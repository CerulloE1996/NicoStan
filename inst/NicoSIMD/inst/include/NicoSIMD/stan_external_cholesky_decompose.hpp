#pragma once
//
// General Cholesky factor with an explicit SIMD (AVX-512 / AVX2) rank-k update. No model inputs, priors, jitter
// or tolerances are embedded. The native Eigen panel schedule, panel factorisation and triangular solve are
// retained. The trailing lower-triangle rank-k update (the "bottleneck" of Eigen's blocked LLT) is computed here
// with explicit packets, reading the panel in place (no packed copies of the panel), in EXACTLY the order of
// floating-point operations of Eigen's own update (selfadjointView<Eigen::Lower>().rankUpdate(A21, -1): its depth
// and
// row blocking, its triangular micro blocks and, in each part of its kernel, the order in which the terms of
// every entry are summed), so that the factor is the same as Stan's, bit for bit. Stan's reverse algorithms and
// its <= 35 dispatch are retained.
//
// Why the order: a factor that differs from Eigen's only in the last bits (any other order of the sums) has a
// gradient that differs by the rounding amplified by the conditioning of the matrix (e.g. 1e-12 in the factor
// and 1e-4 in the gradient for an equicorrelation matrix with rho = 1 - sqrt(machine epsilon)).
//
// The packets are those of Eigen's own vectorisation (AVX-512 when built with -mavx512f, AVX2 otherwise), as
// the order to be matched is that of Eigen's kernel; -DBAYESMVP_FORCE_AVX2 does not change them (it does not
// change Stan math's Eigen code either).
//
// Stan declaration: matrix nico_cholesky_decompose(matrix A);
// Compile with --allow-undefined and this header as the user header. Data double, matrix<var>, and
// var_value<Eigen matrix> are supported. Other Eigen scalar types delegate to native Stan.
//
// This implementation uses Eigen and Stan internal APIs from the pinned Stan Math runtime (Eigen 3.4.0). It
// must be rechecked against every runtime used to build a model. No Eigen flags or external BLAS are changed.

#include <stan/math/rev.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include "nico_simd_intrinsics.hpp"   //// <immintrin.h> on x86-64; SIMDe (NEON) on ARM
#include <new>
#include <ostream>
#include <type_traits>

namespace nico::math {
namespace cholesky {

namespace eigen_internal = Eigen::internal;


// ---- Eigen's packets of double and its kernel's sizes (Eigen 3.4.0, GeneralBlockPanelKernel.h, with FMA):
//      mr = 3 packets of rows, nr = 4 columns, the triangular micro blocks of lcm(mr, nr) columns
using packet         = eigen_internal::packet_traits<double>::type;
using packet_half    = eigen_internal::unpacket_traits<packet>::half;
using packet_quarter = eigen_internal::unpacket_traits<packet_half>::half;
static constexpr Eigen::Index lanes         = eigen_internal::unpacket_traits<packet>::size;
static constexpr Eigen::Index lanes_half    = eigen_internal::unpacket_traits<packet_half>::size;
static constexpr Eigen::Index lanes_quarter = eigen_internal::unpacket_traits<packet_quarter>::size;
static constexpr Eigen::Index columns_of_a_tile = 4;
static constexpr Eigen::Index rows_of_three_packets = 3 * lanes;
static constexpr Eigen::Index size_of_the_micro_blocks = lanes == 8 ? 24 : 12;   // lcm(3 lanes, 4)
static constexpr Eigen::Index peeling_of_the_depth = 8;                          // Eigen's pk
static_assert(lanes == 4 || lanes == 8, "nico_cholesky_decompose: Eigen vectorised with AVX2 or AVX-512");
#if defined(EIGEN_VECTORIZE_AVX512)
static_assert(lanes == 8, "nico_cholesky_decompose: Eigen's packets of double with AVX-512");
#else
static_assert(lanes == 4, "nico_cholesky_decompose: Eigen's packets of double with AVX2");
#endif
// (the sizes of Eigen's own kernel, so that a change of Eigen stops the build instead of changing the order)
using traits_of_eigen = eigen_internal::gebp_traits<double, double>;
static_assert(traits_of_eigen::mr == rows_of_three_packets && traits_of_eigen::nr == columns_of_a_tile,
              "nico_cholesky_decompose: Eigen's kernel sizes mr and nr are not those this order follows");
static_assert(traits_of_eigen::LhsProgress == lanes, "nico_cholesky_decompose: Eigen's LhsProgress");
static_assert(eigen_internal::meta_least_common_multiple<rows_of_three_packets, columns_of_a_tile>::ret
              == size_of_the_micro_blocks, "nico_cholesky_decompose: Eigen's triangular micro blocks");
#ifndef EIGEN_VECTORIZE_FMA
#error "nico_cholesky_decompose: Eigen's kernel order of FMA updates needs -mfma (EIGEN_VECTORIZE_FMA)."
#endif

// ---- one call of Eigen's gebp kernel, with alpha = -1, on the panel in place:
//      result(i, j) -= sum over the depth of lhs(i, k) rhs(j, k), i < rows, j < columns,
//      lhs(i, k) = lhs[i + (first_k + k) stride], rhs(j, k) = rhs[j + (first_k + k) stride],
//      result(i, j) = result[i + j stride_of_result];
//      the rows in Eigen's parts (3, 2 and 1 packets, half and quarter packets, then single rows), each part with
//      its own order of the sums of the terms of an entry, and the columns in tiles of 4, then one at a time.

// (rows of n_packets packets, sequential sums: Eigen's 3pX4 / 2pX4 and 3pX1 / 2pX1 parts)
template <int n_packets>
inline void rows_of_whole_packets(double* result, const Eigen::Index stride_of_result, const double* lhs,
                                  const double* rhs, const Eigen::Index stride, const Eigen::Index first_k,
                                  const Eigen::Index first_row, const Eigen::Index end_row,
                                  const Eigen::Index columns, const Eigen::Index depth) {

    const Eigen::Index columns_in_tiles = (columns / columns_of_a_tile) * columns_of_a_tile;
    for (Eigen::Index i = first_row; i < end_row; i += n_packets * lanes) {
        for (Eigen::Index j = 0; j < columns_in_tiles; j += columns_of_a_tile) {
            packet sums[n_packets][columns_of_a_tile];
            for (int m = 0; m < n_packets; ++m) {
                for (int c = 0; c < columns_of_a_tile; ++c) sums[m][c] = eigen_internal::pset1<packet>(0.0);
            }
            for (Eigen::Index k = 0; k < depth; ++k) {
                const double* lhs_of_k = lhs + (first_k + k) * stride;
                const double* rhs_of_k = rhs + (first_k + k) * stride;
                packet rows_of_lhs[n_packets];
                for (int m = 0; m < n_packets; ++m) {
                    rows_of_lhs[m] = eigen_internal::ploadu<packet>(lhs_of_k + i + m * lanes);
                }
                for (int c = 0; c < columns_of_a_tile; ++c) {
                    const packet broadcast = eigen_internal::pset1<packet>(rhs_of_k[j + c]);
                    for (int m = 0; m < n_packets; ++m) {
                        sums[m][c] = eigen_internal::pmadd(rows_of_lhs[m], broadcast, sums[m][c]);
                    }
                }
            }
            for (int c = 0; c < columns_of_a_tile; ++c) {
                for (int m = 0; m < n_packets; ++m) {
                    double* target = result + i + m * lanes + (j + c) * stride_of_result;
                    eigen_internal::pstoreu(target, eigen_internal::psub(eigen_internal::ploadu<packet>(target),
                                                                         sums[m][c]));
                }
            }
        }
        for (Eigen::Index j = columns_in_tiles; j < columns; ++j) {
            packet sums[n_packets];
            for (int m = 0; m < n_packets; ++m) sums[m] = eigen_internal::pset1<packet>(0.0);
            for (Eigen::Index k = 0; k < depth; ++k) {
                const double* lhs_of_k = lhs + (first_k + k) * stride;
                const packet broadcast = eigen_internal::pset1<packet>(rhs[j + (first_k + k) * stride]);
                for (int m = 0; m < n_packets; ++m) {
                    sums[m] = eigen_internal::pmadd(eigen_internal::ploadu<packet>(lhs_of_k + i + m * lanes),
                                                    broadcast, sums[m]);
                }
            }
            for (int m = 0; m < n_packets; ++m) {
                double* target = result + i + m * lanes + j * stride_of_result;
                eigen_internal::pstoreu(target, eigen_internal::psub(eigen_internal::ploadu<packet>(target),
                                                                     sums[m]));
            }
        }
    }

}

// (rows of one packet, half a packet or a quarter of a packet: Eigen's lhs_process_one_packet; in tiles of 4
//  columns the even and the odd terms of the peeled depth are summed apart, then added, then the rest of the
//  depth; one column at a time, sequential sums)
template <typename packet_type>
inline void rows_of_one_packet_with_even_and_odd_sums(double* result, const Eigen::Index stride_of_result,
                                                      const double* lhs, const double* rhs,
                                                      const Eigen::Index stride, const Eigen::Index first_k,
                                                      const Eigen::Index first_row, const Eigen::Index end_row,
                                                      const Eigen::Index columns, const Eigen::Index depth) {

    constexpr Eigen::Index size_of_packet = eigen_internal::unpacket_traits<packet_type>::size;
    const Eigen::Index columns_in_tiles = (columns / columns_of_a_tile) * columns_of_a_tile;
    const Eigen::Index peeled_depth = depth & ~(peeling_of_the_depth - 1);
    for (Eigen::Index i = first_row; i < end_row; i += size_of_packet) {
        for (Eigen::Index j = 0; j < columns_in_tiles; j += columns_of_a_tile) {
            packet_type even_sums[columns_of_a_tile], odd_sums[columns_of_a_tile];
            for (int c = 0; c < columns_of_a_tile; ++c) {
                even_sums[c] = eigen_internal::pset1<packet_type>(0.0);
                odd_sums[c]  = eigen_internal::pset1<packet_type>(0.0);
            }
            for (Eigen::Index k = 0; k < peeled_depth; ++k) {
                const packet_type row_of_lhs = eigen_internal::ploadu<packet_type>(lhs + i
                                                                                   + (first_k + k) * stride);
                const double* rhs_of_k = rhs + (first_k + k) * stride;
                packet_type* sums = (k % 2 == 0) ? even_sums : odd_sums;
                for (int c = 0; c < columns_of_a_tile; ++c) {
                    sums[c] = eigen_internal::pmadd(row_of_lhs,
                                                    eigen_internal::pset1<packet_type>(rhs_of_k[j + c]), sums[c]);
                }
            }
            for (int c = 0; c < columns_of_a_tile; ++c) {
                even_sums[c] = eigen_internal::padd(even_sums[c], odd_sums[c]);
            }
            for (Eigen::Index k = peeled_depth; k < depth; ++k) {
                const packet_type row_of_lhs = eigen_internal::ploadu<packet_type>(lhs + i
                                                                                   + (first_k + k) * stride);
                const double* rhs_of_k = rhs + (first_k + k) * stride;
                for (int c = 0; c < columns_of_a_tile; ++c) {
                    even_sums[c] = eigen_internal::pmadd(row_of_lhs,
                                                         eigen_internal::pset1<packet_type>(rhs_of_k[j + c]),
                                                         even_sums[c]);
                }
            }
            for (int c = 0; c < columns_of_a_tile; ++c) {
                double* target = result + i + (j + c) * stride_of_result;
                eigen_internal::pstoreu(target, eigen_internal::psub(eigen_internal::ploadu<packet_type>(target),
                                                                     even_sums[c]));
            }
        }
        for (Eigen::Index j = columns_in_tiles; j < columns; ++j) {
            packet_type sum = eigen_internal::pset1<packet_type>(0.0);
            for (Eigen::Index k = 0; k < depth; ++k) {
                sum = eigen_internal::pmadd(eigen_internal::ploadu<packet_type>(lhs + i + (first_k + k) * stride),
                                            eigen_internal::pset1<packet_type>(rhs[j + (first_k + k) * stride]),
                                            sum);
            }
            double* target = result + i + j * stride_of_result;
            eigen_internal::pstoreu(target, eigen_internal::psub(eigen_internal::ploadu<packet_type>(target),
                                                                 sum));
        }
    }

}

// (single rows: Eigen's "remaining rows, 1 at once"; in tiles of 4 columns, the 4 columns are the lanes of the
//  sums, and the terms of the depth are summed in 4 (AVX2) or 8 (AVX-512, two depths per packet) interleaved
//  sums, as Eigen's swapped kernel; one column at a time, a sequential scalar FMA sum)
inline void single_rows(double* result, const Eigen::Index stride_of_result, const double* lhs, const double* rhs,
                        const Eigen::Index stride, const Eigen::Index first_k, const Eigen::Index first_row,
                        const Eigen::Index end_row, const Eigen::Index columns, const Eigen::Index depth) {

    const Eigen::Index columns_in_tiles = (columns / columns_of_a_tile) * columns_of_a_tile;
    for (Eigen::Index j = 0; j < columns_in_tiles; j += columns_of_a_tile) {
        for (Eigen::Index i = first_row; i < end_row; ++i) {
#if defined(EIGEN_VECTORIZE_AVX512)
            // (AVX-512: lanes 0-3 the 4 columns at the first depth of a pair, lanes 4-7 at the second)
            const Eigen::Index end_of_pairs = (depth / 2) * 2;
            const Eigen::Index end_of_eights = (depth / 8) * 8;
            auto columns_at_two_depths = [&](const Eigen::Index k) {

                ALIGN32  __m256d const at_k = _mm256_loadu_pd(rhs + j + (first_k + k) * stride);
                ALIGN32  __m256d const at_k_plus_1 = _mm256_loadu_pd(rhs + j + (first_k + k + 1) * stride);
                return _mm512_insertf64x4(_mm512_castpd256_pd512(at_k), at_k_plus_1, 1);

            };
            auto lhs_at_two_depths = [&](const Eigen::Index k) {

                ALIGN32  __m256d const at_k = _mm256_set1_pd(lhs[i + (first_k + k) * stride]);
                ALIGN32  __m256d const at_k_plus_1 = _mm256_set1_pd(lhs[i + (first_k + k + 1) * stride]);
                return _mm512_insertf64x4(_mm512_castpd256_pd512(at_k), at_k_plus_1, 1);

            };
            packet sums[4];
            for (int s = 0; s < 4; ++s) sums[s] = eigen_internal::pset1<packet>(0.0);
            Eigen::Index k = 0;
            for (; k < end_of_eights; k += 8) {
                for (int s = 0; s < 4; ++s) {
                    sums[s] = eigen_internal::pmadd(columns_at_two_depths(k + 2 * s),
                                                    lhs_at_two_depths(k + 2 * s), sums[s]);
                }
            }
            sums[0] = eigen_internal::padd(eigen_internal::padd(sums[0], sums[1]),
                                           eigen_internal::padd(sums[2], sums[3]));
            for (; k < end_of_pairs; k += 2) {
                sums[0] = eigen_internal::pmadd(columns_at_two_depths(k), lhs_at_two_depths(k), sums[0]);
            }
            packet_half sum_of_4 = eigen_internal::predux_half_dowto4(sums[0]);
            if (depth - end_of_pairs > 0) {
                const Eigen::Index last = first_k + depth - 1;
                sum_of_4 = eigen_internal::pmadd(eigen_internal::ploadu<packet_half>(rhs + j + last * stride),
                                                 eigen_internal::pset1<packet_half>(lhs[i + last * stride]),
                                                 sum_of_4);
            }
#else
            // (AVX2: the 4 columns at one depth per packet)
            const Eigen::Index end_of_fours = (depth / 4) * 4;
            packet sums[4];
            for (int s = 0; s < 4; ++s) sums[s] = eigen_internal::pset1<packet>(0.0);
            Eigen::Index k = 0;
            for (; k < end_of_fours; k += 4) {
                for (int s = 0; s < 4; ++s) {
                    sums[s] = eigen_internal::pmadd(
                        eigen_internal::ploadu<packet>(rhs + j + (first_k + k + s) * stride),
                        eigen_internal::pset1<packet>(lhs[i + (first_k + k + s) * stride]), sums[s]);
                }
            }
            sums[0] = eigen_internal::padd(eigen_internal::padd(sums[0], sums[1]),
                                           eigen_internal::padd(sums[2], sums[3]));
            for (; k < depth; ++k) {
                sums[0] = eigen_internal::pmadd(eigen_internal::ploadu<packet>(rhs + j + (first_k + k) * stride),
                                                eigen_internal::pset1<packet>(lhs[i + (first_k + k) * stride]),
                                                sums[0]);
            }
            const packet sum_of_4 = sums[0];
#endif
            alignas(64) double lanes_of_the_sum[4];
            eigen_internal::pstoreu(lanes_of_the_sum, sum_of_4);
            for (int c = 0; c < columns_of_a_tile; ++c) {
                result[i + (j + c) * stride_of_result] -= lanes_of_the_sum[c];
            }
        }
    }
    for (Eigen::Index j = columns_in_tiles; j < columns; ++j) {
        for (Eigen::Index i = first_row; i < end_row; ++i) {
            double sum = 0.0;
            for (Eigen::Index k = 0; k < depth; ++k) {
                sum = std::fma(lhs[i + (first_k + k) * stride], rhs[j + (first_k + k) * stride], sum);
            }
            result[i + j * stride_of_result] -= sum;
        }
    }

}

inline void kernel_in_the_order_of_eigen(double* result, const Eigen::Index stride_of_result, const double* lhs,
                                         const double* rhs, const Eigen::Index stride, const Eigen::Index first_k,
                                         const Eigen::Index rows, const Eigen::Index columns,
                                         const Eigen::Index depth) {

    if (rows <= 0 || columns <= 0 || depth <= 0) return;
    const Eigen::Index end_of_3 = (rows / (3 * lanes)) * (3 * lanes);
    const Eigen::Index end_of_2 = end_of_3 + ((rows - end_of_3) / (2 * lanes)) * (2 * lanes);
    const Eigen::Index end_of_1 = end_of_2 + ((rows - end_of_2) / lanes) * lanes;
    const Eigen::Index end_of_half = end_of_1 + ((rows - end_of_1) / lanes_half) * lanes_half;
    const Eigen::Index end_of_quarter = end_of_half + ((rows - end_of_half) / lanes_quarter) * lanes_quarter;
    rows_of_whole_packets<3>(result, stride_of_result, lhs, rhs, stride, first_k, 0, end_of_3, columns, depth);
    rows_of_whole_packets<2>(result, stride_of_result, lhs, rhs, stride, first_k, end_of_3, end_of_2, columns,
                             depth);
    rows_of_one_packet_with_even_and_odd_sums<packet>(result, stride_of_result, lhs, rhs, stride, first_k,
                                                      end_of_2, end_of_1, columns, depth);
    if constexpr (lanes_half < lanes) {
        rows_of_one_packet_with_even_and_odd_sums<packet_half>(result, stride_of_result, lhs, rhs, stride,
                                                               first_k, end_of_1, end_of_half, columns, depth);
    }
    if constexpr (lanes_quarter < lanes_half) {
        rows_of_one_packet_with_even_and_odd_sums<packet_quarter>(result, stride_of_result, lhs, rhs, stride,
                                                                  first_k, end_of_half, end_of_quarter, columns,
                                                                  depth);
    }
    single_rows(result, stride_of_result, lhs, rhs, stride, first_k, end_of_quarter, rows, columns, depth);

}

// C <- C - P P^T, lower triangle only. P and C are disjoint blocks of the same column-major matrix, with the same
// leading dimension: rows rows of C and of P, rank columns of P. The schedule is Eigen's
// general_matrix_matrix_triangular_product (its blocking from Eigen's own gemm_blocking_space): the depth in
// blocks of kc, the rows in blocks of mc; in each, the part before the diagonal, then the diagonal block in
// micro blocks of lcm(mr, nr) columns: the micro block on the diagonal through a zeroed buffer (its lower
// triangle added), then the rows below it. The strict upper triangle and every entry outside C are never
// written.
inline void subtract_rank_k(double* C, const double* P, const Eigen::Index rows,
                            const Eigen::Index rank, const Eigen::Index stride) {

    if (rows <= 0 || rank <= 0) return;
    eigen_internal::gemm_blocking_space<Eigen::ColMajor, double, double, Eigen::Dynamic, Eigen::Dynamic,
                                        Eigen::Dynamic> blocking(rows, rows, rank, 1, false);
    const Eigen::Index kc = blocking.kc();
    Eigen::Index mc = std::min(rows, blocking.mc());
    if (mc > columns_of_a_tile) mc = (mc / columns_of_a_tile) * columns_of_a_tile;
    alignas(64) double buffer[size_of_the_micro_blocks * size_of_the_micro_blocks];
    for (Eigen::Index k2 = 0; k2 < rank; k2 += kc) {
        const Eigen::Index depth = std::min(k2 + kc, rank) - k2;
        for (Eigen::Index i2 = 0; i2 < rows; i2 += mc) {
            const Eigen::Index rows_of_block = std::min(i2 + mc, rows) - i2;
            // (1) before the diagonal: rows i2 to i2 + rows_of_block, columns 0 to i2
            kernel_in_the_order_of_eigen(C + i2, stride, P + i2, P, stride, k2, rows_of_block, i2, depth);
            // (2) the diagonal block (Eigen's tribb_kernel)
            for (Eigen::Index j = 0; j < rows_of_block; j += size_of_the_micro_blocks) {
                const Eigen::Index size_of_micro_block = std::min(size_of_the_micro_blocks, rows_of_block - j);
                std::fill(buffer, buffer + size_of_the_micro_blocks * size_of_the_micro_blocks, 0.0);
                kernel_in_the_order_of_eigen(buffer, size_of_the_micro_blocks, P + i2 + j, P + i2 + j, stride, k2,
                                             size_of_micro_block, size_of_micro_block, depth);
                for (Eigen::Index j1 = 0; j1 < size_of_micro_block; ++j1) {
                    double* column = C + (i2 + j) + (i2 + j + j1) * stride;
                    for (Eigen::Index i1 = j1; i1 < size_of_micro_block; ++i1) {
                        column[i1] += buffer[i1 + j1 * size_of_the_micro_blocks];
                    }
                }
                const Eigen::Index i = j + size_of_micro_block;
                kernel_in_the_order_of_eigen(C + (i2 + i) + (i2 + j) * stride, stride, P + i2 + i, P + i2 + j,
                                             stride, k2, rows_of_block - i, size_of_micro_block, depth);
            }
        }
    }

}

// This schedule is copied from the pinned Eigen 3.4.0 LLT, not an independently tuned layout choice.
// Return the failing pivot, or -1 on success, as Eigen's own inplace implementation does.
template <typename Matrix>
inline Eigen::Index factor_lower_in_place(Matrix& storage) {

    // All callers own a contiguous column-major value copy. Eigen's unblocked internal routine directly
    // names Eigen::Block<Matrix>, so it needs an Eigen Map rather than Stan's derived arena_matrix wrapper.
    Eigen::Map<Eigen::Matrix<double, -1, -1>> matrix(storage.data(), storage.rows(), storage.cols());
    const Eigen::Index size = matrix.rows();
    if (size < 32) return Eigen::internal::llt_inplace<double, Eigen::Lower>::unblocked(matrix);

    Eigen::Index block_size = size / 8;
    block_size = (block_size / 16) * 16;
    block_size = std::min(std::max(block_size, Eigen::Index(8)), Eigen::Index(128));
    for (Eigen::Index k = 0; k < size; k += block_size) {
        const Eigen::Index bs = std::min(block_size, size - k);
        const Eigen::Index remaining = size - k - bs;
        auto diagonal = matrix.block(k, k, bs, bs);
        const Eigen::Index failure = Eigen::internal::llt_inplace<double, Eigen::Lower>::unblocked(diagonal);
        if (failure >= 0) return k + failure;
        if (remaining > 0) {
            auto panel = matrix.block(k + bs, k, remaining, bs);
            diagonal.adjoint().template triangularView<Eigen::Upper>()
                .template solveInPlace<Eigen::OnTheRight>(panel);
            double* trailing = matrix.data() + k + bs + (k + bs) * matrix.outerStride();
            const double* panel_values = matrix.data() + k + bs + k * matrix.outerStride();
            subtract_rank_k(trailing, panel_values, remaining, bs, matrix.outerStride());
        }
    }
    return -1;

}

template <typename Matrix>
inline void finish_factorisation(Matrix& matrix, const Eigen::Index failure) {

    // Identical failure predicate and message to check_pos_definite on the native Eigen LLT. No new
    // conditioning, finiteness or positive-pivot tolerance is introduced. In particular +Inf is not
    // rejected by a new finite check. Rounding-sensitive boundary cases still require native comparison.
    if (failure >= 0 || !(matrix.diagonal().array() > 0.0).all()) {
        stan::math::throw_domain_error("cholesky_decompose", "Matrix", " is not positive definite", "m");
    }
    matrix.template triangularView<Eigen::StrictlyUpper>().setZero();

}

// Double-only diagnostic control: the exact native Eigen factorisation core without LLT::compute's
// unused L1-norm calculation. It is not used by nico_cholesky_decompose and is not a model replacement.
inline Eigen::Matrix<double, -1, -1> native_core_control(const Eigen::Matrix<double, -1, -1>& input) {

    stan::math::check_symmetric("cholesky_decompose", "m", input);
    stan::math::check_not_nan("cholesky_decompose", "m", input);
    Eigen::Matrix<double, -1, -1> lower = input;
    const Eigen::Index failure = Eigen::internal::llt_inplace<double, Eigen::Lower>::blocked(lower);
    finish_factorisation(lower, failure);
    return lower;

}


template <typename Input, typename Lower, typename Values>
inline void register_native_reverse(Input& input, Lower& lower, Values& values) {

    if (values.rows() <= 35) {
        stan::math::reverse_pass_callback(stan::math::internal::unblocked_cholesky_lambda(values, lower,
                                                                                      input));
    } else {
        stan::math::reverse_pass_callback(stan::math::internal::cholesky_lambda(values, lower, input));
    }

}

}  // namespace cholesky
}  // namespace nico::math

namespace nico::math {

inline int nico_cholesky_decompose_lanes(std::ostream* /*pstream__*/) {

    return static_cast<int>(nico::math::cholesky::lanes);

}

}  // namespace nico::math

using nico::math::nico_cholesky_decompose_lanes;

namespace nico::math {

template <typename EigMat, stan::require_eigen_t<EigMat>* = nullptr,
          stan::require_not_eigen_vt<stan::is_var, EigMat>* = nullptr>
inline auto nico_cholesky_decompose(const EigMat& input, std::ostream* /*pstream__*/) {

    if constexpr (!std::is_same<stan::value_type_t<EigMat>, double>::value) {
        return stan::math::cholesky_decompose(input);
    } else {
        const stan::eval_return_type_t<EigMat>& evaluated = input.eval();
        stan::math::check_symmetric("cholesky_decompose", "m", evaluated);
        stan::math::check_not_nan("cholesky_decompose", "m", evaluated);
        Eigen::Matrix<double, -1, -1> lower = evaluated;
        const Eigen::Index failure = nico::math::cholesky::factor_lower_in_place(lower);
        nico::math::cholesky::finish_factorisation(lower, failure);
        using return_t = Eigen::Matrix<double, EigMat::RowsAtCompileTime, EigMat::ColsAtCompileTime>;
        return return_t(lower);
    }

}

}  // namespace nico::math

using nico::math::nico_cholesky_decompose;

namespace nico::math {

template <typename EigMat, stan::require_eigen_vt<stan::is_var, EigMat>* = nullptr>
inline auto nico_cholesky_decompose(const EigMat& input, std::ostream* /*pstream__*/) {

    using stan::math::var;
    using stan::math::vari;
    stan::math::check_square("cholesky_decompose", "A", input);
    stan::arena_t<EigMat> input_on_arena = input;
    stan::arena_t<Eigen::Matrix<double, -1, -1>> values(input_on_arena.val());
    stan::math::check_symmetric("cholesky_decompose", "A", input);
    const Eigen::Index failure = nico::math::cholesky::factor_lower_in_place(values);
    nico::math::cholesky::finish_factorisation(values, failure);

    const Eigen::Index size = values.rows();
    stan::arena_t<EigMat> lower(size, size);
    auto& arena = stan::math::ChainableStack::instance_->memalloc_;
    const std::size_t n = static_cast<std::size_t>(size);
    const std::size_t count = n * (n + 1) / 2;
    vari* entries = count == 0 ? nullptr : arena.alloc_array<vari>(count);
    vari* upper_zero = new vari(0.0, false);
    std::size_t position = 0;
    for (Eigen::Index j = 0; j < size; ++j) {
        for (Eigen::Index i = 0; i < j; ++i) lower(i, j) = var(upper_zero);
        for (Eigen::Index i = j; i < size; ++i, ++position) {
            ::new (static_cast<void*>(entries + position)) vari(values(i, j), false);
            lower(i, j) = var(entries + position);
        }
    }
    // Keep the native lower-triangle-only input adjoints, including when distinct entries share a vari.
    // Upper output entries share an independent zero node; their adjoints never reach the input.
    nico::math::cholesky::register_native_reverse(input_on_arena, lower, values);
    return stan::plain_type_t<EigMat>(lower);

}

}  // namespace nico::math

using nico::math::nico_cholesky_decompose;

namespace nico::math {

template <typename T, stan::require_var_matrix_t<T>* = nullptr>
inline auto nico_cholesky_decompose(const T& input, std::ostream* /*pstream__*/) {

    // Native var-matrix Cholesky makes this first A check and then the prim m/NaN checks.
    stan::math::check_symmetric("cholesky_decompose", "A", input.val());
    stan::plain_type_t<T> lower = nico_cholesky_decompose(input.val(), nullptr);
    auto values = lower.val();
    auto input_handle = input;
    nico::math::cholesky::register_native_reverse(input_handle, lower, values);
    return lower;

}

}  // namespace nico::math

using nico::math::nico_cholesky_decompose;






















