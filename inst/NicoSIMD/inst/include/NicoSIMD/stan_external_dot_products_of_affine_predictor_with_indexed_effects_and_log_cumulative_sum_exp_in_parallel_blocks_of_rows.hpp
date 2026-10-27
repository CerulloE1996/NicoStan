#pragma once
//
// -| --------- BayesMVP dot products of an affine predictor with indexed effects and its log cumulative sum of
// -| --------- exponentials, with its blocks of rows run in parallel (Stan external function) -------------------
//
//     nico_dot_products_of_affine_predictor_with_indexed_effects_and_log_cumulative_sum_exp(
//           linear_weights, prefix_weights, X, beta, scale, z, index, rows_in_order_of_index,
//           last_position_of_each_index, offset, rows_per_block_run_in_parallel)
//       = sum_i linear_weights[i] eta[i] + sum_i prefix_weights[i] L[i],
//
//     eta[i] = X[i] beta + scale z[index[i]],     L[i] = log(exp(offset) + sum_{j <= i} exp(eta[j])),
//
// the same value and gradient (up to floating-point rounding) as the function of the same name without its last
// argument (stan_external_dot_products_of_affine_predictor_with_indexed_effects_and_log_cumulative_sum_exp.hpp:
// the partial likelihood of any proportional-hazards model with group effects, and any running total on the log
// scale of an affine predictor with group effects), with the rows split into blocks of
// rows_per_block_run_in_parallel consecutive rows (rounded up to a multiple of 8 rows, so that every block but
// the last is a whole number of SIMD registers with AVX2 and AVX-512 alike), which run in parallel on the
// threads of the calling chain, one TBB task per block, as the chunks of reduce_sum_static (its simple
// partitioner, chunk_size 1). The chunking data of a Stan model (chunk_size = ceiling(N / N_chunks)) is the
// argument, so N_chunks sets the number of blocks run in parallel, as for any tape-chunked model.
//
// Still ONE autodiff node, its partial derivatives made in the forward pass: nothing on the tape but the node,
// no vector of N effects, no copy of the data and no reduce_sum; the effects are read by their index inside
// the node, one partial per group. The running total over the blocks is a two-level scan made inside the node:
//   (1) in parallel, each block's eta, its dot product with the linear weights, its largest value m_b and
//       S_b = the sum of exp(eta - m_b) over its rows (in place of eta);
//   (2) on the calling thread, a scan over the N_blocks block totals: each block's offset o_b, the log of
//       exp(offset) plus the sum of exp(eta) over every row before the block (o_1 = offset);
//   (3) in parallel, each block in units of exp(s_b), with s_b the larger of m_b and o_b (so that no term
//       overflows): its exponentials exp(eta - s_b), its running sums C (from exp(o_b - s_b), a forward scan),
//       the dot product of the prefix weights with log C, prefix_weights / C and their sum R_b over the block;
//   (4) on the calling thread, the backward scan over the blocks of the R_b: each block's sum of
//       prefix_weights / C over every row after it, in its own units (the factors exp(s_a - s_b) are at most 1,
//       since the offsets never decrease and o_b is at least m_a and o_a for every block a before b);
//   (5) in parallel, each block's backward scan of prefix_weights / C from that sum (q), the derivatives of the
//       result at its rows, linear_weights + exp(eta - s_b) q, and their product with its rows of X;
//   (6) in parallel over ranges of the groups, the sums of the derivatives over the rows of each group (the
//       derivatives in the order of the groups, their running sum and its differences at the last position of
//       each group, as in the serial function).
// Every per-element operation is a whole-vector Eigen operation, BayesMVP's exp and log kernels or BayesMVP's
// SIMD scans (stan_external_simd_affine_recurrence_scans.hpp, coefficient 1); the loops are over the blocks (and,
// within each block, over its blocks of 1024 rows, as in the serial function, so that a block's values stay in
// the core's own cache from one step to the next), never over the rows or the groups. The partials of beta and
// the values of the blocks are added in the order of the blocks, so that the result does not depend on which
// thread ran which block. The parallel work runs in an isolated region of the calling arena
// (tbb::this_task_arena::isolate), so that a thread waiting for its blocks never takes up another chain's work,
// which would use the same buffers.
//
// With one block (rows_per_block_run_in_parallel at least the number of rows), with a non-finite eta (the
// serial function reports it) and in the rare case of a block whose first running sum C is under 1e-280 (eta
// far below the largest value of its block and its offset, far outside the posterior), the serial function is
// used: the same values and factors as the unchunked model.
//
// Declared in the Stan functions block (no body; stanc --allow-undefined), next to the serial function:
//
//     real nico_dot_products_of_affine_predictor_with_indexed_effects_and_log_cumulative_sum_exp(
//           data vector linear_weights, data vector prefix_weights, data matrix X, vector beta, real scale,
//           array[] real z, data array[] int index, data array[] int rows_in_order_of_index,
//           data array[] int last_position_of_each_index, real log_offset,
//           data int rows_per_block_run_in_parallel);
//
// -----------------------------------------------------------------------------------------------------------

#include "stan_external_functions.hpp"
#include "stan_external_simd_affine_recurrence_scans.hpp"
#include "stan_external_log_sum_exp_with_nico_exponential.hpp"
#include "stan_external_dot_products_of_affine_predictor_with_indexed_effects_and_log_cumulative_sum_exp.hpp"

#include <stan/math/rev.hpp>
#include <stan/math/prim.hpp>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/partitioner.h>
#include <tbb/task_arena.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace nico::math {

////
//// ---- Packet arithmetic and the weighted scan of the parallel blocks:
////      simd_t holds BMVP_LANES doubles: four with AVX2, eight with AVX-512. The shared packet transfers
////      mask the final register; fixed-lane loops below transfer values or addresses, not mathematics.
////
namespace parallel_affine_prefix {

using namespace log_sum_exp_simd;
using namespace log_cumulative_sum_exp;

inline double nico_sum_parallel_values(const double* values, const Eigen::Index count) {

    simd_t sum = scan_zero();
    for (Eigen::Index first = 0; first < count; first += BMVP_LANES) {
        const int lanes = static_cast<int>(std::min<Eigen::Index>(BMVP_LANES, count - first));
        sum = nico_lse_add(sum, nico_load_prefix_packet(values + first, lanes, 0.0));
    }
    return nico_lse_sum(sum);

}

template <typename T_matrix>
inline simd_t nico_load_parallel_matrix_column(const T_matrix& matrix, const Eigen::Index first,
                                               const Eigen::Index column, const int lanes) {

    alignas(64) double values[BMVP_LANES] = {};
    for (int lane = 0; lane < lanes; ++lane) values[lane] = matrix.coeff(first + lane, column);
    return simd_load(values);

}

inline simd_t nico_load_parallel_indexed_values(const double* values, const int* indices,
                                                const int lanes) {

    alignas(64) double gathered[BMVP_LANES] = {};
    for (int lane = 0; lane < lanes; ++lane) gathered[lane] = values[indices[lane]];
    return simd_load(gathered);

}

//// The carried pair is (maximum, scaled sum). Keeping only its rounded logarithm loses multiplicities
//// when predictors are very large. Each block's initial sum and rescaling factor use the retained pair.
inline bool nico_scan_parallel_block_totals(const double* maxima, const double* sums,
                                           const double* first_exponentials, const Eigen::Index count,
                                           const double offset, double* shifts, double* initial_sums,
                                           double* rescaling_factors) {

    simd_t carry_maximum = scan_set1(offset);
    simd_t carry_sum = scan_set1(offset == -std::numeric_limits<double>::infinity() ? 0.0 : 1.0);
    bool every_first_sum_is_representable = true;
    for (Eigen::Index first = 0; first < count; first += BMVP_LANES) {
        const int lanes = static_cast<int>(std::min<Eigen::Index>(BMVP_LANES, count - first));
        const simd_t block_maximum = nico_load_prefix_packet(maxima + first, lanes,
                                                            -std::numeric_limits<double>::infinity());
        simd_t maximum = block_maximum;
        simd_t sum = nico_load_prefix_packet(sums + first, lanes, 0.0);
        nico_compose_log_prefix_packets(maximum, sum, nico_shift_log_prefix_packet<1>(maximum),
                                        shift_up<1>(sum));
        nico_compose_log_prefix_packets(maximum, sum, nico_shift_log_prefix_packet<2>(maximum),
                                        shift_up<2>(sum));
#if BMVP_LANES == 8
        nico_compose_log_prefix_packets(maximum, sum, nico_shift_log_prefix_packet<4>(maximum),
                                        shift_up<4>(sum));
#endif
        nico_compose_log_prefix_packets(maximum, sum, carry_maximum, carry_sum);
        const simd_t before_maximum = shift_up_one_with(maximum, carry_maximum);
        const simd_t before_sum = shift_up_one_with(sum, carry_sum);
        const simd_t shift = nico_lse_maximum_packet(block_maximum, before_maximum);
        const simd_t initial = scan_mul(before_sum, nico_scale_log_prefix_packet(before_maximum, shift));
        const simd_t factor = nico_scale_log_prefix_packet(block_maximum, shift);
        const simd_t first_sum = scan_fmadd(factor,
                    nico_load_prefix_packet(first_exponentials + first, lanes, 1.0), initial);
        const simd_t valid_first_sum = nico_lse_select(
                    nico_lse_greater(scan_set1(1e-280), first_sum), scan_zero(), scan_set1(1.0));
        alignas(64) double valid[BMVP_LANES];
        simd_store(valid, valid_first_sum);
        for (int lane = 0; lane < lanes; ++lane) {
            every_first_sum_is_representable = every_first_sum_is_representable && valid[lane] != 0.0;
        }
        nico_store_prefix_packet(shifts + first, lanes, shift);
        nico_store_prefix_packet(initial_sums + first, lanes, initial);
        nico_store_prefix_packet(rescaling_factors + first, lanes, factor);
        carry_maximum = broadcast_last_lane(maximum);
        carry_sum = broadcast_last_lane(sum);
    }
    return every_first_sum_is_representable;

}

}  // namespace parallel_affine_prefix

//// (the buffers of the calling thread, kept between calls: two vectors of n values with a leading zero, as in
////  the serial function, which every block's task reads and writes at its own rows only)
struct buffers_of_affine_predictor_with_indexed_effects_in_parallel_blocks_t {
    //// (a leading zero, then: eta, then exp(eta - m_b), then exp(eta - s_b); padded with zeros)
    Eigen::Matrix<double, -1, 1> predictor_then_exponentials;
    //// (a leading zero, then: prefix_weights / C, then, in place, the derivative of the result at each row)
    Eigen::Matrix<double, -1, 1> prefix_weights_over_running_sums_then_derivatives;
    Eigen::Matrix<double, -1, 1> effects_with_a_leading_zero;    // 0, then z[1], ..., z[J]
    void ensure(const Eigen::Index n_padded, const Eigen::Index J) {

        if (predictor_then_exponentials.size() < n_padded + 1) {
            predictor_then_exponentials.resize(n_padded + 1);
            prefix_weights_over_running_sums_then_derivatives.resize(n_padded + 1);
        }
        if (effects_with_a_leading_zero.size() < J + 1) effects_with_a_leading_zero.resize(J + 1);

    }
};
inline buffers_of_affine_predictor_with_indexed_effects_in_parallel_blocks_t&
buffers_of_affine_predictor_with_indexed_effects_in_parallel_blocks() {

    static thread_local buffers_of_affine_predictor_with_indexed_effects_in_parallel_blocks_t buffers;
    return buffers;

}

//// (the buffers of the thread that runs a task: three of one block of 1024 rows, and one of the rows of a range
////  of groups, with a leading zero)
struct buffers_of_one_task_of_affine_predictor_with_indexed_effects_t {
    Eigen::Matrix<double, -1, 1> running_sums_of_the_rows;            // C of one block of 1024 rows
    Eigen::Matrix<double, -1, 1> logs_of_the_rows;                    // log C of one block of 1024 rows
    Eigen::Matrix<double, -1, 1> backward_running_sums_of_the_rows;   // q of one block of 1024 rows
    Eigen::Matrix<double, -1, 1> running_sums_in_order_of_index;
    // 0, then the running sum of a range of groups' rows
    void ensure(const Eigen::Index block_size) {

        if (running_sums_of_the_rows.size() < block_size) {
            running_sums_of_the_rows.resize(block_size);
            logs_of_the_rows.resize(block_size);
            backward_running_sums_of_the_rows.resize(block_size);
        }

    }
};
inline buffers_of_one_task_of_affine_predictor_with_indexed_effects_t&
buffers_of_one_task_of_affine_predictor_with_indexed_effects() {

    static thread_local buffers_of_one_task_of_affine_predictor_with_indexed_effects_t buffers;
    return buffers;

}

//// (the rows of every block run in parallel but the last are a multiple of this number)
inline constexpr Eigen::Index rows_of_parallel_blocks_are_a_multiple_of = 8;

//// log(exp(a) + exp(b)), either of them -infinity allowed
inline double log_sum_exp_of_two_numbers(const double a, const double b) {

    return log_sum_exp_simd::nico_lse_first(log_sum_exp_simd::nico_lse_add(
               log_sum_exp_simd::nico_lse_maximum_packet(scan_set1(a), scan_set1(b)),
               log_sum_exp_simd::nico_lse_log1p_nonnegative(
                   log_cumulative_sum_exp::nico_scale_log_prefix_packet(
                       log_sum_exp_simd::nico_lse_select(
                           log_sum_exp_simd::nico_lse_greater(scan_set1(a), scan_set1(b)),
                           scan_set1(b), scan_set1(a)),
                       log_sum_exp_simd::nico_lse_maximum_packet(scan_set1(a), scan_set1(b))))));

}

//// one TBB task per block (simple partitioner, chunk_size 1, as reduce_sum_static), in an isolated region of the
//// calling arena
template <typename Task_of_one_block>
inline void run_one_task_per_block_in_parallel(const Eigen::Index n_blocks, const Task_of_one_block& task) {

    tbb::this_task_arena::isolate([&]() {

        tbb::parallel_for(tbb::blocked_range<Eigen::Index>(0, n_blocks, 1),
                          [&](const tbb::blocked_range<Eigen::Index>& range) {

                                for (Eigen::Index b = range.begin(); b < range.end(); ++b) task(b);

                          },
                          tbb::simple_partitioner());

    });

}

}  // namespace nico::math

// ---- the Stan-facing function (global scope; stanc calls it as fast_...(..., pstream__))
namespace nico::math {

template <typename T_linear_weights, typename T_prefix_weights, typename T_x, typename T_beta, typename T_scale,
          typename T_z, typename T_offset>
inline stan::return_type_t<T_beta, T_scale, T_z, T_offset>
nico_dot_products_of_affine_predictor_with_indexed_effects_and_log_cumulative_sum_exp(
        const T_linear_weights& linear_weights, const T_prefix_weights& prefix_weights, const T_x& x,
        const T_beta& beta, const T_scale& scale, const std::vector<T_z>& z, const std::vector<int>& index,
        const std::vector<int>& rows_in_order_of_index, const std::vector<int>& last_position_of_each_index,
        const T_offset& offset, const int& rows_per_block_run_in_parallel,
        std::ostream* pstream__ = nullptr) {

    static_assert(stan::is_constant_all<T_linear_weights, T_prefix_weights, T_x>::value,
                  "The weights and the predictor matrix must be data.");
    static const char* function =
          "nico_dot_products_of_affine_predictor_with_indexed_effects_and_log_cumulative_sum_exp";
    const auto& x_ref = stan::math::to_ref(x);
    const auto& beta_ref = stan::math::to_ref(beta);
    const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> linear_values(linear_weights);
    const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> prefix_values(prefix_weights);
    const Eigen::Index n = x_ref.rows();
    const Eigen::Index J = static_cast<Eigen::Index>(z.size());
    //// (the size of the blocks: a multiple of 8 rows; one block: the serial function, which also makes every
    ////  check of the arguments)
    const Eigen::Index multiple = nico::math::rows_of_parallel_blocks_are_a_multiple_of;
    const Eigen::Index rows_per_block =
          ((std::max<Eigen::Index>(1, rows_per_block_run_in_parallel) + multiple - 1) / multiple) * multiple;
    auto serial_function = [&]() {

          return nico_dot_products_of_affine_predictor_with_indexed_effects_and_log_cumulative_sum_exp(
                       linear_weights, prefix_weights, x, beta, scale, z, index, rows_in_order_of_index,
                       last_position_of_each_index, offset, pstream__);

    };
    if (rows_per_block >= n || J == 0) return serial_function();
    stan::math::check_size_match(function, "columns of x", x_ref.cols(), "size of beta", beta_ref.size());
    stan::math::check_size_match(function, "rows of x", n, "size of index", index.size());
    stan::math::check_size_match(function, "rows of x", n, "linear weights", linear_values.size());
    stan::math::check_size_match(function, "rows of x", n, "prefix weights", prefix_values.size());
    stan::math::check_size_match(function, "rows of x", n, "rows in order of index",
                                 rows_in_order_of_index.size());
    stan::math::check_size_match(function, "size of z", J, "last position of each index",
                                 last_position_of_each_index.size());
    //// (the indices, checked as whole vectors)
    const Eigen::Map<const Eigen::Matrix<int, -1, 1>> index_of_each_row(index.data(), n);
    const Eigen::Map<const Eigen::Matrix<int, -1, 1>> rows_in_order(rows_in_order_of_index.data(), n);
    const Eigen::Map<const Eigen::Matrix<int, -1, 1>> last_positions(last_position_of_each_index.data(), J);
    if (index_of_each_row.minCoeff() < 1 || index_of_each_row.maxCoeff() > J || rows_in_order.minCoeff() < 1 ||
        rows_in_order.maxCoeff() > n || last_positions.minCoeff() < 0 || last_positions[J - 1] != n) {

        throw std::domain_error(std::string(function) + ": an index, a row in order of index or a last position "
                                "is outside its range");

    }
    const Eigen::Index n_blocks = (n + rows_per_block - 1) / rows_per_block;
    const Eigen::Index n_padded = nico::math::padded_to_lanes(n);
    const Eigen::Index cache_block_size = nico::math::rows_per_block_of_affine_predictor_with_indexed_effects;
    auto& buffers = nico::math::buffers_of_affine_predictor_with_indexed_effects_in_parallel_blocks();
    buffers.ensure(n_padded, J);
    Eigen::Matrix<double, -1, 1>& first_vector = buffers.predictor_then_exponentials;
    Eigen::Matrix<double, -1, 1>& second_vector = buffers.prefix_weights_over_running_sums_then_derivatives;
    //// (the effects with a leading zero, so that the 1-based indices read them directly)
    buffers.effects_with_a_leading_zero[0] = 0.0;
    buffers.effects_with_a_leading_zero.segment(1, J) =
          stan::math::value_of(Eigen::Map<const Eigen::Matrix<T_z, -1, 1>>(z.data(), J));
    const Eigen::Matrix<double, -1, 1>& effects_with_a_leading_zero = buffers.effects_with_a_leading_zero;
    const Eigen::Matrix<double, -1, 1> beta_values = stan::math::value_of(beta_ref);
    const double scale_value = stan::math::value_of(scale);
    const double offset_value = stan::math::value_of(offset);
    if (std::isnan(offset_value) || offset_value == std::numeric_limits<double>::infinity()) {
          return serial_function();
    }
    //// (row i, 0-based, of eta and then of the exponentials at position i + 1 of the first vector; of
    ////  prefix_weights / C and then of the derivatives at position i + 1 of the second; the padding after row n
    ////  of each is zero, for the scans of the last block)
    double* t = first_vector.data() + 1;
    double* derivatives_of_the_rows = second_vector.data() + 1;
    first_vector.segment(1 + n, n_padded - n).setZero();
    second_vector.segment(1 + n, n_padded - n).setZero();
    const simd_t one = scan_set1(1.0);
    const nico::math::affine_scan_coefficients_t coefficients_of_one{one, one, one, one, one};
    auto first_row_of_block = [&](const Eigen::Index b) { return b * rows_per_block; };
    auto end_row_of_block = [&](const Eigen::Index b) { return std::min(n, (b + 1) * rows_per_block); };
    //// (the values of each block, added in the order of the blocks)
    std::vector<double> linear_part_of_block(n_blocks, 0.0);
    std::vector<double> largest_eta_of_block(n_blocks, -std::numeric_limits<double>::infinity());
    std::vector<double> sum_of_exponentials_of_block(n_blocks, 0.0);
    std::vector<double> first_exponential_of_block(n_blocks, 0.0);
    std::vector<char> eta_of_block_is_finite(n_blocks, 1);
    //// ---- (1) in parallel: each block's eta, its dot product with the linear weights and its largest value
    ////      m_b, then exp(eta - m_b) in place of eta and their sum S_b
    nico::math::run_one_task_per_block_in_parallel(n_blocks, [&](const Eigen::Index b) {

          const Eigen::Index first_row_of_the_block = first_row_of_block(b);
          const Eigen::Index end_row_of_the_block = end_row_of_block(b);
          using namespace parallel_affine_prefix;
          simd_t linear_part = scan_zero();
          simd_t largest = scan_set1(-std::numeric_limits<double>::infinity());
          bool every_eta_is_finite = true;
          for (Eigen::Index first_row = first_row_of_the_block; first_row < end_row_of_the_block;
               first_row += cache_block_size) {

                const Eigen::Index m = std::min(cache_block_size, end_row_of_the_block - first_row);
                for (Eigen::Index row = first_row; row < first_row + m; row += BMVP_LANES) {
                      const int lanes = static_cast<int>(std::min<Eigen::Index>(BMVP_LANES,
                                                                              first_row + m - row));
                      simd_t eta = scan_zero();
                      for (Eigen::Index column = 0; column < beta_values.size(); ++column) {
                            eta = scan_fmadd(nico_load_parallel_matrix_column(x_ref, row, column, lanes),
                                             scan_set1(beta_values[column]), eta);
                      }
                      eta = scan_fmadd(scan_set1(scale_value), nico_load_parallel_indexed_values(
                                  effects_with_a_leading_zero.data(), index.data() + row, lanes), eta);
                      nico_store_prefix_packet(t + row, lanes, eta);
                      alignas(64) double finite[BMVP_LANES];
                      simd_store(finite, nico_lse_select(nico_lse_finite(eta), one, scan_zero()));
                      for (int lane = 0; lane < lanes; ++lane) {
                            every_eta_is_finite = every_eta_is_finite && finite[lane] != 0.0;
                      }
                      linear_part = scan_fmadd(
                                  nico_load_prefix_packet(linear_values.data() + row, lanes, 0.0), eta,
                                  linear_part);
                      alignas(64) double eta_for_maximum[BMVP_LANES];
                      simd_store(eta_for_maximum, eta);
                      largest = nico_lse_maximum_packet(largest, nico_load_prefix_packet(
                                  eta_for_maximum, lanes, -std::numeric_limits<double>::infinity()));
                }

          }
          const double largest_eta = nico_lse_maximum(largest);
          linear_part_of_block[b] = nico_lse_sum(linear_part);
          largest_eta_of_block[b] = largest_eta;
          eta_of_block_is_finite[b] = every_eta_is_finite ? 1 : 0;
          if (!every_eta_is_finite) return;
          simd_t sum_of_exponentials = scan_zero();
          for (Eigen::Index first_row = first_row_of_the_block; first_row < end_row_of_the_block;
               first_row += cache_block_size) {

                const Eigen::Index m = std::min(cache_block_size, end_row_of_the_block - first_row);
                for (Eigen::Index row = first_row; row < first_row + m; row += BMVP_LANES) {
                      const int lanes = static_cast<int>(std::min<Eigen::Index>(BMVP_LANES,
                                                                              first_row + m - row));
                      const simd_t shifted = nico_lse_subtract(nico_load_prefix_packet(t + row, lanes,
                                -std::numeric_limits<double>::infinity()), scan_set1(largest_eta));
                      const simd_t exponentials = nico_exp_of_nonpositive_prefix_argument(shifted);
                      nico_store_prefix_packet(t + row, lanes, exponentials);
                      sum_of_exponentials = nico_lse_add(sum_of_exponentials, exponentials);
                }

          }
          sum_of_exponentials_of_block[b] = nico_lse_sum(sum_of_exponentials);
          first_exponential_of_block[b] = t[first_row_of_the_block];

    });
    //// (a non-finite eta: the serial function, which reports it)
    if (std::find(eta_of_block_is_finite.begin(), eta_of_block_is_finite.end(), 0) !=
        eta_of_block_is_finite.end()) {

          return serial_function();

    }
    //// ---- (2) on the calling thread: each block's offset o_b (the log of exp(offset) plus the sum of exp(eta)
    ////      over every row before the block), its units s_b = max(m_b, o_b), and the check of its first running
    ////      sum
    std::vector<double> shift_of_block(n_blocks);
    std::vector<double> initial_sum_of_block(n_blocks);
    std::vector<double> rescaling_factor_of_block(n_blocks);
    const bool every_first_running_sum_is_representable =
          parallel_affine_prefix::nico_scan_parallel_block_totals(
                largest_eta_of_block.data(), sum_of_exponentials_of_block.data(),
                first_exponential_of_block.data(), n_blocks, offset_value, shift_of_block.data(),
                initial_sum_of_block.data(), rescaling_factor_of_block.data());
    //// (the rare case: the serial function, from eta again)
    if (!every_first_running_sum_is_representable) return serial_function();
    //// ---- (3) in parallel: each block in units of exp(s_b): exp(eta - s_b) (in place), its running sums C
    ////      from exp(o_b - s_b) (the last C of each block of 1024 rows carried to the next), the dot product of
    ////      the prefix weights with log C, prefix_weights / C and their sum R_b
    std::vector<double> prefix_part_of_block(n_blocks, 0.0);
    std::vector<double> sum_of_prefix_weights_over_running_sums_of_block(n_blocks, 0.0);
    nico::math::run_one_task_per_block_in_parallel(n_blocks, [&](const Eigen::Index b) {

          using namespace parallel_affine_prefix;
          auto& buffers_of_the_task =
                nico::math::buffers_of_one_task_of_affine_predictor_with_indexed_effects();
          buffers_of_the_task.ensure(cache_block_size);
          const Eigen::Index first_row_of_the_block = first_row_of_block(b);
          const Eigen::Index end_row_of_the_block = end_row_of_block(b);
          const double factor_to_the_units_of_the_block = rescaling_factor_of_block[b];
          double running_sum_before_the_rows = initial_sum_of_block[b];
          simd_t sum_of_prefix_weights_times_logs = scan_zero();
          simd_t sum_of_prefix_weights_over_running_sums = scan_zero();
          simd_t sum_of_prefix_weights = scan_zero();
          for (Eigen::Index first_row = first_row_of_the_block; first_row < end_row_of_the_block;
               first_row += cache_block_size) {

                const Eigen::Index m = std::min(cache_block_size, end_row_of_the_block - first_row);
                const Eigen::Index m_padded = nico::math::padded_to_lanes(m);
                for (Eigen::Index row = first_row; row < first_row + m; row += BMVP_LANES) {
                      const int lanes = static_cast<int>(std::min<Eigen::Index>(BMVP_LANES,
                                                                              first_row + m - row));
                      nico_store_prefix_packet(t + row, lanes, scan_mul(
                                  nico_load_prefix_packet(t + row, lanes, 0.0),
                                  scan_set1(factor_to_the_units_of_the_block)));
                }
                double* C_of_the_rows = buffers_of_the_task.running_sums_of_the_rows.data();
                nico::math::forward_affine_scan(t + first_row, m_padded, running_sum_before_the_rows,
                                                   C_of_the_rows, coefficients_of_one);
                running_sum_before_the_rows = C_of_the_rows[m - 1];
                for (Eigen::Index row = 0; row < m; row += BMVP_LANES) {
                      const int lanes = static_cast<int>(std::min<Eigen::Index>(BMVP_LANES, m - row));
                      const simd_t running = nico_load_prefix_packet(C_of_the_rows + row, lanes, 1.0);
                      const simd_t weights = nico_load_prefix_packet(
                                  prefix_values.data() + first_row + row, lanes, 0.0);
                      sum_of_prefix_weights_times_logs = scan_fmadd(weights,
                                  nico_lse_log_positive_normal(running), sum_of_prefix_weights_times_logs);
                      const simd_t divided = nico_lse_divide(weights, running);
                      nico_store_prefix_packet(derivatives_of_the_rows + first_row + row, lanes, divided);
                      sum_of_prefix_weights_over_running_sums = nico_lse_add(
                                  sum_of_prefix_weights_over_running_sums, divided);
                      sum_of_prefix_weights = nico_lse_add(sum_of_prefix_weights, weights);
                }

          }
          //// (sum_i prefix_weights[i] L[i] over the block, with L = log C + s_b)
          prefix_part_of_block[b] = nico_lse_sum(scan_fmadd(scan_set1(shift_of_block[b]),
                                    sum_of_prefix_weights, sum_of_prefix_weights_times_logs));
          sum_of_prefix_weights_over_running_sums_of_block[b] =
                                    nico_lse_sum(sum_of_prefix_weights_over_running_sums);

    });
    using namespace parallel_affine_prefix;
    simd_t block_values = scan_zero();
    for (Eigen::Index b = 0; b < n_blocks; b += BMVP_LANES) {
          const int lanes = static_cast<int>(std::min<Eigen::Index>(BMVP_LANES, n_blocks - b));
          block_values = nico_lse_add(block_values, nico_lse_add(
                      nico_load_prefix_packet(linear_part_of_block.data() + b, lanes, 0.0),
                      nico_load_prefix_packet(prefix_part_of_block.data() + b, lanes, 0.0)));
    }
    const double value = nico_lse_sum(block_values);
    if constexpr (stan::is_constant_all<T_beta, T_scale, T_z, T_offset>::value) {
          return value;
    } else {
    //// ---- (4) on the calling thread: each block's sum of prefix_weights / C over every row after it, in its
    ////      own units (the last block: none)
    std::vector<double> backward_sum_after_the_block(n_blocks, 0.0);
    std::vector<double> backward_ratio(n_blocks, 0.0);
    std::vector<double> inclusive_backward_sum(n_blocks, 0.0);
    for (Eigen::Index b = 0; b + 1 < n_blocks; b += BMVP_LANES) {
          const int lanes = static_cast<int>(std::min<Eigen::Index>(BMVP_LANES, n_blocks - 1 - b));
          const simd_t ratio = nico_exp_of_nonpositive_prefix_argument(nico_lse_subtract(
                      nico_load_prefix_packet(shift_of_block.data() + b, lanes, 0.0),
                      nico_load_prefix_packet(shift_of_block.data() + b + 1, lanes, 0.0)));
          nico_store_prefix_packet(backward_ratio.data() + b, lanes, ratio);
    }
    nico_backward_prefix_scan(sum_of_prefix_weights_over_running_sums_of_block.data(), backward_ratio.data(),
                              static_cast<int>(n_blocks), inclusive_backward_sum.data());
    for (Eigen::Index b = 0; b + 1 < n_blocks; b += BMVP_LANES) {
          const int lanes = static_cast<int>(std::min<Eigen::Index>(BMVP_LANES, n_blocks - 1 - b));
          nico_store_prefix_packet(backward_sum_after_the_block.data() + b, lanes, scan_mul(
                      nico_load_prefix_packet(backward_ratio.data() + b, lanes, 0.0),
                      nico_load_prefix_packet(inclusive_backward_sum.data() + b + 1, lanes, 0.0)));
    }
    //// ---- (5) in parallel: each block's q = the backward scan of prefix_weights / C from that sum (the first q
    ////      of each block of 1024 rows carried to the block before), the derivative of the result at each row,
    ////      linear_weights + exp(eta - s_b) q (in place of prefix_weights / C), and its product with the rows
    ////      of x
    const Eigen::Index K = beta_values.size();
    Eigen::Matrix<double, -1, -1> partials_of_beta_of_block = Eigen::Matrix<double, -1, -1>::Zero(K, n_blocks);
    double running_derivative_of_the_first_row = 0.0;
    nico::math::run_one_task_per_block_in_parallel(n_blocks, [&](const Eigen::Index b) {

          auto& buffers_of_the_task =
                nico::math::buffers_of_one_task_of_affine_predictor_with_indexed_effects();
          buffers_of_the_task.ensure(cache_block_size);
          const Eigen::Index first_row_of_the_block = first_row_of_block(b);
          const Eigen::Index end_row_of_the_block = end_row_of_block(b);
          double backward_running_sum_after_the_rows = backward_sum_after_the_block[b];
          const Eigen::Index first_row_of_the_last_rows =
                first_row_of_the_block +
                ((end_row_of_the_block - 1 - first_row_of_the_block) / cache_block_size) * cache_block_size;
          for (Eigen::Index first_row = first_row_of_the_last_rows; first_row >= first_row_of_the_block;
               first_row -= cache_block_size) {

                const Eigen::Index m = std::min(cache_block_size, end_row_of_the_block - first_row);
                const Eigen::Index m_padded = nico::math::padded_to_lanes(m);
                double* q_of_the_rows = buffers_of_the_task.backward_running_sums_of_the_rows.data();
                nico::math::backward_affine_scan(derivatives_of_the_rows + first_row, m_padded,
                                                    backward_running_sum_after_the_rows, q_of_the_rows,
                                                    coefficients_of_one);
                backward_running_sum_after_the_rows = q_of_the_rows[0];
                for (Eigen::Index row = 0; row < m; row += BMVP_LANES) {
                      const int lanes = static_cast<int>(std::min<Eigen::Index>(BMVP_LANES, m - row));
                      const simd_t derivative = scan_fmadd(
                                  nico_load_prefix_packet(t + first_row + row, lanes, 0.0),
                                  nico_load_prefix_packet(q_of_the_rows + row, lanes, 0.0),
                                  nico_load_prefix_packet(linear_values.data() + first_row + row, lanes, 0.0));
                      nico_store_prefix_packet(derivatives_of_the_rows + first_row + row, lanes, derivative);
                }

          }
          if constexpr (!stan::is_constant_all<T_beta>::value) {
                for (Eigen::Index column = 0; column < K; ++column) {
                      simd_t partial = scan_zero();
                      for (Eigen::Index row = first_row_of_the_block; row < end_row_of_the_block;
                           row += BMVP_LANES) {

                            const int lanes = static_cast<int>(std::min<Eigen::Index>(BMVP_LANES,
                                                                                    end_row_of_the_block - row));
                            partial = scan_fmadd(nico_load_parallel_matrix_column(x_ref, row, column, lanes),
                                        nico_load_prefix_packet(derivatives_of_the_rows + row, lanes, 0.0),
                                        partial);

                      }
                      partials_of_beta_of_block(column, b) = nico_lse_sum(partial);
                }
          }
          if (b == 0) running_derivative_of_the_first_row = backward_running_sum_after_the_rows;

    });
    //// ---- (6) in parallel over ranges of the groups: the sums of the derivatives over the rows of each group:
    ////      the derivatives of a range's rows in the order of the groups, their running sum (from zero, after a
    ////      leading zero) and its differences at the last position of each group
    Eigen::Matrix<double, -1, 1> sums_of_each_group = Eigen::Matrix<double, -1, 1>::Zero(J);
    if constexpr (!stan::is_constant_all<T_scale, T_z>::value) {
          second_vector[0] = 0.0;
          const Eigen::Index groups_per_range = (J + std::min<Eigen::Index>(n_blocks, J) - 1) /
                                                std::min<Eigen::Index>(n_blocks, J);
          const Eigen::Index n_ranges = (J + groups_per_range - 1) / groups_per_range;
          nico::math::run_one_task_per_block_in_parallel(n_ranges, [&](const Eigen::Index range) {

                auto& buffers_of_the_task =
                      nico::math::buffers_of_one_task_of_affine_predictor_with_indexed_effects();
                const Eigen::Index first_group = range * groups_per_range;
                const Eigen::Index n_groups = std::min(groups_per_range, J - first_group);
                const Eigen::Index first_position = (first_group == 0) ? 0 : last_positions[first_group - 1];
                const Eigen::Index end_position = last_positions[first_group + n_groups - 1];
                const Eigen::Index n_rows = end_position - first_position;
                if (n_rows == 0) return;
                const Eigen::Index n_rows_padded = nico::math::padded_to_lanes(n_rows);
                Eigen::Matrix<double, -1, 1>& running_sums = buffers_of_the_task.running_sums_in_order_of_index;
                if (running_sums.size() < n_rows_padded + 1) running_sums.resize(n_rows_padded + 1);
                running_sums[0] = 0.0;
                running_sums.segment(1, n_rows) =
                      second_vector.head(n + 1)(rows_in_order.segment(first_position, n_rows));
                running_sums.segment(1 + n_rows, n_rows_padded - n_rows).setZero();
                nico::math::forward_affine_scan(running_sums.data() + 1, n_rows_padded, 0.0,
                                                   running_sums.data() + 1, coefficients_of_one);
                for (Eigen::Index group = first_group; group < first_group + n_groups; group += BMVP_LANES) {
                      const int lanes = static_cast<int>(std::min<Eigen::Index>(BMVP_LANES,
                                                                              first_group + n_groups - group));
                      alignas(64) double end_values[BMVP_LANES] = {};
                      alignas(64) double start_values[BMVP_LANES] = {};
                      for (int lane = 0; lane < lanes; ++lane) {
                            const Eigen::Index g = group + lane;
                            end_values[lane] = running_sums[last_positions[g] - first_position];
                            if (g > first_group) {
                                  start_values[lane] = running_sums[last_positions[g - 1] - first_position];
                            }
                      }
                      nico_store_prefix_packet(sums_of_each_group.data() + group, lanes,
                                  nico_lse_subtract(simd_load(end_values), simd_load(start_values)));
                }

          });
    }
    ////
    //// ---- One callback with SIMD adjoint arithmetic, including shared operands across all arguments:
    ////      Address and value transfers are scalar memory operations; every derivative product, sum and
    ////      adjoint addition is packet arithmetic. Arena allocations live until Stan recovers the tape.
    ////
    const Eigen::Index n_beta_partials = stan::is_constant_all<T_beta>::value ? 0 : K;
    const Eigen::Index n_scale_partials = stan::is_constant_all<T_scale>::value ? 0 : 1;
    const Eigen::Index n_z_partials = stan::is_constant_all<T_z>::value ? 0 : J;
    const Eigen::Index n_offset_partials = stan::is_constant_all<T_offset>::value ? 0 : 1;
    const Eigen::Index n_partials = n_beta_partials + n_scale_partials + n_z_partials + n_offset_partials;
    auto& arena = stan::math::ChainableStack::instance_->memalloc_;
    double** adjoint_addresses = arena.alloc_array<double*>(n_partials);
    double* partial_values = arena.alloc_array<double>(n_partials);
    Eigen::Index position = 0;
    if constexpr (!stan::is_constant_all<T_beta>::value) {
          for (Eigen::Index column = 0; column < K; ++column) {
                simd_t partial = scan_zero();
                for (Eigen::Index b = 0; b < n_blocks; b += BMVP_LANES) {
                      const int lanes = static_cast<int>(std::min<Eigen::Index>(BMVP_LANES, n_blocks - b));
                      alignas(64) double values[BMVP_LANES] = {};
                      for (int lane = 0; lane < lanes; ++lane) {
                            values[lane] = partials_of_beta_of_block(column, b + lane);
                      }
                      partial = nico_lse_add(partial, simd_load(values));
                }
                partial_values[position] = nico_lse_sum(partial);
                if constexpr (stan::is_var_matrix<T_beta>::value) {
                      adjoint_addresses[position] = &beta_ref.adj().coeffRef(column);
                } else {
                      adjoint_addresses[position] = &beta_ref.coeff(column).adj();
                }
                ++position;
          }
    }
    if constexpr (!stan::is_constant_all<T_scale>::value) {
          //// (sum_i z[index[i]] d eta[i] = sum_g z[g] (the sum of d eta over the rows of group g))
          simd_t partial = scan_zero();
          for (Eigen::Index group = 0; group < J; group += BMVP_LANES) {
                const int lanes = static_cast<int>(std::min<Eigen::Index>(BMVP_LANES, J - group));
                partial = scan_fmadd(nico_load_prefix_packet(effects_with_a_leading_zero.data() + 1 + group,
                                                            lanes, 0.0),
                            nico_load_prefix_packet(sums_of_each_group.data() + group, lanes, 0.0), partial);
          }
          partial_values[position] = nico_lse_sum(partial);
          adjoint_addresses[position] = &scale.adj();
          ++position;
    }
    if constexpr (!stan::is_constant_all<T_z>::value) {
          for (Eigen::Index group = 0; group < J; group += BMVP_LANES) {
                const int lanes = static_cast<int>(std::min<Eigen::Index>(BMVP_LANES, J - group));
                nico_store_prefix_packet(partial_values + position + group, lanes,
                            scan_mul(scan_set1(scale_value),
                                nico_load_prefix_packet(sums_of_each_group.data() + group, lanes, 0.0)));
          }
          for (Eigen::Index group = 0; group < J; ++group) adjoint_addresses[position + group] = &z[group].adj();
          position += J;
    }
    if constexpr (!stan::is_constant_all<T_offset>::value) {
          //// (block 1's units are exp(s_1), with o_1 the offset)
          partial_values[position] = nico_lse_first(scan_mul(scan_set1(initial_sum_of_block[0]),
                                                             scan_set1(running_derivative_of_the_first_row)));
          adjoint_addresses[position] = &offset.adj();
    }
    return stan::math::make_callback_var(value, [adjoint_addresses, partial_values, n_partials](auto& result) {

          const simd_t upstream = scan_set1(result.adj());
          for (Eigen::Index first = 0; first < n_partials; first += BMVP_LANES) {
                const int lanes = static_cast<int>(std::min<Eigen::Index>(BMVP_LANES, n_partials - first));
                nico_add_prefix_adjoint_packet(adjoint_addresses + first, lanes,
                            nico_load_prefix_packet(partial_values + first, lanes, 0.0), upstream);
          }

    });
    }

}

}  // namespace nico::math

using nico::math::nico_dot_products_of_affine_predictor_with_indexed_effects_and_log_cumulative_sum_exp;






















