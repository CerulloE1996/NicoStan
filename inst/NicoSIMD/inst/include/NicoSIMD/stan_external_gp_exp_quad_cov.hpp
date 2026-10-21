#pragma once
//
// -| --------- BayesMVP AVX-512 / AVX2 squared exponential covariance, exposed to Stan as an external function --
//
// nico_gp_exp_quad_cov(x, sigma, length_scale) and nico_gp_exp_quad_cov(x1, x2, sigma, length_scale): drop-in
// replacements for Stan's gp_exp_quad_cov with array[] real data points and real sigma and length_scale (each
// data or a parameter):
//
//     K_ij = sigma^2 exp(-(x_i - x_j)^2 / (2 length_scale^2))
//
// The exponentials are BayesMVP's exp kernel (k_exp of stan_external_functions.hpp: 8 lanes with the AVX-512
// kernels, 4 with the AVX2 kernels, including -DBAYESMVP_FORCE_AVX2), one column segment at a time. With one x, K
// is symmetric: only the entries below the diagonal are computed (the segment below the diagonal of each column),
// each is shared by K_ij and K_ji, as in Stan, and the diagonal is sigma^2.
//
// Reverse mode (sigma or length_scale a parameter): every distinct entry is a vari on the nochain stack (its
// chain() is empty), as in Stan's own gp_exp_quad_cov, and ONE callback adds, over the distinct entries,
//
//     adj(sigma)        += sum adj(K_ij) 2 K_ij / sigma
//     adj(length_scale) += sum adj(K_ij) K_ij (x_i - x_j)^2 / length_scale^3
//
// with the squared distances recomputed from the points, which are copied on the arena. Arena per distinct entry:
// 24 bytes (vari); nothing else.
//
// Checks, as Stan's gp_exp_quad_cov: sigma positive, length_scale positive and finite, no point NaN.
//
// Declared in the Stan functions block (no body) as
//
//     matrix nico_gp_exp_quad_cov(data array[] real x, real sigma, real length_scale);
//     matrix nico_gp_exp_quad_cov(data array[] real x1, data array[] real x2, real sigma, real length_scale);
//
// ---------------------------------------------------------------------------------------------------------------

#include "stan_external_functions.hpp"
#include "nico_simd_exp_in_double_precision.hpp"

namespace nico::math {
namespace gp_exp_quad_cov {

static constexpr const char* function_name = "nico_gp_exp_quad_cov";

////
//// ---- SIMD registers and the covariance entries' values and adjoints:
////
//// simd_t is one register of double-precision values: __m256d contains four doubles on AVX2;
//// __m512d contains eight doubles on AVX-512. BMVP_LANES is the corresponding number of doubles.
//// vari is Stan's reverse-mode node; val_ stores its value and adj_ its accumulated derivative.
//// The nodes are contiguous arena objects, not contiguous doubles. Gather offsets therefore use
//// sizeof(vari), rather than assuming a particular node size or copying the fields into another array.
//// count is the number of valid entries in this packet, from one to BMVP_LANES.
////
inline simd_t nico_gp_covariance_products_of_values_and_adjoints(const vari* nodes, const int count) {

#if BMVP_LANES == 8
    const __m512i byte_offsets = _mm512_set_epi64(7 * sizeof(vari), 6 * sizeof(vari),
                                                5 * sizeof(vari), 4 * sizeof(vari),
                                                3 * sizeof(vari), 2 * sizeof(vari), sizeof(vari), 0);
    const __mmask8 valid_lanes = count == 8 ? static_cast<__mmask8>(0xFF)
                                           : static_cast<__mmask8>((1u << count) - 1u);
    const simd_t values = _mm512_mask_i64gather_pd(_mm512_setzero_pd(), valid_lanes, byte_offsets,
                                                  &nodes[0].val_, 1);
    const simd_t adjoints = _mm512_mask_i64gather_pd(_mm512_setzero_pd(), valid_lanes, byte_offsets,
                                                    &nodes[0].adj_, 1);
    return _mm512_mul_pd(values, adjoints);
#else
    const __m256i byte_offsets = _mm256_set_epi64x(3 * sizeof(vari), 2 * sizeof(vari), sizeof(vari), 0);
    const __m256i valid_lanes = _mm256_set_epi64x(count > 3 ? -1 : 0, count > 2 ? -1 : 0,
                                                 count > 1 ? -1 : 0, -1);
    const simd_t mask = _mm256_castsi256_pd(valid_lanes);
    const simd_t values = _mm256_mask_i64gather_pd(_mm256_setzero_pd(), &nodes[0].val_,
                                                  byte_offsets, mask, 1);
    const simd_t adjoints = _mm256_mask_i64gather_pd(_mm256_setzero_pd(), &nodes[0].adj_,
                                                    byte_offsets, mask, 1);
    return _mm256_mul_pd(values, adjoints);
#endif

}


////
//// ---- Sum a fixed-width register, keeping the additions in SIMD registers:
////
inline double nico_gp_covariance_sum_of_packet(const simd_t values) {

#if BMVP_LANES == 8
    return _mm512_reduce_add_pd(values);
#else
    const __m128d halves = _mm_add_pd(_mm256_castpd256_pd128(values),
                                     _mm256_extractf128_pd(values, 1));
    return _mm_cvtsd_f64(_mm_add_sd(halves, _mm_unpackhi_pd(halves, halves)));
#endif

}


// ---- one column segment: out[i] = sigma^2 exp(-0.5 (x_rows[i] - x_column)^2 / length_scale^2), i < n
inline void column_segment(const double* x_rows, const int n, const double x_column,
                           const double minus_half_inverse_squared_length_scale, const double squared_sigma,
                           double* out) {

    ////
    //// ---- Distances, squared distances, exponential arguments and covariance entries:
    ////
    //// The final partial packet uses masked memory operations, not scalar mathematical work.
    //// The squared distance is formed before multiplication by the length-scale coefficient.
    ////
#if BMVP_LANES == 8
    const simd_t column_point = _mm512_set1_pd(x_column);
    const simd_t coefficient = _mm512_set1_pd(minus_half_inverse_squared_length_scale);
    const simd_t magnitude = _mm512_set1_pd(squared_sigma);

    for (int i = 0; i < n; i += BMVP_LANES) {
        const int count = std::min(BMVP_LANES, n - i);
        const __mmask8 valid_lanes = count == 8 ? static_cast<__mmask8>(0xFF)
                                               : static_cast<__mmask8>((1u << count) - 1u);
        const simd_t distance = _mm512_maskz_sub_pd(valid_lanes,
                                                    _mm512_maskz_loadu_pd(valid_lanes, x_rows + i),
                                                    column_point);
        const simd_t distance_squared = _mm512_mul_pd(distance, distance);
        const simd_t argument = _mm512_maskz_mul_pd(valid_lanes, distance_squared, coefficient);
        const simd_t value = _mm512_mul_pd(nico_simd_exp_in_double_precision(argument), magnitude);
        _mm512_mask_storeu_pd(out + i, valid_lanes, value);
    }
#else
    const simd_t column_point = _mm256_set1_pd(x_column);
    const simd_t coefficient = _mm256_set1_pd(minus_half_inverse_squared_length_scale);
    const simd_t magnitude = _mm256_set1_pd(squared_sigma);

    for (int i = 0; i < n; i += BMVP_LANES) {
        const int count = std::min(BMVP_LANES, n - i);
        const __m256i valid_lanes = _mm256_set_epi64x(count > 3 ? -1 : 0, count > 2 ? -1 : 0,
                                                     count > 1 ? -1 : 0, -1);
        const simd_t distance = _mm256_and_pd(_mm256_sub_pd(_mm256_maskload_pd(x_rows + i, valid_lanes),
                                                           column_point),
                                              _mm256_castsi256_pd(valid_lanes));
        const simd_t distance_squared = _mm256_mul_pd(distance, distance);
        const simd_t argument = _mm256_and_pd(_mm256_mul_pd(distance_squared, coefficient),
                                              _mm256_castsi256_pd(valid_lanes));
        const simd_t value = _mm256_mul_pd(nico_simd_exp_in_double_precision(argument), magnitude);
        _mm256_maskstore_pd(out + i, valid_lanes, value);
    }
#endif

}

template <typename T_sigma, typename T_length_scale>
inline void check_arguments(const std::vector<double>& x, const T_sigma& sigma,
                            const T_length_scale& length_scale) {

    stan::math::check_positive(function_name, "magnitude", sigma);
    stan::math::check_positive_finite(function_name, "length scale", length_scale);
    stan::math::check_not_nan(function_name, "x", x);

}

// ---- the reverse pass: the sums over the distinct entries, then the adjoints of sigma and length_scale
struct reverse_pass_of_covariance {
    const double* x_rows      = nullptr;       // the points of the rows (x, or x1)
    const double* x_columns   = nullptr;       // the points of the columns (x, or x2)
    int           n_rows      = 0;
    int           n_columns   = 0;
    bool          symmetric   = false;         // one x: the entries below the diagonal, then the diagonal
    vari*         entries     = nullptr;       // the distinct entries, column by column
    vari*         diagonal    = nullptr;       // (symmetric only)
    vari*         sigma_node  = nullptr;       // nullptr: data
    vari*         length_scale_node = nullptr; // nullptr: data
    double        sigma_value = 0.0;
    double        length_scale_value = 0.0;

    void operator()() const {

        ////
        //// ---- Covariance derivatives, accumulated in packets over the distinct matrix entries:
        ////
        //// Symmetric entries share one node, so its adjoint already includes both returned positions.
        //// Masking the distance also keeps inactive tail lanes out of the derivative reductions.
        //// Node construction, matrix-address selection and copying remain memory/control operations.
        ////
#if BMVP_LANES == 8
        simd_t sums_of_products = _mm512_setzero_pd();
        simd_t sums_of_products_times_squared_distances = _mm512_setzero_pd();
#else
        simd_t sums_of_products = _mm256_setzero_pd();
        simd_t sums_of_products_times_squared_distances = _mm256_setzero_pd();
#endif
        std::ptrdiff_t position = 0;

        for (int j = 0; j < n_columns; ++j) {
            const int first_row = symmetric ? j + 1 : 0;

            for (int i = first_row; i < n_rows; i += BMVP_LANES) {
                const int count = std::min(BMVP_LANES, n_rows - i);
                const simd_t product = nico_gp_covariance_products_of_values_and_adjoints(entries + position,
                                                                                           count);
#if BMVP_LANES == 8
                const __mmask8 valid_lanes = count == 8 ? static_cast<__mmask8>(0xFF)
                                                       : static_cast<__mmask8>((1u << count) - 1u);
                const simd_t distance = _mm512_maskz_sub_pd(valid_lanes,
                                                            _mm512_maskz_loadu_pd(valid_lanes, x_rows + i),
                                                            _mm512_set1_pd(x_columns[j]));
                const simd_t distance_squared = _mm512_mul_pd(distance, distance);
                sums_of_products = _mm512_add_pd(sums_of_products, product);
                sums_of_products_times_squared_distances
                    = _mm512_add_pd(sums_of_products_times_squared_distances,
                                     _mm512_mul_pd(product, distance_squared));
#else
                const __m256i valid_lanes = _mm256_set_epi64x(count > 3 ? -1 : 0, count > 2 ? -1 : 0,
                                                             count > 1 ? -1 : 0, -1);
                const simd_t distance = _mm256_and_pd(_mm256_sub_pd(_mm256_maskload_pd(x_rows + i,
                                                                                     valid_lanes),
                                                                   _mm256_set1_pd(x_columns[j])),
                                                      _mm256_castsi256_pd(valid_lanes));
                const simd_t distance_squared = _mm256_mul_pd(distance, distance);
                sums_of_products = _mm256_add_pd(sums_of_products, product);
                sums_of_products_times_squared_distances
                    = _mm256_add_pd(sums_of_products_times_squared_distances,
                                     _mm256_mul_pd(product, distance_squared));
#endif
                position += count;
            }
        }

        if (symmetric) {
            for (int j = 0; j < n_columns; j += BMVP_LANES) {
                const int count = std::min(BMVP_LANES, n_columns - j);
                const simd_t product = nico_gp_covariance_products_of_values_and_adjoints(diagonal + j, count);
#if BMVP_LANES == 8
                sums_of_products = _mm512_add_pd(sums_of_products, product);
#else
                sums_of_products = _mm256_add_pd(sums_of_products, product);
#endif
            }
        }

        const double sum_of_adjoint_times_entry = nico_gp_covariance_sum_of_packet(sums_of_products);
        const double sum_of_adjoint_times_entry_times_squared_distance
            = nico_gp_covariance_sum_of_packet(sums_of_products_times_squared_distances);

        if (sigma_node != nullptr) sigma_node->adj_ += 2.0 * sum_of_adjoint_times_entry / sigma_value;
        if (length_scale_node != nullptr) {
            length_scale_node->adj_ += sum_of_adjoint_times_entry_times_squared_distance
                                       / (length_scale_value * length_scale_value * length_scale_value);
        }

    }
};

template <typename T_sigma, typename T_length_scale>
inline Eigen::Matrix<stan::return_type_t<T_sigma, T_length_scale>, -1, -1>
covariance(const std::vector<double>& x_rows, const std::vector<double>& x_columns, const bool symmetric,
           const T_sigma& sigma, const T_length_scale& length_scale) {

    using return_t = stan::return_type_t<T_sigma, T_length_scale>;
    constexpr bool any_var = stan::is_var<return_t>::value;
    const int n_rows    = static_cast<int>(x_rows.size());
    const int n_columns = static_cast<int>(x_columns.size());
    const double sigma_value        = stan::math::value_of(sigma);
    const double length_scale_value = stan::math::value_of(length_scale);
    const double squared_sigma      = sigma_value * sigma_value;
    const double minus_half_inverse_squared_length_scale = -0.5 / (length_scale_value * length_scale_value);
    Eigen::Matrix<return_t, -1, -1> result(n_rows, n_columns);
    if (n_rows == 0 || n_columns == 0) return result;

    if constexpr (!any_var) {
        for (int j = 0; j < n_columns; ++j) {
            const int first_row = symmetric ? j + 1 : 0;
            column_segment(x_rows.data() + first_row, n_rows - first_row, x_columns[j],
                           minus_half_inverse_squared_length_scale, squared_sigma,
                           result.data() + static_cast<std::ptrdiff_t>(j) * n_rows + first_row);
        }
        if (symmetric) {
            for (int j = 0; j < n_columns; ++j) {
                result(j, j) = squared_sigma;
                for (int i = j + 1; i < n_rows; ++i) result(j, i) = result(i, j);
            }
        }
        return result;
    } else {
        auto& arena = stan::math::ChainableStack::instance_->memalloc_;
        const std::ptrdiff_t n_distinct_entries = symmetric
            ? static_cast<std::ptrdiff_t>(n_rows) * (n_rows - 1) / 2
            : static_cast<std::ptrdiff_t>(n_rows) * n_columns;
        reverse_pass_of_covariance reverse_pass;
        double* x_rows_on_arena = arena.alloc_array<double>(n_rows);
        std::copy(x_rows.begin(), x_rows.end(), x_rows_on_arena);
        double* x_columns_on_arena = x_rows_on_arena;
        if (!symmetric) {
            x_columns_on_arena = arena.alloc_array<double>(n_columns);
            std::copy(x_columns.begin(), x_columns.end(), x_columns_on_arena);
        }
        reverse_pass.x_rows      = x_rows_on_arena;
        reverse_pass.x_columns   = x_columns_on_arena;
        reverse_pass.n_rows      = n_rows;
        reverse_pass.n_columns   = n_columns;
        reverse_pass.symmetric   = symmetric;
        reverse_pass.entries     = arena.alloc_array<vari>(n_distinct_entries > 0 ? n_distinct_entries : 1);
        reverse_pass.diagonal    = symmetric ? arena.alloc_array<vari>(n_rows) : nullptr;
        reverse_pass.sigma_value = sigma_value;
        reverse_pass.length_scale_value = length_scale_value;
        if constexpr (stan::is_var<T_sigma>::value)        reverse_pass.sigma_node = sigma.vi_;
        if constexpr (stan::is_var<T_length_scale>::value) reverse_pass.length_scale_node = length_scale.vi_;

        // std::vector<double> column_values(n_rows);
        // std::ptrdiff_t position = 0;
        // for (int j = 0; j < n_columns; ++j) {
        //     const int first_row = symmetric ? j + 1 : 0;
        //     const int count     = n_rows - first_row;
        //     column_segment(x_rows.data() + first_row, count, x_columns[j],
        //                    minus_half_inverse_squared_length_scale, squared_sigma, column_values.data());
        //     for (int k = 0; k < count; ++k, ++position) {
        //         vari* node = reverse_pass.entries + position;
        //         ::new (static_cast<void*>(node)) vari(column_values[k], false);     // nochain stack
        //         result.coeffRef(first_row + k, j) = var(node);
        //         if (symmetric) result.coeffRef(j, first_row + k) = var(node);
        //     }
        //     if (symmetric) {
        //         vari* node = reverse_pass.diagonal + j;
        //         ::new (static_cast<void*>(node)) vari(squared_sigma, false);         // nochain stack
        //         result.coeffRef(j, j) = var(node);
        //     }
        // }
        // ---- (1) the values of the distinct entries, column segment by column segment (BayesMVP's exp kernel over
        //      each whole segment), into one buffer in the order of the entries
        scratch_t& buffers = scratch();
        buffers.ensure(static_cast<int>(n_distinct_entries > 0 ? n_distinct_entries : 1));
        double* values_of_entries = buffers.a.data();
        std::ptrdiff_t position = 0;
        for (int j = 0; j < n_columns; ++j) {
            const int first_row = symmetric ? j + 1 : 0;
            const int count     = n_rows - first_row;
            column_segment(x_rows.data() + first_row, count, x_columns[j],
                           minus_half_inverse_squared_length_scale, squared_sigma, values_of_entries + position);
            position += count;
        }
        // ---- (2) the nodes of the entries, in a loop of their own (as in rev_apply)
        for (std::ptrdiff_t p = 0; p < n_distinct_entries; ++p) {
            ::new (static_cast<void*>(reverse_pass.entries + p)) vari(values_of_entries[p],
                   false); // nochain stack
        }
        // ---- (3) the returned matrix: with two sets of points, column by column; with one, in blocks of
        //      size_of_block x size_of_block (as Stan's gp_exp_quad_cov), so that the writes of K_ij and of its
        //      mirror K_ji stay in nearby memory
        if (!symmetric) {
            for (int j = 0; j < n_columns; ++j) {
                var* column = result.data() + static_cast<std::ptrdiff_t>(j) * n_rows;
                vari* nodes_of_column = reverse_pass.entries + static_cast<std::ptrdiff_t>(j) * n_rows;
                for (int i = 0; i < n_rows; ++i) column[i] = var(nodes_of_column + i);
            }
        } else {
            constexpr int size_of_block = 16;
            for (int j = 0; j < n_columns; ++j) {
                vari* node = reverse_pass.diagonal + j;
                ::new (static_cast<void*>(node)) vari(squared_sigma, false);                 // nochain stack
                result.coeffRef(j, j) = var(node);
            }
            for (int first_column_of_block = 0; first_column_of_block < n_columns;
                 first_column_of_block += size_of_block) {

                const int end_column_of_block = std::min(n_columns, first_column_of_block + size_of_block);
                for (int first_row_of_block = first_column_of_block; first_row_of_block < n_rows;
                     first_row_of_block += size_of_block) {

                    const int end_row_of_block = std::min(n_rows, first_row_of_block + size_of_block);
                    for (int j = first_column_of_block; j < end_column_of_block; ++j) {
                        // (column j's entries start after those of columns 0, ..., j - 1: n_rows - 1 - c each)
                        const std::ptrdiff_t start_of_column = static_cast<std::ptrdiff_t>(j) * (n_rows - 1)
                                                               - static_cast<std::ptrdiff_t>(j) * (j - 1) / 2;
                        for (int i = std::max(first_row_of_block, j + 1); i < end_row_of_block; ++i) {
                            vari* node = reverse_pass.entries + start_of_column + (i - j - 1);
                            result.coeffRef(i, j) = var(node);
                            result.coeffRef(j, i) = var(node);
                        }
                    }

                }

            }
        }
        // (moved: the callback stores the functor by value only when it is passed as an rvalue)
        stan::math::reverse_pass_callback(std::move(reverse_pass));
        return result;
    }

}

}  // namespace gp_exp_quad_cov
}  // namespace nico::math

// ---- the Stan-facing functions (global scope; stanc calls them as  nico_gp_exp_quad_cov(x, sigma, length_scale,
// pstream__)); the points must be data
namespace nico::math {

template <typename T_x, typename T_sigma, typename T_length_scale,
          stan::require_all_stan_scalar_t<T_sigma, T_length_scale>* = nullptr>
inline Eigen::Matrix<stan::return_type_t<T_sigma, T_length_scale>, -1, -1>
nico_gp_exp_quad_cov(const std::vector<T_x>& x, const T_sigma& sigma, const T_length_scale& length_scale,
                     std::ostream* /*pstream__*/) {

    static_assert(std::is_arithmetic<T_x>::value, "nico_gp_exp_quad_cov: the points x must be data");
    const std::vector<double> x_as_double(x.begin(), x.end());
    nico::math::gp_exp_quad_cov::check_arguments(x_as_double, sigma, length_scale);
    return nico::math::gp_exp_quad_cov::covariance(x_as_double, x_as_double, true, sigma, length_scale);

}

}  // namespace nico::math

using nico::math::nico_gp_exp_quad_cov;

namespace nico::math {

template <typename T_x1, typename T_x2, typename T_sigma, typename T_length_scale,
          stan::require_all_stan_scalar_t<T_sigma, T_length_scale>* = nullptr>
inline Eigen::Matrix<stan::return_type_t<T_sigma, T_length_scale>, -1, -1>
nico_gp_exp_quad_cov(const std::vector<T_x1>& x1, const std::vector<T_x2>& x2, const T_sigma& sigma,
                     const T_length_scale& length_scale, std::ostream* /*pstream__*/) {

    static_assert(std::is_arithmetic<T_x1>::value && std::is_arithmetic<T_x2>::value,
                  "nico_gp_exp_quad_cov: the points x1 and x2 must be data");
    const std::vector<double> x1_as_double(x1.begin(), x1.end());
    const std::vector<double> x2_as_double(x2.begin(), x2.end());
    nico::math::gp_exp_quad_cov::check_arguments(x1_as_double, sigma, length_scale);
    nico::math::gp_exp_quad_cov::check_arguments(x2_as_double, sigma, length_scale);
    return nico::math::gp_exp_quad_cov::covariance(x1_as_double, x2_as_double, false, sigma, length_scale);

}

}  // namespace nico::math

using nico::math::nico_gp_exp_quad_cov;
























