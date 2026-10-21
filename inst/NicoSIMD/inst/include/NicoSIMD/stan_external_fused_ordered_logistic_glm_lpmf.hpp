#pragma once
//
// -| --------- BayesMVP fused ordered logistic GLM log probability (AVX-512 / AVX2), an external Stan lpmf ------
//
// nico_ordered_logistic_glm_lpmf(y | x, beta, cuts): a drop-in replacement for Stan's ordered_logistic_glm_lpmf
// (the same arguments, value, argument checks and errors), i.e. the log probability
//
//     sum_i log( inv_logit(eta_i - c_{y_i - 1}) - inv_logit(eta_i - c_{y_i}) ),    eta = x beta,
//                                                                                   c_0 = -inf,  c_K = +inf,
//
// of categories y_i in 1, ..., K with ordered cutpoints c_1 < ... < c_{K-1}, for any ordinal regression (an
// ordered logistic model with any number of predictors; e.g. a graded response IRT model with the persons'
// abilities as the one column of x and an item's discrimination as beta, or a proportional-odds regression with a
// design matrix). It is computed in ONE autodiff node (one callback vari, one reverse-pass loop):
//
//   (1) the linear predictor eta = x beta, column by column of x in SIMD (one fused multiply-add per value; the
//       values of a parameter x gathered from its vari pointers in SIMD, and kept for the gradient of beta);
//   (2) u = eta - c_{y-1} and v = eta - c_y, the cutpoints gathered in SIMD from the per-call table of the K
//       categories by the categories themselves (with the range test of the categories in the same pass);
//   (3) the fused pass of nico_ordered_logistic_lpmf (stan_external_fused_ordered_logistic_lpmf.hpp: its
//       fused_pass(), with the same formulation, kernels and precision), which gives the value and, in the same
//       pass, g_i = d log P / d eta_i = F(-u_i) - F(v_i) (F = inv_logit) and the sums of the cutpoint partials;
//   (4) the gradient, directly: d / d beta = x^T g (one SIMD dot product per column of x, in the forward pass),
//       d / d x = g beta^T (only for a parameter x: g and the values of beta are kept, and the reverse pass adds
//       adj(result) beta_p g_i to each x_ip, the products in SIMD), and d / d cuts as
//       nico_ordered_logistic_lpmf's (and Stan's):
//       d / d c_m = sum F(v) (category m) - sum F(-u) (category m + 1) + n_m A_m - n_{m+1} A_{m+1},
//       A_k = 1 / expm1(c_k - c_{k-1}).
//
// No tape node per response: with data x (any GLM with a design matrix), nothing per response is kept on the
// arena (the partials of beta and of the cutpoints are computed in the forward pass); with a parameter x, two
// values per value of x (its vari pointer and g of its response: 16 bytes per response for one column), as
// Stan's ordered_logistic_glm_lpmf keeps (its operands and partials of x).
//
// A row vector x is one row for every response (Stan's broadcast): eta is then one value, d / d x = beta sum(g)
// and d / d beta = x^T sum(g). A single int y is the category of every response.
//
// Precision: that of nico_ordered_logistic_lpmf (nico_exp_1, relative error about 2e-14; nico_log_1, about
// 1.6e-8, in the log1p term of each response's value only, an absolute error of at most about 5e-9 per
// response); eta, every partial and every other term are computed in double precision.
//
// Stan declarations (the signatures of Stan's ordered_logistic_glm_lpmf; in the functions block, no body, and
// compiled with --allow-undefined):
//
//     real nico_ordered_logistic_glm_lpmf(array[] int y, matrix x, vector beta, vector cuts);
//     real nico_ordered_logistic_glm_lpmf(array[] int y, row_vector x, vector beta, vector cuts);
//     real nico_ordered_logistic_glm_lpmf(int y, matrix x, vector beta, vector cuts);
//     real nico_ordered_logistic_glm_lpmf(int y, row_vector x, vector beta, vector cuts);
//
// with data or parameters for x, beta and cuts; stanc calls them as
// nico_ordered_logistic_glm_lpmf<propto__>(y, x, beta, cuts, pstream__), e.g. for
// y ~ nico_ordered_logistic_glm(x, beta, cuts)  or  nico_ordered_logistic_glm_lupmf(y | x, beta, cuts). Stan's
// ordered_logistic_glm has no signature with an array of cutpoint vectors, so neither has this. The checks and
// errors are Stan's, in Stan's order (the categories' range is tested in the SIMD pass of (2), and Stan's own
// check_bounded throws its error when a category is out of range); with propto = true and data x, beta and cuts
// the result is 0, as Stan's.
//
// Include: from stan_external_functions.hpp, after stan_external_fused_ordered_logistic_lpmf.hpp (it uses that
// header's namespace and fused pass); the includes below make it self-sufficient when it is included on its own.
//
// ------------------------------------------------------------------------------------------------------------

#include "stan_external_functions.hpp"
#include "stan_external_fused_ordered_logistic_lpmf.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <ostream>
#include <type_traits>
#include <vector>

namespace nico::math {
namespace fused_ordered_logistic_glm {

using stan::math::var;
using stan::math::vari;

static constexpr int lanes = BMVP_LANES;
static constexpr const char* function_name = "nico_ordered_logistic_glm";

// (a var is exactly one vari pointer, so that a column of a var matrix is a contiguous array of vari pointers)
static_assert(sizeof(var) == sizeof(vari*), "nico_ordered_logistic_glm_lpmf: a var must be one vari pointer");

// ---- SIMD primitives of the GLM: the values of LANES varis, gathered from their vari pointers (the address of
// each value is its vari pointer plus the offset of val_); table[k] at LANES categories; the categories outside
// 1, ..., K replaced by 0 (the padding entry of the tables: -inf and +inf, which contribute exactly 0)
#if BMVP_LANES == 8
using simd_category_t = __m256i;
ALWAYS_INLINE simd_t values_at_offset_of_varis(vari* const* varis, const long long offset_in_vari) {

    ALIGN64  __m512i const addresses = _mm512_add_epi64(_mm512_loadu_si512(reinterpret_cast<const void*>(varis)),
                                               _mm512_set1_epi64(offset_in_vari));
    return _mm512_i64gather_pd(addresses, nullptr, 1);

}
ALWAYS_INLINE simd_category_t load_categories(const int* p) {

    return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));

}
ALWAYS_INLINE simd_t table_at_categories(const double* table, const simd_category_t k) {

    return _mm512_i32gather_pd(k, table, 8);

}
ALWAYS_INLINE simd_category_t categories_in_range_or_0(const simd_category_t k, const int K,
                                                       unsigned& any_category_out_of_range) {

    const __mmask8 in_range = _mm256_cmpgt_epi32_mask(k, _mm256_setzero_si256())
                              & _mm256_cmpgt_epi32_mask(_mm256_set1_epi32(K + 1), k);
    any_category_out_of_range |= static_cast<unsigned>(in_range != 0xFF);
    return _mm256_maskz_mov_epi32(in_range, k);

}
#else
using simd_category_t = __m128i;
ALWAYS_INLINE simd_t values_at_offset_of_varis(vari* const* varis, const long long offset_in_vari) {

    ALIGN32  __m256i const addresses =
            _mm256_add_epi64(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(varis)),
                                               _mm256_set1_epi64x(offset_in_vari));
    return _mm256_i64gather_pd(static_cast<const double*>(nullptr), addresses, 1);

}
ALWAYS_INLINE simd_category_t load_categories(const int* p) {

    return _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));

}
ALWAYS_INLINE simd_t table_at_categories(const double* table, const simd_category_t k) {

    return _mm256_i32gather_pd(table, k, 8);

}
ALWAYS_INLINE simd_category_t categories_in_range_or_0(const simd_category_t k, const int K,
                                                       unsigned& any_category_out_of_range) {

    const __m128i in_range = _mm_and_si128(_mm_cmpgt_epi32(k, _mm_setzero_si128()),
                                           _mm_cmpgt_epi32(_mm_set1_epi32(K + 1), k));
    any_category_out_of_range |= static_cast<unsigned>(_mm_movemask_epi8(in_range) != 0xFFFF);
    return _mm_and_si128(in_range, k);

}
#endif

// the byte offsets of val_ and adj_ in a vari (the same for every vari)
inline long long offset_of_value_in_vari(const vari* v) {

    return static_cast<long long>(reinterpret_cast<const char*>(&v->val_) - reinterpret_cast<const char*>(v));

}
inline long long offset_of_adjoint_in_vari(const vari* v) {

    return static_cast<long long>(reinterpret_cast<const char*>(&v->adj_) - reinterpret_cast<const char*>(v));

}

// ---- the reverse pass of a parameter x: adj_ of each of n varis += factor * partials[i]. The products in SIMD;
// with AVX-512 conflict detection, the LANES adjoints gathered, added to and scattered back when the LANES varis
// are distinct (always, unless x repeats a parameter), otherwise added one after the other; with AVX2 (which has
// no scatter instruction), added one after the other.
inline void add_to_adjoints_of_varis(vari* const* varis, const double* partials, const double factor,
                                     const int n) {

    if (n == 0) return;
    const simd_t factor_in_lanes = fused_ordered_logistic::broadcast(factor);
    int i = 0;
#if BMVP_LANES == 8 && defined(__AVX512CD__)
    ALIGN64  __m512i const offset_of_adjoint_in_lanes = _mm512_set1_epi64(offset_of_adjoint_in_vari(varis[0]));
    for (; i + lanes <= n; i += lanes) {
        ALIGN64  __m512i const vari_addresses = _mm512_loadu_si512(reinterpret_cast<const void*>(varis + i));
        const simd_t contributions = _mm512_mul_pd(factor_in_lanes, simd_loadu(partials + i));
        ALIGN64  __m512i const conflicts = _mm512_conflict_epi64(vari_addresses);
        if (_mm512_test_epi64_mask(conflicts, conflicts) == 0) {
            ALIGN64  __m512i const adjoint_addresses =
                    _mm512_add_epi64(vari_addresses, offset_of_adjoint_in_lanes);
            const simd_t adjoints = _mm512_i64gather_pd(adjoint_addresses, nullptr, 1);
            _mm512_i64scatter_pd(nullptr, adjoint_addresses, _mm512_add_pd(adjoints, contributions), 1);
        } else {
            alignas(64) double contribution_of_each_lane[lanes];
            simd_store(contribution_of_each_lane, contributions);
            for (int j = 0; j < lanes; ++j) varis[i + j]->adj_ += contribution_of_each_lane[j];
        }
    }
#else
    for (; i + lanes <= n; i += lanes) {
        alignas(64) double contribution_of_each_lane[lanes];
        simd_store(contribution_of_each_lane, fused_ordered_logistic::multiply(factor_in_lanes,
                                                                               simd_loadu(partials + i)));
        for (int j = 0; j < lanes; ++j) varis[i + j]->adj_ += contribution_of_each_lane[j];
    }
#endif
    // (the last n mod LANES)
    for (; i < n; ++i) varis[i]->adj_ += factor * partials[i];

}

// ---- the dot product of two arrays of n values (x^T g of one column), and the sum of an array, in SIMD
inline double dot_product(const double* a, const double* b, const int n) {

    simd_t sum = fused_ordered_logistic::broadcast(0.0);
    int i = 0;
    for (; i + lanes <= n; i += lanes) {
        sum = fused_ordered_logistic::multiply_add(simd_loadu(a + i), simd_loadu(b + i), sum);
    }
    double total = fused_ordered_logistic::horizontal_sum(sum);
    for (; i < n; ++i) total += a[i] * b[i];
    return total;

}
inline double sum_of(const double* a, const int n) {

    simd_t sum = fused_ordered_logistic::broadcast(0.0);
    int i = 0;
    for (; i + lanes <= n; i += lanes) sum = fused_ordered_logistic::add(sum, simd_loadu(a + i));
    double total = fused_ordered_logistic::horizontal_sum(sum);
    for (; i < n; ++i) total += a[i];
    return total;

}

// ---- thread-local buffers: the values of a parameter x (column by column), eta, u, v, g (when the reverse pass
// does not need it), the cutpoint values, and the column pointers of x
struct buffers_t {
    std::vector<double> x_values;
    std::vector<double> eta;
    std::vector<double> u;
    std::vector<double> v;
    std::vector<double> eta_partials;
    std::vector<double> cutpoint_values;
    std::vector<const double*> value_columns;
    std::vector<vari* const*> vari_columns;
    void ensure(const int n, const int n_x_values, const int n_cutpoints, const int P) {

        if (static_cast<int>(eta.size()) < n + lanes) {
            eta.resize(n + lanes);
            u.resize(n + lanes);
            v.resize(n + lanes);
            eta_partials.resize(n + lanes);
        }
        if (static_cast<int>(x_values.size()) < n_x_values) x_values.resize(n_x_values);
        if (static_cast<int>(cutpoint_values.size()) < n_cutpoints) cutpoint_values.resize(n_cutpoints);
        if (static_cast<int>(value_columns.size()) < P) {
            value_columns.resize(P);
            vari_columns.resize(P);
        }

    }
};
inline buffers_t& buffers() { thread_local buffers_t b; return b; }

// ---- the whole function (Stan's (array[] int | matrix or row_vector, vector, vector) and its single-int forms)
template <bool propto, typename T_y, typename T_x, typename T_beta, typename T_cuts>
inline stan::return_type_t<T_x, T_beta, T_cuts> ordered_logistic_glm(const T_y& y, const T_x& x,
                                                                     const T_beta& beta_in,
                                                                     const T_cuts& cuts_in) {

    using return_t = stan::return_type_t<T_x, T_beta, T_cuts>;
    using scalar_of_x_t = stan::value_type_t<T_x>;
    constexpr bool x_is_var        = stan::is_var<stan::scalar_type_t<T_x>>::value;
    constexpr bool beta_is_var     = stan::is_var<stan::scalar_type_t<T_beta>>::value;
    constexpr bool cuts_is_var     = stan::is_var<stan::scalar_type_t<T_cuts>>::value;
    constexpr bool x_is_row_vector = (std::decay_t<T_x>::RowsAtCompileTime == 1);
    constexpr bool eta_partials_needed = x_is_var || beta_is_var;
    // (g is kept on the arena for the reverse pass of a parameter matrix x only)
    constexpr bool eta_partials_kept_for_the_reverse_pass = x_is_var && !x_is_row_vector;

    const int n = x_is_row_vector ? static_cast<int>(stan::math::size(y)) : static_cast<int>(x.rows());
    const int P = static_cast<int>(x.cols());
    const int K = static_cast<int>(cuts_in.size()) + 1;

    // Stan's checks, in Stan's order: the sizes, the categories, the cutpoints (ordered; first and last finite)
    stan::math::check_consistent_size(function_name, "Vector of dependent variables", y, n);
    stan::math::check_consistent_size(function_name, "Weight vector", beta_in, P);
    const auto& beta = stan::math::to_ref(beta_in);
    const auto& cuts = stan::math::to_ref(cuts_in);
    buffers_t& b = buffers();
    b.ensure(n, (x_is_var && !x_is_row_vector) ? n * P : P, K, P);
    double* cutpoint_values = b.cutpoint_values.data();
    for (int m = 0; m < K - 1; ++m) cutpoint_values[m] = stan::math::value_of(cuts.coeff(m));
    const auto check_cutpoints = [&]() {

        stan::math::check_ordered(function_name, "Cut-points", cuts);
        if (K > 1) {
            if (K > 2) stan::math::check_finite(function_name, "Final cut-point", cutpoint_values[K - 2]);
            stan::math::check_finite(function_name, "First cut-point", cutpoint_values[0]);
        }

    };
    if (n == 0 || K == 1 || !stan::math::include_summand<propto, T_x, T_beta, T_cuts>::value) {
        // (no responses, no cutpoints, or data x, beta and cuts with propto: only the checks, then 0, as Stan's)
        stan::math::check_bounded(function_name, "Vector of dependent variables", y, 1, K);
        check_cutpoints();
        return return_t(0.0);
    }
    const int* categories = fused_ordered_logistic::contiguous_categories(y, n);
    fused_ordered_logistic::category_tables_t& tables = fused_ordered_logistic::category_tables();
    fused_ordered_logistic::fill_category_tables(cutpoint_values, K, tables);

    // the values of beta (P values: the number of predictors, which does not grow with n), on the arena (the
    // reverse pass of a parameter x reads them)
    auto& arena = stan::math::ChainableStack::instance_->memalloc_;
    double* beta_values = arena.alloc_array<double>(P);
    for (int p = 0; p < P; ++p) beta_values[p] = stan::math::value_of(beta.coeff(p));

    // ---- (1) eta = x beta (and the values of a parameter x), in SIMD. x as a column-major matrix with unit
    // inner stride (bound without a copy to a plain matrix or a block of columns); a row vector x as it is
    using x_reference_t = std::conditional_t<x_is_row_vector, const std::decay_t<T_x>&,
                                             const Eigen::Ref<const Eigen::Matrix<scalar_of_x_t, Eigen::Dynamic,
                                                                                  Eigen::Dynamic>,
                                                              0, Eigen::OuterStride<>>>;
    x_reference_t x_reference(x);
    double* eta = b.eta.data();
    double* x_values = b.x_values.data();
    const double** value_columns = b.value_columns.data();
    vari* const** vari_columns = b.vari_columns.data();
    double eta_of_a_row_vector_x = 0.0;
    double eta_sum = 0.0;
    if constexpr (x_is_row_vector) {
        for (int p = 0; p < P; ++p) {
            x_values[p] = stan::math::value_of(x_reference.coeff(p));
            eta_of_a_row_vector_x += x_values[p] * beta_values[p];
        }
        eta_sum = eta_of_a_row_vector_x;
    } else {
        // (the columns of x: a data x read where it is; a parameter x gathered into x_values)
        for (int p = 0; p < P; ++p) {
            if constexpr (x_is_var) {
                vari_columns[p] = reinterpret_cast<vari* const*>(x_reference.data()
                                                                 + static_cast<long long>(p)
                                                                   * x_reference.outerStride());
                value_columns[p] = x_values + static_cast<long long>(p) * n;
            } else {
                value_columns[p] = x_reference.data() + static_cast<long long>(p) * x_reference.outerStride();
            }
        }
        long long offset_of_value = 0;
        if constexpr (x_is_var) offset_of_value = offset_of_value_in_vari(vari_columns[0][0]);
        simd_t eta_sum_in_lanes = fused_ordered_logistic::broadcast(0.0);
        for (int i = 0; i < n; i += lanes) {
            const int n_valid = std::min(lanes, n - i);
            simd_t eta_block = fused_ordered_logistic::broadcast(0.0);
            for (int p = 0; p < P; ++p) {
                simd_t x_block;
                if (n_valid == lanes) {
                    if constexpr (x_is_var) {
                        x_block = values_at_offset_of_varis(vari_columns[p] + i, offset_of_value);
                        simd_storeu(x_values + static_cast<long long>(p) * n + i, x_block);
                    } else {
                        x_block = simd_loadu(value_columns[p] + i);
                    }
                } else {
                    // (the last block: padding lanes 0, so that they add nothing to eta's sum)
                    alignas(64) double x_padded[lanes];
                    for (int j = 0; j < lanes; ++j) {
                        if (j >= n_valid) {
                            x_padded[j] = 0.0;
                            continue;
                        }
                        if constexpr (x_is_var) {
                            x_padded[j] = vari_columns[p][i + j]->val_;
                            x_values[static_cast<long long>(p) * n + i + j] = x_padded[j];
                        } else {
                            x_padded[j] = value_columns[p][i + j];
                        }
                    }
                    x_block = simd_loadu(x_padded);
                }
                const simd_t beta_p_in_lanes = fused_ordered_logistic::broadcast(beta_values[p]);
                eta_block = fused_ordered_logistic::multiply_add(x_block, beta_p_in_lanes, eta_block);
            }
            if (n_valid == lanes) {
                simd_storeu(eta + i, eta_block);
            } else {
                alignas(64) double eta_padded[lanes];
                simd_store(eta_padded, eta_block);
                for (int j = 0; j < n_valid; ++j) eta[i + j] = eta_padded[j];
            }
            eta_sum_in_lanes = fused_ordered_logistic::add(eta_sum_in_lanes, eta_block);
        }
        eta_sum = fused_ordered_logistic::horizontal_sum(eta_sum_in_lanes);
    }

    // ---- (2) u = eta - c_{y-1}, v = eta - c_y, the cutpoints gathered by category, in SIMD (with the range test
    // of the categories)
    double* u = b.u.data();
    double* v = b.v.data();
    const double* lower_cutpoint = tables.lower_cutpoint.data();
    const double* upper_cutpoint = tables.upper_cutpoint.data();
    unsigned any_category_out_of_range = 0;
    for (int i = 0; i < n; i += lanes) {
        const int n_valid = std::min(lanes, n - i);
        simd_category_t k_block;
        simd_t eta_block;
        if (n_valid == lanes) {
            k_block = load_categories(categories + i);
            if constexpr (x_is_row_vector) eta_block = fused_ordered_logistic::broadcast(eta_of_a_row_vector_x);
            else eta_block = simd_loadu(eta + i);
        } else {
            // (the last block: padding lanes of category 1, never stored)
            alignas(64) int k_padded[lanes];
            alignas(64) double eta_padded[lanes];
            for (int j = 0; j < lanes; ++j) {
                const bool valid = (j < n_valid);
                k_padded[j] = valid ? categories[i + j] : 1;
                if constexpr (x_is_row_vector) eta_padded[j] = eta_of_a_row_vector_x;
                else eta_padded[j] = valid ? eta[i + j] : 0.0;
            }
            k_block = load_categories(k_padded);
            eta_block = simd_loadu(eta_padded);
        }
        k_block = categories_in_range_or_0(k_block, K, any_category_out_of_range);
        const simd_t u_block = fused_ordered_logistic::subtract(eta_block,
                                                                table_at_categories(lower_cutpoint, k_block));
        const simd_t v_block = fused_ordered_logistic::subtract(eta_block,
                                                                table_at_categories(upper_cutpoint, k_block));
        if (n_valid == lanes) {
            simd_storeu(u + i, u_block);
            simd_storeu(v + i, v_block);
        } else {
            alignas(64) double u_padded[lanes], v_padded[lanes];
            simd_store(u_padded, u_block);
            simd_store(v_padded, v_block);
            for (int j = 0; j < n_valid; ++j) {
                u[i + j] = u_padded[j];
                v[i + j] = v_padded[j];
            }
        }
    }
    // Stan's checks, in Stan's order, with Stan's messages (the categories only when the pass found a problem;
    // x and beta only when eta is not finite, as Stan's)
    if (any_category_out_of_range) {
        stan::math::check_bounded(function_name, "Vector of dependent variables", y, 1, K);
    }
    check_cutpoints();
    if (!std::isfinite(eta_sum)) {
        stan::math::check_finite(function_name, "Weight vector", beta);
        stan::math::check_finite(function_name, "Matrix of independent variables", x);
    }

    // ---- (3) the fused pass of nico_ordered_logistic_lpmf: the value, g and the cutpoint sums
    double* eta_partials = nullptr;
    if constexpr (eta_partials_kept_for_the_reverse_pass) {
        eta_partials = arena.alloc_array<double>(n);
    } else {
        eta_partials = b.eta_partials.data();
    }
    double sum_without_L;
    if constexpr (eta_partials_needed) {
        sum_without_L = fused_ordered_logistic::fused_pass<true, cuts_is_var>(u, v, categories, n, K,
                                                                              eta_partials, tables);
    } else {
        sum_without_L = fused_ordered_logistic::fused_pass<false, cuts_is_var>(u, v, categories, n, K,
                                                                               nullptr, tables);
    }
    fused_ordered_logistic::fill_cutpoint_gap_constants_of_categories_with_responses(cutpoint_values, K, tables);
    double log_probability = sum_without_L;
    for (int m = 2; m <= K - 1; ++m) {
        log_probability += static_cast<double>(tables.n_responses[m]) * tables.log1m_exp_of_cutpoint_gap[m];
    }
    if constexpr (!x_is_var && !beta_is_var && !cuts_is_var) {
        return log_probability;
    } else {
        // ---- (4) the gradient: beta (x^T g), a row-vector x (beta sum(g)), the cutpoints (as
        // nico_ordered_logistic_lpmf's); a parameter matrix x in the reverse pass (g beta^T)
        double sum_of_eta_partials = 0.0;
        if constexpr (eta_partials_needed && x_is_row_vector) sum_of_eta_partials = sum_of(eta_partials, n);
        vari** beta_varis = nullptr;
        double* beta_partials = nullptr;
        if constexpr (beta_is_var) {
            beta_varis = arena.alloc_array<vari*>(P);
            beta_partials = arena.alloc_array<double>(P);
            for (int p = 0; p < P; ++p) {
                beta_varis[p] = beta.coeff(p).vi_;
                if constexpr (x_is_row_vector) beta_partials[p] = x_values[p] * sum_of_eta_partials;
                else beta_partials[p] = dot_product(value_columns[p], eta_partials, n);
            }
        }
        vari** x_varis = nullptr;
        double* row_vector_x_partials = nullptr;
        if constexpr (x_is_var) {
            if constexpr (x_is_row_vector) {
                x_varis = arena.alloc_array<vari*>(P);
                row_vector_x_partials = arena.alloc_array<double>(P);
                for (int p = 0; p < P; ++p) {
                    x_varis[p] = x_reference.coeff(p).vi_;
                    row_vector_x_partials[p] = beta_values[p] * sum_of_eta_partials;
                }
            } else {
                x_varis = arena.alloc_array<vari*>(static_cast<size_t>(n) * P);
                for (int p = 0; p < P; ++p) {
                    std::memcpy(x_varis + static_cast<long long>(p) * n, vari_columns[p], sizeof(vari*) * n);
                }
            }
        }
        vari** cutpoint_varis = nullptr;
        double* cutpoint_partials = nullptr;
        const int n_cutpoints = K - 1;
        if constexpr (cuts_is_var) {
            cutpoint_varis = arena.alloc_array<vari*>(n_cutpoints);
            cutpoint_partials = arena.alloc_array<double>(n_cutpoints);
            const double* A = tables.inverse_expm1_of_cutpoint_gap.data();
            const long long* n_responses = tables.n_responses.data();
            for (int m = 1; m <= n_cutpoints; ++m) {
                alignas(64) double lane_sums[lanes];
                simd_store(lane_sums, simd_loadu(tables.cutpoint_sums.data() + (m - 1) * lanes));
                double partial = 0.0;
                for (int j = 0; j < lanes; ++j) partial += lane_sums[j];
                partial += static_cast<double>(n_responses[m]) * A[m];
                if (m + 1 <= K - 1) partial -= static_cast<double>(n_responses[m + 1]) * A[m + 1];
                cutpoint_partials[m - 1] = partial;
                cutpoint_varis[m - 1] = cuts.coeff(m - 1).vi_;
            }
        }
        return stan::math::make_callback_var(
            log_probability,
            [x_varis, eta_partials, beta_values, row_vector_x_partials, beta_varis, beta_partials, cutpoint_varis,
             cutpoint_partials, n, P, n_cutpoints](auto& result_vari) {

                const double adjoint = result_vari.adj_;
                if constexpr (x_is_var) {
                    if constexpr (x_is_row_vector) {
                        for (int p = 0; p < P; ++p) x_varis[p]->adj_ += adjoint * row_vector_x_partials[p];
                    } else {
                        for (int p = 0; p < P; ++p) {
                            add_to_adjoints_of_varis(x_varis + static_cast<long long>(p) * n, eta_partials,
                                                     adjoint * beta_values[p], n);
                        }
                    }
                }
                if constexpr (beta_is_var) {
                    for (int p = 0; p < P; ++p) beta_varis[p]->adj_ += adjoint * beta_partials[p];
                }
                if constexpr (cuts_is_var) {
                    for (int m = 0; m < n_cutpoints; ++m) {
                        cutpoint_varis[m]->adj_ += adjoint * cutpoint_partials[m];
                    }
                }

            });
    }

}

}  // namespace fused_ordered_logistic_glm
}  // namespace nico::math

// ---- the Stan-facing function (global scope; stanc calls nico_ordered_logistic_glm_lpmf<propto__>(y, x, beta,
// cuts, pstream__)): y an int or an array of ints, x a matrix or a row vector, beta and cuts vectors
namespace nico::math {

template <bool propto, typename T_y, typename T_x, typename T_beta, typename T_cuts,
          stan::require_t<std::is_integral<stan::scalar_type_t<T_y>>>* = nullptr,
          stan::require_eigen_t<T_x>* = nullptr,
          stan::require_all_eigen_col_vector_t<T_beta, T_cuts>* = nullptr>
inline stan::return_type_t<T_x, T_beta, T_cuts> nico_ordered_logistic_glm_lpmf(const T_y& y, const T_x& x,
                                                                              const T_beta& beta,
                                                                              const T_cuts& cuts,
                                                                              std::ostream* /*pstream__*/) {

    return nico::math::fused_ordered_logistic_glm::ordered_logistic_glm<propto>(y, x, beta, cuts);

}

}  // namespace nico::math

using nico::math::nico_ordered_logistic_glm_lpmf;























