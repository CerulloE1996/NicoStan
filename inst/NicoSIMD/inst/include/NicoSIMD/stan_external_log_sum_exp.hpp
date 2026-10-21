#pragma once
#if 0
//
// -| --------- BayesMVP AVX-512 / AVX2 log_sum_exp, exposed to Stan as external functions ------------------------
//
// Drop-in replacements for Stan's log_sum_exp, computed with the SIMD exp and log kernels of
// stan_external_functions.hpp (nico_exp_1 and nico_log_1: 8 lanes with the AVX-512 kernels, 4 lanes with the
// AVX2 kernels, including -DBAYESMVP_FORCE_AVX2), used by NicoStan's AVX2 and AVX-512 versions of a Stan model
// (the plain Stan math version of the model keeps Stan's log_sum_exp):
//
//   (1) nico_log_sum_exp(x): the reduction over a vector, row_vector, matrix or array[] real,
//         log_sum_exp(x) = max(x) + log(sum(exp(x - max(x)))),   d log_sum_exp(x) / dx_i = exp(x_i - log_sum_exp(x)),
//       as ONE autodiff node (a callback vari), whatever the number of elements;
//   (2) nico_log_sum_exp(a, b): elementwise over two arguments (real or vector, at least one vector),
//         log_sum_exp(a, b) = max(a, b) + log1p_exp(min(a, b) - max(a, b)),
//         d / da = inv_logit(a - b),   d / db = inv_logit(b - a),
//       as one block of result nodes and ONE reverse callback; with two reals, Stan's own scalar log_sum_exp;
//   (3) nico_log_sum_exp_2d(x): the log_sum_exp of each row of a matrix over its columns (the row-wise
//       reduction which the latent class models compute with their own log_sum_exp_2d(), e.g. the 2-class
//       LC-MVP and LC-MVOP), as a vector: one block of result nodes and ONE reverse callback.
//
// Arena per element: 8 bytes (the operand's node pointer) for (1) and (3), whose reverse pass recomputes
// exp(x_i - result) with the SIMD kernel; 8 bytes (the weight inv_logit(a - b)) + 8 bytes per var vector argument
// (its node pointer) + 24 bytes (the result node) for (2). Edge cases as Stan: an empty reduction is -inf; a
// largest value of -inf gives -inf, and of +inf gives +inf.
//
// Declared in the Stan functions block (no body) as
//     real nico_log_sum_exp(vector x);   real nico_log_sum_exp(row_vector x);   real nico_log_sum_exp(matrix x);
//     real nico_log_sum_exp(array[] real x);   real nico_log_sum_exp(real a, real b);
//     vector nico_log_sum_exp(vector a, vector b);   vector nico_log_sum_exp(real a, vector b);
//     vector nico_log_sum_exp(vector a, real b);   vector nico_log_sum_exp_2d(matrix x);
// (NicoStan's R_fn_write_BayesMVP_AVX_version_of_Stan_model_and_user_header() writes them, and replaces the
//  model's calls of log_sum_exp and log_sum_exp_2d, in the AVX2 and AVX-512 versions of a model).
//
// -----------------------------------------------------------------------------------------------------------------

#include "stan_external_functions.hpp"
#include "stan_external_simd_log_cumulative_sum_exp_scans.hpp"

namespace nico::math {
namespace log_sum_exp_simd {

using stan::math::var;
using stan::math::vari;
using log_cumulative_sum_exp::nico_load_prefix_packet;
using log_cumulative_sum_exp::nico_store_prefix_packet;
using log_cumulative_sum_exp::nico_exp_of_nonpositive_prefix_argument;
using log_cumulative_sum_exp::nico_add_prefix_adjoint_packet;

#if BMVP_LANES == 8
using nico_lse_mask_t = __mmask8;
inline simd_t nico_lse_add(const simd_t a, const simd_t b) { return _mm512_add_pd(a, b); }
inline simd_t nico_lse_subtract(const simd_t a, const simd_t b) { return _mm512_sub_pd(a, b); }
inline simd_t nico_lse_divide(const simd_t a, const simd_t b) { return _mm512_div_pd(a, b); }
inline nico_lse_mask_t nico_lse_greater(const simd_t a, const simd_t b) {

    return _mm512_cmp_pd_mask(a, b, _CMP_GT_OQ);

}
inline nico_lse_mask_t nico_lse_equal(const simd_t a, const simd_t b) {

    return _mm512_cmp_pd_mask(a, b, _CMP_EQ_OQ);

}
inline nico_lse_mask_t nico_lse_nan(const simd_t a) { return _mm512_cmp_pd_mask(a, a, _CMP_UNORD_Q); }
inline nico_lse_mask_t nico_lse_finite(const simd_t a) {

    return _mm512_cmp_pd_mask(_mm512_abs_pd(a), scan_set1(stan::math::INFTY), _CMP_LT_OQ);

}
inline simd_t nico_lse_select(const nico_lse_mask_t mask, const simd_t yes, const simd_t no) {

    return _mm512_mask_blend_pd(mask, no, yes);

}
inline bool nico_lse_any(const nico_lse_mask_t mask) { return mask != 0; }
inline nico_lse_mask_t nico_lse_or(const nico_lse_mask_t a, const nico_lse_mask_t b) { return a | b; }
inline double nico_lse_sum(const simd_t a) { return _mm512_reduce_add_pd(a); }
inline double nico_lse_maximum(const simd_t a) { return _mm512_reduce_max_pd(a); }
#else
using nico_lse_mask_t = simd_t;
inline simd_t nico_lse_add(const simd_t a, const simd_t b) { return _mm256_add_pd(a, b); }
inline simd_t nico_lse_subtract(const simd_t a, const simd_t b) { return _mm256_sub_pd(a, b); }
inline simd_t nico_lse_divide(const simd_t a, const simd_t b) { return _mm256_div_pd(a, b); }
inline nico_lse_mask_t nico_lse_greater(const simd_t a, const simd_t b) {

    return _mm256_cmp_pd(a, b, _CMP_GT_OQ);

}
inline nico_lse_mask_t nico_lse_equal(const simd_t a, const simd_t b) {

    return _mm256_cmp_pd(a, b, _CMP_EQ_OQ);

}
inline nico_lse_mask_t nico_lse_nan(const simd_t a) { return _mm256_cmp_pd(a, a, _CMP_UNORD_Q); }
inline nico_lse_mask_t nico_lse_finite(const simd_t a) {

    return _mm256_cmp_pd(_mm256_abs_pd(a), scan_set1(stan::math::INFTY), _CMP_LT_OQ);

}
inline simd_t nico_lse_select(const nico_lse_mask_t mask, const simd_t yes, const simd_t no) {

    return _mm256_blendv_pd(no, yes, mask);

}
inline bool nico_lse_any(const nico_lse_mask_t mask) { return _mm256_movemask_pd(mask) != 0; }
inline nico_lse_mask_t nico_lse_or(const nico_lse_mask_t a, const nico_lse_mask_t b) {

    return _mm256_or_pd(a, b);

}
inline double nico_lse_sum(const simd_t a) {

    const __m128d pair = _mm_add_pd(_mm256_castpd256_pd128(a), _mm256_extractf128_pd(a, 1));
    return _mm_cvtsd_f64(_mm_add_sd(pair, _mm_unpackhi_pd(pair, pair)));

}
inline double nico_lse_maximum(const simd_t a) {

    const __m128d pair = _mm_max_pd(_mm256_castpd256_pd128(a), _mm256_extractf128_pd(a, 1));
    return _mm_cvtsd_f64(_mm_max_sd(pair, _mm_unpackhi_pd(pair, pair)));

}
#endif

inline simd_t nico_lse_maximum_packet(const simd_t a, const simd_t b) {

    const simd_t ordered = nico_lse_select(nico_lse_greater(a, b), a, b);
    return nico_lse_select(nico_lse_nan(a), a, nico_lse_select(nico_lse_nan(b), b, ordered));

}

//// NaNs are collected separately by the reduction; its maximum need not propagate them at every step.
inline simd_t nico_lse_maximum_before_nan_check(const simd_t a, const simd_t b) {

#if BMVP_LANES == 8
    return _mm512_max_pd(a, b);
#else
    return _mm256_max_pd(a, b);
#endif

}

inline simd_t nico_lse_log_positive_normal(const simd_t a) {

    return BMVP_KERNEL_NAME(nico_log_1_double_precision_wo_checks)(a);

}

inline simd_t nico_lse_log1p_nonnegative(const simd_t a) {

    return BMVP_KERNEL_NAME(nico_log1p_1_double_precision_wo_checks)(a);

}

inline double nico_lse_first(const simd_t a) {

    alignas(64) double values[BMVP_LANES];
    simd_store(values, a);
    return values[0];

}

// ---- the reduction of n doubles; on return, buffer (n doubles) holds exp(x_i - max) when the result is finite
//// A null buffer requests only the value; no exponential vector is then allocated or written.
inline double reduction_of_values(const double* x, const int n, double* buffer) {

    if (n == 0) return stan::math::NEGATIVE_INFTY;
    //// Full packets use unmasked loads. Four independent maxima avoid a serial dependency across packets;
    //// NaN flags remain in SIMD registers until the maximum pass finishes. Only the final tail is masked.
    const int full_packet_end = n - n % BMVP_LANES;
    const int four_packet_end = n - n % (4 * BMVP_LANES);
    simd_t maximum_0 = scan_set1(stan::math::NEGATIVE_INFTY);
    simd_t maximum_1 = maximum_0;
    simd_t maximum_2 = maximum_0;
    simd_t maximum_3 = maximum_0;
    nico_lse_mask_t nan_flags = nico_lse_nan(scan_zero());
    int first = 0;
    for (; first < four_packet_end; first += 4 * BMVP_LANES) {
        const simd_t values_0 = simd_loadu(x + first);
        const simd_t values_1 = simd_loadu(x + first + BMVP_LANES);
        const simd_t values_2 = simd_loadu(x + first + 2 * BMVP_LANES);
        const simd_t values_3 = simd_loadu(x + first + 3 * BMVP_LANES);
        maximum_0 = nico_lse_maximum_before_nan_check(maximum_0, values_0);
        maximum_1 = nico_lse_maximum_before_nan_check(maximum_1, values_1);
        maximum_2 = nico_lse_maximum_before_nan_check(maximum_2, values_2);
        maximum_3 = nico_lse_maximum_before_nan_check(maximum_3, values_3);
        const nico_lse_mask_t first_flags = nico_lse_or(nico_lse_nan(values_0), nico_lse_nan(values_1));
        const nico_lse_mask_t last_flags = nico_lse_or(nico_lse_nan(values_2), nico_lse_nan(values_3));
        nan_flags = nico_lse_or(nan_flags, nico_lse_or(first_flags, last_flags));
    }
    simd_t maxima = nico_lse_maximum_before_nan_check(
        nico_lse_maximum_before_nan_check(maximum_0, maximum_1),
        nico_lse_maximum_before_nan_check(maximum_2, maximum_3));
    for (; first < full_packet_end; first += BMVP_LANES) {
        const simd_t values = simd_loadu(x + first);
        nan_flags = nico_lse_or(nan_flags, nico_lse_nan(values));
        maxima = nico_lse_maximum_before_nan_check(maxima, values);
    }
    if (first < n) {
        const simd_t values = nico_load_prefix_packet(x + first, n - first, stan::math::NEGATIVE_INFTY);
        nan_flags = nico_lse_or(nan_flags, nico_lse_nan(values));
        maxima = nico_lse_maximum_before_nan_check(maxima, values);
    }
    if (nico_lse_any(nan_flags)) return std::numeric_limits<double>::quiet_NaN();
    const double largest = nico_lse_maximum(maxima);
    if (!std::isfinite(largest)) return largest;

    //// The same accurate exponential is evaluated on four independent packets. Buffer writes retain the
    //// derivative caller's contract; value-only callers still allocate and write no exponential buffer.
    const simd_t largest_packet = scan_set1(largest);
    simd_t sum_0 = scan_zero();
    simd_t sum_1 = scan_zero();
    simd_t sum_2 = scan_zero();
    simd_t sum_3 = scan_zero();
    first = 0;
    for (; first < four_packet_end; first += 4 * BMVP_LANES) {
        const simd_t exponential_0 = nico_exp_of_nonpositive_prefix_argument(
            nico_lse_subtract(simd_loadu(x + first), largest_packet));
        const simd_t exponential_1 = nico_exp_of_nonpositive_prefix_argument(
            nico_lse_subtract(simd_loadu(x + first + BMVP_LANES), largest_packet));
        const simd_t exponential_2 = nico_exp_of_nonpositive_prefix_argument(
            nico_lse_subtract(simd_loadu(x + first + 2 * BMVP_LANES), largest_packet));
        const simd_t exponential_3 = nico_exp_of_nonpositive_prefix_argument(
            nico_lse_subtract(simd_loadu(x + first + 3 * BMVP_LANES), largest_packet));
        if (buffer != nullptr) {
            simd_storeu(buffer + first, exponential_0);
            simd_storeu(buffer + first + BMVP_LANES, exponential_1);
            simd_storeu(buffer + first + 2 * BMVP_LANES, exponential_2);
            simd_storeu(buffer + first + 3 * BMVP_LANES, exponential_3);
        }
        sum_0 = nico_lse_add(sum_0, exponential_0);
        sum_1 = nico_lse_add(sum_1, exponential_1);
        sum_2 = nico_lse_add(sum_2, exponential_2);
        sum_3 = nico_lse_add(sum_3, exponential_3);
    }
    simd_t sums = nico_lse_add(nico_lse_add(sum_0, sum_1), nico_lse_add(sum_2, sum_3));
    for (; first < full_packet_end; first += BMVP_LANES) {
        const simd_t shifted = nico_lse_subtract(simd_loadu(x + first), largest_packet);
        const simd_t exponentials = nico_exp_of_nonpositive_prefix_argument(shifted);
        if (buffer != nullptr) simd_storeu(buffer + first, exponentials);
        sums = nico_lse_add(sums, exponentials);
    }
    if (first < n) {
        const int lanes = n - first;
        const simd_t values = nico_load_prefix_packet(x + first, lanes, stan::math::NEGATIVE_INFTY);
        const simd_t shifted = nico_lse_subtract(values, largest_packet);
        const simd_t exponentials = nico_exp_of_nonpositive_prefix_argument(shifted);
        if (buffer != nullptr) nico_store_prefix_packet(buffer + first, lanes, exponentials);
        sums = nico_lse_add(sums, exponentials);
    }
    return nico_lse_first(nico_lse_add(scan_set1(largest),
                                      nico_lse_log_positive_normal(scan_set1(nico_lse_sum(sums)))));

}

// ---- the weights exp(x_i - result) of n doubles, in place on buffer (which holds x on entry)
inline void weights_of_values(double* buffer, const int n, const double result) {

    for (int i = 0; i < n; i += BMVP_LANES) {
        const int lanes = std::min(BMVP_LANES, n - i);
        const simd_t values = nico_load_prefix_packet(buffer + i, lanes, stan::math::NEGATIVE_INFTY);
        const simd_t shifted = nico_lse_subtract(values, scan_set1(result));
        nico_store_prefix_packet(buffer + i, lanes, nico_exp_of_nonpositive_prefix_argument(shifted));
    }

}

// ---- (1) the reduction, for any Eigen object or std::vector of double or var: the argument is evaluated once into
//      a plain object (values, or node pointers), whose contiguous storage is then read
template <typename T, stan::require_eigen_t<T>* = nullptr, stan::require_st_arithmetic<T>* = nullptr>
inline double reduction(const T& x_in) {

    //// Plain double Eigen objects already have contiguous storage. Expressions and strided views retain
    //// their original evaluation into a plain object; reducing their data pointer directly is not safe.
    if constexpr (std::is_same<stan::scalar_type_t<T>, double>::value
                  && std::is_same<std::decay_t<T>, typename std::decay_t<T>::PlainObject>::value) {

        return reduction_of_values(x_in.data(), static_cast<int>(x_in.size()), nullptr);

    } else {
        const Eigen::Matrix<double, -1, -1> x = x_in;
        return reduction_of_values(x.data(), static_cast<int>(x.size()), nullptr);
    }

}

template <typename T, stan::require_std_vector_vt<std::is_arithmetic, T>* = nullptr>
inline double reduction(const T& x) {

    const int n = static_cast<int>(x.size());
    if constexpr (std::is_same<typename T::value_type, double>::value) {
        return reduction_of_values(x.data(), n, nullptr);
    }
    scratch_t& buffers = scratch();
    buffers.ensure(n);
    double* values = buffers.a.data();
    for (int i = 0; i < n; ++i) values[i] = static_cast<double>(x[i]);
    return reduction_of_values(values, n, nullptr);

}

inline var reduction_of_nodes(vari** nodes, const int n) {

    scratch_t& buffers = scratch();
    buffers.ensure(n);
    double* values = buffers.a.data();
    for (int i = 0; i < n; ++i) values[i] = nodes[i]->val_;
    const double result = reduction_of_values(values, n, buffers.b.data());
    return stan::math::make_callback_var(result, [nodes, n](auto& result_node) {

        scratch_t& reverse_buffers = scratch();
        reverse_buffers.ensure(n);
        double* weights = reverse_buffers.a.data();
        for (int i = 0; i < n; ++i) weights[i] = nodes[i]->val_;
        weights_of_values(weights, n, result_node.val());
        const simd_t adjoint = scan_set1(result_node.adj());
        for (int i = 0; i < n; i += BMVP_LANES) {
            const int lanes = std::min(BMVP_LANES, n - i);
            double* targets[BMVP_LANES];
            for (int lane = 0; lane < lanes; ++lane) targets[lane] = &nodes[i + lane]->adj_;
            nico_add_prefix_adjoint_packet(targets, lanes,
                                           nico_load_prefix_packet(weights + i, lanes, 0.0), adjoint);
        }

    });

}

template <typename T, stan::require_eigen_t<T>* = nullptr, stan::require_st_var<T>* = nullptr>
inline var reduction(const T& x_in) {

    const Eigen::Matrix<var, -1, -1> x = x_in;
    const int n = static_cast<int>(x.size());
    auto& arena = stan::math::ChainableStack::instance_->memalloc_;
    vari** nodes = arena.alloc_array<vari*>(n);
    for (int i = 0; i < n; ++i) nodes[i] = x.data()[i].vi_;
    return reduction_of_nodes(nodes, n);

}

template <typename T, stan::require_std_vector_vt<stan::is_var, T>* = nullptr>
inline var reduction(const T& x) {

    const int n = static_cast<int>(x.size());
    auto& arena = stan::math::ChainableStack::instance_->memalloc_;
    vari** nodes = arena.alloc_array<vari*>(n);
    for (int i = 0; i < n; ++i) nodes[i] = x[i].vi_;
    return reduction_of_nodes(nodes, n);

}

// ---- (2) elementwise over two arguments: values and the weights inv_logit(a - b) of n pairs
//// simd_t is one register of four AVX2 doubles or eight AVX-512 doubles. Null weight pointers select
//// value-only arithmetic; the same exponential and non-finite value rules serve data and AD callers.
inline simd_t nico_lse_pair_packet_value(const simd_t av, const simd_t bv,
                                        simd_t* weight_of_a, simd_t* weight_of_b) {

    // log1p_exp's argument, <= 0
    // inv_logit's argument
    const nico_lse_mask_t a_larger = nico_lse_greater(av, bv);
    const simd_t largest = nico_lse_select(a_larger, av, bv);
    const simd_t smallest = nico_lse_select(a_larger, bv, av);
    const simd_t difference = nico_lse_subtract(smallest, largest);
    const simd_t shifted = nico_lse_select(nico_lse_nan(difference),
                                           scan_set1(stan::math::NEGATIVE_INFTY), difference);
    const simd_t exponential = nico_exp_of_nonpositive_prefix_argument(shifted);
    simd_t value = nico_lse_select(nico_lse_finite(largest),
                                  nico_lse_add(largest, nico_lse_log1p_nonnegative(exponential)), largest);
    const nico_lse_mask_t any_nan = nico_lse_or(nico_lse_nan(av), nico_lse_nan(bv));
    value = nico_lse_select(any_nan, nico_lse_add(av, bv), value);
    if (weight_of_a != nullptr || weight_of_b != nullptr) {
        const simd_t denominator = nico_lse_add(scan_set1(1.0), exponential);
        const simd_t nan = scan_set1(std::numeric_limits<double>::quiet_NaN());
        if (weight_of_a != nullptr) {
            simd_t weight = nico_lse_divide(nico_lse_select(a_larger, scan_set1(1.0), exponential),
                                            denominator);
            weight = nico_lse_select(nico_lse_nan(difference), nan, weight);
            *weight_of_a = nico_lse_select(any_nan, nico_lse_add(av, bv), weight);
        }
        if (weight_of_b != nullptr) {
            simd_t weight = nico_lse_divide(nico_lse_select(a_larger, exponential, scan_set1(1.0)),
                                            denominator);
            weight = nico_lse_select(nico_lse_nan(difference), nan, weight);
            *weight_of_b = nico_lse_select(any_nan, nico_lse_add(av, bv), weight);
        }
    }
    return value;

}

inline void nico_lse_pairs_of_values(const double* a, const double* b, const int n, double* result,
                                    double* weight_of_a, double* weight_of_b) {

    for (int i = 0; i < n; i += BMVP_LANES) {
        const int lanes = std::min(BMVP_LANES, n - i);
        simd_t weight_a;
        simd_t weight_b;
        const simd_t value = nico_lse_pair_packet_value(
                    nico_load_prefix_packet(a + i, lanes, 0.0), nico_load_prefix_packet(b + i, lanes, 0.0),
                    weight_of_a == nullptr ? nullptr : &weight_a, weight_of_b == nullptr ? nullptr : &weight_b);
        nico_store_prefix_packet(result + i, lanes, value);
        if (weight_of_a != nullptr) nico_store_prefix_packet(weight_of_a + i, lanes, weight_a);
        if (weight_of_b != nullptr) nico_store_prefix_packet(weight_of_b + i, lanes, weight_b);
    }

}

inline void pairs_of_values(const double* a, const double* b, const int n, double* result, double* weight_of_a) {

    nico_lse_pairs_of_values(a, b, n, result, weight_of_a, nullptr);

}

template <typename T>
inline int length_of_argument(const T& x) {

    if constexpr (stan::is_stan_scalar<T>::value) {
        return 1;
    } else {
        return static_cast<int>(x.size());
    }

}

template <typename T>
inline double value_of_element(const T& x, const int i) {

    if constexpr (stan::is_stan_scalar<T>::value) {
        return stan::math::value_of(x);
    } else {
        return stan::math::value_of(x.coeff(i));
    }

}

template <typename T1, typename T2>
inline auto pairs(const T1& a_in, const T2& b_in) {

    using plain_of_a = std::conditional_t<stan::is_stan_scalar<T1>::value, T1,
                                          Eigen::Matrix<stan::scalar_type_t<T1>, -1, 1>>;
    using plain_of_b = std::conditional_t<stan::is_stan_scalar<T2>::value, T2,
                                          Eigen::Matrix<stan::scalar_type_t<T2>, -1, 1>>;
    const plain_of_a a = a_in;
    const plain_of_b b = b_in;
    const int n_a = length_of_argument(a);
    const int n_b = length_of_argument(b);
    const int n = stan::is_stan_scalar<T1>::value ? n_b : n_a;
    if (!stan::is_stan_scalar<T1>::value && !stan::is_stan_scalar<T2>::value && n_a != n_b) {
        throw std::invalid_argument("nico_log_sum_exp: the two vectors must have the same length");
    }
    if constexpr (!stan::is_var<stan::scalar_type_t<T1>>::value
                  && !stan::is_var<stan::scalar_type_t<T2>>::value) {

        Eigen::Matrix<double, -1, 1> out(n);
        for (int i = 0; i < n; i += BMVP_LANES) {
            const int lanes = std::min(BMVP_LANES, n - i);
            simd_t av;
            simd_t bv;
            if constexpr (stan::is_stan_scalar<T1>::value) {
                av = scan_set1(value_of_element(a, 0));
            } else if constexpr (std::is_same<stan::scalar_type_t<T1>, double>::value) {
                av = nico_load_prefix_packet(a.data() + i, lanes, 0.0);
            } else {
                alignas(64) double values[BMVP_LANES] = {};
                for (int lane = 0; lane < lanes; ++lane) values[lane] = value_of_element(a, i + lane);
                av = simd_load(values);
            }
            if constexpr (stan::is_stan_scalar<T2>::value) {
                bv = scan_set1(value_of_element(b, 0));
            } else if constexpr (std::is_same<stan::scalar_type_t<T2>, double>::value) {
                bv = nico_load_prefix_packet(b.data() + i, lanes, 0.0);
            } else {
                alignas(64) double values[BMVP_LANES] = {};
                for (int lane = 0; lane < lanes; ++lane) values[lane] = value_of_element(b, i + lane);
                bv = simd_load(values);
            }
            nico_store_prefix_packet(out.data() + i, lanes, nico_lse_pair_packet_value(av, bv, nullptr, nullptr));
        }
        return out;

    } else {
        std::vector<double> values_of_a(n), values_of_b(n), results(n), weights(n), other_weights(n);
        for (int i = 0; i < n; ++i) {
            values_of_a[i] = value_of_element(a, stan::is_stan_scalar<T1>::value ? 0 : i);
            values_of_b[i] = value_of_element(b, stan::is_stan_scalar<T2>::value ? 0 : i);
        }
        nico_lse_pairs_of_values(values_of_a.data(), values_of_b.data(), n, results.data(), weights.data(),
                                 other_weights.data());
        auto& arena = stan::math::ChainableStack::instance_->memalloc_;
        double* weight_of_a = arena.alloc_array<double>(n);
        double* weight_of_b = arena.alloc_array<double>(n);
        for (int i = 0; i < n; i += BMVP_LANES) {
            const int lanes = std::min(BMVP_LANES, n - i);
            nico_store_prefix_packet(weight_of_a + i, lanes,
                                      nico_load_prefix_packet(weights.data() + i, lanes, 0.0));
            nico_store_prefix_packet(weight_of_b + i, lanes,
                                      nico_load_prefix_packet(other_weights.data() + i, lanes, 0.0));
        }
        vari* result_nodes = arena.alloc_array<vari>(n);
        VecVar out(n);
        for (int i = 0; i < n; ++i) {
            ::new (static_cast<void*>(result_nodes + i)) vari(results[i], false);   // nochain stack
            out.coeffRef(i) = var(result_nodes + i);
        }
        constexpr bool a_is_var = stan::is_var<stan::scalar_type_t<T1>>::value;
        constexpr bool b_is_var = stan::is_var<stan::scalar_type_t<T2>>::value;
        constexpr bool a_is_scalar = stan::is_stan_scalar<T1>::value;
        constexpr bool b_is_scalar = stan::is_stan_scalar<T2>::value;
        vari** nodes_of_a = nullptr;
        vari** nodes_of_b = nullptr;
        if constexpr (a_is_var) {
            nodes_of_a = arena.alloc_array<vari*>(a_is_scalar ? 1 : n);
            for (int i = 0; i < (a_is_scalar ? 1 : n); ++i) {
                if constexpr (a_is_scalar) nodes_of_a[i] = a.vi_; else nodes_of_a[i] = a.coeff(i).vi_;
            }
        }
        if constexpr (b_is_var) {
            nodes_of_b = arena.alloc_array<vari*>(b_is_scalar ? 1 : n);
            for (int i = 0; i < (b_is_scalar ? 1 : n); ++i) {
                if constexpr (b_is_scalar) nodes_of_b[i] = b.vi_; else nodes_of_b[i] = b.coeff(i).vi_;
            }
        }
        stan::math::reverse_pass_callback([result_nodes, weight_of_a, weight_of_b, nodes_of_a, nodes_of_b, n]() {

            simd_t adjoint_of_scalar_a = scan_zero();
            simd_t adjoint_of_scalar_b = scan_zero();
            for (int i = 0; i < n; i += BMVP_LANES) {
                const int lanes = std::min(BMVP_LANES, n - i);
                alignas(64) double adjoints[BMVP_LANES] = {};
                alignas(64) double values[BMVP_LANES] = {};
                for (int lane = 0; lane < lanes; ++lane) {
                    adjoints[lane] = result_nodes[i + lane].adj_;
                    values[lane] = result_nodes[i + lane].val_;
                }
                const simd_t adjoint = simd_load(adjoints);
                simd_t weight_a = nico_load_prefix_packet(weight_of_a + i, lanes, 0.0);
                simd_t weight_b = nico_load_prefix_packet(weight_of_b + i, lanes, 0.0);
                const nico_lse_mask_t both_negative_infinity =
                    nico_lse_equal(simd_load(values), scan_set1(stan::math::NEGATIVE_INFTY));
                if constexpr (a_is_var && !b_is_var) {
                    weight_a = nico_lse_select(both_negative_infinity, scan_set1(1.0), weight_a);
                }
                if constexpr (b_is_var && !a_is_var) {
                    weight_b = nico_lse_select(both_negative_infinity, scan_set1(1.0), weight_b);
                }
                if constexpr (a_is_var) {
                    if constexpr (a_is_scalar) {
                        adjoint_of_scalar_a = nico_lse_add(adjoint_of_scalar_a, scan_mul(adjoint, weight_a));
                    } else {
                        double* targets[BMVP_LANES];
                        for (int lane = 0; lane < lanes; ++lane) targets[lane] = &nodes_of_a[i + lane]->adj_;
                        nico_add_prefix_adjoint_packet(targets, lanes, weight_a, adjoint);
                    }
                }
                if constexpr (b_is_var) {
                    if constexpr (b_is_scalar) {
                        adjoint_of_scalar_b = nico_lse_add(adjoint_of_scalar_b, scan_mul(adjoint, weight_b));
                    } else {
                        double* targets[BMVP_LANES];
                        for (int lane = 0; lane < lanes; ++lane) targets[lane] = &nodes_of_b[i + lane]->adj_;
                        nico_add_prefix_adjoint_packet(targets, lanes, weight_b, adjoint);
                    }
                }
            }
            if constexpr (a_is_var && a_is_scalar) {
                double* target[1] = {&nodes_of_a[0]->adj_};
                nico_add_prefix_adjoint_packet(target, 1, scan_set1(nico_lse_sum(adjoint_of_scalar_a)),
                                               scan_set1(1.0));
            }
            if constexpr (b_is_var && b_is_scalar) {
                double* target[1] = {&nodes_of_b[0]->adj_};
                nico_add_prefix_adjoint_packet(target, 1, scan_set1(nico_lse_sum(adjoint_of_scalar_b)),
                                               scan_set1(1.0));
            }

        });
        return out;
    }

}

// ---- (3) the log_sum_exp of each row of a matrix (column-major, n_rows x n_columns) of doubles: results (n_rows)
inline void rows_of_values(const double* x, const int n_rows, const int n_columns, double* results,
                           double* buffer) {

    (void)buffer;
    // the row maxima, then the results
    for (int i = 0; i < n_rows; i += BMVP_LANES) {
        const int lanes = std::min(BMVP_LANES, n_rows - i);
        simd_t largest = scan_set1(stan::math::NEGATIVE_INFTY);
        for (int j = 0; j < n_columns; ++j) {
            const double* column = x + static_cast<std::ptrdiff_t>(j) * n_rows + i;
            largest = nico_lse_maximum_packet(largest,
                                               nico_load_prefix_packet(column, lanes,
                                                                        stan::math::NEGATIVE_INFTY));
        }
        const nico_lse_mask_t finite = nico_lse_finite(largest);
        simd_t sums = scan_zero();
        for (int j = 0; j < n_columns; ++j) {
            const double* column = x + static_cast<std::ptrdiff_t>(j) * n_rows + i;
            const simd_t values = nico_load_prefix_packet(column, lanes, stan::math::NEGATIVE_INFTY);
            const simd_t shifted = nico_lse_select(finite, nico_lse_subtract(values, largest),
                                                   scan_set1(stan::math::NEGATIVE_INFTY));
            sums = nico_lse_add(sums, nico_exp_of_nonpositive_prefix_argument(shifted));
        }
        const simd_t logarithms = nico_lse_log_positive_normal(nico_lse_select(finite, sums, scan_set1(1.0)));
        const simd_t value = nico_lse_select(finite, nico_lse_add(largest, logarithms), largest);
        // else: -inf or +inf (the maximum)
        nico_store_prefix_packet(results + i, lanes, value);
    }

}

template <typename T, stan::require_st_arithmetic<T>* = nullptr>
inline Eigen::Matrix<double, -1, 1> rows(const T& x_in) {

    const Eigen::Matrix<double, -1, -1> x = x_in;
    const int n_rows = static_cast<int>(x.rows());
    Eigen::Matrix<double, -1, 1> out(n_rows);
    std::vector<double> buffer(n_rows);
    rows_of_values(x.data(), n_rows, static_cast<int>(x.cols()), out.data(), buffer.data());
    return out;

}

template <typename T, stan::require_st_var<T>* = nullptr>
inline VecVar rows(const T& x_in) {

    const Eigen::Matrix<var, -1, -1> x = x_in;
    const int n_rows = static_cast<int>(x.rows());
    const int n_columns = static_cast<int>(x.cols());
    const int n = n_rows * n_columns;
    auto& arena = stan::math::ChainableStack::instance_->memalloc_;
    vari** nodes = arena.alloc_array<vari*>(n);
    std::vector<double> values(n), results(n_rows), buffer(n_rows);
    for (int j = 0; j < n_columns; ++j) {
        for (int i = 0; i < n_rows; ++i) {
            vari* node = x.coeff(i, j).vi_;
            nodes[static_cast<std::ptrdiff_t>(j) * n_rows + i] = node;
            values[static_cast<std::ptrdiff_t>(j) * n_rows + i] = node->val_;
        }
    }
    rows_of_values(values.data(), n_rows, n_columns, results.data(), buffer.data());
    vari* result_nodes = arena.alloc_array<vari>(n_rows);
    VecVar out(n_rows);
    for (int i = 0; i < n_rows; ++i) {
        ::new (static_cast<void*>(result_nodes + i)) vari(results[i], false);       // nochain stack
        out.coeffRef(i) = var(result_nodes + i);
    }
    stan::math::reverse_pass_callback([nodes, result_nodes, n_rows, n_columns]() {

        for (int i = 0; i < n_rows; i += BMVP_LANES) {
            const int lanes = std::min(BMVP_LANES, n_rows - i);
            alignas(64) double results[BMVP_LANES] = {};
            alignas(64) double adjoints[BMVP_LANES] = {};
            for (int lane = 0; lane < lanes; ++lane) {
                results[lane] = result_nodes[i + lane].val_;
                adjoints[lane] = result_nodes[i + lane].adj_;
            }
            for (int j = 0; j < n_columns; ++j) {
                vari** column = nodes + static_cast<std::ptrdiff_t>(j) * n_rows + i;
                alignas(64) double values[BMVP_LANES] = {};
                double* targets[BMVP_LANES];
                for (int lane = 0; lane < lanes; ++lane) {
                    values[lane] = column[lane]->val_;
                    targets[lane] = &column[lane]->adj_;
                }
                const simd_t shifted = nico_lse_subtract(simd_load(values), simd_load(results));
                nico_add_prefix_adjoint_packet(targets, lanes,
                                               nico_exp_of_nonpositive_prefix_argument(shifted),
                                               simd_load(adjoints));
            }
        }

    });
    return out;

}

}  // namespace log_sum_exp_simd
}  // namespace nico::math

// ---- the Stan-facing functions (global scope; stanc calls them as  nico_log_sum_exp(x, pstream__))
// (1) the reduction: vector, row_vector, matrix (any Eigen object) or array[] real
namespace nico::math {

template <typename T, stan::require_eigen_t<T>* = nullptr>
inline auto nico_log_sum_exp(const T& x, std::ostream* /*pstream__*/) {

    return nico::math::log_sum_exp_simd::reduction(x);

}

}  // namespace nico::math

using nico::math::nico_log_sum_exp;
namespace nico::math {

template <typename T, stan::require_std_vector_t<T>* = nullptr,
          stan::require_stan_scalar_t<stan::value_type_t<T>>* = nullptr>
inline auto nico_log_sum_exp(const T& x, std::ostream* /*pstream__*/) {

    return nico::math::log_sum_exp_simd::reduction(x);

}

}  // namespace nico::math

using nico::math::nico_log_sum_exp;
// (2) two arguments: two reals (Stan's own scalar log_sum_exp), or elementwise with at least one vector
namespace nico::math {

template <typename T1, typename T2, stan::require_all_stan_scalar_t<T1, T2>* = nullptr>
inline auto nico_log_sum_exp(const T1& a, const T2& b, std::ostream* /*pstream__*/) {

    return stan::math::log_sum_exp(a, b);

}

}  // namespace nico::math

using nico::math::nico_log_sum_exp;
namespace nico::math {

template <typename T1, typename T2, stan::require_any_eigen_vector_t<T1, T2>* = nullptr>
inline auto nico_log_sum_exp(const T1& a, const T2& b, std::ostream* /*pstream__*/) {

    return nico::math::log_sum_exp_simd::pairs(a, b);

}

}  // namespace nico::math

using nico::math::nico_log_sum_exp;
// (3) the row-wise reduction of a matrix
namespace nico::math {

template <typename T, stan::require_eigen_t<T>* = nullptr>
inline auto nico_log_sum_exp_2d(const T& x, std::ostream* /*pstream__*/) {

    return nico::math::log_sum_exp_simd::rows(x);

}

}  // namespace nico::math

using nico::math::nico_log_sum_exp_2d;
#endif






















