#pragma once
//
// -| --------- BayesMVP fused AVX-512 / AVX2 Weibull log density and log CCDF, as Stan external functions -------
//
// nico_weibull_lpdf and nico_weibull_lccdf: drop-in replacements for Stan's weibull_lpdf and weibull_lccdf, with
// the same arguments and the same meaning, each computed as ONE fused function: the value AND the partial
// derivatives with respect to every argument which is a parameter (y, alpha, sigma) are computed in one SIMD pass
// over the elements, with the BayesMVP kernels (nico_log_1_*, nico_exp_1_*, as in stan_external_functions.hpp),
// and the result is ONE autodiff node, whose reverse pass is one callback:
//
//     log f(y | alpha, sigma) =  log(alpha) + (alpha - 1) log(y) - alpha log(sigma) - (y / sigma)^alpha
//     log S(y | alpha, sigma) = -(y / sigma)^alpha
//
// with (y / sigma)^alpha = exp(alpha (log(y) - log(sigma))). Stan's weibull_lpdf / weibull_lccdf instead
// evaluate pow(y / sigma, alpha) element by element (scalar pow) and keep several temporary vectors.
//
// Declared in the Stan functions block without a body, e.g.
//
//     real nico_weibull_lpdf(vector y, real alpha, vector sigma);
//     real nico_weibull_lccdf(vector y, real alpha, vector sigma);
//
// and compiled with --allow-undefined, with this file included after stan_external_functions.hpp (at its end).
// Every combination of Stan's argument types is supported (one Stan declaration per combination used):
//
//     y:      vector or array[] real (data or parameters)
//     alpha:  real, vector or array[] real (data or parameters; int data too)
//     sigma:  real, vector or array[] real (data or parameters; int data too)
//
// stanc calls an external lpdf as nico_weibull_lpdf<propto__>(y, alpha, sigma, pstream__) (for
// y ~ nico_weibull(...) and nico_weibull_lupdf(y | ...); nico_weibull_lpdf(y | ...) gives <false>) and an
// external lccdf as nico_weibull_lccdf(y, alpha, sigma, pstream__). The result is a double when every argument
// is data, and a var otherwise.
//
// propto (lpdf only): the same terms as Stan's weibull_lpdf, i.e. log(alpha) is dropped when alpha is data,
// (alpha - 1) log(y) when y and alpha are data, and alpha log(sigma) when alpha and sigma are data (so the value
// equals weibull_lupdf's); with propto = false, the value is weibull_lpdf's.
//
// Checks, as Stan's weibull_lpdf / weibull_lccdf: consistent sizes, y finite (lpdf) or non-negative (lccdf),
// alpha and sigma positive and finite. The checks of the vector arguments are lane masks inside the SIMD pass; a
// failure calls Stan's own check functions, which throw Stan's error messages. As in weibull_lpdf, y < 0 anywhere
// gives a log density of negative infinity. (Stan's weibull_lccdf does not check that the sizes agree; this one
// does.)
//
// Autodiff memory: per element of each vector argument which is a parameter, 8 bytes (its vari pointer) and
// 8 bytes (its partial derivative) on the arena, and one callback vari for the whole call. Reverse pass: the
// adjoint of each operand += the adjoint of the result * its partial derivative, in element order (so a vari
// which appears twice is updated twice).
//
// Tail handling: a partial final block goes through padded stack buffers whose spare lanes repeat the last valid
// element; the spare lanes are neither summed nor stored.
//
// ---------------------------------------------------------------------------------------------------------------

#ifndef BMVP_LANES
#  error "stan_external_fused_weibull_lpdf.hpp: include stan_external_functions.hpp first (it has the kernels)"
#endif

#include <cmath>
#include <ostream>
#include <type_traits>

namespace nico::math {
namespace fused_weibull {

// ---- SIMD helpers, of the same width as the kernels: 8 lanes (AVX-512) or 4 lanes (AVX2)
#if BMVP_LANES == 8
using simd_mask_t = __mmask8;                                   // bit k: lane k
ALWAYS_INLINE simd_t simd_set1(const double x)                               { return _mm512_set1_pd(x); }
ALWAYS_INLINE simd_t simd_add(const simd_t a, const simd_t b)                { return _mm512_add_pd(a, b); }
ALWAYS_INLINE simd_t simd_sub(const simd_t a, const simd_t b)                { return _mm512_sub_pd(a, b); }
ALWAYS_INLINE simd_t simd_mul(const simd_t a, const simd_t b)                { return _mm512_mul_pd(a, b); }
ALWAYS_INLINE simd_t simd_div(const simd_t a, const simd_t b)                { return _mm512_div_pd(a, b); }
ALWAYS_INLINE simd_t simd_fmsub(simd_t a, simd_t b, simd_t c)                { return _mm512_fmsub_pd(a, b, c); }
ALWAYS_INLINE simd_t simd_fnmadd(simd_t a, simd_t b, simd_t c)               { return _mm512_fnmadd_pd(a, b, c); }
ALWAYS_INLINE double simd_sum_of_lanes(const simd_t a)                       { return _mm512_reduce_add_pd(a); }
ALWAYS_INLINE simd_mask_t simd_mask_of_all_lanes()                           { return 0xFF; }
ALWAYS_INLINE simd_mask_t simd_mask_of_no_lanes()                            { return 0x00; }
ALWAYS_INLINE simd_mask_t simd_mask_and(const simd_mask_t a, const simd_mask_t b) { return a & b; }
ALWAYS_INLINE simd_mask_t simd_mask_or(const simd_mask_t a, const simd_mask_t b)  { return a | b; }
ALWAYS_INLINE bool simd_mask_is_all_lanes(const simd_mask_t m)               { return m == 0xFF; }
ALWAYS_INLINE bool simd_mask_is_any_lane(const simd_mask_t m)                { return m != 0x00; }
ALWAYS_INLINE simd_mask_t simd_lanes_finite(const simd_t x) {

    return _mm512_cmp_pd_mask(_mm512_abs_pd(x), _mm512_set1_pd(INFINITY), _CMP_LT_OQ);

}
ALWAYS_INLINE simd_mask_t simd_lanes_positive_and_finite(const simd_t x) {

    return _mm512_cmp_pd_mask(x, _mm512_setzero_pd(), _CMP_GT_OQ)
         & _mm512_cmp_pd_mask(x, _mm512_set1_pd(INFINITY), _CMP_LT_OQ);

}
ALWAYS_INLINE simd_mask_t simd_lanes_nonnegative(const simd_t x) {

    return _mm512_cmp_pd_mask(x, _mm512_setzero_pd(), _CMP_GE_OQ);

}
ALWAYS_INLINE simd_mask_t simd_lanes_negative(const simd_t x) {

    return _mm512_cmp_pd_mask(x, _mm512_setzero_pd(), _CMP_LT_OQ);

}
#else
using simd_mask_t = __m256d;                                    // lane k: all ones (true) or all zeros (false)
ALWAYS_INLINE simd_t simd_set1(const double x)                               { return _mm256_set1_pd(x); }
ALWAYS_INLINE simd_t simd_add(const simd_t a, const simd_t b)                { return _mm256_add_pd(a, b); }
ALWAYS_INLINE simd_t simd_sub(const simd_t a, const simd_t b)                { return _mm256_sub_pd(a, b); }
ALWAYS_INLINE simd_t simd_mul(const simd_t a, const simd_t b)                { return _mm256_mul_pd(a, b); }
ALWAYS_INLINE simd_t simd_div(const simd_t a, const simd_t b)                { return _mm256_div_pd(a, b); }
ALWAYS_INLINE simd_t simd_fmsub(simd_t a, simd_t b, simd_t c)                { return _mm256_fmsub_pd(a, b, c); }
ALWAYS_INLINE simd_t simd_fnmadd(simd_t a, simd_t b, simd_t c)               { return _mm256_fnmadd_pd(a, b, c); }
ALWAYS_INLINE double simd_sum_of_lanes(const simd_t a) {

    const __m128d sum_of_halves = _mm_add_pd(_mm256_castpd256_pd128(a), _mm256_extractf128_pd(a, 1));
    return _mm_cvtsd_f64(_mm_add_sd(sum_of_halves, _mm_unpackhi_pd(sum_of_halves, sum_of_halves)));

}
ALWAYS_INLINE simd_mask_t simd_mask_of_all_lanes() { return _mm256_castsi256_pd(_mm256_set1_epi64x(-1)); }
ALWAYS_INLINE simd_mask_t simd_mask_of_no_lanes()  { return _mm256_setzero_pd(); }
ALWAYS_INLINE simd_mask_t simd_mask_and(const simd_mask_t a, const simd_mask_t b) { return _mm256_and_pd(a, b); }
ALWAYS_INLINE simd_mask_t simd_mask_or(const simd_mask_t a, const simd_mask_t b)  { return _mm256_or_pd(a, b); }
ALWAYS_INLINE bool simd_mask_is_all_lanes(const simd_mask_t m)  { return _mm256_movemask_pd(m) == 0xF; }
ALWAYS_INLINE bool simd_mask_is_any_lane(const simd_mask_t m)   { return _mm256_movemask_pd(m) != 0x0; }
ALWAYS_INLINE simd_mask_t simd_lanes_finite(const simd_t x) {

    // |x| from the bits (an integer mask: a -0.0 constant would not survive -fno-signed-zeros)
    const simd_t absolute_value = _mm256_and_pd(x, _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFF)));
    return _mm256_cmp_pd(absolute_value, _mm256_set1_pd(INFINITY), _CMP_LT_OQ);

}
ALWAYS_INLINE simd_mask_t simd_lanes_positive_and_finite(const simd_t x) {

    return _mm256_and_pd(_mm256_cmp_pd(x, _mm256_setzero_pd(), _CMP_GT_OQ),
                         _mm256_cmp_pd(x, _mm256_set1_pd(INFINITY), _CMP_LT_OQ));

}
ALWAYS_INLINE simd_mask_t simd_lanes_nonnegative(const simd_t x) {

    return _mm256_cmp_pd(x, _mm256_setzero_pd(), _CMP_GE_OQ);

}
ALWAYS_INLINE simd_mask_t simd_lanes_negative(const simd_t x) {

    return _mm256_cmp_pd(x, _mm256_setzero_pd(), _CMP_LT_OQ);

}
#endif

// ---- the two kernels (the checked versions: a lane outside the fast path goes to std::log / std::exp).
//      (nico_log_1 was accurate to about 1e-8 (its ln 2 and polynomial coefficients are single-precision
//      values), so the results agreed with Stan's to about 1e-8 (relative).) The log is now the
//      double-precision nico_log_1_double_precision (below 1 ulp). Compiled with
//      -DBAYESMVP_FUSED_WEIBULL_USE_STD_LOG_AND_EXP, lane-wise std::log / std::exp are used instead (slow): the
//      correctness check of the fused algebra and partials against Stan, without the kernels' approximation.
#ifdef BAYESMVP_FUSED_WEIBULL_USE_STD_LOG_AND_EXP
ALWAYS_INLINE simd_t simd_log(const simd_t x) {

    alignas(64) double lanes[BMVP_LANES];
    simd_store(lanes, x);
    for (int lane = 0; lane < BMVP_LANES; ++lane) lanes[lane] = std::log(lanes[lane]);
    return simd_load(lanes);

}
ALWAYS_INLINE simd_t simd_exp(const simd_t x) {

    alignas(64) double lanes[BMVP_LANES];
    simd_store(lanes, x);
    for (int lane = 0; lane < BMVP_LANES; ++lane) lanes[lane] = std::exp(lanes[lane]);
    return simd_load(lanes);

}
#else
// ALWAYS_INLINE simd_t simd_log(const simd_t x) { return BMVP_KERNEL_NAME(nico_log_1)(x); }
//// (the double-precision log, fdlibm, one division per vector: the log density and the gradient agree with Stan's
////  to about 1e-15 relative, in place of about 1e-8 with nico_log_1; measured on the laptop at full load, 16
////  threads: the fused Weibull 7.2% slower than with nico_log_1, and 2.19x faster than Stan's weibull_lpdf +
////  weibull_lccdf. It uses the double-precision log, which costs 7.2%, and anything under 10% slower is kept.)
ALWAYS_INLINE simd_t simd_log(const simd_t x) { return BMVP_KERNEL_NAME(nico_log_1_double_precision)(x); }
ALWAYS_INLINE simd_t simd_exp(const simd_t x) { return BMVP_KERNEL_NAME(nico_exp_1)(x); }
#endif

// ---- one SIMD block of a vector argument: elements first, ..., first + count - 1. A full block (count =
//      BMVP_LANES) is loaded / stored directly; the final partial block goes through a padded stack buffer.
template <bool full_block>
ALWAYS_INLINE simd_t load_block(const double* values, const int first, const int count) {

    if constexpr (full_block) {
        return simd_loadu(values + first);
    } else {
        alignas(64) double padded[BMVP_LANES];
        for (int lane = 0; lane < BMVP_LANES; ++lane) {
            padded[lane] = values[first + (lane < count ? lane : count - 1)];
        }
        return simd_load(padded);
    }

}

template <bool full_block>
ALWAYS_INLINE void store_block(double* values, const int first, const int count, const simd_t block) {

    if constexpr (full_block) {
        simd_storeu(values + first, block);
    } else {
        alignas(64) double lanes[BMVP_LANES];
        simd_store(lanes, block);
        for (int lane = 0; lane < count; ++lane) values[first + lane] = lanes[lane];
    }

}

// (before a sum, the spare lanes of the partial block are set to 0: they may hold inf or NaN)
template <bool full_block>
ALWAYS_INLINE simd_t zero_spare_lanes(const simd_t block, const int count) {

    if constexpr (full_block) {
        return block;
    } else {
        alignas(64) double lanes[BMVP_LANES];
        simd_store(lanes, block);
        for (int lane = count; lane < BMVP_LANES; ++lane) lanes[lane] = 0.0;
        return simd_load(lanes);
    }

}

// ---- which function: the log density (lpdf) or the log complementary CDF (lccdf)
enum class weibull_function { log_density, log_ccdf };

// ---- everything known at compile time about one call: the argument types and the terms of the log density
template <bool propto, weibull_function function, typename T_y, typename T_shape, typename T_scale>
struct weibull_configuration {
    static constexpr bool is_log_density   = (function == weibull_function::log_density);
    static constexpr bool alpha_is_vector  = !stan::is_stan_scalar<std::decay_t<T_shape>>::value;
    static constexpr bool sigma_is_vector  = !stan::is_stan_scalar<std::decay_t<T_scale>>::value;
    static constexpr bool y_is_var         = !stan::is_constant_all<T_y>::value;
    static constexpr bool alpha_is_var     = !stan::is_constant_all<T_shape>::value;
    static constexpr bool sigma_is_var     = !stan::is_constant_all<T_scale>::value;
    static constexpr bool all_are_data     = stan::is_constant_all<T_y, T_shape, T_scale>::value;
    // the terms of the log density besides -(y / sigma)^alpha (Stan's include_summand: every term when propto is
    // false; when propto is true, a term without parameters is dropped)
    static constexpr bool include_log_alpha_term = is_log_density
                                                && stan::math::include_summand<propto, T_shape>::value;
    static constexpr bool include_log_y_term     = is_log_density
                                                && stan::math::include_summand<propto, T_y, T_shape>::value;
    static constexpr bool include_log_sigma_term = is_log_density
                                                && stan::math::include_summand<propto, T_shape, T_scale>::value;
};

// ---- the inputs and outputs of the SIMD pass: contiguous double values, and arena arrays for the partials
struct weibull_pass_inputs_and_outputs {
    int           n                           = 0;
    const double* y                           = nullptr;   // n values
    const double* alpha                       = nullptr;   // n values, when alpha is a vector
    double        alpha_scalar                = 0.0;       //           when alpha is a scalar
    const double* sigma                       = nullptr;   // n values, when sigma is a vector
    double        sigma_scalar                = 0.0;       //           when sigma is a scalar
    double*       y_partials                  = nullptr;   // n, when y is a vector of var
    double*       alpha_partials              = nullptr;   // n, when alpha is a vector of var
    double*       sigma_partials              = nullptr;   // n, when sigma is a vector of var
    double        value                       = 0.0;       // the log density (log CCDF), summed over the elements
    double        alpha_partial_sum           = 0.0;       // when alpha is a scalar var
    double        sigma_partial_sum           = 0.0;       // when sigma is a scalar var
    bool          vector_arguments_are_valid  = true;
    bool          some_y_is_negative          = false;
};

// ---- the fused SIMD pass: the value and every partial derivative, in one pass over the elements
//
//   z = log(y) - log(sigma),   w = exp(alpha z) = (y / sigma)^alpha
//   log density:  terms   log(alpha) + (alpha - 1) log(y) - alpha log(sigma) - w
//                 d / dy      = (alpha (1 - w) - 1) / y
//                 d / dalpha  = 1 / alpha + (1 - w) z
//                 d / dsigma  = alpha (w - 1) / sigma
//   log CCDF:     term    -w
//                 d / dy      = -alpha w / y
//                 d / dalpha  = -z w
//                 d / dsigma  = alpha w / sigma
//
// A term or partial of a scalar argument which is the same in every element is computed once, after the pass.
template <typename configuration>
inline void run_weibull_pass(weibull_pass_inputs_and_outputs& pass) {

    using c = configuration;
    const int n = pass.n;
    const simd_t one = simd_set1(1.0);
    const simd_t zero = simd_set1(0.0);
    const simd_t alpha_broadcast = simd_set1(pass.alpha_scalar);
    double log_sigma_scalar = 0.0;                              // log(sigma) of a scalar sigma, once
    if constexpr (!c::sigma_is_vector) log_sigma_scalar = std::log(pass.sigma_scalar);
    const simd_t log_sigma_broadcast = simd_set1(log_sigma_scalar);
    simd_t value_sum = zero;
    simd_t alpha_partial_sum = zero;
    simd_t sigma_partial_sum = zero;
    simd_mask_t valid_lanes = simd_mask_of_all_lanes();
    simd_mask_t negative_y_lanes = simd_mask_of_no_lanes();

    const auto update_block = [&](auto full_block_tag, const int first, const int count) {

        constexpr bool full_block = decltype(full_block_tag)::value;
        // the values, and the checks of the vector arguments (as lane masks)
        const simd_t y = load_block<full_block>(pass.y, first, count);
        if constexpr (c::is_log_density) {
            valid_lanes = simd_mask_and(valid_lanes, simd_lanes_finite(y));
            negative_y_lanes = simd_mask_or(negative_y_lanes, simd_lanes_negative(y));
        } else {
            valid_lanes = simd_mask_and(valid_lanes, simd_lanes_nonnegative(y));
        }
        simd_t alpha = alpha_broadcast;
        if constexpr (c::alpha_is_vector) {
            alpha = load_block<full_block>(pass.alpha, first, count);
            valid_lanes = simd_mask_and(valid_lanes, simd_lanes_positive_and_finite(alpha));
        }
        simd_t sigma = zero;
        simd_t log_sigma = log_sigma_broadcast;
        if constexpr (c::sigma_is_vector) {
            sigma = load_block<full_block>(pass.sigma, first, count);
            valid_lanes = simd_mask_and(valid_lanes, simd_lanes_positive_and_finite(sigma));
            log_sigma = simd_log(sigma);
        }
        // z = log(y / sigma) and w = (y / sigma)^alpha
        const simd_t log_y = simd_log(y);
        const simd_t z = simd_sub(log_y, log_sigma);
        const simd_t w = simd_exp(simd_mul(alpha, z));

        if constexpr (c::is_log_density) {
            // value: (alpha - 1) log(y) - w [- alpha log(sigma)] [+ log(alpha)]
            simd_t terms;
            if constexpr (c::include_log_y_term) terms = simd_fmsub(simd_sub(alpha, one), log_y, w);
            else                                 terms = simd_sub(zero, w);
            if constexpr (c::include_log_sigma_term && (c::alpha_is_vector || c::sigma_is_vector)) {
                terms = simd_fnmadd(alpha, log_sigma, terms);
            }
            if constexpr (c::include_log_alpha_term && c::alpha_is_vector) {
                terms = simd_add(terms, simd_log(alpha));
            }
            value_sum = simd_add(value_sum, zero_spare_lanes<full_block>(terms, count));
            // partials
            if constexpr (c::y_is_var) {
                const simd_t y_partial = simd_div(simd_sub(simd_fnmadd(alpha, w, alpha), one), y);
                store_block<full_block>(pass.y_partials, first, count, y_partial);
            }
            if constexpr (c::alpha_is_var) {
                // ((1 - w) z, not z - w z: at y = 0, z = -inf and w = 0, and z - w z would be NaN)
                const simd_t one_minus_w_times_z = simd_mul(simd_sub(one, w), z);
                if constexpr (c::alpha_is_vector) {
                    store_block<full_block>(pass.alpha_partials, first, count,
                                            simd_add(simd_div(one, alpha), one_minus_w_times_z));
                } else {
                    alpha_partial_sum = simd_add(alpha_partial_sum,
                                                 zero_spare_lanes<full_block>(one_minus_w_times_z, count));
                }
            }
            if constexpr (c::sigma_is_var) {
                const simd_t alpha_times_w_minus_one = simd_fmsub(alpha, w, alpha);
                if constexpr (c::sigma_is_vector) {
                    store_block<full_block>(pass.sigma_partials, first, count,
                                            simd_div(alpha_times_w_minus_one, sigma));
                } else {
                    sigma_partial_sum = simd_add(sigma_partial_sum,
                                                 zero_spare_lanes<full_block>(alpha_times_w_minus_one, count));
                }
            }
        } else {
            // value: -w
            value_sum = simd_sub(value_sum, zero_spare_lanes<full_block>(w, count));
            // partials
            if constexpr (c::y_is_var) {
                store_block<full_block>(pass.y_partials, first, count, simd_div(simd_fnmadd(alpha, w, zero), y));
            }
            if constexpr (c::alpha_is_var) {
                const simd_t minus_z_times_w = simd_fnmadd(z, w, zero);
                if constexpr (c::alpha_is_vector) {
                    store_block<full_block>(pass.alpha_partials, first, count, minus_z_times_w);
                } else {
                    alpha_partial_sum = simd_add(alpha_partial_sum,
                                                 zero_spare_lanes<full_block>(minus_z_times_w, count));
                }
            }
            if constexpr (c::sigma_is_var) {
                const simd_t alpha_times_w = simd_mul(alpha, w);
                if constexpr (c::sigma_is_vector) {
                    store_block<full_block>(pass.sigma_partials, first, count, simd_div(alpha_times_w, sigma));
                } else {
                    sigma_partial_sum = simd_add(sigma_partial_sum,
                                                 zero_spare_lanes<full_block>(alpha_times_w, count));
                }
            }
        }

    };
    // (a constant full-block size lets the compiler remove the padding from the hot loop)
    int first = 0;
    for (; first + BMVP_LANES <= n; first += BMVP_LANES) update_block(std::true_type(), first, BMVP_LANES);
    if (first < n) update_block(std::false_type(), first, n - first);

    pass.vector_arguments_are_valid = simd_mask_is_all_lanes(valid_lanes);
    pass.some_y_is_negative = simd_mask_is_any_lane(negative_y_lanes);
    // the terms of scalar arguments which are the same in every element
    double value = simd_sum_of_lanes(value_sum);
    if constexpr (c::include_log_alpha_term && !c::alpha_is_vector) value += n * std::log(pass.alpha_scalar);
    if constexpr (c::include_log_sigma_term && !c::alpha_is_vector && !c::sigma_is_vector) {
        value -= n * pass.alpha_scalar * log_sigma_scalar;
    }
    pass.value = value;
    if constexpr (c::alpha_is_var && !c::alpha_is_vector) {
        double alpha_partial = simd_sum_of_lanes(alpha_partial_sum);
        if constexpr (c::is_log_density) alpha_partial += n / pass.alpha_scalar;
        pass.alpha_partial_sum = alpha_partial;
    }
    if constexpr (c::sigma_is_var && !c::sigma_is_vector) {
        pass.sigma_partial_sum = simd_sum_of_lanes(sigma_partial_sum) / pass.sigma_scalar;
    }

}

// ---- the argument types: y a vector (Eigen column vector or std::vector), alpha and sigma a vector or a scalar;
//      every element (or scalar) a double, an int or a var
template <typename T>
struct is_weibull_scalar_argument
    : std::integral_constant<bool, std::is_arithmetic<std::decay_t<T>>::value
                                   || stan::is_var<std::decay_t<T>>::value> {};

template <typename T>
struct is_weibull_vector_argument
    : std::integral_constant<bool, (stan::is_eigen_col_vector<std::decay_t<T>>::value
                                    || stan::is_std_vector<std::decay_t<T>>::value)
                                   && is_weibull_scalar_argument<stan::value_type_t<std::decay_t<T>>>::value> {};

template <typename T_y, typename T_shape, typename T_scale>
using require_weibull_argument_types_t = std::enable_if_t<
      is_weibull_vector_argument<T_y>::value
      && (is_weibull_scalar_argument<T_shape>::value || is_weibull_vector_argument<T_shape>::value)
      && (is_weibull_scalar_argument<T_scale>::value || is_weibull_vector_argument<T_scale>::value)>;

// ---- contiguous double values of a vector argument of data: its own memory (after to_ref, which evaluates an
//      expression once), or a copy into the scratch buffer when its elements are not doubles (e.g., int)
template <typename T>
inline const double* data_vector_values(const T& x_ref, const int n, double* buffer) {

    if constexpr (std::is_same<stan::value_type_t<T>, double>::value) {
        return x_ref.data();
    } else {
        for (int i = 0; i < n; ++i) buffer[i] = static_cast<double>(x_ref[i]);
        return buffer;
    }

}

// ---- the values and vari pointers of a vector argument of parameters (after to_ref: contiguous vars)
template <typename T>
inline void gather_parameter_vector_values_and_nodes(const T& x_ref, const int n, double* values, vari** nodes) {

    const var* operands = x_ref.data();
    for (int i = 0; i < n; ++i) {
        vari* node = operands[i].vi_;
        nodes[i] = node;
        values[i] = node->val_;
    }

}

// ---- the reverse pass data, captured by value in the callback (trivially destructible: arena pointers)
struct weibull_reverse_pass_data {
    int     n               = 0;
    vari**  y_nodes         = nullptr;
    double* y_partials      = nullptr;
    vari**  alpha_nodes     = nullptr;   // alpha a vector of var
    double* alpha_partials  = nullptr;
    vari*   alpha_node      = nullptr;   // alpha a scalar var
    double  alpha_partial   = 0.0;
    vari**  sigma_nodes     = nullptr;   // sigma a vector of var
    double* sigma_partials  = nullptr;
    vari*   sigma_node      = nullptr;   // sigma a scalar var
    double  sigma_partial   = 0.0;
};

// ---- a failed lane check: Stan's own checks, which throw Stan's error messages
template <typename configuration, typename T_y, typename T_shape, typename T_scale>
__attribute__((noinline, cold)) inline void throw_with_Stan_checks(const char* function_name, const T_y& y,
                                                                   const T_shape& alpha, const T_scale& sigma) {

    using stan::math::check_finite;
    using stan::math::check_nonnegative;
    using stan::math::check_positive_finite;
    if constexpr (configuration::is_log_density) check_finite(function_name, "Random variable", y);
    else                                         check_nonnegative(function_name, "Random variable", y);
    check_positive_finite(function_name, "Shape parameter", alpha);
    check_positive_finite(function_name, "Scale parameter", sigma);

}

// ---- the two Stan-facing functions, both through this one
template <bool propto, weibull_function function, typename T_y, typename T_shape, typename T_scale>
inline stan::return_type_t<T_y, T_shape, T_scale> fused_weibull(const char* function_name, const T_y& y,
                                                                const T_shape& alpha, const T_scale& sigma) {

    using c = weibull_configuration<propto, function, T_y, T_shape, T_scale>;
    using return_t = stan::return_type_t<T_y, T_shape, T_scale>;
    stan::math::check_consistent_sizes(function_name, "Random variable", y, "Shape parameter", alpha,
                                       "Scale parameter", sigma);
    // the scalar arguments are checked first, as Stan checks every argument before an empty y returns 0
    if constexpr (!c::alpha_is_vector) stan::math::check_positive_finite(function_name, "Shape parameter", alpha);
    if constexpr (!c::sigma_is_vector) stan::math::check_positive_finite(function_name, "Scale parameter", sigma);
    const int n = static_cast<int>(stan::math::size(y));
    if (n == 0) return return_t(0.0);
    // (propto with only data: every term is dropped, after the checks, as in weibull_lpdf)
    if constexpr (c::is_log_density && propto && c::all_are_data) {
        throw_with_Stan_checks<c>(function_name, y, alpha, sigma);
        return return_t(0.0);
    }

    const auto& y_ref = stan::math::to_ref(y);
    const auto& alpha_ref = stan::math::to_ref(alpha);
    const auto& sigma_ref = stan::math::to_ref(sigma);
    scratch_t& buffers = scratch();
    buffers.ensure(n);
    weibull_pass_inputs_and_outputs pass;
    weibull_reverse_pass_data reverse;
    pass.n = n;
    reverse.n = n;

    // y: its values, and its vari pointers and partials (arena) when it is a vector of var
    if constexpr (c::y_is_var) {
        auto& arena = stan::math::ChainableStack::instance_->memalloc_;
        reverse.y_nodes = arena.alloc_array<vari*>(n);
        gather_parameter_vector_values_and_nodes(y_ref, n, buffers.a.data(), reverse.y_nodes);
        pass.y = buffers.a.data();
        pass.y_partials = reverse.y_partials = arena.alloc_array<double>(n);
    } else {
        pass.y = data_vector_values(y_ref, n, buffers.a.data());
    }
    // alpha
    if constexpr (c::alpha_is_vector) {
        if constexpr (c::alpha_is_var) {
            auto& arena = stan::math::ChainableStack::instance_->memalloc_;
            reverse.alpha_nodes = arena.alloc_array<vari*>(n);
            gather_parameter_vector_values_and_nodes(alpha_ref, n, buffers.b.data(), reverse.alpha_nodes);
            pass.alpha = buffers.b.data();
            pass.alpha_partials = reverse.alpha_partials = arena.alloc_array<double>(n);
        } else {
            pass.alpha = data_vector_values(alpha_ref, n, buffers.b.data());
        }
    } else {
        pass.alpha_scalar = static_cast<double>(stan::math::value_of(alpha_ref));
        if constexpr (c::alpha_is_var) reverse.alpha_node = alpha_ref.vi_;
    }
    // sigma
    if constexpr (c::sigma_is_vector) {
        if constexpr (c::sigma_is_var) {
            auto& arena = stan::math::ChainableStack::instance_->memalloc_;
            reverse.sigma_nodes = arena.alloc_array<vari*>(n);
            gather_parameter_vector_values_and_nodes(sigma_ref, n, buffers.c.data(), reverse.sigma_nodes);
            pass.sigma = buffers.c.data();
            pass.sigma_partials = reverse.sigma_partials = arena.alloc_array<double>(n);
        } else {
            pass.sigma = data_vector_values(sigma_ref, n, buffers.c.data());
        }
    } else {
        pass.sigma_scalar = static_cast<double>(stan::math::value_of(sigma_ref));
        if constexpr (c::sigma_is_var) reverse.sigma_node = sigma_ref.vi_;
    }

    run_weibull_pass<c>(pass);

    if (!pass.vector_arguments_are_valid) throw_with_Stan_checks<c>(function_name, y, alpha, sigma);
    if constexpr (c::is_log_density) {
        if (pass.some_y_is_negative) return return_t(stan::math::LOG_ZERO);
    }
    if constexpr (c::all_are_data) {
        return pass.value;
    } else {
        reverse.alpha_partial = pass.alpha_partial_sum;
        reverse.sigma_partial = pass.sigma_partial_sum;
        const double value = pass.value;
        return stan::math::make_callback_var(value, [reverse](auto& result) {

            const double adjoint = result.adj_;
            if constexpr (c::y_is_var) {
                for (int i = 0; i < reverse.n; ++i) reverse.y_nodes[i]->adj_ += adjoint * reverse.y_partials[i];
            }
            if constexpr (c::alpha_is_var) {
                if constexpr (c::alpha_is_vector) {
                    for (int i = 0; i < reverse.n; ++i) {
                        reverse.alpha_nodes[i]->adj_ += adjoint * reverse.alpha_partials[i];
                    }
                } else {
                    reverse.alpha_node->adj_ += adjoint * reverse.alpha_partial;
                }
            }
            if constexpr (c::sigma_is_var) {
                if constexpr (c::sigma_is_vector) {
                    for (int i = 0; i < reverse.n; ++i) {
                        reverse.sigma_nodes[i]->adj_ += adjoint * reverse.sigma_partials[i];
                    }
                } else {
                    reverse.sigma_node->adj_ += adjoint * reverse.sigma_partial;
                }
            }

        });
    }

}

}  // namespace fused_weibull
}  // namespace nico::math

// ---- the Stan-facing functions (global scope; see the header comment for the Stan declarations)
namespace nico::math {

template <bool propto, typename T_y, typename T_shape, typename T_scale,
          nico::math::fused_weibull::require_weibull_argument_types_t<T_y, T_shape, T_scale>* = nullptr>
inline stan::return_type_t<T_y, T_shape, T_scale> nico_weibull_lpdf(const T_y& y, const T_shape& alpha,
                                                                    const T_scale& sigma,
                                                                    std::ostream* /*pstream__*/) {

    using namespace nico::math::fused_weibull;
    return nico::math::fused_weibull::fused_weibull<propto, weibull_function::log_density>(
        "nico_weibull_lpdf", y, alpha, sigma);

}

}  // namespace nico::math

using nico::math::nico_weibull_lpdf;

namespace nico::math {

template <typename T_y, typename T_shape, typename T_scale,
          nico::math::fused_weibull::require_weibull_argument_types_t<T_y, T_shape, T_scale>* = nullptr>
inline stan::return_type_t<T_y, T_shape, T_scale> nico_weibull_lccdf(const T_y& y, const T_shape& alpha,
                                                                     const T_scale& sigma,
                                                                     std::ostream* /*pstream__*/) {

    using namespace nico::math::fused_weibull;
    return nico::math::fused_weibull::fused_weibull<false, weibull_function::log_ccdf>(
        "nico_weibull_lccdf", y, alpha, sigma);

}

}  // namespace nico::math

using nico::math::nico_weibull_lccdf;























