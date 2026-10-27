#pragma once
//
// -| --------- sinh, cosh, asinh, acosh and atanh, exposed to Stan as external functions --------------------------------------
//
// nico_sinh, nico_cosh, nico_asinh, nico_acosh, nico_atanh: real, vector, row_vector and matrix; double and var.
// Each is computed with NicoSIMD's double-precision exp and log kernels (8 lanes on AVX-512, 4 on AVX2), with
// the same tape as the other elementwise functions (rev_apply_strip_mined in stan_external_functions.hpp).
//
//   sinh(x)  = sign(x) (e - 1 / e) / 2, e = exp(|x|), where |x| > 0.5; the odd Taylor series to x^15 / 15! where
//              |x| <= 0.5 (the first term left out is below 5e-20 of the value); exp(|x| - log(2)) where
//              |x| > 709 (exp(|x|) overflows there, sinh(x) does not)
//   cosh(x)  = (e + 1 / e) / 2, e = exp(|x|); exp(|x| - log(2)) where |x| > 709
//   asinh(x) = sign(x) log(s + sqrt(s^2 + 1)), s = |x|, where 0.125 < s <= 1e8; log(s) + log(2) where s > 1e8
//              (s^2 + 1 = s^2 to double precision there); the odd Taylor series to x^17 where s <= 0.125
//   acosh(x) = log1p(u), u = (x - 1) + sqrt((x - 1) (x + 1)), x >= 1: log1p(u) = 2 atanh(u / (2 + u)) by its
//              series where u <= 0.25; log(x + sqrt((x - 1) (x + 1))) where u > 0.25 and x <= 1e8; log(x) + log(2)
//              where x > 1e8; NaN where x < 1
//   atanh(x) = sign(x) log((1 + s) / (1 - s)) / 2, s = |x|, where s > 0.125; the odd Taylor series to x^17 where
//              s <= 0.125; NaN where s > 1
//
// Derivatives (from the operand): cosh(x), sinh(x), 1 / sqrt(1 + x^2), 1 / sqrt((x - 1) (x + 1)), 1 / (1 - x^2).
//
// -----------------------------------------------------------------------------------------------------------------------------

#include "stan_external_scalar_functions.hpp"

#include <cmath>
#include <limits>
#include <ostream>

namespace nico::math {

static constexpr double HYPERBOLIC_LN2 = 0.693147180559945309417232121458;

// 1 / 15!, 1 / 13!, ..., 1 / 1! (the odd Taylor series of sinh, in x^2)
static constexpr double sinh_odd_inverse_factorials[8] = { 1.0 / 1307674368000.0, 1.0 / 6227020800.0,
                                                           1.0 / 39916800.0, 1.0 / 362880.0, 1.0 / 5040.0,
                                                           1.0 / 120.0, 1.0 / 6.0, 1.0 };
// (-1)^n (2n - 1)!! / ((2n)!! (2n + 1)), n = 8, ..., 0 (the odd Taylor series of asinh, in x^2)
static constexpr double asinh_odd_coefficients[9] = { 6435.0 / 557056.0, -143.0 / 10240.0, 231.0 / 13312.0,
                                                      -63.0 / 2816.0, 35.0 / 1152.0, -5.0 / 112.0, 3.0 / 40.0,
                                                      -1.0 / 6.0, 1.0 };
// 1 / (2n + 1), n = 8, ..., 0 (the odd Taylor series of atanh, in x^2)
static constexpr double atanh_odd_coefficients[9] = { 1.0 / 17.0, 1.0 / 15.0, 1.0 / 13.0, 1.0 / 11.0, 1.0 / 9.0,
                                                      1.0 / 7.0, 1.0 / 5.0, 1.0 / 3.0, 1.0 };

// ---- the value of each function at one double (the scalar overloads, and the scalar-kernel build)
inline double nico_sinh_value(const double x) {

    const double s = std::abs(x);
    double value;
    if (s <= 0.5) {
        const double squared = s * s;
        double series = sinh_odd_inverse_factorials[0];
        for (int k = 1; k < 8; ++k) series = std::fma(series, squared, sinh_odd_inverse_factorials[k]);
        value = series * s;
    } else if (s <= 709.0) {
        const double e = nico_exp_1(s);
        value = 0.5 * e - 0.5 / e;
    } else {
        value = nico_exp_1(s - HYPERBOLIC_LN2);
    }
    return std::copysign(value, x);

}

inline double nico_cosh_value(const double x) {

    const double s = std::abs(x);
    if (s > 709.0) return nico_exp_1(s - HYPERBOLIC_LN2);
    const double e = nico_exp_1(s);
    return 0.5 * e + 0.5 / e;

}

inline double nico_asinh_value(const double x) {

    const double s = std::abs(x);
    double value;
    if (s <= 0.125) {
        const double squared = s * s;
        double series = asinh_odd_coefficients[0];
        for (int k = 1; k < 9; ++k) series = std::fma(series, squared, asinh_odd_coefficients[k]);
        value = series * s;
    } else if (s <= 1e8) {
        value = nico_log_1_double_precision(s + std::sqrt(std::fma(s, s, 1.0)));
    } else {
        value = nico_log_1_double_precision(s) + HYPERBOLIC_LN2;
    }
    return std::copysign(value, x);

}

// log1p(u) = 2 atanh(w), w = u / (2 + u), by the odd series to w^17, for |u| <= 0.25 (|w| <= 1 / 7; the first term
// left out is below 3e-17 of the value)
inline double nico_log1p_by_atanh_series_value(const double u) {

    const double w = u / (2.0 + u);
    const double squared = w * w;
    double series = atanh_odd_coefficients[0];
    for (int k = 1; k < 9; ++k) series = std::fma(series, squared, atanh_odd_coefficients[k]);
    return 2.0 * series * w;

}

inline double nico_acosh_value(const double x) {

    if (!(x >= 1.0)) return std::numeric_limits<double>::quiet_NaN();
    if (x > 1e8) return nico_log_1_double_precision(x) + HYPERBOLIC_LN2;
    const double t = x - 1.0;
    const double root = std::sqrt(t * (x + 1.0));
    const double u = t + root;
    if (u <= 0.25) return nico_log1p_by_atanh_series_value(u);
    return nico_log_1_double_precision(x + root);

}

inline double nico_atanh_value(const double x) {

    const double s = std::abs(x);
    if (s > 1.0 || std::isnan(x)) return std::numeric_limits<double>::quiet_NaN();
    double value;
    if (s <= 0.125) {
        const double squared = s * s;
        double series = atanh_odd_coefficients[0];
        for (int k = 1; k < 9; ++k) series = std::fma(series, squared, atanh_odd_coefficients[k]);
        value = series * s;
    } else {
        value = 0.5 * nico_log_1_double_precision((1.0 + s) / (1.0 - s));
    }
    return std::copysign(value, x);

}

#if !defined(NICO_SIMD_SCALAR_KERNELS)
// ---- the SIMD kernels: the same branches as the scalar values above, blended lane by lane
#if BMVP_LANES == 8
inline simd_t simd_odd_series_of(const simd_t s,
                                 const double* coefficients,
                                 const int n_coefficients) {

    ALIGN64  __m512d const squared = _mm512_mul_pd(s, s);
    ALIGN64  __m512d series = _mm512_set1_pd(coefficients[0]);
    for (int k = 1; k < n_coefficients; ++k) series = _mm512_fmadd_pd(series, squared, _mm512_set1_pd(coefficients[k]));
    return _mm512_mul_pd(series, s);

}

inline simd_t simd_sinh_of(const simd_t x) {

    ALIGN64  __m512d const s = _mm512_abs_pd(x);
    ALIGN64  __m512d const series = simd_odd_series_of(s, sinh_odd_inverse_factorials, 8);
    const __mmask8 small = _mm512_cmp_pd_mask(s, _mm512_set1_pd(0.5), _CMP_LE_OQ);
    ALIGN64  __m512d value = series;
    if (small != 0xff) {
        const __mmask8 large = _mm512_cmp_pd_mask(s, _mm512_set1_pd(709.0), _CMP_GT_OQ);
        ALIGN64  __m512d const e = nico_exp_1_AVX512(_mm512_mask_blend_pd(large, s,
                                                         _mm512_sub_pd(s, _mm512_set1_pd(HYPERBOLIC_LN2))));
        ALIGN64  __m512d const from_exp = _mm512_mask_blend_pd(large,
            _mm512_sub_pd(_mm512_mul_pd(_mm512_set1_pd(0.5), e), _mm512_div_pd(_mm512_set1_pd(0.5), e)), e);
        value = _mm512_mask_blend_pd(small, from_exp, series);
    }
    return _mm512_or_pd(value, _mm512_and_pd(x, _mm512_set1_pd(-0.0)));

}

inline simd_t simd_cosh_of(const simd_t x) {

    ALIGN64  __m512d const s = _mm512_abs_pd(x);
    const __mmask8 large = _mm512_cmp_pd_mask(s, _mm512_set1_pd(709.0), _CMP_GT_OQ);
    ALIGN64  __m512d const e = nico_exp_1_AVX512(_mm512_mask_blend_pd(large, s,
                                                     _mm512_sub_pd(s, _mm512_set1_pd(HYPERBOLIC_LN2))));
    return _mm512_mask_blend_pd(large,
        _mm512_add_pd(_mm512_mul_pd(_mm512_set1_pd(0.5), e), _mm512_div_pd(_mm512_set1_pd(0.5), e)), e);

}

inline simd_t simd_asinh_of(const simd_t x) {

    ALIGN64  __m512d const s = _mm512_abs_pd(x);
    ALIGN64  __m512d const series = simd_odd_series_of(s, asinh_odd_coefficients, 9);
    const __mmask8 small = _mm512_cmp_pd_mask(s, _mm512_set1_pd(0.125), _CMP_LE_OQ);
    ALIGN64  __m512d value = series;
    if (small != 0xff) {
        const __mmask8 large = _mm512_cmp_pd_mask(s, _mm512_set1_pd(1e8), _CMP_GT_OQ);
        ALIGN64  __m512d const argument = _mm512_mask_blend_pd(large,
            _mm512_add_pd(s, _mm512_sqrt_pd(_mm512_fmadd_pd(s, s, _mm512_set1_pd(1.0)))), s);
        ALIGN64  __m512d const logarithm = nico_log_1_double_precision_AVX512(argument);
        ALIGN64  __m512d const from_log = _mm512_mask_blend_pd(large, logarithm,
                                                  _mm512_add_pd(logarithm, _mm512_set1_pd(HYPERBOLIC_LN2)));
        value = _mm512_mask_blend_pd(small, from_log, series);
    }
    return _mm512_or_pd(value, _mm512_and_pd(x, _mm512_set1_pd(-0.0)));

}

inline simd_t simd_acosh_of(const simd_t x) {

    ALIGN64  __m512d const one = _mm512_set1_pd(1.0);
    ALIGN64  __m512d const t = _mm512_sub_pd(x, one);
    ALIGN64  __m512d const root = _mm512_sqrt_pd(_mm512_mul_pd(t, _mm512_add_pd(x, one)));
    ALIGN64  __m512d const u = _mm512_add_pd(t, root);
    ALIGN64  __m512d const w = _mm512_div_pd(u, _mm512_add_pd(_mm512_set1_pd(2.0), u));
    ALIGN64  __m512d const from_series = _mm512_mul_pd(_mm512_set1_pd(2.0),
                                                       simd_odd_series_of(w, atanh_odd_coefficients, 9));
    const __mmask8 small = _mm512_cmp_pd_mask(u, _mm512_set1_pd(0.25), _CMP_LE_OQ);
    const __mmask8 large = _mm512_cmp_pd_mask(x, _mm512_set1_pd(1e8), _CMP_GT_OQ);
    ALIGN64  __m512d value = from_series;
    if (small != 0xff) {
        ALIGN64  __m512d const logarithm = nico_log_1_double_precision_AVX512(
            _mm512_mask_blend_pd(large, _mm512_add_pd(x, root), x));
        ALIGN64  __m512d const from_log = _mm512_mask_blend_pd(large, logarithm,
                                                  _mm512_add_pd(logarithm, _mm512_set1_pd(HYPERBOLIC_LN2)));
        value = _mm512_mask_blend_pd(small, from_log, from_series);
    }
    const __mmask8 in_domain = _mm512_cmp_pd_mask(x, one, _CMP_GE_OQ);
    return _mm512_mask_blend_pd(in_domain, _mm512_set1_pd(std::numeric_limits<double>::quiet_NaN()), value);

}

inline simd_t simd_atanh_of(const simd_t x) {

    ALIGN64  __m512d const one = _mm512_set1_pd(1.0);
    ALIGN64  __m512d const s = _mm512_abs_pd(x);
    ALIGN64  __m512d const series = simd_odd_series_of(s, atanh_odd_coefficients, 9);
    const __mmask8 small = _mm512_cmp_pd_mask(s, _mm512_set1_pd(0.125), _CMP_LE_OQ);
    ALIGN64  __m512d value = series;
    if (small != 0xff) {
        ALIGN64  __m512d const from_log = _mm512_mul_pd(_mm512_set1_pd(0.5), nico_log_1_double_precision_AVX512(
            _mm512_div_pd(_mm512_add_pd(one, s), _mm512_sub_pd(one, s))));
        value = _mm512_mask_blend_pd(small, from_log, series);
    }
    const __mmask8 in_domain = _mm512_cmp_pd_mask(s, one, _CMP_LE_OQ);
    return _mm512_mask_blend_pd(in_domain, _mm512_set1_pd(std::numeric_limits<double>::quiet_NaN()),
                                _mm512_or_pd(value, _mm512_and_pd(x, _mm512_set1_pd(-0.0))));

}
#else
inline simd_t simd_odd_series_of(const simd_t s,
                                 const double* coefficients,
                                 const int n_coefficients) {

    ALIGN32  __m256d const squared = _mm256_mul_pd(s, s);
    ALIGN32  __m256d series = _mm256_set1_pd(coefficients[0]);
    for (int k = 1; k < n_coefficients; ++k) series = _mm256_fmadd_pd(series, squared, _mm256_set1_pd(coefficients[k]));
    return _mm256_mul_pd(series, s);

}

inline simd_t simd_sinh_of(const simd_t x) {

    ALIGN32  __m256d const s = _mm256_andnot_pd(_mm256_set1_pd(-0.0), x);
    ALIGN32  __m256d const series = simd_odd_series_of(s, sinh_odd_inverse_factorials, 8);
    ALIGN32  __m256d const small = _mm256_cmp_pd(s, _mm256_set1_pd(0.5), _CMP_LE_OQ);
    ALIGN32  __m256d value = series;
    if (_mm256_movemask_pd(small) != 0xf) {
        ALIGN32  __m256d const large = _mm256_cmp_pd(s, _mm256_set1_pd(709.0), _CMP_GT_OQ);
        ALIGN32  __m256d const e = nico_exp_1_AVX2(_mm256_blendv_pd(s,
                                                       _mm256_sub_pd(s, _mm256_set1_pd(HYPERBOLIC_LN2)), large));
        ALIGN32  __m256d const from_exp = _mm256_blendv_pd(
            _mm256_sub_pd(_mm256_mul_pd(_mm256_set1_pd(0.5), e), _mm256_div_pd(_mm256_set1_pd(0.5), e)), e, large);
        value = _mm256_blendv_pd(from_exp, series, small);
    }
    return _mm256_or_pd(value, _mm256_and_pd(x, _mm256_set1_pd(-0.0)));

}

inline simd_t simd_cosh_of(const simd_t x) {

    ALIGN32  __m256d const s = _mm256_andnot_pd(_mm256_set1_pd(-0.0), x);
    ALIGN32  __m256d const large = _mm256_cmp_pd(s, _mm256_set1_pd(709.0), _CMP_GT_OQ);
    ALIGN32  __m256d const e = nico_exp_1_AVX2(_mm256_blendv_pd(s,
                                                   _mm256_sub_pd(s, _mm256_set1_pd(HYPERBOLIC_LN2)), large));
    return _mm256_blendv_pd(
        _mm256_add_pd(_mm256_mul_pd(_mm256_set1_pd(0.5), e), _mm256_div_pd(_mm256_set1_pd(0.5), e)), e, large);

}

inline simd_t simd_asinh_of(const simd_t x) {

    ALIGN32  __m256d const s = _mm256_andnot_pd(_mm256_set1_pd(-0.0), x);
    ALIGN32  __m256d const series = simd_odd_series_of(s, asinh_odd_coefficients, 9);
    ALIGN32  __m256d const small = _mm256_cmp_pd(s, _mm256_set1_pd(0.125), _CMP_LE_OQ);
    ALIGN32  __m256d value = series;
    if (_mm256_movemask_pd(small) != 0xf) {
        ALIGN32  __m256d const large = _mm256_cmp_pd(s, _mm256_set1_pd(1e8), _CMP_GT_OQ);
        ALIGN32  __m256d const argument = _mm256_blendv_pd(
            _mm256_add_pd(s, _mm256_sqrt_pd(_mm256_fmadd_pd(s, s, _mm256_set1_pd(1.0)))), s, large);
        ALIGN32  __m256d const logarithm = nico_log_1_double_precision_AVX2(argument);
        ALIGN32  __m256d const from_log = _mm256_blendv_pd(logarithm,
                                              _mm256_add_pd(logarithm, _mm256_set1_pd(HYPERBOLIC_LN2)), large);
        value = _mm256_blendv_pd(from_log, series, small);
    }
    return _mm256_or_pd(value, _mm256_and_pd(x, _mm256_set1_pd(-0.0)));

}

inline simd_t simd_acosh_of(const simd_t x) {

    ALIGN32  __m256d const one = _mm256_set1_pd(1.0);
    ALIGN32  __m256d const t = _mm256_sub_pd(x, one);
    ALIGN32  __m256d const root = _mm256_sqrt_pd(_mm256_mul_pd(t, _mm256_add_pd(x, one)));
    ALIGN32  __m256d const u = _mm256_add_pd(t, root);
    ALIGN32  __m256d const w = _mm256_div_pd(u, _mm256_add_pd(_mm256_set1_pd(2.0), u));
    ALIGN32  __m256d const from_series = _mm256_mul_pd(_mm256_set1_pd(2.0),
                                                       simd_odd_series_of(w, atanh_odd_coefficients, 9));
    ALIGN32  __m256d const small = _mm256_cmp_pd(u, _mm256_set1_pd(0.25), _CMP_LE_OQ);
    ALIGN32  __m256d const large = _mm256_cmp_pd(x, _mm256_set1_pd(1e8), _CMP_GT_OQ);
    ALIGN32  __m256d value = from_series;
    if (_mm256_movemask_pd(small) != 0xf) {
        ALIGN32  __m256d const logarithm = nico_log_1_double_precision_AVX2(
            _mm256_blendv_pd(_mm256_add_pd(x, root), x, large));
        ALIGN32  __m256d const from_log = _mm256_blendv_pd(logarithm,
                                              _mm256_add_pd(logarithm, _mm256_set1_pd(HYPERBOLIC_LN2)), large);
        value = _mm256_blendv_pd(from_log, from_series, small);
    }
    ALIGN32  __m256d const in_domain = _mm256_cmp_pd(x, one, _CMP_GE_OQ);
    return _mm256_blendv_pd(_mm256_set1_pd(std::numeric_limits<double>::quiet_NaN()), value, in_domain);

}

inline simd_t simd_atanh_of(const simd_t x) {

    ALIGN32  __m256d const one = _mm256_set1_pd(1.0);
    ALIGN32  __m256d const s = _mm256_andnot_pd(_mm256_set1_pd(-0.0), x);
    ALIGN32  __m256d const series = simd_odd_series_of(s, atanh_odd_coefficients, 9);
    ALIGN32  __m256d const small = _mm256_cmp_pd(s, _mm256_set1_pd(0.125), _CMP_LE_OQ);
    ALIGN32  __m256d value = series;
    if (_mm256_movemask_pd(small) != 0xf) {
        ALIGN32  __m256d const from_log = _mm256_mul_pd(_mm256_set1_pd(0.5), nico_log_1_double_precision_AVX2(
            _mm256_div_pd(_mm256_add_pd(one, s), _mm256_sub_pd(one, s))));
        value = _mm256_blendv_pd(from_log, series, small);
    }
    ALIGN32  __m256d const in_domain = _mm256_cmp_pd(s, one, _CMP_LE_OQ);
    return _mm256_blendv_pd(_mm256_set1_pd(std::numeric_limits<double>::quiet_NaN()),
                            _mm256_or_pd(value, _mm256_and_pd(x, _mm256_set1_pd(-0.0))), in_domain);

}
#endif

inline void k_sinh(double* p, int n)  { apply_kernel(p, n, [](const simd_t v) { return simd_sinh_of(v); }); }
inline void k_cosh(double* p, int n)  { apply_kernel(p, n, [](const simd_t v) { return simd_cosh_of(v); }); }
inline void k_asinh(double* p, int n) { apply_kernel(p, n, [](const simd_t v) { return simd_asinh_of(v); }); }
inline void k_acosh(double* p, int n) { apply_kernel(p, n, [](const simd_t v) { return simd_acosh_of(v); }); }
inline void k_atanh(double* p, int n) { apply_kernel(p, n, [](const simd_t v) { return simd_atanh_of(v); }); }
#else
inline void k_sinh(double* p, int n)  { for (int i = 0; i < n; ++i) p[i] = nico_sinh_value(p[i]); }
inline void k_cosh(double* p, int n)  { for (int i = 0; i < n; ++i) p[i] = nico_cosh_value(p[i]); }
inline void k_asinh(double* p, int n) { for (int i = 0; i < n; ++i) p[i] = nico_asinh_value(p[i]); }
inline void k_acosh(double* p, int n) { for (int i = 0; i < n; ++i) p[i] = nico_acosh_value(p[i]); }
inline void k_atanh(double* p, int n) { for (int i = 0; i < n; ++i) p[i] = nico_atanh_value(p[i]); }
#endif

// ---- values and derivatives on buffers (the derivative reads the operand only)
inline void v_sinh(const double* x, int n, double* f)  { copy_n(x, n, f); k_sinh(f, n); }
inline void v_cosh(const double* x, int n, double* f)  { copy_n(x, n, f); k_cosh(f, n); }
inline void v_asinh(const double* x, int n, double* f) { copy_n(x, n, f); k_asinh(f, n); }
inline void v_acosh(const double* x, int n, double* f) { copy_n(x, n, f); k_acosh(f, n); }
inline void v_atanh(const double* x, int n, double* f) { copy_n(x, n, f); k_atanh(f, n); }

// sinh: d/dx = cosh(x)
inline void d_sinh(const double* x,
                   const double*,
                   int n,
                   double* d) { copy_n(x, n, d); k_cosh(d, n); }

// cosh: d/dx = sinh(x)
inline void d_cosh(const double* x,
                   const double*,
                   int n,
                   double* d) { copy_n(x, n, d); k_sinh(d, n); }

// asinh: d/dx = 1 / sqrt(1 + x^2) (1 / |x| where |x| > 1e8, where x^2 + 1 = x^2 and x^2 may overflow)
inline void d_asinh(const double* x,
                    const double*,
                    int n,
                    double* d) {

    for (int i = 0; i < n; ++i) {
        const double s = std::abs(x[i]);
        d[i] = (s > 1e8) ? 1.0 / s : 1.0 / std::sqrt(std::fma(s, s, 1.0));
    }

}

// acosh: d/dx = 1 / sqrt((x - 1) (x + 1)) (1 / x where x > 1e8)
inline void d_acosh(const double* x,
                    const double*,
                    int n,
                    double* d) {

    for (int i = 0; i < n; ++i) {
        d[i] = (x[i] > 1e8) ? 1.0 / x[i] : 1.0 / std::sqrt((x[i] - 1.0) * (x[i] + 1.0));
    }

}

// atanh: d/dx = 1 / ((1 - x) (1 + x))
inline void d_atanh(const double* x,
                    const double*,
                    int n,
                    double* d) {

    for (int i = 0; i < n; ++i) d[i] = 1.0 / ((1.0 - x[i]) * (1.0 + x[i]));

}

}  // namespace nico::math

// ---- the Stan-facing overloads: real (double, var), vector (double, var), row_vector and matrix
#define NICO_HYPERBOLIC_EXTERNAL(NAME, STEM)                                                                           \
namespace nico::math {                                                                                                 \
inline double NAME(const double x,                                                                                     \
                   std::ostream* /*pstream__*/) {                                                                      \
                                                                                                                       \
    return nico_##STEM##_value(x);                                                                                     \
                                                                                                                       \
}                                                                                                                      \
inline var NAME(const var& x,                                                                                          \
                std::ostream* /*pstream__*/) {                                                                         \
                                                                                                                       \
    return nico_apply_scalar_kernel(x, &nico_##STEM##_value, &d_##STEM);                                               \
                                                                                                                       \
}                                                                                                                      \
template <typename T, stan::require_eigen_col_vector_vt<std::is_arithmetic, T>* = nullptr>                            \
inline Eigen::Matrix<double, -1, 1> NAME(const T& x,                                                                   \
                                         std::ostream* /*pstream__*/) {                                                \
                                                                                                                       \
    return prim_apply(x, [](const double* a, int n, double* f) { v_##STEM(a, n, f); },                                 \
                      [](double* values, int n) { k_##STEM(values, n); });                                             \
                                                                                                                       \
}                                                                                                                      \
template <typename T, stan::require_eigen_col_vector_vt<stan::is_var, T>* = nullptr>                                  \
inline Eigen::Matrix<var, -1, 1> NAME(const T& x,                                                                      \
                                      std::ostream* /*pstream__*/) {                                                   \
                                                                                                                       \
    return faster_rev::rev_apply_strip_mined<derivative_values::operand>(                                              \
               x, [](const double* a, int n, double* f) { v_##STEM(a, n, f); },                                        \
               [](double* values, int n) { k_##STEM(values, n); },                                                     \
               [](const double* a, const double* f, int n, double* d) { d_##STEM(a, f, n, d); });                      \
                                                                                                                       \
}                                                                                                                      \
template <typename T, stan::require_eigen_t<T>* = nullptr, stan::require_not_eigen_col_vector_t<T>* = nullptr>        \
inline Eigen::Matrix<stan::value_type_t<T>, std::decay_t<T>::RowsAtCompileTime,                                        \
                     std::decay_t<T>::ColsAtCompileTime>                                                               \
NAME(const T& x,                                                                                                       \
     std::ostream* pstream__) {                                                                                        \
                                                                                                                       \
    return apply_column_vector_function_to_coefficients(                                                               \
               x, [pstream__](const auto& x_as_column) { return NAME(x_as_column, pstream__); });                      \
                                                                                                                       \
}                                                                                                                      \
}                                                                                                                      \
using nico::math::NAME;

NICO_HYPERBOLIC_EXTERNAL(nico_sinh, sinh)
NICO_HYPERBOLIC_EXTERNAL(nico_cosh, cosh)
NICO_HYPERBOLIC_EXTERNAL(nico_asinh, asinh)
NICO_HYPERBOLIC_EXTERNAL(nico_acosh, acosh)
NICO_HYPERBOLIC_EXTERNAL(nico_atanh, atanh)

#undef NICO_HYPERBOLIC_EXTERNAL
