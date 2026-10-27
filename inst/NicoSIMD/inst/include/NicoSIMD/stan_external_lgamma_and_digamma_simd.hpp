#pragma once
//
// ---- lgamma and digamma in one SIMD pass (AVX-512: 8 lanes, AVX2: 4 lanes), for x > 0 and finite.
//
//      x is shifted up to x_s >= 10 with the product p = x (x + 1) ... (x_s - 1) and its derivative dp
//      (10 factors, x + k where x + k < 10 and 1 elsewhere, joined as a tree; no division), then
//          lgamma(x)  = (x_s - 1/2) log(x_s) - x_s + log(2 pi) / 2
//                       + sum_k B_2k / (2k (2k - 1) x_s^(2k - 1)) - log(p),
//          digamma(x) = log(x_s) - 1 / (2 x_s) - sum_k B_2k / (2k x_s^(2k)) - dp / p,
//      the series to x_s^-13 and x_s^-14 (the first term left out is below 5e-17 at x_s >= 10).
//      Two logs (NicoSIMD's double-precision log), one 1 / x_s and one dp / p per vector.
//      A vector with a lane outside 1e-300 < x < 1e300 (0, negative, inf, NaN, subnormal), and the values after
//      the last full vector, go to the scalar lgamma_r / Stan's digamma.
//
//      k_lgamma(p, n):                          lgamma of n values, in place
//      k_lgamma_and_digamma<true>(x, n, lg, dg): lgamma and digamma of n values
//      k_lgamma_and_digamma<false>(x, n, lg, dg): lgamma only (dg not written)
//
#include "stan_external_functions.hpp"
#include "math/nico_double_exp_log.hpp"

#include <cstdlib>
#include <memory>
#include <new>
#include <vector>

namespace nico::math {

static constexpr double LGDG_HALF_LOG_2PI = 0.918938533204672741780329736406;
static constexpr double LGDG_SHIFT_TO = 10.0;
static constexpr double LGDG_SMALLEST = 1e-300;
static constexpr double LGDG_LARGEST = 1e300;

#if !defined(NICO_SIMD_SCALAR_KERNELS)

// ---- the few SIMD operations the pass needs, for 8 or 4 lanes
#if BMVP_LANES == 8
using lgdg_mask_t = __mmask8;
ALWAYS_INLINE simd_t lgdg_set1(const double a) { return _mm512_set1_pd(a); }
ALWAYS_INLINE simd_t lgdg_add(const simd_t a, const simd_t b) { return _mm512_add_pd(a, b); }
ALWAYS_INLINE simd_t lgdg_sub(const simd_t a, const simd_t b) { return _mm512_sub_pd(a, b); }
ALWAYS_INLINE simd_t lgdg_mul(const simd_t a, const simd_t b) { return _mm512_mul_pd(a, b); }
ALWAYS_INLINE simd_t lgdg_div(const simd_t a, const simd_t b) { return _mm512_div_pd(a, b); }
ALWAYS_INLINE simd_t lgdg_fma(const simd_t a, const simd_t b, const simd_t c) { return _mm512_fmadd_pd(a, b, c); }
ALWAYS_INLINE lgdg_mask_t lgdg_lt(const simd_t a, const simd_t b) { return _mm512_cmp_pd_mask(a, b, _CMP_LT_OQ); }
ALWAYS_INLINE bool lgdg_none(const lgdg_mask_t m) { return m == 0; }
ALWAYS_INLINE bool lgdg_all(const lgdg_mask_t m) { return m == 0xff; }
ALWAYS_INLINE simd_t lgdg_select(const lgdg_mask_t m,
                                 const simd_t if_true,
                                 const simd_t if_false) {

    return _mm512_mask_blend_pd(m, if_false, if_true);

}
#else
using lgdg_mask_t = __m256d;
ALWAYS_INLINE simd_t lgdg_set1(const double a) { return _mm256_set1_pd(a); }
ALWAYS_INLINE simd_t lgdg_add(const simd_t a, const simd_t b) { return _mm256_add_pd(a, b); }
ALWAYS_INLINE simd_t lgdg_sub(const simd_t a, const simd_t b) { return _mm256_sub_pd(a, b); }
ALWAYS_INLINE simd_t lgdg_mul(const simd_t a, const simd_t b) { return _mm256_mul_pd(a, b); }
ALWAYS_INLINE simd_t lgdg_div(const simd_t a, const simd_t b) { return _mm256_div_pd(a, b); }
ALWAYS_INLINE simd_t lgdg_fma(const simd_t a, const simd_t b, const simd_t c) { return _mm256_fmadd_pd(a, b, c); }
ALWAYS_INLINE lgdg_mask_t lgdg_lt(const simd_t a, const simd_t b) { return _mm256_cmp_pd(a, b, _CMP_LT_OQ); }
ALWAYS_INLINE bool lgdg_none(const lgdg_mask_t m) { return _mm256_movemask_pd(m) == 0; }
ALWAYS_INLINE bool lgdg_all(const lgdg_mask_t m) { return _mm256_movemask_pd(m) == 0xf; }
ALWAYS_INLINE simd_t lgdg_select(const lgdg_mask_t m,
                                 const simd_t if_true,
                                 const simd_t if_false) {

    return _mm256_blendv_pd(if_false, if_true, m);

}
#endif

// ---- one vector of x (every lane positive, normal and finite)
template <bool with_digamma>
ALWAYS_INLINE void lgamma_and_digamma_block(const simd_t x,
                                            simd_t& lgamma_of_x,
                                            simd_t& digamma_of_x) {

    const simd_t shift_to = lgdg_set1(LGDG_SHIFT_TO);
    const simd_t one = lgdg_set1(1.0);
    const simd_t zero = lgdg_set1(0.0);
    // simd_t x_s = x;
    // simd_t p = one;
    // simd_t dp = lgdg_set1(0.0);
    // for (int step = 0; step < 10; ++step) {
    //     const lgdg_mask_t below = lgdg_lt(x_s, shift_to);
    //     if (lgdg_none(below)) break;
    //     if constexpr (with_digamma) dp = lgdg_select(below, lgdg_fma(dp, x_s, p), dp);
    //     p = lgdg_select(below, lgdg_mul(p, x_s), p);
    //     x_s = lgdg_select(below, lgdg_add(x_s, one), x_s);
    // }
    //// (the same product as a tree, 4 levels deep in place of a chain of 10 steps: factor k is x + k where
    ////  x + k < 10, else 1; a pair (p_a, dp_a), (p_b, dp_b) joins as (p_a p_b, dp_a p_b + p_a dp_b))
    simd_t x_s = x;
    simd_t p = one;
    simd_t dp = zero;
    if (!lgdg_none(lgdg_lt(x, shift_to))) {
        simd_t factor[10], d_factor[10];
        for (int k = 0; k < 10; ++k) {
            const simd_t x_plus_k = lgdg_add(x, lgdg_set1(static_cast<double>(k)));
            const lgdg_mask_t below = lgdg_lt(x_plus_k, shift_to);
            factor[k] = lgdg_select(below, x_plus_k, one);
            d_factor[k] = lgdg_select(below, one, zero);
        }
        simd_t number_of_shifts = d_factor[0];
        for (int k = 1; k < 10; ++k) number_of_shifts = lgdg_add(number_of_shifts, d_factor[k]);
        x_s = lgdg_add(x, number_of_shifts);
        auto join = [&](const int a, const int b) {
            if constexpr (with_digamma) {
                d_factor[a] = lgdg_fma(d_factor[a], factor[b], lgdg_mul(factor[a], d_factor[b]));
            }
            factor[a] = lgdg_mul(factor[a], factor[b]);
        };
        join(0, 1); join(2, 3); join(4, 5); join(6, 7); join(8, 9);
        join(0, 2); join(4, 6);
        join(0, 4);
        join(0, 8);
        p = factor[0];
        if constexpr (with_digamma) dp = d_factor[0];
    }
    const simd_t log_x_s = BMVP_KERNEL_NAME(nico_log_1_double_precision_wo_checks)(x_s);
    const simd_t log_p = BMVP_KERNEL_NAME(nico_log_1_double_precision_wo_checks)(p);
    const simd_t z = lgdg_div(one, x_s);
    const simd_t w = lgdg_mul(z, z);

    // sum_k B_2k / (2k (2k - 1) x_s^(2k - 1)), k = 1, ..., 7
    simd_t series = lgdg_set1(1.0 / 156.0);
    series = lgdg_fma(series, w, lgdg_set1(-691.0 / 360360.0));
    series = lgdg_fma(series, w, lgdg_set1(1.0 / 1188.0));
    series = lgdg_fma(series, w, lgdg_set1(-1.0 / 1680.0));
    series = lgdg_fma(series, w, lgdg_set1(1.0 / 1260.0));
    series = lgdg_fma(series, w, lgdg_set1(-1.0 / 360.0));
    series = lgdg_fma(series, w, lgdg_set1(1.0 / 12.0));
    series = lgdg_mul(series, z);
    lgamma_of_x = lgdg_sub(lgdg_add(lgdg_sub(lgdg_mul(lgdg_sub(x_s, lgdg_set1(0.5)), log_x_s), x_s),
                                    lgdg_add(lgdg_set1(LGDG_HALF_LOG_2PI), series)),
                           log_p);

    if constexpr (with_digamma) {
        // sum_k B_2k / (2k x_s^(2k)), k = 1, ..., 7
        simd_t series_of_digamma = lgdg_set1(1.0 / 12.0);
        series_of_digamma = lgdg_fma(series_of_digamma, w, lgdg_set1(-691.0 / 32760.0));
        series_of_digamma = lgdg_fma(series_of_digamma, w, lgdg_set1(1.0 / 132.0));
        series_of_digamma = lgdg_fma(series_of_digamma, w, lgdg_set1(-1.0 / 240.0));
        series_of_digamma = lgdg_fma(series_of_digamma, w, lgdg_set1(1.0 / 252.0));
        series_of_digamma = lgdg_fma(series_of_digamma, w, lgdg_set1(-1.0 / 120.0));
        series_of_digamma = lgdg_fma(series_of_digamma, w, lgdg_set1(1.0 / 12.0));
        series_of_digamma = lgdg_mul(series_of_digamma, w);
        digamma_of_x = lgdg_sub(lgdg_sub(log_x_s, lgdg_fma(lgdg_set1(0.5), z, series_of_digamma)),
                                lgdg_div(dp, p));
    }

}

#endif

// ---- n values: full vectors in SIMD; the values after the last full vector (all of them when n is below the
//      SIMD width) in one padded vector or with the scalar lgamma_r / Stan's digamma (the rule below); a vector
//      with a lane outside the fast range with the scalar functions
inline void lgamma_and_digamma_scalar(const double x,
                                      double* lgamma_of_x,
                                      double* digamma_of_x) {

    int sign = 1;
    *lgamma_of_x = ::lgamma_r(x, &sign);
    if (digamma_of_x != nullptr) *digamma_of_x = stan::math::digamma(x);

}

// (the last values, fewer than the SIMD width: one padded vector when there are at least tail_in_simd_from of
//  them, else the scalar functions; the two thresholds below are the measured crossings, local HPC, AVX-512)
static constexpr int LGDG_TAIL_IN_SIMD_FROM_WITH_DIGAMMA = 4;
static constexpr int LGDG_TAIL_IN_SIMD_FROM_LGAMMA_ONLY = 6;

template <bool with_digamma>
inline void k_lgamma_and_digamma_with_tail_rule(const double* x,
                                                const int n,
                                                double* lgamma_of_x,
                                                double* digamma_of_x,
                                                const int tail_in_simd_from) {

    int first = 0;
#if !defined(NICO_SIMD_SCALAR_KERNELS)
    // (x may be the lgamma output itself: each vector of x is loaded before its results are stored)
    const simd_t smallest = lgdg_set1(LGDG_SMALLEST);
    const simd_t largest = lgdg_set1(LGDG_LARGEST);
    for (; first + BMVP_LANES <= n; first += BMVP_LANES) {
        const simd_t x_block = simd_loadu(x + first);
        const bool all_inside = lgdg_all(lgdg_lt(smallest, x_block)) && lgdg_all(lgdg_lt(x_block, largest));
        if (!all_inside) {
            for (int i = first; i < first + BMVP_LANES; ++i) {
                lgamma_and_digamma_scalar(x[i], lgamma_of_x + i, with_digamma ? digamma_of_x + i : nullptr);
            }
            continue;
        }
        simd_t lgamma_block, digamma_block;
        lgamma_and_digamma_block<with_digamma>(x_block, lgamma_block, digamma_block);
        simd_storeu(lgamma_of_x + first, lgamma_block);
        if constexpr (with_digamma) simd_storeu(digamma_of_x + first, digamma_block);
    }
    const int count = n - first;
    if (count > 0 && count >= tail_in_simd_from) {
        alignas(64) double lanes[BMVP_LANES];
        bool all_inside = true;
        for (int lane = 0; lane < BMVP_LANES; ++lane) {
            lanes[lane] = (lane < count) ? x[first + lane] : 1.0;
            all_inside &= lanes[lane] > LGDG_SMALLEST && lanes[lane] < LGDG_LARGEST;
        }
        if (all_inside) {
            simd_t lgamma_block, digamma_block;
            lgamma_and_digamma_block<with_digamma>(simd_load(lanes), lgamma_block, digamma_block);
            simd_store(lanes, lgamma_block);
            for (int lane = 0; lane < count; ++lane) lgamma_of_x[first + lane] = lanes[lane];
            if constexpr (with_digamma) {
                simd_store(lanes, digamma_block);
                for (int lane = 0; lane < count; ++lane) digamma_of_x[first + lane] = lanes[lane];
            }
            return;
        }
    }
#endif
    for (int i = first; i < n; ++i) {
        lgamma_and_digamma_scalar(x[i], lgamma_of_x + i, with_digamma ? digamma_of_x + i : nullptr);
    }

}

template <bool with_digamma>
inline void k_lgamma_and_digamma(const double* x,
                                 const int n,
                                 double* lgamma_of_x,
                                 double* digamma_of_x) {

    k_lgamma_and_digamma_with_tail_rule<with_digamma>(x, n, lgamma_of_x, digamma_of_x,
                                                      with_digamma ? LGDG_TAIL_IN_SIMD_FROM_WITH_DIGAMMA
                                                                   : LGDG_TAIL_IN_SIMD_FROM_LGAMMA_ONLY);

}

inline void k_lgamma(double* p,
                     const int n) {

    k_lgamma_and_digamma<false>(p, n, p, nullptr);

}

// ---- log(x) and log1m(y) = log1p(-y) of n values into out: full vectors in SIMD, the last values (fewer than
//      the SIMD width) with the scalar double-precision kernels, with no buffer
inline void k_log_into(const double* x,
                       const int n,
                       double* out) {

    int first = 0;
#if !defined(NICO_SIMD_SCALAR_KERNELS)
    for (; first + BMVP_LANES <= n; first += BMVP_LANES) {
        simd_storeu(out + first, BMVP_KERNEL_NAME(nico_log_1_double_precision)(simd_loadu(x + first)));
    }
#endif
    for (int i = first; i < n; ++i) out[i] = nico_log_1_double_precision(x[i]);

}

inline void k_log1m_into(const double* y,
                         const int n,
                         double* out) {

    int first = 0;
#if !defined(NICO_SIMD_SCALAR_KERNELS)
    const simd_t zero = lgdg_set1(0.0);
    for (; first + BMVP_LANES <= n; first += BMVP_LANES) {
        simd_storeu(out + first,
                    BMVP_KERNEL_NAME(nico_log1p_1_double_precision)(lgdg_sub(zero, simd_loadu(y + first))));
    }
#endif
    for (int i = first; i < n; ++i) out[i] = nico_log1p_1_double_precision(-y[i]);

}

// ---- log(y) and log1m(y) = log1p(-y) of the same n values in one pass (one load of each vector of y, the two
//      kernels side by side), the last values with the scalar kernels
inline void k_log_and_log1m_into(const double* y,
                                 const int n,
                                 double* log_y,
                                 double* log1m_y) {

    int first = 0;
#if !defined(NICO_SIMD_SCALAR_KERNELS)
    const simd_t zero = lgdg_set1(0.0);
    for (; first + BMVP_LANES <= n; first += BMVP_LANES) {
        const simd_t y_vector = simd_loadu(y + first);
        simd_storeu(log_y + first, BMVP_KERNEL_NAME(nico_log_1_double_precision)(y_vector));
        simd_storeu(log1m_y + first, BMVP_KERNEL_NAME(nico_log1p_1_double_precision)(lgdg_sub(zero, y_vector)));
    }
#endif
    for (int i = first; i < n; ++i) {
        log_y[i] = nico_log_1_double_precision(y[i]);
        log1m_y[i] = nico_log1p_1_double_precision(-y[i]);
    }

}

// ---- for a short vector (at most 16 values in all): the logs of x_1 .. x_n (log_x) and of 1 - y_1 .. 1 - y_m
//      (log1m_y) in the same SIMD vectors, no scalar calls. log1m(y) = log(u) + ((1 - u) - y) / u with u = 1 - y
//      rounded (the rounding error of u to first order, as nico_log1p_1_double_precision); either n or m may be 0
inline void k_log_and_log1m_short(const double* x,
                                  const int n,
                                  const double* y,
                                  const int m,
                                  double* log_x,
                                  double* log1m_y) {

    alignas(64) double arguments[16];
    // alignas(64) double corrections[16];
    // const int total = n + m;
    // for (int i = 0; i < n; ++i) {
    //     arguments[i] = x[i];
    //     corrections[i] = 0.0;
    // }
    // for (int j = 0; j < m; ++j) {
    //     const double u = 1.0 - y[j];
    //     arguments[n + j] = u;
    //     corrections[n + j] = ((1.0 - u) - y[j]) / u;
    // }
    //// (the raw values in one array, 1 in the lanes of log1m(y) and 0 in the lanes of log(x); u = 1 - y and its
    ////  correction made in the SIMD vectors, one division per vector)
    // alignas(64) double is_log1m[16];
    // const int total = n + m;
    // for (int i = 0; i < n; ++i) {
    //     arguments[i] = x[i];
    //     is_log1m[i] = 0.0;
    // }
    // for (int j = 0; j < m; ++j) {
    //     arguments[n + j] = y[j];
    //     is_log1m[n + j] = 1.0;
    // }
    // #if defined(NICO_SIMD_SCALAR_KERNELS)
    // for (int i = 0; i < total; ++i) {
    //     if (is_log1m[i] == 0.0) {
    //         arguments[i] = nico_log_1_double_precision(arguments[i]);
    //     } else {
    //         const double u = 1.0 - arguments[i];
    //         arguments[i] = nico_log_1_double_precision(u) + ((1.0 - u) - arguments[i]) / u;
    //     }
    // }
    // #else
    // const int padded = ((total + BMVP_LANES - 1) / BMVP_LANES) * BMVP_LANES;
    // for (int i = total; i < padded; ++i) {
    //     arguments[i] = 1.0;
    //     is_log1m[i] = 0.0;
    // }
    // const simd_t one = lgdg_set1(1.0);
    // const simd_t half = lgdg_set1(0.5);
    // const simd_t zero = lgdg_set1(0.0);
    // for (int first = 0; first < padded; first += BMVP_LANES) {
    //     const simd_t a = simd_load(arguments + first);
    //     const lgdg_mask_t log1m_lanes = lgdg_lt(half, simd_load(is_log1m + first));
    //     const simd_t u = lgdg_select(log1m_lanes, lgdg_sub(one, a), a);
    //     const simd_t correction = lgdg_select(log1m_lanes, lgdg_div(lgdg_sub(lgdg_sub(one, u), a), u), zero);
    //     const simd_t log_of_u = BMVP_KERNEL_NAME(nico_log_1_double_precision)(u);
    //     simd_store(arguments + first, lgdg_add(log_of_u, correction));
    // }
    // #endif
    //// (the correction of u = 1 - y made in scalar code: at most 8 values, cheaper than a masked vector)
    alignas(64) double corrections[16];
    const int total = n + m;
    for (int i = 0; i < n; ++i) {
        arguments[i] = x[i];
        corrections[i] = 0.0;
    }
    for (int j = 0; j < m; ++j) {
        const double u = 1.0 - y[j];
        arguments[n + j] = u;
        corrections[n + j] = ((1.0 - u) - y[j]) / u;
    }
#if defined(NICO_SIMD_SCALAR_KERNELS)
    for (int i = 0; i < total; ++i) arguments[i] = nico_log_1_double_precision(arguments[i]) + corrections[i];
#else
    const int padded = ((total + BMVP_LANES - 1) / BMVP_LANES) * BMVP_LANES;
    for (int i = total; i < padded; ++i) {
        arguments[i] = 1.0;
        corrections[i] = 0.0;
    }
    for (int first = 0; first < padded; first += BMVP_LANES) {
        const simd_t log_of_arguments
            = BMVP_KERNEL_NAME(nico_log_1_double_precision)(simd_load(arguments + first));
        simd_store(arguments + first, lgdg_add(log_of_arguments, simd_load(corrections + first)));
    }
#endif
    for (int i = 0; i < n; ++i) log_x[i] = arguments[i];
    for (int j = 0; j < m; ++j) log1m_y[j] = arguments[n + j];

}

// ---- one scratch buffer of doubles per thread for the values of a call (no allocation once it is large enough);
//      a kernel takes it once and calls no other kernel that takes it
//      (64-byte aligned, the length rounded up to a whole number of 64-byte lines)
struct nico_free_aligned {

    void operator()(double* p) const { std::free(p); }

};

inline double* nico_scratch_of_doubles(const std::size_t n) {

    // static thread_local std::vector<double> scratch;
    // if (scratch.size() < n) scratch.resize(n);
    // return scratch.data();
    static thread_local std::unique_ptr<double[], nico_free_aligned> scratch;
    static thread_local std::size_t scratch_size = 0;
    if (scratch_size < n) {
        const std::size_t size = ((n + 7) / 8) * 8;
        scratch.reset(static_cast<double*>(std::aligned_alloc(64, size * sizeof(double))));
        if (!scratch) throw std::bad_alloc();
        scratch_size = size;
    }
    return scratch.get();

}

// ---- the vector overloads of nico_lgamma (the scalar one, libm's lgamma_r, is in stan_external_lgamma.hpp):
//      data, and parameters (one callback for the vector: adj(x_i) += adj(lgamma(x_i)) digamma(x_i), with the
//      digamma from the same pass as the values)
//      Declared as   vector nico_lgamma(vector x);
template <typename T_x,
          stan::require_eigen_col_vector_vt<std::is_arithmetic, T_x>* = nullptr>
inline Eigen::Matrix<double, Eigen::Dynamic, 1>
nico_lgamma(const T_x& x,
            std::ostream* /*pstream__*/) {

    Eigen::Matrix<double, Eigen::Dynamic, 1> result = x;
    k_lgamma(result.data(), static_cast<int>(result.size()));
    return result;

}

template <typename T_x,
          stan::require_eigen_col_vector_vt<stan::is_var, T_x>* = nullptr>
inline Eigen::Matrix<stan::math::var, Eigen::Dynamic, 1>
nico_lgamma(const T_x& x,
            std::ostream* /*pstream__*/) {

    const int n = static_cast<int>(x.size());
    stan::arena_t<Eigen::Matrix<stan::math::var, Eigen::Dynamic, 1>> x_arena = x;
    stan::arena_t<Eigen::VectorXd> x_values = x_arena.val();
    stan::arena_t<Eigen::VectorXd> lgamma_values(n);
    stan::arena_t<Eigen::VectorXd> digamma_values(n);
    k_lgamma_and_digamma<true>(x_values.data(), n, lgamma_values.data(), digamma_values.data());
    stan::arena_t<Eigen::Matrix<stan::math::var, Eigen::Dynamic, 1>> result = lgamma_values;
    stan::math::reverse_pass_callback([x_arena, result, digamma_values]() mutable {
        x_arena.adj().array() += result.adj().array() * digamma_values.array();
    });
    return Eigen::Matrix<stan::math::var, Eigen::Dynamic, 1>(result);

}

}  // namespace nico::math

using nico::math::nico_lgamma;
