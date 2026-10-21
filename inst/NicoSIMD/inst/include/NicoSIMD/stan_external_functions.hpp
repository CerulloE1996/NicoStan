#pragma once
//
// -| --------- BayesMVP AVX-512 / AVX2 special functions, exposed to Stan as external functions ------------------------------
//
// Vector -> vector versions of the special functions that dominate the LC-MVOP likelihood, computed
// with the BayesMVP SIMD kernels (fast_and_approx_AVX512_fns.hpp / fast_and_approx_AVX2_fns.hpp)
// instead of Stan's scalar implementations. Each function has
//
//     * a PRIM overload  (Eigen column vector of double)  -> Eigen::Matrix<double, -1, 1>
//     * a REV  overload  (Eigen column vector of var)     -> Eigen::Matrix<var, -1, 1>
//
// v4 (this file):
//
//   * SELF-CONTAINED. Only the two kernel files are included, with the three macros they need
//     defined here. v1-v3 included BayesMVP's initial_config / SIMD_config / eigen_config /
//     double_fns.hpp, and double_fns.hpp puts `using namespace Eigen; using namespace stan::math;`
//     and Eigen macro redefinitions into the WHOLE model translation unit (the user header is
//     force-included before the generated model code). Nothing of BayesMVP's leaks into the model now.
//
//   * LEAN TAPE. Forward pass: operand values gathered into thread-local scratch, kernel applied,
//     result varis placement-new'ed as ONE contiguous block on the arena and registered on the
//     nochain stack (their chain() is empty). Only the operand vari pointers are stored. Per element:
//     8 (operand ptr) + 24 (vari) bytes on the arena; nothing else. Stan's own path for
//     Eigen::Matrix<var> input (apply_scalar_unary -> one callback vari per element) is 32 bytes
//     per element plus one virtual call per element in the reverse pass.
//     Reverse pass: one callback for the whole vector; it re-reads operand and result values,
//     recomputes F'(x) with the kernels into scratch, and scatters adj(x) += adj(f) * F'(x).
//
//   In isolation (stan_ext/microbench, 96 threads, length-200 vectors) this is 1.1x (log) to 12x
//   (inv_Phi_approx_from_logit_prob) faster than Stan's built-ins, and 10-30% faster than v3.
//
// Declared in the Stan functions block as e.g.   vector nico_Phi(vector x);   (no body) and compiled
// with --allow-undefined and this file as the user header (cmdstanr: user_header = ...;
// BridgeStan / BayesMVP: stanc_args --allow-undefined + make_args USER_HEADER=...).
//
// Functions:  nico_Phi, nico_log_Phi, nico_inv_Phi, nico_exp, nico_log, nico_log1m, nico_log1p,
//             nico_log1p_exp, nico_log1m_exp, nico_inv_logit, nico_log_inv_logit, nico_Phi_approx,
//             nico_log_Phi_approx, nico_inv_Phi_approx, nico_inv_Phi_approx_from_logit_prob,
//             bmvp_simd_lanes (diagnostic: 8 = AVX-512 kernels, 4 = AVX2 kernels)
//             nico_tanh, nico_square, nico_sqrt, nico_logit, nico_log1m_inv_logit, nico_expm1,
//             nico_inv_Phi_from_log_p(log_p, log_1m_p); and every elementwise function also for a row_vector and a
//             matrix (the sections at the end of this file)
//
// Kernel dispatch: AVX-512 (8 lanes) when compiled with -mavx512f -mavx512vl -mavx512dq, else AVX2
// (4 lanes) with -mavx2; anything less is a compile error (a silently scalar build is what we do NOT
// want to benchmark). Compiling with -DBAYESMVP_FORCE_AVX2 selects the 4-lane AVX2 kernels
// even when AVX-512 is enabled (e.g. BridgeStan make argument CPPFLAGS_OPTIM=-DBAYESMVP_FORCE_AVX2), so an
// AVX-512 machine can run the AVX2 arm with every other compile flag unchanged; bmvp_simd_lanes() then
// returns 4, which callers check against the requested backend. Tail handling: a partial final block goes through a padded stack buffer whose
// spare lanes repeat the last valid element; only the valid lanes are written back.
//
// -----------------------------------------------------------------------------------------------------------------------------

#include <stan/math/rev.hpp>
#include <stan/math/prim.hpp>

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include "nico_simd_intrinsics.hpp"   //// <immintrin.h> on x86-64; SIMDe (NEON) on ARM
#include <new>
#include <ostream>
#include <vector>

// ---- the three macros the kernel files use (same definitions as BayesMVP's initial_config / SIMD_config)
#ifndef ALWAYS_INLINE
#  if defined(__GNUC__) || defined(__clang__)
#    define ALWAYS_INLINE __attribute__((always_inline)) inline
#  else
#    define ALWAYS_INLINE inline
#  endif
#endif
#ifndef VECTORCALL
#  define VECTORCALL
#endif
#ifndef ALIGN64
#  define ALIGN64 alignas(64)
#endif
#ifndef ALIGN32
#  define ALIGN32 alignas(32)
#endif

// ---- the four global constants the kernel files read (same values as BayesMVP's double_fns.hpp)
#ifndef BMVP_STAN_EXT_KERNEL_CONSTANTS
#define BMVP_STAN_EXT_KERNEL_CONSTANTS
static constexpr double NORMAL_TAIL_THRESH  = 2.5;
static constexpr int    MILLS_CF_DEPTH      = 40;
static constexpr double NEG_HALF_LOG_2PI    = -0.91893853320467274178;   // -0.5*log(2*pi)
static constexpr double INV_SQRT_2PI        =  0.39894228040143267794;   //  1/sqrt(2*pi)
#endif

// ---- the SIMD kernels (self-contained: <immintrin.h>, <cmath>, the macros and constants above). Absolute paths: no -I needed.
#include "math/fast_and_approx_AVX2_fns.hpp"
#include "math/fast_and_approx_AVX512_fns.hpp"

#if defined(__AVX512F__) && defined(__AVX512VL__) && defined(__AVX512DQ__) && !defined(BAYESMVP_FORCE_AVX2)
#  define BMVP_LANES 8
#  define BMVP_KERNEL_NAME(stem) stem##_AVX512
#elif defined(NICO_SIMD_AVX2_KERNELS)   //// AVX2 natively, or NEON through SIMDe
#  define BMVP_LANES 4
#  define BMVP_KERNEL_NAME(stem) stem##_AVX2
#else
#  error "BayesMVP_AVX_stan_fns.hpp: build with -mavx512f -mavx512vl -mavx512dq (or at least -mavx2); see BridgeStan make/local"
#endif

namespace nico::math {

using stan::math::var;
using stan::math::vari;
using VecVar = Eigen::Matrix<var, -1, 1>;

// ---- constants
static constexpr double INV_SQRT_2PI = 0.398942280401432677939946059934;
static constexpr double SQRT_2PI     = 2.506628274631000502415765284811;
static constexpr double LN2          = 0.693147180559945309417232121458;
static constexpr double PHI_APPROX_A = 0.07056;   // Stan's Phi_approx: inv_logit(0.07056 x^3 + 1.5976 x)
static constexpr double PHI_APPROX_B = 1.5976;
static constexpr double INV_PHI_APPROX_C1 = 0.34176618822627863;   // the model's inv_Phi_approx: C2 * sinh(asinh(C1 * L) / 3)
static constexpr double INV_PHI_APPROX_C2 = 5.494448514153059;

// ---- generic elementwise application on a raw buffer, in place. Kernels are passed as callables
// (the BayesMVP kernels are always_inline and cannot be called through function pointers).
////
//// ---- SIMD register type: single instruction, multiple data, with double-precision lanes:
////
//// simd_t is one processor register; BMVP_LANES gives its number of double-precision values.
#if BMVP_LANES == 8
//// __m512d is a 512-bit register holding eight doubles, selected for AVX-512.
using simd_t = __m512d;
inline simd_t simd_loadu(const double* p) { return _mm512_loadu_pd(p); }
inline void   simd_storeu(double* p, simd_t v) { _mm512_storeu_pd(p, v); }
inline simd_t simd_load(const double* p) { return _mm512_load_pd(p); }
inline void   simd_store(double* p, simd_t v) { _mm512_store_pd(p, v); }
#else
//// __m256d is a 256-bit register holding four doubles, selected for AVX2.
using simd_t = __m256d;
inline simd_t simd_loadu(const double* p) { return _mm256_loadu_pd(p); }
inline void   simd_storeu(double* p, simd_t v) { _mm256_storeu_pd(p, v); }
inline simd_t simd_load(const double* p) { return _mm256_load_pd(p); }
inline void   simd_store(double* p, simd_t v) { _mm256_store_pd(p, v); }
#endif

template <typename K>
inline void apply_kernel(double* p, const int n, K k) {

    int i = 0;
    for (; i + BMVP_LANES <= n; i += BMVP_LANES) simd_storeu(p + i, k(simd_loadu(p + i)));
    if (i < n) {
        alignas(64) double buf[BMVP_LANES];
        const int rem = n - i;
        for (int j = 0; j < BMVP_LANES; ++j) buf[j] = p[i + (j < rem ? j : rem - 1)];
        simd_store(buf, k(simd_load(buf)));
        for (int j = 0; j < rem; ++j) p[i + j] = buf[j];
    }

}

#define BMVP_K(stem) [](const simd_t v) { return BMVP_KERNEL_NAME(stem)(v); }

// ---- in-place kernels on a buffer of n doubles
inline void k_Phi(double* p, int n)            { apply_kernel(p, n, BMVP_K(nico_Phi)); }
inline void k_log_Phi(double* p, int n)        { apply_kernel(p, n, BMVP_K(nico_log_Phi)); }
inline void k_dlog_Phi_dx(double* p, int n)    { apply_kernel(p, n, BMVP_K(nico_dlog_Phi_dx)); }
inline void k_inv_Phi(double* p, int n)        { apply_kernel(p, n, BMVP_K(nico_inv_Phi_wo_checks)); }
inline void k_exp(double* p, int n)            { apply_kernel(p, n, BMVP_K(nico_exp_1)); }
// (log on plain numbers keeps full double precision; the single-precision polynomial made the two-class
//  LC-MVOP's log density differ from Stan math by 2.2e-9, over the 1e-9 gate)
// inline void k_log(double* p, int n)            { apply_kernel(p, n, BMVP_K(nico_log_1)); }
inline void k_log(double* p, int n)            { apply_kernel(p, n, BMVP_K(nico_log_1_double_precision)); }
inline void k_log1m(double* p, int n)          { apply_kernel(p, n, BMVP_K(nico_log1m_1)); }
inline void k_log1p(double* p, int n)          { apply_kernel(p, n, BMVP_K(nico_log1p_1)); }
inline void k_log1p_exp(double* p, int n)      { apply_kernel(p, n, BMVP_K(nico_log1p_exp_1)); }
inline void k_inv_logit(double* p, int n)      { apply_kernel(p, n, BMVP_K(nico_inv_logit)); }
inline void k_log_inv_logit(double* p, int n)  { apply_kernel(p, n, BMVP_K(nico_log_inv_logit)); }
inline void k_Phi_approx(double* p, int n)     { apply_kernel(p, n, BMVP_K(nico_Phi_approx)); }
inline void k_log_Phi_approx(double* p, int n) { apply_kernel(p, n, BMVP_K(nico_log_Phi_approx)); }
inline void k_inv_Phi_approx(double* p, int n) { apply_kernel(p, n, BMVP_K(nico_inv_Phi_approx)); }
// inline void k_inv_Phi_approx_from_logit_prob(double* p, int n) { apply_kernel(p, n, BMVP_K(nico_inv_Phi_approx_from_logit_prob)); }
// inv_Phi_approx_from_logit_prob(L) = C2 sinh(asinh(C1 L) / 3) is odd in L: computed from s = |C1 L|,
// and the sign of L is put back at the end. (The kernel nico_inv_Phi_approx_from_logit_prob_* takes
// asinh(x) = log(x + sqrt(x^2 + 1)), which cancels for x < 0: -inf for L below about -3e8, and 0.6% off
// at L = -1e8; and it gives 7.3e-14, not 0, at L = 0.) Here asinh(s) = log1p(s + s^2 / (1 + sqrt(1 + s^2)))
// where s <= 1e8, and log1p(2 s - 1) = log(2 s) above (exact to double precision there, and no overflow of
// s^2); sinh(y), y >= 0, is (exp(y) - 1 / exp(y)) / 2 where y > 0.5, and the odd Taylor series to
// y^15 / 15! where y <= 0.5 (the first term left out is below 5e-20 of the value)
inline simd_t simd_inv_Phi_approx_from_logit_prob_of(const simd_t logit_p) {

    // 1 / 15!, 1 / 13!, ..., 1 / 1!
    static constexpr double odd_inverse_factorials[8] = { 1.0 / 1307674368000.0, 1.0 / 6227020800.0,
                                                          1.0 / 39916800.0, 1.0 / 362880.0, 1.0 / 5040.0,
                                                          1.0 / 120.0, 1.0 / 6.0, 1.0 };
#if BMVP_LANES == 8
    ALIGN64  __m512d const one              = _mm512_set1_pd(1.0);
    ALIGN64  __m512d const s                =
            _mm512_abs_pd(_mm512_mul_pd(_mm512_set1_pd(INV_PHI_APPROX_C1), logit_p));
    ALIGN64  __m512d const s_squared        = _mm512_mul_pd(s, s);
    ALIGN64  __m512d const log1p_argument   = _mm512_add_pd(s, _mm512_div_pd(s_squared,
                                         _mm512_add_pd(one, _mm512_sqrt_pd(_mm512_add_pd(one, s_squared)))));
    ALIGN64  __m512d const two_s_minus_one  = _mm512_fmsub_pd(_mm512_set1_pd(2.0), s, one);
    const __mmask8 large           = _mm512_cmp_pd_mask(s, _mm512_set1_pd(1e8), _CMP_GT_OQ);
    ALIGN64  __m512d const y                = _mm512_mul_pd(_mm512_set1_pd(1.0 / 3.0),
                                         nico_log1p_1_AVX512(_mm512_mask_blend_pd(large, log1p_argument,
                                                                                  two_s_minus_one)));
    ALIGN64  __m512d const y_squared        = _mm512_mul_pd(y, y);
    ALIGN64  __m512d series = _mm512_set1_pd(odd_inverse_factorials[0]);
    for (int k = 1; k < 8; ++k) {
        series = _mm512_fmadd_pd(series, y_squared, _mm512_set1_pd(odd_inverse_factorials[k]));
    }
    series = _mm512_mul_pd(series, y);
    ALIGN64  __m512d const exp_y            = nico_exp_1_AVX512(y);
    ALIGN64  __m512d const from_exp         = _mm512_mul_pd(_mm512_set1_pd(0.5),
                                                   _mm512_sub_pd(exp_y, _mm512_div_pd(one, exp_y)));
    const __mmask8 small           = _mm512_cmp_pd_mask(y, _mm512_set1_pd(0.5), _CMP_LE_OQ);
    ALIGN64  __m512d const sinh_y           = _mm512_mask_blend_pd(small, from_exp, series);
    ALIGN64  __m512d const sign_of_logit_p  = _mm512_and_pd(logit_p, _mm512_set1_pd(-0.0));
    return _mm512_xor_pd(_mm512_mul_pd(_mm512_set1_pd(INV_PHI_APPROX_C2), sinh_y), sign_of_logit_p);
#else
    ALIGN32  __m256d const one              = _mm256_set1_pd(1.0);
    ALIGN32  __m256d const s                = _mm256_andnot_pd(_mm256_set1_pd(-0.0),
                                         _mm256_mul_pd(_mm256_set1_pd(INV_PHI_APPROX_C1), logit_p));
    ALIGN32  __m256d const s_squared        = _mm256_mul_pd(s, s);
    ALIGN32  __m256d const log1p_argument   = _mm256_add_pd(s, _mm256_div_pd(s_squared,
                                         _mm256_add_pd(one, _mm256_sqrt_pd(_mm256_add_pd(one, s_squared)))));
    ALIGN32  __m256d const two_s_minus_one  = _mm256_fmsub_pd(_mm256_set1_pd(2.0), s, one);
    ALIGN32  __m256d const large            = _mm256_cmp_pd(s, _mm256_set1_pd(1e8), _CMP_GT_OQ);
    ALIGN32  __m256d const y                = _mm256_mul_pd(_mm256_set1_pd(1.0 / 3.0),
                                         nico_log1p_1_AVX2(_mm256_blendv_pd(log1p_argument, two_s_minus_one,
                                                                            large)));
    ALIGN32  __m256d const y_squared        = _mm256_mul_pd(y, y);
    ALIGN32  __m256d series = _mm256_set1_pd(odd_inverse_factorials[0]);
    for (int k = 1; k < 8; ++k) {
        series = _mm256_fmadd_pd(series, y_squared, _mm256_set1_pd(odd_inverse_factorials[k]));
    }
    series = _mm256_mul_pd(series, y);
    ALIGN32  __m256d const exp_y            = nico_exp_1_AVX2(y);
    ALIGN32  __m256d const from_exp         = _mm256_mul_pd(_mm256_set1_pd(0.5),
                                                   _mm256_sub_pd(exp_y, _mm256_div_pd(one, exp_y)));
    ALIGN32  __m256d const small            = _mm256_cmp_pd(y, _mm256_set1_pd(0.5), _CMP_LE_OQ);
    ALIGN32  __m256d const sinh_y           = _mm256_blendv_pd(from_exp, series, small);
    ALIGN32  __m256d const sign_of_logit_p  = _mm256_and_pd(logit_p, _mm256_set1_pd(-0.0));
    return _mm256_xor_pd(_mm256_mul_pd(_mm256_set1_pd(INV_PHI_APPROX_C2), sinh_y), sign_of_logit_p);
#endif

}
inline void k_inv_Phi_approx_from_logit_prob(double* p, int n) {

    apply_kernel(p, n, [](const simd_t v) { return simd_inv_Phi_approx_from_logit_prob_of(v); });

}
inline void k_log1m_exp(double* p, int n)      { apply_kernel(p, n, BMVP_K(nico_log1m_exp_1)); }
#undef BMVP_K

inline void copy_n(const double* x, int n, double* out) { for (int i = 0; i < n; ++i) out[i] = x[i]; }

// ---- VALUE functions:      f <- F(x)            (x read-only; f written; both length n; f != x)
// ---- DERIVATIVE functions: d <- F'(x)           given the operand values x AND the result values f,
//                                                  so the reverse pass can recompute cheaply.

inline void v_Phi(const double* x, int n, double* f) { copy_n(x, n, f); k_Phi(f, n); }
inline void d_Phi(const double* x, const double*, int n, double* d) {

    for (int i = 0; i < n; ++i) d[i] = -0.5 * x[i] * x[i];
    k_exp(d, n);
    for (int i = 0; i < n; ++i) d[i] *= INV_SQRT_2PI;                          // phi(x)

}

inline void v_log_Phi(const double* x, int n, double* f) { copy_n(x, n, f); k_log_Phi(f, n); }
inline void d_log_Phi(const double* x, const double*, int n, double* d) { copy_n(x, n, d); k_dlog_Phi_dx(d, n); }

inline void v_inv_Phi(const double* p, int n, double* f) { copy_n(p, n, f); k_inv_Phi(f, n); }
inline void d_inv_Phi(const double*, const double* z, int n, double* d) {

    for (int i = 0; i < n; ++i) d[i] = 0.5 * z[i] * z[i];
    k_exp(d, n);
    for (int i = 0; i < n; ++i) d[i] *= SQRT_2PI;                               // 1 / phi(z)

}

inline void v_exp(const double* x, int n, double* f) { copy_n(x, n, f); k_exp(f, n); }
inline void d_exp(const double*, const double* f, int n, double* d) { copy_n(f, n, d); }

inline void v_log(const double* x, int n, double* f) { copy_n(x, n, f); k_log(f, n); }
inline void d_log(const double* x,
                  const double*,
                  int n,
                  double* d) { for (int i = 0; i < n; ++i) d[i] = 1.0 / x[i]; }

inline void v_log1m(const double* x, int n, double* f) { copy_n(x, n, f); k_log1m(f, n); }
inline void d_log1m(const double* x,
                    const double*,
                    int n,
                    double* d) { for (int i = 0; i < n; ++i) d[i] = -1.0 / (1.0 - x[i]); }

inline void v_log1p(const double* x, int n, double* f) { copy_n(x, n, f); k_log1p(f, n); }
inline void d_log1p(const double* x,
                    const double*,
                    int n,
                    double* d) { for (int i = 0; i < n; ++i) d[i] = 1.0 / (1.0 + x[i]); }

inline void v_log1p_exp(const double* x, int n, double* f) { copy_n(x, n, f); k_log1p_exp(f, n); }
inline void d_log1p_exp(const double* x, const double*, int n, double* d) { copy_n(x, n, d); k_inv_logit(d, n); }

// log1m_exp(x) = log(1 - exp(x)), x <= 0: log1p(-exp(x)) via the kernels for x < -ln 2; nearer 0
// that loses digits, so those lanes use the scalar log(-expm1(x)).
// inline void v_log1m_exp(const double* x, int n, double* f) {
//     copy_n(x, n, f); k_exp(f, n);
//     for (int i = 0; i < n; ++i) f[i] = -f[i];
//     k_log1p(f, n);
//     for (int i = 0; i < n; ++i) if (x[i] >= -LN2) f[i] = std::log(-std::expm1(x[i]));
// }
// log1m_exp by the SIMD kernel nico_log1m_exp_1 (math/nico_and_approx_*_fns.hpp): log1p(-exp(x)) for x < -ln 2 (the
// same operations as above), and log(-x) + log(expm1(x) / x) with the double-precision log nearer 0, in SIMD.
inline void v_log1m_exp(const double* x, int n, double* f) { copy_n(x, n, f); k_log1m_exp(f, n); }
inline void d_log1m_exp(const double* x, const double* f, int n, double* d) {

    for (int i = 0; i < n; ++i) d[i] = x[i] - f[i];
    k_exp(d, n);
    for (int i = 0; i < n; ++i) d[i] = -d[i];                                   // -exp(x) / (1 - exp(x))

}

inline void v_inv_logit(const double* x, int n, double* f) { copy_n(x, n, f); k_inv_logit(f, n); }
inline void d_inv_logit(const double*,
                        const double* f,
                        int n,
                        double* d) { for (int i = 0; i < n; ++i) d[i] = f[i] * (1.0 - f[i]); }

inline void v_log_inv_logit(const double* x, int n, double* f) { copy_n(x, n, f); k_log_inv_logit(f, n); }
inline void d_log_inv_logit(const double* x, const double*, int n, double* d) {

    for (int i = 0; i < n; ++i) d[i] = -x[i];
    k_inv_logit(d, n);                                                           // 1 - inv_logit(x)

}

inline double phi_approx_dpoly(double x) { return 3.0 * PHI_APPROX_A * x * x + PHI_APPROX_B; }

inline void v_Phi_approx(const double* x, int n, double* f) { copy_n(x, n, f); k_Phi_approx(f, n); }
inline void d_Phi_approx(const double* x, const double* f, int n, double* d) {

    for (int i = 0; i < n; ++i) d[i] = f[i] * (1.0 - f[i]) * phi_approx_dpoly(x[i]);

}

inline void v_log_Phi_approx(const double* x, int n, double* f) { copy_n(x, n, f); k_log_Phi_approx(f, n); }
inline void d_log_Phi_approx(const double* x, const double* f, int n, double* d) {

    copy_n(f, n, d); k_exp(d, n);                                                // p = exp(f)
    for (int i = 0; i < n; ++i) d[i] = (1.0 - d[i]) * phi_approx_dpoly(x[i]);

}

// z = C2 * sinh(asinh(C1 * L) / 3): dz/dL = (C2 C1 / 3) * sqrt(1 + (z / C2)^2) / sqrt(1 + (C1 L)^2)
inline double inv_phi_approx_dz_dL(double L, double z) {

    const double zc = z / INV_PHI_APPROX_C2;
    const double Lc = INV_PHI_APPROX_C1 * L;
    return (INV_PHI_APPROX_C2 * INV_PHI_APPROX_C1 / 3.0) * std::sqrt(1.0 + zc * zc) / std::sqrt(1.0 + Lc * Lc);

}

inline void v_inv_Phi_approx(const double* p, int n, double* f) { copy_n(p, n, f); k_inv_Phi_approx(f, n); }
inline void d_inv_Phi_approx(const double* p, const double* z, int n, double* d) {

    for (int i = 0; i < n; ++i) {
        const double L = std::log(p[i] / (1.0 - p[i]));
        d[i] = inv_phi_approx_dz_dL(L, z[i]) / (p[i] * (1.0 - p[i]));
    }

}

inline void v_inv_Phi_approx_from_logit_prob(const double* lp, int n, double* f) { copy_n(lp, n, f); k_inv_Phi_approx_from_logit_prob(f, n); }
inline void d_inv_Phi_approx_from_logit_prob(const double* lp, const double* z, int n, double* d) {

    for (int i = 0; i < n; ++i) d[i] = inv_phi_approx_dz_dL(lp[i], z[i]);

}

// ---- tanh, square, sqrt, logit and log1m_inv_logit: in-place kernels on a buffer of n doubles (tanh, logit and
//      log_inv_logit are BayesMVP's kernels of math/nico_and_approx_*_fns.hpp; square and sqrt are one instruction)
#if BMVP_LANES == 8
inline simd_t simd_square_of(const simd_t x)      { return _mm512_mul_pd(x, x); }
inline simd_t simd_square_root_of(const simd_t x) { return _mm512_sqrt_pd(x); }
inline simd_t simd_negative_of(const simd_t x)    { return _mm512_sub_pd(_mm512_setzero_pd(), x); }
#else
inline simd_t simd_square_of(const simd_t x)      { return _mm256_mul_pd(x, x); }
inline simd_t simd_square_root_of(const simd_t x) { return _mm256_sqrt_pd(x); }
inline simd_t simd_negative_of(const simd_t x)    { return _mm256_sub_pd(_mm256_setzero_pd(), x); }
#endif
// expm1(x) = exp(x) - 1: where |x| > 0.5, from the exp kernel (no cancellation there, |exp(x) - 1| > 0.39); where
// |x| <= 0.5, the Taylor series to x^15 / 15! (the first term left out is below 1.2e-18 of the value)
inline simd_t simd_expm1_of(const simd_t x) {

    static constexpr double inverse_factorials[15] = {
        1.0 / 1307674368000.0, 1.0 / 87178291200.0, 1.0 / 6227020800.0, 1.0 / 479001600.0, 1.0 / 39916800.0,
        1.0 / 3628800.0, 1.0 / 362880.0, 1.0 / 40320.0, 1.0 / 5040.0, 1.0 / 720.0, 1.0 / 120.0, 1.0 / 24.0,
        1.0 / 6.0, 1.0 / 2.0, 1.0 };                                            // 1 / 15!, ..., 1 / 1!
#if BMVP_LANES == 8
    ALIGN64  __m512d series = _mm512_set1_pd(inverse_factorials[0]);
    for (int k = 1; k < 15; ++k) series = _mm512_fmadd_pd(series, x, _mm512_set1_pd(inverse_factorials[k]));
    series = _mm512_mul_pd(series, x);
    const __mmask8 small = _mm512_cmp_pd_mask(_mm512_abs_pd(x), _mm512_set1_pd(0.5), _CMP_LE_OQ);
    if (small == 0xff) return series;
    ALIGN64  __m512d const from_exp = _mm512_sub_pd(nico_exp_1_AVX512(x), _mm512_set1_pd(1.0));
    return _mm512_mask_blend_pd(small, from_exp, series);
#else
    ALIGN32  __m256d series = _mm256_set1_pd(inverse_factorials[0]);
    for (int k = 1; k < 15; ++k) series = _mm256_fmadd_pd(series, x, _mm256_set1_pd(inverse_factorials[k]));
    series = _mm256_mul_pd(series, x);
    ALIGN32  __m256d const small = _mm256_cmp_pd(_mm256_abs_pd(x), _mm256_set1_pd(0.5), _CMP_LE_OQ);
    if (_mm256_movemask_pd(small) == 0xf) return series;
    ALIGN32  __m256d const from_exp = _mm256_sub_pd(nico_exp_1_AVX2(x), _mm256_set1_pd(1.0));
    return _mm256_blendv_pd(from_exp, series, small);
#endif

}
inline void k_expm1(double* p, int n) {

    apply_kernel(p, n, [](const simd_t v) { return simd_expm1_of(v); });

}
// tanh(x): where |x| > 0.125, BayesMVP's kernel, 2 / (1 + exp(-2 x)) - 1 (absolute error ~1e-16, which near 0 is a
// large relative error); where |x| <= 0.125, the odd Taylor series to x^15 (the first term left out is below 3e-18
// of the value)
inline simd_t simd_tanh_of(const simd_t x) {

    static constexpr double coefficients[8] = { -929569.0 / 638512875.0, 21844.0 / 6081075.0, -1382.0 / 155925.0,
                                                 62.0 / 2835.0, -17.0 / 315.0, 2.0 / 15.0, -1.0 / 3.0, 1.0 };
#if BMVP_LANES == 8
    ALIGN64  __m512d const squared = _mm512_mul_pd(x, x);
    ALIGN64  __m512d series = _mm512_set1_pd(coefficients[0]);
    for (int k = 1; k < 8; ++k) series = _mm512_fmadd_pd(series, squared, _mm512_set1_pd(coefficients[k]));
    series = _mm512_mul_pd(series, x);
    const __mmask8 small = _mm512_cmp_pd_mask(_mm512_abs_pd(x), _mm512_set1_pd(0.125), _CMP_LE_OQ);
    if (small == 0xff) return series;
    return _mm512_mask_blend_pd(small, nico_tanh_AVX512(x), series);
#else
    ALIGN32  __m256d const squared = _mm256_mul_pd(x, x);
    ALIGN32  __m256d series = _mm256_set1_pd(coefficients[0]);
    for (int k = 1; k < 8; ++k) series = _mm256_fmadd_pd(series, squared, _mm256_set1_pd(coefficients[k]));
    series = _mm256_mul_pd(series, x);
    ALIGN32  __m256d const small = _mm256_cmp_pd(_mm256_abs_pd(x), _mm256_set1_pd(0.125), _CMP_LE_OQ);
    if (_mm256_movemask_pd(small) == 0xf) return series;
    return _mm256_blendv_pd(nico_tanh_AVX2(x), series, small);
#endif

}
inline void k_tanh(double* p, int n) {

    apply_kernel(p, n, [](const simd_t v) { return simd_tanh_of(v); });

}
inline void k_square(double* p, int n) {

    apply_kernel(p, n, [](const simd_t v) { return simd_square_of(v); });

}
inline void k_sqrt(double* p, int n) {

    apply_kernel(p, n, [](const simd_t v) { return simd_square_root_of(v); });

}
inline void k_logit(double* p, int n) {

    apply_kernel(p, n, [](const simd_t v) { return BMVP_KERNEL_NAME(nico_logit)(v); });

}
// log(1 - inv_logit(x)) = log_inv_logit(-x)
inline void k_log1m_inv_logit(double* p, int n) {

    apply_kernel(p, n, [](const simd_t v) { return BMVP_KERNEL_NAME(nico_log_inv_logit)(simd_negative_of(v)); });

}

// tanh(x) (simd_tanh_of above): d/dx = 1 - tanh(x)^2, from the result
inline void v_tanh(const double* x, int n, double* f) { copy_n(x, n, f); k_tanh(f, n); }
inline void d_tanh(const double*,
                   const double* f,
                   int n,
                   double* d) { for (int i = 0; i < n; ++i) d[i] = 1.0 - f[i] * f[i]; }

// square(x) = x * x: d/dx = 2 x, from the operand
inline void v_square(const double* x, int n, double* f) { copy_n(x, n, f); k_square(f, n); }
inline void d_square(const double* x,
                     const double*,
                     int n,
                     double* d) { for (int i = 0; i < n; ++i) d[i] = 2.0 * x[i]; }

// sqrt(x): d/dx = 0.5 / sqrt(x), from the result (infinite at x = 0, as Stan's sqrt)
inline void v_sqrt(const double* x, int n, double* f) { copy_n(x, n, f); k_sqrt(f, n); }
inline void d_sqrt(const double*,
                   const double* f,
                   int n,
                   double* d) { for (int i = 0; i < n; ++i) d[i] = 0.5 / f[i]; }

// logit(x) = log(x) - log1m(x): d/dx = 1 / (x (1 - x)), from the operand
inline void v_logit(const double* x, int n, double* f) { copy_n(x, n, f); k_logit(f, n); }
inline void d_logit(const double* x,
                    const double*,
                    int n,
                    double* d) { for (int i = 0; i < n; ++i) d[i] = 1.0 / (x[i] * (1.0 - x[i])); }

// expm1(x): d/dx = exp(x) = expm1(x) + 1, from the result
inline void v_expm1(const double* x, int n, double* f) { copy_n(x, n, f); k_expm1(f, n); }
inline void d_expm1(const double*,
                    const double* f,
                    int n,
                    double* d) { for (int i = 0; i < n; ++i) d[i] = f[i] + 1.0; }

// log1m_inv_logit(x) = log_inv_logit(-x): d/dx = -inv_logit(x), from the operand
inline void v_log1m_inv_logit(const double* x, int n, double* f) { copy_n(x, n, f); k_log1m_inv_logit(f, n); }
inline void d_log1m_inv_logit(const double* x, const double*, int n, double* d) {

    copy_n(x, n, d); k_inv_logit(d, n);
    for (int i = 0; i < n; ++i) d[i] = -d[i];

}

// ---- thread-local scratch: reused across calls, so no malloc after warm-up and it stays in cache
struct scratch_t {
    std::vector<double> a, b, c;
    void ensure(int n) { if ((int)a.size() < n) { a.resize(n); b.resize(n); c.resize(n); } }
};
inline scratch_t& scratch() { thread_local scratch_t s; return s; }

// ---- the rev-mode wrapper (see the header comment). Arena per element: 8 (operand vari*) + 24 (result vari).
// Select dependencies at compile time: never reread values which the derivative does not use.
// Keep 'both' as the default for existing callers and the matched pre-optimisation benchmark.
// Unused derivative arguments are nullptr; only the declared buffers are populated.
enum class derivative_values { operand, result, both };
template <derivative_values needed = derivative_values::both, typename T, typename V, typename D, typename InPlace = std::nullptr_t>
inline VecVar rev_apply(const T& x_in, V v, D d, InPlace in_place = nullptr) {

    const int n = static_cast<int>(x_in.size());
    auto& mem = stan::math::ChainableStack::instance_->memalloc_;
    vari** xp = mem.alloc_array<vari*>(n);
    scratch_t& sc = scratch(); sc.ensure(n);
    double* xb = sc.a.data(); double* fb = sc.b.data();
    for (int i = 0; i < n; ++i) { xp[i] = x_in.coeff(i).vi_; xb[i] = xp[i]->val_; }
    if constexpr (std::is_same<InPlace, std::nullptr_t>::value) {
        v(xb, n, fb);
    } else {
        in_place(xb, n);
        fb = xb;
    }
    vari* rv = mem.alloc_array<vari>(n);
    VecVar res(n);
    // for (int i = 0; i < n; ++i) {
    //     ::new (static_cast<void*>(rv + i)) vari(fb[i], false);       // nochain stack: chain() is empty
    //     res.coeffRef(i) = var(rv + i);
    // }
    // (the result nodes in a loop of their own, then the returned vars, as in rev_apply_strip_mined)
    for (int i = 0; i < n; ++i) ::new (static_cast<void*>(rv + i)) vari(fb[i], false);   // nochain stack
    var* res_vars = res.data();
    for (int i = 0; i < n; ++i) res_vars[i] = var(rv + i);
    stan::math::reverse_pass_callback([xp, rv, n, d]() {

        scratch_t& s2 = scratch(); s2.ensure(n);
        double* xb2 = s2.a.data(); double* fb2 = s2.b.data(); double* db2 = s2.c.data();
        if constexpr (needed == derivative_values::both) {
            for (int i = 0; i < n; ++i) { xb2[i] = xp[i]->val_; fb2[i] = rv[i].val_; }
        } else if constexpr (needed == derivative_values::operand) {
            for (int i = 0; i < n; ++i) xb2[i] = xp[i]->val_;
            fb2 = nullptr;
        } else {
            for (int i = 0; i < n; ++i) fb2[i] = rv[i].val_;
            xb2 = nullptr;
        }
        d(xb2, fb2, n, db2);
        for (int i = 0; i < n; ++i) xp[i]->adj_ += rv[i].adj_ * db2[i];

    });
    return res;

}

// ---- Specialised log / inv_Phi wrappers: same kernels, derivatives and arena ownership as rev_apply.
// Gather once and transform in place. log's reverse pass needs only operand values; inv_Phi's needs
// only result values. Keep adjoint accumulation scalar and ordered: different inputs can alias one vari.
template <bool inverse_normal, typename T>
inline VecVar rev_apply_log_or_inv_Phi(const T& input) {

    const int n = static_cast<int>(input.size());
    auto& arena = stan::math::ChainableStack::instance_->memalloc_;
    vari** input_nodes = arena.alloc_array<vari*>(n);
    scratch_t& buffers = scratch(); buffers.ensure(n);
    double* values = buffers.a.data();
    for (int i = 0; i < n; ++i) {
        input_nodes[i] = input.coeff(i).vi_;
        values[i] = input_nodes[i]->val_;
    }
    if constexpr (inverse_normal) k_inv_Phi(values, n);
    else                          k_log(values, n);
    vari* result_nodes = arena.alloc_array<vari>(n);
    VecVar result(n);
    for (int i = 0; i < n; ++i) {
        ::new (static_cast<void*>(result_nodes + i)) vari(values[i], false);
        result.coeffRef(i) = var(result_nodes + i);
    }
    stan::math::reverse_pass_callback([input_nodes, result_nodes, n]() {

        const auto update_block = [input_nodes, result_nodes](int first, int count) {

            alignas(64) double derivatives[BMVP_LANES];
            if constexpr (!inverse_normal) {
                for (int lane = 0; lane < BMVP_LANES; ++lane)
                    derivatives[lane] = input_nodes[first + (lane < count ? lane : count - 1)]->val_;
#if BMVP_LANES == 8
                simd_store(derivatives, _mm512_div_pd(_mm512_set1_pd(1.0), simd_load(derivatives)));
#else
                simd_store(derivatives, _mm256_div_pd(_mm256_set1_pd(1.0), simd_load(derivatives)));
#endif
            } else {
                alignas(64) double result_values[BMVP_LANES];
                for (int lane = 0; lane < BMVP_LANES; ++lane)
                    result_values[lane] = result_nodes[first + (lane < count ? lane : count - 1)].val_;
                d_inv_Phi(nullptr, result_values, BMVP_LANES, derivatives);
            }
            for (int lane = 0; lane < count; ++lane)
                input_nodes[first + lane]->adj_ += result_nodes[first + lane].adj_ * derivatives[lane];

        };
        // A constant full-block size lets the compiler remove padding/index tests from the hot loop.
        int first = 0;
        for (; first + BMVP_LANES <= n; first += BMVP_LANES) update_block(first, BMVP_LANES);
        if (first < n) update_block(first, n - first);

    });
    return result;

}

// These two value-only paths can apply the kernel directly to the returned Eigen vector.
template <bool inverse_normal, typename T>
inline Eigen::Matrix<double, -1, 1> prim_apply_log_or_inv_Phi(const T& input) {

    Eigen::Matrix<double, -1, 1> result = input.template cast<double>();
    if constexpr (inverse_normal) k_inv_Phi(result.data(), static_cast<int>(result.size()));
    else                          k_log(result.data(), static_cast<int>(result.size()));
    return result;

}

// ---- the prim (double) path: one output allocation, same as Stan's own vectorised functions.
template <typename T, typename V, typename InPlace = std::nullptr_t>
inline Eigen::Matrix<double, -1, 1> prim_apply(const T& x_in, V v, InPlace in_place = nullptr) {

    const int n = static_cast<int>(x_in.size());
    if constexpr (!std::is_same<InPlace, std::nullptr_t>::value) {
        Eigen::Matrix<double, -1, 1> f = x_in.template cast<double>();
        in_place(f.data(), n);
        return f;
    }
    scratch_t& sc = scratch(); sc.ensure(n);
    double* xb = sc.a.data();
    for (int i = 0; i < n; ++i) xb[i] = x_in.coeff(i);
    Eigen::Matrix<double, -1, 1> f(n);
    v(xb, n, f.data());
    return f;

}

}  // namespace nico::math




// -| --------- strip-mined reverse-mode (var) wrappers: the var overloads of the Stan-facing functions -----------
//
// Reverse-mode versions of the elementwise externals of this file, with the same kernels and
// value functions, the same derivative functions (exceptions below) and the same tape registration: result varis
// placement-new'ed in one arena block on the nochain stack (chain() is empty), operand vari pointers stored, ONE
// reverse callback per call, adjoints accumulated in element order (repeated operands stay correct).
//
// rev_apply (above) makes three full passes over the n elements in each direction,
// through three length-n thread-local scratch buffers: gather all operand values, apply the kernel to all of
// them, construct all result varis; in reverse, gather all values, compute all derivatives, scatter all
// adjoints. Here both directions are strip-mined into blocks of REV_BLOCK_LENGTH elements on stack buffers:
//
//     * forward:  per block, gather the operand vari pointers and values, apply the kernel in place, construct
//                 the result varis and the returned var vector, whilst the block's lines are still in L1.
//     * reverse:  per block, read the operand and / or result values which the derivative needs, compute F'(x)
//                 for the block, and accumulate adj(x) += adj(f) * F'(x) on the same, still-hot lines.
//
// Full blocks have a compile-time trip count (unrolled, no tail tests); one tail block takes the rest. No
// length-n scratch, no thread-local lookups and no scratch resizing per call. Of the block lengths 8, 16, 32,
// 64 and 256, 64 was the fastest or within 1.2% of it (benchmark_wrapper_design_variants.cpp, in
// Alg_paper_analysis/paper_3_general_sampler/fused_AVX_log_density_functions/elementwise).
//
// Function-specific paths:
//
//     log_Phi:          nico_log_Phi and nico_dlog_Phi_dx each evaluate the 40-term Mills-ratio continued
//                       fraction, and rev_apply evaluated it again in the reverse pass. Here ONE evaluation per
//                       element gives both log Phi(x) and its derivative (same operations in the same order as
//                       the two kernels: bit-identical); the derivative is stored on the arena (8 more bytes
//                       per element) and the reverse pass is the adjoint update alone.
//     log, inv_Phi:     the specialised production wrappers (rev_apply_log_or_inv_Phi), already as fast as
//                       the strip-mined generic path.
//     inv_Phi_approx_from_logit_prob:   z = C2 sinh(asinh(C1 L) / 3); with u = z / C2 and t = asinh(u),
//                       C1 L = sinh(3 t) = 3 u + 4 u^3 and cosh(3 t) = cosh(t) (1 + 4 u^2), so
//                       dz/dL = (C1 C2 / 3) / (1 + 4 u^2)                           (result values only)
//     inv_Phi_approx:   dz/dp = (C1 C2 / 3) / ((1 + 4 u^2) p (1 - p))       (no scalar std::log(p / (1 - p))
//                       and no std::sqrt per element in the reverse pass). Exact identities; the gradients differ
//                       from stan_external_functions.hpp at the level of the kernels' own accuracy (z from the
//                       kernel instead of L from std::log), not at rounding level.
//
// The var overloads of the Stan-facing functions below forward to nico::math::faster_rev::fast_*(x, pstream__);
// the prim (double) overloads are unchanged.
//
// ---------------------------------------------------------------------------------------------------------------

namespace nico::math {
namespace faster_rev {

// ---- block length of the strip-mined passes: a multiple of 8 (AVX-512) and 4 (AVX2) lanes, so every block
//      splits into SIMD vectors exactly as the whole vector does in rev_apply (same lane grouping and padding)
static constexpr int REV_BLOCK_LENGTH = 64;
static_assert(REV_BLOCK_LENGTH % 8 == 0, "REV_BLOCK_LENGTH must be a multiple of 8 (AVX-512) and 4 (AVX2) lanes");

// ---- forward and reverse pass of one elementwise function: in-place kernel K(values, n), or, when K is nullptr,
//      value function V(x, n, f); derivative function D(x, f, n, d) as in stan_external_functions.hpp. 'needed'
//      says which values D reads; the argument D does not read is nullptr.
template <derivative_values needed, typename T, typename V, typename K, typename D>
inline VecVar rev_apply_strip_mined(const T& input, V value_function, K in_place_kernel, D derivative_function) {

    const int n = static_cast<int>(input.size());
    auto& arena = stan::math::ChainableStack::instance_->memalloc_;
    vari** input_nodes  = arena.alloc_array<vari*>(n);
    vari*  result_nodes = arena.alloc_array<vari>(n);
    VecVar result(n);

    alignas(64) double operand_values[REV_BLOCK_LENGTH];
    alignas(64) double result_values[REV_BLOCK_LENGTH];

    const auto forward_block = [&](const int first, const int count) {

        for (int i = 0; i < count; ++i) {
            vari* node = input.coeff(first + i).vi_;
            input_nodes[first + i] = node;
            operand_values[i] = node->val_;
        }
        const double* values = operand_values;
        if constexpr (std::is_same<K, std::nullptr_t>::value) {
            value_function(operand_values, count, result_values);
            values = result_values;
        } else {
            in_place_kernel(operand_values, count);
        }
        // for (int i = 0; i < count; ++i) {
        //     vari* node = result_nodes + first + i;
        //     ::new (static_cast<void*>(node)) vari(values[i], false);      // nochain stack: chain() is empty
        //     result.coeffRef(first + i) = var(node);
        // }
        // (the result nodes in a loop of their own, then the returned vars: in one loop, each node's
        //  registration on the nochain stack waits on the stores of the element before it)
        for (int i = 0; i < count; ++i) {
            ::new (static_cast<void*>(result_nodes + first + i)) vari(values[i], false);   // nochain stack
        }
        var* result_vars_of_block = result.data() + first;
        for (int i = 0; i < count; ++i) result_vars_of_block[i] = var(result_nodes + first + i);

    };
    int first = 0;
    for (; first + REV_BLOCK_LENGTH <= n; first += REV_BLOCK_LENGTH) forward_block(first, REV_BLOCK_LENGTH);
    if (first < n) forward_block(first, n - first);

    stan::math::reverse_pass_callback([input_nodes, result_nodes, n, derivative_function]() {

        alignas(64) double operand_values[REV_BLOCK_LENGTH];
        alignas(64) double result_values[REV_BLOCK_LENGTH];
        alignas(64) double derivatives[REV_BLOCK_LENGTH];
        const auto reverse_block = [&](const int first, const int count) {

            vari** block_input_nodes  = input_nodes + first;
            vari*  block_result_nodes = result_nodes + first;
            const double* block_operand_values = nullptr;
            const double* block_result_values  = nullptr;
            if constexpr (needed != derivative_values::result) {
                for (int i = 0; i < count; ++i) operand_values[i] = block_input_nodes[i]->val_;
                block_operand_values = operand_values;
            }
            if constexpr (needed != derivative_values::operand) {
                for (int i = 0; i < count; ++i) result_values[i] = block_result_nodes[i].val_;
                block_result_values = result_values;
            }
            derivative_function(block_operand_values, block_result_values, count, derivatives);
            for (int i = 0; i < count; ++i) {
                block_input_nodes[i]->adj_ += block_result_nodes[i].adj_ * derivatives[i];
            }

        };
        int first = 0;
        for (; first + REV_BLOCK_LENGTH <= n; first += REV_BLOCK_LENGTH) reverse_block(first, REV_BLOCK_LENGTH);
        if (first < n) reverse_block(first, n - first);

    });

    return result;

}

// ---- log Phi(x) and d/dx log Phi(x) from ONE evaluation of the tail Mills ratio (nico_inverse_mills_ratio_tail_*,
//      the degree-8 rational) and of the body: the operations of nico_log_Phi_* and nico_dlog_Phi_dx_*
//      (math/nico_and_approx_*_fns.hpp), in the same order, with the same lane-mask tests (tail only if a lane is
//      in the tail, body only if a lane is not); both outputs are bit-identical to the two kernels.
#if BMVP_LANES == 8
inline void log_Phi_and_dlog_Phi_dx_simd(const __m512d x, __m512d& log_Phi_x, __m512d& dlog_Phi_dx_x) {

    ALIGN64  __m512d const x_aligned = x;
    ALIGN64  __m512d const thresh   = _mm512_set1_pd(NORMAL_TAIL_THRESH);
    const __mmask8 is_tail  = _mm512_cmp_pd_mask(x_aligned, _mm512_sub_pd(_mm512_setzero_pd(), thresh),
            _CMP_LT_OQ);
    ALIGN64  __m512d log_body = _mm512_setzero_pd(), d_body = _mm512_setzero_pd();
    ALIGN64  __m512d log_tail = _mm512_setzero_pd(), d_tail = _mm512_setzero_pd();
    if (is_tail != 0xff) {                                                     // at least one body lane
        ALIGN64  __m512d const Phi_x = nico_Phi_AVX512(x_aligned);
        ALIGN64  __m512d const phi   = _mm512_mul_pd(_mm512_set1_pd(::INV_SQRT_2PI),
                                            nico_exp_1_AVX512(_mm512_mul_pd(_mm512_set1_pd(-0.5),
                                                                            _mm512_mul_pd(x_aligned,
                                                                                    x_aligned))));
        log_body = nico_log_1_AVX512(Phi_x);
        d_body   = _mm512_div_pd(phi, Phi_x);
    }
    if (is_tail != 0) {                                                        // at least one tail lane
        ALIGN64  __m512d const ax        = _mm512_abs_pd(x_aligned);
        ALIGN64  __m512d const inverse_R = nico_inverse_mills_ratio_tail_AVX512(_mm512_max_pd(ax, thresh));
        ALIGN64  __m512d const R         = _mm512_div_pd(_mm512_set1_pd(1.0), inverse_R);
        log_tail = _mm512_add_pd(_mm512_fnmadd_pd(_mm512_set1_pd(0.5), _mm512_mul_pd(x_aligned, x_aligned),
                                                  _mm512_set1_pd(NEG_HALF_LOG_2PI)),
                                 nico_log_1_AVX512(R));
        d_tail   = inverse_R;
    }
    log_Phi_x     = _mm512_mask_blend_pd(is_tail, log_body, log_tail);
    dlog_Phi_dx_x = _mm512_mask_blend_pd(is_tail, d_body, d_tail);

}
#else
inline void log_Phi_and_dlog_Phi_dx_simd(const __m256d x, __m256d& log_Phi_x, __m256d& dlog_Phi_dx_x) {

    ALIGN32  __m256d const x_aligned = x;
    ALIGN32  __m256d const thresh     = _mm256_set1_pd(NORMAL_TAIL_THRESH);
    ALIGN32  __m256d const is_tail    = _mm256_cmp_pd(x_aligned, _mm256_sub_pd(_mm256_setzero_pd(), thresh),
            _CMP_LT_OQ);
    const int      tail_lanes = _mm256_movemask_pd(is_tail);
    ALIGN32  __m256d log_body = _mm256_setzero_pd(), d_body = _mm256_setzero_pd();
    ALIGN32  __m256d log_tail = _mm256_setzero_pd(), d_tail = _mm256_setzero_pd();
    if (tail_lanes != 0xf) {                                                   // at least one body lane
        ALIGN32  __m256d const Phi_x = nico_Phi_AVX2(x_aligned);
        ALIGN32  __m256d const phi   = _mm256_mul_pd(_mm256_set1_pd(::INV_SQRT_2PI),
                                            nico_exp_1_AVX2(_mm256_mul_pd(_mm256_set1_pd(-0.5),
                                                                          _mm256_mul_pd(x_aligned, x_aligned))));
        log_body = nico_log_1_AVX2(Phi_x);
        d_body   = _mm256_div_pd(phi, Phi_x);
    }
    if (tail_lanes != 0) {                                                     // at least one tail lane
        ALIGN32  __m256d const ax        = _mm256_abs_pd(x_aligned);
        ALIGN32  __m256d const inverse_R = nico_inverse_mills_ratio_tail_AVX2(_mm256_max_pd(ax, thresh));
        ALIGN32  __m256d const R         = _mm256_div_pd(_mm256_set1_pd(1.0), inverse_R);
        log_tail = _mm256_add_pd(_mm256_fnmadd_pd(_mm256_set1_pd(0.5), _mm256_mul_pd(x_aligned, x_aligned),
                                                  _mm256_set1_pd(NEG_HALF_LOG_2PI)),
                                 nico_log_1_AVX2(R));
        d_tail   = inverse_R;
    }
    log_Phi_x     = _mm256_blendv_pd(log_body, log_tail, is_tail);
    dlog_Phi_dx_x = _mm256_blendv_pd(d_body, d_tail, is_tail);

}
#endif

// ---- on n doubles: values (in) -> log Phi(values) (out, in place), and derivatives -> d/dx log Phi; a partial
//      final SIMD vector is padded with the last valid element, as in apply_kernel
inline void k_log_Phi_and_dlog_Phi_dx(double* values, const int n, double* derivatives) {

    int i = 0;
    for (; i + BMVP_LANES <= n; i += BMVP_LANES) {
        simd_t log_Phi_x, dlog_Phi_dx_x;
        log_Phi_and_dlog_Phi_dx_simd(simd_loadu(values + i), log_Phi_x, dlog_Phi_dx_x);
        simd_storeu(values + i, log_Phi_x);
        simd_storeu(derivatives + i, dlog_Phi_dx_x);
    }
    if (i < n) {
        alignas(64) double padded_values[BMVP_LANES];
        alignas(64) double padded_derivatives[BMVP_LANES];
        const int remaining = n - i;
        for (int lane = 0; lane < BMVP_LANES; ++lane) {
            padded_values[lane] = values[i + (lane < remaining ? lane : remaining - 1)];
        }
        simd_t log_Phi_x, dlog_Phi_dx_x;
        log_Phi_and_dlog_Phi_dx_simd(simd_load(padded_values), log_Phi_x, dlog_Phi_dx_x);
        simd_store(padded_values, log_Phi_x);
        simd_store(padded_derivatives, dlog_Phi_dx_x);
        for (int lane = 0; lane < remaining; ++lane) {
            values[i + lane]      = padded_values[lane];
            derivatives[i + lane] = padded_derivatives[lane];
        }
    }

}

// ---- log_Phi: value and derivative in the forward pass (one continued fraction per element), the derivative
//      stored on the arena; the reverse pass is the adjoint update alone. Arena per element: 8 + 24 + 8 bytes.
template <typename T>
inline VecVar rev_apply_log_Phi_storing_derivative(const T& input) {

    const int n = static_cast<int>(input.size());
    auto& arena = stan::math::ChainableStack::instance_->memalloc_;
    vari** input_nodes  = arena.alloc_array<vari*>(n);
    vari*  result_nodes = arena.alloc_array<vari>(n);
    double* derivatives = arena.alloc_array<double>(n);
    VecVar result(n);

    alignas(64) double values[REV_BLOCK_LENGTH];

    const auto forward_block = [&](const int first, const int count) {

        for (int i = 0; i < count; ++i) {
            vari* node = input.coeff(first + i).vi_;
            input_nodes[first + i] = node;
            values[i] = node->val_;
        }
        k_log_Phi_and_dlog_Phi_dx(values, count, derivatives + first);
        // for (int i = 0; i < count; ++i) {
        //     vari* node = result_nodes + first + i;
        //     ::new (static_cast<void*>(node)) vari(values[i], false);      // nochain stack: chain() is empty
        //     result.coeffRef(first + i) = var(node);
        // }
        // (the result nodes in a loop of their own, then the returned vars: in one loop, each node's
        //  registration on the nochain stack waits on the stores of the element before it)
        for (int i = 0; i < count; ++i) {
            ::new (static_cast<void*>(result_nodes + first + i)) vari(values[i], false);   // nochain stack
        }
        var* result_vars_of_block = result.data() + first;
        for (int i = 0; i < count; ++i) result_vars_of_block[i] = var(result_nodes + first + i);

    };
    int first = 0;
    for (; first + REV_BLOCK_LENGTH <= n; first += REV_BLOCK_LENGTH) forward_block(first, REV_BLOCK_LENGTH);
    if (first < n) forward_block(first, n - first);

    stan::math::reverse_pass_callback([input_nodes, result_nodes, derivatives, n]() {

        for (int i = 0; i < n; ++i) input_nodes[i]->adj_ += result_nodes[i].adj_ * derivatives[i];

    });

    return result;

}

// ---- inv_Phi_approx derivatives from the result z (see the header comment): u = z / C2,
//      dz/dL = (C1 C2 / 3) / (1 + 4 u^2) and dz/dp = dz/dL / (p (1 - p)). Plain loops (vectorised by g++).
static constexpr double INV_PHI_APPROX_DZ_DL_AT_ZERO = INV_PHI_APPROX_C1 * INV_PHI_APPROX_C2 / 3.0;

inline void d_inv_Phi_approx_from_logit_prob_from_z(const double* /*logit_p*/, const double* z, int n,
                                                    double* d) {

    for (int i = 0; i < n; ++i) {
        const double u = z[i] / INV_PHI_APPROX_C2;
        d[i] = INV_PHI_APPROX_DZ_DL_AT_ZERO / (1.0 + 4.0 * u * u);
    }

}

inline void d_inv_Phi_approx_from_z_and_p(const double* p, const double* z, int n, double* d) {

    for (int i = 0; i < n; ++i) {
        const double u = z[i] / INV_PHI_APPROX_C2;
        d[i] = INV_PHI_APPROX_DZ_DL_AT_ZERO / ((1.0 + 4.0 * u * u) * (p[i] * (1.0 - p[i])));
    }

}

}  // namespace faster_rev
}  // namespace nico::math

// ---- the Stan-facing var overloads, in namespace nico::math::faster_rev (same names and signatures as the
//      rev overloads in stan_external_functions.hpp). NEEDED: the values the derivative reads (operand, result,
//      both); K: the in-place kernel, or nullptr for the copy path through V.
#define BMVP_FASTER_REV_EXTERNAL(NAME, V, K, D, NEEDED)                                                          \
namespace nico::math { namespace faster_rev {                                                                 \
template <typename T, stan::require_eigen_col_vector_vt<stan::is_var, T>* = nullptr>                             \
inline VecVar NAME(const T& x, std::ostream* /*pstream__*/) {                                                    \
                                                                                                                \
    return rev_apply_strip_mined<derivative_values::NEEDED>(                                                     \
               x, [](const double* a, int n, double* f) { V(a, n, f); }, K,                                      \
               [](const double* a, const double* f, int n, double* d) { D(a, f, n, d); });                       \
                                                                                                                \
}                                                                                                                \
} }

#define BMVP_FASTER_REV_IN_PLACE(KERNEL) [](double* values, int n) { KERNEL(values, n); }

BMVP_FASTER_REV_EXTERNAL(nico_Phi,            v_Phi,            BMVP_FASTER_REV_IN_PLACE(k_Phi),
                         d_Phi,                                                                    operand)
BMVP_FASTER_REV_EXTERNAL(nico_exp,            v_exp,            BMVP_FASTER_REV_IN_PLACE(k_exp),
                         d_exp,                                                                    result)
BMVP_FASTER_REV_EXTERNAL(nico_log1m,          v_log1m,          BMVP_FASTER_REV_IN_PLACE(k_log1m),
                         d_log1m,                                                                  operand)
BMVP_FASTER_REV_EXTERNAL(nico_log1p,          v_log1p,          BMVP_FASTER_REV_IN_PLACE(k_log1p),
                         d_log1p,                                                                  operand)
BMVP_FASTER_REV_EXTERNAL(nico_log1p_exp,      v_log1p_exp,      BMVP_FASTER_REV_IN_PLACE(k_log1p_exp),
                         d_log1p_exp,                                                              operand)
// log1m_exp: the SIMD kernel nico_log1m_exp_1 (no scalar near-zero repair), in place
BMVP_FASTER_REV_EXTERNAL(nico_log1m_exp,      v_log1m_exp,      BMVP_FASTER_REV_IN_PLACE(k_log1m_exp),
                         d_log1m_exp,                                                              both)
BMVP_FASTER_REV_EXTERNAL(nico_inv_logit,      v_inv_logit,      BMVP_FASTER_REV_IN_PLACE(k_inv_logit),
                         d_inv_logit,                                                              result)
BMVP_FASTER_REV_EXTERNAL(nico_log_inv_logit,  v_log_inv_logit,  BMVP_FASTER_REV_IN_PLACE(k_log_inv_logit),
                         d_log_inv_logit,                                                          operand)
// 'both': the reverse pass re-reads x from the operand varis, so the forward block can still be overwritten
BMVP_FASTER_REV_EXTERNAL(nico_Phi_approx,     v_Phi_approx,     BMVP_FASTER_REV_IN_PLACE(k_Phi_approx),
                         d_Phi_approx,                                                             both)
BMVP_FASTER_REV_EXTERNAL(nico_log_Phi_approx, v_log_Phi_approx, BMVP_FASTER_REV_IN_PLACE(k_log_Phi_approx),
                         d_log_Phi_approx,                                                         both)
BMVP_FASTER_REV_EXTERNAL(nico_inv_Phi_approx, v_inv_Phi_approx, BMVP_FASTER_REV_IN_PLACE(k_inv_Phi_approx),
                         faster_rev::d_inv_Phi_approx_from_z_and_p,                                both)
BMVP_FASTER_REV_EXTERNAL(nico_inv_Phi_approx_from_logit_prob, v_inv_Phi_approx_from_logit_prob,
                         BMVP_FASTER_REV_IN_PLACE(k_inv_Phi_approx_from_logit_prob),
                         faster_rev::d_inv_Phi_approx_from_logit_prob_from_z,                      result)
BMVP_FASTER_REV_EXTERNAL(nico_tanh,           v_tanh,           BMVP_FASTER_REV_IN_PLACE(k_tanh),
                         d_tanh,                                                                   result)
BMVP_FASTER_REV_EXTERNAL(nico_square,         v_square,         BMVP_FASTER_REV_IN_PLACE(k_square),
                         d_square,                                                                 operand)
BMVP_FASTER_REV_EXTERNAL(nico_sqrt,           v_sqrt,           BMVP_FASTER_REV_IN_PLACE(k_sqrt),
                         d_sqrt,                                                                   result)
BMVP_FASTER_REV_EXTERNAL(nico_logit,          v_logit,          BMVP_FASTER_REV_IN_PLACE(k_logit),
                         d_logit,                                                                  operand)
BMVP_FASTER_REV_EXTERNAL(nico_log1m_inv_logit, v_log1m_inv_logit, BMVP_FASTER_REV_IN_PLACE(k_log1m_inv_logit),
                         d_log1m_inv_logit,                                                        operand)
BMVP_FASTER_REV_EXTERNAL(nico_expm1,          v_expm1,          BMVP_FASTER_REV_IN_PLACE(k_expm1),
                         d_expm1,                                                                  result)

#undef BMVP_FASTER_REV_EXTERNAL
#undef BMVP_FASTER_REV_IN_PLACE

namespace nico::math {
namespace faster_rev {

template <typename T, stan::require_eigen_col_vector_vt<stan::is_var, T>* = nullptr>
inline VecVar nico_log_Phi(const T& x, std::ostream* /*pstream__*/) {

    return rev_apply_log_Phi_storing_derivative(x);

}

template <typename T, stan::require_eigen_col_vector_vt<stan::is_var, T>* = nullptr>
inline VecVar nico_log(const T& x, std::ostream* /*pstream__*/) {

    return rev_apply_log_or_inv_Phi<false>(x);

}

template <typename T, stan::require_eigen_col_vector_vt<stan::is_var, T>* = nullptr>
inline VecVar nico_inv_Phi(const T& x, std::ostream* /*pstream__*/) {

    return rev_apply_log_or_inv_Phi<true>(x);

}

}  // namespace faster_rev
}  // namespace nico::math




// ---- the Stan-facing functions (global scope; stanc calls them as  nico_Phi(x, pstream__)).
#if 0   //// ---- previous macro (var overloads on rev_apply), kept for reference, not compiled:
#define BMVP_STAN_EXTERNAL(NAME, V, D, NEEDED, IN_PLACE)                                                       \
namespace nico::math {                                         \
template <typename T, stan::require_eigen_col_vector_vt<std::is_arithmetic, T>* = nullptr>                     \
inline Eigen::Matrix<double, -1, 1> NAME(const T& x,                                                             \
                                         std::ostream* /*pstream__*/) {                                          \
                                                                                                                \
    return nico::math::prim_apply(x, [](const double* a, int n, double* f) { nico::math::V(a, n, f); }, IN_PLACE); \
                                                                                                                \
}                                                                                                                \
template <typename T, stan::require_eigen_col_vector_vt<stan::is_var, T>* = nullptr>                           \
inline Eigen::Matrix<stan::math::var, -1, 1> NAME(const T& x, std::ostream* /*pstream__*/) {                    \
                                                                                                                \
    return nico::math::rev_apply<nico::math::derivative_values::NEEDED>(                                 \
                                       x, [](const double* a, int n, double* f) { nico::math::V(a, n, f); }, \
                                       [](const double* a, const double* f, int n, double* d) { nico::math::D(a, f, n, d); }, IN_PLACE); \
                                                                                                                \
}                                                                \
}                                                             \
using nico::math::NAME;
#endif
//// ---- the var overloads forward to the strip-mined wrappers in nico::math::faster_rev (above); D and NEEDED
////      stay in the arguments, which the instantiations below still list. The prim overloads are as before.
#define BMVP_STAN_EXTERNAL(NAME, V, D, NEEDED, IN_PLACE)                                                     \
namespace nico::math {                                         \
template <typename T, stan::require_eigen_col_vector_vt<std::is_arithmetic, T>* = nullptr>                   \
inline Eigen::Matrix<double, -1, 1> NAME(const T& x,                                                             \
                                         std::ostream* /*pstream__*/) {                                        \
                                                                                                                \
    return nico::math::prim_apply(x, [](const double* a, int n, double* f) { nico::math::V(a, n, f); },  \
                                     IN_PLACE);                                                               \
                                                                                                                \
}                                                                                                              \
template <typename T, stan::require_eigen_col_vector_vt<stan::is_var, T>* = nullptr>                         \
inline Eigen::Matrix<stan::math::var, -1, 1> NAME(const T& x, std::ostream* pstream__) {                      \
                                                                                                                \
    return nico::math::faster_rev::NAME(x, pstream__);                                                    \
                                                                                                                \
}                                                                \
}                                                             \
using nico::math::NAME;

#define BMVP_IN_PLACE(K) [](double* values, int n) { nico::math::K(values, n); }
BMVP_STAN_EXTERNAL(nico_Phi, v_Phi, d_Phi, operand, BMVP_IN_PLACE(k_Phi))
BMVP_STAN_EXTERNAL(nico_log_Phi, v_log_Phi, d_log_Phi, operand, BMVP_IN_PLACE(k_log_Phi))
BMVP_STAN_EXTERNAL(nico_exp, v_exp, d_exp, result, BMVP_IN_PLACE(k_exp))
BMVP_STAN_EXTERNAL(nico_log1m, v_log1m, d_log1m, operand, BMVP_IN_PLACE(k_log1m))
BMVP_STAN_EXTERNAL(nico_log1p, v_log1p, d_log1p, operand, BMVP_IN_PLACE(k_log1p))
BMVP_STAN_EXTERNAL(nico_log1p_exp, v_log1p_exp, d_log1p_exp, operand, BMVP_IN_PLACE(k_log1p_exp))
// log1m_exp's near-zero repair still needs the original x: do not overwrite it with exp(x).
// (v_log1m_exp is now the SIMD kernel nico_log1m_exp_1, with no scalar repair; the copy path is kept here.)
BMVP_STAN_EXTERNAL(nico_log1m_exp, v_log1m_exp, d_log1m_exp, both, nullptr)
BMVP_STAN_EXTERNAL(nico_inv_logit, v_inv_logit, d_inv_logit, result, BMVP_IN_PLACE(k_inv_logit))
BMVP_STAN_EXTERNAL(nico_log_inv_logit, v_log_inv_logit, d_log_inv_logit, operand, BMVP_IN_PLACE(k_log_inv_logit))
// Keep the copy path here: in-place evaluation regressed complete-call timings in the matched benchmark.
BMVP_STAN_EXTERNAL(nico_Phi_approx, v_Phi_approx, d_Phi_approx, both, nullptr)
BMVP_STAN_EXTERNAL(nico_log_Phi_approx, v_log_Phi_approx, d_log_Phi_approx, both, BMVP_IN_PLACE(k_log_Phi_approx))
BMVP_STAN_EXTERNAL(nico_inv_Phi_approx, v_inv_Phi_approx, d_inv_Phi_approx, both, BMVP_IN_PLACE(k_inv_Phi_approx))
BMVP_STAN_EXTERNAL(nico_inv_Phi_approx_from_logit_prob,
                   v_inv_Phi_approx_from_logit_prob,
                   d_inv_Phi_approx_from_logit_prob,
                   both,
                   BMVP_IN_PLACE(k_inv_Phi_approx_from_logit_prob))
BMVP_STAN_EXTERNAL(nico_tanh, v_tanh, d_tanh, result, BMVP_IN_PLACE(k_tanh))
BMVP_STAN_EXTERNAL(nico_square, v_square, d_square, operand, BMVP_IN_PLACE(k_square))
BMVP_STAN_EXTERNAL(nico_sqrt, v_sqrt, d_sqrt, result, BMVP_IN_PLACE(k_sqrt))
BMVP_STAN_EXTERNAL(nico_logit, v_logit, d_logit, operand, BMVP_IN_PLACE(k_logit))
BMVP_STAN_EXTERNAL(nico_log1m_inv_logit,
                   v_log1m_inv_logit,
                   d_log1m_inv_logit,
                   operand,
                   BMVP_IN_PLACE(k_log1m_inv_logit))
BMVP_STAN_EXTERNAL(nico_expm1, v_expm1, d_expm1, result, BMVP_IN_PLACE(k_expm1))

#undef BMVP_STAN_EXTERNAL
#undef BMVP_IN_PLACE

namespace nico::math {

template <typename T, stan::require_eigen_col_vector_vt<std::is_arithmetic, T>* = nullptr>
inline Eigen::Matrix<double, -1, 1> nico_log(const T& input, std::ostream* /*pstream__*/) {

    return nico::math::prim_apply_log_or_inv_Phi<false>(input);

}

}  // namespace nico::math

using nico::math::nico_log;
namespace nico::math {

template <typename T, stan::require_eigen_col_vector_vt<stan::is_var, T>* = nullptr>
inline nico::math::VecVar nico_log(const T& input, std::ostream* /*pstream__*/) {

    return nico::math::rev_apply_log_or_inv_Phi<false>(input);

}

}  // namespace nico::math

using nico::math::nico_log;
namespace nico::math {

template <typename T, stan::require_eigen_col_vector_vt<std::is_arithmetic, T>* = nullptr>
inline Eigen::Matrix<double, -1, 1> nico_inv_Phi(const T& input, std::ostream* /*pstream__*/) {

    return nico::math::prim_apply_log_or_inv_Phi<true>(input);

}

}  // namespace nico::math

using nico::math::nico_inv_Phi;
namespace nico::math {

template <typename T, stan::require_eigen_col_vector_vt<stan::is_var, T>* = nullptr>
inline nico::math::VecVar nico_inv_Phi(const T& input, std::ostream* /*pstream__*/) {

    return nico::math::rev_apply_log_or_inv_Phi<true>(input);

}

}  // namespace nico::math

using nico::math::nico_inv_Phi;




// -| --------- the exact inverse normal CDF from the log scale: nico_inv_Phi_from_log_p(log_p, log_1m_p) ----------
//
// z = Phi^{-1}(p), given log(p) and log(1 - p) (two vectors of the same length), with BayesMVP's kernel
// nico_inv_Phi_from_log_p_wo_checks (math/nico_and_approx_*_fns.hpp; AS241): in the central region,
// |p - 0.5| <= 0.425 with p = exp(log_p), from p; elsewhere from r = sqrt(-min(log_p, log_1m_p)), negative where
// log_p < log_1m_p. The value depends on log_p in the central region and where log_p < log_1m_p, and on log_1m_p
// elsewhere (p > 0.925); the derivative is taken on the same side, i.e. it is the derivative of the value as it is
// computed, with 1 / phi(z) = exp(0.5 z^2 + 0.5 log(2 pi)):
//
//     log_p side:     dz/dlog_p = exp(log_p + 0.5 z^2 + 0.5 log(2 pi)),       dz/dlog_1m_p = 0
//     log_1m_p side:  dz/dlog_1m_p = -exp(log_1m_p + 0.5 z^2 + 0.5 log(2 pi)), dz/dlog_p = 0
//
// (BayesMVP's own C++ gradient, nico_dZ_dv_from_log_p, takes the side from the sign of z instead; for log_p and
// log_1m_p of the same p both give the same derivative with respect to anything p depends on.)
// The side and both derivatives are computed in the forward pass, in SIMD, and stored on the arena (8 bytes per
// var argument per element); the reverse pass is the adjoint update alone. Either argument may be data.
//
// ---------------------------------------------------------------------------------------------------------------

namespace nico::math {

#if BMVP_LANES == 8
inline void inv_Phi_from_log_p_and_its_derivatives_simd(const __m512d log_p, const __m512d log_1m_p, __m512d& z,
                                                         __m512d& dz_dlog_p, __m512d& dz_dlog_1m_p) {

    ALIGN64  __m512d const log_p_aligned = log_p;
    ALIGN64  __m512d const log_1m_p_aligned = log_1m_p;
    z = nico_inv_Phi_from_log_p_wo_checks_AVX512(log_p_aligned, log_1m_p_aligned);
    // the side of the value, with the kernel's own tests: central, |exp(log_p) - 0.5| <= 0.425; lower, log_p < log_1m_p
    ALIGN64  __m512d const p           = nico_exp_1_AVX512(log_p_aligned);
    const __mmask8 central     = _mm512_cmp_pd_mask(_mm512_abs_pd(_mm512_sub_pd(p, _mm512_set1_pd(0.5))),
                                                    _mm512_set1_pd(0.425), _CMP_LE_OQ);
    const __mmask8 lower       = _mm512_cmp_pd_mask(log_p_aligned, log_1m_p_aligned, _CMP_LT_OQ);
    const __mmask8 uses_log_p  = central | lower;
    ALIGN64  __m512d const log_inv_phi = _mm512_fmadd_pd(_mm512_set1_pd(0.5), _mm512_mul_pd(z, z),
                                                 _mm512_set1_pd(-NEG_HALF_LOG_2PI));
    ALIGN64  __m512d const log_side    = _mm512_mask_blend_pd(uses_log_p, log_1m_p_aligned, log_p_aligned);
    ALIGN64  __m512d const magnitude   = nico_exp_1_AVX512(_mm512_add_pd(log_side, log_inv_phi));
    dz_dlog_p    = _mm512_maskz_mov_pd(uses_log_p, magnitude);
    dz_dlog_1m_p = _mm512_maskz_mov_pd(static_cast<__mmask8>(~uses_log_p),
                                       _mm512_sub_pd(_mm512_setzero_pd(), magnitude));

}
#else
inline void inv_Phi_from_log_p_and_its_derivatives_simd(const __m256d log_p, const __m256d log_1m_p, __m256d& z,
                                                         __m256d& dz_dlog_p, __m256d& dz_dlog_1m_p) {

    ALIGN32  __m256d const log_p_aligned = log_p;
    ALIGN32  __m256d const log_1m_p_aligned = log_1m_p;
    z = nico_inv_Phi_from_log_p_wo_checks_AVX2(log_p_aligned, log_1m_p_aligned);
    // the side of the value, with the kernel's own tests: central, |exp(log_p) - 0.5| <= 0.425; lower, log_p < log_1m_p
    ALIGN32  __m256d const p           = nico_exp_1_AVX2(log_p_aligned);
    ALIGN32  __m256d const central     = _mm256_cmp_pd(_mm256_abs_pd(_mm256_sub_pd(p, _mm256_set1_pd(0.5))),
                                              _mm256_set1_pd(0.425), _CMP_LE_OQ);
    ALIGN32  __m256d const lower       = _mm256_cmp_pd(log_p_aligned, log_1m_p_aligned, _CMP_LT_OQ);
    ALIGN32  __m256d const uses_log_p  = _mm256_or_pd(central, lower);
    ALIGN32  __m256d const log_inv_phi = _mm256_fmadd_pd(_mm256_set1_pd(0.5), _mm256_mul_pd(z, z),
                                                _mm256_set1_pd(-NEG_HALF_LOG_2PI));
    ALIGN32  __m256d const log_side    = _mm256_blendv_pd(log_1m_p_aligned, log_p_aligned, uses_log_p);
    ALIGN32  __m256d const magnitude   = nico_exp_1_AVX2(_mm256_add_pd(log_side, log_inv_phi));
    dz_dlog_p    = _mm256_and_pd(uses_log_p, magnitude);
    dz_dlog_1m_p = _mm256_andnot_pd(uses_log_p, _mm256_sub_pd(_mm256_setzero_pd(), magnitude));

}
#endif

template <typename T_log_p, typename T_log_1m_p>
inline Eigen::Matrix<stan::return_type_t<T_log_p, T_log_1m_p>, -1, 1>
inv_Phi_from_log_p_on_vectors(const T_log_p& log_p, const T_log_1m_p& log_1m_p) {

    constexpr bool log_p_is_var    = stan::is_var<stan::value_type_t<T_log_p>>::value;
    constexpr bool log_1m_p_is_var = stan::is_var<stan::value_type_t<T_log_1m_p>>::value;
    stan::math::check_size_match("nico_inv_Phi_from_log_p", "log_p", log_p.size(), "log_1m_p", log_1m_p.size());
    const int n = static_cast<int>(log_p.size());
    const auto& log_p_evaluated    = stan::math::to_ref(log_p);
    const auto& log_1m_p_evaluated = stan::math::to_ref(log_1m_p);
    Eigen::Matrix<stan::return_type_t<T_log_p, T_log_1m_p>, -1, 1> result(n);

    auto& arena = stan::math::ChainableStack::instance_->memalloc_;
    vari**  log_p_nodes       = log_p_is_var    ? arena.alloc_array<vari*>(n)  : nullptr;
    vari**  log_1m_p_nodes    = log_1m_p_is_var ? arena.alloc_array<vari*>(n)  : nullptr;
    double* dz_dlog_p_values    = log_p_is_var    ? arena.alloc_array<double>(n) : nullptr;
    double* dz_dlog_1m_p_values = log_1m_p_is_var ? arena.alloc_array<double>(n) : nullptr;
    vari*   result_nodes      = (log_p_is_var || log_1m_p_is_var) ? arena.alloc_array<vari>(n) : nullptr;

    // one SIMD vector at a time; a partial final vector is padded with the last valid element, as in apply_kernel
    alignas(64) double log_p_lanes[BMVP_LANES], log_1m_p_lanes[BMVP_LANES];
    alignas(64) double z_lanes[BMVP_LANES], dz_dlog_p_lanes[BMVP_LANES], dz_dlog_1m_p_lanes[BMVP_LANES];
    for (int first = 0; first < n; first += BMVP_LANES) {
        const int count = std::min(BMVP_LANES, n - first);
        for (int lane = 0; lane < BMVP_LANES; ++lane) {
            const int i = first + (lane < count ? lane : count - 1);
            log_p_lanes[lane]    = stan::math::value_of(log_p_evaluated.coeff(i));
            log_1m_p_lanes[lane] = stan::math::value_of(log_1m_p_evaluated.coeff(i));
        }
        simd_t z, dz_dlog_p, dz_dlog_1m_p;
        inv_Phi_from_log_p_and_its_derivatives_simd(simd_load(log_p_lanes), simd_load(log_1m_p_lanes),
                                                    z, dz_dlog_p, dz_dlog_1m_p);
        simd_store(z_lanes, z);
        simd_store(dz_dlog_p_lanes, dz_dlog_p);
        simd_store(dz_dlog_1m_p_lanes, dz_dlog_1m_p);
        for (int lane = 0; lane < count; ++lane) {
            const int i = first + lane;
            if constexpr (log_p_is_var || log_1m_p_is_var) {
                if constexpr (log_p_is_var) {
                    log_p_nodes[i]      = log_p_evaluated.coeff(i).vi_;
                    dz_dlog_p_values[i] = dz_dlog_p_lanes[lane];
                }
                if constexpr (log_1m_p_is_var) {
                    log_1m_p_nodes[i]      = log_1m_p_evaluated.coeff(i).vi_;
                    dz_dlog_1m_p_values[i] = dz_dlog_1m_p_lanes[lane];
                }
                ::new (static_cast<void*>(result_nodes + i)) vari(z_lanes[lane], false);  // nochain stack
                result.coeffRef(i) = var(result_nodes + i);
            } else {
                result.coeffRef(i) = z_lanes[lane];
            }
        }
    }

    if constexpr (log_p_is_var || log_1m_p_is_var) {
        stan::math::reverse_pass_callback([log_p_nodes, log_1m_p_nodes, dz_dlog_p_values, dz_dlog_1m_p_values,
                                           result_nodes, n]() {

            for (int i = 0; i < n; ++i) {
                const double adjoint_of_z = result_nodes[i].adj_;
                if constexpr (log_p_is_var)    log_p_nodes[i]->adj_    += adjoint_of_z * dz_dlog_p_values[i];
                if constexpr (log_1m_p_is_var) log_1m_p_nodes[i]->adj_ += adjoint_of_z * dz_dlog_1m_p_values[i];
            }

        });
    }
    return result;

}

}  // namespace nico::math

namespace nico::math {

template <typename T_log_p, typename T_log_1m_p,
          stan::require_all_eigen_col_vector_t<T_log_p, T_log_1m_p>* = nullptr>
inline Eigen::Matrix<stan::return_type_t<T_log_p, T_log_1m_p>, -1, 1>
nico_inv_Phi_from_log_p(const T_log_p& log_p, const T_log_1m_p& log_1m_p, std::ostream* /*pstream__*/) {

    return nico::math::inv_Phi_from_log_p_on_vectors(log_p, log_1m_p);

}

}  // namespace nico::math

using nico::math::nico_inv_Phi_from_log_p;




// -| --------- the row_vector and matrix overloads of every elementwise function --------------------------------
//
// Each elementwise function above (prim and var) is defined for a column vector. For a row_vector or a matrix
// (or an expression of either), the coefficients are read as one column vector, in Eigen's column-major order (a
// plain row_vector or matrix in place, through a map; an expression evaluated once into a plain matrix), the
// column-vector function is applied (the same kernels, derivatives and tape), and the result is returned with
// the shape of the argument. For var, the coefficients are node pointers: the reshape copies pointers only.
//
// ---------------------------------------------------------------------------------------------------------------

namespace nico::math {

template <typename T, typename F>
inline Eigen::Matrix<stan::value_type_t<T>, std::decay_t<T>::RowsAtCompileTime, std::decay_t<T>::ColsAtCompileTime>
apply_column_vector_function_to_coefficients(const T& x, F column_vector_function) {

    using value_type  = stan::value_type_t<T>;
    using plain_type  = Eigen::Matrix<value_type, std::decay_t<T>::RowsAtCompileTime,
                                      std::decay_t<T>::ColsAtCompileTime>;
    using column_type = Eigen::Matrix<value_type, Eigen::Dynamic, 1>;
    if constexpr (std::is_same<std::decay_t<T>, plain_type>::value) {
        const Eigen::Map<const column_type> x_as_column(x.data(), x.size());
        const column_type result_as_column = column_vector_function(x_as_column);
        return Eigen::Map<const plain_type>(result_as_column.data(), x.rows(), x.cols());
    } else {
        const plain_type x_plain = x;
        const Eigen::Map<const column_type> x_as_column(x_plain.data(), x_plain.size());
        const column_type result_as_column = column_vector_function(x_as_column);
        return Eigen::Map<const plain_type>(result_as_column.data(), x_plain.rows(), x_plain.cols());
    }

}

}  // namespace nico::math

#define BMVP_ROW_VECTOR_AND_MATRIX_OVERLOADS(NAME)                                                               \
namespace nico::math {                                         \
template <typename T, stan::require_eigen_t<T>* = nullptr, stan::require_not_eigen_col_vector_t<T>* = nullptr>   \
inline Eigen::Matrix<stan::value_type_t<T>, std::decay_t<T>::RowsAtCompileTime,                                  \
                     std::decay_t<T>::ColsAtCompileTime>                                                         \
NAME(const T& x, std::ostream* pstream__) {                                                                       \
                                                                                                                \
    return nico::math::apply_column_vector_function_to_coefficients(                                          \
               x, [pstream__](const auto& x_as_column) { return NAME(x_as_column, pstream__); });                \
                                                                                                                \
}                                                                \
}                                                             \
using nico::math::NAME;

BMVP_ROW_VECTOR_AND_MATRIX_OVERLOADS(nico_Phi)
BMVP_ROW_VECTOR_AND_MATRIX_OVERLOADS(nico_log_Phi)
BMVP_ROW_VECTOR_AND_MATRIX_OVERLOADS(nico_inv_Phi)
BMVP_ROW_VECTOR_AND_MATRIX_OVERLOADS(nico_exp)
BMVP_ROW_VECTOR_AND_MATRIX_OVERLOADS(nico_log)
BMVP_ROW_VECTOR_AND_MATRIX_OVERLOADS(nico_log1m)
BMVP_ROW_VECTOR_AND_MATRIX_OVERLOADS(nico_log1p)
BMVP_ROW_VECTOR_AND_MATRIX_OVERLOADS(nico_log1p_exp)
BMVP_ROW_VECTOR_AND_MATRIX_OVERLOADS(nico_log1m_exp)
BMVP_ROW_VECTOR_AND_MATRIX_OVERLOADS(nico_inv_logit)
BMVP_ROW_VECTOR_AND_MATRIX_OVERLOADS(nico_log_inv_logit)
BMVP_ROW_VECTOR_AND_MATRIX_OVERLOADS(nico_Phi_approx)
BMVP_ROW_VECTOR_AND_MATRIX_OVERLOADS(nico_log_Phi_approx)
BMVP_ROW_VECTOR_AND_MATRIX_OVERLOADS(nico_inv_Phi_approx)
BMVP_ROW_VECTOR_AND_MATRIX_OVERLOADS(nico_inv_Phi_approx_from_logit_prob)
BMVP_ROW_VECTOR_AND_MATRIX_OVERLOADS(nico_tanh)
BMVP_ROW_VECTOR_AND_MATRIX_OVERLOADS(nico_square)
BMVP_ROW_VECTOR_AND_MATRIX_OVERLOADS(nico_sqrt)
BMVP_ROW_VECTOR_AND_MATRIX_OVERLOADS(nico_logit)
BMVP_ROW_VECTOR_AND_MATRIX_OVERLOADS(nico_log1m_inv_logit)
BMVP_ROW_VECTOR_AND_MATRIX_OVERLOADS(nico_expm1)

#undef BMVP_ROW_VECTOR_AND_MATRIX_OVERLOADS

// ---- diagnostic: which kernel path this build took. Declare in Stan as   real bmvp_simd_lanes();
namespace nico::math {

inline double bmvp_simd_lanes(std::ostream* /*pstream__*/) { return static_cast<double>(BMVP_LANES); }

}  // namespace nico::math

using nico::math::bmvp_simd_lanes;

// ---- the fused log densities (each also includes this file itself, under #pragma once, so either may come first)
#include "stan_external_fused_normal_lpdf.hpp"
#include "stan_external_fused_student_t_lpdf.hpp"
#include "stan_external_fused_weibull_lpdf.hpp"
#include "stan_external_fused_ordered_logistic_lpmf.hpp"
#include "stan_external_fused_bernoulli_logit_lpmf.hpp"
// left out while stan_external_fused_bernoulli_logit_lpmf.hpp is being included (it includes this file before its
// own namespace, which the GLM header needs); every other entry point gets the GLM header here
#ifndef BMVP_FUSED_BERNOULLI_LOGIT_LPMF_BEING_INCLUDED
#include "stan_external_fused_bernoulli_logit_and_probit_glm_lpmf.hpp"
// (the Student-t identity GLM, the Weibull log GLM and the Bernoulli-logit GLM with coefficients by group,
//  which use the GLM header's code)
#include "stan_external_fused_student_t_id_glm_weibull_log_glm_and_bernoulli_logit_glm_by_group.hpp"
// (the Student-t regression and the grouped Bernoulli-logit regression, private until now, timed
//  against the GLMs above: the faster of each is the one the models use)
#include "stan_external_fused_student_t_regression_lpdf.hpp"
#include "stan_external_fused_bernoulli_logit_grouped_regression_lpmf.hpp"
// (the running version of log_sum_exp, one autodiff step for the whole vector)
#include "stan_external_log_cumulative_sum_exp.hpp"
// (any multi-exponential curve with each group's own coefficients and exponents, one autodiff step)
#include "stan_external_sum_of_exponentials_by_group.hpp"
// (the weighted sum of exponentials of an affine predictor by column, e.g. any cumulative hazard by quadrature
//  with subject effects, one autodiff step)
#include "stan_external_dot_product_exp_of_affine_predictor_by_column.hpp"
#include "stan_external_AR1_path.hpp"
#include "stan_external_sum_log_square_plus.hpp"
// (the fused normal with a log scale)
#include "stan_external_fused_normal_log_scale_lpdf.hpp"
#include "stan_external_fused_affine_compositions.hpp"
// (the dot products of an affine predictor with indexed group effects and its log cumulative sum of
//  exponentials, e.g. the partial likelihood of any proportional-hazards model with group effects, one autodiff
//  step)
#include "stan_external_dot_products_of_affine_predictor_with_indexed_effects_and_log_cumulative_sum_exp.hpp"
#endif























// ---- log_sum_exp: the reduction, elementwise pairs and row-wise reduction with the Nico exponential.
#include "stan_external_log_sum_exp_with_nico_exponential.hpp"

// ---- the squared exponential covariance of data points (as Stan's gp_exp_quad_cov): stan_external_gp_exp_quad_cov.hpp
#include "stan_external_gp_exp_quad_cov.hpp"

// ---- the GHK log likelihood of a finite mixture of multivariate (ordered) probit models (C = 1: the MVP / MVOP;
//      C = 2, 4: the LC-MVP and LC-MVOP), one autodiff node per slice:
//      stan_external_finite_mixture_of_multivariate_ordered_probit_GHK_log_likelihood.hpp
#include "stan_external_finite_mixture_of_multivariate_ordered_probit_GHK_log_likelihood.hpp"























