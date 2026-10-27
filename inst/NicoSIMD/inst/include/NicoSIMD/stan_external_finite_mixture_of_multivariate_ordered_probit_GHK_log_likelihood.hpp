#pragma once
//
// -| --------- BayesMVP GHK log likelihood of a finite mixture of multivariate (ordered) probit models ----------
// -| --------- (Stan external function) -------------------------------------------------------------------------
//
//     nico_finite_mixture_of_multivariate_ordered_probit_GHK_log_likelihood(u_raw, nuisance_transformation,
//           intercept, coefficients, coefficient_set_of_class_and_test, covariates, n_covariates, cutpoints,
//           cutpoint_set_of_class_and_test, position_of_first_cutpoint_of_test, L_Omega, log_class_probability,
//           category, upper_category, test_is_binary, n_categories_of_test, test_at_GHK_position, Phi_type,
//           overflow_threshold, underflow_threshold)
//
// The model: C latent classes (C = 1: the multivariate probit / ordered probit model without latent classes;
// C = 2 and C = 4: the LC-MVP and the LC-MVOP), T tests, each binary (y = 1 when the latent value is above 0) or
// ordinal (category k when the latent value is between the cutpoints k - 1 and k; an interval-censored result,
// categories k, ..., k_upper, when it is between the cutpoints k - 1 and k_upper). In class c the latent values
// of a subject are normal, with mean eta_c (one per test) and correlation matrix L_Omega[c] L_Omega[c]', and
//     eta_c[t] = intercept[c, t] + covariates[., t][m, 1:P] * coefficients[set of (c, t)][1:P, t].
// The log likelihood of the M subjects of the slice, by the GHK method with the nuisance values u (one per
// subject and GHK position, u = Phi(u_raw) or u = (1 + tanh(u_raw)) / 2): at GHK position j, test t =
// test_at_GHK_position[j], with inc = sum_{k < j} L_Omega[c][j, k] Z[k] and s = 1 / L_Omega[c][j, j],
//     a = (lower bound - eta_c[t] - inc) s,   b = (upper bound - eta_c[t] - inc) s,
//     Z[j] = F^{-1}((1 - u) F(a) + u F(b)),    log mass[j] = log(F(b) - F(a)),
// with F the exact Phi (Phi_type 1) or Phi_approx (Phi_type 2); a missing result (category < 0) is by the GHK
// method too, the untruncated draw Z[j] = F^{-1}(u) in its place in the conditioning chain and no likelihood
// term.
// The value is
//     sum_m log sum_c exp(log_class_probability[m, c] + sum_j log mass_c[m, j])  +  sum log |du / du_raw|.
//
// Each step is computed as in BayesMVP's C++ (MVP_log_scale_grad_calc_fns.hpp and
// MVOP_PartialLog_helper_fns.hpp) and the LC-MVP / LC-MVOP Stan models, each Phi_type with its own functions:
// the rows with b < underflow_threshold (both bounds in the lower tail; for a binary test, y = 0 and an
// underflow) and with a > overflow_threshold (both in the upper tail; y = 1 and an overflow) on the log scale,
// with log F(x) (log_Phi or log_Phi_approx) and the inverse from the log scale (inv_Phi_from_log_p or
// inv_Phi_approx_from_logit_prob); the other rows directly, for Phi_type 1 reflected (Phi(x) = 1 - Phi(-x))
// where a > 0 or b = +inf, so that the smaller tail mass is the one formed, and for Phi_type 2 by the stable
// log-scale form of the four-class LC-MVOP; and the rows whose interval is narrower than 1e-5 / (1 + |mid|)
// (both bounds finite) by the four-class LC-MVOP's Gauss-Legendre mass and second-order quantile. An infinite
// bound is exactly infinite here (the Stan models' C_sentinel = 1000 gives the same values).
//
// ONE autodiff node; its partial derivatives are made in the forward pass (the result is one number). Every
// branch computes the same function, so the partial derivatives of each step are those of the function itself,
// with f the density of F, each formed on the log scale (0 for an infinite bound):
//     dZ/da = (1 - u) f(a) / f(Z),   dZ/db = u f(b) / f(Z),   dZ/du_raw = (F(b) - F(a)) |du/du_raw| / f(Z),
//     dlog mass/da = -f(a) / (F(b) - F(a)),   dlog mass/db = f(b) / (F(b) - F(a)),
// and the reverse sweep of the GHK recursion (from the last GHK position to the first, for each class) is done in
// the same call, seeded by the class weights softmax_c of each subject. Every transcendental function is one of
// BayesMVP's SIMD kernels on a whole buffer (AVX-512: 8 lanes; AVX2: 4); every other per-element operation is a
// loop over contiguous buffers without branches, which -O3 -march=native vectorises; the only other loops over
// the subjects are the gathers and scatters of the rows of each branch and the sums of the cutpoints' partials by
// category. Thread-local buffers (one set per thread of reduce_sum); about 8 C T + 9 T + 40 doubles per subject.
//
// Arguments (M = the subjects of the slice; S_B, S_C = the numbers of coefficient and cutpoint sets):
//   u_raw                                 array[M] row_vector[T]: the nuisance parameters, columns in GHK order
//   nuisance_transformation               1: u = Phi(u_raw), log Jacobian -u_raw^2 / 2 - log(2 pi) / 2;
//                                         2: u = (1 + tanh(u_raw)) / 2, log Jacobian log(2 u (1 - u))
//   intercept                             matrix[C, T]: the intercept of each class and test (tests in data
//                                         order)
//   coefficients                          array[S_B] matrix[P_max, T]: the coefficient sets of the covariates
//   coefficient_set_of_class_and_test     array[C, T] int: the coefficient set (1, ..., S_B) of each class and
//                                         test
//   covariates                            array[S_X, T] matrix[M, P_max]: the covariates of the slice's subjects,
//                                         S_X = 1 (the same for every set) or S_X = S_B (one per set); S_X = 0 or
//                                         P_max = 0: no covariates
//   n_covariates                          array[S_X, T] int: the covariates of each test (the first n columns)
//   cutpoints                             array[S_C] vector: the cutpoint sets, each with every ordinal test's
//                                         cutpoints one after the other
//   cutpoint_set_of_class_and_test        array[C, T] int: the cutpoint set (1, ..., S_C) of each class and test
//   position_of_first_cutpoint_of_test    array[T] int: the position in a set of test t's first cutpoint
//                                         (ignored for a binary test)
//   L_Omega                               array[C] matrix[T, T]: the Cholesky factors, rows and columns in GHK
//                                         order
//   log_class_probability                 matrix[M, C]: the log probability of each class for each subject
//   category                              array[M, T] int: the result of each subject and test (binary: 0 or 1;
//                                         ordinal: 1, ..., K_t); any negative value: missing
//   upper_category                        array[M, T] int: the upper category of an interval-censored ordinal
//                                         result (= category otherwise; ignored for binary and missing results)
//   test_is_binary                        array[T] int: 1 binary, 0 ordinal
//   n_categories_of_test                  array[T] int: K_t (2 for a binary test)
//   test_at_GHK_position                  array[T] int: the test at each GHK position (1:T: the data order)
//   Phi_type                              1: exact Phi; 2: Phi_approx
//   overflow_threshold, underflow_threshold   the bounds of the log-scale tails (as in the models)
//
// Declared in the Stan functions block (no body; stanc --allow-undefined):
//
//     real nico_finite_mixture_of_multivariate_ordered_probit_GHK_log_likelihood(
//           array[] row_vector u_raw, data int nuisance_transformation, matrix intercept,
//           array[] matrix coefficients, data array[,] int coefficient_set_of_class_and_test,
//           data array[,] matrix covariates, data array[,] int n_covariates, array[] vector cutpoints,
//           data array[,] int cutpoint_set_of_class_and_test,
//           data array[] int position_of_first_cutpoint_of_test,
//           array[] matrix L_Omega, matrix log_class_probability, data array[,] int category,
//           data array[,] int upper_category, data array[] int test_is_binary,
//           data array[] int n_categories_of_test, data array[] int test_at_GHK_position, data int Phi_type,
//           data real overflow_threshold, data real underflow_threshold);
//
// Included at the end of stan_external_functions.hpp.
//
// -----------------------------------------------------------------------------------------------------------

#include "stan_external_functions.hpp"

#include <stan/math/rev.hpp>
#include <stan/math/prim.hpp>

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace nico::math {

//// ---- the thread-local buffers (each grows to the largest slice seen, then is reused)
struct buffers_of_mixture_of_multivariate_ordered_probit_GHK_t {
    std::vector<double> values_of_each_subject_and_position;
    std::vector<double> values_of_each_class_subject_and_position;
    std::vector<double> values_of_each_subject_and_class;
    std::vector<double> values_of_each_subject;
    std::vector<double> values_of_each_set_test_and_subject;
    std::vector<int>    codes_of_each_subject_and_test;
    std::vector<int>    rows_of_each_branch;
};

inline buffers_of_mixture_of_multivariate_ordered_probit_GHK_t&
buffers_of_mixture_of_multivariate_ordered_probit_GHK() {

    static thread_local buffers_of_mixture_of_multivariate_ordered_probit_GHK_t buffers;
    return buffers;

}

template <typename T_value>
inline T_value* buffer_of_at_least(std::vector<T_value>& buffer, const std::size_t size) {

    if (buffer.size() < size) buffer.resize(size);
    return buffer.data();

}

//// ---- the code of a bound: >= 0, the position of its cutpoint in the cutpoint set; -1, infinite; -2, the fixed
////      cutpoint 0 of a binary test
static constexpr int CODE_OF_AN_INFINITE_BOUND = -1;
static constexpr int CODE_OF_THE_CUTPOINT_ZERO_OF_A_BINARY_TEST = -2;

//// ---- the four kinds of rows of a GHK step (the missing results are the fifth, outside the branches)
static constexpr int ROWS_DIRECTLY = 0;
static constexpr int ROWS_IN_THE_LOWER_TAIL = 1;
static constexpr int ROWS_IN_THE_UPPER_TAIL = 2;
static constexpr int ROWS_OF_A_NARROW_INTERVAL = 3;

//// ---- the Gauss-Legendre rule of the narrow intervals (as the four-class LC-MVOP's narrow_interval_logprob)
static constexpr double NARROW_INTERVAL_NODES[4] = { -0.8611363115940526, -0.3399810435848563,
                                                      0.3399810435848563,  0.8611363115940526 };
static constexpr double NARROW_INTERVAL_WEIGHTS[4] = { 0.1739274225687269, 0.3260725774312731,
                                                        0.3260725774312731, 0.1739274225687269 };
static constexpr double LARGEST_WIDTH_OF_A_NARROW_INTERVAL = 1e-5;

//// ---- the kernels of this function in double precision: BayesMVP's nico_log1p_1, nico_log_inv_logit,
////      nico_log1m_exp_1, nico_log_Phi, nico_inv_Phi_wo_checks and the asinh of
////      inv_Phi_approx_from_logit_prob take a single-precision-accuracy log (about 1.5e-8 relative error, see
////      math/nico_and_approx_*_fns.hpp); the same functions here take BayesMVP's double-precision log and log1p
////      (nico_log_1_double_precision, nico_log1p_1_double_precision) instead, with the same other kernels
////      (nico_exp_1, nico_Phi, nico_inverse_mills_ratio_tail, the AS241 cases of nico_inv_Phi_wo_checks)
#if defined(NICO_SIMD_SCALAR_KERNELS)
using simd_mask_t = bool;
inline simd_t simd_of(const double x) {

    return x;

}
inline simd_t simd_absolute_value_of(const simd_t x) {

    return std::fabs(x);

}
inline simd_mask_t simd_less_than(const simd_t x,
                                  const simd_t y) {

    return x < y;

}
inline simd_mask_t simd_greater_than(const simd_t x,
                                     const simd_t y) {

    return x > y;

}
inline simd_mask_t simd_less_than_or_equal(const simd_t x,
                                           const simd_t y) {

    return x <= y;

}
inline simd_t simd_where(const simd_mask_t mask,
                         const simd_t if_true,
                         const simd_t if_false) {

    return mask ? if_true : if_false;

}
inline bool simd_no_lane(const simd_mask_t mask) {

    return !mask;

}
inline bool simd_every_lane(const simd_mask_t mask) {

    return mask;

}
#elif BMVP_LANES == 8
using simd_mask_t = __mmask8;
inline simd_t simd_of(const double x) { return _mm512_set1_pd(x); }
inline simd_t simd_absolute_value_of(const simd_t x) { return _mm512_abs_pd(x); }
inline simd_mask_t simd_less_than(const simd_t x, const simd_t y) { return _mm512_cmp_pd_mask(x, y, _CMP_LT_OQ); }
inline simd_mask_t simd_greater_than(const simd_t x, const simd_t y) {

    return _mm512_cmp_pd_mask(x, y, _CMP_GT_OQ);

}
inline simd_mask_t simd_less_than_or_equal(const simd_t x, const simd_t y) {

    return _mm512_cmp_pd_mask(x, y, _CMP_LE_OQ);

}
inline simd_t simd_where(const simd_mask_t mask, const simd_t if_true, const simd_t if_false) {

    return _mm512_mask_blend_pd(mask, if_false, if_true);

}
inline bool simd_no_lane(const simd_mask_t mask) { return mask == 0; }
inline bool simd_every_lane(const simd_mask_t mask) { return mask == 0xff; }
#else
using simd_mask_t = __m256d;
inline simd_t simd_of(const double x) { return _mm256_set1_pd(x); }
inline simd_t simd_absolute_value_of(const simd_t x) { return _mm256_andnot_pd(_mm256_set1_pd(-0.0), x); }
inline simd_mask_t simd_less_than(const simd_t x, const simd_t y) { return _mm256_cmp_pd(x, y, _CMP_LT_OQ); }
inline simd_mask_t simd_greater_than(const simd_t x, const simd_t y) { return _mm256_cmp_pd(x, y, _CMP_GT_OQ); }
inline simd_mask_t simd_less_than_or_equal(const simd_t x, const simd_t y) {

    return _mm256_cmp_pd(x, y, _CMP_LE_OQ);

}
inline simd_t simd_where(const simd_mask_t mask, const simd_t if_true, const simd_t if_false) {

    return _mm256_blendv_pd(if_false, if_true, mask);

}
inline bool simd_no_lane(const simd_mask_t mask) { return _mm256_movemask_pd(mask) == 0; }
inline bool simd_every_lane(const simd_mask_t mask) { return _mm256_movemask_pd(mask) == 0xf; }
#endif
inline simd_t simd_sum(const simd_t x, const simd_t y) {

#if defined(NICO_SIMD_SCALAR_KERNELS)
    return x + y;
#elif BMVP_LANES == 8
    return _mm512_add_pd(x, y);
#else
    return _mm256_add_pd(x, y);
#endif

}
inline simd_t simd_difference(const simd_t x, const simd_t y) {

#if defined(NICO_SIMD_SCALAR_KERNELS)
    return x - y;
#elif BMVP_LANES == 8
    return _mm512_sub_pd(x, y);
#else
    return _mm256_sub_pd(x, y);
#endif

}
inline simd_t simd_product(const simd_t x, const simd_t y) {

#if defined(NICO_SIMD_SCALAR_KERNELS)
    return x * y;
#elif BMVP_LANES == 8
    return _mm512_mul_pd(x, y);
#else
    return _mm256_mul_pd(x, y);
#endif

}
inline simd_t simd_quotient(const simd_t x, const simd_t y) {

#if defined(NICO_SIMD_SCALAR_KERNELS)
    return x / y;
#elif BMVP_LANES == 8
    return _mm512_div_pd(x, y);
#else
    return _mm256_div_pd(x, y);
#endif

}
inline simd_t simd_minimum(const simd_t x, const simd_t y) {

#if defined(NICO_SIMD_SCALAR_KERNELS)
    return (x < y) ? x : y;
#elif BMVP_LANES == 8
    return _mm512_min_pd(x, y);
#else
    return _mm256_min_pd(x, y);
#endif

}
inline simd_t simd_maximum(const simd_t x, const simd_t y) {

#if defined(NICO_SIMD_SCALAR_KERNELS)
    return (x > y) ? x : y;
#elif BMVP_LANES == 8
    return _mm512_max_pd(x, y);
#else
    return _mm256_max_pd(x, y);
#endif

}
inline simd_t simd_square_root(const simd_t x) {

#if defined(NICO_SIMD_SCALAR_KERNELS)
    return std::sqrt(x);
#elif BMVP_LANES == 8
    return _mm512_sqrt_pd(x);
#else
    return _mm256_sqrt_pd(x);
#endif

}
inline simd_t simd_log_in_double_precision(const simd_t x) {

#if defined(NICO_SIMD_SCALAR_KERNELS)
    return std::log(x);
#else
    return BMVP_KERNEL_NAME(nico_log_1_double_precision)(x);
#endif

}
inline simd_t simd_log1p_in_double_precision(const simd_t x) {

#if defined(NICO_SIMD_SCALAR_KERNELS)
    return std::log1p(x);
#else
    return BMVP_KERNEL_NAME(nico_log1p_1_double_precision)(x);
#endif

}
//// log_inv_logit(x) = min(x, 0) - log1p(exp(-|x|))
inline simd_t simd_log_inv_logit_in_double_precision(const simd_t x) {

#if defined(NICO_SIMD_SCALAR_KERNELS)
    return stan::math::log_inv_logit(x);
#else
    const simd_t exp_of_minus_absolute_x = BMVP_KERNEL_NAME(nico_exp_1)(
          simd_difference(simd_of(0.0), simd_absolute_value_of(x)));
    return simd_difference(simd_minimum(x, simd_of(0.0)),
                           simd_log1p_in_double_precision(exp_of_minus_absolute_x));
#endif

}
//// log1m_exp(x) = log(1 - exp(x)), x <= 0: log1p(-exp(x)) for x < -ln 2, log(-expm1(x)) nearer 0 (BayesMVP's
//// expm1, simd_expm1_of of stan_external_functions.hpp)
inline simd_t simd_log1m_exp_in_double_precision(const simd_t x) {

#if defined(NICO_SIMD_SCALAR_KERNELS)
    return stan::math::log1m_exp(x);
#else
    const simd_mask_t far_from_zero = simd_less_than(x, simd_of(-LN2));
    simd_t far_result = simd_of(0.0);
    simd_t near_result = simd_of(0.0);
    if (!simd_no_lane(far_from_zero)) {
        far_result = simd_log1p_in_double_precision(simd_difference(simd_of(0.0),
                                                                    BMVP_KERNEL_NAME(nico_exp_1)(x)));
    }
    if (!simd_every_lane(far_from_zero)) {
        near_result = simd_log_in_double_precision(simd_difference(simd_of(0.0), simd_expm1_of(x)));
    }
    return simd_where(far_from_zero, far_result, near_result);
#endif

}
//// log Phi(x): log(nico_Phi(x)) for x >= -2.5; -x^2 / 2 - log(2 pi) / 2 - log(1 / R(-x)) below (R the Mills
//// ratio, nico_inverse_mills_ratio_tail), as nico_log_Phi
inline simd_t simd_log_Phi_in_double_precision(const simd_t x) {

#if defined(NICO_SIMD_SCALAR_KERNELS)
    if (x < -2.5) {
        const double t = -x;
        double f = t;
        for (int k_reversed = 1; k_reversed <= 40; ++k_reversed) f = t + (41 - k_reversed) / f;
        return -0.5 * x * x + (-0.91893853320467274178) + std::log(1.0 / f);
    }
    return std::log(0.5 * std::erfc(-x * 0.70710678118654752440));
#else
    const simd_mask_t in_the_tail = simd_less_than(x, simd_of(-NORMAL_TAIL_THRESH));
    simd_t body = simd_of(0.0);
    simd_t tail = simd_of(0.0);
    if (!simd_every_lane(in_the_tail)) body = simd_log_in_double_precision(BMVP_KERNEL_NAME(nico_Phi)(x));
    if (!simd_no_lane(in_the_tail)) {
        const simd_t inverse_Mills_ratio = BMVP_KERNEL_NAME(nico_inverse_mills_ratio_tail)(
              simd_maximum(simd_absolute_value_of(x), simd_of(NORMAL_TAIL_THRESH)));
        tail = simd_difference(simd_sum(simd_product(simd_of(-0.5), simd_product(x, x)),
                                        simd_of(NEG_HALF_LOG_2PI)),
                               simd_log_in_double_precision(inverse_Mills_ratio));
    }
    return simd_where(in_the_tail, tail, body);
#endif

}
//// Phi^{-1}(p) by AS241, as nico_inv_Phi_wo_checks: the central case for |p - 1/2| <= 0.425, otherwise the tail
//// cases of r = sqrt(-log(min(p, 1 - p)))
inline simd_t simd_inv_Phi_in_double_precision(const simd_t p) {

#if defined(NICO_SIMD_SCALAR_KERNELS)
    return stan::math::inv_Phi(p);
#else
    const simd_t q = simd_difference(p, simd_of(0.5));
    const simd_mask_t central = simd_less_than_or_equal(simd_absolute_value_of(q), simd_of(0.425));
    const simd_t central_value = BMVP_KERNEL_NAME(nico_inv_Phi_wo_checks_case_1)(p, q);
    if (simd_every_lane(central)) return central_value;
    const simd_mask_t q_is_positive = simd_greater_than(q, simd_of(0.0));
    const simd_t smaller_tail = simd_where(q_is_positive, simd_difference(simd_of(1.0), p), p);
    const simd_t r = simd_square_root(simd_difference(simd_of(0.0), simd_log_in_double_precision(smaller_tail)));
    const simd_mask_t r_is_above_5 = simd_greater_than(r, simd_of(5.0));
    simd_t tail_value = BMVP_KERNEL_NAME(nico_inv_Phi_wo_checks_case_2a)(p, r);
    if (!simd_no_lane(r_is_above_5)) {
        tail_value = simd_where(r_is_above_5, BMVP_KERNEL_NAME(nico_inv_Phi_wo_checks_case_2b)(p, r), tail_value);
    }
    tail_value = simd_where(simd_less_than(q, simd_of(0.0)), simd_difference(simd_of(0.0), tail_value),
                            tail_value);
    return simd_where(central, central_value, tail_value);
#endif

}
//// inv_Phi_approx_from_logit_prob(L) = C2 sinh(asinh(C1 L) / 3), as simd_inv_Phi_approx_from_logit_prob_of of
//// stan_external_functions.hpp, with the double-precision log1p in its asinh
inline simd_t simd_inv_Phi_approx_from_logit_prob_in_double_precision(const simd_t logit_p) {

#if defined(NICO_SIMD_SCALAR_KERNELS)
    return 5.494448514153059 * std::sinh(0.33333333333333331483 * std::asinh(0.34176618822627863 * logit_p));
#else
    static constexpr double odd_inverse_factorials[8] = { 1.0 / 1307674368000.0, 1.0 / 6227020800.0,
                                                          1.0 / 39916800.0, 1.0 / 362880.0, 1.0 / 5040.0,
                                                          1.0 / 120.0, 1.0 / 6.0, 1.0 };
    const simd_t one = simd_of(1.0);
    const simd_t s = simd_absolute_value_of(simd_product(simd_of(INV_PHI_APPROX_C1), logit_p));
    const simd_t s_squared = simd_product(s, s);
    const simd_t log1p_argument = simd_sum(s, simd_quotient(s_squared, simd_sum(one, simd_square_root(
                                                                                 simd_sum(one, s_squared)))));
    const simd_t two_s_minus_one = simd_difference(simd_product(simd_of(2.0), s), one);
    const simd_mask_t large = simd_greater_than(s, simd_of(1e8));
    const simd_t y = simd_product(simd_of(1.0 / 3.0), simd_log1p_in_double_precision(
                                                            simd_where(large, two_s_minus_one, log1p_argument)));
    const simd_t y_squared = simd_product(y, y);
    simd_t series = simd_of(odd_inverse_factorials[0]);
    for (int k = 1; k < 8; ++k) {
        series = simd_sum(simd_product(series, y_squared), simd_of(odd_inverse_factorials[k]));
    }
    series = simd_product(series, y);
    const simd_t exp_y = BMVP_KERNEL_NAME(nico_exp_1)(y);
    const simd_t from_exp = simd_product(simd_of(0.5), simd_difference(exp_y, simd_quotient(one, exp_y)));
    const simd_t sinh_y = simd_where(simd_less_than_or_equal(y, simd_of(0.5)), series, from_exp);
    const simd_t value = simd_product(simd_of(INV_PHI_APPROX_C2), sinh_y);
    return simd_where(simd_less_than(logit_p, simd_of(0.0)), simd_difference(simd_of(0.0), value), value);
#endif

}
inline void k_log1p_in_double_precision(double* p, const int n) {

    apply_kernel(p, n, [](const simd_t v) { return simd_log1p_in_double_precision(v); });

}
inline void k_log_inv_logit_in_double_precision(double* p, const int n) {

    apply_kernel(p, n, [](const simd_t v) { return simd_log_inv_logit_in_double_precision(v); });

}
inline void k_log1m_exp_in_double_precision(double* p, const int n) {

    apply_kernel(p, n, [](const simd_t v) { return simd_log1m_exp_in_double_precision(v); });

}
inline void k_log_Phi_in_double_precision(double* p, const int n) {

    apply_kernel(p, n, [](const simd_t v) { return simd_log_Phi_in_double_precision(v); });

}
inline void k_inv_Phi_in_double_precision(double* p, const int n) {

    apply_kernel(p, n, [](const simd_t v) { return simd_inv_Phi_in_double_precision(v); });

}
inline void k_inv_Phi_approx_from_logit_prob_in_double_precision(double* p, const int n) {

    apply_kernel(p, n, [](const simd_t v) { return simd_inv_Phi_approx_from_logit_prob_in_double_precision(v); });

}

//// ---- z = Phi^{-1}(p) from log(p) and log(1 - p), in SIMD (BayesMVP's AS241 kernel), on n doubles; a partial
////      final SIMD vector is padded with the last valid element, as in apply_kernel
#if defined(NICO_SIMD_SCALAR_KERNELS)
inline double log_Phi_stable(const double x) {

    if (x < -10.0) {
        const double t = -x;
        double r = 0.0;
        for (int k = 1; k <= 20; ++k) r = (21.0 - k) / (t + r);
        return -0.5 * stan::math::square(x) - 0.5 * std::log(2.0 * stan::math::pi()) - std::log(t + r);
    }
    if (x <= 0.0) return std::log(stan::math::erfc(-x / std::sqrt(2.0))) - std::log(2.0);
    return stan::math::log1m(0.5 * stan::math::erfc(x / std::sqrt(2.0)));

}
inline double inv_Phi_from_log_lower(const double log_p) {

    double z = -std::sqrt(-2.0 * log_p);
    for (int i = 1; i <= 5; ++i) {
        const double lp = log_Phi_stable(z);
        z -= (lp - log_p) * std::exp(lp - stan::math::std_normal_lpdf(z));
    }
    return z;

}
inline double normal_from_log_uniform(const double log_u,
                                      const double log1m_u) {

    if (log_u <= -std::log(2.0)) return inv_Phi_from_log_lower(log_u);
    return -inv_Phi_from_log_lower(log1m_u);

}
#endif
inline void k_inv_Phi_from_log_p_of_two_buffers(const double* log_p, const double* log_1m_p, const int n,
                                                double* z) {

#if defined(NICO_SIMD_SCALAR_KERNELS)
    for (int i = 0; i < n; ++i) z[i] = normal_from_log_uniform(log_p[i], log_1m_p[i]);
#else
    int i = 0;
    for (; i + BMVP_LANES <= n; i += BMVP_LANES) {
        simd_storeu(z + i, BMVP_KERNEL_NAME(nico_inv_Phi_from_log_p_wo_checks)(simd_loadu(log_p + i),
                                                                               simd_loadu(log_1m_p + i)));
    }
    if (i < n) {
        alignas(64) double padded_log_p[BMVP_LANES];
        alignas(64) double padded_log_1m_p[BMVP_LANES];
        const int remaining = n - i;
        for (int lane = 0; lane < BMVP_LANES; ++lane) {
            padded_log_p[lane]    = log_p[i + (lane < remaining ? lane : remaining - 1)];
            padded_log_1m_p[lane] = log_1m_p[i + (lane < remaining ? lane : remaining - 1)];
        }
        simd_store(padded_log_p, BMVP_KERNEL_NAME(nico_inv_Phi_from_log_p_wo_checks)(simd_load(padded_log_p),
                                                                                     simd_load(padded_log_1m_p)));
        for (int lane = 0; lane < remaining; ++lane) z[i + lane] = padded_log_p[lane];
    }
#endif

}

//// ---- log(exp(x) + exp(y)) on n doubles (both finite): max + log1p(exp(min - max)); work: n doubles
inline void log_sum_exp_of_two_buffers(const double* x, const double* y, const int n, double* out,
                                       double* work) {

    for (int i = 0; i < n; ++i) work[i] = -std::fabs(x[i] - y[i]);
    k_exp(work, n);
    k_log1p_in_double_precision(work, n);
    for (int i = 0; i < n; ++i) out[i] = std::max(x[i], y[i]) + work[i];

}

//// ---- log F(x) on n doubles in place: log Phi(x) (Phi_type 1) or log Phi_approx(x) =
////      log_inv_logit(0.07056 x^3 + 1.5976 x) (Phi_type 2)
inline void k_log_F(double* x, const int n, const int Phi_type) {

    if (Phi_type == 1) {
        k_log_Phi_in_double_precision(x, n);
    } else {
        for (int i = 0; i < n; ++i) x[i] = (PHI_APPROX_A * x[i] * x[i] + PHI_APPROX_B) * x[i];
        k_log_inv_logit_in_double_precision(x, n);
    }

}

//// ---- the log density log f(x) on n doubles in place: -x^2 / 2 - log(2 pi) / 2 (Phi_type 1), or, with p =
////      0.07056 x^3 + 1.5976 x, log(0.21168 x^2 + 1.5976) + log_inv_logit(p) + log_inv_logit(-p) (Phi_type 2);
////      work: 2 n doubles
inline void k_log_density_of_F(double* x, const int n, const int Phi_type, double* work) {

    if (Phi_type == 1) {
        for (int i = 0; i < n; ++i) x[i] = -0.5 * x[i] * x[i] + NEG_HALF_LOG_2PI;
        return;
    }
    double* derivative_of_p = work;
    double* minus_p = work + n;
    for (int i = 0; i < n; ++i) {
        derivative_of_p[i] = 3.0 * PHI_APPROX_A * x[i] * x[i] + PHI_APPROX_B;
        x[i] = (PHI_APPROX_A * x[i] * x[i] + PHI_APPROX_B) * x[i];
        minus_p[i] = -x[i];
    }
    k_log(derivative_of_p, n);
    k_log_inv_logit_in_double_precision(x, n);
    k_log_inv_logit_in_double_precision(minus_p, n);
    for (int i = 0; i < n; ++i) x[i] += derivative_of_p[i] + minus_p[i];

}

//// ---- Z = F^{-1} from log F(Z) and log(1 - F(Z)), on n doubles
inline void inverse_of_F_from_the_log_scale(const double* log_F, const double* log_1m_F, const int n,
                                            const int Phi_type, double* Z) {

    if (Phi_type == 1) {
        k_inv_Phi_from_log_p_of_two_buffers(log_F, log_1m_F, n, Z);
    } else {
        for (int i = 0; i < n; ++i) Z[i] = log_F[i] - log_1m_F[i];
        k_inv_Phi_approx_from_logit_prob_in_double_precision(Z, n);
    }

}

//// ---- the rows of a GHK step computed directly, Phi_type 1: reflected where a > 0 or b = +inf,
////      (a, b, u) -> (-b, -a, 1 - u) and Z -> -Z; prob = Phi(b) - Phi(a), Z = Phi^{-1}(Phi(a) + u prob),
////      log mass = log(prob) (the bounds given to Phi are capped at +-40, where Phi is exactly 0 or 1);
////      work: 3 n
inline void GHK_rows_directly_with_Phi(const int n, const double* a, const double* b, const double* u,
                                       const double* one_minus_u, double* Z, double* log_mass, double* work) {

    const double infinity = std::numeric_limits<double>::infinity();
    double* Phi_of_a = work;
    double* Phi_of_b = work + n;
    double* reflect = work + 2 * n;
    for (int i = 0; i < n; ++i) {
        reflect[i] = ((a[i] > 0.0) || (b[i] == infinity)) ? 1.0 : 0.0;
        const double lower = (reflect[i] == 1.0) ? -b[i] : a[i];
        const double upper = (reflect[i] == 1.0) ? -a[i] : b[i];
        Phi_of_a[i] = std::min(40.0, std::max(-40.0, lower));
        Phi_of_b[i] = std::min(40.0, std::max(-40.0, upper));
    }
    k_Phi(Phi_of_a, n);
    k_Phi(Phi_of_b, n);
    for (int i = 0; i < n; ++i) {
        const double prob = Phi_of_b[i] - Phi_of_a[i];
        const double u_or_1m_u = (reflect[i] == 1.0) ? one_minus_u[i] : u[i];
        Z[i] = Phi_of_a[i] + u_or_1m_u * prob;
        log_mass[i] = prob;
    }
    k_inv_Phi_in_double_precision(Z, n);
    k_log(log_mass, n);
    for (int i = 0; i < n; ++i) Z[i] = (reflect[i] == 1.0) ? -Z[i] : Z[i];

}

//// ---- the rows of a GHK step computed directly, Phi_type 2, by the four-class LC-MVOP's stable form: with
////      p(x) = 0.07056 x^3 + 1.5976 x, F(b) - F(a) = F(b) (1 - F(a)) (1 - exp(-(p(b) - p(a)))),
////      p(b) - p(a) = (b - a) (0.07056 (a^2 + a b + b^2) + 1.5976), and logit F(Z) =
////      log((1 - u) F(a) + u F(b)) - log((1 - u) (1 - F(a)) + u (1 - F(b))), both sums of positive terms (an
////      infinite bound is taken as +-1000, as the models' C_sentinel: F is exactly 0 or 1 there); work: 9 n
inline void GHK_rows_directly_with_Phi_approx(const int n, const double* a, const double* b, const double* log_u,
                                              const double* log_1m_u, double* Z, double* log_mass, double* work) {

    const double infinity = std::numeric_limits<double>::infinity();
    double* log_F_of_a = work;
    double* log_F_of_b = work + n;
    double* log_1m_F_of_a = work + 2 * n;
    double* log_1m_F_of_b = work + 3 * n;
    double* log_of_lower_sum = work + 4 * n;
    double* log_of_upper_sum = work + 5 * n;
    double* first_term = work + 6 * n;
    double* second_term = work + 7 * n;
    double* work_of_sums = work + 8 * n;
    for (int i = 0; i < n; ++i) {
        const double lower = (a[i] == -infinity) ? -1000.0 : a[i];
        const double upper = (b[i] == infinity) ? 1000.0 : b[i];
        const double p_of_a = (PHI_APPROX_A * lower * lower + PHI_APPROX_B) * lower;
        const double p_of_b = (PHI_APPROX_A * upper * upper + PHI_APPROX_B) * upper;
        log_F_of_a[i] = p_of_a;
        log_1m_F_of_a[i] = -p_of_a;
        log_F_of_b[i] = p_of_b;
        log_1m_F_of_b[i] = -p_of_b;
        log_mass[i] = -(upper - lower) * (PHI_APPROX_A * (lower * lower + lower * upper + upper * upper)
                                          + PHI_APPROX_B);
    }
    k_log_inv_logit_in_double_precision(log_F_of_a, n);
    k_log_inv_logit_in_double_precision(log_1m_F_of_a, n);
    k_log_inv_logit_in_double_precision(log_F_of_b, n);
    k_log_inv_logit_in_double_precision(log_1m_F_of_b, n);
    k_log1m_exp_in_double_precision(log_mass, n);
    for (int i = 0; i < n; ++i) {
        log_mass[i] += log_F_of_b[i] + log_1m_F_of_a[i];
        first_term[i] = log_F_of_a[i] + log_1m_u[i];
        second_term[i] = log_F_of_b[i] + log_u[i];
    }
    log_sum_exp_of_two_buffers(first_term, second_term, n, log_of_lower_sum, work_of_sums);
    for (int i = 0; i < n; ++i) {
        first_term[i] = log_1m_F_of_a[i] + log_1m_u[i];
        second_term[i] = log_1m_F_of_b[i] + log_u[i];
    }
    log_sum_exp_of_two_buffers(first_term, second_term, n, log_of_upper_sum, work_of_sums);
    for (int i = 0; i < n; ++i) Z[i] = log_of_lower_sum[i] - log_of_upper_sum[i];
    k_inv_Phi_approx_from_logit_prob_in_double_precision(Z, n);

}

//// ---- the rows in the lower tail (b < underflow_threshold), on the log scale, as BayesMVP's C++
////      (fn_MVP_compute_lp_GHK_cols_log_scale_underflow and the lower tail of the MVOP log-scale tails):
////      log mass = log F(b) + log1m_exp(log F(a) - log F(b)), log F(Z) = log_sum_exp(log(1 - u) + log F(a),
////      log(u) + log F(b)), log(1 - F(Z)) = log1m_exp(log F(Z)) (a = -inf: log mass = log F(b),
////      log F(Z) = log(u) + log F(b)); work: 6 n
inline void GHK_rows_in_the_lower_tail(const int n, const int Phi_type, const double* a, const double* b,
                                       const double* log_u, const double* log_1m_u, double* Z, double* log_mass,
                                       double* work) {

    const double infinity = std::numeric_limits<double>::infinity();
    double* log_F_of_a = work;
    double* log_F_of_b = work + n;
    double* log_F_of_Z = work + 2 * n;
    double* log_1m_F_of_Z = work + 3 * n;
    double* first_term = work + 4 * n;
    double* work_of_sums = work + 5 * n;
    //// (a = -inf: its log F is not used; b in its place keeps the kernels' arguments finite)
    for (int i = 0; i < n; ++i) {
        log_F_of_a[i] = (a[i] == -infinity) ? b[i] : a[i];
        log_F_of_b[i] = b[i];
    }
    k_log_F(log_F_of_a, n, Phi_type);
    k_log_F(log_F_of_b, n, Phi_type);
    for (int i = 0; i < n; ++i) {
        log_mass[i] = (a[i] == -infinity) ? -1.0 : log_F_of_a[i] - log_F_of_b[i];
        first_term[i] = log_1m_u[i] + log_F_of_a[i];
        log_1m_F_of_Z[i] = log_u[i] + log_F_of_b[i];
    }
    k_log1m_exp_in_double_precision(log_mass, n);
    log_sum_exp_of_two_buffers(first_term, log_1m_F_of_Z, n, log_F_of_Z, work_of_sums);
    for (int i = 0; i < n; ++i) {
        log_mass[i] = log_F_of_b[i] + ((a[i] == -infinity) ? 0.0 : log_mass[i]);
        log_F_of_Z[i] = (a[i] == -infinity) ? log_u[i] + log_F_of_b[i] : log_F_of_Z[i];
        log_1m_F_of_Z[i] = log_F_of_Z[i];
    }
    k_log1m_exp_in_double_precision(log_1m_F_of_Z, n);
    inverse_of_F_from_the_log_scale(log_F_of_Z, log_1m_F_of_Z, n, Phi_type, Z);

}

//// ---- the rows in the upper tail (a > overflow_threshold), on the log scale with 1 - F(x) = F(-x), as
////      BayesMVP's C++ (fn_MVP_compute_lp_GHK_cols_log_scale_overflow and the upper tail of the MVOP log-scale
////      tails):
////      log mass = log F(-a) + log1m_exp(log F(-b) - log F(-a)), log(1 - F(Z)) = log_sum_exp(log(1 - u) +
////      log F(-a), log(u) + log F(-b)), log F(Z) = log1m_exp(log(1 - F(Z))) (b = +inf: log mass = log F(-a),
////      log(1 - F(Z)) = log(1 - u) + log F(-a)); work: 6 n
inline void GHK_rows_in_the_upper_tail(const int n, const int Phi_type, const double* a, const double* b,
                                       const double* log_u, const double* log_1m_u, double* Z, double* log_mass,
                                       double* work) {

    const double infinity = std::numeric_limits<double>::infinity();
    double* log_1m_F_of_a = work;
    double* log_1m_F_of_b = work + n;
    double* log_F_of_Z = work + 2 * n;
    double* log_1m_F_of_Z = work + 3 * n;
    double* second_term = work + 4 * n;
    double* work_of_sums = work + 5 * n;
    //// (b = +inf: its log(1 - F) is not used; a in its place keeps the kernels' arguments finite)
    for (int i = 0; i < n; ++i) {
        log_1m_F_of_a[i] = -a[i];
        log_1m_F_of_b[i] = (b[i] == infinity) ? -a[i] : -b[i];
    }
    k_log_F(log_1m_F_of_a, n, Phi_type);
    k_log_F(log_1m_F_of_b, n, Phi_type);
    for (int i = 0; i < n; ++i) {
        log_mass[i] = (b[i] == infinity) ? -1.0 : log_1m_F_of_b[i] - log_1m_F_of_a[i];
        log_F_of_Z[i] = log_1m_u[i] + log_1m_F_of_a[i];
        second_term[i] = log_u[i] + log_1m_F_of_b[i];
    }
    k_log1m_exp_in_double_precision(log_mass, n);
    log_sum_exp_of_two_buffers(log_F_of_Z, second_term, n, log_1m_F_of_Z, work_of_sums);
    for (int i = 0; i < n; ++i) {
        log_mass[i] = log_1m_F_of_a[i] + ((b[i] == infinity) ? 0.0 : log_mass[i]);
        log_1m_F_of_Z[i] = (b[i] == infinity) ? log_1m_u[i] + log_1m_F_of_a[i] : log_1m_F_of_Z[i];
        log_F_of_Z[i] = log_1m_F_of_Z[i];
    }
    k_log1m_exp_in_double_precision(log_F_of_Z, n);
    inverse_of_F_from_the_log_scale(log_F_of_Z, log_1m_F_of_Z, n, Phi_type, Z);

}

//// ---- the rows of a narrow interval (both bounds finite, (b - a) (1 + |mid|) < 1e-5), as the four-class
////      LC-MVOP's narrow_interval_logprob and narrow_interval_z: with w = b - a and mid = (a + b) / 2,
////      log mass = log(w) + log sum_q weight_q f(mid + w node_q / 2) (Gauss-Legendre, 4 nodes),
////      Z = mid + w (u - 1/2) + score w^2 u (1 - u) / 2, score = -mid (Phi_type 1) or, with
////      d = 0.21168 mid^2 + 1.5976, 0.42336 mid / d + d (1 - 2 Phi_approx(mid)) (Phi_type 2); work: 9 n
inline void GHK_rows_of_a_narrow_interval(const int n, const int Phi_type, const double* a, const double* b,
                                          const double* u, const double* one_minus_u, double* Z,
                                          double* log_mass, double* work) {

    double* width = work;
    double* mid = work + n;
    double* terms = work + 2 * n;                 // 4 n: one column per node
    double* largest_term = work + 6 * n;
    double* work_of_log_density = work + 7 * n;   // 2 n
    for (int i = 0; i < n; ++i) {
        width[i] = b[i] - a[i];
        mid[i] = 0.5 * (a[i] + b[i]);
    }
    for (int node = 0; node < 4; ++node) {
        double* terms_of_node = terms + node * n;
        for (int i = 0; i < n; ++i) terms_of_node[i] = mid[i] + 0.5 * width[i] * NARROW_INTERVAL_NODES[node];
        k_log_density_of_F(terms_of_node, n, Phi_type, work_of_log_density);
        const double log_weight = std::log(NARROW_INTERVAL_WEIGHTS[node]);
        for (int i = 0; i < n; ++i) terms_of_node[i] += log_weight;
    }
    for (int i = 0; i < n; ++i) {
        largest_term[i] = std::max(std::max(terms[i], terms[n + i]),
                                   std::max(terms[2 * n + i], terms[3 * n + i]));
    }
    for (int node = 0; node < 4; ++node) {
        double* terms_of_node = terms + node * n;
        for (int i = 0; i < n; ++i) terms_of_node[i] -= largest_term[i];
        k_exp(terms_of_node, n);
    }
    for (int i = 0; i < n; ++i) {
        log_mass[i] = terms[i] + terms[n + i] + terms[2 * n + i] + terms[3 * n + i];
        Z[i] = width[i];
    }
    k_log(log_mass, n);
    k_log(Z, n);
    for (int i = 0; i < n; ++i) log_mass[i] += Z[i] + largest_term[i];
    //// (the score: -mid, or for Phi_type 2 0.42336 mid / d + d (1 - 2 inv_logit(p(mid))))
    double* score = terms;
    if (Phi_type == 1) {
        for (int i = 0; i < n; ++i) score[i] = -mid[i];
    } else {
        double* derivative = terms + n;
        for (int i = 0; i < n; ++i) {
            derivative[i] = 3.0 * PHI_APPROX_A * mid[i] * mid[i] + PHI_APPROX_B;
            score[i] = (PHI_APPROX_A * mid[i] * mid[i] + PHI_APPROX_B) * mid[i];
        }
        k_inv_logit(score, n);
        for (int i = 0; i < n; ++i) {
            score[i] = 6.0 * PHI_APPROX_A * mid[i] / derivative[i] + derivative[i] * (1.0 - 2.0 * score[i]);
        }
    }
    for (int i = 0; i < n; ++i) {
        Z[i] = mid[i] + width[i] * (u[i] - 0.5) + 0.5 * score[i] * width[i] * width[i] * u[i] * one_minus_u[i];
    }

}

inline void stop_with_a_message_about_the_arguments(const std::string& message) {

    throw std::domain_error(std::string("nico_finite_mixture_of_multivariate_ordered_probit_GHK_log_likelihood: ")
                            + message);

}

}  // namespace nico::math

namespace nico::math {

template <typename T_u_raw, typename T_intercept, typename T_coefficients, typename T_cutpoints,
          typename T_L_Omega, typename T_log_class_probability>
inline stan::return_type_t<T_u_raw, T_intercept, T_coefficients, T_cutpoints, T_L_Omega, T_log_class_probability>
nico_finite_mixture_of_multivariate_ordered_probit_GHK_log_likelihood(
        const T_u_raw& u_raw,
        const int nuisance_transformation,
        const T_intercept& intercept,
        const T_coefficients& coefficients,
        const std::vector<std::vector<int>>& coefficient_set_of_class_and_test,
        const std::vector<std::vector<Eigen::Matrix<double, -1, -1>>>& covariates,
        const std::vector<std::vector<int>>& n_covariates,
        const T_cutpoints& cutpoints,
        const std::vector<std::vector<int>>& cutpoint_set_of_class_and_test,
        const std::vector<int>& position_of_first_cutpoint_of_test,
        const T_L_Omega& L_Omega,
        const T_log_class_probability& log_class_probability,
        const std::vector<std::vector<int>>& category,
        const std::vector<std::vector<int>>& upper_category,
        const std::vector<int>& test_is_binary,
        const std::vector<int>& n_categories_of_test,
        const std::vector<int>& test_at_GHK_position,
        const int Phi_type,
        const double overflow_threshold,
        const double underflow_threshold,
        std::ostream* /*pstream__*/ = nullptr) {

    using namespace nico::math;
    using stan::math::value_of;
    const double infinity = std::numeric_limits<double>::infinity();
    constexpr bool gradient_is_needed = !stan::is_constant_all<T_u_raw, T_intercept, T_coefficients, T_cutpoints,
                                                               T_L_Omega, T_log_class_probability>::value;

    const auto& intercept_ref = stan::math::to_ref(intercept);
    const auto& log_class_probability_ref = stan::math::to_ref(log_class_probability);

    //// ---- the sizes, and the checks of the arguments (each O(1) per test, class or set, except the categories,
    ////      checked where their codes are made):
    const int M = static_cast<int>(u_raw.size());
    const int C = static_cast<int>(L_Omega.size());
    const int T = (C > 0) ? static_cast<int>(L_Omega[0].rows()) : 0;
    const int n_coefficient_sets = static_cast<int>(coefficients.size());
    const int n_cutpoint_sets = static_cast<int>(cutpoints.size());
    const int n_covariate_sets = static_cast<int>(covariates.size());
    if (C < 1) stop_with_a_message_about_the_arguments("L_Omega must have at least one class");
    if (Phi_type != 1 && Phi_type != 2) stop_with_a_message_about_the_arguments("Phi_type must be 1 or 2");
    if (nuisance_transformation != 1 && nuisance_transformation != 2) {
        stop_with_a_message_about_the_arguments("nuisance_transformation must be 1 (Phi) or 2 (tanh)");
    }
    if (intercept_ref.rows() != C || intercept_ref.cols() != T) {
        stop_with_a_message_about_the_arguments("intercept must be C x T");
    }
    if (log_class_probability_ref.rows() != M || log_class_probability_ref.cols() != C) {
        stop_with_a_message_about_the_arguments("log_class_probability must be M x C");
    }
    if (static_cast<int>(category.size()) != M || static_cast<int>(upper_category.size()) != M) {
        stop_with_a_message_about_the_arguments("category and upper_category must have M rows");
    }
    if (static_cast<int>(test_is_binary.size()) != T || static_cast<int>(n_categories_of_test.size()) != T
        || static_cast<int>(test_at_GHK_position.size()) != T
        || static_cast<int>(position_of_first_cutpoint_of_test.size()) != T) {

        stop_with_a_message_about_the_arguments("the arrays of the tests must have T elements");

    }
    if (static_cast<int>(coefficient_set_of_class_and_test.size()) != C
        || static_cast<int>(cutpoint_set_of_class_and_test.size()) != C) {

        stop_with_a_message_about_the_arguments("the sets of each class and test must be C x T");

    }
    for (int c = 0; c < C; ++c) {
        if (L_Omega[c].rows() != T || L_Omega[c].cols() != T) {
            stop_with_a_message_about_the_arguments("every L_Omega must be T x T");
        }
        if (static_cast<int>(coefficient_set_of_class_and_test[c].size()) != T
            || static_cast<int>(cutpoint_set_of_class_and_test[c].size()) != T) {

            stop_with_a_message_about_the_arguments("the sets of each class and test must be C x T");

        }
    }
    for (int m = 0; m < M; ++m) {
        if (u_raw[m].size() != T) {
            stop_with_a_message_about_the_arguments("every row of u_raw must have T values");
        }
        if (static_cast<int>(category[m].size()) != T || static_cast<int>(upper_category[m].size()) != T) {
            stop_with_a_message_about_the_arguments("every row of category and upper_category must have T "
                                                    "values");
        }
    }
    //// (covariates: none when there are no covariate sets or no covariates in any of them)
    int largest_number_of_covariates = 0;
    if (n_covariate_sets > 0) {
        if (n_covariate_sets != 1 && n_covariate_sets != n_coefficient_sets) {
            stop_with_a_message_about_the_arguments("covariates must have 1 or S_B sets");
        }
        if (static_cast<int>(n_covariates.size()) != n_covariate_sets) {
            stop_with_a_message_about_the_arguments("n_covariates must have the sets of covariates");
        }
        for (int s = 0; s < n_covariate_sets; ++s) {
            if (static_cast<int>(covariates[s].size()) != T || static_cast<int>(n_covariates[s].size()) != T) {
                stop_with_a_message_about_the_arguments("every set of covariates must have T matrices");
            }
            for (int t = 0; t < T; ++t) {
                const int P = n_covariates[s][t];
                if (P < 0 || P > covariates[s][t].cols() || (P > 0 && covariates[s][t].rows() != M)) {
                    stop_with_a_message_about_the_arguments("covariates must be M x P with n_covariates <= P");
                }
                largest_number_of_covariates = std::max(largest_number_of_covariates, P);
            }
        }
    }
    const bool there_are_covariates = (largest_number_of_covariates > 0);
    if (there_are_covariates) {
        for (int s = 0; s < n_coefficient_sets; ++s) {
            if (coefficients[s].cols() != T || coefficients[s].rows() < largest_number_of_covariates) {
                stop_with_a_message_about_the_arguments("every coefficient set must be P_max x T");
            }
        }
    }
    for (int c = 0; c < C; ++c) {
        for (int t = 0; t < T; ++t) {
            if (there_are_covariates && (coefficient_set_of_class_and_test[c][t] < 1
                                         || coefficient_set_of_class_and_test[c][t] > n_coefficient_sets)) {

                stop_with_a_message_about_the_arguments("a coefficient set is out of range");

            }
            if (test_is_binary[t] != 1 && (cutpoint_set_of_class_and_test[c][t] < 1
                                           || cutpoint_set_of_class_and_test[c][t] > n_cutpoint_sets)) {

                stop_with_a_message_about_the_arguments("a cutpoint set is out of range");

            }
        }
    }
    std::vector<int> position_of_test_in_GHK_order(T, -1);
    for (int j = 0; j < T; ++j) {
        const int t = test_at_GHK_position[j] - 1;
        if (t < 0 || t >= T || position_of_test_in_GHK_order[t] != -1) {
            stop_with_a_message_about_the_arguments("test_at_GHK_position must be a permutation of 1:T");
        }
        position_of_test_in_GHK_order[t] = j;
    }
    for (int t = 0; t < T; ++t) {
        if (test_is_binary[t] == 1) continue;
        const int first = position_of_first_cutpoint_of_test[t] - 1;
        for (int s = 0; s < n_cutpoint_sets; ++s) {
            if (first < 0 || first + n_categories_of_test[t] - 1 > cutpoints[s].size()) {
                stop_with_a_message_about_the_arguments("the cutpoints of an ordinal test are out of range");
            }
        }
    }

    //// ---- the values of the parameters:
    const Eigen::Matrix<double, -1, -1> intercept_values = value_of(intercept_ref);
    std::vector<Eigen::Matrix<double, -1, -1>> L_values(C);
    for (int c = 0; c < C; ++c) L_values[c] = value_of(L_Omega[c]);
    std::vector<Eigen::Matrix<double, -1, 1>> cutpoint_values(n_cutpoint_sets);
    for (int s = 0; s < n_cutpoint_sets; ++s) cutpoint_values[s] = value_of(cutpoints[s]);

    //// ---- the buffers:
    buffers_of_mixture_of_multivariate_ordered_probit_GHK_t& buffers =
          buffers_of_mixture_of_multivariate_ordered_probit_GHK();
    const std::size_t MT = static_cast<std::size_t>(M) * T;
    const std::size_t MC = static_cast<std::size_t>(M) * C;
    double* u_raw_values = buffer_of_at_least(buffers.values_of_each_subject_and_position, 9 * MT);
    double* u_values = u_raw_values + MT;                           // [j * M + m], GHK positions
    double* one_minus_u_values = u_values + MT;
    double* log_u_values = one_minus_u_values + MT;
    double* log_1m_u_values = log_u_values + MT;
    double* log_derivative_of_u = log_1m_u_values + MT;
    double* derivative_of_log_Jacobian = log_derivative_of_u + MT;
    double* Z_of_missing_results = derivative_of_log_Jacobian + MT;
    double* gradient_of_u_raw = Z_of_missing_results + MT;
    double* values_of_each_class_subject_and_position = buffer_of_at_least(
          buffers.values_of_each_class_subject_and_position, 8 * C * MT);
    double* log_mass_sum_of_each_class = buffer_of_at_least(buffers.values_of_each_subject_and_class, 2 * MC);
    double* weight_of_each_class = log_mass_sum_of_each_class + MC;   // [c * M + m]
    double* eta = buffer_of_at_least(buffers.values_of_each_subject,
                                     static_cast<std::size_t>(36 + T) * static_cast<std::size_t>(M));
    double* a = eta + M;
    double* b = a + M;
    double* log_mass = b + M;
    double* gathered = log_mass + M;            // 8 M: the inputs and outputs of the rows of one branch
    double* work = gathered + 8 * M;            // 12 M
    double* exponent = work + 12 * M;           // 5 M: the arguments of the partials' exponentials
    double* log_density = exponent + 5 * M;     // 3 M
    double* derivative_wrt_a = log_density + 3 * M;
    double* derivative_wrt_b = derivative_wrt_a + M;
    double* derivative_wrt_eta = derivative_wrt_b + M;
    double* increment = derivative_wrt_eta + M;
    double* adjoint_of_Z = increment + M;       // T M (one class at a time)
    const std::size_t n_linear_predictors = there_are_covariates
                                             ? static_cast<std::size_t>(n_coefficient_sets) * T * M : 0;
    double* linear_predictor_of_each_set = buffer_of_at_least(buffers.values_of_each_set_test_and_subject,
                                                              2 * n_linear_predictors + 1);
    double* gradient_of_linear_predictor = linear_predictor_of_each_set + n_linear_predictors;
    int* code_of_lower_bound = buffer_of_at_least(buffers.codes_of_each_subject_and_test, 2 * MT);
    int* code_of_upper_bound = code_of_lower_bound + MT;                // [t * M + m], tests in data order
    int* rows_of_each_branch = buffer_of_at_least(buffers.rows_of_each_branch, 4 * static_cast<std::size_t>(M));

    //// ---- the nuisance values, once for every class: u, 1 - u, log(u), log(1 - u), log |du/du_raw| and the
    ////      derivative of the log Jacobian (Phi: -u_raw; tanh, u = inv_logit(2 u_raw): 2 (1 - 2 u))
    for (int m = 0; m < M; ++m) {
        for (int j = 0; j < T; ++j) {
            u_raw_values[static_cast<std::size_t>(j) * M + m] = value_of(u_raw[m].coeff(j));
        }
    }
    if (nuisance_transformation == 1) {
        for (std::size_t i = 0; i < MT; ++i) {
            u_values[i] = u_raw_values[i];
            one_minus_u_values[i] = -u_raw_values[i];
            log_u_values[i] = u_raw_values[i];
            log_1m_u_values[i] = -u_raw_values[i];
            log_derivative_of_u[i] = -0.5 * u_raw_values[i] * u_raw_values[i] + NEG_HALF_LOG_2PI;
            derivative_of_log_Jacobian[i] = -u_raw_values[i];
        }
        k_Phi(u_values, static_cast<int>(MT));
        k_Phi(one_minus_u_values, static_cast<int>(MT));
        k_log_Phi_in_double_precision(log_u_values, static_cast<int>(MT));
        k_log_Phi_in_double_precision(log_1m_u_values, static_cast<int>(MT));
    } else {
        for (std::size_t i = 0; i < MT; ++i) {
            u_values[i] = 2.0 * u_raw_values[i];
            one_minus_u_values[i] = -2.0 * u_raw_values[i];
            log_u_values[i] = 2.0 * u_raw_values[i];
            log_1m_u_values[i] = -2.0 * u_raw_values[i];
        }
        k_inv_logit(u_values, static_cast<int>(MT));
        k_inv_logit(one_minus_u_values, static_cast<int>(MT));
        k_log_inv_logit_in_double_precision(log_u_values, static_cast<int>(MT));
        k_log_inv_logit_in_double_precision(log_1m_u_values, static_cast<int>(MT));
        for (std::size_t i = 0; i < MT; ++i) {
            log_derivative_of_u[i] = LN2 + log_u_values[i] + log_1m_u_values[i];
            derivative_of_log_Jacobian[i] = 2.0 * (one_minus_u_values[i] - u_values[i]);
        }
    }
    double log_Jacobian = 0.0;
    for (std::size_t i = 0; i < MT; ++i) log_Jacobian += log_derivative_of_u[i];
    //// (the untruncated draws of the missing results, Z = F^{-1}(u), once for every class: with u = Phi(u_raw)
    ////  and Phi_type 1, Z = u_raw itself)
    if (nuisance_transformation == 1 && Phi_type == 1) {
        for (std::size_t i = 0; i < MT; ++i) Z_of_missing_results[i] = u_raw_values[i];
    } else {
        inverse_of_F_from_the_log_scale(log_u_values, log_1m_u_values, static_cast<int>(MT), Phi_type,
                                        Z_of_missing_results);
    }
    for (std::size_t i = 0; i < MT; ++i) gradient_of_u_raw[i] = 0.0;

    //// ---- the codes of the bounds of each subject and test, from the data, once for every class (and the
    ////      checks of the categories); a missing result (category < 0) has both bounds infinite
    for (int t = 0; t < T; ++t) {
        const int K = n_categories_of_test[t];
        const int first = position_of_first_cutpoint_of_test[t] - 1;
        for (int m = 0; m < M; ++m) {
            const int k = category[m][t];
            int lower_code = CODE_OF_AN_INFINITE_BOUND;
            int upper_code = CODE_OF_AN_INFINITE_BOUND;
            if (k >= 0) {
                if (test_is_binary[t] == 1) {
                    if (k > 1) stop_with_a_message_about_the_arguments("a binary result must be 0 or 1");
                    lower_code = (k == 1) ? CODE_OF_THE_CUTPOINT_ZERO_OF_A_BINARY_TEST
                                          : CODE_OF_AN_INFINITE_BOUND;
                    upper_code = (k == 1) ? CODE_OF_AN_INFINITE_BOUND
                                          : CODE_OF_THE_CUTPOINT_ZERO_OF_A_BINARY_TEST;
                } else {
                    const int k_upper = upper_category[m][t];
                    if (k < 1 || k_upper < k || k_upper > K) {
                        stop_with_a_message_about_the_arguments("an ordinal category is out of range");
                    }
                    lower_code = (k == 1) ? CODE_OF_AN_INFINITE_BOUND : first + k - 2;
                    upper_code = (k_upper == K) ? CODE_OF_AN_INFINITE_BOUND : first + k_upper - 1;
                }
            }
            code_of_lower_bound[static_cast<std::size_t>(t) * M + m] = lower_code;
            code_of_upper_bound[static_cast<std::size_t>(t) * M + m] = upper_code;
        }
    }

    //// ---- the linear predictors of the covariates of each coefficient set and test, once for every class
    if (there_are_covariates) {
        for (int s = 0; s < n_coefficient_sets; ++s) {
            const int s_X = (n_covariate_sets == 1) ? 0 : s;
            const Eigen::Matrix<double, -1, -1> coefficient_values = value_of(coefficients[s]);
            for (int t = 0; t < T; ++t) {
                Eigen::Map<Eigen::Matrix<double, -1, 1>> linear_predictor(linear_predictor_of_each_set
                                                             + (static_cast<std::size_t>(s) * T + t) * M, M);
                const int P = n_covariates[s_X][t];
                if (P > 0) {
                    linear_predictor.noalias() = covariates[s_X][t].leftCols(P)
                                                 * coefficient_values.col(t).head(P);
                } else {
                    linear_predictor.setZero();
                }
            }
        }
        for (std::size_t i = 0; i < n_linear_predictors; ++i) gradient_of_linear_predictor[i] = 0.0;
    }

    //// ---- the forward pass of the GHK recursion of each class: Z, the log masses and the partial derivatives
    ////      of each step, in the buffers of the class
    for (int c = 0; c < C; ++c) {
        double* Z_of_class = values_of_each_class_subject_and_position + 8 * c * MT;   // [j * M + m]
        double* dZ_da_of_class = Z_of_class + MT;
        double* dZ_db_of_class = dZ_da_of_class + MT;
        double* dZ_du_raw_of_class = dZ_db_of_class + MT;
        double* dlog_mass_da_of_class = dZ_du_raw_of_class + MT;
        double* dlog_mass_db_of_class = dlog_mass_da_of_class + MT;
        double* finite_a_of_class = dlog_mass_db_of_class + MT;     // a where finite, 0 where infinite
        double* finite_b_of_class = finite_a_of_class + MT;
        double* log_mass_sum = log_mass_sum_of_each_class + static_cast<std::size_t>(c) * M;
        for (int m = 0; m < M; ++m) {
            log_mass_sum[m] = value_of(log_class_probability_ref.coeff(m, c));
            increment[m] = 0.0;
        }
        const Eigen::Map<const Eigen::Matrix<double, -1, -1>> Z_matrix(Z_of_class, M, T);

        for (int j = 0; j < T; ++j) {
            const int t = test_at_GHK_position[j] - 1;
            const double s = 1.0 / L_values[c](j, j);
            const std::size_t offset = static_cast<std::size_t>(j) * M;
            const double* cutpoints_of_test = (test_is_binary[t] == 1) ? nullptr
                                              : cutpoint_values[cutpoint_set_of_class_and_test[c][t] - 1].data();
            const double* linear_predictor = there_are_covariates
                  ? linear_predictor_of_each_set
                    + (static_cast<std::size_t>(coefficient_set_of_class_and_test[c][t] - 1) * T + t) * M
                  : nullptr;
            const int* lower_codes = code_of_lower_bound + static_cast<std::size_t>(t) * M;
            const int* upper_codes = code_of_upper_bound + static_cast<std::size_t>(t) * M;
            const double* u_of_position = u_values + offset;
            const double* one_minus_u_of_position = one_minus_u_values + offset;
            const double* log_u_of_position = log_u_values + offset;
            const double* log_1m_u_of_position = log_1m_u_values + offset;
            const double* log_derivative_of_u_of_position = log_derivative_of_u + offset;
            double* Z_of_position = Z_of_class + offset;

            //// (the linear predictor and the bounds of every subject)
            const double intercept_of_class_and_test = intercept_values(c, t);
            for (int m = 0; m < M; ++m) {
                eta[m] = intercept_of_class_and_test + increment[m]
                         + (there_are_covariates ? linear_predictor[m] : 0.0);
            }
            for (int m = 0; m < M; ++m) {
                const int lower_code = lower_codes[m];
                const int upper_code = upper_codes[m];
                const double lower_cutpoint = (lower_code >= 0) ? cutpoints_of_test[lower_code] : 0.0;
                const double upper_cutpoint = (upper_code >= 0) ? cutpoints_of_test[upper_code] : 0.0;
                a[m] = (lower_code == CODE_OF_AN_INFINITE_BOUND) ? -infinity : (lower_cutpoint - eta[m]) * s;
                b[m] = (upper_code == CODE_OF_AN_INFINITE_BOUND) ? infinity : (upper_cutpoint - eta[m]) * s;
            }

            //// (the rows of each branch: narrow intervals first, then the lower and the upper tails, as in the
            ////  four-class LC-MVOP; the missing results are given their untruncated draws)
            int n_rows_of_branch[4] = { 0, 0, 0, 0 };
            int* rows_of_branch[4] = { rows_of_each_branch, rows_of_each_branch + M, rows_of_each_branch + 2 * M,
                                       rows_of_each_branch + 3 * M };
            for (int m = 0; m < M; ++m) {
                if (category[m][t] < 0) {
                    Z_of_position[m] = Z_of_missing_results[offset + m];
                    log_mass[m] = 0.0;
                    continue;
                }
                int branch = ROWS_DIRECTLY;
                const bool both_bounds_are_finite = (a[m] != -infinity) && (b[m] != infinity);
                if (both_bounds_are_finite && (b[m] - a[m]) * (1.0 + std::fabs(0.5 * (a[m] + b[m])))
                                              < LARGEST_WIDTH_OF_A_NARROW_INTERVAL) {

                    branch = ROWS_OF_A_NARROW_INTERVAL;

                } else if (b[m] < underflow_threshold) {
                    branch = ROWS_IN_THE_LOWER_TAIL;
                } else if (a[m] > overflow_threshold) {
                    branch = ROWS_IN_THE_UPPER_TAIL;
                }
                rows_of_branch[branch][n_rows_of_branch[branch]] = m;
                n_rows_of_branch[branch] += 1;
            }
            for (int branch = 0; branch < 4; ++branch) {
                const int n = n_rows_of_branch[branch];
                if (n == 0) continue;
                const int* rows = rows_of_branch[branch];
                double* gathered_a = gathered;
                double* gathered_b = gathered + n;
                double* gathered_u = gathered + 2 * n;
                double* gathered_one_minus_u = gathered + 3 * n;
                double* gathered_log_u = gathered + 4 * n;
                double* gathered_log_1m_u = gathered + 5 * n;
                double* gathered_Z = gathered + 6 * n;
                double* gathered_log_mass = gathered + 7 * n;
                for (int i = 0; i < n; ++i) {
                    const int m = rows[i];
                    gathered_a[i] = a[m];
                    gathered_b[i] = b[m];
                    gathered_u[i] = u_of_position[m];
                    gathered_one_minus_u[i] = one_minus_u_of_position[m];
                    gathered_log_u[i] = log_u_of_position[m];
                    gathered_log_1m_u[i] = log_1m_u_of_position[m];
                }
                if (branch == ROWS_DIRECTLY) {
                    if (Phi_type == 1) {
                        GHK_rows_directly_with_Phi(n, gathered_a, gathered_b, gathered_u, gathered_one_minus_u,
                                                   gathered_Z, gathered_log_mass, work);
                    } else {
                        GHK_rows_directly_with_Phi_approx(n, gathered_a, gathered_b, gathered_log_u,
                                                          gathered_log_1m_u, gathered_Z, gathered_log_mass, work);
                    }
                } else if (branch == ROWS_IN_THE_LOWER_TAIL) {
                    GHK_rows_in_the_lower_tail(n, Phi_type, gathered_a, gathered_b, gathered_log_u,
                                               gathered_log_1m_u, gathered_Z, gathered_log_mass, work);
                } else if (branch == ROWS_IN_THE_UPPER_TAIL) {
                    GHK_rows_in_the_upper_tail(n, Phi_type, gathered_a, gathered_b, gathered_log_u,
                                               gathered_log_1m_u, gathered_Z, gathered_log_mass, work);
                } else {
                    GHK_rows_of_a_narrow_interval(n, Phi_type, gathered_a, gathered_b, gathered_u,
                                                  gathered_one_minus_u, gathered_Z, gathered_log_mass, work);
                }
                for (int i = 0; i < n; ++i) {
                    Z_of_position[rows[i]] = gathered_Z[i];
                    log_mass[rows[i]] = gathered_log_mass[i];
                }
            }
            for (int m = 0; m < M; ++m) log_mass_sum[m] += log_mass[m];

            //// (the partial derivatives of the step, every subject at once; an infinite bound takes exponent 0
            ////  and partial 0)
            if (gradient_is_needed) {
                double* log_density_of_a = log_density;
                double* log_density_of_b = log_density + M;
                double* log_density_of_Z = log_density + 2 * M;
                for (int m = 0; m < M; ++m) {
                    log_density_of_a[m] = (a[m] == -infinity) ? 0.0 : a[m];
                    log_density_of_b[m] = (b[m] == infinity) ? 0.0 : b[m];
                    log_density_of_Z[m] = Z_of_position[m];
                }
                k_log_density_of_F(log_density_of_a, M, Phi_type, work);
                k_log_density_of_F(log_density_of_b, M, Phi_type, work);
                k_log_density_of_F(log_density_of_Z, M, Phi_type, work);
                double* exponent_dZ_da = exponent;
                double* exponent_dZ_db = exponent + M;
                double* exponent_dZ_du_raw = exponent + 2 * M;
                double* exponent_dlog_mass_da = exponent + 3 * M;
                double* exponent_dlog_mass_db = exponent + 4 * M;
                for (int m = 0; m < M; ++m) {
                    const bool a_is_finite = (a[m] != -infinity);
                    const bool b_is_finite = (b[m] != infinity);
                    exponent_dZ_da[m] = a_is_finite ? log_1m_u_of_position[m] + log_density_of_a[m]
                                                      - log_density_of_Z[m] : 0.0;
                    exponent_dZ_db[m] = b_is_finite ? log_u_of_position[m] + log_density_of_b[m]
                                                      - log_density_of_Z[m] : 0.0;
                    exponent_dZ_du_raw[m] = log_mass[m] + log_derivative_of_u_of_position[m]
                                            - log_density_of_Z[m];
                    exponent_dlog_mass_da[m] = a_is_finite ? log_density_of_a[m] - log_mass[m] : 0.0;
                    exponent_dlog_mass_db[m] = b_is_finite ? log_density_of_b[m] - log_mass[m] : 0.0;
                }
                k_exp(exponent, 5 * M);
                for (int m = 0; m < M; ++m) {
                    const bool a_is_finite = (a[m] != -infinity);
                    const bool b_is_finite = (b[m] != infinity);
                    dZ_da_of_class[offset + m] = a_is_finite ? exponent_dZ_da[m] : 0.0;
                    dZ_db_of_class[offset + m] = b_is_finite ? exponent_dZ_db[m] : 0.0;
                    dZ_du_raw_of_class[offset + m] = exponent_dZ_du_raw[m];
                    dlog_mass_da_of_class[offset + m] = a_is_finite ? -exponent_dlog_mass_da[m] : 0.0;
                    dlog_mass_db_of_class[offset + m] = b_is_finite ? exponent_dlog_mass_db[m] : 0.0;
                    finite_a_of_class[offset + m] = a_is_finite ? a[m] : 0.0;
                    finite_b_of_class[offset + m] = b_is_finite ? b[m] : 0.0;
                }
            }

            //// (the increment of the next GHK position: sum_{k <= j} L[j + 1, k] Z[k])
            if (j + 1 < T) {
                Eigen::Map<Eigen::Matrix<double, -1, 1>> increment_vector(increment, M);
                increment_vector.noalias() = Z_matrix.leftCols(j + 1)
                                             * L_values[c].row(j + 1).head(j + 1).transpose();
            }
        }
    }

    //// ---- the mixture over the classes: log sum_c exp(log_class_probability + the class's log masses) of each
    ////      subject, and the weight of each class (the derivative of the value with respect to its terms)
    double value = log_Jacobian;
    {
        double* largest_term = derivative_wrt_a;
        double* sum_of_exponentials = derivative_wrt_b;
        for (int m = 0; m < M; ++m) largest_term[m] = log_mass_sum_of_each_class[m];
        for (int c = 1; c < C; ++c) {
            const double* log_mass_sum = log_mass_sum_of_each_class + static_cast<std::size_t>(c) * M;
            for (int m = 0; m < M; ++m) largest_term[m] = std::max(largest_term[m], log_mass_sum[m]);
        }
        for (int c = 0; c < C; ++c) {
            const double* log_mass_sum = log_mass_sum_of_each_class + static_cast<std::size_t>(c) * M;
            double* weight = weight_of_each_class + static_cast<std::size_t>(c) * M;
            for (int m = 0; m < M; ++m) weight[m] = log_mass_sum[m] - largest_term[m];
        }
        k_exp(weight_of_each_class, static_cast<int>(MC));
        for (int m = 0; m < M; ++m) sum_of_exponentials[m] = 0.0;
        for (int c = 0; c < C; ++c) {
            const double* weight = weight_of_each_class + static_cast<std::size_t>(c) * M;
            for (int m = 0; m < M; ++m) sum_of_exponentials[m] += weight[m];
        }
        for (int c = 0; c < C; ++c) {
            double* weight = weight_of_each_class + static_cast<std::size_t>(c) * M;
            for (int m = 0; m < M; ++m) weight[m] /= sum_of_exponentials[m];
        }
        k_log(sum_of_exponentials, M);
        for (int m = 0; m < M; ++m) value += largest_term[m] + sum_of_exponentials[m];
    }

    if constexpr (!gradient_is_needed) {
        return value;
    } else {

        //// ---- the reverse sweep of the GHK recursion of each class, seeded by the class weights: the gradient
        ////      with respect to u_raw, the intercepts, the linear predictors of the covariates, the cutpoints
        ////      and L_Omega
        Eigen::Matrix<double, -1, -1> gradient_of_intercept = Eigen::Matrix<double, -1, -1>::Zero(C, T);
        std::vector<Eigen::Matrix<double, -1, -1>> gradient_of_L(C, Eigen::Matrix<double, -1, -1>::Zero(T, T));
        std::vector<Eigen::Matrix<double, -1, 1>> gradient_of_cutpoints(n_cutpoint_sets);
        for (int s = 0; s < n_cutpoint_sets; ++s) {
            gradient_of_cutpoints[s] = Eigen::Matrix<double, -1, 1>::Zero(cutpoints[s].size());
        }
        for (int c = 0; c < C; ++c) {
            const double* Z_of_class = values_of_each_class_subject_and_position + 8 * c * MT;
            const double* dZ_da_of_class = Z_of_class + MT;
            const double* dZ_db_of_class = dZ_da_of_class + MT;
            const double* dZ_du_raw_of_class = dZ_db_of_class + MT;
            const double* dlog_mass_da_of_class = dZ_du_raw_of_class + MT;
            const double* dlog_mass_db_of_class = dlog_mass_da_of_class + MT;
            const double* finite_a_of_class = dlog_mass_db_of_class + MT;
            const double* finite_b_of_class = finite_a_of_class + MT;
            const double* weight = weight_of_each_class + static_cast<std::size_t>(c) * M;
            const Eigen::Map<const Eigen::Matrix<double, -1, -1>> Z_matrix(Z_of_class, M, T);
            Eigen::Map<Eigen::Matrix<double, -1, -1>> adjoint_of_Z_matrix(adjoint_of_Z, M, T);
            adjoint_of_Z_matrix.setZero();
            for (int j = T - 1; j >= 0; --j) {
                const int t = test_at_GHK_position[j] - 1;
                const double s = 1.0 / L_values[c](j, j);
                const std::size_t offset = static_cast<std::size_t>(j) * M;
                const double* adjoint_of_Z_of_position = adjoint_of_Z + offset;
                double sum_of_bounds_times_their_derivatives = 0.0;
                double sum_of_derivatives_wrt_eta = 0.0;
                for (int m = 0; m < M; ++m) {
                    derivative_wrt_a[m] = adjoint_of_Z_of_position[m] * dZ_da_of_class[offset + m]
                                          + weight[m] * dlog_mass_da_of_class[offset + m];
                    derivative_wrt_b[m] = adjoint_of_Z_of_position[m] * dZ_db_of_class[offset + m]
                                          + weight[m] * dlog_mass_db_of_class[offset + m];
                    gradient_of_u_raw[offset + m] += adjoint_of_Z_of_position[m] * dZ_du_raw_of_class[offset + m];
                    derivative_wrt_eta[m] = -s * (derivative_wrt_a[m] + derivative_wrt_b[m]);
                    sum_of_bounds_times_their_derivatives +=
                          finite_a_of_class[offset + m] * derivative_wrt_a[m]
                          + finite_b_of_class[offset + m] * derivative_wrt_b[m];
                    sum_of_derivatives_wrt_eta += derivative_wrt_eta[m];
                }
                //// (a = (cutpoint - eta - inc) s: d/dcutpoint = s; d/dL[j, j] = (a / s) (-s^2) = -s a)
                if (test_is_binary[t] != 1) {
                    double* gradient_of_set =
                          gradient_of_cutpoints[cutpoint_set_of_class_and_test[c][t] - 1].data();
                    const int* lower_codes = code_of_lower_bound + static_cast<std::size_t>(t) * M;
                    const int* upper_codes = code_of_upper_bound + static_cast<std::size_t>(t) * M;
                    for (int m = 0; m < M; ++m) {
                        if (lower_codes[m] >= 0) gradient_of_set[lower_codes[m]] += s * derivative_wrt_a[m];
                        if (upper_codes[m] >= 0) gradient_of_set[upper_codes[m]] += s * derivative_wrt_b[m];
                    }
                }
                gradient_of_L[c](j, j) += -s * sum_of_bounds_times_their_derivatives;
                gradient_of_intercept(c, t) += sum_of_derivatives_wrt_eta;
                if (there_are_covariates) {
                    double* gradient_of_linear_predictor_of_set = gradient_of_linear_predictor
                          + (static_cast<std::size_t>(coefficient_set_of_class_and_test[c][t] - 1) * T + t) * M;
                    for (int m = 0; m < M; ++m) gradient_of_linear_predictor_of_set[m] += derivative_wrt_eta[m];
                }
                //// (inc = sum_{k < j} L[j, k] Z[k])
                if (j > 0) {
                    const Eigen::Map<const Eigen::Matrix<double, -1,
                                                         1>> derivative_wrt_eta_vector(derivative_wrt_eta, M);
                    adjoint_of_Z_matrix.leftCols(j).noalias() += derivative_wrt_eta_vector
                                                                 * L_values[c].row(j).head(j);
                    gradient_of_L[c].row(j).head(j).noalias() += derivative_wrt_eta_vector.transpose()
                                                                 * Z_matrix.leftCols(j);
                }
            }
        }
        for (std::size_t i = 0; i < MT; ++i) gradient_of_u_raw[i] += derivative_of_log_Jacobian[i];

        //// ---- the partials of the node
        auto partials = stan::math::make_partials_propagator(u_raw, intercept_ref, coefficients, cutpoints,
                                                             L_Omega, log_class_probability_ref);
        if constexpr (!stan::is_constant_all<T_u_raw>::value) {
            for (int m = 0; m < M; ++m) {
                for (int j = 0; j < T; ++j) {
                    stan::math::partials_vec<0>(partials)[m](j) =
                          gradient_of_u_raw[static_cast<std::size_t>(j) * M + m];
                }
            }
        }
        if constexpr (!stan::is_constant_all<T_intercept>::value) {
            stan::math::partials<1>(partials) = gradient_of_intercept;
        }
        if constexpr (!stan::is_constant_all<T_coefficients>::value) {
            for (int s = 0; s < n_coefficient_sets; ++s) {
                stan::math::partials_vec<2>(partials)[s].setZero();
                if (!there_are_covariates) continue;
                const int s_X = (n_covariate_sets == 1) ? 0 : s;
                for (int t = 0; t < T; ++t) {
                    const int P = n_covariates[s_X][t];
                    if (P == 0) continue;
                    const Eigen::Map<const Eigen::Matrix<double, -1, 1>> gradient_of_linear_predictor_of_set(
                          gradient_of_linear_predictor + (static_cast<std::size_t>(s) * T + t) * M, M);
                    stan::math::partials_vec<2>(partials)[s].col(t).head(P) =
                          covariates[s_X][t].leftCols(P).transpose() * gradient_of_linear_predictor_of_set;
                }
            }
        }
        if constexpr (!stan::is_constant_all<T_cutpoints>::value) {
            for (int s = 0; s < n_cutpoint_sets; ++s) {
                stan::math::partials_vec<3>(partials)[s] = gradient_of_cutpoints[s];
            }
        }
        if constexpr (!stan::is_constant_all<T_L_Omega>::value) {
            for (int c = 0; c < C; ++c) stan::math::partials_vec<4>(partials)[c] = gradient_of_L[c];
        }
        if constexpr (!stan::is_constant_all<T_log_class_probability>::value) {
            for (int c = 0; c < C; ++c) {
                for (int m = 0; m < M; ++m) {
                    stan::math::partials<5>(partials)(m, c) = weight_of_each_class[static_cast<std::size_t>(c) * M
                                                                                   + m];
                }
            }
        }
        return partials.build(value);
    }

}

}  // namespace nico::math

using nico::math::nico_finite_mixture_of_multivariate_ordered_probit_GHK_log_likelihood;























