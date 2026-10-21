#pragma once

/// if have AVX-512
//// -------------------------------------------------------------------------------------------------------------------------------------------------------------
// Simple test function that just multiplies vector by 2
// // =============================================================================================
// //  SELF-TEST  — call once from an Rcpp export or a throwaway main(); prints max errors.
// //  Reference: log Phi(x) = log(0.5*erfc(-x/sqrt2)) is exact to ~1e-15 for x >= -37.
// //  Passing:  err_logPhi <~ 1e-8 (limited by nico_log), err_dlogPhi <~ 1e-8,
// //            err_roundtrip (inv_Phi_from_log_p(log Phi(x), log Phi(-x)) - x) <~ 1e-12.
// // =============================================================================================
// inline void normal_tail_self_test() {
//       double err_logPhi = 0, err_dlogPhi = 0, err_roundtrip = 0, worst_x = 0;
//       for (double x = -37.0; x <= 8.0; x += 0.01) {
//           const double ref_logPhi  = std::log(0.5 * std::erfc(-x * M_SQRT1_2));
//           const double ref_dlogPhi = INV_SQRT_2PI * std::exp(-0.5 * x * x) / (0.5 * std::erfc(-x * M_SQRT1_2));
//           const double e1 = std::fabs(nico_log_Phi(x) - ref_logPhi) / std::max(1.0, std::fabs(ref_logPhi));
//           const double e2 = std::fabs(nico_dlog_Phi_dx(x) - ref_dlogPhi) / std::max(1e-300, std::fabs(ref_dlogPhi));
//           const double lp = nico_log_Phi(x), l1mp = nico_log_Phi(-x);
//           const double e3 = std::fabs(nico_inv_Phi_from_log_p(lp, l1mp) - x) / std::max(1.0, std::fabs(x));
//           if (e1 > err_logPhi)    err_logPhi = e1;
//           if (e2 > err_dlogPhi)   err_dlogPhi = e2;
//           if (e3 > err_roundtrip) { err_roundtrip = e3; worst_x = x; }
//       }
//       std::printf("normal_tail_self_test: max rel err  logPhi=%.3e  dlogPhi/dx=%.3e  inv_Phi roundtrip=%.3e (worst at x=%.2f)\n",
//                   err_logPhi, err_dlogPhi, err_roundtrip, worst_x);
//       //// the SIMD kernels must agree with the scalar ones to ~1e-12 — check a few lanes:
// #if defined(__AVX512VL__) && defined(__AVX512F__) && defined(__AVX512DQ__)
//       {
//           alignas(64) double xs[8]  = {-30.0, -12.0, -6.0, -3.0, -2.6, -1.0, 0.5, 4.0};
//           alignas(64) double out[8];
//           _mm512_store_pd(out, nico_log_Phi_AVX512(_mm512_load_pd(xs)));
//           double e = 0; for (int i = 0; i < 8; ++i) e = std::max(e, std::fabs(out[i] - nico_log_Phi(xs[i])) / std::max(1.0, std::fabs(out[i])));
//           std::printf("  AVX512 vs scalar log_Phi: max rel diff %.3e\n", e);
//       }
// #endif
// }
// 
// 
// 
//// ------------------------------------------------------------------------------------------------------------------------------------------------------------
// comparing value to itself checks for  NaN. is NaN then this = FALSE
//  __mmask8 mask_is_NaN = f(mask_is_not_NaN); // there is no "NOT" or "complement" mask  !!!!
// comparing value to itself checks for  NaN. is NaN then this = FALSE
//  __mmask8 mask_is_NaN = f(mask_is_not_NaN); // there is no "NOT" or "complement" mask  !!!!
// adapted from: https://stackoverflow.com/questions/30674291/how-to-check-inf-for-avx-intrinsic-m256
//x = _mm512_andnot_pd(sign_bit, x); // x  = NOT -0.0 AND x
// adapted from: https://stackoverflow.com/questions/30674291/how-to-check-inf-for-avx-intrinsic-m256
// x = _mm512_andnot_pd(sign_bit, x);
///////////////////  fns - exp   -----------------------------------------------------------------------------------------------------------------------------
/* AVX_a = p * 2^AVX_i */
// ALWAYS_INLINE __m512d nico_ldexp_2( const __m512d AVX_a, 
//                                     const __m512i AVX_i) {
//   
//         const __m512i neg_mask = _mm512_srai_epi64(AVX_i, 63);
//         const __m512i abs_i = _mm512_sub_epi64(_mm512_xor_si512(AVX_i, neg_mask), neg_mask);
//         const __m512i threshold = _mm512_set1_epi64(1000);
//         const __mmask8 cmp_mask = _mm512_cmpgt_epi64_mask(abs_i, threshold); 
//         
//         if (cmp_mask) {
//           
//               const __m512i i1 = _mm512_xor_si512(_mm512_and_si512(neg_mask, _mm512_set1_epi64(-2000)), threshold);
//               const __m512i i2 = _mm512_sub_epi64(AVX_i, i1);
//               const __m512d mid = _mm512_castsi512_pd(_mm512_add_epi64(_mm512_slli_epi64(i1, 52), _mm512_castpd_si512(AVX_a)));
//               
//               // return _mm512_castsi512_pd(_mm512_add_epi64(_mm512_slli_epi64(i2, 52), _mm512_castpd_si512(mid)));
//               return  nico_ldexp(mid, i2); 
//           
//         }
//         
//         // return _mm512_castsi512_pd(_mm512_add_epi64(_mm512_slli_epi64(AVX_i, 52), _mm512_castpd_si512(AVX_a)));
//         return  nico_ldexp(AVX_a, AVX_i); 
//   
// }
// Adapted from: https://stackoverflow.com/questions/48863719/fastest-implementation-of-exponential-function-using-avx
// added   (optional) extra degree(s) for poly approx (oroginal float fn had 4 degrees) - using "minimaxApprox" R package to find coefficient terms
// R code:    minimaxApprox::minimaxApprox(fn = exp, lower = -0.346573590279972643113, upper = 0.346573590279972643113, degree = 5, basis ="Chebyshev")
/* log2(e) */
/* -log(2)_hi, exactly representable */
/* -log(2)_lo */
// /* coefficients for core approximation to exp() in [-log(2)/2, log(2)/2] */
/* exp(x) = 2^i * e^f; i = rint (log2(e) * a), f = a - log(2) * i */
/* t = log2(e) * a */
/* i = (int)rint(t) */
// ((0<<4)| _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC|_MM_FROUND_NO_EXC));
/* a - log(2)_hi * r */
/* f = a - log(2)_hi * r - log(2)_lo * r */
/* p ~= exp (f), -log(2)/2 <= f <= log(2)/2 */
// polynomial approximation
/* exp(x) = 2^i * p */
//// see https://stackoverflow.com/questions/39587752/difference-between-ldexp1-x-and-exp2x
// Keep exceptional-lane libm calls out of the inlined SIMD hot path.
// 708.4 bounds the exponent-bit shortcut, NOT the domain of finite exp().
// Preserve the ordinary SIMD path; only exceptional lanes need libm's full exponent range.
// // see https://stackoverflow.com/questions/39587752/difference-between-ldexp1-x-and-exp2x
// inline __m512d nico_exp_1_AVX512(const __m512d a) {
//   
//   return _mm512_mask_blend_pd(
//     _mm512_cmp_pd_mask(_mm512_abs_pd(a), exp_bound, _CMP_LT_OQ),
//     _mm512_mask_blend_pd(
//       is_not_NaN_mask(a),
//       _mm512_set1_pd(INFINITY - INFINITY),
//       _mm512_mask_blend_pd(
//         _mm512_cmp_pd_mask(a, zero, _CMP_LT_OQ),
//         pos_inf,
//         zero
//       ) 
//     ),
//     nico_exp_1_wo_checks_AVX512(a)
//   );
//   
// } 
///////////////////  fns - log  (+ related e.g. log1m etc) ----------------------------------------------------------------------------------------------------
////// ------  log fn  --------   https://stackoverflow.com/a/65537754/9007125 // vectorized version of the answer by njuffa
//    e = (__double_as_int (a) - i1 )    &   i2   ;
// 0x1.0p-52
//   m = __int_as_double (__double_as_int (a) - e);  
//  m = _mm256_sub_pd(m, one) ; // m - 1.0;  /* m in [2/3, 4/3] */
// m = _mm512_sub_pd(m, one) ; // m - 1.0;  /* m in [2/3, 4/3] */
/* Compute log1p(m) for m in [-1/3, 1/3] */
// -0x1.0ae000p-3
//  0x1.208000p-3
// -0x1.f198b2p-4
//  0x1.1e5740p-3
// -0x1.55b36cp-3
//  0x1.99d8b2p-3
// -0x1.fffe02p-3
//  0x1.5554fap-2
// -0x1.000000p-1
//  0x1.62e430p-1 // log(2)
//// ---- log, double precision (nico_log_1_double_precision_*): the fdlibm (Sun Microsystems, e_log.c) algorithm,
////      vectorised. a = 2^k (1 + f) with 1 + f in [sqrt(2)/2, sqrt(2)) (the exponent trick below, with sqrt(2)/2
////      as the split point); s = f / (2 + f), |s| <= 0.1716; R = s^2 (Lg1 + s^4 (Lg3 + s^4 (Lg5 + s^4 Lg7))) +
////      s^4 (Lg2 + s^4 (Lg4 + s^4 Lg6)); log(a) = k ln2_hi - ((hfsq - (s (hfsq + R) + k ln2_lo)) - f), hfsq =
////      f^2 / 2 (ln 2 split into hi and lo parts). Below 1 ulp (measured 0.83 ulp), but one division per vector:
////      slower than nico_log_1_wo_checks_* (njuffa's single-precision polynomial, about 1.5e-8 relative error),
////      which therefore stays the default log. ---- reduction for a positive normal double: a = 2^k (1 + f) with
////      1 + f in [sqrt(2)/2, sqrt(2)) (k exact, as a double)
// bits of sqrt(2)/2
// e = k << 52; k = e * 2^-52 (exact); f exact
//// ---- the fdlibm core: k ln 2 + log1p(f) for 1 + f in [sqrt(2)/2, sqrt(2)) (shared by nico_log_1_* and
////      nico_log1p_1_*)
// 0x3fe62e42fee00000
// 0x3dea39ef35793c76
// 0x3FE5555555555593
// 0x3FD999999997FA04
// 0x3FD2492494229359
// 0x3FCC71C51D8E78AF
// 0x3FC7466496CB03DE
// 0x3FC39A09D078C69F
// 0x3FC2F112DF3E5244
///// compute log(1+x) without losing precision for small values of x. //  see: https://www.johndcook.com/cpp_log_one_plus_x.html
/// if abs(x) > small
////  if fabs(x) > small
//// ---- log1p, double precision (nico_log1p_1_double_precision_*; the default nico_log1p_1_* is the faster
////      single- precision-accuracy version above). For x in [sqrt(2)/2 - 1, sqrt(2) - 1) the fdlibm core runs on
////      f = x itself (k = 0): 1 + x is never rounded. Otherwise log(u), u = 1 + x, reduced as in nico_log_1,
////      plus (x - (u - 1)) / u, the rounding error of u to first order (a 14-bit reciprocal of u is ample for
////      that term). One core evaluation (one division) for any mix of lanes; the reciprocal only when a lane is
////      outside the core range.
// sqrt(2)/2 - 1
// sqrt(2) - 1
// ///////  compute log(1+x) without losing precision for small values of x. //  see: https://www.johndcook.com/cpp_log_one_plus_x.html
// -x);
/// if x < 0
//// if x > 0 
//////https://stackoverflow.com/a/65537754/9007125
// Isolate rare subnormal/domain repairs from the normal-positive SIMD path.
// The mantissa/exponent shortcut assumes a positive NORMAL double.
// A single unsigned interval test covers positive normal doubles, excluding infinity/NaN.
// double-precision log (fdlibm, one division per vector) with the same scalar repair of non-normal / non-positive lanes
///////  compute log(1+x) without losing precision for small values of x.
/////see: https://www.johndcook.com/cpp_log_one_plus_x.html 
/// if x <= -1.0
//// if abs(x) > small
// is x <= -1? (i.e., is x+1 <= 0?)
// if "is_x_le_or_eq_to_m1_mask" is FALSE (i.e., x+1 > 0)
// if x > 1e-4 - evaluate using normal log fn!
// if FALSE (i.e., x+1  > 0 ) then still pass onto normal log fn as this will handle NaNs etc!
// Isolate x <= -1, infinities and NaN (std::log1p) from the SIMD path.
// //  compute log(1+x) without losing precision for small values of x.
// //  see: https://www.johndcook.com/cpp_log_one_plus_x.html
/// if x > 0
/// if x < 0  ----log1p(exp(x))
/// if x > 0 ---- x + log1p(exp(-x))
//// ---- log(1 - exp(x)) (log1m_exp), x <= 0: x <  -ln 2:  log1p(-exp(x)) with the default nico_exp_1 and
////      nico_log1p_1 (as the Stan external's log1p(-exp(x)) branch: the same values) x >= -ln 2:  log(-expm1(x))
////      = log(-x) + h(x),  h(x) = log(expm1(x) / x) = x/2 + log(sinh(x/2) / (x/2)) = x/2 + sum_k B_2k x^2k / (2k
////      (2k)!)  (B_2k the Bernoulli numbers), to k = 8 (x^16): truncation below 1e-18 and rounding about 3e-17
////      on [-ln 2, 0); log(-x) by the double-precision log (nico_log_1_double_precision_*): no cancellation as x
////      -> 0, and no scalar log / expm1. Each branch is evaluated only when at least one lane needs it. x = 0
////      gives -inf, x > 0 and NaN give NaN, x = -inf gives 0 (exceptional lanes go through the scalar repairs of
////      nico_exp_1, nico_log_1, nico_log1p_1).
// x >= -ln 2 (NaN: false)
// //  compute log(1+x) without losing precision for small values of x. //  see: https://www.johndcook.com/cpp_log_one_plus_x.html
// inline   double nico_log1p_exp_1(const double x)   {
//   
//   if (x > 0.0) return x + nico_log1p_1(nico_exp_1(-x));
//   return nico_log1p_1(nico_exp_1(x));
//   
// }
///////////////////  fns - inv_logit   ----------------------------------------------------------------------------------------------------------------
///////////////////  fns - log_inv_logit   ----------------------------------------------------------------------------------------------------------------
////////////
// inline     __m512d  nico_log_inv_logit_for_x_pos_AVX512(const __m512d x )  {
//   
//            const __m512d   m_x = _mm512_sub_pd(zero, x);
//            return   _mm512_sub_pd(zero, nico_log1p_exp_1_AVX512(nico_exp_1_AVX512(m_x))); ///  - log1p(exp(-x))
//            
// }
// 
// inline      __m512d  nico_log_inv_logit_for_x_neg_AVX512(const __m512d x )  {
//   
//           return _mm512_mask_blend_pd(_mm512_cmp_pd_mask(x,  log_eps, _CMP_GT_OQ), /// if x > log_eps = -18.420680743952367
//                                       x,  /// if x < log_eps
//                                       _mm512_sub_pd(x,   nico_log1p_1_AVX512(nico_exp_1_AVX512(x))) ); /// if x > log_eps = -18.420680743952367
//   
// }
// 
// inline      __m512d  nico_log_inv_logit_AVX512(const __m512d x )  {
//   
//           return  _mm512_mask_blend_pd(_mm512_cmp_pd_mask(x,  zero, _CMP_GE_OQ),  // x > 0
//                                        nico_log_inv_logit_for_x_neg_AVX512(x),   /// x < 0
//                                        nico_log_inv_logit_for_x_pos_AVX512(x));  // x > 0
//   
// }
// For positive x: log(1/(1 + exp(-x))) = -log(1 + exp(-x))
// -x
// exp(-x)
// log(1 + exp(-x))
// -log(1 + exp(-x))
// For negative x: log(exp(x)/(1 + exp(x))) = x - log(1 + exp(x))
// exp(x)
// log(1 + exp(x))
// x - log(1 + exp(x))
// Blend based on x > log_eps 
// if x <= log_eps
// if x > log_eps
// if x < 0 
// if x >= 0
// 
// inline   double  nico_log_inv_logit_for_x_pos(const double x )  {
//   return    - nico_log1p_exp_1(-x);  // return    - nico_log1p_1(nico_exp_1(-x));
// }
// 
// inline   double  nico_log_inv_logit_for_x_neg(const double x )  {
//   const double log_eps = -18.420680743952367;
//   if (x > log_eps) return  x - nico_log1p_1(nico_exp_1(x));
//   return x;
// }
//  
// inline   double  nico_log_inv_logit(const double x )  {
//   if (x > 0.0) return  nico_log_inv_logit_for_x_pos(x);
//   return  nico_log_inv_logit_for_x_neg(x);
// } 
// 
//  
// 
////////////
/// if x > log_eps = -18.420680743952367
// x > 0
// x > 0
/////////////////////  fns - Phi_approx   ----------------------------------------------------------------------------------------------------------------
///////////////////  fns - inv_Phi_approx   -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// logit first
// -0.34176618822627863*m_logit_p;
// now do arc_sinh part
// Avoid 1 / p overflowing for valid subnormal probabilities. The usual path below is unchanged.
// log((1-p)/p), stable near 1
// -0.34176618822627863*m_logit_p;
// now do arc_sinh part
// __mmask8 is_x_valid_prob =  _mm512_cmp_pd_mask(x,  zero, _CMP_GT_OQ) & _mm512_cmp_pd_mask(x,  one, _CMP_LT_OQ); // if x in (0, 1)
// return   _mm512_mask_blend_pd(is_x_valid_prob,
//                               _mm512_mask_blend_pd( (_mm512_cmp_pd_mask(x,  one, _CMP_GT_OQ) | _mm512_cmp_pd_mask(x,  zero, _CMP_GT_OQ)),  //  if x > 1 or x < 0 (i.e. = NaN / throw error)
//                                                    _mm512_mask_blend_pd( _mm512_cmp_pd_mask(x,  zero, _CMP_EQ_OQ),
//                                                                          _mm512_set1_pd(INFINITY), // if x = 1
//                                                                          _mm512_set1_pd(-INFINITY)),  // if x = 0
//                                                    _mm512_set1_pd(std::numeric_limits<double>::quiet_NaN()) ),
//                               _mm512_mul_pd(_mm512_set1_pd(2.7472242570765295),   _mm512_div_pd(_mm512_sub_pd(_mm512_mul_pd(exp_x_i, exp_x_i), one), exp_x_i) ));
// const __m512d  inv_Phi_approx_c1 = _mm512_set1_pd(-0.34176618822627863);
// now do arc_sinh part
// const __m512d  inv_Phi_approx_c1 = _mm512_set1_pd(-0.34176618822627863);
// now do arc_sinh part
///////////////////  fns - log_Phi_approx   ----------------------------------------------------------------------------------------------------------------------------------------
//////////////////// ------------- tanh  --------------------------------------------------------------------------------------------------------------------------------------
// from: https://stackoverflow.com/questions/57870896/writing-a-portable-sse-avx-version-of-stdcopysign
// Extract the signed bit from srcSign
// Extract the number without sign of srcValue (abs(srcValue))
// Merge signed bit with number and return
///////////////////  fns - error functions   ----------------------------------------------------------------------------------------------------------------------------------------
// 
// 
// static const __m512d erf_part_1_c0 =      _mm512_set1_pd(-5.6271698391213282e-18);
// static const __m512d erf_part_1_c1 =      _mm512_set1_pd(4.8565951797366214e-16);
// static const __m512d erf_part_1_c2 =      _mm512_set1_pd(-1.9912968283386570e-14);
// static const __m512d erf_part_1_c3 =      _mm512_set1_pd(5.1614612434698227e-13);
// static const __m512d erf_part_1_c4 =      _mm512_set1_pd(-9.4934693745934645e-12);
// static const __m512d erf_part_1_c5 =      _mm512_set1_pd(1.3183034417605052e-10);
// static const __m512d erf_part_1_c6 =      _mm512_set1_pd(-1.4354030030292210e-09);
// static const __m512d erf_part_1_c7 =      _mm512_set1_pd(1.2558925114413972e-08);
// static const __m512d erf_part_1_c8 =      _mm512_set1_pd(-8.9719702096303798e-08);
// static const __m512d erf_part_1_c9 =      _mm512_set1_pd(5.2832013824348913e-07);
// static const __m512d erf_part_1_c10 =     _mm512_set1_pd(-2.5730580226082933e-06);
// static const __m512d erf_part_1_c11 =     _mm512_set1_pd(1.0322052949676148e-05);
// static const __m512d erf_part_1_c12 =     _mm512_set1_pd(-3.3555264836700767e-05);
// static const __m512d erf_part_1_c13 =     _mm512_set1_pd(8.4667486930266041e-05);
// static const __m512d erf_part_1_c14 =     _mm512_set1_pd(-1.4570926486271945e-04);
// static const __m512d erf_part_1_c15 =     _mm512_set1_pd(7.1877160107954648e-05);
// static const __m512d erf_part_1_c16 =     _mm512_set1_pd(4.9486959714661590e-04);
// static const __m512d erf_part_1_c17 =     _mm512_set1_pd(-1.6221099717135270e-03);
// static const __m512d erf_part_1_c18 =     _mm512_set1_pd(1.6425707149019379e-04);
// static const __m512d erf_part_1_c19 =     _mm512_set1_pd(1.9148914196620660e-02);
// static const __m512d erf_part_1_c20 =     _mm512_set1_pd(-1.0277918343487560e-1);
// static const __m512d erf_part_1_c21 =     _mm512_set1_pd(-6.3661844223699315e-1);
// static const __m512d erf_part_1_c22 =     _mm512_set1_pd(-1.2837929411398119e-1);
// 
// 
// ///// see: https://math.stackexchange.com/questions/42920/efficient-and-accurate-approximation-of-error-function   // max ulp error = 0.97749 (USE_EXPM1 = 1); 1.05364 (USE_EXPM1 = 0)
// inline    __m512d nico_erf_wo_checks_part_1_upper_AVX512(const __m512d a,
//                                                          const __m512d t,
//                                                          const __m512d s) {
// 
//       __m512d r, u;
// 
//       r = _mm512_fmadd_pd (erf_part_1_c0, t, erf_part_1_c1); // -0x1.9f363ba3b515dp-58, 0x1.17f6b1d68f44bp-51
//       u = _mm512_fmadd_pd (erf_part_1_c2, t, erf_part_1_c3); // -0x1.66b85b7fbd01ap-46, 0x1.22907eebc22e0p-41
//       r = _mm512_fmadd_pd (r, s, u);
//       u = _mm512_fmadd_pd (erf_part_1_c4, t, erf_part_1_c5); // -0x1.4e0591fd97592p-37, 0x1.21e5e2d8544d1p-33
//       r = _mm512_fmadd_pd (r, s, u);
//       u = _mm512_fmadd_pd (erf_part_1_c6, t, erf_part_1_c7); // -0x1.8a8f81b7e0e84p-30, 0x1.af85793b93d2fp-27
//       r = _mm512_fmadd_pd (r, s, u);
//       u = _mm512_fmadd_pd (erf_part_1_c8, t, erf_part_1_c9); // -0x1.8157db0edbfa8p-24, 0x1.1ba3c453738fdp-21
//       r = _mm512_fmadd_pd (r, s, u);
//       u = _mm512_fmadd_pd (erf_part_1_c10, t, erf_part_1_c11); // -0x1.595999b7e922dp-19, 0x1.5a59c27b3b856p-17
//       r = _mm512_fmadd_pd (r, s, u);
//       u = _mm512_fmadd_pd (erf_part_1_c12, t, erf_part_1_c13); // -0x1.197b61ee37123p-15, 0x1.631f0597f62b8p-14
//       r = _mm512_fmadd_pd (r, s, u);
//       u = _mm512_fmadd_pd (erf_part_1_c14, t, erf_part_1_c15); // -0x1.319310dfb8583p-13, 0x1.2d798353da894p-14
//       r = _mm512_fmadd_pd (r, s, u);
//       u = _mm512_fmadd_pd (erf_part_1_c16, t, erf_part_1_c17); //  0x1.037445e25d3e5p-11,-0x1.a939f51db8c06p-10
//       r = _mm512_fmadd_pd (r, s, u);
//       u = _mm512_fmadd_pd (erf_part_1_c18, t, erf_part_1_c19); //  0x1.5878d80188695p-13, 0x1.39bc5e0e9e09ap-6
//       r = _mm512_fmadd_pd (r, s, u);
//       r = _mm512_fmadd_pd (r, t, erf_part_1_c20); // -0x1.a4fbc8f8ff7dap-4
//       r = _mm512_fmadd_pd (r, t, erf_part_1_c21); // -0x1.45f2da3ae06f8p-1
//       r = _mm512_fmadd_pd (r, t, erf_part_1_c22); // -0x1.06ebb92d9ffa8p-3
//       r = _mm512_fmadd_pd (r, t, -t);
// 
//       r = _mm512_sub_pd(one, nico_exp_1_AVX512(r)); // r = 1.0 - nico_exp_1_wo_checks_AVX512(r);
// 
//       return CopySign(a, r);    //return copysign(r, a);
// 
// }
// 
// 
// static const __m512d erf_part_2_c0 =      _mm512_set1_pd(-7.7794684889591997e-10);
// static const __m512d erf_part_2_c1 =      _mm512_set1_pd(1.3710980398024347e-8);
// static const __m512d erf_part_2_c2 =      _mm512_set1_pd(-1.6206313758492398e-7);
// static const __m512d erf_part_2_c3 =      _mm512_set1_pd( 1.6447131571278227e-6);
// static const __m512d erf_part_2_c4 =      _mm512_set1_pd(-1.4924712302009488e-5);
// static const __m512d erf_part_2_c5 =      _mm512_set1_pd(1.2055293576900605e-4);
// static const __m512d erf_part_2_c6 =      _mm512_set1_pd(-8.5483259293144627e-4);
// static const __m512d erf_part_2_c7 =      _mm512_set1_pd(5.2239776061185055e-3);
// static const __m512d erf_part_2_c8 =      _mm512_set1_pd(-2.6866170643111514e-2);
// static const __m512d erf_part_2_c9 =      _mm512_set1_pd(1.1283791670944182e-1);
// static const __m512d erf_part_2_c10 =     _mm512_set1_pd(-3.7612638903183515e-1);
// static const __m512d erf_part_2_c11 =     _mm512_set1_pd(1.2837916709551256e-1);
// 
// 
// 
// // see: https://math.stackexchange.com/questions/42920/efficient-and-accurate-approximation-of-error-function //   // max ulp error = 1.01912
// inline    __m512d nico_erf_wo_checks_part_2_lower_AVX512(const __m512d a,
//                                                          const __m512d t,
//                                                          const __m512d s) {
// 
//       __m512d r, u;
// 
// 
//       r =           erf_part_2_c0;                                   // -0x1.abae491c44131p-31
//       r = _mm512_fmadd_pd (r, s, erf_part_2_c1); //  0x1.d71b0f1b10071p-27
//       r = _mm512_fmadd_pd (r, s, erf_part_2_c2); // -0x1.5c0726f04dbc7p-23
//       r = _mm512_fmadd_pd (r, s, erf_part_2_c3); //  0x1.b97fd3d9927cap-20
//       r = _mm512_fmadd_pd (r, s, erf_part_2_c4); // -0x1.f4ca4d6f3e232p-17
//       r = _mm512_fmadd_pd (r, s, erf_part_2_c5); //  0x1.f9a2baa8fedc2p-14
//       r = _mm512_fmadd_pd (r, s, erf_part_2_c6); // -0x1.c02db03dd71bbp-11
//       r = _mm512_fmadd_pd (r, s, erf_part_2_c7); //  0x1.565bccf92b31ep-8
//       r = _mm512_fmadd_pd (r, s, erf_part_2_c8); // -0x1.b82ce311fa94bp-6
//       r = _mm512_fmadd_pd (r, s, erf_part_2_c9); //  0x1.ce2f21a040d14p-4
//       r = _mm512_fmadd_pd (r, s, erf_part_2_c10); // -0x1.812746b0379bcp-2
//       r = _mm512_fmadd_pd (r, s, erf_part_2_c11); //  0x1.06eba8214db68p-3
// 
//       return _mm512_fmadd_pd(r, a, a);
// 
// }
// 
// 
// 
// //// // Combined fast erf function
// ///// see: https://math.stackexchange.com/questions/42920/efficient-and-accurate-approximation-of-error-function
// inline __m512d nico_erf_wo_checks_AVX512(const __m512d a) {
// 
//   const __m512d t = _mm512_abs_pd(a);
//   const __m512d s = _mm512_mul_pd(a, a);
// 
//   return _mm512_mask_blend_pd(
//     _mm512_cmp_pd_mask(t, one, _CMP_GT_OQ),
//     nico_erf_wo_checks_part_2_lower_AVX512(a, t, s),
//     nico_erf_wo_checks_part_1_upper_AVX512(a, t, s)
//   );
// 
// }
// 
// 
// 
// 
// 
// 
// 
// 
///////////////////  fns - Phi functions   ----------------------------------------------------------------------------------------------------------------------------------------
/// use AVX-512
// static const __m512d one_half =     _mm512_set1_pd(0.5);
// static const __m512d sqrt_2_recip = _mm512_set1_pd(0.707106781186547461715);
// 
// inline    __m512d nico_Phi_wo_checks_AVX512(const __m512d x) {
// 
//   return _mm512_mul_pd(one_half, _mm512_add_pd(one, nico_erf_wo_checks_AVX512(_mm512_mul_pd(x, sqrt_2_recip)) ) ) ;
// 
// }
//// ---- nico_Phi: relative-minimax rational approximation of the Mills ratio (replaces Abramowitz & Stegun 26.2.17):
////
////   Phi(-z) = phi(z) * R(z),   R(z) = Phi(-z) / phi(z) = sqrt(pi/2) * erfcx(z / sqrt(2)),   z = |x|,
////   R(z) ~= P6(z) / Q7(z)  on z in [0, 37.5],   max relative error 7.8e-12 (measured on a dense grid in mpmath, double coefficients),
////   so  Phi(-z) = exp(-z^2/2) * N6(z) / Q7(z)  with  N6 = P6 / sqrt(2 pi)  (the 1/sqrt(2 pi) of phi is folded into N6),
////   and Phi(x) = 1 - Phi(-x) for x > 0 (the same reflection as before).
////
//// Why: A&S 26.2.17 (Zelen & Severo 1964 / Hastings 1955 form, published) has absolute error <= 7.5e-8 but RELATIVE error 5e-5 at
//// x = -3, 8e-3 at -7.5 and 0.16 at -37.5, while every caller registers the exact density phi(x) as the gradient, so value and
//// gradient disagreed in the lower tail (and log(nico_Phi) was off by up to 0.17 nats). The rational form is relative-accurate for
//// every x, branch-free, and costs the same as A&S: one exp, one divide and 13 FMAs (A&S: 1 FMA + 9 multiplies + 4 adds).
////
//// Provenance: the Mills-ratio/rational form is standard (e.g. Cody 1969, Math. Comp. 23:631-637, uses rationals for erfc);
//// THESE coefficients were fitted with a relative-error Sanathanan-Koerner / Lawson
//// near-minimax fit in 40-digit mpmath, constrained to N6(0) = 0.5 exactly so that nico_Phi(0) = 0.5 with no jump at x = 0.
//// FAST_PHI_MILLS_Z_CAP caps only the rational's argument (the exp still sees the true z): it keeps N and Q finite for
//// |x| = inf / huge |x| (exp(-z^2/2) = 0 there, so the result is 0 / 1 exactly as before) and does not change any x in [-37.5, 8.25].
//// N6(z) = sum_k c[k] z^k   (includes 1/sqrt(2 pi))
//// Q7(z) = sum_k d[k] z^k
//// Phi(x) = exp(-z^2/2) * N(z) / Q(z) for x <= 0, 1 minus that for x > 0 (z = |x|): see the FAST_PHI_MILLS_* block above.
/////  std::fabs(x);
//// NaN x: min returns its 2nd operand (the cap), and the NaN still propagates through the exp below.
//// N / Q is formed BEFORE multiplying by the exp so the divide runs in parallel with nico_exp_1's polynomial instead of
//// after it (measured: (exp * N) / Q put the divide latency on the critical path and cost ~6% on AVX2).
//// Phi(-z) = exp(-z^2/2) * N(z) / Q(z)
//// if x < 0
//// if x > 0
// inline __m512d nico_Phi_AVX512(const __m512d x) {
//   
//         const __mmask8 is_x_gr_lower = _mm512_cmp_pd_mask(x, Phi_lower_bound, _CMP_GT_OQ); // true where x > -37.5 (in range)
//         const __mmask8 is_x_lt_upper = _mm512_cmp_pd_mask(x, Phi_upper_bound, _CMP_LT_OQ); // true where x < 8.25 (in range)
//         const __mmask8 is_x_in_range = is_x_gr_lower & is_x_lt_upper ; //// _mm512_kand(is_x_gr_lower, is_x_lt_upper);
//         
//         return _mm512_mask_blend_pd(
//                                       is_x_in_range,
//                                                       _mm512_mask_blend_pd( _mm512_cmp_pd_mask(x, Phi_lower_bound, _CMP_LT_OQ),  // if x NOT in range and x < -37.5
//                                                                             one,    // if x NOT in range and x NOT < -37.5 i.e. x must be > 8.25  
//                                                                             zero),    // if x NOT in range and x < -37.5
//                                       nico_Phi_wo_checks_AVX512(x)); /// if x is in range
//   
// }
// equality must not fall through to 1
// true where x < 8.25 (in range)
// Calculate the main Phi function for in-range values
// if x NOT in range and x NOT < -37.5 i.e. x must be > 8.25
// if x NOT in range and x < -37.5
// if x is in range
// Clamp results between 0 and 1
// Preserve NaN input rather than turning it into a valid probability through min/max.
///////////////////  fns - inverse-error functions (for inv_Phi)  ----------------------------------------------------------------------------------------------------------------------------------------
//  compute inverse error functions with maximum error of 2.35793 ulp  // see: https://stackoverflow.com/questions/27229371/inverse-error-function-in-c
// 
// 
// static const __m512d inv_erf_part_1_c0 =     _mm512_set1_pd(3.03697567e-10);
// static const __m512d inv_erf_part_1_c1 =     _mm512_set1_pd(2.93243101e-8);
// static const __m512d inv_erf_part_1_c2 =     _mm512_set1_pd(1.22150334e-6);
// static const __m512d inv_erf_part_1_c3 =     _mm512_set1_pd(2.84108955e-5);
// static const __m512d inv_erf_part_1_c4 =     _mm512_set1_pd(3.93552968e-4);
// static const __m512d inv_erf_part_1_c5 =     _mm512_set1_pd(3.02698812e-3);
// static const __m512d inv_erf_part_1_c6 =     _mm512_set1_pd(4.83185798e-3);
// static const __m512d inv_erf_part_1_c7 =     _mm512_set1_pd(-2.64646143e-1);
// static const __m512d inv_erf_part_1_c8 =     _mm512_set1_pd(8.40016484e-1);
//   
//   
//   
//   
// // see: https://math.stackexchange.com/questions/42920/efficient-and-accurate-approximation-of-error-function   // max ulp error = 0.97749 (USE_EXPM1 = 1); 1.05364 (USE_EXPM1 = 0)
// inline    __m512d nico_inv_erf_wo_checks_part_1_upper_AVX512(const __m512d a, 
//                                                              const __m512d t) {
//     
//     __m512d p =            inv_erf_part_1_c0; //  0x1.4deb44p-32
//     p = _mm512_fmadd_pd (p, t,  inv_erf_part_1_c1); //  0x1.f7c9aep-26
//     p = _mm512_fmadd_pd (p, t,  inv_erf_part_1_c2); //  0x1.47e512p-20
//     p = _mm512_fmadd_pd (p, t,  inv_erf_part_1_c3); //  0x1.dca7dep-16
//     p = _mm512_fmadd_pd (p, t,  inv_erf_part_1_c4); //  0x1.9cab92p-12
//     p = _mm512_fmadd_pd (p, t,  inv_erf_part_1_c5); //  0x1.8cc0dep-9
//     p = _mm512_fmadd_pd (p, t,  inv_erf_part_1_c6); //  0x1.3ca920p-8
//     p = _mm512_fmadd_pd (p, t,  inv_erf_part_1_c7); // -0x1.0eff66p-2
//     p = _mm512_fmadd_pd (p, t,  inv_erf_part_1_c8); //  0x1.ae16a4p-1
//     
//     return  _mm512_mul_pd(a, p); // a * p;
//     
//   }
//   
//   static const __m512d inv_erf_part_2_c0 =     _mm512_set1_pd(5.43877832e-9);
//   static const __m512d inv_erf_part_2_c1 =     _mm512_set1_pd(1.43285448e-7);
//   static const __m512d inv_erf_part_2_c2 =     _mm512_set1_pd(1.22774793e-6);
//   static const __m512d inv_erf_part_2_c3 =     _mm512_set1_pd(1.12963626e-7);
//   static const __m512d inv_erf_part_2_c4 =     _mm512_set1_pd(-5.61530760e-5);
//   static const __m512d inv_erf_part_2_c5 =     _mm512_set1_pd(-1.47697632e-4);
//   static const __m512d inv_erf_part_2_c6 =     _mm512_set1_pd(2.31468678e-3);
//   static const __m512d inv_erf_part_2_c7 =     _mm512_set1_pd(1.15392581e-2);
//   static const __m512d inv_erf_part_2_c8 =     _mm512_set1_pd(-2.32015476e-1);
//   static const __m512d inv_erf_part_2_c9 =     _mm512_set1_pd(8.86226892e-1);
//   
//   // see: https://math.stackexchange.com/questions/42920/efficient-and-accurate-approximation-of-error-function //   // max ulp error = 1.01912
//   inline    __m512d nico_inv_erf_wo_checks_part_2_lower_AVX512(const __m512d a,
//                                                                const __m512d t) {
//     
//     __m512d p =            inv_erf_part_2_c0; //  0x1.4deb44p-32
//     p = _mm512_fmadd_pd (p, t,  inv_erf_part_2_c1); //  0x1.f7c9aep-26
//     p = _mm512_fmadd_pd (p, t,  inv_erf_part_2_c2); //  0x1.47e512p-20
//     p = _mm512_fmadd_pd (p, t,  inv_erf_part_2_c3); //  0x1.dca7dep-16
//     p = _mm512_fmadd_pd (p, t,  inv_erf_part_2_c4); //  0x1.9cab92p-12
//     p = _mm512_fmadd_pd (p, t,  inv_erf_part_2_c5); //  0x1.8cc0dep-9
//     p = _mm512_fmadd_pd (p, t,  inv_erf_part_2_c6); //  0x1.3ca920p-8
//     p = _mm512_fmadd_pd (p, t,  inv_erf_part_2_c7); // -0x1.0eff66p-2
//     p = _mm512_fmadd_pd (p, t,  inv_erf_part_2_c8); //  0x1.ae16a4p-1
//     p = _mm512_fmadd_pd (p, t,  inv_erf_part_2_c9); //  0x1.ae16a4p-1
//     
//     return  _mm512_mul_pd(a, p); // a * p;
//   
// }
// 
// 
// 
// static const __m512d inv_erf_c0 =      _mm512_set1_pd(6.125);
// 
// // see: https://math.stackexchange.com/questions/42920/efficient-and-accurate-approximation-of-error-function
// inline __m512d nico_inv_erf_wo_checks_AVX512(const __m512d a) {
//   
//       __m512d t = _mm512_fmadd_pd(a, _mm512_sub_pd(zero, a), one);
//       t = nico_log_1_AVX512(t);
//       
//       return _mm512_mask_blend_pd(
//         _mm512_cmp_pd_mask(_mm512_abs_pd(t), inv_erf_c0, _CMP_GT_OQ), 
//         nico_inv_erf_wo_checks_part_2_lower_AVX512(a, t),
//         nico_inv_erf_wo_checks_part_1_upper_AVX512(a, t));
//   
// }
//  
// 
// 
///////////////////  fns - inv_Phi functions   ----------------------------------------------------------------------------------------------------------------------------------------
// static const __m512d sqrt_2 =  _mm512_set1_pd(1.41421356237309514547);
// 
// inline    __m512d nico_inv_Phi_wo_checks_AVX512(const __m512d x) {
//   
//     return _mm512_mul_pd(sqrt_2, nico_inv_erf_wo_checks_AVX512( _mm512_sub_pd(_mm512_mul_pd(two, x), one )))  ;
//   
// }
/// vectorised, AVX-512 version of Stan fn provided by Sean Pinkney:  https://github.com/stan-dev/math/issues/2555 
///  CASE 2(a): if abs(q) > 0.425  AND   if r <= 5.0 
// std::cout << "calling  nico_Phi_wo_checks_case_2a_AVX512"   << "\n"; 
//std::cout << "r = " <<  r  << "\n";
///  CASE 2(a): if abs(q) > 0.425  AND   if r > 5.0 
// std::cout << "calling  nico_Phi_wo_checks_case_2b_AVX512"   << "\n";
//std::cout << "r = " <<  r  << "\n";
// inline __m512d nico_inv_Phi_wo_checks_case_2_AVX512(const __m512d p,
//                                                     const __m512d q) {  /// CASE 2 (i.e. one of case 2(a) or 2(b) depending on value of r)
//   
//         const __mmask8 is_q_gr_0 = _mm512_cmp_pd_mask(q, _mm512_setzero_pd(), _CMP_GT_OQ);
//          
//         ////   compute r 
//         __m512d r_if_q_gr_0 =  _mm512_sub_pd(_mm512_set1_pd(1.0), p);
//         __m512d r_if_q_lt_0 = p;
//         __m512d r = _mm512_mask_blend_pd(is_q_gr_0, r_if_q_gr_0, r_if_q_lt_0);
//         r = _mm512_sqrt_pd(_mm512_sub_pd(_mm512_setzero_pd(), nico_log_1_AVX512(r)));
//         
//         /// then call either case 2(a) fn (if r <= 5.0) or case 2(b) fn (if r > 5.0)  
//         const __mmask8 is_r_gr_5 = _mm512_cmp_pd_mask(r, _mm512_set1_pd(5.0), _CMP_GT_OQ);
//         __m512d val =  _mm512_mask_blend_pd( is_r_gr_5, 
//                                              nico_inv_Phi_wo_checks_case_2b_AVX512(p, r),  /// if r > 5.0
//                                              nico_inv_Phi_wo_checks_case_2a_AVX512(p, r));
//         
//         // Flip the sign if q is negative
//         const __mmask8 is_q_lt_zero = _mm512_cmp_pd_mask(q, _mm512_setzero_pd(), _CMP_LT_OQ); // _mm512_cmplt_pd_mask(q, _mm512_set1_pd(0.0));
//         
//         val = _mm512_mask_blend_pd(is_q_lt_zero, 
//                                    _mm512_sub_pd(_mm512_setzero_pd(), val),   //// if q < 0
//                                    val); //// if   q => 0 
//         
//         return val;
//   
//   
// }
// q>0 -> 1-p, else p
// r <= 5
// No need to evaluate the extreme-tail polynomial unless a lane will actually use it.
// q<0 -> -val
///  CASE 1: if abs(q) <= 0.425
// The tail result would be discarded in every lane. Avoid its log and two rational polynomials.
// central lanes -> v1
// inline __m512d nico_inv_Phi_wo_checks_AVX512(const __m512d p) {
//       
//       const __m512d q = _mm512_sub_pd(p, _mm512_set1_pd(0.50));
//       const __mmask8 is_q_le_threshold = _mm512_cmp_pd_mask(_mm512_abs_pd(q), _mm512_set1_pd(0.425), _CMP_LE_OQ);
//        
//       if (_mm512_kortestc(is_q_le_threshold, is_q_le_threshold)) {    // All elements in q are outside the |q| <= 0.425 range, so call case 2
//         
//         return nico_inv_Phi_wo_checks_case_2_AVX512(p, q); 
//         
//       } else { // At least one element is within the |q| <= 0.425 range, so call case 1
//          
//         return nico_inv_Phi_wo_checks_case_1_AVX512(p, q);
//       }
//   
// } 
// x >= NORMAL_TAIL_THRESH
// ---------------------------------------------------------------------------------------------
//  Relative-accurate normal tail kernels for the GHK log-scale branches.
//  Include AFTER the nico_and_approx_*_fns.hpp headers (uses nico_exp_1_*, nico_log_1_*, nico_Phi_*,
//  nico_inv_Phi_wo_checks_case_{1,2a,2b}_* and _mm256_abs_pd / _mm512_abs_pd).
//
//  Everything is built on ONE function, the Mills ratio
//        R(x) = (1 - Phi(x)) / phi(x)  =  sqrt(pi/2) * erfcx(x / sqrt(2)),      x >= 0,
//  which is smooth, O(1/x) and has no cancellation. Then:
//        Phi(x)          = phi(x) * R(-x)                               (x < 0)
//        log Phi(x)      = -x^2/2 - 0.5*log(2*pi) + log R(-x)           (x < 0)
//        log(1 - Phi(x)) = log Phi(-x)
//        d/dx log Phi(x) = phi(x)/Phi(x) = 1 / R(-x)                     (x < 0)
//        inv_Phi from log p : AS241 tail case needs only r = sqrt(-log(min(p, 1-p)))
//        dZ/dlog_p    =  Phi(Z)/phi(Z)     = exp(log_p    + Z^2/2 + 0.5 log 2pi)  (=  R(-Z) for Z<0, stable side)
//        dZ/dlog_1m_p = -(1-Phi(Z))/phi(Z) = -exp(log_1m_p + Z^2/2 + 0.5 log 2pi)  (= -R( Z) for Z>0, stable side)
//
//  R(x) is evaluated with the Laplace continued fraction  R(x) = 1/(x + 1/(x + 2/(x + 3/(x + ...))))
//  by backward recurrence of depth MILLS_CF_DEPTH. Depth 40 is full double precision for x >= 2.5
//  (run normal_tail_self_test() once; lower the depth after benchmarking if needed - the tail
//  path is only taken for |x| > NORMAL_TAIL_THRESH, and a rational minimax in 1/x can replace the
//  fraction later; the self-test validates any replacement).
//
//  Body (|x| <= NORMAL_TAIL_THRESH) keeps the existing nico_Phi (Abramowitz-Stegun). Its absolute
//  error 7.5e-8 is a relative error of 1.2e-5 at the threshold and smaller inside; the tail side is
//  exact, so the join is continuous to ~1e-5 in log Phi. The remaining tail error is the nico_log implementation's
//  (~1e-8); swap in the double-precision log constant for machine precision.
// ---------------------------------------------------------------------------------------------
//// ---- Mills ratio for the lower tail of log_Phi / dlog_Phi_dx, z >= NORMAL_TAIL_THRESH (2.5): R(z) = Phi(-z) /
////      phi(z) = P(t) / (Q(t) z),  t = 1/z^2,  P and Q of degree 8 with P(0) = Q(0) = 1 (the exact 1/z
////      asymptote; t -> 0 without overflow for any z), fitted to the relative error of R on z in [2.5, inf)
////      (relative-weight least squares iterated to near-minimax, in 70-digit mpmath, in u = t / 0.16: fit error
////      2.1e-18). Evaluated as 1/R = z Q / P = z (1 - w), w = D(t) / P(t), D = P - Q: 1/R = z - z w is one FMA
////      and |w| <= 0.16, so the rounding of w hardly reaches 1/R. nico_log_Phi takes log of R = 1 / (1/R) (the
////      same argument the continued fraction gave nico_log_1). Two or three divisions per vector instead of 41
////      for the 40-term continued fraction (still defined above as nico_mills_ratio_tail_*, now unused),
////      1.35e-12 at z = 2.5.
//// P(t) = sum_k c[k] t^k
//// D = P - Q
//// Q(t) = sum_k d[k] t^k
// 1/R(z) = z (1 - w), w = D(t) / P(t), t = 1/z^2, for the tail Mills ratio R (see the coefficient block above);
// z = inf gives inf (w = 0 there, and inf * 0 would give NaN)
// D(t), D(0) = 0
// P(t)
//// ---- previous version (tail and body evaluated for every vector), kept for reference:
// // log Phi(x), all x; relative-accurate in the lower tail.
// ALWAYS_INLINE __m512d nico_log_Phi_AVX512(const __m512d x) {
//
//       const __m512d  thresh   = _mm512_set1_pd(NORMAL_TAIL_THRESH);
//       const __m512d  ax       = _mm512_abs_pd(x);
//       const __m512d  R        = nico_mills_ratio_tail_AVX512(_mm512_max_pd(ax, thresh));   // body lanes compute R(2.5), then discarded
//       const __m512d  log_tail = _mm512_add_pd(_mm512_fnmadd_pd(_mm512_set1_pd(0.5), _mm512_mul_pd(x, x), _mm512_set1_pd(NEG_HALF_LOG_2PI)),
//                                               nico_log_1_AVX512(R));
//       const __m512d  log_body = nico_log_1_AVX512(nico_Phi_AVX512(x));
//       const __mmask8 is_tail  = _mm512_cmp_pd_mask(x, _mm512_sub_pd(_mm512_setzero_pd(), thresh), _CMP_LT_OQ);   // x < -2.5
//       return _mm512_mask_blend_pd(is_tail, log_body, log_tail);
//
// }
// log Phi(x), all x; relative-accurate in the lower tail.
// Tail (x < -2.5): -x^2/2 - 0.5 log(2 pi) + log R(-x), R by the degree-8 rational above (body lanes evaluate R(2.5),
// then discarded). Lane-mask tests: the tail is evaluated only when at least one lane is in the tail, and the body
// (log of nico_Phi, unchanged) only when at least one lane is not.
// x < -2.5
// no tail lane: body only
// every lane in the tail
// log(1 - Phi(x)) = log Phi(-x)
//// ---- previous version (tail and body evaluated for every vector), kept for reference:
// // d/dx log Phi(x) = phi(x)/Phi(x).  Tail: 1/R(-x).  Body: phi / (the same nico_Phi the lp used).
// ALWAYS_INLINE __m512d nico_dlog_Phi_dx_AVX512(const __m512d x) {
//
//       const __m512d  thresh  = _mm512_set1_pd(NORMAL_TAIL_THRESH);
//       const __m512d  ax      = _mm512_abs_pd(x);
//       const __m512d  R       = nico_mills_ratio_tail_AVX512(_mm512_max_pd(ax, thresh));
//       const __m512d  d_tail  = _mm512_div_pd(_mm512_set1_pd(1.0), R);
//       const __m512d  phi     = _mm512_mul_pd(_mm512_set1_pd(INV_SQRT_2PI),
//                                              nico_exp_1_AVX512(_mm512_mul_pd(_mm512_set1_pd(-0.5), _mm512_mul_pd(x, x))));
//       const __m512d  d_body  = _mm512_div_pd(phi, nico_Phi_AVX512(x));
//       const __mmask8 is_tail = _mm512_cmp_pd_mask(x, _mm512_sub_pd(_mm512_setzero_pd(), thresh), _CMP_LT_OQ);
//       return _mm512_mask_blend_pd(is_tail, d_body, d_tail);
//
// }
// d/dx log Phi(x) = phi(x)/Phi(x).  Tail: 1/R(-x) = z (1 - D/P) (the rational above).  Body: phi / (the same
// nico_Phi the lp used). Lane-mask tests as in nico_log_Phi_AVX512.
// x < -2.5
// no tail lane: body only
// every lane in the tail
// d/dx log(1 - Phi(x)) = -nico_dlog_Phi_dx_AVX512(-x)
// inv_Phi given log(p) and log(1-p). AS241; tail case fed r = sqrt(-min(log_p, log_1m_p)) directly.
// only consumed by the central case
//// tail
// log(min(p, 1-p))
// r <= 5
// r >  5
// p < 0.5 -> Z < 0
//// central
// dZ/dv through the stable side (see scalar version for the identities).
// keeps the UNUSED side finite
// Z<0 -> via_lo, else via_hi

#include <NicoSIMD/math/fast_and_approx_AVX512_fns.hpp>






















