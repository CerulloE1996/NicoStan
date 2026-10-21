#pragma once

// #include <R_ext/Print.h>  // For REprintf
// #include <R_ext/Utils.h>  // For R_FlushConsole
//// #if defined(__AVX2__) && !(defined(__AVX512VL__) && defined(__AVX512F__) && defined(__AVX512DQ__)) // use AVX2 if AVX-512 not available 
// compiled on every AVX2-capable build, including AVX-512 ones — the dispatch layer needs both kernel sets
//// -------------------------------------------------------------------------------------------------------------------------------------------------------------
// Simple test function that just multiplies vector by 2
// // _mm256_zeroupper();  // Reset AVX state
//// _mm256_zeroupper();  // Reset AVX state
//// -------------------------------------------------------------------------------------------------------------------------------------------------------------
// _mm256_zeroupper();  // Reset AVX state
// _mm256_zeroupper();  // Reset AVX state
// is_finite_mask and is_not_NaN_mask for AVX2
// _mm256_zeroupper();  // Reset AVX state
// _mm256_zeroupper();  // Reset AVX state
// _mm256_zeroupper();  // Reset AVX state
// bookmark (this doesnt seem to be available on g++ at all at time of writing even with all AVX256 flags!)
//#if !defined(__AVX256VL__)
// this uses AVX2 instructions so moree widely supported
///////////////////  fns - exp   -----------------------------------------------------------------------------------------------------------------------------
// _mm256_zeroupper();  // Reset AVX state
// _mm256_zeroupper();  // Reset AVX state
// // Replace _mm256_srai_epi64 (AVX-512 fn) with this helper function (forAVX2)
// ALWAYS_INLINE __m256i   avx2_srai_epi64 VECTORCALL(  const __m256i x, 
//                                                      const int count) {
//   
//        // _mm256_zeroupper();  // Reset AVX state
//     
//        ALIGN32  __m256i const x_aligned = x;
//     
//        // Get the sign bits
//        ALIGN32  __m256i const sign_bits = _mm256_and_si256(_mm256_srai_epi32(_mm256_permute4x64_epi64(x_aligned, 0xF5), 31),
//                                                               _mm256_set1_epi64x(0xFFFFFFFF00000000));
//        
//        // Perform arithmetic shift on lower 32 bits
//        ALIGN32  __m256i const shift_result = _mm256_srai_epi32(x_aligned, count);
//        
//        // Combine results
//        ALIGN32  __m256i const res = _mm256_or_si256(shift_result,  
//                                          _mm256_and_si256(sign_bits, 
//                                                           _mm256_set1_epi64x(0xFFFFFFFF00000000))); 
//        // _mm256_zeroupper();  // Reset AVX state
//        return res;
//      
// }
// // Fast implementation of ldexp (multiply by power of 2) using AVX2 intrinsics
// // Computes: a * 2^i for 4 pairs of values simultaneously
// // Handles edge cases by splitting large exponents into two steps
// ALWAYS_INLINE __m256d nico_ldexp_2 VECTORCALL( const __m256d AVX_a,
//                                                const __m256i AVX_i) {
//   
//         // _mm256_zeroupper();  // Reset AVX state
//   
//         ALIGN32  __m256d const AVX_a_aligned = AVX_a;
//         ALIGN32  __m256i const AVX_i_aligned = AVX_i;
//   
//         const uint64_t shift_val_52 = 52;  // Double-precision exponent shift
//         const uint64_t shift_val_63 = 63;  // Sign bit position
//         const uint64_t shift_val_1000 = 1000;  // Safe threshold for exponents
//         const uint64_t shift_val_neg_2000 = 2000;  // Double threshold for negative case
//         const uint64_t mask = 0xF;  // 4-bit comparison mask (one bit per double)
//   
//         // Get sign mask: for each 64-bit integer, shift right by 63 bits (arithmetic shift)
//         // This replicates the sign bit across all bits: 0x0000... for positive, 0xFFFF... for negative    ;
//         ALIGN32  __m256i const neg_mask = avx2_srai_epi64(AVX_i_aligned, shift_val_63);
//   
//         // Calculate absolute value using the formula: abs(x) = (x XOR sign_mask) - sign_mask
//         // 1. XOR with sign mask flips all bits if negative, keeps same if positive
//         // 2. Subtract sign mask adds 1 for negative numbers (two's complement)
//         ALIGN32  __m256i const abs_i = _mm256_sub_epi64(_mm256_xor_si256(AVX_i_aligned, neg_mask), neg_mask);
//   
//         // Create vector with all elements = 1000 (our safe threshold for exponents)
//         ALIGN32  __m256i const threshold = _mm256_set1_epi64x(shift_val_1000);
//   
//         // Compare abs_i with threshold by subtraction
//         // If abs_i > threshold, result will be positive
//         ALIGN32  __m256i const cmp = _mm256_sub_epi64(abs_i, threshold);
//   
//         // Create comparison mask by getting sign bits of comparison result
//         // Will be 0xFFFF... where abs_i <= threshold, 0x0000... where abs_i > threshold
//         ALIGN32  __m256i const cmp_mask = avx2_srai_epi64(cmp, shift_val_63);
//   
//         // Convert comparison mask to 4-bit integer (one bit per double)
//         // If result != 0xF (not all bits set), then at least one value exceeded threshold
//         if (_mm256_movemask_pd(_mm256_castsi256_pd(cmp_mask)) != mask) {
//   
//                 // Handle large exponents by splitting into two steps
//   
//                 // Create i1 = ±1000 based on original sign:
//                 // 1. AND neg_mask with -2000 gives -2000 for negative numbers, 0 for positive
//                 // 2. XOR with threshold (1000) gives -1000 for negative numbers, 1000 for positive
//                 ALIGN32  __m256i const i1 = _mm256_xor_si256(_mm256_and_si256(neg_mask, _mm256_set1_epi64x(shift_val_neg_2000)), threshold);
//   
//                 // Calculate remaining exponent: i2 = original_i - i1
//                 ALIGN32  __m256i const i2 = _mm256_sub_epi64(AVX_i_aligned, i1);
//   
//                 // First scaling step: multiply by 2^i1
//                 // 1. Shift i1 left by 52 to position it in double's exponent field
//                 // 2. Add to bit pattern of input doubles (effectively multiplying by 2^i1)
//                 ALIGN32  __m256d const mid = _mm256_castsi256_pd(_mm256_add_epi64(_mm256_slli_epi64(i1, shift_val_52), _mm256_castpd_si256(AVX_a)));
//   
//                 // Second scaling step: multiply intermediate result by 2^i2
//                 // Same process as above but with i2 and intermediate result
//                 // return _mm256_castsi256_pd(_mm256_add_epi64(_mm256_slli_epi64(i2, 52), _mm256_castpd_si256(mid)));
//                 ALIGN32  __m256d const res = nico_ldexp(mid, i2);
//                 return  res;
//   
//         }
//   
//         // For small exponents, perform direct scaling:
//         // 1. Shift exponent left by 52 bits to align with double's exponent field
//         // 2. Add to original number's bit pattern (effectively multiplying by 2^i)
//         // return _mm256_castsi256_pd(_mm256_add_epi64(_mm256_slli_epi64(AVX_i, 52), _mm256_castpd_si256(AVX_a)));
//         ALIGN32  __m256d const res = nico_ldexp(AVX_a_aligned, AVX_i_aligned);
//         // _mm256_zeroupper();  // Reset AVX state
//         return res;
// 
// }
// Helper function for AVX2 64-bit conversion 
// ALWAYS_INLINE __m256i avx2_cvtpd_epi64 VECTORCALL(const __m256d x) {
//   
//         ALIGN32  __m256d const x_aligned = x;
//    
//         // Extract doubles and convert to int64 one at a tim
//         ALIGN32 int64_t result[4];
//         ALIGN32 double temp[4];
//   
//         _mm256_storeu_pd(temp, x_aligned);
//        
//         for(int i = 0; i < 4; i++) {
//             #ifdef _MSC_VER
//                    result[i] = (int64_t)_mm_cvtsd_si64(_mm_load_sd(&temp[i]));  // Use SSE2 conversion
//             #else
//                    result[i] = (int64_t)std::llrint(temp[i]);     // Use llrint for proper rounding
//             #endif
//             // result[i] = (int64_t)_mm_cvtsd_si64(_mm_load_sd(&temp[i]));  // Use SSE2 conversion
//         }
//        
//         //// return _mm256_load_si256((__m256i*)result);
//         ALIGN32  __m256i const res = _mm256_load_si256(reinterpret_cast<const __m256i*>(static_cast<void*>(result))); 
//         return res;
//  
// }
// Round-to-nearest-even double -> int64 for |x| < 2^51. Matches _mm512_cvtpd_epi64 under default
// MXCSR, and matches std::llrint on the current path (|t| < ~1022 whenever the checked exp is used).
// 1.5 * 2^52
// Adapted from: https://stackoverflow.com/questions/48863719/fastest-implementation-of-exponential-function-using-avx
// added   (optional) extra degree(s) for poly approx (oroginal float fn had 4 degrees) - using "minimaxApprox" R package to find coefficient terms
// R code:    minimaxApprox::minimaxApprox(fn = exp, lower = -0.346573590279972643113, upper = 0.346573590279972643113, degree = 5, basis ="Chebyshev")
// REprintf("Entering nico_exp_1_wo_checks_AVX2 \n");
// R_FlushConsole();
// _mm256_zeroupper();  // Reset AVX state
// REprintf("Before x_aligned \n");
// R_FlushConsole();
// REprintf("After x_aligned \n");
// R_FlushConsole();
/* log2(e) */
/* -log(2)_hi, exactly representable */
/* -log(2)_lo */
// /* coefficients for core approximation to exp() in [-log(2)/2, log(2)/2] */
/* exp(x) = 2^i * e^f; i = rint (log2(e) * a), f = a - log(2) * i */
/* t = log2(e) * a */
///  const __m256i i = _mm256_cvttpd_epi32(t);       /* i = (int)rint(t) */
// REprintf("Before avx2_cvtpd_epi64 \n");
// R_FlushConsole();
// REprintf("After avx2_cvtpd_epi64 \n");
// R_FlushConsole();
// REprintf("Before _mm256_round_pd \n");
// R_FlushConsole();
// const __m256d x_2 = _mm256_round_pd(t, _MM_FROUND_TO_NEAREST_INT) ; // ((0<<4)| _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC|_MM_FROUND_NO_EXC));
// REprintf("After _mm256_round_pd \n");
// R_FlushConsole();
// REprintf("Before _mm256_fmadd_pd \n");
// R_FlushConsole();
/* a - log(2)_hi * r */
/* f = a - log(2)_hi * r - log(2)_lo * r */
// REprintf("After _mm256_fmadd_pd \n");
// R_FlushConsole();
/* p ~= exp (f), -log(2)/2 <= f <= log(2)/2 */
// REprintf("Before nico_ldexp \n");
// R_FlushConsole();
// ALIGN32  __m256d const res = nico_ldexp_2(p, i);   /* exp(x) = 2^i * p */
// REprintf("After nico_ldexp \n");
// R_FlushConsole();
// _mm256_zeroupper();  // Reset AVX state
// see https://stackoverflow.com/questions/39587752/difference-between-ldexp1-x-and-exp2x
// ALWAYS_INLINE __m256d nico_exp_1_AVX2  VECTORCALL(const __m256d a) {
//   
//         // _mm256_zeroupper();  // Reset AVX state
//         
//         ALIGN32  __m256d const   exp_bound  =   _mm256_set1_pd(708.4);
//         ALIGN32  __m256d const   pos_inf  =    _mm256_set1_pd(INFINITY);
//         ALIGN32  __m256d const   zero  =   _mm256_setzero_pd();
//         
//         ALIGN32  __m256d const res = _mm256_blendv_pd(
//                                       _mm256_blendv_pd(
//                                         _mm256_blendv_pd( 
//                                           pos_inf,
//                                           zero,
//                                           _mm256_cmp_pd(a, zero, _CMP_LT_OQ)
//                                         ),
//                                         _mm256_set1_pd(INFINITY - INFINITY),
//                                         is_not_NaN_mask(a)
//                                       ),  
//                                       nico_exp_1_wo_checks_AVX2(a),
//                                       _mm256_cmp_pd(_mm256_abs_pd(a), exp_bound, _CMP_LT_OQ)
//                                     );
//         
//         // _mm256_zeroupper();  // Reset AVX state
//         return res;
//         
// }
// see https://stackoverflow.com/questions/39587752/difference-between-ldexp1-x-and-exp2x
// Keep exceptional-lane libm calls out of the inlined SIMD hot path.
// 708.4 bounds the exponent-bit shortcut, NOT the domain of finite exp().
///////////////////  fns - log  (+ related e.g. log1m etc) -----------------------------------------------------------------------------------------------------------------
// ALWAYS_INLINE __m256d avx2_cvtepi64_pd VECTORCALL(const __m256i v) {
//   
//       // Extract high and low 64-bit integers
//       ALIGN32 __m128i const low =  _mm256_castsi256_si128(v);
//       ALIGN32 __m128i const high = _mm256_extracti128_si256(v, 1);
//       
//       // Convert to doubles one element at a time
//       ALIGN32 double vals[4];
//       vals[0] = (double)_mm_extract_epi64(low, 0);
//       vals[1] = (double)_mm_extract_epi64(low, 1);
//       vals[2] = (double)_mm_extract_epi64(high, 0);
//       vals[3] = (double)_mm_extract_epi64(high, 1);
//       
//       // Load back into __m256d
//       return _mm256_loadu_pd(vals);
//   
// }
////
// Only for the exponent field `e` in nico_log_1_wo_checks_AVX2 (low 52 bits are zero).
// Returns (double)e exactly; keep the following fmadd by log_c1 as-is so the structure matches AVX-512.
// high dword of each lane
// == e >> 52
// exact
// * 2^52, exact
//// https://stackoverflow.com/a/65537754/9007125 // vectorized version of the answer by njuffa
// ALWAYS_INLINE     __m256d nico_log_1_wo_checks_AVX2 VECTORCALL(const __m256d a ) {
//   
//       ALIGN32  __m256i const log_i1 =   _mm256_set1_epi64x(0x3fe5555555555555);
//       ALIGN32  __m256i const log_i2 =   _mm256_set1_epi64x(0xFFF0000000000000);
//       
//       ALIGN32  __m256d const log_c1 =   _mm256_set1_pd(0.000000000000000222044604925031308085);
//       ALIGN32  __m256d const log_c2 =   _mm256_set1_pd(-0.13031005859375);
//       ALIGN32  __m256d const log_c3 =   _mm256_set1_pd(0.140869140625);
//       ALIGN32  __m256d const log_c4 =   _mm256_set1_pd(-0.121483256222766876221);
//       ALIGN32  __m256d const log_c5 =   _mm256_set1_pd(0.139814853668212890625);
//       ALIGN32  __m256d const log_c6 =   _mm256_set1_pd(-0.166846126317977905273);
//       ALIGN32  __m256d const log_c7 =   _mm256_set1_pd(0.200120344758033752441);
//       ALIGN32  __m256d const log_c8 =   _mm256_set1_pd(-0.249996200203895568848);
//       ALIGN32  __m256d const log_c9 =   _mm256_set1_pd(0.333331972360610961914);
//       ALIGN32  __m256d const log_c10 =  _mm256_set1_pd(-0.5);
//       ALIGN32  __m256d const log_c11 =  _mm256_set1_pd(0.693147182464599609375);
//       
//       
//       ALIGN32  __m256i const aInt = _mm256_castpd_si256(a);
//       ALIGN32  __m256i const e = _mm256_and_epi64( _mm256_sub_epi64(aInt, log_i1),  log_i2); //    e = (__double_as_int (a) - i1 )    &   i2   ;
//       ALIGN32  __m256d const i = _mm256_fmadd_pd ( avx2_cvtepi64_pd(e), log_c1, _mm256_setzero_pd()); // 0x1.0p-52
//       ALIGN32  __m256d const m = _mm256_sub_pd(_mm256_castsi256_pd( _mm256_sub_epi64(aInt, e)),  _mm256_set1_pd(1.0)) ;  //   m = __int_as_double (__double_as_int (a) - e);   //  m = _mm256_sub_pd(m, one) ; // m - 1.0;  /* m in [2/3, 4/3] */
//       ALIGN32  __m256d const s = _mm256_mul_pd(m, m);  // m = _mm256_sub_pd(m,  _mm256_set1_pd(1.0)) ; // m - 1.0;  /* m in [2/3, 4/3] */
//       
//       /* Compute log1p(m) for m in [-1/3, 1/3] */
//       ALIGN32 __m256d   r, t;
//       r =             log_c2;  // -0x1.0ae000p-3
//       t =             log_c3;  //  0x1.208000p-3
//       r = _mm256_fmadd_pd (r, s, log_c4); // -0x1.f198b2p-4
//       t = _mm256_fmadd_pd (t, s, log_c5); //  0x1.1e5740p-3
//       r = _mm256_fmadd_pd (r, s, log_c6); // -0x1.55b36cp-3
//       t = _mm256_fmadd_pd (t, s, log_c7); //  0x1.99d8b2p-3
//       r = _mm256_fmadd_pd (r, s, log_c8); // -0x1.fffe02p-3
//       r = _mm256_fmadd_pd (t, m, r);
//       r = _mm256_fmadd_pd (r, m, log_c9); //  0x1.5554fap-2
//       r = _mm256_fmadd_pd (r, m, log_c10); // -0x1.000000p-1
//       r = _mm256_fmadd_pd (r, s, m);
//       
//       ALIGN32  __m256d const res = _mm256_fmadd_pd(i,  log_c11, r); //  0x1.62e430p-1 // log(2)
//       return res;
//   
// }
//// https://stackoverflow.com/a/65537754/9007125 // vectorized version of the answer by njuffa
// 2^-52
// -0x1.0ae000p-3
//  0x1.208000p-3
// -0x1.f198b2p-4  <-- was -0.121483256...
//  0x1.1e5740p-3
// -0x1.55b36cp-3
//  0x1.99d8b2p-3
// -0x1.fffe02p-3
//  0x1.5554fap-2
//  0x1.62e430p-1
// <-- was the _mm256_and_epi64 macro
// m in [-1/3, 1/3]
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
// Create mask using _mm256_cmp_pd
//// ---- log1p, double precision (nico_log1p_1_double_precision_*; the default nico_log1p_1_* is the faster
////      single- precision-accuracy version above). For x in [sqrt(2)/2 - 1, sqrt(2) - 1) the fdlibm core runs on
////      f = x itself (k = 0): 1 + x is never rounded. Otherwise log(u), u = 1 + x, reduced as in nico_log_1,
////      plus (x - (u - 1)) / u, the rounding error of u to first order (a 12-bit single-precision reciprocal of
////      u is ample for that term). One core evaluation (one division) for any mix of lanes; the reciprocal only
////      when a lane is outside the core range.
// sqrt(2)/2 - 1
// sqrt(2) - 1
// Create mask using _mm256_cmp_pd
// //https://stackoverflow.com/a/65537754/9007125
// Isolate rare subnormal/domain repairs from the normal-positive SIMD path.
// The mantissa/exponent shortcut assumes a positive NORMAL double.
// double-precision log (fdlibm, one division per vector) with the same scalar repair of non-normal / non-positive lanes
// //  compute log(1+x) without losing precision for small values of x.
// //  see: https://www.johndcook.com/cpp_log_one_plus_x.html
// Isolate x <= -1, infinities and NaN (std::log1p) from the SIMD path.
// //  compute log(1-x) without losing precision for small values of x.
// //  compute log(1+x) without losing precision for small values of x. //  see: https://www.johndcook.com/cpp_log_one_plus_x.html
//// ---- log(1 - exp(x)) (log1m_exp), x <= 0: x <  -ln 2:  log1p(-exp(x)) with the default nico_exp_1 and
////      nico_log1p_1 (as the Stan external's log1p(-exp(x)) branch: the same values) x >= -ln 2:  log(-expm1(x))
////      = log(-x) + h(x),  h(x) = log(expm1(x) / x) = x/2 + log(sinh(x/2) / (x/2)) = x/2 + sum_k B_2k x^2k / (2k
////      (2k)!)  (B_2k the Bernoulli numbers), to k = 8 (x^16): truncation below 1e-18 and rounding about 3e-17
////      on [-ln 2, 0); log(-x) by the double-precision log (nico_log_1_double_precision_*): no cancellation as x
////      -> 0, and no scalar log / expm1. Each branch is evaluated only when at least one lane needs it. x = 0
////      gives -inf, x > 0 and NaN give NaN, x = -inf gives 0 (exceptional lanes go through the scalar repairs of
////      nico_exp_1, nico_log_1, nico_log1p_1).
// x >= -ln 2 (NaN: false)
// //https://stackoverflow.com/a/65537754/9007125
// inline      __m256d nico_log_1_AVX2(const __m256d a ) {
//   
//             const __mmask8 is_a_finite_and_gr0_mask =  is_finite_mask(a) &  _mm256_cmp_pd_mask(a, _mm256_setzero_pd(), _CMP_GT_OQ);
//             
//             return  _mm256_mask_blend_pd(is_a_finite_and_gr0_mask,
//                                          _mm256_mask_blend_pd(_mm256_cmp_pd_mask(_mm256_setzero_pd(), a, _CMP_EQ_OQ),     // cond is "if a = 0 "
//                                                               _mm256_sub_pd(_mm256_set1_pd(INFINITY), _mm256_set1_pd(INFINITY)),  // if a =/= 0  (which implies "if a < 0.0" !!!!!)
//                                                               _mm256_set1_pd(-INFINITY)),  // if a = 0 (i.e. cond=TRUE)
//                                                               nico_log_1_wo_checks_AVX2(a));  // if TRUE
//             
// }
// 
// //  compute log(1+x) without losing precision for small values of x.
// //  see: https://www.johndcook.com/cpp_log_one_plus_x.html
// inline     __m256d nico_log1p_1_AVX2(const __m256d x)   {
//   
//   const __mmask8 is_x_le_or_eq_to_m1_mask =     _mm256_cmp_pd_mask(x,  _mm256_set1_pd(-1.0), _CMP_LE_OQ);
//   const __m256d sign_bit = _mm256_set1_pd(-0.0);
//   const __m256d x_abs = _mm256_andnot_pd(sign_bit, x);
//   
//   return   _mm256_mask_blend_pd(is_x_le_or_eq_to_m1_mask, // is x <= -1? (i.e., is x+1 <= 0?)
//                                 _mm256_mask_blend_pd( _mm256_cmp_pd_mask(x_abs,  _mm256_set1_pd(1e-4), _CMP_GT_OQ), // if "is_x_le_or_eq_to_m1_mask" is FALSE (i.e., x+1 > 0)
//                                                      _mm256_mul_pd(_mm256_fmadd_pd( _mm256_set1_pd(-0.5), x,  _mm256_set1_pd(1.0)), x) ,
//                                                      nico_log_1_AVX2( _mm256_add_pd(x,  _mm256_set1_pd(1.0)) ) ), // if x > 1e-4 - evaluate using normal log fn!
//                                  nico_log_1_AVX2( _mm256_add_pd(x,  _mm256_set1_pd(1.0)) ) ) ;  // if FALSE (i.e., x+1  > 0 ) then still pass onto normal log fn as this will handle NaNs etc!
//   
// }
// 
// //  compute log(1-x) without losing precision for small values of x.
// inline     __m256d nico_log1m_1_AVX2(const __m256d x)   {
//   
//   return nico_log1p_1_AVX2( _mm256_sub_pd(_mm256_setzero_pd() , x)) ; // -x);
//   
// }
// 
// //  compute log(1+x) without losing precision for small values of x. //  see: https://www.johndcook.com/cpp_log_one_plus_x.html
// inline      __m256d nico_log1p_exp_1_AVX2(const __m256d x)   {
//   
//   const __m256d neg_x =  _mm256_sub_pd(_mm256_setzero_pd(), x);
//   const __mmask8 is_x_gr_0_mask =      _mm256_cmp_pd_mask(x,  _mm256_setzero_pd(), _CMP_GT_OQ);
//   
//   return       _mm256_mask_blend_pd(is_x_gr_0_mask,
//                                     nico_log1p_1_AVX2(nico_exp_1_AVX2(x)),
//                                     _mm256_add_pd(x, nico_log1p_1_AVX2(nico_exp_1_AVX2(neg_x))));
//   
// }
///////////////////  fns - inv_logit   ----------------------------------------------------------------------------------------------------------------
// ALWAYS_INLINE     __m256d  nico_inv_logit_AVX2 VECTORCALL(const __m256d x )  {
//   
//           ALIGN32  __m256d const res =  _mm256_blendv_pd(  nico_inv_logit_for_x_neg_AVX2(x),
//                                        nico_inv_logit_for_x_pos_AVX2(x), 
//                                        _mm256_cmp_pd(x, _mm256_setzero_pd(), _CMP_GE_OQ));  // x > 0
//           return res;
//   
// }
// ALWAYS_INLINE      __m256d  nico_inv_logit_wo_checks_AVX2 VECTORCALL(const __m256d x )  {
//   
//           ALIGN32  __m256d const res =  _mm256_blendv_pd(    nico_inv_logit_for_x_neg_wo_checks_AVX2(x),
//                                        nico_inv_logit_for_x_pos_wo_checks_AVX2(x), 
//                                        _mm256_cmp_pd(x,  _mm256_setzero_pd(), _CMP_GT_OQ));  // x > 0
//           return res;
//       
// }
///////////////////  fns - log_inv_logit   ----------------------------------------------------------------------------------------------------------------
// inline     __m256d  nico_log_inv_logit_for_x_pos_AVX2 VECTORCALL(const __m256d x )  {
//    
//       ALIGN32 const    __m256d   m_x = _mm256_sub_pd(_mm256_setzero_pd(), x);
//     
//       ALIGN32  __m256d const res =   _mm256_sub_pd(_mm256_setzero_pd(), nico_log1p_exp_1_AVX2(nico_exp_1_AVX2(m_x)));
//       return res;
//     
// }
// -x
// exp(-x)
// log(1 + exp(-x))
// -log(1 + exp(-x))
// x > 0
////////////
// x > 0
/////////////////////  fns - Phi_approx   ----------------------------------------------------------------------------------------------------------------
///////////////////  fns - inv_Phi_approx   ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// logit first
// -0.34176618822627863*m_logit_p;
// now do arc_sinh part
//   now do sinh parth part
// Avoid 1 / p overflowing for valid subnormal probabilities. The usual path below is unchanged.
// log((1-p)/p), stable near 1
// -0.34176618822627863*m_logit_p;
// now do arc_sinh part
//   now do sinh parth part
// need to add citation to this (slight modification from a forum post)
// ALIGN32  __m256d const  inv_Phi_approx_c1 = _mm256_set1_pd(-0.34176618822627863);
// now do arc_sinh part
//   now do sinh parth part
// ALIGN32  __m256d const  inv_Phi_approx_c1 = _mm256_set1_pd(-0.34176618822627863);
// now do arc_sinh part
//   now do sinh parth part
///////////////////  fns - log_Phi_approx   ----------------------------------------------------------------------------------------------------------------------------------------
//////////////////// ------------- tanh  --------------------------------------------------------------------------------------------------------------------------------------
// from: https://stackoverflow.com/questions/57870896/writing-a-portable-sse-avx-version-of-stdcopysign
// Extract the signed bit from srcSign
// Extract the number without sign of srcValue (abs(srcValue))
// Merge signed bit with number and return
///////////////////  fns - Phi functions   ------------------------------------------------------------------------------------------------------------------------------------
// inline    __m256d nico_Phi_wo_checks_AVX2(const __m256d x) {
//   
//     const __m256d sqrt_2_recip = _mm256_set1_pd(0.707106781186547461715);
//   
//     return _mm256_mul_pd(one_half, _mm256_add_pd(one, nico_erf_wo_checks_AVX2( _mm256_mul_pd(x, sqrt_2_recip)) ) ) ;
//     
// }
//  
//  
//  
// 
// // AVX2 version with consistent mask logic (i.e., consistent with AVX-256 logic)
// inline __m256d nico_Phi_AVX2(const __m256d x) {
//  
//   
//   const __m256d is_x_gr_lower =  _mm256_cmp_pd(x, Phi_lower_bound, _CMP_GT_OQ); // i.e. if x > -37.5  
//   const __m256d is_x_lt_upper =  _mm256_cmp_pd(x, Phi_upper_bound, _CMP_LT_OQ); // i.e. if x < 8.25
//   const __m256d is_x_in_range =  _mm256_and_pd( is_x_gr_lower ,  is_x_lt_upper);
//   
//    
//    return    _mm256_blendv_pd(     
//                              _mm256_blendv_pd(zero,   // if x NOT in range and x NOT < -37.5 i.e. x must be <-37.5
//                                               one,  // if x NOT in range and x > 8.25
//                                               _mm256_cmp_pd(x, Phi_lower_bound, _CMP_LT_OQ)),  // if x NOT in range and x < -37.5
//                             nico_Phi_wo_checks_AVX2(x), // i.e. if x in range ?
//                             is_x_in_range); 
//   
// }
// 
// 
// 
//
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
// equality must not fall through to 1
// true where x < 8.25 (in range)
// Calculate the main Phi function for in-range values
// if x NOT in range and x NOT < -37.5 i.e. x must be > 8.25
// if x NOT in range and x < -37.5
// if x is in range
// Clamp results between 0 and 1
// Preserve NaN input rather than turning it into a valid probability through min/max.
///////////////////  fns - inv_Phi functions   --------------------------------------------------------------------------------------------------------------------------------
//// vectorised, AVX-256 version of Stan fn provided by Sean Pinkney:  https://github.com/stan-dev/math/issues/2555 
///  CASE 2(a): if abs(q) > 0.425  AND   if r <= 5.0 
///  CASE 2(a): if abs(q) > 0.425  AND   if r > 5.0 
// ALWAYS_INLINE __m256d nico_inv_Phi_wo_checks_case_2_AVX2 VECTORCALL( const __m256d p,
//                                                                      const __m256d q) {  /// CASE 2 (i.e. one of case 2(a) or 2(b) depending on value of r)
//   
//   
//       ALIGN32  __m256d const zero =  _mm256_setzero_pd();
//       ALIGN32  __m256d const one =   _mm256_set1_pd(1.0);
//       ALIGN32  __m256d const five =  _mm256_set1_pd(5.0);
//       
//       ALIGN32  __m256d const is_q_gr_0 = _mm256_cmp_pd(q, zero, _CMP_GT_OQ);
//        
//       ////   compute r 
//       ALIGN32 __m256d r_if_q_gr_0 =  _mm256_sub_pd(one, p);
//       ALIGN32 __m256d r_if_q_lt_0 = p;
//       ALIGN32 __m256d r = _mm256_blendv_pd(r_if_q_gr_0, r_if_q_lt_0, is_q_gr_0);
//       
//       r = _mm256_sqrt_pd(_mm256_sub_pd(zero, nico_log_1_AVX2(r)));
//       
//       /// then call either case 2(a) fn (if r <= 5.0) or case 2(b) fn (if r > 5.0)  
//       ALIGN32  __m256d const is_r_gr_5 = _mm256_cmp_pd(r, five, _CMP_GT_OQ);
//       ALIGN32 __m256d val =  _mm256_blendv_pd(     nico_inv_Phi_wo_checks_case_2b_AVX2(p, r),  /// if r > 5.0
//                                            nico_inv_Phi_wo_checks_case_2a_AVX2(p, r),
//                                            is_r_gr_5);
//       
//       // Flip the sign if q is negative
//       ALIGN32  __m256d const is_q_lt_zero = _mm256_cmp_pd(q, zero, _CMP_LT_OQ); // _mm256_cmplt_pd_mask(q, _mm256_set1_pd(0.0));
//       
//       val = _mm256_blendv_pd(    _mm256_sub_pd(zero, val),   //// if q < 0
//                                  val,
//                                  is_q_lt_zero); //// if   q => 0 
//       return val;
// 
// }
// q>0 -> 1-p, else p
// r <= 5
// No need to evaluate the extreme-tail polynomial unless a lane will actually use it.
// q<0 -> -val
///  CASE 1: if abs(q) <= 0.425
// ALWAYS_INLINE __m256d nico_inv_Phi_wo_checks_AVX2 VECTORCALL(const __m256d p) {
//   
//       ALIGN32  __m256d const one_half =  _mm256_set1_pd(0.50);
//       ALIGN32  __m256d const point_425 = _mm256_set1_pd(0.425);
//       
//       ALIGN32  __m256d const q = _mm256_sub_pd(p, one_half);
//       ALIGN32  __m256d const is_q_le_threshold = _mm256_cmp_pd(_mm256_abs_pd(q), point_425, _CMP_LE_OQ);
//       
//      
//       ALIGN32 __m256d res;
//       
//       if (_mm256_movemask_pd(is_q_le_threshold) == 0xF) {  // 0xF is binary 1111 (4 bits)  ///  if (_mm256_kortestc(is_q_le_threshold, is_q_le_threshold)) { 
//         
//         res = nico_inv_Phi_wo_checks_case_2_AVX2(p, q);
//         
//       } else { 
//         
//         res = nico_inv_Phi_wo_checks_case_1_AVX2(p, q);
//         
//       }
//       
//       return res;
// 
// }
// The tail result would be discarded in every lane. Avoid its log and two rational polynomials.
// central lanes -> v1
// =============================================================================================
//  AVX2   (identical logic; blendv(a, b, m) == mask_blend(m, a, b), so argument order is unchanged)
// =============================================================================================
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
// ALWAYS_INLINE __m256d nico_log_Phi_AVX2 VECTORCALL(const __m256d x) {
//       const __m256d thresh   = _mm256_set1_pd(NORMAL_TAIL_THRESH);
//       const __m256d ax       = _mm256_abs_pd(x);
//       const __m256d R        = nico_mills_ratio_tail_AVX2(_mm256_max_pd(ax, thresh));
//       const __m256d log_tail = _mm256_add_pd(_mm256_fnmadd_pd(_mm256_set1_pd(0.5), _mm256_mul_pd(x, x), _mm256_set1_pd(NEG_HALF_LOG_2PI)),
//                                              nico_log_1_AVX2(R));
//       const __m256d log_body = nico_log_1_AVX2(nico_Phi_AVX2(x));
//       const __m256d is_tail  = _mm256_cmp_pd(x, _mm256_sub_pd(_mm256_setzero_pd(), thresh), _CMP_LT_OQ);
//       return _mm256_blendv_pd(log_body, log_tail, is_tail);
// }
// Tail (x < -2.5): -x^2/2 - 0.5 log(2 pi) + log R(-x), R by the degree-8 rational above (body lanes evaluate R(2.5),
// then discarded). Lane-mask tests: the tail is evaluated only when at least one lane is in the tail, and the body
// (log of nico_Phi, unchanged) only when at least one lane is not.
// x < -2.5
// no tail lane: body only
// every lane in the tail
//// ---- previous version (tail and body evaluated for every vector), kept for reference:
// ALWAYS_INLINE __m256d nico_dlog_Phi_dx_AVX2 VECTORCALL(const __m256d x) {
//       const __m256d thresh  = _mm256_set1_pd(NORMAL_TAIL_THRESH);
//       const __m256d ax      = _mm256_abs_pd(x);
//       const __m256d R       = nico_mills_ratio_tail_AVX2(_mm256_max_pd(ax, thresh));
//       const __m256d d_tail  = _mm256_div_pd(_mm256_set1_pd(1.0), R);
//       const __m256d phi     = _mm256_mul_pd(_mm256_set1_pd(INV_SQRT_2PI),
//                                             nico_exp_1_AVX2(_mm256_mul_pd(_mm256_set1_pd(-0.5), _mm256_mul_pd(x, x))));
//       const __m256d d_body  = _mm256_div_pd(phi, nico_Phi_AVX2(x));
//       const __m256d is_tail = _mm256_cmp_pd(x, _mm256_sub_pd(_mm256_setzero_pd(), thresh), _CMP_LT_OQ);
//       return _mm256_blendv_pd(d_body, d_tail, is_tail);
// }
// Tail: 1/R(-x) = z (1 - D/P) (the rational above). Lane-mask tests as in nico_log_Phi_AVX2.
// x < -2.5
// no tail lane: body only
// every lane in the tail

#include <NicoSIMD/math/fast_and_approx_AVX2_fns.hpp>






















