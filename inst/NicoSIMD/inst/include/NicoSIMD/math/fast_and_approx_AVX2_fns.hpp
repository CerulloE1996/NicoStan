
#pragma once 


#ifndef FAST_AND_APPROX_AVX2_FNS_HPP
#define FAST_AND_APPROX_AVX2_FNS_HPP


#include "../nico_simd_intrinsics.hpp"   //// <immintrin.h> on x86-64; SIMDe (NEON) on ARM
#include <cmath>
#include "nico_exp_over_the_full_double_range.hpp"
  
  
// #include <R_ext/Print.h>  // For REprintf
// #include <R_ext/Utils.h>  // For R_FlushConsole
  
  
//// #if defined(__AVX2__) && !(defined(__AVX512VL__) && defined(__AVX512F__) && defined(__AVX512DQ__)) // use AVX2 if AVX-512 not available 
// compiled on every AVX2-capable build, including AVX-512 ones - the dispatch layer needs both kernel sets
#if defined(NICO_SIMD_AVX2_KERNELS)   //// AVX2 natively, or NEON through SIMDe

namespace nico::math {

//// -------------------------------------------------------------------------------------------------------------------------------------------------------------


// Simple test function that just multiplies vector by 2
inline __m256d  test_simple_AVX2 VECTORCALL(const __m256d x) {

       ALIGN32  __m256d const x_aligned = x;
   
       // // _mm256_zeroupper();  // Reset AVX state
       ALIGN32  __m256d const two = _mm256_set1_pd(2.0);
       ALIGN32  __m256d const res = _mm256_mul_pd(x_aligned, two);
       //// _mm256_zeroupper();  // Reset AVX state
       return res; 
   
}




//// -------------------------------------------------------------------------------------------------------------------------------------------------------------
ALWAYS_INLINE __m256d   _mm256_abs_pd VECTORCALL(const __m256d x) {
 
       // _mm256_zeroupper();  // Reset AVX state
       
       ALIGN32  __m256d const x_aligned = x;
       
       ALIGN32  __m256d const sign_bit = _mm256_set1_pd(-0.0);
       ALIGN32  __m256d const x_abs = _mm256_andnot_pd(sign_bit, x_aligned);
       
       // _mm256_zeroupper();  // Reset AVX state
       
       return x_abs;
 
}




// is_finite_mask and is_not_NaN_mask for AVX2
ALWAYS_INLINE __m256d  is_finite_mask VECTORCALL(const __m256d x) {
  
       // _mm256_zeroupper();  // Reset AVX state
  
       ALIGN32  __m256d const x_aligned = x;
       
       ALIGN32  __m256d const INF = _mm256_set1_pd(std::numeric_limits<double>::infinity());
       
       ALIGN32  __m256d const abs_x = _mm256_abs_pd(x_aligned);
       
       ALIGN32  __m256d const res = _mm256_cmp_pd(abs_x, INF, _CMP_NEQ_OQ);
       
       return res; 
   
}




ALWAYS_INLINE __m256d is_not_NaN_mask VECTORCALL(const __m256d x) {
  
       // _mm256_zeroupper();  // Reset AVX state
  
       ALIGN32 __m256d const x_aligned = x;
  
       ALIGN32 __m256d const res = _mm256_cmp_pd(x_aligned, x_aligned, _CMP_EQ_OQ);
       // _mm256_zeroupper();  // Reset AVX state
       
       return res;
 
}




// bookmark (this doesnt seem to be available on g++ at all at time of writing even with all AVX256 flags!)
//#if !defined(__AVX256VL__)
#define _mm256_and_epi64(a, b) _mm256_castpd_si256(_mm256_and_pd(_mm256_castsi256_pd(a), _mm256_castsi256_pd(b))) // this uses AVX2 instructions so moree widely supported




///////////////////  fns - exp   -----------------------------------------------------------------------------------------------------------------------------
ALWAYS_INLINE __m256d   nico_ldexp VECTORCALL(  const __m256d AVX_a,
                                                const __m256i AVX_i) {
  
       // _mm256_zeroupper();  // Reset AVX state
    
       ALIGN32  __m256d const AVX_a_aligned = AVX_a;
       ALIGN32  __m256i const AVX_i_aligned = AVX_i;
    
       ALIGN32 const uint64_t shift_val_52 = 52;
       ALIGN32  __m256i const shifted = _mm256_slli_epi64(AVX_i_aligned, shift_val_52);
       ALIGN32  __m256i const a_bits = _mm256_castpd_si256(AVX_a_aligned);
       ALIGN32  __m256i const result = _mm256_add_epi64(shifted, a_bits);
      
       ALIGN32  __m256d const res = _mm256_castsi256_pd(result);
       // _mm256_zeroupper();  // Reset AVX state
       return res;
  
}




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
ALWAYS_INLINE __m256i avx2_cvtpd_epi64 VECTORCALL(const __m256d x) {

        ALIGN32  __m256d const x_aligned = x;
    
        ALIGN32 __m256d const magic = _mm256_set1_pd(6755399441055744.0);   // 1.5 * 2^52
        ALIGN32 __m256d const y     = _mm256_add_pd(x_aligned, magic);
        return _mm256_sub_epi64(_mm256_castpd_si256(y), _mm256_castpd_si256(magic));
      
}
 
 
// Adapted from: https://stackoverflow.com/questions/48863719/fastest-implementation-of-exponential-function-using-avx
// added   (optional) extra degree(s) for poly approx (oroginal float fn had 4 degrees) - using "minimaxApprox" R package to find coefficient terms
// R code:    minimaxApprox::minimaxApprox(fn = exp, lower = -0.346573590279972643113, upper = 0.346573590279972643113, degree = 5, basis ="Chebyshev")
ALWAYS_INLINE __m256d nico_exp_1_wo_checks_AVX2  VECTORCALL(const __m256d x)  { 
  
        // REprintf("Entering nico_exp_1_wo_checks_AVX2 \n");
        // R_FlushConsole();
      
        // _mm256_zeroupper();  // Reset AVX state
      
        // REprintf("Before x_aligned \n");
        // R_FlushConsole();
        ALIGN32  __m256d const x_aligned = x;
        // REprintf("After x_aligned \n");
        // R_FlushConsole();
        
        ALIGN32  __m256d const exp_l2e = _mm256_set1_pd (1.442695040888963387); /* log2(e) */
        ALIGN32  __m256d const exp_l2h = _mm256_set1_pd (-0.693145751953125); /* -log(2)_hi, exactly representable */
        ALIGN32  __m256d const exp_l2l = _mm256_set1_pd (-1.42860682030941723212e-6); /* -log(2)_lo */
        
        // /* coefficients for core approximation to exp() in [-log(2)/2, log(2)/2] */
        ALIGN32  __m256d const exp_c0 =     _mm256_set1_pd(0.00000276479776161191821278);
        ALIGN32  __m256d const exp_c1 =     _mm256_set1_pd(0.0000248844480527491290235);
        ALIGN32  __m256d const exp_c2 =     _mm256_set1_pd(0.000198411488032534342194);
        ALIGN32  __m256d const exp_c3 =     _mm256_set1_pd(0.00138888017711994078175);
        ALIGN32  __m256d const exp_c4 =     _mm256_set1_pd(0.00833333340524595143906);
        ALIGN32  __m256d const exp_c5 =     _mm256_set1_pd(0.0416666670404215802592);
        ALIGN32  __m256d const exp_c6 =     _mm256_set1_pd(0.166666666664891632843);
        ALIGN32  __m256d const exp_c7 =     _mm256_set1_pd(0.499999999994389376923);
        ALIGN32  __m256d const exp_c8 =     _mm256_set1_pd(1.00000000000001221245);
        ALIGN32  __m256d const exp_c9 =     _mm256_set1_pd(1.00000000000001332268);
        
        ALIGN32  __m256d const input  = x_aligned;
        
        /* exp(x) = 2^i * e^f; i = rint (log2(e) * a), f = a - log(2) * i */
        ALIGN32  __m256d const t = _mm256_mul_pd(x_aligned, exp_l2e);      /* t = log2(e) * a */
        ///  const __m256i i = _mm256_cvttpd_epi32(t);       /* i = (int)rint(t) */
        // REprintf("Before avx2_cvtpd_epi64 \n");
        // R_FlushConsole();
        ALIGN32  __m256i const i = avx2_cvtpd_epi64(t);  
        // REprintf("After avx2_cvtpd_epi64 \n");
        // R_FlushConsole();
        // REprintf("Before _mm256_round_pd \n");
        // R_FlushConsole();
        // const __m256d x_2 = _mm256_round_pd(t, _MM_FROUND_TO_NEAREST_INT) ; // ((0<<4)| _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC|_MM_FROUND_NO_EXC));
        ALIGN32  __m256d const x_2 = _mm256_round_pd(t, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        // REprintf("After _mm256_round_pd \n");
        // R_FlushConsole();
        // REprintf("Before _mm256_fmadd_pd \n");
        // R_FlushConsole();
        ALIGN32  __m256d const f0 = _mm256_fmadd_pd(x_2, exp_l2h, input);
        ALIGN32  __m256d const f =  _mm256_fmadd_pd(x_2, exp_l2l, f0);  /* a - log(2)_hi * r */    /* f = a - log(2)_hi * r - log(2)_lo * r */
        // REprintf("After _mm256_fmadd_pd \n");
        // R_FlushConsole();
        
        /* p ~= exp (f), -log(2)/2 <= f <= log(2)/2 */
        ALIGN32 __m256d p = exp_c0;
        p = _mm256_fmadd_pd(p, f, exp_c1);
        p = _mm256_fmadd_pd(p, f, exp_c2);
        p = _mm256_fmadd_pd(p, f, exp_c3);
        p = _mm256_fmadd_pd(p, f, exp_c4);
        p = _mm256_fmadd_pd(p, f, exp_c5);
        p = _mm256_fmadd_pd(p, f, exp_c6);
        p = _mm256_fmadd_pd(p, f, exp_c7);
        p = _mm256_fmadd_pd(p, f, exp_c8);
        p = _mm256_fmadd_pd(p, f, exp_c9);
        
        // REprintf("Before nico_ldexp \n");
        // R_FlushConsole();
        // ALIGN32  __m256d const res = nico_ldexp_2(p, i);   /* exp(x) = 2^i * p */
        ALIGN32 __m256d const res = nico_ldexp(p, i);
        // REprintf("After nico_ldexp \n");
        // R_FlushConsole();
        // _mm256_zeroupper();  // Reset AVX state
        return  res;  
  
}




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
__attribute__((noinline, cold)) inline __m256d nico_exp_exceptional_lanes_AVX2(
        const __m256d input, const __m256d result, const unsigned ordinary_mask) {

    const __m256i mask_bits = _mm256_set_epi64x(
        ordinary_mask & 8u ? -1 : 0, ordinary_mask & 4u ? -1 : 0,
        ordinary_mask & 2u ? -1 : 0, ordinary_mask & 1u ? -1 : 0);
    return _mm256_blendv_pd(nico_exp_over_the_full_double_range_AVX2(input), result,
                            _mm256_castsi256_pd(mask_bits));

}

ALWAYS_INLINE __m256d nico_exp_1_AVX2 VECTORCALL(const __m256d a) {

      ALIGN32  __m256d const a_aligned = a;

      // 708.4 bounds the exponent-bit shortcut, NOT the domain of finite exp().
      ALIGN32  __m256d const ordinary_lanes = _mm256_cmp_pd(_mm256_abs_pd(a_aligned), _mm256_set1_pd(708.4),
              _CMP_LT_OQ);
      ALIGN32  __m256d const result = nico_exp_1_wo_checks_AVX2(a_aligned);
      const __m256d normal_result = _mm256_cmp_pd(result,
          _mm256_set1_pd(std::numeric_limits<double>::min()), _CMP_GE_OQ);
      const int ordinary_mask = _mm256_movemask_pd(_mm256_and_pd(ordinary_lanes, normal_result));
      if (__builtin_expect(ordinary_mask != 0xf, 0)) return nico_exp_exceptional_lanes_AVX2(a_aligned, result,
              ordinary_mask);
      return result;

}
 
 











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
ALWAYS_INLINE __m256d avx2_cvtepi64_pd VECTORCALL(const __m256i e) {

      ALIGN32  __m256i const e_aligned = e;
  
      // high dword of each lane
      ALIGN32 __m256i const hi = _mm256_permutevar8x32_epi32(e_aligned, _mm256_setr_epi32(1, 3, 5, 7, 1, 3, 5,
              7));
      ALIGN32 __m128i const E  = _mm_srai_epi32(_mm256_castsi256_si128(hi),
                                                20); // == e >> 52
      ALIGN32 __m256d const Ed = _mm256_cvtepi32_pd(E);                                                     // exact
      return _mm256_mul_pd(Ed,
                           _mm256_set1_pd(4503599627370496.0)); // * 2^52, exact
      
}




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
ALWAYS_INLINE __m256d nico_log_1_wo_checks_AVX2 VECTORCALL(const __m256d a) {

      ALIGN32  __m256d const a_aligned = a;
    
      ALIGN32 __m256i const log_i1  = _mm256_set1_epi64x(0x3fe5555555555555);
      ALIGN32 __m256i const log_i2  = _mm256_set1_epi64x(0xFFF0000000000000);
    
      ALIGN32 __m256d const log_c1  = _mm256_set1_pd(0.000000000000000222044604925031308085); // 2^-52
      ALIGN32 __m256d const log_c2  = _mm256_set1_pd(-0.13031005859375);                      // -0x1.0ae000p-3
      ALIGN32 __m256d const log_c3  = _mm256_set1_pd(0.140869140625);                         //  0x1.208000p-3
      ALIGN32 __m256d const log_c4  = _mm256_set1_pd(-0.121483512222766876221);               // -0x1.f198b2p-4  <-- was -0.121483256...
      ALIGN32 __m256d const log_c5  = _mm256_set1_pd(0.139814853668212890625);                //  0x1.1e5740p-3
      ALIGN32 __m256d const log_c6  = _mm256_set1_pd(-0.166846126317977905273);               // -0x1.55b36cp-3
      ALIGN32 __m256d const log_c7  = _mm256_set1_pd(0.200120344758033752441);                //  0x1.99d8b2p-3
      ALIGN32 __m256d const log_c8  = _mm256_set1_pd(-0.249996200203895568848);               // -0x1.fffe02p-3
      ALIGN32 __m256d const log_c9  = _mm256_set1_pd(0.333331972360610961914);                //  0x1.5554fap-2
      ALIGN32 __m256d const log_c10 = _mm256_set1_pd(-0.5);
      ALIGN32 __m256d const log_c11 = _mm256_set1_pd(0.693147182464599609375);                //  0x1.62e430p-1
    
      ALIGN32 __m256i const aInt = _mm256_castpd_si256(a_aligned);
      ALIGN32 __m256i const e    = _mm256_and_si256(_mm256_sub_epi64(aInt, log_i1),
                                                    log_i2); // <-- was the _mm256_and_epi64 macro
      ALIGN32 __m256d const i    = _mm256_fmadd_pd(avx2_cvtepi64_pd(e), log_c1, _mm256_setzero_pd());
      ALIGN32 __m256d const m    = _mm256_sub_pd(_mm256_castsi256_pd(_mm256_sub_epi64(aInt, e)),
                                                 _mm256_set1_pd(1.0)); // m in [-1/3, 1/3]
      ALIGN32 __m256d const s    = _mm256_mul_pd(m, m);
    
      ALIGN32 __m256d r, t;
      r = log_c2;
      t = log_c3;
      r = _mm256_fmadd_pd(r, s, log_c4);
      t = _mm256_fmadd_pd(t, s, log_c5);
      r = _mm256_fmadd_pd(r, s, log_c6);
      t = _mm256_fmadd_pd(t, s, log_c7);
      r = _mm256_fmadd_pd(r, s, log_c8);
      r = _mm256_fmadd_pd(t, m, r);
      r = _mm256_fmadd_pd(r, m, log_c9);
      r = _mm256_fmadd_pd(r, m, log_c10);
      r = _mm256_fmadd_pd(r, s, m);
    
      return _mm256_fmadd_pd(i, log_c11, r);
  
}


//// ---- log, double precision (nico_log_1_double_precision_*): the fdlibm (Sun Microsystems, e_log.c) algorithm,
////      vectorised. a = 2^k (1 + f) with 1 + f in [sqrt(2)/2, sqrt(2)) (the exponent trick below, with sqrt(2)/2
////      as the split point); s = f / (2 + f), |s| <= 0.1716; R = s^2 (Lg1 + s^4 (Lg3 + s^4 (Lg5 + s^4 Lg7))) +
////      s^4 (Lg2 + s^4 (Lg4 + s^4 Lg6)); log(a) = k ln2_hi - ((hfsq - (s (hfsq + R) + k ln2_lo)) - f), hfsq =
////      f^2 / 2 (ln 2 split into hi and lo parts). Below 1 ulp (measured 0.83 ulp), but one division per vector:
////      slower than nico_log_1_wo_checks_* (njuffa's single-precision polynomial, about 1.5e-8 relative error),
////      which therefore stays the default log. ---- reduction for a positive normal double: a = 2^k (1 + f) with
////      1 + f in [sqrt(2)/2, sqrt(2)) (k exact, as a double)
ALWAYS_INLINE void nico_log_reduce_AVX2(const __m256d a, __m256d& f, __m256d& k) {

      ALIGN32  __m256d const a_aligned = a;

      ALIGN32 __m256i const sqrt_half_bits = _mm256_set1_epi64x(0x3fe6a09e667f3bcd);     // bits of sqrt(2)/2
      ALIGN32 __m256i const exponent_mask  = _mm256_set1_epi64x(0xFFF0000000000000);
      ALIGN32 __m256i const a_bits         = _mm256_castpd_si256(a_aligned);
      // e = k << 52; k = e * 2^-52 (exact); f exact
      ALIGN32 __m256i const e = _mm256_and_si256(_mm256_sub_epi64(a_bits, sqrt_half_bits), exponent_mask);
      k = _mm256_mul_pd(avx2_cvtepi64_pd(e), _mm256_set1_pd(0.000000000000000222044604925031308085));
      f = _mm256_sub_pd(_mm256_castsi256_pd(_mm256_sub_epi64(a_bits, e)), _mm256_set1_pd(1.0));

}

//// ---- the fdlibm core: k ln 2 + log1p(f) for 1 + f in [sqrt(2)/2, sqrt(2)) (shared by nico_log_1_* and
////      nico_log1p_1_*)
ALWAYS_INLINE __m256d nico_log_fdlibm_core_AVX2(const __m256d f, const __m256d k) {

      ALIGN32  __m256d const f_aligned = f;
      ALIGN32  __m256d const k_aligned = k;

      ALIGN32 __m256d const ln2_hi = _mm256_set1_pd(6.93147180369123816490e-01);   // 0x3fe62e42fee00000
      ALIGN32 __m256d const ln2_lo = _mm256_set1_pd(1.90821492927058770002e-10);   // 0x3dea39ef35793c76
      ALIGN32 __m256d const Lg1    = _mm256_set1_pd(6.666666666666735130e-01);     // 0x3FE5555555555593
      ALIGN32 __m256d const Lg2    = _mm256_set1_pd(3.999999999940941908e-01);     // 0x3FD999999997FA04
      ALIGN32 __m256d const Lg3    = _mm256_set1_pd(2.857142874366239149e-01);     // 0x3FD2492494229359
      ALIGN32 __m256d const Lg4    = _mm256_set1_pd(2.222219843214978396e-01);     // 0x3FCC71C51D8E78AF
      ALIGN32 __m256d const Lg5    = _mm256_set1_pd(1.818357216161805012e-01);     // 0x3FC7466496CB03DE
      ALIGN32 __m256d const Lg6    = _mm256_set1_pd(1.531383769920937332e-01);     // 0x3FC39A09D078C69F
      ALIGN32 __m256d const Lg7    = _mm256_set1_pd(1.479819860511658591e-01);     // 0x3FC2F112DF3E5244

      ALIGN32 __m256d const s      = _mm256_div_pd(f_aligned, _mm256_add_pd(_mm256_set1_pd(2.0), f_aligned));
      ALIGN32 __m256d const s2     = _mm256_mul_pd(s, s);
      ALIGN32 __m256d const s4     = _mm256_mul_pd(s2, s2);
      ALIGN32 __m256d const t_odd  = _mm256_mul_pd(s2, _mm256_fmadd_pd(s4, _mm256_fmadd_pd(s4,
                                                       _mm256_fmadd_pd(s4, Lg7, Lg5), Lg3), Lg1));
      ALIGN32 __m256d const t_even = _mm256_mul_pd(s4, _mm256_fmadd_pd(s4, _mm256_fmadd_pd(s4, Lg6, Lg4), Lg2));
      ALIGN32 __m256d const R      = _mm256_add_pd(t_odd, t_even);
      ALIGN32 __m256d const hfsq   = _mm256_mul_pd(_mm256_set1_pd(0.5), _mm256_mul_pd(f_aligned, f_aligned));

      ALIGN32 __m256d const inner  = _mm256_fmadd_pd(s, _mm256_add_pd(hfsq, R), _mm256_mul_pd(k_aligned, ln2_lo));
      return _mm256_sub_pd(_mm256_mul_pd(k_aligned, ln2_hi), _mm256_sub_pd(_mm256_sub_pd(hfsq, inner),
              f_aligned));

}

ALWAYS_INLINE __m256d nico_log_1_double_precision_wo_checks_AVX2 VECTORCALL(const __m256d a) {

      ALIGN32  __m256d const a_aligned = a;

      __m256d f, k;
      nico_log_reduce_AVX2(a_aligned, f, k);
      return nico_log_fdlibm_core_AVX2(f, k);

}



ALWAYS_INLINE __m256d nico_log1p_1_wo_checks_AVX2 VECTORCALL(const __m256d x) {

      ALIGN32  __m256d const x_aligned = x;
    
      ALIGN32  __m256d const small =  _mm256_set1_pd(1e-4);
      ALIGN32  __m256d const minus_one_half  =  _mm256_set1_pd(-0.50);
    
      // Create mask using _mm256_cmp_pd
      ALIGN32  __m256d const mask = _mm256_cmp_pd(_mm256_abs_pd(x_aligned), small, _CMP_GT_OQ);
      
      ALIGN32  __m256d const res =  _mm256_blendv_pd(
                                    _mm256_mul_pd(_mm256_fmadd_pd(minus_one_half, x_aligned,  _mm256_set1_pd(
                                            1.0)), x_aligned),
                                    nico_log_1_wo_checks_AVX2(_mm256_add_pd(x_aligned,  _mm256_set1_pd(1.0))),
                                    mask
                                  );
      
      return res;
  
}




//// ---- log1p, double precision (nico_log1p_1_double_precision_*; the default nico_log1p_1_* is the faster
////      single- precision-accuracy version above). For x in [sqrt(2)/2 - 1, sqrt(2) - 1) the fdlibm core runs on
////      f = x itself (k = 0): 1 + x is never rounded. Otherwise log(u), u = 1 + x, reduced as in nico_log_1,
////      plus (x - (u - 1)) / u, the rounding error of u to first order (a 12-bit single-precision reciprocal of
////      u is ample for that term). One core evaluation (one division) for any mix of lanes; the reciprocal only
////      when a lane is outside the core range.
static constexpr double FAST_LOG1P_CORE_LOWER_AVX2 = -0.29289321881345247560;   // sqrt(2)/2 - 1
static constexpr double FAST_LOG1P_CORE_UPPER_AVX2 =  0.41421356237309504880;   // sqrt(2) - 1
ALWAYS_INLINE __m256d nico_log1p_1_double_precision_wo_checks_AVX2 VECTORCALL(const __m256d x) {

      ALIGN32  __m256d const x_aligned = x;

      ALIGN32 __m256d const in_core = _mm256_and_pd(
                          _mm256_cmp_pd(x_aligned, _mm256_set1_pd(FAST_LOG1P_CORE_LOWER_AVX2), _CMP_GE_OQ),
                          _mm256_cmp_pd(x_aligned, _mm256_set1_pd(FAST_LOG1P_CORE_UPPER_AVX2), _CMP_LT_OQ));
      if (_mm256_movemask_pd(in_core) == 0xf) return nico_log_fdlibm_core_AVX2(x_aligned, _mm256_setzero_pd());
      ALIGN32 __m256d const u = _mm256_add_pd(_mm256_set1_pd(1.0), x_aligned);
      __m256d f_u, k_u;
      nico_log_reduce_AVX2(u, f_u, k_u);
      ALIGN32 __m256d const log_value  = nico_log_fdlibm_core_AVX2(
                                             _mm256_blendv_pd(f_u, x_aligned, in_core),
                                             _mm256_blendv_pd(k_u, _mm256_setzero_pd(), in_core));
      ALIGN32 __m256d const reciprocal = _mm256_cvtps_pd(_mm_rcp_ps(_mm256_cvtpd_ps(u)));
      ALIGN32 __m256d const correction = _mm256_mul_pd(_mm256_sub_pd(x_aligned, _mm256_sub_pd(u, _mm256_set1_pd(
              1.0))),
                                                       reciprocal);
      return _mm256_blendv_pd(_mm256_add_pd(log_value, correction), log_value, in_core);

}




ALWAYS_INLINE __m256d nico_log1m_1_wo_checks_AVX2 VECTORCALL(const __m256d x) {

     ALIGN32  __m256d const x_aligned = x;
  
     ALIGN32  __m256d const res =  nico_log1p_1_wo_checks_AVX2(_mm256_sub_pd(_mm256_setzero_pd(), x_aligned));
     return res;
  
} 




// ALWAYS_INLINE __m256d nico_log1p_exp_1_wo_checks_AVX2 VECTORCALL(const __m256d x) {
//
//        ALIGN32  __m256d const neg_x = _mm256_sub_pd(_mm256_setzero_pd(), x);
//
//        ALIGN32  __m256d const mask = _mm256_cmp_pd(x, _mm256_setzero_pd(), _CMP_GT_OQ);     // Create mask using _mm256_cmp_pd
//
//        ALIGN32  __m256d const res =  _mm256_blendv_pd(
//                                     nico_log1p_1_wo_checks_AVX2(nico_exp_1_wo_checks_AVX2(x)),
//                                     _mm256_add_pd(x, nico_log1p_1_wo_checks_AVX2(nico_exp_1_wo_checks_AVX2(neg_x))),
//                                     mask
//                                   );
//
//       return res;
//
// }
// (one exp and one log1p / one division per lane, of the argument which the lane's branch uses, instead of
//  both branches in every lane: the same operations, lane by lane, and so the same values, at half the cost)
ALWAYS_INLINE __m256d nico_log1p_exp_1_wo_checks_AVX2 VECTORCALL(const __m256d x) {

       ALIGN32  __m256d const x_aligned = x;
       ALIGN32  __m256d const is_x_gr_0 = _mm256_cmp_pd(x_aligned, _mm256_setzero_pd(), _CMP_GT_OQ);
       ALIGN32  __m256d const argument = _mm256_blendv_pd(x_aligned, _mm256_sub_pd(_mm256_setzero_pd(),
               x_aligned), is_x_gr_0);
       ALIGN32  __m256d const log1p_exp_of_argument =
           nico_log1p_1_wo_checks_AVX2(nico_exp_1_wo_checks_AVX2(argument));
       return _mm256_blendv_pd(log1p_exp_of_argument,                                             // if x <= 0
                               // if x > 0
                               _mm256_add_pd(x_aligned, log1p_exp_of_argument), is_x_gr_0);

}




// //https://stackoverflow.com/a/65537754/9007125
// Isolate rare subnormal/domain repairs from the normal-positive SIMD path.
__attribute__((noinline, cold)) inline __m256d nico_log_exceptional_lanes_AVX2(const __m256d input,
              const __m256d result,
              const unsigned ordinary_mask) {

    ALIGN32  __m256d const input_aligned = input;
    ALIGN32  __m256d const result_aligned = result;
    alignas(32) double input_values[4], output_values[4];
    _mm256_store_pd(input_values, input_aligned);
    _mm256_store_pd(output_values, result_aligned);
    for (int lane = 0; lane < 4; ++lane) {
        if (!(ordinary_mask & (1u << lane))) output_values[lane] = std::log(input_values[lane]);
    }
    return _mm256_load_pd(output_values);

}

ALWAYS_INLINE __m256d nico_log_1_AVX2 VECTORCALL(const __m256d a) {

      ALIGN32  __m256d const a_aligned = a;

      // The mantissa/exponent shortcut assumes a positive NORMAL double.
      ALIGN32  __m256d const ordinary_lanes = _mm256_and_pd(
          _mm256_cmp_pd(a_aligned, _mm256_set1_pd(std::numeric_limits<double>::min()), _CMP_GE_OQ),
          _mm256_cmp_pd(a_aligned, _mm256_set1_pd(INFINITY), _CMP_LT_OQ));
      const int ordinary_mask = _mm256_movemask_pd(ordinary_lanes);
      ALIGN32  __m256d const result = nico_log_1_wo_checks_AVX2(a_aligned);
      if (__builtin_expect(ordinary_mask != 0xf, 0)) return nico_log_exceptional_lanes_AVX2(a_aligned, result,
              ordinary_mask);
      return result;
  
}

// double-precision log (fdlibm, one division per vector) with the same scalar repair of non-normal / non-positive lanes
ALWAYS_INLINE __m256d nico_log_1_double_precision_AVX2 VECTORCALL(const __m256d a) {

      ALIGN32  __m256d const a_aligned = a;

      ALIGN32  __m256d const ordinary_lanes = _mm256_and_pd(
          _mm256_cmp_pd(a_aligned, _mm256_set1_pd(std::numeric_limits<double>::min()), _CMP_GE_OQ),
          _mm256_cmp_pd(a_aligned, _mm256_set1_pd(INFINITY), _CMP_LT_OQ));
      const int ordinary_mask = _mm256_movemask_pd(ordinary_lanes);
      ALIGN32  __m256d const result = nico_log_1_double_precision_wo_checks_AVX2(a_aligned);
      if (__builtin_expect(ordinary_mask != 0xf, 0)) {
          return nico_log_exceptional_lanes_AVX2(a_aligned, result, ordinary_mask);
      }
      return result;

}




// //  compute log(1+x) without losing precision for small values of x.
// //  see: https://www.johndcook.com/cpp_log_one_plus_x.html
ALWAYS_INLINE __m256d nico_log1p_1_AVX2 VECTORCALL(const __m256d x) {

      ALIGN32  __m256d const x_aligned = x;
  
      ALIGN32  __m256d const minus_1 =  _mm256_set1_pd(-1.0);
      ALIGN32  __m256d const neg_half = _mm256_set1_pd(-0.5);
      ALIGN32  __m256d const sign_bit = _mm256_set1_pd(-0.0);
      
      ALIGN32  __m256d const res =  _mm256_blendv_pd(
                                    _mm256_blendv_pd(
                                      _mm256_mul_pd(_mm256_fmadd_pd(neg_half, x_aligned,  _mm256_set1_pd(1.0)),
                                              x_aligned),
                                      nico_log_1_AVX2(_mm256_add_pd(x_aligned,  _mm256_set1_pd(1.0))),
                                      _mm256_cmp_pd(_mm256_andnot_pd(sign_bit, x_aligned), _mm256_set1_pd(1e-4),
                                              _CMP_GT_OQ)
                                    ), 
                                    nico_log_1_AVX2(_mm256_add_pd(x_aligned,  _mm256_set1_pd(1.0))),
                                    _mm256_cmp_pd(x_aligned, minus_1, _CMP_LE_OQ)
                                  );
      
      return res;
  
} 




// Isolate x <= -1, infinities and NaN (std::log1p) from the SIMD path.
__attribute__((noinline, cold)) inline __m256d nico_log1p_exceptional_lanes_AVX2(const __m256d input,
                                                                                const __m256d result,
                                                                                const unsigned ordinary_mask) {

    ALIGN32  __m256d const input_aligned = input;
    ALIGN32  __m256d const result_aligned = result;
    alignas(32) double input_values[4], output_values[4];
    _mm256_store_pd(input_values, input_aligned);
    _mm256_store_pd(output_values, result_aligned);
    for (int lane = 0; lane < 4; ++lane) {
        if (!(ordinary_mask & (1u << lane))) output_values[lane] = std::log1p(input_values[lane]);
    }
    return _mm256_load_pd(output_values);

}

ALWAYS_INLINE __m256d nico_log1p_1_double_precision_AVX2 VECTORCALL(const __m256d x) {

      ALIGN32  __m256d const x_aligned = x;

      ALIGN32  __m256d const ordinary_lanes = _mm256_and_pd(_mm256_cmp_pd(x_aligned, _mm256_set1_pd(-1.0),
              _CMP_GT_OQ),
                                                   _mm256_cmp_pd(x_aligned, _mm256_set1_pd(INFINITY),
                                                           _CMP_LT_OQ));
      const int ordinary_mask = _mm256_movemask_pd(ordinary_lanes);
      ALIGN32  __m256d const result = nico_log1p_1_double_precision_wo_checks_AVX2(x_aligned);
      if (__builtin_expect(ordinary_mask != 0xf, 0)) {
          return nico_log1p_exceptional_lanes_AVX2(x_aligned, result, ordinary_mask);
      }
      return result;

}




// //  compute log(1-x) without losing precision for small values of x.
ALWAYS_INLINE __m256d nico_log1m_1_AVX2 VECTORCALL(const __m256d x) {

      ALIGN32  __m256d const x_aligned = x;
  
      ALIGN32  __m256d const res =  nico_log1p_1_AVX2(_mm256_sub_pd(_mm256_setzero_pd(), x_aligned));
      return res;
  
}  




// //  compute log(1+x) without losing precision for small values of x. //  see: https://www.johndcook.com/cpp_log_one_plus_x.html
// ALWAYS_INLINE __m256d nico_log1p_exp_1_AVX2 VECTORCALL(const __m256d x) {
//
//       ALIGN32  __m256d const neg_x = _mm256_sub_pd(_mm256_setzero_pd(), x);
//
//       ALIGN32  __m256d const res =  _mm256_blendv_pd(
//                                       nico_log1p_1_AVX2(nico_exp_1_AVX2(x)),
//                                       _mm256_add_pd(x, nico_log1p_1_AVX2(nico_exp_1_AVX2(neg_x))), 
//                                       _mm256_cmp_pd(x, _mm256_setzero_pd(), _CMP_GT_OQ)
//                                     );
//
//       return res;
//
// }
// (one exp and one log1p / one division per lane, of the argument which the lane's branch uses, instead of
//  both branches in every lane: the same operations, lane by lane, and so the same values, at half the cost)
ALWAYS_INLINE __m256d nico_log1p_exp_1_AVX2 VECTORCALL(const __m256d x) {

       ALIGN32  __m256d const x_aligned = x;
       ALIGN32  __m256d const is_x_gr_0 = _mm256_cmp_pd(x_aligned, _mm256_setzero_pd(), _CMP_GT_OQ);
       ALIGN32  __m256d const argument = _mm256_blendv_pd(x_aligned, _mm256_sub_pd(_mm256_setzero_pd(),
               x_aligned), is_x_gr_0);
       ALIGN32  __m256d const log1p_exp_of_argument = nico_log1p_1_AVX2(nico_exp_1_AVX2(argument));
       return _mm256_blendv_pd(log1p_exp_of_argument,                                             // if x <= 0
                               // if x > 0
                               _mm256_add_pd(x_aligned, log1p_exp_of_argument), is_x_gr_0);

}




//// ---- log(1 - exp(x)) (log1m_exp), x <= 0: x <  -ln 2:  log1p(-exp(x)) with the default nico_exp_1 and
////      nico_log1p_1 (as the Stan external's log1p(-exp(x)) branch: the same values) x >= -ln 2:  log(-expm1(x))
////      = log(-x) + h(x),  h(x) = log(expm1(x) / x) = x/2 + log(sinh(x/2) / (x/2)) = x/2 + sum_k B_2k x^2k / (2k
////      (2k)!)  (B_2k the Bernoulli numbers), to k = 8 (x^16): truncation below 1e-18 and rounding about 3e-17
////      on [-ln 2, 0); log(-x) by the double-precision log (nico_log_1_double_precision_*): no cancellation as x
////      -> 0, and no scalar log / expm1. Each branch is evaluated only when at least one lane needs it. x = 0
////      gives -inf, x > 0 and NaN give NaN, x = -inf gives 0 (exceptional lanes go through the scalar repairs of
////      nico_exp_1, nico_log_1, nico_log1p_1).
static constexpr double FAST_LOG1M_EXP_LN2_AVX2 = 0.693147180559945309417232121458;
ALWAYS_INLINE __m256d nico_log1m_exp_1_AVX2 VECTORCALL(const __m256d x) {

      ALIGN32  __m256d const x_aligned = x;

      // x >= -ln 2 (NaN: false)
      ALIGN32  __m256d const near_zero  = _mm256_cmp_pd(x_aligned, _mm256_set1_pd(-FAST_LOG1M_EXP_LN2_AVX2),
              _CMP_GE_OQ);
      const int     near_lanes = _mm256_movemask_pd(near_zero);
      ALIGN32  __m256d far_result  = _mm256_setzero_pd();
      ALIGN32  __m256d near_result = _mm256_setzero_pd();
      if (near_lanes != 0xf) {
          far_result = nico_log1p_1_AVX2(_mm256_sub_pd(_mm256_setzero_pd(), nico_exp_1_AVX2(x_aligned)));
      }
      if (near_lanes != 0) {
          ALIGN32  __m256d const u = _mm256_mul_pd(x_aligned, x_aligned);
          ALIGN32  __m256d e = _mm256_set1_pd(-3617.0 / 170729965486080000.0);
          e = _mm256_fmadd_pd(e, u, _mm256_set1_pd(1.0 / 1046139494400.0));
          e = _mm256_fmadd_pd(e, u, _mm256_set1_pd(-691.0 / 15692092416000.0));
          e = _mm256_fmadd_pd(e, u, _mm256_set1_pd(1.0 / 479001600.0));
          e = _mm256_fmadd_pd(e, u, _mm256_set1_pd(-1.0 / 9676800.0));
          e = _mm256_fmadd_pd(e, u, _mm256_set1_pd(1.0 / 181440.0));
          e = _mm256_fmadd_pd(e, u, _mm256_set1_pd(-1.0 / 2880.0));
          e = _mm256_fmadd_pd(e, u, _mm256_set1_pd(1.0 / 24.0));
          ALIGN32  __m256d const h = _mm256_fmadd_pd(u, e, _mm256_mul_pd(_mm256_set1_pd(0.5), x_aligned));
          ALIGN32  __m256d const log_of_minus_x =
                  nico_log_1_double_precision_AVX2(_mm256_sub_pd(_mm256_setzero_pd(), x_aligned));
          near_result = _mm256_add_pd(log_of_minus_x, h);
      }
      return _mm256_blendv_pd(far_result, near_result, near_zero);

}




ALWAYS_INLINE     __m256d nico_logit_wo_checks_AVX2 VECTORCALL(const __m256d x)   {

      ALIGN32  __m256d const x_aligned = x;
  
      ALIGN32  __m256d const log_x = nico_log_1_AVX2(x_aligned);
      ALIGN32  __m256d const log_1m_x =  nico_log1m_1_AVX2(x_aligned);
      ALIGN32  __m256d const res =   log_x - log_1m_x;
      return res;
  
} 




ALWAYS_INLINE     __m256d nico_logit_AVX2 VECTORCALL(const __m256d x)   {

      ALIGN32  __m256d const x_aligned = x;
  
      ALIGN32  __m256d const log_x = nico_log_1_AVX2(x_aligned);
      ALIGN32  __m256d const log_1m_x =  nico_log1m_1_AVX2(x_aligned);
      ALIGN32  __m256d const res =   log_x - log_1m_x;
      return res;
  
} 




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
ALWAYS_INLINE     __m256d  nico_inv_logit_for_x_pos_AVX2 VECTORCALL(const __m256d x )  {

          ALIGN32  __m256d const x_aligned = x;
  
          ALIGN32  __m256d const  exp_m_x =  nico_exp_1_AVX2(_mm256_sub_pd(_mm256_setzero_pd(), x_aligned)) ;
          ALIGN32  __m256d const res =    _mm256_div_pd( _mm256_set1_pd(1.0),
                                                        _mm256_add_pd( _mm256_set1_pd(1.0), exp_m_x))  ;
          return res;

}




ALWAYS_INLINE     __m256d  nico_inv_logit_for_x_neg_AVX2 VECTORCALL(const __m256d x )  {

          ALIGN32  __m256d const x_aligned = x;
  
          ALIGN32  __m256d const  log_eps  =   _mm256_set1_pd(-18.420680743952367);
          ALIGN32  __m256d const  exp_x =  nico_exp_1_AVX2(x_aligned) ;
          
          ALIGN32  __m256d const res = _mm256_blendv_pd(    exp_x,
                                      _mm256_div_pd(exp_x, _mm256_add_pd( _mm256_set1_pd(1.0), exp_x)), 
                                      _mm256_cmp_pd(x_aligned,  log_eps, _CMP_GT_OQ));
          return res;

}




// ALWAYS_INLINE     __m256d  nico_inv_logit_AVX2 VECTORCALL(const __m256d x )  {
//   
//           ALIGN32  __m256d const res =  _mm256_blendv_pd(  nico_inv_logit_for_x_neg_AVX2(x),
//                                        nico_inv_logit_for_x_pos_AVX2(x), 
//                                        _mm256_cmp_pd(x, _mm256_setzero_pd(), _CMP_GE_OQ));  // x > 0
//           return res;
//   
// }
// ALWAYS_INLINE __m256d nico_inv_logit_AVX2 VECTORCALL(const __m256d x) {
//
//           ALIGN32 __m256d const result = _mm256_blendv_pd(nico_inv_logit_for_x_neg_AVX2(x),
//                                                            nico_inv_logit_for_x_pos_AVX2(x),
//                                                            _mm256_cmp_pd(x, _mm256_setzero_pd(), _CMP_GE_OQ));
//           return _mm256_min_pd(_mm256_set1_pd(1.0), _mm256_max_pd(_mm256_setzero_pd(), result));
//
// }
// (one exp and one log1p / one division per lane, of the argument which the lane's branch uses, instead of
//  both branches in every lane: the same operations, lane by lane, and so the same values, at half the cost)
ALWAYS_INLINE __m256d nico_inv_logit_AVX2 VECTORCALL(const __m256d x) {

          ALIGN32  __m256d const x_aligned = x;
          ALIGN32 __m256d const one = _mm256_set1_pd(1.0);
          ALIGN32 __m256d const is_x_pos = _mm256_cmp_pd(x_aligned, _mm256_setzero_pd(), _CMP_GE_OQ);
          ALIGN32 __m256d const is_x_gr_log_eps = _mm256_cmp_pd(x_aligned, _mm256_set1_pd(-18.420680743952367),
                                                                _CMP_GT_OQ);
          ALIGN32 __m256d const exp_of_argument =
              nico_exp_1_AVX2(_mm256_blendv_pd(x_aligned, _mm256_sub_pd(_mm256_setzero_pd(), x_aligned),
                      is_x_pos));
          //// 1 / (1 + exp(-x)) (x pos), exp(x) / (1 + exp(x)) (x neg, x > log_eps), exp(x) (x <= log_eps)
          ALIGN32 __m256d const quotient = _mm256_div_pd(_mm256_blendv_pd(exp_of_argument, one, is_x_pos),
                                                         _mm256_add_pd(one, exp_of_argument));
          ALIGN32 __m256d const result = _mm256_blendv_pd(exp_of_argument, quotient,
                                                          _mm256_or_pd(is_x_pos, is_x_gr_log_eps));
          return _mm256_min_pd(one, _mm256_max_pd(_mm256_setzero_pd(), result));

}




ALWAYS_INLINE      __m256d  nico_inv_logit_for_x_pos_wo_checks_AVX2 VECTORCALL(const __m256d x )  {

          ALIGN32  __m256d const x_aligned = x;

          ALIGN32  __m256d const  exp_m_x =  nico_exp_1_wo_checks_AVX2(_mm256_sub_pd(_mm256_setzero_pd(),
                  x_aligned)) ;
          ALIGN32  __m256d const res =    _mm256_div_pd( _mm256_set1_pd(1.0),
                                                        _mm256_add_pd( _mm256_set1_pd(1.0), exp_m_x))  ;
          return res;
  
}




ALWAYS_INLINE     __m256d  nico_inv_logit_for_x_neg_wo_checks_AVX2 VECTORCALL(const __m256d x )  {

          ALIGN32  __m256d const x_aligned = x;
  
          ALIGN32  __m256d const  log_eps  =   _mm256_set1_pd(-18.420680743952367);
          ALIGN32  __m256d const  exp_x =  nico_exp_1_wo_checks_AVX2(x_aligned) ;
          
          ALIGN32  __m256d const res = _mm256_blendv_pd(    exp_x,
                                      _mm256_div_pd(exp_x, _mm256_add_pd( _mm256_set1_pd(1.0), exp_x)), 
                                      _mm256_cmp_pd(x_aligned,  log_eps, _CMP_GT_OQ));
          return res;
  
}




// ALWAYS_INLINE      __m256d  nico_inv_logit_wo_checks_AVX2 VECTORCALL(const __m256d x )  {
//   
//           ALIGN32  __m256d const res =  _mm256_blendv_pd(    nico_inv_logit_for_x_neg_wo_checks_AVX2(x),
//                                        nico_inv_logit_for_x_pos_wo_checks_AVX2(x), 
//                                        _mm256_cmp_pd(x,  _mm256_setzero_pd(), _CMP_GT_OQ));  // x > 0
//           return res;
//       
// }
// ALWAYS_INLINE __m256d nico_inv_logit_wo_checks_AVX2 VECTORCALL(const __m256d x) {
//
//           ALIGN32 __m256d const result = _mm256_blendv_pd(nico_inv_logit_for_x_neg_wo_checks_AVX2(x),
//                                                            nico_inv_logit_for_x_pos_wo_checks_AVX2(x),
//                                                            _mm256_cmp_pd(x, _mm256_setzero_pd(), _CMP_GT_OQ));
//           return _mm256_min_pd(_mm256_set1_pd(1.0), _mm256_max_pd(_mm256_setzero_pd(), result));
//
// }
// (one exp and one log1p / one division per lane, of the argument which the lane's branch uses, instead of
//  both branches in every lane: the same operations, lane by lane, and so the same values, at half the cost)
ALWAYS_INLINE __m256d nico_inv_logit_wo_checks_AVX2 VECTORCALL(const __m256d x) {

          ALIGN32  __m256d const x_aligned = x;
          ALIGN32 __m256d const one = _mm256_set1_pd(1.0);
          ALIGN32 __m256d const is_x_pos = _mm256_cmp_pd(x_aligned, _mm256_setzero_pd(), _CMP_GT_OQ);
          ALIGN32 __m256d const is_x_gr_log_eps = _mm256_cmp_pd(x_aligned, _mm256_set1_pd(-18.420680743952367),
                                                                _CMP_GT_OQ);
          ALIGN32 __m256d const exp_of_argument =
              nico_exp_1_wo_checks_AVX2(_mm256_blendv_pd(x_aligned, _mm256_sub_pd(_mm256_setzero_pd(),
                      x_aligned), is_x_pos));
          //// 1 / (1 + exp(-x)) (x pos), exp(x) / (1 + exp(x)) (x neg, x > log_eps), exp(x) (x <= log_eps)
          ALIGN32 __m256d const quotient = _mm256_div_pd(_mm256_blendv_pd(exp_of_argument, one, is_x_pos),
                                                         _mm256_add_pd(one, exp_of_argument));
          ALIGN32 __m256d const result = _mm256_blendv_pd(exp_of_argument, quotient,
                                                          _mm256_or_pd(is_x_pos, is_x_gr_log_eps));
          return _mm256_min_pd(one, _mm256_max_pd(_mm256_setzero_pd(), result));

}




///////////////////  fns - log_inv_logit   ----------------------------------------------------------------------------------------------------------------
// inline     __m256d  nico_log_inv_logit_for_x_pos_AVX2 VECTORCALL(const __m256d x )  {
//    
//       ALIGN32 const    __m256d   m_x = _mm256_sub_pd(_mm256_setzero_pd(), x);
//     
//       ALIGN32  __m256d const res =   _mm256_sub_pd(_mm256_setzero_pd(), nico_log1p_exp_1_AVX2(nico_exp_1_AVX2(m_x)));
//       return res;
//     
// }
inline __m256d nico_log_inv_logit_for_x_pos_AVX2 VECTORCALL(const __m256d x) {

      ALIGN32  __m256d const x_aligned = x;
  
      ALIGN32 __m256d const m_x     = _mm256_sub_pd(_mm256_setzero_pd(), x_aligned);   // -x
      ALIGN32 __m256d const exp_m_x = nico_exp_1_AVX2(m_x);                    // exp(-x)
      ALIGN32 __m256d const log_sum = nico_log1p_1_AVX2(exp_m_x);              // log(1 + exp(-x))
      return _mm256_sub_pd(_mm256_setzero_pd(), log_sum);                      // -log(1 + exp(-x))
      
}




inline     __m256d  nico_log_inv_logit_for_x_neg_AVX2 VECTORCALL(const __m256d x )  {

      ALIGN32  __m256d const x_aligned = x;

      ALIGN32  __m256d const  log_eps  =   _mm256_set1_pd(-18.420680743952367);
    
      ALIGN32  __m256d const res = _mm256_blendv_pd(    x_aligned,
                                  _mm256_sub_pd(x_aligned,   nico_log1p_1_AVX2(nico_exp_1_AVX2(x_aligned))),
                                  _mm256_cmp_pd(x_aligned,  log_eps, _CMP_GT_OQ));
      return res;
    
}




// inline     __m256d  nico_log_inv_logit_AVX2 VECTORCALL(const __m256d x )  {
//
//       ALIGN32  __m256d const res =  _mm256_blendv_pd(    nico_log_inv_logit_for_x_neg_AVX2(x),
//                                      nico_log_inv_logit_for_x_pos_AVX2(x), 
//                                      _mm256_cmp_pd(x,  _mm256_setzero_pd(), _CMP_GE_OQ));  // x > 0
//       return res;
//
// }
// (one exp and one log1p / one division per lane, of the argument which the lane's branch uses, instead of
//  both branches in every lane: the same operations, lane by lane, and so the same values, at half the cost)
inline     __m256d  nico_log_inv_logit_AVX2 VECTORCALL(const __m256d x )  {

      ALIGN32  __m256d const x_aligned = x;
      ALIGN32  __m256d const is_x_ge_0 = _mm256_cmp_pd(x_aligned, _mm256_setzero_pd(), _CMP_GE_OQ);
      ALIGN32  __m256d const is_x_gr_log_eps = _mm256_cmp_pd(x_aligned, _mm256_set1_pd(-18.420680743952367),
              _CMP_GT_OQ);
      ALIGN32  __m256d const log1p_exp_of_argument =
          nico_log1p_1_AVX2(nico_exp_1_AVX2(_mm256_blendv_pd(x_aligned, _mm256_sub_pd(_mm256_setzero_pd(),
                  x_aligned),
                                                             is_x_ge_0)));
      //// x - log(1 + exp(x)) (x < 0, x > log_eps), x (x <= log_eps), -log(1 + exp(-x)) (x >= 0)
      ALIGN32  __m256d const result_of_x_neg = _mm256_blendv_pd(x_aligned, _mm256_sub_pd(x_aligned,
              log1p_exp_of_argument),
                                                                is_x_gr_log_eps);
      return _mm256_blendv_pd(result_of_x_neg, _mm256_sub_pd(_mm256_setzero_pd(), log1p_exp_of_argument),
                              is_x_ge_0);

}




////////////
inline     __m256d  nico_log_inv_logit_for_x_pos_wo_checks_AVX2 VECTORCALL(const __m256d x )  {

      ALIGN32  __m256d const x_aligned = x;
      
      ALIGN32  __m256d const res =   _mm256_sub_pd(_mm256_setzero_pd(), nico_log1p_exp_1_wo_checks_AVX2((
              _mm256_sub_pd(_mm256_setzero_pd(), x_aligned)) ) );
      return res;
   
}




inline     __m256d  nico_log_inv_logit_for_x_neg_wo_checks_AVX2 VECTORCALL(const __m256d x )  {

      ALIGN32  __m256d const x_aligned = x;
  
      ALIGN32  __m256d const  log_eps  =   _mm256_set1_pd(-18.420680743952367);
    
      ALIGN32  __m256d const res = _mm256_blendv_pd( x_aligned,
                                  _mm256_sub_pd(x_aligned,   nico_log1p_1_wo_checks_AVX2(
                                          nico_exp_1_wo_checks_AVX2(x_aligned))),
                                  _mm256_cmp_pd(x_aligned,  log_eps, _CMP_GT_OQ));
      return res;
    
}




// inline     __m256d  nico_log_inv_logit_wo_checks_AVX2 VECTORCALL(const __m256d x )  {
//
//       ALIGN32  __m256d const res =  _mm256_blendv_pd(    nico_log_inv_logit_for_x_neg_wo_checks_AVX2(x),
//                                      nico_log_inv_logit_for_x_pos_wo_checks_AVX2(x),
//                                      _mm256_cmp_pd(x,  _mm256_setzero_pd(), _CMP_GE_OQ));  // x > 0
//       return res;
//
// }
// (one exp and one log1p / one division per lane, of the argument which the lane's branch uses, instead of
//  both branches in every lane: the same operations, lane by lane, and so the same values, at half the cost)
inline     __m256d  nico_log_inv_logit_wo_checks_AVX2 VECTORCALL(const __m256d x )  {

      ALIGN32  __m256d const x_aligned = x;
      ALIGN32  __m256d const is_x_ge_0 = _mm256_cmp_pd(x_aligned, _mm256_setzero_pd(), _CMP_GE_OQ);
      ALIGN32  __m256d const is_x_gr_log_eps = _mm256_cmp_pd(x_aligned, _mm256_set1_pd(-18.420680743952367),
              _CMP_GT_OQ);
      ALIGN32  __m256d const log1p_exp_of_argument =
          nico_log1p_1_wo_checks_AVX2(nico_exp_1_wo_checks_AVX2(
              _mm256_blendv_pd(x_aligned, _mm256_sub_pd(_mm256_setzero_pd(), x_aligned), is_x_ge_0)));
      //// x - log(1 + exp(x)) (x < 0, x > log_eps), x (x <= log_eps), -log(1 + exp(-x)) (x >= 0)
      ALIGN32  __m256d const result_of_x_neg = _mm256_blendv_pd(x_aligned, _mm256_sub_pd(x_aligned,
              log1p_exp_of_argument),
                                                                is_x_gr_log_eps);
      return _mm256_blendv_pd(result_of_x_neg, _mm256_sub_pd(_mm256_setzero_pd(), log1p_exp_of_argument),
                              is_x_ge_0);

}




/////////////////////  fns - Phi_approx   ----------------------------------------------------------------------------------------------------------------
inline     __m256d  nico_Phi_approx_wo_checks_AVX2 VECTORCALL(const __m256d x)  {

      ALIGN32  __m256d const x_aligned = x;
  
      ALIGN32  __m256d const a =     _mm256_set1_pd(0.07056);
      ALIGN32  __m256d const b =     _mm256_set1_pd(1.5976);
      
      ALIGN32  __m256d const   x_sq =  _mm256_mul_pd(x_aligned, x_aligned);
      ALIGN32  __m256d const   a_x_sq_plus_b = _mm256_fmadd_pd(a, x_sq, b);
      ALIGN32  __m256d const   stuff_to_inv_logit =  _mm256_mul_pd(x_aligned, a_x_sq_plus_b);
      
      ALIGN32  __m256d const res = nico_inv_logit_wo_checks_AVX2(stuff_to_inv_logit);
      return res;
  
}




inline     __m256d  nico_Phi_approx_AVX2 VECTORCALL(const __m256d x )  {

      ALIGN32  __m256d const x_aligned = x;
  
      ALIGN32  __m256d const a =     _mm256_set1_pd(0.07056);
      ALIGN32  __m256d const b =     _mm256_set1_pd(1.5976);
      
      ALIGN32  __m256d const  x_sq =  _mm256_mul_pd(x_aligned, x_aligned);
      ALIGN32  __m256d const  a_x_sq_plus_b = _mm256_fmadd_pd(a, x_sq, b);
      ALIGN32  __m256d const   stuff_to_inv_logit =  _mm256_mul_pd(x_aligned, a_x_sq_plus_b);
      
      ALIGN32  __m256d const res = nico_inv_logit_AVX2(stuff_to_inv_logit);
      return res;
  
}




///////////////////  fns - inv_Phi_approx   ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
inline      __m256d  nico_inv_Phi_approx_wo_checks_AVX2 VECTORCALL(const __m256d x )  {

      ALIGN32  __m256d const x_aligned = x;
  
      ALIGN32  __m256d const  inv_Phi_approx_c1 = _mm256_set1_pd(-0.34176618822627863);
      ALIGN32  __m256d const  inv_Phi_approx_c2 = _mm256_set1_pd(2.7472242570765295);
      ALIGN32  __m256d const  one_third =  _mm256_set1_pd(0.33333333333333331483);
      
      // logit first
      ALIGN32  __m256d const  m_logit_p = nico_log_1_wo_checks_AVX2( _mm256_sub_pd(_mm256_div_pd( _mm256_set1_pd(
              1.0),  x_aligned),  _mm256_set1_pd(1.0) ) );
      ALIGN32  __m256d const  x_i =  _mm256_mul_pd(m_logit_p,
                                                   inv_Phi_approx_c1) ; // -0.34176618822627863*m_logit_p;
      ALIGN32  __m256d const  asinh_stuff_div_3 =  _mm256_mul_pd( one_third,  nico_log_1_wo_checks_AVX2(  _mm256_add_pd(x_i, (  _mm256_sqrt_pd(  _mm256_fmadd_pd(x_i, x_i,  _mm256_set1_pd(1.0)) ) )  ) ) ) ;          // now do arc_sinh part
      ALIGN32  __m256d const  exp_x_i = nico_exp_1_wo_checks_AVX2(asinh_stuff_div_3);
      
      ALIGN32  __m256d const res = _mm256_mul_pd(  inv_Phi_approx_c2,   _mm256_div_pd(_mm256_sub_pd(_mm256_mul_pd(exp_x_i, exp_x_i),  _mm256_set1_pd(1.0)), exp_x_i) ) ;    //   now do sinh parth part
      return res;
  
}




inline     __m256d  nico_inv_Phi_approx_AVX2 VECTORCALL(const __m256d x )  {

      ALIGN32  __m256d const x_aligned = x;

      // Avoid 1 / p overflowing for valid subnormal probabilities. The usual path below is unchanged.
      ALIGN32  __m256d const ordinary_lanes = _mm256_and_pd(
          _mm256_cmp_pd(x_aligned, _mm256_set1_pd(std::numeric_limits<double>::min()), _CMP_GE_OQ),
          _mm256_cmp_pd(x_aligned, _mm256_set1_pd(1.0), _CMP_LT_OQ));
      const int ordinary_mask = _mm256_movemask_pd(ordinary_lanes);
      if (ordinary_mask != 0xf) {
          alignas(32) double input_values[4], output_values[4];
          _mm256_store_pd(input_values, x_aligned);
          ALIGN32  __m256d const safe_input = _mm256_blendv_pd(_mm256_set1_pd(0.5), x_aligned, ordinary_lanes);
          _mm256_store_pd(output_values, nico_inv_Phi_approx_AVX2(safe_input));
          for (int lane = 0; lane < 4; ++lane) {
              if (!(ordinary_mask & (1u << lane))) {
                  const double logit_probability = std::log(input_values[lane]) - std::log1p(-input_values[lane]);
                  output_values[lane] = 5.494448514153059 * std::sinh(std::asinh(0.34176618822627863 * logit_probability) / 3.0);
              }
          }
          return _mm256_load_pd(output_values);
      }
  
      ALIGN32  __m256d const  one_third =  _mm256_set1_pd(0.33333333333333331483);
      
      // log((1-p)/p), stable near 1
      ALIGN32  __m256d const m_logit_p = nico_log_1_AVX2(_mm256_div_pd(_mm256_sub_pd(_mm256_set1_pd(1.0),
              x_aligned), x_aligned));
      ALIGN32  __m256d const x_i =  _mm256_mul_pd(m_logit_p, _mm256_set1_pd(-0.34176618822627863)) ; // -0.34176618822627863*m_logit_p;
      ALIGN32  __m256d const asinh_stuff_div_3 =  _mm256_mul_pd(  one_third,  nico_log_1_AVX2(  _mm256_add_pd(x_i, (  _mm256_sqrt_pd(  _mm256_fmadd_pd(x_i, x_i,  _mm256_set1_pd(1.0)) ) )  ) ) ) ;          // now do arc_sinh part
      ALIGN32  __m256d const exp_x_i = nico_exp_1_AVX2(asinh_stuff_div_3);
      
      ALIGN32  __m256d const res = _mm256_mul_pd(  _mm256_set1_pd(2.7472242570765295),   _mm256_div_pd( _mm256_sub_pd(_mm256_mul_pd(exp_x_i, exp_x_i),  _mm256_set1_pd(1.0)), exp_x_i) ) ;    //   now do sinh parth part
      return res;
      
}




// need to add citation to this (slight modification from a forum post)
inline      __m256d  nico_inv_Phi_approx_from_logit_prob_wo_checks_AVX2 VECTORCALL(const __m256d logit_p )  {

      ALIGN32  __m256d const logit_p_aligned = logit_p;
  
      // ALIGN32  __m256d const  inv_Phi_approx_c1 = _mm256_set1_pd(-0.34176618822627863);
      ALIGN32  __m256d const  inv_Phi_approx_c1 = _mm256_set1_pd(0.34176618822627863);
      ALIGN32  __m256d const  inv_Phi_approx_c2 = _mm256_set1_pd(2.7472242570765295);
      ALIGN32  __m256d const  one_third =  _mm256_set1_pd(0.33333333333333331483);
      
      ALIGN32  __m256d const  x_i =  _mm256_mul_pd(logit_p_aligned,  inv_Phi_approx_c1) ;
      ALIGN32  __m256d const  asinh_stuff_div_3 =  _mm256_mul_pd(  one_third,  nico_log_1_wo_checks_AVX2( _mm256_add_pd(x_i  , _mm256_sqrt_pd(  _mm256_fmadd_pd(x_i, x_i,  _mm256_set1_pd(1.0)) ) )  ) ) ;          // now do arc_sinh part
      ALIGN32  __m256d const  exp_x_i = nico_exp_1_wo_checks_AVX2(asinh_stuff_div_3);
      
      ALIGN32  __m256d const res = _mm256_mul_pd(inv_Phi_approx_c2,   _mm256_div_pd(_mm256_sub_pd( _mm256_mul_pd(exp_x_i, exp_x_i),  _mm256_set1_pd(1.0)), exp_x_i) ) ;   //   now do sinh parth part
      return res;
  
  
}




inline     __m256d  nico_inv_Phi_approx_from_logit_prob_AVX2 VECTORCALL(const __m256d logit_p )  {

      ALIGN32  __m256d const logit_p_aligned = logit_p;
  
      // ALIGN32  __m256d const  inv_Phi_approx_c1 = _mm256_set1_pd(-0.34176618822627863);
      ALIGN32  __m256d const  inv_Phi_approx_c1 = _mm256_set1_pd(0.34176618822627863);
      ALIGN32  __m256d const  inv_Phi_approx_c2 = _mm256_set1_pd(2.7472242570765295);
      ALIGN32  __m256d const  one_third =  _mm256_set1_pd(0.33333333333333331483);
      
      ALIGN32  __m256d const  x_i =  _mm256_mul_pd(logit_p_aligned,  inv_Phi_approx_c1) ;
      ALIGN32  __m256d const  asinh_stuff_div_3 =  _mm256_mul_pd( one_third,  nico_log_1_AVX2( _mm256_add_pd(x_i  , _mm256_sqrt_pd(  _mm256_fmadd_pd(x_i, x_i,  _mm256_set1_pd(1.0)) ) )  ) ) ;          // now do arc_sinh part
      ALIGN32  __m256d const  exp_x_i = nico_exp_1_AVX2(asinh_stuff_div_3);
      
      ALIGN32  __m256d const res = _mm256_mul_pd(inv_Phi_approx_c2,   _mm256_div_pd(_mm256_sub_pd( _mm256_mul_pd(exp_x_i, exp_x_i),  _mm256_set1_pd(1.0)), exp_x_i) ) ;   //   now do sinh parth part
      return res;
  
}




///////////////////  fns - log_Phi_approx   ----------------------------------------------------------------------------------------------------------------------------------------
inline      __m256d  nico_log_Phi_approx_wo_checks_AVX2 VECTORCALL(const __m256d x )  {

      ALIGN32  __m256d const x_aligned = x;
  
      ALIGN32  __m256d const a =     _mm256_set1_pd(0.07056);
      ALIGN32  __m256d const b =     _mm256_set1_pd(1.5976);
      
      ALIGN32  __m256d const  x_sq =  _mm256_mul_pd(x_aligned, x_aligned);
      ALIGN32  __m256d const  a_x_sq_plus_b = _mm256_fmadd_pd(a, x_sq, b);
      ALIGN32  __m256d const  stuff_to_inv_logit =  _mm256_mul_pd(x_aligned, a_x_sq_plus_b);
      
      ALIGN32  __m256d const res = nico_log_inv_logit_wo_checks_AVX2(stuff_to_inv_logit);
      return res;
  
}




inline __m256d nico_log_Phi_approx_AVX2 VECTORCALL(const __m256d x) {

      ALIGN32  __m256d const x_aligned = x;
  
      ALIGN32  __m256d const a =     _mm256_set1_pd(0.07056);
      ALIGN32  __m256d const b =     _mm256_set1_pd(1.5976);
      ALIGN32  __m256d const Phi_upper_bound =  _mm256_set1_pd(8.25);
      ALIGN32  __m256d const Phi_lower_bound =  _mm256_set1_pd(-37.5);
      
      ALIGN32  __m256d const x_sq = _mm256_mul_pd(x_aligned, x_aligned);
      ALIGN32  __m256d const x_cubed = _mm256_mul_pd(x_sq, x_aligned);
      ALIGN32  __m256d const result = _mm256_fmadd_pd(a, x_cubed, _mm256_mul_pd(b, x_aligned));
      
      ALIGN32  __m256d const res = nico_log_inv_logit_AVX2(result);
      return res;
  
} 




//////////////////// ------------- tanh  --------------------------------------------------------------------------------------------------------------------------------------
ALWAYS_INLINE      __m256d    nico_tanh_AVX2 VECTORCALL( const  __m256d x  )   {

      ALIGN32  __m256d const x_aligned = x;
  
  
      ALIGN32  __m256d const  two =     _mm256_set1_pd(2.0);
      ALIGN32  __m256d const  m_two =   _mm256_set1_pd(-2.0);
      
      ALIGN32  __m256d const res = _mm256_sub_pd( _mm256_div_pd(two,   _mm256_add_pd( _mm256_set1_pd(1.0),
              nico_exp_1_AVX2(  _mm256_mul_pd(x_aligned, m_two)   ) ) ),  _mm256_set1_pd(1.0) ) ;
      return res;
  
}




ALWAYS_INLINE      __m256d    nico_tanh_wo_checks_AVX2 VECTORCALL(const   __m256d x  )   {

      ALIGN32  __m256d const x_aligned = x;
  
      ALIGN32  __m256d const  two =     _mm256_set1_pd(2.0);
      ALIGN32  __m256d const  m_two =   _mm256_set1_pd(-2.0);
      
      ALIGN32  __m256d const res  =  _mm256_sub_pd( _mm256_div_pd(two,   _mm256_add_pd( _mm256_set1_pd(1.0),
              nico_exp_1_wo_checks_AVX2(  _mm256_mul_pd(x_aligned, m_two)   ) ) ),  _mm256_set1_pd(1.0) ) ;
      return res;
      
}




// from: https://stackoverflow.com/questions/57870896/writing-a-portable-sse-avx-version-of-stdcopysign
ALWAYS_INLINE    __m256d CopySign VECTORCALL(  const __m256d srcSign, 
                                               const __m256d srcValue) {

      ALIGN32  __m256d const srcSign_aligned = srcSign;
      ALIGN32  __m256d const srcValue_aligned = srcValue;
        
      ALIGN32  __m256d const mask0 = _mm256_set1_pd(-0.);
      
      ALIGN32  __m256d const tmp0 = _mm256_and_pd(srcSign_aligned, mask0); // Extract the signed bit from srcSign
      // Extract the number without sign of srcValue (abs(srcValue))
      ALIGN32  __m256d const tmp1 = _mm256_andnot_pd(mask0, srcValue_aligned);
      
      ALIGN32  __m256d const res =  _mm256_or_pd(tmp0, tmp1);  // Merge signed bit with number and return
      return res;
  
}




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
#ifndef BAYESMVP_FAST_PHI_MILLS_COEFFICIENTS_DEFINED
#define BAYESMVP_FAST_PHI_MILLS_COEFFICIENTS_DEFINED
static constexpr double FAST_PHI_MILLS_Z_CAP = 38.5;
static constexpr double FAST_PHI_MILLS_NUMERATOR_COEFFICIENTS[7] = {   //// N6(z) = sum_k c[k] z^k   (includes 1/sqrt(2 pi))
        0.5,
        0.5968126365007661,
        0.34755100079419643,
        0.12199983678527791,
        0.027010741025906854,
        0.0035954377698120227,
        0.0002302422023077616 };
static constexpr double FAST_PHI_MILLS_DENOMINATOR_COEFFICIENTS[8] = {   //// Q7(z) = sum_k d[k] z^k
        1.0,
        1.9915098343653488,
        1.7840969388752417,
        0.9377097714656843,
        0.314821789667328,
        0.06828298351478591,
        0.009012426603260822,
        0.0005771316095331759 };
static_assert(FAST_PHI_MILLS_NUMERATOR_COEFFICIENTS[0] == 0.5,
              "nico_Phi Mills rational: N(0) must be exactly 0.5 so that nico_Phi(0) = 0.5");
static_assert(FAST_PHI_MILLS_DENOMINATOR_COEFFICIENTS[0] == 1.0,
              "nico_Phi Mills rational: Q(0) must be exactly 1");
static_assert(FAST_PHI_MILLS_Z_CAP >= 37.5,
              "nico_Phi Mills rational: the cap must not bind inside nico_Phi's range [-37.5, 8.25]");
#endif




//// Phi(x) = exp(-z^2/2) * N(z) / Q(z) for x <= 0, 1 minus that for x > 0 (z = |x|): see the FAST_PHI_MILLS_* block above.
ALWAYS_INLINE __m256d nico_Phi_wo_checks_AVX2 VECTORCALL(const __m256d x) {

       ALIGN32  __m256d const x_aligned = x;

       ALIGN32  __m256d const z = _mm256_abs_pd(x_aligned); /////  std::fabs(x);
       //// NaN x: min returns its 2nd operand (the cap), and the NaN still propagates through the exp below.
       ALIGN32  __m256d const z_rational = _mm256_min_pd(z, _mm256_set1_pd(FAST_PHI_MILLS_Z_CAP));

       ALIGN32  __m256d numerator = _mm256_set1_pd(FAST_PHI_MILLS_NUMERATOR_COEFFICIENTS[6]);
       numerator = _mm256_fmadd_pd(numerator,
                                   z_rational,
                                   _mm256_set1_pd(FAST_PHI_MILLS_NUMERATOR_COEFFICIENTS[5]));
       numerator = _mm256_fmadd_pd(numerator,
                                   z_rational,
                                   _mm256_set1_pd(FAST_PHI_MILLS_NUMERATOR_COEFFICIENTS[4]));
       numerator = _mm256_fmadd_pd(numerator,
                                   z_rational,
                                   _mm256_set1_pd(FAST_PHI_MILLS_NUMERATOR_COEFFICIENTS[3]));
       numerator = _mm256_fmadd_pd(numerator,
                                   z_rational,
                                   _mm256_set1_pd(FAST_PHI_MILLS_NUMERATOR_COEFFICIENTS[2]));
       numerator = _mm256_fmadd_pd(numerator,
                                   z_rational,
                                   _mm256_set1_pd(FAST_PHI_MILLS_NUMERATOR_COEFFICIENTS[1]));
       numerator = _mm256_fmadd_pd(numerator,
                                   z_rational,
                                   _mm256_set1_pd(FAST_PHI_MILLS_NUMERATOR_COEFFICIENTS[0]));

       ALIGN32  __m256d denominator = _mm256_set1_pd(FAST_PHI_MILLS_DENOMINATOR_COEFFICIENTS[7]);
       denominator = _mm256_fmadd_pd(denominator,
                                     z_rational,
                                     _mm256_set1_pd(FAST_PHI_MILLS_DENOMINATOR_COEFFICIENTS[6]));
       denominator = _mm256_fmadd_pd(denominator,
                                     z_rational,
                                     _mm256_set1_pd(FAST_PHI_MILLS_DENOMINATOR_COEFFICIENTS[5]));
       denominator = _mm256_fmadd_pd(denominator,
                                     z_rational,
                                     _mm256_set1_pd(FAST_PHI_MILLS_DENOMINATOR_COEFFICIENTS[4]));
       denominator = _mm256_fmadd_pd(denominator,
                                     z_rational,
                                     _mm256_set1_pd(FAST_PHI_MILLS_DENOMINATOR_COEFFICIENTS[3]));
       denominator = _mm256_fmadd_pd(denominator,
                                     z_rational,
                                     _mm256_set1_pd(FAST_PHI_MILLS_DENOMINATOR_COEFFICIENTS[2]));
       denominator = _mm256_fmadd_pd(denominator,
                                     z_rational,
                                     _mm256_set1_pd(FAST_PHI_MILLS_DENOMINATOR_COEFFICIENTS[1]));
       denominator = _mm256_fmadd_pd(denominator,
                                     z_rational,
                                     _mm256_set1_pd(FAST_PHI_MILLS_DENOMINATOR_COEFFICIENTS[0]));

       ALIGN32  __m256d const is_x_gr_0 = _mm256_cmp_pd(x_aligned, _mm256_setzero_pd(), _CMP_GT_OQ);

       ALIGN32  __m256d const exp_stuff = nico_exp_1_AVX2(_mm256_mul_pd(_mm256_set1_pd(-0.50), _mm256_mul_pd(z, z)) );
       //// N / Q is formed BEFORE multiplying by the exp so the divide runs in parallel with nico_exp_1's polynomial instead of
       //// after it (measured: (exp * N) / Q put the divide latency on the critical path and cost ~6% on AVX2).
       ALIGN32  __m256d const val = _mm256_mul_pd(exp_stuff, _mm256_div_pd(numerator, denominator));   //// Phi(-z) = exp(-z^2/2) * N(z) / Q(z)
       ALIGN32  __m256d const one_m_res = _mm256_sub_pd(_mm256_set1_pd(1.0), val);

       ALIGN32  __m256d const res = _mm256_blendv_pd(    val,  //// if x < 0
                                                        one_m_res, //// if x > 0
                                                        is_x_gr_0);
       return res;

}




ALWAYS_INLINE __m256d nico_Phi_AVX2 VECTORCALL(const __m256d x) {

      ALIGN32  __m256d const x_aligned = x;
  
      
      ALIGN32  __m256d const Phi_upper_bound =  _mm256_set1_pd(8.25);
      ALIGN32  __m256d const Phi_lower_bound =  _mm256_set1_pd(-37.5);
      
      // equality must not fall through to 1
      ALIGN32  __m256d const is_x_gr_lower = _mm256_cmp_pd(x_aligned, Phi_lower_bound, _CMP_GE_OQ);
      // true where x < 8.25 (in range)
      ALIGN32  __m256d const is_x_lt_upper = _mm256_cmp_pd(x_aligned, Phi_upper_bound, _CMP_LT_OQ);
      ALIGN32  __m256d const is_x_in_range = _mm256_and_pd(is_x_gr_lower, is_x_lt_upper);
      
      // Calculate the main Phi function for in-range values
      ALIGN32  __m256d const val = _mm256_blendv_pd(
                                    _mm256_blendv_pd( _mm256_set1_pd(1.0),
                                                      // if x NOT in range and x NOT < -37.5 i.e. x must be > 8.25
                                                      _mm256_setzero_pd(), // if x NOT in range and x < -37.5
                                                      _mm256_cmp_pd(x_aligned, Phi_lower_bound, _CMP_LT_OQ)),  
                                                      nico_Phi_wo_checks_AVX2(x_aligned), // if x is in range
                                                      is_x_in_range
      );
      
      // Clamp results between 0 and 1
      ALIGN32  __m256d const res = _mm256_min_pd(_mm256_set1_pd(1.0), _mm256_max_pd(_mm256_setzero_pd(), val));
      // Preserve NaN input rather than turning it into a valid probability through min/max.
      return _mm256_blendv_pd(res, x_aligned, _mm256_cmp_pd(x_aligned, x_aligned, _CMP_UNORD_Q));
  
}




///////////////////  fns - inv_Phi functions   --------------------------------------------------------------------------------------------------------------------------------
//// vectorised, AVX-256 version of Stan fn provided by Sean Pinkney:  https://github.com/stan-dev/math/issues/2555 
ALWAYS_INLINE __m256d nico_inv_Phi_wo_checks_case_2a_AVX2 VECTORCALL( const __m256d p,
                                                                      __m256d r) { ///  CASE 2(a): if abs(q) > 0.425  AND   if r <= 5.0 
   
       r = _mm256_add_pd(r, _mm256_set1_pd(-1.60));
       
       ALIGN32 __m256d numerator = _mm256_set1_pd(0.00077454501427834140764);
       numerator = _mm256_fmadd_pd(r, numerator, _mm256_set1_pd(0.0227238449892691845833));
       numerator = _mm256_fmadd_pd(r, numerator, _mm256_set1_pd(0.24178072517745061177));
       numerator = _mm256_fmadd_pd(r, numerator, _mm256_set1_pd(1.27045825245236838258));
       numerator = _mm256_fmadd_pd(r, numerator, _mm256_set1_pd(3.64784832476320460504));
       numerator = _mm256_fmadd_pd(r, numerator, _mm256_set1_pd(5.7694972214606914055));
       numerator = _mm256_fmadd_pd(r, numerator, _mm256_set1_pd(4.6303378461565452959));
       numerator = _mm256_fmadd_pd(r, numerator, _mm256_set1_pd(1.42343711074968357734)); 
       
       ALIGN32 __m256d  denominator = _mm256_set1_pd(0.00000000105075007164441684324);
       denominator = _mm256_fmadd_pd(r,  denominator, _mm256_set1_pd(0.0005475938084995344946));
       denominator = _mm256_fmadd_pd(r,  denominator, _mm256_set1_pd(0.0151986665636164571966));
       denominator = _mm256_fmadd_pd(r,  denominator, _mm256_set1_pd(0.14810397642748007459));
       denominator = _mm256_fmadd_pd(r,  denominator, _mm256_set1_pd(0.68976733498510000455));
       denominator = _mm256_fmadd_pd(r,  denominator, _mm256_set1_pd(1.6763848301838038494));
       denominator = _mm256_fmadd_pd(r,  denominator, _mm256_set1_pd(2.05319162663775882187));
       denominator = _mm256_fmadd_pd(r, denominator,  _mm256_set1_pd(1.0));
       
       ALIGN32  __m256d const val = _mm256_div_pd(numerator, denominator);
       return val;

}  




ALWAYS_INLINE __m256d nico_inv_Phi_wo_checks_case_2b_AVX2 VECTORCALL( const __m256d p,
                                                                      __m256d r) { ///  CASE 2(a): if abs(q) > 0.425  AND   if r > 5.0 
        
      ALIGN32  __m256d const one =   _mm256_set1_pd(1.0);
      
      r = _mm256_add_pd(r, _mm256_set1_pd(-5.0));
      
      ALIGN32 __m256d numerator =  _mm256_set1_pd(0.000000201033439929228813265);
      numerator = _mm256_fmadd_pd(r, numerator,  _mm256_set1_pd(0.0000271155556874348757815));
      numerator = _mm256_fmadd_pd(r, numerator,  _mm256_set1_pd(0.0012426609473880784386));
      numerator = _mm256_fmadd_pd(r, numerator,  _mm256_set1_pd(0.026532189526576123093));
      numerator = _mm256_fmadd_pd(r, numerator,  _mm256_set1_pd(0.29656057182850489123));
      numerator = _mm256_fmadd_pd(r, numerator,  _mm256_set1_pd(1.7848265399172913358));
      numerator = _mm256_fmadd_pd(r, numerator,  _mm256_set1_pd(5.4637849111641143699));
      numerator = _mm256_fmadd_pd(r, numerator,  _mm256_set1_pd(6.6579046435011037772));
      
      ALIGN32 __m256d denominator =  _mm256_set1_pd(0.00000000000000204426310338993978564);
      denominator = _mm256_fmadd_pd(r, denominator,  _mm256_set1_pd(0.00000014215117583164458887));
      denominator = _mm256_fmadd_pd(r, denominator,  _mm256_set1_pd(0.000018463183175100546818));
      denominator = _mm256_fmadd_pd(r, denominator,  _mm256_set1_pd(0.0007868691311456132591));
      denominator = _mm256_fmadd_pd(r, denominator,  _mm256_set1_pd(0.0148753612908506148525));
      denominator = _mm256_fmadd_pd(r, denominator,  _mm256_set1_pd(0.13692988092273580531)); 
      denominator = _mm256_fmadd_pd(r, denominator,  _mm256_set1_pd(0.59983220655588793769));
      denominator = _mm256_fmadd_pd(r, denominator,  one);
      
      ALIGN32  __m256d const val = _mm256_div_pd(numerator, denominator);
      return val;

} 




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
ALWAYS_INLINE __m256d nico_inv_Phi_wo_checks_case_2_AVX2 VECTORCALL( const __m256d p,
                                                                     const __m256d q) {

      ALIGN32  __m256d const p_aligned = p;
      ALIGN32  __m256d const q_aligned = q;
  
      ALIGN32 __m256d const zero = _mm256_setzero_pd();
      ALIGN32 __m256d const one  = _mm256_set1_pd(1.0);
      ALIGN32 __m256d const five = _mm256_set1_pd(5.0);

      ALIGN32 __m256d const q_pos = _mm256_cmp_pd(q_aligned, zero, _CMP_GT_OQ);
      // q>0 -> 1-p, else p
      ALIGN32 __m256d r = _mm256_blendv_pd(p_aligned, _mm256_sub_pd(one, p_aligned), q_pos);
      r = _mm256_sqrt_pd(_mm256_sub_pd(zero, nico_log_1_AVX2(r)));

      ALIGN32 __m256d const r_gr_5 = _mm256_cmp_pd(r, five, _CMP_GT_OQ);
      ALIGN32 __m256d val = nico_inv_Phi_wo_checks_case_2a_AVX2(p_aligned, r);                 // r <= 5
      // No need to evaluate the extreme-tail polynomial unless a lane will actually use it.
      if (_mm256_movemask_pd(r_gr_5) != 0) val = _mm256_blendv_pd(val, nico_inv_Phi_wo_checks_case_2b_AVX2(
              p_aligned, r), r_gr_5);

      ALIGN32 __m256d const q_neg = _mm256_cmp_pd(q_aligned, zero, _CMP_LT_OQ);
      return _mm256_blendv_pd(val, _mm256_sub_pd(zero, val), q_neg);                    // q<0 -> -val
      
}




ALWAYS_INLINE __m256d nico_inv_Phi_wo_checks_case_1_AVX2 VECTORCALL( const __m256d p,
                                                                     const __m256d q) {

      ALIGN32  __m256d const q_aligned = q; ///  CASE 1: if abs(q) <= 0.425
   
      ALIGN32  __m256d const q_sq = _mm256_mul_pd(q_aligned, q_aligned);
      ALIGN32  __m256d const r = _mm256_sub_pd(_mm256_set1_pd(0.180625), q_sq);
      
      ALIGN32 __m256d numerator = _mm256_fmadd_pd(r,
                                                  _mm256_set1_pd(2509.0809287301226727),
                                                  _mm256_set1_pd(33430.575583588128105));
      numerator = _mm256_fmadd_pd(numerator, r, _mm256_set1_pd(67265.770927008700853));
      numerator = _mm256_fmadd_pd(numerator, r, _mm256_set1_pd(45921.953931549871457));
      numerator = _mm256_fmadd_pd(numerator, r, _mm256_set1_pd(13731.693765509461125));
      numerator = _mm256_fmadd_pd(numerator, r, _mm256_set1_pd(1971.5909503065514427)); 
      numerator = _mm256_fmadd_pd(numerator, r, _mm256_set1_pd(133.14166789178437745));
      numerator = _mm256_fmadd_pd(numerator, r, _mm256_set1_pd(3.387132872796366608));
      
      ALIGN32 __m256d denominator = _mm256_fmadd_pd(r,
                                                    _mm256_set1_pd(5226.495278852854561),
                                                    _mm256_set1_pd(28729.085735721942674));
      denominator = _mm256_fmadd_pd(denominator, r, _mm256_set1_pd(39307.89580009271061));
      denominator = _mm256_fmadd_pd(denominator, r, _mm256_set1_pd(21213.794301586595867));
      denominator = _mm256_fmadd_pd(denominator, r, _mm256_set1_pd(5394.1960214247511077));
      denominator = _mm256_fmadd_pd(denominator, r, _mm256_set1_pd(687.1870074920579083));
      denominator = _mm256_fmadd_pd(denominator, r, _mm256_set1_pd(42.313330701600911252)); 
      denominator = _mm256_fmadd_pd(denominator, r,  _mm256_set1_pd(1.0));
      
      ALIGN32 __m256d val = _mm256_div_pd(numerator, denominator); 
      val = _mm256_mul_pd(q_aligned, val);
      return val;

}   




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
ALWAYS_INLINE __m256d nico_inv_Phi_wo_checks_AVX2 VECTORCALL(const __m256d p) {

      ALIGN32  __m256d const p_aligned = p;
  
      ALIGN32 __m256d const q       = _mm256_sub_pd(p_aligned, _mm256_set1_pd(0.5));
      ALIGN32 __m256d const central = _mm256_cmp_pd(_mm256_abs_pd(q), _mm256_set1_pd(0.425), _CMP_LE_OQ);
      ALIGN32 __m256d const v1 = nico_inv_Phi_wo_checks_case_1_AVX2(p_aligned, q);
      // The tail result would be discarded in every lane. Avoid its log and two rational polynomials.
      if (_mm256_movemask_pd(central) == 0xf) return v1;
      ALIGN32 __m256d const v2 = nico_inv_Phi_wo_checks_case_2_AVX2(p_aligned, q);
      return _mm256_blendv_pd(v2, v1, central);   // central lanes -> v1
      
}




// =============================================================================================
//  AVX2   (identical logic; blendv(a, b, m) == mask_blend(m, a, b), so argument order is unchanged)
// =============================================================================================
ALWAYS_INLINE __m256d nico_mills_ratio_tail_AVX2 VECTORCALL(const __m256d x) {

      ALIGN32  __m256d const x_aligned = x;
      ALIGN32  __m256d f = x_aligned;
      for (int k = MILLS_CF_DEPTH; k >= 1; --k) {
          f = _mm256_add_pd(x_aligned, _mm256_div_pd(_mm256_set1_pd(static_cast<double>(k)), f));
      }
      return _mm256_div_pd(_mm256_set1_pd(1.0), f);

}




//// ---- Mills ratio for the lower tail of log_Phi / dlog_Phi_dx, z >= NORMAL_TAIL_THRESH (2.5): R(z) = Phi(-z) /
////      phi(z) = P(t) / (Q(t) z),  t = 1/z^2,  P and Q of degree 8 with P(0) = Q(0) = 1 (the exact 1/z
////      asymptote; t -> 0 without overflow for any z), fitted to the relative error of R on z in [2.5, inf)
////      (relative-weight least squares iterated to near-minimax, in 70-digit mpmath, in u = t / 0.16: fit error
////      2.1e-18). Evaluated as 1/R = z Q / P = z (1 - w), w = D(t) / P(t), D = P - Q: 1/R = z - z w is one FMA
////      and |w| <= 0.16, so the rounding of w hardly reaches 1/R. nico_log_Phi takes log of R = 1 / (1/R) (the
////      same argument the continued fraction gave nico_log_1). Two or three divisions per vector instead of 41
////      for the 40-term continued fraction (still defined above as nico_mills_ratio_tail_*, now unused),
////      1.35e-12 at z = 2.5.
#ifndef BAYESMVP_LOG_PHI_TAIL_MILLS_COEFFICIENTS_DEFINED
#define BAYESMVP_LOG_PHI_TAIL_MILLS_COEFFICIENTS_DEFINED
static constexpr double LOG_PHI_TAIL_MILLS_NUMERATOR_COEFFICIENTS[9] = {     //// P(t) = sum_k c[k] t^k
        1.0,
        71.1021192876737,
        1886.6541015845198,
        23805.12629965518,
        150884.97847730128,
        466855.69935666997,
        630792.3515363043,
        284804.99468690663,
        18191.359378269553 };
static constexpr double LOG_PHI_TAIL_MILLS_NUMERATOR_MINUS_DENOMINATOR_COEFFICIENTS[9] = {   //// D = P - Q
        0.0,
        -1.0,
        -69.10211928769195,
        -1754.4498629989466,
        -20668.839292316028,
        -117585.7095446167,
        -305560.76122112013,
        -306398.0298078199,
        -76229.08437753914 };
static constexpr double LOG_PHI_TAIL_MILLS_DENOMINATOR_COEFFICIENTS[9] = {   //// Q(t) = sum_k d[k] t^k
        1.0,
        72.1021192876737,
        1955.7562208722118,
        25559.576162654128,
        171553.8177696173,
        584441.4089012867,
        936353.1127574245,
        591203.0244947266,
        94420.4437558087 };
#endif

// 1/R(z) = z (1 - w), w = D(t) / P(t), t = 1/z^2, for the tail Mills ratio R (see the coefficient block above);
// z = inf gives inf (w = 0 there, and inf * 0 would give NaN)
ALWAYS_INLINE __m256d nico_inverse_mills_ratio_tail_AVX2(const __m256d z) {

        ALIGN32  __m256d const z_aligned = z;

        ALIGN32  __m256d const t = _mm256_div_pd(_mm256_set1_pd(1.0), _mm256_mul_pd(z_aligned, z_aligned));
        ALIGN32  __m256d d = _mm256_set1_pd(LOG_PHI_TAIL_MILLS_NUMERATOR_MINUS_DENOMINATOR_COEFFICIENTS[8]);
        ALIGN32  __m256d p = _mm256_set1_pd(LOG_PHI_TAIL_MILLS_NUMERATOR_COEFFICIENTS[8]);
        d = _mm256_fmadd_pd(d, t, _mm256_set1_pd(LOG_PHI_TAIL_MILLS_NUMERATOR_MINUS_DENOMINATOR_COEFFICIENTS[7]));
        p = _mm256_fmadd_pd(p, t, _mm256_set1_pd(LOG_PHI_TAIL_MILLS_NUMERATOR_COEFFICIENTS[7]));
        d = _mm256_fmadd_pd(d, t, _mm256_set1_pd(LOG_PHI_TAIL_MILLS_NUMERATOR_MINUS_DENOMINATOR_COEFFICIENTS[6]));
        p = _mm256_fmadd_pd(p, t, _mm256_set1_pd(LOG_PHI_TAIL_MILLS_NUMERATOR_COEFFICIENTS[6]));
        d = _mm256_fmadd_pd(d, t, _mm256_set1_pd(LOG_PHI_TAIL_MILLS_NUMERATOR_MINUS_DENOMINATOR_COEFFICIENTS[5]));
        p = _mm256_fmadd_pd(p, t, _mm256_set1_pd(LOG_PHI_TAIL_MILLS_NUMERATOR_COEFFICIENTS[5]));
        d = _mm256_fmadd_pd(d, t, _mm256_set1_pd(LOG_PHI_TAIL_MILLS_NUMERATOR_MINUS_DENOMINATOR_COEFFICIENTS[4]));
        p = _mm256_fmadd_pd(p, t, _mm256_set1_pd(LOG_PHI_TAIL_MILLS_NUMERATOR_COEFFICIENTS[4]));
        d = _mm256_fmadd_pd(d, t, _mm256_set1_pd(LOG_PHI_TAIL_MILLS_NUMERATOR_MINUS_DENOMINATOR_COEFFICIENTS[3]));
        p = _mm256_fmadd_pd(p, t, _mm256_set1_pd(LOG_PHI_TAIL_MILLS_NUMERATOR_COEFFICIENTS[3]));
        d = _mm256_fmadd_pd(d, t, _mm256_set1_pd(LOG_PHI_TAIL_MILLS_NUMERATOR_MINUS_DENOMINATOR_COEFFICIENTS[2]));
        p = _mm256_fmadd_pd(p, t, _mm256_set1_pd(LOG_PHI_TAIL_MILLS_NUMERATOR_COEFFICIENTS[2]));
        d = _mm256_fmadd_pd(d, t, _mm256_set1_pd(LOG_PHI_TAIL_MILLS_NUMERATOR_MINUS_DENOMINATOR_COEFFICIENTS[1]));
        p = _mm256_fmadd_pd(p, t, _mm256_set1_pd(LOG_PHI_TAIL_MILLS_NUMERATOR_COEFFICIENTS[1]));
        d = _mm256_mul_pd(d, t);                                                            // D(t), D(0) = 0
        p = _mm256_fmadd_pd(p, t, _mm256_set1_pd(LOG_PHI_TAIL_MILLS_NUMERATOR_COEFFICIENTS[0]));     // P(t)
        ALIGN32  __m256d const z_minus_z_w = _mm256_fnmadd_pd(z_aligned, _mm256_div_pd(d, p), z_aligned);
        return _mm256_blendv_pd(z_minus_z_w, z_aligned, _mm256_cmp_pd(z_aligned, _mm256_set1_pd(INFINITY),
                _CMP_EQ_OQ));

}




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
ALWAYS_INLINE __m256d nico_log_Phi_AVX2 VECTORCALL(const __m256d x) {

      ALIGN32  __m256d const x_aligned = x;
      ALIGN32  __m256d const thresh   = _mm256_set1_pd(NORMAL_TAIL_THRESH);
      // x < -2.5
      ALIGN32  __m256d const is_tail  = _mm256_cmp_pd(x_aligned, _mm256_sub_pd(_mm256_setzero_pd(), thresh),
              _CMP_LT_OQ);
      const int     tail_lanes = _mm256_movemask_pd(is_tail);
      // no tail lane: body only
      if (tail_lanes == 0) return nico_log_1_AVX2(nico_Phi_AVX2(x_aligned));
      ALIGN32  __m256d const ax       = _mm256_abs_pd(x_aligned);
      ALIGN32  __m256d const R        = _mm256_div_pd(_mm256_set1_pd(1.0),
                                             nico_inverse_mills_ratio_tail_AVX2(_mm256_max_pd(ax, thresh)));
      ALIGN32  __m256d const log_tail = _mm256_add_pd(_mm256_fnmadd_pd(_mm256_set1_pd(0.5), _mm256_mul_pd(
              x_aligned, x_aligned),
                                                              _mm256_set1_pd(NEG_HALF_LOG_2PI)),
                                             nico_log_1_AVX2(R));
      if (tail_lanes == 0xf) return log_tail;                                           // every lane in the tail
      ALIGN32  __m256d const log_body = nico_log_1_AVX2(nico_Phi_AVX2(x_aligned));
      return _mm256_blendv_pd(log_body, log_tail, is_tail);

}




ALWAYS_INLINE __m256d nico_log_1m_Phi_AVX2 VECTORCALL(const __m256d x) {

      ALIGN32  __m256d const x_aligned = x;
      return nico_log_Phi_AVX2(_mm256_sub_pd(_mm256_setzero_pd(), x_aligned));

}




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
ALWAYS_INLINE __m256d nico_dlog_Phi_dx_AVX2 VECTORCALL(const __m256d x) {

      ALIGN32  __m256d const x_aligned = x;
      ALIGN32  __m256d const thresh  = _mm256_set1_pd(NORMAL_TAIL_THRESH);
      // x < -2.5
      ALIGN32  __m256d const is_tail = _mm256_cmp_pd(x_aligned, _mm256_sub_pd(_mm256_setzero_pd(), thresh),
              _CMP_LT_OQ);
      const int     tail_lanes = _mm256_movemask_pd(is_tail);
      if (tail_lanes == 0) {                                                            // no tail lane: body only
          ALIGN32  __m256d const phi = _mm256_mul_pd(_mm256_set1_pd(INV_SQRT_2PI),
                                            nico_exp_1_AVX2(_mm256_mul_pd(_mm256_set1_pd(-0.5),
                                                                          _mm256_mul_pd(x_aligned, x_aligned))));
          return _mm256_div_pd(phi, nico_Phi_AVX2(x_aligned));
      }
      ALIGN32  __m256d const ax      = _mm256_abs_pd(x_aligned);
      ALIGN32  __m256d const d_tail  = nico_inverse_mills_ratio_tail_AVX2(_mm256_max_pd(ax, thresh));
      if (tail_lanes == 0xf) return d_tail;                                             // every lane in the tail
      ALIGN32  __m256d const phi     = _mm256_mul_pd(_mm256_set1_pd(INV_SQRT_2PI),
                                            nico_exp_1_AVX2(_mm256_mul_pd(_mm256_set1_pd(-0.5), _mm256_mul_pd(
                                                    x_aligned, x_aligned))));
      ALIGN32  __m256d const d_body  = _mm256_div_pd(phi, nico_Phi_AVX2(x_aligned));
      return _mm256_blendv_pd(d_body, d_tail, is_tail);

}




ALWAYS_INLINE __m256d nico_inv_Phi_from_log_p_wo_checks_AVX2 VECTORCALL( const __m256d log_p, 
                                                                         const __m256d log_1m_p) {

      ALIGN32  __m256d const log_p_aligned = log_p;
      ALIGN32  __m256d const log_1m_p_aligned = log_1m_p;
      ALIGN32  __m256d const zero    = _mm256_setzero_pd();
      ALIGN32  __m256d const p       = nico_exp_1_AVX2(log_p_aligned);
      ALIGN32  __m256d const q       = _mm256_sub_pd(p, _mm256_set1_pd(0.5));
      ALIGN32  __m256d const central = _mm256_cmp_pd(_mm256_abs_pd(q), _mm256_set1_pd(0.425), _CMP_LE_OQ);
      ALIGN32  __m256d const log_min = _mm256_min_pd(log_p_aligned, log_1m_p_aligned);
      ALIGN32  __m256d const r       = _mm256_sqrt_pd(_mm256_sub_pd(zero, log_min));
      ALIGN32  __m256d const r_gr_5  = _mm256_cmp_pd(r, _mm256_set1_pd(5.0), _CMP_GT_OQ);
      ALIGN32  __m256d v_tail = _mm256_blendv_pd(nico_inv_Phi_wo_checks_case_2a_AVX2(p, r),
                                        nico_inv_Phi_wo_checks_case_2b_AVX2(p, r),
                                        r_gr_5);
      ALIGN32  __m256d const lower   = _mm256_cmp_pd(log_p_aligned, log_1m_p_aligned, _CMP_LT_OQ);
      v_tail = _mm256_blendv_pd(v_tail, _mm256_sub_pd(zero, v_tail), lower);
      ALIGN32  __m256d const v_central = nico_inv_Phi_wo_checks_case_1_AVX2(p, q);
      return _mm256_blendv_pd(v_tail, v_central, central);

}




ALWAYS_INLINE __m256d nico_dZ_dv_from_log_p_AVX2 VECTORCALL( const __m256d Z, 
                                                             const __m256d log_p, 
                                                             const __m256d log_1m_p,
                                                             const __m256d dlog_p_dv, 
                                                             const __m256d dlog_1m_p_dv) {

      ALIGN32  __m256d const Z_aligned = Z;
      ALIGN32  __m256d const log_p_aligned = log_p;
      ALIGN32  __m256d const log_1m_p_aligned = log_1m_p;
      ALIGN32  __m256d const dlog_p_dv_aligned = dlog_p_dv;
      ALIGN32  __m256d const dlog_1m_p_dv_aligned = dlog_1m_p_dv;
  
      ALIGN32  __m256d const log_inv_phi =
              _mm256_fmadd_pd(_mm256_set1_pd(0.5), _mm256_mul_pd(Z_aligned, Z_aligned), _mm256_set1_pd(
                      -NEG_HALF_LOG_2PI));
      ALIGN32  __m256d const cap    = _mm256_set1_pd(700.0);
      ALIGN32  __m256d const e_lo   = _mm256_min_pd(_mm256_add_pd(log_p_aligned,    log_inv_phi), cap);
      ALIGN32  __m256d const e_hi   = _mm256_min_pd(_mm256_add_pd(log_1m_p_aligned, log_inv_phi), cap);
      ALIGN32  __m256d const via_lo = _mm256_mul_pd(nico_exp_1_AVX2(e_lo), dlog_p_dv_aligned);
      ALIGN32  __m256d const via_hi =
              _mm256_mul_pd(_mm256_sub_pd(_mm256_setzero_pd(), nico_exp_1_AVX2(e_hi)), dlog_1m_p_dv_aligned);
      ALIGN32  __m256d const Z_neg  = _mm256_cmp_pd(Z_aligned, _mm256_setzero_pd(), _CMP_LT_OQ);
      return _mm256_blendv_pd(via_hi, via_lo, Z_neg);
      
}




}  // namespace nico::math

using nico::math::test_simple_AVX2;
using nico::math::_mm256_abs_pd;
using nico::math::is_finite_mask;
using nico::math::is_not_NaN_mask;
using nico::math::nico_ldexp;
using nico::math::avx2_cvtpd_epi64;
using nico::math::nico_exp_1_wo_checks_AVX2;
using nico::math::nico_exp_exceptional_lanes_AVX2;
using nico::math::nico_exp_1_AVX2;
using nico::math::avx2_cvtepi64_pd;
using nico::math::nico_log_1_wo_checks_AVX2;
using nico::math::nico_log_reduce_AVX2;
using nico::math::nico_log_fdlibm_core_AVX2;
using nico::math::nico_log_1_double_precision_wo_checks_AVX2;
using nico::math::nico_log1p_1_wo_checks_AVX2;
using nico::math::nico_log1p_1_double_precision_wo_checks_AVX2;
using nico::math::nico_log1m_1_wo_checks_AVX2;
using nico::math::nico_log1p_exp_1_wo_checks_AVX2;
using nico::math::nico_log_exceptional_lanes_AVX2;
using nico::math::nico_log_1_AVX2;
using nico::math::nico_log_1_double_precision_AVX2;
using nico::math::nico_log1p_1_AVX2;
using nico::math::nico_log1p_exceptional_lanes_AVX2;
using nico::math::nico_log1p_1_double_precision_AVX2;
using nico::math::nico_log1m_1_AVX2;
using nico::math::nico_log1p_exp_1_AVX2;
using nico::math::nico_log1m_exp_1_AVX2;
using nico::math::nico_logit_wo_checks_AVX2;
using nico::math::nico_logit_AVX2;
using nico::math::nico_inv_logit_for_x_pos_AVX2;
using nico::math::nico_inv_logit_for_x_neg_AVX2;
using nico::math::nico_inv_logit_AVX2;
using nico::math::nico_inv_logit_for_x_pos_wo_checks_AVX2;
using nico::math::nico_inv_logit_for_x_neg_wo_checks_AVX2;
using nico::math::nico_inv_logit_wo_checks_AVX2;
using nico::math::nico_log_inv_logit_for_x_pos_AVX2;
using nico::math::nico_log_inv_logit_for_x_neg_AVX2;
using nico::math::nico_log_inv_logit_AVX2;
using nico::math::nico_log_inv_logit_for_x_pos_wo_checks_AVX2;
using nico::math::nico_log_inv_logit_for_x_neg_wo_checks_AVX2;
using nico::math::nico_log_inv_logit_wo_checks_AVX2;
using nico::math::nico_Phi_approx_wo_checks_AVX2;
using nico::math::nico_Phi_approx_AVX2;
using nico::math::nico_inv_Phi_approx_wo_checks_AVX2;
using nico::math::nico_inv_Phi_approx_AVX2;
using nico::math::nico_inv_Phi_approx_from_logit_prob_wo_checks_AVX2;
using nico::math::nico_inv_Phi_approx_from_logit_prob_AVX2;
using nico::math::nico_log_Phi_approx_wo_checks_AVX2;
using nico::math::nico_log_Phi_approx_AVX2;
using nico::math::nico_tanh_AVX2;
using nico::math::nico_tanh_wo_checks_AVX2;
using nico::math::CopySign;
using nico::math::nico_Phi_wo_checks_AVX2;
using nico::math::nico_Phi_AVX2;
using nico::math::nico_inv_Phi_wo_checks_case_2a_AVX2;
using nico::math::nico_inv_Phi_wo_checks_case_2b_AVX2;
using nico::math::nico_inv_Phi_wo_checks_case_2_AVX2;
using nico::math::nico_inv_Phi_wo_checks_case_1_AVX2;
using nico::math::nico_inv_Phi_wo_checks_AVX2;
using nico::math::nico_mills_ratio_tail_AVX2;
using nico::math::nico_inverse_mills_ratio_tail_AVX2;
using nico::math::nico_log_Phi_AVX2;
using nico::math::nico_log_1m_Phi_AVX2;
using nico::math::nico_dlog_Phi_dx_AVX2;
using nico::math::nico_inv_Phi_from_log_p_wo_checks_AVX2;
using nico::math::nico_dZ_dv_from_log_p_AVX2;

#endif




#endif
 
 
 
 
 

















