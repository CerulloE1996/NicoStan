
#pragma once 


#ifndef NICO_AND_APPROX_AVX512_FNS_HPP
#define NICO_AND_APPROX_AVX512_FNS_HPP


#include "../nico_simd_intrinsics.hpp"   //// <immintrin.h> on x86-64; SIMDe (NEON) on ARM
#include <cmath>
#include "nico_exp_over_the_full_double_range.hpp"
#include "nico_normal_constants.hpp"


#if defined(__AVX512VL__) && defined(__AVX512F__)  && defined(__AVX512DQ__)  /// if have AVX-512

namespace nico::math {
 
//// -------------------------------------------------------------------------------------------------------------------------------------------------------------
 

// Simple test function that just multiplies vector by 2
inline __m512d  test_simple_AVX512 VECTORCALL(const __m512d x) {

       ALIGN64  __m512d const x_aligned = x;
     
       ALIGN64  __m512d const two = _mm512_set1_pd(2.0);
       ALIGN64  __m512d const res = _mm512_mul_pd(x_aligned, two);
       return res; 
 
}








// // =============================================================================================
// //  SELF-TEST  - call once from an Rcpp export or a throwaway main(); prints max errors.
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
//       //// the SIMD kernels must agree with the scalar ones to ~1e-12 - check a few lanes:
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
ALWAYS_INLINE    __mmask8 is_NaN_mask(const __m512d x) {

       ALIGN64  __m512d const x_aligned = x;
       
       // comparing value to itself checks for  NaN. is NaN then this = FALSE
       const __mmask8 mask_is_NaN =_mm512_cmp_pd_mask(x_aligned, x_aligned, _CMP_NEQ_OQ);
       //  __mmask8 mask_is_NaN = f(mask_is_not_NaN); // there is no "NOT" or "complement" mask  !!!!
       
       return  mask_is_NaN;
 
}



ALWAYS_INLINE    __mmask8 is_not_NaN_mask(const __m512d x)  {

       ALIGN64  __m512d const x_aligned = x;
 
       // comparing value to itself checks for  NaN. is NaN then this = FALSE
       const __mmask8 mask_is_not_NaN =_mm512_cmp_pd_mask(x_aligned, x_aligned, _CMP_EQ_OQ);
       //  __mmask8 mask_is_NaN = f(mask_is_not_NaN); // there is no "NOT" or "complement" mask  !!!!
       
       return  mask_is_not_NaN;
 
}



// adapted from: https://stackoverflow.com/questions/30674291/how-to-check-inf-for-avx-intrinsic-m256
ALWAYS_INLINE    __mmask8 is_finite_mask(const __m512d x) {

        ALIGN64  __m512d const x_aligned = x;
  
        ALIGN64  __m512d const INF = _mm512_set1_pd(std::numeric_limits<double>::infinity());
        ALIGN64  __m512d const sign_bit = _mm512_set1_pd(-0.0);
       
        //x = _mm512_andnot_pd(sign_bit, x); // x  = NOT -0.0 AND x
        return _mm512_cmp_pd_mask(_mm512_andnot_pd(sign_bit, x_aligned), INF, _CMP_NEQ_OQ);
 
}



// adapted from: https://stackoverflow.com/questions/30674291/how-to-check-inf-for-avx-intrinsic-m256
ALWAYS_INLINE    __mmask8 is_infinity_mask(const __m512d x) {

        ALIGN64  __m512d const x_aligned = x;
  
        ALIGN64  __m512d const INF = _mm512_set1_pd(std::numeric_limits<double>::infinity());
        ALIGN64  __m512d const sign_bit = _mm512_set1_pd(-0.0);
       
        // x = _mm512_andnot_pd(sign_bit, x);
        return _mm512_cmp_pd_mask(_mm512_andnot_pd(sign_bit, x_aligned), INF, _CMP_EQ_OQ);
 
}




ALWAYS_INLINE    __mmask8 is_positive_mask(const __m512d x) {

        ALIGN64  __m512d const x_aligned = x;
 
        return _mm512_cmp_pd_mask(_mm512_setzero_pd(), x_aligned, _CMP_LT_OQ);
 
}



ALWAYS_INLINE    __mmask8 is_negative_mask(const __m512d x) {

        ALIGN64  __m512d const x_aligned = x;
 
        return _mm512_cmp_pd_mask(_mm512_setzero_pd(), x_aligned, _CMP_GT_OQ);
 
}




///////////////////  fns - exp   -----------------------------------------------------------------------------------------------------------------------------
ALWAYS_INLINE    __m512d nico_ldexp( const __m512d AVX_a,
                                     const __m512i AVX_i) {

       ALIGN64  __m512d const AVX_a_aligned = AVX_a;
       ALIGN64  __m512i const AVX_i_aligned = AVX_i;
  
       /* AVX_a = p * 2^AVX_i */
       return _mm512_castsi512_pd (_mm512_add_epi64 (_mm512_slli_epi64 (AVX_i_aligned, 52ULL),
               _mm512_castpd_si512 (AVX_a_aligned)));
   
}




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
ALWAYS_INLINE    __m512d nico_exp_1_wo_checks_AVX512(const __m512d x)  {

        ALIGN64  __m512d const x_aligned = x;
  
        ALIGN64  __m512d const exp_l2e = _mm512_set1_pd (1.442695040888963387); /* log2(e) */
        /* -log(2)_hi, exactly representable */
        ALIGN64  __m512d const exp_l2h = _mm512_set1_pd (-0.693145751953125);
        ALIGN64  __m512d const exp_l2l = _mm512_set1_pd (-1.42860682030941723212e-6); /* -log(2)_lo */
        
        // /* coefficients for core approximation to exp() in [-log(2)/2, log(2)/2] */
        ALIGN64  __m512d const exp_c0 =     _mm512_set1_pd(0.00000276479776161191821278);
        ALIGN64  __m512d const exp_c1 =     _mm512_set1_pd(0.0000248844480527491290235);
        ALIGN64  __m512d const exp_c2 =     _mm512_set1_pd(0.000198411488032534342194);
        ALIGN64  __m512d const exp_c3 =     _mm512_set1_pd(0.00138888017711994078175);
        ALIGN64  __m512d const exp_c4 =     _mm512_set1_pd(0.00833333340524595143906);
        ALIGN64  __m512d const exp_c5 =     _mm512_set1_pd(0.0416666670404215802592);
        ALIGN64  __m512d const exp_c6 =     _mm512_set1_pd(0.166666666664891632843);
        ALIGN64  __m512d const exp_c7 =     _mm512_set1_pd(0.499999999994389376923);
        ALIGN64  __m512d const exp_c8 =     _mm512_set1_pd(1.00000000000001221245);
        ALIGN64  __m512d const exp_c9 =     _mm512_set1_pd(1.00000000000001332268);
    
        ALIGN64  __m512d const input  = x_aligned;
    
        /* exp(x) = 2^i * e^f; i = rint (log2(e) * a), f = a - log(2) * i */
        ALIGN64  __m512d const t = _mm512_mul_pd(x_aligned, exp_l2e);      /* t = log2(e) * a */
        ALIGN64  __m512i const i = _mm512_cvtpd_epi64(t);       /* i = (int)rint(t) */
        // ((0<<4)| _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC|_MM_FROUND_NO_EXC));
        ALIGN64  __m512d const x_2 = _mm512_roundscale_pd(t, _MM_FROUND_TO_NEAREST_INT) ;
        /* a - log(2)_hi * r */    /* f = a - log(2)_hi * r - log(2)_lo * r */
        ALIGN64  __m512d const f = _mm512_fmadd_pd(x_2, exp_l2l, _mm512_fmadd_pd (x_2, exp_l2h, input));
        
        /* p ~= exp (f), -log(2)/2 <= f <= log(2)/2 */
        // polynomial approximation
        ALIGN64  __m512d p = exp_c0;
        p = _mm512_fmadd_pd(p, f, exp_c1);
        p = _mm512_fmadd_pd(p, f, exp_c2);
        p = _mm512_fmadd_pd(p, f, exp_c3);
        p = _mm512_fmadd_pd(p, f, exp_c4);
        p = _mm512_fmadd_pd(p, f, exp_c5);
        p = _mm512_fmadd_pd(p, f, exp_c6);
        p = _mm512_fmadd_pd(p, f, exp_c7);
        p = _mm512_fmadd_pd(p, f, exp_c8);
        p = _mm512_fmadd_pd(p, f, exp_c9);
        
        return  nico_ldexp(p, i) ;    /* exp(x) = 2^i * p */
      
}




//// see https://stackoverflow.com/questions/39587752/difference-between-ldexp1-x-and-exp2x
// Keep exceptional-lane libm calls out of the inlined SIMD hot path.
__attribute__((noinline, cold)) inline __m512d nico_exp_exceptional_lanes_AVX512(
        const __m512d input, const __m512d result, const unsigned ordinary_mask) {

    return _mm512_mask_mov_pd(nico_exp_over_the_full_double_range_AVX512(input),
                              static_cast<__mmask8>(ordinary_mask), result);

}

ALWAYS_INLINE    __m512d nico_exp_1_AVX512(const __m512d a) {

        ALIGN64  __m512d const a_aligned = a;

        // 708.4 bounds the exponent-bit shortcut, NOT the domain of finite exp().
        // Preserve the ordinary SIMD path; only exceptional lanes need libm's full exponent range.
        __mmask8 ordinary_lanes = _mm512_cmp_pd_mask(_mm512_abs_pd(a_aligned), _mm512_set1_pd(708.4),
                _CMP_LT_OQ);
        ALIGN64  __m512d const result = nico_exp_1_wo_checks_AVX512(a_aligned);
        ordinary_lanes &= _mm512_cmp_pd_mask(result,
            _mm512_set1_pd(std::numeric_limits<double>::min()), _CMP_GE_OQ);
        if (__builtin_expect(ordinary_lanes != 0xff, 0)) return nico_exp_exceptional_lanes_AVX512(a_aligned,
                result, ordinary_lanes);
        return result;

}

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
// single-precision log, commented out: every caller uses the double-precision kernel
// ALWAYS_INLINE       __m512d nico_log_1_wo_checks_AVX512(const __m512d a ) {
//
//           ALIGN64  __m512d const a_aligned = a;
//
//           ALIGN64  __m512i const log_i1 =   _mm512_set1_epi64(0x3fe5555555555555);
//           ALIGN64  __m512i const log_i2 =   _mm512_set1_epi64(0xFFF0000000000000);
//
//           ALIGN64  __m512d const log_c1 =   _mm512_set1_pd(0.000000000000000222044604925031308085);
//           ALIGN64  __m512d const log_c2 =   _mm512_set1_pd(-0.13031005859375);
//           ALIGN64  __m512d const log_c3 =   _mm512_set1_pd(0.140869140625);
//           ALIGN64  __m512d const log_c4 =   _mm512_set1_pd(-0.121483512222766876221);           
//           ALIGN64  __m512d const log_c5 =   _mm512_set1_pd(0.139814853668212890625);
//           ALIGN64  __m512d const log_c6 =   _mm512_set1_pd(-0.166846126317977905273);
//           ALIGN64  __m512d const log_c7 =   _mm512_set1_pd(0.200120344758033752441);
//           ALIGN64  __m512d const log_c8 =   _mm512_set1_pd(-0.249996200203895568848);
//           ALIGN64  __m512d const log_c9 =   _mm512_set1_pd(0.333331972360610961914);
//           ALIGN64  __m512d const log_c10 =  _mm512_set1_pd(-0.5);
//           ALIGN64  __m512d const log_c11 =  _mm512_set1_pd(0.693147182464599609375);
//
//           ALIGN64  __m512i const aInt = _mm512_castpd_si512(a_aligned);
//
//           //    e = (__double_as_int (a) - i1 )    &   i2   ;
//           ALIGN64  __m512i const e = _mm512_and_epi64( _mm512_sub_epi64(aInt, log_i1),  log_i2);
//           // 0x1.0p-52
//           ALIGN64  __m512d const i = _mm512_fmadd_pd ( _mm512_cvtepi64_pd(e), log_c1, _mm512_setzero_pd());
//           ALIGN64  __m512d const m =
//                   _mm512_sub_pd(_mm512_castsi512_pd( _mm512_sub_epi64(aInt, e)), _mm512_set1_pd(1.0)) ;
//           //   m = __int_as_double (__double_as_int (a) - e);  
//           //  m = _mm256_sub_pd(m, one) ; // m - 1.0;  /* m in [2/3, 4/3] */
//           // m = _mm512_sub_pd(m, one) ; // m - 1.0;  /* m in [2/3, 4/3] */
//           ALIGN64  __m512d const s = _mm512_mul_pd(m, m);
//
//           /* Compute log1p(m) for m in [-1/3, 1/3] */
//           ALIGN64  __m512d r =             log_c2;  // -0x1.0ae000p-3
//           ALIGN64  __m512d t =             log_c3;  //  0x1.208000p-3
//           r = _mm512_fmadd_pd (r, s, log_c4); // -0x1.f198b2p-4
//           t = _mm512_fmadd_pd (t, s, log_c5); //  0x1.1e5740p-3
//           r = _mm512_fmadd_pd (r, s, log_c6); // -0x1.55b36cp-3
//           t = _mm512_fmadd_pd (t, s, log_c7); //  0x1.99d8b2p-3
//           r = _mm512_fmadd_pd (r, s, log_c8); // -0x1.fffe02p-3
//           r = _mm512_fmadd_pd (t, m, r);
//           r = _mm512_fmadd_pd (r, m, log_c9); //  0x1.5554fap-2
//           r = _mm512_fmadd_pd (r, m, log_c10); // -0x1.000000p-1
//           r = _mm512_fmadd_pd (r, s, m);
//
//           return _mm512_fmadd_pd(i,  log_c11, r); //  0x1.62e430p-1 // log(2)
//
// }


//// ---- log, double precision (nico_log_1_double_precision_*): the fdlibm (Sun Microsystems, e_log.c) algorithm,
////      vectorised. a = 2^k (1 + f) with 1 + f in [sqrt(2)/2, sqrt(2)) (the exponent trick below, with sqrt(2)/2
////      as the split point); s = f / (2 + f), |s| <= 0.1716; R = s^2 (Lg1 + s^4 (Lg3 + s^4 (Lg5 + s^4 Lg7))) +
////      s^4 (Lg2 + s^4 (Lg4 + s^4 Lg6)); log(a) = k ln2_hi - ((hfsq - (s (hfsq + R) + k ln2_lo)) - f), hfsq =
////      f^2 / 2 (ln 2 split into hi and lo parts). Below 1 ulp (measured 0.83 ulp); the only log (the single-
////      precision one, about 1.5e-8 relative error, is commented out). ---- reduction for a positive normal
////      double: a = 2^k (1 + f) with 1 + f in [sqrt(2)/2, sqrt(2)) (k exact, as a double)
ALWAYS_INLINE void nico_log_reduce_AVX512(const __m512d a, __m512d& f, __m512d& k) {

        ALIGN64  __m512d const a_aligned = a;

        ALIGN64  __m512i const sqrt_half_bits = _mm512_set1_epi64(0x3fe6a09e667f3bcd);       // bits of sqrt(2)/2
        ALIGN64  __m512i const exponent_mask  = _mm512_set1_epi64(0xFFF0000000000000);
        ALIGN64  __m512i const a_bits         = _mm512_castpd_si512(a_aligned);
        // e = k << 52; k = e * 2^-52 (exact); f exact
        ALIGN64  __m512i const e              =
                _mm512_and_epi64(_mm512_sub_epi64(a_bits, sqrt_half_bits), exponent_mask);
        k = _mm512_mul_pd(_mm512_cvtepi64_pd(e), _mm512_set1_pd(0.000000000000000222044604925031308085));
        f = _mm512_sub_pd(_mm512_castsi512_pd(_mm512_sub_epi64(a_bits, e)), _mm512_set1_pd(1.0));

}

//// ---- the fdlibm core: k ln 2 + log1p(f) for 1 + f in [sqrt(2)/2, sqrt(2)) (shared by nico_log_1_* and
////      nico_log1p_1_*)
ALWAYS_INLINE __m512d nico_log_fdlibm_core_AVX512(const __m512d f, const __m512d k) {

        ALIGN64  __m512d const f_aligned = f;
        ALIGN64  __m512d const k_aligned = k;

        ALIGN64  __m512d const ln2_hi = _mm512_set1_pd(6.93147180369123816490e-01);   // 0x3fe62e42fee00000
        ALIGN64  __m512d const ln2_lo = _mm512_set1_pd(1.90821492927058770002e-10);   // 0x3dea39ef35793c76
        ALIGN64  __m512d const Lg1    = _mm512_set1_pd(6.666666666666735130e-01);     // 0x3FE5555555555593
        ALIGN64  __m512d const Lg2    = _mm512_set1_pd(3.999999999940941908e-01);     // 0x3FD999999997FA04
        ALIGN64  __m512d const Lg3    = _mm512_set1_pd(2.857142874366239149e-01);     // 0x3FD2492494229359
        ALIGN64  __m512d const Lg4    = _mm512_set1_pd(2.222219843214978396e-01);     // 0x3FCC71C51D8E78AF
        ALIGN64  __m512d const Lg5    = _mm512_set1_pd(1.818357216161805012e-01);     // 0x3FC7466496CB03DE
        ALIGN64  __m512d const Lg6    = _mm512_set1_pd(1.531383769920937332e-01);     // 0x3FC39A09D078C69F
        ALIGN64  __m512d const Lg7    = _mm512_set1_pd(1.479819860511658591e-01);     // 0x3FC2F112DF3E5244

        ALIGN64  __m512d const s      = _mm512_div_pd(f_aligned, _mm512_add_pd(_mm512_set1_pd(2.0), f_aligned));
        ALIGN64  __m512d const s2     = _mm512_mul_pd(s, s);
        ALIGN64  __m512d const s4     = _mm512_mul_pd(s2, s2);
        ALIGN64  __m512d const t_odd  = _mm512_mul_pd(s2, _mm512_fmadd_pd(s4, _mm512_fmadd_pd(s4,
                                                                     _mm512_fmadd_pd(s4, Lg7, Lg5), Lg3), Lg1));
        ALIGN64  __m512d const t_even =
                _mm512_mul_pd(s4, _mm512_fmadd_pd(s4, _mm512_fmadd_pd(s4, Lg6, Lg4), Lg2));
        ALIGN64  __m512d const R      = _mm512_add_pd(t_odd, t_even);
        ALIGN64  __m512d const hfsq   = _mm512_mul_pd(_mm512_set1_pd(0.5), _mm512_mul_pd(f_aligned, f_aligned));

        ALIGN64  __m512d const inner  = _mm512_fmadd_pd(s, _mm512_add_pd(hfsq, R), _mm512_mul_pd(k_aligned,
                ln2_lo));
        return _mm512_sub_pd(_mm512_mul_pd(k_aligned, ln2_hi), _mm512_sub_pd(_mm512_sub_pd(hfsq, inner),
                f_aligned));

}

ALWAYS_INLINE __m512d nico_log_1_double_precision_wo_checks_AVX512(const __m512d a) {

        ALIGN64  __m512d const a_aligned = a;

        __m512d f, k;
        nico_log_reduce_AVX512(a_aligned, f, k);
        return nico_log_fdlibm_core_AVX512(f, k);

}
 
 
///// compute log(1+x) without losing precision for small values of x. //  see: https://www.johndcook.com/cpp_log_one_plus_x.html
// single-precision log, commented out: every caller uses the double-precision kernel
// ALWAYS_INLINE  __m512d nico_log1p_1_wo_checks_AVX512(const __m512d x)   {
//
//           ALIGN64  __m512d const x_aligned = x;
//
//           ALIGN64  __m512d const small =  _mm512_set1_pd(1e-4);
//           ALIGN64  __m512d const minus_one_half  =  _mm512_set1_pd(-0.50);
//
//           /// if abs(x) > small
//           const __mmask8 is_abs_x_gr_1e_m_4 =      _mm512_cmp_pd_mask(_mm512_abs_pd(x_aligned), small,
//                   _CMP_GT_OQ);
//
//           return       _mm512_mask_blend_pd(is_abs_x_gr_1e_m_4,
//                                             _mm512_mul_pd(_mm512_fmadd_pd( minus_one_half, x_aligned,
//                                                     _mm512_set1_pd(1.0)), x_aligned) ,
//                                             ////  if fabs(x) > small
//                                             nico_log_1_wo_checks_AVX512( _mm512_add_pd(x_aligned, _mm512_set1_pd(
//                                                     1.0)) ) ) ;
//
// }




//// ---- log1p, double precision (nico_log1p_1_double_precision_*; the only log1p: the 2-term + single-precision
////      version above is commented out). For x in [sqrt(2)/2 - 1, sqrt(2) - 1) the fdlibm core runs on
////      f = x itself (k = 0): 1 + x is never rounded. Otherwise log(u), u = 1 + x, reduced as in nico_log_1,
////      plus (x - (u - 1)) / u, the rounding error of u to first order (a 14-bit reciprocal of u is ample for
////      that term). One core evaluation (one division) for any mix of lanes; the reciprocal only when a lane is
////      outside the core range.
static constexpr double FAST_LOG1P_CORE_LOWER = -0.29289321881345247560;   // sqrt(2)/2 - 1
static constexpr double FAST_LOG1P_CORE_UPPER =  0.41421356237309504880;   // sqrt(2) - 1
ALWAYS_INLINE __m512d nico_log1p_1_double_precision_wo_checks_AVX512(const __m512d x) {

        ALIGN64  __m512d const x_aligned = x;

        const __mmask8 in_core = _mm512_cmp_pd_mask(x_aligned, _mm512_set1_pd(FAST_LOG1P_CORE_LOWER), _CMP_GE_OQ)
                               & _mm512_cmp_pd_mask(x_aligned, _mm512_set1_pd(FAST_LOG1P_CORE_UPPER), _CMP_LT_OQ);
        if (in_core == 0xff) return nico_log_fdlibm_core_AVX512(x_aligned, _mm512_setzero_pd());
        ALIGN64  __m512d const u = _mm512_add_pd(_mm512_set1_pd(1.0), x_aligned);
        __m512d f_u, k_u;
        nico_log_reduce_AVX512(u, f_u, k_u);
        ALIGN64  __m512d const log_value  = nico_log_fdlibm_core_AVX512(
                                       _mm512_mask_blend_pd(in_core, f_u, x_aligned),
                                       _mm512_mask_blend_pd(in_core, k_u, _mm512_setzero_pd()));
        ALIGN64  __m512d const correction = _mm512_mul_pd(_mm512_sub_pd(x_aligned, _mm512_sub_pd(u,
                _mm512_set1_pd(1.0))),
                                                 _mm512_rcp14_pd(u));
        return _mm512_mask_blend_pd(in_core, _mm512_add_pd(log_value, correction), log_value);

}

// ///////  compute log(1+x) without losing precision for small values of x. //  see: https://www.johndcook.com/cpp_log_one_plus_x.html
ALWAYS_INLINE      __m512d nico_log1m_1_wo_checks_AVX512(const __m512d x)   {

          ALIGN64  __m512d const x_aligned = x;
  
          ALIGN64  __m512d const neg_x = _mm512_sub_pd(_mm512_setzero_pd(), x_aligned);
          return nico_log1p_1_double_precision_wo_checks_AVX512(neg_x) ; // -x);
  
}


// ALWAYS_INLINE      __m512d nico_log1p_exp_1_wo_checks_AVX512(const __m512d x)   {
//
//           const __m512d neg_x =  _mm512_sub_pd(_mm512_setzero_pd(), x);
//           const __mmask8 is_x_gr_0_mask =      _mm512_cmp_pd_mask(x,  _mm512_setzero_pd(), _CMP_GT_OQ);
//
//           return       _mm512_mask_blend_pd(is_x_gr_0_mask,
//                                             nico_log1p_1_wo_checks_AVX512(nico_exp_1_wo_checks_AVX512(x)),  /// if x < 0
//                                             _mm512_add_pd(x, nico_log1p_1_wo_checks_AVX512(nico_exp_1_wo_checks_AVX512(neg_x)))); //// if x > 0 
//
// }
// (one exp and one log1p / one division per lane, of the argument which the lane's branch uses, instead of
//  both branches in every lane: the same operations, lane by lane, and so the same values, at half the cost)
ALWAYS_INLINE      __m512d nico_log1p_exp_1_wo_checks_AVX512(const __m512d x)   {

          ALIGN64  __m512d const x_aligned = x;
          const __mmask8 is_x_gr_0_mask = _mm512_cmp_pd_mask(x_aligned, _mm512_setzero_pd(), _CMP_GT_OQ);
          ALIGN64  __m512d const argument = _mm512_mask_blend_pd(is_x_gr_0_mask, x_aligned,
                                                         _mm512_sub_pd(_mm512_setzero_pd(), x_aligned));
          ALIGN64  __m512d const log1p_exp_of_argument =
              nico_log1p_1_double_precision_wo_checks_AVX512(nico_exp_1_wo_checks_AVX512(argument));
          return _mm512_mask_blend_pd(is_x_gr_0_mask, log1p_exp_of_argument,                      /// if x <= 0
                                      /// if x > 0
                                      _mm512_add_pd(x_aligned, log1p_exp_of_argument));

}






//////https://stackoverflow.com/a/65537754/9007125
// Isolate rare subnormal/domain repairs from the normal-positive SIMD path.
__attribute__((noinline, cold)) inline __m512d nico_log_exceptional_lanes_AVX512(const __m512d input,
              const __m512d result,
              const unsigned ordinary_mask) {

    ALIGN64  __m512d const input_aligned = input;
    ALIGN64  __m512d const result_aligned = result;
    alignas(64) double input_values[8], output_values[8];
    _mm512_store_pd(input_values, input_aligned);
    _mm512_store_pd(output_values, result_aligned);
    for (int lane = 0; lane < 8; ++lane) {
        if (!(ordinary_mask & (1u << lane))) output_values[lane] = std::log(input_values[lane]);
    }
    return _mm512_load_pd(output_values);

}

// single-precision log, commented out: every caller uses the double-precision kernel
// ALWAYS_INLINE      __m512d nico_log_1_AVX512(const __m512d a ) {
//
//         ALIGN64  __m512d const a_aligned = a;
//
//         // The mantissa/exponent shortcut assumes a positive NORMAL double.
//         // A single unsigned interval test covers positive normal doubles, excluding infinity/NaN.
//         const __mmask8 ordinary_lanes = _mm512_cmp_epu64_mask(
//             _mm512_sub_epi64(_mm512_castpd_si512(a_aligned), _mm512_set1_epi64(0x0010000000000000)),
//             _mm512_set1_epi64(0x7fe0000000000000), _MM_CMPINT_LT);
//         ALIGN64  __m512d const result = nico_log_1_wo_checks_AVX512(a_aligned);
//         if (__builtin_expect(ordinary_lanes != 0xff, 0)) return nico_log_exceptional_lanes_AVX512(a_aligned,
//                 result, ordinary_lanes);
//         return result;
//
// }

// double-precision log (fdlibm, one division per vector) with the same scalar repair of non-normal / non-positive lanes
ALWAYS_INLINE __m512d nico_log_1_double_precision_AVX512(const __m512d a) {

        ALIGN64  __m512d const a_aligned = a;

        const __mmask8 ordinary_lanes = _mm512_cmp_epu64_mask(
            _mm512_sub_epi64(_mm512_castpd_si512(a_aligned), _mm512_set1_epi64(0x0010000000000000)),
            _mm512_set1_epi64(0x7fe0000000000000), _MM_CMPINT_LT);
        ALIGN64  __m512d const result = nico_log_1_double_precision_wo_checks_AVX512(a_aligned);
        if (__builtin_expect(ordinary_lanes != 0xff, 0)) {
            return nico_log_exceptional_lanes_AVX512(a_aligned, result, ordinary_lanes);
        }
        return result;

}


///////  compute log(1+x) without losing precision for small values of x.
/////see: https://www.johndcook.com/cpp_log_one_plus_x.html 
// single-precision log, commented out: every caller uses the double-precision kernel
// ALWAYS_INLINE      __m512d nico_log1p_1_AVX512(const __m512d x)   {
//
//   ALIGN64  __m512d const x_aligned = x;
//
//   ALIGN64  __m512d const minus_one  =  _mm512_set1_pd(-1.0);
//   ALIGN64  __m512d const minus_one_half  =  _mm512_set1_pd(-0.50);
//   ALIGN64  __m512d const small =  _mm512_set1_pd(1e-4);
//
//   /// if x <= -1.0
//   const __mmask8 is_x_le_or_eq_to_m1_mask =     _mm512_cmp_pd_mask(x_aligned,  minus_one, _CMP_LE_OQ);
//   //// if abs(x) > small
//   const __mmask8 is_abs_x_gr_1e_m_4_mask  =     _mm512_cmp_pd_mask(_mm512_abs_pd(x_aligned), small, _CMP_GT_OQ);
//
//   return  _mm512_mask_blend_pd(is_x_le_or_eq_to_m1_mask,              // is x <= -1? (i.e., is x+1 <= 0?)
//                                _mm512_mask_blend_pd(is_abs_x_gr_1e_m_4_mask,
//                                                      // if "is_x_le_or_eq_to_m1_mask" is FALSE (i.e., x+1 > 0)
//                                                     _mm512_mul_pd(_mm512_fmadd_pd(minus_one_half, x_aligned,
//                                                             _mm512_set1_pd(1.0)), x_aligned),
//                                                     // if x > 1e-4 - evaluate using normal log fn!
//                                                     nico_log_1_AVX512(_mm512_add_pd(x_aligned,  _mm512_set1_pd(
//                                                             1.0)))),
//                                // if FALSE (i.e., x+1  > 0 ) then still pass onto normal log fn as this will handle NaNs etc!
//                                nico_log_1_AVX512(_mm512_add_pd(x_aligned,  _mm512_set1_pd(1.0)))) ;
//
// }




// Isolate x <= -1, infinities and NaN (std::log1p) from the SIMD path.
__attribute__((noinline, cold)) inline __m512d nico_log1p_exceptional_lanes_AVX512(const __m512d input,
                                                                                  const __m512d result,
                                                                                  const unsigned ordinary_mask) {

    ALIGN64  __m512d const input_aligned = input;
    ALIGN64  __m512d const result_aligned = result;
    alignas(64) double input_values[8], output_values[8];
    _mm512_store_pd(input_values, input_aligned);
    _mm512_store_pd(output_values, result_aligned);
    for (int lane = 0; lane < 8; ++lane) {
        if (!(ordinary_mask & (1u << lane))) output_values[lane] = std::log1p(input_values[lane]);
    }
    return _mm512_load_pd(output_values);

}

ALWAYS_INLINE __m512d nico_log1p_1_double_precision_AVX512(const __m512d x) {

        ALIGN64  __m512d const x_aligned = x;

        const __mmask8 ordinary_lanes = _mm512_cmp_pd_mask(x_aligned, _mm512_set1_pd(-1.0), _CMP_GT_OQ)
                                      & _mm512_cmp_pd_mask(x_aligned, _mm512_set1_pd(INFINITY), _CMP_LT_OQ);
        ALIGN64  __m512d const result = nico_log1p_1_double_precision_wo_checks_AVX512(x_aligned);
        if (__builtin_expect(ordinary_lanes != 0xff, 0)) {
            return nico_log1p_exceptional_lanes_AVX512(x_aligned, result, ordinary_lanes);
        }
        return result;

}

// //  compute log(1+x) without losing precision for small values of x.
// //  see: https://www.johndcook.com/cpp_log_one_plus_x.html
ALWAYS_INLINE     __m512d nico_log1m_1_AVX512(const __m512d x)   {

  ALIGN64  __m512d const x_aligned = x;
   
  ALIGN64  __m512d const neg_x =  _mm512_sub_pd(_mm512_setzero_pd(), x_aligned);
  return nico_log1p_1_double_precision_AVX512(neg_x) ;  
  
} 


// ALWAYS_INLINE      __m512d nico_log1p_exp_1_AVX512(const  __m512d x)   {
//
//   const __m512d neg_x =  _mm512_sub_pd(_mm512_setzero_pd(), x);
//   const __mmask8 is_x_gr_0_mask =      _mm512_cmp_pd_mask(x,  _mm512_setzero_pd(), _CMP_GT_OQ);
//
//   return       _mm512_mask_blend_pd(is_x_gr_0_mask, /// if x > 0
//                                     nico_log1p_1_AVX512(nico_exp_1_AVX512(x)), /// if x < 0  ----log1p(exp(x))
//                                     _mm512_add_pd(x, nico_log1p_1_AVX512(nico_exp_1_AVX512(neg_x)))); /// if x > 0 ---- x + log1p(exp(-x))
//
// }
// (one exp and one log1p / one division per lane, of the argument which the lane's branch uses, instead of
//  both branches in every lane: the same operations, lane by lane, and so the same values, at half the cost)
ALWAYS_INLINE      __m512d nico_log1p_exp_1_AVX512(const __m512d x)   {

          ALIGN64  __m512d const x_aligned = x;
          const __mmask8 is_x_gr_0_mask = _mm512_cmp_pd_mask(x_aligned, _mm512_setzero_pd(), _CMP_GT_OQ);
          ALIGN64  __m512d const argument = _mm512_mask_blend_pd(is_x_gr_0_mask, x_aligned,
                                                         _mm512_sub_pd(_mm512_setzero_pd(), x_aligned));
          ALIGN64  __m512d const log1p_exp_of_argument = nico_log1p_1_double_precision_AVX512(nico_exp_1_AVX512(argument));
          return _mm512_mask_blend_pd(is_x_gr_0_mask, log1p_exp_of_argument,                      /// if x <= 0
                                      /// if x > 0
                                      _mm512_add_pd(x_aligned, log1p_exp_of_argument));

}




//// ---- log(1 - exp(x)) (log1m_exp), x <= 0: x <  -ln 2:  log1p(-exp(x)) with the default nico_exp_1 and
////      nico_log1p_1 (as the Stan external's log1p(-exp(x)) branch: the same values) x >= -ln 2:  log(-expm1(x))
////      = log(-x) + h(x),  h(x) = log(expm1(x) / x) = x/2 + log(sinh(x/2) / (x/2)) = x/2 + sum_k B_2k x^2k / (2k
////      (2k)!)  (B_2k the Bernoulli numbers), to k = 8 (x^16): truncation below 1e-18 and rounding about 3e-17
////      on [-ln 2, 0); log(-x) by the double-precision log (nico_log_1_double_precision_*): no cancellation as x
////      -> 0, and no scalar log / expm1. Each branch is evaluated only when at least one lane needs it. x = 0
////      gives -inf, x > 0 and NaN give NaN, x = -inf gives 0 (exceptional lanes go through the scalar repairs of
////      nico_exp_1, nico_log_1, nico_log1p_1).
static constexpr double FAST_LOG1M_EXP_LN2 = 0.693147180559945309417232121458;
ALWAYS_INLINE __m512d nico_log1m_exp_1_AVX512(const __m512d x) {

        ALIGN64  __m512d const x_aligned = x;

        // x >= -ln 2 (NaN: false)
        const __mmask8 near_zero = _mm512_cmp_pd_mask(x_aligned, _mm512_set1_pd(-FAST_LOG1M_EXP_LN2), _CMP_GE_OQ);
        ALIGN64  __m512d far_result  = _mm512_setzero_pd();
        ALIGN64  __m512d near_result = _mm512_setzero_pd();
        if (near_zero != 0xff) {
            far_result = nico_log1p_1_double_precision_AVX512(_mm512_sub_pd(_mm512_setzero_pd(), nico_exp_1_AVX512(x_aligned)));
        }
        if (near_zero != 0) {
            ALIGN64  __m512d const u = _mm512_mul_pd(x_aligned, x_aligned);
            ALIGN64  __m512d e = _mm512_set1_pd(-3617.0 / 170729965486080000.0);
        e = _mm512_fmadd_pd(e, u, _mm512_set1_pd(1.0 / 1046139494400.0));
        e = _mm512_fmadd_pd(e, u, _mm512_set1_pd(-691.0 / 15692092416000.0));
        e = _mm512_fmadd_pd(e, u, _mm512_set1_pd(1.0 / 479001600.0));
        e = _mm512_fmadd_pd(e, u, _mm512_set1_pd(-1.0 / 9676800.0));
        e = _mm512_fmadd_pd(e, u, _mm512_set1_pd(1.0 / 181440.0));
        e = _mm512_fmadd_pd(e, u, _mm512_set1_pd(-1.0 / 2880.0));
        e = _mm512_fmadd_pd(e, u, _mm512_set1_pd(1.0 / 24.0));
            ALIGN64  __m512d const h = _mm512_fmadd_pd(u, e, _mm512_mul_pd(_mm512_set1_pd(0.5), x_aligned));
            ALIGN64  __m512d const minus_x        = _mm512_sub_pd(_mm512_setzero_pd(), x_aligned);
            ALIGN64  __m512d const log_of_minus_x = nico_log_1_double_precision_AVX512(minus_x);
            near_result = _mm512_add_pd(log_of_minus_x, h);
        }
        return _mm512_mask_blend_pd(near_zero, far_result, near_result);

}



// //  compute log(1+x) without losing precision for small values of x. //  see: https://www.johndcook.com/cpp_log_one_plus_x.html
// inline   double nico_log1p_exp_1(const double x)   {
//   
//   if (x > 0.0) return x + nico_log1p_1(nico_exp_1(-x));
//   return nico_log1p_1(nico_exp_1(x));
//   
// }


 

 

ALWAYS_INLINE     __m512d nico_logit_wo_checks_AVX512(const __m512d x)   {

   ALIGN64  __m512d const x_aligned = x;
   
   ALIGN64  __m512d const log_x = nico_log_1_double_precision_AVX512(x_aligned);
   ALIGN64  __m512d const log_1m_x =  nico_log1m_1_AVX512(x_aligned);
   return  log_x - log_1m_x;
   
} 



ALWAYS_INLINE     __m512d nico_logit_AVX512(const __m512d x)   {

   ALIGN64  __m512d const x_aligned = x;
   
   ALIGN64  __m512d const log_x = nico_log_1_double_precision_AVX512(x_aligned);
   ALIGN64  __m512d const log_1m_x =  nico_log1m_1_AVX512(x_aligned);
   return  log_x - log_1m_x;
   
} 
 












 







///////////////////  fns - inv_logit   ----------------------------------------------------------------------------------------------------------------
ALWAYS_INLINE     __m512d  nico_inv_logit_for_x_pos_AVX512(const __m512d x )  {

      ALIGN64  __m512d const x_aligned = x;

      ALIGN64  __m512d const exp_m_x =  nico_exp_1_AVX512(_mm512_sub_pd(_mm512_setzero_pd(), x_aligned)) ;
      return    _mm512_div_pd(_mm512_set1_pd(1.0), _mm512_add_pd(_mm512_set1_pd(1.0), exp_m_x))  ;
          
}




ALWAYS_INLINE     __m512d  nico_inv_logit_for_x_neg_AVX512(const __m512d x )  {

      ALIGN64  __m512d const x_aligned = x;
  
      ALIGN64  __m512d const log_eps  =   _mm512_set1_pd(-18.420680743952367);
      ALIGN64  __m512d const exp_x =  nico_exp_1_AVX512(x_aligned) ;
      
      return _mm512_mask_blend_pd(_mm512_cmp_pd_mask(x_aligned,  log_eps, _CMP_GT_OQ),
                                  exp_x,
                                  _mm512_div_pd(exp_x, _mm512_add_pd(_mm512_set1_pd(1.0), exp_x)));
  
}




// ALWAYS_INLINE     __m512d  nico_inv_logit_AVX512(const __m512d x )  {
//
//   __m512d result =   _mm512_mask_blend_pd(_mm512_cmp_pd_mask(x,  _mm512_setzero_pd(), _CMP_GE_OQ),
//                                    nico_inv_logit_for_x_neg_AVX512(x),
//                                    nico_inv_logit_for_x_pos_AVX512(x));
//
//   return  _mm512_min_pd(_mm512_set1_pd(1.0), _mm512_max_pd(_mm512_setzero_pd(), result));
//
// }
// (one exp and one log1p / one division per lane, of the argument which the lane's branch uses, instead of
//  both branches in every lane: the same operations, lane by lane, and so the same values, at half the cost)
ALWAYS_INLINE     __m512d  nico_inv_logit_AVX512(const __m512d x )  {

  ALIGN64  __m512d const x_aligned = x;
  ALIGN64  __m512d const one = _mm512_set1_pd(1.0);
  const __mmask8 is_x_pos = _mm512_cmp_pd_mask(x_aligned, _mm512_setzero_pd(), _CMP_GE_OQ);
  const __mmask8 is_x_gr_log_eps = _mm512_cmp_pd_mask(x_aligned, _mm512_set1_pd(-18.420680743952367), _CMP_GT_OQ);
  ALIGN64  __m512d const exp_of_argument = nico_exp_1_AVX512(_mm512_mask_blend_pd(is_x_pos, x_aligned,
                                                                 _mm512_sub_pd(_mm512_setzero_pd(), x_aligned)));
  //// 1 / (1 + exp(-x)) (x pos), exp(x) / (1 + exp(x)) (x neg, x > log_eps), exp(x) (x <= log_eps)
  ALIGN64  __m512d const quotient = _mm512_div_pd(_mm512_mask_blend_pd(is_x_pos, exp_of_argument, one),
                                          _mm512_add_pd(one, exp_of_argument));
  ALIGN64  __m512d const result = _mm512_mask_blend_pd(is_x_pos | is_x_gr_log_eps, exp_of_argument, quotient);
  return  _mm512_min_pd(one, _mm512_max_pd(_mm512_setzero_pd(), result));

}

 






 

 






ALWAYS_INLINE     __m512d  nico_inv_logit_for_x_pos_wo_checks_AVX512(const __m512d x )  {

   ALIGN64  __m512d const x_aligned = x;
  
   ALIGN64  __m512d const exp_m_x =  nico_exp_1_wo_checks_AVX512(_mm512_sub_pd(_mm512_setzero_pd(), x_aligned)) ;
   return    _mm512_div_pd(_mm512_set1_pd(1.0), _mm512_add_pd(_mm512_set1_pd(1.0), exp_m_x))  ;
   
}




ALWAYS_INLINE      __m512d  nico_inv_logit_for_x_neg_wo_checks_AVX512(const __m512d x )  {

  ALIGN64  __m512d const x_aligned = x;
  
  ALIGN64  __m512d const log_eps  =   _mm512_set1_pd(-18.420680743952367);
  ALIGN64  __m512d const exp_x =  nico_exp_1_wo_checks_AVX512(x_aligned) ;
  
  return _mm512_mask_blend_pd(_mm512_cmp_pd_mask(x_aligned,  log_eps, _CMP_GT_OQ),
                              exp_x,
                              _mm512_div_pd(exp_x, _mm512_add_pd(_mm512_set1_pd(1.0), exp_x)));
  
}




// ALWAYS_INLINE     __m512d  nico_inv_logit_wo_checks_AVX512(const __m512d x )  {
//
//   __m512d result =   _mm512_mask_blend_pd(_mm512_cmp_pd_mask(x,  _mm512_setzero_pd(), _CMP_GT_OQ),
//                                   nico_inv_logit_for_x_neg_wo_checks_AVX512(x),
//                                   nico_inv_logit_for_x_pos_wo_checks_AVX512(x));
//
//   return  _mm512_min_pd(_mm512_set1_pd(1.0), _mm512_max_pd(_mm512_setzero_pd(), result));
//
// }
// (one exp and one log1p / one division per lane, of the argument which the lane's branch uses, instead of
//  both branches in every lane: the same operations, lane by lane, and so the same values, at half the cost)
ALWAYS_INLINE     __m512d  nico_inv_logit_wo_checks_AVX512(const __m512d x )  {

  ALIGN64  __m512d const x_aligned = x;
  ALIGN64  __m512d const one = _mm512_set1_pd(1.0);
  const __mmask8 is_x_pos = _mm512_cmp_pd_mask(x_aligned, _mm512_setzero_pd(), _CMP_GT_OQ);
  const __mmask8 is_x_gr_log_eps = _mm512_cmp_pd_mask(x_aligned, _mm512_set1_pd(-18.420680743952367), _CMP_GT_OQ);
  ALIGN64  __m512d const exp_of_argument = nico_exp_1_wo_checks_AVX512(_mm512_mask_blend_pd(is_x_pos, x_aligned,
                                                                 _mm512_sub_pd(_mm512_setzero_pd(), x_aligned)));
  //// 1 / (1 + exp(-x)) (x pos), exp(x) / (1 + exp(x)) (x neg, x > log_eps), exp(x) (x <= log_eps)
  ALIGN64  __m512d const quotient = _mm512_div_pd(_mm512_mask_blend_pd(is_x_pos, exp_of_argument, one),
                                          _mm512_add_pd(one, exp_of_argument));
  ALIGN64  __m512d const result = _mm512_mask_blend_pd(is_x_pos | is_x_gr_log_eps, exp_of_argument, quotient);
  return  _mm512_min_pd(one, _mm512_max_pd(_mm512_setzero_pd(), result));

}







 

 




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
inline __m512d nico_log_inv_logit_for_x_pos_AVX512(const __m512d x) {

      ALIGN64  __m512d const x_aligned = x;
  
      ALIGN64  __m512d const m_x = _mm512_sub_pd(_mm512_setzero_pd(), x_aligned);  // -x
      ALIGN64  __m512d const exp_m_x = nico_exp_1_AVX512(m_x);             // exp(-x)
      ALIGN64  __m512d const log_sum = nico_log1p_1_double_precision_AVX512(exp_m_x);       // log(1 + exp(-x))
      return _mm512_sub_pd(_mm512_setzero_pd(), log_sum);         // -log(1 + exp(-x))
  
}




// For negative x: log(exp(x)/(1 + exp(x))) = x - log(1 + exp(x))
inline __m512d nico_log_inv_logit_for_x_neg_AVX512(const __m512d x) {

      ALIGN64  __m512d const x_aligned = x;
  
      ALIGN64  __m512d const log_eps = _mm512_set1_pd(-18.420680743952367);
      ALIGN64  __m512d const exp_x = nico_exp_1_AVX512(x_aligned);                 // exp(x)
      ALIGN64  __m512d const log_sum = nico_log1p_1_double_precision_AVX512(exp_x);        // log(1 + exp(x))
      ALIGN64  __m512d const result = _mm512_sub_pd(x_aligned, log_sum);          // x - log(1 + exp(x))
      
      // Blend based on x > log_eps 
      __mmask8 mask = _mm512_cmp_pd_mask(x_aligned, log_eps, _CMP_GT_OQ);
      return _mm512_mask_blend_pd(mask,
                                  x_aligned,        // if x <= log_eps
                                  result);  // if x > log_eps
  
}




// inline __m512d nico_log_inv_logit_AVX512(const __m512d x) {
//
//       __mmask8 is_pos = _mm512_cmp_pd_mask(x, _mm512_setzero_pd(), _CMP_GE_OQ);
//
//       return _mm512_mask_blend_pd(is_pos,
//                                   nico_log_inv_logit_for_x_neg_AVX512(x),  // if x < 0 
//                                   nico_log_inv_logit_for_x_pos_AVX512(x)); // if x >= 0
//
// }
// (one exp and one log1p / one division per lane, of the argument which the lane's branch uses, instead of
//  both branches in every lane: the same operations, lane by lane, and so the same values, at half the cost)
inline __m512d nico_log_inv_logit_AVX512(const __m512d x) {

      ALIGN64  __m512d const x_aligned = x;
      const __mmask8 is_x_ge_0 = _mm512_cmp_pd_mask(x_aligned, _mm512_setzero_pd(), _CMP_GE_OQ);
      const __mmask8 is_x_gr_log_eps = _mm512_cmp_pd_mask(x_aligned, _mm512_set1_pd(-18.420680743952367),
              _CMP_GT_OQ);
      ALIGN64  __m512d const log1p_exp_of_argument =
          nico_log1p_1_double_precision_AVX512(nico_exp_1_AVX512(_mm512_mask_blend_pd(is_x_ge_0, x_aligned,
                                                    _mm512_sub_pd(_mm512_setzero_pd(), x_aligned))));
      //// x - log(1 + exp(x)) (x < 0, x > log_eps), x (x <= log_eps), -log(1 + exp(-x)) (x >= 0)
      ALIGN64  __m512d const result_of_x_neg = _mm512_mask_blend_pd(is_x_gr_log_eps, x_aligned,
                                                            _mm512_sub_pd(x_aligned, log1p_exp_of_argument));
      return _mm512_mask_blend_pd(is_x_ge_0, result_of_x_neg,
                                  _mm512_sub_pd(_mm512_setzero_pd(), log1p_exp_of_argument));

}












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
inline     __m512d  nico_log_inv_logit_for_x_pos_wo_checks_AVX512(const __m512d x )  {

        ALIGN64  __m512d const x_aligned = x;
 
        return   _mm512_sub_pd(_mm512_setzero_pd(), nico_log1p_exp_1_wo_checks_AVX512((_mm512_sub_pd(
                _mm512_setzero_pd(), x_aligned)) ) );
  
}




inline     __m512d  nico_log_inv_logit_for_x_neg_wo_checks_AVX512(const __m512d x )  {

        ALIGN64  __m512d const x_aligned = x;
        
        ALIGN64  __m512d const log_eps  =   _mm512_set1_pd(-18.420680743952367);
        
        /// if x > log_eps = -18.420680743952367
        return _mm512_mask_blend_pd(_mm512_cmp_pd_mask(x_aligned,  log_eps, _CMP_GT_OQ),
                                    x_aligned,
                                    _mm512_sub_pd(x_aligned,   nico_log1p_1_double_precision_wo_checks_AVX512(
                                            nico_exp_1_wo_checks_AVX512(x_aligned))));
        
}




// inline     __m512d  nico_log_inv_logit_wo_checks_AVX512(const __m512d x )  {
//
//         return  _mm512_mask_blend_pd(_mm512_cmp_pd_mask(x,  _mm512_setzero_pd(), _CMP_GE_OQ),  // x > 0
//                                      nico_log_inv_logit_for_x_neg_wo_checks_AVX512(x),
//                                      nico_log_inv_logit_for_x_pos_wo_checks_AVX512(x));  // x > 0
//
// }
// (one exp and one log1p / one division per lane, of the argument which the lane's branch uses, instead of
//  both branches in every lane: the same operations, lane by lane, and so the same values, at half the cost)
inline __m512d nico_log_inv_logit_wo_checks_AVX512(const __m512d x) {

      ALIGN64  __m512d const x_aligned = x;
      const __mmask8 is_x_ge_0 = _mm512_cmp_pd_mask(x_aligned, _mm512_setzero_pd(), _CMP_GE_OQ);
      const __mmask8 is_x_gr_log_eps = _mm512_cmp_pd_mask(x_aligned, _mm512_set1_pd(-18.420680743952367),
              _CMP_GT_OQ);
      ALIGN64  __m512d const log1p_exp_of_argument =
          nico_log1p_1_double_precision_wo_checks_AVX512(nico_exp_1_wo_checks_AVX512(_mm512_mask_blend_pd(is_x_ge_0, x_aligned,
                                                    _mm512_sub_pd(_mm512_setzero_pd(), x_aligned))));
      //// x - log(1 + exp(x)) (x < 0, x > log_eps), x (x <= log_eps), -log(1 + exp(-x)) (x >= 0)
      ALIGN64  __m512d const result_of_x_neg = _mm512_mask_blend_pd(is_x_gr_log_eps, x_aligned,
                                                            _mm512_sub_pd(x_aligned, log1p_exp_of_argument));
      return _mm512_mask_blend_pd(is_x_ge_0, result_of_x_neg,
                                  _mm512_sub_pd(_mm512_setzero_pd(), log1p_exp_of_argument));

}




/////////////////////  fns - Phi_approx   ----------------------------------------------------------------------------------------------------------------
inline      __m512d  nico_Phi_approx_wo_checks_AVX512( const __m512d x)  {

        ALIGN64  __m512d const x_aligned = x;
        
        ALIGN64  __m512d const a =     _mm512_set1_pd(0.07056);
        ALIGN64  __m512d const b =     _mm512_set1_pd(1.5976);
      
        ALIGN64  __m512d const x_sq =  _mm512_mul_pd(x_aligned, x_aligned);
        ALIGN64  __m512d const a_x_sq_plus_b = _mm512_fmadd_pd(a, x_sq, b);
        ALIGN64  __m512d const stuff_to_inv_logit =  _mm512_mul_pd(x_aligned, a_x_sq_plus_b);
        
        return    nico_inv_logit_wo_checks_AVX512(stuff_to_inv_logit);
  
}




inline     __m512d  nico_Phi_approx_AVX512(const __m512d x )  {

        ALIGN64  __m512d const x_aligned = x;
  
        ALIGN64  __m512d const a =     _mm512_set1_pd(0.07056);
        ALIGN64  __m512d const b =     _mm512_set1_pd(1.5976);
        
        ALIGN64  __m512d const x_sq =  _mm512_mul_pd(x_aligned, x_aligned);
        ALIGN64  __m512d const a_x_sq_plus_b = _mm512_fmadd_pd(a, x_sq, b);
        ALIGN64  __m512d const stuff_to_inv_logit =  _mm512_mul_pd(x_aligned, a_x_sq_plus_b);
        
        return nico_inv_logit_AVX512(stuff_to_inv_logit);
  
}




///////////////////  fns - inv_Phi_approx   -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
inline     __m512d  nico_inv_Phi_approx_wo_checks_AVX512(const __m512d x )  {

        ALIGN64  __m512d const x_aligned = x;
  
        ALIGN64  __m512d const inv_Phi_approx_c1 = _mm512_set1_pd(-0.34176618822627863);
        ALIGN64  __m512d const inv_Phi_approx_c2 = _mm512_set1_pd(2.7472242570765295);
        ALIGN64  __m512d const one_third =  _mm512_set1_pd(0.33333333333333331483);
        
        // logit first
        ALIGN64  __m512d const m_logit_p =
                nico_log_1_double_precision_wo_checks_AVX512( _mm512_sub_pd(_mm512_div_pd(_mm512_set1_pd(1.0),  x_aligned),
                        _mm512_set1_pd(1.0) ) );
        // -0.34176618822627863*m_logit_p;
        ALIGN64  __m512d const x_i =  _mm512_mul_pd(m_logit_p, inv_Phi_approx_c1) ;
        // now do arc_sinh part
        ALIGN64  __m512d const asinh_stuff_div_3 =
                _mm512_mul_pd(  one_third,  nico_log_1_double_precision_wo_checks_AVX512(  _mm512_add_pd(x_i, (  _mm512_sqrt_pd(  _mm512_fmadd_pd(x_i, x_i, _mm512_set1_pd(1.0)) ) )  ) ) ) ;
        ALIGN64  __m512d const exp_x_i = nico_exp_1_wo_checks_AVX512(asinh_stuff_div_3);
        
        return _mm512_mul_pd(inv_Phi_approx_c2,   _mm512_div_pd(_mm512_sub_pd(_mm512_mul_pd(exp_x_i, exp_x_i), _mm512_set1_pd(1.0)), exp_x_i) );
  
}




inline     __m512d  nico_inv_Phi_approx_AVX512(const __m512d x )  {

        ALIGN64  __m512d const x_aligned = x;

        // Avoid 1 / p overflowing for valid subnormal probabilities. The usual path below is unchanged.
        const __mmask8 ordinary_lanes = _mm512_cmp_pd_mask(x_aligned, _mm512_set1_pd(
                std::numeric_limits<double>::min()), _CMP_GE_OQ)
                                     & _mm512_cmp_pd_mask(x_aligned, _mm512_set1_pd(1.0), _CMP_LT_OQ);
        if (ordinary_lanes != 0xff) {
            alignas(64) double input_values[8], output_values[8];
            _mm512_store_pd(input_values, x_aligned);
            ALIGN64  __m512d const safe_input = _mm512_mask_blend_pd(ordinary_lanes, _mm512_set1_pd(0.5),
                    x_aligned);
            _mm512_store_pd(output_values, nico_inv_Phi_approx_AVX512(safe_input));
            for (int lane = 0; lane < 8; ++lane) {
                if (!(ordinary_lanes & (1u << lane))) {
                    const double logit_probability = std::log(input_values[lane]) - std::log1p(-input_values[lane]);
                    output_values[lane] = 5.494448514153059 * std::sinh(std::asinh(0.34176618822627863 * logit_probability) / 3.0);
                }
            }
            return _mm512_load_pd(output_values);
        }
        
        ALIGN64  __m512d const inv_Phi_approx_c1 = _mm512_set1_pd(-0.34176618822627863);
        ALIGN64  __m512d const inv_Phi_approx_c2 = _mm512_set1_pd(2.7472242570765295);
        ALIGN64  __m512d const one_third =  _mm512_set1_pd(0.33333333333333331483);
        
        // log((1-p)/p), stable near 1
        ALIGN64  __m512d const m_logit_p =
                nico_log_1_double_precision_AVX512(_mm512_div_pd(_mm512_sub_pd(_mm512_set1_pd(1.0), x_aligned), x_aligned));
        // -0.34176618822627863*m_logit_p;
        ALIGN64  __m512d const x_i =  _mm512_mul_pd(m_logit_p, inv_Phi_approx_c1) ;
        // now do arc_sinh part
        ALIGN64  __m512d const asinh_stuff_div_3 =
                _mm512_mul_pd( one_third,  nico_log_1_double_precision_AVX512(  _mm512_add_pd(x_i, (  _mm512_sqrt_pd(  _mm512_fmadd_pd(x_i, x_i, _mm512_set1_pd(1.0)) ) )  ) ) ) ;
        ALIGN64  __m512d const exp_x_i = nico_exp_1_AVX512(asinh_stuff_div_3);
        
        return    _mm512_mul_pd(inv_Phi_approx_c2,   _mm512_div_pd(_mm512_sub_pd(_mm512_mul_pd(exp_x_i, exp_x_i), _mm512_set1_pd(1.0)), exp_x_i) );
        
        // __mmask8 is_x_valid_prob =  _mm512_cmp_pd_mask(x,  zero, _CMP_GT_OQ) & _mm512_cmp_pd_mask(x,  one, _CMP_LT_OQ); // if x in (0, 1)
        // return   _mm512_mask_blend_pd(is_x_valid_prob,
        //                               _mm512_mask_blend_pd( (_mm512_cmp_pd_mask(x,  one, _CMP_GT_OQ) | _mm512_cmp_pd_mask(x,  zero, _CMP_GT_OQ)),  //  if x > 1 or x < 0 (i.e. = NaN / throw error)
        //                                                    _mm512_mask_blend_pd( _mm512_cmp_pd_mask(x,  zero, _CMP_EQ_OQ),
        //                                                                          _mm512_set1_pd(INFINITY), // if x = 1
        //                                                                          _mm512_set1_pd(-INFINITY)),  // if x = 0
        //                                                    _mm512_set1_pd(std::numeric_limits<double>::quiet_NaN()) ),
        //                               _mm512_mul_pd(_mm512_set1_pd(2.7472242570765295),   _mm512_div_pd(_mm512_sub_pd(_mm512_mul_pd(exp_x_i, exp_x_i), one), exp_x_i) ));
  
}




inline      __m512d  nico_inv_Phi_approx_from_logit_prob_wo_checks_AVX512(const __m512d logit_p )  {

        ALIGN64  __m512d const logit_p_aligned = logit_p;
        
        // const __m512d  inv_Phi_approx_c1 = _mm512_set1_pd(-0.34176618822627863);
        ALIGN64  __m512d const inv_Phi_approx_c1 = _mm512_set1_pd(0.34176618822627863);
        ALIGN64  __m512d const inv_Phi_approx_c2 = _mm512_set1_pd(2.7472242570765295);
        ALIGN64  __m512d const one_third =  _mm512_set1_pd(0.33333333333333331483);
       
        ALIGN64  __m512d const x_i =   _mm512_mul_pd(logit_p_aligned, inv_Phi_approx_c1) ;
        // now do arc_sinh part
        ALIGN64  __m512d const asinh_stuff_div_3 =
                _mm512_mul_pd(one_third,  nico_log_1_double_precision_wo_checks_AVX512(  _mm512_add_pd(x_i, (  _mm512_sqrt_pd(  _mm512_fmadd_pd(x_i, x_i, _mm512_set1_pd(1.0)) ) )  ) ) ) ;
        ALIGN64  __m512d const exp_x_i = nico_exp_1_wo_checks_AVX512(asinh_stuff_div_3);
        
        return _mm512_mul_pd(inv_Phi_approx_c2,   _mm512_div_pd(_mm512_sub_pd(_mm512_mul_pd(exp_x_i, exp_x_i), _mm512_set1_pd(1.0)), exp_x_i) );
  
}




inline      __m512d  nico_inv_Phi_approx_from_logit_prob_AVX512(const __m512d logit_p )  {

        ALIGN64  __m512d const logit_p_aligned = logit_p;
        
        // const __m512d  inv_Phi_approx_c1 = _mm512_set1_pd(-0.34176618822627863);
        ALIGN64  __m512d const inv_Phi_approx_c1 = _mm512_set1_pd(0.34176618822627863);
        ALIGN64  __m512d const inv_Phi_approx_c2 = _mm512_set1_pd(2.7472242570765295);
        ALIGN64  __m512d const one_third =  _mm512_set1_pd(0.33333333333333331483);
       
        ALIGN64  __m512d const x_i =   _mm512_mul_pd(logit_p_aligned,  inv_Phi_approx_c1) ;
        // now do arc_sinh part
        ALIGN64  __m512d const asinh_stuff_div_3 =
                _mm512_mul_pd(one_third,  nico_log_1_double_precision_AVX512(  _mm512_add_pd(x_i, (  _mm512_sqrt_pd(  _mm512_fmadd_pd(x_i, x_i, _mm512_set1_pd(1.0)) ) )  ) ) ) ;
        ALIGN64  __m512d const exp_x_i = nico_exp_1_AVX512(asinh_stuff_div_3);
        
        return _mm512_mul_pd(inv_Phi_approx_c2,   _mm512_div_pd(_mm512_sub_pd(_mm512_mul_pd(exp_x_i, exp_x_i), _mm512_set1_pd(1.0)), exp_x_i) );
  
}




///////////////////  fns - log_Phi_approx   ----------------------------------------------------------------------------------------------------------------------------------------
inline      __m512d  nico_log_Phi_approx_wo_checks_AVX512(const __m512d x )  {

        ALIGN64  __m512d const x_aligned = x;
    
        ALIGN64  __m512d const a =     _mm512_set1_pd(0.07056);
        ALIGN64  __m512d const b =     _mm512_set1_pd(1.5976);
        
        ALIGN64  __m512d const x_sq =  _mm512_mul_pd(x_aligned, x_aligned);
        ALIGN64  __m512d const a_x_sq_plus_b = _mm512_fmadd_pd(a, x_sq, b);
        ALIGN64  __m512d const stuff_to_inv_logit =  _mm512_mul_pd(x_aligned, a_x_sq_plus_b);
        
        return nico_log_inv_logit_wo_checks_AVX512(stuff_to_inv_logit);
  
}




inline __m512d nico_log_Phi_approx_AVX512(const __m512d x) {

        ALIGN64  __m512d const x_aligned = x;
      
        ALIGN64  __m512d const Phi_upper_bound =  _mm512_set1_pd(8.25);
        ALIGN64  __m512d const Phi_lower_bound =  _mm512_set1_pd(-37.5);
        
        ALIGN64  __m512d const a =     _mm512_set1_pd(0.07056);
        ALIGN64  __m512d const b =     _mm512_set1_pd(1.5976);
    
        ALIGN64  __m512d const x_sq = _mm512_mul_pd(x_aligned, x_aligned);
        ALIGN64  __m512d const x_cubed = _mm512_mul_pd(x_sq, x_aligned);
        ALIGN64  __m512d const result = _mm512_fmadd_pd(a, x_cubed, _mm512_mul_pd(b, x_aligned));
        
        return nico_log_inv_logit_AVX512(result);
  
}




//////////////////// ------------- tanh  --------------------------------------------------------------------------------------------------------------------------------------
ALWAYS_INLINE      __m512d    nico_tanh_AVX512( const  __m512d  x  )   {

        ALIGN64  __m512d const x_aligned = x;
        
        ALIGN64  __m512d const two =     _mm512_set1_pd(2.0);
        ALIGN64  __m512d const m_two =   _mm512_set1_pd(-2.0);
        
        return        _mm512_sub_pd( _mm512_div_pd(two,   _mm512_add_pd(_mm512_set1_pd(1.0), nico_exp_1_AVX512(
                _mm512_mul_pd(x_aligned, m_two)   ) ) ), _mm512_set1_pd(1.0) ) ;
  
}




ALWAYS_INLINE       __m512d    nico_tanh_wo_checks_AVX512( const  __m512d x  )   {

        ALIGN64  __m512d const x_aligned = x;
  
        ALIGN64  __m512d const two =     _mm512_set1_pd(2.0);
        ALIGN64  __m512d const m_two =   _mm512_set1_pd(-2.0);
        
        return        _mm512_sub_pd( _mm512_div_pd(two,   _mm512_add_pd(_mm512_set1_pd(1.0),
                nico_exp_1_wo_checks_AVX512(  _mm512_mul_pd(x_aligned, m_two)   ) ) ), _mm512_set1_pd(1.0) ) ;
  
}




// from: https://stackoverflow.com/questions/57870896/writing-a-portable-sse-avx-version-of-stdcopysign
ALWAYS_INLINE     __m512d nico_copy_sign( const __m512d srcSign, 
                                    const __m512d srcValue) {

        ALIGN64  __m512d const srcSign_aligned = srcSign;
        ALIGN64  __m512d const srcValue_aligned = srcValue;
  
        ALIGN64  __m512d const mask0 = _mm512_set1_pd(-0.);
        
        ALIGN64  __m512d tmp0 = _mm512_and_pd(srcSign_aligned, mask0); // Extract the signed bit from srcSign
        // Extract the number without sign of srcValue (abs(srcValue))
        ALIGN64  __m512d tmp1 = _mm512_andnot_pd(mask0, srcValue_aligned);
        
        return _mm512_or_pd(tmp0, tmp1);  // Merge signed bit with number and return
  
}
  

  
  
  
  
  
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
ALWAYS_INLINE __m512d nico_Phi_wo_checks_AVX512(__m512d x) {

        ALIGN64  __m512d const x_aligned = x;

        ALIGN64  __m512d const z = _mm512_abs_pd(x_aligned); /////  std::fabs(x);
        //// NaN x: min returns its 2nd operand (the cap), and the NaN still propagates through the exp below.
        ALIGN64  __m512d const z_rational = _mm512_min_pd(z, _mm512_set1_pd(FAST_PHI_MILLS_Z_CAP));

        ALIGN64  __m512d numerator = _mm512_set1_pd(FAST_PHI_MILLS_NUMERATOR_COEFFICIENTS[6]);
        numerator = _mm512_fmadd_pd(numerator,
                                    z_rational,
                                    _mm512_set1_pd(FAST_PHI_MILLS_NUMERATOR_COEFFICIENTS[5]));
        numerator = _mm512_fmadd_pd(numerator,
                                    z_rational,
                                    _mm512_set1_pd(FAST_PHI_MILLS_NUMERATOR_COEFFICIENTS[4]));
        numerator = _mm512_fmadd_pd(numerator,
                                    z_rational,
                                    _mm512_set1_pd(FAST_PHI_MILLS_NUMERATOR_COEFFICIENTS[3]));
        numerator = _mm512_fmadd_pd(numerator,
                                    z_rational,
                                    _mm512_set1_pd(FAST_PHI_MILLS_NUMERATOR_COEFFICIENTS[2]));
        numerator = _mm512_fmadd_pd(numerator,
                                    z_rational,
                                    _mm512_set1_pd(FAST_PHI_MILLS_NUMERATOR_COEFFICIENTS[1]));
        numerator = _mm512_fmadd_pd(numerator,
                                    z_rational,
                                    _mm512_set1_pd(FAST_PHI_MILLS_NUMERATOR_COEFFICIENTS[0]));

        ALIGN64  __m512d denominator = _mm512_set1_pd(FAST_PHI_MILLS_DENOMINATOR_COEFFICIENTS[7]);
        denominator = _mm512_fmadd_pd(denominator,
                                      z_rational,
                                      _mm512_set1_pd(FAST_PHI_MILLS_DENOMINATOR_COEFFICIENTS[6]));
        denominator = _mm512_fmadd_pd(denominator,
                                      z_rational,
                                      _mm512_set1_pd(FAST_PHI_MILLS_DENOMINATOR_COEFFICIENTS[5]));
        denominator = _mm512_fmadd_pd(denominator,
                                      z_rational,
                                      _mm512_set1_pd(FAST_PHI_MILLS_DENOMINATOR_COEFFICIENTS[4]));
        denominator = _mm512_fmadd_pd(denominator,
                                      z_rational,
                                      _mm512_set1_pd(FAST_PHI_MILLS_DENOMINATOR_COEFFICIENTS[3]));
        denominator = _mm512_fmadd_pd(denominator,
                                      z_rational,
                                      _mm512_set1_pd(FAST_PHI_MILLS_DENOMINATOR_COEFFICIENTS[2]));
        denominator = _mm512_fmadd_pd(denominator,
                                      z_rational,
                                      _mm512_set1_pd(FAST_PHI_MILLS_DENOMINATOR_COEFFICIENTS[1]));
        denominator = _mm512_fmadd_pd(denominator,
                                      z_rational,
                                      _mm512_set1_pd(FAST_PHI_MILLS_DENOMINATOR_COEFFICIENTS[0]));

        const __mmask8 is_x_gr_0 = _mm512_cmp_pd_mask(x_aligned, _mm512_setzero_pd(), _CMP_GT_OQ);

        ALIGN64  __m512d const exp_stuff =
                nico_exp_1_AVX512(_mm512_mul_pd(_mm512_set1_pd(-0.50), _mm512_mul_pd(z, z)) );
        //// N / Q is formed BEFORE multiplying by the exp so the divide runs in parallel with nico_exp_1's polynomial instead of
        //// after it (measured: (exp * N) / Q put the divide latency on the critical path and cost ~6% on AVX2).
        //// Phi(-z) = exp(-z^2/2) * N(z) / Q(z)
        ALIGN64  __m512d const res = _mm512_mul_pd(exp_stuff, _mm512_div_pd(numerator, denominator));
        ALIGN64  __m512d const one_m_res = _mm512_sub_pd(_mm512_set1_pd(1.0), res);

        return _mm512_mask_blend_pd(is_x_gr_0,
                                    res,  //// if x < 0
                                    one_m_res); //// if x > 0

}





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


ALWAYS_INLINE __m512d nico_Phi_AVX512(const __m512d x) {

      ALIGN64  __m512d const x_aligned = x;
      
      
      ALIGN64  __m512d const Phi_upper_bound =  _mm512_set1_pd(8.25);
      ALIGN64  __m512d const Phi_lower_bound =  _mm512_set1_pd(-37.5);
  
      // equality must not fall through to 1
      const __mmask8 is_x_gr_lower = _mm512_cmp_pd_mask(x_aligned, Phi_lower_bound, _CMP_GE_OQ);
      // true where x < 8.25 (in range)
      const __mmask8 is_x_lt_upper = _mm512_cmp_pd_mask(x_aligned, Phi_upper_bound, _CMP_LT_OQ);
      const __mmask8 is_x_in_range = is_x_gr_lower & is_x_lt_upper;
      
      // Calculate the main Phi function for in-range values
      ALIGN64  __m512d result = _mm512_mask_blend_pd(
        is_x_in_range,
        _mm512_mask_blend_pd(_mm512_cmp_pd_mask(x_aligned, Phi_lower_bound, _CMP_LT_OQ),
                             _mm512_set1_pd(1.0),    // if x NOT in range and x NOT < -37.5 i.e. x must be > 8.25
                             _mm512_setzero_pd()),  // if x NOT in range and x < -37.5
                             nico_Phi_wo_checks_AVX512(x_aligned) // if x is in range
      );
      
      // Clamp results between 0 and 1
      result = _mm512_min_pd(_mm512_set1_pd(1.0), _mm512_max_pd(_mm512_setzero_pd(), result));
      
      // Preserve NaN input rather than turning it into a valid probability through min/max.
      return _mm512_mask_mov_pd(result, _mm512_cmp_pd_mask(x_aligned, x_aligned, _CMP_UNORD_Q), x_aligned);
  
}












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


ALWAYS_INLINE __m512d nico_inv_Phi_wo_checks_case_2a_AVX512(const __m512d p,
                                                     __m512d r) { ///  CASE 2(a): if abs(q) > 0.425  AND   if r <= 5.0 
  
 // std::cout << "calling  nico_Phi_wo_checks_case_2a_AVX512"   << "\n"; 
  //std::cout << "r = " <<  r  << "\n";
  
  r = _mm512_add_pd(r, _mm512_set1_pd(-1.60));
  
  ALIGN64  __m512d numerator = _mm512_set1_pd(0.00077454501427834140764);
  numerator = _mm512_fmadd_pd(r, numerator, _mm512_set1_pd(0.0227238449892691845833));
  numerator = _mm512_fmadd_pd(r, numerator, _mm512_set1_pd(0.24178072517745061177));
  numerator = _mm512_fmadd_pd(r, numerator, _mm512_set1_pd(1.27045825245236838258));
  numerator = _mm512_fmadd_pd(r, numerator, _mm512_set1_pd(3.64784832476320460504));
  numerator = _mm512_fmadd_pd(r, numerator, _mm512_set1_pd(5.7694972214606914055));
  numerator = _mm512_fmadd_pd(r, numerator, _mm512_set1_pd(4.6303378461565452959));
  numerator = _mm512_fmadd_pd(r, numerator, _mm512_set1_pd(1.42343711074968357734)); 
  
  ALIGN64  __m512d denominator = _mm512_set1_pd(0.00000000105075007164441684324);
  denominator = _mm512_fmadd_pd(r,  denominator, _mm512_set1_pd(0.0005475938084995344946));
  denominator = _mm512_fmadd_pd(r,  denominator, _mm512_set1_pd(0.0151986665636164571966));
  denominator = _mm512_fmadd_pd(r,  denominator, _mm512_set1_pd(0.14810397642748007459));
  denominator = _mm512_fmadd_pd(r,  denominator, _mm512_set1_pd(0.68976733498510000455));
  denominator = _mm512_fmadd_pd(r,  denominator, _mm512_set1_pd(1.6763848301838038494));
  denominator = _mm512_fmadd_pd(r,  denominator, _mm512_set1_pd(2.05319162663775882187));
  denominator = _mm512_fmadd_pd(r, denominator,  _mm512_set1_pd(1.0));
  
  ALIGN64  __m512d const val = _mm512_div_pd(numerator, denominator);
  
  return val;
  
}  


ALWAYS_INLINE __m512d nico_inv_Phi_wo_checks_case_2b_AVX512(const __m512d p,
                                                     __m512d r) { ///  CASE 2(a): if abs(q) > 0.425  AND   if r > 5.0 
  
 // std::cout << "calling  nico_Phi_wo_checks_case_2b_AVX512"   << "\n";
  //std::cout << "r = " <<  r  << "\n";
  
  r = _mm512_add_pd(r, _mm512_set1_pd(-5.0));
  
  ALIGN64  __m512d numerator =  _mm512_set1_pd(0.000000201033439929228813265);
  numerator = _mm512_fmadd_pd(r, numerator,  _mm512_set1_pd(0.0000271155556874348757815));
  numerator = _mm512_fmadd_pd(r, numerator,  _mm512_set1_pd(0.0012426609473880784386));
  numerator = _mm512_fmadd_pd(r, numerator,  _mm512_set1_pd(0.026532189526576123093));
  numerator = _mm512_fmadd_pd(r, numerator,  _mm512_set1_pd(0.29656057182850489123));
  numerator = _mm512_fmadd_pd(r, numerator,  _mm512_set1_pd(1.7848265399172913358));
  numerator = _mm512_fmadd_pd(r, numerator,  _mm512_set1_pd(5.4637849111641143699));
  numerator = _mm512_fmadd_pd(r, numerator,  _mm512_set1_pd(6.6579046435011037772));
  
  ALIGN64  __m512d denominator =  _mm512_set1_pd(0.00000000000000204426310338993978564);
  denominator = _mm512_fmadd_pd(r, denominator,  _mm512_set1_pd(0.00000014215117583164458887));
  denominator = _mm512_fmadd_pd(r, denominator,  _mm512_set1_pd(0.000018463183175100546818));
  denominator = _mm512_fmadd_pd(r, denominator,  _mm512_set1_pd(0.0007868691311456132591));
  denominator = _mm512_fmadd_pd(r, denominator,  _mm512_set1_pd(0.0148753612908506148525));
  denominator = _mm512_fmadd_pd(r, denominator,  _mm512_set1_pd(0.13692988092273580531)); 
  denominator = _mm512_fmadd_pd(r, denominator,  _mm512_set1_pd(0.59983220655588793769));
  denominator = _mm512_fmadd_pd(r, denominator,  _mm512_set1_pd(1.0));
  
  ALIGN64  __m512d const val = _mm512_div_pd(numerator, denominator);
  
  return val;
  
  
} 
 


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



ALWAYS_INLINE __m512d nico_inv_Phi_wo_checks_case_2_AVX512( const __m512d p, 
                                                            const __m512d q) {

        ALIGN64  __m512d const p_aligned = p;
        ALIGN64  __m512d const q_aligned = q;
  
        ALIGN64  __m512d const zero = _mm512_setzero_pd(), one = _mm512_set1_pd(1.0);
        const __mmask8 q_pos = _mm512_cmp_pd_mask(q_aligned, zero, _CMP_GT_OQ);
        // q>0 -> 1-p, else p
        ALIGN64  __m512d r = _mm512_mask_blend_pd(q_pos, p_aligned, _mm512_sub_pd(one, p_aligned));
        r = _mm512_sqrt_pd(_mm512_sub_pd(zero, nico_log_1_double_precision_AVX512(r)));
        const __mmask8 r_gr_5 = _mm512_cmp_pd_mask(r, _mm512_set1_pd(5.0), _CMP_GT_OQ);
        ALIGN64  __m512d val = nico_inv_Phi_wo_checks_case_2a_AVX512(p_aligned, r);                // r <= 5
        // No need to evaluate the extreme-tail polynomial unless a lane will actually use it.
        if (r_gr_5 != 0) val = _mm512_mask_blend_pd(r_gr_5, val, nico_inv_Phi_wo_checks_case_2b_AVX512(p_aligned,
                r));
        const __mmask8 q_neg = _mm512_cmp_pd_mask(q_aligned, zero, _CMP_LT_OQ);
        return _mm512_mask_blend_pd(q_neg, val, _mm512_sub_pd(zero, val));          // q<0 -> -val
  
}



ALWAYS_INLINE __m512d nico_inv_Phi_wo_checks_case_1_AVX512( const __m512d p,
                                                            const __m512d q) {

        ALIGN64  __m512d const q_aligned = q; ///  CASE 1: if abs(q) <= 0.425

        ALIGN64  __m512d const q_sq = _mm512_mul_pd(q_aligned, q_aligned);
        ALIGN64  __m512d const r = _mm512_sub_pd(_mm512_set1_pd(0.180625), q_sq);

        ALIGN64  __m512d numerator =
                _mm512_fmadd_pd(r, _mm512_set1_pd(2509.0809287301226727), _mm512_set1_pd(33430.575583588128105));
        numerator = _mm512_fmadd_pd(numerator, r, _mm512_set1_pd(67265.770927008700853));
        numerator = _mm512_fmadd_pd(numerator, r, _mm512_set1_pd(45921.953931549871457));
        numerator = _mm512_fmadd_pd(numerator, r, _mm512_set1_pd(13731.693765509461125));
        numerator = _mm512_fmadd_pd(numerator, r, _mm512_set1_pd(1971.5909503065514427));
        numerator = _mm512_fmadd_pd(numerator, r, _mm512_set1_pd(133.14166789178437745));
        numerator = _mm512_fmadd_pd(numerator, r, _mm512_set1_pd(3.387132872796366608));

        ALIGN64  __m512d denominator =
                _mm512_fmadd_pd(r, _mm512_set1_pd(5226.495278852854561), _mm512_set1_pd(28729.085735721942674));
        denominator = _mm512_fmadd_pd(denominator, r, _mm512_set1_pd(39307.89580009271061));
        denominator = _mm512_fmadd_pd(denominator, r, _mm512_set1_pd(21213.794301586595867));
        denominator = _mm512_fmadd_pd(denominator, r, _mm512_set1_pd(5394.1960214247511077));
        denominator = _mm512_fmadd_pd(denominator, r, _mm512_set1_pd(687.1870074920579083));
        denominator = _mm512_fmadd_pd(denominator, r, _mm512_set1_pd(42.313330701600911252));
        denominator = _mm512_fmadd_pd(denominator, r,  _mm512_set1_pd(1.0));

        ALIGN64  __m512d val = _mm512_div_pd(numerator, denominator);
        val = _mm512_mul_pd(q_aligned, val);

        return val;

}


ALWAYS_INLINE __m512d nico_inv_Phi_wo_checks_AVX512( const __m512d p) {

        ALIGN64  __m512d const p_aligned = p;
  
        ALIGN64  __m512d const q = _mm512_sub_pd(p_aligned, _mm512_set1_pd(0.5));
        const __mmask8 central = _mm512_cmp_pd_mask(_mm512_abs_pd(q), _mm512_set1_pd(0.425), _CMP_LE_OQ);
        ALIGN64  __m512d const v1 = nico_inv_Phi_wo_checks_case_1_AVX512(p_aligned, q);
        // The tail result would be discarded in every lane. Avoid its log and two rational polynomials.
        if (central == 0xff) return v1;
        ALIGN64  __m512d const v2 = nico_inv_Phi_wo_checks_case_2_AVX512(p_aligned, q);
        return _mm512_mask_blend_pd(central, v2, v1); // central lanes -> v1
  
}

 

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




ALWAYS_INLINE __m512d nico_mills_ratio_tail_AVX512(const __m512d x) {

      ALIGN64  __m512d const x_aligned = x;              // x >= NORMAL_TAIL_THRESH
  
      ALIGN64  __m512d f = x_aligned;
      for (int k = MILLS_CF_DEPTH; k >= 1; --k) {
          f = _mm512_add_pd(x_aligned, _mm512_div_pd(_mm512_set1_pd(static_cast<double>(k)), f));
      }
      return _mm512_div_pd(_mm512_set1_pd(1.0), f);
      
}




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
//  (run normal_tail_self_test() once; lower the depth after benchmarking if desired - the tail
//  path is only taken for |x| > NORMAL_TAIL_THRESH, and a rational minimax in 1/x can replace the
//  fraction later; the self-test validates any replacement).
//
//  Body (|x| <= NORMAL_TAIL_THRESH) keeps the existing nico_Phi (Abramowitz-Stegun). Its absolute
//  error 7.5e-8 is a relative error of 1.2e-5 at the threshold and smaller inside; the tail side is
//  exact, so the join is continuous to ~1e-5 in log Phi. The remaining tail error is the nico_log's
//  (~1e-8); swap in the double-precision log constant if machine precision is required.
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
ALWAYS_INLINE __m512d nico_inverse_mills_ratio_tail_AVX512(const __m512d z) {

        ALIGN64  __m512d const z_aligned = z;

        ALIGN64  __m512d const t = _mm512_div_pd(_mm512_set1_pd(1.0), _mm512_mul_pd(z_aligned, z_aligned));
        ALIGN64  __m512d d = _mm512_set1_pd(LOG_PHI_TAIL_MILLS_NUMERATOR_MINUS_DENOMINATOR_COEFFICIENTS[8]);
        ALIGN64  __m512d p = _mm512_set1_pd(LOG_PHI_TAIL_MILLS_NUMERATOR_COEFFICIENTS[8]);
        d = _mm512_fmadd_pd(d, t, _mm512_set1_pd(LOG_PHI_TAIL_MILLS_NUMERATOR_MINUS_DENOMINATOR_COEFFICIENTS[7]));
        p = _mm512_fmadd_pd(p, t, _mm512_set1_pd(LOG_PHI_TAIL_MILLS_NUMERATOR_COEFFICIENTS[7]));
        d = _mm512_fmadd_pd(d, t, _mm512_set1_pd(LOG_PHI_TAIL_MILLS_NUMERATOR_MINUS_DENOMINATOR_COEFFICIENTS[6]));
        p = _mm512_fmadd_pd(p, t, _mm512_set1_pd(LOG_PHI_TAIL_MILLS_NUMERATOR_COEFFICIENTS[6]));
        d = _mm512_fmadd_pd(d, t, _mm512_set1_pd(LOG_PHI_TAIL_MILLS_NUMERATOR_MINUS_DENOMINATOR_COEFFICIENTS[5]));
        p = _mm512_fmadd_pd(p, t, _mm512_set1_pd(LOG_PHI_TAIL_MILLS_NUMERATOR_COEFFICIENTS[5]));
        d = _mm512_fmadd_pd(d, t, _mm512_set1_pd(LOG_PHI_TAIL_MILLS_NUMERATOR_MINUS_DENOMINATOR_COEFFICIENTS[4]));
        p = _mm512_fmadd_pd(p, t, _mm512_set1_pd(LOG_PHI_TAIL_MILLS_NUMERATOR_COEFFICIENTS[4]));
        d = _mm512_fmadd_pd(d, t, _mm512_set1_pd(LOG_PHI_TAIL_MILLS_NUMERATOR_MINUS_DENOMINATOR_COEFFICIENTS[3]));
        p = _mm512_fmadd_pd(p, t, _mm512_set1_pd(LOG_PHI_TAIL_MILLS_NUMERATOR_COEFFICIENTS[3]));
        d = _mm512_fmadd_pd(d, t, _mm512_set1_pd(LOG_PHI_TAIL_MILLS_NUMERATOR_MINUS_DENOMINATOR_COEFFICIENTS[2]));
        p = _mm512_fmadd_pd(p, t, _mm512_set1_pd(LOG_PHI_TAIL_MILLS_NUMERATOR_COEFFICIENTS[2]));
        d = _mm512_fmadd_pd(d, t, _mm512_set1_pd(LOG_PHI_TAIL_MILLS_NUMERATOR_MINUS_DENOMINATOR_COEFFICIENTS[1]));
        p = _mm512_fmadd_pd(p, t, _mm512_set1_pd(LOG_PHI_TAIL_MILLS_NUMERATOR_COEFFICIENTS[1]));
        d = _mm512_mul_pd(d, t);                                                            // D(t), D(0) = 0
        p = _mm512_fmadd_pd(p, t, _mm512_set1_pd(LOG_PHI_TAIL_MILLS_NUMERATOR_COEFFICIENTS[0]));     // P(t)
        ALIGN64  __m512d const z_minus_z_w = _mm512_fnmadd_pd(z_aligned, _mm512_div_pd(d, p), z_aligned);
        return _mm512_mask_blend_pd(_mm512_cmp_pd_mask(z_aligned, _mm512_set1_pd(INFINITY), _CMP_EQ_OQ),
                z_minus_z_w, z_aligned);

}




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
ALWAYS_INLINE __m512d nico_log_Phi_AVX512(const __m512d x) {

      ALIGN64  __m512d const x_aligned = x;

      ALIGN64  __m512d const thresh   = _mm512_set1_pd(NORMAL_TAIL_THRESH);
      // x < -2.5
      const __mmask8 is_tail  = _mm512_cmp_pd_mask(x_aligned, _mm512_sub_pd(_mm512_setzero_pd(), thresh),
              _CMP_LT_OQ);
      // no tail lane: body only
      if (is_tail == 0) return nico_log_1_double_precision_AVX512(nico_Phi_AVX512(x_aligned));
      ALIGN64  __m512d const ax       = _mm512_abs_pd(x_aligned);
      ALIGN64  __m512d const R        = _mm512_div_pd(_mm512_set1_pd(1.0),
                                              nico_inverse_mills_ratio_tail_AVX512(_mm512_max_pd(ax, thresh)));
      ALIGN64  __m512d const log_tail = _mm512_add_pd(_mm512_fnmadd_pd(_mm512_set1_pd(0.5), _mm512_mul_pd(
              x_aligned, x_aligned),
                                                               _mm512_set1_pd(NEG_HALF_LOG_2PI)),
                                              nico_log_1_double_precision_AVX512(R));
      if (is_tail == 0xff) return log_tail;                                            // every lane in the tail
      ALIGN64  __m512d const log_body = nico_log_1_double_precision_AVX512(nico_Phi_AVX512(x_aligned));
      return _mm512_mask_blend_pd(is_tail, log_body, log_tail);

}




ALWAYS_INLINE __m512d nico_log_1m_Phi_AVX512(const __m512d x) {

      ALIGN64  __m512d const x_aligned = x;                     // log(1 - Phi(x)) = log Phi(-x)
  
      return nico_log_Phi_AVX512(_mm512_sub_pd(_mm512_setzero_pd(), x_aligned));
  
}




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
ALWAYS_INLINE __m512d nico_dlog_Phi_dx_AVX512(const __m512d x) {

      ALIGN64  __m512d const x_aligned = x;

      ALIGN64  __m512d const thresh  = _mm512_set1_pd(NORMAL_TAIL_THRESH);
      // x < -2.5
      const __mmask8 is_tail = _mm512_cmp_pd_mask(x_aligned, _mm512_sub_pd(_mm512_setzero_pd(), thresh),
              _CMP_LT_OQ);
      if (is_tail == 0) {                                                              // no tail lane: body only
          ALIGN64  __m512d const phi = _mm512_mul_pd(_mm512_set1_pd(INV_SQRT_2PI),
                                            nico_exp_1_AVX512(_mm512_mul_pd(_mm512_set1_pd(-0.5),
                                                                            _mm512_mul_pd(x_aligned,
                                                                                    x_aligned))));
          return _mm512_div_pd(phi, nico_Phi_AVX512(x_aligned));
      }
      ALIGN64  __m512d const ax      = _mm512_abs_pd(x_aligned);
      ALIGN64  __m512d const d_tail  = nico_inverse_mills_ratio_tail_AVX512(_mm512_max_pd(ax, thresh));
      if (is_tail == 0xff) return d_tail;                                              // every lane in the tail
      ALIGN64  __m512d const phi     = _mm512_mul_pd(_mm512_set1_pd(INV_SQRT_2PI),
                                             nico_exp_1_AVX512(_mm512_mul_pd(_mm512_set1_pd(-0.5), _mm512_mul_pd(
                                                     x_aligned, x_aligned))));
      ALIGN64  __m512d const d_body  = _mm512_div_pd(phi, nico_Phi_AVX512(x_aligned));
      return _mm512_mask_blend_pd(is_tail, d_body, d_tail);

}
// d/dx log(1 - Phi(x)) = -nico_dlog_Phi_dx_AVX512(-x)




// inv_Phi given log(p) and log(1-p). AS241; tail case fed r = sqrt(-min(log_p, log_1m_p)) directly.
ALWAYS_INLINE __m512d nico_inv_Phi_from_log_p_wo_checks_AVX512(const __m512d log_p,
                                                               const __m512d log_1m_p) {

      ALIGN64  __m512d const log_p_aligned = log_p;
      ALIGN64  __m512d const log_1m_p_aligned = log_1m_p;
  
      ALIGN64  __m512d const zero    = _mm512_setzero_pd();
      // only consumed by the central case
      ALIGN64  __m512d const p       = nico_exp_1_AVX512(log_p_aligned);
      ALIGN64  __m512d const q       = _mm512_sub_pd(p, _mm512_set1_pd(0.5));
      const __mmask8 central = _mm512_cmp_pd_mask(_mm512_abs_pd(q), _mm512_set1_pd(0.425), _CMP_LE_OQ);
      //// tail
      // log(min(p, 1-p))
      ALIGN64  __m512d const log_min = _mm512_min_pd(log_p_aligned, log_1m_p_aligned);
      ALIGN64  __m512d const r       = _mm512_sqrt_pd(_mm512_sub_pd(zero, log_min));
      const __mmask8 r_gr_5  = _mm512_cmp_pd_mask(r, _mm512_set1_pd(5.0), _CMP_GT_OQ);
      ALIGN64  __m512d v_tail = _mm512_mask_blend_pd(r_gr_5,
                                            nico_inv_Phi_wo_checks_case_2a_AVX512(p, r),   // r <= 5
                                            nico_inv_Phi_wo_checks_case_2b_AVX512(p, r));  // r >  5
      // p < 0.5 -> Z < 0
      const __mmask8 lower   = _mm512_cmp_pd_mask(log_p_aligned, log_1m_p_aligned, _CMP_LT_OQ);
      v_tail = _mm512_mask_blend_pd(lower, v_tail, _mm512_sub_pd(zero, v_tail));
      //// central
      ALIGN64  __m512d const v_central = nico_inv_Phi_wo_checks_case_1_AVX512(p, q);
      return _mm512_mask_blend_pd(central, v_tail, v_central);
      
}




// dZ/dv through the stable side (see scalar version for the identities).
ALWAYS_INLINE __m512d nico_dZ_dv_from_log_p_AVX512( const __m512d Z,
                                                    const __m512d log_p, 
                                                    const __m512d log_1m_p,
                                                    const __m512d dlog_p_dv, 
                                                    const __m512d dlog_1m_p_dv) {

      ALIGN64  __m512d const Z_aligned = Z;
      ALIGN64  __m512d const log_p_aligned = log_p;
      ALIGN64  __m512d const log_1m_p_aligned = log_1m_p;
      ALIGN64  __m512d const dlog_p_dv_aligned = dlog_p_dv;
      ALIGN64  __m512d const dlog_1m_p_dv_aligned = dlog_1m_p_dv;
  
      ALIGN64  __m512d const log_inv_phi =
              _mm512_fmadd_pd(_mm512_set1_pd(0.5), _mm512_mul_pd(Z_aligned, Z_aligned), _mm512_set1_pd(
                      -NEG_HALF_LOG_2PI));
      // keeps the UNUSED side finite
      ALIGN64  __m512d const cap    = _mm512_set1_pd(700.0);
      ALIGN64  __m512d const e_lo   = _mm512_min_pd(_mm512_add_pd(log_p_aligned,    log_inv_phi), cap);
      ALIGN64  __m512d const e_hi   = _mm512_min_pd(_mm512_add_pd(log_1m_p_aligned, log_inv_phi), cap);
      ALIGN64  __m512d const via_lo = _mm512_mul_pd(nico_exp_1_AVX512(e_lo), dlog_p_dv_aligned);
      ALIGN64  __m512d const via_hi =
              _mm512_mul_pd(_mm512_sub_pd(_mm512_setzero_pd(), nico_exp_1_AVX512(e_hi)), dlog_1m_p_dv_aligned);
      const __mmask8 Z_neg  = _mm512_cmp_pd_mask(Z_aligned, _mm512_setzero_pd(), _CMP_LT_OQ);
      return _mm512_mask_blend_pd(Z_neg,
                                  via_hi,
                                  via_lo); // Z<0 -> via_lo, else via_hi
      
}


}  // namespace nico::math

using nico::math::test_simple_AVX512;
using nico::math::is_NaN_mask;
using nico::math::is_not_NaN_mask;
using nico::math::is_finite_mask;
using nico::math::is_infinity_mask;
using nico::math::is_positive_mask;
using nico::math::is_negative_mask;
using nico::math::nico_ldexp;
using nico::math::nico_exp_1_wo_checks_AVX512;
using nico::math::nico_exp_exceptional_lanes_AVX512;
using nico::math::nico_exp_1_AVX512;
// using nico::math::nico_log_1_wo_checks_AVX512;
using nico::math::nico_log_reduce_AVX512;
using nico::math::nico_log_fdlibm_core_AVX512;
using nico::math::nico_log_1_double_precision_wo_checks_AVX512;
// using nico::math::nico_log1p_1_wo_checks_AVX512;
using nico::math::nico_log1p_1_double_precision_wo_checks_AVX512;
using nico::math::nico_log1m_1_wo_checks_AVX512;
using nico::math::nico_log1p_exp_1_wo_checks_AVX512;
using nico::math::nico_log_exceptional_lanes_AVX512;
// using nico::math::nico_log_1_AVX512;
using nico::math::nico_log_1_double_precision_AVX512;
// using nico::math::nico_log1p_1_AVX512;
using nico::math::nico_log1p_exceptional_lanes_AVX512;
using nico::math::nico_log1p_1_double_precision_AVX512;
using nico::math::nico_log1m_1_AVX512;
using nico::math::nico_log1p_exp_1_AVX512;
using nico::math::nico_log1m_exp_1_AVX512;
using nico::math::nico_logit_wo_checks_AVX512;
using nico::math::nico_logit_AVX512;
using nico::math::nico_inv_logit_for_x_pos_AVX512;
using nico::math::nico_inv_logit_for_x_neg_AVX512;
using nico::math::nico_inv_logit_AVX512;
using nico::math::nico_inv_logit_for_x_pos_wo_checks_AVX512;
using nico::math::nico_inv_logit_for_x_neg_wo_checks_AVX512;
using nico::math::nico_inv_logit_wo_checks_AVX512;
using nico::math::nico_log_inv_logit_for_x_pos_AVX512;
using nico::math::nico_log_inv_logit_for_x_neg_AVX512;
using nico::math::nico_log_inv_logit_AVX512;
using nico::math::nico_log_inv_logit_for_x_pos_wo_checks_AVX512;
using nico::math::nico_log_inv_logit_for_x_neg_wo_checks_AVX512;
using nico::math::nico_log_inv_logit_wo_checks_AVX512;
using nico::math::nico_Phi_approx_wo_checks_AVX512;
using nico::math::nico_Phi_approx_AVX512;
using nico::math::nico_inv_Phi_approx_wo_checks_AVX512;
using nico::math::nico_inv_Phi_approx_AVX512;
using nico::math::nico_inv_Phi_approx_from_logit_prob_wo_checks_AVX512;
using nico::math::nico_inv_Phi_approx_from_logit_prob_AVX512;
using nico::math::nico_log_Phi_approx_wo_checks_AVX512;
using nico::math::nico_log_Phi_approx_AVX512;
using nico::math::nico_tanh_AVX512;
using nico::math::nico_tanh_wo_checks_AVX512;
using nico::math::nico_copy_sign;
using nico::math::nico_Phi_wo_checks_AVX512;
using nico::math::nico_Phi_AVX512;
using nico::math::nico_inv_Phi_wo_checks_case_2a_AVX512;
using nico::math::nico_inv_Phi_wo_checks_case_2b_AVX512;
using nico::math::nico_inv_Phi_wo_checks_case_2_AVX512;
using nico::math::nico_inv_Phi_wo_checks_case_1_AVX512;
using nico::math::nico_inv_Phi_wo_checks_AVX512;
using nico::math::nico_mills_ratio_tail_AVX512;
using nico::math::nico_inverse_mills_ratio_tail_AVX512;
using nico::math::nico_log_Phi_AVX512;
using nico::math::nico_log_1m_Phi_AVX512;
using nico::math::nico_dlog_Phi_dx_AVX512;
using nico::math::nico_inv_Phi_from_log_p_wo_checks_AVX512;
using nico::math::nico_dZ_dv_from_log_p_AVX512;

#endif
 
 
 
#endif






















