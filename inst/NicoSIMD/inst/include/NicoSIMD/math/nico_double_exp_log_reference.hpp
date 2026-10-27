#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <math.h>
#include <sstream>
#include <stdexcept>
#include <string.h>
#include "nico_double_exp_log.hpp"

namespace nico::math::reference {

double __int_as_double (int64_t a) {  double r; memcpy (&r, &a, sizeof r); return r;}

int64_t __double_as_int (double a) { int64_t r; memcpy (&r, &a, sizeof r); return r;}

inline double nico_ldexp(const double a, 
                         const int i) {

  int64_t ia = ((uint64_t)i << 52ULL) + __double_as_int(a);
  return __int_as_double(ia);

}

inline double nico_ldexp_2(const double a, 
                           const int i) {

  // In IEEE 754 double precision format:
  // Bits 0-51:  Fraction (mantissa)
  // Bits 52-62: Exponent (11 bits)
  // Bit 63:     Sign
  
  // First attempt: try to directly manipulate the exponent bits
  // Shifting i by 52 positions puts it in the exponent field
  int64_t ia = ((uint64_t)i << 52ULL) + __double_as_int(a);
  
  // Check if |i| is too large to prevent exponent overflow
  // For IEEE 754 doubles:
  // - Minimum exponent is -1022 (subnormal numbers)
  // - Maximum exponent is +1023 (before infinity)
  // We use ±1000 as a safe threshold
  if ((unsigned int)(i + 1000) > 2000) { // This checks if |i| > 1000
    // If |i| is too large, split the scaling into two steps
    // to avoid exponent overflow/underflow
     
    // Scale by +/- 1000 (depending on sign of i)
    int i1 = (i < 0) ? -1000 : 1000;
    // Scale by the remaining amount
    int i2 = i - i1;
    
    // Perform the two-step scaling
    // first scale by i1
    int64_t step1 = ((uint64_t)i1 << 52ULL) + __double_as_int(a);
    double mid = __int_as_double(step1);
    
    // then scale by i2
    // int64_t step2 = ((uint64_t)i2 << 52ULL) + __double_as_int(mid);
    // return __int_as_double(step2);
    return nico_ldexp(mid, i2);
    
  }
  
  
  return nico_ldexp(a, i); // if doesnt exceed threshold just use 'standard' nico_ldexp()
  
}

inline double nico_exp_1_wo_checks (const double x)  {
  
  const double input = x;
  const double l2e =   (1.442695040888963387); /* log2(e) */
  const double l2h =   (-0.693145751999999948367); /* -log(2)_hi */
  const double l2l =   (-0.00000142860676999999996193); /* -log(2)_lo */

  // /* coefficients for core approximation to exp() in [-log(2)/2, log(2)/2] */
   // // ///// 9-degree approx:
   const double c0 =     (0.00000276479776161191821278);
   const double c1 =     (0.0000248844480527491290235);
   const double c2 =     (0.000198411488032534342194);
   const double c3 =     (0.00138888017711994078175);
   const double c4 =     (0.00833333340524595143906);
   const double c5 =     (0.0416666670404215802592);
   const double c6 =     (0.166666666664891632843);
   const double c7 =     (0.499999999994389376923);
   const double c8 =     (1.00000000000001221245);
   const double c9 =     (1.00000000000001332268);
   
  /* exp(x) = 2^i * e^f; i = rint (log2(e) * x), f = x - log(2) * i */
  const double  t = x*l2e ; // _mm512_mul_pd (x, l2e);      /* t = log2(e) * x */
  const int i = (int)rint(t) ; // i = _mm512_cvtpd_epi64(t);       /* i = (int)rint(t) */
  const double x_2 = std::round(t) ; //  _mm512_roundscale_pd(t,((0<<4)| _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC|_MM_FROUND_NO_EXC));
  const double f = std::fma (x_2, l2l, std::fma(x_2, l2h, input)); /* x - log(2)_hi * r */    /* f = x - log(2)_hi * r - log(2)_lo * r */
  
  /* p ~= exp (f), -log(2)/2 <= f <= log(2)/2 */
  double  p;
  p = c0;
  p = std::fma (p, f, c1);
  p = std::fma (p, f, c2);
  p = std::fma (p, f, c3);
  p = std::fma (p, f, c4);
  p = std::fma (p, f, c5);
  p = std::fma (p, f, c6);
  p = std::fma (p, f, c7);
  p = std::fma (p, f, c8);
  p = std::fma (p, f, c9);
  
  return nico_ldexp_2(p, i) ;
  
}

inline double nico_exp_1(const double a)    {

  if (std::fabs (a) < 708.4) {
    return  nico_exp_1_wo_checks(a);
  } else {
   // const double large =  8.98846567431157953864652595394512366808e307;
    double r = INFINITY - INFINITY; //  // this handles NaNs
    if (a < 0.0) r = 0.0;
    if (a > 0.0) r = INFINITY; // + INF
    return r;
  }

}

// single-precision log, commented out: every caller uses the double-precision kernel
// inline  double nico_log_1_wo_checks (const double a)  {
//
//         double  r,  t;
//
//         const int64_t e = (__double_as_int (a) - 0x3fe5555555555555 )    &   0xFFF0000000000000   ;
//         const double i = fma ((double)e, 0.000000000000000222044604925031308085, 0.0); // 0x1.0p-52
//         const double m = __int_as_double (__double_as_int (a) - e) - 1.0 ;  /* m in [2/3, 4/3] */
//         const double s = m * m;
//
//         /* Compute log1p(m) for m in [-1/3, 1/3] */
//         r =             -0.13031005859375;  // -0x1.0ae000p-3
//         t =              0.140869140625;  //  0x1.208000p-3
//         r = fma (r, s, -0.121483512222766876221); // -0x1.f198b2p-4
//         t = fma (t, s,  0.139814853668212890625); //  0x1.1e5740p-3
//         r = fma (r, s, -0.166846126317977905273); // -0x1.55b36cp-3
//         t = fma (t, s,  0.200120344758033752441); //  0x1.99d8b2p-3
//         r = fma (r, s, -0.249996200203895568848); // -0x1.fffe02p-3
//         r = fma (t, m, r);
//         r = fma (r, m,  0.333331972360610961914); //  0x1.5554fap-2
//         r = fma (r, m, -0.5); // -0x1.000000p-1
//         r = fma (r, s, m);
//
//         return fma(i,  0.693147182464599609375, r); //  0x1.62e430p-1 // log(2)
//
// }

// single-precision log, commented out: every caller uses the double-precision kernel
// inline   double nico_log1p_1_wo_checks(const double x)   {
//
//   if (fabs(x) > 1e-4)   return nico_log_1_wo_checks(1.0 + x);     // x is large enough that the obvious evaluation is OK
//   return (-0.5*x + 1.0)*x;     // Use Taylor approx. log(1 + x) = x - x^2/2 with error roughly x^3/3     // Since |x| < 10^-4, |x|^3 < 10^-12, relative error less than 10^-8
//
// }

inline   double nico_log1m_1_wo_checks(const double x)   {
  
  return nico_log1p_1_double_precision_wo_checks(-x);
  
}

inline   double nico_log1p_exp_1_wo_checks(const double x)   {
  
  if (x > 0.0) return x + nico_log1p_1_double_precision_wo_checks(nico_exp_1_wo_checks(-x));
  return nico_log1p_1_double_precision_wo_checks(nico_exp_1_wo_checks(x));
  
}

// single-precision log, commented out: every caller uses the double-precision kernel
// inline   double nico_log_1  (const double a)  {
//
//       if (!((a > 0.0) && (a < INFINITY))) {
//
//         double r = a + a;  // silence NaNs if necessary
//         if (a  < 0.0) r =  INFINITY - INFINITY; //  NaN
//         if (a == 0.0) r = -INFINITY;
//         return r;
//
//       }
//
//       return  nico_log_1_wo_checks(a);
//
// }

// single-precision log, commented out: every caller uses the double-precision kernel
// inline   double nico_log1p_1(const double x)   {
//
//   if (fabs(x) > 1e-4)   return nico_log_1(1.0 + x);     // x is large enough that the obvious evaluation is OK
//
//   if (x <= -1.0)   {
//
//     std::stringstream os;
//     os << "Invalid input argument (" << x
//        << "); must be greater than -1.0";
//     throw std::invalid_argument( os.str() );
//
//   }
//
//   return (-0.5*x + 1.0)*x;  // Use Taylor approx. log(1 + x) = x - x^2/2 with error roughly x^3/3     // Since |x| < 10^-4, |x|^3 < 10^-12, relative error less than 10^-8
//
// }

inline  double nico_log1m_1(const double x)   {
  
  return nico_log1p_1_double_precision(-x);
  
}

inline   double nico_log1p_exp_1(const double x)   {
  
  if (x > 0.0) return x + nico_log1p_1_double_precision(nico_exp_1(-x));
  return nico_log1p_1_double_precision(nico_exp_1(x));
  
}

inline  double  nico_inv_logit_for_x_pos(const double x )  {

  return     (1.0 /  (1.0 + nico_exp_1(-x)))  ;

}

inline  double  nico_inv_logit_for_x_neg(const double x )  {

  const double log_eps = -18.420680743952367;
  const double   exp_x =  nico_exp_1(x) ;
  if (x > log_eps) return  exp_x / (1.0 + exp_x);
  return  exp_x;

}

inline  double  nico_inv_logit(const double x )  {
  
  double val;
  
  if (x > 0.0) { 
    val =   nico_inv_logit_for_x_pos(x);
  } else { 
    val =  nico_inv_logit_for_x_neg(x);
  }
 
  if  ((val > 0) && (val < 1))  { 
    return val;
  } else if (val > 1) { 
    return 1;
  } else { 
    return 0;
  }
  
}

inline  double  nico_inv_logit_for_x_pos_wo_checks(const double x )  {

  double     exp_m_x =  nico_exp_1_wo_checks(-x) ;
  return     (1.0 /  (1.0 + exp_m_x))  ;

}

inline   double  nico_inv_logit_for_x_neg_wo_checks(const double x )  {

  double log_eps = -18.420680743952367;
  double   exp_x =  nico_exp_1_wo_checks(x) ;
  if (x > log_eps) return   exp_x / (1.0 + exp_x);
  return  exp_x;

}

inline   double  nico_inv_logit_wo_checks(const double x )  {

  if (x > 0) return  nico_inv_logit_for_x_pos_wo_checks(x);
  return  nico_inv_logit_for_x_neg_wo_checks(x);
  
  double val;
  
  if (x > 0.0) { 
    val =   nico_inv_logit_for_x_pos_wo_checks(x);
  } else { 
    val =  nico_inv_logit_for_x_neg_wo_checks(x);
  }
  
  if  ((val > 0) && (val < 1))  { 
    return val;
  } else if (val > 1) {  
    return 1;
  } else {  
    return 0;
  }
  
}

inline   double  nico_log_inv_logit_for_x_pos(const double x )  {

  return    - nico_log1p_exp_1(-x);  // return    - nico_log1p_1_double_precision(nico_exp_1(-x));

}

inline   double  nico_log_inv_logit_for_x_neg(const double x )  {

  const double log_eps = -18.420680743952367;
  if (x > log_eps) return  x - nico_log1p_1_double_precision(nico_exp_1(x));
  return x;

}

inline   double  nico_log_inv_logit(const double x )  {

  if (x > 0.0) return  nico_log_inv_logit_for_x_pos(x);
  return  nico_log_inv_logit_for_x_neg(x);

}

inline   double  nico_log_inv_logit_for_x_pos_wo_checks(const double x )  {

   return    - nico_log1p_exp_1_wo_checks((-x));

}

inline   double  nico_log_inv_logit_for_x_neg_wo_checks(const double x )  {

  const double log_eps = -18.420680743952367;
  if (x > log_eps) return x - nico_log1p_1_double_precision_wo_checks(nico_exp_1_wo_checks(x));
  return x;

}

inline   double  nico_log_inv_logit_wo_checks(const double x )  {

  if (x > 0.0) return  nico_log_inv_logit_for_x_pos_wo_checks(x);
  return  nico_log_inv_logit_for_x_neg_wo_checks(x);

}

inline   double    nico_tanh(const   double x  )   {

  return    ( 2.0 / (1.0 + nico_exp_1(-2.0*x   )  ) - 1.0 )  ;

}

inline  double    nico_tanh_wo_checks(const   double x  )   {

  return    ( 2.0 / (1.0 + nico_exp_1_wo_checks(-2.0*x   )  ) - 1.0 )  ;

}

}






















