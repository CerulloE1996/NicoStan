#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <math.h>
#include <sstream>
#include <stdexcept>

#include <stan/math/prim/fun/abs.hpp>
#include <stan/math/prim/fun/sqrt.hpp>

#include "nico_double_exp_log_reference.hpp"

namespace nico::math::reference {

static constexpr double NORMAL_TAIL_THRESH  = 2.5;

static constexpr int    MILLS_CF_DEPTH      = 40;

static constexpr double NEG_HALF_LOG_2PI    = -0.91893853320467274178;   // -0.5*log(2*pi)

static constexpr double INV_SQRT_2PI        =  0.39894228040143267794;   //  1/sqrt(2*pi)

inline double nico_mills_ratio_tail(const double x) {                 // x >= NORMAL_TAIL_THRESH
  
      double f = x;
      for (int k = MILLS_CF_DEPTH; k >= 1; --k) f = x + static_cast<double>(k) / f;
      return 1.0 / f;
      
}

inline double nico_log_Phi_tail_only(const double x) {                // x < -NORMAL_TAIL_THRESH

      return -0.5 * x * x + NEG_HALF_LOG_2PI + std::log(nico_mills_ratio_tail(-x));

}

inline double nico_log_Phi(const double x) {
  
      if (x < -NORMAL_TAIL_THRESH) return nico_log_Phi_tail_only(x);
      return std::log(0.5 * std::erfc(-x * M_SQRT1_2));
      
}

inline double nico_log_1m_Phi(const double x) { return nico_log_Phi(-x); }

inline double nico_dlog_Phi_dx(const double x) {                      // phi(x)/Phi(x)
  
      if (x < -NORMAL_TAIL_THRESH) return 1.0 / nico_mills_ratio_tail(-x);
      return INV_SQRT_2PI * std::exp(-0.5 * x * x) / (0.5 * std::erfc(-x * M_SQRT1_2));
      
}

inline double nico_inv_Phi_case_1(const double q) {                   // |q| <= 0.425
  
      const double r = 0.180625 - q * q;
      double num = 2509.0809287301226727 * r + 33430.575583588128105;
      num = num * r + 67265.770927008700853;  num = num * r + 45921.953931549871457;
      num = num * r + 13731.693765509461125;  num = num * r + 1971.5909503065514427;
      num = num * r + 133.14166789178437745;  num = num * r + 3.387132872796366608;
      double den = 5226.495278852854561 * r + 28729.085735721942674;
      den = den * r + 39307.89580009271061;   den = den * r + 21213.794301586595867;
      den = den * r + 5394.1960214247511077;  den = den * r + 687.1870074920579083;
      den = den * r + 42.313330701600911252;  den = den * r + 1.0;
      return q * num / den;
      
}

inline double nico_inv_Phi_case_2a(double r) {                        // r <= 5
  
      r -= 1.60;
      double num = 0.00077454501427834140764;
      num = r * num + 0.0227238449892691845833; num = r * num + 0.24178072517745061177;
      num = r * num + 1.27045825245236838258;   num = r * num + 3.64784832476320460504;
      num = r * num + 5.7694972214606914055;    num = r * num + 4.6303378461565452959;
      num = r * num + 1.42343711074968357734;
      double den = 0.00000000105075007164441684324;
      den = r * den + 0.0005475938084995344946; den = r * den + 0.0151986665636164571966;
      den = r * den + 0.14810397642748007459;   den = r * den + 0.68976733498510000455;
      den = r * den + 1.6763848301838038494;    den = r * den + 2.05319162663775882187;
      den = r * den + 1.0;
      return num / den;
      
}

inline double nico_inv_Phi_case_2b(double r) {                        // r > 5
  
      r -= 5.0;
      double num = 0.000000201033439929228813265;
      num = r * num + 0.0000271155556874348757815; num = r * num + 0.0012426609473880784386;
      num = r * num + 0.026532189526576123093;     num = r * num + 0.29656057182850489123;
      num = r * num + 1.7848265399172913358;       num = r * num + 5.4637849111641143699;
      num = r * num + 6.6579046435011037772;
      double den = 0.00000000000000204426310338993978564;
      den = r * den + 0.00000014215117583164458887; den = r * den + 0.000018463183175100546818;
      den = r * den + 0.0007868691311456132591;     den = r * den + 0.0148753612908506148525;
      den = r * den + 0.13692988092273580531;       den = r * den + 0.59983220655588793769;
      den = r * den + 1.0;
      return num / den;
      
}

inline double nico_inv_Phi_from_log_p(const double log_p, 
                                      const double log_1m_p) {
  
      const double p = std::exp(log_p);                                 // only consumed by the central case
      const double q = p - 0.5;
      if (std::fabs(q) <= 0.425) return nico_inv_Phi_case_1(q);
      const double r = std::sqrt(-std::min(log_p, log_1m_p));           // sqrt(-log(min(p, 1-p)))
      const double v = (r <= 5.0) ? nico_inv_Phi_case_2a(r) : nico_inv_Phi_case_2b(r);
      return (log_p < log_1m_p) ? -v : v;                                // p < 0.5 -> Z < 0
      
}

inline double nico_dZ_dv_from_log_p(const double Z, 
                                    const double log_p, 
                                    const double log_1m_p,
                                    const double dlog_p_dv, 
                                    const double dlog_1m_p_dv) {
  
      const double log_inv_phi = 0.5 * Z * Z - NEG_HALF_LOG_2PI;           // log(1/phi(Z))
      if (Z < 0.0) return  std::exp(log_p    + log_inv_phi) * dlog_p_dv;       //  (Phi/phi)     * dlog_p/dv
      else         return -std::exp(log_1m_p + log_inv_phi) * dlog_1m_p_dv;   // -((1-Phi)/phi) * dlog_1m_p/dv

}

inline  double  nico_Phi_approx_wo_checks(const double x )  {
  
  const double    a =       (0.07056);
  const double    b =       (1.5976);
  const double    x_sq = x*x;
  const double    a_x_sq_plus_b = fma(a, x_sq, b);
  const double    stuff_to_inv_logit =  x*a_x_sq_plus_b;
  
  return    nico_inv_logit_wo_checks(stuff_to_inv_logit);
  
}

inline   double  nico_Phi_approx(const double x )  {
  
  const double    a =       (0.07056);
  const double    b =       (1.5976);
  const double    x_sq = x*x;
  const double    a_x_sq_plus_b = fma(a, x_sq, b);
  const double    stuff_to_inv_logit =  x*a_x_sq_plus_b;
  
  return    nico_inv_logit(stuff_to_inv_logit);
  
}

inline  double  inv_Phi_approx_std( const double x )  {

  const double m_logit_p =   std::log( 1.0/x  - 1.0 )  ; // log first
  const double x_i = -0.34176618822627863*m_logit_p;
  const double asinh_stuff_div_3 =  0.33333333333333331483 * std::log( x_i  +  std::sqrt(  fma(x_i, x_i, 1.0) ) )  ;          // now do arc_sinh part
  const double exp_x_i =  std::exp(asinh_stuff_div_3);
  return  2.7472242570765295 * ( fma(exp_x_i, exp_x_i , -1.0) / exp_x_i ) ;  //   now do sinh parth part

}

inline  double  nico_inv_Phi_approx_wo_checks( const double x )  {
  
  const double m_logit_p = nico_log_1_double_precision_wo_checks( 1.0/x - 1.0); // log first
  const double x_i = -0.34176618822627863*m_logit_p;
  const double asinh_stuff_div_3 =  0.33333333333333331483 * nico_log_1_double_precision_wo_checks( x_i  +  std::sqrt(  fma(x_i, x_i, 1.0) ) ) ;          // now do arc_sinh part
  const double exp_x_i = nico_exp_1_wo_checks(asinh_stuff_div_3);
  return  2.7472242570765295 * ( ((exp_x_i*exp_x_i)  - 1.0) / exp_x_i ) ;  //   now do sinh parth part
  
}

inline  double  nico_inv_Phi_approx( const double x )  {
  
      if ( (x > 0) &&  (x < 1)) {
            const double m_logit_p = nico_log_1_double_precision(1.0/x - 1.0); // log first
            const double x_i = -0.34176618822627863*m_logit_p;
            const double asinh_stuff_div_3 =  0.33333333333333331483 * nico_log_1_double_precision(x_i + std::sqrt(fma(x_i, x_i, 1.0))) ;          // now do arc_sinh part
            const double exp_x_i = nico_exp_1(asinh_stuff_div_3);
            return  2.7472242570765295 * (((exp_x_i*exp_x_i)  - 1.0) / exp_x_i) ;  //   now do sinh parth part
      }  else {
            if ((x < 0) || (x > 1)) {
              // return std::numeric_limits<double>::quiet_NaN();
                std::stringstream os;
                os << "Invalid input argument for inv_Phi_approx (" << x
                   << "); must be in [0, 1]";
                throw std::invalid_argument( os.str() );
            }   else if  (x == 0) {
              return -INFINITY;
            }   else { // if x  == 1
              return INFINITY;
            }
      }
      
}

inline    double inv_Phi_approx_from_logit_prob_std(const double logit_p) {
  
  double x_i = 0.34176618822627863*logit_p;
  double asinh_stuff_div_3 =  0.33333333333333331483 *  std::log( x_i  +   std::sqrt(  fma(x_i, x_i, 1.0) ) ) ;          // now do arc_sinh part
  double exp_x_i =  std::exp(asinh_stuff_div_3);
  return  2.7472242570765295 * ( fma(exp_x_i, exp_x_i , -1.0) / exp_x_i ) ;  //   now do sinh parth part
  
}

inline    double nico_inv_Phi_approx_from_logit_prob_wo_checks(const double  logit_p) {
  
  double x_i = 0.34176618822627863*logit_p;
  double asinh_stuff_div_3 =  0.33333333333333331483 *  nico_log_1_double_precision(x_i  +   std::sqrt(x_i*x_i + 1.0)) ;          // now do arc_sinh part
  double exp_x_i =  nico_exp_1_wo_checks(asinh_stuff_div_3);
  return  2.7472242570765295 * ( (exp_x_i*exp_x_i  - 1.0) / exp_x_i ) ;  //   now do sinh parth part
  
}

inline   double nico_inv_Phi_approx_from_logit_prob(const double  logit_p) {
  
  double x_i = 0.34176618822627863*logit_p;
  double asinh_stuff_div_3 =  0.33333333333333331483 *  nico_log_1_double_precision_wo_checks(x_i  +   std::sqrt(x_i*x_i + 1.0)) ;          // now do arc_sinh part
  double exp_x_i =  nico_exp_1_wo_checks(asinh_stuff_div_3);
  return  2.7472242570765295 * ( (exp_x_i*exp_x_i  - 1.0) / exp_x_i ) ;  //   now do sinh parth part
  
}

inline  double  nico_log_Phi_approx_wo_checks( const double x )  {
  
 const double    a =       (0.07056);
 const double    b =       (1.5976);
 const double    x_sq = x*x;
 const double    a_x_sq_plus_b = fma(a, x_sq, b);
 const double    stuff_to_inv_logit =  x*a_x_sq_plus_b;
 return nico_log_inv_logit_wo_checks(stuff_to_inv_logit);
 
}

inline   double  nico_log_Phi_approx(const double x )  {
 
      const double    a =       (0.07056);
      const double    b =       (1.5976);
      const double    x_sq = x*x;
      const double    a_x_sq_plus_b = fma(a, x_sq, b);
      const double    stuff_to_inv_logit =  x*a_x_sq_plus_b;
      return nico_log_inv_logit(stuff_to_inv_logit);
      
}

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

inline double nico_Phi_wo_checks(double x) {

      const double z = std::fabs(x);
      //// NaN x: the comparison is false, so z_rational = the cap (as _mm512_min_pd), and the NaN still
      //// propagates through std::exp.
      const double z_rational = (z < FAST_PHI_MILLS_Z_CAP) ? z : FAST_PHI_MILLS_Z_CAP;

      double numerator = FAST_PHI_MILLS_NUMERATOR_COEFFICIENTS[6];
      numerator = std::fma(numerator, z_rational, FAST_PHI_MILLS_NUMERATOR_COEFFICIENTS[5]);
      numerator = std::fma(numerator, z_rational, FAST_PHI_MILLS_NUMERATOR_COEFFICIENTS[4]);
      numerator = std::fma(numerator, z_rational, FAST_PHI_MILLS_NUMERATOR_COEFFICIENTS[3]);
      numerator = std::fma(numerator, z_rational, FAST_PHI_MILLS_NUMERATOR_COEFFICIENTS[2]);
      numerator = std::fma(numerator, z_rational, FAST_PHI_MILLS_NUMERATOR_COEFFICIENTS[1]);
      numerator = std::fma(numerator, z_rational, FAST_PHI_MILLS_NUMERATOR_COEFFICIENTS[0]);

      double denominator = FAST_PHI_MILLS_DENOMINATOR_COEFFICIENTS[7];
      denominator = std::fma(denominator, z_rational, FAST_PHI_MILLS_DENOMINATOR_COEFFICIENTS[6]);
      denominator = std::fma(denominator, z_rational, FAST_PHI_MILLS_DENOMINATOR_COEFFICIENTS[5]);
      denominator = std::fma(denominator, z_rational, FAST_PHI_MILLS_DENOMINATOR_COEFFICIENTS[4]);
      denominator = std::fma(denominator, z_rational, FAST_PHI_MILLS_DENOMINATOR_COEFFICIENTS[3]);
      denominator = std::fma(denominator, z_rational, FAST_PHI_MILLS_DENOMINATOR_COEFFICIENTS[2]);
      denominator = std::fma(denominator, z_rational, FAST_PHI_MILLS_DENOMINATOR_COEFFICIENTS[1]);
      denominator = std::fma(denominator, z_rational, FAST_PHI_MILLS_DENOMINATOR_COEFFICIENTS[0]);

      const double lower_tail_probability = std::exp(-0.5 * z * z) * (numerator / denominator);   //// Phi(-z) = exp(-z^2/2) * N(z) / Q(z), same order as the AVX kernels
      double val;

      if (x >= 0.0) {

         val =  1.0 - lower_tail_probability;

      } else {

         val = lower_tail_probability;

      }


      /// ensure output is between 0 and 1
      if  ((val > 0) && (val < 1))  {
        return val;
      } else if (val >= 1) {   //// >=: val == 1.0 (x > ~8.29) must not reach the final branch (which returns 0)
        return 1;
      } else {
        return 0;
      }

}

inline double nico_Phi(const double x) {

    const double sqrt_2_recip = 0.707106781186547461715;

    if (std::isnan(x)) return x;   //// preserve NaN (not 1.0), as nico_Phi_AVX2 / _AVX512 do

    if  ((x >= -37.5) && (x < 8.25)) {   //// >=: x == -37.5 must not fall through to the final branch (which
                                         ////     returns 1.0)
      
          return  nico_Phi_wo_checks(x) ; // 0.5 * (1.0 + nico_erf_wo_checks(x * sqrt_2_recip));
      
    } else if (x < -37.5) {
      
          return 0.0;
      
    } else {  // if x > 8.25
      
          return 1.0;
       
    }

}

inline double  nico_inv_Phi_wo_checks(double p) {
  
  double r; 
  double val;
  const double q = p - 0.5;
  
  if (stan::math::abs(q) <= 0.425) { // CASE 1
    
    r = 0.180625 - q * q;
    
    double numerator =  fma(r, 2509.0809287301226727, 33430.575583588128105);
    numerator =  fma(numerator, r, 67265.770927008700853);
    numerator =  fma(numerator, r, 45921.953931549871457); 
    numerator =  fma(numerator, r, 13731.693765509461125); /// (fma_3 * r + 13731.693765509461125);
    numerator =  fma(numerator, r, 1971.5909503065514427); ///  (fma_4 * r +  1971.5909503065514427);
    numerator =  fma(numerator, r, 133.14166789178437745); /// (fma_5 * r + 133.14166789178437745);
    numerator =  fma(numerator, r, 3.387132872796366608); ///  (fma_6 * r + 3.387132872796366608)e 
    
    
    double denominator =  fma(r, 5226.495278852854561, 28729.085735721942674);
    denominator =  fma(denominator, r, 39307.89580009271061);
    denominator =  fma(denominator, r, 21213.794301586595867);
    denominator =  fma(denominator, r, 5394.1960214247511077);
    denominator =  fma(denominator, r, 687.1870074920579083);
    denominator =  fma(denominator, r, 42.313330701600911252);
    denominator =  fma(denominator, r, 1.0);
    
    val =  numerator / denominator;
    val = q*val;
    
    return val;
    
  } else { /* closer than 0.075 from {0,1} boundary */  //// CASE 2 
    
    //// first, compute r 
    if (q > 0.0) r = 1.0 - p;
    else r = p;
    r = stan::math::sqrt(-nico_log_1_double_precision(r));
    
    
    /// then compute val 
    if (r <= 5.0) { /* <==> min(p,1-p) >= exp(-25) ~= 1.3888e-11 */  //// CASE 2(a)
      
      r += -1.60;
      
      double numerator = (0.00077454501427834140764);
      numerator = fma(r, numerator, (0.0227238449892691845833));
      numerator = fma(r, numerator, (0.24178072517745061177));
      numerator = fma(r, numerator, (1.27045825245236838258));
      numerator = fma(r, numerator, (3.64784832476320460504));
      numerator = fma(r, numerator, (5.7694972214606914055));
      numerator = fma(r, numerator, (4.6303378461565452959));
      numerator = fma(r, numerator, (1.42343711074968357734));
      
      double  denominator = (0.00000000105075007164441684324);
      denominator = fma(r,  denominator, (0.0005475938084995344946));
      denominator = fma(r,  denominator, (0.0151986665636164571966));
      denominator = fma(r,  denominator, (0.14810397642748007459));
      denominator = fma(r,  denominator, (0.68976733498510000455));
      denominator = fma(r,  denominator, (1.6763848301838038494));
      denominator = fma(r,  denominator, (2.05319162663775882187));
      denominator = fma(r, denominator,  (1.0));
      
      val = numerator / denominator;
      
      
    } else { /* very close to  0 or 1 */      //// CASE 2(b)
       
      r += -5.0;
      
      double numerator =  (0.000000201033439929228813265);
      numerator = fma(r, numerator,  (0.0000271155556874348757815));
      numerator = fma(r, numerator,  (0.0012426609473880784386));
      numerator = fma(r, numerator,  (0.026532189526576123093));
      numerator = fma(r, numerator,  (0.29656057182850489123));
      numerator = fma(r, numerator,  (1.7848265399172913358));
      numerator = fma(r, numerator,  (5.4637849111641143699));
      numerator = fma(r, numerator,  (6.6579046435011037772));
      
      double denominator =  (0.00000000000000204426310338993978564);
      denominator = fma(r, denominator,  (0.00000014215117583164458887));
      denominator = fma(r, denominator,  (0.000018463183175100546818));
      denominator = fma(r, denominator,  (0.0007868691311456132591));
      denominator = fma(r, denominator,  (0.0148753612908506148525));
      denominator = fma(r, denominator,  (0.13692988092273580531));
      denominator = fma(r, denominator,  (0.59983220655588793769));
      denominator = fma(r, denominator,  (1.0));
      
      val = numerator / denominator;
      
    }
    
    if (q < 0.0) val = -val;
    
    return val;
    
  } 
  
  
}

}






















