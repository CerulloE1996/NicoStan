
#pragma once 


#ifndef FN_WRAPPERS_LOG_SUM_EXP_SIMD_HPP
#define FN_WRAPPERS_LOG_SUM_EXP_SIMD_HPP


#include <stan/math/prim/fun/sqrt.hpp>
#include <stan/math/prim/fun/log.hpp>
#include <stan/math/prim/prob/std_normal_log_qf.hpp>
#include <stan/math/prim/fun/Phi.hpp>
#include <stan/math/prim/fun/inv_Phi.hpp>
#include <stan/math/prim/fun/Phi_approx.hpp>
#include <stan/math/prim/fun/tanh.hpp>
#include <stan/math/prim/fun/log_inv_logit.hpp>


#include <Eigen/Dense>
#include <Eigen/Core>


#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)   //// (x86-64 only: ARM CPUs have no AVX)
#include <immintrin.h>
#endif

#include <string>
#include <stdexcept>

#include "fn_SIMD_level_resolver.hpp"







//// The AVX2 kernels below are compiled on AVX-512 builds too, so that vect_type = "AVX2" runs the 4-lane kernels
//// there instead of falling through fn_log_sum_exp_2d_double below to Stan's scalar log_sum_exp.
#if BAYESMVP_COMPILED_AVX2_KERNELS
  
namespace nico::math {

ALWAYS_INLINE Eigen::Matrix<double, -1, 1>   nico_log_sum_exp_2d_AVX2_double(  Eigen::Ref<Eigen::Matrix<double, -1, -1>>   x) {
    
      const int N = x.rows();
      const std::string vect_type_exp = "AVX2";
      const std::string vect_type_log = "AVX2";
      
      Eigen::Matrix<double, -1, 1> log_sum_abs_result(N);
      Eigen::Matrix<double, -1, 1> container_max_logs(N);
      
      log_sum_exp_general(x, 
                          vect_type_exp,
                          vect_type_log,
                          log_sum_abs_result,
                          container_max_logs);
      
      return log_sum_abs_result;
    
}

}  // namespace nico::math

using nico::math::nico_log_sum_exp_2d_AVX2_double; 



#endif




#if BAYESMVP_COMPILED_AVX512_KERNELS
  
namespace nico::math {

ALWAYS_INLINE Eigen::Matrix<double, -1, 1>   nico_log_sum_exp_2d_AVX512_double(  Eigen::Ref<Eigen::Matrix<double, -1, -1>> x) {
      
      const int N = x.rows();
      const std::string vect_type_exp = "AVX512";
      const std::string vect_type_log = "AVX512";
      
      Eigen::Matrix<double, -1, 1> log_sum_abs_result(N);
      Eigen::Matrix<double, -1, 1> container_max_logs(N);
      
      log_sum_exp_general(x, 
                          vect_type_exp,
                          vect_type_log,
                          log_sum_abs_result,
                          container_max_logs);
      
      return log_sum_abs_result;
    
}

}  // namespace nico::math

using nico::math::nico_log_sum_exp_2d_AVX512_double;
  
  
#endif




namespace nico::math {

ALWAYS_INLINE  Eigen::Matrix<double, -1, 1> fn_log_sum_exp_2d_double(      Eigen::Ref<Eigen::Matrix<double, -1, -1>>  x,    // Eigen::Matrix<double, -1, 2> &x,
                                                                           const std::string &vect_type = "Stan",
                                                                           const bool &skip_checks = false) {
    
    {
      if (vect_type == "Eigen") {
        return  log_sum_exp_2d_Eigen_double(x);
      } else if (vect_type == "Stan") {
        return  log_sum_exp_2d_Stan_double(x);
      } else if (vect_type == "AVX2") {
        //// a SIMD level this build did not compile THROWS (as in fn_EIGEN_Ref_double;
        //// fn_SIMD_level_resolver.hpp), instead of silently falling through to the Stan return at the bottom of
        //// this function. Plain string comparisons (no extra per-call cost).
#if BAYESMVP_COMPILED_AVX2_KERNELS
        if (skip_checks == false)   return  nico_log_sum_exp_2d_AVX2_double(x);
        else                        return  nico_log_sum_exp_2d_AVX2_double(x);
#else
        throw std::invalid_argument(fn_BayesMVP_SIMD_level_not_compiled_message(vect_type));
#endif
      } else if (vect_type == "AVX512") {
#if BAYESMVP_COMPILED_AVX512_KERNELS
        if (skip_checks == false)   return  nico_log_sum_exp_2d_AVX512_double(x);
        else                        return  nico_log_sum_exp_2d_AVX512_double(x);
#else
        throw std::invalid_argument(fn_BayesMVP_SIMD_level_not_compiled_message(vect_type));
#endif
      } else {
        return  log_sum_exp_2d_Stan_double(x);
      }
      
    }
    
    return  log_sum_exp_2d_Stan_double(x);
    
  }

}  // namespace nico::math

using nico::math::fn_log_sum_exp_2d_double;
  
  
  
  
  






#endif

  
  
  
  
  

















