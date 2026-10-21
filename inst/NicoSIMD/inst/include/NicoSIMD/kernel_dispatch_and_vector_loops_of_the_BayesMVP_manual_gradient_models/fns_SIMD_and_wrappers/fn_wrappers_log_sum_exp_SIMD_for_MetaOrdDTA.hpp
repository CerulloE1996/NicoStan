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
 
#include <immintrin.h>

 

 

 

 
 
 
 
  
 

 
  
  
  
  
#if defined(__AVX2__) && ( !(defined(__AVX512VL__) && defined(__AVX512F__)  && defined(__AVX512DQ__)) ) // use AVX2
  
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





#if defined(__AVX512VL__) && defined(__AVX512F__)  && defined(__AVX512DQ__)
  
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
#if defined(__AVX2__) && ( !(defined(__AVX512VL__) && defined(__AVX512F__)  && defined(__AVX512DQ__)) ) // use AVX2
        if (skip_checks == false)   return  nico_log_sum_exp_2d_AVX2_double(x);
        else                        return  nico_log_sum_exp_2d_AVX2_double(x);
#endif
      } else if (vect_type == "AVX512") {
#if defined(__AVX512VL__) && defined(__AVX512F__)  && defined(__AVX512DQ__)
        if (skip_checks == false)   return  nico_log_sum_exp_2d_AVX512_double(x);
        else                        return  nico_log_sum_exp_2d_AVX512_double(x);
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

  
  
  
  
  
