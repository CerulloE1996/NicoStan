
#pragma once 
 
#ifndef FN_WRAPPERS_SIMD_AVX_GENERAL_HPP
#define FN_WRAPPERS_SIMD_AVX_GENERAL_HPP

 
 
#include <Eigen/Dense>
#include <Eigen/Core>
 
#include <string>
 
#include "fn_SIMD_level_resolver.hpp"
 
 
//// -------------------------------------------------------------------------------------------------------------
//// ---- BOTH SIMD levels are compiled on an AVX-512 build:
////        - fn_process_Ref_double_AVX512  (8 lanes, fast_*_AVX512 kernels) is compiled when
////          BAYESMVP_COMPILED_AVX512_KERNELS,
////        - fn_process_Ref_double_AVX2    (4 lanes, fast_*_AVX2 kernels)   is compiled when
////          BAYESMVP_COMPILED_AVX2_KERNELS (i.e. on AVX2-only AND on AVX-512 builds),
////        - the Eigen helpers take the lane width (8 or 4) as a template argument (a compile-time constant, no
////          per-element branching),
////      so vect_type = "AVX2" on an AVX-512 build runs the 4-lane kernels. fn_EIGEN_Ref_double
////      (fn_wrappers_overall.hpp) picks one via fn_BayesMVP_SIMD_lane_width_for_vect_type()
////      (fn_SIMD_level_resolver.hpp). fn_process_Ref_double_AVX (the name NicoStan's SIMD_config.hpp
////      forward-declares) runs the highest compiled level.
////
#if BAYESMVP_COMPILED_AVX512_KERNELS || BAYESMVP_COMPILED_AVX2_KERNELS
 
 
#include <immintrin.h>
 
 
//// Kernel function-pointer type for each lane width: BayesMVP_SIMD_kernel_pointer<8 or 4>::type is the pointer
//// type FuncAVX of SIMD_config.hpp for each level.
namespace nico::math {

template <int SIMD_lane_width> struct BayesMVP_SIMD_kernel_pointer;

}  // namespace nico::math

using nico::math::BayesMVP_SIMD_kernel_pointer;
#if BAYESMVP_COMPILED_AVX512_KERNELS
namespace nico::math {

template <> struct BayesMVP_SIMD_kernel_pointer<8> { typedef __m512d (*type)(const __m512d); };

}  // namespace nico::math

using nico::math::BayesMVP_SIMD_kernel_pointer;   //// e.g. nico_exp_1_AVX512
#endif
#if BAYESMVP_COMPILED_AVX2_KERNELS
namespace nico::math {

template <> struct BayesMVP_SIMD_kernel_pointer<4> { typedef __m256d (*type)(const __m256d); };

}  // namespace nico::math

using nico::math::BayesMVP_SIMD_kernel_pointer;   //// e.g. nico_exp_1_AVX2
#endif


//// SIMD_lane_width = 8 (AVX-512 kernels) or 4 (AVX2 kernels); a compile-time constant.
namespace nico::math {

template <int SIMD_lane_width, typename T>
ALWAYS_INLINE  void fn_AVX_row_or_col_vector(    Eigen::Ref<T>  x_Ref,
                                                 const typename BayesMVP_SIMD_kernel_pointer<SIMD_lane_width>::type &fn_AVX,
                                                 const FuncDouble &fn_double) {
  
        static_assert((SIMD_lane_width == 8) || (SIMD_lane_width == 4),
                      "SIMD_lane_width must be 8 (AVX-512) or 4 (AVX2)");
  
        const int N = x_Ref.size();
  
        const int vect_size = SIMD_lane_width;
          
        const double vect_siz_dbl = static_cast<double>(vect_size);
        const double N_dbl = static_cast<double>(N);
        const int N_divisible_by_vect_size = std::floor(N_dbl / vect_siz_dbl) * vect_size;
        
       if (N >= vect_size) {
             
                     //// last vect_size elements, copied BEFORE the SIMD pass overwrites them (the scalar
                     //// remainder pass below recomputes them from their original values). Inside
                     //// "if (N >= vect_size)": for N < vect_size, x_Ref(N - vect_size) would read before the
                     //// start of the vector (and the copy is not used then).
                     Eigen::Matrix<double, -1, 1> x_tail = Eigen::Matrix<double, -1, 1>::Zero(vect_size); // last vect_size elements
                     {
                         int counter = 0;
                         for (int i = N - vect_size; i < N; ++i) {
                           x_tail(counter) = x_Ref(i);
                           counter += 1;
                         } 
                     }
                     
                     for (int i = 0; i + vect_size <= N_divisible_by_vect_size; i += vect_size) {
                       
                         if constexpr (SIMD_lane_width == 8) {
                           #if BAYESMVP_COMPILED_AVX512_KERNELS
                             ALIGN64  __m512d const AVX_array = _mm512_loadu_pd(&x_Ref(i));
                             ALIGN64  __m512d const AVX_array_out = fn_AVX(AVX_array);
                             _mm512_storeu_pd(&x_Ref(i), AVX_array_out);
                           #endif
                         } else {
                           #if BAYESMVP_COMPILED_AVX2_KERNELS
                             ALIGN32  __m256d const AVX_array = _mm256_loadu_pd(&x_Ref(i));
                             ALIGN32  __m256d const AVX_array_out = fn_AVX(AVX_array);
                             _mm256_storeu_pd(&x_Ref(i), AVX_array_out);
                           #endif
                         }
                           
                     }
                     
                     if (N_divisible_by_vect_size != N) {    // Handle remainder
                       int counter = 0;
                       for (int i = N - vect_size; i < N; ++i) {
                         x_Ref(i) =  fn_double(x_tail(counter));
                         counter += 1;
                       }
                     }
         
       }  else {   // If N < vect_size, handle everything with scalar operations
         
             for (int i = 0; i < N; ++i) {
               x_Ref(i) = fn_double(x_Ref(i));
             }
         
       }
   
}

}  // namespace nico::math

using nico::math::fn_AVX_row_or_col_vector;
 
 
 
 
 
 
 

namespace nico::math {

template <int SIMD_lane_width, typename T>
ALWAYS_INLINE  void fn_AVX_matrix(    Eigen::Ref<T> x_Ref,
                                      const typename BayesMVP_SIMD_kernel_pointer<SIMD_lane_width>::type &fn_AVX, 
                                      const FuncDouble &fn_double) {
     
     const int n_rows = x_Ref.rows();
     const int n_cols = x_Ref.cols();
     
     if (n_rows > n_cols) { // if data in "long" format
       for (int j = 0; j < n_cols; ++j) {   
         Eigen::Matrix<double, -1, 1> x_col = x_Ref.col(j);
         using ColType = decltype(x_col);
         Eigen::Ref<Eigen::Matrix<double, -1, 1>> x_col_Ref(x_col); 
         fn_AVX_row_or_col_vector<SIMD_lane_width, ColType>(x_col_Ref, fn_AVX, fn_double);
         x_Ref.col(j) = x_col_Ref;
       }
     } else { 
       for (int j = 0; j < n_rows; ++j) {
         Eigen::Matrix<double, 1, -1> x_row = x_Ref.row(j);
         using RowType = decltype(x_row);
         Eigen::Ref<Eigen::Matrix<double, 1, -1>> x_row_Ref(x_row); 
         fn_AVX_row_or_col_vector<SIMD_lane_width, RowType>(x_row_Ref, fn_AVX, fn_double);
         x_Ref.row(j) = x_row_Ref;
       }
     }
   
}

}  // namespace nico::math

using nico::math::fn_AVX_matrix; 
 
 
 
 
 
 
 

namespace nico::math {

template <int SIMD_lane_width, typename T>
ALWAYS_INLINE  void fn_AVX_dbl_Eigen(     Eigen::Ref<T> x_Ref, 
                                          const typename BayesMVP_SIMD_kernel_pointer<SIMD_lane_width>::type &fn_AVX, 
                                          const FuncDouble &fn_double) {
     
     constexpr int n_rows = T::RowsAtCompileTime;
     constexpr int n_cols = T::ColsAtCompileTime;
     
     if constexpr (n_rows == 1 && n_cols == -1) {   // Row vector case
        
           fn_AVX_row_or_col_vector<SIMD_lane_width>(x_Ref, fn_AVX, fn_double);
       
     } else if constexpr (n_rows == -1 && n_cols == 1) {  // Column vector case
       
           fn_AVX_row_or_col_vector<SIMD_lane_width>(x_Ref, fn_AVX, fn_double);
       
     } else {   // General matrix case
       
           fn_AVX_matrix<SIMD_lane_width>(x_Ref, fn_AVX, fn_double);
       
     }
   
}

}  // namespace nico::math

using nico::math::fn_AVX_dbl_Eigen;
  
 
 

 
 
 
  

namespace nico::math {

template <int SIMD_lane_width, typename T>
ALWAYS_INLINE  void    fn_process_double_AVX_sub_function(    Eigen::Ref<T> x_Ref,  
                                                              const typename BayesMVP_SIMD_kernel_pointer<SIMD_lane_width>::type &fn_fast_AVX_function,
                                                              const FuncDouble &fn_fast_double_function,
                                                              const typename BayesMVP_SIMD_kernel_pointer<SIMD_lane_width>::type &fn_fast_AVX_function_wo_checks,
                                                              const FuncDouble &fn_fast_double_function_wo_checks, 
                                                              const bool skip_checks) {
      
      if (skip_checks == false) {
        
           fn_AVX_dbl_Eigen<SIMD_lane_width>(x_Ref, fn_fast_AVX_function, fn_fast_double_function);
        
      } else {
        
           fn_AVX_dbl_Eigen<SIMD_lane_width>(x_Ref,
                                             fn_fast_AVX_function_wo_checks,
                                             fn_fast_double_function_wo_checks);
        
      }
  
}

}  // namespace nico::math

using nico::math::fn_process_double_AVX_sub_function;

 


 

#if BAYESMVP_COMPILED_AVX512_KERNELS
//// ---- 8-lane (AVX-512) processor: the AVX-512 body of fn_process_Ref_double_AVX under its own name, so that it
////      can coexist with the 4-lane one below (with the <8> lane-width argument of the helpers).
 
namespace nico::math {

template <typename T>
ALWAYS_INLINE  void       fn_process_Ref_double_AVX512(    Eigen::Ref<T> x_Ref,
                                                           const std::string &fn,
                                                           const bool &skip_checks) {
  
    if        (fn == "test_simple") {    
          std::cout << "Calling test_simple function" << std::endl;
          try { 
            // fn_process_double_AVX_sub_function(x_Ref, 
            //                                     test_simple_AVX512,  test_simple_double,  
            //                                     test_simple_AVX512,  test_simple_double, skip_checks) ;
            fn_AVX_dbl_Eigen<8>(x_Ref, test_simple_AVX512, test_simple_double);
            // fn_process_double_AVX_sub_function(x_Ref, test_simple_AVX512,  test_simple_double,   test_simple_AVX512, test_simple_double, skip_checks) ;
          } catch (const std::exception& e) { 
              std::cout << "Exception caught: " << e.what() << std::endl;
              throw;
          } catch (...) {
              std::cout << "Unknown exception caught" << std::endl;
              throw;
          }
    } else if (fn == "exp") {    
          fn_process_double_AVX_sub_function<8>( x_Ref, 
                                              nico_exp_1_AVX512, mvp_std_exp, 
                                              nico_exp_1_wo_checks_AVX512, mvp_std_exp, skip_checks) ;
    } else if (fn == "log") {   
          fn_process_double_AVX_sub_function<8>( x_Ref, 
                                              nico_log_1_AVX512, mvp_std_log, 
                                              nico_log_1_wo_checks_AVX512, mvp_std_log, skip_checks) ;
    } else if (fn == "log1p") {    
          fn_process_double_AVX_sub_function<8>( x_Ref, 
                                              nico_log1p_1_AVX512, mvp_std_log1p, 
                                              nico_log1p_1_wo_checks_AVX512, mvp_std_log1p, skip_checks) ;
    } else if (fn == "log1m") {     
          fn_process_double_AVX_sub_function<8>( x_Ref, 
                                              nico_log1m_1_AVX512, mvp_std_log1m, 
                                              nico_log1m_1_wo_checks_AVX512, mvp_std_log1m, skip_checks) ;
    } else if (fn == "logit") {        
          fn_process_double_AVX_sub_function<8>( x_Ref, 
                                              nico_logit_AVX512, mvp_std_logit, 
                                              nico_logit_wo_checks_AVX512, mvp_std_logit, skip_checks) ;
    } else if (fn == "tanh") {  
          fn_process_double_AVX_sub_function<8>( x_Ref, 
                                              nico_tanh_AVX512, mvp_std_tanh, 
                                              nico_tanh_wo_checks_AVX512, mvp_std_tanh, skip_checks) ;
    } else if (fn == "Phi_approx") {    
          fn_process_double_AVX_sub_function<8>( x_Ref, 
                                              nico_Phi_approx_AVX512, mvp_std_Phi_approx, 
                                              nico_Phi_approx_wo_checks_AVX512, mvp_std_Phi_approx, skip_checks) ;
    } else if (fn == "log_Phi_approx") {      
          fn_process_double_AVX_sub_function<8>( x_Ref, 
                                              nico_log_Phi_approx_AVX512, nico_log_Phi_approx, 
                                              nico_log_Phi_approx_wo_checks_AVX512, nico_log_Phi_approx_wo_checks, skip_checks) ;
    } else if (fn == "inv_Phi_approx") {      
          fn_process_double_AVX_sub_function<8>( x_Ref, 
                                              nico_inv_Phi_approx_AVX512, nico_inv_Phi_approx, 
                                              nico_inv_Phi_approx_wo_checks_AVX512, nico_inv_Phi_approx_wo_checks, skip_checks) ;
    } else if (fn == "inv_Phi_approx_from_logit_prob") { 
          fn_process_double_AVX_sub_function<8>( x_Ref, 
                                              nico_inv_Phi_approx_from_logit_prob_AVX512, nico_inv_Phi_approx_from_logit_prob, 
                                              nico_inv_Phi_approx_from_logit_prob_wo_checks_AVX512, nico_inv_Phi_approx_from_logit_prob_wo_checks, skip_checks) ;
    } else if (fn == "Phi") {             
          fn_process_double_AVX_sub_function<8>( x_Ref, 
                                              nico_Phi_AVX512, mvp_std_Phi, 
                                              nico_Phi_wo_checks_AVX512, mvp_std_Phi, skip_checks) ;
    } else if (fn == "inv_Phi") {            
          fn_process_double_AVX_sub_function<8>( x_Ref, 
                                              nico_inv_Phi_wo_checks_AVX512, mvp_std_inv_Phi, 
                                              nico_inv_Phi_wo_checks_AVX512, mvp_std_inv_Phi, skip_checks) ;
    } else if (fn == "inv_logit") {           
          fn_process_double_AVX_sub_function<8>( x_Ref, 
                                              nico_inv_logit_AVX512, mvp_std_inv_logit, 
                                              nico_inv_logit_wo_checks_AVX512, mvp_std_inv_logit, skip_checks) ;
    } else if (fn == "log_inv_logit") {     
          fn_process_double_AVX_sub_function<8>( x_Ref, 
                                              nico_log_inv_logit_AVX512, nico_log_inv_logit, 
                                              nico_log_inv_logit_wo_checks_AVX512, nico_log_inv_logit_wo_checks, skip_checks) ;
    }

}

}  // namespace nico::math

using nico::math::fn_process_Ref_double_AVX512;
 

#endif
 
 
#if BAYESMVP_COMPILED_AVX2_KERNELS
//// ---- 4-lane (AVX2) processor: the AVX2 body of fn_process_Ref_double_AVX under its own name, compiled on
////      AVX-512 builds too, so vect_type = "AVX2" runs the genuine 256-bit fast_*_AVX2 kernels there (with the
////      <4> lane-width argument of the helpers).
 
namespace nico::math {

template <typename T>
ALWAYS_INLINE  void       fn_process_Ref_double_AVX2(       Eigen::Ref<T> x_Ref,
                                                            const std::string &fn,
                                                            const bool &skip_checks) {
   
   if        (fn == "test_simple") {    
     std::cout << "Calling test_simple function" << std::endl;
     try { 
       // fn_process_double_AVX_sub_function(x_Ref, 
       //                                     test_simple_AVX2,  test_simple_double,  
       //                                     test_simple_AVX2,  test_simple_double, skip_checks) ;
       fn_AVX_dbl_Eigen<4>(x_Ref, test_simple_AVX2, test_simple_double);
       // fn_process_double_AVX_sub_function(x_Ref, test_simple_AVX2,  test_simple_double,   test_simple_AVX2, test_simple_double, skip_checks) ;
     } catch (const std::exception& e) { 
       std::cout << "Exception caught: " << e.what() << std::endl;
       throw;
     } catch (...) {
       std::cout << "Unknown exception caught" << std::endl;
       throw;
     }
   } else if (fn == "exp") {    
     fn_process_double_AVX_sub_function<4>( x_Ref, 
                                         nico_exp_1_AVX2, mvp_std_exp, 
                                         nico_exp_1_wo_checks_AVX2, mvp_std_exp, skip_checks) ;
   } else if (fn == "log") {   
     fn_process_double_AVX_sub_function<4>( x_Ref, 
                                         nico_log_1_AVX2, mvp_std_log, 
                                         nico_log_1_wo_checks_AVX2, mvp_std_log, skip_checks) ;
   } else if (fn == "log1p") {    
     fn_process_double_AVX_sub_function<4>( x_Ref, 
                                         nico_log1p_1_AVX2, mvp_std_log1p, 
                                         nico_log1p_1_wo_checks_AVX2, mvp_std_log1p, skip_checks) ;
   } else if (fn == "log1m") {     
     fn_process_double_AVX_sub_function<4>( x_Ref, 
                                         nico_log1m_1_AVX2, mvp_std_log1m, 
                                         nico_log1m_1_wo_checks_AVX2, mvp_std_log1m, skip_checks) ;
   } else if (fn == "logit") {        
     fn_process_double_AVX_sub_function<4>( x_Ref, 
                                         nico_logit_AVX2, mvp_std_logit, 
                                         nico_logit_wo_checks_AVX2, mvp_std_logit, skip_checks) ;
   } else if (fn == "tanh") {  
     fn_process_double_AVX_sub_function<4>( x_Ref, 
                                         nico_tanh_AVX2, mvp_std_tanh, 
                                         nico_tanh_wo_checks_AVX2, mvp_std_tanh, skip_checks) ;
   } else if (fn == "Phi_approx") {    
     fn_process_double_AVX_sub_function<4>( x_Ref, 
                                         nico_Phi_approx_AVX2, mvp_std_Phi_approx, 
                                         nico_Phi_approx_wo_checks_AVX2, mvp_std_Phi_approx, skip_checks) ;
   } else if (fn == "log_Phi_approx") {      
     fn_process_double_AVX_sub_function<4>( x_Ref, 
                                         nico_log_Phi_approx_AVX2, nico_log_Phi_approx, 
                                         nico_log_Phi_approx_wo_checks_AVX2, nico_log_Phi_approx_wo_checks, skip_checks) ;
   } else if (fn == "inv_Phi_approx") {      
     fn_process_double_AVX_sub_function<4>( x_Ref, 
                                         nico_inv_Phi_approx_AVX2, nico_inv_Phi_approx, 
                                         nico_inv_Phi_approx_wo_checks_AVX2, nico_inv_Phi_approx_wo_checks, skip_checks) ;
   } else if (fn == "inv_Phi_approx_from_logit_prob") { 
     fn_process_double_AVX_sub_function<4>( x_Ref, 
                                         nico_inv_Phi_approx_from_logit_prob_AVX2, nico_inv_Phi_approx_from_logit_prob, 
                                         nico_inv_Phi_approx_from_logit_prob_wo_checks_AVX2, nico_inv_Phi_approx_from_logit_prob_wo_checks, skip_checks) ;
   } else if (fn == "Phi") {             
     fn_process_double_AVX_sub_function<4>( x_Ref, 
                                         nico_Phi_AVX2, mvp_std_Phi, 
                                         nico_Phi_wo_checks_AVX2, mvp_std_Phi, skip_checks) ;
   } else if (fn == "inv_Phi") {            
     fn_process_double_AVX_sub_function<4>( x_Ref, 
                                         nico_inv_Phi_wo_checks_AVX2, mvp_std_inv_Phi, 
                                         nico_inv_Phi_wo_checks_AVX2, mvp_std_inv_Phi, skip_checks) ;
   } else if (fn == "inv_logit") {           
     fn_process_double_AVX_sub_function<4>( x_Ref, 
                                         nico_inv_logit_AVX2, mvp_std_inv_logit, 
                                         nico_inv_logit_wo_checks_AVX2, mvp_std_inv_logit, skip_checks) ;
   } else if (fn == "log_inv_logit") {     
     fn_process_double_AVX_sub_function<4>( x_Ref, 
                                         nico_log_inv_logit_AVX2, nico_log_inv_logit, 
                                         nico_log_inv_logit_wo_checks_AVX2, nico_log_inv_logit_wo_checks, skip_checks) ;
   }
   
 }

}  // namespace nico::math

using nico::math::fn_process_Ref_double_AVX2;
 
#endif
 
 
//// ---- fn_process_Ref_double_AVX: the name NicoStan's SIMD_config.hpp forward-declares (whenever USE_AVX2 or
////      USE_AVX_512 is defined). It runs the HIGHEST compiled level. BayesMVP's own dispatch does not call it;
////      fn_EIGEN_Ref_double calls fn_process_Ref_double_AVX512 / fn_process_Ref_double_AVX2 directly.
namespace nico::math {

template <typename T>
ALWAYS_INLINE  void       fn_process_Ref_double_AVX(         Eigen::Ref<T> x_Ref,
                                                             const std::string &fn,
                                                             const bool &skip_checks) {
   
    #if BAYESMVP_COMPILED_AVX512_KERNELS
          fn_process_Ref_double_AVX512(x_Ref, fn, skip_checks);
    #else
          fn_process_Ref_double_AVX2(x_Ref, fn, skip_checks);
    #endif

}

}  // namespace nico::math

using nico::math::fn_process_Ref_double_AVX;
 
 

 
  
  


#endif



#endif

  
  



















