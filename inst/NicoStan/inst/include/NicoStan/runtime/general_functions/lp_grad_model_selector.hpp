
#pragma once

 
 

#include <tbb/task_arena.h>
#include <NicoStan/runtime/general_functions/full_clock_of_the_compute_threads.hpp>
#include <NicoStan/runtime/general_functions/the_TBB_arena_of_each_chain.hpp>
#include <NicoStan/runtime/general_functions/chunk_local_leapfrog_step_of_the_nuisance_block.hpp>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <cstdlib>
#include <memory>
#include <sstream>
#include <stdexcept>  
#include <complex>
///#include <dlfcn.h> // For dynamic loading 
#include <map>
#include <vector>  
#include <string> 
#include <stdexcept>
#include <stdio.h>
#include <iostream>
 
#include <stan/model/model_base.hpp>  
 
#include <stan/io/array_var_context.hpp>
#include <stan/io/var_context.hpp>
#include <stan/io/dump.hpp> 

 
#include <stan/io/json/json_data.hpp>
#include <stan/io/json/json_data_handler.hpp>
#include <stan/io/json/json_error.hpp>
#include <stan/io/json/rapidjson_parser.hpp>   
 
 


#include <Eigen/Dense>
 
 

  
  
  
  
  


inline void  fn_lp_grad_InPlace(         Eigen::Ref<Eigen::Matrix<double, -1, 1>> lp_and_grad_outs,
                                         const std::string  &Model_type,
                                         const bool force_autodiff,
                                         const bool force_PartialLog,
                                         const bool multi_attempts,
                                         const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_main_vec_ref,
                                         const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_us_vec_ref,
                                         const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref,
                                         const std::string &grad_option,
                                         const Model_fn_args_struct  &Model_args_as_cpp_struct,
                                         const Stan_model_struct &Stan_model_as_cpp_struct,
                                         std::vector<LC_MVP_workspace_struct> &LC_MVP_ws_structs,
                                         const int n_threads_WCP
) {

#ifndef NICOSTAN_WITH_BUILTINS
  if (Model_type != "Stan") throw std::invalid_argument("NicoStan requires Model_type = Stan.");
#endif
  const int N = Model_args_as_cpp_struct.N;
  const int n_nuisance =  Model_args_as_cpp_struct.n_nuisance;
  int n_params_main = Model_args_as_cpp_struct.n_params_main;
  int n_params = n_params_main + n_nuisance;
   
#ifdef NICOSTAN_WITH_BUILTINS
   if  ((Model_type == "latent_trait") || (Model_type == "MVP") ||  (Model_type == "MVOP") || (Model_type == "LC_MVP") || (Model_type == "LC_MVOP")) { //// use a built-in, fast manual-gradient model

       // The latent-trait partial-log implementation is not a supported
       // execution path.  Fail before dispatch so that callers never receive
       // an uninitialised or partially populated output vector.
       if (Model_type == "latent_trait" && force_PartialLog) {
         throw std::invalid_argument("force_PartialLog = TRUE is unsupported for Model_type = latent_trait; use force_PartialLog = FALSE.");
       }
     
       if (multi_attempts == true) {
                     
                        if ((Model_type == "LC_MVP") || (Model_type == "MVP")) {
                          
                                          //// force_autodiff / force_PartialLog are passed on and honoured there
                                          //// (they used to stop here, so the multi-attempt chain ignored them):
                                          fn_lp_grad_MVP_multi_attempts_InPlace_process(lp_and_grad_outs,
                                                                                        theta_main_vec_ref,
                                                                                        theta_us_vec_ref,
                                                                                        y_ref,
                                                                                        grad_option,
                                                                                        Model_args_as_cpp_struct,
                                                                                        LC_MVP_ws_structs,
                                                                                        n_threads_WCP,
                                                                                        Model_args_as_cpp_struct.autodiff_fallback,
                                                                                        Model_type == "MVP",
                                                                                        force_autodiff,
                                                                                        force_PartialLog);
                          
                        } else if (Model_type == "latent_trait") {
                          
                          //// #if COMPILE_LATENT_TRAIT
                                          fn_lp_grad_LT_LC_multi_attempts_InPlace_process(lp_and_grad_outs,
                                                                                          theta_main_vec_ref,
                                                                                          theta_us_vec_ref,
                                                                                          y_ref,
                                                                                          grad_option,
                                                                                          Model_args_as_cpp_struct,
                                                                                          !Model_args_as_cpp_struct.autodiff_fallback_option_is_set ||
                                                                                              Model_args_as_cpp_struct.autodiff_fallback);
                          ////  #endif
                                         
                        } else if ((Model_type == "LC_MVOP") || (Model_type == "MVOP")) {
                          
                                          //// force_autodiff / force_PartialLog are passed on and honoured there
                                          //// (they used to stop here, so the multi-attempt chain ignored them):
                                          fn_lp_grad_MVOP_multi_attempts_InPlace_process(lp_and_grad_outs,
                                                                                         theta_main_vec_ref,
                                                                                         theta_us_vec_ref,
                                                                                         y_ref,
                                                                                         grad_option,
                                                                                         Model_args_as_cpp_struct,
                                                                                         LC_MVP_ws_structs,
                                                                                         n_threads_WCP,
                                                                                         Model_args_as_cpp_struct.
                                                                                               autodiff_fallback,
                                                                                         force_autodiff,
                                                                                         force_PartialLog);
                          
                        }
           
       } else {
           
                        if (force_autodiff == false) {  // Handle cases where autodiff is not forced
                           
                                 if ((Model_type == "LC_MVP") || (Model_type == "MVP")) {
                                   
                                       if (force_PartialLog == true) {
                                         
                                              fn_lp_grad_MVP_LC_Pinkney_PartialLog_MD_and_AD_InPlace_process(lp_and_grad_outs,
                                                                                                             theta_main_vec_ref, 
                                                                                                             theta_us_vec_ref,
                                                                                                             y_ref, 
                                                                                                             grad_option, 
                                                                                                             Model_args_as_cpp_struct,
                                                                                                             LC_MVP_ws_structs,
                                                                                                             n_threads_WCP);
                                         
                                       } else {
                                         
                                              fn_lp_grad_MVP_LC_Pinkney_NoLog_MD_and_AD_Inplace_process(  lp_and_grad_outs, 
                                                                                                          theta_main_vec_ref,
                                                                                                          theta_us_vec_ref,
                                                                                                          y_ref, 
                                                                                                          grad_option, 
                                                                                                          Model_args_as_cpp_struct,
                                                                                                          LC_MVP_ws_structs,
                                                                                                          n_threads_WCP);
                                         
                                       }
                                       
                                 } else if (Model_type == "latent_trait") {
                                   
                                       //// #if COMPILE_LATENT_TRAIT
                                       if (force_PartialLog == true) {
                                                
                                                fn_lp_grad_LT_LC_PartialLog_MD_and_AD_InPlace_process(  lp_and_grad_outs, 
                                                                                                        theta_main_vec_ref,
                                                                                                        theta_us_vec_ref, 
                                                                                                        y_ref, 
                                                                                                        grad_option, 
                                                                                                        Model_args_as_cpp_struct);
                                               //// bookmark
                                               //  std::cout << "ERROR: For the latent_trait model, the MANUAL-LOG-SCALE lp_grad function isn't yet working fully, so set force_PartialLog = FALSE" << std::endl;
                                              
                                       } else {
                                         
                                               fn_lp_grad_LT_LC_NoLog_MD_and_AD_InPlace_process(  lp_and_grad_outs,
                                                                                                  theta_main_vec_ref, 
                                                                                                  theta_us_vec_ref, 
                                                                                                  y_ref, 
                                                                                                  grad_option,
                                                                                                  Model_args_as_cpp_struct);
                                         
                                       }
                                       //// #endif
                                       
                                 } else if ((Model_type == "LC_MVOP") || (Model_type == "MVOP")) {
                                   
                                       // fn_lp_grad_MVOP_LC_Pinkney_NoLog_MD_and_AD_Inplace_process(
                                       //       lp_and_grad_outs, theta_main_vec_ref, theta_us_vec_ref, y_ref,
                                       //       grad_option, Model_args_as_cpp_struct, LC_MVP_ws_structs,
                                       //       n_threads_WCP);
                                       ////
                                       //// force_PartialLog = TRUE runs the PartialLog path, as for LC_MVP / MVP
                                       //// above (the call above ran NoLog whatever force_PartialLog was):
                                       ////
                                       if (force_PartialLog == true) {
                                         
                                              fn_lp_grad_MVOP_LC_Pinkney_PartialLog_MD_and_AD_Inplace_process(
                                                    lp_and_grad_outs, theta_main_vec_ref, theta_us_vec_ref,
                                                    y_ref, grad_option, Model_args_as_cpp_struct,
                                                    LC_MVP_ws_structs, n_threads_WCP);
                                         
                                       } else {
                                         
                                              fn_lp_grad_MVOP_LC_Pinkney_NoLog_MD_and_AD_Inplace_process(
                                                    lp_and_grad_outs, theta_main_vec_ref, theta_us_vec_ref,
                                                    y_ref, grad_option, Model_args_as_cpp_struct,
                                                    LC_MVP_ws_structs, n_threads_WCP);
                                         
                                       }
                                   
                                 }
                           
                        } else if (force_autodiff == true) { /// autodiff
                           
                                 if (Model_type == "LC_MVP") {
                                   
                                           fn_lp_and_grad_MVP_Pinkney_AD_log_scale_InPlace_process(lp_and_grad_outs,
                                                                                                   theta_main_vec_ref, 
                                                                                                   theta_us_vec_ref, 
                                                                                                   y_ref,  
                                                                                                   grad_option, 
                                                                                                   Model_args_as_cpp_struct,
                                                                                                   LC_MVP_ws_structs,
                                                                                                   n_threads_WCP);
                                   
                                 } else if (Model_type == "MVP") {
                                   
                                           fn_lp_and_grad_std_MVP_Pinkney_AD_log_scale_InPlace_process(lp_and_grad_outs,
                                                                                                       theta_main_vec_ref, 
                                                                                                       theta_us_vec_ref, 
                                                                                                       y_ref,  
                                                                                                       grad_option,
                                                                                                       Model_args_as_cpp_struct,
                                                                                                       LC_MVP_ws_structs,
                                                                                                       n_threads_WCP); 
                                   
                                 } else if (Model_type == "latent_trait") {
                                   
                                   //// #if COMPILE_LATENT_TRAIT
                                           fn_lp_and_grad_LC_LT_AD_log_scale_InPlace_process(lp_and_grad_outs, 
                                                                                             theta_main_vec_ref, 
                                                                                             theta_us_vec_ref, 
                                                                                             y_ref,
                                                                                             grad_option, 
                                                                                             Model_args_as_cpp_struct); 
                                   //// #endif
                                   
                                 } else if ((Model_type == "LC_MVOP") || (Model_type == "MVOP")) {
                                    
                                           fn_lp_and_grad_MVOP_Pinkney_AD_log_scale_InPlace_process(lp_and_grad_outs,
                                                                                                    theta_main_vec_ref,
                                                                                                    theta_us_vec_ref,
                                                                                                    y_ref,
                                                                                                    grad_option,
                                                                                                    Model_args_as_cpp_struct,
                                                                                                    LC_MVP_ws_structs,
                                                                                                    n_threads_WCP);
                                           
                                 }
                               
                        }
             
       }
         
  } else
#endif
  if (Model_type == "Stan" ) { 
     
     //// #if HAS_BRIDGESTAN_H
                 
                 //// ---- Build the COMPLETE unconstrained vector: nuisance block FIRST,
                 //// main block SECOND. This must hold for EVERY n_nuisance (including 0
                 //// and small blocks <= 10): the Stan log-density is defined on all
                 //// declared coordinates. The previous "n_nuisance > 10" shortcut
                 //// silently dropped small nuisance blocks (and misplaced the gradient),
                 //// which is wrong whenever the model has nuisance parameters.
                 //// (the complete vector and the full gradient are per-thread buffers, made again only when the
                 ////  number of parameters changes, so that no heap allocation is made per gradient)
                 static thread_local Eigen::Matrix<double, -1, 1> params;
                 static thread_local Eigen::Matrix<double, -1, 1> full_lp_grad_buffer;
                 //// (a model in NicoStan's nuisance-block form reads the nuisance values straight from
                 ////  theta_us_vec_ref, so no complete vector is made for it)
                 const bool nuisance_block_owned_by_NicoStan =
                       (Stan_model_as_cpp_struct.nicostan_bs_log_density_gradient_with_the_nuisance_block !=
                        nullptr);
                 if (!nuisance_block_owned_by_NicoStan) {
                       if (params.size() != n_params) params.resize(n_params);
                       params.head(n_nuisance) = theta_us_vec_ref;
                       params.tail(n_params_main) = theta_main_vec_ref;
                 }
                 
                 //// ---- full (log_prob, gradient) on the complete vector: written straight into lp_and_grad_outs
                 //// when the whole gradient is asked for (no copy), otherwise into the buffer and copied as before
                 // const bool write_straight_into_outputs = (grad_option != "main_only") &&
                 //                                          (grad_option != "us_only") &&
                 //                                          (lp_and_grad_outs.size() == 1 + n_params);
                 //// (each chain's lp_and_grad_outs holds 1 + N + n_params values, with N = 100 for Stan models
                 ////  (R_fn_initialise_model.R), the log density first, then the nuisance and main gradients, the
                 ////  N values last and unused for Stan models: the test "== 1 + n_params" was never true, so every
                 ////  gradient went into the buffer and was copied, and the chunk-local leapfrog step was never
                 ////  made; the first 1 + n_params values are the same places the copy writes to)
                 const bool write_straight_into_outputs = (grad_option != "main_only") &&
                                                          (grad_option != "us_only") &&
                                                          (lp_and_grad_outs.size() >= 1 + n_params);
                 if ((!write_straight_into_outputs) && (full_lp_grad_buffer.size() != 1 + n_params)) {
                       full_lp_grad_buffer.resize(1 + n_params);
                 }
                 Eigen::Map<Eigen::Matrix<double, -1, 1>> full_lp_grad(write_straight_into_outputs ?
                                                                       lp_and_grad_outs.data() :
                                                                       full_lp_grad_buffer.data(),
                                                                       1 + n_params);
                 //// (the chunk-local leapfrog step, chunk_local_leapfrog_step_of_the_nuisance_block.hpp: only in
                 ////  the gradient call of a leapfrog step, which sets the step for its duration, of a library
                 ////  built with it, and only when the nuisance gradient is written straight into
                 ////  lp_and_grad_outs, where the step reads it; otherwise the gradient call is made as before
                 ////  and the sampler makes the step)
                 const auto gradient_with_the_chunk_local_leapfrog_step = Stan_model_as_cpp_struct.
                       nicostan_bs_log_density_gradient_with_the_nuisance_block_and_the_chunk_local_leapfrog_step;
                 nicostan_chunk_local_leapfrog_step_t* chunk_local_leapfrog_step =
                       nicostan_chunk_local_leapfrog_step_of_this_gradient_call;
                 const bool make_the_chunk_local_leapfrog_step =
                       nuisance_block_owned_by_NicoStan && (chunk_local_leapfrog_step != nullptr) &&
                       write_straight_into_outputs && (gradient_with_the_chunk_local_leapfrog_step != nullptr);
                 if (chunk_local_leapfrog_step != nullptr) {
                       chunk_local_leapfrog_step->step_made_inside_the_chunks = 0;
                 }
                 //// (the model's gradient call: a model in NicoStan's nuisance-block form gets the main
                 ////  parameters and NicoStan's nuisance values, and writes the nuisance gradient and the main
                 ////  gradient straight into their parts of full_lp_grad; every other model gets the complete
                 ////  vector)
                 const auto compute_log_density_and_gradient = [&]() {

                       if (make_the_chunk_local_leapfrog_step) {
                             double log_density = 0.0;
                             char* error_message = nullptr;
                             const int result = gradient_with_the_chunk_local_leapfrog_step(
                                         Stan_model_as_cpp_struct.bs_model_ptr, true, true,
                                         theta_main_vec_ref.data(), theta_us_vec_ref.data(),
                                         full_lp_grad.data() + 1, n_nuisance, &log_density,
                                         full_lp_grad.data() + 1 + n_nuisance, chunk_local_leapfrog_step,
                                         &error_message);
                             if (result != 0) {
                                   throw std::runtime_error("Gradient computation failed: " +
                                                            std::string(error_message ? error_message :
                                                                                        "Unknown error"));
                             }
                             full_lp_grad(0) = log_density;
                       } else if (nuisance_block_owned_by_NicoStan) {
                             double log_density = 0.0;
                             char* error_message = nullptr;
                             const int result = Stan_model_as_cpp_struct.
                                   nicostan_bs_log_density_gradient_with_the_nuisance_block(
                                         Stan_model_as_cpp_struct.bs_model_ptr, true, true,
                                         theta_main_vec_ref.data(), theta_us_vec_ref.data(),
                                         full_lp_grad.data() + 1, n_nuisance, &log_density,
                                         full_lp_grad.data() + 1 + n_nuisance, &error_message);
                             if (result != 0) {
                                   throw std::runtime_error("Gradient computation failed: " +
                                                            std::string(error_message ? error_message :
                                                                                        "Unknown error"));
                             }
                             full_lp_grad(0) = log_density;
                       } else {
                             fn_Stan_compute_log_prob_grad(Stan_model_as_cpp_struct,
                                                           params, n_params_main, n_nuisance,
                                                           full_lp_grad);
                       }

                 };
                 //// ---- the Stan model's gradient runs inside THIS CHAIN'S OWN TBB arena of exactly n_threads_WCP
 //// threads (1 reserved for the chain's thread): before every chain's reduce_sum put its
                 //// tasks on the one global TBB pool of the process, so the chains fought over the same workers
                 //// (4 x 4 burn-in: 11 ms per gradient against 8.2 ms for one chain alone on 4 CPUs, PK at
                 //// N = 25,000), and with 1 thread per chain a chain's chunks could run on other chains' CPUs.
                 //// (thread_local: one arena per chain thread, made again only when n_threads_WCP changes)
 //// Measured, -20:4x (PK, N = 25,000, local-HPC): with the strict per-chain arena
                 //// Stan math's burn-in was SLOWER at every WCP (4 x 4: 0.222 against 0.159 s per iteration; 4 x 45:
                 //// 0.117 against 0.095), because with the shared pool a chain in its serial part lends its idle
                 //// CPUs to the other chains' tasks. So the arena is OFF by default and switched on with the
                 //// environment variable NICOSTAN_PER_CHAIN_TBB_ARENA=1 (read once per process).
                 //// (the chain's OpenMP team for a model library compiled with NICOSTAN_REDUCE_SUM_OPENMP: the
                 ////  thread count is this thread's own OpenMP setting, read by the library's reduce_sum)
                 #ifdef _OPENMP
                 omp_set_num_threads(std::max(1, n_threads_WCP));
                 #endif
                 // static const bool use_per_chain_TBB_arena = [] {
                 //       const char *value = std::getenv("NICOSTAN_PER_CHAIN_TBB_ARENA");
                 //       return (value != nullptr) && (std::string(value) == "1");
                 // }();
                 // if (use_per_chain_TBB_arena) {
                 //       static thread_local std::unique_ptr<tbb::task_arena> arena_of_this_chain;
                 //       static thread_local int n_threads_of_arena_of_this_chain = -1;
                 //       //// (the WCP threads entering this chain's arena ask for the full clock:
                 //       ////  full_clock_of_the_compute_threads.hpp)
                 //       static thread_local std::unique_ptr<full_clock_of_the_threads_entering_an_arena>
                 //             full_clock_of_the_arena_of_this_chain;
                 //       if (n_threads_of_arena_of_this_chain != n_threads_WCP) {
                 //             full_clock_of_the_arena_of_this_chain.reset();
                 //             arena_of_this_chain.reset(new tbb::task_arena(std::max(1, n_threads_WCP), 1));
                 //             arena_of_this_chain->initialize();
                 //             full_clock_of_the_arena_of_this_chain.reset(
                 //                   new full_clock_of_the_threads_entering_an_arena(*arena_of_this_chain));
                 //             n_threads_of_arena_of_this_chain = n_threads_WCP;
                 //       }
                 //       arena_of_this_chain->execute([&]() {
                 //             compute_log_density_and_gradient();
                 //       });
                 // } else {
                 //// (the chain's own arena, shared with the leapfrog's pass over the nuisance values:
                 ////  the_TBB_arena_of_each_chain.hpp)
                 if (fn_each_chain_runs_its_WCP_work_in_its_own_TBB_arena()) {
                       fn_TBB_arena_of_this_chain(n_threads_WCP).execute([&]() {

                             compute_log_density_and_gradient();

                       });
                 } else {
                       compute_log_density_and_gradient();
                 }
                 
                 if (!write_straight_into_outputs) lp_and_grad_outs(0) = full_lp_grad(0);
                 
                 if (write_straight_into_outputs) {
                   
                     //// (lp_and_grad_outs already holds the log density and the whole gradient)
                   
                 } else if (grad_option == "main_only") { 
                     
                     //// only the MAIN gradient block is copied back (nuisance slots stay 0)
                     lp_and_grad_outs.segment(1 + n_nuisance,
                                              n_params_main) = full_lp_grad.segment(1 + n_nuisance,
                                              n_params_main);
                   
                 } else if (grad_option == "us_only") {  // only nuisance 
                    
                     //// only the NUISANCE gradient block is copied back (main slots stay 0)
                     lp_and_grad_outs.segment(1, n_nuisance) = full_lp_grad.segment(1, n_nuisance);
                   
                 } else {  /// otherwise compute entire gradient
                   
                     lp_and_grad_outs.segment(1, n_params) = full_lp_grad.segment(1, n_params);
                   
                 }
                 
            
            //// #endif
              
  }
   
}










inline  Eigen::Matrix<double, -1, 1 >         fn_lp_grad(    const std::string  &Model_type,
                                                             const bool force_autodiff,
                                                             const bool force_PartialLog,
                                                             const bool multi_attempts,
                                                             const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_main_vec_ref,
                                                             const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_us_vec_ref,
                                                             const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref,
                                                             const std::string &grad_option,
                                                             const Model_fn_args_struct  &Model_args_as_cpp_struct,
                                                             const Stan_model_struct &Stan_model_as_cpp_struct,
                                                             std::vector<LC_MVP_workspace_struct> &LC_MVP_ws_structs,
                                                             const int n_threads_WCP
) {
 
     
      const int N = Model_args_as_cpp_struct.N;
      const int n_nuisance =  Model_args_as_cpp_struct.n_nuisance;
      const int n_params_main = Model_args_as_cpp_struct.n_params_main;
      const int n_params = n_params_main + n_nuisance;
     
       Eigen::Matrix<double, -1, 1> lp_and_grad_outs =     Eigen::Matrix<double, -1, 1>::Zero(1 + N + n_params);
       
       
       fn_lp_grad_InPlace(        lp_and_grad_outs, 
                                  Model_type, 
                                  force_autodiff, force_PartialLog, multi_attempts,
                                  theta_main_vec_ref,
                                  theta_us_vec_ref, 
                                  y_ref,
                                  grad_option,
                                  Model_args_as_cpp_struct,
                                  Stan_model_as_cpp_struct,
                                  LC_MVP_ws_structs,
                                  n_threads_WCP);
       
       
       return  lp_and_grad_outs;
 
}


 
 
 
 
 
 
 
 
 
 
 
//   void  fn_lp_only_InPlace(      double &lp,
//                                        const std::string  &Model_type,
//                                        const bool force_autodiff,
//                                        const bool force_PartialLog,
//                                        const bool multi_attempts,
//                                        const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_main_vec_ref,
//                                        const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_us_vec_ref,
//                                        const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref,
//                                        const Model_fn_args_struct  &Model_args_as_cpp_struct,
//                                        const Stan_model_struct &Stan_model_as_cpp_struct) {
// 
//   const int N = Model_args_as_cpp_struct.N;
//   const int n_nuisance =  Model_args_as_cpp_struct.n_nuisance;
//   int n_params_main = Model_args_as_cpp_struct.n_params_main;
//   int n_params = n_params_main + n_nuisance;
// 
//   const std::string  grad_option = "none";
//   
//   if  ((Model_type == "latent_trait") || (Model_type == "MVP") || (Model_type == "LC_MVP")) { //// use a built-in, fast manual-gradient model
//         
//           // Handle cases where autodiff is not forced
//           if (force_autodiff == false) {
//             
//                   if ((Model_type == "LC_MVP") || (Model_type == "MVP")) {
//                     
//                         if (force_PartialLog == true) {
//                           lp = (   fn_lp_grad_MVP_LC_Pinkney_PartialLog_MD_and_AD(theta_main_vec_ref, theta_us_vec_ref, y_ref, grad_option, Model_args_as_cpp_struct)  ).head(1).eval()(0);
//                         } else {
//                           lp =  (  fn_lp_grad_MVP_LC_Pinkney_NoLog_MD_and_AD(theta_main_vec_ref, theta_us_vec_ref, y_ref, grad_option, Model_args_as_cpp_struct) ).head(1).eval()(0);
//                         }
//                         
//                   } else if (Model_type == "latent_trait") {
//                     //// #if COMPILE_LATENT_TRAIT
//                     
//                         if (force_PartialLog == true) {  
//                           lp = (   fn_lp_grad_LT_LC_PartialLog_MD_and_AD(theta_main_vec_ref, theta_us_vec_ref, y_ref, grad_option, Model_args_as_cpp_struct)  ).head(1).eval()(0);
//                           //// bookmark
//                           // std::cout << "ERROR: For the latent_trait model, the MANUAL-LOG-SCALE lp_grad function isn't yet working fully, so set force_PartialLog = FALSE" << std::endl; 
//                         } else { 
//                           lp =  (  fn_lp_grad_LT_LC_NoLog_MD_and_AD(theta_main_vec_ref, theta_us_vec_ref, y_ref, grad_option, Model_args_as_cpp_struct) ).head(1).eval()(0);
//                         }
//                         //// #endif
//                         
//                   }
//           } else {  /// use autodiff 
//             
//             if ((Model_type == "LC_MVP") || (Model_type == "MVP")) {
//                     
//                     lp =  (  fn_lp_and_grad_MVP_Pinkney_AD_log_scale(theta_main_vec_ref, theta_us_vec_ref, y_ref,  grad_option, Model_args_as_cpp_struct) ).head(1).eval()(0);
//                     
//                   } else if (Model_type == "latent_trait") {
//                     //// #if COMPILE_LATENT_TRAIT
//                     lp =  (  fn_lp_and_grad_LC_LT_AD_log_scale(theta_main_vec_ref, theta_us_vec_ref, y_ref,  grad_option, Model_args_as_cpp_struct) ).head(1).eval()(0);
//                     //// #endif
//                     
//                   } 
//             
//           }
// 
//   }  else if (Model_type == "Stan" )  { 
//     
//     
//               //// #if HAS_BRIDGESTAN_H
//               
//               Eigen::Matrix<double, -1, 1> params(n_params);
//               if (n_nuisance > 10) {
//                   params.head(n_nuisance) = theta_us_vec_ref;
//                   params.tail(n_params_main) = theta_main_vec_ref;
//               } else { 
//                   params = theta_main_vec_ref;
//                   n_params = n_params_main;
//                   params.resize(n_params_main);
//               }
//               
//               Eigen::Matrix<double, -1, 1> lp_and_grad_outs(1 + N + n_params);
//               
//               if (grad_option == "main_only") { 
//                 
//                 lp  =    fn_Stan_compute_log_prob_grad(Stan_model_as_cpp_struct,  params, n_params_main, n_nuisance, lp_and_grad_outs).head(1).eval()(0);
//                 
//               } else if (grad_option == "us_only") {  // only nuisance 
//                 
//                 lp  =  fn_Stan_compute_log_prob_grad(Stan_model_as_cpp_struct,   params, n_params_main, n_nuisance, lp_and_grad_outs).head(1).eval()(0);
//                 
//               } else {  /// otherwise compute entire gradient
//                 
//                 lp  =  fn_Stan_compute_log_prob_grad(Stan_model_as_cpp_struct,  params, n_params_main, n_nuisance, lp_and_grad_outs).head(1).eval()(0);
//                 
//               }
//               
//               //// #endif
//     
//   }
//   
// }


 
 
 
 
 
 
 
 
 
 










