
#pragma once
 

#include <random>
#include <limits>
#include <memory>
#include <Eigen/Dense>
#include <unsupported/Eigen/SpecialFunctions>
#include "EHMC_dual_flow_kick_flow_fns.hpp"
#include <NicoStan/runtime/general_functions/the_TBB_arena_of_each_chain.hpp>
#include <NicoStan/runtime/general_functions/chunk_local_leapfrog_step_of_the_nuisance_block.hpp>
#include <algorithm>
#include <vector>
 
 
 
 

 
 
 
ALWAYS_INLINE  void leapfrog_integrator_dense_M_standard_HMC_dual_InPlace(           Eigen::Matrix<double, -1, 1> &velocity_main_vec_proposed_ref,
                                                                                     Eigen::Matrix<double, -1, 1> &velocity_us_vec_proposed_ref,
                                                                                     Eigen::Matrix<double, -1, 1> &theta_main_vec_proposed_ref,
                                                                                     Eigen::Matrix<double, -1, 1> &theta_us_vec_proposed_ref,
                                                                                     Eigen::Matrix<double, -1, 1> &lp_and_grad_outs,
                                                                                     const Eigen::Matrix<double, -1, 1> &theta_main_vec_initial_ref,
                                                                                     const Eigen::Matrix<double, -1, 1> &theta_us_vec_initial_ref,
                                                                                     const Eigen::Matrix<double, -1, -1> &M_inv_dense_main,
                                                                                     const Eigen::Matrix<double, -1, 1> &M_inv_us_vec,
                                                                                     const Eigen::Matrix<int, -1, -1> &y_ref,
                                                                                     const int L_ii, // Number of leapfrog steps
                                                                                     const double eps,
                                                                                     const std::string &Model_type,
                                                                                     const bool force_autodiff, const bool force_PartialLog, const bool multi_attempts,
                                                                                     const std::string &grad_option,
                                                                                     const Model_fn_args_struct &Model_args_as_cpp_struct,
                                                                                     const Stan_model_struct &Stan_model_as_cpp_struct,
                                                                                     std::vector<LC_MVP_workspace_struct> &LC_MVP_ws_structs,
                                                                                     const int n_threads_WCP,
                                                                                     std::function<void(Eigen::Ref<Eigen::Matrix<double, -1, 1>>,
                                                                                                        const std::string,
                                                                                                        const bool, const bool, const bool,
                                                                                                        const Eigen::Ref<const Eigen::Matrix<double, -1, 1>>,
                                                                                                        const Eigen::Ref<const Eigen::Matrix<double, -1, 1>>,
                                                                                                        const Eigen::Ref<const Eigen::Matrix<int, -1, -1>>,
                                                                                                        const std::string,
                                                                                                        const Model_fn_args_struct &,
                                                                                                        const Stan_model_struct &,
                                                                                                        std::vector<LC_MVP_workspace_struct> &,
                                                                                                        const int &)> fn_lp_grad_InPlace
                                                                                                 
 ) {
   
               const int N = Model_args_as_cpp_struct.N;
               const int n_nuisance =  Model_args_as_cpp_struct.n_nuisance;
               const int n_params_main = Model_args_as_cpp_struct.n_params_main;
               const int n_params = n_params_main + n_nuisance;
               
               Eigen::Matrix<double, -1, 1> grad_main =  ( lp_and_grad_outs.segment(1 + n_nuisance, n_params_main).array()).matrix();
               
               for (int l = 0; l < L_ii; l++) {
            if (Model_args_as_cpp_struct.burnin_leapfrog_steps != nullptr) {
                ++(*Model_args_as_cpp_struct.burnin_leapfrog_steps);
            }
                 
                         // Update velocity (first half step) - main params
                         velocity_main_vec_proposed_ref.array() +=  ( 0.5 * eps * M_inv_dense_main *  grad_main ).array() ; 
                         // Update velocity (first half step) - nuisance params
                         velocity_us_vec_proposed_ref.array() += (0.5 * eps * lp_and_grad_outs.segment(1, n_nuisance).array()  * M_inv_us_vec.array());
                         
                         //// update params by full step - main params
                         theta_main_vec_proposed_ref.array()  +=  eps *     velocity_main_vec_proposed_ref.array() ;
                         //// update params by full step - nuisance params
                         theta_us_vec_proposed_ref.array()  +=  eps *     velocity_us_vec_proposed_ref.array() ;
                         
                         // Update lp and gradients
                         fn_lp_grad_InPlace(lp_and_grad_outs, 
                                            Model_type, force_autodiff, force_PartialLog, multi_attempts,
                                            theta_main_vec_proposed_ref, 
                                            theta_us_vec_proposed_ref,
                                            y_ref,
                                            grad_option, 
                                            Model_args_as_cpp_struct, //MVP_workspace,
                                            Stan_model_as_cpp_struct, 
                                            LC_MVP_ws_structs,
                                            n_threads_WCP);
                         if (!std::isfinite(lp_and_grad_outs(0))) return; 
                         grad_main =  ( lp_and_grad_outs.segment(1 + n_nuisance, n_params_main).array()).matrix();
                         
                         // Update velocity (second half step)
                         velocity_main_vec_proposed_ref.array() +=  ( 0.5 * eps * M_inv_dense_main *  grad_main ).array() ;   
                         // Update velocity (first half step) - nuisance params
                         velocity_us_vec_proposed_ref.array() += (0.5 * eps * lp_and_grad_outs.segment(1, n_nuisance).array()  * M_inv_us_vec.array());
                 
               } // End of leapfrog steps 
                                       
 }
 
 
 
 
 
 
 
 
 
 ALWAYS_INLINE  void leapfrog_integrator_diag_M_standard_HMC_dual_InPlace(     Eigen::Matrix<double, -1, 1> &velocity_main_vec_proposed_ref,
                                                                               Eigen::Matrix<double, -1, 1> &velocity_us_vec_proposed_ref,
                                                                               Eigen::Matrix<double, -1, 1> &theta_main_vec_proposed_ref,
                                                                               Eigen::Matrix<double, -1, 1> &theta_us_vec_proposed_ref,
                                                                               Eigen::Matrix<double, -1, 1> &lp_and_grad_outs,
                                                                               const Eigen::Matrix<double, -1, 1> &theta_main_vec_initial_ref,
                                                                               const Eigen::Matrix<double, -1, 1> &theta_us_vec_initial_ref,
                                                                               const Eigen::Matrix<double, -1, 1> &M_inv_main_vec,
                                                                               const Eigen::Matrix<double, -1, 1> &M_inv_us_vec,
                                                                               const Eigen::Matrix<int, -1, -1> &y_ref,
                                                                               const int L_ii, // Number of leapfrog steps
                                                                               const double eps,
                                                                               const std::string &Model_type,
                                                                               const bool force_autodiff, const bool force_PartialLog, const bool multi_attempts,
                                                                               const std::string &grad_option,
                                                                               const Model_fn_args_struct &Model_args_as_cpp_struct,
                                                                               const Stan_model_struct &Stan_model_as_cpp_struct,
                                                                               std::vector<LC_MVP_workspace_struct> &LC_MVP_ws_structs,
                                                                               const int n_threads_WCP,
                                                                               std::function<void(Eigen::Ref<Eigen::Matrix<double, -1, 1>>,
                                                                                                  const std::string,
                                                                                                  const bool, const bool, const bool,
                                                                                                  const Eigen::Ref<const Eigen::Matrix<double, -1, 1>>,
                                                                                                  const Eigen::Ref<const Eigen::Matrix<double, -1, 1>>,
                                                                                                  const Eigen::Ref<const Eigen::Matrix<int, -1, -1>>,
                                                                                                  const std::string,
                                                                                                  const Model_fn_args_struct &,
                                                                                                  const Stan_model_struct &,
                                                                                                  std::vector<LC_MVP_workspace_struct> &,
                                                                                                  const int &)> fn_lp_grad_InPlace
                                                                                 
 ) {
   
       const int N = Model_args_as_cpp_struct.N;
       const int n_nuisance =  Model_args_as_cpp_struct.n_nuisance;
       const int n_params_main = Model_args_as_cpp_struct.n_params_main;
       const int n_params = n_params_main + n_nuisance;
       
       Eigen::Matrix<double, -1, 1> grad_main =  ( lp_and_grad_outs.segment(1 + n_nuisance, n_params_main).array()).matrix();
       
       for (int l = 0; l < L_ii; l++) {
            if (Model_args_as_cpp_struct.burnin_leapfrog_steps != nullptr) {
                ++(*Model_args_as_cpp_struct.burnin_leapfrog_steps);
            }
         
             // Update velocity (first half step) - main params
             velocity_main_vec_proposed_ref.array() +=  ( 0.5 * eps * M_inv_main_vec.array() *  grad_main.array() ).array() ;
             // Update velocity (first half step) - nuisance params
             velocity_us_vec_proposed_ref.array() += (0.5 * eps * lp_and_grad_outs.segment(1, n_nuisance).array()  * M_inv_us_vec.array());
             
             //// update params by full step - main params 
             theta_main_vec_proposed_ref.array()  +=  eps *     velocity_main_vec_proposed_ref.array() ;
             //// update params by full step - nuisance params
             theta_us_vec_proposed_ref.array()  +=  eps *     velocity_us_vec_proposed_ref.array() ;
             
             // Update lp and gradients
             fn_lp_grad_InPlace(lp_and_grad_outs, 
                                Model_type, force_autodiff, force_PartialLog, multi_attempts,
                                theta_main_vec_proposed_ref, theta_us_vec_proposed_ref, y_ref, grad_option, 
                                Model_args_as_cpp_struct, //MVP_workspace,
                                Stan_model_as_cpp_struct,
                                LC_MVP_ws_structs,
                                n_threads_WCP);
             if (!std::isfinite(lp_and_grad_outs(0))) return; 
             grad_main =  ( lp_and_grad_outs.segment(1 + n_nuisance, n_params_main).array()).matrix();
             
             // Update velocity (second half step)
             velocity_main_vec_proposed_ref.array() +=  ( 0.5 * eps * M_inv_main_vec.array() *  grad_main.array() ).array() ; 
             // Update velocity (first half step) - nuisance params
             velocity_us_vec_proposed_ref.array() += (0.5 * eps * lp_and_grad_outs.segment(1, n_nuisance).array()  * M_inv_us_vec.array());
         
       } // End of leapfrog steps 
   
 }


 
 
 
 
 
  
 
template<typename T = RNG_TYPE_dqrng>
ALWAYS_INLINE  void                         fn_standard_HMC_dual_single_iter_InPlace_process(    HMCResult &result_input,
                                                                                                 T &rng_main,
                                                                                                 T &rng_nuisance,
                                                                                                 const std::string &Model_type,
                                                                                                 const bool  force_autodiff,
                                                                                                 const bool  force_PartialLog,
                                                                                                 const bool  multi_attempts,
                                                                                                 const Eigen::Matrix<int, -1, -1> &y_ref,
                                                                                                 const Model_fn_args_struct &Model_args_as_cpp_struct,
                                                                                                 EHMC_fn_args_struct  &EHMC_args_as_cpp_struct, /// pass by ref. to modify (???)
                                                                                                 const EHMC_Metric_struct   &EHMC_Metric_struct_as_cpp_struct,
                                                                                                 const Stan_model_struct &Stan_model_as_cpp_struct,
                                                                                                 std::vector<LC_MVP_workspace_struct> &LC_MVP_ws_structs,
                                                                                                 const int n_threads_WCP
) {


    //// important params
    const int N = Model_args_as_cpp_struct.N;
    const int n_nuisance =  Model_args_as_cpp_struct.n_nuisance;
    const int n_params_main = Model_args_as_cpp_struct.n_params_main;
    const int n_params = n_params_main + n_nuisance;
  
    const std::string grad_option = "all";
     
    const std::string metric_shape_main = EHMC_Metric_struct_as_cpp_struct.metric_shape_main;
    
    double U_x_initial =  0.0 ; 
    double U_x_prop =  0.0 ; 
    double log_posterior_prop =  0.0 ; 
    double log_posterior_0  =   0.0 ;   
    double log_ratio = 0.0;
    double energy_old = 0.0;
    double energy_new = 0.0;
    
    //// result_input.store_current_state(); // sets initial theta and velocity to current theta and velocity
    result_input.main_theta_vec_0() =  result_input.main_theta_vec();
    result_input.us_theta_vec_0() =    result_input.us_theta_vec();

  {

      { /// draw velocity for main:
          Eigen::Matrix<double, -1, 1> std_norm_vec_main = Eigen::Matrix<double, -1, 1>::Zero(n_params_main);
          generate_random_std_norm_vec_InPlace(std_norm_vec_main, rng_main);
          if (metric_shape_main == "dense") result_input.main_velocity_0_vec()  = EHMC_Metric_struct_as_cpp_struct.M_inv_dense_main_chol * std_norm_vec_main;
          if (metric_shape_main == "diag")  result_input.main_velocity_0_vec().array() = std_norm_vec_main.array() *  (EHMC_Metric_struct_as_cpp_struct.M_inv_main_vec).array().sqrt() ; 
      }
      { /// draw velocity for nuisance:
          Eigen::Matrix<double, -1, 1> std_norm_vec_us = Eigen::Matrix<double, -1, 1>::Zero(n_nuisance);
          generate_random_std_norm_vec_InPlace(std_norm_vec_us, rng_nuisance);
          result_input.us_velocity_0_vec().array() = std_norm_vec_us.array() *  (EHMC_Metric_struct_as_cpp_struct.M_inv_us_vec).array().sqrt();
      }

    {
 
      { 
        
      try {
        
        result_input.main_velocity_vec_proposed()  =   result_input.main_velocity_0_vec(); // set to initial velocity
        result_input.main_theta_vec_proposed() =       result_input.main_theta_vec_0();   // set to initial theta   
        result_input.us_velocity_vec_proposed()  =   result_input.us_velocity_0_vec(); // set to initial velocity
        result_input.us_theta_vec_proposed() =       result_input.us_theta_vec_0();   // set to initial theta 

          //// ---------------------------------------------------------------------------------------------------------------///    Perform L leapfrogs   ///-----------------------------------------
          if (EHMC_args_as_cpp_struct.tau_main < EHMC_args_as_cpp_struct.eps_main) { 
            EHMC_args_as_cpp_struct.tau_main = EHMC_args_as_cpp_struct.eps_main; 
          }
          //
          //// burn-in with share_tau_ii_across_chains: tau_main_ii was set (identically for every chain) by the burn-in worker
          if (!EHMC_args_as_cpp_struct.randomize_tau) {
            EHMC_args_as_cpp_struct.tau_main_ii = EHMC_args_as_cpp_struct.tau_main;
          } else if (!EHMC_args_as_cpp_struct.use_given_tau_main_ii) {
            EHMC_args_as_cpp_struct.tau_main_ii = fn_generate_burnin_tau_ii(EHMC_args_as_cpp_struct.tau_main,
                                                                            rng_main,
                                                                            true);
          }
          if (EHMC_args_as_cpp_struct.tau_main_ii < EHMC_args_as_cpp_struct.eps_main) { 
            EHMC_args_as_cpp_struct.tau_main_ii = EHMC_args_as_cpp_struct.eps_main; 
          }
          //
          const int L_ii = fn_number_of_leapfrog_steps_from_tau_eps_and_max_L(
              EHMC_args_as_cpp_struct.tau_main_ii,
              EHMC_args_as_cpp_struct.eps_main,
              EHMC_args_as_cpp_struct.max_L);
          EHMC_args_as_cpp_struct.L_main_ii = L_ii;  //// recorded per iteration with debug = TRUE
          
          
          //// initial lp  
          fn_lp_grad_InPlace(     result_input.lp_and_grad_outs(), 
                                  Model_type, 
                                  force_autodiff, force_PartialLog, multi_attempts,
                                  result_input.main_theta_vec(),  result_input.us_theta_vec(),
                                  y_ref, 
                                  grad_option,
                                  Model_args_as_cpp_struct,
                                  Stan_model_as_cpp_struct, 
                                  LC_MVP_ws_structs,
                                  n_threads_WCP);
          
          log_posterior_0 =  result_input.lp_and_grad_outs()(0);
          U_x_initial = - log_posterior_0; //// initial energy
          
          result_input.lp_and_grad_outs_0() = result_input.lp_and_grad_outs(); // ----
          
          /////////  do leapfrogs for ALL parameters
          if (metric_shape_main == "dense") {
                                                                                                 
                 leapfrog_integrator_dense_M_standard_HMC_dual_InPlace(     result_input.main_velocity_vec_proposed(), 
                                                                            result_input.us_velocity_vec_proposed(), 
                                                                            result_input.main_theta_vec_proposed(), 
                                                                            result_input.us_theta_vec_proposed(), 
                                                                            result_input.lp_and_grad_outs(), 
                                                                            result_input.main_theta_vec(),
                                                                            result_input.us_theta_vec(),
                                                                            EHMC_Metric_struct_as_cpp_struct.M_inv_dense_main,
                                                                            EHMC_Metric_struct_as_cpp_struct.M_inv_us_vec,
                                                                            y_ref,
                                                                            L_ii, EHMC_args_as_cpp_struct.eps_main,
                                                                            Model_type, 
                                                                            force_autodiff, force_PartialLog, multi_attempts, 
                                                                            grad_option,
                                                                            Model_args_as_cpp_struct, 
                                                                            Stan_model_as_cpp_struct,
                                                                            LC_MVP_ws_structs,
                                                                            n_threads_WCP,
                                                                            fn_lp_grad_InPlace);
            
          } else if (metric_shape_main == "diag") {
            
                leapfrog_integrator_diag_M_standard_HMC_dual_InPlace(      result_input.main_velocity_vec_proposed(), 
                                                                           result_input.us_velocity_vec_proposed(), 
                                                                           result_input.main_theta_vec_proposed(), 
                                                                           result_input.us_theta_vec_proposed(), 
                                                                           result_input.lp_and_grad_outs(), 
                                                                           result_input.main_theta_vec(),
                                                                           result_input.us_theta_vec(),
                                                                           EHMC_Metric_struct_as_cpp_struct.M_inv_main_vec,
                                                                           EHMC_Metric_struct_as_cpp_struct.M_inv_us_vec,
                                                                           y_ref,
                                                                           L_ii, EHMC_args_as_cpp_struct.eps_main,
                                                                           Model_type, 
                                                                           force_autodiff, force_PartialLog, multi_attempts, 
                                                                           grad_option,
                                                                           Model_args_as_cpp_struct, 
                                                                           Stan_model_as_cpp_struct,
                                                                           LC_MVP_ws_structs,
                                                                           n_threads_WCP,
                                                                           fn_lp_grad_InPlace);
            
          }
 
          
          // /// proposed lp  
          log_posterior_prop =  result_input.lp_and_grad_outs()(0);
          if (EHMC_args_as_cpp_struct.record_kinetic_energy_tau_derivatives) result_input.record_kinetic_energy_rate_main();
          U_x_prop = - log_posterior_prop; // initial energy
        
          //////////////////////////////////////////////////////////////////    M-H acceptance step  (i.e, Accept/Reject step)
          if (metric_shape_main == "dense") {
       
                  const Eigen::Matrix<double, 1, -1>  velocity_0_x_M_dense_main = result_input.main_velocity_0_vec().transpose() * EHMC_Metric_struct_as_cpp_struct.M_dense_main; // row-vec
                  energy_old = U_x_initial ;
                  energy_old  +=  0.5 * (velocity_0_x_M_dense_main * result_input.main_velocity_0_vec() ).eval()(0, 0) ; // for main
                  //// for nuisance:
                  energy_old  +=  0.5 * (  result_input.us_velocity_0_vec().array()  *  result_input.us_velocity_0_vec().array() * ( EHMC_Metric_struct_as_cpp_struct.M_us_vec ).array() ).sum() ;  
                
                  const Eigen::Matrix<double, 1, -1>  velocity_prop_x_M_dense_main =  result_input.main_velocity_vec_proposed().transpose() * EHMC_Metric_struct_as_cpp_struct.M_dense_main; // row-vec
                  energy_new  =  U_x_prop;  
                  energy_new  +=  0.5 * (velocity_prop_x_M_dense_main *  result_input.main_velocity_vec_proposed()).eval()(0, 0) ; // for main
                  //// for nuisance:
                  energy_new  +=  0.5 * (result_input.us_velocity_vec_proposed().array() * result_input.us_velocity_vec_proposed().array() * (EHMC_Metric_struct_as_cpp_struct.M_us_vec ).array() ).sum();  
        
                  log_ratio = - energy_new + energy_old;
              
          } else if (metric_shape_main == "diag") {
            
                  energy_old = U_x_initial ;
                  //// for main:
                  energy_old  +=  0.5 * (result_input.main_velocity_0_vec().array() * result_input.main_velocity_0_vec().array() * (stan::math::inv(EHMC_Metric_struct_as_cpp_struct.M_inv_main_vec).array()).array()).sum(); 
                  //// for nuisance:
                  energy_old  +=  0.5 * (result_input.us_velocity_0_vec().array() * result_input.us_velocity_0_vec().array() * (EHMC_Metric_struct_as_cpp_struct.M_us_vec).array()).sum() ; 
                  
                  energy_new  =  U_x_prop;  
                  //// for main:
                  energy_new  +=  0.5 * (result_input.main_velocity_vec_proposed().array() * result_input.main_velocity_vec_proposed().array() * (stan::math::inv(EHMC_Metric_struct_as_cpp_struct.M_inv_main_vec).array()).array()).sum(); 
                  //// for nuisance:
                  energy_new  +=  0.5 * (result_input.us_velocity_vec_proposed().array() * result_input.us_velocity_vec_proposed().array() * (EHMC_Metric_struct_as_cpp_struct.M_us_vec).array() ).sum(); 
                  
                  log_ratio = - energy_new + energy_old; 
            
          }
 
        if  (check_divergence(log_ratio, result_input) == true)    { /// if main_div, reject proposal 
            
                    //// if  div, reject proposal 
                    result_input.main_div() = 1;  
                    result_input.main_p_jump() = 0.0;
                    result_input.us_div() = 1;
                    result_input.us_p_jump() = 0.0;
                    result_input.reject_proposal_main();  // # reject proposal
                    result_input.reject_proposal_us();  // # reject proposal
        
          
        }  else {  // if no main_div, carry on with MH step 
          
                    result_input.main_div() = 0;
                    result_input.main_p_jump() = std::min(1.0, stan::math::exp(log_ratio));
                    result_input.us_div()   = 0;
                    result_input.us_p_jump() =   std::min(1.0, stan::math::exp(log_ratio));
                    
                    const double rand_unif = generate_random_std_uniform(rng_main);
                    const double log_rand_unif = stan::math::log(rand_unif);

                    if  (log_rand_unif > log_ratio)   {  // # reject proposal
                          result_input.reject_proposal_main();  // # reject proposal
                          result_input.reject_proposal_us();  // # reject proposal
                    } else {   // # accept proposal
                          result_input.accept_proposal_main();  // # accept proposal
                          result_input.accept_proposal_us();  // # accept proposal
                    }
 
                
        }

      } catch (...) {
            //// (a diagnostic: the message of the first failed gradient of this chain, written to stderr; the
            ////  failure is recorded below as before; message_of_the_first_failed_gradient_of_each_chain.hpp)
            try {
                  throw;
            } catch (const std::exception &failed_gradient) {
                  fn_write_the_message_of_the_first_failed_gradient_of_this_chain(
                        "main and nuisance parameters, standard HMC",
                        failed_gradient.what());
            } catch (...) {
                  fn_write_the_message_of_the_first_failed_gradient_of_this_chain(
                        "main and nuisance parameters, standard HMC",
                        "not a std::exception");
            }
        
              // std::cout << "  Could not evaluate lp_grad function when sampling main + nuisance parameters " << ")\n";
        
                result_input.us_div() = 1; // record as div 
                result_input.us_p_jump() = 0.0; // set p_jump to zero
                result_input.reject_proposal_us(); // # reject proposal
                
                result_input.main_div() = 1; // record as div 
                result_input.main_p_jump() = 0.0; // set p_jump to zero
                result_input.reject_proposal_main(); // # reject proposal
  
              
      }
      


    }   
      
    }
    

  }
 


}



 
// // ============================================================
// // DIFFUSION-PATHSPACE DUAL LEAPFROG — DENSE M for main
// // ============================================================
// // Main params:  standard HMC (dense M_main)         — UNCHANGED
// // Nuisance:     diffusion-pathspace (A=I, M=I)      — rotation + corrected kick
// //
// // 3-way split:
// //   H₁ = ½θ_us'θ_us + ½‖v_us‖²        (nuisance rotation — exact)
// //   H₂ = ½v_main'M_main v_main        (main kinetic — position update)
// //   H₃ = Φ(θ) = -log_lik - log_prior  (kick for everything)
// //
// // Kick gradients:
// //   -∂Φ/∂θ_main = grad_main_log_post        (same as standard)
// //   -∂Φ/∂θ_us   = grad_us_log_post + θ_us   (corrected)
// // ============================================================
 

// ============================================================
// (the version with one copy of the factors per thread, before they were shared by the chains:)
// // THE PER-UNIT FACTORS OF THE NUISANCE ROTATION, KEPT BETWEEN TRAJECTORIES
// // ============================================================
// // cos(eps / sqrt(M_i)), sqrt(M_i) sin(eps / sqrt(M_i)) and sin(eps / sqrt(M_i)) / sqrt(M_i) of every nuisance
// // parameter, made with exactly the same expressions as in the leapfrog steps below (so every value is the
// // same, to the last bit), but made again only when the step size or the nuisance metric has changed since this
// // thread's last trajectory (in the sampling phase: once), instead of in 7 new arrays of n_nuisance values for
// // every trajectory. A thread whose last trajectory had another metric or step size (e.g. another chain)
// // makes them again.
// struct factors_of_the_nuisance_rotation_t {
//     double eps = std::numeric_limits<double>::quiet_NaN();
//     Eigen::Matrix<double, -1, 1> M_us_vec;       // the nuisance metric they were made for
//     Eigen::Matrix<double, -1, 1> M_inv_us_vec;
//     Eigen::Matrix<double, -1, 1> cos_t;
//     Eigen::Matrix<double, -1, 1> sqrt_M_sin_t;
//     Eigen::Matrix<double, -1, 1> inv_sqrt_M_sin_t;
// };
//
// ALWAYS_INLINE const factors_of_the_nuisance_rotation_t& fn_factors_of_the_nuisance_rotation(
//                                                              const Eigen::Matrix<double, -1, 1> &M_us_vec,
//                                                              const Eigen::Matrix<double, -1, 1> &M_inv_us_vec,
//                                                              const double eps) {
//         thread_local factors_of_the_nuisance_rotation_t factors;
//         const bool made_for_this_metric_and_step_size =
//               (factors.eps == eps) &&
//               (factors.M_us_vec.size() == M_us_vec.size()) &&
//               (factors.M_inv_us_vec.size() == M_inv_us_vec.size()) &&
//               (factors.M_us_vec.array() == M_us_vec.array()).all() &&
//               (factors.M_inv_us_vec.array() == M_inv_us_vec.array()).all();
//         if (!made_for_this_metric_and_step_size) {
//               const Eigen::Matrix<double, -1, 1> sqrt_M     = M_us_vec.array().sqrt();
//               const Eigen::Matrix<double, -1, 1> inv_sqrt_M = M_inv_us_vec.array().sqrt();   // = 1/sqrt(M_i)
//               const Eigen::Matrix<double, -1, 1> angle      = eps * inv_sqrt_M;              // omega_i * eps
//               factors.cos_t = angle.array().cos();
//               const Eigen::Matrix<double, -1, 1> sin_t      = angle.array().sin();
//               factors.sqrt_M_sin_t     = (sqrt_M.array() * sin_t.array()).matrix();
//               factors.inv_sqrt_M_sin_t = (inv_sqrt_M.array() * sin_t.array()).matrix();
//               factors.M_us_vec = M_us_vec;
//               factors.M_inv_us_vec = M_inv_us_vec;
//               factors.eps = eps;
//         }
//         return factors;
// }
//
// THE PER-UNIT FACTORS OF THE NUISANCE ROTATION, KEPT BETWEEN TRAJECTORIES AND SHARED BY THE CHAINS
// ============================================================
// cos(eps / sqrt(M_i)), sqrt(M_i) sin(eps / sqrt(M_i)) and sin(eps / sqrt(M_i)) / sqrt(M_i) of every nuisance
// parameter, made with exactly the same expressions as in the leapfrog steps below (so every value is the same,
// to the last bit), made again only when the step size or the nuisance metric has changed since this thread's
// last trajectory, and ONE copy for every chain that has the same step size and metric (in the sampling phase:
// every chain), instead of one copy per thread: at large N the copies of every chain filled the shared cache
// (e.g. 3 x 25,000 values per chain, 16 chains per L3). Each thread keeps the factors of its last trajectory
// (shared_ptr), takes the process's latest factors when they are for its step size and metric, and otherwise
// makes its own and publishes them as the latest; so chains with different step sizes or metrics (e.g. in a
// burn-in) each keep their own, as before.
struct factors_of_the_nuisance_rotation_t {
    double eps = std::numeric_limits<double>::quiet_NaN();
    Eigen::Matrix<double, -1, 1> M_us_vec;       // the nuisance metric they were made for
    Eigen::Matrix<double, -1, 1> M_inv_us_vec;
    Eigen::Matrix<double, -1, 1> cos_t;
    Eigen::Matrix<double, -1, 1> sqrt_M_sin_t;
    Eigen::Matrix<double, -1, 1> inv_sqrt_M_sin_t;
};

ALWAYS_INLINE bool fn_factors_of_the_nuisance_rotation_are_for(const factors_of_the_nuisance_rotation_t &factors,
                                                               const Eigen::Matrix<double, -1, 1> &M_us_vec,
                                                               const Eigen::Matrix<double, -1, 1> &M_inv_us_vec,
                                                               const double eps) {

        return (factors.eps == eps) &&
               (factors.M_us_vec.size() == M_us_vec.size()) &&
               (factors.M_inv_us_vec.size() == M_inv_us_vec.size()) &&
               (factors.M_us_vec.array() == M_us_vec.array()).all() &&
               (factors.M_inv_us_vec.array() == M_inv_us_vec.array()).all();

}

ALWAYS_INLINE const factors_of_the_nuisance_rotation_t& fn_factors_of_the_nuisance_rotation(
                                                             const Eigen::Matrix<double, -1, 1> &M_us_vec,
                                                             const Eigen::Matrix<double, -1, 1> &M_inv_us_vec,
                                                             const double eps) {

        //// (the process's latest factors, read and replaced atomically; this thread's last ones)
        static std::shared_ptr<const factors_of_the_nuisance_rotation_t> latest_factors_of_the_process;
        thread_local std::shared_ptr<const factors_of_the_nuisance_rotation_t> factors_of_this_thread;
        if (factors_of_this_thread &&
            fn_factors_of_the_nuisance_rotation_are_for(*factors_of_this_thread, M_us_vec, M_inv_us_vec, eps)) {

              return *factors_of_this_thread;

        }
        std::shared_ptr<const factors_of_the_nuisance_rotation_t> latest =
              std::atomic_load(&latest_factors_of_the_process);
        if (latest && fn_factors_of_the_nuisance_rotation_are_for(*latest, M_us_vec, M_inv_us_vec, eps)) {
              factors_of_this_thread = latest;
              return *factors_of_this_thread;
        }
        auto factors = std::make_shared<factors_of_the_nuisance_rotation_t>();
        const Eigen::Matrix<double, -1, 1> sqrt_M     = M_us_vec.array().sqrt();
        const Eigen::Matrix<double, -1, 1> inv_sqrt_M = M_inv_us_vec.array().sqrt();   // = 1/sqrt(M_i)
        const Eigen::Matrix<double, -1, 1> angle      = eps * inv_sqrt_M;              // omega_i * eps
        factors->cos_t = angle.array().cos();
        const Eigen::Matrix<double, -1, 1> sin_t      = angle.array().sin();
        factors->sqrt_M_sin_t     = (sqrt_M.array() * sin_t.array()).matrix();
        factors->inv_sqrt_M_sin_t = (inv_sqrt_M.array() * sin_t.array()).matrix();
        factors->M_us_vec = M_us_vec;
        factors->M_inv_us_vec = M_inv_us_vec;
        factors->eps = eps;
        factors_of_this_thread = factors;
        std::atomic_store(&latest_factors_of_the_process,
                          std::shared_ptr<const factors_of_the_nuisance_rotation_t>(factors));
        return *factors_of_this_thread;

}



// ============================================================
// THE CHUNK-LOCAL LEAPFROG STEP OF THE NUISANCE VALUES (reduce_sum_chunked)
// ============================================================
// For a model library built with options(NicoStan.reduce_sum_chunk_local_leapfrog_step = TRUE) (a model in
// NicoStan's nuisance-block form: its nuisance values are read inside the chunks of its reduce_sum), the pass
// over all the nuisance values that follows each gradient of a leapfrog step (the closing half kick of the step,
// then the next step's half kick and rotation; after the last step the closing half kick alone) is made inside
// the gradient call, by each chunk on its own values, right after their gradient, whilst the values, their
// gradient and their velocities are still in the cache: at large N the N-sized vectors of a chain no longer fit
// in its share of the L3, and every pass over all N streams them from DRAM
// (reduce_sum_chunk_local_leapfrog_step.inc). The step is made here, by lambdas with the same arithmetic as the
// pass, through the function the chunks call; every range is made in whole blocks of 64 values (a part of a block
// in a local copy of the whole block), so that each value is made by the same part of the compiled loop as in the
// pass over all the values. The next values are written into this chain thread's arrays of the next values (the
// values the chunks read are never written during the gradient call); the values not made inside the chunks
// (those read outside their own chunk, whose gradient is complete only after all the chunks, and any value no
// chunk owns) are made here after the gradient call, and so is again any value made inside a chunk whose gradient
// then had adjoints from outside it; the arrays are then swapped with the chain's own (no copy). The same numbers
// as the pass, in the same order of operations of the kick-flow-kick step: the nuisance pass reads only the
// nuisance values, their velocities and their gradient, none of which the main parameters' half kicks and
// position update between them change.
////
//// the arrays of one step of the nuisance values: those it reads, and the next values it writes
struct nuisance_arrays_of_a_leapfrog_step_t {
    const double* theta = nullptr;
    const double* velocity = nullptr;
    const double* gradient = nullptr;
    const double* M_inv = nullptr;
    const double* theta_hat = nullptr;
    const double* cos_of_angle = nullptr;
    const double* sqrt_M_sin = nullptr;
    const double* inv_sqrt_M_sin = nullptr;
    double* theta_next = nullptr;      // (null for the closing half kick alone)
    double* velocity_next = nullptr;
};

//// the next values of the nuisance parameters on this chain thread, and the values read outside their own chunk
//// in its last gradient call (made again only when the number of nuisance values changes)
struct next_nuisance_values_of_this_chain_thread_t {
    Eigen::Matrix<double, -1, 1> theta_next;
    Eigen::Matrix<double, -1, 1> velocity_next;
    std::vector<int> values_read_outside_their_own_chunk_in_the_previous_gradient_call;
};

ALWAYS_INLINE next_nuisance_values_of_this_chain_thread_t& fn_next_nuisance_values_of_this_chain_thread(
                                                                 const int n_nuisance) {

        thread_local next_nuisance_values_of_this_chain_thread_t next_values;
        if ((next_values.theta_next.size() != n_nuisance) || (next_values.velocity_next.size() != n_nuisance)) {
              next_values.theta_next.resize(n_nuisance);
              next_values.velocity_next.resize(n_nuisance);
              next_values.values_read_outside_their_own_chunk_in_the_previous_gradient_call.clear();
        }
        return next_values;

}

//// the step of the values first to end - 1, in whole blocks of 64 values (straight on the arrays), a part of a
//// block in a local copy of the whole block whose other values are 0 (only the part's next values are copied
//// back); the gradient's local copy is one value past a 64-byte boundary, as lp_and_grad_outs.data() + 1 is
template <typename Step_of_the_values_first_to_end>
inline void fn_make_the_step_of_the_nuisance_values_first_to_end_in_blocks_of_64_values(
                   const nuisance_arrays_of_a_leapfrog_step_t &arrays,
                   const int n_nuisance,
                   const int first,
                   const int end,
                   Step_of_the_values_first_to_end &step_of_the_values) {

        const int block = values_per_block_of_the_nuisance_pass_on_the_WCP_threads;
        int value = first;
        while (value < end) {
              const int start_of_block = (value / block) * block;
              const int end_of_block = std::min(n_nuisance, start_of_block + block);
              if ((value == start_of_block) && (end >= end_of_block)) {
                    int end_of_whole_blocks = end_of_block;
                    while ((end_of_whole_blocks < n_nuisance) &&
                           (std::min(n_nuisance, end_of_whole_blocks + block) <= end)) {

                          end_of_whole_blocks = std::min(n_nuisance, end_of_whole_blocks + block);

                    }
                    step_of_the_values(arrays, value, end_of_whole_blocks);
                    value = end_of_whole_blocks;
                    continue;
              }
              const int end_of_part = std::min(end, end_of_block);
              const int n_values_of_block = end_of_block - start_of_block;
              alignas(64) double theta[values_per_block_of_the_nuisance_pass_on_the_WCP_threads];
              alignas(64) double velocity[values_per_block_of_the_nuisance_pass_on_the_WCP_threads];
              alignas(64) double gradient_one_value_past_a_64_byte_boundary[
                    values_per_block_of_the_nuisance_pass_on_the_WCP_threads + 1];
              alignas(64) double M_inv[values_per_block_of_the_nuisance_pass_on_the_WCP_threads];
              alignas(64) double theta_hat[values_per_block_of_the_nuisance_pass_on_the_WCP_threads];
              alignas(64) double cos_of_angle[values_per_block_of_the_nuisance_pass_on_the_WCP_threads];
              alignas(64) double sqrt_M_sin[values_per_block_of_the_nuisance_pass_on_the_WCP_threads];
              alignas(64) double inv_sqrt_M_sin[values_per_block_of_the_nuisance_pass_on_the_WCP_threads];
              alignas(64) double theta_next[values_per_block_of_the_nuisance_pass_on_the_WCP_threads];
              alignas(64) double velocity_next[values_per_block_of_the_nuisance_pass_on_the_WCP_threads];
              double* gradient = gradient_one_value_past_a_64_byte_boundary + 1;
              const auto fn_local_copy = [&](const double* source, double* local) -> const double* {
                    if (source == nullptr) return nullptr;
                    for (int j = 0; j < n_values_of_block; ++j) local[j] = 0.0;
                    for (int j = value; j < end_of_part; ++j) local[j - start_of_block] = source[j];
                    return local;
              };
              nuisance_arrays_of_a_leapfrog_step_t local_arrays;
              local_arrays.theta          = fn_local_copy(arrays.theta, theta);
              local_arrays.velocity       = fn_local_copy(arrays.velocity, velocity);
              local_arrays.gradient       = fn_local_copy(arrays.gradient, gradient);
              local_arrays.M_inv          = fn_local_copy(arrays.M_inv, M_inv);
              local_arrays.theta_hat      = fn_local_copy(arrays.theta_hat, theta_hat);
              local_arrays.cos_of_angle   = fn_local_copy(arrays.cos_of_angle, cos_of_angle);
              local_arrays.sqrt_M_sin     = fn_local_copy(arrays.sqrt_M_sin, sqrt_M_sin);
              local_arrays.inv_sqrt_M_sin = fn_local_copy(arrays.inv_sqrt_M_sin, inv_sqrt_M_sin);
              local_arrays.theta_next     = (arrays.theta_next == nullptr) ? nullptr : theta_next;
              local_arrays.velocity_next  = velocity_next;
              step_of_the_values(local_arrays, 0, n_values_of_block);
              for (int j = value; j < end_of_part; ++j) {
                    if (arrays.theta_next != nullptr) arrays.theta_next[j] = theta_next[j - start_of_block];
                    arrays.velocity_next[j] = velocity_next[j - start_of_block];
              }
              value = end_of_part;
        }

}

//// (what the chunks call, through nicostan_chunk_local_leapfrog_step_t)
template <typename Step_of_the_values_first_to_end>
struct NicoStan_pass_over_the_nuisance_values_t {
    const nuisance_arrays_of_a_leapfrog_step_t* arrays;
    int n_nuisance;
    Step_of_the_values_first_to_end* step_of_the_values;
};

template <typename Step_of_the_values_first_to_end>
inline void fn_make_the_step_of_the_nuisance_values_first_to_end_called_from_the_chunks(
                   void* NicoStan_pass_over_the_values, int first, int end) {

        auto* pass = static_cast<NicoStan_pass_over_the_nuisance_values_t<Step_of_the_values_first_to_end>*>(
                           NicoStan_pass_over_the_values);
        fn_make_the_step_of_the_nuisance_values_first_to_end_in_blocks_of_64_values(*pass->arrays,
                                                                                    pass->n_nuisance, first, end,
                                                                                    *pass->step_of_the_values);

}

//// the gradient call of a leapfrog step with the chunk-local leapfrog step: true when the step of every nuisance
//// value was made (into the arrays of the next values: inside the chunks, and here for the rest), false when it
//// was not (the log density is not finite, or the library did not make it) and the sampler makes it itself
template <typename Gradient_call, typename Step_of_the_values_first_to_end>
ALWAYS_INLINE bool fn_gradient_call_with_the_chunk_local_leapfrog_step_of_the_nuisance_values(
                   Gradient_call &&gradient_call,
                   const Eigen::Matrix<double, -1, 1> &lp_and_grad_outs,
                   const nuisance_arrays_of_a_leapfrog_step_t &arrays,
                   Step_of_the_values_first_to_end &step_of_the_values,
                   const int n_nuisance,
                   std::vector<int> &values_read_outside_their_own_chunk_in_the_previous_gradient_call) {

        NicoStan_pass_over_the_nuisance_values_t<Step_of_the_values_first_to_end> pass{ &arrays, n_nuisance,
                                                                                         &step_of_the_values };
        nicostan_chunk_local_leapfrog_step_t step{};
        step.NicoStan_pass_over_the_values = &pass;
        step.fn_make_the_step_of_the_values_first_to_end =
              &fn_make_the_step_of_the_nuisance_values_first_to_end_called_from_the_chunks<
                    Step_of_the_values_first_to_end>;
        step.values_read_outside_their_own_chunk_in_the_previous_gradient_call =
              values_read_outside_their_own_chunk_in_the_previous_gradient_call.data();
        step.n_values_read_outside_their_own_chunk_in_the_previous_gradient_call =
              static_cast<int>(values_read_outside_their_own_chunk_in_the_previous_gradient_call.size());
        {
              nicostan_scope_of_the_chunk_local_leapfrog_step_of_a_gradient_call scope_of_the_step(&step);
              gradient_call();
        }
        if (!std::isfinite(lp_and_grad_outs(0))) return false;
        if (step.step_made_inside_the_chunks != 1) return false;
        //// ---- the values not made inside the chunks
        const int* runs = step.first_and_end_of_the_runs_of_values_stepped_inside_the_chunks;
        const int n_runs = step.n_runs_of_values_stepped_inside_the_chunks;
        int first_value_not_made = 0;
        for (int run = 0; run < n_runs; ++run) {
              if (runs[2 * run] > first_value_not_made) {
                    fn_make_the_step_of_the_nuisance_values_first_to_end_in_blocks_of_64_values(
                          arrays, n_nuisance, first_value_not_made, runs[2 * run], step_of_the_values);
              }
              first_value_not_made = std::max(first_value_not_made, runs[2 * run + 1]);
        }
        if (first_value_not_made < n_nuisance) {
              fn_make_the_step_of_the_nuisance_values_first_to_end_in_blocks_of_64_values(
                    arrays, n_nuisance, first_value_not_made, n_nuisance, step_of_the_values);
        }
        //// ---- the values made inside a chunk whose gradient then had adjoints from outside it: made again
        for (int k = 0; k < step.n_values_read_outside_their_own_chunk; ++k) {
              const int value = step.values_read_outside_their_own_chunk[k];
              //// (the first run that starts after the value)
              int low = 0;
              int high = n_runs;
              while (low < high) {
                    const int middle = (low + high) / 2;
                    if (runs[2 * middle] <= value) low = middle + 1;
                    else                           high = middle;
              }
              if ((low > 0) && (value < runs[2 * (low - 1) + 1])) {
                    fn_make_the_step_of_the_nuisance_values_first_to_end_in_blocks_of_64_values(
                          arrays, n_nuisance, value, value + 1, step_of_the_values);
              }
        }
        //// ---- the values read outside their own chunk, for the next gradient call
        values_read_outside_their_own_chunk_in_the_previous_gradient_call.assign(
              step.values_read_outside_their_own_chunk,
              step.values_read_outside_their_own_chunk + step.n_values_read_outside_their_own_chunk);
        return true;

}


// ============================================================
// DIFFUSION-PATHSPACE DUAL LEAPFROG — DENSE M for main
// ============================================================
// General version: supports M_us = A_us (Laplace precision)
// When theta_hat = 0, M_us = I: reduces to probit-specific version
// When theta_hat = posterior mean, M_us = posterior precision:
//   gives preconditioning + uniform rotation for normal-like nuisance
// ============================================================
ALWAYS_INLINE void leapfrog_integrator_dense_M_diffusion_HMC_dual_InPlace(   Eigen::Matrix<double, -1, 1> &velocity_main_vec_proposed_ref,
                                                                             Eigen::Matrix<double, -1, 1> &velocity_us_vec_proposed_ref,
                                                                             Eigen::Matrix<double, -1, 1> &theta_main_vec_proposed_ref,
                                                                             Eigen::Matrix<double, -1, 1> &theta_us_vec_proposed_ref,
                                                                             Eigen::Matrix<double, -1, 1> &lp_and_grad_outs,
                                                                             ////
                                                                             Eigen::Matrix<double, -1, 1> &us_theta_vec_current_segment,
                                                                             Eigen::Matrix<double, -1, 1> &us_velocity_vec_current_segment,
                                                                             ////
                                                                             const Eigen::Matrix<double, -1, 1> &theta_main_vec_initial_ref,
                                                                             const Eigen::Matrix<double, -1, 1> &theta_us_vec_initial_ref,
                                                                             const Eigen::Matrix<double, -1, -1> &M_inv_dense_main,
                                                                             const Eigen::Matrix<double, -1, 1> &M_inv_us_vec,
                                                                             const Eigen::Matrix<double, -1, 1> &M_us_vec,
                                                                             const Eigen::Matrix<double, -1, 1> &theta_hat_us_vec,
                                                                             const Eigen::Matrix<int, -1, -1> &y_ref,
                                                                             const int L_ii,
                                                                             const double eps,
                                                                             const std::string &Model_type,
                                                                             const bool force_autodiff, const bool force_PartialLog, const bool multi_attempts,
                                                                             const std::string &grad_option,
                                                                             const Model_fn_args_struct &Model_args_as_cpp_struct,
                                                                             const Stan_model_struct &Stan_model_as_cpp_struct,
                                                                             std::vector<LC_MVP_workspace_struct> &LC_MVP_ws_structs,
                                                                             const int n_threads_WCP,
                                                                             std::function<void(Eigen::Ref<Eigen::Matrix<double, -1, 1>>,
                                                                                                const std::string,
                                                                                                const bool, const bool, const bool,
                                                                                                const Eigen::Ref<const Eigen::Matrix<double, -1, 1>>,
                                                                                                const Eigen::Ref<const Eigen::Matrix<double, -1, 1>>,
                                                                                                const Eigen::Ref<const Eigen::Matrix<int, -1, -1>>,
                                                                                                const std::string,
                                                                                                const Model_fn_args_struct &,
                                                                                                const Stan_model_struct &,
                                                                                                std::vector<LC_MVP_workspace_struct> &,
                                                                                                const int &)> fn_lp_grad_InPlace
) {
  
        const int n_nuisance = Model_args_as_cpp_struct.n_nuisance;
        const int n_params_main = Model_args_as_cpp_struct.n_params_main;
        
        Eigen::Matrix<double, -1, 1> grad_main = lp_and_grad_outs.segment(1 + n_nuisance, n_params_main);
        
        // const Eigen::Matrix<double, -1, 1> sqrt_M     = M_us_vec.array().sqrt();
        // const Eigen::Matrix<double, -1, 1> inv_sqrt_M = M_inv_us_vec.array().sqrt();   // = 1/sqrt(M_i)
        // const Eigen::Matrix<double, -1, 1> angle      = eps * inv_sqrt_M;              // omega_i * eps
        // const Eigen::Matrix<double, -1, 1> cos_t      = angle.array().cos();
        // const Eigen::Matrix<double, -1, 1> sin_t      = angle.array().sin();
        // //// (the two products of the rotation, made once per trajectory, as the separate passes evaluated them)
        // const Eigen::Matrix<double, -1, 1> sqrt_M_sin_t     = (sqrt_M.array() * sin_t.array()).matrix();
        // const Eigen::Matrix<double, -1, 1> inv_sqrt_M_sin_t = (inv_sqrt_M.array() * sin_t.array()).matrix();
        //// (the factors of the rotation: kept between trajectories, made again only when eps or the metric
        ////  change)
        const factors_of_the_nuisance_rotation_t &factors_of_the_rotation =
              fn_factors_of_the_nuisance_rotation(M_us_vec, M_inv_us_vec, eps);
        const Eigen::Matrix<double, -1, 1> &cos_t            = factors_of_the_rotation.cos_t;
        const Eigen::Matrix<double, -1, 1> &sqrt_M_sin_t     = factors_of_the_rotation.sqrt_M_sin_t;
        const Eigen::Matrix<double, -1, 1> &inv_sqrt_M_sin_t = factors_of_the_rotation.inv_sqrt_M_sin_t;
        
        //// ---- the nuisance part of each step in ONE pass over the nuisance parameters (the same arithmetic, in
        ////      the same order, as the half kick, the rotation around theta_hat and, from the second step on, the
        ////      previous step's closing half kick written as separate passes); no copies of theta or v are made
        ////      (us_theta_vec_current_segment and us_velocity_vec_current_segment are not needed). Independent
        ////      elements: vectorised.
        const double half_eps = 0.5 * eps;
        double* __restrict velocity_us = velocity_us_vec_proposed_ref.data();
        double* __restrict theta_us = theta_us_vec_proposed_ref.data();
        const double* __restrict grad_us = lp_and_grad_outs.data() + 1;
        const double* __restrict M_inv_us = M_inv_us_vec.data();
        const double* __restrict theta_hat = theta_hat_us_vec.data();
        const double* __restrict cos_of_angle = cos_t.data();
        const double* __restrict sqrt_M_sin = sqrt_M_sin_t.data();
        const double* __restrict inv_sqrt_M_sin = inv_sqrt_M_sin_t.data();
        //// (over ranges of the values on the chain's own WCP threads when it has several:
        ////  the_TBB_arena_of_each_chain.hpp)
        const auto nuisance_kick_and_rotation = [&](const bool closing_kick_of_the_previous_step) {

          fn_pass_over_the_nuisance_values_on_the_WCP_threads_of_this_chain(n_nuisance, n_threads_WCP,
                                                                            [&](const int first, const int end) {

            for (int i = first; i < end; ++i) {
                const double kick = half_eps * M_inv_us[i] * ((grad_us[i] + theta_us[i]) - theta_hat[i]);
                const double velocity = closing_kick_of_the_previous_step ? (velocity_us[i] + kick) + kick
                                                                          : velocity_us[i] + kick;
                const double distance = theta_us[i] - theta_hat[i];
                theta_us[i] = cos_of_angle[i] * distance + sqrt_M_sin[i] * velocity + theta_hat[i];
                velocity_us[i] = cos_of_angle[i] * velocity - inv_sqrt_M_sin[i] * distance;
            }

          });

        };
        
        //// ---- the chunk-local leapfrog step (a model library built with it: see
        ////      fn_gradient_call_with_the_chunk_local_leapfrog_step_of_the_nuisance_values()): the step of the
        ////      values first to end - 1 into the next values, with the same arithmetic as the pass above (the
        ////      closing half kick of the previous step, the half kick and the rotation), and the closing half
        ////      kick alone after the last step, as the pass below the steps
        const bool chunk_local_leapfrog_step_of_the_library =
              (Model_type == "Stan") &&
              (Stan_model_as_cpp_struct.
                     nicostan_bs_log_density_gradient_with_the_nuisance_block_and_the_chunk_local_leapfrog_step !=
               nullptr);
        next_nuisance_values_of_this_chain_thread_t* next_values = chunk_local_leapfrog_step_of_the_library ?
                                                                   &fn_next_nuisance_values_of_this_chain_thread(
                                                                         n_nuisance) :
                                                                   nullptr;
        const auto closing_kick_kick_and_rotation_into_the_next_values = [half_eps](
                    const nuisance_arrays_of_a_leapfrog_step_t &arrays, const int first, const int end) {

            const bool closing_kick_of_the_previous_step = true;
            const double* __restrict velocity_us = arrays.velocity;
            const double* __restrict theta_us = arrays.theta;
            const double* __restrict grad_us = arrays.gradient;
            const double* __restrict M_inv_us = arrays.M_inv;
            const double* __restrict theta_hat = arrays.theta_hat;
            const double* __restrict cos_of_angle = arrays.cos_of_angle;
            const double* __restrict sqrt_M_sin = arrays.sqrt_M_sin;
            const double* __restrict inv_sqrt_M_sin = arrays.inv_sqrt_M_sin;
            double* __restrict theta_us_next = arrays.theta_next;
            double* __restrict velocity_us_next = arrays.velocity_next;
            for (int i = first; i < end; ++i) {
                const double kick = half_eps * M_inv_us[i] * ((grad_us[i] + theta_us[i]) - theta_hat[i]);
                const double velocity = closing_kick_of_the_previous_step ? (velocity_us[i] + kick) + kick
                                                                          : velocity_us[i] + kick;
                const double distance = theta_us[i] - theta_hat[i];
                theta_us_next[i] = cos_of_angle[i] * distance + sqrt_M_sin[i] * velocity + theta_hat[i];
                velocity_us_next[i] = cos_of_angle[i] * velocity - inv_sqrt_M_sin[i] * distance;
            }

        };
        const auto closing_kick_into_the_next_values = [half_eps](
                    const nuisance_arrays_of_a_leapfrog_step_t &arrays, const int first, const int end) {

            const double* __restrict velocity_us = arrays.velocity;
            const double* __restrict theta_us = arrays.theta;
            const double* __restrict grad_us = arrays.gradient;
            const double* __restrict M_inv_us = arrays.M_inv;
            const double* __restrict theta_hat = arrays.theta_hat;
            double* __restrict velocity_us_next = arrays.velocity_next;
            for (int i = first; i < end; ++i) {
                velocity_us_next[i] = velocity_us[i] +
                                      half_eps * M_inv_us[i] * ((grad_us[i] + theta_us[i]) - theta_hat[i]);
            }

        };
        //// (the arrays of a step: the chain's current nuisance values, their velocities and gradient, the
        ////  factors of the rotation, and this chain thread's arrays of the next values)
        const auto fn_nuisance_arrays_of_this_step = [&](const bool rotation_follows) {

            nuisance_arrays_of_a_leapfrog_step_t arrays;
            arrays.theta = theta_us;
            arrays.velocity = velocity_us;
            arrays.gradient = grad_us;
            arrays.M_inv = M_inv_us;
            arrays.theta_hat = theta_hat;
            arrays.cos_of_angle = cos_of_angle;
            arrays.sqrt_M_sin = sqrt_M_sin;
            arrays.inv_sqrt_M_sin = inv_sqrt_M_sin;
            arrays.theta_next = rotation_follows ? next_values->theta_next.data() : nullptr;
            arrays.velocity_next = next_values->velocity_next.data();
            return arrays;

        };
        //// (true once the gradient call of a step has made the nuisance step that follows it)
        bool nuisance_step_made_with_the_gradient = false;
        
        for (int l = 0; l < L_ii; l++) {
            if (Model_args_as_cpp_struct.burnin_leapfrog_steps != nullptr) {
                ++(*Model_args_as_cpp_struct.burnin_leapfrog_steps);
            }
          
               velocity_main_vec_proposed_ref.array() += (0.5 * eps * M_inv_dense_main * grad_main).array();
               
               theta_main_vec_proposed_ref.array() += eps * velocity_main_vec_proposed_ref.array();
               
               //// the nuisance (closing half kick of the previous step,) half kick and rotation, in one pass
               // if (l == 0) nuisance_kick_and_rotation(false);
               // else        nuisance_kick_and_rotation(true);
               //// (from the second step on, not when the previous step's gradient call made it inside the
               ////  chunks)
               if (l == 0)                                      nuisance_kick_and_rotation(false);
               else if (!nuisance_step_made_with_the_gradient) nuisance_kick_and_rotation(true);
               
               if (chunk_local_leapfrog_step_of_the_library) {
                     const bool rotation_follows = (l + 1 < L_ii);
                     const nuisance_arrays_of_a_leapfrog_step_t arrays = fn_nuisance_arrays_of_this_step(
                                                                           rotation_follows);
                     const auto gradient_call = [&]() {

                           fn_lp_grad_InPlace(lp_and_grad_outs,
                                              Model_type, force_autodiff, force_PartialLog, multi_attempts,
                                              theta_main_vec_proposed_ref, theta_us_vec_proposed_ref,
                                              y_ref, grad_option,
                                              Model_args_as_cpp_struct,
                                              Stan_model_as_cpp_struct,
                                              LC_MVP_ws_structs,
                                              n_threads_WCP);

                     };
                     nuisance_step_made_with_the_gradient = rotation_follows ?
                           fn_gradient_call_with_the_chunk_local_leapfrog_step_of_the_nuisance_values(
                                 gradient_call, lp_and_grad_outs, arrays,
                                 closing_kick_kick_and_rotation_into_the_next_values, n_nuisance,
                                 next_values->values_read_outside_their_own_chunk_in_the_previous_gradient_call) :
                           fn_gradient_call_with_the_chunk_local_leapfrog_step_of_the_nuisance_values(
                                 gradient_call, lp_and_grad_outs, arrays, closing_kick_into_the_next_values,
                                 n_nuisance,
                                 next_values->values_read_outside_their_own_chunk_in_the_previous_gradient_call);
                     if (nuisance_step_made_with_the_gradient) {
                           //// (the next values taken over: the storage of the vectors swapped, no copy)
                           if (rotation_follows) theta_us_vec_proposed_ref.swap(next_values->theta_next);
                           velocity_us_vec_proposed_ref.swap(next_values->velocity_next);
                           theta_us = theta_us_vec_proposed_ref.data();
                           velocity_us = velocity_us_vec_proposed_ref.data();
                     }
               } else {
                     fn_lp_grad_InPlace(lp_and_grad_outs,
                                        Model_type, force_autodiff, force_PartialLog, multi_attempts,
                                        theta_main_vec_proposed_ref, theta_us_vec_proposed_ref,
                                        y_ref, grad_option,
                                        Model_args_as_cpp_struct,
                                        Stan_model_as_cpp_struct,
                                        LC_MVP_ws_structs,
                                        n_threads_WCP);
               }
               if (!std::isfinite(lp_and_grad_outs(0))) return; 
               grad_main = lp_and_grad_outs.segment(1 + n_nuisance, n_params_main);
               
               velocity_main_vec_proposed_ref.array() += (0.5 * eps * M_inv_dense_main * grad_main).array();
               
               //// (the nuisance closing half kick: in the next step's pass, or below after the last step)
        }
        
        //// ---- the last step's closing half kick of the nuisance parameters (unless the last step's gradient
        ////      call made it inside the chunks)
        if (!nuisance_step_made_with_the_gradient) {
        fn_pass_over_the_nuisance_values_on_the_WCP_threads_of_this_chain(n_nuisance, n_threads_WCP,
                                                                          [&](const int first, const int end) {

            for (int i = first; i < end; ++i) {
                velocity_us[i] += half_eps * M_inv_us[i] * ((grad_us[i] + theta_us[i]) - theta_hat[i]);
            }

        });
        }
  
}


ALWAYS_INLINE void leapfrog_integrator_dense_M_diffusion_HMC_dual_Id_M_us_zero_theta_us_InPlace(   Eigen::Matrix<double, -1, 1> &velocity_main_vec_proposed_ref,
                                                                                                   Eigen::Matrix<double, -1, 1> &velocity_us_vec_proposed_ref,
                                                                                                   Eigen::Matrix<double, -1, 1> &theta_main_vec_proposed_ref,
                                                                                                   Eigen::Matrix<double, -1, 1> &theta_us_vec_proposed_ref,
                                                                                                   Eigen::Matrix<double, -1, 1> &lp_and_grad_outs,
                                                                                                   ////
                                                                                                   Eigen::Matrix<double, -1, 1> &us_theta_vec_current_segment,
                                                                                                   Eigen::Matrix<double, -1, 1> &us_velocity_vec_current_segment,
                                                                                                   ////
                                                                                                   const Eigen::Matrix<double, -1, 1> &theta_main_vec_initial_ref,
                                                                                                   const Eigen::Matrix<double, -1, 1> &theta_us_vec_initial_ref,
                                                                                                   const Eigen::Matrix<double, -1, -1> &M_inv_dense_main,
                                                                                                   const Eigen::Matrix<double, -1, 1> &M_us_vec,
                                                                                                   const Eigen::Matrix<int, -1, -1> &y_ref,
                                                                                                   const int L_ii,
                                                                                                   const double eps,
                                                                                                   const std::string &Model_type,
                                                                                                   const bool force_autodiff, const bool force_PartialLog, const bool multi_attempts,
                                                                                                   const std::string &grad_option,
                                                                                                   const Model_fn_args_struct &Model_args_as_cpp_struct,
                                                                                                   const Stan_model_struct &Stan_model_as_cpp_struct,
                                                                                                   std::vector<LC_MVP_workspace_struct> &LC_MVP_ws_structs,
                                                                                                   const int n_threads_WCP,
                                                                                                   std::function<void(Eigen::Ref<Eigen::Matrix<double, -1, 1>>,
                                                                                                                      const std::string,
                                                                                                                      const bool, const bool, const bool,
                                                                                                                      const Eigen::Ref<const Eigen::Matrix<double, -1, 1>>,
                                                                                                                      const Eigen::Ref<const Eigen::Matrix<double, -1, 1>>,
                                                                                                                      const Eigen::Ref<const Eigen::Matrix<int, -1, -1>>,
                                                                                                                      const std::string,
                                                                                                                      const Model_fn_args_struct &,
                                                                                                                      const Stan_model_struct &,
                                                                                                                      std::vector<LC_MVP_workspace_struct> &,
                                                                                                                      const int &)> fn_lp_grad_InPlace
) {
        
        const int n_nuisance = Model_args_as_cpp_struct.n_nuisance;
        const int n_params_main = Model_args_as_cpp_struct.n_params_main;
        
        Eigen::Matrix<double, -1, 1> grad_main = lp_and_grad_outs.segment(1 + n_nuisance, n_params_main);
        
        const double cos_eps = std::cos(eps);
        const double sin_eps = std::sin(eps);
        const double half_eps = 0.5 * eps;
        
        //// ---- the nuisance part of each step in ONE pass over the nuisance parameters (the same arithmetic, in
        ////      the same order, as the half kick, the rotation and, from the second step on, the previous step's
        ////      closing half kick written as separate passes): v += (eps/2) (grad + theta) [twice from the second
        ////      step], then (theta, v) <- (cos theta + sin v, cos v - sin theta). No copies of theta or v are made
        ////      (us_theta_vec_current_segment and us_velocity_vec_current_segment are not needed), so each step
        ////      reads velocity, gradient and position and writes velocity and position once (40 bytes per
        ////      nuisance parameter, where the separate passes moved 144). Independent elements: vectorised.
        double* __restrict velocity_us = velocity_us_vec_proposed_ref.data();
        double* __restrict theta_us = theta_us_vec_proposed_ref.data();
        const double* __restrict grad_us = lp_and_grad_outs.data() + 1;
        
        //// ---- the chunk-local leapfrog step (a model library built with it: see
        ////      fn_gradient_call_with_the_chunk_local_leapfrog_step_of_the_nuisance_values()): the step of the
        ////      values first to end - 1 into the next values, with the same arithmetic as the passes of the
        ////      steps below (the closing half kick of the previous step, the half kick and the rotation), and
        ////      the closing half kick alone after the last step
        const bool chunk_local_leapfrog_step_of_the_library =
              (Model_type == "Stan") &&
              (Stan_model_as_cpp_struct.
                     nicostan_bs_log_density_gradient_with_the_nuisance_block_and_the_chunk_local_leapfrog_step !=
               nullptr);
        next_nuisance_values_of_this_chain_thread_t* next_values = chunk_local_leapfrog_step_of_the_library ?
                                                                   &fn_next_nuisance_values_of_this_chain_thread(
                                                                         n_nuisance) :
                                                                   nullptr;
        const auto closing_kick_kick_and_rotation_into_the_next_values = [half_eps, cos_eps, sin_eps](
                    const nuisance_arrays_of_a_leapfrog_step_t &arrays, const int first, const int end) {

            const double* __restrict velocity_us = arrays.velocity;
            const double* __restrict theta_us = arrays.theta;
            const double* __restrict grad_us = arrays.gradient;
            double* __restrict theta_us_next = arrays.theta_next;
            double* __restrict velocity_us_next = arrays.velocity_next;
            for (int i = first; i < end; ++i) {
                const double kick = half_eps * (grad_us[i] + theta_us[i]);
                const double velocity = (velocity_us[i] + kick) + kick;
                const double theta = theta_us[i];
                theta_us_next[i] = cos_eps * theta + sin_eps * velocity;
                velocity_us_next[i] = cos_eps * velocity - sin_eps * theta;
            }

        };
        const auto closing_kick_into_the_next_values = [half_eps](
                    const nuisance_arrays_of_a_leapfrog_step_t &arrays, const int first, const int end) {

            const double* __restrict velocity_us = arrays.velocity;
            const double* __restrict theta_us = arrays.theta;
            const double* __restrict grad_us = arrays.gradient;
            double* __restrict velocity_us_next = arrays.velocity_next;
            for (int i = first; i < end; ++i) {
                velocity_us_next[i] = velocity_us[i] + half_eps * (grad_us[i] + theta_us[i]);
            }

        };
        //// (the arrays of a step: the chain's current nuisance values, their velocities and gradient, and this
        ////  chain thread's arrays of the next values)
        const auto fn_nuisance_arrays_of_this_step = [&](const bool rotation_follows) {

            nuisance_arrays_of_a_leapfrog_step_t arrays;
            arrays.theta = theta_us;
            arrays.velocity = velocity_us;
            arrays.gradient = grad_us;
            arrays.theta_next = rotation_follows ? next_values->theta_next.data() : nullptr;
            arrays.velocity_next = next_values->velocity_next.data();
            return arrays;

        };
        //// (true once the gradient call of a step has made the nuisance step that follows it)
        bool nuisance_step_made_with_the_gradient = false;
        
        for (int l = 0; l < L_ii; l++) {
            if (Model_args_as_cpp_struct.burnin_leapfrog_steps != nullptr) {
                ++(*Model_args_as_cpp_struct.burnin_leapfrog_steps);
            }
          
                // ---- Half kick: main (UNCHANGED; from the second step on, the previous step's closing half kick
                //      of the main parameters was already applied at the end of that step) ----
                velocity_main_vec_proposed_ref.array() += (0.5 * eps * M_inv_dense_main * grad_main).array();
                
                // ---- Position update: main (UNCHANGED) ----
                theta_main_vec_proposed_ref.array() += eps * velocity_main_vec_proposed_ref.array();
                
                // ---- Nuisance: (closing half kick of the previous step,) half kick and rotation, in one pass ----
                //// (over ranges of the values on the chain's own WCP threads when it has several:
                ////  the_TBB_arena_of_each_chain.hpp)
                if (l == 0) {
                  fn_pass_over_the_nuisance_values_on_the_WCP_threads_of_this_chain(n_nuisance, n_threads_WCP,
                                                                                    [&](const int first,
                                                                                        const int end) {

                    for (int i = first; i < end; ++i) {
                        const double kick = half_eps * (grad_us[i] + theta_us[i]);
                        const double velocity = velocity_us[i] + kick;
                        const double theta = theta_us[i];
                        theta_us[i] = cos_eps * theta + sin_eps * velocity;
                        velocity_us[i] = cos_eps * velocity - sin_eps * theta;
                    }

                  });
                // } else {
                //// (from the second step on, not when the previous step's gradient call made it inside the
                ////  chunks)
                } else if (!nuisance_step_made_with_the_gradient) {
                  fn_pass_over_the_nuisance_values_on_the_WCP_threads_of_this_chain(n_nuisance, n_threads_WCP,
                                                                                    [&](const int first,
                                                                                        const int end) {

                    for (int i = first; i < end; ++i) {
                        const double kick = half_eps * (grad_us[i] + theta_us[i]);
                        const double velocity = (velocity_us[i] + kick) + kick;
                        const double theta = theta_us[i];
                        theta_us[i] = cos_eps * theta + sin_eps * velocity;
                        velocity_us[i] = cos_eps * velocity - sin_eps * theta;
                    }

                  });
                }
                
                // ---- Recompute gradient ----
                if (chunk_local_leapfrog_step_of_the_library) {
                      const bool rotation_follows = (l + 1 < L_ii);
                      const nuisance_arrays_of_a_leapfrog_step_t arrays = fn_nuisance_arrays_of_this_step(
                                                                            rotation_follows);
                      const auto gradient_call = [&]() {

                            fn_lp_grad_InPlace(  lp_and_grad_outs,
                                                 Model_type, force_autodiff, force_PartialLog, multi_attempts,
                                                 theta_main_vec_proposed_ref, theta_us_vec_proposed_ref,
                                                 y_ref, grad_option,
                                                 Model_args_as_cpp_struct,
                                                 Stan_model_as_cpp_struct,
                                                 LC_MVP_ws_structs,
                                                 n_threads_WCP);

                      };
                      nuisance_step_made_with_the_gradient = rotation_follows ?
                            fn_gradient_call_with_the_chunk_local_leapfrog_step_of_the_nuisance_values(
                                  gradient_call, lp_and_grad_outs, arrays,
                                  closing_kick_kick_and_rotation_into_the_next_values, n_nuisance,
                                  next_values->
                                        values_read_outside_their_own_chunk_in_the_previous_gradient_call) :
                            fn_gradient_call_with_the_chunk_local_leapfrog_step_of_the_nuisance_values(
                                  gradient_call, lp_and_grad_outs, arrays, closing_kick_into_the_next_values,
                                  n_nuisance,
                                  next_values->
                                        values_read_outside_their_own_chunk_in_the_previous_gradient_call);
                      if (nuisance_step_made_with_the_gradient) {
                            //// (the next values taken over: the storage of the vectors swapped, no copy)
                            if (rotation_follows) theta_us_vec_proposed_ref.swap(next_values->theta_next);
                            velocity_us_vec_proposed_ref.swap(next_values->velocity_next);
                            theta_us = theta_us_vec_proposed_ref.data();
                            velocity_us = velocity_us_vec_proposed_ref.data();
                      }
                } else {
                      fn_lp_grad_InPlace(  lp_and_grad_outs,
                                           Model_type, force_autodiff, force_PartialLog, multi_attempts,
                                           theta_main_vec_proposed_ref, theta_us_vec_proposed_ref,
                                           y_ref, grad_option,
                                           Model_args_as_cpp_struct,
                                           Stan_model_as_cpp_struct, 
                                           LC_MVP_ws_structs,
                                           n_threads_WCP);
                }
                if (!std::isfinite(lp_and_grad_outs(0))) return; 
                grad_main = lp_and_grad_outs.segment(1 + n_nuisance, n_params_main);
                
                // ---- Half kick: main (UNCHANGED) ----
                velocity_main_vec_proposed_ref.array() += (0.5 * eps * M_inv_dense_main * grad_main).array();
                
                // ---- Half kick: nuisance (with new theta): applied in the next step's pass, or below after the
                //      last step ----
        }
        
        // ---- the last step's closing half kick of the nuisance parameters (unless the last step's gradient
        //      call made it inside the chunks) ----
        if (!nuisance_step_made_with_the_gradient) {
        fn_pass_over_the_nuisance_values_on_the_WCP_threads_of_this_chain(n_nuisance, n_threads_WCP,
                                                                          [&](const int first, const int end) {

            for (int i = first; i < end; ++i) velocity_us[i] += half_eps * (grad_us[i] + theta_us[i]);

        });
        }
  
}



// ============================================================
// DIFFUSION-PATHSPACE DUAL LEAPFROG — DIAG M for main
// ============================================================
ALWAYS_INLINE void leapfrog_integrator_diag_M_diffusion_HMC_dual_InPlace(  Eigen::Matrix<double, -1, 1> &velocity_main_vec_proposed_ref,
                                                                           Eigen::Matrix<double, -1, 1> &velocity_us_vec_proposed_ref,
                                                                           Eigen::Matrix<double, -1, 1> &theta_main_vec_proposed_ref,
                                                                           Eigen::Matrix<double, -1, 1> &theta_us_vec_proposed_ref,
                                                                           Eigen::Matrix<double, -1, 1> &lp_and_grad_outs,
                                                                           ////
                                                                           Eigen::Matrix<double, -1, 1> &us_theta_vec_current_segment,
                                                                           Eigen::Matrix<double, -1, 1> &us_velocity_vec_current_segment,
                                                                           ////
                                                                           const Eigen::Matrix<double, -1, 1> &theta_main_vec_initial_ref,
                                                                           const Eigen::Matrix<double, -1, 1> &theta_us_vec_initial_ref,
                                                                           const Eigen::Matrix<double, -1, 1> &M_inv_main_vec,
                                                                           const Eigen::Matrix<double, -1, 1> &M_inv_us_vec,
                                                                           const Eigen::Matrix<double, -1, 1> &M_us_vec,
                                                                           const Eigen::Matrix<double, -1, 1> &theta_hat_us_vec,
                                                                           const Eigen::Matrix<int, -1, -1> &y_ref,
                                                                           const int L_ii,
                                                                           const double eps,
                                                                           const std::string &Model_type,
                                                                           const bool force_autodiff, const bool force_PartialLog, const bool multi_attempts,
                                                                           const std::string &grad_option,
                                                                           const Model_fn_args_struct &Model_args_as_cpp_struct,
                                                                           const Stan_model_struct &Stan_model_as_cpp_struct,
                                                                           std::vector<LC_MVP_workspace_struct> &LC_MVP_ws_structs,
                                                                           const int n_threads_WCP,
                                                                           std::function<void(Eigen::Ref<Eigen::Matrix<double, -1, 1>>,
                                                                                              const std::string,
                                                                                              const bool, const bool, const bool,
                                                                                              const Eigen::Ref<const Eigen::Matrix<double, -1, 1>>,
                                                                                              const Eigen::Ref<const Eigen::Matrix<double, -1, 1>>,
                                                                                              const Eigen::Ref<const Eigen::Matrix<int, -1, -1>>,
                                                                                              const std::string,
                                                                                              const Model_fn_args_struct &,
                                                                                              const Stan_model_struct &,
                                                                                              std::vector<LC_MVP_workspace_struct> &,
                                                                                              const int &)> fn_lp_grad_InPlace
) {
  
        const int n_nuisance = Model_args_as_cpp_struct.n_nuisance;
        const int n_params_main = Model_args_as_cpp_struct.n_params_main;
        
        Eigen::Matrix<double, -1, 1> grad_main = lp_and_grad_outs.segment(1 + n_nuisance, n_params_main);
        
        // const Eigen::Matrix<double, -1, 1> sqrt_M     = M_us_vec.array().sqrt();
        // const Eigen::Matrix<double, -1, 1> inv_sqrt_M = M_inv_us_vec.array().sqrt();   // = 1/sqrt(M_i)
        // const Eigen::Matrix<double, -1, 1> angle      = eps * inv_sqrt_M;              // omega_i * eps
        // const Eigen::Matrix<double, -1, 1> cos_t      = angle.array().cos();
        // const Eigen::Matrix<double, -1, 1> sin_t      = angle.array().sin();
        // //// (the two products of the rotation, made once per trajectory, as the separate passes evaluated them)
        // const Eigen::Matrix<double, -1, 1> sqrt_M_sin_t     = (sqrt_M.array() * sin_t.array()).matrix();
        // const Eigen::Matrix<double, -1, 1> inv_sqrt_M_sin_t = (inv_sqrt_M.array() * sin_t.array()).matrix();
        //// (the factors of the rotation: kept between trajectories, made again only when eps or the metric
        ////  change)
        const factors_of_the_nuisance_rotation_t &factors_of_the_rotation =
              fn_factors_of_the_nuisance_rotation(M_us_vec, M_inv_us_vec, eps);
        const Eigen::Matrix<double, -1, 1> &cos_t            = factors_of_the_rotation.cos_t;
        const Eigen::Matrix<double, -1, 1> &sqrt_M_sin_t     = factors_of_the_rotation.sqrt_M_sin_t;
        const Eigen::Matrix<double, -1, 1> &inv_sqrt_M_sin_t = factors_of_the_rotation.inv_sqrt_M_sin_t;
        
        //// ---- the nuisance part of each step in ONE pass over the nuisance parameters (the same arithmetic, in
        ////      the same order, as the half kick, the rotation around theta_hat and, from the second step on, the
        ////      previous step's closing half kick written as separate passes); no copies of theta or v are made
        ////      (us_theta_vec_current_segment and us_velocity_vec_current_segment are not needed). Independent
        ////      elements: vectorised.
        const double half_eps = 0.5 * eps;
        double* __restrict velocity_us = velocity_us_vec_proposed_ref.data();
        double* __restrict theta_us = theta_us_vec_proposed_ref.data();
        const double* __restrict grad_us = lp_and_grad_outs.data() + 1;
        const double* __restrict M_inv_us = M_inv_us_vec.data();
        const double* __restrict theta_hat = theta_hat_us_vec.data();
        const double* __restrict cos_of_angle = cos_t.data();
        const double* __restrict sqrt_M_sin = sqrt_M_sin_t.data();
        const double* __restrict inv_sqrt_M_sin = inv_sqrt_M_sin_t.data();
        //// (over ranges of the values on the chain's own WCP threads when it has several:
        ////  the_TBB_arena_of_each_chain.hpp)
        const auto nuisance_kick_and_rotation = [&](const bool closing_kick_of_the_previous_step) {

          fn_pass_over_the_nuisance_values_on_the_WCP_threads_of_this_chain(n_nuisance, n_threads_WCP,
                                                                            [&](const int first, const int end) {

            for (int i = first; i < end; ++i) {
                const double kick = half_eps * M_inv_us[i] * ((grad_us[i] + theta_us[i]) - theta_hat[i]);
                const double velocity = closing_kick_of_the_previous_step ? (velocity_us[i] + kick) + kick
                                                                          : velocity_us[i] + kick;
                const double distance = theta_us[i] - theta_hat[i];
                theta_us[i] = cos_of_angle[i] * distance + sqrt_M_sin[i] * velocity + theta_hat[i];
                velocity_us[i] = cos_of_angle[i] * velocity - inv_sqrt_M_sin[i] * distance;
            }

          });

        };
        
        //// ---- the chunk-local leapfrog step (a model library built with it: see
        ////      fn_gradient_call_with_the_chunk_local_leapfrog_step_of_the_nuisance_values()): the step of the
        ////      values first to end - 1 into the next values, with the same arithmetic as the pass above (the
        ////      closing half kick of the previous step, the half kick and the rotation), and the closing half
        ////      kick alone after the last step, as the pass below the steps
        const bool chunk_local_leapfrog_step_of_the_library =
              (Model_type == "Stan") &&
              (Stan_model_as_cpp_struct.
                     nicostan_bs_log_density_gradient_with_the_nuisance_block_and_the_chunk_local_leapfrog_step !=
               nullptr);
        next_nuisance_values_of_this_chain_thread_t* next_values = chunk_local_leapfrog_step_of_the_library ?
                                                                   &fn_next_nuisance_values_of_this_chain_thread(
                                                                         n_nuisance) :
                                                                   nullptr;
        const auto closing_kick_kick_and_rotation_into_the_next_values = [half_eps](
                    const nuisance_arrays_of_a_leapfrog_step_t &arrays, const int first, const int end) {

            const bool closing_kick_of_the_previous_step = true;
            const double* __restrict velocity_us = arrays.velocity;
            const double* __restrict theta_us = arrays.theta;
            const double* __restrict grad_us = arrays.gradient;
            const double* __restrict M_inv_us = arrays.M_inv;
            const double* __restrict theta_hat = arrays.theta_hat;
            const double* __restrict cos_of_angle = arrays.cos_of_angle;
            const double* __restrict sqrt_M_sin = arrays.sqrt_M_sin;
            const double* __restrict inv_sqrt_M_sin = arrays.inv_sqrt_M_sin;
            double* __restrict theta_us_next = arrays.theta_next;
            double* __restrict velocity_us_next = arrays.velocity_next;
            for (int i = first; i < end; ++i) {
                const double kick = half_eps * M_inv_us[i] * ((grad_us[i] + theta_us[i]) - theta_hat[i]);
                const double velocity = closing_kick_of_the_previous_step ? (velocity_us[i] + kick) + kick
                                                                          : velocity_us[i] + kick;
                const double distance = theta_us[i] - theta_hat[i];
                theta_us_next[i] = cos_of_angle[i] * distance + sqrt_M_sin[i] * velocity + theta_hat[i];
                velocity_us_next[i] = cos_of_angle[i] * velocity - inv_sqrt_M_sin[i] * distance;
            }

        };
        const auto closing_kick_into_the_next_values = [half_eps](
                    const nuisance_arrays_of_a_leapfrog_step_t &arrays, const int first, const int end) {

            const double* __restrict velocity_us = arrays.velocity;
            const double* __restrict theta_us = arrays.theta;
            const double* __restrict grad_us = arrays.gradient;
            const double* __restrict M_inv_us = arrays.M_inv;
            const double* __restrict theta_hat = arrays.theta_hat;
            double* __restrict velocity_us_next = arrays.velocity_next;
            for (int i = first; i < end; ++i) {
                velocity_us_next[i] = velocity_us[i] +
                                      half_eps * M_inv_us[i] * ((grad_us[i] + theta_us[i]) - theta_hat[i]);
            }

        };
        //// (the arrays of a step: the chain's current nuisance values, their velocities and gradient, the
        ////  factors of the rotation, and this chain thread's arrays of the next values)
        const auto fn_nuisance_arrays_of_this_step = [&](const bool rotation_follows) {

            nuisance_arrays_of_a_leapfrog_step_t arrays;
            arrays.theta = theta_us;
            arrays.velocity = velocity_us;
            arrays.gradient = grad_us;
            arrays.M_inv = M_inv_us;
            arrays.theta_hat = theta_hat;
            arrays.cos_of_angle = cos_of_angle;
            arrays.sqrt_M_sin = sqrt_M_sin;
            arrays.inv_sqrt_M_sin = inv_sqrt_M_sin;
            arrays.theta_next = rotation_follows ? next_values->theta_next.data() : nullptr;
            arrays.velocity_next = next_values->velocity_next.data();
            return arrays;

        };
        //// (true once the gradient call of a step has made the nuisance step that follows it)
        bool nuisance_step_made_with_the_gradient = false;
        
        for (int l = 0; l < L_ii; l++) {
            if (Model_args_as_cpp_struct.burnin_leapfrog_steps != nullptr) {
                ++(*Model_args_as_cpp_struct.burnin_leapfrog_steps);
            }
          
               velocity_main_vec_proposed_ref.array() += 0.5 * eps * M_inv_main_vec.array() * grad_main.array();
               
               theta_main_vec_proposed_ref.array() += eps * velocity_main_vec_proposed_ref.array();
               
               //// the nuisance (closing half kick of the previous step,) half kick and rotation, in one pass
               // if (l == 0) nuisance_kick_and_rotation(false);
               // else        nuisance_kick_and_rotation(true);
               //// (from the second step on, not when the previous step's gradient call made it inside the
               ////  chunks)
               if (l == 0)                                      nuisance_kick_and_rotation(false);
               else if (!nuisance_step_made_with_the_gradient) nuisance_kick_and_rotation(true);
               
               if (chunk_local_leapfrog_step_of_the_library) {
                     const bool rotation_follows = (l + 1 < L_ii);
                     const nuisance_arrays_of_a_leapfrog_step_t arrays = fn_nuisance_arrays_of_this_step(
                                                                           rotation_follows);
                     const auto gradient_call = [&]() {

                           fn_lp_grad_InPlace(lp_and_grad_outs,
                                              Model_type, force_autodiff, force_PartialLog, multi_attempts,
                                              theta_main_vec_proposed_ref, theta_us_vec_proposed_ref,
                                              y_ref, grad_option,
                                              Model_args_as_cpp_struct,
                                              Stan_model_as_cpp_struct,
                                              LC_MVP_ws_structs,
                                              n_threads_WCP);

                     };
                     nuisance_step_made_with_the_gradient = rotation_follows ?
                           fn_gradient_call_with_the_chunk_local_leapfrog_step_of_the_nuisance_values(
                                 gradient_call, lp_and_grad_outs, arrays,
                                 closing_kick_kick_and_rotation_into_the_next_values, n_nuisance,
                                 next_values->values_read_outside_their_own_chunk_in_the_previous_gradient_call) :
                           fn_gradient_call_with_the_chunk_local_leapfrog_step_of_the_nuisance_values(
                                 gradient_call, lp_and_grad_outs, arrays, closing_kick_into_the_next_values,
                                 n_nuisance,
                                 next_values->values_read_outside_their_own_chunk_in_the_previous_gradient_call);
                     if (nuisance_step_made_with_the_gradient) {
                           //// (the next values taken over: the storage of the vectors swapped, no copy)
                           if (rotation_follows) theta_us_vec_proposed_ref.swap(next_values->theta_next);
                           velocity_us_vec_proposed_ref.swap(next_values->velocity_next);
                           theta_us = theta_us_vec_proposed_ref.data();
                           velocity_us = velocity_us_vec_proposed_ref.data();
                     }
               } else {
                     fn_lp_grad_InPlace(lp_and_grad_outs,
                                        Model_type, force_autodiff, force_PartialLog, multi_attempts,
                                        theta_main_vec_proposed_ref, theta_us_vec_proposed_ref,
                                        y_ref, grad_option,
                                        Model_args_as_cpp_struct,
                                        Stan_model_as_cpp_struct,
                                        LC_MVP_ws_structs,
                                        n_threads_WCP);
               }
               if (!std::isfinite(lp_and_grad_outs(0))) return; 
               grad_main = lp_and_grad_outs.segment(1 + n_nuisance, n_params_main);
               
               velocity_main_vec_proposed_ref.array() += 0.5 * eps * M_inv_main_vec.array() * grad_main.array();
               
               //// (the nuisance closing half kick: in the next step's pass, or below after the last step)
        }
        
        //// ---- the last step's closing half kick of the nuisance parameters (unless the last step's gradient
        ////      call made it inside the chunks)
        if (!nuisance_step_made_with_the_gradient) {
        fn_pass_over_the_nuisance_values_on_the_WCP_threads_of_this_chain(n_nuisance, n_threads_WCP,
                                                                          [&](const int first, const int end) {

            for (int i = first; i < end; ++i) {
                velocity_us[i] += half_eps * M_inv_us[i] * ((grad_us[i] + theta_us[i]) - theta_hat[i]);
            }

        });
        }
  
}



ALWAYS_INLINE void leapfrog_integrator_diag_M_diffusion_HMC_dual_Id_M_us_zero_theta_us_InPlace(  Eigen::Matrix<double, -1, 1> &velocity_main_vec_proposed_ref,
                                                                                                 Eigen::Matrix<double, -1, 1> &velocity_us_vec_proposed_ref,
                                                                                                 Eigen::Matrix<double, -1, 1> &theta_main_vec_proposed_ref,
                                                                                                 Eigen::Matrix<double, -1, 1> &theta_us_vec_proposed_ref,
                                                                                                 Eigen::Matrix<double, -1, 1> &lp_and_grad_outs,
                                                                                                 ////
                                                                                                 Eigen::Matrix<double, -1, 1> &us_theta_vec_current_segment,
                                                                                                 Eigen::Matrix<double, -1, 1> &us_velocity_vec_current_segment,
                                                                                                 ////
                                                                                                 const Eigen::Matrix<double, -1, 1> &theta_main_vec_initial_ref,
                                                                                                 const Eigen::Matrix<double, -1, 1> &theta_us_vec_initial_ref,
                                                                                                 const Eigen::Matrix<double, -1, 1> &M_inv_main_vec,
                                                                                                 const Eigen::Matrix<double, -1, 1> &M_us_vec,
                                                                                                 const Eigen::Matrix<int, -1, -1> &y_ref,
                                                                                                 const int L_ii,
                                                                                                 const double eps,
                                                                                                 const std::string &Model_type,
                                                                                                 const bool force_autodiff, const bool force_PartialLog, const bool multi_attempts,
                                                                                                 const std::string &grad_option,
                                                                                                 const Model_fn_args_struct &Model_args_as_cpp_struct,
                                                                                                 const Stan_model_struct &Stan_model_as_cpp_struct,
                                                                                                 std::vector<LC_MVP_workspace_struct> &LC_MVP_ws_structs,
                                                                                                 const int n_threads_WCP,
                                                                                                 std::function<void(Eigen::Ref<Eigen::Matrix<double, -1, 1>>,
                                                                                                                    const std::string,
                                                                                                                    const bool, const bool, const bool,
                                                                                                                    const Eigen::Ref<const Eigen::Matrix<double, -1, 1>>,
                                                                                                                    const Eigen::Ref<const Eigen::Matrix<double, -1, 1>>,
                                                                                                                    const Eigen::Ref<const Eigen::Matrix<int, -1, -1>>,
                                                                                                                    const std::string,
                                                                                                                    const Model_fn_args_struct &,
                                                                                                                    const Stan_model_struct &,
                                                                                                                    std::vector<LC_MVP_workspace_struct> &,
                                                                                                                    const int &)> fn_lp_grad_InPlace
) {
  
        const int n_nuisance = Model_args_as_cpp_struct.n_nuisance;
        const int n_params_main = Model_args_as_cpp_struct.n_params_main;
        
        Eigen::Matrix<double, -1, 1> grad_main = lp_and_grad_outs.segment(1 + n_nuisance, n_params_main);
        
        const double cos_eps = std::cos(eps);
        const double sin_eps = std::sin(eps);
        
        for (int l = 0; l < L_ii; l++) {
            if (Model_args_as_cpp_struct.burnin_leapfrog_steps != nullptr) {
                ++(*Model_args_as_cpp_struct.burnin_leapfrog_steps);
            }
                
                // ---- Half kick: main (UNCHANGED) ----
                velocity_main_vec_proposed_ref.array() += 0.5 * eps * M_inv_main_vec.array() * grad_main.array();
                 
                // ---- Half kick: nuisance ----
                // v += (eps/2) * (M⁻¹ * grad_log_post + (θ - θ̂))
                velocity_us_vec_proposed_ref.array() += 0.5 * eps * (lp_and_grad_outs.segment(1, n_nuisance).array() + theta_us_vec_proposed_ref.array());
                                                                     
                
                // ---- Position update: main (UNCHANGED) ----
                theta_main_vec_proposed_ref.array() += eps * velocity_main_vec_proposed_ref.array();
                
                // ---- Position update: nuisance (ROTATION around θ̂) ----
                us_theta_vec_current_segment = theta_us_vec_proposed_ref;
                us_velocity_vec_current_segment      = velocity_us_vec_proposed_ref;
                
                theta_us_vec_proposed_ref.array()    = cos_eps * us_theta_vec_current_segment.array()    + sin_eps * us_velocity_vec_current_segment.array();
                velocity_us_vec_proposed_ref.array() = cos_eps * us_velocity_vec_current_segment.array() - sin_eps * us_theta_vec_current_segment.array();
                
                // ---- Recompute gradient ----
                fn_lp_grad_InPlace(  lp_and_grad_outs,
                                     Model_type, force_autodiff, force_PartialLog, multi_attempts,
                                     theta_main_vec_proposed_ref, theta_us_vec_proposed_ref,
                                     y_ref, grad_option,
                                     Model_args_as_cpp_struct,
                                     Stan_model_as_cpp_struct, 
                                     LC_MVP_ws_structs,
                                     n_threads_WCP);
                if (!std::isfinite(lp_and_grad_outs(0))) return; 
                grad_main = lp_and_grad_outs.segment(1 + n_nuisance, n_params_main);
                
                // ---- Half kick: main (UNCHANGED) ----
                velocity_main_vec_proposed_ref.array() += 0.5 * eps * M_inv_main_vec.array() * grad_main.array();
                
                // ---- Half kick: nuisance (with new θ) ----
                velocity_us_vec_proposed_ref.array() += 0.5 * eps * (lp_and_grad_outs.segment(1, n_nuisance).array() + theta_us_vec_proposed_ref.array());
                                                
        }
  
}

 

// ============================================================
// SINGLE ITERATION — diffusion-pathspace dual (general M_us)
// ============================================================
// Changes from standard dual:
//   1. Nuisance velocity sampled as N(0, M_us⁻¹)
//   2. Calls diffusion leapfrog (rotation around θ̂ + corrected kick)
//   3. Nuisance kinetic energy = ½v_us'M_us v_us
// ============================================================
 
template<typename T = RNG_TYPE_dqrng>
ALWAYS_INLINE void fn_diffusion_HMC_dual_single_iter_InPlace_process(    HMCResult &result_input,
                                                                         T &rng_main,
                                                                         T &rng_nuisance,
                                                                         const std::string &Model_type,
                                                                         const bool force_autodiff,
                                                                         const bool force_PartialLog,
                                                                         const bool multi_attempts,
                                                                         const Eigen::Matrix<int, -1, -1> &y_ref,
                                                                         const Model_fn_args_struct &Model_args_as_cpp_struct,
                                                                         EHMC_fn_args_struct &EHMC_args_as_cpp_struct,
                                                                         const EHMC_Metric_struct &EHMC_Metric_struct_as_cpp_struct,
                                                                         const Stan_model_struct &Stan_model_as_cpp_struct,
                                                                         std::vector<LC_MVP_workspace_struct> &LC_MVP_ws_structs,
                                                                         const int n_threads_WCP,
                                                                         bool *current_lp_grad_valid = nullptr
) {
  
  const int N = Model_args_as_cpp_struct.N;
  const int n_nuisance = Model_args_as_cpp_struct.n_nuisance;
  const int n_params_main = Model_args_as_cpp_struct.n_params_main;
  const int n_params = n_params_main + n_nuisance;
  
  const std::string grad_option = "all";
  const std::string metric_shape_main = EHMC_Metric_struct_as_cpp_struct.metric_shape_main;
  
  double U_x_initial = 0.0;
  double U_x_prop = 0.0;
  double log_posterior_prop = 0.0;
  double log_posterior_0 = 0.0;
  double log_ratio = 0.0;
  double energy_old = 0.0;
  double energy_new = 0.0;
  
  result_input.main_theta_vec_0() = result_input.main_theta_vec();
  result_input.us_theta_vec_0()   = result_input.us_theta_vec(); 
  
  {
    // ---- Draw velocity for main: ----
    {
      Eigen::Matrix<double, -1, 1> std_norm_vec_main = Eigen::Matrix<double, -1, 1>::Zero(n_params_main);
      generate_random_std_norm_vec_InPlace(std_norm_vec_main, rng_main);
      if (metric_shape_main == "dense") result_input.main_velocity_0_vec() = EHMC_Metric_struct_as_cpp_struct.M_inv_dense_main_chol * std_norm_vec_main;
      if (metric_shape_main == "diag")  result_input.main_velocity_0_vec().array() = std_norm_vec_main.array() * EHMC_Metric_struct_as_cpp_struct.M_inv_main_vec.array().sqrt();
    }
    
    // ---- Draw velocity for nuisance: v ~ N(0, M_us⁻¹) ----
    {
      Eigen::Matrix<double, -1, 1> std_norm_vec_us = Eigen::Matrix<double, -1, 1>::Zero(n_nuisance);
      generate_random_std_norm_vec_InPlace(std_norm_vec_us, rng_nuisance);
      result_input.us_velocity_0_vec().array() = std_norm_vec_us.array() * EHMC_Metric_struct_as_cpp_struct.M_inv_us_vec.array().sqrt();
    }
    
    {
      {
        try {
          
          result_input.main_velocity_vec_proposed() = result_input.main_velocity_0_vec();
          result_input.main_theta_vec_proposed() = result_input.main_theta_vec_0();
          result_input.us_velocity_vec_proposed() = result_input.us_velocity_0_vec();
          result_input.us_theta_vec_proposed() = result_input.us_theta_vec_0();
          
          // ---- Compute L ----
          if (EHMC_args_as_cpp_struct.tau_main < EHMC_args_as_cpp_struct.eps_main) {
            EHMC_args_as_cpp_struct.tau_main = EHMC_args_as_cpp_struct.eps_main;
          }
          //// burn-in with share_tau_ii_across_chains: tau_main_ii was set (identically for every chain) by the burn-in worker
          if (!EHMC_args_as_cpp_struct.randomize_tau) {
            EHMC_args_as_cpp_struct.tau_main_ii = EHMC_args_as_cpp_struct.tau_main;
          } else if (!EHMC_args_as_cpp_struct.use_given_tau_main_ii) {
            EHMC_args_as_cpp_struct.tau_main_ii = fn_generate_burnin_tau_ii(EHMC_args_as_cpp_struct.tau_main,
                                                                            rng_main,
                                                                            true);
          }
          if (EHMC_args_as_cpp_struct.tau_main_ii < EHMC_args_as_cpp_struct.eps_main) {
            EHMC_args_as_cpp_struct.tau_main_ii = EHMC_args_as_cpp_struct.eps_main;
          }
          const int L_ii = fn_number_of_leapfrog_steps_from_tau_eps_and_max_L(
              EHMC_args_as_cpp_struct.tau_main_ii,
              EHMC_args_as_cpp_struct.eps_main,
              EHMC_args_as_cpp_struct.max_L);
          EHMC_args_as_cpp_struct.L_main_ii = L_ii;  //// recorded per iteration with debug = TRUE
          
          // ---- Initial lp: BOTH orderings reuse the current-state evaluation within a caller-owned
          // sequence of trajectories for the same target. It is valid for KFK exactly as for FKF: the
          // last evaluation of a KFK trajectory is at the proposal (kept on accept), a reject/divergence
          // restores lp_and_grad_outs_0 (the current state), and an exception clears the flag. KFK used
          // to recompute it every iteration - one uncounted gradient per iteration (1 / (L + 1) of the
          // sampling time) giving bit-identical values. Burn-in passes a fresh (false) flag per call. ----
          const bool reuse_initial_lp =
              current_lp_grad_valid != nullptr && *current_lp_grad_valid;
          if (!reuse_initial_lp) {
              if (current_lp_grad_valid != nullptr) *current_lp_grad_valid = false;
              fn_lp_grad_InPlace(result_input.lp_and_grad_outs(),
                                 Model_type,
                                 force_autodiff, force_PartialLog, multi_attempts,
                                 result_input.main_theta_vec(), result_input.us_theta_vec(),
                                 y_ref,
                                 grad_option,
                                 Model_args_as_cpp_struct,
                                 Stan_model_as_cpp_struct,
                                 LC_MVP_ws_structs,
                                 n_threads_WCP);
              if (EHMC_args_as_cpp_struct.diffusion_HMC_integrator == "flow_kick_flow" &&
                  !std::isfinite(result_input.lp_and_grad_outs()(0))) {

                  throw std::runtime_error("Nonfinite initial density in joint diffusion HMC.");

              }
          }
          
          log_posterior_0 = result_input.lp_and_grad_outs()(0);
          U_x_initial = -log_posterior_0;
          
          result_input.lp_and_grad_outs_0() = result_input.lp_and_grad_outs(); // ----
          // Both rejection methods restore this evaluation; acceptance retains
          // the freshly evaluated endpoint. Metric changes do not change lp.
          if (current_lp_grad_valid != nullptr) *current_lp_grad_valid = true;
          
          // ---- Leapfrog (w/ diffusion-pathspace HMC for nuisance) ----
          // Eigen::Matrix<double, -1, 1>  us_theta_vec_current_segment =     result_input.us_theta_vec();
          // Eigen::Matrix<double, -1, 1>  us_velocity_vec_current_segment =  result_input.us_velocity_vec();
           
          if (EHMC_args_as_cpp_struct.diffusion_HMC_integrator == "flow_kick_flow") {
              
              if (metric_shape_main == "dense") {
                  
                  if (EHMC_Metric_struct_as_cpp_struct.theta_hat_us_vec.isZero(1e-5) && EHMC_Metric_struct_as_cpp_struct.M_us_vec.isOnes(1e-5)) {
                    
                       leapfrog_integrator_dense_M_diffusion_HMC_dual_Id_M_us_zero_theta_us_flow_kick_flow_InPlace(
                                                                                                       result_input.main_velocity_vec_proposed(),
                                                                                                       result_input.us_velocity_vec_proposed(),
                                                                                                       result_input.main_theta_vec_proposed(),
                                                                                                       result_input.us_theta_vec_proposed(),
                                                                                                       result_input.lp_and_grad_outs(),
                                                                                                       result_input.us_theta_vec_current_segment(),
                                                                                                       result_input.us_velocity_vec_current_segment(),
                                                                                                       result_input.main_theta_vec(),
                                                                                                       result_input.us_theta_vec(),
                                                                                                       EHMC_Metric_struct_as_cpp_struct.M_inv_dense_main,
                                                                                                       EHMC_Metric_struct_as_cpp_struct.M_us_vec,
                                                                                                       y_ref,
                                                                                                       L_ii, EHMC_args_as_cpp_struct.eps_main,
                                                                                                       Model_type,
                                                                                                       force_autodiff, force_PartialLog, multi_attempts,
                                                                                                       grad_option,
                                                                                                       Model_args_as_cpp_struct,
                                                                                                       Stan_model_as_cpp_struct, 
                                                                                                       LC_MVP_ws_structs,
                                                                                                       n_threads_WCP,
                                                                                                       fn_lp_grad_InPlace);
                    
                  } else {
                    
                       leapfrog_integrator_dense_M_diffusion_HMC_dual_flow_kick_flow_InPlace(
                                                                                                       result_input.main_velocity_vec_proposed(),
                                                                                                       result_input.us_velocity_vec_proposed(),
                                                                                                       result_input.main_theta_vec_proposed(),
                                                                                                       result_input.us_theta_vec_proposed(),
                                                                                                       result_input.lp_and_grad_outs(),
                                                                                                       result_input.us_theta_vec_current_segment(),
                                                                                                       result_input.us_velocity_vec_current_segment(),
                                                                                                       result_input.main_theta_vec(),
                                                                                                       result_input.us_theta_vec(),
                                                                                                       EHMC_Metric_struct_as_cpp_struct.M_inv_dense_main,
                                                                                                       EHMC_Metric_struct_as_cpp_struct.M_inv_us_vec,
                                                                                                       EHMC_Metric_struct_as_cpp_struct.M_us_vec,
                                                                                                       EHMC_Metric_struct_as_cpp_struct.theta_hat_us_vec,
                                                                                                       y_ref,
                                                                                                       L_ii, EHMC_args_as_cpp_struct.eps_main,
                                                                                                       Model_type,
                                                                                                       force_autodiff, force_PartialLog, multi_attempts,
                                                                                                       grad_option,
                                                                                                       Model_args_as_cpp_struct,
                                                                                                       Stan_model_as_cpp_struct, 
                                                                                                       LC_MVP_ws_structs,
                                                                                                       n_threads_WCP,
                                                                                                       fn_lp_grad_InPlace);
                    
                  }
                
              } else if (metric_shape_main == "diag") {
                  
                  if (EHMC_Metric_struct_as_cpp_struct.theta_hat_us_vec.isZero(1e-5) && EHMC_Metric_struct_as_cpp_struct.M_us_vec.isOnes(1e-5)) {
                    
                       leapfrog_integrator_diag_M_diffusion_HMC_dual_Id_M_us_zero_theta_us_flow_kick_flow_InPlace(
                                                                                                       result_input.main_velocity_vec_proposed(),
                                                                                                       result_input.us_velocity_vec_proposed(),
                                                                                                       result_input.main_theta_vec_proposed(),
                                                                                                       result_input.us_theta_vec_proposed(),
                                                                                                       result_input.lp_and_grad_outs(),
                                                                                                       result_input.us_theta_vec_current_segment(),
                                                                                                       result_input.us_velocity_vec_current_segment(),
                                                                                                       result_input.main_theta_vec(),
                                                                                                       result_input.us_theta_vec(),
                                                                                                       EHMC_Metric_struct_as_cpp_struct.M_inv_main_vec,
                                                                                                       EHMC_Metric_struct_as_cpp_struct.M_us_vec,
                                                                                                       y_ref,
                                                                                                       L_ii, EHMC_args_as_cpp_struct.eps_main,
                                                                                                       Model_type,
                                                                                                       force_autodiff, force_PartialLog, multi_attempts,
                                                                                                       grad_option,
                                                                                                       Model_args_as_cpp_struct,
                                                                                                       Stan_model_as_cpp_struct, 
                                                                                                       LC_MVP_ws_structs,
                                                                                                       n_threads_WCP,
                                                                                                       fn_lp_grad_InPlace);
                    
                  } else {
                    
                       leapfrog_integrator_diag_M_diffusion_HMC_dual_flow_kick_flow_InPlace(
                                                                                                       result_input.main_velocity_vec_proposed(),
                                                                                                       result_input.us_velocity_vec_proposed(),
                                                                                                       result_input.main_theta_vec_proposed(),
                                                                                                       result_input.us_theta_vec_proposed(),
                                                                                                       result_input.lp_and_grad_outs(),
                                                                                                       result_input.us_theta_vec_current_segment(),
                                                                                                       result_input.us_velocity_vec_current_segment(),
                                                                                                       result_input.main_theta_vec(),
                                                                                                       result_input.us_theta_vec(),
                                                                                                       EHMC_Metric_struct_as_cpp_struct.M_inv_main_vec,
                                                                                                       EHMC_Metric_struct_as_cpp_struct.M_inv_us_vec,
                                                                                                       EHMC_Metric_struct_as_cpp_struct.M_us_vec,
                                                                                                       EHMC_Metric_struct_as_cpp_struct.theta_hat_us_vec,
                                                                                                       y_ref,
                                                                                                       L_ii, EHMC_args_as_cpp_struct.eps_main,
                                                                                                       Model_type,
                                                                                                       force_autodiff, force_PartialLog, multi_attempts,
                                                                                                       grad_option,
                                                                                                       Model_args_as_cpp_struct,
                                                                                                       Stan_model_as_cpp_struct, 
                                                                                                       LC_MVP_ws_structs,
                                                                                                       n_threads_WCP,
                                                                                                       fn_lp_grad_InPlace);
                    
                  }
                
              }
              
          } else if (metric_shape_main == "dense") {
                  
                  if (EHMC_Metric_struct_as_cpp_struct.theta_hat_us_vec.isZero(1e-5) && EHMC_Metric_struct_as_cpp_struct.M_us_vec.isOnes(1e-5)) {
                    
                       leapfrog_integrator_dense_M_diffusion_HMC_dual_Id_M_us_zero_theta_us_InPlace(  result_input.main_velocity_vec_proposed(),
                                                                                                      result_input.us_velocity_vec_proposed(),
                                                                                                      result_input.main_theta_vec_proposed(),
                                                                                                      result_input.us_theta_vec_proposed(),
                                                                                                      result_input.lp_and_grad_outs(),
                                                                                                      result_input.us_theta_vec_current_segment(),
                                                                                                      result_input.us_velocity_vec_current_segment(),
                                                                                                      result_input.main_theta_vec(),
                                                                                                      result_input.us_theta_vec(),
                                                                                                      EHMC_Metric_struct_as_cpp_struct.M_inv_dense_main,
                                                                                                      EHMC_Metric_struct_as_cpp_struct.M_us_vec,
                                                                                                      y_ref,
                                                                                                      L_ii, EHMC_args_as_cpp_struct.eps_main,
                                                                                                      Model_type,
                                                                                                      force_autodiff, force_PartialLog, multi_attempts,
                                                                                                      grad_option,
                                                                                                      Model_args_as_cpp_struct,
                                                                                                      Stan_model_as_cpp_struct, 
                                                                                                      LC_MVP_ws_structs,
                                                                                                      n_threads_WCP,
                                                                                                      fn_lp_grad_InPlace);
                    
                  } else { 
                  
                       leapfrog_integrator_dense_M_diffusion_HMC_dual_InPlace(  result_input.main_velocity_vec_proposed(),
                                                                                result_input.us_velocity_vec_proposed(),
                                                                                result_input.main_theta_vec_proposed(),
                                                                                result_input.us_theta_vec_proposed(),
                                                                                result_input.lp_and_grad_outs(),
                                                                                result_input.us_theta_vec_current_segment(),
                                                                                result_input.us_velocity_vec_current_segment(),
                                                                                result_input.main_theta_vec(),
                                                                                result_input.us_theta_vec(),
                                                                                EHMC_Metric_struct_as_cpp_struct.M_inv_dense_main,
                                                                                EHMC_Metric_struct_as_cpp_struct.M_inv_us_vec,
                                                                                EHMC_Metric_struct_as_cpp_struct.M_us_vec,
                                                                                EHMC_Metric_struct_as_cpp_struct.theta_hat_us_vec,
                                                                                y_ref,
                                                                                L_ii, EHMC_args_as_cpp_struct.eps_main,
                                                                                Model_type,
                                                                                force_autodiff, force_PartialLog, multi_attempts,
                                                                                grad_option,
                                                                                Model_args_as_cpp_struct,
                                                                                Stan_model_as_cpp_struct, 
                                                                                LC_MVP_ws_structs,
                                                                                n_threads_WCP,
                                                                                fn_lp_grad_InPlace);
                  }
            
          } else if (metric_shape_main == "diag") {
            
                  if (EHMC_Metric_struct_as_cpp_struct.theta_hat_us_vec.isZero(1e-5) && EHMC_Metric_struct_as_cpp_struct.M_us_vec.isOnes(1e-5)) {
                  
                        leapfrog_integrator_diag_M_diffusion_HMC_dual_Id_M_us_zero_theta_us_InPlace(  result_input.main_velocity_vec_proposed(),
                                                                                                      result_input.us_velocity_vec_proposed(),
                                                                                                      result_input.main_theta_vec_proposed(),
                                                                                                      result_input.us_theta_vec_proposed(),
                                                                                                      result_input.lp_and_grad_outs(),
                                                                                                      result_input.us_theta_vec_current_segment(),
                                                                                                      result_input.us_velocity_vec_current_segment(),
                                                                                                      result_input.main_theta_vec(),
                                                                                                      result_input.us_theta_vec(),
                                                                                                      EHMC_Metric_struct_as_cpp_struct.M_inv_main_vec,
                                                                                                      EHMC_Metric_struct_as_cpp_struct.M_us_vec,
                                                                                                      y_ref,
                                                                                                      L_ii, EHMC_args_as_cpp_struct.eps_main,
                                                                                                      Model_type,
                                                                                                      force_autodiff, force_PartialLog, multi_attempts,
                                                                                                      grad_option,
                                                                                                      Model_args_as_cpp_struct,
                                                                                                      Stan_model_as_cpp_struct, 
                                                                                                      LC_MVP_ws_structs,
                                                                                                      n_threads_WCP,
                                                                                                      fn_lp_grad_InPlace);
                    
                  } else { 
                  
                        leapfrog_integrator_diag_M_diffusion_HMC_dual_InPlace(  result_input.main_velocity_vec_proposed(),
                                                                                result_input.us_velocity_vec_proposed(),
                                                                                result_input.main_theta_vec_proposed(),
                                                                                result_input.us_theta_vec_proposed(),
                                                                                result_input.lp_and_grad_outs(),
                                                                                result_input.us_theta_vec_current_segment(),
                                                                                result_input.us_velocity_vec_current_segment(),
                                                                                result_input.main_theta_vec(),
                                                                                result_input.us_theta_vec(),
                                                                                EHMC_Metric_struct_as_cpp_struct.M_inv_main_vec,
                                                                                EHMC_Metric_struct_as_cpp_struct.M_inv_us_vec,
                                                                                EHMC_Metric_struct_as_cpp_struct.M_us_vec,
                                                                                EHMC_Metric_struct_as_cpp_struct.theta_hat_us_vec,
                                                                                y_ref,
                                                                                L_ii, EHMC_args_as_cpp_struct.eps_main,
                                                                                Model_type,
                                                                                force_autodiff, force_PartialLog, multi_attempts,
                                                                                grad_option,
                                                                                Model_args_as_cpp_struct,
                                                                                Stan_model_as_cpp_struct, 
                                                                                LC_MVP_ws_structs,
                                                                                n_threads_WCP,
                                                                                fn_lp_grad_InPlace);
                    
                  }
            
          }
          
          // ---- Proposed lp ----
          log_posterior_prop = result_input.lp_and_grad_outs()(0);
          if (EHMC_args_as_cpp_struct.record_kinetic_energy_tau_derivatives) result_input.record_kinetic_energy_rate_main();
          U_x_prop = -log_posterior_prop;
          
          // ============================================================
          // M-H ENERGY CALCULATION
          // H = -log_post + ½v_main'M_main v_main + ½v_us'M_us v_us
          // ============================================================
          if (metric_shape_main == "dense") {
                  
                  const Eigen::Matrix<double, 1, -1> v0_x_M = result_input.main_velocity_0_vec().transpose() * EHMC_Metric_struct_as_cpp_struct.M_dense_main;
                  energy_old = U_x_initial;
                  energy_old += 0.5 * (v0_x_M * result_input.main_velocity_0_vec()).eval()(0, 0);
                  //// for nuisance:
                  energy_old += 0.5 * (result_input.us_velocity_0_vec().array().square() * EHMC_Metric_struct_as_cpp_struct.M_us_vec.array()).sum();
                  
                  const Eigen::Matrix<double, 1, -1> vp_x_M = result_input.main_velocity_vec_proposed().transpose() * EHMC_Metric_struct_as_cpp_struct.M_dense_main;
                  energy_new = U_x_prop;
                  energy_new += 0.5 * (vp_x_M * result_input.main_velocity_vec_proposed()).eval()(0, 0);
                  //// for nuisance:
                  energy_new += 0.5 * (result_input.us_velocity_vec_proposed().array().square() * EHMC_Metric_struct_as_cpp_struct.M_us_vec.array()).sum();
                  
                  log_ratio = -energy_new + energy_old;
            
          } else if (metric_shape_main == "diag") {
                  
                  energy_old = U_x_initial;
                  // Main: ½v'M_main v (unchanged):
                  energy_old += 0.5 * (result_input.main_velocity_0_vec().array().square() * stan::math::inv(EHMC_Metric_struct_as_cpp_struct.M_inv_main_vec).array()).sum();
                  //// for nuisance:
                  energy_old += 0.5 * (result_input.us_velocity_0_vec().array().square() * EHMC_Metric_struct_as_cpp_struct.M_us_vec.array()).sum();
                  
                  energy_new = U_x_prop;
                  // Main: ½v'M_main v (unchanged):
                  energy_new += 0.5 * (result_input.main_velocity_vec_proposed().array().square() * stan::math::inv(EHMC_Metric_struct_as_cpp_struct.M_inv_main_vec).array()).sum();
                  //// for nuisance:
                  energy_new += 0.5 * (result_input.us_velocity_vec_proposed().array().square() * EHMC_Metric_struct_as_cpp_struct.M_us_vec.array()).sum();
                  
                  log_ratio = -energy_new + energy_old;
            
          }
          
          // ---- Accept/Reject ----
          if (check_divergence(log_ratio, result_input) == true) {
                  
                  result_input.main_div() = 1;
                  result_input.main_p_jump() = 0.0;
                  result_input.us_div() = 1;
                  result_input.us_p_jump() = 0.0;
                  result_input.reject_proposal_main();
                  result_input.reject_proposal_us();
            
          } else {
            
                  result_input.main_div() = 0;
                  result_input.main_p_jump() = std::min(1.0, stan::math::exp(log_ratio));
                  result_input.us_div() = 0;
                  result_input.us_p_jump() = std::min(1.0, stan::math::exp(log_ratio));
                  
                  const double rand_unif = generate_random_std_uniform(rng_main);
                  const double log_rand_unif = stan::math::log(rand_unif);
                  
                  if (log_rand_unif > log_ratio) {
                    result_input.reject_proposal_main();
                    result_input.reject_proposal_us();
                  } else {
                    result_input.accept_proposal_main();
                    result_input.accept_proposal_us();
                  }
            
          }
          
        } catch (...) {
              //// (a diagnostic: the message of the first failed gradient of this chain, written to stderr; the
              ////  failure is recorded below as before; message_of_the_first_failed_gradient_of_each_chain.hpp)
              try {
                    throw;
              } catch (const std::exception &failed_gradient) {
                    fn_write_the_message_of_the_first_failed_gradient_of_this_chain(
                        "main and nuisance parameters, diffusion HMC",
                        failed_gradient.what());
              } catch (...) {
                    fn_write_the_message_of_the_first_failed_gradient_of_this_chain(
                        "main and nuisance parameters, diffusion HMC",
                        "not a std::exception");
              }
          
                if (current_lp_grad_valid != nullptr) *current_lp_grad_valid = false;
                result_input.us_div() = 1;
                result_input.us_p_jump() = 0.0;
                result_input.reject_proposal_us();
                result_input.main_div() = 1;
                result_input.main_p_jump() = 0.0;
                result_input.reject_proposal_main();
          
        }
      }
    }
  }
  
}






















