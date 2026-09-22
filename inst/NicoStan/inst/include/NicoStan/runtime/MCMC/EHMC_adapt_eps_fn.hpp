


#pragma once

 

#include <Eigen/Dense>
 

 

 
using namespace Eigen;

 
 

 
//// adapt step size (eps) using ADAM
Eigen::Matrix<double, -1, 1>        adapt_eps_ADAM(double eps,   //// updating this  
                                                   double eps_m_adam,   //// updating this 
                                                   double eps_v_adam,  //// updating this 
                                                   const double iter, 
                                                   const double n_burnin, 
                                                   const double LR,  /// ADAM learning rate
                                                   const double p_jump, 
                                                   const double adapt_delta, 
                                                   const double beta1_adam, 
                                                   const double beta2_adam, 
                                                   const double eps_adam) {
 
          const double eps_initial = eps;
          const double eps_m_adam_initial = eps_m_adam;
          const double eps_v_adam_initial = eps_v_adam;
          
          const double eps_noisy_grad_across_chains = p_jump - adapt_delta;
          
          // update moving avg's for ADAM
          const double eps_m_adam_new = beta1_adam * eps_m_adam + (1.0 - beta1_adam) * eps_noisy_grad_across_chains;  
          const double eps_v_adam_new = beta2_adam * eps_v_adam + (1.0 - beta2_adam) * std::pow(eps_noisy_grad_across_chains, 2.0);   
          
          // calc. bias-corrected estimates (local to this fn)
          // const double iter_dbl = (double) iter;
          // const double n_burnin_dbl = (double) n_burnin;
          const double eps_m_hat = eps_m_adam_new / (1.0 - std::pow(beta1_adam, iter));
          const double eps_v_hat = eps_v_adam_new / (1.0 - std::pow(beta2_adam, iter));
          
          const double current_alpha = LR * (  1.0 - (1.0 - LR)*iter/(n_burnin) );
          
          const double  log_h =   stan::math::log(eps) + current_alpha * eps_m_hat/(stan::math::sqrt(eps_v_hat) + eps_adam);
          const double  eps_new = stan::math::exp(log_h);   
  
          Eigen::Matrix<double, -1, 1> out_vec(3);
          out_vec(0) = eps_new;
          out_vec(1) = eps_m_adam_new;
          out_vec(2) = eps_v_adam_new;
          
          if (is_NaN_or_Inf_Eigen(out_vec) == true) {
            out_vec(0) = eps_initial;
            out_vec(1) = eps_m_adam_initial;
            out_vec(2) = eps_v_adam_initial;
          }
          
          return out_vec;
 
 
}











