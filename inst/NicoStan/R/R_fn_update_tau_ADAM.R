




#' R_fn_update_tau_using_ADAM
#' @export
R_fn_update_tau_using_ADAM <- function(debug = FALSE,
                                          n_chains, 
                                          noisy_grads_prop_per_chain,
                                          valid_chains,
                                          ##
                                          tau,
                                          LR,
                                          ii,
                                          n_burnin,
                                          ##
                                          tau_m_adam,
                                          tau_v_adam,
                                          ##
                                          beta1_adam,
                                          beta2_adam,
                                          eps_adam,
                                          ##
                                          ## ---- ADAM bias-correction step (OPTIONAL):
                                          ##
                                          ## The number of tau ADAM updates actually PERFORMED on these moments,
                                          ## counting this one (1 on the first update after the moments are reset).
                                          ## ii keeps driving the learning-rate SCHEDULE below; it cannot drive the
                                          ## bias correction, because ii also advances on iterations where this
                                          ## function returns early without touching tau_m_adam / tau_v_adam. After
                                          ## k such skips the first real step would divide by (1 - beta2^(k + 1))
                                          ## instead of (1 - beta2), inflating it by up to sqrt(1 / (1 - beta2)).
                                          ## NULL keeps the historical behaviour (bias correction uses ii).
                                          ##
                                          bias_correction_step = NULL,
                                          ##
                                          ## ---- Acceptance weighting (both OPTIONAL; corrected sampler calls use a mean).
                                          ##
                                          ## weights_per_chain: per-chain Metropolis acceptance probabilities (p_jump),
                                          ## indexed the SAME way as noisy_grads_prop_per_chain. The ChEES estimator is the
                                          ## expectation estimate mean(alpha_k g_k). Passing p_jump instead of
                                          ## relying on the accept indicator carried implicitly by use_proposed = FALSE gives the
                                          ## same expectation with strictly lower variance, since it replaces a Bernoulli draw by
                                          ## its conditional mean (Rao-Blackwellisation).
                                          weights_per_chain = NULL,
                                          ##
                                          ## aggregation: "median" is the historical behaviour - robust, but it discards the
                                          ## acceptance weighting and is NOT the ChEES estimator. "weighted_mean" uses the
                                          ## weights above (falling back to an unweighted mean if none are supplied).
                                          aggregation = c("median", "weighted_mean")
) {

        aggregation <- match.arg(aggregation)
        ##
        ## ---- bias-correction exponent (defaults to ii, the historical behaviour):
        ##
        bias_correction_step <- if_null_then_set_to(bias_correction_step, ii)
        if (length(bias_correction_step) != 1 || !is.finite(bias_correction_step) || bias_correction_step < 1) {
            stop(paste0("R_fn_update_tau_using_ADAM: bias_correction_step must be a single finite number >= 1; got: ",
                        paste(bias_correction_step, collapse = ", ")))
        }
        ##
        ## ---- every return carries attr "adam_update_performed" so the caller can count real updates:
        ##
        fn_mark_adam_update_performed <- function(returned_vector,
                                                  adam_update_performed) {
            attr(returned_vector, "adam_update_performed") <- adam_update_performed
            return(returned_vector)
        }

        # eta_w = 3.0;
        # rho = 1.0;
        ##
        tau_initial = tau;
        tau_m_adam_initial = tau_m_adam;
        tau_v_adam_initial = tau_v_adam;
        ##
        out_vec <- c(NA, NA, NA)
        ##
        is_NaN_or_Inf_vec <- function(x) { 
          if ((any(is.infinite(x))) || (any(is.nan(x))) || (any(is.na(x)))) {  # Added is.na check
            return(TRUE)
          } else { 
            return(FALSE)
          }
        }
        
        noisy_grads_per_chain <- noisy_grads_prop_per_chain
        n_chains_total <- if (length(n_chains) == 1L && is.finite(n_chains) && n_chains >= 1) {
            as.integer(n_chains)
        } else {
            length(noisy_grads_per_chain)
        }
        # print(noisy_grads_per_chain)
        
        # Filter out NA/NaN/Inf values before computing mean
        valid_mask <- (!is.na(noisy_grads_per_chain) &
                         !is.nan(noisy_grads_per_chain) &
                         !is.infinite(noisy_grads_per_chain))
        valid_grads <- noisy_grads_per_chain[valid_mask]

        if (length(valid_grads) == 0) {

            return(fn_mark_adam_update_performed(returned_vector       = c(tau, tau_m_adam, tau_v_adam),
                                                 adam_update_performed = FALSE))

        } else if (aggregation == "median") {

            # tau_noisy_grad_mean <- mean(valid_grads)
            tau_noisy_grad_mean <- median(valid_grads)

        } else {

            ## ---- acceptance-weighted expectation: mean(alpha_k * g_k).
            ##
            ## Zero-acceptance chains contribute zero, remaining in the denominator. If every
            ## surviving chain has alpha = 0, retain tau AND its moments rather than drifting on old momentum.
            ##
            if (is.null(weights_per_chain)) {

                ## The objective is an expectation over the complete chain
                ## minibatch. Divergent/non-finite chains contribute zero and
                ## stay in the denominator; mean(valid_grads) would silently
                ## condition on the surviving chains.
                tau_noisy_grad_mean <- sum(valid_grads) / n_chains_total

            } else {

                ## noisy_grads_prop_per_chain is filled in only for the non-divergent chains, so it
                ## can be SHORTER than weights_per_chain whenever the last chain diverged. Indexing a
                ## longer weight vector by the shorter mask would silently recycle the mask and pair
                ## each gradient with the wrong chain's acceptance probability, so align explicitly:
                weights_aligned <- rep(0.0, length(noisy_grads_per_chain))
                n_shared <- min(length(weights_per_chain), length(noisy_grads_per_chain))
                if (n_shared > 0) weights_aligned[seq_len(n_shared)] <- weights_per_chain[seq_len(n_shared)]
                ##
                valid_weights <- weights_aligned[valid_mask]
                valid_weights[!is.finite(valid_weights)] <- 0.0
                valid_weights <- pmax(0.0, valid_weights)
                ##
                weight_sum <- sum(valid_weights)
                ##
                if (weight_sum > 0) {
                    ## weights_per_chain are conditional acceptance
                    ## probabilities. The expectation is mean(alpha_k g_k),
                    ## hence divide by the full chain count rather than by
                    ## the number of finite survivors.
                    tau_noisy_grad_mean <- sum(valid_weights * valid_grads) / n_chains_total
                } else {
                    return(fn_mark_adam_update_performed(returned_vector       = c(tau, tau_m_adam, tau_v_adam),
                                                         adam_update_performed = FALSE))
                }

            }

        }
        
        # Ensure tau_noisy_grad_mean is valid
        if (is_NaN_or_Inf_vec(tau_noisy_grad_mean)) { 
          tau_noisy_grad_mean <- 0.0
        }
        
        ##
        # if (ii > eta_w && eigen_max > 0) {
        ##
        pow <- function(x, a) { 
          return(x^a)
        }
        ##
        tau_m_adam = beta1_adam * tau_m_adam + (1.0 - beta1_adam) * tau_noisy_grad_mean;
        tau_v_adam = beta2_adam * tau_v_adam + (1.0 - beta2_adam) * pow(tau_noisy_grad_mean, 2);
        
        # Add small epsilon to denominators to prevent division by zero
        ## Bias correction counts the updates actually applied to these moments (bias_correction_step),
        ## not the iteration index ii (see the bias_correction_step argument).
        tau_m_hat = tau_m_adam / (1.0 - pow(beta1_adam, bias_correction_step) + 1e-10);
        tau_v_hat = tau_v_adam / (1.0 - pow(beta2_adam, bias_correction_step) + 1e-10);
        
        ##
        ## ii and n_burnin here count TAU-adaptation iterations, excluding the preceding ramp.
        ## Start at the full requested LR, then decay over this optimiser's own active window.
        adaptation_progress <- if (n_burnin <= 1) 0 else min(1, max(0, (ii - 1) / (n_burnin - 1)))
        current_alpha = LR * (1.0 - (1.0 - LR) * adaptation_progress);
        
        # CRITICAL FIX: Work in log space more carefully
        # Ensure tau is positive before taking log
        if (tau <= 0 || is.na(tau) || is.nan(tau) || is.infinite(tau)) {
          tau = tau_initial  # Reset to initial if invalid
        }
        
        # Update in log space with bounds
        log_tau_current = log(tau)
        grad_update = current_alpha * tau_m_hat / (sqrt(abs(tau_v_hat)) + eps_adam)
        
        # # Clip the gradient update to prevent extreme changes
        # grad_update = max(-1.0, min(1.0, grad_update))
        # 
        log_tau = log_tau_current + grad_update
        
        ## Numerical bounds only; the caller applies the user's max_tau limit.
        ## An arbitrary 0.05 floor previously overrode smaller valid initial trajectory lengths.
        log_tau = max(log(.Machine$double.xmin), min(log(.Machine$double.xmax) - 1, log_tau))
        
        tau = exp(log_tau);
        
        ##
        # # ## Additional safety: Limit tau changes to "max_change"% per iteration:
        # max_change <- 1.5
        # tau <- tau_initial * min(max_change, max(1/max_change, tau/tau_initial))
        ##
        out_vec[1] <- tau
        out_vec[2] <- tau_m_adam
        out_vec[3] <- tau_v_adam
        # }
        ##
        # Right before the return
        if (debug) {
          if (ii %% 50 == 0) {
            cat("\n=== ADAM Tau Update ===\n")
            cat("tau_noisy_grad_mean:", tau_noisy_grad_mean, "\n")
            cat("tau_m_adam:", tau_m_adam, "\n")
            cat("tau_v_adam:", tau_v_adam, "\n")
            cat("current_alpha:", current_alpha, "\n")
            cat("log_tau old:", log(tau_initial), "\n")
            cat("log_tau new:", log_tau, "\n")
            cat("tau:", tau_initial, "->", tau, "\n")
          }
        }
        
        # Final safety check
        adam_update_performed <- TRUE
        if (is_NaN_or_Inf_vec(out_vec) == TRUE) {
          cat("tau or tau_m_adam or tau_v_adam are not valid, reverting to initial values\n")
          out_vec[1] = tau_initial;
          out_vec[2] = tau_m_adam_initial;
          out_vec[3] = tau_v_adam_initial;
          ## the moments were reverted, so this does not count as a performed update:
          adam_update_performed <- FALSE
        }
        ##
        return(fn_mark_adam_update_performed(returned_vector       = out_vec,
                                             adam_update_performed = adam_update_performed))
  
}











