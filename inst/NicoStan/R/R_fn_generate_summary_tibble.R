


#' generate_summary_tibble
#' @keywords internal
#' @export
generate_summary_tibble <- function(n_threads = NULL,
                                    trace, 
                                    param_names, 
                                    n_to_compute, 
                                    compute_nested_rhat,
                                    n_chains, 
                                    n_superchains,
                                    nested_rhat_grouping = NULL) {
  
        
              ##
              ## ---- Fix: the n_threads argument used to be overwritten here with detectCores() / 2 (96 on the
              ##      local HPC), so the caller's thread count was silently ignored. Honour it; only when it is NULL use
              ##      half of the CPUs this process may run on (taskset / CPU affinity respected).
              ##
              if (is.null(n_threads)) {
                    n_cpus_available_to_this_process <-  length(tryCatch(parallel::mcaffinity(), error = function(error_object) NULL))
                    if (n_cpus_available_to_this_process < 1) n_cpus_available_to_this_process <-  parallel::detectCores()
                    n_threads <-  max(1, round(n_cpus_available_to_this_process / 2, 0))
              }
              if (!(is.numeric(n_threads) && length(n_threads) == 1 && is.finite(n_threads) && n_threads >= 1)) {
                    stop(paste0("generate_summary_tibble: n_threads must be a single number >= 1, got: ",
                                paste(format(n_threads), collapse = ", ")))
              }

              #### Initialize summary dataframe
              summary_df <- data.frame(     parameter = param_names,
                                            mean = NA,
                                            sd = NA,
                                            `2.5%` = NA,
                                            `50%` = NA,
                                            `97.5%` = NA,
                                            n_eff = NA,
                                            Rhat = NA,
                                            n_Rhat = NA,
                                            n_eff_tail = NA,   ## tail ESS (Vehtari et al. 2021): the smaller of the 5% and 95% quantile ESS
                                            n_eff_sd = NA,     ## ESS of the centred squared draws, (x - mean)^2: the ESS for estimating the posterior SD
                                                               ## ("ESS(theta^2)"; the definition behind posterior::ess_sd; Enzo, 4 Oct 2026)
                                            check.names = FALSE)
              
              # # Effective Sample Size (ESS) and Rhat - using the fast custom RcppParallel fn "NicoStan::Rcpp_compute_MCMC_diagnostics()"
              posterior_draws_as_std_vec_of_mats <- list()
              ##
              n_iter <- dim(trace)[2]
              mat <- matrix(nrow = n_iter, ncol = n_chains)

              n_params <- n_to_compute
              
             #  message(print(paste("str(trace) = ")))
             #  message(print(paste(str(trace))))
             # # message(print(str(posterior_draws_as_std_vec_of_mats)))

              for (i in 1:n_params) {
                posterior_draws_as_std_vec_of_mats[[i]] <- mat
                for (kk in 1:n_chains) {
                  posterior_draws_as_std_vec_of_mats[[i]][1:n_iter, kk] <- trace[i, 1:n_iter, kk] 
                }
              }
              
              if (n_params < n_threads) { n_threads = n_params }
              
              #### Compute summary stats using custom Rcpp/C++ functions:
              outs <-  (Rcpp_compute_chain_stats(   posterior_draws_as_std_vec_of_mats,
                                                    stat_type = "mean",
                                                    n_threads = n_threads))
              means_between_chains <- outs$statistics[, 1]
              

              outs <-  (Rcpp_compute_chain_stats(   posterior_draws_as_std_vec_of_mats,
                                                    stat_type = "sd",
                                                    n_threads = n_threads))
              SDs_between_chains <- outs$statistics[, 1]
              

              outs <-  (Rcpp_compute_chain_stats(   posterior_draws_as_std_vec_of_mats,
                                                    stat_type = "quantiles",
                                                    n_threads = n_threads))
              quantiles_between_chains <- outs$statistics
              
              
              #### Compute Effective Sample Size (ESS) and Rhat using custom Rcpp/C++ functions 
              outs <-  (Rcpp_compute_MCMC_diagnostics(  posterior_draws_as_std_vec_of_mats,
                                                        diagnostic = "split_ESS_rank",
                                                        n_threads = n_threads))
              ess_vec <- outs$diagnostics[, 1]
              # ess_tail_vec <- outs$diagnostics[, 2]
              ess_tail_vec <- outs$diagnostics[, 2]   ## the tail ESS "split_ESS_rank" already returns (column 2), now kept
              ## ---- where draws tie at a quantile (e.g. a probability that equals 1 in more than 5% of the
              ##      draws, so that I(x <= q95) never changes), the tail ESS of x is undefined: it is then the
              ##      tail ESS of -x, i.e. the same two tails counted with the tied draws on the other side
              ##      (without ties the two agree to within one draw). Draws that never change stay undefined:
              tail_ESS_undefined <- which(!is.finite(ess_tail_vec))
              if (length(tail_ESS_undefined) > 0) {
                    negated_draws <- lapply(X = posterior_draws_as_std_vec_of_mats[tail_ESS_undefined],
                                            FUN = function(draws_matrix) -draws_matrix)
                    tail_ESS_of_negated_draws <- Rcpp_compute_MCMC_diagnostics( negated_draws,
                                                                                diagnostic = "split_ESS_rank",
                                                                                n_threads = n_threads)
                    ess_tail_vec[tail_ESS_undefined] <- tail_ESS_of_negated_draws$diagnostics[, 2]
              }
              ##
              ## ESS of the centred squared draws, (x - mean(x))^2 with ONE pooled mean over all chains and iterations: the ESS
              ## for estimating the posterior SD (the definition of posterior::ess_sd), computed with the plain split-chain ESS
              ## "split_ESS" of the same C++ routine. Anticorrelated draws inflate the bulk (and, less, the tail) ESS of x while
              ## this one falls, so it is reported for every parameter alongside them (Enzo, 4 Oct 2026).
              posterior_draws_centred_squared <- lapply(X = posterior_draws_as_std_vec_of_mats, FUN = function(draws_matrix) {
                    (draws_matrix - mean(draws_matrix))^2
              })
              outs <-  (Rcpp_compute_MCMC_diagnostics(  posterior_draws_centred_squared,
                                                        diagnostic = "split_ESS",
                                                        n_threads = n_threads))
              ess_sd_vec <- outs$diagnostics[, 1]
              ##
              outs <-  (Rcpp_compute_MCMC_diagnostics(  posterior_draws_as_std_vec_of_mats,
                                                        diagnostic = "split_rhat_rank",
                                                        n_threads = n_threads))
              rhat_bulk_vec <- outs$diagnostics[, 1]
              rhat_tail_vec <- outs$diagnostics[, 2]
              rhat_vec <- pmax(rhat_bulk_vec, rhat_tail_vec)
              Max_rhat_rank <- max(rhat_vec, na.rm = TRUE)
              ##
              for (i in seq_len(n_to_compute)) {
                
                        #### Get all values for this parameter across iterations and chains
                        param_values <- as.vector(trace[i, , ])
                        
                        #### Calculate summary statistics
                        summary_df$mean[i] <- means_between_chains[i]
                        summary_df$sd[i] <- SDs_between_chains[i]
                        try({  
                            summary_df[i, c("2.5%", "50%", "97.5%")] <- quantiles_between_chains[i, ]
                            #### summary_df[i, c("2.5%", "50%", "97.5%")] <- quantiles_between_chains[, i]
                        })
                        summary_df$n_eff[i] <- round(ess_vec[i])
                        summary_df$n_eff_tail[i] <- round(ess_tail_vec[i])
                        summary_df$n_eff_sd[i] <- round(ess_sd_vec[i])
                        summary_df$Rhat[i] <- rhat_vec[i]
                
              }
              
              if (compute_nested_rhat == TRUE) {
                
                        if (is.null(x = nested_rhat_grouping)) {
                            nested_rhat_grouping <-  fn_prepare_nested_rhat_grouping(
                                superchain_ids = create_superchain_ids(n_chains = n_chains, n_superchains = n_superchains),
                                source = "caller_supplied_nominal_groups")
                        }
                        summary_df$n_Rhat[seq_len(length.out = n_to_compute)] <-  fn_nested_rhat_from_draws_array(
                            draws_array = aperm(a = trace[seq_len(length.out = n_to_compute), , , drop = FALSE], perm = c(2, 3, 1)),
                            nested_rhat_grouping = nested_rhat_grouping)
                                    
              }
              
              summary_tibble <- tibble::tibble(summary_df)
              print(summary_tibble, n = 100)
              
              return(summary_tibble)

    
}





 
















