#### =====================================================================================================================================
## R_fns_SNAPER_HMC.R
##
## ---- SNAPER in mandatory metric coordinates -----------------------------------------------------------------------------------------
##
#' Metric-standardised SNAPER trajectory gradient
#' @export
R_fn_compute_gradients_for_tau_using_SNAPER <-  function( eigen_vector,
                                                          theta_vec_initial,
                                                          theta_vec_prop,
                                                          snaper_m_vec,
                                                          velocity_prop,
                                                          mass_matrix,
                                                          tau_ii,
                                                          snaper_m_vec_prop = snaper_m_vec) {

        factor <-  fn_trajectory_metric_factor(mass_matrix, length(theta_vec_initial))
        result <-  fn_metric_position_criterion(
            algorithm = "SNAPER", theta_initial = matrix(theta_vec_initial, ncol = 1),
            theta_proposed = matrix(theta_vec_prop, ncol = 1),
            velocity_proposed = matrix(velocity_prop, ncol = 1),
            mean_initial = snaper_m_vec, mean_proposed = snaper_m_vec_prop,
            metric_factor = factor, tau_values = tau_ii, direction = eigen_vector)
        return(c(result$gradient))

}

fn_update_snaper_w_minibatch <-  function( X,
                                           snaper_m_vec,
                                           snaper_w_vec,
                                           eta_w,
                                           ## metric_factor) {
                                           metric_factor,
                                           ##
                                           ## ---- interest_only (6 Oct 2026): the leading eigen-direction of the covariance of the INTEREST ROWS of
                                           ##      the metric coordinates: the other rows of the states are set to 0 and the direction is 0 on them
                                           ##      (the vector keeps the block's length). NULL = every row, as before (unchanged):
                                           interest_rows = NULL) {

        X <-  as.matrix(X)
        if (nrow(X) != length(snaper_m_vec) || nrow(X) != length(snaper_w_vec)) stop("SNAPER direction dimensions do not agree.")
        X <-  fn_apply_trajectory_metric(metric_factor, X - snaper_m_vec)
        direction <-  c(snaper_w_vec)
        if (!is.null(interest_rows)) {
                X <-  as.matrix(X)
                X[-interest_rows, ] <-  0
                direction[-interest_rows] <-  0
                if (!any(direction != 0)) direction[interest_rows] <-  1 / sqrt(length(interest_rows))
        }
        norm <-  sqrt(sum(direction^2))
        if (!is.finite(norm) || norm == 0) direction <-  fn_initialise_snaper_direction(nrow(X)) else direction <-  direction / norm
        candidate <-  c(X %*% crossprod(X, direction))
        magnitude <-  sqrt(sum(candidate^2))
        if (!is.finite(magnitude)) return(direction)
        if (magnitude == 0) {
                column_norms <-  colSums(X^2)
                if (max(column_norms) == 0) return(direction)
                candidate <-  X[, which.max(column_norms)]
                magnitude <-  sqrt(sum(candidate^2))
        }
        updated <-  direction + eta_w * candidate / magnitude
        norm <-  sqrt(sum(updated^2))
        if (!is.finite(norm) || norm == 0) return(direction)
        return(c(updated) / norm)

}

## ---- Safe first-direction initialisation ------------------------------------
fn_initialise_snaper_direction <- function(n_parameters) {

        if (length(n_parameters) != 1 || !is.finite(n_parameters) || n_parameters < 1 || n_parameters != floor(n_parameters)) {
            stop("n_parameters must be one positive integer.")
        }
        return(rep(1 / sqrt(n_parameters), n_parameters))

}


## ---- Exponentially weighted centre update used by Appendix C ----------------
fn_update_snaper_mean <- function(snaper_mean, theta_mean, ii, kappa = 8.0) {

        if (length(snaper_mean) != length(theta_mean)) stop("SNAPER centre dimensions must agree.")
        if (length(ii) != 1L || !is.finite(ii) || ii < 1) stop("SNAPER iteration must be positive and finite.")
        eta_mean <- 1.0 / (ceiling(ii / kappa) + 1.0)
        if (ii < 2) return(c(theta_mean))
        return((1.0 - eta_mean) * c(snaper_mean) + eta_mean * c(theta_mean))

}


## ---- One trajectory-length algorithm selector -------------------------------
fn_weighted_proposal_mean <-  function( proposals,
                                        acceptance_probabilities,
                                        divergences,
                                        previous_mean) {

        proposals <-  as.matrix(proposals)
        if (length(acceptance_probabilities) != ncol(proposals) ||
            length(divergences) != ncol(proposals)) return(c(previous_mean))
        valid <-  is.finite(acceptance_probabilities) & acceptance_probabilities > 0 &
                  is.finite(divergences) & divergences == 0 &
                  colSums(!is.finite(proposals)) == 0
        if (!any(valid)) return(c(previous_mean))
        weights <-  acceptance_probabilities[valid]
        weights <-  weights / sum(weights)
        return(c(proposals[, valid, drop = FALSE] %*% weights))

}


fn_normalise_burnin_algorithm <- function(burnin_algorithm) {

        if (length(burnin_algorithm) != 1L || is.na(burnin_algorithm)) {
            ## stop("burnin_algorithm must be one of 'KE', 'ChEES', 'CHESSR', 'CHESSR_log' or 'SNAPER'.")
            ## stop("burnin_algorithm must be one of 'KE', 'ChEES', 'CHESSR', 'CHESSR_log', 'SNAPER', 'CHESSR_time' or 'SNAPER_time'.")
            ## stop(paste0("burnin_algorithm must be one of 'ChEES', 'CHESSR', 'CHESSR_log', 'SNAPER', ",
            ##             "'CHESSR_time', 'SNAPER_time', 'ESJD', 'ESJD_CHESSR', 'ESJD_SNAPER' or 'LQ_ESSR'."))
            ## stop(paste0("burnin_algorithm must be one of 'ChEES', 'CHESSR', 'CHESSR_log', 'SNAPER', ",
            ##             "'CHESSR_time', 'SNAPER_time', 'ESJD', 'ESJD_CHESSR', 'ESJD_SNAPER', 'LQ_ESSR', ",
            ##             "'L_ESSR_length_response' or 'LQ_ESSR_length_response'."))
            ## stop(paste0("burnin_algorithm must be one of 'ChEES', 'CHESSR', 'CHESSR_log', 'SNAPER', ",
            ##             "'CHESSR_time', 'SNAPER_time', 'ESJD', 'ESJD_CHESSR', 'ESJD_SNAPER', 'LQ_ESSR', ",
            ##             "'L_ESSR_length_response', 'LQ_ESSR_length_response' or 'LQ_ESSR_spectral'."))
         ## stop(paste0("burnin_algorithm must be one of 'ChEES', 'CHESSR', 'CHESSR_log', 'SNAPER', ",
                     ## "'CHESSR_time', 'SNAPER_time', 'ESJD', 'ESJD_CHESSR', 'ESJD_SNAPER', 'LQ_ESSR', ",
                     ## "'L_ESSR_length_response', 'LQ_ESSR_length_response', 'LQ_ESSR_spectral' or ",
                     ## "'LQ_ESSR_spectral_long_bin_memory'."))
            stop(paste0("burnin_algorithm must be one of 'ChEES', 'CHESSR', 'CHESSR_log', 'SNAPER', ",
                        "'CHESSR_time', 'SNAPER_time', 'ESJD', 'ESJD_CHESSR', 'ESJD_SNAPER', 'LQ_ESSR', ",
                        "'L_ESSR_length_response', 'LQ_ESSR_length_response', 'LQ_ESSR_spectral', ",
                        "'LQ_ESSR_spectral_long_bin_memory', 'LQ_ESSR_spec_bins99_evid_expand' or ",
                        ## "'ESJD_LQ_ESSR_spec_bins99_finite_N_evid_expand'."))
                        "'ESJD_LQ_ESSR_spec_bins99_finite_N_evid_expand' or ",
                        "'ESJD_LQ_ESSR_spec_bins99_evid_expand'."))
        }
        algorithm_key <- tolower(trimws(as.character(burnin_algorithm)))
        ##
        ## ---- CHESS and CHESSR are different criteria: "CHESS"/"ChEES" = ChEES (Hoffman et al., 2021; NOT divided by tau),
        ##      "CHESSR"/"ChEESR" = the ChEES-rate criterion (Sountsov and Hoffman, 2022; divided by tau):
        ##
        algorithm <- switch(algorithm_key,
                           chees = "ChEES",
                           chess = "ChEES",
                           `chees-r` = "CHESSR",
                           chees_r = "CHESSR",
                           chessr = "CHESSR",
                           `chess-r` = "CHESSR",
                           chess_r = "CHESSR",
                           cheesr = "CHESSR",
                           chessr_log = "CHESSR_log",
                           cheesr_log = "CHESSR_log",
                           `chess-r-log` = "CHESSR_log",
                           `chees-r-log` = "CHESSR_log",
                           snaper = "SNAPER",
                           `snaper-hmc` = "SNAPER",
                           snaper_hmc = "SNAPER",
                           ##
                           ## ---- the time-to-target-ESS variants of CHESSR and SNAPER (R_fn_time_criterion.R):
                           ##
                           chessr_time = "CHESSR_time",
                           cheesr_time = "CHESSR_time",
                           `chess-r-time` = "CHESSR_time",
                           `chees-r-time` = "CHESSR_time",
                           snaper_time = "SNAPER_time",
                           `snaper-time` = "SNAPER_time",
                           snaper_hmc_time = "SNAPER_time",
                           ##
                           ## ---- expected squared jumped distance per unit trajectory length (Pasarica and Gelman, 2010; the rate form, as
                           ##      CHESSR is the rate form of ChEES), and the geometric mean of the ESJD and CHESSR criteria
                           ##      (R_fn_metric_trajectory_adaptation.R):
                           ##
                           esjd = "ESJD",
                           esjd_chessr = "ESJD_CHESSR",
                           esjd_cheesr = "ESJD_CHESSR",
                           `esjd-chessr` = "ESJD_CHESSR",
                           `esjd-chees-r` = "ESJD_CHESSR",
                           esjd_snaper = "ESJD_SNAPER",
                           `esjd-snaper` = "ESJD_SNAPER",
                           esjd_snaper_hmc = "ESJD_SNAPER",
                           `esjd-snaper-hmc` = "ESJD_SNAPER",
                           ##
                           ## ---- the soft minimum of the linear and quadratic lag-one ESS bounds per unit trajectory length
                           ##      (R_fn_metric_trajectory_adaptation.R, fn_metric_tau_block_update):
                           ##
                           lq_essr = "LQ_ESSR",
                           lqessr = "LQ_ESSR",
                           `lq-essr` = "LQ_ESSR",
                           ##
                           ## ---- the soft minimum of the linear ("L_") or the linear and quadratic ("LQ_") lag-one ESS
                           ##      bounds per unit trajectory length, maximised over tau from the length response of the
                           ##      jittered trajectories (R_fn_length_response_tau_criterion.R):
                           ##
                           l_essr_length_response = "L_ESSR_length_response",
                           `l-essr-length-response` = "L_ESSR_length_response",
                           lq_essr_length_response = "LQ_ESSR_length_response",
                           `lq-essr-length-response` = "LQ_ESSR_length_response",
                           ##
                           ## ---- the same with a spectral linear term (the flow's fitted cosine mixture per
                           ##      coordinate, composed with the acceptance laziness), per expected gradient
                           ##      (R_fn_spectral_ESS_tau_criterion.R; 6 Oct 2026):
                           ##
                           lq_essr_spectral = "LQ_ESSR_spectral",
                           `lq-essr-spectral` = "LQ_ESSR_spectral",
                           ##
                           ## ---- the same with the length bins forgetting at 0.99 per update instead of 0.95
                           ##      (R_fn_spectral_ESS_tau_criterion.R; 7 Oct 2026):
                           ##
                           lq_essr_spectral_long_bin_memory = "LQ_ESSR_spectral_long_bin_memory",
                           `lq-essr-spectral-long-bin-memory` = "LQ_ESSR_spectral_long_bin_memory",
                           ##
                           ## ---- (7 Oct 2026; R_fn_spectral_ESS_tau_criterion.R) the same with the
                           ##      upper-end doubling only on evidence; and the equal-weight geometric mean of
                           ##      ESJD and that, with the finite-run ESS of the run's n_iter draws:
                           ##
                           lq_essr_spec_bins99_evid_expand = "LQ_ESSR_spec_bins99_evid_expand",
                           esjd_lq_essr_spec_bins99_finite_n_evid_expand =
                                 "ESJD_LQ_ESSR_spec_bins99_finite_N_evid_expand",
                           esjd_lq_essr_spec_bins99_evid_expand = "ESJD_LQ_ESSR_spec_bins99_evid_expand",
                           lq_essr_spec_bins99_evid_expand_jump_accept =
                                 "LQ_ESSR_spec_bins99_evid_expand_jump_accept",
                           esjd_w20_lq_essr_spec_bins99_evid_expand = "ESJD_w20_LQ_ESSR_spec_bins99_evid_expand",
                           esjd_w33_lq_essr_spec_bins99_evid_expand = "ESJD_w33_LQ_ESSR_spec_bins99_evid_expand",
                           ## stop("burnin_algorithm must be one of 'KE', 'ChEES', 'CHESSR', 'CHESSR_log' or 'SNAPER'; got: ", burnin_algorithm))
                           ## stop("burnin_algorithm must be one of 'KE', 'ChEES', 'CHESSR', 'CHESSR_log', 'SNAPER', 'CHESSR_time' or 'SNAPER_time'; got: ", burnin_algorithm))
                           ## stop(paste0("burnin_algorithm must be one of 'KE', 'ChEES', 'CHESSR', 'CHESSR_log', 'SNAPER', 'CHESSR_time', 'SNAPER_time', ",
                           ##             "'ESJD', 'ESJD_CHESSR' or 'ESJD_SNAPER'; got: ", burnin_algorithm)))
                           ## stop(paste0("burnin_algorithm must be one of 'ChEES', 'CHESSR', 'CHESSR_log', ",
                           ##             "'SNAPER', 'CHESSR_time', 'SNAPER_time', 'ESJD', 'ESJD_CHESSR', ",
                           ##             "'ESJD_SNAPER' or 'LQ_ESSR'; got: ", burnin_algorithm)))
                           ## stop(paste0("burnin_algorithm must be one of 'ChEES', 'CHESSR', 'CHESSR_log', ",
                           ##             "'SNAPER', 'CHESSR_time', 'SNAPER_time', 'ESJD', 'ESJD_CHESSR', ",
                           ##             "'ESJD_SNAPER', 'LQ_ESSR', 'L_ESSR_length_response' or ",
                           ##             "'LQ_ESSR_length_response'; got: ", burnin_algorithm)))
                           ## stop(paste0("burnin_algorithm must be one of 'ChEES', 'CHESSR', 'CHESSR_log', ",
                           ##             "'SNAPER', 'CHESSR_time', 'SNAPER_time', 'ESJD', 'ESJD_CHESSR', ",
                           ##             "'ESJD_SNAPER', 'LQ_ESSR', 'L_ESSR_length_response', ",
                           ##             "'LQ_ESSR_length_response' or 'LQ_ESSR_spectral'; got: ",
                           ##             burnin_algorithm)))
                        ## stop(paste0("burnin_algorithm must be one of 'ChEES', 'CHESSR', 'CHESSR_log', ",
                                    ## "'SNAPER', 'CHESSR_time', 'SNAPER_time', 'ESJD', 'ESJD_CHESSR', ",
                                    ## "'ESJD_SNAPER', 'LQ_ESSR', 'L_ESSR_length_response', ",
                                    ## "'LQ_ESSR_length_response', 'LQ_ESSR_spectral' or ",
                                    ## "'LQ_ESSR_spectral_long_bin_memory'; got: ", burnin_algorithm)))
                           stop(paste0("burnin_algorithm must be one of 'ChEES', 'CHESSR', 'CHESSR_log', ",
                                       "'SNAPER', 'CHESSR_time', 'SNAPER_time', 'ESJD', 'ESJD_CHESSR', ",
                                       "'ESJD_SNAPER', 'LQ_ESSR', 'L_ESSR_length_response', ",
                                       "'LQ_ESSR_length_response', 'LQ_ESSR_spectral', ",
                                       "'LQ_ESSR_spectral_long_bin_memory', ",
                                       "'LQ_ESSR_spec_bins99_evid_expand' or ",
                                       ## "'ESJD_LQ_ESSR_spec_bins99_finite_N_evid_expand'; got: ",
                                       "'ESJD_LQ_ESSR_spec_bins99_finite_N_evid_expand' or ",
                                       "'ESJD_LQ_ESSR_spec_bins99_evid_expand'; got: ",
                                       burnin_algorithm)))
        return(algorithm)

}




 

 



#' After updating snaper_w but before computing eigen_max
#' fn_stabilize_snaper_w
#' @export
fn_stabilize_snaper_w <- function(snaper_w_vec, 
                                  ii) {
  
        # Remove NaNs first
        if (any(is.nan(snaper_w_vec))) {
          cat("NaN detected in snaper_w, resetting\n")
          snaper_w_vec[is.nan(snaper_w_vec)] <- 0.01
        }
        
        # # # Option 1: Normalize by iteration (reduces influence of early iterations)
        # if (ii > 10) {
        #   scale_factor <- sqrt(ii) / 10  # Grows slowly
        #   snaper_w_vec <- snaper_w_vec / scale_factor
        # }
        # # 
        # # Option 2: Clip extreme values
        # w_magnitude <- sqrt(sum(snaper_w_vec^2))
        # ##
        # if (w_magnitude > 10) {
        #   snaper_w_vec <- snaper_w_vec * (10 / w_magnitude)
        # }
        # 
        # # Option 3: Use more aggressive decay for large w
        # if (w_magnitude > 5) {
        #   beta_extra <- 0.9  # Extra decay
        #   snaper_w_vec <- snaper_w_vec * beta_extra
        # }
        
        return(snaper_w_vec)
  
}
















 











