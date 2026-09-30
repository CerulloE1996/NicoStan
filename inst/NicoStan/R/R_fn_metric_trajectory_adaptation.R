#### =====================================================================================================================================
## R_fn_metric_trajectory_adaptation.R
##
## ---- Mandatory metric coordinates for position-based trajectory adaptation ---------------------------------------------------------
##
fn_trajectory_metric_factor <-  function( mass_matrix,
                                          dimension) {

        if (is.null(dim(mass_matrix))) {
                if (length(mass_matrix) != dimension || any(!is.finite(mass_matrix)) || any(mass_matrix <= 0)) {
                        stop("The trajectory mass diagonal must be positive, finite and match the adapted block.")
                }
                return(sqrt(c(mass_matrix)))
        }
        ## R's upper Cholesky factor B satisfies t(B) %*% B = M.
        if (!identical(dim(mass_matrix), c(as.integer(dimension), as.integer(dimension))) ||
            any(!is.finite(mass_matrix)) || !isSymmetric(mass_matrix, tol = 1e-8)) {
                stop("The trajectory mass matrix must be finite, symmetric and match the adapted block.")
        }
        return(chol(mass_matrix))

}

##
## ---- Joint (main + nuisance) metric factor: a list applied block-wise, so a dense main metric never has to be expanded to the full
##      (n_main + n_nuisance)^2 matrix. EXPERIMENTAL, for testing tau_adaptation_block = "joint".
##      Rows 1..n_main of a position/velocity are the main block, the remaining rows the nuisance block (diagonal mass M_us).
##
fn_trajectory_joint_metric_factor <-  function( mass_main,
                                                mass_us_vec,
                                                n_main,
                                                n_us) {

        if (length(mass_us_vec) != n_us || any(!is.finite(mass_us_vec)) || any(mass_us_vec <= 0)) {
                stop("The nuisance mass diagonal must be positive, finite and match the nuisance block.")
        }
        return(list(main = fn_trajectory_metric_factor(mass_main, n_main),
                    us = sqrt(c(mass_us_vec)),
                    n_main = as.integer(n_main)))

}

fn_apply_trajectory_metric <-  function( metric_factor,
                                         values) {

        if (is.list(metric_factor)) {
                values <-  as.matrix(values)
                main_rows <-  seq_len(metric_factor$n_main)
                if (nrow(values) != metric_factor$n_main + length(metric_factor$us)) {
                        stop("Joint metric factor and position/velocity dimensions do not agree.")
                }
                return(rbind(fn_apply_trajectory_metric(metric_factor$main, values[main_rows, , drop = FALSE]),
                             fn_apply_trajectory_metric(metric_factor$us, values[-main_rows, , drop = FALSE])))
        }
        if (is.matrix(metric_factor)) {
                if (ncol(metric_factor) != if (is.matrix(values)) nrow(values) else length(values)) {
                        stop("Metric factor and position/velocity dimensions do not agree.")
                }
                return(metric_factor %*% values)
        }
        if (length(metric_factor) != if (is.matrix(values)) nrow(values) else length(values)) {
                stop("Metric factor and position/velocity dimensions do not agree.")
        }
        return(metric_factor * values)

}

fn_transport_snaper_direction <-  function( direction,
                                            previous_factor,
                                            metric_factor) {

        ##
        ## ---- w estimates the leading eigenvector of Cov(z), z = B (theta - m), i.e. a DISPLACEMENT in z-coordinates.
        ##      A theta-space displacement u appears in z-coordinates as B u, so the same theta-space direction in the new
        ##      coordinates is B_new B_prev^{-1} w (for a diagonal metric: (b_new / b_prev) * w). The previous map,
        ##      B_new^{-T} B_prev^T w, carried w over as a covector instead, which is the inverse of this for a diagonal metric.
        ##      (R's chol() returns the upper-triangular B with t(B) %*% B = M, so backsolve() solves B_prev x = w.)
        ##
        if (is.null(previous_factor) || identical(previous_factor, metric_factor)) return(direction)
        if (is.list(metric_factor)) {
                ## joint factor: transport each block with its own factors, then renormalise the whole direction once.
                main_rows <-  seq_len(metric_factor$n_main)
                theta_space_main <-  if (is.matrix(previous_factor$main)) c(backsolve(previous_factor$main, direction[main_rows])) else
                                         direction[main_rows] / previous_factor$main
                transported_main <-  if (is.matrix(metric_factor$main)) c(metric_factor$main %*% theta_space_main) else
                                         metric_factor$main * theta_space_main
                transported <-  c(transported_main,
                                  metric_factor$us * (direction[-main_rows] / previous_factor$us))
                magnitude <-  sqrt(sum(transported^2))
                if (!is.finite(magnitude) || magnitude == 0) return(fn_initialise_snaper_direction(length(direction)))
                return(transported / magnitude)
        }
        theta_space_direction <-  if (is.matrix(previous_factor)) c(backsolve(previous_factor, direction)) else direction / previous_factor
        transported <-  if (is.matrix(metric_factor)) c(metric_factor %*% theta_space_direction) else metric_factor * theta_space_direction
        magnitude <-  sqrt(sum(transported^2))
        if (!is.finite(magnitude) || magnitude == 0) return(fn_initialise_snaper_direction(length(direction)))
        return(transported / magnitude)

}

fn_metric_position_criterion <-  function( algorithm,
                                           theta_initial,
                                           theta_proposed,
                                           velocity_proposed,
                                           mean_initial,
                                           mean_proposed,
                                           metric_factor,
                                           tau_values,
                                           ## direction = NULL) {
                                           direction = NULL,
                                           ##
                                           ## ---- burnin_algorithm = "CHESSR_time" / "SNAPER_time" only (see R_fn_time_criterion.R); the other criteria ignore
                                           ##      them. With both at 0 the "_time" criteria are CHESSR / SNAPER exactly:
                                           tau_offset_from_sampling_overhead      = 0,
                                           ## burnin_to_sampling_leapfrog_time_ratio = 0) {
                                           burnin_to_sampling_leapfrog_time_ratio = 0,
                                           ##
                                           ## ---- "CHESSR_time" / "SNAPER_time" only: MALT's exponent, the penalty times (1 + lag_one_autocorrelation_rho) / 2
                                           ##      (R_fn_time_criterion.R, "MALT exponent"); 1 (default) leaves the penalty unchanged bit for bit:
                                           lag_one_autocorrelation_rho            = 1) {

        theta_initial <-  as.matrix(theta_initial)
        theta_proposed <-  as.matrix(theta_proposed)
        velocity_proposed <-  as.matrix(velocity_proposed)
        if (!identical(dim(theta_initial), dim(theta_proposed)) ||
            !identical(dim(theta_initial), dim(velocity_proposed)) ||
            length(mean_initial) != nrow(theta_initial) || length(mean_proposed) != nrow(theta_initial) ||
            length(tau_values) != ncol(theta_initial)) stop("Trajectory endpoint dimensions do not agree.")
        ## if (!algorithm %in% c("ChEES", "CHESSR", "CHESSR_log", "SNAPER")) stop("Unknown position-based trajectory algorithm.")
        if (!algorithm %in% c("ChEES", "CHESSR", "CHESSR_log", "SNAPER", "CHESSR_time", "SNAPER_time")) stop("Unknown position-based trajectory algorithm.")
        ##
        initial <-  fn_apply_trajectory_metric(metric_factor, theta_initial - mean_initial)
        proposed <-  fn_apply_trajectory_metric(metric_factor, theta_proposed - mean_proposed)
        velocity <-  fn_apply_trajectory_metric(metric_factor, velocity_proposed)
        ## velocity already equals d(theta)/dt. No additional M_inv belongs here.
        ## if (algorithm == "SNAPER") {
        if (algorithm %in% c("SNAPER", "SNAPER_time")) {
                if (length(direction) != nrow(initial) || any(!is.finite(direction))) stop("Invalid SNAPER direction.")
                projection_initial <-  c(crossprod(direction, initial))
                projection_proposed <-  c(crossprod(direction, proposed))
                projection_velocity <-  c(crossprod(direction, velocity))
                delta <-  projection_proposed^2 - projection_initial^2
                derivative <-  2 * projection_proposed * projection_velocity
        } else {
                delta <-  0.5 * (colSums(proposed^2) - colSums(initial^2))
                derivative <-  colSums(proposed * velocity)
        }
        ##
        numerator <-  delta^2
        numerator_gradient <-  2 * delta * derivative * tau_values
        if (algorithm == "ChEES") {
                return(list(gradient = numerator_gradient, criterion = numerator,
                            numerator_gradient = numerator_gradient, numerator = numerator))
        }
        ##
        ## ---- CHESSR_time / SNAPER_time: the rate criteria ascend  ESS_elasticity_wrt_log_tau - 1,  i.e. per chain
        ##      (numerator_gradient - numerator) / tau_values; these ascend  ESS_elasticity_wrt_log_tau - time_to_target_ESS_tau_penalty,
        ##      with the penalty (1 + burnin_to_sampling_leapfrog_time_ratio) * tau_values / (tau_values + tau_offset_from_sampling_overhead)
        ##      at each chain's own trajectory length (R_fn_time_criterion.R). The reported criterion value is the rate criterion's.
        ##      The penalty is multiplied by (1 + lag_one_autocorrelation_rho) / 2 (MALT's exponent), exactly 1 at the default
        ##      lag_one_autocorrelation_rho = 1.
        ##
        if (algorithm %in% c("CHESSR_time", "SNAPER_time")) {
                time_to_target_ESS_tau_penalty <-  fn_time_to_target_ESS_tau_penalty( tau_values                             = tau_values,
                                                                                       tau_offset_from_sampling_overhead      = tau_offset_from_sampling_overhead,
                                                                                       ## burnin_to_sampling_leapfrog_time_ratio = burnin_to_sampling_leapfrog_time_ratio)
                                                                                       burnin_to_sampling_leapfrog_time_ratio = burnin_to_sampling_leapfrog_time_ratio,
                                                                                       lag_one_autocorrelation_rho            = lag_one_autocorrelation_rho)
                return(list(gradient = (numerator_gradient - numerator * time_to_target_ESS_tau_penalty) / tau_values,
                             criterion = numerator / tau_values,
                             numerator_gradient = numerator_gradient,
                             numerator = numerator,
                             time_to_target_ESS_tau_penalty = time_to_target_ESS_tau_penalty))
        }
        return(list(gradient = (numerator_gradient - numerator) / tau_values,
                     criterion = numerator / tau_values,
                     numerator_gradient = numerator_gradient,
                     numerator = numerator))

}

##
## ---- The same criterion from its per-chain reductions, computed elsewhere: the resident burn-in interface computes them from the
##      chains' joint (main + nuisance) state inside the worker (fn_persistent_burnin_joint_position_reductions_resident), for
##      tau_adaptation_block = "joint". The reductions are, per chain,
##        SNAPER:    projection_initial, projection_proposed, projection_velocity = c(crossprod(direction, initial)), c(crossprod(direction, proposed)),
##                   c(crossprod(direction, velocity));
##        otherwise: column_sums_proposed_sq, column_sums_initial_sq, column_sums_proposed_velocity = colSums(proposed^2), colSums(initial^2),
##                   colSums(proposed * velocity);
##      everything computed from them below is the code of fn_metric_position_criterion().
##
fn_metric_position_criterion_from_reductions <-  function( algorithm,
                                                           reductions,
                                                           ## tau_values) {
                                                           tau_values,
                                                           ##
                                                           ## ---- CHESSR_time / SNAPER_time only (as in fn_metric_position_criterion):
                                                           tau_offset_from_sampling_overhead      = 0,
                                                           ## burnin_to_sampling_leapfrog_time_ratio = 0) {
                                                           burnin_to_sampling_leapfrog_time_ratio = 0,
                                                           lag_one_autocorrelation_rho            = 1) {

        ## if (!algorithm %in% c("ChEES", "CHESSR", "CHESSR_log", "SNAPER")) stop("Unknown position-based trajectory algorithm.")
        if (!algorithm %in% c("ChEES", "CHESSR", "CHESSR_log", "SNAPER", "CHESSR_time", "SNAPER_time")) stop("Unknown position-based trajectory algorithm.")
        ## if (algorithm == "SNAPER") {
        if (algorithm %in% c("SNAPER", "SNAPER_time")) {
                projection_initial <-  c(reductions$projection_initial)
                projection_proposed <-  c(reductions$projection_proposed)
                projection_velocity <-  c(reductions$projection_velocity)
                if (length(projection_initial) != length(tau_values) || length(projection_proposed) != length(tau_values) ||
                    length(projection_velocity) != length(tau_values)) stop("Trajectory endpoint dimensions do not agree.")
                delta <-  projection_proposed^2 - projection_initial^2
                derivative <-  2 * projection_proposed * projection_velocity
        } else {
                column_sums_proposed_sq <-  c(reductions$column_sums_proposed_sq)
                column_sums_initial_sq <-  c(reductions$column_sums_initial_sq)
                column_sums_proposed_velocity <-  c(reductions$column_sums_proposed_velocity)
                if (length(column_sums_proposed_sq) != length(tau_values) || length(column_sums_initial_sq) != length(tau_values) ||
                    length(column_sums_proposed_velocity) != length(tau_values)) stop("Trajectory endpoint dimensions do not agree.")
                delta <-  0.5 * (column_sums_proposed_sq - column_sums_initial_sq)
                derivative <-  column_sums_proposed_velocity
        }
        ##
        numerator <-  delta^2
        numerator_gradient <-  2 * delta * derivative * tau_values
        if (algorithm == "ChEES") {
                return(list(gradient = numerator_gradient, criterion = numerator,
                            numerator_gradient = numerator_gradient, numerator = numerator))
        }
        ##
        ## ---- CHESSR_time / SNAPER_time: the rate criteria ascend  ESS_elasticity_wrt_log_tau - 1,  i.e. per chain
        ##      (numerator_gradient - numerator) / tau_values; these ascend  ESS_elasticity_wrt_log_tau - time_to_target_ESS_tau_penalty,
        ##      with the penalty (1 + burnin_to_sampling_leapfrog_time_ratio) * tau_values / (tau_values + tau_offset_from_sampling_overhead)
        ##      at each chain's own trajectory length (R_fn_time_criterion.R). The reported criterion value is the rate criterion's.
        ##      The penalty is multiplied by (1 + lag_one_autocorrelation_rho) / 2 (MALT's exponent), exactly 1 at the default
        ##      lag_one_autocorrelation_rho = 1.
        ##
        if (algorithm %in% c("CHESSR_time", "SNAPER_time")) {
                time_to_target_ESS_tau_penalty <-  fn_time_to_target_ESS_tau_penalty( tau_values                             = tau_values,
                                                                                       tau_offset_from_sampling_overhead      = tau_offset_from_sampling_overhead,
                                                                                       ## burnin_to_sampling_leapfrog_time_ratio = burnin_to_sampling_leapfrog_time_ratio)
                                                                                       burnin_to_sampling_leapfrog_time_ratio = burnin_to_sampling_leapfrog_time_ratio,
                                                                                       lag_one_autocorrelation_rho            = lag_one_autocorrelation_rho)
                return(list(gradient = (numerator_gradient - numerator * time_to_target_ESS_tau_penalty) / tau_values,
                             criterion = numerator / tau_values,
                             numerator_gradient = numerator_gradient,
                             numerator = numerator,
                             time_to_target_ESS_tau_penalty = time_to_target_ESS_tau_penalty))
        }
        return(list(gradient = (numerator_gradient - numerator) / tau_values,
                     criterion = numerator / tau_values,
                     numerator_gradient = numerator_gradient,
                     numerator = numerator))

}

fn_metric_tau_block_update <-  function( algorithm,
                                         theta_initial,
                                         theta_proposed,
                                         theta_accepted,
                                         velocity_proposed,
                                         velocity_accepted,
                                         mean_initial,
                                         mean_proposed,
                                         metric_factor,
                                         direction,
                                         tau_values,
                                         probabilities,
                                         divergences,
                                         weight_by_probability,
                                         tau,
                                         learning_rate,
                                         iteration,
                                         adaptation_iterations,
                                         adam_mean,
                                         adam_variance,
                                         beta1,
                                         beta2,
                                         adam_epsilon,
                                         criterion_ema = NA_real_,
                                         ##
                                         ## ---- number of tau ADAM updates actually performed on these moments, counting
                                         ##      this one; used ONLY for the ADAM bias correction (iteration keeps driving
                                         ##      the learning-rate schedule). NULL = historical behaviour (uses iteration).
                                         ##
                                         bias_correction_step = NULL,
                                         ##
                                         ## ---- the position criterion (the list returned by fn_metric_position_criterion) when it is computed elsewhere,
                                         ##      e.g. by fn_metric_position_criterion_from_reductions() for the resident joint block; NULL = computed here
                                         ##      from the endpoints. When it is supplied, the endpoint arguments, the means, metric_factor and direction
                                         ##      are not used, except the number of columns (chains) of theta_initial.
                                         ##
                                         ## position_criterion = NULL) {
                                         position_criterion = NULL,
                                         ##
                                         ## ---- CHESSR_time / SNAPER_time only: passed on to fn_metric_position_criterion (endpoints). A position_criterion
                                         ##      supplied from elsewhere already carries them (fn_metric_position_criterion_from_reductions):
                                         tau_offset_from_sampling_overhead      = 0,
                                         ## burnin_to_sampling_leapfrog_time_ratio = 0) {
                                         burnin_to_sampling_leapfrog_time_ratio = 0,
                                         ## (and MALT's exponent, lag_one_autocorrelation_rho; 1 = the penalty without it)
                                         ## lag_one_autocorrelation_rho            = 1) {
                                         lag_one_autocorrelation_rho            = 1,
                                         ##
                                         ## ---- position in the tau learning-rate schedule, passed on to R_fn_update_tau_using_ADAM (NULL = iteration and
                                         ##      adaptation_iterations: one linear decay LR -> LR^2 over the tau adaptation):
                                         learning_rate_schedule_iteration       = NULL,
                                         learning_rate_schedule_length          = NULL) {

        use_proposals <-  isTRUE(weight_by_probability)
        if (!is.null(position_criterion)) {
        criterion <-  position_criterion
        } else {
        criterion <-  fn_metric_position_criterion(
            algorithm = algorithm,
            theta_initial = theta_initial,
            theta_proposed = if (use_proposals) theta_proposed else theta_accepted,
            velocity_proposed = if (use_proposals) velocity_proposed else velocity_accepted,
            mean_initial = mean_initial,
            mean_proposed = if (use_proposals) mean_proposed else mean_initial,
            metric_factor = metric_factor,
            tau_values = tau_values,
            ## direction = direction)
            direction = direction,
            tau_offset_from_sampling_overhead = tau_offset_from_sampling_overhead,
            ## burnin_to_sampling_leapfrog_time_ratio = burnin_to_sampling_leapfrog_time_ratio)
            burnin_to_sampling_leapfrog_time_ratio = burnin_to_sampling_leapfrog_time_ratio,
            lag_one_autocorrelation_rho = lag_one_autocorrelation_rho)
        }  ## end of: if (!is.null(position_criterion))
        ##
        if (length(probabilities) != ncol(as.matrix(theta_initial)) ||
            length(divergences) != length(probabilities)) stop("Acceptance probabilities and divergences must match the chains.")
        valid <-  is.finite(criterion$numerator_gradient) & is.finite(criterion$numerator) &
                  is.finite(tau_values) & tau_values > 0 & is.finite(divergences) & divergences == 0
        weights <-  if (use_proposals) pmin(1, pmax(0, probabilities)) else rep(1, length(probabilities))
        weights[!is.finite(weights) | !valid] <-  0
        gradients <-  if (algorithm == "CHESSR_log") criterion$numerator_gradient else criterion$gradient
        values <-  if (algorithm == "CHESSR_log") criterion$numerator else criterion$criterion
        gradients[!valid] <-  0
        values[!valid] <-  0
        ## Invalid/divergent chains contribute zero to the complete minibatch.
        gradient_used <-  sum(weights * gradients) / length(weights)
        criterion_mean <-  sum(weights * values) / length(weights)
        criterion_updated <-  if (is.finite(criterion_ema) && criterion_ema > 0) {
                0.9 * criterion_ema + 0.1 * criterion_mean
        } else criterion_mean
        if (algorithm == "CHESSR_log") {
                rate <-  fn_normalise_ChEES_per_tau(gradients = gradients,
                                                    criteria = values,
                                                    tau_values = tau_values,
                                                    criterion_ema = criterion_ema,
                                                    weights = weights)
                gradient_used <-  rate$gradient
                criterion_updated <-  rate$criterion_ema
        }
        ##
        if (!any(weights > 0 & values > 0) || !is.finite(gradient_used)) {
                ## skipped: tau and the ADAM moments are unchanged, so this is NOT a performed update.
                updated <-  c(tau, adam_mean, adam_variance)
                attr(updated, "adam_update_performed") <-  FALSE
        } else {
                updated <-  R_fn_update_tau_using_ADAM(
                    n_chains = 1L, noisy_grads_prop_per_chain = gradient_used,
                    valid_chains = 1L, tau = tau, LR = learning_rate,
                    ii = iteration, n_burnin = adaptation_iterations,
                    tau_m_adam = adam_mean, tau_v_adam = adam_variance,
                    beta1_adam = beta1, beta2_adam = beta2, eps_adam = adam_epsilon,
                    bias_correction_step = bias_correction_step,
                    ## aggregation = "weighted_mean")
                    aggregation = "weighted_mean",
                    learning_rate_schedule_iteration = learning_rate_schedule_iteration,
                    learning_rate_schedule_length = learning_rate_schedule_length)
        }
        ##
        ## ---- CHESSR_time / SNAPER_time: also return the chain-mean estimate of ESS_elasticity_wrt_log_tau that the criterion drives
        ##      towards time_to_target_ESS_tau_penalty, with the weights and the 1 / tau_values of the ascended quantity:
        ##        ESS_elasticity_wrt_log_tau = sum(weights * numerator_gradient / tau_values) / sum(weights * numerator / tau_values),
        ##      the value of the penalty at which the mean ascended quantity is 0 when the chains share tau. Its two chain means are
        ##      returned as well, so that estimates over several updates can be pooled as a ratio of sums. The other criteria return
        ##      the list below unchanged:
        ##
        if (algorithm %in% c("CHESSR_time", "SNAPER_time")) {
                ChEES_or_SNAPER_statistic_per_tau_chain_mean <-  sum(ifelse(valid, weights * criterion$numerator / tau_values, 0)) / length(weights)
                ChEES_or_SNAPER_statistic_log_tau_derivative_per_tau_chain_mean <-  sum(ifelse(valid, weights * criterion$numerator_gradient / tau_values, 0)) /
                                                                                    length(weights)
                ESS_elasticity_wrt_log_tau <-  if (is.finite(ChEES_or_SNAPER_statistic_per_tau_chain_mean) && ChEES_or_SNAPER_statistic_per_tau_chain_mean > 0) {
                        ChEES_or_SNAPER_statistic_log_tau_derivative_per_tau_chain_mean / ChEES_or_SNAPER_statistic_per_tau_chain_mean
                } else NA_real_
                return(list(updated = updated,
                             adam_update_performed = isTRUE(attr(updated, "adam_update_performed")),
                             gradient = gradient_used,
                             criterion = sum(weights * values) / length(weights),
                             criterion_ema = criterion_updated,
                             ChEES_or_SNAPER_statistic_per_tau_chain_mean = ChEES_or_SNAPER_statistic_per_tau_chain_mean,
                             ChEES_or_SNAPER_statistic_log_tau_derivative_per_tau_chain_mean = ChEES_or_SNAPER_statistic_log_tau_derivative_per_tau_chain_mean,
                             ESS_elasticity_wrt_log_tau = ESS_elasticity_wrt_log_tau))
        }
        return(list(updated = updated,
                     adam_update_performed = isTRUE(attr(updated, "adam_update_performed")),
                     gradient = gradient_used,
                     criterion = sum(weights * values) / length(weights),
                     criterion_ema = criterion_updated))

}






















