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
                                           lag_one_autocorrelation_rho            = 1,
                                           ##
                                           ## ---- the trajectory cost offset (tau units): eps x (0.5 + the gradient evaluations per iteration beyond
                                           ##      the leapfrog steps), so that every rate criterion divides by the EXPECTED gradient cost of a
                                           ##      trajectory, eps x E[ceil(t / eps)] + the endpoint evaluation = t + offset, instead of t alone
                                           ##      (4 Oct 2026; 0 = the earlier behaviour, valid only when L is large). The _time criteria add it to
                                           ##      tau_offset_from_sampling_overhead:
                                           tau_cost_offset                        = 0) {

        theta_initial <-  as.matrix(theta_initial)
        theta_proposed <-  as.matrix(theta_proposed)
        velocity_proposed <-  as.matrix(velocity_proposed)
        if (!identical(dim(theta_initial), dim(theta_proposed)) ||
            !identical(dim(theta_initial), dim(velocity_proposed)) ||
            length(mean_initial) != nrow(theta_initial) || length(mean_proposed) != nrow(theta_initial) ||
            length(tau_values) != ncol(theta_initial)) stop("Trajectory endpoint dimensions do not agree.")
        ## if (!algorithm %in% c("ChEES", "CHESSR", "CHESSR_log", "SNAPER")) stop("Unknown position-based trajectory algorithm.")
        ## if (!algorithm %in% c("ChEES", "CHESSR", "CHESSR_log", "SNAPER", "CHESSR_time", "SNAPER_time")) stop("Unknown position-based trajectory algorithm.")
        ## if (!algorithm %in% c("ChEES", "CHESSR", "CHESSR_log", "SNAPER", "CHESSR_time", "SNAPER_time", "ESJD", "ESJD_CHESSR", "ESJD_SNAPER")) {
        if (!algorithm %in% c("ChEES", "CHESSR", "CHESSR_log", "SNAPER", "CHESSR_time", "SNAPER_time", "ESJD", "ESJD_CHESSR",
                              "ESJD_SNAPER", "LQ_ESSR")) {
                stop("Unknown position-based trajectory algorithm.")
        }
        ##
        initial <-  fn_apply_trajectory_metric(metric_factor, theta_initial - mean_initial)
        proposed <-  fn_apply_trajectory_metric(metric_factor, theta_proposed - mean_proposed)
        velocity <-  fn_apply_trajectory_metric(metric_factor, velocity_proposed)
        ## velocity already equals d(theta)/dt. No additional M_inv belongs here.
        ##
        ## ---- LQ_ESSR: per chain (columns) and monitored coordinate (rows), in the metric coordinates z = B (theta - m), the statistics
        ##      of the lag-one ESS bounds of the linear statistic z_j and of the centred squared statistic z_j^2
        ##      (fn_metric_tau_block_update() turns them into the criterion):
        ##        linear_jump_squared       = (z_t,j - z_0,j)^2,        from the uncentred jump B (theta_t - theta_0), as for ESJD;
        ##        quadratic_jump_squared    = (z_t,j^2 - z_0,j^2)^2,    both ends centred at the SAME m (mean_initial);
        ##      their log-tau derivatives along the trajectory, with dz_t/dt = B v_t (velocity) and dt / dlog(tau) = t (tau_values),
        ##        2 (z_t,j - z_0,j) (B v_t)_j t     and     2 (z_t,j^2 - z_0,j^2) 2 z_t,j (B v_t)_j t;
        ##      and z_0,j^2, z_0,j^4 at the trajectory start (for Var(z_j) and Var(z_j^2)). The acceptance probability enters as the
        ##      chain weight in fn_metric_tau_block_update() (or a rejected chain's accepted end equals its start). Every row of the
        ##      adapted block is monitored: the main rows for the main block, the main and nuisance rows for a joint block (as for the
        ##      other criteria). For a dense metric the coordinates are those of z (whitened combinations of the parameters), not the
        ##      parameters themselves:
        ##
        if (algorithm == "LQ_ESSR") {
                ## monitored_rows <-  if (is.list(metric_factor)) seq_len(metric_factor$n_main) else seq_len(nrow(initial))
                monitored_rows <-  seq_len(nrow(initial))
                jump_in_metric_coordinates <-  fn_apply_trajectory_metric(metric_factor,
                                                                          theta_proposed - theta_initial)[monitored_rows, , drop = FALSE]
                initial_monitored <-  initial[monitored_rows, , drop = FALSE]
                ## proposed_monitored <-  proposed[monitored_rows, , drop = FALSE]
                ## the squared statistic f(x) = (B (x - m))^2 must use ONE centre m at both endpoints (the stationarity identity
                ## b = 1 - E[w (f(Y) - f(X))^2] / (2 Var f) needs the same f), so the proposed end is centred at mean_initial too;
                ## with mean_proposed an unmoved chain would report a non-zero squared jump:
                proposed_monitored <-  fn_apply_trajectory_metric(metric_factor, theta_proposed - mean_initial)[monitored_rows, , drop = FALSE]
                velocity_monitored <-  velocity[monitored_rows, , drop = FALSE]
                trajectory_length_per_entry <-  matrix(tau_values, nrow = length(monitored_rows), ncol = length(tau_values), byrow = TRUE)
                squared_statistic_change <-  proposed_monitored^2 - initial_monitored^2
                linear_jump_squared_log_tau_derivative <-  2 * jump_in_metric_coordinates * velocity_monitored *
                                                           trajectory_length_per_entry
                return(list(linear_jump_squared                       = jump_in_metric_coordinates^2,
                            linear_jump_squared_log_tau_derivative    = linear_jump_squared_log_tau_derivative,
                            quadratic_jump_squared                    = squared_statistic_change^2,
                            quadratic_jump_squared_log_tau_derivative = 4 * squared_statistic_change * proposed_monitored *
                                                                        velocity_monitored * trajectory_length_per_entry,
                            initial_second_moment                     = initial_monitored^2,
                            initial_fourth_moment                     = initial_monitored^4,
                            ## per-chain sums, in the fields the other criteria use (chain validity and the probe's reporting):
                            numerator                                 = colSums(jump_in_metric_coordinates^2),
                            numerator_gradient                        = colSums(linear_jump_squared_log_tau_derivative),
                            tau_cost_exponent                         = 1))
        }
        ##
        ## ---- ESJD: expected squared jumped distance (Pasarica and Gelman, 2010) per unit trajectory length, in the same metric
        ##      coordinates z = B (theta - m) as the other criteria. The statistic is the squared jump of the chain,
        ##        numerator = ||z_t - z_0||^2 = ||B (theta_t - theta_0)||^2,
        ##      which does not depend on the centres m (a jump is a difference of two positions), so the uncentred positions are used and
        ##      the change is not squared again. Along the trajectory d/dt ||z_t - z_0||^2 = 2 (z_t - z_0) . dz_t/dt, with dz_t/dt = B v_t
        ##      (velocity above), and t_i = u_i * tau gives d t_i / d log(tau) = t_i, hence
        ##        numerator_gradient = 2 * ((z_t - z_0) . B v_t) * tau_values = d(numerator) / d log(tau).
        ##      Everything after the statistic is the CHESSR code below: per chain criterion = numerator / tau_values (the per-trajectory
        ##      rate) and gradient = (numerator_gradient - numerator) / tau_values = d(numerator / tau_values) / d log(tau); the acceptance
        ##      weighting and the chain mean are applied by the caller (fn_metric_tau_block_update), exactly as for CHESSR.
        ##      With the accepted endpoints (weight_by_probability = FALSE) a rejected chain has theta_t = theta_0 and contributes 0.
        ##
        ## ---- ESJD_CHESSR: both statistics of the same endpoints, returned side by side as list(ESJD = ..., CHESSR = ...), each with the
        ##      four per-chain fields of its own criterion; fn_metric_tau_block_update() combines them (geometric mean of the two criteria):
        ##
        if (algorithm %in% c("ESJD", "ESJD_CHESSR", "ESJD_SNAPER")) {
                jump_in_metric_coordinates <-  fn_apply_trajectory_metric(metric_factor, theta_proposed - theta_initial)
                ESJD_numerator <-  colSums(jump_in_metric_coordinates^2)
                ESJD_numerator_gradient <-  2 * colSums(jump_in_metric_coordinates * velocity) * tau_values
                ## cost = t + tau_cost_offset: d/dlog(tau) of N / (t + o) = (N' t - N t / (t + o)) / (t + o):
                ESJD_position_criterion <-  list(gradient           = (ESJD_numerator_gradient - ESJD_numerator *
                                                                       (tau_values / (tau_values + tau_cost_offset))) / (tau_values + tau_cost_offset),
                                                 criterion          = ESJD_numerator / (tau_values + tau_cost_offset),
                                                 numerator_gradient = ESJD_numerator_gradient,
                                                 numerator          = ESJD_numerator)
                if (algorithm == "ESJD") return(ESJD_position_criterion)
        }
        ##
        ## if (algorithm == "SNAPER") {
        if (algorithm %in% c("SNAPER", "SNAPER_time", "ESJD_SNAPER")) {
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
                                                                                       tau_offset_from_sampling_overhead      = tau_offset_from_sampling_overhead +
                                                                                                                                tau_cost_offset,
                                                                                       ## burnin_to_sampling_leapfrog_time_ratio = burnin_to_sampling_leapfrog_time_ratio)
                                                                                       burnin_to_sampling_leapfrog_time_ratio = burnin_to_sampling_leapfrog_time_ratio,
                                                                                       lag_one_autocorrelation_rho            = lag_one_autocorrelation_rho)
                return(list(gradient = (numerator_gradient - numerator * time_to_target_ESS_tau_penalty) / tau_values,
                             criterion = numerator / tau_values,
                             numerator_gradient = numerator_gradient,
                             numerator = numerator,
                             time_to_target_ESS_tau_penalty = time_to_target_ESS_tau_penalty))
        }
        ##
        ## ---- ESJD_CHESSR: the ESJD criterion (above) and the CHESSR criterion (the rate criterion below, from the ChEES delta), side by side:
        ##
        ## ---- ESJD_SNAPER: return the ESJD and SNAPER per-trajectory rates from the same endpoints. The SNAPER direction is held fixed
        ##      for this transition and is learned/transported by the burn-in driver in the same way as for SNAPER.
        if (algorithm == "ESJD_SNAPER") {
                return(list(ESJD   = ESJD_position_criterion,
                            SNAPER = list(gradient           = (numerator_gradient - numerator * (tau_values / (tau_values + tau_cost_offset))) /
                                                               (tau_values + tau_cost_offset),
                                          criterion          = numerator / (tau_values + tau_cost_offset),
                                          numerator_gradient = numerator_gradient,
                                          numerator          = numerator)))
        }
        if (algorithm == "ESJD_CHESSR") {
                return(list(ESJD   = ESJD_position_criterion,
                            CHESSR = list(gradient           = (numerator_gradient - numerator * (tau_values / (tau_values + tau_cost_offset))) /
                                                               (tau_values + tau_cost_offset),
                                          criterion          = numerator / (tau_values + tau_cost_offset),
                                          numerator_gradient = numerator_gradient,
                                          numerator          = numerator)))
        }
        ## cost = t + tau_cost_offset (see the argument): the rate criterion per unit of expected gradient cost
        return(list(gradient = (numerator_gradient - numerator * (tau_values / (tau_values + tau_cost_offset))) / (tau_values + tau_cost_offset),
                     criterion = numerator / (tau_values + tau_cost_offset),
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
                                                           lag_one_autocorrelation_rho            = 1,
                                                           ## the trajectory cost offset (as fn_metric_position_criterion; 4 Oct 2026):
                                                           tau_cost_offset                        = 0) {

        ##
        ## ---- ESJD / ESJD_CHESSR need ||z_t - z_0||^2 and (z_t - z_0) . dz_t/dt, i.e. the cross products z_0 . z_t and z_0 . dz_t/dt of the
        ##      uncentred positions, which these reductions do not carry; the resident joint block therefore passes the endpoints to
        ##      fn_metric_position_criterion() for them (R_fn_init_and_run_burnin_CHESS.R, the "status 2" route):
        ##
        if (algorithm %in% c("ESJD", "ESJD_CHESSR", "ESJD_SNAPER")) {
                stop(paste0("burnin_algorithm = '", algorithm, "' is not available from the per-chain reductions (no z_0 . z_t cross products); ",
                            "use fn_metric_position_criterion() on the endpoints."))
        }
        ## if (!algorithm %in% c("ChEES", "CHESSR", "CHESSR_log", "SNAPER")) stop("Unknown position-based trajectory algorithm.")
        if (!algorithm %in% c("ChEES", "CHESSR", "CHESSR_log", "SNAPER", "CHESSR_time", "SNAPER_time", "ESJD_SNAPER")) stop("Unknown position-based trajectory algorithm.")
        ## if (algorithm == "SNAPER") {
        if (algorithm %in% c("SNAPER", "SNAPER_time", "ESJD_SNAPER")) {
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
                                                                                       tau_offset_from_sampling_overhead      = tau_offset_from_sampling_overhead +
                                                                                                                                tau_cost_offset,
                                                                                       ## burnin_to_sampling_leapfrog_time_ratio = burnin_to_sampling_leapfrog_time_ratio)
                                                                                       burnin_to_sampling_leapfrog_time_ratio = burnin_to_sampling_leapfrog_time_ratio,
                                                                                       lag_one_autocorrelation_rho            = lag_one_autocorrelation_rho)
                return(list(gradient = (numerator_gradient - numerator * time_to_target_ESS_tau_penalty) / tau_values,
                             criterion = numerator / tau_values,
                             numerator_gradient = numerator_gradient,
                             numerator = numerator,
                             time_to_target_ESS_tau_penalty = time_to_target_ESS_tau_penalty))
        }
        ## cost = t + tau_cost_offset (see the argument): the rate criterion per unit of expected gradient cost
        return(list(gradient = (numerator_gradient - numerator * (tau_values / (tau_values + tau_cost_offset))) / (tau_values + tau_cost_offset),
                     criterion = numerator / (tau_values + tau_cost_offset),
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
                                         ## learning_rate_schedule_length          = NULL) {
                                         learning_rate_schedule_length          = NULL,
                                         ##
                                         ## ---- "ESJD_CHESSR" only: the exponential moving averages of the chain-mean ESJD and CHESSR criteria carried
                                         ##      between updates, c(ESJD = , CHESSR = ); NULL or NA = none yet. The other criteria ignore it:
                                         ## component_criterion_ema                = NULL) {
                                         ## component_criterion_ema) {
                                         component_criterion_ema,
                                         ##
                                         ## ---- "LQ_ESSR" only: the rows (main-block indices) the criterion monitors; NULL = every row of the block:
                                         interest_rows                          = NULL,
                                         ##
                                         ## ---- the trajectory cost offset of the rate criteria (fn_metric_position_criterion; 4 Oct 2026):
                                         tau_cost_offset                        = 0) {

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
            lag_one_autocorrelation_rho = lag_one_autocorrelation_rho,
            tau_cost_offset = tau_cost_offset)
        }  ## end of: if (!is.null(position_criterion))
        ##
        if (length(probabilities) != ncol(as.matrix(theta_initial)) ||
            length(divergences) != length(probabilities)) stop("Acceptance probabilities and divergences must match the chains.")
        ##
        ## ---- ESJD_CHESSR: ascend the geometric mean of the ESJD and CHESSR criteria with equal weights,
        ##        C = sqrt( E[w * ESJD rate] * E[w * CHESSR] ),
        ##        d log C / d log(tau) = 0.5 * d log E[w * ESJD rate] / d log(tau) + 0.5 * d log E[w * CHESSR] / d log(tau),
        ##      so the scale of either criterion (a squared distance, or the squared change of a squared distance) does not set the
        ##      weights. Each term is its own criterion exactly as the code below forms it (validity, weights, zero for an invalid chain,
        ##      chain mean sum(weights * gradients) / n_chains), divided by the level of that criterion: an exponential moving average
        ##      (decay 0.9, as criterion_ema and fn_normalise_ChEES_per_tau) of its chain mean sum(weights * values) / n_chains,
        ##      including this update's mean. The moving average, rather than this update's mean alone, keeps the ratio clear of the
        ##      E[a / b] != E[a] / E[b] bias of a minibatch of a few burn-in chains. The two moving averages are carried between updates
        ##      in component_criterion_ema and returned updated; a component whose chain mean is not positive keeps its moving average,
        ##      and the update is skipped (tau and the ADAM moments unchanged) while either component has no positive weighted value
        ##      or the combined gradient is not finite. The reported criterion is sqrt(ESJD chain mean * CHESSR chain mean):
        ##
        ## ---- ESJD_SNAPER uses the same equal-weight geometric-mean update as ESJD_CHESSR, with SNAPER's learned direction as its
        ##      second per-trajectory rate. The component names below keep the two criteria's EMA, masks and probe reductions separate.
        ##
        ## ---- LQ_ESSR: ascend the soft minimum, over the monitored coordinates j, of the lag-one ESS bounds of the linear statistic
        ##      z_j and of the centred squared statistic z_j^2, per unit trajectory length:
        ##        a_j = 1 - E[w (z_t,j - z_0,j)^2] / (2 Var(z_j)),          b_j = 1 - E[w (z_t,j^2 - z_0,j^2)^2] / (2 Var(z_j^2)),
        ##      the lag-one autocorrelations of z_j and z_j^2 at stationarity (w = acceptance probability, or 1 with the accepted end),
        ##        g(r) = (1 - r) / (1 + r),        S = ( sum over the 2 x n_coordinates scores s = g(a_j), g(b_j) of s^(-q) )^(-1/q),
        ##        criterion = S / tau^c,           d log(criterion) / d log(tau) = sum_i omega_i d log(s_i) / d log(tau) - c,
        ##      with omega_i = s_i^(-q) / sum_k s_k^(-q), d log g(a_j) / d log(tau) = (d E[w (z_t,j - z_0,j)^2] / d log(tau)) /
        ##      (Var(z_j) (1 - a_j^2)) (and the same for b_j with Var(z_j^2)), and c the cost exponent (1: per unit trajectory length,
        ##      i.e. per gradient at a fixed step size). g(r) is the ESS fraction of a statistic whose autocorrelation at lag k is r^k:
        ##      exact for a normal mode of a Gaussian target with exact dynamics and a full momentum refresh, where
        ##      a = E[cos(omega t)] and b = E[cos^2(omega t)] (b = a^2 only without jitter), and an upper bound on the ESS fraction of
        ##      a general statistic of a reversible chain. g decreases in r, so it rewards anti-correlation of z_j, which does improve
        ##      the estimate of a mean; the quadratic score is what stops a mean-only optimum (b_j >= 0 always under exact Gaussian
        ##      dynamics), and the soft minimum stops the best coordinates from hiding the worst. The chain means of the four
        ##      statistics, and z_0,j^2 and z_0,j^4 (for Var(z_j) = E[z_j^2] and Var(z_j^2) = E[z_j^4] - E[z_j^2]^2, unweighted over the
        ##      valid chains), are smoothed by exponential moving averages (decay 0.9, as criterion_ema) carried between updates in
        ##      component_criterion_ema; the derivatives use this update's chain means, divided by the smoothed levels, as for
        ##      ESJD_CHESSR. The update is skipped (tau and the ADAM moments unchanged) when no chain is valid or the gradient is not
        ##      finite:
        if (algorithm == "LQ_ESSR") {
                ##
                soft_minimum_power <-  8
                lag_one_autocorrelation_bound <-  1 - 1e-6
                required_fields <-  c("linear_jump_squared", "linear_jump_squared_log_tau_derivative", "quadratic_jump_squared",
                                      "quadratic_jump_squared_log_tau_derivative", "initial_second_moment", "initial_fourth_moment")
                if (!is.list(criterion) || !all(required_fields %in% names(criterion))) {
                        stop("burnin_algorithm = 'LQ_ESSR' needs the per-coordinate statistics of fn_metric_position_criterion().")
                }
                ## interest_only: keep only the rows of the parameters of interest (the main rows lead every block):
                if (!is.null(interest_rows)) {
                        if (any(interest_rows > nrow(criterion$linear_jump_squared))) stop("interest_rows exceed the rows of the adapted block.")
                        for (field in required_fields) criterion[[field]] <-  criterion[[field]][interest_rows, , drop = FALSE]
                }
                tau_cost_exponent <-  if (is.null(criterion$tau_cost_exponent)) 1 else criterion$tau_cost_exponent
                n_chains <-  length(probabilities)
                statistics_are_finite <-  apply(is.finite(criterion$linear_jump_squared) &
                                                is.finite(criterion$linear_jump_squared_log_tau_derivative) &
                                                is.finite(criterion$quadratic_jump_squared) &
                                                is.finite(criterion$quadratic_jump_squared_log_tau_derivative) &
                                                is.finite(criterion$initial_fourth_moment), 2, all)
                valid <-  statistics_are_finite & is.finite(tau_values) & tau_values > 0 & is.finite(divergences) & divergences == 0
                weights <-  if (use_proposals) pmin(1, pmax(0, probabilities)) else rep(1, n_chains)
                weights[!is.finite(weights) | !valid] <-  0
                ##
                ## ---- chain means over the complete minibatch (an invalid chain contributes zero), and unweighted moments of the
                ##      trajectory starts over the valid chains:
                fn_weighted_chain_mean <-  function(statistic) {
                        statistic[, !valid] <-  0
                        c(statistic %*% weights) / n_chains
                }
                fn_valid_chain_mean <-  function(statistic) {
                        if (!any(valid)) return(rep(NA_real_, nrow(statistic)))
                        rowMeans(statistic[, valid, drop = FALSE])
                }
                current <-  list(linear_jump_squared       = fn_weighted_chain_mean(criterion$linear_jump_squared),
                                 quadratic_jump_squared    = fn_weighted_chain_mean(criterion$quadratic_jump_squared),
                                 initial_second_moment     = fn_valid_chain_mean(criterion$initial_second_moment),
                                 initial_fourth_moment     = fn_valid_chain_mean(criterion$initial_fourth_moment))
                linear_derivative_chain_mean <-  fn_weighted_chain_mean(criterion$linear_jump_squared_log_tau_derivative)
                quadratic_derivative_chain_mean <-  fn_weighted_chain_mean(criterion$quadratic_jump_squared_log_tau_derivative)
                ##
                ## ---- moving averages (a previous one is used only if it is a list of the same fields and lengths):
                previous_is_usable <-  is.list(component_criterion_ema) &&
                                       all(names(current) %in% names(component_criterion_ema)) &&
                                       all(vapply(names(current), function(field) {
                                               length(component_criterion_ema[[field]]) == length(current[[field]]) &&
                                               all(is.finite(component_criterion_ema[[field]]))
                                       }, logical(1)))
                component_criterion_ema_updated <-  if (!any(valid)) {
                        if (previous_is_usable) component_criterion_ema else NULL
                } else if (previous_is_usable) {
                        stats::setNames(lapply(names(current), function(field) {
                                0.9 * component_criterion_ema[[field]] + 0.1 * current[[field]]
                        }), names(current))
                } else current
                ##
                gradient_used <-  NA_real_
                criterion_mean <-  NA_real_
                linear_lag_one_autocorrelation <-  quadratic_lag_one_autocorrelation <-  NULL
                if (!is.null(component_criterion_ema_updated) && any(valid)) {
                        variance_linear <-  pmax(component_criterion_ema_updated$initial_second_moment, 1e-12)
                        variance_quadratic <-  component_criterion_ema_updated$initial_fourth_moment -
                                               component_criterion_ema_updated$initial_second_moment^2
                        ## (a non-positive or non-finite estimate falls back to the Gaussian value 2 Var(z_j)^2)
                        variance_quadratic <-  ifelse(is.finite(variance_quadratic) & variance_quadratic > 0, variance_quadratic,
                                                      2 * variance_linear^2)
                        linear_lag_one_autocorrelation_unclipped <-  1 - component_criterion_ema_updated$linear_jump_squared / (2 * variance_linear)
                        quadratic_lag_one_autocorrelation_unclipped <-  1 - component_criterion_ema_updated$quadratic_jump_squared /
                                                                            (2 * variance_quadratic)
                        linear_lag_one_autocorrelation <-  pmin(pmax(linear_lag_one_autocorrelation_unclipped, -lag_one_autocorrelation_bound),
                                                                lag_one_autocorrelation_bound)
                        quadratic_lag_one_autocorrelation <-  pmin(pmax(quadratic_lag_one_autocorrelation_unclipped, -lag_one_autocorrelation_bound),
                                                                   lag_one_autocorrelation_bound)
                        ## where the clip binds the clipped score is flat in tau, so its derivative is 0:
                        linear_score_is_clipped <-  abs(linear_lag_one_autocorrelation_unclipped) >= lag_one_autocorrelation_bound
                        quadratic_score_is_clipped <-  abs(quadratic_lag_one_autocorrelation_unclipped) >= lag_one_autocorrelation_bound
                        scores <-  c((1 - linear_lag_one_autocorrelation) / (1 + linear_lag_one_autocorrelation),
                                     (1 - quadratic_lag_one_autocorrelation) / (1 + quadratic_lag_one_autocorrelation))
                        log_score_derivatives <-  c(ifelse(linear_score_is_clipped, 0, linear_derivative_chain_mean /
                                                               (variance_linear * (1 - linear_lag_one_autocorrelation^2))),
                                                    ifelse(quadratic_score_is_clipped, 0, quadratic_derivative_chain_mean /
                                                               (variance_quadratic * (1 - quadratic_lag_one_autocorrelation^2))))
                        ## the soft minimum (sum s^(-q))^(-1/q), between n^(-1/q) min(s) and min(s), computed relative to the smallest
                        ## score (no underflow of s^(-q)):
                        smallest_score <-  min(scores)
                        relative_weights <-  (smallest_score / scores)^soft_minimum_power
                        soft_minimum <-  smallest_score * sum(relative_weights)^(-1 / soft_minimum_power)
                        omega <-  relative_weights / sum(relative_weights)
                        ## cost (mean(t) + tau_cost_offset)^c: d log cost / d log(tau) = c mean(t) / (mean(t) + offset)
                        gradient_used <-  sum(omega * log_score_derivatives) -
                                          tau_cost_exponent * (mean(tau_values[valid]) / (mean(tau_values[valid]) + tau_cost_offset))
                        criterion_mean <-  soft_minimum / (mean(tau_values[valid]) + tau_cost_offset)^tau_cost_exponent
                }
                criterion_updated <-  criterion_mean
                ##
                if (!any(valid) || !is.finite(gradient_used)) {
                        ## skipped: tau and the ADAM moments are unchanged, so this is NOT a performed update.
                        updated <-  c(tau, adam_mean, adam_variance)
                        attr(updated, "adam_update_performed") <-  FALSE
                        if (!is.finite(criterion_updated)) criterion_updated <-  criterion_ema
                } else {
                        updated <-  R_fn_update_tau_using_ADAM( n_chains                         = 1,
                                                                noisy_grads_prop_per_chain       = gradient_used,
                                                                valid_chains                     = 1,
                                                                tau                              = tau,
                                                                LR                               = learning_rate,
                                                                ii                               = iteration,
                                                                n_burnin                         = adaptation_iterations,
                                                                tau_m_adam                       = adam_mean,
                                                                tau_v_adam                       = adam_variance,
                                                                beta1_adam                       = beta1,
                                                                beta2_adam                       = beta2,
                                                                eps_adam                         = adam_epsilon,
                                                                bias_correction_step             = bias_correction_step,
                                                                aggregation                      = "weighted_mean",
                                                                learning_rate_schedule_iteration = learning_rate_schedule_iteration,
                                                                learning_rate_schedule_length    = learning_rate_schedule_length)
                }
                ##
                return(list(updated                            = updated,
                            adam_update_performed              = isTRUE(attr(updated, "adam_update_performed")),
                            gradient                           = gradient_used,
                            criterion                          = criterion_mean,
                            criterion_ema                      = criterion_updated,
                            component_criterion_ema            = component_criterion_ema_updated,
                            linear_lag_one_autocorrelation     = linear_lag_one_autocorrelation,
                            quadratic_lag_one_autocorrelation  = quadratic_lag_one_autocorrelation))
        }
        if (algorithm %in% c("ESJD_CHESSR", "ESJD_SNAPER")) {
                ##
                component_names <-  if (algorithm == "ESJD_SNAPER") c("ESJD", "SNAPER") else c("ESJD", "CHESSR")
                if (is.null(component_criterion_ema)) component_criterion_ema <-  stats::setNames(rep(NA_real_, 2), component_names)
                if (!is.numeric(component_criterion_ema) || !all(component_names %in% names(component_criterion_ema))) {
                        stop(paste0("component_criterion_ema must be a numeric vector named ESJD and ", component_names[[2]],
                                    " (NA = no moving average yet)."))
                }
                if (!is.list(criterion[[component_names[[1]]]]) || !is.list(criterion[[component_names[[2]]]])) {
                        stop(paste0("burnin_algorithm = '", algorithm, "' needs the ESJD and ", component_names[[2]],
                                    " criteria of fn_metric_position_criterion()."))
                }
                ##
                valid <-  is.finite(criterion[[component_names[[1]]]]$numerator_gradient) & is.finite(criterion[[component_names[[1]]]]$numerator) &
                          is.finite(criterion[[component_names[[2]]]]$numerator_gradient) & is.finite(criterion[[component_names[[2]]]]$numerator) &
                          is.finite(tau_values) & tau_values > 0 & is.finite(divergences) & divergences == 0
                weights <-  if (use_proposals) pmin(1, pmax(0, probabilities)) else rep(1, length(probabilities))
                weights[!is.finite(weights) | !valid] <-  0
                ##
                component_gradient_chain_mean <-  stats::setNames(rep(NA_real_, 2), component_names)
                component_criterion_chain_mean <-  stats::setNames(rep(NA_real_, 2), component_names)
                component_criterion_ema_updated <-  stats::setNames(rep(NA_real_, 2), component_names)
                component_log_tau_derivative <-  stats::setNames(rep(NA_real_, 2), component_names)
                component_informative <-  stats::setNames(rep(FALSE, 2), component_names)
                for (component_name in component_names) {
                        ##
                        component_gradients <-  criterion[[component_name]]$gradient
                        component_values <-  criterion[[component_name]]$criterion
                        component_gradients[!valid] <-  0
                        component_values[!valid] <-  0
                        component_gradient_chain_mean[[component_name]] <-  sum(weights * component_gradients) / length(weights)
                        component_criterion_chain_mean[[component_name]] <-  sum(weights * component_values) / length(weights)
                        ##
                        previous_component_ema <-  component_criterion_ema[[component_name]]
                        component_chain_mean_is_positive <-  is.finite(component_criterion_chain_mean[[component_name]]) &&
                                                             component_criterion_chain_mean[[component_name]] > 0
                        component_criterion_ema_updated[[component_name]] <-  if (!component_chain_mean_is_positive) {
                                previous_component_ema
                        } else if (is.finite(previous_component_ema) && previous_component_ema > 0) {
                                0.9 * previous_component_ema + 0.1 * component_criterion_chain_mean[[component_name]]
                        } else component_criterion_chain_mean[[component_name]]
                        ##
                        component_informative[[component_name]] <-  any(weights > 0 & component_values > 0) &&
                                                                     is.finite(component_criterion_ema_updated[[component_name]]) &&
                                                                     component_criterion_ema_updated[[component_name]] > 0
                        component_log_tau_derivative[[component_name]] <-  if (component_informative[[component_name]]) {
                                component_gradient_chain_mean[[component_name]] / component_criterion_ema_updated[[component_name]]
                        } else NA_real_
                }
                ##
                gradient_used <-  0.5 * component_log_tau_derivative[[component_names[[1]]]] + 0.5 * component_log_tau_derivative[[component_names[[2]]]]
                criterion_mean <-  sqrt(component_criterion_chain_mean[[component_names[[1]]]] * component_criterion_chain_mean[[component_names[[2]]]])
                criterion_updated <-  if (is.finite(criterion_ema) && criterion_ema > 0) {
                        0.9 * criterion_ema + 0.1 * criterion_mean
                } else criterion_mean
                ##
                if (!all(component_informative) || !is.finite(gradient_used)) {
                        ## skipped: tau and the ADAM moments are unchanged, so this is NOT a performed update.
                        updated <-  c(tau, adam_mean, adam_variance)
                        attr(updated, "adam_update_performed") <-  FALSE
                } else {
                        updated <-  R_fn_update_tau_using_ADAM( n_chains                         = 1,
                                                                noisy_grads_prop_per_chain       = gradient_used,
                                                                valid_chains                     = 1,
                                                                tau                              = tau,
                                                                LR                               = learning_rate,
                                                                ii                               = iteration,
                                                                n_burnin                         = adaptation_iterations,
                                                                tau_m_adam                       = adam_mean,
                                                                tau_v_adam                       = adam_variance,
                                                                beta1_adam                       = beta1,
                                                                beta2_adam                       = beta2,
                                                                eps_adam                         = adam_epsilon,
                                                                bias_correction_step             = bias_correction_step,
                                                                aggregation                      = "weighted_mean",
                                                                learning_rate_schedule_iteration = learning_rate_schedule_iteration,
                                                                learning_rate_schedule_length    = learning_rate_schedule_length)
                }
                ##
                return(list(updated                        = updated,
                            adam_update_performed          = isTRUE(attr(updated, "adam_update_performed")),
                            gradient                       = gradient_used,
                            criterion                      = criterion_mean,
                            criterion_ema                  = criterion_updated,
                            component_criterion_ema        = component_criterion_ema_updated,
                            component_criterion_chain_mean = component_criterion_chain_mean,
                            component_gradient_chain_mean  = component_gradient_chain_mean,
                            component_log_tau_derivative   = component_log_tau_derivative))
        }
        ##
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



























