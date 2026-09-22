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

fn_apply_trajectory_metric <-  function( metric_factor,
                                         values) {

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
                                           direction = NULL) {

        theta_initial <-  as.matrix(theta_initial)
        theta_proposed <-  as.matrix(theta_proposed)
        velocity_proposed <-  as.matrix(velocity_proposed)
        if (!identical(dim(theta_initial), dim(theta_proposed)) ||
            !identical(dim(theta_initial), dim(velocity_proposed)) ||
            length(mean_initial) != nrow(theta_initial) || length(mean_proposed) != nrow(theta_initial) ||
            length(tau_values) != ncol(theta_initial)) stop("Trajectory endpoint dimensions do not agree.")
        if (!algorithm %in% c("ChEES", "CHESSR", "CHESSR_log", "SNAPER")) stop("Unknown position-based trajectory algorithm.")
        ##
        initial <-  fn_apply_trajectory_metric(metric_factor, theta_initial - mean_initial)
        proposed <-  fn_apply_trajectory_metric(metric_factor, theta_proposed - mean_proposed)
        velocity <-  fn_apply_trajectory_metric(metric_factor, velocity_proposed)
        ## velocity already equals d(theta)/dt. No additional M_inv belongs here.
        if (algorithm == "SNAPER") {
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
                                         bias_correction_step = NULL) {

        use_proposals <-  isTRUE(weight_by_probability)
        criterion <-  fn_metric_position_criterion(
            algorithm = algorithm,
            theta_initial = theta_initial,
            theta_proposed = if (use_proposals) theta_proposed else theta_accepted,
            velocity_proposed = if (use_proposals) velocity_proposed else velocity_accepted,
            mean_initial = mean_initial,
            mean_proposed = if (use_proposals) mean_proposed else mean_initial,
            metric_factor = metric_factor,
            tau_values = tau_values,
            direction = direction)
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
                    aggregation = "weighted_mean")
        }
        return(list(updated = updated,
                     adam_update_performed = isTRUE(attr(updated, "adam_update_performed")),
                     gradient = gradient_used,
                     criterion = sum(weights * values) / length(weights),
                     criterion_ema = criterion_updated))

}
