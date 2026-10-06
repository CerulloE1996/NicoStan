#### =====================================================================================================================================
## R_fn_trajectory_criterion_options.R
##
## ---- Advanced trajectory-criterion options and pure endpoint reductions ---------------------------------------------------------------
##
fn_validate_trajectory_criterion_options <-  function( tau_gradient_estimator,
                                                       tau_cost_exponent,
                                                       esjd_jump_power,
                                                       tau_jitter_burnin,
                                                       algorithm,
                                                       randomize_tau_burnin) {

        algorithm_values <-  c("KE", "ChEES", "CHESSR", "CHESSR_log", "SNAPER",
                               "CHESSR_time", "SNAPER_time", "ESJD", "ESJD_CHESSR", "ESJD_SNAPER")
        if (!is.character(tau_gradient_estimator) || length(tau_gradient_estimator) != 1 ||
            is.na(tau_gradient_estimator) || !tau_gradient_estimator %in% c("forward", "two_ended")) {
                stop("tau_gradient_estimator must be 'forward' or 'two_ended'.")
        }
        if (!is.numeric(tau_cost_exponent) || length(tau_cost_exponent) != 1 ||
            !is.finite(tau_cost_exponent) || tau_cost_exponent < 0 || tau_cost_exponent > 1.5) {
                stop("tau_cost_exponent must be one finite number in [0, 1.5].")
        }
        if (!is.numeric(esjd_jump_power) || length(esjd_jump_power) != 1 ||
            !is.finite(esjd_jump_power) || esjd_jump_power < 2 || esjd_jump_power > 4 ||
            esjd_jump_power != floor(esjd_jump_power)) {
                stop("esjd_jump_power must be one of 2, 3 or 4.")
        }
        if (!is.character(tau_jitter_burnin) || length(tau_jitter_burnin) != 1 ||
            is.na(tau_jitter_burnin) || !tau_jitter_burnin %in% c("uniform", "halton")) {
                stop("tau_jitter_burnin must be 'uniform' or 'halton'.")
        }
        if (!is.character(algorithm) || length(algorithm) != 1 ||
            is.na(algorithm) || !algorithm %in% algorithm_values) {
                stop(paste0("algorithm must be one of ", paste(algorithm_values, collapse = ", "), "."))
        }
        if (!is.logical(randomize_tau_burnin) || length(randomize_tau_burnin) != 1 ||
            is.na(randomize_tau_burnin)) {
                stop("randomize_tau_burnin must be TRUE or FALSE.")
        }
        nondefault_options <- !identical(tau_gradient_estimator, "forward") ||
                              !isTRUE(tau_cost_exponent == 1) ||
                              !isTRUE(esjd_jump_power == 2)
        if (algorithm == "KE" && nondefault_options) {
                stop("KE does not accept non-default trajectory-criterion options.")
        }
        if (algorithm == "ChEES" && !isTRUE(tau_cost_exponent == 1)) {
                stop("ChEES has an unnormalised criterion; tau_cost_exponent is not applied and must retain its legacy value 1.")
        }
        if (algorithm %in% c("CHESSR_log", "CHESSR_time", "SNAPER_time") && !isTRUE(tau_cost_exponent == 1)) {
                stop("The log and _time criteria use their existing penalties; tau_cost_exponent must retain its legacy value 1.")
        }
        if (!algorithm %in% c("ESJD", "ESJD_CHESSR", "ESJD_SNAPER") && !isTRUE(esjd_jump_power == 2)) {
                stop("esjd_jump_power is applied only by ESJD, ESJD_CHESSR and ESJD_SNAPER; other criteria require 2.")
        }
        if (identical(tau_jitter_burnin, "halton") && !isTRUE(randomize_tau_burnin)) {
                stop("Halton burn-in jitter requires randomize_tau_burnin = TRUE.")
        }
        return(list(tau_gradient_estimator = tau_gradient_estimator,
                    tau_cost_exponent = as.numeric(tau_cost_exponent),
                    esjd_jump_power = as.numeric(esjd_jump_power),
                    tau_jitter_burnin = tau_jitter_burnin,
                    algorithm = algorithm,
                    randomize_tau_burnin = randomize_tau_burnin))

}

fn_trajectory_criterion_options_are_legacy <-  function(options) {

        return(identical(options$tau_gradient_estimator, "forward") &&
               isTRUE(options$tau_cost_exponent == 1) &&
               isTRUE(options$esjd_jump_power == 2))

}

fn_trajectory_criterion_rate_result <-  function(numerator,
                                                  numerator_gradient,
                                                  tau_values,
                                                  tau_cost_exponent) {

        denominator <-  tau_values^tau_cost_exponent
        return(list(gradient = (numerator_gradient - tau_cost_exponent * numerator) / denominator,
                    criterion = numerator / denominator,
                    numerator_gradient = numerator_gradient,
                    numerator = numerator))

}

fn_trajectory_criterion_time_result <-  function(numerator,
                                                  numerator_gradient,
                                                  tau_values,
                                                  tau_offset_from_sampling_overhead,
                                                  burnin_to_sampling_leapfrog_time_ratio,
                                                  lag_one_autocorrelation_rho) {

        time_to_target_ESS_tau_penalty <-  fn_time_to_target_ESS_tau_penalty(
            tau_values = tau_values,
            tau_offset_from_sampling_overhead = tau_offset_from_sampling_overhead,
            burnin_to_sampling_leapfrog_time_ratio = burnin_to_sampling_leapfrog_time_ratio,
            lag_one_autocorrelation_rho = lag_one_autocorrelation_rho)
        return(list(gradient = (numerator_gradient - numerator * time_to_target_ESS_tau_penalty) / tau_values,
                    criterion = numerator / tau_values,
                    numerator_gradient = numerator_gradient,
                    numerator = numerator,
                    time_to_target_ESS_tau_penalty = time_to_target_ESS_tau_penalty))

}

fn_trajectory_criterion_position_statistic <-  function( statistic_algorithm,
                                                          initial,
                                                          proposed,
                                                          velocity_initial,
                                                          velocity_proposed,
                                                          direction,
                                                          tau_values,
                                                          tau_gradient_estimator) {

        if (statistic_algorithm %in% c("SNAPER", "SNAPER_time")) {
                if (length(direction) != nrow(initial) || any(!is.finite(direction))) stop("Invalid SNAPER direction.")
                projection_initial <-  c(crossprod(direction, initial))
                projection_proposed <-  c(crossprod(direction, proposed))
                projection_velocity_initial <- c(crossprod(direction, velocity_initial))
                projection_velocity_proposed <- c(crossprod(direction, velocity_proposed))
                delta <-  projection_proposed^2 - projection_initial^2
                derivative_forward <-  2 * projection_proposed * projection_velocity_proposed
                derivative_symmetric <- projection_proposed * projection_velocity_proposed +
                                        projection_initial * projection_velocity_initial
        } else {
                delta <-  0.5 * (colSums(proposed^2) - colSums(initial^2))
                derivative_forward <- colSums(proposed * velocity_proposed)
                derivative_symmetric <- 0.5 * (derivative_forward + colSums(initial * velocity_initial))
        }
        numerator <-  delta^2
        numerator_gradient <- if (tau_gradient_estimator == "two_ended") {
                2 * delta * derivative_symmetric * tau_values
        } else {
                2 * delta * derivative_forward * tau_values
        }
        return(list(numerator = numerator, numerator_gradient = numerator_gradient))

}

fn_trajectory_criterion_esjd_statistic <-  function( theta_initial,
                                                     theta_proposed,
                                                     velocity_initial,
                                                     velocity_proposed,
                                                     metric_factor,
                                                     tau_values,
                                                     tau_gradient_estimator,
                                                     esjd_jump_power) {

        jump <-  fn_apply_trajectory_metric(metric_factor, theta_proposed - theta_initial)
        velocity_initial_metric <- fn_apply_trajectory_metric(metric_factor, velocity_initial)
        velocity_proposed_metric <- fn_apply_trajectory_metric(metric_factor, velocity_proposed)
        jump_norm <- sqrt(pmax(colSums(jump^2), 0))
        jump_power_factor <- ifelse(jump_norm > 0, jump_norm^(esjd_jump_power - 2),
                                    if (esjd_jump_power == 2) 1 else 0)
        dot_proposed <- colSums(jump * velocity_proposed_metric)
        dot_initial <- colSums(jump * velocity_initial_metric)
        numerator <- jump_norm^esjd_jump_power
        numerator_gradient <- if (tau_gradient_estimator == "two_ended") {
                (esjd_jump_power / 2) * jump_power_factor * (dot_proposed + dot_initial) * tau_values
        } else {
                esjd_jump_power * jump_power_factor * dot_proposed * tau_values
        }
        return(list(numerator = numerator, numerator_gradient = numerator_gradient))

}

fn_metric_position_criterion_advanced <-  function( algorithm,
                                                    theta_initial,
                                                    theta_proposed,
                                                    velocity_initial,
                                                    velocity_proposed,
                                                    mean_initial,
                                                    mean_proposed,
                                                    metric_factor,
                                                    tau_values,
                                                    direction,
                                                    tau_gradient_estimator,
                                                    tau_cost_exponent,
                                                    esjd_jump_power,
                                                    tau_jitter_burnin,
                                                    randomize_tau_burnin,
                                                    tau_offset_from_sampling_overhead,
                                                    burnin_to_sampling_leapfrog_time_ratio,
                                                    lag_one_autocorrelation_rho) {

        options <- fn_validate_trajectory_criterion_options(
            tau_gradient_estimator = tau_gradient_estimator,
            tau_cost_exponent = tau_cost_exponent,
            esjd_jump_power = esjd_jump_power,
            tau_jitter_burnin = tau_jitter_burnin,
            algorithm = algorithm,
            randomize_tau_burnin = randomize_tau_burnin)
        if (fn_trajectory_criterion_options_are_legacy(options)) {
                return(fn_metric_position_criterion(
                    algorithm = algorithm,
                    theta_initial = theta_initial,
                    theta_proposed = theta_proposed,
                    velocity_proposed = velocity_proposed,
                    mean_initial = mean_initial,
                    mean_proposed = mean_proposed,
                    metric_factor = metric_factor,
                    tau_values = tau_values,
                    direction = direction,
                    tau_offset_from_sampling_overhead = tau_offset_from_sampling_overhead,
                    burnin_to_sampling_leapfrog_time_ratio = burnin_to_sampling_leapfrog_time_ratio,
                    lag_one_autocorrelation_rho = lag_one_autocorrelation_rho))
        }

        theta_initial <- as.matrix(theta_initial)
        theta_proposed <- as.matrix(theta_proposed)
        velocity_initial <- as.matrix(velocity_initial)
        velocity_proposed <- as.matrix(velocity_proposed)
        tau_values <- c(tau_values)
        if (!identical(dim(theta_initial), dim(theta_proposed)) ||
            !identical(dim(theta_initial), dim(velocity_initial)) ||
            !identical(dim(theta_initial), dim(velocity_proposed)) ||
            length(mean_initial) != nrow(theta_initial) || length(mean_proposed) != nrow(theta_initial) ||
            length(tau_values) != ncol(theta_initial) || any(!is.finite(tau_values)) || any(tau_values <= 0)) {
                stop("Trajectory endpoint dimensions do not agree.")
        }
        initial <- fn_apply_trajectory_metric(metric_factor, theta_initial - mean_initial)
        proposed <- fn_apply_trajectory_metric(metric_factor, theta_proposed - mean_proposed)
        velocity_initial_metric <- fn_apply_trajectory_metric(metric_factor, velocity_initial)
        velocity_proposed_metric <- fn_apply_trajectory_metric(metric_factor, velocity_proposed)

        is_esjd <- algorithm %in% c("ESJD", "ESJD_CHESSR", "ESJD_SNAPER")
        if (is_esjd) {
                esjd <- fn_trajectory_criterion_esjd_statistic(
                    theta_initial = theta_initial,
                    theta_proposed = theta_proposed,
                    velocity_initial = velocity_initial,
                    velocity_proposed = velocity_proposed,
                    metric_factor = metric_factor,
                    tau_values = tau_values,
                    tau_gradient_estimator = options$tau_gradient_estimator,
                    esjd_jump_power = options$esjd_jump_power)
                esjd_result <- fn_trajectory_criterion_rate_result(
                    numerator = esjd$numerator,
                    numerator_gradient = esjd$numerator_gradient,
                    tau_values = tau_values,
                    tau_cost_exponent = options$tau_cost_exponent)
                if (algorithm == "ESJD") return(esjd_result)
        }

        statistic_algorithm <- if (algorithm == "ESJD_CHESSR") "CHESSR" else
                               if (algorithm == "ESJD_SNAPER") "SNAPER" else algorithm
        if (!statistic_algorithm %in% c("ChEES", "CHESSR", "CHESSR_log", "SNAPER", "CHESSR_time", "SNAPER_time")) {
                stop(paste0("Unknown position-based trajectory algorithm: ", algorithm))
        }
        statistic <- fn_trajectory_criterion_position_statistic(
            statistic_algorithm = statistic_algorithm,
            initial = initial,
            proposed = proposed,
            velocity_initial = velocity_initial_metric,
            velocity_proposed = velocity_proposed_metric,
            direction = direction,
            tau_values = tau_values,
            tau_gradient_estimator = options$tau_gradient_estimator)
        secondary_result <- if (statistic_algorithm == "ChEES") {
                list(gradient = statistic$numerator_gradient, criterion = statistic$numerator,
                     numerator_gradient = statistic$numerator_gradient, numerator = statistic$numerator)
        } else if (statistic_algorithm %in% c("CHESSR_time", "SNAPER_time")) {
                fn_trajectory_criterion_time_result(
                    numerator = statistic$numerator,
                    numerator_gradient = statistic$numerator_gradient,
                    tau_values = tau_values,
                    tau_offset_from_sampling_overhead = tau_offset_from_sampling_overhead,
                    burnin_to_sampling_leapfrog_time_ratio = burnin_to_sampling_leapfrog_time_ratio,
                    lag_one_autocorrelation_rho = lag_one_autocorrelation_rho)
        } else {
                fn_trajectory_criterion_rate_result(
                    numerator = statistic$numerator,
                    numerator_gradient = statistic$numerator_gradient,
                    tau_values = tau_values,
                    tau_cost_exponent = options$tau_cost_exponent)
        }
        if (algorithm == "ESJD_CHESSR") return(list(ESJD = esjd_result, CHESSR = secondary_result))
        if (algorithm == "ESJD_SNAPER") return(list(ESJD = esjd_result, SNAPER = secondary_result))
        return(secondary_result)

}

fn_trajectory_criterion_reduction_component <-  function(reductions,
                                                          candidate_names,
                                                          label) {

        sources <- list(reductions)
        if (is.list(reductions$initial_contractions)) {
                sources[[length(sources) + 1]] <- reductions$initial_contractions
        }
        for (source in sources) {
                for (candidate_name in candidate_names) {
                        if (!is.null(source[[candidate_name]])) return(c(source[[candidate_name]]))
                }
        }
        stop(paste0("Missing ", label, " in trajectory reductions."))

}

fn_trajectory_criterion_reduction_optional <-  function(reductions,
                                                         candidate_names) {

        sources <- list(reductions)
        if (is.list(reductions$initial_contractions)) {
                sources[[length(sources) + 1]] <- reductions$initial_contractions
        }
        for (source in sources) {
                for (candidate_name in candidate_names) {
                        if (!is.null(source[[candidate_name]])) return(c(source[[candidate_name]]))
                }
        }
        return(NULL)

}

fn_trajectory_criterion_reduction_aligned <-  function(reductions,
                                                        candidate_names,
                                                        label,
                                                        tau_values) {

        value <- fn_trajectory_criterion_reduction_component(
            reductions = reductions,
            candidate_names = candidate_names,
            label = label)
        if (length(value) != length(tau_values)) stop(paste0(label, " must match tau_values."))
        return(value)

}

fn_trajectory_criterion_reduction_component_combination <-  function(reductions,
                                                                     candidate_names,
                                                                     first_component_names,
                                                                     second_component_names,
                                                                     label,
                                                                     tau_values) {

        direct <- fn_trajectory_criterion_reduction_optional(reductions, candidate_names)
        if (!is.null(direct)) {
                if (length(direct) != length(tau_values)) stop(paste0(label, " must match tau_values."))
                return(direct)
        }
        first_component <- fn_trajectory_criterion_reduction_aligned(
            reductions, first_component_names, paste0(label, " first contraction"), tau_values)
        second_component <- fn_trajectory_criterion_reduction_aligned(
            reductions, second_component_names, paste0(label, " second contraction"), tau_values)
        return(first_component - second_component)

}

fn_trajectory_criterion_reduction_statistic <-  function( statistic_algorithm,
                                                           reductions,
                                                           tau_values,
                                                           tau_gradient_estimator) {

        if (statistic_algorithm %in% c("SNAPER", "SNAPER_time")) {
                projection_initial <- fn_trajectory_criterion_reduction_aligned(
                    reductions, c("projection_initial"), "projection_initial", tau_values)
                projection_proposed <- fn_trajectory_criterion_reduction_aligned(
                    reductions, c("projection_proposed"), "projection_proposed", tau_values)
                projection_velocity_proposed <- fn_trajectory_criterion_reduction_aligned(
                    reductions, c("projection_velocity"), "projection_velocity", tau_values)
                projection_velocity_initial <- if (tau_gradient_estimator == "two_ended") {
                        fn_trajectory_criterion_reduction_aligned(
                            reductions, c("projection_initial_velocity", "projection_velocity_initial"),
                            "projection_initial_velocity", tau_values)
                } else 0
                delta <- projection_proposed^2 - projection_initial^2
                derivative_forward <- 2 * projection_proposed * projection_velocity_proposed
                derivative_symmetric <- projection_proposed * projection_velocity_proposed +
                                        projection_initial * projection_velocity_initial
        } else {
                proposed_sq <- fn_trajectory_criterion_reduction_aligned(
                    reductions, c("column_sums_proposed_sq"), "column_sums_proposed_sq", tau_values)
                initial_sq <- fn_trajectory_criterion_reduction_aligned(
                    reductions, c("column_sums_initial_sq"), "column_sums_initial_sq", tau_values)
                proposed_velocity <- fn_trajectory_criterion_reduction_aligned(
                    reductions, c("column_sums_proposed_velocity"), "column_sums_proposed_velocity", tau_values)
                initial_velocity <- if (tau_gradient_estimator == "two_ended") {
                        fn_trajectory_criterion_reduction_aligned(
                            reductions, c("column_sums_initial_velocity", "column_sums_initial_x_velocity"),
                            "column_sums_initial_velocity", tau_values)
                } else 0
                delta <- 0.5 * (proposed_sq - initial_sq)
                derivative_forward <- proposed_velocity
                derivative_symmetric <- 0.5 * (proposed_velocity + initial_velocity)
        }
        numerator <- delta^2
        numerator_gradient <- if (tau_gradient_estimator == "two_ended") {
                2 * delta * derivative_symmetric * tau_values
        } else {
                2 * delta * derivative_forward * tau_values
        }
        return(list(numerator = numerator, numerator_gradient = numerator_gradient))

}

fn_metric_position_criterion_from_reductions_advanced <-  function( algorithm,
                                                                    reductions,
                                                                    tau_values,
                                                                    tau_gradient_estimator,
                                                                    tau_cost_exponent,
                                                                    esjd_jump_power,
                                                                    tau_jitter_burnin,
                                                                    randomize_tau_burnin,
                                                                    tau_offset_from_sampling_overhead,
                                                                    burnin_to_sampling_leapfrog_time_ratio,
                                                                    lag_one_autocorrelation_rho) {

        options <- fn_validate_trajectory_criterion_options(
            tau_gradient_estimator = tau_gradient_estimator,
            tau_cost_exponent = tau_cost_exponent,
            esjd_jump_power = esjd_jump_power,
            tau_jitter_burnin = tau_jitter_burnin,
            algorithm = algorithm,
            randomize_tau_burnin = randomize_tau_burnin)
        if (fn_trajectory_criterion_options_are_legacy(options) &&
            !algorithm %in% c("ESJD", "ESJD_CHESSR", "ESJD_SNAPER")) {
                return(fn_metric_position_criterion_from_reductions(
                    algorithm = algorithm,
                    reductions = reductions,
                    tau_values = tau_values,
                    tau_offset_from_sampling_overhead = tau_offset_from_sampling_overhead,
                    burnin_to_sampling_leapfrog_time_ratio = burnin_to_sampling_leapfrog_time_ratio,
                    lag_one_autocorrelation_rho = lag_one_autocorrelation_rho))
        }

        tau_values <- c(tau_values)
        if (length(tau_values) < 1 || any(!is.finite(tau_values)) || any(tau_values <= 0)) {
                stop("tau_values must be positive and finite.")
        }
        is_esjd <- algorithm %in% c("ESJD", "ESJD_CHESSR", "ESJD_SNAPER")
        if (is_esjd) {
                generic_esjd_contractions_are_raw <-  isTRUE(reductions$esjd_reductions_are_uncentred) ||
                                                      isTRUE(reductions$esjd_endpoint_centres_are_equal)
                ## ESJD uses the uncentred position difference. Direct jump contractions keep it separate from the centred
                ## contractions used by the position statistics, including when the two endpoint centres differ.
                jump_squared <-  fn_trajectory_criterion_reduction_optional(
                    reductions, c("column_sums_jump_sq", "column_sums_raw_jump_sq", "jump_squared"))
                if (is.null(jump_squared)) {
                        if (!generic_esjd_contractions_are_raw) {
                                stop("ESJD reductions need a raw jump square or an explicit uncentred/equal-centres contract.")
                        }
                        proposed_sq <- fn_trajectory_criterion_reduction_aligned(
                            reductions, c("column_sums_proposed_sq"), "column_sums_proposed_sq", tau_values)
                        initial_sq <- fn_trajectory_criterion_reduction_aligned(
                            reductions, c("column_sums_initial_sq"), "column_sums_initial_sq", tau_values)
                        initial_proposed <- fn_trajectory_criterion_reduction_aligned(
                            reductions,
                            c("column_sums_initial_proposed", "column_sums_proposed_initial", "column_sums_initial_x_proposed"),
                            "column_sums_initial_proposed", tau_values)
                        jump_squared <- pmax(proposed_sq + initial_sq - 2 * initial_proposed, 0)
                } else {
                        if (length(jump_squared) != length(tau_values)) stop("Raw jump squares must match tau_values.")
                        jump_squared <-  pmax(jump_squared, 0)
                }
                jump_norm <- sqrt(jump_squared)
                jump_power_factor <- ifelse(jump_norm > 0, jump_norm^(options$esjd_jump_power - 2),
                                             if (options$esjd_jump_power == 2) 1 else 0)
                if (is.null(fn_trajectory_criterion_reduction_optional(reductions,
                            c("column_sums_jump_velocity_proposed", "column_sums_raw_jump_velocity_proposed"))) &&
                    !generic_esjd_contractions_are_raw) {
                        stop("ESJD reductions need a raw proposed-velocity jump contraction or an explicit uncentred/equal-centres contract.")
                }
                jump_velocity_proposed <- fn_trajectory_criterion_reduction_component_combination(
                    reductions,
                    c("column_sums_jump_velocity_proposed", "column_sums_raw_jump_velocity_proposed"),
                    c("column_sums_proposed_velocity"),
                    c("column_sums_initial_velocity_proposed", "column_sums_initial_x_velocity_proposed"),
                    "jump velocity at proposed endpoint", tau_values)
                jump_velocity_initial <- if (options$tau_gradient_estimator == "two_ended") {
                        if (is.null(fn_trajectory_criterion_reduction_optional(reductions,
                                    c("column_sums_jump_velocity_initial", "column_sums_raw_jump_velocity_initial"))) &&
                            !generic_esjd_contractions_are_raw) {
                                stop("ESJD reductions need a raw initial-velocity jump contraction or an explicit uncentred/equal-centres contract.")
                        }
                        fn_trajectory_criterion_reduction_component_combination(
                            reductions,
                            c("column_sums_jump_velocity_initial", "column_sums_raw_jump_velocity_initial"),
                            c("column_sums_proposed_velocity_initial", "column_sums_proposed_x_velocity_initial"),
                            c("column_sums_initial_velocity", "column_sums_initial_x_velocity"),
                            "jump velocity at initial endpoint", tau_values)
                } else 0
                esjd_numerator <- jump_norm^options$esjd_jump_power
                esjd_numerator_gradient <- if (options$tau_gradient_estimator == "two_ended") {
                        (options$esjd_jump_power / 2) * jump_power_factor *
                            (jump_velocity_proposed + jump_velocity_initial) * tau_values
                } else {
                        options$esjd_jump_power * jump_power_factor * jump_velocity_proposed * tau_values
                }
                esjd_result <- fn_trajectory_criterion_rate_result(
                    numerator = esjd_numerator,
                    numerator_gradient = esjd_numerator_gradient,
                    tau_values = tau_values,
                    tau_cost_exponent = options$tau_cost_exponent)
                if (algorithm == "ESJD") return(esjd_result)
        }

        statistic_algorithm <- if (algorithm == "ESJD_CHESSR") "CHESSR" else
                               if (algorithm == "ESJD_SNAPER") "SNAPER" else algorithm
        if (!statistic_algorithm %in% c("ChEES", "CHESSR", "CHESSR_log", "SNAPER", "CHESSR_time", "SNAPER_time")) {
                stop(paste0("Unknown position-based trajectory algorithm: ", algorithm))
        }
        statistic <- fn_trajectory_criterion_reduction_statistic(
            statistic_algorithm = statistic_algorithm,
            reductions = reductions,
            tau_values = tau_values,
            tau_gradient_estimator = options$tau_gradient_estimator)
        secondary_result <- if (statistic_algorithm == "ChEES") {
                list(gradient = statistic$numerator_gradient, criterion = statistic$numerator,
                     numerator_gradient = statistic$numerator_gradient, numerator = statistic$numerator)
        } else if (statistic_algorithm %in% c("CHESSR_time", "SNAPER_time")) {
                fn_trajectory_criterion_time_result(
                    numerator = statistic$numerator,
                    numerator_gradient = statistic$numerator_gradient,
                    tau_values = tau_values,
                    tau_offset_from_sampling_overhead = tau_offset_from_sampling_overhead,
                    burnin_to_sampling_leapfrog_time_ratio = burnin_to_sampling_leapfrog_time_ratio,
                    lag_one_autocorrelation_rho = lag_one_autocorrelation_rho)
        } else {
                fn_trajectory_criterion_rate_result(
                    numerator = statistic$numerator,
                    numerator_gradient = statistic$numerator_gradient,
                    tau_values = tau_values,
                    tau_cost_exponent = options$tau_cost_exponent)
        }
        if (algorithm == "ESJD_CHESSR") return(list(ESJD = esjd_result, CHESSR = secondary_result))
        if (algorithm == "ESJD_SNAPER") return(list(ESJD = esjd_result, SNAPER = secondary_result))
        return(secondary_result)

}


























