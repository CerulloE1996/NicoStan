##################################################################################################################
## R_fn_bulk_local_tuner.R
##################################################################################################################
##
## Local rank-normalised, multi-lag importance estimates for shorter nominal trajectories.
## Fixed geometry is a condition for the path calculation; it does not establish stationarity.
## The estimator does not assert exact equivalence to native finite-window bulk ESS.
##

fn_bulk_local_tuner_settings <-  function( options = NULL ) {

        if (is.null(options)) return(NULL)
        if (!is.list(options) || is.null(names(options)) && length(options) > 0) {
                stop("bulk_local_tuner must be NULL or a named list.")
        }
        defaults <-  list(parameter_families = NULL,
                          history_length = 64,
                          min_history = 32,
                          max_lag = 8,
                          candidate_scales = c(1, 0.95, 0.9),
                          min_effective_paths = 20,
                          min_time_clusters = 12,
                          min_gain = 0.02,
                          uncertainty_multiplier = 2,
                          stationarity_rhat_max = 1.05,
                          freeze_start_iter = NULL,
                          settle_iter = 8)
        if (anyDuplicated(names(options)) || any(!names(options) %in% names(defaults))) {
                stop("Unknown or duplicated bulk_local_tuner setting.")
        }
        for (field in names(options)) defaults[field] <-  options[field]
        ##
        integer_fields <-  c("history_length", "min_history", "max_lag", "min_effective_paths",
                             "min_time_clusters", "settle_iter")
        for (field in integer_fields) {
                value <-  defaults[[field]]
                lower <-  if (field == "settle_iter") 0 else 1
                if (!is.numeric(value) || length(value) != 1 || !is.finite(value) ||
                    value < lower || value != floor(value)) stop(paste0("Invalid ", field, "."))
        }
        if (defaults$min_history > defaults$history_length || defaults$max_lag < 3 ||
            defaults$max_lag >= defaults$min_history) stop("Incompatible history and lag settings.")
        for (field in c("min_gain", "uncertainty_multiplier", "stationarity_rhat_max")) {
                value <-  defaults[[field]]
                if (!is.numeric(value) || length(value) != 1 || !is.finite(value) || value < 0) {
                        stop(paste0("Invalid ", field, "."))
                }
        }
        if (defaults$stationarity_rhat_max < 1 || defaults$uncertainty_multiplier < 1) {
                stop("Invalid stationarity or uncertainty threshold.")
        }
        scales <-  defaults$candidate_scales
        if (!is.numeric(scales) || !length(scales) || any(!is.finite(scales)) ||
            any(scales <= 0 | scales > 1) || !any(scales == 1)) {
                stop("Candidate scales must include 1 and lie in (0, 1].")
        }
        defaults$candidate_scales <-  unique(scales)
        families <-  defaults$parameter_families
        if (!is.null(families) && (!is.character(families) || !length(families) ||
                                  anyNA(families) || any(!nzchar(families)))) {
                stop("parameter_families must be NULL or a non-empty character vector.")
        }
        freeze <-  defaults$freeze_start_iter
        if (!is.null(freeze) && (!is.numeric(freeze) || length(freeze) != 1 || !is.finite(freeze) ||
                                freeze < 0 || freeze != floor(freeze))) {
                stop("freeze_start_iter must be NULL or a non-negative integer.")
        }
        return(defaults)

}


fn_bulk_local_tuner_chain_vector <-  function( value, n_chains, name, positive = TRUE ) {

        if (!is.numeric(value) || !length(value) || !length(value) %in% c(1, n_chains) ||
            any(!is.finite(value)) || any(value < 0) || positive && any(value == 0)) {
                stop(paste0("Invalid ", name, "."))
        }
        return(rep(value, length.out = n_chains))

}


fn_bulk_local_tuner_init <-  function( settings,
                                       parameter_names,
                                       n_chains,
                                       cost_contract ) {

        settings <-  fn_bulk_local_tuner_settings(settings)
        if (!is.character(parameter_names) || !length(parameter_names) || anyNA(parameter_names) ||
            anyDuplicated(parameter_names) || any(!nzchar(parameter_names))) {
                stop("parameter_names must contain distinct main-parameter names.")
        }
        if (!is.numeric(n_chains) || length(n_chains) != 1 || !is.finite(n_chains) ||
            n_chains < 1 || n_chains != floor(n_chains)) stop("Invalid n_chains.")
        fields <-  c("endpoint_evaluations_per_iteration", "start_of_call_evaluations_per_chain",
                     "n_iter_sampling")
        if (!is.list(cost_contract) || !all(fields %in% names(cost_contract))) {
                stop("The native first-call gradient-cost contract is required.")
        }
        cost_contract$endpoint_evaluations_per_iteration <-  fn_bulk_local_tuner_chain_vector(
                cost_contract$endpoint_evaluations_per_iteration, n_chains, "endpoint evaluations", FALSE)
        cost_contract$start_of_call_evaluations_per_chain <-  fn_bulk_local_tuner_chain_vector(
                cost_contract$start_of_call_evaluations_per_chain, n_chains, "first-call evaluations", FALSE)
        sampling <-  cost_contract$n_iter_sampling
        if (!is.numeric(sampling) || length(sampling) != 1 || !is.finite(sampling) ||
            sampling < 1 || sampling != floor(sampling)) stop("Invalid n_iter_sampling.")
        selected <-  seq_along(parameter_names)
        if (!is.null(settings$parameter_families)) {
                families <-  sub("[.\\[].*$", "", parameter_names)
                if (any(!settings$parameter_families %in% families)) stop("Unknown parameter family.")
                selected <-  which(families %in% settings$parameter_families)
        }
        return(list(enabled = !is.null(settings),
                    settings = settings,
                    parameter_names = parameter_names,
                    selected_indices = selected,
                    n_chains = n_chains,
                    cost_contract = cost_contract,
                    draws = NULL,
                    leapfrog_steps = matrix(numeric(0), nrow = 0, ncol = n_chains),
                    nominal_tau = NULL,
                    eps = NULL,
                    kernel_signature = NULL,
                    shared_jitter = FALSE,
                    buffer_count = 0,
                    iteration_count = 0,
                    epoch_count = 0,
                    epoch_iterations = 0,
                    last_reset_reason = NULL))

}


fn_bulk_local_tuner_reset <-  function( state, reason ) {

        state$draws <-  NULL
        state$leapfrog_steps <-  matrix(numeric(0), nrow = 0, ncol = state$n_chains)
        state$buffer_count <-  0
        state$epoch_iterations <-  0
        state$epoch_count <-  state$epoch_count + 1
        state$last_reset_reason <-  reason
        return(state)

}


fn_bulk_local_tuner_length_probability <-  function( leapfrog_steps, nominal_tau, eps ) {

        upper <-  2 * pmax(nominal_tau, eps)
        lower_edge <-  pmax(0, (leapfrog_steps - 1) * eps)
        upper_edge <-  pmin(upper, leapfrog_steps * eps)
        probability <-  pmax(0, upper_edge - lower_edge) / upper
        probability[leapfrog_steps < 1 | leapfrog_steps != floor(leapfrog_steps)] <-  0
        return(probability)

}


fn_bulk_local_tuner_expected_length <-  function( nominal_tau, eps ) {

        width <-  2 * pmax(nominal_tau, eps) / eps
        full_bins <-  floor(width)
        remainder <-  width - full_bins
        return((full_bins * (full_bins + 1) / 2 + (full_bins + 1) * remainder) / width)

}


fn_bulk_local_tuner_append <-  function( state,
                                         theta_initial,
                                         theta_accepted,
                                         leapfrog_steps,
                                         nominal_tau,
                                         eps,
                                         kernel_signature,
                                         divergences = NULL,
                                         shared_jitter = FALSE ) {

        if (!state$enabled) return(state)
        state$iteration_count <-  state$iteration_count + 1
        valid_matrix <-  function(value) {
                return(is.matrix(value) && is.numeric(value) &&
                       identical(dim(value), as.integer(c(length(state$parameter_names), state$n_chains))) &&
                       all(is.finite(value)))
        }
        if (!valid_matrix(theta_initial) || !valid_matrix(theta_accepted)) {
                return(fn_bulk_local_tuner_reset(state, "invalid_or_nonfinite_accepted_state"))
        }
        if (!is.null(divergences) && (!is.numeric(divergences) && !is.logical(divergences) ||
                                    anyNA(divergences) || any(divergences != 0))) {
                return(fn_bulk_local_tuner_reset(state, "divergent_or_invalid_transition"))
        }
        trajectory <-  tryCatch(list(
                nominal_tau = fn_bulk_local_tuner_chain_vector(nominal_tau, state$n_chains, "nominal_tau"),
                eps = fn_bulk_local_tuner_chain_vector(eps, state$n_chains, "eps"),
                steps = fn_bulk_local_tuner_chain_vector(leapfrog_steps, state$n_chains, "leapfrog_steps")),
                error = function(error) NULL)
        if (is.null(trajectory) || any(trajectory$steps != floor(trajectory$steps)) ||
            !is.logical(shared_jitter) || length(shared_jitter) != 1 || is.na(shared_jitter) ||
            is.null(kernel_signature) || !length(kernel_signature) || anyNA(kernel_signature, recursive = TRUE)) {
                return(fn_bulk_local_tuner_reset(state, "invalid_trajectory_metadata"))
        }
        probability <-  fn_bulk_local_tuner_length_probability(trajectory$steps,
                                                               trajectory$nominal_tau,
                                                               trajectory$eps)
        if (any(!is.finite(probability) | probability <= 0)) {
                return(fn_bulk_local_tuner_reset(state, "length_outside_exact_jitter_support"))
        }
        geometry_changed <-  !identical(state$nominal_tau, trajectory$nominal_tau) ||
                             !identical(state$eps, trajectory$eps) ||
                             !identical(state$kernel_signature, kernel_signature) ||
                             !identical(state$shared_jitter, shared_jitter)
        if (geometry_changed) state <-  fn_bulk_local_tuner_reset(state, "geometry_or_nominal_law_changed")
        if (!is.null(state$draws)) {
                last <-  matrix(state$draws[, , dim(state$draws)[3]],
                                nrow = length(state$parameter_names), ncol = state$n_chains)
                if (!isTRUE(all.equal(last, theta_initial, tolerance = 0, check.attributes = FALSE))) {
                        state <-  fn_bulk_local_tuner_reset(state, "accepted_state_discontinuity")
                }
        }
        state$nominal_tau <-  trajectory$nominal_tau
        state$eps <-  trajectory$eps
        state$kernel_signature <-  kernel_signature
        state$shared_jitter <-  shared_jitter
        if (is.null(state$draws)) {
                state$draws <-  array(theta_initial,
                                     dim = c(length(state$parameter_names), state$n_chains, 1))
        }
        new_draws <-  array(0, dim = c(length(state$parameter_names), state$n_chains,
                                      dim(state$draws)[3] + 1))
        new_draws[, , seq_len(dim(state$draws)[3])] <-  state$draws
        new_draws[, , dim(new_draws)[3]] <-  theta_accepted
        state$draws <-  new_draws
        state$leapfrog_steps <-  rbind(state$leapfrog_steps, trajectory$steps)
        state$epoch_iterations <-  state$epoch_iterations + 1
        retained <-  min(nrow(state$leapfrog_steps), state$settings$history_length)
        if (nrow(state$leapfrog_steps) > retained) {
                first <-  nrow(state$leapfrog_steps) - retained + 1
                state$leapfrog_steps <-  state$leapfrog_steps[first:nrow(state$leapfrog_steps), , drop = FALSE]
                state$draws <-  state$draws[, , first:dim(state$draws)[3], drop = FALSE]
        }
        state$buffer_count <-  retained
        return(state)

}


fn_bulk_local_tuner_rank <-  function( value ) {

        ranks <-  rank(value, ties.method = "average")
        return(stats::qnorm((ranks - 3 / 8) / (length(ranks) + 1 / 4)))

}


fn_bulk_local_tuner_rhat <-  function( draws ) {

        n_draws <-  nrow(draws)
        half <-  floor(n_draws / 2)
        if (half < 2) return(NA_real_)
        split <-  cbind(draws[seq_len(half), , drop = FALSE],
                        draws[(n_draws - half + 1):n_draws, , drop = FALSE])
        within <-  mean(apply(split, 2, stats::var))
        between <-  half * stats::var(colMeans(split))
        if (!is.finite(within) || within <= 0) return(Inf)
        return(sqrt(((half - 1) * within / half + between / half) / within))

}


fn_bulk_local_tuner_stationarity <-  function( state ) {

        if (state$n_chains < 2) {
                return(list(passed = FALSE,
                            reason = "fewer_than_two_chains",
                            max_rank_folded_split_rhat = NA_real_,
                            parameter_rhat = list(),
                            screen_parameters = state$parameter_names,
                            nuisance_not_screened = TRUE,
                            interpretation = "A finite-window stationarity screen, not proof of stationarity."))
        }
        values <-  lapply(seq_along(state$parameter_names), function(index) {
                raw <-  t(matrix(state$draws[index, , ], nrow = state$n_chains))
                ranked <-  matrix(fn_bulk_local_tuner_rank(as.vector(raw)), nrow = nrow(raw))
                folded <-  abs(raw - stats::median(raw))
                folded <-  matrix(fn_bulk_local_tuner_rank(as.vector(folded)), nrow = nrow(raw))
                return(c(rank = fn_bulk_local_tuner_rhat(ranked),
                         folded = fn_bulk_local_tuner_rhat(folded)))
        })
        names(values) <-  state$parameter_names
        maximum <-  max(unlist(values))
        return(list(passed = is.finite(maximum) && maximum <= state$settings$stationarity_rhat_max,
                    reason = if (is.finite(maximum) && maximum <= state$settings$stationarity_rhat_max) {
                            "main_parameter_screen_passed"
                    } else "main_parameter_screen_failed",
                    max_rank_folded_split_rhat = maximum,
                    parameter_rhat = values,
                    screen_parameters = state$parameter_names,
                    nuisance_not_screened = TRUE,
                    interpretation = "A finite-window stationarity screen, not proof of stationarity."))

}


fn_bulk_local_tuner_observables <-  function( state ) {

        n_draws <-  dim(state$draws)[3]
        output <-  array(0, dim = c(length(state$selected_indices), state$n_chains, n_draws))
        for (position in seq_along(state$selected_indices)) {
                raw <-  state$draws[state$selected_indices[position], , ]
                ranked <-  fn_bulk_local_tuner_rank(as.vector(raw))
                output[position, , ] <-  matrix(ranked - mean(ranked), nrow = state$n_chains)
        }
        return(output)

}


fn_bulk_local_tuner_support <-  function( weights, shared_jitter ) {

        weights <-  as.matrix(weights)
        path_weights <-  if (shared_jitter) weights[, 1] else as.vector(weights)
        time_weights <-  rowSums(weights)
        kish <-  function(value) {
                if (!length(value) || sum(value * value) <= 0) return(0)
                return(sum(value)^2 / sum(value * value))
        }
        return(c(effective_paths = kish(path_weights),
                 effective_time_clusters = kish(time_weights),
                 nonzero_time_clusters = sum(time_weights > 0)))

}


fn_bulk_local_tuner_score <-  function( state, observables, scale, deleted_draws = integer(0) ) {

        n_steps <-  state$buffer_count
        n_draws <-  n_steps + 1
        n_parameters <-  dim(observables)[1]
        max_lag <-  state$settings$max_lag
        probability_base <-  matrix(fn_bulk_local_tuner_length_probability(
                as.vector(state$leapfrog_steps), rep(state$nominal_tau, each = n_steps),
                rep(state$eps, each = n_steps)), nrow = n_steps)
        probability_candidate <-  matrix(fn_bulk_local_tuner_length_probability(
                as.vector(state$leapfrog_steps), rep(state$nominal_tau * scale, each = n_steps),
                rep(state$eps, each = n_steps)), nrow = n_steps)
        step_weights <-  probability_candidate / probability_base
        gamma <-  matrix(NA_real_, nrow = n_parameters, ncol = max_lag + 1)
        keep_draws <-  setdiff(seq_len(n_draws), deleted_draws)
        if (!length(keep_draws)) return(list(defined = FALSE, reason = "no_retained_draws"))
        for (position in seq_len(n_parameters)) {
                gamma[position, 1] <-  mean(observables[position, , keep_draws, drop = FALSE]^2)
        }
        supports <-  matrix(0, nrow = max_lag, ncol = 3,
                            dimnames = list(NULL, c("effective_paths", "effective_time_clusters",
                                                   "nonzero_time_clusters")))
        for (lag in seq_len(max_lag)) {
                starts <-  seq_len(n_draws - lag)
                if (length(deleted_draws)) {
                        keep <-  vapply(starts, function(start) {
                                return(!any(seq.int(start, start + lag) %in% deleted_draws))
                        }, logical(1))
                        starts <-  starts[keep]
                }
                if (!length(starts)) return(list(defined = FALSE, reason = "no_retained_lag_paths"))
                path_weights <-  matrix(1, nrow = length(starts), ncol = state$n_chains)
                for (offset in seq_len(lag)) {
                        path_weights <-  path_weights * step_weights[starts + offset - 1, , drop = FALSE]
                }
                supports[lag, ] <-  fn_bulk_local_tuner_support(path_weights, state$shared_jitter)
                for (position in seq_len(n_parameters)) {
                        initial <-  t(matrix(observables[position, , starts], nrow = state$n_chains))
                        final <-  t(matrix(observables[position, , starts + lag], nrow = state$n_chains))
                        ## The denominator counts all eligible paths, including zero-weight paths.
                        gamma[position, lag + 1] <-  sum(path_weights * initial * final) / length(path_weights)
                }
        }
        support <-  apply(supports, 2, min)
        if (any(!is.finite(gamma)) || any(gamma[, 1] <= 0)) {
                return(list(defined = FALSE, reason = "undefined_rank_covariance", support = support,
                            gamma = gamma, step_weights = step_weights, path_support = supports))
        }
        correlations <-  gamma / gamma[, 1]
        if (any(abs(correlations) > 1 + sqrt(.Machine$double.eps))) {
                return(list(defined = FALSE, reason = "undefined_rank_autocorrelation", support = support,
                            correlations = correlations, gamma = gamma,
                            step_weights = step_weights, path_support = supports))
        }
        sample_size <-  length(keep_draws) * state$n_chains
        ess <-  rep(NA_real_, n_parameters)
        for (position in seq_len(n_parameters)) {
                pairs <-  numeric(0)
                stopped <-  FALSE
                for (lag in seq.int(0, max_lag - 1, by = 2)) {
                        pair <-  correlations[position, lag + 1] + correlations[position, lag + 2]
                        if (pair <= 0) {
                                if (lag == 0) {
                                        return(list(defined = FALSE, reason = "invalid_spectral_estimate",
                                                    support = support, correlations = correlations,
                                                    gamma = gamma, step_weights = step_weights,
                                                    path_support = supports))
                                }
                                stopped <-  TRUE
                                break
                        }
                        if (length(pairs)) pair <-  min(pair, pairs[length(pairs)])
                        pairs <-  c(pairs, pair)
                }
                if (!stopped) {
                        return(list(defined = FALSE, reason = "positive_sequence_unresolved_at_max_lag",
                                    support = support, correlations = correlations, gamma = gamma,
                                    step_weights = step_weights, path_support = supports))
                }
                integrated <-  -1 + 2 * sum(pairs)
                if (!is.finite(integrated) || integrated <= 0) {
                        return(list(defined = FALSE, reason = "invalid_spectral_estimate", support = support,
                                    correlations = correlations, gamma = gamma,
                                    step_weights = step_weights, path_support = supports))
                }
                integrated <-  max(1 / log10(sample_size), integrated)
                ess[position] <-  sample_size / integrated
        }
        costs <-  fn_bulk_local_tuner_expected_length(state$nominal_tau * scale, state$eps) +
                  state$cost_contract$endpoint_evaluations_per_iteration +
                  state$cost_contract$start_of_call_evaluations_per_chain / state$cost_contract$n_iter_sampling
        names(ess) <-  state$parameter_names[state$selected_indices]
        return(list(defined = TRUE,
                    reason = "defined",
                    bulk_ess = min(ess),
                    parameter_bulk_ess = ess,
                    efficiency = min(ess) / (n_steps * sum(costs)),
                    expected_gradient_cost_per_iteration_per_chain = costs,
                    support = support,
                    correlations = correlations,
                    gamma = gamma,
                    step_weights = step_weights,
                    path_support = supports,
                    ess_numeric_cap = "S * log10(S)"))

}


fn_bulk_local_tuner_decide <-  function( state ) {

        method <-  list(estimator = "rank_normalised_multilag_path_importance_estimate",
                        native_finite_window_equivalence = FALSE,
                        native_finite_window_equivalent = FALSE,
                        uncertainty = "paired_overlap_aware_delete_time_block_jackknife_heuristic",
                        covariance_denominator = "all_eligible_paths",
                        tail_ess_optimised = FALSE,
                        geometry_establishes_stationarity = FALSE,
                        cost = "exact_discrete_expected_L_plus_endpoint_plus_first_call_amortisation")
        result <-  list(tau_selected = state$nominal_tau,
                        status = "retain_baseline",
                        reason = "insufficient_history",
                        candidates = list(),
                        support = NULL,
                        stationarity = NULL,
                        method = method,
                        buffer_count = state$buffer_count,
                        raw_buffer_bytes = as.numeric(object.size(state$draws)) +
                                           as.numeric(object.size(state$leapfrog_steps)),
                        raw_parameter_count = length(state$parameter_names),
                        selected_parameter_count = length(state$selected_indices),
                        realised_leapfrog_steps = state$leapfrog_steps)
        result$realised_gradient_cost_per_iteration_per_chain <-  if (state$buffer_count > 0) {
                colMeans(state$leapfrog_steps) + state$cost_contract$endpoint_evaluations_per_iteration +
                state$cost_contract$start_of_call_evaluations_per_chain / state$cost_contract$n_iter_sampling
        } else NULL
        if (!state$enabled) {
                result$reason <-  "disabled"
                return(result)
        }
        if (state$buffer_count < state$settings$min_history) return(result)
        if (state$shared_jitter && (length(unique(state$nominal_tau)) != 1 ||
                                   length(unique(state$eps)) != 1 ||
                                   any(apply(state$leapfrog_steps, 1, function(value) {
                                           return(length(unique(value)) != 1)
                                   })))) {
                result$reason <-  "unsupported_unequal_chain_law_for_shared_jitter"
                return(result)
        }
        result$stationarity <-  fn_bulk_local_tuner_stationarity(state)
        if (!result$stationarity$passed) {
                result$reason <-  if (state$n_chains < 2) "insufficient_stationarity_chains" else {
                        "stationarity_screen_failed"
                }
                return(result)
        }
        observables <-  fn_bulk_local_tuner_observables(state)
        baseline <-  fn_bulk_local_tuner_score(state, observables, 1)
        result$candidates[["1"]] <-  c(list(scale = 1, tau = state$nominal_tau), baseline)
        result$support <-  baseline$support
        if (!baseline$defined) {
                result$reason <-  paste0("baseline_", baseline$reason)
                return(result)
        }
        n_draws <-  state$buffer_count + 1
        n_blocks <-  floor(n_draws / (state$settings$max_lag + 1))
        if (n_blocks < 4) {
                result$reason <-  "insufficient_jackknife_blocks"
                return(result)
        }
        block_labels <-  pmin(n_blocks, ceiling(seq_len(n_draws) / floor(n_draws / n_blocks)))
        blocks <-  lapply(seq_len(n_blocks), function(block) which(block_labels == block))
        jack_baseline <-  lapply(blocks, function(block) fn_bulk_local_tuner_score(state, observables, 1, block))
        best_gain <-  log1p(state$settings$min_gain)
        result$reason <-  "no_supported_conservative_improvement"
        for (scale in state$settings$candidate_scales[state$settings$candidate_scales < 1]) {
                if (any(state$nominal_tau * scale < state$eps)) {
                        result$candidates[[as.character(scale)]] <-  list(
                                scale = scale, tau = state$nominal_tau * scale,
                                defined = FALSE, accepted = FALSE, reason = "candidate_below_step_size")
                        next
                }
                candidate <-  fn_bulk_local_tuner_score(state, observables, scale)
                candidate$scale <-  scale
                candidate$tau <-  state$nominal_tau * scale
                candidate$accepted <-  FALSE
                if (candidate$defined && (candidate$support["effective_paths"] <
                                          state$settings$min_effective_paths ||
                                          candidate$support["effective_time_clusters"] <
                                          state$settings$min_time_clusters)) {
                        candidate$reason <-  "insufficient_path_or_time_cluster_support"
                } else if (candidate$defined) {
                        jack_candidate <-  lapply(blocks, function(block) {
                                return(fn_bulk_local_tuner_score(state, observables, scale, block))
                        })
                        valid <-  vapply(seq_len(n_blocks), function(index) {
                                defined <-  isTRUE(jack_baseline[[index]]$defined) &&
                                            isTRUE(jack_candidate[[index]]$defined)
                                if (!defined) return(FALSE)
                                support <-  jack_candidate[[index]]$support
                                return(support["effective_paths"] >= state$settings$min_effective_paths &&
                                       support["effective_time_clusters"] >= state$settings$min_time_clusters)
                        }, logical(1))
                        if (!all(valid)) {
                                candidate$reason <-  "undefined_or_unsupported_delete_block_comparison"
                        } else {
                                log_ratios <-  vapply(seq_len(n_blocks), function(index) {
                                        return(log(jack_candidate[[index]]$efficiency /
                                                   jack_baseline[[index]]$efficiency))
                                }, numeric(1))
                                se <-  sqrt((n_blocks - 1) / n_blocks *
                                            sum((log_ratios - mean(log_ratios))^2))
                                candidate$log_ratio <-  log(candidate$efficiency / baseline$efficiency)
                                candidate$log_ratio_se <-  se
                                candidate$conservative_gain <-  candidate$log_ratio -
                                                                state$settings$uncertainty_multiplier * se
                                candidate$time_blocks <-  blocks
                                if (is.finite(candidate$conservative_gain) &&
                                    candidate$conservative_gain > best_gain) {
                                        best_gain <-  candidate$conservative_gain
                                        result$tau_selected <-  candidate$tau
                                        result$status <-  "select_shorter_tau"
                                        result$reason <-  "supported_conservative_bulk_efficiency_gain"
                                        candidate$accepted <-  TRUE
                                }
                        }
                }
                result$candidates[[as.character(scale)]] <-  candidate
        }
        for (key in names(result$candidates)) {
                result$candidates[[key]]$accepted <-  result$status == "select_shorter_tau" &&
                                                     identical(result$candidates[[key]]$tau, result$tau_selected)
        }
        return(result)

}

























