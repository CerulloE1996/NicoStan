#### =====================================================================================================================================
## R_fn_metric_adaptive_shrinkage.R
##
## ---- Diagonal-target correlation shrinkage with streamed adaptation-noise estimates ---------------------------------------------------
##
## Correlation shrinkage follows the diagonal-target rule of Schafer and Strimmer (2005).
## Iteration-level batches estimate serial dependence without storing the draw stream. The pooled estimate uses complete
## K-chain iteration blocks; the per-iteration estimate uses the covariance proposals and the actual metric blend weights.
## Batch means during a changing warm-up kernel and the weighted proposal-noise correction are approximations.
##


#' Initialise streamed adaptive metric shrinkage
#'
#' @param n_main Number of main parameters.
#' @param n_chains Number of burn-in chains.
#' @param metric_estimator Either "pooled" or "per_iteration".
#' @param n_window_updates Number of observed iterations or metric updates in the adaptation window.
#' @param initial_covariance Initial unshrunk main covariance.
#' @return A main-block state with no retained draw stream.
#' @export
fn_init_adaptive_metric_shrinkage <-  function( n_main,
                                                 n_chains,
                                                 metric_estimator,
                                                 n_window_updates,
                                                 initial_covariance) {

        for (value in list(n_main, n_chains, n_window_updates)) {
              if (!is.numeric(value) || length(value) != 1 || !is.finite(value) || value < 1 || value != floor(value)) {
                    stop("Adaptive metric dimensions and window updates must be positive whole numbers.")
              }
        }
        if (!metric_estimator %in% c("pooled", "per_iteration")) {
              stop("Adaptive metric shrinkage supports pooled and per_iteration estimators.")
        }
        .fn_metric_shrinkage_check_covariance(initial_covariance, n_main)
        ##
        pairs <-  which(upper.tri(matrix(FALSE, n_main, n_main)), arr.ind = TRUE)
        feature_pairs <-  which(upper.tri(matrix(FALSE, 5, 5), diag = TRUE), arr.ind = TRUE)
        n_pairs <-  nrow(pairs)
        ##
        return(list( metric_estimator = metric_estimator,
                     n_main = n_main,
                     n_chains = n_chains,
                     n_window_updates = n_window_updates,
                     pair_indices = pairs,
                     feature_pair_indices = feature_pairs,
                     batch_length = max(1, ceiling(sqrt(n_window_updates))),
                     n_events = 0,
                     n_draws = 0,
                     n_batches = 0,
                     batch_count = 0,
                     weight_sum_squared = 0,
                     initial_covariance = initial_covariance,
                     unshrunk_covariance = initial_covariance,
                     reference_centre = rep(0, n_main),
                     pooled_mean = rep(0, n_main),
                     pooled_M2 = matrix(0, n_main, n_main),
                     feature_mean = matrix(0, n_pairs, 5),
                     feature_M2 = matrix(0, n_pairs, nrow(feature_pairs)),
                     batch_sum = matrix(0, n_pairs, 5),
                     batch_mean = matrix(0, n_pairs, 5),
                     batch_M2 = matrix(0, n_pairs, nrow(feature_pairs))))

}



.fn_metric_shrinkage_check_covariance <-  function( covariance_matrix,
                                                     n_main) {

        if (!is.matrix(covariance_matrix) || !is.numeric(covariance_matrix) ||
            !identical(dim(covariance_matrix), c(as.integer(n_main), as.integer(n_main))) ||
            any(!is.finite(covariance_matrix)) || any(diag(covariance_matrix) <= 0)) {
              stop("Adaptive shrinkage requires a finite square covariance with positive diagonal.")
        }
        if (!isTRUE(all.equal(covariance_matrix, t(covariance_matrix), tolerance = sqrt(.Machine$double.eps),
                             check.attributes = FALSE))) {
              stop("Adaptive shrinkage requires a symmetric covariance.")
        }
        return(invisible(TRUE))

}



.fn_metric_shrinkage_update_feature_moments <-  function( feature,
                                                           feature_mean,
                                                           feature_M2,
                                                           n_previous,
                                                           feature_pairs) {

        delta <-  feature - feature_mean
        n_new <-  n_previous + 1
        feature_mean <-  feature_mean + delta / n_new
        for (pair in seq_len(nrow(feature_pairs))) {
              feature_M2[, pair] <-  feature_M2[, pair] +
                  delta[, feature_pairs[pair, 1]] * delta[, feature_pairs[pair, 2]] * (n_previous / n_new)
        }
        return(list(mean = feature_mean, M2 = feature_M2))

}



#' Observe one main-draw block or one unshrunk covariance proposal
#'
#' @param state Adaptive shrinkage state.
#' @param main_draws Main-parameter by chain matrix for pooled observation; NULL for per_iteration.
#' @param covariance_proposal Unshrunk regularised covariance proposal for per_iteration; NULL for pooled.
#' @param ratio_M_effective Actual metric blend weight for per_iteration; NULL for pooled.
#' @return Updated adaptive shrinkage state.
#' @export
fn_update_adaptive_metric_shrinkage <-  function( state,
                                                   main_draws,
                                                   covariance_proposal,
                                                   ratio_M_effective) {

        pairs <-  state$pair_indices
        if (identical(state$metric_estimator, "pooled")) {
              if (!is.matrix(main_draws) || !is.numeric(main_draws) ||
                  !identical(dim(main_draws), c(as.integer(state$n_main), as.integer(state$n_chains))) ||
                  any(!is.finite(main_draws)) || !is.null(covariance_proposal) || !is.null(ratio_M_effective)) {
                    stop("Pooled adaptive shrinkage requires a finite main-parameter by chain matrix and NULL proposal/weight.")
              }
              ##
              block_mean <-  rowMeans(main_draws)
              centred_draws <-  main_draws - block_mean
              if (state$n_events == 0) state$reference_centre <-  block_mean
              feature_draws <-  main_draws - state$reference_centre
              feature_block_mean <-  rowMeans(feature_draws)
              raw_second_moment <-  tcrossprod(feature_draws) / state$n_chains
              delta <-  block_mean - state$pooled_mean
              n_new <-  state$n_draws + state$n_chains
              state$pooled_M2 <-  state$pooled_M2 + tcrossprod(centred_draws) +
                  tcrossprod(delta) * (state$n_draws * state$n_chains / n_new)
              state$pooled_mean <-  state$pooled_mean + delta * (state$n_chains / n_new)
              state$n_draws <-  n_new
              if (n_new > 1) state$unshrunk_covariance <-  state$pooled_M2 / (n_new - 1)
              ##
              raw_diagonal <-  diag(raw_second_moment)
              feature <-  cbind(feature_block_mean[pairs[, 1]], feature_block_mean[pairs[, 2]],
                                 raw_diagonal[pairs[, 1]], raw_diagonal[pairs[, 2]], raw_second_moment[pairs])
        } else {
              .fn_metric_shrinkage_check_covariance(covariance_proposal, state$n_main)
              if (!is.null(main_draws) || !is.numeric(ratio_M_effective) || length(ratio_M_effective) != 1 ||
                  !is.finite(ratio_M_effective) || ratio_M_effective < 0 || ratio_M_effective > 1) {
                    stop("Per-iteration adaptive shrinkage requires NULL draws and a metric weight in [0, 1].")
              }
              ##
              state$unshrunk_covariance <-  ratio_M_effective * covariance_proposal +
                  (1 - ratio_M_effective) * state$unshrunk_covariance
              state$weight_sum_squared <-  ratio_M_effective^2 +
                  (1 - ratio_M_effective)^2 * state$weight_sum_squared
              state$n_draws <-  state$n_draws + state$n_chains
              proposal_diagonal <-  diag(covariance_proposal)
              feature <-  cbind(rep(0, nrow(pairs)), rep(0, nrow(pairs)),
                                 proposal_diagonal[pairs[, 1]], proposal_diagonal[pairs[, 2]], covariance_proposal[pairs])
        }
        ##
        moments <-  .fn_metric_shrinkage_update_feature_moments(feature, state$feature_mean, state$feature_M2,
                                                                state$n_events, state$feature_pair_indices)
        state$feature_mean <-  moments$mean
        state$feature_M2 <-  moments$M2
        state$n_events <-  state$n_events + 1
        ##
        state$batch_sum <-  state$batch_sum + feature
        state$batch_count <-  state$batch_count + 1
        if (state$batch_count == state$batch_length) {
              batch_feature <-  state$batch_sum / state$batch_length
              batch_moments <-  .fn_metric_shrinkage_update_feature_moments(batch_feature, state$batch_mean,
                                                                           state$batch_M2, state$n_batches,
                                                                           state$feature_pair_indices)
              state$batch_mean <-  batch_moments$mean
              state$batch_M2 <-  batch_moments$M2
              state$n_batches <-  state$n_batches + 1
              state$batch_sum[] <-  0
              state$batch_count <-  0
        }
        return(state)

}



#' Reset shrinkage moments at a pooled metric-window boundary
#'
#' @param state Adaptive shrinkage state.
#' @param n_window_updates Length of the new adaptation window.
#' @return A fresh observer retaining dimensions and the current unshrunk covariance.
#' @export
fn_reset_adaptive_metric_shrinkage <-  function( state,
                                                  n_window_updates) {

        return(fn_init_adaptive_metric_shrinkage(n_main = state$n_main,
                                                  n_chains = state$n_chains,
                                                  metric_estimator = state$metric_estimator,
                                                  n_window_updates = n_window_updates,
                                                  initial_covariance = if (identical(state$metric_estimator, "pooled"))
                                                      state$initial_covariance else state$unshrunk_covariance))

}



.fn_metric_shrinkage_influence_variance <-  function( coefficients,
                                                       feature_M2,
                                                       n_observations,
                                                       feature_pairs) {

        variance <-  rep(0, nrow(coefficients))
        for (pair in seq_len(nrow(feature_pairs))) {
              first <-  feature_pairs[pair, 1]
              second <-  feature_pairs[pair, 2]
              variance <-  variance + coefficients[, first] * coefficients[, second] *
                  feature_M2[, pair] * if (first == second) 1 else 2
        }
        return(pmax(0, variance / (n_observations - 1)))

}



#' Resolve diagonal-target shrinkage from the streamed correlation-noise estimate
#'
#' @param state Adaptive shrinkage state.
#' @param covariance_matrix Covariance to shrink; the pooled ridge proposal or per_iteration unshrunk blended estimate.
#' @return Shrunk covariance, intensity and compact diagnostics. Variances are unchanged.
#' @export
fn_resolve_adaptive_metric_shrinkage <-  function( state,
                                                    covariance_matrix) {

        .fn_metric_shrinkage_check_covariance(covariance_matrix, state$n_main)
        ##
        lambda <-  1
        reason <-  "insufficient_complete_batches"
        noise_n_used <-  state$n_batches * state$batch_length
        correlation_signal <-  0
        correlation_noise <-  NA_real_
        raw_noise <-  NA_real_
        batch_noise <-  NA_real_
        valid_pairs <-  0
        noise_multiplier <-  NA_real_
        batch_noise_multiplier <-  NA_real_
        pairs <-  state$pair_indices
        ##
        if (nrow(pairs) == 0) {
              reason <-  "one_parameter"
        } else if (state$n_batches >= 2 && state$n_events >= 2) {
              raw_covariance <-  state$unshrunk_covariance
              variances <-  diag(raw_covariance)
              centre <-  if (identical(state$metric_estimator, "pooled")) state$pooled_mean - state$reference_centre else
                             rep(0, state$n_main)
              if (identical(state$metric_estimator, "pooled")) {
                    variances <-  variances * (state$n_draws - 1) / state$n_draws
                    raw_covariance <-  raw_covariance * (state$n_draws - 1) / state$n_draws
              }
              ##
              variance_first <-  variances[pairs[, 1]]
              variance_second <-  variances[pairs[, 2]]
              sd_product <-  sqrt(variance_first) * sqrt(variance_second)
              correlation <-  raw_covariance[pairs] / sd_product
              mean_first <-  centre[pairs[, 1]]
              mean_second <-  centre[pairs[, 2]]
              coefficients <-  cbind(-mean_second / sd_product + correlation * mean_first / variance_first,
                                       -mean_first / sd_product + correlation * mean_second / variance_second,
                                       -0.5 * correlation / variance_first,
                                       -0.5 * correlation / variance_second,
                                       1 / sd_product)
              valid <-  is.finite(variance_first) & variance_first > 0 & is.finite(variance_second) & variance_second > 0 &
                  is.finite(correlation) & rowSums(!is.finite(coefficients)) == 0
              ##
              raw_variance <-  .fn_metric_shrinkage_influence_variance(coefficients, state$feature_M2,
                                                                        state$n_events, state$feature_pair_indices)
              batch_variance <-  state$batch_length *
                  .fn_metric_shrinkage_influence_variance(coefficients, state$batch_M2,
                                                            state$n_batches, state$feature_pair_indices)
              valid <-  valid & is.finite(raw_variance) & is.finite(batch_variance)
              valid_pairs <-  sum(valid)
              ##
              noise_multiplier <-  if (identical(state$metric_estimator, "pooled")) {
                    1 / state$n_events
              } else {
                    state$weight_sum_squared
              }
              batch_noise_multiplier <-  if (identical(state$metric_estimator, "pooled")) {
                    1 / noise_n_used
              } else {
                    state$weight_sum_squared
              }
              if (valid_pairs > 0) {
                    correlation_signal <-  sum(correlation[valid]^2)
                    raw_noise <-  sum(raw_variance[valid]) * noise_multiplier
                    batch_noise <-  sum(batch_variance[valid]) * batch_noise_multiplier
                    correlation_noise <-  sum(pmax(raw_variance[valid] * noise_multiplier,
                                                   batch_variance[valid] * batch_noise_multiplier))
                    if (is.finite(correlation_signal) && correlation_signal > 0 && is.finite(correlation_noise)) {
                          lambda <-  min(1, max(0, correlation_noise / correlation_signal))
                          reason <-  "estimated"
                    } else {
                          reason <-  "zero_or_invalid_correlation_signal"
                    }
              } else {
                    reason <-  "no_valid_pairs"
              }
        }
        ##
        covariance_diagonal <-  diag(covariance_matrix)
        shrunk_covariance <-  (1 - lambda) * covariance_matrix
        diag(shrunk_covariance) <-  covariance_diagonal
        ##
        return(list( covariance_matrix = shrunk_covariance,
                     shrinkage = lambda,
                     diagnostics = list( method = "schafer_strimmer_diagonal_target",
                                         noise_method = if (identical(state$metric_estimator, "pooled"))
                                             "iteration_cluster_batch_means" else "weighted_covariance_proposal_batch_means_approximation",
                                         reason = reason,
                                         n_events = state$n_events,
                                         n_draws = state$n_draws,
                                         n_batches = state$n_batches,
                                         batch_length = state$batch_length,
                                         noise_n_used = noise_n_used,
                                         valid_pairs = valid_pairs,
                                         correlation_signal = correlation_signal,
                                         correlation_noise = correlation_noise,
                                         raw_noise = raw_noise,
                                         batch_noise = batch_noise,
                                         noise_multiplier = noise_multiplier,
                                         batch_noise_multiplier = batch_noise_multiplier,
                                         weight_sum_squared = state$weight_sum_squared,
                                         dependence_estimate = "max_raw_and_batch_long_run_variance",
                                         adaptation_noise_approximation = TRUE)))

}


























