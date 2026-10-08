#### ===========================================================================================================
## R_fn_length_response_tau_criterion.R
##
## ---- burnin_algorithm = "L_ESSR_length_response" / "LQ_ESSR_length_response" (6 Oct 2026) -------------------
##
##      The soft minimum, over the monitored coordinates j, of the lag-one ESS bounds g(a) = (1 - a) / (1 + a) of
##      the linear statistics z_j ("L_ESSR_length_response"), or of the linear statistics and the centred squares
##      z_j^2 ("LQ_ESSR_length_response", the objective of "LQ_ESSR"), per unit trajectory length (with the cost
##      offset and exponent of the rate criteria), MAXIMISED OVER TAU DIRECTLY from the "length response" of the
##      jittered trajectories, instead of by stochastic gradient ascent ("LQ_ESSR"):
##
##      Every burn-in trajectory has its own jittered length t. Its acceptance-weighted squared jump of a
##      statistic, divided by twice the statistic's variance, is a draw of m_j(t) = 1 - rho_j(t), rho_j(t) the
##      autocorrelation of the statistic after a trajectory of length t. Pooled over the chains and the recent
##      updates in bins of log2(t) (a quarter of a doubling wide; every earlier update keeps the weight
##      forgetting_factor at each new update), these give the curve m_j(t) over the explored lengths, and with
##      t ~ U(0, 2 tau') the lag-one autocorrelation at ANY candidate tau' is
##        a_j(tau') = 1 - (1 / (2 tau')) * integral_0^{2 tau'} m_j(t) dt
##      (m_j piecewise linear through (0, 0) and the bin means, the last bin's mean carried to its longest
##      trajectory). The criterion is evaluated on a grid of tau' (1/16 of a doubling apart) up to the longest
##      explored length / 2 (about tau), and tau moves the fraction learning_rate_now (the linear LR -> LR^2
##      schedule of R_fn_update_tau_using_ADAM) of the distance in log(tau) towards its maximiser, or towards
##      twice the larger of tau and the longest candidate when the maximiser is within
##      upper_end_zone_in_doublings of the longest candidate (the criterion is then still rising at the longest
##      length that the data cover).
##
##      Why (phase 1 of the LQ_ESSR redesign, 6 Oct 2026, in
##      validation_staging/bulk-local-tuner-y0658d/lq_essr_redesign_2026_10_06): "LQ_ESSR" estimated
##      d log(criterion) / d log(tau) from one update's 4-chain product statistics of the slowest coordinates
##      divided by small smoothed levels; that gradient was heavy-tailed (max / median 22-122 vs ESJD 3-7), so
##      ADAM moved tau only at a few random spikes and tau ended at 0.74-1.19 x its handover value at b125, b250
##      and b500. No derivative is estimated here, there are no local basins within the explored
##      lengths (the global maximiser is found), and tau moves from any start.
##
##      Exactness: g(a) is the ESS fraction of a statistic whose lag-k autocorrelation is a^k: exact for a normal
##      mode of a Gaussian target with exact dynamics and a full momentum refresh, and an upper bound otherwise.
##

##
## ---- the state carried between updates: the bin sums (bins of log2(t / reference_length), a quarter of a
##      doubling wide, from 2^-10 to 2^6 times the reference length, plus a catch-all bin at each end), and the
##      moving averages of the start moments (decay 0.9, as "LQ_ESSR") for the statistics' variances:
##
fn_length_response_initialise_state <-  function( n_statistics,
                                                  reference_length,
                                                  bin_width_in_doublings   = 0.25,
                                                  lowest_bin_in_doublings  = -10,
                                                  highest_bin_in_doublings = 6) {

        bin_edges_in_doublings <-  seq( from = lowest_bin_in_doublings,
                                        to   = highest_bin_in_doublings,
                                        by   = bin_width_in_doublings)
        n_bins <-  length(bin_edges_in_doublings) + 1
        ##
        return(list(reference_length                      = reference_length,
                    bin_edges_in_doublings                = bin_edges_in_doublings,
                    sum_of_normalised_squared_jumps       = matrix(0, nrow = n_statistics, ncol = n_bins),
                    sum_of_chain_counts                   = numeric(n_bins),
                    sum_of_trajectory_lengths             = numeric(n_bins),
                    largest_trajectory_length_in_bin      = numeric(n_bins),
                    second_moment_moving_average          = NULL,
                    fourth_moment_moving_average          = NULL,
                    n_updates                             = 0))

}

##
## ---- add one update's trajectories to the state:
##        squared_jumps      = n_statistics x n_chains, the weighted squared jump of each statistic (zero for a
##                             chain that does not contribute: divergent, or non-finite statistics);
##        variances          = the variance of each statistic now;
##        trajectory_lengths = each chain's own trajectory length;
##        binned_chains      = the chains whose trajectory length is finite and positive (every one of them
##                             counts, with its zero jump if it does not contribute, as in the complete-minibatch
##                             convention of the other criteria).
##
fn_length_response_add_update <-  function( state,
                                            squared_jumps,
                                            variances,
                                            trajectory_lengths,
                                            binned_chains,
                                            forgetting_factor = 0.95) {

        if (!any(binned_chains)) return(state)
        ##
        state$sum_of_normalised_squared_jumps <-  forgetting_factor * state$sum_of_normalised_squared_jumps
        state$sum_of_chain_counts <-  forgetting_factor * state$sum_of_chain_counts
        state$sum_of_trajectory_lengths <-  forgetting_factor * state$sum_of_trajectory_lengths
        ##
        bin_index <-  findInterval(log2(trajectory_lengths / state$reference_length),
                                   state$bin_edges_in_doublings) + 1
        normalised_squared_jumps <-  squared_jumps / (2 * variances)
        ##
        for (chain_index in which(binned_chains)) {

                bin <-  bin_index[chain_index]
                state$sum_of_normalised_squared_jumps[, bin] <-  state$sum_of_normalised_squared_jumps[, bin] +
                                                                 normalised_squared_jumps[, chain_index]
                state$sum_of_chain_counts[bin] <-  state$sum_of_chain_counts[bin] + 1
                state$sum_of_trajectory_lengths[bin] <-  state$sum_of_trajectory_lengths[bin] +
                                                         trajectory_lengths[chain_index]
                state$largest_trajectory_length_in_bin[bin] <-  max(state$largest_trajectory_length_in_bin[bin],
                                                                    trajectory_lengths[chain_index])

        }
        state$n_updates <-  state$n_updates + 1
        ##
        return(state)

}

##
## ---- integral_0^upper of the piecewise-linear interpolation through (0, 0), (x_1, y_1), ..., (x_K, y_K), as
##      weights on y_1..y_K (upper <= x_K); on a segment [left, right] covered up to segment_end, the weight of
##      y_left is covered - covered^2 / (2 width) and that of y_right is covered^2 / (2 width), covered =
##      segment_end - left, width = right - left:
##
fn_length_response_piecewise_linear_integral_weights <-  function( node_positions,
                                                                   upper_limits) {

        nodes <-  c(0, node_positions)
        weights <-  matrix(0, nrow = length(nodes), ncol = length(upper_limits))
        ##
        for (limit_index in seq_along(upper_limits)) {

                upper <-  upper_limits[limit_index]
                for (segment in seq_len(length(nodes) - 1)) {

                        left <-  nodes[segment]
                        right <-  nodes[segment + 1]
                        if (upper <= left) break
                        width <-  right - left
                        covered <-  min(right, upper) - left
                        weights[segment, limit_index] <-  weights[segment, limit_index] +
                                                          covered - covered^2 / (2 * width)
                        weights[segment + 1, limit_index] <-  weights[segment + 1, limit_index] +
                                                              covered^2 / (2 * width)

                }

        }
        ##
        return(weights[-1, , drop = FALSE])   ## the node at 0 has y = 0

}

##
## ---- the criterion on the grid of candidate tau' and its maximiser (NULL while fewer than 3 bins hold at least
##      minimum_bin_count forgotten chain counts):
##
fn_length_response_evaluate <-  function( state,
                                          tau_cost_offset             = 0,
                                          tau_cost_exponent           = 1,
                                          soft_minimum_power          = 8,
                                          minimum_bin_count           = 2,
                                          grid_step_in_doublings      = 1 / 16,
                                          upper_end_zone_in_doublings = 1 / 8) {

        usable_bins <-  which(state$sum_of_chain_counts >= minimum_bin_count)
        if (length(usable_bins) < 3) return(NULL)
        ##
        bin_centres <-  state$sum_of_trajectory_lengths[usable_bins] / state$sum_of_chain_counts[usable_bins]
        bin_means <-  state$sum_of_normalised_squared_jumps[, usable_bins, drop = FALSE] /
                      rep(state$sum_of_chain_counts[usable_bins],
                          each = nrow(state$sum_of_normalised_squared_jumps))
        ##
        ## the curve is known up to the longest trajectory of the last usable bin, not only up to that bin's
        ## centre: its mean is carried as a constant from the centre to that length, so that the candidates reach
        ## about tau:
        longest_length_with_data <-  state$largest_trajectory_length_in_bin[usable_bins[length(usable_bins)]]
        if (longest_length_with_data > bin_centres[length(bin_centres)]) {
                bin_centres <-  c(bin_centres, longest_length_with_data)
                bin_means <-  cbind(bin_means, bin_means[, ncol(bin_means)])
        }
        ##
        smallest_candidate <-  bin_centres[1]
        largest_candidate <-  bin_centres[length(bin_centres)] / 2
        if (!(largest_candidate > smallest_candidate)) return(NULL)
        candidate_tau <-  2^seq(log2(smallest_candidate), log2(largest_candidate), by = grid_step_in_doublings)
        if (utils::tail(candidate_tau, 1) < largest_candidate) {
                candidate_tau <-  c(candidate_tau, largest_candidate)
        }
        ##
        integral_weights <-  fn_length_response_piecewise_linear_integral_weights(bin_centres, 2 * candidate_tau)
        lag_one_autocorrelation <-  1 - sweep(bin_means %*% integral_weights, 2, 2 * candidate_tau, "/")
        lag_one_autocorrelation <-  pmin(pmax(lag_one_autocorrelation, -1 + 1e-6), 1 - 1e-6)
        scores <-  (1 - lag_one_autocorrelation) / (1 + lag_one_autocorrelation)
        ##
        ## the soft minimum (sum s^(-q))^(-1/q) per candidate, relative to the smallest score (no underflow):
        smallest_scores <-  apply(scores, 2, min)
        relative_inverse_scores <-  sweep(1 / scores, 2, 1 / smallest_scores, "/")
        soft_minimum <-  smallest_scores *
                         colSums(relative_inverse_scores^soft_minimum_power)^(-1 / soft_minimum_power)
        criterion <-  soft_minimum / (candidate_tau + tau_cost_offset)^tau_cost_exponent
        best <-  which.max(criterion)
        ##
        ## the scores near the longest candidate rest on the sparsest bins, and the soft minimum of noisier scores
        ## is lower, which pulls the maximiser just inside the longest candidate; a maximiser within
        ## upper_end_zone_in_doublings of it therefore counts as at the upper end:
        maximum_is_at_upper_end <-  candidate_tau[best] >= largest_candidate * 2^(-upper_end_zone_in_doublings)
        return(list(candidate_tau             = candidate_tau,
                    criterion                 = criterion,
                    tau_maximising_criterion  = candidate_tau[best],
                    criterion_maximum         = criterion[best],
                    maximum_is_at_upper_end   = maximum_is_at_upper_end,
                    largest_candidate         = largest_candidate))

}

##
## ---- the full update for fn_metric_tau_block_update(): criterion = the per-coordinate statistics of
##      fn_metric_position_criterion() (as for "LQ_ESSR"); state = the list carried between updates
##      (component_criterion_ema; NULL = none yet). Returns the list that fn_metric_tau_block_update() returns,
##      with "updated" = c(new tau, ADAM mean, ADAM variance) (the ADAM moments are not used and stay as passed),
##      "gradient" = log(target tau) - log(tau) (the direction and full distance of the move, for the records)
##      and "criterion" = the criterion at its maximiser:
##
fn_length_response_soft_minimum_ESS_bound_tau_update <-  function( criterion,
                                                                   include_centred_squared_statistics,
                                                                   state,
                                                                   tau,
                                                                   tau_values,
                                                                   probabilities,
                                                                   divergences,
                                                                   use_proposals,
                                                                   learning_rate,
                                                                   iteration,
                                                                   adaptation_iterations,
                                                                   learning_rate_schedule_iteration = NULL,
                                                                   learning_rate_schedule_length    = NULL,
                                                                   adam_mean,
                                                                   adam_variance,
                                                                   interest_rows                    = NULL,
                                                                   tau_cost_offset                  = 0) {

        required_fields <-  c("linear_jump_squared", "quadratic_jump_squared", "initial_second_moment",
                              "initial_fourth_moment")
        if (!is.list(criterion) || !all(required_fields %in% names(criterion))) {
                stop(paste0("The length-response criteria need the per-coordinate statistics of ",
                            "fn_metric_position_criterion()."))
        }
        ## interest_only: keep only the rows of the parameters of interest (the main rows lead every block):
        if (!is.null(interest_rows)) {
                if (any(interest_rows > nrow(criterion$linear_jump_squared))) {
                        stop("interest_rows exceed the rows of the adapted block.")
                }
                for (field in required_fields) {
                        criterion[[field]] <-  criterion[[field]][interest_rows, , drop = FALSE]
                }
        }
        tau_cost_exponent <-  if (is.null(criterion$tau_cost_exponent)) 1 else criterion$tau_cost_exponent
        n_chains <-  length(probabilities)
        ##
        ## ---- which chains contribute (as "LQ_ESSR"), and which are binned (every chain with a usable length):
        statistics_are_finite <-  apply(is.finite(criterion$linear_jump_squared) &
                                        is.finite(criterion$quadratic_jump_squared) &
                                        is.finite(criterion$initial_fourth_moment), 2, all)
        binned_chains <-  is.finite(tau_values) & tau_values > 0
        valid <-  statistics_are_finite & binned_chains & is.finite(divergences) & divergences == 0
        weights <-  if (isTRUE(use_proposals)) pmin(1, pmax(0, probabilities)) else rep(1, n_chains)
        weights[!is.finite(weights) | !valid] <-  0
        ##
        if (!any(binned_chains)) {
                updated <-  c(tau, adam_mean, adam_variance)
                attr(updated, "adam_update_performed") <-  FALSE
                return(list(updated = updated, adam_update_performed = FALSE, gradient = NA_real_,
                            criterion = NA_real_, criterion_ema = NA_real_, component_criterion_ema = state))
        }
        ##
        ## ---- the moving averages (decay 0.9) of the start moments over the valid chains:
        if (any(valid)) {
                second_moment_now <-  rowMeans(criterion$initial_second_moment[, valid, drop = FALSE])
                fourth_moment_now <-  rowMeans(criterion$initial_fourth_moment[, valid, drop = FALSE])
        } else {
                second_moment_now <-  fourth_moment_now <-  NULL
        }
        if (is.null(state)) {
                if (is.null(second_moment_now)) {
                        updated <-  c(tau, adam_mean, adam_variance)
                        attr(updated, "adam_update_performed") <-  FALSE
                        return(list(updated = updated, adam_update_performed = FALSE, gradient = NA_real_,
                                    criterion = NA_real_, criterion_ema = NA_real_,
                                    component_criterion_ema = NULL))
                }
                n_statistics <-  nrow(criterion$linear_jump_squared) *
                                 (if (isTRUE(include_centred_squared_statistics)) 2 else 1)
                state <-  fn_length_response_initialise_state( n_statistics     = n_statistics,
                                                               reference_length = mean(tau_values[binned_chains]))
        }
        if (!is.null(second_moment_now)) {
                state$second_moment_moving_average <-  if (is.null(state$second_moment_moving_average)) {
                        second_moment_now
                } else 0.9 * state$second_moment_moving_average + 0.1 * second_moment_now
                state$fourth_moment_moving_average <-  if (is.null(state$fourth_moment_moving_average)) {
                        fourth_moment_now
                } else 0.9 * state$fourth_moment_moving_average + 0.1 * fourth_moment_now
        }
        variance_linear <-  pmax(state$second_moment_moving_average, 1e-12)
        ##
        ## ---- the weighted squared jumps (zero for a chain that does not contribute):
        linear_jumps <-  criterion$linear_jump_squared
        linear_jumps[!is.finite(linear_jumps)] <-  0
        squared_jumps <-  sweep(linear_jumps, 2, weights, "*")
        variances <-  variance_linear
        if (isTRUE(include_centred_squared_statistics)) {
                ## (a non-positive or non-finite estimate falls back to the Gaussian value 2 Var(z_j)^2, as
                ##  "LQ_ESSR")
                variance_quadratic <-  state$fourth_moment_moving_average - state$second_moment_moving_average^2
                variance_quadratic <-  ifelse(is.finite(variance_quadratic) & variance_quadratic > 0,
                                              variance_quadratic,
                                              2 * variance_linear^2)
                quadratic_jumps <-  criterion$quadratic_jump_squared
                quadratic_jumps[!is.finite(quadratic_jumps)] <-  0
                squared_jumps <-  rbind(squared_jumps, sweep(quadratic_jumps, 2, weights, "*"))
                variances <-  c(variance_linear, variance_quadratic)
        }
        state <-  fn_length_response_add_update( state              = state,
                                                 squared_jumps      = squared_jumps,
                                                 variances          = variances,
                                                 trajectory_lengths = tau_values,
                                                 binned_chains      = binned_chains)
        ##
        ## ---- evaluate, and move tau:
        evaluation <-  fn_length_response_evaluate( state             = state,
                                                    tau_cost_offset   = tau_cost_offset,
                                                    tau_cost_exponent = tau_cost_exponent)
        if (is.null(evaluation)) {
                updated <-  c(tau, adam_mean, adam_variance)
                attr(updated, "adam_update_performed") <-  FALSE
                return(list(updated = updated, adam_update_performed = FALSE, gradient = NA_real_,
                            criterion = NA_real_, criterion_ema = NA_real_, component_criterion_ema = state))
        }
        schedule_iteration <-  if (is.null(learning_rate_schedule_iteration)) iteration else
                                                                              learning_rate_schedule_iteration
        schedule_length <-  if (is.null(learning_rate_schedule_length)) adaptation_iterations else
                                                                        learning_rate_schedule_length
        adaptation_progress <-  if (schedule_length <= 1) 0 else
                                min(1, max(0, (schedule_iteration - 1) / (schedule_length - 1)))
        learning_rate_now <-  learning_rate * (1 - (1 - learning_rate) * adaptation_progress)
        target_tau <-  if (isTRUE(evaluation$maximum_is_at_upper_end)) {
                2 * max(tau, evaluation$largest_candidate)
        } else evaluation$tau_maximising_criterion
        new_tau <-  exp(log(tau) + learning_rate_now * (log(target_tau) - log(tau)))
        ##
        updated <-  c(new_tau, adam_mean, adam_variance)
        attr(updated, "adam_update_performed") <-  TRUE
        return(list(updated                    = updated,
                    adam_update_performed      = TRUE,
                    gradient                   = log(target_tau) - log(tau),
                    criterion                  = evaluation$criterion_maximum,
                    criterion_ema              = evaluation$criterion_maximum,
                    component_criterion_ema    = state,
                    tau_maximising_criterion   = evaluation$tau_maximising_criterion,
                    maximum_is_at_upper_end    = evaluation$maximum_is_at_upper_end,
                    largest_candidate          = evaluation$largest_candidate,
                    learning_rate_now          = learning_rate_now))

}























