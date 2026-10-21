#### =============================================================================================================
## R_fn_sampling_eps_to_adapt_delta_from_acceptance_of_burnin_chains.R
##
## ---- The sampling eps from the acceptance probabilities which the burn-in chains recorded (no added iterations)
##
## The burn-in adapts ONE eps, shared by its chains, with ADAM on log(eps) towards a cross-chain mean acceptance
## probability of adapt_delta (eps_acceptance_mean; "harmonic" as in ChEES-HMC and SNAPER-HMC: Hoffman, Radul &
## Sountsov 2021, Algorithm 1; Sountsov & Hoffman 2022, eq. (15)). The harmonic mean of the chains' acceptance
## probabilities is never above their ordinary mean, so with the harmonic rule the ORDINARY mean acceptance
## probability at the eps handed over lies above adapt_delta; and the eps handed over is the last iterate of the
## adaptation.
##
## This sets the sampling eps so that the ordinary mean acceptance probability is adapt_delta, from what the
## burn-in chains have already recorded: at each iteration t of the last fraction of the burn-in, the eps which
## the iteration's transitions ran with (eps_t) and the ordinary mean over the burn-in chains of their acceptance
## probabilities (alpha_bar_t; a divergent proposal counts as 0, as in the sampling). The burn-in itself is
## unchanged, and no iteration is added, so there are no gradient evaluations and no time beyond the burn-in's.
##
## As in R_fn_calibrate_eps_on_sampling_chains_to_adapt_delta.R, the curve
##
##       alpha(eps) = 2 Phi(-c eps^2)
##
## has one unknown, c, which solves the pooled moment equation over these iterations
##
##       sum_t (alpha_bar_t - 2 Phi(-c eps_t^2)) = 0      (every iteration has the same number of burn-in chains),
##
## with each eps_t first expressed for the final metric: a metric update rescales eps and tau
## (NicoStan_metric_update_rescales_eps_and_tau), so eps_t is multiplied by the rescale factors of every later
## update. The sampling eps is the one with alpha(eps) = adapt_delta:
##
##       eps = sqrt(-qnorm(adapt_delta / 2) / c)      (capped at max_eps_main).
##
## If the moment equation has no root (e.g. every recorded acceptance probability is 1), the eps handed over by
## the burn-in is kept, and the result says so.
##
## The burn-in iterations used: those from first_burnin_iteration_used_for_sampling_eps to the last, which
## R_fn_sample.R sets to the iteration after the final metric update (metric_adaptation_end_iter + 1 of the
## burn-in schedule). From there the metric and the centre of the nuisance parameters are fixed and only eps and
## tau still adapt, so no metric update follows any iteration used and every rescale factor above is 1. The
## metric-update rescale of eps (1 / sqrt(c), with c from the main block's diagonal) does not keep the
## acceptance probability when the nuisance parameters dominate it (the joint sampler), so the eps of an
## iteration before the final metric update, expressed for the final metric through these factors, is placed at
## too large an eps for its acceptance probability, and the fitted c is too small. With
## first_burnin_iteration_used_for_sampling_eps = NULL, the last fraction_of_burnin_iterations_at_end_used of the
## burn-in is used instead (the window described above).
##
#' The sampling eps from the acceptance probabilities which the burn-in chains recorded
#'
#' @param eps_main_of_transition_during_burnin_vec The eps of each burn-in iteration's transitions (after the
#'   iteration's metric update, before its eps update).
#' @param p_jump_main_during_burnin_vec The ordinary mean over the burn-in chains of their acceptance
#'   probabilities, at each burn-in iteration.
#' @param metric_update_eps_tau_rescale_factor_vec The factor by which the metric update of each burn-in iteration
#'   rescaled eps and tau (NA = no rescale).
#' @param eps_main_handed_over_by_burnin The eps which the burn-in hands over.
#' @param n_chains_burnin Number of burn-in chains.
#' @param adapt_delta The target ordinary mean acceptance probability.
#' @param fraction_of_burnin_iterations_at_end_used The last fraction of the burn-in iterations which are used.
#'   (only when first_burnin_iteration_used_for_sampling_eps is NULL)
#' @param max_eps_main Upper bound of the step size.
#' @param first_burnin_iteration_used_for_sampling_eps The first burn-in iteration whose acceptance probabilities
#'   are used (every iteration from it to the last is used); R_fn_sample.R gives the iteration after the final
#'   metric update. NULL (default) uses the last fraction_of_burnin_iterations_at_end_used of the burn-in. A value
#'   given that is not one whole number from 1 to the number of burn-in iterations stops with an error.
#' @return A list: eps_main (the sampling eps), c_main (the fitted constant), eps_main_handed_over_by_burnin,
#'   predicted_mean_acceptance_at_eps_handed_over_by_burnin, first_burnin_iteration_used,
#'   n_burnin_iterations_used, n_transitions_used, mean_acceptance_of_iterations_used,
#'   fraction_of_burnin_iterations_at_end_used and kept_eps_handed_over_by_burnin (TRUE when the moment equation
#'   had no root).
#'   Also first_burnin_iteration_used_for_sampling_eps (the value given; NA for NULL) and
#'   burnin_iterations_used_for_sampling_eps ("from_first_burnin_iteration_given" or
#'   "last_fraction_of_burnin"); fraction_of_burnin_iterations_at_end_used is NA when the fraction is not used.
#' @noRd
fn_sampling_eps_to_adapt_delta_from_acceptance_of_burnin_chains <-  function(
                                                  eps_main_of_transition_during_burnin_vec,
                                                  p_jump_main_during_burnin_vec,
                                                  metric_update_eps_tau_rescale_factor_vec,
                                                  eps_main_handed_over_by_burnin,
                                                  n_chains_burnin,
                                                  adapt_delta,
                                                  fraction_of_burnin_iterations_at_end_used,
                                                  ## max_eps_main
                                                  max_eps_main,
                                                  first_burnin_iteration_used_for_sampling_eps = NULL
) {

        ##
        n_burnin_iterations <-  length(eps_main_of_transition_during_burnin_vec)
        if ((n_burnin_iterations == 0) ||
            (length(p_jump_main_during_burnin_vec) != n_burnin_iterations)) {
              stop(paste0("fn_sampling_eps_to_adapt_delta_from_acceptance_of_burnin_chains: the burn-in record ",
                          "has no eps of each iteration's transitions, or not one acceptance probability per ",
                          "iteration."))
        }
        ##
        ## ---- each iteration's eps expressed for the final metric (times the rescale factors of every later
        ##      metric update; no rescale = factor 1):
        rescale_factors <-  metric_update_eps_tau_rescale_factor_vec
        if (length(rescale_factors) != n_burnin_iterations) rescale_factors <-  rep(1, n_burnin_iterations)
        rescale_factors[!is.finite(rescale_factors)] <-  1
        log_rescale_factors <-  log(rescale_factors)
        ## (the sum of the log factors of the updates AFTER each iteration)
        log_rescale_of_later_updates <-  rev(cumsum(rev(log_rescale_factors))) - log_rescale_factors
        eps_for_final_metric <-  eps_main_of_transition_during_burnin_vec * exp(log_rescale_of_later_updates)
        ##
        ## ---- the iterations used: the last fraction of the burn-in, with a finite eps and acceptance
        ##      probability:
        ## n_burnin_iterations_at_end <-  max(1, floor(fraction_of_burnin_iterations_at_end_used *
        ##                                             n_burnin_iterations))
        ## first_burnin_iteration_used <-  n_burnin_iterations - n_burnin_iterations_at_end + 1
        ## ---- the iterations used: from first_burnin_iteration_used_for_sampling_eps to the last (the iterations
        ##      after the final metric update, as R_fn_sample.R gives it), or, when it is NULL, the last
        ##      fraction_of_burnin_iterations_at_end_used of the burn-in; with a finite eps and acceptance
        ##      probability:
        if (is.null(first_burnin_iteration_used_for_sampling_eps)) {
              burnin_iterations_used_for_sampling_eps <-  "last_fraction_of_burnin"
              n_burnin_iterations_at_end <-  max(1, floor(fraction_of_burnin_iterations_at_end_used *
                                                          n_burnin_iterations))
              first_burnin_iteration_used <-  n_burnin_iterations - n_burnin_iterations_at_end + 1
        } else {
              burnin_iterations_used_for_sampling_eps <-  "from_first_burnin_iteration_given"
              first_iteration_given <-  first_burnin_iteration_used_for_sampling_eps
              if ((length(first_iteration_given) != 1) ||
                  !is.numeric(first_iteration_given) ||
                  !is.finite(first_iteration_given) ||
                  (first_iteration_given != round(first_iteration_given)) ||
                  (first_iteration_given < 1) ||
                  (first_iteration_given > n_burnin_iterations)) {
                    stop(paste0("fn_sampling_eps_to_adapt_delta_from_acceptance_of_burnin_chains: ",
                                "first_burnin_iteration_used_for_sampling_eps must be one whole number from 1 ",
                                "to the number of burn-in iterations (", n_burnin_iterations, "), e.g. the ",
                                "iteration after the final metric update ",
                                "(burnin_schedule$metric_adaptation_end_iter + 1 of the burn-in record); given: ",
                                if (length(first_iteration_given) == 0) "nothing (length 0)" else
                                      paste(first_iteration_given, collapse = ", "), "."))
              }
              first_burnin_iteration_used <-  first_iteration_given
        }
        iterations_used <-  seq(from = first_burnin_iteration_used, to = n_burnin_iterations)
        iterations_used <-  iterations_used[is.finite(eps_for_final_metric[iterations_used]) &
                                            (eps_for_final_metric[iterations_used] > 0) &
                                            is.finite(p_jump_main_during_burnin_vec[iterations_used])]
        eps_used <-  eps_for_final_metric[iterations_used]
        mean_acceptance_used <-  p_jump_main_during_burnin_vec[iterations_used]
        ##
        ## ---- c from the pooled moment equation (increasing in c: at c -> 0 every 2 Phi(-c eps^2) -> 1, at
        ##      c -> Inf every one -> 0):
        fn_moment_equation <-  function(log_c) {

              return(sum(mean_acceptance_used - 2 * stats::pnorm(-exp(log_c) * eps_used^2)))
        }

        c_main <-  NA_real_
        if (length(iterations_used) > 0) {
              log_c_of_typical_eps <-  -2 * log(stats::median(eps_used))
              log_c_bounds <-  log_c_of_typical_eps + c(-30, 30)
              if ((fn_moment_equation(log_c_bounds[1]) < 0) && (fn_moment_equation(log_c_bounds[2]) > 0)) {
                    c_main <-  exp(stats::uniroot(f = fn_moment_equation, interval = log_c_bounds,
                                                  tol = 1e-10)$root)
              }
        }
        kept_eps_handed_over_by_burnin <-  !is.finite(c_main)
        eps_main_for_sampling <-  if (kept_eps_handed_over_by_burnin) eps_main_handed_over_by_burnin else
                                        min(max_eps_main, sqrt(-stats::qnorm(adapt_delta / 2) / c_main))
        predicted_mean_acceptance_at_eps_handed_over_by_burnin <-  if (kept_eps_handed_over_by_burnin) {
              NA_real_
        } else {
              2 * stats::pnorm(-c_main * eps_main_handed_over_by_burnin^2)
        }
        ##
        ## message(colourise(paste0("sampling eps from the acceptance probabilities of the burn-in chains ",
        ##                          "(iterations ",
        ##                          first_burnin_iteration_used, "-", n_burnin_iterations, ", ",
        ##                          length(iterations_used) * n_chains_burnin, " transitions, mean acceptance ",
        ##                          signif(mean(mean_acceptance_used), 3), "): eps ",
        ##                          signif(eps_main_handed_over_by_burnin, 4), " -> ",
        ##                          signif(eps_main_for_sampling, 4),
        ##                          if (kept_eps_handed_over_by_burnin) " (no root: eps handed over kept)" else
        ##                                paste0(" (predicted mean acceptance at the eps handed over ",
        ##                                      signif(predicted_mean_acceptance_at_eps_handed_over_by_burnin, 3),
        ##                                       ")")),
        ##                   "cyan"))
        ## (the same line, with the rule that chose the iterations used)
        message(colourise(paste0("sampling eps from the acceptance probabilities of the burn-in chains ",
                                 "(iterations ",
                                 first_burnin_iteration_used, "-", n_burnin_iterations,
                                 if (identical(burnin_iterations_used_for_sampling_eps,
                                               "from_first_burnin_iteration_given")) {
                                       ", from first_burnin_iteration_used_for_sampling_eps; "
                                 } else {
                                       paste0(", the last ", fraction_of_burnin_iterations_at_end_used,
                                              " of the burn-in; ")
                                 },
                                 length(iterations_used) * n_chains_burnin, " transitions, mean acceptance ",
                                 signif(mean(mean_acceptance_used), 3), "): eps ",
                                 signif(eps_main_handed_over_by_burnin, 4), " -> ",
                                 signif(eps_main_for_sampling, 4),
                                 if (kept_eps_handed_over_by_burnin) " (no root: eps handed over kept)" else
                                       paste0(" (predicted mean acceptance at the eps handed over ",
                                              signif(predicted_mean_acceptance_at_eps_handed_over_by_burnin, 3),
                                              ")")),
                          "cyan"))
        return(list( eps_main = eps_main_for_sampling,
                     c_main = c_main,
                     eps_main_handed_over_by_burnin = eps_main_handed_over_by_burnin,
                     predicted_mean_acceptance_at_eps_handed_over_by_burnin =
                           predicted_mean_acceptance_at_eps_handed_over_by_burnin,
                     first_burnin_iteration_used = first_burnin_iteration_used,
                     n_burnin_iterations_used = length(iterations_used),
                     n_transitions_used = length(iterations_used) * n_chains_burnin,
                     mean_acceptance_of_iterations_used = mean(mean_acceptance_used),
                     ## fraction_of_burnin_iterations_at_end_used = fraction_of_burnin_iterations_at_end_used,
                     ## (NA when the iterations used start at first_burnin_iteration_used_for_sampling_eps)
                     fraction_of_burnin_iterations_at_end_used =
                           if (identical(burnin_iterations_used_for_sampling_eps, "last_fraction_of_burnin"))
                                 fraction_of_burnin_iterations_at_end_used else NA_real_,
                     first_burnin_iteration_used_for_sampling_eps =
                           if (is.null(first_burnin_iteration_used_for_sampling_eps)) NA_real_ else
                                 first_burnin_iteration_used_for_sampling_eps,
                     burnin_iterations_used_for_sampling_eps = burnin_iterations_used_for_sampling_eps,
                     kept_eps_handed_over_by_burnin = kept_eps_handed_over_by_burnin))
##
}






















