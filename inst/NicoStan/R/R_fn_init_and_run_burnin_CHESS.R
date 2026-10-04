

## R_fn_init_and_run_burnin_CHESS.R

# Burnin - using R fn (which calls Rcpp fn @ each iter)  -------------------------------------------------------------------------------------


#' init_and_run_burnin_ChESSR
#' @param debug_burnin_timing Collect native component timings and per-chain integrator step counts; default FALSE.
#' @param diffusion_HMC_integrator Joint diffusion integrator: "kick_flow_kick" (default) or "flow_kick_flow".
## #' @param eps_acceptance_mean Cross-chain mean acceptance targeted by the step-size adaptation: "harmonic" (default) or "arithmetic".
#' @param eps_acceptance_mean Cross-chain mean acceptance targeted by the step-size adaptation: "harmonic" (default), "arithmetic"
#'   or "geometric".
#' @param tau_shrink_on_divergence,tau_shrink_factor,tau_shrink_min_divergent_chains Divergence-triggered tau shrink during
#'   adaptation; default FALSE (off). TRUE multiplies tau by tau_shrink_factor (default 0.95) at each adaptation iteration in
#'   which at least tau_shrink_min_divergent_chains (default 2) burn-in chains diverge, which is the rule used in earlier runs.
#' @param metric_pooled_window_resets Pooled metric estimator only: iterations r at which its Welford accumulators (main
#'   covariance and nuisance variances) are reset, at the start of iteration r + 1. "stan_style" (default) =
#'   round(c(0.30, 0.60) * n_adapt), the resets used in earlier runs; "none" = one window from metric_start_iter to
#'   metric_adaptation_end_iter; or a numeric vector of whole numbers >= 0.
#' @param metric_pooled_offdiagonal_shrinkage Pooled metric estimator only: s in [0, 1] (default 0, as in earlier runs); each
#'   pooled main covariance proposal becomes (1 - s) * cov + s * diag(diag(cov)). 0 = off-diagonals unshrunk.
#' @param time_criterion_sampling_quantities For burnin_algorithm = "CHESSR_time" / "SNAPER_time": the sampling-side quantities of the
#'   time-to-target-ESS criterion, as returned by fn_resolve_time_criterion_sampling_quantities() (see R_fn_time_criterion.R). NULL
#'   (default) with a "_time" criterion gives the plain CHESSR / SNAPER criterion, with a warning; ignored by the other criteria.
#' @param tau_gradient_estimator "forward" uses the proposed endpoint gradient; "two_ended" uses the time-reversal-averaged endpoint
#'   gradient. Required for internal callers.
#' @param tau_cost_exponent Exponent in [0, 1.5] applied to the trajectory-length cost. Required for internal callers.
#' @param esjd_jump_power Power in {2, 3, 4} of the metric jump used by ESJD. Required for internal callers.
#' @param tau_jitter_burnin Burn-in trajectory-length jitter: "uniform" or "halton". Required for internal callers.
#' @export
init_and_run_burnin_ChESSR   <- function(  debug,
                                           ##
                                           init_object,
                                           ##
                                           Model_args_as_Rcpp_List,
                                           ##
                                           Model_type,
                                           ##
                                           n_chains_burnin,
                                           ##
                                           theta_main_vectors_all_chains_input_from_R,
                                           theta_nuisance_vectors_all_chains_input_from_R,
                                           ##
                                           parallel_method,
                                           ##
                                           Stan_data_list,
                                           model_args_list,
                                           ##
                                           sample_nuisance,
                                           n_nuisance_override,
                                           ##
                                           seed,
                                           n_burnin,
                                           n_adapt,
                                           gap,
                                           ##
                                           adapt_delta,
                                           LR_main,
                                           LR_us,
                                           ##
                                           learning_rate_initial = NULL,
                                           learning_rate_initial_iter = NULL,
                                           tau_mult,
                                           tau_initial,
                                           ## tau_initial: one positive number (e.g. pi, 2*pi), or "adaptive". With "adaptive" the ramp before the
                                           ## handover targets pi (exactly as tau_initial = pi), and the handover sets tau = (pi/2) * sqrt(lambda_max),
                                           ## lambda_max being the largest eigenvalue of the burn-in draws' covariance in metric coordinates over
                                           ## [clip_iter, gap]: main block from the pooled (Welford) covariance, nuisance block from the pooled
                                           ## within-individual (n_tests x n_tests) second moment (see the handover block in the burn-in loop).
                                           ##
                                           manual_tau,
                                           tau_if_manual,
                                           ##
                                           ## When TRUE, tau_if_manual is read as a number of LEAPFROG STEPS rather than as an
                                           ## integration time, and tau is re-derived as L * eps at every iteration from the
                                           ## CURRENTLY adapted step size. That pins the trajectory length while leaving the
                                           ## step size free to keep dual-averaging to adapt_delta, which is what a
                                           ## "sweep L, measure ESS per gradient" experiment needs; passing a raw tau instead
                                           ## lets L drift as eps adapts, so the swept quantity is not the one being held fixed.
                                           ## Note tau_ii ~ Uniform(0, 2*tau), so this fixes the MEAN number of leapfrog steps.
                                           tau_if_manual_in_L_units = FALSE,
                                           ##
                                           ## Weight each chain's gradient by its Metropolis acceptance probability rather than
                                           ## by the realised accept indicator. Both options now aggregate by a mean.
                                           ## Proposal weighting never also applies the accept indicator.
                                           tau_weight_by_p_jump = NULL,
                                           ##
                                           ## Burn-in tau ramp over [clip_iter, gap), before the tau adaptation takes over:
                                           ##   "original" - 5*eps, then tau_initial/4, tau_initial/2, tau_initial.
                                           ##   "staged"   - 5, 10, 20 steps, then tau_initial times (1/8, 1/4, 1/2, 1).
                                           ## Both use equal slices of the actual window, cap their step-count floor at
                                           ## tau_initial, and hand over once. Neither is a bit-identical historical replay.
                                           tau_ramp = c("original", "staged"),
                                           ##
                                           ## eps at iteration gap (the handover to the tau adaptation):
                                           ##   TRUE  - re-initialise eps with find_initial_eps (halves from 0.5 until ONE leapfrog step
                                           ##           from the chain mean accepts at >= 0.8, so it returns 0.0625, 0.125, ...) and reset
                                           ##           the eps ADAM moments. The long-standing behaviour.
                                           ##   FALSE - keep the eps (and its ADAM moments) adapted up to gap. tau is reset either way.
                                           eps_reinit_at_ChEES_handover = TRUE,
                                           ##
                                           ## Cross-chain mean of the per-chain acceptance probabilities that the eps (step-size) ADAM update
                                           ## targets to adapt_delta (see R_fn_harmonic_mean_acceptance.R):
                                           ##   "harmonic"   - K / sum_k (1 / alpha_k), as in ChEES-HMC (Hoffman, Radul & Sountsov 2021, Algorithm 1)
                                           ##                  and SNAPER-HMC (Sountsov & Hoffman 2022, eq. (15)). A chain with zero acceptance
                                           ##                  (a divergent proposal, which includes |log ratio| > 1000 in either direction, or an
                                           ##                  exp(log ratio) that underflowed to 0) makes it exactly 0.
                                           ##   "arithmetic" - mean(alpha_k, na.rm = TRUE), the rule used before this option existed.
                                           ##   "geometric"  - exp(mean(log(alpha_k))), each alpha_k clamped to [1e-8, 1] first; between the
                                           ##                  harmonic and the arithmetic mean.
                                           ## eps_acceptance_mean = c("harmonic", "arithmetic"),
                                           eps_acceptance_mean = c("harmonic", "arithmetic", "geometric"),
                                           ##
                                           ## Divergence-triggered tau shrink during adaptation (a heuristic; not part of ChEES, ChEES-R or SNAPER):
                                           ##   TRUE  - at the end of every iteration ii with clip_iter < ii < n_adapt (adaptive tau only), if at least
                                           ##           tau_shrink_min_divergent_chains burn-in chains diverged in the main block, tau_main and tau_us
                                           ##           are multiplied by tau_shrink_factor before the trajectory-criterion update. With
                                           ##           tau_shrink_factor = 0.95 and tau_shrink_min_divergent_chains = 2 this is the rule used in
                                           ##           all earlier runs.
                                           ##   FALSE - (default) tau changes only through the ramp, the handover, the criterion update and the
                                           ##           max_tau caps.
                                           ## The iterations at which the shrink fired are returned as tau_shrink_fired_vec.
                                           tau_shrink_on_divergence        = FALSE,
                                           tau_shrink_factor               = 0.95,
                                           tau_shrink_min_divergent_chains = 2,
                                           ##
                                           ## burnin_algorithm = "CHESSR_time" / "SNAPER_time" only: the sampling-side quantities of the time-to-target-ESS
                                           ## criterion (time_per_leapfrog_step_sampling, sampling_overhead_in_leapfrog_steps, n_iter_sampling_for_time_criterion,
                                           ## their sources, and an optional user-supplied time_per_leapfrog_step_burnin), from
                                           ## fn_resolve_time_criterion_sampling_quantities() (R_fn_time_criterion.R). time_per_leapfrog_step_burnin is
                                           ## measured below, during this burn-in.
                                           time_criterion_sampling_quantities = NULL,
                                           bulk_local_tuner = NULL,
                                           bulk_local_tuner_names = NULL,
                                           bulk_local_tuner_cost_contract = NULL,
                                           ##
                                           ## Centre (theta_hat_us) of the exact Gaussian rotation for the nuisance block:
                                           ##   "running_mean"        - re-set EVERY iteration to the running mean of the chains' current
                                           ##                           nuisance states. The centre then follows the chains, which inflates
                                           ##                           the burn-in acceptance (same state/eps/metric: ~0.7 vs ~0.45 with the
                                           ##                           centre frozen, as it is in sampling), so eps is tuned too big for
                                           ##                           sampling. The earlier behaviour (default here, so a call
                                           ##                           that does not pass it - e.g. the pre-burnin - is unchanged).
                                           ##   "running_mean_frozen" - the same running mean, but FROZEN from theta_hat_us_freeze_iter on,
                                           ##                           so eps finishes adapting against exactly the kernel sampling uses.
                                           ##   "zero"                - centre fixed at 0 throughout (the Feb 2026 code).
                                           theta_hat_us_rule = c("running_mean", "running_mean_frozen", "zero"),
                                           theta_hat_us_freeze_iter = NULL,   ## automatic: linked to the final metric update; legacy: round(0.6 * n_adapt)
                                           burnin_schedule = "legacy",       ## preserve ordering pre-burn-in and direct historical callers
                                           metric_adaptation_end_iter = NULL,
                                           ##
                                           ## Non-adapting iterations to run after n_adapt. NULL = all of them, up to n_burnin (the behaviour
                                           ## without this option). eps, the metric and tau are frozen after n_adapt, so these iterations only let
                                           ## the few burn-in chains settle - which the many sampling chains do far more cheaply. e.g. 5.
                                           ## Historical early-stop description above is retired: the loop now always runs to n_burnin.
                                           ##
                                           ## TRUE = one trajectory length tau_ii per iteration, shared by ALL burn-in chains (the C++ burn-in
                                           ## worker draws it from a separate RNG; each chain's momentum stays its own). Iterations then stop
                                           ## waiting for whichever chain drew the longest trajectory. FALSE = per-chain tau_ii (as before).
                                           share_tau_ii_across_chains_in_burnin = FALSE,
                                           randomize_tau_burnin,
                                           tau_gradient_estimator,
                                           tau_cost_exponent,
                                           esjd_jump_power,
                                           ## burnin_algorithm = "LQ_ESSR" only: the main-block rows (indices) the criterion monitors, from
                                           ## interest_only in R_fn_sample_model; NULL = every row of the adapted block:
                                           interest_rows = NULL,
                                           tau_jitter_burnin,
                                           ##
                                           ## ---- tau_adaptation_block ("main" | "joint"; EXPERIMENTAL):
                                           ##      which parameters feed the trajectory-length criterion. "main" = the main block only (the
                                           ##      behaviour before this option existed). "joint" = main and nuisance blocks concatenated,
                                           ##      still adapting the ONE joint tau_main. Models without a sampled nuisance block fall back
                                           ##      to "main"; the effective value is reported as model_results$tau_adaptation_block.
                                           ##
                                           tau_adaptation_block = "main",
                                           ##
                                           ## Sole trajectory-length selector. Canonical values are
                                           ## "KE", "ChEES", "CHESSR", "CHESSR_log" and "SNAPER".
                                           ## "CHESS"/"ChEES" select ChEES; "CHESSR"/"ChEESR" select the ChEES-rate criterion (different criteria).
                                           ## "CHESSR_time" / "SNAPER_time": the time-to-target-ESS variants of CHESSR / SNAPER (R_fn_time_criterion.R).
                                           ## "ESJD": expected squared jumped distance per unit trajectory length (Pasarica and Gelman, 2010), the
                                           ## CHESSR estimator with the statistic ||z_t - z_0||^2; "ESJD_CHESSR": the geometric mean of the ESJD and
                                           ## CHESSR criteria, equal weights (both in R_fn_metric_trajectory_adaptation.R).
                                           ## "ESJD_SNAPER": the experimental equal-weight geometric mean of the ESJD and SNAPER per-trajectory rates; its
                                           ## learned SNAPER direction is updated and transported wherever this criterion is used.
                                           burnin_algorithm,
                                           diffusion_HMC,
                                           partitioned_HMC,
                                           ##
                                           clip_iter,
                                           clip_iter_tau,
                                           ##
                                           n_refresh,
                                           use_proposed,
                                           ##
                                           beta1_adam,
                                           beta2_adam,
                                           eps_adam,
                                           ##
                                           force_autodiff,
                                           force_PartialLog,
                                           multi_attempts,
                                           ##
                                           force_autodiff_for_metric,
                                           force_PartialLog_for_metric,
                                           force_multi_attempts_for_metric,
                                           ##
                                           ## (vect_type / Phi_type / inv_Phi_type were removed from this signature: they were
                                           ##  accepted but never used. The native models read them from model_args_list at initialisation;
                                           ##  Stan models do not use them. A caller still passing them now gets an "unused argument" error.)
                                           ##
                                           metric_type_main,
                                           metric_shape_main,
                                           ratio_M_main,
                                           interval_width_main,
                                           ##
                                           M_decay_type,
                                           M_decay_power,
                                           M_decay_scale,
                                           ##
                                           metric_type_nuisance,
                                           metric_shape_nuisance,
                                           ratio_M_nuisance,
                                           interval_width_nuisance,
                                           ##
                                           max_tau_main,
                                           max_tau_nuisance,
                                           ##
                                           max_eps_main,
                                           max_eps_nuisance,
                                           ##
                                           max_L,
                                           ##
                                           n_nuisance_to_track,
                                           ##
                                           n_threads_WCP,
                                           ##
                                           time_pre_burnin,
                                           ##
                                           metric_start_iter = NULL,  # first iteration whose draws feed the metric; NULL = max(clip_iter, 0.35*n_adapt)
                                           metric_estimator = "pooled",
                                           ##
                                           ## Pooled metric estimator only (metric_estimator = "pooled"; see R_fn_metric_pooled_settings.R). The
                                           ## defaults are the rule of all earlier runs:
                                           ##   metric_pooled_window_resets          - iterations r at which the pooled Welford accumulators (main
                                           ##                                          covariance and nuisance variances) are reset (at the start of
                                           ##                                          iteration r + 1):
                                           ##                                            "stan_style" - (default) round(c(0.30, 0.60) * n_adapt), the
                                           ##                                                           resets used in all earlier runs.
                                           ##                                            "none"       - one window, metric_start_iter to
                                           ##                                                           metric_adaptation_end_iter.
                                           ##                                            numeric      - whole numbers >= 0.
                                           ##   metric_pooled_offdiagonal_shrinkage  - s in [0, 1], applied once to each pooled main covariance proposal:
                                           ##                                          (1 - s) * cov + s * diag(diag(cov)). 0 (default) = all
                                           ##                                          off-diagonals unshrunk, as in all earlier runs.
                                           metric_pooled_window_resets          = "stan_style",
                                           metric_pooled_offdiagonal_shrinkage,
                                           ##
                                           ## NOTE: 'eps_init' and 'eps_initial' are DIFFERENT things.
                                           ## eps_init  = carry the step-size over from a previous stage
                                           ##             (used by the reorder pre-burnin); it replaces the
                                           ##             heuristic starting eps for the WHOLE run.
                                           ## eps_initial = a deliberately LARGE step-size held for the
                                           ##             first eps_initial_iter iterations only, after
                                           ##             which eps is handed back to the starting value
                                           ##             (heuristic or eps_init) and adapted as normal.
                                           eps_init = NULL,
                                           ##
                                           ## eps_carry_over = the final step-sizes of a previous stage (the reorder pre-burnin),
                                           ##             as list(eps_main = , eps_us = ). NULL (default) = the find_initial_eps
                                           ##             search runs at the start of this stage. Non-NULL = this
                                           ##             stage starts from these values and the search is NOT run at all (no
                                           ##             search cost; unlike eps_init, which still runs it and then overrides
                                           ##             its result). The joint sampler (partitioned_HMC = FALSE) keeps
                                           ##             eps_us = eps_main, as after the search. eps_initial still applies.
                                           eps_carry_over = NULL,
                                           ##
                                           eps_initial = NULL,
                                           eps_initial_iter = NULL,
                                           ##
                                           diffusion_HMC_integrator = "kick_flow_kick",
                                           debug_burnin_timing = FALSE
) {
  if (!is.logical(debug_burnin_timing) || length(debug_burnin_timing) != 1 || is.na(debug_burnin_timing)) {
      stop("debug_burnin_timing must be TRUE or FALSE.")
  }
  
  # Rprof("burnin_prof.out", interval = 0.005)
  
  ratio_M_us <- ratio_M_nuisance
  ##
  max_tau_us <- max_tau_nuisance
  max_eps_us <- max_eps_nuisance
  ##
  theta_us_vectors_all_chains_input_from_R <- theta_nuisance_vectors_all_chains_input_from_R 
  ##
  # Model_args_as_Rcpp_List <- init_object$Model_args_as_Rcpp_List
  n_nuisance <- init_object$n_nuisance
  n_params_main <- init_object$n_params_main
  n_params <- n_nuisance + n_params_main
  ##
  Model_args_as_Rcpp_List$n_nuisance <- n_nuisance
  ##
  index_nuisance <- seq_len(n_nuisance)
  index_us <- index_nuisance
  index_main <- (1 + n_nuisance):n_params
  
  y <- model_args_list$y ; y
  N <- model_args_list$N ; N
  n_class <- model_args_list$n_class ; n_class
  
  if (Model_type == "Stan") {
      N <- 1000
      latent_results <- rlogis(n = N, location =  0.0)
      y <- ifelse(latent_results > 0, 1, 0)
      y <- matrix(data = c(y), ncol = 1)
  } 
  
                                        
  
  {  # ---------------------------------------------------------------- list for EHMC params / EHMC struct in C++
    
    EHMC_Metric_as_Rcpp_List <- init_EHMC_Metric_as_Rcpp_List(   n_params_main = n_params_main, 
                                                                 n_nuisance = n_nuisance, 
                                                                 metric_shape_main = metric_shape_main)  
    
  }
  
  
  
  
  
  { # ----------------------------------------------------------------- list for EHMC params / EHMC struct in C++  #
    
    EHMC_args_as_Rcpp_List <- init_EHMC_args_as_Rcpp_List( diffusion_HMC = diffusion_HMC,
                                                       diffusion_HMC_integrator = diffusion_HMC_integrator)
    ##
    if (diffusion_HMC_integrator == "flow_kick_flow" &&
        (!isTRUE(x = diffusion_HMC) || !isFALSE(x = partitioned_HMC) || n_nuisance < 1L)) {
        stop("flow_kick_flow requires diffusion_HMC = TRUE, partitioned_HMC = FALSE and a nonempty nuisance block.")
    }

  }


  { # ----------------------------------------------------------------- trajectory-length adaptation options

    burnin_algorithm <- fn_normalise_burnin_algorithm(burnin_algorithm)
    fn_validate_trajectory_criterion_options(
        tau_gradient_estimator = tau_gradient_estimator,
        tau_cost_exponent = tau_cost_exponent,
        esjd_jump_power = esjd_jump_power,
        tau_jitter_burnin = tau_jitter_burnin,
        algorithm = burnin_algorithm,
        randomize_tau_burnin = randomize_tau_burnin)
    trajectory_criterion_advanced_active <- !identical(tau_gradient_estimator, "forward") ||
                                             !identical(as.numeric(tau_cost_exponent), 1) ||
                                             !identical(as.numeric(esjd_jump_power), 2) ||
                                             !identical(tau_jitter_burnin, "uniform")
    if (is.null(tau_weight_by_p_jump)) tau_weight_by_p_jump <- TRUE
    tau_ramp <- match.arg(arg = tau_ramp, choices = c("original", "staged"))
    ## eps_acceptance_mean <- match.arg(arg = eps_acceptance_mean, choices = c("harmonic", "arithmetic"))
    eps_acceptance_mean <- match.arg(arg = eps_acceptance_mean, choices = c("harmonic", "arithmetic", "geometric"))
    if (!is.logical(eps_reinit_at_ChEES_handover) || length(eps_reinit_at_ChEES_handover) != 1 || is.na(eps_reinit_at_ChEES_handover)) {
      stop("eps_reinit_at_ChEES_handover must be TRUE or FALSE.")
    }
    theta_hat_us_rule <- match.arg(arg = theta_hat_us_rule, choices = c("running_mean", "running_mean_frozen", "zero"))
    resolved_burnin_schedule <- fn_burnin_adaptation_schedule(
        n_burnin = n_burnin,
        n_adapt = n_adapt,
        burnin_schedule = burnin_schedule,
        metric_adaptation_end_iter = metric_adaptation_end_iter,
        theta_hat_us_freeze_iter = theta_hat_us_freeze_iter,
        theta_hat_us_rule = theta_hat_us_rule,
        clip_iter = clip_iter,
        clip_iter_tau = clip_iter_tau)
    n_adapt <- resolved_burnin_schedule$n_adapt
    metric_adaptation_end_iter <- resolved_burnin_schedule$metric_adaptation_end_iter
    theta_hat_us_freeze_iter <- resolved_burnin_schedule$theta_hat_us_freeze_iter
    bulk_local_tuner_state <-  NULL
    bulk_local_tuner_metadata <-  NULL
    bulk_local_tuner_freeze_active <-  FALSE
    bulk_local_tuner_collect_start <-  Inf
    bulk_local_tuner_decision_iter <-  Inf
    bulk_local_tuner_frozen_kernel <-  NULL
    if (identical(burnin_schedule, "automatic")) {
        cat("\nMain burn-in schedule (version ", resolved_burnin_schedule$schedule_version, "):\n", sep = "")
        print(x = resolved_burnin_schedule$phase_table, row.names = FALSE)
    }
    ##
    ## Metric estimator. "chain_mean" and "chain_mean_scaled" both estimate the variance of the
    ## CROSS-CHAIN MEAN of the burn-in chains, which is ~ Sigma / n_chains_burnin when the chains are
    ## independent; "chain_mean_scaled" multiplies that estimate by n_chains_burnin before it enters
    ## the metric, putting it back on the posterior (Sigma) scale. "pooled" estimates Sigma directly.
    ## Any other value used to leave the metric silently un-updated (metric_ready never TRUE).
    metric_estimator <- as.character(metric_estimator)   ## a factor (e.g. from expand.grid) would otherwise skip the x n_chains_burnin below
    ## "per_iteration" estimates Sigma from each iteration's burn-in draws alone: the cross-chain (co)variance of the
    ## n_chains_burnin draws of that iteration around their cross-chain mean (divisor n_chains_burnin - 1), with nothing
    ## accumulated across iterations; ratio_M (and M_decay) then blend these proposals into the metric, so early draws fade.
    ## if (length(metric_estimator) != 1 || !(metric_estimator %in% c("pooled", "chain_mean", "chain_mean_scaled"))) {
    if (length(metric_estimator) != 1 || !(metric_estimator %in% c("pooled", "chain_mean", "chain_mean_scaled", "per_iteration"))) {
        ## stop("metric_estimator must be 'pooled', 'chain_mean' or 'chain_mean_scaled'; got: ",
        stop("metric_estimator must be 'pooled', 'chain_mean', 'chain_mean_scaled' or 'per_iteration'; got: ",
             paste(as.character(metric_estimator), collapse = ", "))
    }
    metric_estimator <- as.character(metric_estimator)
    metric_variance_scale <- if (metric_estimator == "chain_mean_scaled") n_chains_burnin else 1
    ##
    ## ---- pooled metric estimator: window resets and off-diagonal shrinkage (read only when metric_estimator = "pooled"):
    ##
    metric_pooled_window_resets         <- fn_validate_metric_pooled_window_resets(metric_pooled_window_resets)
    metric_pooled_offdiagonal_shrinkage <- fn_validate_metric_pooled_offdiagonal_shrinkage(metric_pooled_offdiagonal_shrinkage)
    metric_pooled_offdiagonal_shrinkage_resolution <- fn_resolve_metric_pooled_offdiagonal_shrinkage(
        metric_pooled_offdiagonal_shrinkage = metric_pooled_offdiagonal_shrinkage,
        metric_estimator = metric_estimator,
        metric_type_main = metric_type_main,
        metric_shape_main = metric_shape_main)
    metric_pooled_offdiagonal_shrinkage_adaptive_active <- isTRUE(metric_pooled_offdiagonal_shrinkage_resolution$active)
    metric_pooled_offdiagonal_shrinkage_numeric <- if (identical(metric_pooled_offdiagonal_shrinkage, "adaptive")) 0 else
                                                   metric_pooled_offdiagonal_shrinkage
    ##
    if (!is.logical(tau_weight_by_p_jump) || length(tau_weight_by_p_jump) != 1 || is.na(tau_weight_by_p_jump)) {
        stop("tau_weight_by_p_jump must be a single TRUE or FALSE.")
    }
    ##
    if (!is.logical(tau_if_manual_in_L_units) || length(tau_if_manual_in_L_units) != 1 || is.na(tau_if_manual_in_L_units)) {
        stop("tau_if_manual_in_L_units must be a single TRUE or FALSE.")
    }
    ##
    ## ---- divergence-triggered tau shrink (tau_shrink_on_divergence):
    ##
    if (!is.logical(tau_shrink_on_divergence) || length(tau_shrink_on_divergence) != 1 || is.na(tau_shrink_on_divergence)) {
        stop("tau_shrink_on_divergence must be a single TRUE or FALSE.")
    }
    if (!is.numeric(tau_shrink_factor) || length(tau_shrink_factor) != 1 || !is.finite(tau_shrink_factor) ||
        tau_shrink_factor <= 0 || tau_shrink_factor > 1) {
        stop(paste0("tau_shrink_factor must be a single number in (0, 1]; got: ",
                    paste(as.character(tau_shrink_factor), collapse = ", ")))
    }
    if (!is.numeric(tau_shrink_min_divergent_chains) || length(tau_shrink_min_divergent_chains) != 1 ||
        !is.finite(tau_shrink_min_divergent_chains) || tau_shrink_min_divergent_chains < 1 ||
        tau_shrink_min_divergent_chains != round(tau_shrink_min_divergent_chains)) {
        stop(paste0("tau_shrink_min_divergent_chains must be a single whole number >= 1; got: ",
                    paste(as.character(tau_shrink_min_divergent_chains), collapse = ", ")))
    }
    ##
    message(colourise(paste0("divergence-triggered tau shrink = ", tau_shrink_on_divergence,
                             if (isTRUE(tau_shrink_on_divergence)) paste0(" (tau x ", tau_shrink_factor,
                                                                          " when >= ", tau_shrink_min_divergent_chains,
                                                                          " burn-in chains diverge, clip_iter < iteration < n_adapt)") else ""),
                      "cyan"))
    ##
    ## ---- CHESSR_time / SNAPER_time: the time-to-target-ESS criteria (R_fn_time_criterion.R). Active only with an adapted tau:
    ##
    time_criterion_active <-  fn_burnin_algorithm_is_time_criterion(burnin_algorithm) && !isTRUE(manual_tau)
    if (time_criterion_active) {
        if (isTRUE(partitioned_HMC)) {
            stop(paste0("burnin_algorithm = '", burnin_algorithm, "' is defined for the joint sampler (partitioned_HMC = FALSE), whose single tau ",
                        "sets the leapfrog steps of every gradient evaluation; use '", sub("_time$", "", burnin_algorithm), "' with partitioned_HMC = TRUE."))
        }
        if (is.null(time_criterion_sampling_quantities)) {
            warning(paste0("burnin_algorithm = '", burnin_algorithm, "' without time_criterion_sampling_quantities: ",
                           "sampling_overhead_in_leapfrog_steps = 0 and burnin_to_sampling_leapfrog_time_ratio = 0, i.e. the plain ",
                           sub("_time$", "", burnin_algorithm), " criterion."))
            time_criterion_sampling_quantities <-  suppressWarnings(
                fn_resolve_time_criterion_sampling_quantities( time_criterion_settings                = fn_default_time_criterion_settings(),
                                                               time_criterion_previous_run_quantities = NULL,
                                                               sampling_timing_probe_result           = NULL,
                                                               n_iter_planned                         = NA_real_))
        }
        ##
        ## ---- n_iter_burnin of the time model: the main burn-in iterations whose number of leapfrog steps is set by the adapted tau,
        ##      i.e. iterations clip_iter_tau + 1 to n_burnin (the update at iteration ii sets the tau of iteration ii + 1). The earlier
        ##      iterations (one leapfrog step before clip_iter, then the ramp to tau_initial up to the handover at clip_iter_tau) and the
        ##      pre-burnin (pre_burnin_L steps) take the same time whatever tau is chosen, so, like time_per_iter_overhead_burnin, they
        ##      drop out of the derivative of the time to the target ESS. The count is fixed for the whole adaptation:
        ##
        n_iter_burnin_at_adapted_tau <-  n_burnin - clip_iter_tau
        ##
        ## ---- with per-chain jitter of the burn-in trajectory lengths (randomize_tau_burnin = TRUE and tau_ii not shared across the
        ##      chains), each burn-in iteration waits for the longest of n_chains_burnin draws tau_ii ~ U(0, 2 tau), whose mean is about
        ##      2 * n_chains_burnin / (n_chains_burnin + 1) times tau, while the time model charges tau / eps leapfrog steps per burn-in
        ##      iteration; burnin_to_sampling_leapfrog_time_ratio is then too small by about that factor:
        ##
        if (isTRUE(randomize_tau_burnin) && !isTRUE(share_tau_ii_across_chains_in_burnin) && n_chains_burnin > 1) {
            warning(paste0("burnin_algorithm = '", burnin_algorithm, "' with randomize_tau_burnin = TRUE and share_tau_ii_across_chains_in_burnin = FALSE: ",
                           "each burn-in iteration waits for the longest of the ", n_chains_burnin, " jittered trajectories (on average about ",
                           signif(2 * n_chains_burnin / (n_chains_burnin + 1), 3), " times tau), which the time model does not charge, so ",
                           "burnin_to_sampling_leapfrog_time_ratio is too small by about that factor. share_tau_ii_across_chains_in_burnin = TRUE ",
                           "or randomize_tau_burnin = FALSE avoids this."))
        }
        message(colourise(paste0("time-to-target-ESS criterion (", burnin_algorithm, "): ",
                                 "sampling_overhead_in_leapfrog_steps = ", signif(time_criterion_sampling_quantities$sampling_overhead_in_leapfrog_steps, 4),
                                 " (", time_criterion_sampling_quantities$sampling_overhead_in_leapfrog_steps_source, ")",
                                 " | time_per_leapfrog_step_sampling = ", signif(time_criterion_sampling_quantities$time_per_leapfrog_step_sampling, 4), " s",
                                 " (", time_criterion_sampling_quantities$time_per_leapfrog_step_sampling_source, ")",
                                 " | n_iter_sampling_for_time_criterion = ", signif(time_criterion_sampling_quantities$n_iter_sampling_for_time_criterion, 5),
                                 " (", time_criterion_sampling_quantities$n_iter_sampling_for_time_criterion_source, ")",
                                 " | n_iter_burnin = n_burnin - clip_iter_tau = ", n_burnin, " - ", clip_iter_tau, " = ", n_iter_burnin_at_adapted_tau,
                                 " (the iterations run at the adapted tau)"),
                          "cyan"))
        ##
        ## ---- MALT's exponent (lag_one_autocorrelation_rho: a number in [0, 1] or "adaptive"; R_fn_time_criterion.R). Quantities from an
        ##      older resolver lack these elements and give the defaults (lag_one_autocorrelation_rho = 1; offset 8). The
        ##      sampling_overhead_in_leapfrog_steps of the time criterion (including "auto": a built-in default or the sampling timing
        ##      probe, for the sampling configuration) is resolved before this burn-in and is fixed for all of it:
        ##
        lag_one_autocorrelation_rho_setting <-  if (is.null(time_criterion_sampling_quantities$lag_one_autocorrelation_rho)) 1 else
                                                time_criterion_sampling_quantities$lag_one_autocorrelation_rho
        lag_one_autocorrelation_rho_is_adaptive <-  identical(lag_one_autocorrelation_rho_setting, "adaptive")
        lag_one_autocorrelation_rho_moment_averaging_offset <-  if (is.null(time_criterion_sampling_quantities$lag_one_autocorrelation_rho_moment_averaging_offset)) 8 else
                                                                time_criterion_sampling_quantities$lag_one_autocorrelation_rho_moment_averaging_offset
        ##
        ## ---- the console messages show lag_one_autocorrelation_rho only when it is not at its default 1, so that at the default the
        ##      messages are those of the criterion without it:
        ##
        time_criterion_messages_show_rho <-  !isTRUE(lag_one_autocorrelation_rho_setting == 1)
        if (time_criterion_messages_show_rho) {
            message(colourise(paste0("time-to-target-ESS criterion (", burnin_algorithm, "): lag_one_autocorrelation_rho = ", lag_one_autocorrelation_rho_setting,
                                     if (lag_one_autocorrelation_rho_is_adaptive) paste0(" (MALT's running moments, lag_one_autocorrelation_rho_moment_averaging_offset = ",
                                                                                          lag_one_autocorrelation_rho_moment_averaging_offset,
                                                                                          "; lag_one_autocorrelation_rho = 1 until the first moment update)") else "",
                                     " | penalty times (1 + lag_one_autocorrelation_rho) / 2"),
                              "cyan"))
        }
    }
    ##
    ## Position criteria use the mass metric of each adapted block. The joint
    ## sampler scores MAIN parameters only, as in the existing PS7 contract.
    ## Main parameters use ordinary HMC even when nuisance parameters use
    ## diffusion-pathspace dynamics. No nuisance score enters ChEES-R or SNAPER.
    ##
    if (isTRUE(tau_if_manual_in_L_units)) {
        if (!isTRUE(manual_tau)) {
            stop("tau_if_manual_in_L_units = TRUE requires manual_tau = TRUE.")
        }
        if (!is.numeric(tau_if_manual) || any(!is.finite(tau_if_manual)) || any(tau_if_manual < 1)) {
            stop("With tau_if_manual_in_L_units = TRUE, tau_if_manual is a leapfrog-step count and must be >= 1; got: ",
                 paste(as.character(tau_if_manual), collapse = ", "))
        }
    }
    ##
    message(paste0("metric estimator = ", metric_estimator,
                   if (metric_variance_scale != 1) paste0(" (x ", metric_variance_scale, ")") else "",
                   if (metric_estimator == "pooled") paste0(" (window resets = ", paste(metric_pooled_window_resets, collapse = "/"),
                                                            ", off-diagonal shrinkage = ", metric_pooled_offdiagonal_shrinkage,
                                                            if (identical(metric_pooled_offdiagonal_shrinkage, "adaptive") &&
                                                                !metric_pooled_offdiagonal_shrinkage_adaptive_active)
                                                                paste0("; not applied: ", metric_pooled_offdiagonal_shrinkage_resolution$reason) else "",
                                                            ")") else "",
                   " | tau adaptation: algorithm = ", burnin_algorithm,
                   ", ramp = ", tau_ramp,
                   ", eps re-init at handover = ", eps_reinit_at_ChEES_handover,
                   " | eps acceptance mean = ", eps_acceptance_mean,
                   " | nuisance centre: ", theta_hat_us_rule,
                   if (identical(theta_hat_us_rule, "running_mean_frozen")) paste0(" (frozen from iteration ", theta_hat_us_freeze_iter, ")") else "",
                   ", weight by p_jump = ", tau_weight_by_p_jump,
                   ", manual_tau = ", manual_tau,
                   if (isTRUE(manual_tau)) paste0(" (tau_if_manual = ", paste(tau_if_manual, collapse = "/"),
                                                  if (isTRUE(tau_if_manual_in_L_units)) " leapfrog steps)" else ")") else ""))

  }
  
  
  {  # ----------------------------------------------------------------- list for EHMC params / EHMC struct in C++
    
    EHMC_burnin_as_Rcpp_List <- init_EHMC_burnin_as_Rcpp_List( n_params_main = n_params_main,
                                                               n_nuisance = n_nuisance,
                                                               adapt_delta = adapt_delta,
                                                               LR_main = LR_main,
                                                               LR_us = LR_us)
    
  }
  
  # str(EHMC_Metric_as_Rcpp_List)
  # str(EHMC_args_as_Rcpp_List)
  # str(EHMC_burnin_as_Rcpp_List)
  
  
 # theta_main_vectors_all_chains_input_from_R <- init_object$theta_main_vectors_all_chains_input_from_R
#  theta_us_vectors_all_chains_input_from_R <-   init_object$theta_us_vectors_all_chains_input_from_R
  ##
  theta_main_vectors_all_chains_input_from_R
  theta_us_vectors_all_chains_input_from_R
  ##
  velocity_main_vectors_all_chains_input_from_R <- theta_main_vectors_all_chains_input_from_R
  velocity_us_vectors_all_chains_input_from_R   <- theta_us_vectors_all_chains_input_from_R
  
  ###  theta_vec_mean <- rowMeans(rbind(theta_us_vectors_all_chains_input_from_R, theta_main_vectors_all_chains_input_from_R))
  ##
  EHMC_args_as_Rcpp_List$diffusion_HMC <- diffusion_HMC
  ##
  tictoc::tic()

  debug <- FALSE ## BOOKMARK
  
  message("Printing from init_and_run_burnin_ChESSR:")

  # RcppParallel::setThreadOptions(numThreads = n_chains_burnin * n_threads_WCP_burnin);

  Model_type_R <- Model_type
  
  # ## Fixed SNAPER-HMC / ADAM constants: 
  # beta1_adam = 0.00
  # beta2_adam = 0.95
  # eps_adam = 1e-8
  ##
  kappa = 8.0
  eta_w <- 3
  
  if (sample_nuisance == TRUE) { 
    n_params <- n_params_main + n_nuisance
    index_nuisance <- seq_len(n_nuisance)
    index_main <- (1 + n_nuisance):n_params
  } else { 
    n_params <- n_params_main  
    index_main <- 1:n_params
    if (Model_type == "Stan") {
      ## ---- Stan models with sample_nuisance = FALSE have NO nuisance block:
      ## n_nuisance stays 0 (the complete unconstrained vector is main). No dummy
      ## coordinate is introduced: the C++ gradient/sampler paths handle empty
      ## nuisance blocks, and BridgeStan must always see the full main vector.
      index_nuisance <- integer(0)
      n_nuisance <- 0
    } else {
      ## ---- built-in models: keep the established 1-coordinate dummy convention.
      index_nuisance <- 1
      n_nuisance <- 1
    }
  }

  Model_args_as_Rcpp_List$n_nuisance <- n_nuisance
  Model_args_as_Rcpp_List$n_params_main <- n_params_main
  ##
  ## ---- tau_adaptation_block: validate, and resolve the EFFECTIVE block (reported back as model_results$tau_adaptation_block):
  ##
  tau_adaptation_block <- if_null_then_set_to(tau_adaptation_block, "main")
  if (!is.character(tau_adaptation_block) || length(tau_adaptation_block) != 1 || !tau_adaptation_block %in% c("main", "joint")) {
      stop(paste0("tau_adaptation_block must be 'main' or 'joint'; got: ", paste(as.character(tau_adaptation_block), collapse = ", ")))
  }
  tau_adaptation_block_effective <- tau_adaptation_block
  if (identical(tau_adaptation_block, "joint") && !(isTRUE(sample_nuisance) && n_nuisance > 0)) {
      tau_adaptation_block_effective <- "main"
      message(paste0("\033[36mtau_adaptation_block = 'joint' requested but this model has no sampled nuisance block; using 'main'.\033[0m"))
  }
  if (identical(tau_adaptation_block_effective, "joint") && isTRUE(partitioned_HMC)) {
      stop("tau_adaptation_block = 'joint' needs the joint (diffusion) HMC sampler with one tau; it is not defined for partitioned_HMC = TRUE.")
  }
  message(paste0("\033[36mtau_adaptation_block: requested = ", tau_adaptation_block, " | effective = ", tau_adaptation_block_effective, "\033[0m"))

  if (Model_type != "Stan") {
   # n_tests <- ncol(y)
    Model_args_as_Rcpp_List$Model_args_bools[15, 1] <- FALSE # debug
  }
  
  
  n_chains <- n_chains_burnin
  
  print(paste("n_params_main = ", n_params_main))
  print(paste("n_nuisance = ", n_nuisance))
 
  if (sample_nuisance == TRUE) { 
     theta_vec_mean <-  rowMeans( rbind(theta_us_vectors_all_chains_input_from_R, theta_main_vectors_all_chains_input_from_R))
  } else { 
     #### theta_main_vectors_all_chains_input_from_R[,] <- 0 
     theta_vec_mean <-  rowMeans( rbind(theta_main_vectors_all_chains_input_from_R))
  }
  length(theta_vec_mean)
  

 #   theta_vec <- c(theta_us_vectors_all_chains_input_from_R[, 1], theta_main_vectors_all_chains_input_from_R[, 1]) # using inits from chain 1

  ## Historical option, now retired:
  ## burnin_post_adapt_iter: stop this many iterations after n_adapt instead of at n_burnin (NULL = run to n_burnin):
  ## n_burnin is the total main burn-in length; its non-adapting tail is determined by n_adapt, not a separate input.
  last_burnin_iteration <- n_burnin
  iter_seq_burnin <- seq(from = 1, to = last_burnin_iteration, by = 1)

  
  if (metric_shape_main == "diag") {
    EHMC_Metric_as_Rcpp_List$M_dense_main          <- diag(10)
    EHMC_burnin_as_Rcpp_List$M_dense_sqrt          <- diag(10)
    EHMC_Metric_as_Rcpp_List$M_inv_dense_main      <- diag(10)
    EHMC_Metric_as_Rcpp_List$M_inv_dense_main_chol <- diag(10)
  }
  
  M_dense_main_non_scaled <-           EHMC_Metric_as_Rcpp_List$M_dense_main
  M_inv_dense_main_non_scaled <-       EHMC_Metric_as_Rcpp_List$M_inv_dense_main
  M_inv_dense_main_chol_non_scaled <-  EHMC_Metric_as_Rcpp_List$M_inv_dense_main_chol
  ##
  M_main_diag <- EHMC_Metric_as_Rcpp_List$M_main_vec
  M_inv_main_diag <- EHMC_Metric_as_Rcpp_List$M_inv_main_vec
  
 
  {
      
          ## --------  SNAPER stuff for main params (to load into C++ structs)
          EHMC_burnin_as_Rcpp_List$snaper_m_vec_main <-    theta_vec_mean[index_main]
          EHMC_burnin_as_Rcpp_List$snaper_s_vec_main_empirical <- rep(1, n_params_main)
          ##
          EHMC_burnin_as_Rcpp_List$snaper_w_vec_main <- fn_initialise_snaper_direction(n_params_main)
          EHMC_burnin_as_Rcpp_List$eigen_max_main <- sqrt(sum(EHMC_burnin_as_Rcpp_List$snaper_w_vec_main^2))
          EHMC_burnin_as_Rcpp_List$eigen_vector_main <- EHMC_burnin_as_Rcpp_List$snaper_w_vec_main / EHMC_burnin_as_Rcpp_List$eigen_max_main
          ##
          ## --------  SNAPER stuff for nuisance (to load into C++ structs)
          EHMC_burnin_as_Rcpp_List$snaper_m_vec_us <-     theta_vec_mean[index_nuisance]
          EHMC_burnin_as_Rcpp_List$snaper_s_vec_us_empirical  <- rep(1, n_nuisance)
          if (n_nuisance > 0L) {
              EHMC_burnin_as_Rcpp_List$snaper_w_vec_us <- fn_initialise_snaper_direction(n_nuisance)
              EHMC_burnin_as_Rcpp_List$eigen_max_us <- sqrt(sum(EHMC_burnin_as_Rcpp_List$snaper_w_vec_us^2))
              EHMC_burnin_as_Rcpp_List$eigen_vector_us <- EHMC_burnin_as_Rcpp_List$snaper_w_vec_us / EHMC_burnin_as_Rcpp_List$eigen_max_us
          } else {
              EHMC_burnin_as_Rcpp_List$snaper_w_vec_us <- numeric(0)
              EHMC_burnin_as_Rcpp_List$eigen_max_us <- 0.0
              EHMC_burnin_as_Rcpp_List$eigen_vector_us <- numeric(0)
          }
          ##
          EHMC_Metric_as_Rcpp_List$M_us_vec <-   1.0 / EHMC_Metric_as_Rcpp_List$M_inv_us_vec  
          ##
          EHMC_burnin_as_Rcpp_List$M_dense_sqrt <- diag(n_params_main)
          ##
          snaper_m_vec_all <-    theta_vec_mean
          snaper_m_prop_vec_all <- theta_vec_mean
          snaper_s_vec_all_empirical <- rep(1, n_params)
          trajectory_metric <- list(main = NULL, us = NULL, joint = NULL)
          trajectory_mass <- list(main = NULL, us = NULL, joint = NULL)
          trajectory_direction <- list(main = fn_initialise_snaper_direction(n_params_main),
                                       us = if (n_nuisance > 0) fn_initialise_snaper_direction(n_nuisance) else numeric(0),
                                       ## "joint" = main rows first, then the nuisance rows (tau_adaptation_block = "joint"):
                                       joint = if (n_nuisance > 0) fn_initialise_snaper_direction(n_params_main + n_nuisance) else numeric(0))

  }
  
   
  EHMC_args_as_Rcpp_List$eps_main <- 0.00001 # just in case fn_find_initial_eps_main_and_us fails
  EHMC_args_as_Rcpp_List$eps_us   <- 0.00001 # just in case fn_find_initial_eps_main_and_us fails

  
  # At initialization
  EHMC_burnin_as_Rcpp_List$tau_m_adam_main <- 0
  EHMC_burnin_as_Rcpp_List$tau_v_adam_main <- 0
  ##
  EHMC_burnin_as_Rcpp_List$tau_m_adam_us <- 0
  EHMC_burnin_as_Rcpp_List$tau_v_adam_us <- 0
  
  
  # print(paste(" -------------------- Model_args_as_Rcpp_List = -------------------- "))
  # str(Model_args_as_Rcpp_List)
  
  
  ## ---- Threads for the step-size search fn_find_initial_eps_main_and_us() (here and at the ChEES handover). A build whose
  ##      search has the n_threads argument spreads each lp/grad evaluation of the search over the burn-in's within-chain
  ##      (WCP) threads and adds the chunk contributions in the serial order, so eps is bitwise the same as single-threaded;
  ##      a build without it gets the original single-threaded call.
  eps_search_has_n_threads <- isTRUE(tryCatch("n_threads" %in% names(formals(fn_find_initial_eps_main_and_us)), error = function(e) FALSE))
  eps_search_n_threads <- if (is.numeric(n_threads_WCP) && (length(n_threads_WCP) == 1) && is.finite(n_threads_WCP) && (n_threads_WCP >= 1)) floor(n_threads_WCP) else 1


  ## INITIAL VALUE(S) FOR EPSILON (I.E. THE HMC STEP-SIZE(S)) ---- BOOKMARK ------------------------------------------------------------------------------:
  ##
  ## ---- eps_carry_over: start from the previous stage's final step-sizes and skip the find_initial_eps search below:
  ##
  if (!is.null(eps_carry_over)) {

        eps_carry_over_main <- eps_carry_over$eps_main
        eps_carry_over_us   <- if (isTRUE(partitioned_HMC)) eps_carry_over$eps_us else eps_carry_over_main
        ##
        for (eps_carry_over_value in list(eps_carry_over_main, eps_carry_over_us)) {
              if (!is.numeric(eps_carry_over_value) || length(eps_carry_over_value) != 1 ||
                  !is.finite(eps_carry_over_value) || eps_carry_over_value <= 0) {
                    stop("eps_carry_over must be NULL or list(eps_main = , eps_us = ) holding single positive finite step-sizes.")
              }
        }
        ##
        EHMC_args_as_Rcpp_List$eps_main <- eps_carry_over_main
        ## nuisance: its own carried value under partitioned HMC; the joint sampler keeps eps_us = eps_main:
        EHMC_args_as_Rcpp_List$eps_us   <- eps_carry_over_us
        ##
        message(colourise(paste0("Initial step_sizes carried over from the previous stage (find_initial_eps search skipped): eps_main = ",
                                 signif(EHMC_args_as_Rcpp_List$eps_main, 4), ", eps_us = ", signif(EHMC_args_as_Rcpp_List$eps_us, 4)),
                          "cyan"))

  } else {
  try({

        if (sample_nuisance == TRUE) {
          theta_us_vec <-   c(theta_vec_mean[index_nuisance])
        } else {
          theta_us_vec <-   c(rep(1, n_nuisance))
        }

        if (isTRUE(eps_search_has_n_threads)) {

              par_res <- (fn_find_initial_eps_main_and_us(     theta_main_vec_initial_ref = matrix(c(theta_vec_mean[index_main]), ncol = 1),
                                                                             theta_us_vec_initial_ref = matrix(c(theta_us_vec), ncol = 1),
                                                                             partitioned_HMC = partitioned_HMC,
                                                                             seed = seed,
                                                                             Model_type = Model_type,
                                                                             force_autodiff = force_autodiff,
                                                                             force_PartialLog = force_PartialLog,
                                                                             multi_attempts = multi_attempts,
                                                                             y_ref = y,
                                                                             Model_args_as_Rcpp_List = Model_args_as_Rcpp_List,
                                                                             EHMC_args_as_Rcpp_List = EHMC_args_as_Rcpp_List,
                                                                             EHMC_Metric_as_Rcpp_List = EHMC_Metric_as_Rcpp_List,
                                                                             n_threads = eps_search_n_threads))

        } else {

        par_res <- (fn_find_initial_eps_main_and_us(     theta_main_vec_initial_ref = matrix(c(theta_vec_mean[index_main]), ncol = 1),
                                                                       theta_us_vec_initial_ref = matrix(c(theta_us_vec), ncol = 1),
                                                                       partitioned_HMC = partitioned_HMC,
                                                                       seed = seed,
                                                                       Model_type = Model_type,
                                                                       force_autodiff = force_autodiff,
                                                                       force_PartialLog = force_PartialLog,
                                                                       multi_attempts = multi_attempts,
                                                                       y_ref = y,
                                                                       Model_args_as_Rcpp_List = Model_args_as_Rcpp_List,
                                                                       EHMC_args_as_Rcpp_List = EHMC_args_as_Rcpp_List,
                                                                       EHMC_Metric_as_Rcpp_List = EHMC_Metric_as_Rcpp_List))

        }


        message(paste("Initial step_sizes = "))
        ## main
        EHMC_args_as_Rcpp_List$eps_main <- min(0.50, par_res[[1]])
        cat(paste("eps_main = "))
        print(EHMC_args_as_Rcpp_List$eps_main)

        ## nuisance
        if (partitioned_HMC == TRUE) {

              EHMC_args_as_Rcpp_List$eps_us <-   min(0.50, par_res[[2]])
              cat(paste("eps_us = "))
              print(EHMC_args_as_Rcpp_List$eps_us)

        } else if (partitioned_HMC == FALSE) {

              EHMC_args_as_Rcpp_List$eps_us <-  EHMC_args_as_Rcpp_List$eps_main

        }

  })
  } ## end of "else" (no eps_carry_over: find_initial_eps search) of "if (!is.null(eps_carry_over))"
  
  if (!is.null(eps_init)) {
    EHMC_args_as_Rcpp_List$eps_main <- eps_init
    EHMC_args_as_Rcpp_List$eps_us   <- eps_init
    message(paste("eps initialised from previous stage:", signif(eps_init, 4)))
  }
  ##
  ## ---- LARGE-STEP WARM START (eps_initial) ----------------------------------------------
  ##
  ## The step-size found by the heuristic (or carried over via eps_init) can be small enough
  ## that the first iterations barely move, which wastes warm-up on a region the chains should
  ## be crossing quickly. Holding eps at a deliberately LARGE value for the first
  ## eps_initial_iter iterations lets them cover ground first; eps is then handed back to the
  ## value it would have started at and adapted from there as normal.
  ##
  ## During this window tau is 1 * eps (the clip_iter MALA branch below), so these are big
  ## single-leapfrog (MALA) steps rather than long trajectories.
  ##
  ## eps_initial_iter defaults to n_burnin / 10 - i.e. the 50 iterations that were found to
  ## work at n_burnin = 500, scaled proportionally to any other burnin length.
  ##
  eps_main_after_warm_start <- EHMC_args_as_Rcpp_List$eps_main
  eps_us_after_warm_start   <- EHMC_args_as_Rcpp_List$eps_us
  ##
  if (is.null(eps_initial_iter)) {
    eps_initial_iter <- round(n_burnin / 10)
  }
  eps_initial_iter <- max(0L, min(as.integer(round(eps_initial_iter)), as.integer(n_burnin)))
  ##
  use_eps_warm_start <- (!is.null(eps_initial)) && (eps_initial_iter >= 1L)
  ##
  if (use_eps_warm_start) {

        if (!is.numeric(eps_initial) || length(eps_initial) != 1L || !is.finite(eps_initial) || eps_initial <= 0) {
          stop("'eps_initial' must be NULL or a single positive finite number.")
        }
        ##
        ## respect the configured step-size ceilings - a warm start must not exceed them:
        eps_initial_main <- min(eps_initial, max_eps_main)
        eps_initial_us   <- min(eps_initial, max_eps_us)
        ##
        EHMC_args_as_Rcpp_List$eps_main <- eps_initial_main
        EHMC_args_as_Rcpp_List$eps_us   <- eps_initial_us
        ##
        message(paste0( "eps warm start: holding eps at ", signif(eps_initial_main, 4),
                        " for the first ", eps_initial_iter, " of ", n_burnin,
                        " burnin iterations, then reverting to ",
                        signif(eps_main_after_warm_start, 4), " (main)."))

  }

  fn_update_eps_ADAM <-  function( eps,
                                    eps_m_adam,
                                    eps_v_adam,
                                    iter,
                                    n_burnin,
                                    LR,
                                    p_jump,
                                    adapt_delta,
                                    beta1_adam,
                                    beta2_adam,
                                    eps_adam,
                                    bias_correction_step = iter) {

        previous_state <-  c(eps, eps_m_adam, eps_v_adam)
        attr(previous_state, "adam_update_performed") <-  FALSE
        if (length(bias_correction_step) != 1 || !is.finite(bias_correction_step) || bias_correction_step < 1) {
            stop("Step-size ADAM bias_correction_step must be a single finite number >= 1.")
        }
        ## Gradient moments count updates since their last reset; the learning-rate schedule retains its global iteration.
        gradient <-  p_jump - adapt_delta
        first_moment <-  beta1_adam * eps_m_adam + (1 - beta1_adam) * gradient
        second_moment <-  beta2_adam * eps_v_adam + (1 - beta2_adam) * gradient^2
        first_moment_corrected <-  first_moment / (1 - beta1_adam^bias_correction_step)
        second_moment_corrected <-  second_moment / (1 - beta2_adam^bias_correction_step)
        current_alpha <-  LR * (1 - (1 - LR) * iter / n_burnin)
        next_eps <-  exp(log(eps) + current_alpha * first_moment_corrected / (sqrt(second_moment_corrected) + eps_adam))
        next_state <-  c(next_eps, first_moment, second_moment)
        if (any(!is.finite(next_state)) || next_eps <= 0) return(previous_state)
        attr(next_state, "adam_update_performed") <-  TRUE
        return(next_state)

  }

  {
      # --------  eps for main (to load into C++ structs)
      ## EHMC_burnin_as_Rcpp_List$eps_m_adam_main <- EHMC_args_as_Rcpp_List$eps_main
      EHMC_burnin_as_Rcpp_List$eps_m_adam_main <- 0   ## first moment of the acceptance error (p_jump - adapt_delta), not of eps
      EHMC_burnin_as_Rcpp_List$eps_v_adam_main <- 0
      eps_adam_updates_main <- 0
  
      # --------  eps for nuisance (to load into C++ structs)
      ## EHMC_burnin_as_Rcpp_List$eps_m_adam_us <- EHMC_args_as_Rcpp_List$eps_us
      EHMC_burnin_as_Rcpp_List$eps_m_adam_us <- 0     ## first moment of the acceptance error (p_jump - adapt_delta), not of eps
      EHMC_burnin_as_Rcpp_List$eps_v_adam_us  <- 0
      eps_adam_updates_us <- 0
  
      # --------  tau for main (to load into C++ structs)
      EHMC_args_as_Rcpp_List$tau_main  <-  EHMC_args_as_Rcpp_List$eps_main
      EHMC_burnin_as_Rcpp_List$tau_m_adam_main <- 0
      EHMC_burnin_as_Rcpp_List$tau_v_adam_main <- 0
  
      # --------  tau for nuisance (to load into C++ structs)
      EHMC_args_as_Rcpp_List$tau_us  <-  EHMC_args_as_Rcpp_List$eps_us
      EHMC_burnin_as_Rcpp_List$tau_m_adam_us <- 0
      EHMC_burnin_as_Rcpp_List$tau_v_adam_us <- 0
  
      # --------  M for nuisance (to load into C++ structs)
      M_us_vec <-   rep(1, n_nuisance)
      EHMC_Metric_as_Rcpp_List$M_inv_us_vec <- 1 / M_us_vec
      EHMC_burnin_as_Rcpp_List$sqrt_M_us_vec <- rep(1, n_nuisance)
  
      # -------- M for main (to load into C++ structs)
      EHMC_Metric_as_Rcpp_List$M_inv_dense_main <- diag(rep(1, n_params_main))
      EHMC_Metric_as_Rcpp_List$M_dense_main <- diag(rep(1, n_params_main))
      EHMC_Metric_as_Rcpp_List$M_inv_dense_main_chol <- diag(rep(1, n_params_main))
  }
 
  if (metric_shape_main == "diag") {
    EHMC_Metric_as_Rcpp_List$M_dense_main          <- diag(10)
    EHMC_burnin_as_Rcpp_List$M_dense_sqrt          <- diag(10)
    EHMC_Metric_as_Rcpp_List$M_inv_dense_main      <- diag(10)
    EHMC_Metric_as_Rcpp_List$M_inv_dense_main_chol <- diag(10)
  }

  div_main <- p_jump_per_chain <- div_us <- p_jump_us_per_chain <- c()
  
  time_burnin <- 0 
  
  ####  print(theta_main_vectors_all_chains_input_from_R)
  
  shrinkage_factor <- 1
  
  # ## ---- Stan-style windowed dense-metric adaptation (main params) ----
  # win_init_end <- max(clip_iter, round(0.10 * n_adapt))  # eps-only warmup buffer
  # win_slow_end <- round(0.80 * n_adapt)                  # metric frozen after this; eps adapts to n_adapt
  # ##
  # compute_window_ends <- function(first_iter, last_iter, base_size = 25) {
  #   ends <- integer(0); pos <- first_iter; size <- base_size
  #   while (pos <= last_iter) {
  #     this_end <- pos + size - 1
  #     if (this_end + 2 * size > last_iter) this_end <- last_iter  # absorb remainder
  #     this_end <- min(this_end, last_iter)
  #     ends <- c(ends, this_end)
  #     pos  <- this_end + 1
  #     size <- 2 * size
  #   }
  #   ends
  # }
  # win_ends <- compute_window_ends(win_init_end + 1, win_slow_end)
  # if (length(win_ends) == 0) warning("burnin too short for windowed metric adaptation!")
  # message("Metric windows end at iterations: ", paste(win_ends, collapse = ", "))
  # ##
  # win_sum_x   <- rep(0.0, n_params_main)
  # win_sum_xxT <- matrix(0.0, n_params_main, n_params_main)
  # win_n       <- 0L
  
  # ## ---- Phase-2: windowed empirical refinement of the Hessian metric ----
  # win_first_start <- round(0.50 * n_adapt)   # Hessian-only before this
  # win_metric_end  <- round(0.80 * n_adapt)   # metric frozen after this; eps keeps adapting
  # ##
  # compute_window_ends <- function(first_iter, last_iter, base_size = 50) {
  #   ends <- integer(0); pos <- first_iter; size <- base_size
  #   while (pos <= last_iter) {
  #     this_end <- pos + size - 1
  #     if (this_end + 2 * size > last_iter) this_end <- last_iter
  #     this_end <- min(this_end, last_iter)
  #     ends <- c(ends, this_end); pos <- this_end + 1; size <- 2 * size
  #   }
  #   ends
  # }
  # win_ends <- compute_window_ends(win_first_start + 1, win_metric_end)
  # message("Phase-2 metric windows end at: ", paste(win_ends, collapse = ", "))
  # ##
  # win_sum_x    <- rep(0.0, n_params_main)
  # win_sum_xxT  <- matrix(0.0, n_params_main, n_params_main)
  # win_n        <- 0L
  # win_prev_X   <- NULL      # previous iteration's chain states (for lag-1 rho)
  # win_rho_sum  <- 0.0
  # win_rho_n    <- 0L
  # H_inv_seed   <- NULL      # snapshot of Hessian-derived M_inv, taken at first window start
  
  # ## ---- Phase-2: windowed empirical refinement of the Hessian metric ----
  # # win_first_start <- round(0.50 * n_adapt)
  # win_first_start <- n_adapt + 999999   ## Phase-2 disabled
  # ##
  # win_metric_end  <- round(0.80 * n_adapt)
  # compute_window_ends <- function(first_iter, last_iter, base_size = 50) {
  #   ends <- integer(0); pos <- first_iter; size <- base_size
  #   while (pos <= last_iter) {
  #     this_end <- pos + size - 1
  #     if (this_end + 2 * size > last_iter) this_end <- last_iter
  #     this_end <- min(this_end, last_iter)
  #     ends <- c(ends, this_end); pos <- this_end + 1; size <- 2 * size
  #   }
  #   ends
  # }
  # win_ends <- compute_window_ends(win_first_start + 1, win_metric_end)
  # message("Phase-2 metric windows end at: ", paste(win_ends, collapse = ", "))
  # win_sum_x    <- rep(0.0, n_params_main)
  # win_sum_xxT  <- matrix(0.0, n_params_main, n_params_main)
  # win_n        <- 0L
  # win_prev_X   <- NULL
  # win_rho_sum  <- 0.0
  # win_rho_n    <- 0L
  # H_inv_seed   <- NULL
  
  # ## Cross-chain covariance accumulator (replaces Welford-of-the-mean):
  # # cov_crosschain_ema <- NULL
  # # ema_alpha <- 0.05   ## smoothing weight per iteration; ~1/alpha iter memory
  # cov_crosschain_ema <- NULL
  # ema_alpha <- 0.10                                  # faster forgetting
  # cov_start_iter <- round(0.5 * n_adapt)             # accumulate late only
  
  L_main_during_burnin_vec <- numeric(length = 0L)
  L_us_during_burnin_vec <-   numeric(length = 0L)
  ## Post-update settings at each iteration, not the realised per-chain jittered integration times.
  tau_main_during_burnin_vec <- eps_main_during_burnin_vec <- rep(x = NA_real_, times = n_burnin)
  tau_us_during_burnin_vec <- eps_us_during_burnin_vec <- rep(x = NA_real_, times = n_burnin)
  ## Burn-in record for diagnosing the adapted metric (saved with the run): every burn-in chain's main-parameter draws after
  ## each iteration (iteration x parameter x chain), the main metric variances (diagonal of M_inv) used in that iteration, and
  ## quantiles of the nuisance metric variances (M_inv_us_vec) in that iteration:
  burnin_trace_main_all_chains <-  array(data = NA_real_, dim = c(n_burnin, n_params_main, n_chains_burnin))
  burnin_metric_main_variance_history <-  matrix(data = NA_real_, nrow = n_burnin, ncol = n_params_main)
  burnin_metric_nuisance_variance_quantiles_history <-  matrix(data = NA_real_, nrow = n_burnin, ncol = 5,
                                                              dimnames = list(NULL, c("q05", "q25", "q50", "q75", "q95")))
  ## Cross-chain acceptance at each iteration: the arithmetic mean (printed as p_jump) and the mean that the eps update used
  ## (eps_acceptance_mean; identical to the arithmetic mean when eps_acceptance_mean = "arithmetic"):
  p_jump_main_during_burnin_vec <- p_jump_main_for_eps_during_burnin_vec <- rep(x = NA_real_, times = n_burnin)
  p_jump_us_during_burnin_vec   <- p_jump_us_for_eps_during_burnin_vec   <- rep(x = NA_real_, times = n_burnin)
  ChEES_criterion_ema_vec <- ChEES_per_tau_gradient_vec <- rep(x = NA_real_, times = n_burnin)
  ChEES_criterion_ema <- NA_real_
  ## burnin_algorithm = "ESJD_CHESSR" only: moving averages of the chain-mean ESJD and CHESSR criteria (the levels that turn each
  ## gradient into a log-tau derivative, fn_metric_tau_block_update); reset with ChEES_criterion_ema at the handover:
  ESJD_CHESSR_component_criterion_ema <-  c(ESJD = NA_real_, CHESSR = NA_real_)
  ## burnin_algorithm = "ESJD_SNAPER" only: the same component moving averages, with the learned SNAPER rate as the second component:
  ESJD_SNAPER_component_criterion_ema <-  c(ESJD = NA_real_, SNAPER = NA_real_)
  ## burnin_algorithm = "LQ_ESSR" only: moving averages of the per-coordinate chain means of its jump statistics and start moments
  ## (fn_metric_tau_block_update); NULL = none yet, reset with ChEES_criterion_ema at the handover:
  LQ_ESSR_component_criterion_ema <-  NULL
  ## Iterations at which the divergence-triggered tau shrink fired (tau_shrink_on_divergence = TRUE):
  tau_shrink_fired_vec <- rep(x = FALSE, times = n_burnin)
  ##
  ## ---- CHESSR_time / SNAPER_time: per-iteration record of the online burn-in timing and of the criterion's quantities
  ##      (R_fn_time_criterion.R); the two quantities stay 0 for every other criterion, which ignores them:
  ##
  tau_offset_from_sampling_overhead_at_update      <-  0
  burnin_to_sampling_leapfrog_time_ratio_at_update <-  0
  ## (MALT's exponent: 1 = the penalty without it; the other criteria ignore it)
  lag_one_autocorrelation_rho_at_update            <-  1
  if (time_criterion_active) {
      time_per_iter_native_call_burnin_vec                 <-  rep(x = NA_real_, times = n_burnin)
      n_leapfrog_steps_per_iter_burnin_vec                 <-  rep(x = NA_real_, times = n_burnin)
      used_in_time_per_leapfrog_step_burnin_fit_vec        <-  rep(x = FALSE,    times = n_burnin)
      time_per_leapfrog_step_burnin_vec                    <-  rep(x = NA_real_, times = n_burnin)
      tau_offset_from_sampling_overhead_vec                <-  rep(x = NA_real_, times = n_burnin)
      burnin_to_sampling_leapfrog_time_ratio_vec           <-  rep(x = NA_real_, times = n_burnin)
      time_to_target_ESS_tau_penalty_at_tau_main_vec       <-  rep(x = NA_real_, times = n_burnin)
      ESS_elasticity_wrt_log_tau_vec                       <-  rep(x = NA_real_, times = n_burnin)
      ChEES_or_SNAPER_statistic_per_tau_chain_mean_vec     <-  rep(x = NA_real_, times = n_burnin)
      ChEES_or_SNAPER_statistic_log_tau_derivative_per_tau_chain_mean_vec <-  rep(x = NA_real_, times = n_burnin)
      iteration_of_no_interior_optimum_warning             <-  NA_real_
      time_per_leapfrog_step_burnin_fit_sums               <-  fn_time_per_leapfrog_step_burnin_fit_init()
      time_criterion_at_handover                           <-  NULL
      time_criterion_at_last_update                        <-  NULL
      eps_main_used_for_iteration                          <-  NA_real_
      ##
      ## ---- MALT's exponent (lag_one_autocorrelation_rho) used at each tau update and ("adaptive") its running moments:
      ##
      lag_one_autocorrelation_rho_used_vec                              <-  rep(x = NA_real_, times = n_burnin)
      lag_one_autocorrelation_rho_estimate_after_update_vec             <-  rep(x = NA_real_, times = n_burnin)
      lag_one_autocorrelation_rho_n_valid_chains_vec                    <-  rep(x = NA_real_, times = n_burnin)
      lag_one_autocorrelation_rho_moments                               <-  fn_lag_one_autocorrelation_rho_moments_init()
      lag_one_autocorrelation_rho_source_at_update                      <-  "setting"
  }
  
  ## print every n_refresh iterations (Stan's 'refresh' convention). This used to be
  ## hard-wired to n_burnin/25, which silently ignored the n_refresh the caller passed:
  n_refresh_iter_counter <- max(1L, as.integer(round(n_refresh)))
  
  EHMC_args_as_Rcpp_List$tau_main <- EHMC_args_as_Rcpp_List$eps_main
  
  
  gap <- clip_iter_tau
  ## Tau's optimiser starts at the handover, not at the beginning of the epsilon/metric warm-up.
  n_tau_adaptation_iterations <- n_adapt - gap
  ##
  ## ---- tau_initial = "adaptive": everything before the handover runs exactly as with tau_initial = pi (the ramp's
  ##      stage target); only the handover itself (ii == gap) uses the eigenvalue estimate. A numeric tau_initial is untouched.
  ##
  tau_initial_adaptive <-  identical(tau_initial, "adaptive")
  if (tau_initial_adaptive) tau_initial <-  pi
  ##
  ## ---- "all" uses every iteration from 1 to the handover. Numeric fractions retain the windows described below.
  ##      The 30-iteration minimum applies only to numeric fractions; the default remains 0.5.
  ##
  ## ---- tau_initial = "adaptive": the part of [clip_iter, gap] whose draws feed lambda_max. Option
  ##      NicoStan_tau_initial_moments_window_fraction (default 0.5 = the last half; 1 = the whole of [clip_iter, gap], the original
  ##      behaviour) keeps only the last part of it, so that the early burn-in draws taken while the chains are still moving into the typical set do not
  ##      inflate lambda_max. The window never has fewer than NicoStan_tau_initial_moments_window_min_iter iterations (default 30)
  ##      and never starts before clip_iter:
  ##
  # tau_initial_moments_window_fraction <-  as.numeric(getOption("NicoStan_tau_initial_moments_window_fraction", default = 1))
  tau_initial_moments_window_fraction <-  getOption("NicoStan_tau_initial_moments_window_fraction", default = 0.5)
  tau_initial_moments_all <-  identical(tau_initial_moments_window_fraction, "all")
  ##
  if (tau_initial_moments_all) {

        tau_initial_moments_window_min_iter <-  NA_real_
        tau_initial_moments_window_length <-  gap
        tau_initial_moments_start_iter <-  1

  } else {

        tau_initial_moments_window_fraction <-  as.numeric(tau_initial_moments_window_fraction)
        tau_initial_moments_window_min_iter <-  as.numeric(getOption("NicoStan_tau_initial_moments_window_min_iter", default = 30))
        if (length(tau_initial_moments_window_fraction) != 1 || !is.finite(tau_initial_moments_window_fraction) ||
            tau_initial_moments_window_fraction <= 0 || tau_initial_moments_window_fraction > 1) {
              stop("NicoStan_tau_initial_moments_window_fraction must be \"all\" or a single number in (0, 1].")
        }
        if (length(tau_initial_moments_window_min_iter) != 1 || !is.finite(tau_initial_moments_window_min_iter) || tau_initial_moments_window_min_iter < 1) {
              stop("NicoStan_tau_initial_moments_window_min_iter must be a single number >= 1.")
        }
        tau_initial_moments_window_length <-  max(tau_initial_moments_window_min_iter,
                                                  ceiling(tau_initial_moments_window_fraction * (gap - clip_iter + 1)))
        tau_initial_moments_start_iter <-  max(clip_iter, gap - tau_initial_moments_window_length + 1)

  }
  ##
  if (!isTRUE(manual_tau)) {
      if (!is.numeric(tau_initial) || length(tau_initial) != 1 || !is.finite(tau_initial) || tau_initial <= 0) {
          ## stop("tau_initial must be a single positive finite number.")
          stop("tau_initial must be a single positive finite number or \"adaptive\".")
      }
      if (gap < clip_iter || clip_iter < 1 || gap != round(gap) || clip_iter != round(clip_iter) ||
          n_tau_adaptation_iterations < 1) {
          stop("Adaptive tau requires 1 <= clip_iter <= clip_iter_tau < n_adapt.")
      }
      message("tau adaptation v2: ", tau_ramp, " ramp to tau_initial = ", signif(tau_initial, 5),
              "; one handover at iteration ", gap, "; updates ", gap, "-", n_adapt - 1,
              "; then frozen. No later resets.")
      if (tau_initial_adaptive) {
          message(colourise(paste0("tau_initial = \"adaptive\": the ramp targets pi; at the handover (iteration ", gap,
                                   ") tau = (pi/2) * sqrt(lambda_max), lambda_max from the burn-in draws of iterations ",
                                   ## clip_iter, "-", gap,
                                   tau_initial_moments_start_iter, "-", gap, " in metric coordinates (main: pooled covariance; nuisance: pooled",
                                   " within-individual second moment)."), "cyan"))
      }
  }
  ##
  ## ---- tau-only ADAM first-moment weight. Option NicoStan_tau_adam_beta1 (default NULL = beta1_adam, i.e. the tau ADAM as before;
  ##      beta1_adam is 0, as in the ChEES / SNAPER papers). A number in [0, 1) replaces beta1_adam in every tau ADAM update (the
  ##      first moment and its bias correction; every trajectory criterion, block and path). The step-size (eps) ADAM keeps
  ##      beta1_adam:
  ##
  tau_adam_beta1_option <-  getOption("NicoStan_tau_adam_beta1", default = NULL)
    if (!is.null(bulk_local_tuner)) {
        bulk_local_tuner <-  fn_bulk_local_tuner_settings(bulk_local_tuner)
        if (isTRUE(tau_shrink_on_divergence) ||
            !identical(getOption("NicoStan_tau_adaptation_scheme", "adam_decay"), "adam_decay")) {
            stop("bulk_local_tuner excludes divergence shrink and probe/averaging tau schemes.")
        }
        if (!identical(burnin_algorithm, "ESJD") || isTRUE(partitioned_HMC) || isTRUE(manual_tau) ||
            isTRUE(tau_if_manual_in_L_units) || !isTRUE(randomize_tau_burnin) ||
            !identical(tau_jitter_burnin, "uniform")) {
            stop("bulk_local_tuner requires the supported ESJD uniform-jitter nonpartitioned kernel.")
        }
        if (isTRUE(sample_nuisance) && identical(theta_hat_us_rule, "running_mean")) {
            stop("bulk_local_tuner requires a frozen nuisance centre.")
        }
        bulk_local_tuner_geometry_end <-  max(metric_adaptation_end_iter,
              if (isTRUE(sample_nuisance) && !identical(theta_hat_us_rule, "zero")) {
                    theta_hat_us_freeze_iter
              } else 0)
        bulk_local_tuner_collect_start <-  if (is.null(bulk_local_tuner$freeze_start_iter)) {
              max(n_adapt, bulk_local_tuner_geometry_end + 1, gap + 1)
        } else bulk_local_tuner$freeze_start_iter
        if (bulk_local_tuner_collect_start <= max(bulk_local_tuner_geometry_end, gap)) {
            stop("bulk_local_tuner freeze_start_iter must follow metric/centre freezing and tau handover.")
        }
        if (use_eps_warm_start && bulk_local_tuner_collect_start <= eps_initial_iter) {
            stop("bulk_local_tuner collection must follow the epsilon warm-start window.")
        }
        bulk_local_tuner_decision_iter <-  n_burnin - bulk_local_tuner$settle_iter
        bulk_local_tuner_required_history <-  max(bulk_local_tuner$min_history,
              4 * (bulk_local_tuner$max_lag + 1) - 1)
        bulk_local_tuner_metadata <-  list(settings = bulk_local_tuner,
              status = "skipped", reason = "insufficient_fixed_kernel_history",
              requested_freeze_start_iter = bulk_local_tuner$freeze_start_iter,
              collection_start_iter = bulk_local_tuner_collect_start,
              decision_iter = bulk_local_tuner_decision_iter,
              required_history = bulk_local_tuner_required_history,
              legacy_adaptation_end_iter = n_adapt - 1,
              actual_eps_adaptation_end_iter = n_adapt - 1,
              actual_tau_adaptation_end_iter = n_adapt - 1,
              metric_adaptation_end_iter = metric_adaptation_end_iter,
              nuisance_centre_freeze_iter = theta_hat_us_freeze_iter)
        if (n_chains_burnin < 2) {
            bulk_local_tuner_metadata$reason <-  "insufficient_chains_for_stationarity"
            bulk_local_tuner_collect_start <-  Inf
        } else if (bulk_local_tuner_decision_iter - bulk_local_tuner_collect_start + 1 >=
                   bulk_local_tuner_required_history) {
            bulk_local_tuner_state <-  fn_bulk_local_tuner_init(bulk_local_tuner,
                  bulk_local_tuner_names, n_chains_burnin, bulk_local_tuner_cost_contract)
        } else bulk_local_tuner_collect_start <-  Inf
    }
  if (!is.null(tau_adam_beta1_option) &&
      (!is.numeric(tau_adam_beta1_option) || length(tau_adam_beta1_option) != 1 || !is.finite(tau_adam_beta1_option) ||
       tau_adam_beta1_option < 0 || tau_adam_beta1_option >= 1)) {
        stop(paste0("NicoStan_tau_adam_beta1 must be NULL (tau uses beta1_adam) or a single number in [0, 1); got: ",
                    paste(as.character(tau_adam_beta1_option), collapse = ", "), "."))
  }
  tau_adam_beta1 <-  if (is.null(tau_adam_beta1_option)) beta1_adam else as.numeric(tau_adam_beta1_option)
  ##
  ## ---- eps-only ADAM first-moment weight. Option NicoStan_eps_adam_beta1 (default NULL = beta1_adam, i.e. the step-size ADAM as
  ##      before). A number in [0, 1) replaces beta1_adam in both step-size (eps) ADAM updates (main and nuisance); the tau ADAM
  ##      keeps beta1_adam (or NicoStan_tau_adam_beta1 when that is set):
  ##
  eps_adam_beta1_option <-  getOption("NicoStan_eps_adam_beta1", default = NULL)
  if (!is.null(eps_adam_beta1_option) &&
      (!is.numeric(eps_adam_beta1_option) || length(eps_adam_beta1_option) != 1 || !is.finite(eps_adam_beta1_option) ||
       eps_adam_beta1_option < 0 || eps_adam_beta1_option >= 1)) {
        stop(paste0("NicoStan_eps_adam_beta1 must be NULL (eps uses beta1_adam) or a single number in [0, 1); got: ",
                    paste(as.character(eps_adam_beta1_option), collapse = ", "), "."))
  }
  eps_adam_beta1 <-  if (is.null(eps_adam_beta1_option)) beta1_adam else as.numeric(eps_adam_beta1_option)
  message(colourise(paste0("step-size (eps) ADAM beta1 = ", eps_adam_beta1,
                           if (is.null(eps_adam_beta1_option)) " (beta1_adam)" else " (NicoStan_eps_adam_beta1)"), "cyan"))
  ##
  ## ---- second-moment decay and denominator constant, separately for the step-size (eps) and the trajectory-length (tau) ADAM updates.
  ##      Options NicoStan_eps_adam_beta2, NicoStan_eps_adam_epsilon, NicoStan_tau_adam_beta2 and NicoStan_tau_adam_epsilon; default NULL =
  ##      the sampler arguments beta2_adam / eps_adam, so a caller that sets none of them is unchanged. A beta2 is a number in [0, 1), a
  ##      denominator constant a positive number:
  ##
  fn_adam_option_or_argument <-  function(option_name,
                                          argument_value,
                                          is_decay) {

        option_value <-  getOption(option_name, default = NULL)
        if (is.null(option_value)) return(argument_value)
        value_is_valid <-  is.numeric(option_value) && length(option_value) == 1 && is.finite(option_value) &&
                           (if (is_decay) (option_value >= 0 && option_value < 1) else option_value > 0)
        if (!value_is_valid) {
              stop(paste0(option_name, " must be NULL (the sampler argument) or a single ", if (is_decay) "number in [0, 1)" else "positive number",
                          "; got: ", paste(as.character(option_value), collapse = ", "), "."))
        }
        return(as.numeric(option_value))

  }
  eps_adam_beta2   <-  fn_adam_option_or_argument("NicoStan_eps_adam_beta2",   beta2_adam, is_decay = TRUE)
  eps_adam_epsilon <-  fn_adam_option_or_argument("NicoStan_eps_adam_epsilon", eps_adam,   is_decay = FALSE)
  tau_adam_beta2   <-  fn_adam_option_or_argument("NicoStan_tau_adam_beta2",   beta2_adam, is_decay = TRUE)
  tau_adam_epsilon <-  fn_adam_option_or_argument("NicoStan_tau_adam_epsilon", eps_adam,   is_decay = FALSE)
  message(colourise(paste0("eps ADAM: beta1 = ", eps_adam_beta1, ", beta2 = ", eps_adam_beta2, ", denominator constant = ", eps_adam_epsilon), "cyan"))
  ##
  ## ---- tau learning-rate restart at the metric freeze. Option NicoStan_tau_learning_rate_restart_at_metric_end (default FALSE = one
  ##      linear decay of the tau learning rate from LR to LR^2 over the tau adaptation, iterations gap to n_adapt - 1).
  ##      TRUE: from the first iteration after metric_adaptation_end_iter (the first whose trajectories use the final metric), the
  ##      decay restarts at the full LR and falls linearly to LR^2 at the last tau update (iteration n_adapt - 1); tau itself is
  ##      kept. The ADAM second moment (tau_v_adam) and the bias-correction counter are deliberately NOT reset: only the
  ##      learning-rate schedule restarts, so the restarted steps are still normalised by the gradient scale already learned.
  ##      It needs gap < metric_adaptation_end_iter and at least one tau update after metric_adaptation_end_iter; otherwise it
  ##      has no effect (one message):
  ##
  ## ---- restart iteration = metric_adaptation_end_iter itself (not the iteration after it). Within burn-in iteration ii the order is:
  ##      final metric update (forced at ii == metric_adaptation_end_iter by the automatic schedule), then the trajectory-metric factor,
  ##      then the sampler step ("Perform iteration"), then the eps and tau updates. The trajectories scored by the tau update of
  ##      iteration metric_adaptation_end_iter therefore already run with the final metric, and that update is the first one after
  ##      the metric freeze. tau_v_adam, tau_m_adam and the bias-correction counter are kept (see above), which keeps the bias
  ##      correction consistent (it counts the updates applied to the moments it corrects).
  ##
  tau_learning_rate_restart_at_metric_end <-  getOption("NicoStan_tau_learning_rate_restart_at_metric_end", default = FALSE)
  ## ---- Step-size ADAM at the final metric update (4 Oct 2026). Two switches, main burn-in only (the pre-burn-in
  ##      stage
  ##      runs this function with manual_tau = TRUE and is left as it was):
  ##      NicoStan_eps_adam_reset_at_metric_end (default TRUE): reset the step-size ADAM moments at the final
  ##      metric
  ##      update. The acceptance crashes inside the metric windows inflate the ADAM's second moment, which then
  ##      throttles
  ##      every later step: at b125 the step size had reached ~0.18 inside the last window, fell to ~0.13 with the
  ##      final
  ##      metric update and did not recover in the 22 adaptation iterations left (sampling acceptance 0.89 against
  ##      the
  ##      0.8 target). The reset is what the code already does at the ChEES handover and at the end of the eps
  ##      warm start,
  ##      and what Stan does after every metric window.
  ##      NicoStan_eps_learning_rate_restart_at_metric_end (default TRUE): from the final metric update the eps
  ##      ADAM
  ##      learning-rate schedule restarts (alpha = LR again, decaying to LR^2 over the remaining adaptation
  ##      iterations),
  ##      the step-size analogue of NicoStan_tau_learning_rate_restart_at_metric_end above. Together with the
  ##      reset and
  ##      adapt_delta = 0.7 (27 seeds, LC-MVP N = 10,000,
  ##      validation_staging/bulk-local-tuner-y0658d/eps_target_confirm):
  ##      b125 time to the target min ESS 0.84 [0.74, 0.95] of the previous default, bulk ESS per gradient 1.23
  ##      [1.06, 1.43], tail 1.27 [1.07, 1.50]; b250 within-chain ESS per gradient 1.11 [1.04, 1.18]; b500
  ##      unchanged.
  ##      FALSE restores the previous behaviour.
  eps_adam_reset_at_metric_end <-  isTRUE(getOption("NicoStan_eps_adam_reset_at_metric_end", default = TRUE)) &&
                                   !isTRUE(manual_tau)
  eps_learning_rate_restart_at_metric_end <-  isTRUE(getOption("NicoStan_eps_learning_rate_restart_at_metric_end",
                                                               default = TRUE)) && !isTRUE(manual_tau)
  if (eps_adam_reset_at_metric_end || eps_learning_rate_restart_at_metric_end) {
        message(colourise(paste0("step-size ADAM at the final metric update (iteration ",
                                 metric_adaptation_end_iter, "): moments ",
                                 if (eps_adam_reset_at_metric_end) "reset" else "kept",
                                 ", learning-rate schedule ",
                                 if (eps_learning_rate_restart_at_metric_end) "restarted" else "continued"),
                          "cyan"))
  }
  ##
  ## ---- Trajectory cost offset of the rate criteria (4 Oct 2026): every rate criterion (ESJD, CHESSR, SNAPER, their hybrids,
  ##      LQ_ESSR, and the _time criteria on top of their sampling overhead) divides by the EXPECTED gradient cost of a
  ##      trajectory, t + eps x c, with c = 0.5 (E[ceil(t / eps)] - t / eps under the jitter) + the gradient evaluations per
  ##      iteration beyond the leapfrog steps (1: the endpoint evaluation of the main-only, Stan-model, standard joint and
  ##      flow_kick_flow paths; 0: the joint diffusion kick_flow_kick path, which reuses it), instead of t alone. Dividing by t
  ##      alone is the large-L limit and undercounts the cost of short trajectories (at L ~ 2-3 by 30-50%), which kept CHESSR
  ##      and SNAPER short of the ESS-per-gradient optimum on the German Credit regression (the SNAPER-HMC authors' code divides
  ##      by t + eps). Option NicoStan_rate_criterion_cost_offset_steps: 0 (DEFAULT: the cost t alone, as before), "auto" (the
  ##      path-aware c above) or a number (c). The default stays 0 because on the latent-class benchmarks (LC-MVP N = 10,000,
  ##      b125, 27 seeds) the corrected cost made ESJD worse (time to target ESS 1.20 [1.06, 1.35], bulk ESS per gradient 0.82
  ##      [0.71, 0.95]) and sent CHESSR / SNAPER to 2-3x longer trajectories with half the tail ESS; on the German Credit
  ##      regression it puts CHESSR on the ESS-per-gradient optimum (0.116 vs 0.100) but lowers SNAPER (0.094 vs 0.112) and
  ##      ESJD (0.055 vs 0.084). The offset is eps_<block> x c at every tau update (tau_cost_offset_vec records it).
  ##
  rate_criterion_cost_offset_steps_setting <-  getOption("NicoStan_rate_criterion_cost_offset_steps", default = 0)
  rate_criterion_cost_offset_steps <-  if (identical(rate_criterion_cost_offset_steps_setting, "auto")) {
        has_nuisance_block_for_cost <-  isTRUE(sample_nuisance) && isTRUE(n_nuisance > 0)
        endpoint_gradient_reused <-  has_nuisance_block_for_cost && isTRUE(diffusion_HMC) && !isTRUE(partitioned_HMC) &&
                                     identical(diffusion_HMC_integrator, "kick_flow_kick")
        0.5 + (if (endpoint_gradient_reused) 0 else 1)
  } else {
        as.numeric(rate_criterion_cost_offset_steps_setting)
  }
  if (length(rate_criterion_cost_offset_steps) != 1 || !is.finite(rate_criterion_cost_offset_steps) ||
      rate_criterion_cost_offset_steps < 0) {
        stop("NicoStan_rate_criterion_cost_offset_steps must be \"auto\" or one number >= 0.")
  }
  tau_cost_offset_vec <-  rep(x = NA_real_, times = n_burnin)
  tau_cost_offset_at_update <-  0
  if (!isTRUE(manual_tau) && rate_criterion_cost_offset_steps > 0) {
        message(colourise(paste0("rate criteria: trajectory cost = tau + eps x ", signif(rate_criterion_cost_offset_steps, 3),
                                 " (expected gradient evaluations beyond tau / eps; the default 0 is the cost tau alone)"), "cyan"))
  }
  fn_eps_schedule_iter <-  function(ii) {
        if (eps_learning_rate_restart_at_metric_end && ii >= metric_adaptation_end_iter) {
              return(ii - metric_adaptation_end_iter + 1)
        }
        return(ii)
  }
  fn_eps_schedule_length <-  function(ii) {
        if (eps_learning_rate_restart_at_metric_end && ii >= metric_adaptation_end_iter) {
              return(n_adapt - metric_adaptation_end_iter + 1)
        }
        return(n_adapt)
  }
  if (!is.logical(tau_learning_rate_restart_at_metric_end) || length(tau_learning_rate_restart_at_metric_end) != 1 ||
      is.na(tau_learning_rate_restart_at_metric_end)) {
        stop("NicoStan_tau_learning_rate_restart_at_metric_end must be TRUE or FALSE.")
  }
  # tau_learning_rate_restart_iter <-  metric_adaptation_end_iter + 1
  tau_learning_rate_restart_iter <-  metric_adaptation_end_iter
  tau_learning_rate_restart_active <-  tau_learning_rate_restart_at_metric_end && !isTRUE(manual_tau) &&
                                       (metric_adaptation_end_iter > gap) && (tau_learning_rate_restart_iter < n_adapt)
  if (tau_learning_rate_restart_at_metric_end && !isTRUE(manual_tau) && !tau_learning_rate_restart_active) {
        message(colourise(paste0("NicoStan_tau_learning_rate_restart_at_metric_end = TRUE has no effect in this burn-in: metric_adaptation_end_iter (",
                                 metric_adaptation_end_iter, ") must be after the tau handover (iteration ", gap,
                                 ## ") and before the last tau update (iteration ", n_adapt - 1, ")."), "cyan"))
                                 ") and at or before the last tau update (iteration ", n_adapt - 1, ")."), "cyan"))
  }
  ##
  ## ---- tau ADAM settings of this burn-in, reported once when tau is adapted:
  ##
  if (!isTRUE(manual_tau)) {
        # message(colourise(paste0("tau ADAM: beta1 = ", tau_adam_beta1,
        #                          if (is.null(tau_adam_beta1_option)) " (= beta1_adam, shared with eps)" else paste0(" (NicoStan_tau_adam_beta1; eps keeps beta1_adam = ", beta1_adam, ")"),
        #                          ", beta2 = ", beta2_adam, ", eps_adam = ", eps_adam,
        message(colourise(paste0("tau ADAM: beta1 = ", tau_adam_beta1, ", beta2 = ", tau_adam_beta2, ", denominator constant = ", tau_adam_epsilon,
                                 " | learning-rate restart at the metric freeze = ", tau_learning_rate_restart_at_metric_end,
                                 if (tau_learning_rate_restart_active) paste0(" (from the tau update of iteration ", tau_learning_rate_restart_iter,
                                                                               " to iteration ", n_adapt - 1, ")") else ""), "cyan"))
  }
  ##
  ## ---- tau adaptation scheme. Option NicoStan_tau_adaptation_scheme:
  ##        "adam_decay"          (default) ADAM on log tau from the handover tau, learning rate decaying linearly from LR to LR^2 (the
  ##                              scheme above; the code without this option, bit for bit).
  ##        "probe_then_average"  1. at the handover tau = tau_handover / 8 (rung -3 of tau_handover x 2^rung);
  ##                              2. probe: the aggregated gradient of every tau update (the one the ADAM update would use; every criterion
  ##                                 and block) is summed over blocks of 4 tau updates. After a block with a sum >= 0 tau doubles (rung + 1),
  ##                                 up to tau_handover x 4 (rung 2; a block with a sum >= 0 at rung 2 ends the probe there); the first block
  ##                                 with a sum < 0 halves tau (rung - 1, never below rung -3) and ends the probe. tau changes only at the end
  ##                                 of a block; the ADAM step computed during the probe is discarded and the ADAM moments are not touched;
  ##                              3. from the tau update after the one that ends the probe: ADAM on log tau with the tau ADAM settings, fresh
  ##                                 moments and the CONSTANT learning rate LR (so the learning-rate restart has no effect);
  ##                              4. log tau after every tau update from iteration metric_adaptation_end_iter on (after the max_tau_main
  ##                                 ceiling) is averaged; the tau handed to sampling is exp(that mean) (the last tau when no update was
  ##                                 averaged), before any tau_sampling_scale factor.
  ##        "fixed_length_probe_then_decay_and_average"  as "probe_then_average" with three changes: (a) the probe's updates run fixed-length
  ##                              trajectories (tau_ii = tau) also when randomize_tau_burnin = TRUE, and the jitter resumes at the iteration
  ##                              after the probe ends; (b) the top rung is 0 (tau_handover), not 2; (c) after the probe the learning rate falls
  ##                              linearly from LR (first ADAM update) to LR^2 (last tau update, iteration n_adapt - 1).
  ##                              Only tau_main is probed and averaged: tau_us follows it for the joint sampler; with partitioned_HMC = TRUE
  ##                              tau_us is not adapted in this burn-in and is left as it is.
  ##
  tau_adaptation_scheme <-  getOption("NicoStan_tau_adaptation_scheme", default = "adam_decay")
  # if (!is.character(tau_adaptation_scheme) || length(tau_adaptation_scheme) != 1 || !tau_adaptation_scheme %in% c("adam_decay", "probe_then_average")) {
  #       stop(paste0("NicoStan_tau_adaptation_scheme must be \"adam_decay\" or \"probe_then_average\"; got: ",
  if (!is.character(tau_adaptation_scheme) || length(tau_adaptation_scheme) != 1 ||
      !tau_adaptation_scheme %in% c("adam_decay", "probe_then_average", "fixed_length_probe_then_decay_and_average")) {
        stop(paste0("NicoStan_tau_adaptation_scheme must be \"adam_decay\", \"probe_then_average\" or \"fixed_length_probe_then_decay_and_average\"; got: ",
                    paste(as.character(tau_adaptation_scheme), collapse = ", "), "."))
  }
  # tau_probe_then_average <-  identical(tau_adaptation_scheme, "probe_then_average") && !isTRUE(manual_tau)
  tau_probe_fixed_length_decay <-  identical(tau_adaptation_scheme, "fixed_length_probe_then_decay_and_average") && !isTRUE(manual_tau)
  tau_probe_then_average <-  (identical(tau_adaptation_scheme, "probe_then_average") || tau_probe_fixed_length_decay) && !isTRUE(manual_tau)
  tau_probe_block_length <-  4
  tau_probe_rung_start   <-  -3
  # tau_probe_rung_max     <-  2
  tau_probe_rung_max     <-  if (tau_probe_fixed_length_decay) 0 else 2
  # tau_probe_state <-  list(probing = FALSE, rung = tau_probe_rung_start, gradient_block_sum = 0, n_in_block = 0,
  #                          log_tau_start = NA_real_, end_iteration = NA_real_)
  ## (component_block_sums: burnin_algorithm = "ESJD_CHESSR" only, the block sums of the chain means of its two components; see fn_tau_probe_step)
  tau_probe_state <-  list(probing = FALSE, rung = tau_probe_rung_start, gradient_block_sum = 0, n_in_block = 0,
                           log_tau_start = NA_real_, end_iteration = NA_real_,
                           component_block_sums = c(ESJD_gradient = 0, ESJD_criterion = 0, CHESSR_gradient = 0, CHESSR_criterion = 0,
                                                    SNAPER_gradient = 0, SNAPER_criterion = 0))
  tau_probe_tau_path_vec <-  rep(x = NA_real_, times = n_burnin)
  tau_average_start_iteration <-  metric_adaptation_end_iter
  tau_average_sum_log_tau <-  0
  tau_average_n_updates <-  0
  tau_main_last_iterate <-  NA_real_
  tau_main_averaged <-  NA_real_
  ##
  ## one tau update of the probe: add the aggregated gradient to the block; at the end of a block move the rung and, when it
  ## ends, record the iteration (the probe's tau is exp(log_tau_start + rung * log(2))):
  fn_tau_probe_step <-  function(state,
                                 aggregated_gradient,
                                 ## iteration) {
                                 iteration,
                                 ##
                                 ## ---- burnin_algorithm = "ESJD_CHESSR" only (NULL for the other criteria): this update's chain means of the two
                                 ##      components, c(ESJD_gradient = , ESJD_criterion = , CHESSR_gradient = , CHESSR_criterion = ):
                                 component_chain_means) {

        state$gradient_block_sum <-  state$gradient_block_sum + aggregated_gradient
        ## The probe carries either the existing ESJD/CHESSR pair or the ESJD/SNAPER pair. Keep both named slots in the state so
        ## that the second component is selected by its criterion name without changing the existing CHESSR path.
        if (!is.null(component_chain_means)) {
              state$component_block_sums[["ESJD_gradient"]] <-
                  state$component_block_sums[["ESJD_gradient"]] + component_chain_means[["ESJD_gradient"]]
              state$component_block_sums[["ESJD_criterion"]] <-
                  state$component_block_sums[["ESJD_criterion"]] + component_chain_means[["ESJD_criterion"]]
              second_component_name <-  if ("SNAPER_gradient" %in% names(component_chain_means)) "SNAPER" else "CHESSR"
              state$component_block_sums[[paste0(second_component_name, "_gradient")]] <-
                  state$component_block_sums[[paste0(second_component_name, "_gradient")]] + component_chain_means[[paste0(second_component_name, "_gradient")]]
              state$component_block_sums[[paste0(second_component_name, "_criterion")]] <-
                  state$component_block_sums[[paste0(second_component_name, "_criterion")]] + component_chain_means[[paste0(second_component_name, "_criterion")]]
        }
        state$n_in_block <-  state$n_in_block + 1
        if (state$n_in_block >= tau_probe_block_length) {
              # block_sum_negative <-  state$gradient_block_sum < 0
              ##
              ## ---- "ESJD_CHESSR": the block's sign is that of the block-pooled log-tau derivative of the geometric mean,
              ##        0.5 * sum(ESJD gradient) / sum(ESJD criterion) + 0.5 * sum(CHESSR gradient) / sum(CHESSR criterion),
              ##      the sums over the block's updates (all at the one tau of this rung). The per-update gradients divide by moving
              ##      averages that still carry the levels of the previous rung (tau was half as long), which would weight the two
              ##      terms by the wrong levels; near the optimum the two terms have opposite signs, so this could flip the decision.
              ##      A component with no positive criterion sum gives 0, as a non-finite aggregated gradient does:
              ##
              block_decision_value <-  state$gradient_block_sum
              if (!is.null(component_chain_means)) {
                    block_sums <-  state$component_block_sums
                    second_component_name <-  if (!is.null(component_chain_means) && "SNAPER_gradient" %in% names(component_chain_means)) "SNAPER" else "CHESSR"
                    block_decision_value <-  if (is.finite(block_sums[["ESJD_criterion"]]) && block_sums[["ESJD_criterion"]] > 0 &&
                                                 is.finite(block_sums[[paste0(second_component_name, "_criterion")]]) &&
                                                 block_sums[[paste0(second_component_name, "_criterion")]] > 0) {
                          0.5 * block_sums[["ESJD_gradient"]] / block_sums[["ESJD_criterion"]] +
                          0.5 * block_sums[[paste0(second_component_name, "_gradient")]] / block_sums[[paste0(second_component_name, "_criterion")]]
                    } else 0
                    if (!is.finite(block_decision_value)) block_decision_value <-  0
              }
              block_sum_negative <-  block_decision_value < 0
              probe_at_top <-  !block_sum_negative && (state$rung >= tau_probe_rung_max)
              probe_goes_up <-  !block_sum_negative && (state$rung < tau_probe_rung_max)
              if (block_sum_negative && (state$rung > tau_probe_rung_start)) state$rung <-  state$rung - 1
              if (probe_goes_up) state$rung <-  state$rung + 1
              if (block_sum_negative || probe_at_top) {
                    state$probing <-  FALSE
                    state$end_iteration <-  iteration
              }
              state$gradient_block_sum <-  0
              state$component_block_sums[] <-  0
              state$n_in_block <-  0
        }
        return(state)

  }
  if (tau_probe_then_average) {
        # message(colourise(paste0("tau adaptation scheme = probe_then_average: tau_handover / 8, doubling in blocks of ", tau_probe_block_length,
        #                          " tau updates up to tau_handover x 4, then ADAM with constant LR; tau for sampling = exp(mean log tau) over the tau updates of iterations ",
        message(colourise(paste0("tau adaptation scheme = ", tau_adaptation_scheme, ": tau_handover / 8, doubling in blocks of ", tau_probe_block_length,
                                 " tau updates up to tau_handover x ", 2^tau_probe_rung_max,
                                 if (tau_probe_fixed_length_decay) "; fixed-length probe; LR decays LR -> LR^2 after the probe" else ", then ADAM with constant LR",
                                 "; tau for sampling = exp(mean log tau) over the tau updates of iterations ",
                                 max(gap, tau_average_start_iteration), "-", n_adapt - 1,
                                 if (tau_learning_rate_restart_at_metric_end) " (the learning-rate restart has no effect with this scheme)" else ""), "cyan"))
  }
  ##
  ## ---- tau_initial = "adaptive": lambda_max = the largest eigenvalue of each block's burn-in draws in metric coordinates, from the
  ##      states of all burn-in chains at the start of every iteration in [clip_iter, gap]. For a Gaussian target with covariance Sigma
  ##      and mass matrix M, the slowest oscillation has period 2 pi sqrt(lambda_max(M^(1/2) Sigma M^(1/2))), so
  ##      tau = (pi/2) * sqrt(lambda_max) is a quarter of it.
  ##
  ##        - main block (d = n_params_main): Sigma_hat = the pooled sample covariance of the main parameter vectors over all chains
  ##          and iterations (batch Welford update, the same update as wf_m / wf_C2 of the pooled metric estimator), and M = the main
  ##          mass in force at the handover, as the trajectory criterion reads it (diag: 1 / M_inv_main_vec; dense: M_dense_main;
  ##          unit: the identity). lambda_max_main = the largest eigenvalue of B Sigma_hat t(B), t(B) %*% B = M, by base::eigen()
  ##          at the handover only. The metric's own moments are not the same quantity: wf_C2 exists only for
  ##          metric_estimator = "pooled" (from metric_start_iter, with window resets), and empicical_cov_main of the chain_mean
  ##          estimators is the Welford covariance of the across-chain MEAN.
  ##
  ##        - nuisance block (d = N x n_tests): each coordinate is standardised as y = (u - m) / s, m = the running centre
  ##          snaper_m_vec_us and s = sqrt(M_inv_us_vec), the nuisance metric's own scale at that iteration (Empirical, uniform_diag
  ##          and unit alike). The y of individual i (its n_tests coordinates, in the model's chunk-major storage order; see
  ##          fn_nuisance_chunk_layout_positions) give y_i t(y_i), and the n_tests x n_tests sum over the N individuals and all
  ##          chains is formed in compiled code (inst/src_extra/tau_initial_nuisance_moments.cpp, compiled at first use by
  ##          Rcpp::sourceCpp and cached, so NicoStan / BayesMVP need no rebuild). R keeps the running sum and the count of terms;
  ##          lambda_max_us = the largest eigenvalue of sum / count. External Stan models have no known (individual x test)
  ##          layout, so there every nuisance coordinate is a 1 x 1 block.
  ##
  ##        - partitioned_HMC = TRUE: tau_main from lambda_max_main, tau_us from lambda_max_us. partitioned_HMC = FALSE (one tau):
  ##          tau = (pi/2) * sqrt(max(lambda_max_main, lambda_max_us)).
  ##          With one tau, the block(s) follow the trajectory criterion: tau_adaptation_block "main" gives tau = (pi/2) * sqrt(lambda_max_main)
  ##          (the nuisance estimate is still formed and recorded, not used); "joint" gives the larger of the two as above.
  ##
  ##      A block whose estimate is missing, not finite or not positive uses pi (the numeric path with tau_initial = pi), with a
  ##      message. The accumulation reads the chains, centres and metric only: it changes no sampler state, adaptation or random number.
  ##
  ##      Its cost falls inside time_burnin (adaptive fits only) and is recorded separately (proc.time() elapsed seconds): seconds_moments
  ##      = the per-iteration accumulation, seconds_loader = finding / loading the compiled accumulator, seconds_handover = the two
  ##      eigenvalue computations at the handover.
  ##
  tau_initial_moments <-  list(n_updates = 0,
                               n_errors = 0,
                               nuisance_sampled = isTRUE(sample_nuisance) && (n_nuisance > 0),
                               seconds_moments = 0,
                               seconds_loader = 0,
                               seconds_handover = 0,
                               main = list(n = 0, mean = NULL, C2 = NULL),
                               us = list(sum = NULL, count = 0, positions = NULL, layout = NA_character_,
                                         moments_function = NULL, loader_failed = FALSE,
                                         scale_source = NULL, scale = NULL))
  tau_initial_adaptive_record <-  NULL
  ## the source file of this function when it was sourced from a development tree (keep.source = TRUE), else character(0);
  ## the compiled nuisance accumulator is looked for next to it (R/ -> inst/src_extra/):
  tau_initial_burnin_source_file <-  if (tau_initial_adaptive) utils::getSrcFilename(sys.function(), full.names = TRUE) else character(0)
  ##
  ## batch Welford update of the pooled mean and cross-product sum, one draw per column of X (the update of wf_m / wf_C2):
  fn_tau_initial_welford_update <-  function( moments,
                                              X) {

          X <-  as.matrix(X)
          n_draws_new <-  ncol(X)
          if (is.null(moments$mean)) {
              moments$mean <-  rep(0, nrow(X))
              moments$C2   <-  matrix(0, nrow = nrow(X), ncol = nrow(X))
          }
          batch_mean  <-  rowMeans(X)
          batch_dev   <-  X - batch_mean
          mean_shift  <-  batch_mean - moments$mean
          n_total_new <-  moments$n + n_draws_new
          moments$C2   <-  moments$C2 + tcrossprod(batch_dev) + tcrossprod(mean_shift) * (moments$n * n_draws_new / n_total_new)
          moments$mean <-  moments$mean + mean_shift * (n_draws_new / n_total_new)
          moments$n    <-  n_total_new
          return(moments)

  }
  ##
  ## lambda_max of the pooled main covariance in metric coordinates, B Sigma_hat t(B), B = the trajectory criterion's metric factor:
  fn_tau_initial_lambda_max_main <-  function( moments,
                                               mass) {

          if (is.null(moments$C2) || moments$n < 2) return(NA_real_)
          covariance <-  moments$C2 / (moments$n - 1)
          metric_factor <-  fn_trajectory_metric_factor(mass, nrow(covariance))
          covariance_metric <-  fn_apply_trajectory_metric(metric_factor, t(fn_apply_trajectory_metric(metric_factor, covariance)))
          covariance_metric <-  0.5 * (covariance_metric + t(covariance_metric))
          if (!all(is.finite(covariance_metric))) return(NA_real_)
          return(base::eigen(covariance_metric, symmetric = TRUE, only.values = TRUE)$values[1])

  }
  ##
  ## lambda_max of the pooled within-individual second moment of the standardised nuisance coordinates:
  fn_tau_initial_lambda_max_us <-  function(moments_us) {

          if (is.null(moments_us$sum) || !(moments_us$count > 0)) return(NA_real_)
          second_moment <-  moments_us$sum / moments_us$count
          second_moment <-  0.5 * (second_moment + t(second_moment))
          if (!all(is.finite(second_moment))) return(NA_real_)
          return(base::eigen(second_moment, symmetric = TRUE, only.values = TRUE)$values[1])

  }
  ##
  ## position (1-based) of every nuisance entry, one row per individual and one column per test: the built-in models store the
  ## nuisance vector chunk by chunk (fn_nuisance_chunk_layout_positions, with the burn-in's chunk count and vectorisation);
  ## without that layout, one column (each coordinate on its own):
  fn_tau_initial_nuisance_positions <-  function() {

          n_tests_us <-  if ((Model_type != "Stan") && is.matrix(y)) ncol(y) else NA_real_
          N_us       <-  if ((Model_type != "Stan") && is.matrix(y)) nrow(y) else NA_real_
          fn_layout_positions <-  get0(x = "fn_nuisance_chunk_layout_positions", mode = "function", inherits = TRUE)
          if (is.finite(n_tests_us) && is.finite(N_us) && (N_us * n_tests_us == n_nuisance) && !is.null(fn_layout_positions)) {
              positions <-  fn_layout_positions( N = N_us,
                                                 n_tests = n_tests_us,
                                                 n_chunks = Model_args_as_Rcpp_List$Model_args_ints[4],
                                                 vect_type = Model_args_as_Rcpp_List$Model_args_strings[1])
              return(list(positions = positions, layout = "chunk_major"))
          }
          message(colourise(paste0("tau_initial = \"adaptive\": no (individual x test) layout is known for the nuisance block (n_nuisance = ",
                                   n_nuisance, "); each nuisance coordinate is taken as its own 1 x 1 block."), "cyan"))
          return(list(positions = matrix(seq_len(n_nuisance), ncol = 1), layout = "coordinatewise"))

  }
  ##
  ## the compiled accumulator: option NicoStan_tau_initial_nuisance_moments_cpp, else the development tree this function was
  ## sourced from, else the installed package; compiled once per source version into the NicoStan user cache (or the folder in
  ## option NicoStan_src_extra_cache_dir). Once loaded, it is kept for the R session (option
  ## NicoStan_tau_initial_nuisance_moments_loaded, keyed on the file and its md5 sum), so later burn-ins of the same session do not
  ## call Rcpp::sourceCpp() again. The cache folder is shared by every R process of the user; if loading from it fails (e.g. two
  ## processes compiling into it at the same time), the file is compiled once more into a folder of this process only.
  fn_tau_initial_nuisance_moments_function <-  function() {

          cpp_file_name <-  "tau_initial_nuisance_moments.cpp"
          cpp_function_name <-  "fn_tau_initial_nuisance_second_moment_sum"
          cpp_file_candidates <-  c(getOption("NicoStan_tau_initial_nuisance_moments_cpp", default = NA_character_),
                                    if (length(tau_initial_burnin_source_file) == 1 && nzchar(tau_initial_burnin_source_file))
                                        file.path(dirname(dirname(tau_initial_burnin_source_file)), "inst", "src_extra", cpp_file_name) else NA_character_,
                                    system.file("src_extra", cpp_file_name, package = "NicoStan"))
          cpp_file_candidates <-  cpp_file_candidates[!is.na(cpp_file_candidates) & nzchar(cpp_file_candidates)]
          cpp_file <-  cpp_file_candidates[file.exists(cpp_file_candidates)][1]
          if (is.na(cpp_file)) {
              message(colourise(paste0("tau_initial = \"adaptive\": ", cpp_file_name, " not found (looked in: ",
                                       paste(cpp_file_candidates, collapse = ", "), "); the nuisance block has no estimate."), "red"))
              return(NULL)
          }
          ##
          ## already loaded in this R session (same file, same contents):
          cpp_file <-  normalizePath(cpp_file, mustWork = TRUE)
          cpp_file_md5 <-  unname(tools::md5sum(cpp_file))
          session_loaded <-  getOption("NicoStan_tau_initial_nuisance_moments_loaded", default = NULL)
          if (is.list(session_loaded) && identical(session_loaded$cpp_file, cpp_file) && identical(session_loaded$cpp_file_md5, cpp_file_md5) &&
              is.function(session_loaded$moments_function)) {
              return(session_loaded$moments_function)
          }
          ##
          fn_source_cpp_into <-  function(cache_dir) {
              dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)
              moments_env <-  new.env()
              load_result <-  try(Rcpp::sourceCpp(file = cpp_file, cacheDir = cache_dir, env = moments_env), silent = TRUE)
              if (inherits(load_result, "try-error") || !exists(cpp_function_name, envir = moments_env, mode = "function", inherits = FALSE)) {
                  return(list(moments_function = NULL, error = trimws(as.character(load_result))))
              }
              return(list(moments_function = get(cpp_function_name, envir = moments_env, inherits = FALSE), error = NA_character_))
          }
          cache_dir <-  getOption("NicoStan_src_extra_cache_dir", default = file.path(tools::R_user_dir(package = "NicoStan", which = "cache"), "src_extra"))
          loaded <-  fn_source_cpp_into(cache_dir)
          if (is.null(loaded$moments_function)) {
              process_cache_dir <-  file.path(cache_dir, paste0("process_", Sys.getpid()))
              message(colourise(paste0("tau_initial = \"adaptive\": ", cpp_file, " did not load from the shared cache ", cache_dir, " (",
                                       loaded$error, "); compiling it again into ", process_cache_dir, "."), "cyan"))
              loaded <-  fn_source_cpp_into(process_cache_dir)
          }
          if (is.null(loaded$moments_function)) {
              message(colourise(paste0("tau_initial = \"adaptive\": ", cpp_file, " did not compile or load (",
                                       loaded$error, "); the nuisance block has no estimate."), "red"))
              return(NULL)
          }
          options(NicoStan_tau_initial_nuisance_moments_loaded = list(cpp_file = cpp_file,
                                                                     cpp_file_md5 = cpp_file_md5,
                                                                     moments_function = loaded$moments_function))
          message(colourise(paste0("tau_initial = \"adaptive\": nuisance accumulator loaded from ", cpp_file), "cyan"))
          return(loaded$moments_function)

  }
  ##
  fn_tau_initial_from_eigen_max <-  function( eigen_max,
                                              block_label) {

          if (length(eigen_max) == 1 && is.finite(eigen_max) && eigen_max > 0) return(0.5 * pi * sqrt(eigen_max))
          message(colourise(paste0("tau_initial = \"adaptive\": lambda_max (", block_label, ") = ", signif(eigen_max, 5),
                                   " is not a positive finite number; tau falls back to pi."), "red"))
          return(pi)

  }
  EHMC_args_as_Rcpp_List$record_kinetic_energy_tau_derivatives <- FALSE
  tau_adaptation_iteration_vec <- rep(x = NA_real_, times = n_burnin)
  ##
  ## ---- tau ADAM updates actually PERFORMED, per block (drives the ADAM bias correction only):
  ##      tau_adaptation_iteration also advances on iterations where the update is skipped (no chain with
  ##      alpha > 0 and a positive criterion, or every chain divergent) and tau / the moments are left
  ##      unchanged, so it cannot be the bias-correction exponent. Reset wherever the tau ADAM moments are.
  ##
  tau_adam_updates_performed_by_block <- list(main = 0,
                                              us   = 0)
  tau_adam_bias_correction_step_vec <- rep(x = NA_real_, times = n_burnin)
  
  print(paste("clip_iter_tau = ", clip_iter_tau))
  
  ii <- 1
  
  # lp_grad_outs_AD <- BayesMVP:::Rcpp_wrapper_fn_lp_grad(Model_type = Model_type,
  #                                                          force_autodiff = TRUE,
  #                                                          force_PartialLog = FALSE,
  #                                                          multi_attempts = FALSE,
  #                                                          theta_main_vec = theta_vec_mean,
  #                                                          theta_us_vec   =  theta_us_vec,
  #                                                          y = y,
  #                                                          grad_option = "all",
  #                                                          Model_args_as_Rcpp_List = Model_args_as_Rcpp_List)
  # ##
  # print(paste("lp_grad_outs_AD = ", head(lp_grad_outs_AD)))
  # ##
  # lp_grad_outs_MD <- BayesMVP:::Rcpp_wrapper_fn_lp_grad(Model_type = Model_type,
  #                                                          force_autodiff = FALSE,
  #                                                          force_PartialLog = FALSE,
  #                                                          multi_attempts = FALSE,
  #                                                          theta_main_vec = theta_vec_mean,
  #                                                          theta_us_vec   = theta_us_vec,
  #                                                          y = y,
  #                                                          grad_option = "all",
  #                                                          Model_args_as_Rcpp_List = Model_args_as_Rcpp_List)
  # ##
  # print(paste("lp_grad_outs_MD = ", head(lp_grad_outs_MD)))
  # ##
  # print(paste("diff_sum = ", sum(lp_grad_outs_AD - lp_grad_outs_MD)))
  # ##
  # print(paste("diffs (head) = ", head(lp_grad_outs_AD - lp_grad_outs_MD, 100)))
  # ##
  # print(paste("diffs (grad of last 100 params) = ", head( tail((lp_grad_outs_AD - lp_grad_outs_MD), N + 100) , 100)  ))
  # ##
  # print(paste("AD (grad of last 100 params) = ", head( tail((lp_grad_outs_AD), N + 200) , 200)  ))
  # ##
  # print(paste("MD (grad of last 100 params) = ", head( tail((lp_grad_outs_MD), N + 200) , 200)  ))
  
  
  t_cpp_total <- 0
  t_push_total <- 0
  t_tau <- 0
  ##
  ## The separate profiling entry point leaves the normal native return format unchanged.
  ## No timing records are allocated when profiling is disabled.
  ##
  burnin_profile <- NULL
  run_burnin_iteration <- fn_persistent_burnin_run_one_iter
  if (debug_burnin_timing) {
      run_burnin_iteration <- fn_persistent_burnin_run_one_iter_profiled
      burnin_iteration_profiles <- vector(mode = "list", length = last_burnin_iteration)
      burnin_chain_profiles <- vector(mode = "list", length = last_burnin_iteration)
  }
  
  ##
  ## ---- share_tau_ii_across_chains_in_burnin: read by the C++ burn-in worker from this list, which is pushed to it every iteration:
  ##
  EHMC_args_as_Rcpp_List$share_tau_ii_across_chains <- isTRUE(share_tau_ii_across_chains_in_burnin)
  EHMC_args_as_Rcpp_List$randomize_tau <- randomize_tau_burnin
  ##
  ## ---- resident burn-in interface: the nuisance-sized state and statistics stay inside the worker between iterations,
  ##      and only main-sized quantities cross to R every iteration. Used only when the compiled code that creates the
  ##      worker provides it (the functions are looked up in the namespace of fn_create_persistent_burnin_worker, i.e. the
  ##      package whose compiled code owns the worker); otherwise the original interface, unchanged.
  ##      options(NicoStan_resident_burnin = FALSE) keeps the original interface; options(NicoStan_resident_burnin_joint_block = FALSE)
  ##      keeps it for tau_adaptation_block = "joint" only.
  ##
  worker_api_env <-  environment(fn_create_persistent_burnin_worker)
  tau_jitter_burnin_state <-  fn_tau_jitter_burnin_init(
                                tau_jitter_burnin = tau_jitter_burnin,
                                randomize_tau_burnin = randomize_tau_burnin,
                                seed = seed,
                                n_chains = n_chains_burnin,
                                share_tau_ii_across_chains = isTRUE(share_tau_ii_across_chains_in_burnin),
                                partitioned_HMC = partitioned_HMC)
  if (!is.null(tau_jitter_burnin_state)) {
        run_burnin_iteration_tau_jitter <-  fn_tau_jitter_burnin_worker_api(worker_api_env)
  }
  resident_api_names <-  c("fn_persistent_burnin_resident_api_version",
                           "fn_persistent_burnin_run_one_iter_main_only",
                           "fn_persistent_burnin_run_one_iter_main_only_profiled",
                           "fn_persistent_burnin_update_adaptation_main_only",
                           "fn_persistent_burnin_init_resident_statistics",
                           "fn_persistent_burnin_get_resident_statistic",
                           "fn_persistent_burnin_set_resident_statistic",
                           "fn_persistent_burnin_get_state",
                           "fn_persistent_burnin_state_row_means_resident",
                           "fn_persistent_burnin_pooled_welford_nuisance_resident",
                           "fn_persistent_burnin_update_snaper_m_and_s_resident",
                           "fn_persistent_burnin_update_snaper_m_prop_resident",
                           "fn_persistent_burnin_set_nuisance_centre_resident")
  ## ---- the joint (main + nuisance) trajectory block (tau_adaptation_block = "joint"):
  resident_joint_api_names <-  c("fn_persistent_burnin_resident_joint_api_version",
                                 "fn_persistent_burnin_joint_direction_set",
                                 "fn_persistent_burnin_joint_direction_get",
                                 "fn_persistent_burnin_joint_direction_transport_resident",
                                 "fn_persistent_burnin_joint_direction_update_snaper_resident",
                                 "fn_persistent_burnin_joint_position_reductions_resident")
  resident_joint_two_ended_api_name <-  "fn_persistent_burnin_joint_position_two_ended_reductions_resident"
  ##
  fn_worker_api_has_functions <-  function(function_names) {
        all(vapply(X = function_names,
                   FUN = function(name) exists(x = name, envir = worker_api_env, mode = "function", inherits = FALSE),
                   FUN.VALUE = logical(1)))
  }
  resident_api_available <-  fn_worker_api_has_functions(resident_api_names)
  if (resident_api_available) {
        resident_api <-  mget(x = resident_api_names, envir = worker_api_env, inherits = FALSE)
        resident_api_available <-  isTRUE(tryCatch(resident_api$fn_persistent_burnin_resident_api_version() >= 1,
                                                   error = function(e) FALSE))
  }
  resident_joint_api_available <-  resident_api_available && fn_worker_api_has_functions(resident_joint_api_names)
  if (resident_joint_api_available) {
        resident_api <-  c(resident_api, mget(x = resident_joint_api_names, envir = worker_api_env, inherits = FALSE))
        resident_joint_api_available <-  isTRUE(tryCatch(resident_api$fn_persistent_burnin_resident_joint_api_version() >= 1,
                                                         error = function(e) FALSE))
  }
  resident_joint_two_ended_api_available <-  resident_joint_api_available &&
                                             fn_worker_api_has_functions(resident_joint_two_ended_api_name)
  if (resident_joint_two_ended_api_available) {
        resident_api[[resident_joint_two_ended_api_name]] <-  get(resident_joint_two_ended_api_name,
                                                                  envir = worker_api_env,
                                                                  inherits = FALSE)
  }
  ##
  use_resident_burnin <-  resident_api_available &&
                          isTRUE(getOption("NicoStan_resident_burnin", TRUE)) &&
                          isTRUE(sample_nuisance) && (n_nuisance > 0) &&
                          (identical(tau_adaptation_block_effective, "main") ||
                           (identical(tau_adaptation_block_effective, "joint") && resident_joint_api_available &&
                            isTRUE(getOption("NicoStan_resident_burnin_joint_block", TRUE)))) &&
                          identical(getOption("matprod", "default"), "default")
  ## the joint trajectory block on the resident interface (every nuisance-sized quantity of its criterion is computed in the worker):
  use_resident_joint_block <-  use_resident_burnin && identical(tau_adaptation_block_effective, "joint")
  if (use_resident_joint_block && identical(tau_gradient_estimator, "two_ended") && !resident_joint_two_ended_api_available) {
        stop("tau_gradient_estimator = 'two_ended' with tau_adaptation_block = 'joint' needs the resident two-ended reduction helper; rebuild the native NicoStan worker.")
  }
  ##
  ## ---- tau_initial = "adaptive" on the resident interface: the nuisance block's running n_tests x n_tests sum and its count
  ##      (see tau_initial_moments above) are kept in the worker, which reads its own nuisance state and centre
  ##      (fn_persistent_burnin_tau_initial_nuisance_moments_*_resident, chains in parallel), so the nuisance state is not
  ##      copied to R at each iteration of [clip_iter, gap] and the separate accumulator is not loaded. Used only when the
  ##      worker's compiled code provides these functions; otherwise (an older build, the original interface, or
  ##      options(NicoStan_tau_initial_resident_moments = FALSE)) the state is copied and inst/src_extra/tau_initial_nuisance_moments.cpp
  ##      is used, as before. The two give the same sum up to the order in which the chains are added (rounding only).
  ##
  resident_tau_initial_api_names <-  c("fn_persistent_burnin_tau_initial_api_version",
                                       "fn_persistent_burnin_tau_initial_nuisance_moments_reset_resident",
                                       "fn_persistent_burnin_tau_initial_nuisance_moments_accumulate_resident",
                                       "fn_persistent_burnin_tau_initial_nuisance_moments_get_resident")
  use_resident_tau_initial_moments <-  FALSE
  tau_initial_resident_moments_reset <-  FALSE
  if (tau_initial_adaptive && use_resident_burnin && isTRUE(getOption("NicoStan_tau_initial_resident_moments", TRUE)) &&
      fn_worker_api_has_functions(resident_tau_initial_api_names)) {
        resident_tau_initial_api <-  mget(x = resident_tau_initial_api_names, envir = worker_api_env, inherits = FALSE)
        use_resident_tau_initial_moments <-  isTRUE(tryCatch(resident_tau_initial_api$fn_persistent_burnin_tau_initial_api_version() >= 1,
                                                             error = function(e) FALSE))
  }
  if (tau_initial_adaptive && isTRUE(tau_initial_moments$nuisance_sampled)) {
        message(colourise(paste0("tau_initial = \"adaptive\": nuisance moments accumulated ",
                                 if (use_resident_tau_initial_moments) "inside the burn-in worker (no copy of the nuisance state)" else
                                                                       "by the separate compiled accumulator (nuisance state copied to R)"), "cyan"))
  }
  ##
  if (use_resident_burnin) {
        run_burnin_iteration <-  if (debug_burnin_timing) resident_api$fn_persistent_burnin_run_one_iter_main_only_profiled else
                                                          resident_api$fn_persistent_burnin_run_one_iter_main_only
  }
  message(colourise(paste0("burn-in state transfer: ",
                            if (use_resident_burnin) "resident (main-sized quantities per iteration)" else "original (full state per iteration)"),
                     "cyan"))
  ##
  worker_ptr <- fn_create_persistent_burnin_worker(
                n_threads_R = n_chains_burnin,
                partitioned_HMC_R = partitioned_HMC,
                diffusion_HMC_R = diffusion_HMC,
                Model_type_R = Model_type,
                sample_nuisance_R = sample_nuisance,
                force_autodiff_R = force_autodiff,
                force_PartialLog_R = force_PartialLog,
                multi_attempts_R = multi_attempts,
                y_Eigen_R = y,
                Model_args_as_Rcpp_List = Model_args_as_Rcpp_List,   ## final version, incl. n_nuisance / bools edits
                EHMC_args_as_Rcpp_List = EHMC_args_as_Rcpp_List,
                EHMC_Metric_as_Rcpp_List = EHMC_Metric_as_Rcpp_List,
                n_threads_WCP = n_threads_WCP)

  ## Load initial theta into the resident state (ONCE):
  fn_persistent_burnin_set_theta(
                worker_ptr,
                theta_main_vectors_all_chains_input_from_R,
                theta_us_vectors_all_chains_input_from_R)

  ## IMPORTANT: worker_ptr must be created AFTER all edits to Model_args_as_Rcpp_List
  ## (n_nuisance, n_params_main, Model_args_bools[15,1] etc.) -- the worker snapshots
  ## Model_args at construction and never re-reads it.
  #   
  # if (is.null(metric_start_iter)) metric_start_iter <- max(clip_iter, round(0.35 * n_adapt))
  # wf_n  <- 0
  # wf_m  <- rep(0, n_params)                          # running mean of the DRAWS (all chains pooled)
  # wf_M2 <- rep(0, n_params)                          # running sum of squared deviations, per coordinate
  # wf_C2 <- matrix(0, n_params_main, n_params_main)   # running cross-products, main block
  # empicical_cov_main <- diag(rep(1, n_params_main))
  # var_draws_all      <- rep(1, n_params)
  # metric_ready       <- FALSE
  ##
  if (is.null(metric_start_iter)) metric_start_iter <- round(n_burnin/10)     # same start as old ii_min
  ## metric_window_resets <- round(c(0.30, 0.60) * n_adapt)                      # discard early, drift-contaminated draws
  metric_window_resets <- fn_resolve_metric_pooled_window_reset_iterations( metric_pooled_window_resets = metric_pooled_window_resets,
                                                                            n_adapt                     = n_adapt)
  if (metric_estimator == "pooled") {
      message(colourise(paste0("pooled metric: draws from iteration ", metric_start_iter, " to ", metric_adaptation_end_iter,
                               if (length(metric_window_resets) == 0) " in one window (no reset)" else
                                   paste0(", accumulators reset at the start of iteration(s) ", paste(metric_window_resets + 1, collapse = ", ")),
                               " | off-diagonal shrinkage = ", metric_pooled_offdiagonal_shrinkage),
                        "cyan"))
  }
  if (metric_estimator == "per_iteration") {
      message(colourise(paste0("per-iteration metric: cross-chain (co)variance of each iteration's draws, iterations ", metric_start_iter,
                               " to ", metric_adaptation_end_iter, ", blended into the metric with ratio_M (no accumulation, no window resets)",
                               if (identical(metric_shape_main, "dense")) paste0(" | off-diagonal shrinkage = ", metric_pooled_offdiagonal_shrinkage,
                                                                                  if (identical(metric_pooled_offdiagonal_shrinkage, "adaptive") &&
                                                                                      !metric_pooled_offdiagonal_shrinkage_adaptive_active)
                                                                                      paste0(" (not applied: ", metric_pooled_offdiagonal_shrinkage_resolution$reason, ")") else "") else ""),
                        "cyan"))
  }
  ## wf_min_draws: the pooled estimator updates the metric once a window holds this many draws (chains x
  ## iterations), whatever the metric shape.
  ## Until 4 Oct 2026 this was n_params_main + 1 (the draws needed for a full-rank p x p covariance), which with 4
  ## burn-in chains and the stan_style window resets is never reached for a model with more main parameters than
  ## 4 x the window length: the LC-MVOP (141 main parameters) at b125 ran its whole burn-in and sampling with the
  ## unit metric (eps 0.005, 359 leapfrog steps per draw). Enzo, 4 Oct 2026: "this seems like a dumb rule to me.
  ## just delete it". Two draws are all a variance needs (the nuisance pooled estimator already uses 2); the dense
  ## covariance stays usable through the existing ridge (w = n / (n + 5) towards 1e-3 I) and the ratio_M blending.
  wf_min_draws <- 2
  ##
  wf_n  <- 0
  wf_m  <- rep(0, n_params)
  wf_M2 <- rep(0, n_params)
  wf_C2 <- matrix(0, n_params_main, n_params_main)
  empicical_cov_main <- diag(rep(1, n_params_main))
  var_draws_all      <- rep(1, n_params)
  ##
  ## per-iteration metric estimator (metric_estimator = "per_iteration"): the main-parameter variances of the current iteration's
  ## draws (the diagonal main metric reads them only once metric_ready is TRUE, by which time they have been computed), and the
  ## number of draws that have entered the per-iteration proposals so far (the n of the Stan-style regularisation of the dense proposal):
  per_iteration_variance_vec_main          <- rep(1, n_params_main)
  per_iteration_metric_n_draws_since_start <- 0
  adaptive_metric_shrinkage_state <- NULL
  adaptive_metric_shrinkage_last_diagnostics <- NULL
  adaptive_metric_shrinkage_lambda_history <- rep(NA_real_, n_burnin)
  adaptive_metric_shrinkage_window_updates <- function(window_start) {
        next_reset <- metric_window_resets[metric_window_resets >= window_start]
        window_end <- if (length(next_reset) > 0) next_reset[1] else metric_adaptation_end_iter
        return(max(1, window_end - window_start + 1))
  }
  adaptive_metric_shrinkage_per_iteration_updates <- function() {
        ii_min_for_metric <- round(n_burnin / 10)
        ii_max_for_metric <- n_adapt
        first_update <- max(metric_start_iter, ii_min_for_metric + 1)
        last_update <- min(metric_adaptation_end_iter, ii_max_for_metric - 1)
        if (first_update > last_update) return(1)
        update_iterations <- seq.int(first_update, last_update)
        n_updates <- sum(update_iterations %% interval_width_main == 0)
        if (identical(burnin_schedule, "automatic") && metric_adaptation_end_iter > ii_min_for_metric &&
            metric_adaptation_end_iter < ii_max_for_metric && metric_adaptation_end_iter %% interval_width_main != 0) {
              n_updates <- n_updates + 1
        }
        return(max(1, n_updates))
  }
  if (metric_pooled_offdiagonal_shrinkage_adaptive_active) {
        n_window_updates_initial <- if (identical(metric_estimator, "pooled")) {
              adaptive_metric_shrinkage_window_updates(metric_start_iter)
        } else {
              adaptive_metric_shrinkage_per_iteration_updates()
        }
        adaptive_metric_shrinkage_state <- fn_init_adaptive_metric_shrinkage(
              n_main = n_params_main,
              n_chains = n_chains_burnin,
              metric_estimator = metric_estimator,
              n_window_updates = n_window_updates_initial,
              initial_covariance = EHMC_Metric_as_Rcpp_List$M_inv_dense_main)
  }
  ##
  metric_ready <- (metric_estimator %in% c("chain_mean", "chain_mean_scaled"))   # pooled: wait for draws; chain_mean(_scaled): behave exactly as before
  ##
  if (use_resident_burnin) {
        ##
        ## ---- the nuisance-sized statistics move into the worker, with exactly the initial values set above:
        resident_api$fn_persistent_burnin_init_resident_statistics( worker_ptr,
                                                                    snaper_m_vec_us = EHMC_burnin_as_Rcpp_List$snaper_m_vec_us,
                                                                    snaper_s_vec_us_empirical = EHMC_burnin_as_Rcpp_List$snaper_s_vec_us_empirical,
                                                                    snaper_m_prop_vec_us = snaper_m_prop_vec_all[index_nuisance],
                                                                    var_draws_us = var_draws_all[index_nuisance])
        ##
        ## ---- R keeps only the main rows of the pooled estimator (every operation of it is row-local):
        wf_m  <-  wf_m[index_main]
        wf_M2 <-  wf_M2[index_main]
        ##
        ## ---- the joint SNAPER direction (tau_adaptation_block = "joint") moves into the worker too:
        if (use_resident_joint_block) {
              resident_api$fn_persistent_burnin_joint_direction_set(worker_ptr, trajectory_direction$joint)
        }
        ##
        ## ---- the nuisance state is resident: any use of these would now be stale, so a missed use raises an error instead
        ##      (inside the loop's try() blocks that error is printed and the burn-in continues, so check the console of an A/B run):
        theta_us_vectors_all_chains_input_from_R <-  NULL
        velocity_us_vectors_all_chains_input_from_R <-  NULL
        ##
  }
  
  ## ---- HOLD THE ADAM LEARNING RATE HIGH TO START (learning_rate_initial) ----------------
  ##
  ## The ADAM learning rate drives BOTH the step-size (eps) and the path-length (tau)
  ## adaptation. Starting it high lets the adaptation move a long way in the first few
  ## iterations instead of creeping; it is then dropped back to the run's normal learning rate
  ## for the rest of the burnin, so the later, finer adaptation is not made noisy by a big rate.
  ##
  ## learning_rate_initial_iter defaults to n_burnin / 10 - i.e. the 50 iterations that work at
  ## n_burnin = 500, scaled proportionally to any other burnin length.
  ##
  learning_rate_main_after_hold <- LR_main
  learning_rate_us_after_hold   <- LR_us
  ##
  if (is.null(learning_rate_initial_iter)) {
    learning_rate_initial_iter <- round(n_burnin / 10)
  }
  learning_rate_initial_iter <- max(0L, min(as.integer(round(learning_rate_initial_iter)), as.integer(n_burnin)))
  ##
  use_learning_rate_hold <- (!is.null(learning_rate_initial)) && (learning_rate_initial_iter >= 1L)
  ##
  if (use_learning_rate_hold) {

        if (!is.numeric(learning_rate_initial) || length(learning_rate_initial) != 1L ||
            !is.finite(learning_rate_initial) || learning_rate_initial <= 0) {
          stop("'learning_rate_initial' must be NULL or a single positive finite number.")
        }
        ##
        message(paste0( "learning-rate hold: LR held at ", signif(learning_rate_initial, 4),
                        " for the first ", learning_rate_initial_iter, " of ", n_burnin,
                        " burnin iterations, then set to ", signif(learning_rate_main_after_hold, 4),
                        " (the run's learning_rate) for the remaining ",
                        n_burnin - learning_rate_initial_iter, "."))
        ##
        ## ---- the hold only does something if it is ABOVE the rate it hands back to. Equal values
        ## are almost always an accident (a learning_rate that was raised to the hold value, or a
        ## hold left at the default), and silently doing nothing is the wrong answer:
        ##
        if (learning_rate_initial <= learning_rate_main_after_hold) {

              warning(paste0(
                          "learning_rate_initial (", trimws(formatC(x = learning_rate_initial, format = "g", digits = 4, width = 0)),
                          ") is NOT above the run's learning_rate (",
                          trimws(formatC(x = learning_rate_main_after_hold, format = "g", digits = 4, width = 0)),
                          "), so the learning-rate hold ",
                          if (isTRUE(all.equal(learning_rate_initial, learning_rate_main_after_hold)))
                            "changes NOTHING - the rate is the same before and after. Either raise learning_rate_initial, or lower learning_rate so there is something to hand back down to."
                          else
                            "RAISES the rate after the hold rather than lowering it. Either raise learning_rate_initial, or lower learning_rate so there is something to hand back down to."),
                       call. = FALSE,
                       immediate. = TRUE)

        }

  }
  ##
  ## ---- Per-iteration constants, built once here before the loop (the same values at every iteration; R copies an
  ##      object before any in-place change, so the list elements that share them cannot alter them):
  ##        - the unit nuisance metric (metric_type_nuisance = "unit"), one vector per list element;
  ##        - the zero centre theta_hat_us_vec (theta_hat_us_rule = "zero", and every iteration up to clip_iter).
  ##
  unit_M_inv_us_vec     <- rep(1, n_nuisance)
  unit_M_us_vec         <- rep(1, n_nuisance)
  unit_sqrt_M_us_vec    <- rep(1, n_nuisance)
  theta_hat_us_vec_zero <- matrix(rep(0.0, n_nuisance))
  ##
  ## ---- Proposal-weighted centre (snaper_m_prop_vec_all): with tau_adaptation_block = "main" the trajectory criterion reads
  ##      only its MAIN rows, so from iteration 2 on only those rows are updated (see the proposal-centre update in the loop).
  ##      proposal_entry_limit is half the largest double: two entries below it in absolute value cannot overflow when added.
  ##      The main rows of the matrix-vector product are the same with or without the nuisance rows for R's reference BLAS
  ##      (libRblas: each row is accumulated on its own over the columns, in the same order whatever the number of rows).
  ##      Optimised BLAS kernels (OpenBLAS, MKL, Accelerate/vecLib) may treat rows differently by position, so with any
  ##      other BLAS, or one R does not name, the full update is kept (normalizePath() follows a libRblas link to another BLAS).
  ##
  blas_library_file <- tryCatch(basename(normalizePath(extSoftVersion()[["BLAS"]], mustWork = FALSE)),
                                error = function(e) "")
  blas_is_reference_Rblas <- isTRUE(grepl("^libRblas(\\.0)?\\.(so|dylib)$", blas_library_file))
  ##
  proposal_centre_main_rows_only <- isTRUE(sample_nuisance) && (n_nuisance > 0) && identical(tau_adaptation_block_effective, "main") &&
                                    blas_is_reference_Rblas
  proposal_entry_limit <- 0.5 * .Machine$double.xmax
  ##
 ####  Start burnin   ------------------------------------------------------------------------------------------------------------------------------------------------
 for (ii in iter_seq_burnin) {

              ## ---- learning-rate hold. Nothing else writes LR_main / LR_us inside the loop, so
              ## setting them here, at the top, is what every adaptation in THIS iteration sees:
              if (use_learning_rate_hold) {

                    if (ii <= learning_rate_initial_iter) {
                          EHMC_burnin_as_Rcpp_List$LR_main <- learning_rate_initial
                          EHMC_burnin_as_Rcpp_List$LR_us   <- learning_rate_initial
                    } else {
                          EHMC_burnin_as_Rcpp_List$LR_main <- learning_rate_main_after_hold
                          EHMC_burnin_as_Rcpp_List$LR_us   <- learning_rate_us_after_hold
                          ##
                          if (ii == learning_rate_initial_iter + 1L) {
                            message(paste0( "learning-rate hold finished at iteration ", ii, ": LR ",
                                            signif(learning_rate_initial, 4), " -> ",
                                            signif(learning_rate_main_after_hold, 4),
                                            if (learning_rate_initial <= learning_rate_main_after_hold)
                                              "  (NOT a reduction - see the warning at the start of the burnin)"
                                            else ""))
                          }
                    }

              }
   
              if (metric_type_nuisance == "unit") { 
               
                   #### ---- for unit metric --------------------------------------
                   ## (the unit vectors are built once, before the loop)
                   EHMC_Metric_as_Rcpp_List$M_inv_us_vec  <- unit_M_inv_us_vec
                   EHMC_Metric_as_Rcpp_List$M_us_vec      <- unit_M_us_vec
                   EHMC_burnin_as_Rcpp_List$sqrt_M_us_vec <- unit_sqrt_M_us_vec
               
              }
                        
              if (ii %% n_refresh_iter_counter == 0) {
                    print(ii)
              }
            
              # if (ii %% round(n_burnin/2) == 0) { 
              #       gc() ## ; gc()
              # }
              
              ## "all" starts this accumulation at iteration 1; numeric fractions use the historical window below.
              ## ---- tau_initial = "adaptive": accumulate the moments behind lambda_max (see tau_initial_moments above) on [clip_iter, gap],
              ##      BEFORE the handover below, so that iteration gap reads the states at the start of that iteration. It only reads
              ##      the chains, centres and metric: no sampler state, adaptation or random number is changed, so the burn-in up to
              ##      the handover is the same as with tau_initial = pi.
              ## if (tau_initial_adaptive && !isTRUE(manual_tau) && ii >= clip_iter && ii <= gap) {
  if (tau_initial_adaptive && !isTRUE(manual_tau) && ii >= tau_initial_moments_start_iter && ii <= gap) {

                tau_initial_moments_t0 <-  proc.time()[[3]]
                tau_initial_loader_seconds_ii <-  0
                tau_initial_moments_try <-  try({

                      tau_initial_moments$n_updates <-  tau_initial_moments$n_updates + 1
                      ##
                      ## main block: the pooled (Welford) mean and cross-product sum of the main parameter vectors of all chains:
                      tau_initial_moments$main <-  fn_tau_initial_welford_update( moments = tau_initial_moments$main,
                                                                                  X = theta_main_vectors_all_chains_input_from_R)
                      ##
                      ## nuisance block: the n_tests x n_tests sum of y_i t(y_i), y = (u - snaper_m_vec_us) / sqrt(M_inv_us_vec), over the
                      ## individuals and chains; on the resident interface the states and the centre are read from the worker:
                      if (tau_initial_moments$nuisance_sampled && !tau_initial_moments$us$loader_failed) {
                          if (is.null(tau_initial_moments$us$positions)) {
                              tau_initial_nuisance_layout <-  fn_tau_initial_nuisance_positions()
                              tau_initial_moments$us$positions <-  tau_initial_nuisance_layout$positions
                              tau_initial_moments$us$layout    <-  tau_initial_nuisance_layout$layout
                          }
                          ## on the worker's accumulator (use_resident_tau_initial_moments): the layout is given to the worker once, and the
                          ## worker adds this iteration from its own theta_us and snaper_m_vec_us; only the scale is passed from R:
                          if (use_resident_tau_initial_moments) {
                              if (!tau_initial_resident_moments_reset) {
                                  resident_tau_initial_api$fn_persistent_burnin_tau_initial_nuisance_moments_reset_resident( worker_ptr,
                                                                                                                            tau_initial_moments$us$positions)
                                  tau_initial_resident_moments_reset <-  TRUE
                              }
                              if (!identical(tau_initial_moments$us$scale_source, EHMC_Metric_as_Rcpp_List$M_inv_us_vec)) {
                                  tau_initial_moments$us$scale_source <-  EHMC_Metric_as_Rcpp_List$M_inv_us_vec
                                  tau_initial_moments$us$scale        <-  sqrt(as.numeric(EHMC_Metric_as_Rcpp_List$M_inv_us_vec))
                              }
                              resident_tau_initial_api$fn_persistent_burnin_tau_initial_nuisance_moments_accumulate_resident( worker_ptr,
                                                                                                                             "snaper_m_vec_us",
                                                                                                                             tau_initial_moments$us$scale)
                          }
                          ## if (is.null(tau_initial_moments$us$moments_function)) {
                          if (!use_resident_tau_initial_moments && is.null(tau_initial_moments$us$moments_function)) {
                              tau_initial_loader_t0 <-  proc.time()[[3]]
                              tau_initial_moments$us$moments_function <-  fn_tau_initial_nuisance_moments_function()
                              tau_initial_moments$us$loader_failed    <-  is.null(tau_initial_moments$us$moments_function)
                              tau_initial_loader_seconds_ii <-  proc.time()[[3]] - tau_initial_loader_t0
                          }
                          ## if (!tau_initial_moments$us$loader_failed) {
                          if (!use_resident_tau_initial_moments && !tau_initial_moments$us$loader_failed) {
                              tau_initial_us_states <-  if (use_resident_burnin) resident_api$fn_persistent_burnin_get_state(worker_ptr, "theta_us_vectors_all_chains_output_to_R") else
                                                                                 theta_us_vectors_all_chains_input_from_R
                              tau_initial_us_centre <-  if (use_resident_burnin) resident_api$fn_persistent_burnin_get_resident_statistic(worker_ptr, "snaper_m_vec_us") else
                                                                                 EHMC_burnin_as_Rcpp_List$snaper_m_vec_us
                              ## s = sqrt(M_inv_us_vec), recomputed only when the nuisance metric has changed (every change assigns a new
                              ## vector, so an unchanged metric is the same object and identical() returns at once):
                              if (!identical(tau_initial_moments$us$scale_source, EHMC_Metric_as_Rcpp_List$M_inv_us_vec)) {
                                  tau_initial_moments$us$scale_source <-  EHMC_Metric_as_Rcpp_List$M_inv_us_vec
                                  tau_initial_moments$us$scale        <-  sqrt(as.numeric(EHMC_Metric_as_Rcpp_List$M_inv_us_vec))
                              }
                              tau_initial_us_sum <-  tau_initial_moments$us$moments_function( as.matrix(tau_initial_us_states),
                                                                                             as.numeric(tau_initial_us_centre),
                                                                                             tau_initial_moments$us$scale,
                                                                                             tau_initial_moments$us$positions)
                              tau_initial_moments$us$sum   <-  if (is.null(tau_initial_moments$us$sum)) tau_initial_us_sum else tau_initial_moments$us$sum + tau_initial_us_sum
                              tau_initial_moments$us$count <-  tau_initial_moments$us$count + nrow(tau_initial_moments$us$positions) * ncol(as.matrix(tau_initial_us_states))
                              rm(tau_initial_us_states, tau_initial_us_centre, tau_initial_us_sum)
                          }
                      }

                })
                if (inherits(tau_initial_moments_try, "try-error")) tau_initial_moments$n_errors <-  tau_initial_moments$n_errors + 1
                tau_initial_moments$seconds_loader  <-  tau_initial_moments$seconds_loader + tau_initial_loader_seconds_ii
                tau_initial_moments$seconds_moments <-  tau_initial_moments$seconds_moments +
                                                        (proc.time()[[3]] - tau_initial_moments_t0 - tau_initial_loader_seconds_ii)

              }

              if (manual_tau == FALSE) {
                      
                try({

                      ## Initialise ONCE. Repeated resets here used to discard already-learned tau values.
                      if (ii == gap) {

                            EHMC_burnin_as_Rcpp_List$tau_m_adam_main <- 0
                            EHMC_burnin_as_Rcpp_List$tau_v_adam_main <- 0
                            EHMC_burnin_as_Rcpp_List$tau_m_adam_us <- 0
                            EHMC_burnin_as_Rcpp_List$tau_v_adam_us <- 0
                            ## the moments are reset, so the performed-update counters restart with them:
                            tau_adam_updates_performed_by_block$main <- 0
                            tau_adam_updates_performed_by_block$us   <- 0
                            ChEES_criterion_ema <- NA_real_
                            ESJD_CHESSR_component_criterion_ema <-  c(ESJD = NA_real_, CHESSR = NA_real_)
                            ESJD_SNAPER_component_criterion_ema <-  c(ESJD = NA_real_, SNAPER = NA_real_)
                            LQ_ESSR_component_criterion_ema <-  NULL

                            ## ---- tau_initial = "adaptive": tau = (pi/2) * sqrt(lambda_max) per block (see tau_initial_moments above), the main
                            ##      mass being the one in force now. The joint sampler (partitioned_HMC = FALSE) has one tau for both blocks,
                            ##      so it takes the larger of the two, i.e. (pi/2) * sqrt(max(lambda_max_main, lambda_max_us)). A block whose
                            ##      estimate is missing, not finite or not positive uses pi (the numeric path with tau_initial = pi).
                            ##      With tau_adaptation_block "main" the joint sampler's tau comes from the main block alone.
                            if (tau_initial_adaptive) {

                                  tau_initial_handover_t0 <-  proc.time()[[3]]
                                  ## on the worker's accumulator, the running sum and count are read from the worker (one n_tests x n_tests matrix
                                  ## and one number); without a term the block keeps no estimate, as before:
                                  if (use_resident_tau_initial_moments && tau_initial_resident_moments_reset) {
                                      tau_initial_resident_moments <-  tryCatch(resident_tau_initial_api$fn_persistent_burnin_tau_initial_nuisance_moments_get_resident(worker_ptr),
                                                                               error = function(e) { message(colourise(paste0("tau_initial = \"adaptive\": reading the nuisance moments from the worker failed: ",
                                                                                                                              conditionMessage(e)), "red")) ; NULL })
                                      if (!is.null(tau_initial_resident_moments) && (tau_initial_resident_moments$count > 0)) {
                                          tau_initial_moments$us$sum   <-  tau_initial_resident_moments$sum
                                          tau_initial_moments$us$count <-  tau_initial_resident_moments$count
                                      }
                                  }
                                  tau_initial_mass_main <-  if (metric_shape_main == "diag") 1 / c(EHMC_Metric_as_Rcpp_List$M_inv_main_vec) else
                                                                                            EHMC_Metric_as_Rcpp_List$M_dense_main
                                  lambda_max_main_handover <-  tryCatch(fn_tau_initial_lambda_max_main(tau_initial_moments$main, tau_initial_mass_main),
                                                                        error = function(e) { message(colourise(paste0("tau_initial = \"adaptive\": lambda_max (main) failed: ",
                                                                                                                       conditionMessage(e)), "red")) ; NA_real_ })
                                  lambda_max_us_handover   <-  if (!tau_initial_moments$nuisance_sampled) NA_real_ else
                                                               tryCatch(fn_tau_initial_lambda_max_us(tau_initial_moments$us),
                                                                        error = function(e) { message(colourise(paste0("tau_initial = \"adaptive\": lambda_max (nuisance) failed: ",
                                                                                                                       conditionMessage(e)), "red")) ; NA_real_ })
                                  ## the Marchenko-Pastur upper edge (1 + sqrt(d / n))^2: the lambda_max that sampling noise alone gives for n
                                  ## independent draws with covariance I in metric coordinates (diagnostic, not used for tau):
                                  lambda_max_main_noise_edge <-  (1 + sqrt(n_params_main / max(1, tau_initial_moments$main$n)))^2
                                  tau_initial_moments$seconds_handover <-  proc.time()[[3]] - tau_initial_handover_t0
                                  ##
                                  tau_block_main_handover <-  fn_tau_initial_from_eigen_max(lambda_max_main_handover, "main")
                                  ## a nuisance block that is not sampled has no estimate: its tau is then set as by the numeric path with pi.
                                  tau_block_us_handover   <-  if (tau_initial_moments$nuisance_sampled) fn_tau_initial_from_eigen_max(lambda_max_us_handover, "nuisance") else
                                                                                                        tau_initial
                                  lambda_max_joint_handover <-  if (tau_initial_moments$nuisance_sampled) max(lambda_max_main_handover, lambda_max_us_handover) else
                                                                                                          lambda_max_main_handover
                                  ##
                                  ## the joint sampler (partitioned_HMC = FALSE) has one tau: from the main block when tau is adapted on the main
                                  ## block (tau_adaptation_block "main"), from the larger of the two blocks when it is adapted on both ("joint"):
                                  tau_initial_uses_nuisance_block <-  (partitioned_HMC == FALSE) && identical(tau_adaptation_block_effective, "joint") &&
                                                                      isTRUE(tau_initial_moments$nuisance_sampled)
                                  tau_initial_tau_source <-  if (partitioned_HMC == TRUE) "each block its own" else
                                                             if (tau_initial_uses_nuisance_block) "larger of main and nuisance" else "main block"
                                  ## tau_main_adaptive_handover <-  if (partitioned_HMC == TRUE) tau_block_main_handover else
                                  ##                                if (tau_initial_moments$nuisance_sampled) max(tau_block_main_handover, tau_block_us_handover) else
                                  ##                                                                          tau_block_main_handover
                                  tau_main_adaptive_handover <-  if (partitioned_HMC == TRUE) tau_block_main_handover else
                                                                 if (tau_initial_uses_nuisance_block) max(tau_block_main_handover, tau_block_us_handover) else
                                                                                                      tau_block_main_handover
                                  tau_us_adaptive_handover   <-  if (partitioned_HMC == TRUE) tau_block_us_handover else tau_main_adaptive_handover
                                  ##
                                  ## ---- LQ_ESSR: its criterion has several local optima in tau (the squared statistic oscillates with tau even under
                                  ##      the uniform jitter), and the first one is the global one. (pi/2) sqrt(lambda_max) is the fixed-length optimum
                                  ##      of the slowest direction for ChEES, which on a Gaussian lies in the basin of LQ_ESSR's SECOND optimum, so
                                  ##      gradient ascent stops there. LQ_ESSR is therefore handed over at its own unit-Gaussian optimum for the
                                  ##      slowest direction, 1.0696 sqrt(lambda_max) (t ~ U(0, 2 tau), exact dynamics), i.e. the factor 1.0696 / (pi/2):
                                  if (identical(burnin_algorithm, "LQ_ESSR")) {
                                        tau_main_adaptive_handover <-  tau_main_adaptive_handover * 1.0696 / (pi / 2)
                                        tau_us_adaptive_handover <-  tau_us_adaptive_handover * 1.0696 / (pi / 2)
                                  }
                                  ##
                                  message(colourise(paste0("tau_initial = \"adaptive\" handover (iteration ", ii, ", ", tau_initial_moments$n_updates,
                                                           " iterations of moments, ", tau_initial_moments$main$n, " main draws): lambda_max main = ",
                                                           signif(lambda_max_main_handover, 5),
                                                           ", nuisance = ", if (tau_initial_moments$nuisance_sampled) signif(lambda_max_us_handover, 5) else "not sampled",
                                                           if (partitioned_HMC == TRUE) "" else paste0(", joint (larger) = ", signif(lambda_max_joint_handover, 5)),
                                                           "; tau from: ", tau_initial_tau_source, " (tau_adaptation_block = ", tau_adaptation_block_effective, ")",
                                                           " -> tau_main = ", signif(tau_main_adaptive_handover, 5),
                                                           ", tau_us = ", signif(tau_us_adaptive_handover, 5)), "cyan"))
                                  message(colourise(paste0("tau_initial = \"adaptive\" diagnostics (not used for tau): main noise edge (1 + sqrt(d/n))^2 = ",
                                                           signif(lambda_max_main_noise_edge, 5), " (d = ", n_params_main, ", n = ", tau_initial_moments$main$n, ")",
                                                           "; nuisance layout = ", tau_initial_moments$us$layout,
                                                           " (", if (is.null(tau_initial_moments$us$positions)) NA else ncol(tau_initial_moments$us$positions), " per individual, ",
                                                           tau_initial_moments$us$count, " terms)",
                                                           "; errors = ", tau_initial_moments$n_errors,
                                                           "; seconds (inside time_burnin): moments = ", signif(tau_initial_moments$seconds_moments, 4),
                                                           ", loader = ", signif(tau_initial_moments$seconds_loader, 4),
                                                           ", handover = ", signif(tau_initial_moments$seconds_handover, 4)), "cyan"))
                                  if (tau_main_adaptive_handover > max_tau_main || tau_us_adaptive_handover > max_tau_us) {
                                      message(colourise(paste0("tau_initial = \"adaptive\": handover tau exceeds its ceiling (tau_main = ", signif(tau_main_adaptive_handover, 5),
                                                               " vs max_tau_main = ", max_tau_main, "; tau_us = ", signif(tau_us_adaptive_handover, 5),
                                                               " vs max_tau_us = ", max_tau_us, "); the ceiling is applied later in this iteration."), "red"))
                                  }

                            }

                            if (partitioned_HMC == TRUE) {

                                        ## For main:
                                        ## or: ## tau_mult * sqrt(EHMC_burnin_as_Rcpp_List$eigen_max_main)
                                        tau_main_prop <-    tau_initial
                                        if (tau_initial_adaptive) tau_main_prop <-  tau_main_adaptive_handover
                                        EHMC_args_as_Rcpp_List$tau_main  <-   if_not_NA_or_INF_else(tau_main_prop, 5.0 *   EHMC_args_as_Rcpp_List$eps_main)
                                        message(paste("tau handover: main =", EHMC_args_as_Rcpp_List$tau_main))

                                        ## For nuisance:
                                        tau_us_prop <-   tau_initial
                                        if (tau_initial_adaptive) tau_us_prop <-  tau_us_adaptive_handover
                                        EHMC_args_as_Rcpp_List$tau_us  <-   if_not_NA_or_INF_else(tau_us_prop, 5.0 * EHMC_args_as_Rcpp_List$eps_us)
                                        message(paste("tau handover: nuisance =", EHMC_args_as_Rcpp_List$tau_us))

                            } else if (partitioned_HMC == FALSE) {

                                        ## For ALL:
                                        tau_main_prop <- tau_initial
                                        if (tau_initial_adaptive) tau_main_prop <-  tau_main_adaptive_handover
                                        EHMC_args_as_Rcpp_List$tau_main <-   if_not_NA_or_INF_else(tau_main_prop, 5.0 * EHMC_args_as_Rcpp_List$eps_main)
                                        message(paste("tau handover: joint =", EHMC_args_as_Rcpp_List$tau_main))
                                        ##
                                        # EHMC_args_as_Rcpp_List$tau_main <- EHMC_args_as_Rcpp_List$tau_main
                                        EHMC_args_as_Rcpp_List$tau_us   <- EHMC_args_as_Rcpp_List$tau_main
                            }

                            ## ---- tau adaptation scheme "probe_then_average": the probe starts at tau_handover / 8 (rung -3), with fresh ADAM moments:
                            if (tau_probe_then_average) {
                                  tau_probe_state$log_tau_start <-  log(EHMC_args_as_Rcpp_List$tau_main)
                                  tau_probe_state$probing <-  TRUE
                                  tau_probe_state$rung <-  tau_probe_rung_start
                                  tau_probe_state$gradient_block_sum <-  0
                                  tau_probe_state$component_block_sums[] <-  0
                                  tau_probe_state$n_in_block <-  0
                                  ## "fixed_length_probe_then_decay_and_average": fixed-length trajectories (tau_ii = tau) for the probe's updates:
                                  if (tau_probe_fixed_length_decay) EHMC_args_as_Rcpp_List$randomize_tau <-  FALSE
                                  EHMC_args_as_Rcpp_List$tau_main <-  exp(tau_probe_state$log_tau_start + tau_probe_state$rung * log(2))
                                  if (partitioned_HMC == FALSE) EHMC_args_as_Rcpp_List$tau_us <-  EHMC_args_as_Rcpp_List$tau_main
                                  message(paste("tau probe start (tau_handover / 8): main =", EHMC_args_as_Rcpp_List$tau_main))
                            }

                            if (tau_initial_adaptive) {
                                  ## tau as set here; the max_tau_main / max_tau_us ceilings are applied later in this iteration.
                                  tau_initial_adaptive_record <-  list( estimator = "pooled_welford_main_and_within_individual_nuisance",
                                                                        handover_iteration = ii,
                                                                        clip_iter = clip_iter,
                                                                        moments_start_iter = tau_initial_moments_start_iter,
                                                                        moments_window = if (tau_initial_moments_all) "all" else "after_clip_iter",
                                                                        moments_window_fraction = if (tau_initial_moments_all) NA_real_ else tau_initial_moments_window_fraction,
                                                                        moments_window_min_iter = tau_initial_moments_window_min_iter,
                                                                        ## tau ADAM options of this burn-in (NA = beta1_adam for tau; the restart iteration is NA when there was no restart):
                                                                        tau_adam_beta1_option = if (is.null(tau_adam_beta1_option)) NA_real_ else as.numeric(tau_adam_beta1_option),
                                                                        tau_adam_beta1_used = tau_adam_beta1,
                                                                        tau_learning_rate_restart_at_metric_end = tau_learning_rate_restart_at_metric_end,
                                                                        tau_learning_rate_restart_iteration = if (tau_learning_rate_restart_active) tau_learning_rate_restart_iter else NA_real_,
                                                                        n_updates = tau_initial_moments$n_updates,
                                                                        n_errors = tau_initial_moments$n_errors,
                                                                        partitioned_HMC = partitioned_HMC,
                                                                        nuisance_sampled = tau_initial_moments$nuisance_sampled,
                                                                        ##
                                                                        lambda_max_main = lambda_max_main_handover,
                                                                        lambda_max_us = lambda_max_us_handover,
                                                                        lambda_max_joint = if (partitioned_HMC == TRUE) NA_real_ else lambda_max_joint_handover,
                                                                        tau_adaptation_block = tau_adaptation_block_effective,
                                                                        tau_source = tau_initial_tau_source,
                                                                        ##
                                                                        dimension_main = n_params_main,
                                                                        n_draws_main = tau_initial_moments$main$n,
                                                                        lambda_max_main_noise_edge = lambda_max_main_noise_edge,
                                                                        dimension_us = if (tau_initial_moments$nuisance_sampled) n_nuisance else 0,
                                                                        nuisance_layout = tau_initial_moments$us$layout,
                                                                        nuisance_accumulator = if (!tau_initial_moments$nuisance_sampled) NA_character_ else
                                                                                               if (use_resident_tau_initial_moments) "worker" else "standalone",
                                                                        n_tests_us = if (is.null(tau_initial_moments$us$positions)) NA_real_ else ncol(tau_initial_moments$us$positions),
                                                                        n_terms_us = tau_initial_moments$us$count,
                                                                        second_moment_us = if (is.null(tau_initial_moments$us$sum) || !(tau_initial_moments$us$count > 0)) NULL else
                                                                                               tau_initial_moments$us$sum / tau_initial_moments$us$count,
                                                                        ##
                                                                        tau_block_main = tau_block_main_handover,
                                                                        tau_block_us = tau_block_us_handover,
                                                                        tau_main_handover = EHMC_args_as_Rcpp_List$tau_main,
                                                                        tau_us_handover = EHMC_args_as_Rcpp_List$tau_us,
                                                                        max_tau_main = max_tau_main,
                                                                        max_tau_us = max_tau_us,
                                                                        tau_main_handover_capped = min(max_tau_main, EHMC_args_as_Rcpp_List$tau_main),
                                                                        tau_us_handover_capped = min(max_tau_us, EHMC_args_as_Rcpp_List$tau_us),
                                                                        ##
                                                                        ## elapsed seconds inside time_burnin (see tau_initial_moments above):
                                                                        seconds_moments = tau_initial_moments$seconds_moments,
                                                                        seconds_loader = tau_initial_moments$seconds_loader,
                                                                        seconds_handover = tau_initial_moments$seconds_handover)
                            }

                    }
                })
                
              }
                
               if (manual_tau == TRUE) {

                     ## In L units the target is a LEAPFROG-STEP COUNT, so tau is rebuilt from the
                     ## step size as it currently stands. eps keeps adapting to adapt_delta; only the
                     ## trajectory length is pinned.
                     if (isTRUE(tau_if_manual_in_L_units)) {
                           tau_main_manual <- tau_if_manual[1] * EHMC_args_as_Rcpp_List$eps_main
                           tau_us_manual   <- if (length(tau_if_manual) >= 2)
                                                  tau_if_manual[2] * EHMC_args_as_Rcpp_List$eps_us else tau_main_manual
                     } else {
                           tau_main_manual <- tau_if_manual[1]
                           tau_us_manual   <- if (length(tau_if_manual) >= 2) tau_if_manual[2] else tau_if_manual[1]
                     }
                     ##
                     if (length(tau_if_manual) == 1) {

                       EHMC_args_as_Rcpp_List$tau_main <- tau_main_manual
                       EHMC_args_as_Rcpp_List$tau_us   <- tau_main_manual
                       if (partitioned_HMC == FALSE) EHMC_args_as_Rcpp_List$tau_main <- tau_main_manual

                     } else if (length(tau_if_manual) == 2) {

                       EHMC_args_as_Rcpp_List$tau_main <- tau_main_manual
                       EHMC_args_as_Rcpp_List$tau_us   <- tau_us_manual
                       if (partitioned_HMC == FALSE) EHMC_args_as_Rcpp_List$tau_main <- tau_main_manual

                     }
                 
               }
   
                try({
                     if (!isTRUE(manual_tau) && ii >= clip_iter && ii < gap) {

                             ## Position within the actual ramp window; its final iteration reaches the last stage.
                             tau_ramp_position <- (ii - clip_iter) / max(1, gap - clip_iter - 1)
    
                             if (identical(tau_ramp, "original")) {

                             ## ---- Original four-stage shape: 5*eps, then 1/4, 1/2 and all of tau_initial.
                             ## Historically these were fractions of absolute gap and hard-coded pi, skipping
                             ## the first stage and jumping at handover whenever tau_initial was not pi.
                             if (tau_ramp_position < 0.25)   {

                                     tau_main_prop  <-  min(tau_initial, 5.0 * EHMC_args_as_Rcpp_List$eps_main)
                                     tau_us_prop  <-    min(tau_initial, 5.0 * EHMC_args_as_Rcpp_List$eps_us)
                                     tau_overall_prop <- tau_main_prop

                             } else if (tau_ramp_position < 0.5) {

                                     tau_main_prop  <-  min(tau_initial, max(0.25 * tau_initial, 5.0 * EHMC_args_as_Rcpp_List$eps_main))
                                     tau_us_prop  <-    min(tau_initial, max(0.25 * tau_initial, 5.0 * EHMC_args_as_Rcpp_List$eps_us))
                                     tau_overall_prop <- tau_main_prop

                             } else if (tau_ramp_position < 0.75) {

                                     tau_main_prop  <-  min(tau_initial, max(0.5 * tau_initial, 5.0 * EHMC_args_as_Rcpp_List$eps_main))
                                     tau_us_prop  <-    min(tau_initial, max(0.5 * tau_initial, 5.0 * EHMC_args_as_Rcpp_List$eps_us))
                                     tau_overall_prop <- tau_main_prop

                             } else { 
                               
                                     tau_main_prop  <-  tau_initial
                                     tau_us_prop  <-    tau_initial
                                     tau_overall_prop <- tau_main_prop
                               
                             }
         
                             if (partitioned_HMC == TRUE) {

                                     EHMC_args_as_Rcpp_List$tau_main  <-   if_not_NA_or_INF_else(tau_main_prop, 5.0 * EHMC_args_as_Rcpp_List$eps_main)
                                     EHMC_args_as_Rcpp_List$tau_us    <-   if_not_NA_or_INF_else(tau_us_prop,   5.0 * EHMC_args_as_Rcpp_List$eps_us)

                             } else if (partitioned_HMC == FALSE) {

                                     EHMC_args_as_Rcpp_List$tau_main <- if_not_NA_or_INF_else(tau_main_prop, 5.0 * EHMC_args_as_Rcpp_List$eps_main)
                                     ##
                                     EHMC_args_as_Rcpp_List$tau_main <- EHMC_args_as_Rcpp_List$tau_main
                                     EHMC_args_as_Rcpp_List$tau_us   <- EHMC_args_as_Rcpp_List$tau_main

                             }

                             } else { ## tau_ramp == "staged"

                             ## ---- tau ramp across the clip window [clip_iter, gap).
                             ##
                             ## Short trajectories first - a handful of leapfrog steps, i.e.
                             ## tau = L * eps - lengthening, then fixed fractions of a period.
                             ## Each stage gets an equal slice of the window. To change the ramp,
                             ## edit the two vectors below; nothing else needs touching.
                             ##
                             ## gap is the absolute handover iteration (clip_iter_tau), not an added duration.
                             ## Both ramps use the window itself and the same user-selected endpoint.
                             ##
                             tau_ramp_leapfrog_steps <- c(5, 10, 20)                   ## tau = L * eps
                             tau_ramp_fixed_tau <- tau_initial * c(0.125, 0.25, 0.50, 1.00)
                             ##
                             n_tau_ramp_stages <- length(tau_ramp_leapfrog_steps) + length(tau_ramp_fixed_tau)
                             ##
                             ## 0 at clip_iter, approaching 1 at gap:
                             ## tau_ramp_position was computed above for both ramp shapes.
                             ##
                             tau_ramp_stage <- min( n_tau_ramp_stages,
                                                    1 + floor(tau_ramp_position * n_tau_ramp_stages))
                             ##
                             if (tau_ramp_stage <= length(tau_ramp_leapfrog_steps)) {

                                     tau_main_prop <- min(tau_initial, tau_ramp_leapfrog_steps[tau_ramp_stage] * EHMC_args_as_Rcpp_List$eps_main)
                                     tau_us_prop   <- min(tau_initial, tau_ramp_leapfrog_steps[tau_ramp_stage] * EHMC_args_as_Rcpp_List$eps_us)

                             } else {

                                     ## The leapfrog stages scale with eps; the fixed stages do not.
                                     ## If eps has grown enough that the last leapfrog stage already
                                     ## exceeds the first fixed stage, the ramp would SHORTEN at the
                                     ## handover (e.g. 20 * eps = 1.00 dropping to 0.125 * pi = 0.39
                                     ## at eps = 0.05). Floor the fixed stages at where the leapfrog
                                     ## phase ended so the ramp only ever lengthens; the fixed targets
                                     ## take over again as soon as they exceed that floor.
                                     ## (Restored: this floor was active in every earlier run;
                                     ## 09-18 10:21; removing it changed the burn-in relative to those runs.)
                                     ##
                                     tau_ramp_floor <- max(tau_ramp_leapfrog_steps) * EHMC_args_as_Rcpp_List$eps_main
                                     ##
                                     tau_main_prop <- min(tau_initial, max(tau_ramp_fixed_tau[tau_ramp_stage - length(tau_ramp_leapfrog_steps)],
                                                                          tau_ramp_floor))
                                     tau_us_prop <- min(tau_initial, max(tau_ramp_fixed_tau[tau_ramp_stage - length(tau_ramp_leapfrog_steps)],
                                                                        max(tau_ramp_leapfrog_steps) * EHMC_args_as_Rcpp_List$eps_us))

                             }
                             ##
                             tau_overall_prop <- tau_main_prop
         
                             if (partitioned_HMC == TRUE) {

                                     EHMC_args_as_Rcpp_List$tau_main  <-   if_not_NA_or_INF_else(tau_main_prop, 5.0 * EHMC_args_as_Rcpp_List$eps_main)
                                     EHMC_args_as_Rcpp_List$tau_us    <-   if_not_NA_or_INF_else(tau_us_prop,   5.0 * EHMC_args_as_Rcpp_List$eps_us)

                             } else if (partitioned_HMC == FALSE) {

                                     ## the fallback here used to be 5 * tau_main, which compounds the
                                     ## CURRENT path length rather than falling back to a short one;
                                     ## matched to the partitioned branch (5 * eps_main):
                                     EHMC_args_as_Rcpp_List$tau_main <- if_not_NA_or_INF_else(tau_main_prop, 5.0 * EHMC_args_as_Rcpp_List$eps_main)
                                     ##
                                     EHMC_args_as_Rcpp_List$tau_us   <- EHMC_args_as_Rcpp_List$tau_main

                             }

                             } ## end tau_ramp switch
    
                     }
                })
              # }
    

          
               ## ---- Re-assert the manual trajectory length.
               ##
               ## The tau ramp above fires on ii in [clip_iter, gap) and is NOT conditioned on
               ## manual_tau, so without this a manual run spends the whole ramp window at the ramp's
               ## tau rather than the requested one - and in L units the requested tau also has to be
               ## rebuilt here because eps has moved since it was set at the top of the iteration.
               ## Placed BEFORE the max_tau clip so that clip still applies, and before the
               ## ii < clip_iter rule so the L = 1 warm start still happens.
               if (manual_tau == TRUE) {
                     if (isTRUE(tau_if_manual_in_L_units)) {
                           EHMC_args_as_Rcpp_List$tau_main <- tau_if_manual[1] * EHMC_args_as_Rcpp_List$eps_main
                           EHMC_args_as_Rcpp_List$tau_us   <- if (length(tau_if_manual) >= 2)
                                                                  tau_if_manual[2] * EHMC_args_as_Rcpp_List$eps_us else
                                                                  EHMC_args_as_Rcpp_List$tau_main
                     } else {
                           EHMC_args_as_Rcpp_List$tau_main <- tau_if_manual[1]
                           EHMC_args_as_Rcpp_List$tau_us   <- if (length(tau_if_manual) >= 2)
                                                                  tau_if_manual[2] else tau_if_manual[1]
                     }
                     if (partitioned_HMC == FALSE) EHMC_args_as_Rcpp_List$tau_us <- EHMC_args_as_Rcpp_List$tau_main
               }
               if ( EHMC_args_as_Rcpp_List$tau_main  > max_tau_main) {
                 EHMC_args_as_Rcpp_List$tau_main  <- max_tau_main
               }
               if ( EHMC_args_as_Rcpp_List$tau_us  > max_tau_us) {
                 EHMC_args_as_Rcpp_List$tau_us  <- max_tau_us
               }
               if (partitioned_HMC == FALSE) { 
                   if ( EHMC_args_as_Rcpp_List$tau_main  > max_tau_main) {
                     EHMC_args_as_Rcpp_List$tau_main  <- max_tau_main
                   }
               }
               ## Make sure this comes last in the tau adaptation above:
               if (ii < clip_iter) {
                     EHMC_args_as_Rcpp_List$tau_us  <-    1 * EHMC_args_as_Rcpp_List$eps_us;
                     EHMC_args_as_Rcpp_List$tau_main  <-  1 * EHMC_args_as_Rcpp_List$eps_main;
                     if (partitioned_HMC == FALSE) EHMC_args_as_Rcpp_List$tau_main <- 1 * EHMC_args_as_Rcpp_List$eps_main;
               }
   
               #### ---------------------------------------------------------------------------------------------------------------------------------------------------------------
               #### current mean theta across all K chains:
               if (use_resident_burnin) {
                        ## the nuisance part (and theta_vec_current_mean / theta_vec_current_us) is formed in the worker, where it is used;
                        ## rowMeans() of the main block is exactly theta_vec_current_mean[index_main] of the full mean:
                        theta_vec_current_main <-  rowMeans(theta_main_vectors_all_chains_input_from_R)
               } else {
               if (sample_nuisance == TRUE) { 
                        ## same values as rowMeans(rbind(us, main)), without copying the (n_nuisance + n_main) x K matrix every iteration:
                        theta_vec_current_mean <- c(rowMeans(theta_us_vectors_all_chains_input_from_R), rowMeans(theta_main_vectors_all_chains_input_from_R))
                        ## only the partitioned sampler's nuisance update reads this nuisance-length copy:
                        if (partitioned_HMC == TRUE) theta_vec_current_us <-   theta_vec_current_mean[index_nuisance]
                        theta_vec_current_main <- theta_vec_current_mean[index_main]
               } else { 
                        theta_vec_current_mean <- rowMeans(rbind(theta_main_vectors_all_chains_input_from_R))
                        theta_vec_current_main <- theta_vec_current_mean
               }
               }  ## end of: if (use_resident_burnin)
               # ##
               # if (ii >= metric_start_iter) {
               #     
               #          X_all <- if (sample_nuisance) rbind(theta_us_vectors_all_chains_input_from_R,
               #                                              theta_main_vectors_all_chains_input_from_R)
               #                   else theta_main_vectors_all_chains_input_from_R                       # n_params x K
               #          K     <- ncol(X_all)
               #          m_b   <- rowMeans(X_all)                                  # ONE common mean this iteration, same for every chain
               #          D_b   <- X_all - m_b
               #          delta <- m_b - wf_m
               #          n_new <- wf_n + K
               #          wf_M2 <- wf_M2 + rowSums(D_b^2) + delta^2 * (wf_n * K / n_new)
               #          wf_C2 <- wf_C2 + tcrossprod(D_b[index_main, , drop = FALSE]) +
               #                           tcrossprod(delta[index_main]) * (wf_n * K / n_new)
               #          wf_m  <- wf_m + delta * (K / n_new)
               #          wf_n  <- n_new
               #          ##
               #          if (wf_n >= 2 * n_params_main) {                          # need > p draws for a usable 44x44
               #              var_draws_all <- wf_M2 / (wf_n - 1)
               #              cov_draws     <- wf_C2 / (wf_n - 1)
               #              w <- wf_n / (wf_n + 5)                                # Stan-style regularisation, keeps it PD early on
               #              empicical_cov_main <- w * cov_draws + (1 - w) * 1e-3 * diag(n_params_main)
               #              empicical_cov_main <- 0.5 * (empicical_cov_main + t(empicical_cov_main))
               #              metric_ready <- TRUE
               #          }
               #      
               # }
               ## The pooled moments (var_draws_all, empicical_cov_main, metric_ready) are read only by the metric updates,
               ## all at or before metric_adaptation_end_iter, so they are not accumulated after it:
               if (use_resident_burnin) {
                    if ((metric_estimator == "pooled") && (ii >= metric_start_iter) && (ii <= metric_adaptation_end_iter)) {
                         ##
                         ## ---- the same update: the main rows here (the same code on the main rows only; every operation of it is row-local),
                         ##      the nuisance rows of wf_m, wf_M2 and var_draws_all in the worker:
                         window_reset_now <-  ii %in% (metric_window_resets + 1)
                         ## if (window_reset_now) { wf_n <- 0; wf_m[] <- 0; wf_M2[] <- 0; wf_C2[] <- 0 }
                         ## after a window reset the metric is held (no blending towards the previous window's proposal) until the new window has wf_min_draws draws:
                         if (window_reset_now) {
                               wf_n <- 0; wf_m[] <- 0; wf_M2[] <- 0; wf_C2[] <- 0; metric_ready <- FALSE
                               if (metric_pooled_offdiagonal_shrinkage_adaptive_active) {
                                     adaptive_metric_shrinkage_state <- fn_reset_adaptive_metric_shrinkage(
                                           state = adaptive_metric_shrinkage_state,
                                           n_window_updates = adaptive_metric_shrinkage_window_updates(ii))
                               }
                         }
                         wf_n_before_update <-  wf_n
                         ##
                         X_main <- theta_main_vectors_all_chains_input_from_R
                         K     <- ncol(X_main)
                         m_b   <- rowMeans(X_main)
                         D_b   <- X_main - m_b
                         delta <- m_b - wf_m
                         n_new <- wf_n + K
                         wf_M2 <- wf_M2 + rowSums(D_b^2) + delta^2 * (wf_n * K / n_new)
                         wf_C2 <- wf_C2 + tcrossprod(D_b) +
                                          tcrossprod(delta) * (wf_n * K / n_new)
                         wf_m  <- wf_m + delta * (K / n_new)
                         wf_n  <- n_new
                         if (metric_pooled_offdiagonal_shrinkage_adaptive_active) {
                               adaptive_metric_shrinkage_state <- fn_update_adaptive_metric_shrinkage(
                                     state = adaptive_metric_shrinkage_state,
                                     main_draws = X_main,
                                     covariance_proposal = NULL,
                                     ratio_M_effective = NULL)
                         }
                         ##
                         n_new_resident <-  resident_api$fn_persistent_burnin_pooled_welford_nuisance_resident( worker_ptr,
                                                                                                               wf_n_R = wf_n_before_update,
                                                                                                               reset_R = window_reset_now,
                                                                                                               wf_min_draws_R = wf_min_draws)
                         if (!identical(n_new_resident, wf_n)) stop("BUG: the resident pooled-estimator count differs from wf_n.")
                         ##
                         if (wf_n >= wf_min_draws) {
                             cov_draws     <- wf_C2 / (wf_n - 1)
                             w <- wf_n / (wf_n + 5)
                             empicical_cov_main <- w * cov_draws + (1 - w) * 1e-3 * diag(n_params_main)
                             if (metric_pooled_offdiagonal_shrinkage_adaptive_active) {
                                   adaptive_shrinkage_resolution <- fn_resolve_adaptive_metric_shrinkage(
                                         state = adaptive_metric_shrinkage_state,
                                         covariance_matrix = empicical_cov_main)
                                   empicical_cov_main <- adaptive_shrinkage_resolution$covariance_matrix
                                   adaptive_metric_shrinkage_lambda_history[ii] <- adaptive_shrinkage_resolution$shrinkage
                                   adaptive_metric_shrinkage_last_diagnostics <- adaptive_shrinkage_resolution$diagnostics
                             } else {
                                   ## off-diagonals x (1 - metric_pooled_offdiagonal_shrinkage), diagonal kept (0 = unchanged):
                                   empicical_cov_main <- if (identical(metric_pooled_offdiagonal_shrinkage, "adaptive")) {
                                         fn_shrink_metric_pooled_offdiagonal( covariance_matrix                   = empicical_cov_main,
                                                                              metric_pooled_offdiagonal_shrinkage = metric_pooled_offdiagonal_shrinkage_numeric)
                                   } else {
                                         fn_shrink_metric_pooled_offdiagonal( covariance_matrix                   = empicical_cov_main,
                                                                              metric_pooled_offdiagonal_shrinkage = metric_pooled_offdiagonal_shrinkage)
                                   }
                             }
                             empicical_cov_main <- 0.5 * (empicical_cov_main + t(empicical_cov_main))
                             metric_ready <- TRUE
                         }
                    }
               } else {
               if ((metric_estimator == "pooled") && (ii >= metric_start_iter) && (ii <= metric_adaptation_end_iter)) {
                    ## if (ii %in% (metric_window_resets + 1)) { wf_n <- 0; wf_m[] <- 0; wf_M2[] <- 0; wf_C2[] <- 0 }
                    ## after a window reset the metric is held (no blending towards the previous window's proposal) until the new window has wf_min_draws draws:
                    if (ii %in% (metric_window_resets + 1)) {
                          wf_n <- 0; wf_m[] <- 0; wf_M2[] <- 0; wf_C2[] <- 0; metric_ready <- FALSE
                          if (metric_pooled_offdiagonal_shrinkage_adaptive_active) {
                                adaptive_metric_shrinkage_state <- fn_reset_adaptive_metric_shrinkage(
                                      state = adaptive_metric_shrinkage_state,
                                      n_window_updates = adaptive_metric_shrinkage_window_updates(ii))
                          }
                    }
                    ##
                    X_all <- if (sample_nuisance) rbind(theta_us_vectors_all_chains_input_from_R,
                                                        theta_main_vectors_all_chains_input_from_R)
                             else theta_main_vectors_all_chains_input_from_R
                    K     <- ncol(X_all)
                    m_b   <- rowMeans(X_all)
                    D_b   <- X_all - m_b
                    delta <- m_b - wf_m
                    n_new <- wf_n + K
                    wf_M2 <- wf_M2 + rowSums(D_b^2) + delta^2 * (wf_n * K / n_new)
                    wf_C2 <- wf_C2 + tcrossprod(D_b[index_main, , drop = FALSE]) +
                                     tcrossprod(delta[index_main]) * (wf_n * K / n_new)
                    wf_m  <- wf_m + delta * (K / n_new)
                    wf_n  <- n_new
                    if (metric_pooled_offdiagonal_shrinkage_adaptive_active) {
                          adaptive_metric_shrinkage_state <- fn_update_adaptive_metric_shrinkage(
                                state = adaptive_metric_shrinkage_state,
                                main_draws = X_all[index_main, , drop = FALSE],
                                covariance_proposal = NULL,
                                ratio_M_effective = NULL)
                    }
                    ##
                    if (wf_n >= wf_min_draws) {
                        var_draws_all <- wf_M2 / (wf_n - 1)
                        cov_draws     <- wf_C2 / (wf_n - 1)
                        w <- wf_n / (wf_n + 5)
                        empicical_cov_main <- w * cov_draws + (1 - w) * 1e-3 * diag(n_params_main)
                        if (metric_pooled_offdiagonal_shrinkage_adaptive_active) {
                              adaptive_shrinkage_resolution <- fn_resolve_adaptive_metric_shrinkage(
                                    state = adaptive_metric_shrinkage_state,
                                    covariance_matrix = empicical_cov_main)
                              empicical_cov_main <- adaptive_shrinkage_resolution$covariance_matrix
                              adaptive_metric_shrinkage_lambda_history[ii] <- adaptive_shrinkage_resolution$shrinkage
                              adaptive_metric_shrinkage_last_diagnostics <- adaptive_shrinkage_resolution$diagnostics
                        } else {
                              ## off-diagonals x (1 - metric_pooled_offdiagonal_shrinkage), diagonal kept (0 = unchanged):
                              empicical_cov_main <- if (identical(metric_pooled_offdiagonal_shrinkage, "adaptive")) {
                                    fn_shrink_metric_pooled_offdiagonal( covariance_matrix                   = empicical_cov_main,
                                                                         metric_pooled_offdiagonal_shrinkage = metric_pooled_offdiagonal_shrinkage_numeric)
                              } else {
                                    fn_shrink_metric_pooled_offdiagonal( covariance_matrix                   = empicical_cov_main,
                                                                         metric_pooled_offdiagonal_shrinkage = metric_pooled_offdiagonal_shrinkage)
                              }
                        }
                        empicical_cov_main <- 0.5 * (empicical_cov_main + t(empicical_cov_main))
                        metric_ready <- TRUE
                    }
               }
               }  ## end of: if (use_resident_burnin)
               ##
               ## ---- per-iteration metric estimator (metric_estimator = "per_iteration"): at iteration ii the proposal is the
               ##      cross-chain (co)variance of THIS iteration's n_chains_burnin draws around their cross-chain mean (divisor
               ##      n_chains_burnin - 1), over the same iterations as the pooled moments (metric_start_iter to
               ##      metric_adaptation_end_iter). Nothing is accumulated across iterations, so there are no window resets; the
               ##      metric updates blend each proposal into the metric with ratio_M (and M_decay), as for every estimator:
               ##        main, diagonal   per_iteration_variance_vec_main (passed to update_M_Empirical_main);
               ##        main, dense      empicical_cov_main, with the Stan-style regularisation w * cov + (1 - w) * 1e-3 * I,
               ##                         w = n / (n + 5), n = the draws that have entered the per-iteration proposals so far,
               ##                         then the off-diagonal shrinkage of the pooled estimator. Each proposal has rank at most
               ##                         n_chains_burnin - 1 and w goes to 1 as n grows, so with metric_pooled_offdiagonal_shrinkage
               ##                         < 1 the dense metric is well conditioned only when n_chains_burnin - 1 >= n_params_main;
               ##        nuisance         var_draws_all[index_nuisance]; in the resident burn-in var_draws_us in the worker, from the
               ##                         pooled nuisance Welford update with its accumulators reset at every iteration (wf_n_R = 0).
               if ((metric_estimator == "per_iteration") && (ii >= metric_start_iter) && (ii <= metric_adaptation_end_iter)) {
                    ##
                    per_iteration_draws_main <- theta_main_vectors_all_chains_input_from_R
                    per_iteration_n_chains   <- ncol(per_iteration_draws_main)
                    ##
                    if (per_iteration_n_chains >= 2) {
                         ##
                         per_iteration_deviations_main   <- per_iteration_draws_main - rowMeans(per_iteration_draws_main)
                         per_iteration_variance_vec_main <- rowSums(per_iteration_deviations_main^2) / (per_iteration_n_chains - 1)
                         ##
                         if (metric_shape_main == "dense") {
                              per_iteration_metric_n_draws_since_start <- per_iteration_metric_n_draws_since_start + per_iteration_n_chains
                              per_iteration_covariance_main       <- tcrossprod(per_iteration_deviations_main) / (per_iteration_n_chains - 1)
                              per_iteration_regularisation_weight <- per_iteration_metric_n_draws_since_start / (per_iteration_metric_n_draws_since_start + 5)
                              empicical_cov_main <- per_iteration_regularisation_weight * per_iteration_covariance_main +
                                                    (1 - per_iteration_regularisation_weight) * 1e-3 * diag(n_params_main)
                              if (!metric_pooled_offdiagonal_shrinkage_adaptive_active) {
                                    ## off-diagonals x (1 - metric_pooled_offdiagonal_shrinkage), diagonal kept (0 = unchanged):
                                    empicical_cov_main <- if (identical(metric_pooled_offdiagonal_shrinkage, "adaptive")) {
                                          fn_shrink_metric_pooled_offdiagonal( covariance_matrix                   = empicical_cov_main,
                                                                               metric_pooled_offdiagonal_shrinkage = metric_pooled_offdiagonal_shrinkage_numeric)
                                    } else {
                                          fn_shrink_metric_pooled_offdiagonal( covariance_matrix                   = empicical_cov_main,
                                                                               metric_pooled_offdiagonal_shrinkage = metric_pooled_offdiagonal_shrinkage)
                                    }
                              }
                              empicical_cov_main <- 0.5 * (empicical_cov_main + t(empicical_cov_main))
                         }
                         ##
                         if ((sample_nuisance == TRUE) && (metric_type_nuisance %in% c("Empirical", "uniform_diag"))) {
                              if (use_resident_burnin) {
                                   ## wf_n_R = 0 and reset_R = TRUE: var_draws_us = rowSums(D_b^2) / (n_chains_burnin - 1) of this iteration's
                                   ## nuisance draws, D_b = draws - their cross-chain mean (wf_min_draws_R = 2: any 2 or more draws):
                                   per_iteration_n_draws_resident <- resident_api$fn_persistent_burnin_pooled_welford_nuisance_resident( worker_ptr,
                                                                                                                                       wf_n_R = 0,
                                                                                                                                       reset_R = TRUE,
                                                                                                                                       wf_min_draws_R = 2)
                                   if (per_iteration_n_draws_resident != per_iteration_n_chains) stop("BUG: the resident per-iteration draw count differs from n_chains_burnin.")
                              } else {
                                   per_iteration_deviations_nuisance <- theta_us_vectors_all_chains_input_from_R - rowMeans(theta_us_vectors_all_chains_input_from_R)
                                   var_draws_all[index_nuisance]     <- rowSums(per_iteration_deviations_nuisance^2) / (per_iteration_n_chains - 1)
                              }
                         }
                         ##
                         metric_ready <- TRUE
                    }
               }
               ##
               if (ii < 0.333333 * n_burnin)  {
                        shrinkage_factor <- 0.75
                        main_vec_for_Hessian <- theta_vec_current_main
               } else {
                        shrinkage_factor <- 0.25
                        main_vec_for_Hessian <- EHMC_burnin_as_Rcpp_List$snaper_m_vec_main
               }
              
            #### ////  updates for MAIN: --------------------------------------------------------------------------------------------------------------------
            if (partitioned_HMC == TRUE) {
                             ## update snaper_m and snaper_s_empirical (for MAIN):
                             try({
                               delta_old_main <- c(c(theta_vec_current_main) - c(EHMC_burnin_as_Rcpp_List$snaper_m_vec_main))
                               outs_update_snaper_m_and_s <- fn_update_snaper_m_and_s( EHMC_burnin_as_Rcpp_List$snaper_m_vec_main,
                                                                                                      EHMC_burnin_as_Rcpp_List$snaper_s_vec_main_empirical,
                                                                                                      theta_vec_current_main,
                                                                                                      ii)
                               EHMC_burnin_as_Rcpp_List$snaper_m_vec_main <- outs_update_snaper_m_and_s[,1]
                               EHMC_burnin_as_Rcpp_List$snaper_s_vec_main_empirical <- outs_update_snaper_m_and_s[,2]
                               delta_new_main <- c(c(theta_vec_current_main) - c(EHMC_burnin_as_Rcpp_List$snaper_m_vec_main))
                             })
            }
            #### ////  updates for NUISANCE: ----------------------------------------------------------------------------------------------------------------
            if (sample_nuisance == TRUE) {

                   if ((partitioned_HMC == TRUE) && use_resident_burnin) {
                              ## update snaper_m and snaper_s_empirical (for NUISANCE), in the worker (the same compiled update on the resident values):
                              try({
                                resident_api$fn_persistent_burnin_update_snaper_m_and_s_resident( worker_ptr,
                                                                                                  snaper_m_vec_main = EHMC_burnin_as_Rcpp_List$snaper_m_vec_main,
                                                                                                  snaper_s_vec_main_empirical = EHMC_burnin_as_Rcpp_List$snaper_s_vec_main_empirical,
                                                                                                  theta_vec_current_mean_main = theta_vec_current_main,
                                                                                                  ii_R = ii,
                                                                                                  joint_layout_R = FALSE)
                              })
                   } else {
                   if (partitioned_HMC == TRUE) { 
                    
                              ## update snaper_m and snaper_s_empirical (for NUISANCE):
                              try({
                                outs_update_snaper_m_and_s <-   fn_update_snaper_m_and_s( EHMC_burnin_as_Rcpp_List$snaper_m_vec_us,  
                                                                                                         EHMC_burnin_as_Rcpp_List$snaper_s_vec_us_empirical, 
                                                                                                         theta_vec_current_us, 
                                                                                                         ii)
                                EHMC_burnin_as_Rcpp_List$snaper_m_vec_us <- outs_update_snaper_m_and_s[,1]
                                EHMC_burnin_as_Rcpp_List$snaper_s_vec_us_empirical <- outs_update_snaper_m_and_s[,2]
                    
                              })
                   }
                   }  ## end of: if ((partitioned_HMC == TRUE) && use_resident_burnin)
            }
            #### ////  updates for ALL: --------------------------------------------------------------------------------------------------------------------
            if (partitioned_HMC == FALSE) {
              
                   if (use_resident_burnin) {
                        ##
                        ## ---- the same update in the worker: fn_update_snaper_m_and_s on c(nuisance, main), with the is.finite() fallback.
                        ##      The nuisance parts stay there; the main parts come back:
                        try({
                           resident_snaper_update <-  resident_api$fn_persistent_burnin_update_snaper_m_and_s_resident( worker_ptr,
                                                                                                                        snaper_m_vec_main = EHMC_burnin_as_Rcpp_List$snaper_m_vec_main,
                                                                                                                        snaper_s_vec_main_empirical = EHMC_burnin_as_Rcpp_List$snaper_s_vec_main_empirical,
                                                                                                                        theta_vec_current_mean_main = theta_vec_current_main,
                                                                                                                        ii_R = ii,
                                                                                                                        joint_layout_R = TRUE)
                           if (isTRUE(resident_snaper_update$non_finite_fallback)) {
                             cat("Warning: NaN in initial snaper_m_vec_all, using theta_vec_current_mean\n")
                           }
                           ## the main centre before the update (theta_vec_current_main when the fallback was taken):
                           snaper_m_vec_main_before_update <-  if (isTRUE(resident_snaper_update$non_finite_fallback)) theta_vec_current_main else
                                                                                                                         EHMC_burnin_as_Rcpp_List$snaper_m_vec_main
                           delta_old_main <- c(c(theta_vec_current_main) - snaper_m_vec_main_before_update)
                           EHMC_burnin_as_Rcpp_List$snaper_m_vec_main           <- resident_snaper_update$snaper_m_vec_main
                           EHMC_burnin_as_Rcpp_List$snaper_s_vec_main_empirical <- resident_snaper_update$snaper_s_vec_main_empirical
                           delta_new_main <- c(c(theta_vec_current_main) - c(EHMC_burnin_as_Rcpp_List$snaper_m_vec_main))
                        })
                   } else {
                   snaper_m_vec_all <- c(EHMC_burnin_as_Rcpp_List$snaper_m_vec_us, EHMC_burnin_as_Rcpp_List$snaper_m_vec_main)
                   # Check for NaN in initial values
                   ## (min() and max() reach any NA, NaN or +/-Inf, so this is the same test as any(!is.finite(snaper_m_vec_all))
                   ##  without two nuisance-length logical copies of the vector)
                   if ((length(snaper_m_vec_all) > 0) &&
                       !(is.finite(min(snaper_m_vec_all)) && is.finite(max(snaper_m_vec_all)))) {
                     cat("Warning: NaN in initial snaper_m_vec_all, using theta_vec_current_mean\n")
                     snaper_m_vec_all <- theta_vec_current_mean
                   }
                   snaper_s_vec_all_empirical <- c(EHMC_burnin_as_Rcpp_List$snaper_s_vec_us_empirical, 
                                                   EHMC_burnin_as_Rcpp_List$snaper_s_vec_main_empirical)
                  
                   ## update snaper_m and snaper_s_empirical (for MAIN):
                   try({
                      delta_old_main <- c(c(theta_vec_current_main) - snaper_m_vec_all[index_main])
                      outs_update_snaper_m_and_s <- fn_update_snaper_m_and_s( snaper_m = snaper_m_vec_all,
                                                                                             snaper_s_empirical = snaper_s_vec_all_empirical,
                                                                                             theta_vec_mean = theta_vec_current_mean,
                                                                                             ii = ii)
                      ## Each block is read straight from the returned (n_params x 2) matrix: the same values as extracting the
                      ## two full-length columns first and then subsetting them, without the two full-length copies.
                      snaper_m_vec_main_updated <- outs_update_snaper_m_and_s[index_main, 1]
                      delta_new_main <- c(c(theta_vec_current_main) - c(snaper_m_vec_main_updated))
                      ##
                      EHMC_burnin_as_Rcpp_List$snaper_m_vec_main           <- snaper_m_vec_main_updated
                      EHMC_burnin_as_Rcpp_List$snaper_s_vec_main_empirical <- outs_update_snaper_m_and_s[index_main, 2]
                      EHMC_burnin_as_Rcpp_List$snaper_m_vec_us             <- outs_update_snaper_m_and_s[index_nuisance, 1]
                      EHMC_burnin_as_Rcpp_List$snaper_s_vec_us_empirical   <- outs_update_snaper_m_and_s[index_nuisance, 2]
                   })
                   }  ## end of: if (use_resident_burnin)
                   ##
                   ## Appendix C keeps a proposal centre for z' based on the
                   ## acceptance-probability weighted proposal minibatch.
                   ## With realised accept/reject endpoints, the current mean
                   ## is the corresponding lower-variance-free fallback.
                   ## if (!isTRUE(manual_tau) && ii < n_adapt && burnin_algorithm %in% c("ChEES", "CHESSR", "CHESSR_log", "SNAPER")) {
                   ## if (!isTRUE(manual_tau) && ii < n_adapt && burnin_algorithm %in% c("ChEES", "CHESSR", "CHESSR_log", "SNAPER", "CHESSR_time", "SNAPER_time")) {
                   ## ("ESJD_CHESSR" reads the proposal centre through its CHESSR term. "ESJD" alone does not use it, since a jump
                   ##  ||z_t - z_0||^2 does not depend on any centre, but the centre is kept exactly as for CHESSR, so every array the
                   ##  criterion code receives, including the resident worker's copy, has the same shape and state as for CHESSR)
                   ## if (!isTRUE(manual_tau) && ii < n_adapt &&
                   ##     burnin_algorithm %in% c("ChEES", "CHESSR", "CHESSR_log", "SNAPER", "CHESSR_time", "SNAPER_time", "ESJD", "ESJD_CHESSR", "ESJD_SNAPER")) {
                   ## ("LQ_ESSR" centres its squared statistic exactly as CHESSR does, so it keeps the same centre)
                   if (!isTRUE(manual_tau) && ii < n_adapt &&
                       burnin_algorithm %in% c("ChEES", "CHESSR", "CHESSR_log", "SNAPER", "CHESSR_time", "SNAPER_time", "ESJD", "ESJD_CHESSR",
                                               "ESJD_SNAPER", "LQ_ESSR")) {
                     if (use_resident_burnin) {
                         ##
                         ## ---- the full update below (every row, as in its last branch), in the worker on the resident proposals of the
                         ##      previous iteration: the nuisance rows of the centre stay there, its main rows come back into
                         ##      snaper_m_prop_vec_all[index_main], the only rows of it that R reads in this mode:
                         use_weighted_proposal_mean <-  (ii > 1) && isTRUE(tau_weight_by_p_jump) && (length(p_jump_per_chain) == n_chains_burnin)
                         resident_proposal_update <-  resident_api$fn_persistent_burnin_update_snaper_m_prop_resident( worker_ptr,
                                                                                                                      snaper_m_prop_vec_main = snaper_m_prop_vec_all[index_main],
                                                                                                                      theta_vec_current_mean_main = theta_vec_current_main,
                                                                                                                      ii_R = ii,
                                                                                                                      use_weighted_proposal_mean_R = use_weighted_proposal_mean,
                                                                                                                      acceptance_probabilities = as.numeric(p_jump_per_chain),
                                                                                                                      divergences = as.numeric(div_main))
                         if (isTRUE(resident_proposal_update$weighted_mean_status == 2)) {
                             ## R's %*% would not use the BLAS for these proposals (two adjacent entries overflow): the original R code on the
                             ## resident values, and the result goes back to the worker:
                             snaper_m_prop_vec_all_R <-  c(resident_api$fn_persistent_burnin_get_resident_statistic(worker_ptr, "snaper_m_prop_vec_us"),
                                                          snaper_m_prop_vec_all[index_main])
                             proposal_weighted_mean <- fn_weighted_proposal_mean(
                                 proposals = rbind(resident_api$fn_persistent_burnin_get_state(worker_ptr, "theta_us_prop_burnin_tau_adapt_all_chains_input_from_R"),
                                                   theta_main_prop_burnin_tau_adapt_all_chains_input_from_R),
                                 acceptance_probabilities = p_jump_per_chain,
                                 divergences = div_main,
                                 previous_mean = snaper_m_prop_vec_all_R)
                             snaper_m_prop_vec_all_R <- fn_update_snaper_mean(
                                 snaper_mean = snaper_m_prop_vec_all_R,
                                 theta_mean = proposal_weighted_mean,
                                 ii = ii)
                             resident_api$fn_persistent_burnin_set_resident_statistic(worker_ptr, "snaper_m_prop_vec_us", snaper_m_prop_vec_all_R[index_nuisance])
                             snaper_m_prop_vec_all[index_main] <-  snaper_m_prop_vec_all_R[index_main]
                         } else {
                             snaper_m_prop_vec_all[index_main] <-  resident_proposal_update$snaper_m_prop_vec_main
                         }
                     } else {
                     ##
                     ## ---- tau_adaptation_block = "main": from iteration 2 on only the MAIN rows of the centre are updated, the
                     ##      only rows the trajectory criterion reads (each row is its own weighted mean and moving average, so
                     ##      these rows are unchanged). The nuisance proposals still decide which chains are valid: when every
                     ##      proposal entry is finite and below proposal_entry_limit in absolute value, no chain is dropped for a
                     ##      nuisance entry and the weighted mean takes R's BLAS matrix-vector path with or without the nuisance
                     ##      rows; otherwise the stacked (nuisance, main) proposals are used, exactly as below.
                     ##      Iteration 1, "joint", models without a nuisance block and any BLAS other than R's reference BLAS keep the
                     ##      full update below. This relies on fn_weighted_proposal_mean() dropping a chain for any non-finite entry.
                     ##
                     if (proposal_centre_main_rows_only && ii > 1) {
                       if (isTRUE(tau_weight_by_p_jump) && length(p_jump_per_chain) == n_chains_burnin) {
                           proposal_entries_bounded <- isTRUE(min(theta_us_prop_burnin_tau_adapt_all_chains_input_from_R)   > -proposal_entry_limit) &&
                                                       isTRUE(max(theta_us_prop_burnin_tau_adapt_all_chains_input_from_R)   <  proposal_entry_limit) &&
                                                       isTRUE(min(theta_main_prop_burnin_tau_adapt_all_chains_input_from_R) > -proposal_entry_limit) &&
                                                       isTRUE(max(theta_main_prop_burnin_tau_adapt_all_chains_input_from_R) <  proposal_entry_limit)
                           if (proposal_entries_bounded) {
                               proposal_weighted_mean_main <- fn_weighted_proposal_mean(
                                   proposals = theta_main_prop_burnin_tau_adapt_all_chains_input_from_R,
                                   acceptance_probabilities = p_jump_per_chain,
                                   divergences = div_main,
                                   previous_mean = snaper_m_prop_vec_all[index_main])
                           } else {
                               proposal_weighted_mean_main <- fn_weighted_proposal_mean(
                                   proposals = rbind(theta_us_prop_burnin_tau_adapt_all_chains_input_from_R,
                                                     theta_main_prop_burnin_tau_adapt_all_chains_input_from_R),
                                   acceptance_probabilities = p_jump_per_chain,
                                   divergences = div_main,
                                   previous_mean = snaper_m_prop_vec_all)[index_main]
                           }
                       } else {
                           proposal_weighted_mean_main <- theta_vec_current_main
                       }
                       snaper_m_prop_vec_all[index_main] <- fn_update_snaper_mean(
                           snaper_mean = snaper_m_prop_vec_all[index_main],
                           theta_mean = proposal_weighted_mean_main,
                           ii = ii)
                     } else {
                       if (ii > 1 && isTRUE(tau_weight_by_p_jump) && length(p_jump_per_chain) == n_chains_burnin) {
                           proposal_weighted_mean <- fn_weighted_proposal_mean(
                               proposals = if (sample_nuisance) {
                                   rbind(theta_us_prop_burnin_tau_adapt_all_chains_input_from_R,
                                         theta_main_prop_burnin_tau_adapt_all_chains_input_from_R)
                               } else {
                                   theta_main_prop_burnin_tau_adapt_all_chains_input_from_R
                               },
                               acceptance_probabilities = p_jump_per_chain,
                               divergences = div_main,
                               previous_mean = snaper_m_prop_vec_all)
                       } else {
                           proposal_weighted_mean <- theta_vec_current_mean
                       }
                       snaper_m_prop_vec_all <- fn_update_snaper_mean(
                           snaper_mean = snaper_m_prop_vec_all,
                           theta_mean = proposal_weighted_mean,
                           ii = ii)
                     }
                     }  ## end of: if (use_resident_burnin)
                   }
                   ##
                   # try({
                   #    for (kk in 1:n_chains_burnin) {
                   #      n_draws_kk   <- (ii - 1) * n_chains_burnin + kk                         # Welford count = draws seen so far
                   #      theta_kk     <- c(theta_us_vectors_all_chains_input_from_R[, kk], theta_main_vectors_all_chains_input_from_R[, kk])
                   #      delta_old_kk <- theta_kk[index_main] - snaper_m_vec_all[index_main]
                   #      outs <- BayesMVP:::fn_update_snaper_m_and_s(snaper_m = snaper_m_vec_all,
                   #                                                      snaper_s_empirical = snaper_s_vec_all_empirical,
                   #                                                      theta_vec_mean = theta_kk,
                   #                                                      ii = n_draws_kk)
                   #      snaper_m_vec_all           <- outs[, 1]
                   #      snaper_s_vec_all_empirical <- outs[, 2]
                   #      delta_new_kk <- theta_kk[index_main] - snaper_m_vec_all[index_main]
                   #      if ((metric_type_main == "Empirical") && (metric_shape_main == "dense") && (ii >= 20)) {
                   #        empicical_cov_main <- update_cov_Welford(delta_old = delta_old_kk, delta_new = delta_new_kk,
                   #                                                 ii = n_draws_kk, cov_mat = empicical_cov_main)
                   #      }
                   #    }
                   #    ##
                   #    # if ((metric_type_main == "Empirical") && (metric_shape_main == "dense")) {
                   #    #   if (ii < 20) empicical_cov_main <- diag(snaper_s_vec_all_empirical[index_main])
                   #    #   else         empicical_cov_main <- BayesMVP:::Rcpp_shrink_matrix(shrinkage_factor = shrinkage_factor, mat = empicical_cov_main)
                   #    # }
                   #    EHMC_burnin_as_Rcpp_List$snaper_m_vec_main           <- snaper_m_vec_all[index_main]
                   #    EHMC_burnin_as_Rcpp_List$snaper_s_vec_main_empirical <- snaper_s_vec_all_empirical[index_main]
                   #    EHMC_burnin_as_Rcpp_List$snaper_m_vec_us             <- snaper_m_vec_all[index_nuisance]
                   #    EHMC_burnin_as_Rcpp_List$snaper_s_vec_us_empirical   <- snaper_s_vec_all_empirical[index_nuisance]
                   # })
                   ##
                   # Before computing sqrt_M_all_vec
                   ## (min(x, Inf, na.rm = TRUE) <= 0 is the same test as any(x <= 0, na.rm = TRUE), without a nuisance-length
                   ##  logical vector; the Inf keeps an empty or all-NA x at FALSE without a warning)
                   if (min(EHMC_Metric_as_Rcpp_List$M_us_vec, Inf, na.rm = TRUE) <= 0) {
                     EHMC_Metric_as_Rcpp_List$M_us_vec[EHMC_Metric_as_Rcpp_List$M_us_vec <= 0] <- 1.0
                   }
                   if (any(EHMC_Metric_as_Rcpp_List$M_main_vec <= 0, na.rm = TRUE)) {
                     EHMC_Metric_as_Rcpp_List$M_main_vec[EHMC_Metric_as_Rcpp_List$M_main_vec <= 0] <- 1.0
                   }
                   ##
                   ## (sqrt_M_all_vec was computed here every iteration but never used - removed; the guards above stay)
                   
           }
           ##
           {
                  # # EHMC_Metric_as_Rcpp_List$theta_hat_us_vec <- matrix(EHMC_burnin_as_Rcpp_List$snaper_m_vec_us) ## bookmark
                  # EHMC_Metric_as_Rcpp_List$theta_hat_us_vec <- matrix(rep(0.0, n_nuisance)) # -------------------
                  
                  ## theta_hat_us = centre of the exact Gaussian rotation for the nuisance (see theta_hat_us_rule).
                  ## Sampling uses whatever centre burn-in ends with, FROZEN - so eps must finish adapting against a
                  ## frozen centre too, or the burn-in acceptance is inflated and eps comes out too big.
                  if (use_resident_burnin) {
                    ## the same rule, applied to every chain's centre in the worker (R's copy is refreshed where R reads it, and at the end):
                    if (identical(theta_hat_us_rule, "zero") || (ii <= clip_iter)) {
                      resident_api$fn_persistent_burnin_set_nuisance_centre_resident(worker_ptr, "zero")
                    } else if (identical(theta_hat_us_rule, "running_mean") || (ii < theta_hat_us_freeze_iter)) {
                      resident_api$fn_persistent_burnin_set_nuisance_centre_resident(worker_ptr, "running_mean")
                    } else if (ii == theta_hat_us_freeze_iter) {
                      resident_api$fn_persistent_burnin_set_nuisance_centre_resident(worker_ptr, "running_mean")
                      cat("nuisance centre (theta_hat_us) FROZEN from iteration", ii, "(theta_hat_us_rule = running_mean_frozen)\n")
                    } ## else: running_mean_frozen past the freeze iteration - the centre stays exactly as it is
                  } else {
                  if (identical(theta_hat_us_rule, "zero") || (ii <= clip_iter)) {
                    EHMC_Metric_as_Rcpp_List$theta_hat_us_vec <- theta_hat_us_vec_zero   ## = matrix(rep(0.0, n_nuisance)), built once before the loop
                  } else if (identical(theta_hat_us_rule, "running_mean") || (ii < theta_hat_us_freeze_iter)) {
                    EHMC_Metric_as_Rcpp_List$theta_hat_us_vec <- matrix(EHMC_burnin_as_Rcpp_List$snaper_m_vec_us)
                  } else if (ii == theta_hat_us_freeze_iter) {
                    EHMC_Metric_as_Rcpp_List$theta_hat_us_vec <- matrix(EHMC_burnin_as_Rcpp_List$snaper_m_vec_us)
                    cat("nuisance centre (theta_hat_us) FROZEN from iteration", ii, "(theta_hat_us_rule = running_mean_frozen)\n")
                  } ## else: running_mean_frozen past the freeze iteration - keep the centre exactly as it is
                  }  ## end of: if (use_resident_burnin)
           }
           ##
           # if ( (metric_type_main == "Empirical") && (metric_shape_main == "dense") ) {
           #   
           #        if (ii < 20) {
           #              empicical_cov_main <- diag(EHMC_burnin_as_Rcpp_List$snaper_s_vec_main_empirical)
           #        } else {
           #              ## update covariance using Welford's algorithm
           #              try({  
           #                              cov_outs <- update_cov_Welford(  
           #                                                                delta_old = delta_old_main,
           #                                                                delta_new = delta_new_main,
           #                                                                ii = ii,
           #                                                                cov_mat = empicical_cov_main)
           #                              ## Get cov mat:
           #                              empicical_cov_main <- cov_outs # for unbiased estimate
           #                              empicical_cov_main <- BayesMVP:::Rcpp_shrink_matrix( shrinkage_factor = shrinkage_factor, 
           #                                                                                      mat = empicical_cov_main)
           #              })
           #        }
           # }
           if (metric_estimator %in% c("chain_mean", "chain_mean_scaled")) {
              if ((metric_type_main == "Empirical") && (metric_shape_main == "dense")) {
                  if (ii < 20) {
                    ## nrow = length(...) is needed for models with ONE main parameter: diag(x) of a length-1
                    ## vector builds a floor(x) x floor(x) identity (0x0 for a variance below 1), so update_cov_Welford then
                    ## failed at every iteration ("non-conformable arrays", swallowed by try()) and the dense metric was never adapted.
                    empicical_cov_main <- diag(c(EHMC_burnin_as_Rcpp_List$snaper_s_vec_main_empirical),
                                               nrow = length(EHMC_burnin_as_Rcpp_List$snaper_s_vec_main_empirical))
                  } else {
                    try({
                      empicical_cov_main <- update_cov_Welford(delta_old = delta_old_main, delta_new = delta_new_main,
                                                               ii = ii, cov_mat = empicical_cov_main)
                      empicical_cov_main <- Rcpp_shrink_matrix(shrinkage_factor = shrinkage_factor, mat = empicical_cov_main)
                    })
                  }
              }
          }
          ##
          { ## //// Update metric(s):
                    
                    # ii_min <- round(n_burnin/10) ## (- 1 + round(n_burnin/5))
                    ##
                    if (n_burnin > 500) {
                       # ii_min <- 50
                       ii_min <- round(n_burnin/10)
                       # ii_max <- 750
                       ii_max <- n_adapt
                    } else { 
                       ii_min <- round(n_burnin/10)
                       ii_max <- n_adapt
                    }
            
                    #### ii_max <- (n_burnin - round(n_burnin/20))
                    num_diff_e <- 0.0001
            
                   if (burnin_schedule == "automatic" && ii == metric_adaptation_end_iter && !metric_ready) {
                       warning("Final metric update skipped: the pooled estimate is not ready. Existing metrics remain fixed; consider longer burn-in.")
                   }
                   ## if  ((ii < n_adapt) && (ii > ii_min) && (ii %% interval_width_main == 0) && (ii < ii_max))  {
                   ## Request the final update even when the interval width does not divide the cutoff.
                   if ((ii <= metric_adaptation_end_iter) && (ii > ii_min) &&
                       ((ii %% interval_width_main == 0) || (burnin_schedule == "automatic" && ii == metric_adaptation_end_iter)) &&
                       (ii < ii_max) && metric_ready)  {
                                   
                         if (metric_type_main == "unit") { 
                           
                                                   EHMC_Metric_as_Rcpp_List$M_inv_main_vec <- c(rep(1, n_params_main))
                                                   EHMC_Metric_as_Rcpp_List$M_inv_dense_main <-  diag(c(EHMC_Metric_as_Rcpp_List$M_inv_main_vec))
                                                   EHMC_Metric_as_Rcpp_List$M_inv_dense_main_chol <-  EHMC_Metric_as_Rcpp_List$M_inv_dense_main
                                                   EHMC_Metric_as_Rcpp_List$M_dense_sqrt <-           EHMC_Metric_as_Rcpp_List$M_inv_dense_main
                           
                         } else { 
                           
                                 if (metric_type_main == "Hessian") {
                                   
                                     # if (ii < win_first_start) {
                                       
                                             if (use_resident_burnin) {
                                                 ## update_M_Hessian_main() reads the nuisance centre snaper_m_vec_us, which is kept in the worker:
                                                 EHMC_burnin_as_Rcpp_List$snaper_m_vec_us <-  resident_api$fn_persistent_burnin_get_resident_statistic(worker_ptr, "snaper_m_vec_us")
                                             }
                                             outs <- update_M_Hessian_main(
                                                     metric_shape_main = metric_shape_main,
                                                     ##
                                                     EHMC_Metric_as_Rcpp_List = EHMC_Metric_as_Rcpp_List,
                                                     EHMC_burnin_as_Rcpp_List = EHMC_burnin_as_Rcpp_List,
                                                     ##
                                                     ratio_M_main = ratio_M_main,
                                                     interval_width_main = interval_width_main,
                                                     ##
                                                     force_autodiff_for_metric = force_autodiff_for_metric,
                                                     force_PartialLog_for_metric = force_PartialLog_for_metric,
                                                     force_multi_attempts_for_metric = force_multi_attempts_for_metric,
                                                     ##
                                                     shrinkage_factor = shrinkage_factor,
                                                     num_diff_e = num_diff_e,
                                                     ##
                                                     Model_type = Model_type,
                                                     y = y,
                                                     ##
                                                     Model_args_as_Rcpp_List = Model_args_as_Rcpp_List,
                                                     ii = ii,
                                                     n_adapt = n_adapt,
                                                     main_vec_for_Hessian = main_vec_for_Hessian)
                                             
                                             EHMC_Metric_as_Rcpp_List <- outs$EHMC_Metric_as_Rcpp_List
                                             EHMC_burnin_as_Rcpp_List <- outs$EHMC_burnin_as_Rcpp_List
                                             
                                     # }
                                   
                                 # } else if (metric_type_main == "Empirical") {
                                 #   
                                 #           outs <-  update_M_Empirical_main(
                                 #                    debug = debug,
                                 #                    ##
                                 #                    metric_shape_main = metric_shape_main,
                                 #                    ##
                                 #                    EHMC_Metric_as_Rcpp_List = EHMC_Metric_as_Rcpp_List,
                                 #                    EHMC_burnin_as_Rcpp_List = EHMC_burnin_as_Rcpp_List,
                                 #                    ##
                                 #                    # empicical_cov_main = empicical_cov_main,
                                 #                    empicical_cov_main = cov_crosschain_ema, 
                                 #                    ##
                                 #                    ratio_M_main = ratio_M_main,
                                 #                    ##
                                 #                    ii = ii,
                                 #                    n_adapt = n_adapt,
                                 #                    ##
                                 #                    M_decay_type = M_decay_type,
                                 #                    # M_decay_type = "inverse",
                                 #                    # M_decay_type = "linear",
                                 #                    # M_decay_type = "exponential",
                                 #                    # M_decay_type = "cosine",
                                 #                    ##
                                 #                    M_decay_power = M_decay_power,
                                 #                    M_decay_scale = M_decay_scale)
                                 #           
                                 #           EHMC_Metric_as_Rcpp_List <- outs$EHMC_Metric_as_Rcpp_List
                                 #           EHMC_burnin_as_Rcpp_List <- outs$EHMC_burnin_as_Rcpp_List
                                 #           
                                 #           # str(EHMC_Metric_as_Rcpp_List)
                                 #           
                                 # }
                           
                           
                         # } else if (metric_type_main == "Empirical") {
                         #       
                         #       if (metric_shape_main == "dense" && is.null(cov_crosschain_ema)) {
                         #         message("ii = ", ii, ": cross-chain cov not ready yet - skipping metric update this interval")
                         #       } else {
                         #         outs <- update_M_Empirical_main(  debug = debug,
                         #                                           metric_shape_main = metric_shape_main,
                         #                                           EHMC_Metric_as_Rcpp_List = EHMC_Metric_as_Rcpp_List,
                         #                                           EHMC_burnin_as_Rcpp_List = EHMC_burnin_as_Rcpp_List,
                         #                                           empicical_cov_main = cov_crosschain_ema,     ## the new estimator
                         #                                           ratio_M_main = ratio_M_main,
                         #                                           ii = ii, n_adapt = n_adapt,
                         #                                           M_decay_type = M_decay_type,
                         #                                           M_decay_power = M_decay_power,
                         #                                           M_decay_scale = M_decay_scale)
                         #         
                         #         EHMC_Metric_as_Rcpp_List <- outs$EHMC_Metric_as_Rcpp_List
                         #         EHMC_burnin_as_Rcpp_List <- outs$EHMC_burnin_as_Rcpp_List
                         #       }
                         #   
                         # }
                                           
                           } else if (metric_type_main == "Empirical") {
                             
                             if (metric_shape_main == "dense") {
                               ##### dense metric handled by the windowed scheme (EDIT B) - do nothing here
                               outs <- update_M_Empirical_main(  debug = debug,
                                                                 metric_shape_main = metric_shape_main,
                                                                 EHMC_Metric_as_Rcpp_List = EHMC_Metric_as_Rcpp_List,
                                                                 EHMC_burnin_as_Rcpp_List = EHMC_burnin_as_Rcpp_List,
                                                                 empicical_cov_main = empicical_cov_main,   ## reverted (diag path ignores it anyway)
                                                                 ratio_M_main = ratio_M_main,
                                                                 ii = ii, n_adapt = n_adapt,
                                                                 M_decay_type = M_decay_type,
                                                                 M_decay_power = M_decay_power,
                                                                 M_decay_scale = M_decay_scale,
                                                                 variance_scale = metric_variance_scale,   ## x n_chains_burnin only for chain_mean_scaled
                                                                 adaptive_shrinkage_state = if (metric_pooled_offdiagonal_shrinkage_adaptive_active &&
                                                                                                 identical(metric_estimator, "per_iteration"))
                                                                                                    adaptive_metric_shrinkage_state else NULL,
                                                                 compute_M_dense_sqrt = FALSE)             ## M_dense_sqrt is only read by SNAPER
                               
                               EHMC_Metric_as_Rcpp_List <- outs$EHMC_Metric_as_Rcpp_List
                               EHMC_burnin_as_Rcpp_List <- outs$EHMC_burnin_as_Rcpp_List
                               if (metric_pooled_offdiagonal_shrinkage_adaptive_active && identical(metric_estimator, "per_iteration")) {
                                     adaptive_metric_shrinkage_state <- outs$adaptive_shrinkage_state
                                     adaptive_metric_shrinkage_last_diagnostics <- outs$adaptive_shrinkage_diagnostics
                                     if (!is.null(adaptive_metric_shrinkage_last_diagnostics) &&
                                         isTRUE(adaptive_metric_shrinkage_last_diagnostics$applied) &&
                                         is.finite(adaptive_metric_shrinkage_last_diagnostics$shrinkage)) {
                                           adaptive_metric_shrinkage_lambda_history[ii] <- adaptive_metric_shrinkage_last_diagnostics$shrinkage
                                     }
                               }
                             } else {
                               outs <- update_M_Empirical_main(  debug = debug,
                                                                 metric_shape_main = metric_shape_main,
                                                                 EHMC_Metric_as_Rcpp_List = EHMC_Metric_as_Rcpp_List,
                                                                 EHMC_burnin_as_Rcpp_List = EHMC_burnin_as_Rcpp_List,
                                                                 empicical_cov_main = empicical_cov_main,   ## reverted (diag path ignores it anyway)
                                                                 ratio_M_main = ratio_M_main,
                                                                 ii = ii, n_adapt = n_adapt,
                                                                 M_decay_type = M_decay_type,
                                                                 M_decay_power = M_decay_power,
                                                                 M_decay_scale = M_decay_scale,
                                                                 variance_scale = metric_variance_scale,   ## x n_chains_burnin only for chain_mean_scaled
                                                                 ## per-iteration estimator: this iteration's cross-chain variances of the main parameters (NULL keeps
                                                                 ## variance_scale * snaper_s_vec_main_empirical, for every other estimator):
                                                                 ## per_iteration_variance_vec_main = if (metric_estimator == "per_iteration") per_iteration_variance_vec_main else NULL,
                                                                 ## pooled estimator, diagonal shape: its own regularised pooled variances (the diagonal of empicical_cov_main, set by the
                                                                 ## pooled blocks above), not the running variance of the cross-chain mean (snaper_s_vec_main_empirical, ~ Sigma / n_chains):
                                                                 per_iteration_variance_vec_main = if (metric_estimator == "per_iteration") per_iteration_variance_vec_main else
                                                                                                   if (metric_estimator == "pooled") diag(empicical_cov_main) else NULL,
                                                                 adaptive_shrinkage_state = NULL,
                                                                 compute_M_dense_sqrt = FALSE)             ## M_dense_sqrt is only read by SNAPER
                               
                               EHMC_Metric_as_Rcpp_List <- outs$EHMC_Metric_as_Rcpp_List
                               EHMC_burnin_as_Rcpp_List <- outs$EHMC_burnin_as_Rcpp_List
                             }
                             
                           }
                      
                           
                         }
                           
                           if (sample_nuisance == TRUE)  {
                             
                             # if  ((ii < n_adapt) && (ii > ii_min) && (ii %% interval_width_nuisance == 0) && (ii < ii_max))  {
                             # if ((ii < n_adapt) && (ii > ii_min) && (ii %% interval_width_main == 0) && (ii < ii_max) && metric_ready)  {
                             if ((ii <= metric_adaptation_end_iter) && (ii > ii_min) &&
                                 ((ii %% interval_width_nuisance == 0) || (burnin_schedule == "automatic" && ii == metric_adaptation_end_iter)) &&
                                 (ii < ii_max) && metric_ready)  {
                               
                                     try({
                                       
                                       if (use_resident_burnin && (metric_type_nuisance %in% c("Empirical", "uniform_diag"))) {
                                               ## the nuisance statistic read below comes from the worker (R's copy of it is not updated during a
                                               ## resident burn-in); the nuisance metric computed from it reaches the worker with the next push:
                                               ## if (metric_estimator == "pooled") {
                                               if (metric_estimator %in% c("pooled", "per_iteration")) {
                                                    var_draws_all[index_nuisance] <-  resident_api$fn_persistent_burnin_get_resident_statistic(worker_ptr, "var_draws_us")
                                               } else {
                                                    EHMC_burnin_as_Rcpp_List$snaper_s_vec_us_empirical <-  resident_api$fn_persistent_burnin_get_resident_statistic(worker_ptr, "snaper_s_vec_us_empirical")
                                               }
                                       }
                                       if (metric_type_nuisance == "unit") { 
                                         
                                               #### ---- for unit metric --------------------------------------
                                               ## (the unit vectors are built once, before the loop)
                                               EHMC_Metric_as_Rcpp_List$M_inv_us_vec  <- unit_M_inv_us_vec
                                               EHMC_Metric_as_Rcpp_List$M_us_vec      <- unit_M_us_vec
                                               EHMC_burnin_as_Rcpp_List$sqrt_M_us_vec <- unit_sqrt_M_us_vec
                                         
                                       } else if (metric_type_nuisance == "Empirical") {
                                         
                                               ## if (metric_estimator == "pooled") {
                                               if (metric_estimator %in% c("pooled", "per_iteration")) {
                                                    proposed_variance_vec_us <- var_draws_all[index_nuisance]
                                               } else {
                                                    proposed_variance_vec_us <- metric_variance_scale * EHMC_burnin_as_Rcpp_List$snaper_s_vec_us_empirical
                                               }
                                               if (length(proposed_variance_vec_us) != n_nuisance) stop("BUG: proposed_variance_vec_us wrong length!")
                                               ##
                                               outs <- update_M_diag_Empirical_nuisance(  EHMC_Metric_as_Rcpp_List = EHMC_Metric_as_Rcpp_List,
                                                                                          EHMC_burnin_as_Rcpp_List = EHMC_burnin_as_Rcpp_List,
                                                                                          ##
                                                                                          proposed_variance_vec = proposed_variance_vec_us,
                                                                                          ##
                                                                                          ratio = ratio_M_us,
                                                                                          ##
                                                                                          ii = ii,
                                                                                          n_adapt = n_adapt,
                                                                                          ##
                                                                                          M_decay_type = M_decay_type,
                                                                                          # M_decay_type = "inverse",
                                                                                          # M_decay_type = "linear",
                                                                                          # M_decay_type = "exponential",
                                                                                          # M_decay_type = "cosine",
                                                                                          ##
                                                                                          M_decay_power = M_decay_power,
                                                                                          M_decay_scale = M_decay_scale)
                                               ##
                                               EHMC_Metric_as_Rcpp_List <- outs$EHMC_Metric_as_Rcpp_List
                                               EHMC_burnin_as_Rcpp_List <- outs$EHMC_burnin_as_Rcpp_List
                                               # M_inv_us_vec <- EHMC_Metric_as_Rcpp_List$M_inv_us_vec
                                         
                                       } else if (metric_type_nuisance == "uniform_diag") {
                                         
                                               ## if (metric_estimator == "pooled") {
                                               if (metric_estimator %in% c("pooled", "per_iteration")) {
                                                    proposed_variance_vec_us <- var_draws_all[index_nuisance]
                                               } else {
                                                    proposed_variance_vec_us <- metric_variance_scale * EHMC_burnin_as_Rcpp_List$snaper_s_vec_us_empirical
                                               }
                                               if (length(proposed_variance_vec_us) != n_nuisance) stop("BUG: proposed_variance_vec_us wrong length!")
                                               ##
                                               outs <- update_M_diag_Empirical_nuisance(   EHMC_Metric_as_Rcpp_List = EHMC_Metric_as_Rcpp_List,
                                                                                           EHMC_burnin_as_Rcpp_List = EHMC_burnin_as_Rcpp_List,
                                                                                           ##
                                                                                           proposed_variance_vec = proposed_variance_vec_us,
                                                                                           ##
                                                                                           ratio = ratio_M_us,
                                                                                           ##
                                                                                           ii = ii,
                                                                                           n_adapt = n_adapt,
                                                                                           ##
                                                                                           M_decay_type = M_decay_type,
                                                                                           # M_decay_type = "inverse",
                                                                                           # M_decay_type = "linear",
                                                                                           # M_decay_type = "exponential",
                                                                                           # M_decay_type = "cosine",
                                                                                           ##
                                                                                           M_decay_power = M_decay_power,
                                                                                           M_decay_scale = M_decay_scale)
                                               ##
                                               EHMC_Metric_as_Rcpp_List <- outs$EHMC_Metric_as_Rcpp_List
                                               EHMC_burnin_as_Rcpp_List <- outs$EHMC_burnin_as_Rcpp_List
                                               ##
                                               EHMC_Metric_as_Rcpp_List$M_inv_us_vec <- rep(stats::median(EHMC_Metric_as_Rcpp_List$M_inv_us_vec), n_nuisance)
                                               EHMC_Metric_as_Rcpp_List$M_us_vec <- 1 / EHMC_Metric_as_Rcpp_List$M_inv_us_vec
                                               EHMC_burnin_as_Rcpp_List$sqrt_M_us_vec <- sqrt(EHMC_Metric_as_Rcpp_List$M_us_vec)
                                         
                                       }
                                       
                                     })
                               
                             }
                             
                           }
                      
                    } ## // end of "if  ( (ii >  (- 1 + round(n_burnin/5))) && (ii %% interval_width_main == 0) && (ii < (n_burnin - round(n_burnin/10))) )  {" loop
            
          }  ## // end of metric block
   
          ## ---- re-initialise eps against the settled metric at ChEES handover (eps_reinit_at_ChEES_handover = TRUE) ----
          if ((ii == gap) && !isTRUE(eps_reinit_at_ChEES_handover)) {
            cat("eps KEPT at ChEES handover (ii =", ii, "):", signif(EHMC_args_as_Rcpp_List$eps_main, 4), "(eps_reinit_at_ChEES_handover = FALSE)\n")
          }
          if ((ii == gap) && isTRUE(eps_reinit_at_ChEES_handover)) {
            try({
              if (use_resident_burnin) {
                  ## the search below reads the nuisance centre from the metric list; the centre is kept in the worker:
                  EHMC_Metric_as_Rcpp_List$theta_hat_us_vec <-  matrix(resident_api$fn_persistent_burnin_get_resident_statistic(worker_ptr, "theta_hat_us_vec"))
              }
              if (isTRUE(eps_search_has_n_threads)) {   ## threads for the search: see the initial step-size block above
                par_res <- fn_find_initial_eps_main_and_us(
                    theta_main_vec_initial_ref = matrix(rowMeans(theta_main_vectors_all_chains_input_from_R), ncol = 1),
                    theta_us_vec_initial_ref   = matrix(if (use_resident_burnin) resident_api$fn_persistent_burnin_state_row_means_resident(worker_ptr, "theta_us") else
                                                                             rowMeans(theta_us_vectors_all_chains_input_from_R),   ncol = 1),
                    partitioned_HMC = partitioned_HMC,
                    seed = seed + ii,
                    Model_type = Model_type,
                    force_autodiff = force_autodiff,
                    force_PartialLog = force_PartialLog,
                    multi_attempts = multi_attempts,
                    y_ref = y,
                    Model_args_as_Rcpp_List = Model_args_as_Rcpp_List,
                    EHMC_args_as_Rcpp_List = EHMC_args_as_Rcpp_List,
                    EHMC_Metric_as_Rcpp_List = EHMC_Metric_as_Rcpp_List,
                    n_threads = eps_search_n_threads)
              } else {
              par_res <- fn_find_initial_eps_main_and_us(
                  theta_main_vec_initial_ref = matrix(rowMeans(theta_main_vectors_all_chains_input_from_R), ncol = 1),
                  theta_us_vec_initial_ref   = matrix(if (use_resident_burnin) resident_api$fn_persistent_burnin_state_row_means_resident(worker_ptr, "theta_us") else
                                                                           rowMeans(theta_us_vectors_all_chains_input_from_R),   ncol = 1),
                  partitioned_HMC = partitioned_HMC,
                  seed = seed + ii,
                  Model_type = Model_type,
                  force_autodiff = force_autodiff,
                  force_PartialLog = force_PartialLog,
                  multi_attempts = multi_attempts,
                  y_ref = y,
                  Model_args_as_Rcpp_List = Model_args_as_Rcpp_List,
                  EHMC_args_as_Rcpp_List = EHMC_args_as_Rcpp_List,
                  EHMC_Metric_as_Rcpp_List = EHMC_Metric_as_Rcpp_List)
              }
              eps_new <- min(max_eps_main, par_res[[1]])
              EHMC_args_as_Rcpp_List$eps_main <- eps_new
              EHMC_args_as_Rcpp_List$eps_us   <- eps_new
              ## EHMC_burnin_as_Rcpp_List$eps_m_adam_main <- eps_new ; EHMC_burnin_as_Rcpp_List$eps_v_adam_main <- 0
              ## EHMC_burnin_as_Rcpp_List$eps_m_adam_us   <- eps_new ; EHMC_burnin_as_Rcpp_List$eps_v_adam_us   <- 0
              EHMC_burnin_as_Rcpp_List$eps_m_adam_main <- 0 ; EHMC_burnin_as_Rcpp_List$eps_v_adam_main <- 0   ## first moment of the acceptance error, not of eps
              EHMC_burnin_as_Rcpp_List$eps_m_adam_us   <- 0 ; EHMC_burnin_as_Rcpp_List$eps_v_adam_us   <- 0
              eps_adam_updates_main <- 0
              eps_adam_updates_us <- 0
              cat("eps re-initialised at ChEES handover (ii =", ii, "):", signif(eps_new, 4), "\n")
            })
          }
          ##
          if (isTRUE(getOption("BayesMVP_force_L1", FALSE))) {
              EHMC_args_as_Rcpp_List$tau_main <- EHMC_args_as_Rcpp_List$eps_main
              EHMC_args_as_Rcpp_List$tau_us   <- EHMC_args_as_Rcpp_List$eps_us
          }
          ##
          ## ---- Metric-coordinate principal directions, held fixed during this transition -------------------------------------------------
          if (!isTRUE(manual_tau)) {
              adapted_blocks <- tau_adaptation_block_effective   ## "main" (default) or "joint" = main + nuisance (EXPERIMENTAL)
              for (adapted_block in adapted_blocks) {
                  if (use_resident_joint_block) {
                  ## the joint states, centre and direction stay in the worker (their nuisance rows are read there, below):
                  states <-  NULL
                  centre <-  NULL
                  } else {
                  states <- if (adapted_block == "main") theta_main_vectors_all_chains_input_from_R else
                            if (adapted_block == "joint") rbind(theta_main_vectors_all_chains_input_from_R, theta_us_vectors_all_chains_input_from_R) else
                            theta_us_vectors_all_chains_input_from_R
                  centre <- if (adapted_block == "main") EHMC_burnin_as_Rcpp_List$snaper_m_vec_main else
                            if (adapted_block == "joint") c(EHMC_burnin_as_Rcpp_List$snaper_m_vec_main, EHMC_burnin_as_Rcpp_List$snaper_m_vec_us) else
                            EHMC_burnin_as_Rcpp_List$snaper_m_vec_us
                  }  ## end of: if (use_resident_joint_block)
                  mass_main_block <- if (metric_shape_main == "diag") {
                      1 / c(EHMC_Metric_as_Rcpp_List$M_inv_main_vec)
                  } else EHMC_Metric_as_Rcpp_List$M_dense_main
                  mass <- if (adapted_block == "us") c(EHMC_Metric_as_Rcpp_List$M_us_vec) else
                          if (adapted_block == "joint") list(main = mass_main_block, us = c(EHMC_Metric_as_Rcpp_List$M_us_vec)) else mass_main_block
                  if (!identical(mass, trajectory_mass[[adapted_block]])) {
                      new_factor <- if (adapted_block == "joint") {
                          fn_trajectory_joint_metric_factor(mass_main = mass$main, mass_us_vec = mass$us, n_main = n_params_main, n_us = n_nuisance)
                      } else fn_trajectory_metric_factor(mass, nrow(states))
                      if (use_resident_joint_block) {
                      ## ---- fn_transport_snaper_direction() of the resident joint direction: the main-block part here, with the same R
                      ##      code on the direction's main rows; the nuisance part and the joint normalisation in the worker:
                      previous_joint_factor <-  trajectory_metric[[adapted_block]]
                      if (burnin_algorithm %in% c("SNAPER", "SNAPER_time", "ESJD_SNAPER") &&
                          !(is.null(previous_joint_factor) || identical(previous_joint_factor, new_factor))) {
                          ## Both blocks retain their transported magnitudes until the single joint normalisation.
                          ## The complete direction is a temporary vector; no chain draws or direction history are stored.
                          direction_joint <-  resident_api$fn_persistent_burnin_joint_direction_get(worker_ptr, FALSE)
                          transported_joint <-  fn_transport_snaper_direction(direction_joint, previous_joint_factor, new_factor)
                          resident_api$fn_persistent_burnin_joint_direction_set(worker_ptr, transported_joint)
                      }
                      } else {
                      trajectory_direction[[adapted_block]] <- fn_transport_snaper_direction(
                          trajectory_direction[[adapted_block]], trajectory_metric[[adapted_block]], new_factor)
                      }  ## end of: if (use_resident_joint_block)
                      trajectory_metric[[adapted_block]] <- new_factor
                      trajectory_mass[[adapted_block]] <- mass
                  }
                  ## if (ii < n_adapt && burnin_algorithm == "SNAPER") {
                  if (ii < n_adapt && burnin_algorithm %in% c("SNAPER", "SNAPER_time", "ESJD_SNAPER")) {
                      if (use_resident_joint_block) {
                      ## ---- fn_update_snaper_w_minibatch() of the resident joint direction: the main rows of the states in metric
                      ##      coordinates here (the same R code on the main rows), the nuisance rows and the update in the worker:
                      X_main_metric <-  fn_apply_trajectory_metric(trajectory_metric[[adapted_block]]$main,
                                                                    theta_main_vectors_all_chains_input_from_R - c(EHMC_burnin_as_Rcpp_List$snaper_m_vec_main))
                      snaper_direction_status <-  resident_api$fn_persistent_burnin_joint_direction_update_snaper_resident( worker_ptr,
                                                                                                                          X_main_metric = X_main_metric,
                                                                                                                          factor_us = trajectory_metric[[adapted_block]]$us,
                                                                                                                          eta_w_R = 8 / max(1, ii))
                      if (snaper_direction_status == 2) {
                          ## one of R's two matrix products would not use the BLAS for these numbers: the R code on the joint values
                          ## from the worker, and the result goes back to the worker:
                          snaper_direction_joint <-  fn_update_snaper_w_minibatch(
                              X = rbind(theta_main_vectors_all_chains_input_from_R,
                                        resident_api$fn_persistent_burnin_get_state(worker_ptr, "theta_us_vectors_all_chains_output_to_R")),
                              snaper_m_vec = c(EHMC_burnin_as_Rcpp_List$snaper_m_vec_main,
                                               resident_api$fn_persistent_burnin_get_resident_statistic(worker_ptr, "snaper_m_vec_us")),
                              snaper_w_vec = resident_api$fn_persistent_burnin_joint_direction_get(worker_ptr, FALSE),
                              eta_w = 8 / max(1, ii), metric_factor = trajectory_metric[[adapted_block]])
                          resident_api$fn_persistent_burnin_joint_direction_set(worker_ptr, snaper_direction_joint)
                      }
                      } else {
                      trajectory_direction[[adapted_block]] <- fn_update_snaper_w_minibatch(
                          X = states, snaper_m_vec = c(centre), snaper_w_vec = trajectory_direction[[adapted_block]],
                          eta_w = 8 / max(1, ii), metric_factor = trajectory_metric[[adapted_block]])
                      }  ## end of: if (use_resident_joint_block)
                  }
              }
          }
          ##
          ## //////////////////   --------------------------------  Perform iteration  ------------------------------------------------------------------------------
          if (!is.null(bulk_local_tuner_state) && ii >= bulk_local_tuner_collect_start) {
              if (!bulk_local_tuner_freeze_active) {
                  bulk_local_tuner_frozen_kernel <-  list(eps_main = EHMC_args_as_Rcpp_List$eps_main,
                        eps_us = EHMC_args_as_Rcpp_List$eps_us,
                        tau_main = EHMC_args_as_Rcpp_List$tau_main,
                        tau_us = EHMC_args_as_Rcpp_List$tau_us)
                  bulk_local_tuner_metadata$actual_eps_adaptation_end_iter <-
                        min(n_adapt - 1, ii - 1)
                  bulk_local_tuner_metadata$actual_tau_adaptation_end_iter <-
                        min(n_adapt - 1, ii - 1)
                  bulk_local_tuner_metadata$final_sampling_kernel_start_iter <-  ii
                  bulk_local_tuner_freeze_active <-  TRUE
              }
              EHMC_args_as_Rcpp_List$eps_main <-  bulk_local_tuner_frozen_kernel$eps_main
              EHMC_args_as_Rcpp_List$eps_us <-  bulk_local_tuner_frozen_kernel$eps_us
              EHMC_args_as_Rcpp_List$tau_main <-  bulk_local_tuner_frozen_kernel$tau_main
              EHMC_args_as_Rcpp_List$tau_us <-  bulk_local_tuner_frozen_kernel$tau_us
          }
          try({
            
                                        # if (parallel_method == "RcppParallel") {
                                        #   fn <- BayesMVP:::fn_R_RcppParallel_EHMC_single_iter_burnin
                                        # } else {  ### OpenMP
                                        #   fn <- BayesMVP:::fn_R_OpenMP_EHMC_single_iter_burnin
                                        # }
                                        # # 
                                        # # # t0 <- proc.time()[3]
                                        # # ##
                                        # result <-   fn(      n_threads_R = n_chains_burnin,
                                        #                      seed_R = seed + ii, ## seed + ii
                                        #                      sample_nuisance_R = sample_nuisance,
                                        #                      n_nuisance_to_track = 1, ## n_nuisance_to_track,
                                        #                      n_iter_R = 1,
                                        #                      current_iter_R = ii,
                                        #                      n_adapt  = 0,
                                        #                      partitioned_HMC_R = partitioned_HMC,
                                        #                      diffusion_HMC_R = diffusion_HMC,
                                        #                      clip_iter = clip_iter,
                                        #                      gap =  gap,
                                        #                      burnin_indicator = FALSE,
                                        #                      metric_type_nuisance = metric_type_nuisance,
                                        #                      metric_type_main = metric_type_main,
                                        #                      shrinkage_factor = shrinkage_factor,
                                        #                      max_eps_main = max_eps_main,
                                        #                      max_eps_us = max_eps_us,
                                        #                      tau_main_target = 0, # dummy arg
                                        #                      tau_us_target = 0,   # dummy arg
                                        #                      main_L_manual = FALSE,  # dummy arg
                                        #                      L_main_if_manual = 0,   # dummy arg
                                        #                      us_L_manual = FALSE,    # dummy arg
                                        #                      L_us_if_manual = 0,     # dummy arg
                                        #                      max_L = max_L,
                                        #                      tau_mult = tau_mult,
                                        #                      ratio_M_us = ratio_M_us,
                                        #                      ratio_Hess_main = ratio_M_main,
                                        #                      M_interval_width = interval_width_main,
                                        #                      Model_type_R =  Model_type,
                                        #                      force_autodiff_R = force_autodiff,
                                        #                      force_PartialLog_R = force_PartialLog ,
                                        #                      multi_attempts_R = multi_attempts,
                                        #                      ##
                                        #                      theta_main_vectors_all_chains_input_from_R = theta_main_vectors_all_chains_input_from_R,
                                        #                      velocity_main_vectors_all_chains_input_from_R = velocity_main_vectors_all_chains_input_from_R,
                                        #                      ##
                                        #                      theta_us_vectors_all_chains_input_from_R = theta_us_vectors_all_chains_input_from_R,
                                        #                      velocity_us_vectors_all_chains_input_from_R = velocity_us_vectors_all_chains_input_from_R,
                                        #                      ##
                                        #                      y_Eigen_R =  y,
                                        #                      Model_args_as_Rcpp_List =  Model_args_as_Rcpp_List,
                                        #                      EHMC_args_as_Rcpp_List =   EHMC_args_as_Rcpp_List,
                                        #                      EHMC_Metric_as_Rcpp_List = EHMC_Metric_as_Rcpp_List,
                                        #                      EHMC_burnin_as_Rcpp_List = EHMC_burnin_as_Rcpp_List,
                                        #                      ##
                                        #                      n_threads_WCP = n_threads_WCP)
                                        #
                                        # ## Push this iteration's eps / tau / metric (same wholesale overwrite the old
                                        # ## wrapper did internally every call -- adaptation semantics identical):
                                        # BayesMVP:::fn_persistent_burnin_update_adaptation( worker_ptr,
                                        #                                                       EHMC_args_as_Rcpp_List,
                                        #                                                       EHMC_Metric_as_Rcpp_List)
                                        # 
                                        # t0 <- proc.time()[3]
                                        ##
                                        t0 <- proc.time()[3]
                                        if (use_resident_burnin) {
                                            ## eps / tau and the main metric every iteration; the nuisance metric is copied whenever it differs from the
                                            ## worker's copy (so R need not flag its changes); the nuisance centre is set by the centre rule above:
                                            resident_api$fn_persistent_burnin_update_adaptation_main_only( worker_ptr,
                                                                                                           EHMC_args_as_Rcpp_List,
                                                                                                           EHMC_Metric_as_Rcpp_List,
                                                                                                           push_nuisance_metric_R = FALSE,
                                                                                                           push_nuisance_centre_R = FALSE)
                                        } else {
                                        fn_persistent_burnin_update_adaptation( worker_ptr,
                                                                                              EHMC_args_as_Rcpp_List,
                                                                                              EHMC_Metric_as_Rcpp_List)
                                        }  ## end of: if (use_resident_burnin)
                                        ##
                                        push_elapsed_seconds <- proc.time()[3] - t0
                                        t_push_total <- t_push_total + push_elapsed_seconds
                                        t0 <- proc.time()[3]
                                        ##
                                        ## Run one iteration (all heavy state resident in C++):
                                        ## (CHESSR_time / SNAPER_time: the step size this iteration runs with, and the wall time of the native call)
                                        if (bulk_local_tuner_freeze_active) {
                                            bulk_local_tuner_eps_used <-  EHMC_args_as_Rcpp_List$eps_main
                                            bulk_local_tuner_tau_used <-  EHMC_args_as_Rcpp_List$tau_main
                                            bulk_local_tuner_signature <-  list(
                                                  model_type = Model_type,
                                                  model_args = Model_args_as_Rcpp_List,
                                                  metric = EHMC_Metric_as_Rcpp_List,
                                                  eps = EHMC_args_as_Rcpp_List$eps_main,
                                                  eps_us = EHMC_args_as_Rcpp_List$eps_us,
                                                  diffusion_HMC = EHMC_args_as_Rcpp_List$diffusion_HMC,
                                                  diffusion_HMC_integrator = diffusion_HMC_integrator,
                                                  nuisance_metric = if (use_resident_burnin) {
                                                        resident_api$fn_persistent_burnin_get_resident_statistic(
                                                              worker_ptr, "M_us_vec")
                                                  } else EHMC_Metric_as_Rcpp_List$M_us_vec,
                                                  nuisance_centre = if (use_resident_burnin) {
                                                        resident_api$fn_persistent_burnin_get_resident_statistic(
                                                              worker_ptr, "theta_hat_us_vec")
                                                  } else EHMC_Metric_as_Rcpp_List$theta_hat_us_vec)
                                        }
                                        if (time_criterion_active) {
                                            eps_main_used_for_iteration <-  EHMC_args_as_Rcpp_List$eps_main
                                            native_call_start_time <-  Sys.time()
                                        }
                                        tau_jitter_iteration <-  if (ii < clip_iter) NULL else
                                                                   fn_tau_jitter_burnin_draw(tau_jitter_burnin_state,
                                                                                              ii - clip_iter + 1,
                                                                                              EHMC_args_as_Rcpp_List)
                                        if (is.null(tau_jitter_iteration)) {
                                            result <- run_burnin_iteration(
                                                      worker_ptr,
                                                      seed_R = seed + ii,
                                                      current_iter_R = ii)
                                        } else {
                                            result <-  run_burnin_iteration_tau_jitter(
                                                       worker_ptr,
                                                       seed_R = seed + ii,
                                                       current_iter_R = ii,
                                                       tau_main_ii_R = tau_jitter_iteration$tau_main_ii,
                                                       tau_us_ii_R = tau_jitter_iteration$tau_us_ii,
                                                       main_only_R = use_resident_burnin,
                                                       profile_burnin_R = debug_burnin_timing)
                                        }
                                        if (time_criterion_active) {
                                            time_per_iter_native_call_burnin_vec[ii] <-  as.numeric(difftime(Sys.time(), native_call_start_time, units = "secs"))
                                        }
                                        ##
                                        ## Everything AFTER this point in the loop stays byte-for-byte identical:
                                        ## the returned list has the same element names, so all the
                                        ## "theta_main_vectors_all_chains_input_from_R <- result$..." unpacking,
                                        ## snaper updates, eps/tau ADAM, metric updates, printing -- all unchanged.
                                        # ##
                                        cpp_elapsed_seconds <- proc.time()[3] - t0
                                        t_cpp_total <- t_cpp_total + cpp_elapsed_seconds
                                        if (debug_burnin_timing) {
                                            profile <- result$burnin_profile
                                            if (is.null(profile)) stop("The native burn-in profiler returned no diagnostics.")
                                            burnin_iteration_profiles[[ii]] <- data.frame(
                                                iter = ii,
                                                n_chains_burnin = n_chains_burnin,
                                                n_threads_WCP = n_threads_WCP,
                                                cpp_call_seconds = unname(cpp_elapsed_seconds),
                                                push_seconds = unname(push_elapsed_seconds),
                                                parallel_seconds = profile$parallel_seconds,
                                                output_seconds = profile$output_seconds,
                                                call_residual_seconds = unname(cpp_elapsed_seconds) - profile$parallel_seconds - profile$output_seconds,
                                                max_chain_seconds = max(profile$chain_seconds),
                                                mean_chain_seconds = mean(profile$chain_seconds),
                                                total_leapfrog_steps = sum(profile$n_leapfrog_steps))
                                            burnin_chain_profiles[[ii]] <- data.frame(
                                                iter = ii,
                                                chain = seq_len(n_chains_burnin),
                                                elapsed_seconds = profile$chain_seconds,
                                                start_offset_seconds = profile$chain_start_offset_seconds,
                                                finish_offset_seconds = profile$chain_start_offset_seconds + profile$chain_seconds,
                                                n_leapfrog_steps = profile$n_leapfrog_steps)
                                        }
                                    
                                    
                                        # if (ii %% 25 == 0) {
                                        # 
                                        #     ## us_only
                                        #     ## corr_only
                                        #     ## coeff_only
                                        #     ## prev_only
                                        #     ## cutpoints_only
                                        #     ##
                                        #     ## L_diag
                                        # 
                                        #     # ##
                                        #     BayesMVP:::set_debug_cutpoint_grads(TRUE)   # debug on
                                        # 
                                        #     Model_args_as_Rcpp_List$Model_args_ints[4, 1] <- 10
                                        #     ##
                                        #     lp_grad_outs_AD <- BayesMVP:::Rcpp_wrapper_fn_lp_grad(Model_type = Model_type,
                                        #                                                              force_autodiff = FALSE,
                                        #                                                              force_PartialLog = FALSE,
                                        #                                                              multi_attempts = FALSE,
                                        #                                                              theta_main_vec = theta_main_vectors_all_chains_input_from_R[, 1],
                                        #                                                              theta_us_vec   =  theta_us_vectors_all_chains_input_from_R[, 1],
                                        #                                                              y = y,
                                        #                                                              grad_option = "all",
                                        #                                                              Model_args_as_Rcpp_List = Model_args_as_Rcpp_List)
                                        #     ##
                                        #     Model_args_as_Rcpp_List$Model_args_ints[4, 1] <- 10
                                        #     ##
                                        #     # print(paste("lp_grad_outs_AD = ", head(lp_grad_outs_AD)))
                                        #     ##
                                        #     lp_grad_outs_MD <- BayesMVP:::Rcpp_wrapper_fn_lp_grad(Model_type = Model_type,
                                        #                                                              force_autodiff = FALSE,
                                        #                                                              force_PartialLog = FALSE,
                                        #                                                              multi_attempts = FALSE,
                                        #                                                              theta_main_vec = theta_main_vectors_all_chains_input_from_R[, 1],
                                        #                                                              theta_us_vec   = theta_us_vectors_all_chains_input_from_R[, 1],
                                        #                                                              y = y,
                                        #                                                              grad_option = "all",
                                        #                                                              Model_args_as_Rcpp_List = Model_args_as_Rcpp_List)
                                        #     ##
                                        #     # print(paste("lp_grad_outs_MD = ", head(lp_grad_outs_MD)))
                                        #     # ##
                                        #     # print(paste("diff_sum = ", sum(lp_grad_outs_AD - lp_grad_outs_MD)))
                                        #     ##
                                        #     print(paste("diffs (head - lp + first 99 u's) = ", head(lp_grad_outs_AD - lp_grad_outs_MD, 100)))
                                        #     ##
                                        #     ## ---- Main params:
                                        #     ##
                                        #     n_pars_main <- n_params_main
                                        #     ##
                                        #     print(paste("AD (grad of last n_pars_main params) = ",
                                        #                 head( tail((lp_grad_outs_AD), N + n_pars_main) , n_pars_main)  ))
                                        #     print(paste("MD (grad of last n_pars_main params) = ",
                                        #                 head( tail((lp_grad_outs_MD), N + n_pars_main) , n_pars_main)  ))
                                        #     ##
                                        #     # print(paste("AD (grad of last n_pars_main params - index 74 to 79) = ",
                                        #     #             head( tail((lp_grad_outs_AD), N + n_pars_main) , n_pars_main)[74:79]   ))
                                        #     # print(paste("MD (grad of last n_pars_main params - index 74 to 79) = ",
                                        #     #             head( tail((lp_grad_outs_MD), N + n_pars_main) , n_pars_main)[74:79]   ))
                                        #     ##
                                        #     print(paste("diffs (grad of last n_pars_main params) = ",
                                        #                 head( tail((lp_grad_outs_AD - lp_grad_outs_MD), N + n_pars_main) , n_pars_main)  ))
                                        #     # ##
                                        #     # print(paste("diffs (grad of last n_pars_main params - index 74:79) = ",
                                        #     #             head( tail((lp_grad_outs_AD - lp_grad_outs_MD), N + n_pars_main) , n_pars_main)[74:79] ))
                                        #     ##
                                        #     ll_1  <- tail(lp_grad_outs_AD, N)
                                        #     ll_10 <- tail(lp_grad_outs_MD, N)
                                        #     d <- ll_1 - ll_10
                                        #     print(c(sum_d = sum(d), max_abs = max(abs(d)), n_big = sum(abs(d) > 0.05), n_zero_slots = sum(ll_10 == 0)))
                                        #     print(which(abs(d) > 0.05))
                                        #     ##
                                        #     BayesMVP:::set_debug_cutpoint_grads(FALSE)  # debug off
                                        #     ##
                                        #     # ## The main params are ordered: correlations, coefficients, prev
                                        #     # ## With n_corrs=30, what are params 46 and 52?
                                        #     # n_corrs <- 30
                                        #     # n_covariates_total <- sum(model_args_list$n_covariates_per_outcome_mat)
                                        #     # cat(sprintf("n_corrs=%d, n_covs_total=%d, n_pops=%d, total=%d\n",
                                        #     #             n_corrs, n_covariates_total, model_args_list$n_pops,
                                        #     #             n_corrs + n_covariates_total + model_args_list$n_pops))
                                        #     #
                                        #     # ## Map index to parameter:
                                        #     # for (idx in c(35, 41)) {
                                        #     #   if (idx <= n_corrs) {
                                        #     #     cat(sprintf("Index %d = correlation param %d\n", idx, idx))
                                        #     #   } else if (idx <= n_corrs + n_covariates_total) {
                                        #     #     coeff_idx <- idx - n_corrs
                                        #     #     cat(sprintf("Index %d = coefficient %d\n", idx, coeff_idx))
                                        #     #   } else {
                                        #     #     prev_idx <- idx - n_corrs - n_covariates_total
                                        #     #     cat(sprintf("Index %d = prevalence pop %d\n", idx, prev_idx))
                                        #     #   }
                                        #     # }
                                        # 
                                        # 
                                        # }

                           #  result <- parallel::mccollect(result)

                            theta_main_vectors_all_chains_input_from_R <-                  result$theta_main_vectors_all_chains_output_to_R
                            theta_main_0_burnin_tau_adapt_all_chains_input_from_R <-       result$theta_main_0_burnin_tau_adapt_all_chains_input_from_R
                            theta_main_prop_burnin_tau_adapt_all_chains_input_from_R <-    result$theta_main_prop_burnin_tau_adapt_all_chains_input_from_R
                            ##
                            velocity_main_vectors_all_chains_input_from_R <-               result$velocity_main_vectors_all_chains_output_to_R
                            velocity_main_0_burnin_tau_adapt_all_chains_input_from_R <-    result$velocity_main_0_burnin_tau_adapt_all_chains_input_from_R
                            velocity_main_prop_burnin_tau_adapt_all_chains_input_from_R <- result$velocity_main_prop_burnin_tau_adapt_all_chains_input_from_R
                            ##
                            tau_main_ii_vec <- result[[3]][6,]
                            if (bulk_local_tuner_freeze_active && ii <= bulk_local_tuner_decision_iter) {
                                bulk_local_tuner_state <-  fn_bulk_local_tuner_append(
                                      state = bulk_local_tuner_state,
                                      theta_initial = theta_main_0_burnin_tau_adapt_all_chains_input_from_R,
                                      theta_accepted = theta_main_vectors_all_chains_input_from_R,
                                      leapfrog_steps = pmax(1, ceiling(tau_main_ii_vec /
                                                                       bulk_local_tuner_eps_used)),
                                      nominal_tau = bulk_local_tuner_tau_used,
                                      eps = bulk_local_tuner_eps_used,
                                      kernel_signature = bulk_local_tuner_signature,
                                      divergences = result$other_main_out_vector_all_chains_output_to_R[2, ],
                                      shared_jitter = isTRUE(share_tau_ii_across_chains_in_burnin))
                                if (ii == bulk_local_tuner_decision_iter) {
                                    bulk_local_tuner_decision <-  fn_bulk_local_tuner_decide(
                                          bulk_local_tuner_state)
                                    bulk_local_tuner_metadata$decision <-  bulk_local_tuner_decision
                                    bulk_local_tuner_metadata$status <-  bulk_local_tuner_decision$status
                                    bulk_local_tuner_metadata$reason <-  bulk_local_tuner_decision$reason
                                    bulk_local_tuner_selected <-  bulk_local_tuner_decision$tau_selected
                                    if (all(is.finite(bulk_local_tuner_selected)) &&
                                        all(bulk_local_tuner_selected > 0) &&
                                        length(unique(bulk_local_tuner_selected)) == 1 &&
                                        all(bulk_local_tuner_selected <= max_tau_main) &&
                                        (!isTRUE(sample_nuisance) ||
                                         all(bulk_local_tuner_selected <= max_tau_nuisance))) {
                                        bulk_local_tuner_frozen_kernel$tau_main <-  bulk_local_tuner_selected[1]
                                        bulk_local_tuner_frozen_kernel$tau_us <-  bulk_local_tuner_selected[1]
                                        if (identical(bulk_local_tuner_decision$status, "select_shorter_tau")) {
                                            bulk_local_tuner_metadata$actual_tau_adaptation_end_iter <-  ii
                                            bulk_local_tuner_metadata$final_sampling_kernel_start_iter <-  ii + 1
                                        }
                                    } else {
                                        bulk_local_tuner_metadata$status <-  "skipped"
                                        bulk_local_tuner_metadata$reason <-
                                              "selection_outside_shared_tau_contract"
                                    }
                                }
                            }
                            
                            # print( result$theta_us_prop_burnin_tau_adapt_all_chains_input_from_R)
                            
                         # if (sample_nuisance == TRUE) { 
                           
                                if (!use_resident_burnin) {   ## (resident interface: the nuisance-sized state stays in the worker)
                                theta_us_vectors_all_chains_input_from_R <-                  result$theta_us_vectors_all_chains_output_to_R
                                theta_us_0_burnin_tau_adapt_all_chains_input_from_R <-       result$theta_us_0_burnin_tau_adapt_all_chains_input_from_R
                                theta_us_prop_burnin_tau_adapt_all_chains_input_from_R <-    result$theta_us_prop_burnin_tau_adapt_all_chains_input_from_R
                                ##
                                velocity_us_vectors_all_chains_input_from_R <-               result$velocity_us_vectors_all_chains_output_to_R
                                velocity_us_0_burnin_tau_adapt_all_chains_input_from_R <-    result$velocity_us_0_burnin_tau_adapt_all_chains_input_from_R
                                velocity_us_prop_burnin_tau_adapt_all_chains_input_from_R <- result$velocity_us_prop_burnin_tau_adapt_all_chains_input_from_R
                                ##
                                }  ## end of: if (!use_resident_burnin)
                                tau_us_ii_vec <-   result[[6]][6,]
                                if (!use_resident_burnin) {
                                if (debug) {
                                  if (ii %% 25 == 0) {
                                      cat("velocity_us_0 range:", range(velocity_us_0_burnin_tau_adapt_all_chains_input_from_R), "\n")
                                      cat("velocity_us_prop range:", range(velocity_us_prop_burnin_tau_adapt_all_chains_input_from_R), "\n")
                                  }
                                }
                                }  ## end of: if (!use_resident_burnin)
                         # }
 
          })
   
   
   
   
                        try({
                          
                              div_main <- div_us <- c()
                              p_jump_per_chain <- p_jump_us_per_chain <- c()
                              
                              for (kk in 1:n_chains_burnin) {
                                  div_main[kk] <- result$other_main_out_vector_all_chains_output_to_R[, kk][2]
                                  p_jump_per_chain[kk] <-   result$other_main_out_vector_all_chains_output_to_R[, kk][1]
                                  if (sample_nuisance == TRUE) { 
                                      div_us[kk] <- result$other_us_out_vector_all_chains_output_to_R[, kk][2]
                                      p_jump_us_per_chain[kk] <-   result$other_us_out_vector_all_chains_output_to_R[, kk][1]
                                  }
                              }
                          
                        })
                         
                         # ## ---- Phase-2 accumulation + blended assignment (TOP LEVEL of loop) ----
                         # if (metric_type_main == "Hessian" && metric_shape_main == "dense") {
                         #   tryCatch({
                         #     
                         #     if (ii == win_first_start) {
                         #       H_inv_seed <- EHMC_Metric_as_Rcpp_List$M_inv_dense_main
                         #       message("ii = ", ii, ": Hessian metric snapshotted as Phase-2 prior")
                         #     }
                         #     
                         #     if (ii > win_first_start && ii <= win_metric_end) {
                         #       Xc <- theta_main_vectors_all_chains_input_from_R
                         #       win_sum_x   <- win_sum_x   + rowSums(Xc)
                         #       win_sum_xxT <- win_sum_xxT + tcrossprod(Xc)
                         #       win_n       <- win_n + ncol(Xc)
                         #       if (!is.null(win_prev_X)) {
                         #         for (jj in 1:min(5, n_params_main)) {
                         #           r1 <- suppressWarnings(cor(Xc[jj, ], win_prev_X[jj, ]))
                         #           if (is.finite(r1)) { win_rho_sum <- win_rho_sum + r1; win_rho_n <- win_rho_n + 1L }
                         #         }
                         #       }
                         #       win_prev_X <- Xc
                         #     }
                         #     
                         #     if ((ii %in% win_ends) && (win_n > 5L) && !is.null(H_inv_seed)) {
                         #       p      <- n_params_main
                         #       n_used <- win_n
                         #       mean_w <- win_sum_x / n_used
                         #       cov_w  <- (win_sum_xxT - n_used * tcrossprod(mean_w)) / (n_used - 1)
                         #       cov_w  <- 0.5 * (cov_w + t(cov_w))
                         #       rho_hat <- if (win_rho_n > 0) max(0, min(0.99, win_rho_sum / win_rho_n)) else 0.9
                         #       tau_hat <- (1 + rho_hat) / (1 - rho_hat)
                         #       n_eff   <- n_used / tau_hat
                         #       # w_blend <- n_eff / (n_eff + 5 * p)
                         #       w_blend <- n_eff / (n_eff + 20 * p)
                         #       cov_blend <- w_blend * cov_w + (1 - w_blend) * H_inv_seed
                         #       cov_blend <- cov_blend + 1e-4 * mean(diag(cov_blend)) * diag(p)
                         #       cov_blend <- BayesMVP:::Rcpp_near_PD(0.5 * (cov_blend + t(cov_blend)))
                         #       EHMC_Metric_as_Rcpp_List$M_inv_dense_main      <- cov_blend
                         #       EHMC_Metric_as_Rcpp_List$M_dense_main          <- BayesMVP:::Rcpp_solve(cov_blend)
                         #       EHMC_Metric_as_Rcpp_List$M_inv_dense_main_chol <- BayesMVP:::Rcpp_Chol(cov_blend)
                         #       EHMC_burnin_as_Rcpp_List$M_dense_sqrt          <- pracma::sqrtm(EHMC_Metric_as_Rcpp_List$M_dense_main)[[1]]
                         #       EHMC_Metric_as_Rcpp_List$M_inv_main_vec  <- diag(cov_blend)
                         #       EHMC_Metric_as_Rcpp_List$M_main_vec      <- 1 / diag(cov_blend)
                         #       EHMC_burnin_as_Rcpp_List$sqrt_M_main_vec <- sqrt(EHMC_Metric_as_Rcpp_List$M_main_vec)
                         #       message(sprintf("ii = %d: Phase-2 metric assigned (n=%d, n_eff~%.0f, w=%.2f, tau_hat~%.0f)",
                         #                       ii, n_used, n_eff, w_blend, tau_hat))
                         #       
                         #       # ## re-initialize eps against the NEW metric + restore tau:
                         #       # try({
                         #       #   par_res <- BayesMVP:::fn_find_initial_eps_main_and_us(
                         #       #     theta_main_vec_initial_ref = matrix(rowMeans(theta_main_vectors_all_chains_input_from_R), ncol = 1),
                         #       #     theta_us_vec_initial_ref   = matrix(rowMeans(theta_us_vectors_all_chains_input_from_R), ncol = 1),
                         #       #     partitioned_HMC = partitioned_HMC, seed = seed + ii,
                         #       #     Model_type = Model_type,
                         #       #     force_autodiff = force_autodiff, force_PartialLog = force_PartialLog,
                         #       #     multi_attempts = multi_attempts, y_ref = y,
                         #       #     Model_args_as_Rcpp_List = Model_args_as_Rcpp_List,
                         #       #     EHMC_args_as_Rcpp_List = EHMC_args_as_Rcpp_List,
                         #       #     EHMC_Metric_as_Rcpp_List = EHMC_Metric_as_Rcpp_List)
                         #       #   EHMC_args_as_Rcpp_List$eps_main <- min(max_eps_main, par_res[[1]])
                         #       #   EHMC_burnin_as_Rcpp_List$eps_m_adam_main <- EHMC_args_as_Rcpp_List$eps_main
                         #       #   EHMC_burnin_as_Rcpp_List$eps_v_adam_main <- 0
                         #       #   EHMC_args_as_Rcpp_List$tau_main <- max(EHMC_args_as_Rcpp_List$tau_main,
                         #       #                                          12 * EHMC_args_as_Rcpp_List$eps_main)
                         #       #   message(sprintf("ii = %d: eps re-initialized to %.4f, tau restored to %.3f",
                         #       #                   ii, EHMC_args_as_Rcpp_List$eps_main, EHMC_args_as_Rcpp_List$tau_main))
                         #       # })
                         #       
                         #       ## analytic eps rescale against the new metric:
                         #       ev_old <- eigen(H_inv_seed, symmetric = TRUE, only.values = TRUE)$values
                         #       ev_new <- eigen(cov_blend,  symmetric = TRUE, only.values = TRUE)$values
                         #       scale_fac <- sqrt(max(ev_old) / max(ev_new))
                         #       EHMC_args_as_Rcpp_List$eps_main <- max(0.01, min(max_eps_main,
                         #                                                        EHMC_args_as_Rcpp_List$eps_main * scale_fac))
                         #       EHMC_burnin_as_Rcpp_List$eps_m_adam_main <- EHMC_args_as_Rcpp_List$eps_main
                         #       EHMC_burnin_as_Rcpp_List$eps_v_adam_main <- 0
                         #       EHMC_args_as_Rcpp_List$tau_main <- min(EHMC_args_as_Rcpp_List$tau_main,
                         #                                              40 * EHMC_args_as_Rcpp_List$eps_main)
                         #       message(sprintf("ii = %d: eps rescaled to %.4f (factor %.2f), tau capped to %.3f",
                         #                       ii, EHMC_args_as_Rcpp_List$eps_main, scale_fac, EHMC_args_as_Rcpp_List$tau_main))
                         #       
                         #       win_sum_x[] <- 0.0; win_sum_xxT[] <- 0.0; win_n <- 0L
                         #       win_prev_X <- NULL; win_rho_sum <- 0.0; win_rho_n <- 0L
                         #     }
                         #     
                         #   }, error = function(e) message("PHASE-2 METRIC FAILED at ii = ", ii, ": ", conditionMessage(e)))
                         # }
   
                         ## ---- Per-iteration cross-chain covariance (the correct estimator) ----
                         # try({
                         #   # if (ii > clip_iter) {   ## skip the earliest transient
                         #     if (ii > cov_start_iter) {
                         #       cov_ii <- cov(t(theta_main_vectors_all_chains_input_from_R))
                         #       cov_ii <- cov_ii + 1e-3 * mean(diag(cov_ii)) * diag(n_params_main)   # rank floor
                         #     # # cov_ii <- cov(t(theta_main_vectors_all_chains_input_from_R))   ## K draws -> p x p
                         #     # if (is.null(cov_crosschain_ema)) {
                         #     #   cov_crosschain_ema <- cov_ii
                         #     # } else {
                         #     #   cov_crosschain_ema <- ema_alpha * cov_ii + (1 - ema_alpha) * cov_crosschain_ema
                         #     # }
                         #   }
                         # })
   
                     
                     # tryCatch({
                     #   if (ii > cov_start_iter) {
                     #     cov_ii <- cov(t(theta_main_vectors_all_chains_input_from_R))
                     #     cov_ii <- cov_ii + 1e-3 * mean(diag(cov_ii)) * diag(nrow(cov_ii))
                     #     if (is.null(cov_crosschain_ema)) {
                     #       cov_crosschain_ema <- cov_ii
                     #     } else {
                     #       cov_crosschain_ema <- ema_alpha * cov_ii + (1 - ema_alpha) * cov_crosschain_ema
                     #     }
                     #   }
                     # }, error = function(e) message("COV ACCUM FAILED at ii=", ii, ": ", conditionMessage(e)))
                       
                       # ## ---- Windowed metric: accumulate + assign at window ends ----
                       # if (metric_type_main == "Empirical" && metric_shape_main == "dense") {
                       #   tryCatch({
                       #     if (ii > win_init_end && ii <= win_slow_end) {
                       #       Xc <- theta_main_vectors_all_chains_input_from_R          # p x K
                       #       win_sum_x   <- win_sum_x   + rowSums(Xc)
                       #       win_sum_xxT <- win_sum_xxT + tcrossprod(Xc)
                       #       win_n       <- win_n + ncol(Xc)
                       #     }
                       #     if ((ii %in% win_ends) && (win_n > 5L)) {
                       #       p      <- n_params_main
                       #       n_used <- win_n
                       #       mean_w <- win_sum_x / n_used
                       #       cov_w  <- (win_sum_xxT - n_used * tcrossprod(mean_w)) / (n_used - 1)
                       #       cov_w  <- 0.5 * (cov_w + t(cov_w))                                    # symmetrize
                       #       cov_w  <- (n_used / (n_used + 5)) * cov_w +
                       #         (5 / (n_used + 5) + 1e-4) * mean(diag(cov_w)) * diag(p)     # Stan reg + ridge floor
                       #       cov_w  <- BayesMVP:::Rcpp_near_PD(cov_w)
                       #       ## DIRECT assignment of the full consistent triple:
                       #       EHMC_Metric_as_Rcpp_List$M_inv_dense_main      <- cov_w
                       #       EHMC_Metric_as_Rcpp_List$M_dense_main          <- BayesMVP:::Rcpp_solve(cov_w)
                       #       EHMC_Metric_as_Rcpp_List$M_inv_dense_main_chol <- BayesMVP:::Rcpp_Chol(cov_w)
                       #       EHMC_burnin_as_Rcpp_List$M_dense_sqrt          <- pracma::sqrtm(EHMC_Metric_as_Rcpp_List$M_dense_main)[[1]]
                       #       ## diagonal mirrors:
                       #       EHMC_Metric_as_Rcpp_List$M_inv_main_vec  <- diag(cov_w)
                       #       EHMC_Metric_as_Rcpp_List$M_main_vec      <- 1 / diag(cov_w)
                       #       EHMC_burnin_as_Rcpp_List$sqrt_M_main_vec <- sqrt(EHMC_Metric_as_Rcpp_List$M_main_vec)
                       #       ## restart eps adaptation against the new metric (Stan-style):
                       #       EHMC_burnin_as_Rcpp_List$eps_v_adam_main <- 0
                       #       EHMC_burnin_as_Rcpp_List$eps_m_adam_main <- EHMC_args_as_Rcpp_List$eps_main
                       #       ## reset window accumulators:
                       #       win_sum_x[]   <- 0.0
                       #       win_sum_xxT[] <- 0.0
                       #       win_n         <- 0L
                       #       message("ii = ", ii, ": metric DIRECTLY assigned from window (", n_used, " draws)")
                       #     }
                       #   }, error = function(e) message("WINDOWED METRIC FAILED at ii = ", ii, ": ", conditionMessage(e)))
                       # }
                       
                       # ## ---- Phase-2 accumulation + blended assignment ----
                       # if (metric_type_main == "Hessian" && metric_shape_main == "dense") {
                       #   tryCatch({
                       #     
                       #     ## snapshot the Hessian metric once, at handover:
                       #     if (ii == win_first_start) {
                       #       H_inv_seed <- EHMC_Metric_as_Rcpp_List$M_inv_dense_main
                       #       message("ii = ", ii, ": Hessian metric snapshotted as Phase-2 prior")
                       #     }
                       #     
                       #     ## accumulate:
                       #     if (ii > win_first_start && ii <= win_metric_end) {
                       #       Xc <- theta_main_vectors_all_chains_input_from_R           # p x K
                       #       win_sum_x   <- win_sum_x   + rowSums(Xc)
                       #       win_sum_xxT <- win_sum_xxT + tcrossprod(Xc)
                       #       win_n       <- win_n + ncol(Xc)
                       #       ## lag-1 autocorrelation, first 5 params, averaged over chains:
                       #       if (!is.null(win_prev_X)) {
                       #         for (jj in 1:min(5, n_params_main)) {
                       #           r1 <- suppressWarnings(cor(Xc[jj, ], win_prev_X[jj, ]))
                       #           if (is.finite(r1)) { win_rho_sum <- win_rho_sum + r1; win_rho_n <- win_rho_n + 1L }
                       #         }
                       #       }
                       #       win_prev_X <- Xc
                       #     }
                       #     
                       #     ## at window end: blend with Hessian prior, assign directly:
                       #     if ((ii %in% win_ends) && (win_n > 5L) && !is.null(H_inv_seed)) {
                       #       p      <- n_params_main
                       #       n_used <- win_n
                       #       mean_w <- win_sum_x / n_used
                       #       cov_w  <- (win_sum_xxT - n_used * tcrossprod(mean_w)) / (n_used - 1)
                       #       cov_w  <- 0.5 * (cov_w + t(cov_w))
                       #       ## effective sample size of the window:
                       #       rho_hat <- if (win_rho_n > 0) max(0, min(0.99, win_rho_sum / win_rho_n)) else 0.9
                       #       tau_hat <- (1 + rho_hat) / (1 - rho_hat)
                       #       n_eff   <- n_used / tau_hat
                       #       w_blend <- n_eff / (n_eff + 5 * p)
                       #       ## shrinkage blend toward the Hessian prior:
                       #       cov_blend <- w_blend * cov_w + (1 - w_blend) * H_inv_seed
                       #       cov_blend <- cov_blend + 1e-4 * mean(diag(cov_blend)) * diag(p)
                       #       cov_blend <- BayesMVP:::Rcpp_near_PD(0.5 * (cov_blend + t(cov_blend)))
                       #       ## direct assignment of the consistent triple:
                       #       EHMC_Metric_as_Rcpp_List$M_inv_dense_main      <- cov_blend
                       #       EHMC_Metric_as_Rcpp_List$M_dense_main          <- BayesMVP:::Rcpp_solve(cov_blend)
                       #       EHMC_Metric_as_Rcpp_List$M_inv_dense_main_chol <- BayesMVP:::Rcpp_Chol(cov_blend)
                       #       EHMC_burnin_as_Rcpp_List$M_dense_sqrt          <- pracma::sqrtm(EHMC_Metric_as_Rcpp_List$M_dense_main)[[1]]
                       #       EHMC_Metric_as_Rcpp_List$M_inv_main_vec  <- diag(cov_blend)
                       #       EHMC_Metric_as_Rcpp_List$M_main_vec      <- 1 / diag(cov_blend)
                       #       EHMC_burnin_as_Rcpp_List$sqrt_M_main_vec <- sqrt(EHMC_Metric_as_Rcpp_List$M_main_vec)
                       #       ## restart eps adaptation against the new metric:
                       #       EHMC_burnin_as_Rcpp_List$eps_v_adam_main <- 0
                       #       EHMC_burnin_as_Rcpp_List$eps_m_adam_main <- EHMC_args_as_Rcpp_List$eps_main
                       #       ## reset window:
                       #       win_sum_x[] <- 0.0; win_sum_xxT[] <- 0.0; win_n <- 0L
                       #       win_prev_X <- NULL; win_rho_sum <- 0.0; win_rho_n <- 0L
                       #       message(sprintf("ii = %d: Phase-2 metric assigned (n=%d, n_eff~%.0f, w=%.2f, tau_hat~%.0f)",
                       #                       ii, n_used, n_eff, w_blend, tau_hat))
                       #     }
                       #     
                       #   }, error = function(e) message("PHASE-2 METRIC FAILED at ii = ", ii, ": ", conditionMessage(e)))
                       # }
   
                        ## ---- CHESSR_time / SNAPER_time: add this iteration to the online fit of time_per_leapfrog_step_burnin (iterations from
                        ##      clip_iter on with no divergent chain; step count = the longest trajectory of the iteration, max(1, ceiling(tau_ii / eps)),
                        ##      see fn_time_per_leapfrog_step_burnin_fit_update in R_fn_time_criterion.R):
                        ##
                        if (time_criterion_active) {
                            try({
                                n_leapfrog_steps_per_iter_burnin_vec[ii] <-  max(1, ceiling(max(tau_main_ii_vec) / eps_main_used_for_iteration))
                                if (ii >= clip_iter && length(div_main) == n_chains_burnin && all(is.finite(div_main)) && sum(div_main) == 0) {
                                    time_per_leapfrog_step_burnin_fit_sums <-  fn_time_per_leapfrog_step_burnin_fit_update(
                                        time_per_leapfrog_step_burnin_fit_sums = time_per_leapfrog_step_burnin_fit_sums,
                                        n_leapfrog_steps_per_iter              = n_leapfrog_steps_per_iter_burnin_vec[ii],
                                        time_per_iter                          = time_per_iter_native_call_burnin_vec[ii])
                                    used_in_time_per_leapfrog_step_burnin_fit_vec[ii] <-  TRUE
                                }
                            })
                        }
   
                        ## ---- divergence-triggered tau shrink (tau_shrink_on_divergence; OFF by default). div_main holds one
                        ##      0/1 divergence flag per burn-in chain, so "sum(div_main) > (tau_shrink_min_divergent_chains - 1)"
                        ##      with the defaults (0.95, 2) is exactly the original "sum(div_main) > 1" rule:
                        ##
                        ## if (!isTRUE(manual_tau) && ii > clip_iter && ii < n_adapt && sum(div_main) > 1) {
                        if (isTRUE(tau_shrink_on_divergence) && !isTRUE(manual_tau) && ii > clip_iter && ii < n_adapt &&
                            sum(div_main) > (tau_shrink_min_divergent_chains - 1)) {
                             warning(paste0("Too many divergences at iter ", ii, ", reducing tau"))
                             ##
                             ## EHMC_args_as_Rcpp_List$tau_main <- 0.95 * EHMC_args_as_Rcpp_List$tau_main
                             ## EHMC_args_as_Rcpp_List$tau_us   <- 0.95 * EHMC_args_as_Rcpp_List$tau_us
                             EHMC_args_as_Rcpp_List$tau_main <- tau_shrink_factor * EHMC_args_as_Rcpp_List$tau_main
                             EHMC_args_as_Rcpp_List$tau_us   <- tau_shrink_factor * EHMC_args_as_Rcpp_List$tau_us
                             ##
                             tau_shrink_fired_vec[ii] <- TRUE
                        }
   
                        # if (ii == cov_start_iter) {
                        #      cat("cov accumulation starts; ensemble spread (max sd):",
                        #      max(apply(theta_main_vectors_all_chains_input_from_R, 1, sd)), "\n")
                        # }
  
                        p_jump_main <- mean(p_jump_per_chain, na.rm = TRUE)
                        if (sample_nuisance == TRUE) {
                           p_jump_us <- mean(p_jump_us_per_chain, na.rm = TRUE)
                        }
                         # p_jump_main <- median(p_jump_per_chain, na.rm = TRUE)
                         # if (sample_nuisance == TRUE) { 
                         #   p_jump_us <- median(p_jump_us_per_chain, na.rm = TRUE)
                         # }

                        ## ---- cross-chain acceptance fed to the eps update (eps_acceptance_mean). "arithmetic" reuses p_jump_main /
                        ##      p_jump_us above unchanged; "harmonic" is K / sum_k (1 / alpha_k), exactly 0 when any chain has zero
                        ##      acceptance (see R_fn_harmonic_mean_acceptance.R). p_jump_main / p_jump_us stay the arithmetic means (printed).
                        ##      "geometric" is exp(mean(log(alpha_k))) with alpha_k clamped to [1e-8, 1] (fn_geometric_mean_acceptance).
                        # p_jump_main_for_eps <- if (identical(eps_acceptance_mean, "arithmetic")) p_jump_main else
                        #                        fn_harmonic_mean_acceptance( p     = p_jump_per_chain,
                        #                                                     floor = 0)
                        p_jump_main_for_eps <- if (identical(eps_acceptance_mean, "arithmetic")) p_jump_main else
                                               if (identical(eps_acceptance_mean, "geometric"))
                                               fn_geometric_mean_acceptance( p           = p_jump_per_chain,
                                                                             lower_clamp = 1e-8) else
                                               fn_harmonic_mean_acceptance( p     = p_jump_per_chain,
                                                                            floor = 0)
                        p_jump_main_during_burnin_vec[ii]         <- p_jump_main
                        p_jump_main_for_eps_during_burnin_vec[ii] <- p_jump_main_for_eps
                        if (sample_nuisance == TRUE) {
                           # p_jump_us_for_eps <- if (identical(eps_acceptance_mean, "arithmetic")) p_jump_us else
                           #                      fn_harmonic_mean_acceptance( p     = p_jump_us_per_chain,
                           #                                                   floor = 0)
                           p_jump_us_for_eps <- if (identical(eps_acceptance_mean, "arithmetic")) p_jump_us else
                                                if (identical(eps_acceptance_mean, "geometric"))
                                                fn_geometric_mean_acceptance( p           = p_jump_us_per_chain,
                                                                              lower_clamp = 1e-8) else
                                                fn_harmonic_mean_acceptance( p     = p_jump_us_per_chain,
                                                                             floor = 0)
                           p_jump_us_during_burnin_vec[ii]         <- p_jump_us
                           p_jump_us_for_eps_during_burnin_vec[ii] <- p_jump_us_for_eps
                        }

                        if (ii < n_adapt && !bulk_local_tuner_freeze_active) {

                            ## Step-size ADAM moments reset at the final metric update (see the switches above):
                            ## the first
                            ## acceptance measured with the final metric starts the ADAM afresh.
                            if (eps_adam_reset_at_metric_end && ii == metric_adaptation_end_iter) {
                                    EHMC_burnin_as_Rcpp_List$eps_m_adam_main <- 0
                                    EHMC_burnin_as_Rcpp_List$eps_v_adam_main <- 0
                                    EHMC_burnin_as_Rcpp_List$eps_m_adam_us   <- 0
                                    EHMC_burnin_as_Rcpp_List$eps_v_adam_us   <- 0
                                    eps_adam_updates_main <- 0
                                    eps_adam_updates_us <- 0
                                    message(colourise(paste0("step-size ADAM moments reset at the final ",
                                                             "metric update (iteration ", ii, "); eps_main = ",
                                                             signif(EHMC_args_as_Rcpp_List$eps_main, 4)),
                                                      "cyan"))
                            }
    
                            ## ii_eps / n_eps: the iteration and the length that the eps learning-rate
                            ## schedule sees at iteration ii (restarted at the final metric update when
                            ## the switch above is on); short names only because of the depth of the two
                            ## ADAM calls below.
                            ii_eps <-  fn_eps_schedule_iter(ii)
                            n_eps <-  fn_eps_schedule_length(ii)
                            ## //////////////////   --------------------------------  update eps (step-size) for main ----------------------------------------------------------
                                            adapt_eps_outs <-  fn_update_eps_ADAM( EHMC_args_as_Rcpp_List$eps_main,
                                                                                               EHMC_burnin_as_Rcpp_List$eps_m_adam_main,
                                                                                               EHMC_burnin_as_Rcpp_List$eps_v_adam_main,
                                                                                               ii_eps,
                                                                                               n_eps,
                                                                                               EHMC_burnin_as_Rcpp_List$LR_main,
                                                                                               ## p_jump_main,
                                                                                               p_jump_main_for_eps,
                                                                                               EHMC_burnin_as_Rcpp_List$adapt_delta_main,
                                                                                               ## beta1_adam,
                                                                                               eps_adam_beta1,
                                                                                               ## beta2_adam, 
                                                                                               ## eps_adam)
                                                                                               eps_adam_beta2,
                                                                                               eps_adam_epsilon,
                                                                                               bias_correction_step = eps_adam_updates_main + 1)
    
                                            EHMC_args_as_Rcpp_List$eps_main <-    min(max_eps_main, adapt_eps_outs[1])   ;  EHMC_args_as_Rcpp_List$eps_main
                                            EHMC_burnin_as_Rcpp_List$eps_m_adam_main <-  adapt_eps_outs[2]   ; EHMC_burnin_as_Rcpp_List$eps_m_adam_main
                                            EHMC_burnin_as_Rcpp_List$eps_v_adam_main <-  adapt_eps_outs[3]   ;    EHMC_burnin_as_Rcpp_List$eps_v_adam_main
                                            if (isTRUE(attr(adapt_eps_outs, "adam_update_performed"))) eps_adam_updates_main <- eps_adam_updates_main + 1
    
                            ## //////////////////   --------------------------------  update eps (step-size) for nuisance ------------------------------------------------------
                                            if ((partitioned_HMC == TRUE) && (sample_nuisance == TRUE)) {
                                                    adapt_eps_outs <-  fn_update_eps_ADAM(  EHMC_args_as_Rcpp_List$eps_us,
                                                                                                        EHMC_burnin_as_Rcpp_List$eps_m_adam_us,
                                                                                                        EHMC_burnin_as_Rcpp_List$eps_v_adam_us ,
                                                                                                        ii_eps,
                                                                                                        n_eps,
                                                                                                        EHMC_burnin_as_Rcpp_List$LR_us,
                                                                                                        ## p_jump_us,
                                                                                                        p_jump_us_for_eps,
                                                                                                        EHMC_burnin_as_Rcpp_List$adapt_delta_us,
                                                                                                        ## beta1_adam, 
                                                                                                        eps_adam_beta1, 
                                                                                                        ## beta2_adam, 
                                                                                                        ## eps_adam)
                                                                                                        eps_adam_beta2,
                                                                                                        eps_adam_epsilon,
                                                                                                        bias_correction_step = eps_adam_updates_us + 1)
            
                                                    EHMC_args_as_Rcpp_List$eps_us <-          min(max_eps_us,   adapt_eps_outs[1] )
                                                    EHMC_burnin_as_Rcpp_List$eps_m_adam_us  <- adapt_eps_outs[2]
                                                    EHMC_burnin_as_Rcpp_List$eps_v_adam_us <- adapt_eps_outs[3]
                                                    if (isTRUE(attr(adapt_eps_outs, "adam_update_performed"))) eps_adam_updates_us <- eps_adam_updates_us + 1
                                            }

                        }

                        ## //////////////////   --------------------------------  eps warm start (eps_initial) ------------------------------------------------------------
                        ##
                        ## eps set at the END of iteration ii is what iteration ii + 1 runs with, so:
                        ##   ii <  eps_initial_iter -> iteration ii + 1 is still inside the window: re-impose
                        ##                             eps_initial, discarding what ADAM just proposed;
                        ##   ii == eps_initial_iter -> the window ends here, so hand eps back to the value it
                        ##                             would have started at and reset the ADAM moments to it,
                        ##                             so the warm-start value is not carried forward.
                        ## Iterations 1 .. eps_initial_iter therefore run at eps_initial (it is also set before
                        ## the loop, which is what iteration 1 uses).
                        ##
                        if (use_eps_warm_start && !bulk_local_tuner_freeze_active) {

                              if (ii < eps_initial_iter) {

                                    EHMC_args_as_Rcpp_List$eps_main <- eps_initial_main
                                    EHMC_args_as_Rcpp_List$eps_us   <- eps_initial_us
                                    ##
                                    EHMC_burnin_as_Rcpp_List$eps_m_adam_main <- 0
                                    EHMC_burnin_as_Rcpp_List$eps_v_adam_main <- 0
                                    EHMC_burnin_as_Rcpp_List$eps_m_adam_us   <- 0
                                    EHMC_burnin_as_Rcpp_List$eps_v_adam_us   <- 0
                                    eps_adam_updates_main <- 0
                                    eps_adam_updates_us <- 0

                              } else if (ii == eps_initial_iter) {

                                    EHMC_args_as_Rcpp_List$eps_main <- eps_main_after_warm_start
                                    EHMC_args_as_Rcpp_List$eps_us   <- eps_us_after_warm_start
                                    ##
                                    EHMC_burnin_as_Rcpp_List$eps_m_adam_main <- 0
                                    EHMC_burnin_as_Rcpp_List$eps_v_adam_main <- 0
                                    EHMC_burnin_as_Rcpp_List$eps_m_adam_us   <- 0
                                    EHMC_burnin_as_Rcpp_List$eps_v_adam_us   <- 0
                                    eps_adam_updates_main <- 0
                                    eps_adam_updates_us <- 0
                                    ##
                                    message(paste0( "eps warm start finished at iteration ", ii,
                                                    ": eps reset to ", signif(eps_main_after_warm_start, 4),
                                                    " (main) and adapting normally from here."))

                              }

                        }

                        ## ---- Update the selected trajectory criterion through one block interface ------------------------------------------------
                        if (ii >= gap && ii < n_adapt && !isTRUE(manual_tau) &&
                            !bulk_local_tuner_freeze_active) {
                            tau_adaptation_iteration <- ii - gap + 1
                            tau_adaptation_iteration_vec[ii] <- tau_adaptation_iteration
                            ##
                            ## ---- position in the tau learning-rate schedule (LR -> LR^2): the tau adaptation iteration and window, or, after the
                            ##      restart at the metric freeze (NicoStan_tau_learning_rate_restart_at_metric_end), the iteration counted from
                            ##      tau_learning_rate_restart_iter and the window from there to the last tau update (iteration n_adapt - 1):
                            ##
                            tau_learning_rate_schedule_iteration <- tau_adaptation_iteration
                            tau_learning_rate_schedule_length <- n_tau_adaptation_iterations
                            if (tau_learning_rate_restart_active && (ii >= tau_learning_rate_restart_iter)) {
                                tau_learning_rate_schedule_iteration <- ii - tau_learning_rate_restart_iter + 1
                                tau_learning_rate_schedule_length <- n_adapt - tau_learning_rate_restart_iter
                                if (ii == tau_learning_rate_restart_iter) {
                                    # message(colourise(paste0("tau learning rate restarted at iteration ", ii, " (the first after metric_adaptation_end_iter = ",
                                    message(colourise(paste0("tau learning rate restarted at iteration ", ii, " (the first tau update with the final metric; metric_adaptation_end_iter = ",
                                                             metric_adaptation_end_iter, "): full LR, decaying linearly to LR^2 at iteration ", n_adapt - 1,
                                                             "; tau and the ADAM moments are kept."), "cyan"))
                                }
                            }
                            ##
                            ## ---- "probe_then_average": the constant learning rate LR (schedule position 1 of 1), and whether this tau update belongs
                            ##      to the probe (decided before the update, so the update that ends the probe is not also an ADAM step):
                            tau_probe_probing_now <-  tau_probe_then_average && isTRUE(tau_probe_state$probing)
                            if (tau_probe_then_average) {
                                tau_learning_rate_schedule_iteration <- 1
                                tau_learning_rate_schedule_length <- 1
                            }
                            ## ---- "fixed_length_probe_then_decay_and_average": after the probe, LR at the first ADAM update (end_iteration + 1) -> LR^2 at iteration n_adapt - 1:
                            if (tau_probe_fixed_length_decay && !tau_probe_probing_now && is.finite(tau_probe_state$end_iteration)) {
                                tau_learning_rate_schedule_iteration <- ii - tau_probe_state$end_iteration
                                tau_learning_rate_schedule_length <- max(1, (n_adapt - 1) - tau_probe_state$end_iteration)
                            }
                            ##
                            ## ---- CHESSR_time / SNAPER_time: tau_offset_from_sampling_overhead (at the step size the scored trajectories ran
                            ##      with) and burnin_to_sampling_leapfrog_time_ratio (at the current time_per_leapfrog_step_burnin) for this update:
                            ##
                            if (time_criterion_active) {
                                time_per_leapfrog_step_burnin_estimate <-  fn_time_per_leapfrog_step_burnin_fit_estimate(
                                    time_per_leapfrog_step_burnin_fit_sums       = time_per_leapfrog_step_burnin_fit_sums,
                                    time_per_leapfrog_step_burnin_min_iterations = time_criterion_sampling_quantities$time_per_leapfrog_step_burnin_min_iterations)
                                time_per_leapfrog_step_burnin_used <-  if (is.finite(time_criterion_sampling_quantities$time_per_leapfrog_step_burnin_user_supplied))
                                    time_criterion_sampling_quantities$time_per_leapfrog_step_burnin_user_supplied else
                                    time_per_leapfrog_step_burnin_estimate$time_per_leapfrog_step_burnin
                                ##
                                ## ---- MALT's exponent: lag_one_autocorrelation_rho for this update (the setting, or "adaptive": from the moments of the
                                ##      earlier updates):
                                ##
                                lag_one_autocorrelation_rho_for_update <-  fn_lag_one_autocorrelation_rho_for_tau_update(
                                    lag_one_autocorrelation_rho_setting = lag_one_autocorrelation_rho_setting,
                                    lag_one_autocorrelation_rho_moments = lag_one_autocorrelation_rho_moments)
                                lag_one_autocorrelation_rho_at_update <-  lag_one_autocorrelation_rho_for_update$value
                                lag_one_autocorrelation_rho_source_at_update <-  lag_one_autocorrelation_rho_for_update$source
                                lag_one_autocorrelation_rho_used_vec[ii] <-  lag_one_autocorrelation_rho_at_update
                                ##
                                time_criterion_quantities_at_update <-  fn_time_criterion_quantities_at_tau_update(
                                    time_criterion_sampling_quantities = time_criterion_sampling_quantities,
                                    time_per_leapfrog_step_burnin      = time_per_leapfrog_step_burnin_used,
                                    eps_used_for_iteration             = eps_main_used_for_iteration,
                                    n_iter_burnin                      = n_iter_burnin_at_adapted_tau)
                                tau_offset_from_sampling_overhead_at_update      <-  time_criterion_quantities_at_update$tau_offset_from_sampling_overhead
                                burnin_to_sampling_leapfrog_time_ratio_at_update <-  time_criterion_quantities_at_update$burnin_to_sampling_leapfrog_time_ratio
                                ##
                                ## ---- with no sampling overhead the penalty is 1 + burnin_to_sampling_leapfrog_time_ratio at every tau; at 2 or more it
                                ##      exceeds the ChEES / SNAPER elasticity at every tau on a Gaussian target (2 tau cot(tau) < 2 on the unit Gaussian),
                                ##      so the criterion has no interior optimum and shortens tau towards one leapfrog step:
                                ##
                                ## ---- (with MALT's exponent the penalty with no sampling overhead is ((1 + lag_one_autocorrelation_rho) / 2) *
                                ##      (1 + burnin_to_sampling_leapfrog_time_ratio), and the same holds at 2 or more. At lag_one_autocorrelation_rho = 1 the
                                ##      test is burnin_to_sampling_leapfrog_time_ratio >= 1 and the warning is the one of the criterion without
                                ##      lag_one_autocorrelation_rho)
                                ##
                                time_to_target_ESS_tau_penalty_without_sampling_overhead <-  ((1 + lag_one_autocorrelation_rho_at_update) / 2) *
                                                                                            (1 + burnin_to_sampling_leapfrog_time_ratio_at_update)
                                ## if (is.na(iteration_of_no_interior_optimum_warning) &&
                                ##     isTRUE(time_criterion_sampling_quantities$sampling_overhead_in_leapfrog_steps == 0) &&
                                ##     burnin_to_sampling_leapfrog_time_ratio_at_update >= 1) {
                                penalty_has_no_interior_optimum_at_update <-  if (isTRUE(lag_one_autocorrelation_rho_at_update == 1))
                                                                                  burnin_to_sampling_leapfrog_time_ratio_at_update >= 1 else
                                                                                  time_to_target_ESS_tau_penalty_without_sampling_overhead >= 2
                                if (is.na(iteration_of_no_interior_optimum_warning) &&
                                    isTRUE(time_criterion_sampling_quantities$sampling_overhead_in_leapfrog_steps == 0) &&
                                    penalty_has_no_interior_optimum_at_update) {
                                    iteration_of_no_interior_optimum_warning <-  ii
                                    if (isTRUE(lag_one_autocorrelation_rho_at_update == 1)) {
                                        warning(paste0("burnin_algorithm = '", burnin_algorithm, "' at iteration ", ii, ": sampling_overhead_in_leapfrog_steps = 0 (",
                                                       time_criterion_sampling_quantities$sampling_overhead_in_leapfrog_steps_source, ") and ",
                                                       "burnin_to_sampling_leapfrog_time_ratio = ", signif(burnin_to_sampling_leapfrog_time_ratio_at_update, 4),
                                                       " >= 1, so time_to_target_ESS_tau_penalty = ", signif(1 + burnin_to_sampling_leapfrog_time_ratio_at_update, 4),
                                                       " at every tau. On a Gaussian target the ChEES / SNAPER elasticity is below 2 at every tau, so the criterion ",
                                                       "has no interior optimum and shortens tau towards one leapfrog step. Supply time_per_iter_overhead_sampling and ",
                                                       "time_per_iter_summaries_sampling (or a previous run), or let the sampling timing probe time the summaries."))
                                    } else {
                                        warning(paste0("burnin_algorithm = '", burnin_algorithm, "' at iteration ", ii, ": sampling_overhead_in_leapfrog_steps = 0 (",
                                                       time_criterion_sampling_quantities$sampling_overhead_in_leapfrog_steps_source, "), ",
                                                       "burnin_to_sampling_leapfrog_time_ratio = ", signif(burnin_to_sampling_leapfrog_time_ratio_at_update, 4),
                                                       " and lag_one_autocorrelation_rho = ", signif(lag_one_autocorrelation_rho_at_update, 4),
                                                       ", so time_to_target_ESS_tau_penalty = ((1 + lag_one_autocorrelation_rho) / 2) * (1 + burnin_to_sampling_leapfrog_time_ratio) = ",
                                                       signif(time_to_target_ESS_tau_penalty_without_sampling_overhead, 4),
                                                       " >= 2 at every tau. On a Gaussian target the ChEES / SNAPER elasticity is below 2 at every tau, so the criterion ",
                                                       "has no interior optimum and shortens tau towards one leapfrog step. Supply time_per_iter_overhead_sampling and ",
                                                       "time_per_iter_summaries_sampling (or a previous run), set sampling_overhead_in_leapfrog_steps (a number or \"auto\"), ",
                                                       "or let the sampling timing probe time the summaries."))
                                    }
                                }
                                ##
                                time_per_leapfrog_step_burnin_vec[ii]              <-  time_per_leapfrog_step_burnin_used
                                tau_offset_from_sampling_overhead_vec[ii]          <-  tau_offset_from_sampling_overhead_at_update
                                burnin_to_sampling_leapfrog_time_ratio_vec[ii]     <-  burnin_to_sampling_leapfrog_time_ratio_at_update
                                time_to_target_ESS_tau_penalty_at_tau_main_vec[ii] <-  fn_time_to_target_ESS_tau_penalty(
                                    tau_values                             = EHMC_args_as_Rcpp_List$tau_main,
                                    tau_offset_from_sampling_overhead      = tau_offset_from_sampling_overhead_at_update +
                                                                             EHMC_args_as_Rcpp_List$eps_main * rate_criterion_cost_offset_steps,
                                    ## burnin_to_sampling_leapfrog_time_ratio = burnin_to_sampling_leapfrog_time_ratio_at_update)
                                    burnin_to_sampling_leapfrog_time_ratio = burnin_to_sampling_leapfrog_time_ratio_at_update,
                                    lag_one_autocorrelation_rho            = lag_one_autocorrelation_rho_at_update)
                                time_criterion_snapshot <-  list( iteration                                 = ii,
                                                                  eps_main_used_for_iteration               = eps_main_used_for_iteration,
                                                                  tau_main_before_update                    = EHMC_args_as_Rcpp_List$tau_main,
                                                                  time_per_leapfrog_step_burnin             = time_per_leapfrog_step_burnin_used,
                                                                  time_per_leapfrog_step_burnin_source      = if (is.finite(time_criterion_sampling_quantities$time_per_leapfrog_step_burnin_user_supplied))
                                                                                                                  "user_supplied" else "online_burnin_fit",
                                                                  time_per_leapfrog_step_burnin_fit         = time_per_leapfrog_step_burnin_estimate,
                                                                  sampling_overhead_in_leapfrog_steps       = time_criterion_sampling_quantities$sampling_overhead_in_leapfrog_steps,
                                                                  tau_offset_from_sampling_overhead         = tau_offset_from_sampling_overhead_at_update,
                                                                  burnin_to_sampling_leapfrog_time_ratio    = burnin_to_sampling_leapfrog_time_ratio_at_update,
                                                                  lag_one_autocorrelation_rho               = lag_one_autocorrelation_rho_at_update,
                                                                  lag_one_autocorrelation_rho_source        = lag_one_autocorrelation_rho_source_at_update,
                                                                  time_to_target_ESS_tau_penalty_at_tau_main = time_to_target_ESS_tau_penalty_at_tau_main_vec[ii],
                                                                  ESS_elasticity_wrt_log_tau                = NA_real_)
                                if (ii == gap) {
                                    time_criterion_at_handover <-  time_criterion_snapshot
                                    message(colourise(paste0("time-to-target-ESS criterion at the handover (iteration ", ii, "): ",
                                                             "time_per_leapfrog_step_burnin = ", signif(time_per_leapfrog_step_burnin_used, 4), " s (",
                                                             time_criterion_snapshot$time_per_leapfrog_step_burnin_source, ", ",
                                                             time_per_leapfrog_step_burnin_estimate$status, ")",
                                                             " | tau_offset_from_sampling_overhead = ", signif(tau_offset_from_sampling_overhead_at_update, 4),
                                                             ## " | burnin_to_sampling_leapfrog_time_ratio = ", signif(burnin_to_sampling_leapfrog_time_ratio_at_update, 4)),
                                                             " | burnin_to_sampling_leapfrog_time_ratio = ", signif(burnin_to_sampling_leapfrog_time_ratio_at_update, 4),
                                                             if (time_criterion_messages_show_rho) paste0(
                                                                 " | lag_one_autocorrelation_rho = ", signif(lag_one_autocorrelation_rho_at_update, 4), " (", lag_one_autocorrelation_rho_source_at_update, ")") else ""),
                                                      "cyan"))
                                }
                                time_criterion_at_last_update <-  time_criterion_snapshot
                            }
                            adapted_blocks <- tau_adaptation_block_effective   ## "main" (default) or "joint" (EXPERIMENTAL)
                            for (adapted_block in adapted_blocks) {
                                is_main <- adapted_block == "main"
                                is_joint <- adapted_block == "joint"   ## main + nuisance concatenated (main rows first); updates tau_main
                                block_name <- if (is_main || is_joint) "main" else "us"
                                if (use_resident_joint_block) {
                                ## the joint endpoints stay in the worker; their main rows are read below, their nuisance rows in the worker:
                                theta_initial <- theta_proposed <- theta_accepted <- NULL
                                velocity_initial <- velocity_proposed <- velocity_accepted <- NULL
                                } else {
                                theta_initial <- if (is_main) theta_main_0_burnin_tau_adapt_all_chains_input_from_R else if (is_joint) rbind(theta_main_0_burnin_tau_adapt_all_chains_input_from_R, theta_us_0_burnin_tau_adapt_all_chains_input_from_R) else theta_us_0_burnin_tau_adapt_all_chains_input_from_R
                                theta_proposed <- if (is_main) theta_main_prop_burnin_tau_adapt_all_chains_input_from_R else if (is_joint) rbind(theta_main_prop_burnin_tau_adapt_all_chains_input_from_R, theta_us_prop_burnin_tau_adapt_all_chains_input_from_R) else theta_us_prop_burnin_tau_adapt_all_chains_input_from_R
                                theta_accepted <- if (is_main) theta_main_vectors_all_chains_input_from_R else if (is_joint) rbind(theta_main_vectors_all_chains_input_from_R, theta_us_vectors_all_chains_input_from_R) else theta_us_vectors_all_chains_input_from_R
                                velocity_initial <- if (is_main) velocity_main_0_burnin_tau_adapt_all_chains_input_from_R else if (is_joint) rbind(velocity_main_0_burnin_tau_adapt_all_chains_input_from_R, velocity_us_0_burnin_tau_adapt_all_chains_input_from_R) else velocity_us_0_burnin_tau_adapt_all_chains_input_from_R
                                velocity_proposed <- if (is_main) velocity_main_prop_burnin_tau_adapt_all_chains_input_from_R else if (is_joint) rbind(velocity_main_prop_burnin_tau_adapt_all_chains_input_from_R, velocity_us_prop_burnin_tau_adapt_all_chains_input_from_R) else velocity_us_prop_burnin_tau_adapt_all_chains_input_from_R
                                velocity_accepted <- if (is_main) velocity_main_vectors_all_chains_input_from_R else if (is_joint) rbind(velocity_main_vectors_all_chains_input_from_R, velocity_us_vectors_all_chains_input_from_R) else velocity_us_vectors_all_chains_input_from_R
                                }  ## end of: if (use_resident_joint_block)
                                probabilities <- if (is_main || is_joint) p_jump_per_chain else p_jump_us_per_chain
                                divergences <- if (is_main) div_main else if (is_joint) pmax(div_main, div_us) else div_us
                                tau_values <- if (is_main || is_joint) tau_main_ii_vec else tau_us_ii_vec
                                tau_name <- paste0("tau_", block_name)
                                ## the trajectory cost offset of this update's block (eps_<block> x c; see its definition):
                                tau_cost_offset_at_update <-  EHMC_args_as_Rcpp_List[[paste0("eps_", block_name)]] * rate_criterion_cost_offset_steps
                                tau_cost_offset_vec[ii] <-  tau_cost_offset_at_update
                                adam_mean_name <- paste0("tau_m_adam_", block_name)
                                adam_variance_name <- paste0("tau_v_adam_", block_name)
                                ## this update would be number (performed so far + 1) on these moments:
                                tau_adam_bias_correction_step <- tau_adam_updates_performed_by_block[[block_name]] + 1
                                t0 <- proc.time()[3]
                                        ## the nuisance kinetic-energy sums of every chain, from the velocities inside the worker:
                                            ## the main rows of the joint velocities are the main velocities; the nuisance change comes from the worker:
                                               ## end of: if (use_resident_joint_block)
                                        ## beta1_adam = beta1_adam, beta2_adam = beta2_adam, eps_adam = eps_adam,
                                        ## beta1_adam = tau_adam_beta1, beta2_adam = beta2_adam, eps_adam = eps_adam,
                                        ## aggregation = "weighted_mean")
                                {
                                    if (use_resident_joint_block) {
                                    ##
                                    ## ---- the joint criterion: the main rows of the endpoints in metric coordinates here (the same R code on the main
                                    ##      rows: fn_apply_trajectory_metric() of the main block of trajectory_metric$joint), their nuisance rows and the
                                    ##      per-chain sums / projections over all the joint rows in the worker:
                                    use_proposals_joint <-  isTRUE(tau_weight_by_p_jump)
                                    joint_metric_factor <-  trajectory_metric[[adapted_block]]
                                    initial_main_metric <-  fn_apply_trajectory_metric(joint_metric_factor$main,
                                                                                       theta_main_0_burnin_tau_adapt_all_chains_input_from_R - c(EHMC_burnin_as_Rcpp_List$snaper_m_vec_main))
                                    proposed_main_metric <-  fn_apply_trajectory_metric(joint_metric_factor$main,
                                                                                        if (use_proposals_joint) theta_main_prop_burnin_tau_adapt_all_chains_input_from_R - c(snaper_m_prop_vec_all[index_main]) else
                                                                                                                 theta_main_vectors_all_chains_input_from_R - c(EHMC_burnin_as_Rcpp_List$snaper_m_vec_main))
                                    velocity_main_metric <-  fn_apply_trajectory_metric(joint_metric_factor$main,
                                                                                        if (use_proposals_joint) velocity_main_prop_burnin_tau_adapt_all_chains_input_from_R else
                                                                                                                 velocity_main_vectors_all_chains_input_from_R)
                                    ##
                                    ## ---- ESJD / ESJD_CHESSR: the worker's reductions carry no z_0 . z_t cross products (the squared jump
                                    ##      ||z_t - z_0||^2 needs them), so these criteria always take the "status 2" route below: the joint endpoints
                                    ##      come from the worker and fn_metric_position_criterion() computes the criterion in R (no C++ change):
                                    ##
                                    ## joint_reductions <-  resident_api$fn_persistent_burnin_joint_position_reductions_resident( worker_ptr,
                                    ## joint_reductions <-  if (burnin_algorithm %in% c("ESJD", "ESJD_CHESSR", "ESJD_SNAPER")) list(status = 2) else
                                    ## ("LQ_ESSR" needs every row of the endpoints, not per-chain sums: the same route)
                                    joint_reductions <-  if (burnin_algorithm %in% c("ESJD", "ESJD_CHESSR", "ESJD_SNAPER",
                                                                                     "LQ_ESSR")) list(status = 2) else
                                                         if (identical(tau_gradient_estimator, "two_ended"))
                                                             resident_api[[resident_joint_two_ended_api_name]]( worker_ptr,
                                                                                                                 projection_R = burnin_algorithm %in% c("SNAPER", "SNAPER_time", "ESJD_SNAPER"),
                                                                                                                 use_proposals_R = use_proposals_joint,
                                                                                                                 initial_main = initial_main_metric,
                                                                                                                 proposed_main = proposed_main_metric,
                                                                                                                 velocity_main = velocity_main_metric,
                                                                                                                 velocity_initial_main = fn_apply_trajectory_metric(joint_metric_factor$main,
                                                                                                                                                                     velocity_main_0_burnin_tau_adapt_all_chains_input_from_R),
                                                                                                                 factor_us = joint_metric_factor$us) else
                                                         resident_api$fn_persistent_burnin_joint_position_reductions_resident( worker_ptr,
                                                                                                                              ## projection_R = identical(burnin_algorithm, "SNAPER"),
                                                                                                                              projection_R = burnin_algorithm %in% c("SNAPER", "SNAPER_time", "ESJD_SNAPER"),
                                                                                                                              use_proposals_R = use_proposals_joint,
                                                                                                                              initial_main = initial_main_metric,
                                                                                                                              proposed_main = proposed_main_metric,
                                                                                                                              velocity_main = velocity_main_metric,
                                                                                                                              factor_us = joint_metric_factor$us)
                                    if (joint_reductions$status == 3) stop("Invalid SNAPER direction.")
                                    joint_position_criterion <-  NULL
                                    joint_mean_initial <-  joint_mean_proposed <-  joint_direction <-  NULL
                                    if (joint_reductions$status == 2) {
                                        ## R's crossprod() would not use the BLAS for these numbers: the joint endpoints come from the worker and the
                                        ## criterion is computed by the R code itself for this iteration:
                                        theta_initial <-  rbind(theta_main_0_burnin_tau_adapt_all_chains_input_from_R,
                                                                resident_api$fn_persistent_burnin_get_state(worker_ptr, "theta_us_0_burnin_tau_adapt_all_chains_input_from_R"))
                                        theta_proposed <-  rbind(theta_main_prop_burnin_tau_adapt_all_chains_input_from_R,
                                                                 resident_api$fn_persistent_burnin_get_state(worker_ptr, "theta_us_prop_burnin_tau_adapt_all_chains_input_from_R"))
                                        theta_accepted <-  rbind(theta_main_vectors_all_chains_input_from_R,
                                                                 resident_api$fn_persistent_burnin_get_state(worker_ptr, "theta_us_vectors_all_chains_output_to_R"))
                                        velocity_proposed <-  rbind(velocity_main_prop_burnin_tau_adapt_all_chains_input_from_R,
                                                                    resident_api$fn_persistent_burnin_get_state(worker_ptr, "velocity_us_prop_burnin_tau_adapt_all_chains_input_from_R"))
                                        velocity_accepted <-  rbind(velocity_main_vectors_all_chains_input_from_R,
                                                                    resident_api$fn_persistent_burnin_get_state(worker_ptr, "velocity_us_vectors_all_chains_output_to_R"))
                                        velocity_initial <-  rbind(velocity_main_0_burnin_tau_adapt_all_chains_input_from_R,
                                                                   resident_api$fn_persistent_burnin_get_state(worker_ptr, "velocity_us_0_burnin_tau_adapt_all_chains_input_from_R"))
                                        joint_mean_initial <-  c(EHMC_burnin_as_Rcpp_List$snaper_m_vec_main,
                                                                 resident_api$fn_persistent_burnin_get_resident_statistic(worker_ptr, "snaper_m_vec_us"))
                                        joint_mean_proposed <-  c(snaper_m_prop_vec_all[index_main],
                                                                  resident_api$fn_persistent_burnin_get_resident_statistic(worker_ptr, "snaper_m_prop_vec_us"))
                                        joint_direction <-  resident_api$fn_persistent_burnin_joint_direction_get(worker_ptr, FALSE)
                                        if (trajectory_criterion_advanced_active) {
                                            joint_position_criterion <-  fn_metric_position_criterion_advanced(
                                                algorithm = burnin_algorithm,
                                                theta_initial = theta_initial,
                                                theta_proposed = if (use_proposals_joint) theta_proposed else theta_accepted,
                                                velocity_initial = velocity_initial,
                                                velocity_proposed = if (use_proposals_joint) velocity_proposed else velocity_accepted,
                                                mean_initial = joint_mean_initial,
                                                mean_proposed = if (use_proposals_joint) joint_mean_proposed else joint_mean_initial,
                                                metric_factor = joint_metric_factor,
                                                tau_values = tau_values,
                                                direction = joint_direction,
                                                tau_gradient_estimator = tau_gradient_estimator,
                                                tau_cost_exponent = tau_cost_exponent,
                                                esjd_jump_power = esjd_jump_power,
                                                tau_jitter_burnin = tau_jitter_burnin,
                                                randomize_tau_burnin = randomize_tau_burnin,
                                                tau_offset_from_sampling_overhead = tau_offset_from_sampling_overhead_at_update,
                                                burnin_to_sampling_leapfrog_time_ratio = burnin_to_sampling_leapfrog_time_ratio_at_update,
                                                lag_one_autocorrelation_rho = lag_one_autocorrelation_rho_at_update,
                                                tau_cost_offset = tau_cost_offset_at_update)
                                        }
                                    } else {
                                        joint_position_criterion <-  if (trajectory_criterion_advanced_active)
                                            fn_metric_position_criterion_from_reductions_advanced(
                                                algorithm = burnin_algorithm,
                                                reductions = joint_reductions,
                                                tau_values = tau_values,
                                                tau_gradient_estimator = tau_gradient_estimator,
                                                tau_cost_exponent = tau_cost_exponent,
                                                esjd_jump_power = esjd_jump_power,
                                                tau_jitter_burnin = tau_jitter_burnin,
                                                randomize_tau_burnin = randomize_tau_burnin,
                                                tau_offset_from_sampling_overhead = tau_offset_from_sampling_overhead_at_update,
                                                burnin_to_sampling_leapfrog_time_ratio = burnin_to_sampling_leapfrog_time_ratio_at_update,
                                                lag_one_autocorrelation_rho = lag_one_autocorrelation_rho_at_update,
                                                tau_cost_offset = tau_cost_offset_at_update) else
                                            fn_metric_position_criterion_from_reductions(algorithm = burnin_algorithm,
                                                                                          reductions = joint_reductions,
                                                                                                                  ## tau_values = tau_values)
                                                                                          tau_values = tau_values,
                                                                                          tau_offset_from_sampling_overhead = tau_offset_from_sampling_overhead_at_update,
                                                                                                                  ## burnin_to_sampling_leapfrog_time_ratio = burnin_to_sampling_leapfrog_time_ratio_at_update)
                                                                                          burnin_to_sampling_leapfrog_time_ratio = burnin_to_sampling_leapfrog_time_ratio_at_update,
                                                                                          lag_one_autocorrelation_rho = lag_one_autocorrelation_rho_at_update,
                                                                                          tau_cost_offset = tau_cost_offset_at_update)
                                        ## (with the criterion given, only the number of columns (chains) of theta_initial is read)
                                        theta_initial <-  theta_main_0_burnin_tau_adapt_all_chains_input_from_R
                                    }
                                    update <- fn_metric_tau_block_update(
                                        algorithm = burnin_algorithm, theta_initial = theta_initial, theta_proposed = theta_proposed,
                                        theta_accepted = theta_accepted, velocity_proposed = velocity_proposed, velocity_accepted = velocity_accepted,
                                        mean_initial = joint_mean_initial, mean_proposed = joint_mean_proposed,
                                        metric_factor = joint_metric_factor, direction = joint_direction,
                                        tau_values = tau_values, probabilities = probabilities, divergences = divergences,
                                        weight_by_probability = tau_weight_by_p_jump, tau = EHMC_args_as_Rcpp_List[[tau_name]],
                                        learning_rate = EHMC_burnin_as_Rcpp_List[[paste0("LR_", block_name)]],
                                        iteration = tau_adaptation_iteration, adaptation_iterations = n_tau_adaptation_iterations,
                                        adam_mean = EHMC_burnin_as_Rcpp_List[[adam_mean_name]],
                                        adam_variance = EHMC_burnin_as_Rcpp_List[[adam_variance_name]],
                                        ## beta1 = beta1_adam, beta2 = beta2_adam, adam_epsilon = eps_adam,
                                        ## beta1 = tau_adam_beta1, beta2 = beta2_adam, adam_epsilon = eps_adam,
                                        beta1 = tau_adam_beta1, beta2 = tau_adam_beta2, adam_epsilon = tau_adam_epsilon,
                                        criterion_ema = ChEES_criterion_ema,
                                        bias_correction_step = tau_adam_bias_correction_step,
                                        ## position_criterion = joint_position_criterion)
                                        position_criterion = joint_position_criterion,
                                        tau_offset_from_sampling_overhead = tau_offset_from_sampling_overhead_at_update,
                                        ## burnin_to_sampling_leapfrog_time_ratio = burnin_to_sampling_leapfrog_time_ratio_at_update)
                                        burnin_to_sampling_leapfrog_time_ratio = burnin_to_sampling_leapfrog_time_ratio_at_update,
                                        ## lag_one_autocorrelation_rho = lag_one_autocorrelation_rho_at_update)
                                        lag_one_autocorrelation_rho = lag_one_autocorrelation_rho_at_update,
                                        tau_cost_offset = tau_cost_offset_at_update,
                                        learning_rate_schedule_iteration = tau_learning_rate_schedule_iteration,
                                        ## learning_rate_schedule_length = tau_learning_rate_schedule_length)
                                        learning_rate_schedule_length = tau_learning_rate_schedule_length,
                                        ## component_criterion_ema = if (burnin_algorithm == "ESJD_SNAPER") ESJD_SNAPER_component_criterion_ema else
                                        ##                           ESJD_CHESSR_component_criterion_ema)
                                        component_criterion_ema = if (burnin_algorithm == "ESJD_SNAPER") ESJD_SNAPER_component_criterion_ema else
                                                                  if (burnin_algorithm == "LQ_ESSR") LQ_ESSR_component_criterion_ema else
                                                                  ESJD_CHESSR_component_criterion_ema,
                                        ## LQ_ESSR: the interest rows index the main block, which leads the main and the joint block:
                                        interest_rows = if (is_main || is_joint) interest_rows else NULL)
                                    } else {
                                    initial_mean <- if (is_main) EHMC_burnin_as_Rcpp_List$snaper_m_vec_main else
                                                    if (is_joint) c(EHMC_burnin_as_Rcpp_List$snaper_m_vec_main, EHMC_burnin_as_Rcpp_List$snaper_m_vec_us) else
                                                    EHMC_burnin_as_Rcpp_List$snaper_m_vec_us
                                    proposal_mean <- snaper_m_prop_vec_all[if (is_main) index_main else if (is_joint) c(index_main, index_nuisance) else index_nuisance]
                                    position_criterion <-  NULL
                                    if (trajectory_criterion_advanced_active) {
                                        position_criterion <-  fn_metric_position_criterion_advanced(
                                            algorithm = burnin_algorithm,
                                            theta_initial = theta_initial,
                                            theta_proposed = if (isTRUE(tau_weight_by_p_jump)) theta_proposed else theta_accepted,
                                            velocity_initial = velocity_initial,
                                            velocity_proposed = if (isTRUE(tau_weight_by_p_jump)) velocity_proposed else velocity_accepted,
                                            mean_initial = c(initial_mean),
                                            mean_proposed = if (isTRUE(tau_weight_by_p_jump)) c(proposal_mean) else c(initial_mean),
                                            metric_factor = trajectory_metric[[adapted_block]],
                                            tau_values = tau_values,
                                            direction = trajectory_direction[[adapted_block]],
                                            tau_gradient_estimator = tau_gradient_estimator,
                                            tau_cost_exponent = tau_cost_exponent,
                                            esjd_jump_power = esjd_jump_power,
                                            tau_jitter_burnin = tau_jitter_burnin,
                                            randomize_tau_burnin = randomize_tau_burnin,
                                            tau_offset_from_sampling_overhead = tau_offset_from_sampling_overhead_at_update,
                                            burnin_to_sampling_leapfrog_time_ratio = burnin_to_sampling_leapfrog_time_ratio_at_update,
                                            lag_one_autocorrelation_rho = lag_one_autocorrelation_rho_at_update,
                                            tau_cost_offset = tau_cost_offset_at_update)
                                    }
                                    update <- fn_metric_tau_block_update(
                                        algorithm = burnin_algorithm, theta_initial = theta_initial, theta_proposed = theta_proposed,
                                        theta_accepted = theta_accepted, velocity_proposed = velocity_proposed, velocity_accepted = velocity_accepted,
                                        mean_initial = c(initial_mean), mean_proposed = c(proposal_mean),
                                        metric_factor = trajectory_metric[[adapted_block]], direction = trajectory_direction[[adapted_block]],
                                        tau_values = tau_values, probabilities = probabilities, divergences = divergences,
                                        weight_by_probability = tau_weight_by_p_jump, tau = EHMC_args_as_Rcpp_List[[tau_name]],
                                        learning_rate = EHMC_burnin_as_Rcpp_List[[paste0("LR_", block_name)]],
                                        iteration = tau_adaptation_iteration, adaptation_iterations = n_tau_adaptation_iterations,
                                        adam_mean = EHMC_burnin_as_Rcpp_List[[adam_mean_name]],
                                        adam_variance = EHMC_burnin_as_Rcpp_List[[adam_variance_name]],
                                        ## beta1 = beta1_adam, beta2 = beta2_adam, adam_epsilon = eps_adam,
                                        ## beta1 = tau_adam_beta1, beta2 = beta2_adam, adam_epsilon = eps_adam,
                                        beta1 = tau_adam_beta1, beta2 = tau_adam_beta2, adam_epsilon = tau_adam_epsilon,
                                        criterion_ema = ChEES_criterion_ema,
                                        ## bias_correction_step = tau_adam_bias_correction_step)
                                        bias_correction_step = tau_adam_bias_correction_step,
                                        ## position_criterion = position_criterion)
                                        position_criterion = position_criterion,
                                        tau_offset_from_sampling_overhead = tau_offset_from_sampling_overhead_at_update,
                                        ## burnin_to_sampling_leapfrog_time_ratio = burnin_to_sampling_leapfrog_time_ratio_at_update)
                                        burnin_to_sampling_leapfrog_time_ratio = burnin_to_sampling_leapfrog_time_ratio_at_update,
                                        ## lag_one_autocorrelation_rho = lag_one_autocorrelation_rho_at_update)
                                        lag_one_autocorrelation_rho = lag_one_autocorrelation_rho_at_update,
                                        tau_cost_offset = tau_cost_offset_at_update,
                                        learning_rate_schedule_iteration = tau_learning_rate_schedule_iteration,
                                        ## learning_rate_schedule_length = tau_learning_rate_schedule_length)
                                        learning_rate_schedule_length = tau_learning_rate_schedule_length,
                                        ## component_criterion_ema = if (burnin_algorithm == "ESJD_SNAPER") ESJD_SNAPER_component_criterion_ema else
                                        ##                           ESJD_CHESSR_component_criterion_ema)
                                        component_criterion_ema = if (burnin_algorithm == "ESJD_SNAPER") ESJD_SNAPER_component_criterion_ema else
                                                                  if (burnin_algorithm == "LQ_ESSR") LQ_ESSR_component_criterion_ema else
                                                                  ESJD_CHESSR_component_criterion_ema,
                                        ## LQ_ESSR: the interest rows index the main block, which leads the main and the joint block:
                                        interest_rows = if (is_main || is_joint) interest_rows else NULL)
                                    }  ## end of: if (use_resident_joint_block)
                                    updated <- update$updated
                                    tau_adam_update_performed <- isTRUE(update$adam_update_performed)
                                    if (is_main || is_joint) {
                                        ChEES_criterion_ema <- update$criterion_ema
                                        ChEES_criterion_ema_vec[ii] <- update$criterion_ema
                                        ChEES_per_tau_gradient_vec[ii] <- update$gradient
                                        ## "ESJD_CHESSR": the two component moving averages, carried to the next update:
                                        if (!is.null(update$component_criterion_ema)) {
                                            if (burnin_algorithm == "ESJD_SNAPER") {
                                                ## "ESJD_SNAPER": carry its ESJD/SNAPER component moving averages to the next update.
                                                ESJD_SNAPER_component_criterion_ema <-  update$component_criterion_ema
                                            } else if (burnin_algorithm == "LQ_ESSR") {
                                                ## "LQ_ESSR": carry its per-coordinate moving averages to the next update.
                                                LQ_ESSR_component_criterion_ema <-  update$component_criterion_ema
                                            } else {
                                                ESJD_CHESSR_component_criterion_ema <-  update$component_criterion_ema
                                            }
                                        }
                                    }
                                    ##
                                    ## ---- CHESSR_time / SNAPER_time: the chain-mean estimate of ESS_elasticity_wrt_log_tau at this update (the snapshots
                                    ##      were taken before the update, so it is added to them here):
                                    ##
                                    if (time_criterion_active && (is_main || is_joint) && !is.null(update$ESS_elasticity_wrt_log_tau)) {
                                        ESS_elasticity_wrt_log_tau_vec[ii] <-  update$ESS_elasticity_wrt_log_tau
                                        ChEES_or_SNAPER_statistic_per_tau_chain_mean_vec[ii] <-  update$ChEES_or_SNAPER_statistic_per_tau_chain_mean
                                        ChEES_or_SNAPER_statistic_log_tau_derivative_per_tau_chain_mean_vec[ii] <-  update$ChEES_or_SNAPER_statistic_log_tau_derivative_per_tau_chain_mean
                                        time_criterion_at_last_update$ESS_elasticity_wrt_log_tau <-  update$ESS_elasticity_wrt_log_tau
                                        if (ii == gap) time_criterion_at_handover$ESS_elasticity_wrt_log_tau <-  update$ESS_elasticity_wrt_log_tau
                                    }
                                    ##
                                    ## ---- lag_one_autocorrelation_rho = "adaptive": this update's criterion statistic at the trajectory start and at the ACCEPTED
                                    ##      end, in the metric coordinates, centre and direction of the criterion, enters MALT's running moments AFTER the tau
                                    ##      update, so the lag_one_autocorrelation_rho of the next update comes from the moments up to this one
                                    ##      (R_fn_time_criterion.R). The resident joint block takes the criterion statistic from the worker's reductions of the
                                    ##      accepted end (a second reductions call when the criterion's were of the proposed end), or from the joint states when
                                    ##      they were copied to R (status 2 above):
                                    ##
                                    if (time_criterion_active && lag_one_autocorrelation_rho_is_adaptive && (is_main || is_joint)) {
                                        lag_one_autocorrelation_rho_statistic <-  tryCatch({
                                            if (use_resident_joint_block && is.null(joint_mean_initial)) {
                                                joint_reductions_of_accepted_end <-  if (!use_proposals_joint) joint_reductions else
                                                    resident_api$fn_persistent_burnin_joint_position_reductions_resident( worker_ptr,
                                                                                                                          projection_R = burnin_algorithm %in% c("SNAPER", "SNAPER_time", "ESJD_SNAPER"),
                                                                                                                          use_proposals_R = FALSE,
                                                                                                                          initial_main = initial_main_metric,
                                                                                                                          proposed_main = fn_apply_trajectory_metric(joint_metric_factor$main,
                                                                                                                                                                     theta_main_vectors_all_chains_input_from_R - c(EHMC_burnin_as_Rcpp_List$snaper_m_vec_main)),
                                                                                                                          velocity_main = fn_apply_trajectory_metric(joint_metric_factor$main,
                                                                                                                                                                     velocity_main_vectors_all_chains_input_from_R),
                                                                                                                          factor_us = joint_metric_factor$us)
                                                fn_lag_one_autocorrelation_rho_statistic_from_reductions( algorithm                  = burnin_algorithm,
                                                                                                          reductions_of_accepted_end = joint_reductions_of_accepted_end)
                                            } else {
                                                fn_lag_one_autocorrelation_rho_statistic_per_chain( algorithm      = burnin_algorithm,
                                                                                                    theta_initial  = theta_initial,
                                                                                                    theta_accepted = theta_accepted,
                                                                                                    mean_initial   = if (use_resident_joint_block) joint_mean_initial else c(initial_mean),
                                                                                                    metric_factor  = if (use_resident_joint_block) joint_metric_factor else trajectory_metric[[adapted_block]],
                                                                                                    direction      = if (use_resident_joint_block) joint_direction else trajectory_direction[[adapted_block]])
                                            }
                                        }, error = function(error_object) {
                                            message(colourise(paste0("lag_one_autocorrelation_rho = \"adaptive\" at iteration ", ii, ": the criterion statistic is not available (",
                                                                     conditionMessage(error_object), "); the moments are not updated."), "red"))
                                            NULL
                                        })
                                        if (!is.null(lag_one_autocorrelation_rho_statistic)) {
                                            lag_one_autocorrelation_rho_moments <-  fn_lag_one_autocorrelation_rho_moments_update(
                                                lag_one_autocorrelation_rho_moments                 = lag_one_autocorrelation_rho_moments,
                                                criterion_statistic_at_trajectory_start             = lag_one_autocorrelation_rho_statistic$criterion_statistic_at_trajectory_start,
                                                criterion_statistic_at_accepted_end                 = lag_one_autocorrelation_rho_statistic$criterion_statistic_at_accepted_end,
                                                chain_is_valid                                      = is.finite(divergences) & divergences == 0,
                                                lag_one_autocorrelation_rho_moment_averaging_offset = lag_one_autocorrelation_rho_moment_averaging_offset)
                                            lag_one_autocorrelation_rho_n_valid_chains_vec[ii] <-  lag_one_autocorrelation_rho_moments$n_valid_chains_last_update
                                        }
                                        lag_one_autocorrelation_rho_estimate_after_update_vec[ii] <-  fn_lag_one_autocorrelation_rho_estimate(lag_one_autocorrelation_rho_moments)
                                    }
                                }
                                ##
                                ## ---- "probe_then_average", probe phase: the ADAM step computed above is discarded; its aggregated gradient goes into
                                ##      the block sum, tau is set by the rung, and the ADAM moments stay as they are (fresh):
                                if (tau_probe_probing_now && (is_main || is_joint)) {
                                    tau_probe_aggregated_gradient <-  update$gradient
                                    if (is.null(tau_probe_aggregated_gradient) || length(tau_probe_aggregated_gradient) != 1 ||
                                        !is.finite(tau_probe_aggregated_gradient)) tau_probe_aggregated_gradient <-  0
                                    ## "ESJD_CHESSR": the chain means of its two components go into the block sums of the probe (fn_tau_probe_step):
                                    tau_probe_component_chain_means <-  NULL
                                    if (burnin_algorithm == "ESJD_CHESSR") {
                                        tau_probe_component_chain_means <-  c(ESJD_gradient    = update$component_gradient_chain_mean[["ESJD"]],
                                                                              ESJD_criterion   = update$component_criterion_chain_mean[["ESJD"]],
                                                                              CHESSR_gradient  = update$component_gradient_chain_mean[["CHESSR"]],
                                                                              CHESSR_criterion = update$component_criterion_chain_mean[["CHESSR"]])
                                        tau_probe_component_chain_means[!is.finite(tau_probe_component_chain_means)] <-  0
                                    } else if (burnin_algorithm == "ESJD_SNAPER") {
                                        ## The ESJD_SNAPER probe pools the same ESJD component with the SNAPER rate, so its rung decision
                                        ## cannot accidentally use the CHESSR component from the existing ESJD_CHESSR path.
                                        tau_probe_component_chain_means <-  c(ESJD_gradient    = update$component_gradient_chain_mean[["ESJD"]],
                                                                              ESJD_criterion   = update$component_criterion_chain_mean[["ESJD"]],
                                                                              SNAPER_gradient  = update$component_gradient_chain_mean[["SNAPER"]],
                                                                              SNAPER_criterion = update$component_criterion_chain_mean[["SNAPER"]])
                                        tau_probe_component_chain_means[!is.finite(tau_probe_component_chain_means)] <-  0
                                    }
                                    tau_probe_rung_before_step <-  tau_probe_state$rung
                                    tau_probe_state <-  fn_tau_probe_step( state = tau_probe_state,
                                                                           aggregated_gradient = tau_probe_aggregated_gradient,
                                                                           ## iteration = ii)
                                                                           iteration = ii,
                                                                           component_chain_means = tau_probe_component_chain_means)
                                    ##
                                    ## ---- "ESJD_CHESSR": when the probe moves tau to a new rung, or ends (ADAM from the next tau update; with
                                    ##      "fixed_length_probe_then_decay_and_average" the jitter also resumes), the two component moving averages
                                    ##      restart, so that the levels dividing the next gradients are those of the new tau, not of the previous rung:
                                    if (burnin_algorithm == "ESJD_CHESSR" && (tau_probe_state$rung != tau_probe_rung_before_step || !tau_probe_state$probing)) {
                                        ESJD_CHESSR_component_criterion_ema <-  c(ESJD = NA_real_, CHESSR = NA_real_)
                                    }
                                    if (burnin_algorithm == "ESJD_SNAPER" && (tau_probe_state$rung != tau_probe_rung_before_step || !tau_probe_state$probing)) {
                                        ESJD_SNAPER_component_criterion_ema <-  c(ESJD = NA_real_, SNAPER = NA_real_)
                                    }
                                    ## "LQ_ESSR": its per-coordinate moving averages restart at a new rung for the same reason:
                                    if (burnin_algorithm == "LQ_ESSR" &&
                                        (tau_probe_state$rung != tau_probe_rung_before_step || !tau_probe_state$probing)) {
                                        LQ_ESSR_component_criterion_ema <-  NULL
                                    }
                                    updated <-  c(exp(tau_probe_state$log_tau_start + tau_probe_state$rung * log(2)),
                                                  EHMC_burnin_as_Rcpp_List[[adam_mean_name]],
                                                  EHMC_burnin_as_Rcpp_List[[adam_variance_name]])
                                    tau_adam_update_performed <-  FALSE
                                    tau_probe_tau_path_vec[ii] <-  updated[1]
                                    if (!tau_probe_state$probing) {
                                        # message(colourise(paste0("tau probe ended at iteration ", ii, ": tau = ", signif(updated[1], 5),
                                        #                          " (tau_handover x 2^", tau_probe_state$rung, "); ADAM with constant LR from the next tau update."), "cyan"))
                                        message(colourise(paste0("tau probe ended at iteration ", ii, ": tau = ", signif(updated[1], 5),
                                                                 " (tau_handover x 2^", tau_probe_state$rung, "); ADAM with ",
                                                                 if (tau_probe_fixed_length_decay) "LR decaying to LR^2" else "constant LR",
                                                                 " from the next tau update."), "cyan"))
                                        ## "fixed_length_probe_then_decay_and_average": the jitter (if any) resumes at the next iteration:
                                        if (tau_probe_fixed_length_decay) EHMC_args_as_Rcpp_List$randomize_tau <-  randomize_tau_burnin
                                    }
                                }
                                EHMC_args_as_Rcpp_List[[tau_name]] <- updated[1]
                                EHMC_burnin_as_Rcpp_List[[adam_mean_name]] <- updated[2]
                                EHMC_burnin_as_Rcpp_List[[adam_variance_name]] <- updated[3]
                                ##
                                ## ---- count only updates that actually changed the moments:
                                ##
                                if (tau_adam_update_performed) {
                                    tau_adam_updates_performed_by_block[[block_name]] <- tau_adam_bias_correction_step
                                    if (is_main || is_joint) tau_adam_bias_correction_step_vec[ii] <- tau_adam_bias_correction_step
                                }
                                if (tau_adam_updates_performed_by_block[[block_name]] > tau_adaptation_iteration) {
                                    stop(paste0("tau ADAM update counter (", tau_adam_updates_performed_by_block[[block_name]],
                                                ") exceeds the tau adaptation iteration (", tau_adaptation_iteration,
                                                ") for block ", block_name, " at iteration ", ii, "."))
                                }
                                t_tau <- t_tau + (proc.time()[3] - t0)
                            }
                            if (!isTRUE(partitioned_HMC)) EHMC_args_as_Rcpp_List$tau_us <- EHMC_args_as_Rcpp_List$tau_main
                            L_main_during_burnin_vec[ii] <- EHMC_args_as_Rcpp_List$tau_main / EHMC_args_as_Rcpp_List$eps_main
                            L_us_during_burnin_vec[ii] <- EHMC_args_as_Rcpp_List$tau_us / EHMC_args_as_Rcpp_List$eps_us
                        }

                            ## Apply user ceilings immediately, including the last update before adaptation freezes.
                            if (!isTRUE(manual_tau) && ii >= gap && ii < n_adapt) {
                                EHMC_args_as_Rcpp_List$tau_main <- min(max_tau_main, EHMC_args_as_Rcpp_List$tau_main)
                                EHMC_args_as_Rcpp_List$tau_us <- if (isTRUE(partitioned_HMC))
                                    min(max_tau_us, EHMC_args_as_Rcpp_List$tau_us) else EHMC_args_as_Rcpp_List$tau_main
                            }
                            ## ---- "probe_then_average": log tau after the ceiling, averaged over the tau updates from iteration metric_adaptation_end_iter on:
                            if (tau_probe_then_average && ii >= gap && ii < n_adapt && ii >= tau_average_start_iteration) {
                                tau_average_sum_log_tau <-  tau_average_sum_log_tau + log(EHMC_args_as_Rcpp_List$tau_main)
                                tau_average_n_updates <-  tau_average_n_updates + 1
                            }
                            ## //////////////////   --------------------------------  update tau for ALL PARAMS ------------------------------------------------------------------------------
                            ## n_refresh is the print INTERVAL, i.e. report every n_refresh
                            ## iterations. (This block used to overwrite the argument with a
                            ## hard-coded 20 and then divide n_burnin by it, so nothing the caller
                            ## passed had any effect.) max(1L, ...) guards ii %% 0 = NA:
                            n_refresh_steps <- max(1L, as.integer(round(n_refresh)))
                            ##
                            # print(n_burnin)
                            # print(n_refresh)
                            # print(ii)
                            # print(p_jump_main)
                            # print(EHMC_args_as_Rcpp_List$eps_main)
                            # print(EHMC_args_as_Rcpp_List$tau_main)
                            # print(div_main)
                            ##
                            try({  
                            if (ii %% n_refresh_steps == 0) {
                              
                                if (n_nuisance > 0) {
                                  if (use_resident_burnin) {
                                      ## the nuisance centre is kept in the worker:
                                      EHMC_Metric_as_Rcpp_List$theta_hat_us_vec <-  matrix(resident_api$fn_persistent_burnin_get_resident_statistic(worker_ptr, "theta_hat_us_vec"))
                                  }
                                  cat("max |theta_hat|:", max(abs(EHMC_Metric_as_Rcpp_List$theta_hat_us_vec)), "\n")
                                  cat("max M_i:", max(EHMC_Metric_as_Rcpp_List$M_us_vec), "\n")
                                  cat("min M_i:", min(EHMC_Metric_as_Rcpp_List$M_us_vec), "\n")
                                  cat("max/min M ratio:", max(EHMC_Metric_as_Rcpp_List$M_us_vec)/min(EHMC_Metric_as_Rcpp_List$M_us_vec), "\n")
                                }
                                # 
                                # ## --------------------------------
                                # var_draws_us <- apply(theta_us_vectors_all_chains_input_from_R, 1, var)   # across the 4 chains, this iteration
                                # cat("median var across chains:", median(var_draws_us), " vs 1/M_us:", 1/median(EHMC_Metric_as_Rcpp_List$M_us_vec), "\n")
                                # cat("ratio (should be ~1 if estimator is right):", median(var_draws_us) * median(EHMC_Metric_as_Rcpp_List$M_us_vec), "\n")
                                # ## --------------------------------

                              
                                        # print(paste("iter = ", ii))
                                        cat(colourise(    (paste("iter = ", ii))          , "purple"), "\n")
      
                                        if (partitioned_HMC == TRUE) { # if NOT sampling all parameters at once
                                                  cat(colourise(    (paste("p_jump_main = ", round(p_jump_main, 3)))          , "green"), "\n")
                                                  # if (identical(eps_acceptance_mean, "harmonic")) {
                                                  #     cat(colourise(    (paste0("p_jump_main (harmonic mean, eps target) = ", round(p_jump_main_for_eps, 3)))          , "green"), "\n")
                                                  # }
                                                  if (eps_acceptance_mean %in% c("harmonic", "geometric")) {
                                                      cat(colourise(    (paste0("p_jump_main (", eps_acceptance_mean, " mean, eps target) = ", round(p_jump_main_for_eps, 3)))          , "green"), "\n")
                                                  }
                                                  cat(colourise(    (paste("eps_main = ", round(EHMC_args_as_Rcpp_List$eps_main, 3)))          , "blue"), "\n")
                                                  ##
                                                  message((paste("tau_main = ",  round(EHMC_args_as_Rcpp_List$tau_main, 3))))#
                                                  ##
                                                  message((paste("L_main = ",  round(ceiling(EHMC_args_as_Rcpp_List$tau_main/EHMC_args_as_Rcpp_List$eps_main), 0))))
                                                  cat(colourise(    (paste("div_main = ", sum(div_main)))          , "red"), "\n")
                                        } else {
                                                  cat(colourise(    (paste("p_jump = ", round(p_jump_main, 3)))          , "green"), "\n")
                                                  # if (identical(eps_acceptance_mean, "harmonic")) {
                                                  #     cat(colourise(    (paste0("p_jump (harmonic mean, eps target) = ", round(p_jump_main_for_eps, 3)))          , "green"), "\n")
                                                  # }
                                                  if (eps_acceptance_mean %in% c("harmonic", "geometric")) {
                                                      cat(colourise(    (paste0("p_jump (", eps_acceptance_mean, " mean, eps target) = ", round(p_jump_main_for_eps, 3)))          , "green"), "\n")
                                                  }
                                                  cat(colourise(    (paste("eps = ", round(EHMC_args_as_Rcpp_List$eps_main, 3)))          , "blue"), "\n")
                                                  ##
                                                  message((paste("tau = ",  signif(EHMC_args_as_Rcpp_List$tau_main, 3))))
                                                  ##
                                                  message((paste("L = ",  round(ceiling(EHMC_args_as_Rcpp_List$tau_main/EHMC_args_as_Rcpp_List$eps_main), 0))))
                                                  cat(colourise(    (paste("div = ", sum(div_main)))          , "red"), "\n")
                                                  if (time_criterion_active && ii >= gap && ii < n_adapt) {
                                                      message(colourise(paste0("time criterion: time_per_leapfrog_step_burnin = ", signif(time_per_leapfrog_step_burnin_vec[ii], 4), " s",
                                                                               " | tau_offset_from_sampling_overhead = ", signif(tau_offset_from_sampling_overhead_vec[ii], 4),
                                                                               " | burnin_to_sampling_leapfrog_time_ratio = ", signif(burnin_to_sampling_leapfrog_time_ratio_vec[ii], 4),
                                                                               " | time_to_target_ESS_tau_penalty (at tau) = ", signif(time_to_target_ESS_tau_penalty_at_tau_main_vec[ii], 4),
                                                                               ## " | ESS_elasticity_wrt_log_tau (estimate) = ", signif(ESS_elasticity_wrt_log_tau_vec[ii], 4)),
                                                                               " | ESS_elasticity_wrt_log_tau (estimate) = ", signif(ESS_elasticity_wrt_log_tau_vec[ii], 4),
                                                                               if (time_criterion_messages_show_rho) paste0(
                                                                                   " | lag_one_autocorrelation_rho = ", signif(lag_one_autocorrelation_rho_used_vec[ii], 4),
                                                                                   if (lag_one_autocorrelation_rho_is_adaptive) paste0(" (adaptive; estimate after this update ",
                                                                                                                                       signif(lag_one_autocorrelation_rho_estimate_after_update_vec[ii], 4), ")") else "") else ""),
                                                                        "cyan"))
                                                  }
                                        }


                                        if (partitioned_HMC == TRUE) { # if NOT sampling all parameters at once
                                            if     (sample_nuisance == TRUE)   {
                                                cat(colourise(    (paste("p_jump_us = ", round(p_jump_us, 3)))          , "green"), "\n")
                                                # if (identical(eps_acceptance_mean, "harmonic")) {
                                                #     cat(colourise(    (paste0("p_jump_us (harmonic mean, eps target) = ", round(p_jump_us_for_eps, 3)))          , "green"), "\n")
                                                # }
                                                if (eps_acceptance_mean %in% c("harmonic", "geometric")) {
                                                    cat(colourise(    (paste0("p_jump_us (", eps_acceptance_mean, " mean, eps target) = ", round(p_jump_us_for_eps, 3)))          , "green"), "\n")
                                                }
                                                cat(colourise(    (paste("eps_us = ", round(EHMC_args_as_Rcpp_List$eps_us, 3)))          , "blue"), "\n")
                                                ##
                                                message((paste("tau_us = ",  round(EHMC_args_as_Rcpp_List$tau_us, 3))))
                                                ##
                                                message((paste("L_us = ",  round(ceiling(EHMC_args_as_Rcpp_List$tau_us / EHMC_args_as_Rcpp_List$eps_us), 0))))
                                                cat(colourise(    (paste("div_us = ", sum(div_us)))          , "red"), "\n")
                                            }
                                        }
                                          
                                        # print(head(c(EHMC_Metric_as_Rcpp_List$M_inv_main_vec)))
                                      
                            }
                            })
                                    
                            if (ii == last_burnin_iteration) {

                                    try({
                                        burnin_toc_result <- tictoc::toc(log = TRUE)
                                        print(burnin_toc_result)
                                        tictoc::tic.clearlog()
                                        ## elapsed seconds straight from toc()'s numbers. Parsing the "X sec elapsed" text with "\\d+\\.\\d+"
                                        ## gave NA whenever the time rounded to a whole second (tictoc then prints e.g. "4 sec elapsed"):
                                        time_burnin <- as.numeric(burnin_toc_result$toc - burnin_toc_result$tic)
                                    })
                              
                            } 
                        
                        if (bulk_local_tuner_freeze_active) {
                            EHMC_args_as_Rcpp_List$eps_main <-  bulk_local_tuner_frozen_kernel$eps_main
                            EHMC_args_as_Rcpp_List$eps_us <-  bulk_local_tuner_frozen_kernel$eps_us
                            EHMC_args_as_Rcpp_List$tau_main <-  bulk_local_tuner_frozen_kernel$tau_main
                            EHMC_args_as_Rcpp_List$tau_us <-  bulk_local_tuner_frozen_kernel$tau_us
                        }
                        L_main_iter_ii <-  EHMC_args_as_Rcpp_List$tau_main /  EHMC_args_as_Rcpp_List$eps_main 
                        L_main_during_burnin_vec[ii] <- L_main_iter_ii
                        tau_main_during_burnin_vec[ii] <- EHMC_args_as_Rcpp_List$tau_main
                        eps_main_during_burnin_vec[ii] <- EHMC_args_as_Rcpp_List$eps_main
                        if (ii <= n_burnin) {
                            if (identical(dim(theta_main_vectors_all_chains_input_from_R), c(as.integer(n_params_main), as.integer(n_chains_burnin)))) {
                                burnin_trace_main_all_chains[ii, , ] <-  theta_main_vectors_all_chains_input_from_R
                            }
                            burnin_metric_main_variance_history[ii, ] <-  if (identical(metric_shape_main, "dense")) diag(EHMC_Metric_as_Rcpp_List$M_inv_dense_main) else
                                                                              c(EHMC_Metric_as_Rcpp_List$M_inv_main_vec)
                            if (isTRUE(sample_nuisance) && length(EHMC_Metric_as_Rcpp_List$M_inv_us_vec) > 0) {
                                burnin_metric_nuisance_variance_quantiles_history[ii, ] <-  stats::quantile(c(EHMC_Metric_as_Rcpp_List$M_inv_us_vec),
                                                                                                           probs = c(0.05, 0.25, 0.50, 0.75, 0.95), names = FALSE)
                            }
                        }
                        if (isTRUE(partitioned_HMC) && isTRUE(sample_nuisance)) {
                            tau_us_during_burnin_vec[ii] <- EHMC_args_as_Rcpp_List$tau_us
                            eps_us_during_burnin_vec[ii] <- EHMC_args_as_Rcpp_List$eps_us
                        }


    }
    ##
    stage_wall <- time_burnin                      # wall time of THIS stage only
    time_burnin <- time_burnin + time_pre_burnin   # cumulative (returned, leave as-is)
    ##
    L_main_during_burnin <- if (length(x = L_main_during_burnin_vec) > 0L) mean(L_main_during_burnin_vec, na.rm = TRUE) else NA_real_
    L_us_during_burnin <-   if (length(x = L_us_during_burnin_vec) > 0L) mean(L_us_during_burnin_vec, na.rm = TRUE) else NA_real_
    ##
    ## ---- "probe_then_average": the tau handed to sampling is exp(mean log tau) over the averaged tau updates (before any tau_sampling_scale):
    if (tau_probe_then_average && !bulk_local_tuner_freeze_active) {
          ## "fixed_length_probe_then_decay_and_average": randomize_tau back to randomize_tau_burnin (a probe that never ended left it FALSE):
          if (tau_probe_fixed_length_decay) EHMC_args_as_Rcpp_List$randomize_tau <-  randomize_tau_burnin
          tau_main_last_iterate <-  EHMC_args_as_Rcpp_List$tau_main
          if (tau_average_n_updates > 0) {
                tau_main_averaged <-  exp(tau_average_sum_log_tau / tau_average_n_updates)
                EHMC_args_as_Rcpp_List$tau_main <-  tau_main_averaged
                if (!isTRUE(partitioned_HMC)) EHMC_args_as_Rcpp_List$tau_us <-  tau_main_averaged
          }
          message(colourise(paste0("tau for sampling (probe_then_average): exp(mean log tau) over ", tau_average_n_updates, " tau updates = ",
                                   signif(EHMC_args_as_Rcpp_List$tau_main, 5), " (last tau of the burn-in: ", signif(tau_main_last_iterate, 5),
                                   "; probe ended at iteration ", tau_probe_state$end_iteration, ")"), "cyan"))
    }
    ##
    cat("stage wall time:        ", stage_wall, "\n")
    cat("time inside C++ call:   ", t_cpp_total, "\n")
    cat("time pushing state to C++:", t_push_total, "\n")
    cat("R adaptation + misc:    ", stage_wall - t_cpp_total - t_push_total, "\n")
    ##
    # time_burnin <- time_burnin + time_pre_burnin
    # ##
    # L_main_during_burnin <- mean(L_main_during_burnin_vec, na.rm = TRUE)
    # L_us_during_burnin <-   mean(L_us_during_burnin_vec, na.rm = TRUE)
    # ##
    # cat("total burnin wall time:", time_burnin, "\n")
    # cat("time inside C++ call:  ", t_cpp_total, "\n")
    # cat("R overhead + adaptation:", time_burnin - t_cpp_total, "\n")
    ##
    cat("time inside tau C++ call:  ", t_tau, "\n")
    if (debug_burnin_timing) {
        burnin_profile <- list(
            iterations = do.call(what = rbind, args = burnin_iteration_profiles),
            chains = do.call(what = rbind, args = burnin_chain_profiles),
            stage_totals = data.frame(stage_wall_seconds = stage_wall,
                                      cpp_call_seconds = t_cpp_total,
                                      push_seconds = t_push_total,
                                      R_adaptation_and_misc_seconds = stage_wall - t_cpp_total - t_push_total))
        burnin_profile$stage_totals$parallel_seconds <- sum(burnin_profile$iterations$parallel_seconds)
        burnin_profile$stage_totals$output_seconds <- sum(burnin_profile$iterations$output_seconds)
        burnin_profile$stage_totals$call_residual_seconds <- sum(burnin_profile$iterations$call_residual_seconds)
        cat("native parallel chains: ", sum(burnin_profile$iterations$parallel_seconds), "\n")
        cat("native output allocation/copy: ", sum(burnin_profile$iterations$output_seconds), "\n")
        ## The call residual includes diagnostic packaging and R/native boundary costs.
        ## It may be slightly negative because proc.time() has coarser resolution than steady_clock.
        ## Chain times overlap: do not sum them to estimate wall time. Step counts include an entered
        ## divergent step; for partitioned HMC they sum main and nuisance integrator steps.
    }
    ##
    # Rprof(NULL)
    # summaryRprof("burnin_prof.out")$by.self[1:15, ]
    ##
    ## ---- CHESSR_time / SNAPER_time: the record of the time-to-target-ESS criterion (NULL for every other criterion):
    ##
    time_criterion_burnin_record <-  NULL
    if (time_criterion_active) {
        time_per_leapfrog_step_burnin_estimate_at_end <-  fn_time_per_leapfrog_step_burnin_fit_estimate(
            time_per_leapfrog_step_burnin_fit_sums       = time_per_leapfrog_step_burnin_fit_sums,
            time_per_leapfrog_step_burnin_min_iterations = time_criterion_sampling_quantities$time_per_leapfrog_step_burnin_min_iterations)
        ##
        ## ---- at the end: the lag_one_autocorrelation_rho the next update would use (the setting, or the "adaptive" estimate from all the
        ##      moment updates):
        ##
        lag_one_autocorrelation_rho_at_end <-  fn_lag_one_autocorrelation_rho_for_tau_update( lag_one_autocorrelation_rho_setting = lag_one_autocorrelation_rho_setting,
                                                                                              lag_one_autocorrelation_rho_moments = lag_one_autocorrelation_rho_moments)
        time_criterion_quantities_at_end <-  fn_time_criterion_quantities_at_tau_update(
            time_criterion_sampling_quantities = time_criterion_sampling_quantities,
            time_per_leapfrog_step_burnin      = if (is.finite(time_criterion_sampling_quantities$time_per_leapfrog_step_burnin_user_supplied))
                                                     time_criterion_sampling_quantities$time_per_leapfrog_step_burnin_user_supplied else
                                                     time_per_leapfrog_step_burnin_estimate_at_end$time_per_leapfrog_step_burnin,
            eps_used_for_iteration             = EHMC_args_as_Rcpp_List$eps_main,
            n_iter_burnin                      = n_iter_burnin_at_adapted_tau)
        ##
        ## ---- ESS_elasticity_wrt_log_tau pooled over the second half of the tau updates (ratio of the summed chain means), next to the
        ##      mean penalty over the same updates: where the criterion has converged the two agree:
        ##
        tau_update_iterations_with_elasticity <-  which(is.finite(ChEES_or_SNAPER_statistic_per_tau_chain_mean_vec) &
                                                        is.finite(ChEES_or_SNAPER_statistic_log_tau_derivative_per_tau_chain_mean_vec))
        tau_update_iterations_second_half <-  utils::tail(tau_update_iterations_with_elasticity, ceiling(length(tau_update_iterations_with_elasticity) / 2))
        ChEES_or_SNAPER_statistic_per_tau_sum_second_half <-  sum(ChEES_or_SNAPER_statistic_per_tau_chain_mean_vec[tau_update_iterations_second_half])
        ESS_elasticity_wrt_log_tau_pooled_over_second_half_of_tau_updates <-  if (length(tau_update_iterations_second_half) > 0 &&
                                                                                  ChEES_or_SNAPER_statistic_per_tau_sum_second_half > 0) {
            sum(ChEES_or_SNAPER_statistic_log_tau_derivative_per_tau_chain_mean_vec[tau_update_iterations_second_half]) /
            ChEES_or_SNAPER_statistic_per_tau_sum_second_half
        } else NA_real_
        time_to_target_ESS_tau_penalty_mean_over_second_half_of_tau_updates <-  if (length(tau_update_iterations_second_half) > 0) {
            mean(time_to_target_ESS_tau_penalty_at_tau_main_vec[tau_update_iterations_second_half], na.rm = TRUE)
        } else NA_real_
        ##
        time_criterion_burnin_record <-  list(
            burnin_algorithm                                   = burnin_algorithm,
            n_iter_burnin                                      = n_iter_burnin_at_adapted_tau,
            n_iter_burnin_definition                           = paste0("iterations clip_iter_tau + 1 to n_burnin (", clip_iter_tau + 1, " to ", n_burnin,
                                                                        "), which run at the adapted tau; n_burnin - clip_iter_tau"),
            n_burnin                                           = n_burnin,
            clip_iter_tau                                      = clip_iter_tau,
            time_criterion_sampling_quantities                 = time_criterion_sampling_quantities,
            time_per_leapfrog_step_burnin_measurement          = paste0("wall time (Sys.time()) of each native burn-in iteration call, all burn-in chains in parallel, ",
                                                                        "regressed on max(1, ceiling(max tau_ii / eps)) by running least squares over iterations ",
                                                                        ">= clip_iter with no divergent chain; slope = time_per_leapfrog_step_burnin"),
            time_per_leapfrog_step_burnin_at_end               = time_per_leapfrog_step_burnin_estimate_at_end,
            time_per_leapfrog_step_burnin_fit_sums_at_end      = time_per_leapfrog_step_burnin_fit_sums,
            at_handover                                        = time_criterion_at_handover,
            at_last_update                                     = time_criterion_at_last_update,
            at_end                                             = list( eps_main                               = EHMC_args_as_Rcpp_List$eps_main,
                                                                       tau_main                               = EHMC_args_as_Rcpp_List$tau_main,
                                                                       tau_offset_from_sampling_overhead      = time_criterion_quantities_at_end$tau_offset_from_sampling_overhead,
                                                                       burnin_to_sampling_leapfrog_time_ratio = time_criterion_quantities_at_end$burnin_to_sampling_leapfrog_time_ratio,
                                                                       time_to_target_ESS_tau_penalty_at_tau_main = fn_time_to_target_ESS_tau_penalty(
                                                                           tau_values                             = EHMC_args_as_Rcpp_List$tau_main,
                                                                           tau_offset_from_sampling_overhead      = time_criterion_quantities_at_end$tau_offset_from_sampling_overhead,
                                                                           ## burnin_to_sampling_leapfrog_time_ratio = time_criterion_quantities_at_end$burnin_to_sampling_leapfrog_time_ratio),
                                                                           burnin_to_sampling_leapfrog_time_ratio = time_criterion_quantities_at_end$burnin_to_sampling_leapfrog_time_ratio,
                                                                           lag_one_autocorrelation_rho            = lag_one_autocorrelation_rho_at_end$value),
                                                                       lag_one_autocorrelation_rho              = lag_one_autocorrelation_rho_at_end$value,
                                                                       lag_one_autocorrelation_rho_source       = lag_one_autocorrelation_rho_at_end$source,
                                                                       ESS_elasticity_wrt_log_tau_pooled_over_second_half_of_tau_updates   = ESS_elasticity_wrt_log_tau_pooled_over_second_half_of_tau_updates,
                                                                       time_to_target_ESS_tau_penalty_mean_over_second_half_of_tau_updates = time_to_target_ESS_tau_penalty_mean_over_second_half_of_tau_updates,
                                                                       n_tau_updates_in_second_half                                        = length(tau_update_iterations_second_half)),
            iteration_of_no_interior_optimum_warning           = iteration_of_no_interior_optimum_warning,
            time_per_iter_native_call_burnin_vec               = time_per_iter_native_call_burnin_vec,
            n_leapfrog_steps_per_iter_burnin_vec               = n_leapfrog_steps_per_iter_burnin_vec,
            used_in_time_per_leapfrog_step_burnin_fit_vec      = used_in_time_per_leapfrog_step_burnin_fit_vec,
            time_per_leapfrog_step_burnin_vec                  = time_per_leapfrog_step_burnin_vec,
            tau_offset_from_sampling_overhead_vec              = tau_offset_from_sampling_overhead_vec,
            burnin_to_sampling_leapfrog_time_ratio_vec         = burnin_to_sampling_leapfrog_time_ratio_vec,
            time_to_target_ESS_tau_penalty_at_tau_main_vec     = time_to_target_ESS_tau_penalty_at_tau_main_vec,
            ESS_elasticity_wrt_log_tau_vec                     = ESS_elasticity_wrt_log_tau_vec,
            ChEES_or_SNAPER_statistic_per_tau_chain_mean_vec   = ChEES_or_SNAPER_statistic_per_tau_chain_mean_vec,
            ## ChEES_or_SNAPER_statistic_log_tau_derivative_per_tau_chain_mean_vec = ChEES_or_SNAPER_statistic_log_tau_derivative_per_tau_chain_mean_vec)
            ChEES_or_SNAPER_statistic_log_tau_derivative_per_tau_chain_mean_vec = ChEES_or_SNAPER_statistic_log_tau_derivative_per_tau_chain_mean_vec,
            ##
            ## ---- MALT's exponent: the setting, the lag_one_autocorrelation_rho used at each tau update and ("adaptive") the estimate after
            ##      each moment update, the valid chains of each update and the final running moments:
            lag_one_autocorrelation_rho_setting                   = lag_one_autocorrelation_rho_setting,
            lag_one_autocorrelation_rho_moment_averaging_offset   = lag_one_autocorrelation_rho_moment_averaging_offset,
            lag_one_autocorrelation_rho_used_vec                  = lag_one_autocorrelation_rho_used_vec,
            lag_one_autocorrelation_rho_estimate_after_update_vec = lag_one_autocorrelation_rho_estimate_after_update_vec,
            lag_one_autocorrelation_rho_n_valid_chains_vec        = lag_one_autocorrelation_rho_n_valid_chains_vec,
            lag_one_autocorrelation_rho_moments_at_end            = lag_one_autocorrelation_rho_moments)
        message(colourise(paste0("time-to-target-ESS criterion at the end of the burn-in: time_per_leapfrog_step_burnin = ",
                                 signif(time_per_leapfrog_step_burnin_estimate_at_end$time_per_leapfrog_step_burnin, 4), " s (",
                                 time_per_leapfrog_step_burnin_estimate_at_end$status, ", ",
                                 time_per_leapfrog_step_burnin_estimate_at_end$n_iterations_used, " iterations)",
                                 " | tau_offset_from_sampling_overhead = ", signif(time_criterion_quantities_at_end$tau_offset_from_sampling_overhead, 4),
                                 " | burnin_to_sampling_leapfrog_time_ratio = ", signif(time_criterion_quantities_at_end$burnin_to_sampling_leapfrog_time_ratio, 4),
                                 " | over the second half of the tau updates: ESS_elasticity_wrt_log_tau (pooled estimate) = ",
                                 signif(ESS_elasticity_wrt_log_tau_pooled_over_second_half_of_tau_updates, 4),
                                 ## ", mean time_to_target_ESS_tau_penalty = ", signif(time_to_target_ESS_tau_penalty_mean_over_second_half_of_tau_updates, 4)),
                                 ", mean time_to_target_ESS_tau_penalty = ", signif(time_to_target_ESS_tau_penalty_mean_over_second_half_of_tau_updates, 4),
                                 if (time_criterion_messages_show_rho) paste0(
                                     " | lag_one_autocorrelation_rho at the end = ", signif(lag_one_autocorrelation_rho_at_end$value, 4),
                                     " (", lag_one_autocorrelation_rho_at_end$source, ")") else ""),
                          "cyan"))
    }
    ##
    if (use_resident_burnin) {
          ##
          ## ---- the resident nuisance-sized results come back: the same objects the original loop leaves in R
          theta_us_vectors_all_chains_input_from_R <-  resident_api$fn_persistent_burnin_get_state(worker_ptr, "theta_us_vectors_all_chains_output_to_R")
          EHMC_burnin_as_Rcpp_List$snaper_m_vec_us <-  resident_api$fn_persistent_burnin_get_resident_statistic(worker_ptr, "snaper_m_vec_us")
          EHMC_burnin_as_Rcpp_List$snaper_s_vec_us_empirical <-  resident_api$fn_persistent_burnin_get_resident_statistic(worker_ptr, "snaper_s_vec_us_empirical")
          EHMC_Metric_as_Rcpp_List$theta_hat_us_vec <-  matrix(resident_api$fn_persistent_burnin_get_resident_statistic(worker_ptr, "theta_hat_us_vec"))
          if (use_resident_joint_block) {
                trajectory_direction$joint <-  resident_api$fn_persistent_burnin_joint_direction_get(worker_ptr, FALSE)
          }
          ##
    }
    ##
    adaptive_metric_shrinkage_history <- adaptive_metric_shrinkage_lambda_history[is.finite(adaptive_metric_shrinkage_lambda_history)]
    names(adaptive_metric_shrinkage_history) <- as.character(which(is.finite(adaptive_metric_shrinkage_lambda_history)))
    metric_adaptive_shrinkage <- list(
        requested = metric_pooled_offdiagonal_shrinkage_resolution$requested,
        effective = metric_pooled_offdiagonal_shrinkage_resolution$effective,
        active = metric_pooled_offdiagonal_shrinkage_adaptive_active,
        reason = metric_pooled_offdiagonal_shrinkage_resolution$reason,
        metric_estimator = metric_estimator,
        intensity_scope = if (identical(metric_estimator, "pooled")) "last_resolved_covariance_proposal" else
                              "last_applied_unshrunk_blended_metric",
        history_scope = if (identical(metric_estimator, "pooled")) "covariance_proposal_resolutions" else
                            "applied_metric_updates",
        final_shrinkage = if (length(adaptive_metric_shrinkage_history) > 0) utils::tail(adaptive_metric_shrinkage_history, 1) else NA_real_,
        n_events = if (metric_pooled_offdiagonal_shrinkage_adaptive_active) adaptive_metric_shrinkage_state$n_events else 0,
        n_draws = if (metric_pooled_offdiagonal_shrinkage_adaptive_active) adaptive_metric_shrinkage_state$n_draws else 0,
        n_batches = if (metric_pooled_offdiagonal_shrinkage_adaptive_active) adaptive_metric_shrinkage_state$n_batches else 0,
        batch_length = if (metric_pooled_offdiagonal_shrinkage_adaptive_active) adaptive_metric_shrinkage_state$batch_length else NA_real_,
        weight_sum_squared = if (metric_pooled_offdiagonal_shrinkage_adaptive_active) adaptive_metric_shrinkage_state$weight_sum_squared else NA_real_,
        diagnostics = if (metric_pooled_offdiagonal_shrinkage_adaptive_active) adaptive_metric_shrinkage_last_diagnostics else NULL)
    ##
    rm(worker_ptr); 
    # gc()
    ##
    ## ---- debugging hook, INERT unless the environment variable is set: save the end-of-burn-in state and everything
    ##      needed to rebuild the persistent burn-in worker, so the C++ kernel can be replayed offline on it.
    ##      One file per burn-in (the pre-burnin and the main burn-in differ in n_burnin).
    dump_burnin_end_state_path <- Sys.getenv("BAYESMVP_DUMP_BURNIN_END_STATE_PATH")
    if (nzchar(dump_burnin_end_state_path)) {
          dump_burnin_end_state_file <- paste0(dump_burnin_end_state_path, "_nburnin", n_burnin, ".rds")
          saveRDS(list( y                                          = y,
                        Model_args_as_Rcpp_List                    = Model_args_as_Rcpp_List,
                        EHMC_args_as_Rcpp_List                     = EHMC_args_as_Rcpp_List,
                        EHMC_Metric_as_Rcpp_List                   = EHMC_Metric_as_Rcpp_List,
                        theta_main_vectors_all_chains_input_from_R = theta_main_vectors_all_chains_input_from_R,
                        theta_us_vectors_all_chains_input_from_R   = theta_us_vectors_all_chains_input_from_R,
                        n_chains_burnin                            = n_chains_burnin,
                        partitioned_HMC                            = partitioned_HMC,
                        diffusion_HMC                              = diffusion_HMC,
                        Model_type                                 = Model_type,
                        sample_nuisance                            = sample_nuisance,
                        force_autodiff                             = force_autodiff,
                        force_PartialLog                           = force_PartialLog,
                        multi_attempts                             = multi_attempts,
                        n_threads_WCP                              = n_threads_WCP,
                        seed                                       = seed,
                        eps_main_during_burnin_vec                 = eps_main_during_burnin_vec),
                  file = dump_burnin_end_state_file)
          message("burn-in end state saved to ", dump_burnin_end_state_file)
          if (identical(Sys.getenv("BAYESMVP_STOP_AFTER_BURNIN_DUMP"), as.character(n_burnin))) {
                stop("BAYESMVP_STOP_AFTER_BURNIN_DUMP = ", n_burnin, ": stopping after the burn-in dump.")
          }
    }
    ##
    if (!is.null(bulk_local_tuner_state)) {
        bulk_local_tuner_metadata$accepted_draws <-  if (is.null(bulk_local_tuner_state$draws)) NULL else
              bulk_local_tuner_state$draws[bulk_local_tuner_state$selected_indices, , , drop = FALSE]
        bulk_local_tuner_metadata$leapfrog_steps <-  bulk_local_tuner_state$leapfrog_steps
        bulk_local_tuner_metadata$selected_indices <-  bulk_local_tuner_state$selected_indices
        bulk_local_tuner_metadata$parameter_names <-  bulk_local_tuner_names
        bulk_local_tuner_metadata$selected_parameter_names <-
              bulk_local_tuner_names[bulk_local_tuner_state$selected_indices]
        bulk_local_tuner_metadata$cost_contract <-  bulk_local_tuner_cost_contract
        bulk_local_tuner_metadata$epoch_count <-  bulk_local_tuner_state$epoch_count
        bulk_local_tuner_metadata$nominal_tau_for_sampling <-  EHMC_args_as_Rcpp_List$tau_main
        bulk_local_tuner_metadata$eps_for_sampling <-  EHMC_args_as_Rcpp_List$eps_main
    }
    return(list(n_chains_burnin = n_chains_burnin,
                bulk_local_tuner_metadata = bulk_local_tuner_metadata,
                burnin_profile = burnin_profile,
                burnin_schedule = resolved_burnin_schedule,
                n_burnin = n_burnin,
                time_burnin = time_burnin,
                ##
                eps_main =  EHMC_args_as_Rcpp_List$eps_main,
                tau_main =  EHMC_args_as_Rcpp_List$tau_main,
                eps_us =  EHMC_args_as_Rcpp_List$eps_us,
                tau_us =  EHMC_args_as_Rcpp_List$tau_us,
                tau_main_during_burnin_vec = tau_main_during_burnin_vec,
                eps_main_during_burnin_vec = eps_main_during_burnin_vec,
                ## the rate criteria's trajectory cost offset (4 Oct 2026): eps x c per tau update, and c:
                rate_criterion_cost_offset_steps = rate_criterion_cost_offset_steps,
                tau_cost_offset_vec = tau_cost_offset_vec,
                ## burn-in record for metric diagnosis (see its allocation):
                burnin_trace_main_all_chains = burnin_trace_main_all_chains,
                burnin_metric_main_variance_history = burnin_metric_main_variance_history,
                burnin_metric_nuisance_variance_quantiles_history = burnin_metric_nuisance_variance_quantiles_history,
                tau_us_during_burnin_vec = tau_us_during_burnin_vec,
                eps_us_during_burnin_vec = eps_us_during_burnin_vec,
                ## cross-chain mean acceptance for the eps update ("harmonic" | "arithmetic") and its per-iteration values:
                eps_acceptance_mean = eps_acceptance_mean,
                p_jump_main_during_burnin_vec = p_jump_main_during_burnin_vec,
                p_jump_main_for_eps_during_burnin_vec = p_jump_main_for_eps_during_burnin_vec,
                p_jump_us_during_burnin_vec = p_jump_us_during_burnin_vec,
                p_jump_us_for_eps_during_burnin_vec = p_jump_us_for_eps_during_burnin_vec,
                ## pooled metric estimator: window resets (as given, and the iterations r they resolve to; reset at the start
                ## of iteration r + 1) and off-diagonal shrinkage. Read only when metric_estimator = "pooled":
                metric_estimator                      = metric_estimator,
                metric_pooled_window_resets           = metric_pooled_window_resets,
                metric_pooled_window_reset_iterations = metric_window_resets,
                metric_pooled_offdiagonal_shrinkage   = metric_pooled_offdiagonal_shrinkage,
                metric_adaptive_shrinkage = metric_adaptive_shrinkage,
                metric_adaptive_shrinkage_history = adaptive_metric_shrinkage_history,
                ChEES_criterion_ema_vec = ChEES_criterion_ema_vec,
                ChEES_per_tau_gradient_vec = ChEES_per_tau_gradient_vec,
                tau_adaptation_version = 4,
                trajectory_parameter_block = tau_adaptation_block_effective,
                ## EXPERIMENTAL: requested and effective block feeding the trajectory-length criterion ("main" | "joint"):
                tau_adaptation_block_requested = tau_adaptation_block,
                tau_adaptation_block = tau_adaptation_block_effective,
                tau_gradient_estimator = tau_gradient_estimator,
                tau_cost_exponent = tau_cost_exponent,
                esjd_jump_power = esjd_jump_power,
                tau_jitter_burnin = tau_jitter_burnin,
                snaper_direction_main = trajectory_direction$main,
                snaper_direction_joint = trajectory_direction$joint,
                trajectory_metric_factor_main = trajectory_metric$main,
                tau_adaptation_iteration_vec = tau_adaptation_iteration_vec,
                ## tau_initial = "adaptive": the lambda_max estimates and the tau set at the handover (NULL for a numeric tau_initial):
                tau_initial_adaptive = tau_initial_adaptive_record,
                ## ADAM bias correction counts performed tau updates, not iterations (skipped updates excluded):
                tau_adam_bias_correction = "performed_update_counter",
                tau_adam_updates_main = tau_adam_updates_performed_by_block$main,
                tau_adam_updates_us = tau_adam_updates_performed_by_block$us,
                tau_adam_bias_correction_step_vec = tau_adam_bias_correction_step_vec,
                ## tau-only ADAM beta1 (option NicoStan_tau_adam_beta1; NA = beta1_adam) and the beta1 the tau updates used; the tau
                ## learning-rate restart at the metric freeze (option NicoStan_tau_learning_rate_restart_at_metric_end) and the
                ## iteration from which the restarted schedule applied (NA = no restart):
                tau_adam_beta1_option = if (is.null(tau_adam_beta1_option)) NA_real_ else as.numeric(tau_adam_beta1_option),
                tau_adam_beta1_used = tau_adam_beta1,
                tau_learning_rate_restart_at_metric_end = tau_learning_rate_restart_at_metric_end,
                tau_learning_rate_restart_iteration = if (tau_learning_rate_restart_active) tau_learning_rate_restart_iter else NA_real_,
                ## tau adaptation scheme (option NicoStan_tau_adaptation_scheme) and, for "probe_then_average", the probe (end iteration, rung
                ## and tau after every probe update) and the averaged tau handed to sampling (NA for "adam_decay"):
                tau_adaptation_scheme = tau_adaptation_scheme,
                tau_probe_end_iteration = tau_probe_state$end_iteration,
                tau_probe_end_rung = if (tau_probe_then_average) tau_probe_state$rung else NA_real_,
                tau_probe_tau_path_vec = tau_probe_tau_path_vec,
                tau_average_start_iteration = if (tau_probe_then_average) tau_average_start_iteration else NA_real_,
                tau_average_n_updates = tau_average_n_updates,
                tau_main_last_iterate = tau_main_last_iterate,
                tau_main_averaged = tau_main_averaged,
                ## the sampler arguments beta1_adam / beta2_adam / eps_adam (the defaults of the options below):
                beta1_adam = beta1_adam,
                beta2_adam = beta2_adam,
                eps_adam = eps_adam,
                ## ADAM settings the eps and the tau updates actually used (options NicoStan_eps_adam_* / NicoStan_tau_adam_*):
                eps_adam_beta1_used = eps_adam_beta1,
                eps_adam_beta2_used = eps_adam_beta2,
                eps_adam_epsilon_used = eps_adam_epsilon,
                tau_adam_beta2_used = tau_adam_beta2,
                tau_adam_epsilon_used = tau_adam_epsilon,
                L_main_during_burnin_vec = L_main_during_burnin_vec,
                L_us_during_burnin_vec = L_us_during_burnin_vec,
                L_main_during_burnin = L_main_during_burnin,
                L_us_during_burnin = L_us_during_burnin,
                ##
                ## divergence-triggered tau shrink: settings, and the iterations at which it fired:
                tau_shrink_on_divergence        = tau_shrink_on_divergence,
                tau_shrink_factor               = tau_shrink_factor,
                tau_shrink_min_divergent_chains = tau_shrink_min_divergent_chains,
                tau_shrink_fired_vec            = tau_shrink_fired_vec,
                tau_shrink_n_fired              = sum(tau_shrink_fired_vec),
                ##
                ## CHESSR_time / SNAPER_time: the record of the time-to-target-ESS criterion (NULL for every other criterion):
                time_criterion = time_criterion_burnin_record,
                ##
                Model_args_as_Rcpp_List = Model_args_as_Rcpp_List,
                EHMC_Metric_as_Rcpp_List = EHMC_Metric_as_Rcpp_List,
                EHMC_args_as_Rcpp_List = EHMC_args_as_Rcpp_List,
                EHMC_burnin_as_Rcpp_List = EHMC_burnin_as_Rcpp_List,
                ##
                theta_main_vectors_all_chains_input_from_R = theta_main_vectors_all_chains_input_from_R,
                ##
                theta_nuisance_vectors_all_chains_input_from_R = theta_us_vectors_all_chains_input_from_R,
                theta_us_vectors_all_chains_input_from_R = theta_us_vectors_all_chains_input_from_R))
    
  
  
}























