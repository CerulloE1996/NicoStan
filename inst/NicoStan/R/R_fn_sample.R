

## R_fn_sample.R
##
##
## ---- Unit-Gaussian factor for tau_sampling_scale = "gaussian_matched":
##
## Unit-Gaussian heuristic, derived from the Gaussian analysis of
## Hoffman, Radul and Sountsov (2021, AISTATS, "An adaptive MCMC scheme for setting trajectory lengths
## in Hamiltonian Monte Carlo"). It is NOT a published method. Full derivation: docs/adaptation-notes.md.
##
## For a unit Gaussian with exact dynamics, a trajectory of length t gives ChEES(t) = sin^2(t) (up to a
## constant). NicoStan adapts tau with a FIXED length in burn-in but samples with tau_ii ~ U(0, 2 * tau_bar),
## and the optimum of each criterion differs between the two:
##
##   KE / ChEES   fixed:  max sin^2(t)                                  -> t = pi/2    = 1.5708
##                jitter: max E[sin^2(t)] = 1/2 - sin(4T) / (8T)        -> T = 1.1234   factor 0.7151
##   CHESSR /     fixed:  max sin^2(t) / t                              -> t = 1.1656
##   SNAPER       jitter: max E[sin^2(t) / t] (per-chain mean of C_i / tau_i, as coded) -> T = 0.8950  factor 0.7678
##   CHESSR_log   fixed:  max sin^2(t) / t                              -> t = 1.1656
##                jitter: max E[sin^2(t)] / T (ratio of means, as coded) -> T = pi/4 = 0.7854  factor 0.6738
##
## (KE uses the squared kinetic-energy change, which for the unit Gaussian has the same optimum as ChEES.)
## CHESSR_time / SNAPER_time (R_fn_time_criterion.R) use the CHESSR / SNAPER factor 0.7678. It is exact when their penalty is 1
## (sampling_overhead_in_leapfrog_steps = 0 and burnin_to_sampling_leapfrog_time_ratio = 0); with a positive
## tau_offset_from_sampling_overhead the jittered optimum lies between the ChEES and the rate factors, and that offset is not a
## unit-Gaussian constant, so 0.7678 is the documented approximation.
## With MALT's exponent, lag_one_autocorrelation_rho < 1 (the penalty times (1 + lag_one_autocorrelation_rho) / 2; R_fn_time_criterion.R),
## the fixed-length optimum moves from sin^2(t) / t towards sin^2(t) / t^((1 + lag_one_autocorrelation_rho) / 2), so 0.7678 is a further
## approximation there.
## The factors are computed here from those formulas rather than typed in, so they trace to the derivation.
##
## ESJD (squared jump E||x_t - x_0||^2 = 2 (1 - cos(t)) on the unit Gaussian) and ESJD_CHESSR (the geometric mean of the ESJD and
## CHESSR rates), both per-chain means of C_i / tau_i as coded (R_fn_metric_trajectory_adaptation.R):
##   ESJD         fixed:  max 2 (1 - cos(t)) / t                          -> t = 2.3311
##                jitter: max E[2 (1 - cos(t)) / t]                       -> T = 1.7899   factor 0.7678
##   ESJD_CHESSR  fixed:  max sqrt(2 (1 - cos(t)) / t * sin^2(t) / t)     -> t = 1.4465
##                jitter: max sqrt(E[2 (1 - cos(t)) / t] * E[sin^2(t) / t]) -> T = 1.1693   factor 0.8083
## (ESJD has the CHESSR factor: each is one term (1 - cos(omega t)) / t, omega = 1 for ESJD and 2 for CHESSR, and both optima scale
## as 1 / omega.)
##
## ESJD_SNAPER has the same unit-Gaussian trajectory-length factor as ESJD_CHESSR: the SNAPER squared-change rate is proportional to
## sin^2(t) / t for a fixed principal direction, so the equal-weight geometric mean has the same optimiser and jittered factor.
##
fn_tau_sampling_scale_gaussian_factor <- function(burnin_algorithm) {

        ##
        ## ---- criterion values on a unit Gaussian (exact dynamics):
        ##
        fn_expected_chees_jittered <- function(tau_bar) {
                0.5 - sin(4 * tau_bar) / (8 * tau_bar)
        }
        fn_chees_rate_fixed <- function(tau_fixed) {
                sin(tau_fixed)^2 / tau_fixed
        }
        fn_expected_per_chain_rate_jittered <- function(tau_bar) {
                integral_value <- stats::integrate(f = function(tau_value) ifelse(tau_value == 0, 0, sin(tau_value)^2 / tau_value),
                                                   lower = 0,
                                                   upper = 2 * tau_bar,
                                                   rel.tol = 1e-10)$value
                integral_value / (2 * tau_bar)
        }
        fn_ratio_of_means_rate_jittered <- function(tau_bar) {
                fn_expected_chees_jittered(tau_bar) / tau_bar
        }
        fn_first_maximiser <- function(criterion_function) {
                ## (0.3, 2.0) brackets the FIRST local maximum of every criterion above and no other.
                stats::optimize(f = criterion_function,
                                interval = c(0.3, 2.0),
                                maximum = TRUE,
                                tol = 1e-10)$maximum
        }
        ##
        ## ---- fixed-length optimum / jittered-mean optimum, per criterion:
        ##
        tau_fixed_optimum_chees <- pi / 2
        tau_fixed_optimum_rate <- fn_first_maximiser(fn_chees_rate_fixed)
        ##
        if (burnin_algorithm == "ChEES") {
                return(fn_first_maximiser(fn_expected_chees_jittered) / tau_fixed_optimum_chees)
        ## } else if (burnin_algorithm %in% c("CHESSR", "SNAPER")) {
        } else if (burnin_algorithm %in% c("CHESSR", "SNAPER", "CHESSR_time", "SNAPER_time")) {
                return(fn_first_maximiser(fn_expected_per_chain_rate_jittered) / tau_fixed_optimum_rate)
        } else if (burnin_algorithm == "CHESSR_log") {
                return(fn_first_maximiser(fn_ratio_of_means_rate_jittered) / tau_fixed_optimum_rate)
        }
        ##
        ## ---- ESJD and ESJD_CHESSR (see the header): the fixed-length ESJD optimum 2.3311 lies beyond 2.0, so ESJD takes the bracket
        ##      (0.3, 4.0), which holds its first local maximum (fixed and jittered) and no other; ESJD_CHESSR keeps (0.3, 2.0):
        ##
        fn_squared_jump_rate_fixed <- function(tau_fixed) {
                2 * (1 - cos(tau_fixed)) / tau_fixed
        }
        fn_expected_per_chain_squared_jump_rate_jittered <- function(tau_bar) {
                integral_value <- stats::integrate(f = function(tau_value) ifelse(tau_value == 0, 0, 2 * (1 - cos(tau_value)) / tau_value),
                                                   lower = 0,
                                                   upper = 2 * tau_bar,
                                                   rel.tol = 1e-10)$value
                integral_value / (2 * tau_bar)
        }
        fn_first_maximiser_up_to_four <- function(criterion_function) {
                stats::optimize(f = criterion_function,
                                interval = c(0.3, 4.0),
                                maximum = TRUE,
                                tol = 1e-10)$maximum
        }
        if (burnin_algorithm == "ESJD") {
                return(fn_first_maximiser_up_to_four(fn_expected_per_chain_squared_jump_rate_jittered) /
                       fn_first_maximiser_up_to_four(fn_squared_jump_rate_fixed))
        } else if (burnin_algorithm == "ESJD_SNAPER") {
                return(fn_first_maximiser(function(tau_bar) sqrt(fn_expected_per_chain_squared_jump_rate_jittered(tau_bar) * fn_expected_per_chain_rate_jittered(tau_bar))) /
                       fn_first_maximiser(function(tau_fixed) sqrt(fn_squared_jump_rate_fixed(tau_fixed) * fn_chees_rate_fixed(tau_fixed))))
        } else if (burnin_algorithm == "ESJD_CHESSR") {
                return(fn_first_maximiser(function(tau_bar) sqrt(fn_expected_per_chain_squared_jump_rate_jittered(tau_bar) * fn_expected_per_chain_rate_jittered(tau_bar))) /
                       fn_first_maximiser(function(tau_fixed) sqrt(fn_squared_jump_rate_fixed(tau_fixed) * fn_chees_rate_fixed(tau_fixed))))
        }
        ##
        ## ---- LQ_ESSR: on a unit Gaussian with exact dynamics and a full momentum refresh, the lag-one autocorrelations of z and z^2
        ##      are a = E[cos(t)] and b = E[cos(t)^2] (t ~ U(0, 2 tau_bar) when jittered, t = tau when fixed), and the criterion is
        ##      min(g(a), g(b)) / tau with g(r) = (1 - r) / (1 + r). Its first maximisers are 1.0696 (jittered) and pi / 2 (fixed),
        ##      both inside (0.3, 2.0):
        ##
        if (burnin_algorithm == "LQ_ESSR") {
                fn_ESS_fraction_of_lag_one_autocorrelation <- function(lag_one_autocorrelation) {
                        (1 - lag_one_autocorrelation) / (1 + lag_one_autocorrelation)
                }
                fn_LQ_ESSR_jittered <- function(tau_bar) {
                        linear_lag_one_autocorrelation <- sin(2 * tau_bar) / (2 * tau_bar)
                        quadratic_lag_one_autocorrelation <- 0.5 + sin(4 * tau_bar) / (8 * tau_bar)
                        min(fn_ESS_fraction_of_lag_one_autocorrelation(linear_lag_one_autocorrelation),
                            fn_ESS_fraction_of_lag_one_autocorrelation(quadratic_lag_one_autocorrelation)) / tau_bar
                }
                fn_LQ_ESSR_fixed <- function(tau_fixed) {
                        min(fn_ESS_fraction_of_lag_one_autocorrelation(cos(tau_fixed)),
                            fn_ESS_fraction_of_lag_one_autocorrelation(cos(tau_fixed)^2)) / tau_fixed
                }
                return(fn_first_maximiser(fn_LQ_ESSR_jittered) / fn_first_maximiser(fn_LQ_ESSR_fixed))
        }
        stop(paste0("tau_sampling_scale = 'gaussian_matched' has no factor for burnin_algorithm = '", burnin_algorithm, "'."))

}
##

# debug = TRUE
# ##
# stream = MCMC_seed
# ##
# init_object = init_object
# ##
# n_chains_burnin = n_chains_burnin
# init_lists_per_chain = init_lists_per_chain
# ##
# parallel_method = "RcppParallel"
# ## parallel_method = "OpenMP"
# ##
# Stan_data_list = NULL
# model_args_list = model_args_list
# ##
# sample_nuisance = TRUE
# n_nuisance_override = NULL
# ##
# seed = MCMC_seed ## fixed MCMC seed
# ##
# n_burnin = n_burnin
# n_adapt = NULL
# gap = NULL
# ##
# n_chains_sampling = n_chains_sampling
# n_superchains = n_chains_sampling
# ##
# n_iter = n_iter
# ##
# adapt_delta = 0.80
# ##
# learning_rate = learning_rate
# ##s
# tau_mult = 1.60
# tau_initial = 1.0*6.283185
# ##
# manual_tau = FALSE
# tau_if_manual = 3.0
# ##
# burnin_algorithm = "CHESSR"
# ##
# diffusion_HMC = FALSE
# partitioned_HMC = FALSE
# ##
# clip_iter = 50
# clip_iter_tau = 50
# ##
# n_refresh = 100
# use_proposed = TRUE
# ##
# beta1_adam = 0.00
# beta2_adam = 0.95
# eps_adam = 1e-8
# ##
# force_autodiff = FALSE
# force_PartialLog = FALSE
# multi_attempts = FALSE
# ##
# force_autodiff_for_metric = TRUE
# force_PartialLog_for_metric = FALSE
# force_multi_attempts_for_metric = FALSE
# ##
# vect_type = "AVX512"
# ##
# Phi_type = "Phi"
# inv_Phi_type = "inv_Phi"
# ##
# # metric_type_main = "Hessian",
# # metric_shape_main = "dense",
# # ratio_M_main = 0.50,
# # interval_width_main = round(n_burnin/10),
# ##
# metric_type_main = "Empirical"
# metric_shape_main = "diag"
# ## ratio_M_main = 0.25
# ratio_M_main = 0.50
# interval_width_main = round(n_burnin/10)
# ##
# metric_type_nuisance = "Empirical"
# metric_shape_nuisance = "diag"
# ratio_M_nuisance = 0.25
# interval_width_nuisance = round(n_burnin/10)
# ##
# max_tau_main = 25.0
# max_tau_nuisance = 25.0
# ##
# max_eps_main = 0.75
# max_eps_nuisance = 0.75
# ##
# max_L = 512
# n_nuisance_to_track = NULL
##
# M_decay_type = "inverse"
# M_decay_power = 0.50
# M_decay_scale <- n_adapt / 100


#' R_fn_sample_model
#' @details Set \code{options(BayesMVP_autodiff_fallback = TRUE)} before a fit
#'   to enable an autodiff lp/gradient fallback for built-in models with multi_attempts = TRUE.
#'   If unset, it defaults to FALSE for MVP/LC-MVP/MVOP/LC-MVOP and preserves the existing
#'   TRUE default for latent_trait. Standard-scale and partial-log evaluations are tried first;
#'   autodiff is used only if both return nonfinite output, at the same parameter values.
#'   The latent_trait model skips its known-unfinished partial-log path and uses standard -> autodiff.
#'   This does not retry trajectories or rescue finite energy-error divergences. The option
#'   is frozen for the fit and returned as autodiff_fallback. For separate R worker
#'   processes, set the option in the process which calls this function.
#' @param debug_burnin_timing Collect per-iteration native timings and per-chain elapsed times and integrator step counts during burn-in.
#' @param run_in_fresh_R_process TRUE (default) runs the complete burn-in/sampling fit in a fresh R process, releasing its
#'   native worker memory when it exits. Draws and diagnostics are returned normally. FALSE runs in the calling session.
#' @param diffusion_HMC_integrator Joint diffusion integrator: "kick_flow_kick" (default) or "flow_kick_flow".
#' @param reorder_cols_MVP Run a short pre-burnin and then re-fit with the test/outcome
#'   columns permuted into the Dissmann (2013) pair-first order of the estimated
#'   correlation matrix. Supported for Model_type = "LC_MVP" and for USER-SUPPLIED
#'   Stan models (Model_type = "Stan") through the BayesMVP provider.
#'   (Model_type = "LC_MVOP" is supported as well, with binary and ordinal tests in any positions.)
#' NicoStan alone does not supply model-specific column reordering.
#'   Default (NULL, or not given): TRUE for Model_type = "LC_MVP" and "LC_MVOP" and for a Stan model
#'   whose data has \code{latent_test_order} (see below), FALSE for every other model.
#'
#'   For a Stan model whose data has the field \code{latent_test_order} (a permutation of 1:n_tests
#'   that the model uses as the order in which the tests' latent variables are conditioned on each
#'   other, e.g. the four-class LC-MVOP model), the chosen order is written into
#'   \code{latent_test_order} and nothing else is permuted: the data, the per-test entries and the
#'   outputs all stay in the original test order, so the refusal below does not apply (the model
#'   must declare \code{latent_test_order}; otherwise sample() stops).
#'
#'   For Model_type = "Stan" only \code{y} is permuted - BayesMVP cannot know which of
#'   an external model's other data objects are test-indexed - so the option is REFUSED
#'   (with a large warning banner, after which sampling continues in the original column
#'   order) unless all of the following hold: the outcome matrix in the Stan data is
#'   called \code{y} and is N x n_tests with one column per test; the model exposes a
#'   correlation matrix called \code{Omega} whose dimension matches; and NO other entry
#'   of the Stan data is shaped like n_tests. That last condition is the strict one, and
#'   deliberately so: permuting \code{y} while leaving a per-test prior, covariate or
#'   flag in place would pair each test's responses with a different test's metadata,
#'   silently. Note also that test-indexed OUTPUTS are not un-permuted (slot j corresponds
#'   to original test \code{test_perm[j]}; the permutation and its inverse are returned as
#'   \code{test_perm} / \code{test_inv_perm}), and that user-supplied test-indexed initial
#'   values are left alone. This has only been tested to work and/or be beneficial for
#'   multivariate probit-based models. For Model_type = "LC_MVP" (y, the coefficient priors, the
#'   correlation bounds, priors and known values, the covariates and the beta initial values are permuted
#'   with the tests) the summaries and traces are returned in the ORIGINAL test order: beta, Omega and
#'   L_Omega are re-indexed from the fitted order by create_summary_and_traces(), and Omega_orig,
#'   L_Omega_orig, Xbeta_baseline_nd / _d and Se / Sp / Fp_baseline are computed in the original order by
#'   the model; Omega_unconstrained_vec, the raw sampler draws, the burn-in draws and the metric stay in
#'   the sampler's coordinates of the fitted order (HMC_info$parameter_families_left_in_fitted_test_order).
#'   For Model_type = "LC_MVOP" the same holds (the ordinal metadata, the Dirichlet priors and the cutpoint
#'   initial values are permuted with the tests as well), and the cutpoints C_unc_vec, C_raw_vec and C_vec
#'   are also re-indexed into the original order of the ordinal tests (HMC_info lists every re-indexed family).
#' @param test_order_rule_for_reorder_cols_MVP The rule that sets the test (column) order after the pre-burnin
#'   of reorder_cols_MVP = TRUE (unused when reorder_cols_MVP = FALSE or when test_perm_override is given).
#'   \code{"most_nearly_deterministic_test_last_then_greedy_correlation_order"} places last the test whose
#'   class-conditional latent means sit furthest in the tails at the pre-burnin estimate (the sum over classes of
#'   the mean absolute probit linear predictor; BayesMVP's compute_tail_extremeness_per_test_LC_MVP()), and orders
#'   the other tests by the greedy pair-first rule of compute_optimal_test_order() on their own estimated
#'   correlations. \code{"greedy_correlation_order"} orders all tests by compute_optimal_test_order(), the rule
#'   used before this option existed. NULL (default, not set): the first rule for Model_type = "LC_MVP" and
#'   "greedy_correlation_order" for USER-SUPPLIED Stan models, whose intercepts are not known here; the first
#'   rule set explicitly for a Stan model stops with an error. The rule used is returned as
#'   \code{test_order_rule_for_reorder_cols_MVP}.
#'   \code{"test_with_largest_tail_category_distance_from_latent_mean_last_then_greedy_correlation_order"}
#'   (binary and ordinal tests; the default for Model_type = "LC_MVOP") places last the test whose tail
#'   categories sit furthest from the class-conditional latent mean at the pre-burnin estimate (BayesMVP's
#'   compute_tail_category_distance_from_latent_mean_per_test(), from the cutpoints and category probabilities;
#'   for a binary test it equals the measure of the first rule), and orders the other tests by the greedy rule.
#'   The first rule is for binary tests only (it stops for LC_MVOP); neither is available for a Stan model.
#'   \code{"tests_missing_for_most_people_last_then_greedy_correlation_order"} (Stan models; the default for a
#'   Stan model whose data has \code{latent_test_order}) places last the tests missing for more than half of the
#'   people (a value of \code{y} that is NA or negative counts as missing, as in the four-class LC-MVOP model),
#'   and keeps the greedy order of compute_optimal_test_order() within each of the two groups (the tests observed
#'   for most people first, then the mostly missing ones).
#' @param n_chunks_multiplier_for_PartialLog_log_scale_evaluation A positive whole number (default 3): every
#'   PartialLog (log-scale) evaluation of the built-in LC_MVP / MVP and LC_MVOP / MVOP models (the fallback of
#'   multi_attempts when the standard evaluation fails, and the evaluation forced by force_PartialLog = TRUE)
#'   requests this many times the number of chunks of the fit, without a cap, so that each
#'   PartialLog chunk, which needs about 3 times more memory per individual than a standard one, stays about the
#'   same size in memory. The nuisance values and their gradient are re-laid out exactly between the two chunk
#'   layouts. 1 = the number of chunks of the fit. Returned as
#'   \code{n_chunks_multiplier_for_PartialLog_log_scale_evaluation} and in HMC_info.
#' @param burnin_TBB_pool_equals_n_chains NULL selects TRUE for built-in models, whose within-chain work uses OpenMP.
#'   External Stan models always use FALSE, including when TRUE is supplied, so their nested TBB work can use
#'   n_chains_burnin * n_threads_WCP_burnin threads. An explicit FALSE remains available for built-in models.
#' @param tau_sampling_scale "none" (default; unchanged behaviour), "gaussian_matched" or one positive number. Multiplies the
#'   adapted tau once at the switch from burn-in to sampling, only when tau was adapted with a fixed length
#'   (randomize_tau_burnin = FALSE) and sampling is randomised; eps is not re-initialised. "gaussian_matched" uses the
#'   criterion-specific unit-Gaussian factor (an experimental heuristic, not a published method; see
#'   docs/adaptation-notes.md). The requested value, effective value, factor and tau before/after are returned.
#' @param tau_gradient_estimator "forward" (default) uses the proposed endpoint gradient; "two_ended" uses the
#'   time-reversal-averaged endpoint gradient for the trajectory-length criterion.
#' @param tau_cost_exponent Exponent in [0, 1.5] applied to the trajectory-length cost in the trajectory criterion. Default 1.
#' @param esjd_jump_power Power in {2, 3, 4} of the metric jump used by the ESJD criterion. Default 2.
#' @param tau_jitter_burnin Burn-in trajectory-length jitter: "uniform" (default) or "halton".
#' @param eps_reinit_after_pre_burnin NULL/TRUE (default) re-initialises eps with find_initial_eps at the start of the main burn-in.
#'   FALSE carries the final eps (main and nuisance) of the test-order pre-burnin (reorder_cols_MVP = TRUE) into the main burn-in
#'   and skips that search; it has no effect when no pre-burnin runs.
#' @param eps_acceptance_mean NULL/"harmonic" (default) adapts the burn-in step size (eps) so that the HARMONIC mean over the
#'   burn-in chains of the per-chain acceptance probabilities, K / sum_k (1 / alpha_k), reaches adapt_delta, as in ChEES-HMC
#'   (Hoffman, Radul & Sountsov, 2021) and SNAPER-HMC (Sountsov & Hoffman, 2022); a chain with zero acceptance (a divergent
#'   proposal, which includes |log ratio| > 1000 in either direction, or an exp(log ratio) that underflowed to 0) makes this
#'   mean 0. "arithmetic" uses mean(alpha_k), the rule used before this option existed. "geometric" uses
#'   exp(mean(log(alpha_k))), each alpha_k clamped to [1e-8, 1] first, which lies between the harmonic and the arithmetic
#'   mean. Applies to the pre-burnin and the main burn-in; the value used is returned.
#' @param tau_shrink_on_divergence,tau_shrink_factor,tau_shrink_min_divergent_chains Divergence-triggered tau shrink in the
#'   burn-in. NULL/FALSE (default) = off. TRUE multiplies tau by tau_shrink_factor (NULL = 0.95) at each adaptation iteration in
#'   which at least tau_shrink_min_divergent_chains (NULL = 2) burn-in chains diverge, which is the rule used in earlier runs.
#'   The settings and the number of iterations at which the shrink fired are returned.
#' @param metric_pooled_window_resets Pooled metric estimator (metric_estimator = "pooled") only: iterations r at which its
#'   Welford accumulators (main covariance and nuisance variances) are reset, at the start of iteration r + 1.
#'   NULL/"stan_style" (default) = round(c(0.30, 0.60) * n_adapt), the resets used before this option existed; "none" = one
#'   window from metric_start_iter to metric_adaptation_end_iter; or a numeric vector of whole numbers >= 0. Applies to the
#'   pre-burnin and the main burn-in; the value used and the reset iterations of the main burn-in are returned.
#' @param metric_pooled_offdiagonal_shrinkage Pooled metric estimator only: s in [0, 1] (NULL = 0, as before this option
#'   existed), applied once to each pooled covariance proposal of the dense main metric as (1 - s) * cov + s * diag(diag(cov)).
#'   0 keeps all off-diagonals unshrunk. Applies to the pre-burnin and the main burn-in; the value used is returned.
#'   "adaptive" (the default of $sample()) estimates s from the pooled burn-in draws of all the chains
#'   (the diagonal-target shrinkage of Schafer and Strimmer, 2005; R_fn_metric_adaptive_shrinkage.R).
#' @param time_criterion_settings burnin_algorithm = "CHESSR_time" / "SNAPER_time" only: NULL (default) or a named list of the
#'   time-to-target-ESS criterion settings (sampling timing probe, previous saved run(s), user-supplied times, ESS target); see
#'   fn_default_time_criterion_settings() and R_fn_time_criterion.R. The quantities used, their sources, the probe result and
#'   the burn-in record are returned as time_criterion; the probe's wall time is included in time_burnin.
#'   sampling_overhead_in_leapfrog_steps: NULL (default; user times > previous run > probe), a number >= 0 (used as it is; no
#'   probe) or "auto": a value for the sampling configuration (n_chains_sampling chains, n_threads_WCP_sampling threads per chain,
#'   the sampling chunk count), never from the burn-in's timings or a saved run: the built-in default for the model, that
#'   configuration and the machine's number of physical cores when one exists (fn_default_sampling_overhead_in_leapfrog_steps),
#'   otherwise the sampling timing probe in its "lite" mode. With the built-in default, as with a number, no probe runs, and
#'   burnin_to_sampling_leapfrog_time_ratio uses a user-supplied or previous-run time_per_leapfrog_step_sampling (else 0, with a
#'   warning). sampling_timing_probe_mode: NULL (default; "lite" with "auto", else "full"), "full" or "lite" (the smallest and the
#'   largest sampling_timing_probe_L_values only, and the summaries timed once, less their fixed set-up, which is measured once per
#'   R session and cached). With the default sampling_timing_probe_L_values = c(2, 8) and no cached set-up (the first lite probe
#'   of a configuration in an R session, and always with run_in_fresh_R_process = TRUE), a lite probe makes exactly the full
#'   probe's calls. The source used (built_in_default, probe_lite, probe_full, probe_lite_without_summaries,
#'   probe_full_without_summaries or user_supplied) and the two parts of the value are in
#'   time_criterion$time_criterion_sampling_quantities.
#'   lag_one_autocorrelation_rho: MALT's exponent, a number in [0, 1] (default 1 = the criterion without it) or "adaptive"
#'   (estimated during the tau adaptation); lag_one_autocorrelation_rho_moment_averaging_offset: the offset in MALT's running-moment
#'   weight (default 8). The values used at each tau update are in time_criterion$burnin.
#' @param vect_type,Phi_type,inv_Phi_type NULL (default) = not supplied, which changes nothing. These settings are NOT
#'   chosen here: for Model_type = "Stan" they are not used at all (the .stan file defines its own maths), so a non-NULL
#'   value stops; for the built-in models they are set at model initialisation via model_args_list$vect_type /
#'   $Phi_type / $inv_Phi_type (validated by BayesMVP), and a non-NULL value here must equal the initialised value,
#'   otherwise it stops and points to model_args_list.
#' @export
R_fn_sample_model  <-    function(      debug = FALSE,
                                        ##
                                        stream = NULL,
                                        ##
                                        init_object, ## This may be updated within this function
                                        ##
                                        # Model_type, ## removed this
                                        ##
                                        n_chains_burnin,  ## here as may be updated
                                        init_lists_per_chain, ## This may be updated within this function
                                        ##
                                        parallel_method,
                                        ##
                                        Stan_data_list,  ## here as may be updated
                                        model_args_list, ## here as may be updated
                                        ##
                                        sample_nuisance,  ## here as may be updated
                                        n_nuisance_override = NULL,
                                        ##
                                        seed,
                                        n_burnin,
                                        n_adapt,
                                        gap,
                                        ##
                                        n_chains_sampling,
                                        n_superchains,
                                        nuisance_jitter_scale = 0.25,
                                        ##
                                        n_iter,
                                        ##
                                        adapt_delta,
                                        ##
                                        learning_rate,
                                        ##
                                        tau_mult,
                                        tau_initial,
                                        ##
                                        manual_tau,
                                        tau_if_manual,
                                        ##
                                        ## Read tau_if_manual as a LEAPFROG-STEP count instead of an integration time, so a
                                        ## fixed-L sweep stays at the requested L while eps keeps adapting. See
                                        ## R_fn_init_and_run_burnin_CHESS for the full note.
                                        tau_if_manual_in_L_units = NULL,
                                        ##
                                        ##
                                        ## Weight the per-chain tau gradients by the Metropolis acceptance probability and
                                        ## aggregate by the acceptance-weighted mean rather than the median.
                                        tau_weight_by_p_jump = NULL,
                                        ##
                                        ## Burn-in tau ramp before the adaptation takes over: "original" (default; the
                                        ## ramp used in all earlier runs) or "staged" (5/10/20 leapfrog steps then
                                        ## pi/8 .. pi, floored at 20 * eps). See init_and_run_burnin_ChESSR.
                                        tau_ramp = NULL,
                                        ##
                                        ## eps at the ChEES handover (iteration gap) of the MAIN burn-in: TRUE (default) re-initialises
                                        ## it with find_initial_eps; FALSE keeps the eps adapted up to gap. See init_and_run_burnin_ChESSR.
                                        eps_reinit_at_ChEES_handover = NULL,
                                        ##
                                        ## Cross-chain mean acceptance targeted by the burn-in eps adaptation: "harmonic" (default; K / sum_k (1 / alpha_k),
                                        ## as in ChEES-HMC and SNAPER-HMC) or "arithmetic" (the earlier rule). See init_and_run_burnin_ChESSR.
                                        eps_acceptance_mean = NULL,
                                        ##
                                        ## Divergence-triggered tau shrink in the burn-in: NULL/FALSE (default) = off; TRUE = multiply tau by
                                        ## tau_shrink_factor (NULL = 0.95) at each adaptation iteration in which at least tau_shrink_min_divergent_chains
                                        ## (NULL = 2) burn-in chains diverge (the rule used in earlier runs). See init_and_run_burnin_ChESSR.
                                        tau_shrink_on_divergence        = NULL,
                                        tau_shrink_factor               = NULL,
                                        tau_shrink_min_divergent_chains = NULL,
                                        ##
                                        ## Pooled metric estimator (metric_estimator = "pooled") only, in the pre-burnin and the main burn-in.
                                        ## NULL for both = the earlier rule ("stan_style", 0), so a call that passes neither runs exactly as before:
                                        ##   metric_pooled_window_resets          - NULL/"stan_style" (default) = reset the Welford accumulators (main
                                        ##                                          covariance and nuisance variances) at round(c(0.30, 0.60) * n_adapt);
                                        ##                                          "none" = one window from metric_start_iter to
                                        ##                                          metric_adaptation_end_iter; or a numeric vector of reset iterations.
                                        ##   metric_pooled_offdiagonal_shrinkage  - NULL = 0; each pooled main covariance proposal becomes
                                        ##                                          (1 - s) * cov + s * diag(diag(cov)).
                                        ## See init_and_run_burnin_ChESSR and R_fn_metric_pooled_settings.R.
                                        metric_pooled_window_resets          = NULL,
                                        metric_pooled_offdiagonal_shrinkage,
                                        ## burnin_algorithm = "CHESSR_time" / "SNAPER_time" only: settings of the time-to-target-ESS criterion (sampling timing
                                        ## probe, previous saved run(s), user-supplied times, ESS target), as a named list; NULL = the defaults
                                        ## (fn_default_time_criterion_settings(), R_fn_time_criterion.R). Ignored by the other criteria.
                                        ## Also: sampling_overhead_in_leapfrog_steps = a number (no probe) or "auto" (the built-in default for the sampling
                                        ## configuration and machine, else the sampling timing probe in its "lite" mode; never the burn-in's timings or a saved run),
                                        ## sampling_timing_probe_mode = "full" / "lite", and MALT's exponent lag_one_autocorrelation_rho = a number in [0, 1]
                                        ## (default 1) or "adaptive".
                                        time_criterion_settings = NULL,
                                        ##
                                        ## eps at the START of the MAIN burn-in, after the test-order pre-burnin (reorder_cols_MVP = TRUE): TRUE (default)
                                        ## re-initialises it with find_initial_eps; FALSE carries the pre-burnin's final eps (main and nuisance) into the
                                        ## main burn-in and skips that search. Inert when no pre-burnin runs. See init_and_run_burnin_ChESSR (eps_carry_over).
                                        eps_reinit_after_pre_burnin = NULL,
                                        ##
                                        ## Centre of the nuisance Gaussian rotation in the MAIN burn-in: "running_mean_frozen" (default;
                                        ## running mean, frozen from theta_hat_us_freeze_iter so eps is tuned against the kernel sampling
                                        ## uses), "running_mean" (never frozen - the earlier behaviour, which tunes eps too big)
                                        ## or "zero" (the Feb 2026 code). See init_and_run_burnin_ChESSR.
                                        theta_hat_us_rule = NULL,
                                        theta_hat_us_freeze_iter = NULL,
                                        burnin_schedule = "automatic",
                                        metric_adaptation_end_iter = NULL,
                                        ##
                                        ## Burn-in speed-ups (NULL = unchanged behaviour):
                                        ## Historical early-stop option below is retired; n_burnin always specifies the full main burn-in.
                                        ##   burnin_post_adapt_iter - non-adapting iterations kept after n_adapt in the MAIN burn-in (NULL = all of
                                        ##                            them, i.e. n_burnin - n_adapt). eps/metric/tau are frozen there. e.g. 5.
                                        ##   pre_burnin_n_iter      - length of the test-order pre-burnin (NULL = 125), with its schedule scaled to match.
                                        pre_burnin_n_iter = NULL,
                                        ##   pre_burnin_L           - fix the pre-burnin trajectory at this many leapfrog steps (NULL = tau fixed at 3.0,
                                        ##                            or 1.5 with "pooled"). A fixed tau costs L = tau / eps, i.e. 60-100 steps while eps is
                                        ##                            still small early on; the pre-burnin only needs rough correlations. e.g. 10.
                                        pre_burnin_L = NULL,
                                        ##   share_tau_ii_across_chains_in_burnin - TRUE = one tau_ii per iteration shared by all MAIN burn-in chains
                                        ##                                          (momenta stay per-chain); NULL/FALSE = per-chain tau_ii (as before).
                                        share_tau_ii_across_chains_in_burnin = NULL,
                                        # randomize_tau_burnin = FALSE,
                                        randomize_tau_burnin,                  ## tau adapted under the sampling jitter U(0, 2 tau), as in ChEES and SNAPER
                                        randomize_tau_sampling = TRUE,
                                        tau_gradient_estimator,
                                        tau_cost_exponent,
                                        esjd_jump_power,
                                        ## every trajectory criterion: the parameter families whose main-block
                                        ## rows the criterion monitors (NULL = all rows of the adapted block; with
                                        ## tau_adaptation_block also NULL, LC_MVP / LC_MVOP use
                                        ## parms_that_Se_Sp_and_prev_depend_on_for_built_in_LCMs(); see
                                        ## tau_adaptation_block):
                                        interest_only = NULL,
                                        ## the parameters of interest (parameter families) over which every
                                        ## min-ESS-based efficiency measure of the summary is computed, whatever
                                        ## the burn-in adaptation block; NULL = interest_only, else the built-in
                                        ## LC_MVP / LC_MVOP set, else every main parameter:
                                        parameter_families_of_interest_for_min_ESS = NULL,
                                        ## every position-based trajectory criterion: TRUE = its statistic on
                                        ## the reported quantities f(theta_main) instead of the main parameters
                                        ## (main block only, interest_only NULL; see the R6 sample() method), f =
                                        ## reported_quantities_function, NULL = the built-in one of LC_MVP /
                                        ## LC_MVOP (fn_reported_quantities_function_for_built_in_LCMs()), each
                                        ## quantity centred and scaled over the trajectory starts of the last n
                                        ## burn-in iterations; FALSE (default) = the main parameters, unchanged:
                                        tau_criteria_monitor_reported_quantities = FALSE,
                                        reported_quantities_function = NULL,
                                        reported_quantities_centre_and_scale_from_last_n_iterations = 10,
                                        tau_jitter_burnin,
                                        ##   tau_sampling_scale                   - multiply the adapted tau ONCE at the burn-in -> sampling switch,
                                        ##                                          only when tau was adapted with a fixed length (randomize_tau_burnin
                                        ##                                          = FALSE) and sampling is randomised. "none" (default) = factor 1,
                                        ##                                          the behaviour before this option existed; "gaussian_matched" = the
                                        ##                                          criterion-specific unit-Gaussian factor (KE/ChEES 0.7151, CHESSR/SNAPER
                                        ##                                          0.7678, CHESSR_log 0.6738); or one positive number. eps is NOT
                                        ##                                          re-initialised. Experimental heuristic derived from the Gaussian
                                        ##                                          analysis of Hoffman, Radul and Sountsov (2021) - NOT a published method.
                                        ##                                          See docs/adaptation-notes.md.
                                        tau_sampling_scale = "none",
                                        ## sampling-phase jitter of the trajectory length: "uniform" (iid U(0, 2 tau)) or "halton" (the base-2
                                        tau_jitter_sampling_type = "uniform",
                                        bulk_local_tuner = NULL,
                                        ##   tau_adaptation_block                 - "main" (default; the behaviour before this option existed): the
                                        ##                                          trajectory-length criterion uses the main block only. "joint":
                                        ##                                          main + nuisance concatenated, still adapting the one joint tau.
                                        ##                                          EXPERIMENTAL, for testing only;
                                        ##                                          models without a sampled nuisance block fall back to "main".
                                        ##                                          "interest_only": the criterion uses only the main-block rows of the
                                        ##                                          parameter families named in interest_only (needs interest_only).
                                        ## tau_adaptation_block NULL (not set): "interest_only" whenever interest_only
                                        ## is given; for LC_MVP / LC_MVOP with interest_only also NULL and tau adapted,
                                        ## interest_only = the parameters Se, Sp and prevalence depend on (LC_MVP
                                        ## beta, p_raw; LC_MVOP beta, p_raw, C_unc_vec), so "interest_only" too;
                                        ## "main" only for a model with no parameter families of interest at all.
                                        ## interest_only given with "main" is "interest_only"; "joint" with
                                        ## interest_only stops.
                                        ## tau_adaptation_block = "main",
                                        tau_adaptation_block = NULL,
                                        ## SpESS-R only (see the R6 sample() method): the largest number
                                        ## of rows fitted per tau update (0 = every row), and whether the
                                        ## bins decay lazily:
                                        SpESS_R_max_rows_fitted_per_update = 0,
                                        SpESS_R_bins_decay_lazily = FALSE,
                                        ## SpESS-R only: the power of the soft minimum over the monitored
                                        ## rows (see the R6 sample() method; 8 = the default):
                                        SpESS_R_soft_minimum_power = 8,
                                        ## SpESS-R only: the tau move (see the R6 sample() method):
                                        SpESS_R_first_update_jumps_to_the_maximiser = FALSE,
                                        SpESS_R_doubling_needs_evidence_on_updates_in_a_row = 1,
                                        ## SpESS-R only, with a dense main metric: its statistics in the main
                                        ## parameters' own coordinates (see the R6 sample() method):
                                        SpESS_R_statistics_in_parameter_coordinates = FALSE,
                                        ## SpESS-R only: the tau handed to sampling (see the R6 sample() method):
                                        SpESS_R_sampling_tau_from_last_fraction_of_updates = 0,
                                        ## SpESS-R only: the tau handed to sampling from the lengths that
                                        ## maximised its criterion at its performed updates (see the R6
                                        ## sample() method; NULL = off):
                                        SpESS_R_sampling_tau_from_maximiser_over_last_fraction_of_updates = NULL,
                                        ## SpESS-R only: the forgetting factor of its length bins (see the R6
                                        ## sample() method; NULL = its own value):
                                        SpESS_R_bin_forgetting_factor = NULL,
                                        ## SpESS-R only: the variances its squared jumps are divided by (see the
                                        ## R6 sample() method; 0 = the moving averages, as built):
                                        SpESS_R_normalising_variances_from_last_n_updates = 0,
                                        ## SpESS-R only: the factors on its linear and quadratic ESS fractions
                                        ## (see the R6 sample() method; 1 = as built):
                                        SpESS_R_linear_ESS_fraction_multiplier = 1,
                                        SpESS_R_quadratic_ESS_fraction_multiplier = 1,
                                        ## SpESS-R only: its quadratic part from the normal-mode spectral fit of
                                        ## the squares (see the R6 sample() method; FALSE = as built):
                                        SpESS_R_quadratic_part_normal_mode_spectral_fit = FALSE,
                                        ## SpESS-R only (its update in R; needs the option above TRUE): its
                                        ## quadratic part from the linear part's spectral mixture with the pair
                                        ## modes (see the R6 sample() method; FALSE = as built):
                                        SpESS_R_quadratic_part_from_linear_spectrum_with_pair_modes = FALSE,
                                        ## SpESS-R only (its update in R): the target tau of each update, the
                                        ## shortest candidate within that fraction of the largest objective (see
                                        ## the R6 sample() method; NULL = off, the maximiser):
                                        SpESS_R_target_tau_is_shortest_within_fraction_of_best_objective = NULL,
                                        ## SpESS-R only, with the reported quantities monitored (its update in
                                        ## R): its linear part corrected by each reported quantity's integrated
                                        ## autocorrelation time measured over the last that many burn-in
                                        ## iterations (see the R6 sample() method; NULL = off, as built):
                                        SpESS_R_linear_part_measured_persistence_from_last_n_burnin_iterations =
                                              NULL,
                                        ##   burnin_TBB_pool_equals_n_chains      - NULL defaults to TRUE for built-in models (separate OpenMP WCP teams).
                                        ##                                          Stan models FORCE FALSE, even if TRUE is supplied: nested TBB work
                                        ##                                          shares the n_threads_WCP_burnin * n_chains_burnin thread budget.
                                        burnin_TBB_pool_equals_n_chains = NULL,
                                        ##   store_log_lik_trace                  - FALSE = sampling does not keep the N x n_iter per-chain log-lik trace
                                        ##                                          (720 MB at N = 10,000, 180 chains, 50 iterations, zero-filled and copied
                                        ##                                          into R). Only create_summary_and_traces(save_log_lik_trace = TRUE) reads
                                        ##                                          it. NULL/TRUE = keep it (as before). RcppParallel sampling only.
                                        store_log_lik_trace = NULL,
                                        ##
                                        burnin_algorithm = "CHESSR",
                                        diffusion_HMC,
                                        partitioned_HMC,
                                        ##
                                        clip_iter,
                                        clip_iter_tau,
                                        metric_start_iter,
                                        ##
                                        n_refresh = NULL,
                                        use_proposed = TRUE,
                                        ##
                                        beta1_adam = 0.00,
                                        beta2_adam = 0.95,
                                        eps_adam = 1e-8,
                                        ##
                                        force_autodiff,
                                        force_PartialLog,
                                        multi_attempts,
                                        ##
                                        force_autodiff_for_metric = TRUE,
                                        force_PartialLog_for_metric = FALSE,
                                        force_multi_attempts_for_metric = FALSE,
                                        ##
                                        ## vect_type / Phi_type / inv_Phi_type: NULL (default) = not supplied. For Model_type = "Stan" a
                                        ## non-NULL value stops (the .stan file defines its own maths); for the built-in models these are
                                        ## set at initialisation via model_args_list and a non-NULL value here must equal the initialised
                                        ## value (see fn_check_sample_math_settings_match_native_model_args).
                                        vect_type = NULL,
                                        Phi_type = NULL,
                                        inv_Phi_type = NULL,
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
                                        max_tau_main = 25.0,
                                        max_tau_nuisance = 25.0,
                                        ##
                                        max_eps_main,
                                        max_eps_nuisance,
                                        ##
                                        learning_rate_initial = NULL,
                                        learning_rate_initial_iter = NULL,
                                        ##
                                        ## (eps warm start: OFF unless asked for - see eps_initial below)
                                        eps_initial = NULL,
                                        eps_initial_iter = NULL,
                                        ##
                                        max_L,
                                        ##
                                        nuisance_to_track = NULL,
                                        ##
                                        use_disk,
                                        use_disk_path = "/tmp/hmc_traces",
                                        ##
                                        n_threads_WCP_burnin,
                                        n_threads_WCP_sampling,
                                        ##
                                        reorder_cols_MVP,
                                        ##
                                        metric_estimator = "pooled",
                                        ##
                                        ## Fix the test (column) order instead of estimating it from the pre-burnin, e.g.
                                        ## c(5, 4, 6, 1, 3, 2). Only used when reorder_cols_MVP = TRUE; the pre-burnin still
                                        ## runs (so a run is otherwise identical), only its estimated order is replaced.
                                        test_perm_override = NULL,
                                        ##
                                        ## The rule that sets the test order after the pre-burnin when
                                        ## reorder_cols_MVP = TRUE (see the roxygen entry):
                                        ## "most_nearly_deterministic_test_last_then_greedy_correlation_order" or
                                        ## "greedy_correlation_order"; NULL (not set) = the first rule for the
                                        ## built-in LC_MVP model and the greedy rule for external Stan models:
                                        test_order_rule_for_reorder_cols_MVP = NULL,
                                        ##
                                        ## The number-of-chunks multiplier of every PartialLog (log-scale)
                                        ## evaluation (see the roxygen entry):
                                        n_chunks_multiplier_for_PartialLog_log_scale_evaluation = 3,
                                        ##
                                        num_chunks_burnin = NULL,
                                        num_chunks_sampling = NULL,
                                        diffusion_HMC_integrator = "kick_flow_kick",
                                        ##
                                        ## ---- burn-in adaptation settings (documented in $sample()):
##
                                        tau_initial_moments_window_fraction = 0.5,
                                        tau_initial_moments_window_min_iter = 1,
                                        tau_adam_beta1 = NULL,
                                        tau_adam_beta2 = NULL,
                                        tau_adam_epsilon = NULL,
                                        eps_adam_beta1 = NULL,
                                        eps_adam_beta2 = NULL,
                                        eps_adam_epsilon = NULL,
                                        tau_adaptation_scheme = "adam_decay",
                                        tau_learning_rate_restart_at_metric_end = FALSE,
                                        eps_adam_reset_at_metric_end = TRUE,
                                        eps_learning_rate_restart_at_metric_end = TRUE,
                                        calibrate_eps_on_sampling_chains_before_sampling = TRUE,
                                        n_transitions_per_eps_calibration_block = 300,
                                        n_eps_calibration_blocks = 4,
                                        ## (NULL: never; c(lower, upper): see the documentation)
                                        repeat_sampling_with_eps_refitted_if_mean_acceptance_outside = NULL,
                                        n_eps_refits_and_repeated_samplings = 2,
                                        set_sampling_eps_to_adapt_delta_from_acceptance_of_burnin_chains = FALSE,
                                        fraction_of_burnin_iterations_at_end_used_for_sampling_eps = 0.5,
                                        ## choose_threads_and_chunks_not_given_with_Paper_1_functions = TRUE,
                                        ## time_rule_num_chunks_candidates_for_sampling = TRUE,
                                        choose_num_chunks_and_n_threads_WCP_with_Paper_1_algorithm = TRUE,
                                        budget_of_the_layout_choice_in_lp_grad_calls = 2000,
                                        fewest_seconds_of_the_budget_of_the_layout_choice = 0.5,
                                        rate_criterion_cost_offset_steps = 0,
                                        pooled_metric_warm_up = 8,
                                        pooled_metric_soft_full_weight_draws = NULL,
                                        pooled_metric_weight_k_draws = 5,
                                        pooled_metric_signal_to_noise_centred = FALSE,
                                        metric_update_rescales_eps_and_tau = TRUE,
                                        # metric_update_rescales_eps_and_tau = FALSE,
                                        learning_rate_initial_n_tau_updates_held_from_tau_handover = NULL,
                                        resident_burnin = TRUE,
                                        resident_burnin_joint_block = TRUE,
                                        tau_initial_resident_moments = TRUE,
                                        record_burnin_metric_nuisance_variance_quantiles_history = TRUE,
                                        burnin_tau_set_to_eps_so_every_trajectory_is_one_leapfrog_step = FALSE,
                                        Hessian_metric_offdiagonal_shrinkage = "adaptive",
                                        ## NULL = TRUE for the latent_trait model, FALSE for the others:
                                        autodiff_fallback = NULL,
                                        ##
                                        debug_burnin_timing = FALSE,
                                        run_in_fresh_R_process = TRUE
) {
                ## ---- global R options are not read for sampler settings: each of these is an argument of this
                ##      function (and of $sample()), so a session that still sets one of the options stops here
                ##      instead of having it ignored:
                global_R_option_names_that_the_sampler_does_not_read <-  paste0("NicoStan_", c(
                      "tau_initial_moments_window_fraction",
                      "tau_initial_moments_window_min_iter",
                      "tau_adam_beta1",
                      "tau_adam_beta2",
                      "tau_adam_epsilon",
                      "eps_adam_beta1",
                      "eps_adam_beta2",
                      "eps_adam_epsilon",
                      "tau_adaptation_scheme",
                      "tau_learning_rate_restart_at_metric_end",
                      "eps_adam_reset_at_metric_end",
                      "eps_learning_rate_restart_at_metric_end",
                      "rate_criterion_cost_offset_steps",
                      "pooled_metric_warm_up",
                      "pooled_metric_soft_full_weight_draws",
                      "pooled_metric_weight_k_draws",
                      "pooled_metric_signal_to_noise_centred",
                      "metric_update_rescales_eps_and_tau",
                      "learning_rate_initial_n_tau_updates_held_from_tau_handover",
                      "resident_burnin",
                      "resident_burnin_joint_block",
                      "tau_initial_resident_moments",
                      "record_burnin_metric_nuisance_variance_quantiles_history"),
                      ## (and the two BayesMVP options that were read the same way:)
                      "BayesMVP_autodiff_fallback", "BayesMVP_force_L1")
                global_R_options_set_that_the_sampler_does_not_read <-
                      global_R_option_names_that_the_sampler_does_not_read[
                            !vapply(X = global_R_option_names_that_the_sampler_does_not_read,
                                    FUN.VALUE = logical(1),
                                    FUN = function(option_name) is.null(getOption(option_name)))]
                if (length(global_R_options_set_that_the_sampler_does_not_read) > 0) {
                    stop(paste0("These global R options are not read by the sampler: ",
                                paste(global_R_options_set_that_the_sampler_does_not_read, collapse = ", "),
                                ". Give each setting as the $sample() argument of the same name without the ",
                                "\"NicoStan_\" prefix, and remove the option, e.g. options(",
                                global_R_options_set_that_the_sampler_does_not_read[1], " = NULL)."))
                }
                if (!is.logical(run_in_fresh_R_process) || length(run_in_fresh_R_process) != 1L || is.na(run_in_fresh_R_process)) {
                    stop("run_in_fresh_R_process must be TRUE or FALSE.")
                }
                if (!is.numeric(SpESS_R_soft_minimum_power) || length(SpESS_R_soft_minimum_power) != 1 ||
                    !is.finite(SpESS_R_soft_minimum_power) || SpESS_R_soft_minimum_power <= 0) {
                    stop("SpESS_R_soft_minimum_power must be one positive number (8 = the default).")
                }
                if (!is.logical(SpESS_R_first_update_jumps_to_the_maximiser) ||
                    length(SpESS_R_first_update_jumps_to_the_maximiser) != 1 ||
                    is.na(SpESS_R_first_update_jumps_to_the_maximiser)) {
                    stop(paste0("SpESS_R_first_update_jumps_to_the_maximiser must be TRUE or FALSE (FALSE = the ",
                                "default)."))
                }
                if (!is.numeric(SpESS_R_doubling_needs_evidence_on_updates_in_a_row) ||
                    length(SpESS_R_doubling_needs_evidence_on_updates_in_a_row) != 1 ||
                    !is.finite(SpESS_R_doubling_needs_evidence_on_updates_in_a_row) ||
                    SpESS_R_doubling_needs_evidence_on_updates_in_a_row < 1 ||
                    SpESS_R_doubling_needs_evidence_on_updates_in_a_row !=
                          floor(SpESS_R_doubling_needs_evidence_on_updates_in_a_row)) {
                    stop(paste0("SpESS_R_doubling_needs_evidence_on_updates_in_a_row must be one whole number ",
                                "of at least 1 (1 = the default)."))
                }
                if (!is.logical(SpESS_R_statistics_in_parameter_coordinates) ||
                    length(SpESS_R_statistics_in_parameter_coordinates) != 1 ||
                    is.na(SpESS_R_statistics_in_parameter_coordinates)) {
                    stop(paste0("SpESS_R_statistics_in_parameter_coordinates must be TRUE or FALSE (FALSE = the ",
                                "default)."))
                }
                if (!is.numeric(SpESS_R_sampling_tau_from_last_fraction_of_updates) ||
                    length(SpESS_R_sampling_tau_from_last_fraction_of_updates) != 1 ||
                    !is.finite(SpESS_R_sampling_tau_from_last_fraction_of_updates) ||
                    SpESS_R_sampling_tau_from_last_fraction_of_updates < 0 ||
                    SpESS_R_sampling_tau_from_last_fraction_of_updates > 1) {
                    stop(paste0("SpESS_R_sampling_tau_from_last_fraction_of_updates must be one number from ",
                                "0 to 1 (0 = the default)."))
                }
                if (!is.null(SpESS_R_sampling_tau_from_maximiser_over_last_fraction_of_updates) &&
                    (!is.numeric(SpESS_R_sampling_tau_from_maximiser_over_last_fraction_of_updates) ||
                     length(SpESS_R_sampling_tau_from_maximiser_over_last_fraction_of_updates) != 1 ||
                     !is.finite(SpESS_R_sampling_tau_from_maximiser_over_last_fraction_of_updates) ||
                     SpESS_R_sampling_tau_from_maximiser_over_last_fraction_of_updates <= 0 ||
                     SpESS_R_sampling_tau_from_maximiser_over_last_fraction_of_updates > 1)) {
                    stop(paste0("SpESS_R_sampling_tau_from_maximiser_over_last_fraction_of_updates must be ",
                                "NULL (the default: off) or one number above 0 and at most 1."))
                }
                ## (the two options each set the tau handed to sampling: at most one of them may be set)
                if (!is.null(SpESS_R_sampling_tau_from_maximiser_over_last_fraction_of_updates) &&
                    SpESS_R_sampling_tau_from_last_fraction_of_updates > 0) {
                    stop(paste0("SpESS_R_sampling_tau_from_last_fraction_of_updates and ",
                                "SpESS_R_sampling_tau_from_maximiser_over_last_fraction_of_updates each set the ",
                                "tau handed to sampling: set at most one of them (the other at its default, 0 ",
                                "or NULL respectively)."))
                }
                if (!is.null(SpESS_R_bin_forgetting_factor) &&
                    (!is.numeric(SpESS_R_bin_forgetting_factor) || length(SpESS_R_bin_forgetting_factor) != 1 ||
                     !is.finite(SpESS_R_bin_forgetting_factor) || SpESS_R_bin_forgetting_factor <= 0 ||
                     SpESS_R_bin_forgetting_factor > 1)) {
                    stop(paste0("SpESS_R_bin_forgetting_factor must be NULL (the default: SpESS-R's own 0.99) ",
                                "or one number above 0 and at most 1."))
                }
                if (!is.numeric(SpESS_R_normalising_variances_from_last_n_updates) ||
                    length(SpESS_R_normalising_variances_from_last_n_updates) != 1 ||
                    !is.finite(SpESS_R_normalising_variances_from_last_n_updates) ||
                    SpESS_R_normalising_variances_from_last_n_updates < 0 ||
                    SpESS_R_normalising_variances_from_last_n_updates !=
                          round(SpESS_R_normalising_variances_from_last_n_updates)) {
                    stop(paste0("SpESS_R_normalising_variances_from_last_n_updates must be one whole number ",
                                ">= 0 (0 = the default: the moving averages of every update)."))
                }
                for (multiplier_name in c("SpESS_R_linear_ESS_fraction_multiplier",
                                          "SpESS_R_quadratic_ESS_fraction_multiplier")) {
                    multiplier_value <-  get(multiplier_name)
                    if (!is.numeric(multiplier_value) || length(multiplier_value) != 1 ||
                        is.na(multiplier_value) || !(multiplier_value > 0)) {
                        stop(paste0(multiplier_name, " must be one number above 0 (1 = the default; Inf leaves ",
                                    "that part out of SpESS-R's soft minimum)."))
                    }
                }
                if (!is.logical(SpESS_R_quadratic_part_normal_mode_spectral_fit) ||
                    length(SpESS_R_quadratic_part_normal_mode_spectral_fit) != 1 ||
                    is.na(SpESS_R_quadratic_part_normal_mode_spectral_fit)) {
                    stop(paste0("SpESS_R_quadratic_part_normal_mode_spectral_fit must be TRUE or FALSE (FALSE = ",
                                "the default)."))
                }
                if (!is.logical(SpESS_R_quadratic_part_from_linear_spectrum_with_pair_modes) ||
                    length(SpESS_R_quadratic_part_from_linear_spectrum_with_pair_modes) != 1 ||
                    is.na(SpESS_R_quadratic_part_from_linear_spectrum_with_pair_modes)) {
                    stop(paste0("SpESS_R_quadratic_part_from_linear_spectrum_with_pair_modes must be TRUE or ",
                                "FALSE (FALSE = the default)."))
                }
                ## (the pair-mode quadratic part replaces the normal-mode fit of the squares in SpESS-R's
                ##  update in R, so it needs that option)
                if (isTRUE(SpESS_R_quadratic_part_from_linear_spectrum_with_pair_modes) &&
                    !isTRUE(SpESS_R_quadratic_part_normal_mode_spectral_fit)) {
                    stop(paste0("SpESS_R_quadratic_part_from_linear_spectrum_with_pair_modes = TRUE needs ",
                                "SpESS_R_quadratic_part_normal_mode_spectral_fit = TRUE: the pair-mode ",
                                "quadratic part replaces the normal-mode fit of the squares in SpESS-R's ",
                                "update in R, which that option selects."))
                }
                if (!is.null(SpESS_R_target_tau_is_shortest_within_fraction_of_best_objective) &&
                    (!is.numeric(SpESS_R_target_tau_is_shortest_within_fraction_of_best_objective) ||
                     length(SpESS_R_target_tau_is_shortest_within_fraction_of_best_objective) != 1 ||
                     !is.finite(SpESS_R_target_tau_is_shortest_within_fraction_of_best_objective) ||
                     SpESS_R_target_tau_is_shortest_within_fraction_of_best_objective <= 0 ||
                     SpESS_R_target_tau_is_shortest_within_fraction_of_best_objective >= 0.5)) {
                    stop(paste0("SpESS_R_target_tau_is_shortest_within_fraction_of_best_objective must be NULL ",
                                "(the default: off) or one number above 0 and below 0.5."))
                }
                if (!is.null(SpESS_R_linear_part_measured_persistence_from_last_n_burnin_iterations) &&
                    (!is.numeric(SpESS_R_linear_part_measured_persistence_from_last_n_burnin_iterations) ||
                     length(SpESS_R_linear_part_measured_persistence_from_last_n_burnin_iterations) != 1 ||
                     !is.finite(SpESS_R_linear_part_measured_persistence_from_last_n_burnin_iterations) ||
                     SpESS_R_linear_part_measured_persistence_from_last_n_burnin_iterations < 10 ||
                     SpESS_R_linear_part_measured_persistence_from_last_n_burnin_iterations !=
                           floor(SpESS_R_linear_part_measured_persistence_from_last_n_burnin_iterations))) {
                    stop(paste0("SpESS_R_linear_part_measured_persistence_from_last_n_burnin_iterations must be ",
                                "NULL (the default: off) or one whole number of burn-in iterations, 10 or more."))
                }
                if (is.infinite(SpESS_R_linear_ESS_fraction_multiplier) &&
                    is.infinite(SpESS_R_quadratic_ESS_fraction_multiplier)) {
                    stop(paste0("SpESS_R_linear_ESS_fraction_multiplier and ",
                                "SpESS_R_quadratic_ESS_fraction_multiplier cannot both be Inf (that would leave ",
                                "both parts out)."))
                }
                ## ---- tau_criteria_monitor_reported_quantities: TRUE or FALSE; with TRUE and tau adapted (with
                ##      manual tau the option is unused), its function (NULL = the built-in one, resolved
                ##      below once the model is initialised) and its window, and neither the joint block nor
                ##      interest_only (the reported quantities replace the monitored rows):
                if (!is.logical(tau_criteria_monitor_reported_quantities) ||
                    length(tau_criteria_monitor_reported_quantities) != 1 ||
                    is.na(tau_criteria_monitor_reported_quantities)) {
                    stop(paste0("tau_criteria_monitor_reported_quantities must be TRUE or FALSE (FALSE = the ",
                                "default)."))
                }
                if (isTRUE(tau_criteria_monitor_reported_quantities) && !isTRUE(manual_tau)) {
                    if (!is.null(reported_quantities_function) && !is.function(reported_quantities_function)) {
                        stop(paste0("reported_quantities_function must be NULL (the built-in reported ",
                                    "quantities of LC_MVP / LC_MVOP) or a function of a matrix of main ",
                                    "parameters (n_params_main rows x chains) that returns the reported ",
                                    "quantities (rows, with rownames, x chains)."))
                    }
                    if (!is.numeric(reported_quantities_centre_and_scale_from_last_n_iterations) ||
                        length(reported_quantities_centre_and_scale_from_last_n_iterations) != 1 ||
                        !is.finite(reported_quantities_centre_and_scale_from_last_n_iterations) ||
                        reported_quantities_centre_and_scale_from_last_n_iterations < 1 ||
                        reported_quantities_centre_and_scale_from_last_n_iterations !=
                              floor(reported_quantities_centre_and_scale_from_last_n_iterations)) {
                        stop(paste0("reported_quantities_centre_and_scale_from_last_n_iterations must be one ",
                                    "whole number of at least 1 (10 = the default)."))
                    }
                    if (identical(tau_adaptation_block, "joint")) {
                        stop(paste0("tau_criteria_monitor_reported_quantities = TRUE replaces the parameters ",
                                    "that the tau criterion monitors by the reported quantities of the main ",
                                    "block; use it with tau_adaptation_block = \"main\" (or NULL), not ",
                                    "\"joint\"."))
                    }
                    if (!is.null(interest_only)) {
                        stop(paste0("tau_criteria_monitor_reported_quantities = TRUE replaces the parameters ",
                                    "that the tau criterion monitors by the reported quantities; use it with ",
                                    "interest_only = NULL."))
                    }
                }
                ## ---- the retired burnin_algorithm values are no longer selectable (their names normalised as by
                ##      fn_normalise_burnin_algorithm(); an unknown name still stops there, below, as before):
                retired_burnin_algorithms_not_selectable <-  c("LQ_ESSR", "L_ESSR_length_response",
                                                               "LQ_ESSR_length_response", "LQ_ESSR_spectral",
                                                               "LQ_ESSR_spectral_long_bin_memory")
                burnin_algorithm_requested_normalised <-  tryCatch(
                      fn_normalise_burnin_algorithm(if_null_then_set_to(burnin_algorithm, "CHESSR")),
                      error = function(error_object) NA_character_)
                if (isTRUE(burnin_algorithm_requested_normalised %in% retired_burnin_algorithms_not_selectable)) {
                    stop(paste0("burnin_algorithm = \"", burnin_algorithm, "\" is retired and can no longer be ",
                                "selected (retired: ", paste(retired_burnin_algorithms_not_selectable,
                                                             collapse = ", "),
                                "). SpESS-R (\"LQ_ESSR_spec_bins99_evid_expand\", also \"SpESSR\"), its ESJD ",
                                "blends and \"LQ_ESSR_spec_bins99_evid_expand_jump_accept\" remain."))
                }
                ## Pass only supplied arguments, preserving defaults and missing-argument handling in the sampler.
                if (isTRUE(x = run_in_fresh_R_process)) {
                    argument_names <-  names(as.list(match.call())[-1L])
                    argument_names <-  setdiff(argument_names, "run_in_fresh_R_process")
                    ## A NULL eps_reinit_after_pre_burnin is the default: it is not forwarded, so a child process whose installed
                    ## build has no such argument still runs.
                    if (is.null(eps_reinit_after_pre_burnin)) {
                        argument_names <-  setdiff(argument_names, "eps_reinit_after_pre_burnin")
                    }
                    ## The same for a NULL eps_acceptance_mean (the default, "harmonic"):
                    if (is.null(eps_acceptance_mean)) {
                        argument_names <-  setdiff(argument_names, "eps_acceptance_mean")
                    }
                    ## The same for a NULL time_criterion_settings (the default):
                    if (is.null(time_criterion_settings)) {
                        argument_names <-  setdiff(argument_names, "time_criterion_settings")
                    }
                    ## The same holds for the divergence-triggered tau shrink options:
                    for (tau_shrink_argument_name in c("tau_shrink_on_divergence", "tau_shrink_factor", "tau_shrink_min_divergent_chains")) {
                        if (is.null(get(tau_shrink_argument_name, envir = environment(), inherits = FALSE))) {
                            argument_names <-  setdiff(argument_names, tau_shrink_argument_name)
                        }
                    }
                    ## and for the pooled metric estimator options:
                    for (metric_pooled_argument_name in c("metric_pooled_window_resets", "metric_pooled_offdiagonal_shrinkage")) {
                        if (is.null(get(metric_pooled_argument_name, envir = environment(), inherits = FALSE))) {
                            argument_names <-  setdiff(argument_names, metric_pooled_argument_name)
                        }
                    }
                    ## and for a NULL test_order_rule_for_reorder_cols_MVP (the default, not set):
                    if (is.null(test_order_rule_for_reorder_cols_MVP)) {
                        argument_names <-  setdiff(argument_names, "test_order_rule_for_reorder_cols_MVP")
                    }
                    ## and for tau_criteria_monitor_reported_quantities = FALSE (the default), with its two
                    ## settings, which are then unused:
                    if (!isTRUE(tau_criteria_monitor_reported_quantities)) {
                        reported_quantities_argument_names <-  c(
                              "tau_criteria_monitor_reported_quantities",
                              "reported_quantities_function",
                              "reported_quantities_centre_and_scale_from_last_n_iterations")
                        argument_names <-  setdiff(argument_names, reported_quantities_argument_names)
                    }
                    ## and for a NULL SpESS_R_sampling_tau_from_maximiser_over_last_fraction_of_updates (the
                    ## default, off), so that a child process whose installed build has no such argument still
                    ## runs:
                    if (is.null(SpESS_R_sampling_tau_from_maximiser_over_last_fraction_of_updates)) {
                        argument_names <-  setdiff(
                              argument_names, "SpESS_R_sampling_tau_from_maximiser_over_last_fraction_of_updates")
                    }
                    ## and the same for a NULL SpESS_R_target_tau_is_shortest_within_fraction_of_best_objective
                    ## (the default, off):
                    if (is.null(SpESS_R_target_tau_is_shortest_within_fraction_of_best_objective)) {
                        argument_names <-  setdiff(
                              argument_names, "SpESS_R_target_tau_is_shortest_within_fraction_of_best_objective")
                    }
                    ## and the same for a NULL
                    ## SpESS_R_linear_part_measured_persistence_from_last_n_burnin_iterations (the default, off):
                    if (is.null(SpESS_R_linear_part_measured_persistence_from_last_n_burnin_iterations)) {
                        argument_names <-  setdiff(argument_names,
                              "SpESS_R_linear_part_measured_persistence_from_last_n_burnin_iterations")
                    }
                    fit_frame <-  environment()
                    fit_arguments <-  setNames(lapply(argument_names, function(argument_name) {
                        get(argument_name, envir = fit_frame, inherits = FALSE)
                    }), argument_names)
                    return(fn_sample_model_in_fresh_R_process(arguments = fit_arguments))
                }
                ##
                ## Built-in likelihoods use OpenMP WCP teams; external Stan likelihoods share the outer TBB arena.
                if (identical(x = init_object$Model_type, y = "Stan")) {
                    if (isTRUE(x = burnin_TBB_pool_equals_n_chains)) {
                        message("burnin_TBB_pool_equals_n_chains = TRUE is overridden to FALSE for Stan models to allow TBB within-chain parallelism.")
                    }
                    burnin_TBB_pool_equals_n_chains <-  FALSE
                } else {
                    burnin_TBB_pool_equals_n_chains <-  if (is.null(x = burnin_TBB_pool_equals_n_chains)) TRUE else
                                                          isTRUE(x = burnin_TBB_pool_equals_n_chains)
                }
                ##
                if (!is.logical(debug_burnin_timing) || length(debug_burnin_timing) != 1 || is.na(debug_burnin_timing)) {
                    stop("debug_burnin_timing must be TRUE or FALSE.")
                }
                ##
                ##
                ## ---- vect_type / Phi_type / inv_Phi_type are not used for Stan models: stop if supplied (
                ##      previously silently ignored). Built-in models are checked after the model is (re-)initialised below:
                ##
                fn_stop_if_sample_math_settings_supplied_for_Stan_model( Model_type   = init_object$Model_type,
                                                                         vect_type    = vect_type,
                                                                         Phi_type     = Phi_type,
                                                                         inv_Phi_type = inv_Phi_type)
  
                ##
                message("Printing from R_fn_sample_model:")
                ##
                ## Helper function:
                ## ---- Should the sampler KEEP the nuisance (latent-variable) trace?
                ##
                ## Its only consumer is the post-hoc param_constrain, which needs the complete
                ## unconstrained draw [nuisance | main] ONLY when the model being constrained
                ## declares the nuisance coordinates: an external Stan model with a nuisance block,
                ## or the latent_trait skeleton. The LC_MVP / LC_MVOP / MVP / MVOP skeletons declare
                ## the main parameters only, so for them the trace has no reader anywhere - yet
                ## storing it costs n_nuisance x n_iter x n_chains doubles of memory traffic per run
                ## (86 GB at n_nuisance = 60,000, 180 chains, 1000 iterations), which showed up as
                ## a ~20% drop in gradient evaluations per second. Decided from the model's own
                ## unconstrained dimension, not from Model_type, so it stays right if a skeleton
                ## changes. An explicit n_nuisance_to_track always wins.
                ##
                ## ---- (the nuisance values the caller asked to keep: the BLOCK NAMES become positions into the
                ##      nuisance vector, 0-based, which is how the sampler stores them) ------------------------
                fn_positions_of_the_nuisance_values_to_track <- function(nuisance_to_track,
                                                                         nuisance_blocks,
                                                                         Stan_data_list,
                                                                         n_nuisance) {
                      ## (none: "none" or 0. NULL, the default, keeps every block NicoStan holds, in memory and
                      ##  never on disk, which is what a fit returns when the user named the nuisance blocks)
                      if (is.character(nuisance_to_track) && length(nuisance_to_track) == 1 &&
                          identical(nuisance_to_track, "none")) {
                            return(integer(0))
                      }
                      if (is.numeric(nuisance_to_track) && length(nuisance_to_track) == 1 &&
                          is.finite(nuisance_to_track) && nuisance_to_track == 0) {
                            return(integer(0))
                      }
                      ## (a count is never a setting the user gives: the blocks are NAMED, never counted)
                      if (!is.null(nuisance_to_track) && is.numeric(nuisance_to_track)) {
                            stop(paste0("$sample(nuisance_to_track = ): give the NAMES of the nuisance blocks to ",
                                        "keep, e.g. c(\"beta_z\", \"C_raw\"), or \"none\"; never a count."))
                      }
                      if (!is.null(nuisance_to_track)) {
                            stopifnot(is.character(nuisance_to_track), length(nuisance_to_track) >= 1)
                      }
                      if (is.null(nuisance_blocks) || nrow(nuisance_blocks) == 0) {
                            if (is.null(nuisance_to_track)) return(integer(0))
                            stop(paste0("$sample(nuisance_to_track = ): this model has no nuisance blocks held ",
                                        "by NicoStan, so there is nothing to keep."))
                      }
                      if (is.null(nuisance_to_track)) {
                            message(paste0("nuisance_to_track: keeping all ", n_nuisance, " nuisance values of ",
                                           "every draw in memory (never on disk), the blocks ",
                                           paste(nuisance_blocks$name, collapse = ", "), "."))
                            return(as.integer(seq_len(n_nuisance) - 1L))
                      }
                      names_asked_for <- unique(nuisance_to_track)
                      names_not_held <- setdiff(names_asked_for, nuisance_blocks$name)
                      if (length(names_not_held) > 0) {
                            stop(paste0("$sample(nuisance_to_track = ): ", paste(names_not_held, collapse = ", "),
                                        " are not nuisance blocks held by NicoStan; the blocks are ",
                                        paste(nuisance_blocks$name, collapse = ", "), "."))
                      }
                      ## (every block asked for: the whole nuisance vector in its own order, so that no block size
                      ##  has to be worked out - a block whose size is set in the transformed data cannot be, as
                      ##  in the network meta-analysis models, and this is the case that would otherwise stop the
                      ##  sampler for the one setting that keeps everything)
                      if (setequal(names_asked_for, nuisance_blocks$name)) {
                            message(paste0("nuisance_to_track: keeping all ", n_nuisance, " nuisance values of ",
                                           "every draw in memory (never on disk), the blocks ",
                                           paste(names_asked_for, collapse = ", "), "."))
                            return(as.integer(seq_len(n_nuisance) - 1L))
                      }
                      ## (a subset: each block's size from the data, exactly as the model's own
                      ##  nicostan_offset_of_<block> is built - the offsets are the running sum of the sizes in the
                      ##  declared order. A size that cannot be worked out from the data stops here, naming the
                      ##  block, as fn_size_of_the_nuisance_blocks_held_by_NicoStan_from_the_data() does.)
                      positions_of_every_block <- list()
                      offset_running <- 0
                      for (block_index in seq_len(nrow(nuisance_blocks))) {
                            size_of_the_block <- tryCatch(
                                  eval(parse(text = nuisance_blocks$terms[block_index]),
                                       envir = Stan_data_list) *
                                  eval(parse(text = nuisance_blocks$values_per_term[block_index]),
                                       envir = Stan_data_list),
                                  error = function(e) NA_real_)
                            if (!is.numeric(size_of_the_block) || length(size_of_the_block) != 1 ||
                                is.na(size_of_the_block) || size_of_the_block != round(size_of_the_block)) {
                                  stop(paste0("$sample(nuisance_to_track = ): the size of the nuisance block ",
                                              nuisance_blocks$name[block_index], " (",
                                              nuisance_blocks$terms[block_index], " x ",
                                              nuisance_blocks$values_per_term[block_index],
                                              ") cannot be worked out from the data, so a SUBSET of the blocks ",
                                              "cannot be given; name every block to keep them all."))
                            }
                            positions_of_every_block[[nuisance_blocks$name[block_index]]] <-
                                  offset_running + seq_len(size_of_the_block) - 1L
                            offset_running <- offset_running + size_of_the_block
                      }
                      positions <- sort(unique(unlist(positions_of_every_block[names_asked_for],
                                                   use.names = FALSE)))
                      if (any(positions < 0 | positions >= n_nuisance)) {
                            stop(paste0("$sample(nuisance_to_track = ): a position lies outside the ", n_nuisance,
                                        "-value nuisance vector; the blocks and the compiled model disagree."))
                      }
                      message(paste0("nuisance_to_track: keeping ", length(positions), " of the ", n_nuisance,
                                     " nuisance values of every draw in memory (never on disk), the blocks ",
                                     paste(names_asked_for, collapse = ", "), "."))
                      return(as.integer(positions))
                }
                ##
                if_null_then_set_to <- function(x, 
                                                set_to_this_if_null) { 
                  
                  if (is.null(x)) {
                    return(set_to_this_if_null)
                  }
                  return(x)
                  
                }
                ##
                if (is.null(seed)) {
                  stop("Specify a seed for MCMC sampling")
                }
                if (!is.null(adapt_delta) && (adapt_delta <= 0 || adapt_delta >= 1)) {
                  stop("adapt_delta must be between 0 and 1")
                }
                ##
                ##
                # ## bookmark : For the latent_trait model, the MANUAL-LOG-SCALE lp_grad function isn't yet working fully, so don't use it and print error
                # if (Model_type == "latent_trait") {
                #   if (force_PartialLog == TRUE) {
                #      stop("Error: the * MANUAL * LOG-SCALE lp_grad function isn't yet working fully, set force_PartialLog = FALSE\n
                #           However, note that the autodiff (AD) version is working. ")
                #   }
                # }
                ##
                # if (partitioned_HMC == FALSE) {
                #   if (diffusion_HMC == TRUE) {
                #     stop("Diffusion-pathspace HMC is only allowed if partitioned_HMC is set to TRUE - \n
                #                      since we can only sample the nuisance parameters using diffusion-pathspace HMC; \n
                #                      also, ensure that the model has latent variables/ nuisasnce parameters wich \n
                #                      are (approximately) normaly distributed on the raw (unconstrained) scale")
                #   }
                # }
                ##
                ## ---- Print warnings:
                ##
                if (is.null(diffusion_HMC)) {
                  warning("'diffusion_HMC' not specificed (either TRUE of FALSE) - using default (standard, non-diffusion HMC)")
                  diffusion_HMC <- FALSE 
                }
                
                if (partitioned_HMC == TRUE && is.null(sample_nuisance)) {
                  sample_nuisance <- TRUE
                } 
                ##
                ## ---- Set some MCMC defaults:
                ##
                ratio_M_us <- ratio_M_nuisance
                max_eps_us <- max_eps_nuisance
                ##
                burnin_algorithm <- fn_normalise_burnin_algorithm(if_null_then_set_to(burnin_algorithm, "CHESSR"))
                ##
                ## ---- CHESSR_time / SNAPER_time: settings of the time-to-target-ESS criterion (defaults filled in; NULL stays NULL for
                ##      the other criteria, which ignore them):
                ##
                if (fn_burnin_algorithm_is_time_criterion(burnin_algorithm) || !is.null(time_criterion_settings)) {
                    time_criterion_settings <- fn_validate_time_criterion_settings(time_criterion_settings)
                }
                ##
                parallel_method <- if_null_then_set_to(parallel_method, "RcppParallel")
                ##
                n_burnin <- if_null_then_set_to(n_burnin, 500)
                ## Resolve the main schedule once; the ordering pre-burn-in deliberately does not receive this option.
                resolved_burnin_schedule <- fn_burnin_adaptation_schedule(
                    n_burnin = n_burnin,
                    n_adapt = n_adapt,
                    burnin_schedule = if_null_then_set_to(burnin_schedule, "automatic"),
                    metric_adaptation_end_iter = metric_adaptation_end_iter,
                    theta_hat_us_freeze_iter = theta_hat_us_freeze_iter,
                    theta_hat_us_rule = if_null_then_set_to(theta_hat_us_rule, "running_mean_frozen"),
                    clip_iter = clip_iter,
                    clip_iter_tau = clip_iter_tau)
                burnin_schedule <- resolved_burnin_schedule$burnin_schedule
                n_adapt <- resolved_burnin_schedule$n_adapt
                metric_adaptation_end_iter <- resolved_burnin_schedule$metric_adaptation_end_iter
                theta_hat_us_freeze_iter <- resolved_burnin_schedule$theta_hat_us_freeze_iter
                clip_iter <- resolved_burnin_schedule$clip_iter
                clip_iter_tau <- resolved_burnin_schedule$clip_iter_tau
                ##
                ## ---- clip_iter / clip_iter_tau defaults: the configurations selected by pilot
                ## study 7 (ps_7_MCMC_settings_BayesMVP.R), which sets them per burnin length and
                ## derives clip_iter_tau as clip_iter + int (see ps_7_..._functions.R):
                ##
                ##    n_burnin   clip_iter   int   clip_iter_tau
                ##      1000        150      200       350          (0.15 / 0.35 of n_burnin)
                ##       500         75      100       175          (0.15 / 0.35)
                ##       250         50       75       125          (0.20 / 0.50)
                ##       125         20       30        50          (0.16 / 0.40; requested earlier handover)
                ##
                ## Historical 125-iteration schedule, retained for reference:
                ##       125         30       50        80          (0.24 / 0.64)
                ##
                ## Previous banding rationale (the exact 125-iteration schedule is now the exception):
                ## Shorter burnins need proportionally MORE clipping, so this is a banding rather
                ## than one ratio. The four studied lengths are reproduced exactly; outside that
                ## range the end bands' ratios are extrapolated so that very long or very short
                ## burnins still scale (and clip_iter_tau can never exceed n_burnin).
                ##
                if (is.null(clip_iter) || is.null(clip_iter_tau)) {

                      if (n_burnin >= 1000) {
                          clip_iter_default     <- round(0.15 * n_burnin)
                          clip_iter_tau_default <- round(0.35 * n_burnin)
                      } else if (n_burnin >= 500) {
                          clip_iter_default     <- 75
                          clip_iter_tau_default <- 175
                      } else if (n_burnin >= 250) {
                          clip_iter_default     <- 50
                          clip_iter_tau_default <- 125
                      } else if (n_burnin == 125) {
                          ## 40% of the 250-iteration ramp: 20 MALA iterations + 30 ramp iterations.
                          clip_iter_default     <- 20
                          clip_iter_tau_default <- 50
                      } else if (n_burnin >= 125) {
                          clip_iter_default     <- 30
                          clip_iter_tau_default <- 80
                      } else {
                          clip_iter_default     <- round(0.24 * n_burnin)
                          clip_iter_tau_default <- round(0.64 * n_burnin)
                      }
                      ##
                      clip_iter <- if_null_then_set_to(clip_iter, clip_iter_default)
                      clip_iter_tau <- if_null_then_set_to(clip_iter_tau, clip_iter_tau_default)

                }
                ##
                n_adapt <- if_null_then_set_to(n_adapt, n_burnin - round(n_burnin/10))
                # gap <- if_null_then_set_to(gap, clip_iter  + round(n_adapt / 5)) 
                ##
                gap <- if_null_then_set_to(gap, clip_iter + clip_iter_tau)
                ##
                ## (whether the sampling chains and superchains were given in $sample(): when they were not, Paper
                ##  1's check of the sampling layout chooses the chains, and the superchains follow them)
                n_chains_sampling_given_in_sample <-  !is.null(n_chains_sampling)
                n_superchains_given_in_sample <-  !is.null(n_superchains)
                n_chains_sampling <- if_null_then_set_to(n_chains_sampling, parallel::detectCores())
                ##
                # ## per-chain jitter of the inherited nuisance init (see cpp_fn_post_burnin_prep_for_sampling);
                # ## 0 = old behaviour, where n_chains_sampling / n_chains_burnin chains share one exact vector
                # nuisance_jitter_scale <- if_null_then_set_to(nuisance_jitter_scale, 0.25)
                # if (!is.numeric(nuisance_jitter_scale) || length(nuisance_jitter_scale) != 1 ||
                #     !is.finite(nuisance_jitter_scale) || nuisance_jitter_scale < 0) {
                #     stop("nuisance_jitter_scale must be a single non-negative finite number (0 disables the jitter).")
                # }
                nuisance_jitter_scale <- 0.0
                ##
                if (is.null(n_superchains)) {
                    if (n_chains_sampling < 16) { 
                      n_superchains <- n_chains_sampling
                    } else { 
                      if (n_chains_sampling %in% c(17, 31)) { 
                           n_superchains <- round(n_chains_sampling/2)
                      } else if (n_chains_sampling %in% c(32, 63)) { 
                           n_superchains <- round(n_chains_sampling/4)
                      } else if (n_chains_sampling %in% c(64, 127)) { 
                           n_superchains <- round(n_chains_sampling/8)
                      } else if (n_chains_sampling > 127) { 
                           n_superchains <- round(n_chains_sampling/16)
                      }
                    }
                }
                ##
                n_superchains <- if_null_then_set_to(n_superchains, "ChESSR")
                ##
                n_iter <- if_null_then_set_to(n_iter, 1000)
                ##
                ## adapt_delta default 0.80 (Stan's robustness choice; the Paper 2 Stan benchmarks ran at 0.80, so the
                ## with the step-size ADAM reset and learning-rate restart at the final metric update
                ## (R_fn_init_and_run_burnin_CHESS.R), 27 seeds on the LC-MVP at N = 10,000 gave at b125 time to the
                ## target min ESS 0.84 [0.74, 0.95] of the 0.80 default, bulk ESS per gradient 1.23 [1.06, 1.43] and
                ## tail 1.27 [1.07, 1.50]; b250 within-chain ESS per gradient 1.11 [1.04, 1.18]; b500 unchanged. The
                ## target alone gave time 0.90 [0.80, 1.01] at b125 and nothing significant at b250 / b500; 0.65 (the
                ## HMC optimum of Beskos et al.) was no better than 0.70 at 6 seeds. 0.70 remains available as an
                ## argument.
                adapt_delta <- if_null_then_set_to(adapt_delta, 0.80)
                ##
                ## ---- learning_rate default: the pilot study 7 pairing (the same one encoded by
                ## fn_sampler_settings_APMS_BayesMVP()), INTERPOLATED so that every burnin length
                ## gets a rate, not just the four studied ones:
                ##
                ##    n_burnin    1000     500     250     125
                ##    LR        0.0125   0.025   0.05    0.075
                ##
                ## Interpolation is linear in log(n_burnin) vs log(LR), i.e. piecewise power-law.
                ## That is the natural scale here: over [250, 1000] the studied rates are EXACTLY
                ## inversely proportional to the burnin length (LR * n_burnin = 12.5 at all three
                ## anchors), so that whole stretch is one straight line and is reproduced exactly;
                ## only the short [125, 250] segment is shallower (the rate is deliberately held
                ## below what 1/n_burnin would give, which at 125 would be 0.10).
                ##
                ## Outside [125, 1000] the nearest segment's slope is continued, so a 2000-iteration
                ## burnin gets 0.00625 and short ones keep rising - capped at 0.125, the value the
                ## previous banding used for short burnins, so that a very small n_burnin cannot
                ## produce an unstable rate.
                ##
                ## (This REPLACES a banding that tested n_burnin %in% c(601, 900) and similar, which
                ## matched only those exact values: n_burnin of 250, 500 or 700 fell through every
                ## branch and left learning_rate NULL, which then propagated into LR_main / LR_us.)
                ##
                if (is.null(learning_rate)) {

                      log_n_burnin_anchors <- log(c(125, 250, 500, 1000))
                      log_learning_rate_anchors <- log(c(0.075, 0.05, 0.025, 0.0125))
                      ##
                      ## all.inside = TRUE clamps to a real segment, so values below 125 / above 1000
                      ## continue the first / last segment rather than falling off the end:
                      segment_index <- findInterval( x = log(n_burnin),
                                                     vec = log_n_burnin_anchors,
                                                     all.inside = TRUE)
                      ##
                      log_slope <- (log_learning_rate_anchors[segment_index + 1] - log_learning_rate_anchors[segment_index]) /
                                   (log_n_burnin_anchors[segment_index + 1] - log_n_burnin_anchors[segment_index])
                      ##
                      learning_rate <- exp(log_learning_rate_anchors[segment_index] +
                                           log_slope * (log(n_burnin) - log_n_burnin_anchors[segment_index]))
                      ##
                      learning_rate <- min(signif(x = learning_rate, digits = 6), 0.125)

                }
                ##
                if (is.null(learning_rate) || !is.finite(learning_rate) || learning_rate <= 0) {
                  stop("'learning_rate' must be a single positive finite number (or NULL to use the n_burnin-based default).")
                }
                ##
                LR_main <- LR_us <- learning_rate
                ##
                tau_mult <- if_null_then_set_to(tau_mult, 1.60)
                ## tau_initial = "adaptive": the burn-in ramp targets pi and the handover sets tau = (pi/2) * sqrt(lambda_max) from the
                ## burn-in draws (see init_and_run_burnin_ChESSR); it is passed through unchanged.
                ## if (!is.null(x = tau_initial)) {
                if (!is.null(x = tau_initial) && !identical(x = tau_initial, y = "adaptive")) {
                  if (!is.numeric(x = tau_initial) || length(x = tau_initial) != 1L ||
                      !is.finite(x = tau_initial) || tau_initial <= 0) {
                    ## stop("'tau_initial' must be NULL or a single positive finite number.")
                    stop("'tau_initial' must be NULL, \"adaptive\" or a single positive finite number.")
                  }
                }
                ##
                manual_tau <- if_null_then_set_to(manual_tau, FALSE)
                tau_if_manual <- if_null_then_set_to(tau_if_manual, 3.0)
                ##
                ##      models (LC_MVP, LC_MVOP), with interest_only also NULL and tau adapted (manual_tau =
                ##      FALSE), the criterion monitors the parameters Se, Sp and prevalence depend on
                ##      (parms_that_Se_Sp_and_prev_depend_on_for_built_in_LCMs()), restricted below to the
                ##      families this model has. An explicit tau_adaptation_block ("main" = every main row, or
                ##      "joint") or interest_only always wins; every other model is unchanged ("main", every main
                ##      row):
                interest_only_from_built_in_default_for_LCMs <-  FALSE
                if (is.null(tau_adaptation_block)) {
                      tau_adaptation_block <-  "main"
                      ## if (is.null(interest_only) && !isTRUE(manual_tau)) {
                      ## (not with tau_criteria_monitor_reported_quantities = TRUE: the reported quantities
                      ##  replace the monitored rows)
                      if (is.null(interest_only) && !isTRUE(manual_tau) &&
                          !isTRUE(tau_criteria_monitor_reported_quantities)) {
                            interest_only <-  parms_that_Se_Sp_and_prev_depend_on_for_built_in_LCMs(
                                  Model_type = init_object$Model_type)
                            interest_only_from_built_in_default_for_LCMs <-  !is.null(interest_only)
                      }
                }
                ##
                ## ---- tau_adaptation_block "interest_only": the criterion uses only the main-block rows of the
                ##      parameter families named in interest_only. It is the block whenever interest_only is given
                ##      (with "joint" it stops below); it needs interest_only. The interest rows are rows of the main
                ##      block, so the burn-in runs the main block's route with only those rows in the criterion:
                if (!is.null(interest_only) && !identical(tau_adaptation_block, "joint")) {
                      tau_adaptation_block <-  "interest_only"
                }
                if (identical(tau_adaptation_block, "interest_only") && is.null(interest_only)) {
                      stop(paste0("tau_adaptation_block = \"interest_only\" needs interest_only, the parameter ",
                                  "families of interest (e.g. interest_only = c(\"beta_mu\", \"C_raw_vec\"))."))
                }
                tau_adaptation_block_as_given <-  tau_adaptation_block
                if (identical(tau_adaptation_block, "interest_only")) {
                      tau_adaptation_block <-  "main"
                }
                ##
                ## ---- the parameters of interest, over which every min-ESS-based efficiency measure of the
                ##      summary is computed, whatever the burn-in adaptation block (main, joint or
                ##      interest_only), so that fits with different blocks are compared on the same parameters:
                ##      as given; NULL = interest_only when it is given; otherwise, for LC_MVP / LC_MVOP, the
                ##      parameters Se, Sp and prevalence depend on; otherwise NULL (every main parameter). The
                ##      summary also reports the min ESS over every main parameter (the parameters block except
                ##      the nuisance parameters):
                ##
                if (is.null(parameter_families_of_interest_for_min_ESS)) {
                      if (!is.null(interest_only)) {
                            parameter_families_of_interest_for_min_ESS <-  interest_only
                      } else {
                            parameter_families_of_interest_for_min_ESS <-
                                  parms_that_Se_Sp_and_prev_depend_on_for_built_in_LCMs(
                                        Model_type = init_object$Model_type)
                      }
                }
                if (!is.null(parameter_families_of_interest_for_min_ESS) &&
                    (!is.character(parameter_families_of_interest_for_min_ESS) ||
                     length(parameter_families_of_interest_for_min_ESS) == 0 ||
                     anyNA(parameter_families_of_interest_for_min_ESS))) {
                      stop(paste0("parameter_families_of_interest_for_min_ESS must be NULL or a character ",
                                  "vector of parameter families (the names before \"[\" or \".\"), e.g. ",
                                  "c(\"beta\", \"p_raw\")."))
                }
                ##
                ## ---- Trajectory-length adaptation selected by burnin_algorithm.
                ##      The frozen legacy package preserves the historical defaults.
                tau_if_manual_in_L_units <- if_null_then_set_to(tau_if_manual_in_L_units, FALSE)

                tau_weight_by_p_jump <- if_null_then_set_to(tau_weight_by_p_jump, TRUE)
                tau_ramp <- if_null_then_set_to(tau_ramp, "original")
                eps_reinit_at_ChEES_handover <- if_null_then_set_to(eps_reinit_at_ChEES_handover, TRUE)
                eps_acceptance_mean <- if_null_then_set_to(eps_acceptance_mean, "harmonic")
                eps_reinit_after_pre_burnin <- if_null_then_set_to(eps_reinit_after_pre_burnin, TRUE)
                tau_shrink_on_divergence        <- if_null_then_set_to(tau_shrink_on_divergence, FALSE)
                tau_shrink_factor               <- if_null_then_set_to(tau_shrink_factor, 0.95)
                tau_shrink_min_divergent_chains <- if_null_then_set_to(tau_shrink_min_divergent_chains, 2)
                share_tau_ii_across_chains_in_burnin <- isTRUE(if_null_then_set_to(share_tau_ii_across_chains_in_burnin, FALSE))
                for (flag in c("randomize_tau_burnin", "randomize_tau_sampling")) {
                    value <-  get(flag)
                    if (!is.logical(value) || length(value) != 1L || is.na(value)) {
                        stop(paste0(flag, " must be TRUE or FALSE."))
                    }
                }
                fn_validate_trajectory_criterion_options(
                    tau_gradient_estimator = tau_gradient_estimator,
                    tau_cost_exponent = tau_cost_exponent,
                    esjd_jump_power = esjd_jump_power,
                    tau_jitter_burnin = tau_jitter_burnin,
                    algorithm = burnin_algorithm,
                    randomize_tau_burnin = randomize_tau_burnin)
                ##
                ## ---- tau_sampling_scale: "none", "gaussian_matched" or one positive finite number (see the argument note):
                ##
                tau_sampling_scale <- if_null_then_set_to(tau_sampling_scale, "none")
                tau_jitter_sampling_type <- if_null_then_set_to(tau_jitter_sampling_type, "uniform")
                if (!is.character(tau_jitter_sampling_type) || length(tau_jitter_sampling_type) != 1 || !tau_jitter_sampling_type %in% c("uniform", "halton")) {
                      stop("tau_jitter_sampling_type must be \"uniform\" or \"halton\".")
                }
                if (!is.null(bulk_local_tuner)) {
                    bulk_local_tuner <-  fn_bulk_local_tuner_settings(bulk_local_tuner)
                    if (!identical(burnin_algorithm, "ESJD") || isTRUE(partitioned_HMC) ||
                        !isTRUE(randomize_tau_burnin) || !isTRUE(randomize_tau_sampling) ||
                        !identical(tau_jitter_burnin, "uniform") || isTRUE(manual_tau) ||
                        isTRUE(tau_if_manual_in_L_units) || !identical(tau_sampling_scale, "none") ||
                        isTRUE(tau_shrink_on_divergence) ||
                        !identical(tau_adaptation_scheme, "adam_decay")) {
                        stop(paste0("bulk_local_tuner requires ESJD uniform-jitter adaptive tau ",
                                    "without extra tau schemes."))
                    }
                }
                {
                    tau_sampling_scale_is_valid_string <- is.character(tau_sampling_scale) &&
                                                          length(tau_sampling_scale) == 1 &&
                                                          tau_sampling_scale %in% c("none", "gaussian_matched")
                    tau_sampling_scale_is_valid_number <- is.numeric(tau_sampling_scale) &&
                                                          length(tau_sampling_scale) == 1 &&
                                                          is.finite(tau_sampling_scale) &&
                                                          tau_sampling_scale > 0
                    if (!tau_sampling_scale_is_valid_string && !tau_sampling_scale_is_valid_number) {
                        stop(paste0("tau_sampling_scale must be 'none', 'gaussian_matched' or one positive finite number; got: ",
                                    paste(as.character(tau_sampling_scale), collapse = ", ")))
                    }
                }
                store_log_lik_trace <- if_null_then_set_to(store_log_lik_trace, TRUE)
                if (!is.logical(store_log_lik_trace) || length(store_log_lik_trace) != 1 || is.na(store_log_lik_trace)) {
                    stop("store_log_lik_trace must be NULL, TRUE or FALSE.")
                }
                theta_hat_us_rule <- if_null_then_set_to(theta_hat_us_rule, "running_mean_frozen")
                if (length(theta_hat_us_rule) != 1 || !theta_hat_us_rule %in% c("running_mean_frozen", "running_mean", "zero")) {
                    stop("theta_hat_us_rule must be 'running_mean_frozen', 'running_mean' or 'zero'; got: ", paste(as.character(theta_hat_us_rule), collapse = ", "))
                }
                if (!is.null(pre_burnin_n_iter) && (length(pre_burnin_n_iter) != 1 || !is.finite(pre_burnin_n_iter) || pre_burnin_n_iter < 30)) {
                    stop("pre_burnin_n_iter must be NULL or a single number >= 30.")
                }
                if (!is.null(pre_burnin_L) && (length(pre_burnin_L) != 1 || !is.finite(pre_burnin_L) || pre_burnin_L < 1)) {
                    stop("pre_burnin_L must be NULL or a single leapfrog-step count >= 1.")
                }
                if (!is.logical(eps_reinit_at_ChEES_handover) || length(eps_reinit_at_ChEES_handover) != 1 || is.na(eps_reinit_at_ChEES_handover)) {
                    stop("eps_reinit_at_ChEES_handover must be TRUE or FALSE; got: ", paste(as.character(eps_reinit_at_ChEES_handover), collapse = ", "))
                }
                if (!is.logical(eps_reinit_after_pre_burnin) || length(eps_reinit_after_pre_burnin) != 1 || is.na(eps_reinit_after_pre_burnin)) {
                    stop("eps_reinit_after_pre_burnin must be TRUE or FALSE; got: ", paste(as.character(eps_reinit_after_pre_burnin), collapse = ", "))
                }
                if (!is.logical(tau_shrink_on_divergence) || length(tau_shrink_on_divergence) != 1 || is.na(tau_shrink_on_divergence)) {
                    stop("tau_shrink_on_divergence must be TRUE or FALSE; got: ", paste(as.character(tau_shrink_on_divergence), collapse = ", "))
                }
                if (!is.numeric(tau_shrink_factor) || length(tau_shrink_factor) != 1 || !is.finite(tau_shrink_factor) ||
                    tau_shrink_factor <= 0 || tau_shrink_factor > 1) {
                    stop("tau_shrink_factor must be a single number in (0, 1]; got: ", paste(as.character(tau_shrink_factor), collapse = ", "))
                }
                if (!is.numeric(tau_shrink_min_divergent_chains) || length(tau_shrink_min_divergent_chains) != 1 ||
                    !is.finite(tau_shrink_min_divergent_chains) || tau_shrink_min_divergent_chains < 1 ||
                    tau_shrink_min_divergent_chains != round(tau_shrink_min_divergent_chains)) {
                    stop("tau_shrink_min_divergent_chains must be a single whole number >= 1; got: ",
                         paste(as.character(tau_shrink_min_divergent_chains), collapse = ", "))
                }
                metric_estimator <- as.character(metric_estimator)
                ## if (length(metric_estimator) != 1 || !(metric_estimator %in% c("pooled", "chain_mean", "chain_mean_scaled"))) {
                if (length(metric_estimator) != 1 || !(metric_estimator %in% c("pooled", "chain_mean", "chain_mean_scaled", "per_iteration"))) {
                    ## stop("metric_estimator must be 'pooled', 'chain_mean' or 'chain_mean_scaled'; got: ",
                    stop("metric_estimator must be 'pooled', 'chain_mean', 'chain_mean_scaled' or 'per_iteration'; got: ",
                         paste(as.character(metric_estimator), collapse = ", "))
                }
                ## pooled metric estimator options (NULL = "stan_style" and 0, the earlier rule; read only when metric_estimator = "pooled"):
                metric_pooled_window_resets         <- fn_validate_metric_pooled_window_resets(metric_pooled_window_resets)
                metric_pooled_offdiagonal_shrinkage <- fn_validate_metric_pooled_offdiagonal_shrinkage(metric_pooled_offdiagonal_shrinkage)
                if (length(tau_ramp) != 1 || !tau_ramp %in% c("original", "staged")) {
                    stop("tau_ramp must be 'original' or 'staged'; got: ", paste(as.character(tau_ramp), collapse = ", "))
                }
                # if (!is.character(eps_acceptance_mean) || length(eps_acceptance_mean) != 1 || !eps_acceptance_mean %in% c("harmonic", "arithmetic")) {
                #     stop(paste0("eps_acceptance_mean must be 'harmonic' or 'arithmetic'; got: ", paste(as.character(eps_acceptance_mean), collapse = ", ")))
                # }
                # if (!is.character(eps_acceptance_mean) || length(eps_acceptance_mean) != 1 || !eps_acceptance_mean %in% c("harmonic", "arithmetic", "geometric")) {
                #     stop(paste0("eps_acceptance_mean must be 'harmonic', 'arithmetic' or 'geometric'; got: ", paste(as.character(eps_acceptance_mean), collapse = ", ")))
                # }
                ## (the harmonic mean of the chains' acceptance, in the burn-in AND in the step-size calibration on the
                ##  sampling chains, is the only one allowed: it is the fix that made every check against cmdstanr pass
                ##  MAKE SURE THATS ENFORCED")
                if (!is.character(eps_acceptance_mean) || length(eps_acceptance_mean) != 1 || !identical(eps_acceptance_mean, "harmonic")) {
                    stop(paste0("eps_acceptance_mean must be 'harmonic' (the harmonic mean is enforced everywhere); got: ",
                                paste(as.character(eps_acceptance_mean), collapse = ", ")))
                }
                ##
                if (!is.logical(calibrate_eps_on_sampling_chains_before_sampling) ||
                    length(calibrate_eps_on_sampling_chains_before_sampling) != 1 ||
                    is.na(calibrate_eps_on_sampling_chains_before_sampling)) {
                    stop("calibrate_eps_on_sampling_chains_before_sampling must be a single TRUE or FALSE.")
                }
                if (!is.null(repeat_sampling_with_eps_refitted_if_mean_acceptance_outside)) {
                    bounds <-  repeat_sampling_with_eps_refitted_if_mean_acceptance_outside
                    if (!is.numeric(bounds) || (length(bounds) != 2) || any(!is.finite(bounds)) ||
                        !(bounds[1] > 0) || !(bounds[2] < 1) || !(bounds[1] < bounds[2])) {
                        stop(paste0("repeat_sampling_with_eps_refitted_if_mean_acceptance_outside must be NULL ",
                                    "or two numbers, lower < upper, in (0, 1)."))
                    }
                    if (!isTRUE(calibrate_eps_on_sampling_chains_before_sampling)) {
                        stop(paste0("repeat_sampling_with_eps_refitted_if_mean_acceptance_outside needs ",
                                    "calibrate_eps_on_sampling_chains_before_sampling = TRUE (its last point is ",
                                    "one of the two points of the refit)."))
                    }
                    if (!is.numeric(n_eps_refits_and_repeated_samplings) ||
                        (length(n_eps_refits_and_repeated_samplings) != 1) ||
                        !(n_eps_refits_and_repeated_samplings >= 1) ||
                        (n_eps_refits_and_repeated_samplings != round(n_eps_refits_and_repeated_samplings))) {
                        stop("n_eps_refits_and_repeated_samplings must be one whole number >= 1.")
                    }
                }
                if (!is.logical(set_sampling_eps_to_adapt_delta_from_acceptance_of_burnin_chains) ||
                    length(set_sampling_eps_to_adapt_delta_from_acceptance_of_burnin_chains) != 1 ||
                    is.na(set_sampling_eps_to_adapt_delta_from_acceptance_of_burnin_chains)) {
                    stop(paste0("set_sampling_eps_to_adapt_delta_from_acceptance_of_burnin_chains must be a ",
                                "single TRUE or FALSE."))
                }
                if (isTRUE(set_sampling_eps_to_adapt_delta_from_acceptance_of_burnin_chains) &&
                    isTRUE(calibrate_eps_on_sampling_chains_before_sampling)) {
                    stop(paste0("Set either calibrate_eps_on_sampling_chains_before_sampling or ",
                                "set_sampling_eps_to_adapt_delta_from_acceptance_of_burnin_chains to TRUE, ",
                                "not both."))
                }
                if (!is.numeric(fraction_of_burnin_iterations_at_end_used_for_sampling_eps) ||
                    length(fraction_of_burnin_iterations_at_end_used_for_sampling_eps) != 1 ||
                    !is.finite(fraction_of_burnin_iterations_at_end_used_for_sampling_eps) ||
                    (fraction_of_burnin_iterations_at_end_used_for_sampling_eps <= 0) ||
                    (fraction_of_burnin_iterations_at_end_used_for_sampling_eps > 1)) {
                    stop("fraction_of_burnin_iterations_at_end_used_for_sampling_eps must be in (0, 1].")
                }
                ## if (!is.logical(choose_threads_and_chunks_not_given_with_Paper_1_functions) ||
                ##     length(choose_threads_and_chunks_not_given_with_Paper_1_functions) != 1 ||
                ##     is.na(choose_threads_and_chunks_not_given_with_Paper_1_functions)) {
                ##     stop(paste0("choose_threads_and_chunks_not_given_with_Paper_1_functions must be a single ",
                ##                 "TRUE or FALSE."))
                ## }
                ## if (!is.logical(time_rule_num_chunks_candidates_for_sampling) ||
                ##     length(time_rule_num_chunks_candidates_for_sampling) != 1 ||
                ##     is.na(time_rule_num_chunks_candidates_for_sampling)) {
                ##     stop("time_rule_num_chunks_candidates_for_sampling must be a single TRUE or FALSE.")
                ## }
                if (!is.logical(choose_num_chunks_and_n_threads_WCP_with_Paper_1_algorithm) ||
                    length(choose_num_chunks_and_n_threads_WCP_with_Paper_1_algorithm) != 1 ||
                    is.na(choose_num_chunks_and_n_threads_WCP_with_Paper_1_algorithm)) {
                    stop(paste0("choose_num_chunks_and_n_threads_WCP_with_Paper_1_algorithm must be a single ",
                                "TRUE or FALSE."))
                }
                if (!is.logical(tau_weight_by_p_jump) || length(tau_weight_by_p_jump) != 1 || is.na(tau_weight_by_p_jump)) {
                    stop("tau_weight_by_p_jump must be a single TRUE or FALSE.")
                }
                if (!is.logical(tau_if_manual_in_L_units) || length(tau_if_manual_in_L_units) != 1 || is.na(tau_if_manual_in_L_units)) {
                    stop("tau_if_manual_in_L_units must be a single TRUE or FALSE.")
                }
                ##
                diffusion_HMC <- if_null_then_set_to(diffusion_HMC, FALSE)
                partitioned_HMC <- if_null_then_set_to(partitioned_HMC, FALSE)
                ##
                ## ---- n_refresh is the burnin print INTERVAL: progress is reported every
                ## n_refresh iterations. NULL keeps roughly the cadence the burnin used before it
                ## was made configurable (about 20 reports across the burnin), so leaving it unset
                ## does not change how chatty a run is:
                ##
                n_refresh <- if_null_then_set_to(n_refresh, max(1L, round(n_burnin / 20)))
                ##
                if (!is.numeric(n_refresh) || length(n_refresh) != 1L || !is.finite(n_refresh) || n_refresh < 1) {
                  stop("'n_refresh' must be NULL or a single finite number >= 1 (the burnin print interval, in iterations).")
                }
                use_proposed <- if_null_then_set_to(use_proposed, TRUE)
                ##
                beta1_adam <- if_null_then_set_to(beta1_adam, 0.00)
                beta2_adam <- if_null_then_set_to(beta2_adam, 0.95)
                eps_adam <- if_null_then_set_to(eps_adam, 1e-8)
                ##
                force_autodiff <- if_null_then_set_to(force_autodiff, FALSE)
                force_PartialLog <- if_null_then_set_to(force_PartialLog, FALSE)
                multi_attempts <- if_null_then_set_to(multi_attempts, TRUE)
                ##
                if (is.null(autodiff_fallback)) {
                    autodiff_fallback <-  identical(init_object$Model_type, "latent_trait")
                }
                ##
                if (!is.logical(autodiff_fallback) || length(autodiff_fallback) != 1 || is.na(autodiff_fallback)) {
                    stop("autodiff_fallback must be NULL or a single TRUE or FALSE.")
                }
                ##
                autodiff_fallback <-  init_object$Model_type %in% c("LC_MVP", "MVP", "LC_MVOP", "MVOP", "latent_trait") && isTRUE(multi_attempts) && autodiff_fallback
                ##
                if (autodiff_fallback) {
                    cat(init_object$Model_type, " lp_grad fallback: ",
                        if (identical(init_object$Model_type, "latent_trait")) "standard -> autodiff" else "standard -> partial-log -> autodiff",
                        " (same position; no trajectory retry).\n", sep = "")
                }
                ##
                ## ---- the lp / gradient evaluation path, named when force_autodiff or force_PartialLog is TRUE
                ##      (the evaluation selector and the LC_MVP / MVP / LC_MVOP / MVOP multi-attempt functions
                ##      honour both; force_autodiff wins over force_PartialLog):
                ##
                if (isTRUE(force_autodiff) || isTRUE(force_PartialLog)) {
                      Model_type_of_this_fit <-  init_object$Model_type
                      is_latent_trait_with_multi_attempts <-  identical(Model_type_of_this_fit, "latent_trait") &&
                                                              isTRUE(multi_attempts)
                      forced_evaluation_path_text <-
                            if (identical(Model_type_of_this_fit, "Stan")) {
                                  paste0("force_autodiff / force_PartialLog have no effect for ",
                                         "Model_type = 'Stan': the Stan model's own autodiff gradient is used.")
                            } else if (is_latent_trait_with_multi_attempts) {
                                  paste0("force_autodiff / force_PartialLog have no effect for latent_trait ",
                                         "with multi_attempts = TRUE (its multi-attempt chain does not read ",
                                         "them).")
                            } else if (isTRUE(force_autodiff)) {
                                  paste0("Evaluation path forced (force_autodiff = TRUE): autodiff ",
                                         "(stan::math::var, log scale) only; NoLog and PartialLog are skipped; ",
                                         "a failed evaluation is flagged, with no further attempt.")
                            } else {
                                  paste0("Evaluation path forced (force_PartialLog = TRUE): PartialLog ",
                                         "(log scale) first; NoLog is skipped; ",
                                         if (isTRUE(autodiff_fallback)) "autodiff if it fails." else
                                               "a failed evaluation is flagged.")
                            }
                      message(colourise(forced_evaluation_path_text, "cyan"))
                }
                ##
                force_autodiff_for_metric <- if_null_then_set_to(force_autodiff_for_metric, TRUE)
                force_PartialLog_for_metric <- if_null_then_set_to(force_PartialLog_for_metric, FALSE)
                force_multi_attempts_for_metric <- if_null_then_set_to(force_multi_attempts_for_metric, FALSE)
                ##
                ##
                ## ---- (vect_type / Phi_type / inv_Phi_type are NOT defaulted here any more: the unconditional
                ##      "vect_type <- detect_vectorization_support()" and the Phi_type / inv_Phi_type defaults that used to sit
                ##      here were never used by anything. See R_fn_check_sample_math_settings.R.)
                ##
                metric_type_main <- if_null_then_set_to(metric_type_main, "Empirical")
                metric_shape_main <- if_null_then_set_to(metric_shape_main, "diag")
                ## ratio_M_main <- if_null_then_set_to(ratio_M_main, 0.25)
                ratio_M_main <- if_null_then_set_to(ratio_M_main, 0.90)
                ## interval_width_main <- if_null_then_set_to(interval_width_main, 10)
                interval_width_main <- if_null_then_set_to(interval_width_main, 1)
                metric_pooled_offdiagonal_shrinkage_resolution <- fn_resolve_metric_pooled_offdiagonal_shrinkage(
                    metric_pooled_offdiagonal_shrinkage = metric_pooled_offdiagonal_shrinkage,
                    metric_estimator = metric_estimator,
                    metric_type_main = metric_type_main,
                    metric_shape_main = metric_shape_main)
                ##
                metric_type_nuisance <- if_null_then_set_to(metric_type_nuisance, "Empirical")
                metric_shape_nuisance <- if_null_then_set_to(metric_shape_nuisance, "diag")
                ## ratio_M_nuisance <- if_null_then_set_to(ratio_M_nuisance, 0.25)
                ratio_M_nuisance <- if_null_then_set_to(ratio_M_nuisance, 0.90)
                # interval_width_nuisance <- if_null_then_set_to(interval_width_nuisance, 10)
                ## (the nuisance metric is updated at every burn-in iteration unless set otherwise)
                interval_width_nuisance <- if_null_then_set_to(interval_width_nuisance, 1)
                ##
                # max_eps_main <- if_null_then_set_to(max_eps_main, 0.75)
                # max_eps_nuisance <- if_null_then_set_to(max_eps_nuisance, 0.75)
                max_eps_main <- if_null_then_set_to(max_eps_main, 1.0)
                max_eps_nuisance <- if_null_then_set_to(max_eps_nuisance, 1.0)
                ##
                ## ---- eps warm start (see init_and_run_burnin_ChESSR): hold the HMC step-size at a
                ## deliberately LARGE 'eps_initial' for the first 'eps_initial_iter' burnin iterations,
                ## then hand it back to the usual starting step-size and adapt from there. Default is
                ## ON at eps_initial = 0.10 held for n_burnin / 10 iterations - i.e. the 50 iterations
                ## found to work at n_burnin = 500, scaled proportionally. Pass eps_initial = NULL (or
                ## NA) to switch the warm start off entirely.
                ##
                if (!is.null(eps_initial)) {
                      if (length(eps_initial) != 1L || is.na(eps_initial)) {
                            eps_initial <- NULL
                      } else if (!is.numeric(eps_initial) || !is.finite(eps_initial) || eps_initial <= 0) {
                            stop("'eps_initial' must be a single positive finite number, or NULL / NA to disable the eps warm start.")
                      }
                }
                ##
                if (!is.null(eps_initial_iter)) {
                      if (!is.numeric(eps_initial_iter) || length(eps_initial_iter) != 1L ||
                          !is.finite(eps_initial_iter) || eps_initial_iter < 0) {
                            stop("'eps_initial_iter' must be NULL (n_burnin / 10) or a single non-negative finite number of iterations.")
                      }
                }
                ##
                ## ---- learning-rate hold (see init_and_run_burnin_ChESSR): hold the ADAM learning
                ## rate at a HIGH 'learning_rate_initial' for the first 'learning_rate_initial_iter'
                ## burnin iterations - which accelerates BOTH the step-size and the path-length
                ## adaptation - then drop it to the run's normal learning rate. Default is ON at 0.10
                ## held for n_burnin / 10 iterations, i.e. the 50 iterations found to work at
                ## n_burnin = 500, scaled proportionally. Pass NULL (or NA) to switch the hold off.
                ##
                # if (!is.null(learning_rate_initial)) {
                #       if (length(learning_rate_initial) != 1L || is.na(learning_rate_initial)) {
                #             learning_rate_initial <- NULL
                #       } else if (!is.numeric(learning_rate_initial) || !is.finite(learning_rate_initial) ||
                #                  learning_rate_initial <= 0) {
                #             stop("'learning_rate_initial' must be a single positive finite number, or NULL / NA to disable the learning-rate hold.")
                #       }
                # }
                # if (is.null(learning_rate_initial)) { 
                #   learning_rate_initial <- learning_rate
                # }
                if (is.null(learning_rate_initial) ||
                    (length(learning_rate_initial) == 1 && is.na(learning_rate_initial))) {
                  learning_rate_initial <- learning_rate
                }
                ##
                if (!is.null(learning_rate_initial_iter)) {
                      if (!is.numeric(learning_rate_initial_iter) || length(learning_rate_initial_iter) != 1L ||
                          !is.finite(learning_rate_initial_iter) || learning_rate_initial_iter < 0) {
                            stop("'learning_rate_initial_iter' must be NULL (n_burnin / 10) or a single non-negative finite number of iterations.")
                      }
                }
                ##
                max_L <- if_null_then_set_to(max_L, 1024)
                # ##
                # n_nuisance_to_track <- if_null_then_set_to(n_nuisance_to_track, n_nuisance)
                ##
                Model_type <- init_object$Model_type
                ##
                Stan_model_file_path <- init_object$Stan_model_file_path
                Stan_cpp_user_header <- init_object$Stan_cpp_user_header
                Stan_cpp_flags <- init_object$Stan_cpp_flags
                stanc_args <- init_object$stanc_args
                make_args <- init_object$make_args
                ##
                ## ---- 'reorder_cols_MVP' for USER-SUPPLIED Stan models:
                ##
                ##      For the built-in models BayesMVP owns the data layout, so it knows
                ##      exactly which objects are test-indexed and can permute all of them.
                ##      For an EXTERNAL Stan model it knows nothing about the data block
                ##      beyond what is in 'Stan_data_list', so the ONLY thing it permutes is
                ##      the outcome matrix - which must therefore exist and be called 'y'.
                ##      If it is missing, reordering is REFUSED (loudly) instead of being
                ##      applied to something arbitrary:
                ##
                # reorder_cols_MVP <- if (missing(reorder_cols_MVP)) FALSE else
                #                     if_null_then_set_to(reorder_cols_MVP, FALSE)
                ## A reorder_cols_MVP that is NULL or not given is TRUE for the latent class probit models
                ## (LC_MVP, LC_MVOP) and FALSE for every other model (a value given explicitly is used as it is):
                # reorder_cols_MVP_when_not_given <-  Model_type %in% c("LC_MVP", "LC_MVOP")
                ## (and TRUE for a Stan model whose data has latent_test_order, e.g. the four-class LC-MVOP model:
                ##  its test order is then set automatically after the pre-burnin, as for the built-in models)
                reorder_cols_MVP_when_not_given <-  (Model_type %in% c("LC_MVP", "LC_MVOP")) ||
                                                    ((Model_type == "Stan") &&
                                                     !is.null(Stan_data_list$latent_test_order))
                reorder_cols_MVP <-  if (missing(reorder_cols_MVP)) reorder_cols_MVP_when_not_given else
                                     if_null_then_set_to(reorder_cols_MVP, reorder_cols_MVP_when_not_given)
                ##
                if (isTRUE(reorder_cols_MVP) && (Model_type == "Stan")) {
                      ## (BayesMVP's functions are called with their package, so that they are found in the
                      ##  fresh R process the sampler runs in, where BayesMVP is not attached)
                      if (!requireNamespace("BayesMVP", quietly = TRUE)) {
                          stop("reorder_cols_MVP requires the BayesMVP package (not installed).")
                      }
                      reorder_cols_MVP <- BayesMVP::check_reorder_cols_MVP_for_Stan(
                                              Stan_data_list = Stan_data_list)
                      ##
                      ## ---- A Stan data field 'latent_test_order' is what reordering changes for such a model
                      ##      (see the reorder block), so the model itself must declare it; otherwise the chosen
                      ##      order would be silently ignored:
                      ##
                      if (isTRUE(reorder_cols_MVP) && !is.null(Stan_data_list$latent_test_order)) {
                            model_code_for_latent_test_order_check <-
                                  paste(readLines(Stan_model_file_path, warn = FALSE), collapse = "\n")
                            if (!grepl("latent_test_order", model_code_for_latent_test_order_check,
                                       fixed = TRUE)) {
                                  stop(paste0("reorder_cols_MVP: the Stan data has 'latent_test_order', but the ",
                                              "Stan model (", Stan_model_file_path, ") does not declare it, so ",
                                              "the chosen test order would be ignored. Use the model's ",
                                              "latent-order version, or remove latent_test_order from the data."))
                            }
                      }
                }
                ##
                ## A fixed test order is only applied inside the reorder_cols_MVP block, so refuse it anywhere
                ## else rather than silently ignoring it (the ps7 filename would otherwise claim "_tp...").
                if (!is.null(test_perm_override) && !((Model_type %in% c("LC_MVP", "Stan")) && isTRUE(reorder_cols_MVP))) {
                    stop("test_perm_override is only used when reorder_cols_MVP = TRUE (Model_type LC_MVP or Stan); ",
                         "here it would be silently ignored.")
                }
                ##
                ## ---- test_order_rule_for_reorder_cols_MVP: NULL (not set) = the first rule for the built-in
                ##      models and "greedy_correlation_order" for external Stan models, whose intercepts are not
                ##      known here; a rule set explicitly is the rule that runs, or the fit stops:
                ##
                # test_order_rules_for_reorder_cols_MVP_allowed <-
                #       c("most_nearly_deterministic_test_last_then_greedy_correlation_order",
                #         "greedy_correlation_order")
                # if (is.null(test_order_rule_for_reorder_cols_MVP)) {
                #     test_order_rule_for_reorder_cols_MVP <-
                #           if (Model_type == "Stan") "greedy_correlation_order" else
                #                                     test_order_rules_for_reorder_cols_MVP_allowed[1]
                # } else
                ##
                ## ---- the third rule measures binary AND ordinal tests (tail categories from the cutpoints);
                ##      it is the default for LC_MVOP, whose ordinal tests the first rule (intercepts only)
                ##      cannot measure:
                ##
                test_order_rules_for_reorder_cols_MVP_allowed <-
                      c("most_nearly_deterministic_test_last_then_greedy_correlation_order",
                        "greedy_correlation_order",
                        paste0("test_with_largest_tail_category_distance_from_latent_mean_last_then_greedy_",
                               "correlation_order"),
                        "tests_missing_for_most_people_last_then_greedy_correlation_order")
                if (is.null(test_order_rule_for_reorder_cols_MVP)) {
                    # test_order_rule_for_reorder_cols_MVP <-
                    #       if (Model_type == "Stan") "greedy_correlation_order" else
                    #       if (Model_type == "LC_MVOP") test_order_rules_for_reorder_cols_MVP_allowed[3] else
                    #                                 test_order_rules_for_reorder_cols_MVP_allowed[1]
                    ## (a Stan model whose data has latent_test_order: the fourth rule)
                    test_order_rule_for_reorder_cols_MVP <-
                          if ((Model_type == "Stan") && !is.null(Stan_data_list$latent_test_order))
                                test_order_rules_for_reorder_cols_MVP_allowed[4] else
                          if (Model_type == "Stan") "greedy_correlation_order" else
                          if (Model_type == "LC_MVOP") test_order_rules_for_reorder_cols_MVP_allowed[3] else
                                                    test_order_rules_for_reorder_cols_MVP_allowed[1]
                } else if ((Model_type != "Stan") &&
                           identical(test_order_rule_for_reorder_cols_MVP,
                                     test_order_rules_for_reorder_cols_MVP_allowed[4])) {
                    stop(paste0("test_order_rule_for_reorder_cols_MVP = \"",
                                test_order_rules_for_reorder_cols_MVP_allowed[4], "\" reads the missing values ",
                                "of a Stan model's data y (NA or negative); it is for Model_type = \"Stan\"."))
                } else if ((Model_type == "LC_MVOP") &&
                           identical(test_order_rule_for_reorder_cols_MVP,
                                     test_order_rules_for_reorder_cols_MVP_allowed[1])) {
                    stop(paste0("test_order_rule_for_reorder_cols_MVP = \"",
                                test_order_rules_for_reorder_cols_MVP_allowed[1], "\" reads the intercepts ",
                                "only, which do not measure an ordinal test; use \"",
                                test_order_rules_for_reorder_cols_MVP_allowed[3], "\" for LC_MVOP."))
                } else if ((Model_type == "Stan") &&
                           identical(test_order_rule_for_reorder_cols_MVP,
                                     test_order_rules_for_reorder_cols_MVP_allowed[3])) {
                    stop(paste0("test_order_rule_for_reorder_cols_MVP = \"",
                                test_order_rules_for_reorder_cols_MVP_allowed[3], "\" needs the cutpoints ",
                                "and intercepts of a built-in model; use \"greedy_correlation_order\" for an ",
                                "external Stan model."))
                } else if (!is.character(test_order_rule_for_reorder_cols_MVP) ||
                           (length(test_order_rule_for_reorder_cols_MVP) != 1) ||
                           !(test_order_rule_for_reorder_cols_MVP %in%
                             test_order_rules_for_reorder_cols_MVP_allowed)) {
                    stop(paste0("test_order_rule_for_reorder_cols_MVP must be NULL or one of \"",
                                paste(test_order_rules_for_reorder_cols_MVP_allowed, collapse = "\", \""),
                                "\"."))
                } else if ((Model_type == "Stan") &&
                           (test_order_rule_for_reorder_cols_MVP ==
                            test_order_rules_for_reorder_cols_MVP_allowed[1])) {
                    stop(paste0("test_order_rule_for_reorder_cols_MVP = \"",
                                test_order_rules_for_reorder_cols_MVP_allowed[1], "\" needs the intercepts ",
                                "of the built-in LC_MVP model; use \"greedy_correlation_order\" for an ",
                                "external Stan model."))
                }
                ##
                ## ---- n_chunks_multiplier_for_PartialLog_log_scale_evaluation: a single positive whole number:
                ##
                if (!is.numeric(n_chunks_multiplier_for_PartialLog_log_scale_evaluation) ||
                    (length(n_chunks_multiplier_for_PartialLog_log_scale_evaluation) != 1) ||
                    !is.finite(n_chunks_multiplier_for_PartialLog_log_scale_evaluation) ||
                    (n_chunks_multiplier_for_PartialLog_log_scale_evaluation < 1) ||
                    (n_chunks_multiplier_for_PartialLog_log_scale_evaluation !=
                     round(n_chunks_multiplier_for_PartialLog_log_scale_evaluation))) {
                    stop(paste0("n_chunks_multiplier_for_PartialLog_log_scale_evaluation must be a single ",
                                "positive whole number (default 3)."))
                }
                n_chunks_multiplier_for_PartialLog_log_scale_evaluation <-
                      as.numeric(n_chunks_multiplier_for_PartialLog_log_scale_evaluation)
                ##
                # if (is.null(Stan_data_list$test_perm))      Stan_data_list$test_perm      <- 1:model_args_list$n_tests
                # if (is.null(Stan_data_list$n_cat_per_test)) Stan_data_list$n_cat_per_test <- model_args_list$n_cat_per_test
                ##
                ## (the rule before every model except the binary LC-MVP had to use Paper 1's functions; kept
                ##  switched off:)
##              ## ---- Burn-in within-chain threads and chunks of an external Stan model with the N / chunk_size
##              ##      data contract (reduce_sum_static), when neither n_threads_WCP_burnin nor
##              ##      num_chunks_burnin is given: chosen by fn_compute_burnin_n_threads_WCP_and_num_chunks()
##              ##      (Paper 1's burn-in rule) for the CPUs of this process and the measured M_bytes/chain.
##              ##      A value given is used as given; otherwise (a built-in model, or a Stan model without
##              ##      the contract) n_threads_WCP_burnin = 1, as before:
##              ##
##              burnin_n_threads_WCP_and_num_chunks_rule <-  NULL
##              if (is.null(n_threads_WCP_burnin)) {
##                  if (identical(Model_type, "Stan") && is.null(num_chunks_burnin) &&
##                      fn_Stan_data_has_N_and_chunk_size(Stan_data_list = Stan_data_list)) {
##                      model_so_file_for_burnin_rule <-
##                            if (is.character(init_object$model_so_file) &&
##                                (length(init_object$model_so_file) == 1) &&
##                                file.exists(init_object$model_so_file)) init_object$model_so_file else
##                            normalizePath(transform_stan_path(Stan_model_file_path))
##                      burnin_n_threads_WCP_and_num_chunks_rule <-
##                            fn_compute_burnin_n_threads_WCP_and_num_chunks_for_Stan_model(
##                                  model_so_file   = model_so_file_for_burnin_rule,
##                                  Stan_data_list  = Stan_data_list,
##                                  n_chains_burnin = n_chains_burnin,
##                                  init_object     = init_object)
##                      n_threads_WCP_burnin <-  burnin_n_threads_WCP_and_num_chunks_rule$n_threads_WCP_burnin
##                      num_chunks_burnin <-  burnin_n_threads_WCP_and_num_chunks_rule$num_chunks_burnin
##                      M_MiB_per_chain <-  burnin_n_threads_WCP_and_num_chunks_rule$M_bytes_per_chain / 1024^2
##                      how_chosen <-  if (isTRUE(burnin_n_threads_WCP_and_num_chunks_rule$reused_saved_choice)) {
##                            "reused: the choice saved for this model, N, burn-in chains and CPUs"
##                      } else {
##                            paste0("chosen in ",
##                                   round(burnin_n_threads_WCP_and_num_chunks_rule$seconds_spent_choosing, 1),
##                                   " s, not counted in the burn-in time")
##                      }
##                      message(paste0("\033[36mBurn-in: ", n_chains_burnin, " chains x ", n_threads_WCP_burnin,
##                                     " within-chain threads, ", num_chunks_burnin, " chunks (M_bytes/chain = ",
##                                     round(M_MiB_per_chain, 2), " MiB; ", how_chosen,
##                                     "; fn_compute_burnin_n_threads_WCP_and_num_chunks())\033[0m"))
##                  } else {
##                      n_threads_WCP_burnin <-  1
##                  }
##              }
                ##
                ## ---- Within-chain threads and chunks. The binary LC-MVP's are given in $sample() (Paper 1's E1
                ##      results for its N and machine). Every other model's are always chosen with Paper 1's
                ##      functions, and giving any of them in $sample() stops with an error:
                ##        - burn-in: fn_compute_burnin_n_threads_WCP_and_num_chunks() (Paper 1's burn-in
                ##          rule), from the measured M_bytes/chain and the CPUs of this process (a Stan model
                ##          with the N / chunk_size data contract:
                ##          fn_compute_burnin_n_threads_WCP_and_num_chunks_for_Stan_model(), which also times
                ##          the rule's configurations; a built-in model:
                ##          fn_compute_burnin_n_threads_WCP_and_num_chunks_for_built_in_model());
                ##        - sampling: one thread per chain, and the chunks of fn_compute_initial_N_chunks()
                ##          for the sampling chains and the CPUs of this process
                ##          (fn_compute_sampling_num_chunks_for_CPUs_of_this_process());
                ##        - a Stan model without the N / chunk_size data contract has no within-chain parallelism:
                ##          one thread per chain in both phases.
                ##      Now, for every model other than the binary LC-MVP: each of them given in $sample()
                ##      (n_threads_WCP_burnin and num_chunks_burnin together; n_threads_WCP_sampling;
                ##      num_chunks_sampling) is used as given; those not given are chosen with Paper 1's functions
                ##      when choose_threads_and_chunks_not_given_with_Paper_1_functions = TRUE (default), and must
                ##      all be given when it is FALSE. A Stan model's num_chunks_sampling is chosen with
                ##      fn_compute_sampling_num_chunks_for_Stan_model(): Paper 1's rule from the model's own
                ##      measured M_bytes/chain, timed against the powers of two around it with the sampling layout
                ##      (time_rule_num_chunks_candidates_for_sampling = TRUE, default; FALSE uses the rule's
                ##      value).
                ##      The sampling threads not given are 1 per chain.
                ##
                threads_and_chunks_given_in_sample <-  identical(Model_type, "LC_MVP")
                ##
                if (!threads_and_chunks_given_in_sample) {
                    settings_given_in_sample <-  c( n_threads_WCP_burnin    = !is.null(n_threads_WCP_burnin),
                                                    num_chunks_burnin       = !is.null(num_chunks_burnin),
                                                    n_threads_WCP_sampling  = !is.null(n_threads_WCP_sampling),
                                                    num_chunks_sampling     = !is.null(num_chunks_sampling))
                    ## if (any(settings_given_in_sample)) {
                    ##  stop(paste0("For Model_type = \"", Model_type, "\", the within-chain threads and the ",
                    ##              "chunks are always chosen with Paper 1's functions and cannot be given in ",
                    ##              "$sample() (only the binary LC-MVP's are given, from Paper 1's E1 results); ",
                    ##              "remove: ", paste(names(settings_given_in_sample)[settings_given_in_sample],
                    ##                                collapse = ", "), "."))
                    ## }
                    ## (each one given is used as given; with the automatic choice off, all four must be given)
                    ## if (!isTRUE(choose_threads_and_chunks_not_given_with_Paper_1_functions) &&
                    ##     !all(settings_given_in_sample)) {
                    ##    stop(paste0("choose_threads_and_chunks_not_given_with_Paper_1_functions = FALSE: give ",
                    ##                 "n_threads_WCP_burnin, num_chunks_burnin, n_threads_WCP_sampling and ",
                    ##                 "num_chunks_sampling in $sample(); missing: ",
                    ##                 paste(names(settings_given_in_sample)[!settings_given_in_sample],
                    ##                       collapse = ", "), "."))
                    ## }
                    if (!isTRUE(choose_num_chunks_and_n_threads_WCP_with_Paper_1_algorithm) &&
                        !all(settings_given_in_sample)) {
                        stop(paste0("choose_num_chunks_and_n_threads_WCP_with_Paper_1_algorithm = FALSE: ",
                                    "give n_threads_WCP_burnin, num_chunks_burnin, ",
                                    "n_threads_WCP_sampling and ",
                                    "num_chunks_sampling in $sample() (e.g. from NicoStan's ",
                                    "fn_compute_burnin_n_threads_WCP_and_num_chunks_for_Stan_model() and ",
                                    "fn_compute_sampling_num_chunks_for_Stan_model(), or BayesMVP's ",
                                    "fn_compute_burnin_n_threads_WCP_and_num_chunks_for_BayesMVP_model() and ",
                                    "fn_compute_sampling_num_chunks_for_BayesMVP_model()), or set it to TRUE ",
                                    "(the default); ",
                                    "missing: ",
                                    paste(names(settings_given_in_sample)[!settings_given_in_sample],
                                          collapse = ", "), "."))
                    }
                    if (settings_given_in_sample[["n_threads_WCP_burnin"]] !=
                        settings_given_in_sample[["num_chunks_burnin"]]) {
                        stop(paste0("Give both n_threads_WCP_burnin and num_chunks_burnin in $sample(), or ",
                                    "neither (Paper 1's burn-in rule chooses them together)."))
                    }
                }
                ##
                burnin_n_threads_WCP_and_num_chunks_rule <-  NULL
                sampling_num_chunks_rule <-  NULL
                ##
                if (threads_and_chunks_given_in_sample) {
                    ## (the binary LC-MVP: as given; a thread count not given is 1, as before)
                    if (is.null(n_threads_WCP_burnin)) n_threads_WCP_burnin <-  1
                    if (is.null(n_threads_WCP_sampling)) n_threads_WCP_sampling <-  1
                } else if (identical(Model_type, "Stan") &&
                           !fn_Stan_data_has_N_and_chunk_size(Stan_data_list = Stan_data_list)) {
                    ## n_threads_WCP_burnin <-  1
                    ## n_threads_WCP_sampling <-  1
                    ## (one thread per chain unless given)
                    if (is.null(n_threads_WCP_burnin)) n_threads_WCP_burnin <-  1
                    if (is.null(n_threads_WCP_sampling)) n_threads_WCP_sampling <-  1
                } else {
                    ##
                    ## ---- burn-in (Paper 1's burn-in rule):
                    ##
                    burnin_threads_and_chunks_given_in_sample <-  !is.null(n_threads_WCP_burnin) &&
                                                                  !is.null(num_chunks_burnin)
                    M_bytes_per_chain_of_model <-  NULL
                    ##
                    ## ---- the chains' own starting values, on the unconstrained scale, for the layout timings of
                    ##      the pickers below (Stan models): every timed chain starts where a chain of the
                    ##      sampler starts (where the log density is finite), never at the pickers' own small
                    ##      default values, where some models' log density is -Inf (e.g. the two-class LC-MVOP,
                    ##      whose small alternating cutpoint values leave an interval empty); built as the
                    ##      sampler's starting values are built below (R_fn_init_initial_values()), the chains
                    ##      given used in turn for as many timed chains as there are CPUs:
                    ##
                    initial_values_of_the_chains_for_the_layout_timings <-  NULL
                    if (identical(Model_type, "Stan") && is.list(init_lists_per_chain) &&
                        (length(init_lists_per_chain) > 0)) {
                          ## (in a fresh R process the init_object arrives without its BridgeStan model, which
                          ##  the parent keeps (R_fn_fresh_process.R), and the model is only made below
                          ##  (initialise_model); so the starting values here use a BridgeStan model made from
                          ##  the library and data files, as the parent does after the child; without it
                          ##  R_fn_init_initial_values() stopped with "attempt to apply non-function (in
                          ##  bs_model$param_names())", the end-to-end check of the two-class LC-MVOP after
                          bs_model_for_the_starting_values <-  init_object$bs_model
                          if (!is.function(bs_model_for_the_starting_values$param_names)) {
                                ## (the child's init_object has no json_file_path field either: the fresh
                                ##  process is given Model_args_as_Rcpp_List, whose json_file_path
                                ##  initialise_model() set to the same data file (R_fn_initialise_model.R))
                                json_file_path_for_the_starting_values <-  init_object$json_file_path
                                if (is.null(json_file_path_for_the_starting_values)) {
                                      json_file_path_for_the_starting_values <-
                                            init_object$Model_args_as_Rcpp_List$json_file_path
                                }
                                if (is.null(json_file_path_for_the_starting_values) ||
                                    !file.exists(json_file_path_for_the_starting_values)) {
                                      stop(paste0("the starting values of the layout timings need the model's ",
                                                  "data file, and init_object has no json_file_path (nor ",
                                                  "Model_args_as_Rcpp_List$json_file_path) that exists"))
                                }
                                bs_model_for_the_starting_values <-  R_fn_create_BridgeStan_model(
                                      lib  = init_object$model_so_file,
                                      # data = init_object$json_file_path,
                                      data = json_file_path_for_the_starting_values,
                                      seed = 123)
                          }
                          starting_values_given <-  R_fn_init_initial_values(
                                Model_type                                 = Model_type,
                                # bs_model                                   = init_object$bs_model,
                                bs_model                                   = bs_model_for_the_starting_values,
                                n_chains_burnin                            = length(init_lists_per_chain),
                                init_lists_per_chain                       = init_lists_per_chain,
                                sample_nuisance                            = sample_nuisance,
                                n_nuisance                                 = init_object$n_nuisance,
                                n_params_main                              = init_object$n_params_main,
                                chunk_layout_of_nuisance_vector_in_burnin  = NULL,
                                nuisance_blocks_held_by_NicoStan           =
                                      init_object$nuisance_blocks_held_by_NicoStan,
                                Stan_data_list                             = Stan_data_list)
                          n_timed_chains_at_most <-  max(n_chains_burnin, length(
                                fn_read_L3_cache_topology_of_available_CPUs()$CPU_ids_available))
                          column_of_each_timed_chain <-  ((seq_len(n_timed_chains_at_most) - 1) %%
                                                          length(init_lists_per_chain)) + 1
                          main_values_given <-  matrix(
                                starting_values_given$theta_main_vectors_all_chains_input_from_R,
                                nrow = init_object$n_params_main)
                          nuisance_values_given <-  matrix(
                                starting_values_given$theta_nuisance_vectors_all_chains_input_from_R,
                                nrow = init_object$n_nuisance)
                          initial_values_of_the_chains_for_the_layout_timings <-  list(
                                main      = main_values_given[, column_of_each_timed_chain, drop = FALSE],
                                nuisance  = nuisance_values_given[, column_of_each_timed_chain, drop = FALSE])
                    }
                    ##
                    ## ---- a Stan model with no layout given (none of the four, nor n_chains_sampling): both
                    ##      layouts chosen within one budget, budget_of_the_layout_choice_in_lp_grad_calls of the
                    ##      model's own lp_grad calls (at least fewest_seconds_of_the_budget_of_the_layout_choice),
                    ##      by fn_compute_burnin_and_sampling_layouts_of_Stan_model_within_a_budget_of_lp_grad_calls();
                    ##      budget_of_the_layout_choice_in_lp_grad_calls = NULL: each phase chosen as before, without
                    ##      a budget
                    ##
                    ##      from it): "nico_math" when NicoStan compiled its AVX version or the Stan file calls a
                    ##      nico_ function, "Stan_math" otherwise
                    ##
                    kind_of_tape_of_the_model <-  NULL
                    if (identical(Model_type, "Stan")) {
                          kind_of_tape_of_the_model <-
                                fn_kind_of_tape_of_Stan_model_from_the_nico_calls_in_its_Stan_file(
                                      Stan_model_file_path                = init_object$Stan_model_file_path,
                                      BayesMVP_AVX_version_of_Stan_model  =
                                            init_object$BayesMVP_AVX_version_of_Stan_model)
                    }
                    ##
                    layouts_within_a_budget <-  NULL
                    if (identical(Model_type, "Stan") && !is.null(budget_of_the_layout_choice_in_lp_grad_calls) &&
                        !burnin_threads_and_chunks_given_in_sample && is.null(n_threads_WCP_sampling) &&
                        is.null(num_chunks_sampling) && !n_chains_sampling_given_in_sample) {
                        model_so_file_for_burnin_rule <-
                              if (is.character(init_object$model_so_file) &&
                                  (length(init_object$model_so_file) == 1) &&
                                  file.exists(init_object$model_so_file)) init_object$model_so_file else
                              normalizePath(transform_stan_path(Stan_model_file_path))
                        layouts_within_a_budget <-
                              fn_compute_burnin_and_sampling_layouts_of_Stan_model_within_a_budget_of_lp_grad_calls(
                                    model_so_file                               = model_so_file_for_burnin_rule,
                                    Stan_data_list                              = Stan_data_list,
                                    init_object                                 = init_object,
                                    kind_of_tape                                = kind_of_tape_of_the_model,
                                    initial_values_of_the_chains                =
                                          initial_values_of_the_chains_for_the_layout_timings,
                                    ## (the memory measurement starts from the user's starting values of the
                                    ##  first chain, where the log density is finite)
                                    init_list                                   =
                                          if (is.list(init_lists_per_chain) && length(init_lists_per_chain) > 0)
                                                init_lists_per_chain[[1]] else NULL,
                                    n_chains_burnin                             = n_chains_burnin,
                                    budget_of_the_choice_in_lp_grad_calls       =
                                          budget_of_the_layout_choice_in_lp_grad_calls,
                                    fewest_seconds_of_the_budget_of_the_choice  =
                                          fewest_seconds_of_the_budget_of_the_layout_choice,
                                    reuse_saved_burnin_choice                   = TRUE)
                        burnin_n_threads_WCP_and_num_chunks_rule <-  layouts_within_a_budget$burnin
                        n_threads_WCP_burnin <-  layouts_within_a_budget$n_threads_WCP_burnin
                        num_chunks_burnin <-  layouts_within_a_budget$num_chunks_burnin
                        n_chains_sampling <-  layouts_within_a_budget$n_chains_sampling
                        n_threads_WCP_sampling <-  layouts_within_a_budget$n_threads_WCP_sampling
                        num_chunks_sampling <-  layouts_within_a_budget$num_chunks_sampling
                        M_bytes_per_chain_of_model <-  layouts_within_a_budget$M_bytes_per_chain
                        N_units_of_model <-  Stan_data_list$N
                        if (!n_superchains_given_in_sample) {
                              n_superchains <-  fn_n_superchains_of_n_chains_sampling(n_chains_sampling)
                        }
                        how_chosen <-  paste0(
                              "burn-in and sampling layouts from Paper 1's check of Paper 1's equations in ",
                              round(layouts_within_a_budget$seconds_spent_choosing_the_burnin_and_sampling_layouts,
                                    1), " s (budget ",
                              round(layouts_within_a_budget$most_seconds_spent_choosing_the_burnin_and_sampling_layouts,
                                    1), " s), not counted in the burn-in time")
                        how_chosen_sampling <-  "chosen with the burn-in layout"
                    }
                    if (is.null(layouts_within_a_budget)) {
                    if (identical(Model_type, "Stan")) {
                        model_so_file_for_burnin_rule <-
                              if (is.character(init_object$model_so_file) &&
                                  (length(init_object$model_so_file) == 1) &&
                                  file.exists(init_object$model_so_file)) init_object$model_so_file else
                              normalizePath(transform_stan_path(Stan_model_file_path))
                        if (!burnin_threads_and_chunks_given_in_sample) {
                        burnin_n_threads_WCP_and_num_chunks_rule <-
                              fn_compute_burnin_n_threads_WCP_and_num_chunks_for_Stan_model(
                                    model_so_file   = model_so_file_for_burnin_rule,
                                    Stan_data_list  = Stan_data_list,
                                    n_chains_burnin = n_chains_burnin,
                                    init_object     = init_object,
                                    kind_of_tape    = kind_of_tape_of_the_model,
                                    initial_values_of_the_chains =
                                          initial_values_of_the_chains_for_the_layout_timings,
                                    ## (the memory measurement starts from the user's starting values of the
                                    ##  first chain, never from a random point)
                                    init_list       = if (is.list(init_lists_per_chain) &&
                                                          length(init_lists_per_chain) > 0)
                                                            init_lists_per_chain[[1]] else NULL,
                                    time_Paper_1_check_of_num_chunks = TRUE)
                        how_chosen <-  if (isTRUE(burnin_n_threads_WCP_and_num_chunks_rule$reused_saved_choice)) {
                              "reused: the choice saved for this model, N, burn-in chains and CPUs"
                        } else {
                              paste0("chosen in ",
                                     round(burnin_n_threads_WCP_and_num_chunks_rule$seconds_spent_choosing, 1),
                                     " s, not counted in the burn-in time")
                        }
                        }  ## end of: if (!burnin_threads_and_chunks_given_in_sample)
                        N_units_of_model <-  Stan_data_list$N
                    } else {
                        if (!burnin_threads_and_chunks_given_in_sample) {
                        ## burnin_n_threads_WCP_and_num_chunks_rule <-
                        ##       fn_compute_burnin_n_threads_WCP_and_num_chunks_for_built_in_model(
                        ##             Model_type      = Model_type,
                        ##             init_object     = init_object,
                        ##             n_chains_burnin = n_chains_burnin)
                        ## (BayesMVP's models have manual gradients: BayesMVP's function, i.e. Paper 1's equations
                        ##  for a manual gradient, then Paper 1's check)
                        burnin_n_threads_WCP_and_num_chunks_rule <-
                              BayesMVP::fn_compute_burnin_n_threads_WCP_and_num_chunks_for_BayesMVP_model(
                                    Model_type                        = Model_type,
                                    init_object                       = init_object,
                                    n_chains_burnin                   = n_chains_burnin,
                                    time_Paper_1_check_of_num_chunks  = TRUE)
                        reused_saved_measurement <-
                              burnin_n_threads_WCP_and_num_chunks_rule$reused_saved_measurement
                        how_chosen <-  if (isTRUE(reused_saved_measurement)) {
                              "M_bytes/chain reused: the measurement saved for this model and data size"
                        } else {
                              paste0("M_bytes/chain measured in ",
                                     round(burnin_n_threads_WCP_and_num_chunks_rule$seconds_spent_choosing, 1),
                                     " s, not counted in the burn-in time")
                        }
                        }  ## end of: if (!burnin_threads_and_chunks_given_in_sample)
                        N_units_of_model <-  nrow(init_object$model_args_list$y)
                    }
                    ## n_threads_WCP_burnin <-  burnin_n_threads_WCP_and_num_chunks_rule$n_threads_WCP_burnin
                    ## num_chunks_burnin <-  burnin_n_threads_WCP_and_num_chunks_rule$num_chunks_burnin
                    ## M_bytes_per_chain_of_model <-  burnin_n_threads_WCP_and_num_chunks_rule$M_bytes_per_chain
                    if (burnin_threads_and_chunks_given_in_sample) {
                        how_chosen <-  "burn-in threads and chunks given in $sample()"
                    } else {
                        n_threads_WCP_burnin <-  burnin_n_threads_WCP_and_num_chunks_rule$n_threads_WCP_burnin
                        num_chunks_burnin <-  burnin_n_threads_WCP_and_num_chunks_rule$num_chunks_burnin
                        M_bytes_per_chain_of_model <-  burnin_n_threads_WCP_and_num_chunks_rule$M_bytes_per_chain
                    }
                    ##
                    ## ---- sampling (one thread per chain; Paper 1's chunks for the sampling chains and the CPUs
                    ##      of this process):
                    ##
                    ## (the earlier sampling block, before the sampling threads and chunks could be given and the
                    ##  rule's chunk count of a Stan model was timed:)
                    ## n_threads_WCP_sampling <-  1
                    ## sampling_num_chunks_rule <-  fn_compute_sampling_num_chunks_for_CPUs_of_this_process(
                    ##                                    M_bytes_per_chain       = M_bytes_per_chain_of_model,
                    ##                                    n_chains_sampling       = n_chains_sampling,
                    ##                                    n_threads_WCP_sampling  = n_threads_WCP_sampling,
                    ##                                    gradient_type           =
                    ##                                          if (identical(Model_type, "Stan")) "autodiff" else
                    ##                                          "manual")
                    ## num_chunks_sampling <-  sampling_num_chunks_rule$N_chunks
                    ## if (identical(Model_type, "Stan")) {
                    ##     ## (reduce_sum_static halves the range of units until each piece holds at most
                    ##  ##  chunk_size units, so the data are cut into a power of two of pieces: the chunk count
                    ##     ##  is rounded up to the next power of two, i.e. to the number of pieces actually used)
                    ##     num_chunks_sampling <-  min(2^ceiling(log2(num_chunks_sampling)),
                    ##                                 2^floor(log2(N_units_of_model)))
                    ## } else {
                    ##  ## (a built-in model: at most one chunk per SIMD vector of observations, for both phases)
                    ##     vect_type_of_model <-
                    ##           as.character(init_object$Model_args_as_Rcpp_List$Model_args_strings[1])
                    ##     vec_size_of_model <-  if (identical(vect_type_of_model, "AVX512")) 8 else
                    ##                           if (identical(vect_type_of_model, "AVX2")) 4 else
                    ##                           if (identical(vect_type_of_model, "AVX")) 2 else 1
                    ##     most_chunks_of_model <-  max(1, floor(N_units_of_model / vec_size_of_model))
                    ##     num_chunks_burnin <-  min(num_chunks_burnin, most_chunks_of_model)
                    ##     num_chunks_sampling <-  min(num_chunks_sampling, most_chunks_of_model)
                    ## }
                    ##
                    ## ---- sampling: the threads given (1 per chain when not given); the chunks given, or
                    ##      Paper 1's rule from the model's own measured M_bytes/chain (a Stan model: timed
                    ##      against the powers of two around it with the sampling layout,
                    ##      fn_compute_sampling_num_chunks_for_Stan_model()):
                    ##
                    ## (the sampling threads given in $sample(), before a thread count not given is set to 1)
                    sampling_threads_given_in_sample <-  !is.null(n_threads_WCP_sampling)
                    if (is.null(n_threads_WCP_sampling)) n_threads_WCP_sampling <-  1
                    sampling_num_chunks_given_in_sample <-  !is.null(num_chunks_sampling)
                    if (sampling_num_chunks_given_in_sample) {
                        how_chosen_sampling <-  "sampling chunks given in $sample()"
                    } else if (!sampling_threads_given_in_sample) {
                        ## (Paper 1's check of the whole sampling layout, WCP and chunking, chunking alone being
                        ##  its 1 thread per chain; the chains given in $sample() are kept)
                        n_chains_sampling_of_check <-  NULL
                        if (n_chains_sampling_given_in_sample) n_chains_sampling_of_check <-  n_chains_sampling
                        if (identical(Model_type, "Stan")) {
                              sampling_layout <-
                              fn_compute_sampling_n_chains_n_threads_WCP_and_num_chunks_for_Stan_model(
                                    model_so_file      = model_so_file_for_burnin_rule,
                                    Stan_data_list     = Stan_data_list,
                                    M_bytes_per_chain  = M_bytes_per_chain_of_model,
                                    n_chains_sampling  = n_chains_sampling_of_check,
                                    init_object        = init_object,
                                    kind_of_tape       = kind_of_tape_of_the_model,
                                    initial_values_of_the_chains =
                                          initial_values_of_the_chains_for_the_layout_timings,
                                    init_list          = if (is.list(init_lists_per_chain) &&
                                                             length(init_lists_per_chain) > 0)
                                                               init_lists_per_chain[[1]] else NULL)
                        } else {
                        sampling_layout <-
                        BayesMVP::fn_compute_sampling_n_chains_n_threads_WCP_and_num_chunks_for_BayesMVP_model(
                              Model_type         = Model_type,
                              init_object        = init_object,
                              n_chains_sampling  = n_chains_sampling_of_check)
                        }
                        n_chains_sampling <-  sampling_layout$n_chains_sampling
                        n_threads_WCP_sampling <-  sampling_layout$n_threads_WCP_sampling
                        num_chunks_sampling <-  sampling_layout$num_chunks_sampling
                        M_bytes_per_chain_of_model <-  sampling_layout$M_bytes_per_chain
                        if (!n_superchains_given_in_sample) {
                              n_superchains <-  fn_n_superchains_of_n_chains_sampling(n_chains_sampling)
                        }
                        how_chosen_sampling <-  paste0(
                              "sampling layout from Paper 1's check (WCP and chunking) of Paper 1's equations (",
                              sampling_layout$n_chains_sampling_from_equations, " chains x ",
                              sampling_layout$n_threads_WCP_sampling_from_equations, " threads x ",
                              sampling_layout$num_chunks_sampling_from_equations, " chunks) in ",
                              round(sampling_layout$seconds_spent_choosing, 1),
                              " s, not counted in the burn-in time")
                    } else if (identical(Model_type, "Stan")) {
                        sampling_num_chunks_rule <-  fn_compute_sampling_num_chunks_for_Stan_model(
                              model_so_file                     = model_so_file_for_burnin_rule,
                              Stan_data_list                    = Stan_data_list,
                              n_chains_sampling                 = n_chains_sampling,
                              n_threads_WCP_sampling            = n_threads_WCP_sampling,
                              M_bytes_per_chain                 = M_bytes_per_chain_of_model,
                              init_object                       = init_object,
                              kind_of_tape                      = kind_of_tape_of_the_model,
                              time_Paper_1_check_of_num_chunks  = TRUE)
                        num_chunks_sampling <-  sampling_num_chunks_rule$num_chunks_sampling
                        M_bytes_per_chain_of_model <-  sampling_num_chunks_rule$M_bytes_per_chain
                        how_chosen_sampling <-  if (isTRUE(sampling_num_chunks_rule$reused_saved_choice)) {
                              paste0("sampling chunks reused: the choice saved for this model, N, chains, ",
                                     "threads and CPUs")
                        } else if (is.null(sampling_num_chunks_rule$timings)) {
                              paste0("sampling chunks from Paper 1's rule (",
                                     sampling_num_chunks_rule$num_chunks_sampling_from_rule, "), not timed")
                        } else {
                              paste0("sampling chunks from Paper 1's check of Paper 1's rule (",
                                     sampling_num_chunks_rule$num_chunks_sampling_from_rule, ") in ",
                                     round(sampling_num_chunks_rule$seconds_spent_choosing, 1),
                                     " s, not counted in the burn-in time; within 1% of the fastest: ",
                                     paste(sampling_num_chunks_rule$num_chunks_within_tie_of_fastest,
                                           collapse = ", "))
                        }
                    } else {
                        ## if (is.null(M_bytes_per_chain_of_model)) {
                        ##     M_bytes_per_chain_of_model <-  fn_M_bytes_per_chain_of_built_in_model(
                        ##                                          Model_type   = Model_type,
                        ##                                          init_object  = init_object)$M_bytes_per_chain
                        ## }
                        ## sampling_num_chunks_rule <-  fn_compute_sampling_num_chunks_for_CPUs_of_this_process(
                        ##                                  M_bytes_per_chain       = M_bytes_per_chain_of_model,
                        ##                                    n_chains_sampling       = n_chains_sampling,
                        ##                                    n_threads_WCP_sampling  = n_threads_WCP_sampling,
                        ##                                    gradient_type           = "manual")
                        ## num_chunks_sampling <-  sampling_num_chunks_rule$N_chunks
                        ## how_chosen_sampling <-  "sampling chunks from Paper 1's rule"
                        ## (BayesMVP's models have manual gradients: BayesMVP's function, i.e. Paper 1's equations
                        ##  for a manual gradient, then Paper 1's check)
                        sampling_num_chunks_rule <-  BayesMVP::fn_compute_sampling_num_chunks_for_BayesMVP_model(
                              Model_type                        = Model_type,
                              init_object                       = init_object,
                              n_chains_sampling                 = n_chains_sampling,
                              n_threads_WCP_sampling            = n_threads_WCP_sampling,
                              time_Paper_1_check_of_num_chunks  = TRUE)
                        num_chunks_sampling <-  sampling_num_chunks_rule$num_chunks_sampling
                        M_bytes_per_chain_of_model <-  sampling_num_chunks_rule$M_bytes_per_chain
                        how_chosen_sampling <-  paste0(
                              "sampling chunks from Paper 1's check of Paper 1's equations (",
                              sampling_num_chunks_rule$num_chunks_sampling_from_Paper_1_equations, ") in ",
                              round(sampling_num_chunks_rule$seconds_spent_choosing, 1),
                              " s, not counted in the burn-in time")
                    }
                    }  ## end of: if (is.null(layouts_within_a_budget))
                    ##
                    message(colourise(paste0("Burn-in: ", n_chains_burnin, " chains x ", n_threads_WCP_burnin,
                                             " within-chain threads, ", num_chunks_burnin, " chunks; sampling: ",
                                             n_chains_sampling, " chains x ", n_threads_WCP_sampling,
                                             " thread, ", num_chunks_sampling, " chunks (M_bytes/chain = ",
                                             if (is.null(M_bytes_per_chain_of_model)) "not measured" else
                                                   paste0(round(M_bytes_per_chain_of_model / 1024^2, 2), " MiB"),
                                             "; ", how_chosen, "; ", how_chosen_sampling,
                                             "; Paper 1's functions: ",
                                             "fn_compute_burnin_n_threads_WCP_and_num_chunks() and ",
                                             "fn_compute_initial_N_chunks())"), "cyan"))
                }
                ##
                ## ---- Explicit phase chunk counts for external Stan models use the model's N / chunk_size data contract:
                ##
                stan_chunk_size_sampling <- NULL
                if (Model_type == "Stan" && (!is.null(num_chunks_burnin) || !is.null(num_chunks_sampling))) {
                    for (chunk_count in list(num_chunks_burnin, num_chunks_sampling)) {
                        if (is.null(chunk_count)) next
                        if (!is.numeric(chunk_count) || length(chunk_count) != 1L || !is.finite(chunk_count) ||
                            chunk_count < 1 || chunk_count != floor(chunk_count) || chunk_count > .Machine$integer.max) {
                            stop("Stan phase chunk counts must be NULL or positive integers within Stan's integer range.")
                        }
                    }
                    ## (a model without reduce_sum - no data fields N and chunk_size - has nothing to chunk: an
                    ##  explicit N_chunks = 1 for both phases is that model as it is; any other count stops)
                    model_has_the_chunk_data <-  all(vapply(c("N", "chunk_size"), function(data_name) {
                          data_value <-  Stan_data_list[[data_name]]
                          is.numeric(data_value) && (length(data_value) == 1) && is.finite(data_value) &&
                                (data_value >= 1) && (data_value == floor(data_value)) &&
                                (data_value <= .Machine$integer.max)
                    }, logical(1)))
                    every_given_chunk_count_is_1 <-  all(unlist(list(num_chunks_burnin, num_chunks_sampling)) == 1)
                    if (!model_has_the_chunk_data && !every_given_chunk_count_is_1) {
                        stop(paste0("Explicit Stan phase chunk counts above 1 require positive integer data fields ",
                                    "N and chunk_size (a model with reduce_sum); this model has none, so only ",
                                    "num_chunks_burnin = 1 and num_chunks_sampling = 1 (or NULL) are possible."))
                    }
                    # for (data_name in c("N", "chunk_size")) {
                    #     data_value <- Stan_data_list[[data_name]]
                    #     if (!is.numeric(data_value) || length(data_value) != 1L || !is.finite(data_value) ||
                    #         data_value < 1 || data_value != floor(data_value) || data_value > .Machine$integer.max) {
                    #         stop("Explicit Stan phase chunk counts require positive integer data fields N and chunk_size.")
                    #     }
                    # }
                    if (model_has_the_chunk_data) {
                        ## NULL retains the supplied data chunk_size independently for that phase.
                        stan_chunk_size_sampling <- if (is.null(num_chunks_sampling)) Stan_data_list$chunk_size else
                            as.integer(ceiling(Stan_data_list$N / num_chunks_sampling))
                        if (!is.null(num_chunks_burnin)) {
                            Stan_data_list$chunk_size <- as.integer(ceiling(Stan_data_list$N / num_chunks_burnin))
                        }
                    }
                }
                ##
                if (is.null(num_chunks_burnin)) { 
                  num_chunks_burnin <- model_args_list$num_chunks
                } else { 
                  model_args_list$num_chunks <- num_chunks_burnin
                }
                ##
                ## ---- Initialise model:
                ##
                ## model_args_list$num_chunks <- 4
                ##
                ## ---- Reuse of the supplied initialisation object:
                ##
                ##      The init_object passed in is normally built by initialise_model() for the same model and data, and
                ##      rebuilding it here repeats the BridgeStan set-up (JSON data file, compile check, dyn.load and model
                ##      construction) for the same result. The supplied object is kept, and initialise_model() skipped, ONLY
                ##      when a rebuild could not give a different object:
                ##        - it holds a usable BridgeStan model: a StanModel whose shared library is loaded in this R session
                ##          and whose unconstrained dimension equals the stored n_params (Stan models) or n_params_main
                ##          (built-in models, whose .stan skeleton holds only the main parameters); the fresh-process path
                ##          passes an init_object without a bs_model, so it always rebuilds;
                ##        - the same stream (it names the JSON data file; stream = NULL makes initialise_model() draw a new
                ##          one, so it always rebuilds), the same sample_nuisance and the same n_nuisance_override;
                ##        - built-in models: init_hard_coded_model() applied to model_args_list (after the num_chunks patch
                ##          above) gives exactly the stored model_args_list and Model_args_as_Rcpp_List, and the stored
                ##          Stan_data_list is the one derived from it, so a different num_chunks, prior, bound or threshold
                ##          rebuilds;
                ##        - Stan models: identical model_args_list and Stan_data_list (after the chunk_size patch above).
                ##      Any error in these checks means a rebuild. The re-initialisation on column-permuted data after the
                ##      pre-burn-in test reordering (further below) always rebuilds.
                ##
                ## ---- a Stan model whose nuisance blocks are held by NicoStan (nuisance_params,
                ##      nico_reduce_sum): its model file is the nico_reduce_sum version, compiled
                ##      against NicoStan's private runtime with the chunk-local bookkeeping and the nuisance
                ##      block (the two options, set here too, as this may be a fresh R process); the size of
                ##      its nuisance block was worked out at its initialisation (used when n_nuisance_override
                ##      is not given), and the writer's record of the blocks gives their initial values their
                ##      order; both are kept through a rebuild:
                ##
                nuisance_blocks_held_by_NicoStan_of_the_model <-  NULL
                if (is.list(init_object)) {
                      nuisance_blocks_held_by_NicoStan_of_the_model <-
                            init_object$nuisance_blocks_held_by_NicoStan
                }
                if (!is.null(nuisance_blocks_held_by_NicoStan_of_the_model)) {
                      options(NicoStan.reduce_sum_lean_slicing = TRUE, NicoStan.reduce_sum_nuisance_block = TRUE)
                      if (is.null(n_nuisance_override)) {
                            n_nuisance_override <-  init_object$n_nuisance_override
                      }
                }
                ##
                init_reuse_supplied_object <-  FALSE
                ##
                init_reuse_bs_model <-  if (is.list(init_object)) init_object$bs_model else NULL
                ##
                if (inherits(x = init_reuse_bs_model, what = "StanModel")) {
                      ##
                      init_reuse_supplied_object <-  tryCatch({
                            ##
                            init_reuse_bs_lib_name <-  init_reuse_bs_model$.__enclos_env__$private$lib_name
                            ##
                            init_reuse_bs_model_usable <-  is.character(init_reuse_bs_lib_name) &&
                                                           (length(init_reuse_bs_lib_name) == 1) &&
                                                           is.loaded("bs_model_construct_R", PACKAGE = init_reuse_bs_lib_name) &&
                                                           identical(as.numeric(init_reuse_bs_model$param_unc_num()),
                                                                     as.numeric(if (Model_type == "Stan") init_object$n_params else init_object$n_params_main)) &&
                                                           ## (the C++ sampler re-opens the JSON data file and the model .so by path, and a
                                                           ##  rebuild would re-write a JSON file removed since the caller's initialisation):
                                                           is.character(init_object$json_file_path) && (length(init_object$json_file_path) == 1) &&
                                                           is.character(init_object$model_so_file) && (length(init_object$model_so_file) == 1) &&
                                                           file.exists(init_object$json_file_path) &&
                                                           file.exists(init_object$model_so_file)
                            ##
                            init_reuse_same_settings <-  identical(stream, init_object$stream) &&
                                                         identical(sample_nuisance, init_object$sample_nuisance) &&
                                                         identical(n_nuisance_override, init_object$n_nuisance_override)
                            ##
                            if (!(isTRUE(init_reuse_bs_model_usable) && isTRUE(init_reuse_same_settings))) {

                                  FALSE

                            } else if (Model_type == "Stan") {

                                  identical(model_args_list, init_object$model_args_list) &&
                                  identical(Stan_data_list, init_object$Stan_data_list)

                            } else {

                                  ## (the same derivation as in initialise_model(), without the BridgeStan step):
                                  init_reuse_outs <-  BayesMVP::init_hard_coded_model( Model_type = Model_type,
                                                                             model_args_list = model_args_list)
                                  ##
                                  init_reuse_Model_args_as_Rcpp_List <-  init_reuse_outs$Model_args_as_Rcpp_List
                                  init_reuse_Model_args_as_Rcpp_List$json_file_path <-  init_object$json_file_path
                                  init_reuse_Model_args_as_Rcpp_List$model_so_file <-  init_object$model_so_file
                                  ##
                                  identical(init_reuse_outs$model_args_list, init_object$model_args_list) &&
                                  identical(init_reuse_Model_args_as_Rcpp_List, init_object$Model_args_as_Rcpp_List) &&
                                  identical(BayesMVP::make_Stan_data_list_for_internal_models(
                                                                                     Model_type = Model_type,
                                                                                     model_args_list = init_reuse_outs$model_args_list),
                                            init_object$Stan_data_list)

                            }
                            ##
                      }, error = function(e) FALSE)
                      ##
                      init_reuse_supplied_object <-  isTRUE(init_reuse_supplied_object)
                }
                ##
                if (init_reuse_supplied_object) {
                      ##
                      message(colourise(paste0("Reusing the supplied initialisation object (same model, data, model_args_list, ",
                                               "stream and nuisance settings); initialise_model() skipped."), "cyan"))
                      ##
                } else {
                ##
                init_object <- initialise_model( Model_type =  Model_type,
                                                 ##
                                                 stream = stream,
                                                 ##
                                                 sample_nuisance = sample_nuisance,
                                                 n_nuisance_override = n_nuisance_override,
                                                 ##
                                                 model_args_list = model_args_list,
                                                 ##
                                                 compile = init_object$compile,
                                                 force_recompile = init_object$force_recompile,
                                                 ##
                                                 cmdstanr_model_fit_obj = NULL,
                                                 ##
                                                 Stan_data_list = Stan_data_list,
                                                 ##
                                                 Stan_model_file_path = Stan_model_file_path,
                                                 Stan_cpp_user_header = Stan_cpp_user_header,
                                                 Stan_cpp_flags = Stan_cpp_flags,
                                                 stanc_args = stanc_args,
                                                 make_args = make_args)
                ##
                ## (the writer's record of the nuisance blocks held by NicoStan, kept through the rebuild)
                init_object$nuisance_blocks_held_by_NicoStan <-  nuisance_blocks_held_by_NicoStan_of_the_model
                } ## end of "else" (rebuild) of the "if (init_reuse_supplied_object)"
                ##
                ##
                ## ---- Built-in models: any vect_type / Phi_type / inv_Phi_type supplied to $sample() must equal the value the
                ##      (re-)initialised model resolved from model_args_list (and BayesMVP validated); otherwise stop, since
                ##      $sample() cannot change it. Prints the effective values when any were supplied:
                ##
                fn_check_sample_math_settings_match_native_model_args( Model_type                = Model_type,
                                                                       model_args_list_in_effect = init_object$model_args_list,
                                                                       vect_type                 = vect_type,
                                                                       Phi_type                  = Phi_type,
                                                                       inv_Phi_type              = inv_Phi_type)
                ##
                y <- init_object$model_args_list$y ; y
                N <- init_object$model_args_list$N ; N
                n_class <- init_object$model_args_list$n_class ; n_class
                ##
                ##
                Model_args_as_Rcpp_List <- init_object$Model_args_as_Rcpp_List ; Model_args_as_Rcpp_List
                Model_args_as_Rcpp_List$autodiff_fallback <-  autodiff_fallback
                Model_args_as_Rcpp_List$n_chunks_multiplier_for_PartialLog_log_scale_evaluation <-
                      n_chunks_multiplier_for_PartialLog_log_scale_evaluation
                model_args_list         <- init_object$model_args_list
                ##
                # Model_args_as_Rcpp_List$Model_args_ints[4, 1] <- 4
                # str(Model_args_as_Rcpp_List)
                ##
                n_nuisance <- init_object$n_nuisance ; n_nuisance
                n_params_main <- init_object$n_params_main ; n_params_main
                ##
                ## ---- Models with NO nuisance block (n_nuisance == 0): there are no
                ## nuisance coordinates to partition off or diffuse, so the COMPLETE
                ## unconstrained vector is sampled jointly as the main block. Main-block
                ## settings (metric, tau, eps, adapt_delta, n_iter, seeds, ...) are
                ## unchanged. This only triggers when n_nuisance resolves to 0 (e.g. a
                ## Stan model run with sample_nuisance = FALSE, or a nuisance declaration
                ## that evaluates to zero length); built-in models always have n_nuisance > 0.
                ##
                ## A Stan model can have an empty nuisance block for SOME data and not others - e.g. a
                ## declaration like "array[prior_only == 1 ? 0 : N] row_vector[n_tests] u_raw;", which
                ## is exactly what a prior-only run produces. So this has to be resolved from the
                ## realised n_nuisance, not from the settings the caller wrote.
                ##
                if (n_nuisance == 0 && (isTRUE(partitioned_HMC) || isTRUE(diffusion_HMC) || isTRUE(sample_nuisance))) {
                    message( "n_nuisance = 0: no nuisance coordinates to partition or diffuse - ",
                             "disabling partitioned / diffusion HMC and nuisance updates; ",
                             "the complete parameter vector is sampled jointly as the main block.")
                    ##
                    sample_nuisance <- FALSE
                    partitioned_HMC <- FALSE
                    diffusion_HMC <- FALSE
                    ##
                    ## ---- The joint-diffusion integrators are only defined WITH a nuisance block, so
                    ## disabling diffusion_HMC while leaving diffusion_HMC_integrator = "flow_kick_flow"
                    ## leaves the two contradicting each other, and the burnin then refuses to start
                    ## ("flow_kick_flow requires diffusion_HMC = TRUE ... and a nonempty nuisance
                    ## block"). Fall back to the default ordering alongside the disable:
                    ##
                    if (!identical(diffusion_HMC_integrator, "kick_flow_kick")) {
                          message( "n_nuisance = 0: diffusion_HMC_integrator '", diffusion_HMC_integrator,
                                   "' is only defined for joint diffusion over a nuisance block - ",
                                   "falling back to 'kick_flow_kick' for this run.")
                          ##
                          diffusion_HMC_integrator <- "kick_flow_kick"
                    }
                }
                ##
                ## the caller's choice (block NAMES or "none"), kept so a re-initialisation below
                ## re-resolves from the SAME instruction rather than from an already-resolved value:
                nuisance_to_track_user <- nuisance_to_track
                ##
                positions_of_the_nuisance_values_to_keep <- fn_positions_of_the_nuisance_values_to_track(
                      nuisance_to_track = nuisance_to_track_user,
                      nuisance_blocks   = init_object$nuisance_blocks_held_by_NicoStan,
                      Stan_data_list    = Stan_data_list,
                      n_nuisance        = n_nuisance)
                ##
                ## (the sampler stores the values at these positions of every draw, in memory, never on disk;
                ##  its own count is the length of the positions)
                n_nuisance_to_track <- length(positions_of_the_nuisance_values_to_keep)
                if (n_nuisance_to_track > 0L) {
                      Rcpp_fn_set_the_positions_of_the_nuisance_values_kept_in_the_trace(
                            positions_from_0 = as.integer(positions_of_the_nuisance_values_to_keep))
                }
                # ##
                # if (is.null(n_nuisance_to_track)) {
                #   if (sample_nuisance == FALSE) {
                #     n_nuisance <- 1 # dummy (or 9? try both)
                #     n_nuisance_to_track <- 1 # dummy (or 9? try both)
                #   } else { 
                #     n_nuisance_to_track <- n_nuisance
                #   }
                # }
                n_params <- n_nuisance + n_params_main
                ##
                print(paste("n_params = ", n_params))
                print(paste("n_params_main = ",  n_params_main))
                print(paste("n_nuisance = ",  n_nuisance))
                ##
                Model_args_as_Rcpp_List$n_nuisance <- n_nuisance
                bs_model <- init_object$bs_model
                ##
                index_nuisance <- seq_len(n_nuisance)
                index_main <- (1 + n_nuisance):n_params
                ##
                ## ---- Process initial values BEFORE burnin:
                ##
                ## ---- the chunk layout of the nuisance vector in the burn-in (built-in models): the
                ##      burn-in's number of chunks and vectorisation, as the C++ reads them:
                chunk_layout_of_nuisance_vector_in_burnin <-  NULL
                if (Model_type != "Stan") {
                      chunk_layout_of_nuisance_vector_in_burnin <-
                            list( N = nrow(y),
                                  n_tests = ncol(y),
                                  n_chunks = Model_args_as_Rcpp_List$Model_args_ints[4],
                                  vect_type = Model_args_as_Rcpp_List$Model_args_strings[1])
                }
                outs <- R_fn_init_initial_values( Model_type = Model_type,
                                                  bs_model = bs_model,
                                                  ##
                                                  n_chains_burnin = n_chains_burnin,
                                                  init_lists_per_chain = init_lists_per_chain,
                                                  ##
                                                  sample_nuisance = sample_nuisance,
                                                  n_nuisance = n_nuisance,
                                                  n_params_main = n_params_main,
                                                  ##
                                                  chunk_layout_of_nuisance_vector_in_burnin =
                                                        chunk_layout_of_nuisance_vector_in_burnin,
                                                  ##
                                                  nuisance_blocks_held_by_NicoStan =
                                                        init_object$nuisance_blocks_held_by_NicoStan,
                                                  Stan_data_list = Stan_data_list)
                ##
                inits_unconstrained_vec_per_chain <- outs$inits_unconstrained_vec_per_chain
                ##
                theta_nuisance_vectors_all_chains_input_from_R <- outs$theta_nuisance_vectors_all_chains_input_from_R
                theta_main_vectors_all_chains_input_from_R <- outs$theta_main_vectors_all_chains_input_from_R
                ##
                n_chains_burnin <- outs$n_chains_burnin
                init_lists_per_chain <- outs$init_lists_per_chain
                ##
                if (Model_type != "Stan")  {
                    n_params_main <- Model_args_as_Rcpp_List$n_params_main # <- n_params_main
                    n_nuisance <-    Model_args_as_Rcpp_List$n_nuisance #<- n_nuisance
                } else if  (Model_type == "Stan")   {
                    Model_args_as_Rcpp_List$n_params_main <- n_params_main
                    Model_args_as_Rcpp_List$n_nuisance <- n_nuisance
                }
                if (Model_type == "Stan") {
                  Model_args_as_Rcpp_List$model_so_file <-    init_object$model_so_file
                  Model_args_as_Rcpp_List$json_file_path <-   init_object$json_file_path
                } else {
                  Model_args_as_Rcpp_List$model_so_file <-  "none" #   init_object$dummy_model_so_file
                  Model_args_as_Rcpp_List$json_file_path <- "none" #  init_object$dummy_json_file_path
                }
                ##
                if (Model_type == "Stan") {
                   ## a model with NO data gets "{}" (not cmdstanr's empty ARRAY). Written ATOMICALLY (temporary file +
                   ## rename; the shared stan_data folder is read by concurrent fits - see fn_write_file_atomically):
                   fn_write_stan_json_atomically( stan_data_list = Stan_data_list,
                                                  json_file_path = Model_args_as_Rcpp_List$json_file_path)
                } else { 
                   # cmdstanr::write_stan_json(data = Stan_data_list, file = Model_args_as_Rcpp_List$dummy_json_file_path)  
                }
                ##
                print(  Model_args_as_Rcpp_List$model_so_file)
                print(  Model_args_as_Rcpp_List$json_file_path)
                
                ## Built-in default: n_chains_burnin TBB threads plus each chain's OpenMP WCP team.
                ## Stan models always use the full shared TBB budget; built-in models can request it with explicit FALSE.
                RcppParallel::setThreadOptions(numThreads = if (burnin_TBB_pool_equals_n_chains) n_chains_burnin else n_threads_WCP_burnin * n_chains_burnin);
                
                ## One maintained burn-in driver selects its criterion through burnin_algorithm.
                fn_burnin <- init_and_run_burnin_ChESSR
                ##
                time_pre_burnin <- 0.0
                ##
                ## ---- Second half of the 'reorder_cols_MVP' gate for external Stan models:
##
                ##      the ordering is chosen from the model's estimated correlation
                ##      matrix, so the model must expose one named "Omega" whose dimension
                ##      matches the number of columns of 'y'. Checked HERE (before the
                ##      pre-burnin) so that a model which cannot be reordered does not waste
                ##      a pre-burnin first:
                ##
                Omega_dims <- NULL
                ##
                if (isTRUE(reorder_cols_MVP) && (Model_type == "Stan")) {

                      Omega_dims <- BayesMVP::infer_Omega_dims_from_bs_model(bs_model = bs_model)
                      n_tests_y  <- ncol(Stan_data_list$y)
                      ##
                      if (Omega_dims$n_found == 0) {
                            big_warning_banner(
                                  title = "reorder_cols_MVP: no 'Omega' parameter found in the Stan model",
                                  body = c("The column ordering is chosen from the estimated correlation matrix, which is",
                                           "located by NAME: BayesMVP looks for a parameter / transformed parameter /",
                                           "generated quantity called 'Omega' (e.g. 'Omega[i,j]' or 'Omega[c,i,j]').",
                                           "The model declares no such quantity.",
                                           "",
                                           "==> reorder_cols_MVP has been DISABLED for this run; sampling continues with",
                                           "    the data in its original column order."))
                            reorder_cols_MVP <- FALSE
                      } else if (!identical(as.integer(Omega_dims$n_tests), as.integer(n_tests_y))) {
                            big_warning_banner(
                                  title = "reorder_cols_MVP: 'Omega' does not match the number of columns of 'y'",
                                  body = c(paste0("Omega in the Stan model is ", Omega_dims$n_tests, " x ", Omega_dims$n_tests,
                                                  ", but 'y' has ", n_tests_y, " columns."),
                                           "Reordering permutes the columns of 'y' using the ordering of Omega, so the two",
                                           "must refer to the same set of tests/outcomes, in the same order.",
                                           "",
                                           "==> reorder_cols_MVP has been DISABLED for this run; sampling continues with",
                                           "    the data in its original column order."))
                            reorder_cols_MVP <- FALSE
                      }

                }
                ##
                test_perm     <- NULL
                test_inv_perm <- NULL
                ##
                ## Default = identity for any model that didn't reorder:
                if (is.null(test_perm)) {
                  ## for external Stan models the test/outcome count is only knowable from
                  ## the user-supplied outcome matrix 'y':
                  n_tests_tmp <- if (Model_type == "Stan") {
                                    tryCatch(ncol(Stan_data_list$y), error = function(e) NULL)
                                 } else {
                                    tryCatch(init_object$model_args_list$n_tests, error = function(e) NULL)
                                 }
                  if (!is.null(n_tests_tmp)) {
                    test_perm     <- 1:n_tests_tmp
                    test_inv_perm <- 1:n_tests_tmp
                  }
                }
                ##
                # reorder_cols_MVP <- TRUE
                ##
                # if ((Model_type %in% c("LC_MVP", "Stan")) &&
                #     (reorder_cols_MVP == TRUE)) {
                ##
                ## ---- LC_MVOP too: its re-layout block and its outputs (create_summary_and_traces) handle any
                ##      test order:
                ##
                if ((Model_type %in% c("LC_MVP", "LC_MVOP", "Stan")) &&
                    (reorder_cols_MVP == TRUE)) {

                    ## ---- Number of tests + number of latent classes: known from
                    ## 'model_args_list' for the built-in models; for an external Stan model
                    ## n_tests is the column count of the user-supplied 'y' and n_class is
                    ## read off the model's own "Omega" parameter names:
                    if (Model_type == "Stan") {
                          n_tests <- ncol(Stan_data_list$y)
                          ##
                          if (is.null(Omega_dims)) {
                            Omega_dims <- BayesMVP::infer_Omega_dims_from_bs_model(bs_model = bs_model)
                          }
                          n_class_for_reorder <- if (is.null(Omega_dims$n_class)) 1L else Omega_dims$n_class
                    } else {
                          n_tests <- init_object$model_args_list$n_tests
                          n_class_for_reorder <- n_class
                    }
                    ##
                    theta_main_orig <- theta_main_vectors_all_chains_input_from_R
                    theta_nuisance_orig <- theta_nuisance_vectors_all_chains_input_from_R
                    ##
                    pre_burnin_params <- list()
                    ##
                    # pre_burnin_params$n_burnin <- 250
                    # pre_burnin_params$n_adapt  <- 200
                    # ##
                    # pre_burnin_params$gap <- 100
                    # ##
                    # pre_burnin_params$LR_main <- 0.10
                    # pre_burnin_params$LR_us <- 0.10
                    # ##
                    # pre_burnin_params$manual_tau <- TRUE
                    # pre_burnin_params$tau_if_manual <- c(3.0, 3.0)
                    # ##
                    # int <- 75
                    # pre_burnin_params$clip_iter <- 50
                    # pre_burnin_params$clip_iter_tau <- pre_burnin_params$clip_iter + int
                    # ##
                    # pre_burnin_params$M_decay_scale <- pre_burnin_params$n_adapt / 100
                    ##
                    pre_burnin_params$n_burnin <- 125
                    pre_burnin_params$n_adapt  <- 100
                    ##
                    pre_burnin_params$gap <- 50
                    ##
                    pre_burnin_params$LR_main <- 0.10
                    pre_burnin_params$LR_us <- 0.10
                    ##
                    pre_burnin_params$manual_tau <- TRUE
                    # pre_burnin_params$tau_if_manual <- c(3.0, 3.0)
                       # pre_burnin_params$tau_if_manual <- c(1.5, 1.5)
                    pre_burnin_params$tau_if_manual <- if (metric_estimator == "pooled") c(1.5, 1.5) else c(3.0, 3.0)
                    ## pre_burnin_L: a fixed number of leapfrog steps instead of a fixed tau (tau is then L * eps each iteration):
                    pre_burnin_params$tau_if_manual_in_L_units <- !is.null(pre_burnin_L)
                    if (!is.null(pre_burnin_L)) pre_burnin_params$tau_if_manual <- c(pre_burnin_L, pre_burnin_L)
                    ##
                    int <- 40
                    pre_burnin_params$clip_iter <- 25
                    pre_burnin_params$clip_iter_tau <- pre_burnin_params$clip_iter + int
                    ##
                    pre_burnin_params$M_decay_scale <-  pre_burnin_params$n_adapt / 5
                    ##
                    ## ---- shorter pre-burnin (pre_burnin_n_iter): the same schedule, scaled to the new length:
##
                    if (!is.null(pre_burnin_n_iter)) {
                          pre_burnin_length_scale <- pre_burnin_n_iter / pre_burnin_params$n_burnin
                          pre_burnin_params$n_burnin      <- as.integer(round(pre_burnin_n_iter))
                          pre_burnin_params$n_adapt       <- round(pre_burnin_params$n_adapt * pre_burnin_length_scale)
                          pre_burnin_params$gap           <- round(pre_burnin_params$gap * pre_burnin_length_scale)
                          pre_burnin_params$clip_iter     <- max(5, round(pre_burnin_params$clip_iter * pre_burnin_length_scale))
                          pre_burnin_params$clip_iter_tau <- pre_burnin_params$clip_iter + round(int * pre_burnin_length_scale)
                          pre_burnin_params$M_decay_scale <-  pre_burnin_params$n_adapt / 5
                          message("pre-burnin shortened to ", pre_burnin_params$n_burnin, " iterations (n_adapt = ", pre_burnin_params$n_adapt,
                                  ", clip_iter = ", pre_burnin_params$clip_iter, ", clip_iter_tau = ", pre_burnin_params$clip_iter_tau, ")")
                    }
                    ##
                    pre_burnin_object <-             fn_burnin(  init_object = init_object,
                                                                 debug_burnin_timing = debug_burnin_timing,
##
                        ## ---- burn-in adaptation settings:
##
                        tau_initial_moments_window_fraction = tau_initial_moments_window_fraction,
                        tau_initial_moments_window_min_iter = tau_initial_moments_window_min_iter,
                        tau_adam_beta1 = tau_adam_beta1,
                        tau_adam_beta2 = tau_adam_beta2,
                        tau_adam_epsilon = tau_adam_epsilon,
                        eps_adam_beta1 = eps_adam_beta1,
                        eps_adam_beta2 = eps_adam_beta2,
                        eps_adam_epsilon = eps_adam_epsilon,
                        tau_adaptation_scheme = tau_adaptation_scheme,
                        tau_learning_rate_restart_at_metric_end = tau_learning_rate_restart_at_metric_end,
                        eps_adam_reset_at_metric_end = eps_adam_reset_at_metric_end,
                        eps_learning_rate_restart_at_metric_end = eps_learning_rate_restart_at_metric_end,
                        rate_criterion_cost_offset_steps = rate_criterion_cost_offset_steps,
                        pooled_metric_warm_up = pooled_metric_warm_up,
                        pooled_metric_soft_full_weight_draws = pooled_metric_soft_full_weight_draws,
                        pooled_metric_weight_k_draws = pooled_metric_weight_k_draws,
                        pooled_metric_signal_to_noise_centred = pooled_metric_signal_to_noise_centred,
                        metric_update_rescales_eps_and_tau = metric_update_rescales_eps_and_tau,
                        learning_rate_initial_n_tau_updates_held_from_tau_handover =
                              learning_rate_initial_n_tau_updates_held_from_tau_handover,
                        resident_burnin = resident_burnin,
                        resident_burnin_joint_block = resident_burnin_joint_block,
                        tau_initial_resident_moments = tau_initial_resident_moments,
                        record_burnin_metric_nuisance_variance_quantiles_history =
                              record_burnin_metric_nuisance_variance_quantiles_history,
                        burnin_tau_set_to_eps_so_every_trajectory_is_one_leapfrog_step =
                              burnin_tau_set_to_eps_so_every_trajectory_is_one_leapfrog_step,
                        Hessian_metric_offdiagonal_shrinkage = Hessian_metric_offdiagonal_shrinkage,
                                                                 ## finite-run ESS of a burnin_algorithm:
                                                                 n_iter_sampling_per_chain = n_iter,
                                                                 ##
                                                                 Model_args_as_Rcpp_List = Model_args_as_Rcpp_List,
                                                                 ##
                                                                 Model_type = Model_type,
                                                                 ##
                                                                 n_chains_burnin = n_chains_burnin,
                                                                 ##
                                                                 theta_main_vectors_all_chains_input_from_R = theta_main_vectors_all_chains_input_from_R,
                                                                 theta_nuisance_vectors_all_chains_input_from_R = theta_nuisance_vectors_all_chains_input_from_R,
                                                                 ##
                                                                 parallel_method = "RcppParallel", ## no OpenMP for burnin yet (only sampling!)
                                                                 ##
                                                                 Stan_data_list = Stan_data_list,
                                                                 model_args_list = model_args_list,
                                                                 ##
                                                                 sample_nuisance = sample_nuisance,
                                                                 n_nuisance_override = n_nuisance_override,
                                                                 ##
                                                                 seed = seed,
                                                                 n_burnin = pre_burnin_params$n_burnin,
                                                                 n_adapt = pre_burnin_params$n_adapt,
                                                                 gap = pre_burnin_params$gap,
                                                                 ##
                                                                 adapt_delta = adapt_delta,
                                                                 eps_acceptance_mean = eps_acceptance_mean,
                                                                 tau_weight_by_p_jump = tau_weight_by_p_jump,
                                                                 LR_main = pre_burnin_params$LR_main,
                                                                 LR_us = pre_burnin_params$LR_us,
                                                                 tau_mult = tau_mult,
                                                                  tau_initial = if_null_then_set_to(
                                                                      x = tau_initial,
                                                                      set_to_this_if_null = 2*pi),
                                                                 ##
                                                                 manual_tau = pre_burnin_params$manual_tau,
                                                                 randomize_tau_burnin = randomize_tau_burnin,
                                                                 ## The ordering pre-burnin is a fixed-length/manual warm-start; advanced trajectory options belong to the main burn-in.
                                                                 tau_gradient_estimator = "forward",
                                                                 tau_cost_exponent = 1,
                                                                 esjd_jump_power = 2,
                                                                 tau_jitter_burnin = "uniform",
                                                                 tau_adaptation_block = tau_adaptation_block,
                                                                 tau_if_manual = pre_burnin_params$tau_if_manual,
                                                                 tau_if_manual_in_L_units = pre_burnin_params$tau_if_manual_in_L_units,
                                                                 ## (inert here: the pre-burnin uses manual_tau = TRUE)
                                                                 tau_shrink_on_divergence        = tau_shrink_on_divergence,
                                                                 tau_shrink_factor               = tau_shrink_factor,
                                                                 tau_shrink_min_divergent_chains = tau_shrink_min_divergent_chains,
                                                                 ##
                                                                 burnin_algorithm = burnin_algorithm,
                                                                 diffusion_HMC = diffusion_HMC,
                                                                 diffusion_HMC_integrator = diffusion_HMC_integrator,
                                                                 partitioned_HMC = partitioned_HMC,
                                                                 ##
                                                                 clip_iter = pre_burnin_params$clip_iter,
                                                                 clip_iter_tau = pre_burnin_params$clip_iter_tau,
                                                                 ##
                                                                 n_refresh = n_refresh,
                                                                 use_proposed = use_proposed,
                                                                 ##
                                                                 beta1_adam = beta1_adam,
                                                                 beta2_adam = beta2_adam,
                                                                 eps_adam = eps_adam,
                                                                 ##
                                                                 force_autodiff = force_autodiff,
                                                                 force_PartialLog = force_PartialLog,
                                                                 multi_attempts = multi_attempts,
                                                                 ##
                                                                 force_autodiff_for_metric = force_autodiff_for_metric,
                                                                 force_PartialLog_for_metric = force_PartialLog_for_metric,
                                                                 force_multi_attempts_for_metric = force_multi_attempts_for_metric,
                                                                 ##
                                                                 metric_type_main = metric_type_main,
                                                                 metric_shape_main = metric_shape_main,
                                                                 ratio_M_main = ratio_M_main,
                                                                 interval_width_main = interval_width_main,
                                                                 ##
                                                                 M_decay_type = M_decay_type,
                                                                 M_decay_power = M_decay_power,
                                                                 M_decay_scale = pre_burnin_params$M_decay_scale,
                                                                 ##
                                                                 metric_type_nuisance = metric_type_nuisance,
                                                                 metric_shape_nuisance = metric_shape_nuisance,
                                                                 ratio_M_nuisance = ratio_M_nuisance,
                                                                 interval_width_nuisance = interval_width_nuisance,
                                                                 ##
                                                                 max_tau_main = max_tau_main,
                                                                 max_tau_nuisance = max_tau_nuisance,
                                                                 ##
                                                                 max_eps_main = max_eps_main,
                                                                 max_eps_nuisance = max_eps_nuisance,
                                                                 ##
                                                                 max_L = max_L,
                                                                 ##
                                                                 n_nuisance_to_track = n_nuisance_to_track,
                                                                 ##
                                                                 n_threads_WCP = n_threads_WCP_burnin,
                                                                 ##
                                                                 time_pre_burnin = time_pre_burnin,
                                                                 ##
                                                                 metric_start_iter = NULL,
                                                                 ##
                                                                 eps_init = NULL,
                                                                 ##
                                                                 ## same holds, scaled to the (short) pre-burnin length:
                                                                 learning_rate_initial = learning_rate_initial,
                                                                 learning_rate_initial_iter = NULL,
                                                                 ##
                                                                 eps_initial = eps_initial,
                                                                 eps_initial_iter = NULL,
                                                                 ##
                                                                 metric_pooled_window_resets         = metric_pooled_window_resets,
                                                                 metric_pooled_offdiagonal_shrinkage = metric_pooled_offdiagonal_shrinkage,
                                                                 ##
                                                                 metric_estimator = metric_estimator)
                      # ##
                      # theta_main_pre <- pre_burnin_object$theta_main_vectors_all_chains_input_from_R
                      # # theta_main_median <- apply(theta_main_pre, 1, median)
                      # theta_main_median <- apply(theta_main_pre, 1, mean)
                      ##
                      time_pre_burnin <- pre_burnin_object$time_burnin
                      ##
                      theta_main_median <- pre_burnin_object$EHMC_burnin_as_Rcpp_List$snaper_m_vec_main
                      ##
                      ## ---- For an external Stan model a failure to read Omega back out of
                      ## the pre-burnin must NOT abort the whole (expensive) run: fall back
                      ## to the identity ordering instead:
                      if (Model_type == "Stan") {
                            ##
                            ## ---- param_constrain() needs the COMPLETE unconstrained vector. The
                            ## built-in skeletons declare no nuisance block, so theta_main alone is the
                            ## whole vector there; a user-supplied Stan model may declare one FIRST, in
                            ## which case the nuisance mean has to be prepended (same layout as
                            ## 'snaper_m_vec_all' = c(nuisance, main) used by the burnin):
                            ##
                            theta_full_median <- if (n_nuisance > 0) {
                                  c(as.numeric(pre_burnin_object$EHMC_burnin_as_Rcpp_List$snaper_m_vec_us),
                                    as.numeric(theta_main_median))
                            } else {
                                  as.numeric(theta_main_median)
                            }
                            ##
                            Omega_hat_list <- tryCatch(
                                  BayesMVP::extract_Omega_via_bridgestan( bs_model = bs_model,
                                                                theta_main = theta_full_median,
                                                                n_tests = n_tests,
                                                                n_class = n_class_for_reorder),
                                  error = function(e) {
                                        big_warning_banner(
                                              title = "reorder_cols_MVP: could not extract 'Omega' after the pre-burnin",
                                              body = c(paste0("Error was: ", conditionMessage(e)),
                                                       "",
                                                       "==> continuing WITHOUT reordering (original column order)."))
                                        NULL
                                  })
                      } else {
                            Omega_hat_list <- BayesMVP::extract_Omega_via_bridgestan( bs_model = bs_model,
                                                                            theta_main = theta_main_median,
                                                                            n_tests = n_tests,
                                                                            n_class = n_class_for_reorder)
                      }
                      # ##
                      # test_perm <- compute_optimal_test_order(Omega_hat_list)
                      # ##
                      # # # test_perm <- c(1,2,3,4,5,6)
                      # # test_perm <- c(4, 5, 3, 1, 6, 2)
                      # test_inv_perm <- order(test_perm)
                      # cat("Optimal test order:", test_perm, "\n")
                      ##
                #     ##
                #     ## Carry forward non-Omega params from pre-burnin
                #     ##
                #     theta_main_vectors_all_chains_input_from_R     <- pre_burnin_object$theta_main_vectors_all_chains_input_from_R
                #     theta_nuisance_vectors_all_chains_input_from_R <- pre_burnin_object$theta_us_vectors_all_chains_input_from_R
                #     ##
                      # theta_main_vectors_all_chains_input_from_R <- theta_main_orig
                      # theta_nuisance_vectors_all_chains_input_from_R <- theta_nuisance_orig
                #     
                #     
                #     test_perm <- c(4, 5, 3, 1, 6, 2) ## ----------------------------------- debug
                      
                      if (!is.null(test_perm_override)) {
                        ## User-fixed order (the pre-burnin estimate above is discarded):
                        test_perm <- as.integer(test_perm_override)
                        if (!identical(sort(test_perm), seq_len(as.integer(n_tests)))) {
                          stop("test_perm_override must be a permutation of 1:", n_tests, "; got: ", paste(test_perm_override, collapse = ", "))
                        }
                      } else if (is.null(Omega_hat_list)) {
                        ## Omega unavailable (external Stan model only - see above): identity order.
                        test_perm <- seq_len(n_tests)
                      } else {
                        # test_perm <- compute_optimal_test_order(Omega_hat_list)
                        ##
                        ## ---- test_order_rule_for_reorder_cols_MVP (resolved above; the first rule is
                        ##      never resolved for an external Stan model).
                        ##      "most_nearly_deterministic_test_last_then_greedy_correlation_order": the
                        ##      test whose class-conditional latent means sit furthest in the tails (the
                        ##      most nearly deterministic test) is placed last, so that no other test's
                        ##      truncation bound depends on its steep-tail nuisance values; the other tests
                        ##      keep the greedy C-vine order. Both use the same pre-burnin estimate
                        ##      (theta_main_median). "greedy_correlation_order": the greedy C-vine order of
                        ##      all tests, the rule used before the option existed:
                        ##
                        if (identical(test_order_rule_for_reorder_cols_MVP,
                                      "most_nearly_deterministic_test_last_then_greedy_correlation_order")) {
                              tail_extremeness_per_test <-  BayesMVP::compute_tail_extremeness_per_test_LC_MVP(
                                    theta_main                   = theta_main_median,
                                    n_tests                      = n_tests,
                                    n_class                      = n_class_for_reorder,
                                    n_covariates_per_outcome_mat = model_args_list$n_covariates_per_outcome_mat,
                                    X                            = model_args_list$X)
                              test_perm <-  BayesMVP::compute_test_order_with_most_nearly_deterministic_test_last(
                                    Omega_hat_list            = Omega_hat_list,
                                    tail_extremeness_per_test = tail_extremeness_per_test)
                              message(colourise(paste0("Test tail extremeness (sum over classes of ",
                                                       "|probit linear predictor|): ",
                                                       paste(formatC(tail_extremeness_per_test, digits = 2,
                                                                     format = "f"), collapse = " "),
                                                       "; most nearly deterministic test placed last: ",
                                                       test_perm[n_tests]), "cyan"))
                        } else if (identical(test_order_rule_for_reorder_cols_MVP,
                                             paste0("test_with_largest_tail_category_distance_from_latent_mean_",
                                                    "last_then_greedy_correlation_order"))) {
                              ##
                              ## ---- binary and ordinal tests: the probability-weighted distance of the tail
                              ##      categories from the class-conditional latent mean (cutpoints from the
                              ##      skeleton's C_vec at the pre-burnin estimate; 0 = the cut of a binary test):
                              ##
                              n_cat_per_test_for_reorder <-  if (is.null(model_args_list$n_cat_per_test)) {
                                    rep(2, n_tests)
                              } else {
                                    as.numeric(model_args_list$n_cat_per_test)
                              }
                              cutpoints_per_class_and_test <-
                                    BayesMVP::extract_cutpoints_per_class_and_test_via_bridgestan(
                                    bs_model           = bs_model,
                                    theta_main         = theta_main_median,
                                    n_tests            = n_tests,
                                    n_class            = n_class_for_reorder,
                                    n_cat_per_test     = n_cat_per_test_for_reorder,
                                    n_thr_per_ord_test = model_args_list$n_thr_per_ord_test)
                              tail_category_distance_per_test <-
                                    BayesMVP::compute_tail_category_distance_from_latent_mean_per_test(
                                    theta_main                   = theta_main_median,
                                    n_tests                      = n_tests,
                                    n_class                      = n_class_for_reorder,
                                    n_covariates_per_outcome_mat = model_args_list$n_covariates_per_outcome_mat,
                                    X                            = model_args_list$X,
                                    cutpoints_per_class_and_test = cutpoints_per_class_and_test)
                              test_perm <-  BayesMVP::compute_test_order_with_most_nearly_deterministic_test_last(
                                    Omega_hat_list            = Omega_hat_list,
                                    tail_extremeness_per_test = tail_category_distance_per_test)
                              message(colourise(paste0("Test tail-category distance from the latent mean (sum ",
                                                       "over classes): ",
                                                       paste(formatC(tail_category_distance_per_test, digits = 2,
                                                                     format = "f"), collapse = " "),
                                                       "; test with the largest distance placed last: ",
                                                       test_perm[n_tests]), "cyan"))
                        } else if (identical(test_order_rule_for_reorder_cols_MVP,
                                             paste0("tests_missing_for_most_people_last_then_greedy_",
                                                    "correlation_order"))) {
                              ##
                              ## ---- the tests missing for more than half of the people (y NA or negative) last;
                              ##      the greedy order within each group:
                              ##
                              greedy_test_order <-  BayesMVP::compute_optimal_test_order(Omega_hat_list)
                              y_for_missingness <-  as.matrix(Stan_data_list$y)
                              proportion_missing_per_test <-  colMeans(is.na(y_for_missingness) |
                                                                       (y_for_missingness < 0))
                              test_is_missing_for_most_people <-  proportion_missing_per_test > 0.5
                              missing_for_most_people_in_greedy_order <-
                                    test_is_missing_for_most_people[greedy_test_order]
                              test_perm <-  c(greedy_test_order[!missing_for_most_people_in_greedy_order],
                                              greedy_test_order[missing_for_most_people_in_greedy_order])
                              message(colourise(paste0("Proportion of people missing each test: ",
                                                       paste(formatC(proportion_missing_per_test, digits = 3,
                                                                     format = "f"), collapse = " "),
                                                       "; greedy order ",
                                                       paste(greedy_test_order, collapse = " "),
                                                       "; tests missing for most people placed last: ",
                                                       paste(test_perm, collapse = " ")), "cyan"))
                        } else {
                              test_perm <- BayesMVP::compute_optimal_test_order(Omega_hat_list)
                        }
                      }
                      test_inv_perm <- order(test_perm)
                      cat(if (!is.null(test_perm_override)) "Test order (test_perm_override):" else "Optimal test order:", test_perm, "\n")
                      ##
                      if (!identical(as.integer(test_perm), seq_len(as.integer(n_tests)))) {
                        ##
                        n_tests_loc <- n_tests
                        ##
                      if ((Model_type == "Stan") && !is.null(Stan_data_list$latent_test_order)) {
                        ##
                        ## ---- USER-SUPPLIED Stan model with the data field 'latent_test_order' (e.g. the
                        ##      four-class LC-MVOP model): the model sets the order in which the tests' latent
                        ##      variables are conditioned on each other (the GHK sequence, the LDL order of the
                        ##      correlation matrices and the nuisance columns) from latent_test_order itself,
                        ##      whilst its data, its other parameters and its outputs stay in the original test
                        ##      order. So the chosen order goes into latent_test_order and NOTHING else in the
                        ##      Stan data is changed (y and every per-test entry stay as supplied):
                        ##
                        Stan_data_list$latent_test_order <- as.integer(test_perm)
                        ##
                        message("reorder_cols_MVP (Stan, latent_test_order): latent_test_order set to [",
                                paste(test_perm, collapse = ", "), "] (latent position j = original test ",
                                "test_perm[j]); the data and the outputs stay in the original test order.")
                        ##
                      } else if (Model_type == "Stan") {
                        ##
                        ## ---- USER-SUPPLIED Stan model: the ONLY object permuted is the outcome
                        ##      matrix 'y' of the supplied Stan data - BayesMVP cannot know which of
                        ##      the supplied other data objects (priors, covariates, ...) are
                        ##      test-indexed, so it touches none of them, and it does not permute
                        ##      user-supplied initial values either:
                        ##
                        Stan_data_list$y <- Stan_data_list$y[, test_perm, drop = FALSE]
                        ##
                        ## ---- If the external Stan model declares a 'test_perm' data variable
                        ##      (the convention used by the built-in skeletons, which un-permute
                        ##      test-indexed quantities in generated quantities), keep it in sync so
                        ##      that the model can un-permute its own outputs:
                        ##
                        if (!is.null(Stan_data_list$test_perm)) {
                          Stan_data_list$test_perm <- as.integer(test_perm)
                        }
                        ##
                        message("reorder_cols_MVP (Stan): permuted the columns of 'y' to order [",
                                paste(test_perm, collapse = ", "), "]; ",
                                "fitted slot j holds the ORIGINAL test test_perm[j]. ",
                                "Only 'y'", if (!is.null(Stan_data_list$test_perm)) " (and 'test_perm')" else "",
                                " in the Stan data was changed.")
                        ##
                      } else {
                        ##
                        ## ---- The correlation bounds and known correlation values in the ORIGINAL test
                        ##      order, kept for the re-layout of the Omega_unconstrained_vec initial values
                        ##      below (made after the bounds and known values have been permuted, so that the
                        ##      fitted-order ones are those the model reads):
                        ##
                        correlation_bounds_and_known_values_in_original_test_order <-
                              list( lb_corr = model_args_list$lb_corr,
                                    ub_corr = model_args_list$ub_corr,
                                    known_values_indicator_list = model_args_list$known_values_indicator_list,
                                    known_values_list = model_args_list$known_values_list)
                        ##
                        ## ---- Permute y:
                        ##
                        y_swapped <- model_args_list$y[, test_perm]
                        y <- y_swapped
                        model_args_list$y <- y_swapped
                        Stan_data_list$y <- y_swapped
                        ##
                        ## ---- Permute correlation bounds / known corrs (existing):
                        ##
                        # perm_data <- permute_corr_data( model_args_list$lb_corr,
                        #                                 model_args_list$ub_corr,
                        #                                 model_args_list$known_values_indicator,
                        #                                 model_args_list$known_values,
                        #                                 test_perm)
                        # model_args_list$lb_corr <- perm_data$lb_corr
                        # model_args_list$ub_corr <- perm_data$ub_corr
                        # model_args_list$known_values_indicator <- perm_data$known_values_indicator
                        # model_args_list$known_values <- perm_data$known_values
                        ##
                        ## ---- The built-in models (C++ and Stan skeletons) read the known correlations
                        ##      from known_values_indicator_list / known_values_list
                        ##      (init_hard_coded_model_args), so these are the fields permuted here (the
                        ##      lines above permuted known_values_indicator / known_values, which no model
                        ##      reads). The correlation priors prior_for_corr_a / prior_for_corr_b are
                        ##      indexed by test pair in the same way. permute_corr_data() reads each matrix
                        ##      through its lower triangle, the triangle the models read:
                        ##
                        perm_data <-  BayesMVP::permute_corr_data( lb_corr = model_args_list$lb_corr,
                                                         ub_corr = model_args_list$ub_corr,
                                                         known_values_indicator =
                                                               model_args_list$known_values_indicator_list,
                                                         known_values = model_args_list$known_values_list,
                                                         perm = test_perm)
                        model_args_list$lb_corr                     <-  perm_data$lb_corr
                        model_args_list$ub_corr                     <-  perm_data$ub_corr
                        model_args_list$known_values_indicator_list <-  perm_data$known_values_indicator
                        model_args_list$known_values_list           <-  perm_data$known_values
                        ##
                        for (prior_for_corr_name in c("prior_for_corr_a", "prior_for_corr_b")) {
                              if (!is.null(model_args_list[[prior_for_corr_name]])) {
                                    model_args_list[[prior_for_corr_name]] <-
                                          lapply( model_args_list[[prior_for_corr_name]],
                                                  fn_permute_test_pair_matrix_reading_its_lower_triangle,
                                                  test_perm = test_perm)
                              }
                        }
                        ##
                        ## ---- Permute coefficient priors, X, covariate counts, baseline cases:
                        ##
                        for (c in 1:n_class) {
                          model_args_list$prior_coeffs_mean_mat[[c]] <- model_args_list$prior_coeffs_mean_mat[[c]][, test_perm, drop = FALSE]
                          model_args_list$prior_coeffs_sd_mat[[c]]   <- model_args_list$prior_coeffs_sd_mat[[c]][, test_perm, drop = FALSE]
                          model_args_list$X[[c]] <- model_args_list$X[[c]][test_perm]
                        }
                        # model_args_list$n_covs_per_outcome <- model_args_list$n_covs_per_outcome[, test_perm, drop = FALSE]   ## CHECK-NAME-1
                        # model_args_list$baseline_case_nd   <- model_args_list$baseline_case_nd[test_perm]                     ## CHECK-NAME-1
                        # model_args_list$baseline_case_d    <- model_args_list$baseline_case_d[test_perm]                      ## CHECK-NAME-1
                        ##
                        ## ---- Permute beta inits (existing):
                        ##
                        for (kk in 1:n_chains_burnin) {
                          for (c in 1:n_class) {
                            init_lists_per_chain[[kk]]$beta[[c]][,] <- init_lists_per_chain[[kk]]$beta[[c]][, test_perm, drop = FALSE]
                          }
                        }
                        ##
                        ## ---- Permute the u_raw (nuisance) initial values with the tests: u_raw holds one
                        ##      column per test (N x n_tests), so fitted slot j takes the column of ORIGINAL
                        ##      test test_perm[j]:
                        ##
                        for (kk in 1:n_chains_burnin) {
                          if (!is.null(init_lists_per_chain[[kk]]$u_raw)) {
                            init_lists_per_chain[[kk]]$u_raw <-
                                  BayesMVP:::fn_u_raw_initial_values_in_fitted_test_order(
                                        u_raw = init_lists_per_chain[[kk]]$u_raw,
                                        test_perm = test_perm,
                                        N = nrow(y_swapped))
                          }
                        }
                        ##
                        ## ---- Map the Omega_unconstrained_vec initial values into the fitted test order: each
                        ##      class's raw vector holds the bounded LDL ("Pinkney") coordinates of its
                        ##      correlation matrix in the ORIGINAL order; it is mapped through the correlation
                        ##      matrix itself, raw (original order, original-order bounds / known values) ->
                        ##      Omega -> Omega[test_perm, test_perm] -> raw (fitted order, with the bounds /
                        ##      known values the model now reads):
                        ##
                        correlation_bounds_and_known_values_in_fitted_test_order <-
                              list( lb_corr = model_args_list$lb_corr,
                                    ub_corr = model_args_list$ub_corr,
                                    known_values_indicator_list = model_args_list$known_values_indicator_list,
                                    known_values_list = model_args_list$known_values_list)
                        for (kk in 1:n_chains_burnin) {
                          if (!is.null(init_lists_per_chain[[kk]]$Omega_unconstrained_vec)) {
                            init_lists_per_chain[[kk]]$Omega_unconstrained_vec <-
                                  BayesMVP:::fn_Omega_unconstrained_vec_initial_values_in_fitted_test_order(
                                        Omega_unconstrained_vec =
                                              init_lists_per_chain[[kk]]$Omega_unconstrained_vec,
                                        test_perm = test_perm,
                                        n_class = n_class,
                                        bounds_and_known_values_in_original_test_order =
                                              correlation_bounds_and_known_values_in_original_test_order,
                                        bounds_and_known_values_in_fitted_test_order =
                                              correlation_bounds_and_known_values_in_fitted_test_order,
                                        corr_force_positive = isTRUE(model_args_list$corr_force_positive))
                          }
                        }
                        message(colourise(paste0("reorder_cols_MVP: u_raw and Omega_unconstrained_vec initial ",
                                                 "values re-laid out into the fitted test order [",
                                                 paste(test_perm, collapse = ", "), "]"), "cyan"))
                        ##
                        ## ================================================================
                        ## vvvvvvvvvvvv  NEW BLOCK - PASTE ALL OF THIS HERE  vvvvvvvvvvvv
                        ## ================================================================
                        ##
                        model_args_list$n_covariates_per_outcome_mat <- model_args_list$n_covariates_per_outcome_mat[, test_perm, drop = FALSE]
                        ##
                        model_args_list$test_perm <- test_perm   ## picked up by make_Stan_data_list at re-init
                        ##
                        # if (Model_type %in% c("LC_MVOP",
                        #                       "MVOP")) {
                        #   
                        #       n_bin_orig    <- model_args_list$n_binary_tests
                        #       ord_of_fitted <- test_perm[test_perm > n_bin_orig] - n_bin_orig
                        #       ##
                        #       if (!identical(ord_of_fitted, seq_along(ord_of_fitted))) {
                        #         
                        #             n_thr_old <- model_args_list$n_thr_per_ord_test
                        #             ##
                        #             model_args_list$prior_dirichlet_alpha <- model_args_list$prior_dirichlet_alpha[, ord_of_fitted, drop = FALSE]
                        #             ##
                        #             model_args_list$n_cat_per_ord_test <- model_args_list$n_cat_per_ord_test[ord_of_fitted]
                        #             model_args_list$n_thr_per_ord_test <- model_args_list$n_thr_per_ord_test[ord_of_fitted]
                        #             ##
                        #             end_old   <- cumsum(n_thr_old)
                        #             start_old <- c(1, head(end_old, -1) + 1)
                        #             ##
                        #             for (kk in 1:n_chains_burnin) {
                        #                 if (is.null(init_lists_per_chain[[kk]]$C_raw_vec)) {
                        #                   stop("permute block: init_lists_per_chain[[kk]]$C_raw_vec not found -- check the actual C_raw init field name!")
                        #                 }
                        #                 for (c in 1:2) {
                        #                   C_raw  <- init_lists_per_chain[[kk]]$C_raw_vec[[c]]
                        #                   blocks <- lapply(seq_along(n_thr_old), function(tt) C_raw[start_old[tt]:end_old[tt]])
                        #                   init_lists_per_chain[[kk]]$C_raw_vec[[c]] <- unlist(blocks[ord_of_fitted])
                        #                 }
                        #             }
                        #         
                        #       }
                        #       
                        # }
                        ##
                        ## ---- Ordinal metadata permutation (FULLY GENERAL -- makes NO assumption
                        ##      that the original data is binaries-first, nor that the permutation
                        ##      keeps binary and ordinal tests in separate blocks).
                        ##
                        ##      All ordinal-indexed metadata (n_cat_per_ord_test, n_thr_per_ord_test,
                        ##      prior_dirichlet_alpha, C_raw init blocks) is stored in ORDINAL-SLOT
                        ##      order: the k-th entry describes the k-th ordinal test *in slot order*.
                        ##      Under a permutation, slot j holds ORIGINAL test test_perm[j], so the
                        ##      new ordinal ordering is read straight off which fitted slots are
                        ##      ordinal, mapped back to their original ordinal indices.
                        ##
                        if (Model_type %in% c("LC_MVOP", 
                                              "MVOP")) {
                          
                              ##
                              ## ---- Which ORIGINAL tests are ordinal, and their ordinal index (NA for binary).
                              ##      Derived from n_cat_per_test, which is in ORIGINAL order at this point
                              ##      (the permutation has not been applied to it yet):
                              ##
                              n_cat_orig    <- model_args_list$n_cat_per_test
                              is_ord_orig   <- (n_cat_orig > 2L)
                              ##
                              ord_idx_orig            <- rep(NA_integer_, n_tests_loc)
                              ord_idx_orig[is_ord_orig] <- seq_len(sum(is_ord_orig))
                              ##
                              ## ---- ord_of_fitted[k] = ORIGINAL ordinal index of the k-th ordinal FITTED slot.
                              ##      Walk the fitted slots in order; slot j holds original test test_perm[j];
                              ##      keep the ordinal ones:
                              ##
                              ord_of_fitted <- ord_idx_orig[test_perm]
                              ord_of_fitted <- ord_of_fitted[!is.na(ord_of_fitted)]
                              ##
                              stopifnot(length(ord_of_fitted) == sum(is_ord_orig))
                              ##
                              if (!identical(ord_of_fitted, seq_along(ord_of_fitted))) {
                                
                                    n_thr_old <- model_args_list$n_thr_per_ord_test
                                    ##
                                    ## ---- Ordinal-indexed metadata:
                                    ##
                                    model_args_list$n_cat_per_ord_test    <- model_args_list$n_cat_per_ord_test[ord_of_fitted]
                                    model_args_list$n_thr_per_ord_test    <- model_args_list$n_thr_per_ord_test[ord_of_fitted]
                                    # model_args_list$prior_dirichlet_alpha <-
                                    #       model_args_list$prior_dirichlet_alpha[, ord_of_fitted, drop = FALSE]
                                    ##
                                    ## ---- prior_dirichlet_alpha is a LIST: one matrix
                                    ##      (max(n_cat_per_ord_test) x n_ordinal_tests) per latent class
                                    ##      (init_hard_coded_model_args). Each class's columns go into
                                    ##      fitted ordinal-slot order (the line above indexed the list as a
                                    ##      matrix):
                                    ##
                                    fn_dirichlet_alpha_columns_in_fitted_ordinal_slot_order <-
                                          function(prior_dirichlet_alpha_matrix) {
                                                return(prior_dirichlet_alpha_matrix[, ord_of_fitted,
                                                                                    drop = FALSE])
                                          }
                                    if (is.list(model_args_list$prior_dirichlet_alpha)) {
                                          model_args_list$prior_dirichlet_alpha <-
                                                lapply( model_args_list$prior_dirichlet_alpha,
                                                        fn_dirichlet_alpha_columns_in_fitted_ordinal_slot_order)
                                    } else {
                                          model_args_list$prior_dirichlet_alpha <-
                                                fn_dirichlet_alpha_columns_in_fitted_ordinal_slot_order(
                                                      model_args_list$prior_dirichlet_alpha)
                                    }
                                    ##
                                    ## ---- Ragged C_raw init blocks: whole-block permutation (blocks are
                                    ##      independent per test -- 1st elem is the first cutpoint, rest are
                                    ##      log-gaps WITHIN that test):
                                    ##
                                    end_old   <- cumsum(n_thr_old)
                                    start_old <- c(1, head(end_old, -1) + 1)
                                    ##
                                    # for (kk in 1:n_chains_burnin) {
                                    #       if (is.null(init_lists_per_chain[[kk]]$C_raw_vec)) {
                                    #         stop("permute block: init_lists_per_chain[[kk]]$C_raw_vec not ",
                                    #              "found -- check the init field name!")
                                    #       }
                                    #       for (c in 1:2) {
                                    #         C_raw  <- init_lists_per_chain[[kk]]$C_raw_vec[[c]]
                                    #         blocks <- lapply(seq_along(n_thr_old),
                                    #                          function(tt) C_raw[start_old[tt]:end_old[tt]])
                                    #         init_lists_per_chain[[kk]]$C_raw_vec[[c]] <-
                                    #               unlist(blocks[ord_of_fitted])
                                    #       }
                                    # }
                                    ##
                                    ## ---- The cutpoint PARAMETER of the LC_MVOP / MVOP skeletons, i.e. the
                                    ##      initial-value field, is C_unc_vec: one vector of length
                                    ##      sum(n_thr_per_ord_test) per latent class (C_raw_vec is a transformed
                                    ##      parameter there). Its per-test blocks have different lengths and
                                    ##      move as whole blocks, in every latent class:
                                    ##
                                    fn_C_unc_vec_blocks_in_fitted_ordinal_slot_order <-
                                          function(C_unc_vec_of_one_class) {
                                                block_of_ordinal_slot <-  function(tt) {
                                                      C_unc_vec_of_one_class[start_old[tt]:end_old[tt]]
                                                }
                                                blocks <-  lapply(seq_along(n_thr_old), block_of_ordinal_slot)
                                                return(unlist(blocks[ord_of_fitted]))
                                          }
                                    ##
                                    for (kk in 1:n_chains_burnin) {
                                          C_unc_vec_initial_values <-  init_lists_per_chain[[kk]]$C_unc_vec
                                          if (is.null(C_unc_vec_initial_values)) {
                                                stop(paste0("test re-ordering: init_lists_per_chain[[", kk,
                                                            "]] has no C_unc_vec (the cutpoint parameter of ",
                                                            "the ", Model_type, " model)."))
                                          }
                                          if (is.matrix(C_unc_vec_initial_values)) {   ## one row per class
                                                for (c in seq_len(nrow(C_unc_vec_initial_values))) {
                                                      C_unc_vec_initial_values[c, ] <-
                                                            fn_C_unc_vec_blocks_in_fitted_ordinal_slot_order(
                                                                  C_unc_vec_initial_values[c, ])
                                                }
                                          } else {   ## a list: one vector per latent class
                                                for (c in seq_along(C_unc_vec_initial_values)) {
                                                      C_unc_vec_initial_values[[c]] <-
                                                            fn_C_unc_vec_blocks_in_fitted_ordinal_slot_order(
                                                                  C_unc_vec_initial_values[[c]])
                                                }
                                          }
                                          init_lists_per_chain[[kk]]$C_unc_vec <-  C_unc_vec_initial_values
                                    }
                                
                              }
                              ##
                              ## ---- n_cat_per_test is TEST-indexed, so permute directly (independent of
                              ##      whether the ordinal sub-order changed):
                              ##
                              model_args_list$n_cat_per_test <- n_cat_orig[test_perm]
                          
                        }
                        ##
                        ## ================================================================
                        ## ^^^^^^^^^^^^^^^^^^  END OF NEW BLOCK  ^^^^^^^^^^^^^^^^^^^^^^^^
                        ## ================================================================
                        # ##
                        # ## ---- Slot-type vector (single source of truth) + ordinal metadata (LC_MVOP):
                        # ##
                        # model_args_list$n_cat_per_test <- model_args_list$n_cat_per_test[test_perm]                           ## CHECK-NAME-2
                        # ##
                        # if (Model_type == "LC_MVOP") {
                        #   
                        #       n_bin_orig <- sum(model_args_list$n_cat_per_test == 2)   ## count is perm-invariant
                        #       ##
                        #       ## fitted-slot-order list of ORIGINAL ordinal indices:
                        #       # ord_of_fitted <- test_perm[ model_args_list$n_cat_per_test[order(test_perm)][test_perm] > 2 ]  ## see simpler line below
                        #       ord_of_fitted <- test_perm[test_perm > n_bin_orig] - n_bin_orig   ## USE THIS ONE (original data = binaries-first)
                        #       ##
                        #       if (!identical(ord_of_fitted, seq_along(ord_of_fitted))) {
                        #         n_thr_old <- model_args_list$n_thr_per_ord_test
                        #         ##
                        #         model_args_list$n_thr_per_ord_test    <- model_args_list$n_thr_per_ord_test[ord_of_fitted]
                        #         model_args_list$n_cat_per_ord_test    <- model_args_list$n_cat_per_ord_test[ord_of_fitted]
                        #         model_args_list$prior_dirichlet_alpha <- model_args_list$prior_dirichlet_alpha[, ord_of_fitted, drop = FALSE]
                        #         ##
                        #         ## ---- Ragged C_raw init blocks: whole-block permutation (blocks are
                        #         ##      independent per test: 1st elem unconstrained, rest log-diffs WITHIN test):
                        #         end_old   <- cumsum(n_thr_old)
                        #         start_old <- c(1, head(end_old, -1) + 1)
                        #         for (kk in 1:n_chains_burnin) {
                        #           for (c in 1:2) {
                        #             C_raw <- init_lists_per_chain[[kk]]$C_raw_vec[[c]]                               ## CHECK-NAME-3
                        #             blocks <- lapply(seq_along(n_thr_old), function(tt) C_raw[start_old[tt]:end_old[tt]])
                        #             init_lists_per_chain[[kk]]$C_raw_vec[[c]] <- unlist(blocks[ord_of_fitted])       ## CHECK-NAME-3
                        #           }
                        #         }
                        #       }
                        #   
                        # }
                        # ##
                        # ## ---- Pass permutation + slot types to the skeleton:
                        # Stan_data_list$test_perm      <- test_perm
                        # Stan_data_list$n_cat_per_test <- model_args_list$n_cat_per_test
                        # ##
                        # ## ---- Mirror metadata into Stan_data_list (maintained in parallel with model_args_list):
                        # if (Model_type == "LC_MVOP") {
                        #   Stan_data_list$n_thr_per_ord_test    <- model_args_list$n_thr_per_ord_test
                        #   Stan_data_list$n_cat_per_ord_test    <- model_args_list$n_cat_per_ord_test
                        #   Stan_data_list$prior_dirichlet_alpha <- model_args_list$prior_dirichlet_alpha
                        # }
                # #     ##
                #       if (!identical(test_perm, 1:init_object$model_args_list$n_tests)) {
                # #       
                #             ##
                #             ## ---- Permute y:
                #             ##
                #             y_swapped <- model_args_list$y[, test_perm]
                #             y <- y_swapped
                #             model_args_list$y <- y_swapped
                #             Stan_data_list$y <- y_swapped
                #             ##
                #             ## ---- Permute correlation bounds and any known corr's:
                #             ##
                #             perm_data <- permute_corr_data(  model_args_list$lb_corr,
                #                                              model_args_list$ub_corr,
                #                                              model_args_list$known_values_indicator,
                #                                              model_args_list$known_values,
                #                                              test_perm)
                #             model_args_list$lb_corr <- perm_data$lb_corr
                #             model_args_list$ub_corr <- perm_data$ub_corr
                #             model_args_list$known_values_indicator <- perm_data$known_values_indicator
                #             model_args_list$known_values <- perm_data$known_values
                #             ##
                #             ## ---- Permute coefficient priors and X:
                #             ##
                #             for (c in 1:n_class) {
                #               model_args_list$prior_coeffs_mean_mat[[c]] <- model_args_list$prior_coeffs_mean_mat[[c]][, test_perm, drop = FALSE]
                #               model_args_list$prior_coeffs_sd_mat[[c]]   <- model_args_list$prior_coeffs_sd_mat[[c]][, test_perm, drop = FALSE]
                #               model_args_list$X[[c]] <- model_args_list$X[[c]][test_perm]
                #             }
                #             ##
                #             for (kk in 1:n_chains_burnin) { 
                #               for (c in 1:n_class) {
                #                 init_lists_per_chain[[kk]]$beta[[c]][,] <- init_lists_per_chain[[kk]]$beta[[c]][, test_perm, drop = FALSE]
                #               }
                #             }
                # #           ##
                # #           ## Zero out Omega params; keep beta/prevalence from pre-burnin:
                # #           ##
                # #           theta_main_vectors_all_chains_input_from_R <- zero_omega_in_theta( theta_main_pre,
                # #                                                                              bs_model)
                            ##
                      } ## end of "else" (built-in models) of the "if (Model_type == 'Stan')" permute-branch
                            ##
                            ## ---- Update init_object:
                            ##
                            # Stan_data_list$y <- Stan_data_list$y[, test_perm]
                            ##
                            init_object <- initialise_model( Model_type =  Model_type,
                                                             ##
                                                             stream = stream,
                                                             ##
                                                             sample_nuisance = sample_nuisance,
                                                             n_nuisance_override = n_nuisance_override,
                                                             ##
                                                             model_args_list = model_args_list,
                                                             ##
                                                             compile = init_object$compile,
                                                             force_recompile = init_object$force_recompile,
                                                             ##
                                                             cmdstanr_model_fit_obj = NULL,
                                                             ##
                                                             Stan_data_list = Stan_data_list,
                                                             ##
                                                             Stan_model_file_path = Stan_model_file_path,
                                                             Stan_cpp_user_header = Stan_cpp_user_header,
                                                             Stan_cpp_flags = Stan_cpp_flags,
                                                             stanc_args = stanc_args,
                                                             make_args = make_args)
                            ##
                            ## (the writer's record of the nuisance blocks held by NicoStan, kept through the
                            ##  rebuild)
                            init_object$nuisance_blocks_held_by_NicoStan <-
                                  nuisance_blocks_held_by_NicoStan_of_the_model
                            ##
                            Model_args_as_Rcpp_List <- init_object$Model_args_as_Rcpp_List
                            Model_args_as_Rcpp_List$autodiff_fallback <-  autodiff_fallback
                            Model_args_as_Rcpp_List$n_chunks_multiplier_for_PartialLog_log_scale_evaluation <-
                                  n_chunks_multiplier_for_PartialLog_log_scale_evaluation
                            model_args_list         <- init_object$model_args_list 
                            ##
                            n_nuisance <- init_object$n_nuisance ; n_nuisance
                            n_params_main <- init_object$n_params_main ; n_params_main
                            ##
                            positions_of_the_nuisance_values_to_keep <- fn_positions_of_the_nuisance_values_to_track(
                                  nuisance_to_track = nuisance_to_track_user,
                                  nuisance_blocks   = init_object$nuisance_blocks_held_by_NicoStan,
                                  Stan_data_list    = Stan_data_list,
                                  n_nuisance        = n_nuisance)
                            n_nuisance_to_track <- length(positions_of_the_nuisance_values_to_keep)
                            if (n_nuisance_to_track > 0L) {
                                  Rcpp_fn_set_the_positions_of_the_nuisance_values_kept_in_the_trace(
                                        positions_from_0 = as.integer(positions_of_the_nuisance_values_to_keep))
                            }
                            # ##
                            # if (is.null(n_nuisance_to_track)) {
                            #   if (sample_nuisance == FALSE) {
                            #     n_nuisance <- 1 # dummy (or 9? try both)
                            #     n_nuisance_to_track <- 1 # dummy (or 9? try both)
                            #   } else {
                            #     n_nuisance_to_track <- n_nuisance
                            #   }
                            # }
                            n_params <- n_nuisance + n_params_main
                            ##
                            print(paste("n_params = ", n_params))
                            print(paste("n_params_main = ",  n_params_main))
                            print(paste("n_nuisance = ",  n_nuisance))
                            ##
                            Model_args_as_Rcpp_List$n_nuisance <- n_nuisance
                            bs_model <- init_object$bs_model
                            ##
                            ## ---- For external Stan models the DATA reaches the C++ sampler through the
                            ##      JSON file (not through Model_args_as_Rcpp_List), and the JSON path is
                            ##      content-addressed (hashed), so re-initialising on the permuted data
                            ##      produced a NEW json file: point the sampler at it (and re-write it from
                            ##      the permuted Stan_data_list, matching what is done pre-reorder):
                            ##
                            if (Model_type == "Stan") {
                                  ##
                                  Model_args_as_Rcpp_List$n_params_main <- n_params_main
                                  ##
                                  Model_args_as_Rcpp_List$model_so_file  <- init_object$model_so_file
                                  Model_args_as_Rcpp_List$json_file_path <- init_object$json_file_path
                                  ##
                                  ## (atomic write - temporary file + rename; see fn_write_file_atomically)
                                  fn_write_stan_json_atomically( stan_data_list = Stan_data_list,
                                                                 json_file_path = Model_args_as_Rcpp_List$json_file_path)
                                  ##
                                  print(paste("post-reorder model_so_file = ",  Model_args_as_Rcpp_List$model_so_file))
                                  print(paste("post-reorder json_file_path = ", Model_args_as_Rcpp_List$json_file_path))
                                  ##
                            }
                            ##
                            index_nuisance <- seq_len(n_nuisance)
                            index_main <- (1 + n_nuisance):n_params
                            ##
                            ## ---- Process initial values BEFORE burnin:
                            ##
                            ## ---- the chunk layout of the nuisance vector in the burn-in (built-in models): the
                            ##      burn-in's number of chunks and vectorisation, as the C++ reads them:
                            chunk_layout_of_nuisance_vector_in_burnin <-  NULL
                            if (Model_type != "Stan") {
                                  chunk_layout_of_nuisance_vector_in_burnin <-
                                        list( N = nrow(y),
                                              n_tests = ncol(y),
                                              n_chunks = Model_args_as_Rcpp_List$Model_args_ints[4],
                                              vect_type = Model_args_as_Rcpp_List$Model_args_strings[1])
                            }
                            outs <- R_fn_init_initial_values( Model_type = Model_type,
                                                              bs_model = bs_model,
                                                              ##
                                                              n_chains_burnin = n_chains_burnin,
                                                              init_lists_per_chain = init_lists_per_chain,
                                                              ##
                                                              sample_nuisance = sample_nuisance,
                                                              n_nuisance = n_nuisance,
                                                              n_params_main = n_params_main,
                                                              ##
                                                              chunk_layout_of_nuisance_vector_in_burnin =
                                                                    chunk_layout_of_nuisance_vector_in_burnin,
                                                              ##
                                                              nuisance_blocks_held_by_NicoStan =
                                                                    init_object$nuisance_blocks_held_by_NicoStan,
                                                              Stan_data_list = Stan_data_list)
                            ##
                            inits_unconstrained_vec_per_chain <- outs$inits_unconstrained_vec_per_chain
                            ##
                            theta_nuisance_vectors_all_chains_input_from_R <- outs$theta_nuisance_vectors_all_chains_input_from_R
                            theta_main_vectors_all_chains_input_from_R     <- outs$theta_main_vectors_all_chains_input_from_R
                            # theta_main_vectors_all_chains_input_from_R <- theta_main_pre ## -------------------
                            ##
                            n_chains_burnin <- outs$n_chains_burnin
                            init_lists_per_chain <- outs$init_lists_per_chain
                            # ##
                            # if (Model_type != "Stan")  {
                            #   n_params_main <- Model_args_as_Rcpp_List$n_params_main # <- n_params_main
                            #   n_nuisance <-    Model_args_as_Rcpp_List$n_nuisance #<- n_nuisance
                            # } else if  (Model_type == "Stan")   {
                            #   Model_args_as_Rcpp_List$n_params_main <- n_params_main
                            #   Model_args_as_Rcpp_List$n_nuisance <- n_nuisance
                            # }
                            # if (Model_type == "Stan") {
                            #   Model_args_as_Rcpp_List$model_so_file <-    init_object$model_so_file
                            #   Model_args_as_Rcpp_List$json_file_path <-   init_object$json_file_path
                            # } else {
                            #   Model_args_as_Rcpp_List$model_so_file <-  "none" #   init_object$dummy_model_so_file
                            #   Model_args_as_Rcpp_List$json_file_path <- "none" #  init_object$dummy_json_file_path
                            # }
                            # ##
                            # if (Model_type == "Stan") {
                            #   cmdstanr::write_stan_json(data = Stan_data_list, file = Model_args_as_Rcpp_List$json_file_path)
                            # } else {
                            #   # cmdstanr::write_stan_json(data = Stan_data_list, file = Model_args_as_Rcpp_List$dummy_json_file_path)
                            # }
                            # ##
                            # print(  Model_args_as_Rcpp_List$model_so_file)
                            # print(  Model_args_as_Rcpp_List$json_file_path)
                            # ##
                            # RcppParallel::setThreadOptions(numThreads = n_chains_burnin);
                            # ## else if (burnin_algorithm %in% c("SNAPER", "snaper")) {
                            #   fn_burnin <- init_and_run_burnin_SNAPER
                            # }
                            # ##
                            # n_tests <- init_object$model_args_list$n_tests
                            ##
                            ## ---- Use FRESH inits from the new init_object, not pre-burnin values:
                            ##
                            # theta_main_vectors_all_chains_input_from_R <- theta_main_orig
                            # theta_nuisance_vectors_all_chains_input_from_R <- theta_nuisance_orig
                            # ##
                            # theta_main_vectors_all_chains_input_from_R <- pre_burnin_object$theta_main_vectors_all_chains_input_from_R  # inits stored here
                            # theta_nuisance_vectors_all_chains_input_from_R <- pre_burnin_object$theta_us_vectors_all_chains_input_from_R
                      }
                      
                } else {
                    pre_burnin_object <- NULL
                }
                ##
                ## ---- Run burnin:
                ##
                ## Explicit starting path lengths take precedence over the metric-estimator default:
                tau_initial <- if_null_then_set_to(x = tau_initial,
                                                  set_to_this_if_null = if (metric_estimator == "pooled") pi else 2*pi)
                ##
                # if (!(is.null(pre_burnin_object))) { 
                #     clip_iter <- 5
                #     clip_iter_tau <- 15
                # }
                ##
                ## ---- eps_reinit_after_pre_burnin = FALSE: the main burn-in starts from the pre-burnin's final eps (main and
                ##      nuisance) and skips its find_initial_eps search. NULL = the main burn-in runs that search itself:
                ##
                eps_carry_over <- NULL
                if (!isTRUE(eps_reinit_after_pre_burnin)) {
                      if (is.null(pre_burnin_object)) {
                            message(colourise("eps_reinit_after_pre_burnin = FALSE has no effect: no test-order pre-burnin ran (it runs only with reorder_cols_MVP = TRUE), so the main burn-in runs its own eps search.",
                                              "cyan"))
                      } else {
                            eps_carry_over <- list(eps_main = pre_burnin_object$eps_main,
                                                   eps_us   = pre_burnin_object$eps_us)
                      }
                }
                ##
                ## ---- CHESSR_time / SNAPER_time: the sampling-side quantities of the time-to-target-ESS criterion, resolved BEFORE the
                ##      main burn-in (R_fn_time_criterion.R). The sampling timing probe uses the native sampling entry point
                ##      and the summaries routine of this function's own environment (a model package's backend supplies its natives), so
                ##      it is bound to that environment:
                ##
                time_criterion_sampling_quantities      <- NULL
                time_criterion_previous_run_quantities  <- NULL
                sampling_timing_probe_result            <- NULL
                ## (sampling_overhead_in_leapfrog_steps = "auto": the built-in default for this model and sampling configuration, when one
                ##  exists, see below)
                sampling_overhead_in_leapfrog_steps_built_in_default <- NULL
                if (fn_burnin_algorithm_is_time_criterion(burnin_algorithm) && !isTRUE(manual_tau)) {
                      ##
                      ## ---- the burn-in refuses these criteria with partitioned_HMC = TRUE (see R_fn_time_criterion.R); stop here, before
                      ##      the previous-run read and the probe, instead of after them (partitioned_HMC is final at this point):
                      ##
                      if (isTRUE(partitioned_HMC)) {
                            stop(paste0("burnin_algorithm = '", burnin_algorithm, "' is defined for the joint sampler (partitioned_HMC = FALSE), whose single tau ",
                                        "sets the leapfrog steps of every gradient evaluation; use '", sub("_time$", "", burnin_algorithm), "' with partitioned_HMC = TRUE."))
                      }
                      if (!is.null(time_criterion_settings$time_criterion_previous_run_path)) {
                            ## ESS_per_iter_sampling_expected is the minimum ESS over time_criterion_ess_parameter_set ("diagnostic" by default:
                            ## the generated quantities of the model's Stan skeleton file, from its BridgeStan metadata, unless the saved run
                            ## records its own diagnostic_parameter_names; never the parameters block):
                            model_diagnostic_parameter_names_for_time_criterion <- if (is.character(init_object$stan_main_and_tp_and_gq_param_names) &&
                                                                                       is.character(init_object$stan_main_and_tp_param_names))
                                  setdiff(init_object$stan_main_and_tp_and_gq_param_names, init_object$stan_main_and_tp_param_names) else NULL
                            time_criterion_previous_run_quantities <- fn_time_criterion_quantities_from_saved_run(
                                  time_criterion_previous_run_path  = time_criterion_settings$time_criterion_previous_run_path,
                                  time_criterion_ess_parameter_set  = time_criterion_settings$time_criterion_ess_parameter_set,
                                  model_diagnostic_parameter_names  = model_diagnostic_parameter_names_for_time_criterion)
                            message(colourise(paste0("time criterion: previous run(s) ", paste(basename(time_criterion_settings$time_criterion_previous_run_path), collapse = ", "),
                                                     " (", time_criterion_previous_run_quantities$sampling_time_estimation_method, ")",
                                                     " | time_per_leapfrog_step_sampling = ", signif(time_criterion_previous_run_quantities$time_per_leapfrog_step_sampling, 4), " s",
                                                     " | sampling_overhead_in_leapfrog_steps = ", signif(time_criterion_previous_run_quantities$sampling_overhead_in_leapfrog_steps, 4),
                                                     " | ESS_per_iter_sampling_expected = ", signif(time_criterion_previous_run_quantities$ESS_per_iter_sampling_expected, 4),
                                                     " (min ESS over '", time_criterion_previous_run_quantities$ESS_per_iter_sampling_expected_parameter_set, "': ",
                                                     time_criterion_previous_run_quantities$ESS_per_iter_sampling_expected_min_ESS_source, ")"),
                                              "cyan"))
                      }
                      ##
                      ## ---- sampling_overhead_in_leapfrog_steps = "auto": a value for the SAMPLING configuration (n_chains_sampling chains,
                      ##      n_threads_WCP_sampling threads per chain, the sampling chunk count), never from the burn-in, whose configuration
                      ##      differs, or from a saved run: the built-in default for this model, configuration and machine when one exists
                      ##      (fn_sampling_overhead_in_leapfrog_steps_built_in_default_for_run and fn_default_sampling_overhead_in_leapfrog_steps,
                      ##      R_fn_time_criterion.R), otherwise the sampling timing probe below in its "lite" mode (sampling_timing_probe_mode;
                      ##      fn_sampling_timing_probe_mode_for_run):
                      ##
                      sampling_timing_probe_mode_for_run <- fn_sampling_timing_probe_mode_for_run(time_criterion_settings = time_criterion_settings)
                      if (identical(time_criterion_settings$sampling_overhead_in_leapfrog_steps, "auto")) {
                            built_in_default_lookup_for_run <- fn_sampling_overhead_in_leapfrog_steps_built_in_default_for_run(
                                  Model_type              = Model_type,
                                  init_object             = init_object,
                                  model_args_list         = model_args_list,
                                  num_chunks_sampling     = num_chunks_sampling,
                                  n_chains_sampling       = n_chains_sampling,
                                  n_threads_WCP_sampling  = n_threads_WCP_sampling)
                            sampling_overhead_in_leapfrog_steps_built_in_default <- built_in_default_lookup_for_run$sampling_overhead_in_leapfrog_steps_built_in_default
                            n_observations_for_built_in_default <- built_in_default_lookup_for_run$n_observations
                            num_chunks_sampling_for_built_in_default <- built_in_default_lookup_for_run$num_chunks_sampling
                            if (!is.null(sampling_overhead_in_leapfrog_steps_built_in_default)) {
                                  message(colourise(paste0("time criterion: sampling_overhead_in_leapfrog_steps = \"auto\": built-in default for ", Model_type, ", ",
                                                           n_observations_for_built_in_default, " observations, ", n_chains_sampling, " chains x ",
                                                           n_threads_WCP_sampling, " thread(s), sampling chunk count ", num_chunks_sampling_for_built_in_default, " = ",
                                                           signif(sampling_overhead_in_leapfrog_steps_built_in_default$sampling_overhead_in_leapfrog_steps, 4),
                                                           " (iteration overhead ", signif(sampling_overhead_in_leapfrog_steps_built_in_default$sampling_overhead_in_leapfrog_steps_from_iteration_overhead, 4),
                                                           " + summaries ", signif(sampling_overhead_in_leapfrog_steps_built_in_default$sampling_overhead_in_leapfrog_steps_from_summaries, 4),
                                                           " leapfrog steps; ", sampling_overhead_in_leapfrog_steps_built_in_default$calibration_method, "); no sampling timing probe"),
                                                    "cyan"))
                            } else {
                                  message(colourise(paste0("time criterion: sampling_overhead_in_leapfrog_steps = \"auto\": no built-in default for ", Model_type, ", ",
                                                           n_observations_for_built_in_default, " observations, ", n_chains_sampling, " chains x ",
                                                           n_threads_WCP_sampling, " thread(s), sampling chunk count ", num_chunks_sampling_for_built_in_default,
                                                           ", ", built_in_default_lookup_for_run$n_physical_cores_of_this_machine, " physical cores",
                                                           if (isTRUE(time_criterion_settings$run_sampling_timing_probe)) paste0("; the sampling timing probe runs in its \"",
                                                                                                                                 sampling_timing_probe_mode_for_run, "\" mode") else
                                                               "; run_sampling_timing_probe = FALSE, so no probe runs"),
                                                    "cyan"))
                            }
                      }
                      if (isTRUE(time_criterion_settings$run_sampling_timing_probe) &&
                          fn_sampling_timing_probe_is_needed( time_criterion_settings                = time_criterion_settings,
                                                              ## time_criterion_previous_run_quantities = time_criterion_previous_run_quantities)) {
                                                              time_criterion_previous_run_quantities = time_criterion_previous_run_quantities,
                                                              sampling_overhead_in_leapfrog_steps_built_in_default = sampling_overhead_in_leapfrog_steps_built_in_default)) {
                            fn_run_sampling_timing_probe_in_sampler_environment <- fn_run_sampling_timing_probe
                            environment(fn_run_sampling_timing_probe_in_sampler_environment) <- parent.env(environment())
                            nested_rhat_grouping_for_probe <- tryCatch(fn_nested_rhat_grouping_from_burnin( n_chains_sampling     = n_chains_sampling,
                                                                                                            n_superchains         = n_superchains,
                                                                                                            n_chains_burnin       = n_chains_burnin,
                                                                                                            nuisance_jitter_scale = if (n_nuisance == 0) 0 else nuisance_jitter_scale),
                                                                       error = function(error_object) NULL)
                            ## the start time is taken here so that a probe that fails part-way still has its wall time counted in time_burnin:
                            sampling_timing_probe_call_start_time <- Sys.time()
                            sampling_timing_probe_result <- tryCatch(
                                fn_run_sampling_timing_probe_in_sampler_environment(
                                      time_criterion_settings                         = time_criterion_settings,
                                      ##
                                      Model_type                                      = Model_type,
                                      init_object                                     = init_object,
                                      Model_args_as_Rcpp_List                         = Model_args_as_Rcpp_List,
                                      model_args_list                                 = model_args_list,
                                      Stan_data_list                                  = Stan_data_list,
                                      stan_chunk_size_sampling                        = stan_chunk_size_sampling,
                                      y                                               = y,
                                      ##
                                      theta_main_vectors_all_chains_input_from_R      = theta_main_vectors_all_chains_input_from_R,
                                      theta_nuisance_vectors_all_chains_input_from_R  = theta_nuisance_vectors_all_chains_input_from_R,
                                      n_chains_burnin                                 = n_chains_burnin,
                                      n_params_main                                   = n_params_main,
                                      n_nuisance                                      = n_nuisance,
                                      ##
                                      n_chains_sampling                               = n_chains_sampling,
                                      n_threads_WCP_sampling                          = n_threads_WCP_sampling,
                                      n_threads_WCP_burnin                            = n_threads_WCP_burnin,
                                      num_chunks_sampling                             = num_chunks_sampling,
                                      n_threads_to_restore_after_probe                = if (burnin_TBB_pool_equals_n_chains) n_chains_burnin else n_threads_WCP_burnin * n_chains_burnin,
                                      parallel_method                                 = parallel_method,
                                      use_disk                                        = use_disk,
                                      use_disk_path                                   = use_disk_path,
                                      store_log_lik_trace                             = store_log_lik_trace,
                                      n_nuisance_to_track                             = n_nuisance_to_track,
                                      ##
                                      sample_nuisance                                 = sample_nuisance,
                                      partitioned_HMC                                 = partitioned_HMC,
                                      diffusion_HMC                                   = diffusion_HMC,
                                      diffusion_HMC_integrator                        = diffusion_HMC_integrator,
                                      metric_shape_main                               = metric_shape_main,
                                      force_autodiff                                  = force_autodiff,
                                      force_PartialLog                                = force_PartialLog,
                                      multi_attempts                                  = multi_attempts,
                                      seed                                            = seed,
                                      eps_carry_over                                  = eps_carry_over,
                                      ##
                                      ## "full" (every value of sampling_timing_probe_L_values; the summaries timed by four calls) or "lite" (the smallest and the
                                      ## largest number of leapfrog steps; the summaries timed once, less their fixed set-up cached for this R session):
                                      sampling_timing_probe_mode                      = sampling_timing_probe_mode_for_run,
                                      ##
                                      model_results_template_for_summaries            = list( init_object             = init_object,
                                                                                              test_perm               = test_perm,
                                                                                              test_inv_perm           = test_inv_perm,
                                                                                              LR_main                 = LR_main,
                                                                                              LR_us                   = LR_us,
                                                                                              adapt_delta             = adapt_delta,
                                                                                              burnin_schedule         = resolved_burnin_schedule,
                                                                                              n_burnin                = n_burnin,
                                                                                              metric_type_main        = metric_type_main,
                                                                                              metric_shape_main       = metric_shape_main,
                                                                                              metric_type_nuisance    = metric_type_nuisance,
                                                                                              metric_shape_nuisance   = metric_shape_nuisance,
                                                                                              diffusion_HMC           = diffusion_HMC,
                                                                                              partitioned_HMC         = partitioned_HMC,
                                                                                              n_superchains           = n_superchains,
                                                                                              nested_rhat_grouping    = nested_rhat_grouping_for_probe,
                                                                                              interval_width_main     = interval_width_main,
                                                                                              interval_width_nuisance = interval_width_nuisance,
                                                                                              force_autodiff          = force_autodiff,
                                                                                              force_PartialLog        = force_PartialLog,
                                                                                              multi_attempts          = multi_attempts)),
                                error = function(error_object) {
                                      warning(paste0("sampling timing probe failed: ", conditionMessage(error_object)))
                                      list(status                          = "failed",
                                           error_message                   = conditionMessage(error_object),
                                           sampling_timing_probe_wall_time = as.numeric(difftime(Sys.time(), sampling_timing_probe_call_start_time, units = "secs")))
                                })
                      }
                      time_criterion_sampling_quantities <- fn_resolve_time_criterion_sampling_quantities(
                            time_criterion_settings                = time_criterion_settings,
                            time_criterion_previous_run_quantities = time_criterion_previous_run_quantities,
                            sampling_timing_probe_result           = sampling_timing_probe_result,
                            ## (sampling_overhead_in_leapfrog_steps = "auto" only; NULL = none for this model and sampling configuration):
                            sampling_overhead_in_leapfrog_steps_built_in_default = sampling_overhead_in_leapfrog_steps_built_in_default,
                            n_iter_planned                         = n_iter)
                }
                ##
                ##
                ##      rows of the named parameter families, over which the trajectory criterion computes its statistic. The parameter
                ##      vector is (nuisance, main), so the main names are the last n_params_main BridgeStan names; a family is the name
                ##      before its first "." or "[" (e.g. "beta.1.2.1" -> "beta"). NULL = every row of the adapted block:
                interest_rows <-  NULL
                ## interest_only is its own choice of the criterion's parameters, the third beside "main" (the 44 main parameters) and
                ## "joint" (every parameter): the named subset of the main block, so it runs on the main block's route and is not
                ## combined with "joint":
                if (!is.null(interest_only) && identical(tau_adaptation_block, "joint")) {
                      stop("interest_only is its own choice of the criterion's parameters (a subset of the main block); use it with tau_adaptation_block = 'main', not 'joint'.")
                }
                if (!is.null(interest_only)) {
                      ## (the unconstrained names: one per coordinate; the constrained names are longer than the
                      ##  coordinates for a simplex, which would shift every name after it)
                      # main_parameter_names <-  utils::tail(init_object$bs_main_param_names, n_params_main)
                      main_parameter_names <-  utils::tail(if (!is.null(init_object$bs_main_param_unc_names))
                                                                 init_object$bs_main_param_unc_names else
                                                                 init_object$bs_main_param_names, n_params_main)
                      main_parameter_families <-  sub(pattern = "[.\\[].*$", replacement = "", x = main_parameter_names)
                      ## a family that matches no main parameter (e.g. a misspelt name) stops for a user-supplied
                      ## interest_only; the built-in LC_MVP / LC_MVOP default keeps the families this model has
                      ## (e.g. no C_unc_vec without ordinal tests):
                      families_matching_no_main_parameter <-  setdiff(interest_only, main_parameter_families)
                      if (isTRUE(interest_only_from_built_in_default_for_LCMs)) {
                            interest_only <-  intersect(interest_only, main_parameter_families)
                      } else if (length(families_matching_no_main_parameter) > 0) {
                            stop(paste0("interest_only: ",
                                        paste(families_matching_no_main_parameter, collapse = ", "),
                                        " match(es) no main parameter; the main parameter families are: ",
                                        paste(unique(main_parameter_families), collapse = ", "), "."))
                      }
                      interest_rows <-  which(main_parameter_families %in% interest_only)
                      if (length(interest_rows) == 0) {
                            stop(paste0("interest_only matches no main parameter; the main parameter families are: ",
                                        paste(unique(main_parameter_families), collapse = ", "), "."))
                      }
                      ## message(colourise(paste0("interest_only = ", paste(interest_only, collapse = ", "), ": ", length(interest_rows), " of ",
                      ##                          n_params_main, " main parameters monitored by the LQ_ESSR criterion."), "cyan"))
                      message(colourise(paste0("interest_only = ", paste(interest_only, collapse = ", "), ": ", length(interest_rows), " of ",
                                               n_params_main, " main parameters monitored by the ",
                                               burnin_algorithm, " trajectory criterion."), "cyan"))
                      if (isTRUE(interest_only_from_built_in_default_for_LCMs)) {
                            message(colourise(paste0("interest_only is the built-in default for ",
                                                     init_object$Model_type,
                                                     " (the parameters Se, Sp and prevalence depend on); ",
                                                     "tau_adaptation_block = \"main\" monitors every main ",
                                                     "parameter."),
                                              "cyan"))
                      }
                }
                ##
                ## ---- tau_criteria_monitor_reported_quantities = TRUE (tau adapted): the function of the
                ##      reported quantities, f(theta_main), that every position-based trajectory criterion
                ##      monitors in the main burn-in (init_and_run_burnin_ChESSR): reported_quantities_function,
                ##      or, when it is NULL, the built-in one of LC_MVP / LC_MVOP
                ##      (R_fn_reported_quantities_for_tau_criteria.R). It is built here, from the init_object that
                ##      the main burn-in runs with, i.e. after any test re-ordering of reorder_cols_MVP (the
                ##      re-initialisation after the pre-burnin, above). A model without built-in reported
                ##      quantities needs reported_quantities_function:
                if (isTRUE(tau_criteria_monitor_reported_quantities) && !isTRUE(manual_tau) &&
                    is.null(reported_quantities_function)) {
                      reported_quantities_function <-  fn_reported_quantities_function_for_built_in_LCMs(
                            init_object = init_object,
                            n_params_main = n_params_main)
                      if (is.null(reported_quantities_function)) {
                            stop(paste0("tau_criteria_monitor_reported_quantities = TRUE: Model_type \"",
                                        init_object$Model_type, "\" has no built-in reported quantities (only ",
                                        "LC_MVP and LC_MVOP have them). Give reported_quantities_function: a ",
                                        "function of a numeric matrix of main parameters (n_params_main = ",
                                        n_params_main, " rows in the sampler's main-block order, unconstrained ",
                                        "scale, x chains) that returns a numeric matrix of the reported ",
                                        "quantities (one row per quantity, with rownames, x chains)."))
                      }
                }
                bulk_local_tuner_names <-  NULL
                bulk_local_tuner_cost_contract <-  NULL
                if (!is.null(bulk_local_tuner)) {
                      bulk_local_tuner <-  fn_bulk_local_tuner_settings(bulk_local_tuner)
                      if (isTRUE(tau_shrink_on_divergence) ||
                          !identical(tau_adaptation_scheme, "adam_decay")) {
                            stop("bulk_local_tuner excludes divergence shrink and probe/averaging tau schemes.")
                      }
                      if (!identical(burnin_algorithm, "ESJD") || isTRUE(partitioned_HMC) ||
                          !isTRUE(randomize_tau_burnin) || !isTRUE(randomize_tau_sampling) ||
                          !identical(tau_jitter_burnin, "uniform") || isTRUE(manual_tau) ||
                          isTRUE(tau_if_manual_in_L_units) || !identical(tau_sampling_scale, "none")) {
                            stop(paste0("bulk_local_tuner requires ESJD, nonpartitioned HMC, uniform jitter, ",
                                        "adaptive tau and tau_sampling_scale = 'none'."))
                      }
                      bulk_local_tuner_names <-  utils::tail(init_object$bs_main_param_names, n_params_main)
                      if (length(bulk_local_tuner_names) != n_params_main ||
                          anyNA(bulk_local_tuner_names) || anyDuplicated(bulk_local_tuner_names)) {
                            stop("bulk_local_tuner requires distinct original main parameter names.")
                      }
                      bulk_local_tuner_families <-  sub("[.\\[].*$", "", bulk_local_tuner_names)
                      if (any(!bulk_local_tuner$parameter_families %in% bulk_local_tuner_families)) {
                            stop("bulk_local_tuner contains an unknown main parameter family.")
                      }
                      bulk_local_tuner_joint_diffusion <-  isTRUE(sample_nuisance) && n_nuisance > 0 &&
                                                          isTRUE(diffusion_HMC)
                      bulk_local_tuner_integrator <-  if (is.null(diffusion_HMC_integrator)) {
                            "kick_flow_kick"
                      } else diffusion_HMC_integrator
                      bulk_local_tuner_cost_contract <-  list(
                            endpoint_evaluations_per_iteration = if (bulk_local_tuner_joint_diffusion &&
                                  identical(bulk_local_tuner_integrator, "kick_flow_kick")) 0 else 1,
                            start_of_call_evaluations_per_chain = if (bulk_local_tuner_joint_diffusion) 1 else 0,
                            n_iter_sampling = n_iter)
                }
                burnin_object <-                 fn_burnin(  init_object = init_object,
                                                             ## debug = TRUE: the burn-in keeps its debug record
                                                             ## (R_fn_init_and_run_burnin_CHESS.R):
                                                             debug = debug,
                                                             ## ESS of a burnin_algorithm:
                                                             n_iter_sampling_per_chain = n_iter,
                                                             bulk_local_tuner = bulk_local_tuner,
                                                             bulk_local_tuner_names = bulk_local_tuner_names,
                                                             bulk_local_tuner_cost_contract =
                                                                   bulk_local_tuner_cost_contract,
                                                             debug_burnin_timing = debug_burnin_timing,
##
                        ## ---- burn-in adaptation settings:
##
                        tau_initial_moments_window_fraction = tau_initial_moments_window_fraction,
                        tau_initial_moments_window_min_iter = tau_initial_moments_window_min_iter,
                        tau_adam_beta1 = tau_adam_beta1,
                        tau_adam_beta2 = tau_adam_beta2,
                        tau_adam_epsilon = tau_adam_epsilon,
                        eps_adam_beta1 = eps_adam_beta1,
                        eps_adam_beta2 = eps_adam_beta2,
                        eps_adam_epsilon = eps_adam_epsilon,
                        tau_adaptation_scheme = tau_adaptation_scheme,
                        tau_learning_rate_restart_at_metric_end = tau_learning_rate_restart_at_metric_end,
                        eps_adam_reset_at_metric_end = eps_adam_reset_at_metric_end,
                        eps_learning_rate_restart_at_metric_end = eps_learning_rate_restart_at_metric_end,
                        rate_criterion_cost_offset_steps = rate_criterion_cost_offset_steps,
                        pooled_metric_warm_up = pooled_metric_warm_up,
                        pooled_metric_soft_full_weight_draws = pooled_metric_soft_full_weight_draws,
                        pooled_metric_weight_k_draws = pooled_metric_weight_k_draws,
                        pooled_metric_signal_to_noise_centred = pooled_metric_signal_to_noise_centred,
                        metric_update_rescales_eps_and_tau = metric_update_rescales_eps_and_tau,
                        learning_rate_initial_n_tau_updates_held_from_tau_handover =
                              learning_rate_initial_n_tau_updates_held_from_tau_handover,
                        resident_burnin = resident_burnin,
                        resident_burnin_joint_block = resident_burnin_joint_block,
                        tau_initial_resident_moments = tau_initial_resident_moments,
                        record_burnin_metric_nuisance_variance_quantiles_history =
                              record_burnin_metric_nuisance_variance_quantiles_history,
                        burnin_tau_set_to_eps_so_every_trajectory_is_one_leapfrog_step =
                              burnin_tau_set_to_eps_so_every_trajectory_is_one_leapfrog_step,
                        Hessian_metric_offdiagonal_shrinkage = Hessian_metric_offdiagonal_shrinkage,
                                                             ##
                                                             ## CHESSR_time / SNAPER_time only (NULL otherwise):
                                                             time_criterion_sampling_quantities = time_criterion_sampling_quantities,
                                                             ##
                                                             Model_args_as_Rcpp_List = Model_args_as_Rcpp_List,
                                                             ##
                                                             Model_type = Model_type,
                                                             ##
                                                             n_chains_burnin = n_chains_burnin,
                                                             ##
                                                             theta_main_vectors_all_chains_input_from_R = theta_main_vectors_all_chains_input_from_R,
                                                             theta_nuisance_vectors_all_chains_input_from_R = theta_nuisance_vectors_all_chains_input_from_R,
                                                             ##
                                                             parallel_method = "RcppParallel", ## no OpenMP for burnin yet (only sampling!)
                                                             ##
                                                             Stan_data_list = Stan_data_list,
                                                             model_args_list = model_args_list,
                                                             ##
                                                             sample_nuisance = sample_nuisance,
                                                             n_nuisance_override = n_nuisance_override,
                                                             ##
                                                             seed = seed,
                                                             n_burnin = n_burnin,
                                                             n_adapt = n_adapt,
                                                             gap = gap,
                                                             ##
                                                             adapt_delta = adapt_delta,
                                                             LR_main = LR_main,
                                                             LR_us = LR_us,
                                                             tau_mult = tau_mult,
                                                             ##
                                                             # tau_initial = tau_initial,
                                                             # tau_initial = pre_burnin_object$tau_main,
                                                              tau_initial = tau_initial,
                                                             ##
                                                             manual_tau = manual_tau,
                                                             tau_if_manual = tau_if_manual,
                                                             tau_if_manual_in_L_units = tau_if_manual_in_L_units,
                                                             ##
                                                             tau_weight_by_p_jump = tau_weight_by_p_jump,
                                                             tau_ramp = tau_ramp,
                                                             eps_reinit_at_ChEES_handover = eps_reinit_at_ChEES_handover,
                                                             eps_acceptance_mean = eps_acceptance_mean,
                                                             tau_shrink_on_divergence        = tau_shrink_on_divergence,
                                                             tau_shrink_factor               = tau_shrink_factor,
                                                             tau_shrink_min_divergent_chains = tau_shrink_min_divergent_chains,
                                                             theta_hat_us_rule = theta_hat_us_rule,
                                                             theta_hat_us_freeze_iter = theta_hat_us_freeze_iter,
                                                             burnin_schedule = burnin_schedule,
                                                             metric_adaptation_end_iter = metric_adaptation_end_iter,
                                                             share_tau_ii_across_chains_in_burnin = share_tau_ii_across_chains_in_burnin,
                                                             randomize_tau_burnin = randomize_tau_burnin,
                                                             tau_gradient_estimator = tau_gradient_estimator,
                                                             tau_cost_exponent = tau_cost_exponent,
                                                             esjd_jump_power = esjd_jump_power,
                                                             interest_rows = interest_rows,
                                                             tau_criteria_monitor_reported_quantities =
                                                                   tau_criteria_monitor_reported_quantities,
                                                             reported_quantities_function =
                                                                   reported_quantities_function,
                                                ## (indented less: the argument name is long)
                                                reported_quantities_centre_and_scale_from_last_n_iterations =
                                                      reported_quantities_centre_and_scale_from_last_n_iterations,
                                                             tau_jitter_burnin = tau_jitter_burnin,
                                                             tau_adaptation_block = tau_adaptation_block_as_given,
                                                             SpESS_R_max_rows_fitted_per_update =
                                                                   SpESS_R_max_rows_fitted_per_update,
                                                             SpESS_R_bins_decay_lazily =
                                                                   SpESS_R_bins_decay_lazily,
                                                             SpESS_R_soft_minimum_power =
                                                                   SpESS_R_soft_minimum_power,
                                                             SpESS_R_first_update_jumps_to_the_maximiser =
                                                                   SpESS_R_first_update_jumps_to_the_maximiser,
                                                             SpESS_R_doubling_needs_evidence_on_updates_in_a_row =
                                                              SpESS_R_doubling_needs_evidence_on_updates_in_a_row,
                                                             SpESS_R_statistics_in_parameter_coordinates =
                                                                   SpESS_R_statistics_in_parameter_coordinates,
                                                             SpESS_R_sampling_tau_from_last_fraction_of_updates =
                                                              SpESS_R_sampling_tau_from_last_fraction_of_updates,
                                          ## (indented less: the argument name is long)
                                          SpESS_R_sampling_tau_from_maximiser_over_last_fraction_of_updates =
                                                SpESS_R_sampling_tau_from_maximiser_over_last_fraction_of_updates,
                                                             SpESS_R_bin_forgetting_factor =
                                                                   SpESS_R_bin_forgetting_factor,
                                                             SpESS_R_normalising_variances_from_last_n_updates =
                                                              SpESS_R_normalising_variances_from_last_n_updates,
                                                             SpESS_R_linear_ESS_fraction_multiplier =
                                                              SpESS_R_linear_ESS_fraction_multiplier,
                                                             SpESS_R_quadratic_ESS_fraction_multiplier =
                                                              SpESS_R_quadratic_ESS_fraction_multiplier,
                                                             SpESS_R_quadratic_part_normal_mode_spectral_fit =
                                                              SpESS_R_quadratic_part_normal_mode_spectral_fit,
                                          ## (indented less: the argument name is long)
                                          SpESS_R_quadratic_part_from_linear_spectrum_with_pair_modes =
                                                SpESS_R_quadratic_part_from_linear_spectrum_with_pair_modes,
                                          ## (indented less: the argument name is long)
                                          SpESS_R_target_tau_is_shortest_within_fraction_of_best_objective =
                                                SpESS_R_target_tau_is_shortest_within_fraction_of_best_objective,
                                  ## (indented less: the argument name is long)
                                  SpESS_R_linear_part_measured_persistence_from_last_n_burnin_iterations =
                                        SpESS_R_linear_part_measured_persistence_from_last_n_burnin_iterations,
                                                             ##
                                                             burnin_algorithm = burnin_algorithm,
                                                             diffusion_HMC = diffusion_HMC,
                                                             diffusion_HMC_integrator = diffusion_HMC_integrator,
                                                             partitioned_HMC = partitioned_HMC,
                                                             ##
                                                             clip_iter = clip_iter,
                                                             clip_iter_tau = clip_iter_tau,
                                                             ##
                                                             n_refresh = n_refresh,
                                                             use_proposed = use_proposed,
                                                             ##
                                                             beta1_adam = beta1_adam,
                                                             beta2_adam = beta2_adam,
                                                             eps_adam = eps_adam,
                                                             ##
                                                             force_autodiff = force_autodiff,
                                                             force_PartialLog = force_PartialLog,
                                                             multi_attempts = multi_attempts,
                                                             ##
                                                             force_autodiff_for_metric = force_autodiff_for_metric,
                                                             force_PartialLog_for_metric = force_PartialLog_for_metric,
                                                             force_multi_attempts_for_metric = force_multi_attempts_for_metric,
                                                             ##
                                                             metric_type_main = metric_type_main,
                                                             metric_shape_main = metric_shape_main,
                                                             ratio_M_main = ratio_M_main,
                                                             interval_width_main = interval_width_main,
                                                             ##
                                                             M_decay_type = M_decay_type,
                                                             M_decay_power = M_decay_power,
                                                             M_decay_scale = M_decay_scale,
                                                             ##
                                                             metric_type_nuisance = metric_type_nuisance,
                                                             metric_shape_nuisance = metric_shape_nuisance,
                                                             ratio_M_nuisance = ratio_M_nuisance,
                                                             interval_width_nuisance = interval_width_nuisance,
                                                             ##
                                                             max_tau_main = max_tau_main,
                                                             max_tau_nuisance = max_tau_nuisance,
                                                             ##
                                                             max_eps_main = max_eps_main,
                                                             max_eps_nuisance = max_eps_nuisance,
                                                             ##
                                                             max_L = max_L,
                                                             ##
                                                             n_nuisance_to_track = n_nuisance_to_track,
                                                             ##
                                                             n_threads_WCP = n_threads_WCP_burnin,
                                                             ##
                                                             time_pre_burnin = time_pre_burnin,
                                                             ##
                                                             metric_start_iter = metric_start_iter,
                                                             ##
                                                             eps_init = NULL,
                                                             ##
                                                             ## NULL unless eps_reinit_after_pre_burnin = FALSE (see above):
                                                             eps_carry_over = eps_carry_over,
                                                             ##
                                                             learning_rate_initial = learning_rate_initial,
                                                             learning_rate_initial_iter = learning_rate_initial_iter,
                                                             ##
                                                             eps_initial = eps_initial,
                                                             eps_initial_iter = eps_initial_iter,
                                                             ##
                                                             metric_pooled_window_resets         = metric_pooled_window_resets,
                                                             metric_pooled_offdiagonal_shrinkage = metric_pooled_offdiagonal_shrinkage,
                                                             ##
                                                             metric_estimator = metric_estimator)
                                                            
                ##
                ## ---- CHESSR_time / SNAPER_time: the sampling timing probe ran for this burn-in, so its wall time is part of the burn-in
                ##      time (time_burnin, and burnin_object$time_burnin, which the summaries read). The time without it is kept:
                ##
                time_burnin_without_sampling_timing_probe <- burnin_object$time_burnin
                sampling_timing_probe_fraction_of_burnin  <- NA_real_
                if (!is.null(sampling_timing_probe_result) && is.finite(sampling_timing_probe_result$sampling_timing_probe_wall_time)) {
                      burnin_object$time_burnin <- burnin_object$time_burnin + sampling_timing_probe_result$sampling_timing_probe_wall_time
                      sampling_timing_probe_fraction_of_burnin <- sampling_timing_probe_result$sampling_timing_probe_wall_time /
                                                                  (time_burnin_without_sampling_timing_probe - time_pre_burnin)
                      message(colourise(paste0("sampling timing probe wall time = ", signif(sampling_timing_probe_result$sampling_timing_probe_wall_time, 4),
                                               " s = ", signif(100 * sampling_timing_probe_fraction_of_burnin, 3), "% of the main burn-in (included in time_burnin)"),
                                        "cyan"))
                      if (is.finite(sampling_timing_probe_fraction_of_burnin) && sampling_timing_probe_fraction_of_burnin > 0.05) {
                            warning(paste0("The sampling timing probe took ", signif(100 * sampling_timing_probe_fraction_of_burnin, 3),
                                           "% of the main burn-in; a previous saved run (time_criterion_previous_run_path), fewer probe iterations ",
                                           "or sampling_timing_probe_max_wall_time lowers this."))
                      }
                }
                
                {

                          theta_main_vectors_all_chains_input_from_R <- burnin_object$theta_main_vectors_all_chains_input_from_R  # inits stored here
                          theta_nuisance_vectors_all_chains_input_from_R <- burnin_object$theta_us_vectors_all_chains_input_from_R
        
                          Model_args_as_Rcpp_List <-  burnin_object$Model_args_as_Rcpp_List
                          EHMC_args_as_Rcpp_List  <-  burnin_object$EHMC_args_as_Rcpp_List
                          EHMC_args_as_Rcpp_List$max_L <-  max_L
                          ## The sampling phase has its own trajectory randomisation setting.
                          EHMC_args_as_Rcpp_List$randomize_tau <-  randomize_tau_sampling
                          EHMC_args_as_Rcpp_List$share_tau_ii_across_chains <-  FALSE
                          EHMC_args_as_Rcpp_List$use_given_tau_main_ii <-  FALSE
                          ## The burn-in jitter sequence is deliberately cleared at the sampling handover; the sampling phase keeps
                          ## its established uniform RNG path.
                          tau_jitter_burnin_for_sampling <-  "uniform"
                          if (identical(tau_jitter_burnin, "halton")) {
                              EHMC_args_as_Rcpp_List <-  fn_tau_jitter_burnin_reset(EHMC_args_as_Rcpp_List)
                          }
                          ##
                          ## ---- tau_sampling_scale: rescale the adapted tau ONCE for the randomised sampling phase.
                          ##      Burn-in adapts tau with a FIXED trajectory length (randomize_tau_burnin = FALSE, deliberately:
                          ##      the burn-in chains run in lockstep, so per-chain jitter would make every iteration wait for
                          ##      the longest chain), but sampling draws tau_ii ~ U(0, 2 * tau_bar), whose optimal mean differs
                          ##      from the fixed-length optimum. Only applied when tau was adapted (manual_tau = FALSE) with a
                          ##      fixed length and sampling is randomised; otherwise the factor is 1. eps is NOT re-initialised.
                          ##      Experimental heuristic (Gaussian analysis, Hoffman et al. 2021); see docs/adaptation-notes.md.
                          ##
                          {
                                tau_sampling_scale_applicable <- !isTRUE(manual_tau) &&
                                                                 !isTRUE(randomize_tau_burnin) &&
                                                                 isTRUE(randomize_tau_sampling)
                                tau_sampling_scale_not_applied_reason <- if (isTRUE(manual_tau)) "tau_not_adapted_manual_tau" else
                                                                         if (isTRUE(randomize_tau_burnin)) "tau_adapted_under_jitter" else
                                                                         if (!isTRUE(randomize_tau_sampling)) "sampling_not_randomised" else NA_character_
                                ##
                                tau_sampling_scale_effective <- if (!tau_sampling_scale_applicable || identical(tau_sampling_scale, "none")) "none" else
                                                                if (identical(tau_sampling_scale, "gaussian_matched")) "gaussian_matched" else
                                                                as.character(tau_sampling_scale)
                                tau_sampling_scale_factor <- if (identical(tau_sampling_scale_effective, "none")) 1 else
                                                             if (identical(tau_sampling_scale_effective, "gaussian_matched"))
                                                                 fn_tau_sampling_scale_gaussian_factor(burnin_algorithm = burnin_algorithm) else
                                                             as.numeric(tau_sampling_scale)
                                ##
                                tau_main_before_sampling_scale <- EHMC_args_as_Rcpp_List$tau_main
                                tau_us_before_sampling_scale <- EHMC_args_as_Rcpp_List$tau_us
                                ##
                                if (tau_sampling_scale_factor != 1) {
                                      EHMC_args_as_Rcpp_List$tau_main <- min(max_tau_main,
                                                                             tau_sampling_scale_factor * tau_main_before_sampling_scale)
                                      ## same rule as the burn-in: the joint sampler keeps tau_us = tau_main.
                                      EHMC_args_as_Rcpp_List$tau_us <- if (isTRUE(partitioned_HMC))
                                          min(max_tau_nuisance, tau_sampling_scale_factor * tau_us_before_sampling_scale) else
                                          EHMC_args_as_Rcpp_List$tau_main
                                      ##
                                      ## create_summary_and_traces() computes the sampling L (and so the gradient counts behind
                                      ## ESS / grad) from burnin_object$EHMC_args_as_Rcpp_List, so it must carry the tau that
                                      ## sampling actually used. The burn-in end value stays in burnin_object$tau_main and in
                                      ## tau_main_before_sampling_scale below.
                                      burnin_object$EHMC_args_as_Rcpp_List$tau_main <- EHMC_args_as_Rcpp_List$tau_main
                                      burnin_object$EHMC_args_as_Rcpp_List$tau_us <- EHMC_args_as_Rcpp_List$tau_us
                                }
                                ##
                                tau_main_after_sampling_scale <- EHMC_args_as_Rcpp_List$tau_main
                                tau_us_after_sampling_scale <- EHMC_args_as_Rcpp_List$tau_us
                                ##
                                if (tau_main_after_sampling_scale != tau_sampling_scale_factor * tau_main_before_sampling_scale) {
                                      warning(paste0("tau_sampling_scale: tau_main capped at max_tau_main = ", max_tau_main,
                                                     "; the applied ratio is ", signif(tau_main_after_sampling_scale / tau_main_before_sampling_scale, 6),
                                                     ", not the factor ", signif(tau_sampling_scale_factor, 6), "."))
                                }
                                message(paste0("tau_sampling_scale: requested = ", as.character(tau_sampling_scale),
                                               " | effective = ", tau_sampling_scale_effective,
                                               " | factor = ", signif(tau_sampling_scale_factor, 6),
                                               if (!is.na(tau_sampling_scale_not_applied_reason) && !identical(tau_sampling_scale, "none"))
                                                   paste0(" (not applied: ", tau_sampling_scale_not_applied_reason, ")") else "",
                                               " | tau_main ", signif(tau_main_before_sampling_scale, 6),
                                               " -> ", signif(tau_main_after_sampling_scale, 6)))
                          }
                          EHMC_Metric_as_Rcpp_List <- burnin_object$EHMC_Metric_as_Rcpp_List
                          EHMC_burnin_as_Rcpp_List <- burnin_object$EHMC_burnin_as_Rcpp_List
        
                          time_burnin <- burnin_object$time_burnin
                          # time_burnin <- time_burnin + time_pre_burnin
                          
                          n_chains_burnin <-  burnin_object$n_chains_burnin
                          n_burnin <-  burnin_object$n_burnin

                }

                {
                          
                          if (is.null(n_superchains)) { 
                            n_superchains <- n_chains_sampling
                          }
                          if (n_superchains > n_chains_sampling) {
                            n_superchains <- n_chains_sampling
                          }
                          
                          print(paste("Start of post_burnin_prep_inits fn"))
                          # post_burnin_prep_inits <- R_fn_post_burnin_prep_for_sampling( 
                          #                           n_chains_burnin = n_chains_burnin,
                          #                           n_chains_sampling = n_chains_sampling,
                          #                           n_superchains = n_superchains,
                          #                           n_params_main = n_params_main,
                          #                           n_nuisance = n_nuisance,
                          #                           theta_main_vectors_all_chains_input_from_R,
                          #                           theta_nuisance_vectors_all_chains_input_from_R)
                          ## nuisance_jitter_scale: each sampling chain starts from its superchain's burn-in
                          ## nuisance endpoint PLUS this multiple of the per-coordinate spread of the burn-in
                          ## endpoints. Without it, n_chains_sampling / n_chains_burnin chains share one exact
                          ## nuisance vector, so a badly-placed burn-in chain is inherited by all of them.
                          ## 0 reproduces the old (duplicated) behaviour.
                          post_burnin_prep_inits <- cpp_fn_post_burnin_prep_for_sampling(  n_chains_burnin = n_chains_burnin,
                                                                                           n_chains_sampling = n_chains_sampling,
                                                                                           n_superchains = n_superchains,
                                                                                           n_params_main = n_params_main,
                                                                                           n_nuisance = n_nuisance,
                                                                                           theta_main_in = theta_main_vectors_all_chains_input_from_R,
                                                                                           theta_us_in = theta_nuisance_vectors_all_chains_input_from_R,
                                                                                           nuisance_jitter_scale = nuisance_jitter_scale,
                                                                                           seed = seed)
                          print(paste("End of post_burnin_prep_inits fn"))
                          ## Record the actual shared starting states, including repeated endpoints across nominal superchains.
                          ## Nonzero per-chain nuisance jitter breaks shared FULL states; do not claim valid nesting then.
                          nested_rhat_grouping <-  fn_nested_rhat_grouping_from_burnin(
                              n_chains_sampling = n_chains_sampling,
                              n_superchains = n_superchains,
                              n_chains_burnin = n_chains_burnin,
                              nuisance_jitter_scale = if (n_nuisance == 0) 0 else nuisance_jitter_scale)
        
                          theta_main_vectors_all_chains_input_from_R <- post_burnin_prep_inits$theta_main_vectors_all_chains_input_from_R
                          # theta_us_vectors_all_chains_input_from_R   <- post_burnin_prep_inits$theta_us_vectors_all_chains_input_from_R
                          theta_nuisance_vectors_all_chains_input_from_R <- post_burnin_prep_inits$theta_us_vectors_all_chains_input_from_R

                }

                # EHMC_Metric_as_Rcpp_List$M_inv_dense_main      <- S_post_pd
                # EHMC_Metric_as_Rcpp_List$M_dense_main          <- Rcpp_solve(S_post_pd)
                # EHMC_Metric_as_Rcpp_List$M_inv_dense_main_chol <- Rcpp_Chol(S_post_pd)
                ##
                # EHMC_args_as_Rcpp_List$eps_main <- 0.5/4
                
                cat("SAMPLING WILL USE: eps_main =", EHMC_args_as_Rcpp_List$eps_main,
                    "| tau_main =", EHMC_args_as_Rcpp_List$tau_main,
                    "| implied L =", ceiling(EHMC_args_as_Rcpp_List$tau_main / EHMC_args_as_Rcpp_List$eps_main), "\n")
                
                # cat("metric check:",
                #     max(abs(EHMC_Metric_as_Rcpp_List$M_dense_main %*% EHMC_Metric_as_Rcpp_List$M_inv_dense_main - diag(43))),
                #     max(abs(EHMC_Metric_as_Rcpp_List$M_inv_dense_main_chol %*% t(EHMC_Metric_as_Rcpp_List$M_inv_dense_main_chol) - EHMC_Metric_as_Rcpp_List$M_inv_dense_main)), "\n")
                
                {

                          # gc()
                          ##
                          ## External Stan models use the chunk_size in their JSON data, not this built-in argument matrix.
                          ## Writing [4] into their absent Model_args_ints creates a vector and the native converter rejects it.
                          if (Model_type != "Stan") {
                            num_chunks_used_in_burnin <- Model_args_as_Rcpp_List$Model_args_ints[4]
                            ##
                            if (is.null(num_chunks_sampling)) {
                              Model_args_as_Rcpp_List$Model_args_ints[4] <- model_args_list$num_chunks
                            } else {
                              # model_args_list$num_chunks <- num_chunks_sampling
                              Model_args_as_Rcpp_List$Model_args_ints[4] <- num_chunks_sampling
                            }
                            num_chunks_used_in_sampling <- Model_args_as_Rcpp_List$Model_args_ints[4]
                          }
                          ##
                          ## ---- Re-lay-out every nuisance-indexed quantity for the sampling chunking.
                          ##
                          ## The built-in C++ log densities store the nuisance vector chunk by chunk (see
                          ## fn_nuisance_chunk_layout_positions), so entry i belongs to a DIFFERENT (observation,
                          ## test) when the number of chunks changes. Without this, a burn-in with 10 chunks handed
                          ## to a sampler with 25 gives every chain latent variables assigned to the wrong
                          ## observations and a scrambled diffusion centre theta_hat_us for the whole sampling
                          ## phase: mass divergences at the start of sampling and chains stuck for good.
                          ## External Stan models keep their own (chunk-independent) layout and are left alone.
                          ##
                          if ((Model_type != "Stan") && (n_nuisance > 0) &&
                              (num_chunks_used_in_sampling != num_chunks_used_in_burnin)) {
                                N_for_remap       <- init_object$model_args_list$N
                                n_tests_for_remap <- init_object$model_args_list$n_tests
                                if (is.null(N_for_remap) || is.null(n_tests_for_remap) || (N_for_remap * n_tests_for_remap != n_nuisance)) {
                                  stop("Cannot re-lay-out the nuisance vector for num_chunks_sampling != num_chunks_burnin: ",
                                       "expected n_nuisance = N * n_tests. Use the same chunk count for burn-in and sampling.")
                                }
                                nuisance_chunk_remap_index <- BayesMVP:::fn_nuisance_chunk_remap_index(
                                                                                             N             = N_for_remap,
                                                                                             n_tests       = n_tests_for_remap,
                                                                                             n_chunks_from = num_chunks_used_in_burnin,
                                                                                             n_chunks_to   = num_chunks_used_in_sampling,
                                                                                             vect_type     = Model_args_as_Rcpp_List$Model_args_strings[1])
                                ##
                                ## Re-orders the rows of anything indexed by nuisance coordinate (vector of length
                                ## n_nuisance, or matrix with n_nuisance rows); returns anything else unchanged:
                                fn_remap_nuisance_indexed_rows <- function(object_to_remap) {
                                  if (is.numeric(object_to_remap) && is.matrix(object_to_remap) && nrow(object_to_remap) == n_nuisance) {
                                    return(object_to_remap[nuisance_chunk_remap_index, , drop = FALSE])
                                  }
                                  if (is.numeric(object_to_remap) && !is.matrix(object_to_remap) && length(object_to_remap) == n_nuisance) {
                                    return(object_to_remap[nuisance_chunk_remap_index])
                                  }
                                  object_to_remap
                                }
                                ##
                                theta_nuisance_vectors_all_chains_input_from_R <- fn_remap_nuisance_indexed_rows(theta_nuisance_vectors_all_chains_input_from_R)
                                ##
                                ## Only per-coordinate VALUES move. Position lists (index_us, index_main, ...) say
                                ## WHERE the nuisance block sits in the parameter vector, not which entry is which,
                                ## so they must not be permuted.
                                names_of_remapped_fields <- character(0)
                                for (metric_field_name in names(EHMC_Metric_as_Rcpp_List)) {
                                  if (grepl("^index", metric_field_name) || is.integer(EHMC_Metric_as_Rcpp_List[[metric_field_name]])) next
                                  metric_field_remapped <- fn_remap_nuisance_indexed_rows(EHMC_Metric_as_Rcpp_List[[metric_field_name]])
                                  if (!identical(metric_field_remapped, EHMC_Metric_as_Rcpp_List[[metric_field_name]])) {
                                    names_of_remapped_fields <- c(names_of_remapped_fields, paste0("Metric$", metric_field_name))
                                  }
                                  EHMC_Metric_as_Rcpp_List[[metric_field_name]] <- metric_field_remapped
                                }
                                for (burnin_field_name in names(EHMC_burnin_as_Rcpp_List)) {
                                  if (grepl("^index", burnin_field_name) || is.integer(EHMC_burnin_as_Rcpp_List[[burnin_field_name]])) next
                                  burnin_field_remapped <- fn_remap_nuisance_indexed_rows(EHMC_burnin_as_Rcpp_List[[burnin_field_name]])
                                  if (!identical(burnin_field_remapped, EHMC_burnin_as_Rcpp_List[[burnin_field_name]])) {
                                    names_of_remapped_fields <- c(names_of_remapped_fields, paste0("burnin$", burnin_field_name))
                                  }
                                  EHMC_burnin_as_Rcpp_List[[burnin_field_name]] <- burnin_field_remapped
                                }
                                cat("nuisance layout remapped for sampling: num_chunks ", num_chunks_used_in_burnin, " -> ",
                                    num_chunks_used_in_sampling, " (inits + ", paste(names_of_remapped_fields, collapse = ", "), ")\n", sep = "")
                          }
                          ##
                          ## ---- the dense main metric made exactly symmetric before it is frozen: the inverse metric
                          ##      (the covariance estimate) is averaged with its transpose, and the metric and the
                          ##      Cholesky factor are rebuilt from it, so that the three agree (the asymmetry it had,
                          ##      relative to its largest entry, is printed):
                          ##
                          if (identical(EHMC_Metric_as_Rcpp_List$metric_shape_main, "dense") &&
                              is.matrix(EHMC_Metric_as_Rcpp_List$M_inv_dense_main)) {
                                M_inv_dense_main_as_estimated <-  EHMC_Metric_as_Rcpp_List$M_inv_dense_main
                                relative_asymmetry <-  max(abs(M_inv_dense_main_as_estimated - t(M_inv_dense_main_as_estimated))) /
                                                       max(abs(M_inv_dense_main_as_estimated))
                                M_inv_dense_main_symmetric <-  0.5 * (M_inv_dense_main_as_estimated + t(M_inv_dense_main_as_estimated))
                                M_dense_main_symmetric <-  solve(M_inv_dense_main_symmetric)
                                M_dense_main_symmetric <-  0.5 * (M_dense_main_symmetric + t(M_dense_main_symmetric))
                                EHMC_Metric_as_Rcpp_List$M_inv_dense_main <-  M_inv_dense_main_symmetric
                                EHMC_Metric_as_Rcpp_List$M_dense_main <-  M_dense_main_symmetric
                                EHMC_Metric_as_Rcpp_List$M_inv_dense_main_chol <-  t(base::chol(M_inv_dense_main_symmetric))
                                cat("dense main metric made symmetric before sampling: relative asymmetry it had = ",
                                    format(relative_asymmetry, digits = 3), "\n", sep = "")
                          }
                          ##
                          ## ---- validate the frozen sampling state and geometry before the timer/native call:
                          ##
                          R_fn_validate_frozen_sampling_geometry(
                              EHMC_args_as_Rcpp_List = EHMC_args_as_Rcpp_List,
                              EHMC_Metric_as_Rcpp_List = EHMC_Metric_as_Rcpp_List,
                              theta_main = theta_main_vectors_all_chains_input_from_R,
                              theta_nuisance = theta_nuisance_vectors_all_chains_input_from_R,
                              n_chains_sampling = n_chains_sampling,
                              sample_nuisance = sample_nuisance,
                              partitioned_HMC = partitioned_HMC,
                              Model_args_as_Rcpp_List = Model_args_as_Rcpp_List)
                          ##
                          ## ---- keep the per-chain log-lik trace during sampling? (read by the C++ as an optional list element):
                          ##
                          if (!store_log_lik_trace && (parallel_method == "OpenMP")) {
                              stop("store_log_lik_trace = FALSE is only implemented for parallel_method = 'RcppParallel'.")
                          }
                EHMC_args_as_Rcpp_List$store_log_lik_trace <- store_log_lik_trace
                EHMC_args_as_Rcpp_List$record_kinetic_energy_tau_derivatives <- FALSE
                ## debug = TRUE: the sampler records each iteration's own trajectory length and main-parameter
                ## proposal (sampling_diagnostics; read by the C++ as an optional list element, absent otherwise):
                if (isTRUE(debug)) EHMC_args_as_Rcpp_List$debug <- TRUE
                          ##
                          tictoc::tic("post-burnin timer")
                          ##
                          if (Model_type == "Stan" && !is.null(stan_chunk_size_sampling)) {
                              Stan_data_list$chunk_size <- stan_chunk_size_sampling
                              ## Sampling workers load this phase's content-addressed data file when constructed.
                              Model_args_as_Rcpp_List$json_file_path <- convert_stan_data_list_to_JSON(
                                  stan_data_list = Stan_data_list,
                                  pkg_data_dir = dirname(init_object$json_file_path))
                          }
                          ##
                          if (Model_type != "Stan") {
                              Model_args_as_Rcpp_List$model_so_file <- "none"
                              Model_args_as_Rcpp_List$json_file_path <- "none"
                          }
        
                          Model_args_as_Rcpp_List$n_nuisance
                          #  RcppParallel::setThreadOptions(numThreads = n_chains_sampling); #### BOOKMARK
                          RcppParallel::setThreadOptions(numThreads = n_chains_sampling * n_threads_WCP_sampling);
                          
                          # if (parallel_method == "OpenMP") { 
                          #   fn <- Rcpp_fn_OpenMP_EHMC_sampling
                          # } else { ###  use RcppParallel
                          #   fn <- Rcpp_fn_RcppParallel_EHMC_sampling
                          # }
                           
                          if (Model_type == "Stan") {
                             N <- 1000
                             latent_results <- rlogis(n = N, location =  0.0)
                             y <- ifelse(latent_results > 0, 1, 0)
                             y <- matrix(data = c(y), ncol = 1)
                          } 
                          
                           ### Call C++ parallel sampling function
                          print(paste("Start of sampling"))
                          ## the Halton sampling jitter is switched on in the C++ runtime for this sampling call
                          ## only (cleared after the call and on exit). The flag is a static of the runtime
                          ## headers, compiled into each package's own shared object (NicoStan.so; BayesMVP.so,
                          ## whose main.cpp is generated from the same native_api.cpp.in), so it is set through
                          ## the setter of the package whose sampling function runs this fit, looked up in that
                          ## function's namespace (for BayesMVP's backend: BayesMVP's, not NicoStan's); a build
                          ## without the setter stops here:
                          fn_set_sampler_halton_flag <-  NULL
                          if (identical(tau_jitter_sampling_type, "halton")) {
                                sampling_function_in_use <-  if (parallel_method == "OpenMP") {
                                                                   Rcpp_fn_OpenMP_EHMC_sampling
                                                             } else {
                                                                   Rcpp_fn_RcppParallel_EHMC_sampling
                                                             }
                                sampler_package_namespace <-  environment(sampling_function_in_use)
                                fn_set_sampler_halton_flag <-  get0( x = "fn_set_sampling_tau_jitter_halton",
                                                                     envir = sampler_package_namespace,
                                                                     mode = "function",
                                                                     inherits = FALSE)
                                if (is.null(fn_set_sampler_halton_flag)) {
                                      stop(paste0("tau_jitter_sampling_type = \"halton\", but the installed ",
                                                  environmentName(sampler_package_namespace), " build has no ",
                                                  "fn_set_sampling_tau_jitter_halton(); rebuild it."))
                                }
                                fn_set_sampler_halton_flag(TRUE)
                                on.exit(fn_set_sampler_halton_flag(FALSE), add = TRUE)
                          }
                          ##
                          ## ---- step-size (eps) calibration on the sampling chains
                          ##      (fn_calibrate_eps_on_sampling_chains_from_one_call_to_adapt_delta in
                          ##      R_fn_calibrate_eps_on_sampling_chains_to_adapt_delta.R): one sampling call of
                          ##      every sampling chain, from the initial states of the sampling, at the eps
                          ##      handed over by the burn-in, without any nuisance trace; its first block of
                          ##      iterations is discarded and the acceptance of the rest sets the sampling eps
                          ##      so that the mean acceptance probability over the sampling chains is
                          ##      adapt_delta. The sampling then starts again from the same initial states (the
                          ##      sampler copies them). Its time is counted with the burn-in time (below) and its
                          ##      gradient evaluations with the burn-in's (summary):
                          ##
                          eps_calibration_on_sampling_chains <-  NULL
                          time_eps_calibration_on_sampling_chains <-  0
                          if (isTRUE(calibrate_eps_on_sampling_chains_before_sampling)) {
                                time_start_of_eps_calibration <-  Sys.time()
                                ## ## (every nuisance parameter of the sampler's vector, so that the last draw of
                                ## ##  the block is the chain's whole nuisance state; 0 without nuisance
                                ## ##  parameters)
                                ## n_nuisance_in_last_state_of_calibration_block <-  if (isTRUE(sample_nuisance))
                                      ## NROW(theta_nuisance_vectors_all_chains_input_from_R) else 0
                                ## fn_run_sampling_iterations_continuing_the_chains <-  function( n_iter,
                                                                                               ## eps_main,
                                                                                               ## eps_us,
                                                                                               ## seed) {
                                      ## EHMC_args_of_calibration_block <-  EHMC_args_as_Rcpp_List
                                      ## EHMC_args_of_calibration_block$eps_main <-  eps_main
                                      ## EHMC_args_of_calibration_block$eps_us <-  eps_us
                                      ## if (parallel_method == "OpenMP") {
                                            ## result_of_block <-  Rcpp_fn_OpenMP_EHMC_sampling(
                                                  ## n_threads_R = n_chains_sampling,
                                                  ## sample_nuisance_R = sample_nuisance,
                                                  ## n_nuisance_to_track =
                                                        ## n_nuisance_in_last_state_of_calibration_block,
                                                  ## seed_R = seed,
                                                  ## iter_one_by_one = FALSE,
                                                  ## n_iter_R = n_iter,
                                                  ## partitioned_HMC_R = partitioned_HMC,
                                                  ## diffusion_HMC_R = diffusion_HMC,
                                                  ## Model_type_R = Model_type,
                                                  ## force_autodiff_R = force_autodiff,
                                                  ## force_PartialLog = force_PartialLog,
                                                  ## multi_attempts_R = multi_attempts,
                                                  ## theta_main_vectors_all_chains_input_from_R =
                                                        ## theta_main_vectors_all_chains_input_from_R,
                                                  ## theta_us_vectors_all_chains_input_from_R =
                                                        ## theta_nuisance_vectors_all_chains_input_from_R,
                                                  ## y = y,
                                                  ## Model_args_as_Rcpp_List = Model_args_as_Rcpp_List,
                                                  ## EHMC_args_as_Rcpp_List = EHMC_args_of_calibration_block,
                                                  ## EHMC_Metric_as_Rcpp_List = EHMC_Metric_as_Rcpp_List,
                                                  ## n_threads_WCP = n_threads_WCP_sampling)
                                      ## } else {
                                            ## EHMC_args_of_calibration_block$store_log_lik_trace <-  FALSE
                                            ## result_of_block <-  Rcpp_fn_RcppParallel_EHMC_sampling(
                                                  ## n_threads_R = n_chains_sampling,
                                                  ## sample_nuisance_R = sample_nuisance,
                                                  ## n_nuisance_to_track =
                                                        ## n_nuisance_in_last_state_of_calibration_block,
                                                  ## seed_R = seed,
                                                  ## iter_one_by_one = FALSE,
                                                  ## n_iter_R = n_iter,
                                                  ## partitioned_HMC_R = partitioned_HMC,
                                                  ## diffusion_HMC_R = diffusion_HMC,
                                                  ## Model_type_R = Model_type,
                                                  ## force_autodiff_R = force_autodiff,
                                                  ## force_PartialLog = force_PartialLog,
                                                  ## multi_attempts_R = multi_attempts,
                                                  ## theta_main_vectors_all_chains_input_from_R =
                                                        ## theta_main_vectors_all_chains_input_from_R,
                                                  ## theta_us_vectors_all_chains_input_from_R =
                                                        ## theta_nuisance_vectors_all_chains_input_from_R,
                                                  ## y = y,
                                                  ## Model_args_as_Rcpp_List = Model_args_as_Rcpp_List,
                                                  ## EHMC_args_as_Rcpp_List = EHMC_args_of_calibration_block,
                                                  ## EHMC_Metric_as_Rcpp_List = EHMC_Metric_as_Rcpp_List,
                                                  ## use_disk = FALSE,
                                                  ## trace_dir = use_disk_path,
                                                  ## n_threads_WCP = n_threads_WCP_sampling)
                                      ## }
                                      ## ## (the last state of every chain becomes the start of the next block and
                                      ## ##  of the sampling: the last column of each chain's main and nuisance
                                      ## ##  traces, in the sampler's own layout; fn_sample's variables are
                                      ## ##  updated)
                                      ## main_traces_of_block <-  result_of_block[[1]]
                                      ## for (chain in seq_len(n_chains_sampling)) {
                                            ## last_iteration <-  ncol(main_traces_of_block[[chain]])
                                            ## theta_main_vectors_all_chains_input_from_R[, chain] <<-
                                                  ## main_traces_of_block[[chain]][, last_iteration]
                                      ## }
                                      ## if (n_nuisance_in_last_state_of_calibration_block > 0) {
                                            ## nuisance_traces_of_block <-  result_of_block[[3]]
                                            ## for (chain in seq_len(n_chains_sampling)) {
                                                  ## last_iteration <-  ncol(nuisance_traces_of_block[[chain]])
                                                  ## theta_nuisance_vectors_all_chains_input_from_R[, chain] <<-
                                                        ## nuisance_traces_of_block[[chain]][, last_iteration]
                                            ## }
                                            ## rm(nuisance_traces_of_block)
                                      ## }
                                      ## ## (only the acceptance probabilities are returned: no trace is kept)
                                      ## return(list(sampling_diagnostics = result_of_block$sampling_diagnostics))
                                ## }
                             ## ## (the sampling chains in a persistent native worker - the burn-in's own worker
                             ## ##  type - with the sampling's settings: per-chain trajectory lengths, the
                             ## ##  sampling metric and nuisance centre; the functions of the package whose worker
                             ## ##  the burn-in used)
                             ## worker_api_env_of_calibration <-  environment(fn_create_persistent_burnin_worker)
                             ## fn_run_one_iteration_of_calibration_worker <-
                                   ## get(x = "fn_persistent_burnin_run_one_iter_main_only",
                                       ## envir = worker_api_env_of_calibration, mode = "function")
                             ## fn_state_of_calibration_worker <-
                                   ## get(x = "fn_persistent_burnin_get_state",
                                       ## envir = worker_api_env_of_calibration, mode = "function")
                             ## calibration_worker <-  fn_create_persistent_burnin_worker(
                                   ## n_threads_R = n_chains_sampling,
                                   ## partitioned_HMC_R = partitioned_HMC,
                                   ## diffusion_HMC_R = diffusion_HMC,
                                   ## Model_type_R = Model_type,
                                   ## sample_nuisance_R = sample_nuisance,
                                   ## force_autodiff_R = force_autodiff,
                                   ## force_PartialLog_R = force_PartialLog,
                                   ## multi_attempts_R = multi_attempts,
                                   ## y_Eigen_R = y,
                                   ## Model_args_as_Rcpp_List = Model_args_as_Rcpp_List,
                                   ## EHMC_args_as_Rcpp_List = EHMC_args_as_Rcpp_List,
                                   ## EHMC_Metric_as_Rcpp_List = EHMC_Metric_as_Rcpp_List,
                                   ## n_threads_WCP = n_threads_WCP_sampling)
                             ## fn_persistent_burnin_set_theta( calibration_worker,
                                                             ## theta_main_vectors_all_chains_input_from_R,
                                                             ## theta_nuisance_vectors_all_chains_input_from_R)
                             ## ## (the iteration counter of the worker's random-number streams, over every block)
                             ## calibration_iteration <-  0
                             ## fn_run_sampling_iterations_continuing_the_chains <-  function( n_iter,
                                                                                            ## eps_main,
                                                                                            ## eps_us,
                                                                                            ## seed) {
                                   ## EHMC_args_of_calibration_block <-  EHMC_args_as_Rcpp_List
                                   ## EHMC_args_of_calibration_block$eps_main <-  eps_main
                                   ## EHMC_args_of_calibration_block$eps_us <-  eps_us
                                   ## fn_persistent_burnin_update_adaptation( calibration_worker,
                                                                           ## EHMC_args_of_calibration_block,
                                                                           ## EHMC_Metric_as_Rcpp_List)
                                   ## p_jump_main <-  matrix(data = NA_real_, nrow = n_iter,
                                                          ## ncol = n_chains_sampling)
                                   ## p_jump_nuisance <-  matrix(data = NA_real_, nrow = n_iter,
                                                              ## ncol = n_chains_sampling)
                                   ## for (iteration in seq_len(n_iter)) {
                                         ## calibration_iteration <<-  calibration_iteration + 1
                                         ## result_of_iteration <-  fn_run_one_iteration_of_calibration_worker(
                                               ## calibration_worker, seed, calibration_iteration)
                                         ## ## (the acceptance probability of each chain: row 1 of the main and of
                                         ## ##  the nuisance outputs, as the burn-in reads them)
                                         ## main_outputs <-
                                               ## result_of_iteration$other_main_out_vector_all_chains_output_to_R
                                         ## nuisance_outputs <-
                                               ## result_of_iteration$other_us_out_vector_all_chains_output_to_R
                                         ## p_jump_main[iteration, ] <-  main_outputs[1, ]
                                         ## p_jump_nuisance[iteration, ] <-  nuisance_outputs[1, ]
                                   ## }
                                   ## return(list(sampling_diagnostics = list(p_jump_main = p_jump_main,
                                                                           ## p_jump_nuisance = p_jump_nuisance)))
                             ## }
                                fn_run_sampling_iterations_from_initial_states <-  function( n_iter,
                                                                                             eps_main,
                                                                                             eps_us,
                                                                                             seed) {
                                      EHMC_args_of_calibration_call <-  EHMC_args_as_Rcpp_List
                                      EHMC_args_of_calibration_call$eps_main <-  eps_main
                                      EHMC_args_of_calibration_call$eps_us <-  eps_us
                                      if (parallel_method == "OpenMP") {
                                            result_of_call <-  Rcpp_fn_OpenMP_EHMC_sampling(
                                                  n_threads_R = n_chains_sampling,
                                                  sample_nuisance_R = sample_nuisance,
                                                  n_nuisance_to_track = 0,
                                                  seed_R = seed,
                                                  iter_one_by_one = FALSE,
                                                  n_iter_R = n_iter,
                                                  partitioned_HMC_R = partitioned_HMC,
                                                  diffusion_HMC_R = diffusion_HMC,
                                                  Model_type_R = Model_type,
                                                  force_autodiff_R = force_autodiff,
                                                  force_PartialLog = force_PartialLog,
                                                  multi_attempts_R = multi_attempts,
                                                  theta_main_vectors_all_chains_input_from_R =
                                                        theta_main_vectors_all_chains_input_from_R,
                                                  theta_us_vectors_all_chains_input_from_R =
                                                        theta_nuisance_vectors_all_chains_input_from_R,
                                                  y = y,
                                                  Model_args_as_Rcpp_List = Model_args_as_Rcpp_List,
                                                  EHMC_args_as_Rcpp_List = EHMC_args_of_calibration_call,
                                                  EHMC_Metric_as_Rcpp_List = EHMC_Metric_as_Rcpp_List,
                                                  n_threads_WCP = n_threads_WCP_sampling)
                                      } else {
                                            EHMC_args_of_calibration_call$store_log_lik_trace <-  FALSE
                                            result_of_call <-  Rcpp_fn_RcppParallel_EHMC_sampling(
                                                  n_threads_R = n_chains_sampling,
                                                  sample_nuisance_R = sample_nuisance,
                                                  n_nuisance_to_track = 0,
                                                  seed_R = seed,
                                                  iter_one_by_one = FALSE,
                                                  n_iter_R = n_iter,
                                                  partitioned_HMC_R = partitioned_HMC,
                                                  diffusion_HMC_R = diffusion_HMC,
                                                  Model_type_R = Model_type,
                                                  force_autodiff_R = force_autodiff,
                                                  force_PartialLog = force_PartialLog,
                                                  multi_attempts_R = multi_attempts,
                                                  theta_main_vectors_all_chains_input_from_R =
                                                        theta_main_vectors_all_chains_input_from_R,
                                                  theta_us_vectors_all_chains_input_from_R =
                                                        theta_nuisance_vectors_all_chains_input_from_R,
                                                  y = y,
                                                  Model_args_as_Rcpp_List = Model_args_as_Rcpp_List,
                                                  EHMC_args_as_Rcpp_List = EHMC_args_of_calibration_call,
                                                  EHMC_Metric_as_Rcpp_List = EHMC_Metric_as_Rcpp_List,
                                                  use_disk = FALSE,
                                                  trace_dir = use_disk_path,
                                                  n_threads_WCP = n_threads_WCP_sampling)
                                      }
                                      ## (only the acceptance probabilities are returned)
                                      return(list(sampling_diagnostics = result_of_call$sampling_diagnostics))
                                }
                                ## eps_calibration_on_sampling_chains <-
                                ## fn_calibrate_eps_on_sampling_chains_to_adapt_delta( ... )   (chained blocks)
                                eps_calibration_on_sampling_chains <-
                                fn_calibrate_eps_on_sampling_chains_from_one_call_to_adapt_delta(
                                      fn_run_sampling_iterations_from_initial_states =
                                            fn_run_sampling_iterations_from_initial_states,
                                      eps_main = EHMC_args_as_Rcpp_List$eps_main,
                                      eps_us = EHMC_args_as_Rcpp_List$eps_us,
                                      calibrate_eps_us = isTRUE(partitioned_HMC) && isTRUE(sample_nuisance),
                                      adapt_delta = adapt_delta,
                                      n_chains_sampling = n_chains_sampling,
                                      n_transitions_per_eps_calibration_block =
                                            n_transitions_per_eps_calibration_block,
                                      n_eps_calibration_blocks = n_eps_calibration_blocks,
                                      max_eps_main = max_eps_main,
                                      max_eps_us = max_eps_us,
                                      seed = seed)
                             ## ## (the chains' states after the last block, read once, become the initial states
                             ## ##  of the sampling; then the worker is released)
                             ## theta_main_vectors_all_chains_input_from_R <-
                                   ## fn_state_of_calibration_worker(calibration_worker,
                                                                  ## "theta_main_vectors_all_chains_output_to_R")
                             ## if (isTRUE(sample_nuisance) &&
                                 ## (NROW(theta_nuisance_vectors_all_chains_input_from_R) > 0)) {
                                   ## theta_nuisance_vectors_all_chains_input_from_R <-
                                         ## fn_state_of_calibration_worker(
                                               ## calibration_worker, "theta_us_vectors_all_chains_output_to_R")
                             ## }
                             ## rm(calibration_worker)
                             ## invisible(gc())
                                ## (also in the burn-in record's EHMC_args_as_Rcpp_List, from which the summary
                                ##  takes the sampling eps for the sampling gradient evaluations; the record's
                                ##  eps_main and eps_us keep the eps handed over by the burn-in)
                                EHMC_args_as_Rcpp_List$eps_main <-  eps_calibration_on_sampling_chains$eps_main
                                burnin_object$EHMC_args_as_Rcpp_List$eps_main <-
                                      eps_calibration_on_sampling_chains$eps_main
                                if (isTRUE(partitioned_HMC) && isTRUE(sample_nuisance)) {
                                      EHMC_args_as_Rcpp_List$eps_us <-  eps_calibration_on_sampling_chains$eps_us
                                      burnin_object$EHMC_args_as_Rcpp_List$eps_us <-
                                            eps_calibration_on_sampling_chains$eps_us
                                }
                                ## (the tau of the blocks, for their gradient evaluations in the summary)
                                eps_calibration_on_sampling_chains$tau_main <-  EHMC_args_as_Rcpp_List$tau_main
                                eps_calibration_on_sampling_chains$tau_us <-  EHMC_args_as_Rcpp_List$tau_us
                                time_eps_calibration_on_sampling_chains <-
                                      as.numeric(difftime( Sys.time(),
                                                           time_start_of_eps_calibration,
                                                           units = "secs"))
                                eps_calibration_on_sampling_chains$time_seconds <-
                                      time_eps_calibration_on_sampling_chains
                          }
                          ##
                          ## ---- the sampling eps from the acceptance probabilities which the burn-in chains
                          ##      recorded at each iteration of the last part of the burn-in, so that the ORDINARY
                          ##      mean acceptance probability is adapt_delta. The burn-in, with its harmonic-mean
                          ##      eps adaptation, is unchanged, and no iteration is added: the burn-in time and
                          ##      gradient evaluations are the burn-in's own
                          ##      (R_fn_sampling_eps_to_adapt_delta_from_acceptance_of_burnin_chains.R):
                          ##
                          sampling_eps_from_acceptance_of_burnin_chains <-  NULL
                          if (isTRUE(set_sampling_eps_to_adapt_delta_from_acceptance_of_burnin_chains)) {
                                if (isTRUE(partitioned_HMC) && isTRUE(sample_nuisance)) {
                                      stop(paste0("set_sampling_eps_to_adapt_delta_from_acceptance_of_burnin_",
                                                  "chains sets the main eps only: it is not available with ",
                                                  "partitioned HMC."))
                                }
                                sampling_eps_from_acceptance_of_burnin_chains <-
                                      fn_sampling_eps_to_adapt_delta_from_acceptance_of_burnin_chains(
                                            eps_main_of_transition_during_burnin_vec =
                                                  burnin_object$eps_main_of_transition_during_burnin_vec,
                                            p_jump_main_during_burnin_vec =
                                                  burnin_object$p_jump_main_during_burnin_vec,
                                            metric_update_eps_tau_rescale_factor_vec =
                                                  burnin_object$metric_update_eps_tau_rescale_factor_vec,
                                            eps_main_handed_over_by_burnin = EHMC_args_as_Rcpp_List$eps_main,
                                            n_chains_burnin = n_chains_burnin,
                                            adapt_delta = adapt_delta,
                                            fraction_of_burnin_iterations_at_end_used =
                                                  fraction_of_burnin_iterations_at_end_used_for_sampling_eps,
                                            ## max_eps_main = max_eps_main)
                                            max_eps_main = max_eps_main,
                                            ## (the iterations after the final metric update; a burn-in record
                                            ##  without burnin_schedule$metric_adaptation_end_iter stops there)
                                            first_burnin_iteration_used_for_sampling_eps =
                                                  burnin_object$burnin_schedule$metric_adaptation_end_iter + 1)
                                ## (also in the burn-in record's EHMC_args_as_Rcpp_List, from which the summary
                                ##  takes the sampling eps for the sampling gradient evaluations; the record's
                                ##  eps_main keeps the eps handed over by the burn-in)
                                EHMC_args_as_Rcpp_List$eps_main <-
                                      sampling_eps_from_acceptance_of_burnin_chains$eps_main
                                burnin_object$EHMC_args_as_Rcpp_List$eps_main <-
                                      sampling_eps_from_acceptance_of_burnin_chains$eps_main
                          }
                          ## (the start of the sampling call, whose seconds a repeated sampling counts as burn-in)
                          time_start_of_sampling_call <-  Sys.time()
                          if (parallel_method == "OpenMP") {
                             sampling_object <- Rcpp_fn_OpenMP_EHMC_sampling( n_threads_R = n_chains_sampling,
                                                                                            sample_nuisance_R = sample_nuisance,
                                                                                            n_nuisance_to_track = n_nuisance_to_track,
                                                                                            seed_R = seed,
                                                                                            iter_one_by_one = FALSE,
                                                                                            n_iter_R = n_iter,
                                                                                            partitioned_HMC_R = partitioned_HMC,
                                                                                            diffusion_HMC_R = diffusion_HMC,
                                                                                            Model_type_R = Model_type,
                                                                                            force_autodiff_R = force_autodiff,
                                                                                            force_PartialLog = force_PartialLog,
                                                                                            multi_attempts_R = multi_attempts,
                                                                                            theta_main_vectors_all_chains_input_from_R = theta_main_vectors_all_chains_input_from_R,
                                                                                            theta_us_vectors_all_chains_input_from_R = theta_nuisance_vectors_all_chains_input_from_R,
                                                                                            y =  y,  ## only used in C++ for mnl models! (data passed via Stan_data_list / JSON for Stan models!)
                                                                                            Model_args_as_Rcpp_List =  Model_args_as_Rcpp_List,
                                                                                            EHMC_args_as_Rcpp_List =   EHMC_args_as_Rcpp_List,
                                                                                            EHMC_Metric_as_Rcpp_List = EHMC_Metric_as_Rcpp_List,
                                                                                            n_threads_WCP = n_threads_WCP_sampling)


                          } else {
                               sampling_object <- Rcpp_fn_RcppParallel_EHMC_sampling(   n_threads_R = n_chains_sampling,
                                                          sample_nuisance_R = sample_nuisance,
                                                          n_nuisance_to_track = n_nuisance_to_track,
                                                          seed_R = seed,
                                                          iter_one_by_one = FALSE,
                                                          n_iter_R = n_iter,
                                                          partitioned_HMC_R = partitioned_HMC,
                                                          diffusion_HMC_R = diffusion_HMC,
                                                          Model_type_R = Model_type,
                                                          force_autodiff_R = force_autodiff,
                                                          force_PartialLog = force_PartialLog,
                                                          multi_attempts_R = multi_attempts,
                                                          theta_main_vectors_all_chains_input_from_R = theta_main_vectors_all_chains_input_from_R,
                                                          theta_us_vectors_all_chains_input_from_R = theta_nuisance_vectors_all_chains_input_from_R,
                                                          y =  y,  ## only used in C++ for mnl models! (data passed via Stan_data_list / JSON for Stan models!) 
                                                          Model_args_as_Rcpp_List =  Model_args_as_Rcpp_List,
                                                          EHMC_args_as_Rcpp_List =   EHMC_args_as_Rcpp_List,
                                                          EHMC_Metric_as_Rcpp_List = EHMC_Metric_as_Rcpp_List,
                                                          use_disk = use_disk,
                                                          trace_dir = use_disk_path,
                                                          n_threads_WCP = n_threads_WCP_sampling)
                          }
                          print(paste("End of sampling"))
                          if (!is.null(fn_set_sampler_halton_flag)) fn_set_sampler_halton_flag(FALSE)
                          ##
                          ## ---- the sampling repeated with a refitted eps, when its mean acceptance probability
                          ##      (over every sampling chain and iteration) is outside
                          ##      repeat_sampling_with_eps_refitted_if_mean_acceptance_outside, up to
                          ##      n_eps_refits_and_repeated_samplings times: the eps is refitted from the two
                          ##      most recent points (the last calibration point and the sampling), with the
                          ##      curve alpha(eps) = 2 Phi(-c eps^k) through both (z = -qnorm(alpha / 2) =
                          ##      c eps^k; the one-call calibration assumes k = 2 through one point; with less
                          ##      than 5% between the two eps, or k outside [1, 6], k = 2 through the second
                          ##      point). The sampling starts again from the same initial states, with the same
                          ##      seed. Each earlier sampling counts as burn-in: it is added to the calibration's
                          ##      blocks (whose gradient evaluations the summary counts with the burn-in's) and
                          ##      its seconds to the calibration's time (moved to the burn-in time below):
                          ##
                          eps_refits_after_mean_sampling_acceptance_outside_bounds <-  NULL
                          if (!is.null(repeat_sampling_with_eps_refitted_if_mean_acceptance_outside) &&
                              !is.null(eps_calibration_on_sampling_chains)) {
                                bounds <-  repeat_sampling_with_eps_refitted_if_mean_acceptance_outside
                                for (refit_number in seq_len(n_eps_refits_and_repeated_samplings + 1)) {

                                      seconds_of_sampling_call <-  as.numeric(difftime(
                                                                             Sys.time(),
                                                                             time_start_of_sampling_call,
                                                                             units = "secs"))
                                      mean_sampling_acceptance <-
                                            mean(sampling_object$sampling_diagnostics$p_jump_main)
                                      if ((mean_sampling_acceptance >= bounds[1]) &&
                                          (mean_sampling_acceptance <= bounds[2])) break
                                      if (refit_number > n_eps_refits_and_repeated_samplings) {
                                            warning(paste0("The mean sampling acceptance ",
                                                           round(mean_sampling_acceptance, 3),
                                                           " is still outside [", bounds[1], ", ", bounds[2],
                                                           "] after ", n_eps_refits_and_repeated_samplings,
                                                           " refits of the sampling eps."))
                                            break
                                      }
                                      ##
                                      ## ---- the refit from the two most recent points:
##
                                      blocks <-  eps_calibration_on_sampling_chains$blocks
                                      calibration_points <-  blocks[blocks$used_in_fit, , drop = FALSE]
                                      last_calibration_point <-  calibration_points[nrow(calibration_points), ]
                                      eps_of_sampling <-  EHMC_args_as_Rcpp_List$eps_main
                                      acceptance_of_first_point <-  last_calibration_point$mean_acceptance_main
                                      z_of_first_point <-  -stats::qnorm(acceptance_of_first_point / 2)
                                      z_of_second_point <-  -stats::qnorm(mean_sampling_acceptance / 2)
                                      z_of_adapt_delta <-  -stats::qnorm(adapt_delta / 2)
                                      log_ratio_of_eps <-  log(eps_of_sampling / last_calibration_point$eps_main)
                                      k <-  log(z_of_second_point / z_of_first_point) / log_ratio_of_eps
                                      too_little_change_of_eps <-  abs(log_ratio_of_eps) < log(1.05)
                                      if (!is.finite(k) || too_little_change_of_eps || (k < 1) || (k > 6)) {
                                            k <-  2
                                      }
                                      eps_main_refitted <-  eps_of_sampling *
                                                            (z_of_adapt_delta / z_of_second_point)^(1 / k)
                                      eps_main_refitted <-  min(max_eps_main, eps_main_refitted)
                                      message(colourise(paste0(
                                            "Mean sampling acceptance ", round(mean_sampling_acceptance, 3),
                                            " outside [", bounds[1], ", ", bounds[2], "]: refit ", refit_number,
                                            " of the sampling eps from (",
                                            signif(last_calibration_point$eps_main, 4), ", ",
                                            round(acceptance_of_first_point, 3), ") and (",
                                            signif(eps_of_sampling, 4), ", ", round(mean_sampling_acceptance, 3),
                                            "): k = ", round(k, 2), ", eps_main = ", signif(eps_main_refitted, 4),
                                            "; the sampling is repeated"),
                                            "cyan"))
                                      ##
                                      ## ---- the sampling just run, counted as burn-in (one more calibration
                                      ##      block):
                                      sampling_as_calibration_block <-  data.frame(
                                            block = nrow(blocks) + 1,
                                            n_iter = n_iter,
                                            n_iter_discarded = 0,
                                            n_chains = n_chains_sampling,
                                            eps_main = eps_of_sampling,
                                            mean_acceptance_main = mean_sampling_acceptance,
                                            eps_us = EHMC_args_as_Rcpp_List$eps_us,
                                            mean_acceptance_us = NA_real_,
                                            used_in_fit = TRUE,
                                            stringsAsFactors = FALSE)
                                      eps_calibration_on_sampling_chains$blocks <-
                                            rbind(blocks, sampling_as_calibration_block)
                                      eps_calibration_on_sampling_chains$time_seconds <-
                                            eps_calibration_on_sampling_chains$time_seconds +
                                            seconds_of_sampling_call
                                      time_eps_calibration_on_sampling_chains <-
                                            time_eps_calibration_on_sampling_chains + seconds_of_sampling_call
                                      eps_refits_after_mean_sampling_acceptance_outside_bounds <-  rbind(
                                            eps_refits_after_mean_sampling_acceptance_outside_bounds,
                                            data.frame( refit = refit_number,
                                                        eps_main_of_last_calibration_point =
                                                              last_calibration_point$eps_main,
                                                        mean_acceptance_of_last_calibration_point =
                                                              last_calibration_point$mean_acceptance_main,
                                                        eps_main_of_sampling = eps_of_sampling,
                                                        mean_sampling_acceptance = mean_sampling_acceptance,
                                                        exponent_k_of_refit = k,
                                                        eps_main_refitted = eps_main_refitted,
                                                        seconds_counted_as_burnin = seconds_of_sampling_call,
                                                        stringsAsFactors = FALSE))
                                      ##
                                      ## ---- the sampling again, from the same initial states and seed, with
                                      ##      the refitted eps (also in the burn-in record, from which the summary
                                      ##      takes the sampling eps):
                                      EHMC_args_as_Rcpp_List$eps_main <-  eps_main_refitted
                                      burnin_object$EHMC_args_as_Rcpp_List$eps_main <-  eps_main_refitted
                                      eps_calibration_on_sampling_chains$eps_main <-  eps_main_refitted
                                      rm(sampling_object)
                                      invisible(gc(verbose = FALSE))
                                      if (!is.null(fn_set_sampler_halton_flag)) fn_set_sampler_halton_flag(TRUE)
                                      time_start_of_sampling_call <-  Sys.time()
                                      if (parallel_method == "OpenMP") {
                                            sampling_object <-  Rcpp_fn_OpenMP_EHMC_sampling(
                                                  n_threads_R = n_chains_sampling,
                                                  sample_nuisance_R = sample_nuisance,
                                                  n_nuisance_to_track = n_nuisance_to_track,
                                                  seed_R = seed,
                                                  iter_one_by_one = FALSE,
                                                  n_iter_R = n_iter,
                                                  partitioned_HMC_R = partitioned_HMC,
                                                  diffusion_HMC_R = diffusion_HMC,
                                                  Model_type_R = Model_type,
                                                  force_autodiff_R = force_autodiff,
                                                  force_PartialLog = force_PartialLog,
                                                  multi_attempts_R = multi_attempts,
                                                  theta_main_vectors_all_chains_input_from_R =
                                                        theta_main_vectors_all_chains_input_from_R,
                                                  theta_us_vectors_all_chains_input_from_R =
                                                        theta_nuisance_vectors_all_chains_input_from_R,
                                                  y = y,
                                                  Model_args_as_Rcpp_List = Model_args_as_Rcpp_List,
                                                  EHMC_args_as_Rcpp_List = EHMC_args_as_Rcpp_List,
                                                  EHMC_Metric_as_Rcpp_List = EHMC_Metric_as_Rcpp_List,
                                                  n_threads_WCP = n_threads_WCP_sampling)
                                      } else {
                                            sampling_object <-  Rcpp_fn_RcppParallel_EHMC_sampling(
                                                  n_threads_R = n_chains_sampling,
                                                  sample_nuisance_R = sample_nuisance,
                                                  n_nuisance_to_track = n_nuisance_to_track,
                                                  seed_R = seed,
                                                  iter_one_by_one = FALSE,
                                                  n_iter_R = n_iter,
                                                  partitioned_HMC_R = partitioned_HMC,
                                                  diffusion_HMC_R = diffusion_HMC,
                                                  Model_type_R = Model_type,
                                                  force_autodiff_R = force_autodiff,
                                                  force_PartialLog = force_PartialLog,
                                                  multi_attempts_R = multi_attempts,
                                                  theta_main_vectors_all_chains_input_from_R =
                                                        theta_main_vectors_all_chains_input_from_R,
                                                  theta_us_vectors_all_chains_input_from_R =
                                                        theta_nuisance_vectors_all_chains_input_from_R,
                                                  y = y,
                                                  Model_args_as_Rcpp_List = Model_args_as_Rcpp_List,
                                                  EHMC_args_as_Rcpp_List = EHMC_args_as_Rcpp_List,
                                                  EHMC_Metric_as_Rcpp_List = EHMC_Metric_as_Rcpp_List,
                                                  use_disk = use_disk,
                                                  trace_dir = use_disk_path,
                                                  n_threads_WCP = n_threads_WCP_sampling)
                                      }
                                      print(paste("End of the repeated sampling"))
                                      if (!is.null(fn_set_sampler_halton_flag)) fn_set_sampler_halton_flag(FALSE)

                                }
                          }
                           
                           # str(sampling_object)
                          
                          try({
                              sampling_toc_result <- tictoc::toc(log = TRUE)
                              print(sampling_toc_result)
                              tictoc::tic.clearlog()
                              ## elapsed seconds straight from toc()'s numbers. Parsing the "X sec elapsed" text with "\\d+\\.\\d+"
                              ## gave NA whenever the time rounded to a whole second (tictoc then prints e.g. "4 sec elapsed"):
                              time_sampling <- as.numeric(sampling_toc_result$toc - sampling_toc_result$tic)
                          })
                          ## (the eps calibration on the sampling chains ran inside the post-burnin timer: its
                          ##  time moves from the sampling time to the burn-in time)
                          if (time_eps_calibration_on_sampling_chains > 0) {
                                time_sampling <-  time_sampling - time_eps_calibration_on_sampling_chains
                                burnin_object$time_burnin <-  burnin_object$time_burnin +
                                                              time_eps_calibration_on_sampling_chains
                                time_burnin <-  time_burnin + time_eps_calibration_on_sampling_chains
                          }

                          print(paste("time_sampling = ",  time_sampling))
                          
                          # gc()
                          
                }
                
                time_total <- NULL
                try({
                   time_total <- time_sampling + time_burnin
                   ##
                   print(paste("time_pre_burnin = ",  time_pre_burnin))
                   print(paste("time_burnin = ",  time_burnin))
                   print(paste("time_total = ",  time_total))
                })

                ##
                ## ---- Names on every per-coordinate object of the returned burn-in record (draws, adapted
                ##      metric, theta_hat_us, snaper / eigen vectors, final states), each the name of its
                ##      coordinate of the sampler's vector in the sampler's own layout
                ##      (R_fn_name_burnin_record_by_sampler_coordinate.R). Values are unchanged, and sampling has
                ##      finished, so nothing handed to the sampler is touched. A failure here leaves the record
                ##      without names, with a warning:
                ##
                burnin_object <-  tryCatch(
                      fn_burnin_record_with_names_of_sampler_coordinates(
                            burnin_object    = burnin_object,
                            coordinate_names = fn_names_of_sampler_coordinates_in_sampler_layout(
                                  init_object                          = init_object,
                                  Model_type                           = Model_type,
                                  test_perm                            = test_perm,
                                  num_chunks_of_burnin_nuisance_layout =
                                        if (exists("num_chunks_used_in_burnin", inherits = FALSE))
                                              num_chunks_used_in_burnin else NULL,
                                  vect_type_of_burnin_nuisance_layout  =
                                        if (Model_type != "Stan") Model_args_as_Rcpp_List$Model_args_strings[1]
                                        else NULL,
                                  fn_nuisance_chunk_layout_positions   =
                                        if (exists("fn_nuisance_chunk_layout_positions", mode = "function"))
                                              get("fn_nuisance_chunk_layout_positions", mode = "function") else
                                              NULL)),
                      error = function(error_object) {
                            warning(paste0("burn-in record left without coordinate names: ",
                                           conditionMessage(error_object)))
                            return(burnin_object)
                      })

  out_list <- list(init_object = init_object,
                   burnin_object = burnin_object,
                   burnin_schedule = burnin_object$burnin_schedule,
                   burnin_profile = if (isTRUE(debug_burnin_timing)) list(
                       pre_burnin = pre_burnin_object$burnin_profile,
                       main_burnin = burnin_object$burnin_profile) else NULL,
                   sampling_object = sampling_object,
                   ## the eps calibration on the sampling chains
                   ## (R_fn_calibrate_eps_on_sampling_chains_to_adapt_delta.R; NULL when
                   ## calibrate_eps_on_sampling_chains_before_sampling = FALSE): the sampling eps, its blocks
                   ## (counted with the burn-in's gradient evaluations by the summary), the tau of the blocks and
                   ## the time (counted with the burn-in time):
                   eps_calibration_on_sampling_chains =
                         if (exists("eps_calibration_on_sampling_chains", inherits = FALSE))
                               eps_calibration_on_sampling_chains else NULL,
                   ## the refits of the sampling eps after a mean sampling acceptance outside
                   ## repeat_sampling_with_eps_refitted_if_mean_acceptance_outside (NULL when there was none):
                   eps_refits_after_mean_sampling_acceptance_outside_bounds =
                         if (exists("eps_refits_after_mean_sampling_acceptance_outside_bounds", inherits = FALSE))
                               eps_refits_after_mean_sampling_acceptance_outside_bounds else NULL,
                   ## the sampling eps from the acceptance probabilities which the burn-in chains recorded
                   ## (R_fn_sampling_eps_to_adapt_delta_from_acceptance_of_burnin_chains.R; NULL when
                   ## set_sampling_eps_to_adapt_delta_from_acceptance_of_burnin_chains = FALSE):
                   sampling_eps_from_acceptance_of_burnin_chains =
                         if (exists("sampling_eps_from_acceptance_of_burnin_chains", inherits = FALSE))
                               sampling_eps_from_acceptance_of_burnin_chains else NULL,
                   ##
                   test_perm = test_perm,
                   test_inv_perm = test_inv_perm,
                   ## the rule that sets test_perm after the reorder_cols_MVP pre-burnin, as resolved:
                   test_order_rule_for_reorder_cols_MVP = test_order_rule_for_reorder_cols_MVP,
                   ## the number-of-chunks multiplier of every PartialLog (log-scale) evaluation:
                   n_chunks_multiplier_for_PartialLog_log_scale_evaluation =
                         n_chunks_multiplier_for_PartialLog_log_scale_evaluation,
                   ##
                   LR_main = LR_main, 
                   LR_us = LR_us, 
                   adapt_delta = adapt_delta,
                   ## cross-chain mean acceptance targeted by the burn-in eps adaptation ("harmonic" | "arithmetic"):
                   eps_acceptance_mean = eps_acceptance_mean,
                   burnin_algorithm = burnin_algorithm,
                   ## the parameters of interest (parameter families; NULL = every main parameter), over which
                   ## the summary computes every min-ESS-based efficiency measure, whatever the burn-in
                   ## adaptation block:
                   parameter_families_of_interest_for_min_ESS = parameter_families_of_interest_for_min_ESS,
                   trajectory_adaptation_version = "metric_main_v4",
                   ## trajectory_criterion = if (burnin_algorithm == "CHESSR_log") "ema_log_expected_numerator_over_mean_tau" else
                   ##                        if (burnin_algorithm %in% c("CHESSR", "SNAPER")) "expected_per_trajectory_rate" else
                   ##                        if (burnin_algorithm %in% c("CHESSR_time", "SNAPER_time")) "expected_per_trajectory_rate_with_time_to_target_ESS_penalty" else
                   ##                        if (burnin_algorithm == "ChEES") "expected_squared_position_statistic_change" else "squared_kinetic_energy_change",
                   trajectory_criterion = if (burnin_algorithm == "CHESSR_log") "ema_log_expected_numerator_over_mean_tau" else
                                          if (burnin_algorithm %in% c("CHESSR", "SNAPER")) "expected_per_trajectory_rate" else
                                          if (burnin_algorithm %in% c("CHESSR_time", "SNAPER_time")) "expected_per_trajectory_rate_with_time_to_target_ESS_penalty" else
                                          if (burnin_algorithm == "ESJD") "expected_per_trajectory_rate_of_squared_jumped_distance" else
                                          if (burnin_algorithm == "ESJD_CHESSR") "geometric_mean_of_ESJD_and_CHESSR_per_trajectory_rates" else
                                          if (burnin_algorithm == "ESJD_SNAPER") "geometric_mean_of_ESJD_and_SNAPER_per_trajectory_rates" else
                                          if (burnin_algorithm == "ESJD_CHESSR_SNAPER")
                                              paste0("geometric_mean_of_ESJD_CHESSR_and_SNAPER_per_trajectory_",
                                                     "rates_with_equal_weights") else
                                          if (burnin_algorithm == "ESJD_w50_CHESSR_w25_SNAPER_w25")
                                              paste0("weighted_geometric_mean_of_ESJD_weight_0.50_",
                                                     "CHESSR_weight_0.25_and_SNAPER_weight_0.25_",
                                                     "per_trajectory_rates") else
                                          if (burnin_algorithm == "LQ_ESSR")
                                              "soft_minimum_of_linear_and_quadratic_lag_one_ESS_bounds_per_unit_trajectory_length" else
                                          if (burnin_algorithm == "L_ESSR_length_response")
                                              paste0("soft_minimum_of_linear_lag_one_ESS_bounds_",
                                                     "per_unit_trajectory_length_maximised_over_tau_",
                                                     "from_the_length_response_of_jittered_trajectories") else
                                          if (burnin_algorithm == "LQ_ESSR_length_response")
                                              paste0("soft_minimum_of_linear_and_quadratic_lag_one_ESS_bounds_",
                                                     "per_unit_trajectory_length_maximised_over_tau_",
                                                     "from_the_length_response_of_jittered_trajectories") else
                                          if (burnin_algorithm == "LQ_ESSR_spectral")
                                              paste0("soft_minimum_of_spectral_linear_ESS_and_quadratic_lag_one_",
                                                     "ESS_bound_per_expected_leapfrog_step_maximised_over_tau_",
                                                     "from_the_length_response_of_jittered_trajectories") else
                                          if (burnin_algorithm == "ESJD_w20_LQ_ESSR_spec_bins99_evid_expand")
                                              paste0("weighted_geometric_mean_of_ESJD_weight_0.20_and_",
                                                     "the_soft_minimum_of_spectral_linear_ESS_and_",
                                                     "quadratic_lag_one_ESS_per_expected_leapfrog_step_",
                                                     "maximised_over_tau_with_length_bins_forgetting_",
                                                     "at_0.99_per_update_doubling_at_the_upper_end_",
                                                     "only_on_evidence") else
                                          if (burnin_algorithm == "ESJD_w33_LQ_ESSR_spec_bins99_evid_expand")
                                              paste0("weighted_geometric_mean_of_ESJD_weight_0.33_and_",
                                                     "the_soft_minimum_of_spectral_linear_ESS_and_",
                                                     "quadratic_lag_one_ESS_per_expected_leapfrog_step_",
                                                     "maximised_over_tau_with_length_bins_forgetting_",
                                                     "at_0.99_per_update_doubling_at_the_upper_end_",
                                                     "only_on_evidence") else
                                          if (burnin_algorithm == "LQ_ESSR_spec_bins99_evid_expand_jump_accept")
                                              paste0("soft_minimum_of_spectral_linear_ESS_and_quadratic_lag_one_",
                                                     "ESS_bound_per_expected_leapfrog_step_maximised_over_tau_",
                                                     "from_the_length_response_of_jittered_trajectories_",
                                                     "with_length_bins_forgetting_at_0.99_per_update_",
                                                     "doubling_at_the_upper_end_only_on_evidence_",
                                                     "with_jump_weighted_acceptance_per_row_and_length_bin") else
                                          if (burnin_algorithm == "LQ_ESSR_spec_bins99_evid_expand")
                                              paste0("soft_minimum_of_spectral_linear_ESS_and_quadratic_lag_one_",
                                                     "ESS_bound_per_expected_leapfrog_step_maximised_over_tau_",
                                                     "from_the_length_response_of_jittered_trajectories_",
                                                     "with_length_bins_forgetting_at_0.99_per_update_",
                                                     "doubling_at_the_upper_end_only_on_evidence") else
                                          if (burnin_algorithm == "ESJD_LQ_ESSR_spec_bins99_evid_expand")
                                              paste0("geometric_mean_of_ESJD_and_the_soft_minimum_of_",
                                                     "spectral_linear_ESS_and_quadratic_lag_one_ESS_",
                                                     "per_expected_leapfrog_step_maximised_over_tau_",
                                                     "with_length_bins_forgetting_at_0.99_per_update_",
                                                     "doubling_at_the_upper_end_only_on_evidence") else
                                          if (burnin_algorithm ==
                                              "ESJD_LQ_ESSR_spec_bins99_finite_N_evid_expand")
                                              paste0("geometric_mean_of_ESJD_and_the_soft_minimum_of_",
                                                     "finite_run_spectral_linear_ESS_and_quadratic_lag_one_ESS_",
                                                     "per_expected_leapfrog_step_maximised_over_tau_",
                                                     "with_length_bins_forgetting_at_0.99_per_update_",
                                                     "doubling_at_the_upper_end_only_on_evidence") else
                                          if (burnin_algorithm == "LQ_ESSR_spectral_long_bin_memory")
                                              paste0("soft_minimum_of_spectral_linear_ESS_and_quadratic_lag_one_",
                                                     "ESS_bound_per_expected_leapfrog_step_maximised_over_tau_",
                                                     "from_the_length_response_of_jittered_trajectories_",
                                                     "with_length_bins_forgetting_at_0.99_per_update") else
                                          if (burnin_algorithm == "ChEES")
                                              "expected_squared_position_statistic_change" else NA_character_,
                   trajectory_coordinates = "mass_metric",
                   trajectory_parameter_block = if_null_then_set_to(burnin_object$trajectory_parameter_block, "main"),
                   ## EXPERIMENTAL: block feeding the trajectory-length criterion, as requested and as actually used:
                   tau_adaptation_block_requested = tau_adaptation_block_as_given,
                   tau_adaptation_block = if (identical(tau_adaptation_block_as_given, "interest_only"))
                                                "interest_only" else
                                                if_null_then_set_to(burnin_object$tau_adaptation_block, "main"),
                   tau_adaptation_enabled = !isTRUE(manual_tau),
                   randomize_tau_burnin = randomize_tau_burnin,
                   randomize_tau_sampling = randomize_tau_sampling,
                   tau_gradient_estimator = tau_gradient_estimator,
                   tau_cost_exponent = tau_cost_exponent,
                   esjd_jump_power = esjd_jump_power,
                   interest_only = interest_only,
                   tau_jitter_burnin = tau_jitter_burnin,
                   tau_jitter_burnin_for_sampling = tau_jitter_burnin_for_sampling,
                   tau_jitter_sampling_type = tau_jitter_sampling_type,
                   ## tau_sampling_scale: requested value, effective value / factor, and tau either side of the one-off rescaling:
                   tau_sampling_scale = tau_sampling_scale,
                   bulk_local_tuner = bulk_local_tuner,
                   bulk_local_tuner_metadata = burnin_object$bulk_local_tuner_metadata,
                   tau_sampling_scale_effective = tau_sampling_scale_effective,
                   tau_sampling_scale_factor = tau_sampling_scale_factor,
                   tau_sampling_scale_not_applied_reason = tau_sampling_scale_not_applied_reason,
                   tau_main_before_sampling_scale = tau_main_before_sampling_scale,
                   tau_main_after_sampling_scale = tau_main_after_sampling_scale,
                   tau_us_before_sampling_scale = tau_us_before_sampling_scale,
                   tau_us_after_sampling_scale = tau_us_after_sampling_scale,
                   ## effective starting path length and learning-rate hold value passed to the main burn-in:
                   tau_initial = tau_initial,
                   learning_rate_initial = learning_rate_initial,
                   manual_tau = manual_tau,
                   tau_weight_by_p_jump = tau_weight_by_p_jump,
                   ## divergence-triggered tau shrink in the main burn-in: settings and number of iterations at which it fired:
                   tau_shrink_on_divergence        = tau_shrink_on_divergence,
                   tau_shrink_factor               = tau_shrink_factor,
                   tau_shrink_min_divergent_chains = tau_shrink_min_divergent_chains,
                   tau_shrink_n_fired              = burnin_object$tau_shrink_n_fired,
                   ## pooled metric estimator (read only when metric_estimator = "pooled"): window resets as given, the main
                   ## burn-in iterations r they resolved to (reset at the start of iteration r + 1), and the off-diagonal shrinkage:
                   metric_estimator                      = metric_estimator,
                   metric_pooled_window_resets           = metric_pooled_window_resets,
                   metric_pooled_window_reset_iterations = burnin_object$metric_pooled_window_reset_iterations,
                   metric_pooled_offdiagonal_shrinkage   = metric_pooled_offdiagonal_shrinkage,
                   metric_adaptive_shrinkage              = burnin_object$metric_adaptive_shrinkage,
                   metric_adaptive_shrinkage_history     = burnin_object$metric_adaptive_shrinkage_history,
                   ## CHESSR_time / SNAPER_time: settings, the quantities the criterion used and their sources, the sampling timing probe,
                   ## the previous-run quantities and the burn-in record (NULL for every other criterion):
                   time_criterion = if (fn_burnin_algorithm_is_time_criterion(burnin_algorithm) && !isTRUE(manual_tau)) list(
                       time_criterion_settings                   = time_criterion_settings,
                       time_criterion_sampling_quantities        = time_criterion_sampling_quantities,
                       time_criterion_previous_run_quantities    = time_criterion_previous_run_quantities,
                       sampling_timing_probe                     = sampling_timing_probe_result,
                       sampling_timing_probe_wall_time           = if (is.null(sampling_timing_probe_result)) 0 else sampling_timing_probe_result$sampling_timing_probe_wall_time,
                       sampling_timing_probe_fraction_of_burnin  = sampling_timing_probe_fraction_of_burnin,
                       time_burnin_without_sampling_timing_probe = time_burnin_without_sampling_timing_probe,
                       burnin                                    = burnin_object$time_criterion) else NULL,
                   ## tau ADAM bias correction counts PERFORMED updates (skipped iterations excluded); final counts per block:
                   tau_adam_bias_correction = burnin_object$tau_adam_bias_correction,
                   tau_adam_updates_main = burnin_object$tau_adam_updates_main,
                   tau_adam_updates_us = burnin_object$tau_adam_updates_us,
                   n_chains_burnin = n_chains_burnin,
                   burnin_TBB_pool_equals_n_chains = burnin_TBB_pool_equals_n_chains,
                   n_burnin = n_burnin,
                   ## the pre-burn-in actually run (0 iterations when it did not run), for the gradient count:
                   pre_burnin_n_iter_run = if (!exists("pre_burnin_object", inherits = FALSE) ||
                                               is.null(pre_burnin_object)) 0 else pre_burnin_params$n_burnin,
                   pre_burnin_L = pre_burnin_L,
                   pre_burnin_L_main_during_burnin_vec = if (!exists("pre_burnin_object", inherits = FALSE) ||
                                                             is.null(pre_burnin_object)) NULL else
                                                             pre_burnin_object$L_main_during_burnin_vec,
                   metric_type_main = metric_type_main,
                   metric_shape_main = metric_shape_main,
                   metric_type_nuisance = metric_type_nuisance,
                   metric_shape_nuisance = metric_shape_nuisance,
                   theta_hat_us_rule = theta_hat_us_rule,
                   ##
                   diffusion_HMC = diffusion_HMC,
                   diffusion_HMC_integrator = diffusion_HMC_integrator,
                   partitioned_HMC = partitioned_HMC,
                   n_superchains = n_superchains,
                   nested_rhat_grouping = nested_rhat_grouping,
                   interval_width_main = interval_width_main,
                   interval_width_nuisance = interval_width_nuisance,
                   ##
                   force_autodiff = force_autodiff,
                   force_PartialLog = force_PartialLog,
                   multi_attempts = multi_attempts,
                   autodiff_fallback = autodiff_fallback,
                   ##
                   time_burnin = time_burnin,
                   time_sampling = time_sampling,
                   time_total = time_total,
                   ##
                   n_threads_WCP_burnin = n_threads_WCP_burnin,
                   num_chunks_burnin = num_chunks_burnin,
                   burnin_n_threads_WCP_and_num_chunks_rule = burnin_n_threads_WCP_and_num_chunks_rule)
  
  
  return(out_list)

}























