#### =====================================================================================================================================
## R_fn_time_criterion.R
##
## ---- Time-to-target-ESS trajectory-length criteria: burnin_algorithm = "CHESSR_time" and "SNAPER_time" -------------------------------
##
## The rate criteria CHESSR (ChEES-rate) and SNAPER (Sountsov & Hoffman, 2022, eq. (2) and (5)) choose tau to maximise the
## expected squared-jump statistic per unit of integration time, i.e. per leapfrog step. They charge a trajectory for its
## leapfrog steps only. The two "_time" criteria below choose tau to minimise the WALL-CLOCK time the whole run needs to reach
## a target ESS, which also charges (i) the fixed non-leapfrog wall-clock time of every sampling iteration (the initial log density and
## gradient, the per-iteration bookkeeping, the trace storage and the summaries), and (ii) the burn-in's own leapfrog steps.
##
## ---- Time model (every timing quantity is a wall-clock time in seconds, measured for ALL chains of a phase running in parallel):
##
##   n_leapfrog_steps_per_iter  = tau / eps
##
##   time_to_target_ESS(tau)    = time_burnin_not_depending_on_tau
##                                + n_iter_burnin * (time_per_iter_overhead_burnin + n_leapfrog_steps_per_iter * time_per_leapfrog_step_burnin)
##                                + (ESS_target_for_time_criterion / ESS_per_iter_sampling(tau))
##                                  * (time_per_iter_overhead_sampling + time_per_iter_summaries_sampling
##                                     + n_leapfrog_steps_per_iter * time_per_leapfrog_step_sampling)
##
##   n_iter_burnin is the number of main burn-in iterations whose number of leapfrog steps is set by the adapted tau: iterations
##   clip_iter_tau + 1 to n_burnin, i.e. n_burnin - clip_iter_tau (the tau update at iteration ii sets the tau of iteration ii + 1).
##   time_burnin_not_depending_on_tau is the rest of the burn-in: the pre-burnin (pre_burnin_L leapfrog steps), the iterations before
##   clip_iter (one leapfrog step) and the ramp to tau_initial up to the handover at clip_iter_tau. Their wall time is the same whatever
##   tau is chosen, so they drop out of the derivative below, and n_iter_burnin stays fixed for the whole adaptation.
##
##   ESS_target_for_time_criterion / ESS_per_iter_sampling(tau) is the number of sampling iterations needed, called
##   n_iter_sampling_for_time_criterion below (evaluated at the current tau and then held fixed, see "Measurements").
##
## ---- Stationary point in log(tau). Write
##
##   ESS_elasticity_wrt_log_tau              = d log(ESS_per_iter_sampling) / d log(tau)          (what ChEES / SNAPER estimate)
##   sampling_overhead_in_leapfrog_steps     = (time_per_iter_overhead_sampling + time_per_iter_summaries_sampling)
##                                             / time_per_leapfrog_step_sampling
##   tau_offset_from_sampling_overhead       = eps * sampling_overhead_in_leapfrog_steps
##   burnin_to_sampling_leapfrog_time_ratio  = (n_iter_burnin * time_per_leapfrog_step_burnin)
##                                             / (n_iter_sampling_for_time_criterion * time_per_leapfrog_step_sampling)
##
##   d time_to_target_ESS / d log(tau)
##     = n_leapfrog_steps_per_iter * (n_iter_burnin * time_per_leapfrog_step_burnin
##                                    + n_iter_sampling_for_time_criterion * time_per_leapfrog_step_sampling)
##       - ESS_elasticity_wrt_log_tau * n_iter_sampling_for_time_criterion
##         * (time_per_iter_overhead_sampling + time_per_iter_summaries_sampling + n_leapfrog_steps_per_iter * time_per_leapfrog_step_sampling),
##
##   and setting it to zero and dividing by n_iter_sampling_for_time_criterion * time_per_leapfrog_step_sampling / eps gives
##
##     ESS_elasticity_wrt_log_tau = time_to_target_ESS_tau_penalty
##                                = (1 + burnin_to_sampling_leapfrog_time_ratio) * tau / (tau + tau_offset_from_sampling_overhead).
##
##   time_per_iter_overhead_burnin and time_burnin_not_depending_on_tau do not depend on tau, so they drop out; neither is measured.
##
## ---- The criteria. The rate criteria (CHESSR, SNAPER) ascend  ESS_elasticity_wrt_log_tau - 1  with ADAM on log(tau); per chain
##      (fn_metric_position_criterion) the ascended quantity is  (numerator_gradient - numerator) / tau_values, where numerator is
##      the chain's squared-jump statistic and numerator_gradient its derivative with respect to log(tau). The "_time" criteria
##      replace the 1 by the penalty, per chain at the chain's own tau_values:
##
##        gradient = (numerator_gradient - numerator * time_to_target_ESS_tau_penalty) / tau_values.
##
##      The acceptance weighting, the mean over the chains, the skip rule and the ADAM step are those of CHESSR / SNAPER.
##
## ---- Special cases:
##      burnin_to_sampling_leapfrog_time_ratio = 0 and sampling_overhead_in_leapfrog_steps = 0: the penalty is tau / tau = 1 exactly
##        (IEEE division of a finite positive number by itself), and numerator * 1 = numerator, so CHESSR_time and SNAPER_time are
##        byte-identical to CHESSR and SNAPER.
##      sampling_overhead_in_leapfrog_steps -> infinity (a per-iteration non-leapfrog wall-clock time that dwarfs the trajectory): the penalty -> 0 and the
##        ascended quantity -> numerator_gradient / tau_values, the ChEES gradient up to the per-chain weights 1 / tau_values
##        (the same weight for every chain when the burn-in chains share tau_ii), i.e. the ChEES optimum.
##      (The first holds at lag_one_autocorrelation_rho = 1, the default: with lag_one_autocorrelation_rho < 1 the penalty is also
##      multiplied by (1 + lag_one_autocorrelation_rho) / 2, see "MALT exponent" below. The second holds for any lag_one_autocorrelation_rho.)
##
## ---- MALT exponent, lag_one_autocorrelation_rho (Riou-Durand, Sountsov, Vogrinc, Margossian and Power, 2023, AISTATS, "Adaptive
##      tuning for Metropolis adjusted Langevin trajectories", section 3.4 and appendix A). Mapping to MALT's notation, used only
##      here: MALT's phi is criterion_statistic below; its rho is lag_one_autocorrelation_rho; its ESJD_phi is
##      ESJD_of_criterion_statistic; its C_rho is the family in (i); its a, beta, m2, s2 and c are
##      lag_one_autocorrelation_rho_moment_averaging_offset, running_moment_weight, mean_criterion_statistic_at_accepted_end,
##      variance_criterion_statistic_at_accepted_end and lag_one_autocovariance_criterion_statistic
##      (fn_lag_one_autocorrelation_rho_moments_update).
##
##      criterion_statistic is the statistic whose squared jump the criterion scores, of the position in metric coordinates about the
##      running mean: CHESSR: 0.5 * |position_in_metric_coordinates|^2; SNAPER: (direction' position_in_metric_coordinates)^2.
##      ESJD_of_criterion_statistic(tau) is its expected squared jump over one iteration, and
##
##        ESJD_elasticity_wrt_log_tau = d log(ESJD_of_criterion_statistic) / d log(tau)
##
##      is the quantity the ChEES / SNAPER gradient estimates (numerator_gradient / numerator, per chain).
##
##      (i)   MALT's family
##
##              ESJD_of_criterion_statistic(tau) / tau^((1 + lag_one_autocorrelation_rho) / 2),   lag_one_autocorrelation_rho in [0, 1],
##
##            has its stationary point in log(tau) at  ESJD_elasticity_wrt_log_tau = (1 + lag_one_autocorrelation_rho) / 2;
##            lag_one_autocorrelation_rho = 1 is the rate criterion.
##
##      (ii)  MALT's model (section 3.4 and appendix A, Proposition 1; exact when the autocorrelations of criterion_statistic decay
##            geometrically, e.g. on Gaussian targets), with lag_one_autocorrelation_rho(tau) the correlation of criterion_statistic at the
##            start of an iteration and at its accepted end:
##
##              ESS_per_iter_sampling(tau)        proportional to  (1 - lag_one_autocorrelation_rho(tau)) / (1 + lag_one_autocorrelation_rho(tau)),
##              ESJD_of_criterion_statistic(tau)  proportional to   1 - lag_one_autocorrelation_rho(tau).
##
##            Differentiating both logarithms in log(tau), d log(ESS_per_iter_sampling) = -2 d lag_one_autocorrelation_rho / (1 - lag_one_autocorrelation_rho^2)
##            and d log(ESJD_of_criterion_statistic) = -d lag_one_autocorrelation_rho / (1 - lag_one_autocorrelation_rho), so that
##
##              ESS_elasticity_wrt_log_tau = (2 / (1 + lag_one_autocorrelation_rho)) * ESJD_elasticity_wrt_log_tau.
##
##            At lag_one_autocorrelation_rho = 1 the two elasticities are equal, which is how the time model above reads
##            ESS_elasticity_wrt_log_tau ("what ChEES / SNAPER estimate").
##
##      (iii) Substituting (ii) into the stationary point of the time model above (ESS_elasticity_wrt_log_tau = time_to_target_ESS_tau_penalty),
##            for any burnin_to_sampling_leapfrog_time_ratio, gives the condition on what the criteria estimate:
##
##              ESJD_elasticity_wrt_log_tau = ((1 + lag_one_autocorrelation_rho) / 2) * time_to_target_ESS_tau_penalty
##                                          = ((1 + lag_one_autocorrelation_rho) / 2) * (1 + burnin_to_sampling_leapfrog_time_ratio)
##                                            * tau / (tau + tau_offset_from_sampling_overhead),
##
##            i.e. the penalty of the ascended quantity (fn_time_to_target_ESS_tau_penalty) is time_to_target_ESS_tau_penalty times
##            (1 + lag_one_autocorrelation_rho) / 2. With lag_one_autocorrelation_rho = 1 it is time_to_target_ESS_tau_penalty itself.
##
##      (iv)  With burnin_to_sampling_leapfrog_time_ratio = 0, (iii) is the stationary point in log(tau) of
##
##              ESJD_of_criterion_statistic(tau) / (tau + eps * sampling_overhead_in_leapfrog_steps)^((1 + lag_one_autocorrelation_rho) / 2),
##
##            and with sampling_overhead_in_leapfrog_steps = 0 as well it is the family in (i).
##
##      lag_one_autocorrelation_rho is the setting of the same name:
##        a number in [0, 1]: 1 (default) gives (1 + 1) / 2 = 1 exactly, and 1 * x = x in IEEE arithmetic, so the penalty, and
##          CHESSR_time / SNAPER_time, are byte-identical to the criterion without lag_one_autocorrelation_rho above;
##        "adaptive": estimated during the tau adaptation as the lag-one autocorrelation of criterion_statistic across the burn-in chains,
##          with MALT's running moments (Algorithm 3, lines 6, 11 and 20-22; see fn_lag_one_autocorrelation_rho_moments_update), clipped
##          to [0, 1].
##      lag_one_autocorrelation_rho enters CHESSR_time and SNAPER_time only; the rate criteria CHESSR and SNAPER keep the exponent 1.
##
## ---- tau_sampling_scale = "gaussian_matched" uses the CHESSR / SNAPER unit-Gaussian factor for the "_time" criteria (see
##      fn_tau_sampling_scale_gaussian_factor). With a zero penalty shift that factor is exact; with a positive
##      tau_offset_from_sampling_overhead the jittered optimum lies between the ChEES and the rate factors, and the shift is not
##      a unit-Gaussian constant, so the rate factor is the documented approximation.
##
## ---- Measurements (see fn_run_sampling_timing_probe, fn_time_criterion_quantities_from_saved_run and the burn-in):
##      time_per_leapfrog_step_sampling, time_per_iter_overhead_sampling, time_per_iter_summaries_sampling: a short sampling timing
##        probe before the main burn-in, or a previous saved run, or values supplied by the user (in that order of precedence:
##        user-supplied > previous run > probe).
##      time_per_leapfrog_step_burnin: measured online during the burn-in (slope of a running least-squares fit of the iteration time
##        on its number of leapfrog steps, see fn_time_per_leapfrog_step_burnin_fit_update), or supplied by the user.
##      n_iter_burnin: n_burnin - clip_iter_tau (see the time model above).
##      n_iter_sampling_for_time_criterion: supplied by the user; else ESS_target_for_time_criterion / ESS_per_iter_sampling_expected
##        when an ESS target is given; else the planned n_iter of the run.
##      Which ESS: ESS_target_for_time_criterion and ESS_per_iter_sampling_expected refer to the minimum ESS over the parameters chosen
##        by time_criterion_ess_parameter_set: "diagnostic" (default) = the model's diagnostic parameters, i.e. the generated
##        quantities of its Stan skeleton file (for LC_MVP the accuracy parameters p[1..2], Se_baseline[..], Sp_baseline[..] and
##        Fp_baseline[..]), never the parameters block; "main" = the parameters block; or explicit parameter names. The set is used
##        when ESS_per_iter_sampling_expected is read from previous saved run(s) (fn_time_criterion_min_ESS_of_saved_run); a
##        user-supplied ESS_per_iter_sampling_expected must refer to the same set.
##      Nothing available: burnin_to_sampling_leapfrog_time_ratio = 0 and sampling_overhead_in_leapfrog_steps = 0, i.e. plain
##        CHESSR / SNAPER, with a warning.
##      sampling_overhead_in_leapfrog_steps = "auto": a value for the SAMPLING configuration of the run (n_chains_sampling chains,
##        n_threads_WCP_sampling threads per chain, the sampling chunk count), resolved before the main burn-in and fixed for all of
##        it: (a) the built-in default for the model, that configuration and the machine's number of physical cores when one exists
##        (fn_default_sampling_overhead_in_leapfrog_steps), else (b) the sampling timing probe's own three times, with the probe in its
##        cut-down mode, sampling_timing_probe_mode = "lite" (unless sampling_timing_probe_mode = "full" is set). It is never taken from
##        a saved run or from the burn-in's own timings: the burn-in runs another configuration (n_chains_burnin chains,
##        n_threads_WCP_burnin threads per chain, the burn-in chunk count), whose within-chain parallelism and chunk count change the
##        per-iteration costs. User-supplied time_per_iter_overhead_sampling and time_per_iter_summaries_sampling do not enter it (NULL
##        uses them). A time_criterion_previous_run_path that is supplied is still read, for time_per_leapfrog_step_sampling
##        (burnin_to_sampling_leapfrog_time_ratio) and ESS_per_iter_sampling_expected only. With the built-in default, as with a number,
##        the time_per_leapfrog_step_sampling of burnin_to_sampling_leapfrog_time_ratio is the user's, else a previous run's; with
##        neither, burnin_to_sampling_leapfrog_time_ratio = 0, with a warning (the default's calibrated leapfrog time is of the
##        calibration machine and is not used). A probe with no summaries time (use_disk = TRUE, sampling_timing_probe_time_summaries =
##        FALSE, a failed routine or timing noise) gives the iteration overhead part only, with a warning and the source
##        "probe_lite_without_summaries" (or "probe_full_without_summaries").
##      sampling_overhead_in_leapfrog_steps supplied as a number: used as it is, and the sampling timing probe does not run. The
##        time_per_leapfrog_step_sampling of burnin_to_sampling_leapfrog_time_ratio is then the user's, else a previous run's; with
##        neither, burnin_to_sampling_leapfrog_time_ratio = 0, with a warning.
##
## ---- Scope: the joint sampler (partitioned_HMC = FALSE), whose single tau sets the number of leapfrog steps of every
##      gradient evaluation. With partitioned_HMC = TRUE the main and nuisance trajectories have separate taus and step counts,
##      which this time model does not describe, so the burn-in refuses the "_time" criteria there.
##


##
## ---- Algorithm names -------------------------------------------------------------------------------------------------------------
##
fn_burnin_algorithm_is_time_criterion <-  function(burnin_algorithm) {

        return(length(burnin_algorithm) == 1 && !is.na(burnin_algorithm) &&
               burnin_algorithm %in% c("CHESSR_time", "SNAPER_time"))

}


##
## ---- The penalty that replaces the 1 of the rate criteria (vectorised over chains) ----------------------------------------------
##
fn_time_to_target_ESS_tau_penalty <-  function( tau_values,
                                                tau_offset_from_sampling_overhead,
                                                ## burnin_to_sampling_leapfrog_time_ratio) {
                                                burnin_to_sampling_leapfrog_time_ratio,
                                                ##
                                                ## ---- MALT exponent (see "MALT exponent" above): the penalty times (1 + lag_one_autocorrelation_rho) / 2,
                                                ##      which is exactly 1 at the default 1 and leaves the penalty unchanged bit for bit:
                                                lag_one_autocorrelation_rho = 1) {

        ## return((1 + burnin_to_sampling_leapfrog_time_ratio) * tau_values / (tau_values + tau_offset_from_sampling_overhead))
        time_to_target_ESS_tau_penalty_at_rho_one <-  (1 + burnin_to_sampling_leapfrog_time_ratio) * tau_values / (tau_values + tau_offset_from_sampling_overhead)
        return(((1 + lag_one_autocorrelation_rho) / 2) * time_to_target_ESS_tau_penalty_at_rho_one)

}


##
## ---- Settings: defaults and validation -------------------------------------------------------------------------------------------
##
##      time_criterion_settings is one list; every element is optional (NULL = default). Elements:
##
##        run_sampling_timing_probe                     TRUE (default) / FALSE: run the sampling timing probe before the main burn-in
##                                                      when a sampling quantity is neither supplied nor available from a previous run.
##        sampling_timing_probe_L_values                fixed leapfrog-step counts of the probe; default c(2, 8).
##        sampling_timing_probe_n_iter_per_L            iterations of the long probe call at each step count; default 10.
##        sampling_timing_probe_n_iter_short_call       iterations of the short probe call at each step count; default 3. The
##                                                      difference between the two calls removes the per-call set-up time.
##        sampling_timing_probe_n_iter_warm_up_call     iterations of the untimed warm-up call made before the timed calls; default 2.
##        sampling_timing_probe_max_wall_time           seconds; the probe makes no further call once its wall time exceeds this.
##                                                      NULL (default) = no limit.
##        sampling_timing_probe_time_summaries          TRUE (default) / FALSE: time the summaries routine on the probe draws.
##        sampling_timing_probe_summaries_n_iter_tiled_short_call, sampling_timing_probe_summaries_n_iter_tiled_long_call
##                                                      numbers of draw rows (iterations over all chains) the probe draws are tiled to
##                                                      for the two timed summaries calls; defaults 10 and 100.
##        sampling_timing_probe_summary_arguments       arguments for create_summary_and_traces() on the probe draws; default: the
##                                                      arguments pilot study 7 uses for its summaries.
##        time_criterion_previous_run_path              path(s) of saved run(s) used instead of the probe (see
##                                                      fn_time_criterion_quantities_from_saved_run).
##        time_per_leapfrog_step_sampling               user-supplied values (seconds), which take precedence over the previous run
##        time_per_iter_overhead_sampling                 and the probe.
##        time_per_iter_summaries_sampling
##        sampling_overhead_in_leapfrog_steps           user-supplied directly (takes precedence over the value built from the
##                                                      three times above).
##        time_per_leapfrog_step_burnin                 user-supplied value (seconds); replaces the online burn-in measurement.
##        time_per_leapfrog_step_burnin_min_iterations  burn-in iterations (after clip_iter) before the online fit is used; default 20.
##        ESS_target_for_time_criterion                 target ESS over all sampling chains.
##        ESS_per_iter_sampling_expected                expected ESS per sampling iteration over all chains.
##        n_iter_sampling_for_time_criterion            sampling iterations the criterion assumes (user-supplied).
##        time_criterion_ess_parameter_set              the parameters whose minimum ESS the ESS target and ESS_per_iter_sampling_expected
##                                                      refer to: "diagnostic" (default; the model's diagnostic parameters = the
##                                                      generated quantities of its Stan skeleton file, e.g. the accuracy parameters
##                                                      Se, Sp and prevalence), "main" (the parameters block), or a character vector of
##                                                      parameter names ("Se_baseline[1]") and / or base names ("Se_baseline" = every
##                                                      element). Used to read ESS_per_iter_sampling_expected from previous saved run(s).
##        sampling_overhead_in_leapfrog_steps           also "auto": a value for the sampling configuration (see "Measurements" above): the
##                                                      built-in default for the model, that configuration and the machine's number of
##                                                      physical cores when one exists
##                                                      (fn_default_sampling_overhead_in_leapfrog_steps), else the sampling timing probe
##                                                      in the mode sampling_timing_probe_mode ("lite" by default); never a saved run or
##                                                      the burn-in's timings. A time_criterion_previous_run_path that is supplied is still
##                                                      read, for time_per_leapfrog_step_sampling and ESS_per_iter_sampling_expected only.
##                                                      A number (>= 0) is used as it is and runs no probe. NULL (default) = the
##                                                      precedence user times > previous run > probe.
##        sampling_timing_probe_mode                    NULL (default) = "lite" with sampling_overhead_in_leapfrog_steps = "auto", else
##                                                      "full"; "full" = the probe described below (every value of
##                                                      sampling_timing_probe_L_values; the summaries timed by four calls); "lite" = the
##                                                      same sampling configuration and native entry point, with the timed calls at the
##                                                      smallest and the largest value of sampling_timing_probe_L_values only, and the
##                                                      summaries timed once on the long tiled draws, less the routine's fixed set-up,
##                                                      which is measured once per R session and cached (see "Lite mode" in the probe's
##                                                      description). With the default sampling_timing_probe_L_values = c(2, 8) the
##                                                      native calls are the full probe's, and a lite probe whose set-up is not yet
##                                                      cached (the first of its configuration in the R session, and every one with
##                                                      run_in_fresh_R_process = TRUE) makes the full probe's four summaries calls, so
##                                                      it then costs as much as the full probe.
##        lag_one_autocorrelation_rho                   MALT's exponent (1 + lag_one_autocorrelation_rho) / 2 on the cost (see "MALT exponent"
##                                                      above): a number in [0, 1], 1 (default) = the criterion without it; or "adaptive" =
##                                                      estimated during the tau adaptation as the lag-one autocorrelation of the criterion's
##                                                      statistic across the burn-in chains. CHESSR_time / SNAPER_time only.
##        lag_one_autocorrelation_rho_moment_averaging_offset
##                                                      "adaptive" only: the offset in the running-moment weight
##                                                      running_moment_weight = n_earlier_moment_updates / (n_earlier_moment_updates +
##                                                      lag_one_autocorrelation_rho_moment_averaging_offset); default 8 (MALT's value);
##                                                      1 = equal weights over the tau updates; larger values forget the early updates faster.
##
fn_default_time_criterion_settings <-  function() {

        return(list( run_sampling_timing_probe                    = TRUE,
                     sampling_timing_probe_L_values               = c(2, 8),
                     sampling_timing_probe_n_iter_per_L           = 10,
                     sampling_timing_probe_n_iter_short_call      = 3,
                     sampling_timing_probe_n_iter_warm_up_call    = 2,
                     sampling_timing_probe_max_wall_time          = NULL,
                     sampling_timing_probe_time_summaries         = TRUE,
                     sampling_timing_probe_summaries_n_iter_tiled_short_call = 10,
                     sampling_timing_probe_summaries_n_iter_tiled_long_call  = 100,
                     sampling_timing_probe_summary_arguments      = list( compute_main_params            = TRUE,
                                                                          compute_transformed_parameters = TRUE,
                                                                          compute_generated_quantities   = TRUE,
                                                                          save_log_lik_trace             = FALSE,
                                                                          save_nuisance_trace            = FALSE,
                                                                          compute_nested_rhat            = TRUE,
                                                                          n_superchains                  = NULL,
                                                                          save_trace_tibbles             = FALSE),
                     time_criterion_previous_run_path             = NULL,
                     time_per_leapfrog_step_sampling              = NULL,
                     time_per_iter_overhead_sampling              = NULL,
                     time_per_iter_summaries_sampling             = NULL,
                     sampling_overhead_in_leapfrog_steps          = NULL,
                     time_per_leapfrog_step_burnin                = NULL,
                     time_per_leapfrog_step_burnin_min_iterations = 20,
                     ESS_target_for_time_criterion                = NULL,
                     ESS_per_iter_sampling_expected               = NULL,
                     n_iter_sampling_for_time_criterion           = NULL,
                     ## time_criterion_ess_parameter_set             = "diagnostic"))
                     time_criterion_ess_parameter_set             = "diagnostic",
                     lag_one_autocorrelation_rho                  = 1,
                     lag_one_autocorrelation_rho_moment_averaging_offset = 8,
                     sampling_timing_probe_mode                   = NULL))

}

fn_validate_time_criterion_settings <-  function(time_criterion_settings) {

        default_settings <-  fn_default_time_criterion_settings()
        if (is.null(time_criterion_settings)) return(default_settings)
        if (!is.list(time_criterion_settings) || (length(time_criterion_settings) > 0 && is.null(names(time_criterion_settings))) ||
            any(names(time_criterion_settings) == "")) {
              stop("time_criterion_settings must be NULL or a named list (see fn_default_time_criterion_settings()).")
        }
        unknown_setting_names <-  setdiff(names(time_criterion_settings), names(default_settings))
        if (length(unknown_setting_names) > 0) {
              stop(paste0("time_criterion_settings: unknown element(s) ", paste(unknown_setting_names, collapse = ", "),
                          ". Known elements: ", paste(names(default_settings), collapse = ", "), "."))
        }
        ##
        validated_settings <-  default_settings
        for (setting_name in names(time_criterion_settings)) {
              if (!is.null(time_criterion_settings[[setting_name]])) {
                    validated_settings[[setting_name]] <-  time_criterion_settings[[setting_name]]
              }
        }
        ##
        fn_is_single_logical <-  function(setting_value) is.logical(setting_value) && length(setting_value) == 1 && !is.na(setting_value)
        fn_is_single_positive_number <-  function(setting_value) {
              is.numeric(setting_value) && length(setting_value) == 1 && is.finite(setting_value) && setting_value > 0
        }
        fn_is_single_non_negative_number <-  function(setting_value) {
              is.numeric(setting_value) && length(setting_value) == 1 && is.finite(setting_value) && setting_value >= 0
        }
        fn_is_single_whole_number_at_least <-  function(setting_value, minimum_whole_number) {
              fn_is_single_positive_number(setting_value) && setting_value == round(setting_value) && setting_value >= minimum_whole_number
        }
        ##
        for (setting_name in c("run_sampling_timing_probe", "sampling_timing_probe_time_summaries")) {
              if (!fn_is_single_logical(validated_settings[[setting_name]])) {
                    stop(paste0("time_criterion_settings$", setting_name, " must be TRUE or FALSE."))
              }
        }
        ##
        sampling_timing_probe_L_values <-  validated_settings$sampling_timing_probe_L_values
        if (!is.numeric(sampling_timing_probe_L_values) || length(sampling_timing_probe_L_values) < 2 ||
            any(!is.finite(sampling_timing_probe_L_values)) || any(sampling_timing_probe_L_values < 1) ||
            any(sampling_timing_probe_L_values != round(sampling_timing_probe_L_values)) ||
            length(unique(sampling_timing_probe_L_values)) < 2) {
              stop("time_criterion_settings$sampling_timing_probe_L_values must hold at least two different whole numbers >= 1.")
        }
        validated_settings$sampling_timing_probe_L_values <-  sort(unique(as.numeric(sampling_timing_probe_L_values)))
        ##
        if (!fn_is_single_whole_number_at_least(validated_settings$sampling_timing_probe_n_iter_short_call, 1)) {
              stop("time_criterion_settings$sampling_timing_probe_n_iter_short_call must be a single whole number >= 1.")
        }
        if (!fn_is_single_whole_number_at_least(validated_settings$sampling_timing_probe_n_iter_per_L,
                                                validated_settings$sampling_timing_probe_n_iter_short_call + 1)) {
              stop("time_criterion_settings$sampling_timing_probe_n_iter_per_L must be a single whole number greater than ",
                   "sampling_timing_probe_n_iter_short_call.")
        }
        if (!fn_is_single_whole_number_at_least(validated_settings$sampling_timing_probe_n_iter_warm_up_call, 1)) {
              stop("time_criterion_settings$sampling_timing_probe_n_iter_warm_up_call must be a single whole number >= 1.")
        }
        if (!fn_is_single_whole_number_at_least(validated_settings$sampling_timing_probe_summaries_n_iter_tiled_short_call, 1)) {
              stop("time_criterion_settings$sampling_timing_probe_summaries_n_iter_tiled_short_call must be a single whole number >= 1.")
        }
        if (!fn_is_single_whole_number_at_least(validated_settings$sampling_timing_probe_summaries_n_iter_tiled_long_call,
                                                validated_settings$sampling_timing_probe_summaries_n_iter_tiled_short_call + 1)) {
              stop("time_criterion_settings$sampling_timing_probe_summaries_n_iter_tiled_long_call must be a single whole number greater than ",
                   "sampling_timing_probe_summaries_n_iter_tiled_short_call.")
        }
        if (!is.null(validated_settings$sampling_timing_probe_max_wall_time) &&
            !fn_is_single_positive_number(validated_settings$sampling_timing_probe_max_wall_time)) {
              stop("time_criterion_settings$sampling_timing_probe_max_wall_time must be NULL or a single positive number of seconds.")
        }
        if (!is.list(validated_settings$sampling_timing_probe_summary_arguments)) {
              stop("time_criterion_settings$sampling_timing_probe_summary_arguments must be a named list of create_summary_and_traces() arguments.")
        }
        if (!is.null(validated_settings$time_criterion_previous_run_path) &&
            (!is.character(validated_settings$time_criterion_previous_run_path) ||
             length(validated_settings$time_criterion_previous_run_path) < 1 ||
             anyNA(validated_settings$time_criterion_previous_run_path))) {
              stop("time_criterion_settings$time_criterion_previous_run_path must be NULL or one or more file paths.")
        }
        ##
        for (setting_name in c("time_per_leapfrog_step_sampling", "time_per_leapfrog_step_burnin",
                               "ESS_target_for_time_criterion", "ESS_per_iter_sampling_expected",
                               "n_iter_sampling_for_time_criterion")) {
              if (!is.null(validated_settings[[setting_name]]) && !fn_is_single_positive_number(validated_settings[[setting_name]])) {
                    stop(paste0("time_criterion_settings$", setting_name, " must be NULL or a single positive number."))
              }
        }
        ## for (setting_name in c("time_per_iter_overhead_sampling", "time_per_iter_summaries_sampling",
        ##                        "sampling_overhead_in_leapfrog_steps")) {
        for (setting_name in c("time_per_iter_overhead_sampling", "time_per_iter_summaries_sampling")) {
              if (!is.null(validated_settings[[setting_name]]) && !fn_is_single_non_negative_number(validated_settings[[setting_name]])) {
                    stop(paste0("time_criterion_settings$", setting_name, " must be NULL or a single number >= 0."))
              }
        }
        ##
        ## ---- sampling_overhead_in_leapfrog_steps: NULL, a number >= 0 (leapfrog-step equivalents) or "auto" (the built-in default for the
        ##      sampling configuration, else the sampling timing probe):
        ##
        sampling_overhead_in_leapfrog_steps_setting <-  validated_settings$sampling_overhead_in_leapfrog_steps
        if (!is.null(sampling_overhead_in_leapfrog_steps_setting) && !identical(sampling_overhead_in_leapfrog_steps_setting, "auto") &&
            !fn_is_single_non_negative_number(sampling_overhead_in_leapfrog_steps_setting)) {
              stop(paste0("time_criterion_settings$sampling_overhead_in_leapfrog_steps must be NULL (user times > previous run > probe), ",
                          "a single number >= 0 (leapfrog-step equivalents; no probe) or \"auto\" (the built-in default for the sampling ",
                          "configuration, else the sampling timing probe in its \"lite\" mode). Got: ",
                          paste(format(sampling_overhead_in_leapfrog_steps_setting), collapse = ", "), "."))
        }
        ##
        ## ---- sampling_timing_probe_mode: NULL ("lite" with sampling_overhead_in_leapfrog_steps = "auto", else "full"), "full" or "lite":
        ##
        sampling_timing_probe_mode_setting <-  validated_settings$sampling_timing_probe_mode
        if (!is.null(sampling_timing_probe_mode_setting) &&
            !(is.character(sampling_timing_probe_mode_setting) && length(sampling_timing_probe_mode_setting) == 1 &&
              sampling_timing_probe_mode_setting %in% c("full", "lite"))) {
              stop(paste0("time_criterion_settings$sampling_timing_probe_mode must be NULL (\"lite\" with sampling_overhead_in_leapfrog_steps = ",
                          "\"auto\", else \"full\"), \"full\" or \"lite\". Got: ", paste(format(sampling_timing_probe_mode_setting), collapse = ", "), "."))
        }
        ##
        ## ---- lag_one_autocorrelation_rho: a number in [0, 1] or "adaptive"; lag_one_autocorrelation_rho_moment_averaging_offset >= 1:
        ##
        lag_one_autocorrelation_rho_setting <-  validated_settings$lag_one_autocorrelation_rho
        lag_one_autocorrelation_rho_is_number_in_unit_interval <-  is.numeric(lag_one_autocorrelation_rho_setting) &&
                                                                   length(lag_one_autocorrelation_rho_setting) == 1 &&
                                                                   is.finite(lag_one_autocorrelation_rho_setting) &&
                                                                   lag_one_autocorrelation_rho_setting >= 0 && lag_one_autocorrelation_rho_setting <= 1
        if (!identical(lag_one_autocorrelation_rho_setting, "adaptive") && !lag_one_autocorrelation_rho_is_number_in_unit_interval) {
              stop(paste0("time_criterion_settings$lag_one_autocorrelation_rho must be a single number in [0, 1] (MALT's exponent ",
                          "(1 + lag_one_autocorrelation_rho) / 2; ",
                          "1 = the criterion without it) or \"adaptive\" (the lag-one autocorrelation of the criterion's statistic, estimated ",
                          "during the tau adaptation). Got: ", paste(format(lag_one_autocorrelation_rho_setting), collapse = ", "), "."))
        }
        lag_one_autocorrelation_rho_moment_averaging_offset <-  validated_settings$lag_one_autocorrelation_rho_moment_averaging_offset
        if (!(is.numeric(lag_one_autocorrelation_rho_moment_averaging_offset) && length(lag_one_autocorrelation_rho_moment_averaging_offset) == 1 &&
              is.finite(lag_one_autocorrelation_rho_moment_averaging_offset) && lag_one_autocorrelation_rho_moment_averaging_offset >= 1)) {
              stop(paste0("time_criterion_settings$lag_one_autocorrelation_rho_moment_averaging_offset must be a single number >= 1 (the offset in ",
                          "the running-moment weight n_earlier_moment_updates / (n_earlier_moment_updates + lag_one_autocorrelation_rho_moment_averaging_offset); ",
                          "1 = equal weights over the tau updates). Got: ",
                          paste(format(lag_one_autocorrelation_rho_moment_averaging_offset), collapse = ", "), "."))
        }
        if (!fn_is_single_whole_number_at_least(validated_settings$time_per_leapfrog_step_burnin_min_iterations, 2)) {
              stop("time_criterion_settings$time_per_leapfrog_step_burnin_min_iterations must be a single whole number >= 2.")
        }
        time_criterion_ess_parameter_set <-  validated_settings$time_criterion_ess_parameter_set
        if (!is.character(time_criterion_ess_parameter_set) || length(time_criterion_ess_parameter_set) < 1 ||
            anyNA(time_criterion_ess_parameter_set) || any(!nzchar(trimws(time_criterion_ess_parameter_set))) ||
            (length(time_criterion_ess_parameter_set) > 1 && any(time_criterion_ess_parameter_set %in% c("diagnostic", "main")))) {
              stop(paste0("time_criterion_settings$time_criterion_ess_parameter_set must be \"diagnostic\", \"main\" or a character vector of ",
                          "parameter names / base names (e.g. c(\"p\", \"Se_baseline\", \"Sp_baseline\"))."))
        }
        validated_settings$time_criterion_ess_parameter_set <-  trimws(time_criterion_ess_parameter_set)
        ##
        return(validated_settings)

}


##
## ---- Least-squares fit of a per-iteration wall time on the number of leapfrog steps per iteration ----------------------------------
##
##      time_per_iter = time_per_iter_overhead + n_leapfrog_steps_per_iter * time_per_leapfrog_step (ordinary least squares).
##      Status: "fitted"; "not_identified" (fewer than two different step counts); "non_positive_slope" (the leapfrog time is not
##      identified as positive). A negative intercept is set to 0 and reported ("fitted_intercept_set_to_zero").
##
fn_fit_time_per_iter_on_n_leapfrog_steps <-  function( n_leapfrog_steps_per_iter,
                                                       time_per_iter) {

        observation_is_finite <-  is.finite(n_leapfrog_steps_per_iter) & is.finite(time_per_iter)
        n_leapfrog_steps_per_iter <-  n_leapfrog_steps_per_iter[observation_is_finite]
        time_per_iter <-  time_per_iter[observation_is_finite]
        ##
        time_per_iter_least_squares_fit <-  list( time_per_iter_overhead  = NA_real_,
                                                  time_per_leapfrog_step  = NA_real_,
                                                  n_observations_used     = length(time_per_iter),
                                                  status                  = "not_identified")
        if (length(unique(n_leapfrog_steps_per_iter)) < 2) return(time_per_iter_least_squares_fit)
        ##
        design_matrix_intercept_and_n_leapfrog_steps <-  cbind(1, n_leapfrog_steps_per_iter)
        least_squares_intercept_and_slope <-  qr.solve(design_matrix_intercept_and_n_leapfrog_steps, time_per_iter)
        time_per_iter_overhead_least_squares_intercept <-  unname(least_squares_intercept_and_slope[1])
        time_per_leapfrog_step_least_squares_slope     <-  unname(least_squares_intercept_and_slope[2])
        ##
        if (!is.finite(time_per_leapfrog_step_least_squares_slope) || time_per_leapfrog_step_least_squares_slope <= 0) {
              time_per_iter_least_squares_fit$status <-  "non_positive_slope"
              return(time_per_iter_least_squares_fit)
        }
        time_per_iter_least_squares_fit$time_per_leapfrog_step <-  time_per_leapfrog_step_least_squares_slope
        time_per_iter_least_squares_fit$time_per_iter_overhead <-  max(0, time_per_iter_overhead_least_squares_intercept)
        time_per_iter_least_squares_fit$status <-  if (time_per_iter_overhead_least_squares_intercept < 0) "fitted_intercept_set_to_zero" else "fitted"
        return(time_per_iter_least_squares_fit)

}


##
## ---- Online burn-in measurement of time_per_leapfrog_step_burnin -------------------------------------------------------------------
##
##      What is timed: the wall time (Sys.time(), microsecond resolution) of the native burn-in iteration call
##      (run_burnin_iteration()), i.e. one iteration of ALL burn-in chains in parallel, including the copy of their output to R;
##      the push of eps / tau / metric to the worker and the R adaptation are outside it. The call returns when the longest
##      trajectory ends, so the step count of the iteration is the largest over the chains: max(1, ceiling(tau_ii / eps)), with
##      tau_ii the per-chain trajectory lengths the sampler returns (row 6 of other_main_out_vector) and eps the step size the
##      iteration ran with (the native samplers' rule).
##      Which iterations: every iteration from clip_iter on in which no chain diverged (a trajectory whose log density becomes
##      non-finite stops early, so its step count is not known without the profiler).
##      The estimate: a running least-squares fit of the iteration time on the step count, updated with centred (Welford-type)
##      sums; time_per_leapfrog_step_burnin is its slope, and the intercept (the per-iteration non-leapfrog time, which includes
##      the initial gradient) is kept as a by-product. The tau ramp from clip_iter to clip_iter_tau varies the step count, so the
##      slope is identified before the handover. If, after the minimum number of iterations, the slope is not identified or not
##      positive, the fallback is the running mean of iteration time / (step count + 1), which attributes all of the call to its
##      gradient evaluations (an upper bound).
##
fn_time_per_leapfrog_step_burnin_fit_init <-  function() {

        return(list( n_iterations_used                                         = 0,
                     n_leapfrog_steps_per_iter_running_mean                    = 0,
                     time_per_iter_running_mean                                = 0,
                     sum_squared_deviations_n_leapfrog_steps                   = 0,
                     sum_cross_deviations_n_leapfrog_steps_and_time_per_iter   = 0,
                     time_per_gradient_evaluation_running_mean                 = 0))

}

fn_time_per_leapfrog_step_burnin_fit_update <-  function( time_per_leapfrog_step_burnin_fit_sums,
                                                          n_leapfrog_steps_per_iter,
                                                          time_per_iter) {

        if (!is.finite(n_leapfrog_steps_per_iter) || !is.finite(time_per_iter) || n_leapfrog_steps_per_iter < 1 || time_per_iter <= 0) {
              return(time_per_leapfrog_step_burnin_fit_sums)
        }
        time_per_leapfrog_step_burnin_fit_sums$n_iterations_used <-  time_per_leapfrog_step_burnin_fit_sums$n_iterations_used + 1
        deviation_n_leapfrog_steps_from_previous_mean <-  n_leapfrog_steps_per_iter - time_per_leapfrog_step_burnin_fit_sums$n_leapfrog_steps_per_iter_running_mean
        time_per_leapfrog_step_burnin_fit_sums$n_leapfrog_steps_per_iter_running_mean <-  time_per_leapfrog_step_burnin_fit_sums$n_leapfrog_steps_per_iter_running_mean +
                                                                                          deviation_n_leapfrog_steps_from_previous_mean / time_per_leapfrog_step_burnin_fit_sums$n_iterations_used
        time_per_leapfrog_step_burnin_fit_sums$time_per_iter_running_mean <-  time_per_leapfrog_step_burnin_fit_sums$time_per_iter_running_mean +
                                                                              (time_per_iter - time_per_leapfrog_step_burnin_fit_sums$time_per_iter_running_mean) / time_per_leapfrog_step_burnin_fit_sums$n_iterations_used
        time_per_leapfrog_step_burnin_fit_sums$sum_squared_deviations_n_leapfrog_steps <-  time_per_leapfrog_step_burnin_fit_sums$sum_squared_deviations_n_leapfrog_steps +
                                                                                           deviation_n_leapfrog_steps_from_previous_mean *
                                                                                           (n_leapfrog_steps_per_iter - time_per_leapfrog_step_burnin_fit_sums$n_leapfrog_steps_per_iter_running_mean)
        time_per_leapfrog_step_burnin_fit_sums$sum_cross_deviations_n_leapfrog_steps_and_time_per_iter <-  time_per_leapfrog_step_burnin_fit_sums$sum_cross_deviations_n_leapfrog_steps_and_time_per_iter +
                                                                                                           deviation_n_leapfrog_steps_from_previous_mean *
                                                                                                           (time_per_iter - time_per_leapfrog_step_burnin_fit_sums$time_per_iter_running_mean)
        time_per_leapfrog_step_burnin_fit_sums$time_per_gradient_evaluation_running_mean <-  time_per_leapfrog_step_burnin_fit_sums$time_per_gradient_evaluation_running_mean +
                                                                                             (time_per_iter / (n_leapfrog_steps_per_iter + 1) - time_per_leapfrog_step_burnin_fit_sums$time_per_gradient_evaluation_running_mean) /
                                                                                             time_per_leapfrog_step_burnin_fit_sums$n_iterations_used
        return(time_per_leapfrog_step_burnin_fit_sums)

}

fn_time_per_leapfrog_step_burnin_fit_estimate <-  function( time_per_leapfrog_step_burnin_fit_sums,
                                                            time_per_leapfrog_step_burnin_min_iterations) {

        time_per_leapfrog_step_burnin_estimate <-  list( time_per_leapfrog_step_burnin          = NA_real_,
                                                         time_per_iter_overhead_burnin_fitted   = NA_real_,
                                                         n_iterations_used                      = time_per_leapfrog_step_burnin_fit_sums$n_iterations_used,
                                                         status                                 = "too_few_iterations")
        if (time_per_leapfrog_step_burnin_fit_sums$n_iterations_used < time_per_leapfrog_step_burnin_min_iterations) {
              return(time_per_leapfrog_step_burnin_estimate)
        }
        ##
        least_squares_slope_is_identified <-  time_per_leapfrog_step_burnin_fit_sums$sum_squared_deviations_n_leapfrog_steps > 0
        time_per_leapfrog_step_burnin_least_squares_slope <-  if (least_squares_slope_is_identified) {
              time_per_leapfrog_step_burnin_fit_sums$sum_cross_deviations_n_leapfrog_steps_and_time_per_iter /
              time_per_leapfrog_step_burnin_fit_sums$sum_squared_deviations_n_leapfrog_steps
        } else NA_real_
        if (least_squares_slope_is_identified && is.finite(time_per_leapfrog_step_burnin_least_squares_slope) &&
            time_per_leapfrog_step_burnin_least_squares_slope > 0) {
              time_per_leapfrog_step_burnin_estimate$time_per_leapfrog_step_burnin <-  time_per_leapfrog_step_burnin_least_squares_slope
              time_per_leapfrog_step_burnin_estimate$time_per_iter_overhead_burnin_fitted <-  time_per_leapfrog_step_burnin_fit_sums$time_per_iter_running_mean -
                                                                                              time_per_leapfrog_step_burnin_least_squares_slope *
                                                                                              time_per_leapfrog_step_burnin_fit_sums$n_leapfrog_steps_per_iter_running_mean
              time_per_leapfrog_step_burnin_estimate$status <-  "fitted"
        } else {
              time_per_leapfrog_step_burnin_estimate$time_per_leapfrog_step_burnin <-  time_per_leapfrog_step_burnin_fit_sums$time_per_gradient_evaluation_running_mean
              time_per_leapfrog_step_burnin_estimate$status <-  "running_mean_time_per_gradient_evaluation_fallback"
        }
        return(time_per_leapfrog_step_burnin_estimate)

}


##
## ---- The two quantities that enter the criterion at one tau update ----------------------------------------------------------------
##
##      eps_used_for_iteration is the step size the scored trajectories ran with. n_iter_burnin is the number of main burn-in
##      iterations run at the adapted tau, n_burnin - clip_iter_tau of init_and_run_burnin_ChESSR (see the time model at the top of
##      this file). A missing time_per_leapfrog_step_burnin or time_per_leapfrog_step_sampling gives
##      burnin_to_sampling_leapfrog_time_ratio = 0.
##
fn_time_criterion_quantities_at_tau_update <-  function( time_criterion_sampling_quantities,
                                                         time_per_leapfrog_step_burnin,
                                                         eps_used_for_iteration,
                                                         n_iter_burnin) {

        tau_offset_from_sampling_overhead <-  eps_used_for_iteration * time_criterion_sampling_quantities$sampling_overhead_in_leapfrog_steps
        ##
        time_per_leapfrog_step_sampling <-  time_criterion_sampling_quantities$time_per_leapfrog_step_sampling
        n_iter_sampling_for_time_criterion <-  time_criterion_sampling_quantities$n_iter_sampling_for_time_criterion
        burnin_to_sampling_leapfrog_time_ratio <-  if (is.finite(time_per_leapfrog_step_burnin) && is.finite(time_per_leapfrog_step_sampling) &&
                                                      time_per_leapfrog_step_sampling > 0 && is.finite(n_iter_sampling_for_time_criterion) &&
                                                      n_iter_sampling_for_time_criterion > 0) {
              (n_iter_burnin * time_per_leapfrog_step_burnin) / (n_iter_sampling_for_time_criterion * time_per_leapfrog_step_sampling)
        } else 0
        ##
        return(list( tau_offset_from_sampling_overhead      = tau_offset_from_sampling_overhead,
                     burnin_to_sampling_leapfrog_time_ratio = burnin_to_sampling_leapfrog_time_ratio))

}


##
## ---- sampling_overhead_in_leapfrog_steps = "auto": the built-in default ---------------------------------------------------------------
##
##      sampling_overhead_in_leapfrog_steps describes the SAMPLING phase: n_chains_sampling chains with n_threads_WCP_sampling threads
##      per chain and the sampling chunk count. The burn-in runs another configuration (n_chains_burnin chains with n_threads_WCP_burnin
##      threads per chain and the burn-in chunk count), and within-chain parallelism and the chunk count change the per-iteration costs
##      differently in the two phases, so the burn-in's own timings are not used for it.
##
##      The calibration table (fn_sampling_overhead_in_leapfrog_steps_calibration_table) holds measured wall times of the sampling
##      phase, per sampling iteration of all n_chains_sampling chains in parallel:
##        time_per_leapfrog_step_sampling    the time per leapfrog step;
##        time_per_iter_overhead_sampling    the fixed time per iteration of the sampling loop (the initial log density and gradient,
##                                           the per-iteration bookkeeping and the trace storage);
##        time_per_iter_summaries_sampling   the summaries routine's time per draw row;
##      for one model type (Model_type), number of observations (n_observations; for LC_MVP, N individuals times n_tests tests),
##      n_chains_sampling, n_threads_WCP_sampling and sampling chunk count (num_chunks_sampling; a row applies to another sampling chunk
##      count only when its applies_to_any_num_chunks_sampling is TRUE), measured on a machine with n_physical_cores_of_calibration_machine
##      physical cores (calibration_machine_description). The times, and the ratios of times below, are those of that machine and that
##      sampling configuration: a run on a machine with another number of physical cores (where the same n_chains_sampling x
##      n_threads_WCP_sampling threads can be oversubscribed), or with another sampling chunk count, has no default. Then
##
##        sampling_overhead_in_leapfrog_steps = sampling_overhead_in_leapfrog_steps_from_iteration_overhead + sampling_overhead_in_leapfrog_steps_from_summaries
##                                            = time_per_iter_overhead_sampling / time_per_leapfrog_step_sampling
##                                              + time_per_iter_summaries_sampling / time_per_leapfrog_step_sampling.
##
##      Calibration points:
##        LC_MVP (BayesMVP), N = 10,000 individuals x 6 tests = 60,000 observations, 180 chains x 1 thread, 25 sampling chunks (the
##        sampling chunk count of pilot study 7 at N = 10,000), on the local HPC (AMD EPYC 9654, 96 physical cores):
##        time_per_leapfrog_step_sampling = 9.54 ms, time_per_iter_overhead_sampling = 16.1 ms (1.69 leapfrog steps),
##        time_per_iter_summaries_sampling = 41.6 ms (4.36 leapfrog steps), sampling_overhead_in_leapfrog_steps = 6.05.
##
##      Interpolation rule (fn_default_sampling_overhead_in_leapfrog_steps): only the rows of the same model type, n_chains_sampling,
##      n_threads_WCP_sampling, sampling chunk count and number of physical cores are used (a row with the same chunk count is preferred
##      to one with applies_to_any_num_chunks_sampling = TRUE at the same n_observations). An n_observations equal to that of a row gives
##      that row ("calibration_point"). One strictly between two rows gives each of the three times interpolated linearly in
##      log(n_observations) on the log scale between the two neighbouring rows ("log_log_interpolation_between_calibration_points", a
##      power law in n_observations between them: the leapfrog time grows with n_observations, and the fixed and summaries times need
##      not), and the two parts are the interpolated fixed and summaries times over the interpolated leapfrog time. Outside the
##      calibrated range of n_observations, or for a model type, sampling configuration, sampling chunk count or machine with no row,
##      there is no default (NULL), and "auto" runs the sampling timing probe: no extrapolation.
##
##      The default's time_per_leapfrog_step_sampling is returned for the record only. burnin_to_sampling_leapfrog_time_ratio divides
##      a burn-in time measured on the machine of the run, so it uses a user-supplied or previous-run time_per_leapfrog_step_sampling
##      and never the calibration machine's (fn_resolve_time_criterion_sampling_quantities).
##
fn_sampling_overhead_in_leapfrog_steps_calibration_table <-  function() {

        return(data.frame( model_type                                 = c("LC_MVP"),
                           n_individuals                              = c(10000),
                           n_tests                                    = c(6),
                           n_observations                             = c(60000),
                           n_chains_sampling                          = c(180),
                           n_threads_WCP_sampling                     = c(1),
                           num_chunks_sampling                        = c(25),
                           applies_to_any_num_chunks_sampling         = c(FALSE),
                           n_physical_cores_of_calibration_machine    = c(96),
                           calibration_machine_description            = c("local HPC, AMD EPYC 9654 (96 physical cores, 192 logical)"),
                           time_per_leapfrog_step_sampling            = c(0.00954),
                           time_per_iter_overhead_sampling            = c(0.0161),
                           time_per_iter_summaries_sampling           = c(0.0416),
                           calibration_description                    = c("LC_MVP (BayesMVP), N = 10,000 x 6 tests, 180 chains x 1 thread, 25 sampling chunks, local HPC (96 physical cores)"),
                           stringsAsFactors                           = FALSE))

}

##
##      The built-in default for one model type, sampling configuration and machine: a list with sampling_overhead_in_leapfrog_steps,
##      its two parts, the three (interpolated) times, the configuration it was looked up for, the calibration_method and the
##      calibration rows used; NULL when the table has no row for the configuration (sampling chunk count and number of physical cores
##      included) or n_observations lies outside the calibrated range. n_physical_cores_of_this_machine is the number of physical cores
##      of the machine of the run (parallel::detectCores(logical = FALSE); NA = unknown, which matches no row).
##
fn_default_sampling_overhead_in_leapfrog_steps <-  function( model_type,
                                                             n_observations,
                                                             n_chains_sampling,
                                                             n_threads_WCP_sampling  = 1,
                                                             num_chunks_sampling     = NA_real_,
                                                             calibration_table       = fn_sampling_overhead_in_leapfrog_steps_calibration_table(),
                                                             n_physical_cores_of_this_machine = parallel::detectCores(logical = FALSE)) {

        fn_is_single_finite_number <-  function(candidate_value) is.numeric(candidate_value) && length(candidate_value) == 1 && is.finite(candidate_value)
        if (!is.character(model_type) || length(model_type) != 1 || is.na(model_type) ||
            !fn_is_single_finite_number(n_observations) || n_observations <= 0 ||
            !fn_is_single_finite_number(n_chains_sampling) || !fn_is_single_finite_number(n_threads_WCP_sampling)) {
              return(NULL)
        }
        if (!fn_is_single_finite_number(num_chunks_sampling)) num_chunks_sampling <-  NA_real_
        ## (parallel::detectCores() gives NA where the number of cores cannot be read; an unknown machine matches no row:)
        if (!fn_is_single_finite_number(n_physical_cores_of_this_machine)) return(NULL)
        ##
        ## ---- the rows of this model type, sampling configuration and machine (a row applies to another sampling chunk count only when
        ##      its applies_to_any_num_chunks_sampling is TRUE; an unknown sampling chunk count, NA, matches only such rows):
        ##
        calibration_row_matches <-  calibration_table$model_type == model_type &
                                    calibration_table$n_chains_sampling == n_chains_sampling &
                                    calibration_table$n_threads_WCP_sampling == n_threads_WCP_sampling &
                                    calibration_table$n_physical_cores_of_calibration_machine == n_physical_cores_of_this_machine &
                                    ((calibration_table$applies_to_any_num_chunks_sampling %in% TRUE) |
                                     (!is.na(num_chunks_sampling) & calibration_table$num_chunks_sampling == num_chunks_sampling))
        calibration_row_matches[is.na(calibration_row_matches)] <-  FALSE
        matching_calibration_rows <-  calibration_table[calibration_row_matches, , drop = FALSE]
        if (nrow(matching_calibration_rows) == 0) return(NULL)
        matching_calibration_rows <-  matching_calibration_rows[order(matching_calibration_rows$n_observations,
                                                                      !(!is.na(num_chunks_sampling) & matching_calibration_rows$num_chunks_sampling %in% num_chunks_sampling)), , drop = FALSE]
        matching_calibration_rows <-  matching_calibration_rows[!duplicated(matching_calibration_rows$n_observations), , drop = FALSE]
        ##
        ## ---- no extrapolation outside the calibrated range of n_observations:
        ##
        if (n_observations < min(matching_calibration_rows$n_observations) || n_observations > max(matching_calibration_rows$n_observations)) {
              return(NULL)
        }
        ##
        calibration_time_names <-  c("time_per_leapfrog_step_sampling", "time_per_iter_overhead_sampling", "time_per_iter_summaries_sampling")
        calibration_row_at_n_observations <-  which(matching_calibration_rows$n_observations == n_observations)
        if (length(calibration_row_at_n_observations) == 1) {
              times_at_n_observations <-  unlist(matching_calibration_rows[calibration_row_at_n_observations, calibration_time_names])
              calibration_method <-  "calibration_point"
              calibration_rows_used <-  matching_calibration_rows$calibration_description[calibration_row_at_n_observations]
        } else {
              lower_calibration_row <-  max(which(matching_calibration_rows$n_observations < n_observations))
              upper_calibration_row <-  lower_calibration_row + 1
              log_n_observations_interpolation_weight <-  (log(n_observations) - log(matching_calibration_rows$n_observations[lower_calibration_row])) /
                                                          (log(matching_calibration_rows$n_observations[upper_calibration_row]) -
                                                           log(matching_calibration_rows$n_observations[lower_calibration_row]))
              times_at_n_observations <-  exp((1 - log_n_observations_interpolation_weight) * log(unlist(matching_calibration_rows[lower_calibration_row, calibration_time_names])) +
                                              log_n_observations_interpolation_weight * log(unlist(matching_calibration_rows[upper_calibration_row, calibration_time_names])))
              calibration_method <-  "log_log_interpolation_between_calibration_points"
              calibration_rows_used <-  matching_calibration_rows$calibration_description[c(lower_calibration_row, upper_calibration_row)]
        }
        ##
        time_per_leapfrog_step_sampling   <-  unname(times_at_n_observations["time_per_leapfrog_step_sampling"])
        time_per_iter_overhead_sampling   <-  unname(times_at_n_observations["time_per_iter_overhead_sampling"])
        time_per_iter_summaries_sampling  <-  unname(times_at_n_observations["time_per_iter_summaries_sampling"])
        if (!fn_is_single_finite_number(time_per_leapfrog_step_sampling) || time_per_leapfrog_step_sampling <= 0) return(NULL)
        sampling_overhead_in_leapfrog_steps_from_iteration_overhead <-  time_per_iter_overhead_sampling / time_per_leapfrog_step_sampling
        sampling_overhead_in_leapfrog_steps_from_summaries <-  time_per_iter_summaries_sampling / time_per_leapfrog_step_sampling
        ##
        return(list( sampling_overhead_in_leapfrog_steps                          = sampling_overhead_in_leapfrog_steps_from_iteration_overhead +
                                                                                    sampling_overhead_in_leapfrog_steps_from_summaries,
                     sampling_overhead_in_leapfrog_steps_from_iteration_overhead  = sampling_overhead_in_leapfrog_steps_from_iteration_overhead,
                     sampling_overhead_in_leapfrog_steps_from_summaries           = sampling_overhead_in_leapfrog_steps_from_summaries,
                     time_per_leapfrog_step_sampling                              = time_per_leapfrog_step_sampling,
                     time_per_iter_overhead_sampling                              = time_per_iter_overhead_sampling,
                     time_per_iter_summaries_sampling                             = time_per_iter_summaries_sampling,
                     model_type                                                   = model_type,
                     n_observations                                               = n_observations,
                     n_chains_sampling                                            = n_chains_sampling,
                     n_threads_WCP_sampling                                       = n_threads_WCP_sampling,
                     num_chunks_sampling                                          = num_chunks_sampling,
                     n_physical_cores_of_this_machine                             = n_physical_cores_of_this_machine,
                     calibration_method                                           = calibration_method,
                     calibration_rows_used                                        = calibration_rows_used))

}

##
##      The built-in default for a run of R_fn_sample_model (sampling_overhead_in_leapfrog_steps = "auto"): the model type, the number
##      of observations and the sampling chunk count are taken from the run's objects, and the default is looked up for the run's
##      sampling configuration and machine (fn_default_sampling_overhead_in_leapfrog_steps):
##        n_observations       = init_object$model_args_list$N * init_object$model_args_list$n_tests (NA when either is missing, as for
##                               Stan models, which then have no default);
##        num_chunks_sampling  = num_chunks_sampling, else model_args_list$num_chunks (the chunk count the sampling phase uses when
##                               num_chunks_sampling is NULL); NA for Stan models.
##      Returns a list: sampling_overhead_in_leapfrog_steps_built_in_default (NULL = no default) and the n_observations,
##      num_chunks_sampling and n_physical_cores_of_this_machine it was looked up with.
##
fn_sampling_overhead_in_leapfrog_steps_built_in_default_for_run <-  function( Model_type,
                                                                              init_object,
                                                                              model_args_list,
                                                                              num_chunks_sampling,
                                                                              n_chains_sampling,
                                                                              n_threads_WCP_sampling,
                                                                              n_physical_cores_of_this_machine = parallel::detectCores(logical = FALSE),
                                                                              calibration_table                = fn_sampling_overhead_in_leapfrog_steps_calibration_table()) {

        fn_is_single_finite_number <-  function(candidate_value) is.numeric(candidate_value) && length(candidate_value) == 1 && is.finite(candidate_value)
        N_of_run <-  init_object$model_args_list$N
        n_tests_of_run <-  init_object$model_args_list$n_tests
        n_observations_for_built_in_default <-  if (fn_is_single_finite_number(N_of_run) && fn_is_single_finite_number(n_tests_of_run))
                                                    N_of_run * n_tests_of_run else NA_real_
        num_chunks_sampling_for_built_in_default <-  if (identical(Model_type, "Stan")) NA_real_ else
                                                     if (is.null(num_chunks_sampling)) model_args_list$num_chunks else num_chunks_sampling
        if (!fn_is_single_finite_number(num_chunks_sampling_for_built_in_default)) num_chunks_sampling_for_built_in_default <-  NA_real_
        ##
        sampling_overhead_in_leapfrog_steps_built_in_default <-  fn_default_sampling_overhead_in_leapfrog_steps(
              model_type                        = Model_type,
              n_observations                    = n_observations_for_built_in_default,
              n_chains_sampling                 = n_chains_sampling,
              n_threads_WCP_sampling            = n_threads_WCP_sampling,
              num_chunks_sampling               = num_chunks_sampling_for_built_in_default,
              calibration_table                 = calibration_table,
              n_physical_cores_of_this_machine  = n_physical_cores_of_this_machine)
        ##
        return(list( sampling_overhead_in_leapfrog_steps_built_in_default  = sampling_overhead_in_leapfrog_steps_built_in_default,
                     n_observations                                        = n_observations_for_built_in_default,
                     num_chunks_sampling                                   = num_chunks_sampling_for_built_in_default,
                     n_physical_cores_of_this_machine                      = n_physical_cores_of_this_machine))

}


##
## ---- lag_one_autocorrelation_rho = "adaptive": the criterion's statistic at both ends of each chain's trajectory -------------------
##
##      The criterion's statistic is the one whose squared jump the criterion scores (fn_metric_position_criterion), of the position in
##      metric coordinates, position_in_metric_coordinates = metric_factor (theta - mean_initial):
##        CHESSR / CHESSR_time:  0.5 * |position_in_metric_coordinates|^2;
##        SNAPER / SNAPER_time:  (direction' position_in_metric_coordinates)^2 (MALT's statistic, see "MALT exponent" at the top of this file).
##      Both ends use the same centre (the running mean before the iteration), metric factor and direction:
##      criterion_statistic_at_trajectory_start at the start of the trajectory and criterion_statistic_at_accepted_end at the ACCEPTED
##      end (the chain's next state), whatever tau_weight_by_p_jump is. One value per chain (column).
##
fn_lag_one_autocorrelation_rho_statistic_per_chain <-  function( algorithm,
                                                                 theta_initial,
                                                                 theta_accepted,
                                                                 mean_initial,
                                                                 metric_factor,
                                                                 direction = NULL) {

        theta_initial <-  as.matrix(theta_initial)
        theta_accepted <-  as.matrix(theta_accepted)
        if (!identical(dim(theta_initial), dim(theta_accepted)) || length(mean_initial) != nrow(theta_initial)) {
              stop("fn_lag_one_autocorrelation_rho_statistic_per_chain: trajectory endpoint dimensions do not agree.")
        }
        initial_in_metric_coordinates <-  fn_apply_trajectory_metric(metric_factor, theta_initial - mean_initial)
        accepted_in_metric_coordinates <-  fn_apply_trajectory_metric(metric_factor, theta_accepted - mean_initial)
        if (algorithm %in% c("SNAPER", "SNAPER_time")) {
              if (length(direction) != nrow(initial_in_metric_coordinates) || any(!is.finite(direction))) stop("Invalid SNAPER direction.")
              return(list( criterion_statistic_at_trajectory_start = c(crossprod(direction, initial_in_metric_coordinates))^2,
                           criterion_statistic_at_accepted_end     = c(crossprod(direction, accepted_in_metric_coordinates))^2))
        }
        return(list( criterion_statistic_at_trajectory_start = 0.5 * colSums(initial_in_metric_coordinates^2),
                     criterion_statistic_at_accepted_end     = 0.5 * colSums(accepted_in_metric_coordinates^2)))

}

##
##      The same from the per-chain reductions of the resident joint block (fn_persistent_burnin_joint_position_reductions_resident with
##      use_proposals_R = FALSE, i.e. of the ACCEPTED end centred at the initial mean; see fn_metric_position_criterion_from_reductions).
##      NULL when the reductions were not computed (status other than 0).
##
fn_lag_one_autocorrelation_rho_statistic_from_reductions <-  function( algorithm,
                                                                       reductions_of_accepted_end) {

        if (is.null(reductions_of_accepted_end) || !identical(as.numeric(reductions_of_accepted_end$status), 0)) return(NULL)
        if (algorithm %in% c("SNAPER", "SNAPER_time")) {
              return(list( criterion_statistic_at_trajectory_start = c(reductions_of_accepted_end$projection_initial)^2,
                           criterion_statistic_at_accepted_end     = c(reductions_of_accepted_end$projection_proposed)^2))
        }
        return(list( criterion_statistic_at_trajectory_start = 0.5 * c(reductions_of_accepted_end$column_sums_initial_sq),
                     criterion_statistic_at_accepted_end     = 0.5 * c(reductions_of_accepted_end$column_sums_proposed_sq)))

}


##
## ---- lag_one_autocorrelation_rho = "adaptive": MALT's running moments and the lag_one_autocorrelation_rho they give ------------------
##
##      MALT (Algorithm 3, lines 6, 11 and 20-22; the mapping to its notation is in "MALT exponent" at the top of this file), over the valid
##      chains (no divergence; the criterion's statistic finite at both ends), at the moment update that follows n_earlier_moment_updates
##      earlier ones (one moment update per tau update); "mean over the valid chains" is written mean_over_valid_chains():
##
##        running_moment_weight                         = n_earlier_moment_updates / (n_earlier_moment_updates + lag_one_autocorrelation_rho_moment_averaging_offset)
##        mean_criterion_statistic_at_accepted_end     <- running_moment_weight * mean_criterion_statistic_at_accepted_end
##                                                         + (1 - running_moment_weight) * mean_over_valid_chains(criterion_statistic_at_accepted_end)
##        variance_criterion_statistic_at_accepted_end <- running_moment_weight * variance_criterion_statistic_at_accepted_end
##                                                         + (1 - running_moment_weight)
##                                                           * mean_over_valid_chains((criterion_statistic_at_accepted_end - mean_criterion_statistic_at_accepted_end)^2)
##        lag_one_autocovariance_criterion_statistic   <- running_moment_weight * lag_one_autocovariance_criterion_statistic
##                                                         + (1 - running_moment_weight)
##                                                           * mean_over_valid_chains((criterion_statistic_at_accepted_end - mean_criterion_statistic_at_accepted_end)
##                                                                                    * (criterion_statistic_at_trajectory_start - mean_criterion_statistic_at_accepted_end))
##        lag_one_autocorrelation_rho                   = min(1, max(0, lag_one_autocovariance_criterion_statistic / variance_criterion_statistic_at_accepted_end))
##
##      At the first moment update running_moment_weight = 0, so the moments are that update's cross-chain moments and
##      lag_one_autocorrelation_rho is the cross-chain sample lag-one autocorrelation (clipped). MALT centres its autocovariance term at the
##      mean from before the update (its line 11 precedes line 22); here both products use the updated mean_criterion_statistic_at_accepted_end,
##      so that the first update gives the cross-chain sample estimate; the two agree as n_earlier_moment_updates grows.
##      lag_one_autocorrelation_rho_moment_averaging_offset = 1 weights every update equally; MALT uses 8, which forgets the early updates
##      (MALT, appendix D, table 5).
##      An update with fewer than 2 valid chains leaves the moments unchanged.
##      The lag_one_autocorrelation_rho of the tau update after n moment updates comes from the moments up to update n, before this
##      update's trajectories are scored (MALT's line 6 precedes its line 10); before the first moment update there are none, and
##      lag_one_autocorrelation_rho = 1 (the rate criteria's exponent).
##
fn_lag_one_autocorrelation_rho_moments_init <-  function() {

        return(list( n_moment_updates                              = 0,
                     mean_criterion_statistic_at_accepted_end      = 0,
                     variance_criterion_statistic_at_accepted_end  = 0,
                     lag_one_autocovariance_criterion_statistic    = 0,
                     n_valid_chains_last_update                    = 0))

}

fn_lag_one_autocorrelation_rho_moments_update <-  function( lag_one_autocorrelation_rho_moments,
                                                            criterion_statistic_at_trajectory_start,
                                                            criterion_statistic_at_accepted_end,
                                                            chain_is_valid,
                                                            lag_one_autocorrelation_rho_moment_averaging_offset) {

        if (length(criterion_statistic_at_trajectory_start) != length(criterion_statistic_at_accepted_end) ||
            length(chain_is_valid) != length(criterion_statistic_at_accepted_end)) {
              stop(paste0("fn_lag_one_autocorrelation_rho_moments_update: criterion_statistic_at_trajectory_start, criterion_statistic_at_accepted_end ",
                          "and chain_is_valid must have one value per chain."))
        }
        chain_is_used <-  !is.na(chain_is_valid) & chain_is_valid &
                          is.finite(criterion_statistic_at_trajectory_start) & is.finite(criterion_statistic_at_accepted_end)
        lag_one_autocorrelation_rho_moments$n_valid_chains_last_update <-  sum(chain_is_used)
        if (sum(chain_is_used) < 2) return(lag_one_autocorrelation_rho_moments)
        ##
        criterion_statistic_at_trajectory_start_of_used_chains <-  criterion_statistic_at_trajectory_start[chain_is_used]
        criterion_statistic_at_accepted_end_of_used_chains     <-  criterion_statistic_at_accepted_end[chain_is_used]
        n_earlier_moment_updates <-  lag_one_autocorrelation_rho_moments$n_moment_updates
        running_moment_weight <-  n_earlier_moment_updates / (n_earlier_moment_updates + lag_one_autocorrelation_rho_moment_averaging_offset)
        ##
        mean_criterion_statistic_at_accepted_end <-  running_moment_weight * lag_one_autocorrelation_rho_moments$mean_criterion_statistic_at_accepted_end +
                                                     (1 - running_moment_weight) * mean(criterion_statistic_at_accepted_end_of_used_chains)
        variance_criterion_statistic_at_accepted_end <-  running_moment_weight * lag_one_autocorrelation_rho_moments$variance_criterion_statistic_at_accepted_end +
                                                         (1 - running_moment_weight) * mean((criterion_statistic_at_accepted_end_of_used_chains -
                                                                                             mean_criterion_statistic_at_accepted_end)^2)
        lag_one_autocovariance_criterion_statistic <-  running_moment_weight * lag_one_autocorrelation_rho_moments$lag_one_autocovariance_criterion_statistic +
                                                       (1 - running_moment_weight) * mean((criterion_statistic_at_accepted_end_of_used_chains - mean_criterion_statistic_at_accepted_end) *
                                                                                          (criterion_statistic_at_trajectory_start_of_used_chains - mean_criterion_statistic_at_accepted_end))
        ##
        lag_one_autocorrelation_rho_moments$n_moment_updates <-  n_earlier_moment_updates + 1
        lag_one_autocorrelation_rho_moments$mean_criterion_statistic_at_accepted_end <-  mean_criterion_statistic_at_accepted_end
        lag_one_autocorrelation_rho_moments$variance_criterion_statistic_at_accepted_end <-  variance_criterion_statistic_at_accepted_end
        lag_one_autocorrelation_rho_moments$lag_one_autocovariance_criterion_statistic <-  lag_one_autocovariance_criterion_statistic
        return(lag_one_autocorrelation_rho_moments)

}

fn_lag_one_autocorrelation_rho_estimate <-  function(lag_one_autocorrelation_rho_moments) {

        variance_criterion_statistic_at_accepted_end <-  lag_one_autocorrelation_rho_moments$variance_criterion_statistic_at_accepted_end
        lag_one_autocovariance_criterion_statistic   <-  lag_one_autocorrelation_rho_moments$lag_one_autocovariance_criterion_statistic
        if (lag_one_autocorrelation_rho_moments$n_moment_updates < 1 ||
            !is.finite(variance_criterion_statistic_at_accepted_end) || variance_criterion_statistic_at_accepted_end <= 0 ||
            !is.finite(lag_one_autocovariance_criterion_statistic)) {
              return(NA_real_)
        }
        return(min(1, max(0, lag_one_autocovariance_criterion_statistic / variance_criterion_statistic_at_accepted_end)))

}

##
##      The lag_one_autocorrelation_rho that a tau update uses: the number set by the setting, or ("adaptive") the estimate from the moments of
##      the earlier updates, 1 while there is none. Returns the value and where it came from.
##
fn_lag_one_autocorrelation_rho_for_tau_update <-  function( lag_one_autocorrelation_rho_setting,
                                                            lag_one_autocorrelation_rho_moments) {

        if (!identical(lag_one_autocorrelation_rho_setting, "adaptive")) {
              return(list(value = lag_one_autocorrelation_rho_setting, source = "setting"))
        }
        lag_one_autocorrelation_rho_estimate <-  fn_lag_one_autocorrelation_rho_estimate(lag_one_autocorrelation_rho_moments)
        if (!is.finite(lag_one_autocorrelation_rho_estimate)) {
              return(list(value = 1, source = "adaptive_no_estimate_yet_rho_one"))
        }
        return(list(value = lag_one_autocorrelation_rho_estimate, source = "adaptive_running_moments"))

}


##
## ---- Quantities from previous saved run(s) ---------------------------------------------------------------------------------------
##
##      Reads one or more saved runs (the lists saved by pilot study 7, or any list with the same fields) and returns
##      time_per_leapfrog_step_sampling, time_per_iter_overhead_sampling, time_per_iter_summaries_sampling,
##      sampling_overhead_in_leapfrog_steps and ESS_per_iter_sampling_expected.
##
##      Fields read, at the top level of the run or, failing that, in the place pilot study 7 stores them:
##        time_sampling           (efficiency_info$time_sampling)   wall time of the whole sampling call, all chains in parallel
##        time_summaries          (efficiency_info$time_summaries)  wall time of the summaries routine over all the draws
##        n_iter                  (efficiency_info$n_iter)          sampling iterations per chain
##        n_chains_sampling       (n_chains, HMC_info$n_chains_sampling)
##        per-parameter ESS       (n_eff of tibble_main, tibble_tp, tibble_gq; diagnostic_parameter_names; min_ESS;
##                                efficiency_info$Min_ESS_main): the minimum ESS over ALL chains of the parameters chosen by
##                                time_criterion_ess_parameter_set (see fn_time_criterion_min_ESS_of_saved_run)
##        L_main_during_sampling  (efficiency_info$L_main_during_sampling, else HMC_info$tau_main / HMC_info$eps_main)  = tau / eps,
##                                the number of leapfrog steps per sampling iteration before rounding
##        randomize_tau_sampling  (adaptation$randomize_tau_sampling; TRUE when absent, the sampler default)
##        N, n_threads_WCP_sampling, num_chunks_sampling  (top level, else settings$...; NA when absent), for the configuration check
##
##      Which ESS: the minimum ESS over the parameter set chosen by time_criterion_ess_parameter_set ("diagnostic" by default: the
##      model's diagnostic parameters, never the parameters block; see fn_time_criterion_min_ESS_of_saved_run). per_saved_run_table
##      records the set and the field it was read from (min_ESS_parameter_set, min_ESS_source). A run without that ESS is left out
##      of ESS_per_iter_sampling_expected (NA when no run has it). The diagnostic and the main minimum ESS measure different
##      quantities (in pilot study 7 they differed by a factor of about 12), so the main one never stands in for the diagnostic one.
##      Configuration check: the runs must agree on n_chains_sampling, N, n_threads_WCP_sampling and num_chunks_sampling wherever
##      they record them (the caller checks them against the current run).
##
##      Conventions: every "per iteration" quantity is per sampling iteration of ALL n_chains_sampling chains running in parallel
##      (not per chain): time per iteration = time_sampling / n_iter_sampling_of_saved_run, summaries time per iteration = time_summaries / n_iter,
##      ESS_per_iter_sampling_expected = min_ESS / n_iter (ESS summed over the chains; the per-chain value would also be divided
##      by n_chains_sampling). The mean number of leapfrog steps per sampling iteration is L_main_during_sampling + 0.5 when the sampling
##      trajectories are jittered (tau_ii ~ U(0, 2 tau), and ceiling() adds half a step on average), else ceiling(L_main_during_sampling).
##
##      One run (or runs with only one step count): the per-iteration sampling time cannot be split into a fixed part and a
##      leapfrog part, so time_per_iter_overhead_sampling is set to 0 and all of time_sampling / n_iter is attributed to the leapfrog
##      steps. sampling_overhead_in_leapfrog_steps then counts the summaries only (a lower bound). time_sampling also includes the
##      once-per-call set-up, which inflates the per-iteration time slightly.
##      Several runs with at least two different step counts (e.g. runs of other burn-in algorithms with the same sampling
##      configuration): least-squares fit of time_sampling / n_iter on the step count; intercept = time_per_iter_overhead_sampling,
##      slope = time_per_leapfrog_step_sampling. The summaries time per iteration is the mean over the runs and
##      ESS_per_iter_sampling_expected the median over the runs. All runs must share n_chains_sampling.
##
##
## ---- The minimum ESS of one saved run over the chosen parameter set -------------------------------------------------------------
##
##      time_criterion_ess_parameter_set:
##        "diagnostic"  the model's diagnostic parameters: the run's own diagnostic_parameter_names (pilot study 7 saves them: the
##                      generated quantities it checks, e.g. p[1..2], Se_baseline[1..6], Sp_baseline[1..6], Fp_baseline[1..6] for
##                      LC_MVP), else model_diagnostic_parameter_names (the generated quantities of the current model's Stan
##                      skeleton file, from its BridgeStan metadata), looked up in the run's per-parameter ESS (n_eff of
##                      tibble_main, tibble_tp and tibble_gq); else the run's min_ESS (pilot study 7: the minimum over the same
##                      diagnostic parameters); else not available. efficiency_info$Min_ESS_main (the parameters block) is never used.
##        "main"        the parameters block: n_eff of tibble_main, else efficiency_info$Min_ESS_main.
##        other         parameter names ("Se_baseline[1]") and / or base names ("Se_baseline" = every element of it), looked up in
##                      the per-parameter ESS. A name the run does not have stops with a message; elements with a non-finite
##                      n_eff (e.g. a fixed element of a Cholesky factor) are left out, with a warning. For LC_MVP,
##                      c("beta", "p_raw") (the intercepts, beta[class, 1, test] without covariates, and the prevalence
##                      parameters of the parameters block) gives the same minimum ESS as the accuracy parameters.
##
fn_time_criterion_min_ESS_of_saved_run <-  function( saved_run,
                                                     time_criterion_ess_parameter_set = "diagnostic",
                                                     model_diagnostic_parameter_names = NULL,
                                                     saved_run_label                  = "saved run") {

        fn_first_single_finite_number <-  function(candidate_value) {
              if (is.numeric(candidate_value) && length(candidate_value) == 1 && is.finite(candidate_value)) as.numeric(candidate_value) else NA_real_
        }
        fn_per_parameter_ESS_table <-  function(tibble_names) {
              per_parameter_tables <-  lapply(X = tibble_names, FUN = function(tibble_name) {
                    per_parameter_summary <-  saved_run[[tibble_name]]
                    if (!is.data.frame(per_parameter_summary) || !all(c("parameter", "n_eff") %in% names(per_parameter_summary))) return(NULL)
                    data.frame(parameter = as.character(per_parameter_summary$parameter), n_eff = as.numeric(per_parameter_summary$n_eff),
                               stringsAsFactors = FALSE)
              })
              per_parameter_tables <-  per_parameter_tables[!vapply(X = per_parameter_tables, FUN = is.null, FUN.VALUE = logical(1))]
              if (length(per_parameter_tables) == 0) return(NULL)
              return(do.call(what = rbind, args = per_parameter_tables))
        }
        fn_min_ESS_result <-  function(min_ESS_value, min_ESS_source, n_parameters) {
              list(min_ESS = min_ESS_value, min_ESS_source = min_ESS_source, n_parameters_in_min_ESS = n_parameters)
        }
        not_available_result <-  fn_min_ESS_result(NA_real_, "not_available", 0)
        efficiency_info <-  if (is.list(saved_run$efficiency_info)) saved_run$efficiency_info else list()
        ##
        if (identical(time_criterion_ess_parameter_set, "main")) {
              per_parameter_ESS_main <-  fn_per_parameter_ESS_table("tibble_main")
              if (!is.null(per_parameter_ESS_main) && nrow(per_parameter_ESS_main) > 0 && all(is.finite(per_parameter_ESS_main$n_eff))) {
                    return(fn_min_ESS_result(min(per_parameter_ESS_main$n_eff), "tibble_main$n_eff", nrow(per_parameter_ESS_main)))
              }
              if (is.finite(fn_first_single_finite_number(efficiency_info$Min_ESS_main))) {
                    return(fn_min_ESS_result(fn_first_single_finite_number(efficiency_info$Min_ESS_main), "efficiency_info$Min_ESS_main", NA_real_))
              }
              return(not_available_result)
        }
        ##
        per_parameter_ESS_all <-  fn_per_parameter_ESS_table(c("tibble_main", "tibble_tp", "tibble_gq"))
        ##
        if (identical(time_criterion_ess_parameter_set, "diagnostic")) {
              saved_run_diagnostic_parameter_names <-  saved_run$diagnostic_parameter_names
              diagnostic_parameter_names_source <-  "diagnostic_parameter_names_of_saved_run"
              if (!is.character(saved_run_diagnostic_parameter_names) || length(saved_run_diagnostic_parameter_names) == 0) {
                    saved_run_diagnostic_parameter_names <-  model_diagnostic_parameter_names
                    diagnostic_parameter_names_source <-  "generated_quantities_of_model"
              }
              if (is.character(saved_run_diagnostic_parameter_names) && length(saved_run_diagnostic_parameter_names) > 0 && !is.null(per_parameter_ESS_all)) {
                    diagnostic_rows <-  match(x = saved_run_diagnostic_parameter_names, table = per_parameter_ESS_all$parameter)
                    if (!anyNA(diagnostic_rows) && all(is.finite(per_parameter_ESS_all$n_eff[diagnostic_rows]))) {
                          return(fn_min_ESS_result(min(per_parameter_ESS_all$n_eff[diagnostic_rows]),
                                                   paste0("n_eff of ", diagnostic_parameter_names_source), length(diagnostic_rows)))
                    }
              }
              if (is.finite(fn_first_single_finite_number(saved_run$min_ESS))) {
                    return(fn_min_ESS_result(fn_first_single_finite_number(saved_run$min_ESS), "min_ESS", NA_real_))
              }
              return(not_available_result)
        }
        ##
        if (is.null(per_parameter_ESS_all)) return(not_available_result)
        per_parameter_base_names <-  sub(pattern = "\\[.*$", replacement = "", x = per_parameter_ESS_all$parameter)
        requested_name_found <-  vapply(X = time_criterion_ess_parameter_set, FUN.VALUE = logical(1), FUN = function(requested_name) {
              any(per_parameter_ESS_all$parameter == requested_name) || any(per_parameter_base_names == requested_name)
        })
        if (!all(requested_name_found)) {
              stop(paste0("fn_time_criterion_min_ESS_of_saved_run: ", saved_run_label, " has no ESS for ",
                          paste(time_criterion_ess_parameter_set[!requested_name_found], collapse = ", "),
                          " (time_criterion_ess_parameter_set)."))
        }
        requested_rows <-  per_parameter_ESS_all$parameter %in% time_criterion_ess_parameter_set | per_parameter_base_names %in% time_criterion_ess_parameter_set
        requested_rows_finite <-  requested_rows & is.finite(per_parameter_ESS_all$n_eff)
        if (any(requested_rows & !requested_rows_finite)) {
              warning(paste0("fn_time_criterion_min_ESS_of_saved_run: ", saved_run_label, ": ", sum(requested_rows & !requested_rows_finite),
                             " requested element(s) with a non-finite ESS left out (",
                             paste(utils::head(per_parameter_ESS_all$parameter[requested_rows & !requested_rows_finite], 5), collapse = ", "), ")."))
        }
        if (!any(requested_rows_finite)) return(not_available_result)
        return(fn_min_ESS_result(min(per_parameter_ESS_all$n_eff[requested_rows_finite]), "n_eff of the requested parameter names",
                                 sum(requested_rows_finite)))

}


#' Time-criterion quantities from previous saved run(s)
#'
#' @param time_criterion_previous_run_path One or more paths of saved runs (RDS files), or a list of already-read runs.
#' @param time_criterion_ess_parameter_set The parameters whose minimum ESS gives ESS_per_iter_sampling_expected: "diagnostic"
#'   (default; the model's diagnostic parameters, e.g. Se, Sp and prevalence), "main" (the parameters block) or parameter names /
#'   base names. See fn_time_criterion_min_ESS_of_saved_run.
#' @param model_diagnostic_parameter_names The generated quantities of the current model's Stan skeleton file, used for
#'   "diagnostic" when a saved run does not record its own diagnostic_parameter_names; NULL = not used.
#' @return A list with time_per_leapfrog_step_sampling, time_per_iter_overhead_sampling, time_per_iter_summaries_sampling,
#'   sampling_overhead_in_leapfrog_steps, ESS_per_iter_sampling_expected, the sampling-time estimation method used and a per-saved-run table.
#' @export
fn_time_criterion_quantities_from_saved_run <-  function( time_criterion_previous_run_path,
                                                          time_criterion_ess_parameter_set  = "diagnostic",
                                                          model_diagnostic_parameter_names  = NULL) {

        saved_runs <-  if (is.character(time_criterion_previous_run_path)) {
              lapply(X = time_criterion_previous_run_path, FUN = function(saved_run_path) {
                    if (!file.exists(saved_run_path)) stop(paste0("fn_time_criterion_quantities_from_saved_run: no file at ", saved_run_path))
                    readRDS(file = saved_run_path)
              })
        } else if (is.list(time_criterion_previous_run_path) && length(time_criterion_previous_run_path) > 0 &&
                   all(vapply(X = time_criterion_previous_run_path, FUN = is.list, FUN.VALUE = logical(1)))) {
              time_criterion_previous_run_path
        } else {
              stop("fn_time_criterion_quantities_from_saved_run: supply one or more saved-run paths, or a list of saved runs.")
        }
        saved_run_labels <-  if (is.character(time_criterion_previous_run_path)) time_criterion_previous_run_path else
                             paste0("saved_run_", seq_along(saved_runs))
        ##
        fn_first_single_number <-  function(...) {
              for (candidate_value in list(...)) {
                    if (is.numeric(candidate_value) && length(candidate_value) == 1 && is.finite(candidate_value)) return(as.numeric(candidate_value))
              }
              return(NA_real_)
        }
        ##
        per_saved_run_table <-  do.call(what = rbind, args = lapply(X = seq_along(saved_runs), FUN = function(saved_run_index) {

              saved_run <-  saved_runs[[saved_run_index]]
              efficiency_info <-  if (is.list(saved_run$efficiency_info)) saved_run$efficiency_info else list()
              HMC_info <-  if (is.list(saved_run$HMC_info)) saved_run$HMC_info else list()
              saved_run_settings <-  if (is.list(saved_run$settings)) saved_run$settings else list()
              ##
              time_sampling <-  fn_first_single_number(saved_run$time_sampling, efficiency_info$time_sampling)
              time_summaries <-  fn_first_single_number(saved_run$time_summaries, efficiency_info$time_summaries)
              n_iter_sampling_of_saved_run <-  fn_first_single_number(saved_run$n_iter, efficiency_info$n_iter, HMC_info$n_iter)
              n_chains_sampling <-  fn_first_single_number(saved_run$n_chains_sampling, saved_run$n_chains, HMC_info$n_chains_sampling)
              min_ESS_of_saved_run_result <-  fn_time_criterion_min_ESS_of_saved_run( saved_run                        = saved_run,
                                                                                      time_criterion_ess_parameter_set = time_criterion_ess_parameter_set,
                                                                                      model_diagnostic_parameter_names = model_diagnostic_parameter_names,
                                                                                      saved_run_label                  = saved_run_labels[saved_run_index])
              min_ESS_source <-  min_ESS_of_saved_run_result$min_ESS_source
              min_ESS_of_saved_run <-  min_ESS_of_saved_run_result$min_ESS
              L_main_during_sampling <-  fn_first_single_number(saved_run$L_main_during_sampling, efficiency_info$L_main_during_sampling,
                                                                if (is.numeric(HMC_info$tau_main) && is.numeric(HMC_info$eps_main))
                                                                    HMC_info$tau_main / HMC_info$eps_main else NULL)
              randomize_tau_sampling <-  if (is.list(saved_run$adaptation) && is.logical(saved_run$adaptation$randomize_tau_sampling) &&
                                             length(saved_run$adaptation$randomize_tau_sampling) == 1) {
                    isTRUE(saved_run$adaptation$randomize_tau_sampling)
              } else TRUE
              ##
              if (!is.finite(time_sampling) || !is.finite(n_iter_sampling_of_saved_run) || n_iter_sampling_of_saved_run < 1 ||
                  !is.finite(L_main_during_sampling) || L_main_during_sampling <= 0) {
                    stop(paste0("fn_time_criterion_quantities_from_saved_run: ", saved_run_labels[saved_run_index],
                                " lacks a usable time_sampling, n_iter or L_main_during_sampling."))
              }
              ##
              data.frame( saved_run                         = saved_run_labels[saved_run_index],
                          time_sampling                     = time_sampling,
                          time_summaries                    = time_summaries,
                          n_iter_sampling_of_saved_run      = n_iter_sampling_of_saved_run,
                          n_chains_sampling                 = n_chains_sampling,
                          min_ESS_of_saved_run              = min_ESS_of_saved_run,
                          min_ESS_source                    = min_ESS_source,
                          min_ESS_parameter_set             = paste(time_criterion_ess_parameter_set, collapse = ","),
                          n_parameters_in_min_ESS           = min_ESS_of_saved_run_result$n_parameters_in_min_ESS,
                          N                                 = fn_first_single_number(saved_run$N, saved_run_settings$N),
                          n_threads_WCP_sampling            = fn_first_single_number(saved_run$n_threads_WCP_sampling, saved_run_settings$n_threads_WCP_sampling),
                          num_chunks_sampling               = fn_first_single_number(saved_run$num_chunks_sampling, saved_run_settings$num_chunks_sampling),
                          L_main_during_sampling            = L_main_during_sampling,
                          randomize_tau_sampling            = randomize_tau_sampling,
                          n_leapfrog_steps_per_iter         = if (randomize_tau_sampling) L_main_during_sampling + 0.5 else ceiling(L_main_during_sampling),
                          time_per_iter_sampling            = time_sampling / n_iter_sampling_of_saved_run,
                          time_per_iter_summaries_sampling  = if (is.finite(time_summaries)) time_summaries / n_iter_sampling_of_saved_run else NA_real_,
                          ESS_per_iter_sampling             = if (is.finite(min_ESS_of_saved_run)) min_ESS_of_saved_run / n_iter_sampling_of_saved_run else NA_real_,
                          stringsAsFactors                  = FALSE)

        }))
        ##
        for (configuration_column in c("n_chains_sampling", "N", "n_threads_WCP_sampling", "num_chunks_sampling")) {
              configuration_values <-  per_saved_run_table[[configuration_column]]
              if (length(unique(configuration_values[is.finite(configuration_values)])) > 1) {
                    stop(paste0("fn_time_criterion_quantities_from_saved_run: the runs have different ", configuration_column, " (",
                                paste(unique(configuration_values[is.finite(configuration_values)]), collapse = ", "),
                                "); use runs of one N and one sampling configuration."))
              }
        }
        ##
        time_per_iter_fit_across_saved_runs <-  fn_fit_time_per_iter_on_n_leapfrog_steps( n_leapfrog_steps_per_iter = per_saved_run_table$n_leapfrog_steps_per_iter,
                                                                                          time_per_iter             = per_saved_run_table$time_per_iter_sampling)
        if (time_per_iter_fit_across_saved_runs$status %in% c("fitted", "fitted_intercept_set_to_zero")) {
              sampling_time_estimation_method <-  "least_squares_across_runs"
              time_per_leapfrog_step_sampling <-  time_per_iter_fit_across_saved_runs$time_per_leapfrog_step
              time_per_iter_overhead_sampling <-  time_per_iter_fit_across_saved_runs$time_per_iter_overhead
        } else {
              sampling_time_estimation_method <-  "single_step_count_overhead_set_to_zero"
              time_per_iter_overhead_sampling <-  0
              time_per_leapfrog_step_sampling <-  mean(per_saved_run_table$time_per_iter_sampling / per_saved_run_table$n_leapfrog_steps_per_iter)
        }
        time_per_iter_summaries_sampling <-  if (any(is.finite(per_saved_run_table$time_per_iter_summaries_sampling)))
              mean(per_saved_run_table$time_per_iter_summaries_sampling, na.rm = TRUE) else NA_real_
        ## (every run's minimum ESS is over the same parameter set, time_criterion_ess_parameter_set, so the runs that have it are pooled:)
        saved_run_is_used_for_ESS <-  is.finite(per_saved_run_table$ESS_per_iter_sampling)
        if (any(!saved_run_is_used_for_ESS)) {
              warning(paste0("fn_time_criterion_quantities_from_saved_run: ", sum(!saved_run_is_used_for_ESS), " run(s) have no minimum ESS over the ",
                             "parameter set '", paste(time_criterion_ess_parameter_set, collapse = ","), "' and are left out of ESS_per_iter_sampling_expected."))
        }
        ESS_per_iter_sampling_expected <-  if (any(saved_run_is_used_for_ESS))
              stats::median(per_saved_run_table$ESS_per_iter_sampling[saved_run_is_used_for_ESS]) else NA_real_
        ESS_per_iter_sampling_expected_source <-  if (any(saved_run_is_used_for_ESS))
              paste(unique(per_saved_run_table$min_ESS_source[saved_run_is_used_for_ESS]), collapse = ", ") else "not_available"
        sampling_overhead_in_leapfrog_steps <-  (time_per_iter_overhead_sampling +
                                                 (if (is.finite(time_per_iter_summaries_sampling)) time_per_iter_summaries_sampling else 0)) /
                                                time_per_leapfrog_step_sampling
        ##
        return(list( time_per_leapfrog_step_sampling             = time_per_leapfrog_step_sampling,
                     time_per_iter_overhead_sampling             = time_per_iter_overhead_sampling,
                     time_per_iter_summaries_sampling            = time_per_iter_summaries_sampling,
                     sampling_overhead_in_leapfrog_steps         = sampling_overhead_in_leapfrog_steps,
                     ESS_per_iter_sampling_expected              = ESS_per_iter_sampling_expected,
                     ESS_per_iter_sampling_expected_min_ESS_source = ESS_per_iter_sampling_expected_source,
                     ESS_per_iter_sampling_expected_parameter_set  = paste(time_criterion_ess_parameter_set, collapse = ","),
                     sampling_time_estimation_method             = sampling_time_estimation_method,
                     time_per_iter_fit_across_saved_runs_status  = time_per_iter_fit_across_saved_runs$status,
                     per_saved_run_table                         = per_saved_run_table))

}


##
## ---- Sampling timing probe --------------------------------------------------------------------------------------------------------
##
##      Runs before the main burn-in (after the test-order pre-burn-in, if any), at the initial state, in the sampling configuration
##      of the run: n_chains_sampling chains, n_threads_WCP_sampling threads per chain, the sampling chunk count (the nuisance layout
##      re-mapped as the sampling phase does), store_log_lik_trace, use_disk and n_nuisance_to_track as sampling will use them.
##      It calls the same native sampling entry point as the sampling phase (Rcpp_fn_RcppParallel_EHMC_sampling, or
##      Rcpp_fn_OpenMP_EHMC_sampling with parallel_method = "OpenMP"), with no adaptation (sampling adapts nothing) and a FIXED
##      number of leapfrog steps: randomize_tau = FALSE and tau = (n_leapfrog_steps_for_call - 0.5) * eps, the midpoint of the
##      interval that the sampler's rule ceiling(tau / eps) maps to n_leapfrog_steps_for_call (the fixed-number-of-leapfrog-steps
##      convention of tau_if_manual_in_L_units, placed so that rounding in tau / eps cannot give n_leapfrog_steps_for_call + 1).
##
##      Starting states: sampling chain number sampling_chain_index starts from burn-in chain number
##      ((sampling_chain_index - 1) mod n_chains_burnin) + 1, as after the burn-in. Metric:
##      the unit metric the burn-in starts from, with the nuisance centre at the mean of the starting nuisance states. eps: the
##      pre-burn-in's final eps when it is carried over, else the find_initial_eps search the burn-in itself runs.
##
##      Warm-up: before the timed calls, one untimed call of sampling_timing_probe_n_iter_warm_up_call iterations at the largest number
##      of leapfrog steps. The first native call with n_chains_sampling threads pays once for the thread-pool workers, the per-thread
##      memory arenas and first-touch page faults; without the warm-up that wall time falls on the first timed (short) call only, which
##      biases its long-minus-short difference (the first call of a pair is colder than the second).
##
##      Calls: for each n_leapfrog_steps_for_call in sampling_timing_probe_L_values, one call of
##      sampling_timing_probe_n_iter_short_call iterations and one of sampling_timing_probe_n_iter_per_L iterations. The native call
##      runs all its iterations inside C++ and does not return per-iteration times, so each call is timed as a whole (Sys.time()), and
##        time_per_iter_sampling at n_leapfrog_steps_for_call = (time_of_native_call of the long call - time_of_native_call of the short call)
##                                                              / (n_iter of the long call - n_iter of the short call),
##      which removes the per-call set-up (struct copies, worker construction, collection and conversion to R), a wall-clock time
##      paid once per sampling run and independent of tau. A difference that is not positive is not used (NA, with a warning). Then
##        time_per_iter_sampling = time_per_iter_overhead_sampling + n_leapfrog_steps_for_call * time_per_leapfrog_step_sampling
##      is fitted by least squares over the numbers of leapfrog steps with a usable difference (fn_fit_time_per_iter_on_n_leapfrog_steps);
##      with fewer than two of them the fit is not identified and the probe status is not "fitted". The intercept includes the
##      initial log density and gradient evaluation of every iteration.
##
##      Summaries: the summaries routine (create_summary_and_traces, with sampling_timing_probe_summary_arguments) has a fixed set-up
##      of about a second, and its wall time per draw row is small, so a difference over a few probe draws is lost in the run-to-run noise
##      of the set-up. The draws of the long call at the first number of leapfrog steps are therefore tiled (repeated along the
##      iterations; the routine's wall time does not depend on the draw values) to sampling_timing_probe_summaries_n_iter_tiled_short_call
##      and sampling_timing_probe_summaries_n_iter_tiled_long_call rows. The routine is first called twice, untimed, on the short tiled
##      draws (the first call loads packages; R's byte-code compiler compiles a closure at its second call), and then timed once on
##      each tiled set:
##        time_per_iter_summaries_sampling = (time_summaries_on_draws of the long tiled draws - time_summaries_on_draws of the short tiled draws)
##                                           / (sampling_timing_probe_summaries_n_iter_tiled_long_call - sampling_timing_probe_summaries_n_iter_tiled_short_call),
##      the wall-clock time per draw row over all chains, without the routine's fixed set-up. If that difference is not positive, or
##      the routine fails, time_per_iter_summaries_sampling is NA with a warning; an
##      upper bound such as the long call's time / its n_iter, which includes the whole set-up, is never used. With use_disk = TRUE the
##      routine reads the traces from the files the sampling call wrote, which cannot be tiled, so the summaries are not timed (NA,
##      with a warning); time_per_iter_summaries_sampling can then be supplied or taken from a previous run. The routine writes its
##      post-hoc files to use_disk_path_post_hoc_dir = paste0(use_disk_path, "_timing_probe_post_hoc") unless
##      sampling_timing_probe_summary_arguments sets it.
##
##      Lite mode (sampling_timing_probe_mode = "lite"; the default with sampling_overhead_in_leapfrog_steps = "auto"): the same
##      sampling configuration, starting states, step size, native entry point and untimed warm-up call, but the short and the long
##      call are made only at the smallest and the largest value of sampling_timing_probe_L_values (two numbers of leapfrog steps,
##      which identify the fit; with the default c(2, 8) these are the full probe's native calls), and the summaries routine is
##      timed ONCE, on the long tiled draws:
##        time_per_iter_summaries_sampling = (time_summaries_on_draws of the long tiled draws - time_of_summaries_fixed_set_up)
##                                           / sampling_timing_probe_summaries_n_iter_tiled_long_call,
##      with time_of_summaries_fixed_set_up measured once per R session and configuration and cached
##      (fn_time_summaries_per_draw_row_lite_with_cached_set_up, fn_sampling_timing_probe_summaries_set_up_cache). The first "lite"
##      probe of a configuration in an R session measures the set-up with the four summaries calls above, so its summaries timing
##      costs as much as the full probe's; every later one makes one summaries call.
##      A "lite" probe therefore makes fewer calls than the full probe only when sampling_timing_probe_L_values has more than two
##      values (fewer native calls) or the set-up is already cached (one summaries call instead of four). With the default c(2, 8) and
##      no cached set-up, which is always the case with run_in_fresh_R_process = TRUE, it makes exactly the full probe's calls; its
##      start message says which case applies.
##
##      The probe draws are discarded. The probe leaves the sampler state as it was: every list it changes is a local copy (R copies
##      on modification), the native samplers seed their own random number streams from seed_R at every call, R's .Random.seed is
##      saved and restored, and the RcppParallel thread count is set back to the burn-in's. With use_disk the trace files are
##      overwritten by the sampling phase. The probe's wall time is returned (sampling_timing_probe_wall_time), and no further
##      call starts once it exceeds sampling_timing_probe_max_wall_time.
##
##      Calls to native entry points and to the summaries routine are resolved in the environment the caller binds this function to
##      (R_fn_sample_model binds it to its own enclosing environment, so a model package's backend supplies its own natives).
##
fn_run_sampling_timing_probe <-  function( time_criterion_settings,
                                           ##
                                           Model_type,
                                           init_object,
                                           Model_args_as_Rcpp_List,
                                           model_args_list,
                                           Stan_data_list,
                                           stan_chunk_size_sampling,
                                           y,
                                           ##
                                           theta_main_vectors_all_chains_input_from_R,
                                           theta_nuisance_vectors_all_chains_input_from_R,
                                           n_chains_burnin,
                                           n_params_main,
                                           n_nuisance,
                                           ##
                                           n_chains_sampling,
                                           n_threads_WCP_sampling,
                                           n_threads_WCP_burnin,
                                           num_chunks_sampling,
                                           n_threads_to_restore_after_probe,
                                           parallel_method,
                                           use_disk,
                                           use_disk_path,
                                           store_log_lik_trace,
                                           n_nuisance_to_track,
                                           ##
                                           sample_nuisance,
                                           partitioned_HMC,
                                           diffusion_HMC,
                                           diffusion_HMC_integrator,
                                           metric_shape_main,
                                           force_autodiff,
                                           force_PartialLog,
                                           multi_attempts,
                                           seed,
                                           eps_carry_over,
                                           ##
                                           ## model_results_template_for_summaries) {
                                           model_results_template_for_summaries,
                                           ##
                                           ## ---- "full" (default; the probe described above) or "lite" (see "Lite mode" above):
                                           sampling_timing_probe_mode = "full") {

        sampling_timing_probe_start_time <-  Sys.time()
        fn_wall_time_in_seconds_since <-  function(start_time) as.numeric(difftime(Sys.time(), start_time, units = "secs"))
        ##
        sampling_timing_probe_L_values <-  time_criterion_settings$sampling_timing_probe_L_values
        sampling_timing_probe_n_iter_per_L <-  time_criterion_settings$sampling_timing_probe_n_iter_per_L
        sampling_timing_probe_n_iter_short_call <-  time_criterion_settings$sampling_timing_probe_n_iter_short_call
        sampling_timing_probe_max_wall_time <-  time_criterion_settings$sampling_timing_probe_max_wall_time
        sampling_timing_probe_n_iter_warm_up_call <-  time_criterion_settings$sampling_timing_probe_n_iter_warm_up_call
        ##
        ## ---- sampling_timing_probe_mode = "lite" (see "Lite mode" above): the timed calls at the smallest and the largest number of
        ##      leapfrog steps only:
        ##
        if (!(is.character(sampling_timing_probe_mode) && length(sampling_timing_probe_mode) == 1 && sampling_timing_probe_mode %in% c("full", "lite"))) {
              stop("sampling timing probe: sampling_timing_probe_mode must be \"full\" or \"lite\".")
        }
        sampling_timing_probe_is_lite <-  identical(sampling_timing_probe_mode, "lite")
        n_distinct_sampling_timing_probe_L_values_of_full_probe <-  length(unique(sampling_timing_probe_L_values))
        if (sampling_timing_probe_is_lite) sampling_timing_probe_L_values <-  range(sampling_timing_probe_L_values)
        ##
        ## ---- leave R's random number stream and the thread count as they were:
        ##
        random_seed_existed_before_probe <-  exists(x = ".Random.seed", envir = globalenv(), inherits = FALSE)
        if (random_seed_existed_before_probe) random_seed_before_probe <-  get(x = ".Random.seed", envir = globalenv(), inherits = FALSE)
        on.exit({
              if (random_seed_existed_before_probe) {
                    assign(x = ".Random.seed", value = random_seed_before_probe, envir = globalenv())
              } else if (exists(x = ".Random.seed", envir = globalenv(), inherits = FALSE)) {
                    rm(list = ".Random.seed", envir = globalenv())
              }
              RcppParallel::setThreadOptions(numThreads = n_threads_to_restore_after_probe)
        }, add = TRUE)
        ##
        message(colourise(paste0("sampling timing probe: ", n_chains_sampling, " chains x ", n_threads_WCP_sampling,
                                 " thread(s), fixed leapfrog steps ", paste(sampling_timing_probe_L_values, collapse = " / "),
                                 ", calls of ", sampling_timing_probe_n_iter_short_call, " and ", sampling_timing_probe_n_iter_per_L,
                                 " iterations at each, after an untimed ", sampling_timing_probe_n_iter_warm_up_call, "-iteration warm-up call at ",
                                 max(sampling_timing_probe_L_values)), "cyan"))
        ## (sampling_timing_probe_mode = "lite": the cache key of the summaries' fixed set-up for this configuration and data, and whether
        ##  the set-up is already cached in this R session; see "Lite mode" above:)
        sampling_timing_probe_summaries_set_up_cache_key_for_probe <-  NULL
        if (sampling_timing_probe_is_lite) {
              sampling_timing_probe_summaries_set_up_cache_key_for_probe <-  fn_sampling_timing_probe_summaries_set_up_cache_key(
                    Model_type                         = Model_type,
                    model_so_file                      = Model_args_as_Rcpp_List$model_so_file,
                    n_params_main                      = n_params_main,
                    n_nuisance                         = n_nuisance,
                    n_nuisance_to_track                = n_nuisance_to_track,
                    n_chains_sampling                  = n_chains_sampling,
                    store_log_lik_trace                = store_log_lik_trace,
                    summaries_arguments_from_settings  = time_criterion_settings$sampling_timing_probe_summary_arguments,
                    data_for_cache_key                 = if (Model_type == "Stan") Stan_data_list else list(y = y))
              summaries_set_up_is_cached_at_probe_start <-  exists(x        = sampling_timing_probe_summaries_set_up_cache_key_for_probe,
                                                                   envir    = fn_sampling_timing_probe_summaries_set_up_cache(),
                                                                   inherits = FALSE)
              lite_probe_makes_fewer_native_calls <-  n_distinct_sampling_timing_probe_L_values_of_full_probe > 2
              lite_probe_times_summaries <-  isTRUE(time_criterion_settings$sampling_timing_probe_time_summaries) && !isTRUE(use_disk)
              lite_probe_makes_fewer_summaries_calls <-  lite_probe_times_summaries && summaries_set_up_is_cached_at_probe_start
              message(colourise(paste0("sampling timing probe: \"lite\" mode (the smallest and the largest number of leapfrog steps only; the summaries ",
                                       "timed once, less their fixed set-up, which is measured once per R session and cached)"), "cyan"))
              message(colourise(paste0("sampling timing probe (lite): native calls ",
                                       if (lite_probe_makes_fewer_native_calls) paste0("at 2 of the ", n_distinct_sampling_timing_probe_L_values_of_full_probe,
                                                                                       " numbers of leapfrog steps of the full probe") else
                                                                                "the same as the full probe's (sampling_timing_probe_L_values has at most 2 values)",
                                       if (!lite_probe_times_summaries) "; summaries not timed (use_disk = TRUE or sampling_timing_probe_time_summaries = FALSE), as in the full probe" else
                                       if (summaries_set_up_is_cached_at_probe_start) "; summaries set-up cached, so one summaries call" else
                                           "; summaries set-up not yet cached in this R session, so the full probe's four summaries calls",
                                       if (lite_probe_makes_fewer_native_calls || lite_probe_makes_fewer_summaries_calls) "" else
                                           "; this probe costs as much as the full probe"), "cyan"))
        }
        ##
        ## ---- y for the native calls: the built-in models read it; for Stan models it is a placeholder (the data come from the
        ##      JSON file), made here without drawing random numbers:
        ##
        y_data_for_probe <-  if (Model_type == "Stan") matrix(data = rep(1, 1000), ncol = 1) else y
        ##
        ## ---- Model arguments as the sampling phase will have them (the edits init_and_run_burnin_ChESSR and the sampling block of
        ##      R_fn_sample_model make):
        ##
        Model_args_for_probe <-  Model_args_as_Rcpp_List
        Model_args_for_probe$n_nuisance <-  n_nuisance
        Model_args_for_probe$n_params_main <-  n_params_main
        if (Model_type != "Stan") Model_args_for_probe$Model_args_bools[15, 1] <-  FALSE
        ##
        ## ---- step size: carried over from the pre-burn-in, else the burn-in's own search (burn-in model arguments and layout):
        ##
        EHMC_Metric_for_probe <-  init_EHMC_Metric_as_Rcpp_List( n_params_main     = n_params_main,
                                                                 n_nuisance        = n_nuisance,
                                                                 metric_shape_main = metric_shape_main)
        EHMC_args_for_probe <-  init_EHMC_args_as_Rcpp_List( diffusion_HMC            = diffusion_HMC,
                                                             diffusion_HMC_integrator = diffusion_HMC_integrator)
        EHMC_args_for_probe$eps_main <-  0.00001
        EHMC_args_for_probe$eps_us   <-  0.00001
        ##
        if (!is.null(eps_carry_over)) {
              eps_main_for_probe <-  eps_carry_over$eps_main
              eps_us_for_probe   <-  if (isTRUE(partitioned_HMC)) eps_carry_over$eps_us else eps_main_for_probe
              eps_source_for_probe <-  "carried_over_from_pre_burnin"
        } else {
              theta_mean_main_for_eps_search <-  rowMeans(theta_main_vectors_all_chains_input_from_R)
              theta_mean_us_for_eps_search <-  if (isTRUE(sample_nuisance) && n_nuisance > 0) rowMeans(theta_nuisance_vectors_all_chains_input_from_R) else
                                               rep(1, n_nuisance)
              eps_search_arguments <-  list( theta_main_vec_initial_ref = matrix(c(theta_mean_main_for_eps_search), ncol = 1),
                                             theta_us_vec_initial_ref   = matrix(c(theta_mean_us_for_eps_search), ncol = 1),
                                             partitioned_HMC            = partitioned_HMC,
                                             seed                       = seed,
                                             Model_type                 = Model_type,
                                             force_autodiff             = force_autodiff,
                                             force_PartialLog           = force_PartialLog,
                                             multi_attempts             = multi_attempts,
                                             y_ref                      = y_data_for_probe,
                                             Model_args_as_Rcpp_List    = Model_args_for_probe,
                                             EHMC_args_as_Rcpp_List     = EHMC_args_for_probe,
                                             EHMC_Metric_as_Rcpp_List   = EHMC_Metric_for_probe)
              if (isTRUE(tryCatch("n_threads" %in% names(formals(fn_find_initial_eps_main_and_us)), error = function(error_object) FALSE))) {
                    eps_search_arguments$n_threads <-  max(1, floor(n_threads_WCP_burnin))
              }
              eps_search_result <-  do.call(what = fn_find_initial_eps_main_and_us, args = eps_search_arguments)
              eps_main_for_probe <-  min(0.50, eps_search_result[[1]])
              eps_us_for_probe   <-  if (isTRUE(partitioned_HMC)) min(0.50, eps_search_result[[2]]) else eps_main_for_probe
              eps_source_for_probe <-  "find_initial_eps_search"
        }
        if (!is.finite(eps_main_for_probe) || eps_main_for_probe <= 0 || !is.finite(eps_us_for_probe) || eps_us_for_probe <= 0) {
              stop("sampling timing probe: no usable step size.")
        }
        EHMC_args_for_probe$eps_main <-  eps_main_for_probe
        EHMC_args_for_probe$eps_us   <-  eps_us_for_probe
        EHMC_args_for_probe$randomize_tau <-  FALSE
        EHMC_args_for_probe$share_tau_ii_across_chains <-  FALSE
        EHMC_args_for_probe$use_given_tau_main_ii <-  FALSE
        EHMC_args_for_probe$store_log_lik_trace <-  store_log_lik_trace
        EHMC_args_for_probe$record_kinetic_energy_tau_derivatives <-  FALSE
        ##
        ## ---- sampling chunk count and nuisance layout (the sampling block of R_fn_sample_model):
        ##
        theta_nuisance_for_probe <-  theta_nuisance_vectors_all_chains_input_from_R
        if (Model_type != "Stan") {
              num_chunks_used_in_burnin <-  Model_args_for_probe$Model_args_ints[4]
              Model_args_for_probe$Model_args_ints[4] <-  if (is.null(num_chunks_sampling)) model_args_list$num_chunks else num_chunks_sampling
              num_chunks_used_in_sampling <-  Model_args_for_probe$Model_args_ints[4]
              if ((n_nuisance > 0) && (num_chunks_used_in_sampling != num_chunks_used_in_burnin)) {
                    N_for_remap       <-  init_object$model_args_list$N
                    n_tests_for_remap <-  init_object$model_args_list$n_tests
                    if (is.null(N_for_remap) || is.null(n_tests_for_remap) || (N_for_remap * n_tests_for_remap != n_nuisance)) {
                          stop("sampling timing probe: cannot re-lay-out the nuisance vector for the sampling chunk count.")
                    }
                    nuisance_chunk_remap_index <-  fn_nuisance_chunk_remap_index( N             = N_for_remap,
                                                                                  n_tests       = n_tests_for_remap,
                                                                                  n_chunks_from = num_chunks_used_in_burnin,
                                                                                  n_chunks_to   = num_chunks_used_in_sampling,
                                                                                  vect_type     = Model_args_for_probe$Model_args_strings[1])
                    theta_nuisance_for_probe <-  theta_nuisance_for_probe[nuisance_chunk_remap_index, , drop = FALSE]
              }
              Model_args_for_probe$model_so_file <-  "none"
              Model_args_for_probe$json_file_path <-  "none"
        } else if (!is.null(stan_chunk_size_sampling)) {
              Stan_data_for_probe <-  Stan_data_list
              Stan_data_for_probe$chunk_size <-  stan_chunk_size_sampling
              Model_args_for_probe$json_file_path <-  convert_stan_data_list_to_JSON( stan_data_list = Stan_data_for_probe,
                                                                                      pkg_data_dir   = dirname(init_object$json_file_path))
        }
        ##
        ## ---- starting states of the sampling chains (sampling chain number sampling_chain_index <- burn-in chain number
        ##      ((sampling_chain_index - 1) mod n_chains_burnin) + 1):
        ##
        sampling_chain_to_burnin_chain_index <-  rep_len(seq_len(n_chains_burnin), n_chains_sampling)
        theta_main_for_probe <-  theta_main_vectors_all_chains_input_from_R[, sampling_chain_to_burnin_chain_index, drop = FALSE]
        theta_nuisance_for_probe <-  theta_nuisance_for_probe[, sampling_chain_to_burnin_chain_index, drop = FALSE]
        if (n_nuisance > 0 && nrow(theta_nuisance_for_probe) == n_nuisance) {
              EHMC_Metric_for_probe$theta_hat_us_vec <-  matrix(rowMeans(theta_nuisance_for_probe), ncol = 1)
        }
        ##
        RcppParallel::setThreadOptions(numThreads = n_chains_sampling * n_threads_WCP_sampling)
        ##
        fn_call_sampling_entry_point <-  function(n_iter_for_call, EHMC_args_for_call) {
              if (parallel_method == "OpenMP") {
                    return(Rcpp_fn_OpenMP_EHMC_sampling( n_threads_R                                 = n_chains_sampling,
                                                         sample_nuisance_R                           = sample_nuisance,
                                                         n_nuisance_to_track                         = n_nuisance_to_track,
                                                         seed_R                                      = seed,
                                                         iter_one_by_one                             = FALSE,
                                                         n_iter_R                                    = n_iter_for_call,
                                                         partitioned_HMC_R                           = partitioned_HMC,
                                                         diffusion_HMC_R                             = diffusion_HMC,
                                                         Model_type_R                                = Model_type,
                                                         force_autodiff_R                            = force_autodiff,
                                                         force_PartialLog                            = force_PartialLog,
                                                         multi_attempts_R                            = multi_attempts,
                                                         theta_main_vectors_all_chains_input_from_R  = theta_main_for_probe,
                                                         theta_us_vectors_all_chains_input_from_R    = theta_nuisance_for_probe,
                                                         y                                           = y_data_for_probe,
                                                         Model_args_as_Rcpp_List                     = Model_args_for_probe,
                                                         EHMC_args_as_Rcpp_List                      = EHMC_args_for_call,
                                                         EHMC_Metric_as_Rcpp_List                    = EHMC_Metric_for_probe,
                                                         n_threads_WCP                               = n_threads_WCP_sampling))
              }
              return(Rcpp_fn_RcppParallel_EHMC_sampling( n_threads_R                                 = n_chains_sampling,
                                                         sample_nuisance_R                           = sample_nuisance,
                                                         n_nuisance_to_track                         = n_nuisance_to_track,
                                                         seed_R                                      = seed,
                                                         iter_one_by_one                             = FALSE,
                                                         n_iter_R                                    = n_iter_for_call,
                                                         partitioned_HMC_R                           = partitioned_HMC,
                                                         diffusion_HMC_R                             = diffusion_HMC,
                                                         Model_type_R                                = Model_type,
                                                         force_autodiff_R                            = force_autodiff,
                                                         force_PartialLog                            = force_PartialLog,
                                                         multi_attempts_R                            = multi_attempts,
                                                         theta_main_vectors_all_chains_input_from_R  = theta_main_for_probe,
                                                         theta_us_vectors_all_chains_input_from_R    = theta_nuisance_for_probe,
                                                         y                                           = y_data_for_probe,
                                                         Model_args_as_Rcpp_List                     = Model_args_for_probe,
                                                         EHMC_args_as_Rcpp_List                      = EHMC_args_for_call,
                                                         EHMC_Metric_as_Rcpp_List                    = EHMC_Metric_for_probe,
                                                         use_disk                                    = use_disk,
                                                         trace_dir                                   = use_disk_path,
                                                         n_threads_WCP                               = n_threads_WCP_sampling))
        }
        ##
        ## ---- the draws of one call tiled along the iterations to n_iter_target rows (the per-chain matrices and vectors of the
        ##      main, divergence, nuisance and log-likelihood traces; every other element is left as it is):
        ##
        fn_tile_sampling_object_draws <-  function(sampling_object, n_iter_source, n_iter_target) {
              iteration_index_after_tiling <-  rep_len(seq_len(n_iter_source), n_iter_target)
              fn_tile_one_chain_trace <-  function(chain_trace) {
                    if (is.matrix(chain_trace) && ncol(chain_trace) == n_iter_source) return(chain_trace[, iteration_index_after_tiling, drop = FALSE])
                    if (is.numeric(chain_trace) && is.null(dim(chain_trace)) && length(chain_trace) == n_iter_source) return(chain_trace[iteration_index_after_tiling])
                    return(chain_trace)
              }
              for (trace_element_index in intersect(c(1, 2, 3, 6), seq_along(sampling_object))) {
                    if (is.list(sampling_object[[trace_element_index]])) {
                          sampling_object[[trace_element_index]] <-  lapply(X = sampling_object[[trace_element_index]], FUN = fn_tile_one_chain_trace)
                    }
              }
              return(sampling_object)
        }
        ##
        ## ---- the summaries routine on one set of draws; returns its wall time, or NA if it failed:
        ##
        summaries_arguments_from_settings <-  time_criterion_settings$sampling_timing_probe_summary_arguments
        if (is.null(summaries_arguments_from_settings$use_disk_path_post_hoc_dir) && is.character(use_disk_path) && length(use_disk_path) == 1) {
              summaries_arguments_from_settings$use_disk_path_post_hoc_dir <-  paste0(use_disk_path, "_timing_probe_post_hoc")
        }
        fn_time_summaries_on_probe_draws <-  function(sampling_object, EHMC_args_for_call, time_of_native_call, n_leapfrog_steps_for_call) {
              probe_model_results <-  model_results_template_for_summaries
              probe_model_results$sampling_object <-  sampling_object
              probe_model_results$burnin_object <-  list( EHMC_args_as_Rcpp_List = EHMC_args_for_call,
                                                          time_burnin            = 0,
                                                          n_chains_burnin        = n_chains_burnin,
                                                          L_main_during_burnin   = n_leapfrog_steps_for_call - 0.5,
                                                          L_us_during_burnin     = n_leapfrog_steps_for_call - 0.5)
              probe_model_results$time_burnin <-  0
              probe_model_results$time_sampling <-  time_of_native_call
              summaries_arguments <-  c(list( model_results = probe_model_results,
                                              use_disk      = use_disk,
                                              use_disk_path = use_disk_path),
                                        summaries_arguments_from_settings)
              ##
              ## a guard timer on the tictoc stack: the routine starts its own timer (tictoc::tic()) and stops it near its end, so after the
              ## call, whether it succeeded or failed before or after its own toc(), the timers are stopped down to and including the guard
              ## and no timer of a caller is touched. tictoc records proc.time() elapsed seconds (millisecond resolution), so the clock is
              ## let move past the guard's start before the call: every timer the routine starts then has a later start than the guard:
              time_of_tictoc_guard_start <-  tictoc::tic("sampling_timing_probe_summaries_tictoc_guard")
              while (proc.time()[["elapsed"]] <= time_of_tictoc_guard_start) Sys.sleep(0.0005)
              summaries_start_time <-  Sys.time()
              summaries_completed <-  FALSE
              tryCatch({
                    ## (invisible(): capture.output() would otherwise print the routine's whole returned object, draws included, into its text
                    ##  connection, which takes far longer than the routine itself)
                    suppressMessages(utils::capture.output(invisible(do.call(what = create_summary_and_traces, args = summaries_arguments)), type = "output"))
                    summaries_completed <-  TRUE
              }, error = function(error_object) {
                    message(colourise(paste0("sampling timing probe: the summaries routine failed on the probe draws: ",
                                             conditionMessage(error_object)), "red"))
              })
              time_summaries_on_draws <-  if (summaries_completed) fn_wall_time_in_seconds_since(summaries_start_time) else NA_real_
              repeat {
                    stopped_timer <-  tryCatch(tictoc::toc(quiet = TRUE), error = function(error_object) NULL)
                    if (is.null(stopped_timer) || unname(stopped_timer$tic) <= time_of_tictoc_guard_start) break
              }
              return(time_summaries_on_draws)
        }
        ##
        fn_is_over_wall_time_limit <-  function() {
              !is.null(sampling_timing_probe_max_wall_time) &&
              fn_wall_time_in_seconds_since(sampling_timing_probe_start_time) > sampling_timing_probe_max_wall_time
        }
        ##
        ## ---- untimed warm-up call at the largest number of leapfrog steps (see "Warm-up" above):
        ##
        n_leapfrog_steps_for_warm_up_call <-  max(sampling_timing_probe_L_values)
        EHMC_args_for_warm_up_call <-  EHMC_args_for_probe
        EHMC_args_for_warm_up_call$tau_main <-  (n_leapfrog_steps_for_warm_up_call - 0.5) * eps_main_for_probe
        EHMC_args_for_warm_up_call$tau_main_ii <-  EHMC_args_for_warm_up_call$tau_main
        EHMC_args_for_warm_up_call$tau_us <-  if (isTRUE(partitioned_HMC)) (n_leapfrog_steps_for_warm_up_call - 0.5) * eps_us_for_probe else EHMC_args_for_warm_up_call$tau_main
        EHMC_args_for_warm_up_call$tau_us_ii <-  EHMC_args_for_warm_up_call$tau_us
        warm_up_call_start_time <-  Sys.time()
        warm_up_sampling_object <-  fn_call_sampling_entry_point(n_iter_for_call = sampling_timing_probe_n_iter_warm_up_call, EHMC_args_for_call = EHMC_args_for_warm_up_call)
        time_of_warm_up_native_call <-  fn_wall_time_in_seconds_since(warm_up_call_start_time)
        rm(warm_up_sampling_object)
        ##
        ## ---- the timed calls:
        ##
        sampling_timing_probe_calls_table <-  data.frame( n_leapfrog_steps_per_iter = numeric(0),
                                                          n_iter                    = numeric(0),
                                                          time_of_native_call       = numeric(0),
                                                          n_divergent_trajectories  = numeric(0))
        stopped_at_wall_time_limit <-  FALSE
        sampling_object_for_summaries <-  NULL
        for (sampling_timing_probe_L_value_index in seq_along(sampling_timing_probe_L_values)) {
              n_leapfrog_steps_for_call <-  sampling_timing_probe_L_values[sampling_timing_probe_L_value_index]
              EHMC_args_for_call <-  EHMC_args_for_probe
              EHMC_args_for_call$tau_main <-  (n_leapfrog_steps_for_call - 0.5) * eps_main_for_probe
              EHMC_args_for_call$tau_main_ii <-  EHMC_args_for_call$tau_main
              EHMC_args_for_call$tau_us <-  if (isTRUE(partitioned_HMC)) (n_leapfrog_steps_for_call - 0.5) * eps_us_for_probe else EHMC_args_for_call$tau_main
              EHMC_args_for_call$tau_us_ii <-  EHMC_args_for_call$tau_us
              for (n_iter_for_call in c(sampling_timing_probe_n_iter_short_call, sampling_timing_probe_n_iter_per_L)) {
                    if (fn_is_over_wall_time_limit()) {
                          stopped_at_wall_time_limit <-  TRUE
                          break
                    }
                    call_start_time <-  Sys.time()
                    sampling_object_of_call <-  fn_call_sampling_entry_point(n_iter_for_call = n_iter_for_call, EHMC_args_for_call = EHMC_args_for_call)
                    time_of_native_call <-  fn_wall_time_in_seconds_since(call_start_time)
                    n_divergent_trajectories <-  sum(vapply(X = sampling_object_of_call[[2]], FUN = function(divergence_trace) sum(divergence_trace != 0),
                                                            FUN.VALUE = numeric(1)))
                    ## the long call's draws at the first number of leapfrog steps are kept for the summaries timing:
                    if (sampling_timing_probe_L_value_index == 1 && n_iter_for_call == sampling_timing_probe_n_iter_per_L) {
                          sampling_object_for_summaries <-  list( sampling_object           = sampling_object_of_call,
                                                                  EHMC_args_for_call        = EHMC_args_for_call,
                                                                  time_of_native_call       = time_of_native_call,
                                                                  n_leapfrog_steps_for_call = n_leapfrog_steps_for_call)
                    }
                    rm(sampling_object_of_call)
                    sampling_timing_probe_calls_table <-  rbind(sampling_timing_probe_calls_table,
                                                                data.frame( n_leapfrog_steps_per_iter = n_leapfrog_steps_for_call,
                                                                            n_iter                    = n_iter_for_call,
                                                                            time_of_native_call       = time_of_native_call,
                                                                            n_divergent_trajectories  = n_divergent_trajectories))
              }
              if (stopped_at_wall_time_limit) break
        }
        ##
        ## ---- time_per_iter_sampling at each number of leapfrog steps (NA when the long-minus-short difference is not positive), then
        ##      the least-squares fit over the numbers of leapfrog steps with a usable difference:
        ##
        time_per_iter_sampling_by_step_count_table <-  do.call(what = rbind, args = lapply(X   = unique(sampling_timing_probe_calls_table$n_leapfrog_steps_per_iter),
                                                                                          FUN = function(n_leapfrog_steps_for_call) {
              probe_calls_at_this_n_leapfrog_steps <-  sampling_timing_probe_calls_table[sampling_timing_probe_calls_table$n_leapfrog_steps_per_iter ==
                                                                                         n_leapfrog_steps_for_call, , drop = FALSE]
              short_probe_call_row <-  probe_calls_at_this_n_leapfrog_steps[probe_calls_at_this_n_leapfrog_steps$n_iter ==
                                                                            sampling_timing_probe_n_iter_short_call, , drop = FALSE]
              long_probe_call_row  <-  probe_calls_at_this_n_leapfrog_steps[probe_calls_at_this_n_leapfrog_steps$n_iter ==
                                                                            sampling_timing_probe_n_iter_per_L, , drop = FALSE]
              if (nrow(short_probe_call_row) != 1 || nrow(long_probe_call_row) != 1) return(NULL)
              time_per_iter_long_minus_short_call <-  (long_probe_call_row$time_of_native_call - short_probe_call_row$time_of_native_call) /
                                                      (long_probe_call_row$n_iter - short_probe_call_row$n_iter)
              long_minus_short_call_is_usable <-  is.finite(time_per_iter_long_minus_short_call) && time_per_iter_long_minus_short_call > 0
              data.frame( n_leapfrog_steps_per_iter      = n_leapfrog_steps_for_call,
                          time_per_iter_sampling         = if (long_minus_short_call_is_usable) time_per_iter_long_minus_short_call else NA_real_,
                          time_per_iter_sampling_method  = if (long_minus_short_call_is_usable) "long_minus_short_call" else
                                                               "long_minus_short_call_not_positive_not_used",
                          stringsAsFactors               = FALSE)
        }))
        if (!is.null(time_per_iter_sampling_by_step_count_table) &&
            any(time_per_iter_sampling_by_step_count_table$time_per_iter_sampling_method == "long_minus_short_call_not_positive_not_used")) {
              warning(paste0("sampling timing probe: the long call was not slower than the short call at ",
                             paste(time_per_iter_sampling_by_step_count_table$n_leapfrog_steps_per_iter[
                                   time_per_iter_sampling_by_step_count_table$time_per_iter_sampling_method == "long_minus_short_call_not_positive_not_used"],
                                   collapse = ", "),
                             " leapfrog step(s); those step counts are left out of the fit (timing noise; more probe iterations reduce it)."))
        }
        sampling_timing_probe_least_squares_fit <-  if (is.null(time_per_iter_sampling_by_step_count_table)) {
              fn_fit_time_per_iter_on_n_leapfrog_steps(n_leapfrog_steps_per_iter = numeric(0), time_per_iter = numeric(0))
        } else {
              fn_fit_time_per_iter_on_n_leapfrog_steps( n_leapfrog_steps_per_iter = time_per_iter_sampling_by_step_count_table$n_leapfrog_steps_per_iter,
                                                        time_per_iter             = time_per_iter_sampling_by_step_count_table$time_per_iter_sampling)
        }
        ##
        ## ---- summaries per iteration, on tiled probe draws (see "Summaries" above):
        ##
        time_per_iter_summaries_sampling <-  NA_real_
        sampling_timing_probe_summaries_calls_table <-  NULL
        ## (sampling_timing_probe_mode = "lite": the cache key of the summaries' fixed set-up and the cached set-up used or stored)
        sampling_timing_probe_summaries_set_up_cache_record <-  NULL
        time_per_iter_summaries_sampling_method <-  if (isTRUE(time_criterion_settings$sampling_timing_probe_time_summaries)) "not_available" else "not_timed"
        if (isTRUE(time_criterion_settings$sampling_timing_probe_time_summaries)) {
              if (isTRUE(use_disk)) {
                    time_per_iter_summaries_sampling_method <-  "not_timed_with_use_disk"
                    ## (sampling_overhead_in_leapfrog_steps = "auto" uses neither a user-supplied time_per_iter_summaries_sampling nor a previous
                    ##  run's value, so its warning names the setting that can replace the value:)
                    if (identical(time_criterion_settings$sampling_overhead_in_leapfrog_steps, "auto")) {
                          warning(paste0("sampling timing probe: with use_disk = TRUE the summaries routine reads the traces from the files of the sampling ",
                                         "call, which cannot be tiled, so the summaries are not timed; with sampling_overhead_in_leapfrog_steps = \"auto\" ",
                                         "the summaries part of sampling_overhead_in_leapfrog_steps is then 0 (a user-supplied time_per_iter_summaries_sampling and ",
                                         "a previous run's value are not used by \"auto\"); supply sampling_overhead_in_leapfrog_steps as a number instead."))
                    } else {
                          warning(paste0("sampling timing probe: with use_disk = TRUE the summaries routine reads the traces from the files of the sampling ",
                                         "call, which cannot be tiled, so the summaries are not timed; supply time_per_iter_summaries_sampling or ",
                                         "time_criterion_previous_run_path."))
                    }
              } else if (is.null(sampling_object_for_summaries) || fn_is_over_wall_time_limit()) {
                    time_per_iter_summaries_sampling_method <-  if (is.null(sampling_object_for_summaries)) "no_probe_draws" else "stopped_at_wall_time_limit"
                    warning("sampling timing probe: the summaries could not be timed (no probe draws, or the wall-time limit was reached).")
              ## } else {
              } else if (!sampling_timing_probe_is_lite) {
                    n_iter_of_draws_for_summaries <-  sampling_timing_probe_n_iter_per_L
                    summaries_n_iter_tiled_short_call <-  time_criterion_settings$sampling_timing_probe_summaries_n_iter_tiled_short_call
                    summaries_n_iter_tiled_long_call  <-  time_criterion_settings$sampling_timing_probe_summaries_n_iter_tiled_long_call
                    fn_time_summaries_on_tiled_draws <-  function(n_iter_target) {
                          fn_time_summaries_on_probe_draws( sampling_object           = fn_tile_sampling_object_draws( sampling_object = sampling_object_for_summaries$sampling_object,
                                                                                                                      n_iter_source   = n_iter_of_draws_for_summaries,
                                                                                                                      n_iter_target   = n_iter_target),
                                                            EHMC_args_for_call        = sampling_object_for_summaries$EHMC_args_for_call,
                                                            time_of_native_call       = sampling_object_for_summaries$time_of_native_call,
                                                            n_leapfrog_steps_for_call = sampling_object_for_summaries$n_leapfrog_steps_for_call)
                    }
                    ## two untimed warm-up calls (package loading; byte-code compilation at the second call of a closure):
                    time_summaries_on_warm_up_draws <-  c(fn_time_summaries_on_tiled_draws(n_iter_target = summaries_n_iter_tiled_short_call),
                                                          fn_time_summaries_on_tiled_draws(n_iter_target = summaries_n_iter_tiled_short_call))
                    time_summaries_on_short_tiled_draws <-  fn_time_summaries_on_tiled_draws(n_iter_target = summaries_n_iter_tiled_short_call)
                    time_summaries_on_long_tiled_draws  <-  fn_time_summaries_on_tiled_draws(n_iter_target = summaries_n_iter_tiled_long_call)
                    sampling_timing_probe_summaries_calls_table <-  data.frame( summaries_call          = c("warm_up_untimed_1", "warm_up_untimed_2", "short_tiled", "long_tiled"),
                                                                                n_iter_of_draws         = c(summaries_n_iter_tiled_short_call, summaries_n_iter_tiled_short_call,
                                                                                                            summaries_n_iter_tiled_short_call, summaries_n_iter_tiled_long_call),
                                                                                time_summaries_on_draws = c(time_summaries_on_warm_up_draws, time_summaries_on_short_tiled_draws,
                                                                                                            time_summaries_on_long_tiled_draws),
                                                                                stringsAsFactors        = FALSE)
                    time_per_iter_summaries_long_minus_short_draws <-  (time_summaries_on_long_tiled_draws - time_summaries_on_short_tiled_draws) /
                                                                       (summaries_n_iter_tiled_long_call - summaries_n_iter_tiled_short_call)
                    if (is.finite(time_per_iter_summaries_long_minus_short_draws) && time_per_iter_summaries_long_minus_short_draws > 0) {
                          time_per_iter_summaries_sampling <-  time_per_iter_summaries_long_minus_short_draws
                          time_per_iter_summaries_sampling_method <-  "long_minus_short_tiled_draws"
                    } else if (!is.finite(time_summaries_on_short_tiled_draws) || !is.finite(time_summaries_on_long_tiled_draws)) {
                          time_per_iter_summaries_sampling_method <-  "summaries_routine_failed"
                          warning("sampling timing probe: the summaries routine failed on the tiled probe draws; time_per_iter_summaries_sampling is not available.")
                    } else {
                          time_per_iter_summaries_sampling_method <-  "long_minus_short_tiled_draws_not_positive_not_used"
                          warning(paste0("sampling timing probe: the summaries took no longer on ", summaries_n_iter_tiled_long_call, " than on ",
                                         summaries_n_iter_tiled_short_call, " draw rows (timing noise); time_per_iter_summaries_sampling is not available."))
                    }
              } else {
                    ##
                    ## ---- sampling_timing_probe_mode = "lite": the summaries routine timed once on the long tiled draws, less its fixed set-up
                    ##      cached for this R session and configuration (see "Lite mode" above; fn_time_summaries_per_draw_row_lite_with_cached_set_up):
                    ##
                    fn_time_summaries_on_tiled_probe_draws <-  function(n_iter_target) {
                          fn_time_summaries_on_probe_draws( sampling_object           = fn_tile_sampling_object_draws( sampling_object = sampling_object_for_summaries$sampling_object,
                                                                                                                      n_iter_source   = sampling_timing_probe_n_iter_per_L,
                                                                                                                      n_iter_target   = n_iter_target),
                                                            EHMC_args_for_call        = sampling_object_for_summaries$EHMC_args_for_call,
                                                            time_of_native_call       = sampling_object_for_summaries$time_of_native_call,
                                                            n_leapfrog_steps_for_call = sampling_object_for_summaries$n_leapfrog_steps_for_call)
                    }
                    summaries_lite_result <-  fn_time_summaries_per_draw_row_lite_with_cached_set_up(
                          fn_time_summaries_on_tiled_draws   = fn_time_summaries_on_tiled_probe_draws,
                          summaries_n_iter_tiled_short_call  = time_criterion_settings$sampling_timing_probe_summaries_n_iter_tiled_short_call,
                          summaries_n_iter_tiled_long_call   = time_criterion_settings$sampling_timing_probe_summaries_n_iter_tiled_long_call,
                          summaries_set_up_cache_key         = sampling_timing_probe_summaries_set_up_cache_key_for_probe)
                    time_per_iter_summaries_sampling <-  summaries_lite_result$time_per_draw_row_summaries
                    time_per_iter_summaries_sampling_method <-  summaries_lite_result$method
                    sampling_timing_probe_summaries_calls_table <-  summaries_lite_result$summaries_calls_table
                    sampling_timing_probe_summaries_set_up_cache_record <-  summaries_lite_result[c("summaries_set_up_cache_key",
                                                                                                    "summaries_set_up_was_cached_before_this_probe",
                                                                                                    "summaries_set_up_cache_entry")]
              }
        }
        rm(sampling_object_for_summaries)
        ##
        sampling_timing_probe_n_iter_total <-  sum(sampling_timing_probe_calls_table$n_iter)
        sampling_timing_probe_n_divergent_trajectories <-  sum(sampling_timing_probe_calls_table$n_divergent_trajectories)
        sampling_timing_probe_divergent_trajectory_fraction <-  if (sampling_timing_probe_n_iter_total > 0) {
              sampling_timing_probe_n_divergent_trajectories / (sampling_timing_probe_n_iter_total * n_chains_sampling)
        } else NA_real_
        if (is.finite(sampling_timing_probe_divergent_trajectory_fraction) && sampling_timing_probe_divergent_trajectory_fraction > 0.10) {
              warning(paste0("sampling timing probe: ", signif(100 * sampling_timing_probe_divergent_trajectory_fraction, 3), "% of the probe trajectories diverged; ",
                             "a trajectory whose log density becomes non-finite stops early, which lowers the measured leapfrog time."))
        }
        ##
        sampling_timing_probe_status <-  if (sampling_timing_probe_least_squares_fit$status %in% c("fitted", "fitted_intercept_set_to_zero")) "fitted" else
                                         if (stopped_at_wall_time_limit) "stopped_at_wall_time_limit_before_fit" else paste0("fit_", sampling_timing_probe_least_squares_fit$status)
        sampling_timing_probe_wall_time <-  fn_wall_time_in_seconds_since(sampling_timing_probe_start_time)
        ##
        sampling_timing_probe_result <-  list( status                                          = sampling_timing_probe_status,
                                               sampling_timing_probe_L_values                  = sampling_timing_probe_L_values,
                                               sampling_timing_probe_n_iter_per_L              = sampling_timing_probe_n_iter_per_L,
                                               sampling_timing_probe_n_iter_short_call         = sampling_timing_probe_n_iter_short_call,
                                               sampling_timing_probe_max_wall_time             = sampling_timing_probe_max_wall_time,
                                               stopped_at_wall_time_limit                      = stopped_at_wall_time_limit,
                                               eps_main_for_probe                              = eps_main_for_probe,
                                               eps_us_for_probe                                = eps_us_for_probe,
                                               eps_source_for_probe                            = eps_source_for_probe,
                                               n_chains_sampling                               = n_chains_sampling,
                                               n_threads_WCP_sampling                          = n_threads_WCP_sampling,
                                               sampling_timing_probe_n_iter_total              = sampling_timing_probe_n_iter_total,
                                               sampling_timing_probe_n_divergent_trajectories  = sampling_timing_probe_n_divergent_trajectories,
                                               sampling_timing_probe_n_iter_warm_up_call       = sampling_timing_probe_n_iter_warm_up_call,
                                               time_of_warm_up_native_call                     = time_of_warm_up_native_call,
                                               sampling_timing_probe_calls_table               = sampling_timing_probe_calls_table,
                                               sampling_timing_probe_summaries_calls_table     = sampling_timing_probe_summaries_calls_table,
                                               time_per_iter_sampling_by_step_count_table      = time_per_iter_sampling_by_step_count_table,
                                               sampling_timing_probe_fit_status                = sampling_timing_probe_least_squares_fit$status,
                                               time_per_leapfrog_step_sampling                 = sampling_timing_probe_least_squares_fit$time_per_leapfrog_step,
                                               time_per_iter_overhead_sampling                 = sampling_timing_probe_least_squares_fit$time_per_iter_overhead,
                                               time_per_iter_summaries_sampling                = time_per_iter_summaries_sampling,
                                               time_per_iter_summaries_sampling_method         = time_per_iter_summaries_sampling_method,
                                               ## sampling_timing_probe_wall_time                 = sampling_timing_probe_wall_time)
                                               sampling_timing_probe_wall_time                 = sampling_timing_probe_wall_time,
                                               sampling_timing_probe_mode                      = sampling_timing_probe_mode,
                                               sampling_timing_probe_summaries_set_up_cache_record = sampling_timing_probe_summaries_set_up_cache_record)
        ##
        message(colourise(paste0("sampling timing probe: ", sampling_timing_probe_status,
                                 " | time_per_leapfrog_step_sampling = ", signif(sampling_timing_probe_result$time_per_leapfrog_step_sampling, 4), " s",
                                 " | time_per_iter_overhead_sampling = ", signif(sampling_timing_probe_result$time_per_iter_overhead_sampling, 4), " s",
                                 " | time_per_iter_summaries_sampling = ", signif(time_per_iter_summaries_sampling, 4), " s",
                                 " | wall time ", signif(sampling_timing_probe_wall_time, 4), " s"), "cyan"))
        return(sampling_timing_probe_result)

}


##
## ---- The mode of the sampling timing probe for a run ------------------------------------------------------------------------------
##
##      sampling_timing_probe_mode as set; NULL (default) = "lite" with sampling_overhead_in_leapfrog_steps = "auto" and "full" (the probe
##      described in "Sampling timing probe" above, without its "Lite mode") otherwise.
##
fn_sampling_timing_probe_mode_for_run <-  function(time_criterion_settings) {

        if (!is.null(time_criterion_settings$sampling_timing_probe_mode)) return(time_criterion_settings$sampling_timing_probe_mode)
        if (identical(time_criterion_settings$sampling_overhead_in_leapfrog_steps, "auto")) return("lite")
        return("full")

}


##
## ---- Sampling timing probe, "lite" mode: the summaries routine's fixed set-up, cached for the R session --------------------------------
##
##      The summaries routine's wall time on n draw rows is about time_of_summaries_fixed_set_up + n * time_per_draw_row. At the first
##      "lite" probe of a configuration in an R session the set-up is measured by the full probe's method (two untimed calls on the
##      short tiled draws, then one timed call on the short and one on the long tiled draws):
##        time_per_draw_row               = (time on the long tiled draws - time on the short tiled draws) / (n_iter_tiled_long - n_iter_tiled_short),
##        time_of_summaries_fixed_set_up  = time on the short tiled draws - n_iter_tiled_short * time_per_draw_row,
##      and this probe's time_per_iter_summaries_sampling is that time_per_draw_row. The set-up (when it is not negative) is stored in the
##      cache under the configuration's key (fn_sampling_timing_probe_summaries_set_up_cache_key: model type, full path and modification
##      time of the model file, the size of every data element, numbers of main and nuisance parameters, nuisance parameters tracked,
##      n_chains_sampling, store_log_lik_trace and the summary arguments other than the output folder use_disk_path_post_hoc_dir, which
##      changes with use_disk_path from run to run). Every later "lite" probe
##      with the same key times the routine ONCE, on the long tiled draws:
##        time_per_iter_summaries_sampling = (time on the long tiled draws - time_of_summaries_fixed_set_up) / n_iter_tiled_long.
##      A difference that is not positive, or a routine that fails, gives NA with a warning, as in the full probe.
##
##      The cache is an environment that belongs to the package namespace (fn_sampling_timing_probe_summaries_set_up_cache), so it
##      lasts as long as the R session: a new R session, or a fit run in a fresh R process (run_in_fresh_R_process = TRUE), starts with an
##      empty cache, and its first "lite" probe measures the set-up again. NicoStan:::fn_clear_sampling_timing_probe_summaries_set_up_cache()
##      empties it (for example to measure the set-up again after the load on the machine has changed).
##
fn_sampling_timing_probe_summaries_set_up_cache <-  local({

        sampling_timing_probe_summaries_set_up_cache_environment <-  new.env(parent = emptyenv())
        function() sampling_timing_probe_summaries_set_up_cache_environment

})

fn_clear_sampling_timing_probe_summaries_set_up_cache <-  function() {

        summaries_set_up_cache_environment <-  fn_sampling_timing_probe_summaries_set_up_cache()
        cached_summaries_set_up_keys <-  ls(envir = summaries_set_up_cache_environment, all.names = TRUE)
        rm(list = cached_summaries_set_up_keys, envir = summaries_set_up_cache_environment)
        message(colourise(paste0("sampling timing probe: ", length(cached_summaries_set_up_keys), " cached summaries set-up time(s) cleared."), "cyan"))
        return(invisible(length(cached_summaries_set_up_keys)))

}

##
##      The key records the data by the size of each element (data_for_cache_key: Stan_data_list for Stan models, list(y = y) for the
##      built-in models): the dimensions, or the length, of every element, and the value of every element of length one (such as N),
##      because the routine's fixed set-up (the Stan data JSON file it writes and the BridgeStan models it builds with those data)
##      grows with the data size. The compiled model is recorded by its full path and modification time (model_so_file = "none", the
##      built-in models, as it is), so two models with the same file name in different folders, or a rebuilt model, have
##      different keys.
##
fn_sampling_timing_probe_summaries_set_up_cache_key <-  function( Model_type,
                                                                  model_so_file,
                                                                  n_params_main,
                                                                  n_nuisance,
                                                                  n_nuisance_to_track,
                                                                  n_chains_sampling,
                                                                  store_log_lik_trace,
                                                                  summaries_arguments_from_settings,
                                                                  data_for_cache_key = NULL) {

        ## (the output folder of the routine's post-hoc files does not change its work, and it changes with use_disk_path from run to run:)
        summaries_arguments_for_cache_key <-  summaries_arguments_from_settings
        summaries_arguments_for_cache_key$use_disk_path_post_hoc_dir <-  NULL
        ##
        model_so_file_for_cache_key <-  if (is.character(model_so_file) && length(model_so_file) == 1 && !is.na(model_so_file) && file.exists(model_so_file)) {
              paste0(normalizePath(model_so_file, winslash = "/", mustWork = TRUE), " (modified ", format(as.numeric(file.mtime(model_so_file)), digits = 15), ")")
        } else if (is.character(model_so_file)) paste(model_so_file, collapse = ",") else "none"
        ##
        fn_size_of_one_data_element_for_cache_key <-  function(data_element) {
              if (is.list(data_element) && !is.data.frame(data_element)) return(paste0("list(", length(data_element), ")"))
              if (!is.null(dim(data_element))) return(paste0("dim ", paste(dim(data_element), collapse = "x")))
              if (length(data_element) == 1 && (is.numeric(data_element) || is.logical(data_element) || is.character(data_element))) {
                    return(paste0("value ", format(data_element, digits = 15)))
              }
              return(paste0("length ", length(data_element)))
        }
        data_sizes_for_cache_key <-  if (is.list(data_for_cache_key) && length(data_for_cache_key) > 0) {
              data_element_names <-  names(data_for_cache_key)
              if (is.null(data_element_names)) data_element_names <-  rep("", length(data_for_cache_key))
              paste(paste0(data_element_names, ": ", vapply(X = data_for_cache_key, FUN = fn_size_of_one_data_element_for_cache_key, FUN.VALUE = character(1))),
                    collapse = "; ")
        } else "none"
        ##
        return(paste0("Model_type=", paste(Model_type, collapse = ","),
                      " | model_so_file=", model_so_file_for_cache_key,
                      " | data=", data_sizes_for_cache_key,
                      " | n_params_main=", paste(n_params_main, collapse = ","),
                      " | n_nuisance=", paste(n_nuisance, collapse = ","),
                      " | n_nuisance_to_track=", paste(n_nuisance_to_track, collapse = ","),
                      " | n_chains_sampling=", paste(n_chains_sampling, collapse = ","),
                      " | store_log_lik_trace=", paste(store_log_lik_trace, collapse = ","),
                      " | summary_arguments=", paste(deparse(summaries_arguments_for_cache_key), collapse = "")))

}

##
##      fn_time_summaries_on_tiled_draws(n_iter_target) is the probe's timing of the summaries routine on its draws tiled to n_iter_target
##      draw rows (its wall time in seconds, NA when the routine failed). Returns time_per_draw_row_summaries, the method, the calls table,
##      the cache key, whether the set-up was cached before this probe and the cache entry used or stored.
##
fn_time_summaries_per_draw_row_lite_with_cached_set_up <-  function( fn_time_summaries_on_tiled_draws,
                                                                     summaries_n_iter_tiled_short_call,
                                                                     summaries_n_iter_tiled_long_call,
                                                                     summaries_set_up_cache_key) {

        summaries_set_up_cache_environment <-  fn_sampling_timing_probe_summaries_set_up_cache()
        summaries_set_up_cache_entry <-  if (exists(x = summaries_set_up_cache_key, envir = summaries_set_up_cache_environment, inherits = FALSE))
                                             get(x = summaries_set_up_cache_key, envir = summaries_set_up_cache_environment, inherits = FALSE) else NULL
        summaries_set_up_was_cached_before_this_probe <-  !is.null(summaries_set_up_cache_entry)
        time_per_draw_row_summaries <-  NA_real_
        ##
        if (!summaries_set_up_was_cached_before_this_probe) {
              ##
              ## ---- the first "lite" probe of this configuration in the R session: the full probe's four calls, and the set-up is cached:
              ##
              time_summaries_on_warm_up_draws <-  c(fn_time_summaries_on_tiled_draws(n_iter_target = summaries_n_iter_tiled_short_call),
                                                    fn_time_summaries_on_tiled_draws(n_iter_target = summaries_n_iter_tiled_short_call))
              time_summaries_on_short_tiled_draws <-  fn_time_summaries_on_tiled_draws(n_iter_target = summaries_n_iter_tiled_short_call)
              time_summaries_on_long_tiled_draws  <-  fn_time_summaries_on_tiled_draws(n_iter_target = summaries_n_iter_tiled_long_call)
              summaries_calls_table <-  data.frame( summaries_call          = c("warm_up_untimed_1", "warm_up_untimed_2", "short_tiled", "long_tiled"),
                                                    n_iter_of_draws         = c(summaries_n_iter_tiled_short_call, summaries_n_iter_tiled_short_call,
                                                                                summaries_n_iter_tiled_short_call, summaries_n_iter_tiled_long_call),
                                                    time_summaries_on_draws = c(time_summaries_on_warm_up_draws, time_summaries_on_short_tiled_draws,
                                                                                time_summaries_on_long_tiled_draws),
                                                    stringsAsFactors        = FALSE)
              time_per_draw_row_long_minus_short_draws <-  (time_summaries_on_long_tiled_draws - time_summaries_on_short_tiled_draws) /
                                                           (summaries_n_iter_tiled_long_call - summaries_n_iter_tiled_short_call)
              if (is.finite(time_per_draw_row_long_minus_short_draws) && time_per_draw_row_long_minus_short_draws > 0) {
                    time_per_draw_row_summaries <-  time_per_draw_row_long_minus_short_draws
                    time_of_summaries_fixed_set_up <-  time_summaries_on_short_tiled_draws - summaries_n_iter_tiled_short_call * time_per_draw_row_long_minus_short_draws
                    if (is.finite(time_of_summaries_fixed_set_up) && time_of_summaries_fixed_set_up >= 0) {
                          summaries_set_up_cache_entry <-  list( time_of_summaries_fixed_set_up          = time_of_summaries_fixed_set_up,
                                                                 time_per_draw_row_when_set_up_measured  = time_per_draw_row_long_minus_short_draws,
                                                                 summaries_n_iter_tiled_short_call       = summaries_n_iter_tiled_short_call,
                                                                 summaries_n_iter_tiled_long_call        = summaries_n_iter_tiled_long_call,
                                                                 time_when_set_up_measured               = Sys.time())
                          assign(x = summaries_set_up_cache_key, value = summaries_set_up_cache_entry, envir = summaries_set_up_cache_environment)
                          method <-  "long_minus_short_tiled_draws_set_up_measured_and_cached"
                    } else {
                          method <-  "long_minus_short_tiled_draws_set_up_negative_not_cached"
                    }
              } else if (!is.finite(time_summaries_on_short_tiled_draws) || !is.finite(time_summaries_on_long_tiled_draws)) {
                    method <-  "summaries_routine_failed"
                    warning("sampling timing probe (lite): the summaries routine failed on the tiled probe draws; time_per_iter_summaries_sampling is not available.")
              } else {
                    method <-  "long_minus_short_tiled_draws_not_positive_not_used"
                    warning(paste0("sampling timing probe (lite): the summaries took no longer on ", summaries_n_iter_tiled_long_call, " than on ",
                                   summaries_n_iter_tiled_short_call, " draw rows (timing noise); time_per_iter_summaries_sampling is not available."))
              }
        } else {
              ##
              ## ---- the set-up is cached: one timed call on the long tiled draws:
              ##
              time_summaries_on_long_tiled_draws <-  fn_time_summaries_on_tiled_draws(n_iter_target = summaries_n_iter_tiled_long_call)
              summaries_calls_table <-  data.frame( summaries_call          = "long_tiled_single_call",
                                                    n_iter_of_draws         = summaries_n_iter_tiled_long_call,
                                                    time_summaries_on_draws = time_summaries_on_long_tiled_draws,
                                                    stringsAsFactors        = FALSE)
              time_per_draw_row_single_call_minus_cached_set_up <-  (time_summaries_on_long_tiled_draws - summaries_set_up_cache_entry$time_of_summaries_fixed_set_up) /
                                                                    summaries_n_iter_tiled_long_call
              if (is.finite(time_per_draw_row_single_call_minus_cached_set_up) && time_per_draw_row_single_call_minus_cached_set_up > 0) {
                    time_per_draw_row_summaries <-  time_per_draw_row_single_call_minus_cached_set_up
                    method <-  "single_long_tiled_call_minus_cached_set_up"
              } else if (!is.finite(time_summaries_on_long_tiled_draws)) {
                    method <-  "summaries_routine_failed"
                    warning("sampling timing probe (lite): the summaries routine failed on the tiled probe draws; time_per_iter_summaries_sampling is not available.")
              } else {
                    method <-  "single_long_tiled_call_minus_cached_set_up_not_positive_not_used"
                    warning(paste0("sampling timing probe (lite): the summaries on ", summaries_n_iter_tiled_long_call, " draw rows took no longer than the ",
                                   "cached fixed set-up (", signif(summaries_set_up_cache_entry$time_of_summaries_fixed_set_up, 4), " s; timing noise, or a set-up ",
                                   "that has changed: NicoStan:::fn_clear_sampling_timing_probe_summaries_set_up_cache() measures it again); ",
                                   "time_per_iter_summaries_sampling is not available."))
              }
        }
        ##
        message(colourise(paste0("sampling timing probe (lite): summaries ", method,
                                 if (!is.null(summaries_set_up_cache_entry)) paste0(" | fixed set-up ", signif(summaries_set_up_cache_entry$time_of_summaries_fixed_set_up, 4), " s") else "",
                                 " | time_per_iter_summaries_sampling = ", signif(time_per_draw_row_summaries, 4), " s"), "cyan"))
        return(list( time_per_draw_row_summaries                    = time_per_draw_row_summaries,
                     method                                         = method,
                     summaries_calls_table                          = summaries_calls_table,
                     summaries_set_up_cache_key                     = summaries_set_up_cache_key,
                     summaries_set_up_was_cached_before_this_probe  = summaries_set_up_was_cached_before_this_probe,
                     summaries_set_up_cache_entry                   = summaries_set_up_cache_entry))

}


##
## ---- Is the probe needed? ------------------------------------------------------------------------------------------------------
##
fn_sampling_timing_probe_is_needed <-  function( time_criterion_settings,
                                                 ## time_criterion_previous_run_quantities) {
                                                 time_criterion_previous_run_quantities,
                                                 ##
                                                 ## ---- sampling_overhead_in_leapfrog_steps = "auto" only: the built-in default for the model and sampling
                                                 ##      configuration (fn_default_sampling_overhead_in_leapfrog_steps); NULL = none:
                                                 sampling_overhead_in_leapfrog_steps_built_in_default = NULL) {

        fn_is_single_finite_number <-  function(candidate_value) is.numeric(candidate_value) && length(candidate_value) == 1 && is.finite(candidate_value)
        previous_run_quantities <-  if (is.null(time_criterion_previous_run_quantities)) list() else time_criterion_previous_run_quantities
        ##
        ## ---- sampling_overhead_in_leapfrog_steps supplied as a number: no probe. The sampling leapfrog time of
        ##      burnin_to_sampling_leapfrog_time_ratio then comes from the user or a previous run; failing both,
        ##      burnin_to_sampling_leapfrog_time_ratio = 0 with a warning (fn_resolve_time_criterion_sampling_quantities).
        ##      "auto": the probe is needed exactly when there is no built-in default for the model and sampling configuration (a
        ##      previous run or user-supplied times do not replace it):
        ##
        if (fn_is_single_finite_number(time_criterion_settings$sampling_overhead_in_leapfrog_steps)) {
              return(FALSE)
        }
        if (identical(time_criterion_settings$sampling_overhead_in_leapfrog_steps, "auto")) {
              return(is.null(sampling_overhead_in_leapfrog_steps_built_in_default))
        }
        ##
        time_per_leapfrog_step_sampling_available <-  fn_is_single_finite_number(time_criterion_settings$time_per_leapfrog_step_sampling) ||
                                                      fn_is_single_finite_number(previous_run_quantities$time_per_leapfrog_step_sampling)
        sampling_overhead_in_leapfrog_steps_available <-  fn_is_single_finite_number(time_criterion_settings$sampling_overhead_in_leapfrog_steps) ||
                                                          fn_is_single_finite_number(previous_run_quantities$sampling_overhead_in_leapfrog_steps) ||
                                                          (fn_is_single_finite_number(time_criterion_settings$time_per_iter_overhead_sampling) &&
                                                           fn_is_single_finite_number(time_criterion_settings$time_per_iter_summaries_sampling) &&
                                                           time_per_leapfrog_step_sampling_available)
        return(!(time_per_leapfrog_step_sampling_available && sampling_overhead_in_leapfrog_steps_available))

}


##
## ---- Resolve the sampling-side quantities ----------------------------------------------------------------------------------------
##
fn_resolve_time_criterion_sampling_quantities <-  function( time_criterion_settings,
                                                            time_criterion_previous_run_quantities,
                                                            sampling_timing_probe_result,
                                                            ## n_iter_planned) {
                                                            n_iter_planned,
                                                            ##
                                                            ## ---- sampling_overhead_in_leapfrog_steps = "auto" only: the built-in default for the model and sampling
                                                            ##      configuration (fn_default_sampling_overhead_in_leapfrog_steps); NULL = none:
                                                            sampling_overhead_in_leapfrog_steps_built_in_default = NULL) {

        fn_is_single_finite_number <-  function(candidate_value) is.numeric(candidate_value) && length(candidate_value) == 1 && is.finite(candidate_value)
        previous_run_quantities <-  if (is.null(time_criterion_previous_run_quantities)) list() else time_criterion_previous_run_quantities
        sampling_timing_probe_quantities <-  if (is.null(sampling_timing_probe_result) || !identical(sampling_timing_probe_result$status, "fitted")) list() else
                                             sampling_timing_probe_result
        ##
        fn_resolve_one_quantity <-  function(quantity_name) {
              if (fn_is_single_finite_number(time_criterion_settings[[quantity_name]]))          return(list(value = time_criterion_settings[[quantity_name]],          source = "user_supplied"))
              if (fn_is_single_finite_number(previous_run_quantities[[quantity_name]]))          return(list(value = previous_run_quantities[[quantity_name]],          source = "previous_run"))
              if (fn_is_single_finite_number(sampling_timing_probe_quantities[[quantity_name]])) return(list(value = sampling_timing_probe_quantities[[quantity_name]], source = "sampling_timing_probe"))
              return(list(value = NA_real_, source = "not_available"))
        }
        ##
        time_per_leapfrog_step_sampling_resolved <-  fn_resolve_one_quantity("time_per_leapfrog_step_sampling")
        time_per_iter_overhead_sampling_resolved <-  fn_resolve_one_quantity("time_per_iter_overhead_sampling")
        time_per_iter_summaries_sampling_resolved <-  fn_resolve_one_quantity("time_per_iter_summaries_sampling")
        ##
        ## ---- sampling_overhead_in_leapfrog_steps: user-supplied directly; else from the three times above; else from the previous run:
        ##
        ## ---- (and before these: "auto" = a value for the sampling configuration: the built-in default, else the sampling timing
        ##      probe's own three times, (overhead + summaries) / leapfrog time, with the source "probe_lite" or "probe_full"
        ##      ("probe_lite_without_summaries" or "probe_full_without_summaries", with a warning, when the probe has no summaries time);
        ##      never a saved run, the burn-in or user-supplied component times)
        ##
        sampling_overhead_in_leapfrog_steps_is_auto <-  identical(time_criterion_settings$sampling_overhead_in_leapfrog_steps, "auto")
        sampling_overhead_in_leapfrog_steps_from_iteration_overhead <-  NA_real_
        sampling_overhead_in_leapfrog_steps_from_summaries <-  NA_real_
        ## if (fn_is_single_finite_number(time_criterion_settings$sampling_overhead_in_leapfrog_steps)) {
        if (sampling_overhead_in_leapfrog_steps_is_auto && !is.null(sampling_overhead_in_leapfrog_steps_built_in_default)) {
              sampling_overhead_in_leapfrog_steps_resolved <-  list(value  = sampling_overhead_in_leapfrog_steps_built_in_default$sampling_overhead_in_leapfrog_steps,
                                                                    source = "built_in_default")
              sampling_overhead_in_leapfrog_steps_from_iteration_overhead <-  sampling_overhead_in_leapfrog_steps_built_in_default$sampling_overhead_in_leapfrog_steps_from_iteration_overhead
              sampling_overhead_in_leapfrog_steps_from_summaries <-  sampling_overhead_in_leapfrog_steps_built_in_default$sampling_overhead_in_leapfrog_steps_from_summaries
        } else if (sampling_overhead_in_leapfrog_steps_is_auto &&
                   fn_is_single_finite_number(sampling_timing_probe_quantities$time_per_leapfrog_step_sampling) &&
                   sampling_timing_probe_quantities$time_per_leapfrog_step_sampling > 0 &&
                   fn_is_single_finite_number(sampling_timing_probe_quantities$time_per_iter_overhead_sampling)) {
              sampling_overhead_in_leapfrog_steps_from_iteration_overhead <-  sampling_timing_probe_quantities$time_per_iter_overhead_sampling /
                                                                              sampling_timing_probe_quantities$time_per_leapfrog_step_sampling
              sampling_timing_probe_has_summaries_time <-  fn_is_single_finite_number(sampling_timing_probe_quantities$time_per_iter_summaries_sampling)
              sampling_overhead_in_leapfrog_steps_from_summaries <-  (if (sampling_timing_probe_has_summaries_time)
                                                                         sampling_timing_probe_quantities$time_per_iter_summaries_sampling else 0) /
                                                                     sampling_timing_probe_quantities$time_per_leapfrog_step_sampling
              sampling_timing_probe_mode_of_result <-  if (is.null(sampling_timing_probe_quantities$sampling_timing_probe_mode)) "full" else
                                                           sampling_timing_probe_quantities$sampling_timing_probe_mode
              sampling_overhead_in_leapfrog_steps_resolved <-  list(value  = sampling_overhead_in_leapfrog_steps_from_iteration_overhead + sampling_overhead_in_leapfrog_steps_from_summaries,
                                                                    source = paste0("probe_", sampling_timing_probe_mode_of_result,
                                                                                    if (sampling_timing_probe_has_summaries_time) "" else "_without_summaries"))
              ## (no summaries time from the probe: use_disk = TRUE, sampling_timing_probe_time_summaries = FALSE, a failed summaries routine
              ##  or a long-minus-short difference that is not positive; "auto" takes no user-supplied or previous-run summaries time, so the
              ##  summaries part is 0 and the warning names the setting that can replace the value:)
              if (!sampling_timing_probe_has_summaries_time) {
                    warning(paste0("CHESSR_time / SNAPER_time: sampling_overhead_in_leapfrog_steps = \"auto\": the sampling timing probe (",
                                   sampling_timing_probe_mode_of_result, ") has no time_per_iter_summaries_sampling (",
                                   if (is.null(sampling_timing_probe_quantities$time_per_iter_summaries_sampling_method)) "method not recorded" else
                                       sampling_timing_probe_quantities$time_per_iter_summaries_sampling_method,
                                   "), so sampling_overhead_in_leapfrog_steps = ", signif(sampling_overhead_in_leapfrog_steps_from_iteration_overhead, 4),
                                   " is the iteration overhead part only (summaries part 0); supply sampling_overhead_in_leapfrog_steps as a number ",
                                   "to include the summaries."))
              }
        } else if (sampling_overhead_in_leapfrog_steps_is_auto) {
              sampling_overhead_in_leapfrog_steps_resolved <-  list(value = 0, source = "not_available_set_to_zero")
              warning(paste0("CHESSR_time / SNAPER_time: sampling_overhead_in_leapfrog_steps = \"auto\" has no built-in default for this model and ",
                             "sampling configuration and no successful sampling timing probe; it is set to 0."))
        } else if (fn_is_single_finite_number(time_criterion_settings$sampling_overhead_in_leapfrog_steps)) {
              sampling_overhead_in_leapfrog_steps_resolved <-  list(value = time_criterion_settings$sampling_overhead_in_leapfrog_steps, source = "user_supplied")
        } else if (fn_is_single_finite_number(time_per_leapfrog_step_sampling_resolved$value) && time_per_leapfrog_step_sampling_resolved$value > 0 &&
                   fn_is_single_finite_number(time_per_iter_overhead_sampling_resolved$value)) {
              sampling_overhead_in_leapfrog_steps_resolved <-  list(value  = (time_per_iter_overhead_sampling_resolved$value +
                                                                              (if (fn_is_single_finite_number(time_per_iter_summaries_sampling_resolved$value))
                                                                                   time_per_iter_summaries_sampling_resolved$value else 0)) /
                                                                             time_per_leapfrog_step_sampling_resolved$value,
                                                                    source = paste0("computed_from_times (overhead: ", time_per_iter_overhead_sampling_resolved$source,
                                                                                    ", summaries: ", time_per_iter_summaries_sampling_resolved$source,
                                                                                    ", leapfrog: ", time_per_leapfrog_step_sampling_resolved$source, ")"))
        } else if (fn_is_single_finite_number(previous_run_quantities$sampling_overhead_in_leapfrog_steps)) {
              sampling_overhead_in_leapfrog_steps_resolved <-  list(value = previous_run_quantities$sampling_overhead_in_leapfrog_steps, source = "previous_run")
        } else {
              sampling_overhead_in_leapfrog_steps_resolved <-  list(value = 0, source = "not_available_set_to_zero")
              warning(paste0("CHESSR_time / SNAPER_time: sampling_overhead_in_leapfrog_steps is not available (no user value, previous run ",
                             "or successful sampling timing probe); it is set to 0."))
        }
        ##
        ## ---- the two parts of a sampling_overhead_in_leapfrog_steps computed from the three times (NULL setting), as for "auto":
        ##
        if (!sampling_overhead_in_leapfrog_steps_is_auto && startsWith(sampling_overhead_in_leapfrog_steps_resolved$source, "computed_from_times")) {
              sampling_overhead_in_leapfrog_steps_from_iteration_overhead <-  time_per_iter_overhead_sampling_resolved$value / time_per_leapfrog_step_sampling_resolved$value
              sampling_overhead_in_leapfrog_steps_from_summaries <-  (if (fn_is_single_finite_number(time_per_iter_summaries_sampling_resolved$value))
                                                                         time_per_iter_summaries_sampling_resolved$value else 0) /
                                                                     time_per_leapfrog_step_sampling_resolved$value
        }
        ##
        ## ---- (sampling_overhead_in_leapfrog_steps = "auto" with the built-in default, where no probe runs, as with a number: the
        ##      time_per_leapfrog_step_sampling of burnin_to_sampling_leapfrog_time_ratio is the user's, else a previous run's; with
        ##      neither, burnin_to_sampling_leapfrog_time_ratio = 0 with the warning below. The default's calibrated leapfrog time is of the
        ##      calibration machine, and the ratio divides a burn-in time measured on the machine of the run, so it is never used here.)
        ##
        if (!fn_is_single_finite_number(time_per_leapfrog_step_sampling_resolved$value)) {
              warning(paste0("CHESSR_time / SNAPER_time: time_per_leapfrog_step_sampling is not available; ",
                             "burnin_to_sampling_leapfrog_time_ratio will be 0."))
        }
        ##
        ## ---- n_iter_sampling_for_time_criterion: user-supplied; else ESS target / expected ESS per iteration; else the planned n_iter:
        ##
        ESS_per_iter_sampling_expected_resolved <-  fn_resolve_one_quantity("ESS_per_iter_sampling_expected")
        if (fn_is_single_finite_number(time_criterion_settings$n_iter_sampling_for_time_criterion)) {
              n_iter_sampling_for_time_criterion_resolved <-  list(value = time_criterion_settings$n_iter_sampling_for_time_criterion, source = "user_supplied")
        } else if (fn_is_single_finite_number(time_criterion_settings$ESS_target_for_time_criterion) &&
                   fn_is_single_finite_number(ESS_per_iter_sampling_expected_resolved$value) &&
                   ESS_per_iter_sampling_expected_resolved$value > 0) {
              n_iter_sampling_for_time_criterion_resolved <-  list(value  = time_criterion_settings$ESS_target_for_time_criterion / ESS_per_iter_sampling_expected_resolved$value,
                                                                   source = paste0("ESS_target_for_time_criterion / ESS_per_iter_sampling_expected (",
                                                                                   ESS_per_iter_sampling_expected_resolved$source, ")"))
        } else {
              n_iter_sampling_for_time_criterion_resolved <-  list(value = n_iter_planned, source = "planned_n_iter_of_this_run")
        }
        ##
        return(list( time_per_leapfrog_step_sampling               = time_per_leapfrog_step_sampling_resolved$value,
                     time_per_leapfrog_step_sampling_source        = time_per_leapfrog_step_sampling_resolved$source,
                     time_per_iter_overhead_sampling               = time_per_iter_overhead_sampling_resolved$value,
                     time_per_iter_overhead_sampling_source        = time_per_iter_overhead_sampling_resolved$source,
                     time_per_iter_summaries_sampling              = time_per_iter_summaries_sampling_resolved$value,
                     time_per_iter_summaries_sampling_source       = time_per_iter_summaries_sampling_resolved$source,
                     sampling_overhead_in_leapfrog_steps           = sampling_overhead_in_leapfrog_steps_resolved$value,
                     sampling_overhead_in_leapfrog_steps_source    = sampling_overhead_in_leapfrog_steps_resolved$source,
                     ESS_target_for_time_criterion                 = if (fn_is_single_finite_number(time_criterion_settings$ESS_target_for_time_criterion))
                                                                         time_criterion_settings$ESS_target_for_time_criterion else NA_real_,
                     ESS_per_iter_sampling_expected                = ESS_per_iter_sampling_expected_resolved$value,
                     ESS_per_iter_sampling_expected_source         = ESS_per_iter_sampling_expected_resolved$source,
                     n_iter_sampling_for_time_criterion            = n_iter_sampling_for_time_criterion_resolved$value,
                     n_iter_sampling_for_time_criterion_source     = n_iter_sampling_for_time_criterion_resolved$source,
                     time_criterion_ess_parameter_set              = paste(if (is.null(time_criterion_settings$time_criterion_ess_parameter_set)) "diagnostic" else
                                                                               time_criterion_settings$time_criterion_ess_parameter_set, collapse = ","),
                     time_per_leapfrog_step_burnin_user_supplied   = if (fn_is_single_finite_number(time_criterion_settings$time_per_leapfrog_step_burnin))
                                                                         time_criterion_settings$time_per_leapfrog_step_burnin else NA_real_,
                     ## time_per_leapfrog_step_burnin_min_iterations  = time_criterion_settings$time_per_leapfrog_step_burnin_min_iterations))
                     time_per_leapfrog_step_burnin_min_iterations  = time_criterion_settings$time_per_leapfrog_step_burnin_min_iterations,
                     ##
                     ## ---- sampling_overhead_in_leapfrog_steps = "auto" (its source above: built_in_default, probe_lite, probe_full,
                     ##      probe_lite_without_summaries or probe_full_without_summaries), the two
                     ##      parts of sampling_overhead_in_leapfrog_steps (time_per_iter_overhead_sampling and time_per_iter_summaries_sampling over
                     ##      time_per_leapfrog_step_sampling; NA when it was supplied as a number or read from a previous run), the built-in default
                     ##      (NULL = none, or not "auto"), and MALT's exponent (NULL settings = the defaults: lag_one_autocorrelation_rho = 1,
                     ##      lag_one_autocorrelation_rho_moment_averaging_offset = 8):
                     sampling_overhead_in_leapfrog_steps_is_auto                  = sampling_overhead_in_leapfrog_steps_is_auto,
                     sampling_overhead_in_leapfrog_steps_from_iteration_overhead  = sampling_overhead_in_leapfrog_steps_from_iteration_overhead,
                     sampling_overhead_in_leapfrog_steps_from_summaries           = sampling_overhead_in_leapfrog_steps_from_summaries,
                     sampling_overhead_in_leapfrog_steps_built_in_default         = sampling_overhead_in_leapfrog_steps_built_in_default,
                     lag_one_autocorrelation_rho                                  = if (is.null(time_criterion_settings$lag_one_autocorrelation_rho)) 1 else
                                                                                        time_criterion_settings$lag_one_autocorrelation_rho,
                     lag_one_autocorrelation_rho_moment_averaging_offset          = if (is.null(time_criterion_settings$lag_one_autocorrelation_rho_moment_averaging_offset)) 8 else
                                                                                        time_criterion_settings$lag_one_autocorrelation_rho_moment_averaging_offset))

}























