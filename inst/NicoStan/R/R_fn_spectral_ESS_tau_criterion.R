#### ===========================================================================================================
## R_fn_spectral_ESS_tau_criterion.R
##
## ---- burnin_algorithm = "LQ_ESSR_spectral" (6 Oct 2026) ------------------------------------------------------
##
##      The soft minimum (power 8), over the monitored coordinates j, of two ESS fractions per expected
##      gradient (the expected executed leapfrog steps E[L] of a trajectory at the current eps), maximised over
##      tau directly as "LQ_ESSR_length_response" (R_fn_length_response_tau_criterion.R), with a different
##      LINEAR term:
##
##      Linear term (spectral): the flow's decorrelation curve of z_j, m_j(t) = 1 - rho_j(t), from the
##      UNWEIGHTED proposal jumps (z_t,j - z_0,j)^2 / (2 Var(z_j)) of the valid chains, binned by the EXECUTED
##      trajectory length t = L eps (L = max(1, ceiling(tau_ii / eps))), with the forgotten sums of
##      cos(omega_k t) per bin for a log-spaced grid of frequencies omega_k (a quarter of a doubling apart), is
##      fitted by non-negative weights w_jk summing to 1:  m_j(t) = sum_k w_jk (1 - cos(omega_k t)),  for the
##      grid frequencies from omega_min to pi / eps. omega_min = the first grid frequency >= pi / (2 T_max),
##      T_max = the longest executed length with data: the slowest cosine that reaches 0 within the explored
##      lengths (with t ~ U(0, 2 tau), T_max is about 2 tau, so omega_min tau is about pi / 4). Motion slower
##      than that cannot be resolved by the data and can only take its weight at omega_min, the most optimistic
##      place the data allow, so the term stays an upper bound and pushes tau up until that motion is resolved.
##      The constant is generic (the definition of the explored range), not calibrated on any model.
##      The iteration kernel composes the flow with acceptance as laziness: for a candidate tau',
##        lambda_k(tau') = 1 - a_bar (1 - E[cos(omega_k L eps)]),   L = max(1, ceiling(U 2 tau' / eps)),
##      a_bar = the moving average (decay 0.9) of the chains' mean acceptance probability, and the ESS fraction of
##      z_j is  1 / sum_k w_jk (1 + lambda_k) / (1 - lambda_k)  (exact for the normal modes of a Gaussian target
##      with exact dynamics, a full momentum refresh and state-independent rejections).
##      (Why unweighted proposal jumps: acceptance-weighted jumps give 1 - rho(t) = a (1 - rho_flow(t)), and a fit
##      with weights summing to 1 would read the constant 1 - a as unresolved slow motion and push tau long. With
##      accepted ends instead of proposals (weight_by_probability = FALSE) the jumps already hold the rejections,
##      so a_bar = 1 is used.)
##
##      Quadratic term: unchanged from "LQ_ESSR_length_response", the lag-one bound g(b) = (1 - b) / (1 + b) of
##      the centred squares z_j^2 from the acceptance-weighted jumps, binned by the jittered length tau_ii,
##      through the piecewise-linear integral b_j(tau') = 1 - (1 / (2 tau')) int_0^{2 tau'} m_j(t) dt: the
##      iteration kernel's own lag-one autocorrelation (rejections included), with no extrapolation in t.
##
##      Candidates, the upper-end rule and the tau move are those of "LQ_ESSR_length_response" (a grid of tau'
##      1/16 of a doubling apart from the first bin centre up to the longest length / 2; tau moves the fraction
##      learning_rate_now of the distance in log(tau) to the maximiser, or towards twice the larger of tau and the
##      largest candidate when the maximiser is within 1/8 doubling of it).
##      The NNLS weights: Lawson-Hanson on the normal equations
##      (src_extra/spectral_ESS_nnls_normal_equations.cpp, compiled at first use), weighted by the bins'
##      forgotten chain counts, sum(w) = 1 by a heavily weighted row.
##
## ---- burnin_algorithm = "LQ_ESSR_spectral_long_bin_memory" (7 Oct 2026): exactly "LQ_ESSR_spectral" with the
##      length bins (the flow bins and the quadratic bins) forgetting at bin_forgetting_factor = 0.99 per update
##      instead of 0.95, so that the objective rests on about 5 times more trajectories. 0.99 was chosen in an
##      offline stationary stress test on four development models (PS7 binary LC-MVP, PS7 ordinal LC-MVOP, German
##      Credit, stochastic volatility).
##
## ---- burnin_algorithm = "LQ_ESSR_spec_bins99_evid_expand" (7 Oct 2026): "LQ_ESSR_spectral_long_bin_memory"
##      that doubles tau at the upper end of the candidates only on evidence that the objective still rises there.
##      More precisely (expansion_rule = "evidence"): a maximiser at the upper end targets 2 max(tau, largest
##      candidate) only when the objective at the largest candidate exceeds the objective one bin width (a quarter
##      of a doubling) below it by more than the one-sided 5% normal quantile times the standard deviation of
##      that difference over a parametric bootstrap of the bin means (every flow and quadratic bin mean perturbed
##      by its own estimated standard error, i.e. the forgotten within-bin variance over the forgotten chain
##      count; n_bootstrap_draws = 32 draws from their own random-number stream, so that R's stream is left as it
##      was); otherwise the target is the maximiser itself.
## ---- burnin_algorithm = "ESJD_w20_LQ_ESSR_spec_bins99_evid_expand" and
##      "ESJD_w33_LQ_ESSR_spec_bins99_evid_expand"
##      (7 Oct 2026): the weighted geometric mean of the ESJD objective (weight 0.20 / 0.33) and the objective
##      of "LQ_ESSR_spec_bins99_evid_expand" (weight 0.80 / 0.67), each divided by its maximum over the
##      candidates: "ESJD_LQ_ESSR_spec_bins99_evid_expand" with a lighter ESJD weight (objective_mix_ESJD_weight;
##      0.5 = the equal-weight mean of H and D, computed exactly as before).
## ---- burnin_algorithm = "LQ_ESSR_spec_bins99_evid_expand_jump_accept" (7 Oct 2026):
##      "LQ_ESSR_spec_bins99_evid_expand" with the jump-weighted acceptance per monitored row and flow length
##      bin in the spectral term (jump_weighted_acceptance = TRUE): the lag-one factor of row j and
##      frequency k is 1 - E_T[a_j(T) (1 - cos(omega_k T))] with a_j(T) = sum(acceptance x jump) / sum(jump)
##      of row j in the flow bin of T, instead of the single moving-average acceptance a_bar (the same when
##      the acceptance is constant).
## ---- burnin_algorithm = "ESJD_LQ_ESSR_spec_bins99_evid_expand" (7 Oct 2026): the same equal-weight geometric
##      mean as below without the finite-run ESS (finite_run_draws = NULL).
## ---- burnin_algorithm = "ESJD_LQ_ESSR_spec_bins99_finite_N_evid_expand" (7 Oct 2026): the equal-weight
##      geometric mean of ESJD and "LQ_ESSR_spec_bins99_evid_expand" with the finite-run ESS of the run's
##      N = n_iter draws. More precisely: objective_mix = "geometric_mean_with_ESJD", each objective divided by
##      its maximum over the candidates; the ESJD objective = the mean acceptance times the sum over the
##      monitored rows of the flow bins' normalised squared jumps averaged over executed lengths
##      s ~ U(0, 2 tau') (piecewise-linear in s through 0, constant beyond the longest bin), per expected
##      leapfrog step; finite_run_draws = N replaces (1 + lambda) / (1 - lambda) by
##      V_N(lambda) = 1 + 2 sum_{k=1}^{N-1} (1 - k / N) lambda^k, the variance inflation of the mean of N draws,
##      for every spectral component and for the quadratic term's lag-one b.
##      Both were selected from seven variants in an offline stationary stress test on five development
##      models (the four above and a joint longitudinal-survival model).
##

##
## ---- NicoStan's jitter (EHMC "Compute L"): L = max(1, ceiling(U 2 tau / eps)) leapfrog steps, executed
##      length t = L eps. With a = 2 tau / eps and n = ceiling(a): P(L = l) = 1 / a for l < n and
##      (a - n + 1) / a for l = n. E[cos(omega L eps)] in closed form from
##      S(n) = sum_{l = 1}^{n} cos(l x) = sin(n x / 2) cos((n + 1) x / 2) / sin(x / 2), x = omega eps
##      (S(n) = n where sin(x / 2) = 0):
##
fn_spectral_ESS_sum_of_cosines <-  function( x,
                                             n) {

        half_sine <-  sin(x / 2)
        value <-  sin(n * x / 2) * cos((n + 1) * x / 2) / half_sine
        exactly_periodic <-  abs(half_sine) < 1e-12
        value[exactly_periodic] <-  n[exactly_periodic]
        return(value)

}
fn_spectral_ESS_expected_cosine_under_jitter <-  function( omega,
                                                           tau,
                                                           eps) {

        a <-  2 * tau / eps
        if (a <= 1) return(cos(omega * eps))
        n <-  ceiling(a)
        x <-  omega * eps
        return((fn_spectral_ESS_sum_of_cosines(x, rep(n - 1, length(x))) + (a - n + 1) * cos(n * x)) / a)

}
fn_spectral_ESS_expected_leapfrog_steps_under_jitter <-  function( tau,
                                                                   eps) {

        a <-  2 * tau / eps
        if (a <= 1) return(1)
        n <-  ceiling(a)
        return(((n - 1) * n / 2 + (a - n + 1) * n) / a)

}

##
## ---- the NNLS solver, compiled once per R session at first use (the same scheme as the tau_initial nuisance
##      moments in init_and_run_burnin_ChESSR: the shared src_extra cache, then a cache of this process only):
##
fn_spectral_ESS_nnls_function <-  function() {

        cpp_file_name <-  "spectral_ESS_nnls_normal_equations.cpp"
        cpp_function_name <-  "fn_spectral_ESS_nnls_normal_equations"
        this_source_file <-  tryCatch(utils::getSrcFilename(fn_spectral_ESS_nnls_function, full.names = TRUE),
                                      error = function(e) character(0))
        cpp_file_candidates <-  c(getOption("NicoStan_spectral_ESS_nnls_cpp", default = NA_character_),
                                  if (length(this_source_file) == 1 && nzchar(this_source_file)) {
                                          file.path(dirname(dirname(this_source_file)), "inst", "src_extra",
                                                    cpp_file_name)
                                  } else NA_character_,
                                  system.file("src_extra", cpp_file_name, package = "NicoStan"))
        cpp_file_candidates <-  cpp_file_candidates[!is.na(cpp_file_candidates) & nzchar(cpp_file_candidates)]
        cpp_file <-  cpp_file_candidates[file.exists(cpp_file_candidates)][1]
        if (is.na(cpp_file)) {
                stop(paste0("burnin_algorithm = \"LQ_ESSR_spectral\": ", cpp_file_name, " not found (looked in: ",
                            paste(cpp_file_candidates, collapse = ", "), ")."))
        }
        cpp_file <-  normalizePath(cpp_file, mustWork = TRUE)
        cpp_file_md5 <-  unname(tools::md5sum(cpp_file))
        session_loaded <-  getOption("NicoStan_spectral_ESS_nnls_loaded", default = NULL)
        if (is.list(session_loaded) && identical(session_loaded$cpp_file, cpp_file) &&
            identical(session_loaded$cpp_file_md5, cpp_file_md5) && is.function(session_loaded$nnls_function)) {
                return(session_loaded$nnls_function)
        }
        fn_source_cpp_into <-  function(cache_dir) {
                dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)
                nnls_env <-  new.env()
                load_result <-  try(Rcpp::sourceCpp(file = cpp_file, cacheDir = cache_dir, env = nnls_env),
                                    silent = TRUE)
                if (inherits(load_result, "try-error") ||
                    !exists(cpp_function_name, envir = nnls_env, mode = "function", inherits = FALSE)) {
                        return(list(nnls_function = NULL, error = trimws(as.character(load_result))))
                }
                return(list(nnls_function = get(cpp_function_name, envir = nnls_env, inherits = FALSE),
                            error = NA_character_))
        }
        cache_dir <-  getOption("NicoStan_src_extra_cache_dir",
                                default = file.path(tools::R_user_dir(package = "NicoStan", which = "cache"),
                                                    "src_extra"))
        loaded <-  fn_source_cpp_into(cache_dir)
        if (is.null(loaded$nnls_function)) {
                process_cache_dir <-  file.path(cache_dir, paste0("process_", Sys.getpid()))
                message(colourise(paste0("LQ_ESSR_spectral: ", cpp_file, " did not load from the shared cache ",
                                         cache_dir, " (", loaded$error, "); compiling it again into ",
                                         process_cache_dir, "."), "cyan"))
                loaded <-  fn_source_cpp_into(process_cache_dir)
        }
        if (is.null(loaded$nnls_function)) {
                stop(paste0("burnin_algorithm = \"LQ_ESSR_spectral\": ", cpp_file, " did not compile or load (",
                            loaded$error, ")."))
        }
        options(NicoStan_spectral_ESS_nnls_loaded = list(cpp_file = cpp_file, cpp_file_md5 = cpp_file_md5,
                                                        nnls_function = loaded$nnls_function))
        return(loaded$nnls_function)

}

##
## ---- the flow state: the bins of log2(t / reference_length) a quarter of a doubling wide (as the
##      length-response state), over the EXECUTED lengths of the valid chains, with the forgotten sums of
##      cos(omega_k t) per bin for the grid frequencies omega_k (a quarter of a doubling apart, from the slowest
##      frequency any bin could resolve up to four times pi / eps at the start, so that a smaller eps later is
##      still covered):
##
fn_spectral_ESS_initialise_flow_state <-  function( n_linear_statistics,
                                                    reference_length,
                                                    eps_at_start,
                                                    bin_width_in_doublings        = 0.25,
                                                    lowest_bin_in_doublings       = -10,
                                                    highest_bin_in_doublings      = 6,
                                                    frequency_grid_step_in_doublings = 0.25) {

        bin_edges_in_doublings <-  seq( from = lowest_bin_in_doublings,
                                        to   = highest_bin_in_doublings,
                                        by   = bin_width_in_doublings)
        n_bins <-  length(bin_edges_in_doublings) + 1
        lowest_frequency <-  pi / (2 * 2^highest_bin_in_doublings * reference_length)
        highest_frequency <-  4 * pi / eps_at_start
        frequencies <-  2^seq(log2(lowest_frequency), log2(highest_frequency),
                              by = frequency_grid_step_in_doublings)
        ##
        return(list(reference_length                 = reference_length,
                    bin_edges_in_doublings           = bin_edges_in_doublings,
                    frequencies                      = frequencies,
                    sum_of_normalised_squared_jumps  = matrix(0, nrow = n_linear_statistics, ncol = n_bins),
                    sum_of_chain_counts              = numeric(n_bins),
                    sum_of_executed_lengths          = numeric(n_bins),
                    largest_executed_length_in_bin   = numeric(n_bins),
                    sum_of_cosines                   = matrix(0, nrow = n_bins, ncol = length(frequencies)),
                    n_updates                        = 0))

}
fn_spectral_ESS_add_flow_update <-  function( flow_state,
                                              normalised_squared_jumps,
                                              executed_lengths,
                                              flow_chains,
                                              forgetting_factor = 0.95) {

        if (!any(flow_chains)) return(flow_state)
        ##
        flow_state$sum_of_normalised_squared_jumps <-  forgetting_factor *
                                                       flow_state$sum_of_normalised_squared_jumps
        flow_state$sum_of_chain_counts <-  forgetting_factor * flow_state$sum_of_chain_counts
        flow_state$sum_of_executed_lengths <-  forgetting_factor * flow_state$sum_of_executed_lengths
        flow_state$sum_of_cosines <-  forgetting_factor * flow_state$sum_of_cosines
        bin_index <-  findInterval(log2(executed_lengths / flow_state$reference_length),
                                   flow_state$bin_edges_in_doublings) + 1
        for (chain_index in which(flow_chains)) {

                bin <-  bin_index[chain_index]
                flow_state$sum_of_normalised_squared_jumps[, bin] <-
                      flow_state$sum_of_normalised_squared_jumps[, bin] + normalised_squared_jumps[, chain_index]
                flow_state$sum_of_chain_counts[bin] <-  flow_state$sum_of_chain_counts[bin] + 1
                flow_state$sum_of_executed_lengths[bin] <-  flow_state$sum_of_executed_lengths[bin] +
                                                            executed_lengths[chain_index]
                flow_state$largest_executed_length_in_bin[bin] <-
                      max(flow_state$largest_executed_length_in_bin[bin], executed_lengths[chain_index])
                flow_state$sum_of_cosines[bin, ] <-  flow_state$sum_of_cosines[bin, ] +
                                                     cos(flow_state$frequencies * executed_lengths[chain_index])

        }
        flow_state$n_updates <-  flow_state$n_updates + 1
        ##
        return(flow_state)

}

##
## ---- the spectral weights (rows = the frequencies used, columns = the monitored coordinates); NULL while
##      fewer than 3 flow bins hold at least minimum_bin_count forgotten chain counts:
##
fn_spectral_ESS_fit_flow_weights <-  function( flow_state,
                                               eps_now,
                                               minimum_bin_count = 2) {

        usable_bins <-  which(flow_state$sum_of_chain_counts >= minimum_bin_count)
        if (length(usable_bins) < 3) return(NULL)
        longest_executed_length_with_data <-  max(flow_state$largest_executed_length_in_bin[usable_bins])
        lowest_resolvable_frequency <-  pi / (2 * longest_executed_length_with_data)
        highest_frequency_used <-  pi / eps_now
        first <-  which(flow_state$frequencies >= lowest_resolvable_frequency)[1]
        if (is.na(first)) first <-  length(flow_state$frequencies)
        last <-  max(c(first, which(flow_state$frequencies <= highest_frequency_used * (1 + 1e-9))))
        used <-  seq(first, last)
        counts <-  flow_state$sum_of_chain_counts[usable_bins]
        design <-  1 - flow_state$sum_of_cosines[usable_bins, used, drop = FALSE] / counts
        bin_means <-  flow_state$sum_of_normalised_squared_jumps[, usable_bins, drop = FALSE] /
                      rep(counts, each = nrow(flow_state$sum_of_normalised_squared_jumps))
        ## sum(w) = 1 by a heavily weighted row of ones:
        equality_weight <-  1e4 * sum(counts)
        AtA <-  crossprod(design, counts * design) + equality_weight
        Atb <-  crossprod(design, counts * t(bin_means)) + equality_weight
        nnls <-  fn_spectral_ESS_nnls_function()
        weights <-  nnls(AtA, Atb, 1e-12 * max(abs(Atb)))
        weights <-  sweep(weights, 2, pmax(colSums(weights), 1e-300), "/")
        ##
        return(list(frequencies                  = flow_state$frequencies[used],
                    weights                      = weights,
                    lowest_resolvable_frequency  = lowest_resolvable_frequency,
                    longest_executed_length      = longest_executed_length_with_data))

}

##
## ---- the full update for fn_metric_tau_block_update(): criterion = the per-coordinate statistics of
##      fn_metric_position_criterion() (as "LQ_ESSR"); state = the list carried between updates
##      (component_criterion_ema; NULL = none yet); eps_used_for_trajectories = the step size of this update's
##      trajectories (for their executed lengths); eps_now = the step size the next trajectories will use (the
##      highest frequency pi / eps_now and the expected leapfrog steps of the candidates). Returns the list that
##      fn_metric_tau_block_update() returns, as the length-response criteria:
##
## ---- (7 Oct 2026) the finite-run variance inflation of the mean of N draws of an AR(1)-type component,
##      V_N(lambda) = 1 + 2 sum_{k=1}^{N-1} (1 - k / N) lambda^k: the closed form, and its first-order expansion
##      N - (1 - lambda) (N^2 - 1) / 3 where N (1 - lambda) < 0.01 (the closed form cancels there):
##
fn_spectral_ESS_finite_run_variance_inflation <-  function( lambda,
                                                            N) {

        lambda <-  pmin(pmax(lambda, -1 + 1e-12), 1)
        one_minus <-  1 - lambda
        closed <-  1 + 2 * (lambda / one_minus - lambda * (1 - lambda^N) / (N * one_minus^2))
        near_one <-  one_minus * N < 0.01
        closed[near_one] <-  N - one_minus[near_one] * (N^2 - 1) / 3
        return(closed)

}

##
## ---- (7 Oct 2026) the forgotten sums of squares of the per-chain values in every length bin (the bins of
##      fn_length_response_add_update / fn_spectral_ESS_add_flow_update, with the same forgetting):
##
fn_spectral_ESS_add_bin_sum_of_squares <-  function( sum_of_squares,
                                                     values,
                                                     bin_index,
                                                     chains,
                                                     forgetting_factor) {

        sum_of_squares <-  forgetting_factor * sum_of_squares
        for (chain_index in which(chains)) {
                sum_of_squares[, bin_index[chain_index]] <-  sum_of_squares[, bin_index[chain_index]] +
                                                             values[, chain_index]^2
        }
        return(sum_of_squares)

}

##
## ---- (7 Oct 2026) the parametric-bootstrap draws of the objective per candidate (expansion_rule =
##      "evidence"): every quadratic and flow bin mean perturbed by its own estimated standard error (the
##      forgotten within-bin variance over the forgotten chain count; means kept >= 0), the objective recomputed
##      (fn_objective); the normal draws come from their own stream (seed 104729 + the number of bin updates so
##      far), and R's random-number stream is restored afterwards, so that the rest of the run draws exactly what
##      it would draw without the bootstrap:
##
fn_spectral_ESS_bootstrap_objective_draws <-  function( state,
                                                        usable_bins,
                                                        bin_means,
                                                        minimum_bin_count,
                                                        n_bootstrap_draws,
                                                        fn_objective,
                                                        n_candidates) {

        n_rows <-  nrow(bin_means)
        quadratic_state <-  state$quadratic_state
        quadratic_counts <-  quadratic_state$sum_of_chain_counts[usable_bins]
        quadratic_means <-  quadratic_state$sum_of_normalised_squared_jumps[, usable_bins, drop = FALSE] /
                            rep(quadratic_counts, each = n_rows)
        quadratic_variances <-  pmax(state$quadratic_sum_of_squares[, usable_bins, drop = FALSE] /
                                     rep(quadratic_counts, each = n_rows) - quadratic_means^2, 0)
        quadratic_standard_errors <-  sqrt(quadratic_variances / rep(quadratic_counts, each = n_rows))
        ## (the extra node at the longest length carries the last bin's mean, so also its standard error)
        if (ncol(bin_means) > ncol(quadratic_standard_errors)) {
                quadratic_standard_errors <-  cbind(quadratic_standard_errors,
                                                    quadratic_standard_errors[, ncol(quadratic_standard_errors)])
        }
        flow <-  state$flow_state
        flow_usable <-  which(flow$sum_of_chain_counts >= minimum_bin_count)
        flow_counts <-  flow$sum_of_chain_counts[flow_usable]
        flow_means <-  flow$sum_of_normalised_squared_jumps[, flow_usable, drop = FALSE] /
                       rep(flow_counts, each = n_rows)
        flow_standard_errors <-  sqrt(pmax(state$flow_sum_of_squares[, flow_usable, drop = FALSE] /
                                           rep(flow_counts, each = n_rows) - flow_means^2, 0) /
                                      rep(flow_counts, each = n_rows))
        ## ---- the bootstrap's own random-number stream (R's stream restored on exit):
        global_environment <-  globalenv()
        had_seed <-  exists(".Random.seed", envir = global_environment, inherits = FALSE)
        if (had_seed) saved_seed <-  get(".Random.seed", envir = global_environment, inherits = FALSE)
        on.exit({
                if (had_seed) {
                        assign(".Random.seed", saved_seed, envir = global_environment)
                } else if (exists(".Random.seed", envir = global_environment, inherits = FALSE)) {
                        rm(".Random.seed", envir = global_environment)
                }
        })
        set.seed(104729 + quadratic_state$n_updates)
        draws <-  vapply(seq_len(n_bootstrap_draws), function(draw_index) {
                perturbed_quadratic <-  pmax(bin_means + quadratic_standard_errors *
                                                         stats::rnorm(length(bin_means)), 0)
                perturbed_flow <-  flow
                perturbed_flow$sum_of_normalised_squared_jumps[, flow_usable] <-
                      pmax(flow_means + flow_standard_errors * stats::rnorm(length(flow_means)), 0) *
                      rep(flow_counts, each = n_rows)
                draw <-  fn_objective(perturbed_quadratic, perturbed_flow)
                if (is.null(draw)) return(rep(NA_real_, n_candidates))
                return(draw$objective)
        }, numeric(n_candidates))
        return(matrix(draws, nrow = n_candidates))

}

##
##
## ---- (7 Oct 2026) the linear scores (rows x candidates) of
##      "LQ_ESSR_spec_bins99_evid_expand_jump_accept": the lag-one factor of row j and frequency k is
##      1 - E_T[a_j(T) (1 - cos(omega_k T))] over NicoStan's jitter of the
##      executed lengths T = L eps_now, with a_j(T) = the forgotten sum of acceptance probability x normalised
##      proposal jump of row j over the forgotten sum of its normalised proposal jumps, in T's flow bin (valid
##      chains); a bin without a jump of row j uses the moving-average acceptance. The acceptance by row and bin
##      comes from the unperturbed state (also in the bootstrap draws):
##
fn_spectral_ESS_jump_weighted_acceptance_linear_scores <-  function( state,
                                                                     fit,
                                                                     candidate_tau,
                                                                     eps_now,
                                                                     finite_run_draws = NULL) {

        flow <-  state$flow_state
        jumps <-  flow$sum_of_normalised_squared_jumps
        acceptance_by_bin <-  state$flow_sum_of_acceptance_weighted_jumps / jumps
        acceptance_by_bin[!is.finite(acceptance_by_bin) | jumps <= 0] <-  NA_real_
        acceptance_by_bin <-  pmin(pmax(acceptance_by_bin, 0), 1)
        scores <-  matrix(NA_real_, nrow = nrow(jumps), ncol = length(candidate_tau))
        for (candidate_index in seq_along(candidate_tau)) {
                a <-  2 * candidate_tau[candidate_index] / eps_now
                if (a <= 1) {
                        steps <-  1
                        step_weights <-  1
                } else {
                        n <-  ceiling(a)
                        steps <-  seq_len(n)
                        step_weights <-  c(rep(1 / a, n - 1), (a - n + 1) / a)
                }
                lengths <-  steps * eps_now
                bins <-  findInterval(log2(lengths / flow$reference_length), flow$bin_edges_in_doublings) + 1
                acceptance <-  acceptance_by_bin[, bins, drop = FALSE]
                acceptance[is.na(acceptance)] <-  state$acceptance_moving_average
                one_minus_cosines <-  1 - cos(outer(fit$frequencies, lengths))
                lag_one <-  pmin(1 - one_minus_cosines %*% t(sweep(acceptance, 2, step_weights, "*")), 1 - 1e-9)
                variance_inflation <-  if (is.null(finite_run_draws)) {
                        (1 + lag_one) / (1 - lag_one)
                } else {
                        matrix(fn_spectral_ESS_finite_run_variance_inflation(lag_one, finite_run_draws),
                               nrow = nrow(lag_one))
                }
                scores[, candidate_index] <-  1 / colSums(fit$weights * variance_inflation)
        }
        return(scores)

}
fn_spectral_ESS_soft_minimum_tau_update <-  function( criterion,
                                                      state,
                                                      tau,
                                                      tau_values,
                                                      eps_used_for_trajectories,
                                                      eps_now,
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
                                                      soft_minimum_power               = 8,
                                                      minimum_bin_count                = 2,
                                                      grid_step_in_doublings           = 1 / 16,
                                                      upper_end_zone_in_doublings      = 1 / 8,
                                                      bin_forgetting_factor            = 0.95,
                                                      finite_run_draws                 = NULL,
                                                      expansion_rule                   = "as_built",
                                                      objective_mix                    = "none",
                                                      ## n_bootstrap_draws                = 32) {
                                                      n_bootstrap_draws                = 32,
                                                      ## jump_weighted_acceptance         = FALSE) {
                                                      jump_weighted_acceptance         = FALSE,
                                                      objective_mix_ESJD_weight        = 0.5) {

        required_fields <-  c("linear_jump_squared", "quadratic_jump_squared", "initial_second_moment",
                              "initial_fourth_moment")
        if (!is.list(criterion) || !all(required_fields %in% names(criterion))) {
                stop(paste0("LQ_ESSR_spectral needs the per-coordinate statistics of ",
                            "fn_metric_position_criterion()."))
        }
        if (!isTRUE(is.finite(eps_now)) || eps_now <= 0) {
                stop("LQ_ESSR_spectral needs the current step size (eps_now).")
        }
        if (!isTRUE(is.finite(eps_used_for_trajectories)) || eps_used_for_trajectories <= 0) {
                eps_used_for_trajectories <-  eps_now
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
        n_chains <-  length(probabilities)
        n_rows <-  nrow(criterion$linear_jump_squared)
        unchanged <-  function(state_to_return) {
                updated <-  c(tau, adam_mean, adam_variance)
                attr(updated, "adam_update_performed") <-  FALSE
                return(list(updated = updated, adam_update_performed = FALSE, gradient = NA_real_,
                            criterion = NA_real_, criterion_ema = NA_real_,
                            component_criterion_ema = state_to_return))
        }
        ##
        ## ---- which chains contribute (as "LQ_ESSR_length_response"):
        statistics_are_finite <-  apply(is.finite(criterion$linear_jump_squared) &
                                        is.finite(criterion$quadratic_jump_squared) &
                                        is.finite(criterion$initial_fourth_moment), 2, all)
        binned_chains <-  is.finite(tau_values) & tau_values > 0
        valid <-  statistics_are_finite & binned_chains & is.finite(divergences) & divergences == 0
        weights <-  if (isTRUE(use_proposals)) pmin(1, pmax(0, probabilities)) else rep(1, n_chains)
        weights[!is.finite(weights) | !valid] <-  0
        if (!any(binned_chains)) return(unchanged(state))
        executed_lengths <-  eps_used_for_trajectories * pmax(1, ceiling(tau_values / eps_used_for_trajectories))
        ##
        ## ---- the moving averages (decay 0.9) of the start moments over the valid chains, and of the chains'
        ##      mean acceptance probability (the laziness of the iteration kernel; 1 with accepted ends):
        if (any(valid)) {
                second_moment_now <-  rowMeans(criterion$initial_second_moment[, valid, drop = FALSE])
                fourth_moment_now <-  rowMeans(criterion$initial_fourth_moment[, valid, drop = FALSE])
        } else {
                second_moment_now <-  fourth_moment_now <-  NULL
        }
        acceptance_finite <-  binned_chains & is.finite(probabilities)
        acceptance_now <-  if (isTRUE(use_proposals) && any(acceptance_finite)) {
                mean(pmin(1, pmax(0, probabilities[acceptance_finite])))
        } else 1
        if (is.null(state)) {
                if (is.null(second_moment_now)) return(unchanged(NULL))
                state <-  list(quadratic_state             = fn_length_response_initialise_state(
                                       n_statistics     = n_rows,
                                       reference_length = mean(tau_values[binned_chains])),
                               flow_state                  = fn_spectral_ESS_initialise_flow_state(
                                       n_linear_statistics = n_rows,
                                       reference_length    = mean(executed_lengths[binned_chains]),
                                       eps_at_start        = eps_used_for_trajectories),
                               second_moment_moving_average = NULL,
                               fourth_moment_moving_average = NULL,
                               acceptance_moving_average    = NULL)
        }
        if (!is.null(second_moment_now)) {
                state$second_moment_moving_average <-  if (is.null(state$second_moment_moving_average)) {
                        second_moment_now
                } else 0.9 * state$second_moment_moving_average + 0.1 * second_moment_now
                state$fourth_moment_moving_average <-  if (is.null(state$fourth_moment_moving_average)) {
                        fourth_moment_now
                } else 0.9 * state$fourth_moment_moving_average + 0.1 * fourth_moment_now
        }
        state$acceptance_moving_average <-  if (is.null(state$acceptance_moving_average)) acceptance_now else
                                            0.9 * state$acceptance_moving_average + 0.1 * acceptance_now
        variance_linear <-  pmax(state$second_moment_moving_average, 1e-12)
        variance_quadratic <-  state$fourth_moment_moving_average - state$second_moment_moving_average^2
        variance_quadratic <-  ifelse(is.finite(variance_quadratic) & variance_quadratic > 0,
                                      variance_quadratic,
                                      2 * variance_linear^2)
        ##
        ## ---- the quadratic bins (as "LQ_ESSR_length_response": acceptance-weighted, by the jittered length):
        quadratic_jumps <-  criterion$quadratic_jump_squared
        quadratic_jumps[!is.finite(quadratic_jumps)] <-  0
        state$quadratic_state <-  fn_length_response_add_update( state              = state$quadratic_state,
                                                                 squared_jumps      = sweep(quadratic_jumps, 2,
                                                                                            weights, "*"),
                                                                 variances          = variance_quadratic,
                                                                 trajectory_lengths = tau_values,
                                                                 binned_chains      = binned_chains,
                                                                 forgetting_factor  = bin_forgetting_factor)
        ## (expansion_rule = "evidence" only) the forgotten sums of squares of the per-chain values in every
        ## quadratic bin, for the bin means' standard errors:
        if (identical(expansion_rule, "evidence")) {
                if (is.null(state$quadratic_sum_of_squares)) {
                        state$quadratic_sum_of_squares <-
                              0 * state$quadratic_state$sum_of_normalised_squared_jumps
                }
                state$quadratic_sum_of_squares <-  fn_spectral_ESS_add_bin_sum_of_squares(
                      sum_of_squares    = state$quadratic_sum_of_squares,
                      values            = sweep(quadratic_jumps, 2, weights, "*") / (2 * variance_quadratic),
                      bin_index         = findInterval(log2(tau_values / state$quadratic_state$reference_length),
                                                       state$quadratic_state$bin_edges_in_doublings) + 1,
                      chains            = binned_chains,
                      forgetting_factor = bin_forgetting_factor)
        }
        ##
        ## ---- the flow bins (unweighted proposal jumps of the valid chains, by the executed length):
        linear_jumps <-  criterion$linear_jump_squared
        linear_jumps[!is.finite(linear_jumps)] <-  0
        state$flow_state <-  fn_spectral_ESS_add_flow_update( flow_state               = state$flow_state,
                                                              normalised_squared_jumps = linear_jumps /
                                                                                         (2 * variance_linear),
                                                              executed_lengths         = executed_lengths,
                                                              flow_chains              = valid,
                                                              forgetting_factor        = bin_forgetting_factor)
        ## (expansion_rule = "evidence" only) the same for the flow bins:
        if (identical(expansion_rule, "evidence")) {
                if (is.null(state$flow_sum_of_squares)) {
                        state$flow_sum_of_squares <-  0 * state$flow_state$sum_of_normalised_squared_jumps
                }
                if (any(valid)) {
                        state$flow_sum_of_squares <-  fn_spectral_ESS_add_bin_sum_of_squares(
                              sum_of_squares    = state$flow_sum_of_squares,
                              values            = linear_jumps / (2 * variance_linear),
                              bin_index         = findInterval(log2(executed_lengths /
                                                                    state$flow_state$reference_length),
                                                               state$flow_state$bin_edges_in_doublings) + 1,
                              chains            = valid,
                              forgetting_factor = bin_forgetting_factor)
                }
        }
        ## (jump_weighted_acceptance only) the forgotten sums of acceptance probability x normalised proposal
        ## jump per row and flow bin (valid chains; forgotten only when a valid chain enters, as the flow state):
        if (isTRUE(jump_weighted_acceptance)) {
                if (is.null(state$flow_sum_of_acceptance_weighted_jumps)) {
                        state$flow_sum_of_acceptance_weighted_jumps <-  0 *
                              state$flow_state$sum_of_normalised_squared_jumps
                }
                if (any(valid)) {
                        flow_bin_index <-  findInterval(log2(executed_lengths /
                                                             state$flow_state$reference_length),
                                                        state$flow_state$bin_edges_in_doublings) + 1
                        state$flow_sum_of_acceptance_weighted_jumps <-  bin_forgetting_factor *
                              state$flow_sum_of_acceptance_weighted_jumps
                        for (chain_index in which(valid)) {
                                bin <-  flow_bin_index[chain_index]
                                state$flow_sum_of_acceptance_weighted_jumps[, bin] <-
                                      state$flow_sum_of_acceptance_weighted_jumps[, bin] +
                                      weights[chain_index] * linear_jumps[, chain_index] /
                                      (2 * variance_linear)
                        }
                }
        }
        ##
        ## ---- the candidates and the quadratic lag-one scores (as "LQ_ESSR_length_response"):
        quadratic_state <-  state$quadratic_state
        usable_bins <-  which(quadratic_state$sum_of_chain_counts >= minimum_bin_count)
        if (length(usable_bins) < 3) return(unchanged(state))
        bin_centres <-  quadratic_state$sum_of_trajectory_lengths[usable_bins] /
                        quadratic_state$sum_of_chain_counts[usable_bins]
        bin_means <-  quadratic_state$sum_of_normalised_squared_jumps[, usable_bins, drop = FALSE] /
                      rep(quadratic_state$sum_of_chain_counts[usable_bins],
                          each = nrow(quadratic_state$sum_of_normalised_squared_jumps))
        longest_length_with_data <-
              quadratic_state$largest_trajectory_length_in_bin[usable_bins[length(usable_bins)]]
        if (longest_length_with_data > bin_centres[length(bin_centres)]) {
                bin_centres <-  c(bin_centres, longest_length_with_data)
                bin_means <-  cbind(bin_means, bin_means[, ncol(bin_means)])
        }
        smallest_candidate <-  bin_centres[1]
        largest_candidate <-  bin_centres[length(bin_centres)] / 2
        if (!(largest_candidate > smallest_candidate)) return(unchanged(state))
        candidate_tau <-  2^seq(log2(smallest_candidate), log2(largest_candidate), by = grid_step_in_doublings)
        if (utils::tail(candidate_tau, 1) < largest_candidate) {
                candidate_tau <-  c(candidate_tau, largest_candidate)
        }
        integral_weights <-  fn_length_response_piecewise_linear_integral_weights(bin_centres, 2 * candidate_tau)
     ## quadratic_lag_one <-  1 - sweep(bin_means %*% integral_weights, 2, 2 * candidate_tau, "/")
     ## quadratic_lag_one <-  pmin(pmax(quadratic_lag_one, -1 + 1e-6), 1 - 1e-6)
     ## quadratic_scores <-  (1 - quadratic_lag_one) / (1 + quadratic_lag_one)
     ## ##
     ## ## ---- the spectral linear scores (the flow's fitted cosine mixture, composed with the laziness a_bar):
     ## fit <-  fn_spectral_ESS_fit_flow_weights( flow_state        = state$flow_state,
                                               ## eps_now           = eps_now,
                                               ## minimum_bin_count = minimum_bin_count)
     ## if (is.null(fit)) return(unchanged(state))
     ## expected_cosines <-  vapply(candidate_tau, function(candidate) {
             ## fn_spectral_ESS_expected_cosine_under_jitter(fit$frequencies, candidate, eps_now)
     ## }, numeric(length(fit$frequencies)))
     ## expected_cosines <-  matrix(expected_cosines, nrow = length(fit$frequencies))
     ## lag_one_per_frequency <-  1 - state$acceptance_moving_average * (1 - expected_cosines)
     ## lag_one_per_frequency <-  pmin(lag_one_per_frequency, 1 - 1e-9)
     ## integrated_autocorrelation_times <-  crossprod(fit$weights,
                                                    ## (1 + lag_one_per_frequency) / (1 - lag_one_per_frequency))
     ## linear_scores <-  1 / integrated_autocorrelation_times
     ## ##
     ## ## ---- the soft minimum per expected gradient, its maximiser, and the tau move:
     ## scores <-  rbind(linear_scores, quadratic_scores)
     ## smallest_scores <-  apply(scores, 2, min)
     ## relative_inverse_scores <-  sweep(1 / scores, 2, 1 / smallest_scores, "/")
     ## soft_minimum <-  smallest_scores *
                      ## colSums(relative_inverse_scores^soft_minimum_power)^(-1 / soft_minimum_power)
     ## expected_leapfrog_steps <-  vapply(candidate_tau, fn_spectral_ESS_expected_leapfrog_steps_under_jitter,
                                        ## numeric(1), eps = eps_now)
     ## criterion_per_candidate <-  soft_minimum / expected_leapfrog_steps
        ## ---- (7 Oct 2026) the same computation as a function of the quadratic bin means and the flow state
        ##      (the estimates, or a bootstrap draw of them), with the round-5 options finite_run_draws and
        ##      objective_mix (the header); with their defaults it returns exactly the objective as before:
        expected_leapfrog_steps <-  vapply(candidate_tau, fn_spectral_ESS_expected_leapfrog_steps_under_jitter,
                                           numeric(1), eps = eps_now)
        fn_objective <-  function(quadratic_bin_means, flow_state_used) {
                ## ---- the quadratic lag-one scores:
                quadratic_lag_one <-  1 - sweep(quadratic_bin_means %*% integral_weights, 2,
                                                2 * candidate_tau, "/")
                quadratic_lag_one <-  pmin(pmax(quadratic_lag_one, -1 + 1e-6), 1 - 1e-6)
                quadratic_scores <-  if (is.null(finite_run_draws)) {
                        (1 - quadratic_lag_one) / (1 + quadratic_lag_one)
                } else 1 / fn_spectral_ESS_finite_run_variance_inflation(quadratic_lag_one, finite_run_draws)
                ## ---- the spectral linear scores (the flow's fitted cosine mixture, with the laziness a_bar):
                fit <-  fn_spectral_ESS_fit_flow_weights( flow_state        = flow_state_used,
                                                          eps_now           = eps_now,
                                                          minimum_bin_count = minimum_bin_count)
                if (is.null(fit)) return(NULL)
                expected_cosines <-  vapply(candidate_tau, function(candidate) {
                        fn_spectral_ESS_expected_cosine_under_jitter(fit$frequencies, candidate, eps_now)
                }, numeric(length(fit$frequencies)))
                expected_cosines <-  matrix(expected_cosines, nrow = length(fit$frequencies))
                lag_one_per_frequency <-  1 - state$acceptance_moving_average * (1 - expected_cosines)
                lag_one_per_frequency <-  pmin(lag_one_per_frequency, 1 - 1e-9)
                variance_inflation <-  if (is.null(finite_run_draws)) {
                        (1 + lag_one_per_frequency) / (1 - lag_one_per_frequency)
                } else {
                        matrix(fn_spectral_ESS_finite_run_variance_inflation(lag_one_per_frequency,
                                                                             finite_run_draws),
                               nrow = nrow(lag_one_per_frequency))
                }
                integrated_autocorrelation_times <-  crossprod(fit$weights, variance_inflation)
                linear_scores <-  1 / integrated_autocorrelation_times
                if (isTRUE(jump_weighted_acceptance)) {
                        linear_scores <-  fn_spectral_ESS_jump_weighted_acceptance_linear_scores(
                              state, fit, candidate_tau, eps_now, finite_run_draws)
                }
                ## ---- the soft minimum per expected gradient:
                scores <-  rbind(linear_scores, quadratic_scores)
                smallest_scores <-  apply(scores, 2, min)
                relative_inverse_scores <-  sweep(1 / scores, 2, 1 / smallest_scores, "/")
                soft_minimum <-  smallest_scores *
                                 colSums(relative_inverse_scores^soft_minimum_power)^(-1 / soft_minimum_power)
                objective <-  soft_minimum / expected_leapfrog_steps
                ## ---- objective_mix = "geometric_mean_with_ESJD": the geometric mean with the ESJD objective:
                if (identical(objective_mix, "geometric_mean_with_ESJD")) {
                        flow_usable <-  which(flow_state_used$sum_of_chain_counts >= minimum_bin_count)
                        flow_counts <-  flow_state_used$sum_of_chain_counts[flow_usable]
                        flow_centres <-  flow_state_used$sum_of_executed_lengths[flow_usable] / flow_counts
                        flow_means <-  colSums(flow_state_used$sum_of_normalised_squared_jumps[, flow_usable,
                                                                                                 drop = FALSE]) /
                                       flow_counts
                        flow_order <-  order(flow_centres)
                        flow_centres <-  flow_centres[flow_order]
                        flow_means <-  flow_means[flow_order]
                        upper_limits <-  pmin(2 * candidate_tau, max(flow_centres))
                        flow_integral_weights <-  fn_length_response_piecewise_linear_integral_weights(
                              node_positions = flow_centres,
                              upper_limits   = upper_limits)
                        integral <-  as.numeric(flow_means %*% flow_integral_weights)
                        ## (beyond the longest bin centre the mean squared jump is carried as constant)
                        integral <-  integral + pmax(2 * candidate_tau - upper_limits, 0) *
                                                flow_means[length(flow_means)]
                        ESJD_objective <-  state$acceptance_moving_average * 2 * integral / (2 * candidate_tau) /
                                           expected_leapfrog_steps
                        ## objective <-  sqrt(objective / max(objective) * ESJD_objective / max(ESJD_objective))
                        ## (7 Oct 2026) the ESJD weight objective_mix_ESJD_weight; 0.5 = exactly as before:
                        objective <-  if (identical(objective_mix_ESJD_weight, 0.5)) {
                                sqrt(objective / max(objective) * ESJD_objective / max(ESJD_objective))
                        } else {
                                (objective / max(objective))^(1 - objective_mix_ESJD_weight) *
                                      (ESJD_objective / max(ESJD_objective))^objective_mix_ESJD_weight
                        }
                }
                return(list(objective = objective, fit = fit))
        }
        objective_estimate <-  fn_objective(bin_means, state$flow_state)
        if (is.null(objective_estimate)) return(unchanged(state))
        fit <-  objective_estimate$fit
        criterion_per_candidate <-  objective_estimate$objective
        best <-  which.max(criterion_per_candidate)
        maximum_is_at_upper_end <-  candidate_tau[best] >= largest_candidate * 2^(-upper_end_zone_in_doublings)
        schedule_iteration <-  if (is.null(learning_rate_schedule_iteration)) iteration else
                                                                              learning_rate_schedule_iteration
        schedule_length <-  if (is.null(learning_rate_schedule_length)) adaptation_iterations else
                                                                        learning_rate_schedule_length
        adaptation_progress <-  if (schedule_length <= 1) 0 else
                                min(1, max(0, (schedule_iteration - 1) / (schedule_length - 1)))
        learning_rate_now <-  learning_rate * (1 - (1 - learning_rate) * adaptation_progress)
     ## target_tau <-  if (isTRUE(maximum_is_at_upper_end)) {
             ## 2 * max(tau, largest_candidate)
     ## } else candidate_tau[best]
        ## (7 Oct 2026) expansion_rule = "as_built": every maximiser at the upper end targets
        ## 2 max(tau, largest candidate); "evidence": only when the bootstrap shows the objective still rising
        ## at the top (the header):
        expand <-  isTRUE(maximum_is_at_upper_end)
        evidence_z <-  NA_real_
        if (expand && identical(expansion_rule, "evidence")) {
                top <-  length(candidate_tau)
                reference <-  which.min(abs(log2(candidate_tau) - (log2(candidate_tau[top]) - 0.25)))
                draws <-  fn_spectral_ESS_bootstrap_objective_draws( state             = state,
                                                                     usable_bins       = usable_bins,
                                                                     bin_means         = bin_means,
                                                                     minimum_bin_count = minimum_bin_count,
                                                                     n_bootstrap_draws = n_bootstrap_draws,
                                                                     fn_objective      = fn_objective,
                                                                     n_candidates      = length(candidate_tau))
                difference_draws <-  draws[top, ] - draws[reference, ]
                difference_estimate <-  criterion_per_candidate[top] - criterion_per_candidate[reference]
                spread <-  stats::sd(difference_draws, na.rm = TRUE)
                evidence_z <-  if (is.finite(spread) && spread > 0) difference_estimate / spread else -Inf
                expand <-  reference < top && evidence_z > stats::qnorm(0.95)
        }
        target_tau <-  if (expand) 2 * max(tau, largest_candidate) else candidate_tau[best]
        new_tau <-  exp(log(tau) + learning_rate_now * (log(target_tau) - log(tau)))
        ##
        updated <-  c(new_tau, adam_mean, adam_variance)
        attr(updated, "adam_update_performed") <-  TRUE
        ## (expansion_rule = "evidence" only) also whether tau moved towards the doubling, and the test's z:
        if (identical(expansion_rule, "evidence")) {
                return(list(updated                      = updated,
                            adam_update_performed        = TRUE,
                            gradient                     = log(target_tau) - log(tau),
                            criterion                    = criterion_per_candidate[best],
                            criterion_ema                = criterion_per_candidate[best],
                            component_criterion_ema      = state,
                            tau_maximising_criterion     = candidate_tau[best],
                            maximum_is_at_upper_end      = maximum_is_at_upper_end,
                            largest_candidate            = largest_candidate,
                            learning_rate_now            = learning_rate_now,
                            lowest_resolvable_frequency  = fit$lowest_resolvable_frequency,
                            acceptance_moving_average    = state$acceptance_moving_average,
                            expanded_towards_doubling    = expand,
                            expansion_evidence_z         = evidence_z))
        }
        return(list(updated                      = updated,
                    adam_update_performed        = TRUE,
                    gradient                     = log(target_tau) - log(tau),
                    criterion                    = criterion_per_candidate[best],
                    criterion_ema                = criterion_per_candidate[best],
                    component_criterion_ema      = state,
                    tau_maximising_criterion     = candidate_tau[best],
                    maximum_is_at_upper_end      = maximum_is_at_upper_end,
                    largest_candidate            = largest_candidate,
                    learning_rate_now            = learning_rate_now,
                    lowest_resolvable_frequency  = fit$lowest_resolvable_frequency,
                    acceptance_moving_average    = state$acceptance_moving_average))

}
























