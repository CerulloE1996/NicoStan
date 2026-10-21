#### ===========================================================================================================
## R_fn_spectral_ESS_tau_criterion.R
##
## ---- burnin_algorithm = "LQ_ESSR_spectral" ------------------------------------------------------
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
## ---- burnin_algorithm = "LQ_ESSR_spectral_long_bin_memory": exactly "LQ_ESSR_spectral" with the
##      length bins (the flow bins and the quadratic bins) forgetting at bin_forgetting_factor = 0.99 per update
##      instead of 0.95, so that the objective rests on about 5 times more trajectories. 0.99 was chosen in an
##      offline stationary stress test on four development models (PS7 binary LC-MVP, PS7 ordinal LC-MVOP, German
##      Credit, stochastic volatility).
##
## ---- burnin_algorithm = "LQ_ESSR_spec_bins99_evid_expand": "LQ_ESSR_spectral_long_bin_memory"
##      that doubles tau at the upper end of the candidates only on evidence that the objective still rises there.
##      More precisely (expansion_rule = "evidence"): a maximiser at the upper end targets 2 max(tau, largest
##      candidate) only when the objective at the largest candidate exceeds the objective one bin width (a quarter
##      of a doubling) below it by more than the one-sided 5% normal quantile times the standard deviation of
##      that difference over a parametric bootstrap of the bin means (every flow and quadratic bin mean perturbed
##      by its own estimated standard error, i.e. the forgotten within-bin variance over the forgotten chain
##      count; n_bootstrap_draws = 32 draws from their own random-number stream, so that R's stream is left as it
##      was); otherwise the target is the maximiser itself.
## ---- burnin_algorithm = "ESJD_w20_LQ_ESSR_spec_bins99_evid_expand" and
##      "ESJD_w33_LQ_ESSR_spec_bins99_evid_expand":
##      the weighted geometric mean of the ESJD objective (weight 0.20 / 0.33) and the objective
##      of "LQ_ESSR_spec_bins99_evid_expand" (weight 0.80 / 0.67), each divided by its maximum over the
##      candidates: "ESJD_LQ_ESSR_spec_bins99_evid_expand" with a lighter ESJD weight (objective_mix_ESJD_weight;
##      0.5 = the equal-weight mean of H and D, computed exactly as before).
##
## ---- burnin_algorithm = "LQ_ESSR_spec_bins99_evid_expand_jump_accept":
##
##      "LQ_ESSR_spec_bins99_evid_expand" with the jump-weighted acceptance per monitored row and flow length
##      bin in the spectral term (jump_weighted_acceptance = TRUE): the lag-one factor of row j and
##      frequency k is 1 - E_T[a_j(T) (1 - cos(omega_k T))] with a_j(T) = sum(acceptance x jump) / sum(jump)
##      of row j in the flow bin of T, instead of the single moving-average acceptance a_bar (the same when
##      the acceptance is constant).
## ---- burnin_algorithm = "ESJD_LQ_ESSR_spec_bins99_evid_expand": the same equal-weight geometric
##      mean as below without the finite-run ESS (finite_run_draws = NULL).
## ---- burnin_algorithm = "ESJD_LQ_ESSR_spec_bins99_finite_N_evid_expand": the equal-weight
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
## ---- the spectral weights (rows = the frequencies used, columns = the monitored coordinates); NULL whilst
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
## ---- the finite-run variance inflation of the mean of N draws of an AR(1)-type component,
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
## ---- the forgotten sums of squares of the per-chain values in every length bin (the bins of
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
## ---- the parametric-bootstrap draws of the objective per candidate (expansion_rule =
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
        ## (SpESS_R_quadratic_part_normal_mode_spectral_fit only: the squares' flow bins, perturbed the same way)
        squares_flow <-  state$squares_flow_state
        if (!is.null(squares_flow)) {
                squares_usable <-  which(squares_flow$sum_of_chain_counts >= minimum_bin_count)
                squares_counts <-  squares_flow$sum_of_chain_counts[squares_usable]
                squares_means <-  squares_flow$sum_of_normalised_squared_jumps[, squares_usable, drop = FALSE] /
                                  rep(squares_counts, each = n_rows)
                squares_standard_errors <-  sqrt(pmax(state$squares_flow_sum_of_squares[, squares_usable,
                                                                                          drop = FALSE] /
                                                      rep(squares_counts, each = n_rows) - squares_means^2, 0) /
                                                 rep(squares_counts, each = n_rows))
        }
##
        ## ---- the bootstrap's own random-number stream (R's stream restored on exit):
##
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
                ## draw <-  fn_objective(perturbed_quadratic, perturbed_flow)
                if (is.null(squares_flow)) {
                        draw <-  fn_objective(perturbed_quadratic, perturbed_flow)
                } else {
                        perturbed_squares_flow <-  squares_flow
                        perturbed_squares_flow$sum_of_normalised_squared_jumps[, squares_usable] <-
                              pmax(squares_means + squares_standard_errors *
                                                   stats::rnorm(length(squares_means)), 0) *
                              rep(squares_counts, each = n_rows)
                        draw <-  fn_objective(perturbed_quadratic, perturbed_flow, perturbed_squares_flow)
                }
                if (is.null(draw)) return(rep(NA_real_, n_candidates))
                return(draw$objective)
        }, numeric(n_candidates))

        return(matrix(draws, nrow = n_candidates))

}

##
##
## ---- the linear scores (rows x candidates) of
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
##
## ---- the complete update in C++ (src_extra/spectral_LQ_ESSR_update.cpp; the moving averages, the
##      length bins, the cosine-mixture fit, the objective on the candidates, the bootstrap test of the expansion
##      rule "evidence" and the tau move, with the same arithmetic as the R code of
##      fn_spectral_ESS_soft_minimum_tau_update() below), compiled at first use as the NNLS solver. It is used for
##      the options it implements (finite_run_draws = NULL, objective_mix = "none", jump_weighted_acceptance =
##      FALSE: "LQ_ESSR_spectral", "LQ_ESSR_spectral_long_bin_memory" and "LQ_ESSR_spec_bins99_evid_expand"), so
##      that the criterion adds no computation in R beyond passing its inputs; options(
##      NicoStan_spectral_LQ_ESSR_update_in_R = TRUE) runs the R code instead (for checks):
##
fn_spectral_LQ_ESSR_update_cpp_function <-  function() {

        ## (once compiled and loaded in this R session, the function is returned at once: no file look-up or md5
        ##  per update; a changed .cpp is picked up by a new R session)
        session_loaded <-  getOption("NicoStan_spectral_LQ_ESSR_update_loaded", default = NULL)
        if (is.list(session_loaded) && is.function(session_loaded$update_function)) {
                return(session_loaded$update_function)
        }
        cpp_file_name <-  "spectral_LQ_ESSR_update.cpp"
        cpp_function_name <-  "fn_spectral_LQ_ESSR_update_cpp"
        this_source_file <-  tryCatch(utils::getSrcFilename(fn_spectral_LQ_ESSR_update_cpp_function,
                                                            full.names = TRUE),
                                      error = function(e) character(0))
        cpp_file_candidates <-  c(getOption("NicoStan_spectral_LQ_ESSR_update_cpp", default = NA_character_),
                                  if (length(this_source_file) == 1 && nzchar(this_source_file)) {

                                          file.path(dirname(dirname(this_source_file)), "inst", "src_extra",
                                                    cpp_file_name)
                                  } else NA_character_,

                                  system.file("src_extra", cpp_file_name, package = "NicoStan"))
        cpp_file_candidates <-  cpp_file_candidates[!is.na(cpp_file_candidates) & nzchar(cpp_file_candidates)]
        cpp_file <-  cpp_file_candidates[file.exists(cpp_file_candidates)][1]
        if (is.na(cpp_file)) {
                stop(paste0("the spectral LQ_ESSR criterion: ", cpp_file_name, " not found (looked in: ",
                            paste(cpp_file_candidates, collapse = ", "), ")."))
        }
        cpp_file <-  normalizePath(cpp_file, mustWork = TRUE)
        cpp_file_md5 <-  unname(tools::md5sum(cpp_file))
        session_loaded <-  getOption("NicoStan_spectral_LQ_ESSR_update_loaded", default = NULL)
        if (is.list(session_loaded) && identical(session_loaded$cpp_file, cpp_file) &&
            identical(session_loaded$cpp_file_md5, cpp_file_md5) && is.function(session_loaded$update_function)) {
                return(session_loaded$update_function)
        }
        fn_source_cpp_into <-  function(cache_dir) {

                dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)
                update_env <-  new.env()
                load_result <-  try(Rcpp::sourceCpp(file = cpp_file, cacheDir = cache_dir, env = update_env),
                                    silent = TRUE)
                if (inherits(load_result, "try-error") ||
                    !exists(cpp_function_name, envir = update_env, mode = "function", inherits = FALSE)) {
                        return(list(update_function = NULL, error = trimws(as.character(load_result))))
                }
                ## return(list(update_function = get(cpp_function_name, envir = update_env, inherits = FALSE),
                ##             error = NA_character_))
                ## (and the function returning a state with its bin matrices in R, from the same compiled file)
                return(list(update_function = get(cpp_function_name, envir = update_env, inherits = FALSE),
                            state_with_bin_matrices_in_R_function = get(
                                  "fn_spectral_LQ_ESSR_state_with_bin_matrices_in_R", envir = update_env,
                                  inherits = FALSE),
                            statistics_function = get("fn_spectral_LQ_ESSR_statistics_cpp", envir = update_env,
                                                      inherits = FALSE),
                            error = NA_character_))
        }

        cache_dir <-  getOption("NicoStan_src_extra_cache_dir",
                                default = file.path(tools::R_user_dir(package = "NicoStan", which = "cache"),
                                                    "src_extra"))
        loaded <-  fn_source_cpp_into(cache_dir)
        if (is.null(loaded$update_function)) {
                process_cache_dir <-  file.path(cache_dir, paste0("process_", Sys.getpid()))
                message(colourise(paste0("spectral LQ_ESSR: ", cpp_file, " did not load from the shared cache ",
                                         cache_dir, " (", loaded$error, "); compiling it again into ",
                                         process_cache_dir, "."), "cyan"))
                loaded <-  fn_source_cpp_into(process_cache_dir)
        }
        if (is.null(loaded$update_function)) {
                stop(paste0("the spectral LQ_ESSR criterion: ", cpp_file, " did not compile or load (",
                            loaded$error, ")."))
        }
        ## options(NicoStan_spectral_LQ_ESSR_update_loaded = list(cpp_file = cpp_file,
        ##                                                       cpp_file_md5 = cpp_file_md5,
        ##                                                       update_function = loaded$update_function))
        options(NicoStan_spectral_LQ_ESSR_update_loaded = list(
              cpp_file                              = cpp_file,
              cpp_file_md5                          = cpp_file_md5,
              update_function                       = loaded$update_function,
              state_with_bin_matrices_in_R_function = loaded$state_with_bin_matrices_in_R_function,
              statistics_function                   = loaded$statistics_function))
        return(loaded$update_function)

}
##
## ---- the criterion's state with its four bin matrices in R (copies of the matrices that the C++ update holds in
##      C++ memory between updates; see spectral_LQ_ESSR_update.cpp), as the update returned it before it held
##      them; a NULL state, or one with its matrices in R already, is returned as it is. The R code of the update
##      (options(NicoStan_spectral_LQ_ESSR_update_in_R = TRUE)) and the burn-in debug record use it, and the C++
##      update accepts this form too (a state saved with saveRDS() must be saved in this form):
##
fn_spectral_LQ_ESSR_state_with_bin_matrices_in_R <-  function(state) {

        if (!is.list(state) || is.null(state[["bin_matrices_held_in_cpp_between_updates"]])) return(state)
        fn_spectral_LQ_ESSR_update_cpp_function()
        session_loaded <-  getOption("NicoStan_spectral_LQ_ESSR_update_loaded", default = NULL)
        if (!is.function(session_loaded$state_with_bin_matrices_in_R_function)) {
                stop(paste0("the spectral LQ_ESSR criterion: the compiled update in this R session has no ",
                            "fn_spectral_LQ_ESSR_state_with_bin_matrices_in_R() (an older .cpp); start a new R ",
                            "session."))
        }
        return(session_loaded$state_with_bin_matrices_in_R_function(state))

}
##
## ---- the C++ function of SpESS-R's per-coordinate statistics for a diagonal trajectory metric (the same numbers
##      as the R code of fn_metric_position_criterion(); spectral_LQ_ESSR_update.cpp):
##
fn_spectral_LQ_ESSR_statistics_cpp_function <-  function() {

        fn_spectral_LQ_ESSR_update_cpp_function()
        session_loaded <-  getOption("NicoStan_spectral_LQ_ESSR_update_loaded", default = NULL)
        if (!is.function(session_loaded$statistics_function)) {
                stop(paste0("the spectral LQ_ESSR criterion: the compiled update in this R session has no ",
                            "fn_spectral_LQ_ESSR_statistics_cpp() (an older .cpp); start a new R session."))
        }
        return(session_loaded$statistics_function)

}
##
## ---- (SpESS_R_normalising_variances_from_last_n_updates = n > 0) the variances that SpESS-R divides the squared
##      jumps of z_j and of z_j^2 by, from the start moments z_j^2 and z_j^4 of the valid chains (finite
##      statistics, a trajectory, no divergence: the chains that enter the moving averages) of the last n updates
##      with a valid chain only, as plain means. As built, the moving averages (decay 0.9) of every update's start
##      moments keep the start of the burn-in, when the chains are far from the posterior, for tens of updates:
##      on the ordinal LC-MVOP model at N = 1,000 they overestimated Var(z_j^2) of the row that set the soft
##      minimum by about 10 to 40 times through the burn-in. The same formulas as for the moving averages:
##      Var(z_j) = max(m2, 1e-12) and Var(z_j^2) = m4 - m2^2 (2 Var(z_j)^2 when that is not positive).
##      window_in_state = the sums of the last updates (NULL = none yet). Returns the window with this update's
##      sums added and those beyond the last n dropped, and the two variances (NULL whilst no update has had a
##      valid chain):
##
fn_SpESS_R_variances_from_last_n_updates <-  function( window_in_state,
                                                       linear_jump_squared,
                                                       quadratic_jump_squared,
                                                       initial_second_moment,
                                                       initial_fourth_moment,
                                                       tau_values,
                                                       divergences,
                                                       n_updates) {

        statistics_are_finite <-  apply(is.finite(linear_jump_squared) & is.finite(quadratic_jump_squared) &
                                        is.finite(initial_fourth_moment), 2, all)
        valid <-  statistics_are_finite & is.finite(tau_values) & tau_values > 0 & is.finite(divergences) &
                  divergences == 0
        window <-  window_in_state
        if (any(valid)) {
                window <-  list(second_moment_sums = cbind(window$second_moment_sums,
                                                           rowSums(initial_second_moment[, valid, drop = FALSE])),
                                fourth_moment_sums = cbind(window$fourth_moment_sums,
                                                           rowSums(initial_fourth_moment[, valid, drop = FALSE])),
                                valid_chain_counts = c(window$valid_chain_counts, sum(valid)))
                kept <-  utils::tail(seq_along(window$valid_chain_counts), n_updates)
                window <-  list(second_moment_sums = window$second_moment_sums[, kept, drop = FALSE],
                                fourth_moment_sums = window$fourth_moment_sums[, kept, drop = FALSE],
                                valid_chain_counts = window$valid_chain_counts[kept])
        }
        if (is.null(window)) return(list(window = NULL, variance_linear = NULL, variance_quadratic = NULL))
        second_moment <-  rowSums(window$second_moment_sums) / sum(window$valid_chain_counts)
        fourth_moment <-  rowSums(window$fourth_moment_sums) / sum(window$valid_chain_counts)
        variance_linear <-  pmax(second_moment, 1e-12)
        variance_quadratic <-  fourth_moment - second_moment^2
        variance_quadratic <-  ifelse(is.finite(variance_quadratic) & variance_quadratic > 0,
                                      variance_quadratic,
                                      2 * variance_linear^2)
        return(list(window = window, variance_linear = variance_linear, variance_quadratic = variance_quadratic))

}
##
## ---- the call of the C++ update, and the list that fn_spectral_ESS_soft_minimum_tau_update() returns. The C++
##      bootstrap draws from its own generator (BootstrapNormalGenerator, seeded from the update count), so R's
##      random-number stream is not touched and is not saved or restored here (the R code's draws, with
##      options(NicoStan_spectral_LQ_ESSR_update_in_R = TRUE), come from R's generator):
##
fn_spectral_LQ_ESSR_update_in_cpp <-  function( criterion,
                                               state,
                                               tau,
                                               tau_values,
                                               eps_used_for_trajectories,
                                               eps_now,
                                               probabilities,
                                               divergences,
                                               use_proposals,
                                               learning_rate,
                                               schedule_iteration,
                                               schedule_length,
                                               adam_mean,
                                               adam_variance,
                                               interest_rows,
                                               soft_minimum_power,
                                               minimum_bin_count,
                                               grid_step_in_doublings,
                                               upper_end_zone_in_doublings,
                                               bin_forgetting_factor,
                                               expansion_rule,
                                               ## n_bootstrap_draws) {
                                               n_bootstrap_draws,
                                               ## the threads over which the C++ update spreads its loops over the
                                               ## parameters (and candidates); its results are the same with any
                                               ## number of threads (1 = the serial loops):
                                               n_threads_for_update_loops = 1,
                                               ## at most this many rows fitted and scored per update (the
                                               ## slowest-moving ones; 0 = every row), and whether the bins decay
                                               ## lazily (spectral_LQ_ESSR_update.cpp):
                                               largest_number_of_rows_fitted_per_update = 0,
                                               ## bins_decay_lazily = FALSE) {
                                               bins_decay_lazily = FALSE,
                                               ## the tau move (see fn_spectral_ESS_soft_minimum_tau_update()):
                                               ## the first performed update moves tau all the way to the
                                               ## maximiser (TRUE), and the number of performed updates in a row
                                               ## with the doubling evidence that a doubling needs (1 = each):
                                               SpESS_R_first_update_jumps_to_the_maximiser = FALSE,
                                               ## SpESS_R_doubling_needs_evidence_on_updates_in_a_row = 1) {
                                               SpESS_R_doubling_needs_evidence_on_updates_in_a_row = 1,
                                               ## the variances that the squared jumps are divided by: 0 = the
                                               ## moving averages (decay 0.9) of every update's start moments (as
                                               ## built); n > 0 = the plain means of the start moments of the last
                                               ## n updates' valid chains only
                                               ## (fn_SpESS_R_variances_from_last_n_updates()):
                                               ## SpESS_R_normalising_variances_from_last_n_updates = 0) {
                                               SpESS_R_normalising_variances_from_last_n_updates = 0,
                                               ## the factors on the linear and on the quadratic ESS fractions
                                               ## before the soft minimum (1 = as built; above 1 weighs that part
                                               ## lower; Inf leaves it out):
                                               SpESS_R_linear_ESS_fraction_multiplier = 1,
                                               SpESS_R_quadratic_ESS_fraction_multiplier = 1) {

        rows <-  if (is.null(interest_rows)) seq_len(nrow(criterion$linear_jump_squared)) else interest_rows
        ## global_environment <-  globalenv()
        ## had_seed <-  exists(".Random.seed", envir = global_environment, inherits = FALSE)
        ## if (had_seed) saved_seed <-  get(".Random.seed", envir = global_environment, inherits = FALSE)
        update_function <-  fn_spectral_LQ_ESSR_update_cpp_function()
        ## (every row: the matrices themselves, not copies of them)
        fn_rows_used <-  function(statistic) {

                if (is.null(interest_rows)) statistic else statistic[rows, , drop = FALSE]
        }

        ## (SpESS_R_normalising_variances_from_last_n_updates > 0) the variances of the last n updates' valid
        ## chains, given to the C++ update in place of those of the moving averages, and their window, kept in the
        ## state below:
        variances_from_last_updates <-  NULL
        if (SpESS_R_normalising_variances_from_last_n_updates > 0) {
                variances_from_last_updates <-  fn_SpESS_R_variances_from_last_n_updates(
                      window_in_state        = if (is.list(state)) {
                                                       state$SpESS_R_start_moment_sums_of_last_n_updates
                                               } else NULL,
                      linear_jump_squared    = fn_rows_used(criterion$linear_jump_squared),
                      quadratic_jump_squared = fn_rows_used(criterion$quadratic_jump_squared),
                      initial_second_moment  = fn_rows_used(criterion$initial_second_moment),
                      initial_fourth_moment  = fn_rows_used(criterion$initial_fourth_moment),
                      tau_values             = as.numeric(tau_values),
                      divergences            = as.numeric(divergences),
                      n_updates              = SpESS_R_normalising_variances_from_last_n_updates)
        }
        ## (the variances of the last n updates given, or a factor on an ESS fraction other than 1: the same call
        ##  with the two variances and the two factors added at the end, which needs the development tree's C++
        ##  update, options(NicoStan_spectral_LQ_ESSR_update_cpp = ...); otherwise the call below, unchanged,
        ##  which also runs with the C++ update of the installed build)
        ## if (!is.null(variances_from_last_updates$variance_linear)) {
        if (!is.null(variances_from_last_updates$variance_linear) ||
            SpESS_R_linear_ESS_fraction_multiplier != 1 || SpESS_R_quadratic_ESS_fraction_multiplier != 1) {
                out <-  update_function(state, fn_rows_used(criterion$linear_jump_squared),
                                        fn_rows_used(criterion$quadratic_jump_squared),
                                        fn_rows_used(criterion$initial_second_moment),
                                        fn_rows_used(criterion$initial_fourth_moment),
                                        tau, as.numeric(tau_values),
                                        if (isTRUE(is.finite(eps_used_for_trajectories))) {
                                                eps_used_for_trajectories
                                        } else NA_real_,
                                        eps_now, as.numeric(probabilities), as.numeric(divergences),
                                        isTRUE(use_proposals),
                                        learning_rate, schedule_iteration, schedule_length, soft_minimum_power,
                                        minimum_bin_count, grid_step_in_doublings, upper_end_zone_in_doublings,
                                        bin_forgetting_factor, identical(expansion_rule, "evidence"),
                                        n_bootstrap_draws,
                                        max(1, floor(n_threads_for_update_loops)),
                                        max(0, floor(largest_number_of_rows_fitted_per_update)),
                                        isTRUE(bins_decay_lazily),
                                        variances_from_last_updates$variance_linear,
                                        variances_from_last_updates$variance_quadratic,
                                        SpESS_R_linear_ESS_fraction_multiplier,
                                        SpESS_R_quadratic_ESS_fraction_multiplier)
        } else {
        ## out <-  update_function(state, criterion$linear_jump_squared[rows, , drop = FALSE],
        ##                         criterion$quadratic_jump_squared[rows, , drop = FALSE],
        ##                         criterion$initial_second_moment[rows, , drop = FALSE],
        ##                         criterion$initial_fourth_moment[rows, , drop = FALSE],
        out <-  update_function(state, fn_rows_used(criterion$linear_jump_squared),
                                fn_rows_used(criterion$quadratic_jump_squared),
                                fn_rows_used(criterion$initial_second_moment),
                                fn_rows_used(criterion$initial_fourth_moment),
                                tau, as.numeric(tau_values),
                                if (isTRUE(is.finite(eps_used_for_trajectories))) {
                                        eps_used_for_trajectories
                                } else NA_real_,
                                eps_now, as.numeric(probabilities), as.numeric(divergences),
                                isTRUE(use_proposals),
                                learning_rate, schedule_iteration, schedule_length, soft_minimum_power,
                                minimum_bin_count, grid_step_in_doublings, upper_end_zone_in_doublings,
                                ## bin_forgetting_factor, identical(expansion_rule, "evidence"),
                                ## n_bootstrap_draws)
                                bin_forgetting_factor, identical(expansion_rule, "evidence"), n_bootstrap_draws,
                                max(1, floor(n_threads_for_update_loops)),
                                max(0, floor(largest_number_of_rows_fitted_per_update)),
                                isTRUE(bins_decay_lazily))
        }
        if (!is.null(variances_from_last_updates) && is.list(out$state)) {
                out$state$SpESS_R_start_moment_sums_of_last_n_updates <-  variances_from_last_updates$window
        }
        ## if (isTRUE(out$bootstrap_ran)) {
        ##         if (had_seed) {
        ##                 assign(".Random.seed", saved_seed, envir = global_environment)
        ##         } else if (exists(".Random.seed", envir = global_environment, inherits = FALSE)) {
        ##                 rm(".Random.seed", envir = global_environment)
        ##         }
        ## }
        ##
        ## ---- the options of the tau move (at their defaults nothing below changes the update):
##
        ##      SpESS_R_doubling_needs_evidence_on_updates_in_a_row > 1: a doubling chosen by the C++ update is
        ##      made only when the doubling evidence has held on that many performed updates in a row (the count
        ##      is kept in the state and restarts after each doubling made); otherwise tau moves the usual
        ##      fraction learning_rate_now towards the maximiser, as without the evidence;
        ##      SpESS_R_first_update_jumps_to_the_maximiser = TRUE: the first performed update moves tau all the
        ##      way to the maximiser (whether or not the C++ update chose a doubling).
        doubling_held_for_evidence_in_a_row <-  FALSE
        first_update_jump_made <-  FALSE
        if (isTRUE(out$performed) && SpESS_R_doubling_needs_evidence_on_updates_in_a_row > 1) {
                updates_in_a_row <-  if (is.null(state$SpESS_R_updates_in_a_row_with_doubling_evidence)) 0 else
                                           state$SpESS_R_updates_in_a_row_with_doubling_evidence
                if (isTRUE(out$expanded_towards_doubling)) {
                        updates_in_a_row <-  updates_in_a_row + 1
                        if (updates_in_a_row < SpESS_R_doubling_needs_evidence_on_updates_in_a_row) {
                                out$new_tau <-  exp(log(tau) + out$learning_rate_now *
                                                    (log(out$tau_maximising_criterion) - log(tau)))
                                out$gradient <-  log(out$tau_maximising_criterion) - log(tau)
                                out$expanded_towards_doubling <-  FALSE
                                doubling_held_for_evidence_in_a_row <-  TRUE
                        } else updates_in_a_row <-  0
                } else updates_in_a_row <-  0
                out$state$SpESS_R_updates_in_a_row_with_doubling_evidence <-  updates_in_a_row
        }
        if (isTRUE(out$performed) && isTRUE(SpESS_R_first_update_jumps_to_the_maximiser) &&
            !isTRUE(state$SpESS_R_first_update_jump_made)) {
                out$new_tau <-  out$tau_maximising_criterion
                out$gradient <-  log(out$tau_maximising_criterion) - log(tau)
                first_update_jump_made <-  TRUE
        }
        if (isTRUE(SpESS_R_first_update_jumps_to_the_maximiser)) {
                out$state$SpESS_R_first_update_jump_made <-  isTRUE(state$SpESS_R_first_update_jump_made) ||
                                                             first_update_jump_made
        }
        if (!isTRUE(out$performed)) {
                updated <-  c(tau, adam_mean, adam_variance)
                attr(updated, "adam_update_performed") <-  FALSE
                return(list(updated = updated, adam_update_performed = FALSE, gradient = NA_real_,
                            criterion = NA_real_, criterion_ema = NA_real_, component_criterion_ema = out$state))
        }
        updated <-  c(out$new_tau, adam_mean, adam_variance)
        attr(updated, "adam_update_performed") <-  TRUE
        result <-  list(updated                      = updated,
                        adam_update_performed        = TRUE,
                        gradient                     = out$gradient,
                        criterion                    = out$criterion,
                        criterion_ema                = out$criterion,
                        component_criterion_ema      = out$state,
                        tau_maximising_criterion     = out$tau_maximising_criterion,
                        maximum_is_at_upper_end      = out$maximum_is_at_upper_end,
                        largest_candidate            = out$largest_candidate,
                        learning_rate_now            = out$learning_rate_now,
                        lowest_resolvable_frequency  = out$lowest_resolvable_frequency,
                        acceptance_moving_average    = out$acceptance_moving_average)
        if (identical(expansion_rule, "evidence")) {
                result$expanded_towards_doubling <-  out$expanded_towards_doubling
                result$expansion_evidence_z <-  out$expansion_evidence_z
        }
        if (SpESS_R_doubling_needs_evidence_on_updates_in_a_row > 1) {
                result$doubling_held_for_evidence_in_a_row <-  doubling_held_for_evidence_in_a_row
        }
        if (isTRUE(SpESS_R_first_update_jumps_to_the_maximiser)) {
                result$first_update_jump_made <-  first_update_jump_made
        }
        return(result)

}
##
## ---- (SpESS_R_quadratic_part_from_linear_spectrum_with_pair_modes = TRUE) the quadratic ESS fractions (rows =
##      the monitored rows, columns = the candidate taus) from the LINEAR part's spectral mixture, with its pair
##      modes. For a row z = sum_k b_k x_k of the normal modes x_k (frequencies omega_k) of a normal target, with
##      exact dynamics, a full momentum refresh and state-independent rejections (the linear part's assumptions),
##      the centred square z^2 - E[z^2] is the sum over the PAIRS (j, k) of b_j b_k (x_j x_k - delta_jk); the
##      iteration kernel multiplies each pair's product in expectation by its own lag-one factor
##        rho_jk = (1 - a_bar) + a_bar E_T[cos(omega_j T) cos(omega_k T)]
##               = (1 - a_bar) + a_bar (E_T[cos((omega_j - omega_k) T)] + E_T[cos((omega_j + omega_k) T)]) / 2,
##      and the products are uncorrelated at every lag, so the autocorrelation of z^2 at lag n is
##      sum_{j,k} w_j w_k rho_jk^n and its ESS fraction is
##        1 / sum_{j,k} w_j w_k (1 + rho_jk) / (1 - rho_jk),   w_k = b_k^2 / sum b^2.
##      w and omega are the linear part's weights (summing to 1 per monitored row) and frequencies, E_T is
##      NicoStan's jitter of the candidate tau at eps_now (fn_spectral_ESS_expected_cosine_under_jitter(), as in
##      the linear part, at the difference and at the sum frequencies), a_bar the moving-average acceptance, and
##      rho_jk is capped below 1 as the linear part's lag-one values are. The pair j = k is the normal-mode factor
##      1 - a_bar (1 - E_T[cos(2 omega_k T)]) / 2 of SpESS_R_quadratic_part_normal_mode_spectral_fit (written
##      below as 1 - a_bar (2 - E_T[cos((omega_j - omega_k) T)] - E_T[cos((omega_j + omega_k) T)]) / 2, so that
##      with E_T[cos(0 T)] = 1, exactly, it is computed with the same arithmetic); the pairs j != k are the cross
##      terms that the normal-mode fit of the squares does not model (two equal modes far apart, at a long tau
##      and a_bar = 1: ESS fraction 1/2, against its 1/3). The double sum runs over the frequencies with a
##      non-zero weight in at least one row (a zero weight adds exactly 0 to a row's sum):
##
##
## ---- (SpESS_R_linear_part_measured_persistence_from_last_n_burnin_iterations) each reported quantity's
##      integrated autocorrelation time over the burn-in iterations of a window: window_of_values = a list of
##      matrices, one per burn-in iteration, oldest first (rows = the quantities, columns = the chains; their
##      values at that iteration's trajectory starts). Each quantity's values are centred on their mean over all
##      the chains and iterations, so that slow differences between the chains stay in its variance; its
##      autocorrelations at lags 1, 2, ... are pooled over the chains; and its integrated autocorrelation time is
##      Geyer's initial positive sequence, -1 + 2 (the sum of the pairs of autocorrelations, from lag 0, up to
##      the last positive pair). NA for a quantity with a value that is not finite in the window or no spread;
##      NULL whilst the window holds fewer than 4 iterations:
##
fn_SpESS_R_measured_integrated_autocorrelation_times <-  function(window_of_values) {

        n_iterations <-  length(window_of_values)
        if (n_iterations < 4) return(NULL)
        quantity_names <-  rownames(window_of_values[[1]])
        ## (quantities x chains x iterations)
        values <-  simplify2array(lapply(window_of_values, as.matrix))
        if (length(dim(values)) != 3) return(NULL)
        integrated_autocorrelation_times <-  vapply(seq_len(dim(values)[1]), function(quantity) {

                draws <-  matrix(values[quantity, , ], nrow = dim(values)[2])
                if (any(!is.finite(draws))) return(NA_real_)
                centred <-  draws - mean(draws)
                variance <-  mean(centred^2)
                if (!(variance > 0)) return(NA_real_)
                autocorrelations <-  c(1, vapply(seq_len(n_iterations - 1), function(lag) {

                        mean(centred[, seq_len(n_iterations - lag), drop = FALSE] *
                             centred[, lag + seq_len(n_iterations - lag), drop = FALSE]) / variance
                }, numeric(1)))

                n_pairs <-  floor(length(autocorrelations) / 2)
                pair_sums <-  autocorrelations[2 * seq_len(n_pairs) - 1] + autocorrelations[2 * seq_len(n_pairs)]
                positive <-  cumprod(pair_sums > 0) == 1
                return(-1 + 2 * sum(pair_sums[positive]))
        }, numeric(1))

        names(integrated_autocorrelation_times) <-  quantity_names
        ##
        return(integrated_autocorrelation_times)

}
fn_SpESS_R_quadratic_ESS_fractions_from_linear_spectrum_with_pair_modes <-  function( linear_spectral_weights,
                                                                                    linear_spectral_frequencies,
                                                                                    candidate_tau,
                                                                                    eps_now,
                                                                                    acceptance_moving_average) {

        indices_of_frequencies_with_non_zero_weight_in_some_row <-
              which(rowSums(linear_spectral_weights > 0) > 0)
        weights <-  linear_spectral_weights[indices_of_frequencies_with_non_zero_weight_in_some_row, ,
                                            drop = FALSE]
        frequencies <-  linear_spectral_frequencies[indices_of_frequencies_with_non_zero_weight_in_some_row]
        difference_frequencies <-  outer(frequencies, frequencies, "-")
        sum_frequencies <-  outer(frequencies, frequencies, "+")
        quadratic_ESS_fractions <-  matrix(NA_real_, nrow = ncol(weights), ncol = length(candidate_tau))
        for (candidate_index in seq_along(candidate_tau)) {
                expected_cosines_at_differences <-  fn_spectral_ESS_expected_cosine_under_jitter(
                      c(difference_frequencies), candidate_tau[candidate_index], eps_now)
                expected_cosines_at_sums <-  fn_spectral_ESS_expected_cosine_under_jitter(
                      c(sum_frequencies), candidate_tau[candidate_index], eps_now)
                pair_lag_one <-  pmin(1 - acceptance_moving_average *
                                          (2 - expected_cosines_at_differences - expected_cosines_at_sums) / 2,
                                      1 - 1e-9)
                pair_variance_inflation <-  matrix((1 + pair_lag_one) / (1 - pair_lag_one),
                                                   nrow = length(frequencies))
                quadratic_ESS_fractions[, candidate_index] <-  1 / colSums(weights *
                                                                           (pair_variance_inflation %*% weights))
        }
        return(quadratic_ESS_fractions)

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
                                                      ## objective_mix_ESJD_weight        = 0.5) {
                                                      objective_mix_ESJD_weight        = 0.5,
                                                      ## threads for the C++ update's loops:
                                                      n_threads_for_update_loops       = 1,
                                                      ## the C++ update's limit on the rows fitted per update (0 =
                                                      ## every row) and its lazily decaying bins:
                                                      largest_number_of_rows_fitted_per_update = 0,
                                                      ## bins_decay_lazily                = FALSE) {
                                                      bins_decay_lazily                = FALSE,
                                                      ## the tau move of the C++ update (both at their defaults:
                                                      ## unchanged): see fn_spectral_LQ_ESSR_update_in_cpp():
                                                      SpESS_R_first_update_jumps_to_the_maximiser = FALSE,
                                                   ## SpESS_R_doubling_needs_evidence_on_updates_in_a_row = 1) {
                                                      SpESS_R_doubling_needs_evidence_on_updates_in_a_row = 1,
                                                      ## the variances that the squared jumps are divided by (0 =
                                                      ## the moving averages, as built; n > 0 = the last n
                                                      ## updates' valid chains only): see
                                                      ## fn_SpESS_R_variances_from_last_n_updates():
                                                      ## SpESS_R_normalising_variances_from_last_n_updates = 0) {
                                                      SpESS_R_normalising_variances_from_last_n_updates = 0,
                                                      ## the factors on the linear and on the quadratic ESS
                                                      ## fractions before the soft minimum (1 = as built; above 1
                                                      ## weighs that part lower; Inf leaves it out):
                                                      SpESS_R_linear_ESS_fraction_multiplier = 1,
                                                      ## SpESS_R_quadratic_ESS_fraction_multiplier = 1) {
                                                      SpESS_R_quadratic_ESS_fraction_multiplier = 1,
                                                      ## the quadratic part from the normal-mode spectral fit of
                                                      ## the squares instead of the lag-one bound (FALSE = as
                                                      ## built; TRUE runs the R code of the update): see below:
                                                   ## SpESS_R_quadratic_part_normal_mode_spectral_fit = FALSE) {
                                                      SpESS_R_quadratic_part_normal_mode_spectral_fit = FALSE,
                                                      ## the quadratic part from the linear part's spectral
                                                      ## mixture with its pair modes, in place of the normal-mode
                                                      ## fit of the squares (the R code only; needs the option
                                                      ## above TRUE; FALSE = off, as built): see below and the
                                                      ## function just above this one:
                                          ## (indented less: the argument name is long)
                                          SpESS_R_quadratic_part_from_linear_spectrum_with_pair_modes = FALSE,
                                                      ## the target tau of the update (the R code only): NULL =
                                                      ## off, the candidate that maximises the objective (as
                                                      ## built); a number f above 0 and below 0.5 = the shortest
                                                      ## candidate whose objective is at least (1 - f) times the
                                                      ## largest objective over the candidates: see below:
                                          ## (indented less: the argument name is long)
                                          ## SpESS_R_target_tau_is_shortest_within_fraction_of_best_objective =
                                          ##       NULL) {
                                          SpESS_R_target_tau_is_shortest_within_fraction_of_best_objective =
                                                NULL,
                                                      ## each monitored row's integrated autocorrelation time
                                                      ## measured over the last burn-in iterations (computed by
                                                      ## the burn-in with
                                                      ## fn_SpESS_R_measured_integrated_autocorrelation_times();
                                                      ## NULL = off, as built; the R code only): its excess over
                                                      ## the linear part's own at the current tau is added to the
                                                      ## linear part as motion slower than the explored lengths
                                                      ## resolve; see below:
                                          SpESS_R_measured_integrated_autocorrelation_times = NULL) {

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
        ## (the quadratic part from the linear spectrum with its pair modes takes the place of the normal-mode fit
        ##  of the squares in the R code, so it needs that option)
        if (isTRUE(SpESS_R_quadratic_part_from_linear_spectrum_with_pair_modes) &&
            !isTRUE(SpESS_R_quadratic_part_normal_mode_spectral_fit)) {
                stop(paste0("SpESS_R_quadratic_part_from_linear_spectrum_with_pair_modes = TRUE needs ",
                            "SpESS_R_quadratic_part_normal_mode_spectral_fit = TRUE: the pair-mode quadratic ",
                            "part replaces the normal-mode fit of the squares in SpESS-R's update in R, which ",
                            "that option selects."))
        }
        ## the options that the C++ update implements run in C++ (the R code below otherwise, or with
        ##  options(NicoStan_spectral_LQ_ESSR_update_in_R = TRUE)):
        ## (the normal-mode spectral fit of the squares exists in the R code only)
        ## if (is.null(finite_run_draws) && identical(objective_mix, "none") &&
        ##     !isTRUE(jump_weighted_acceptance) &&
        ##     expansion_rule %in% c("as_built", "evidence") &&
        ##     !isTRUE(SpESS_R_quadratic_part_normal_mode_spectral_fit) &&
        ##     !isTRUE(getOption("NicoStan_spectral_LQ_ESSR_update_in_R", default = FALSE))) {
        ## (the correction of the linear part by the measured integrated autocorrelation times exists in the R
        ##  code only)
        if (is.null(finite_run_draws) && identical(objective_mix, "none") && !isTRUE(jump_weighted_acceptance) &&
            expansion_rule %in% c("as_built", "evidence") &&
            !isTRUE(SpESS_R_quadratic_part_normal_mode_spectral_fit) &&
            is.null(SpESS_R_measured_integrated_autocorrelation_times) &&
            !isTRUE(getOption("NicoStan_spectral_LQ_ESSR_update_in_R", default = FALSE))) {
                ## (the target tau within a fraction of the best objective exists in the R code only, so it stops
                ##  here, where the C++ update would be used)
                if (!is.null(SpESS_R_target_tau_is_shortest_within_fraction_of_best_objective)) {
                        stop(paste0("SpESS_R_target_tau_is_shortest_within_fraction_of_best_objective needs ",
                                    "SpESS_R_quadratic_part_normal_mode_spectral_fit = TRUE: the option acts in ",
                                    "SpESS-R's update in R only, and with ",
                                    "SpESS_R_quadratic_part_normal_mode_spectral_fit = FALSE SpESS-R's C++ ",
                                    "update would be used."))
                }
                return(fn_spectral_LQ_ESSR_update_in_cpp(
                      criterion                   = criterion,
                      state                       = state,
                      tau                         = tau,
                      tau_values                  = tau_values,
                      eps_used_for_trajectories   = eps_used_for_trajectories,
                      eps_now                     = eps_now,
                      probabilities               = probabilities,
                      divergences                 = divergences,
                      use_proposals               = use_proposals,
                      learning_rate               = learning_rate,
                      schedule_iteration          = if (is.null(learning_rate_schedule_iteration)) iteration else
                                                    learning_rate_schedule_iteration,
                      schedule_length             = if (is.null(learning_rate_schedule_length)) {
                                                            adaptation_iterations
                                                    } else learning_rate_schedule_length,
                      adam_mean                   = adam_mean,
                      adam_variance               = adam_variance,
                      interest_rows               = interest_rows,
                      soft_minimum_power          = soft_minimum_power,
                      minimum_bin_count           = minimum_bin_count,
                      grid_step_in_doublings      = grid_step_in_doublings,
                      upper_end_zone_in_doublings = upper_end_zone_in_doublings,
                      bin_forgetting_factor       = bin_forgetting_factor,
                      expansion_rule              = expansion_rule,
                      ## n_bootstrap_draws           = n_bootstrap_draws))
                      n_bootstrap_draws           = n_bootstrap_draws,
                      n_threads_for_update_loops  = n_threads_for_update_loops,
                      largest_number_of_rows_fitted_per_update = largest_number_of_rows_fitted_per_update,
                      ## bins_decay_lazily           = bins_decay_lazily))
                      bins_decay_lazily           = bins_decay_lazily,
                      SpESS_R_first_update_jumps_to_the_maximiser = SpESS_R_first_update_jumps_to_the_maximiser,
                      ## SpESS_R_doubling_needs_evidence_on_updates_in_a_row =
                      ##       SpESS_R_doubling_needs_evidence_on_updates_in_a_row))
                      SpESS_R_doubling_needs_evidence_on_updates_in_a_row =
                            SpESS_R_doubling_needs_evidence_on_updates_in_a_row,
                      ## SpESS_R_normalising_variances_from_last_n_updates =
                      ##       SpESS_R_normalising_variances_from_last_n_updates))
                      SpESS_R_normalising_variances_from_last_n_updates =
                            SpESS_R_normalising_variances_from_last_n_updates,
                      SpESS_R_linear_ESS_fraction_multiplier = SpESS_R_linear_ESS_fraction_multiplier,
                      SpESS_R_quadratic_ESS_fraction_multiplier = SpESS_R_quadratic_ESS_fraction_multiplier))
        }
        ## (the two options of the tau move exist in the C++ update only)
        ## if (isTRUE(SpESS_R_first_update_jumps_to_the_maximiser) ||
        ##     SpESS_R_doubling_needs_evidence_on_updates_in_a_row > 1) {
        ##         stop(paste0("SpESS_R_first_update_jumps_to_the_maximiser and ",
        ##                     "SpESS_R_doubling_needs_evidence_on_updates_in_a_row act in the C++ update only, ",
        ##                     "which this call does not use."))
        ## }
        ## (the first-update jump exists in the C++ update only; the doubling evidence on updates in a row also in
        ##  the R code below, at the doubling)
        if (isTRUE(SpESS_R_first_update_jumps_to_the_maximiser)) {
                stop(paste0("SpESS_R_first_update_jumps_to_the_maximiser acts in the C++ update only, which ",
                            "this call does not use."))
        }
        ## (the R code below needs the state's bin matrices in R; the C++ update may hold them in C++ memory)
        state <-  fn_spectral_LQ_ESSR_state_with_bin_matrices_in_R(state)
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
##
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
        ## (SpESS_R_quadratic_part_normal_mode_spectral_fit) the squares' flow bins, on the same grid of lengths
        ## and frequencies as the flow bins, empty when they are created:
        if (isTRUE(SpESS_R_quadratic_part_normal_mode_spectral_fit) && is.null(state$squares_flow_state)) {
                squares_flow_state <-  state$flow_state
                squares_flow_state$sum_of_normalised_squared_jumps[] <-  0
                squares_flow_state$sum_of_chain_counts[] <-  0
                squares_flow_state$sum_of_executed_lengths[] <-  0
                squares_flow_state$largest_executed_length_in_bin[] <-  0
                squares_flow_state$sum_of_cosines[] <-  0
                squares_flow_state$n_updates <-  0
                state$squares_flow_state <-  squares_flow_state
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
        ## (SpESS_R_normalising_variances_from_last_n_updates > 0) the variances of the last n updates' valid
        ## chains in place of those of the moving averages (which are still kept in the state, as built), and
        ## their window in the state (fn_SpESS_R_variances_from_last_n_updates()):
        if (SpESS_R_normalising_variances_from_last_n_updates > 0) {
                variances_from_last_updates <-  fn_SpESS_R_variances_from_last_n_updates(
                      window_in_state        = state$SpESS_R_start_moment_sums_of_last_n_updates,
                      linear_jump_squared    = criterion$linear_jump_squared,
                      quadratic_jump_squared = criterion$quadratic_jump_squared,
                      initial_second_moment  = criterion$initial_second_moment,
                      initial_fourth_moment  = criterion$initial_fourth_moment,
                      tau_values             = tau_values,
                      divergences            = divergences,
                      n_updates              = SpESS_R_normalising_variances_from_last_n_updates)
                state$SpESS_R_start_moment_sums_of_last_n_updates <-  variances_from_last_updates$window
                if (!is.null(variances_from_last_updates$variance_linear)) {
                        variance_linear <-  variances_from_last_updates$variance_linear
                        variance_quadratic <-  variances_from_last_updates$variance_quadratic
                }
        }
        ##
        ## ---- the quadratic bins (as "LQ_ESSR_length_response": acceptance-weighted, by the jittered length):
##
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
##
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
        ## (SpESS_R_quadratic_part_normal_mode_spectral_fit) the squares' flow bins: the unweighted proposal
        ## jumps of the centred squares of the valid chains divided by Var(z^2) (twice the quadratic bins'
        ## normalisation: for one normal mode of a normal target their mean is 2 sin^2(w t) = 1 - cos(2 w t)), by
        ## the executed length, as the flow bins, with their forgotten sums of squares for the bootstrap
        ## (expansion_rule = "evidence"):
        if (isTRUE(SpESS_R_quadratic_part_normal_mode_spectral_fit)) {
                squares_jumps <-  criterion$quadratic_jump_squared
                squares_jumps[!is.finite(squares_jumps)] <-  0
                state$squares_flow_state <-  fn_spectral_ESS_add_flow_update(
                      flow_state               = state$squares_flow_state,
                      normalised_squared_jumps = squares_jumps / variance_quadratic,
                      executed_lengths         = executed_lengths,
                      flow_chains              = valid,
                      forgetting_factor        = bin_forgetting_factor)
                if (identical(expansion_rule, "evidence")) {
                        if (is.null(state$squares_flow_sum_of_squares)) {
                                state$squares_flow_sum_of_squares <-  0 *
                                      state$squares_flow_state$sum_of_normalised_squared_jumps
                        }
                        if (any(valid)) {
                                state$squares_flow_sum_of_squares <-  fn_spectral_ESS_add_bin_sum_of_squares(
                                      sum_of_squares    = state$squares_flow_sum_of_squares,
                                      values            = squares_jumps / variance_quadratic,
                                      bin_index         = findInterval(
                                            log2(executed_lengths / state$squares_flow_state$reference_length),
                                            state$squares_flow_state$bin_edges_in_doublings) + 1,
                                      chains            = valid,
                                      forgetting_factor = bin_forgetting_factor)
                        }
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
##
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
        ## ---- the same computation as a function of the quadratic bin means and the flow state
        ##      (the estimates, or a bootstrap draw of them), with the round-5 options finite_run_draws and
        ##      objective_mix (the header); with their defaults it returns exactly the objective as before:
        expected_leapfrog_steps <-  vapply(candidate_tau, fn_spectral_ESS_expected_leapfrog_steps_under_jitter,
                                           numeric(1), eps = eps_now)
        ## fn_objective <-  function(quadratic_bin_means, flow_state_used) {
        fn_objective <-  function(quadratic_bin_means, flow_state_used,
                                  squares_flow_state_used = state$squares_flow_state) {

##
                ## ---- the quadratic lag-one scores:
##
                quadratic_lag_one <-  1 - sweep(quadratic_bin_means %*% integral_weights, 2,
                                                2 * candidate_tau, "/")
                quadratic_lag_one <-  pmin(pmax(quadratic_lag_one, -1 + 1e-6), 1 - 1e-6)
                quadratic_scores <-  if (is.null(finite_run_draws)) {
                        (1 - quadratic_lag_one) / (1 + quadratic_lag_one)
                } else 1 / fn_spectral_ESS_finite_run_variance_inflation(quadratic_lag_one, finite_run_draws)
                ## (SpESS_R_quadratic_part_normal_mode_spectral_fit) the normal-mode model of the squares in
                ## place of the lag-one bound: for one normal mode of a normal target the squares' lag-one
                ## autocorrelation under the jitter is E[cos^2(w T)] = (1 + E[cos(2 w T)]) / 2 >= 0, so the
                ## squares' flow bins are fitted by the cosine mixture (frequencies 2 w), each frequency's lag-one
                ## is 1 - a_bar (1 - E[cos(2 w T)]) / 2 and the ESS fraction 1 / sum_k w_k (1 + mu_k) / (1 - mu_k)
                ## (an assumption: pair modes, the cross terms of a row that combines several normal modes, and
                ## the antithetic squares of non-normal targets are not modelled):
                ## if (isTRUE(SpESS_R_quadratic_part_normal_mode_spectral_fit)) {
                ## (with SpESS_R_quadratic_part_from_linear_spectrum_with_pair_modes the squares' own fit is not
                ##  computed: the pair-mode quadratic part below, from the linear fit, takes its place)
                if (isTRUE(SpESS_R_quadratic_part_normal_mode_spectral_fit) &&
                    !isTRUE(SpESS_R_quadratic_part_from_linear_spectrum_with_pair_modes)) {
                        squares_fit <-  fn_spectral_ESS_fit_flow_weights(
                              flow_state        = squares_flow_state_used,
                              eps_now           = eps_now,
                              minimum_bin_count = minimum_bin_count)
                        if (is.null(squares_fit)) return(NULL)
                        squares_cosines <-  matrix(vapply(candidate_tau, function(candidate) {

                                fn_spectral_ESS_expected_cosine_under_jitter(squares_fit$frequencies, candidate,
                                                                             eps_now)
                        }, numeric(length(squares_fit$frequencies))), nrow = length(squares_fit$frequencies))

                        squares_lag_one <-  pmin(1 - state$acceptance_moving_average * (1 - squares_cosines) / 2,
                                                 1 - 1e-9)
                        quadratic_scores <-  1 / crossprod(squares_fit$weights,
                                                           (1 + squares_lag_one) / (1 - squares_lag_one))
                }
##
                ## ---- the spectral linear scores (the flow's fitted cosine mixture, with the laziness a_bar):
##
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
                ## (SpESS_R_measured_integrated_autocorrelation_times) each row's integrated autocorrelation time
                ## measured over the last burn-in iterations against the linear part's own at the candidate
                ## nearest the current tau: the excess is motion that the flow's fit does not hold (slower than
                ## the explored lengths resolve, or between the chains), whose integrated autocorrelation time
                ## scales as (tau / tau')^2 (a normal mode far below its frequency: 1 - lambda ~ (w tau')^2), and
                ## is added to the linear part's at every candidate tau':
                if (!is.null(SpESS_R_measured_integrated_autocorrelation_times)) {
                        candidate_nearest_current_tau <-  which.min(abs(log(candidate_tau) - log(tau)))
                        excess_integrated_autocorrelation_times <-
                              SpESS_R_measured_integrated_autocorrelation_times -
                              1 / linear_scores[, candidate_nearest_current_tau]
                        excess_integrated_autocorrelation_times[
                              !is.finite(excess_integrated_autocorrelation_times)] <-  0
                        excess_integrated_autocorrelation_times <-
                              pmax(excess_integrated_autocorrelation_times, 0)
                        linear_scores <-  1 / (1 / linear_scores +
                                               outer(excess_integrated_autocorrelation_times,
                                                     (tau / candidate_tau)^2))
                }
                ## (SpESS_R_quadratic_part_from_linear_spectrum_with_pair_modes) the quadratic part from this
                ## linear fit's weights and frequencies with the pair modes (the function above
                ## fn_spectral_ESS_soft_minimum_tau_update()), in place of the normal-mode fit of the squares; in
                ## the bootstrap draws of the doubling evidence the linear fit is that of the perturbed flow bins,
                ## so the pair-mode part uses the perturbed linear weights:
                if (isTRUE(SpESS_R_quadratic_part_from_linear_spectrum_with_pair_modes)) {
                        quadratic_scores <-
                              fn_SpESS_R_quadratic_ESS_fractions_from_linear_spectrum_with_pair_modes(
                                    linear_spectral_weights     = fit$weights,
                                    linear_spectral_frequencies = fit$frequencies,
                                    candidate_tau               = candidate_tau,
                                    eps_now                     = eps_now,
                                    acceptance_moving_average   = state$acceptance_moving_average)
                }
                ## (the factors on the two ESS fractions; 1 = as built, the same numbers; Inf leaves a part out)
                if (SpESS_R_linear_ESS_fraction_multiplier != 1) {
                        linear_scores <-  SpESS_R_linear_ESS_fraction_multiplier * linear_scores
                }
                if (SpESS_R_quadratic_ESS_fraction_multiplier != 1) {
                        quadratic_scores <-  SpESS_R_quadratic_ESS_fraction_multiplier * quadratic_scores
                }
##
                ## ---- the soft minimum per expected gradient:
##
                scores <-  rbind(linear_scores, quadratic_scores)
                smallest_scores <-  apply(scores, 2, min)
                relative_inverse_scores <-  sweep(1 / scores, 2, 1 / smallest_scores, "/")
                soft_minimum <-  smallest_scores *
                                 colSums(relative_inverse_scores^soft_minimum_power)^(-1 / soft_minimum_power)
                objective <-  soft_minimum / expected_leapfrog_steps
##
                ## ---- objective_mix = "geometric_mean_with_ESJD": the geometric mean with the ESJD objective:
##
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
                        ## the ESJD weight objective_mix_ESJD_weight; 0.5 = the equal-weight geometric mean above:
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
        ## (SpESS_R_target_tau_is_shortest_within_fraction_of_best_objective = f set: the target is the shortest
        ##  candidate whose objective is at least (1 - f) times the largest objective over the candidates, on the
        ##  grid of candidates evaluated above (grid_step_in_doublings apart, with no refinement between them).
        ##  It takes the place of the maximiser everywhere below: the upper-end rule and the doubling (so
        ##  maximum_is_at_upper_end is then whether the target is at the upper end), the tau move, and the
        ##  tau_maximising_criterion and criterion returned; the maximiser and the largest objective are also
        ##  returned, as tau_maximising_criterion_before_shortest_within_fraction and
        ##  largest_criterion_over_candidates. The objective is flat near its maximum, where its noise moves the
        ##  maximiser; NULL: the maximiser, as built)
        if (!is.null(SpESS_R_target_tau_is_shortest_within_fraction_of_best_objective)) {
                index_of_maximiser_before_shortest_within_fraction <-  best
                candidates_within_fraction_of_best_objective <-  which(
                      criterion_per_candidate >=
                            (1 - SpESS_R_target_tau_is_shortest_within_fraction_of_best_objective) *
                            criterion_per_candidate[index_of_maximiser_before_shortest_within_fraction])
                if (length(candidates_within_fraction_of_best_objective) > 0) {
                        best <-  candidates_within_fraction_of_best_objective[
                              which.min(candidate_tau[candidates_within_fraction_of_best_objective])]
                }
        }
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
        ## expansion_rule = "as_built": every maximiser at the upper end targets
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
        ## (SpESS_R_doubling_needs_evidence_on_updates_in_a_row > 1, as in the C++ update's wrapper: a doubling is
        ##  made only when the doubling evidence has held on that many performed updates in a row, the count kept
        ##  in the state and restarted after each doubling made; otherwise tau moves the usual fraction
        ##  learning_rate_now towards the maximiser)
        doubling_held_for_evidence_in_a_row <-  FALSE
        if (SpESS_R_doubling_needs_evidence_on_updates_in_a_row > 1) {
                updates_in_a_row <-  if (is.null(state$SpESS_R_updates_in_a_row_with_doubling_evidence)) 0 else
                                           state$SpESS_R_updates_in_a_row_with_doubling_evidence
                if (expand) {
                        updates_in_a_row <-  updates_in_a_row + 1
                        if (updates_in_a_row < SpESS_R_doubling_needs_evidence_on_updates_in_a_row) {
                                expand <-  FALSE
                                doubling_held_for_evidence_in_a_row <-  TRUE
                        } else updates_in_a_row <-  0
                } else updates_in_a_row <-  0
                state$SpESS_R_updates_in_a_row_with_doubling_evidence <-  updates_in_a_row
        }
        target_tau <-  if (expand) 2 * max(tau, largest_candidate) else candidate_tau[best]
        new_tau <-  exp(log(tau) + learning_rate_now * (log(target_tau) - log(tau)))
        ##
        updated <-  c(new_tau, adam_mean, adam_variance)
        attr(updated, "adam_update_performed") <-  TRUE
        ## (SpESS_R_target_tau_is_shortest_within_fraction_of_best_objective set: the same result as the two
        ##  below, with the maximiser before the rule and the largest objective over the candidates added; NULL:
        ##  the two below, unchanged)
        if (!is.null(SpESS_R_target_tau_is_shortest_within_fraction_of_best_objective)) {
                result <-  list(updated                      = updated,
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
                                acceptance_moving_average    = state$acceptance_moving_average)
                if (identical(expansion_rule, "evidence")) {
                        result$expanded_towards_doubling <-  expand
                        result$expansion_evidence_z <-  evidence_z
                        result$doubling_held_for_evidence_in_a_row <-  doubling_held_for_evidence_in_a_row
                }
                result$tau_maximising_criterion_before_shortest_within_fraction <-
                      candidate_tau[index_of_maximiser_before_shortest_within_fraction]
                result$largest_criterion_over_candidates <-
                      criterion_per_candidate[index_of_maximiser_before_shortest_within_fraction]
                return(result)
        }
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
                            ## expansion_evidence_z         = evidence_z))
                            expansion_evidence_z         = evidence_z,
                            doubling_held_for_evidence_in_a_row =
                                  doubling_held_for_evidence_in_a_row))
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




















