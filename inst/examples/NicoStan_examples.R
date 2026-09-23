##
## =====================================================================================================================================
## NicoStan examples
##
## Source this file to define the functions; no fits run on sourcing.
## The smoke profile checks execution. The analysis profile is a starting configuration, not a convergence guarantee.
##
.NicoStan_examples_source_files <-  as.character(unlist(lapply(sys.frames(), function(frame) frame$ofile)))
.NicoStan_examples_source_directory <-  if (length(.NicoStan_examples_source_files)) {
        dirname(normalizePath(tail(.NicoStan_examples_source_files, 1L), mustWork = TRUE))
} else system.file("examples", package = "NicoStan")
##
NicoStan_example_registry <-  function() {
        data.frame(
            model = c("cox_frailty", "weibull", "robust_t4", "hierarchical_logistic", "joint_longitudinal_survival", "gaussian_process", "stochastic_volatility",
                      "latent_diffusion_survival"),
            file = c("Frailty_surv_shared_frailty_Cox.stan", "standard_surv_Weibull.stan", "robust_regression_t4.stan",
                     "hierarchical_logistic_regression.stan", "Joint_longitudinal_surv.stan", "Gaussian_process.stan", "Stochastic_volatility.stan",
                     "Latent_diffusion_survival.stan"),
            diffusion_eligible = c(TRUE, FALSE, FALSE, TRUE, TRUE, FALSE, TRUE, TRUE),
            nuisance = c("log_frailty_raw[J]", "none", "none", "beta_raw[K,2]", "b_raw[2,N_subj]", "none: latent GP integrated out", "h_std[T]",
                         "w[M]: standard normal Brownian increments"),
            stringsAsFactors = FALSE)
}
##
NicoStan_quadrature <-  function(n_points = 15L) {
        index <-  seq_len(n_points - 1L)
        off_diagonal <-  index / sqrt(4 * index^2 - 1)
        jacobi <-  matrix(0, nrow = n_points, ncol = n_points)
        jacobi[cbind(index, index + 1L)] <-  off_diagonal
        jacobi[cbind(index + 1L, index)] <-  off_diagonal
        eig <-  eigen(jacobi, symmetric = TRUE)
        order_points <-  order(eig$values)
        list(points = eig$values[order_points], weights = 2 * eig$vectors[1L, order_points]^2)
}
##
## ---- Latent diffusion survival simulator (Beskos, Kalogeropoulos and Pazos, 2013, Section 6.3):
##
## One latent path X is shared by every subject. The hazard is h(u) = hazard_scale * (X(u)^2 + x_squared_hazard_offset)
## and dX_u = -(drift_sin_coefficient * sin(X_u) + drift_constant) du + sigma_x dB_u, X_0 = x_0.
## The defaults are their eq. (51) on the window [0, 1] shown in their Figure 1, with h(x) = x^2 + 0.01.
##
## x_squared_hazard_offset (round-4 change, 2026-09-22, assistant-introduced; not in Beskos et al., who use 0):
## with h(x) = x^2 exactly, log h(t) = 2 log|x(t)| is -Inf at x(t) = 0 and h(x) = h(-x), so once the path nears
## zero the posterior splits into sign patterns separated by -Inf walls at the event times, and HMC diverges
## there (cmdstanr: divergences and x_path R-hat 1.5 even with every parameter fixed at the Beskos et al. values).
## The offset keeps log h finite. The data are generated with the same offset, so the model is correctly specified.
## The true path is simulated by Euler-Maruyama on a grid n_fine_steps_per_model_step times finer
## than the M-step grid used by the Stan model, and event times are drawn exactly given that fine
## path, by inverting its trapezoid cumulative hazard at independent Exp(1) thresholds. Subjects
## whose threshold exceeds H(t_max) are right censored at t_max.
##
fn_simulate_latent_diffusion_survival_data <- function( N_subjects,
                                                        M_model_steps                = 50,
                                                        t_max                        = 1,
                                                        x_0                          = 2,
                                                        sigma_x                      = 1,
                                                        true_drift_sin_coefficient   = 1.4,
                                                        true_drift_constant          = 1,
                                                        true_hazard_scale            = 1,
                                                        x_squared_hazard_offset      = 0.01,
                                                        n_fine_steps_per_model_step  = 20) {
        {
                stopifnot(length(N_subjects) == 1,
                          N_subjects >= 1,
                          N_subjects == round(N_subjects),
                          length(M_model_steps) == 1,
                          M_model_steps >= 2,
                          M_model_steps == round(M_model_steps),
                          t_max > 0,
                          sigma_x > 0,
                          true_hazard_scale > 0,
                          length(x_squared_hazard_offset) == 1,
                          is.finite(x_squared_hazard_offset),
                          x_squared_hazard_offset >= 0,
                          n_fine_steps_per_model_step >= 1,
                          n_fine_steps_per_model_step == round(n_fine_steps_per_model_step))
        }
        ##
        ## ---- true latent path on the fine grid:
        ##
        {
                n_fine_steps <- M_model_steps * n_fine_steps_per_model_step
                fine_step_length <- t_max / n_fine_steps
                fine_grid_times <- seq(from = 0, to = t_max, length.out = n_fine_steps + 1)
                fine_brownian_increments <- rnorm(n_fine_steps)
                true_x_path_fine <- numeric(n_fine_steps + 1)
                true_x_path_fine[1] <- x_0
                for (fine_step_index in seq_len(n_fine_steps)) {
                        current_x_value <- true_x_path_fine[fine_step_index]
                        drift_value <- -(true_drift_sin_coefficient * sin(current_x_value) + true_drift_constant)
                        true_x_path_fine[fine_step_index + 1] <- current_x_value +
                                                                 drift_value * fine_step_length +
                                                                 sigma_x * sqrt(fine_step_length) * fine_brownian_increments[fine_step_index]
                }
        }
        ##
        ## ---- exact event times given the fine path (piecewise-linear hazard between fine grid points):
        ##
        {
                true_hazard_fine <- true_hazard_scale * (true_x_path_fine^2 + x_squared_hazard_offset)
                true_cumulative_hazard_fine <- c(0, cumsum(0.5 * fine_step_length * (head(true_hazard_fine, -1) + tail(true_hazard_fine, -1))))
                exponential_thresholds <- rexp(N_subjects)
                event_indicator <- as.integer(exponential_thresholds <= true_cumulative_hazard_fine[n_fine_steps + 1])
                observed_times <- rep(t_max, N_subjects)
                for (subject_index in which(event_indicator == 1)) {
                        ##
                        ## Within the fine cell, H is quadratic in the time offset when h is linear;
                        ## solve H(lower) + s * h_lower + 0.5 * s^2 * slope = threshold for s.
                        ##
                        threshold_value <- exponential_thresholds[subject_index]
                        lower_fine_index <- max(1, findInterval(threshold_value, true_cumulative_hazard_fine, left.open = TRUE))
                        lower_fine_index <- min(lower_fine_index, n_fine_steps)
                        remaining_cumulative_hazard <- threshold_value - true_cumulative_hazard_fine[lower_fine_index]
                        hazard_at_lower <- true_hazard_fine[lower_fine_index]
                        hazard_slope <- (true_hazard_fine[lower_fine_index + 1] - hazard_at_lower) / fine_step_length
                        ##
                        ## Stable form of the smaller positive root; it also covers a zero slope.
                        ##
                        root_denominator <- hazard_at_lower + sqrt(max(0, hazard_at_lower^2 + 2 * hazard_slope * remaining_cumulative_hazard))
                        time_offset <- if (root_denominator > 0) 2 * remaining_cumulative_hazard / root_denominator else 0
                        observed_times[subject_index] <- min(t_max, max(fine_grid_times[lower_fine_index] + time_offset, 1e-10))
                }
        }
        ##
        ## ---- Stan data list (names match Latent_diffusion_survival.stan):
        ##
        {
                stan_data_list <- list(N                       = as.integer(N_subjects),
                                       M                       = as.integer(M_model_steps),
                                       t_max                   = t_max,
                                       x_0                     = x_0,
                                       sigma_x                 = sigma_x,
                                       x_squared_hazard_offset = x_squared_hazard_offset,
                                       t                       = observed_times,
                                       event                   = event_indicator)
        }
        list(stan_data_list              = stan_data_list,
             true_x_path_fine            = true_x_path_fine,
             fine_grid_times             = fine_grid_times,
             true_cumulative_hazard_fine = true_cumulative_hazard_fine,
             true_parameter_values       = c(drift_sin_coefficient = true_drift_sin_coefficient,
                                             drift_constant        = true_drift_constant,
                                             hazard_scale          = true_hazard_scale))
}
##
make_NicoStan_example <-  function( model,
                                    profile = c("smoke", "analysis"),
                                    seed = 2026L,
                                    chains = 4L,
                                    N = NULL) {
        profile <-  match.arg(profile)
        registry <-  NicoStan_example_registry()
        stopifnot(length(model) == 1L, model %in% registry$model, chains >= 2L)
        set.seed(seed)
        small <-  profile == "smoke"
        if (!is.null(N)) stopifnot(length(N) == 1L, is.finite(N), N >= 12L, N == as.integer(N))
        size <-  function(default) if (is.null(N)) as.integer(default) else as.integer(N)
        n_nuisance <-  0L
        ##
        if (model == "cox_frailty") {
                n <-  size(if (small) 64L else 600L)
                n_groups <-  if (is.null(N)) if (small) 8L else 30L else max(4L, as.integer(ceiling(n / 8)))
                group <-  rep(seq_len(n_groups), length.out = n)
                design <-  matrix(rnorm(n * 2L), nrow = n)
                beta <-  c(0.5, -0.3)
                latent <-  rnorm(n_groups)
                hazard <-  0.15 * exp(c(design %*% beta) + 0.5 * latent[group])
                event_time <-  rexp(n, rate = hazard)
                censor_time <-  runif(n, min = 2, max = 8)
                observed_time <-  pmin(event_time, censor_time)
                data <-  list(N = n, K = 2L, J = n_groups, X = design, t = observed_time,
                              event = as.integer(event_time <= censor_time), group = group,
                              at_risk_count = vapply(observed_time, function(t) sum(observed_time >= t), integer(1L)))
                initial <-  list(log_frailty_raw = rep(0, n_groups), beta = c(0, 0), sigma_frailty = 0.5)
                main <-  c("beta", "sigma_frailty")
                latent_name <-  "log_frailty_raw"
                n_nuisance <-  n_groups
        } else if (model == "weibull") {
                n <-  size(if (small) 64L else 600L)
                design <-  matrix(rnorm(n * 2L), nrow = n)
                shape <-  1.4
                scale <-  exp(-(-1 + c(design %*% c(0.4, -0.25))) / shape)
                event_time <-  rweibull(n, shape = shape, scale = scale)
                censor_time <-  runif(n, min = 1, max = 4)
                observed <-  event_time <= censor_time
                data <-  list(N = sum(observed), N_cens = sum(!observed), K = 2L,
                              t = event_time[observed], t_cens = censor_time[!observed],
                              X = design[observed, , drop = FALSE], X_cens = design[!observed, , drop = FALSE])
                initial <-  list(alpha = 1.2, beta = c(0, 0), mu = -0.8)
                main <-  c("alpha", "beta", "mu")
        } else if (model == "robust_t4") {
                n <-  size(if (small) 64L else 600L)
                design <-  matrix(rnorm(n * 2L), nrow = n)
                data <-  list(N = n, K = 2L, X = design,
                              y = 0.3 + c(design %*% c(0.7, -0.4)) + 0.8 * rt(n, df = 4), nu = 4)
                initial <-  list(alpha = 0, beta = c(0, 0), sigma = 1)
                main <-  c("alpha", "beta", "sigma")
        } else if (model == "hierarchical_logistic") {
                n <-  size(if (small) 96L else 1200L)
                n_groups <-  if (is.null(N)) if (small) 8L else 40L else max(4L, as.integer(ceiling(n / 12)))
                group <-  rep(seq_len(n_groups), length.out = n)
                x <-  rnorm(n)
                coefficients <-  cbind(rnorm(n_groups, -0.3, 0.6), rnorm(n_groups, 0.7, 0.4))
                probability <-  plogis(coefficients[group, 1L] + coefficients[group, 2L] * x)
                data <-  list(N = n, K = n_groups, group = group, x = x, y = rbinom(n, size = 1L, prob = probability))
                initial <-  list(beta_raw = matrix(0, nrow = n_groups, ncol = 2L), mu = c(0, 0), sigma = c(0.6, 0.4))
                main <-  c("mu", "sigma")
                latent_name <-  "beta_raw"
                n_nuisance <-  2L * n_groups
        } else if (model == "stochastic_volatility") {
                n <-  size(if (small) 60L else 500L)
                innovations <-  rnorm(n)
                h <-  numeric(n)
                h[1L] <-  -1 + 0.25 * innovations[1L] / sqrt(1 - 0.85^2)
                for (i in 2:n) h[i] <-  -1 + 0.85 * (h[i - 1L] + 1) + 0.25 * innovations[i]
                data <-  list(T = n, y = rnorm(n, sd = exp(h / 2)))
                initial <-  list(h_std = rep(0, n), mu = log(mean(data$y^2)), phi = 0.7, sigma = 0.3)
                main <-  c("mu", "phi", "sigma")
                latent_name <-  "h_std"
                n_nuisance <-  n
        } else if (model == "latent_diffusion_survival") {
                ##
                ## ---- latent diffusion survival (Beskos, Kalogeropoulos and Pazos, 2013, Section 6.3):
                ##
                ## N counts subjects, all sharing one latent hazard path. The Stan grid is fixed at
                ## M = 50 Euler-Maruyama steps on [0, 1] (dt = 0.02, the only step Beskos et al. report, which is for
                ## their diffusion-bridge experiment; they give none for the survival one), so the nuisance dimension is 50 for every N.
                ##
                n <-  size(if (small) 64 else 200)
                simulated_latent_diffusion_survival <-  fn_simulate_latent_diffusion_survival_data(N_subjects              = n,
                                                                                                   M_model_steps           = 50,
                                                                                                   x_squared_hazard_offset = 0.01)
                data <-  simulated_latent_diffusion_survival$stan_data_list
                if (!identical(data$x_squared_hazard_offset, 0.01)) stop("latent_diffusion_survival: x_squared_hazard_offset did not reach the Stan data list.")
                initial <-  list(w                     = rep(0, data$M),
                                 drift_sin_coefficient = 1,
                                 drift_constant        = 0.5,
                                 hazard_scale          = 0.8)
                main <-  c("drift_sin_coefficient", "drift_constant", "hazard_scale")
                latent_name <-  "w"
                n_nuisance <-  data$M
        } else if (model == "gaussian_process") {
                n <-  size(if (small) 24L else 100L)
                x <-  seq(-2, 2, length.out = n)
                data <-  list(N = n, x = x, y = sin(1.5 * x) + rnorm(n, sd = 0.25),
                              N_pred = 12L, x_pred = seq(-2.2, 2.2, length.out = 12L))
                initial <-  list(lengthscale = 1, sigma_f = 0.8, sigma_n = 0.3)
                main <-  c("lengthscale", "sigma_f", "sigma_n")
        } else if (model == "joint_longitudinal_survival") {
                n_subjects <-  size(if (small) 12L else 100L)
                q <-  NicoStan_quadrature()
                x_surv <-  matrix(rnorm(n_subjects), ncol = 1L)
                raw <-  matrix(rnorm(2L * n_subjects), nrow = 2L)
                b0 <-  0.5 * raw[1L, ]
                b1 <-  0.2 * (0.25 * raw[1L, ] + sqrt(1 - 0.25^2) * raw[2L, ])
                observed_time <-  numeric(n_subjects)
                event <-  integer(n_subjects)
                longitudinal <-  vector("list", n_subjects)
                for (i in seq_len(n_subjects)) {
                        rate <-  0.2 * exp(0.3 * x_surv[i, 1L] + 0.4 * (0.5 + b0[i]))
                        slope <-  0.4 * (0.2 + b1[i])
                        cumulative_hazard <-  function(t) {
                                if (abs(slope) < 1e-8) rate * t else rate * expm1(slope * t) / slope
                        }
                        followup <-  runif(1L, min = 3, max = 5)
                        threshold <-  rexp(1L)
                        event[i] <-  as.integer(threshold <= cumulative_hazard(followup))
                        observed_time[i] <-  if (event[i]) uniroot(function(t) cumulative_hazard(t) - threshold,
                                                                 interval = c(0, followup), tol = 1e-10)$root else followup
                        times <-  seq(0, observed_time[i], by = 0.5)
                        longitudinal[[i]] <-  data.frame(subject = i, time = times,
                            y = 0.5 + 0.2 * times + b0[i] + b1[i] * times + rnorm(length(times), sd = 0.3))
                }
                long <-  do.call(rbind, longitudinal)
                designs_quad <-  lapply(observed_time, function(t) cbind(1, t * (q$points + 1) / 2))
                data <-  list(N_long = nrow(long), N_subj = n_subjects, K_long = 2L,
                              X_long = cbind(1, long$time), Y_long = long$y, subj_long = long$subject, time_long = long$time,
                              K_surv = 1L, X_surv = x_surv, T_surv = observed_time, event = event,
                              Q = length(q$points), quad_points = q$points, quad_weights = q$weights,
                              X_long_event = cbind(1, observed_time), X_long_quad = designs_quad)
                initial <-  list(b_raw = matrix(0, nrow = 2L, ncol = n_subjects), beta_long = c(0.5, 0.2),
                                 sigma_long = 0.4, sigma_b0 = 0.5, sigma_b1 = 0.25, rho_b = 0,
                                 beta_surv = array(0, dim = 1L), lambda = 0.2, alpha_assoc = 0.2)
                main <-  c("beta_long", "sigma_long", "sigma_b0", "sigma_b1", "rho_b", "beta_surv", "lambda", "alpha_assoc")
                latent_name <-  "b_raw"
                n_nuisance <-  2L * n_subjects
        }
        ## Identical simulated data and initial values are reconstructed for each engine.
        initial_values <-  lapply(seq_len(chains), function(chain) {
                value <-  initial
                if (n_nuisance > 0L) value[[latent_name]][] <-  rnorm(n_nuisance, sd = 0.05)
                value
        })
        list(model = model, data = data, initial_values = initial_values, main_parameters = main,
             n_nuisance = n_nuisance, diffusion_eligible = n_nuisance > 0L,
             N = if (model == "joint_longitudinal_survival") n_subjects else n,
             N_unit = if (model %in% c("joint_longitudinal_survival", "latent_diffusion_survival")) "subjects" else if (model == "stochastic_volatility") "time points" else "observations",
             source_file = registry$file[match(model, registry$model)], seed = seed, profile = profile)
}
##
NicoStan_check_gradient <-  function(model, example, expected_simd_lanes = NULL) {
        bs <-  bridgestan::StanModel$new(lib = model$init_object$model_so_file,
                                        data = model$init_object$json_file_path, seed = example$seed, warn = FALSE)
        density_environment <-  new.env(parent = environment())
        sys.source(file.path(.NicoStan_examples_source_directory, "check_model_densities.R"), envir = density_environment)
        density_error <-  density_environment$NicoStan_check_density(bs, example, tolerance = if (is.null(expected_simd_lanes)) 1e-7 else 1e-6)
        observed_simd_lanes <-  NULL
        unconstrained <-  bs$param_unconstrain_json(jsonlite::toJSON(example$initial_values[[1L]], auto_unbox = TRUE, digits = NA))
        if (!is.null(expected_simd_lanes)) {
                names_with_gq <-  unlist(bs$param_names(include_tp = FALSE, include_gq = TRUE))
                generated <-  bs$param_constrain(unconstrained, include_tp = FALSE, include_gq = TRUE, rng = bs$new_rng(example$seed))
                observed_simd_lanes <-  unname(generated[match("simd_lanes", names_with_gq)])
                stopifnot(identical(as.numeric(observed_simd_lanes), as.numeric(expected_simd_lanes)))
        }
        ## The first declared parameter must be the complete standard normal nuisance block.
        expected_latent_name <-  switch(example$model, cox_frailty = "log_frailty_raw", hierarchical_logistic = "beta_raw",
                                       joint_longitudinal_survival = "b_raw", stochastic_volatility = "h_std",
                                       latent_diffusion_survival = "w", NULL)
        if (example$n_nuisance > 0L) {
                names <-  unlist(bs$param_names(include_tp = FALSE, include_gq = FALSE))
                first_names <-  names[seq_len(example$n_nuisance)]
                stopifnot(all(startsWith(first_names, paste0(expected_latent_name, "."))) ||
                          all(startsWith(first_names, paste0(expected_latent_name, "["))))
        }
        gradient <-  bs$log_density_gradient(unconstrained)$gradient
        native_args <-  model$init_object$Model_args_as_Rcpp_List
        n_us <-  example$n_nuisance
        native_gradient <-  get("Rcpp_wrapper_fn_lp_grad", envir = asNamespace("NicoStan"))
        native <-  native_gradient(Model_type = "Stan", force_autodiff = FALSE,
            force_PartialLog = FALSE, multi_attempts = FALSE,
            theta_main_vec = matrix(unconstrained[seq.int(n_us + 1L, length(unconstrained))], ncol = 1L),
            theta_us_vec = matrix(unconstrained[seq_len(n_us)], ncol = 1L),
            y = matrix(0L, nrow = native_args$N, ncol = native_args$n_tests),
            grad_option = "all", Model_args_as_Rcpp_List = native_args)
        step <-  1e-5
        finite_difference <-  vapply(seq_along(unconstrained), function(index) {
                upper <-  lower <-  unconstrained
                upper[index] <-  upper[index] + step
                lower[index] <-  lower[index] - step
                (bs$log_density(upper) - bs$log_density(lower)) / (2 * step)
        }, numeric(1L))
        native_error <-  max(abs(c(native[1L + seq_along(unconstrained)]) - gradient))
        difference_error <-  max(abs(finite_difference - gradient) / pmax(1, abs(gradient)))
        stopifnot(all(is.finite(gradient)), native_error < 1e-8, difference_error < 1e-4)
        list(native_max_absolute_error = native_error, finite_difference_max_scaled_error = difference_error,
             density_max_absolute_error = density_error, n_checked = length(unconstrained), simd_lanes = observed_simd_lanes)
}
## Structural checks are kept separate from the numerical gradient check so
## benchmark runs can still verify automatic nuisance detection and the AVX
## generated quantity without paying for an independent finite-difference pass
## for every arm and sample size.
NicoStan_check_model_structure <- function(model, example, expected_simd_lanes = NULL) {
        init <- model$init_object
        stopifnot(identical(as.integer(init$n_nuisance), as.integer(example$n_nuisance)))
        bs <- init$bs_model
        unconstrained_names <- as.character(bs$param_unc_names())
        stopifnot(length(unconstrained_names) == as.integer(init$n_nuisance + init$n_params_main))
        expected_latent_name <- switch(example$model, cox_frailty = "log_frailty_raw", hierarchical_logistic = "beta_raw",
                                       joint_longitudinal_survival = "b_raw", stochastic_volatility = "h_std",
                                       latent_diffusion_survival = "w", NULL)
        if (example$n_nuisance > 0L) {
                unconstrained_bases <- sub("\\..*$", "", unconstrained_names[seq_len(example$n_nuisance)])
                stopifnot(all(unconstrained_bases == expected_latent_name))
        }
        observed_simd_lanes <- NULL
        if (!is.null(expected_simd_lanes)) {
                names_with_gq <- as.character(bs$param_names(include_tp = FALSE, include_gq = TRUE))
                unconstrained <- bs$param_unconstrain_json(jsonlite::toJSON(example$initial_values[[1L]], auto_unbox = TRUE, digits = NA))
                generated <- bs$param_constrain(unconstrained, include_tp = FALSE, include_gq = TRUE,
                                                rng = bs$new_rng(example$seed))
                observed_simd_lanes <- unname(generated[match("simd_lanes", names_with_gq)])
                stopifnot(length(observed_simd_lanes) == 1L,
                          identical(as.numeric(observed_simd_lanes), as.numeric(expected_simd_lanes)))
        }
        list(n_nuisance = as.integer(init$n_nuisance), n_params_main = as.integer(init$n_params_main),
             n_params = as.integer(init$n_params), simd_lanes = observed_simd_lanes)
}
##
NicoStan_compare_avx <-  function(avx_model, example, plain_file) {
        plain_library <-  bridgestan::compile_model(plain_file, make_args = c("STAN_THREADS=true", "PRECOMPILED_HEADERS=false"))
        plain <-  bridgestan::StanModel$new(plain_library, data = avx_model$init_object$json_file_path, seed = example$seed, warn = FALSE)
        avx <-  bridgestan::StanModel$new(avx_model$init_object$model_so_file, data = avx_model$init_object$json_file_path,
                                         seed = example$seed, warn = FALSE)
        u <-  plain$param_unconstrain_json(jsonlite::toJSON(example$initial_values[[1L]], auto_unbox = TRUE, digits = NA))
        set.seed(example$seed + 41L)
        points <-  c(list(u), lapply(seq_len(5L), function(i) u + rnorm(length(u), sd = 0.3)))
        errors <-  do.call(rbind, lapply(points, function(point) {
                a <-  avx$log_density_gradient(point, propto = FALSE, jacobian = TRUE)
                b <-  plain$log_density_gradient(point, propto = FALSE, jacobian = TRUE)
                c(log_density_absolute_error = abs(a$val - b$val),
                  gradient_max_scaled_error = max(abs(a$gradient - b$gradient) / pmax(1, abs(b$gradient))))
        }))
        stopifnot(all(is.finite(errors)), max(errors[, 1L]) < 1e-6, max(errors[, 2L]) < 1e-6)
        errors
}
##
run_NicoStan_example <-  function( model,
                                   engine = c("NicoStan", "cmdstanr"),
                                   math_backend = c("Stan", "AVX512", "AVX2"),
                                   profile = c("smoke", "analysis"),
                                   burnin_algorithm = "CHESSR",
                                   diffusion = NULL,
                                   seed = 2026L,
                                   chains = 4L,
                                   warmup = NULL,
                                   iterations = NULL,
                                   adapt_delta = NULL,
                                   N = NULL,
                                   validate = NULL,
                                   output_dir = file.path(getwd(), "NicoStan_example_results"),
                                   model_dir = file.path(.NicoStan_examples_source_directory, "models")) {
        engine <-  match.arg(engine)
        math_backend <-  match.arg(math_backend)
        if (engine == "cmdstanr" && math_backend != "Stan") stop("The CmdStanR comparison arm uses the plain Stan model.")
        profile <-  match.arg(profile)
        if (is.null(warmup)) warmup <-  if (profile == "smoke") if (model == "stochastic_volatility") 500L else 200L else 1000L
        if (is.null(iterations)) iterations <-  if (profile == "smoke") if (model == "stochastic_volatility") 500L else 250L else 1000L
        if (is.null(adapt_delta)) adapt_delta <-  if (model == "stochastic_volatility") 0.999 else 0.9
        if (is.null(validate)) validate <-  profile == "smoke"
        stopifnot(is.numeric(adapt_delta), length(adapt_delta) == 1L, adapt_delta > 0, adapt_delta < 1)
        example <-  make_NicoStan_example(model = model, profile = profile, seed = seed, chains = chains, N = N)
        if (is.null(diffusion)) diffusion <-  example$diffusion_eligible
        if (isTRUE(diffusion) && !example$diffusion_eligible) stop("This model has no explicit Gaussian nuisance block.")
        if (engine == "cmdstanr") diffusion <-  FALSE
        ##
        plain_source <-  file.path(model_dir, example$source_file)
        selected_source <-  if (math_backend == "Stan") example$source_file else sub(".stan$", "_avx.stan", example$source_file)
        source_file <-  file.path(model_dir, selected_source)
        stopifnot(file.exists(source_file), warmup >= 50L, iterations >= 20L)
        source_hash <-  digest::digest(file = source_file, algo = "sha256")
        model_build_dir <-  file.path(output_dir, "models", paste0(model, "_", substr(source_hash, 1L, 12L), "_", math_backend))
        dir.create(model_build_dir, recursive = TRUE, showWarnings = FALSE)
        ## 2026-09-22: AVX builds carry the backend in the file name (<model>_avx_AVX2.stan / _AVX512.stan): BridgeStan's R interface
        ## finds a loaded model library by its base name, so an AVX2 and an AVX512 build of the same _avx.stan loaded in one R
        ## session shared a name, and the build loaded first then returned an all-zero gradient (A2_simd test 4b).
        stan_file <-  file.path(model_build_dir, if (math_backend == "Stan") selected_source else sub(".stan$", paste0("_", math_backend, ".stan"), selected_source))
        if (!file.exists(stan_file)) stopifnot(file.copy(source_file, stan_file))
        stan_file <-  normalizePath(stan_file, mustWork = TRUE)
        suffix <-  if (engine == "NicoStan") paste0(burnin_algorithm, "_", math_backend, "_diffusion_", diffusion) else "NUTS"
        settings_hash <-  substr(digest::digest(list(chains, warmup, iterations, adapt_delta, burnin_algorithm, diffusion, source_hash, example$data, example$initial_values, digest::digest(file = file.path(.NicoStan_examples_source_directory, "NicoStan_examples.R"), algo = "sha256")), algo = "sha256"), 1L, 10L)
        run_dir <-  file.path(output_dir, paste0(model, "_N", example$N, "_", engine, "_", suffix, "_", profile, "_seed_", seed, "_", settings_hash))
        dir.create(run_dir, recursive = TRUE, showWarnings = FALSE)
        data_file <-  file.path(run_dir, "data.json")
        cmdstanr::write_stan_json(example$data, data_file)
        message(paste0("Running ", model, " with ", engine, " (", suffix, ")"))
        ##
        compile_started <-  proc.time()[[3L]]
        gradient_check <-  NULL
        structure_check <-  NULL
        avx_comparison <-  NULL
        header <-  NULL
        header_hashes <-  NULL
        compile_arguments <-  c("STAN_THREADS=true", "PRECOMPILED_HEADERS=false")
        if (math_backend != "Stan") {
                header <-  system.file("include", "BayesMVP", "stan_external_functions.hpp", package = "BayesMVP")
                if (!nzchar(header) || !file.exists(header)) stop("Install the new BayesMVP extension to use its custom AVX header.")
                if (file.exists("/proc/cpuinfo")) {
                        flags <-  paste(readLines("/proc/cpuinfo", warn = FALSE)[startsWith(readLines("/proc/cpuinfo", warn = FALSE), "flags")], collapse = " ")
                        required <-  if (math_backend == "AVX512") c("avx512f", "avx512vl", "avx512dq", "fma") else c("avx2", "fma")
                        if (!all(vapply(required, function(flag) grepl(paste0("\\b", flag, "\\b"), flags), logical(1L)))) {
                                stop(paste0("This CPU does not expose the required ", math_backend, " instruction set."))
                        }
                }
                ## 2026-09-22: ISA flags APPENDED via CXXFLAGS_OPTIM (a command-line CXXFLAGS replaced BridgeStan's make/local
                ## CXXFLAGS, dropping -DNDEBUG and -fno-math-errno etc. from the AVX arms only), and "AVX2" = the same build as
                ## "AVX512" plus -DBAYESMVP_FORCE_AVX2 (4-lane BayesMVP kernels; simd_lanes is checked below). Same rule as
                ## NicoStan_benchmark_compile_arguments() in NicoStan_benchmarks.R.
                isa_flags <-  if (math_backend == "AVX512") "-march=native -mtune=native -mfma -mavx2 -mavx512f -mavx512vl -mavx512dq" else
                              "-march=native -mtune=native -mfma -mavx2"
                compile_arguments <-  c(compile_arguments, paste0("CXXFLAGS_OPTIM=", shQuote(isa_flags)))
                if (math_backend == "AVX2") compile_arguments <-  c(compile_arguments, "CPPFLAGS_OPTIM=-DBAYESMVP_FORCE_AVX2")
                header_files <-  c(header, file.path(dirname(header), "math", c("fast_and_approx_AVX2_fns.hpp", "fast_and_approx_AVX512_fns.hpp")))
                header_hashes <-  setNames(vapply(header_files, function(path) digest::digest(file = path, algo = "sha256"), character(1L)), basename(header_files))
        }
        if (engine == "NicoStan") {
                fit <-  NicoStan::MVP_model$new(Model_type = "Stan", Stan_data_list = example$data,
                    Stan_model_file_path = stan_file, sample_nuisance = example$n_nuisance > 0L,
                    Stan_cpp_user_header = header, make_args = compile_arguments)
                compilation_seconds <-  proc.time()[[3L]] - compile_started
                structure_check <- NicoStan_check_model_structure(fit, example,
                    expected_simd_lanes = if (math_backend == "Stan") NULL else if (math_backend == "AVX512") 8 else 4)
                if (isTRUE(validate)) gradient_check <-  NicoStan_check_gradient(model = fit, example = example,
                    expected_simd_lanes = if (math_backend == "Stan") NULL else if (math_backend == "AVX512") 8 else 4)
                if (math_backend != "Stan" && isTRUE(validate)) {
                        plain_build_dir <-  file.path(output_dir, "models", paste0(model, "_", substr(digest::digest(file = plain_source, algo = "sha256"), 1L, 12L), "_Stan"))
                        dir.create(plain_build_dir, recursive = TRUE, showWarnings = FALSE)
                        plain_file <-  file.path(plain_build_dir, example$source_file)
                        if (!file.exists(plain_file)) stopifnot(file.copy(plain_source, plain_file))
                        avx_comparison <-  NicoStan_compare_avx(fit, example, normalizePath(plain_file, mustWork = TRUE))
                }
                sample_started <-  proc.time()[[3L]]
                set.seed(seed)
                fit$sample(n_burnin = warmup, n_iter = iterations, n_chains_burnin = chains,
                    n_chains_sampling = chains, n_superchains = max(2L, floor(chains / 2L)),
                    init_lists_per_chain = example$initial_values,
                    burnin_algorithm = burnin_algorithm, diffusion_HMC = diffusion, partitioned_HMC = FALSE,
                    metric_type_main = "Empirical", metric_shape_main = "diag", metric_type_nuisance = "uniform_diag",
                    M_decay_type = "inverse", M_decay_power = 0.5, M_decay_scale = 1.13,
                    ratio_M_main = 0.9, ratio_M_nuisance = 0.9, learning_rate = 0.05, learning_rate_initial = 0.1,
                    n_threads_WCP_burnin = 1L, n_threads_WCP_sampling = 1L, adapt_delta = adapt_delta,
                    n_refresh = 100L, seed = seed, stream = seed, reorder_cols_MVP = FALSE)
                fitting_seconds <-  proc.time()[[3L]] - sample_started
                stopifnot(identical(fit$result$randomize_tau_burnin, FALSE), identical(fit$result$randomize_tau_sampling, TRUE),
                          identical(fit$result$burnin_object$EHMC_args_as_Rcpp_List$randomize_tau, FALSE))
                fit$summary(save_log_lik_trace = FALSE, compute_nested_rhat = TRUE, compute_generated_quantities = math_backend == "Stan")
                draws <-  fit$model_fit_object$traces$traces_as_arrays$trace_params_main
                ## `result$time_sampling` is the unrounded tictoc interval
                ## captured around the native sampling call.  Do not replace a
                ## missing sampling timer with total fitting time: that would
                ## mislabel compilation, burn-in, and post-processing as
                ## post-warmup sampling.
                sampling_seconds <- as.numeric(fit$result$time_sampling)
                sampling_time_basis <- "NicoStan result$time_sampling: unrounded native sampling interval"
                if (!is.finite(sampling_seconds) || sampling_seconds <= 0) {
                        sampling_seconds <- NA_real_
                        sampling_time_basis <- "NicoStan sampling timer unavailable; sampling efficiency is NA"
                }
                divergences <-  fit$model_fit_object$summaries$divergences$n_divs
                n_leapfrog <-  NA_real_
                efficiency <-  fit$model_fit_object$summaries$efficiency_info
                gradient_work <-  efficiency$L_main_during_sampling * iterations * chains
                gradient_work_basis <-  "NicoStan adapted tau/epsilon times iterations and chains (APMS gradient-work estimate)"
                max_nested_rhat_main <- if (is.null(efficiency$Max_nested_rhat_main)) NA_real_ else as.numeric(efficiency$Max_nested_rhat_main)
                nested_rhat_grouping <- fit$model_fit_object$summaries$HMC_info$nested_rhat_grouping
                if (is.null(nested_rhat_grouping)) nested_rhat_grouping <- list(status = "unavailable")
                adaptation <-  fit$model_fit_object$adaptation
                native_library <-  getLoadedDLLs()[["NicoStan"]][["path"]]
                build <-  list(package_version = as.character(packageVersion("NicoStan")),
                               package_path = find.package("NicoStan"),
                               native_sha256 = digest::digest(file = native_library, algo = "sha256"),
                               rdb_sha256 = digest::digest(file = file.path(find.package("NicoStan"), "R", "NicoStan.rdb"), algo = "sha256"),
                               bridgestan_version = as.character(packageVersion("bridgestan")))
        } else {
                stan_model <-  cmdstanr::cmdstan_model(stan_file = stan_file, cpp_options = list(stan_threads = TRUE), quiet = TRUE)
                compilation_seconds <-  proc.time()[[3L]] - compile_started
                sample_started <-  proc.time()[[3L]]
                set.seed(seed)
                fit <-  stan_model$sample(data = example$data, init = example$initial_values, chains = chains,
                    parallel_chains = chains, threads_per_chain = 1L, iter_warmup = warmup, iter_sampling = iterations,
                    adapt_delta = adapt_delta, max_treedepth = 12L, seed = seed, refresh = 100L, output_dir = run_dir)
                fitting_seconds <-  proc.time()[[3L]] - sample_started
                draws <-  as.array(fit$draws(variables = example$main_parameters))
                sampling_seconds <-  max(fit$time()$chains$sampling)
                sampling_time_basis <- "CmdStanR fit$time()$chains$sampling"
                if (!is.finite(sampling_seconds) || sampling_seconds <= 0) {
                        sampling_seconds <- NA_real_
                        sampling_time_basis <- "CmdStanR sampling timer unavailable; sampling efficiency is NA"
                }
                sampler_diagnostics <-  as.array(fit$sampler_diagnostics())
                divergences <-  sum(sampler_diagnostics[, , "divergent__"])
                n_leapfrog <-  sum(sampler_diagnostics[, , "n_leapfrog__"])
                gradient_work <-  n_leapfrog
                gradient_work_basis <-  "CmdStan sum of n_leapfrog__ during sampling (APMS gradient-work proxy)"
                adaptation <-  list(burnin_algorithm = "NUTS")
                max_nested_rhat_main <- NA_real_
                nested_rhat_grouping <- list(status = "not_applicable_cmdstanr")
                build <-  list(cmdstanr_version = as.character(packageVersion("cmdstanr")),
                               cmdstan_version = cmdstanr::cmdstan_version())
        }
        stopifnot(all(is.finite(draws)), identical(as.integer(dim(draws)[1:2]), as.integer(c(iterations, chains))))
        summary <-  posterior::summarise_draws(posterior::as_draws_array(draws), "mean", "sd", "mcse_mean", "ess_bulk", "ess_tail", "rhat")
        selected_parameters <-  as.character(summary$variable)
        selected_bases <-  sub("\\[.*$", "", selected_parameters)
        stopifnot(length(selected_parameters) > 0L, all(selected_bases %in% example$main_parameters),
                  setequal(unique(selected_bases), example$main_parameters))
        stopifnot(all(is.finite(summary$mean)), all(is.finite(summary$sd)), all(summary$sd > 0),
                  all(is.finite(summary$ess_bulk)), all(is.finite(summary$rhat)))
        record <-  list(example = example, engine = engine, math_backend = math_backend, diffusion = diffusion, settings = list(chains = chains,
            warmup = warmup, iterations = iterations, seed = seed, burnin_algorithm = burnin_algorithm, adapt_delta = adapt_delta),
            source_sha256 = source_hash, plain_source_sha256 = digest::digest(file = plain_source, algo = "sha256"),
            external_header_sha256 = header_hashes, compile_arguments = compile_arguments,
            harness_sha256 = digest::digest(file = file.path(.NicoStan_examples_source_directory, "NicoStan_examples.R"), algo = "sha256"),
            data_sha256 = digest::digest(example$data, algo = "sha256"),
            initial_values_sha256 = digest::digest(example$initial_values, algo = "sha256"),
            structure_check = structure_check, gradient_check = gradient_check, avx_comparison = avx_comparison,
            prior_specification = "example_priors_2026_09_21", compilation_seconds = compilation_seconds,
            fitting_seconds = fitting_seconds, sampling_seconds = sampling_seconds,
            sampling_time_basis = sampling_time_basis,
            divergences = divergences, n_leapfrog_sampling = n_leapfrog,
            ess_parameter_names = selected_parameters, ess_parameter_block = "non_nuisance_parameters_only",
            ess_parameter_bases = example$main_parameters,
            min_ESS_main = min(summary$ess_bulk), min_ESS_tail_main = min(summary$ess_tail),
            min_ESS_per_sec_sampling = min(summary$ess_bulk) / sampling_seconds,
            min_ESS_per_sec_fitting = min(summary$ess_bulk) / fitting_seconds,
            gradient_work_sampling = gradient_work, gradient_work_basis = gradient_work_basis,
            min_ESS_per_1000_grad_sampling = 1000 * min(summary$ess_bulk) / gradient_work,
            exact_gradient_evaluations = NA_real_, max_nested_rhat_main = max_nested_rhat_main,
            nested_rhat_grouping = nested_rhat_grouping, adaptation = adaptation, build = build,
            summary = summary, draws_main = draws,
            diagnostic_target_met = max(summary$rhat) < 1.05 && min(summary$ess_bulk) >= 100 && divergences == 0,
            run_dir = normalizePath(run_dir, mustWork = TRUE),
            session = sessionInfo())
        saveRDS(record, file.path(run_dir, "result.rds"))
        utils::write.csv(as.data.frame(summary), file.path(run_dir, "summary.csv"), row.names = FALSE)
        writeLines(selected_parameters, file.path(run_dir, "ESS_parameter_names.txt"))
        message(paste0("Completed ", model, ": ", nrow(summary), " main parameters; nuisance dimension ", example$n_nuisance,
                       "; maximum R-hat ", round(max(summary$rhat), 3), "; divergences ", divergences))
        record
}
