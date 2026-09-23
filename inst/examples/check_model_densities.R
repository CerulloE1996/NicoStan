#### =====================================================================================================================================
## Independent R log-density checks for the Stan examples, including all density constants but no constraint Jacobian.
## Non-centred Gaussian priors are checked through their equivalent centred densities and transformation Jacobians.
##
NicoStan_reference_log_density <-  function(example, p) {
        d <-  example$data
        normal <-  function(x, mean = 0, sd = 1) sum(dnorm(x, mean, sd, log = TRUE))
        if (example$model == "cox_frailty") {
                eta <-  c(d$X %*% p$beta) + p$sigma_frailty * p$log_frailty_raw[d$group]
                likelihood <-  sum(vapply(which(d$event == 1L), function(i) {
                        risk <-  eta[d$t >= d$t[i]]
                        eta[i] - (max(risk) + log(sum(exp(risk - max(risk)))))
                }, numeric(1L)))
                return(likelihood + normal(p$beta, sd = 2) + normal(p$log_frailty_raw) + dexp(p$sigma_frailty, log = TRUE))
        }
        if (example$model == "weibull") {
                observed_scale <-  exp(-(p$mu + c(d$X %*% p$beta)) / p$alpha)
                censored_scale <-  exp(-(p$mu + c(d$X_cens %*% p$beta)) / p$alpha)
                return(sum(dweibull(d$t, shape = p$alpha, scale = observed_scale, log = TRUE)) +
                       sum(pweibull(d$t_cens, shape = p$alpha, scale = censored_scale, lower.tail = FALSE, log.p = TRUE)) +
                       dlnorm(p$alpha, sdlog = 0.5, log = TRUE) + normal(p$beta, sd = 1) + normal(p$mu, mean = -1, sd = 1.5))
        }
        if (example$model == "robust_t4") {
                mu <-  p$alpha + c(d$X %*% p$beta)
                return(sum(dt((d$y - mu) / p$sigma, df = d$nu, log = TRUE) - log(p$sigma)) +
                       normal(p$alpha, sd = 10) + normal(p$beta, sd = 2.5) + dexp(p$sigma, log = TRUE))
        }
        if (example$model == "hierarchical_logistic") {
                beta <-  sweep(sweep(p$beta_raw, 2L, p$sigma, "*"), 2L, p$mu, "+")
                eta <-  beta[d$group, 1L] + beta[d$group, 2L] * d$x
                centred_prior <-  sum(vapply(1:2, function(i) normal(beta[, i], p$mu[i], p$sigma[i]), numeric(1L)))
                log_jacobian <-  d$K * sum(log(p$sigma))
                return(sum(dbinom(d$y, 1L, plogis(eta), log = TRUE)) + centred_prior + log_jacobian +
                       normal(p$mu, sd = 2) + normal(p$sigma, sd = 2))
        }
        if (example$model == "stochastic_volatility_discrete_time") {
                h <-  numeric(d$T)
                h[1L] <-  p$mu + p$sigma * p$h_std[1L] / sqrt(1 - p$phi^2)
                if (d$T > 1L) for (i in 2:d$T) h[i] <-  p$mu + p$phi * (h[i - 1L] - p$mu) + p$sigma * p$h_std[i]
                return(normal(d$y, sd = exp(h / 2)) + normal(p$h_std) + dunif(p$phi, -1, 1, log = TRUE) +
                       normal(p$sigma, sd = 0.5) + normal(p$mu, sd = 2))
        }
        if (example$model == "gaussian_process") {
                x <-  (d$x - mean(d$x)) / sd(d$x)
                y <-  (d$y - mean(d$y)) / sd(d$y)
                covariance <-  p$sigma_f^2 * exp(-outer(x, x, "-")^2 / (2 * p$lengthscale^2)) +
                               diag(0.05^2 + p$sigma_n^2, d$N)
                factor <-  chol(covariance)
                whitened <-  forwardsolve(t(factor), y)
                likelihood <-  -0.5 * (d$N * log(2 * pi) + 2 * sum(log(diag(factor))) + sum(whitened^2))
                return(likelihood + dgamma(1 / p$lengthscale, shape = 5, rate = 5, log = TRUE) - 2 * log(p$lengthscale) +
                       normal(p$sigma_f) + normal(p$sigma_n))
        }
        if (example$model == "joint_longitudinal_survival") {
                factor <-  matrix(c(p$sigma_b0, p$rho_b * p$sigma_b1, 0,
                                    p$sigma_b1 * sqrt(1 - p$rho_b^2)), nrow = 2L)
                effects <-  factor %*% p$b_raw
                inverse_covariance <-  solve(tcrossprod(factor))
                centred_prior <-  -d$N_subj * (log(2 * pi) + sum(log(diag(factor)))) -
                                  0.5 * sum(effects * (inverse_covariance %*% effects))
                log_jacobian <-  d$N_subj * sum(log(diag(factor)))
                mu <-  c(d$X_long %*% p$beta_long) + effects[1L, d$subj_long] + effects[2L, d$subj_long] * d$time_long
                ## The example's fixed effects are intercept + time. Use the analytic integral as an independent check of quadrature.
                rate <-  p$lambda * exp(c(d$X_surv %*% p$beta_surv) + p$alpha_assoc * (p$beta_long[1L] + effects[1L, ]))
                slope <-  p$alpha_assoc * (p$beta_long[2L] + effects[2L, ])
                cumulative_hazard <-  rate * d$T_surv
                use_ratio <-  abs(slope) > 1e-8
                cumulative_hazard[use_ratio] <-  rate[use_ratio] * expm1(slope[use_ratio] * d$T_surv[use_ratio]) / slope[use_ratio]
                log_hazard <-  log(rate) + slope * d$T_surv
                return(normal(d$Y_long, mu, p$sigma_long) + sum(d$event * log_hazard - cumulative_hazard) +
                       centred_prior + log_jacobian + normal(p$beta_long, sd = 2) + normal(p$beta_surv, sd = 2) +
                       sum(dexp(c(p$sigma_long, p$sigma_b0, p$sigma_b1), log = TRUE)) +
                       dunif(p$rho_b, -1, 1, log = TRUE) + dlnorm(p$lambda, meanlog = log(0.2), sdlog = 1, log = TRUE) + normal(p$alpha_assoc))
        }
        if (example$model == "latent_diffusion_survival") {
                ##
                ## ---- independent check: Euler-Maruyama path, then the trapezoid rule on the node set
                ## {grid points up to t} plus t itself, with x(t) from approx() on the grid path.
                ## h(x) = hazard_scale * (x^2 + x_squared_hazard_offset) (round-4 offset, 2026-09-22).
                ##
                if (is.null(d$x_squared_hazard_offset)) stop("latent_diffusion_survival: the data list has no x_squared_hazard_offset.")
                grid_step_length <- d$t_max / d$M
                grid_times <- grid_step_length * (0:d$M)
                x_path <- numeric(d$M + 1)
                x_path[1] <- d$x_0
                for (grid_step_index in seq_len(d$M)) {
                        x_path[grid_step_index + 1] <- x_path[grid_step_index] -
                                                       (p$drift_sin_coefficient * sin(x_path[grid_step_index]) + p$drift_constant) * grid_step_length +
                                                       d$sigma_x * sqrt(grid_step_length) * p$w[grid_step_index]
                }
                x_at_event_or_censoring_time <- approx(x = grid_times, y = x_path, xout = d$t, rule = 2)$y
                log_likelihood_by_subject <- vapply(seq_len(d$N), function(subject_index) {
                        subject_time <- d$t[subject_index]
                        interior_grid_indices <- which(grid_times <= subject_time)
                        interior_grid_indices <- interior_grid_indices[interior_grid_indices <= d$M]
                        node_times <- c(grid_times[interior_grid_indices], subject_time)
                        node_hazards <- p$hazard_scale * (c(x_path[interior_grid_indices], x_at_event_or_censoring_time[subject_index])^2 + d$x_squared_hazard_offset)
                        integrated_hazard <- sum(diff(node_times) * (head(node_hazards, -1) + tail(node_hazards, -1)) / 2)
                        log_hazard_at_time <- log(p$hazard_scale) + log(x_at_event_or_censoring_time[subject_index]^2 + d$x_squared_hazard_offset)
                        d$event[subject_index] * log_hazard_at_time - integrated_hazard
                }, numeric(1))
                return(sum(log_likelihood_by_subject) + normal(p$w) + normal(p$drift_sin_coefficient, mean = 1.4, sd = 1) +
                       normal(p$drift_constant, mean = 1, sd = 1) + dlnorm(p$hazard_scale, meanlog = 0, sdlog = 1, log = TRUE))
        }
        stop("Unknown example model.")
}
##
NicoStan_check_density <-  function(bs, example, tolerance = 1e-7) {
        errors <-  vapply(example$initial_values, function(point) {
                u <-  bs$param_unconstrain_json(jsonlite::toJSON(point, auto_unbox = TRUE, digits = NA))
                abs(bs$log_density(u, propto = FALSE, jacobian = FALSE) - NicoStan_reference_log_density(example, point))
        }, numeric(1L))
        stopifnot(all(is.finite(errors)), max(errors) < tolerance)
        max(errors)
}






















