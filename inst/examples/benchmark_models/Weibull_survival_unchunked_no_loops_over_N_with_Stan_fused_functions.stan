// Weibull_survival_unchunked_no_loops_over_N_with_Stan_fused_functions.stan
// Unchunked Stan::math version of Weibull_survival_chunked_no_loops_over_N_v2 (the Stan math file
// of the chunking paper's
// layout timings): the same log density as that file at N_chunks = 1, and as Weibull_survival.stan, as one pass
// over all observations, without reduce_sum_static. The same data (chunk_size, when it is in the data, is not
// read) and the same unconstrained parameters, in the same order and with the same names (alpha, beta, mu), as
// both files.
// The likelihood is Stan's fused Poisson-log GLM, one call on the observed failures and the censored
// observations together: with the scale exp(-(mu + x' beta) / alpha) of Weibull_survival.stan and
// lambda = alpha log t + mu + x' beta,
//     weibull_lpdf(t | alpha, scale)   = log(alpha) - log(t) + lambda - exp(lambda)
//                                      = poisson_log_glm_lpmf(1 | lambda) + log(alpha) - log(t),
//     weibull_lccdf(t | alpha, scale)  = -exp(lambda) = poisson_log_glm_lpmf(0 | lambda)
// (lgamma(1 + 1) = lgamma(0 + 1) = 0), i.e. poisson_log_glm with the covariates [1, X], the coefficients
// append_row(mu, beta) and the intercept alpha log t of each observation, plus N log(alpha) - sum(log(t)), whose
// second term is data (made once in transformed data). The same identity as
// Weibull_survival_chunked_no_loops_over_N_v2_with_poisson_log_glm, here in one call over all observations.
// lambda and lambda_cens, transformed parameters of Weibull_survival.stan, are reported as generated quantities
// (the same values).
// No loop over N anywhere.


data {
  int<lower=0> N;                    // number of observations (the observed failures)
  int<lower=0> N_cens;               // number of censored observations
  int<lower=0> K;                    // number of covariates
  vector[N] t;                       // observed failure times
  vector[N_cens] t_cens;             // censored times
  matrix[N, K] X;                    // covariate matrix for failures
  matrix[N_cens, K] X_cens;          // covariate matrix for censored
}


transformed data {
  // the observed failures and then the censored observations, as one set of observations of the GLM: their
  // covariates with a first column of ones (so that mu is the first coefficient), their log times, and their
  // outcomes (1 for an observed failure, 0 for a censored observation)
  matrix[N + N_cens, K + 1] X_with_ones_of_all_observations
        = append_row(append_col(rep_vector(1, N), X), append_col(rep_vector(1, N_cens), X_cens));
  vector[N + N_cens] log_t_of_all_observations = append_row(log(t), log(t_cens));
  array[N + N_cens] int failure_indicator_of_all_observations = append_array(rep_array(1, N),
                                                                             rep_array(0, N_cens));
  real sum_of_log_t = sum(log(t));   // the data term -sum(log(t)) of the observed failures' densities
}


parameters {
  real<lower=0> alpha;               // Weibull shape parameter
  vector[K] beta;                    // regression coefficients
  real mu;                           // intercept
}


model {
  // Priors
  // Example-scale priors: centred/unit-scale covariates and survival times in the chosen reference unit.
  alpha ~ lognormal(0, 0.5);
  beta ~ normal(0, 1);
  // mu ~ normal(-1, 1.5);
  mu ~ normal(-1, 1.5);

  // Likelihood for observed failures and for censored observations (Stan's fused Poisson-log GLM; see the
  // header)
  target += poisson_log_glm_lupmf(failure_indicator_of_all_observations | X_with_ones_of_all_observations,
                                  alpha * log_t_of_all_observations, append_row(mu, beta))
            + N * log(alpha) - sum_of_log_t;
}


generated quantities {
  // (lambda and lambda_cens: the transformed parameters of Weibull_survival.stan, the same values)
  vector[N] lambda;                  // scale parameters for failures
  vector[N_cens] lambda_cens;        // scale parameters for censored
  vector[N + N_cens] log_lik;
  vector[N + N_cens] t_pred;         // posterior predictive samples

  // Weibull parameterization: scale = exp(-(mu + X*beta)/alpha)
  lambda = exp(-(mu + X * beta) / alpha);
  lambda_cens = exp(-(mu + X_cens * beta) / alpha);

  // Log-likelihood for observed (weibull_lpdf(t[n] | alpha, lambda[n]) of every observation, whole-vector), and
  // its posterior predictive samples (weibull_rng over the vector lambda: the same draws, in the same order), as
  // in the chunked file
  if (N > 0) {
    log_lik[1:N] = log(alpha) + (alpha - 1) * log(t) - alpha * log(lambda) - (t ./ lambda) .^ alpha;
    t_pred[1:N] = to_vector(weibull_rng(alpha, lambda));
  }

  // Log-likelihood for censored (weibull_lccdf(t_cens[n] | alpha, lambda_cens[n]) of every censored observation,
  // whole-vector: -(t_cens / lambda_cens)^alpha), and its posterior predictive samples
  if (N_cens > 0) {
    log_lik[(N + 1):(N + N_cens)] = -(t_cens ./ lambda_cens) .^ alpha;
    t_pred[(N + 1):(N + N_cens)] = to_vector(weibull_rng(alpha, lambda_cens));
  }
}






















