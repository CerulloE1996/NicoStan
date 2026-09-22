

data {
  int<lower=0> N;                    // number of observations
  int<lower=0> N_cens;               // number of censored observations
  int<lower=0> K;                    // number of covariates
  vector[N] t;                       // observed failure times
  vector[N_cens] t_cens;             // censored times
  matrix[N, K] X;                    // covariate matrix for failures
  matrix[N_cens, K] X_cens;          // covariate matrix for censored
}


parameters {
  real<lower=0> alpha;               // Weibull shape parameter
  vector[K] beta;                    // regression coefficients
  real mu;                           // intercept
}


transformed parameters {
  vector[N] lambda;                  // scale parameters for failures
  vector[N_cens] lambda_cens;        // scale parameters for censored
  
  // Weibull parameterization: scale = exp(-(mu + X*beta)/alpha)
  for (n in 1:N) {
    lambda[n] = exp(-(mu + X[n] * beta) / alpha);
  }
  
  for (n in 1:N_cens) {
    lambda_cens[n] = exp(-(mu + X_cens[n] * beta) / alpha);
  }
}


model {
  // Priors
  // Example-scale priors: centred/unit-scale covariates and survival times in the chosen reference unit.
  alpha ~ lognormal(0, 0.5);
  beta ~ normal(0, 1);
  mu ~ normal(-1, 1.5);
  
  // Likelihood for observed failures
  t ~ weibull(alpha, lambda);
  
  // Likelihood for censored observations
  for (n in 1:N_cens) {
    target += weibull_lccdf(t_cens[n] | alpha, lambda_cens[n]);
  }
}


generated quantities {
  vector[N + N_cens] log_lik;
  vector[N + N_cens] t_pred;         // posterior predictive samples
  
  // Log-likelihood for observed
  for (n in 1:N) {
    log_lik[n] = weibull_lpdf(t[n] | alpha, lambda[n]);
    t_pred[n] = weibull_rng(alpha, lambda[n]);
  }
  
  // Log-likelihood for censored
  for (n in 1:N_cens) {
    log_lik[N + n] = weibull_lccdf(t_cens[n] | alpha, lambda_cens[n]);
    t_pred[N + n] = weibull_rng(alpha, lambda_cens[n]);
  }
  
}
