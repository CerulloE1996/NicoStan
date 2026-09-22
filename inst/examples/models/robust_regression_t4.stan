

// robust_regression_t4.stan


data {
    int<lower=1> N;                    // number of observations
    int<lower=1> K;                    // number of predictors
    matrix[N, K] X;                    // design matrix (without intercept)
    vector[N] y;                       // response variable
    real<lower=1> nu;                  // degrees of freedom (fixed at 4)
}


parameters {
    real alpha;                        // intercept
    vector[K] beta;                    // regression coefficients
    real<lower=0> sigma;               // scale parameter
}


transformed parameters {
    vector[N] mu = alpha + X * beta;   // linear predictor
}


model {
    // Priors
    alpha ~ normal(0, 10);
    beta ~ normal(0, 2.5);
    sigma ~ exponential(1);
    
    // Likelihood with t-distributed errors
    y ~ student_t(nu, mu, sigma);
}


generated quantities {
    vector[N] log_lik;
    vector[N] y_rep;
    
    for (n in 1:N) {
      log_lik[n] = student_t_lpdf(y[n] | nu, mu[n], sigma);
      y_rep[n] = student_t_rng(nu, mu[n], sigma);
    }
}



