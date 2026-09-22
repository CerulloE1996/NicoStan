

// Cox partial likelihood with lognormal shared frailty. Ties use the Breslow form.

data {
    int<lower=0> N;                    // number of observations
    int<lower=0> K;                    // number of covariates
    int<lower=0> J;                    // number of groups/clusters
    matrix[N, K] X;                    // covariate matrix
    vector[N] t;                       // survival times
    array[N] int<lower=0,upper=1> event;     // event indicator
    array[N] int<lower=1,upper=J> group;     // group assignment
    array[N] int<lower=0> at_risk_count;     // number at risk at each event time
}


parameters {
    vector[J] log_frailty_raw;         // raw log frailties (nuisance parameters)
    vector[K] beta;                    // regression coefficients
    real<lower=0> sigma_frailty;       // frailty standard deviation
}


transformed parameters {
    vector[J] log_frailty;
    vector[N] frailty;
    
    // non-centered parameterization for frailties
    log_frailty = sigma_frailty * log_frailty_raw;
    
    // assign frailties to observations
    for (n in 1:N) {
      frailty[n] = log_frailty[group[n]];
    }
}


model {
    // priors
    beta ~ normal(0, 2);
    log_frailty_raw ~ normal(0, 1);
    sigma_frailty ~ exponential(1);
    
    // partial likelihood for Cox model with frailties
    for (n in 1:N) {
      if (event[n] == 1) {
        // numerator
        target += X[n] * beta + frailty[n];
        
        // denominator
        real log_sum_exp_denom = negative_infinity();
        for (j in 1:N) {
          if (t[j] >= t[n]) {  // j is at risk at time t[n]
            log_sum_exp_denom = log_sum_exp(log_sum_exp_denom, 
                                          X[j] * beta + frailty[j]);
          }
        }
        target += -log_sum_exp_denom;
      }
    }
}


generated quantities {
    vector[N] log_lik;
    vector[J] frailty_exp = exp(log_frailty);
    
    for (n in 1:N) {
      if (event[n] == 1) {
        real numerator = X[n] * beta + frailty[n];
        real log_sum_exp_denom = negative_infinity();
        for (j in 1:N) {
          if (t[j] >= t[n]) {
            log_sum_exp_denom = log_sum_exp(log_sum_exp_denom, 
                                          X[j] * beta + frailty[j]);
          }
        }
        log_lik[n] = numerator - log_sum_exp_denom;
      } else {
        log_lik[n] = 0;
      }
    }
}



