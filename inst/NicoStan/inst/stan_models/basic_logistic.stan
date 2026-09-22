



data {
  
  int<lower=1> N;
  array[N] int y;
  int<lower=1> n_covs;
  matrix[N, n_covs] X;
  
}

parameters {
  
  real alpha; // intercept
  vector[n_covs] beta; // coefficients for cts covariates 
  
}

 


model {
  
  //// likelihood
  {
 
    /// vector[N] alpha_p_Xbeta = alpha + X * beta;
    vector[N] probs_vector = inv_logit(alpha + X * beta);
    vector[N] log_lik = log(to_vector(y).*probs_vector + (1 - to_vector(y)) .* (1.0 - probs_vector));
 
     target +=  sum(log_lik);
  }
  
  /// priors
  alpha ~ normal(0.0, 1.0);
  beta  ~ normal(0.0, 5.0);
   
}


