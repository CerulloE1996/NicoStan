data {
  int<lower=0> N;          // Number of observations
  array[N] real y;               // Observations
}
parameters {
  real mu;                 // Mean parameter
}
model {
  mu ~ normal(0, 10);      // Prior on the mean
  y ~ normal(mu, 1);       // Likelihood
}
