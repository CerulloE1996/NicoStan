// hierarchical_logistic_regression_unchunked_no_loops_over_N_with_Stan_fused_functions.stan
// Unchunked Stan::math version of the hierarchical logistic model of the tape-chunking paper's layout timings
// (hierarchical_logistic_regression_chunked_no_loops_over_N_automatic_reduce_sum_chunked, from its source
// hierarchical_logistic_regression_chunked_no_loops_over_N): the same log density as
// hierarchical_logistic_regression.stan, as one pass over all N observations, without reduce_sum_static.
// The same data (chunk_size, when it is in the data, is not read).
// Its parameters are those of hierarchical_logistic_regression.stan, in the same order and with the same names:
// beta_raw as a K x 2 matrix (column 1: the K groups' raw intercepts; column 2: their raw slopes). The chunked
// file's array[K] row_vector[2] beta_raw (the same K x 2 values, group by group) exists only so that
// reduce_sum_static can slice the groups, and orders the same values row by row; a point-by-point check of the
// log density and gradient is hence made against hierarchical_logistic_regression.stan.
// The likelihood is Stan's fused logistic GLM: with the covariates [1, x] (made once in transformed data) and
// the coefficients mu (the population intercept and slope), the intercept of each observation is its group's
// deviation from mu, sigma[1] beta_raw[group, 1] + sigma[2] beta_raw[group, 2] x, i.e.
//     beta[group, 1] + beta[group, 2] x = x_with_ones * mu + (the deviations),
// the same linear predictor as hierarchical_logistic_regression.stan. The groups' deviations are made once per
// group (K values), then read at each observation's group.
// beta, the transformed parameter of hierarchical_logistic_regression.stan, is reported as a generated quantity
// (the same values).
// No loop over N anywhere.


data {
    int<lower=0> N;                    // number of observations
    int<lower=1> K;                    // number of groups
    array[N] int<lower=1, upper=K> group; // group indicator for each observation
    vector[N] x;                       // predictor variable
    array[N] int<lower=0, upper=1> y;  // binary outcome
}


transformed data {
    // the covariates with a first column of ones, so that mu is the coefficient vector of the GLM
    matrix[N, 2] x_with_ones = append_col(rep_vector(1, N), x);
}


parameters {
    matrix[K, 2] beta_raw;             // standard normal nuisance block, declared first
    vector[2] mu;                      // population means for [intercept, slope]
    vector<lower=0>[2] sigma;          // population standard deviations
}


model {
    // Priors
    mu ~ normal(0, 2);
    sigma ~ normal(0, 2);

    // Equivalent non-centred form of beta[, i] ~ normal(mu[i], sigma[i]).
    to_vector(beta_raw) ~ std_normal();

    // Likelihood (Stan's fused logistic GLM; see the header)
    {
      vector[K] deviation_of_intercept_of_group = sigma[1] * beta_raw[, 1];
      vector[K] deviation_of_slope_of_group = sigma[2] * beta_raw[, 2];
      y ~ bernoulli_logit_glm(x_with_ones,
                              deviation_of_intercept_of_group[group]
                              + deviation_of_slope_of_group[group] .* x,
                              mu);
    }
}


generated quantities {
    matrix[K, 2] beta;                 // group-specific intercepts and slopes
    vector[N] log_lik;
    vector[N] y_rep;

    for (i in 1:2) {
      beta[, i] = mu[i] + sigma[i] * beta_raw[, i];
    }

    // (on the whole vectors, as in the chunked file: each observation's bernoulli_logit_lpmf(y[n] | eta[n]) is
    //  log_inv_logit((2 y[n] - 1) eta[n]), and bernoulli_logit_rng draws the replicates in the same order)
    {
      vector[N] eta = beta[group, 1] + beta[group, 2] .* x;
      log_lik = log_inv_logit((2 * to_vector(y) - 1) .* eta);
      y_rep = to_vector(bernoulli_logit_rng(eta));
    }
}






















