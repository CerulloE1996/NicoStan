// robust_Student_t_regression_unchunked_no_loops_over_N_with_Stan_fused_functions.stan
// Unchunked plain Stan version of robust_Student_t_regression_chunked_no_loops_over_N_v2 (the Stan math file of
// Paper 1's layout timings): the same log density as that file at N_chunks = 1, and as
// robust_Student_t_regression.stan, as one pass over all N observations, without reduce_sum_static. The same
// data (chunk_size, when it is in the data, is not read) and the same unconstrained parameters, in the same
// order and with the same names (alpha, beta, sigma), as both files.
// Stan has no fused Student-t GLM: the likelihood is Stan's vectorised student_t on the whole vector of
// responses, with alpha folded into the coefficients of one product per observation ([1, X] times
// append_row(alpha, beta), the covariates' first column of ones made once in transformed data), as in v2.
// mu, the transformed parameter of robust_Student_t_regression.stan, is reported as a generated quantity (the
// same values), so that no N-sized value is saved for the log density.
// No loop over N anywhere.


data {
    int<lower=1> N;                    // number of observations
    int<lower=1> K;                    // number of predictors
    matrix[N, K] X;                    // design matrix (without intercept)
    vector[N] y;                       // response variable
    real<lower=1> nu;                  // degrees of freedom (fixed at 4)
}


transformed data {
    // the covariates with a first column of ones, so that alpha is the first coefficient of the product
    matrix[N, K + 1] X_with_ones = append_col(rep_vector(1, N), X);
}


parameters {
    real alpha;                        // intercept
    vector[K] beta;                    // regression coefficients
    real<lower=0> sigma;               // scale parameter
}


model {
    // Priors
    alpha ~ normal(0, 10);
    beta ~ normal(0, 2.5);
    sigma ~ exponential(1);

    // Likelihood with t-distributed errors (alpha + X * beta = [1, X] * append_row(alpha, beta))
    y ~ student_t(nu, X_with_ones * append_row(alpha, beta), sigma);
}


generated quantities {
    // (mu: the transformed parameter of robust_Student_t_regression.stan, the same values)
    vector[N] mu = alpha + X * beta;   // linear predictor
    vector[N] log_lik;
    vector[N] y_rep;

    // (on the whole vectors, as in the chunked file: each observation's log likelihood,
    //  student_t_lpdf(y[n] | nu, mu[n], sigma), in closed form, and the replicates drawn by
    //  student_t_rng(nu, mu, sigma) on the whole vector, in the same order)
    log_lik = lgamma((nu + 1) / 2) - lgamma(nu / 2) - 0.5 * log(nu * pi()) - log(sigma)
              - (nu + 1) / 2 * log1p(square((y - mu) / sigma) / nu);
    y_rep   = to_vector(student_t_rng(nu, mu, sigma));
}






















