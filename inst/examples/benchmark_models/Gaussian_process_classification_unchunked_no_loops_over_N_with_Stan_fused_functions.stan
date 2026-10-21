// Gaussian_process_classification_unchunked_no_loops_over_N_with_Stan_fused_functions.stan
// Unchunked plain Stan version of Gaussian_process_classification_chunked (the Stan math file of Paper 1's
// layout timings): the same log density as that file at N_chunks = 1, and as
// Gaussian_process_classification.stan, without reduce_sum_static. The same data (chunk_size, when it is in the data, is not read) and the same
// unconstrained parameters, in the same order and with the same names (eta, mu, lengthscale, sigma_f), as both
// files.
// Stan's fused functions: gp_exp_quad_cov for the covariance matrix, add_diag for its 1e-8 jitter,
// cholesky_decompose, and Stan's fused logistic GLM for the likelihood: the latent log-odds f = mu + L_K * eta
// are the GLM's linear predictor, with the Cholesky factor L_K as its covariates, mu as its intercept and the
// nuisance block eta as its coefficients, i.e. y ~ bernoulli_logit_glm(L_K, mu, eta), the same density as
// y ~ bernoulli_logit(mu + L_K * eta) in one node (the product, the likelihood and their gradients), in place of
// the N x N product and the N log-odds on the autodiff tape. (In the chunked file f is made on the main tape and
// sliced; the GLM needs the whole of L_K, so it has no chunked form without L_K shared by every chunk.)
// The covariance has only a 1e-8 jitter on its diagonal (its condition number can be about 1e10), so the
// gradient of the GLM's product can differ from that of L_K * eta by more than rounding at some points; its
// check is against the reference's own sensitivity to a tiny move of the point.
// No loop over N anywhere.
//
// Gaussian process classification: a Bernoulli-logit outcome y (e.g. disease status) whose log-odds is a latent
// Gaussian process f of an input x (e.g. a standardised biomarker), with a squared-exponential covariance
// (lengthscale, sigma_f) and a constant mean mu, non-centred: f = mu + L_K * eta, eta ~ N(0, I).


data {
    int<lower=1> N;                           // number of observations
    array[N] real x;                          // input (e.g. a standardised biomarker)
    array[N] int<lower=0, upper=1> y;         // binary outcome (e.g. disease status)
}


parameters {
    vector[N] eta;                            // standard normal nuisance block, declared first
    real mu;                                  // constant mean of the latent log-odds
    real<lower=0> lengthscale;                // lengthscale of the squared-exponential covariance
    real<lower=0> sigma_f;                    // marginal standard deviation of the latent log-odds
}


model {
    matrix[N, N] L_K = cholesky_decompose(add_diag(gp_exp_quad_cov(x, sigma_f, lengthscale), 1e-8));
    mu ~ normal(0, 1);
    lengthscale ~ inv_gamma(5, 5);
    sigma_f ~ normal(0, 1.5);
    eta ~ std_normal();
    // (y ~ bernoulli_logit(mu + L_K * eta), with Stan's fused logistic GLM; see the header)
    y ~ bernoulli_logit_glm(L_K, mu, eta);
}






















