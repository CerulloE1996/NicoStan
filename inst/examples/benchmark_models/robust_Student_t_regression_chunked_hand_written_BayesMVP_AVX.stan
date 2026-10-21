// (v2: alpha folded into the coefficients of one product per observation, with the covariates' first column
//  of ones made once in transformed data; the same values)
// robust_Student_t_regression_chunked_no_loops_over_N_hand_written_BayesMVP_AVX.stan
// Hand-written AVX version of models_staging_no_loops_over_N_hierarchical_robust_t_IRT/robust_Student_t_regression_chunked_no_loops_over_N.stan: every log density and elementwise function on a
// vector or matrix is BayesMVP's fast_ function, declared in the functions block; calls on single
// reals stay on Stan math. One text for both instruction sets (NicoStan's AVX2 or AVX-512 make
// arguments).


// robust_Student_t_regression_chunked_no_loops_over_N.stan
// Tape-chunked version of robust_Student_t_regression.stan with no loop over the observations: the log density
// and its chunks as in robust_Student_t_regression_chunked.stan (below; already on whole vectors, with nothing
// whose size grows with N outside the chunks), and the generated quantities computed on whole vectors too: each
// observation's log likelihood, student_t_lpdf(y[n] | nu, mu[n], sigma), from the closed form of the Student-t
// log density, and the replicates drawn by student_t_rng(nu, mu, sigma) on the whole vector (in the same order).
//
// The header of the tape-chunked model this one comes from (the next 10 lines):
// robust_Student_t_regression_chunked.stan
// Tape-chunked version of robust_Student_t_regression.stan (robust t4 regression): the same log density, with the
// likelihood summed over the observations by reduce_sum_static, in chunks of chunk_size observations (data), so
// that each chunk's autodiff tape is built, differentiated and released before the next one (the
// chunking paper's tape
// chunking; N_chunks from NicoStan::fn_compute_initial_N_chunks()). Each chunk computes its observations' linear
// predictor and terms with the whole-vector operations of robust_Student_t_regression.stan; mu, a transformed
// parameter there, is reported as a generated quantity (the same values).
// The chunking data are NicoStan's for Stan models: N (the observations, the units of the chunks; already in the
// data) and chunk_size, which NicoStan sets to ceiling(N / N_chunks) in each phase (num_chunks_burnin,
// num_chunks_sampling).

functions {
    real bmvp_simd_lanes();
    vector nico_log1p(vector x);
    real nico_normal_lpdf(vector y, real mu, real sigma);
    real nico_student_t_lpdf(array[] real y, real nu, vector mu, real sigma);
    // real partial_sum_observations_lpdf(array[] real y_slice, int start, int end,
    //                                    matrix X, real nu, real alpha, vector beta, real sigma) {
    //     return student_t_lupdf(y_slice | nu, alpha + X[start:end] * beta, sigma);
    // }
    // (X and nu declared as data, as they are)
    real partial_sum_observations_lpdf(array[] real y_slice, int start, int end,
                                       data matrix X_with_ones, data real nu, real alpha, vector beta,
                                       real sigma) {
        // return nico_student_t_lupdf(y_slice | nu, alpha + X[start:end] * beta, sigma);
        // (the same values, with alpha folded into the coefficients: alpha + X * beta =
        //  [1, X] * append_row(alpha, beta), so that each observation's location takes one product, in place
        //  of a product and an addition per observation)
        return nico_student_t_lupdf(y_slice | nu, X_with_ones[start:end] * append_row(alpha, beta), sigma);
    }
}


data {
    int<lower=1> N;                    // number of observations
    int<lower=1> K;                    // number of predictors
    matrix[N, K] X;                    // design matrix (without intercept)
    vector[N] y;                       // response variable
    real<lower=1> nu;                  // degrees of freedom (fixed at 4)
    int<lower=1> chunk_size;           // observations per chunk (reduce_sum_static)
}


transformed data {
    array[N] real y_array = to_array_1d(y);   // the responses, the sliced argument of reduce_sum_static
    // the covariates with a first column of ones, so that alpha is the first coefficient of the products
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
    beta ~ nico_normal(0, 2.5);
    sigma ~ exponential(1);

    // Likelihood with t-distributed errors (reduce_sum_static over the observations, in chunks of chunk_size;
    // each chunk's linear predictor is alpha + X * beta at its observations)
    // target += reduce_sum_static(partial_sum_observations_lupdf, y_array, chunk_size, X, nu, alpha, beta,
    //                             sigma);
    target += reduce_sum_static(partial_sum_observations_lupdf, y_array, chunk_size, X_with_ones, nu, alpha, beta,
                                sigma);
}


generated quantities {
    real simd_lanes = bmvp_simd_lanes();
    // (mu: the transformed parameter of robust_Student_t_regression.stan, the same values)
    vector[N] mu = alpha + X * beta;   // linear predictor
    vector[N] log_lik;
    vector[N] y_rep;

    // for (n in 1:N) {
    //   log_lik[n] = student_t_lpdf(y[n] | nu, mu[n], sigma);
    //   y_rep[n]   = student_t_rng(nu, mu[n], sigma);
    // }
    // (on the whole vectors: each observation's log likelihood, student_t_lpdf(y[n] | nu, mu[n], sigma), is
    //  lgamma((nu + 1) / 2) - lgamma(nu / 2) - log(nu pi) / 2 - log(sigma)
    //  - (nu + 1) / 2 log1p(((y[n] - mu[n]) / sigma)^2 / nu), and student_t_rng(nu, mu, sigma) draws the
    //  replicates in the same order, each as student_t_rng(nu, mu[n], sigma))
    log_lik = lgamma((nu + 1) / 2) - lgamma(nu / 2) - 0.5 * log(nu * pi()) - log(sigma)
              - (nu + 1) / 2 * nico_log1p(square((y - mu) / sigma) / nu);
    y_rep   = to_vector(student_t_rng(nu, mu, sigma));
}
