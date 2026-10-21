// Gaussian_process_classification_chunked_hand_written_BayesMVP_AVX.stan
// Hand-written AVX version of models_staging_chunked_Gaussian_process_classification/Gaussian_process_classification_chunked.stan: every log density and elementwise function on a
// vector or matrix is BayesMVP's fast_ function, declared in the functions block; calls on single
// reals stay on Stan math. One text for both instruction sets (NicoStan's AVX2 or AVX-512 make
// arguments).
// Gaussian_process_classification_chunked.stan
// Tape-chunked version of Gaussian_process_classification.stan: the same log density, with every term per
// observation inside reduce_sum_static (the tape-chunking paper's chunking; N_chunks from
// NicoStan::fn_compute_initial_N_chunks()), so that each chunk's autodiff tape is built, differentiated and
// released before the next one. As in the reduce_sum_static LC-MVP (LC_MVP_bin_PartialLog_v5_reduce_sum_static
// .stan), everything per observation that is a parameter is packed into the sliced argument, and the shared
// arguments are only data: the units of the chunks are the observations, and observation n's slice holds its
// latent log-odds f[n] and its nuisance parameter eta[n] (to_array_2d(append_col(f, eta)), an N x 2 array), so
// that each chunk copies only its own observations' values onto its tape; the outcomes y are shared as data
// (not copied). Each chunk adds, with whole-block operations (no loop over its observations), the
// Bernoulli-logit log-likelihood of its outcomes and its share of the nuisance block's prior,
// eta[start:end] ~ std_normal().
// The covariance matrix, its Cholesky factorisation and f = mu + L_K * eta stay on the outer tape, exactly as in
// Gaussian_process_classification.stan: the Cholesky factorisation of the dense N x N covariance is not a sum of
// terms over the observations (each row of L_K depends on every row of the covariance before it), so it cannot be
// split into partial sums, and the covariance is its input, which cannot be built inside the chunks (a partial
// sum of reduce_sum_static returns one real, and each chunk's tape is released after the chunk); f needs the
// whole of L_K, which is already on the outer tape, so it is computed there once, with one matrix-vector product
// (rather than slicing the rows of L_K, which would also need the whole of eta as a shared argument in every
// chunk).
// The chunking data are NicoStan's for Stan models: N (the observations, the units of the chunks; already in the
// data) and chunk_size, which NicoStan sets to ceiling(N / N_chunks) in each phase (num_chunks_burnin,
// num_chunks_sampling).

functions {
    real bmvp_simd_lanes();
    real nico_normal_lpdf(vector y, real mu, real sigma);
    real nico_bernoulli_logit_lpmf(array[] int y, vector f);
    matrix nico_gp_exp_quad_cov(array[] real x, real sigma, real length_scale);
    //
    // The log density of the observations start to end (one chunk of reduce_sum_static): the Bernoulli-logit
    // log-likelihood of their outcomes (y ~ bernoulli_logit(f) in Gaussian_process_classification.stan) and their
    // share of the nuisance block's prior (eta ~ std_normal() there). Column 1 of the slice holds their f, and
    // column 2 their eta.
    //
    real partial_sum_observations_lpdf(array[,] real f_and_eta_slice, int start, int end,
                                       data array[] int y) {
        matrix[end - start + 1, 2] f_and_eta = to_matrix(f_and_eta_slice);
        return nico_normal_lupdf(f_and_eta[, 2] | 0, 1) + nico_bernoulli_logit_lupmf(y[start:end] | f_and_eta[, 1]);
    }
}


data {
    int<lower=1> N;                           // number of observations (the units of the chunks)
    array[N] real x;                          // input (e.g. a standardised biomarker)
    array[N] int<lower=0, upper=1> y;         // binary outcome (e.g. disease status)
    int<lower=1> chunk_size;                  // observations per chunk (reduce_sum_static)
}


parameters {
    vector[N] eta;                            // standard normal nuisance block, declared first
    real mu;                                  // constant mean of the latent log-odds
    real<lower=0> lengthscale;                // lengthscale of the squared-exponential covariance
    real<lower=0> sigma_f;                    // marginal standard deviation of the latent log-odds
}


model {
    matrix[N, N] L_K = cholesky_decompose(add_diag(nico_gp_exp_quad_cov(x, sigma_f, lengthscale), 1e-8));
    vector[N] f = mu + L_K * eta;
    mu ~ normal(0, 1);
    lengthscale ~ inv_gamma(5, 5);
    sigma_f ~ normal(0, 1.5);

    // Likelihood and the nuisance block's prior (reduce_sum_static over the observations, in chunks of
    // chunk_size; each observation's f and eta sliced). The sliced argument is a named local variable, which
    // reduce_sum_static holds by reference; an expression passed in its place is held by value and copied at
    // every split of the partition into chunks, i.e. the whole N x 2 array copied about N_chunks times per
    // gradient.
    array[N, 2] real f_and_eta_by_observation = to_array_2d(append_col(f, eta));
    target += reduce_sum_static(partial_sum_observations_lupdf, f_and_eta_by_observation, chunk_size, y);
}


generated quantities {
    real simd_lanes = bmvp_simd_lanes();
}
