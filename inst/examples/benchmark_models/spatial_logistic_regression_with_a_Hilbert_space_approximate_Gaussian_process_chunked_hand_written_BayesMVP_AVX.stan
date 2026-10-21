// spatial_logistic_regression_with_a_Hilbert_space_approximate_Gaussian_process_chunked_hand_written_
// BayesMVP_AVX.stan
// Hand-written BayesMVP AVX version of spatial_logistic_regression_with_a_Hilbert_space_approximate_Gaussian_
// process_chunked.stan (one text, compiled for AVX2 and for AVX-512: only the compile flags differ): every
// expensive call on a vector is BayesMVP's (nico_exp for the spectral density, nico_normal(0, 1) for the prior of
// z, and the fused nico_bernoulli_logit_glm for the likelihood of each chunk); everything else as in the chunked
// file. The header of the chunked file follows.
// Tape-chunked version of spatial_logistic_regression_with_a_Hilbert_space_approximate_Gaussian_process_
// unchunked_no_loops_over_N_with_Stan_fused_functions.stan (see its header for the model): the same log density,
// with the likelihood, Stan's fused logistic GLM, summed by reduce_sum_static over the people, in chunks of
// chunk_size people (Paper 1's tape chunking; N_chunks from NicoStan::fn_compute_initial_N_chunks()), so that
// each chunk's autodiff tape is built, differentiated and released before the next one.
// The model has no parameter whose size grows with N: its parameters are the M coefficients z of the basis
// functions, alpha, beta, sigma_gp and length_scale (M + 6 values), shared by every chunk; what grows with N is
// data (the outcomes, the sliced argument, and the rows of the design matrix of each chunk). The M coefficients
// of the GLM, sqrt(S) .* z, are made once on the outer tape (M values, not N) and shared.
// The chunking data are NicoStan's for Stan models: N (the people, the units of the chunks; already in the data)
// and chunk_size, which NicoStan sets to ceiling(N / N_chunks) in each phase (num_chunks_burnin,
// num_chunks_sampling).


functions {
    real bmvp_simd_lanes();
    vector nico_exp(vector x);
    real nico_normal_lpdf(vector y, real mu, real sigma);
    real nico_bernoulli_logit_glm_lpmf(array[] int y, matrix x, real alpha, vector beta);

    // the square root of the spectral density of the squared exponential kernel in two dimensions at the square
    // roots of the eigenvalues: sigma_gp sqrt(2 pi) length_scale exp(-length_scale^2 lambda / 4)
    vector square_root_of_spectral_density_of_squared_exponential_kernel_in_two_dimensions(real sigma_gp,
                                                                                           real length_scale,
                                                                                           vector eigenvalues) {
        return sigma_gp * sqrt(2 * pi()) * length_scale * nico_exp(-0.25 * square(length_scale) * eigenvalues);
    }

    // the log likelihood of the people start to end (one chunk of reduce_sum_static): the logistic GLM of their
    // outcomes, with their rows of the design matrix
    real partial_sum_people_lpmf(array[] int y_slice, int start, int end,
                                 data matrix design_matrix,
                                 real alpha,
                                 vector coefficients) {
        return nico_bernoulli_logit_glm_lupmf(y_slice | design_matrix[start:end], alpha, coefficients);
    }
}


data {
    int<lower=1> N;                           // number of people (the units of the chunks)
    matrix[N, 2] location;                    // where each person lives (two coordinates)
    matrix[N, 3] covariates;                  // age (standardised), sex (0 / 1), deprivation (standardised)
    array[N] int<lower=0, upper=1> y;         // the binary outcome (e.g. the disease present)
    int<lower=1> M_per_dimension;             // basis functions per dimension of the location
    real<lower=1> boundary_factor;            // the box of the basis functions: this times the data's half-range
    int<lower=1> chunk_size;                  // people per chunk (reduce_sum_static)
}


transformed data {
    int M = M_per_dimension * M_per_dimension;
    // the centred locations and the half-widths of the box
    matrix[N, 2] location_centred;
    vector[2] L;
    for (d in 1:2) {
        location_centred[, d] = location[, d] - (max(location[, d]) + min(location[, d])) / 2;
        L[d] = boundary_factor * max(abs(location_centred[, d]));
    }
    // the basis functions of each dimension (N x M_per_dimension each), then their products (N x M), with the
    // covariates in the first 3 columns of the design matrix of the GLM
    array[2] matrix[N, M_per_dimension] basis_of_dimension;
    for (d in 1:2) {
        for (m in 1:M_per_dimension) {
            basis_of_dimension[d][, m] = sin(pi() * m * (location_centred[, d] + L[d]) / (2 * L[d])) / sqrt(L[d]);
        }
    }
    vector[M] eigenvalues;
    matrix[N, 3 + M] design_matrix;
    design_matrix[, 1:3] = covariates;
    for (m_1 in 1:M_per_dimension) {
        for (m_2 in 1:M_per_dimension) {
            int j = (m_1 - 1) * M_per_dimension + m_2;
            eigenvalues[j] = square(pi() * m_1 / (2 * L[1])) + square(pi() * m_2 / (2 * L[2]));
            design_matrix[, 3 + j] = basis_of_dimension[1][, m_1] .* basis_of_dimension[2][, m_2];
        }
    }
}


parameters {
    vector[M] z;                              // the standard normal coefficients of the basis functions
    real alpha;                               // the intercept (log-odds at the average location and covariates)
    vector[3] beta;                           // the coefficients of the covariates
    real<lower=0> sigma_gp;                   // the marginal standard deviation of the spatial effect
    real<lower=0> length_scale;               // the length scale of the spatial effect (units of location)
}


model {
    alpha ~ normal(0, 2);
    beta ~ normal(0, 1);
    sigma_gp ~ normal(0, 1);
    length_scale ~ inv_gamma(5, 1);
    target += nico_normal_lupdf(z | 0, 1);

    // Likelihood: the logistic GLM, summed by reduce_sum_static over the people, in chunks of chunk_size; the
    // outcomes y (data) are the sliced argument, passed by name (a named variable is held by reference, an
    // expression would be copied at every split)
    vector[3 + M] coefficients
        = append_row(beta,
                     square_root_of_spectral_density_of_squared_exponential_kernel_in_two_dimensions(
                             sigma_gp, length_scale, eigenvalues) .* z);
    target += reduce_sum_static(partial_sum_people_lupmf, y, chunk_size, design_matrix, alpha, coefficients);
}


generated quantities {
    real simd_lanes = bmvp_simd_lanes();
}
