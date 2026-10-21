


// Gaussian_process_regression_chunked.stan
// Tape-chunked version of Gaussian_process_regression.stan (Gaussian process regression, marginal likelihood):
// the same log density, with the likelihood written as the sum over the observations of its terms and summed by
// reduce_sum_static, in chunks of chunk_size observations (data), so that each chunk's autodiff tape is built,
// differentiated and released before the next one (the tape-chunking paper's chunking; N_chunks from
// NicoStan::fn_compute_initial_N_chunks()).
// The likelihood of Gaussian_process_regression.stan, multi_normal_cholesky_lpdf(y_norm | 0, L_K), equals
//     sum over the observations n of  std_normal_lpdf(z[n]) - log(L_K[n, n]),   with z = L_K \ y_norm,
// since log det(L_K * L_K') = 2 * sum(log(diagonal(L_K))) and y_norm' * inverse(L_K * L_K') * y_norm = z' * z.
// Each chunk computes these terms of its observations with whole-vector operations; its observations' z and
// diagonal of L_K are the sliced argument of reduce_sum_static, packed per observation as the rows of an
// N x 2 array (column 1: z; column 2: the diagonal of L_K), so that no O(N) parameter is shared by the chunks.
// What stays on the outer tape, as whole-matrix operations (no loop over the observations), and why it cannot be
// split over the observations by reduce_sum_static, which returns only the sum of its chunks' values:
//   (1) the covariance matrix K = gp_exp_quad_cov(x_norm, sigma_f, lengthscale) plus its two diagonal terms:
//       the N x N input of the Cholesky factorisation, which needs all of it at once, and no chunk can return a
//       block of K to the outer tape;
//   (2) the Cholesky factorisation L_K of K: every column of L_K depends on every column before it;
//   (3) the forward substitution z = L_K \ y_norm: z[n] depends on z[1:(n - 1)], i.e. the block of z of each
//       chunk needs the blocks of every chunk before it.
// The priors on lengthscale, sigma_f and sigma_n (3 values) are also on the outer tape.
// The chunking data are NicoStan's for Stan models: N (the observations, the units of the chunks; already in the
// data) and chunk_size, which NicoStan sets to ceiling(N / N_chunks) in each phase (num_chunks_burnin,
// num_chunks_sampling).


// gaussian_process.stan


functions {
    vector gp_pred_rng(array[] real x2,
                       vector y1,
                       array[] real x1,
                       real sigma_f,
                       real lengthscale_f,
                       real sigma,
                       real fixed_nugget_sd,
                       real jitter) {
      int N1 = rows(y1);
      int N2 = size(x2);
      vector[N2] f2;
      {
        matrix[N1, N1] L_K;
        vector[N1] K_div_y1;
        matrix[N1, N2] k_x1_x2;
        matrix[N1, N2] v_pred;
        vector[N2] f2_mu;
        matrix[N2, N2] cov_f2;
        matrix[N1, N1] K;

        // K = gp_exp_quad_cov(x1, sigma_f, lengthscale_f);
        // for (n in 1:N1) {
        //   K[n, n] = K[n,n] + square(sigma) + square(fixed_nugget_sd);
        // }
        // (the same diagonal terms, added in the same order by add_diag, without the loop over the observations)
        K = add_diag(add_diag(gp_exp_quad_cov(x1, sigma_f, lengthscale_f), square(sigma)),
                     square(fixed_nugget_sd));
        L_K = cholesky_decompose(K);
        K_div_y1 = mdivide_left_tri_low(L_K, y1);
        K_div_y1 = mdivide_right_tri_low(K_div_y1', L_K)';
        k_x1_x2 = gp_exp_quad_cov(x1, x2, sigma_f, lengthscale_f);
        f2_mu = (k_x1_x2' * K_div_y1);
        v_pred = mdivide_left_tri_low(L_K, k_x1_x2);
        cov_f2 = gp_exp_quad_cov(x2, sigma_f, lengthscale_f) - v_pred' * v_pred;
        f2 = multi_normal_rng(f2_mu, add_diag(cov_f2, rep_vector(jitter, N2)));
      }
      return f2;
    }

    // the likelihood terms of the chunk's observations (start to end): std_normal_lpdf(z[n]) - log(L_K[n, n]),
    // from the chunk's rows of the packed N x 2 array (column 1: z; column 2: the diagonal of L_K)
    real partial_sum_observations_lpdf(array[,] real z_and_L_K_diagonal_slice, int start, int end) {
        vector[end - start + 1] z_slice = to_vector(z_and_L_K_diagonal_slice[:, 1]);
        vector[end - start + 1] L_K_diagonal_slice = to_vector(z_and_L_K_diagonal_slice[:, 2]);
        return std_normal_lupdf(z_slice) - sum(log(L_K_diagonal_slice));
    }
}


data {
    int<lower=2> N;          // at least two observations for standardisation
    vector[N] x;             // input locations
    vector[N] y;             // target variable
    int<lower=1> N_pred;     // number of prediction points
    vector[N_pred] x_pred;   // prediction input locations
    int<lower=1> chunk_size; // observations per chunk (reduce_sum_static)
}


transformed data {
    // Normalize data
    real x_mean = mean(x);
    real y_mean = mean(y);
    real x_sd = sd(x);
    real y_sd = sd(y);

    if (x_sd <= 0 || y_sd <= 0) {
      reject("Gaussian_process requires non-constant x and y for standardisation.");
    }

    array[N] real x_norm = to_array_1d((x - x_mean) / x_sd);
    array[N_pred] real x_pred_norm = to_array_1d((x_pred - x_mean) / x_sd);
    vector[N] y_norm = (y - y_mean) / y_sd;

    // real fixed_nugget_sd = 0.05; // independent diagonal noise in standardised outcome units, not a shared intercept
    real fixed_nugget_sd = 1e-4;  // a numerical jitter only (a variance of 1e-8 on the diagonal)
    // vector[N] zeros = rep_vector(0, N);
    // (zeros: the mean of multi_normal_cholesky in Gaussian_process_regression.stan; the terms of the chunks
    //  below take the mean 0 directly)
}


parameters {
    real<lower=0> lengthscale;   // lengthscale parameter
    real<lower=0> sigma_f;       // signal standard deviation
    real<lower=0> sigma_n;       // noise standard deviation
}


model {
    // Covariance matrix (both diagonal terms added by add_diag, without the N x N diag_matrix whose
    // addition made N^2 more autodiff variables; NicoStan's example, Gaussian_process.stan, kept below,
    // commented out)
    matrix[N, N] L_K = cholesky_decompose(add_diag(gp_exp_quad_cov(x_norm, sigma_f, lengthscale),
                                                   square(fixed_nugget_sd) + square(sigma_n)));
    // matrix[N, N] K = gp_exp_quad_cov(x_norm, sigma_f, lengthscale) +
    //                  diag_matrix(rep_vector(square(fixed_nugget_sd), N));
    // matrix[N, N] L_K = cholesky_decompose(add_diag(K, square(sigma_n)));

    // Priors
    lengthscale ~ inv_gamma(5, 5);
    sigma_f ~ std_normal();
    sigma_n ~ std_normal();

    // Likelihood
    // y_norm ~ multi_normal_cholesky(zeros, L_K);
    // (the same log density: the forward substitution z = L_K \ y_norm on the outer tape, then the terms
    //  std_normal_lpdf(z[n]) - log(L_K[n, n]) summed by reduce_sum_static over the observations, in chunks of
    //  chunk_size; z and the diagonal of L_K packed per observation as the rows of an N x 2 array, the sliced
    //  argument; see the header)
    {
      vector[N] z = mdivide_left_tri_low(L_K, y_norm);
      array[N, 2] real z_and_L_K_diagonal = to_array_2d(append_col(z, diagonal(L_K)));
      target += reduce_sum_static(partial_sum_observations_lupdf, z_and_L_K_diagonal, chunk_size);
    }
}


generated quantities {
    // Predictions scaled back to original scale
    vector[N_pred] f_pred = gp_pred_rng(x_pred_norm, y_norm, x_norm,
                                        sigma_f, lengthscale, sigma_n, fixed_nugget_sd, 1e-9) * y_sd + y_mean;
    real sigma = sigma_n * y_sd;
}

