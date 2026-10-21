// Gaussian_process_regression_unchunked_no_loops_over_N_with_Stan_fused_functions.stan
// Unchunked Stan::math version of Gaussian_process_regression_chunked (the Stan math file of the
// chunking paper's layout
// timings): the same log density as that file at N_chunks = 1, and as Gaussian_process_regression.stan, without
// reduce_sum_static. The same data (chunk_size, when it is in the data, is not read) and the same unconstrained
// parameters, in the same order and with the same names (lengthscale, sigma_f, sigma_n), as both files.
// Stan's fused functions: gp_exp_quad_cov for the covariance matrix, add_diag for its two diagonal terms (no
// N x N diagonal matrix), cholesky_decompose, and multi_normal_cholesky for the likelihood (the triangular
// solve, the log determinant and the density in one node), as in Gaussian_process_regression.stan and
// Gaussian_process_regression_chunked_with_multi_normal_cholesky; its zero mean is made once in transformed data.
// The model block is that of Gaussian_process_regression.stan, which already has no loop over N; the one loop
// over the observations of that file, in gp_pred_rng() of its generated quantities, is replaced by add_diag, as
// in the chunked file (the same values).
// No loop over N anywhere.


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

        // (the diagonal terms square(sigma) + square(fixed_nugget_sd), added in the same order by add_diag,
        //  without the loop over the observations, as in the chunked file)
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
}


data {
    int<lower=2> N;          // at least two observations for standardisation
    vector[N] x;             // input locations
    vector[N] y;             // target variable
    int<lower=1> N_pred;     // number of prediction points
    vector[N_pred] x_pred;   // prediction input locations
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

    // real fixed_nugget_sd = 0.05; // independent diagonal noise in standardised outcome units, not a shared
    real fixed_nugget_sd = 1e-4;  // a numerical jitter only (a variance of 1e-8 on the diagonal)
                                 // intercept
    vector[N] zeros = rep_vector(0, N);   // the mean of multi_normal_cholesky
}


parameters {
    real<lower=0> lengthscale;   // lengthscale parameter
    real<lower=0> sigma_f;       // signal standard deviation
    real<lower=0> sigma_n;       // noise standard deviation
}


model {
    // Covariance matrix (both diagonal terms added by add_diag) and its Cholesky factor
    matrix[N, N] L_K = cholesky_decompose(add_diag(gp_exp_quad_cov(x_norm, sigma_f, lengthscale),
                                                   square(fixed_nugget_sd) + square(sigma_n)));

    // Priors
    lengthscale ~ inv_gamma(5, 5);
    sigma_f ~ std_normal();
    sigma_n ~ std_normal();

    // Likelihood (Stan's fused multivariate normal on the Cholesky factor)
    y_norm ~ multi_normal_cholesky(zeros, L_K);
}


generated quantities {
    // Predictions scaled back to original scale
    vector[N_pred] f_pred = gp_pred_rng(x_pred_norm, y_norm, x_norm,
                                        sigma_f, lengthscale, sigma_n, fixed_nugget_sd, 1e-9) * y_sd + y_mean;
    real sigma = sigma_n * y_sd;
}






















