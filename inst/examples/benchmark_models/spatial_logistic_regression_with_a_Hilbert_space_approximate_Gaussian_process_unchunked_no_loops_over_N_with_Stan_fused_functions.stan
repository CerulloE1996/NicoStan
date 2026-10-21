// spatial_logistic_regression_with_a_Hilbert_space_approximate_Gaussian_process_unchunked_no_loops_over_N_with_
// Stan_fused_functions.stan
// A spatial logistic regression of individual-level data, as in spatial epidemiology (e.g. the presence of a
// disease in each person, by where they live): the log-odds of person n are
//     alpha + covariates[n] * beta + f(location[n]),
// with the covariates age (standardised), sex (0 / 1) and deprivation (standardised), and f a zero-mean Gaussian
// process of the two-dimensional location, with the squared exponential kernel (sigma_gp, length_scale), in its
// Hilbert space approximation (Solin and Saerkkae, 2020; Riutort-Mayol et al., 2023; brms's gp(..., k = ...)):
//     f(s) = sum_j phi_j(s) sqrt(S(sqrt(lambda_j))) z_j,   z_j ~ normal(0, 1),
// with, on the box [-L_1, L_1] x [-L_2, L_2] around the centred locations (L_d = boundary_factor times the
// largest |centred coordinate| of dimension d), the M = M_per_dimension^2 basis functions
//     phi_j(s) = prod_d sin(pi m_jd (s_d + L_d) / (2 L_d)) / sqrt(L_d),
// their eigenvalues lambda_j = sum_d (pi m_jd / (2 L_d))^2 (m_jd = 1, ..., M_per_dimension), and the spectral
// density of the squared exponential kernel in two dimensions, S(w) = sigma_gp^2 2 pi length_scale^2
// exp(-length_scale^2 w^2 / 2). The basis functions are data (made once in transformed data, N x M), so the
// whole likelihood is ONE logistic GLM with the covariates and the basis functions as its columns, and the
// coefficients beta and sqrt(S) .* z: Stan's fused bernoulli_logit_glm. Its cost grows as N x (3 + M), not N^3.
// Unchunked plain Stan version (the reference of the chunked version and of its AVX versions). No loop over N
// anywhere (the loop over the M basis functions makes their columns, each a whole-vector operation, once).


functions {
    // the square root of the spectral density of the squared exponential kernel in two dimensions at the square
    // roots of the eigenvalues: sigma_gp sqrt(2 pi) length_scale exp(-length_scale^2 lambda / 4)
    vector square_root_of_spectral_density_of_squared_exponential_kernel_in_two_dimensions(real sigma_gp,
                                                                                           real length_scale,
                                                                                           vector eigenvalues) {
        return sigma_gp * sqrt(2 * pi()) * length_scale * exp(-0.25 * square(length_scale) * eigenvalues);
    }
}


data {
    int<lower=1> N;                           // number of people (the units of the chunks)
    matrix[N, 2] location;                    // where each person lives (two coordinates)
    matrix[N, 3] covariates;                  // age (standardised), sex (0 / 1), deprivation (standardised)
    array[N] int<lower=0, upper=1> y;         // the binary outcome (e.g. the disease present)
    int<lower=1> M_per_dimension;             // basis functions per dimension of the location
    real<lower=1> boundary_factor;            // the box of the basis functions: this times the data's half-range
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
    z ~ std_normal();

    // Likelihood: one logistic GLM with the covariates and the basis functions (see the header)
    vector[3 + M] coefficients
        = append_row(beta,
                     square_root_of_spectral_density_of_squared_exponential_kernel_in_two_dimensions(
                             sigma_gp, length_scale, eigenvalues) .* z);
    y ~ bernoulli_logit_glm(design_matrix, alpha, coefficients);
}
