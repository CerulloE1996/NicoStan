////
//// ---- One-class multivariate ordered probit (MVOP) model, with NicoSIMD's fused GHK log likelihood -----------
////
//// N subjects and T outcomes (tests), each binary (y = 1 when the latent value is above 0) or ordinal (category
//// k when the latent value is between the cutpoints k - 1 and k). The latent values of a subject are
//// multivariate normal, with mean X_t[i, ] * beta[, t] for outcome t (each outcome with its OWN covariates, the
//// first being the intercept) and correlation matrix Omega = L_Omega L_Omega' (i.e., a truncated multivariate
//// normal for each subject, whose truncation region is set by the subject's results). A missing result (y = -1)
//// is handled by the GHK method too: its latent value is the untruncated GHK draw, with no likelihood term.
////
//// The log likelihood of each chunk of subjects is ONE call of
//// nico_finite_mixture_of_multivariate_ordered_probit_GHK_log_likelihood, which computes the GHK recursion, and
//// its gradient, in one autodiff node, with NicoSIMD's AVX-512 or AVX2 functions (x86-64) or with NEON (64-bit
//// ARM, e.g. Apple Silicon, through SIMDe). The same function also takes 2 or 4 latent classes,
//// interval-censored ordinal results and either Phi_type. The nuisance block u_raw is the sliced argument of
//// reduce_sum_static, so that everything whose size grows with N is inside the chunks.
////
//// The priors are those of BayesMVP's MVOP model (BayesMVP::MVP_model$new(Model_type = "MVOP")): normal priors
//// on the coefficients, an LKJ prior on the correlation matrix and an induced Dirichlet prior on the cutpoints
//// of each ordinal test; with the same values, this model and BayesMVP's manual-gradient MVOP have the same
//// posterior.
////
functions {
      ////
      //// ---- NicoSIMD's fused GHK log likelihood (declared without a body: its C++ is NicoSIMD's user header)
      ////
      real nico_finite_mixture_of_multivariate_ordered_probit_GHK_log_likelihood(
            array[] row_vector u_raw, data int nuisance_transformation, matrix intercept,
            array[] matrix coefficients, data array[,] int coefficient_set_of_class_and_test,
            data array[,] matrix covariates, data array[,] int n_covariates, array[] vector cutpoints,
            data array[,] int cutpoint_set_of_class_and_test,
            data array[] int position_of_first_cutpoint_of_test,
            array[] matrix L_Omega, matrix log_class_probability, data array[,] int category,
            data array[,] int upper_category, data array[] int test_is_binary,
            data array[] int n_categories_of_test, data array[] int test_at_GHK_position, data int Phi_type,
            data real overflow_threshold, data real underflow_threshold);
      ////
      //// ---- the partial sum of reduce_sum_static: the log likelihood of the subjects start, ..., end (with the
      //// log Jacobian of u = Phi(u_raw)); their covariates and results are indexed in the call itself, so that
      //// they stay data
      ////
      real partial_sum_of_the_GHK_log_likelihood_lpdf( array[] row_vector u_raw_slice, int start, int end,
                                                       matrix intercept, array[] matrix coefficients,
                                                       data array[,] int set_of_class_and_test,
                                                       data array[,] matrix covariates,
                                                       data array[,] int n_covariates, array[] vector cutpoints,
                                                       data array[] int position_of_first_cutpoint_of_test,
                                                       array[] matrix L_Omega, data array[,] int y,
                                                       data array[] int test_is_binary,
                                                       data array[] int n_categories_of_test,
                                                       data array[] int test_at_GHK_position, data int Phi_type,
                                                       data real overflow_threshold,
                                                       data real underflow_threshold) {
            matrix[end - start + 1, 1] log_class_probability = rep_matrix(0.0, end - start + 1, 1);
            return nico_finite_mixture_of_multivariate_ordered_probit_GHK_log_likelihood(
                         u_raw_slice, 1, intercept, coefficients, set_of_class_and_test,
                         covariates[:, :, start:end], n_covariates, cutpoints, set_of_class_and_test,
                         position_of_first_cutpoint_of_test, L_Omega, log_class_probability,
                         y[start:end], y[start:end], test_is_binary, n_categories_of_test, test_at_GHK_position,
                         Phi_type, overflow_threshold, underflow_threshold);
      }
      ////
      //// ---- the induced Dirichlet prior on an ordinal test's cutpoints (probit link), anchored at "anchor":
      //// the Dirichlet prior on the category probabilities Phi(cutpoints - anchor), times its Jacobian
      ////
      real induced_Dirichlet_probit_lpdf(vector cutpoints, vector alpha, real anchor) {
            int n_cutpoints = num_elements(cutpoints);
            vector[n_cutpoints] cumulative_probability = Phi(cutpoints - anchor);
            vector[n_cutpoints + 1] category_probability =  append_row(cumulative_probability, 1.0)
                                                          - append_row(0.0, cumulative_probability);
            return dirichlet_lpdf(category_probability | alpha) + std_normal_lpdf(cutpoints - anchor);
      }
}
data {
      int<lower=1> N;                                       //// the subjects
      int<lower=1> n_tests;                                 //// the outcomes, binary and ordinal, in any order
      array[n_tests] int<lower=0, upper=1> test_is_binary;  //// 1: binary; 0: ordinal
      array[n_tests] int<lower=2> n_categories_of_test;     //// 2 if binary; K_t (at least 3) if ordinal
      array[N, n_tests] int<lower=-1> y;                    //// binary: 0 or 1; ordinal: 1, ..., K_t; missing: -1
      int<lower=1> n_covariates_max;                        //// the most covariates of an outcome (intercept too)
      array[n_tests] int<lower=1, upper=n_covariates_max> n_covariates_of_test;
      array[n_tests] matrix[N, n_covariates_max] X;         //// each outcome's covariates in its first columns
                                                            //// (the first column 1, the intercept), then 0
      matrix[n_covariates_max, n_tests] prior_beta_mean;
      matrix<lower=0>[n_covariates_max, n_tests] prior_beta_sd;
      real<lower=0> prior_LKJ_eta;                          //// the LKJ prior on the correlation matrix
      real<lower=0> prior_Dirichlet_alpha;                  //// the induced Dirichlet prior on the cutpoints
      real C_raw_lower;                                     //// the bounds of the first cutpoint of an ordinal
      real<lower=C_raw_lower> C_raw_upper;                  //// outcome, and of the logs of its gaps
      int<lower=1, upper=2> Phi_type;                       //// 1: the exact Phi; 2: Phi_approx
      real overflow_threshold;                              //// the bounds of the log-scale tails
      real underflow_threshold;
      int<lower=1> chunk_size;                              //// the subjects of each chunk (set by NicoStan)
}
transformed data {
      ////
      //// ---- where each ordinal outcome's cutpoints are, in the one vector of every ordinal outcome's cutpoints
      ////
      int n_cutpoints_total = 0;
      array[n_tests] int position_of_first_cutpoint_of_test = rep_array(1, n_tests);
      for (t in 1:n_tests) {
            if (test_is_binary[t] == 0) {
                  position_of_first_cutpoint_of_test[t] = n_cutpoints_total + 1;
                  n_cutpoints_total += n_categories_of_test[t] - 1;
            }
      }
      ////
      //// ---- the coefficients of every outcome, one after the other, and their priors
      ////
      int n_coefficients_total = sum(n_covariates_of_test);
      vector[n_coefficients_total] prior_beta_mean_vec;
      vector[n_coefficients_total] prior_beta_sd_vec;
      {
            int position = 1;
            for (t in 1:n_tests) {
                  int n_t = n_covariates_of_test[t];
                  prior_beta_mean_vec[position:(position + n_t - 1)] = prior_beta_mean[1:n_t, t];
                  prior_beta_sd_vec[position:(position + n_t - 1)] = prior_beta_sd[1:n_t, t];
                  position += n_t;
            }
      }
      ////
      //// ---- one class: one coefficient set, one covariate set and one cutpoint set; GHK order = the data order
      ////
      array[1, n_tests] int set_of_class_and_test = rep_array(1, 1, n_tests);
      array[1, n_tests] matrix[N, n_covariates_max] covariates;
      array[1, n_tests] int n_covariates;
      for (t in 1:n_tests) {
            covariates[1, t] = X[t];
            n_covariates[1, t] = n_covariates_of_test[t];
      }
      matrix[1, n_tests] intercept_of_the_class = rep_matrix(0.0, 1, n_tests);  //// (in the coefficients instead)
      array[n_tests] int test_at_GHK_position = linspaced_int_array(n_tests, 1, n_tests);
}
parameters {
      array[N] row_vector[n_tests] u_raw;     //// the nuisance block (declared first): one GHK value per result
      vector[n_coefficients_total] beta_vec;  //// the coefficients of each outcome (its intercept first)
      cholesky_factor_corr[n_tests] L_Omega;
      vector<lower=C_raw_lower, upper=C_raw_upper>[n_cutpoints_total] C_raw;   //// first cutpoint, then log gaps
}
transformed parameters {
      ////
      //// ---- the coefficient matrix (outcome t's coefficients in the first rows of column t, then 0)
      ////
      matrix[n_covariates_max, n_tests] beta = rep_matrix(0.0, n_covariates_max, n_tests);
      {
            int position = 1;
            for (t in 1:n_tests) {
                  int n_t = n_covariates_of_test[t];
                  beta[1:n_t, t] = beta_vec[position:(position + n_t - 1)];
                  position += n_t;
            }
      }
      ////
      //// ---- the ordered cutpoints of each ordinal outcome: the first, then the cumulative sums of the gaps
      ////
      vector[n_cutpoints_total] cutpoints = C_raw;
      for (t in 1:n_tests) {
            if (test_is_binary[t] == 0) {
                  int first = position_of_first_cutpoint_of_test[t];
                  int n_gaps = n_categories_of_test[t] - 2;
                  vector[n_gaps] gaps = exp(C_raw[(first + 1):(first + n_gaps)]);
                  cutpoints[(first + 1):(first + n_gaps)] = C_raw[first] + cumulative_sum(gaps);
            }
      }
}
model {
      ////
      //// ---- priors (BayesMVP's MVOP priors)
      ////
      beta_vec ~ normal(prior_beta_mean_vec, prior_beta_sd_vec);
      L_Omega ~ lkj_corr_cholesky(prior_LKJ_eta);
      for (t in 1:n_tests) {
            if (test_is_binary[t] == 0) {
                  int first = position_of_first_cutpoint_of_test[t];
                  int n_gaps = n_categories_of_test[t] - 2;
                  target += induced_Dirichlet_probit_lpdf(cutpoints[first:(first + n_gaps)] |
                                                          rep_vector(prior_Dirichlet_alpha, n_gaps + 2),
                                                          prior_beta_mean[1, t]);
                  target += sum(C_raw[(first + 1):(first + n_gaps)]);    //// the Jacobian of the exp() gaps
            }
      }
      ////
      //// ---- the log likelihood, in chunks of chunk_size subjects (u_raw sliced), each chunk one fused call
      ////
      array[1] matrix[n_covariates_max, n_tests] coefficients_of_the_class = { beta };
      array[1] vector[n_cutpoints_total] cutpoints_of_the_class = { cutpoints };
      array[1] matrix[n_tests, n_tests] L_Omega_of_the_class = { L_Omega };
      target += reduce_sum_static( partial_sum_of_the_GHK_log_likelihood_lpdf, u_raw, chunk_size,
                                   intercept_of_the_class, coefficients_of_the_class, set_of_class_and_test,
                                   covariates, n_covariates, cutpoints_of_the_class,
                                   position_of_first_cutpoint_of_test, L_Omega_of_the_class, y,
                                   test_is_binary, n_categories_of_test, test_at_GHK_position, Phi_type,
                                   overflow_threshold, underflow_threshold);
}
generated quantities {
      matrix[n_tests, n_tests] Omega = multiply_lower_tri_self_transpose(L_Omega);
}
