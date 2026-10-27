// network_meta_analysis_Xu_fixed_cutpoints_unchunked_no_loops_over_N_with_Stan_fused_functions.stan
// Unchunked Stan::math version of network_meta_analysis_Xu_fixed_cutpoints.stan (MetaOrdDTA's
// DTA_NMA_Nyaga_Xu_FIXEDthr.stan, copied unchanged): the same model, the same log density (with every constant,
// as the original's binomial and induced-Dirichlet terms keep theirs), and the same parameters, in the same order
// and with the same names, with no loop over the studies, the study-test pairs or the reported cutpoints.
//
// The model (Nyaga et al. 2016's network meta-analysis of diagnostic test accuracy, with Xu's fixed between-study
// cutpoints, in MetaOrdDTA's parameterisation): for study s, index test t and class c (1 = non-diseased,
// 2 = diseased), the location of the latent (probit or logit) score is
//     Xbeta[t][s, c] = X_c[t][s, ] beta_mu[t][c, ]' + (beta_L_Sigma beta_eta_z[, s])[c]
//                      + beta_tau[t, c] beta_delta_z_vec[c][o],
// with o the study-test pair (s, t) counted study by study, test by test; the cutpoints C_c,t (fixed across the
// studies) are made from C_raw_vec with softplus (or exp) differences; at the study's i-th reported cutpoint
// k_i, the survival probability is S_i = Phi(Xbeta - C_c,t[k_i]) (inv_logit with the logit link), and the count
// above k_i is binomial(the count above k_{i - 1} (the class total for i = 1), S_i / S_{i - 1}) (S_0 = 1), the
// probability kept within [1e-10, 1 - 1e-10] as in the original's safe_binomial_lpmf.
//
// What differs from the original, and why:
//   - the likelihood is ONE vectorised binomial_lpmf (Stan's fused binomial log mass) per class over every
//     reported cutpoint of every study-test pair, on whole vectors (the original loops over the tests, studies
//     and cutpoints, with one scalar binomial_lpmf per cutpoint, and builds n_studies x n_thr matrices of every
//     test for every study, the studies without the test included);
//   - the locations are made once per study-test pair (n_study_test_pairs values per class), with
//     rows_dot_product for the covariates, beta_tau and the between-study effects read by index, and then read
//     at each reported cutpoint by index;
//   - Phi (or inv_logit) on the vector of every reported cutpoint; the original's safe_Phi(x) is Phi(x) for
//     every x that is not NaN (Stan's Phi is itself 0 below -37.5 and 1 above 8.25);
//   - the binomial probability clamp fmax(1e-10, fmin(1 - 1e-10, p)) on the whole vector (the original's, in the
//     same order, so that a NaN ratio gives 1 - 1e-10 as there);
//   - S_{i - 1} is the previous reported cutpoint's S of the same pair, read from the vector of every S shifted
//     by one, and 1 at each pair's first reported cutpoint (is_first_reported_cutpoint_of_its_pair);
//   - the induced-Dirichlet prior's sum of normal_lpdf(C[k] | 0, 1) over the cutpoints is std_normal_lpdf(C)
//     (the same value), and beta_sigma's two normal statements one vectorised statement.
// Stan has no fused function of this likelihood (binomial_logit_glm is for a logit-scale probability, and here
// the probability is a ratio of two survival probabilities), so binomial_lpmf on whole vectors is the fused
// function used.
// The indices of the study-test pairs and of the reported cutpoints are data, made once by
// make_model_and_data_network_meta_analysis_Xu_fixed_cutpoints.R from the original data (x, cutpoint_index,
// n_obs_cutpoints, indicator_index_test_in_study, X_nd and X_d), in the order of the studies sorted by their
// number of index tests (then by study, each study's tests in increasing order), which is the order of the
// chunked version's nuisance blocks; the original data are read too, unchanged.
// The generated quantities of the original are not computed here (they do not change the log density).
// No loop over the studies, the study-test pairs or the reported cutpoints anywhere; the loops left are over the
// index tests and the two classes, whose numbers do not grow with N.


functions {
        ////
        //// ---- MetaOrdDTA's functions used by the model (copied unchanged from inst/stan_functions/):
        ////
        //// (Stan_fns_ragged.stan)
        array[] int calculate_start_indices( array[] int n_thr,
                                             int n_tests) {

              array[n_tests] int start_index;

              start_index[1] = 1;
              for (t in 2:n_tests) {
                start_index[t] = start_index[t - 1] + n_thr[t - 1];
              }

              return start_index;

        }
        array[] int calculate_end_indices( array[] int n_thr,
                                           int n_tests,
                                           array[] int start_index) {

              array[n_tests] int end_index;

              for (t in 1:n_tests) {
                end_index[t] = start_index[t] + n_thr[t] - 1;
              }

              return end_index;

        }
        vector get_test_values( vector flat_values,
                                data array[] int start_index,
                                data array[] int end_index,
                                data int test_index) {

                int n_elements = end_index[test_index] - start_index[test_index] + 1;
                vector[n_elements] result;

                for (i in 1:n_elements) {
                  result[i] = flat_values[start_index[test_index] + i - 1];
                }

                return result;

        }
        //// (Stan_fns_ordinal.stan)
        vector construct_C(  vector C_raw_vec,
                             int softplus) {


                  int n_total_cutpoints = num_elements(C_raw_vec);
                  vector[n_total_cutpoints] C_vec;
                  //// ----  1st cutpoint (for each test) is unconstrained (no Jacobian needed):
                  C_vec[1] = C_raw_vec[1];
                  //// ---- Rest of cutpoints are made using LOG-differences (OR softplus-differences):
                  if (softplus == 1) {
                          vector[n_total_cutpoints - 1] softplus_C_vec = log1p_exp(C_raw_vec[2:n_total_cutpoints]);
                          for (k in 2:n_total_cutpoints) {
                                     C_vec[k] = C_vec[k - 1] + softplus_C_vec[k - 1];
                                     // jacobian += log_inv_logit(C_raw_vec[k]);   //// Jacobian for trans. C_raw -> C:
                          }
                  } else {
                          vector[n_total_cutpoints - 1] exp_C_vec = exp(C_raw_vec[2:n_total_cutpoints]);
                          for (k in 2:n_total_cutpoints) {
                                     C_vec[k] = C_vec[k - 1] + exp_C_vec[k - 1];
                                     // jacobian += C_raw_vec[k];  //// Jacobian for trans. C_raw -> C:
                          }
                  }
                  //// ---- Output cutpoints + Jacobian (as 1st element in output vector):
                  return(C_vec);

        }
        vector cumul_probs_to_ord_probs(vector cumul_probs) {

                int n_thr = num_elements(cumul_probs);
                int n_cat = n_thr + 1;
                vector[n_cat] ord_probs;

                ord_probs[1] = cumul_probs[1] - 0.0;
                for (k in 2:n_thr) {
                     ord_probs[k] = cumul_probs[k] - cumul_probs[k - 1]; // since probs are INCREASING with k
                }
                ord_probs[n_cat] =  1.0 - cumul_probs[n_cat - 1];

                return ord_probs;

        }
        //// (Stan_fns_corr.stan)
        matrix make_bivariate_L_Omega(real corr) {

                matrix[2, 2] L_Omega = rep_matrix(0.0, 2, 2);
                L_Omega[1, 1] = 1.0;
                L_Omega[2, 1] = corr;
                L_Omega[2, 2] = sqrt(1.0 - square(corr));
                return L_Omega;

        }
        //// (Stan_fns_Jacobian.stan)
        real raw_C_to_C_log_det_J_lp(   vector raw_C,
                                        int softplus) {

               int n_cutpoints = num_elements(raw_C);
               real log_det_J = 0.0;
               if (softplus == 1)  log_det_J += sum(raw_C[2:n_cutpoints]);
               else                log_det_J += sum(log_inv_logit(raw_C[2:n_cutpoints]));
               return log_det_J;

        }
}


data {
          int<lower=1> n_studies;
          int<lower=1> n_index_tests;
          ////
          array[n_index_tests] int<lower=1> n_studies_per_test;
          ////
          array[n_index_tests] int<lower=1> n_thr;
          array[n_index_tests] int<lower=2> n_cat;
          int<lower=1> n_thr_max; // corresponding to the test with the most thresholds/cutpoints.
          int<lower=2> n_cat_max; // corresponding to the test with the most thresholds/cutpoints.
          ////
          array[n_index_tests, n_studies] int n_obs_cutpoints; //// OBSERVED cutpoints for test t in study s
          ////
          array[n_studies, n_index_tests] int<lower=0, upper=1> indicator_index_test_in_study;   // Binary indicator if test t is in study s
          ////
          //// ---- Data:
          ////
          array[n_index_tests, 2, n_studies, n_thr_max + 1] int x;
          array[n_index_tests, 2, n_studies, n_thr_max + 1] int cutpoint_index;
          ////
          array[n_index_tests] int n_covariates_nd; // total INCLUDING ones w/ missing
          array[n_index_tests] int n_covariates_d;  // total INCLUDING ones w/ missing
          array[n_index_tests] matrix[n_studies, max(n_covariates_nd)] X_nd; // "observed" with 999's for missing
          array[n_index_tests] matrix[n_studies, max(n_covariates_d)]  X_d;   // "observed" with 999's for missing
          array[n_index_tests] vector[max(n_covariates_nd)] baseline_case_nd;
          array[n_index_tests] vector[max(n_covariates_d)]  baseline_case_d;
          ////
          //// ---- Priors for beta:
          ////
          array[n_index_tests] matrix[2, max(max(n_covariates_nd), max(n_covariates_d))] prior_beta_mu_mean;
          array[n_index_tests] matrix<lower=0.0>[2, max(max(n_covariates_nd), max(n_covariates_d))] prior_beta_mu_SD;
          matrix<lower=0.0>[n_index_tests, 2] prior_beta_tau_SD;
          vector<lower=0.0>[2] prior_beta_sigma_SD; //// "sigma's (Nyaga et al. notation) are shared between tests.
          ////
          //// Priors (and possible restrictions) for between-study correlations:
          ////
          real<lower=-1.0, upper=+1.0> beta_corr_lb;
          real<lower=beta_corr_lb, upper=+1.0>  beta_corr_ub;
          real<lower=0.0>  prior_beta_corr_LKJ;
          ////
          //// ---- Induced-Dirichlet priors:
          ////
          array[2] matrix<lower=0.0>[n_index_tests, n_thr_max + 1] prior_dirichlet_alpha;
          ////
          //// ---- Other:
          ////
          int<lower=0, upper=1> softplus;
          ////
          int use_probit_link;
          ////
          int hetero_sigma;
          int compound_symmetry;
          ////
          array[n_studies] int<lower=0, upper=1> K_fold_CV_indicator;
          ////
          //// ---- The same data, by study-test pair and by reported cutpoint (made once from the data above by
          ////      make_model_and_data_network_meta_analysis_Xu_fixed_cutpoints.R), in the order of the studies
          ////      sorted by their number of index tests (then by study; each study's tests in increasing order):
          ////
          array[4] int<lower=0> n_studies_with_each_number_of_index_tests;
          int<lower=1> n_study_test_pairs;
          array[n_study_test_pairs] int<lower=1, upper=n_studies> study_of_each_study_test_pair;
          array[n_study_test_pairs] int<lower=1, upper=n_index_tests> test_of_each_study_test_pair;
          //// (the pair's place in the original's order of beta_delta_z_vec[c]: study by study, test by test)
          array[n_study_test_pairs] int<lower=1, upper=n_study_test_pairs>
                                                 position_of_each_study_test_pair_in_the_original_order;
          //// (the pair's covariates X_nd[t][s, ] and X_d[t][s, ], 0 after the test's own n_covariates)
          matrix[n_study_test_pairs, max(max(n_covariates_nd), max(n_covariates_d))] X_nd_of_each_study_test_pair;
          matrix[n_study_test_pairs, max(max(n_covariates_nd), max(n_covariates_d))] X_d_of_each_study_test_pair;
          array[n_study_test_pairs] int<lower=1> first_reported_cutpoint_of_each_study_test_pair;
          array[n_study_test_pairs] int<lower=1> last_reported_cutpoint_of_each_study_test_pair;
          int<lower=1> n_reported_cutpoints;
          array[n_reported_cutpoints] int<lower=1, upper=n_study_test_pairs>
                                                 study_test_pair_of_each_reported_cutpoint;
          //// (1 at each pair's first reported cutpoint, 0 at the others)
          vector<lower=0, upper=1>[n_reported_cutpoints] is_first_reported_cutpoint_of_its_pair;
          //// (the place of the reported cutpoint's C in C_vec[c]: start_index[t] + k - 1)
          array[2, n_reported_cutpoints] int<lower=1, upper=sum(n_thr)>
                                                 position_in_C_vec_of_each_reported_cutpoint;
          //// (x_2[t, c, s, i] and n[t, c, s, i] of the original's transformed data)
          array[2, n_reported_cutpoints] int<lower=0> count_above_each_reported_cutpoint;
          array[2, n_reported_cutpoints] int<lower=0> count_above_the_previous_reported_cutpoint_or_class_total;
}

transformed data {
          ////
          int n_covariates_max = max(max(n_covariates_nd), max(n_covariates_d));
          ////
          int n_total_C_per_class_if_fixed = sum(n_thr); // total # of fixed-effect cutpoints
          int n_tests = n_index_tests;
          ////
          //// ---- Calculate indices:
          ////
          array[n_tests] int start_index = calculate_start_indices(n_thr, n_tests);
          array[n_tests] int end_index   = calculate_end_indices(n_thr, n_tests, start_index);
          ////
          //// (n_total_obs of the original: the number of study-test pairs)
          int n_total_obs = n_study_test_pairs;
          if (sum(to_array_1d(indicator_index_test_in_study)) != n_total_obs) {
              reject("n_study_test_pairs must be the number of 1s of indicator_index_test_in_study");
          }
          if (min(K_fold_CV_indicator) != 1) {
              reject("this version is for K_fold_CV_indicator = 1 in every study (the whole likelihood)");
          }
          ////
          //// (0 at each pair's first reported cutpoint, 1 at the others)
          vector[n_reported_cutpoints] is_not_first_reported_cutpoint_of_its_pair =
                1.0 - is_first_reported_cutpoint_of_its_pair;
}


parameters {
          matrix[2, n_studies] beta_eta_z;
          array[2] vector[n_total_obs] beta_delta_z_vec; //// Standard normal RVs for test-specific effects
          ////
          array[n_index_tests] matrix[2, n_covariates_max] beta_mu;
          ////
          array[2] vector<lower=-15, upper=2.5>[n_total_C_per_class_if_fixed] C_raw_vec;  //// RAW LOG-DIFFS - Global cutpoints for each test (staggered array/matrix using "n_thr[t]" to index correctly)
          ////
          //// ---- "NMA" params:
          ////
          vector<lower=0.0>[2] beta_sigma;  //// Var component - Between-study SD (Nyaga's σ)
          ////
          matrix<lower=0.0>[n_index_tests, 2] beta_tau_raw;
          ////
          real<lower=beta_corr_lb, upper=beta_corr_ub> beta_corr;  //// between-study corr (possibly restricted)
}


model {
        ////
        //// (beta_tau of the original's transformed parameters: with compound symmetry, every test's row is the
        ////  first test's)
        matrix[n_index_tests, 2] beta_tau;
        if (compound_symmetry) {
              beta_tau = rep_matrix(beta_tau_raw[1, ], n_index_tests);
        } else {
              beta_tau = beta_tau_raw;
        }
        ////
        //// ---- Construct (global) cutpoints:
        ////
        array[2] vector[n_total_C_per_class_if_fixed] C_vec;
        for (t in 1:n_index_tests) {
            for (c in 1:2) {
                  int n_thr_t = n_thr[t];
                  vector[n_thr_t] C_raw_vec_test_t = get_test_values(C_raw_vec[c], start_index, end_index, t);
                  vector[n_thr_t] C_vec_test_t =  construct_C(C_raw_vec_test_t, softplus);
                  C_vec[c][start_index[t]:end_index[t]] = C_vec_test_t;
            }
        }
        ////
        //// ---- Construct simple 2x2 (bivariate) between-study corr matrices:
        ////
        matrix[2, 2] beta_L_Omega = make_bivariate_L_Omega(beta_corr);
        matrix[2, 2] beta_L_Sigma = diag_pre_multiply(beta_sigma, beta_L_Omega);
        ////
        //// ---- Priors:
        ////
        for (t in 1:n_index_tests) {
            beta_mu[t][1, 1:n_covariates_nd[t]] ~ normal(
                                                  prior_beta_mu_mean[t][1, 1:n_covariates_nd[t]],
                                                  prior_beta_mu_SD[t][1, 1:n_covariates_nd[t]]);
            beta_mu[t][2, 1:n_covariates_d[t]]  ~ normal(
                                                  prior_beta_mu_mean[t][2, 1:n_covariates_d[t]],
                                                  prior_beta_mu_SD[t][2, 1:n_covariates_d[t]]);
        }
        ////
        to_vector(beta_tau_raw) ~ normal(0.0, to_vector(prior_beta_tau_SD));  //// delta_{s, i, t} ~ normal(0, tau_{i, t}):
        beta_L_Omega ~ lkj_corr_cholesky(prior_beta_corr_LKJ);
        ////
        //// (the original's two statements beta_sigma[c] ~ normal(0.0, prior_beta_sigma_SD[c]), as one)
        beta_sigma ~ normal(0.0, prior_beta_sigma_SD);
        ////
        //// ---- Induced-dirichlet ** Prior ** model (the original's induced_dirichlet_given_C_lpdf, whose terms
        ////      keep their constants: its sum over k of normal_lpdf(C[k] | 0, 1) is std_normal_lpdf(C)):
        ////
        for (c in 1:2) {
            for (t in 1:n_index_tests) {
                  int n_thr_t = n_thr[t];
                  int n_cat_t = n_thr_t + 1;
                  vector[n_thr_t] C_vec_t = C_vec[c][start_index[t]:end_index[t]];
                  vector[n_thr_t] Ind_Dir_cumul_prob = (use_probit_link == 1) ? Phi(C_vec_t) : inv_logit(C_vec_t);
                  vector[n_cat_t] Ind_Dir_ord_prob   = cumul_probs_to_ord_probs(Ind_Dir_cumul_prob);
                  if (use_probit_link == 1) {
                        target += std_normal_lpdf(C_vec_t);
                  } else {
                        target += sum(log_inv_logit(C_vec_t) + log1m_inv_logit(C_vec_t));
                  }
                  target += dirichlet_lpdf(Ind_Dir_ord_prob | to_vector(prior_dirichlet_alpha[c][t, 1:n_cat_t]));
            }
        }
        ////
        //// ---- Likelihood / Observational Model:
        ////
        target += std_normal_lpdf(to_vector(beta_eta_z));
        ////
        for (c in 1:2) {
          target += std_normal_lpdf(to_vector(beta_delta_z_vec[c])); // part of between-study model, NOT prior
        }
        ////
        for (t in 1:n_index_tests) {
          for (c in 1:2) {
               target += raw_C_to_C_log_det_J_lp(C_raw_vec[c][start_index[t]:end_index[t]], softplus);
          }
        }
        ////
        //// ---- Log-likelihood (each class: one vectorised binomial over every reported cutpoint):
        ////
        {
              //// (the between-study effects of every study, 2 x n_studies: the original's
              ////  beta_L_Sigma * beta_eta_z[1:2, s] of each study)
              matrix[2, n_studies] beta_eta = beta_L_Sigma * beta_eta_z;
              for (c in 1:2) {
                    //// (the covariates' coefficients of class c of every test, one row per test)
                    matrix[n_index_tests, n_covariates_max] beta_mu_of_class_c;
                    for (t in 1:n_index_tests) {
                          beta_mu_of_class_c[t, ] = beta_mu[t][c, ];
                    }
                    //// (the location Xbeta[t][s, c] of every study-test pair; the covariates are data, read in
                    ////  place)
                    vector[n_study_test_pairs] Xbeta_of_each_pair =
                          rows_dot_product((c == 1) ? X_nd_of_each_study_test_pair : X_d_of_each_study_test_pair,
                                           beta_mu_of_class_c[test_of_each_study_test_pair])
                          + beta_eta[c, study_of_each_study_test_pair]'
                          + beta_tau[test_of_each_study_test_pair, c]
                            .* beta_delta_z_vec[c][position_of_each_study_test_pair_in_the_original_order];
                    //// (the latent survival scale at every reported cutpoint, and its survival probability)
                    vector[n_reported_cutpoints] latent_surv =
                          Xbeta_of_each_pair[study_test_pair_of_each_reported_cutpoint]
                          - C_vec[c][position_in_C_vec_of_each_reported_cutpoint[c]];
                    vector[n_reported_cutpoints] surv_prob = (use_probit_link == 1) ? Phi(latent_surv) :
                                                                                      inv_logit(latent_surv);
                    //// (the previous reported cutpoint's survival probability of the same pair, 1 at the pair's
                    ////  first one)
                    vector[n_reported_cutpoints] surv_prob_of_previous_reported_cutpoint =
                          append_row(1.0, surv_prob[1:(n_reported_cutpoints - 1)])
                          .* is_not_first_reported_cutpoint_of_its_pair
                          + is_first_reported_cutpoint_of_its_pair;
                    vector[n_reported_cutpoints] cond_prob = surv_prob ./ surv_prob_of_previous_reported_cutpoint;
                    //// (the original's safe_binomial_lpmf: the probability within [1e-10, 1 - 1e-10])
                    vector[n_reported_cutpoints] cond_prob_safe = fmax(1e-10, fmin(1 - 1e-10, cond_prob));
                    target += binomial_lpmf(count_above_each_reported_cutpoint[c] |
                                            count_above_the_previous_reported_cutpoint_or_class_total[c],
                                            cond_prob_safe);
              }
        }
}
























