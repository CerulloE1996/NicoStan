##
## ==== Run_one_class_MVOP_with_the_fused_GHK_log_likelihood_and_BayesMVP.R ====================================
##
## ---- The one-class multivariate ordered probit (MVOP) model, fitted to the real NHANES data twice:
##
##      (1) as a Stan model with NicoSIMD's fused GHK log likelihood
##          (models/one_class_MVOP_with_the_fused_GHK_log_likelihood.stan), by NicoStan, on any CPU with AVX-512,
##          AVX2 or NEON (e.g. Apple Silicon Macs, through SIMDe);
##      (2) by BayesMVP's built-in MVOP model, whose gradients are derived by hand (manual gradients).
##      Both have the same priors (BayesMVP's MVOP priors, with the values below), hence the same posterior;
##      their posterior means of the correlation matrix are printed side by side at the end.
##
## ---- Each part shows, step by step:
##
##        (a) how the priors are set;
##        (b) how the nuisance parameters are given;
##        (c) the initial values of the burn-in chains;
##        (d) how the package finds the optimal within-chain threads (WCP) and N_chunks of the burn-in and of the
##            sampling by itself, from the L2 and L3 caches of the computer it runs on (Paper 1's equations, then
##            a short timed check of the layouts around them), and how they are carried into $sample();
##        (e) the fit and its summaries.
## ---- The data: the 9575 US adults of NHANES I (the NHANES data of the hexbin package), with 4 outcomes, each
##      with its OWN covariates, and every missing result coded -1 (handled by the GHK method):
##        iron deficiency (transferrin saturation below 16%), low albumin (below 3.5 g/dL) and cancer (binary);
##        anaemia (ordinal: the WHO categories of haemoglobin for each sex, none / mild / moderate / severe).
##
## Usage:  Rscript Run_one_class_MVOP_with_the_fused_GHK_log_likelihood_and_BayesMVP.R
##         (or source() it; the settings and the priors are in the two blocks below; BayesMVP is needed for
##          part (2) only)
##
##
## ------- Settings:  --------------------------------------------------------------------------------------------
##
{
      seed                          <-  123
      n_burnin                      <-  500
      n_iter                        <-  500
      n_chains_burnin               <-  4
      fit_with_BayesMVP_too         <-  TRUE
}
##
##
## ------- The priors (the same in both parts, hence the same posterior):  ---------------------------------------
##
{
      ## (an LKJ prior on the correlation matrix Omega of the latent values; 2 is BayesMVP's default)
      prior_LKJ_eta                         <-  2
      ## (a normal prior on every coefficient: mean 0; SD 2.5 for the intercepts, since the outcomes are rare
      ##  (e.g. low albumin in 0.6% of people, i.e. an intercept near -2.5 on the probit scale), and SD 1 for
      ##  every other coefficient)
      prior_mean_of_each_coefficient        <-  0
      prior_SD_of_each_intercept            <-  2.5
      prior_SD_of_each_other_coefficient    <-  1
      ## (the induced Dirichlet prior on the cutpoints of each ordinal outcome, with every alpha equal to 1)
      prior_Dirichlet_alpha                 <-  1
      ## (the bounds of the first cutpoint, and of the logs of the gaps between the cutpoints)
      C_raw_lower                           <-  -10
      C_raw_upper                           <-  2.5
}
##
##
## ------- The outcomes and covariates of the real NHANES data:  -------------------------------------------------
##
utils::data(NHANES, package = "hexbin")
N <-  nrow(NHANES)
female <-  as.numeric(NHANES$Sex == "F")
##
## ---- Anaemia (the WHO categories for each sex): 1 = none, 2 = mild, 3 = moderate, 4 = severe:
##
cut_off_of_anaemia <-  ifelse(female == 1, 12, 13)
anaemia <-  ifelse(NHANES$Hemoglobin >= cut_off_of_anaemia, 1,
                   ifelse(NHANES$Hemoglobin >= 11, 2,
                          ifelse(NHANES$Hemoglobin >= 8, 3, 4)))
##
## ---- The binary outcomes (first), then anaemia; every missing result coded -1:
##
iron_deficiency <-  as.numeric(NHANES$Transferin < 16)
low_albumin <-  as.numeric(NHANES$Albumin < 3.5)
cancer <-  as.numeric(NHANES$Cancer.Incidence == "Yes")
y <-  cbind(iron_deficiency, low_albumin, cancer, anaemia)
y[is.na(y)] <-  -1
storage.mode(y) <-  "integer"
n_tests <-  ncol(y)
test_is_binary <-  c(1, 1, 1, 0)
n_categories_of_test <-  c(2, 2, 2, 4)
##
## ---- The covariates (age per 10 years from 50, weight per 10 kg from 75 kg (BMI is missing for most people),
##      smoking against never smoked, and the 0 / 1 codes of education and race):
X_all <-  cbind(intercept                 = 1,
                age_per_10_years_from_50  = (NHANES$Age - 50) / 10,
                female                    = female,
                weight_per_10_kg_from_75  = (NHANES$Weight - 75) / 10,
                current_smoker            = as.numeric(NHANES$Smoke == "Current"),
                past_smoker               = as.numeric(NHANES$Smoke == "Past"),
                smoking_unknown           = as.numeric(NHANES$Smoke == "Unknown"),
                education                 = NHANES$Ed,
                race                      = NHANES$Race)
stopifnot(!anyNA(X_all))
n_covariates_max <-  ncol(X_all)
##
## ---- Each outcome's own covariates (in the first columns of its X matrix, the rest 0):
##
smoking <-  c("current_smoker", "past_smoker", "smoking_unknown")
covariates_of_each_outcome <-  list(
      iron_deficiency = c("intercept", "age_per_10_years_from_50", "female", "weight_per_10_kg_from_75",
                          "race", "education"),
      low_albumin     = c("intercept", "age_per_10_years_from_50", "female", "weight_per_10_kg_from_75",
                          smoking, "education"),
      cancer          = c("intercept", "age_per_10_years_from_50", "female", "weight_per_10_kg_from_75",
                          smoking, "race"),
      anaemia         = c("intercept", "age_per_10_years_from_50", "female", "weight_per_10_kg_from_75",
                          smoking, "race", "education"))
X_of_each_outcome <-  list()
for (t in 1:n_tests) {
      covariates_t <-  covariates_of_each_outcome[[t]]
      X_t <-  matrix(0, nrow = N, ncol = n_covariates_max)
      X_t[, seq_along(covariates_t)] <-  X_all[, covariates_t]
      X_of_each_outcome[[t]] <-  X_t
}
n_covariates_of_test <-  as.numeric(lengths(covariates_of_each_outcome))
message(paste0("\033[36m", "NHANES: N = ", N, "; missing results of each outcome: ",
               paste0(colSums(y == -1), collapse = ", "), "\033[0m"))
##
## ---- The priors of the coefficients, as matrices (covariates x outcomes; the intercepts in the first row):
##
prior_beta_mean <-  matrix(prior_mean_of_each_coefficient, nrow = n_covariates_max, ncol = n_tests)
prior_beta_sd <-  matrix(prior_SD_of_each_other_coefficient, nrow = n_covariates_max, ncol = n_tests)
prior_beta_sd[1, ] <-  prior_SD_of_each_intercept
##
## ---- The initial values of the cutpoints of anaemia (the first cutpoint, then the logs of the 2 gaps):
##
C_raw_initial <-  c(1.0, log(0.5), log(1.5))
##
##
## ------- (1) The Stan model with NicoSIMD's fused GHK log likelihood, by NicoStan:  ----------------------------
##
## ---- NicoSIMD gives the user header and build arguments of this CPU (AVX-512, AVX2, or NEON on 64-bit ARM):
##
NicoSIMD_build <-  NicoSIMD::fn_include_folder_Stan_user_header_and_build_arguments_of_NicoSIMD_for_this_CPU()
message(paste0("\033[36m", "NicoSIMD's functions on this CPU: ", NicoSIMD_build$instruction_set, "\033[0m"))
##
## ---- The Stan model, copied to a writable folder (it is compiled next to its .stan file):
##
Stan_model_file_path <-  file.path(tempdir(), "one_class_MVOP_with_the_fused_GHK_log_likelihood.stan")
file.copy(from = system.file("examples", "models", "one_class_MVOP_with_the_fused_GHK_log_likelihood.stan",
                             package = "NicoStan"),
          to = Stan_model_file_path, overwrite = TRUE)
##
## ---- (1a) The priors: the Stan model's model block sets them, and their values are passed to it as data:
##
##          beta_vec ~ normal(prior_beta_mean_vec, prior_beta_sd_vec);   (prior_beta_mean, prior_beta_sd)
##          L_Omega ~ lkj_corr_cholesky(prior_LKJ_eta);                   (prior_LKJ_eta)
##          target += induced_Dirichlet_probit_lpdf(the cutpoints of each ordinal outcome |
##                                                  rep_vector(prior_Dirichlet_alpha, ...), ...);
##      (to change a prior's form, edit the model block; to change its values, edit the priors block above)
Stan_data_list <-  list(N                      = N,
                        n_tests                = n_tests,
                        test_is_binary         = test_is_binary,
                        n_categories_of_test   = n_categories_of_test,
                        y                      = y,
                        n_covariates_max       = n_covariates_max,
                        n_covariates_of_test   = n_covariates_of_test,
                        X                      = X_of_each_outcome,
                        ## (the priors' values)
                        prior_beta_mean        = prior_beta_mean,
                        prior_beta_sd          = prior_beta_sd,
                        prior_LKJ_eta          = prior_LKJ_eta,
                        prior_Dirichlet_alpha  = prior_Dirichlet_alpha,
                        C_raw_lower            = C_raw_lower,
                        C_raw_upper            = C_raw_upper,
                        ## (Phi_type 1: the exact normal CDF; the bounds of the log-scale tails)
                        Phi_type               = 1,
                        overflow_threshold     = 7.5,
                        underflow_threshold    = -7.5,
                        ## (NicoStan's reduce_sum data: N and chunk_size, which NicoStan sets for each layout)
                        chunk_size             = N)
##
## ---- (1b) The nuisance parameters: u_raw, the N x n_tests GHK values (one per subject and outcome), declared in
##      the parameters block of the Stan model,
##          array[N] row_vector[n_tests] u_raw;
##      and given to NicoStan by name with nuisance_params (any number of nuisance blocks may be given, e.g.
##      nuisance_params = c("z_intercept", "z_slope")). NicoStan samples them with diffusion-pathspace HMC
##      (diffusion_HMC = TRUE in $sample()), and the main parameters with standard HMC; as u_raw is the sliced
##      argument of the model's reduce_sum_static, NicoStan holds its values inside the chunks
##      (reduce_sum_chunked). (NicoStan never chooses the nuisance blocks itself: they are always named.)
model_NicoStan <-  NicoStan::Nico_model$new(Model_type            = "Stan",
                                            Stan_data_list        = Stan_data_list,
                                            Stan_model_file_path  = Stan_model_file_path,
                                            Stan_cpp_user_header  = NicoSIMD_build$Stan_user_header,
                                            make_args             = NicoSIMD_build$make_args,
                                            nuisance_params       = "u_raw",
                                            sample_nuisance       = TRUE)
##
## ---- (1c) The initial values of the burn-in chains (u_raw as an N x n_tests matrix, one row per subject):
##
initial_values_NicoStan <-  rep(list(list(u_raw     = matrix(0, nrow = N, ncol = n_tests),
                                          beta_vec  = rep(0, sum(n_covariates_of_test)),
                                          L_Omega   = diag(n_tests),
                                          C_raw     = C_raw_initial)), n_chains_burnin)
##
## ---- (1d) The optimal within-chain threads (WCP) and N_chunks of the burn-in and of the sampling, found by
##      NicoStan from the L2 and L3 caches of this computer (Paper 1's equations), then checked by timing a few
##      layouts around them (Paper 1's check), within a budget of 2000 of the model's own gradient evaluations:
layouts_NicoStan <-
      NicoStan::fn_compute_burnin_and_sampling_layouts_of_Stan_model_within_a_budget_of_lp_grad_calls(
            model_so_file    = model_NicoStan$init_object$model_so_file,
            Stan_data_list   = Stan_data_list,
            init_object      = model_NicoStan$init_object,
            n_chains_burnin  = n_chains_burnin)
message(paste0("\033[36m", "NicoStan's layouts (found in ",
               round(layouts_NicoStan$seconds_spent_choosing_the_burnin_and_sampling_layouts, 1), " s):",
               " burn-in ", n_chains_burnin, " chains x ", layouts_NicoStan$n_threads_WCP_burnin,
               " threads per chain, N_chunks = ", layouts_NicoStan$num_chunks_burnin,
               "; sampling ", layouts_NicoStan$n_chains_sampling, " chains x ",
               layouts_NicoStan$n_threads_WCP_sampling, " threads per chain, N_chunks = ",
               layouts_NicoStan$num_chunks_sampling, "\033[0m"))
##
## ---- (1e) The fit, with these layouts carried over into $sample().
##      Note: if you leave these out (or set them to NULL), NicoStan will find them for you, as long as your
##      model uses reduce_sum() (with N and chunk_size in its data) or is a built-in BayesMVP model.
model_NicoStan$sample(init_lists_per_chain     = initial_values_NicoStan,
                      n_burnin                 = n_burnin,
                      n_iter                   = n_iter,
                      n_chains_burnin          = n_chains_burnin,
                      burnin_algorithm         = "CHESSR",
                      diffusion_HMC            = TRUE,
                      partitioned_HMC          = FALSE,
                      n_threads_WCP_burnin     = layouts_NicoStan$n_threads_WCP_burnin,
                      num_chunks_burnin        = layouts_NicoStan$num_chunks_burnin,
                      n_chains_sampling        = layouts_NicoStan$n_chains_sampling,
                      n_threads_WCP_sampling   = layouts_NicoStan$n_threads_WCP_sampling,
                      num_chunks_sampling      = layouts_NicoStan$num_chunks_sampling,
                      seed                     = seed)
fit_summary_NicoStan <-  model_NicoStan$summary(save_log_lik_trace = FALSE)
print(fit_summary_NicoStan$get_summary_main(), n = 50)
print(fit_summary_NicoStan$get_summary_transformed(), n = 50)
##
##
## ------- (2) The same model by BayesMVP's built-in MVOP (manual gradients):  -----------------------------------
##
if (fit_with_BayesMVP_too) {
      ##
      ## ---- (2a) The priors: BayesMVP's MVOP takes them in model_args_list (each a list with one element per
      ##      latent class; one class here): prior_coeffs_mean_mat and prior_coeffs_sd_mat (covariates x
      ##      outcomes), lkj_cholesky_eta (one per class), prior_dirichlet_alpha (categories x ordinal outcomes),
      ##      and the bounds C_raw_lower and C_raw_upper; the same values as in part (1):
      model_args_list <-  list(y                             = y,
                               n_cat_per_ord_test            = 4,
                               X                             = list(X_of_each_outcome),
                               n_covariates_per_outcome_mat  = matrix(n_covariates_of_test, nrow = 1),
                               ## (the priors)
                               prior_coeffs_mean_mat         = list(prior_beta_mean),
                               prior_coeffs_sd_mat           = list(prior_beta_sd),
                               lkj_cholesky_eta              = matrix(prior_LKJ_eta, nrow = 1, ncol = 1),
                               prior_dirichlet_alpha         = list(matrix(prior_Dirichlet_alpha, nrow = 4,
                                                                           ncol = 1)),
                               C_raw_lower                   = C_raw_lower,
                               C_raw_upper                   = C_raw_upper,
                               ## (Phi_type "Phi": the exact normal CDF)
                               Phi_type                      = "Phi")
      ##
      ## ---- (2b) The nuisance parameters: BayesMVP's MVOP has them built in (u_raw, the N x n_tests GHK values);
      ##      sample_nuisance = TRUE samples them with diffusion-pathspace HMC, as in part (1):
      model_BayesMVP <-  BayesMVP::MVP_model$new(Model_type       = "MVOP",
                                                 model_args_list  = model_args_list,
                                                 sample_nuisance  = TRUE)
      ##
      ## ---- (2c) The same initial values (BayesMVP's cutpoints are unconstrained by tanh, between the bounds):
##
      C_unc_initial <-  atanh(2 * (C_raw_initial - C_raw_lower) / (C_raw_upper - C_raw_lower) - 1)
      initial_values_BayesMVP <-  rep(list(list(u_raw                    = matrix(0, nrow = N, ncol = n_tests),
                                                beta                     = list(prior_beta_mean),
                                                Omega_unconstrained_vec  = list(rep(0, choose(n_tests, 2))),
                                                C_unc_vec                = list(C_unc_initial))),
                                      n_chains_burnin)
      ##
      ## ---- (2d) The optimal within-chain threads (WCP) and N_chunks of the burn-in and of the sampling,
      ##      found by BayesMVP from the L2 and L3 caches of this computer (Paper 1's equations for a manual
      ##      gradient), then checked by timing a few layouts around them (Paper 1's check):
      burnin_layout_BayesMVP <-  BayesMVP::fn_compute_burnin_n_threads_WCP_and_num_chunks_for_BayesMVP_model(
                                       Model_type       = "MVOP",
                                       init_object      = model_BayesMVP$init_object,
                                       n_chains_burnin  = n_chains_burnin)
      sampling_layout_BayesMVP <-
            BayesMVP::fn_compute_sampling_n_chains_n_threads_WCP_and_num_chunks_for_BayesMVP_model(
                  Model_type   = "MVOP",
                  init_object  = model_BayesMVP$init_object)
      message(paste0("\033[36m", "BayesMVP's layouts: burn-in ", n_chains_burnin, " chains x ",
                     burnin_layout_BayesMVP$n_threads_WCP_burnin, " threads per chain, N_chunks = ",
                     burnin_layout_BayesMVP$num_chunks_burnin, " (found in ",
                     round(burnin_layout_BayesMVP$seconds_spent_choosing, 1), " s); sampling ",
                     sampling_layout_BayesMVP$n_chains_sampling, " chains x ",
                     sampling_layout_BayesMVP$n_threads_WCP_sampling, " threads per chain, N_chunks = ",
                     sampling_layout_BayesMVP$num_chunks_sampling, " (found in ",
                     round(sampling_layout_BayesMVP$seconds_spent_choosing, 1), " s)", "\033[0m"))
      ##
      ## ---- (2e) The fit, with these layouts carried over into $sample().
      ##      Note: if you leave these out (or set them to NULL), NicoStan will find them for you, as long as your
      ##      model uses reduce_sum() (with N and chunk_size in its data) or is a built-in BayesMVP model.
      model_BayesMVP$sample(init_lists_per_chain     = initial_values_BayesMVP,
                            n_burnin                 = n_burnin,
                            n_iter                   = n_iter,
                            n_chains_burnin          = n_chains_burnin,
                            burnin_algorithm         = "CHESSR",
                            diffusion_HMC            = TRUE,
                            partitioned_HMC          = FALSE,
                            n_threads_WCP_burnin     = burnin_layout_BayesMVP$n_threads_WCP_burnin,
                            num_chunks_burnin        = burnin_layout_BayesMVP$num_chunks_burnin,
                            n_chains_sampling        = sampling_layout_BayesMVP$n_chains_sampling,
                            n_threads_WCP_sampling   = sampling_layout_BayesMVP$n_threads_WCP_sampling,
                            num_chunks_sampling      = sampling_layout_BayesMVP$num_chunks_sampling,
                            seed                     = seed,
                            reorder_cols_MVP         = FALSE)
      fit_summary_BayesMVP <-  model_BayesMVP$summary(save_log_lik_trace = FALSE)
      print(fit_summary_BayesMVP$get_summary_main(), n = 50)
      ##
      ## ---- The posterior means of the correlation matrix, side by side (they should agree within MCMC error):
##
      ##      the rows of the summaries named Omega[..., t1, t2], whatever their first indices
      fn_posterior_mean_of_the_correlation_matrix <-  function(summary_tibble,
                                                               n_tests) {

              rows_of_Omega <-  grepl(pattern = "^Omega\\[", x = summary_tibble$parameter)
              indices_of_each_row <-  regmatches(summary_tibble$parameter[rows_of_Omega],
                                                 gregexpr(pattern = "[0-9]+",
                                                          text = summary_tibble$parameter[rows_of_Omega]))
              posterior_mean_of_Omega <-  matrix(NA, nrow = n_tests, ncol = n_tests)
              for (r in seq_along(indices_of_each_row)) {
                    t1_t2 <-  as.numeric(utils::tail(indices_of_each_row[[r]], 2))
                    posterior_mean_of_Omega[t1_t2[1], t1_t2[2]] <-  summary_tibble$mean[rows_of_Omega][r]
              }
              dimnames(posterior_mean_of_Omega) <-  list(colnames(y), colnames(y))
              return(posterior_mean_of_Omega)
      }

      summary_of_NicoStan <-  rbind(fit_summary_NicoStan$get_summary_main(),
                                    fit_summary_NicoStan$get_summary_transformed(),
                                    fit_summary_NicoStan$get_summary_generated_quantities())
      summary_of_BayesMVP <-  rbind(fit_summary_BayesMVP$get_summary_main(),
                                    fit_summary_BayesMVP$get_summary_transformed(),
                                    fit_summary_BayesMVP$get_summary_generated_quantities())
      message("\033[32m", "Posterior means of the correlation matrix, NicoStan (fused GHK):", "\033[0m")
      print(round(fn_posterior_mean_of_the_correlation_matrix(summary_of_NicoStan, n_tests), 3))
      message("\033[32m", "Posterior means of the correlation matrix, BayesMVP (manual gradients):", "\033[0m")
      print(round(fn_posterior_mean_of_the_correlation_matrix(summary_of_BayesMVP, n_tests), 3))
}





















