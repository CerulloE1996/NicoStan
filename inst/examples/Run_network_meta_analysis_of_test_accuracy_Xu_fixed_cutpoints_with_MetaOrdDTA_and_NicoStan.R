##
## ==== Run_network_meta_analysis_of_test_accuracy_Xu_fixed_cutpoints_with_MetaOrdDTA_and_NicoStan.R ===========
##
## ---- MetaOrdDTA's network meta-analysis (NMA) of the accuracy of several ordinal diagnostic tests with multiple
##      thresholds (Nyaga et al.'s NMA model, with Xu et al.'s fixed cutpoints: one set of cutpoints per test and
##      disease class, shared by every study), fitted by NicoStan to simulated data:
##        - 4 ordinal anxiety screening tests (GAD-2, GAD-7, HADS and BAI), 80 studies, each with 1 to 3 of them;
##        - at each threshold of each test, the number of non-diseased and of diseased patients above it, with
##          the thresholds a study did not report coded -1 ("missing thresholds", which the model handles);
##        - simulated with MetaOrdDTA's own simulator (R_fn_sim_NMA_data_with_covariates_varying()), from a
##          realistic design.
## ---- MetaOrdDTA prepares the data and the priors (exactly as it gives them to Stan), and NicoStan fits its Stan
##      model, unchanged. Step by step:
##        (a) how the priors are set;
##        (b) how the nuisance parameters are given;
##        (c) the initial values of the burn-in chains;
##        (d) the layouts (no reduce_sum() here, hence one thread per chain and no chunks);
##        (e) the fit and its summaries.
##
## Usage:  Rscript Run_network_meta_analysis_of_test_accuracy_Xu_fixed_cutpoints_with_MetaOrdDTA_and_NicoStan.R
##         (or source() it; the settings and the priors are in the two blocks below; MetaOrdDTA is needed)
##
##
## ------- Settings:  --------------------------------------------------------------------------------------------
##
{
      seed                          <-  123
      n_burnin                      <-  500
      n_iter                        <-  500
      n_chains_burnin               <-  4
}
##
##
## ------- The priors (MetaOrdDTA's defaults for this model, written out so that they can be changed):  ----------
##
{
      ## (a normal prior on each test's mean location in each class, beta_mu: mean 0, SD 1)
      prior_mean_of_each_beta_mu         <-  0
      prior_SD_of_each_beta_mu           <-  1
      ## (half-normal priors on the between-study SDs: beta_sigma, shared by the tests, and beta_tau, per test)
      prior_SD_of_each_beta_sigma        <-  1
      prior_SD_of_each_beta_tau          <-  1
      ## (an LKJ prior on the between-study correlation of the two classes, within its bounds)
      prior_LKJ_eta_of_beta_corr         <-  2
      ## (the induced Dirichlet prior on the cutpoints of each test and class, with every alpha equal to 1)
      prior_Dirichlet_alpha              <-  1
}
##
##
## ------- The simulated data:  ----------------------------------------------------------------------------------
##
simulated_NMA_data <-  readRDS(system.file("examples", "data",
                                           paste0("simulated_network_meta_analysis_of_four_ordinal_screening_",
                                                  "tests_with_missing_thresholds.rds"),
                                           package = "NicoStan"))
## (x[[t]][[c]]: test t, class c (1 = non-diseased, 2 = diseased), one row per study; its first column is the
##  number of patients of the class, the others the number above each threshold, -1 when it was not reported)
x <-  lapply(simulated_NMA_data$x, unname)
indicator_index_test_in_study <-  simulated_NMA_data$indicator_index_test_in_study
n_studies <-  simulated_NMA_data$n_studies
message(paste0("\033[36m", "Simulated NMA data: ", n_studies, " studies; studies per test (",
               paste0(simulated_NMA_data$names_of_the_index_tests, collapse = ", "), "): ",
               paste0(colSums(indicator_index_test_in_study), collapse = ", "), "\033[0m"))
##
##
## ---- MetaOrdDTA's data list and priors for this model (Xu, fixed cutpoints, intercept only), exactly as it
##      gives them to Stan; nothing is compiled or fitted by MetaOrdDTA here: ----------------------------------
##
model_MetaOrdDTA <-  MetaOrdDTA::MetaOrd_model$new(x                              = x,
                                                   indicator_index_test_in_study  = indicator_index_test_in_study,
                                                   intercept_only                 = TRUE,
                                                   network                        = TRUE,
                                                   model_parameterisation         = "Xu",
                                                   random_thresholds              = FALSE,
                                                   Dirichlet_random_effects_type  = "fixed",
                                                   n_chains                       = n_chains_burnin,
                                                   compile                        = FALSE)
##
## ---- (a) The priors: the Stan model's model block sets them, and their values are passed to it as data:
##
##          beta_mu[t][c, ] ~ normal(prior_beta_mu_mean[t][c, ], prior_beta_mu_SD[t][c, ]);
##          beta_sigma[c] ~ normal(0, prior_beta_sigma_SD[c]);           (half-normal: beta_sigma >= 0)
##          to_vector(beta_tau_raw) ~ normal(0, to_vector(prior_beta_tau_SD));   (half-normal)
##          beta_L_Omega ~ lkj_corr_cholesky(prior_beta_corr_LKJ);
##          Ind_Dir_ord_prob ~ induced_dirichlet_given_C(C_vec_t, prior_dirichlet_alpha[c][t, ], ...);
##      (to change a prior's form, edit the model block; to change its values, edit the priors block above)
priors <-  model_MetaOrdDTA$priors
for (t in seq_along(priors$prior_beta_mu_mean)) {
      priors$prior_beta_mu_mean[[t]][] <-  prior_mean_of_each_beta_mu
      priors$prior_beta_mu_SD[[t]][] <-  prior_SD_of_each_beta_mu
}
priors$prior_beta_sigma_SD[] <-  prior_SD_of_each_beta_sigma
priors$prior_beta_tau_SD[] <-  prior_SD_of_each_beta_tau
priors$prior_beta_corr_LKJ <-  prior_LKJ_eta_of_beta_corr
## (one matrix per class, tests x categories, as the Stan model declares it: array[2] matrix[...] (older
##  MetaOrdDTA installs give a single matrix here))
n_thr_max <-  model_MetaOrdDTA$internal_obj$outs_data$stan_data_list$n_thr_max
alpha_of_each_class <-  matrix(prior_Dirichlet_alpha, nrow = length(x), ncol = n_thr_max + 1)
priors$prior_dirichlet_alpha <-  list(alpha_of_each_class, alpha_of_each_class)
## (the data list, then the priors, with the names given twice kept once, as MetaOrdDTA does)
Stan_data_list <-  MetaOrdDTA:::R_fn_remove_duplicates_from_list(c(model_MetaOrdDTA$internal_obj$outs_data$
                                                                       stan_data_list,
                                                                   priors))
## (the model's 0 / 1 switches as numbers: MetaOrdDTA keeps them as TRUE / FALSE, which cmdstanr writes as
##  1 / 0, whereas the JSON NicoStan writes for BridgeStan keeps true / false, which Stan's int data refuse)
for (name_of_switch in c("softplus", "hetero_sigma", "compound_symmetry")) {
      Stan_data_list[[name_of_switch]] <-  as.integer(Stan_data_list[[name_of_switch]])
}
##
## ---- MetaOrdDTA's Stan model (unchanged) and the folder of its #include files, copied to a writable folder
##      (NicoStan compiles it next to its .stan file):
Stan_model_folder <-  file.path(tempdir(), "MetaOrdDTA_NMA_Xu_fixed_cutpoints")
dir.create(Stan_model_folder, showWarnings = FALSE)
Stan_model_file_path <-  file.path(Stan_model_folder, "DTA_NMA_Nyaga_Xu_FIXEDthr.stan")
file.copy(from = system.file("stan_models", "stan_models_NMA", "DTA_NMA_Nyaga_Xu_FIXEDthr.stan",
                             package = "MetaOrdDTA"),
          to = Stan_model_file_path, overwrite = TRUE)
Stan_functions_folder <-  system.file("stan_functions", package = "MetaOrdDTA")
##
## ---- (b) The nuisance parameters: the two blocks of standard-normal between-study values of the Stan model,
##          matrix[2, n_studies] beta_eta_z;               (each study's effect in each class)
##          array[2] vector[n_total_obs] beta_delta_z_vec;  (each study-test pair's effect in each class)
##      with n_total_obs the number of study-test pairs, given to NicoStan by name with nuisance_params (NicoStan
##      never chooses the nuisance blocks itself). NicoStan writes the model's version with the two blocks held
##      by NicoStan (DTA_NMA_Nyaga_Xu_FIXEDthr_reduce_sum_chunked.stan, next to the model; the rest of the model
##      unchanged), and samples them with diffusion-pathspace HMC (diffusion_HMC = TRUE in $sample()), and the
##      main parameters with standard HMC.
n_total_obs <-  sum(indicator_index_test_in_study)
n_nuisance <-  2 * n_studies + 2 * n_total_obs
model_NicoStan <-  NicoStan::Nico_model$new(Model_type            = "Stan",
                                            Stan_data_list        = Stan_data_list,
                                            Stan_model_file_path  = Stan_model_file_path,
                                            stanc_args            = list(paste0("--include-paths=",
                                                                                Stan_functions_folder)),
                                            nuisance_params       = c("beta_eta_z", "beta_delta_z_vec"),
                                            ## n_nuisance_override   = n_nuisance,
                                            sample_nuisance       = TRUE)
message(paste0("\033[36m", "Nuisance parameters given to NicoStan: ", n_nuisance, " (2 x ", n_studies,
               " studies + 2 x ", n_total_obs, " study-test pairs)", "\033[0m"))
##
## ---- (c) The initial values of the burn-in chains, by name (the nuisance blocks at 0, i.e. every study at the
##      average; the cutpoints' raw values at 0; the SDs and the correlation small):
n_index_tests <-  length(x)
initial_values_NicoStan <-  rep(list(list(beta_eta_z        = matrix(0, nrow = 2, ncol = n_studies),
                                          beta_delta_z_vec  = list(rep(0, n_total_obs), rep(0, n_total_obs)),
                                          beta_mu           = rep(list(matrix(0, nrow = 2, ncol = 1)),
                                                                  n_index_tests),
                                          C_raw_vec         = list(rep(0, sum(Stan_data_list$n_thr)),
                                                                   rep(0, sum(Stan_data_list$n_thr))),
                                          beta_sigma        = c(0.25, 0.25),
                                          beta_tau_raw      = matrix(0.25, nrow = n_index_tests, ncol = 2),
                                          beta_corr         = 0)),
                                n_chains_burnin)
##
## ---- (d) The layouts: this model has no reduce_sum() (its N is the number of studies, which is small), so
##      every chain runs on one thread, without chunks (n_threads_WCP = 1, num_chunks = 1), and NicoStan runs one
##      chain per CPU in the sampling. For a model whose likelihood is in reduce_sum() chunks, see worked example
##      3, where NicoStan finds the optimal within-chain threads (WCP) and N_chunks of the burn-in and of the
##      sampling by itself, from the L2 and L3 caches of the computer.
##
## ---- (e) The fit.
##      Note: if you leave the layouts out (or set them to NULL), NicoStan will find them for you, as long as
##      your model uses reduce_sum() or is a built-in BayesMVP model.
model_NicoStan$sample(init_lists_per_chain     = initial_values_NicoStan,
                      n_burnin                 = n_burnin,
                      n_iter                   = n_iter,
                      n_chains_burnin          = n_chains_burnin,
                      burnin_algorithm         = "CHESSR",
                      diffusion_HMC            = TRUE,
                      partitioned_HMC          = FALSE,
                      seed                     = seed)
fit_summary <-  model_NicoStan$summary(save_log_lik_trace = FALSE)
print(fit_summary$get_summary_main(), n = 50)






















