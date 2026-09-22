#### =====================================================================================================================================
## random_intercepts.R
##
## A simulated logistic random-intercept example. This is a usage example, not a comparative benchmark.
##
run_random_intercepts <-  function( burnin_algorithm = "CHESSR",
                                     model_file = NULL) {

        ## Compile in a writable temporary directory rather than inside the installed package.
        if (is.null(model_file)) {
                model_directory <-  tempfile(pattern = "nicostan_random_intercepts_")
                dir.create(path = model_directory)
                model_file <-  file.path(model_directory, "random_intercepts.stan")
                stopifnot(file.copy(from = system.file("examples", "random_intercepts.stan", package = "NicoStan"),
                                   to = model_file))
        }
        stopifnot(file.exists(model_file))
        set.seed(seed = 2026)
        n_groups <-  20L
        n_observations <-  200L
        group <-  rep(x = seq_len(n_groups), each = n_observations / n_groups)
        x <-  rnorm(n = n_observations)
        group_effect <-  rnorm(n = n_groups, sd = 0.6)
        probability <-  plogis(q = -0.4 + 0.8 * x + group_effect[group])
        stan_data <-  list(N = n_observations, J = n_groups, group = group, x = x,
                           y = rbinom(n = n_observations, size = 1, prob = probability))
        ##
        model <-  NicoStan::MVP_model$new(Model_type = "Stan",
                                           Stan_data_list = stan_data,
                                           Stan_model_file_path = model_file,
                                           sample_nuisance = TRUE)
        initial_values <-  lapply(X = seq_len(4L), FUN = function(chain) {
                list(z = rep(x = 0, times = n_groups), alpha = 0.1 * (chain - 2.5), beta = 0, sigma = 0.6)
        })
        ##
        model$sample(n_burnin = 125, n_iter = 250,
                      n_chains_burnin = 4, n_chains_sampling = 4, n_superchains = 2,
                      init_lists_per_chain = initial_values,
                      burnin_algorithm = burnin_algorithm,
                      diffusion_HMC = TRUE, partitioned_HMC = FALSE,
                      metric_type_main = "Empirical", metric_shape_main = "diag",
                      metric_type_nuisance = "uniform_diag",
                      M_decay_type = "inverse", M_decay_power = 0.5, M_decay_scale = 1.13,
                      ratio_M_main = 0.9, ratio_M_nuisance = 0.9,
                      learning_rate = 0.05, learning_rate_initial = 0.1,
                      n_threads_WCP_burnin = 1, n_threads_WCP_sampling = 1,
                      adapt_delta = 0.80, n_refresh = 25,
                      seed = 2026, stream = 2026, reorder_cols_MVP = FALSE)
        return(model)

}
