#### =====================================================================================================================================
## Native multivariate probit examples supplied by the BayesMVP extension.
##
run_BayesMVP_native_example <-  function(Model_type = c("MVP", "LC_MVP", "MVOP", "LC_MVOP", "latent_trait"), ...) {
        Model_type <-  match.arg(Model_type)
        example_file <-  system.file("examples", "run_native_example.R", package = "BayesMVP")
        if (!nzchar(example_file)) stop("Install the current BayesMVP extension before running its native examples.")
        example_environment <-  new.env(parent = parent.frame())
        sys.source(example_file, envir = example_environment)
        example_environment$run_native_example(Model_type = Model_type, ...)
}
##
benchmark_BayesMVP_native_example <-  function(Model_type,
                                               N_values = c(48L, 150L, 500L),
                                               burnin_algorithm = "CHESSR",
                                               n_burnin = 200L,
                                               n_iter = 250L,
                                               output_directory = file.path(getwd(), paste0("BayesMVP_native_", Model_type))) {
        stopifnot(length(N_values) == 3L, length(unique(N_values)) == 3L)
        dir.create(output_directory, recursive = TRUE, showWarnings = FALSE)
        records <-  lapply(N_values, function(N) {
                fit <-  run_BayesMVP_native_example(Model_type = Model_type, N = N,
                    burnin_algorithm = burnin_algorithm, n_burnin = n_burnin, n_iter = n_iter)
                draws <-  fit$model_fit_object$traces$traces_as_arrays$trace_params_main
                summary <-  posterior::summarise_draws(posterior::as_draws_array(draws), "ess_bulk", "ess_tail", "rhat")
                stopifnot(dim(draws)[3L] == fit$init_object$n_params_main)
                time_sampling <-  as.numeric(fit$result$time_sampling)
                time_fitting <-  time_sampling + as.numeric(fit$result$time_burnin)
                if (!is.finite(time_sampling) || time_sampling <= 0) time_sampling <-  NA_real_
                if (!is.finite(time_fitting) || time_fitting <= 0) time_fitting <-  NA_real_
                efficiency <-  fit$model_fit_object$summaries$efficiency_info
                gradient_work <-  efficiency$L_main_during_sampling * n_iter * dim(draws)[2L]
                build <-  list(BayesMVP_version = as.character(utils::packageVersion("BayesMVP")),
                    NicoStan_version = as.character(utils::packageVersion("NicoStan")),
                    BayesMVP_library = find.package("BayesMVP"), NicoStan_library = find.package("NicoStan"),
                    BayesMVP_native_sha256 = digest::digest(file = getLoadedDLLs()[["BayesMVP"]][["path"]], algo = "sha256"),
                    BayesMVP_R_sha256 = digest::digest(file = file.path(find.package("BayesMVP"), "R", "BayesMVP.rdb"), algo = "sha256"),
                    NicoStan_R_sha256 = digest::digest(file = file.path(find.package("NicoStan"), "R", "NicoStan.rdb"), algo = "sha256"))
                fingerprint <-  substr(digest::digest(list(Model_type, N, burnin_algorithm, n_burnin, n_iter, build),
                                                       algo = "sha256"), 1L, 16L)
                result <-  list(Model_type = Model_type, N = N, summary_main = summary,
                    main_parameter_names = summary$variable, n_nuisance = fit$n_nuisance,
                    min_ESS_main = min(summary$ess_bulk), min_ESS_tail_main = min(summary$ess_tail),
                    max_Rhat_main = max(summary$rhat), sampling_seconds = time_sampling,
                    fitting_seconds = time_fitting,
                    min_ESS_per_sec_sampling = min(summary$ess_bulk) / time_sampling,
                    min_ESS_per_sec_fitting = min(summary$ess_bulk) / time_fitting,
                    min_ESS_per_1000_grad_sampling = 1000 * min(summary$ess_bulk) / gradient_work,
                    gradient_work_basis = "APMS estimate: adapted L_main times sampling iterations and chains",
                    divergences = fit$model_fit_object$summaries$divergences,
                    settings = list(burnin_algorithm = burnin_algorithm, n_burnin = n_burnin, n_iter = n_iter),
                    engine = "BayesMVP native manual-gradient extension", build = build, fingerprint = fingerprint)
                saveRDS(result, file.path(output_directory, paste0(Model_type, "_N", N, "_", fingerprint, ".rds")))
                result
        })
        table <-  do.call(rbind, lapply(records, function(result) {
                as.data.frame(result[c("Model_type", "N", "n_nuisance", "min_ESS_main", "min_ESS_tail_main",
                    "max_Rhat_main", "sampling_seconds", "fitting_seconds", "min_ESS_per_sec_sampling",
                    "min_ESS_per_sec_fitting", "min_ESS_per_1000_grad_sampling", "gradient_work_basis", "engine", "fingerprint")])
        }))
        utils::write.csv(table, file.path(output_directory, "native_efficiency.csv"), row.names = FALSE)
        list(summary = table, results = records)
}
