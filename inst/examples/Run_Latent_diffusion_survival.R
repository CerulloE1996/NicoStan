#### =====================================================================================================================================
## latent_diffusion_survival: three sample sizes, common data/initial values, and main-parameter efficiency benchmarks.
## Model: Beskos, Kalogeropoulos and Pazos (2013, Section 6.3), after Roberts and Sangalli (2010). All subjects share one
## latent hazard path h(u) = hazard_scale * (X(u)^2 + 0.01), with dX_u = -(1.4 sin(X_u) + 1) du + dB_u, X_0 = 2, on [0, 1].
## The 0.01 hazard offset and the normal(1.4, 1) / normal(1, 1) drift priors are round-4 changes (2026-09-22, assistant-
## introduced, not in Beskos et al.) that remove the divergences; see MODEL_NOTES.md and the .stan header.
## The data are simulated by fn_simulate_latent_diffusion_survival_data() in NicoStan_examples.R (fine-grid path, exact
## event times given that path, right censoring at t = 1). NicoStan estimates the drift coefficients and the hazard
## scale jointly with the 50 standard normal Brownian increments w (the nuisance block, sample_nuisance = TRUE).
## Source this file in R or run with Rscript. Edit the settings below for the research run.
##
##
## ---- settings:
##
{
        model <- "latent_diffusion_survival"
        N_values <- c(100, 200, 800)
        profile <- Sys.getenv("NICOSTAN_EXAMPLE_PROFILE", unset = "analysis")
        burnin_algorithm <- "CHESSR"
}
##
## ---- resolve this example folder without changing the working directory:
##
{
        example_source_file_paths <- as.character(unlist(lapply(sys.frames(), function(stack_frame) stack_frame$ofile)))
        if (!length(example_source_file_paths)) {
                example_source_file_paths <- sub("^--file=", "", commandArgs()[startsWith(commandArgs(), "--file=")])
        }
        if (!length(example_source_file_paths)) stop("Source this file or run it with Rscript.")
        example_directory <- dirname(normalizePath(tail(example_source_file_paths, 1), mustWork = TRUE))
}
##
## ---- select a separate library only when explicitly requested:
##
{
        example_library_path <- Sys.getenv("NICOSTAN_EXAMPLE_R_LIB", unset = "")
        if (nzchar(example_library_path)) {
                if ("NicoStan" %in% loadedNamespaces() &&
                    normalizePath(getNamespaceInfo(asNamespace("NicoStan"), "path")) !=
                    normalizePath(file.path(example_library_path, "NicoStan"), mustWork = FALSE)) {
                        stop("Restart R before selecting a different NicoStan library.")
                }
                .libPaths(c(example_library_path, .libPaths()))
        }
        source(file.path(example_directory, "NicoStan_examples.R"))
        source(file.path(example_directory, "NicoStan_benchmarks.R"))
}
##
## ---- benchmark arms:
##
## On AVX-512 hardware this gives CmdStanR, NicoStan, and NicoStan + AVX-512.
## On AVX2-only hardware the third arm uses AVX2 and is labelled accordingly.
##
{
        benchmark_arms <- NicoStan_benchmark_arms()
        if (identical(NicoStan_benchmark_cpu_has_isa("AVX512"), FALSE)) {
                benchmark_arms <- benchmark_arms[benchmark_arms$math_backend != "AVX512", ]
                if (isTRUE(NicoStan_benchmark_cpu_has_isa("AVX2"))) {
                        benchmark_arms_with_avx2 <- NicoStan_benchmark_arms(include_avx2 = TRUE)
                        benchmark_arms <- rbind(benchmark_arms, benchmark_arms_with_avx2[benchmark_arms_with_avx2$math_backend == "AVX2", ])
                }
                message("AVX-512 is unavailable on this CPU; the arm table records the supported comparisons.")
        }
}
##
## ---- run the three sample sizes:
##
{
        benchmark_result <- NicoStan_benchmark_run(models           = model,
                                                   N_grid           = setNames(list(N_values), model),
                                                   arms             = benchmark_arms,
                                                   include_avx2     = any(benchmark_arms$math_backend == "AVX2"),
                                                   profile          = profile,
                                                   burnin_algorithm = burnin_algorithm,
                                                   validate         = profile == "smoke",
                                                   output_dir       = file.path(getwd(), paste0("NicoStan_benchmark_", model)),
                                                   model_dir        = file.path(example_directory, "models"))
}
##
## ---- check that the requested model and sample sizes are the ones that ran:
##
{
        if (nrow(benchmark_result$summary)) {
                effective_models <- unique(benchmark_result$summary$model)
                effective_N_values <- sort(unique(benchmark_result$summary$N))
                message(paste0("effective model: ", paste(effective_models, collapse = ", "),
                               "; effective N values: ", paste(effective_N_values, collapse = ", ")))
                if (!identical(effective_models, model)) stop(paste0("Benchmark ran model(s) ", paste(effective_models, collapse = ", "), ", not ", model, "."))
                if (!setequal(effective_N_values, N_values)) stop(paste0("Benchmark ran N = ", paste(effective_N_values, collapse = ", "), ", not the requested ", paste(N_values, collapse = ", "), "."))
        }
        benchmark_result$comparison$efficiency
}
