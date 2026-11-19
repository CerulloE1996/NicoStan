#### =====================================================================================================================================
## rnn_ssm: non-centred Bayesian gated nonlinear state-space model (adapted from BPA/Ch.05/ssm.stan).
## Three sample sizes matching stochastic volatility, common data/initial values, and main-parameter efficiency benchmarks.
## Source this file in R or run with Rscript. Edit the settings below for the research run.
##
model <- "rnn_ssm"
N_values <- c(500, 10000, 100000)
profile <- Sys.getenv("NICOSTAN_EXAMPLE_PROFILE", unset = "analysis")
burnin_algorithm <- "CHESSR"
##
## Resolve this example folder without changing the working directory.
.example_sources <- as.character(unlist(lapply(sys.frames(), function(frame) frame$ofile)))
if (!length(.example_sources)) .example_sources <- sub("^--file=", "", commandArgs()[startsWith(commandArgs(), "--file=")])
if (!length(.example_sources)) stop("Source this file or run it with Rscript.")
.example_directory <- dirname(normalizePath(tail(.example_sources, 1), mustWork = TRUE))
## Select a separate library only when explicitly requested.
.example_library <- Sys.getenv("NICOSTAN_EXAMPLE_R_LIB", unset = "")
if (nzchar(.example_library)) {
        if ("NicoStan" %in% loadedNamespaces() &&
            normalizePath(getNamespaceInfo(asNamespace("NicoStan"), "path")) !=
            normalizePath(file.path(.example_library, "NicoStan"), mustWork = FALSE)) {
                stop("Restart R before selecting a different NicoStan library.")
        }
        .libPaths(c(.example_library, .libPaths()))
}

source(file.path(.example_directory, "NicoStan_examples.R"))
source(file.path(.example_directory, "NicoStan_benchmarks.R"))
##
## Standard Stan math arms only (CmdStanR and NicoStan; no separate _avx.stan file for this model).
benchmark_arms <- NicoStan_benchmark_arms()
benchmark_arms <- benchmark_arms[benchmark_arms$math_backend == "Stan", , drop = FALSE]
benchmark_result <- NicoStan_benchmark_run(models = model, N_grid = setNames(list(N_values), model),
    arms = benchmark_arms, include_avx2 = FALSE, profile = profile,
    burnin_algorithm = burnin_algorithm, validate = profile == "smoke",
    output_dir = file.path(getwd(), paste0("NicoStan_benchmark_", model)),
    model_dir = file.path(.example_directory, "models"))
benchmark_result$comparison$efficiency
