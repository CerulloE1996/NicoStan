#### =====================================================================================================================================
## robust_t4: three sample sizes, common data/initial values, and main-parameter efficiency benchmarks.
## Source this file in R or run with Rscript. Edit the settings below for the research run.
##
model <-  "robust_t4"
N_values <-  c(250L, 1000L, 3000L)
profile <-  Sys.getenv("NICOSTAN_EXAMPLE_PROFILE", unset = "analysis")
burnin_algorithm <-  "CHESSR"
##
## Resolve this example folder without changing the working directory.
.example_sources <-  as.character(unlist(lapply(sys.frames(), function(frame) frame$ofile)))
if (!length(.example_sources)) .example_sources <-  sub("^--file=", "", commandArgs()[startsWith(commandArgs(), "--file=")])
if (!length(.example_sources)) stop("Source this file or run it with Rscript.")
.example_directory <-  dirname(normalizePath(tail(.example_sources, 1L), mustWork = TRUE))
## Select a separate library only when explicitly requested.
.example_library <-  Sys.getenv("NICOSTAN_EXAMPLE_R_LIB", unset = "")
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
## On AVX-512 hardware this gives CmdStanR, NicoStan, and NicoStan + AVX-512.
## On AVX2-only hardware the third arm uses AVX2 and is labelled accordingly.
benchmark_arms <-  NicoStan_benchmark_arms()
if (identical(NicoStan_benchmark_cpu_has_isa("AVX512"), FALSE)) {
        benchmark_arms <-  benchmark_arms[benchmark_arms$math_backend != "AVX512", ]
        if (isTRUE(NicoStan_benchmark_cpu_has_isa("AVX2"))) {
                avx2 <-  NicoStan_benchmark_arms(include_avx2 = TRUE)
                benchmark_arms <-  rbind(benchmark_arms, avx2[avx2$math_backend == "AVX2", ])
        }
        message("AVX-512 is unavailable on this CPU; the arm table records the supported comparisons.")
}
benchmark_result <-  NicoStan_benchmark_run(models = model, N_grid = setNames(list(N_values), model),
    arms = benchmark_arms, include_avx2 = any(benchmark_arms$math_backend == "AVX2"), profile = profile,
    burnin_algorithm = burnin_algorithm, validate = profile == "smoke",
    output_dir = file.path(getwd(), paste0("NicoStan_benchmark_", model)),
    model_dir = file.path(.example_directory, "models"))
benchmark_result$comparison$efficiency
