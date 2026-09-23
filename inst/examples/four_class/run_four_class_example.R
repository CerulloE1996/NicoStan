## Portable four-class example runner.
##
## Usage from this directory:
##   Rscript run_four_class_example.R NicoStan Stan 24
##   Rscript run_four_class_example.R NicoStan AVX512 24
##   Rscript run_four_class_example.R NicoStan AVX2 24     (AVX2 kernels; also works on an AVX-512 machine)
##   Rscript run_four_class_example.R cmdstanr Stan 24
## Add a fourth argument to choose the writable output directory.
##
## The default N grid is 24, 48, 96.  The three arms use the same synthetic
## fixture and the same main-parameter registry.  The AVX arm needs the general
## BayesMVP external header from an installed BayesMVP extension package.

script_file <- sub("^--file=", "", commandArgs()[startsWith(commandArgs(), "--file=")])
example_directory <- dirname(normalizePath(script_file, mustWork = TRUE))
source(file.path(example_directory, "four_class_example.R"), local = TRUE)

arguments <- commandArgs(trailingOnly = TRUE)
engine <- if (length(arguments) >= 1L) arguments[1L] else "NicoStan"
backend <- if (length(arguments) >= 2L) arguments[2L] else "Stan"
N <- if (length(arguments) >= 3L) as.integer(arguments[3L]) else 24L
output_directory <- if (length(arguments) >= 4L) normalizePath(arguments[4L], mustWork = FALSE) else
    file.path(getwd(), "NicoStan_four_class_results")
if (!engine %in% c("NicoStan", "cmdstanr")) stop("engine must be NicoStan or cmdstanr")
if (!backend %in% c("Stan", "AVX512", "AVX2")) stop("backend must be Stan, AVX512 or AVX2")
if (engine == "cmdstanr" && backend != "Stan") stop("CmdStanR uses the plain Stan arm")
if (!N %in% c(24L, 48L, 96L)) stop("N must be one of 24, 48, or 96")

paths <- four_class_example_paths(example_directory = example_directory)
paths$runtime_directory <- output_directory
dir.create(paths$runtime_directory, recursive = TRUE, showWarnings = FALSE)
fixture_directory <- file.path(paths$fixture_directory, paste0("N", N))
fixture_file <- file.path(fixture_directory, "fixture.R")
if (engine == "cmdstanr") {
    result <- run_four_class_cmdstanr_smoke(
        paths = paths,
        fixture_file = fixture_file,
        stan_file = paths$source_plain,
        output_rds = file.path(paths$runtime_directory,
                               paste0("cmdstanr_plain_N", N, ".rds"))
    )
} else {
    stan_file <- if (backend == "Stan") paths$source_plain else paths$source_avx
    avx_header <- system.file("include", "BayesMVP", "stan_external_functions.hpp",
                              package = "BayesMVP")
    if (backend %in% c("AVX512", "AVX2") && (!nzchar(avx_header) || !file.exists(avx_header)))
        stop("Install the BayesMVP extension package before running the ", backend, " arm")
    result <- run_four_class_nicostan_smoke(
        paths = paths,
        fixture_file = fixture_file,
        stan_file = stan_file,
        math_backend = backend,
        avx_header = if (backend %in% c("AVX512", "AVX2")) avx_header else NULL,
        output_rds = file.path(paths$runtime_directory,
                               paste0("nicostan_", tolower(backend), "_N", N, ".rds"))
    )
}
print(result[c("math_backend", "simd_lanes", "n_nuisance", "n_params_main", "automatic_nuisance_detection")])
