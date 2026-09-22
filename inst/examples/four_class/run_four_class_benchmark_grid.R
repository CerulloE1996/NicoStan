## Run the three four-class arms at N = 24, 48, and 96.
##
## Usage:
##   Rscript run_four_class_benchmark_grid.R /path/to/writable/output analysis
##   Rscript run_four_class_benchmark_grid.R /path/to/writable/output smoke
##
## The output directory must be writable. It is deliberately outside the installed
## example directory because CmdStanR and BridgeStan write compiled libraries and
## chain output alongside their model files.

script_file <- sub("^--file=", "", commandArgs()[startsWith(commandArgs(), "--file=")])
example_directory <- dirname(normalizePath(script_file, mustWork = TRUE))
source(file.path(example_directory, "four_class_example.R"), local = TRUE)

arguments <- commandArgs(trailingOnly = TRUE)
output_directory <- if (length(arguments)) normalizePath(arguments[1L], mustWork = FALSE) else
    file.path(tempdir(), "nicostan_four_class_benchmarks")
run_mode <- if (length(arguments) >= 2L) match.arg(arguments[2L], c("analysis", "smoke")) else "analysis"
all_Ns <- c(24L, 48L, 96L)
selected_Ns <- if (length(arguments) >= 3L) {
    selected <- strsplit(arguments[3L], ",", fixed = TRUE)[[1L]]
    selected <- suppressWarnings(as.integer(selected))
    if (!length(selected) || anyNA(selected) || any(!selected %in% all_Ns))
        stop("N_values must be a comma-separated subset of 24, 48, and 96")
    unique(selected)
} else all_Ns
settings <- if (run_mode == "analysis") {
    list(chains = 4L, warmup = 1000L, iterations = 1000L, adapt_delta = 0.99, phi_type = 2L)
} else {
    list(chains = 2L, warmup = 60L, iterations = 80L, adapt_delta = 0.9, phi_type = 1L)
}
dir.create(output_directory, recursive = TRUE, showWarnings = FALSE)

paths <- four_class_example_paths(example_directory = example_directory)
Ns <- selected_Ns
arms <- c("cmdstanr_plain", "nicostan_plain", "nicostan_avx512")
avx_header <- system.file("include", "BayesMVP", "stan_external_functions.hpp", package = "BayesMVP")
if (!nzchar(avx_header) || !file.exists(avx_header))
    stop("Install the BayesMVP extension package before running the AVX512 benchmark arm")

records <- list()
for (N in Ns) {
    fixture_file <- file.path(paths$fixture_directory, paste0("N", N), "fixture.R")
    for (arm in arms) {
        message(paste0("Running ", arm, " at N = ", N))
        arm_paths <- paths
        arm_paths$runtime_directory <- file.path(output_directory, arm, paste0("N", N))
        dir.create(arm_paths$runtime_directory, recursive = TRUE, showWarnings = FALSE)
        result <- if (arm == "cmdstanr_plain") {
            run_four_class_cmdstanr_smoke(
                paths = arm_paths, fixture_file = fixture_file,
                stan_file = paths$source_plain,
                chains = settings$chains, warmup = settings$warmup,
                iterations = settings$iterations, adapt_delta = settings$adapt_delta,
                phi_type = settings$phi_type,
                output_rds = file.path(arm_paths$runtime_directory, "result.rds")
            )
        } else {
            run_four_class_nicostan_smoke(
                paths = arm_paths, fixture_file = fixture_file,
                stan_file = if (arm == "nicostan_plain") paths$source_plain else paths$source_avx,
                math_backend = if (arm == "nicostan_plain") "Stan" else "AVX512",
                avx_header = if (arm == "nicostan_plain") NULL else avx_header,
                chains = settings$chains, warmup = settings$warmup,
                iterations = settings$iterations, adapt_delta = settings$adapt_delta,
                phi_type = settings$phi_type,
                output_rds = file.path(arm_paths$runtime_directory, "result.rds")
            )
        }
        records[[length(records) + 1L]] <- data.frame(
            arm = arm,
            N = N,
            n_nuisance = result$n_nuisance,
            n_params_main_unconstrained = result$n_params_main_unconstrained,
            main_parameter_count_constrained = result$main_parameter_count_constrained,
            min_ess_main = result$min_ess_main,
            max_rhat_main = result$max_rhat_main,
            divergences = result$divergences,
            sampling_seconds = result$sampling_seconds,
            total_seconds = result$total_seconds,
            min_ess_per_sec_sampling = result$min_ess_per_sec_sampling,
            min_ess_per_sec_total = result$min_ess_per_sec_total,
            min_ess_per_1000_gradient_work = result$min_ess_per_1000_gradient_work,
            gradient_work_basis = result$gradient_work_basis,
            phi_type = result$phi_type,
            avx_kernel_path_active = result$avx_kernel_path_active,
            avx_kernel_path = result$avx_kernel_path,
            metric_type_main = result$metric_type_main,
            M_decay_type = result$M_decay_type,
            fit_wall_seconds = result$fit_wall_seconds,
            run_mode = run_mode,
            stringsAsFactors = FALSE
        )
    }
}
benchmark_table <- do.call(rbind, records)
output_csv <- file.path(output_directory, "four_class_benchmark_grid.csv")
write.csv(benchmark_table, output_csv, row.names = FALSE)
print(benchmark_table)
message(paste0("Wrote benchmark grid: ", output_csv))
