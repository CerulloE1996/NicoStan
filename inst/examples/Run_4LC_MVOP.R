## Source-friendly four-class entrypoint.
##
## Visible benchmark settings:
N_vec <- c(24L, 48L, 96L)
four_class_benchmark_mode <- "analysis"  # or "smoke"
four_class_benchmark_output <- file.path(getwd(), "NicoStan_four_class_results")

## Capture this entrypoint's own source path while source() still exposes it.
## Looking up sys.frame(1)$ofile inside run_4LC_MVOP() instead sees the caller's
## frame and can resolve the wrong four_class runner (or no runner at all).
.run_4lc_sources <- as.character(unlist(lapply(sys.frames(), function(frame) frame$ofile)))
if (!length(.run_4lc_sources))
    .run_4lc_sources <- sub("^--file=", "", commandArgs()[startsWith(commandArgs(), "--file=")])
.run_4lc_source_file <- if (length(.run_4lc_sources)) tail(.run_4lc_sources, 1L) else NULL

run_4LC_MVOP <- function(
    N_values = N_vec,
    mode = four_class_benchmark_mode,
    output_directory = four_class_benchmark_output) {
    N_values <- as.integer(N_values)
    if (!length(N_values) || anyNA(N_values) || any(!N_values %in% N_vec))
        stop("N_values must be selected from 24, 48, and 96")
    if (!mode %in% c("analysis", "smoke"))
        stop("mode must be 'analysis' or 'smoke'")
    N_values <- unique(N_values)
    example_runner <- if (!is.null(.run_4lc_source_file)) file.path(
        dirname(normalizePath(.run_4lc_source_file, mustWork = FALSE)),
                                "four_class", "run_four_class_benchmark_grid.R")
    else NA_character_
    if (!file.exists(example_runner)) {
        example_runner <- file.path(system.file("examples", "four_class", package = "NicoStan"),
                                    "run_four_class_benchmark_grid.R")
    }
    if (!file.exists(example_runner)) stop("Could not locate run_four_class_benchmark_grid.R")
    ## Pass the validated selection explicitly. The child runner owns the arm
    ## loop, while this entrypoint owns the requested N subset.
    status <- system2(command = file.path(R.home("bin"), "Rscript"),
                      args = c(shQuote(example_runner), shQuote(output_directory),
                               shQuote(mode), shQuote(paste(N_values, collapse = ","))))
    if (!identical(as.integer(status), 0L))
        stop("run_four_class_benchmark_grid.R failed with status ", status)
    invisible(normalizePath(file.path(output_directory, "four_class_benchmark_grid.csv"),
                            mustWork = FALSE))
}

if (!interactive() && identical(Sys.getenv("NICO_RUN_4LC_MVOP"), "1"))
    run_4LC_MVOP()
