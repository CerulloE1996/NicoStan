#### =====================================================================================================================================
## Rscript run_one.R MODEL ENGINE MATH_BACKEND OUTPUT_DIRECTORY [PROFILE]
## R_LIBS determines which installed NicoStan build is used.
##
script <-  sub("^--file=", "", commandArgs()[startsWith(commandArgs(), "--file=")])
source(file.path(dirname(normalizePath(script, mustWork = TRUE)), "paper3_examples.R"))
arguments <-  commandArgs(trailingOnly = TRUE)
stopifnot(length(arguments) >= 4L)
result <-  run_paper3_example(model = arguments[1L], engine = arguments[2L], math_backend = arguments[3L],
                              output_dir = arguments[4L], profile = if (length(arguments) >= 5L) arguments[5L] else "smoke")
message(paste0("PASS: ", result$example$model, " / ", result$engine, " / ", result$math_backend,
               "; diagnostic target met: ", result$diagnostic_target_met))
