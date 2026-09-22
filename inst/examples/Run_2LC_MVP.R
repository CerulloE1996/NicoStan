#### =====================================================================================================================================
## LC_MVP: native manual-gradient example from the BayesMVP extension.
## Edit N_values to choose the three native-model benchmark sizes.
##
N_values <-  c(48L, 150L, 500L)
.example_sources <-  as.character(unlist(lapply(sys.frames(), function(frame) frame$ofile)))
if (!length(.example_sources)) .example_sources <-  sub("^--file=", "", commandArgs()[startsWith(commandArgs(), "--file=")])
if (!length(.example_sources)) stop("Source this file or run it with Rscript.")
.example_directory <-  dirname(normalizePath(tail(.example_sources, 1L), mustWork = TRUE))
source(file.path(.example_directory, "BayesMVP_native_examples.R"))
native_benchmark <-  benchmark_BayesMVP_native_example(Model_type = "LC_MVP", N_values = N_values, burnin_algorithm = "CHESSR")
native_benchmark$summary
