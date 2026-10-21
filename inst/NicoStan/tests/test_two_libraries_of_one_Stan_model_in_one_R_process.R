##
## ==== test_two_libraries_of_one_Stan_model_in_one_R_process.R ==================================================
##
## ---- Loads two or more libraries of the same Stan model (e.g. its Stan math library, then its AVX2 library)
##      into ONE R process with BridgeStan, in the order given, evaluates the log density and its gradient of each
##      at the same points, and reports the largest relative difference of each from the first: they must agree to
##      rounding (about 1e-8), whatever library was loaded first -
##
## Usage:   Rscript test_two_libraries_of_one_Stan_model_in_one_R_process.R
## <data .rds (the data list, or a list with stan_data_list)> <N_chunks> <library .so> <library .so> ...
##
{
      arguments <-  commandArgs(trailingOnly = TRUE)
      data_file <-  arguments[1]
      N_chunks <-  as.numeric(arguments[2])
      libraries <-  arguments[-(1:2)]
}
##
## ---- the data, with chunk_size from N_chunks, written once as JSON:
##
data_object <-  readRDS(data_file)
data_list <-  if (!is.null(data_object$stan_data_list)) data_object$stan_data_list else data_object
N <-  if (!is.null(data_list$N)) data_list$N else data_list$T
data_list$chunk_size <-  as.integer(ceiling(N / N_chunks))
json_file <-  tempfile(fileext = ".json")
jsonlite::write_json(data_list, json_file, auto_unbox = TRUE, digits = NA)
##
## ---- every library in this one process, in the order given:
##
models <-  lapply(libraries, function(library) bridgestan::StanModel$new(library, data = json_file, seed = 1,
                                                                         warn = TRUE))
n_unconstrained <-  models[[1]]$param_unc_num()
##
set.seed(2026)
points <-  lapply(1:8, function(point_index) stats::rnorm(n_unconstrained, mean = 0,
                                                           sd = if (point_index <= 4) 0.3 else 1))
first <-  lapply(points, function(point) models[[1]]$log_density_gradient(point, propto = TRUE, jacobian = TRUE))
for (index in seq_along(models)[-1]) {

      largest_relative_difference <-  0
      for (point_index in seq_along(points)) {
            other <-  models[[index]]$log_density_gradient(points[[point_index]], propto = TRUE, jacobian = TRUE)
            differences <-  c(abs(first[[point_index]]$val - other$val) / max(1, abs(first[[point_index]]$val)),
                              abs(first[[point_index]]$gradient - other$gradient) /
                                  pmax(1, abs(first[[point_index]]$gradient)))
            largest_relative_difference <-  max(largest_relative_difference, differences)
      }
      message(paste0(basename(libraries[index]), " vs ", basename(libraries[1]), ": largest relative difference ",
                     formatC(largest_relative_difference, format = "e", digits = 2),
                     if (largest_relative_difference < 1e-6) "  (agree)" else "  (DISAGREE)"))
}





















