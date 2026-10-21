##
## ==== R_fn_compare_log_density_and_gradient_of_compiled_Stan_models_at_points.R ================================
##
## ---- The log density and its gradient of several compiled Stan models (BridgeStan libraries) of the same
##      parameters, compared with those of a reference model at the same unconstrained points -
##
## Each library is called directly through BridgeStan's C entry points (bs_model_construct_R, bs_param_unc_num_R,
## bs_log_density_gradient_R and bs_model_destruct_R; no R6 class), as in
## fn_seconds_per_lp_grad_call_of_compiled_Stan_model(). Each library must not have been loaded before in this R
## process (BridgeStan: a library loaded again is not updated if its file has changed), so the comparison is run
## in a fresh R process, and every library has a base name of its own.
## By default, each library is also loaded and evaluated in an R process of its own (callr::r()), so that no two
## libraries ever share an R process and the values of each library do not depend on which other libraries were
## loaded before it.
##
##
##
##
#### ---- the log density and gradient of one library at each point ----------------------------------------------
#' Log density and gradient of a compiled Stan model at each unconstrained point, without any R6 class
#'
#' @param model_so_file The model's compiled BridgeStan library (.so).
#' @param JSON_text_of_Stan_data The Stan data as JSON text (e.g. from fn_JSON_text_of_Stan_data_list()).
#' @param points A list of unconstrained parameter vectors, or a function of the model's unconstrained parameter
#'   names that returns that list.
#' @param propto,jacobian As in BridgeStan's log_density_gradient (default FALSE: with every constant; TRUE).
#' @param seed The seed of the model's construction (default 1).
#' @return A list with log_density (one value per point; NA where the call failed), gradient (a matrix, one row
#'   per point), points, unconstrained_parameter_names, n_unconstrained_parameters and
#'   error_message_of_each_point.
#' @noRd
fn_log_density_and_gradient_of_compiled_Stan_model_at_points <-  function( model_so_file,
                                                                           JSON_text_of_Stan_data,
                                                                           points,
                                                                           propto    = FALSE,
                                                                           jacobian  = TRUE,
                                                                           seed      = 1
) {

        model_so_file <-  normalizePath(model_so_file, mustWork = TRUE)
        library_name <-  tools::file_path_sans_ext(basename(model_so_file))
        if (is.loaded("bs_model_construct_R", PACKAGE = library_name)) {
              stop(paste0("The library ", library_name, " is already loaded in this R process; check each model ",
                          "in a fresh R process (a library loaded again is not updated if its file changed)."))
        }
        dyn.load(model_so_file)
        ##
        construction <-  .C( "bs_model_construct_R",
                             as.character(JSON_text_of_Stan_data),
                             as.integer(seed),
                             ptr_out  = raw(8),
                             err_msg  = as.character(""),
                             err_ptr  = raw(8),
                             PACKAGE  = library_name)
        if (all(construction$ptr_out == 0)) {
              stop(paste0("BridgeStan could not construct the model ", library_name, ": ", construction$err_msg))
        }
        model_pointer <-  construction$ptr_out
        n_unconstrained_parameters <-  .C( "bs_param_unc_num_R",
                                           as.raw(model_pointer),
                                           num      = as.integer(0),
                                           PACKAGE  = library_name)$num
        unconstrained_parameter_names <-  strsplit(.C( "bs_param_unc_names_R",
                                                       as.raw(model_pointer),
                                                       names_out  = as.character(""),
                                                       PACKAGE    = library_name)$names_out, ",")[[1]]
        if (is.function(points)) points <-  points(unconstrained_parameter_names)
        ##
        log_density <-  rep(NA_real_, length(points))
        gradient <-  matrix(NA_real_, nrow = length(points), ncol = n_unconstrained_parameters)
        error_message_of_each_point <-  rep("", length(points))
        for (point_index in seq_along(points)) {
              stopifnot(length(points[[point_index]]) == n_unconstrained_parameters)
              ## (a point of a model whose nuisance blocks NicoStan holds, nico_reduce_sum, carries the values
              ##  of those blocks: set before the call, as NicoStan's readers use them outside NicoStan's gradient
              ##  call; without them every log density of such a model fails, "no nuisance values")
              nuisance_values_of_the_point <-  attr(points[[point_index]], "nicostan_nuisance_values")
              if (!is.null(nuisance_values_of_the_point) &&
                  is.loaded("nicostan_set_the_nuisance_values_used_outside_NicoStan_R", PACKAGE = library_name)) {
                    ## (the routine of the model library named through do.call(), as in
                    ##  R_fn_write_nico_reduce_sum_version_of_Stan_model.R)
                    do.call(".C", list( "nicostan_set_the_nuisance_values_used_outside_NicoStan_R",
                                        as.double(nuisance_values_of_the_point),
                                        as.integer(length(nuisance_values_of_the_point)),
                                        PACKAGE = library_name))
              }
              call <-  .C( "bs_log_density_gradient_R",
                           as.raw(model_pointer),
                           as.logical(propto),
                           as.logical(jacobian),
                           as.double(points[[point_index]]),
                           val          = double(1),
                           gradient     = double(n_unconstrained_parameters),
                           return_code  = as.integer(0),
                           err_msg      = as.character(""),
                           err_ptr      = raw(8),
                           NAOK         = TRUE,
                           PACKAGE      = library_name)
              if (call$return_code == 0) {
                    log_density[point_index] <-  call$val
                    gradient[point_index, ] <-  call$gradient
              } else {
                    error_message_of_each_point[point_index] <-  call$err_msg
              }
        }
        ##
        .C("bs_model_destruct_R", as.raw(model_pointer), PACKAGE = library_name)
        ##
        return(list( log_density                    = log_density,
                     gradient                       = gradient,
                     points                         = points,
                     unconstrained_parameter_names  = unconstrained_parameter_names,
                     n_unconstrained_parameters     = n_unconstrained_parameters,
                     error_message_of_each_point    = error_message_of_each_point))

}
##
##
##
##
#### ---- each model against the reference model, for each set of points -----------------------------------------
#' Largest absolute and relative differences of the log density and gradient of each compiled Stan model from
#' those of a reference model, for each set of points
#'
#' @param reference_model_so_file The reference model's compiled BridgeStan library (.so).
#' @param other_model_so_files A named vector of the other models' libraries (same unconstrained parameters).
#' @param JSON_text_of_Stan_data The Stan data as JSON text (the same data for every model).
#' @param points A list of unconstrained parameter vectors, or a function of the reference model's unconstrained
#'   parameter names that returns that list.
#' @param point_set_of_each_point The name of the set of each point (e.g. "ordinary", "tails").
#' @param propto,jacobian As in BridgeStan's log_density_gradient (default FALSE: with every constant; TRUE).
#' @param evaluate_each_library_in_its_own_fresh_R_process TRUE (default): each library (the reference and every
#'   other model) is loaded and evaluated in an R process of its own (callr::r()), so that no two libraries are
#'   ever loaded in the same R process; FALSE: every library is loaded in this R process (the previous behaviour).
#' @return A data frame with one row per other model and set of points: the largest absolute difference and the
#'   largest relative difference (|other - reference| / |reference|, over the values whose reference is not 0)
#'   of the log density and of the gradient elements, the points whose log density is not finite (or whose call
#'   failed) in the reference and in the other model, and the reference's largest |log density|.
#' @noRd
fn_compare_log_density_and_gradient_of_compiled_Stan_models_at_points <-  function(
        reference_model_so_file,
        other_model_so_files,
        JSON_text_of_Stan_data,
        points,
        point_set_of_each_point,
        propto                                            = FALSE,
        jacobian                                          = TRUE,
        evaluate_each_library_in_its_own_fresh_R_process  = TRUE
) {

        stopifnot(!is.null(names(other_model_so_files)))
        ##
        ## ---- each library in its own fresh R process (callr::r()), or (FALSE: the previous behaviour) every
        ##      library in this R process:
        ##
        fn_evaluate_one_library <-  function(model_so_file, points_of_evaluation) {

              if (!evaluate_each_library_in_its_own_fresh_R_process) {
                    return(fn_log_density_and_gradient_of_compiled_Stan_model_at_points(
                                 model_so_file           = model_so_file,
                                 JSON_text_of_Stan_data  = JSON_text_of_Stan_data,
                                 points                  = points_of_evaluation,
                                 propto                  = propto,
                                 jacobian                = jacobian))
              }
              return(callr::r(func  = fn_log_density_and_gradient_of_compiled_Stan_model_at_points,
                              args  = list(model_so_file           = model_so_file,
                                           JSON_text_of_Stan_data  = JSON_text_of_Stan_data,
                                           points                  = points_of_evaluation,
                                           propto                  = propto,
                                           jacobian                = jacobian)))
        }

        ##
        ## (points given as a function of the unconstrained parameter names, with fresh R processes: the names
        ##  first, from the reference evaluated at no point, then the points, here, with this R process's random
        ##  numbers, as before; without fresh R processes the function is given to the reference's evaluation, as
        ##  before)
        if (is.function(points) && evaluate_each_library_in_its_own_fresh_R_process) {
              names_of_reference <-  fn_evaluate_one_library(model_so_file         = reference_model_so_file,
                                                             points_of_evaluation  = list())
              points <-  points(names_of_reference$unconstrained_parameter_names)
        }
        reference <-  fn_evaluate_one_library(model_so_file         = reference_model_so_file,
                                              points_of_evaluation  = points)
        # reference <-  fn_log_density_and_gradient_of_compiled_Stan_model_at_points(
        #                     model_so_file           = reference_model_so_file,
        #                     JSON_text_of_Stan_data  = JSON_text_of_Stan_data,
        #                     points                  = points,
        #                     propto                  = propto,
        #                     jacobian                = jacobian)
        points <-  reference$points
        stopifnot(length(point_set_of_each_point) == length(points))
        ##
        fn_largest_absolute_difference <-  function(other_values, reference_values) {

              both_finite <-  is.finite(other_values) & is.finite(reference_values)
              if (!any(both_finite)) return(NA_real_)
              return(max(abs(other_values[both_finite] - reference_values[both_finite])))
        }

        fn_largest_relative_difference <-  function(other_values, reference_values) {

              both_finite_and_reference_not_0 <-  is.finite(other_values) & is.finite(reference_values) &
                                                  (reference_values != 0)
              if (!any(both_finite_and_reference_not_0)) return(NA_real_)
              return(max(abs(other_values[both_finite_and_reference_not_0] -
                             reference_values[both_finite_and_reference_not_0]) /
                         abs(reference_values[both_finite_and_reference_not_0])))
        }

        ##
        rows <-  list()
        for (other_model_name in names(other_model_so_files)) {
              other <-  fn_evaluate_one_library(model_so_file         = other_model_so_files[[other_model_name]],
                                                points_of_evaluation  = points)
              # other <-  fn_log_density_and_gradient_of_compiled_Stan_model_at_points(
              #               model_so_file           = other_model_so_files[[other_model_name]],
              #               JSON_text_of_Stan_data  = JSON_text_of_Stan_data,
              #               points                  = points,
              #               propto                  = propto,
              #               jacobian                = jacobian)
              stopifnot(other$n_unconstrained_parameters == reference$n_unconstrained_parameters)
              for (point_set in unique(point_set_of_each_point)) {
                    in_set <-  which(point_set_of_each_point == point_set)
                    rows[[length(rows) + 1]] <-  data.frame(
                          model                                       = other_model_name,
                          point_set                                   = point_set,
                          n_points                                    = length(in_set),
                          n_points_not_finite_in_reference            = sum(!is.finite(
                                                                              reference$log_density[in_set])),
                          n_points_not_finite_in_model                = sum(!is.finite(
                                                                              other$log_density[in_set])),
                          largest_abs_log_density_of_reference        = suppressWarnings(max(abs(
                                                                              reference$log_density[in_set]),
                                                                              na.rm = TRUE)),
                          largest_absolute_difference_of_log_density  = fn_largest_absolute_difference(
                                                                              other$log_density[in_set],
                                                                              reference$log_density[in_set]),
                          largest_relative_difference_of_log_density  = fn_largest_relative_difference(
                                                                              other$log_density[in_set],
                                                                              reference$log_density[in_set]),
                          largest_absolute_difference_of_gradient     = fn_largest_absolute_difference(
                                                                              other$gradient[in_set, ],
                                                                              reference$gradient[in_set, ]),
                          largest_relative_difference_of_gradient     = fn_largest_relative_difference(
                                                                              other$gradient[in_set, ],
                                                                              reference$gradient[in_set, ]),
                          stringsAsFactors                            = FALSE)
              }
        }
        ##
        return(do.call(rbind, rows))

}

##
##
## fit draws"): a short fit of the reference's own Stan file with cmdstanr (NUTS, one chain), its draws spread
##      over its sampling iterations, then each draw put on the unconstrained scale by the reference's library
##      (BridgeStan's param_unconstrain, in a fresh R process), in the order of its unconstrained parameters.
##      Before, the points were random around 0 on the unconstrained scale, at some of which the reference itself
##      is not finite (e.g. the PK at 1,000 patients: exp(-rate x t) = 0) -
##
#' Points on the unconstrained scale from a short fit of a Stan model (for the checks of its libraries)
#'
#' @param Stan_model_file_path The Stan file of the reference library (its build arguments say which).
#' @param Stan_data_list The data of the check: a list, or the path of its JSON file.
#' @param model_so_file The reference library (BridgeStan), which puts the draws on its unconstrained scale.
#' @param n_points The number of points (draws, 10 sampling iterations apart).
#' @param n_burnin The burn-in iterations of the short fit.
#' @param seed The seed of the fit.
#' @return A list of n_points numeric vectors (unconstrained, in the library's order).
#' @noRd
fn_points_on_the_unconstrained_scale_from_a_short_fit_of_a_Stan_model <-  function(
                    Stan_model_file_path,
                    Stan_data_list,
                    model_so_file,
                    n_points,
                    n_burnin,
                    seed
) {

##
        ## ---- the data (a list, or the path of its JSON file):
##
        JSON_text_of_Stan_data <-  if (is.character(Stan_data_list) && length(Stan_data_list) == 1) {
              paste(readLines(Stan_data_list, warn = FALSE), collapse = "\n")
        } else {
              fn_JSON_text_of_Stan_data_list(Stan_data_list = Stan_data_list)
        }
        ##
        ## ---- the short fit (one chain; n_points draws, 10 iterations apart):
##
        # Stan_model <-  cmdstanr::cmdstan_model( stan_file    = Stan_model_file_path,
        #                                         quiet        = TRUE,
        #                                         cpp_options  = list(stan_threads = TRUE))
        ## (the cmdstan executable beside the Stan file, <Stan file without .stan>_cmdstan_STAN_THREADS, as
        ##  run_one_Stan_cmdstanr.R names it, built once beforehand, in its own process, and only loaded here:
        ##  a check never builds a model, and stops when the executable is missing or older than its Stan file;
        ## (a reference that only NicoStan can build - its nico_reduce_sum version,
        ##  "_automatic_nico_reduce_sum...", or a version whose nuisance block NicoStan owns,
        ##  "_nuisance_block_owned_by_NicoStan..." - is fitted through the version of the same model it was made
        ##  from, which cmdstan builds: the same file name without that part and what follows it, up to the math
        ##  backend's ending; its draws hold the reference's parameters, matched by name below)
        Stan_file_of_the_short_fit <-  sub(paste0("_(automatic_reduce_sum_chunked|automatic_nico_reduce_sum|",
                                                  "nuisance_block_owned_by_NicoStan)",
                                                  ".*?(_Stan_math_for_layout_timings)?\\.stan$"),
                                           "\\2.stan", Stan_model_file_path, perl = TRUE)
        ## (the references made from another nico_reduce_sum version rather than from a version of their own,
        ##  so that the name rule gives a file that does not exist: the version their build arguments name as
        ##  their origin, with the same data block and the same parameters apart from the nuisance block held by
        start_of_the_names_of_the_chunked_PK_versions <-  paste0("two_compartment_population_pharmacokinetics_",
                                                                 "chunked_no_loops_over_N_")
        references_made_from_another_nico_reduce_sum_version <-  data.frame(
              file_by_the_name_rule  = paste0(start_of_the_names_of_the_chunked_PK_versions, c("v3", "v4"),
                                              "_Stan_math_for_layout_timings.stan"),
              file_of_the_short_fit  = paste0(start_of_the_names_of_the_chunked_PK_versions,
                                              "v2_Stan_math_for_layout_timings.stan"),
              stringsAsFactors       = FALSE)
        made_from_another <-  match(basename(Stan_file_of_the_short_fit),
                                    references_made_from_another_nico_reduce_sum_version$file_by_the_name_rule)
        if (!is.na(made_from_another)) {
              Stan_file_of_the_short_fit <-  file.path(dirname(Stan_file_of_the_short_fit),
                                                       references_made_from_another_nico_reduce_sum_version$
                                                             file_of_the_short_fit[made_from_another])
        }
        if (Stan_file_of_the_short_fit != Stan_model_file_path) {
              if (!file.exists(Stan_file_of_the_short_fit)) {
                    stop(paste0(basename(Stan_model_file_path), " can only be built by NicoStan, and the ",
                                "version of the same model it was made from, whose short fit gives the points, ",
                                "does not exist: ", Stan_file_of_the_short_fit))
              }
              message(paste0("The points come from the short fit of ", basename(Stan_file_of_the_short_fit),
                             " (cmdstan builds it; ", basename(Stan_model_file_path), " can only be built by ",
                             "NicoStan)"))
        }
        # cmdstan_executable_file <-  sub("\\.stan$", "_cmdstan_STAN_THREADS", Stan_model_file_path)
        cmdstan_executable_file <-  sub("\\.stan$", "_cmdstan_STAN_THREADS", Stan_file_of_the_short_fit)
        if (!file.exists(cmdstan_executable_file) ||
            (file.mtime(cmdstan_executable_file) < file.mtime(Stan_file_of_the_short_fit))) {
              stop(paste0("The cmdstan executable of ", basename(Stan_file_of_the_short_fit), " is missing or ",
                          "older than its Stan file: ", cmdstan_executable_file, ". Build it first, once, in ",
                          "its own process: bash ",
                          "build_the_cmdstan_executables_of_the_reference_Stan_files_of_the_checks.sh ",
                          "(benchmark folder), or Rscript -e 'cmdstanr::cmdstan_model(stan_file = \"",
                          Stan_file_of_the_short_fit, "\", exe_file = \"", cmdstan_executable_file,
                          "\", cpp_options = list(stan_threads = TRUE))'"))
        }
        # Stan_model <-  cmdstanr::cmdstan_model( stan_file    = Stan_model_file_path,
        #                                         exe_file     = cmdstan_executable_file,
        #                                         compile      = FALSE)
        Stan_model <-  cmdstanr::cmdstan_model( stan_file    = Stan_file_of_the_short_fit,
                                                exe_file     = cmdstan_executable_file,
                                                cpp_options  = list(stan_threads = TRUE),
                                                compile      = FALSE)
        message(paste0("cmdstan executable reused: ", cmdstan_executable_file))
        ## (fixed starting values, never the sampler's random ones: every unconstrained parameter at 0)
        fit <-  Stan_model$sample( data               = Stan_data_list,
                                   init               = 0,
                                   chains             = 1,
                                   threads_per_chain  = 1,
                                   iter_warmup        = n_burnin,
                                   iter_sampling      = 10 * n_points,
                                   thin               = 10,
                                   seed               = seed,
                                   refresh            = 0,
                                   show_messages      = FALSE)
        draws <-  unclass(posterior::as_draws_matrix(fit$draws()))
        ## (a short fit with no finite draw gives no point where the reference is finite: the check would prove
        ##  nothing, so it stops here)
        if (nrow(draws) == 0 || !any(is.finite(draws[, "lp__"]))) {
              stop(paste0("The short fit of ", basename(Stan_file_of_the_short_fit), " has no finite draw: no ",
                          "point where a fit goes, and a check at its points would prove nothing"))
        }
        ##
        ## ---- each draw on the library's unconstrained scale (its constrained parameters only, in its order:
##
        ##      e.g. "omega.1" for cmdstanr's "omega[1]"), in a fresh R process:
        fn_unconstrain_the_draws <-  function(model_so_file, JSON_text_of_Stan_data, draws, seed) {

              library_name <-  tools::file_path_sans_ext(basename(model_so_file))
              dyn.load(model_so_file)
              construction <-  .C( "bs_model_construct_R", as.character(JSON_text_of_Stan_data),
                                   as.integer(seed), ptr_out = raw(8), err_msg = as.character(""),
                                   err_ptr = raw(8), PACKAGE = library_name)
              if (all(construction$ptr_out == 0)) stop(paste0("BridgeStan: ", construction$err_msg))
              model_pointer <-  construction$ptr_out
              n_parameters <-  .C( "bs_param_num_R", as.raw(model_pointer), as.logical(FALSE),
                                   as.logical(FALSE), num = as.integer(0), PACKAGE = library_name)$num
              names_of_parameters <-  strsplit(.C( "bs_param_names_R", as.raw(model_pointer), as.logical(FALSE),
                                                   as.logical(FALSE), names_out = as.character(""),
                                                   PACKAGE = library_name)$names_out, ",")[[1]]
              n_unconstrained <-  .C( "bs_param_unc_num_R", as.raw(model_pointer), num = as.integer(0),
                                      PACKAGE = library_name)$num
              names_of_the_draws <-  gsub("\\]", "", gsub("[\\[,]", ".", colnames(draws)))
              column_of_each_parameter <-  match(names_of_parameters, names_of_the_draws)
              if (length(names_of_parameters) != n_parameters || anyNA(column_of_each_parameter)) {
                    stop("The fit's draws do not have every parameter of the library")
              }
              points <-  lapply(seq_len(nrow(draws)), function(draw_index) {

                    call <-  .C( "bs_param_unconstrain_R", as.raw(model_pointer),
                                 as.double(draws[draw_index, column_of_each_parameter]),
                                 theta_unc = double(n_unconstrained), return_code = as.integer(0),
                                 err_msg = as.character(""), err_ptr = raw(8), PACKAGE = library_name)
                    if (call$return_code != 0) stop(paste0("BridgeStan param_unconstrain: ", call$err_msg))
                    call$theta_unc
              })

              .C("bs_model_destruct_R", as.raw(model_pointer), PACKAGE = library_name)
              return(points)
        }

        points <-  callr::r(func  = fn_unconstrain_the_draws,
                            args  = list(model_so_file           = normalizePath(model_so_file, mustWork = TRUE),
                                         JSON_text_of_Stan_data  = JSON_text_of_Stan_data,
                                         draws                   = draws,
                                         seed                    = seed))
        ##
        ## ---- (a reference whose nuisance blocks NicoStan holds, nico_reduce_sum, as its build arguments say:
##
        ##      its library has no Stan parameters for them, and outside NicoStan's gradient call it reads the
        ##      values set with nicostan_set_the_nuisance_values_used_outside_NicoStan_R, without which every log
        ##      every point. Each point then carries its draw's values of those blocks, in NicoStan's order:
        ##      block after block, in each block term by term (the first index) with the values of each term
        ##      together; fn_log_density_and_gradient_of_compiled_Stan_model_at_points sets them before the call)
        file_of_the_build_arguments <-  sub("_model\\.so$", "_arguments_of_build.rds", model_so_file)
        nuisance_params_held_by_NicoStan <-  NULL
        if (file.exists(file_of_the_build_arguments)) {
              arguments_of_build <-  readRDS(file_of_the_build_arguments)
              if (isTRUE(arguments_of_build$reduce_sum_nuisance_block)) {
                    nuisance_params_held_by_NicoStan <-  arguments_of_build$nuisance_params
              }
        }
        if (length(nuisance_params_held_by_NicoStan) > 0) {
              names_of_the_draws <-  colnames(draws)
              columns_of_the_nuisance_values <-  integer(0)
              for (block in nuisance_params_held_by_NicoStan) {
                    columns_of_the_block <-  which(startsWith(names_of_the_draws, paste0(block, "[")))
                    ## (a block declared in the Stan file of the short fit but of size 0 in these data, e.g. the
                    ##  network meta-analysis's beta_z_of_the_studies_with_4_index_tests when no study has 4 index
                    ## tests, has no columns in the draws and gives NicoStan no values: it is passed over; a block
                    ##  the Stan file does not declare at all still stops)
                    if (length(columns_of_the_block) == 0) {
                          text_of_the_short_fit <-  paste(readLines(Stan_file_of_the_short_fit, warn = FALSE),
                                                          collapse = "\n")
                          if (grepl(paste0("[] ;\n]", block, "\\s*;"), text_of_the_short_fit)) {
                                message(paste0("The nuisance block ", block, " has no values in these data (size ",
                                               "0 in the short fit of ", basename(Stan_file_of_the_short_fit),
                                               "): passed over"))
                                next
                          }
                          stop(paste0("The short fit of ", basename(Stan_file_of_the_short_fit),
                                      " has no values of the nuisance block ", block, " that NicoStan holds in ",
                                      basename(model_so_file)))
                    }
                    indices_of_each_value <-  do.call(rbind, lapply(strsplit(sub("^[^\\[]*\\[(.*)\\]$", "\\1",
                                                                                 names_of_the_draws[
                                                                                       columns_of_the_block]),
                                                                             ","), as.numeric))
                    order_term_by_term <-  do.call(order, as.data.frame(indices_of_each_value))
                    columns_of_the_nuisance_values <-  c(columns_of_the_nuisance_values,
                                                         columns_of_the_block[order_term_by_term])
              }
              for (point_index in seq_along(points)) {
                    attr(points[[point_index]], "nicostan_nuisance_values") <-
                          as.numeric(draws[point_index, columns_of_the_nuisance_values])
              }
              message(paste0("Each point carries the ", length(columns_of_the_nuisance_values), " values of the ",
                             "nuisance blocks NicoStan holds (", paste(nuisance_params_held_by_NicoStan,
                                                                       collapse = ", "),
                             "), from the same draw"))
        }
        return(points)

}




















