##
## ==== R_fn_seconds_per_lp_grad_call_and_L_and_N_iter_of_each_timed_call.R ====================================
##
## ---- The seconds of one lp_grad call of a compiled Stan model, and the leapfrog steps (L) and iterations
##      (N_iter) of each timed call of Paper 1's two-run timing that follow from them -
##
## Paper 1's runners time each layout with an untimed warm-up call, then a short call and a long call, back to
## back, with the same seed and settings and a fixed trajectory length; the seconds per iteration are
## (T_long - T_short) / (n_long - n_short), so that the start-up of each call cancels. Their N_iter is lowered as
## N grows (Stan: 200 / 40 / 20 / 10 at N = 500 / 2500 / 10,000 / 50,000), so that each long call is short.
##
## Here, the same choice is made for any Stan model, from the time of its own lp_grad call (the log density and
## its gradient), which depends on neither N_iter nor L:
##   (1) fn_seconds_per_lp_grad_call_of_compiled_Stan_model() times lp_grad of the model's BridgeStan library,
##       called directly through BridgeStan's C entry points (bs_model_construct_R, bs_param_unc_num_R,
##       bs_log_density_gradient_R and bs_model_destruct_R; no R6 class), at one point of the unconstrained
##       parameters, as the median over batches of repeated calls;
##   (2) fn_L_and_N_iter_of_each_timed_call_from_seconds_per_lp_grad_call() gives the leapfrog steps of each
##       iteration (enough that one iteration has at least shortest_seconds_per_iteration of lp_grad calls, so
##       that the sampler's own work at each iteration is small next to the gradients) and the iterations of the
##       long call (enough that it lasts target_seconds_of_long_call, i.e. a fraction of a second, and never fewer
##       than fewest_iterations_of_long_call, so that the timing stays accurate).
##
## The library must not have been loaded before in this R process (BridgeStan: a library loaded again is not
## updated if its file has changed), so each model is timed in a fresh R process.
##
##
##
##
#### ---- the seconds of one lp_grad call ------------------------------------------------------------------------
#' Seconds of one lp_grad call (log density and gradient) of a compiled Stan model, without any R6 class
#'
#' @param model_so_file The model's compiled BridgeStan library (.so).
#' @param JSON_text_of_Stan_data The Stan data as JSON text (e.g. from fn_JSON_text_of_Stan_data_list()).
#' @param seconds_of_repeated_calls The total seconds of the timed calls (default 0.1).
#' @param n_batches The batches of repeated calls; the seconds per call are the median over the batches
#'   (default 5).
#' @param unconstrained_parameter_vector The point of the timing (default NULL: zeros, or else uniform values in
#'   (-2, 2) until the log density is finite).
#' @param propto,jacobian As in NicoStan's sampler (default TRUE and TRUE).
#' @param seed The seed of the model's construction and of the uniform values (default 123).
#' @param stop_if_library_already_loaded TRUE (default): stop when the library is already loaded in this R
#'   process; FALSE: use it (only when this same fresh R process loaded it, from the current file).
#' @param n_nuisance_owned_by_NicoStan 0 (default), or the size of the nuisance block of a library in NicoStan's
#'   nuisance-block form: each call is then NicoStan's gradient call of that form, with the nuisance values at
#'   the same point as the main parameters (0, or the same random values).
#' @param measure_the_memory_of_one_gradient_in_this_R_process TRUE: before the timed calls, M_bytes/chain, the
#'   memory of one gradient of this model and data, is measured in this R process
#'   ([fn_memory_of_one_gradient_measured_in_this_R_process()], the same steps as the fresh child process of
#'   [fn_estimate_initial_M_bytes_per_chain()]), so that no child process is started for it; FALSE (default):
#'   not measured.
#' @param BridgeStan_model_already_built_from_the_same_JSON_text NULL (no default; given by every caller): the
#'   model is built here from
#'   JSON_text_of_Stan_data and released at the end; or a bridgestan::StanModel already built from the same
#'   library and the same JSON text (e.g. init_object$bs_model of the fit), used as it is and never released here.
#' @param fn_whether_the_repeated_calls_are_needed NULL (no default; given by every caller): the repeated calls
#'   are always made; or a
#'   function of the estimated seconds per lp_grad call (two untimed calls, then calls lasting at least 0.01 s)
#'   that returns FALSE when the repeated calls are not needed (e.g. a picker's search that its budget rule
#'   skips): the estimate is then returned as seconds_per_lp_grad_call, with repeated_calls_made = FALSE.
#' @return A list with seconds_per_lp_grad_call, seconds_per_lp_grad_call_of_each_batch, n_calls_per_batch,
#'   n_calls_of_the_estimate, repeated_calls_made,
#'   n_unconstrained_parameters, log_density, model_so_file, memory_of_one_gradient (the list of
#'   fn_memory_of_one_gradient_measured_in_this_R_process(); NULL when not measured),
#'   seconds_of_memory_measurement and seconds_of_the_whole_call (from the start of this call, the library loaded
#'   and the model constructed, to its end).
#' @noRd
fn_seconds_per_lp_grad_call_of_compiled_Stan_model <-  function( model_so_file,
                                                                 JSON_text_of_Stan_data,
                                                                 seconds_of_repeated_calls       = 0.1,
                                                                 n_batches                       = 5,
                                                                 unconstrained_parameter_vector  = NULL,
                                                                 propto                          = TRUE,
                                                                 jacobian                        = TRUE,
                                                                 seed                            = 123,
                                                                 stop_if_library_already_loaded  = TRUE,
                                                                 n_nuisance_owned_by_NicoStan    = 0,
                                                    measure_the_memory_of_one_gradient_in_this_R_process  = FALSE,
                                            BridgeStan_model_already_built_from_the_same_JSON_text,
                                                    fn_whether_the_repeated_calls_are_needed
) {

        time_of_call_start <-  proc.time()[["elapsed"]]

        ##
        ## ---- the library, loaded once in this R process:
        ##
        model_so_file <-  normalizePath(model_so_file, mustWork = TRUE)
        library_name <-  tools::file_path_sans_ext(basename(model_so_file))
        ## (a BridgeStan model already built from the same library and the same JSON text, e.g.
        ##  init_object$bs_model of the fit: it is used as it is, never built a second time, and never released
        model_given <-  !is.null(BridgeStan_model_already_built_from_the_same_JSON_text)
        if (model_given) {
              private_fields_of_the_model <-
                    BridgeStan_model_already_built_from_the_same_JSON_text$.__enclos_env__$private
              if (!identical(normalizePath(private_fields_of_the_model$lib, mustWork = TRUE), model_so_file)) {
                    stop(paste0("fn_seconds_per_lp_grad_call_of_compiled_Stan_model: the BridgeStan model given ",
                                "was built from ", private_fields_of_the_model$lib, ", not from ", model_so_file))
              }
              library_name <-  private_fields_of_the_model$lib_name
        }
        library_already_loaded <-  is.loaded("bs_model_construct_R", PACKAGE = library_name)
        if (library_already_loaded && isTRUE(stop_if_library_already_loaded) && !model_given) {
              stop(paste0("The library ", library_name, " is already loaded in this R process; time each model ",
                          "in a fresh R process (a library loaded again is not updated if its file changed)."))
        }
        if (!library_already_loaded) dyn.load(model_so_file)
        ##
        ## ---- the model, constructed with the data (or the model given):
        ##
        # construction <-  .C( "bs_model_construct_R",
        #                      as.character(JSON_text_of_Stan_data),
        #                      as.integer(seed),
        #                      ptr_out  = raw(8),
        #                      err_msg  = as.character(""),
        #                      err_ptr  = raw(8),
        #                      PACKAGE  = library_name)
        # if (all(construction$ptr_out == 0)) {
        #       stop(paste0("BridgeStan could not construct the model ", library_name, ": ",
        #                   construction$err_msg))
        # }
        # model_pointer <-  construction$ptr_out
        if (model_given) {
              model_pointer <-  private_fields_of_the_model$model
        } else {
              construction <-  .C( "bs_model_construct_R",
                                   as.character(JSON_text_of_Stan_data),
                                   as.integer(seed),
                                   ptr_out  = raw(8),
                                   err_msg  = as.character(""),
                                   err_ptr  = raw(8),
                                   PACKAGE  = library_name)
              if (all(construction$ptr_out == 0)) {
                    stop(paste0("BridgeStan could not construct the model ", library_name, ": ",
                                construction$err_msg))
              }
              model_pointer <-  construction$ptr_out
        }
        n_unconstrained_parameters <-  .C( "bs_param_unc_num_R",
                                           as.raw(model_pointer),
                                           num      = as.integer(0),
                                           PACKAGE  = library_name)$num
        ##
        ## ---- one lp_grad call:
        ##
        fn_lp_grad_call <-  function(theta) {

              if (n_nuisance_owned_by_NicoStan > 0) {
                    ## (NicoStan's gradient call of a library in its nuisance-block form: theta holds the nuisance
                    ##  values first, then the main parameters, as NicoStan's complete vector)
                    nuisance <-  seq_len(n_nuisance_owned_by_NicoStan)
                    ## (called through do.call(): the routine is in the model library, not in NicoStan's own, and
                    ##  a .C() call written out would be registered with NicoStan's own routines when it is built)
                    return(do.call(".C", list( "nicostan_bs_log_density_gradient_with_the_nuisance_block_R",
                                               as.raw(model_pointer),
                                               as.integer(propto),
                                               as.integer(jacobian),
                                               as.double(theta[-nuisance]),
                                               as.double(theta[nuisance]),
                                               nuisance_gradient  = double(n_nuisance_owned_by_NicoStan),
                                               as.integer(n_nuisance_owned_by_NicoStan),
                                               val                = double(1),
                                               gradient           = double(n_unconstrained_parameters),
                                               return_code        = as.integer(0),
                                               err_msg            = as.character(""),
                                               err_ptr            = raw(8),
                                               NAOK               = TRUE,
                                               PACKAGE            = library_name)))
              }
              .C( "bs_log_density_gradient_R",
                  as.raw(model_pointer),
                  as.logical(propto),
                  as.logical(jacobian),
                  as.double(theta),
                  val          = double(1),
                  gradient     = double(n_unconstrained_parameters),
                  return_code  = as.integer(0),
                  err_msg      = as.character(""),
                  err_ptr      = raw(8),
                  NAOK         = TRUE,
                  PACKAGE      = library_name)
        }

        ##
        ## ---- M_bytes/chain, the memory of the first gradient of this model and data, measured in this R process
        ##      (when asked), before any timed call:
        ##
        memory_of_one_gradient <-  NULL
        seconds_of_memory_measurement <-  0
        if (isTRUE(measure_the_memory_of_one_gradient_in_this_R_process)) {
              time_of_memory_measurement_start <-  proc.time()[["elapsed"]]
              memory_of_one_gradient <-  fn_memory_of_one_gradient_measured_in_this_R_process(
                                               library_name                  = library_name,
                                               model_pointer                 = model_pointer,
                                               n_unconstrained_parameters    = n_unconstrained_parameters,
                                               n_nuisance_owned_by_NicoStan  = n_nuisance_owned_by_NicoStan,
                                               propto                        = propto,
                                               jacobian                      = jacobian,
                                               seed                          = seed)
              seconds_of_memory_measurement <-  proc.time()[["elapsed"]] - time_of_memory_measurement_start
        }
        ##
        ## ---- the point of the timing (finite log density):
        ##
        n_values_of_a_point <-  n_unconstrained_parameters + n_nuisance_owned_by_NicoStan
        theta <-  unconstrained_parameter_vector
        if (is.null(theta)) theta <-  rep(0, n_values_of_a_point)
        first_call <-  fn_lp_grad_call(theta)
        n_attempts <-  1
        set.seed(seed)
        while (((first_call$return_code != 0) || !is.finite(first_call$val)) && (n_attempts < 10)) {
              theta <-  stats::runif(n_values_of_a_point, min = -2, max = 2)
              first_call <-  fn_lp_grad_call(theta)
              n_attempts <-  n_attempts + 1
        }
        if ((first_call$return_code != 0) || !is.finite(first_call$val)) {
              stop(paste0("No point with a finite log density was found for ", library_name, " in ", n_attempts,
                          " attempts: ", first_call$err_msg))
        }
        ##
        ## ---- the calls of each batch (from the time of one more call), then the batches:
        ##
        # time_start <-  proc.time()[["elapsed"]]
        # fn_lp_grad_call(theta)
        # seconds_of_one_call <-  max(proc.time()[["elapsed"]] - time_start, 1e-6)
        ## (from enough calls to last at least 0.01 s: proc.time() counts whole milliseconds, and one call of a
        ##  small model lasts microseconds, so that one call alone can measure 0 s and every batch 0 s too)
        # n_calls_of_estimate <-  1
        # repeat {
        #       time_start <-  proc.time()[["elapsed"]]
        #       for (call in seq_len(n_calls_of_estimate)) {
        #             fn_lp_grad_call(theta)
        #       }
        #       seconds_of_estimate <-  proc.time()[["elapsed"]] - time_start
        #       if (seconds_of_estimate >= 0.01) break
        #       n_calls_of_estimate <-  2 * n_calls_of_estimate
        # }
        ## (the same, after two untimed calls, with the microsecond clock of Sys.time(): with proc.time()'s whole
        ##  milliseconds, one slow call in the estimate gave a batch of one or two calls of a fast library, which
        ##  measured 0 s, so that every batch measured 0 s, the median was 0 and the N_iter of the timed calls
        ##  was Inf, e.g. the AVX-512 Cox library at N = 50,000)
        fn_lp_grad_call(theta)
        fn_lp_grad_call(theta)
        n_calls_of_estimate <-  1
        repeat {
              time_start <-  as.numeric(Sys.time())
              for (call in seq_len(n_calls_of_estimate)) {
                    fn_lp_grad_call(theta)
              }
              seconds_of_estimate <-  as.numeric(Sys.time()) - time_start
              if (seconds_of_estimate >= 0.01) break
              n_calls_of_estimate <-  2 * n_calls_of_estimate
        }
        seconds_of_one_call <-  seconds_of_estimate / n_calls_of_estimate
        ## (the repeated calls only when they are needed, e.g. only when the search they size will run: the
        ##  pickers' budget rule is decided from this estimate, and a skipped search then costs the two untimed
        ##  still come from the full measurement below)
        if (!is.null(fn_whether_the_repeated_calls_are_needed) &&
            !isTRUE(fn_whether_the_repeated_calls_are_needed(seconds_of_one_call))) {
              fn_record_that_a_gradient_of_the_library_ran_in_this_R_process(model_so_file = model_so_file)
              if (!model_given && is.loaded("bs_model_destruct_R", PACKAGE = library_name)) {
                    .C("bs_model_destruct_R", as.raw(model_pointer), PACKAGE = library_name)
              }
              return(list( seconds_per_lp_grad_call                = seconds_of_one_call,
                           seconds_per_lp_grad_call_of_each_batch  = NA_real_,
                           n_calls_per_batch                       = 0,
                           n_calls_of_the_estimate                 = n_calls_of_estimate,
                           repeated_calls_made                     = FALSE,
                           n_unconstrained_parameters              = n_unconstrained_parameters,
                           log_density                             = first_call$val,
                           model_so_file                           = model_so_file,
                           memory_of_one_gradient                  = memory_of_one_gradient,
                           seconds_of_memory_measurement           = seconds_of_memory_measurement,
                           seconds_of_the_whole_call               = proc.time()[["elapsed"]] -
                                                                     time_of_call_start))
        }
        n_calls_per_batch <-  max(1, ceiling((seconds_of_repeated_calls / n_batches) / seconds_of_one_call))
        ##
        seconds_per_lp_grad_call_of_each_batch <-  rep(NA_real_, n_batches)
        # for (batch in seq_len(n_batches)) {
        #       time_start <-  proc.time()[["elapsed"]]
        #       for (call in seq_len(n_calls_per_batch)) {
        #             fn_lp_grad_call(theta)
        #       }
        #       seconds_per_lp_grad_call_of_each_batch[batch] <-  (proc.time()[["elapsed"]] - time_start) /
        #                                                         n_calls_per_batch
        # }
        ## (the same with the microsecond clock of Sys.time())
        for (batch in seq_len(n_batches)) {
              time_start <-  as.numeric(Sys.time())
              for (call in seq_len(n_calls_per_batch)) {
                    fn_lp_grad_call(theta)
              }
              seconds_per_lp_grad_call_of_each_batch[batch] <-  (as.numeric(Sys.time()) - time_start) /
                                                                n_calls_per_batch
        }
        seconds_per_lp_grad_call <-  stats::median(seconds_per_lp_grad_call_of_each_batch)
        if (!is.finite(seconds_per_lp_grad_call) || (seconds_per_lp_grad_call <= 0)) {
              stop(paste0("The seconds per lp_grad call of ", library_name, " are not a positive number: ",
                          seconds_per_lp_grad_call, " (", n_calls_per_batch, " calls per batch; seconds per ",
                          "call of each batch: ", paste(seconds_per_lp_grad_call_of_each_batch, collapse = " "),
                          ")"))
        }
        ## (a gradient of this library has now run in this R process: a later memory measurement in this process
        ##  would not count the gradient's code pages, already resident, so it is made in a fresh process)
        fn_record_that_a_gradient_of_the_library_ran_in_this_R_process(model_so_file = model_so_file)
        ##
        ## ---- the model is released (the library stays loaded until the R process ends), unless it was given:
        ##
        # if (is.loaded("bs_model_destruct_R", PACKAGE = library_name)) {
        #       .C("bs_model_destruct_R", as.raw(model_pointer), PACKAGE = library_name)
        # }
        if (!model_given && is.loaded("bs_model_destruct_R", PACKAGE = library_name)) {
              .C("bs_model_destruct_R", as.raw(model_pointer), PACKAGE = library_name)
        }
        ##
        return(list( seconds_per_lp_grad_call                = seconds_per_lp_grad_call,
                     seconds_per_lp_grad_call_of_each_batch  = seconds_per_lp_grad_call_of_each_batch,
                     n_calls_per_batch                       = n_calls_per_batch,
                     n_calls_of_the_estimate                 = n_calls_of_estimate,
                     repeated_calls_made                     = TRUE,
                     n_unconstrained_parameters              = n_unconstrained_parameters,
                     log_density                             = first_call$val,
                     model_so_file                           = model_so_file,
                     memory_of_one_gradient                  = memory_of_one_gradient,
                     seconds_of_memory_measurement           = seconds_of_memory_measurement,
                     seconds_of_the_whole_call               = proc.time()[["elapsed"]] - time_of_call_start))

}




#### ---- the data's JSON text and the BridgeStan model the fit has already made -------------------------------
#' The data's JSON text of init_object's data file, when it is the JSON text of the Stan data given
#'
#' NicoStan's initialise_model() writes the Stan data to a JSON file named after the digest of the data list
#' (convert_stan_data_list_to_JSON(): data_<digest>.json) and builds init_object$bs_model from it. A picker given
#' that init_object reads this text (milliseconds) instead of writing the data's JSON text again (about 0.09 s
#' for 3 MB, robust Student-t at N = 50,000), and can use init_object$bs_model for its lp_grad timing instead of
#' data given, with the chunk_size of the file, is the digest in the file's name; otherwise NULL is returned and
#' the picker writes the JSON text itself.
#' @param init_object NicoStan's init_object (its json_file_path, or Model_args_as_Rcpp_List$json_file_path).
#' @param Stan_data_list The Stan data given to the picker (with integer fields N and chunk_size).
#' @return NULL, or a list with JSON_text and chunk_size (the chunk_size written in the file).
#' @noRd
fn_JSON_text_of_the_data_file_of_init_object_when_it_is_of_the_Stan_data_given <-  function( init_object,
                                                                                            Stan_data_list
) {

        if (!is.list(init_object)) return(NULL)
        json_file_path <-  init_object$json_file_path
        if (is.null(json_file_path) && is.list(init_object$Model_args_as_Rcpp_List)) {
              json_file_path <-  init_object$Model_args_as_Rcpp_List$json_file_path
        }
        if (!is.character(json_file_path) || (length(json_file_path) != 1) || !file.exists(json_file_path)) {
              return(NULL)
        }
        digest_in_the_file_name <-  regmatches(basename(json_file_path),
                                               regexpr("data_[0-9a-f]{32}", basename(json_file_path)))
        if (length(digest_in_the_file_name) != 1) return(NULL)
        digest_in_the_file_name <-  sub("^data_", "", digest_in_the_file_name)
        JSON_text <-  readChar(con = json_file_path, nchars = file.info(json_file_path)$size, useBytes = TRUE)
        position_of_field <-  regexpr(pattern = "\"chunk_size\":", text = JSON_text, fixed = TRUE,
                                      useBytes = TRUE)
        if (position_of_field < 0) return(NULL)
        position_after_field <-  position_of_field + attr(position_of_field, "match.length")
        text_after_field <-  substr(x = JSON_text, start = position_after_field, stop = position_after_field + 30)
        chunk_size_of_the_file <-  suppressWarnings(as.integer(regmatches(text_after_field,
                                                                         regexpr("[0-9]+", text_after_field))))
        if ((length(chunk_size_of_the_file) != 1) || is.na(chunk_size_of_the_file)) return(NULL)
        ## (the Stan data given, with the chunk_size of the file, must be the data the file was written from)
        Stan_data_list_with_the_chunk_size_of_the_file <-  Stan_data_list
        Stan_data_list_with_the_chunk_size_of_the_file$chunk_size <-  chunk_size_of_the_file
        if (!identical(digest::digest(Stan_data_list_with_the_chunk_size_of_the_file), digest_in_the_file_name)) {
              return(NULL)
        }
        return(list(JSON_text = JSON_text, chunk_size = chunk_size_of_the_file))

}




#### ---- the libraries whose gradient ran in this R process ----------------------------------------------------
## (M_bytes/chain is measured in this R process only for a library none of whose gradients ran here before: a
##  second measurement would not count the code pages of the gradient, already resident; e.g. 0.91 to 0.95 of the
##  fresh process's bytes, robust Student-t and hierarchical logistic regression, N = 25,000. NicoStan records the
##  library at its lp_grad timing, its layout timings and $sample(); a gradient made outside NicoStan, e.g. with
##  bridgestan::StanModel$log_density_gradient(), is not seen)
environment_of_the_libraries_whose_gradient_ran_in_this_R_process <-  new.env()

#' Records that a gradient of a compiled Stan model's library ran in this R process
#' @param model_so_file The model's compiled BridgeStan library (.so).
#' @return NULL (invisibly).
#' @noRd
fn_record_that_a_gradient_of_the_library_ran_in_this_R_process <-  function(model_so_file) {

        if (!is.character(model_so_file) || (length(model_so_file) != 1) || !file.exists(model_so_file)) {
              return(invisible(NULL))
        }
        assign(x = normalizePath(model_so_file), value = TRUE,
               envir = environment_of_the_libraries_whose_gradient_ran_in_this_R_process)
        return(invisible(NULL))

}

#' Whether a gradient of a compiled Stan model's library ran in this R process (as recorded by NicoStan)
#' @param model_so_file The model's compiled BridgeStan library (.so).
#' @return TRUE or FALSE.
#' @noRd
fn_a_gradient_of_the_library_ran_in_this_R_process <-  function(model_so_file) {

        return(exists(x = normalizePath(model_so_file, mustWork = FALSE),
                      envir = environment_of_the_libraries_whose_gradient_ran_in_this_R_process,
                      inherits = FALSE))

}




#### ---- M_bytes/chain measured in this R process, with a model already constructed ----------------------------
#' The memory of one gradient (M_bytes/chain) of a compiled Stan model, measured in this R process
#'
#' The same steps as the fresh child process of [fn_estimate_initial_M_bytes_per_chain()]
#' (fn_measure_one_gradient_memory_in_this_process()), with the model already constructed by
#' [fn_seconds_per_lp_grad_call_of_compiled_Stan_model()], so that no child process (about 0.5 s) is started:
#' the nuisance values of a library in NicoStan's nuisance-block form set to 0; the evaluation point zero on the
#' unconstrained scale (or uniform values on (-2, 2), seeds seed to seed + 9, until the log density is finite);
#' the log density once without the gradient (propto = FALSE: no autodiff tape); gc() of R's younger generations
#' and the free heap memory returned to the kernel (malloc_trim(0), NicoStan's
#' Rcpp_fn_release_free_heap_memory()); and the pages of this thread's autodiff arena and stacks given back to the
#' kernel (the library's nicostan_bs_release_thread_memory_v1(), of NicoStan's page-release runtime), so that the
#' gradient touches them again as in a fresh process. It is used only for a library none of whose gradients ran
#' in this R process before (see fn_a_gradient_of_the_library_ran_in_this_R_process()). Then this thread is
#' pinned to the CPU it runs on, the peak resident memory is reset (/proc/self/clear_refs), the resident memory is
#' read (Rss of /proc/self/smaps_rollup), ONE log density and gradient is evaluated, and the resident memory,
#' VmHWM and VmRSS are read again; the thread's CPUs are then put back. M_bytes/chain is the resident growth plus
#' the transient part max(0, VmHWM - VmRSS).
#'
#' It differs from the fresh process only by what this process already holds: the code pages of the gradient,
#' when a gradient of this library ran here before (not counted, as in a fresh process they are; hence the
#' condition above), and free heap pages that malloc_trim(0) cannot return (as in the fresh process).
#'
#' @param library_name,model_pointer,n_unconstrained_parameters The model, as constructed by
#'   fn_seconds_per_lp_grad_call_of_compiled_Stan_model().
#' @param n_nuisance_owned_by_NicoStan The size of the nuisance block NicoStan holds (0: none).
#' @param propto,jacobian As in NicoStan's sampler.
#' @param seed The seed of the random evaluation points.
#' @return A list with M_bytes_per_chain, resident_memory_growth_bytes, transient_peak_bytes,
#'   peak_reset_succeeded, thread_memory_released (TRUE: the library's nicostan_bs_release_thread_memory_v1()
#'   was called; FALSE: the library has none), free_heap_release_result, pinned_CPU_index and
#'   evaluation_point_description.
#' @noRd
fn_memory_of_one_gradient_measured_in_this_R_process <-  function( library_name,
                                                                   model_pointer,
                                                                   n_unconstrained_parameters,
                                                                   n_nuisance_owned_by_NicoStan  = 0,
                                                                   propto                        = TRUE,
                                                                   jacobian                      = TRUE,
                                                                   seed                          = 123
) {

        ##
        ## ---- one field of /proc/self/status or /proc/self/smaps_rollup (in kB = 1024 bytes), in bytes:
        ##
        fn_field_bytes <-  function( lines_of_file,
                                     field_name) {

              field_line <-  lines_of_file[startsWith(x = lines_of_file, prefix = paste0(field_name, ":"))][1]
              field_value_text <-  trimws(x = sub(pattern = paste0("^", field_name, ":"), replacement = "",
                                                  x = field_line))
              return(1024 * as.numeric(strsplit(x = field_value_text, split = "[[:space:]]+")[[1]][1]))
        }

        fn_VmHWM_and_VmRSS_bytes <-  function() {

              status_lines <-  readLines(con = "/proc/self/status", warn = FALSE)
              return(c( VmHWM = fn_field_bytes(lines_of_file = status_lines, field_name = "VmHWM"),
                        VmRSS = fn_field_bytes(lines_of_file = status_lines, field_name = "VmRSS")))
        }

        fn_exact_resident_memory_bytes <-  function() {

              if (file.exists("/proc/self/smaps_rollup")) {
                    smaps_rollup_lines <-  readLines(con = "/proc/self/smaps_rollup", warn = FALSE)
                    return(fn_field_bytes(lines_of_file = smaps_rollup_lines, field_name = "Rss"))
              }
              return(fn_VmHWM_and_VmRSS_bytes()[["VmRSS"]])
        }

        ##
        ## ---- the nuisance values of a library in NicoStan's nuisance-block form, at 0 (the values used outside
        ##      NicoStan's gradient call, as in the fresh process):
        ##
        if (n_nuisance_owned_by_NicoStan > 0) {
              do.call(".C", list( "nicostan_set_the_nuisance_values_used_outside_NicoStan_R",
                                  as.double(rep(0, n_nuisance_owned_by_NicoStan)),
                                  as.integer(n_nuisance_owned_by_NicoStan),
                                  PACKAGE = library_name))
        }
        ##
        ## ---- the evaluation point, and the log density without the gradient (propto = FALSE: no autodiff tape):
        ##
        fn_log_density_without_gradient <-  function(theta) {

              call <-  .C( "bs_log_density_R",
                           as.raw(model_pointer),
                           as.logical(FALSE),
                           as.logical(jacobian),
                           as.double(theta),
                           val          = double(1),
                           return_code  = as.integer(0),
                           err_msg      = as.character(""),
                           err_ptr      = raw(8),
                           NAOK         = TRUE,
                           PACKAGE      = library_name)
              if (call$return_code != 0) return(NA_real_)
              return(call$val)
        }

        theta <-  rep(0, n_unconstrained_parameters)
        evaluation_point_description <-  "zero on the unconstrained scale"
        log_density_without_gradient <-  fn_log_density_without_gradient(theta)
        attempt_index <-  0
        while (!is.finite(log_density_without_gradient) && (attempt_index < 10)) {
              attempt_index <-  attempt_index + 1
              set.seed(seed = seed + attempt_index - 1)
              theta <-  stats::runif(n = n_unconstrained_parameters, min = -2, max = 2)
              evaluation_point_description <-  paste0("uniform(-2, 2) on the unconstrained scale, seed ",
                                                      seed + attempt_index - 1)
              log_density_without_gradient <-  fn_log_density_without_gradient(theta)
        }
        if (!is.finite(log_density_without_gradient)) {
              stop(paste0("fn_memory_of_one_gradient_measured_in_this_R_process: the log density of ",
                          library_name, " is not finite at zero or at 10 random points."))
        }
        ##
        ## ---- R's own allocations settled, the free heap memory returned to the kernel, and this thread's
        ##      autodiff arena and stacks given back to the kernel (a gradient of this library may have run here):
        ##
        ## invisible(gc(verbose = FALSE, full = TRUE))
        ## (a collection of R's younger generations only: a full one costs about 0.1 s in a process with NicoStan
        ##  loaded, and gave the same bytes to within 1%, robust Student-t and hierarchical logistic regression,
        ##  N = 25,000)
        invisible(gc(verbose = FALSE, full = FALSE))
        free_heap_release_result <-  tryCatch(expr  = as.numeric(Rcpp_fn_release_free_heap_memory()),
                                              error = function(condition) NA_real_)
        thread_memory_released <-  FALSE
        ## (the routine called by its address in the model's library, never by its name in .C(): NicoStan's
        ##  routine registration lists the name in every .C() call - a string or a variable - as a routine of
        ##  NicoStan.so itself, and this routine is only in the model's library, not in NicoStan's dummy model,
        ##  so NicoStan.so would not load)
        if (is.loaded("nicostan_bs_release_thread_memory_v1", PACKAGE = library_name)) {

              page_release_routine_of_the_model_library <-  getNativeSymbolInfo(
                                                                      "nicostan_bs_release_thread_memory_v1",
                                                                      PACKAGE = library_name)
              ## (an int function without arguments, called for its effect; its status is not returned by .C())
              invisible(.C(page_release_routine_of_the_model_library$address))
              thread_memory_released <-  TRUE
        }

        ##
        ## ---- this thread pinned to the CPU it runs on (field 39 of /proc/self/stat, 0-based), so that the
        ##      kernel's per-CPU counters of resident memory add at most one batch of error; put back afterwards:
        ##
        CPUs_of_thread_before_pinning <-  tryCatch(expr  = parallel::mcaffinity(),
                                                   error = function(condition) NULL)
        on.exit(if (length(CPUs_of_thread_before_pinning) > 1) {

                      invisible(tryCatch(expr = parallel::mcaffinity(affinity = CPUs_of_thread_before_pinning),
                                         error = function(condition) NULL))
                }, add = TRUE)

        pinned_CPU_index <-  NA_real_
        if (length(CPUs_of_thread_before_pinning) > 1) {

              process_stat_line <-  readLines(con = "/proc/self/stat", warn = FALSE)[1]
              process_stat_fields <-  strsplit(x = sub(pattern = "^.*\\) ", replacement = "",
                                                       x = process_stat_line),
                                               split = " ", fixed = TRUE)[[1]]
              current_CPU_index <-  suppressWarnings(as.numeric(process_stat_fields[37])) + 1
              pinned_CPU_index <-  if (isTRUE(current_CPU_index %in% CPUs_of_thread_before_pinning)) {
                    current_CPU_index
              } else {
                    CPUs_of_thread_before_pinning[1]
              }
              invisible(tryCatch(expr = parallel::mcaffinity(affinity = pinned_CPU_index),
                                 error = function(condition) NULL))
        }

        ##
        ## ---- the peak reset, the readings before, ONE log density and gradient, and the readings after:
        ##
        page_size_bytes <-  4096
        resident_memory_counter_batch_bytes <-  max(32, 2 * parallel::detectCores(logical = TRUE)) *
                                                page_size_bytes
        peak_reset_tolerance_bytes <-  max(1024^2, 4 * resident_memory_counter_batch_bytes)
        VmHWM_before_reset_bytes <-  fn_VmHWM_and_VmRSS_bytes()[["VmHWM"]]
        peak_reset_write_succeeded <-  tryCatch(expr    = { cat("5", file = "/proc/self/clear_refs"); TRUE },

                                                warning = function(condition) FALSE,
                                                error   = function(condition) FALSE)
        VmHWM_and_VmRSS_before_gradient <-  fn_VmHWM_and_VmRSS_bytes()
        peak_reset_succeeded <-  peak_reset_write_succeeded &&
                                 ((VmHWM_and_VmRSS_before_gradient[["VmHWM"]] < VmHWM_before_reset_bytes) ||
                                  (VmHWM_and_VmRSS_before_gradient[["VmHWM"]] -
                                   VmHWM_and_VmRSS_before_gradient[["VmRSS"]] <= peak_reset_tolerance_bytes))
        resident_memory_before_gradient_bytes <-  fn_exact_resident_memory_bytes()
        gradient_call <-  .C( "bs_log_density_gradient_R",
                              as.raw(model_pointer),
                              as.logical(propto),
                              as.logical(jacobian),
                              as.double(theta),
                              val          = double(1),
                              gradient     = double(n_unconstrained_parameters),
                              return_code  = as.integer(0),
                              err_msg      = as.character(""),
                              err_ptr      = raw(8),
                              NAOK         = TRUE,
                              PACKAGE      = library_name)
        VmHWM_and_VmRSS_after_gradient <-  fn_VmHWM_and_VmRSS_bytes()
        resident_memory_after_gradient_bytes <-  fn_exact_resident_memory_bytes()
        if (gradient_call$return_code != 0) {

              stop(paste0("fn_memory_of_one_gradient_measured_in_this_R_process: the gradient of ", library_name,
                          " failed: ", gradient_call$err_msg))
        }

        ##
        ## ---- the resident growth (exact) plus the transient part (touched and freed before the gradient ended):
        ##
        resident_memory_growth_bytes <-  resident_memory_after_gradient_bytes -
                                         resident_memory_before_gradient_bytes
        transient_peak_bytes <-  max(0, VmHWM_and_VmRSS_after_gradient[["VmHWM"]] -
                                        VmHWM_and_VmRSS_after_gradient[["VmRSS"]])
        M_bytes_per_chain <-  resident_memory_growth_bytes + if (peak_reset_succeeded) transient_peak_bytes else 0
        ##
        return(list( M_bytes_per_chain                       = M_bytes_per_chain,
                     resident_memory_growth_bytes            = resident_memory_growth_bytes,
                     transient_peak_bytes                    = transient_peak_bytes,
                     peak_reset_succeeded                    = peak_reset_succeeded,
                     thread_memory_released                  = thread_memory_released,
                     free_heap_release_result                = free_heap_release_result,
                     pinned_CPU_index                        = pinned_CPU_index,
                     evaluation_point_description            = evaluation_point_description))

}




## (the version before L was lowered only after N_iter reaches 1, kept as it was)
##### ---- L and N_iter of each timed call ------------------------------------------------------------------------
# #' The leapfrog steps (L) and the iterations (N_iter) of each timed call of Paper 1's two-run timing, from the
# #' seconds of one lp_grad call
# #'
# #' L stays at L_preferred (10, as in Paper 1 and its reference grid), and N_iter of the long call is what is
# #' lowered: N_iter = target_seconds_of_long_call / (L x seconds per lp_grad call), so that it scales with the
# #' model's lp_grad time (and hence with N), never below fewest_iterations_of_long_call. Only when even
# #' fewest_iterations_of_long_call iterations at L_preferred would last longer than
# #' target_seconds_of_long_call is L lowered (a last resort; with L = 1 the timings were found to be inaccurate).
# #'
# #' @param seconds_per_lp_grad_call From fn_seconds_per_lp_grad_call_of_compiled_Stan_model().
# #' @param target_seconds_of_long_call The seconds of lp_grad calls in the long call (default 0.2).
# #' @param L_preferred The leapfrog steps of each timed iteration (default 10).
# #' @param fewest_iterations_of_long_call The fewest iterations of the long call (default 3).
# #' @param n_iterations_of_short_call The iterations of the short call (default 1, as in Paper 1's runners).
# #' @param L_may_be_lowered_when_the_fewest_iterations_last_longer_than_the_target TRUE (default): L lowered as a
# #'   last resort; FALSE: L stays at L_preferred whatever the seconds of the long call (the ground truth, where
# #'   every math backend is timed at the same L, so that their throughputs are compared at the same L).
# #' @return A list with L, N_iter_of_long_call, N_iter_of_short_call, predicted_seconds_per_iteration,
# #'   predicted_seconds_of_long_call (one chain, one thread, without the sampler's own work) and L_lowered (TRUE
# #'   when L is below L_preferred).
# #' @noRd
# fn_L_and_N_iter_of_each_timed_call_from_seconds_per_lp_grad_call <-  function(
#         seconds_per_lp_grad_call,
#         target_seconds_of_long_call     = 0.2,
#         L_preferred                     = 10,
#         fewest_iterations_of_long_call  = 3,
#         n_iterations_of_short_call      = 1,
#         L_may_be_lowered_when_the_fewest_iterations_last_longer_than_the_target  = TRUE
# ) {
#
#         ##
#         ## ---- N_iter of the long call at L_preferred, never fewer than fewest_iterations_of_long_call:
#         ##
#         L <-  L_preferred
#         N_iter_of_long_call <-  floor(target_seconds_of_long_call / (L * seconds_per_lp_grad_call))
#         N_iter_of_long_call <-  max(fewest_iterations_of_long_call,
#                                     n_iterations_of_short_call + 1,
#                                     N_iter_of_long_call)
#         ##
#         ## ---- (a last resort) L lowered only when even the fewest iterations at L_preferred last longer than
#         ##      target_seconds_of_long_call:
#         ##
#         if (isTRUE(L_may_be_lowered_when_the_fewest_iterations_last_longer_than_the_target) &&
#             (N_iter_of_long_call * L * seconds_per_lp_grad_call > target_seconds_of_long_call)) {
#              L <-  max(1, floor(target_seconds_of_long_call / (N_iter_of_long_call * seconds_per_lp_grad_call)))
#         }
#         predicted_seconds_per_iteration <-  L * seconds_per_lp_grad_call
#         ##
#         return(list( L                                = L,
#                      N_iter_of_long_call              = N_iter_of_long_call,
#                      N_iter_of_short_call             = n_iterations_of_short_call,
#                      predicted_seconds_per_iteration  = predicted_seconds_per_iteration,
#                      predicted_seconds_of_long_call   = N_iter_of_long_call * predicted_seconds_per_iteration,
#                      L_lowered                        = (L < L_preferred)))
#
# }
##
##
#### ---- L and N_iter of each timed call ------------------------------------------------------------------------
#' The leapfrog steps (L) and the iterations (N_iter) of each timed call, from the measured seconds of one
#' lp_grad call, case by case
#'
#' The algorithm: L starts at L_preferred (10, the baseline of Paper 1); N_iter is lowered first, as
#' N_iter = floor(target_seconds_of_long_call / (L x seconds per lp_grad call)), down to 1; only when even
#' 1 iteration at L_preferred lasts longer than target_seconds_of_long_call is L lowered, as
#' L = floor(target_seconds_of_long_call / seconds per lp_grad call), down to 1. Hence a timed call lasts between
#' about half of and target_seconds_of_long_call, whatever the model, N and math backend. Throughputs at
#' different L are compared per L_preferred leapfrog steps (TP_iter x L / L_preferred).
#'
#' @param seconds_per_lp_grad_call From fn_seconds_per_lp_grad_call_of_compiled_Stan_model(), or the seconds per
#'   gradient of a layout already timed (its seconds per iteration / its L).
#' @param target_seconds_of_long_call The seconds of the timed call (default 0.2).
#' @param L_preferred The leapfrog steps of each timed iteration before any is taken away (default 10).
#' @param fewest_iterations_of_long_call The fewest iterations of the timed call before L is lowered (default 1).
#' @param n_iterations_of_short_call The iterations of the short call of a two-run timing (default 0: the one-call
#'   timing, whose timed iterations follow its untimed iteration in the same C++ call).
#' @param L_may_be_lowered_when_the_fewest_iterations_last_longer_than_the_target Retired: L is always lowered
#'   FALSE stops with an error.
#' @return A list with L, N_iter_of_long_call, N_iter_of_short_call, predicted_seconds_per_iteration,
#'   predicted_seconds_of_long_call (one chain, one thread, without the sampler's own work) and L_lowered (TRUE
#'   when L is below L_preferred).
#' @noRd
fn_L_and_N_iter_of_each_timed_call_from_seconds_per_lp_grad_call <-  function(
        seconds_per_lp_grad_call,
        target_seconds_of_long_call     = 0.2,
        L_preferred                     = 10,
        fewest_iterations_of_long_call  = 1,
        n_iterations_of_short_call      = 0,
        L_may_be_lowered_when_the_fewest_iterations_last_longer_than_the_target  = TRUE
) {

        if (!isTRUE(L_may_be_lowered_when_the_fewest_iterations_last_longer_than_the_target)) {
              stop(paste0("L_may_be_lowered_when_the_fewest_iterations_last_longer_than_the_target = FALSE is ",
                          "retired: L and N_iter always come from the lp_grad time (N_iter down to 1 first, ",
                          "then L)"))
        }
        ##
        ## ---- N_iter of the timed call at L_preferred, lowered first, down to fewest_iterations_of_long_call:
        ##
        if (!is.finite(seconds_per_lp_grad_call) || (seconds_per_lp_grad_call <= 0)) {
              stop(paste0("seconds_per_lp_grad_call must be a positive number, not ", seconds_per_lp_grad_call))
        }
        L <-  L_preferred
        N_iter_of_long_call <-  floor(target_seconds_of_long_call / (L * seconds_per_lp_grad_call))
        N_iter_of_long_call <-  max(fewest_iterations_of_long_call,
                                    n_iterations_of_short_call + 1,
                                    N_iter_of_long_call)
        ##
        ## ---- then L, only when even those iterations at L_preferred last longer than the target:
        ##
        if (N_iter_of_long_call * L * seconds_per_lp_grad_call > target_seconds_of_long_call) {
              L <-  max(1, floor(target_seconds_of_long_call / (N_iter_of_long_call * seconds_per_lp_grad_call)))
        }
        predicted_seconds_per_iteration <-  L * seconds_per_lp_grad_call
        ##
        return(list( L                                = L,
                     N_iter_of_long_call              = N_iter_of_long_call,
                     N_iter_of_short_call             = n_iterations_of_short_call,
                     predicted_seconds_per_iteration  = predicted_seconds_per_iteration,
                     predicted_seconds_of_long_call   = N_iter_of_long_call * predicted_seconds_per_iteration,
                     L_lowered                        = (L < L_preferred)))

}























