#### =================================================================================================
## R_fn_fresh_process.R
##
## Run one complete fit in a clean R process. BridgeStan's R6 object is rebuilt in the parent after
## the child has returned, rather than attempting to serialise its external pointer.
#### =================================================================================================

#' Run R_fn_sample_model() in a fresh R process
#'
#' @param arguments Named list of arguments supplied to R_fn_sample_model().
#' @param provider_package Name of the package providing R_fn_sample_model().
#' @return The standard R_fn_sample_model() result, with execution_process metadata.
#' @keywords internal
fn_sample_model_in_fresh_R_process <-  function(arguments,
                                               provider_package = .nicostan_provider_package) {

        stop_unserialisable <-  function(path, kind) {
                stop(paste0("Fresh-process sampler cannot transport ", kind, " at ", path,
                            ". Set run_in_fresh_R_process = FALSE to diagnose or run this fit in the current R process."),
                     call. = FALSE)
        }

        ##
        validate_transport <-  function(x, path = "value") {
                if (is.environment(x)) stop_unserialisable(path = path, kind = "an environment")
                if (typeof(x) %in% c("externalptr", "weakref")) stop_unserialisable(path = path, kind = "an external pointer")
                if (inherits(x, "connection")) stop_unserialisable(path = path, kind = "a connection")
                if (is.function(x)) stop_unserialisable(path = path, kind = "a function closure")
                if (is.null(x)) return(invisible(TRUE))
                if (is.atomic(x)) {
                        x_attributes <-  attributes(x)
                        for (attribute_name in names(x_attributes)) {
                                validate_transport(x = x_attributes[[attribute_name]],
                                                   path = paste0(path, " attribute ", attribute_name))
                        }
                        return(invisible(TRUE))
                }
                ##
                if (isS4(x)) {
                        for (slot_name in methods::slotNames(x)) {
                                validate_transport(x = methods::slot(object = x, name = slot_name),
                                                   path = paste0(path, "@", slot_name))
                        }
                }
                ##
                if (is.list(x)) {
                        x_names <-  names(x)
                        for (ii in seq_along(x)) {
                                item_name <-  if (!is.null(x_names) && nzchar(x_names[[ii]])) paste0("$", x_names[[ii]]) else paste0("[[", ii, "]]")
                                validate_transport(x = x[[ii]], path = paste0(path, item_name))
                        }
                } else if (is.pairlist(x)) {
                        x_names <-  names(x)
                        for (ii in seq_along(x)) {
                                item_name <-  if (!is.null(x_names) && nzchar(x_names[[ii]])) paste0("$", x_names[[ii]]) else paste0("[[", ii, "]]")
                                validate_transport(x = x[[ii]], path = paste0(path, item_name))
                        }
                } else if (is.language(x) && !is.symbol(x)) {
                        validate_transport(x = as.list(x), path = paste0(path, " call"))
                }
                ##
                x_attributes <-  attributes(x)
                for (attribute_name in names(x_attributes)) {
                        validate_transport(x = x_attributes[[attribute_name]],
                                           path = paste0(path, " attribute ", attribute_name))
                }
                invisible(TRUE)
        }

        ##
        capture_rng <-  function() {
                has_seed <-  exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
                list(kind = RNGkind(),
                     has_seed = has_seed,
                     state = if (has_seed) get(".Random.seed", envir = .GlobalEnv, inherits = FALSE) else NULL)
        }

        restore_rng <-  function(snapshot) {
                if (!is.list(snapshot) || is.null(snapshot$kind)) {
                        stop("Invalid RNG snapshot returned by the fresh-process sampler.", call. = FALSE)
                }
                do.call(what = RNGkind, args = as.list(snapshot$kind))
                if (isTRUE(snapshot$has_seed)) {
                        assign(".Random.seed", snapshot$state, envir = .GlobalEnv)
                } else if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
                        rm(".Random.seed", envir = .GlobalEnv)
                }
                invisible(NULL)
        }

        ##
        if (!is.list(arguments) || is.null(names(arguments)) || any(!nzchar(names(arguments)))) {
                stop("arguments must be a named list of R_fn_sample_model() arguments.", call. = FALSE)
        }
        if (!is.character(provider_package) || length(provider_package) != 1L ||
            is.na(provider_package) || !nzchar(provider_package)) {
                stop("provider_package must be one non-empty package name.", call. = FALSE)
        }
        if (!"init_object" %in% names(arguments)) {
                stop("Fresh-process sampling requires an init_object argument.", call. = FALSE)
        }
        if (!is.list(arguments$init_object) || is.environment(arguments$init_object)) {
                stop_unserialisable(path = "arguments$init_object", kind = "an unsupported initialisation object")
        }

        ##
        ## R_fn_sample_model() reads exactly these fields before its first internal reinitialisation.
        ## Keeping this whitelist avoids sending a BridgeStan R6 object or stale native state.
        init_metadata_names <-  c("Model_type",
                                 "Stan_model_file_path",
                                 "Stan_cpp_user_header",
                                 "Stan_cpp_flags",
                                 "stanc_args",
                                 "make_args")
        arguments$init_object <-  setNames( object = lapply(X = init_metadata_names, FUN = function(field_name) {
                                                                arguments$init_object[[field_name]]
                                                        }),
                                            nm = init_metadata_names)

        ##
        model_options <-  options()
        model_options <-  model_options[startsWith(names(model_options), "BayesMVP_")]
        parent_rng <-  capture_rng()
        parent_libpath <-  .libPaths()
        parent_working_dir <-  getwd()
        validate_transport(x = arguments, path = "arguments")
        validate_transport(x = model_options, path = "BayesMVP options")
        validate_transport(x = parent_rng, path = "RNG snapshot")
        validate_transport(x = parent_libpath, path = ".libPaths()")
        validate_transport(x = parent_working_dir, path = "getwd()")

        ##
        ## Keep the pre-call stream if no child result arrives, otherwise the child's post-fit stream.
        ## This also covers an error while reconstructing the returned BridgeStan object.
        child_rng_applied <-  FALSE
        on.exit({
                try(expr = restore_rng(snapshot = if (isTRUE(child_rng_applied)) child_rng else parent_rng), silent = TRUE)
        }, add = TRUE)

        ##
        ## This worker is self-contained and runs with baseenv() as its enclosing environment. It loads
        ## the provider only after callr has set the requested library paths and working directory.
        worker <-  function(arguments,
                           provider_package,
                           libpath,
                           working_dir,
                           rng_snapshot,
                           model_options) {

                validate_worker_transport <-  function(x, path = "value") {
                        fail <-  function(kind) {
                                stop(paste0("Fresh-process sampler cannot transport ", kind, " at ", path,
                                            ". Set run_in_fresh_R_process = FALSE to diagnose or run this fit in the current R process."),
                                     call. = FALSE)
                        }
                        if (is.environment(x)) fail(kind = "an environment")
                        if (typeof(x) %in% c("externalptr", "weakref")) fail(kind = "an external pointer")
                        if (inherits(x, "connection")) fail(kind = "a connection")
                        if (is.function(x)) fail(kind = "a function closure")
                        if (is.null(x)) return(invisible(TRUE))
                        if (is.atomic(x)) {
                                x_attributes <-  attributes(x)
                                for (attribute_name in names(x_attributes)) {
                                        validate_worker_transport(x = x_attributes[[attribute_name]],
                                                                  path = paste0(path, " attribute ", attribute_name))
                                }
                                return(invisible(TRUE))
                        }
                        ##
                        if (isS4(x)) {
                                for (slot_name in methods::slotNames(x)) {
                                        validate_worker_transport(x = methods::slot(object = x, name = slot_name),
                                                                  path = paste0(path, "@", slot_name))
                                }
                        }
                        ##
                        if (is.list(x)) {
                                x_names <-  names(x)
                                for (ii in seq_along(x)) {
                                        item_name <-  if (!is.null(x_names) && nzchar(x_names[[ii]])) paste0("$", x_names[[ii]]) else paste0("[[", ii, "]]")
                                        validate_worker_transport(x = x[[ii]], path = paste0(path, item_name))
                                }
                        } else if (is.pairlist(x)) {
                                x_names <-  names(x)
                                for (ii in seq_along(x)) {
                                        item_name <-  if (!is.null(x_names) && nzchar(x_names[[ii]])) paste0("$", x_names[[ii]]) else paste0("[[", ii, "]]")
                                        validate_worker_transport(x = x[[ii]], path = paste0(path, item_name))
                                }
                        } else if (is.language(x) && !is.symbol(x)) {
                                validate_worker_transport(x = as.list(x), path = paste0(path, " call"))
                        }
                        ##
                        x_attributes <-  attributes(x)
                        for (attribute_name in names(x_attributes)) {
                                validate_worker_transport(x = x_attributes[[attribute_name]],
                                                          path = paste0(path, " attribute ", attribute_name))
                        }
                        invisible(TRUE)
                }

                ##
                capture_worker_rng <-  function() {
                        has_seed <-  exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
                        list(kind = RNGkind(),
                             has_seed = has_seed,
                             state = if (has_seed) get(".Random.seed", envir = .GlobalEnv, inherits = FALSE) else NULL)
                }

                ##
                condition_record <-  function(condition) {
                        condition_call <-  conditionCall(condition)
                        list(message = conditionMessage(condition),
                             class = class(condition),
                             call = if (is.null(condition_call)) NULL else paste(deparse(condition_call), collapse = ""))
                }

                setwd(dir = working_dir)
                .libPaths(new = libpath)
                fit_result <-  tryCatch({
                        ## Loading a namespace may use RNG, so restore caller state after it has loaded.
                        loadNamespace(package = provider_package, lib.loc = libpath)
                        do.call(what = RNGkind, args = as.list(rng_snapshot$kind))
                        if (isTRUE(rng_snapshot$has_seed)) {
                                assign(".Random.seed", rng_snapshot$state, envir = .GlobalEnv)
                        } else if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
                                rm(".Random.seed", envir = .GlobalEnv)
                        }
                        if (length(model_options)) do.call(what = options, args = model_options)

                        provider_entrypoint <-  getExportedValue(ns = provider_package, name = "R_fn_sample_model")
                        if (!is.function(provider_entrypoint) ||
                            !"run_in_fresh_R_process" %in% names(formals(provider_entrypoint))) {
                                stop(paste0("The loaded provider '", provider_package,
                                            "' does not expose the updated R_fn_sample_model(run_in_fresh_R_process = ...) entry point."),
                                     call. = FALSE)
                        }
                        ## Explicit recursion guard for direct helper calls as well as normal dispatch.
                        arguments$run_in_fresh_R_process <-  FALSE
                        validate_worker_transport(x = arguments, path = "arguments")
                        output <-  do.call(what = provider_entrypoint, args = arguments)
                        if (!is.list(output) || is.null(output$init_object) || !is.list(output$init_object)) {
                                stop("R_fn_sample_model() returned no serialisable init_object.", call. = FALSE)
                        }
                        ## Remove the only expected native pointer before the result crosses the boundary.
                        output$init_object$bs_model <-  NULL
                        validate_worker_transport(x = output, path = "R_fn_sample_model result")
                        list(ok = TRUE, output = output)
                }, error = function(condition) {
                        list(ok = FALSE, error = condition_record(condition))
                })

                ##
                postfit_rng <-  capture_worker_rng()
                fit_result$rng_kind <-  postfit_rng$kind
                fit_result$rng_has_seed <-  postfit_rng$has_seed
                ## Keep a NULL state as a named list element so a caller that had no .Random.seed
                ## still receives a complete transport record.
                fit_result["rng_state"] <-  list(postfit_rng$state)
                fit_result$child_pid <-  Sys.getpid()
                validate_worker_transport(x = fit_result, path = "fresh-process result")
                fit_result
        }
        ##
        environment(worker) <-  baseenv()

        ##
        child_process <-  NULL
        on.exit({
                if (!is.null(x = child_process) && isTRUE(x = child_process$is_alive())) {
                        try(expr = child_process$kill_tree(), silent = TRUE)
                }
        }, add = TRUE)
        child_process <-  tryCatch({
                callr::r_bg(func = worker,
                            args = list(arguments = arguments,
                                        provider_package = provider_package,
                                        libpath = parent_libpath,
                                        working_dir = parent_working_dir,
                                        rng_snapshot = parent_rng,
                                        model_options = model_options),
                            libpath = parent_libpath,
                            wd = parent_working_dir,
                            stdout = "",
                            stderr = "",
                            system_profile = FALSE,
                            user_profile = FALSE,
                            supervise = TRUE,
                            error = "error")
        }, error = function(condition) {
                stop(paste0("Fresh-process sampler child could not be started: ",
                            conditionMessage(condition),
                            ". Set run_in_fresh_R_process = FALSE to diagnose or run this fit in the current R process."),
                     call. = FALSE)
        })
        tryCatch({
                child_process$wait()
        }, interrupt = function(condition) {
                if (isTRUE(x = child_process$is_alive())) try(expr = child_process$kill_tree(), silent = TRUE)
                stop("Fresh-process sampler was interrupted while the child process was running. Set run_in_fresh_R_process = FALSE to diagnose or run this fit in the current R process.",
                     call. = FALSE)
        })
        child_exit_status <-  child_process$get_exit_status()
        if (length(x = child_exit_status) != 1L || is.na(x = child_exit_status) || child_exit_status != 0L) {
                stop(paste0("Fresh-process sampler child did not exit cleanly (exit status ",
                            if (length(x = child_exit_status) == 1L && !is.na(x = child_exit_status)) child_exit_status else "unavailable",
                            "). Its returned fit and RNG state were rejected. Set run_in_fresh_R_process = FALSE to diagnose or run this fit in the current R process."),
                     call. = FALSE)
        }
        child_result <-  tryCatch({
                child_process$get_result()
        }, error = function(condition) {
                stop(paste0("Fresh-process sampler child terminated before returning a result: ",
                            conditionMessage(condition),
                            ". Set run_in_fresh_R_process = FALSE to diagnose or run this fit in the current R process."),
                     call. = FALSE)
        })

        ##
        if (!is.list(child_result) || is.null(child_result$rng_kind) ||
            is.null(child_result$rng_has_seed) || !"rng_state" %in% names(child_result)) {
                stop("Fresh-process sampler returned an incomplete child result. Set run_in_fresh_R_process = FALSE to diagnose or run this fit in the current R process.",
                     call. = FALSE)
        }
        child_rng <-  list(kind = child_result$rng_kind,
                          has_seed = child_result$rng_has_seed,
                          state = child_result$rng_state)
        validate_transport(x = child_rng, path = "fresh-process RNG result")
        restore_rng(snapshot = child_rng)
        child_rng_applied <-  TRUE

        ##
        if (!isTRUE(child_result$ok)) {
                child_error <-  child_result$error
                child_message <-  if (is.list(child_error) && !is.null(child_error$message)) child_error$message else "the child reported an unspecified error"
                child_class <-  if (is.list(child_error) && !is.null(child_error$class)) paste(child_error$class, collapse = "/") else "error"
                stop(paste0("Fresh-process sampler failed in the child [", child_class, "]: ", child_message),
                     call. = FALSE)
        }

        ##
        output <-  child_result$output
        validate_transport(x = output, path = "fresh-process output")
        if (!is.list(output) || is.null(output$init_object) || !is.list(output$init_object)) {
                stop("Fresh-process sampler returned no serialisable init_object. Set run_in_fresh_R_process = FALSE to diagnose or run this fit in the current R process.",
                     call. = FALSE)
        }
        if (is.null(output$init_object$model_so_file) || is.null(output$init_object$json_file_path)) {
                stop("Fresh-process sampler cannot rebuild the returned BridgeStan model because model_so_file or json_file_path is missing. Set run_in_fresh_R_process = FALSE to diagnose or run this fit in the current R process.",
                     call. = FALSE)
        }

        ##
        ## Recreate only bs_model. Draws, diagnostics, metadata and paths remain child results.
        output$init_object$bs_model <-  tryCatch({
                R_fn_create_BridgeStan_model( lib = output$init_object$model_so_file,
                                              data = output$init_object$json_file_path,
                                              seed = 123)
        }, error = function(condition) {
                stop(paste0("Fresh-process sampler completed, but the parent could not rebuild init_object$bs_model: ",
                            conditionMessage(condition),
                            ". Set run_in_fresh_R_process = FALSE to diagnose or run this fit in the current R process."),
                     call. = FALSE)
        })
        ## Keep the post-fit state exact even if BridgeStan construction changes R's RNG in future.
        restore_rng(snapshot = child_rng)
        output$execution_process <-  list(isolated = TRUE,
                                         child_pid = child_result$child_pid,
                                         parent_pid = Sys.getpid(),
                                         provider = provider_package)
        ##
        return(output)
}






















