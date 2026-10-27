#### =====================================================================================================================================
## R_fn_estimate_initial_M_bytes_per_chain.R
##
## ---- Initial M_bytes/chain: memory used by ONE full-data log-density-and-gradient evaluation of one chain, for any Stan model -------
##
## M_bytes/chain is the starting value of the memory that one gradient evaluation of one chain touches (for a Stan model: the
## autodiff tape, its stacks, the temporary containers and the parameter and gradient copies). It is measured, not modelled,
## so it applies to any Stan model compiled for BridgeStan:
##
##   for each repeat, in a FRESH child R process (callr), pinned to one CPU:
##     1. load the compiled model and the data with bridgestan;
##     2. evaluate the log density once WITHOUT the gradient (propto = FALSE, i.e. on doubles, with no autodiff), which loads
##        the code and data pages but leaves the autodiff arena and stacks untouched;
##     3. run R's gc() and return the free heap memory to the kernel (malloc_trim(0), through NicoStan's
##        Rcpp_fn_release_free_heap_memory(); glibc only), so that the gradient cannot reuse free pages that are already resident;
##     4. reset the process's peak resident memory (write "5" to /proc/self/clear_refs, so that VmHWM = VmRSS) and read the
##        resident memory (Rss of /proc/self/smaps_rollup);
##     5. run ONE log_density_gradient and read the resident memory again, and VmHWM and VmRSS of /proc/self/status;
##     6. return the resident growth plus the transient part, i.e. (Rss after - Rss before) + max(0, VmHWM - VmRSS after), in bytes.
##
## The evaluation point is the supplied initial values (init_list) or unconstrained point if given, and otherwise zero on the
## unconstrained scale. The tape of a Stan model depends on the point only through the branches of the model code; points far
## in the tails (e.g. random values on (-2, 2) for every unconstrained parameter) can take more expensive branches than the
## points the sampler visits, so they are used only when the log density is not finite at zero (or when requested).
##
## A fresh process is essential: Stan keeps its arena blocks and the capacity of its stacks after a gradient (recover_memory()
## rewinds them but does not free them), so a second measurement in the same process would reuse pages that are already
## resident and show no growth. The parent returns the median over the repeats, every repeat and the method.
##
## The result is a lower value of the memory touched: pages that are already resident before the reset and are reused by the
## gradient are not counted (the data rows, heap temporaries freed after step 2 that malloc_trim(0) cannot return because they
## share a page with live memory, and R's own free memory). The peak (VmHWM) and VmRSS come from the kernel's resident-memory
## counters, which on Linux 6.2 and later are updated in per-CPU batches of max(32, 2 x online CPUs) pages, so the transient part
## is known to within about one batch (128 KiB with up to 16 CPUs, 1.5 MiB with 192); pinning the child to one CPU keeps the error
## to one batch. The resident growth is read from /proc/self/smaps_rollup, which walks the page tables and is exact.
##
## Linux only (it reads /proc/self/status and /proc/self/smaps_rollup and writes /proc/self/clear_refs). If the kernel does not
## allow the peak to be reset, only the resident growth is used and the method records this; that misses temporary containers
## freed before the gradient returns, so it is a lower value still.
##




#### ---- child-process measurement --------------------------------------------------------------------------------------------------------

#' Measure the memory touched by one log_density_gradient in the current (fresh) R process
#'
#' Runs in a fresh child R process (its enclosing environment is baseenv(), so every non-base function is called with its
#' namespace). Pins the process to one CPU, loads the model and data with bridgestan, chooses the evaluation point, evaluates
#' the log density once without the gradient, releases the free heap memory, resets the peak resident memory, runs one
#' log_density_gradient and returns the resident growth plus the transient part of the peak.
#'
#' @keywords internal
#' @noRd
fn_measure_one_gradient_memory_in_this_process <-  function( model_so_file,
                                                             json_file_path,
                                                             unconstrained_parameter_vector,
                                                             init_list,
                                                             default_evaluation_point,
                                                             initial_values_seed,
                                                             initial_values_range_unconstrained,
                                                             max_attempts_for_finite_log_density,
                                                             propto,
                                                             jacobian,
                                                             free_heap_release_package_name,
                                                             free_heap_release_function_name,
                                                             n_nuisance_owned_by_NicoStan = 0) {

        ##
        ## ---- one field of /proc/self/status or /proc/self/smaps_rollup (reported in kB = 1024 bytes), in bytes:
        ##
        fn_read_proc_file_field_bytes <-  function( proc_file_path,
                                                    field_name) {

              proc_file_lines <-  readLines(con = proc_file_path, warn = FALSE)
              field_line <-  proc_file_lines[startsWith(x = proc_file_lines, prefix = paste0(field_name, ":"))]
              ##
              if (length(x = field_line) < 1) {
                    stop(paste0("field ", field_name, " not found in ", proc_file_path))
              }
              ##
              field_value_text <-  trimws(x = sub( pattern = paste0("^", field_name, ":"),
                                                   replacement = "",
                                                   x = field_line[1]))
              field_value_kB <-  as.numeric(strsplit(x = field_value_text, split = "[[:space:]]+")[[1]][1])
              ##
              return(field_value_kB * 1024)

        }
        ##
        ## ---- VmHWM and VmRSS from ONE read of /proc/self/status (so that both come from the same state of the counters):
        ##
        fn_read_VmHWM_and_VmRSS_bytes <-  function() {

              status_lines <-  readLines(con = "/proc/self/status", warn = FALSE)
              ##
              fn_field_bytes <-  function(field_name) {

                    field_line <-  status_lines[startsWith(x = status_lines, prefix = paste0(field_name, ":"))]
                    if (length(x = field_line) != 1) stop(paste0( "field ",
                                                                  field_name,
                                                                  " not found in /proc/self/status"))
                    field_value_text <-  trimws(x = sub( pattern = paste0("^", field_name, ":"),
                                                         replacement = "",
                                                         x = field_line))
                    return(1024 * as.numeric(strsplit(x = field_value_text, split = "[[:space:]]+")[[1]][1]))
              }

              ##
              return(c( VmHWM = fn_field_bytes(field_name = "VmHWM"),
                        VmRSS = fn_field_bytes(field_name = "VmRSS")))

        }
        ##
        ## ---- exact resident memory (page-table walk of /proc/self/smaps_rollup; VmRSS if smaps_rollup is not available):
        ##
        smaps_rollup_available <-  file.exists("/proc/self/smaps_rollup")
        ##
        fn_read_exact_resident_memory_bytes <-  function() {

              if (smaps_rollup_available) {
                    return(fn_read_proc_file_field_bytes( proc_file_path = "/proc/self/smaps_rollup",
                                                          field_name = "Rss"))
              }
              return(fn_read_proc_file_field_bytes(proc_file_path = "/proc/self/status", field_name = "VmRSS"))
        }

        ##
        ## ---- pin this process to the CPU it is running on (the kernel's per-CPU counter batches then add at most one batch of
        ##      error to each reading); /proc/self/stat field 39 is the CPU last run on (0-based), mcaffinity() is 1-based:
        ##
        CPU_affinity_before_pinning <-  tryCatch( expr  = parallel::mcaffinity(),
                                                  error = function(condition) NULL)
        pinned_CPU_index <-  NA_real_
        ##
        if (length(x = CPU_affinity_before_pinning) > 1) {

              process_stat_line <-  readLines(con = "/proc/self/stat", warn = FALSE)[1]
              process_stat_fields <-  strsplit(x = sub(pattern = "^.*\\) ", replacement = "", x = process_stat_line), split = " ", fixed = TRUE)[[1]]
              current_CPU_index <-  suppressWarnings(as.numeric(process_stat_fields[37])) + 1
              ##
              pinned_CPU_index <-  if (isTRUE(current_CPU_index %in% CPU_affinity_before_pinning)) {
                    current_CPU_index
              } else {
                    CPU_affinity_before_pinning[1]
              }
              ##
              pinned_CPU_affinity <-  tryCatch( expr  = parallel::mcaffinity(affinity = pinned_CPU_index),
                                                error = function(condition) NULL)
              if (!identical( x = as.numeric(pinned_CPU_affinity),
                              y = as.numeric(pinned_CPU_index))) pinned_CPU_index <-  NA_real_

        } else if (length(x = CPU_affinity_before_pinning) == 1) {

              pinned_CPU_index <-  CPU_affinity_before_pinning

        }
        ##
        ## ---- the package that provides the free heap release (NicoStan) is loaded first, as in a real run (this also loads
        ##      RcppParallel's TBB library before the model's):
        ##
        free_heap_release_function <-  NULL
        free_heap_release_note <-  "free heap memory not released (NicoStan's Rcpp_fn_release_free_heap_memory() is not available)"
        ##
        if (!is.null(x = free_heap_release_package_name)) {

              free_heap_release_function <-  tryCatch( expr  = get(x        = free_heap_release_function_name,
                                                                   envir    = loadNamespace(package = free_heap_release_package_name),
                                                                   inherits = FALSE),
                                                       error = function(condition) {

                                                             load_error_message <-  substr(x = conditionMessage(condition), start = 1, stop = 300)
                                                             free_heap_release_note <<-  paste0("free heap memory not released (",
                                                                                                free_heap_release_package_name,
                                                                                                " could not be loaded: ", load_error_message, ")")
                                                             NULL
                                                       })


        }
        ##
        ## ---- load the compiled model and the data:
        ##
        bridgestan_model <-  bridgestan::StanModel$new( lib  = model_so_file,
                                                        data = json_file_path,
                                                        seed = initial_values_seed)
        ##
        ## ---- a library in NicoStan's nuisance-block form (n_nuisance_owned_by_NicoStan > 0): its nuisance
        ##      values are not BridgeStan parameters, all 0 here; the log densities without the gradient (the
        ##      search of a finite point) use the values set for evaluations outside NicoStan's gradient call,
        ##      and the ONE measured gradient goes through NicoStan's own gradient call, with the nuisance block
        ##      held, as NicoStan's sampler runs it: one node per nuisance value on the main tape, where the
        ##      values used outside NicoStan make a node at every read (a model that reads a value more than
        ##      once on the main tape, e.g. the Cox packing, would be measured too large)
        ##
        if (n_nuisance_owned_by_NicoStan > 0) {
              ## (called through do.call(): the routine is in the model library, not in NicoStan's own, and a
              ##  .C() call written out would be registered with NicoStan's own routines when it is built)
              do.call(".C", list( "nicostan_set_the_nuisance_values_used_outside_NicoStan_R",
                                  as.double(rep(0, n_nuisance_owned_by_NicoStan)),
                                  as.integer(n_nuisance_owned_by_NicoStan),
                                  PACKAGE = tools::file_path_sans_ext(basename(model_so_file))))
        }
        nuisance_values_of_the_gradient <-  as.double(rep(0, n_nuisance_owned_by_NicoStan))
        nuisance_gradient_of_the_gradient <-  double(n_nuisance_owned_by_NicoStan)
        ##
        ## ---- one log density and gradient, through NicoStan's own gradient call for a nuisance-block library
        ##      (the .C copies of its two nuisance-sized vectors, 16 bytes per nuisance value, are in the measurement)
        ##
        fn_log_density_gradient_of_the_measurement <-  function(theta_unconstrained) {

              if (n_nuisance_owned_by_NicoStan == 0) {

                    return(bridgestan_model$log_density_gradient( theta_unc = theta_unconstrained,
                                                                  propto    = propto,
                                                                  jacobian  = jacobian))

              }
              private_fields <-  bridgestan_model$.__enclos_env__$private
              out <-  do.call(".C", list( "nicostan_bs_log_density_gradient_with_the_nuisance_block_R",
                                          as.raw(private_fields$model),
                                          as.logical(propto),
                                          as.logical(jacobian),
                                          as.double(theta_unconstrained),
                                          nuisance_values_of_the_gradient,
                                          nuisance_gradient = nuisance_gradient_of_the_gradient,
                                          as.integer(n_nuisance_owned_by_NicoStan),
                                          val = double(1),
                                          gradient = double(length(theta_unconstrained)),
                                          return_code = as.integer(0),
                                          err_msg = as.character(""),
                                          err_ptr = raw(8),
                                          NAOK = TRUE,
                                          PACKAGE = private_fields$lib_name))
              if (out$return_code != 0) {

                    stop(paste0("fn_estimate_initial_M_bytes_per_chain: NicoStan's gradient call with the nuisance ",
                                "block failed: ", out$err_msg))

              }
              return(list(val = out$val, gradient = out$gradient))

        }
        ##
        n_unconstrained_parameters <-  bridgestan_model$param_unc_num()
        ##
        ## ---- evaluation point, and the log density WITHOUT the gradient (propto = FALSE: doubles, no autodiff tape):
        ##
        fn_log_density_without_gradient <-  function(theta_unconstrained) {

              tryCatch( expr  = bridgestan_model$log_density( theta_unc = theta_unconstrained,
                                                              propto    = FALSE,
                                                              jacobian  = jacobian),
                        error = function(condition) NA_real_)
        }

        ##
        init_list_names_not_in_model <-  character(0)
        init_list_parameters_completed_from_zero <-  character(0)
        ##
        if (!is.null(x = unconstrained_parameter_vector)) {

              ## (1) a supplied point on the unconstrained scale:
              if (length(x = unconstrained_parameter_vector) != n_unconstrained_parameters) {
                    stop(paste0("unconstrained_parameter_vector has length ", length(x = unconstrained_parameter_vector),
                                " but the model has ", n_unconstrained_parameters, " unconstrained parameters"))
              }
              ##
              theta_unconstrained <-  as.numeric(unconstrained_parameter_vector)
              evaluation_point_description <-  "supplied unconstrained_parameter_vector"
              log_density_without_gradient <-  fn_log_density_without_gradient(theta_unconstrained = theta_unconstrained)

        } else if (!is.null(x = init_list)) {

              ## (2) supplied initial values on the constrained scale; parameters not given are completed with the model's own
              ##     defaults (the constrained transform of zero on the unconstrained scale), as R_fn_init_initial_values() does.
              ##     BridgeStan names the entries "name.i.j" in column-major order, which is R's order for as.numeric() of an
              ##     array, so every parameter is matched by name and filled with as.numeric() of its value:
              parameter_names <-  bridgestan_model$param_names(include_tp = FALSE, include_gq = FALSE)
              parameter_base_names <-  sub(pattern = "\\..*$", replacement = "", x = parameter_names)
              ##
              theta_constrained <-  bridgestan_model$param_constrain( theta_unc  = rep(0, times = n_unconstrained_parameters),
                                                                      include_tp = FALSE,
                                                                      include_gq = FALSE,
                                                                      rng        = bridgestan_model$new_rng(seed = initial_values_seed))
              ##
              for (init_list_name in names(x = init_list)) {

                    matching_indices <-  which(parameter_base_names == init_list_name)
                    ##
                    if (length(x = matching_indices) == 0) {
                          init_list_names_not_in_model <-  c(init_list_names_not_in_model, init_list_name)
                          next
                    }
                    ##
                    init_list_values <-  init_list[[init_list_name]]
                    ##
                    if (is.list(x = init_list_values) || (!is.numeric(init_list_values) && !is.logical(init_list_values))) {
                          stop(paste0( "init_list$",
                                       init_list_name,
                                       " must be a number, vector, matrix or array (not a list)"))
                    }
                    ##
                    init_list_values <-  as.numeric(init_list_values)
                    ##
                    if (length(x = init_list_values) != length(x = matching_indices)) {
                          stop(paste0("init_list$", init_list_name, " has ", length(x = init_list_values), " value(s) but the model's parameter ",
                                      init_list_name, " has ", length(x = matching_indices)))
                    }
                    ##
                    theta_constrained[matching_indices] <-  init_list_values

              }
              ##
              init_list_parameters_completed_from_zero <-  setdiff( x = unique(x = parameter_base_names),
                                                                    y = names(x = init_list))
              ##
              theta_unconstrained <-  tryCatch( expr  = bridgestan_model$param_unconstrain(theta = theta_constrained),
                                                error = function(condition) {

                                                      stop(paste0("init_list could not be unconstrained by the model: ",
                                                                  substr( x = conditionMessage(condition),
                                                                          start = 1,
                                                                          stop = 500)))
                                                })

              ##
              evaluation_point_description <-  if (length(x = init_list_parameters_completed_from_zero) == 0) {
                    "supplied init_list"
              } else {
                    paste0("supplied init_list (", paste( init_list_parameters_completed_from_zero,
                                                          collapse = ", "),
                           " completed with the model's values at zero on the unconstrained scale)")
              }
              log_density_without_gradient <-  fn_log_density_without_gradient(theta_unconstrained = theta_unconstrained)

        } else {

              ## (3) the default point: zero on the unconstrained scale, or random; random points are also tried when the
              ##     log density is not finite at zero:
              log_density_without_gradient <-  NA_real_
              evaluation_point_description <-  ""
              ##
              if (default_evaluation_point == "zero") {

                    theta_unconstrained <-  rep(0, times = n_unconstrained_parameters)
                    evaluation_point_description <-  "zero on the unconstrained scale"
                    log_density_without_gradient <-  fn_log_density_without_gradient(theta_unconstrained = theta_unconstrained)
                    ##
                    if (!is.finite(x = log_density_without_gradient)) {
                          evaluation_point_description <-  "the log density was not finite at zero, so "
                    }

              }
              ##
              ##  NicoStan's own fixed small values, the starting values of its layout timings, and otherwise
              ##  the stop below, which asks for the model's fixed starting values)
#             if (!is.finite(x = log_density_without_gradient)) {

#                   for (attempt_index in seq_len(length.out = max_attempts_for_finite_log_density)) {

#                         set.seed(seed = initial_values_seed + attempt_index - 1)
#                         ##
#                         theta_unconstrained <-  stats::runif( n   = n_unconstrained_parameters,
#                                                               min = initial_values_range_unconstrained[1],
#                                                               max = initial_values_range_unconstrained[2])
#                         ##
#                         log_density_without_gradient <-  fn_log_density_without_gradient(theta_unconstrained = theta_unconstrained)
#                         ##
#                         if (is.finite(x = log_density_without_gradient)) break

#                   }
#                   ##
#                   evaluation_point_description <-  paste0(evaluation_point_description,
#                                                           "uniform(", initial_values_range_unconstrained[1], ", ",
#                                                           initial_values_range_unconstrained[2], ") on the unconstrained scale, seed(s) ",
#                                                           initial_values_seed, " to ", initial_values_seed + attempt_index - 1)

#             }
              if (!is.finite(x = log_density_without_gradient)) {
                    theta_unconstrained <-  0.005 * (((seq_len(n_unconstrained_parameters) * 7) %% 5) - 2)
                    log_density_without_gradient <-  fn_log_density_without_gradient(
                                                           theta_unconstrained = theta_unconstrained)
                    evaluation_point_description <-  paste0(
                          evaluation_point_description,
                          "NicoStan's fixed small values on the unconstrained scale")
              }

        }
        ##
        if (!is.finite(x = log_density_without_gradient)) {
              stop(paste0("the log density is not finite at the evaluation point (", evaluation_point_description,
                          "); supply an init_list or an unconstrained_parameter_vector at which it is finite"))
        }
        ##
        ## ---- settle R's own allocations and return the free heap memory to the kernel:
        ##
        invisible(x = gc(verbose = FALSE, full = TRUE))
        ##
        free_heap_release_result <-  NA_real_
        ##
        if (!is.null(x = free_heap_release_function)) {

              free_heap_release_result <-  tryCatch( expr  = as.numeric(free_heap_release_function()),
                                                     error = function(condition) NA_real_)
              ##
              free_heap_release_note <-  if (identical(x = free_heap_release_result, y = -1)) {

                    "free heap memory not released (malloc_trim is not available: not glibc)"
              } else if (is.na(x = free_heap_release_result)) {

                    "free heap memory not released (the release function failed)"
              } else {
                    "free heap memory released with malloc_trim(0)"
              }

        }
        ##
        ## ---- reset the peak resident memory to the current value. The reset worked if the write raised no error and either the
        ##      peak fell, or it is now within a few batches of the kernel's per-CPU resident-memory counters (max(32, 2 x online
        ##      CPUs) pages each) of VmRSS:
        ##
        page_size_bytes <-  tryCatch( expr  = fn_read_proc_file_field_bytes( proc_file_path = "/proc/self/smaps",
                                                                             field_name = "KernelPageSize"),
                                      error = function(condition) 4096)
        N_online_CPUs <-  parallel::detectCores(logical = TRUE)
        if (!is.finite(x = N_online_CPUs)) N_online_CPUs <-  1
        ##
        resident_memory_counter_batch_bytes <-  max(32, 2 * N_online_CPUs) * page_size_bytes
        peak_reset_tolerance_bytes <-  max(1024^2, 4 * resident_memory_counter_batch_bytes)
        ##
        VmHWM_before_reset_bytes <-  fn_read_VmHWM_and_VmRSS_bytes()[["VmHWM"]]
        ##
        peak_reset_write_succeeded <-  tryCatch( expr    = { cat("5", file = "/proc/self/clear_refs"); TRUE },

                                                 warning = function(condition) FALSE,
                                                 error   = function(condition) FALSE)
        ##
        VmHWM_and_VmRSS_before_gradient <-  fn_read_VmHWM_and_VmRSS_bytes()
        VmHWM_before_gradient_bytes <-  VmHWM_and_VmRSS_before_gradient[["VmHWM"]]
        VmRSS_before_gradient_bytes <-  VmHWM_and_VmRSS_before_gradient[["VmRSS"]]
        ##
        peak_reset_succeeded <-  peak_reset_write_succeeded &&
                                 ((VmHWM_before_gradient_bytes < VmHWM_before_reset_bytes) ||
                                  (VmHWM_before_gradient_bytes - VmRSS_before_gradient_bytes <= peak_reset_tolerance_bytes))
        ##
        resident_memory_before_gradient_bytes <-  fn_read_exact_resident_memory_bytes()
        ##
        ## ---- ONE log density and gradient:
        ##
        log_density_and_gradient <-  fn_log_density_gradient_of_the_measurement(theta_unconstrained)
        ##
        VmHWM_and_VmRSS_after_gradient <-  fn_read_VmHWM_and_VmRSS_bytes()
        VmHWM_after_gradient_bytes <-  VmHWM_and_VmRSS_after_gradient[["VmHWM"]]
        VmRSS_after_gradient_bytes <-  VmHWM_and_VmRSS_after_gradient[["VmRSS"]]
        ##
        resident_memory_after_gradient_bytes <-  fn_read_exact_resident_memory_bytes()
        ##
        ## ---- resident growth (exact) plus the transient part (memory touched and freed again before the gradient returned):
        ##
        resident_memory_growth_bytes <-  resident_memory_after_gradient_bytes - resident_memory_before_gradient_bytes
        transient_peak_bytes <-  max(0, VmHWM_after_gradient_bytes - VmRSS_after_gradient_bytes)
        ##
        resident_memory_source <-  if (smaps_rollup_available) "Rss of /proc/self/smaps_rollup" else "VmRSS of /proc/self/status"
        ##
        if (peak_reset_succeeded) {

              gradient_memory_bytes <-  resident_memory_growth_bytes + transient_peak_bytes
              measurement_method_code <-  "peak"
              measurement_method <-  paste0("resident growth (", resident_memory_source, ") + transient peak (VmHWM - VmRSS after the gradient), ",
                                            "peak reset with /proc/self/clear_refs")
        } else {

              gradient_memory_bytes <-  resident_memory_growth_bytes
              measurement_method_code <-  "resident_growth_only"
              measurement_method <-  paste0("resident growth only (", resident_memory_source, "); the peak could not be reset, ",
                                            "so temporaries freed before the gradient returns are missed")
        }
        ##
        return(list( gradient_memory_bytes                      = gradient_memory_bytes,
                     measurement_method_code                    = measurement_method_code,
                     measurement_method                         = measurement_method,
                     resident_memory_growth_bytes               = resident_memory_growth_bytes,
                     transient_peak_bytes                       = transient_peak_bytes,
                     resident_memory_before_gradient_bytes      = resident_memory_before_gradient_bytes,
                     resident_memory_after_gradient_bytes       = resident_memory_after_gradient_bytes,
                     VmHWM_before_reset_bytes                   = VmHWM_before_reset_bytes,
                     VmHWM_before_gradient_bytes                = VmHWM_before_gradient_bytes,
                     VmRSS_before_gradient_bytes                = VmRSS_before_gradient_bytes,
                     VmHWM_after_gradient_bytes                 = VmHWM_after_gradient_bytes,
                     VmRSS_after_gradient_bytes                 = VmRSS_after_gradient_bytes,
                     peak_reset_write_succeeded                 = peak_reset_write_succeeded,
                     peak_reset_tolerance_bytes                 = peak_reset_tolerance_bytes,
                     resident_memory_counter_batch_bytes        = resident_memory_counter_batch_bytes,
                     free_heap_release_result                   = free_heap_release_result,
                     free_heap_release_note                     = free_heap_release_note,
                     pinned_CPU_index                           = pinned_CPU_index,
                     n_unconstrained_parameters                 = n_unconstrained_parameters,
                     evaluation_point_description               = evaluation_point_description,
                     init_list_names_not_in_model               = init_list_names_not_in_model,
                     log_density_without_gradient               = log_density_without_gradient,
                     log_density_with_gradient                  = log_density_and_gradient$val,
                     gradient_norm                              = sqrt(sum(log_density_and_gradient$gradient^2)),
                     child_process_id                           = Sys.getpid()))

}




#### ---- child-process measurement for a built-in model (manual gradients) ----------------------------------

#' Measure the memory touched by one log density and manual gradient of a built-in NicoStan model in the
#' current (fresh) R process
#'
#' Runs in a fresh child R process (its enclosing environment is baseenv(), so every non-base function is
#' called with its namespace). Pins the process to one CPU, evaluates the log density and gradient of the
#' built-in model once with Rcpp_wrapper_fn_lp_grad() of the package that provides the built-in models' C++
#' code (gradient_package_name, i.e. BayesMVP: NicoStan's own compiled code evaluates Stan models only). This
#' loads the code and data pages; the model's workspace is allocated inside the call and freed when it
#' returns, so none of it stays resident. It then releases the free heap memory, resets the peak resident
#' memory, evaluates the log density and gradient once more and returns the resident growth plus the transient
#' part of the peak.
#'
#' @keywords internal
#' @noRd
fn_measure_one_gradient_memory_of_built_in_model_in_this_process <-  function( Model_type,
                                                                               theta_main_vec,
                                                                               theta_us_vec,
                                                                               y,
                                                                               Model_args_as_Rcpp_List,
                                                                               gradient_package_name,
                                                                               free_heap_release_package_name,
                                                                               free_heap_release_function_name) {

        ##
        ## ---- one field of /proc/self/status or /proc/self/smaps_rollup (reported in kB = 1024 bytes), in
        ##      bytes:
        ##
        fn_read_proc_file_field_bytes <-  function( proc_file_path,
                                                    field_name) {

              proc_file_lines <-  readLines(con = proc_file_path, warn = FALSE)
              field_line <-  proc_file_lines[startsWith(x = proc_file_lines, prefix = paste0(field_name, ":"))]
              ##
              if (length(x = field_line) < 1) {
                    stop(paste0("field ", field_name, " not found in ", proc_file_path))
              }
              ##
              field_value_text <-  trimws(x = sub(pattern = paste0("^", field_name, ":"), replacement = "",
                                                  x = field_line[1]))
              field_value_kB <-  as.numeric(strsplit(x = field_value_text, split = "[[:space:]]+")[[1]][1])
              ##
              return(field_value_kB * 1024)

        }
        ##
        ## ---- VmHWM and VmRSS from ONE read of /proc/self/status (so that both come from the same state of the
        ##      counters):
        ##
        fn_read_VmHWM_and_VmRSS_bytes <-  function() {

              status_lines <-  readLines(con = "/proc/self/status", warn = FALSE)
              ##
              fn_field_bytes <-  function(field_name) {

                    field_line <-  status_lines[startsWith(x = status_lines, prefix = paste0(field_name, ":"))]
                    if (length(x = field_line) != 1) {
                          stop(paste0("field ", field_name, " not found in /proc/self/status"))
                    }
                    field_value_text <-  trimws(x = sub(pattern = paste0("^", field_name, ":"), replacement = "",
                                                        x = field_line))
                    return(1024 * as.numeric(strsplit(x = field_value_text, split = "[[:space:]]+")[[1]][1]))
              }

              ##
              return(c( VmHWM = fn_field_bytes(field_name = "VmHWM"),
                        VmRSS = fn_field_bytes(field_name = "VmRSS")))

        }
        ##
        ## ---- exact resident memory (page-table walk of /proc/self/smaps_rollup; VmRSS if smaps_rollup is not
        ##      available):
        ##
        smaps_rollup_available <-  file.exists("/proc/self/smaps_rollup")
        ##
        fn_read_exact_resident_memory_bytes <-  function() {

              if (smaps_rollup_available) {
                    return(fn_read_proc_file_field_bytes( proc_file_path = "/proc/self/smaps_rollup",
                                                          field_name     = "Rss"))
              }
              return(fn_read_proc_file_field_bytes( proc_file_path = "/proc/self/status",
                                                    field_name     = "VmRSS"))
        }

        ##
        ## ---- pin this process to the CPU it is running on (the kernel's per-CPU counter batches then add at
        ##      most one batch of error to each reading); /proc/self/stat field 39 is the CPU last run on
        ##      (0-based), mcaffinity() is 1-based:
        ##
        CPU_affinity_before_pinning <-  tryCatch( expr  = parallel::mcaffinity(),
                                                  error = function(condition) NULL)
        pinned_CPU_index <-  NA_real_
        ##
        if (length(x = CPU_affinity_before_pinning) > 1) {

              process_stat_line <-  readLines(con = "/proc/self/stat", warn = FALSE)[1]
              process_stat_line_after_name <-  sub(pattern = "^.*\\) ", replacement = "", x = process_stat_line)
              process_stat_fields <-  strsplit(x = process_stat_line_after_name, split = " ", fixed = TRUE)[[1]]
              current_CPU_index <-  suppressWarnings(as.numeric(process_stat_fields[37])) + 1
              ##
              pinned_CPU_index <-  if (isTRUE(current_CPU_index %in% CPU_affinity_before_pinning)) {
                    current_CPU_index
              } else {
                    CPU_affinity_before_pinning[1]
              }
              ##
              pinned_CPU_affinity <-  tryCatch( expr  = parallel::mcaffinity(affinity = pinned_CPU_index),
                                                error = function(condition) NULL)
              if (!identical(x = as.numeric(pinned_CPU_affinity), y = as.numeric(pinned_CPU_index))) {

                    pinned_CPU_index <-  NA_real_
              }


        } else if (length(x = CPU_affinity_before_pinning) == 1) {

              pinned_CPU_index <-  CPU_affinity_before_pinning

        }
        ##
        ## ---- the package that provides the free heap release (NicoStan) is loaded first, as in a real run:
        ##
        free_heap_release_function <-  NULL
        free_heap_release_note <-  paste0("free heap memory not released (NicoStan's ",
                                          "Rcpp_fn_release_free_heap_memory() is not available)")
        ##
        if (!is.null(x = free_heap_release_package_name)) {

              free_heap_release_function <-  tryCatch(
                    expr  = get(x        = free_heap_release_function_name,
                                envir    = loadNamespace(package = free_heap_release_package_name),
                                inherits = FALSE),
                    error = function(condition) {

                          load_error_message <-  substr(x = conditionMessage(condition), start = 1, stop = 300)
                          free_heap_release_note <<-  paste0("free heap memory not released (",
                                                             free_heap_release_package_name,
                                                             " could not be loaded: ", load_error_message, ")")
                          NULL
                    })


        }
        ##
        ## ---- the built-in log density and manual gradient (the standard evaluation, not the log-scale one):
        ##
        lp_grad_function <-  get(x        = "Rcpp_wrapper_fn_lp_grad",
                                 envir    = loadNamespace(package = gradient_package_name),
                                 inherits = FALSE)
        ##
        fn_log_density_and_gradient <-  function() {

              lp_grad_function( Model_type               = Model_type,
                                force_autodiff           = FALSE,
                                force_PartialLog         = FALSE,
                                multi_attempts           = FALSE,
                                theta_main_vec           = theta_main_vec,
                                theta_us_vec             = theta_us_vec,
                                y                        = y,
                                grad_option              = "all",
                                Model_args_as_Rcpp_List  = Model_args_as_Rcpp_List)
        }

        ##
        ## ---- the first evaluation (loads the code and data pages; its workspace is freed when it returns):
        ##
        first_log_density_and_gradient <-  fn_log_density_and_gradient()
        log_density_first_evaluation <-  first_log_density_and_gradient[1]
        first_evaluation_all_finite <-  all(is.finite(x = first_log_density_and_gradient))
        rm(first_log_density_and_gradient)
        ##
        if (!is.finite(x = log_density_first_evaluation)) {
              stop(paste0("the log density of the built-in model is not finite at the evaluation point ",
                          "(zero on the unconstrained scale)"))
        }
        ##
        ## ---- settle R's own allocations and return the free heap memory to the kernel:
        ##
        invisible(x = gc(verbose = FALSE, full = TRUE))
        ##
        free_heap_release_result <-  NA_real_
        ##
        if (!is.null(x = free_heap_release_function)) {

              free_heap_release_result <-  tryCatch( expr  = as.numeric(free_heap_release_function()),
                                                     error = function(condition) NA_real_)
              ##
              free_heap_release_note <-  if (identical(x = free_heap_release_result, y = -1)) {

                    "free heap memory not released (malloc_trim is not available: not glibc)"
              } else if (is.na(x = free_heap_release_result)) {

                    "free heap memory not released (the release function failed)"
              } else {
                    "free heap memory released with malloc_trim(0)"
              }

        }
        ##
        ## ---- reset the peak resident memory to the current value. The reset worked if the write raised no error
        ##      and either the peak fell, or it is now within a few batches of the kernel's per-CPU
        ##      resident-memory counters (max(32, 2 x online CPUs) pages each) of VmRSS:
        ##
        page_size_bytes <-  tryCatch(
              expr  = fn_read_proc_file_field_bytes( proc_file_path = "/proc/self/smaps",
                                                     field_name     = "KernelPageSize"),
              error = function(condition) 4096)
        N_online_CPUs <-  parallel::detectCores(logical = TRUE)
        if (!is.finite(x = N_online_CPUs)) N_online_CPUs <-  1
        ##
        resident_memory_counter_batch_bytes <-  max(32, 2 * N_online_CPUs) * page_size_bytes
        peak_reset_tolerance_bytes <-  max(1024^2, 4 * resident_memory_counter_batch_bytes)
        ##
        VmHWM_before_reset_bytes <-  fn_read_VmHWM_and_VmRSS_bytes()[["VmHWM"]]
        ##
        peak_reset_write_succeeded <-  tryCatch( expr    = { cat("5", file = "/proc/self/clear_refs"); TRUE },

                                                 warning = function(condition) FALSE,
                                                 error   = function(condition) FALSE)
        ##
        VmHWM_and_VmRSS_before_gradient <-  fn_read_VmHWM_and_VmRSS_bytes()
        VmHWM_before_gradient_bytes <-  VmHWM_and_VmRSS_before_gradient[["VmHWM"]]
        VmRSS_before_gradient_bytes <-  VmHWM_and_VmRSS_before_gradient[["VmRSS"]]
        ##
        peak_fell <-  VmHWM_before_gradient_bytes < VmHWM_before_reset_bytes
        peak_within_tolerance_of_VmRSS <-  (VmHWM_before_gradient_bytes - VmRSS_before_gradient_bytes) <=
                                           peak_reset_tolerance_bytes
        peak_reset_succeeded <-  peak_reset_write_succeeded && (peak_fell || peak_within_tolerance_of_VmRSS)
        ##
        resident_memory_before_gradient_bytes <-  fn_read_exact_resident_memory_bytes()
        ##
        ## ---- the measured log density and gradient:
        ##
        log_density_and_gradient <-  fn_log_density_and_gradient()
        ##
        VmHWM_and_VmRSS_after_gradient <-  fn_read_VmHWM_and_VmRSS_bytes()
        VmHWM_after_gradient_bytes <-  VmHWM_and_VmRSS_after_gradient[["VmHWM"]]
        VmRSS_after_gradient_bytes <-  VmHWM_and_VmRSS_after_gradient[["VmRSS"]]
        ##
        resident_memory_after_gradient_bytes <-  fn_read_exact_resident_memory_bytes()
        ##
        ## ---- resident growth (exact) plus the transient part (memory touched and freed again before the
        ##      gradient returned):
        ##
        resident_memory_growth_bytes <-  resident_memory_after_gradient_bytes -
                                         resident_memory_before_gradient_bytes
        transient_peak_bytes <-  max(0, VmHWM_after_gradient_bytes - VmRSS_after_gradient_bytes)
        ##
        resident_memory_source <-  if (smaps_rollup_available) {

              "Rss of /proc/self/smaps_rollup"
        } else {

              "VmRSS of /proc/self/status"
        }
        ##
        if (peak_reset_succeeded) {
              gradient_memory_bytes <-  resident_memory_growth_bytes + transient_peak_bytes
              measurement_method_code <-  "peak"
              measurement_method <-  paste0("resident growth (", resident_memory_source, ") + transient peak ",
                                            "(VmHWM - VmRSS after the gradient), peak reset with ",
                                            "/proc/self/clear_refs")
        } else {
              gradient_memory_bytes <-  resident_memory_growth_bytes
              measurement_method_code <-  "resident_growth_only"
              measurement_method <-  paste0("resident growth only (", resident_memory_source, "); the peak ",
                                            "could not be reset, so temporaries freed before the gradient ",
                                            "returns are missed")
        }
        ##
        return(list( gradient_memory_bytes                      = gradient_memory_bytes,
                     measurement_method_code                    = measurement_method_code,
                     measurement_method                         = measurement_method,
                     resident_memory_growth_bytes               = resident_memory_growth_bytes,
                     transient_peak_bytes                       = transient_peak_bytes,
                     resident_memory_before_gradient_bytes      = resident_memory_before_gradient_bytes,
                     resident_memory_after_gradient_bytes       = resident_memory_after_gradient_bytes,
                     VmHWM_before_reset_bytes                   = VmHWM_before_reset_bytes,
                     VmHWM_before_gradient_bytes                = VmHWM_before_gradient_bytes,
                     VmRSS_before_gradient_bytes                = VmRSS_before_gradient_bytes,
                     VmHWM_after_gradient_bytes                 = VmHWM_after_gradient_bytes,
                     VmRSS_after_gradient_bytes                 = VmRSS_after_gradient_bytes,
                     peak_reset_write_succeeded                 = peak_reset_write_succeeded,
                     peak_reset_tolerance_bytes                 = peak_reset_tolerance_bytes,
                     resident_memory_counter_batch_bytes        = resident_memory_counter_batch_bytes,
                     free_heap_release_result                   = free_heap_release_result,
                     free_heap_release_note                     = free_heap_release_note,
                     pinned_CPU_index                           = pinned_CPU_index,
                     log_density_first_evaluation               = log_density_first_evaluation,
                     first_evaluation_all_finite                = first_evaluation_all_finite,
                     log_density_with_gradient                  = log_density_and_gradient[1],
                     all_outputs_finite                         = all(is.finite(log_density_and_gradient)),
                     child_process_id                           = Sys.getpid()))

}




#### ---- free heap release: where the child process finds it ----------------------------------------------------------------------------

#' Locate NicoStan's free heap release function for the child process
#'
#' Returns the package and function name of NicoStan's Rcpp_fn_release_free_heap_memory() (malloc_trim(0)) when the installed
#' NicoStan provides it, and NULL otherwise (the measurement then runs without releasing free heap memory, and says so).
#'
#' @keywords internal
#' @noRd
fn_locate_free_heap_release_function <-  function() {

        NicoStan_provides_release <-  requireNamespace(package = "NicoStan", quietly = TRUE) &&
                                      exists( x = "Rcpp_fn_release_free_heap_memory",
                                              envir = asNamespace(ns = "NicoStan"),
                                              inherits = FALSE)
        ##
        if (!NicoStan_provides_release) {
              return(NULL)
        }
        ##
        return(list( package_name  = "NicoStan",
                     function_name = "Rcpp_fn_release_free_heap_memory"))

}




#### ---- exported function ----------------------------------------------------------------------------------------------------------------

#' Estimate the initial M_bytes/chain: memory used by one full-data log-density-and-gradient evaluation of one chain
#'
#' Measures, for any Stan model compiled for BridgeStan, the memory (in bytes) that ONE evaluation of the log density and its
#' gradient on the full data uses in one chain. This is the starting value of M_bytes/chain, from which
#' [fn_compute_initial_N_chunks()] chooses the starting number of chunks.
#'
#' For each repeat, a fresh child R process, pinned to one CPU, loads the model and the data with bridgestan, evaluates the log
#' density once without the gradient (this loads the code and data pages but does not use the autodiff tape), runs gc() and
#' returns the free heap memory to the kernel (malloc_trim(0)), resets its peak resident memory (by writing "5" to
#' /proc/self/clear_refs), reads its resident memory (Rss of /proc/self/smaps_rollup), runs one log_density_gradient, and
#' reads the resident memory again and the peak (VmHWM) and VmRSS of /proc/self/status. The result of the repeat is the growth
#' of the resident memory plus the transient part max(0, VmHWM - VmRSS) after the gradient. The value returned is the median
#' over the repeats. Each repeat needs a new process because Stan keeps its arena blocks and stack capacity after a gradient,
#' so a second gradient in the same process reuses memory that is already resident and shows no growth.
#'
#' Limits of the measurement:
#' \itemize{
#'   \item It is a lower value of the memory touched by the evaluation: pages that are already resident before the reset and
#'     are reused by the gradient are not counted (the data rows, heap temporaries freed after the log density without the
#'     gradient that malloc_trim(0) cannot return, and R's own free memory). For the LC_MVP_bin_PartialLog_v5 reduce_sum_static
#'     model (6 binary tests), the slope over 1,000 to 5,000 individuals was 4 to 5% below the working set of its partial sum
#'     counted directly from Stan's arena, stacks and heap. This is a starting value; the check during burn-in corrects it.
#'   \item Without NicoStan's Rcpp_fn_release_free_heap_memory() (or on a C library other than glibc), free heap memory that is
#'     already resident is not returned to the kernel before the reset, and the gradient reuses it without it being counted;
#'     for the LC_MVP_bin_PartialLog_v5 models this lowered the result by 5 to 10% (1.9 to 4.8 MB at 1,000 to 5,000
#'     individuals). free_heap_release_note records whether it was released.
#'   \item It includes a fixed part (code pages first touched by the gradient, the copies of the parameters and of the
#'     gradient, and the first arena block), which matters only for very small data.
#'   \item Precision: the peak (VmHWM) and VmRSS come from the kernel's resident-memory counters, which on Linux 6.2 and later
#'     are updated in per-CPU batches of max(32, 2 x online CPUs) pages (128 KiB with up to 16 CPUs, 1.5 MiB with 192 CPUs),
#'     so the transient part is known to within about one batch; the child process is pinned to one CPU so that the error does
#'     not grow with the number of CPUs it runs on. The resident growth, from /proc/self/smaps_rollup, is exact.
#'   \item The tape of a Stan model can depend on the evaluation point, through the branches of the model code. Points far in
#'     the tails (e.g. random values on (-2, 2) for every unconstrained parameter) can take more expensive branches than the
#'     points the sampler visits (for the LC_MVP_bin_PartialLog_v5 model (6 binary tests), 1.1 to 2.4 times the memory at zero
#'     in tests at 1,000 and 5,000 individuals, depending on the seed), so the initial values of the real run (init_list) or,
#'     by default, zero on the unconstrained scale are used.
#'   \item It must use the same compiled model, the same data (the full data, not a subset) and the same settings as the real
#'     run. For a model with within-chain chunking (e.g. reduce_sum), the model should be evaluated as one chunk (a chunk
#'     size equal to the number of observations), otherwise the result is the memory of one chunk, not of the full data.
#'   \item The evaluation is single-threaded (STAN_NUM_THREADS = 1 in the child process).
#'   \item Linux only: it reads /proc/self/status and /proc/self/smaps_rollup and writes /proc/self/clear_refs. If the peak
#'     cannot be reset in a repeat, that repeat uses the resident growth only (which misses temporaries freed before the
#'     gradient returns); the median is then taken over the repeats that did reset the peak, and a warning is given if none did.
#'   \item Built-in NicoStan models (Model_type other than "Stan") use manual gradients whose memory is not measured here.
#' }
#'
#' @param model_so_file Path of the compiled BridgeStan model library (.so), as NicoStan builds it. Not needed when
#'   model_object is given.
#' @param json_file_path Path of the JSON data file. Give either json_file_path or Stan_data_list (not needed when model_object
#'   is given).
#' @param Stan_data_list The data as an R list (written to a temporary JSON file with NicoStan's JSON writer).
#' @param model_object A NicoStan model object (from Nico_model$new() / MVP_model$new()) or its init_object; the compiled model
#'   (init_object$model_so_file) and the data (init_object$json_file_path) are taken from it.
#' @param init_list Optional initial values for one chain on the constrained scale, in the format of one element of NicoStan's
#'   init_lists_per_chain; the gradient is evaluated at this point. Parameters not given are completed with the model's values
#'   at zero on the unconstrained scale, as in NicoStan's sampler. Each value is matched to the model's parameter of the same
#'   name and read in R's (column-major) order, so a length-1 array parameter can be given as a plain number. Recommended: the
#'   initial values of the real run.
#' @param unconstrained_parameter_vector Optional point on the unconstrained scale at which the gradient is evaluated (in place
#'   of init_list).
#' @param default_evaluation_point Point used when neither init_list nor unconstrained_parameter_vector is given: "zero"
#'   (default; zero on the unconstrained scale) or "random" (uniform on initial_values_range_unconstrained). When the log
#'   density is not finite at zero, random points are tried.
#' @param initial_values_seed Seed of the random evaluation point (default 123), and of the model's random number generator.
#'   If the log density is not finite there, the next seeds are tried (up to max_attempts_for_finite_log_density).
#' @param initial_values_range_unconstrained Range of the random evaluation point on the unconstrained scale (default c(-2, 2),
#'   as for Stan's default random initial values).
#' @param max_attempts_for_finite_log_density Number of random points tried (default 10; a whole number of at least 1).
#' @param n_repeats Number of fresh-process repeats (default 3); the median is returned.
#' @param N_observations Optional number of observations (e.g. individuals) in the data; if given, bytes per observation are
#'   also returned.
#' @param propto,jacobian Passed to log_density_gradient (default TRUE and TRUE, as in NicoStan's sampler).
#' @return A list with M_bytes_per_chain (median over the repeats used), M_bytes_per_chain_each_repeat, repeats_used_for_median,
#'   bytes_per_observation (NA when N_observations is not given), N_observations, n_repeats, measurement_method,
#'   measurement_method_each_repeat, median_note, free_heap_release_note, evaluation_point_description,
#'   n_unconstrained_parameters, log_density_with_gradient, gradient_norm, the readings of every repeat (resident memory before
#'   and after the gradient, resident growth, transient part, VmHWM before the reset, VmHWM and VmRSS before and after the
#'   gradient), child_process_id_each_repeat, pinned_CPU_index_each_repeat, resident_memory_counter_batch_bytes, model_so_file,
#'   json_file_path, propto and jacobian.
#' @examples
#' \dontrun{
#' ## From a compiled model and its data file:
#' M_estimate <- fn_estimate_initial_M_bytes_per_chain( model_so_file   = "LC_MVP_bin_PartialLog_v5_model.so",
#'                                                      json_file_path  = "data_N5000.json",
#'                                                      N_observations  = 5000)
#' M_estimate$M_bytes_per_chain
#'
#' ## At the initial values of the real run (one chain's element of init_lists_per_chain):
#' M_estimate <- fn_estimate_initial_M_bytes_per_chain( model_so_file   = "LC_MVP_bin_PartialLog_v5_model.so",
#'                                                      json_file_path  = "data_N5000.json",
#'                                                      init_list       = init_lists_per_chain[[1]],
#'                                                      N_observations  = 5000)
#'
#' ## From a NicoStan model object, then the starting number of chunks for 64 chains:
#' model <- Nico_model$new( Model_type           = "Stan",
#'                          Stan_model_file_path = "model.stan",
#'                          Stan_data_list       = Stan_data_list)
#' M_estimate <- fn_estimate_initial_M_bytes_per_chain(model_object = model)
#' fn_compute_initial_N_chunks( M_bytes_per_chain = M_estimate,
#'                              N_chains          = 64)
#' }
#' @export
fn_estimate_initial_M_bytes_per_chain <-  function( model_so_file                        = NULL,
                                                    json_file_path                       = NULL,
                                                    Stan_data_list                       = NULL,
                                                    model_object                         = NULL,
                                                    init_list                            = NULL,
                                                    unconstrained_parameter_vector       = NULL,
                                                    default_evaluation_point             = "zero",
                                                    initial_values_seed                  = 123,
                                                    initial_values_range_unconstrained   = c(-2, 2),
                                                    max_attempts_for_finite_log_density  = 10,
                                                    n_repeats                            = 3,
                                                    N_observations                       = NULL,
                                                    propto                               = TRUE,
                                                    jacobian                             = TRUE,
                                                    n_nuisance_owned_by_NicoStan         = 0
) {

        ##
        ## ---- Linux only:
        ##
        if (!identical(x = Sys.info()[["sysname"]], y = "Linux") || !file.exists("/proc/self/status")) {
              stop(paste0("fn_estimate_initial_M_bytes_per_chain: the measurement reads /proc/self/status and writes /proc/self/clear_refs, ",
                          "so it runs on Linux only (this system: ", Sys.info()[["sysname"]], ")."))
        }
        ##
        ## ---- compiled model and data from a NicoStan model object (or its init_object):
        ##
        if (!is.null(x = model_object)) {

              if (!is.null(x = model_so_file) || !is.null(x = json_file_path) || !is.null(x = Stan_data_list)) {
                    stop(paste0("fn_estimate_initial_M_bytes_per_chain: give either model_object, or model_so_file with ",
                                "json_file_path / Stan_data_list, not both."))
              }
              ##
              init_object <-  if (is.list(x = model_object) && !is.null(x = model_object$model_so_file)) model_object else model_object$init_object
              ##
              if (is.null(x = init_object) || is.null(x = init_object$model_so_file) || is.null(x = init_object$json_file_path)) {
                    stop(paste0("fn_estimate_initial_M_bytes_per_chain: model_object has no init_object with model_so_file and ",
                                "json_file_path; initialise the model first (e.g. Nico_model$new(..., Stan_data_list = ...)) or give ",
                                "model_so_file and json_file_path."))
              }
              ##
              if (!is.null(x = init_object$Model_type) && !identical(x = init_object$Model_type, y = "Stan")) {
                    stop(paste0("fn_estimate_initial_M_bytes_per_chain: Model_type = \"", init_object$Model_type, "\" uses NicoStan's manual ",
                                "gradient, whose memory is not measured by this function (it measures Stan models through BridgeStan)."))
              }
              ##
              model_so_file <-  init_object$model_so_file
              json_file_path <-  init_object$json_file_path

        }
        ##
        ## ---- checks:
        ##
        if (is.null(x = model_so_file) || !is.character(model_so_file) || length(x = model_so_file) != 1 || !file.exists(model_so_file)) {
              stop(paste0("fn_estimate_initial_M_bytes_per_chain: model_so_file must be the path of an existing compiled BridgeStan ",
                          "model (.so); got: ", paste(as.character(model_so_file), collapse = ", ")))
        }
        ##
        if (is.null(x = json_file_path) == is.null(x = Stan_data_list)) {
              stop("fn_estimate_initial_M_bytes_per_chain: give exactly one of json_file_path and Stan_data_list (the full data of the real run).")
        }
        ##
        fn_is_single_whole_number <-  function(value, minimum_value) {

              return(is.numeric(value) && length(x = value) == 1 && is.finite(x = value) && value >= minimum_value && value == round(x = value))
        }

        ##
        if (!fn_is_single_whole_number(value = n_repeats, minimum_value = 1)) {
              stop("fn_estimate_initial_M_bytes_per_chain: n_repeats must be a single positive whole number.")
        }
        ##
        if (!fn_is_single_whole_number(value = max_attempts_for_finite_log_density, minimum_value = 1)) {
              stop("fn_estimate_initial_M_bytes_per_chain: max_attempts_for_finite_log_density must be a single positive whole number.")
        }
        ##
        if (!fn_is_single_whole_number(value = initial_values_seed, minimum_value = -Inf)) {
              stop("fn_estimate_initial_M_bytes_per_chain: initial_values_seed must be a single finite whole number.")
        }
        ##
        if (!is.null(x = N_observations) &&
            (!is.numeric(N_observations) || length(x = N_observations) != 1 || !is.finite(x = N_observations) || N_observations <= 0)) {
              stop("fn_estimate_initial_M_bytes_per_chain: N_observations must be NULL or a single positive number.")
        }
        ##
        if (!is.null(x = unconstrained_parameter_vector) &&
            (!is.numeric(unconstrained_parameter_vector) || any(!is.finite(x = unconstrained_parameter_vector)))) {
              stop("fn_estimate_initial_M_bytes_per_chain: unconstrained_parameter_vector must be NULL or a numeric vector of finite values.")
        }
        ##
        if (!is.null(x = unconstrained_parameter_vector) && !is.null(x = init_list)) {
              stop("fn_estimate_initial_M_bytes_per_chain: give at most one of init_list and unconstrained_parameter_vector.")
        }
        ##
        if (!is.null(x = init_list) &&
            (!is.list(x = init_list) || length(x = init_list) == 0 || is.null(x = names(x = init_list)) || any(!nzchar(x = names(x = init_list))))) {
              stop("fn_estimate_initial_M_bytes_per_chain: init_list must be a named list of initial values for one chain.")
        }
        ##
        if (!is.character(default_evaluation_point) || length(x = default_evaluation_point) != 1 ||
            !(default_evaluation_point %in% c("zero", "random"))) {
              stop("fn_estimate_initial_M_bytes_per_chain: default_evaluation_point must be \"zero\" or \"random\".")
        }
        ##
        if (!is.numeric(initial_values_range_unconstrained) || length(x = initial_values_range_unconstrained) != 2 ||
            any(!is.finite(x = initial_values_range_unconstrained)) ||
            initial_values_range_unconstrained[1] >= initial_values_range_unconstrained[2]) {
              stop(paste0("fn_estimate_initial_M_bytes_per_chain: initial_values_range_unconstrained must be two finite numbers ",
                          "c(lower, upper) with lower < upper."))
        }
        ##
        ## ---- data written to a JSON file (when given as a list), then the paths:
        ##
        if (!is.null(x = Stan_data_list)) {
              json_file_path <-  convert_stan_data_list_to_JSON( stan_data_list = Stan_data_list,
                                                                 pkg_data_dir   = tempdir())
        }
        ##
        if (!is.character(json_file_path) || length(x = json_file_path) != 1 || !file.exists(json_file_path)) {
              stop(paste0("fn_estimate_initial_M_bytes_per_chain: json_file_path must be the path of an existing JSON data file; got: ",
                          paste(as.character(json_file_path), collapse = ", ")))
        }
        ##
        model_so_file <-  normalizePath(path = model_so_file, mustWork = TRUE)
        json_file_path <-  normalizePath(path = json_file_path, mustWork = TRUE)
        ##
        ## ---- the measurement kept from an earlier call on the same compiled library (unchanged: the same
        ##      size and modification time) with data of the same size in bytes, the same nuisance block and the
        ##      same propto and jacobian, so that each model and N is measured once: each measurement is a fresh
        ##      the sampling picker spent choosing for the robust Student-t's AVX-512 library at N = 50,000
        ##      (kept in a file beside the library, one entry per such case):
        ##
        file_of_the_kept_measurements <-  paste0(tools::file_path_sans_ext(model_so_file),
                                                 "_M_bytes_per_chain_kept.rds")
        earlier_file_of_the_kept_measurements <-  file_of_the_kept_measurements
        if (nchar(paste0(basename(file_of_the_kept_measurements),
                         ".being_written_by_", Sys.getpid()), type = "bytes") > 255) {
              folder_of_the_kept_measurements <-  tools::file_path_sans_ext(model_so_file)
              file_of_the_kept_measurements <-  file.path(folder_of_the_kept_measurements,
                                                          "M_bytes_per_chain_kept.rds")
        }
        # name_of_this_case <-  paste0("library_", file.size(model_so_file), "_bytes_modified_",
        #                              format(as.numeric(file.mtime(model_so_file)), digits = 17), "_data_",
        #                              file.size(json_file_path), "_bytes_nuisance_", n_nuisance_owned_by_NicoStan,
        #                              "_propto_", propto, "_jacobian_", jacobian)
        ##
        ## ---- chunk_size and N_blocks of the data are part of the case (data of one size in bytes can hold
        ##      different chunk sizes), N_blocks as given in the data or ceiling(N / chunk_size)
        ##
        text_of_the_data <-  readChar(con = json_file_path, nchars = file.size(json_file_path), useBytes = TRUE)
        fn_integer_of_the_data <-  function(name) {

              found <-  regmatches(x = text_of_the_data,
                                   m = regexec(pattern = paste0("\"", name, "\"\\s*:\\s*\\[?\\s*([0-9]+)"),
                                               text = text_of_the_data))[[1]]
              if (length(x = found) == 2) return(as.numeric(found[2]))
              return(NA_real_)

        }
        chunk_size_of_the_data <-  fn_integer_of_the_data(name = "chunk_size")
        N_blocks_of_the_data <-  fn_integer_of_the_data(name = "N_blocks")
        if (is.na(N_blocks_of_the_data) && !is.na(chunk_size_of_the_data)) {
              N_blocks_of_the_data <-  ceiling(fn_integer_of_the_data(name = "N") / chunk_size_of_the_data)
        }
        ## (an AUTO-CHUNKING library's number of chunks, read from its data: part of the case, so that two numbers
        ##  of chunks whose data files have the same size are never one measurement)
        N_chunks_of_auto_chunking_of_the_data <-  fn_integer_of_the_data(name = "N_chunks_of_auto_chunking")
        name_of_this_case <-  paste0("library_", file.size(model_so_file), "_bytes_modified_",
                                     format(as.numeric(file.mtime(model_so_file)), digits = 17), "_data_",
                                     file.size(json_file_path), "_bytes_nuisance_", n_nuisance_owned_by_NicoStan,
                                     "_propto_", propto, "_jacobian_", jacobian,
                                     "_chunk_size_", chunk_size_of_the_data, "_N_blocks_", N_blocks_of_the_data,
                                     if (!is.na(N_chunks_of_auto_chunking_of_the_data))
                                           paste0("_N_chunks_of_auto_chunking_",
                                                  N_chunks_of_auto_chunking_of_the_data) else "",
                                     ## (a nuisance-block library measured through NicoStan's own gradient call:
                                     ##  never one of the measurements kept before, through the values used outside)
                                     if (n_nuisance_owned_by_NicoStan > 0) "_through_NicoStans_gradient_call" else "")
        kept_measurements <-  list()
        for (file_to_read in unique(c(earlier_file_of_the_kept_measurements,
                                     file_of_the_kept_measurements))) {
              if (file.exists(file_to_read)) {
                    earlier_measurements <-  tryCatch(readRDS(file_to_read),
                                                       error = function(condition) list())
                    kept_measurements[names(earlier_measurements)] <-  earlier_measurements
              }
        }
        if (!is.null(kept_measurements[[name_of_this_case]])) {

              message(colourise(paste0("M_bytes/chain = ",
                                       formatC(x = kept_measurements[[name_of_this_case]]$M_bytes_per_chain,
                                               format = "f", digits = 0, big.mark = ","),
                                       " bytes, kept from the earlier measurement of ",
                                       basename(path = model_so_file), " on data of the same size (",
                                       file_of_the_kept_measurements, ")"), "cyan"))
              return(kept_measurements[[name_of_this_case]])
        }

        ##
        ## ---- the child's function has baseenv() as its enclosing environment, so it carries nothing from this session:
        ##
        fn_child_measurement <-  fn_measure_one_gradient_memory_in_this_process
        environment(fn = fn_child_measurement) <-  baseenv()
        ##
        child_process_environment <-  c(callr::rcmd_safe_env(),
                                        STAN_NUM_THREADS = "1")
        ##
        free_heap_release_location <-  fn_locate_free_heap_release_function()
        ##
        message(colourise(paste0("M_bytes/chain: ", n_repeats, " fresh-process repeat(s) of one log density and gradient of ",
                                 basename(path = model_so_file), " on ", basename(path = json_file_path)), "cyan"))
        ##
        repeat_results <-  vector(mode = "list", length = n_repeats)
        ##
        for (repeat_index in seq_len(length.out = n_repeats)) {

              repeat_results[[repeat_index]] <-  tryCatch(
                    expr  = callr::r( func           = fn_child_measurement,
                                      args           = list( model_so_file                        = model_so_file,
                                                             json_file_path                       = json_file_path,
                                                             unconstrained_parameter_vector       = unconstrained_parameter_vector,
                                                             init_list                            = init_list,
                                                             default_evaluation_point             = default_evaluation_point,
                                                             initial_values_seed                  = initial_values_seed,
                                                             initial_values_range_unconstrained   = initial_values_range_unconstrained,
                                                             max_attempts_for_finite_log_density  = max_attempts_for_finite_log_density,
                                                             propto                               = propto,
                                                             jacobian                             = jacobian,
                                                             free_heap_release_package_name       = free_heap_release_location$package_name,
                                                             free_heap_release_function_name      = free_heap_release_location$function_name,
                                                             n_nuisance_owned_by_NicoStan         =
                                                                   n_nuisance_owned_by_NicoStan),
                                      libpath        = .libPaths(),
                                      env            = child_process_environment,
                                      system_profile = FALSE,
                                      user_profile   = FALSE,
                                      error          = "error"),
                    error = function(condition) {

                          ## (callr wraps the child's error; its own message is the parent condition's):
                          child_error_message <-  conditionMessage(condition)
                          if (!is.null(x = condition$parent)) child_error_message <-  conditionMessage(condition$parent)
                          ##
                          ## (bridgestan's errors can contain the whole data; only the start is kept):
                          if (nchar(x = child_error_message) > 500) {
                                child_error_message <-  paste0(substr(x = child_error_message, start = 1, stop = 500), " ...")
                          }
                          ##
                          stop(paste0("fn_estimate_initial_M_bytes_per_chain: repeat ", repeat_index, " failed in the child process: ",
                                      child_error_message), call. = FALSE)
                    })

              ##
              message(colourise(paste0("  repeat ", repeat_index, " of ", n_repeats, ": ",
                                       formatC( x = repeat_results[[repeat_index]]$gradient_memory_bytes,
                                                format = "f",
                                                digits = 0,
                                                big.mark = ","),
                                       " bytes (", repeat_results[[repeat_index]]$measurement_method_code, ")"), "cyan"))

        }
        ##
        ## ---- summary over the repeats (the two estimators are never mixed in one median: when some repeats could not reset the
        ##      peak, the median is over the repeats that did):
        ##
        fn_extract_from_repeats <-  function(field_name) {

              sapply(X = repeat_results, FUN = function(repeat_result) repeat_result[[field_name]])
        }

        ##
        M_bytes_per_chain_each_repeat <-  fn_extract_from_repeats(field_name = "gradient_memory_bytes")
        measurement_method_code_each_repeat <-  fn_extract_from_repeats(field_name = "measurement_method_code")
        measurement_method_each_repeat <-  fn_extract_from_repeats(field_name = "measurement_method")
        ##
        repeat_used_peak <-  measurement_method_code_each_repeat == "peak"
        median_note <-  ""
        ##
        if (all(repeat_used_peak)) {

              repeats_used_for_median <-  seq_len(length.out = n_repeats)

        } else if (any(repeat_used_peak)) {

              repeats_used_for_median <-  which(repeat_used_peak)
              median_note <-  paste0("median over the ", length(x = repeats_used_for_median), " repeat(s) that reset the peak; ",
                                     n_repeats - length(x = repeats_used_for_median), " repeat(s) could not reset it and are not used")

        } else {

              repeats_used_for_median <-  seq_len(length.out = n_repeats)
              median_note <-  "no repeat could reset the peak, so the median is of the resident growth only (a lower value)"
              warning(paste0("fn_estimate_initial_M_bytes_per_chain: the peak resident memory could not be reset (/proc/self/clear_refs) ",
                             "in any repeat, so M_bytes/chain is the growth of the resident memory only (",
                             measurement_method_each_repeat[1], "); this misses temporaries freed before the gradient returns."))

        }
        ##
        M_bytes_per_chain <-  stats::median(M_bytes_per_chain_each_repeat[repeats_used_for_median])
        measurement_method <-  unique(x = measurement_method_each_repeat[repeats_used_for_median])
        ##
        bytes_per_observation <-  if (is.null(x = N_observations)) NA_real_ else M_bytes_per_chain / N_observations
        ##
        free_heap_release_note <-  paste(unique(x = fn_extract_from_repeats(field_name = "free_heap_release_note")), collapse = " | ")
        ##
        init_list_names_not_in_model <-  repeat_results[[1]]$init_list_names_not_in_model
        if (length(x = init_list_names_not_in_model) > 0) {
              warning(paste0("fn_estimate_initial_M_bytes_per_chain: init_list entries that match no parameter of the model were ignored: ",
                             paste(init_list_names_not_in_model, collapse = ", ")))
        }
        ##
        M_range_description <-  paste0("range ", formatC( x = min(M_bytes_per_chain_each_repeat) / 1e6,
                                                          format = "f",
                                                          digits = 2), " to ",
                                       formatC( x = max(M_bytes_per_chain_each_repeat) / 1e6,
                                                format = "f",
                                                digits = 2), " MB")
        ##
        bytes_per_observation_description <-  if (is.null(x = N_observations)) "" else {
              paste0("; ", formatC(x = bytes_per_observation, format = "f", digits = 0, big.mark = ","),
                     " bytes per observation (N_observations = ", N_observations, ")")
        }
        ##
        message(colourise(paste0("initial M_bytes/chain = ", formatC( x = M_bytes_per_chain,
                                                                      format = "f",
                                                                      digits = 0,
                                                                      big.mark = ","),
                                 " bytes (", formatC( x = M_bytes_per_chain / 1e6,
                                                      format = "f",
                                                      digits = 2), " MB; median of ",
                                 length(x = repeats_used_for_median), " fresh-process repeat(s), ", M_range_description, ")",
                                 bytes_per_observation_description), "cyan"))
        ##
        message(colourise(paste0("  ", free_heap_release_note, if (nzchar(x = median_note)) paste0("; ", median_note) else ""), "cyan"))
        ##
        # return(list( M_bytes_per_chain                                 = M_bytes_per_chain,
        measurement <-  list( M_bytes_per_chain                                 = M_bytes_per_chain,
                     M_bytes_per_chain_each_repeat                     = M_bytes_per_chain_each_repeat,
                     repeats_used_for_median                           = repeats_used_for_median,
                     bytes_per_observation                             = bytes_per_observation,
                     N_observations                                    = if (is.null(x = N_observations)) NA_real_ else N_observations,
                     n_repeats                                         = n_repeats,
                     measurement_method                                = measurement_method,
                     measurement_method_each_repeat                    = measurement_method_code_each_repeat,
                     median_note                                       = median_note,
                     free_heap_release_note                            = free_heap_release_note,
                     evaluation_point_description                      = repeat_results[[1]]$evaluation_point_description,
                     n_unconstrained_parameters                        = repeat_results[[1]]$n_unconstrained_parameters,
                     log_density_with_gradient                         = repeat_results[[1]]$log_density_with_gradient,
                     gradient_norm                                     = repeat_results[[1]]$gradient_norm,
                     resident_memory_growth_bytes_each_repeat          = fn_extract_from_repeats(field_name = "resident_memory_growth_bytes"),
                     transient_peak_bytes_each_repeat                  = fn_extract_from_repeats(field_name = "transient_peak_bytes"),
                     resident_memory_before_gradient_bytes_each_repeat = fn_extract_from_repeats(field_name = "resident_memory_before_gradient_bytes"),
                     resident_memory_after_gradient_bytes_each_repeat  = fn_extract_from_repeats(field_name = "resident_memory_after_gradient_bytes"),
                     VmHWM_before_reset_bytes_each_repeat              = fn_extract_from_repeats(field_name = "VmHWM_before_reset_bytes"),
                     VmHWM_before_gradient_bytes_each_repeat           = fn_extract_from_repeats(field_name = "VmHWM_before_gradient_bytes"),
                     VmRSS_before_gradient_bytes_each_repeat           = fn_extract_from_repeats(field_name = "VmRSS_before_gradient_bytes"),
                     VmHWM_after_gradient_bytes_each_repeat            = fn_extract_from_repeats(field_name = "VmHWM_after_gradient_bytes"),
                     VmRSS_after_gradient_bytes_each_repeat            = fn_extract_from_repeats(field_name = "VmRSS_after_gradient_bytes"),
                     child_process_id_each_repeat                      = fn_extract_from_repeats(field_name = "child_process_id"),
                     pinned_CPU_index_each_repeat                      = fn_extract_from_repeats(field_name = "pinned_CPU_index"),
                     resident_memory_counter_batch_bytes               = repeat_results[[1]]$resident_memory_counter_batch_bytes,
                     model_so_file                                     = model_so_file,
                     json_file_path                                    = json_file_path,
                     propto                                            = propto,
                     jacobian                                          = jacobian)
        ##
        ## ---- kept for the next call on this library and data size (the file written whole, then renamed):
        ##
        kept_measurements[[name_of_this_case]] <-  measurement
        file_being_written <-  paste0(file_of_the_kept_measurements, ".being_written_by_", Sys.getpid())
        kept <-  tryCatch({
              dir.create(dirname(file_of_the_kept_measurements), recursive = TRUE, showWarnings = FALSE)
              saveRDS(kept_measurements, file_being_written)
              file.rename(file_being_written, file_of_the_kept_measurements)
        }, error = function(condition) FALSE)
        if (!isTRUE(kept)) {

              message(colourise(paste0("M_bytes/chain: the measurement could not be kept in ",
                                       file_of_the_kept_measurements), "cyan"))
        }

        return(measurement)

}




#### ---- exported function for a built-in model (manual gradients) ------------------------------------------

#' Estimate the initial M_bytes/chain of a built-in NicoStan model: memory used by one full-data
#' log-density-and-gradient evaluation of one chain (manual gradients)
#'
#' Measures, for a built-in NicoStan model (Model_type "MVP", "MVOP", "LC_MVP" or "LC_MVOP", whose manual
#' gradients are computed over the N observations in num_chunks chunks), the memory (in bytes) that ONE
#' evaluation of the log density and its gradient on the full data uses in one chain, with the model evaluated
#' as one chunk. This is the M_bytes/chain from which [fn_compute_burnin_n_threads_WCP_and_num_chunks()] and
#' [fn_compute_initial_N_chunks()] choose the threads and chunks of a built-in model.
#'
#' For each repeat, a fresh child R process, pinned to one CPU, evaluates the log density and gradient once with
#' BayesMVP's Rcpp_wrapper_fn_lp_grad() (BayesMVP provides the built-in models' C++ code; NicoStan's own compiled
#' code evaluates Stan models only), with the chunk count of Model_args_as_Rcpp_List set to 1, runs gc()
#' and returns the free heap memory to the kernel (malloc_trim(0)), resets its peak resident memory (by writing
#' "5" to /proc/self/clear_refs), reads its resident memory, evaluates the log density and gradient once more,
#' and reads the resident memory again and the peak (VmHWM) and VmRSS. The model's workspace is allocated inside
#' each evaluation and freed when it returns, so the first evaluation leaves only the code and data pages
#' resident; the result of the repeat is the growth of the resident memory plus the transient part
#' max(0, VmHWM - VmRSS) of the second evaluation. The value returned is the median over the repeats.
#'
#' Limits of the measurement, in addition to those of [fn_estimate_initial_M_bytes_per_chain()] (a lower value
#' for pages reused without being returned to the kernel; the peak known to within about one batch of the
#' kernel's resident-memory counters; Linux only):
#' \itemize{
#'   \item It includes the copies that Rcpp_wrapper_fn_lp_grad() makes of its inputs (the main and nuisance
#'     parameters and y) and its output vector, which the sampler does not make for each gradient; these are a
#'     small part of the total.
#'   \item The evaluation point is zero on the unconstrained scale, for the main and the nuisance parameters.
#'     The memory of the manual gradient depends on the point only through the log-scale (PartialLog)
#'     evaluation, which is not used here.
#'   \item It uses the C++ code of the installed NicoStan.
#' }
#'
#' @param Model_type The built-in model: "MVP", "MVOP", "LC_MVP" or "LC_MVOP".
#' @param init_object The model's initialisation object (it holds Model_args_as_Rcpp_List, n_params_main,
#'   n_nuisance and model_args_list$y).
#' @param y The data: an integer matrix with one row per observation and one column per test (default: the
#'   data of init_object).
#' @param n_repeats Number of fresh-process repeats (default 3); the median is returned.
#' @return A list with M_bytes_per_chain (median over the repeats used), M_bytes_per_chain_each_repeat,
#'   repeats_used_for_median, bytes_per_observation, N_observations, n_tests, n_repeats, measurement_method,
#'   measurement_method_each_repeat, median_note, free_heap_release_note, evaluation_point_description,
#'   n_main_parameters, n_nuisance_parameters, log_density_with_gradient, the readings of every repeat,
#'   child_process_id_each_repeat, pinned_CPU_index_each_repeat, Model_type and vect_type.
#' @export
fn_estimate_initial_M_bytes_per_chain_of_built_in_model <-  function( Model_type,
                                                                      init_object,
                                                                      y          = NULL,
                                                                      n_repeats  = 3
) {

        ##
        ## ---- Linux only, built-in models only:
        ##
        if (!identical(x = Sys.info()[["sysname"]], y = "Linux") || !file.exists("/proc/self/status")) {
              stop(paste0("fn_estimate_initial_M_bytes_per_chain_of_built_in_model: the measurement reads ",
                          "/proc/self/status and writes /proc/self/clear_refs, so it runs on Linux only (this ",
                          "system: ", Sys.info()[["sysname"]], ")."))
        }
        ##
        built_in_model_types <-  c("MVP", "MVOP", "LC_MVP", "LC_MVOP")
        if (!is.character(Model_type) || (length(x = Model_type) != 1) ||
            !(Model_type %in% built_in_model_types)) {
              stop(paste0("fn_estimate_initial_M_bytes_per_chain_of_built_in_model: Model_type must be one of ",
                          paste(built_in_model_types, collapse = ", "), " (for a Stan model, use ",
                          "fn_estimate_initial_M_bytes_per_chain()); got: ",
                          paste(as.character(Model_type), collapse = ", ")))
        }
        ##
        if (is.null(x = init_object$Model_args_as_Rcpp_List) || is.null(x = init_object$n_params_main) ||
            is.null(x = init_object$n_nuisance)) {
              stop(paste0("fn_estimate_initial_M_bytes_per_chain_of_built_in_model: init_object must hold ",
                          "Model_args_as_Rcpp_List, n_params_main and n_nuisance (the model's initialisation ",
                          "object)."))
        }
        ##
        if (is.null(x = y)) y <-  init_object$model_args_list$y
        if (!is.matrix(x = y) || (nrow(x = y) < 1) || (ncol(x = y) < 1)) {
              stop(paste0("fn_estimate_initial_M_bytes_per_chain_of_built_in_model: y must be the data matrix ",
                          "(one row per observation, one column per test)."))
        }
        ##
        if (!is.numeric(n_repeats) || (length(x = n_repeats) != 1) || !is.finite(x = n_repeats) ||
            (n_repeats < 1) || (n_repeats != round(x = n_repeats))) {
              stop(paste0("fn_estimate_initial_M_bytes_per_chain_of_built_in_model: n_repeats must be a single ",
                          "positive whole number."))
        }
        ##
        ## ---- the model evaluated as one chunk (the full data at once), at zero on the unconstrained scale:
        ##
        Model_args_as_Rcpp_List_one_chunk <-  init_object$Model_args_as_Rcpp_List
        Model_args_as_Rcpp_List_one_chunk$Model_args_ints[4] <-  1
        Model_args_as_Rcpp_List_one_chunk$model_so_file <-  "none"
        Model_args_as_Rcpp_List_one_chunk$json_file_path <-  "none"
        ##
        theta_main_vec <-  rep(x = 0, times = init_object$n_params_main)
        theta_us_vec <-  rep(x = 0, times = init_object$n_nuisance)
        storage.mode(y) <-  "integer"
        ##
        N_observations <-  nrow(x = y)
        n_tests <-  ncol(x = y)
        vect_type <-  as.character(Model_args_as_Rcpp_List_one_chunk$Model_args_strings[1])
        ##
        ## ---- the child's function has baseenv() as its enclosing environment, so it carries nothing from this
        ##      session:
        ##
        fn_child_measurement <-  fn_measure_one_gradient_memory_of_built_in_model_in_this_process
        environment(fn = fn_child_measurement) <-  baseenv()
        ##
        free_heap_release_location <-  fn_locate_free_heap_release_function()
        ##
        message(colourise(paste0("M_bytes/chain: ", n_repeats, " fresh-process repeat(s) of one log density and ",
                                 "manual gradient of the built-in ", Model_type, " model (N = ", N_observations,
                                 ", ", n_tests, " tests, one chunk)"), "cyan"))
        ##
        repeat_results <-  vector(mode = "list", length = n_repeats)
        ##
        for (repeat_index in seq_len(length.out = n_repeats)) {

              repeat_results[[repeat_index]] <-  tryCatch(
                    expr  = callr::r( func           = fn_child_measurement,
                                      args           = list( Model_type                       = Model_type,
                                                             theta_main_vec                   = theta_main_vec,
                                                             theta_us_vec                     = theta_us_vec,
                                                             y                                = y,
                                                             Model_args_as_Rcpp_List          =
                                                                   Model_args_as_Rcpp_List_one_chunk,
                                                             gradient_package_name            = "BayesMVP",
                                                             free_heap_release_package_name   =
                                                                   free_heap_release_location$package_name,
                                                             free_heap_release_function_name  =
                                                                   free_heap_release_location$function_name),
                                      libpath        = .libPaths(),
                                      env            = callr::rcmd_safe_env(),
                                      system_profile = FALSE,
                                      user_profile   = FALSE,
                                      error          = "error"),
                    error = function(condition) {

                          ## (callr wraps the child's error; its own message is the parent condition's):
                          child_error_message <-  conditionMessage(condition)
                          if (!is.null(x = condition$parent)) {
                                child_error_message <-  conditionMessage(condition$parent)
                          }
                          if (nchar(x = child_error_message) > 500) {
                                child_error_message <-  paste0(substr(x = child_error_message, start = 1,
                                                                      stop = 500), " ...")
                          }
                          stop(paste0("fn_estimate_initial_M_bytes_per_chain_of_built_in_model: repeat ",
                                      repeat_index, " failed in the child process: ", child_error_message),
                               call. = FALSE)
                    })

              ##
              message(colourise(paste0("  repeat ", repeat_index, " of ", n_repeats, ": ",
                                       formatC(x = repeat_results[[repeat_index]]$gradient_memory_bytes,
                                               format = "f", digits = 0, big.mark = ","),
                                       " bytes (", repeat_results[[repeat_index]]$measurement_method_code, ")"),
                                "cyan"))

        }
        ##
        ## ---- summary over the repeats (the two estimators are never mixed in one median: when some repeats
        ##      could not reset the peak, the median is over the repeats that did):
        ##
        fn_extract_from_repeats <-  function(field_name) {

              sapply(X = repeat_results, FUN = function(repeat_result) repeat_result[[field_name]])
        }

        ##
        M_bytes_per_chain_each_repeat <-  fn_extract_from_repeats(field_name = "gradient_memory_bytes")
        measurement_method_code_each_repeat <-  fn_extract_from_repeats(field_name = "measurement_method_code")
        measurement_method_each_repeat <-  fn_extract_from_repeats(field_name = "measurement_method")
        ##
        repeat_used_peak <-  measurement_method_code_each_repeat == "peak"
        median_note <-  ""
        ##
        if (all(repeat_used_peak)) {

              repeats_used_for_median <-  seq_len(length.out = n_repeats)

        } else if (any(repeat_used_peak)) {

              repeats_used_for_median <-  which(repeat_used_peak)
              median_note <-  paste0("median over the ", length(x = repeats_used_for_median), " repeat(s) that ",
                                     "reset the peak; ", n_repeats - length(x = repeats_used_for_median),
                                     " repeat(s) could not reset it and are not used")

        } else {

              repeats_used_for_median <-  seq_len(length.out = n_repeats)
              median_note <-  paste0("no repeat could reset the peak, so the median is of the resident growth ",
                                     "only (a lower value)")
              warning(paste0("fn_estimate_initial_M_bytes_per_chain_of_built_in_model: the peak resident memory ",
                             "could not be reset (/proc/self/clear_refs) in any repeat, so M_bytes/chain is the ",
                             "growth of the resident memory only (", measurement_method_each_repeat[1], "); ",
                             "this misses temporaries freed before the gradient returns."))

        }
        ##
        M_bytes_per_chain <-  stats::median(M_bytes_per_chain_each_repeat[repeats_used_for_median])
        measurement_method <-  unique(x = measurement_method_each_repeat[repeats_used_for_median])
        bytes_per_observation <-  M_bytes_per_chain / N_observations
        free_heap_release_notes <-  fn_extract_from_repeats(field_name = "free_heap_release_note")
        free_heap_release_note <-  paste(unique(x = free_heap_release_notes), collapse = " | ")
        ##
        message(colourise(paste0("initial M_bytes/chain = ",
                                 formatC(x = M_bytes_per_chain, format = "f", digits = 0, big.mark = ","),
                                 " bytes (", formatC(x = M_bytes_per_chain / 1e6, format = "f", digits = 2),
                                 " MB; median of ", length(x = repeats_used_for_median), " fresh-process ",
                                 "repeat(s); ", formatC(x = bytes_per_observation, format = "f", digits = 0,
                                                        big.mark = ","),
                                 " bytes per observation)"), "cyan"))
        message(colourise(paste0("  ", free_heap_release_note,
                                 if (nzchar(x = median_note)) paste0("; ", median_note) else ""), "cyan"))
        ##
        return(list( M_bytes_per_chain                                 = M_bytes_per_chain,
                     M_bytes_per_chain_each_repeat                     = M_bytes_per_chain_each_repeat,
                     repeats_used_for_median                           = repeats_used_for_median,
                     bytes_per_observation                             = bytes_per_observation,
                     N_observations                                    = N_observations,
                     n_tests                                           = n_tests,
                     n_repeats                                         = n_repeats,
                     measurement_method                                = measurement_method,
                     measurement_method_each_repeat                    = measurement_method_code_each_repeat,
                     median_note                                       = median_note,
                     free_heap_release_note                            = free_heap_release_note,
                     evaluation_point_description                      = "zero on the unconstrained scale",
                     n_main_parameters                                 = init_object$n_params_main,
                     n_nuisance_parameters                             = init_object$n_nuisance,
                     log_density_with_gradient                         =
                           repeat_results[[1]]$log_density_with_gradient,
                     all_outputs_finite_each_repeat                    =
                           fn_extract_from_repeats(field_name = "all_outputs_finite"),
                     resident_memory_growth_bytes_each_repeat          =
                           fn_extract_from_repeats(field_name = "resident_memory_growth_bytes"),
                     transient_peak_bytes_each_repeat                  =
                           fn_extract_from_repeats(field_name = "transient_peak_bytes"),
                     child_process_id_each_repeat                      =
                           fn_extract_from_repeats(field_name = "child_process_id"),
                     pinned_CPU_index_each_repeat                      =
                           fn_extract_from_repeats(field_name = "pinned_CPU_index"),
                     resident_memory_counter_batch_bytes               =
                           repeat_results[[1]]$resident_memory_counter_batch_bytes,
                     Model_type                                        = Model_type,
                     vect_type                                         = vect_type))

}





#### ---- M_bytes/chain of a built-in model counted from the workspace of its manual gradient --------------------------------------------

#' M_bytes/chain of a built-in model, counted from the workspace of its manual gradient
#'
#' Paper 1's M_bytes/chain for a manual gradient: the doubles that the gradient's workspace holds for each
#' individual (one row of the data, with all n_tests tests), counted from the arrays which BayesMVP's C++ resizes
#' for each chunk, times 8 bytes, times N, as in Paper 1 for the binary LC-MVP (31 T + 15 doubles, i.e. 1608 bytes
#' per individual with T = 6 tests):
#' \itemize{
#'   \item the binary models (Model_type "MVP" and "LC_MVP"; resize_MVP_workspace() in
#'     MVP_lp_grad_MD_AD_fns.hpp): 31 T + 13 + n_class doubles per individual (the five arrays of the latent
#'     classes are made for 2 classes, also for the MVP; lp_array has one column per class);
#'   \item the ordinal models (Model_type "MVOP" and "LC_MVOP"; fn_MVOP_resize_ws_for_chunk() in
#'     MVOP_lp_grad_MD_AD_fns.hpp): (36 + n_class) T + 15 + n_class doubles per individual (the same five arrays
#'     for 2 classes, plus Upper_Bound_Z for each class).
#' }
#' The inputs read for each individual from outside the workspace (its rows of y and of the covariates, and its
#' nuisance parameters) and the nuisance gradient written to the output are not counted, as in Paper 1. Nothing is
#' measured: unlike fn_estimate_initial_M_bytes_per_chain_of_built_in_model(), which measures the growth of the
#' resident memory of a whole gradient evaluation (2 to 3 times this count for these models), this is the working
#' set from which Paper 1's rule sizes the chunks of a manual gradient.
#'
#' @param Model_type The built-in model: "MVP", "MVOP", "LC_MVP" or "LC_MVOP".
#' @param N The number of individuals (rows of the data).
#' @param n_tests The number of tests (columns of the data), T.
#' @return A list: M_bytes_per_chain, bytes_per_individual, doubles_per_individual, N, n_tests, n_class and
#'   Model_type.
#' @export
fn_M_bytes_per_chain_counted_from_workspace_of_built_in_model <-  function( Model_type,
                                                                           N,
                                                                           n_tests
) {

        if (!is.character(Model_type) || (length(Model_type) != 1) ||
            !(Model_type %in% c("MVP", "MVOP", "LC_MVP", "LC_MVOP"))) {
              stop(paste0("fn_M_bytes_per_chain_counted_from_workspace_of_built_in_model: Model_type must be ",
                          "\"MVP\", \"MVOP\", \"LC_MVP\" or \"LC_MVOP\"; got: ",
                          paste(as.character(Model_type), collapse = ", ")))
        }
        n_class <-  if (Model_type %in% c("LC_MVP", "LC_MVOP")) 2 else 1
        ##
        ## ---- the doubles of the workspace per individual (as counted from the C++ above):
        ##
        if (Model_type %in% c("MVP", "LC_MVP")) {
              doubles_per_individual <-  31 * n_tests + 13 + n_class
        } else {
              doubles_per_individual <-  (36 + n_class) * n_tests + 15 + n_class
        }
        bytes_per_individual <-  8 * doubles_per_individual
        return(list( M_bytes_per_chain       = bytes_per_individual * N,
                     bytes_per_individual    = bytes_per_individual,
                     doubles_per_individual  = doubles_per_individual,
                     N                       = N,
                     n_tests                 = n_tests,
                     n_class                 = n_class,
                     Model_type              = Model_type))

}






















