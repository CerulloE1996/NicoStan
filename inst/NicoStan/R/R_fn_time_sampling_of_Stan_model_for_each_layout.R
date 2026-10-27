##
## ==== R_fn_time_sampling_of_Stan_model_for_each_layout.R =======================================================
##
## ---- Seconds per iteration of a Stan model (NicoStan's N / chunk_size data contract for reduce_sum_static) for
##      each sampling layout of chains, within-chain threads and chunks, in the sampler of the sampling phase -
##
## Each layout is timed in the sampler that $sample() runs for the sampling
## (Rcpp_fn_RcppParallel_EHMC_sampling()): every chain runs its iterations independently in one call, so that the
## call waits for the slowest chain once, as the sampling does - and not at every iteration, as the persistent
## burn-in worker of fn_time_burnin_iteration_for_each_num_chunks() does, which makes a layout with more chains
## look slower than it is in the sampling. The trajectory length is fixed (L_main leapfrog steps, randomize_tau =
## FALSE), at a tiny step size (eps = 1e-5), so that every layout does the same work. The seconds per iteration
## are the difference of a long call and a short call (n_iterations_of_short_call iterations) over the difference
## of their iterations, so that the setup of each call (the chains' workspaces) is not counted, as it is not in a
## sampling of hundreds of iterations. The seconds of each call are those of its chains alone (from the start of
## the chains to the end of the last chain), when the sampler returns them (sampling_call_timing), and otherwise
## its wall seconds, setup included (the difference then cancels the setup only as far as it is the same in both
## calls). The seconds setting caps each layout separately; only its shortest call may exceed it. Each
## layout's data are the Stan data with chunk_size = ceiling(N / N_chunks), as in
## fn_time_burnin_iteration_for_each_num_chunks(). The user-mode and kernel-mode CPU seconds of the two calls are
## returned too, and, when fn_snapshot_hardware_counters is given, it is called just before and just after each
## of them (e.g. to snapshot Paper 1's E4 counter tool pmc_stat, which counts user-mode cycles), so that the clock
## of the busy threads during the iterations is the difference of the two calls' user-mode cycles over the
## difference of their user-mode CPU seconds.
##
##
##
##
#### ---- the data files of each chunk size written by this R process ------------------------------------------
## (one entry per data text and chunk size: the JSON text the file was written from and the file's path in
##  tempdir(); the files are kept until R ends, so that every call of fn_time_sampling_of_Stan_model_for_each_
##  layout() in the process reads the same file of a chunk size and the Stan model kept between sampling calls
##  is used again)
environment_of_the_data_files_of_each_chunk_size_written_by_this_R_process <-  new.env(parent = emptyenv())





#### ---- the timing --------------------------------------------------------------------------------------------
#' Seconds per iteration of a Stan model in the sampler of the sampling phase, for each layout of chains,
#' within-chain threads and chunks
#'
#' @param init_object The model's initialisation object (it holds Model_args_as_Rcpp_List, with the model's
#'   compiled BridgeStan library, n_params_main and n_nuisance).
#' @param Stan_data_list The Stan data (with integer fields N and chunk_size).
#' @param layouts A data frame with columns n_chains (the sampling chains, run in parallel), n_threads_WCP
#'   (within-chain threads) and num_chunks.
#' @param minimum_seconds_per_timing The cap on each layout timed in one call; 0 keeps a fixed call.
#'   Its shortest call may exceed the cap when one iteration with one leapfrog step is already longer.
#' @param n_iterations_of_short_call The iterations of the short call (default 1).
#' @param time_layout_in_one_call TRUE (default): each layout is timed in ONE sampling call: each chain runs
#'   n_iterations_of_untimed_warm_up_call untimed iterations, then its timed iterations
#'   (n_iterations_of_long_call),
#'   timed by the chain itself in C++ (needs a NicoStan build that returns
#'   seconds_of_timed_iterations_of_each_chain); the layout's seconds per iteration are those of its slowest
#'   chain. FALSE: the untimed warm-up call, the short call and the long call (Paper 1's two-run difference).
#' @param n_iterations_of_long_call The iterations of the long call (default NULL: sized from the short call to
#'   last minimum_seconds_per_timing; given: used as they are, e.g. from the model's lp_grad time).
#' @param n_iterations_of_untimed_warm_up_call The iterations of an untimed call just before the short call, so
#'   that the short call is as warm as the long call, as in Paper 1's runners (default 1; 0: none).
#' @param L_main,eps_main The leapfrog steps (fixed) and step size (default 10 and 1e-5).
#' @param JSON_text_of_Stan_data_list The data's JSON text from fn_JSON_text_of_Stan_data_list(), so that each
#'   layout's data file only changes its chunk_size (default NULL: the whole data written for each layout).
#' @param fn_snapshot_hardware_counters A function without arguments, called just before and just after each of
#'   the two calls used, which returns the number of its snapshot (default NULL: none).
#' @param initial_values_of_the_chains A list with the matrices main (n_params_main rows) and nuisance
#'   (n_nuisance rows), one column per chain (at least as many as the layouts' chains; each layout takes its first
#'   n_chains columns), e.g. for a model whose log density is -Inf at some of the default starting values
#'   (default NULL: the small values 0.005 x (((1:n) x 7) mod 5 - 2), the same for every model).
#' @param nuisance_metric_of_the_timings "identity" (default: every nuisance variance 1 and every centre 0, for
#'   which the sampler takes its shorter per-unit path) or "diagonal" (the nuisance variances
#'   exp(0.5 sin(1:n)), between 0.61 and 1.65, and the centres 0.01 cos(1:n), the same for every model and
#'   layout and without random numbers, for which the sampler takes the full per-unit path of a fit after its
#'   burn-in, with every per-unit array of the diagonal metric read and written).
#' @param pass_the_data_to_the_sampler_as_JSON_text_instead_of_a_file TRUE: each layout's sampler is given the
#'   data's JSON text itself, with the layout's chunk_size (BridgeStan reads a string that does not end in .json
#'   as JSON), so that no data file is written per layout; FALSE (default): a data file per layout, as before.
#'   Needs JSON_text_of_Stan_data_list.
#' @return The data frame layouts with seconds_per_iteration, seconds_per_draw_over_all_chains (the seconds per
#'   iteration divided by the layout's chains), timed_by_chains_seconds_of_the_sampler (TRUE: from the seconds of
#'   the calls' chains alone; FALSE: from their wall seconds, setup included),
#'   seconds_per_iteration_from_wall_seconds_of_the_calls, n_iterations_of_long_call, seconds_spent_timing (the
#'   wall time of the layout's calls, setup included), user_CPU_seconds_of_short_call,
#'   kernel_CPU_seconds_of_short_call, user_CPU_seconds_of_long_call and kernel_CPU_seconds_of_long_call (of the R
#'   process and all its threads; the long call used, i.e. the one made again when it was), and
#'   counter_snapshot_before_short_call, counter_snapshot_after_short_call, counter_snapshot_before_long_call and
#'   counter_snapshot_after_long_call (NA without fn_snapshot_hardware_counters), and
#'   n_divergent_transitions_of_timed_calls and n_transitions_of_timed_calls (of the short call and the long call
#'   used, from the sampler's divergence trace: a timing with any divergent transition is invalid).
#' @export
fn_time_sampling_of_Stan_model_for_each_layout <-  function( init_object,
                                                            Stan_data_list,
                                                            layouts,
                                                            minimum_seconds_per_timing     = 2,
                                                            n_iterations_of_short_call     = 1,
##                                                            n_iterations_of_untimed_warm_up_call = 1,
                                                            n_iterations_of_untimed_warm_up_call = 0,
                                                            n_iterations_of_long_call      = NULL,
                                                            time_layout_in_one_call        = TRUE,
                                                            L_main                         = 10,
                                                            eps_main                       = 1e-05,
                                                            JSON_text_of_Stan_data_list    = NULL,
                                                            fn_snapshot_hardware_counters  = NULL,
                                                            initial_values_of_the_chains   = NULL,
                                                            nuisance_metric_of_the_timings = "identity",
                                                     pass_the_data_to_the_sampler_as_JSON_text_instead_of_a_file =
                                                                  FALSE,
                                                            ## (the leapfrog steps of each untimed iteration of a
                                                            ##  layout timed in one call; NULL: L_main, as the
                                                            L_of_the_untimed_iterations    = NULL) {

        minimum_seconds_argument_was_given <-  !missing(minimum_seconds_per_timing)
        seconds_cap <-  minimum_seconds_per_timing
        if (!minimum_seconds_argument_was_given && !is.null(n_iterations_of_long_call)) seconds_cap <-  0
        if (!is.finite(seconds_cap) || seconds_cap < 0) {
              stop("The timed-call cap must be finite and nonnegative.")
        }
        ##
        columns_of_layouts <-  c("n_chains", "n_threads_WCP", "num_chunks")
        ## (a gradient of the model's library runs in this R process: recorded, see
        ##  fn_a_gradient_of_the_library_ran_in_this_R_process())
        fn_record_that_a_gradient_of_the_library_ran_in_this_R_process(model_so_file = init_object$model_so_file)
        if (!is.data.frame(layouts) || !all(columns_of_layouts %in% names(layouts))) {
              stop(paste0("fn_time_sampling_of_Stan_model_for_each_layout: layouts must be a data frame with ",
                          "columns n_chains, n_threads_WCP and num_chunks."))
        }
        n_params_main <-  init_object$n_params_main
        n_nuisance <-  init_object$n_nuisance
        ## (the data's JSON text, with a chunk_size, written once here when it is not given: without it, every
        ##  call wrote a new data file, and the Stan model kept between sampling calls, found again by its data
        ##  file's path, size and time, was built again at every call)
        if (is.null(JSON_text_of_Stan_data_list)) {
              Stan_data_with_a_chunk_size <-  Stan_data_list
              if (is.null(Stan_data_with_a_chunk_size$chunk_size)) {
                    # Stan_data_with_a_chunk_size$chunk_size <-  as.integer(Stan_data_list$N)
                    Stan_data_with_a_chunk_size <-  fn_Stan_data_list_with_the_chunks_of_N_chunks(
                                                          Stan_data_list  = Stan_data_list,
                                                          N_chunks        = 1)
              }
              JSON_text_of_Stan_data_list <-  fn_JSON_text_of_Stan_data_list(
                                                    Stan_data_list = Stan_data_with_a_chunk_size)
        }
        ##
        ## ---- AUTO-CHUNKING's number of chunks, read from the data of a library built by
        ##      R_fn_prepare_auto_chunked_BridgeStan_build() (N_chunks_of_auto_chunking): with the layouts'
        ##      num_chunks alone, AUTO-CHUNKING takes num_chunks and the model's reduce_sum one chunk
        ##      (chunk_size = N); with a column num_chunks_of_auto_chunking too (both on), the model's reduce_sum
        ##      and AUTO-CHUNKING take the same num_chunks, one shared count, never two:
        ##
        library_is_auto_chunked <-  grepl(pattern = "_auto_chunked_model[.]so$", x = init_object$model_so_file)
        layouts_give_the_chunks_of_auto_chunking <-  "num_chunks_of_auto_chunking" %in% names(layouts)
        if (layouts_give_the_chunks_of_auto_chunking && !library_is_auto_chunked) {
              stop("num_chunks_of_auto_chunking is given for a library that is not auto-chunked: ",
                   init_object$model_so_file)
        }
        if (layouts_give_the_chunks_of_auto_chunking &&
            !isTRUE(all(layouts$num_chunks_of_auto_chunking == layouts$num_chunks))) {
              stop("with both on, reduce_sum and AUTO-CHUNKING take the same num_chunks: a layout gives two counts")
        }
        if (library_is_auto_chunked && !layouts_give_the_chunks_of_auto_chunking &&
            !isTRUE(all(layouts$n_threads_WCP == 1L))) {
              stop("AUTO-CHUNKING alone has no WCP: every layout has 1 thread per chain")
        }
        if (grepl(pattern = "\"N_chunks_of_auto_chunking\"", x = JSON_text_of_Stan_data_list, fixed = TRUE)) {
              stop("the Stan data already hold N_chunks_of_auto_chunking: it is set for each layout")
        }
        fn_chunks_of_the_layout <-  function(layout_index) {

              if (!library_is_auto_chunked) {
                    return(list(num_chunks                   = layouts$num_chunks[layout_index],
                                num_chunks_of_auto_chunking  = NA_integer_))
              }
              if (layouts_give_the_chunks_of_auto_chunking) {
                    return(list(num_chunks                   = layouts$num_chunks[layout_index],
                                num_chunks_of_auto_chunking  = layouts$num_chunks_of_auto_chunking[layout_index]))
              }
              return(list(num_chunks                   = 1L,
                          num_chunks_of_auto_chunking  = layouts$num_chunks[layout_index]))

        }
        ##
        ## ---- each layout's data file: the Stan data with chunk_size = ceiling(N / N_chunks). With the data's
        ##      JSON text given, the file of a chunk size is written ONCE in this R process and kept in tempdir()
        ##      (removed with it when R ends): a layout visited again, or the same chunk size at another number
        ##      of threads per chain, reads the same file, so that the Stan model kept between sampling calls
        ##      (the same path, size and modification time) is used again, and the data's text is not
        ##      substituted and written again for every layout (at N = 50,000 that substitution and write took
        ##      up to seconds per layout for the models with the largest data):
        ##
        fn_write_JSON_data_file_of_layout <-  function(num_chunks,
                                                       num_chunks_of_auto_chunking) {

              # chunk_size <-  as.integer(ceiling(Stan_data_list$N / num_chunks))
              chunk_size <-  fn_chunk_size_and_number_of_blocks_from_N_and_N_chunks(
                                   N         = Stan_data_list$N,
                                   N_chunks  = num_chunks)$chunk_size
              ## (an auto-chunked library's data also hold N_chunks_of_auto_chunking, first in the JSON text)
              fn_with_the_chunks_of_auto_chunking <-  function(JSON_text) {

                    if (is.na(num_chunks_of_auto_chunking)) return(JSON_text)
                    return(sub(pattern = "^\\s*\\{",
                               replacement = paste0("{\"N_chunks_of_auto_chunking\": ",
                                                    as.integer(num_chunks_of_auto_chunking), ", "),
                               x = JSON_text))

              }
              ## (the data's JSON text itself, with the layout's chunks, given to the sampler in place of a
              ##  file)
              if (isTRUE(pass_the_data_to_the_sampler_as_JSON_text_instead_of_a_file) &&
                  !is.null(JSON_text_of_Stan_data_list)) {
                    # return(fn_JSON_text_of_Stan_data_list_with_chunk_size(
                    #              JSON_text_of_Stan_data_list  = JSON_text_of_Stan_data_list,
                    #              chunk_size                   = chunk_size))
                    return(fn_with_the_chunks_of_auto_chunking(
                                 JSON_text = fn_JSON_text_of_Stan_data_list_with_the_chunks_of_N_chunks(
                                                   JSON_text_of_Stan_data_list  = JSON_text_of_Stan_data_list,
                                                   N                            = Stan_data_list$N,
                                                   N_chunks                     = num_chunks)))
              }
              if (is.null(JSON_text_of_Stan_data_list)) {
                    json_file_path <-  tempfile(pattern = paste0("sampling_chunks_", num_chunks, "_"),
                                                fileext = ".json")
                    # Stan_data_of_layout <-  Stan_data_list
                    # Stan_data_of_layout$chunk_size <-  chunk_size
                    Stan_data_of_layout <-  fn_Stan_data_list_with_the_chunks_of_N_chunks(
                                                  Stan_data_list  = Stan_data_list,
                                                  N_chunks        = num_chunks)
                    if (!is.na(num_chunks_of_auto_chunking)) {
                          Stan_data_of_layout$N_chunks_of_auto_chunking <-  as.integer(
                                                                                 num_chunks_of_auto_chunking)
                    }
                    cmdstanr::write_stan_json(data = Stan_data_of_layout, file = json_file_path)
                    return(normalizePath(json_file_path, mustWork = TRUE))
              }
              ## (the file of this data text and N_chunks, if this R process has written it already: the key is
              ##  the text's length in bytes and the N_chunks (two N_chunks with one chunk_size have the same
              ##  chunks, so the same data, kept once per N_chunks), and the text itself is compared, so that two
              ##  data sets of the same length never share a file)
              # key_of_data_file <-  paste0(nchar(JSON_text_of_Stan_data_list, type = "bytes"),
              #                             "_bytes_chunk_size_", chunk_size)
              key_of_data_file <-  paste0(nchar(JSON_text_of_Stan_data_list, type = "bytes"),
                                          "_bytes_N_chunks_", num_chunks,
                                          "_N_chunks_of_auto_chunking_", num_chunks_of_auto_chunking)
              data_file_kept <-  environment_of_the_data_files_of_each_chunk_size_written_by_this_R_process[[
                                       key_of_data_file]]
              if (!is.null(data_file_kept) &&
                  identical(data_file_kept$JSON_text_of_Stan_data_list, JSON_text_of_Stan_data_list) &&
                  file.exists(data_file_kept$json_file_path)) {
                    return(data_file_kept$json_file_path)
              }
              json_file_path <-  tempfile(pattern = paste0("sampling_data_chunk_size_", chunk_size, "_N_chunks_",
                                                           num_chunks, "_"),
                                          fileext = ".json")
              # JSON_text_of_layout <-  sub(pattern = "\"chunk_size\":[[:space:]]*[0-9]+",
              #                             replacement = paste0("\"chunk_size\": ", chunk_size),
              #                             x = JSON_text_of_Stan_data_list)
              JSON_text_of_the_chunks <-  fn_JSON_text_of_Stan_data_list_with_the_chunks_of_N_chunks(
                                                JSON_text_of_Stan_data_list  = JSON_text_of_Stan_data_list,
                                                N                            = Stan_data_list$N,
                                                N_chunks                     = num_chunks)
              JSON_text_of_layout <-  fn_with_the_chunks_of_auto_chunking(JSON_text = JSON_text_of_the_chunks)
              writeChar(object = JSON_text_of_layout, con = json_file_path, eos = NULL, useBytes = TRUE)
              json_file_path <-  normalizePath(json_file_path, mustWork = TRUE)
              assign(x = key_of_data_file,
                     value = list( JSON_text_of_Stan_data_list  = JSON_text_of_Stan_data_list,
                                   json_file_path               = json_file_path),
                     envir = environment_of_the_data_files_of_each_chunk_size_written_by_this_R_process)
              return(json_file_path)
        }

        fn_data_files_kept_for_this_R_process <-  function() {

              data_files_kept <-  as.list(
                    environment_of_the_data_files_of_each_chunk_size_written_by_this_R_process)
              return(vapply(X = data_files_kept, FUN = function(data_file_kept) data_file_kept$json_file_path,
                            FUN.VALUE = ""))
        }

        json_file_paths <-  character(0)
        ## on.exit(unlink(x = json_file_paths), add = TRUE)
        ## (only data files are removed: the data given as JSON text is not a file)
        ## on.exit(unlink(x = json_file_paths[grepl(pattern = "\\.json$", x = json_file_paths)]), add = TRUE)
        ## (only the data files written for this call alone are removed: the data given as JSON text is not a
        ##  file, and the file of each chunk size written from the JSON text is kept for the R process)
        on.exit(unlink(x = setdiff(json_file_paths[grepl(pattern = "\\.json$", x = json_file_paths)],
                                   fn_data_files_kept_for_this_R_process())), add = TRUE)
        ##
        ## ---- the sampler's arguments: diffusion HMC (kick-flow-kick), a fixed trajectory of L_main leapfrog
        ##      steps and a diagonal main metric:
        ##
        EHMC_args_as_Rcpp_List <-  init_EHMC_args_as_Rcpp_List( diffusion_HMC            = TRUE,
                                                                diffusion_HMC_integrator = "kick_flow_kick")
        EHMC_args_as_Rcpp_List$eps_main <-  eps_main
        EHMC_args_as_Rcpp_List$tau_main <-  L_main * eps_main
        EHMC_args_as_Rcpp_List$eps_us <-  eps_main
        EHMC_args_as_Rcpp_List$tau_us <-  L_main * eps_main
        EHMC_args_as_Rcpp_List$randomize_tau <-  FALSE
        EHMC_args_as_Rcpp_List$store_log_lik_trace <-  FALSE
        EHMC_Metric_as_Rcpp_List <-  init_EHMC_Metric_as_Rcpp_List( n_params_main     = n_params_main,
                                                                    n_nuisance        = n_nuisance,
                                                                    ##  a diagonal main metric, as Stan's
                                                                    metric_shape_main = "diag")
        stopifnot(nuisance_metric_of_the_timings %in% c("identity", "diagonal"))
        if (nuisance_metric_of_the_timings == "diagonal" && n_nuisance > 0) {

              variances_of_the_nuisance <-  exp(0.5 * sin(seq_len(n_nuisance)))
              EHMC_Metric_as_Rcpp_List$M_inv_us_vec <-  matrix(variances_of_the_nuisance, ncol = 1)
              EHMC_Metric_as_Rcpp_List$M_us_vec <-  matrix(1 / variances_of_the_nuisance, ncol = 1)
              EHMC_Metric_as_Rcpp_List$theta_hat_us_vec <-  matrix(0.01 * cos(seq_len(n_nuisance)), ncol = 1)
        }

        fn_small_values <-  function(n_values) 0.005 * (((seq_len(n_values) * 7) %% 5) - 2)
        ##
        layouts$seconds_per_iteration <-  NA_real_
        layouts$seconds_per_draw_over_all_chains <-  NA_real_
        layouts$timed_by_chains_seconds_of_the_sampler <-  NA
        layouts$largest_seconds_after_the_timed_iterations_of_a_chain <-  NA_real_
        layouts$seconds_per_iteration_from_wall_seconds_of_the_calls <-  NA_real_
        layouts$n_iterations_of_long_call <-  NA_real_
        layouts$seconds_spent_timing <-  NA_real_
        layouts$user_CPU_seconds_of_short_call <-  NA_real_
        layouts$kernel_CPU_seconds_of_short_call <-  NA_real_
        layouts$user_CPU_seconds_of_long_call <-  NA_real_
        layouts$kernel_CPU_seconds_of_long_call <-  NA_real_
        layouts$counter_snapshot_before_short_call <-  NA_real_
        layouts$counter_snapshot_after_short_call <-  NA_real_
        layouts$counter_snapshot_before_long_call <-  NA_real_
        layouts$counter_snapshot_after_long_call <-  NA_real_
        layouts$n_divergent_transitions_of_timed_calls <-  NA_real_
        layouts$n_transitions_of_timed_calls <-  NA_real_
        layouts$n_timed_sampling_attempts <-  NA_real_
        layouts$total_seconds_of_timed_sampling_attempts <-  NA_real_
        layouts$total_wall_seconds_of_sampling_attempts <-  NA_real_
        layouts$total_user_CPU_seconds_of_sampling_attempts <-  NA_real_
        layouts$total_kernel_CPU_seconds_of_sampling_attempts <-  NA_real_
        layouts$sampling_timing_attempts_URL_encoded_JSON <-  NA_character_
        ##
        ## ---- one call, with its wall seconds, the seconds of its chains alone (when the sampler returns them),
        ##      its user-mode and kernel-mode CPU seconds and the counter snapshots just before and just after it
        ##      (outside its wall seconds):
        ##
        fn_snapshot_or_NA <-  function() {

              if (is.null(fn_snapshot_hardware_counters)) return(NA_real_)
              return(fn_snapshot_hardware_counters())
        }

        fn_timed_call <-  function(fn_call) {

              snapshot_before <-  fn_snapshot_or_NA()
              process_time_before <-  proc.time()
              result_of_call <-  fn_call()
              process_time_after <-  proc.time()
              snapshot_after <-  fn_snapshot_or_NA()
              chains_seconds <-  NA_real_
              seconds_of_timed_iterations_of_each_chain <-  NULL
              seconds_after_the_timed_iterations_of_each_chain <-  NULL
              ## (an untimed L asked for must have been used: a NicoStan build without it ignores the setting)
              if (!is.null(L_of_the_untimed_iterations) &&
                  !identical(as.integer(result_of_call$sampling_call_timing$
                                              L_of_the_untimed_iterations_before_timed_iterations),
                             as.integer(L_of_the_untimed_iterations))) {
                    stop(paste0("L_of_the_untimed_iterations = ", L_of_the_untimed_iterations,
                                " was not used by the sampler: the installed NicoStan does not have it; ",
                                "install it again."))
              }
              if (!is.null(result_of_call$sampling_call_timing)) {
                    chains_seconds <-  result_of_call$sampling_call_timing$chains_seconds
                    seconds_of_timed_iterations_of_each_chain <-
                          result_of_call$sampling_call_timing$seconds_of_timed_iterations_of_each_chain
                    ## (each chain's seconds after its timed iterations, the destruction of its model: never
                    ##  in the timed seconds)
                    seconds_after_the_timed_iterations_of_each_chain <-
                          result_of_call$sampling_call_timing$seconds_after_the_timed_iterations_of_each_chain
              }
              ## (the divergent transitions of the call, from the sampler's divergence trace, its second element:
              ##  a timing with any divergent transition is invalid, as a divergent trajectory stops early)
              n_divergent_transitions <-  NA_real_
              n_transitions <-  NA_real_
              if (length(result_of_call) >= 2) {
                    n_divergent_transitions <-  sum(unlist(result_of_call[[2]]))
                    n_transitions <-  length(unlist(result_of_call[[2]]))
              }
              rm(result_of_call)
              ## (as Paper 1's burn-in timings, R_fn_run_ps_1_burnin_benchmark.R: eps is tiny and L fixed, so
              ##  nothing should diverge; a divergent transition makes the configuration invalid)
              if (isTRUE(n_divergent_transitions > 0)) {
                    stop("Sampling timing encountered ", n_divergent_transitions,
                         " divergent transition(s) in a timed call. The configuration is invalid for a timing ",
                         "comparison.")
              }
              return(list( seconds              = process_time_after[["elapsed"]] -
                                                  process_time_before[["elapsed"]],
                           chains_seconds       = chains_seconds,
                           user_CPU_seconds     = process_time_after[["user.self"]] -
                                                  process_time_before[["user.self"]],
                           kernel_CPU_seconds   = process_time_after[["sys.self"]] -
                                                  process_time_before[["sys.self"]],
                           snapshot_before      = snapshot_before,
                           snapshot_after       = snapshot_after,
                           n_divergent_transitions  = n_divergent_transitions,
                           n_transitions            = n_transitions,
                           seconds_of_timed_iterations_of_each_chain = seconds_of_timed_iterations_of_each_chain,
                           seconds_after_the_timed_iterations_of_each_chain =
                                 seconds_after_the_timed_iterations_of_each_chain))
        }

        ## (the seconds per iteration of the previous layout timed in one call: the size of the next one)
        seconds_per_iteration_of_previous_layout <-  NA_real_
        for (layout_index in seq_len(nrow(layouts))) {
              ##
              time_layout_start <-  proc.time()[["elapsed"]]
              n_chains <-  layouts$n_chains[layout_index]
              n_threads_WCP <-  layouts$n_threads_WCP[layout_index]
              chunks_of_the_layout <-  fn_chunks_of_the_layout(layout_index = layout_index)
              if (library_is_auto_chunked) {
                    message(paste0("layout ", layout_index, ": ", n_chains, " chains x ", n_threads_WCP,
                                   " threads; chunks of the model's reduce_sum ", chunks_of_the_layout$num_chunks,
                                   ", of AUTO-CHUNKING ", chunks_of_the_layout$num_chunks_of_auto_chunking))
              }
              json_file_path <-  fn_write_JSON_data_file_of_layout(
                                       num_chunks                   = chunks_of_the_layout$num_chunks,
                                       num_chunks_of_auto_chunking  =
                                             chunks_of_the_layout$num_chunks_of_auto_chunking)
              json_file_paths <-  c(json_file_paths, json_file_path)
              Model_args_as_Rcpp_List <-  init_object$Model_args_as_Rcpp_List
              Model_args_as_Rcpp_List$json_file_path <-  json_file_path
              ## (one column per chain also for a model without nuisance parameters, i.e. a 0 x n_chains matrix;
              ##  the starting values given are used as they are when they have exactly n_chains columns, so
              ##  that no copy of the n_nuisance x n_chains matrix is made per layout, and the small default
              ##  values are only made when no starting values are given)
              if (is.null(initial_values_of_the_chains)) {
                    initial_main <-  matrix(fn_small_values(n_params_main * n_chains), nrow = n_params_main,
                                            ncol = n_chains)
                    initial_nuisance <-  matrix(fn_small_values(n_nuisance * n_chains), nrow = n_nuisance,
                                                ncol = n_chains)
              } else {
                    stopifnot(nrow(initial_values_of_the_chains$main) == n_params_main,
                              nrow(initial_values_of_the_chains$nuisance) == n_nuisance,
                              ncol(initial_values_of_the_chains$main) >= n_chains,
                              ncol(initial_values_of_the_chains$nuisance) >= n_chains)
                    if (ncol(initial_values_of_the_chains$main) == n_chains) {
                          initial_main <-  initial_values_of_the_chains$main
                    } else {
                          initial_main <-  initial_values_of_the_chains$main[, seq_len(n_chains), drop = FALSE]
                    }
                    if (ncol(initial_values_of_the_chains$nuisance) == n_chains) {
                          initial_nuisance <-  initial_values_of_the_chains$nuisance
                    } else {
                          initial_nuisance <-  initial_values_of_the_chains$nuisance[, seq_len(n_chains),
                                                                                     drop = FALSE]
                    }
              }
              RcppParallel::setThreadOptions(numThreads = n_chains * n_threads_WCP)
              fn_sampling_call <-  function( n_iterations,
                                             seed_of_call) {

                    Rcpp_fn_RcppParallel_EHMC_sampling(
                          n_threads_R                                 = n_chains,
                          seed_R                                      = seed_of_call,
                          n_iter_R                                    = n_iterations,
                          iter_one_by_one                             = FALSE,
                          partitioned_HMC_R                           = FALSE,
                          diffusion_HMC_R                             = TRUE,
                          Model_type_R                                = "Stan",
                          sample_nuisance_R                           = TRUE,
                          force_autodiff_R                            = TRUE,
                          force_PartialLog_R                          = FALSE,
                          multi_attempts_R                            = FALSE,
                          n_nuisance_to_track                         = 0,
                          theta_main_vectors_all_chains_input_from_R  = initial_main,
                          theta_us_vectors_all_chains_input_from_R    = initial_nuisance,
                          y_Eigen_R                                   = matrix(0, nrow = 1, ncol = 1),
                          Model_args_as_Rcpp_List                     = Model_args_as_Rcpp_List,
                          EHMC_args_as_Rcpp_List                      = EHMC_args_as_Rcpp_List,
                          EHMC_Metric_as_Rcpp_List                    = EHMC_Metric_as_Rcpp_List,
                          use_disk                                    = FALSE,
                          trace_dir                                   = tempdir(),
                          n_threads_WCP                               = n_threads_WCP)
              }

              ##
              ## ---- the short call, then the long call, each with its setup:
              ##
              ## time_before_short_call <-  proc.time()[["elapsed"]]
              ## invisible(fn_sampling_call(n_iterations_of_short_call, 1))
              ## seconds_of_short_call <-  proc.time()[["elapsed"]] - time_before_short_call
              ## ---- the layout timed in one call: each chain's untimed iterations, then its timed iterations,
              ##      timed by the chain itself (no warm-up call, no short call, so no start-up is paid twice):
              ##
              if (isTRUE(time_layout_in_one_call)) {
                    # n_untimed_iterations <-  max(1, n_iterations_of_untimed_warm_up_call)
                    ## (0 allowed: the timed window then starts at the chain's first iteration, so that the
                    ##  check of the timing method can measure what the untimed iteration absorbs)
                    n_untimed_iterations <-  max(0, n_iterations_of_untimed_warm_up_call)
                    ##
                    ## ---- Each layout starts with its shortest call -----------------------------------------------------------------
                    ##
                    L_of_this_layout <-  L_main
                    predicted_seconds_of_call <-  NA_real_
                    seconds_per_iteration_measured <-  NA_real_
                    sizing_the_layout <-  seconds_cap > 0
                    if (sizing_the_layout) {
                          n_timed_iterations <-  1L
                          L_of_this_layout <-  1L
                    } else {
                          n_timed_iterations <-  n_iterations_of_long_call
                          if (is.null(n_timed_iterations)) n_timed_iterations <-  1L
                    }
                    EHMC_args_of_one_call <-  EHMC_args_as_Rcpp_List
                    EHMC_args_of_one_call$n_untimed_iterations_before_timed_iterations <-
                          as.integer(n_untimed_iterations)
                    ## (no chain of a timing call hands its memory pages back to the system: a finished chain's
                    ##  release would interrupt the chains still being timed)
                    EHMC_args_of_one_call$release_memory_of_the_Stan_model_after_sampling <-  0
                    if (!is.null(L_of_the_untimed_iterations)) {
                          EHMC_args_of_one_call$L_of_the_untimed_iterations_before_timed_iterations <-
                                as.integer(L_of_the_untimed_iterations)
                    }
                    EHMC_args_as_Rcpp_List_of_all_calls <-  EHMC_args_as_Rcpp_List
                    sampling_timing_attempts <-  list()
                    lowering_L_between_straddling_counts <-  FALSE
                    longest_call_under_cap <-  NULL
                    use_longest_call_under_cap <-  FALSE
                    ##
                    ## ---- Measure and resize each layout -----------------------------------------------------------------------------
                    ##
                    repeat {
                          EHMC_args_of_one_call$tau_main <-  L_of_this_layout * eps_main
                          EHMC_args_of_one_call$tau_us <-  L_of_this_layout * eps_main
                          EHMC_args_as_Rcpp_List <-  EHMC_args_of_one_call
                          one_call <-  fn_timed_call(function() fn_sampling_call(n_untimed_iterations +
                                                                                n_timed_iterations, 1))
                          EHMC_args_as_Rcpp_List <-  EHMC_args_as_Rcpp_List_of_all_calls
                          seconds_of_timed_iterations_of_each_chain <-
                                one_call$seconds_of_timed_iterations_of_each_chain
                          if (is.null(seconds_of_timed_iterations_of_each_chain) ||
                              length(seconds_of_timed_iterations_of_each_chain) == 0 ||
                              !all(is.finite(seconds_of_timed_iterations_of_each_chain)) ||
                              max(seconds_of_timed_iterations_of_each_chain) <= 0) {

                                stop("The sampler did not return finite positive timed sampling seconds.")
                          }

                          seconds_of_timed_iterations <-  max(seconds_of_timed_iterations_of_each_chain)
                          attempt <-  data.frame(
                                attempt_index             = length(sampling_timing_attempts) + 1,
                                n_timed_iterations        = n_timed_iterations,
                                L                         = L_of_this_layout,
                                seconds_cap               = seconds_cap,
                                band_lower_seconds        = max(0, seconds_cap - 0.1),
                                predicted_seconds         = predicted_seconds_of_call,
                                shortest_call             = n_timed_iterations == 1 && L_of_this_layout == 1,
                                in_band                   = seconds_of_timed_iterations <= seconds_cap &&
                                      seconds_of_timed_iterations >= max(0, seconds_cap - 0.1),
                                seconds_of_timed_iterations = seconds_of_timed_iterations,
                                wall_seconds              = one_call$seconds,
                                user_CPU_seconds          = one_call$user_CPU_seconds,
                                kernel_CPU_seconds        = one_call$kernel_CPU_seconds,
                                counter_snapshot_before   = one_call$snapshot_before,
                                counter_snapshot_after    = one_call$snapshot_after,
                                n_divergent_transitions    = one_call$n_divergent_transitions,
                                n_transitions             = one_call$n_transitions,
                                seconds_setting           = minimum_seconds_per_timing)
                          attempt$seconds_of_timed_iterations_of_each_chain <-
                                I(list(seconds_of_timed_iterations_of_each_chain))
                          sampling_timing_attempts[[length(sampling_timing_attempts) + 1]] <-  attempt
                          if (seconds_cap <= 0) break
                          ##
                          ## ---- Retain the longest measured call below the cap ---------------------------------------------------------
                          ##
                          if (seconds_of_timed_iterations <= seconds_cap &&
                              (is.null(longest_call_under_cap) ||
                               seconds_of_timed_iterations > longest_call_under_cap$seconds)) {

                                longest_call_under_cap <-  list(
                                      seconds = seconds_of_timed_iterations,
                                      one_call = one_call,
                                      iterations = n_timed_iterations,
                                      L = L_of_this_layout,
                                      predicted = predicted_seconds_of_call)

                          }
                          ##
                          ## ---- Bound noisy resizing without interrupting a sampling call ------------------------------------------------
                          ##
                          if (length(sampling_timing_attempts) >= 2 * L_main + 4 &&
                              !is.null(longest_call_under_cap)) {

                                use_longest_call_under_cap <-  TRUE
                                break

                          }
                          seconds_per_iteration_measured <-  seconds_of_timed_iterations / n_timed_iterations
                          if (sizing_the_layout) {
                                sizing_the_layout <-  FALSE
                                if (seconds_of_timed_iterations > seconds_cap) break
                                L_of_this_layout <-  max(1, min(L_main,
                                      floor(seconds_cap / seconds_per_iteration_measured)))
                                n_timed_iterations <-  max(1, floor(seconds_cap /
                                      (L_of_this_layout * seconds_per_iteration_measured)))
                                predicted_seconds_of_call <-  n_timed_iterations * L_of_this_layout *
                                      seconds_per_iteration_measured
                                if (n_timed_iterations == 1 && L_of_this_layout == 1 &&
                                    seconds_of_timed_iterations < max(0, seconds_cap - 0.1)) {
                                      n_timed_iterations <-  2L
                                }
                                next
                          }
                          if (seconds_of_timed_iterations <= seconds_cap) {
                                if (seconds_of_timed_iterations >= max(0, seconds_cap - 0.1)) break
                                if (lowering_L_between_straddling_counts) {
                                      use_longest_call_under_cap <-  TRUE
                                      break
                                }
                                seconds_per_step_measured <-  seconds_per_iteration_measured / L_of_this_layout
                                L_next <-  min(L_main, max(1, floor(seconds_cap / seconds_per_step_measured)))
                                iterations_next <-  max(1, floor(seconds_cap /
                                      (L_next * seconds_per_step_measured)))
                                tried_next <-  vapply(sampling_timing_attempts, function(previous) {
                                      previous$L == L_next && previous$n_timed_iterations == iterations_next
                                }, logical(1))
                                ##
                                ## ---- Adjacent iteration counts straddle the band -------------------------------------------------------
                                ##
                                if (any(tried_next)) {

                                      if (lowering_L_between_straddling_counts) {
                                            use_longest_call_under_cap <-  TRUE
                                            break
                                      }
                                      larger_count <-  n_timed_iterations + 1L
                                      over_at_this_L <-  vapply(sampling_timing_attempts, function(previous) {

                                            previous$L == L_of_this_layout &&
                                                  previous$n_timed_iterations == larger_count &&
                                                  previous$seconds_of_timed_iterations > seconds_cap

                                      }, logical(1))
                                      n_timed_iterations <-  larger_count
                                      lowering_L_between_straddling_counts <-  TRUE
                                      if (any(over_at_this_L) && L_of_this_layout > 1) {
                                            L_of_this_layout <-  L_of_this_layout - 1L
                                      }
                                      predicted_seconds_of_call <-  n_timed_iterations * L_of_this_layout *
                                            seconds_per_step_measured
                                      next

                                }
                                L_of_this_layout <-  L_next
                                n_timed_iterations <-  iterations_next
                                predicted_seconds_of_call <-  n_timed_iterations * L_of_this_layout *
                                      seconds_per_step_measured
                                next
                          }
                          ##
                          ## ---- Keep the larger count while closing a measured gap -----------------------------------------------------
                          ##
                          if (lowering_L_between_straddling_counts) {

                                if (L_of_this_layout <= 1) {
                                      use_longest_call_under_cap <-  TRUE
                                      break
                                }
                                L_of_this_layout <-  L_of_this_layout - 1L
                                predicted_seconds_of_call <-  n_timed_iterations * seconds_per_iteration_measured *
                                      L_of_this_layout / attempt$L
                                next

                          }
                          if (n_timed_iterations > 1) {
                                n_timed_iterations <-  max(1, min(n_timed_iterations - 1,
                                      floor(n_timed_iterations * seconds_cap / seconds_of_timed_iterations)))
                          } else if (L_of_this_layout > 1) {
                                L_of_this_layout <-  max(1, min(L_of_this_layout - 1,
                                      floor(L_of_this_layout * seconds_cap / seconds_of_timed_iterations)))
                          } else {
                                break
                          }
                          predicted_seconds_of_call <-  n_timed_iterations * seconds_per_iteration_measured *
                                L_of_this_layout / attempt$L

                    }
                    ##
                    ## ---- Keep a measured under-cap call when whole counts miss the band -------------------------------------------------
                    ##
                    if (use_longest_call_under_cap) {

                          one_call <-  longest_call_under_cap$one_call
                          n_timed_iterations <-  longest_call_under_cap$iterations
                          L_of_this_layout <-  longest_call_under_cap$L
                          predicted_seconds_of_call <-  longest_call_under_cap$predicted
                          seconds_of_timed_iterations <-  longest_call_under_cap$seconds
                          seconds_of_timed_iterations_of_each_chain <-
                                one_call$seconds_of_timed_iterations_of_each_chain

                    }
                    sampling_timing_attempts <-  do.call(rbind, sampling_timing_attempts)
                    layouts$n_timed_sampling_attempts[layout_index] <-  nrow(sampling_timing_attempts)
                    layouts$total_seconds_of_timed_sampling_attempts[layout_index] <-
                          sum(sampling_timing_attempts$seconds_of_timed_iterations)
                    layouts$total_wall_seconds_of_sampling_attempts[layout_index] <-
                          sum(sampling_timing_attempts$wall_seconds)
                    layouts$total_user_CPU_seconds_of_sampling_attempts[layout_index] <-
                          sum(sampling_timing_attempts$user_CPU_seconds)
                    layouts$total_kernel_CPU_seconds_of_sampling_attempts[layout_index] <-
                          sum(sampling_timing_attempts$kernel_CPU_seconds)
                    layouts$sampling_timing_attempts_URL_encoded_JSON[layout_index] <-  utils::URLencode(
                          URL = as.character(jsonlite::toJSON(
                                x = sampling_timing_attempts, dataframe = "rows", digits = NA, na = "null")),
                          reserved = TRUE)
                    seconds_per_iteration <-  max(seconds_of_timed_iterations_of_each_chain) / n_timed_iterations
                    seconds_per_iteration_of_previous_layout <-  seconds_per_iteration
                    layouts$L_of_the_timed_call[layout_index] <-  L_of_this_layout
                    layouts$seconds_cap[layout_index] <-  seconds_cap
                    layouts$predicted_seconds_of_timed_call[layout_index] <-  predicted_seconds_of_call
                    layouts$actual_seconds_of_timed_call[layout_index] <-  seconds_of_timed_iterations
                    layouts$in_band[layout_index] <-  seconds_cap > 0 &&
                          seconds_of_timed_iterations >= max(0, seconds_cap - 0.1) &&
                          seconds_of_timed_iterations <= seconds_cap
                    layouts$used_longest_call_under_cap[layout_index] <-  use_longest_call_under_cap
                    layouts$shortest_call_exceeds_cap[layout_index] <-  seconds_cap > 0 &&
                          seconds_of_timed_iterations > seconds_cap && n_timed_iterations == 1 &&
                          L_of_this_layout == 1
                    layouts$seconds_per_iteration_at_requested_L[layout_index] <-
                          seconds_per_iteration * L_main / L_of_this_layout
                    layouts$seconds_per_iteration[layout_index] <-  seconds_per_iteration
                    layouts$seconds_per_draw_over_all_chains[layout_index] <-  seconds_per_iteration / n_chains
                    layouts$timed_by_chains_seconds_of_the_sampler[layout_index] <-  TRUE
                    if (!is.null(one_call$seconds_after_the_timed_iterations_of_each_chain)) {
                          layouts$largest_seconds_after_the_timed_iterations_of_a_chain[layout_index] <-
                                max(one_call$seconds_after_the_timed_iterations_of_each_chain)
                    }
                    layouts$n_iterations_of_long_call[layout_index] <-  n_timed_iterations
                    layouts$seconds_spent_timing[layout_index] <-  proc.time()[["elapsed"]] - time_layout_start
                    layouts$n_divergent_transitions_of_timed_calls[layout_index] <-
                          one_call$n_divergent_transitions
                    layouts$n_transitions_of_timed_calls[layout_index] <-  one_call$n_transitions
                    ## (the counter snapshots just before and just after the one call, in the long call's
                    ##  columns; they also count its n_untimed_iterations untimed iterations, i.e. the counters
                    ##  per iteration are their differences over n_untimed_iterations + n_iterations_of_long_call)
                    layouts$counter_snapshot_before_long_call[layout_index] <-  one_call$snapshot_before
                    layouts$counter_snapshot_after_long_call[layout_index] <-  one_call$snapshot_after
                    layouts$user_CPU_seconds_of_long_call[layout_index] <-  one_call$user_CPU_seconds
                    layouts$kernel_CPU_seconds_of_long_call[layout_index] <-  one_call$kernel_CPU_seconds
                    next
              }
              ## (Paper 1's runners: one untimed warm-up call before every timed pair, so that the short call is
              ##  not colder than the long call)
              if (n_iterations_of_untimed_warm_up_call > 0) {
                    invisible(fn_sampling_call(n_iterations_of_untimed_warm_up_call, 1))
              }
              short_call <-  fn_timed_call(function() fn_sampling_call(n_iterations_of_short_call, 1))
              ## (the seconds of the calls' chains alone when the sampler returns them, otherwise their wall
              ##  seconds)
              timed_by_chains_seconds <-  !is.na(short_call$chains_seconds)
              fn_seconds_of_call <-  function(call) {

                    if (timed_by_chains_seconds) return(call$chains_seconds)
                    return(call$seconds)
              }

              seconds_of_short_call <-  fn_seconds_of_call(short_call)
              seconds_per_iteration_at_most <-  seconds_of_short_call / n_iterations_of_short_call
              n_iterations_of_long_call_given <-  !is.null(n_iterations_of_long_call)
              if (!n_iterations_of_long_call_given) {
                    n_iterations_of_long_call <-  n_iterations_of_short_call +
                                                  max(2, ceiling(minimum_seconds_per_timing /
                                                                 seconds_per_iteration_at_most))
              }
              ## time_before_long_call <-  proc.time()[["elapsed"]]
              ## invisible(fn_sampling_call(n_iterations_of_long_call, 2))
              ## seconds_of_long_call <-  proc.time()[["elapsed"]] - time_before_long_call
              long_call <-  fn_timed_call(function() fn_sampling_call(n_iterations_of_long_call, 2))
              seconds_of_long_call <-  fn_seconds_of_call(long_call)
              ##
              ## ---- the seconds per iteration: the difference of the two calls over the difference of their
              ##      iterations (the long call's seconds per iteration if the difference is not positive):
              ##
              seconds_per_iteration <-  (seconds_of_long_call - seconds_of_short_call) /
                                        (n_iterations_of_long_call - n_iterations_of_short_call)
              ##
              ## ---- (the short call's seconds include the setup, so that its estimate of the seconds per
              ##       iteration is too high and the long call can have fewer than minimum_seconds_per_timing
              ##       seconds of iterations: then the long call is made again, sized with the seconds per
              ##       iteration found)
              ##
              seconds_of_iterations_of_long_call <-  seconds_per_iteration *
                                                     (n_iterations_of_long_call - n_iterations_of_short_call)
              long_call_too_short <-  seconds_of_iterations_of_long_call < minimum_seconds_per_timing
              if ((seconds_per_iteration > 0) && long_call_too_short && !n_iterations_of_long_call_given) {

                    n_iterations_of_long_call <-  n_iterations_of_short_call +
                                                  max(2, ceiling(minimum_seconds_per_timing /
                                                                 seconds_per_iteration))
                    ## time_before_long_call <-  proc.time()[["elapsed"]]
                    ## invisible(fn_sampling_call(n_iterations_of_long_call, 3))
                    ## seconds_of_long_call <-  proc.time()[["elapsed"]] - time_before_long_call
                    long_call <-  fn_timed_call(function() fn_sampling_call(n_iterations_of_long_call, 3))
                    seconds_of_long_call <-  fn_seconds_of_call(long_call)
                    seconds_per_iteration <-  (seconds_of_long_call - seconds_of_short_call) /
                                              (n_iterations_of_long_call - n_iterations_of_short_call)
              }

              if (!(seconds_per_iteration > 0)) {

                    seconds_per_iteration <-  seconds_of_long_call / n_iterations_of_long_call
              }

              layouts$seconds_per_iteration[layout_index] <-  seconds_per_iteration
              layouts$seconds_per_draw_over_all_chains[layout_index] <-  seconds_per_iteration / n_chains
              layouts$timed_by_chains_seconds_of_the_sampler[layout_index] <-  timed_by_chains_seconds
              layouts$seconds_per_iteration_from_wall_seconds_of_the_calls[layout_index] <-
                    (long_call$seconds - short_call$seconds) /
                    (n_iterations_of_long_call - n_iterations_of_short_call)
              layouts$n_iterations_of_long_call[layout_index] <-  n_iterations_of_long_call
              layouts$seconds_spent_timing[layout_index] <-  proc.time()[["elapsed"]] - time_layout_start
              layouts$user_CPU_seconds_of_short_call[layout_index] <-  short_call$user_CPU_seconds
              layouts$kernel_CPU_seconds_of_short_call[layout_index] <-  short_call$kernel_CPU_seconds
              layouts$user_CPU_seconds_of_long_call[layout_index] <-  long_call$user_CPU_seconds
              layouts$kernel_CPU_seconds_of_long_call[layout_index] <-  long_call$kernel_CPU_seconds
              layouts$counter_snapshot_before_short_call[layout_index] <-  short_call$snapshot_before
              layouts$counter_snapshot_after_short_call[layout_index] <-  short_call$snapshot_after
              layouts$counter_snapshot_before_long_call[layout_index] <-  long_call$snapshot_before
              layouts$counter_snapshot_after_long_call[layout_index] <-  long_call$snapshot_after
              ## (the short call and the long call used: any divergent transition makes the timing invalid)
              layouts$n_divergent_transitions_of_timed_calls[layout_index] <-
                    short_call$n_divergent_transitions + long_call$n_divergent_transitions
              layouts$n_transitions_of_timed_calls[layout_index] <-
                    short_call$n_transitions + long_call$n_transitions
              ## invisible(gc(verbose = FALSE))
              ## (no gc() after each layout: it cost about 0.08 s per layout, at N = 25,000)
        ##
        }
        return(layouts)
##
}






















