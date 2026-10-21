##
## ==== R_fn_compute_sampling_num_chunks_for_Stan_model.R =======================================================
##
## ---- Sampling chunks (num_chunks_sampling) of a Stan model whose likelihood is summed by reduce_sum_static:
##
##      Paper 1's rule from the model's own measured M_bytes/chain, confirmed by timing ----
##
## For one compiled model (each math backend, e.g. Stan math or the AVX2 / AVX-512 functions, is its own .so, so
## each gets its own memory and its own chunks):
##
##   1. M_bytes/chain is measured on the full data evaluated as one piece
##      (fn_estimate_initial_M_bytes_per_chain(), as for the burn-in), unless it is given;
##   2. Paper 1's rule gives the chunks for the sampling layout and the CPUs of this process
##      (fn_compute_sampling_num_chunks_for_CPUs_of_this_process(), i.e. fn_compute_initial_N_chunks()), rounded
##      up to a power of two, as reduce_sum_static cuts the data into a power of two of pieces;
##   3. the rule's chunk count and the powers of two around it (1/4, 1/2, 2 and 4 times it, and 1, i.e. no
##      chunking) are timed in NicoStan's own worker with the SAMPLING layout: all n_chains_sampling chains
##      advanced one iteration at once, n_threads_WCP_sampling threads each
##      (fn_time_burnin_iteration_for_each_num_chunks(), the timing that Paper 1's burn-in benchmark and the
##      burn-in choice use), and the fastest is used.
##
## The result says whether the rule's chunk count is the fastest or within fraction_slower_counted_as_tie (default
## 1%) of the fastest, and lists every chunk count within that fraction (near-ties).
##
## The choice is saved in NicoStan's cache folder and reused, without measuring or timing again, only for the same
## model file, N, number of sampling chains, within-chain threads and CPUs. The seconds the choice took are
## returned (seconds_spent_choosing; 0 when reused) and are not part of a fit's burn-in or sampling time.
##
## This function stands alone (it can be called before any fit, e.g. for each math backend of a benchmark), and
## $sample() calls it to choose num_chunks_sampling of a Stan model when num_chunks_sampling is not given and
## choose_threads_and_chunks_not_given_with_Paper_1_functions = TRUE.
##
##
##
##
#### ---- the sampling choice saved for reuse: same model, N, sampling chains, threads and CPUs ------------------
#' The file of NicoStan's saved sampling choices (num_chunks_sampling)
#'
#' One row per model file, N, number of sampling chains, within-chain threads and CPU list, in NicoStan's cache
#' folder (tools::R_user_dir("NicoStan", "cache")).
#' @return The path of the file.
#' @keywords internal
fn_saved_sampling_choices_file <-  function() {
        ##
        return(file.path(tools::R_user_dir(package = "NicoStan", which = "cache"),
                         "sampling_num_chunks_chosen_per_model_N_chains_threads_and_CPUs.rds"))
##
}
##
#' The saved sampling choice for one model file, N, sampling chains, threads and CPU list (NULL if there is none)
#' @param model_so_file Path of the compiled BridgeStan model library (.so).
#' @param N_units The N of the Stan data.
#' @param n_chains_sampling The number of sampling chains.
#' @param n_threads_WCP_sampling The within-chain threads of each sampling chain.
#' @param CPU_list The CPUs available, as fn_format_CPU_list() writes them.
#' @return The saved row, or NULL.
#' @keywords internal
fn_read_saved_sampling_choice <-  function( model_so_file,
                                            N_units,
                                            n_chains_sampling,
                                            n_threads_WCP_sampling,
                                            CPU_list
) {
        ##
        saved_choices_file <-  fn_saved_sampling_choices_file()
        if (!file.exists(saved_choices_file)) return(NULL)
        saved_choices <-  tryCatch(expr = readRDS(file = saved_choices_file), error = function(condition) NULL)
        if (!is.data.frame(saved_choices) || nrow(saved_choices) == 0) return(NULL)
        same_model_N_chains_threads_and_CPUs <-  saved_choices$model_so_file == normalizePath(model_so_file) &
                                                 saved_choices$N_units == N_units &
                                                 saved_choices$n_chains_sampling == n_chains_sampling &
                                                 saved_choices$n_threads_WCP_sampling == n_threads_WCP_sampling &
                                                 saved_choices$CPU_list == CPU_list
        if (!any(same_model_N_chains_threads_and_CPUs)) return(NULL)
        return(saved_choices[which(same_model_N_chains_threads_and_CPUs)[1], , drop = FALSE])
##
}
##
#' Save a sampling choice (replacing the saved one for the same model file, N, sampling chains, threads and CPUs)
#' @param model_so_file,N_units,n_chains_sampling,n_threads_WCP_sampling,CPU_list As for
#'   fn_read_saved_sampling_choice().
#' @param num_chunks_sampling The chosen chunk count.
#' @param num_chunks_sampling_from_rule The chunk count of Paper 1's rule.
#' @param M_bytes_per_chain The measured M_bytes/chain.
#' @param seconds_spent_choosing The time the choice took (memory measurement and timings).
#' @return The path of the file, invisibly.
#' @keywords internal
fn_save_sampling_choice <-  function( model_so_file,
                                      N_units,
                                      n_chains_sampling,
                                      n_threads_WCP_sampling,
                                      CPU_list,
                                      num_chunks_sampling,
                                      num_chunks_sampling_from_rule,
                                      M_bytes_per_chain,
                                      seconds_spent_choosing
) {
        ##
        saved_choices_file <-  fn_saved_sampling_choices_file()
        dir.create(path = dirname(saved_choices_file), recursive = TRUE, showWarnings = FALSE)
        new_choice <-  data.frame( model_so_file                  = normalizePath(model_so_file),
                                   N_units                        = N_units,
                                   n_chains_sampling              = n_chains_sampling,
                                   n_threads_WCP_sampling         = n_threads_WCP_sampling,
                                   CPU_list                       = CPU_list,
                                   num_chunks_sampling            = num_chunks_sampling,
                                   num_chunks_sampling_from_rule  = num_chunks_sampling_from_rule,
                                   M_bytes_per_chain              = M_bytes_per_chain,
                                   seconds_spent_choosing         = seconds_spent_choosing,
                                   stringsAsFactors               = FALSE)
        saved_choices <-  if (file.exists(saved_choices_file)) {
              tryCatch(expr = readRDS(file = saved_choices_file), error = function(condition) NULL)
        } else NULL
        if (is.data.frame(saved_choices) && nrow(saved_choices) > 0) {
              same_model_N_chains_threads_and_CPUs <-  saved_choices$model_so_file == new_choice$model_so_file &
                                                       saved_choices$N_units == N_units &
                                                       saved_choices$n_chains_sampling == n_chains_sampling &
                                                       saved_choices$n_threads_WCP_sampling ==
                                                             n_threads_WCP_sampling &
                                                       saved_choices$CPU_list == CPU_list
              saved_choices <-  rbind(saved_choices[!same_model_N_chains_threads_and_CPUs, , drop = FALSE],
                                      new_choice)
        } else {
              saved_choices <-  new_choice
        }
        ## (written through a temporary file renamed over it, as several fits can save at once)
        temporary_file <-  tempfile(pattern = "saved_sampling_choices_", tmpdir = dirname(saved_choices_file),
                                    fileext = ".rds.tmp")
        saveRDS(object = saved_choices, file = temporary_file)
        file.rename(from = temporary_file, to = saved_choices_file)
        invisible(saved_choices_file)
##
}
##
##
#### ---- the sampling chunks of a Stan model: Paper 1's rule from its own memory, confirmed by timing ----------
##
#' Sampling chunks of a Stan model: Paper 1's rule from the model's own M_bytes/chain, confirmed by timing
#'
#' For a Stan model with NicoStan's N / chunk_size data contract (integer data fields N, the units summed by
#' reduce_sum_static, and chunk_size, the most units per piece): measures M_bytes/chain of this compiled model
#' (unless M_bytes_per_chain is given), applies Paper 1's rule for the sampling layout and the CPUs of this
#' process, and, when init_object is given and time_Paper_1_check_of_num_chunks = TRUE, applies Paper 1's check
#' ([fn_check_num_chunks_by_doubling_and_halving()]) with the sampling layout in NicoStan's own worker: the rule's
#' chunk count, twice it and half it, then the next doubling (or halving) for as long as it is faster (powers of
#' two, the partial sums of reduce_sum_static); the fastest value timed is used.
#'
#' @param model_so_file Path of the compiled BridgeStan model library (.so) of this model and math backend.
#' @param Stan_data_list The Stan data (with integer fields N and chunk_size).
#' @param n_chains_sampling Number of sampling chains run in parallel.
#' @param n_threads_WCP_sampling Within-chain threads of each sampling chain (default 1).
#' @param M_bytes_per_chain M_bytes/chain of this model, if already measured (default NULL: measured here).
#' @param init_list Initial values of one chain (the memory measurement's point; default: zero on the
#'   unconstrained scale).
#' @param init_object The model's initialisation object (default NULL: the chunk counts are not timed, and the
#'   rule's chunk count is used).
#' @param time_Paper_1_check_of_num_chunks TRUE (default): Paper 1's check of the rule's chunk count (doubling
#'   or halving while it gets faster), and the fastest value timed is used; FALSE: Paper 1's equations alone
#'   (the rule's chunk count).
#' @param fraction_slower_counted_as_tie A chunk count at most this fraction slower than the fastest is a near-tie
#'   (default 0.01, i.e. 1%).
#' @param reuse_saved_choice TRUE (default): reuse the choice saved for the same model file, N, number of sampling
#'   chains, within-chain threads and CPUs, when there is one.
#' @param n_rounds,n_timed_iterations,minimum_seconds_per_timing Settings of the timing
#'   (fn_time_burnin_iteration_for_each_num_chunks(); defaults 3, 3 and 0.2 s).
#' @return A list: num_chunks_sampling (the chunk count to use), chunk_size_sampling,
#'   num_chunks_sampling_from_rule, rule_is_fastest_or_near_tie (NA when not timed),
#'   num_chunks_within_tie_of_fastest, timings (a data frame:
#'   num_chunks, median_seconds_per_iteration, percent_slower_than_fastest; NULL when not timed),
#'   M_bytes_per_chain, rule (the list of fn_compute_sampling_num_chunks_for_CPUs_of_this_process()), N_units,
#'   n_chains_sampling, n_threads_WCP_sampling, CPU_list, reused_saved_choice and seconds_spent_choosing.
#' @export
fn_compute_sampling_num_chunks_for_Stan_model <-  function( model_so_file,
                                                           Stan_data_list,
                                                           ## (the kind of autodiff tape of the library, for
                                                           ##  Paper 1's equations: "Stan_math"
                                                           ##  or "nico_math"; no default)
                                                           kind_of_tape,
                                                           n_chains_sampling,
                                                           n_threads_WCP_sampling          = 1,
                                                           M_bytes_per_chain               = NULL,
                                                           init_list                       = NULL,
                                                           init_object                     = NULL,
                                                           time_Paper_1_check_of_num_chunks = TRUE,
                                                           fraction_slower_counted_as_tie  = 0.01,
                                                           reuse_saved_choice              = TRUE,
                                                           n_rounds                        = 3,
                                                           n_timed_iterations              = 3,
                                                           minimum_seconds_per_timing      = 0.2,
                                                           leapfrog_steps_per_timed_iteration = 20
) {
        ##
        if (!fn_Stan_data_has_N_and_chunk_size(Stan_data_list = Stan_data_list)) {
              stop(paste0("fn_compute_sampling_num_chunks_for_Stan_model: the Stan data need integer fields N ",
                          "and chunk_size (NicoStan's N / chunk_size data contract for reduce_sum_static)."))
        }
        N_units <-  Stan_data_list$N
        CPU_list <-  fn_format_CPU_list(CPU_ids = fn_read_L3_cache_topology_of_available_CPUs()$CPU_ids_available)
        ##
        ## ---- the choice saved for the same model file, N, sampling chains, threads and CPUs, reused as it is:
        ##
        if (isTRUE(reuse_saved_choice)) {
              saved_choice <-  fn_read_saved_sampling_choice( model_so_file           = model_so_file,
                                                              N_units                 = N_units,
                                                              n_chains_sampling       = n_chains_sampling,
                                                              n_threads_WCP_sampling  = n_threads_WCP_sampling,
                                                              CPU_list                = CPU_list)
              if (!is.null(saved_choice)) {
                    return(list( num_chunks_sampling              = saved_choice$num_chunks_sampling,
                                 chunk_size_sampling              = ceiling(N_units /
                                                                            saved_choice$num_chunks_sampling),
                                 num_chunks_sampling_from_rule    = saved_choice$num_chunks_sampling_from_rule,
                                 rule_is_fastest_or_near_tie      = NA,
                                 num_chunks_within_tie_of_fastest = NA,
                                 timings                          = NULL,
                                 M_bytes_per_chain                = saved_choice$M_bytes_per_chain,
                                 rule                             = NULL,
                                 N_units                          = N_units,
                                 n_chains_sampling                = n_chains_sampling,
                                 n_threads_WCP_sampling           = n_threads_WCP_sampling,
                                 CPU_list                         = CPU_list,
                                 reused_saved_choice              = TRUE,
                                 saved_choice                     = saved_choice,
                                 seconds_spent_choosing           = 0))
              }
        }
        time_choice_start <-  proc.time()[["elapsed"]]
        ##
        ## ---- M_bytes/chain of this compiled model (the full data as one piece):
        ##
        if (is.null(M_bytes_per_chain)) {
              Stan_data_list_one_piece <-  Stan_data_list
              Stan_data_list_one_piece$chunk_size <-  as.integer(N_units)
              n_nuisance_owned_by_NicoStan <-  0
              if (!is.null(init_object) && length(init_object$nuisance_blocks_held_by_NicoStan) > 0 &&
                  !is.null(init_object$n_nuisance_override)) {
                    n_nuisance_owned_by_NicoStan <-  init_object$n_nuisance_override
              }
              M_bytes_per_chain <-  fn_estimate_initial_M_bytes_per_chain(
                                          model_so_file                 = model_so_file,
                                          Stan_data_list                = Stan_data_list_one_piece,
                                          init_list                     = init_list,
                                          n_nuisance_owned_by_NicoStan  = n_nuisance_owned_by_NicoStan)
        }
        ## (the measurement's list, as fn_estimate_initial_M_bytes_per_chain() returns it: its number of bytes)
        if (is.list(M_bytes_per_chain)) M_bytes_per_chain <-  M_bytes_per_chain$M_bytes_per_chain
        ##
        ## ---- Paper 1's rule for the sampling layout and CPUs, rounded up to a power of two; no chunk cap:
        ##
        # rule <-  fn_compute_sampling_num_chunks_for_CPUs_of_this_process(
        #                M_bytes_per_chain       = M_bytes_per_chain,
        #                n_chains_sampling       = n_chains_sampling,
        #                n_threads_WCP_sampling  = n_threads_WCP_sampling,
        #                gradient_type           = "autodiff")
        ## (Paper 1's sampling equation, from the L2 cache per hardware thread and the kind of tape)
        L2_cache_and_threads_per_core <-
              fn_read_threads_per_core_and_L2_cache_size_of_first_CPU_of_this_process()
        sampling_equation_of_18_Oct <-  fn_sampling_num_chunks_of_Paper_1_equation(
                                              M_bytes_per_chain             = M_bytes_per_chain,
                                              N_units                       = N_units,
                                              kind_of_tape                  = kind_of_tape,
                                              L2_cache_size_bytes_per_core  =
                                                    L2_cache_and_threads_per_core$L2_cache_size_bytes,
                                              N_threads_per_core            =
                                                    L2_cache_and_threads_per_core$N_threads_per_core)
        rule <-  list(N_chunks = sampling_equation_of_18_Oct$num_chunks_sampling,
                      sampling_equation_of_18_Oct = sampling_equation_of_18_Oct)
        most_num_chunks <-  Inf
        num_chunks_sampling_from_rule <-  2^ceiling(log2(rule$N_chunks))
        ##
        ## ---- the rule's chunk count and the powers of two around it, timed with the sampling layout:
        ##
        ## num_chunks_candidates <-  sort(unique(pmin(most_num_chunks,
        ##                                            pmax(1, c(1, num_chunks_sampling_from_rule *
        ##                                                         c(1 / 4, 1 / 2, 1, 2, 4))))))
        ## (Paper 1's check, with the sampling layout: the rule's chunk count, twice it and half it, then the
        ##  next doubling (or halving) for as long as it is faster; powers of two, the partial sums of
        ##  reduce_sum_static)
        timings <-  NULL
        check <-  NULL
        L_main_of_timings <-  NA
        if (isTRUE(time_Paper_1_check_of_num_chunks) && !is.null(init_object) && (most_num_chunks > 1)) {
##            ##
##            ## (the leapfrog steps per timed iteration, the same for every timing of this model, N and
##            ##  layout: 20, or fewer for a slow model, so that each timing lasts about
##            ##  minimum_seconds_per_timing)
##            L_main_of_timings <-
##                  fn_leapfrog_steps_per_timed_iteration_for_timings_of_at_least_minimum_seconds(
##                                        init_object                 = init_object,
##                                        Stan_data_list              = Stan_data_list,
##                                        n_chains                    = n_chains_sampling,
##                                        n_threads_WCP               = n_threads_WCP_sampling,
##                                        num_chunks                  = num_chunks_sampling_from_rule,
##                                        n_timed_iterations          = n_timed_iterations,
##                                        minimum_seconds_per_timing  = minimum_seconds_per_timing)
              ## (Paper 1's L = 20 leapfrog steps for every timing by default; a slow model runs fewer timed
              ##  iterations instead, down to n_timed_iterations, i.e. the iterations fall with N, as in Paper 1;
              ##  leapfrog_steps_per_timed_iteration = NULL: fewer steps for a slow model too, so that each
              ##  timing lasts about minimum_seconds_per_timing)
              ## L_main_of_timings <-  20
              L_main_of_timings <-  leapfrog_steps_per_timed_iteration
              if (is.null(L_main_of_timings)) {
                    L_main_of_timings <-
                          fn_leapfrog_steps_per_timed_iteration_for_timings_of_at_least_minimum_seconds(
                                init_object                 = init_object,
                                Stan_data_list              = Stan_data_list,
                                n_chains                    = n_chains_sampling,
                                n_threads_WCP               = n_threads_WCP_sampling,
                                num_chunks                  = num_chunks_sampling_from_rule,
                                n_timed_iterations          = n_timed_iterations,
                                minimum_seconds_per_timing  = minimum_seconds_per_timing)
              }
              ## (the data's JSON text, written once: each chunk count's data file only changes its chunk_size)
              JSON_text_of_Stan_data_list <-  fn_JSON_text_of_Stan_data_list(Stan_data_list = Stan_data_list)
              check <-  tryCatch(
                    expr  = fn_check_num_chunks_by_doubling_and_halving(
                                  num_chunks_start        = num_chunks_sampling_from_rule,
                                  fn_median_seconds_per_iteration_of_each_num_chunks =
                                        function(num_chunks_candidates) {
                                        ##       fn_time_burnin_iteration_for_each_num_chunks(
                                        ##             init_object                 = init_object,
                                        ##             Stan_data_list              = Stan_data_list,
                                        ##             n_chains_burnin             = n_chains_sampling,
                                        ##             n_threads_WCP_burnin        = n_threads_WCP_sampling,
                                        ##             num_chunks_candidates       = num_chunks_candidates,
                                        ##             n_rounds                    = n_rounds,
                                        ##             n_timed_iterations          = n_timed_iterations,
                                        ##             minimum_seconds_per_timing  = minimum_seconds_per_timing,
                                        ##             L_main                      = L_main_of_timings,
                                        ##             JSON_text_of_Stan_data_list = JSON_text_of_Stan_data_list
                                        ##       )$median_seconds_per_burnin_iteration
                                              ## (timed in the sampler of the sampling phase: the chains
                                              ##  independent, as in the sampling)
                                              layouts_timed <-  data.frame(
                                                    n_chains       = n_chains_sampling,
                                                    n_threads_WCP  = n_threads_WCP_sampling,
                                                    num_chunks     = num_chunks_candidates)
                                              fn_time_sampling_of_Stan_model_for_each_layout(
                                                    init_object                  = init_object,
                                                    Stan_data_list               = Stan_data_list,
                                                    layouts                      = layouts_timed,
                                                    minimum_seconds_per_timing   = minimum_seconds_per_timing,
                                                    L_main                       = L_main_of_timings,
                                                    JSON_text_of_Stan_data_list  = JSON_text_of_Stan_data_list
                                              )$seconds_per_iteration
                                        },
                                  fewest_num_chunks       = max(1, n_threads_WCP_sampling),
                                  most_num_chunks         = most_num_chunks,
                                  multiple_of_num_chunks  = n_threads_WCP_sampling,
                                  powers_of_two           = TRUE),
                    error = function(condition) {
                          message(paste0("Sampling chunk counts not timed (", conditionMessage(condition),
                                         "); the rule's chunk count is used."))
                          NULL
                    })
              if (!is.null(check)) timings <-  check$timings
        }
        num_chunks_sampling <-  num_chunks_sampling_from_rule
        rule_is_fastest_or_near_tie <-  NA
        num_chunks_within_tie_of_fastest <-  NA
        if (!is.null(timings)) {
              names(timings)[names(timings) == "median_seconds_per_burnin_iteration"] <-
                    "median_seconds_per_iteration"
              fastest_seconds <-  min(timings$median_seconds_per_iteration)
              timings$percent_slower_than_fastest <-  100 * (timings$median_seconds_per_iteration /
                                                             fastest_seconds - 1)
              ## num_chunks_sampling <-  timings$num_chunks[which.min(timings$median_seconds_per_iteration)]
              ## (the value which Paper 1's check returns: the fastest value timed)
              num_chunks_sampling <-  check$num_chunks
              num_chunks_within_tie_of_fastest <-  timings$num_chunks[timings$percent_slower_than_fastest <=
                                                                      100 * fraction_slower_counted_as_tie]
              rule_is_fastest_or_near_tie <-  num_chunks_sampling_from_rule %in% num_chunks_within_tie_of_fastest
        }
        seconds_spent_choosing <-  proc.time()[["elapsed"]] - time_choice_start
        ## (saved for reuse when the chunk counts were timed)
        if (!is.null(timings)) {
              fn_save_sampling_choice( model_so_file                  = model_so_file,
                                       N_units                        = N_units,
                                       n_chains_sampling              = n_chains_sampling,
                                       n_threads_WCP_sampling         = n_threads_WCP_sampling,
                                       CPU_list                       = CPU_list,
                                       num_chunks_sampling            = num_chunks_sampling,
                                       num_chunks_sampling_from_rule  = num_chunks_sampling_from_rule,
                                       M_bytes_per_chain              = M_bytes_per_chain,
                                       seconds_spent_choosing         = seconds_spent_choosing)
        }
        return(list( num_chunks_sampling              = num_chunks_sampling,
                     chunk_size_sampling              = ceiling(N_units / num_chunks_sampling),
                     num_chunks_sampling_from_rule    = num_chunks_sampling_from_rule,
                     rule_is_fastest_or_near_tie      = rule_is_fastest_or_near_tie,
                     num_chunks_within_tie_of_fastest = num_chunks_within_tie_of_fastest,
                     timings                          = timings,
                     M_bytes_per_chain                = M_bytes_per_chain,
                     rule                             = rule,
                     N_units                          = N_units,
                     n_chains_sampling                = n_chains_sampling,
                     n_threads_WCP_sampling           = n_threads_WCP_sampling,
                     CPU_list                         = CPU_list,
                     reused_saved_choice              = FALSE,
                     L_main_of_timings                = L_main_of_timings,
                     seconds_spent_choosing           = seconds_spent_choosing))
##
}






















