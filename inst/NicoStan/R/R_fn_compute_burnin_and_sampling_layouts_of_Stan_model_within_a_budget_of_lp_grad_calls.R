##
## ==== R_fn_compute_burnin_and_sampling_layouts_of_Stan_model_within_a_budget_of_lp_grad_calls.R ==============
##
## ---- The burn-in layout AND the sampling layout of a Stan model (NicoStan's N / chunk_size data contract for
##      reduce_sum_static), chosen with Paper 1's equations and Paper 1's check, both within ONE budget of time
##      proportional to the model's own seconds per lp_grad call -
##
## The whole choice - the data's JSON text, the lp_grad time, M_bytes/chain, the burn-in layout (chains x threads
## per chain x N_chunks) and the sampling layout (threads per chain x N_chunks, every CPU used) - lasts at most
##
##   most seconds spent choosing  =  max( fewest_seconds_of_the_budget_of_the_choice,
##                                        budget_of_the_choice_in_lp_grad_calls x seconds per lp_grad call )
##
## i.e. a fixed number of the model's own gradients (a typical fit without chunking is about 26,000 to 124,000
## lp_grad calls of wall time, every model and N of Paper 3's benchmark on the laptop; 2000 is about 2% of it),
## with a floor so that the cheapest models still get the fewest valid timings. The steps:
##   (1) the data's JSON text, written once, for every step below (each layout only changes its chunk_size);
##   (2) M_bytes/chain of the full data as one piece: in a fresh child process (Paper 1's measurement, by default
##       started in the background so that it runs whilst the lp_grad time is measured), or in this R process
##       (where_M_bytes_per_chain_is_measured = "this_R_process": no child process, but 0.91 to 1.11 times the
##       fresh process's bytes on 11 models of Paper 3, i.e. not the same bytes);
##   (3) the seconds of one lp_grad call (fn_seconds_per_lp_grad_call_of_compiled_Stan_model()), measured once,
##       which sets the budget and the iterations of every timed call of both phases;
##   (4) the burn-in layout (fn_compute_burnin_n_threads_WCP_and_num_chunks_for_Stan_model()) within
##       fraction_of_the_budget_left_given_to_the_burnin_layout of the seconds left, then the sampling layout
##       (fn_compute_sampling_n_chains_n_threads_WCP_and_num_chunks_for_Stan_model()) within all the seconds left.
## Each phase starts from Paper 1's equations and keeps the move threshold of Paper 1's check
## (fraction_faster_needed_to_move); within its budget, each timed call lasts the seconds left divided by the
## calls still planned, a layout is timed only when one more call fits, and when not even two calls fit, Paper 1's
## equations alone choose that phase's layout, nothing timed (and a message says so).
##
##
##
##
#### ---- M_bytes/chain in a fresh child process, started in the background ------------------------------------

#' M_bytes/chain of a compiled Stan model in a fresh child process, started in the background
#'
#' Starts the fresh child process of [fn_estimate_initial_M_bytes_per_chain()] (one repeat, the same child
#' function, fn_measure_one_gradient_memory_in_this_process(), with the same arguments and environment), without
#' waiting for it, so that this R process can measure the lp_grad time meanwhile. The function returned waits for
#' the child and returns M_bytes/chain.
#'
#' @param model_so_file The model's compiled BridgeStan library (.so).
#' @param JSON_text_of_Stan_data The full data as one piece (chunk_size = N), as JSON text.
#' @param init_list Initial values of one chain (the measurement's point; default NULL: zero on the unconstrained
#'   scale).
#' @param n_nuisance_owned_by_NicoStan The size of the nuisance block NicoStan holds (0: none).
#' @return A function without arguments, which waits for the child and returns a list: M_bytes_per_chain,
#'   measurement_method_code and the child's whole result.
#' @noRd
fn_start_the_measurement_of_M_bytes_per_chain_in_a_fresh_process_in_the_background <-  function(
        model_so_file,
        JSON_text_of_Stan_data,
        init_list                     = NULL,
        n_nuisance_owned_by_NicoStan  = 0
) {

        ##
        ## ---- the data file of the child (it reads a file, as in fn_estimate_initial_M_bytes_per_chain()):
        ##
        json_file_path <-  tempfile(pattern = "data_of_the_memory_measurement_", fileext = ".json")
        writeChar(object = JSON_text_of_Stan_data, con = json_file_path, eos = NULL, useBytes = TRUE)
        ##
        ## ---- the child's function, with baseenv() as its enclosing environment, as in
        ##      fn_estimate_initial_M_bytes_per_chain():
        ##
        fn_child_measurement <-  fn_measure_one_gradient_memory_in_this_process
        environment(fn = fn_child_measurement) <-  baseenv()
        free_heap_release_location <-  fn_locate_free_heap_release_function()
        child_process <-  callr::r_bg( func            = fn_child_measurement,
                                       args            = list( model_so_file                        =
                                                                     normalizePath(model_so_file),
                                                               json_file_path                       =
                                                                     normalizePath(json_file_path),
                                                               unconstrained_parameter_vector       = NULL,
                                                               init_list                            = init_list,
                                                               default_evaluation_point             = "zero",
                                                               initial_values_seed                  = 123,
                                                               initial_values_range_unconstrained   = c(-2, 2),
                                                               max_attempts_for_finite_log_density  = 10,
                                                               propto                               = TRUE,
                                                               jacobian                             = TRUE,
                                                               free_heap_release_package_name       =
                                                                     free_heap_release_location$package_name,
                                                               free_heap_release_function_name      =
                                                                     free_heap_release_location$function_name,
                                                               n_nuisance_owned_by_NicoStan         =
                                                                     n_nuisance_owned_by_NicoStan),
                                       libpath         = .libPaths(),
                                       env             = c(callr::rcmd_safe_env(), STAN_NUM_THREADS = "1"),
                                       system_profile  = FALSE,
                                       user_profile    = FALSE)
        ##
        ## ---- the function that waits for the child:
        ##
        fn_wait_for_M_bytes_per_chain <-  function() {

              child_process$wait()
              result_of_child <-  tryCatch(expr  = child_process$get_result(),
                                           error = function(condition) {

                                                 child_error_message <-  conditionMessage(condition)
                                                 if (!is.null(condition$parent)) {
                                                       child_error_message <-  conditionMessage(condition$parent)
                                                 }
                                                 stop(paste0("The measurement of M_bytes/chain in the ",
                                                             "background failed in the child process: ",
                                                             substr(x = child_error_message, start = 1,
                                                                    stop = 500)), call. = FALSE)
                                           })

              unlink(x = json_file_path)
              return(list( M_bytes_per_chain        = result_of_child$gradient_memory_bytes,
                           measurement_method_code  = result_of_child$measurement_method_code,
                           result_of_child          = result_of_child))
        }

        return(fn_wait_for_M_bytes_per_chain)

}




#### ---- the burn-in and sampling layouts, within one budget of lp_grad calls --------------------------------

#' The burn-in and sampling layouts of a Stan model, chosen within one budget proportional to its lp_grad time
#'
#' NicoStan's choice of both layouts for an end user (and $sample() when no layout is given): the burn-in chains x
#' threads per chain x N_chunks and the sampling threads per chain x N_chunks (every CPU used), each from Paper
#' 1's equations and Paper 1's check, the whole choice within
#' max(fewest_seconds_of_the_budget_of_the_choice, budget_of_the_choice_in_lp_grad_calls x the model's seconds per
#' lp_grad call) seconds (see the head of this file for the steps).
#'
#' @param model_so_file Path of the compiled BridgeStan model library (.so) of this model and math backend.
#' @param Stan_data_list The Stan data (with integer fields N and chunk_size).
#' @param initial_values_of_the_chains A list with the matrices main (n_params_main rows) and nuisance
#'   (n_nuisance rows), one column per chain (at least as many as the chains of any layout timed), e.g.
#'   the starting values of a model whose constraint rejects the default ones (the two-class LC-MVOP);
#'   given on to both pickers (default NULL: NicoStan's own small starting values).
#' @param init_object The model's initialisation object (from NicoStan's initialise_model()).
#' @param n_chains_burnin Number of burn-in chains run in parallel (default 4).
#' @param n_chains_sampling The sampling chains, when they are given (default NULL: every CPU used).
#' @param budget_of_the_choice_in_lp_grad_calls The most seconds spent choosing both layouts, as a number of the
#'   model's own lp_grad calls (default 2000: about 2% of a typical fit without chunking of Paper 3's models).
#' @param fewest_seconds_of_the_budget_of_the_choice The least budget of the whole choice, in seconds, so that the
#'   cheapest models still get the fewest valid timings (default 0.5 s).
#' @param fraction_of_the_budget_left_given_to_the_burnin_layout The share of the seconds left after the lp_grad
#'   time and M_bytes/chain given to the burn-in layout (default 0.5); the sampling layout gets every second left
#'   after it.
#' @param n_timed_calls_planned_for_each_phase_within_the_budget The timed calls each phase's budget is shared out
#'   over (default 10).
#' @param where_M_bytes_per_chain_is_measured "fresh_process_in_the_background" (default: Paper 1's fresh-process
#'   measurement, started before the lp_grad timing and waited for after it), "fresh_process" (the same, after the
#'   lp_grad timing) or "this_R_process" (at the lp_grad timing's first gradient, no child process; when a
#'   gradient of this library ran in this R process before, the fresh process instead).
#' @param seconds_of_repeated_calls_of_the_lp_grad_timing,n_batches_of_the_lp_grad_timing The lp_grad timing's
#'   seconds of repeated calls and its batches (defaults 0.03 s and 3; see
#'   fn_seconds_per_lp_grad_call_of_compiled_Stan_model()).
#' @param leapfrog_steps_per_timed_iteration The leapfrog steps of each timed iteration (default 10, never
#'   lowered).
#' @param fraction_faster_needed_to_move The move threshold of Paper 1's check (default 0.03).
#' @param reuse_saved_burnin_choice TRUE: the burn-in choice saved for the same model file, N, burn-in chains and
#'   CPUs is reused (fn_read_saved_burnin_choice()); FALSE (default): chosen again.
#' @param keep_burnin_workers_for_reuse_in_the_check TRUE (default): a burn-in layout timed again reuses its
#'   worker, without its start-up (keep_workers_for_reuse_in_check of the burn-in choice).
#' @param pass_the_data_to_the_workers_as_JSON_text_instead_of_files TRUE (default): no data file per layout.
#' @param nuisance_metric_of_the_sampling_timings The nuisance metric of the sampling timings (default "diagonal",
#'   as a fit's sampling after its burn-in, and the ground truth of Paper 3's benchmark).
#' @param fraction_slower_within_which_a_neighbour_is_timed_again_in_the_last_round Passed to both phases' check
#'   (default NULL: every neighbour; see [fn_search_threads_per_chain_and_num_chunks_by_doubling_and_halving()]).
#' @param time_the_doubling_after_a_doubling_of_num_chunks_within_the_move_threshold Passed to both phases' check
#'   (default TRUE; see [fn_search_threads_per_chain_and_num_chunks_by_doubling_and_halving()]).
#' @param init_object_of_model_without_chunking,Stan_data_list_of_model_without_chunking The same model without
#'   tape chunking, for the sampling choice (default NULL: none).
#' @param n_nuisance_owned_by_NicoStan The size of the nuisance block NicoStan holds (default NULL: init_object's
#'   n_nuisance_override when the option NicoStan.reduce_sum_nuisance_block is set, otherwise 0).
#' @param init_list Initial values of one chain for the memory measurement (default NULL: zero on the
#'   unconstrained scale; given, M_bytes/chain is measured in a fresh process).
#' @param fewest_seconds_of_gradient_per_WCP_thread The burn-in rule's WCP limit (default 1e-5 s).
#' @param one_piece_per_thread_when_one_thread_per_chain The burn-in rule's pieces at one thread (default TRUE).
#' @param S_target_of_the_L2_cache_per_core_when_every_burnin_thread_has_a_core The burn-in rule's S_target
#'   (default TRUE). These three are the burn-in rule's changes (see
#'   [fn_compute_burnin_n_threads_WCP_and_num_chunks()]): on the burn-in ground truth of Paper 3's models, 211
#'   cases of both machines and every math backend, the median % slower than the fastest at the rule's layout
#'   fell from 4.6% to 0.6%, the largest from 1068% to 176%
#'   (score_changes_to_Paper_1_burnin_rule_on_the_ground_truth_of_every_model.R).
#' @param time_the_doubling_of_num_chunks_first_and_the_others_only_if_it_is_not_faster The sampling check
#'   upwards first (default TRUE).
#' @param compare_the_layout_found_with_one_thread_per_chain_and_N_chunks_1 The sampling check's comparison with
#'   one thread per chain and N_chunks = 1 (default TRUE). These two are the sampling check's changes (see
#'   [fn_compute_sampling_n_chains_n_threads_WCP_and_num_chunks_for_Stan_model()]).
#' @return A list: n_threads_WCP_burnin, num_chunks_burnin, n_chains_sampling, n_threads_WCP_sampling,
#'   num_chunks_sampling, use_model_without_chunking; burnin and sampling (the lists of the two choices);
#'   seconds_per_lp_grad_call, M_bytes_per_chain, where_M_bytes_per_chain_was_measured;
#'   most_seconds_spent_choosing_the_burnin_and_sampling_layouts (the budget),
#'   seconds_spent_choosing_the_burnin_and_sampling_layouts (the whole choice), and its parts:
#'   seconds_of_the_JSON_text, seconds_of_the_lp_grad_timing, seconds_waited_for_M_bytes_per_chain,
#'   seconds_spent_choosing_the_burnin_layout and seconds_spent_choosing_the_sampling_layout.
#' @export
fn_compute_burnin_and_sampling_layouts_of_Stan_model_within_a_budget_of_lp_grad_calls <-  function(
        model_so_file,
        Stan_data_list,
        init_object,
        ##  a Stan::math library, "nico_math" for a nico::math (AVX) library; no default, every caller passes
        ##  it)
        kind_of_tape,
        n_chains_burnin                                                            = 4,
        n_chains_sampling                                                          = NULL,
        budget_of_the_choice_in_lp_grad_calls                                      = 2000,
        fewest_seconds_of_the_budget_of_the_choice                                 = 0.5,
        fraction_of_the_budget_left_given_to_the_burnin_layout                     = 0.5,
        n_timed_calls_planned_for_each_phase_within_the_budget                     = 10,
        where_M_bytes_per_chain_is_measured                                        =
              "fresh_process_in_the_background",
        seconds_of_repeated_calls_of_the_lp_grad_timing                            = 0.03,
        n_batches_of_the_lp_grad_timing                                            = 3,
        leapfrog_steps_per_timed_iteration                                         = 10,
        fraction_faster_needed_to_move                                             = 0.03,
        reuse_saved_burnin_choice                                                  = FALSE,
        keep_burnin_workers_for_reuse_in_the_check                                 = TRUE,
        pass_the_data_to_the_workers_as_JSON_text_instead_of_files                 = TRUE,
        nuisance_metric_of_the_sampling_timings                                    = "diagonal",
        fraction_slower_within_which_a_neighbour_is_timed_again_in_the_last_round  = NULL,
        time_the_doubling_after_a_doubling_of_num_chunks_within_the_move_threshold = TRUE,
        init_object_of_model_without_chunking                                      = NULL,
        Stan_data_list_of_model_without_chunking                                   = NULL,
        n_nuisance_owned_by_NicoStan                                               = NULL,
        init_list                                                                  = NULL,
        initial_values_of_the_chains                                               = NULL,
        fewest_seconds_of_gradient_per_WCP_thread                                  = 1e-5,
        one_piece_per_thread_when_one_thread_per_chain                             = TRUE,
        S_target_of_the_L2_cache_per_core_when_every_burnin_thread_has_a_core      = TRUE,
        time_the_doubling_of_num_chunks_first_and_the_others_only_if_it_is_not_faster = TRUE,
        compare_the_layout_found_with_one_thread_per_chain_and_N_chunks_1          = TRUE
) {

        time_choice_start <-  proc.time()[["elapsed"]]
        if (!fn_Stan_data_has_N_and_chunk_size(Stan_data_list = Stan_data_list)) {
              stop(paste0("fn_compute_burnin_and_sampling_layouts_of_Stan_model_within_a_budget_of_lp_grad_",
                          "calls: the Stan data need integer fields N and chunk_size (NicoStan's N / chunk_size ",
                          "data contract for reduce_sum_static)."))
        }
        stopifnot(where_M_bytes_per_chain_is_measured %in% c("fresh_process_in_the_background", "fresh_process",
                                                             "this_R_process"))
        N_units <-  Stan_data_list$N
        ##
        ## ---- a library in NicoStan's nuisance-block form: the size of its nuisance block, as in both choices:
        ##
        if (is.null(n_nuisance_owned_by_NicoStan)) {
              ## (the held-block record identifies ownership, independently of the current session option)
              n_nuisance_owned_by_NicoStan <-  0
              if (!is.null(init_object) && length(init_object$nuisance_blocks_held_by_NicoStan) > 0 &&
                  !is.null(init_object$n_nuisance_override)) {
                    n_nuisance_owned_by_NicoStan <-  init_object$n_nuisance_override
              }
        }
        ##
        ## ---- (1) the data's JSON text, the full data as one piece (chunk_size = N), written once:
        ##
        Stan_data_list_one_piece <-  Stan_data_list
        Stan_data_list_one_piece$chunk_size <-  as.integer(N_units)
        JSON_text_of_Stan_data_list <-  fn_JSON_text_of_Stan_data_list(Stan_data_list = Stan_data_list_one_piece)
        seconds_of_the_JSON_text <-  proc.time()[["elapsed"]] - time_choice_start
        ##
        ## ---- the burn-in choice saved for the same model file, N, burn-in chains and CPUs (when reused, its
        ##      M_bytes/chain too):
        ##
        saved_burnin_choice <-  NULL
        CPU_ids_of_this_process <-  fn_read_L3_cache_topology_of_available_CPUs()$CPU_ids_available
        CPU_list_of_this_process <-  fn_format_CPU_list(CPU_ids = CPU_ids_of_this_process)
        if (isTRUE(reuse_saved_burnin_choice)) {
              saved_burnin_choice <-  fn_read_saved_burnin_choice(
                                            model_so_file    = model_so_file,
                                            N_units          = N_units,
                                            n_chains_burnin  = n_chains_burnin,
                                            CPU_list         = CPU_list_of_this_process)
        }
        ##
        ## ---- (2) M_bytes/chain: where it is measured (in this R process only when no gradient of this library
        ##      ran here before, and at the default evaluation point):
        ##
        where_M_bytes_per_chain_was_measured <-  where_M_bytes_per_chain_is_measured
        if (identical(where_M_bytes_per_chain_was_measured, "this_R_process") &&
            (fn_a_gradient_of_the_library_ran_in_this_R_process(model_so_file = model_so_file) ||
             !is.null(init_list))) {
              where_M_bytes_per_chain_was_measured <-  "fresh_process"
              message(paste0("M_bytes/chain measured in a fresh process, not in this R process: ",
                             if (!is.null(init_list)) "initial values given" else
                                   "a gradient of this library ran in this R process before"))
        }
        if (!is.null(saved_burnin_choice)) where_M_bytes_per_chain_was_measured <-  "saved_with_the_burnin_choice"
        fn_wait_for_M_bytes_per_chain <-  NULL
        if (identical(where_M_bytes_per_chain_was_measured, "fresh_process_in_the_background")) {
              fn_wait_for_M_bytes_per_chain <-
                    fn_start_the_measurement_of_M_bytes_per_chain_in_a_fresh_process_in_the_background(
                          model_so_file                 = model_so_file,
                          JSON_text_of_Stan_data        = JSON_text_of_Stan_data_list,
                          init_list                     = init_list,
                          n_nuisance_owned_by_NicoStan  = n_nuisance_owned_by_NicoStan)
        }
        ##
        ## ---- (3) the seconds of one lp_grad call, measured once for both phases (with M_bytes/chain at its
        ##      first gradient, when measured in this R process):
        ##
        time_of_lp_grad_timing_start <-  proc.time()[["elapsed"]]
        lp_grad <-  fn_seconds_per_lp_grad_call_of_compiled_Stan_model(
                          model_so_file                    = model_so_file,
                          JSON_text_of_Stan_data           = JSON_text_of_Stan_data_list,
                          seconds_of_repeated_calls        = seconds_of_repeated_calls_of_the_lp_grad_timing,
                          n_batches                        = n_batches_of_the_lp_grad_timing,
                          stop_if_library_already_loaded   = FALSE,
                          n_nuisance_owned_by_NicoStan     = n_nuisance_owned_by_NicoStan,
                          measure_the_memory_of_one_gradient_in_this_R_process =
                                identical(where_M_bytes_per_chain_was_measured, "this_R_process"),
                          BridgeStan_model_already_built_from_the_same_JSON_text  = NULL,
                          fn_whether_the_repeated_calls_are_needed                = NULL)
        seconds_of_the_lp_grad_timing <-  proc.time()[["elapsed"]] - time_of_lp_grad_timing_start
        seconds_per_lp_grad_call <-  lp_grad$seconds_per_lp_grad_call
        ##
        ## ---- the budget of the whole choice, from the lp_grad time:
        ##
        most_seconds_spent_choosing <-  max(fewest_seconds_of_the_budget_of_the_choice,
                                            budget_of_the_choice_in_lp_grad_calls * seconds_per_lp_grad_call)
        ##
        ## ---- M_bytes/chain: waited for (background), measured now (fresh process), or from the lp_grad timing:
        ##
        time_of_wait_start <-  proc.time()[["elapsed"]]
        if (identical(where_M_bytes_per_chain_was_measured, "fresh_process_in_the_background")) {
              M_bytes_per_chain <-  fn_wait_for_M_bytes_per_chain()$M_bytes_per_chain
        } else if (identical(where_M_bytes_per_chain_was_measured, "fresh_process")) {
              M_bytes_per_chain <-  fn_estimate_initial_M_bytes_per_chain(
                                          model_so_file                 = model_so_file,
                                          Stan_data_list                = Stan_data_list_one_piece,
                                          init_list                     = init_list,
                                          n_repeats                     = 1,
                                          n_nuisance_owned_by_NicoStan  = n_nuisance_owned_by_NicoStan
                                    )$M_bytes_per_chain
        } else if (identical(where_M_bytes_per_chain_was_measured, "this_R_process")) {
              M_bytes_per_chain <-  lp_grad$memory_of_one_gradient$M_bytes_per_chain
        } else {
              M_bytes_per_chain <-  saved_burnin_choice$M_bytes_per_chain
        }
        seconds_waited_for_M_bytes_per_chain <-  proc.time()[["elapsed"]] - time_of_wait_start
        ##
        ## ---- (4) the burn-in layout, within its share of the seconds left (or the saved choice):
        ##
        fn_seconds_left <-  function() {

              return(most_seconds_spent_choosing - (proc.time()[["elapsed"]] - time_choice_start))
        }

        time_of_burnin_choice_start <-  proc.time()[["elapsed"]]
        if (!is.null(saved_burnin_choice)) {
              burnin_choice <-  list( n_threads_WCP_burnin    = saved_burnin_choice$n_threads_WCP_burnin,
                                      num_chunks_burnin       = saved_burnin_choice$num_chunks_burnin,
                                      M_bytes_per_chain       = saved_burnin_choice$M_bytes_per_chain,
                                      reused_saved_choice     = TRUE,
                                      seconds_spent_choosing  = 0)
        } else {
              burnin_choice <-  fn_compute_burnin_n_threads_WCP_and_num_chunks_for_Stan_model(
                    model_so_file                                    = model_so_file,
                    Stan_data_list                                   = Stan_data_list,
                    kind_of_tape                                     = kind_of_tape,
                    n_chains_burnin                                  = n_chains_burnin,
                    init_object                                      = init_object,
                    initial_values_of_the_chains                     = initial_values_of_the_chains,
                    reuse_saved_choice                               = FALSE,
                    M_bytes_per_chain                                = M_bytes_per_chain,
                    time_Paper_1_check_of_num_chunks                 = TRUE,
                    leapfrog_steps_per_timed_iteration               = leapfrog_steps_per_timed_iteration,
                    keep_workers_for_reuse_in_check                  = keep_burnin_workers_for_reuse_in_the_check,
                    fraction_faster_needed_to_move                   = fraction_faster_needed_to_move,
                    n_nuisance_owned_by_NicoStan                     = n_nuisance_owned_by_NicoStan,
                    seconds_per_lp_grad_call                         = seconds_per_lp_grad_call,
                    JSON_text_of_Stan_data_list                      = JSON_text_of_Stan_data_list,
                    most_seconds_spent_choosing_the_burnin_layout    =
                          fraction_of_the_budget_left_given_to_the_burnin_layout * max(0, fn_seconds_left()),
                    n_timed_calls_planned_within_the_budget          =
                          n_timed_calls_planned_for_each_phase_within_the_budget,
                    pass_the_data_to_the_workers_as_JSON_text_instead_of_files =
                          pass_the_data_to_the_workers_as_JSON_text_instead_of_files,
                    fraction_slower_within_which_a_neighbour_is_timed_again_in_the_last_round =
                          fraction_slower_within_which_a_neighbour_is_timed_again_in_the_last_round,
                    time_the_doubling_after_a_doubling_of_num_chunks_within_the_move_threshold =
                          time_the_doubling_after_a_doubling_of_num_chunks_within_the_move_threshold,
                    fewest_seconds_of_gradient_per_WCP_thread        = fewest_seconds_of_gradient_per_WCP_thread,
                    one_piece_per_thread_when_one_thread_per_chain   =
                          one_piece_per_thread_when_one_thread_per_chain,
                    S_target_of_the_L2_cache_per_core_when_every_burnin_thread_has_a_core =
                          S_target_of_the_L2_cache_per_core_when_every_burnin_thread_has_a_core)
        }
        seconds_spent_choosing_the_burnin_layout <-  proc.time()[["elapsed"]] - time_of_burnin_choice_start
        ##
        ## ---- (4) the sampling layout, within every second left:
        ##
        time_of_sampling_choice_start <-  proc.time()[["elapsed"]]
        sampling_choice <-  fn_compute_sampling_n_chains_n_threads_WCP_and_num_chunks_for_Stan_model(
              model_so_file                                    = model_so_file,
              Stan_data_list                                   = Stan_data_list,
              kind_of_tape                                     = kind_of_tape,
              M_bytes_per_chain                                = M_bytes_per_chain,
              n_chains_sampling                                = n_chains_sampling,
              init_object                                      = init_object,
              initial_values_of_the_chains                     = initial_values_of_the_chains,
              time_Paper_1_check                               = TRUE,
              leapfrog_steps_per_timed_iteration               = leapfrog_steps_per_timed_iteration,
              fraction_faster_needed_to_move                   = fraction_faster_needed_to_move,
              init_object_of_model_without_chunking            = init_object_of_model_without_chunking,
              Stan_data_list_of_model_without_chunking         = Stan_data_list_of_model_without_chunking,
              n_nuisance_owned_by_NicoStan                     = n_nuisance_owned_by_NicoStan,
              seconds_per_lp_grad_call                         = seconds_per_lp_grad_call,
              JSON_text_of_Stan_data_list                      = JSON_text_of_Stan_data_list,
              most_seconds_spent_choosing_the_sampling_layout  = max(0, fn_seconds_left()),
              n_timed_calls_planned_within_the_budget          =
                    n_timed_calls_planned_for_each_phase_within_the_budget,
              pass_the_data_to_the_samplers_as_JSON_text_instead_of_files =
                    pass_the_data_to_the_workers_as_JSON_text_instead_of_files,
              nuisance_metric_of_the_timings                   = nuisance_metric_of_the_sampling_timings,
              fraction_slower_within_which_a_neighbour_is_timed_again_in_the_last_round =
                    fraction_slower_within_which_a_neighbour_is_timed_again_in_the_last_round,
              time_the_doubling_after_a_doubling_of_num_chunks_within_the_move_threshold =
                    time_the_doubling_after_a_doubling_of_num_chunks_within_the_move_threshold,
              time_the_doubling_of_num_chunks_first_and_the_others_only_if_it_is_not_faster =
                    time_the_doubling_of_num_chunks_first_and_the_others_only_if_it_is_not_faster,
              compare_the_layout_found_with_one_thread_per_chain_and_N_chunks_1 =
                    compare_the_layout_found_with_one_thread_per_chain_and_N_chunks_1)
        seconds_spent_choosing_the_sampling_layout <-  proc.time()[["elapsed"]] - time_of_sampling_choice_start
        seconds_spent_choosing <-  proc.time()[["elapsed"]] - time_choice_start
        ##
        message(paste0("Burn-in and sampling layouts chosen in ", round(seconds_spent_choosing, 2), " s (budget ",
                       round(most_seconds_spent_choosing, 2), " s = ",
                       if (most_seconds_spent_choosing > fewest_seconds_of_the_budget_of_the_choice) paste0(
                             budget_of_the_choice_in_lp_grad_calls, " lp_grad calls of ",
                             signif(1000 * seconds_per_lp_grad_call, 3), " ms") else "its floor",
                       "): the data's JSON text ", round(seconds_of_the_JSON_text, 3), " s, the lp_grad time ",
                       round(seconds_of_the_lp_grad_timing, 3), " s, M_bytes/chain (",
                       where_M_bytes_per_chain_was_measured, ") ", round(seconds_waited_for_M_bytes_per_chain, 3),
                       " s more, the burn-in layout ", round(seconds_spent_choosing_the_burnin_layout, 3),
                       " s, the sampling layout ", round(seconds_spent_choosing_the_sampling_layout, 3), " s"))
        ##
        return(list(
              n_threads_WCP_burnin                                         = burnin_choice$n_threads_WCP_burnin,
              num_chunks_burnin                                            = burnin_choice$num_chunks_burnin,
              n_chains_sampling                                            = sampling_choice$n_chains_sampling,
              n_threads_WCP_sampling                                       =
                    sampling_choice$n_threads_WCP_sampling,
              num_chunks_sampling                                          = sampling_choice$num_chunks_sampling,
              use_model_without_chunking                                   =
                    sampling_choice$use_model_without_chunking,
              burnin                                                       = burnin_choice,
              sampling                                                     = sampling_choice,
              seconds_per_lp_grad_call                                     = seconds_per_lp_grad_call,
              M_bytes_per_chain                                            = M_bytes_per_chain,
              where_M_bytes_per_chain_was_measured                         = where_M_bytes_per_chain_was_measured,
              budget_of_the_choice_in_lp_grad_calls                        =
                    budget_of_the_choice_in_lp_grad_calls,
              most_seconds_spent_choosing_the_burnin_and_sampling_layouts  = most_seconds_spent_choosing,
              seconds_spent_choosing_the_burnin_and_sampling_layouts       = seconds_spent_choosing,
              seconds_of_the_JSON_text                                     = seconds_of_the_JSON_text,
              seconds_of_the_lp_grad_timing                                = seconds_of_the_lp_grad_timing,
              seconds_waited_for_M_bytes_per_chain                         = seconds_waited_for_M_bytes_per_chain,
              seconds_spent_choosing_the_burnin_layout                     =
                    seconds_spent_choosing_the_burnin_layout,
              seconds_spent_choosing_the_sampling_layout                   =
                    seconds_spent_choosing_the_sampling_layout))

}




















