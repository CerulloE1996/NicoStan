




#### ---- the sampling layout of a Stan model on the CPUs of this process: chains, WCP threads and N_chunks ------

#' The sampling layout of a Stan model on all the CPUs of this process: the chains, the within-chain threads (WCP)
#' and N_chunks, from Paper 1's equations and Paper 1's check
#'
#' For a Stan model with NicoStan's N / chunk_size data contract (integer data fields N, the units summed by
#' reduce_sum_static, and chunk_size, the most units per piece), with every CPU of this process used, i.e.
#' floor(CPUs / threads per chain) sampling chains (or the n_chains_sampling given, kept as given: then at most
#' floor(CPUs / n_chains_sampling) threads per chain). Paper 1's equations give one chain per CPU with 1 thread
#' (no WCP), or, when the chains are given, the most threads per chain (a power of two) that the CPUs allow, and
#' the N_chunks of Paper 1's rule for that layout (one chunk fits in the L3 cache per active thread, for an
#' autodiff gradient), rounded up to a power of two, at least one chunk per thread. When the equations give one
#' chain per CPU and N_chunks = 1 (the model's tape fits in its share of the cache), one chain per CPU with 1
#' thread and 1 chunk is used, without a check. Otherwise, Paper 1's check
#' ([fn_search_threads_per_chain_and_num_chunks_by_doubling_and_halving()]) searches the threads per chain and
#' N_chunks together from the equations' layout, so that WCP with chunking is chosen when it is faster than
#' chunking alone; each layout is timed in the sampler of the sampling phase
#' ([fn_time_sampling_of_Stan_model_for_each_layout()]: the chains run independently in one call, as in the
#' sampling, and the call's setup is not counted), and the layouts are compared by their seconds per draw over all
#' chains (the seconds per iteration divided by the number of chains), i.e. by their sampling throughput.
#' When the same model without tape chunking (its original model) is given, the layout of one thread per chain and
#' N_chunks = 1 is timed with both models - the original model and this model with one chunk - and the faster is
#' used: one chunk only adds the nested autodiff of reduce_sum_static over the whole data to the original model's
#' work, unless the tape-chunked model's own code is faster (e.g. better vectorised). When the equations give that
#' layout and nothing is timed, the original model is used.
#'
#' @param model_so_file Path of the compiled BridgeStan model library (.so) of this model and math backend.
#' @param Stan_data_list The Stan data (with integer fields N and chunk_size).
#' @param M_bytes_per_chain M_bytes/chain of this model, if already measured (default NULL: measured here).
#' @param n_chains_sampling The sampling chains, when they are given (default NULL: floor(CPUs / threads per
#'   chain), every CPU used).
#' @param init_list Initial values of one chain (the memory measurement's point; default: zero on the
#'   unconstrained scale).
#' @param initial_values_of_the_chains A list with the matrices main (n_params_main rows) and nuisance
#'   (n_nuisance rows), one column per chain (at least as many as the chains of any layout timed), e.g.
#'   the starting values of a model whose constraint rejects the default ones (the two-class LC-MVOP);
#'   given on to every timing (default NULL: NicoStan's own small starting values).
#' @param init_object The model's initialisation object (default NULL: not timed, and the equations' layout is
#'   used).
#' @param time_Paper_1_check TRUE (default): Paper 1's check from the equations' layout; FALSE: the equations'
#'   layout.
#' @param n_timed_iterations No longer used (the burn-in worker's timings; kept so that calls which give it still
#'   work).
#' @param minimum_seconds_per_timing The shortest time of the iterations of each layout's long timing call
#'   (default 0.25 s; see [fn_time_sampling_of_Stan_model_for_each_layout()]).
#' @param leapfrog_steps_per_timed_iteration The leapfrog steps of each timed iteration, i.e. a fixed trajectory
#'   length (default 10), lowered only when even the fewest timed iterations at it would last longer than
#'   minimum_seconds_per_timing (from the model's lp_grad time; see
#'   R_fn_seconds_per_lp_grad_call_and_L_and_N_iter_of_each_timed_call.R).
#' @param fraction_faster_needed_to_move A neighbour must be faster by more than this fraction for the check to
#'   move to it (default 0.03; see fn_search_threads_per_chain_and_num_chunks_by_doubling_and_halving()).
#' @param keep_workers_for_reuse_in_check No longer used (the burn-in worker's timings; kept so that calls which
#'   give it still work).
#' @param init_object_of_model_without_chunking The initialisation object of the same model without tape
#'   chunking, i.e. its original model (default NULL: none, and N_chunks = 1 means this model with one chunk).
#' @param Stan_data_list_of_model_without_chunking The Stan data of the original model (default NULL: none).
#' @param n_nuisance_owned_by_NicoStan The size of the nuisance block of a library in NicoStan's nuisance-block
#'   form (default NULL: init_object's n_nuisance_override when the option NicoStan.reduce_sum_nuisance_block is
#'   set, otherwise 0), whose values the memory measurement sets (see [fn_estimate_initial_M_bytes_per_chain()]).
#' @param seconds_per_lp_grad_call The seconds of one lp_grad call of this model and data (default NULL: measured
#'   here), e.g. the seconds_per_lp_grad_call returned by
#'   [fn_compute_burnin_n_threads_WCP_and_num_chunks_for_Stan_model()].
#' @param JSON_text_of_Stan_data_list The data's JSON text (fn_JSON_text_of_Stan_data_list()), when already
#'   written (default NULL: written here).
#' @param most_seconds_spent_choosing_the_sampling_layout The budget of this choice, in seconds, from the start of
#'   this function (default NULL: no budget, as before). With it: the leapfrog steps of the timings stay at
#'   leapfrog_steps_per_timed_iteration (never lowered); each timed call lasts the seconds left divided by the
#'   timed calls still planned (n_timed_calls_planned_within_the_budget), less the start-up of a call (measured
#'   from the calls made), its iterations from the seconds per iteration of the call before it (the first from
#'   the lp_grad time), at least 3; a layout is timed only when one more call fits in the seconds left, and the
#'   search ends, with the fastest layout timed, when none does; and when not even two of the shortest calls fit,
#'   Paper 1's equations alone choose the layout, nothing timed.
#' @param n_timed_calls_planned_within_the_budget The timed calls the budget is shared out over (default 10).
#' @param seconds_of_start_up_of_a_timed_call_before_one_is_measured The start-up of a timed call (the sampler's
#'   workers and untimed iteration) assumed until the first call has measured it (default 0.05 s).
#' @param pass_the_data_to_the_samplers_as_JSON_text_instead_of_files TRUE: each layout's sampler is given the
#'   data's JSON text, without a data file (see fn_time_sampling_of_Stan_model_for_each_layout()); FALSE
#'   (default): a data file per layout, as before.
#' @param nuisance_metric_of_the_timings The nuisance metric of the timed calls, "identity" (default, as before)
#'   or "diagonal" (as a fit's sampling after its burn-in; see fn_time_sampling_of_Stan_model_for_each_layout()).
#' @param fraction_slower_within_which_a_neighbour_is_timed_again_in_the_last_round Passed to
#'   [fn_search_threads_per_chain_and_num_chunks_by_doubling_and_halving()] (default NULL: every neighbour timed
#'   again).
#' @param time_the_doubling_after_a_doubling_of_num_chunks_within_the_move_threshold Passed to
#'   [fn_search_threads_per_chain_and_num_chunks_by_doubling_and_halving()] (default TRUE).
#' @param time_the_doubling_of_num_chunks_first_and_the_others_only_if_it_is_not_faster Passed to
#'   [fn_search_threads_per_chain_and_num_chunks_by_doubling_and_halving()] (default FALSE: every neighbour at
#'   once, as before; TRUE: upwards first, as Paper 1's equations' N_chunks is at or below the fastest in 211 of
#'   213 sampling cases of Paper 3's ground truth).
#' @param compare_the_layout_found_with_one_thread_per_chain_and_N_chunks_1 TRUE: the layout found by the check is
#'   compared with one thread per chain and N_chunks = 1 (timed when the check did not, budget permitting), and
#'   (default): as before.
#'   decision): Paper 1's check runs only when its cheapest form, two timed calls (the equations' layout and one
#'   neighbour), costs at most this fraction of a typical fit without chunking, its seconds from the lp_grad time;
#'   otherwise any search would cost more than it can save, and Paper 1's equations alone choose the layout
#'   (default 0.02; NULL: no such rule, as before).
#' @param iterations_of_a_typical_fit_without_chunking The iterations of a typical fit without chunking, each of
#'   sampling iterations).
#' @return A list: n_chains_sampling, n_threads_WCP_sampling, num_chunks_sampling and chunk_size_sampling (the
#'   layout to use); use_model_without_chunking (TRUE: the original model is used, with that layout of one thread
#'   per chain and N_chunks = 1); n_chains_sampling_from_equations, n_threads_WCP_sampling_from_equations and
#'   num_chunks_sampling_from_equations; tape_fits_in_cache (TRUE: the equations' N_chunks is 1 and the tape per
#'   chain fits in the L2 cache of one hardware thread, and no check);
#'   timings (a data frame: n_chains_sampling, n_threads_WCP_sampling, num_chunks,
#'   mean_seconds_per_iteration, mean_seconds_per_draw_over_all_chains and n_timings; NULL when not timed);
#'   mean_seconds_per_draw_of_one_chunk_of_this_model and mean_seconds_per_draw_of_model_without_chunking (the
#'   means of the timings of the two models at one thread per chain and N_chunks = 1; NA when not timed);
#'   n_timings, M_bytes_per_chain, rule, N_units, CPU_list, L_main_of_timings and seconds_spent_choosing;
#'   seconds_of_a_typical_fit_without_chunking, seconds_of_the_cheapest_search and
#'   search_skipped_because_it_would_cost_more_than_its_share_of_a_typical_fit (the budget rule).
#' @export
fn_compute_sampling_n_chains_n_threads_WCP_and_num_chunks_for_Stan_model <-  function(
        model_so_file,
        Stan_data_list,
        ##  a Stan::math library, "nico_math" for a nico::math (AVX) library; no default, every caller passes
        ##  it)
        kind_of_tape,
        M_bytes_per_chain                         = NULL,
        n_chains_sampling                         = NULL,
        init_list                                 = NULL,
        init_object                               = NULL,
        initial_values_of_the_chains              = NULL,
        time_Paper_1_check                        = TRUE,
        n_timed_iterations                        = 1,
        minimum_seconds_per_timing                = 0.25,
        leapfrog_steps_per_timed_iteration        = 10,
        keep_workers_for_reuse_in_check           = FALSE,
        fraction_faster_needed_to_move            = 0.03,
        init_object_of_model_without_chunking     = NULL,
        Stan_data_list_of_model_without_chunking  = NULL,
        n_nuisance_owned_by_NicoStan              = NULL,
        seconds_per_lp_grad_call                  = NULL,
        JSON_text_of_Stan_data_list                                                = NULL,
        most_seconds_spent_choosing_the_sampling_layout                            = NULL,
        n_timed_calls_planned_within_the_budget                                    = 10,
        seconds_of_start_up_of_a_timed_call_before_one_is_measured                 = 0.05,
        pass_the_data_to_the_samplers_as_JSON_text_instead_of_files                = FALSE,
        nuisance_metric_of_the_timings                                             = "identity",
        fraction_slower_within_which_a_neighbour_is_timed_again_in_the_last_round  = NULL,
        time_the_doubling_after_a_doubling_of_num_chunks_within_the_move_threshold = TRUE,
        time_the_doubling_of_num_chunks_first_and_the_others_only_if_it_is_not_faster = FALSE,
      # compare_the_layout_found_with_one_thread_per_chain_and_N_chunks_1          = FALSE,
        compare_the_layout_found_with_one_thread_per_chain_and_N_chunks_1          = TRUE,
      # largest_fraction_of_a_typical_fit_without_chunking_spent_searching         = 0.02,
      ##  the 2% skip left the search out at N = 500, where the equations' layout was 57-186% slower)
        largest_fraction_of_a_typical_fit_without_chunking_spent_searching         = NULL,
        iterations_of_a_typical_fit_without_chunking                               = 1500
) {

        if (!fn_Stan_data_has_N_and_chunk_size(Stan_data_list = Stan_data_list)) {
              stop(paste0("fn_compute_sampling_n_chains_n_threads_WCP_and_num_chunks_for_Stan_model: ",
                          "the Stan data need integer fields N and chunk_size (NicoStan's N / chunk_size data ",
                          "contract for reduce_sum_static)."))
        }
        time_choice_start <-  proc.time()[["elapsed"]]
        N_units <-  Stan_data_list$N
        CPU_ids <-  fn_read_L3_cache_topology_of_available_CPUs()$CPU_ids_available
        CPU_list <-  fn_format_CPU_list(CPU_ids = CPU_ids)
        n_CPUs <-  length(CPU_ids)
        ## (the seconds of each step of this choice, printed at its end: the CPU topology, Paper 1's equations,
        ##  the data's JSON text and the lp_grad timing, the L2 cache read, and the rest)
        seconds_of_the_topology_read <-  proc.time()[["elapsed"]] - time_choice_start
        seconds_of_the_lp_grad_step <-  0
        data_file_of_the_fit_used <-  "not needed"
        model_of_the_fit_used <-  "not needed"
        repeated_calls_of_the_lp_grad_timing_made <-  "not needed"
        ## (the chains given are kept; otherwise every CPU is used, floor(CPUs / threads per chain) chains)
        n_chains_given <-  n_chains_sampling
        fn_n_chains_of_threads_per_chain <-  function(threads_per_chain) {

              if (!is.null(n_chains_given)) return(n_chains_given)
              return(floor(n_CPUs / threads_per_chain))
        }

        most_threads_per_chain <-  if (is.null(n_chains_given)) n_CPUs else max(1, floor(n_CPUs / n_chains_given))
        ##
        ## ---- a library in NicoStan's nuisance-block form: the size of its nuisance block (given, or that of
        ##      init_object when the option NicoStan.reduce_sum_nuisance_block is set), whose values the memory
        ##      measurement sets before its evaluation; 0 for every other library:
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
        ## ---- M_bytes/chain of this compiled model (the full data as one piece):
        ##
        if (is.null(M_bytes_per_chain)) {
              # Stan_data_list_one_piece <-  Stan_data_list
              # Stan_data_list_one_piece$chunk_size <-  as.integer(N_units)
              Stan_data_list_one_piece <-  fn_Stan_data_list_with_the_chunks_of_N_chunks(
                                                 Stan_data_list  = Stan_data_list,
                                                 N_chunks        = 1)
              ## (one fresh-process measurement: the peak memory of one log density and gradient is the same in
              ##  every repeat to 0.1% (the three repeats of every case of the 18 Oct calibration agreed to that),
              ##  and each repeat is a fresh R process that loads the library and the data, seconds each at
              ##  large N)
              M_bytes_per_chain <-  fn_estimate_initial_M_bytes_per_chain(
                                          model_so_file                 = model_so_file,
                                          Stan_data_list                = Stan_data_list_one_piece,
                                          init_list                     = init_list,
                                          n_repeats                     = 1,
                                          n_nuisance_owned_by_NicoStan  = n_nuisance_owned_by_NicoStan)
        }
        if (is.list(M_bytes_per_chain)) M_bytes_per_chain <-  M_bytes_per_chain$M_bytes_per_chain
        ##
        ## ---- the leapfrog steps and the timed iterations of every timed call of the check, from the model's own
        ##      lp_grad time (R_fn_seconds_per_lp_grad_call_and_L_and_N_iter_of_each_timed_call.R): the fewest
        ##      timed iterations at leapfrog_steps_per_timed_iteration lasting minimum_seconds_per_timing, and the
        ##      leapfrog steps lowered only when even those iterations would last longer; the same for every
        ##      layout, so that no layout pays a sizing call of its own:
        ##
        L_of_timed_calls <-  leapfrog_steps_per_timed_iteration
        N_iter_of_timed_calls <-  NULL
        ##
        ## ---- the budget of this choice (when given): the seconds left, the start-up of the calls made, the
        ##      seconds per iteration of the call before (the next call's iterations come from it), and whether
        ##      one more timed call fits:
        ##
        budget_given <-  !is.null(most_seconds_spent_choosing_the_sampling_layout)
        equations_alone_because_of_the_budget <-  FALSE
        n_timed_calls_made <-  0
        seconds_of_start_up_of_each_call_made <-  numeric(0)
        ## (the whole timing of the last call made, for the details of each timing given to the search)
        timing_of_the_last_call <-  NULL
        seconds_per_iteration_of_the_call_before <-  NA_real_
        fewest_iterations_of_a_timed_call <-  3
        fn_seconds_left_of_the_budget <-  function() {

              seconds_spent_so_far <-  proc.time()[["elapsed"]] - time_choice_start
              return(most_seconds_spent_choosing_the_sampling_layout - seconds_spent_so_far)
        }

        fn_seconds_of_start_up_of_a_call <-  function() {

              if (length(seconds_of_start_up_of_each_call_made) == 0) {
                    return(seconds_of_start_up_of_a_timed_call_before_one_is_measured)
              }
              return(mean(seconds_of_start_up_of_each_call_made))
        }

        fn_seconds_per_iteration_expected <-  function() {

              if (is.finite(seconds_per_iteration_of_the_call_before)) {
                    return(seconds_per_iteration_of_the_call_before)
              }
              return(L_of_timed_calls * seconds_per_lp_grad_call)
        }

        ## (the shortest call: its start-up, then its untimed iteration and the fewest timed iterations)
        fn_seconds_of_the_shortest_call <-  function() {

              return(fn_seconds_of_start_up_of_a_call() +
                     (1 + fewest_iterations_of_a_timed_call) * fn_seconds_per_iteration_expected())
        }

        fn_another_timing_fits_in_the_budget <-  NULL
        if (budget_given) {
              fn_another_timing_fits_in_the_budget <-  function() {

                    return(fn_seconds_left_of_the_budget() >= fn_seconds_of_the_shortest_call())
              }

        }
        ## (one layout timed in the sampler: under a budget, its iterations from the seconds left and the seconds
        ##  per iteration of the call before; otherwise N_iter_of_timed_calls, as before)
        ## (the seconds of each timed call, of its timed iterations and of the gap before it, i.e. the work of
        ##  this function between two calls, printed at the end of the choice)
        time_at_the_end_of_the_last_step <-  NA_real_
        seconds_of_each_timed_call <-  numeric(0)
        seconds_of_the_timed_iterations_of_each_call <-  numeric(0)
        seconds_of_the_gap_before_each_call <-  numeric(0)
        n_recorded_timed_sampling_attempts <-  0
        fn_seconds_per_iteration_of_one_layout <-  function(init_object_of_call, Stan_data_list_of_call,
                                                            layout_timed, JSON_text_of_call) {

              n_iterations_of_call <-  N_iter_of_timed_calls
              if (budget_given) {
                    n_timed_calls_left <-  max(1, n_timed_calls_planned_within_the_budget - n_timed_calls_made)
                    seconds_of_timed_iterations_of_call <-  fn_seconds_left_of_the_budget() / n_timed_calls_left -
                                                            fn_seconds_of_start_up_of_a_call()
                    n_iterations_of_call <-  max(fewest_iterations_of_a_timed_call,
                                                 floor(seconds_of_timed_iterations_of_call /
                                                       fn_seconds_per_iteration_expected()))
              }
              time_of_the_call_start <-  proc.time()[["elapsed"]]
              seconds_of_the_gap_before_each_call <<-  c(seconds_of_the_gap_before_each_call,
                                                         time_of_the_call_start -
                                                         time_at_the_end_of_the_last_step)
              timing <-  fn_time_sampling_of_Stan_model_for_each_layout(
                               init_object                  = init_object_of_call,
                               initial_values_of_the_chains = initial_values_of_the_chains,
                               Stan_data_list               = Stan_data_list_of_call,
                               layouts                      = layout_timed,
                               minimum_seconds_per_timing   = minimum_seconds_per_timing,
                               n_iterations_of_long_call    = n_iterations_of_call,
                               L_main                       = L_of_timed_calls,
                               JSON_text_of_Stan_data_list  = JSON_text_of_call,
                               nuisance_metric_of_the_timings  = nuisance_metric_of_the_timings,
                               pass_the_data_to_the_sampler_as_JSON_text_instead_of_a_file =
                                     pass_the_data_to_the_samplers_as_JSON_text_instead_of_files)
              time_at_the_end_of_the_last_step <<-  proc.time()[["elapsed"]]
              seconds_of_each_timed_call <<-  c(seconds_of_each_timed_call,
                                                time_at_the_end_of_the_last_step - time_of_the_call_start)
              seconds_of_timed_work_of_call <-  timing$total_seconds_of_timed_sampling_attempts[1]
              if (is.null(seconds_of_timed_work_of_call) || !is.finite(seconds_of_timed_work_of_call)) {
                    seconds_of_timed_work_of_call <-  timing$seconds_per_iteration[1] *
                                                       timing$n_iterations_of_long_call[1]
              }
              seconds_of_the_timed_iterations_of_each_call <<-  c(seconds_of_the_timed_iterations_of_each_call,
                                                                  seconds_of_timed_work_of_call)
              n_timed_calls_made <<-  n_timed_calls_made + 1
              timing_of_the_last_call <<-  timing
              n_attempts_of_call <-  timing$n_timed_sampling_attempts[1]
              if (is.null(n_attempts_of_call) || !is.finite(n_attempts_of_call)) n_attempts_of_call <-  1
              n_recorded_timed_sampling_attempts <<-  n_recorded_timed_sampling_attempts + n_attempts_of_call
              seconds_per_iteration_of_the_call_before <<-  timing$seconds_per_iteration[1]
              seconds_of_start_up_of_each_call_made <<-  c(seconds_of_start_up_of_each_call_made,
                                                           max(0, timing$seconds_spent_timing[1] -
                                                                  seconds_of_timed_work_of_call))
              return(timing$seconds_per_iteration[1] * L_of_timed_calls /
                     timing$L_of_the_timed_call[1])
        }

        ##
        ## ---- Paper 1's equations: one chain per CPU, 1 thread per chain, and the rule's N_chunks (a power of
        ##      two, at most the largest power of two of pieces that the N units allow):
        ##
        ## ## (one chain per CPU: whether the model's tape fits in its share of the cache)
        ## rule_at_one_chain_per_CPU <-  fn_compute_sampling_num_chunks_for_CPUs_of_this_process(
        ##                                     M_bytes_per_chain       = M_bytes_per_chain,
        ##                                     n_chains_sampling       = fn_n_chains_of_threads_per_chain(1),
        ##                                     n_threads_WCP_sampling  = 1,
        ##                                     gradient_type           = "autodiff")
        ## tape_fits_in_cache <-  (rule_at_one_chain_per_CPU$N_chunks <= 1)
        ## ## (one chain per physical core, the core's hardware threads being its within-chain threads)
        ## threads_per_core <-  if (tape_fits_in_cache) 1 else min(most_threads_per_chain,
        ##                                                         fn_read_hardware_threads_per_core())
        ## rule <-  fn_compute_sampling_num_chunks_for_CPUs_of_this_process(
        ##                M_bytes_per_chain       = M_bytes_per_chain,
        ##                n_chains_sampling       = fn_n_chains_of_threads_per_chain(threads_per_core),
        ##                n_threads_WCP_sampling  = threads_per_core,
        ##                gradient_type           = "autodiff")
        ## most_num_chunks <-  2^floor(log2(N_units))
        ## n_chains_sampling_from_equations <-  fn_n_chains_of_threads_per_chain(threads_per_core)
        ## n_threads_WCP_sampling_from_equations <-  threads_per_core
        ## num_chunks_sampling_from_equations <-  min(most_num_chunks, max(threads_per_core,
        ##                                                                 2^ceiling(log2(rule$N_chunks))))
        ## if (tape_fits_in_cache) num_chunks_sampling_from_equations <-  1
        ##
        # ## (every CPU used: one chain per CPU with 1 thread when the chains are not given, otherwise the most
        # ##  threads per chain, a power of two, that the CPUs allow for the chains given)
        # threads_per_chain_from_equations <-  1
        # if (!is.null(n_chains_given)) {
        #       threads_per_chain_from_equations <-  2^floor(log2(most_threads_per_chain))
        # }
        # rule <-  fn_compute_sampling_num_chunks_for_CPUs_of_this_process(
        #                M_bytes_per_chain       = M_bytes_per_chain,
        #                n_chains_sampling       = fn_n_chains_of_threads_per_chain(
        #                                                threads_per_chain_from_equations),
        #                n_threads_WCP_sampling  = threads_per_chain_from_equations,
        #                gradient_type           = "autodiff")
        ##  tape)
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
        ##
        ## ---- Every CPU used: the equation's threads per chain and N_chunks ---------------------------------------------------------
        ##
        ## (when the chains are not given: the equation's threads per chain, 2 for a Stan-like tape and 1
        ##  otherwise, and its N_chunks; otherwise the most threads per chain, a power of two, that the CPUs allow
        ##  for the chains given, with the equation's N_chunks at 1 thread per chain)
        threads_per_chain_from_equations <-  min(sampling_equation_of_18_Oct$n_threads_WCP_sampling,
                                                 2^floor(log2(most_threads_per_chain)))
        num_chunks_of_the_equation <-  sampling_equation_of_18_Oct$num_chunks_at_one_thread_per_chain /
                                       threads_per_chain_from_equations
        if (!is.null(n_chains_given)) {

              threads_per_chain_from_equations <-  2^floor(log2(most_threads_per_chain))
              num_chunks_of_the_equation <-  sampling_equation_of_18_Oct$num_chunks_at_one_thread_per_chain

        }
        rule <-  list(N_chunks = num_chunks_of_the_equation,
                      sampling_equation_of_18_Oct = sampling_equation_of_18_Oct)
        # most_num_chunks <-  Inf
        ## (at most N chunks: exactly N_chunks chunks run, none may be empty)
        most_num_chunks <-  N_units
        n_chains_sampling_from_equations <-  fn_n_chains_of_threads_per_chain(threads_per_chain_from_equations)
        n_threads_WCP_sampling_from_equations <-  threads_per_chain_from_equations
        ## (N_chunks: the rule's N_chunks, rounded up to a power of two, at least one chunk per thread)
        num_chunks_sampling_from_equations <-  2^ceiling(log2(rule$N_chunks))
        num_chunks_sampling_from_equations <-  max(threads_per_chain_from_equations,
                                                    num_chunks_sampling_from_equations)
        time_after_the_equations <-  proc.time()[["elapsed"]]
        ## (the lp_grad time and L and N_iter of the timed calls, at the equations' N_chunks)
        if (isTRUE(time_Paper_1_check) && !is.null(init_object)) {
              ## (the lp_grad time given, e.g. measured by the burn-in pick of the same model and data, or else
              ##  measured here, at the equations' sampling N_chunks, the layout the search starts from, never
              if (is.null(seconds_per_lp_grad_call) || !is.finite(seconds_per_lp_grad_call)) {
                    # Stan_data_list_of_lp_grad_timing <-  Stan_data_list
                    # Stan_data_list_of_lp_grad_timing$chunk_size <-  as.integer(N_units)
                    # Stan_data_list_of_lp_grad_timing$chunk_size <-
                    #       as.integer(ceiling(N_units / num_chunks_sampling_from_equations))
                    Stan_data_list_of_lp_grad_timing <-  fn_Stan_data_list_with_the_chunks_of_N_chunks(
                                                               Stan_data_list  = Stan_data_list,
                                                               N_chunks        =
                                                                     num_chunks_sampling_from_equations)
                    ## seconds_per_lp_grad_call <-  fn_seconds_per_lp_grad_call_of_compiled_Stan_model(
                    ##                                    model_so_file                   = model_so_file,
                    ##                                    JSON_text_of_Stan_data          =
                    ##                                          fn_JSON_text_of_Stan_data_list(
                    ##                                                Stan_data_list =
                    ##                                                      Stan_data_list_of_lp_grad_timing),
                    ## (the data's JSON text written once, here, and the lp_grad timing's text made from it by
                    ##  changing its chunk_size only: writing the JSON text twice, through a temporary file each
                    # if (is.null(JSON_text_of_Stan_data_list)) {
                    #       JSON_text_of_Stan_data_list <-  fn_JSON_text_of_Stan_data_list(
                    #                                             Stan_data_list = Stan_data_list)
                    # }
                    ##  init_object's data file, read instead of written again, when it is of the Stan data given;
                    ##  (b) the fit's own BridgeStan model, init_object$bs_model, used for the lp_grad timing when
                    ##  it was built from that same text with the same chunk_size, never a second model; (c) the
                    ##  repeated calls of the lp_grad timing made only when the search will run, the budget rule
                    ##  being decided from the timing's own estimate first)
                    data_file_of_init_object <-
                          fn_JSON_text_of_the_data_file_of_init_object_when_it_is_of_the_Stan_data_given(
                                init_object     = init_object,
                                Stan_data_list  = Stan_data_list)
                    if (is.null(JSON_text_of_Stan_data_list)) {
                          JSON_text_of_Stan_data_list <-  if (!is.null(data_file_of_init_object))
                                data_file_of_init_object$JSON_text else
                                fn_JSON_text_of_Stan_data_list(Stan_data_list = Stan_data_list)
                    }
                    model_of_the_fit <-  NULL
                    if (!is.null(data_file_of_init_object) &&
                        identical(as.integer(data_file_of_init_object$chunk_size),
                                  as.integer(Stan_data_list_of_lp_grad_timing$chunk_size)) &&
                        ## (and, for data with blocks, the same N_blocks: one chunk_size can have other blocks)
                        (is.null(Stan_data_list$N_blocks) ||
                         identical(as.integer(Stan_data_list$N_blocks),
                                   as.integer(Stan_data_list_of_lp_grad_timing$N_blocks))) &&
                        inherits(x = init_object$bs_model, what = "StanModel") &&
                        identical(normalizePath(init_object$bs_model$.__enclos_env__$private$lib,
                                                mustWork = FALSE),
                                  normalizePath(model_so_file, mustWork = FALSE))) {
                          model_of_the_fit <-  init_object$bs_model
                    }
                    ## (whether the search will run at an lp_grad time: the budget rule below, with the timed
                    ##  calls' L and N_iter from that lp_grad time)
                    fn_the_search_will_run_at_this_lp_grad_time <-  function(seconds_per_lp_grad_call_estimated) {

                          return(TRUE)
                          if (is.null(largest_fraction_of_a_typical_fit_without_chunking_spent_searching)) {
                                return(TRUE)
                          }
                          L_and_N_iter_estimated <-
                                fn_L_and_N_iter_of_each_timed_call_from_seconds_per_lp_grad_call(
                                seconds_per_lp_grad_call     = seconds_per_lp_grad_call_estimated,
                                target_seconds_of_long_call  = minimum_seconds_per_timing,
                                L_preferred                  = leapfrog_steps_per_timed_iteration,
                                L_may_be_lowered_when_the_fewest_iterations_last_longer_than_the_target =
                                      !budget_given)
                          seconds_of_a_typical_fit <-  iterations_of_a_typical_fit_without_chunking *
                                                       leapfrog_steps_per_timed_iteration *
                                                       seconds_per_lp_grad_call_estimated
                          iterations_of_a_timed_call <-  if (budget_given) fewest_iterations_of_a_timed_call else
                                                               L_and_N_iter_estimated$N_iter_of_long_call
                          seconds_of_a_timed_call <-  fn_seconds_of_start_up_of_a_call() +
                                                      (1 + iterations_of_a_timed_call) *
                                                      L_and_N_iter_estimated$L *
                                                      seconds_per_lp_grad_call_estimated
                          return(2 * seconds_of_a_timed_call <=
                                 largest_fraction_of_a_typical_fit_without_chunking_spent_searching *
                                 seconds_of_a_typical_fit)
                    }

                    # seconds_per_lp_grad_call <-  fn_seconds_per_lp_grad_call_of_compiled_Stan_model(
                    #                                    ...
                    #                              )$seconds_per_lp_grad_call
                    lp_grad_timing <-  fn_seconds_per_lp_grad_call_of_compiled_Stan_model(
                                             model_so_file                   = model_so_file,
                                             JSON_text_of_Stan_data          =
                                                   # fn_JSON_text_of_Stan_data_list_with_chunk_size(
                                                   #       JSON_text_of_Stan_data_list =
                                                   #             JSON_text_of_Stan_data_list,
                                                   #       chunk_size                  =
                                                   #             Stan_data_list_of_lp_grad_timing$chunk_size),
                                                   fn_JSON_text_of_Stan_data_list_with_the_chunks_of_N_chunks(
                                                         JSON_text_of_Stan_data_list  =
                                                               JSON_text_of_Stan_data_list,
                                                         N                            = N_units,
                                                         N_chunks                     =
                                                               num_chunks_sampling_from_equations),
                                             stop_if_library_already_loaded  = FALSE,
                                             n_nuisance_owned_by_NicoStan    = n_nuisance_owned_by_NicoStan,
                                             BridgeStan_model_already_built_from_the_same_JSON_text =
                                                   model_of_the_fit,
                                             fn_whether_the_repeated_calls_are_needed =
                                                   fn_the_search_will_run_at_this_lp_grad_time)
                    seconds_per_lp_grad_call <-  lp_grad_timing$seconds_per_lp_grad_call
                    data_file_of_the_fit_used <-  if (is.null(data_file_of_init_object)) "no" else "yes"
                    model_of_the_fit_used <-  if (is.null(model_of_the_fit)) "no" else "yes"
                    repeated_calls_of_the_lp_grad_timing_made <-  if (isTRUE(lp_grad_timing$repeated_calls_made))
                                                                       "yes" else "no"
                    seconds_of_the_lp_grad_step <-  proc.time()[["elapsed"]] - time_after_the_equations
              }
              ## (under a budget, L is never lowered: a model too slow for the budget gets the equations' layout
              ##  instead, see below)
              L_and_N_iter <-  fn_L_and_N_iter_of_each_timed_call_from_seconds_per_lp_grad_call(
                                     seconds_per_lp_grad_call     = seconds_per_lp_grad_call,
                                     target_seconds_of_long_call  = minimum_seconds_per_timing,
                                     L_preferred                  = leapfrog_steps_per_timed_iteration,
                                     L_may_be_lowered_when_the_fewest_iterations_last_longer_than_the_target =
                                           !budget_given)
              L_of_timed_calls <-  L_and_N_iter$L
              N_iter_of_timed_calls <-  L_and_N_iter$N_iter_of_long_call
        }
        ## (one chain per CPU and N_chunks = 1: the model's tape fits in its share of the cache, and the layout
        ##  is not checked)
        # tape_fits_in_cache <-  (threads_per_chain_from_equations == 1) &&
        #                        (num_chunks_sampling_from_equations <= 1)
        ## (one chain per CPU and N_chunks = 1, the layout not checked, only when the tape per chain also fits in
        ##  the L2 cache of one hardware thread, the L2 cache of a core over its hardware threads: on the ground
        ##  truth of every model, N, math backend and machine, N_chunks = 1 is then at most 1.1% slower than the
        ##  fastest; a larger tape whose share of the L3 cache fits on paper can still spill, e.g. the centred SV
        ##  with AVX2 at N = 10,000, 38% slower at N_chunks = 1, so its layout is searched from N_chunks = 1)
        time_before_the_L2_cache_read <-  proc.time()[["elapsed"]]
        L2_cache_of_the_first_CPU <-  fn_read_threads_per_core_and_L2_cache_size_of_first_CPU_of_this_process()
        seconds_of_the_L2_cache_read <-  proc.time()[["elapsed"]] - time_before_the_L2_cache_read
        time_at_the_end_of_the_last_step <-  proc.time()[["elapsed"]]
        L2_cache_bytes_per_hardware_thread <-  0
        if (!is.null(L2_cache_of_the_first_CPU)) {
              L2_cache_bytes_per_hardware_thread <-  L2_cache_of_the_first_CPU$L2_cache_size_bytes /
                                                     L2_cache_of_the_first_CPU$N_threads_per_core
        }
        # tape_fits_in_cache <-  (threads_per_chain_from_equations == 1) &&
        #                        (num_chunks_sampling_from_equations <= 1) &&
        #                        (M_bytes_per_chain <= L2_cache_bytes_per_hardware_thread)
        ##  neighbours are always timed, never 0 layouts: on 18 Oct the robust Student-t's AVX-512 library at
        ##  N = 50,000, its tape 0.42 MB per chain within the L2 cache of one hardware thread, got 180 x 1 / 1
        ##  with 0 layouts timed, 6.6% slower than the ground truth's fastest, 180 x 1 / 2)
        tape_fits_in_cache <-  FALSE
        ##
        ## ---- the same model without tape chunking (its original model), when given: at one thread per chain and
        ##      N_chunks = 1, both models are timed and the faster is used (untimed: the original model):
        ##
        model_without_chunking_given <-  !is.null(init_object_of_model_without_chunking) &&
                                         !is.null(Stan_data_list_of_model_without_chunking)
        seconds_per_draw_of_one_chunk_of_this_model <-  c()
        seconds_per_draw_of_model_without_chunking <-  c()
        if (model_without_chunking_given) {
              ## (the timing's data contract, N and chunk_size, which the original model does not read)
              if (is.null(Stan_data_list_of_model_without_chunking$N)) {
                    Stan_data_list_of_model_without_chunking$N <-  as.integer(N_units)
              }
              # Stan_data_list_of_model_without_chunking$chunk_size <-
              #       as.integer(Stan_data_list_of_model_without_chunking$N)
              Stan_data_list_of_model_without_chunking <-  fn_Stan_data_list_with_the_chunks_of_N_chunks(
                                                                 Stan_data_list  =
                                                                       Stan_data_list_of_model_without_chunking,
                                                                 N_chunks        = 1)
        }
        ## fn_seconds_per_draw_of_model_without_chunking <-  function(n_chains) {
        ##       seconds_per_iteration <-  fn_time_sampling_of_Stan_model_for_each_layout(
        ##             init_object                 = init_object_of_model_without_chunking,
        ##             Stan_data_list              = Stan_data_list_of_model_without_chunking,
        ##             layouts                     = data.frame( n_chains       = n_chains,
        ##                                                       n_threads_WCP  = 1,
        ##                                                       num_chunks     = 1),
        ##             minimum_seconds_per_timing  = minimum_seconds_per_timing,
        ##             n_iterations_of_long_call   = N_iter_of_timed_calls,
        ##             L_main                      = L_of_timed_calls
        ##       )$seconds_per_iteration[1]
        ##       return(seconds_per_iteration / n_chains)
        ## }
        ## (the original model's data as JSON text, written once when it is first timed)
        JSON_text_of_model_without_chunking <-  NULL
        fn_seconds_per_draw_of_model_without_chunking <-  function(n_chains) {

              if (is.null(JSON_text_of_model_without_chunking) &&
                  isTRUE(pass_the_data_to_the_samplers_as_JSON_text_instead_of_files)) {
                    JSON_text_of_model_without_chunking <<-  fn_JSON_text_of_Stan_data_list(
                                                                   Stan_data_list =
                                                                         Stan_data_list_of_model_without_chunking)
              }
              seconds_per_iteration <-  fn_seconds_per_iteration_of_one_layout(
                    init_object_of_call     = init_object_of_model_without_chunking,
                    Stan_data_list_of_call  = Stan_data_list_of_model_without_chunking,
                    layout_timed            = data.frame( n_chains       = n_chains,
                                                          n_threads_WCP  = 1,
                                                          num_chunks     = 1),
                    JSON_text_of_call       = JSON_text_of_model_without_chunking)
              return(seconds_per_iteration / n_chains)
        }

        ##
        ## ---- Paper 1's check (unless the tape fits in the cache): the threads per chain and N_chunks searched
        ##      together, every CPU used, the layouts compared by their seconds per draw over all chains:
        ##
        n_threads_WCP_sampling <-  n_threads_WCP_sampling_from_equations
        num_chunks_sampling <-  num_chunks_sampling_from_equations
        search_ended_by_the_budget <-  FALSE
        timings <-  NULL
        n_timings <-  0
        every_timing_of_the_sampling_search <-  NULL
        L_main_of_timings <-  NA
        ## (the tape fits in the cache and the original model is given: this model with one chunk and the original
        ##  model, each timed once, with one thread per chain)
        ## (the data's JSON text, given, or written once here when timed: each layout's data only change their
        ##  chunk_size)
        ## if (isTRUE(time_Paper_1_check) && !is.null(init_object) && is.null(JSON_text_of_Stan_data_list)) {
        ##       JSON_text_of_Stan_data_list <-  fn_JSON_text_of_Stan_data_list(Stan_data_list = Stan_data_list)
        ## }
        ## (written below, after the budget rule, and only when the search runs: a skipped search never needs it)
      #   ## (under a budget: when not even two of the shortest calls fit in the seconds left, Paper 1's equations
      #   ##  alone choose the layout, nothing timed)
      #   if (budget_given && isTRUE(time_Paper_1_check) && !is.null(init_object) &&
      #       (fn_seconds_left_of_the_budget() < 2 * fn_seconds_of_the_shortest_call())) {
      #         equations_alone_because_of_the_budget <-  TRUE
      # message(paste0("Sampling layout from Paper 1's equations alone (", n_chains_sampling_from_equations,
      #                        " chains x ", n_threads_WCP_sampling_from_equations, " threads x ",
      #                        num_chunks_sampling_from_equations, " chunks), nothing timed: ",
      #                        round(max(0, fn_seconds_left_of_the_budget()), 3), " s of the budget of ",
      # round(most_seconds_spent_choosing_the_sampling_layout, 3), " s left, less than two ",
      #                        "timed calls of ", L_of_timed_calls, " leapfrog steps"))
      #   }
        ##  equations' layout alone, untimed, was 57-186% slower than the ground truth at N = 500)

        ##  calls, costs at most largest_fraction_of_a_typical_fit_without_chunking_spent_searching of a typical
        ##  fit without chunking, its seconds from the lp_grad time (iterations_of_a_typical_fit_without_chunking
        ##  iterations of leapfrog_steps_per_timed_iteration leapfrog steps); otherwise Paper 1's equations alone
        ##  choose the layout, nothing timed. The lp_grad time is that of the equations' N_chunks, never measured
        ##  unchunked, so that a typical fit is if anything underestimated and the search skipped sooner. The move
        ##  threshold of the check and the tolerance of the checks are unchanged)
        seconds_of_a_typical_fit_without_chunking <-  NA_real_
        seconds_of_the_cheapest_search <-  NA_real_
        search_skipped_because_it_would_cost_more_than_its_share_of_a_typical_fit <-  FALSE
        # if (!is.null(largest_fraction_of_a_typical_fit_without_chunking_spent_searching) &&
        #     isTRUE(time_Paper_1_check) && !is.null(init_object) && !equations_alone_because_of_the_budget &&
        #     isTRUE(is.finite(seconds_per_lp_grad_call))) {
        #       seconds_of_a_typical_fit_without_chunking <-  iterations_of_a_typical_fit_without_chunking *
        #                                                     leapfrog_steps_per_timed_iteration *
        #                                                     seconds_per_lp_grad_call
        #       seconds_of_a_timed_call <-  if (budget_given) fn_seconds_of_the_shortest_call() else
        #             fn_seconds_of_start_up_of_a_call() +
        #             (1 + N_iter_of_timed_calls) * L_of_timed_calls * seconds_per_lp_grad_call
        #       seconds_of_the_cheapest_search <-  2 * seconds_of_a_timed_call
        #       if (seconds_of_the_cheapest_search >
        #           largest_fraction_of_a_typical_fit_without_chunking_spent_searching *
        #           seconds_of_a_typical_fit_without_chunking) {
        #             search_skipped_because_it_would_cost_more_than_its_share_of_a_typical_fit <-  TRUE
        #             equations_alone_because_of_the_budget <-  TRUE
        #             message(paste0("Sampling layout: search skipped: a typical fit of ",
        #                            signif(seconds_of_a_typical_fit_without_chunking, 3), " s; the equations' ",
        #                            "layout ", n_chains_sampling_from_equations, " x ",
        #                            n_threads_WCP_sampling_from_equations, " / ",
        #                            num_chunks_sampling_from_equations, "; any search would cost more than it ",
        #                            "can save (at least ",
        #                            signif(seconds_of_the_cheapest_search, 3), " s, more than ",
        #                            100 * largest_fraction_of_a_typical_fit_without_chunking_spent_searching,
        #                            "% of the typical fit)"))
        #       }
        # }
        if (isTRUE(time_Paper_1_check) && !is.null(init_object) && !equations_alone_because_of_the_budget &&
            is.null(JSON_text_of_Stan_data_list)) {
              # JSON_text_of_Stan_data_list <-  fn_JSON_text_of_Stan_data_list(Stan_data_list = Stan_data_list)
              ## (the fit's data file read when it is of the Stan data given, otherwise written here)
              data_file_of_init_object <-
                    fn_JSON_text_of_the_data_file_of_init_object_when_it_is_of_the_Stan_data_given(
                          init_object     = init_object,
                          Stan_data_list  = Stan_data_list)
              JSON_text_of_Stan_data_list <-  if (!is.null(data_file_of_init_object))
                    data_file_of_init_object$JSON_text else
                    fn_JSON_text_of_Stan_data_list(Stan_data_list = Stan_data_list)
        }
        if (model_without_chunking_given && tape_fits_in_cache && isTRUE(time_Paper_1_check) &&
            !is.null(init_object) && !equations_alone_because_of_the_budget) {
              L_main_of_timings <-  L_of_timed_calls
              n_chains_with_one_thread_per_chain <-  fn_n_chains_of_threads_per_chain(1)
              ## seconds_per_draw_of_one_chunk_of_this_model <-  fn_time_sampling_of_Stan_model_for_each_layout(
              ##       init_object                 = init_object,
              ##       Stan_data_list              = Stan_data_list,
              ##       layouts                     = data.frame(
              ##                                           n_chains       = n_chains_with_one_thread_per_chain,
              ##                                           n_threads_WCP  = 1,
              ##                                           num_chunks     = 1),
              ##       minimum_seconds_per_timing  = minimum_seconds_per_timing,
              ##       n_iterations_of_long_call   = N_iter_of_timed_calls,
              ##       L_main                      = L_of_timed_calls
              ## )$seconds_per_iteration[1] / n_chains_with_one_thread_per_chain
              seconds_per_draw_of_one_chunk_of_this_model <-  fn_seconds_per_iteration_of_one_layout(
                    init_object_of_call     = init_object,
                    Stan_data_list_of_call  = Stan_data_list,
                    layout_timed            = data.frame( n_chains       = n_chains_with_one_thread_per_chain,
                                                          n_threads_WCP  = 1,
                                                          num_chunks     = 1),
                    JSON_text_of_call       = JSON_text_of_Stan_data_list) / n_chains_with_one_thread_per_chain
              seconds_per_draw_of_model_without_chunking <-  fn_seconds_per_draw_of_model_without_chunking(
                                                                   n_chains = n_chains_with_one_thread_per_chain)
              n_timings <-  2
        }
        if (isTRUE(time_Paper_1_check) && !is.null(init_object) && !tape_fits_in_cache && (most_num_chunks > 1) &&
            !equations_alone_because_of_the_budget) {
              L_main_of_timings <-  L_of_timed_calls
              workers_kept_for_reuse <-  NULL
              if (isTRUE(keep_workers_for_reuse_in_check)) workers_kept_for_reuse <-  new.env()
              ## (the data's JSON text, written once: each layout's data file only changes its chunk_size)
              ## JSON_text_of_Stan_data_list <-  fn_JSON_text_of_Stan_data_list(Stan_data_list = Stan_data_list)
              ## (written above, or given)
              ## (the seconds per draw over all chains of each layout: its seconds per iteration, with every CPU
              ##  used, divided by its number of chains)
              fn_seconds_per_draw_of_each_layout <-  function(layouts) {

                    seconds_per_draw <-  numeric(nrow(layouts))
                    ## (the details of each timing: the call's wall seconds, the seconds per iteration timed by
                    ##  the chains themselves and from the wall seconds of the calls, whether the chains' own
                    ##  seconds were used, the timed iterations and the divergent transitions)
                    details_of_each_timing <-  list()
                    for (index in seq_len(nrow(layouts))) {
                          n_chains <-  fn_n_chains_of_threads_per_chain(layouts$threads_per_chain[index])
                          ## (timed in the sampler of the sampling phase: the chains independent, as in the
                          ##  sampling)
                          layout_timed <-  data.frame( n_chains       = n_chains,
                                                       n_threads_WCP  = layouts$threads_per_chain[index],
                                                       num_chunks     = layouts$num_chunks[index])
                          ## seconds_per_iteration <-  fn_time_sampling_of_Stan_model_for_each_layout(
                          ##       init_object                  = init_object,
                          ##       Stan_data_list               = Stan_data_list,
                          ##       layouts                      = layout_timed,
                          ##       minimum_seconds_per_timing   = minimum_seconds_per_timing,
                          ##       n_iterations_of_long_call    = N_iter_of_timed_calls,
                          ##       L_main                       = L_main_of_timings,
                          ##       JSON_text_of_Stan_data_list  = JSON_text_of_Stan_data_list
                          ## )$seconds_per_iteration[1]
                          ## (the same call, its iterations from the budget when one is given)
                          seconds_per_iteration <-  fn_seconds_per_iteration_of_one_layout(
                                init_object_of_call     = init_object,
                                Stan_data_list_of_call  = Stan_data_list,
                                layout_timed            = layout_timed,
                                JSON_text_of_call       = JSON_text_of_Stan_data_list)
                          ## seconds_per_iteration <-  fn_time_burnin_iteration_for_each_num_chunks(
                                                          ## init_object                = init_object,
                                                          ## Stan_data_list             = Stan_data_list,
                                                          ## n_chains_burnin            = n_chains,
                                                          ## n_threads_WCP_burnin        =
                                                                ## layouts$threads_per_chain[index],
                                                       ## num_chunks_candidates      = layouts$num_chunks[index],
                                                          ## n_rounds                   = 1,
                                                          ## n_timed_iterations         = n_timed_iterations,
                                                       ## minimum_seconds_per_timing = minimum_seconds_per_timing,
                                                          ## L_main                     = L_main_of_timings,
                                                          ## environment_of_workers_kept_for_reuse  =
                                                                ## workers_kept_for_reuse,
                                                          ## JSON_text_of_Stan_data_list            =
                                                                ## JSON_text_of_Stan_data_list
                                                    ## )$median_seconds_per_burnin_iteration[1]
                          seconds_per_draw[index] <-  seconds_per_iteration / n_chains
                          details_of_each_timing[[index]] <-  data.frame(
                                n_chains                               = n_chains,
                                seconds_of_the_call                    =
                                      timing_of_the_last_call$seconds_spent_timing[1],
                                seconds_per_iteration                  = seconds_per_iteration,
                                seconds_per_iteration_from_wall_seconds_of_the_calls =
                                      timing_of_the_last_call[[paste0("seconds_per_iteration_from_wall_",
                                                                      "seconds_of_the_calls")]][1],
                                timed_by_chains_seconds_of_the_sampler =
                                      timing_of_the_last_call$timed_by_chains_seconds_of_the_sampler[1],
                                n_iterations_of_long_call              =
                                      timing_of_the_last_call$n_iterations_of_long_call[1],
                                n_divergent_transitions_of_timed_calls =
                                      timing_of_the_last_call$n_divergent_transitions_of_timed_calls[1],
                                seconds_of_the_model_without_chunking_used  = FALSE)
                          attempt_columns <-  c(
                                "L_of_the_timed_call",
                                "seconds_cap",
                                "predicted_seconds_of_timed_call",
                                "actual_seconds_of_timed_call",
                                "in_band",
                                "used_longest_call_under_cap",
                                "shortest_call_exceeds_cap",
                                "n_timed_sampling_attempts",
                                "total_seconds_of_timed_sampling_attempts",
                                "total_wall_seconds_of_sampling_attempts",
                                "total_user_CPU_seconds_of_sampling_attempts",
                                "total_kernel_CPU_seconds_of_sampling_attempts",
                                "sampling_timing_attempts_URL_encoded_JSON")
                          for (column in attempt_columns) {
                                details_of_each_timing[[index]][[column]] <-
                                      timing_of_the_last_call[[column]][1]
                          }
                          ## (one thread per chain and N_chunks = 1, with the original model given: the original
                          ##  model timed too, and the seconds of the faster of the two models, by the means of
                          ##  their timings at this layout)
                          if (model_without_chunking_given && (layouts$threads_per_chain[index] == 1) &&
                              (layouts$num_chunks[index] == 1)) {
                                seconds_per_draw_of_one_chunk_of_this_model <<-
                                      c(seconds_per_draw_of_one_chunk_of_this_model, seconds_per_draw[index])
                                seconds_per_draw_of_model_without_chunking <<-
                                      c(seconds_per_draw_of_model_without_chunking,
                                        fn_seconds_per_draw_of_model_without_chunking(n_chains = n_chains))
                                if (mean(seconds_per_draw_of_model_without_chunking) <
                                    mean(seconds_per_draw_of_one_chunk_of_this_model)) {
                                      seconds_per_draw[index] <-
                                            utils::tail(seconds_per_draw_of_model_without_chunking, 1)
                                      name_of_the_detail <-  "seconds_of_the_model_without_chunking_used"
                                      details_of_each_timing[[index]][[name_of_the_detail]] <-  TRUE
                                }
                          }
                    }
                    attr(seconds_per_draw, "details_of_each_timing") <-  do.call(rbind, details_of_each_timing)
                    return(seconds_per_draw)
              }

              doubling_of_num_chunks_first <-
                    time_the_doubling_of_num_chunks_first_and_the_others_only_if_it_is_not_faster
              layout_search <-  tryCatch(
                    expr  = fn_search_threads_per_chain_and_num_chunks_by_doubling_and_halving(
                                  threads_per_chain_start    = n_threads_WCP_sampling_from_equations,
                                  num_chunks_start           = num_chunks_sampling_from_equations,
                                  fn_seconds_of_each_layout  = fn_seconds_per_draw_of_each_layout,
                                  most_threads_per_chain     = most_threads_per_chain,
                                  most_num_chunks            = most_num_chunks,
                                  fraction_faster_needed_to_move  = fraction_faster_needed_to_move,
                                  fn_another_timing_fits_in_the_budget  = fn_another_timing_fits_in_the_budget,
                                  fraction_slower_within_which_a_neighbour_is_timed_again_in_the_last_round =
                                        fraction_slower_within_which_a_neighbour_is_timed_again_in_the_last_round,
                                  time_the_doubling_after_a_doubling_of_num_chunks_within_the_move_threshold =
                                      time_the_doubling_after_a_doubling_of_num_chunks_within_the_move_threshold,
                                  time_the_doubling_of_num_chunks_first_and_the_others_only_if_it_is_not_faster =
                                        doubling_of_num_chunks_first,
                                  also_time_2_threads_per_chain_at_2_chunks_when_the_equations_say_no_chunking =
                                        FALSE),
                    error = function(condition) {

                          message(paste0("Sampling layout not checked (", conditionMessage(condition),
                                         "); Paper 1's equations are used."))
                          NULL
                    })

              if (!is.null(workers_kept_for_reuse)) {
                    fn_remove_workers_kept_for_reuse(
                          environment_of_workers_kept_for_reuse = workers_kept_for_reuse)
              }
              if (!is.null(layout_search)) {
                    search_ended_by_the_budget <-  isTRUE(layout_search$search_ended_by_the_budget)
                    n_threads_WCP_sampling <-  layout_search$threads_per_chain
                    num_chunks_sampling <-  layout_search$num_chunks
                    n_timings <-  layout_search$n_timings
                    every_timing_of_the_sampling_search <-  layout_search$every_timing_made
                    n_chains_of_each_layout_timed <-  vapply(X = layout_search$timings$threads_per_chain,
                                                             FUN = fn_n_chains_of_threads_per_chain,
                                                             FUN.VALUE = 0)
                    timings <-  data.frame(
                          n_chains_sampling                       = n_chains_of_each_layout_timed,
                          n_threads_WCP_sampling                  = layout_search$timings$threads_per_chain,
                          num_chunks                              = layout_search$timings$num_chunks,
                          mean_seconds_per_iteration              = layout_search$timings$mean_seconds *
                                                                    n_chains_of_each_layout_timed,
                          mean_seconds_per_draw_over_all_chains   = layout_search$timings$mean_seconds,
                          n_timings                               = layout_search$timings$n_timings)
                    ##
                    ## ---- (when asked) the layout found against one thread per chain and N_chunks = 1, timed
                    ##      when the check did not time it (and one more call fits in the budget), the faster
                    ##      used:
                    ##
                    if (isTRUE(compare_the_layout_found_with_one_thread_per_chain_and_N_chunks_1) &&
                        !((n_threads_WCP_sampling == 1) && (num_chunks_sampling == 1))) {
                          of_one_by_one <-  (timings$n_threads_WCP_sampling == 1) & (timings$num_chunks == 1)
                          # if (!any(of_one_by_one) &&
                          #     (is.null(fn_another_timing_fits_in_the_budget) ||
                          #      isTRUE(fn_another_timing_fits_in_the_budget()))) {
                          if (!any(of_one_by_one)) {
                                seconds_per_draw_of_one_by_one <-  fn_seconds_per_draw_of_each_layout(
                                      data.frame(threads_per_chain = 1, num_chunks = 1))
                                n_timings <-  n_timings + 1
                                n_chains_of_one_by_one <-  fn_n_chains_of_threads_per_chain(1)
                                timings <-  rbind(timings, data.frame(
                                      n_chains_sampling                      = n_chains_of_one_by_one,
                                      n_threads_WCP_sampling                 = 1,
                                      num_chunks                             = 1,
                                      mean_seconds_per_iteration             = seconds_per_draw_of_one_by_one *
                                                                               n_chains_of_one_by_one,
                                      mean_seconds_per_draw_over_all_chains  = seconds_per_draw_of_one_by_one,
                                      n_timings                              = 1))
                                of_one_by_one <-  (timings$n_threads_WCP_sampling == 1) &
                                                  (timings$num_chunks == 1)
                          }
                          of_found <-  (timings$n_threads_WCP_sampling == n_threads_WCP_sampling) &
                                       (timings$num_chunks == num_chunks_sampling)
                          if (any(of_one_by_one) && any(of_found) &&
                              (timings$mean_seconds_per_draw_over_all_chains[which(of_one_by_one)[1]] <
                               timings$mean_seconds_per_draw_over_all_chains[which(of_found)[1]])) {
                                n_threads_WCP_sampling <-  1
                                num_chunks_sampling <-  1
                          }
                    }
              }
        }
        ##
        ## ---- one thread per chain and N_chunks = 1, with the original model given: the original model, unless
        ##      this model with one chunk was faster in the timings at that layout:
        ##
        use_model_without_chunking <-  FALSE
        if (model_without_chunking_given && (n_threads_WCP_sampling == 1) && (num_chunks_sampling == 1)) {
              use_model_without_chunking <-  TRUE
              if ((length(seconds_per_draw_of_model_without_chunking) > 0) &&
                  (length(seconds_per_draw_of_one_chunk_of_this_model) > 0)) {
                    use_model_without_chunking <-  (mean(seconds_per_draw_of_model_without_chunking) <=
                                                    mean(seconds_per_draw_of_one_chunk_of_this_model))
              }
        }
        fn_mean_or_NA <-  function(seconds) if (length(seconds) == 0) NA else mean(seconds)
        n_chains_sampling <-  fn_n_chains_of_threads_per_chain(n_threads_WCP_sampling)
        ## message(paste0("Sampling layout chosen in ", round(proc.time()[["elapsed"]] - time_choice_start, 2),
        ##                " s: ", n_timings, " layouts timed (L = ", L_of_timed_calls, ", ",
        ##                if (is.null(N_iter_of_timed_calls)) "sized" else N_iter_of_timed_calls,
        ##                " timed iterations each)"))
        message(paste0("Sampling layout chosen in ", round(proc.time()[["elapsed"]] - time_choice_start, 2),
                       " s", if (budget_given) paste0(" (budget ",
                                                      round(most_seconds_spent_choosing_the_sampling_layout, 2),
                                                      " s)") else "",
                       ": ", n_timings, " layouts timed (L = ", L_of_timed_calls, ", ",
                       if (budget_given) "iterations from the budget" else if (is.null(N_iter_of_timed_calls))
                             "sized" else paste0(N_iter_of_timed_calls, " initial timed iterations per layout"),
                       ")",
                       if (search_ended_by_the_budget) "; the search ended by the budget" else ""))
        seconds_of_the_whole_choice <-  proc.time()[["elapsed"]] - time_choice_start
        seconds_of_the_equations <-  time_after_the_equations - time_choice_start - seconds_of_the_topology_read
        seconds_of_the_rest <-  seconds_of_the_whole_choice - seconds_of_the_topology_read -
                                seconds_of_the_equations - seconds_of_the_lp_grad_step -
                                seconds_of_the_L2_cache_read - sum(seconds_of_each_timed_call) -
                                sum(seconds_of_the_gap_before_each_call)
        message(paste0("  the seconds of each step: CPU topology ", round(seconds_of_the_topology_read, 3),
                       ", Paper 1's equations ", round(seconds_of_the_equations, 3),
                       ", the data's JSON text and the lp_grad timing ", round(seconds_of_the_lp_grad_step, 3),
                       " (data file of the fit used: ", data_file_of_the_fit_used, "; its model used: ",
                       model_of_the_fit_used, "; repeated calls made: ",
                       repeated_calls_of_the_lp_grad_timing_made, "), L2 cache read ",
                       round(seconds_of_the_L2_cache_read, 3), ", ", n_recorded_timed_sampling_attempts,
                       " recorded timed sampling attempts over ", length(seconds_of_each_timed_call), " layouts: ",
                       round(sum(seconds_of_each_timed_call), 3), " (their timed iterations ",
                       round(sum(seconds_of_the_timed_iterations_of_each_call), 3), "), the gaps before them ",
                       round(sum(seconds_of_the_gap_before_each_call), 3), ", the rest ",
                       round(seconds_of_the_rest, 3), " s"))
        if (length(seconds_of_each_timed_call) > 0) {

              message(paste0("  each timed call, its timed iterations and the gap before it (s): ",
                             paste0(round(seconds_of_each_timed_call, 3), " (",
                                    round(seconds_of_the_timed_iterations_of_each_call, 3), "; gap ",
                                    round(seconds_of_the_gap_before_each_call, 3), ")", collapse = ", ")))
        }

        return(list( n_chains_sampling                       = n_chains_sampling,
                     n_threads_WCP_sampling                  = n_threads_WCP_sampling,
                     num_chunks_sampling                     = num_chunks_sampling,
                     # chunk_size_sampling                     = ceiling(N_units / num_chunks_sampling),
                     chunk_size_sampling                     =
                           fn_chunk_size_and_number_of_blocks_from_N_and_N_chunks(
                                 N         = N_units,
                                 N_chunks  = num_chunks_sampling)$chunk_size,
                     use_model_without_chunking              = use_model_without_chunking,
                     mean_seconds_per_draw_of_one_chunk_of_this_model =
                           fn_mean_or_NA(seconds_per_draw_of_one_chunk_of_this_model),
                     mean_seconds_per_draw_of_model_without_chunking =
                           fn_mean_or_NA(seconds_per_draw_of_model_without_chunking),
                     n_chains_sampling_from_equations        = n_chains_sampling_from_equations,
                     n_threads_WCP_sampling_from_equations   = n_threads_WCP_sampling_from_equations,
                     num_chunks_sampling_from_equations      = num_chunks_sampling_from_equations,
                     tape_fits_in_cache                      = tape_fits_in_cache,
                     timings                                 = timings,
                     n_timings                               = n_timings,
                     every_timing_of_the_sampling_search     = every_timing_of_the_sampling_search,
                     M_bytes_per_chain                       = M_bytes_per_chain,
                     rule                                    = rule,
                     N_units                                 = N_units,
                     CPU_list                                = CPU_list,
                     L_main_of_timings                       = L_main_of_timings,
                     most_seconds_spent_choosing_the_sampling_layout =
                           if (budget_given) most_seconds_spent_choosing_the_sampling_layout else NA,
                     sampling_layout_from_the_equations_alone_because_of_the_budget =
                           equations_alone_because_of_the_budget,
                     sampling_search_ended_by_the_budget     = search_ended_by_the_budget,
                     n_timed_calls_of_the_sampling_choice    = n_timed_calls_made,
                     seconds_per_lp_grad_call                = if (is.null(seconds_per_lp_grad_call)) NA else
                                                                     seconds_per_lp_grad_call,
                     seconds_of_a_typical_fit_without_chunking  = seconds_of_a_typical_fit_without_chunking,
                     seconds_of_the_cheapest_search          = seconds_of_the_cheapest_search,
                     search_skipped_because_it_would_cost_more_than_its_share_of_a_typical_fit =
                           search_skipped_because_it_would_cost_more_than_its_share_of_a_typical_fit,
                     seconds_spent_choosing                  = proc.time()[["elapsed"]] - time_choice_start))

}




















