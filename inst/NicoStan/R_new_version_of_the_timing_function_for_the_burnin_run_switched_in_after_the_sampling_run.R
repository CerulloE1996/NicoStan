##
## ==== R_fn_time_burnin_and_sampling_layouts_of_Stan_model_on_each_math_backend.R ===============================
##
## ---- The fastest burn-in layout AND the fastest sampling layout of a Stan model (NicoStan's N / chunk_size data
##      contract for reduce_sum_static), separately for each math backend (Stan math, AVX2 and AVX-512) and
##      each N, with Paper 1's method -
##
## For each math backend and N:
##   (1) the model's library (built beforehand: a library that is missing, or older than its .stan file, stops the
##       timings at once; nothing is compiled here) is initialised with the arguments of its build;
##   (2) M_bytes/chain (the full data as one piece) gives Paper 1's equations' N_chunks
##       (fn_compute_sampling_num_chunks_for_CPUs_of_this_process()) for each number of threads per chain;
##   (3) the seconds of one lp_grad call give the leapfrog steps (L = 10) and the iterations of each timed call
##       (about target_seconds_of_timed_call; fn_L_and_N_iter_of_each_timed_call_from_seconds_per_lp_grad_call());
##   (4) the sampling layouts (n_chains_sampling chains, one per CPU: 180 on the local-HPC and 16 on the laptop,
##       as in Paper 1; N_threads/chain = 1): N_chunks = 1 (no chunking) and the equations' N_chunks times
##       2^(-3, ..., 3), doubled again while the largest is the fastest;
##       each timed by NicoStan's one-call timing in the sampler of the sampling phase (untimed, then timed
##       iterations in one C++ call, timed by the chains' own seconds;
##       fn_time_sampling_of_Stan_model_for_each_layout());
##   (5) the burn-in layouts (n_chains_burnin = 4 chains, as in Paper 1; N_threads/chain from
##       threads_per_chain_of_burnin_grid, at most n_CPUs / n_chains_burnin, i.e. WCP and chunking):
##       N_chunks = 1, N_chunks = N_threads/chain and the equations' N_chunks, each timed in NicoStan's
##       persistent burn-in worker,
##       whose chains wait for each other at every iteration, as in the burn-in
##       (fn_time_burnin_iteration_for_each_num_chunks(), one untimed iteration first).
## phases_to_time = "sampling" times the sampling layouts only (the preliminary AVX check: each math backend's own
## optimal sampling N_chunks); the real benchmarks time both phases (the default).
## Layouts are ranked by the total throughput TP_iter = N_chains N_iter / T_e (iterations per second over all
## chains; 1 / the seconds per draw over all chains). The counter snapshots (e.g. of Paper 1's pmc_stat) are taken
## just before and just after each timed call when fn_snapshot_hardware_counters is given. One CSV row is written
## per layout, as soon as it is timed. Before any timing, the predicted total time is printed, and the timings
## stop when it is more than most_minutes_of_everything. With dry_run = TRUE, only the checks of the libraries and
## data and the prediction are made (nothing is built, loaded or timed).
##
##
##
##
#### ---- the timings ------------------------------------------------------------------------------------------
#' Fastest burn-in and sampling layouts of a Stan model, for each math backend and N, with Paper 1's method
#'
#' @param arguments_of_build_of_each_math_backend A named list (e.g. Stan, AVX2, AVX512) of the arguments with
#'   which each backend's library was built: Stan_model_file_path, Stan_cpp_user_header, make_args and
#'   stanc_args.
#' @param Stan_data_list_of_each_N A named list (names: N) of the Stan data lists (made beforehand).
#' @param output_file The CSV file (one row per layout; rows are appended).
#' @param n_CPUs The CPUs of the timings (default: those of this process).
#' @param n_chains_burnin The burn-in chains (default 4).
#' @param n_chains_sampling The sampling chains with one thread each (default n_CPUs: 180 on the local-HPC, 16 on
#'   the laptop, as in Paper 1); with N_threads/chain threads each, n_chains_sampling / N_threads/chain chains, so
#'   that every CPU is used.
#' @param threads_per_chain_of_sampling_grid The N_threads/chain of the sampling layouts (default c(1, 2, 4):
#'   WCP and chunking are searched in the sampling phase too, never chunking only).
#' @param threads_per_chain_of_burnin_grid Default c(1, 2, 4, 8, 16, 32), and n_CPUs / n_chains_burnin.
#' @param phases_to_time The phases whose layouts are timed: "sampling" and / or "burnin" (default both; the
#'   preliminary AVX check uses "sampling" only).
#' @param multipliers_of_equations_num_chunks Default 2^(-3:3).
#' @param most_num_chunks_of_grid Default 2048.
#' @param fewest_num_chunks_of_grid The fewest N_chunks of every grid (default 1: the unchunked layouts too).
#' @param L_preferred The leapfrog steps of each iteration (default 10).
#' @param L_may_be_lowered_when_the_fewest_iterations_last_longer_than_the_target Default TRUE; FALSE keeps
#'   L_preferred for every math backend (the ground truth: throughputs compared at the same L).
#' @param target_seconds_of_timed_call Default 0.2.
#' @param minimum_seconds_of_timed_call Each timed call lasts at least this (default 0: no minimum); a shorter one
#'   is timed again with more iterations, which the case's later layouts of that phase then use.
#' @param predicted_seconds_of_setup_of_each_layout The start-up of each layout's timing, for the prediction
#'   (default 0.3).
#' @param most_minutes_of_everything Default 60.
#' @param fn_snapshot_hardware_counters A function without arguments that snapshots the counters and returns the
#'   number of its snapshot (default NULL: none).
#' @param dry_run TRUE: checks and prediction only (default FALSE).
#' @param initial_values_of_the_chains_of_each_N A named list (names: N) of the starting values of the sampling
#'   chains, as initial_values_of_the_chains of fn_time_sampling_of_Stan_model_for_each_layout() (default NULL:
#'   its default); the burn-in layouts use the first n_chains_burnin of them.
#' @param n_nuisance_owned_by_NicoStan_of_each_N For libraries in NicoStan's nuisance-block form: a named list
#'   (names: N) of the size of the nuisance block that NicoStan samples (default NULL: no such library).
#' @param nuisance_metric_of_the_sampling_timings nuisance_metric_of_the_timings of
#'   fn_time_sampling_of_Stan_model_for_each_layout(): "identity" (default) or "diagonal".
#' @param L_of_the_untimed_warm_up_iteration The leapfrog steps of each layout's untimed iteration before its
#'   timed iterations (default NULL: L, as the timed iterations; 1, since the untimed iteration only starts the
#'   threads, builds the tapes and touches the memory). Saved in every row.
#' @param seconds_of_the_short_call_that_screens_each_layout The seconds of a short timed call that every chunked
#'   layout gets first (default 0: none, every layout gets the full timed call). A layout whose short call is more
#'   than fraction_slower_than_the_fastest_so_far_above_which_a_layout_keeps_its_short_call_only slower than the
#'   fastest full call so far of its phase, N and math backend keeps its short call (its row says so); every other
#'   layout, and every unchunked layout, gets the full timed call.
#' @param fraction_slower_than_the_fastest_so_far_above_which_a_layout_keeps_its_short_call_only Default 0.3
#'   (30%). Sampling is compared by seconds per draw over all chains, burn-in by seconds per
#'   iteration (over every chain count timed so far in this R process). Every row of a layout that had a short
#'   call saves its measure (seconds per draw over all chains for sampling, seconds per iteration for burn-in).
#' @return The data frame of every layout timed (invisibly), or the prediction when dry_run = TRUE.
#' @noRd
## (no default arguments: every caller passes every argument)
fn_time_burnin_and_sampling_layouts_of_Stan_model_on_each_math_backend <-  function(
        arguments_of_build_of_each_math_backend,
        Stan_data_list_of_each_N,
        output_file,
        n_CPUs,
        n_chains_burnin,
        n_chains_sampling,
        threads_per_chain_of_sampling_grid,
        threads_per_chain_of_burnin_grid,
        phases_to_time,
        multipliers_of_equations_num_chunks,
        most_num_chunks_of_grid,
        fewest_num_chunks_of_grid,
        L_preferred,
        L_may_be_lowered_when_the_fewest_iterations_last_longer_than_the_target,
        target_seconds_of_timed_call,
        minimum_seconds_of_timed_call,
        predicted_seconds_of_setup_of_each_layout,
        most_minutes_of_everything,
        fn_snapshot_hardware_counters,
        dry_run,
        initial_values_of_the_chains_of_each_N,
        maximum_seconds_of_each_sampling_timed_call,
        n_nuisance_owned_by_NicoStan_of_each_N,
        nuisance_metric_of_the_sampling_timings,
        seconds_of_the_short_call_that_screens_each_layout,
        fraction_slower_than_the_fastest_so_far_above_which_a_layout_keeps_its_short_call_only,
        L_of_the_untimed_warm_up_iteration,
        mode_of_the_timings,
        fewest_seconds_of_each_ground_truth_timed_call,
        most_seconds_of_each_ground_truth_timed_call
) {

        ##
        ## ---- (1) every library exists and is newer than its .stan file (nothing is compiled here):
        ##
        for (math_backend in names(arguments_of_build_of_each_math_backend)) {
              stan_file <-  arguments_of_build_of_each_math_backend[[math_backend]]$Stan_model_file_path
              so_file <-  sub("\\.stan$", "_model.so", stan_file)
              if (!file.exists(stan_file) || !file.exists(so_file)) {
                    stop(paste0(math_backend, ": the library ", so_file, " (or its .stan file) is missing; ",
                                "build it first, on the CPUs not used for timings."))
              }
              if (file.info(so_file)$mtime < file.info(stan_file)$mtime) {
                    stop(paste0(math_backend, ": the library ", so_file, " is older than its .stan file; build ",
                                "it again first."))
              }
        }
        for (N_name in names(Stan_data_list_of_each_N)) {
              stopifnot(as.numeric(N_name) == Stan_data_list_of_each_N[[N_name]]$N)
        }
        ##
        ## ---- the layouts of the grids (before the equations, the largest number of them, for the prediction):
        ##
        threads_per_chain_of_burnin_grid <-  sort(unique(c(threads_per_chain_of_burnin_grid,
                                                           floor(n_CPUs / n_chains_burnin))))
        threads_per_chain_of_burnin_grid <-  threads_per_chain_of_burnin_grid[
                                                   threads_per_chain_of_burnin_grid * n_chains_burnin <= n_CPUs]
        n_num_chunks_per_number_of_threads <-  length(multipliers_of_equations_num_chunks) + 2 + 1
        stopifnot(n_chains_sampling <= n_CPUs, all(phases_to_time %in% c("sampling", "burnin")))
        threads_per_chain_of_sampling_layouts <-  c()
        # if ("sampling" %in% phases_to_time) threads_per_chain_of_sampling_layouts <-  1
        if ("sampling" %in% phases_to_time) {
              divides_the_sampling_chains <-  (n_chains_sampling %% threads_per_chain_of_sampling_grid == 0)
              threads_per_chain_of_sampling_layouts <-  threads_per_chain_of_sampling_grid[
                                                              divides_the_sampling_chains]
        }
        threads_per_chain_of_burnin_layouts <-  c()
        if ("burnin" %in% phases_to_time) threads_per_chain_of_burnin_layouts <-  threads_per_chain_of_burnin_grid
        n_layouts_of_each_N_and_backend <-  n_num_chunks_per_number_of_threads *
                                            (length(threads_per_chain_of_sampling_layouts) +
                                             length(threads_per_chain_of_burnin_layouts))
        n_cases <-  length(arguments_of_build_of_each_math_backend) * length(Stan_data_list_of_each_N)
        ## (each layout: one untimed iteration and the timed call, plus its start-up; each case: initialisation,
        ##  M_bytes/chain and the lp_grad timing, about 5 s)
        predicted_seconds_of_everything <-  n_cases * (5 + n_layouts_of_each_N_and_backend *
                                                       (max(2 * target_seconds_of_timed_call,
                                                            1.2 * minimum_seconds_of_timed_call) +
                                                        predicted_seconds_of_setup_of_each_layout))
        message(paste0("\033[36mPredicted total time: ", round(predicted_seconds_of_everything / 60, 1),
                       " minutes (", n_cases, " cases of math backend and N, at most ",
                       n_layouts_of_each_N_and_backend, " layouts each; phases: ",
                       paste(phases_to_time, collapse = " and "), "; burn-in N_threads/chain ",
                       paste(threads_per_chain_of_burnin_layouts, collapse = ", "), " with ", n_chains_burnin,
                       " chains; sampling N_threads/chain ",
                       paste(threads_per_chain_of_sampling_layouts, collapse = ", "), " with ", n_chains_sampling,
                       " / N_threads/chain chains).\033[0m"))
        if (predicted_seconds_of_everything > 60 * most_minutes_of_everything) {
              stop(paste0("The predicted total time is more than ", most_minutes_of_everything, " minutes."))
        }
        if (isTRUE(dry_run)) {
              return(list(predicted_minutes_of_everything = predicted_seconds_of_everything / 60,
                          n_cases = n_cases,
                          most_layouts_of_each_case = n_layouts_of_each_N_and_backend,
                          threads_per_chain_of_burnin_grid = threads_per_chain_of_burnin_grid,
                          n_chains_sampling = n_chains_sampling))
        }
        ##
        fn_snapshot_or_NA <-  function() {

              if (is.null(fn_snapshot_hardware_counters)) return(NA_real_)
              return(fn_snapshot_hardware_counters())
        }

        fn_write_rows <-  function(rows) {

              rows$saved_at <-  format(Sys.time(), "%Y-%m-%d %H:%M:%S")
              if (file.exists(output_file)) {
                    rows_saved <-  utils::read.csv(output_file, stringsAsFactors = FALSE)
                    for (column in setdiff(names(rows), names(rows_saved))) rows_saved[[column]] <-  NA
                    for (column in setdiff(names(rows_saved), names(rows))) rows[[column]] <-  NA
                    rows <-  rbind(rows_saved, rows[, names(rows_saved)])
              }
              utils::write.csv(rows, output_file, row.names = FALSE)
        }

        ## (the fastest burn-in seconds per iteration of the full calls saved by this R process for an N and math
        ##  backend: the calls of the other chain counts, made before this one in the same process, are in the
        ##  output file; rows saved before this process started are left out)
        time_this_R_process_started <-  Sys.time() - proc.time()[["elapsed"]] - 1
        fn_fastest_seconds_per_iteration_of_burnin_saved_by_this_process <-  function(N, math_backend) {

              if (!file.exists(output_file)) return(Inf)
              rows_saved <-  utils::read.csv(output_file, stringsAsFactors = FALSE)
              if (!all(c("phase", "saved_at") %in% names(rows_saved))) return(Inf)
              if (!("timed_with_the_short_call_only" %in% names(rows_saved))) {
                    rows_saved$timed_with_the_short_call_only <-  FALSE
              }
              rows <-  rows_saved[(rows_saved$phase %in% "burnin") &
                                  (rows_saved$N %in% N) &
                                  (rows_saved$math_backend %in% math_backend) &
                                  !(rows_saved$timed_with_the_short_call_only %in% TRUE) &
                                  (as.POSIXct(rows_saved$saved_at) >= time_this_R_process_started), ]
              if (nrow(rows) == 0) return(Inf)
              return(min(rows$seconds_per_iteration, na.rm = TRUE))
        }

        seconds_of_short_call <-  seconds_of_the_short_call_that_screens_each_layout
        screening_is_on <-  isTRUE(seconds_of_short_call > 0)
        most_ratio_to_the_fastest_for_the_full_call <-  1 +
              fraction_slower_than_the_fastest_so_far_above_which_a_layout_keeps_its_short_call_only
        ##
        ## (the ground truth: every full timed call lasts between fewest_seconds_of_each_ground_truth_timed_call
        ##  and most_seconds_of_each_ground_truth_timed_call; its iterations and L come from the layout's own
        ##  short call, aiming at the middle; a full call outside them is a hard error that stops everything)
        stopifnot(mode_of_the_timings %in% c("ground_truth", "not_ground_truth"))
        is_the_ground_truth <-  identical(mode_of_the_timings, "ground_truth")
        seconds_aimed_at_by_each_ground_truth_timed_call <-  (fewest_seconds_of_each_ground_truth_timed_call +
                                                              most_seconds_of_each_ground_truth_timed_call) / 2
        ## (no short calls in the ground truth, every call lasts 0.5 to 1 s; each layout's call is sized from
        ##  the measured speed of the layout timed before it, aiming at the middle of the window)
        # if (is_the_ground_truth && !screening_is_on) {
        #       stop(paste0("The ground truth needs seconds_of_the_short_call_that_screens_each_layout > 0: each ",
        #                   "layout's own short call sizes its full timed call."))
        # }
        fn_N_iter_and_L_of_a_ground_truth_timed_call <-  function(seconds_per_iteration_at_L, L, what) {

              seconds_per_leapfrog_step <-  seconds_per_iteration_at_L / L
              if (!isTRUE(seconds_per_leapfrog_step <= most_seconds_of_each_ground_truth_timed_call)) {
                    stop(paste0(what, ": one leapfrog step of every chain lasts ",
                                formatC(seconds_per_leapfrog_step, format = "f", digits = 2), " s, more than the ",
                                most_seconds_of_each_ground_truth_timed_call, " s that a ground-truth timed call ",
                                "may last."))
              }
              if (seconds_per_iteration_at_L <= most_seconds_of_each_ground_truth_timed_call) {
                    N_iter <-  max(1, round(seconds_aimed_at_by_each_ground_truth_timed_call /
                                            seconds_per_iteration_at_L))
                    return(list(N_iter = N_iter, L = L))
              }
              L_of_the_call <-  max(1, min(L, round(seconds_aimed_at_by_each_ground_truth_timed_call /
                                                    seconds_per_leapfrog_step)))
              return(list(N_iter = 1, L = L_of_the_call))
        }

        fn_stop_if_a_ground_truth_timed_call_lasted_too_long_or_too_short <-  function(seconds_of_the_call, what) {

              message(paste0("  ", what, ": ", formatC(seconds_of_the_call, format = "f", digits = 2), " s"))
              if (!is_the_ground_truth) return(invisible(NULL))
              if (!isTRUE(seconds_of_the_call >= fewest_seconds_of_each_ground_truth_timed_call) ||
                  !isTRUE(seconds_of_the_call <= most_seconds_of_each_ground_truth_timed_call)) {
                    stop(paste0(what, ": the ground-truth timed call lasted ",
                                formatC(seconds_of_the_call, format = "f", digits = 2), " s, outside ",
                                fewest_seconds_of_each_ground_truth_timed_call, " to ",
                                most_seconds_of_each_ground_truth_timed_call, " s."))
              }
              return(invisible(NULL))
        }

        ##
        ## (the N_chunks of a grid: none, one chunk per thread, and the equations' N_chunks times each multiplier;
        ##  powers of two, at least one chunk per thread, at most most_num_chunks)
        fn_num_chunks_of_grid <-  function(threads_per_chain, num_chunks_from_equations, most_num_chunks) {

              num_chunks <-  c(1, threads_per_chain,
                               num_chunks_from_equations * multipliers_of_equations_num_chunks)
              num_chunks <-  2^round(log2(pmax(1, num_chunks)))
              num_chunks <-  pmin(most_num_chunks, pmax(threads_per_chain, num_chunks))
              if (threads_per_chain == 1) num_chunks <-  c(1, num_chunks)
              ## (no layout with fewer than fewest_num_chunks_of_grid chunks, e.g. none unchunked at a large N,
              ##  where those layouts are far too slow to time)
              num_chunks <-  pmax(fewest_num_chunks_of_grid, num_chunks)
              return(sort(unique(num_chunks)))
        }

        ##
        all_rows <-  list()
        for (math_backend in names(arguments_of_build_of_each_math_backend)) {
          for (N_name in names(Stan_data_list_of_each_N)) {

              arguments_of_build <-  arguments_of_build_of_each_math_backend[[math_backend]]
              Stan_data_list <-  Stan_data_list_of_each_N[[N_name]]
              N <-  Stan_data_list$N
              Stan_data_list$chunk_size <-  as.integer(N)
              most_num_chunks <-  min(most_num_chunks_of_grid, 2^floor(log2(N)))
              so_file <-  sub("\\.stan$", "_model.so", arguments_of_build$Stan_model_file_path)
              so_file_mtime_before <-  file.info(so_file)$mtime
              message(paste0("\033[36m", math_backend, ", N = ", N, " ...\033[0m"))
              ##
              ## ---- (1) the library, initialised with the arguments of its build (never rebuilt):
              ##
              ## (a library in NicoStan's nuisance-block form: the size of its nuisance block, which NicoStan
              ##  samples; 0 for every other library)
              n_nuisance_owned_by_NicoStan <-  if (is.null(n_nuisance_owned_by_NicoStan_of_each_N)) 0 else
                                                     n_nuisance_owned_by_NicoStan_of_each_N[[N_name]]
              init_object <-  initialise_model( Model_type            = "Stan",
                                                sample_nuisance       = TRUE,
                                                Stan_data_list        = Stan_data_list,
                                                Stan_model_file_path  = arguments_of_build$Stan_model_file_path,
                                                Stan_cpp_user_header  = arguments_of_build$Stan_cpp_user_header,
                                                stanc_args            = arguments_of_build$stanc_args,
                                                make_args             = arguments_of_build$make_args,
                                                compile               = TRUE,
                                                n_nuisance_override   = if (n_nuisance_owned_by_NicoStan > 0)
                                                                              n_nuisance_owned_by_NicoStan else
                                                                              NULL)
              if (!identical(file.info(so_file)$mtime, so_file_mtime_before)) {
                    stop(paste0(math_backend, ": the library ", so_file, " was rebuilt by initialise_model(); ",
                                "build it beforehand with the same arguments."))
              }
              JSON_text <-  fn_JSON_text_of_Stan_data_list(Stan_data_list = Stan_data_list)
              ##
              ## ---- (2) M_bytes/chain, (3) L and N_iter from the seconds of one lp_grad call:
              ##
              M_bytes <-  fn_estimate_initial_M_bytes_per_chain(
                                model_so_file                 = init_object$model_so_file,
                                Stan_data_list                = Stan_data_list,
                                n_repeats                     = 1,
                                n_nuisance_owned_by_NicoStan  = n_nuisance_owned_by_NicoStan)$M_bytes_per_chain
              lp_grad <-  fn_seconds_per_lp_grad_call_of_compiled_Stan_model(
                                model_so_file                   = init_object$model_so_file,
                                JSON_text_of_Stan_data          = JSON_text,
                                stop_if_library_already_loaded  = FALSE,
                                n_nuisance_owned_by_NicoStan    = n_nuisance_owned_by_NicoStan)
              L_and_N_iter <-  fn_L_and_N_iter_of_each_timed_call_from_seconds_per_lp_grad_call(
                                     seconds_per_lp_grad_call     = lp_grad$seconds_per_lp_grad_call,
                                     target_seconds_of_long_call  = target_seconds_of_timed_call,
                                     L_preferred                  = L_preferred,
                                     L_may_be_lowered_when_the_fewest_iterations_last_longer_than_the_target =
                                     L_may_be_lowered_when_the_fewest_iterations_last_longer_than_the_target)
              ## (the iterations of each phase's timed calls: from the lp_grad time, then more if a call is
              ##  shorter than minimum_seconds_of_timed_call at full load)
              ## (a bounded call is sized from one iteration at full load, not a serial speed estimate)
              N_iter_of_sampling_calls <-  if (is.finite(maximum_seconds_of_each_sampling_timed_call)) 1 else
                                                L_and_N_iter$N_iter_of_long_call
              N_iter_of_burnin_calls <-  L_and_N_iter$N_iter_of_long_call
              ## (the untimed iteration's leapfrog steps, as saved in the rows)
              L_of_the_untimed_warm_up_iteration_of_the_rows <-  L_and_N_iter$L
              if (!is.null(L_of_the_untimed_warm_up_iteration)) {
                    L_of_the_untimed_warm_up_iteration_of_the_rows <-  L_of_the_untimed_warm_up_iteration
              }
              ## (the fastest full calls so far of this N and math backend, for the screening by a short call)
              fastest_seconds_per_draw_of_sampling_so_far <-  Inf
              fastest_seconds_per_iteration_of_burnin_so_far <-
                    fn_fastest_seconds_per_iteration_of_burnin_saved_by_this_process(N, math_backend)
              ##
              ## ---- (4) the sampling layouts (n_chains_sampling / N_threads/chain chains: every CPU is used):
              ##
              ## (the memory available now, from /proc/meminfo on Linux: a layout whose gradients cannot fit in
              ##  it is not timed, see below; Inf where it cannot be read)
              bytes_of_memory_available <-  Inf
              if (file.exists("/proc/meminfo")) {
                    line_of_memory_available <-  grep("^MemAvailable:", readLines("/proc/meminfo"), value = TRUE)
                    if (length(line_of_memory_available) == 1) {
                          bytes_of_memory_available <-  1024 * as.numeric(gsub("[^0-9]", "",
                                                                                line_of_memory_available))
                    }
              }
              for (threads_per_chain in threads_per_chain_of_sampling_layouts) {

                    # n_chains <-  n_chains_sampling
                    n_chains <-  n_chains_sampling / threads_per_chain
                    equations <-  fn_compute_sampling_num_chunks_for_CPUs_of_this_process(
                                        M_bytes_per_chain       = M_bytes,
                                        n_chains_sampling       = n_chains,
                                        n_threads_WCP_sampling  = threads_per_chain,
                                        gradient_type           = "autodiff")
                    num_chunks_from_equations <-  2^ceiling(log2(max(1, equations$N_chunks)))
                    num_chunks_of_grid <-  fn_num_chunks_of_grid(threads_per_chain, num_chunks_from_equations,
                                                                 most_num_chunks)
                    TP_iter_of_grid <-  c()
                    index_of_num_chunks <-  1
                    while (index_of_num_chunks <= length(num_chunks_of_grid)) {
                          layout <-  data.frame( n_chains       = n_chains,
                                                 n_threads_WCP  = threads_per_chain,
                                                 num_chunks     = num_chunks_of_grid[index_of_num_chunks])
                          ## (a layout whose chains' gradients need more than 70% of the memory available, i.e.
                          ##  N_chains x M_bytes_per_chain / N_chunks, is not timed, so that the machine never
                          ##  runs out of memory, e.g. the unchunked Stan math layouts with 180 chains at
                          ##  N = 200,000; it is named in the log and has no row)
                          bytes_of_the_gradients_of_the_layout <-  n_chains * M_bytes / layout$num_chunks
                          if (bytes_of_the_gradients_of_the_layout > 0.7 * bytes_of_memory_available) {
                                message(paste0("  ", n_chains, " x ", threads_per_chain, " / ", layout$num_chunks,
                                               ": not timed, its gradients need about ",
                                               round(bytes_of_the_gradients_of_the_layout / 1e9), " GB of the ",
                                               round(bytes_of_memory_available / 1e9), " GB available"))
                                index_of_num_chunks <-  index_of_num_chunks + 1
                                next
                          }
                          ## (the leapfrog steps of this layout's timed call: L, fewer only when one iteration at
                          ##  L lasts longer than maximum_seconds_of_each_sampling_timed_call; see below)
                          L_of_the_timed_call <-  L_and_N_iter$L
                          ## (the screening: a chunked layout first gets a short call, and keeps it only when it
                          ##  is more than the fraction slower than the fastest full call so far; otherwise it
                          ##  gets the full timed call below, its iterations sized from the short call's speed)
                          timed_with_the_short_call_only <-  FALSE
                          seconds_spent_on_the_short_call <-  0
                          ## (the short call's own measure, saved for every layout that had one, so that the short
                          ##  and the full call of each layout timed both ways can be compared)
                          seconds_per_draw_or_iteration_of_the_short_call <-  NA_real_
                          ## (in the ground truth every layout gets its short call, which sizes its full call; it
                          ##  screens as above only when a full call is already timed)
                          # if (screening_is_on && (layout$num_chunks > 1) &&
                          #     is.finite(fastest_seconds_per_draw_of_sampling_so_far)) {
                          if (screening_is_on &&
                              (is_the_ground_truth || ((layout$num_chunks > 1) &&
                                                       is.finite(fastest_seconds_per_draw_of_sampling_so_far)))) {
                                # N_iter_of_the_short_call <-  max(1, floor(N_iter_of_sampling_calls *
                                #                                           seconds_of_short_call /
                                #                                           target_seconds_of_timed_call))
                                ## (the ground truth: one untimed iteration of this layout at full load sizes its
                                ##  timed call; it is never a result, and every layout still gets its timed call)
                                N_iter_of_the_short_call <-  if (is_the_ground_truth) 1 else
                                                                  max(1, floor(N_iter_of_sampling_calls *
                                                                               seconds_of_short_call /
                                                                               target_seconds_of_timed_call))
                                seconds_spent_on_the_short_call <-  system.time({
                                short_timing <-  fn_time_sampling_of_Stan_model_for_each_layout(
                                      init_object                           = init_object,
                                      Stan_data_list                        = Stan_data_list,
                                      layouts                               = layout,
                                      time_layout_in_one_call               = TRUE,
                                      n_iterations_of_long_call             = N_iter_of_the_short_call,
                                      n_iterations_of_untimed_warm_up_call  = 0,  ## (no warm-up: Enzo)
                                      L_main                                = L_of_the_timed_call,
                                      JSON_text_of_Stan_data_list           = JSON_text,
                                      fn_snapshot_hardware_counters         = fn_snapshot_hardware_counters,
                                      initial_values_of_the_chains          =
                                            initial_values_of_the_chains_of_each_N[[as.character(N)]],
                                      nuisance_metric_of_the_timings        =
                                            nuisance_metric_of_the_sampling_timings,
                                      L_of_the_untimed_iterations           = L_of_the_untimed_warm_up_iteration)
                                })[["elapsed"]]
                                if (!isTRUE(all(short_timing$n_divergent_transitions_of_timed_calls == 0))) {
                                      stop(paste0(math_backend, ", N = ", N, ", sampling ", n_chains, " x ",
                                                  threads_per_chain, ", ", layout$num_chunks, " chunks: ",
                                                  short_timing$n_divergent_transitions_of_timed_calls,
                                                  " divergent transition(s) in the short call."))
                                }
                                seconds_per_draw_or_iteration_of_the_short_call <-
                                      short_timing$seconds_per_draw_over_all_chains
                                seconds_of_the_short_call <-  short_timing$seconds_per_iteration *
                                                              N_iter_of_the_short_call
                                message(paste0("  sampling ", n_chains, " x ", threads_per_chain, " / ",
                                               layout$num_chunks, ", short call: ",
                                               formatC(seconds_of_the_short_call, format = "f", digits = 2),
                                               " s"))
                                if ((layout$num_chunks > 1) &&
                                    isTRUE(short_timing$seconds_per_draw_over_all_chains >
                                           most_ratio_to_the_fastest_for_the_full_call *
                                           fastest_seconds_per_draw_of_sampling_so_far)) {
                                      timing <-  short_timing
                                      timed_with_the_short_call_only <-  TRUE
                                } else if (is_the_ground_truth) {
                                      ## (a second untimed sizing call of about 0.25 s, so that every chain runs
                                      ##  beside the others as in the timed call: one iteration alone overlaps too
                                      ##  little, e.g. 0.08 s against 0.23 s per iteration at full load, hierarchical
                                      ##  N = 50,000, 180 x 1 / 1; it is never a result)
                                      N_iter_of_the_second_sizing_call <-  max(1, round(0.25 /
                                                                                  short_timing$seconds_per_iteration))
                                      short_timing <-  fn_time_sampling_of_Stan_model_for_each_layout(
                                            init_object                           = init_object,
                                            Stan_data_list                        = Stan_data_list,
                                            layouts                               = layout,
                                            time_layout_in_one_call               = TRUE,
                                            n_iterations_of_long_call             = N_iter_of_the_second_sizing_call,
                                            n_iterations_of_untimed_warm_up_call  = 0,  ## (no warm-up: Enzo)
                                            L_main                                = L_of_the_timed_call,
                                            JSON_text_of_Stan_data_list           = JSON_text,
                                            fn_snapshot_hardware_counters         = fn_snapshot_hardware_counters,
                                            initial_values_of_the_chains          =
                                                  initial_values_of_the_chains_of_each_N[[as.character(N)]],
                                            nuisance_metric_of_the_timings        =
                                                  nuisance_metric_of_the_sampling_timings,
                                            L_of_the_untimed_iterations           = L_of_the_untimed_warm_up_iteration)
                                      message(paste0("  sampling ", n_chains, " x ", threads_per_chain, " / ",
                                                     layout$num_chunks, ", second untimed sizing call: ",
                                                     N_iter_of_the_second_sizing_call, " iterations, ",
                                                     formatC(short_timing$seconds_per_iteration *
                                                             N_iter_of_the_second_sizing_call, format = "f",
                                                             digits = 2), " s"))
                                      N_iter_and_L <-  fn_N_iter_and_L_of_a_ground_truth_timed_call(
                                            seconds_per_iteration_at_L  = short_timing$seconds_per_iteration,
                                            L                           = L_of_the_timed_call,
                                            what                        = paste0(math_backend, ", N = ", N,
                                                                                 ", sampling ", n_chains, " x ",
                                                                                 threads_per_chain, " / ",
                                                                                 layout$num_chunks))
                                      N_iter_of_sampling_calls <-  N_iter_and_L$N_iter
                                      L_of_the_timed_call <-  N_iter_and_L$L
                                } else if (isTRUE(short_timing$seconds_per_iteration > 0) &&
                                           !is.finite(maximum_seconds_of_each_sampling_timed_call)) {
                                      N_iter_of_sampling_calls <-
                                            max(1, ceiling(target_seconds_of_timed_call /
                                                           short_timing$seconds_per_iteration))
                                }
                          }
                          if (!timed_with_the_short_call_only) repeat {
                                timing <-  fn_time_sampling_of_Stan_model_for_each_layout(
                                                 init_object                           = init_object,
                                                 Stan_data_list                        = Stan_data_list,
                                                 layouts                               = layout,
                                                 time_layout_in_one_call               = TRUE,
                                                 n_iterations_of_long_call             = N_iter_of_sampling_calls,
                                                 n_iterations_of_untimed_warm_up_call  = 0,  ## (no warm-up: Enzo)
                                                 L_main                                = L_of_the_timed_call,
                                                 JSON_text_of_Stan_data_list           = JSON_text,
                                                 fn_snapshot_hardware_counters         =
                                                       fn_snapshot_hardware_counters,
                                                 initial_values_of_the_chains          =
                                                       initial_values_of_the_chains_of_each_N[[as.character(N)]],
                                                 nuisance_metric_of_the_timings        =
                                                       nuisance_metric_of_the_sampling_timings,
                                                 L_of_the_untimed_iterations           =
                                                       L_of_the_untimed_warm_up_iteration)
                                ## (not a single divergent transition is allowed: one makes the timing invalid and
                                ##  stops the timings of this model)
                                if (!isTRUE(all(timing$n_divergent_transitions_of_timed_calls == 0))) {
                                      stop(paste0(math_backend, ", N = ", N, ", sampling ", n_chains, " x ",
                                                  threads_per_chain, ", ", layout$num_chunks, " chunks: ",
                                                  timing$n_divergent_transitions_of_timed_calls,
                                                  " divergent transition(s) in the timed calls."))
                                }
                                seconds_of_timed_call <-  timing$seconds_per_iteration * N_iter_of_sampling_calls
                                ## (the ground truth: the call is printed with its seconds, and one outside 0.5 to
                                ##  1 s stops everything; it is never timed again with other iterations)
                                if (is_the_ground_truth) {
                                      fn_stop_if_a_ground_truth_timed_call_lasted_too_long_or_too_short(
                                            seconds_of_the_call  = seconds_of_timed_call,
                                            what                 = paste0(math_backend, ", N = ", N, ", sampling ",
                                                                          n_chains, " x ", threads_per_chain, " / ",
                                                                          layout$num_chunks, ", ",
                                                                          N_iter_of_sampling_calls, " iterations at L = ",
                                                                          L_of_the_timed_call))
                                      break
                                }
                                ## (a call too short for the chains' clock measures 0 s, e.g. one iteration of a
                                ##  model whose iteration lasts under a millisecond: the call is made 10 times
                                ##  longer and timed again, so that no number of iterations is divided by 0 s)
                                if (!isTRUE(seconds_of_timed_call > 0)) {
                                      N_iter_of_sampling_calls <-  10 * N_iter_of_sampling_calls
                                      next
                                }
                                if (seconds_of_timed_call > maximum_seconds_of_each_sampling_timed_call) {
                                      ## (one iteration is already longer than the most a timed call may last,
                                      ##  e.g. a layout with too few chunks at a large N: the layout is timed
                                      ##  again with fewer leapfrog steps, so that its call lasts about
                                      ##  target_seconds_of_timed_call; its seconds per iteration are scaled
                                      ##  back to L steps below, each step being one gradient)
                                      if (N_iter_of_sampling_calls == 1) {
                                            # if (L_of_the_timed_call == 1) {
                                            #       stop(paste0("One timed iteration at L = 1 exceeds ",
                                            #                   "maximum_seconds_of_each_sampling_timed_call; the ",
                                            #                   "measured call is retained in the run log."))
                                            # }
                                            ## (one iteration at L = 1 still lasts longer: that call is the
                                            ##  measurement of this layout, being longer than the call it
                                            ##  replaces, e.g. the unchunked layout at a large N)
                                            if (L_of_the_timed_call == 1) {
                                                  message(paste0("  ", n_chains, " x ", threads_per_chain, " / ",
                                                                 layout$num_chunks, ": one iteration at L = 1 ",
                                                                 "lasts ", formatC(seconds_of_timed_call,
                                                                                   format = "f", digits = 2),
                                                                 " s; that call is this layout's timing"))
                                                  break
                                            }
                                            L_of_the_timed_call <-  max(1, floor(L_of_the_timed_call *
                                                  target_seconds_of_timed_call / seconds_of_timed_call))
                                            next
                                      }
                                      N_iter_of_sampling_calls <-  max(1, floor(N_iter_of_sampling_calls *
                                            target_seconds_of_timed_call / seconds_of_timed_call))
                                      next
                                }
                                if (!isTRUE(seconds_of_timed_call < minimum_seconds_of_timed_call)) break
                                N_iter_of_sampling_calls <-  ceiling(1.1 * N_iter_of_sampling_calls *
                                                                     minimum_seconds_of_timed_call /
                                                                     seconds_of_timed_call)
                          }
                          ## (a layout timed with fewer leapfrog steps: its seconds per iteration and per draw
                          ##  scaled back to L steps, so that every layout of this N is compared at the same L)
                          timing$L_of_the_timed_call <-  L_of_the_timed_call
                          timing$seconds_of_the_timed_call <-  timing$seconds_per_iteration *
                                                               timing$n_iterations_of_long_call
                          if (L_of_the_timed_call != L_and_N_iter$L) {
                                scale_to_L <-  L_and_N_iter$L / L_of_the_timed_call
                                timing$seconds_per_iteration <-  timing$seconds_per_iteration * scale_to_L
                                timing$seconds_per_draw_over_all_chains <-
                                      timing$seconds_per_draw_over_all_chains * scale_to_L
                          }
                          ## (the next layout's timed call: its iterations from this layout's measured speed,
                          ##  so that it lasts about target_seconds_of_timed_call and is rarely timed twice)
                          # N_iter_of_sampling_calls <-  max(1, ceiling(target_seconds_of_timed_call /
                          #                                             timing$seconds_per_iteration))
                          seconds_aimed_at_by_the_next_call <-  if (is_the_ground_truth)
                                                                      seconds_aimed_at_by_each_ground_truth_timed_call
                                                                else target_seconds_of_timed_call
                          N_iter_of_sampling_calls <-  max(1, round(seconds_aimed_at_by_the_next_call /
                                                                    timing$seconds_per_iteration))
                          timing$phase <-  "sampling"
                          timing$timed_with_the_short_call_only <-  timed_with_the_short_call_only
                          timing$L_of_the_untimed_warm_up_iteration <-
                                L_of_the_untimed_warm_up_iteration_of_the_rows
                          timing$seconds_spent_on_the_short_call <-  seconds_spent_on_the_short_call
                          timing$seconds_per_draw_or_iteration_of_the_short_call <-
                                seconds_per_draw_or_iteration_of_the_short_call
                          if (!timed_with_the_short_call_only) {
                                fastest_seconds_per_draw_of_sampling_so_far <-
                                      min(fastest_seconds_per_draw_of_sampling_so_far,
                                          timing$seconds_per_draw_over_all_chains, na.rm = TRUE)
                          }
                          timing$TP_iter <-  1 / timing$seconds_per_draw_over_all_chains
                          timing$num_chunks_from_equations <-  num_chunks_from_equations
                          timing$math_backend <-  math_backend
                          timing$N <-  N
                          timing$L <-  L_and_N_iter$L
                          timing$seconds_per_lp_grad_call <-  lp_grad$seconds_per_lp_grad_call
                          timing$M_bytes_per_chain <-  M_bytes
                          fn_write_rows(timing)
                          all_rows[[length(all_rows) + 1]] <-  timing
                          TP_iter_of_grid <-  c(TP_iter_of_grid, timing$TP_iter)
                          ## (the largest N_chunks is the fastest: twice it is timed too)
                          if ((index_of_num_chunks == length(num_chunks_of_grid)) &&
                              (which.max(TP_iter_of_grid) == length(TP_iter_of_grid)) &&
                              (num_chunks_of_grid[index_of_num_chunks] < most_num_chunks)) {
                                num_chunks_of_grid <-  c(num_chunks_of_grid,
                                                         2 * num_chunks_of_grid[index_of_num_chunks])
                          }
                          index_of_num_chunks <-  index_of_num_chunks + 1
                    }

              }
              ##
              ## ---- (5) the burn-in layouts (the persistent burn-in worker, one untimed iteration first):
              ##
              for (threads_per_chain in threads_per_chain_of_burnin_layouts) {

                    equations <-  fn_compute_sampling_num_chunks_for_CPUs_of_this_process(
                                        M_bytes_per_chain       = M_bytes,
                                        n_chains_sampling       = n_chains_burnin,
                                        n_threads_WCP_sampling  = threads_per_chain,
                                        gradient_type           = "autodiff")
                    num_chunks_from_equations <-  2^ceiling(log2(max(1, equations$N_chunks)))
                    num_chunks_of_grid <-  fn_num_chunks_of_grid(threads_per_chain, num_chunks_from_equations,
                                                                 most_num_chunks)
                    TP_iter_of_grid <-  c()
                    index_of_num_chunks <-  1
                    while (index_of_num_chunks <= length(num_chunks_of_grid)) {
                          num_chunks <-  num_chunks_of_grid[index_of_num_chunks]
                          ## (the screening, as for the sampling layouts: by seconds per iteration)
                          timed_with_the_short_call_only <-  FALSE
                          seconds_spent_on_the_short_call <-  0
                          seconds_per_draw_or_iteration_of_the_short_call <-  NA_real_
                          ## (the leapfrog steps of this layout's burn-in call: L, fewer only in the ground truth
                          ##  when one iteration at L lasts longer than a ground-truth timed call may last)
                          L_of_the_burnin_call <-  L_and_N_iter$L
                          # if (screening_is_on && (num_chunks > 1) &&
                          #     is.finite(fastest_seconds_per_iteration_of_burnin_so_far)) {
                          if (screening_is_on &&
                              (is_the_ground_truth || ((num_chunks > 1) &&
                                                       is.finite(fastest_seconds_per_iteration_of_burnin_so_far)))) {
                                # N_iter_of_the_short_call <-  max(1, floor(N_iter_of_burnin_calls *
                                                                          # seconds_of_short_call /
                                                                          # target_seconds_of_timed_call))
                                N_iter_of_the_short_call <-  if (is_the_ground_truth) 1 else
                                                                  max(1, floor(N_iter_of_burnin_calls *
                                                                               seconds_of_short_call /
                                                                               target_seconds_of_timed_call))
                                snapshot_before <-  fn_snapshot_or_NA()
                                seconds_spent_on_the_short_call <-  system.time({
                                burnin_timing <-  fn_time_burnin_iteration_for_each_num_chunks(
                                      init_object                         = init_object,
                                      Stan_data_list                      = Stan_data_list,
                                      n_chains_burnin                     = n_chains_burnin,
                                      n_threads_WCP_burnin                = threads_per_chain,
                                      num_chunks_candidates               = num_chunks,
                                      n_rounds                            = 1,
                                      n_timed_iterations                  = N_iter_of_the_short_call,
                                      minimum_seconds_per_timing          = 0,
                                      L_main                              = L_and_N_iter$L,
                                      JSON_text_of_Stan_data_list         = JSON_text,
                                      n_untimed_iterations_before_timing  = 1,
                                      initial_values_of_the_chains        =
                                            initial_values_of_the_chains_of_each_N[[as.character(N)]],
                                      L_of_the_untimed_iterations_before_timing  =
                                            L_of_the_untimed_warm_up_iteration)
                                })[["elapsed"]]
                                snapshot_after <-  fn_snapshot_or_NA()
                                n_divergent_transitions_of_burnin_calls <-
                                      burnin_timing$n_divergent_transitions_of_timed_iterations[1]
                                n_transitions_of_burnin_calls <-
                                      burnin_timing$n_transitions_of_timed_iterations[1]
                                if (!isTRUE(n_divergent_transitions_of_burnin_calls == 0)) {
                                      stop(paste0(math_backend, ", N = ", N, ", burn-in ", n_chains_burnin, " x ",
                                                  threads_per_chain, ", ", num_chunks, " chunks: ",
                                                  n_divergent_transitions_of_burnin_calls,
                                                  " divergent transition(s) in the short call."))
                                }
                                seconds_per_iteration_of_the_short_call <-
                                      burnin_timing$median_seconds_per_burnin_iteration[1]
                                seconds_per_draw_or_iteration_of_the_short_call <-
                                      seconds_per_iteration_of_the_short_call
                                message(paste0("  burn-in ", n_chains_burnin, " x ", threads_per_chain, " / ",
                                               num_chunks, ", short call: ",
                                               formatC(seconds_per_iteration_of_the_short_call *
                                                       N_iter_of_the_short_call, format = "f", digits = 2), " s"))
                                if ((num_chunks > 1) &&
                                    isTRUE(seconds_per_iteration_of_the_short_call >
                                           most_ratio_to_the_fastest_for_the_full_call *
                                           fastest_seconds_per_iteration_of_burnin_so_far)) {
                                      timed_with_the_short_call_only <-  TRUE
                                      seconds_per_iteration <-  seconds_per_iteration_of_the_short_call
                                      wall_seconds <-  seconds_spent_on_the_short_call
                                      N_iter_of_burnin_calls <-  N_iter_of_the_short_call
                                } else if (is_the_ground_truth) {
                                      N_iter_and_L <-  fn_N_iter_and_L_of_a_ground_truth_timed_call(
                                            seconds_per_iteration_at_L  = seconds_per_iteration_of_the_short_call,
                                            L                           = L_and_N_iter$L,
                                            what                        = paste0(math_backend, ", N = ", N,
                                                                                 ", burn-in ", n_chains_burnin, " x ",
                                                                                 threads_per_chain, " / ", num_chunks))
                                      N_iter_of_burnin_calls <-  N_iter_and_L$N_iter
                                      L_of_the_burnin_call <-  N_iter_and_L$L
                                } else if (isTRUE(seconds_per_iteration_of_the_short_call > 0)) {
                                      N_iter_of_burnin_calls <-
                                            max(1, ceiling(target_seconds_of_timed_call /
                                                           seconds_per_iteration_of_the_short_call))
                                }
                          }
                          if (!timed_with_the_short_call_only) repeat {
                          snapshot_before <-  fn_snapshot_or_NA()
                          wall_seconds <-  system.time(
                                burnin_timing <-  fn_time_burnin_iteration_for_each_num_chunks(
                                                        init_object                         = init_object,
                                                        Stan_data_list                      = Stan_data_list,
                                                        n_chains_burnin                     = n_chains_burnin,
                                                        n_threads_WCP_burnin                = threads_per_chain,
                                                        num_chunks_candidates               = num_chunks,
                                                        n_rounds                            = 1,
                                                        n_timed_iterations                  =
                                                              N_iter_of_burnin_calls,
                                                        minimum_seconds_per_timing          = 0,
                                                        # L_main                              = L_and_N_iter$L,
                                                        L_main                              = L_of_the_burnin_call,
                                                        JSON_text_of_Stan_data_list         = JSON_text,
                                                        n_untimed_iterations_before_timing  = 1,
                                                        initial_values_of_the_chains        =
                                                              initial_values_of_the_chains_of_each_N[[
                                                                    as.character(N)]],
                                                        L_of_the_untimed_iterations_before_timing  =
                                                              L_of_the_untimed_warm_up_iteration)
                          )[["elapsed"]]
                          snapshot_after <-  fn_snapshot_or_NA()
                          ## (not a single divergent transition is allowed: the burn-in timing stops at the first,
                          ##  in its untimed and its timed iterations; its count of them is saved in every row)
                          n_divergent_transitions_of_burnin_calls <-
                                burnin_timing$n_divergent_transitions_of_timed_iterations[1]
                          n_transitions_of_burnin_calls <-  burnin_timing$n_transitions_of_timed_iterations[1]
                          if (!isTRUE(n_divergent_transitions_of_burnin_calls == 0)) {
                                stop(paste0(math_backend, ", N = ", N, ", burn-in ", n_chains_burnin, " x ",
                                            threads_per_chain, ", ", num_chunks, " chunks: ",
                                            n_divergent_transitions_of_burnin_calls,
                                            " divergent transition(s) in the timed calls."))
                          }
                          seconds_per_iteration <-  burnin_timing$median_seconds_per_burnin_iteration[1]
                          seconds_of_timed_call <-  seconds_per_iteration * N_iter_of_burnin_calls
                          ## (the ground truth: the call is printed with its seconds, one outside 0.5 to 1 s stops
                          ##  everything; a call with fewer leapfrog steps is scaled back to L steps)
                          if (is_the_ground_truth) {
                                fn_stop_if_a_ground_truth_timed_call_lasted_too_long_or_too_short(
                                      seconds_of_the_call  = seconds_of_timed_call,
                                      what                 = paste0(math_backend, ", N = ", N, ", burn-in ",
                                                                    n_chains_burnin, " x ", threads_per_chain, " / ",
                                                                    num_chunks, ", ", N_iter_of_burnin_calls,
                                                                    " iterations at L = ", L_of_the_burnin_call))
                                seconds_per_iteration <-  seconds_per_iteration * L_and_N_iter$L / L_of_the_burnin_call
                                break
                          }
                          if (!isTRUE(seconds_of_timed_call < minimum_seconds_of_timed_call)) break
                          N_iter_of_burnin_calls <-  ceiling(1.1 * N_iter_of_burnin_calls *
                                                             minimum_seconds_of_timed_call /
                                                             seconds_of_timed_call)
                          }
                          timing <-  data.frame( phase                               = "burnin",
                                                 n_chains                            = n_chains_burnin,
                                                 n_threads_WCP                       = threads_per_chain,
                                                 num_chunks                          = num_chunks,
                                                 seconds_per_iteration               = seconds_per_iteration,
                                                 seconds_per_draw_over_all_chains    = seconds_per_iteration /
                                                                                       n_chains_burnin,
                                                 TP_iter                             = n_chains_burnin /
                                                                                       seconds_per_iteration,
                                                 n_iterations_of_long_call           = N_iter_of_burnin_calls,
                                                 seconds_spent_timing                = wall_seconds,
                                                 n_divergent_transitions_of_timed_calls =
                                                       n_divergent_transitions_of_burnin_calls,
                                                 n_transitions_of_timed_calls        =
                                                       n_transitions_of_burnin_calls,
                                                 counter_snapshot_before_long_call   = snapshot_before,
                                                 counter_snapshot_after_long_call    = snapshot_after,
                                                 num_chunks_from_equations           = num_chunks_from_equations,
                                                 math_backend                        = math_backend,
                                                 N                                   = N,
                                                 L                                   = L_and_N_iter$L,
                                                 seconds_per_lp_grad_call            =
                                                       lp_grad$seconds_per_lp_grad_call,
                                                 M_bytes_per_chain                   = M_bytes,
                                                 timed_with_the_short_call_only      =
                                                       timed_with_the_short_call_only,
                                                 L_of_the_untimed_warm_up_iteration  =
                                                       L_of_the_untimed_warm_up_iteration_of_the_rows,
                                                 seconds_spent_on_the_short_call     =
                                                       seconds_spent_on_the_short_call,
                                                 seconds_per_draw_or_iteration_of_the_short_call =
                                                       seconds_per_draw_or_iteration_of_the_short_call,
                                                 stringsAsFactors                    = FALSE)
                          if (!timed_with_the_short_call_only) {
                                fastest_seconds_per_iteration_of_burnin_so_far <-
                                      min(fastest_seconds_per_iteration_of_burnin_so_far, seconds_per_iteration,
                                          na.rm = TRUE)
                          }
                          fn_write_rows(timing)
                          all_rows[[length(all_rows) + 1]] <-  timing
                          ## (the next layout's timed call: its iterations from this layout's measured speed)
                          # N_iter_of_burnin_calls <-  max(1, ceiling(target_seconds_of_timed_call /
                          #                                           seconds_per_iteration))
                          ## (never divided by 0 s: a layout whose timing measured 0 s gives the next one 10
                          ##  times its iterations)
                          # N_iter_of_burnin_calls <-  if (isTRUE(seconds_per_iteration > 0)) {
                          #                                  max(1, ceiling(target_seconds_of_timed_call /
                          #                                                 seconds_per_iteration))
                          #                            } else 10 * N_iter_of_burnin_calls
                          ## (the ground truth aims at the middle of its 0.5 to 1 s window)
                          seconds_aimed_at_by_the_next_call <-  if (is_the_ground_truth)
                                                                      seconds_aimed_at_by_each_ground_truth_timed_call
                                                                else target_seconds_of_timed_call
                          N_iter_of_burnin_calls <-  if (isTRUE(seconds_per_iteration > 0)) {
                                                           max(1, round(seconds_aimed_at_by_the_next_call /
                                                                        seconds_per_iteration))
                                                     } else 10 * N_iter_of_burnin_calls
                          TP_iter_of_grid <-  c(TP_iter_of_grid, timing$TP_iter)
                          if ((index_of_num_chunks == length(num_chunks_of_grid)) &&
                              (which.max(TP_iter_of_grid) == length(TP_iter_of_grid)) &&
                              (num_chunks < most_num_chunks)) {
                                num_chunks_of_grid <-  c(num_chunks_of_grid, 2 * num_chunks)
                          }
                          index_of_num_chunks <-  index_of_num_chunks + 1
                    }

              }

          }
        }
        ##
        return(invisible(all_rows))

}





















