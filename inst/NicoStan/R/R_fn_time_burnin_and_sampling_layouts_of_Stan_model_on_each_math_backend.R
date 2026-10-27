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
##   (4) the sampling layouts (n_chains_sampling chains; N_threads/chain from threads_per_chain_of_sampling_grid,
##       at most n_CPUs / n_chains_sampling): the N_chunks of num_chunks_of_each_listed_N for the case's N (the
##       list of the next listed N up; the largest listed N's list above it), at least N_threads/chain, doubled
##       again while the largest timed is within 10% of the fastest;
##       each timed by NicoStan's one-call timing in the sampler of the sampling phase (untimed, then timed
##       iterations in one C++ call, timed by the chains' own seconds;
##       fn_time_sampling_of_Stan_model_for_each_layout());
##   (5) the burn-in layouts (n_chains_burnin chains; N_threads/chain from threads_per_chain_of_burnin_grid, at
##       most n_CPUs / n_chains_burnin, i.e. WCP and chunking): the same N_chunks as the sampling layouts, each
##       timed in NicoStan's persistent burn-in worker,
##       whose chains wait for each other at every iteration, as in the burn-in
##       (fn_time_burnin_iteration_for_each_num_chunks(), one untimed iteration first).
## An AUTO-CHUNKING library takes each layout's N_chunks as N_chunks_of_auto_chunking, its model's reduce_sum
## the same N_chunks (both on) or 1 chunk (AUTO-CHUNKING alone, 1 thread per chain), in both phases.
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
#' @param n_chains_sampling The sampling chains, the same at every N_threads/chain of the sampling grid.
#' @param threads_per_chain_of_sampling_grid The N_threads/chain of the sampling layouts (those with
#'   n_chains_sampling x N_threads/chain <= n_CPUs are timed).
#' @param threads_per_chain_of_burnin_grid The N_threads/chain of the burn-in layouts (those with
#'   n_chains_burnin x N_threads/chain <= n_CPUs are timed).
#' @param phases_to_time The phases whose layouts are timed: "sampling" and / or "burnin" (default both; the
#'   preliminary AVX check uses "sampling" only).
#' @param num_chunks_of_each_listed_N A named list (names: N) of the N_chunks of the grids at each listed N; a
#'   case's N takes the list of the next listed N up, and an N above the largest listed N takes its list.
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
#' @param names_of_the_nuisance_blocks_of_a_regular_library For libraries NOT in NicoStan's nuisance-block form:
#'   the names of the model's nuisance blocks (e.g. "w"), as declared in its parameters block, or NULL (the first
#'   declaration, as before). The named blocks must be the library's first unconstrained coordinates (a regular
#'   library cannot reorder them); their size is read from the library's own unconstrained names and given to
#'   initialise_model() as n_nuisance_override, so that nothing is rewritten or rebuilt.
#' @param nuisance_metric_of_the_sampling_timings nuisance_metric_of_the_timings of
#'   fn_time_sampling_of_Stan_model_for_each_layout(): "identity" (default) or "diagonal".
#' @param L_of_the_untimed_warm_up_iteration The leapfrog steps of each layout's untimed iteration before its
#'   only starts the threads, builds the tapes and touches the memory). Saved in every row.
#' @param seconds_of_the_short_call_that_screens_each_layout The seconds of a short timed call that every chunked
#'   layout gets first (default 0: none, every layout gets the full timed call). A layout whose short call is more
#'   than fraction_slower_than_the_fastest_so_far_above_which_a_layout_keeps_its_short_call_only slower than the
#'   fastest full call so far of its phase, N and math backend keeps its short call (its row says so); every other
#'   layout, and every unchunked layout, gets the full timed call.
#' @param fraction_slower_than_the_fastest_so_far_above_which_a_layout_keeps_its_short_call_only Default 0.3
#'   iteration (over every chain count timed so far in this R process). Every row of a layout that had a short
#'   call saves its measure (seconds per draw over all chains for sampling, seconds per iteration for burn-in).
#' @return The data frame of every layout timed (invisibly), or the prediction when dry_run = TRUE.
#' @noRd
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
        num_chunks_of_each_listed_N,
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
        names_of_the_nuisance_blocks_of_a_regular_library,
        nuisance_metric_of_the_sampling_timings,
        seconds_of_the_short_call_that_screens_each_layout,
        fraction_slower_than_the_fastest_so_far_above_which_a_layout_keeps_its_short_call_only,
        L_of_the_untimed_warm_up_iteration,
        mode_of_the_timings,
        fewest_seconds_of_each_ground_truth_timed_call,
        most_seconds_of_each_ground_truth_timed_call,
        most_seconds_of_a_ground_truth_timed_call_of_one_leapfrog_step
) {

        ##
        ## ---- (1) every library exists and is newer than its .stan file (nothing is compiled here):
        ##
        for (math_backend in names(arguments_of_build_of_each_math_backend)) {
              stan_file <-  arguments_of_build_of_each_math_backend[[math_backend]]$Stan_model_file_path
              ## (an AUTO-CHUNKING library is built from the cell's own .stan file, auto_chunking = TRUE in its
              ##  record; a copy of the .stan file named _auto_chunked is the withdrawn route)
              if (grepl("_auto_chunked[.]stan$", stan_file)) {
                    stop(paste0(math_backend, ": ", stan_file, " is a copy of a Stan file; build the ",
                                "AUTO-CHUNKING library from the cell's own .stan file (auto_chunking = TRUE)."))
              }
              library_is_auto_chunked <-  isTRUE(arguments_of_build_of_each_math_backend[[
                                                         math_backend]]$auto_chunking)
              so_file <-  R_fn_library_of_Stan_model_file(model_file     = stan_file,
                                                          auto_chunking  = library_is_auto_chunked)
              if (!file.exists(stan_file) || !file.exists(so_file)) {
                    stop(paste0(math_backend, ": the library ", so_file, " (or its .stan file) is missing; ",
                                "build it first, on the CPUs not used for timings."))
              }
              if (file.info(so_file)$mtime < file.info(stan_file)$mtime) {
                    stop(paste0(math_backend, ": the library ", so_file, " is older than its .stan file; build ",
                                "it again first."))
              }
              ## (an AUTO-CHUNKING library: its build arguments say whether its reduce_sum is on as well, both
              ##  then taking the layout's num_chunks; AUTO-CHUNKING alone has no WCP, so 1 thread per chain, in
              ##  both phases)
              if (library_is_auto_chunked) {
                    with_reduce_sum <-  arguments_of_build_of_each_math_backend[[
                                              math_backend]]$auto_chunking_with_reduce_sum
                    if (!(isTRUE(with_reduce_sum) || isFALSE(with_reduce_sum))) {
                          stop(paste0(math_backend, ": the AUTO-CHUNKING library ", so_file, " needs ",
                                      "auto_chunking_with_reduce_sum (TRUE or FALSE) in its build arguments."))
                    }
                    if (isFALSE(with_reduce_sum) && any(c(threads_per_chain_of_sampling_grid,
                                                          threads_per_chain_of_burnin_grid) != 1)) {
                          stop(paste0(math_backend, ": AUTO-CHUNKING alone has no WCP: 1 thread per chain only."))
                    }
              }
        }
        for (N_name in names(Stan_data_list_of_each_N)) {
              stopifnot(as.numeric(N_name) == Stan_data_list_of_each_N[[N_name]]$N)
        }
        ##
        ## ---- The N_chunks of a layout's grid ---------------------------------------------------------------------------------------
        ##
        ## (the list of the case's N in num_chunks_of_each_listed_N - the next listed N up, the largest listed N's
        ##  list above it - with at least one chunk per thread, at least fewest_num_chunks_of_grid chunks, and
        ##  never more chunks than N; the doublings past its largest come in the loops below)
        fn_num_chunks_of_grid <-  function(threads_per_chain,
                                           N_of_the_case) {

              listed_N <-  as.numeric(names(num_chunks_of_each_listed_N))
              index_of_the_listed_N <-  if (any(listed_N >= N_of_the_case))
                                              which(listed_N == min(listed_N[listed_N >= N_of_the_case])) else
                                              which.max(listed_N)
              num_chunks <-  num_chunks_of_each_listed_N[[index_of_the_listed_N]]
              num_chunks <-  num_chunks[num_chunks >= threads_per_chain]
              num_chunks <-  pmax(fewest_num_chunks_of_grid, num_chunks)
              num_chunks <-  sort(unique(num_chunks))
              return(num_chunks[num_chunks <= N_of_the_case])

        }
        ##
        ## ---- The layouts of the grids ----------------------------------------------------------------------------------------------
        ##
        stopifnot(n_chains_sampling <= n_CPUs, all(phases_to_time %in% c("sampling", "burnin")))
        threads_per_chain_of_sampling_layouts <-  c()
        if ("sampling" %in% phases_to_time) {
              threads_per_chain_of_sampling_layouts <-  threads_per_chain_of_sampling_grid[
                    threads_per_chain_of_sampling_grid * n_chains_sampling <= n_CPUs]
        }
        threads_per_chain_of_burnin_layouts <-  c()
        if ("burnin" %in% phases_to_time) {
              threads_per_chain_of_burnin_layouts <-  threads_per_chain_of_burnin_grid[
                    threads_per_chain_of_burnin_grid * n_chains_burnin <= n_CPUs]
        }
        ##
        ## ---- Every layout of the grids, before the doublings, and whether a full call of it is saved -------------------------------
        ##
        planned_layouts <-  list()
        for (math_backend in names(arguments_of_build_of_each_math_backend)) {
              for (N_name in names(Stan_data_list_of_each_N)) {
                    for (phase in phases_to_time) {
                          n_chains_of_the_phase <-  if (phase == "sampling") n_chains_sampling else
                                                          n_chains_burnin
                          threads_per_chain_of_the_phase <-  if (phase == "sampling")
                                                                   threads_per_chain_of_sampling_layouts else
                                                                   threads_per_chain_of_burnin_layouts
                          for (threads_per_chain in threads_per_chain_of_the_phase) {
                                num_chunks <-  fn_num_chunks_of_grid(threads_per_chain,
                                                                     Stan_data_list_of_each_N[[N_name]]$N)
                                if (length(num_chunks) == 0) next
                                planned_layouts[[length(planned_layouts) + 1]] <-  data.frame(
                                      math_backend   = math_backend,
                                      N              = Stan_data_list_of_each_N[[N_name]]$N,
                                      phase          = phase,
                                      n_chains       = n_chains_of_the_phase,
                                      n_threads_WCP  = threads_per_chain,
                                      num_chunks     = num_chunks,
                                      stringsAsFactors = FALSE)
                          }
                    }
              }
        }
        planned_layouts <-  if (length(planned_layouts) == 0) data.frame() else do.call(rbind, planned_layouts)
        if (nrow(planned_layouts) > 0) planned_layouts$already_saved <-  FALSE
        columns_of_a_layout <-  c("math_backend", "N", "phase", "n_chains", "n_threads_WCP", "num_chunks")
        if ((nrow(planned_layouts) > 0) && file.exists(output_file)) {
              rows_saved_before <-  utils::read.csv(output_file, stringsAsFactors = FALSE)
              if (!("timed_with_the_short_call_only" %in% names(rows_saved_before))) {
                    rows_saved_before$timed_with_the_short_call_only <-  FALSE
              }
              is_a_full_call <-  !(rows_saved_before$timed_with_the_short_call_only %in% TRUE)
              rows_saved_before <-  rows_saved_before[is_a_full_call, , drop = FALSE]
              if (all(columns_of_a_layout %in% names(rows_saved_before))) {
                    planned_layouts$already_saved <-  do.call(paste, planned_layouts[columns_of_a_layout]) %in%
                                                      do.call(paste, rows_saved_before[columns_of_a_layout])
              }
        }
        n_cases <-  length(arguments_of_build_of_each_math_backend) * length(Stan_data_list_of_each_N)
        n_layouts_to_time <-  if (nrow(planned_layouts) == 0) 0 else sum(!planned_layouts$already_saved)
        ## (each layout: one untimed iteration and the timed call, plus its start-up; each case: initialisation,
        ##  M_bytes/chain and the lp_grad timing, about 5 s; the doublings past each list are not counted)
        predicted_seconds_of_everything <-  5 * n_cases + n_layouts_to_time *
                                                          (max(2 * target_seconds_of_timed_call,
                                                               1.2 * minimum_seconds_of_timed_call) +
                                                           predicted_seconds_of_setup_of_each_layout)
        message(paste0("\033[36mPredicted total time: ", round(predicted_seconds_of_everything / 60, 1),
                       " minutes (", n_cases, " cases of math backend and N; ", nrow(planned_layouts),
                       " layouts before the doublings, ", n_layouts_to_time, " of them not saved yet; phases: ",
                       paste(phases_to_time, collapse = " and "), "; burn-in N_threads/chain ",
                       paste(threads_per_chain_of_burnin_layouts, collapse = ", "), " with ", n_chains_burnin,
                       " chains; sampling N_threads/chain ",
                       paste(threads_per_chain_of_sampling_layouts, collapse = ", "), " with ", n_chains_sampling,
                       " chains).\033[0m"))
        if (predicted_seconds_of_everything > 60 * most_minutes_of_everything) {
              stop(paste0("The predicted total time is more than ", most_minutes_of_everything, " minutes."))
        }
        if (isTRUE(dry_run)) {
              return(list(predicted_minutes_of_everything = predicted_seconds_of_everything / 60,
                          n_cases = n_cases,
                          most_layouts_of_each_case = if (n_cases == 0) 0 else nrow(planned_layouts) / n_cases,
                          threads_per_chain_of_burnin_grid = threads_per_chain_of_burnin_layouts,
                          n_chains_sampling = n_chains_sampling,
                          planned_layouts = planned_layouts))
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
                    for (column in setdiff(names(rows), names(rows_saved))) {
                          rows_saved[[column]] <-  rep(NA, nrow(rows_saved))
                    }
                    for (column in setdiff(names(rows_saved), names(rows))) rows[[column]] <-  NA
                    rows <-  rbind(rows_saved, rows[, names(rows_saved)])
              }
              utils::write.csv(rows, output_file, row.names = FALSE)
        }

        ##  a full call - same N, math backend, phase, N_chains, N_threads/chain and N_chunks - and saved in
        ##  output_file is read back and never timed again, so that a search stopped part way goes on where it
        ##  stopped, along the same path; each search of a phase that runs to its end, for an N and math backend,
        ##  is written to the file of finished searches beside output_file, which the runner reads to skip it)
        # file_of_finished_searches <-  sub("\\.csv$", "_finished_N_and_math_backend.csv", output_file)
        ## (a name longer than the 255 characters a file system allows, e.g. that of the tensor-product HSGP's
        ##  finished searches beside it; the shorter names stay where they were, so that nothing is timed again)
        file_of_finished_searches <-  sub("\\.csv$", "_finished_N_and_math_backend.csv", output_file)
        if (nchar(basename(file_of_finished_searches)) > 255) {
              folder_of_the_finished_searches <-  file.path(dirname(output_file),
                                                            "finished_N_and_math_backend_of_each_rows_file")
              dir.create(folder_of_the_finished_searches, showWarnings = FALSE)
              file_of_finished_searches <-  file.path(folder_of_the_finished_searches, basename(output_file))
        }
        fn_rows_saved_by_full_calls_of_the_case <-  function(N, math_backend) {

              if (!file.exists(output_file)) return(NULL)
              rows_saved <-  utils::read.csv(output_file, stringsAsFactors = FALSE)
              if (!("timed_with_the_short_call_only" %in% names(rows_saved))) {
                    rows_saved$timed_with_the_short_call_only <-  FALSE
              }
              rows_saved <-  rows_saved[(rows_saved$N %in% N) &
                                        (rows_saved$math_backend %in% math_backend) &
                                        !(rows_saved$timed_with_the_short_call_only %in% TRUE), ]
              if (nrow(rows_saved) == 0) return(NULL)
              return(rows_saved)
        }

        fn_saved_rows_of_layouts <-  function(rows_saved_of_the_case,
                                              phase,
                                              n_chains,
                                              n_threads_WCP,
                                              num_chunks) {

              if (is.null(rows_saved_of_the_case)) return(NULL)
              is_layout <-  (rows_saved_of_the_case$phase %in% phase) &
                            (rows_saved_of_the_case$n_chains %in% n_chains) &
                            (rows_saved_of_the_case$n_threads_WCP %in% n_threads_WCP)
              if (!is.null(num_chunks)) {
                    is_layout <-  is_layout & (rows_saved_of_the_case$num_chunks %in% num_chunks)
              }
              rows <-  rows_saved_of_the_case[is_layout, ]
              if (nrow(rows) == 0) return(NULL)
              return(rows)
        }

        fn_search_is_finished <-  function(N, math_backend, phase, n_chains_of_the_search) {

              if (!file.exists(file_of_finished_searches)) return(FALSE)
              finished <-  utils::read.csv(file_of_finished_searches, stringsAsFactors = FALSE)
              ## (n_chains_of_the_search NA: a sampling search inferred from the rows saved before the file of
              ##  finished searches existed, whatever its number of chains)
              is_this_search <-  (finished$N == N) & (finished$math_backend == math_backend) &
                                 (finished$phase == phase) &
                                 (is.na(finished$n_chains_of_the_search) |
                                  (finished$n_chains_of_the_search == n_chains_of_the_search))
              ## (a search marked finished on another grid of N_chunks, e.g. the earlier one down to 50
              ##  observations per chunk, is searched again: its saved rows are used, and only the layouts it lacks
              ##  are timed)
              if (!("grid_of_num_chunks" %in% names(finished))) return(FALSE)
              is_this_search <-  is_this_search &
                                 (finished$grid_of_num_chunks %in% "E1_E2_chunk_counts_rounded_to_powers_of_2")
              return(any(is_this_search, na.rm = TRUE))
        }

        fn_write_finished_search <-  function(N, math_backend, phase, n_chains_of_the_search) {

              finished <-  data.frame(N                               = N,
                                      math_backend                    = math_backend,
                                      phase                           = phase,
                                      n_chains_of_the_search          = n_chains_of_the_search,
                                      how_it_is_known_to_be_finished  = "searched_to_the_end",
                                      grid_of_num_chunks              = "E1_E2_chunk_counts_rounded_to_powers_of_2",
                                      saved_at                        = format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
                                      stringsAsFactors                = FALSE)
              if (file.exists(file_of_finished_searches)) {
                    finished_before <-  utils::read.csv(file_of_finished_searches, stringsAsFactors = FALSE)
                    if (!("grid_of_num_chunks" %in% names(finished_before))) {
                          finished_before$grid_of_num_chunks <-  NA_character_
                    }
                    finished <-  rbind(finished_before,
                                       finished)
              }
              utils::write.csv(finished, file_of_finished_searches, row.names = FALSE)
              message(paste0("\033[32m", math_backend, ", N = ", N, ", ", phase, " with ", n_chains_of_the_search,
                             " chains: search finished, written to ", basename(file_of_finished_searches),
                             "\033[0m"))
        }

        ##
        ## ---- the unbracketed lines:
        ##
        fn_write_unbracketed_line <-  function(N,
                                               math_backend,
                                               phase,
                                               n_chains_of_the_line,
                                               threads_per_chain_of_the_line,
                                               largest_num_chunks) {

              line <-  data.frame(rows_file           = basename(output_file),
                                  N                   = N,
                                  math_backend        = math_backend,
                                  phase               = phase,
                                  n_chains            = n_chains_of_the_line,
                                  n_threads_WCP       = threads_per_chain_of_the_line,
                                  largest_num_chunks  = largest_num_chunks,
                                  saved_at            = format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
                                  stringsAsFactors    = FALSE)
              file_of_unbracketed_lines <-  file.path(dirname(output_file), "unbracketed_lines.csv")
              if (file.exists(file_of_unbracketed_lines)) {
                    line <-  rbind(utils::read.csv(file_of_unbracketed_lines, stringsAsFactors = FALSE),
                                   line)
              }
              utils::write.csv(line, file_of_unbracketed_lines, row.names = FALSE)
              message(paste0("\033[31m", math_backend, ", N = ", N, ", ", phase, " ", n_chains_of_the_line,
                             " x ", threads_per_chain_of_the_line, ": peak not bracketed at ", largest_num_chunks,
                             " chunks; line not complete\033[0m"))

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
        ##  is sized from the measured speed of the layout timed before it, aiming at the middle of the window)
        # if (is_the_ground_truth && !screening_is_on) {
        #       stop(paste0("The ground truth needs seconds_of_the_short_call_that_screens_each_layout > 0: each ",
        #                   "layout's own short call sizes its full timed call."))
        # }
        ## (the seconds aimed at by the timed call after a layout whose saved row is used, as after a timed
        ##  layout)
        seconds_aimed_at_by_the_call_after_a_saved_layout <-  target_seconds_of_timed_call
        if (is_the_ground_truth) {
              seconds_aimed_at_by_the_call_after_a_saved_layout <-
                    seconds_aimed_at_by_each_ground_truth_timed_call
        }
        fn_N_iter_and_L_of_a_ground_truth_timed_call <-  function(seconds_per_iteration_at_L, L, what) {

              ## (a short call too quick for the chains' timer gives 0 s per iteration (the laptop, Weibull at
              ##  N = 337, burn-in 4 x 4 / 8: "1 timed iterations in 0 s", which stopped the burn-in ground truth
              ##  with "missing value where TRUE/FALSE needed"): such a call is sized from 0.001 s per iteration,
              ##  the smallest time the timer tells apart, and a call outside its limits is then sized again from
              ##  its own seconds per iteration, as every other call)
              if (!is.finite(seconds_per_iteration_at_L) || seconds_per_iteration_at_L <= 0) {
                    message(paste0("\033[33m", what, ": the short call gave ", seconds_per_iteration_at_L,
                                   " s per iteration (too quick for the timer): sized from 0.001 s per ",
                                   "iteration instead\033[0m"))
                    seconds_per_iteration_at_L <-  0.001
              }
              seconds_per_leapfrog_step <-  seconds_per_iteration_at_L / L
              # if (!isTRUE(seconds_per_leapfrog_step <= most_seconds_of_each_ground_truth_timed_call)) {
              ## (a call of one leapfrog step cannot be made shorter: it may last up to
              ##  most_seconds_of_a_ground_truth_timed_call_of_one_leapfrog_step, above the upper limit of the
              ##  other calls)
              if (!isTRUE(seconds_per_leapfrog_step <=
                          most_seconds_of_a_ground_truth_timed_call_of_one_leapfrog_step)) {
                    stop(paste0(what, ": one leapfrog step of every chain lasts ",
                                formatC( seconds_per_leapfrog_step,
                                         format = "f",
                                         digits = 2), " s, more than the ",
                                most_seconds_of_a_ground_truth_timed_call_of_one_leapfrog_step,
                                " s that a ground-truth timed call of one leapfrog step may last."))
              }
              ##  lowered only when one iteration at L_preferred lasts longer than the most seconds allowed, so a
              ##  call at a lowered L with more than 1 iteration breaks it: the iterations and L are worked out
              ##  from the seconds of one leapfrog step, never kept at the lowered L of the call before. 18 Oct
              ##  2026: 63 rows at N >= 100,000 had L below 10 with 2 or more iterations, e.g. 2 iterations at
              ##  L = 2 for 180 x 1 / 1024 of JLS on Stan math at N = 200,000)
              seconds_per_iteration_at_L <-  seconds_per_leapfrog_step * L_preferred
              L <-  L_preferred
              if (seconds_per_iteration_at_L <= most_seconds_of_each_ground_truth_timed_call) {
                    N_iter <-  max(1, round(seconds_aimed_at_by_each_ground_truth_timed_call /
                                            seconds_per_iteration_at_L))
                    ## (a whole number of iterations at L can miss the limits, e.g. 0.8 s per iteration with
                    ##  limits of 1 and 1.5 s: 1 iteration is too short and 2 too long; then the fewest
                    ##  iterations that reach the fewest seconds, at the L that brings the call to the aim)
                    if (N_iter * seconds_per_iteration_at_L < fewest_seconds_of_each_ground_truth_timed_call) {
                          N_iter <-  ceiling(fewest_seconds_of_each_ground_truth_timed_call /
                                             seconds_per_iteration_at_L)
                    }
                    ##  would last longer than the most seconds allowed)
                    while (N_iter > 1 &&
                           N_iter * seconds_per_iteration_at_L > most_seconds_of_each_ground_truth_timed_call) {
                          N_iter <-  N_iter - 1
                    }
                    # if (N_iter * seconds_per_iteration_at_L > most_seconds_of_each_ground_truth_timed_call) {
                    #       L_of_the_call <-  max(1, min(L, round(seconds_aimed_at_by_each_ground_truth_timed_call /
                    #                                             (N_iter * seconds_per_leapfrog_step))))
                    #       return(list(N_iter = N_iter, L = L_of_the_call))
                    # }
                    return(list(N_iter = N_iter, L = L))
              }
              L_of_the_call <-  max(1, min(L, round(seconds_aimed_at_by_each_ground_truth_timed_call /
                                                    seconds_per_leapfrog_step)))
              return(list(N_iter = 1, L = L_of_the_call))
        }

        ## (the ground truth: a timed call outside its limits is sized again from its own seconds per iteration -
        ##  N_iter first, then L once N_iter is 1, from the L of the lp_grad rule - and timed again, at most 3
        ##  times; NULL when the call is within its limits or has been sized again 3 times already)
        fn_N_iter_and_L_of_a_ground_truth_call_outside_its_limits <-  function(seconds_of_the_call,
                                                                              seconds_per_iteration_of_the_call,
                                                                              L_of_the_call,
                                                                              L_of_the_lp_grad_rule,
                                                                              n_times_sized_again,
                                                                              what) {

              if (!is_the_ground_truth) return(NULL)
              if (isTRUE(seconds_of_the_call >= fewest_seconds_of_each_ground_truth_timed_call &&
                         seconds_of_the_call <= most_seconds_of_each_ground_truth_timed_call)) return(NULL)
              if (n_times_sized_again >= 3) return(NULL)
              ## (a call of one leapfrog step - 1 iteration at L = 1 - cannot be made shorter: it is kept when it
              ##  lasts at most most_seconds_of_a_ground_truth_timed_call_of_one_leapfrog_step)
              is_a_call_of_one_leapfrog_step <-  isTRUE(L_of_the_call == 1 &&
                                                        abs(seconds_of_the_call -
                                                            seconds_per_iteration_of_the_call) <=
                                                        1e-9 * seconds_of_the_call)
              if (is_a_call_of_one_leapfrog_step &&
                  isTRUE(seconds_of_the_call > most_seconds_of_each_ground_truth_timed_call) &&
                  isTRUE(seconds_of_the_call <= most_seconds_of_a_ground_truth_timed_call_of_one_leapfrog_step)) {
                    return(NULL)
              }
              message(paste0("  ", what, ": ", formatC(seconds_of_the_call, format = "f", digits = 2),
                             " s, outside the limits: sized again from this call and timed again"))
              return(fn_N_iter_and_L_of_a_ground_truth_timed_call(
                           seconds_per_iteration_at_L  = seconds_per_iteration_of_the_call *
                                                         L_of_the_lp_grad_rule / L_of_the_call,
                           L                           = L_of_the_lp_grad_rule,
                           what                        = what))
        }

        # fn_stop_if_a_ground_truth_timed_call_lasted_too_long_or_too_short <-  function(seconds_of_the_call,
        #                                                                               what) {
        fn_stop_if_a_ground_truth_timed_call_lasted_too_long_or_too_short <-  function(seconds_of_the_call,
                                                                                      N_iter_of_the_call,
                                                                                      L_of_the_call,
                                                                                      what) {

              message(paste0("  ", what, ": ", formatC(seconds_of_the_call, format = "f", digits = 2), " s"))
              if (!is_the_ground_truth) return(invisible(NULL))
              ## (a call of one leapfrog step - 1 iteration at L = 1 - cannot be made shorter: it may last up to
              ##  most_seconds_of_a_ground_truth_timed_call_of_one_leapfrog_step)
              most_seconds_of_this_call <-  most_seconds_of_each_ground_truth_timed_call
              if (isTRUE(N_iter_of_the_call == 1 && L_of_the_call == 1)) {
                    most_seconds_of_this_call <-  most_seconds_of_a_ground_truth_timed_call_of_one_leapfrog_step
              }
              if (!isTRUE(seconds_of_the_call >= fewest_seconds_of_each_ground_truth_timed_call) ||
                  # !isTRUE(seconds_of_the_call <= most_seconds_of_each_ground_truth_timed_call)) {
                  !isTRUE(seconds_of_the_call <= most_seconds_of_this_call)) {
                    stop(paste0(what, ": the ground-truth timed call lasted ",
                                formatC(seconds_of_the_call, format = "f", digits = 2), " s, outside ",
                                fewest_seconds_of_each_ground_truth_timed_call, " to ",
                                most_seconds_of_this_call, " s."))
              }
              return(invisible(NULL))
        }

        ##
        ## (the N_chunks of a grid: none, one chunk per thread, and the equations' N_chunks times each multiplier;
        ##  powers of two, at least one chunk per thread, without an upper chunk-count limit)
        ##  chunk or fewer, at every N and in every model; and then on, in the loops below, whilst the largest
        ##  N_chunks timed is within 10% of the fastest of its line, so that the fastest is never at the edge)
        # fn_num_chunks_of_grid <-  function(threads_per_chain, num_chunks_from_equations, most_num_chunks) {
        #       num_chunks <-  c(1, threads_per_chain,
        #                        num_chunks_from_equations * multipliers_of_equations_num_chunks)
        ## (the earlier grid, replaced by fn_num_chunks_of_grid() above: the equations' N_chunks times each
        ##  multiplier and the powers of two down to 50 observations per chunk)
        # fn_num_chunks_of_grid <-  function(threads_per_chain, num_chunks_from_equations, most_num_chunks,
        #                                    N_of_the_case) {
        #
        #       log2_of_one_chunk_per_thread <-  ceiling(log2(max(1, threads_per_chain)))
        #       # log2_of_50_observations_per_chunk <-  max(log2_of_one_chunk_per_thread,
        #       #                                           ceiling(log2(N_of_the_case / 50)))
        #       ##
        #       ## ---- The doublings go on to the first power of two whose chunk_size is 50 or fewer ---------------------------------
        #       ##
        #       log2_of_50_observations_per_chunk <-  log2_of_one_chunk_per_thread
        #       while ((2^log2_of_50_observations_per_chunk < N_of_the_case) &&
        #              (fn_chunk_size_and_number_of_blocks_from_N_and_N_chunks(
        #                     N         = N_of_the_case,
        #                     N_chunks  = 2^log2_of_50_observations_per_chunk)$chunk_size > 50)) {
        #
        #             log2_of_50_observations_per_chunk <-  log2_of_50_observations_per_chunk + 1
        #
        #       }
        #       num_chunks <-  c(1, threads_per_chain,
        #                        num_chunks_from_equations * multipliers_of_equations_num_chunks,
        #                        2^(log2_of_one_chunk_per_thread:log2_of_50_observations_per_chunk))
        #       num_chunks <-  2^round(log2(pmax(1, num_chunks)))
        #       # num_chunks <-  pmax(threads_per_chain, num_chunks)
        #       ## (at least the smallest power of two with one chunk per thread, e.g. 32 at 24 threads: TBB's
        #       ##  halving runs a power of two of chunks, so any other N_chunks runs more chunks than it says)
        #       num_chunks <-  pmax(2^log2_of_one_chunk_per_thread, num_chunks)
        #       if (threads_per_chain == 1) num_chunks <-  c(1, num_chunks)
        #       ## (no layout with fewer than fewest_num_chunks_of_grid chunks, e.g. none unchunked at a large N,
        #       ##  where those layouts are far too slow to time)
        #       num_chunks <-  pmax(fewest_num_chunks_of_grid, num_chunks)
        #       num_chunks <-  sort(unique(num_chunks))
        #       ## (the same chunk_size is the same model data: retain the smallest requested count once.)
        #       # return(num_chunks[!duplicated(ceiling(N_of_the_case / num_chunks))])
        #       ## (the N_chunks of the grid are powers of two, each with its own chunk_size and so its own chunks of
        #       ##  reduce_sum_static; more chunks than N cannot run)
        #       return(num_chunks[num_chunks <= N_of_the_case])
        # }

        ##
        all_rows <-  list()
        for (math_backend in names(arguments_of_build_of_each_math_backend)) {
          for (N_name in names(Stan_data_list_of_each_N)) {

              arguments_of_build <-  arguments_of_build_of_each_math_backend[[math_backend]]
              Stan_data_list <-  Stan_data_list_of_each_N[[N_name]]
              N <-  Stan_data_list$N
              # Stan_data_list$chunk_size <-  as.integer(N)
              ## (the data as one chunk: chunk_size and N_blocks of N_chunks = 1, from the one chunk_size formula)
              Stan_data_list <-  fn_Stan_data_list_with_the_chunks_of_N_chunks(Stan_data_list  = Stan_data_list,
                                                                               N_chunks        = 1)
              ## (an AUTO-CHUNKING library reads N_chunks_of_auto_chunking from its data: 1, no chunking, for its
              ##  set-up, M_bytes/chain and lp_grad timing; the timer sets it for each layout)
              library_is_auto_chunked <-  isTRUE(arguments_of_build$auto_chunking)
              Stan_data_list_of_the_set_up <-  Stan_data_list
              if (library_is_auto_chunked) Stan_data_list_of_the_set_up$N_chunks_of_auto_chunking <-  1L
              most_num_chunks <-  Inf
              so_file <-  R_fn_library_of_Stan_model_file(model_file     = arguments_of_build$Stan_model_file_path,
                                                          auto_chunking  = library_is_auto_chunked)
              so_file_mtime_before <-  file.info(so_file)$mtime
              message(paste0("\033[36m", math_backend, ", N = ", N, " ...\033[0m"))
              ## (the searches of this N and math backend already finished are skipped, the whole case when every
              ##  phase of this call is finished: nothing is loaded or timed again)
              sampling_search_is_finished <-  ("sampling" %in% phases_to_time) &&
                                              fn_search_is_finished(N, math_backend, "sampling",
                                                                    n_chains_sampling)
              burnin_search_is_finished <-  ("burnin" %in% phases_to_time) &&
                                            fn_search_is_finished(N, math_backend, "burnin", n_chains_burnin)
              sampling_line_not_bracketed <-  FALSE
              burnin_line_not_bracketed <-  FALSE
              if (sampling_search_is_finished) {
                    message(paste0("\033[36m", math_backend, ", N = ", N, ": sampling search already finished ",
                                   "(", basename(file_of_finished_searches), "): not timed again\033[0m"))
              }
              if (burnin_search_is_finished) {
                    message(paste0("\033[36m", math_backend, ", N = ", N, ": burn-in search with ",
                                   n_chains_burnin, " chains already finished (",
                                   basename(file_of_finished_searches),
                                   "): not timed again\033[0m"))
              }
              if (!("sampling" %in% phases_to_time) || sampling_search_is_finished) {
                    if (!("burnin" %in% phases_to_time) || burnin_search_is_finished) next
              }
              rows_saved_of_the_case <-  fn_rows_saved_by_full_calls_of_the_case(N, math_backend)
              ##
              ## ---- (1) the library, initialised with the arguments of its build (never rebuilt):
              ##
              ## (a library in NicoStan's nuisance-block form: the size of its nuisance block, which NicoStan
              ##  samples; 0 for every other library)
              n_nuisance_owned_by_NicoStan <-  if (is.null(n_nuisance_owned_by_NicoStan_of_each_N)) 0 else
                                                     n_nuisance_owned_by_NicoStan_of_each_N[[N_name]]
              init_object <-  initialise_model( Model_type            = "Stan",
                                                sample_nuisance       = TRUE,
                                                Stan_data_list        = Stan_data_list_of_the_set_up,
                                                Stan_model_file_path  = arguments_of_build$Stan_model_file_path,
                                                Stan_cpp_user_header  = arguments_of_build$Stan_cpp_user_header,
                                                stanc_args            = arguments_of_build$stanc_args,
                                                make_args             = arguments_of_build$make_args,
                                                auto_chunking         = library_is_auto_chunked,
                                                compile               = TRUE,
                                                n_nuisance_override   = if (n_nuisance_owned_by_NicoStan > 0)
                                                                              n_nuisance_owned_by_NicoStan else
                                                                              NULL,
                                                ## (a regular library: the nuisance blocks named by the user,
                                                names_of_the_nuisance_blocks_of_a_regular_library  =
                                                      if (n_nuisance_owned_by_NicoStan > 0) NULL else
                                                            names_of_the_nuisance_blocks_of_a_regular_library)
              if (!identical(file.info(so_file)$mtime, so_file_mtime_before)) {
                    stop(paste0(math_backend, ": the library ", so_file, " was rebuilt by initialise_model(); ",
                                "build it beforehand with the same arguments."))
              }
              ##
              ## (a regular library with named nuisance blocks: the blocks must be its first unconstrained
              ##  coordinates; their size, from the library's own unconstrained names, is its n_nuisance, never
              ##  the size of whichever block happens to be declared first)
              names_of_nuisance_blocks <-  names_of_the_nuisance_blocks_of_a_regular_library
              if ((n_nuisance_owned_by_NicoStan == 0) && is.null(names_of_nuisance_blocks)) {
                    stop(paste0(math_backend, ", N = ", N, ": the nuisance blocks of the regular library ",
                                so_file, " are not named (names_of_the_nuisance_blocks_of_a_regular_library; ",
                                "character(0) for none): NicoStan never chooses them itself."))
              }
              if ((n_nuisance_owned_by_NicoStan == 0) && !is.null(names_of_nuisance_blocks)) {
                    unconstrained_names <-  init_object$bs_model$param_unc_names()
                    is_nuisance_coordinate <-  rep(FALSE, length(unconstrained_names))
                    ## (the parameters block of the model's Stan file: a named block with no values at this N,
                    ##  e.g. an array of the studies with 4 index tests when there are none, must still be
                    ##  declared there)
                    Stan_model_text <-  paste(readLines(arguments_of_build$Stan_model_file_path), collapse = "\n")
                    parameters_block_text <-  regmatches(Stan_model_text,
                                                         regexpr("(?s)\\nparameters\\s*\\{.*?\\n\\}",
                                                                 Stan_model_text, perl = TRUE))
                    for (name_of_block in names_of_nuisance_blocks) {
                          is_coordinate_of_block <-  (unconstrained_names == name_of_block) |
                                                     startsWith(unconstrained_names, paste0(name_of_block, "."))
                          is_declared <-  grepl(paste0("\\b", name_of_block, "\\s*;"), parameters_block_text,
                                                perl = TRUE)
                          if (!any(is_coordinate_of_block) && !isTRUE(is_declared)) {
                                stop(paste0(math_backend, ", N = ", N, ": the nuisance block ", name_of_block,
                                            " is not a parameter of the library ", so_file, "."))
                          }
                          is_nuisance_coordinate <-  is_nuisance_coordinate | is_coordinate_of_block
                    }
                    n_nuisance_named <-  sum(is_nuisance_coordinate)
                    if (!identical(which(is_nuisance_coordinate), seq_len(n_nuisance_named))) {
                          stop(paste0(math_backend, ", N = ", N, ": the nuisance blocks ",
                                      paste(names_of_nuisance_blocks, collapse = ", "),
                                      " are not the first unconstrained coordinates of the library ", so_file,
                                      "; a regular library cannot reorder them (NicoStan's nuisance-block form ",
                                      "can: nuisance_params of initialise_model())."))
                    }
                    if (n_nuisance_named != init_object$n_nuisance) {
                          init_object <-  initialise_model(
                                Model_type            = "Stan",
                                sample_nuisance       = TRUE,
                                Stan_data_list        = Stan_data_list_of_the_set_up,
                                Stan_model_file_path  = arguments_of_build$Stan_model_file_path,
                                Stan_cpp_user_header  = arguments_of_build$Stan_cpp_user_header,
                                stanc_args            = arguments_of_build$stanc_args,
                                make_args             = arguments_of_build$make_args,
                                auto_chunking         = library_is_auto_chunked,
                                compile               = TRUE,
                                n_nuisance_override   = n_nuisance_named)
                          if (!identical(file.info(so_file)$mtime, so_file_mtime_before)) {
                                stop(paste0(math_backend, ": the library ", so_file, " was rebuilt by ",
                                            "initialise_model(); build it beforehand with the same arguments."))
                          }
                    }
                    message(colourise(paste0(math_backend, ", N = ", N, ": nuisance blocks ",
                                             paste(names_of_nuisance_blocks, collapse = ", "), " = the first ",
                                             init_object$n_nuisance, " unconstrained coordinates; ",
                                             init_object$n_params_main, " main parameters"), "cyan"))
              }
              JSON_text <-  fn_JSON_text_of_Stan_data_list(Stan_data_list = Stan_data_list)
              JSON_text_of_the_set_up <-  if (library_is_auto_chunked) fn_JSON_text_of_Stan_data_list(
                                                Stan_data_list = Stan_data_list_of_the_set_up) else JSON_text
              ##
              ## ---- (2) M_bytes/chain, (3) L and N_iter from the seconds of one lp_grad call:
              ##
              M_bytes <-  fn_estimate_initial_M_bytes_per_chain(
                                model_so_file                 = init_object$model_so_file,
                                Stan_data_list                = Stan_data_list_of_the_set_up,
                                n_repeats                     = 1,
                                n_nuisance_owned_by_NicoStan  = n_nuisance_owned_by_NicoStan)$M_bytes_per_chain
              ##  calls of its Stan file, and the L2 cache per core, threads per core and cores of this process's
              ##  CPUs)
              kind_of_tape_of_this_library <-  fn_kind_of_tape_of_Stan_model_from_the_nico_calls_in_its_Stan_file(
                                                     Stan_model_file_path                =
                                                           init_object$Stan_model_file_path,
                                                     BayesMVP_AVX_version_of_Stan_model  =
                                                           init_object$BayesMVP_AVX_version_of_Stan_model)
              topology_of_the_equations <-  fn_read_L3_cache_topology_of_available_CPUs()
              L2_cache_and_threads_per_core <-
                    fn_read_threads_per_core_and_L2_cache_size_of_first_CPU_of_this_process()
              N_cores_of_the_equations <-
                    max(1, floor(topology_of_the_equations$N_hardware_threads_available /
                                 L2_cache_and_threads_per_core$N_threads_per_core))
              lp_grad <-  fn_seconds_per_lp_grad_call_of_compiled_Stan_model(
                                model_so_file                   = init_object$model_so_file,
                                JSON_text_of_Stan_data          = JSON_text_of_the_set_up,
                                stop_if_library_already_loaded  = FALSE,
                                n_nuisance_owned_by_NicoStan    = n_nuisance_owned_by_NicoStan,
                                BridgeStan_model_already_built_from_the_same_JSON_text  = NULL,
                                fn_whether_the_repeated_calls_are_needed                = NULL)
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
              # for (threads_per_chain in threads_per_chain_of_sampling_layouts) {
              threads_per_chain_of_sampling_layouts_of_this_case <-  threads_per_chain_of_sampling_layouts
              if (sampling_search_is_finished) threads_per_chain_of_sampling_layouts_of_this_case <-  c()
              for (threads_per_chain in threads_per_chain_of_sampling_layouts_of_this_case) {

                    # n_chains <-  n_chains_sampling / threads_per_chain
                    ## (the same chains at every N_threads/chain: the one grid of chains x threads per chain of
                    ##  both phases)
                    n_chains <-  n_chains_sampling
                    equations <-  fn_compute_sampling_num_chunks_for_CPUs_of_this_process(
                                        M_bytes_per_chain       = M_bytes,
                                        n_chains_sampling       = n_chains,
                                        n_threads_WCP_sampling  = threads_per_chain,
                                        gradient_type           = "autodiff")
                    num_chunks_from_equations <-  2^ceiling(log2(max(1, equations$N_chunks)))
                    ##  thread per chain only; more threads per chain keep the rule above)
                    if (threads_per_chain == 1) {
                          num_chunks_from_equations <-  fn_sampling_num_chunks_of_Paper_1_equation(
                                M_bytes_per_chain             = M_bytes,
                                N_units                       = N,
                                kind_of_tape                  = kind_of_tape_of_this_library,
                                L2_cache_size_bytes_per_core  = L2_cache_and_threads_per_core$L2_cache_size_bytes,
                                N_threads_per_core            = L2_cache_and_threads_per_core$N_threads_per_core
                          )$num_chunks_at_one_thread_per_chain
                    }
                    ## (a search already started for these chains and threads: the equations' N_chunks saved with
                    ##  its rows, so that the grid, and hence the search's path, is the same as before it stopped)
                    rows_saved_of_these_threads <-  fn_saved_rows_of_layouts(rows_saved_of_the_case,
                                                                             "sampling",
                                                                             n_chains,
                                                                             threads_per_chain,
                                                                             NULL)
                    if (!is.null(rows_saved_of_these_threads) &&
                        is.finite(rows_saved_of_these_threads$num_chunks_from_equations[1])) {
                          num_chunks_from_equations <-  rows_saved_of_these_threads$num_chunks_from_equations[1]
                    }
                    # num_chunks_of_grid <-  fn_num_chunks_of_grid(threads_per_chain, num_chunks_from_equations,
                    #                                              most_num_chunks, N)
                    num_chunks_of_grid <-  fn_num_chunks_of_grid(threads_per_chain, N)
                    TP_iter_of_grid <-  c()
                    index_of_num_chunks <-  1
                    while (index_of_num_chunks <= length(num_chunks_of_grid)) {
                          layout <-  data.frame( n_chains       = n_chains,
                                                 n_threads_WCP  = threads_per_chain,
                                                 num_chunks     = num_chunks_of_grid[index_of_num_chunks])
                          ## (an AUTO-CHUNKING library with its reduce_sum on: both take the layout's num_chunks)
                          if (library_is_auto_chunked &&
                              isTRUE(arguments_of_build$auto_chunking_with_reduce_sum)) {
                                layout$num_chunks_of_auto_chunking <-  layout$num_chunks
                          }
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
                          ## (a layout already timed by a full call and saved: its row is used, never timed again)
                          saved_rows_of_this_layout <-  fn_saved_rows_of_layouts(rows_saved_of_the_case,
                                                                                 "sampling",
                                                                                 n_chains,
                                                                                 threads_per_chain,
                                                                                 layout$num_chunks)
                          if (!is.null(saved_rows_of_this_layout)) {
                                saved_row <-  saved_rows_of_this_layout[nrow(saved_rows_of_this_layout), ]
                                message(paste0("  sampling ", n_chains, " x ", threads_per_chain, " / ",
                                               layout$num_chunks, ": already timed (saved at ",
                                               saved_row$saved_at, ", TP_iter ",
                                               formatC(saved_row$TP_iter, format = "f", digits = 2),
                                               "): not timed again"))
                                fastest_seconds_per_draw_of_sampling_so_far <-
                                      min(fastest_seconds_per_draw_of_sampling_so_far,
                                          saved_row$seconds_per_draw_over_all_chains, na.rm = TRUE)
                                if (isTRUE(saved_row$seconds_per_iteration > 0)) {
                                      N_iter_of_sampling_calls <-  max(1, round(
                                            seconds_aimed_at_by_the_call_after_a_saved_layout /
                                            saved_row$seconds_per_iteration))
                                }
                                all_rows[[length(all_rows) + 1]] <-  saved_row
                                TP_iter_of_grid <-  c(TP_iter_of_grid, saved_row$TP_iter)
                                ## (the largest N_chunks is the fastest: twice it is timed too, as below)
                                if ((index_of_num_chunks == length(num_chunks_of_grid)) &&
                                    # (which.max(TP_iter_of_grid) == length(TP_iter_of_grid)) &&
                                    ##  of the fastest)
                                    (TP_iter_of_grid[length(TP_iter_of_grid)] > max(TP_iter_of_grid) / 1.1)) {
                                      # if (ceiling(N / (2 * num_chunks_of_grid[index_of_num_chunks])) ==
                                      #     ceiling(N / num_chunks_of_grid[index_of_num_chunks])) {
                                      ## (twice the chunks would be more chunks than N: the doublings end here)
                                      if ((!missing(most_num_chunks_of_grid)) &&
                                          (2 * num_chunks_of_grid[index_of_num_chunks] >
                                             most_num_chunks_of_grid)) {
                                            ## (the most chunks of the grid, e.g. 1 with no chunking at all:
                                            ##  the doublings end here, and the search is finished)
                                      } else if (2 * num_chunks_of_grid[index_of_num_chunks] > N) {
                                            sampling_line_not_bracketed <-  TRUE
                                            fn_write_unbracketed_line(N,
                                                                      math_backend,
                                                                      "sampling",
                                                                      n_chains,
                                                                      threads_per_chain,
                                                                      num_chunks_of_grid[index_of_num_chunks])
                                      } else {
                                            num_chunks_of_grid <-  c(num_chunks_of_grid,
                                                                     2 * num_chunks_of_grid[index_of_num_chunks])
                                      }
                                }
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
                                      n_iterations_of_untimed_warm_up_call  = 0,
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
                                      ## (aimed at 0.5 s, repeated until it lasts at least 0.4 s: the first calls of a fresh
                                      ##  library are slower per iteration than its later calls, e.g. Weibull v2 AVX-512,
                                      ##  N = 50,000, 180 x 1 / 1: short call 0.02 s, a second sizing call of 15 iterations in
                                      ##  0.08 s, then the timed call of 144 iterations in 0.46 s)
                                      for (sizing_call in 1:4) {
                                        N_iter_of_the_second_sizing_call <-  max(1, round(0.5 / short_timing$seconds_per_iteration))
                                        short_timing <-  fn_time_sampling_of_Stan_model_for_each_layout(
                                              init_object                           = init_object,
                                              Stan_data_list                        = Stan_data_list,
                                              layouts                               = layout,
                                              time_layout_in_one_call               = TRUE,
                                              n_iterations_of_long_call             = N_iter_of_the_second_sizing_call,
                                              n_iterations_of_untimed_warm_up_call  = 0,
                                              L_main                                = L_of_the_timed_call,
                                              JSON_text_of_Stan_data_list           = JSON_text,
                                              fn_snapshot_hardware_counters         = fn_snapshot_hardware_counters,
                                              initial_values_of_the_chains          =
                                                    initial_values_of_the_chains_of_each_N[[as.character(N)]],
                                              nuisance_metric_of_the_timings        = nuisance_metric_of_the_sampling_timings,
                                              L_of_the_untimed_iterations           = L_of_the_untimed_warm_up_iteration)
                                        seconds_of_the_second_sizing_call <-  short_timing$seconds_per_iteration *
                                                                              N_iter_of_the_second_sizing_call
                                        message(paste0("  sampling ", n_chains, " x ", threads_per_chain, " / ", layout$num_chunks,
                                                       ", second untimed sizing call: ", N_iter_of_the_second_sizing_call,
                                                       " iterations, ", formatC(seconds_of_the_second_sizing_call, format = "f",
                                                                                digits = 2), " s"))
                                        if (isTRUE(seconds_of_the_second_sizing_call >= 0.4)) break
                                      }
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
                          n_timed_calls_sized_again <-  0
                          if (!timed_with_the_short_call_only) repeat {
                                timing <-  fn_time_sampling_of_Stan_model_for_each_layout(
                                                 init_object                           = init_object,
                                                 Stan_data_list                        = Stan_data_list,
                                                 layouts                               = layout,
                                                 time_layout_in_one_call               = TRUE,
                                                 n_iterations_of_long_call             = N_iter_of_sampling_calls,
                                                 n_iterations_of_untimed_warm_up_call  = 0,
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
                                      what_of_the_call <-  paste0(math_backend, ", N = ", N, ", sampling ",
                                                                  n_chains, " x ", threads_per_chain, " / ",
                                                                  layout$num_chunks, ", ",
                                                                  N_iter_of_sampling_calls, " iterations at L = ",
                                                                  L_of_the_timed_call)
                                      N_iter_and_L <-  fn_N_iter_and_L_of_a_ground_truth_call_outside_its_limits(
                                            seconds_of_the_call                = seconds_of_timed_call,
                                            seconds_per_iteration_of_the_call  = timing$seconds_per_iteration,
                                            L_of_the_call                      = L_of_the_timed_call,
                                            L_of_the_lp_grad_rule              = L_and_N_iter$L,
                                            n_times_sized_again                = n_timed_calls_sized_again,
                                            what                               = what_of_the_call)
                                      if (!is.null(N_iter_and_L)) {
                                            n_timed_calls_sized_again <-  n_timed_calls_sized_again + 1
                                            N_iter_of_sampling_calls <-  N_iter_and_L$N_iter
                                            L_of_the_timed_call <-  N_iter_and_L$L
                                            next
                                      }
                                      fn_stop_if_a_ground_truth_timed_call_lasted_too_long_or_too_short(
                                            seconds_of_the_call  = seconds_of_timed_call,
                                            N_iter_of_the_call   = N_iter_of_sampling_calls,
                                            L_of_the_call        = L_of_the_timed_call,
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
                          ## (the N_chunks asked for and, beside it, the chunk_size and the N_blocks that ran)
                          blocks_of_layout <-  fn_chunk_size_and_number_of_blocks_from_N_and_N_chunks(
                                                     N         = N,
                                                     N_chunks  = layout$num_chunks)
                          timing$chunk_size <-  blocks_of_layout$chunk_size
                          timing$N_blocks <-  blocks_of_layout$N_blocks
                          ## (an AUTO-CHUNKING library: its N_chunks_of_auto_chunking, the layout's num_chunks)
                          if (library_is_auto_chunked) timing$num_chunks_of_auto_chunking <-  layout$num_chunks
                          # timing$n_chunks_of_size_chunk_size <-  blocks_of_layout$n_chunks_of_size_chunk_size
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
                              # (which.max(TP_iter_of_grid) == length(TP_iter_of_grid)) &&
                              (TP_iter_of_grid[length(TP_iter_of_grid)] > max(TP_iter_of_grid) / 1.1)) {
                                # if (ceiling(N / (2 * num_chunks_of_grid[index_of_num_chunks])) ==
                                #     ceiling(N / num_chunks_of_grid[index_of_num_chunks])) {
                                ## (twice the chunks would be more chunks than N: the doublings end here)
                                if ((!missing(most_num_chunks_of_grid)) &&
                                    (2 * num_chunks_of_grid[index_of_num_chunks] > most_num_chunks_of_grid)) {
                                      ## (the most chunks of the grid, e.g. 1 with no chunking at all:
                                      ##  the doublings end here, and the search is finished)
                                } else if (2 * num_chunks_of_grid[index_of_num_chunks] > N) {
                                      sampling_line_not_bracketed <-  TRUE
                                      fn_write_unbracketed_line(N,
                                                                math_backend,
                                                                "sampling",
                                                                n_chains,
                                                                threads_per_chain,
                                                                num_chunks_of_grid[index_of_num_chunks])
                                } else {
                                      num_chunks_of_grid <-  c(num_chunks_of_grid,
                                                               2 * num_chunks_of_grid[index_of_num_chunks])
                                }
                          }
                          index_of_num_chunks <-  index_of_num_chunks + 1
                    }

              }
              ## (the sampling search of this N and math backend ran to its end: written to the file of finished
              ##  searches)
              if (("sampling" %in% phases_to_time) && !sampling_search_is_finished &&
                  !sampling_line_not_bracketed) {
                    fn_write_finished_search(N, math_backend, "sampling", n_chains_sampling)
              }
              ##
              ## ---- (5) the burn-in layouts (the persistent burn-in worker, one untimed iteration first):
              ##
              # for (threads_per_chain in threads_per_chain_of_burnin_layouts) {
              threads_per_chain_of_burnin_layouts_of_this_case <-  threads_per_chain_of_burnin_layouts
              if (burnin_search_is_finished) threads_per_chain_of_burnin_layouts_of_this_case <-  c()
              for (threads_per_chain in threads_per_chain_of_burnin_layouts_of_this_case) {

                    # equations <-  fn_compute_sampling_num_chunks_for_CPUs_of_this_process(
                    #                     M_bytes_per_chain       = M_bytes,
                    #                     n_chains_sampling       = n_chains_burnin,
                    #                     n_threads_WCP_sampling  = threads_per_chain,
                    #                     gradient_type           = "autodiff")
                    # num_chunks_from_equations <-  2^ceiling(log2(max(1, equations$N_chunks)))
                    num_chunks_from_equations <-  fn_burnin_num_chunks_of_Paper_1_equation_at_threads_per_chain(
                          M_bytes_per_chain             = M_bytes,
                          N_units                       = N,
                          n_chains_burnin               = n_chains_burnin,
                          n_threads_WCP_burnin          = threads_per_chain,
                          kind_of_tape                  = kind_of_tape_of_this_library,
                          N_cores_available             = N_cores_of_the_equations,
                          L2_cache_size_bytes_per_core  = L2_cache_and_threads_per_core$L2_cache_size_bytes
                    )$num_chunks_burnin
                    ## (a search already started for these chains and threads: the equations' N_chunks saved with
                    ##  its rows, so that the grid, and hence the search's path, is the same as before it stopped)
                    rows_saved_of_these_threads <-  fn_saved_rows_of_layouts(rows_saved_of_the_case,
                                                                             "burnin",
                                                                             n_chains_burnin,
                                                                             threads_per_chain,
                                                                             NULL)
                    if (!is.null(rows_saved_of_these_threads) &&
                        is.finite(rows_saved_of_these_threads$num_chunks_from_equations[1])) {
                          num_chunks_from_equations <-  rows_saved_of_these_threads$num_chunks_from_equations[1]
                    }
                    # num_chunks_of_grid <-  fn_num_chunks_of_grid(threads_per_chain, num_chunks_from_equations,
                    #                                              most_num_chunks, N)
                    num_chunks_of_grid <-  fn_num_chunks_of_grid(threads_per_chain, N)
                    TP_iter_of_grid <-  c()
                    index_of_num_chunks <-  1
                    while (index_of_num_chunks <= length(num_chunks_of_grid)) {
                          num_chunks <-  num_chunks_of_grid[index_of_num_chunks]
                          ## (a layout already timed by a full call and saved: its row is used, never timed again)
                          saved_rows_of_this_layout <-  fn_saved_rows_of_layouts(rows_saved_of_the_case,
                                                                                 "burnin",
                                                                                 n_chains_burnin,
                                                                                 threads_per_chain,
                                                                                 num_chunks)
                          if (!is.null(saved_rows_of_this_layout)) {
                                saved_row <-  saved_rows_of_this_layout[nrow(saved_rows_of_this_layout), ]
                                message(paste0("  burn-in ", n_chains_burnin, " x ", threads_per_chain, " / ",
                                               num_chunks, ": already timed (saved at ", saved_row$saved_at,
                                               ", ", formatC(saved_row$seconds_per_iteration, format = "f",
                                                             digits = 4),
                                               " s per iteration): not timed again"))
                                fastest_seconds_per_iteration_of_burnin_so_far <-
                                      min(fastest_seconds_per_iteration_of_burnin_so_far,
                                          saved_row$seconds_per_iteration, na.rm = TRUE)
                                if (isTRUE(saved_row$seconds_per_iteration > 0)) {
                                      N_iter_of_burnin_calls <-  max(1, round(
                                            seconds_aimed_at_by_the_call_after_a_saved_layout /
                                            saved_row$seconds_per_iteration))
                                }
                                all_rows[[length(all_rows) + 1]] <-  saved_row
                                TP_iter_of_grid <-  c(TP_iter_of_grid, saved_row$TP_iter)
                                ## (the largest N_chunks is the fastest: twice it is timed too, as below)
                                if ((index_of_num_chunks == length(num_chunks_of_grid)) &&
                                    # (which.max(TP_iter_of_grid) == length(TP_iter_of_grid)) &&
                                    ##  of the fastest)
                                    (TP_iter_of_grid[length(TP_iter_of_grid)] > max(TP_iter_of_grid) / 1.1)) {
                                      # if (ceiling(N / (2 * num_chunks)) == ceiling(N / num_chunks)) {
                                      ## (twice the chunks would be more chunks than N: the doublings end here)
                                      if ((!missing(most_num_chunks_of_grid)) &&
                                          (2 * num_chunks > most_num_chunks_of_grid)) {
                                            ## (the most chunks of the grid, e.g. 1 with no chunking at all:
                                            ##  the doublings end here, and the search is finished)
                                      } else if (2 * num_chunks > N) {
                                            burnin_line_not_bracketed <-  TRUE
                                            fn_write_unbracketed_line(N,
                                                                      math_backend,
                                                                      "burnin",
                                                                      n_chains_burnin,
                                                                      threads_per_chain,
                                                                      num_chunks)
                                      } else {
                                            num_chunks_of_grid <-  c(num_chunks_of_grid, 2 * num_chunks)
                                      }
                                }
                                index_of_num_chunks <-  index_of_num_chunks + 1
                                next
                          }
                          ##
                          ## ---- An AUTO-CHUNKING library: its chunks in the data given to the worker --------------------------------
                          ##
                          ## (the layout's num_chunks is N_chunks_of_auto_chunking, put first in the data's JSON
                          ##  text, and the model's reduce_sum takes the same num_chunks (both on) or 1 chunk
                          ##  (AUTO-CHUNKING alone))
                          num_chunks_of_the_models_reduce_sum <-  num_chunks
                          JSON_text_of_the_layout <-  JSON_text
                          if (library_is_auto_chunked) {

                                if (!isTRUE(arguments_of_build$auto_chunking_with_reduce_sum)) {
                                      num_chunks_of_the_models_reduce_sum <-  1
                                }
                                position_of_the_first_brace <-  regexpr(pattern  = "{",
                                                                        text     = substr(JSON_text, 1, 64),
                                                                        fixed    = TRUE)
                                stopifnot(position_of_the_first_brace > 0)
                                JSON_text_of_the_layout <-  paste0("{\"N_chunks_of_auto_chunking\": ",
                                                                   as.integer(num_chunks), ", ",
                                                                   ## (to the last character: substring() stops at
                                                                   ##  1,000,000 characters by default)
                                                                   substring(JSON_text,
                                                                             position_of_the_first_brace + 1,
                                                                             nchar(JSON_text)))

                          }
                          ## (the layout's burn-in worker kept from its short call for its timed call(s): its data
                          ##  file, worker and untimed iteration once per layout, not once per call, about 0.075 s
                          ##  data file removed, without gc(), which costs about 0.1 s)
                          if (exists("workers_of_the_layout", inherits = FALSE)) {
                                for (name_of_kept_layout in ls(envir = workers_of_the_layout)) {
                                      kept_worker <-  workers_of_the_layout[[name_of_kept_layout]]
                                      path_of_kept_data <-  kept_worker$json_file_path
                                      if (is.character(path_of_kept_data) &&
                                          grepl("\\.json$", path_of_kept_data)) {
                                            unlink(path_of_kept_data)
                                      }
                                }
                                rm(list = ls(envir = workers_of_the_layout), envir = workers_of_the_layout)
                          }
                          workers_of_the_layout <-  new.env()
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
                                      num_chunks_candidates               = num_chunks_of_the_models_reduce_sum,
                                      n_rounds                            = 1,
                                      environment_of_workers_kept_for_reuse  = workers_of_the_layout,
                                      n_timed_iterations                  = N_iter_of_the_short_call,
                                      minimum_seconds_per_timing          = 0,
                                      L_main                              = L_and_N_iter$L,
                                      JSON_text_of_Stan_data_list         = JSON_text_of_the_layout,
                                      ## (the data given as its JSON text, with the layout's chunk_size: no data
                                      ##  file per call, and the worker takes the Stan model kept for that text,
                                      pass_the_data_to_the_worker_as_JSON_text_instead_of_a_file  = TRUE,
##                                      n_untimed_iterations_before_timing  = 1,
                                      n_untimed_iterations_before_timing  = 0,
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
                          n_timed_calls_sized_again <-  0
                          if (!timed_with_the_short_call_only) repeat {
                          snapshot_before <-  fn_snapshot_or_NA()
                          wall_seconds <-  system.time(
                                burnin_timing <-  fn_time_burnin_iteration_for_each_num_chunks(
                                                        init_object                         = init_object,
                                                        Stan_data_list                      = Stan_data_list,
                                                        n_chains_burnin                     = n_chains_burnin,
                                                        n_threads_WCP_burnin                = threads_per_chain,
                                                        num_chunks_candidates               =
                                                              num_chunks_of_the_models_reduce_sum,
                                                        n_rounds                            = 1,
                                                        environment_of_workers_kept_for_reuse  =
                                                              workers_of_the_layout,
                                                        n_timed_iterations                  =
                                                              N_iter_of_burnin_calls,
                                                        minimum_seconds_per_timing          = 0,
                                                        # L_main                              = L_and_N_iter$L,
                                                        L_main                              = L_of_the_burnin_call,
                                                        JSON_text_of_Stan_data_list         = JSON_text_of_the_layout,
                                                  pass_the_data_to_the_worker_as_JSON_text_instead_of_a_file =
                                                              TRUE,
##                                                        n_untimed_iterations_before_timing  = 1,
                                                        n_untimed_iterations_before_timing  = 0,
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
                                what_of_the_call <-  paste0(math_backend, ", N = ", N, ", burn-in ",
                                                            n_chains_burnin, " x ", threads_per_chain, " / ",
                                                            num_chunks, ", ",
                                                            N_iter_of_burnin_calls, " iterations at L = ",
                                                            L_of_the_burnin_call)
                                N_iter_and_L <-  fn_N_iter_and_L_of_a_ground_truth_call_outside_its_limits(
                                      seconds_of_the_call                = seconds_of_timed_call,
                                      seconds_per_iteration_of_the_call  = seconds_per_iteration,
                                      L_of_the_call                      = L_of_the_burnin_call,
                                      L_of_the_lp_grad_rule              = L_and_N_iter$L,
                                      n_times_sized_again                = n_timed_calls_sized_again,
                                      what                               = what_of_the_call)
                                if (!is.null(N_iter_and_L)) {
                                      n_timed_calls_sized_again <-  n_timed_calls_sized_again + 1
                                      N_iter_of_burnin_calls <-  N_iter_and_L$N_iter
                                      L_of_the_burnin_call <-  N_iter_and_L$L
                                      next
                                }
                                fn_stop_if_a_ground_truth_timed_call_lasted_too_long_or_too_short(
                                      seconds_of_the_call  = seconds_of_timed_call,
                                      N_iter_of_the_call   = N_iter_of_burnin_calls,
                                      L_of_the_call        = L_of_the_burnin_call,
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
                          ## (the N_chunks asked for and, beside it, the chunk_size and the N_blocks that ran)
                          blocks_of_layout <-  fn_chunk_size_and_number_of_blocks_from_N_and_N_chunks(
                                                     N         = N,
                                                     N_chunks  = num_chunks)
                          timing$chunk_size <-  blocks_of_layout$chunk_size
                          timing$N_blocks <-  blocks_of_layout$N_blocks
                          ## (an AUTO-CHUNKING library: its N_chunks_of_auto_chunking, the layout's num_chunks)
                          if (library_is_auto_chunked) timing$num_chunks_of_auto_chunking <-  num_chunks
                          # timing$n_chunks_of_size_chunk_size <-  blocks_of_layout$n_chunks_of_size_chunk_size
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
                              # (which.max(TP_iter_of_grid) == length(TP_iter_of_grid)) &&
                              (TP_iter_of_grid[length(TP_iter_of_grid)] > max(TP_iter_of_grid) / 1.1)) {
                                # if (ceiling(N / (2 * num_chunks)) == ceiling(N / num_chunks)) {
                                ## (twice the chunks would be more chunks than N: the doublings end here)
                                if ((!missing(most_num_chunks_of_grid)) &&
                                    (2 * num_chunks > most_num_chunks_of_grid)) {
                                      ## (the most chunks of the grid, e.g. 1 with no chunking at all:
                                      ##  the doublings end here, and the search is finished)
                                } else if (2 * num_chunks > N) {
                                      burnin_line_not_bracketed <-  TRUE
                                      fn_write_unbracketed_line(N,
                                                                math_backend,
                                                                "burnin",
                                                                n_chains_burnin,
                                                                threads_per_chain,
                                                                num_chunks)
                                } else {
                                      num_chunks_of_grid <-  c(num_chunks_of_grid, 2 * num_chunks)
                                }
                          }
                          index_of_num_chunks <-  index_of_num_chunks + 1
                    }

              }
              ## (the burn-in search of this N, math backend and number of chains ran to its end: written to the
              ##  file of finished searches)
              if (("burnin" %in% phases_to_time) && !burnin_search_is_finished &&
                  !burnin_line_not_bracketed) {
                    fn_write_finished_search(N, math_backend, "burnin", n_chains_burnin)
              }

          }
        }
        ##
        return(invisible(all_rows))

}





















