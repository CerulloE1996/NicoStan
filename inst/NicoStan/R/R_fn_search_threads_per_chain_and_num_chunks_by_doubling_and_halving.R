




#### ---- Paper 1's check of the layout: the threads per chain and N_chunks doubled and halved together ----------

#' Paper 1's check of the layout: the threads per chain and N_chunks doubled and halved together
#'
#' The check of Paper 1's algorithm for the layout of any model, for the burn-in and for the sampling. From the
#' equations' layout (threads per chain and N_chunks), its four neighbours are timed: the threads per chain
#' doubled and halved, and N_chunks doubled and halved (powers of two, with at least one chunk per thread). While
#' the fastest neighbour is faster than the current layout, it becomes the current layout and its neighbours not
#' yet timed are timed. When no neighbour is faster, the current layout and its neighbours are timed once more in
#' one round, and the means of their timings decide: the search goes on from a neighbour that is now faster, or
#' else it ends, with the fastest of the current layout and its neighbours (by the means of their timings; a
#' neighbour faster by less than fraction_faster_needed_to_move is not searched from, but is still chosen when it
#' is the fastest). The threads per chain and N_chunks are searched together, as they interact (e.g. more
#' threads per chain may need more chunks); the search never times a full grid. Two looks past the neighbours,
#' at most 3 more timings in a search: when the doubling of the threads per chain is not faster, the threads per
#' chain times 4 are timed (at the same N_chunks) and moved to when faster (on a CPU of 2 threads per core, 2
#' threads per chain can be the slowest and 4 the fastest); and the round that confirms the current layout also
#' times its two diagonal neighbours (the threads per chain doubled and N_chunks halved, and the reverse).
#'
#' @param threads_per_chain_start,num_chunks_start The equations' layout. When the equations give N_chunks = 1
#'   (no chunking), only N_chunks = 1 and N_chunks = 2 are timed, at the equations' threads per chain (and, when
#'   also_time_2_threads_per_chain_at_2_chunks_when_the_equations_say_no_chunking is TRUE, 2 threads per chain
#' @param fn_seconds_of_each_layout A function of a data frame of layouts (columns threads_per_chain and
#'   num_chunks) which times each of them once and returns their seconds, in the same order (the seconds
#'   compared: per burn-in iteration for the burn-in, per draw over all chains for the sampling).
#' @param most_threads_per_chain The most threads per chain (e.g. the CPUs divided by the burn-in chains).
#' @param most_moves The most moves of the search (default 30).
#' @param fraction_faster_needed_to_move A neighbour must be faster than the current layout by more than this
#'   fraction for the search to move to it and go on from it (default 0.03, i.e. 3%: about the noise of one
#'   timing), so that the search does not wander over near-ties.
#' @param timings_already_made Layouts timed before the search (a data frame: threads_per_chain, num_chunks and
#'   seconds), e.g. the candidates its start was chosen from (default NULL).
#' @param fn_another_timing_fits_in_the_budget A function without arguments that returns TRUE when one more
#'   timing of a layout fits in the seconds left for choosing the layout (default NULL: no budget, every timing
#'   the search needs is made). With it, the layouts are timed one at a time, and when it returns FALSE the
#'   search ends at once, with the fastest of the last layout searched from and its neighbours timed so far;
#'   the round that times the current layout and its neighbours once more is then not made.
#' @param fraction_slower_within_which_a_neighbour_is_timed_again_in_the_last_round Only the neighbours at most
#'   this fraction slower than the current layout (by the means of their timings) are timed once more in the
#'   round that confirms the current layout, with the current layout (default NULL: every neighbour, as Paper 1's
#'   check does).
#' @param time_the_doubling_after_a_doubling_of_num_chunks_within_the_move_threshold TRUE (default): when no
#'   neighbour is faster by more than fraction_faster_needed_to_move, but the doubling of N_chunks is within it of
#'   the current layout (a tie, i.e. a flat step, faster or slower), the doubling after it (4 times N_chunks) is
#'   timed too, and the search moves there only if it is faster than the current layout by more than
#'   fraction_faster_needed_to_move (e.g. a latent diffusion model whose burn-in was 2% faster at 16 chunks than
#'   at 8, then 7% at 32 and 8% at 64); FALSE: the search stops at a flat step of N_chunks, as before.
#' @param time_the_doubling_of_num_chunks_first_and_the_others_only_if_it_is_not_faster TRUE: from
#'   each layout, its N_chunks doubled is timed first, and the search moves to it when it is faster by more than
#'   fraction_faster_needed_to_move; the other neighbours (N_chunks halved, the threads per chain doubled and
#'   halved) are timed only when it is not (the sampling: Paper 1's equations' N_chunks was at or below the
#'   fastest in 211 of 213 cases of Paper 3's ground truth); FALSE (default): every neighbour at once, as before.
#' @return A list: threads_per_chain and num_chunks (the layout found: the fastest of the last layout searched
#'   from and its neighbours), threads_per_chain_last_searched_from and num_chunks_last_searched_from,
#'   threads_per_chain_start and
#'   num_chunks_start (the equations' layout, as allowed), timings (a data frame: threads_per_chain, num_chunks,
#'   mean_seconds (the mean of its timings) and n_timings, one row per layout timed) and n_timings (every
#'   timing, counting a layout each time it was timed), every_timing_made (a data frame, one row per timing in
#'   the order made: threads_per_chain, num_chunks, seconds, why_it_was_timed, and the columns of the attribute
#'   details_of_each_timing of the seconds given by fn_seconds_of_each_layout, when it gives one, e.g. the
#'   seconds of the call and the chains' own seconds) and n_timings_past_the_neighbours (at most 3).
#' @export
## fn_search_threads_per_chain_and_num_chunks_by_doubling_and_halving <-  function(
##                                                                         threads_per_chain_start,
##                                                                         num_chunks_start,
##                                                                         fn_seconds_of_each_layout,
##                                                                         most_threads_per_chain,
##                                                                         most_num_chunks,
##                                                                         most_moves  = 30,
##                                                                         fraction_faster_needed_to_move =
##                                                                               0.03,
##                                                                         timings_already_made  = NULL
## ) {
## (the arguments one per line, as the names of the budget's arguments are long)
fn_search_threads_per_chain_and_num_chunks_by_doubling_and_halving <-  function(
        threads_per_chain_start,
        num_chunks_start,
        fn_seconds_of_each_layout,
        most_threads_per_chain,
        most_num_chunks,
        most_moves                                                                 = 30,
        fraction_faster_needed_to_move                                             = 0.03,
        timings_already_made                                                       = NULL,
        fn_another_timing_fits_in_the_budget                                       = NULL,
        fraction_slower_within_which_a_neighbour_is_timed_again_in_the_last_round  = NULL,
        time_the_doubling_after_a_doubling_of_num_chunks_within_the_move_threshold = TRUE,
        time_the_doubling_of_num_chunks_first_and_the_others_only_if_it_is_not_faster = FALSE,
        also_time_2_threads_per_chain_at_2_chunks_when_the_equations_say_no_chunking
) {

        ##
        ## ---- powers of two; the CPU limit applies to threads, with no upper limit on N_chunks:
        ##
        largest_threads_per_chain <-  2^floor(log2(max(1, most_threads_per_chain)))
        fn_allowed_layout <-  function(threads_per_chain, num_chunks) {

              threads_per_chain <-  min(largest_threads_per_chain, max(1, 2^round(log2(threads_per_chain))))
              num_chunks <-  max(threads_per_chain, 2^ceiling(log2(num_chunks)))
              return(c(threads_per_chain, num_chunks))
        }

        fn_name_of_layout <-  function(layout) paste0(layout[1], "_threads_x_", layout[2], "_chunks")
        ##
        ## ---- the neighbours of a layout: the threads per chain doubled and halved, N_chunks doubled and halved:
        ##
        fn_neighbours_of_layout <-  function(layout) {

              neighbours <-  list( fn_allowed_layout(2 * layout[1], layout[2]),
                                   fn_allowed_layout(layout[1] / 2, layout[2]),
                                   fn_allowed_layout(layout[1], 2 * layout[2]),
                                   fn_allowed_layout(layout[1], layout[2] / 2))
              names_of_neighbours <-  vapply(X = neighbours, FUN = fn_name_of_layout, FUN.VALUE = "")
              # keep <-  !duplicated(names_of_neighbours) & (names_of_neighbours != fn_name_of_layout(layout))
              ## (most_num_chunks is the N of the data: exactly N_chunks chunks run, so that more chunks than units
              ##  would leave chunks empty)
              num_chunks_of_neighbours <-  vapply(X = neighbours, FUN = function(neighbour) neighbour[2],
                                                  FUN.VALUE = 0)
              keep <-  !duplicated(names_of_neighbours) & (names_of_neighbours != fn_name_of_layout(layout)) &
                       (num_chunks_of_neighbours <= most_num_chunks)
              return(neighbours[keep])
        }

        ##
        ## ---- every timing of every layout timed, by the name of the layout:
        ##
        seconds_of_each_layout_timed <-  list()
        n_timings <-  0
        if (!is.null(timings_already_made)) {
              for (index in seq_len(nrow(timings_already_made))) {
                    name <-  fn_name_of_layout(c(timings_already_made$threads_per_chain[index],
                                                 timings_already_made$num_chunks[index]))
                    seconds_of_each_layout_timed[[name]] <-  c(seconds_of_each_layout_timed[[name]],
                                                               timings_already_made$seconds[index])
              }
              n_timings <-  nrow(timings_already_made)
        }
        ## fn_time_layouts <-  function(layouts) {
        ##       layouts_to_time <-  data.frame(
        ##             threads_per_chain  = vapply(X = layouts, FUN = function(layout) layout[1], FUN.VALUE = 0),
        ##             num_chunks         = vapply(X = layouts, FUN = function(layout) layout[2], FUN.VALUE = 0))
        ##       seconds <-  fn_seconds_of_each_layout(layouts_to_time)
        ##       for (index in seq_along(layouts)) {
        ##             name <-  fn_name_of_layout(layouts[[index]])
        ##             seconds_of_each_layout_timed[[name]] <<-  c(seconds_of_each_layout_timed[[name]],
        ##                                                         seconds[index])
        ##       }
        ##       n_timings <<-  n_timings + length(layouts)
        ## }
        ## (without a budget, the layouts are timed together, as before; with a budget, one at a time, each only
        ##  when one more timing fits in the seconds left, and the search ends at the first that does not)
        budget_spent <-  FALSE
        ## (every timing made by the search, in order, with why the layout was timed and, when
        ##  fn_seconds_of_each_layout gives them as the attribute details_of_each_timing (a data frame, one row
        ##  per layout timed), the details of each timing, e.g. the seconds of the call and the chains' own
        ##  seconds, so that a pick can be judged from its rows, never by timing it again)
        every_timing_made <-  list()
        why_the_layouts_are_timed <-  "start and neighbours"
        fn_record_the_timings <-  function(layouts, seconds) {

              details <-  attr(seconds, "details_of_each_timing")
              for (index in seq_along(layouts)) {
                    row <-  data.frame( order_of_the_timing  = length(every_timing_made) + 1,
                                        threads_per_chain    = layouts[[index]][1],
                                        num_chunks           = layouts[[index]][2],
                                        seconds              = seconds[index],
                                        why_it_was_timed     = why_the_layouts_are_timed)
                    if (is.data.frame(details) && (nrow(details) >= index)) {
                          row <-  cbind(row, details[index, , drop = FALSE])
                    }
                    every_timing_made[[length(every_timing_made) + 1]] <<-  row
              }
              return(invisible(NULL))
        }

        fn_time_layouts <-  function(layouts) {

              if (is.null(fn_another_timing_fits_in_the_budget)) {
                    layouts_to_time <-  data.frame(
                          threads_per_chain  = vapply(X = layouts, FUN = function(layout) layout[1],
                                                      FUN.VALUE = 0),
                          num_chunks         = vapply(X = layouts, FUN = function(layout) layout[2],
                                                      FUN.VALUE = 0))
                    seconds <-  fn_seconds_of_each_layout(layouts_to_time)
                    for (index in seq_along(layouts)) {

                          name <-  fn_name_of_layout(layouts[[index]])
                          seconds_of_each_layout_timed[[name]] <<-  c(seconds_of_each_layout_timed[[name]],
                                                                      seconds[index])
                    }

                    n_timings <<-  n_timings + length(layouts)
                    fn_record_the_timings(layouts = layouts, seconds = seconds)
                    return(invisible(NULL))
              }
              for (index in seq_along(layouts)) {
                    if (!isTRUE(fn_another_timing_fits_in_the_budget())) {
                          budget_spent <<-  TRUE
                          return(invisible(NULL))
                    }
                    layout_to_time <-  data.frame( threads_per_chain  = layouts[[index]][1],
                                                   num_chunks         = layouts[[index]][2])
                    name <-  fn_name_of_layout(layouts[[index]])
                    ## seconds_of_each_layout_timed[[name]] <<-  c(seconds_of_each_layout_timed[[name]],
                    ##                                             fn_seconds_of_each_layout(layout_to_time)[1])
                    seconds <-  fn_seconds_of_each_layout(layout_to_time)
                    seconds_of_each_layout_timed[[name]] <<-  c(seconds_of_each_layout_timed[[name]], seconds[1])
                    n_timings <<-  n_timings + 1
                    fn_record_the_timings(layouts = layouts[index], seconds = seconds)
              }
              return(invisible(NULL))
        }

        fn_mean_seconds <-  function(layout) mean(seconds_of_each_layout_timed[[fn_name_of_layout(layout)]])
        fn_is_timed <-  function(layout) !is.null(seconds_of_each_layout_timed[[fn_name_of_layout(layout)]])
        ##
        ## ---- a tie at the doubling of N_chunks (within fraction_faster_needed_to_move of the current layout,
        ##      faster or slower): the doubling after it, timed once (if not yet timed), and returned when it is
        ##      faster than the current layout by more than fraction_faster_needed_to_move (otherwise NULL):
        ##
        names_of_layouts_looked_past_a_tie <-  character(0)
        fn_layout_past_a_tie_at_the_doubling_of_num_chunks <-  function(layout) {

              if (!isTRUE(time_the_doubling_after_a_doubling_of_num_chunks_within_the_move_threshold)) {
                    return(NULL)
              }
              layout_doubled <-  fn_allowed_layout(layout[1], 2 * layout[2])
              layout_doubled_twice <-  fn_allowed_layout(layout[1], 4 * layout[2])
              if (identical(fn_name_of_layout(layout_doubled), fn_name_of_layout(layout)) ||
                  identical(fn_name_of_layout(layout_doubled_twice), fn_name_of_layout(layout_doubled)) ||
                  !fn_is_timed(layout_doubled)) {
                    return(NULL)
              }
              seconds_of_current <-  fn_mean_seconds(layout)
              doubling_is_a_tie <-  abs(fn_mean_seconds(layout_doubled) - seconds_of_current) <=
                                    fraction_faster_needed_to_move * seconds_of_current
              if (!doubling_is_a_tie) return(NULL)
              if (!fn_is_timed(layout_doubled_twice)) {
                    ## (once per layout, so that a layout confirmed again does not time it twice)
                    if (fn_name_of_layout(layout) %in% names_of_layouts_looked_past_a_tie) return(NULL)
                    names_of_layouts_looked_past_a_tie <<-  c(names_of_layouts_looked_past_a_tie,
                                                              fn_name_of_layout(layout))
                    why_the_layouts_are_timed <<-  "N_chunks x 4, past a tie at the doubling of N_chunks"
                    fn_time_layouts(list(layout_doubled_twice))
                    why_the_layouts_are_timed <<-  "start and neighbours"
                    if (!fn_is_timed(layout_doubled_twice)) return(NULL)
              }
              if (fn_mean_seconds(layout_doubled_twice) < (1 - fraction_faster_needed_to_move) *
                                                          seconds_of_current) {
                    return(layout_doubled_twice)
              }
              return(NULL)
        }

        ##
        ## ---- at most this many more timings in a search for the two looks below, past a doubling of the threads
        ##      per chain that is not faster and at the diagonal neighbours (the threshold, the timing and the
        ##      budget are unchanged; on the laptop, 8 cores of 2 threads, 2 threads per chain is often the
        ##      slowest and 4 the fastest, so a search that only doubles and halves stops before 4):
        ##
        most_timings_past_the_neighbours <-  3
        n_timings_past_the_neighbours <-  0
        ##
        ## ---- the doubling of the threads per chain not faster than the current layout by more than
        ##      fraction_faster_needed_to_move: the threads per chain times 4, at the same N_chunks (as allowed),
        ##      timed once (if not yet timed), and returned when it is faster than the current layout by more than
        ##      fraction_faster_needed_to_move (otherwise NULL):
        ##
        names_of_layouts_looked_past_the_doubling_of_the_threads <-  character(0)
        fn_layout_past_a_doubling_of_the_threads_per_chain_that_is_not_faster <-  function(layout) {

              layout_threads_doubled <-  fn_allowed_layout(2 * layout[1], layout[2])
              layout_threads_times_4 <-  fn_allowed_layout(4 * layout[1], layout[2])
              if (identical(fn_name_of_layout(layout_threads_doubled), fn_name_of_layout(layout)) ||
                  identical(fn_name_of_layout(layout_threads_times_4),
                            fn_name_of_layout(layout_threads_doubled)) ||
                  !fn_is_timed(layout_threads_doubled)) {
                    return(NULL)
              }
              seconds_of_current <-  fn_mean_seconds(layout)
              if (fn_mean_seconds(layout_threads_doubled) < (1 - fraction_faster_needed_to_move) *
                                                            seconds_of_current) {
                    return(NULL)
              }
              if (!fn_is_timed(layout_threads_times_4)) {
                    ## (once per layout, and only within the timings allowed past the neighbours)
                    if (fn_name_of_layout(layout) %in% names_of_layouts_looked_past_the_doubling_of_the_threads) {
                          return(NULL)
                    }
                    if (n_timings_past_the_neighbours >= most_timings_past_the_neighbours) return(NULL)
                    names_of_layouts_looked_past_the_doubling_of_the_threads <<-
                          c(names_of_layouts_looked_past_the_doubling_of_the_threads, fn_name_of_layout(layout))
                    why_the_layouts_are_timed <<-  "threads per chain x 4, past a doubling of them not faster"
                    fn_time_layouts(list(layout_threads_times_4))
                    why_the_layouts_are_timed <<-  "start and neighbours"
                    if (!fn_is_timed(layout_threads_times_4)) return(NULL)
                    n_timings_past_the_neighbours <<-  n_timings_past_the_neighbours + 1
              }
              if (fn_mean_seconds(layout_threads_times_4) < (1 - fraction_faster_needed_to_move) *
                                                            seconds_of_current) {
                    return(layout_threads_times_4)
              }
              return(NULL)
        }

        ##
        ## ---- the diagonal neighbours of a layout (the threads per chain doubled and N_chunks halved, the
        ##      threads per chain halved and N_chunks doubled, as allowed), other than the layout and its
        ##      neighbours:
        ##
        fn_diagonal_neighbours_of_layout <-  function(layout) {

              diagonals <-  list( fn_allowed_layout(2 * layout[1], layout[2] / 2),
                                  fn_allowed_layout(layout[1] / 2, 2 * layout[2]))
              names_taken <-  c(fn_name_of_layout(layout),
                                vapply(X = fn_neighbours_of_layout(layout), FUN = fn_name_of_layout,
                                       FUN.VALUE = ""))
              names_of_diagonals <-  vapply(X = diagonals, FUN = fn_name_of_layout, FUN.VALUE = "")
              keep <-  !duplicated(names_of_diagonals) & !(names_of_diagonals %in% names_taken)
              return(diagonals[keep])
        }

        ##
        ## ---- the search, from the equations' layout:
        ##
        layout_current <-  fn_allowed_layout(threads_per_chain_start, num_chunks_start)
        layout_start <-  layout_current
        name_of_layout_last_confirmed <-  ""
        n_moves <-  0
        ##
        ## ---- the equations say no chunking (N_chunks = 1): only N_chunks = 1 and N_chunks = 2 are timed, at the
        ##      equations' threads per chain, and the search ends there: no further doubling, no look past a tie
        ##      IT"); the layout found is the faster of the two, below:
        ##
        equations_say_no_chunking <-  (num_chunks_start == 1)
        if (equations_say_no_chunking) {
              why_the_layouts_are_timed <-  "the equations say no chunking: N_chunks = 1 and N_chunks = 2 only"
              layouts_of_no_chunking_and_2_chunks <-  list( fn_allowed_layout(threads_per_chain_start, 1),
                                                            fn_allowed_layout(threads_per_chain_start, 2))
              ## (as his PS1 burn-in runner, where a thread needs at least one chunk: at 2 chunks, 2 threads per
              ##  OBVIOUSLY ... COPY THEM")
              if (isTRUE(also_time_2_threads_per_chain_at_2_chunks_when_the_equations_say_no_chunking)) {
                    layouts_of_no_chunking_and_2_chunks <-  c(layouts_of_no_chunking_and_2_chunks,
                                                              list(fn_allowed_layout(2, 2)))
              }
              names_of_the_two <-  vapply(X = layouts_of_no_chunking_and_2_chunks, FUN = fn_name_of_layout,
                                          FUN.VALUE = "")
              layouts_of_no_chunking_and_2_chunks <-
                    layouts_of_no_chunking_and_2_chunks[!duplicated(names_of_the_two)]
              layouts_not_timed <-  Filter(f = function(layout) !fn_is_timed(layout),
                                           x = layouts_of_no_chunking_and_2_chunks)
              if (length(layouts_not_timed) > 0) fn_time_layouts(layouts_not_timed)
        }
        # while (n_moves < most_moves) {
        while (!equations_say_no_chunking && (n_moves < most_moves)) {

              why_the_layouts_are_timed <-  "start and neighbours"
              neighbours <-  fn_neighbours_of_layout(layout_current)
              ## (upwards first: the doubling of N_chunks timed, with the current layout when not yet timed, and
              ##  moved to when it is faster; the other neighbours only when it is not)
              doubling_of_num_chunks_first <-
                    time_the_doubling_of_num_chunks_first_and_the_others_only_if_it_is_not_faster
              if (isTRUE(doubling_of_num_chunks_first)) {
                    layout_doubled <-  fn_allowed_layout(layout_current[1], 2 * layout_current[2])
                    if (!identical(fn_name_of_layout(layout_doubled), fn_name_of_layout(layout_current))) {
                          layouts_first <-  list()
                          if (!fn_is_timed(layout_current)) {
                                layouts_first <-  c(layouts_first, list(layout_current))
                          }
                          if (!fn_is_timed(layout_doubled)) {
                                layouts_first <-  c(layouts_first, list(layout_doubled))
                          }
                          if (length(layouts_first) > 0) fn_time_layouts(layouts_first)
                          if (fn_is_timed(layout_current) && fn_is_timed(layout_doubled) &&
                              (fn_mean_seconds(layout_doubled) < (1 - fraction_faster_needed_to_move) *
                                                                 fn_mean_seconds(layout_current))) {
                                layout_current <-  layout_doubled
                                n_moves <-  n_moves + 1
                                if (budget_spent) break
                                next
                          }
                          if (budget_spent) break
                    }
              }
              layouts_not_timed <-  Filter(f = function(layout) !fn_is_timed(layout), x = neighbours)
              if (!fn_is_timed(layout_current)) layouts_not_timed <-  c(list(layout_current), layouts_not_timed)
              if (length(layouts_not_timed) > 0) fn_time_layouts(layouts_not_timed)
              ## (the budget spent before the current layout was timed: the search ends with the layouts timed)
              if (!fn_is_timed(layout_current)) break
              ## (only the neighbours timed are compared: all of them, unless the budget was spent)
              neighbours <-  Filter(f = fn_is_timed, x = neighbours)
              if (length(neighbours) == 0) break
              ##
              ## (a neighbour faster than the current layout: the search moves to the fastest neighbour)
              seconds_of_neighbours <-  vapply(X = neighbours, FUN = fn_mean_seconds, FUN.VALUE = 0)
              if (min(seconds_of_neighbours) < (1 - fraction_faster_needed_to_move) *
                                                fn_mean_seconds(layout_current)) {

                    layout_current <-  neighbours[[which.min(seconds_of_neighbours)]]
                    n_moves <-  n_moves + 1
                    if (budget_spent) break
                    next
              }

              if (budget_spent) break
              ##
              ## (no neighbour faster, but a tie at the doubling of N_chunks: the doubling after it, moved to when
              ##  it is faster)
              layout_past_a_tie <-  fn_layout_past_a_tie_at_the_doubling_of_num_chunks(layout_current)
              if (!is.null(layout_past_a_tie)) {
                    layout_current <-  layout_past_a_tie
                    n_moves <-  n_moves + 1
                    if (budget_spent) break
                    next
              }
              if (budget_spent) break
              ##
              ## (no neighbour faster, and the doubling of the threads per chain not faster: the threads per chain
              ##  times 4, moved to when it is faster)
              layout_past_the_threads <-  fn_layout_past_a_doubling_of_the_threads_per_chain_that_is_not_faster(
                                                layout_current)
              if (!is.null(layout_past_the_threads)) {
                    layout_current <-  layout_past_the_threads
                    n_moves <-  n_moves + 1
                    if (budget_spent) break
                    next
              }
              if (budget_spent) break
              ##
              ## (no neighbour faster: the current layout and its neighbours timed once more, in one round, and
              ##  the means of their timings decide; a layout is confirmed only once)
              if (identical(name_of_layout_last_confirmed, fn_name_of_layout(layout_current))) break
              ## fn_time_layouts(c(list(layout_current), neighbours))
              ## (every neighbour, or only those at most the given fraction slower than the current layout)
              neighbours_timed_again <-  neighbours
              if (!is.null(fraction_slower_within_which_a_neighbour_is_timed_again_in_the_last_round)) {
                    seconds_of_current <-  fn_mean_seconds(layout_current)
                    neighbours_timed_again <-  Filter(f = function(layout) {

                          fn_mean_seconds(layout) <= (1 +
                                fraction_slower_within_which_a_neighbour_is_timed_again_in_the_last_round) *
                                seconds_of_current
                    }, x = neighbours)

              }
              ## (the diagonal neighbours not yet timed, in the same round, within the timings allowed past the
              ##  neighbours)
              diagonals_not_timed <-  Filter(f = function(layout) !fn_is_timed(layout),
                                             x = fn_diagonal_neighbours_of_layout(layout_current))
              n_diagonals_allowed <-  max(0, most_timings_past_the_neighbours - n_timings_past_the_neighbours)
              diagonals_not_timed <-  utils::head(diagonals_not_timed, n_diagonals_allowed)
              why_the_layouts_are_timed <-  "the round that confirms the current layout"
              fn_time_layouts(c(list(layout_current), neighbours_timed_again))
              if (!budget_spent && (length(diagonals_not_timed) > 0)) {

                    why_the_layouts_are_timed <-  paste0("diagonal neighbour, in the round that confirms the ",
                                                         "current layout")
                    fn_time_layouts(diagonals_not_timed)
                    n_timings_past_the_neighbours <-  n_timings_past_the_neighbours +
                                                      sum(vapply(X = diagonals_not_timed, FUN = fn_is_timed,
                                                                 FUN.VALUE = TRUE))
              }

              why_the_layouts_are_timed <-  "start and neighbours"
              name_of_layout_last_confirmed <-  fn_name_of_layout(layout_current)
              ## seconds_of_neighbours <-  vapply(X = neighbours, FUN = fn_mean_seconds, FUN.VALUE = 0)
              ## if (min(seconds_of_neighbours) < (1 - fraction_faster_needed_to_move) *
              ##                                   fn_mean_seconds(layout_current)) {
              ##       layout_current <-  neighbours[[which.min(seconds_of_neighbours)]]
              ## (the neighbours and the diagonal neighbours timed: the fastest is moved to when it is faster)
              neighbours_and_diagonals <-  c(neighbours,
                                             Filter(f = fn_is_timed,
                                                    x = fn_diagonal_neighbours_of_layout(layout_current)))
              seconds_of_neighbours <-  vapply(X = neighbours_and_diagonals, FUN = fn_mean_seconds, FUN.VALUE = 0)
              if (min(seconds_of_neighbours) < (1 - fraction_faster_needed_to_move) *
                                                fn_mean_seconds(layout_current)) {
                    layout_current <-  neighbours_and_diagonals[[which.min(seconds_of_neighbours)]]
                    n_moves <-  n_moves + 1
                    if (budget_spent) break
                    next
              }
              ## (a tie at the doubling of N_chunks after the confirming round: the doubling after it, as above)
              if (!budget_spent) {
                    layout_past_a_tie <-  fn_layout_past_a_tie_at_the_doubling_of_num_chunks(layout_current)
                    if (!is.null(layout_past_a_tie)) {
                          layout_current <-  layout_past_a_tie
                          n_moves <-  n_moves + 1
                          if (budget_spent) break
                          next
                    }
              }
              ## (the threads per chain times 4 after the confirming round, as above)
              if (!budget_spent) {
                    layout_past_the_threads <-
                          fn_layout_past_a_doubling_of_the_threads_per_chain_that_is_not_faster(layout_current)
                    if (!is.null(layout_past_the_threads)) {
                          layout_current <-  layout_past_the_threads
                          n_moves <-  n_moves + 1
                          if (budget_spent) break
                          next
                    }
              }
              break

        }
        ##
        ## ---- the layout found: the fastest of the last layout searched from and its neighbours, by the means of
        ##      their timings (a neighbour faster by less than fraction_faster_needed_to_move, i.e. a near-tie, is
        ##      not searched from, but is still chosen when it is the fastest):
        ##
        layout_last_searched_from <-  layout_current
        ## layouts_of_last_neighbourhood <-  c(list(layout_current),
        ##                                     Filter(f = fn_is_timed,
        ##                                            x = fn_neighbours_of_layout(layout_current)))
        ## (the diagonal neighbours timed are in the last neighbourhood too)
        layouts_of_last_neighbourhood <-  c(list(layout_current),
                                            Filter(f = fn_is_timed, x = fn_neighbours_of_layout(layout_current)),
                                            Filter(f = fn_is_timed,
                                                   x = fn_diagonal_neighbours_of_layout(layout_current)))
        seconds_of_last_neighbourhood <-  vapply(X = layouts_of_last_neighbourhood, FUN = fn_mean_seconds,
                                                 FUN.VALUE = 0)
        layout_found <-  layouts_of_last_neighbourhood[[which.min(seconds_of_last_neighbourhood)]]
        ##
        ## ---- every layout timed, with the mean of its timings:
        ##
        names_timed <-  names(seconds_of_each_layout_timed)
        timings <-  data.frame(
              threads_per_chain  = as.numeric(sub("_threads_x_.*$", "", names_timed)),
              num_chunks         = as.numeric(sub("^.*_threads_x_([0-9.e+]+)_chunks$", "\\1", names_timed)),
              mean_seconds       = vapply(X = seconds_of_each_layout_timed, FUN = mean, FUN.VALUE = 0),
              n_timings          = vapply(X = seconds_of_each_layout_timed, FUN = length, FUN.VALUE = 0))
        timings <-  timings[order(timings$threads_per_chain, timings$num_chunks), , drop = FALSE]
        rownames(timings) <-  NULL
        every_timing_made_table <-  NULL
        if (length(every_timing_made) > 0) {
              every_timing_made_table <-  do.call(rbind, every_timing_made)
              rownames(every_timing_made_table) <-  NULL
        }
        return(list( threads_per_chain                       = layout_found[1],
                     num_chunks                              = layout_found[2],
                     threads_per_chain_last_searched_from    = layout_last_searched_from[1],
                     num_chunks_last_searched_from           = layout_last_searched_from[2],
                     threads_per_chain_start                 = layout_start[1],
                     num_chunks_start                        = layout_start[2],
                     timings                                 = timings,
                     n_timings                               = n_timings,
                     every_timing_made                       = every_timing_made_table,
                     n_timings_past_the_neighbours           = n_timings_past_the_neighbours,
                     search_ended_by_the_budget              = budget_spent))

}




















