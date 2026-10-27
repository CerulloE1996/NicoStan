##
##
#### ---- Paper 1's check of N_chunks: doubling (or halving) while it gets faster --------------------------------
##
#' Paper 1's check of the number of chunks: doubling (or halving) while it gets faster
#'
#' The check of Paper 1's algorithm for choosing N_chunks for any model (algorithm "Choosing the number of chunks
#' (N_chunks) for any model", after its line for equation 7h): the gradient is timed at N_chunks, 2 N_chunks and
#' N_chunks / 2 (with WCP, rounded up to multiples of N_threads/chain); while 2 N_chunks or N_chunks / 2 is faster
#' than N_chunks, N_chunks becomes the faster of these, and the next doubling (or halving) is timed; the fastest
#' value timed is returned. Each next doubling (or halving) is timed in the same rounds as the current N_chunks,
#' so that the two are compared under the same conditions.
#'
#' @param num_chunks_start N_chunks from Paper 1's equations (equation 7h; for reduce_sum_static, the power of two
#'   of partial sums which it runs).
#' @param fn_median_seconds_per_iteration_of_each_num_chunks A function of a vector of chunk counts which times
#'   them, in the same rounds, and returns the median seconds per iteration of each, in the same order.
#' @param fewest_num_chunks The fewest chunks allowed (with WCP, N_threads/chain: at least one chunk per thread;
#'   default 1).
#' @param multiple_of_num_chunks Every chunk count is rounded up to a multiple of this (with WCP,
#'   N_threads/chain; default 1).
#' @param powers_of_two TRUE: every chunk count is a power of two (the partial sums of reduce_sum_static, see
#'   Paper 1's online appendix); FALSE (default): a multiple of multiple_of_num_chunks.
#' @param fraction_within_which_a_doubling_is_a_tie_and_the_doubling_after_it_is_timed When the check stops
#'   because the doubling of N_chunks is not faster, but its seconds are within this fraction of the current
#'   N_chunks' (a tie, i.e. a flat step), the doubling after it (4 times N_chunks) is timed too, with the current
#'   N_chunks, and the check goes on from it only if it is faster than the current N_chunks by more than this
#'   fraction (default 0.03, the move threshold of Paper 1's layout search; NULL: the check stops at a flat step,
#'   as before).
#' @return A list: num_chunks (the fastest value timed), num_chunks_start, timings (a data frame: num_chunks and
#'   median_seconds_per_iteration of every chunk count timed, its latest timing) and n_timings (the number of
#'   chunk-count timings, counting a chunk count each time it was timed).
#' @export
fn_check_num_chunks_by_doubling_and_halving <-  function( num_chunks_start,
                                                          fn_median_seconds_per_iteration_of_each_num_chunks,
                                                          fewest_num_chunks       = 1,
                                                          most_num_chunks,
                                                          multiple_of_num_chunks  = 1,
                                                          powers_of_two           = FALSE,
                                fraction_within_which_a_doubling_is_a_tie_and_the_doubling_after_it_is_timed =
                                                                0.03
) {

        ##
        ##
        ## ---- a chunk count as allowed: a power of two, or rounded up to a multiple of multiple_of_num_chunks,
        ##      at least fewest_num_chunks, with no upper limit:
        ##
        fn_allowed_num_chunks <-  function(num_chunks) {

              if (isTRUE(powers_of_two)) {
                    num_chunks <-  2^ceiling(log2(num_chunks))
              } else {
                    num_chunks <-  multiple_of_num_chunks * ceiling(num_chunks / multiple_of_num_chunks)
              }
              return(max(fewest_num_chunks, num_chunks))
        }

        ##
        ## ---- N_chunks, 2 N_chunks and N_chunks / 2, timed in the same rounds:
        ##
        num_chunks_current <-  fn_allowed_num_chunks(num_chunks_start)
        num_chunks_doubled <-  fn_allowed_num_chunks(2 * num_chunks_current)
        num_chunks_halved <-  fn_allowed_num_chunks(num_chunks_current / 2)
        num_chunks_timed <-  unique(c(num_chunks_current, num_chunks_doubled, num_chunks_halved))
        ## (most_num_chunks is the N of the data: exactly N_chunks chunks run, so that more chunks than units
        ##  would leave chunks empty)
        num_chunks_timed <-  num_chunks_timed[num_chunks_timed <= most_num_chunks]
        if (num_chunks_current > most_num_chunks) {
              stop(paste0("fn_check_num_chunks_by_doubling_and_halving: the starting N_chunks = ", num_chunks_current,
                          " is larger than most_num_chunks = ", most_num_chunks, " (the N of the data)."))
        }
        seconds_timed <-  fn_median_seconds_per_iteration_of_each_num_chunks(num_chunks_timed)
        timings <-  data.frame( num_chunks                    = num_chunks_timed,
                                median_seconds_per_iteration  = seconds_timed)
        n_timings <-  length(num_chunks_timed)
        ##
        ## ---- the direction: towards the faster of 2 N_chunks and N_chunks / 2, when it is faster than N_chunks:
        ##
        seconds_current <-  seconds_timed[num_chunks_timed == num_chunks_current]
        multiplier <-  0
        if (min(seconds_timed) < seconds_current) {
              num_chunks_fastest <-  num_chunks_timed[which.min(seconds_timed)]
              multiplier <-  if (num_chunks_fastest > num_chunks_current) 2 else 1 / 2
              num_chunks_current <-  num_chunks_fastest
        }
        ##
        ## ---- a tie at the doubling of N_chunks (within the fraction of the current N_chunks' seconds, faster
        ##      or slower): the doubling after it, timed with the current N_chunks, and N_chunks moved to it only
        ##      when it is faster by more than the fraction (otherwise NULL):
        ##
        fn_num_chunks_past_a_tie_at_the_doubling <-  function( num_chunks,
                                                               seconds_of_num_chunks,
                                                               seconds_of_doubling) {

              fraction <-  fraction_within_which_a_doubling_is_a_tie_and_the_doubling_after_it_is_timed
              if (is.null(fraction)) return(NULL)
              num_chunks_doubled <-  fn_allowed_num_chunks(2 * num_chunks)
              num_chunks_doubled_twice <-  fn_allowed_num_chunks(4 * num_chunks)
              # if ((num_chunks_doubled == num_chunks) || (num_chunks_doubled_twice == num_chunks_doubled)) {
              if ((num_chunks_doubled == num_chunks) || (num_chunks_doubled_twice == num_chunks_doubled) ||
                  (num_chunks_doubled_twice > most_num_chunks)) {
                    return(NULL)
              }
              if (abs(seconds_of_doubling - seconds_of_num_chunks) > fraction * seconds_of_num_chunks) {
                    return(NULL)
              }
              num_chunks_pair <-  c(num_chunks, num_chunks_doubled_twice)
              seconds_pair <-  fn_median_seconds_per_iteration_of_each_num_chunks(num_chunks_pair)
              n_timings <<-  n_timings + 2
              timings <<-  rbind(timings[!(timings$num_chunks %in% num_chunks_pair), , drop = FALSE],
                                 data.frame( num_chunks                    = num_chunks_pair,
                                             median_seconds_per_iteration  = seconds_pair))
              if (seconds_pair[2] < (1 - fraction) * seconds_pair[1]) return(num_chunks_doubled_twice)
              return(NULL)
        }

        if (multiplier == 0) {
              num_chunks_doubled <-  fn_allowed_num_chunks(2 * num_chunks_current)
              num_chunks_past_a_tie <-  fn_num_chunks_past_a_tie_at_the_doubling(
                                              num_chunks             = num_chunks_current,
                                              seconds_of_num_chunks  = seconds_current,
                                              seconds_of_doubling    = seconds_timed[num_chunks_timed ==
                                                                                     num_chunks_doubled][1])
              if (!is.null(num_chunks_past_a_tie)) {
                    num_chunks_current <-  num_chunks_past_a_tie
                    multiplier <-  2
              }
        }
        ##
        ## ---- the next doubling (or halving), timed with the current N_chunks, for as long as it is faster:
        ##
        while (multiplier != 0) {
              ##
              num_chunks_next <-  fn_allowed_num_chunks(multiplier * num_chunks_current)
              # if (num_chunks_next == num_chunks_current) break
              ## (and never more chunks than most_num_chunks, the N of the data)
              if ((num_chunks_next == num_chunks_current) || (num_chunks_next > most_num_chunks)) break
              num_chunks_pair <-  c(num_chunks_current, num_chunks_next)
              seconds_pair <-  fn_median_seconds_per_iteration_of_each_num_chunks(num_chunks_pair)
              n_timings <-  n_timings + 2
              ## (the latest timing of each chunk count is kept)
              timings <-  rbind(timings[!(timings$num_chunks %in% num_chunks_pair), , drop = FALSE],
                                data.frame( num_chunks                    = num_chunks_pair,
                                            median_seconds_per_iteration  = seconds_pair))
              if (seconds_pair[2] < seconds_pair[1]) {
                    num_chunks_current <-  num_chunks_next
              } else {
                    multiplier <-  0
                    ## (a flat step of the doubling: the doubling after it, as above)
                    if (num_chunks_next > num_chunks_current) {
                          num_chunks_past_a_tie <-  fn_num_chunks_past_a_tie_at_the_doubling(
                                                          num_chunks             = num_chunks_current,
                                                          seconds_of_num_chunks  = seconds_pair[1],
                                                          seconds_of_doubling    = seconds_pair[2])
                          if (!is.null(num_chunks_past_a_tie)) {
                                num_chunks_current <-  num_chunks_past_a_tie
                                multiplier <-  2
                          }
                    }
              }
        ##
        }
        timings <-  timings[order(timings$num_chunks), , drop = FALSE]
        rownames(timings) <-  NULL
        return(list( num_chunks        = num_chunks_current,
                     num_chunks_start  = fn_allowed_num_chunks(num_chunks_start),
                     timings           = timings,
                     n_timings         = n_timings))
##
}

##
##
#### ---- the leapfrog steps per timed iteration, so that every timing lasts about the same time -----------------
##
#' Leapfrog steps per timed iteration, so that each timing of a model lasts about minimum_seconds_per_timing
#'
#' Times one iteration of one leapfrog step (after one untimed iteration) with the given layout, i.e. about one
#' gradient per chain, and returns the leapfrog steps per iteration with which n_timed_iterations iterations take
#' about minimum_seconds_per_timing: most_leapfrog_steps (20, as in Paper 1's burn-in benchmark) for a fast model,
#' fewer for a slow one (at least 1). With these steps, every timing of every model, N and math backend lasts
#' about minimum_seconds_per_timing (a fast model runs more iterations, see
#' [fn_time_burnin_iteration_for_each_num_chunks()]), rather than many seconds for a slow model.
#'
#' @param init_object,Stan_data_list,n_chains,n_threads_WCP,num_chunks The model and the layout timed.
#' @param n_timed_iterations,minimum_seconds_per_timing The timings these steps are for.
#' @param most_leapfrog_steps The most leapfrog steps per iteration (default 20).
#' @return The leapfrog steps per timed iteration (a whole number from 1 to most_leapfrog_steps).
#' @export
fn_leapfrog_steps_per_timed_iteration_for_timings_of_at_least_minimum_seconds <-  function(
        init_object,
        Stan_data_list,
        n_chains,
        n_threads_WCP,
        num_chunks,
        n_timed_iterations,
        minimum_seconds_per_timing,
        most_leapfrog_steps  = 20
) {

        ##
        seconds_per_one_step_iteration <-  fn_time_burnin_iteration_for_each_num_chunks(
                                                 init_object                 = init_object,
                                                 Stan_data_list              = Stan_data_list,
                                                 n_chains_burnin             = n_chains,
                                                 n_threads_WCP_burnin        = n_threads_WCP,
                                                 num_chunks_candidates       = num_chunks,
                                                 n_rounds                    = 1,
                                                 n_timed_iterations          = 1,
                                                 minimum_seconds_per_timing  = 0,
                                                 L_main                      = 1
                                           )$median_seconds_per_burnin_iteration[1]
        leapfrog_steps <-  floor(minimum_seconds_per_timing /
                                 (n_timed_iterations * seconds_per_one_step_iteration))
        return(max(1, min(most_leapfrog_steps, leapfrog_steps)))
##
}






















