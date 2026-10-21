##
## ==== R_fn_hardware_counters_of_each_timed_layout_from_pmc_stat_snapshots.R ===================================
##
## The hardware counters of each timed layout, from the snapshots of Paper 1's counter tool pmc_stat taken just
## before and just after each timing call of fn_time_sampling_of_Stan_model_for_each_layout() (its argument
## fn_snapshot_hardware_counters): instructions per cycle, and L1 data fills from DRAM per gradient over all
## chains.
## With the one-call timing (time_layout_in_one_call = TRUE: no short call), the window is the whole call, its
## untimed warm-up iterations included, i.e. (n_iterations_of_untimed_warm_up_call + n_iterations_of_long_call)
## iterations of L leapfrog steps for each of the layout's chains. With Paper 1's two calls, the window is the
## long call minus the short call, as the timing itself, i.e. n_iterations_of_long_call -
## n_iterations_of_short_call iterations.
## (The same reading of the snapshots as the scripts
##  summarise_sampling_layout_grid_of_Stan_models_by_N_and_machine.R and
##  make_Paper_1_LaTeX_rows_of_smallest_N_with_chunking_fastest_by_machine_and_math_backend.R of Paper 3's
##  benchmark folder, which keep their own copies for Paper 1's two calls.)
##
##
## ------- The snapshots of one pmc_stat file:  ------------------------------------------------------------------
##
## (pmc_stat's snapshot lines: "<label>:snapshot_<k>", wall seconds, exit status, max RSS, then cycles,
##  instructions, fills from the local L2, the local L3, another CCX and DRAM; the last line of each key is kept,
##  so that a file appended by a restarted run keeps the snapshots of the restarted run)
fn_counter_snapshots_of_pmc_stat_file <-  function(pmc_stat_file) {

        lines <-  readLines(pmc_stat_file, warn = FALSE)
        lines <-  lines[grepl(":snapshot_", lines, fixed = TRUE)]
        fields <-  strsplit(trimws(lines), "[ ,]+")
        snapshots <-  data.frame( key = vapply(fields, function(x) x[1], character(1)),
                                  cycles = vapply(fields, function(x) as.numeric(x[5]), numeric(1)),
                                  instructions = vapply(fields, function(x) as.numeric(x[6]), numeric(1)),
                                  fills_DRAM = vapply(fields, function(x) as.numeric(x[10]), numeric(1)),
                                  stringsAsFactors = FALSE)
        return(snapshots[!duplicated(snapshots$key, fromLast = TRUE), ])
}

##
##
## ------- The counters of one timed layout (one row of fn_time_sampling_of_Stan_model_for_each_layout()):  ------
##
fn_hardware_counters_of_timed_layout <-  function(row,
                                                  snapshots,
                                                  label_of_counter_lines,
                                                  L,
##                                                  n_iterations_of_untimed_warm_up_call  = 1,
                                                  n_iterations_of_untimed_warm_up_call  = 0,
                                                  n_iterations_of_short_call            = 1) {

        fn_counters_at <-  function(k) {

                return(snapshots[match(paste0(label_of_counter_lines, ":snapshot_", k), snapshots$key), ])
        }

        window <-  fn_counters_at(row$counter_snapshot_after_long_call)[, -1] -
                   fn_counters_at(row$counter_snapshot_before_long_call)[, -1]
        n_iterations_of_window <-  n_iterations_of_untimed_warm_up_call + row$n_iterations_of_long_call
        ## (Paper 1's two calls: the long call minus the short call)
        if (!is.na(row$counter_snapshot_before_short_call)) {
                window <-  window - (fn_counters_at(row$counter_snapshot_after_short_call)[, -1] -
                                     fn_counters_at(row$counter_snapshot_before_short_call)[, -1])
                n_iterations_of_window <-  row$n_iterations_of_long_call - n_iterations_of_short_call
        }
        n_gradients_of_window <-  n_iterations_of_window * L * row$n_chains
        return(list(instructions_per_cycle   = window$instructions / window$cycles,
                    DRAM_fills_per_gradient  = window$fills_DRAM / n_gradients_of_window))
}






















