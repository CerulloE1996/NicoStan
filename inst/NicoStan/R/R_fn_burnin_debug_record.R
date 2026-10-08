#### ===========================================================================================================
## R_fn_burnin_debug_record.R
##
## ---- the burn-in debug record of the tau updates (debug = TRUE only; 6 Oct 2026) ---------------------------
##
##      One entry per tau update of the main or the joint block (record$records), for offline tests of
##      trajectory-length criteria on the real models. Each entry holds the iteration, the block and its number
##      of rows, the centre and the proposal centre of the main rows, the index of the main rows' trajectory
##      metric factor in record$metric_factors_main (each distinct factor is kept once), the nominal tau before
##      the update, the step size of the update's trajectories and of the next ones, whether the criterion used
##      the proposals (weighted by their acceptance probabilities) and the interest rows. With the per-iteration
##      record of R_fn_init_and_run_burnin_CHESS.R (burnin_start_main_all_chains, burnin_proposal_main_all_chains,
##      burnin_trace_main_all_chains, burnin_p_jump_main_all_chains, ...) it rebuilds the per-coordinate inputs of
##      every criterion of the main block.
##
##      "LQ_ESSR_spectral" on the main block (update$spectral_update_inputs, returned by
##      fn_metric_tau_block_update() when asked): also the complete arguments of
##      fn_spectral_ESS_soft_minimum_tau_update() except the state it carries between updates (with whether
##      that state was NULL, as after the burn-in's reset at the handover), and its result
##      except that state; at the first performed update and at every 25th spectral update, the full replay
##      record: the complete arguments including the state before the update, the per-coordinate criterion
##      inputs and the complete result. The state (bins and moving averages of every monitored coordinate) is too
##      large to keep at every update; replaying the updates in between from the saved arguments rebuilds it.
##      The joint block's state has a row per nuisance parameter: it is never kept.
##
##      A failure here is added to record$errors and never stops or changes the burn-in.
##
fn_burnin_debug_record_add_tau_update <-  function( record,
                                                    iteration,
                                                    block,
                                                    n_rows_in_block,
                                                    centre_main,
                                                    proposal_centre_main,
                                                    metric_factor_of_block,
                                                    tau_before_update,
                                                    eps_used_for_trajectories,
                                                    eps_now,
                                                    use_proposals,
                                                    interest_rows,
                                                    update) {

        new_record <-  tryCatch({
                ##
                ## ---- the metric factor of the main rows (the joint block's factor holds them as $main), kept
                ##      once per distinct factor:
                metric_factor_main <-  if (identical(block, "joint")) metric_factor_of_block$main else
                                                                      metric_factor_of_block
                n_factors <-  length(record$metric_factors_main)
                if (n_factors == 0 || !identical(record$metric_factors_main[[n_factors]], metric_factor_main)) {
                        record$metric_factors_main[[n_factors + 1]] <-  metric_factor_main
                        n_factors <-  n_factors + 1
                }
                ##
                entry <-  list(iteration                 = iteration,
                               block                     = block,
                               n_rows_in_block           = n_rows_in_block,
                               centre_main               = centre_main,
                               proposal_centre_main      = proposal_centre_main,
                               metric_factor_main_index  = n_factors,
                               tau_before_update         = tau_before_update,
                               eps_used_for_trajectories = eps_used_for_trajectories,
                               eps_now                   = eps_now,
                               use_proposals             = use_proposals,
                               interest_rows             = interest_rows,
                               adam_update_performed     = isTRUE(update$adam_update_performed))
                ##
                ## ---- "LQ_ESSR_spectral", main block: the update's arguments and result (see above):
                spectral_inputs <-  update$spectral_update_inputs
                if (!is.null(spectral_inputs)) {
                        is_spectral_entry <-  vapply(record$records, function(previous) {
                                !is.null(previous$spectral_update)
                        }, logical(1))
                        was_performed <-  vapply(record$records, function(previous) {
                                isTRUE(previous$adam_update_performed)
                        }, logical(1))
                        n_spectral_updates <-  sum(is_spectral_entry) + 1
                        first_performed <-  isTRUE(update$adam_update_performed) &&
                                            !any(is_spectral_entry & was_performed)
                        arguments <-  spectral_inputs$tau_update_arguments
                        entry$spectral_update <-  list(
                              ## (the burn-in resets the carried state to NULL at the handover)
                              state_before_update_is_null = is.null(arguments[["state"]]),
                              arguments_except_state = arguments[names(arguments) != "state"],
                              result_except_state    = update[!names(update) %in% c("component_criterion_ema",
                                                                                      "spectral_update_inputs")])
                        if (first_performed || n_spectral_updates %% 25 == 0) {
                                entry$spectral_update_full_replay_record <-  list(
                                      tau_update_arguments  = arguments,
                                      live_criterion_inputs = spectral_inputs$live_criterion_inputs,
                                      live_update_result    = update[names(update) != "spectral_update_inputs"])
                        }
                }
                ##
                record$records[[length(record$records) + 1]] <-  entry
                record
        }, error = function(error_object) {
                record$errors <-  c(record$errors, paste0("iteration ", iteration, ": ",
                                                          conditionMessage(error_object)))
                record
        })
        ##
        return(new_record)

}
























