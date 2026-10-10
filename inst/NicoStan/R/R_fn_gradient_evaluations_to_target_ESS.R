##
## ==== R_fn_gradient_evaluations_to_target_ESS.R ===============================================================
##
## ---- The gradient evaluations a fit needs to reach a target ESS, from the efficiency_info of its summary ------
##
##
##
##
#' Gradient evaluations to a target ESS
#'
#' The gradient evaluations that a fit needs to reach target_ESS: all of its burn-in gradient evaluations
#' (n_grad_evals_burnin, all burn-in chains, with the pre-burn-in), plus its sampling gradient evaluations
#' (n_grad_evals_sampling, all sampling chains) scaled to the target, target_ESS / ESS x n_grad_evals_sampling,
#' where ESS is the fit's smallest ESS of the chosen type over the chosen quantities. A time proxy that does not
#' depend on the load of the computer.
#'
#' @param efficiency_info The efficiency_info of a fit's summary, or the list of its
#'   get_efficiency_metrics().
#' @param target_ESS The target ESS: one positive number.
#' @param quantities "generated_quantities" (default; e.g. the sensitivities, specificities and prevalences of a
#'   latent class model) or "main" (the main parameters used for the summary's diagnostics).
#' @param ESS_type "min_of_bulk_tail_and_sd" (default: the smallest of the three), "bulk", "tail" or "sd" (the ESS
#'   of the centred squared draws, ESS(theta^2), i.e. the ESS for the posterior SD).
#' @return The gradient evaluations, or NA when one of the ingredients is missing.
#' @examples
#' \dontrun{
#' fit_summary <- model_samples$summary()
#' fn_gradient_evaluations_to_target_ESS(efficiency_info = fit_summary$get_efficiency_metrics(),
#'                                       target_ESS = 1000)
#' }
#' @export
fn_gradient_evaluations_to_target_ESS <-  function( efficiency_info,
                                                    target_ESS,
                                                    quantities = "generated_quantities",
                                                    ESS_type = "min_of_bulk_tail_and_sd"
) {

        if (!is.list(efficiency_info)) {
              stop(paste0("fn_gradient_evaluations_to_target_ESS: efficiency_info must be the ",
                          "efficiency_info list of a summary or the list of its ",
                          "get_efficiency_metrics()."))
        }
        if (!is.numeric(target_ESS) || length(target_ESS) != 1 || !is.finite(target_ESS) || target_ESS <= 0) {
              stop("fn_gradient_evaluations_to_target_ESS: target_ESS must be one positive number.")
        }
        if (!identical(quantities, "generated_quantities") && !identical(quantities, "main")) {
              stop(paste0("fn_gradient_evaluations_to_target_ESS: quantities must be \"generated_quantities\" ",
                          "or \"main\"."))
        }
        ESS_types <-  c("min_of_bulk_tail_and_sd", "bulk", "tail", "sd")
        if (!is.character(ESS_type) || length(ESS_type) != 1 || !(ESS_type %in% ESS_types)) {
              stop(paste0("fn_gradient_evaluations_to_target_ESS: ESS_type must be one of ",
                          paste(ESS_types, collapse = ", "), "."))
        }
        ##
        ## ---- the smallest ESS of each type over the chosen quantities:
        ##
        fn_value_or_NA <-  function(value) {
              if (is.null(value) || length(value) != 1 || !is.finite(value)) NA_real_ else as.numeric(value)
        }
        ESS_by_type <-  if (identical(quantities, "generated_quantities")) {
              c( bulk = fn_value_or_NA(efficiency_info$Min_ESS_gq),
                 tail = fn_value_or_NA(efficiency_info$Min_ESS_tail_gq),
                 sd   = fn_value_or_NA(efficiency_info$Min_ESS_sd_gq))
        } else {
              c( bulk = fn_value_or_NA(efficiency_info$Min_ESS_main),
                 tail = fn_value_or_NA(efficiency_info$Min_ESS_tail_main),
                 sd   = fn_value_or_NA(efficiency_info$Min_ESS_sd_main))
        }
        ESS <-  if (identical(ESS_type, "min_of_bulk_tail_and_sd")) {
              if (anyNA(ESS_by_type)) NA_real_ else min(ESS_by_type)
        } else ESS_by_type[[ESS_type]]
        ##
        n_grad_evals_burnin <-  fn_value_or_NA(efficiency_info$n_grad_evals_burnin)
        n_grad_evals_sampling <-  fn_value_or_NA(efficiency_info$n_grad_evals_sampling)
        if (is.na(ESS) || ESS <= 0 || is.na(n_grad_evals_burnin) || is.na(n_grad_evals_sampling)) {
              return(NA_real_)
        }
        ##
        return(n_grad_evals_burnin + (target_ESS / ESS) * n_grad_evals_sampling)

}























