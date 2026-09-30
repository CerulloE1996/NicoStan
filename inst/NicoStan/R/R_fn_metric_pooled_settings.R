#### =====================================================================================================================================
## R_fn_metric_pooled_settings.R
##
## ---- Settings of the pooled metric estimator (metric_estimator = "pooled") in the burn-in ----------------------------------------------
##
## The pooled estimator accumulates Welford moments of the draws of ALL burn-in chains, one iteration at a time, from
## metric_start_iter to metric_adaptation_end_iter (R_fn_init_and_run_burnin_CHESS.R). Its main-block covariance proposal is
##
##     empicical_cov_main = w * cov_draws + (1 - w) * 1e-3 * I,      w = wf_n / (wf_n + 5)     (Stan-style regularisation)
##
## Its per-coordinate variances var_draws_all[index_nuisance] feed the NUISANCE diagonal metric (metric_type_nuisance =
## "Empirical" or "uniform_diag"). The main diagonal metric does not read the pooled moments: it reads
## snaper_s_vec_main_empirical (update_M_Empirical_main). Two settings:
##
##   metric_pooled_window_resets          - iterations at which the Welford accumulators (wf_n, wf_m, wf_M2, wf_C2, and the
##                                          nuisance statistics held by the resident worker) are reset to zero. The resets
##                                          therefore apply to the pooled main covariance AND to the pooled nuisance
##                                          variances (var_draws_all[index_nuisance]):
##                                            "stan_style" - (default; NULL = "stan_style") round(c(0.30, 0.60) * n_adapt),
##                                                           the resets used by every run before this option existed; the
##                                                           final metric then uses only the draws after iteration
##                                                           round(0.60 * n_adapt).
##                                            "none"       - no reset: ONE window from metric_start_iter to
##                                                           metric_adaptation_end_iter, so the final metric uses every
##                                                           pooled draw in that range.
##                                            numeric      - whole numbers r >= 0, e.g. c(50, 100).
##                                          For each value r the accumulators are reset at the START of iteration r + 1, so the
##                                          draws up to and including iteration r are discarded (the convention of the original
##                                          code, "ii %in% (metric_window_resets + 1)"). A value outside
##                                          [metric_start_iter - 1, metric_adaptation_end_iter - 1] has no effect.
##
##   metric_pooled_offdiagonal_shrinkage  - s in [0, 1], applied ONCE to each pooled covariance proposal, after the Stan-style
##                                          regularisation above and before the symmetrisation:
##                                            empicical_cov_main = (1 - s) * empicical_cov_main + s * diag(diag(empicical_cov_main))
##                                          i.e. every off-diagonal element is multiplied by (1 - s) and the diagonal is kept.
##                                          s = 0 (default) keeps all off-diagonals unshrunk (every run before this option
##                                          existed); s = 1 gives a diagonal metric with the pooled variances. The nuisance
##                                          metric is diagonal, so s does not change it.
##
## Neither setting is read by metric_estimator = "chain_mean" / "chain_mean_scaled" (whose covariance is shrunk every
## iteration by Rcpp_shrink_matrix(shrinkage_factor)). The earlier behaviour is reproduced exactly by
## metric_pooled_window_resets = "stan_style" with metric_pooled_offdiagonal_shrinkage = 0, which are also the defaults, so a
## caller that passes neither setting runs exactly as before.
##
#' Validate metric_pooled_window_resets (pooled metric estimator)
#'
#' @param metric_pooled_window_resets NULL or "stan_style" (round(c(0.30, 0.60) * n_adapt), the default), "none" (no reset;
#'   one window), or a numeric vector of whole numbers >= 0 (reset at the start of iteration r + 1 for each value r).
#' @return "none", "stan_style" or a plain numeric vector. NULL returns "stan_style" (the earlier rule); a zero-length numeric
#'   vector (no reset iterations) returns "none".
#' @noRd
fn_validate_metric_pooled_window_resets <-  function(metric_pooled_window_resets) {

        if (is.null(metric_pooled_window_resets)) {
              return("stan_style")
        }
        ##
        if (is.character(metric_pooled_window_resets)) {
              if (length(metric_pooled_window_resets) != 1 || is.na(metric_pooled_window_resets) ||
                  !(metric_pooled_window_resets %in% c("none", "stan_style"))) {
                    stop(paste0("metric_pooled_window_resets must be 'none', 'stan_style' or a numeric vector of whole numbers >= 0; got: ",
                                paste(as.character(metric_pooled_window_resets), collapse = ", ")))
              }
              return(metric_pooled_window_resets)
        }
        ##
        if (is.numeric(metric_pooled_window_resets) && !is.factor(metric_pooled_window_resets)) {
              if (length(metric_pooled_window_resets) == 0) {
                    return("none")
              }
              if (any(!is.finite(metric_pooled_window_resets)) ||
                  any(metric_pooled_window_resets < 0) ||
                  any(metric_pooled_window_resets != round(metric_pooled_window_resets))) {
                    stop(paste0("metric_pooled_window_resets: a numeric value must hold whole numbers >= 0 (iterations); got: ",
                                paste(as.character(metric_pooled_window_resets), collapse = ", ")))
              }
              return(as.numeric(metric_pooled_window_resets))
        }
        ##
        stop(paste0("metric_pooled_window_resets must be 'none', 'stan_style' or a numeric vector of whole numbers >= 0; got an object of class ",
                    paste(class(metric_pooled_window_resets), collapse = "/")))

}



#' Reset iterations of the pooled metric estimator
#'
#' @param metric_pooled_window_resets As for fn_validate_metric_pooled_window_resets.
#' @param n_adapt Number of adaptation iterations of the burn-in (after the burn-in schedule is resolved).
#' @return Numeric vector of iterations r; the accumulators are reset at the start of iteration r + 1. numeric(0) for "none".
#' @noRd
fn_resolve_metric_pooled_window_reset_iterations <-  function( metric_pooled_window_resets,
                                                               n_adapt) {

        metric_pooled_window_resets <-  fn_validate_metric_pooled_window_resets(metric_pooled_window_resets)
        ##
        if (identical(metric_pooled_window_resets, "none")) {
              return(numeric(0))
        }
        ##
        if (identical(metric_pooled_window_resets, "stan_style")) {
              if (!is.numeric(n_adapt) || length(n_adapt) != 1 || !is.finite(n_adapt)) {
                    stop(paste0("fn_resolve_metric_pooled_window_reset_iterations: n_adapt must be a single finite number; got: ",
                                paste(as.character(n_adapt), collapse = ", ")))
              }
              ## the same expression as the original code:
              return(round(c(0.30, 0.60) * n_adapt))
        }
        ##
        return(metric_pooled_window_resets)

}



#' Validate metric_pooled_offdiagonal_shrinkage (pooled metric estimator)
#'
#' @param metric_pooled_offdiagonal_shrinkage A single number s in [0, 1]; the callers' default is 0 (the earlier rule).
#' @return s as a plain number.
#' @noRd
fn_validate_metric_pooled_offdiagonal_shrinkage <-  function(metric_pooled_offdiagonal_shrinkage) {

        if (!is.numeric(metric_pooled_offdiagonal_shrinkage) || is.factor(metric_pooled_offdiagonal_shrinkage) ||
            length(metric_pooled_offdiagonal_shrinkage) != 1 || !is.finite(metric_pooled_offdiagonal_shrinkage) ||
            metric_pooled_offdiagonal_shrinkage < 0 || metric_pooled_offdiagonal_shrinkage > 1) {
              stop(paste0("metric_pooled_offdiagonal_shrinkage must be a single number in [0, 1]; got: ",
                          paste(as.character(metric_pooled_offdiagonal_shrinkage), collapse = ", ")))
        }
        ##
        return(as.numeric(metric_pooled_offdiagonal_shrinkage))

}



#' Shrink the off-diagonal elements of a pooled covariance proposal towards zero
#'
#' Computes (1 - s) * covariance_matrix + s * diag(diag(covariance_matrix)): each off-diagonal element is multiplied by
#' (1 - s) and the diagonal is copied unchanged (so it is exact, not (1 - s) * d + s * d in floating point). s = 0 returns
#' covariance_matrix itself, unchanged.
#'
#' @param covariance_matrix Square numeric matrix (the pooled main-block covariance proposal).
#' @param metric_pooled_offdiagonal_shrinkage A single number s in [0, 1].
#' @return A matrix of the same dimensions.
#' @noRd
fn_shrink_metric_pooled_offdiagonal <-  function( covariance_matrix,
                                                  metric_pooled_offdiagonal_shrinkage) {

        if (metric_pooled_offdiagonal_shrinkage == 0) {
              return(covariance_matrix)
        }
        ##
        covariance_diagonal <-  diag(covariance_matrix)
        covariance_matrix   <-  (1 - metric_pooled_offdiagonal_shrinkage) * covariance_matrix
        diag(covariance_matrix) <-  covariance_diagonal
        ##
        return(covariance_matrix)

}
























