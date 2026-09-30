#### =====================================================================================================================================
## R_fn_harmonic_mean_acceptance.R
##
## ---- Harmonic mean over the burn-in chains of the per-chain acceptance probabilities, for the step-size (eps) adaptation ---------------
##
## The burn-in adapts ONE eps per block, shared by every burn-in chain, with ADAM on log(eps) and the synthetic
## gradient  g = alpha_bar - adapt_delta  (EHMC_adapt_eps_fn.hpp). alpha_bar is a mean over the chains of the per-chain
## acceptance probabilities alpha_k (p_jump) of the current iteration, as the C++ samplers record them: alpha_k =
## min(1, exp(log_ratio_k)) for a proposal that check_divergence() does not flag, and alpha_k = 0 for a flagged (divergent)
## proposal, which is rejected (see "alpha_k = 0" below). The burn-in argument eps_acceptance_mean chooses the mean:
##
##   "harmonic"   - alpha_bar = K / sum_k (1 / alpha_k), computed here.
##                  Hoffman, Radul & Sountsov (2021), Algorithm 1 ("Compute harmonic-mean acceptance probability"), and
##                  Sountsov & Hoffman (2022), Appendix, eq. (15). One chain with a very low acceptance probability pulls
##                  alpha_bar down and so reduces eps for all the chains, so that a "straggler" chain is not left stalled
##                  at an eps tuned for the others.
##   "arithmetic" - alpha_bar = mean(alpha_k, na.rm = TRUE), the rule used before this option existed (computed in the
##                  burn-in itself, as p_jump_main / p_jump_us).
##   "geometric"  - alpha_bar = exp(mean(log(alpha_k))), with each alpha_k first clamped to [1e-8, 1]
##                  (fn_geometric_mean_acceptance below). It lies between the harmonic and the arithmetic mean: a chain with
##                  zero acceptance pulls alpha_bar down (to at most 1e-8^(1/K) times the geometric mean of the other chains)
##                  without forcing it to exactly 0, as the harmonic mean does.
##
## Zero and missing acceptance probabilities:
##
##   alpha_k = 0  - either a divergent proposal or an acceptance probability that underflows to 0:
##                  - check_divergence() (EHMC_main_sampler_fns.hpp; also used by the nuisance and dual samplers) flags
##                    the proposal when the state is invalid, the log ratio is non-finite, the log density or gradient
##                    holds a NaN / Inf, or |log_ratio| > 1000. The C++ then records div = 1 and p_jump = 0 and rejects.
##                    The |log_ratio| > 1000 rule applies in BOTH directions, so a very large energy DECREASE
##                    (log_ratio > +1000) also gives alpha_k = 0, where ChEES (Hoffman, Radul & Sountsov 2021, Algorithm 1),
##                    SNAPER (Sountsov & Hoffman 2022, eq. (15)) and TensorFlow Probability all have
##                    alpha_k = min(1, exp(log_ratio_k)) = 1 (TFP's hmc_like_log_accept_prob_getter_fn maps only a
##                    NON-FINITE log accept ratio to -Inf, then takes min(., 0)).
##                  - a proposal that is not flagged but has log_ratio below about -745 gives exp(log_ratio) = 0 (or a
##                    denormal number) with div = 0, i.e. alpha_k = 0 (or effectively 0) without a divergence.
##                  Such a chain is KEPT: with floor = 0 (the value the burn-in uses) 1 / 0 = Inf, so the harmonic mean is
##                  exactly 0, i.e. g = -adapt_delta, the largest downward push on eps (a denormal alpha_k gives a harmonic
##                  mean of at most K * alpha_k, i.e. 0 or effectively 0). This is the limit of K / sum_k (1 / alpha_k) as
##                  any alpha_k -> 0, and is what TensorFlow Probability's reduce_log_harmonic_mean_exp returns for
##                  log(alpha_k) = -Inf; it is NOT what TFP returns for a finite log_ratio > +1000 (there alpha_k = 1). The
##                  value is exactly 0, which is finite, so the eps update never receives NaN from a zero chain. A positive
##                  floor bounds each alpha_k below before it is inverted (pmax(alpha_k, floor)).
##   alpha_k = NA - no acceptance probability was returned for that chain (a failed read on the R side, NOT a rejected
##                  proposal, which the C++ already records as 0). NA / NaN values are removed, as in the arithmetic
##                  rule. If no value remains the result is NA, and the eps update keeps eps and its ADAM moments
##                  unchanged (the NaN guard in adapt_eps_ADAM), as the arithmetic rule does for an empty vector.
##
## Values are clamped to [floor, 1] before averaging; the samplers already return values in [0, 1].
##
#' Harmonic mean over the chains of the per-chain acceptance probabilities (burn-in step-size adaptation)
#'
#' @param p Numeric vector of per-chain Metropolis acceptance probabilities for one burn-in iteration.
#' @param floor Lower bound applied to each acceptance probability before it is inverted; default 0 (no floor: a chain
#'   with zero acceptance makes the harmonic mean exactly 0).
#' @return A single number, K / sum_k (1 / pmax(p_k, floor)) over the K non-missing entries; NA when every entry is missing.
#' @noRd
fn_harmonic_mean_acceptance <-  function( p,
                                          floor = 0) {

        if (!is.numeric(floor) || length(floor) != 1 || !is.finite(floor) || floor < 0 || floor >= 1) {
              stop(paste0("fn_harmonic_mean_acceptance: floor must be a single number in [0, 1); got: ",
                          paste(as.character(floor), collapse = ", ")))
        }
        ##
        p <-  as.numeric(p)
        p <-  p[!is.na(p)]    ## NA / NaN = missing, removed (see above)
        ##
        n_chains_with_value <-  length(p)
        if (n_chains_with_value == 0) {
              return(NA_real_)
        }
        ##
        p <-  pmin(pmax(p, floor), 1)
        ##
        ## any chain with zero acceptance (divergent proposal, including |log_ratio| > 1000 in either direction, or an
        ## exp(log_ratio) that underflowed to 0; floor = 0) makes the harmonic mean exactly 0:
        if (any(p == 0)) {
              return(0)
        }
        ##
        return(n_chains_with_value / sum(1 / p))

}
##
##
## ---- Geometric mean over the burn-in chains of the per-chain acceptance probabilities (eps_acceptance_mean = "geometric") ---------
##
## alpha_bar = exp(mean(log(alpha_k))) over the K non-missing entries, each alpha_k clamped to [lower_clamp, 1] first
## (lower_clamp = 1e-8), so that a chain with zero acceptance (divergent or underflowed, see above) gives log(1e-8) instead of
## -Inf. NA / NaN values are removed as in the harmonic and arithmetic rules, and an empty / all-missing vector gives NA (the
## NaN guard in adapt_eps_ADAM then keeps eps and its ADAM moments unchanged).
##
#' Geometric mean over the chains of the per-chain acceptance probabilities (burn-in step-size adaptation)
#'
#' @param p Numeric vector of per-chain Metropolis acceptance probabilities for one burn-in iteration.
#' @param lower_clamp Lower bound applied to each acceptance probability before its log is taken; default 1e-8.
#' @return A single number, exp(mean(log(pmin(pmax(p_k, lower_clamp), 1)))) over the K non-missing entries; NA when every
#'   entry is missing.
#' @noRd
fn_geometric_mean_acceptance <-  function( p,
                                           lower_clamp = 1e-8) {

        if (!is.numeric(lower_clamp) || length(lower_clamp) != 1 || !is.finite(lower_clamp) || lower_clamp <= 0 || lower_clamp >= 1) {
              stop(paste0("fn_geometric_mean_acceptance: lower_clamp must be a single number in (0, 1); got: ",
                          paste(as.character(lower_clamp), collapse = ", ")))
        }
        ##
        p <-  as.numeric(p)
        p <-  p[!is.na(p)]    ## NA / NaN = missing, removed (as in fn_harmonic_mean_acceptance)
        ##
        n_chains_with_value <-  length(p)
        if (n_chains_with_value == 0) {
              return(NA_real_)
        }
        ##
        p <-  pmin(pmax(p, lower_clamp), 1)
        ##
        return(exp(mean(log(p))))

}
























