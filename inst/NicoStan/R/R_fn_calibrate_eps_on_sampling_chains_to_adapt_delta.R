#### =============================================================================================================
## R_fn_calibrate_eps_on_sampling_chains_to_adapt_delta.R
##
## ---- Step-size (eps) calibration on the sampling chains, between the burn-in and the sampling
##
## The burn-in adapts eps on its few burn-in chains (e.g. 4), one transition per chain per iteration, with ADAM
## on log(eps). The per-transition acceptance probability has a standard deviation of about 0.2 around a mean of
## 0.8, so at the end of a short burn-in (e.g. N_burn = 125, about 20-40 iterations after the final metric update)
## the burn-in has measured the acceptance probability at its final eps only to about +-0.02 to +-0.035 (one
## standard deviation), and the eps it hands over can give a sampling acceptance well away from adapt_delta.
##
## This calibration measures the acceptance probability on the SAMPLING chains instead (e.g. 180 chains on the
## local-HPC, 16 on the laptop), with enough transitions to know it to about +-0.01, and sets the sampling eps so
## that the mean acceptance probability over the sampling chains is adapt_delta:
##
##   - block b = 1, ..., n_eps_calibration_blocks: ceiling(n_transitions_per_eps_calibration_block /
##     n_chains_sampling) iterations of every sampling chain at eps_b, with a seed of its own. The chains run on
##     from one block to the next (block 1 from the initial states of the sampling, i.e. the burn-in's final
##     states), and the sampling continues from the end of the last block. Their draws are discarded. The chains
##     run on, rather than each block starting again from the burn-in's final states, because the sampling chains
##     start from only a few distinct states (one per burn-in chain), and the acceptance probability depends on
##     where a chain is: blocks that all started from those few states measured the acceptance near them, which
##     differed from the acceptance over the sampling, after the chains had moved away from them, by up to about
##     +-0.03 (binary LC-MVP, N = 2500, 16 sampling chains);
##   - the mean acceptance probability of each block over all its transitions, alpha_bar_b, enters the curve
##
##         alpha(eps) = 2 Phi(-c eps^2),
##
##     the acceptance probability of HMC with the leapfrog integrator in high dimension (the energy error is
##     approximately N(mu, 2 mu) with mu proportional to eps^4; Neal 2011; Beskos, Pillai, Roberts, Sanz-Serna &
##     Stuart 2013). The one unknown, c, solves the pooled moment equation
##
##         sum_b n_b (alpha_bar_b - 2 Phi(-c eps_b^2)) = 0      (n_b = transitions of block b),
##
##     over every informative block so far except the first (whose chains are still close to their few starting
##     states; it is used alone only for the eps of the next block), and the next eps is the one with alpha(eps) =
##     adapt_delta:
##
##         eps_{b + 1} = sqrt(-qnorm(adapt_delta / 2) / c)      (capped at max_eps);
##
##   - a block with alpha_bar_b above 0.99 or below 0.01 carries almost no information on c: eps is doubled or
##     halved instead, and the block does not count towards n_eps_calibration_blocks (at most 6 such blocks).
##
## The eps after the last block is the sampling eps. Each block's iterations are gradient evaluations of the
## sampling chains before the sampling, counted with the burn-in's gradient evaluations by the summary (from the
## blocks returned here); their time is counted with the burn-in time.
##
## With partitioned HMC (a separate nuisance eps), the nuisance eps is calibrated the same way, from the nuisance
## acceptance probabilities, in the same blocks.
##
#' Calibrate the sampling eps on the sampling chains so that their mean acceptance probability is adapt_delta
#'
#' @param fn_run_sampling_iterations_continuing_the_chains function(n_iter, eps_main, eps_us, seed) that runs
#'   n_iter iterations of every sampling chain, with these step sizes, from the chains' last states (the initial
#'   states of the sampling at the first call), keeps the chains' new last states for the next call and the
#'   sampling, and returns a list whose element sampling_diagnostics holds p_jump_main and p_jump_nuisance, each
#'   a matrix of n_iter rows and one column per chain.
#' @param eps_main,eps_us The step sizes handed over by the burn-in.
#' @param calibrate_eps_us TRUE (partitioned HMC) also calibrates eps_us from the nuisance acceptance
#'   probabilities.
#' @param adapt_delta The target mean acceptance probability.
#' @param n_chains_sampling Number of sampling chains.
#' @param n_transitions_per_eps_calibration_block Transitions (iterations x sampling chains) of each block, at
#'   least; the iterations of a block are ceiling(n_transitions_per_eps_calibration_block / n_chains_sampling).
#' @param n_eps_calibration_blocks Number of informative blocks.
#' @param max_eps_main,max_eps_us Upper bounds of the step sizes.
#' @param seed The sampling seed; block b runs with its own seed derived from it (never the sampling seed).
#' @return A list: eps_main and eps_us for the sampling, c_main and c_us (the fitted constants), and blocks, a
#'   data frame with one row per block (block, n_iter, n_chains, eps_main, mean_acceptance_main, eps_us,
#'   mean_acceptance_us, used_in_fit).
#' @noRd
##
## ---- the acceptance of a block or call: the HARMONIC mean over the chains at each iteration, K / sum_k (1 /
##      alpha_k) (0 when a chain has zero acceptance, e.g. a divergent proposal), averaged over the iterations, as
##      the burn-in adapts eps (eps_acceptance_mean = "harmonic"), so that adapt_delta means the same in the
##      over every transition, which gave a larger sampling eps than the burn-in's for the same adapt_delta:
##
fn_mean_over_iterations_of_the_harmonic_mean_acceptance_over_chains <-  function(p_jump) {

        p_jump <-  as.matrix(p_jump)
        harmonic_mean_of_each_iteration <-  apply(p_jump, 1, function(alpha) {

              alpha <-  alpha[is.finite(alpha)]
              if (length(alpha) == 0) return(NA_real_)
              if (any(alpha <= 0)) return(0)
              length(alpha) / sum(1 / alpha)
        })

        mean(harmonic_mean_of_each_iteration, na.rm = TRUE)
}

fn_calibrate_eps_on_sampling_chains_to_adapt_delta <-  function( fn_run_sampling_iterations_continuing_the_chains,
                                                                 eps_main,
                                                                 eps_us,
                                                                 calibrate_eps_us,
                                                                 adapt_delta,
                                                                 n_chains_sampling,
                                                                 n_transitions_per_eps_calibration_block,
                                                                 n_eps_calibration_blocks,
                                                                 max_eps_main,
                                                                 max_eps_us,
                                                                 seed) {

        ##
        if (!is.numeric(adapt_delta) || length(adapt_delta) != 1 || !(adapt_delta > 0 && adapt_delta < 1)) {
              stop("eps calibration: adapt_delta must be a single number in (0, 1).")
        }
        if (!is.numeric(n_transitions_per_eps_calibration_block) ||
            length(n_transitions_per_eps_calibration_block) != 1 || n_transitions_per_eps_calibration_block < 1) {
              stop("n_transitions_per_eps_calibration_block must be a single number >= 1.")
        }
        if (!is.numeric(n_eps_calibration_blocks) || length(n_eps_calibration_blocks) != 1 ||
            n_eps_calibration_blocks < 1) {
              stop("n_eps_calibration_blocks must be a single number >= 1.")
        }
        ##
        n_iter_per_block <-  ceiling(n_transitions_per_eps_calibration_block / n_chains_sampling)
        max_uninformative_blocks <-  6
        ## (the eps of 2 Phi(-c eps^2) = adapt_delta)
        minus_qnorm_of_half_adapt_delta <-  -stats::qnorm(adapt_delta / 2)
        ##
        ## ---- c from the informative blocks so far: the root of the pooled moment equation (increasing in c):
        ##
        fn_fit_c <-  function( eps_of_blocks,
                               mean_acceptance_of_blocks,
                               n_transitions_of_blocks) {

              fn_moment_equation <-  function(log_c) {

                    sum(n_transitions_of_blocks *
                        (mean_acceptance_of_blocks - 2 * stats::pnorm(-exp(log_c) * eps_of_blocks^2)))
              }

              root <-  stats::uniroot(f = fn_moment_equation, lower = -60, upper = 60, tol = 1e-10)
              return(exp(root$root))
        }

        ##
        blocks <-  list()
        n_informative_blocks <-  0
        n_uninformative_blocks <-  0
        eps_main_of_block <-  eps_main
        eps_us_of_block <-  eps_us
        c_main <-  NA_real_
        c_us <-  NA_real_
        block <-  0
        while ((n_informative_blocks < n_eps_calibration_blocks) &&
               (n_uninformative_blocks <= max_uninformative_blocks)) {
              ##
              block <-  block + 1
              ## (a seed of its own for each block, never the sampling seed, within the range of an R integer)
              seed_of_block <-  ((seed + 1000003 * block) %% 2147483000) + 1
              result_of_block <-  fn_run_sampling_iterations_continuing_the_chains( n_iter   = n_iter_per_block,
                                                                                    eps_main = eps_main_of_block,
                                                                                    eps_us   = eps_us_of_block,
                                                                                    seed     = seed_of_block)
              # p_jump_main <-  as.numeric(result_of_block$sampling_diagnostics$p_jump_main)
              # mean_acceptance_main <-  mean(p_jump_main, na.rm = TRUE)
              mean_acceptance_main <-  fn_mean_over_iterations_of_the_harmonic_mean_acceptance_over_chains(
                                           result_of_block$sampling_diagnostics$p_jump_main)
              mean_acceptance_us <-  NA_real_
              if (isTRUE(calibrate_eps_us)) {
                    # mean_acceptance_us <- 
                    #  mean(as.numeric(result_of_block$sampling_diagnostics$p_jump_nuisance),
                    #                             na.rm = TRUE)
                    mean_acceptance_us <-  fn_mean_over_iterations_of_the_harmonic_mean_acceptance_over_chains(
                                                 result_of_block$sampling_diagnostics$p_jump_nuisance)
              }
              rm(result_of_block)
              ##
              block_is_informative <-  is.finite(mean_acceptance_main) &&
                                       (mean_acceptance_main <= 0.99) && (mean_acceptance_main >= 0.01)
              if (isTRUE(calibrate_eps_us)) {
                    block_is_informative <-  block_is_informative && is.finite(mean_acceptance_us) &&
                                             (mean_acceptance_us <= 0.99) && (mean_acceptance_us >= 0.01)
              }
              blocks[[block]] <-  data.frame( block = block,
                                              n_iter = n_iter_per_block,
                                              n_chains = n_chains_sampling,
                                              eps_main = eps_main_of_block,
                                              mean_acceptance_main = mean_acceptance_main,
                                              eps_us = eps_us_of_block,
                                              mean_acceptance_us = mean_acceptance_us,
                                              used_in_fit = block_is_informative,
                                              stringsAsFactors = FALSE)
              message(colourise(paste0("eps calibration on the ", n_chains_sampling, " sampling chains, block ",
                                       block, " (", n_iter_per_block, " iterations): eps_main = ",
                                       signif(eps_main_of_block, 4), ", mean acceptance = ",
                                       round(mean_acceptance_main, 3),
                                       if (isTRUE(calibrate_eps_us)) {
                                             paste0("; eps_us = ", signif(eps_us_of_block, 4),
                                                    ", mean acceptance = ", round(mean_acceptance_us, 3))
                                       } else ""),
                                "green"))
              ##
              if (!block_is_informative) {
                    ## (doubled when nearly every proposal is accepted, halved when nearly every one is rejected,
                    ##  or when the acceptance probabilities are missing)
                    n_uninformative_blocks <-  n_uninformative_blocks + 1
                    eps_main_of_block <-  if (isTRUE(mean_acceptance_main > 0.99)) min(max_eps_main,
                                                                                       2 * eps_main_of_block) else
                                          eps_main_of_block / 2
                    if (isTRUE(calibrate_eps_us)) {
                          eps_us_of_block <-  if (isTRUE(mean_acceptance_us > 0.99)) min(max_eps_us,
                                                                                         2 * eps_us_of_block) else
                                              eps_us_of_block / 2
                    }
                    next
              }
              ##
              n_informative_blocks <-  n_informative_blocks + 1
              blocks_so_far <-  do.call(rbind, blocks)
              blocks_so_far <-  blocks_so_far[blocks_so_far$used_in_fit, , drop = FALSE]
              ## (the first informative block alone sets the eps of the next block; every later eps comes from the
              ##  informative blocks after it)
              if (nrow(blocks_so_far) >= 2) {
                    blocks_so_far <-  blocks_so_far[-1, , drop = FALSE]
              }
              n_transitions_of_blocks <-  blocks_so_far$n_iter * blocks_so_far$n_chains
              c_main <-  fn_fit_c( eps_of_blocks = blocks_so_far$eps_main,
                                   mean_acceptance_of_blocks = blocks_so_far$mean_acceptance_main,
                                   n_transitions_of_blocks = n_transitions_of_blocks)
              eps_main_of_block <-  min(max_eps_main, sqrt(minus_qnorm_of_half_adapt_delta / c_main))
              if (isTRUE(calibrate_eps_us)) {
                    c_us <-  fn_fit_c( eps_of_blocks = blocks_so_far$eps_us,
                                       mean_acceptance_of_blocks = blocks_so_far$mean_acceptance_us,
                                       n_transitions_of_blocks = n_transitions_of_blocks)
                    eps_us_of_block <-  min(max_eps_us, sqrt(minus_qnorm_of_half_adapt_delta / c_us))
              }
        ##
        }
        ##
        if (n_informative_blocks < n_eps_calibration_blocks) {
              stop(paste0("eps calibration on the sampling chains: only ", n_informative_blocks, " of ",
                          n_eps_calibration_blocks, " blocks had a mean acceptance probability in [0.01, 0.99] ",
                          "after ", block, " blocks (last eps_main = ", signif(eps_main_of_block, 4), ")."))
        }
        message(colourise(paste0("eps calibration: sampling eps_main = ", signif(eps_main_of_block, 4),
                                 " (burn-in: ", signif(eps_main, 4), "), for a mean acceptance of ",
                                 adapt_delta,
                                 if (isTRUE(calibrate_eps_us)) paste0("; eps_us = ", signif(eps_us_of_block, 4),
                                                                      " (burn-in: ", signif(eps_us, 4), ")")
                                 else ""),
                          "blue"))
        ##
        return(list( eps_main = eps_main_of_block,
                     eps_us = eps_us_of_block,
                     c_main = c_main,
                     c_us = c_us,
                     blocks = do.call(rbind, blocks)))
##
}

##
##
#### =============================================================================================================
##
## ---- The one-call version (the calibration that $sample() uses)
##
## One sampling call of every sampling chain, from the initial states of the sampling (the burn-in's final
## states), at the eps handed over by the burn-in, without any nuisance trace, of n_eps_calibration_blocks blocks
## of ceiling(n_transitions_per_eps_calibration_block / n_chains_sampling) iterations, cut to at most
## max_n_iter_of_eps_calibration_call iterations in all: its first block of iterations is discarded,
## whilst the chains move away from their few distinct starting states (one per burn-in chain), and the mean
## acceptance probability of the next (n_eps_calibration_blocks - 1) blocks, alpha_bar, gives c of the curve
## alpha(eps) = 2 Phi(-c eps^2) (as above) and the sampling eps
##
##         eps = sqrt(-qnorm(adapt_delta / 2) / c),      c = -qnorm(alpha_bar / 2) / eps_burnin^2      (capped).
##
## The sampling then starts again from the same initial states (the sampler copies them), so only its eps differs.
## One call, rather than chained blocks, has no state to hand over between calls and runs the chains independently
## for the whole calibration (one wait for the slowest chain, rather than one per block or per iteration). A call
## whose measured acceptance is above 0.99 or below 0.01 carries almost no information on c: eps is doubled or
## halved and the call repeated (at most 6 times).
##
## ---- the eps of the next call after an uninformative one: double it when every call so far accepted almost
##      every proposal, halve it when every call so far rejected almost every one, and, once both kinds of call
##      have been seen, the geometric mean of the largest eps accepted almost always and the smallest eps rejected
##      almost always (a bisection on the log scale), capped at max_eps:
##
fn_next_eps_of_an_uninformative_call <-  function(eps_accepted_almost_always,
                                                  eps_rejected_almost_always,
                                                  max_eps) {

        if (is.finite(eps_accepted_almost_always) && is.finite(eps_rejected_almost_always)) {
              return(min(max_eps, sqrt(eps_accepted_almost_always * eps_rejected_almost_always)))
        }
        if (is.finite(eps_accepted_almost_always)) return(min(max_eps, 2 * eps_accepted_almost_always))
        return(eps_rejected_almost_always / 2)

}
##
##
#' Calibrate the sampling eps from one sampling call so that the mean acceptance probability is adapt_delta
#'
#' @param fn_run_sampling_iterations_from_initial_states function(n_iter, eps_main, eps_us, seed) that runs
#'   n_iter iterations of every sampling chain from the initial states of the sampling, with these step sizes, and
#'   returns a list whose element sampling_diagnostics holds p_jump_main and p_jump_nuisance, each a matrix of
#'   n_iter rows and one column per chain.
#' @param eps_main,eps_us,calibrate_eps_us,adapt_delta,n_chains_sampling,max_eps_main,max_eps_us,seed As above.
#' @param n_transitions_per_eps_calibration_block Transitions (iterations x sampling chains) of each block of the
#'   call, at least; the first block is discarded.
#' @param n_eps_calibration_blocks Number of blocks in the call, the discarded first one included.
#' @param max_n_iter_of_eps_calibration_call Most iterations of each call: the blocks are cut to this many
#'   iterations in all, and the first ceiling(n_iter_of_call / n_eps_calibration_blocks) of the call's
#'   n_iter_of_call iterations are discarded.
#' @return As above; blocks has one row per call (block = the call's number, n_iter = all its iterations,
#'   n_iter_discarded, n_chains, eps_main, mean_acceptance_main (over the iterations after the discarded ones),
#'   eps_us, mean_acceptance_us, used_in_fit).
#' @noRd
fn_calibrate_eps_on_sampling_chains_from_one_call_to_adapt_delta <-  function(
                                                           fn_run_sampling_iterations_from_initial_states,
                                                           eps_main,
                                                           eps_us,
                                                           calibrate_eps_us,
                                                           adapt_delta,
                                                           n_chains_sampling,
                                                           n_transitions_per_eps_calibration_block,
                                                           n_eps_calibration_blocks,
                                                           max_n_iter_of_eps_calibration_call,
                                                           max_eps_main,
                                                           max_eps_us,
                                                           seed) {

        ##
        if (!is.numeric(adapt_delta) || length(adapt_delta) != 1 || !(adapt_delta > 0 && adapt_delta < 1)) {
              stop("eps calibration: adapt_delta must be a single number in (0, 1).")
        }
        if (!is.numeric(n_eps_calibration_blocks) || length(n_eps_calibration_blocks) != 1 ||
            n_eps_calibration_blocks < 2) {
              stop("n_eps_calibration_blocks must be a single number >= 2 (the first block is discarded).")
        }
        ##
        ## ---- At most max_n_iter_of_eps_calibration_call iterations in each call ----------------------------------------------------
        ##
        if (!is.numeric(max_n_iter_of_eps_calibration_call) || length(max_n_iter_of_eps_calibration_call) != 1 ||
            max_n_iter_of_eps_calibration_call < 2 || max_n_iter_of_eps_calibration_call %% 1 != 0) {

              stop("max_n_iter_of_eps_calibration_call must be a single whole number >= 2.")

        }
        ##
        ## (n_eps_calibration_blocks blocks of ceiling(n_transitions_per_eps_calibration_block /
        ##  n_chains_sampling) iterations, cut to max_n_iter_of_eps_calibration_call iterations in all; the first
        ##  block is discarded)
        n_iter_per_block <-  ceiling(n_transitions_per_eps_calibration_block / n_chains_sampling)
        n_iter_of_call <-  min(max_n_iter_of_eps_calibration_call,
                               n_eps_calibration_blocks * n_iter_per_block)
        n_iter_discarded <-  ceiling(n_iter_of_call / n_eps_calibration_blocks)
        rows_measured <-  (n_iter_discarded + 1):n_iter_of_call
        minus_qnorm_of_half_adapt_delta <-  -stats::qnorm(adapt_delta / 2)
        max_calls <-  7
        ##
        calls <-  list()
        eps_main_of_call <-  eps_main
        eps_us_of_call <-  eps_us
        ## (the largest eps found with almost every proposal accepted and the smallest with almost every one
        ##  rejected: once both are known, the next eps is their geometric mean, so that the calls close in on the
        ##  edge between them rather than doubling and halving back and forth across it)
        eps_main_accepted_almost_always <-  NA_real_
        eps_main_rejected_almost_always <-  NA_real_
        eps_us_accepted_almost_always <-  NA_real_
        eps_us_rejected_almost_always <-  NA_real_
        c_main <-  NA_real_
        c_us <-  NA_real_
        for (call in seq_len(max_calls)) {
              ##
              ## (a seed of its own for each call, never the sampling seed, within the range of an R integer)
              seed_of_call <-  ((seed + 1000003 * call) %% 2147483000) + 1
              result_of_call <-  fn_run_sampling_iterations_from_initial_states( n_iter   = n_iter_of_call,
                                                                                 eps_main = eps_main_of_call,
                                                                                 eps_us   = eps_us_of_call,
                                                                                 seed     = seed_of_call)
              p_jump_main <-  result_of_call$sampling_diagnostics$p_jump_main
              # mean_acceptance_main <-  mean(p_jump_main[rows_measured, , drop = FALSE], na.rm = TRUE)
              mean_acceptance_main <-  fn_mean_over_iterations_of_the_harmonic_mean_acceptance_over_chains(
                                           p_jump_main[rows_measured, , drop = FALSE])
              mean_acceptance_us <-  NA_real_
              if (isTRUE(calibrate_eps_us)) {
                    p_jump_nuisance <-  result_of_call$sampling_diagnostics$p_jump_nuisance
                    # mean_acceptance_us <-  mean(p_jump_nuisance[rows_measured, , drop = FALSE], na.rm = TRUE)
                    mean_acceptance_us <-  fn_mean_over_iterations_of_the_harmonic_mean_acceptance_over_chains(
                                                 p_jump_nuisance[rows_measured, , drop = FALSE])
              }
              rm(result_of_call)
              ##
              call_is_informative <-  is.finite(mean_acceptance_main) &&
                                      (mean_acceptance_main <= 0.99) && (mean_acceptance_main >= 0.01)
              if (isTRUE(calibrate_eps_us)) {
                    call_is_informative <-  call_is_informative && is.finite(mean_acceptance_us) &&
                                            (mean_acceptance_us <= 0.99) && (mean_acceptance_us >= 0.01)
              }
              calls[[call]] <-  data.frame( block = call,
                                            n_iter = n_iter_of_call,
                                            n_iter_discarded = n_iter_discarded,
                                            n_chains = n_chains_sampling,
                                            eps_main = eps_main_of_call,
                                            mean_acceptance_main = mean_acceptance_main,
                                            eps_us = eps_us_of_call,
                                            mean_acceptance_us = mean_acceptance_us,
                                            used_in_fit = call_is_informative,
                                            stringsAsFactors = FALSE)
              message(colourise(paste0("eps calibration on the ", n_chains_sampling, " sampling chains, ",
                                       "one call of ", n_iter_of_call, " iterations (first ", n_iter_discarded,
                                       " discarded): ",
                                       "eps_main = ", signif(eps_main_of_call, 4), ", mean acceptance = ",
                                       round(mean_acceptance_main, 3)),
                                "green"))
              ##
              if (!call_is_informative) {
                    ##
                    if (isTRUE(mean_acceptance_main > 0.99)) {
                          eps_main_accepted_almost_always <-  max(eps_main_accepted_almost_always,
                                                                  eps_main_of_call,
                                                                  na.rm = TRUE)
                    } else {
                          eps_main_rejected_almost_always <-  min(eps_main_rejected_almost_always,
                                                                  eps_main_of_call,
                                                                  na.rm = TRUE)
                    }
                    eps_main_of_call <-  fn_next_eps_of_an_uninformative_call(
                                             eps_accepted_almost_always = eps_main_accepted_almost_always,
                                             eps_rejected_almost_always = eps_main_rejected_almost_always,
                                             max_eps = max_eps_main)
                    ##
                    if (isTRUE(calibrate_eps_us)) {
                          if (isTRUE(mean_acceptance_us > 0.99)) {
                                eps_us_accepted_almost_always <-  max(eps_us_accepted_almost_always,
                                                                      eps_us_of_call,
                                                                      na.rm = TRUE)
                          } else {
                                eps_us_rejected_almost_always <-  min(eps_us_rejected_almost_always,
                                                                      eps_us_of_call,
                                                                      na.rm = TRUE)
                          }
                          eps_us_of_call <-  fn_next_eps_of_an_uninformative_call(
                                                 eps_accepted_almost_always = eps_us_accepted_almost_always,
                                                 eps_rejected_almost_always = eps_us_rejected_almost_always,
                                                 max_eps = max_eps_us)
                    }
                    next
              }
              ##
              c_main <-  -stats::qnorm(mean_acceptance_main / 2) / eps_main_of_call^2
              eps_main_of_call <-  min(max_eps_main, sqrt(minus_qnorm_of_half_adapt_delta / c_main))
              if (isTRUE(calibrate_eps_us)) {
                    c_us <-  -stats::qnorm(mean_acceptance_us / 2) / eps_us_of_call^2
                    eps_us_of_call <-  min(max_eps_us, sqrt(minus_qnorm_of_half_adapt_delta / c_us))
              }
              break
        ##
        }
        if (is.na(c_main)) {
              stop(paste0("eps calibration on the sampling chains: no call had a mean acceptance probability in ",
                          "[0.01, 0.99] after ", max_calls, " calls (last eps_main = ",
                          signif(eps_main_of_call, 4), ")."))
        }
        message(colourise(paste0("eps calibration: sampling eps_main = ", signif(eps_main_of_call, 4),
                                 " (burn-in: ", signif(eps_main, 4), "), for a mean acceptance of ", adapt_delta),
                          "blue"))
        ##
        return(list( eps_main = eps_main_of_call,
                     eps_us = eps_us_of_call,
                     c_main = c_main,
                     c_us = c_us,
                     blocks = do.call(rbind, calls)))
##
}






















