#### =============================================================================================================
## R_fn_print_the_layout_and_settings_of_a_timed_call_or_fit.R
##
## ---- One block before every timed call or fit: the layout and every setting of the call, each name in its own
##      colour through colourise(). Plain text when the terminal has no colour; saved data is never coloured.
##
#' Print the layout and the settings of a timed call or a fit
#'
#' Prints one block, each item as "name = value", before a timed call or a fit. The names are coloured through
#' colourise(). The five fixed meanings are purple for iteration counters, blue for the step size, cyan for the
#' progress phase, green for success and red for divergences. Plain text when the terminal has no colour.
#'
#' @param model The model, e.g. "PK".
#' @param version The version code, one of the eight version codes, e.g. "PK_AVX_fused_RSC".
#' @param phase "burnin" or "sampling".
#' @param machine The machine, e.g. "local_HPC" or "laptop".
#' @param CPUs The CPUs used, e.g. "0-179".
#' @param math_backend The math backend, e.g. "Stan", "AVX512" or "AVX2".
#' @param N The N of the call, in the units the model counts.
#' @param what_N_counts What N counts for this model, e.g. "patients".
#' @param n_chains The number of chains of this phase.
#' @param threads_per_chain The threads per chain.
#' @param n_chunks The number of chunks.
#' @param L The leapfrog steps per iteration.
#' @param N_iter The iterations of the timed call or fit.
#' @param seconds_per_timed_call The target seconds of the timed call; NA_real_ for a fit.
#' @param main_metric The main metric, e.g. "diagonal" or "dense".
#' @param nuisance_metric The nuisance metric, e.g. "diagonal" or "identity".
#' @param eps The step size.
#' @param seed The seed.
#' @param nuisance_block The name of the nuisance block, or "none".
#' @return NULL, invisibly.
#' @export
fn_print_the_layout_and_settings_of_a_timed_call_or_fit <-  function(
        model,
        version,
        phase,
        machine,
        CPUs,
        math_backend,
        N,
        what_N_counts,
        n_chains,
        threads_per_chain,
        n_chunks,
        L,
        N_iter,
        seconds_per_timed_call,
        main_metric,
        nuisance_metric,
        eps,
        seed,
        nuisance_block ) {

        if (!(phase %in% c("burnin", "sampling"))) {
              stop(paste0("phase must be burnin or sampling, not ", phase))
        }
        n_chains_name <-  if (phase == "burnin") "n_chains_burnin" else "n_chains_sampling"
        display_phase <-  if (phase == "burnin") "burn-in" else "sampling"
        n_threads_total <-  n_chains * threads_per_chain
        chunk_size <-  ceiling(N / n_chunks)
        names_of_math_backends <-  c(Stan = "Stan::math", AVX512 = "AVX-512", AVX2 = "AVX2")
        display_math_backend <-  names_of_math_backends[[math_backend]]
        if (is.null(display_math_backend)) display_math_backend <-  math_backend
        seconds_text <-  if (is.na(seconds_per_timed_call)) "not a timed call" else {
              format(seconds_per_timed_call, trim = TRUE)
        }
        message(colourise("---- the layout and settings of this timed call or fit ----", fg = "cyan"))
        message(colourise("model", fg = "light blue"), " = ", model)
        message(colourise("version", fg = "violet"), " = ", version)
        message(colourise("phase", fg = "cyan"), " = ", display_phase)
        message(colourise("machine", fg = "white"), " = ", machine)
        message(colourise("CPUs", fg = "light gray"), " = ", CPUs)
        message(colourise("math_backend", fg = "light green"), " = ", display_math_backend)
        message(colourise("N", fg = "yellow"), " = ", format(N, big.mark = ",", trim = TRUE), " (",
                what_N_counts, ")")
        message(colourise("threads_per_chain", fg = "light purple"), " = ", threads_per_chain)
        message(colourise(n_chains_name, fg = "dark gray"), " = ", n_chains)
        message(colourise("n_threads_total", fg = "brown"), " = ", n_threads_total)
        message(colourise("n_chunks", fg = "light red"), " = ", n_chunks)
        message(colourise("chunk_size", fg = "orange"), " = ", chunk_size)
        message(colourise("L", fg = "light cyan"), " = ", L)
        message(colourise("N_iter", fg = "purple"), " = ", N_iter)
        message(colourise("seconds_per_timed_call", fg = "sky blue"), " = ", seconds_text)
        message(colourise("metric", fg = "lime"), " = main: ", main_metric, "; nuisance: ", nuisance_metric)
        message(colourise("eps", fg = "blue"), " = ", format(eps, trim = TRUE))
        message(colourise("seed", fg = "pink"), " = ", seed)
        message(colourise("nuisance_block", fg = "teal"), " = ", nuisance_block)
        invisible(NULL)
}






















