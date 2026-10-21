




#### ---- the superchains of a number of sampling chains (the rule of $sample()) ---------------------------------

#' The superchains of a number of sampling chains, by the rule of $sample() when n_superchains is not given
#'
#' The same rule as $sample() applies to the n_chains_sampling given or defaulted: n_chains_sampling superchains
#' below 16 chains, half of 17 or 31, a quarter of 32 or 63, an eighth of 64 or 127, a sixteenth above 127, and
#' "ChESSR" otherwise. Used when Paper 1's check of the sampling layout changes the sampling chains.
#'
#' @param n_chains_sampling The sampling chains.
#' @return The superchains (a number), or "ChESSR".
#' @keywords internal
fn_n_superchains_of_n_chains_sampling <-  function(n_chains_sampling) {

        n_superchains <-  NULL
        if (n_chains_sampling < 16) {
              n_superchains <-  n_chains_sampling
        } else if (n_chains_sampling %in% c(17, 31)) {
              n_superchains <-  round(n_chains_sampling / 2)
        } else if (n_chains_sampling %in% c(32, 63)) {
              n_superchains <-  round(n_chains_sampling / 4)
        } else if (n_chains_sampling %in% c(64, 127)) {
              n_superchains <-  round(n_chains_sampling / 8)
        } else if (n_chains_sampling > 127) {
              n_superchains <-  round(n_chains_sampling / 16)
        }
        if (is.null(n_superchains)) n_superchains <-  "ChESSR"
        return(n_superchains)

}





















