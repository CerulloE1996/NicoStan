

## ---- The parameter families that Se, Sp and prevalence depend on, for the built-in latent-class models --------
##
## interest_only (R_fn_sample_model): every tau criterion computes its statistic over the main-block rows of a set
## of parameter families only. For the built-in latent-class models of diagnostic accuracy (BayesMVP's LC_MVP and
## LC_MVOP) the reported quantities - Se, Sp and prevalence - are the same for every user, so the package supplies
## the set here. Every other model (user-supplied Stan models, and the built-in MVP / MVOP / latent_trait models,
## which report no Se, Sp or prevalence) has no built-in set: NULL is returned, and interest_only needs a
## user-supplied character vector of parameter families there.
##
## Derivation (the reported quantities and the main parameter families they are computed from):
##   LC_MVP  (binary tests):   the prevalence from p_raw; Se_j and Sp_j (and Fp_j = 1 - Sp_j) of test j from the
##                             latent-class means of test j in beta; hence {beta, p_raw};
##   LC_MVOP (ordinal tests):  the prevalence from p_raw; Se and Sp of test j at each threshold from the
##                             latent-class means in beta AND the cutpoints of test j in C_unc_vec; hence
##                             {beta, p_raw, C_unc_vec}.
##   Neither depends on Omega_unconstrained_vec (the within-class correlations), the only other main family: the
##   marginal of one test of a multivariate probit has unit latent variance whatever the correlations.
##
## ---- Renamed 6 Oct 2026 (no alias); the previous name:
## parameters_that_Se_Sp_and_prevalence_depend_on_for_built_in_latent_class_models <-  function(Model_type) {
parms_that_Se_Sp_and_prev_depend_on_for_built_in_LCMs <-  function(Model_type) {

        if (identical(Model_type, "LC_MVP"))   return(c("beta", "p_raw"))
        if (identical(Model_type, "LC_MVOP"))  return(c("beta", "p_raw", "C_unc_vec"))
        return(NULL)

}
























