fn_tau_jitter_radical_inverse <-  function(index, base) {

    if (!is.numeric(index) || any(!is.finite(index)) || any(index < 1) || any(index != round(index)) ||
        length(base) != 1 || !base %in% c(2, 3)) {
        stop("Halton indices must be positive finite whole numbers and the base must be 2 or 3.")
    }

    remaining <-  index
    value <-  rep(0, length(index))
    digit_scale <-  1 / base

    while (any(remaining > 0)) {
        value <-  value + (remaining %% base) * digit_scale
        remaining <-  floor(remaining / base)
        digit_scale <-  digit_scale / base
    }

    return(value)

}


fn_tau_jitter_burnin_init <-  function(tau_jitter_burnin,
                                       randomize_tau_burnin,
                                       seed,
                                       n_chains,
                                       share_tau_ii_across_chains,
                                       partitioned_HMC) {

    if (!identical(tau_jitter_burnin, "halton") || !isTRUE(randomize_tau_burnin)) return(NULL)
    if (!is.numeric(seed) || length(seed) != 1 || !is.finite(seed) || seed != round(seed) ||
        abs(seed) > .Machine$integer.max) {
        stop("The Halton burn-in seed must be a finite whole number within the R seed range.")
    }
    if (!is.numeric(n_chains) || length(n_chains) != 1 || !is.finite(n_chains) || n_chains < 1 ||
        n_chains != round(n_chains)) {
        stop("Halton burn-in needs a positive whole number of chains.")
    }
    for (flag in list(share_tau_ii_across_chains, partitioned_HMC)) {
        if (!is.logical(flag) || length(flag) != 1 || is.na(flag)) {
            stop("Halton burn-in sharing and partitioning flags must be TRUE or FALSE.")
        }
    }

    ## Each chain has a random shift; shared trajectories use one shift for the whole chain group.
    seed_existed <-  exists(".Random.seed", envir = globalenv(), inherits = FALSE)
    if (seed_existed) previous_seed <-  get(".Random.seed", envir = globalenv(), inherits = FALSE)
    on.exit({
        if (seed_existed) {
            assign(".Random.seed", previous_seed, envir = globalenv())
        } else if (exists(".Random.seed", envir = globalenv(), inherits = FALSE)) {
            rm(list = ".Random.seed", envir = globalenv())
        }
    }, add = TRUE)
    set.seed(seed)
    n_shifts <-  if (share_tau_ii_across_chains) 1 else n_chains
    main_shift <-  stats::runif(n_shifts)
    nuisance_shift <-  if (partitioned_HMC) stats::runif(n_shifts) else main_shift

    return(list(main_shift = main_shift,
                nuisance_shift = nuisance_shift,
                n_chains = n_chains,
                partitioned_HMC = partitioned_HMC,
                share_tau_ii_across_chains = share_tau_ii_across_chains))

}


fn_tau_jitter_burnin_draw <-  function(state, iteration, EHMC_args_as_Rcpp_List) {

    if (is.null(state)) return(NULL)
    if (!isTRUE(EHMC_args_as_Rcpp_List$randomize_tau)) return(NULL)
    if (!is.numeric(iteration) || length(iteration) != 1 || !is.finite(iteration) ||
        iteration < 1 || iteration != round(iteration)) {
        stop("The Halton burn-in iteration must be a positive finite whole number.")
    }
    argument_names <-  c("tau_main", "eps_main")
    if (state$partitioned_HMC) argument_names <-  c(argument_names, "tau_us", "eps_us")
    for (name in argument_names) {
        value <-  EHMC_args_as_Rcpp_List[[name]]
        if (!is.numeric(value) || length(value) != 1 || !is.finite(value) || value <= 0) {
            stop(paste0("Halton burn-in needs one positive finite ", name, "."))
        }
    }

    main_fraction <-  (fn_tau_jitter_radical_inverse(iteration, 2) + state$main_shift) %% 1
    tau_main_ii <-  rep(2 * max(EHMC_args_as_Rcpp_List$tau_main, EHMC_args_as_Rcpp_List$eps_main) * main_fraction,
                         length.out = state$n_chains)
    tau_us_ii <-  tau_main_ii
    if (state$partitioned_HMC) {
        nuisance_fraction <-  (fn_tau_jitter_radical_inverse(iteration, 3) + state$nuisance_shift) %% 1
        tau_us_ii <-  rep(2 * max(EHMC_args_as_Rcpp_List$tau_us, EHMC_args_as_Rcpp_List$eps_us) * nuisance_fraction,
                           length.out = state$n_chains)
    }

    return(list(tau_main_ii = tau_main_ii, tau_us_ii = tau_us_ii))

}


fn_tau_jitter_burnin_worker_api <-  function(worker_api_env) {

    api_name <-  "fn_persistent_burnin_run_one_iter_tau_jitter"
    if (!exists(api_name, envir = worker_api_env, mode = "function", inherits = FALSE)) {
        stop("tau_jitter_burnin = 'halton' needs the native burn-in jitter entry point in the package that owns the worker; rebuild that native module.")
    }

    return(get(api_name, envir = worker_api_env, inherits = FALSE))

}


fn_tau_jitter_burnin_reset <-  function(EHMC_args_as_Rcpp_List) {

    EHMC_args_as_Rcpp_List$use_given_tau_main_ii <-  FALSE
    EHMC_args_as_Rcpp_List$share_tau_ii_across_chains <-  FALSE
    EHMC_args_as_Rcpp_List$tau_main_ii <-  EHMC_args_as_Rcpp_List$tau_main
    EHMC_args_as_Rcpp_List$tau_us_ii <-  EHMC_args_as_Rcpp_List$tau_us

    return(EHMC_args_as_Rcpp_List)

}
























