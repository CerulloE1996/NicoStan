## Synthetic four-class LC-MVOP example for NicoStan.
##
## This runtime example uses text fixtures, the bundled Stan sources, and the
## portable AVX headers. It never reads or copies clinical data.

four_class_example_paths <- function(
    analysis_directory = file.path(getwd(), "Alg_paper_analysis"),
    example_directory = NULL) {
    if (!is.null(example_directory)) {
        example_directory <- normalizePath(example_directory, mustWork = FALSE)
        model_directory <- file.path(example_directory, "models")
        fixture_directory <- file.path(example_directory, "fixture")
        return(list(
            example_directory = example_directory,
            paper3_directory = dirname(dirname(example_directory)),
            validation_directory = dirname(example_directory),
            fixture_directory = fixture_directory,
            runtime_directory = model_directory,
            source_plain = file.path(model_directory, "four_class_plain.stan"),
            source_avx = file.path(model_directory, "four_class_avx.stan"),
            source_fused = file.path(model_directory, "four_class_avx_fused.stan"),
            fixture_file = file.path(fixture_directory, "fixture.R"),
            fixture_json = file.path(fixture_directory, "data_current.json"),
            init_directory = file.path(fixture_directory, "initial_values")
        ))
    }
    paper3_directory <- normalizePath(file.path(analysis_directory, "paper_3_general_sampler"),
                                      mustWork = FALSE)
    validation_directory <- file.path(paper3_directory, "validation")
    fixture_directory <- file.path(validation_directory, "four_class_fixture")
    runtime_directory <- file.path(validation_directory, "four_class_runtime")
    list(
        example_directory = file.path(paper3_directory, "four_class"),
        paper3_directory = paper3_directory,
        validation_directory = validation_directory,
        fixture_directory = fixture_directory,
        runtime_directory = runtime_directory,
        source_plain = file.path(runtime_directory, "four_class_plain.stan"),
        source_avx = file.path(runtime_directory, "four_class_avx.stan"),
        source_fused = file.path(runtime_directory, "four_class_avx_fused.stan"),
        fixture_file = file.path(fixture_directory, "fixture.R"),
        fixture_json = file.path(fixture_directory, "data_current.json"),
        init_directory = file.path(fixture_directory, "initial_values_current")
    )
}

four_class_add_current_data_fields <- function(stan_data) {
    stopifnot(is.list(stan_data), identical(as.integer(stan_data$n_class), 4L))
    ## Current four-class source requires these fields even when the optional OR
    ## parameterisation and generated-quantity diagnostics are disabled.
    defaults <- list(
        ref_cutoff_category_A = 2L,
        ref_cutoff_category_B = 2L,
        use_OR_prior = 0L,
        log_OR_shared = 1L,
        prior_mu_logit_A_mean = -2,
        prior_mu_logit_A_sd = 1,
        prior_mu_logit_B_mean = -2,
        prior_mu_logit_B_sd = 1,
        prior_sigma_logit_sd = 0.5,
        prior_log_OR_mean = 0,
        prior_log_OR_sd = 1,
        use_cond_prior = 0L,
        prior_logit_Q_mean = -1,
        prior_logit_Q_sd = 1
    )
    for (nm in names(defaults)) if (is.null(stan_data[[nm]])) stan_data[[nm]] <- defaults[[nm]]
    ## These are declared as vectors in Stan.  Explicit dim attributes are
    ## required for the one-element cases because a bare R vector can otherwise
    ## be unboxed by the JSON writer.
    stan_data$pop_weight <- array(as.numeric(stan_data$pop_weight),
                                  dim = length(stan_data$pop_weight))
    stan_data$standardisation_index <- array(as.integer(stan_data$standardisation_index),
                                             dim = length(stan_data$standardisation_index))
    stan_data$standardisation_weight <- array(as.numeric(stan_data$standardisation_weight),
                                              dim = length(stan_data$standardisation_weight))
    stan_data$grid_standardisation_position <- array(as.integer(stan_data$grid_standardisation_position),
                                                     dim = length(stan_data$grid_standardisation_position))
    stan_data$grid_standardisation_weight <- array(as.numeric(stan_data$grid_standardisation_weight),
                                                   dim = length(stan_data$grid_standardisation_weight))
    stopifnot(length(stan_data$pop_weight) == stan_data$n_pops,
              length(stan_data$standardisation_index) == stan_data$n_standardisation,
              length(stan_data$standardisation_weight) == stan_data$n_standardisation,
              length(stan_data$grid_standardisation_position) == stan_data$n_grid_standardisation,
              length(stan_data$grid_standardisation_weight) == stan_data$n_grid_standardisation)
    stan_data
}

four_class_write_json <- function(stan_data, json_file) {
    if (!requireNamespace("cmdstanr", quietly = TRUE)) stop("cmdstanr is required")
    dir.create(dirname(json_file), recursive = TRUE, showWarnings = FALSE)
    cmdstanr::write_stan_json(data = stan_data, file = json_file)
    invisible(json_file)
}

four_class_read_fixture <- function(fixture_file) {
    if (!file.exists(fixture_file)) stop("Synthetic fixture not found: ", fixture_file)
    if (identical(tolower(tools::file_ext(fixture_file)), "r"))
        return(dget(fixture_file))
    stop("Portable fixtures must be supplied as fixture.R text files")
}
four_class_model_data_string <- function(data_json) {
    paste(readLines(data_json, warn = FALSE), collapse = "")
}

four_class_external_model_cache <- function(stan_file, paths, output_directory = NULL,
                                             build_signature = NULL) {
    if (!requireNamespace("digest", quietly = TRUE)) stop("digest is required")
    example_directory <- normalizePath(paths$example_directory, mustWork = FALSE)
    if (is.null(output_directory) ||
        normalizePath(output_directory, mustWork = FALSE) == example_directory ||
        startsWith(normalizePath(output_directory, mustWork = FALSE),
                   paste0(example_directory, .Platform$file.sep))) {
        output_directory <- file.path(tempdir(), "NicoStan_four_class_results")
    }
    dir.create(output_directory, recursive = TRUE, showWarnings = FALSE)
    source_hash <- digest::digest(file = normalizePath(stan_file), algo = "sha256")
    signature <- if (is.null(build_signature)) "plain" else substr(build_signature, 1L, 12L)
    cached_file <- file.path(output_directory,
                             paste0(tools::file_path_sans_ext(basename(stan_file)),
                                    "_", substr(source_hash, 1L, 12L), "_", signature, ".stan"))
    if (!file.exists(cached_file) ||
        !identical(digest::digest(file = cached_file, algo = "sha256"), source_hash)) {
        stopifnot(file.copy(normalizePath(stan_file), cached_file, overwrite = TRUE))
    }
    list(stan_file = normalizePath(cached_file), output_directory = normalizePath(output_directory),
         source_sha256 = source_hash)
}

four_class_external_output_file <- function(output_rds, cache) {
    if (is.null(output_rds) ||
        startsWith(normalizePath(output_rds, mustWork = FALSE),
                   paste0(normalizePath(cache$output_directory), .Platform$file.sep)))
        return(output_rds)
    file.path(cache$output_directory, basename(output_rds))
}

four_class_assert_simd_lanes <- function(fit, expected_lanes = 8L) {
    bs_model <- fit$init_object$bs_model
    gq_names <- bs_model$param_names(include_tp = TRUE, include_gq = TRUE)
    if (!"simd_lanes" %in% gq_names)
        stop("The AVX model has no generated-quantity SIMD lane probe")
    constrained <- bs_model$param_constrain(rep(0, bs_model$param_unc_num()),
                                             include_tp = TRUE, include_gq = TRUE,
                                             rng = bs_model$new_rng(seed = 2026L))
    lane_index <- match("simd_lanes", gq_names)
    lane <- unname(constrained[lane_index])
    if (!is.finite(lane) || abs(lane - expected_lanes) > 1e-8)
        stop("AVX SIMD lane probe returned ", lane, "; expected ", expected_lanes)
    as.numeric(lane)
}

check_four_class_bridge_parity <- function(
    paths = four_class_example_paths(),
    data_json = paths$fixture_json,
    avx_header = NULL) {
    if (!requireNamespace("bridgestan", quietly = TRUE)) stop("bridgestan is required")
    if (!file.exists(paths$source_plain) || !file.exists(paths$source_avx)) stop("Plain/AVX source missing")
    data_string <- four_class_model_data_string(data_json)
    load_model <- function(stan_file, data_string, stanc_args = NULL, make_args = NULL) {
        so_file <- paste0(tools::file_path_sans_ext(normalizePath(stan_file)), "_model.so")
        if (!file.exists(so_file)) {
            so_file <- bridgestan::compile_model(normalizePath(stan_file),
                                                 stanc_args = stanc_args,
                                                 make_args = make_args)
        }
        bridgestan::StanModel$new(lib = normalizePath(so_file), data = data_string, seed = 123L)
    }
    if (is.null(avx_header)) stop("Supply the BayesMVP AVX header for the AVX comparison")
    if (!file.exists(avx_header)) stop("AVX header not found: ", avx_header)
    compare_one_data <- function(data_string, label) {
        plain <- load_model(paths$source_plain, data_string)
        avx <- load_model(paths$source_avx, data_string, stanc_args = "--allow-undefined",
                          make_args = c("STAN_THREADS=true",
                                        paste0("USER_HEADER=", normalizePath(avx_header))))
        set.seed(2026L)
        points <- c(list(rep(0, plain$param_unc_num())),
                    lapply(seq_len(3L), function(ii) rnorm(plain$param_unc_num(), sd = 0.2)))
        errors <- lapply(points, function(theta) {
            p <- plain$log_density_gradient(theta, propto = FALSE, jacobian = TRUE)
            a <- avx$log_density_gradient(theta, propto = FALSE, jacobian = TRUE)
            finite <- is.finite(p$val) && is.finite(a$val) && all(is.finite(p$gradient)) &&
                      all(is.finite(a$gradient))
            list(log_density_absolute_error = abs(p$val - a$val),
                 maximum_scaled_gradient_error = if (finite) max(abs(p$gradient - a$gradient) /
                                                                  pmax(1, abs(p$gradient))) else NA_real_,
                 finite = finite)
        })
        result <- list(
            label = label,
            unconstrained_dimension = plain$param_unc_num(),
            maximum_log_density_absolute_error = max(vapply(errors, `[[`, numeric(1),
                                                          "log_density_absolute_error")),
            maximum_scaled_gradient_error = max(vapply(errors, `[[`, numeric(1),
                                                     "maximum_scaled_gradient_error")),
            finite = all(vapply(errors, `[[`, logical(1), "finite"))
        )
        stopifnot(result$finite, result$maximum_log_density_absolute_error < 1e-5,
                  result$maximum_scaled_gradient_error < 1e-5)
        result
    }
    results <- list(default_normal = compare_one_data(data_string, "Phi_type=1"))
    ## The approximation branch is the one where AVX and Stan are expected to
    ## differ slightly.  It exercises the custom fast_Phi_approx/log kernels.
    approximation_data <- sub('"Phi_type"[[:space:]]*:[[:space:]]*1',
                              '"Phi_type": 2', data_string, perl = TRUE)
    results$approximation <- compare_one_data(approximation_data, "Phi_type=2")
    results
}

run_four_class_nicostan_smoke <- function(
    paths = four_class_example_paths(),
    fixture_file = paths$fixture_file,
    stan_file = paths$source_plain,
    math_backend = c("Stan", "AVX512", "AVX2"),
    avx_header = file.path(dirname(dirname(paths$paper3_directory)), "R_packages", "BayesMVP",
                           "inst", "BayesMVP", "inst", "include", "BayesMVP",
                           "stan_external_functions.hpp"),
    chains = 2L, warmup = 60L, iterations = 80L, adapt_delta = 0.9, seed = 2026L,
    metric_type_main = "Empirical", metric_shape_main = "diag",
    M_decay_type = "inverse", phi_type = NULL,
    output_rds = file.path(paths$runtime_directory, "nicostan_plain_smoke.rds")) {
    math_backend <- match.arg(math_backend)
    if (!requireNamespace("NicoStan", quietly = TRUE)) stop("NicoStan is required")
    fixture <- four_class_read_fixture(fixture_file)
    fixture$data <- four_class_add_current_data_fields(fixture$data)
    if (!is.null(phi_type)) fixture$data$Phi_type <- as.integer(phi_type)
    if (!file.exists(stan_file)) stop("Stan source missing: ", stan_file)
    if (chains > length(fixture$initial_values))
        stop("The fixture contains only ", length(fixture$initial_values), " initial-value lists")
    ## math_backend = "AVX2" compiles the same AVX model with the same flags as "AVX512" plus
    ## -DBAYESMVP_FORCE_AVX2 (make argument CPPFLAGS_OPTIM), which makes the BayesMVP header use its 4-lane AVX2 kernels on an
    ## AVX-512 machine too. (-mno-avx512f in CXXFLAGS would not work here: BridgeStan's make/local APPENDS -mavx512f to the
    ## CXXFLAGS environment variable.) The define is part of the build signature, so the AVX2 and AVX512 builds never share
    ## a cached binary, and four_class_assert_simd_lanes() stops unless the model reports 4 (AVX2) / 8 (AVX512) lanes.
    uses_avx_backend <- math_backend %in% c("AVX512", "AVX2")
    if (uses_avx_backend && !file.exists(avx_header))
        stop("AVX header not found: ", avx_header)
    avx_flags <- if (uses_avx_backend)
        "-O3 -std=c++17 -march=native -mtune=native -mfma -mavx2 -mavx512f -mavx512vl -mavx512dq" else
        "-O3 -std=c++17 -march=native -mtune=native -mfma"
    avx_level_make_args <- if (identical(math_backend, "AVX2")) "CPPFLAGS_OPTIM=-DBAYESMVP_FORCE_AVX2" else character(0)
    expected_simd_lanes <- if (identical(math_backend, "AVX512")) 8 else if (identical(math_backend, "AVX2")) 4 else NA_real_
    build_signature <- if (uses_avx_backend)
        digest::digest(c(list(header = digest::digest(file = avx_header, algo = "sha256"),
                              flags = avx_flags),
                         if (length(avx_level_make_args) > 0) list(level = avx_level_make_args)), algo = "sha256") else NULL
    cache <- four_class_external_model_cache(stan_file, paths, paths$runtime_directory,
                                             build_signature = build_signature)
    output_rds <- four_class_external_output_file(output_rds, cache)
    stan_file <- cache$stan_file
    old_cxxflags <- Sys.getenv("CXXFLAGS", unset = NA_character_)
    if (uses_avx_backend) Sys.setenv(CXXFLAGS = avx_flags)
    on.exit({
        if (is.na(old_cxxflags)) Sys.unsetenv("CXXFLAGS") else Sys.setenv(CXXFLAGS = old_cxxflags)
    }, add = TRUE)
    fit <- NicoStan::MVP_model$new(
        Model_type = "Stan",
        Stan_data_list = fixture$data,
        Stan_model_file_path = normalizePath(stan_file),
        sample_nuisance = TRUE,
        Stan_cpp_user_header = if (uses_avx_backend) normalizePath(avx_header) else NULL,
        make_args = if (uses_avx_backend)
            c("STAN_THREADS=true", paste0("USER_HEADER=", normalizePath(avx_header)), avx_level_make_args) else NULL
    )
    simd_lanes <- if (uses_avx_backend) four_class_assert_simd_lanes(fit, expected_lanes = expected_simd_lanes) else NA_real_
    ## Deliberately no n_nuisance_override: the first u_raw declaration is
    ## recovered from stanc metadata, and its unconstrained dimension is used.
    set.seed(seed)
    fit_started <- proc.time()[["elapsed"]]
    fit$sample(
        n_burnin = warmup, n_iter = iterations,
        n_chains_burnin = chains, n_chains_sampling = chains,
        n_superchains = chains, init_lists_per_chain = fixture$initial_values[seq_len(chains)],
        burnin_algorithm = "CHESSR", diffusion_HMC = FALSE,
        partitioned_HMC = FALSE, metric_type_main = metric_type_main,
        metric_shape_main = metric_shape_main, metric_type_nuisance = "uniform_diag",
        M_decay_type = M_decay_type,
        n_threads_WCP_burnin = 1L, n_threads_WCP_sampling = 1L,
        adapt_delta = adapt_delta, n_refresh = 20L, seed = seed, stream = seed,
        reorder_cols_MVP = FALSE
    )
    fit_wall_seconds <- proc.time()[["elapsed"]] - fit_started
    fit$summary(save_log_lik_trace = FALSE, compute_nested_rhat = FALSE,
                compute_generated_quantities = FALSE)
    efficiency <- fit$model_fit_object$summaries$efficiency_info
    main_summary <- fit$model_fit_object$summaries$summary_tibbles$summary_tibble_main_params
    posterior_summary <- posterior::summarise_draws(
        posterior::as_draws_array(fit$model_fit_object$traces$traces_as_arrays$trace_params_main),
        "mean", "sd", "ess_bulk", "ess_tail", "rhat")
    min_bulk_ess <- min(posterior_summary$ess_bulk, na.rm = TRUE)
    max_bulk_rhat <- max(posterior_summary$rhat, na.rm = TRUE)
    gradient_work <- efficiency$L_main_during_sampling * iterations * chains
    main_bases <- c("Omega_unconstrained_vec", "beta_own_absent_free",
                    "beta_own_present_free", "delta_cross_free",
                    "beta_cov_raw", "prev_simplex", "C_raw_vec",
                    "d_raw_present")
    result <- list(
        fit = fit,
        math_backend = math_backend,
        n_nuisance = fit$n_nuisance,
        n_params_main = fit$n_params_main,
        n_params = fit$init_object$n_params,
        n_params_main_unconstrained = fit$n_params_main,
        main_parameter_bases = main_bases,
        main_parameter_count_constrained = nrow(main_summary),
        automatic_nuisance_detection = isTRUE(fit$n_nuisance > 0),
        randomize_tau_burnin = fit$result$randomize_tau_burnin,
        randomize_tau_sampling = fit$result$randomize_tau_sampling,
        divergences = fit$model_fit_object$summaries$divergences$n_divs,
        min_ess_main = min_bulk_ess,
        max_rhat_main = max_bulk_rhat,
        sampling_seconds = efficiency$time_sampling,
        fit_wall_seconds = fit_wall_seconds,
        total_seconds = fit_wall_seconds,
        min_ess_per_sec_sampling = min_bulk_ess / efficiency$time_sampling,
        min_ess_per_sec_total = min_bulk_ess / fit_wall_seconds,
        gradient_work = gradient_work,
        min_ess_per_1000_gradient_work = 1000 * min_bulk_ess / gradient_work,
        gradient_work_basis = "NicoStan L_main_during_sampling times iterations times chains; gradient-work proxy",
        metric_type_main = metric_type_main,
        M_decay_type = M_decay_type,
        phi_type = fixture$data$Phi_type,
        avx_kernel_path_active = uses_avx_backend && fixture$data$Phi_type != 1L,
        avx_kernel_path = if (uses_avx_backend && fixture$data$Phi_type != 1L)
            "BayesMVP vectorized external branch" else if (uses_avx_backend)
            "Stan exact-CDF branch; external vector functions bypassed" else "Stan built-in math",
        simd_lanes = simd_lanes,
        avx_compile_flags = if (uses_avx_backend) paste(c(avx_flags, avx_level_make_args), collapse = " ") else NA_character_,
        avx_build_signature = if (uses_avx_backend) build_signature else NA_character_
    )
    stopifnot(result$automatic_nuisance_detection,
              is.null(fit$n_nuisance_override),
              identical(result$randomize_tau_burnin, FALSE),
              identical(result$randomize_tau_sampling, TRUE))
    saveRDS(result, output_rds)
    result
}

run_four_class_cmdstanr_smoke <- function(
    paths = four_class_example_paths(),
    fixture_file = paths$fixture_file,
    stan_file = paths$source_plain,
    chains = 2L, warmup = 60L, iterations = 80L, adapt_delta = 0.9, phi_type = NULL, seed = 2026L,
    output_rds = file.path(paths$runtime_directory, "cmdstanr_plain_smoke.rds")) {
    if (!requireNamespace("cmdstanr", quietly = TRUE)) stop("cmdstanr is required")
    fixture <- four_class_read_fixture(fixture_file)
    fixture$data <- four_class_add_current_data_fields(fixture$data)
    if (!is.null(phi_type)) fixture$data$Phi_type <- as.integer(phi_type)
    if (chains > length(fixture$initial_values))
        stop("The fixture contains only ", length(fixture$initial_values), " initial-value lists")
    cache <- four_class_external_model_cache(stan_file, paths, paths$runtime_directory)
    output_rds <- four_class_external_output_file(output_rds, cache)
    stan_file <- cache$stan_file
    model <- cmdstanr::cmdstan_model(normalizePath(stan_file),
                                     cpp_options = list(stan_threads = TRUE), quiet = TRUE)
    fit_started <- proc.time()[["elapsed"]]
    fit <- model$sample(
        data = fixture$data,
        init = fixture$initial_values[seq_len(chains)],
        chains = chains, parallel_chains = chains, threads_per_chain = 1L,
        iter_warmup = warmup, iter_sampling = iterations,
        adapt_delta = adapt_delta, seed = seed, refresh = 20L,
        output_dir = cache$output_directory
    )
    fit_wall_seconds <- proc.time()[["elapsed"]] - fit_started
    main_parameters <- c("Omega_unconstrained_vec", "beta_own_absent_free",
                         "beta_own_present_free", "delta_cross_free",
                         "beta_cov_raw", "prev_simplex", "C_raw_vec",
                         "d_raw_present")
    summary <- fit$summary(variables = main_parameters)
    diagnostics <- as.array(fit$sampler_diagnostics())
    gradient_work <- sum(diagnostics[, , "n_leapfrog__"])
    sampling_seconds <- max(fit$time()$chains$sampling)
    total_seconds <- fit$time()$total
    posterior_summary <- posterior::summarise_draws(
        posterior::as_draws_array(fit$draws(variables = main_parameters)),
        "mean", "sd", "ess_bulk", "ess_tail", "rhat")
    min_ess <- min(posterior_summary$ess_bulk, na.rm = TRUE)
    max_rhat <- max(posterior_summary$rhat, na.rm = TRUE)
    result <- list(
        fit = fit,
        math_backend = "Stan",
        n_nuisance = as.integer(fixture$data$N * fixture$data$n_tests),
        n_params_main = 67L,
        main_parameter_names = summary$variable,
        n_params_main_unconstrained = 67L,
        main_parameter_count_constrained = nrow(summary),
        min_ess_main = min_ess,
        max_rhat_main = max_rhat,
        divergences = sum(diagnostics[, , "divergent__"]),
        sampling_seconds = sampling_seconds,
        fit_wall_seconds = fit_wall_seconds,
        total_seconds = fit_wall_seconds,
        min_ess_per_sec_sampling = min_ess / sampling_seconds,
        min_ess_per_sec_total = min_ess / fit_wall_seconds,
        gradient_work = gradient_work,
        min_ess_per_1000_gradient_work = 1000 * min_ess / gradient_work,
        gradient_work_basis = "CmdStan sum of n_leapfrog__; gradient-work proxy", 
        metric_type_main = "NUTS",
        M_decay_type = NA_character_,
        phi_type = fixture$data$Phi_type,
        avx_kernel_path_active = FALSE,
        avx_kernel_path = "Stan built-in math",
        timings = fit$time()
    )
    saveRDS(result, output_rds)
    result
}
