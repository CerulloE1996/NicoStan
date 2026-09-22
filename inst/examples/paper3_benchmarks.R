#### =====================================================================================================================================
## Paper 3 / NicoStan benchmark orchestration
##
## This file is deliberately separate from the sampler implementation.  It runs
## the same simulated example, seed, initial values, and main-parameter ESS
## scope through CmdStanR, NicoStan, and NicoStan with the external AVX-512
## functions.  Each completed arm is written immediately and can be resumed by
## its configuration fingerprint.
##

paper3_benchmark_registry <- function() {
        data.frame(
            model = c("cox_frailty", "weibull", "robust_t4", "hierarchical_logistic",
                      "joint_longitudinal_survival", "gaussian_process", "stochastic_volatility",
                      "latent_diffusion_survival"),
            N_1 = c(100L, 100L, 250L, 300L, 25L, 40L, 250L, 100L),
            N_2 = c(250L, 250L, 1000L, 1200L, 75L, 80L, 1000L, 200L),
            N_3 = c(500L, 500L, 3000L, 4000L, 200L, 160L, 4000L, 800L),
            units = c("observations", "observations", "observations", "observations",
                      "subjects", "observations", "time_points", "subjects"),
            diffusion_eligible = c(TRUE, FALSE, FALSE, TRUE, TRUE, FALSE, TRUE, TRUE),
            stringsAsFactors = FALSE)
}

paper3_benchmark_arms <- function(include_avx2 = FALSE) {
        arms <- data.frame(
            arm = c("CmdStanR", "NicoStan", "NicoStan_AVX512"),
            engine = c("cmdstanr", "NicoStan", "NicoStan"),
            math_backend = c("Stan", "Stan", "AVX512"),
            stringsAsFactors = FALSE)
        if (isTRUE(include_avx2)) {
                arms <- rbind(arms, data.frame(arm = "NicoStan_AVX2", engine = "NicoStan",
                                               math_backend = "AVX2", stringsAsFactors = FALSE))
        }
        arms
}

paper3_benchmark_grid <- function(models = NULL, N_grid = NULL) {
        registry <- paper3_benchmark_registry()
        if (is.null(models)) models <- registry$model
        stopifnot(all(models %in% registry$model))
        rows <- lapply(models, function(model) {
                row <- registry[match(model, registry$model), , drop = FALSE]
                if (is.null(N_grid)) sizes <- as.integer(unlist(row[c("N_1", "N_2", "N_3")])) else {
                        sizes <- N_grid[[model]]
                        if (is.null(sizes)) stop(paste0("N_grid has no entry for model ", model, "."))
                }
                stopifnot(length(sizes) == 3L, all(is.finite(sizes)), all(sizes == as.integer(sizes)), all(sizes >= 12L))
                data.frame(model = model, N = as.integer(sizes), units = row$units,
                           diffusion_eligible = row$diffusion_eligible, stringsAsFactors = FALSE)
        })
        do.call(rbind, rows)
}

paper3_benchmark_cpu_has_isa <- function(math_backend) {
        if (math_backend == "Stan") return(TRUE)
        if (!file.exists("/proc/cpuinfo")) return(NA)
        lines <- readLines("/proc/cpuinfo", warn = FALSE)
        flags <- paste(lines[startsWith(lines, "flags")], collapse = " ")
        required <- if (math_backend == "AVX512") c("avx512f", "avx512vl", "avx512dq", "fma") else c("avx2", "fma")
        all(vapply(required, function(flag) grepl(paste0("\\b", flag, "\\b"), flags), logical(1L)))
}

paper3_benchmark_hash_files <- function(paths) {
        paths <- unique(as.character(paths))
        paths <- paths[nzchar(paths)]
        setNames(vapply(paths, function(path) {
                if (!file.exists(path)) return(NA_character_)
                digest::digest(file = path, algo = "sha256")
        }, character(1L)), basename(paths))
}

paper3_benchmark_package_fingerprint <- function(package_name) {
        package_path <- tryCatch(find.package(package_name), error = function(error) "")
        if (!nzchar(package_path)) return(list(path = NA_character_, files = NA_character_))
        package_files <- list.files(package_path, recursive = TRUE, full.names = TRUE)
        package_files <- package_files[grepl("(^|/)(R/.*\\.(rdb|rdx|rds)|libs/.*\\.(so|dll|dylib))$", package_files)]
        list(path = normalizePath(package_path, mustWork = TRUE), files = paper3_benchmark_hash_files(package_files))
}

paper3_benchmark_compile_arguments <- function(math_backend) {
        compile_arguments <- c("STAN_THREADS=true", "PRECOMPILED_HEADERS=false")
        if (math_backend == "Stan") return(compile_arguments)
        isa_flags <- if (math_backend == "AVX512") "-mavx2 -mavx512f -mavx512vl -mavx512dq" else
                     "-mavx2 -mno-avx512f -mno-avx512vl -mno-avx512dq"
        c(compile_arguments, paste0("CXXFLAGS=", shQuote(paste("-O3 -std=c++17 -march=native -mtune=native -mfma", isa_flags))))
}

paper3_benchmark_build_fingerprint <- function(engine, math_backend) {
        package_fingerprints <- list()
        if (engine == "NicoStan") package_fingerprints$NicoStan <- paper3_benchmark_package_fingerprint("NicoStan")
        if (engine == "cmdstanr") package_fingerprints$cmdstanr <- paper3_benchmark_package_fingerprint("cmdstanr")
        if (math_backend != "Stan") {
                package_fingerprints$BayesMVP <- paper3_benchmark_package_fingerprint("BayesMVP")
                header <- system.file("include", "BayesMVP", "stan_external_functions.hpp", package = "BayesMVP")
                header_files <- if (nzchar(header)) c(header, file.path(dirname(header), "math", c("fast_and_approx_AVX2_fns.hpp", "fast_and_approx_AVX512_fns.hpp"))) else character()
        } else header_files <- character()
        cmdstan_path <- tryCatch(cmdstanr::cmdstan_path(), error = function(error) NA_character_)
        cmdstan_version <- tryCatch(as.character(cmdstanr::cmdstan_version()), error = function(error) NA_character_)
        compiler <- tryCatch(system2("R", c("CMD", "config", "CXX17"), stdout = TRUE, stderr = TRUE), error = function(error) NA_character_)
        payload <- list(engine = engine, math_backend = math_backend,
                        compile_arguments = paper3_benchmark_compile_arguments(math_backend),
                        package_fingerprints = package_fingerprints,
                        external_header_hashes = paper3_benchmark_hash_files(header_files),
                        R = R.version$version.string, platform = R.version$platform,
                        cmdstan_path = cmdstan_path, cmdstan_version = cmdstan_version,
                        R_CXX17 = compiler,
                        environment = Sys.getenv(c("CC", "CXX", "CXX17", "CXXFLAGS", "MAKEFLAGS")))
        list(hash = digest::digest(payload, algo = "sha256"), payload = payload)
}

paper3_benchmark_item_fingerprint <- function(model, N, arm, seed, profile, chains, warmup,
                                               iterations, adapt_delta, burnin_algorithm,
                                               include_avx2, harness_path, model_dir,
                                               build_fingerprint = NULL) {
        registry <- paper3_example_registry()
        source_file <- registry$file[match(model, registry$model)]
        avx_file <- if (arm$math_backend == "Stan") source_file else sub(".stan$", "_avx.stan", source_file)
        source_paths <- file.path(model_dir, c(source_file, avx_file))
        source_hashes <- vapply(unique(source_paths), function(path) {
                if (!file.exists(path)) return(NA_character_)
                digest::digest(file = path, algo = "sha256")
        }, character(1L))
        harness_hash <- digest::digest(file = harness_path, algo = "sha256")
        orchestration_path <- file.path(dirname(harness_path), "paper3_benchmarks.R")
        orchestration_hash <- if (file.exists(orchestration_path)) digest::digest(file = orchestration_path, algo = "sha256") else NA_character_
        substr(digest::digest(list(version = "paper3_benchmark_v1", model = model, N = N,
                                   arm = arm, seed = seed, profile = profile, chains = chains,
                                   warmup = warmup, iterations = iterations, adapt_delta = adapt_delta,
                                   burnin_algorithm = burnin_algorithm, include_avx2 = include_avx2,
                                   source_hashes = source_hashes, harness_hash = harness_hash,
                                   orchestration_hash = orchestration_hash,
                                   build_fingerprint = build_fingerprint), algo = "sha256"), 1L, 16L)
}

paper3_benchmark_read_manifest <- function(path) {
        if (!file.exists(path)) return(data.frame(stringsAsFactors = FALSE))
        utils::read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
}

paper3_benchmark_write_manifest <- function(manifest, path) {
        dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
        utils::write.csv(manifest, path, row.names = FALSE)
        invisible(manifest)
}

paper3_benchmark_upsert_manifest <- function(manifest, row) {
        if (!nrow(manifest)) return(row)
        for (name in setdiff(names(row), names(manifest))) manifest[[name]] <- NA_character_
        for (name in setdiff(names(manifest), names(row))) row[[name]] <- NA_character_
        row <- row[, names(manifest), drop = FALSE]
        key <- manifest$fingerprint == row$fingerprint
        if (any(key)) manifest[which(key)[1L], names(row)] <- row[1L, names(row)] else manifest <- rbind(manifest, row)
        manifest
}

paper3_benchmark_summary_row <- function(record, model, N, arm, replicate, fingerprint) {
        settings <- record$settings
        nested_status <- if (!is.null(record$nested_rhat_grouping$status)) record$nested_rhat_grouping$status else "unavailable"
        nested_value <- if (!is.null(record$max_nested_rhat_main)) record$max_nested_rhat_main else NA_real_
        data.frame(
            model = model, N = as.integer(N), arm = arm, replicate = as.integer(replicate),
            fingerprint = fingerprint, engine = record$engine, math_backend = record$math_backend,
            build_fingerprint = if (!is.null(record$benchmark$build_fingerprint)) record$benchmark$build_fingerprint$hash else NA_character_,
            diffusion = isTRUE(record$diffusion), seed = settings$seed, chains = settings$chains,
            warmup = settings$warmup, iterations = settings$iterations,
            compilation_seconds = record$compilation_seconds, fitting_seconds = record$fitting_seconds,
            sampling_seconds = record$sampling_seconds, sampling_time_basis = record$sampling_time_basis,
            divergences = record$divergences,
            max_rhat_main = max(record$summary$rhat), min_ess_bulk_main = record$min_ESS_main,
            min_ess_tail_main = record$min_ESS_tail_main,
            ess_per_second_sampling_main = record$min_ESS_per_sec_sampling,
            ess_per_second_fitting_main = record$min_ESS_per_sec_fitting,
            gradient_work_sampling_proxy = record$gradient_work_sampling,
            gradient_work_basis = record$gradient_work_basis,
            exact_gradient_evaluations = record$exact_gradient_evaluations,
            ess_per_1000_gradient_work_main = record$min_ESS_per_1000_grad_sampling,
            max_nested_rhat_main = nested_value,
            nested_rhat_grouping_status = nested_status,
            diagnostic_target_met = isTRUE(record$diagnostic_target_met),
            data_sha256 = record$data_sha256, initial_values_sha256 = record$initial_values_sha256,
            initial_values_identity_scope = "user-provided input initial values before sampler adaptation",
            ess_parameter_set = paste(record$ess_parameter_names, collapse = "|"),
            ess_parameter_bases = paste(record$ess_parameter_bases, collapse = "|"),
            run_dir = record$run_dir, stringsAsFactors = FALSE)
}

paper3_benchmark_run <- function(models = NULL, N_grid = NULL,
                                  arms = NULL,
                                  profile = c("analysis", "smoke"), seed = 2026L,
                                  replicate = 1L, chains = 4L, warmup = NULL, iterations = NULL,
                                  adapt_delta = NULL, burnin_algorithm = "CHESSR",
                                  include_avx2 = FALSE, validate = FALSE, fail_fast = FALSE,
                                  output_dir = file.path(getwd(), "paper3_benchmark_results"),
                                  model_dir = file.path(.paper3_source_directory, "models")) {
        profile <- match.arg(profile)
        stopifnot(length(replicate) == 1L, replicate >= 1L, replicate == as.integer(replicate))
        if (is.null(warmup)) warmup <- if (profile == "smoke") 100L else 1000L
        if (is.null(iterations)) iterations <- if (profile == "smoke") 100L else 1000L
        grid <- paper3_benchmark_grid(models = models, N_grid = N_grid)
        if (is.null(arms)) arms <- paper3_benchmark_arms(include_avx2 = include_avx2)
        arms <- as.data.frame(arms, stringsAsFactors = FALSE)
        required_arm_columns <- c("arm", "engine", "math_backend")
        stopifnot(all(required_arm_columns %in% names(arms)), nrow(arms) > 0L)
        if (is.null(models)) models <- unique(grid$model)
        harness_path <- file.path(.paper3_source_directory, "paper3_examples.R")
        manifest_path <- file.path(output_dir, "benchmark_manifest.csv")
        manifest <- paper3_benchmark_read_manifest(manifest_path)
        dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
        summary_rows <- list()
        counter <- 0L
        total <- nrow(grid) * nrow(arms)
        for (grid_index in seq_len(nrow(grid))) {
                model <- grid$model[grid_index]
                N <- grid$N[grid_index]
                model_registry <- paper3_example_registry()
                diffusion_eligible <- model_registry$diffusion_eligible[match(model, model_registry$model)]
                for (arm_index in seq_len(nrow(arms))) {
                        arm <- arms[arm_index, , drop = FALSE]
                        if (arm$engine == "cmdstanr" && arm$math_backend != "Stan") stop("CmdStanR benchmark arms must use math_backend='Stan'.")
                        if (arm$engine == "NicoStan" && arm$math_backend == "AVX2" && !isTRUE(include_avx2)) stop("AVX2 requested but include_avx2=FALSE.")
                        if (arm$math_backend != "Stan" && identical(paper3_benchmark_cpu_has_isa(arm$math_backend), FALSE)) {
                                stop(paste0("This CPU does not expose the required ", arm$math_backend, " instruction set."))
                        }
                        ## The arm is deliberately absent from the seed.  This is
                        ## what makes the simulated data and every chain's initial
                        ## values identical across CmdStanR, NicoStan, and AVX.
                        item_seed <- as.integer(seed + 100000L * (grid_index - 1L) + replicate - 1L)
                        build_fingerprint <- paper3_benchmark_build_fingerprint(engine = arm$engine, math_backend = arm$math_backend)
                        fingerprint <- paper3_benchmark_item_fingerprint(model = model, N = N, arm = arm,
                            seed = item_seed, profile = profile, chains = chains, warmup = warmup,
                            iterations = iterations, adapt_delta = adapt_delta, burnin_algorithm = burnin_algorithm,
                            include_avx2 = include_avx2, harness_path = harness_path, model_dir = model_dir,
                            build_fingerprint = build_fingerprint$hash)
                        counter <- counter + 1L
                        existing <- nrow(manifest) && any(manifest$fingerprint == fingerprint & manifest$status == "complete")
                        if (existing) {
                                result_path <- manifest$result_path[which(manifest$fingerprint == fingerprint & manifest$status == "complete")[1L]]
                                if (file.exists(result_path)) {
                                        record <- tryCatch(readRDS(result_path), error = function(error) NULL)
                                        if (is.list(record) && is.list(record$benchmark) && identical(record$benchmark$fingerprint, fingerprint)) {
                                                summary_rows[[length(summary_rows) + 1L]] <- paper3_benchmark_summary_row(record, model, N, arm$arm, replicate, fingerprint)
                                                next
                                        }
                                }
                        }
                        item_output <- file.path(output_dir, "fits", paste0(model, "_N", N, "_rep", replicate, "_", arm$arm, "_", fingerprint))
                        status <- "error"
                        result_path <- NA_character_
                        error_message <- NA_character_
                        started <- Sys.time()
                        message(paste0("[", counter, "/", total, "] ", model, " N=", N, " / ", arm$arm))
                        result <- tryCatch({
                                run_paper3_example(model = model, engine = arm$engine, math_backend = arm$math_backend,
                                    profile = profile, burnin_algorithm = burnin_algorithm, seed = item_seed,
                                    chains = chains, warmup = warmup, iterations = iterations, adapt_delta = adapt_delta,
                                    N = N, validate = validate, output_dir = item_output, model_dir = model_dir)
                        }, error = function(error) {
                                error_message <<- conditionMessage(error)
                                if (isTRUE(fail_fast)) stop(error)
                                NULL
                        })
                        if (!is.null(result)) {
                                status <- "complete"
                                result$benchmark <- list(fingerprint = fingerprint, build_fingerprint = build_fingerprint,
                                                         model = model, N = N, arm = arm$arm, replicate = replicate)
                                result_path <- file.path(item_output, "benchmark_record.rds")
                                saveRDS(result, result_path)
                                summary_rows[[length(summary_rows) + 1L]] <- paper3_benchmark_summary_row(result, model, N, arm$arm, replicate, fingerprint)
                        }
                        manifest_row <- data.frame(fingerprint = fingerprint, model = model, N = N, arm = arm$arm,
                            replicate = replicate, seed = item_seed, profile = profile, status = status,
                            started = as.character(started), finished = as.character(Sys.time()),
                            build_fingerprint = build_fingerprint$hash, result_path = result_path,
                            error_message = error_message, stringsAsFactors = FALSE)
                        manifest <- paper3_benchmark_upsert_manifest(manifest, manifest_row)
                        paper3_benchmark_write_manifest(manifest, manifest_path)
                }
        }
        summary <- if (length(summary_rows)) do.call(rbind, summary_rows) else data.frame(stringsAsFactors = FALSE)
        comparison <- paper3_benchmark_compare(summary)
        output <- list(summary = summary, comparison = comparison, manifest = manifest,
                       grid = grid, arms = arms, output_dir = normalizePath(output_dir, mustWork = TRUE))
        saveRDS(output, file.path(output_dir, "benchmark_results.rds"))
        if (nrow(summary)) utils::write.csv(summary, file.path(output_dir, "benchmark_summary.csv"), row.names = FALSE)
        if (nrow(comparison$efficiency)) utils::write.csv(comparison$efficiency, file.path(output_dir, "efficiency_comparisons.csv"), row.names = FALSE)
        if (nrow(comparison$posterior_agreement)) utils::write.csv(comparison$posterior_agreement, file.path(output_dir, "posterior_agreement.csv"), row.names = FALSE)
        if (nrow(comparison$checks)) utils::write.csv(comparison$checks, file.path(output_dir, "comparison_checks.csv"), row.names = FALSE)
        output
}

paper3_benchmark_compare <- function(summary) {
        empty <- data.frame(stringsAsFactors = FALSE)
        if (!nrow(summary)) return(list(efficiency = empty, posterior_agreement = empty, checks = empty))
        keys <- unique(summary[c("model", "N", "replicate")])
        efficiency_rows <- list()
        agreement_rows <- list()
        check_rows <- list()
        for (index in seq_len(nrow(keys))) {
                subset <- summary[summary$model == keys$model[index] & summary$N == keys$N[index] & summary$replicate == keys$replicate[index], , drop = FALSE]
                reference <- subset[subset$arm == "CmdStanR", , drop = FALSE]
                if (nrow(reference) != 1L) next
                check_rows[[length(check_rows) + 1L]] <- data.frame(
                    model = keys$model[index], N = keys$N[index], replicate = keys$replicate[index],
                    data_identical = length(unique(subset$data_sha256)) == 1L,
                    initial_values_identical = length(unique(subset$initial_values_sha256)) == 1L,
                    ess_parameter_set_identical = length(unique(subset$ess_parameter_set)) == 1L,
                    ess_parameter_bases_identical = length(unique(subset$ess_parameter_bases)) == 1L,
                    arms_completed = nrow(subset), all_diagnostic_targets_met = all(subset$diagnostic_target_met),
                    nested_rhat_statuses = paste(unique(subset$nested_rhat_grouping_status), collapse = "|"),
                    stringsAsFactors = FALSE)
                for (row_index in seq_len(nrow(subset))) {
                        row <- subset[row_index, , drop = FALSE]
                        efficiency_rows[[length(efficiency_rows) + 1L]] <- data.frame(
                            model = row$model, N = row$N, replicate = row$replicate, arm = row$arm,
                            reference_arm = "CmdStanR", fitting_time_ratio_vs_cmdstanr = row$fitting_seconds / reference$fitting_seconds,
                            sampling_time_ratio_vs_cmdstanr = row$sampling_seconds / reference$sampling_seconds,
                            sampling_speedup_vs_cmdstanr = reference$sampling_seconds / row$sampling_seconds,
                            ess_per_second_sampling_ratio_vs_cmdstanr = row$ess_per_second_sampling_main / reference$ess_per_second_sampling_main,
                            ess_per_1000_gradient_work_ratio_vs_cmdstanr = row$ess_per_1000_gradient_work_main / reference$ess_per_1000_gradient_work_main,
                            time_to_ESS_100_sampling_extrapolated_s = 100 / row$ess_per_second_sampling_main,
                            time_to_ESS_1000_sampling_extrapolated_s = 1000 / row$ess_per_second_sampling_main,
                            time_to_ESS_10000_sampling_extrapolated_s = 10000 / row$ess_per_second_sampling_main,
                            time_to_ESS_basis = "linear extrapolation from observed main-parameter min ESS/sec; diagnostic status retained separately",
                            gradient_work_basis = row$gradient_work_basis,
                            max_nested_rhat_main = row$max_nested_rhat_main,
                            nested_rhat_grouping_status = row$nested_rhat_grouping_status,
                            diagnostic_target_met = row$diagnostic_target_met, stringsAsFactors = FALSE)
                }
                common_parameters <- strsplit(reference$ess_parameter_set, "\\|", fixed = FALSE)[[1L]]
                for (row_index in seq_len(nrow(subset))) {
                        row <- subset[row_index, , drop = FALSE]
                        result_path <- summary$run_dir[match(row$fingerprint, summary$fingerprint)]
                        reference_path <- summary$run_dir[match(reference$fingerprint, summary$fingerprint)]
                        if (!length(result_path) || !length(reference_path) || !file.exists(file.path(result_path, "result.rds")) || !file.exists(file.path(reference_path, "result.rds"))) next
                        ## The benchmark wrapper is the durable record; fall back to the
                        ## native run record for older results that predate this wrapper.
                        record_path <- file.path(dirname(result_path), "benchmark_record.rds")
                        ref_record_path <- file.path(dirname(reference_path), "benchmark_record.rds")
                        record <- if (file.exists(record_path)) readRDS(record_path) else readRDS(file.path(result_path, "result.rds"))
                        ref_record <- if (file.exists(ref_record_path)) readRDS(ref_record_path) else readRDS(file.path(reference_path, "result.rds"))
                        current <- record$summary
                        ref <- ref_record$summary
                        keep <- intersect(common_parameters, intersect(as.character(current$variable), as.character(ref$variable)))
                        if (!length(keep)) next
                        current <- current[match(keep, current$variable), , drop = FALSE]
                        ref <- ref[match(keep, ref$variable), , drop = FALSE]
                        agreement_rows[[length(agreement_rows) + 1L]] <- data.frame(
                            model = row$model, N = row$N, replicate = row$replicate, arm = row$arm,
                            reference_arm = "CmdStanR", parameters_compared = length(keep),
                            mean_absolute_posterior_mean_difference = mean(abs(current$mean - ref$mean)),
                            max_absolute_posterior_mean_difference = max(abs(current$mean - ref$mean)),
                            mean_relative_sd_difference = mean(abs(current$sd - ref$sd) / pmax(1e-12, ref$sd)),
                            max_absolute_standardised_mean_difference = max(abs(current$mean - ref$mean) / pmax(1e-12, ref$sd)),
                            stringsAsFactors = FALSE)
                }
        }
        list(efficiency = if (length(efficiency_rows)) do.call(rbind, efficiency_rows) else empty,
             posterior_agreement = if (length(agreement_rows)) do.call(rbind, agreement_rows) else empty,
             checks = if (length(check_rows)) do.call(rbind, check_rows) else empty)
}

paper3_benchmark_collect <- function(output_dir) {
        path <- file.path(output_dir, "benchmark_results.rds")
        if (file.exists(path)) return(readRDS(path))
        manifest <- paper3_benchmark_read_manifest(file.path(output_dir, "benchmark_manifest.csv"))
        if (!nrow(manifest)) return(list(summary = data.frame(stringsAsFactors = FALSE)))
        records <- lapply(manifest$result_path[manifest$status == "complete"], function(path) if (file.exists(path)) readRDS(path) else NULL)
        records <- Filter(Negate(is.null), records)
        if (!length(records)) return(list(summary = data.frame(stringsAsFactors = FALSE)))
        summary <- do.call(rbind, lapply(records, function(record) paper3_benchmark_summary_row(record,
            record$benchmark$model, record$benchmark$N, record$benchmark$arm, record$benchmark$replicate,
            record$benchmark$fingerprint)))
        list(summary = summary, comparison = paper3_benchmark_compare(summary), manifest = manifest)
}
