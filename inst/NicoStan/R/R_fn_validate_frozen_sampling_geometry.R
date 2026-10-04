#### =============================================================================================================
## R_fn_validate_frozen_sampling_geometry.R
##
## ---- sampling-boundary validation for the frozen state, trajectory and metric ---------------------------------


R_fn_validate_frozen_sampling_geometry <-  function( EHMC_args_as_Rcpp_List,
        EHMC_Metric_as_Rcpp_List,
        theta_main,
        theta_nuisance,
        n_chains_sampling,
        sample_nuisance,
        partitioned_HMC = TRUE,
        Model_args_as_Rcpp_List = NULL,
        tolerance = 1e-7) {

        fn_stop <-  function(reason) {
                stop(paste0("Frozen sampling geometry is invalid: ", reason), call. = FALSE)
        }

        fn_require_list <-  function(value, label) {
                if (!is.list(value)) fn_stop(paste0(label, " must be a list."))
                invisible(TRUE)
        }

        fn_get_field <-  function(value, field, label) {
                if (!(field %in% names(value))) {
                        fn_stop(paste0(label, "$", field, " is missing."))
                }
                value[[field]]
        }

        fn_require_flag <-  function(value, label) {
                if (!is.logical(value) || length(value) != 1 || is.na(value)) {
                        fn_stop(paste0(label, " must be one non-missing logical value."))
                }
                isTRUE(value)
        }

        fn_require_numeric <-  function(value,
                                       label,
                                       expected_length = NULL,
                                       positive = FALSE) {
                if (!is.numeric(value) || is.complex(value) ||
                    (!is.null(expected_length) && length(value) != expected_length) ||
                    any(!is.finite(value))) {
                        length_text <-  if (is.null(expected_length)) "" else
                                paste0(" of length ", expected_length)
                        fn_stop(paste0(label, " must be a finite numeric value", length_text, "."))
                }
                if (positive && any(value <= 0)) {
                        fn_stop(paste0(label, " must contain only positive values."))
                }
                invisible(value)
        }

        fn_require_matrix <-  function(value,
                                      label,
                                      expected_nrow = NULL,
                                      expected_ncol) {
                if (!is.matrix(value) || !is.numeric(value) || is.complex(value) ||
                    (!is.null(expected_nrow) && nrow(value) != expected_nrow) ||
                    ncol(value) != expected_ncol) {
                        dimension_text <-  if (is.null(expected_nrow)) {
                                paste0("? x ", expected_ncol)
                        } else {
                                paste0(expected_nrow, " x ", expected_ncol)
                        }
                        fn_stop(paste0(label, " must be a numeric matrix with dimensions ",
                                       dimension_text, "."))
                }
                if (any(!is.finite(value))) {
                        fn_stop(paste0(label, " must contain only finite values."))
                }
                invisible(value)
        }

        fn_require_close <-  function(actual, target, label) {
                residual <-  actual - target
                if (any(!is.finite(residual)) || any(abs(residual) > tolerance)) {
                        fn_stop(paste0(label, " exceeds tolerance = ", format(tolerance, scientific = TRUE), "."))
                }
                invisible(TRUE)
        }

        fn_require_spd <-  function(value, label) {
                if (any(abs(value - t(value)) > tolerance)) {
                        fn_stop(paste0(label, " must be symmetric within tolerance = ",
                                       format(tolerance, scientific = TRUE), "."))
                }
                tryCatch(
                        base::chol(value),
                        error = function(error_condition) {
                                fn_stop(paste0(label, " must be positive definite."))
                        })
                invisible(TRUE)
        }

        fn_require_list(EHMC_args_as_Rcpp_List, "EHMC_args_as_Rcpp_List")
        fn_require_list(EHMC_Metric_as_Rcpp_List, "EHMC_Metric_as_Rcpp_List")

        if (!is.numeric(tolerance) || is.complex(tolerance) || length(tolerance) != 1 ||
            !is.finite(tolerance) || tolerance < 0) {
                fn_stop("tolerance must be one finite non-negative numeric value.")
        }

        if (!is.numeric(n_chains_sampling) || is.complex(n_chains_sampling) ||
            length(n_chains_sampling) != 1 || !is.finite(n_chains_sampling) ||
            n_chains_sampling < 1 || n_chains_sampling != floor(n_chains_sampling) ||
            n_chains_sampling > .Machine$integer.max) {
                fn_stop("n_chains_sampling must be one positive finite whole number.")
        }
        n_chains_sampling <-  as.integer(n_chains_sampling)

        sample_nuisance <-  fn_require_flag(sample_nuisance, "sample_nuisance")
        partitioned_HMC <-  fn_require_flag(partitioned_HMC, "partitioned_HMC")
        diffusion_HMC <-  fn_require_flag(
                fn_get_field(EHMC_args_as_Rcpp_List, "diffusion_HMC", "EHMC_args_as_Rcpp_List"),
                "EHMC_args_as_Rcpp_List$diffusion_HMC")

        randomize_tau <-  TRUE
        if ("randomize_tau" %in% names(EHMC_args_as_Rcpp_List)) {
                randomize_tau <-  fn_require_flag(
                        EHMC_args_as_Rcpp_List[["randomize_tau"]],
                        "EHMC_args_as_Rcpp_List$randomize_tau")
        }

        expected_n_main <-  NULL
        expected_n_nuisance <-  NULL
        if (!is.null(Model_args_as_Rcpp_List)) {
                fn_require_list(Model_args_as_Rcpp_List, "Model_args_as_Rcpp_List")
                expected_n_main <-  fn_get_field(Model_args_as_Rcpp_List,
                                                "n_params_main", "Model_args_as_Rcpp_List")
                expected_n_nuisance <-  fn_get_field(Model_args_as_Rcpp_List,
                                                    "n_nuisance", "Model_args_as_Rcpp_List")
                fn_require_numeric(expected_n_main, "Model_args_as_Rcpp_List$n_params_main", 1, TRUE)
                fn_require_numeric(expected_n_nuisance, "Model_args_as_Rcpp_List$n_nuisance", 1)
                if (expected_n_main != floor(expected_n_main) ||
                    expected_n_nuisance != floor(expected_n_nuisance) || expected_n_nuisance < 0) {
                        fn_stop("Model parameter counts must be whole numbers with non-negative nuisance count.")
                }
        }

        fn_require_matrix(theta_main,
                          "theta_main",
                          expected_nrow = expected_n_main,
                          expected_ncol = n_chains_sampling)
        n_main <-  nrow(theta_main)
        if (n_main < 1) fn_stop("theta_main must have at least one row.")

        fn_require_matrix(theta_nuisance,
                          "theta_nuisance",
                          expected_nrow = expected_n_nuisance,
                          expected_ncol = n_chains_sampling)
        n_nuisance <-  nrow(theta_nuisance)
        nuisance_active <-  n_nuisance > 0 &&
                ((!partitioned_HMC && diffusion_HMC) ||
                 (partitioned_HMC && sample_nuisance))

        fn_require_numeric(
                fn_get_field(EHMC_args_as_Rcpp_List, "eps_main", "EHMC_args_as_Rcpp_List"),
                "EHMC_args_as_Rcpp_List$eps_main",
                expected_length = 1,
                positive = TRUE)
        fn_require_numeric(
                fn_get_field(EHMC_args_as_Rcpp_List, "tau_main", "EHMC_args_as_Rcpp_List"),
                "EHMC_args_as_Rcpp_List$tau_main",
                expected_length = 1,
                positive = TRUE)

        if (nuisance_active && partitioned_HMC) {
                fn_require_numeric(
                        fn_get_field(EHMC_args_as_Rcpp_List, "eps_us", "EHMC_args_as_Rcpp_List"),
                        "EHMC_args_as_Rcpp_List$eps_us",
                        expected_length = 1,
                        positive = TRUE)
                fn_require_numeric(
                        fn_get_field(EHMC_args_as_Rcpp_List, "tau_us", "EHMC_args_as_Rcpp_List"),
                        "EHMC_args_as_Rcpp_List$tau_us",
                        expected_length = 1,
                        positive = TRUE)
        }

        use_given_tau_main_ii <-  FALSE
        if ("use_given_tau_main_ii" %in% names(EHMC_args_as_Rcpp_List)) {
                use_given_tau_main_ii <-  fn_require_flag(
                        EHMC_args_as_Rcpp_List[["use_given_tau_main_ii"]],
                        "EHMC_args_as_Rcpp_List$use_given_tau_main_ii")
        }
        if (use_given_tau_main_ii && randomize_tau) {
                fn_require_numeric(
                        fn_get_field(EHMC_args_as_Rcpp_List, "tau_main_ii", "EHMC_args_as_Rcpp_List"),
                        "EHMC_args_as_Rcpp_List$tau_main_ii",
                        expected_length = 1,
                        positive = TRUE)
        }

        metric_shape_main <-  fn_get_field(
                EHMC_Metric_as_Rcpp_List,
                "metric_shape_main",
                "EHMC_Metric_as_Rcpp_List")
        if (!is.character(metric_shape_main) || length(metric_shape_main) != 1 ||
            is.na(metric_shape_main) || !(metric_shape_main %in% c("diag", "dense"))) {
                fn_stop("EHMC_Metric_as_Rcpp_List$metric_shape_main must be 'diag' or 'dense'.")
        }

        if (identical(metric_shape_main, "diag")) {
                M_inv_main_vec <-  fn_get_field(
                        EHMC_Metric_as_Rcpp_List,
                        "M_inv_main_vec",
                        "EHMC_Metric_as_Rcpp_List")
                fn_require_numeric(
                        M_inv_main_vec,
                        "EHMC_Metric_as_Rcpp_List$M_inv_main_vec",
                        expected_length = n_main,
                        positive = TRUE)
        } else {
                M_dense_main <-  fn_get_field(
                        EHMC_Metric_as_Rcpp_List,
                        "M_dense_main",
                        "EHMC_Metric_as_Rcpp_List")
                M_inv_dense_main <-  fn_get_field(
                        EHMC_Metric_as_Rcpp_List,
                        "M_inv_dense_main",
                        "EHMC_Metric_as_Rcpp_List")
                M_inv_dense_main_chol <-  fn_get_field(
                        EHMC_Metric_as_Rcpp_List,
                        "M_inv_dense_main_chol",
                        "EHMC_Metric_as_Rcpp_List")

                fn_require_matrix(M_dense_main, "EHMC_Metric_as_Rcpp_List$M_dense_main", n_main, n_main)
                fn_require_matrix(M_inv_dense_main, "EHMC_Metric_as_Rcpp_List$M_inv_dense_main", n_main, n_main)
                fn_require_matrix(M_inv_dense_main_chol,
                                  "EHMC_Metric_as_Rcpp_List$M_inv_dense_main_chol",
                                  n_main,
                                  n_main)
                fn_require_spd(M_dense_main, "EHMC_Metric_as_Rcpp_List$M_dense_main")
                fn_require_spd(M_inv_dense_main, "EHMC_Metric_as_Rcpp_List$M_inv_dense_main")
                fn_require_close(M_dense_main %*% M_inv_dense_main,
                                 diag(n_main),
                                 "EHMC_Metric_as_Rcpp_List$M_dense_main %*% M_inv_dense_main")

                upper_entries <-  M_inv_dense_main_chol[upper.tri(M_inv_dense_main_chol)]
                if (length(upper_entries) > 0 && any(abs(upper_entries) > tolerance)) {
                        fn_stop("EHMC_Metric_as_Rcpp_List$M_inv_dense_main_chol must be lower triangular.")
                }
                if (any(diag(M_inv_dense_main_chol) <= 0)) {
                        fn_stop(paste0("EHMC_Metric_as_Rcpp_List$M_inv_dense_main_chol must have ",
                                       "positive diagonal entries."))
                }
                fn_require_close(M_inv_dense_main_chol %*% t(M_inv_dense_main_chol),
                                 M_inv_dense_main,
                                 "EHMC_Metric_as_Rcpp_List$M_inv_dense_main_chol %*% t(M_inv_dense_main_chol)")
        }

        if (nuisance_active) {
                M_us_vec <-  fn_get_field(
                        EHMC_Metric_as_Rcpp_List,
                        "M_us_vec",
                        "EHMC_Metric_as_Rcpp_List")
                M_inv_us_vec <-  fn_get_field(
                        EHMC_Metric_as_Rcpp_List,
                        "M_inv_us_vec",
                        "EHMC_Metric_as_Rcpp_List")
                fn_require_numeric(M_us_vec,
                                   "EHMC_Metric_as_Rcpp_List$M_us_vec",
                                   expected_length = n_nuisance,
                                   positive = TRUE)
                fn_require_numeric(M_inv_us_vec,
                                   "EHMC_Metric_as_Rcpp_List$M_inv_us_vec",
                                   expected_length = n_nuisance,
                                   positive = TRUE)
                fn_require_close(as.numeric(M_us_vec) * as.numeric(M_inv_us_vec),
                                 rep(1, n_nuisance),
                                 "EHMC_Metric_as_Rcpp_List$M_us_vec * M_inv_us_vec")

                if (diffusion_HMC) {
                        theta_hat_us_vec <-  fn_get_field(
                                EHMC_Metric_as_Rcpp_List,
                                "theta_hat_us_vec",
                                "EHMC_Metric_as_Rcpp_List")
                        fn_require_numeric(theta_hat_us_vec,
                                           "EHMC_Metric_as_Rcpp_List$theta_hat_us_vec",
                                           expected_length = n_nuisance)
                }
        }

        return(invisible(TRUE))
}























