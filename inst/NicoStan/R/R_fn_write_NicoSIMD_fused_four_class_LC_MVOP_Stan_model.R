##
## ---- writes the four-class native sources from the Stan source: --------------------------------------------------------------------
##

fn_write_NicoSIMD_four_class_LC_MVOP_Stan_model <-  function(
      source_Stan_model_file_path,
      NicoSIMD_Stan_model_file_path,
      fuse_likelihood
##
## ---- Block -------------------------------------------------------------------------------------------------------------------------
##
) {

##
## ---- Conditions --------------------------------------------------------------------------------------------------------------------
##
      if (!is.logical(fuse_likelihood) || length(fuse_likelihood) != 1L || is.na(fuse_likelihood)) {
            stop("The likelihood category must be supplied explicitly.")
      }
      source_path <-  normalizePath(
            source_Stan_model_file_path,
            mustWork = TRUE
      )
      output_directory <-  normalizePath(
            dirname(NicoSIMD_Stan_model_file_path),
            mustWork = TRUE
      )
      output_path <-  file.path(
            output_directory,
            basename(NicoSIMD_Stan_model_file_path)
      )
      source_name <-  paste0(
            "four_class_LC_MVOP_no_loops_over_N_or_standardisation_profiles_",
                             "Stan_math_for_layout_timings.stan"
      )
      if (!identical(
            basename(source_path),
            source_name
##
## ---- Block -------------------------------------------------------------------------------------------------------------------------
##
      )) {
            stop("The four-class source is not the supported Stan reference.")
      }
      if (identical(
            source_path,
            output_path
##
## ---- Block -------------------------------------------------------------------------------------------------------------------------
##
      ) || file.exists(output_path)) {
            stop("The output must be a new file, not the source.")
      }

      source_connection <-  file(
            source_path,
            open = "rb"
      )
      source_bytes <-  readBin(
            source_connection,
            what = "raw",
            n = file.info(source_path)$size
      )
      close(source_connection)
      if (length(source_bytes) == 0) stop("The source is empty.")

##
## ---- masks comments and strings, keeping byte positions: ---------------------------------------------------------------------------
##

##
## ---- Function ----------------------------------------------------------------------------------------------------------------------
##
      fn_mask <-  function(bytes) {

            masked <-  bytes
            values <-  as.integer(bytes)
            state <-  "code"
            escaped <-  FALSE
            position <-  1L
##
## ---- Loop --------------------------------------------------------------------------------------------------------------------------
##
            while (position <= length(values)) {
                  current <-  values[position]
                  following <-  if (position < length(values)) values[position + 1L] else 0L
##
## ---- Conditions --------------------------------------------------------------------------------------------------------------------
##
                  if (state == "line") {
                        if (current %in% c(
                              10L,
                              13L
                        )) state <-  "code"
                        else masked[position] <-  as.raw(32L)
##
## ---- Conditions --------------------------------------------------------------------------------------------------------------------
##
                  } else if (state == "block") {
##
## ---- Conditions --------------------------------------------------------------------------------------------------------------------
##
                        if (current == 42L && following == 47L) {
                              masked[position:(position + 1L)] <-  as.raw(32L)
                              position <-  position + 1L
                              state <-  "code"
                        } else if (!(current %in% c(
                              10L,
                              13L
                        ))) masked[position] <-  as.raw(32L)
##
## ---- Conditions --------------------------------------------------------------------------------------------------------------------
##
                  } else if (state == "string") {
                        if (!(current %in% c(
                              10L,
                              13L
                        ))) masked[position] <-  as.raw(32L)
                        if (escaped) escaped <-  FALSE
                        else if (current == 92L) escaped <-  TRUE
                        else if (current == 34L) state <-  "code"
                  } else if (current == 47L && following %in% c(
                        47L,
                        42L
##
## ---- Block -------------------------------------------------------------------------------------------------------------------------
##
                  )) {
                        masked[position:(position + 1L)] <-  as.raw(32L)
                        state <-  if (following == 47L) "line" else "block"
                        position <-  position + 1L
##
## ---- Conditions --------------------------------------------------------------------------------------------------------------------
##
                  } else if (current == 34L) {
                        masked[position] <-  as.raw(32L)
                        state <-  "string"
                  }
                  position <-  position + 1L
            }
            if (state %in% c(
                  "block",
                  "string"
            )) stop("The source has an unterminated comment or string.")
            return(masked)

      }

      masked <-  rawToChar(fn_mask(source_bytes))
      if (grepl(
            "(?<![A-Za-z0-9_])nico_[A-Za-z0-9_]+",
            masked,
            perl = TRUE
##
## ---- Block -------------------------------------------------------------------------------------------------------------------------
##
      )) {
            stop("The reference already contains native backend calls.")
      }

##
## ---- Function ----------------------------------------------------------------------------------------------------------------------
##
      fn_one_match <-  function(pattern) {

            positions <-  gregexpr(
                  pattern,
                  masked,
                  perl = TRUE,
                  useBytes = TRUE
            )[[1]]
##
## ---- Conditions --------------------------------------------------------------------------------------------------------------------
##
            if (length(positions) != 1L || positions[1] < 1L) {
                  stop("The source span is missing or ambiguous.")
            }
            return(c(
                  start = positions[1],
                  length = attr(
                  positions,
                  "match.length"
            )[1]
            ))

      }

##
## ---- Function ----------------------------------------------------------------------------------------------------------------------
##
      fn_block <-  function(pattern) {

            span <-  fn_one_match(pattern)
            opening <-  unname(span["start"] + span["length"] - 1L)
            values <-  as.integer(charToRaw(masked))
            depth <-  1L
            position <-  opening + 1L
##
## ---- Loop --------------------------------------------------------------------------------------------------------------------------
##
            while (position <= length(values) && depth > 0L) {
                  if (values[position] == 123L) depth <-  depth + 1L
                  if (values[position] == 125L) depth <-  depth - 1L
                  position <-  position + 1L
            }
            if (depth != 0L) stop("The source block has no closing brace.")
            return(c(
                  opening = opening,
                  closing = position - 1L
            ))

      }

      functions_block <-  fn_block("(?<![A-Za-z0-9_])functions\\s*\\{")
      data_block <-  fn_block("(?m)^data\\s*\\{")
      transformed_data_block <-  fn_block("(?<![A-Za-z0-9_])transformed\\s+data\\s*\\{")
      parameters_block <-  fn_block("(?m)^parameters\\s*\\{")
      model_block <-  fn_block("(?m)^model\\s*\\{")

##
## ---- checks the complete signatures and the data types read by the mapping: --------------------------------------------------------
##

      fn_guard_signature <-  function(
            return_type,
                                        function_name,
                                        arguments
##
## ---- Block -------------------------------------------------------------------------------------------------------------------------
##
      ) {

            span <-  fn_one_match(paste0(
                  "(?<![A-Za-z0-9_])",
                  return_type,
                  "\\s+",
                  function_name,
                                          "\\s*\\([^{};]*\\)\\s*\\{"
            ))
            observed <-  substr(
                  masked,
                  span["start"],
                  span["start"] + span["length"] - 1L
            )
            expected <-  paste0(
                  return_type,
                  function_name,
                  "(",
                  paste(
                  arguments,
                  collapse = ","
            ),
                  "){"
            )
            if (!identical(
                  gsub(
                  "[[:space:]]",
                  "",
                  observed
            ),
                  gsub(
                  "[[:space:]]",
                  "",
                  expected
            )
##
## ---- Block -------------------------------------------------------------------------------------------------------------------------
##
            )) {
                  stop(paste0(
                        "The ",
                        function_name,
                        " signature is not the supported reference signature."
                  ))
            }

      }

      fn_guard_signature(
            "matrix",
            "log_prev_for_slice",
            c(
                  "data matrix X_prevalence_s",
                  "data array[] int pop_s",
                  "data int n_class",
              "data int use_OR_prior",
                  "array[] vector prev_simplex",
                  "array[] real mu_A",
              "array[] real sigma_A",
                  "array[] real z_A",
                  "vector beta_prevalence_A",
              "array[] real mu_B",
                  "array[] real sigma_B",
                  "array[] real z_B",
                  "vector beta_prevalence_B",
              "array[] real log_OR_par",
                  "data int log_OR_shared",
                  "data int use_cond_prior"
            )
      )
      fn_guard_signature(
            "real",
            "partial_log_lik",
            c(
                  "array[] row_vector u_raw_slice",
                  "int start",
                  "int end",
                  "data matrix y_use",
              "data array[,] int obs_mask",
                  "data array[,] int y_ord",
                  "data array[,] int y_ord_upper",
              "data array[] matrix X",
                  "data array[] int n_covs_per_outcome",
                  "data array[] int pop",
              "data array[,] int class_D",
                  "data array[] int test_condition",
                  "matrix beta",
              "array[] matrix beta_cov",
                  "array[] matrix L_Omega",
                  "matrix L_Omega_diag_recip",
              "data matrix X_prevalence",
                  "data int use_OR_prior",
                  "array[] vector prev_simplex",
              "array[] real mu_logit_A",
                  "array[] real sigma_logit_A",
                  "array[] real z_logit_A",
              "vector beta_prevalence_A",
                  "array[] real mu_logit_B",
                  "array[] real sigma_logit_B",
              "array[] real z_logit_B",
                  "vector beta_prevalence_B",
                  "array[] real log_OR_par",
              "data int log_OR_shared",
                  "data int use_cond_prior",
                  "array[] vector C_vec_tp",
              "array[] vector log_C_gap_tp",
                  "data array[] int ord_start_index",
                  "data array[] int ord_end_index",
              "data array[] int n_thr_per_ord_test",
                  "data array[] int n_cat_per_ord_test",
              "data int n_binary_tests",
                  "data array[] int latent_test_order",
              "data array[] int latent_position_of_test",
                  "data real C_sentinel",
                  "data int n_tests",
              "data int n_class",
                  "data int n_covariates_max",
                  "data int Phi_type",
              "data real overflow_threshold",
                  "data real underflow_threshold"
            )
      )

      fn_guard_declaration <-  function(
            declaration,
                                          block
##
## ---- Block -------------------------------------------------------------------------------------------------------------------------
##
      ) {

            tokens <-  strsplit(
                  gsub(
                  "[[:space:]]",
                  "",
                  declaration
            ),
                  "",
                  fixed = TRUE
            )[[1]]
            pattern <-  paste0(
                  "(?m)^\\h*",
                  paste0(
                  "\\Q",
                  tokens,
                  "\\E",
                  collapse = "\\s*"
            )
            )
            span <-  fn_one_match(pattern)
            if (span["start"] <= block["opening"] ||
##
## ---- Block -------------------------------------------------------------------------------------------------------------------------
##
                span["start"] + span["length"] - 1L >= block["closing"]) {
                  stop("The mapping declaration is outside the supported block.")
            }

      }

      for (declaration in c(
            "int<lower=1> N;",
            "int<lower=2> n_tests;",
            "int<lower=4, upper=4> n_class;",
            "array[n_class, 2] int<lower=0, upper=1> class_D;",
            "array[n_tests] int<lower=1, upper=2> test_condition;",
            "int<lower=0> n_binary_tests;",
            "int<lower=0> n_ordinal_tests;",
            "array[n_ordinal_tests] int<lower=2> n_cat_per_ord_test;",
            "int n_covariates_max;",
            "array[n_tests] matrix[N, n_covariates_max] X;",
            "matrix[N, n_tests] y;"
##
## ---- Block -------------------------------------------------------------------------------------------------------------------------
##
      )) {
            fn_guard_declaration(
                  declaration,
                  data_block
            )
      }
      for (declaration in c(
            "array[max(n_ordinal_tests, 1)] int ord_start_index;",
            "array[N, n_tests] int<lower=0, upper=1> obs_mask;",
            "matrix[N, n_tests] y_use = y;",
            "array[max(n_ordinal_tests, 1), N] int y_ord;",
            "array[max(n_ordinal_tests, 1), N] int y_ord_upper = y_ord;"
##
## ---- Block -------------------------------------------------------------------------------------------------------------------------
##
      )) {
            fn_guard_declaration(
                  declaration,
                  transformed_data_block
            )
      }

##
## ---- the concentration is data at the sole call of the prior: ----------------------------------------------------------------------
##

      fn_guard_declaration(
            paste0(
                  "array[2] matrix<lower=0>[n_ordinal_tests > 0 ? max(n_cat_per_ord_test) : 1, ",
                   "max(n_ordinal_tests, 1)] prior_dirichlet_alpha;"
            ),
            data_block
      )
      fn_guard_signature(
            "real",
            "induced_dirichlet_lpdf",
            c(
                  "vector C_vec",
                  "vector log_spacing",
                  "vector alpha"
            )
      )
      prior_signature <-  fn_one_match(paste0(
            "(?<![A-Za-z0-9_])real\\s+induced_dirichlet_lpdf\\s*\\([^{};]*\\)\\s*\\{"))
      prior_signature_text <-  substr(
            masked,
            prior_signature["start"],
                                       prior_signature["start"] + prior_signature["length"] - 1L
      )
      alpha_position <-  regexpr(
            "(?<![A-Za-z0-9_])vector\\s+alpha(?=\\s*\\))",
                                   prior_signature_text,
            perl = TRUE,
            useBytes = TRUE
      )[1]
      prior_calls <-  gregexpr(
            "(?<![A-Za-z0-9_])induced_dirichlet_lpdf\\s*\\(",
                                masked,
            perl = TRUE,
            useBytes = TRUE
      )[[1]]
      if (length(prior_calls) != 2L || prior_calls[1] < 1L || alpha_position < 1L ||
          prior_signature["start"] <= functions_block["opening"] ||
##
## ---- Block -------------------------------------------------------------------------------------------------------------------------
##
          prior_signature["start"] + prior_signature["length"] >= functions_block["closing"]) {
            stop("The concentration signature or its call count is not the supported reference.")
      }
      prior_call <-  fn_one_match(paste0(
            "target\\s*\\+=\\s*induced_dirichlet_lpdf\\s*\\(\\s*C_t\\s*\\|\\s*",
            "get_test_values\\s*\\(\\s*log_C_gap_tp\\s*\\[\\s*s\\s*\\]\\s*,\\s*",
            "ord_start_index\\s*,\\s*ord_end_index\\s*,\\s*t_ord\\s*\\)\\s*,\\s*",
            "to_vector\\s*\\(\\s*prior_dirichlet_alpha\\s*\\[\\s*s\\s*\\]\\s*",
            "\\[\\s*1\\s*:\\s*n_cat_t\\s*,\\s*t_ord\\s*\\]\\s*\\)\\s*\\)\\s*;"
      ))
      if (prior_call["start"] <= model_block["opening"] ||
##
## ---- Block -------------------------------------------------------------------------------------------------------------------------
##
          prior_call["start"] + prior_call["length"] >= model_block["closing"]) {
            stop("The data concentration call is outside the supported model block.")
      }
      edits <-  list(list(
            span = c(
                  start = prior_signature["start"] + alpha_position - 1L,
                  length = 0
            ),
            text = "data "
      ))

      nuisance <-  fn_one_match(paste0(
            "array\\s*\\[\\s*n_u_rows\\s*\\]\\s*row_vector\\s*",
                                        "\\[\\s*n_tests\\s*\\]\\s*u_raw\\s*;"
      ))
      if (nuisance["start"] <= parameters_block["opening"] ||
##
## ---- Block -------------------------------------------------------------------------------------------------------------------------
##
          nuisance["start"] >= parameters_block["closing"]) {
            stop("The nuisance block is not the reference's array of rows.")
      }

##
## ---- replaces only the two original GHK sum calls: ---------------------------------------------------------------------------------
##

      old_arguments <-  c(
            "chunk_size",
            "y_use",
            "obs_mask",
            "y_ord",
            "y_ord_upper",
            "X",
            "n_covs_per_outcome",
            "pop",
            "class_D",
            "test_condition",
            "beta",
            "beta_cov_by_own_status",
            "L_Omega",
            "L_Omega_diag_recip",
            "X_prevalence",
            "use_OR_prior",
            "prev_simplex",
            "mu_logit_A",
            "sigma_logit_A",
            "z_logit_A",
            "beta_prevalence_A",
            "mu_logit_B",
            "sigma_logit_B",
            "z_logit_B",
            "beta_prevalence_B",
            "log_OR_par",
            "log_OR_shared",
            "use_cond_prior",
            "C_vec_tp",
            "log_C_gap_tp",
            "ord_start_index",
            "ord_end_index",
            "n_thr_per_ord_test_pad",
            "n_cat_per_ord_test_pad",
            "n_binary_tests",
            "latent_test_order",
            "latent_position_of_test",
            "C_sentinel",
            "n_tests",
            "n_class",
            "n_covariates_max",
            "Phi_type",
            "overflow_threshold",
            "underflow_threshold"
      )
      new_arguments <-  c(
            "chunk_size",
            "category_of_subject_and_test",
            "upper_category_of_subject_and_test",
            "covariates_of_each_set_and_test",
            "n_covs_per_outcome",
            "pop",
            "beta",
            "beta_cov_by_own_status",
            "L_Omega",
            "X_prevalence",
            "use_OR_prior",
            "prev_simplex",
            "mu_logit_A",
            "sigma_logit_A",
            "z_logit_A",
            "beta_prevalence_A",
            "mu_logit_B",
            "sigma_logit_B",
            "z_logit_B",
            "beta_prevalence_B",
            "log_OR_par",
            "log_OR_shared",
            "use_cond_prior",
            "C_vec_tp",
            "own_status_set_of_class_and_test",
            "position_of_first_cutpoint_of_test",
            "test_is_binary_of_test",
            "n_categories_of_test",
            "latent_test_order",
            "n_tests",
            "n_class",
            "n_covariates_max",
            "Phi_type",
            "overflow_threshold",
            "underflow_threshold"
      )
      for (nuisance_name in c(
            "u_raw_CI",
            "u_raw"
##
## ---- Block -------------------------------------------------------------------------------------------------------------------------
##
      )) {
            expected <-  paste0(
                  "target\\s*\\+=\\s*reduce_sum_static\\s*\\(\\s*partial_log_lik\\s*,\\s*",
                                 nuisance_name,
                  "\\s*,\\s*",
                  paste(
                                       old_arguments,
                                       collapse = "\\s*,\\s*"
                                 ),
                                 "\\s*\\)\\s*;"
            )
            span <-  fn_one_match(expected)
##
## ---- Conditions --------------------------------------------------------------------------------------------------------------------
##
            if (span["start"] <= model_block["opening"] || span["start"] >= model_block["closing"]) {
                  stop("A GHK sum is not in the model block.")
            }
            replacement <-  paste0(
                  "target += reduce_sum_static(partial_log_lik_by_the_fused_GHK_mixture_log_likelihood,\n",
                  "                               ",
                  nuisance_name,
                  ",\n",
                  paste0(
                        "                               ",
                        new_arguments,
                        collapse = ",\n"
                  ),
                  ");"
            )
##
## ---- Conditions --------------------------------------------------------------------------------------------------------------------
##
            if (fuse_likelihood) {
                  edits[[length(edits) + 1L]] <-  list(
                        span = span,
                        text = replacement
                  )
            }
      }

##
## ---- the existing tiled GHK declaration and slice wrapper: -------------------------------------------------------------------------
##

      wrapper <-  c(
            "",
            paste0(
                  "      real nico_finite_mixture_of_multivariate_ordered_probit_GHK_",
                   "log_likelihood_in_tiles_of_subjects("
            ),
            "            matrix u_raw_of_each_test_and_subject,",
            "            data int first_subject,",
            "            data int nuisance_transformation,",
            "            matrix intercept,",
            "            array[] matrix coefficients,",
            "            data array[,] int coefficient_set_of_class_and_test,",
            "            data array[,] matrix covariates,",
            "            data array[,] int n_covariates,",
            "            array[] vector cutpoints,",
            "            data array[,] int cutpoint_set_of_class_and_test,",
            "            data array[] int position_of_first_cutpoint_of_test,",
            "            array[] matrix L_Omega,",
            "            matrix log_class_probability,",
            "            data array[,] int category,",
            "            data array[,] int upper_category,",
            "            data array[] int test_is_binary,",
            "            data array[] int n_categories_of_test,",
            "            data array[] int test_at_GHK_position,",
            "            data int Phi_type,",
            "            data real overflow_threshold,",
            "            data real underflow_threshold);",
            "",
            "      real partial_log_lik_by_the_fused_GHK_mixture_log_likelihood(",
            "            array[] row_vector u_raw_slice,",
            "            int start,",
            "            int end,",
            "            data array[,] int category_of_subject_and_test,",
            "            data array[,] int upper_category_of_subject_and_test,",
            "            data array[,] matrix covariates_of_each_set_and_test,",
            "            data array[] int n_covs_per_outcome,",
            "            data array[] int pop,",
            "            matrix beta,",
            "            array[] matrix beta_cov,",
            "            array[] matrix L_Omega,",
            "            data matrix X_prevalence,",
            "            data int use_OR_prior,",
            "            array[] vector prev_simplex,",
            "            array[] real mu_logit_A,",
            "            array[] real sigma_logit_A,",
            "            array[] real z_logit_A,",
            "            vector beta_prevalence_A,",
            "            array[] real mu_logit_B,",
            "            array[] real sigma_logit_B,",
            "            array[] real z_logit_B,",
            "            vector beta_prevalence_B,",
            "            array[] real log_OR_par,",
            "            data int log_OR_shared,",
            "            data int use_cond_prior,",
            "            array[] vector C_vec_tp,",
            "            data array[,] int own_status_set_of_class_and_test,",
            "            data array[] int position_of_first_cutpoint_of_test,",
            "            data array[] int test_is_binary_of_test,",
            "            data array[] int n_categories_of_test,",
            "            data array[] int latent_test_order,",
            "            data int n_tests,",
            "            data int n_class,",
            "            data int n_covariates_max,",
            "            data int Phi_type,",
            "            data real overflow_threshold,",
            "            data real underflow_threshold) {",
            "",
            "            int M = end - start + 1;",
            "            matrix[M, n_class] log_prev = log_prev_for_slice(",
            "                  X_prevalence[start:end, ],",
            "                  pop[start:end],",
            "                  n_class,",
            "                  use_OR_prior,",
            "                  prev_simplex,",
            "                  mu_logit_A,",
            "                  sigma_logit_A,",
            "                  z_logit_A,",
            "                  beta_prevalence_A,",
            "                  mu_logit_B,",
            "                  sigma_logit_B,",
            "                  z_logit_B,",
            "                  beta_prevalence_B,",
            "                  log_OR_par,",
            "                  log_OR_shared,",
            "                  use_cond_prior);",
            "            int n_covariate_sets = (n_covariates_max > 0) ? 1 : 0;",
            "            array[n_covariate_sets, n_tests] int n_covariates_s;",
            "            for (s in 1:n_covariate_sets) {",
            "                  n_covariates_s[s] = n_covs_per_outcome;",
            "            }",
            "            matrix[n_tests, M] u_raw_of_each_test_and_subject = to_matrix(u_raw_slice)';",
            "            return",
            paste0(
                  "                  nico_finite_mixture_of_multivariate_ordered_probit_GHK_",
                   "log_likelihood_in_tiles_of_subjects("
            ),
            "                  u_raw_of_each_test_and_subject,",
            "                  start,",
            "                  2,",
            "                  beta,",
            "                  beta_cov,",
            "                  own_status_set_of_class_and_test,",
            "                  covariates_of_each_set_and_test,",
            "                  n_covariates_s,",
            "                  C_vec_tp,",
            "                  own_status_set_of_class_and_test,",
            "                  position_of_first_cutpoint_of_test,",
            "                  L_Omega,",
            "                  log_prev,",
            "                  category_of_subject_and_test,",
            "                  upper_category_of_subject_and_test,",
            "                  test_is_binary_of_test,",
            "                  n_categories_of_test,",
            "                  latent_test_order,",
            "                  Phi_type,",
            "                  overflow_threshold,",
            "                  underflow_threshold);",
            "",
            "      }",
            "",
            ""
      )
##
## ---- Conditions --------------------------------------------------------------------------------------------------------------------
##
      if (fuse_likelihood) {
            edits[[length(edits) + 1L]] <-  list(
                  span = c(
                        start = functions_block["closing"],
                        length = 0
                  ),
                  text = paste(
                        wrapper,
                        collapse = "\n"
                  )
            )
      }

##
## ---- the existing data mappings, without a full nuisance copy: ---------------------------------------------------------------------
##

      covariates <-  c(
            "",
            "      int n_covariate_sets = (n_covariates_max > 0) ? 1 : 0;",
            "      array[n_covariate_sets, n_tests] matrix[N, n_covariates_max] covariates_of_each_set_and_test;",
            "      for (covariate_set in 1:n_covariate_sets) {",
            "            for (t in 1:n_tests) {",
            "                  covariates_of_each_set_and_test[covariate_set, t] = X[t];",
            "            }",
            "      }",
            ""
      )
##
## ---- Conditions --------------------------------------------------------------------------------------------------------------------
##
      if (fuse_likelihood) {
            edits[[length(edits) + 1L]] <-  list(
                  span = c(
                        start = transformed_data_block["opening"] + 1L,
                        length = 0
                  ),
                  text = paste(
                        covariates,
                        collapse = "\n"
                  )
            )
      }
      mapping_anchor <-  fn_one_match(paste0(
            "(?<![A-Za-z0-9_])array\\s*\\[\\s*n_class\\s*\\]\\s*",
                                              "matrix\\s*\\[\\s*n_tests\\s*,\\s*n_tests\\s*\\]\\s*",
                                              "lb_corr_actual\\s*=\\s*lb_corr\\s*;"
      ))
      if (mapping_anchor["start"] <= transformed_data_block["opening"] ||
##
## ---- Block -------------------------------------------------------------------------------------------------------------------------
##
          mapping_anchor["start"] >= transformed_data_block["closing"]) {
            stop("The correlation data mapping is outside transformed data.")
      }
      mapping <-  c(
            "array[N, n_tests] int category_of_subject_and_test;",
            "      array[N, n_tests] int upper_category_of_subject_and_test;",
            "      array[n_tests] int test_is_binary_of_test = rep_array(0, n_tests);",
            "      array[n_tests] int n_categories_of_test = rep_array(2, n_tests);",
            "      array[n_tests] int position_of_first_cutpoint_of_test = rep_array(1, n_tests);",
            "      array[n_class, n_tests] int own_status_set_of_class_and_test;",
            "      for (t in 1:n_tests) {",
            "            vector[N] observed = to_vector(obs_mask[, t]);",
            "            if (t <= n_binary_tests) {",
            "                  test_is_binary_of_test[t] = 1;",
            "                  category_of_subject_and_test[, t]",
            "                        = to_int(to_array_1d((y_use[, t] + 1.0) .* observed - 1.0));",
            "                  upper_category_of_subject_and_test[, t] = category_of_subject_and_test[, t];",
            "            } else {",
            "                  int t_ord = t - n_binary_tests;",
            "                  n_categories_of_test[t] = n_cat_per_ord_test[t_ord];",
            "                  position_of_first_cutpoint_of_test[t] = ord_start_index[t_ord];",
            "                  category_of_subject_and_test[, t]",
            "                        = to_int(to_array_1d((to_vector(y_ord[t_ord]) + 1.0) .* observed - 1.0));",
            "                  upper_category_of_subject_and_test[, t] = y_ord_upper[t_ord];",
            "            }",
            "            for (c in 1:n_class) {",
            "                  own_status_set_of_class_and_test[c, t] = class_D[c, test_condition[t]] + 1;",
            "            }",
            "      }",
            "      "
      )
##
## ---- Conditions --------------------------------------------------------------------------------------------------------------------
##
      if (fuse_likelihood) {
            edits[[length(edits) + 1L]] <-  list(
                  span = c(
                        start = mapping_anchor["start"],
                        length = 0
                  ),
                  text = paste(
                        mapping,
                        collapse = "\n"
                  )
            )
      }

##
## ---- applies the byte spans and writes without replacing an existing file: ---------------------------------------------------------
##

      line_ending <-  if (grepl(
            "\r\n",
            rawToChar(source_bytes),
            fixed = TRUE
      )) "\r\n" else "\n"
      order_of_edits <-  order(
            vapply(
            edits,
##
## ---- Function ----------------------------------------------------------------------------------------------------------------------
##
            function(edit) {

            unname(edit$span[1])

      },
            numeric(1)
      ),
            decreasing = TRUE
      )
      output_bytes <-  source_bytes
##
## ---- Loop --------------------------------------------------------------------------------------------------------------------------
##
      for (index in order_of_edits) {
            edit <-  edits[[index]]
            first <-  unname(edit$span[1])
            count <-  unname(edit$span[2])
            before <-  if (first > 1L) output_bytes[seq_len(first - 1L)] else raw(0)
##
## ---- Conditions --------------------------------------------------------------------------------------------------------------------
##
            after <-  if (first + count <= length(output_bytes)) {
                  output_bytes[seq.int(
                        first + count,
                        length(output_bytes)
                  )]
            } else raw(0)
            inserted <-  gsub(
                  "\n",
                  line_ending,
                  edit$text,
                  fixed = TRUE
            )
            output_bytes <-  c(
                  before,
                  charToRaw(inserted),
                  after
            )
      }
      temporary_path <-  tempfile(
            pattern = ".NicoSIMD_fused_",
            tmpdir = output_directory
      )
      output_connection <-  file(
            temporary_path,
            open = "wb"
      )
      on.exit(
##
## ---- Block -------------------------------------------------------------------------------------------------------------------------
##
            {
            try(
                  close(output_connection),
                  silent = TRUE
            )
            if (file.exists(temporary_path)) unlink(temporary_path)
      },
            add = TRUE
      )
      writeBin(
            output_bytes,
            output_connection
      )
      close(output_connection)
      if (!file.link(
            temporary_path,
            output_path
##
## ---- Block -------------------------------------------------------------------------------------------------------------------------
##
      )) {
            stop("The generated source could not be written without replacing a file.")
      }
      return(list(
            NicoSIMD_Stan_model_file_path = output_path,
##
## ---- Conditions --------------------------------------------------------------------------------------------------------------------
##
            fused_calls = if (fuse_likelihood) {
                  paste0(
                        "nico_finite_mixture_of_multivariate_ordered_probit_GHK_",
                         "log_likelihood_in_tiles_of_subjects"
                  )
##
## ---- Conditions --------------------------------------------------------------------------------------------------------------------
##
            } else {
                  character(0)
            },
            source_for_remaining_backend_conversion = output_path
      ))

}

fn_write_NicoSIMD_fused_four_class_LC_MVOP_Stan_model <-  function(
      source_Stan_model_file_path,
      NicoSIMD_fused_Stan_model_file_path
##
## ---- Block -------------------------------------------------------------------------------------------------------------------------
##
) {

      result <-  fn_write_NicoSIMD_four_class_LC_MVOP_Stan_model(
            source_Stan_model_file_path = source_Stan_model_file_path,
            NicoSIMD_Stan_model_file_path = NicoSIMD_fused_Stan_model_file_path,
            fuse_likelihood = TRUE
      )
      return(list(
            NicoSIMD_fused_Stan_model_file_path = result$NicoSIMD_Stan_model_file_path,
            fused_calls = result$fused_calls,
            source_for_remaining_backend_conversion = result$source_for_remaining_backend_conversion
      ))

}

fn_write_NicoSIMD_non_fused_four_class_LC_MVOP_Stan_model <-  function(
      source_Stan_model_file_path,
      NicoSIMD_Stan_model_file_path
##
## ---- Block -------------------------------------------------------------------------------------------------------------------------
##
) {

      return(fn_write_NicoSIMD_four_class_LC_MVOP_Stan_model(
            source_Stan_model_file_path = source_Stan_model_file_path,
            NicoSIMD_Stan_model_file_path = NicoSIMD_Stan_model_file_path,
            fuse_likelihood = FALSE
      ))

}






















