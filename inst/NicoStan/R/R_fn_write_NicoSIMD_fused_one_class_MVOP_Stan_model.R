##
## ---- Writes the one-class MVOP native sources from the Stan source -----------------------------------------------------------------
##

fn_write_NicoSIMD_one_class_MVOP_Stan_model <-  function(
      source_Stan_model_file_path,
      NicoSIMD_Stan_model_file_path,
      fuse_likelihood
) {

      ##
      ## ---- Checks the likelihood category ------------------------------------------------------------------------------------------
      ##
      if (!is.logical(fuse_likelihood) || length(fuse_likelihood) != 1L || is.na(fuse_likelihood)) {
            stop("The likelihood category must be supplied explicitly.")
      }
      source_path <-  normalizePath(source_Stan_model_file_path, mustWork = TRUE)
      output_directory <-  normalizePath(dirname(NicoSIMD_Stan_model_file_path), mustWork = TRUE)
      output_path <-  file.path(output_directory, basename(NicoSIMD_Stan_model_file_path))
      source_name <-  "MVOP_PartialLog_v2_reduce_sum_one_class_Stan_math_for_layout_timings.stan"
      ##
      ## ---- Checks the source is the Stan reference ---------------------------------------------------------------------------------
      ##
      if (!identical(basename(source_path), source_name)) {
            stop("The one-class MVOP source is not the supported Stan reference.")
      }
      ##
      ## ---- Checks the output is a new file -----------------------------------------------------------------------------------------
      ##
      if (identical(source_path, output_path) || file.exists(output_path)) {
            stop("The output must be a new file, not the source.")
      }
      source_lines <-  readLines(source_path, warn = FALSE)
      if (length(source_lines) == 0) stop("The source is empty.")
      ##
      ## ---- Checks the reference has no native calls --------------------------------------------------------------------------------
      ##
      if (any(grepl("nico_", sub("//.*$", "", source_lines), fixed = TRUE))) {
            stop("The reference already contains native backend calls.")
      }
      ##
      ## ---- Finds one line of the source --------------------------------------------------------------------------------------------
      ##
      fn_one_line <-  function(line) {

            position <-  which(source_lines == line)
            if (length(position) != 1L) stop(paste0("The source line is missing or ambiguous: ", line))
            return(position)

      }

      ##
      ## ---- The concentration is data at the sole call of the prior -----------------------------------------------------------------
      ##

      alpha_line <-  fn_one_line("                                   vector alpha, ")
      code_lines <-  sub("//.*$", "", source_lines)
      prior_calls <-  grep("induced_dirichlet_lpdf\\s*\\(", code_lines, perl = TRUE)
      prior_calls <-  prior_calls[!grepl("real\\s+induced_dirichlet_lpdf", code_lines[prior_calls], perl = TRUE)]
      ##
      ## ---- Checks the sole call of the prior ---------------------------------------------------------------------------------------
      ##
      if (length(prior_calls) != 1L ||
          !grepl("to_vector(prior_dirichlet_alpha[c][1:n_cat_t, t_ord])", code_lines[prior_calls],
                 fixed = TRUE) ||
          !grepl("real induced_dirichlet_lpdf(", source_lines[alpha_line - 2L], fixed = TRUE)) {
            stop("The concentration signature or its call count is not the supported reference.")
      }
      source_lines[alpha_line] <-  "                                   data vector alpha, "

      ##
      ## ---- The fused GHK log likelihood, copied from the hand-written fused GHK version --------------------------------------------
      ##

      GHK_declaration <-  c(
            "    real nico_finite_mixture_of_multivariate_ordered_probit_GHK_log_likelihood(",
            "          array[] row_vector u_raw, data int nuisance_transformation, matrix intercept,",
            "          array[] matrix coefficients, data array[,] int coefficient_set_of_class_and_test,",
            paste0("          data array[,] matrix covariates, data array[,] int n_covariates, array[] vector ",
                   "cutpoints,"),
            "          data array[,] int cutpoint_set_of_class_and_test,",
            "          data array[] int position_of_first_cutpoint_of_test,",
            "          array[] matrix L_Omega, matrix log_class_probability, data array[,] int category,",
            "          data array[,] int upper_category, data array[] int test_is_binary,",
            paste0("          data array[] int n_categories_of_test, data array[] int test_at_GHK_position, da",
                   "ta int Phi_type,"),
            "          data real overflow_threshold, data real underflow_threshold);",
            paste0("    real nico_finite_mixture_of_multivariate_ordered_probit_GHK_log_likelihood_in_tiles_of_",
                   "subjects("),
            paste0("          matrix u_raw_of_each_test_and_subject, data int first_subject, data int nuisance_t",
                   "ransformation,"),
            paste0("          matrix intercept, array[] matrix coefficients, data array[,] int coefficient_set_o",
                   "f_class_and_test,"),
            paste0("          data array[,] matrix covariates, data array[,] int n_covariates, array[] vector ",
                   "cutpoints,"),
            "          data array[,] int cutpoint_set_of_class_and_test,",
            "          data array[] int position_of_first_cutpoint_of_test,",
            "          array[] matrix L_Omega, matrix log_class_probability, data array[,] int category,",
            "          data array[,] int upper_category, data array[] int test_is_binary,",
            paste0("          data array[] int n_categories_of_test, data array[] int test_at_GHK_position, da",
                   "ta int Phi_type,"),
            "          data real overflow_threshold, data real underflow_threshold);")

      GHK_partial_sum_function <-  c(
            "      ////",
            paste0("      //// ---- The partial sum of reduce_sum_static by BayesMVP's fused GHK log likelihoo",
                   "d of a finite mixture of"),
            paste0("      //// multivariate ordered probit models (stan_external_finite_mixture_of_multivariat",
                   "e_ordered_probit_GHK_"),
            paste0("      //// log_likelihood.hpp): the same log likelihood of the slice's subjects, and the s",
                   "ame log Jacobian of"),
            paste0("      //// u = Phi(u_raw), as partial_log_lik() above, in one autodiff node per slice (its",
                   " value and gradient made"),
            paste0("      //// in BayesMVP's SIMD kernels). The covariates of class 1 are X_nd and those of cl",
                   "ass 2 X_d (one covariate"),
            paste0("      //// set per class, n_covariates_max > 1), otherwise the intercepts alone; the cutpo",
                   "ints of class c are"),
            "      //// C_vec[c]; the GHK order is the data order.",
            "      ////",
            "      real partial_log_lik_by_the_fused_GHK_mixture_log_likelihood( array[] row_vector u_raw_slice,",
            "                            int start,",
            "                            int end,",
            "                            data array[,] int category_of_subject_and_test,",
            "                            // data array[] matrix X_nd,",
            "                            // data array[] matrix X_d,",
            "                            data array[,] matrix covariates_of_each_class_and_test,",
            "                            data array[,] int n_covs_per_outcome,",
            "                            data array[] int pop,",
            "                            matrix intercept_of_each_class_and_test,",
            "                            array[] matrix coefficients_of_each_class,",
            "                            array[] matrix L_Omega,",
            "                            vector prev,",
            "                            array[] vector C_vec,",
            "                            data array[,] int set_of_each_class_and_test,",
            "                            data array[] int position_of_first_cutpoint_of_test,",
            "                            data array[] int test_is_binary,",
            "                            data array[] int n_categories_of_test,",
            "                            data array[] int test_at_GHK_position,",
            "                            data int n_tests,",
            "                            data int n_class,",
            "                            data int n_covariates_max,",
            "                            data int Phi_type,",
            "                            data real overflow_threshold,",
            "                            data real underflow_threshold) {",
            "",
            "            int M = end - start + 1;",
            "            ////",
            paste0("            //// ---- the log class probabilities of the slice's subjects, as partial_log_",
                   "lik() (one class: 0)"),
            "            ////",
            "            matrix[M, n_class] log_prev = rep_matrix(0.0, M, n_class);",
            "            if (n_class == 2) {",
            "                  array[M] int pop_s = pop[start:end];",
            "                  vector[num_elements(prev)] log1m_prev_of_population = log1m(prev);",
            "                  vector[num_elements(prev)] log_prev_of_population = log(prev);",
            "                  log_prev[, 1] = log1m_prev_of_population[pop_s];",
            "                  log_prev[, 2] = log_prev_of_population[pop_s];",
            "            }",
            "            ////",
            paste0("            //// ---- the covariates of the slice's subjects: the rows start:end of covari",
                   "ates_of_each_class_and_test"),
            paste0("            //// (made in transformed data), indexed in the call itself, so that they stay",
                   " data (a local variable of"),
            "            //// a function is not data, which the fused function's covariates must be)",
            "            ////",
            "            // int n_covariate_sets = (n_covariates_max > 1) ? n_class : 0;",
            "            // array[n_covariate_sets, n_tests] matrix[M, n_covariates_max] covariates_s;",
            "            // for (s in 1:n_covariate_sets) {",
            "            //       for (t in 1:n_tests) {",
            "            //             covariates_s[s, t] = rep_matrix(0.0, M, n_covariates_max);",
            paste0("            //             if (s == 1)  covariates_s[s, t][, 1:cols(X_nd[t])] = X_nd[t][st",
                   "art:end, ];"),
            "            //             else         covariates_s[s, t][, 1:cols(X_d[t])] = X_d[t][start:end, ];",
            "            //       }",
            "            // }",
            "            ////",
            paste0("            //// ---- (the version in tiles of subjects: the nuisance values of the slice ",
                   "as one T x M matrix,"),
            paste0("            //// and the full data arrays read at the rows start, ..., end by the tiled fu",
                   "sed function, so that"),
            paste0("            //// no slice of the data is copied; the call of the function above is kept be",
                   "low as comments)"),
            "            ////",
            "            // array[M, n_tests] int category_s = category_of_subject_and_test[start:end];",
            "            // return nico_finite_mixture_of_multivariate_ordered_probit_GHK_log_likelihood(",
            paste0("            //              u_raw_slice, 1, intercept_of_each_class_and_test, coefficient",
                   "s_of_each_class,"),
            paste0("            //              // set_of_each_class_and_test, covariates_s, n_covs_per_outco",
                   "me, C_vec,"),
            paste0("            //              set_of_each_class_and_test, covariates_of_each_class_and_test",
                   "[:, :, start:end],"),
            "            //              n_covs_per_outcome, C_vec,",
            paste0("            //              set_of_each_class_and_test, position_of_first_cutpoint_of_tes",
                   "t, L_Omega, log_prev,"),
            paste0("            //              category_s, category_s, test_is_binary, n_categories_of_test,",
                   " test_at_GHK_position,"),
            "            //              Phi_type, overflow_threshold, underflow_threshold);",
            "            matrix[n_tests, M] u_raw_of_each_test_and_subject = to_matrix(u_raw_slice)';",
            paste0("            return nico_finite_mixture_of_multivariate_ordered_probit_GHK_log_likelihood_",
                   "in_tiles_of_subjects("),
            paste0("                         u_raw_of_each_test_and_subject, start, 1, intercept_of_each_class_",
                   "and_test,"),
            "                         coefficients_of_each_class, set_of_each_class_and_test,",
            "                         covariates_of_each_class_and_test, n_covs_per_outcome, C_vec,",
            paste0("                         set_of_each_class_and_test, position_of_first_cutpoint_of_test, L",
                   "_Omega, log_prev,"),
            paste0("                         category_of_subject_and_test, category_of_subject_and_test, test_",
                   "is_binary,"),
            "                         n_categories_of_test, test_at_GHK_position, Phi_type, overflow_threshold,",
            "                         underflow_threshold);",
            "",
            "      }",
            "")

      binary_check_on_the_observed_results <-  c(
            "              // if (max(abs(y[, t] .* (y[, t] - 1.0))) > 0) column_t_is_binary = 0;",
            paste0("              //// (the same check on the observed results only: a missing result, y < 0,",
                   " counts as 0;"),
            "              ////  fmax(y, 0) is y for every observed result and 0 for a missing one)",
            paste0("              if (max(abs(fmax(y[, t], 0.0) .* (fmax(y[, t], 0.0) - 1.0))) > 0) column_t",
                   "_is_binary = 0;"))

      covariates_of_each_class_and_test <-  c(
            paste0("      //// ---- The covariates of each class and test, for the fused GHK log likelihood: o",
                   "ne set per class (X_nd,"),
            paste0("      //// X_d) when there are covariates (n_covariates_max > 1), each in the first column",
                   "s of a matrix of"),
            "      //// n_covariates_max columns (zeros after them); each slice reads its own rows of it",
            "      ////",
            "      int n_covariate_sets = (n_covariates_max > 1) ? n_class : 0;",
            paste0("      array[n_covariate_sets, n_tests] matrix[N, n_covariates_max] covariates_of_each_clas",
                   "s_and_test;"),
            "      for (covariate_set in 1:n_covariate_sets) {",
            "            for (t in 1:n_tests) {",
            paste0("                  covariates_of_each_class_and_test[covariate_set, t] = rep_matrix(0.0, N,",
                   " n_covariates_max);"),
            "                  if (covariate_set == 1) {",
            paste0("                        covariates_of_each_class_and_test[covariate_set, t][, 1:cols(X_nd[",
                   "t])] = X_nd[t];"),
            "                  } else {",
            paste0("                        covariates_of_each_class_and_test[covariate_set, t][, 1:cols(X_d[t",
                   "])] = X_d[t];"),
            "                  }",
            "            }",
            "      }",
            "      ////")

      data_of_the_GHK_likelihood <-  c(
            "      ////",
            paste0("      //// ---- The data of BayesMVP's fused GHK likelihood: the result of each subject an",
                   "d test (binary 0 / 1,"),
            paste0("      //// ordinal 1, ..., K_t), the categories and the position of the first cutpoint of ",
                   "each test in C_vec[c],"),
            paste0("      //// the set of each class and test (class c: its own covariates and cutpoints) and ",
                   "the GHK order (the data"),
            "      //// order)",
            "      ////",
            "      array[N, n_tests] int category_of_subject_and_test;",
            "      array[n_tests] int n_categories_of_test = rep_array(2, n_tests);",
            "      array[n_tests] int position_of_first_cutpoint_of_test = rep_array(1, n_tests);",
            "      array[n_class, n_tests] int set_of_each_class_and_test;",
            "      array[n_tests] int test_at_GHK_position = linspaced_int_array(n_tests, 1, n_tests);",
            paste0("      //// A MISSING result (y < 0 in the data, use -1) stays negative in category_of_subj",
                   "ect_and_test (to_int()"),
            paste0("      //// of y for a binary test, y_ord for an ordinal one): the fused function takes any",
                   " negative category as"),
            paste0("      //// missing, by the GHK method (the untruncated draw Z = F^{-1}(u) in the condition",
                   "ing chain, and no"),
            "      //// likelihood term).",
            "      for (t in 1:n_tests) {",
            "            if (test_is_binary[t] == 1) {",
            "                  category_of_subject_and_test[, t] = to_int(to_array_1d(y[, t]));",
            "            } else {",
            "                  category_of_subject_and_test[, t] = y_ord[ordinal_slot_of_test[t]];",
            "                  n_categories_of_test[t] = n_cat_per_ord_test[ordinal_slot_of_test[t]];",
            "                  position_of_first_cutpoint_of_test[t] = ord_start_index[ordinal_slot_of_test[t]];",
            "            }",
            "      }",
            "      for (c in 1:n_class) set_of_each_class_and_test[c] = rep_array(c, n_tests);")

      GHK_sum <-  c(
            paste0("          //// (the GHK likelihood of each slice by BayesMVP's fused GHK log likelihood; t",
                   "he reduce_sum_static call"),
            "          ////  of partial_log_lik() is kept below as comments)",
            paste0("          matrix[n_class, n_tests] intercept_of_each_class_and_test = rep_matrix(0.0, n_cl",
                   "ass, n_tests);"),
            "          array[n_class] matrix[n_covariates_max, n_tests] coefficients_of_each_class;",
            "          for (c in 1:n_class) {",
            "                coefficients_of_each_class[c] = rep_matrix(0.0, n_covariates_max, n_tests);",
            "                for (t in 1:n_tests) {",
            "                      if (n_covariates_max > 1) {",
            "                            coefficients_of_each_class[c][1:n_covs_per_outcome[c, t], t]",
            "                                  = to_vector(beta[c, t, 1:n_covs_per_outcome[c, t]]);",
            "                      } else {",
            "                            intercept_of_each_class_and_test[c, t] = beta[c, t, 1];",
            "                      }",
            "                }",
            "          }",
            "          target += reduce_sum_static(partial_log_lik_by_the_fused_GHK_mixture_log_likelihood,",
            "                               u_raw,",
            "                               chunk_size,",
            "                               category_of_subject_and_test,",
            "                               // X_nd,",
            "                               // X_d,",
            "                               covariates_of_each_class_and_test,",
            "                               n_covs_per_outcome,",
            "                               pop,",
            "                               intercept_of_each_class_and_test,",
            "                               coefficients_of_each_class,",
            "                               L_Omega,",
            "                               prev,",
            "                               C_vec,",
            "                               set_of_each_class_and_test,",
            "                               position_of_first_cutpoint_of_test,",
            "                               test_is_binary,",
            "                               n_categories_of_test,",
            "                               test_at_GHK_position,",
            "                               n_tests,",
            "                               n_class,",
            "                               n_covariates_max,",
            "                               Phi_type,",
            "                               overflow_threshold,",
            "                               underflow_threshold);",
            "//        target += reduce_sum_static(partial_log_lik,",
            "//                             u_raw,",
            "//                             chunk_size,",
            "//                             y,",
            "//                             y_ord,",
            "//                             X_nd,",
            "//                             X_d,",
            "//                             n_covs_per_outcome,",
            "//                             pop,",
            "//                             beta,",
            "//                             L_Omega,",
            "//                             L_Omega_diag_recip,",
            "//                             prev,",
            "//                             C_vec,",
            "//                             ord_start_index,",
            "//                             ord_end_index,",
            "//                             n_thr_per_ord_test_pad,",
            "//                             n_cat_per_ord_test_pad,",
            "//                             n_binary_tests,",
            "//                             test_is_binary,",
            "//                             ordinal_slot_of_test,",
            "//                             C_sentinel,",
            "//                             n_tests,",
            "//                             n_class,",
            "//                             n_covariates_max,",
            "//                             Phi_type,",
            "//                             overflow_threshold,",
            "//                             underflow_threshold);")

      ##
      ## ---- Inserts the fused GHK code ----------------------------------------------------------------------------------------------
      ##
      if (fuse_likelihood) {
            functions_line <-  fn_one_line("functions {")
            data_line <-  fn_one_line("data {")
            closing_of_functions <-  max(which(source_lines[seq_len(data_line - 1L)] == "}"))
            binary_check_line <-  fn_one_line(
                  "              if (max(abs(y[, t] .* (y[, t] - 1.0))) > 0) column_t_is_binary = 0;")
            source_lines <-  c(source_lines[seq_len(binary_check_line - 1L)],
                               binary_check_on_the_observed_results,
                               source_lines[(binary_check_line + 1L):length(source_lines)])
            covariates_anchor <-  fn_one_line("      real s = 1 / 1.702;")
            corrs_line <-  fn_one_line("      int n_corrs = n_class*n_corrs_per_class;")
            sum_start <-  fn_one_line("          target += reduce_sum_static(partial_log_lik,")
            sum_end <-  sum_start - 1L + which(trimws(source_lines[sum_start:length(source_lines)]) ==
                                                 "underflow_threshold);")[1]
            stopifnot(functions_line < closing_of_functions, closing_of_functions < data_line,
                      data_line < covariates_anchor, covariates_anchor < corrs_line, corrs_line < sum_start,
                      !is.na(sum_end))
            source_lines <-  c(source_lines[seq_len(functions_line)],
                               GHK_declaration,
                               source_lines[(functions_line + 1L):(closing_of_functions - 1L)],
                               GHK_partial_sum_function,
                               source_lines[closing_of_functions:(covariates_anchor - 1L)],
                               covariates_of_each_class_and_test,
                               source_lines[covariates_anchor:corrs_line],
                               data_of_the_GHK_likelihood,
                               source_lines[(corrs_line + 1L):(sum_start - 1L)],
                               GHK_sum,
                               source_lines[(sum_end + 1L):length(source_lines)])
      }

      ##
      ## ---- Writes without replacing an existing file -------------------------------------------------------------------------------
      ##

      temporary_path <-  tempfile(pattern = ".NicoSIMD_one_class_", tmpdir = output_directory)
      ##
      ## ---- Removes the temporary file on exit --------------------------------------------------------------------------------------
      ##
      on.exit({
            if (file.exists(temporary_path)) unlink(temporary_path)
      }, add = TRUE)
      writeLines(source_lines, temporary_path)
      ##
      ## ---- Links the temporary file to the output ----------------------------------------------------------------------------------
      ##
      if (!file.link(temporary_path, output_path)) {
            stop("The generated source could not be written without replacing a file.")
      }
      ##
      ## ---- Returns the output and its fused calls ----------------------------------------------------------------------------------
      ##
      return(list(
            NicoSIMD_Stan_model_file_path = output_path,
            fused_calls = if (fuse_likelihood) {
                  "nico_finite_mixture_of_multivariate_ordered_probit_GHK_log_likelihood_in_tiles_of_subjects"
            } else {
                  character(0)
            },
            source_for_remaining_backend_conversion = output_path))

}

##
## ---- The fused writer --------------------------------------------------------------------------------------------------------------
##
fn_write_NicoSIMD_fused_one_class_MVOP_Stan_model <-  function(
      source_Stan_model_file_path,
      NicoSIMD_fused_Stan_model_file_path
) {

      result <-  fn_write_NicoSIMD_one_class_MVOP_Stan_model(
            source_Stan_model_file_path = source_Stan_model_file_path,
            NicoSIMD_Stan_model_file_path = NicoSIMD_fused_Stan_model_file_path,
            fuse_likelihood = TRUE)
      return(list(
            NicoSIMD_fused_Stan_model_file_path = result$NicoSIMD_Stan_model_file_path,
            fused_calls = result$fused_calls,
            source_for_remaining_backend_conversion = result$source_for_remaining_backend_conversion))

}

##
## ---- The non-fused writer ----------------------------------------------------------------------------------------------------------
##
fn_write_NicoSIMD_non_fused_one_class_MVOP_Stan_model <-  function(
      source_Stan_model_file_path,
      NicoSIMD_Stan_model_file_path
) {

      return(fn_write_NicoSIMD_one_class_MVOP_Stan_model(
            source_Stan_model_file_path = source_Stan_model_file_path,
            NicoSIMD_Stan_model_file_path = NicoSIMD_Stan_model_file_path,
            fuse_likelihood = FALSE))

}
