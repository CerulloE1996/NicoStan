##
## ---- R_fn_write_NicoSIMD_fused_hierarchical_logistic_regression_Stan_model.R -------------------------------------------------------
##
## ---- writes the fused Bernoulli-logit GLM with coefficients by group in place of Stan's bernoulli_logit_glm ------------------------
##
## The source is the hierarchical logistic regression's Stan::math-fused file (bernoulli_logit_glm on the
## columns [1, x]), with Stan's reduce_sum or with nico_reduce_sum. The groups' intercepts and slopes
## (mu + sigma * beta_raw) are made once per slice and passed by group to
## nico_bernoulli_logit_glm_with_coefficients_by_group, on the data column x. Every other line is left as it is,
## for the remaining NicoSIMD conversion.
##
##
## ---- The function ------------------------------------------------------------------------------------------------------------------
##
fn_write_NicoSIMD_fused_hierarchical_logistic_regression_Stan_model <-  function(
      source_Stan_model_file_path,
      NicoSIMD_fused_Stan_model_file_path
) {

      ##
      ## ---- Paths -------------------------------------------------------------------------------------------------------------------
      ##
      source_path <-  normalizePath(source_Stan_model_file_path,
                                     mustWork = TRUE)
      output_directory <-  normalizePath(dirname(NicoSIMD_fused_Stan_model_file_path),
                                          mustWork = TRUE)
      output_path <-  file.path(output_directory,
                                 basename(NicoSIMD_fused_Stan_model_file_path))
      source_stem <-  "hierarchical_logistic_regression_chunked_no_loops_over_N_with_bernoulli_logit_glm"
      source_names <-  paste0(source_stem,
                               c("", "_automatic_reduce_sum_chunked"),
                               "_Stan_math_for_layout_timings.stan")
      ##
      ## ---- Checks ------------------------------------------------------------------------------------------------------------------
      ##
      if (!(basename(source_path) %in% source_names)) {
            stop("The source is not a supported hierarchical logistic regression bernoulli_logit_glm file.")
      }
      if (identical(source_path,
                    output_path) || file.exists(output_path)) {
            stop("The output must be a new file, not the source.")
      }
      nico_reduce_sum <-  identical(basename(source_path),
                                    source_names[2])
      source_connection <-  file(source_path,
                                  open = "rb")
      source_bytes <-  readBin(source_connection,
                                what = "raw",
                                n = file.info(source_path)$size)
      close(source_connection)
      if (length(source_bytes) == 0) stop("The source is empty.")
      ##
      ## ---- Masks comments and strings, keeping byte positions ----------------------------------------------------------------------
      ##
      fn_mask <-  function(bytes) {

            masked <-  bytes
            values <-  as.integer(bytes)
            state <-  "code"
            escaped <-  FALSE
            position <-  1L
            while (position <= length(values)) {
                  current <-  values[position]
                  following <-  if (position < length(values)) values[position + 1L] else 0L
                  if (state == "line") {
                        if (current %in% c(10L,
                                          13L)) state <-  "code"
                        else masked[position] <-  as.raw(32L)
                  } else if (state == "block") {
                        if (current == 42L && following == 47L) {
                              masked[position:(position + 1L)] <-  as.raw(32L)
                              position <-  position + 1L
                              state <-  "code"
                        } else if (!(current %in% c(10L,
                                                    13L))) masked[position] <-  as.raw(32L)
                  } else if (state == "string") {
                        if (!(current %in% c(10L,
                                             13L))) masked[position] <-  as.raw(32L)
                        if (escaped) escaped <-  FALSE
                        else if (current == 92L) escaped <-  TRUE
                        else if (current == 34L) state <-  "code"
                  } else if (current == 47L && following %in% c(47L,
                                                               42L)) {
                        masked[position:(position + 1L)] <-  as.raw(32L)
                        state <-  if (following == 47L) "line" else "block"
                        position <-  position + 1L
                  } else if (current == 34L) {
                        masked[position] <-  as.raw(32L)
                        state <-  "string"
                  }
                  position <-  position + 1L
            }
            if (state %in% c("block",
                             "string")) stop("The source has an unterminated comment or string.")
            return(masked)

      }
      ##
      ## ---- Checks ------------------------------------------------------------------------------------------------------------------
      ##
      masked_source <-  rawToChar(fn_mask(source_bytes))
      if (grepl("(?<![A-Za-z0-9_])nico_[A-Za-z0-9_]+",
                masked_source,
                perl = TRUE)) {
            stop("The source already contains native backend calls.")
      }
      ##
      ## ---- Checks source slices ----------------------------------------------------------------------------------------------------
      ##

      whole_raw_block <-  "(?<![A-Za-z0-9_])to_matrix\\s*\\(\\s*beta_raw\\s*\\)"
      whole_predictor <-  "(?<![A-Za-z0-9_])vector\\s*\\[\\s*N\\s*\\]\\s+eta(?![A-Za-z0-9_])"
      whole_group_gather <-  "(?<![A-Za-z0-9_])beta\\s*\\[\\s*group\\s*,"
      if (grepl(whole_raw_block,
                masked_source,
                perl = TRUE) ||
          grepl(whole_predictor,
                masked_source,
                perl = TRUE) ||
          grepl(whole_group_gather,
                masked_source,
                perl = TRUE)) {

            stop("The generated quantities must read only their required parameter and observation slices.")

      }
      line_ending <-  if (grepl("\r\n",
                               rawToChar(source_bytes),
                               fixed = TRUE)) "\r\n" else "\n"
      ##
      ## ---- The edits: old text, new text, the number of times each is in the source, and whether it is code ------------------------
      ##
      continuation <-  strrep(" ",
                              75L)
      slice_lines <-  paste0("        vector[end - start + 1] intercepts_of_slice = mu[1] + sigma[1] * ",
                             "beta_raw_of_slice[, 1];\n",
                             "        matrix[end - start + 1, 1] slopes_of_slice = mu[2] + sigma[2] * ",
                             "beta_raw_of_slice[, 2:2];\n")
      edits <-  list(
            list(old = paste0(strrep(" ", 33L), "data vector x_sorted,\n",
                              strrep(" ", 33L), "data matrix x_with_ones_sorted,\n"),
                 new = paste0(strrep(" ", 33L), "data matrix x_sorted_as_matrix,\n"),
                 times = c(1L, 1L),
                 active = TRUE),
            list(old = paste0("data array[] int y_sorted, data vector x_sorted, data matrix x_with_ones_sorted, ",
                              "data array[] int group_sorted,"),
                 new = paste0("data array[] int y_sorted, data matrix x_sorted_as_matrix, ",
                              "data array[] int group_sorted,"),
                 times = c(0L, 1L),
                 active = TRUE),
            list(old = paste0("        matrix[end - start + 1, 2] beta_raw_of_slice = ",
                              "to_matrix(beta_raw_slice);\n"),
                 new = paste0("        matrix[end - start + 1, 2] beta_raw_of_slice = ",
                              "to_matrix(beta_raw_slice);\n",
                              slice_lines),
                 times = c(1L, 1L),
                 active = TRUE),
            list(old = "        matrix[end - start + 1, 2] beta_raw_of_slice = beta_raw_slice;\n",
                 new = paste0("        matrix[end - start + 1, 2] beta_raw_of_slice = beta_raw_slice;\n",
                              slice_lines),
                 times = c(0L, 1L),
                 active = TRUE),
            list(old = paste0(strrep(" ", 15L), "// (the same values, with Stan's fused Bernoulli-logit GLM: ",
                              "the population coefficients mu on the\n",
                              strrep(" ", 15L), "//  columns [1, x], each observation's group deviations ",
                              "sigma .* beta_raw as its intercept)"),
                 new = paste0(strrep(" ", 15L), "// (the same values, with NicoSIMD's Bernoulli-logit GLM ",
                              "by group: each group's own intercept and\n",
                              strrep(" ", 15L), "//  coefficients, for any number of columns of x)"),
                 times = c(1L, 2L),
                 active = FALSE),
            list(old = paste0("+ bernoulli_logit_glm_lupmf(y_sorted[first:last] | ",
                              "x_with_ones_sorted[first:last],\n",
                              strrep(" ", 43L), "sigma[1] * beta_raw_of_slice[position_in_slice, 1]\n",
                              strrep(" ", 43L), "+ (sigma[2] * beta_raw_of_slice[position_in_slice, 2])\n",
                              strrep(" ", 45L), ".* x_sorted[first:last],\n",
                              strrep(" ", 43L), "mu);"),
                 new = paste0("+ nico_bernoulli_logit_glm_with_coefficients_by_group_lupmf(",
                              "y_sorted[first:last] |\n",
                              continuation, "x_sorted_as_matrix[first:last],\n",
                              continuation, "position_in_slice,\n",
                              continuation, "intercepts_of_slice,\n",
                              continuation, "slopes_of_slice);"),
                 times = c(1L, 2L),
                 active = TRUE),
            list(old = "    matrix[N, 2] x_with_ones_sorted = append_col(rep_vector(1, N), x_sorted);",
                 new = "    matrix[N, 1] x_sorted_as_matrix = to_matrix(x_sorted);",
                 times = c(1L, 1L),
                 active = TRUE),
            list(old = paste0("y_sorted, x_sorted, x_with_ones_sorted, group_sorted, first_of_group,\n",
                              strrep(" ", 32L), "last_of_group);"),
                 new = "y_sorted, x_sorted_as_matrix, group_sorted, first_of_group, last_of_group);",
                 times = c(1L, 0L),
                 active = TRUE),
            list(old = paste0("mu, sigma, y_sorted, x_sorted, x_with_ones_sorted, group_sorted, first_of_group, ",
                              "last_of_group, nicostan_values_per_term_of_beta_raw"),
                 new = paste0("mu, sigma, y_sorted, x_sorted_as_matrix, group_sorted, first_of_group, ",
                              "last_of_group, nicostan_values_per_term_of_beta_raw"),
                 times = c(0L, 1L),
                 active = TRUE),
            list(old = "functions {\n",
                 new = paste0("functions {\n",
                              "    real nico_bernoulli_logit_glm_with_coefficients_by_group_lpmf(array[] int y, ",
                              "matrix x, array[] int group,\n",
                              strrep(" ", 66L), "vector alpha, matrix B);\n"),
                 times = c(1L, 1L),
                 active = TRUE))
      ##
      ## ---- Applies each edit, after checking the number of times it is there and that it is code or a comment ----------------------
      ##
      output_text <-  rawToChar(source_bytes)
      ##
      ## ---- Chunk-size spelling in inherited comments -----------------------------------------------------------------------------------
      ##
      output_text <-  gsub(paste0("grain",
                                 "size"),
                           "chunk_size",
                           output_text,
                           ignore.case = TRUE)
      ##
      ## ---- Applies the checked edits ----------------------------------------------------------------------------------------------------
      ##
      for (edit in edits) {

            old <-  gsub("\n",
                         line_ending,
                         edit$old,
                         fixed = TRUE)
            new <-  gsub("\n",
                         line_ending,
                         edit$new,
                         fixed = TRUE)
            masked_text <-  rawToChar(fn_mask(charToRaw(output_text)))
            positions <-  gregexpr(old,
                                    output_text,
                                    fixed = TRUE,
                                    useBytes = TRUE)[[1]]
            n_found <-  if (positions[1] < 1L) 0L else length(positions)
            if (n_found != edit$times[1L + nico_reduce_sum]) {
                  stop(paste0("The text to replace is not in the source the expected number of times: ",
                              substr(edit$old, 1L, 60L)))
            }
            if (n_found > 0L) {
                  for (position in positions) {
                        masked_span <-  substr(masked_text,
                                               position,
                                               position + nchar(old, type = "bytes") - 1L)
                        if (edit$active && !identical(masked_span,
                                                      old)) {
                              stop("The source code to replace is not active code.")
                        }
                        if (!edit$active && !grepl("^[ \r\n]*$",
                                                   masked_span)) {
                              stop("The source comment to replace is not a comment.")
                        }
                  }
                  output_text <-  gsub(old,
                                       new,
                                       output_text,
                                       fixed = TRUE,
                                       useBytes = TRUE)
            }

      }
      ##
      ## ---- Checks: no Stan GLM call and no column of ones left in the active code --------------------------------------------------
      ##
      masked_output <-  rawToChar(fn_mask(charToRaw(output_text)))
      if (grepl("(?<![A-Za-z0-9_])bernoulli_logit_glm_lu?pmf",
                masked_output,
                perl = TRUE) || grepl("x_with_ones_sorted",
                                      masked_output,
                                      fixed = TRUE)) {
            stop("The written file still has Stan's bernoulli_logit_glm or the column of ones.")
      }
      ##
      ## ---- Checks fused calls ------------------------------------------------------------------------------------------------------
      ##

      fused_call_positions <-  gregexpr(
            paste0("(?<![A-Za-z0-9_])",
                   "nico_bernoulli_logit_glm_with_coefficients_by_group_lupmf\\s*\\("),
            masked_output,
            perl = TRUE)[[1]]
      n_fused_calls <-  if (fused_call_positions[1] < 1L) 0L else length(fused_call_positions)
      if (n_fused_calls != 1L + as.integer(nico_reduce_sum)) {

            stop("The written file does not contain every expected fused likelihood call.")

      }
      ##
      ## ---- Writes the new file, never in place of another --------------------------------------------------------------------------
      ##
      temporary_path <-  tempfile(pattern = ".NicoSIMD_fused_",
                                   tmpdir = output_directory)
      output_connection <-  file(temporary_path,
                                  open = "wb")
      on.exit({
            try(close(output_connection),
                 silent = TRUE)
            if (file.exists(temporary_path)) unlink(temporary_path)
      }, add = TRUE)
      writeBin(charToRaw(output_text),
                output_connection)
      close(output_connection)
      if (!file.link(temporary_path,
                     output_path)) {
            stop("The generated source could not be written without replacing a file.")
      }
      return(list(source_Stan_model_file_path = source_path,
                  source_Stan_model_md5 = unname(tools::md5sum(source_path)),
                  NicoSIMD_fused_Stan_model_file_path = output_path,
                  fused_calls = "nico_bernoulli_logit_glm_with_coefficients_by_group_lpmf",
                  source_for_remaining_backend_conversion = output_path))

}
