##
## ---- Function ----------------------------------------------------------------------------------------------------------------------
##
fn_write_NicoSIMD_fused_NMA_ord_biv_Stan_model <-  function(
      source_Stan_model_file_path,
      NicoSIMD_fused_Stan_model_file_path
) {

##
## ---- Controls ----------------------------------------------------------------------------------------------------------------------
##
      ## (writes the NMA ord_biv (fixed cutpoints) nico::math-fused source from the Stan source)
      source_path <-  normalizePath(source_Stan_model_file_path, mustWork = TRUE)
      output_directory <-  normalizePath(dirname(NicoSIMD_fused_Stan_model_file_path), mustWork = TRUE)
      output_path <-  file.path(output_directory, basename(NicoSIMD_fused_Stan_model_file_path))
      source_stem <-  paste0("network_meta_analysis_Xu_fixed_cutpoints_chunked_no_loops_over_N_with_the_fused_",
                             "binomial_counts_on_AVX_and_the_constants_of_the_data_left_out")
      source_names <-  paste0(source_stem, c("", "_automatic_reduce_sum_chunked"),
                              "_Stan_math_for_layout_timings.stan")
##
## ---- Conditions --------------------------------------------------------------------------------------------------------------------
##
      if (!(basename(source_path) %in% source_names)) {
            stop("The NMA ord_biv source is not a supported Stan reference.")
      }
##
## ---- Conditions --------------------------------------------------------------------------------------------------------------------
##
      if (identical(source_path, output_path) || file.exists(output_path)) {
            stop("The output must be a new file, not the source.")
      }
      ## (the partial sums: one with Stan's reduce_sum; with nico_reduce_sum, the source's own and one per
      ##  nuisance block)
      n_partial_sums <-  if (identical(basename(source_path), source_names[1])) 1L else 5L
      source_connection <-  file(source_path, open = "rb")
      source_bytes <-  readBin(source_connection, what = "raw", n = file.info(source_path)$size)
      close(source_connection)
      if (length(source_bytes) == 0) stop("The source is empty.")
##
## ---- Function ----------------------------------------------------------------------------------------------------------------------
##
      ## (masks comments and strings, keeping byte positions)
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
                        if (current %in% c(10L, 13L)) state <-  "code"
                        else masked[position] <-  as.raw(32L)
                  } else if (state == "block") {
                        if (current == 42L && following == 47L) {
                              masked[position:(position + 1L)] <-  as.raw(32L)
                              position <-  position + 1L
                              state <-  "code"
                        } else if (!(current %in% c(10L, 13L))) masked[position] <-  as.raw(32L)
                  } else if (state == "string") {
                        if (!(current %in% c(10L, 13L))) masked[position] <-  as.raw(32L)
                        if (escaped) escaped <-  FALSE
                        else if (current == 92L) escaped <-  TRUE
                        else if (current == 34L) state <-  "code"
                  } else if (current == 47L && following %in% c(47L, 42L)) {
                        masked[position:(position + 1L)] <-  as.raw(32L)
                        state <-  if (following == 47L) "line" else "block"
                        position <-  position + 1L
                  } else if (current == 34L) {
                        masked[position] <-  as.raw(32L)
                        state <-  "string"
                  }
                  position <-  position + 1L
            }
            if (state %in% c("block", "string")) stop("The source has an unterminated comment or string.")
            return(masked)

      }
##
## ---- Controls ----------------------------------------------------------------------------------------------------------------------
##
      source_text <-  rawToChar(source_bytes)
      masked <-  rawToChar(fn_mask(source_bytes))
##
## ---- Conditions --------------------------------------------------------------------------------------------------------------------
##
      if (grepl("(?<![A-Za-z0-9_])nico_[A-Za-z0-9_]+", masked, perl = TRUE)) {
            stop("The reference already contains native backend calls.")
      }
##
## ---- Function ----------------------------------------------------------------------------------------------------------------------
##
      ## (the spans of a pattern in the masked source, exactly expected_count of them)
      fn_matches <-  function(pattern,
                              expected_count) {

            positions <-  gregexpr(pattern, masked, perl = TRUE, useBytes = TRUE)[[1]]
##
## ---- Conditions --------------------------------------------------------------------------------------------------------------------
##
            if (positions[1] < 1L || length(positions) != expected_count) {
                  stop("The source span is missing or ambiguous.")
            }
##
## ---- Function ----------------------------------------------------------------------------------------------------------------------
##
            return(lapply(seq_along(positions), function(index) {

                  c(start = positions[index], length = attr(positions, "match.length")[index])

            }))

      }
##
## ---- Function ----------------------------------------------------------------------------------------------------------------------
##
      ## (the closing brace of the block opened at a position)
      fn_closing_brace <-  function(opening) {

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
            return(position - 1L)

      }
##
## ---- Function ----------------------------------------------------------------------------------------------------------------------
##
      ## (the opening and closing braces of each block whose header matches)
      fn_blocks <-  function(pattern,
                             expected_count) {

##
## ---- Function ----------------------------------------------------------------------------------------------------------------------
##
            return(lapply(fn_matches(pattern, expected_count), function(span) {

                  opening <-  unname(span["start"] + span["length"] - 1L)
                  c(opening = opening, closing = fn_closing_brace(opening))

            }))

      }
##
## ---- Function ----------------------------------------------------------------------------------------------------------------------
##
      ## (a pattern of the code's tokens, any whitespace between them)
      fn_tokens <-  function(code) {

            tokens <-  strsplit(gsub("[[:space:]]", "", code), "", fixed = TRUE)[[1]]
            return(paste0("\\Q", tokens, "\\E", collapse = "\\s*"))

      }
##
## ---- Function ----------------------------------------------------------------------------------------------------------------------
##
      ## (TRUE when the span lies inside the block)
      fn_inside <-  function(span,
                             block) {

            return(span["start"] > block["opening"] &&
                   span["start"] + span["length"] - 1L < block["closing"])

      }
##
## ---- Controls ----------------------------------------------------------------------------------------------------------------------
##
      functions_block <-  fn_blocks("(?<![A-Za-z0-9_])functions\\s*\\{", 1L)[[1]]
      model_block <-  fn_blocks("(?<![A-Za-z0-9_])model\\s*\\{", 1L)[[1]]
      partial_sum_bodies <-  fn_blocks(paste0("(?<![A-Za-z0-9_])real\\s+partial_sum_of_the_studies_with_",
                                              "the_same_number_of_index_tests[A-Za-z0-9_]*_lp[dm]f\\s*",
                                              "\\([^{]*\\{"),
                                       n_partial_sums)
##
## ---- Likelihood --------------------------------------------------------------------------------------------------------------------
##
      ## (each partial sum's likelihood: one call of the fused binomial node of the counts: S = F(latent), the
      ##  previous reported cutpoint's S, their ratio within [1e-10, 1 - 1e-10] and the binomial)
      likelihoods <-  fn_matches(paste0("(?<![A-Za-z0-9_])", fn_tokens(paste0(
            "vector[n_cutpoints_of_chunk] surv_prob = (use_probit_link == 1) ? Phi(latent_surv) : ",
            "inv_logit(latent_surv); ",
            "vector[n_cutpoints_of_chunk] surv_prob_of_previous_reported_cutpoint = ",
            "append_row(1.0, surv_prob[1:(n_cutpoints_of_chunk - 1)]) ",
            ".* is_not_first_reported_cutpoint_of_its_pair[first_cutpoint:last_cutpoint] ",
            "+ is_first_reported_cutpoint_of_its_pair[first_cutpoint:last_cutpoint]; ",
            "vector[n_cutpoints_of_chunk] cond_prob = surv_prob ./ surv_prob_of_previous_reported_cutpoint; ",
            "vector[n_cutpoints_of_chunk] cond_prob_safe = fmax(1e-10, fmin(1 - 1e-10, cond_prob)); ",
            "log_density += binomial_lupmf(",
            "count_above_each_reported_cutpoint[c, first_cutpoint:last_cutpoint] | ",
            "count_above_the_previous_reported_cutpoint_or_class_total[c, first_cutpoint:last_cutpoint], ",
            "cond_prob_safe);"))), n_partial_sums)
##
## ---- Loop --------------------------------------------------------------------------------------------------------------------------
##
      for (body in partial_sum_bodies) {
##
## ---- Conditions --------------------------------------------------------------------------------------------------------------------
##
            if (sum(vapply(likelihoods, fn_inside, logical(1), block = body)) != 1L) {
                  stop("Each partial sum must hold exactly one likelihood.")
            }
##
## ---- Conditions --------------------------------------------------------------------------------------------------------------------
##
            if (!fn_inside(c(start = unname(body["opening"]), length = 1L), functions_block)) {
                  stop("A partial sum is not in the functions block.")
            }
      }
##
## ---- Function ----------------------------------------------------------------------------------------------------------------------
##
      edits <-  lapply(likelihoods, function(span) {

            list(span = span,
                 text = paste0(
                       "//// (the survival probabilities, their conditional probabilities within\n",
                       strrep(" ", 18), "////  [1e-10, 1 - 1e-10], as the original's safe_binomial_lpmf, ",
                       "and the binomial\n",
                       strrep(" ", 18), "////  likelihood of the counts with every constant, as ONE autodiff ",
                       "node: NicoSIMD's\n",
                       strrep(" ", 18), "////  general nico_binomial_counts_above_successive_cutpoints_of_a_",
                       "cumulative_link_lpmf;\n",
                       strrep(" ", 18), "////  the chunk's first cutpoint is its first pair's first)\n",
                       strrep(" ", 18), "log_density += nico_binomial_counts_above_successive_cutpoints_of_a_",
                       "cumulative_link_lupmf(\n",
                       strrep(" ", 24), "count_above_each_reported_cutpoint[c, first_cutpoint:last_cutpoint] |\n",
                       strrep(" ", 24), "count_above_the_previous_reported_cutpoint_or_class_total[c,\n",
                       strrep(" ", 82), "first_cutpoint:last_cutpoint],\n",
                       strrep(" ", 24), "latent_surv,\n",
                       strrep(" ", 24), "is_first_reported_cutpoint_of_its_pair[first_cutpoint:last_cutpoint],\n",
                       strrep(" ", 24), "use_probit_link,\n",
                       strrep(" ", 24), "1e-10);"))

      })
##
## ---- Between-study effects ---------------------------------------------------------------------------------------------------------
##
      ## (each partial sum's between-study effects: one call of the fused node of a mean plus random effects
      ##  times transposed loadings, with a zero mean and not exponentiated)
      between_study_effects <-  fn_matches(paste0("(?<![A-Za-z0-9_])", fn_tokens(
            "matrix[n_studies_of_chunk, 2] beta_eta = beta_z[, 1:2] * beta_L_Sigma';")), n_partial_sums)
##
## ---- Loop --------------------------------------------------------------------------------------------------------------------------
##
      for (body in partial_sum_bodies) {
##
## ---- Conditions --------------------------------------------------------------------------------------------------------------------
##
            if (sum(vapply(between_study_effects, fn_inside, logical(1), block = body)) != 1L) {
                  stop("Each partial sum must hold exactly one product of the between-study effects.")
            }
      }
##
## ---- Function ----------------------------------------------------------------------------------------------------------------------
##
      edits <-  c(edits, lapply(between_study_effects, function(span) {

            list(span = span,
                 text = paste0(
                       "//// (as ONE autodiff node: the zero mean plus beta_z[, 1:2] beta_L_Sigma', ",
                       "not exponentiated)\n",
                       strrep(" ", 12), "matrix[n_studies_of_chunk, 2] beta_eta =\n",
                       strrep(" ", 18), "nico_rows_of_mean_plus_random_effects_times_transposed_loadings(",
                       "beta_z[, 1:2],\n",
                       strrep(" ", 82), "rep_row_vector(0, 2),\n",
                       strrep(" ", 82), "beta_L_Sigma', 0);"))

      }))
##
## ---- Induced-Dirichlet prior -------------------------------------------------------------------------------------------------------
##
      ## (the induced-Dirichlet prior of each test's cutpoints: one call of the fused node: F(C), the category
      ##  probabilities, the log density of F at C and the dirichlet)
      induced_dirichlet <-  fn_matches(paste0("(?<![A-Za-z0-9_])", fn_tokens(paste0(
            "vector[n_thr_t] Ind_Dir_cumul_prob = (use_probit_link == 1) ? Phi(C_vec_t) : inv_logit(C_vec_t); ",
            "vector[n_cat_t] Ind_Dir_ord_prob = cumul_probs_to_ord_probs(Ind_Dir_cumul_prob); ",
            "if (use_probit_link == 1) { ",
            "target += std_normal_lupdf(C_vec_t); ",
            "} else { ",
            "target += sum(log_inv_logit(C_vec_t) + log1m_inv_logit(C_vec_t)); ",
            "} ",
            "target += dirichlet_lupdf(Ind_Dir_ord_prob | to_vector(prior_dirichlet_alpha[c][t, 1:n_cat_t]));"))),
            1L)[[1]]
##
## ---- Conditions --------------------------------------------------------------------------------------------------------------------
##
      if (!fn_inside(induced_dirichlet, model_block)) {
            stop("The induced-Dirichlet prior is not in the model block.")
      }
      edits[[length(edits) + 1L]] <-  list(
            span = induced_dirichlet,
            text = paste0(
                  "//// (the induced-Dirichlet prior of the test's cutpoints, the original's F(C), category\n",
                  strrep(" ", 18), "////  probabilities, log density of F at C and dirichlet, as ONE autodiff ",
                  "node: NicoSIMD's\n",
                  strrep(" ", 18), "////  general nico_induced_dirichlet_prior_of_ordered_cutpoints_lpdf)\n",
                  strrep(" ", 18), "target += nico_induced_dirichlet_prior_of_ordered_cutpoints_lupdf(\n",
                  strrep(" ", 24), "C_vec_t | to_vector(prior_dirichlet_alpha[c][t, 1:n_cat_t]), ",
                  "use_probit_link);"))
##
## ---- Jacobian of the cutpoints -----------------------------------------------------------------------------------------------------
##
      ## (the log determinant of the Jacobian of each test's cutpoints: one call of the fused node, the same value
      ##  and derivatives as raw_C_to_C_log_det_J_lp)
      jacobian <-  fn_matches(paste0("(?<![A-Za-z0-9_])", fn_tokens(paste0(
            "target += raw_C_to_C_log_det_J_lp(C_raw_vec[c][start_index[t]:end_index[t]], softplus);"))), 1L)[[1]]
##
## ---- Conditions --------------------------------------------------------------------------------------------------------------------
##
      if (!fn_inside(jacobian, model_block)) {
            stop("The Jacobian of the cutpoints is not in the model block.")
      }
      edits[[length(edits) + 1L]] <-  list(
            span = jacobian,
            text = paste0(
                  "//// (the log determinant of the Jacobian of the test's cutpoints, as ONE autodiff node)\n",
                  strrep(" ", 15), "target += nico_log_det_jacobian_of_cutpoints_from_raw_differences(\n",
                  strrep(" ", 21), "C_raw_vec[c][start_index[t]:end_index[t]], softplus);"))
##
## ---- Declarations ------------------------------------------------------------------------------------------------------------------
##
      ## (the declarations of the existing nodes: the first 2 as in the hand-written files, the third as in the
      ##  PK fused files, the fourth as in its header)
      declarations <-  c(
            "",
            "        real nico_binomial_counts_above_successive_cutpoints_of_a_cumulative_link_lpmf(",
            paste0("              array[] int n, array[] int n_previous, vector latent, data vector is_first, ",
                   "data int link,"),
            "              data real bound);",
            paste0("        real nico_induced_dirichlet_prior_of_ordered_cutpoints_lpdf(vector C, ",
                   "data vector alpha, data int link);"),
            paste0("        matrix nico_rows_of_mean_plus_random_effects_times_transposed_loadings(matrix z, ",
                   "row_vector mean,"),
            "              matrix transposed_loadings, data int exponentiate);",
            paste0("        real nico_log_det_jacobian_of_cutpoints_from_raw_differences(vector C_raw, ",
                   "data int softplus);"))
      edits[[length(edits) + 1L]] <-  list(
            span = c(start = unname(functions_block["opening"]) + 1L, length = 0),
            text = paste(declarations, collapse = "\n"))
##
## ---- Output ------------------------------------------------------------------------------------------------------------------------
##
      ## (applies the byte spans, last first, and writes without replacing an existing file)
      line_ending <-  if (grepl("\r\n", source_text, fixed = TRUE)) "\r\n" else "\n"
##
## ---- Function ----------------------------------------------------------------------------------------------------------------------
##
      order_of_edits <-  order(vapply(edits, function(edit) {

            unname(edit$span[1])

      }, numeric(1)), decreasing = TRUE)
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
                  output_bytes[seq.int(first + count, length(output_bytes))]
            } else raw(0)
            inserted <-  gsub("\n", line_ending, edit$text, fixed = TRUE)
            output_bytes <-  c(before, charToRaw(inserted), after)
      }
      temporary_path <-  tempfile(pattern = ".NicoSIMD_fused_", tmpdir = output_directory)
      output_connection <-  file(temporary_path, open = "wb")
##
## ---- Block -------------------------------------------------------------------------------------------------------------------------
##
      on.exit({
            try(close(output_connection), silent = TRUE)
            if (file.exists(temporary_path)) unlink(temporary_path)
      }, add = TRUE)
      writeBin(output_bytes, output_connection)
      close(output_connection)
##
## ---- Conditions --------------------------------------------------------------------------------------------------------------------
##
      if (!file.link(temporary_path, output_path)) {
            stop("The generated source could not be written without replacing a file.")
      }
      return(list(
            NicoSIMD_fused_Stan_model_file_path = output_path,
            fused_calls = c("nico_binomial_counts_above_successive_cutpoints_of_a_cumulative_link_lupmf",
                            "nico_rows_of_mean_plus_random_effects_times_transposed_loadings",
                            "nico_induced_dirichlet_prior_of_ordered_cutpoints_lupdf",
                            "nico_log_det_jacobian_of_cutpoints_from_raw_differences"),
            source_for_remaining_backend_conversion = output_path))

}






















