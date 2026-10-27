##
## ---- writes the fused longitudinal and survival calls ------------------------------------------------------------------------------
##

fn_write_NicoSIMD_fused_JLS_Stan_model <-  function(
      source_Stan_model_file_path,
      NicoSIMD_fused_Stan_model_file_path
##
## ---- applies the source checks -----------------------------------------------------------------------------------------------------
##

) {

      source_path <-  normalizePath(source_Stan_model_file_path,
                                    mustWork = TRUE)
      output_directory <-  normalizePath(dirname(NicoSIMD_fused_Stan_model_file_path),
                                         mustWork = TRUE)
      output_path <-  file.path(output_directory,
                                basename(NicoSIMD_fused_Stan_model_file_path))
      source_names <-  c(
            paste0("joint_longitudinal_survival_chunked_no_loops_over_N_v5_",
                   "with_normal_lupdf_Stan_math_for_layout_timings.stan"),
            "JLS_normal_lupdf_Stan_math_generated.stan")
      ##
      ## ---- applies the source checks -----------------------------------------------------------------------------------------------
      ##

      if (!(basename(source_path) %in% source_names)) {
            stop("The source is not a supported longitudinal and survival reference.")
      }
      ##
      ## ---- applies the source checks -----------------------------------------------------------------------------------------------
      ##

      if (identical(source_path,
                    output_path) || file.exists(output_path)) {
            stop("The output must be a new file, not the source.")
      }

      source_connection <-  file(source_path,
                                 open = "rb")
      source_bytes <-  readBin(source_connection,
                               what = "raw",
                               n = file.info(source_path)$size)
      close(source_connection)
      if (length(source_bytes) == 0) stop("The source is empty.")

      ##
      ## ---- masks comments and strings, keeping byte positions ----------------------------------------------------------------------
      ##

      fn_mask <-  function(bytes) {

            masked <-  bytes
            values <-  as.integer(bytes)
            state <-  "code"
            escaped <-  FALSE
            position <-  1L
            ##
            ## ---- applies the source checks -----------------------------------------------------------------------------------------
            ##

            while (position <= length(values)) {
                  current <-  values[position]
                  following <-  if (position < length(values)) values[position + 1L] else 0L
                  ##
                  ## ---- applies the source checks -----------------------------------------------------------------------------------
                  ##

                  if (state == "line") {
                        if (current %in% c(10L,
                                           13L)) state <-  "code"
                        else masked[position] <-  as.raw(32L)
                  ##
                  ## ---- applies the source checks -----------------------------------------------------------------------------------
                  ##

                  } else if (state == "block") {
                        if (current == 42L && following == 47L) {
                              masked[position:(position + 1L)] <-  as.raw(32L)
                              position <-  position + 1L
                              state <-  "code"
                        } else if (!(current %in% c(10L,
                                                    13L))) masked[position] <-  as.raw(32L)
                  ##
                  ## ---- applies the source checks -----------------------------------------------------------------------------------
                  ##

                  } else if (state == "string") {
                        if (!(current %in% c(10L,
                                             13L))) masked[position] <-  as.raw(32L)
                        if (escaped) escaped <-  FALSE
                        else if (current == 92L) escaped <-  TRUE
                        else if (current == 34L) state <-  "code"
                  ##
                  ## ---- applies the source checks -----------------------------------------------------------------------------------
                  ##

                  } else if (current == 47L && following %in% c(47L,
                                                                42L)) {
                        masked[position:(position + 1L)] <-  as.raw(32L)
                        state <-  if (following == 47L) "line" else "block"
                        position <-  position + 1L
                  ##
                  ## ---- applies the source checks -----------------------------------------------------------------------------------
                  ##

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

      masked <-  rawToChar(fn_mask(source_bytes))
      ##
      ## ---- applies the source checks -----------------------------------------------------------------------------------------------
      ##

      if (grepl("(?<![A-Za-z0-9_])nico_[A-Za-z0-9_]+",
                masked,
                perl = TRUE)) {
            stop("The reference already contains native backend calls.")
      }

      ##
      ## ---- applies the source checks -----------------------------------------------------------------------------------------------
      ##

      fn_one_match <-  function(pattern) {

            positions <-  gregexpr(pattern,
                                   masked,
                                   perl = TRUE,
                                   useBytes = TRUE)[[1]]
            ##
            ## ---- applies the source checks -----------------------------------------------------------------------------------------
            ##

            if (length(positions) != 1L || positions[1] < 1L) {
                  stop("The source span is missing or ambiguous.")
            }
            return(c(start = positions[1],
                     length = attr(positions,
                                   "match.length")[1]))

      }

      ##
      ## ---- applies the source checks -----------------------------------------------------------------------------------------------
      ##

      fn_block <-  function(pattern) {

            span <-  fn_one_match(pattern)
            opening <-  unname(span["start"] + span["length"] - 1L)
            values <-  as.integer(charToRaw(masked))
            depth <-  1L
            position <-  opening + 1L
            ##
            ## ---- applies the source checks -----------------------------------------------------------------------------------------
            ##

            while (position <= length(values) && depth > 0L) {
                  if (values[position] == 123L) depth <-  depth + 1L
                  if (values[position] == 125L) depth <-  depth - 1L
                  position <-  position + 1L
            }
            if (depth != 0L) stop("The source block has no closing brace.")
            return(c(opening = opening,
                     closing = position - 1L))

      }

      ##
      ## ---- applies the source checks -----------------------------------------------------------------------------------------------
      ##

      fn_tokens <-  function(code) {

            tokens <-  strsplit(gsub("[[:space:]]",
                                     "",
                                     code),
                                "",
                                fixed = TRUE)[[1]]
            return(paste0("\\Q",
                          tokens,
                          "\\E",
                          collapse = "\\s*"))

      }

      fn_inside <-  function(span,
                             ##
                             ## ---- applies the source checks ------------------------------------------------------------------------
                             ##

                             block) {

            return(span["start"] > block["opening"] &&
                   span["start"] + span["length"] - 1L < block["closing"])

      }

      ##
      ## ---- applies the source checks -----------------------------------------------------------------------------------------------
      ##

      fn_all_matches <-  function(pattern) {

            positions <-  gregexpr(pattern,
                                   masked,
                                   perl = TRUE,
                                   useBytes = TRUE)[[1]]
            if (positions[1] < 1L) stop("The source span is missing.")
            ##
            ## ---- applies the source checks -----------------------------------------------------------------------------------------
            ##

            return(lapply(seq_along(positions),
                          function(k) {

                  c(start = positions[k],
                    length = attr(positions,
                                  "match.length")[k])

            }))

      }

      functions_block <-  fn_block("(?<![A-Za-z0-9_])functions\\s*\\{")
      n_partial_sums <-  if (identical(basename(source_path),
                                       ##
                                       ## ---- applies the source checks --------------------------------------------------------------
                                       ##

                                       "JLS_normal_lupdf_Stan_math_generated.stan")) {
            2L
      } else 1L

      edits <-  list()

      ##
      ## ---- fuses each longitudinal likelihood --------------------------------------------------------------------------------------
      ##

      operation_spans <-  fn_all_matches(fn_tokens(paste0(
            "lp += normal_lupdf(Y_long[n1:n2] | X_long[n1:n2] * beta_long + random_effects_chunk, ",
            "sigma_long);")))
      if (length(operation_spans) != n_partial_sums ||
          !all(vapply(operation_spans,
                       fn_inside,
                       logical(1),
                       ##
                       ## ---- applies the source checks ------------------------------------------------------------------------------
                       ##

                       block = functions_block))) {
            stop("The supported operation is missing or outside the functions block.")
      }
      ##
      ## ---- applies the source checks -----------------------------------------------------------------------------------------------
      ##

      for (span in operation_spans) {
            edits[[length(edits) + 1L]] <-  list(
                  span = span,
                  text = paste0(
                        "lp += nico_normal_id_glm_lupdf(Y_long[n1:n2] | X_long[n1:n2],\n",
                        "              random_effects_chunk, beta_long, sigma_long);"))
      }

      ##
      ## ---- removes the intermediate hazard matrix ----------------------------------------------------------------------------------
      ##

      operation_spans <-  fn_all_matches(fn_tokens(paste0(
            "matrix[Q, n_subjects] log_hazard_quad_minus_log_lambda = rep_matrix(subject_part', Q) + ",
            "to_matrix(X_long_quad_stacked[r1:r2] * (alpha_assoc * beta_long), Q, n_subjects) + ",
            "quad_time_fraction * (T_surv[start:end] .* subject_slope)';")))
      if (length(operation_spans) != n_partial_sums ||
          !all(vapply(operation_spans,
                       fn_inside,
                       logical(1),
                       ##
                       ## ---- applies the source checks ------------------------------------------------------------------------------
                       ##

                       block = functions_block))) {
            stop("The supported operation is missing or outside the functions block.")
      }
      ##
      ## ---- applies the source checks -----------------------------------------------------------------------------------------------
      ##

      for (span in operation_spans) {
            edits[[length(edits) + 1L]] <-  list(
                  span = span,
                  text = "")
      }

      ##
      ## ---- fuses the weighted cumulative hazards -----------------------------------------------------------------------------------
      ##

      operation_spans <-  fn_all_matches(fn_tokens(paste0(
            "lp += -lambda * dot_product(to_vector(quad_weights * half_T_surv[start:end]'), ",
            "exp(to_vector(log_hazard_quad_minus_log_lambda)));")))
      if (length(operation_spans) != n_partial_sums ||
          !all(vapply(operation_spans,
                       fn_inside,
                       logical(1),
                       ##
                       ## ---- applies the source checks ------------------------------------------------------------------------------
                       ##

                       block = functions_block))) {
            stop("The supported operation is missing or outside the functions block.")
      }
      ##
      ## ---- applies the source checks -----------------------------------------------------------------------------------------------
      ##

      for (span in operation_spans) {
            edits[[length(edits) + 1L]] <-  list(
                  span = span,
                  text = paste0(
                        "lp += -lambda * ",
                        "nico_dot_product_exp_of_affine_predictor_at_quadrature_nodes_by_column(\n",
                        "              quad_weights, quad_time_fraction, half_T_surv[start:end], ",
                        "T_surv[start:end],\n",
                        "              X_long_quad_stacked[r1:r2], subject_part, subject_slope, ",
                        "alpha_assoc * beta_long);"))
      }

      declarations <-  c(
            "",
            "    real nico_normal_id_glm_lpdf(vector y, matrix x, vector alpha, vector beta, real sigma);",
            "    real nico_dot_product_exp_of_affine_predictor_at_quadrature_nodes_by_column(",
            "          data vector node_weights, data vector node_positions, data vector weight_scale,",
            "          data vector position_scale, data matrix X, vector a, vector b, vector beta);")
      edits[[length(edits) + 1L]] <-  list(
            span = c(start = functions_block["opening"] + 1L,
                       length = 0L),
            text = paste(declarations,
                           collapse = "\n"))

      line_ending <-  if (grepl("\r\n",
                                rawToChar(source_bytes),
                                fixed = TRUE)) "\r\n" else "\n"
      ##
      ## ---- applies the source checks -----------------------------------------------------------------------------------------------
      ##

      order_of_edits <-  order(vapply(edits,
                                      function(edit) {

            unname(edit$span[1])

      },
                                      numeric(1)),
                               decreasing = TRUE)
      output_bytes <-  source_bytes
      ##
      ## ---- applies the source checks -----------------------------------------------------------------------------------------------
      ##

      for (index in order_of_edits) {
            edit <-  edits[[index]]
            first <-  unname(edit$span[1])
            count <-  unname(edit$span[2])
            before <-  if (first > 1L) output_bytes[seq_len(first - 1L)] else raw(0)
            ##
            ## ---- applies the source checks -----------------------------------------------------------------------------------------
            ##

            after <-  if (first + count <= length(output_bytes)) {
                  output_bytes[seq.int(first + count,
                                       length(output_bytes))]
            } else raw(0)
            inserted <-  gsub("\n",
                              line_ending,
                              edit$text,
                              fixed = TRUE)
            output_bytes <-  c(before,
                               charToRaw(inserted),
                               after)
      }
      temporary_path <-  tempfile(pattern = ".NicoSIMD_fused_",
                                  tmpdir = output_directory)
      output_connection <-  file(temporary_path,
                                 open = "wb")
      ##
      ## ---- applies the source checks -----------------------------------------------------------------------------------------------
      ##

      on.exit({
            try(close(output_connection),
                silent = TRUE)
            if (file.exists(temporary_path)) unlink(temporary_path)
      },
              add = TRUE)
      writeBin(output_bytes,
               output_connection)
      close(output_connection)
      ##
      ## ---- applies the source checks -----------------------------------------------------------------------------------------------
      ##

      if (!file.link(temporary_path,
                     output_path)) {
            stop("The generated source could not be written without replacing a file.")
      }
      return(list(
            source_Stan_model_file_path = source_path,
            source_Stan_model_md5 = unname(tools::md5sum(source_path)),
            NicoSIMD_fused_Stan_model_file_path = output_path,
            fused_calls = c("nico_normal_id_glm_lpdf",
                            "nico_dot_product_exp_of_affine_predictor_at_quadrature_nodes_by_column"),
            source_for_remaining_backend_conversion = output_path))

}

