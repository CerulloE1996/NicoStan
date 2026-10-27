##
## ---- writes the existing fused tensor-product logistic operation:
##

fn_write_NicoSIMD_fused_HSGP_Stan_model <-  function(
      source_Stan_model_file_path,
      NicoSIMD_fused_Stan_model_file_path
) {

      source_path <-  normalizePath(source_Stan_model_file_path,
                                     mustWork = TRUE)
      output_directory <-  normalizePath(dirname(NicoSIMD_fused_Stan_model_file_path),
                                          mustWork = TRUE)
      output_path <-  file.path(output_directory,
                                 basename(NicoSIMD_fused_Stan_model_file_path))
      source_name <-  paste0(
            "spatial_logistic_regression_with_a_Hilbert_space_approximate_Gaussian_process_",
            "chunked_with_a_tensor_product_basis_Stan_math_for_layout_timings.stan")
      if (!identical(basename(source_path),
                     source_name)) {
            stop("The source is not the supported tensor-product logistic reference.")
      }
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

      ## ---- masks comments and strings, keeping byte positions:

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

      masked <-  rawToChar(fn_mask(source_bytes))
      if (grepl("(?<![A-Za-z0-9_])nico_[A-Za-z0-9_]+",
                masked,
                perl = TRUE)) {
            stop("The reference already contains native backend calls.")
      }

      fn_one_match <-  function(pattern) {

            positions <-  gregexpr(pattern,
                                    masked,
                                    perl = TRUE,
                                    useBytes = TRUE)[[1]]
            if (length(positions) != 1L || positions[1] < 1L) {
                  stop("The source span is missing or ambiguous.")
            }
            return(c(start = positions[1],
                     length = attr(positions,
                                   "match.length")[1]))

      }

      fn_block <-  function(pattern) {

            span <-  fn_one_match(pattern)
            opening <-  unname(span["start"] + span["length"] - 1L)
            values <-  as.integer(charToRaw(masked))
            depth <-  1L
            position <-  opening + 1L
            while (position <= length(values) && depth > 0L) {
                  if (values[position] == 123L) depth <-  depth + 1L
                  if (values[position] == 125L) depth <-  depth - 1L
                  position <-  position + 1L
            }
            if (depth != 0L) stop("The source has an unmatched block.")
            return(c(opening = opening,
                     closing = position - 1L))

      }

      fn_one_fixed_match <-  function(source_text) {

            positions <-  gregexpr(source_text,
                                    rawToChar(source_bytes),
                                    fixed = TRUE,
                                    useBytes = TRUE)[[1]]
            if (length(positions) != 1L || positions[1] < 1L) {
                  stop("The exact source span is missing or ambiguous.")
            }
            span <-  c(start = positions[1],
                         length = attr(positions,
                                         "match.length")[1])
            if (!identical(substr(masked,
                                   span[1],
                                   sum(span) - 1L),
                           source_text)) {
                  stop("The exact source span is not active code.")
            }
            return(span)

      }

      functions_span <-  fn_one_match("(?m)^functions[[:space:]]*\\{")
      parameters_block <-  fn_block("(?m)^parameters[[:space:]]*\\{")
      parameter_text <-  substr(masked,
                                 parameters_block[1] + 1L,
                                 parameters_block[2] - 1L)
      expected_parameters <-  paste0("vector[M]z;realalpha;vector[3]beta;",
                                      "real<lower=0>sigma_gp;real<lower=0>length_scale;")
      if (!identical(gsub("[[:space:]]",
                          "",
                          parameter_text),
                     expected_parameters)) {
            stop("The parameter declarations differ from the supported reference.")
      }
      operation_text <-  paste0(
            "return bernoulli_logit_glm_lupmf(y_slice | covariates[start:end],\n",
            "                                         alpha + rows_dot_product(basis_1[start:end] * weights,\n",
            "                                                                  basis_2[start:end]),\n",
            "                                         beta);")
      partial_sum_text <-  paste0(
            "real partial_sum_people_lpmf(array[] int y_slice, int start, int end,\n",
            "                                 data matrix covariates,\n",
            "                                 data matrix basis_1,\n",
            "                                 data matrix basis_2,\n",
            "                                 real alpha,\n",
            "                                 vector beta,\n",
            "                                 matrix weights) {\n",
            "        ",
            operation_text,
            "\n    }")
      partial_sum_span <-  fn_one_fixed_match(partial_sum_text)
      operation <-  fn_one_fixed_match(operation_text)
      if (operation[1] <= partial_sum_span[1] || sum(operation) > sum(partial_sum_span)) {
            stop("The supported operation is not inside the partial-sum function.")
      }
      sum_call_text <-  paste0(
            "target += reduce_sum_static(partial_sum_people_lupmf, y, chunk_size, covariates, ",
            "basis_1, basis_2, alpha,\n",
            "                                beta, weights);")
      fn_one_fixed_match(sum_call_text)

      operation_name <-  "nico_bernoulli_logit_glm_with_a_tensor_product_basis_lpmf"
      declaration <-  paste0(
            "\n    real ",
            operation_name,
            "(array[] int y,\n",
            "           data matrix x,\n",
            "           real alpha,\n",
            "           vector beta,\n",
            "           data matrix basis_1,\n",
            "           data matrix basis_2,\n",
            "           matrix weights);\n")
      replacement <-  paste0(
            "return nico_bernoulli_logit_glm_with_a_tensor_product_basis_lupmf(\n",
            "              y_slice | covariates[start:end], alpha, beta,\n",
            "              basis_1[start:end], basis_2[start:end], weights);")
      edits <-  list(list(span = operation,
                          text = replacement),
                     list(span = c(start = sum(functions_span),
                                    length = 0L),
                          text = declaration))
      line_ending <-  if (grepl("\r\n",
                               rawToChar(source_bytes),
                               fixed = TRUE)) "\r\n" else "\n"
      output_bytes <-  source_bytes
      for (edit in edits) {
            first <-  unname(edit$span[1])
            count <-  unname(edit$span[2])
            before <-  if (first > 1L) output_bytes[seq_len(first - 1L)] else raw(0)
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
      on.exit({
            try(close(output_connection),
                 silent = TRUE)
            if (file.exists(temporary_path)) unlink(temporary_path)
      }, add = TRUE)
      writeBin(output_bytes,
                output_connection)
      close(output_connection)
      if (!file.link(temporary_path,
                     output_path)) {
            stop("The generated source could not be written without replacing a file.")
      }
      return(list(source_Stan_model_file_path = source_path,
                  source_Stan_model_md5 = unname(tools::md5sum(source_path)),
                  NicoSIMD_fused_Stan_model_file_path = output_path,
                  fused_calls = operation_name,
                  source_for_remaining_backend_conversion = output_path))

}
























