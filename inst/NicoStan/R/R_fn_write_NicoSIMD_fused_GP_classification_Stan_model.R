##
## ---- writes the existing fused latent Gaussian-process operation:
##

fn_write_NicoSIMD_fused_GP_classification_Stan_model <-  function(
      source_Stan_model_file_path,
      NicoSIMD_fused_Stan_model_file_path
) {

      source_path <-  normalizePath(source_Stan_model_file_path,
                                     mustWork = TRUE)
      output_directory <-  normalizePath(dirname(NicoSIMD_fused_Stan_model_file_path),
                                          mustWork = TRUE)
      output_path <-  file.path(output_directory,
                                 basename(NicoSIMD_fused_Stan_model_file_path))
      source_name <-  "Gaussian_process_classification_chunked_Stan_math_for_layout_timings.stan"
      if (!identical(basename(source_path),
                     source_name)) {
            stop("The source is not the supported Gaussian-process classification reference.")
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

      functions_span <-  fn_one_match("(?m)^functions[[:space:]]*\\{")
      model_block <-  fn_block("(?m)^model[[:space:]]*\\{")
      parameters_block <-  fn_block("(?m)^parameters[[:space:]]*\\{")
      parameter_text <-  substr(masked,
                                 parameters_block[1] + 1L,
                                 parameters_block[2] - 1L)
      expected_parameters <-  paste0("vector[N]eta;realmu;real<lower=0>lengthscale;",
                                      "real<lower=0>sigma_f;")
      if (!identical(gsub("[[:space:]]",
                          "",
                          parameter_text),
                     expected_parameters)) {
            stop("The parameter declarations differ from the supported reference.")
      }
      operation <-  fn_one_match(paste0(
            "matrix[[:space:]]*\\[[[:space:]]*N[[:space:]]*,[[:space:]]*N[[:space:]]*\\]",
            "[[:space:]]+L_K[[:space:]]*=[[:space:]]*cholesky_decompose[[:space:]]*\\(",
            "[[:space:]]*add_diag[[:space:]]*\\([[:space:]]*gp_exp_quad_cov[[:space:]]*\\(",
            "[[:space:]]*x[[:space:]]*,[[:space:]]*sigma_f[[:space:]]*,[[:space:]]*lengthscale",
            "[[:space:]]*\\)[[:space:]]*,[[:space:]]*1e-8[[:space:]]*\\)[[:space:]]*\\)",
            "[[:space:]]*;[[:space:]]*array[[:space:]]*\\[[[:space:]]*N[[:space:]]*\\][[:space:]]*real",
            "[[:space:]]+f[[:space:]]*=[[:space:]]*to_array_1d[[:space:]]*\\([[:space:]]*mu[[:space:]]*\\+",
            "[[:space:]]*L_K[[:space:]]*\\*[[:space:]]*eta[[:space:]]*\\)[[:space:]]*;"))
      if (operation[1] <= model_block[1] || sum(operation) > model_block[2]) {
            stop("The supported operation is not inside the model block.")
      }
      uses_of_factor <-  gregexpr("(?<![A-Za-z0-9_])L_K(?![A-Za-z0-9_])",
                                   masked,
                                   perl = TRUE)[[1]]
      if (length(uses_of_factor) != 2L) stop("The factor is used outside the supported operation.")
      fn_one_match(paste0("reduce_sum_static[[:space:]]*\\([[:space:]]*partial_sum_observations_lupdf",
                          "[[:space:]]*,[[:space:]]*f[[:space:]]*,",
                          "[[:space:]]*chunk_size[[:space:]]*,[[:space:]]*y[[:space:]]*,[[:space:]]*eta",
                          "[[:space:]]*\\)"))

      operation_name <-  "nico_gp_cholesky_factor_of_covariance_plus_diagonal_times_vector"
      declaration <-  paste0(
            "\n    vector ",
            operation_name,
            "(data array[] real x,\n",
            "           data int kernel,\n",
            "           real sigma,\n",
            "           real length_scale,\n",
            "           real diagonal,\n",
            "           vector z);\n")
      replacement <-  paste0("array[N] real f = to_array_1d(mu + ",
                              operation_name,
                              "(\n",
                              "                          x, 1, sigma_f, lengthscale, 1e-8, eta));")
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
      return(list(NicoSIMD_fused_Stan_model_file_path = output_path,
                  fused_calls = operation_name,
                  source_for_remaining_backend_conversion = output_path))

}






















