##
## ---- writes the GP regression nico::math-fused source from the Stan source:
##

fn_write_NicoSIMD_fused_GP_regression_Stan_model <-  function(
      source_Stan_model_file_path,
      NicoSIMD_fused_Stan_model_file_path
) {

      source_path <-  normalizePath(source_Stan_model_file_path, mustWork = TRUE)
      output_directory <-  normalizePath(dirname(NicoSIMD_fused_Stan_model_file_path), mustWork = TRUE)
      output_path <-  file.path(output_directory, basename(NicoSIMD_fused_Stan_model_file_path))
      source_name <-  "Gaussian_process_regression_chunked_Stan_math_for_layout_timings.stan"
      if (!identical(basename(source_path), source_name)) {
            stop("The GP regression source is not the supported Stan reference.")
      }
      if (identical(source_path, output_path) || file.exists(output_path)) {
            stop("The output must be a new file, not the source.")
      }

      source_connection <-  file(source_path, open = "rb")
      source_bytes <-  readBin(source_connection, what = "raw", n = file.info(source_path)$size)
      close(source_connection)
      if (length(source_bytes) == 0) stop("The source is empty.")

      ##
      ## ---- masks comments and strings, keeping byte positions:
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

      masked <-  rawToChar(fn_mask(source_bytes))
      if (grepl("(?<![A-Za-z0-9_])nico_[A-Za-z0-9_]+", masked, perl = TRUE)) {
            stop("The reference already contains native backend calls.")
      }

      fn_one_match <-  function(pattern) {

            positions <-  gregexpr(pattern, masked, perl = TRUE, useBytes = TRUE)[[1]]
            if (length(positions) != 1L || positions[1] < 1L) {
                  stop("The source span is missing or ambiguous.")
            }
            return(c(start = positions[1], length = attr(positions, "match.length")[1]))

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
            if (depth != 0L) stop("The source block has no closing brace.")
            return(c(opening = opening, closing = position - 1L))

      }

      fn_tokens <-  function(code) {

            tokens <-  strsplit(gsub("[[:space:]]", "", code), "", fixed = TRUE)[[1]]
            return(paste0("\\Q", tokens, "\\E", collapse = "\\s*"))

      }

      fn_inside <-  function(span,
                             block) {

            return(span["start"] > block["opening"] &&
                   span["start"] + span["length"] - 1L < block["closing"])

      }

      functions_block <-  fn_block("(?<![A-Za-z0-9_])functions\\s*\\{")
      model_block <-  fn_block("(?<![A-Za-z0-9_])model\\s*\\{")

      ##
      ## ---- the covariance, its Cholesky factor and the likelihood: nico_gp_normal_marginal_lupdf (kernel 1):
      ##

      cholesky_factor <-  fn_one_match(paste0("(?m)^\\h*", fn_tokens(paste0(
            "matrix[N, N] L_K = cholesky_decompose(add_diag(gp_exp_quad_cov(x_norm, sigma_f, lengthscale), ",
            "square(fixed_nugget_sd) + square(sigma_n)));")), "\\h*\\R"))
      likelihood <-  fn_one_match(fn_tokens(paste0(
            "{ vector[N] z = mdivide_left_tri_low(L_K, y_norm); ",
            "array[N, 2] real z_and_L_K_diagonal = to_array_2d(append_col(z, diagonal(L_K))); ",
            "target += reduce_sum_static(partial_sum_observations_lupdf, z_and_L_K_diagonal, chunk_size); }")))
      if (!fn_inside(cholesky_factor, model_block) ||
          !fn_inside(likelihood, model_block) ||
          likelihood["start"] <= cholesky_factor["start"]) {
            stop("The likelihood is not in its reference model block.")
      }
      edits <-  list(list(span = cholesky_factor, text = ""))
      edits[[length(edits) + 1L]] <-  list(
            span = likelihood,
            text = paste0(
                  "target += nico_gp_normal_marginal_lupdf(y_norm | 0, x_norm, 1, sigma_f, lengthscale,\n",
                  "                                            square(fixed_nugget_sd) + square(sigma_n));"))

      ##
      ## ---- the declaration of the existing node:
      ##

      declarations <-  c(
            "",
            "    real nico_gp_normal_marginal_lpdf(vector y, real mu, data array[] real x, data int kernel,",
            "                                      real sigma, real length_scale, real diagonal);")
      edits[[length(edits) + 1L]] <-  list(
            span = c(start = functions_block["opening"] + 1L, length = 0),
            text = paste(declarations, collapse = "\n"))

      ##
      ## ---- applies the byte spans and writes without replacing an existing file:
      ##

      line_ending <-  if (grepl("\r\n", rawToChar(source_bytes), fixed = TRUE)) "\r\n" else "\n"
      order_of_edits <-  order(vapply(edits, function(edit) {

            unname(edit$span[1])

      }, numeric(1)), decreasing = TRUE)
      output_bytes <-  source_bytes
      for (index in order_of_edits) {
            edit <-  edits[[index]]
            first <-  unname(edit$span[1])
            count <-  unname(edit$span[2])
            before <-  if (first > 1L) output_bytes[seq_len(first - 1L)] else raw(0)
            after <-  if (first + count <= length(output_bytes)) {
                  output_bytes[seq.int(first + count, length(output_bytes))]
            } else raw(0)
            inserted <-  gsub("\n", line_ending, edit$text, fixed = TRUE)
            output_bytes <-  c(before, charToRaw(inserted), after)
      }
      temporary_path <-  tempfile(pattern = ".NicoSIMD_fused_", tmpdir = output_directory)
      output_connection <-  file(temporary_path, open = "wb")
      on.exit({
            try(close(output_connection), silent = TRUE)
            if (file.exists(temporary_path)) unlink(temporary_path)
      }, add = TRUE)
      writeBin(output_bytes, output_connection)
      close(output_connection)
      if (!file.link(temporary_path, output_path)) {
            stop("The generated source could not be written without replacing a file.")
      }
      return(list(
            NicoSIMD_fused_Stan_model_file_path = output_path,
            fused_calls = c("nico_gp_normal_marginal_lpdf"),
            source_for_remaining_backend_conversion = output_path))

}






















