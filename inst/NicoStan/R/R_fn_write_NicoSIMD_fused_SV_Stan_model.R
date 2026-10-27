##
## ---- writes the stochastic volatility nico::math-fused source from the Stan source:
##

fn_write_NicoSIMD_fused_SV_Stan_model <-  function(
      source_Stan_model_file_path,
      NicoSIMD_fused_Stan_model_file_path
) {

      source_path <-  normalizePath(source_Stan_model_file_path, mustWork = TRUE)
      output_directory <-  normalizePath(dirname(NicoSIMD_fused_Stan_model_file_path), mustWork = TRUE)
      output_path <-  file.path(output_directory, basename(NicoSIMD_fused_Stan_model_file_path))
      source_names <-  paste0("stochastic_volatility_chunked_no_loops_over_N_fewest_main_tape_copies",
                              c("", "_automatic_reduce_sum_chunked"), "_Stan_math_for_layout_timings.stan")
      if (!(basename(source_path) %in% source_names)) {
            stop("The stochastic volatility source is not a supported Stan reference.")
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

      ##
      ## ---- the AR(1) path in sub-blocks: nico_AR1_recursion:
      ##

      recursion <-  fn_one_match(paste0("(?<![A-Za-z0-9_])", fn_tokens(paste0(
            "vector fn_AR1_recursion_in_sub_blocks_without_a_loop(vector shocks, real carry, real a, ",
            "int sub_block_length, data array[] int index_of_recursion_matrix) { ",
            "int L = num_elements(shocks); ",
            "int P = sub_block_length; ",
            "int n_sub = (L + P - 1) %/% P; ",
            "vector[P + 1] a_power = append_row(1, append_row(a, pow(a, linspaced_vector(P - 1, 2, P)))); ",
            "matrix[P, P] recursion_matrix = to_matrix(append_row(a_power[1:P], 0)[index_of_recursion_matrix], ",
            "P, P); ",
            "matrix[P, n_sub] shocks_by_sub_block = to_matrix(append_row(shocks, rep_vector(0, n_sub * P - L)), ",
            "P, n_sub); ",
            "row_vector[n_sub] carry_into_sub_block; ",
            "vector[n_sub * P] d_padded; ",
            "carry_into_sub_block[1] = carry; ",
            "if (n_sub > 1) { ",
            "row_vector[n_sub - 1] end_of_sub_block_from_zero = reverse(a_power[1:P])' ",
            "* shocks_by_sub_block[, 1:(n_sub - 1)]; ",
            "carry_into_sub_block[2:n_sub] = fn_AR1_recursion_in_sub_blocks_without_a_loop(",
            "end_of_sub_block_from_zero', carry, a_power[P + 1], P, index_of_recursion_matrix)'; ",
            "} ",
            "shocks_by_sub_block[1] = shocks_by_sub_block[1] + a * carry_into_sub_block; ",
            "d_padded = to_vector(recursion_matrix * shocks_by_sub_block); ",
            "return d_padded[1:L]; }"))))
      if (!fn_inside(recursion, functions_block)) {
            stop("The AR(1) path is not in the functions block.")
      }
      edits <-  list(list(
            span = recursion,
            text = paste0(
                  "vector fn_AR1_recursion_in_sub_blocks_without_a_loop(vector shocks, real carry, real a, ",
                  "int sub_block_length,\n",
                  strrep(" ", 56), "data array[] int index_of_recursion_matrix) {\n",
                  "        return nico_AR1_recursion(shocks, carry, a);\n",
                  "    }")))

      ##
      ## ---- the likelihood of the returns: nico_normal_log_scale_lupdf:
      ##

      likelihood <-  fn_one_match(paste0("(?<![A-Za-z0-9_])", fn_tokens(
            "log_density += normal_lupdf(y[first:last] | 0, exp((mu + d) / 2));")))
      if (!fn_inside(likelihood, functions_block)) {
            stop("The likelihood is not in the partial sum.")
      }
      edits[[length(edits) + 1L]] <-  list(
            span = likelihood,
            text = "log_density += nico_normal_log_scale_lupdf(y[first:last] | 0, (mu + d) / 2);")

      ##
      ## ---- the declarations of the existing nodes:
      ##

      declarations <-  c(
            "",
            "    real nico_normal_log_scale_lpdf(vector y, real mu, vector log_scale);",
            "    vector nico_AR1_recursion(vector shocks, real carry, real a);")
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
            fused_calls = c("nico_AR1_recursion", "nico_normal_log_scale_lpdf"),
            source_for_remaining_backend_conversion = output_path))

}






















