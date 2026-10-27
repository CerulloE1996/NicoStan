##
## ---- writes the Cox nico::math-fused source from the Stan source:
##

fn_write_NicoSIMD_fused_Cox_Stan_model <-  function(
      source_Stan_model_file_path,
      NicoSIMD_fused_Stan_model_file_path
) {

      source_path <-  normalizePath(source_Stan_model_file_path, mustWork = TRUE)
      output_directory <-  normalizePath(dirname(NicoSIMD_fused_Stan_model_file_path), mustWork = TRUE)
      output_path <-  file.path(output_directory, basename(NicoSIMD_fused_Stan_model_file_path))
      source_name <-  "shared_frailty_Cox_survival_chunked_no_loops_over_N_v2_Stan_math_for_layout_timings.stan"
      if (!identical(basename(source_path), source_name)) {
            stop("The Cox source is not the supported Stan reference.")
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
      block_function <-  fn_block(paste0(
            fn_tokens(paste0("vector log_sum_exp_eta_of_block_from_its_raw_log_frailties(",
                             "vector beta_and_sigma_frailty, vector log_frailty_raw_of_block, ",
                             "data array[] real X_block_by_column, data array[] int K_and_chunk_size)")),
            "\\s*\\{"))
      partial_sum_function <-  fn_block(paste0(
            fn_tokens(paste0("real partial_sum_packed_blocks_lpdf(array[] row_vector packed_of_block_slice, ",
                             "int start, int end, data matrix X_decreasing_time, ",
                             "data vector event_decreasing_time, data vector w_decreasing_time, ",
                             "data array[] int block_first, data array[] int block_last, ",
                             "data array[] int N_prior_groups_of_block, data array[] int block_size, ",
                             "data int chunk_size_of_blocks, vector beta, real sigma_frailty)")),
            "\\s*\\{"))
      if (block_function["opening"] <= functions_block["opening"] ||
          block_function["closing"] >= functions_block["closing"] ||
          partial_sum_function["opening"] <= functions_block["opening"] ||
          partial_sum_function["closing"] >= functions_block["closing"]) {
            stop("A Cox function is outside the functions block.")
      }

      ##
      ## ---- The log risk of each block (a map_rect job): nico_log_sum_exp_of_affine_predictor ---------------------------------------
      ##

      ## (the block's row holds W slots; its own observations are the first M)
      block_eta_and_return <-  fn_one_match(fn_tokens(paste0(
            "vector[M] eta = fma(beta_and_sigma_frailty[K + 1], log_frailty_raw_of_block[1:M], ",
            "to_matrix(X_block_by_column, W, K)[1:M] * beta_and_sigma_frailty[1:K]); ",
            "real largest_eta = max(eta); ",
            "return [largest_eta + log(sum(exp(eta - largest_eta)))]';")))
      if (!fn_inside(block_eta_and_return, block_function)) {
            stop("The block log risk is not in its reference function.")
      }
      edits <-  list(list(
            span = block_eta_and_return,
            text = paste0(
                  "return [nico_log_sum_exp_of_affine_predictor(to_matrix(X_block_by_column, W, K)[1:M],\n",
                  "                                                     beta_and_sigma_frailty[1:K],\n",
                  "                                                     beta_and_sigma_frailty[K + 1],\n",
                  "                                                     log_frailty_raw_of_block[1:M])]';")))

      ##
      ## ---- The partial likelihood of each slice: nico_dot_products_of_affine_predictor_and_log_cumulative_sum_exp ------------------
      ##

      slice_eta_and_lp <-  fn_one_match(paste0("(?m)^\\h*", fn_tokens(paste0(
            "vector[M] eta = fma(sigma_frailty, log_frailty_raw_of_slice, ",
            "X_decreasing_time[first:last] * beta); ",
            "real lp = dot_product(event_decreasing_time[first:last], eta);")), "\\h*\\R"))
      slice_risk <-  fn_one_match(fn_tokens(paste0(
            "lp -= weighted_sum_of_log_cumulative_sum_exp(w_decreasing_time[first:last], eta, log_offset);")))
      if (!fn_inside(slice_eta_and_lp, partial_sum_function) ||
          !fn_inside(slice_risk, partial_sum_function) ||
          slice_risk["start"] <= slice_eta_and_lp["start"]) {
            stop("The slice likelihood is not in its reference function.")
      }
      edits[[length(edits) + 1L]] <-  list(span = slice_eta_and_lp, text = "")
      edits[[length(edits) + 1L]] <-  list(
            span = slice_risk,
            text = paste0(
                  "real lp = nico_dot_products_of_affine_predictor_and_log_cumulative_sum_exp(\n",
                  "            event_decreasing_time[first:last], -w_decreasing_time[first:last],\n",
                  "            X_decreasing_time[first:last], beta, sigma_frailty, log_frailty_raw_of_slice,\n",
                  "            log_offset);"))

      ##
      ## ---- One block (no chunking): the same fused node over every observation -----------------------------------------------------
      ##

      one_block_eta_and_target <-  fn_one_match(fn_tokens(paste0(
            "vector[N] eta_decreasing_time = fma(sigma_frailty, log_frailty_raw[group_decreasing_time], ",
            "X_decreasing_time * beta); ",
            "target += dot_product(event_decreasing_time, eta_decreasing_time) ",
            "- weighted_sum_of_log_cumulative_sum_exp(w_decreasing_time, eta_decreasing_time, ",
            "negative_infinity());")))
      if (one_block_eta_and_target["start"] <= functions_block["closing"]) {
            stop("The one-block likelihood is not in the model block.")
      }
      edits[[length(edits) + 1L]] <-  list(
            span = one_block_eta_and_target,
            text = paste0(
                  "target += nico_dot_products_of_affine_predictor_and_log_cumulative_sum_exp(\n",
                  "                  event_decreasing_time, -w_decreasing_time, X_decreasing_time, beta,\n",
                  "                  sigma_frailty, log_frailty_raw[group_decreasing_time],\n",
                  "                  negative_infinity());"))

      ##
      ## ---- The scan over the block totals, and the scan of the generated quantities: nico_log_cumulative_sum_exp -------------------
      ##

      for (scan_code in c(paste0("log_risk_before_block[2:N_blocks] = ",
                                 "log_cumulative_sum_exp(log_risk_of_block, negative_infinity());"),
                          paste0("vector[N] log_cumulative_risk = ",
                                 "log_cumulative_sum_exp(eta[order_decreasing_time], negative_infinity());"))) {

            scan_span <-  fn_one_match(fn_tokens(scan_code))
            if (scan_span["start"] <= functions_block["closing"]) {
                  stop("A scan is not after the functions block.")
            }
            edits[[length(edits) + 1L]] <-  list(
                  span = scan_span,
                  text = sub("= log_cumulative_sum_exp(", "= nico_log_cumulative_sum_exp(", scan_code,
                             fixed = TRUE))

      }

      ##
      ## ---- The declarations of the three fused nodes -------------------------------------------------------------------------------
      ##

      declarations <-  c(
            "",
            "    real nico_dot_products_of_affine_predictor_and_log_cumulative_sum_exp(",
            "        vector linear_weights, vector prefix_weights, data matrix x, vector beta,",
            "        real scale, vector z, real log_offset);",
            "    real nico_log_sum_exp_of_affine_predictor(data matrix x, vector beta, real scale, vector z);",
            "    vector nico_log_cumulative_sum_exp(vector x, real log_offset);")
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
            fused_calls = c("nico_log_sum_exp_of_affine_predictor",
                            "nico_dot_products_of_affine_predictor_and_log_cumulative_sum_exp",
                            "nico_log_cumulative_sum_exp"),
            source_for_remaining_backend_conversion = output_path))

}
























