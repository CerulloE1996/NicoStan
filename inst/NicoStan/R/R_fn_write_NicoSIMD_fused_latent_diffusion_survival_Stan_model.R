##
## ---- writes the latent diffusion survival nico::math-fused source from the Stan source:
##

fn_write_NicoSIMD_fused_latent_diffusion_survival_Stan_model <-  function(
      source_Stan_model_file_path,
      NicoSIMD_fused_Stan_model_file_path
) {

      source_path <-  normalizePath(source_Stan_model_file_path, mustWork = TRUE)
      output_directory <-  normalizePath(dirname(NicoSIMD_fused_Stan_model_file_path), mustWork = TRUE)
      output_path <-  file.path(output_directory, basename(NicoSIMD_fused_Stan_model_file_path))
      source_names <-  paste0("latent_diffusion_survival_chunked_no_loops_over_N_v3_with_the_fused_",
                              "proportional_hazards_node", c("", "_automatic_reduce_sum_chunked"),
                              "_Stan_math_for_layout_timings.stan")
      if (!(basename(source_path) %in% source_names)) {
            stop("The latent diffusion survival source is not a supported Stan reference.")
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

      source_text <-  rawToChar(source_bytes)
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

      fn_number_of_matches <-  function(pattern) {

            positions <-  gregexpr(pattern, masked, perl = TRUE, useBytes = TRUE)[[1]]
            return(sum(positions > 0L))

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
      transformed_data_block <-  fn_block("(?<![A-Za-z0-9_])transformed\\s+data\\s*\\{")
      transformed_parameters_block <-  fn_block("(?<![A-Za-z0-9_])transformed\\s+parameters\\s*\\{")
      model_block <-  fn_block("(?<![A-Za-z0-9_])model\\s*\\{")

      ##
      ## ---- the chunk's likelihood: one call of the fused proportional-hazards node:
      ##

      ## (the source in blocks: each chunk one block of chunk_size subjects, block_first sliced, chunk size 1)
      pattern_of_the_partial_sum_in_blocks <-  paste0("(?<![A-Za-z0-9_])", fn_tokens(paste0(
            "real partial_sum_subjects_lpmf(array[] int block_first_slice, int start, int end, ",
            "data array[] int block_last, data array[] int event, ",
            "data matrix covariates, data vector weights_of_the_grids_at_t, ",
            "data array[] int columns_of_the_grids_at_t, data array[] int events_before, ",
            "data vector weights_of_the_hazard_grid_at_the_events, ",
            "data array[] int columns_of_the_hazard_grid_at_the_events, ",
            "vector hazard_grids, vector beta) { ",
            "int first = block_first_slice[1]; ",
            "int last = block_last[end]; ",
            "int n_subjects = last - first + 1; ",
            "vector[n_subjects] linear_predictor = covariates[first:last] * beta; ",
            "vector[n_subjects] cumulative_baseline_hazard_at_t ",
            "= csr_matrix_times_vector(n_subjects, rows(hazard_grids), ",
            "weights_of_the_grids_at_t[(3 * first - 2):(3 * last)], ",
            "columns_of_the_grids_at_t[(3 * first - 2):(3 * last)], ",
            "linspaced_int_array(n_subjects + 1, 1, 3 * n_subjects + 1), hazard_grids); ",
            "real lp = dot_product(to_vector(event[first:last]), linear_predictor) ",
            "- dot_product(exp(linear_predictor), cumulative_baseline_hazard_at_t); ",
            "int e1 = events_before[first] + 1; ",
            "int e2 = events_before[last + 1]; ",
            "if (e2 >= e1) { ",
            "int n_events = e2 - e1 + 1; ",
            "vector[n_events] baseline_hazard_at_the_events ",
            "= csr_matrix_times_vector(n_events, rows(hazard_grids), ",
            "weights_of_the_hazard_grid_at_the_events[(2 * e1 - 1):(2 * e2)], ",
            "columns_of_the_hazard_grid_at_the_events[(2 * e1 - 1):(2 * e2)], ",
            "linspaced_int_array(n_events + 1, 1, 2 * n_events + 1), hazard_grids); ",
            "lp += sum(log(baseline_hazard_at_the_events)); ",
            "} ",
            "return lp; }")))
      source_in_blocks <-  fn_number_of_matches(pattern_of_the_partial_sum_in_blocks) == 1L
      partial_sum <-  if (source_in_blocks) fn_one_match(pattern_of_the_partial_sum_in_blocks) else
      fn_one_match(paste0("(?<![A-Za-z0-9_])", fn_tokens(paste0(
            "real partial_sum_subjects_lpmf(array[] int event_slice, int start, int end, ",
            "data matrix covariates, data vector weights_of_the_grids_at_t, ",
            "data array[] int columns_of_the_grids_at_t, data array[] int events_before, ",
            "data vector weights_of_the_hazard_grid_at_the_events, ",
            "data array[] int columns_of_the_hazard_grid_at_the_events, ",
            "vector hazard_grids, vector beta) { ",
            "int n_subjects = end - start + 1; ",
            "vector[n_subjects] linear_predictor = covariates[start:end] * beta; ",
            "vector[n_subjects] cumulative_baseline_hazard_at_t ",
            "= csr_matrix_times_vector(n_subjects, rows(hazard_grids), ",
            "weights_of_the_grids_at_t[(3 * start - 2):(3 * end)], ",
            "columns_of_the_grids_at_t[(3 * start - 2):(3 * end)], ",
            "linspaced_int_array(n_subjects + 1, 1, 3 * n_subjects + 1), hazard_grids); ",
            "real lp = dot_product(to_vector(event_slice), linear_predictor) ",
            "- dot_product(exp(linear_predictor), cumulative_baseline_hazard_at_t); ",
            "int e1 = events_before[start] + 1; ",
            "int e2 = events_before[end + 1]; ",
            "if (e2 >= e1) { ",
            "int n_events = e2 - e1 + 1; ",
            "vector[n_events] baseline_hazard_at_the_events ",
            "= csr_matrix_times_vector(n_events, rows(hazard_grids), ",
            "weights_of_the_hazard_grid_at_the_events[(2 * e1 - 1):(2 * e2)], ",
            "columns_of_the_hazard_grid_at_the_events[(2 * e1 - 1):(2 * e2)], ",
            "linspaced_int_array(n_events + 1, 1, 2 * n_events + 1), hazard_grids); ",
            "lp += sum(log(baseline_hazard_at_the_events)); ",
            "} ",
            "return lp; }"))))
      if (!fn_inside(partial_sum, functions_block)) {
            stop("The partial sum is not in the functions block.")
      }
      text_of_the_partial_sum_in_blocks <-  paste0(
                  "real partial_sum_subjects_lpmf(array[] int block_first_slice, int start, int end,\n",
                  strrep(" ", 35), "data array[] int block_last, data array[] int event_in_time_order,\n",
                  strrep(" ", 35), "data matrix covariates_in_time_order,\n",
                  strrep(" ", 35), "data array[] int cell_in_time_order,\n",
                  strrep(" ", 35), "data vector time_into_cell_in_time_order, data vector cell_widths,\n",
                  strrep(" ", 35), "vector baseline_hazard_grid, vector beta) {\n",
                  "        int first = block_first_slice[1];\n",
                  "        int last = block_last[end];\n",
                  "        return nico_proportional_hazards_log_likelihood_with_a_piecewise_linear_baseline_",
                  "hazard_on_a_grid(\n",
                  strrep(" ", 21), "event_in_time_order[first:last], covariates_in_time_order[first:last], ",
                  "beta,\n",
                  strrep(" ", 21), "cell_in_time_order[first:last], time_into_cell_in_time_order[first:last], ",
                  "cell_widths,\n",
                  strrep(" ", 21), "baseline_hazard_grid);\n",
                  "    }")
      edits <-  if (source_in_blocks) {
            list(list(span = partial_sum, text = text_of_the_partial_sum_in_blocks))
      } else
      list(list(
            span = partial_sum,
            text = paste0(
                  "real partial_sum_subjects_lpmf(array[] int event_slice, int start, int end,\n",
                  strrep(" ", 35), "data matrix covariates_in_time_order,\n",
                  strrep(" ", 35), "data array[] int cell_in_time_order,\n",
                  strrep(" ", 35), "data vector time_into_cell_in_time_order, data vector cell_widths,\n",
                  strrep(" ", 35), "vector baseline_hazard_grid, vector beta) {\n",
                  "        return nico_proportional_hazards_log_likelihood_with_a_piecewise_linear_baseline_",
                  "hazard_on_a_grid(\n",
                  strrep(" ", 21), "event_slice, covariates_in_time_order[start:end], beta, ",
                  "cell_in_time_order[start:end],\n",
                  strrep(" ", 21), "time_into_cell_in_time_order[start:end], cell_widths, ",
                  "baseline_hazard_grid);\n",
                  "    }")))

      ##
      ## ---- the sparse interpolation rows (used only by the partial sum) give way to the subjects in time order:
      ##

      sparse_rows <-  fn_one_match(paste0("(?<![A-Za-z0-9_])", fn_tokens(paste0(
            "vector[3 * N] weights_of_the_grids_at_t ",
            "= to_vector(append_row(append_row(rep_row_vector(1, N), weight_of_lower_grid_point'), ",
            "weight_of_upper_grid_point')); ",
            "array[3 * N] int columns_of_the_grids_at_t ",
            "= to_int(to_array_1d(to_vector(append_row(append_row(to_row_vector(cell_index), ",
            "M + 1 + to_row_vector(cell_index)), ",
            "M + 1 + to_row_vector(cell_index_upper))))); ",
            "vector[2 * N_event] weights_of_the_hazard_grid_at_the_events ",
            "= to_vector(append_row(one_minus_event_cell_fraction', event_cell_fraction')); ",
            "array[2 * N_event] int columns_of_the_hazard_grid_at_the_events ",
            "= to_int(to_array_1d(to_vector(append_row(M + 1 + to_row_vector(event_cell_index), ",
            "M + 1 + to_row_vector(event_cell_index_upper)))));"))))
      if (!fn_inside(sparse_rows, transformed_data_block)) {
            stop("The sparse interpolation rows are not in the transformed data.")
      }
      ## (their comment, just before them, goes with them)
      comment_of_the_sparse_rows <-  gregexpr(
            paste0("// the sparse interpolation matrices, as the entries of their rows in the subjects' ",
                   "(and the events') order:"),
            source_text, fixed = TRUE, useBytes = TRUE)[[1]]
      if (length(comment_of_the_sparse_rows) != 1L || comment_of_the_sparse_rows[1] < 1L) {
            stop("The comment of the sparse interpolation rows is missing or ambiguous.")
      }
      comment_start <-  comment_of_the_sparse_rows[1]
      if (comment_start >= sparse_rows["start"] ||
          grepl("[^[:space:]]", substr(masked, comment_start, sparse_rows["start"] - 1L))) {
            stop("The comment of the sparse interpolation rows is not just before them.")
      }
      edits[[length(edits) + 1L]] <-  list(
            span = c(start = comment_start,
                     length = unname(sparse_rows["start"] + sparse_rows["length"] - comment_start)),
            text = paste0(
                  "// the subjects in time order (so in cell order) for the fused node, with each one's time ",
                  "into its cell\n",
                  "    // (within [0, dt], the bounds of its cell, against rounding) and the widths of the ",
                  "cells\n",
                  "    array[N] int subjects_in_time_order = sort_indices_asc(t);\n",
                  "    array[N] int event_in_time_order = event[subjects_in_time_order];\n",
                  "    matrix[N, K] covariates_in_time_order = covariates[subjects_in_time_order];\n",
                  "    array[N] int cell_in_time_order = cell_index[subjects_in_time_order];\n",
                  "    vector[N] time_into_cell_in_time_order\n",
                  "          = fmax(0, fmin(dt, t[subjects_in_time_order]\n",
                  "                             - (to_vector(cell_in_time_order) - 1) * dt));\n",
                  "    vector[M] cell_widths = rep_vector(dt, M);"))

      ##
      ## ---- the Euler-Maruyama path: one call of nico_AR1_recursion (d[k] = a d[k - 1] + shocks[k], d[0] = x_0):
      ##

      path_loop <-  fn_one_match(paste0("(?<![A-Za-z0-9_])", fn_tokens(paste0(
            "x_path[1] = x_0; ",
            "for (k in 1:M) { ",
            "x_path[k + 1] = one_minus_kappa_dt * x_path[k] + increment_without_the_path[k]; ",
            "}"))))
      if (!fn_inside(path_loop, transformed_parameters_block)) {
            stop("The loop of the path is not in the transformed parameters.")
      }
      edits[[length(edits) + 1L]] <-  list(
            span = path_loop,
            text = paste0("x_path = append_row(x_0, nico_AR1_recursion(increment_without_the_path, x_0, ",
                          "one_minus_kappa_dt));"))

      ##
      ## ---- the model's call of the partial sum, on the baseline hazard grid alone:
      ##

      grids_before_the_call <-  paste0(
            "vector[M + 1] cumulative_baseline_hazard_grid; ",
            "cumulative_baseline_hazard_grid[1] = 0; ",
            "cumulative_baseline_hazard_grid[2:(M + 1)] = cumulative_sum(0.5 * dt * (baseline_hazard_grid[1:M] ",
            "+ baseline_hazard_grid[2:(M + 1)])); ",
            "vector[2 * (M + 1)] hazard_grids = append_row(cumulative_baseline_hazard_grid, ",
            "baseline_hazard_grid); ")
      ## (in blocks: block_first sliced, a chunk size of 1 block, block_last with the data)
      call_of_the_partial_sum <-  if (source_in_blocks) fn_one_match(paste0("(?<![A-Za-z0-9_])", fn_tokens(paste0(
            grids_before_the_call,
            "target += reduce_sum_static(partial_sum_subjects_lupmf, block_first, 1, block_last, event, ",
            "covariates, weights_of_the_grids_at_t, columns_of_the_grids_at_t, events_before, ",
            "weights_of_the_hazard_grid_at_the_events, ",
            "columns_of_the_hazard_grid_at_the_events, hazard_grids, beta);")))) else
      fn_one_match(paste0("(?<![A-Za-z0-9_])", fn_tokens(paste0(
            grids_before_the_call,
            "target += reduce_sum_static(partial_sum_subjects_lupmf, event, chunk_size, covariates, ",
            "weights_of_the_grids_at_t, columns_of_the_grids_at_t, events_before, ",
            "weights_of_the_hazard_grid_at_the_events, ",
            "columns_of_the_hazard_grid_at_the_events, hazard_grids, beta);"))))
      if (!fn_inside(call_of_the_partial_sum, model_block)) {
            stop("The call of the partial sum is not in the model block.")
      }
      text_of_the_call <-  if (source_in_blocks) {
            paste0(
                  "// (one block of chunk_size subjects per chunk: reduce_sum_static's chunk size is 1 block)\n",
                  "      target += reduce_sum_static(partial_sum_subjects_lupmf, block_first, 1, block_last,\n",
                  strrep(" ", 34), "event_in_time_order, covariates_in_time_order, cell_in_time_order,\n",
                  strrep(" ", 34), "time_into_cell_in_time_order, cell_widths, baseline_hazard_grid, beta);")
      } else {
            paste0(
                  "target += reduce_sum_static(partial_sum_subjects_lupmf, event_in_time_order, chunk_size,\n",
                  strrep(" ", 34), "covariates_in_time_order, cell_in_time_order,\n",
                  strrep(" ", 34), "time_into_cell_in_time_order, cell_widths, baseline_hazard_grid, beta);")
      }
      edits[[length(edits) + 1L]] <-  list(
            span = call_of_the_partial_sum,
            text = text_of_the_call)

      ##
      ## ---- the declarations of the existing nodes:
      ##

      declarations <-  c(
            "",
            paste0("    real nico_proportional_hazards_log_likelihood_with_a_piecewise_linear_baseline_",
                   "hazard_on_a_grid("),
            "          data array[] int event, data matrix X, vector beta, data array[] int cell,",
            "          data vector time_into_cell, data vector cell_widths, vector h);",
            "    vector nico_AR1_recursion(vector shocks, real carry, real a);")
      edits[[length(edits) + 1L]] <-  list(
            span = c(start = functions_block["opening"] + 1L, length = 0),
            text = paste(declarations, collapse = "\n"))

      ##
      ## ---- applies the byte spans and writes without replacing an existing file:
      ##

      line_ending <-  if (grepl("\r\n", source_text, fixed = TRUE)) "\r\n" else "\n"
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
            fused_calls = c(paste0("nico_proportional_hazards_log_likelihood_with_a_piecewise_linear_",
                                   "baseline_hazard_on_a_grid"),
                            "nico_AR1_recursion"),
            source_for_remaining_backend_conversion = output_path))

}






















