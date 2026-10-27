##
## ---- writes the PK nico::math-fused source from the Stan source:
##

fn_write_NicoSIMD_fused_PK_Stan_model <-  function(
      source_Stan_model_file_path,
      NicoSIMD_fused_Stan_model_file_path
) {

      source_path <-  normalizePath(source_Stan_model_file_path, mustWork = TRUE)
      output_directory <-  normalizePath(dirname(NicoSIMD_fused_Stan_model_file_path), mustWork = TRUE)
      output_path <-  file.path(output_directory, basename(NicoSIMD_fused_Stan_model_file_path))
      source_names <-  paste0("two_compartment_population_pharmacokinetics_chunked_no_loops_over_N_v3",
                              c("", "_automatic_reduce_sum_chunked"),
                              "_Stan_math_for_layout_timings.stan")
      if (!(basename(source_path) %in% source_names)) {
            stop("The PK source is not a supported Stan reference.")
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

      fn_all_matches <-  function(pattern) {

            positions <-  gregexpr(pattern, masked, perl = TRUE, useBytes = TRUE)[[1]]
            if (positions[1] < 1L) stop("The source span is missing.")
            return(lapply(seq_along(positions), function(k) {

                  c(start = positions[k], length = attr(positions, "match.length")[k])

            }))

      }

      functions_block <-  fn_block("(?<![A-Za-z0-9_])functions\\s*\\{")
      n_partial_sums <-  if (grepl("_automatic_reduce_sum_chunked_", basename(source_path), fixed = TRUE)) {
            2L
      } else 1L

      ##
      ## ---- the concentrations from the rates: nico_two_compartment_first_order_input_exponents_and_coefficients
      ## ---- and nico_sum_of_exponentials_by_group:
      ##

      concentrations <-  fn_block(paste0("(?<![A-Za-z0-9_])vector\\s+",
                                         "two_compartment_oral_concentrations_from_log_rates\\s*\\([^{]*\\{"))
      concentrations_start <-  fn_one_match(paste0("(?<![A-Za-z0-9_])vector\\s+",
                                                   "two_compartment_oral_concentrations_from_log_rates\\s*\\("))
      concentrations_span <-  c(start = unname(concentrations_start["start"]),
                                length = unname(concentrations["closing"] - concentrations_start["start"] + 1L))
      if (!fn_inside(concentrations_span, functions_block)) {
            stop("The concentrations function is not in its reference functions block.")
      }
      concentrations_code <-  gsub("[[:space:]]", "",
                                   substr(masked, concentrations_span["start"],
                                          concentrations_span["start"] + concentrations_span["length"] - 1L))
      concentrations_code_file <-  tempfile()
      writeLines(concentrations_code, concentrations_code_file, sep = "")
      if (!identical(unname(tools::md5sum(concentrations_code_file)), "3a07278bcbf666d55ad5dc48890fc31d")) {
            stop("The concentrations function is not the reference one.")
      }
      unlink(concentrations_code_file)
      concentrations_from_rates <-  c(
            "    vector two_compartment_oral_concentrations_from_rates(matrix rates_by_patient,",
            paste0("                                                          data array[] int patient, data",
                   " vector minus_time,"),
            paste0("                                                          data vector time, data array[]",
                   " int first_sample,"),
            paste0("                                                          data array[] int last_sample,",
                   " int start, int end,"),
            "                                                          data real minimum_rate_separation) {",
            "        int N = rows(rates_by_patient);",
            "        matrix[N, 7] exponents_and_coefficients =",
            "              nico_two_compartment_first_order_input_exponents_and_coefficients(rates_by_patient,",
            paste0("                                                                                ",
                   " minimum_rate_separation);"),
            "        vector[N] ka = rates_by_patient[ : , 4];",
            "        vector[N] smallest_rate_separation = exponents_and_coefficients[ : , 7];",
            paste0("        int n_stable_form = number_of_values_below(smallest_rate_separation,",
                   " minimum_rate_separation);"),
            "        array[N] int by_smallest_rate_separation = linspaced_int_array(N, 1, N);",
            "        if (n_stable_form > 0 && n_stable_form < N) {",
            "            by_smallest_rate_separation = sort_indices_asc(smallest_rate_separation);",
            "        }",
            "        array[n_stable_form] int stable_form = by_smallest_rate_separation[1:n_stable_form];",
            "        vector[num_elements(minus_time)] concentration =",
            paste0("              nico_sum_of_exponentials_by_group(minus_time, patient,",
                   " exponents_and_coefficients[ : , 1:3],"),
            paste0("                                                append_col(exponents_and_coefficients[ : ,",
                   " 4:5], ka));"),
            "        if (n_stable_form > 0) {",
            "            vector[N] alpha = exponents_and_coefficients[ : , 4];",
            "            vector[N] beta = exponents_and_coefficients[ : , 5];",
            "            vector[N] k21_minus_beta = exponents_and_coefficients[ : , 6];",
            "            vector[N] dose_ka_over_V1 = rates_by_patient[ : , 5];",
            "            int s1 = first_sample[start];",
            "            int s2 = last_sample[end];",
            paste0("            array[N] int first_sample_in_chunk =",
                   " to_int(to_array_1d(to_vector(first_sample[start:end])"),
            "                                                                    - (s1 - 1)));",
            paste0("            array[N] int last_sample_in_chunk =",
                   " to_int(to_array_1d(to_vector(last_sample[start:end])"),
            "                                                                   - (s1 - 1)));",
            "            int n_stable_form_samples = sum(last_sample_in_chunk[stable_form])",
            "                                        - sum(first_sample_in_chunk[stable_form]) + n_stable_form;",
            "            array[2, n_stable_form_samples] int samples_and_their_patients =",
            paste0("                  samples_of_patients(first_sample_in_chunk[stable_form],",
                   " last_sample_in_chunk[stable_form]);"),
            "            array[n_stable_form_samples] int sample_on_stable_form = samples_and_their_patients[1];",
            paste0("            array[n_stable_form_samples] int patient_on_stable_form =",
                   " stable_form[samples_and_their_patients[2]];"),
            "            concentration[sample_on_stable_form] =",
            "                  dose_ka_over_V1[patient_on_stable_form]",
            "                  .* (k21_minus_beta[patient_on_stable_form]",
            "                      .* second_divided_difference_of_decay(alpha[patient_on_stable_form],",
            "                                                            beta[patient_on_stable_form],",
            "                                                            ka[patient_on_stable_form],",
            "                                                            time[s1:s2][sample_on_stable_form])",
            "                      + minus_first_divided_difference_of_decay(alpha[patient_on_stable_form],",
            "                                                                ka[patient_on_stable_form],",
            paste0("                                                               ",
                   " time[s1:s2][sample_on_stable_form]));"),
            "        }",
            "        return concentration;",
            "    }")
      edits <-  list(list(
            span = concentrations_span,
            text = sub("^    ", "", paste(concentrations_from_rates, collapse = "\n"))))

      ##
      ## ---- each partial sum's rates: nico_rows_of_mean_plus_random_effects_times_transposed_loadings:
      ##

      rates_spans <-  fn_all_matches(fn_tokens(paste0(
            "matrix[n_patients, 5] log_rates_by_patient = rep_matrix(mean_of_log_rates, n_patients) ",
            "+ z_by_patient * transposed_loadings_of_log_rates;")))
      calls <-  fn_all_matches(fn_tokens(
            "two_compartment_oral_concentrations_from_log_rates(log_rates_by_patient,"))
      if (length(rates_spans) != n_partial_sums || length(calls) != n_partial_sums ||
          !all(vapply(c(rates_spans, calls), fn_inside, logical(1), block = functions_block))) {
            stop("The rates of the partial sums are not those of the reference.")
      }
      for (span in rates_spans) {
            edits[[length(edits) + 1L]] <-  list(
                  span = span,
                  text = paste0(
                        "matrix[n_patients, 5] rates_by_patient =\n",
                        "              nico_rows_of_mean_plus_random_effects_times_transposed_loadings(",
                        "z_by_patient, mean_of_log_rates,\n",
                        "                                                                              ",
                        "transposed_loadings_of_log_rates,\n",
                        "                                                                              1);"))
      }
      for (span in calls) {
            edits[[length(edits) + 1L]] <-  list(
                  span = span,
                  text = "two_compartment_oral_concentrations_from_rates(rates_by_patient,")
      }

      ##
      ## ---- the declarations of the existing nodes:
      ##

      declarations <-  c(
            "",
            paste0("    vector nico_sum_of_exponentials_by_group(data vector t, data array[] int group, ",
                   "matrix A, matrix B);"),
            "    matrix nico_two_compartment_first_order_input_exponents_and_coefficients(matrix rates,",
            paste0("                                                                              ",
                   "data real minimum_rate_separation);"),
            paste0("    matrix nico_rows_of_mean_plus_random_effects_times_transposed_loadings(",
                   "matrix z, row_vector mean,"),
            paste0("                                                                           ",
                   "matrix transposed_loadings,"),
            paste0("                                                                           ",
                   "data int exponentiate);"))
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
            fused_calls = c("nico_two_compartment_first_order_input_exponents_and_coefficients",
                            "nico_sum_of_exponentials_by_group",
                            "nico_rows_of_mean_plus_random_effects_times_transposed_loadings"),
            source_for_remaining_backend_conversion = output_path))

}






















