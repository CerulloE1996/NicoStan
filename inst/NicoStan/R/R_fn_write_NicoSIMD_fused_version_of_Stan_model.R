##
## ---- writes the supported Weibull NicoSIMD fused calls:
##


fn_write_NicoSIMD_fused_version_of_Stan_model <-  function( source_Stan_model_file_path,
                                                             NicoSIMD_fused_Stan_model_file_path
) {

      ##
      ## ---- checks the source and output paths:
      ##

      if (!base::file.exists(source_Stan_model_file_path)) {
            base::stop("The source Stan file does not exist.")
      }

      if (!base::dir.exists(base::dirname(NicoSIMD_fused_Stan_model_file_path))) {
            base::stop("The output directory does not exist.")
      }

      source_path <-  base::normalizePath(source_Stan_model_file_path,
                                           winslash = "/",
                                           mustWork = TRUE)
      output_directory <-  base::normalizePath(
            base::dirname(NicoSIMD_fused_Stan_model_file_path),
            winslash = "/",
            mustWork = TRUE)
      output_path <-  base::file.path(output_directory,
                                      base::basename(NicoSIMD_fused_Stan_model_file_path))

      if (base::identical(source_path, output_path)) {
            base::stop("The source and output paths must differ.")
      }

      if (base::file.exists(output_path)) {
            base::stop("The output file already exists.")
      }

      if (base::identical(base::basename(source_path),
            "Gaussian_process_classification_chunked_Stan_math_for_layout_timings.stan")) {
            classification_writer <-  base::get0(
                  "fn_write_NicoSIMD_fused_GP_classification_Stan_model",
                  envir = base::environment(),
                  mode = "function",
                  inherits = TRUE)
            if (base::is.null(classification_writer)) {
                  base::stop("The guarded Gaussian-process classification source writer has not been loaded.")
            }
            return(classification_writer(
                  source_Stan_model_file_path = source_path,
                  NicoSIMD_fused_Stan_model_file_path = output_path))
      }

      if (base::identical(base::basename(source_path), base::paste0(
            "four_class_LC_MVOP_no_loops_over_N_or_standardisation_profiles_",
            "Stan_math_for_layout_timings.stan"))) {
            four_class_writer <-  base::get0(
                  "fn_write_NicoSIMD_fused_four_class_LC_MVOP_Stan_model",
                  envir = base::environment(),
                  mode = "function",
                  inherits = TRUE)
            if (base::is.null(four_class_writer)) {
                  base::stop("The guarded four-class source writer has not been loaded.")
            }
            return(four_class_writer(
                  source_Stan_model_file_path = source_path,
                  NicoSIMD_fused_Stan_model_file_path = output_path))
      }

      if (base::identical(base::basename(source_path), base::paste0(
            "spatial_logistic_regression_with_a_Hilbert_space_approximate_Gaussian_process_",
            "chunked_with_a_tensor_product_basis_Stan_math_for_layout_timings.stan"))) {
            HSGP_writer <-  base::get0(
                  "fn_write_NicoSIMD_fused_HSGP_Stan_model",
                  envir = base::environment(),
                  mode = "function",
                  inherits = TRUE)
            if (base::is.null(HSGP_writer)) {
                  base::stop("The guarded spatial Gaussian-process source writer has not been loaded.")
            }
            return(HSGP_writer(
                  source_Stan_model_file_path = source_path,
                  NicoSIMD_fused_Stan_model_file_path = output_path))
      }

      if (base::basename(source_path) %in% base::paste0(
            "two_compartment_population_pharmacokinetics_chunked_no_loops_over_N_v3",
            c("", "_automatic_reduce_sum_chunked"),
            "_Stan_math_for_layout_timings.stan")) {
            PK_writer <-  base::get0(
                  "fn_write_NicoSIMD_fused_PK_Stan_model",
                  envir = base::environment(),
                  mode = "function",
                  inherits = TRUE)
            if (base::is.null(PK_writer)) {
                  base::stop("The guarded PK source writer has not been loaded.")
            }
            return(PK_writer(
                  source_Stan_model_file_path = source_path,
                  NicoSIMD_fused_Stan_model_file_path = output_path))
      }

      if (base::basename(source_path) %in% base::paste0(
            "stochastic_volatility_chunked_no_loops_over_N_fewest_main_tape_copies",
            c("", "_automatic_reduce_sum_chunked"),
            "_Stan_math_for_layout_timings.stan")) {
            SV_writer <-  base::get0(
                  "fn_write_NicoSIMD_fused_SV_Stan_model",
                  envir = base::environment(),
                  mode = "function",
                  inherits = TRUE)
            if (base::is.null(SV_writer)) {
                  base::stop("The guarded stochastic volatility source writer has not been loaded.")
            }
            return(SV_writer(
                  source_Stan_model_file_path = source_path,
                  NicoSIMD_fused_Stan_model_file_path = output_path))
      }

      ##
      ## ---- uses the guarded longitudinal and survival writer -----------------------------------------------------------------------
      ##

      if (base::basename(source_path) %in% c(
            base::paste0("joint_longitudinal_survival_chunked_no_loops_over_N_v5_",
                         "with_normal_lupdf_Stan_math_for_layout_timings.stan"),
            "JLS_normal_lupdf_Stan_math_generated.stan")) {
            JLS_writer <-  base::get0(
                  "fn_write_NicoSIMD_fused_JLS_Stan_model",
                  envir = base::environment(),
                  mode = "function",
                  inherits = TRUE)
            ##
            ## ---- checks the loaded writer ------------------------------------------------------------------------------------------
            ##

            if (base::is.null(JLS_writer)) {
                  base::stop("The guarded longitudinal and survival writer has not been loaded.")
            }
            return(JLS_writer(
                  source_Stan_model_file_path = source_path,
                  NicoSIMD_fused_Stan_model_file_path = output_path))
      }
      if (base::identical(base::basename(source_path),
            "Gaussian_process_regression_chunked_Stan_math_for_layout_timings.stan")) {
            GPR_writer <-  base::get0(
                  "fn_write_NicoSIMD_fused_GP_regression_Stan_model",
                  envir = base::environment(),
                  mode = "function",
                  inherits = TRUE)
            if (base::is.null(GPR_writer)) {
                  base::stop("The guarded GP regression source writer has not been loaded.")
            }
            return(GPR_writer(
                  source_Stan_model_file_path = source_path,
                  NicoSIMD_fused_Stan_model_file_path = output_path))
      }

      if (base::basename(source_path) %in% base::paste0(
            "latent_diffusion_survival_chunked_no_loops_over_N_v3_with_the_fused_proportional_hazards_node",
            c("", "_automatic_reduce_sum_chunked"),
            "_Stan_math_for_layout_timings.stan")) {
            latent_writer <-  base::get0(
                  "fn_write_NicoSIMD_fused_latent_diffusion_survival_Stan_model",
                  envir = base::environment(),
                  mode = "function",
                  inherits = TRUE)
            if (base::is.null(latent_writer)) {
                  base::stop("The guarded latent diffusion survival source writer has not been loaded.")
            }
            return(latent_writer(
                  source_Stan_model_file_path = source_path,
                  NicoSIMD_fused_Stan_model_file_path = output_path))
      }

      ##
      ## ---- uses the guarded hierarchical logistic regression writer ----------------------------------------------------------------
      ##

      if (base::basename(source_path) %in% base::paste0(
            "hierarchical_logistic_regression_chunked_no_loops_over_N_with_bernoulli_logit_glm",
            c("", "_automatic_reduce_sum_chunked"),
            "_Stan_math_for_layout_timings.stan")) {
            HLR_writer <-  base::get0(
                  "fn_write_NicoSIMD_fused_hierarchical_logistic_regression_Stan_model",
                  envir = base::environment(),
                  mode = "function",
                  inherits = TRUE)
            if (base::is.null(HLR_writer)) {
                  base::stop("The guarded hierarchical logistic regression source writer has not been loaded.")
            }
            return(HLR_writer(
                  source_Stan_model_file_path = source_path,
                  NicoSIMD_fused_Stan_model_file_path = output_path))
      }

      if (base::basename(source_path) %in% base::paste0(
            "network_meta_analysis_Xu_fixed_cutpoints_chunked_no_loops_over_N_with_the_fused_binomial_",
            "counts_on_AVX_and_the_constants_of_the_data_left_out", c("", "_automatic_reduce_sum_chunked"),
            "_Stan_math_for_layout_timings.stan")) {
            NMA_writer <-  base::get0(
                  "fn_write_NicoSIMD_fused_NMA_ord_biv_Stan_model",
                  envir = base::environment(),
                  mode = "function",
                  inherits = TRUE)
            if (base::is.null(NMA_writer)) {
                  base::stop("The guarded NMA ord_biv source writer has not been loaded.")
            }
            return(NMA_writer(
                  source_Stan_model_file_path = source_path,
                  NicoSIMD_fused_Stan_model_file_path = output_path))
      }

      source_file_info <-  base::file.info(source_path)
      source_size <-  source_file_info$size
      if (source_file_info$isdir || !base::is.finite(source_size) || source_size < 1) {
            base::stop("The source Stan file is empty or unreadable.")
      }

      source_connection <-  base::file(source_path, open = "rb")
      source_bytes <-  base::readBin(source_connection,
                                     what = "raw",
                                     n = base::as.integer(source_size))
      base::close(source_connection)

      if (base::length(source_bytes) != source_size) {
            base::stop("The source Stan file could not be read completely.")
      }

      ##
      ## ---- masks comments and strings without changing byte positions:
      ##

      fn_mask_Stan_comments_and_strings <-  function( source_bytes
      ) {

            masked_bytes <-  source_bytes
            source_values <-  base::as.integer(source_bytes)
            n_bytes <-  base::length(source_bytes)
            in_line_comment <-  FALSE
            in_block_comment <-  FALSE
            in_string <-  FALSE
            escaped <-  FALSE
            position <-  1

            while (position <= n_bytes) {

                  current_byte <-  source_values[position]
                  next_byte <-  if (position < n_bytes) source_values[position + 1] else 0

                  if (in_line_comment) {

                        if (current_byte %in% c(10, 13)) {
                              in_line_comment <-  FALSE
                        } else {
                              masked_bytes[position] <-  base::as.raw(32)
                        }

                  } else if (in_block_comment) {

                        if ((current_byte == 42) && (next_byte == 47)) {
                              masked_bytes[position] <-  base::as.raw(32)
                              masked_bytes[position + 1] <-  base::as.raw(32)
                              in_block_comment <-  FALSE
                              position <-  position + 1
                        } else if (!(current_byte %in% c(10, 13))) {
                              masked_bytes[position] <-  base::as.raw(32)
                        }

                  } else if (in_string) {

                        if (!(current_byte %in% c(10, 13))) {
                              masked_bytes[position] <-  base::as.raw(32)
                        }

                        if (escaped) {
                              escaped <-  FALSE
                        } else if (current_byte == 92) {
                              escaped <-  TRUE
                        } else if (current_byte == 34) {
                              in_string <-  FALSE
                        }

                  } else if ((current_byte == 47) && (next_byte == 47)) {

                        masked_bytes[position] <-  base::as.raw(32)
                        masked_bytes[position + 1] <-  base::as.raw(32)
                        in_line_comment <-  TRUE
                        position <-  position + 1

                  } else if ((current_byte == 47) && (next_byte == 42)) {

                        masked_bytes[position] <-  base::as.raw(32)
                        masked_bytes[position + 1] <-  base::as.raw(32)
                        in_block_comment <-  TRUE
                        position <-  position + 1

                  } else if (current_byte == 34) {

                        masked_bytes[position] <-  base::as.raw(32)
                        in_string <-  TRUE

                  }

                  position <-  position + 1

            }

            if (in_block_comment || in_string || escaped) {
                  base::stop("The source has an unterminated comment or string.")
            }

            return(masked_bytes)

      }

      ##
      ## ---- finds one supported source span:
      ##

      fn_find_one_span <-  function( pattern,
                                     source_text,
                                     span_name
      ) {

            positions <-  base::gregexpr(pattern = pattern,
                                          text = source_text,
                                          perl = TRUE,
                                          useBytes = TRUE)[[1]]

            if ((base::length(positions) != 1) || (positions[1] < 1)) {
                  base::stop(base::paste0("Unsupported or ambiguous source: ", span_name, "."))
            }

            return(c(start = positions[1],
                     length = base::attr(positions, "match.length")[1]))

      }

      ##
      ## ---- replaces a byte span without reserialising the source:
      ##

      fn_splice_raw <-  function( input_bytes,
                                  start_position,
                                  remove_length,
                                  replacement_bytes
      ) {

            n_bytes <-  base::length(input_bytes)
            suffix_start <-  start_position + remove_length

            if ((start_position < 1) || (start_position > n_bytes + 1) ||
                (remove_length < 0) || (suffix_start > n_bytes + 1)) {
                  base::stop("The requested source span is outside the file.")
            }

            prefix_bytes <-  if (start_position > 1) {
                  input_bytes[base::seq_len(start_position - 1)]
            } else {
                  base::raw(0)
            }
            suffix_bytes <-  if (suffix_start <= n_bytes) {
                  input_bytes[base::seq.int(suffix_start, n_bytes)]
            } else {
                  base::raw(0)
            }

            return(c(prefix_bytes, replacement_bytes, suffix_bytes))

      }

      ##
      ## ---- finds a matching closing brace:
      ##

      fn_find_closing_brace <-  function( masked_bytes,
                                          opening_position
      ) {

            byte_values <-  base::as.integer(masked_bytes)
            brace_depth <-  0

            for (position in base::seq.int(opening_position, base::length(byte_values))) {

                  if (byte_values[position] == 123) {
                        brace_depth <-  brace_depth + 1
                  } else if (byte_values[position] == 125) {
                        brace_depth <-  brace_depth - 1
                        if (brace_depth == 0) {
                              return(position)
                        }
                  }

            }

            base::stop("The source has an unmatched code block.")

      }

      ##
      ## ---- counts calls in code, not in comments or strings:
      ##

      fn_code_matches <-  function( pattern,
                                    source_text
      ) {

            positions <-  base::gregexpr(pattern = pattern,
                                          text = source_text,
                                          perl = TRUE,
                                          useBytes = TRUE)[[1]]

            if ((base::length(positions) == 1) && (positions[1] < 1)) {
                  return(base::character(0))
            }

            return(base::regmatches(source_text, list(positions))[[1]])

      }

      ##
      ## ---- checks the supported source structure:
      ##

      masked_bytes <-  fn_mask_Stan_comments_and_strings(source_bytes)
      masked_text <-  base::rawToChar(masked_bytes)

      robust_t_calls <-  fn_code_matches(
            "(?<![A-Za-z0-9_])student_t_lupdf\\s*\\(",
            masked_text)
      robust_t_fused_calls <-  fn_code_matches(
            "(?<![A-Za-z0-9_])nico_student_t_id_glm_[A-Za-z0-9_]+(?=\\s*\\()",
            masked_text)

      if ((base::length(robust_t_calls) > 0) ||
          (base::length(robust_t_fused_calls) > 0)) {

            # if ((base::length(robust_t_calls) != 1) ||
            #     (base::length(robust_t_fused_calls) != 0)) {
            #       base::stop("The robust-t likelihood is ambiguous or already fused.")
            # }

            ##
            ## ---- a second call is taken only as the one-chunk statement on y and X_with_ones whole:
            ##

            robust_t_one_chunk_pattern <-  paste0(
                  "(?<![A-Za-z0-9_])target\\s*\\+=\\s*student_t_lupdf\\s*\\(\\s*",
                  "y\\s*\\|\\s*nu\\s*,\\s*",
                  "X_with_ones\\s*\\*\\s*",
                  "append_row\\s*\\(\\s*alpha\\s*,\\s*beta\\s*\\)\\s*,\\s*",
                  "sigma\\s*\\)\\s*;")
            robust_t_one_chunk_statements <-  fn_code_matches(
                  robust_t_one_chunk_pattern,
                  masked_text)

            if ((base::length(robust_t_calls) != 1 + base::length(robust_t_one_chunk_statements)) ||
                (base::length(robust_t_one_chunk_statements) > 1) ||
                (base::length(robust_t_fused_calls) != 0)) {
                  base::stop("The robust-t likelihood is ambiguous or already fused.")
            }

            robust_t_partial_sum_names <-  fn_code_matches(
                  "(?<![A-Za-z0-9_])partial_sum_observations_lpdf\\s*\\(",
                  masked_text)
            if (base::length(robust_t_partial_sum_names) != 1) {
                  base::stop("The robust-t partial-sum function is ambiguous.")
            }

            robust_functions_span <-  fn_find_one_span(
                  "(?<![A-Za-z0-9_])functions\\s*\\{",
                  masked_text,
                  "functions block")
            robust_functions_opening_position <-  robust_functions_span["start"] +
                                                  robust_functions_span["length"] - 1
            robust_functions_closing_position <-  fn_find_closing_brace(
                  masked_bytes,
                  robust_functions_opening_position)

            robust_t_signature_pattern <-  paste0(
                  "(?<![A-Za-z0-9_])real\\s+partial_sum_observations_lpdf\\s*\\(\\s*",
                  "array\\s*\\[\\s*\\]\\s*real\\s+y_slice\\s*,\\s*",
                  "int\\s+start\\s*,\\s*int\\s+end\\s*,\\s*",
                  "data\\s+matrix\\s+X_with_ones\\s*,\\s*data\\s+real\\s+nu\\s*,\\s*",
                  "real\\s+alpha\\s*,\\s*vector\\s+beta\\s*,\\s*",
                  "real\\s+sigma\\s*\\)\\s*\\{")
            robust_t_signature_span <-  fn_find_one_span(
                  robust_t_signature_pattern,
                  masked_text,
                  "robust-t partial-sum signature")
            robust_t_signature_opening <-  robust_t_signature_span["start"] +
                                          robust_t_signature_span["length"] - 1
            robust_t_signature_closing <-  fn_find_closing_brace(
                  masked_bytes,
                  robust_t_signature_opening)

            if ((robust_t_signature_span["start"] <= robust_functions_opening_position) ||
                (robust_t_signature_closing >= robust_functions_closing_position)) {
                  base::stop("The robust-t partial-sum function is outside the functions block.")
            }

            robust_t_body_start <-  robust_t_signature_opening + 1
            robust_t_body_end <-  robust_t_signature_closing - 1
            robust_t_body_bytes <-  if (robust_t_body_start <= robust_t_body_end) {
                  masked_bytes[base::seq.int(robust_t_body_start, robust_t_body_end)]
            } else {
                  base::raw(0)
            }
            robust_t_body_text <-  base::rawToChar(robust_t_body_bytes)

            robust_t_return_pattern <-  paste0(
                  "(?<![A-Za-z0-9_])return\\s+student_t_lupdf\\s*\\(\\s*",
                  "y_slice\\s*\\|\\s*nu\\s*,\\s*",
                  "X_with_ones\\s*\\[\\s*start\\s*:\\s*end\\s*\\]\\s*\\*\\s*",
                  "append_row\\s*\\(\\s*alpha\\s*,\\s*beta\\s*\\)\\s*,\\s*",
                  "sigma\\s*\\)\\s*;")
            robust_t_return_span <-  fn_find_one_span(
                  robust_t_return_pattern,
                  robust_t_body_text,
                  "robust-t likelihood expression")
            robust_t_body_tokens <-  base::gsub("[ \\t\\r\\n]+", "",
                                                 robust_t_body_text,
                                                 perl = TRUE,
                                                 useBytes = TRUE)

            robust_t_expected_body_tokens <-  paste0(
                  "returnstudent_t_lupdf(y_slice|nu,",
                  "X_with_ones[start:end]*append_row(alpha,beta),sigma);")
            if (!base::identical(robust_t_body_tokens,
                                 robust_t_expected_body_tokens)) {
                  base::stop("The robust-t partial-sum body has unsupported statements.")
            }

            robust_t_return_source_start <-  robust_t_body_start +
                                             robust_t_return_span["start"] - 1
            robust_t_return_source_end <-  robust_t_return_source_start +
                                           robust_t_return_span["length"] - 1
            robust_t_return_source_text <-  base::rawToChar(
                  source_bytes[base::seq.int(robust_t_return_source_start,
                                             robust_t_return_source_end)])
            robust_t_return_source_tokens <-  base::gsub(
                  "[ \\t\\r\\n]+",
                  "",
                  robust_t_return_source_text,
                  perl = TRUE,
                  useBytes = TRUE)
            robust_t_expected_return_tokens <-  paste0(
                  "returnstudent_t_lupdf(y_slice|nu,",
                  "X_with_ones[start:end]*append_row(alpha,beta),sigma);")
            if (!base::identical(robust_t_return_source_tokens,
                                 robust_t_expected_return_tokens)) {
                  base::stop("The robust-t likelihood expression contains unsupported source text.")
            }

            newline_match <-  base::regexpr("\\r\\n|\\n|\\r",
                                            masked_text,
                                            perl = TRUE,
                                            useBytes = TRUE)
            if (newline_match[1] < 1) {
                  base::stop("The source has no line ending.")
            }
            newline_length <-  base::attr(newline_match, "match.length")[1]
            newline_bytes <-  source_bytes[base::seq.int(newline_match[1],
                                                         newline_match[1] + newline_length - 1)]
            newline_text <-  base::rawToChar(newline_bytes)

            robust_t_replacement <-  base::paste0(
                  "return nico_student_t_id_glm_lupdf(y_slice | X_with_ones[start:end], ",
                  "0.0, append_row(alpha, beta),",
                  newline_text,
                  "                                           nu, sigma);")
            # robust_t_output_bytes <-  fn_splice_raw(
            #       source_bytes,
            #       robust_t_return_source_start,
            #       robust_t_return_span["length"],
            #       base::charToRaw(robust_t_replacement))

            ##
            ## ---- the one-chunk statement in the model block, spliced first (it comes after the partial sum):
            ##

            robust_t_input_bytes <-  source_bytes

            if (base::length(robust_t_one_chunk_statements) == 1) {

                  robust_model_span <-  fn_find_one_span(
                        "(?<![A-Za-z0-9_])model\\s*\\{",
                        masked_text,
                        "model block")
                  robust_model_opening_position <-  robust_model_span["start"] +
                                                    robust_model_span["length"] - 1
                  robust_model_closing_position <-  fn_find_closing_brace(
                        masked_bytes,
                        robust_model_opening_position)

                  robust_t_one_chunk_span <-  fn_find_one_span(
                        robust_t_one_chunk_pattern,
                        masked_text,
                        "robust-t one-chunk statement")
                  robust_t_one_chunk_end <-  robust_t_one_chunk_span["start"] +
                                             robust_t_one_chunk_span["length"] - 1

                  if ((robust_t_one_chunk_span["start"] <= robust_model_opening_position) ||
                      (robust_t_one_chunk_end >= robust_model_closing_position) ||
                      (robust_t_one_chunk_span["start"] <= robust_t_return_source_end)) {
                        base::stop("The robust-t one-chunk statement is outside the model block.")
                  }

                  robust_t_one_chunk_source_tokens <-  base::gsub(
                        "[ \\t\\r\\n]+",
                        "",
                        base::rawToChar(source_bytes[base::seq.int(robust_t_one_chunk_span["start"],
                                                                   robust_t_one_chunk_end)]),
                        perl = TRUE,
                        useBytes = TRUE)
                  if (!base::identical(robust_t_one_chunk_source_tokens,
                                       paste0("target+=student_t_lupdf(y|nu,",
                                              "X_with_ones*append_row(alpha,beta),sigma);"))) {
                        base::stop("The robust-t one-chunk statement contains unsupported source text.")
                  }

                  ## (y_array: the responses as the array the declaration takes, already in transformed data)
                  robust_t_one_chunk_replacement <-  base::paste0(
                        "target += nico_student_t_id_glm_lupdf(y_array | X_with_ones, ",
                        "0.0, append_row(alpha, beta),",
                        newline_text,
                        "                                              nu, sigma);")
                  robust_t_input_bytes <-  fn_splice_raw(
                        source_bytes,
                        robust_t_one_chunk_span["start"],
                        robust_t_one_chunk_span["length"],
                        base::charToRaw(robust_t_one_chunk_replacement))

            }

            robust_t_output_bytes <-  fn_splice_raw(
                  robust_t_input_bytes,
                  robust_t_return_source_start,
                  robust_t_return_span["length"],
                  base::charToRaw(robust_t_replacement))

            robust_t_signature_line_start <-  robust_t_signature_span["start"]
            if (robust_t_signature_line_start > 1) {
                  robust_t_preceding_positions <-  base::seq_len(
                        robust_t_signature_line_start - 1)
                  robust_t_preceding_values <-  base::as.integer(
                        source_bytes[robust_t_preceding_positions])
                  robust_t_line_break_positions <-  base::which(
                        robust_t_preceding_values %in% c(10, 13))
                  if (base::length(robust_t_line_break_positions) > 0) {
                        robust_t_signature_line_start <-  base::max(
                              robust_t_line_break_positions) + 1
                  } else {
                        robust_t_signature_line_start <-  1
                  }
            }

            robust_t_signature_indent_bytes <-  if (
                  robust_t_signature_line_start < robust_t_signature_span["start"]) {
                  source_bytes[base::seq.int(robust_t_signature_line_start,
                                             robust_t_signature_span["start"] - 1)]
            } else {
                  base::raw(0)
            }
            robust_t_signature_indent_values <-  base::as.integer(
                  robust_t_signature_indent_bytes)
            if (base::any(!(robust_t_signature_indent_values %in% c(9, 32)))) {
                  base::stop("The robust-t signature has unsupported line prefixes.")
            }

            robust_t_signature_indent_text <-  base::rawToChar(
                  robust_t_signature_indent_bytes)
            robust_t_argument_indent_text <-  base::paste0(
                  robust_t_signature_indent_text,
                  base::strrep(" ",
                               base::nchar("real nico_student_t_id_glm_lpdf(")))
            robust_t_declaration_text <-  base::paste0(
                  robust_t_signature_indent_text,
                  "real nico_student_t_id_glm_lpdf(array[] real y,",
                  newline_text,
                  robust_t_argument_indent_text,
                  "matrix x,",
                  newline_text,
                  robust_t_argument_indent_text,
                  "real alpha,",
                  newline_text,
                  robust_t_argument_indent_text,
                  "vector beta,",
                  newline_text,
                  robust_t_argument_indent_text,
                  "real nu,",
                  newline_text,
                  robust_t_argument_indent_text,
                  "real sigma);",
                  newline_text,
                  newline_text)

            robust_t_output_bytes <-  fn_splice_raw(
                  robust_t_output_bytes,
                  robust_t_signature_line_start,
                  0,
                  base::charToRaw(robust_t_declaration_text))

            if (base::file.exists(output_path)) {
                  base::stop("The output file appeared before writing; no file was replaced.")
            }

            temporary_output_path <-  base::tempfile(
                  pattern = ".NicoSIMD_fused_",
                  tmpdir = output_directory)
            write_connection <-  base::file(temporary_output_path, open = "wb")
            base::on.exit({
                  base::try(base::close(write_connection), silent = TRUE)
                  if (base::file.exists(temporary_output_path)) {
                        base::unlink(temporary_output_path)
                  }
            }, add = TRUE)

            base::writeBin(robust_t_output_bytes, write_connection)
            base::close(write_connection)

            if (!base::file.link(temporary_output_path, output_path)) {
                  base::stop("The generated source could not be written without overwriting a file.")
            }

            base::message(base::paste0(
                  "Prepared the custom Student-t GLM call; all other Stan calls remain for backend conversion: ",
                  output_path))

            return(base::list(
                  NicoSIMD_fused_Stan_model_file_path = output_path,
                  fused_calls = "nico_student_t_id_glm_lupdf",
                  source_for_remaining_backend_conversion = output_path))

      }

      functions_span <-  fn_find_one_span("(?<![A-Za-z0-9_])functions\\s*\\{",
                                            masked_text,
                                            "functions block")
      functions_opening_position <-  functions_span["start"] + functions_span["length"] - 1
      functions_closing_position <-  fn_find_closing_brace(masked_bytes,
                                                            functions_opening_position)

      partial_sum_signature_pattern <-  paste0(
            "(?<![A-Za-z0-9_])real\\s+partial_sum_observations_lpdf\\s*\\(\\s*",
            "array\\s*\\[\\s*\\]\\s*real\\s+t_slice\\s*,\\s*",
            "int\\s+start\\s*,\\s*int\\s+end\\s*,\\s*",
            "vector\\s+t_cens\\s*,\\s*matrix\\s+X_with_ones\\s*,\\s*",
            "matrix\\s+X_cens_with_ones\\s*,\\s*array\\s*\\[\\s*\\]\\s*int\\s+",
            "censored_before\\s*,\\s*real\\s+alpha\\s*,\\s*vector\\s+beta\\s*,\\s*",
            "real\\s+mu\\s*",
            "(?:,\\s*data\\s+vector\\s+log_t\\s*,\\s*data\\s+vector\\s+log_t_cens\\s*)?",
            "\\)\\s*\\{"
      )

      partial_sum_signature_span <-  fn_find_one_span(
            partial_sum_signature_pattern,
            masked_text,
            "partial-sum signature")
      partial_sum_signature_opening <-  partial_sum_signature_span["start"] +
                                        partial_sum_signature_span["length"] - 1
      partial_sum_signature_closing <-  fn_find_closing_brace(masked_bytes,
                                                               partial_sum_signature_opening)

      if ((partial_sum_signature_span["start"] <= functions_opening_position) ||
          (partial_sum_signature_closing >= functions_closing_position)) {
            base::stop("The partial-sum function is outside the functions block.")
      }

      if (base::length(fn_code_matches(
                "(?<![A-Za-z0-9_])nico_weibull_log_glm_[A-Za-z0-9_]+(?=\\s*\\()",
                masked_text)) > 0) {
            base::stop("The source already contains a custom Weibull GLM call.")
      }

      body_start <-  partial_sum_signature_opening + 1
      body_end <-  partial_sum_signature_closing - 1
      body_bytes <-  if (body_start <= body_end) {
            masked_bytes[base::seq.int(body_start, body_end)]
      } else {
            base::raw(0)
      }
      body_text <-  base::rawToChar(body_bytes)

      minus_alpha_pattern <-  paste0(
            "(?<![A-Za-z0-9_])real\\s+minus_alpha\\s*=\\s*-\\s*alpha\\s*;")
      coefficients_pattern <-  paste0(
            "(?<![A-Za-z0-9_])vector\\s*\\[\\s*cols\\s*\\(\\s*X_with_ones\\s*\\)\\s*\\]",
            "\\s+coefficients_over_minus_alpha\\s*=\\s*append_row\\s*\\(\\s*mu\\s*,\\s*beta",
            "\\s*\\)\\s*/\\s*minus_alpha\\s*;")
      guard_pattern <-  paste0("(?<![A-Za-z0-9_])if\\s*\\(\\s*c2\\s*>=\\s*c1\\s*\\)\\s*\\{")

      minus_alpha_span <-  fn_find_one_span(minus_alpha_pattern,
                                             body_text,
                                             "minus-alpha construction")
      coefficients_span <-  fn_find_one_span(coefficients_pattern,
                                              body_text,
                                              "coefficient construction")
      guard_span <-  fn_find_one_span(guard_pattern,
                                      body_text,
                                      "censored guard")

      failure_density_pattern <-  paste0(
            "real\\s+lp\\s*=\\s*weibull_lupdf\\s*\\(\\s*",
            "t_slice\\s*\\|\\s*alpha\\s*,\\s*exp\\s*\\(\\s*",
            "X_with_ones\\s*\\[\\s*start\\s*:\\s*end\\s*\\]\\s*\\*\\s*",
            "coefficients_over_minus_alpha\\s*\\)\\s*\\)\\s*;")
      failure_explicit_pattern <-  paste0(
            "vector\\s*\\[\\s*end\\s*-\\s*start\\s*\\+\\s*1\\s*\\]\\s+lambda\\s*=\\s*",
            "exp\\s*\\(\\s*X_with_ones\\s*\\[\\s*start\\s*:\\s*end\\s*\\]\\s*\\*\\s*",
            "coefficients_over_minus_alpha\\s*\\)\\s*;\\s*",
            "real\\s+lp\\s*=\\s*sum\\s*\\(\\s*log\\s*\\(\\s*alpha\\s*\\)\\s*\\+\\s*",
            "\\(\\s*alpha\\s*-\\s*1\\s*\\)\\s*\\*\\s*log_t\\s*\\[\\s*start\\s*:\\s*end\\s*\\]",
            "\\s*-\\s*alpha\\s*\\*\\s*log\\s*\\(\\s*lambda\\s*\\)\\s*-\\s*",
            "\\(\\s*to_vector\\s*\\(\\s*t_slice\\s*\\)\\s*\\./\\s*lambda\\s*\\)",
            "\\s*\\.\\^\\s*alpha\\s*\\)\\s*;")
      failure_pattern <-  paste0("(?<![A-Za-z0-9_])(?:",
                                 failure_density_pattern, "|", failure_explicit_pattern, ")")
      censored_density_pattern <-  paste0(
            "lp\\s*\\+=\\s*weibull_lccdf\\s*\\(\\s*",
            "t_cens\\s*\\[\\s*c1\\s*:\\s*c2\\s*\\]\\s*\\|\\s*alpha\\s*,\\s*",
            "exp\\s*\\(\\s*X_cens_with_ones\\s*\\[\\s*c1\\s*:\\s*c2\\s*\\]\\s*",
            "\\*\\s*coefficients_over_minus_alpha\\s*\\)\\s*\\)\\s*;")
      censored_explicit_pattern <-  paste0(
            "vector\\s*\\[\\s*c2\\s*-\\s*c1\\s*\\+\\s*1\\s*\\]\\s+lambda_cens\\s*=\\s*",
            "exp\\s*\\(\\s*X_cens_with_ones\\s*\\[\\s*c1\\s*:\\s*c2\\s*\\]\\s*\\*\\s*",
            "coefficients_over_minus_alpha\\s*\\)\\s*;\\s*",
            "lp\\s*\\+=\\s*-\\s*sum\\s*\\(\\s*\\(\\s*t_cens\\s*\\[\\s*c1\\s*:\\s*c2\\s*\\]",
            "\\s*\\./\\s*lambda_cens\\s*\\)\\s*\\.\\^\\s*alpha\\s*\\)\\s*;")
      censored_pattern <-  paste0("(?<![A-Za-z0-9_])(?:",
                                  censored_density_pattern, "|", censored_explicit_pattern, ")")

      failure_span <-  fn_find_one_span(failure_pattern,
                                         body_text,
                                         "failure likelihood")
      censored_span <-  fn_find_one_span(censored_pattern,
                                          body_text,
                                          "censored likelihood")

      if (!((minus_alpha_span["start"] < coefficients_span["start"]) &&
            (coefficients_span["start"] < failure_span["start"]) &&
            (failure_span["start"] < guard_span["start"]) &&
            (guard_span["start"] < censored_span["start"]))) {
            base::stop("The partial-sum likelihood has an unsupported statement order.")
      }

      weibull_calls <-  fn_code_matches(
            "(?<![A-Za-z0-9_])weibull_[A-Za-z0-9_]+(?=\\s*\\()",
            body_text)
      explicit_likelihood <-  base::grepl(failure_explicit_pattern, body_text, perl = TRUE)
      signature_text <-  base::substr(
            masked_text,
            partial_sum_signature_span["start"],
            partial_sum_signature_opening)
      if (explicit_likelihood && !base::grepl(
                "data\\s+vector\\s+log_t\\s*,\\s*data\\s+vector\\s+log_t_cens",
                signature_text,
                perl = TRUE)) {

            base::stop("The explicit likelihood requires data-qualified time logarithms.")

      }
      expected_weibull_calls <-  if (explicit_likelihood) base::character(0) else
            c("weibull_lupdf", "weibull_lccdf")
      if (!base::identical(base::sort(weibull_calls), base::sort(expected_weibull_calls))) {
            base::stop("The partial-sum function has unsupported Weibull calls.")
      }
      if (!base::identical(explicit_likelihood,
                            base::grepl(censored_explicit_pattern, body_text, perl = TRUE))) {
            base::stop("The observed and censored likelihood forms differ.")
      }

      guard_opening_position <-  body_start + guard_span["start"] + guard_span["length"] - 2
      guard_closing_position <-  fn_find_closing_brace(masked_bytes,
                                                        guard_opening_position)
      censored_call_position <-  body_start + censored_span["start"] - 1

      if ((censored_call_position <= guard_opening_position) ||
          (censored_call_position >= guard_closing_position)) {
            base::stop("The censored likelihood is not inside the supported guard.")
      }

      ##
      ## ---- preserves comments and strings inside matched expressions:
      ##

      failure_source_start <-  body_start + failure_span["start"] - 1
      failure_source_end <-  failure_source_start + failure_span["length"] - 1
      censored_source_start <-  body_start + censored_span["start"] - 1
      censored_source_end <-  censored_source_start + censored_span["length"] - 1

      failure_source_text <-  base::rawToChar(
            source_bytes[base::seq.int(failure_source_start, failure_source_end)])
      censored_source_text <-  base::rawToChar(
            source_bytes[base::seq.int(censored_source_start, censored_source_end)])

      failure_source_tokens <-  base::gsub("[ \\t\\r\\n]+", "", failure_source_text,
                                            perl = TRUE,
                                            useBytes = TRUE)
      censored_source_tokens <-  base::gsub("[ \\t\\r\\n]+", "", censored_source_text,
                                             perl = TRUE,
                                             useBytes = TRUE)

      ##
      ## ---- Checks the exact observed expression --------------------------------------------------------------------------------------
      ##
      expected_failure_tokens <-  if (explicit_likelihood) {

            base::paste0(
                  "vector[end-start+1]lambda=exp(X_with_ones[start:end]*",
                  "coefficients_over_minus_alpha);",
                  "reallp=sum(log(alpha)+(alpha-1)*log_t[start:end]-alpha*log(lambda)-",
                  "(to_vector(t_slice)./lambda).^alpha);")

      } else {

            base::paste0(
                  "reallp=weibull_lupdf(t_slice|alpha,exp(X_with_ones[start:end]*",
                  "coefficients_over_minus_alpha));")

      }
      if (!base::identical(failure_source_tokens, expected_failure_tokens)) {
            base::stop("The failure likelihood contains unsupported source text.")
      }

      expected_censored_tokens <-  if (explicit_likelihood) {

            base::paste0(
                  "vector[c2-c1+1]lambda_cens=exp(X_cens_with_ones[c1:c2]*",
                  "coefficients_over_minus_alpha);",
                  "lp+=-sum((t_cens[c1:c2]./lambda_cens).^alpha);")

      } else {

            base::paste0("lp+=weibull_lccdf(t_cens[c1:c2]|alpha,",
                         "exp(X_cens_with_ones[c1:c2]*coefficients_over_minus_alpha));")

      }
      if (!base::identical(censored_source_tokens, expected_censored_tokens)) {
            base::stop("The censored likelihood contains unsupported source text.")
      }

      ##
      ## ---- replaces only the two reference expressions:
      ##

      newline_match <-  base::regexpr("\\r\\n|\\n|\\r",
                                      masked_text,
                                      perl = TRUE,
                                      useBytes = TRUE)
      if (newline_match[1] < 1) {
            base::stop("The source has no line ending.")
      }
      newline_length <-  base::attr(newline_match, "match.length")[1]
      newline_bytes <-  source_bytes[base::seq.int(newline_match[1],
                                                   newline_match[1] + newline_length - 1)]
      newline_text <-  base::rawToChar(newline_bytes)

      failure_replacement <-  base::paste0(
            "real lp = nico_weibull_log_glm_lupdf(t_slice | X_with_ones[start:end], 0.0,",
            newline_text,
            "                                             coefficients_over_minus_alpha, alpha);")
      censored_replacement <-  base::paste0(
            "lp += nico_weibull_log_glm_lccdf(t_cens[c1:c2] | X_cens_with_ones[c1:c2], 0.0,",
            newline_text,
            "                                           coefficients_over_minus_alpha, alpha);")

      ##
      ## ---- Transform the exact whole-expression one-chunk body before the earlier partial-sum spans --------------------------------
      ##
      output_bytes <-  source_bytes
      direct_body <-  FALSE
      direct_guard_pattern <-  "(?<![A-Za-z0-9_])if\\s*\\(\\s*chunk_size\\s*>=\\s*N\\s*\\)\\s*\\{"
      direct_guards <-  fn_code_matches(direct_guard_pattern, masked_text)
      whole_failure_expressions <-  fn_code_matches(
            "vector\\s*\\[\\s*N\\s*\\]\\s+lambda\\s*=\\s*exp\\s*\\(\\s*X_with_ones\\s*\\*",
            masked_text)
      if (base::length(whole_failure_expressions) > 0 && base::length(direct_guards) != 1) {

            base::stop("The Weibull whole-expression likelihood requires the supported one-chunk guard.")

      }
      if (base::length(direct_guards) > 0) {

            direct_guard <-  fn_find_one_span(direct_guard_pattern,
                                               masked_text,
                                               "Weibull one-chunk guard")
            direct_opening <-  direct_guard["start"] + direct_guard["length"] - 1
            direct_closing <-  fn_find_closing_brace(masked_bytes,
                                                      direct_opening)
            direct_text <-  base::rawToChar(source_bytes[base::seq.int(direct_opening + 1,
                                                                        direct_closing - 1)])
            direct_tokens <-  base::gsub("[ \\t\\r\\n]+", "", direct_text, perl = TRUE, useBytes = TRUE)
            direct_masked <-  base::rawToChar(masked_bytes[base::seq.int(direct_opening + 1,
                                                                          direct_closing - 1)])
            direct_body <-  !base::grepl("partial_sum_observations_lupdf\\s*\\(",
                                          direct_masked,
                                          perl = TRUE)
            if (direct_body) {

                  model_span <-  fn_find_one_span("(?<![A-Za-z0-9_])model\\s*\\{",
                                                   masked_text,
                                                   "model block")
                  model_opening <-  model_span["start"] + model_span["length"] - 1
                  model_closing <-  fn_find_closing_brace(masked_bytes,
                                                           model_opening)
                  expected_direct_tokens <-  base::paste0(
                        "realminus_alpha=-alpha;",
                        "vector[cols(X_with_ones)]coefficients_over_minus_alpha=append_row(mu,beta)/minus_alpha;",
                        "vector[N]lambda=exp(X_with_ones*coefficients_over_minus_alpha);",
                        "target+=sum(log(alpha)+(alpha-1)*log_t-alpha*log(lambda)-(t./lambda).^alpha);",
                        "if(N_cens>0){",
                        "vector[N_cens]lambda_cens=exp(X_cens_with_ones*coefficients_over_minus_alpha);",
                        "target+=-sum((t_cens./lambda_cens).^alpha);}")
                  if (!explicit_likelihood || direct_opening <= model_opening ||
                      direct_closing >= model_closing || direct_opening <= censored_source_end ||
                      !base::identical(direct_tokens, expected_direct_tokens)) {

                        base::stop("The Weibull one-chunk body contains unsupported source text.")

                  }
                  direct_replacement <-  base::paste(c(
                        "",
                        "            real minus_alpha = -alpha;",
                        paste0("            vector[cols(X_with_ones)] coefficients_over_minus_alpha = ",
                               "append_row(mu, beta) / minus_alpha;"),
                        "            target += nico_weibull_log_glm_lupdf(",
                        "                  t | X_with_ones,",
                        "                  0.0,",
                        "                  coefficients_over_minus_alpha,",
                        "                  alpha);",
                        "            if (N_cens > 0) {",
                        "                  target += nico_weibull_log_glm_lccdf(",
                        "                        t_cens | X_cens_with_ones,",
                        "                        0.0,",
                        "                        coefficients_over_minus_alpha,",
                        "                        alpha);",
                        "            }",
                        "",
                        "      "), collapse = newline_text)
                  output_bytes <-  fn_splice_raw(output_bytes,
                                                 direct_opening + 1,
                                                 direct_closing - direct_opening - 1,
                                                 base::charToRaw(direct_replacement))

            }

      }
      if (censored_source_start > failure_source_start) {
            output_bytes <-  fn_splice_raw(output_bytes,
                                           censored_source_start,
                                           censored_span["length"],
                                           base::charToRaw(censored_replacement))
            output_bytes <-  fn_splice_raw(output_bytes,
                                           failure_source_start,
                                           failure_span["length"],
                                           base::charToRaw(failure_replacement))
      } else {
            output_bytes <-  fn_splice_raw(output_bytes,
                                           failure_source_start,
                                           failure_span["length"],
                                           base::charToRaw(failure_replacement))
            output_bytes <-  fn_splice_raw(output_bytes,
                                           censored_source_start,
                                           censored_span["length"],
                                           base::charToRaw(censored_replacement))
      }

      ##
      ## ---- adds the two external declarations once:
      ##

      output_masked_bytes <-  fn_mask_Stan_comments_and_strings(output_bytes)
      output_masked_text <-  base::rawToChar(output_masked_bytes)
      declaration_names <-  fn_code_matches(
            "(?m)^\\s*real\\s+nico_weibull_log_glm_(?:lpdf|lccdf)\\s*\\(",
            output_masked_text)

      if (base::length(declaration_names) != 0) {
            base::stop("The generated source already contains a custom Weibull GLM declaration.")
      }

      signature_line_start <-  partial_sum_signature_span["start"]
      if (signature_line_start > 1) {
            preceding_positions <-  base::seq_len(signature_line_start - 1)
            preceding_values <-  base::as.integer(source_bytes[preceding_positions])
            line_break_positions <-  base::which(preceding_values %in% c(10, 13))
            if (base::length(line_break_positions) > 0) {
                  signature_line_start <-  base::max(line_break_positions) + 1
            } else {
                  signature_line_start <-  1
            }
      }

      signature_indent_bytes <-  if (signature_line_start < partial_sum_signature_span["start"]) {
            source_bytes[base::seq.int(signature_line_start,
                                       partial_sum_signature_span["start"] - 1)]
      } else {
            base::raw(0)
      }
      signature_indent_values <-  base::as.integer(signature_indent_bytes)

      if (base::any(!(signature_indent_values %in% c(9, 32)))) {
            base::stop("The partial-sum signature has unsupported line prefixes.")
      }

      signature_indent_text <-  base::rawToChar(signature_indent_bytes)
      declaration_text <-  base::paste0(
            signature_indent_text,
            "real nico_weibull_log_glm_lpdf(array[] real y,",
            newline_text,
            "                               matrix x,",
            newline_text,
            "                               real alpha,",
            newline_text,
            "                               vector beta,",
            newline_text,
            "                               real shape);",
            newline_text,
            signature_indent_text,
            "real nico_weibull_log_glm_lccdf(vector y,",
            newline_text,
            "                                matrix x,",
            newline_text,
            "                                real alpha,",
            newline_text,
            "                                vector beta,",
            newline_text,
            "                                real shape);",
            newline_text,
            newline_text
      )

      ##
      ## ---- The direct branch passes the original data vector without making an array copy -----------------------------------------
      ##
      if (direct_body) {

            declaration_text <-  base::paste0(
                  declaration_text,
                  signature_indent_text,
                  "real nico_weibull_log_glm_lpdf(vector y,",
                  newline_text,
                  "                               matrix x,",
                  newline_text,
                  "                               real alpha,",
                  newline_text,
                  "                               vector beta,",
                  newline_text,
                  "                               real shape);",
                  newline_text,
                  newline_text)

      }
      declaration_bytes <-  base::charToRaw(declaration_text)
      output_bytes <-  fn_splice_raw(output_bytes,
                                     signature_line_start,
                                     0,
                                     declaration_bytes)

      ##
      ## ---- writes the generated source without overwriting a file:
      ##

      if (base::file.exists(output_path)) {
            base::stop("The output file appeared before writing; no file was replaced.")
      }

      temporary_output_path <-  base::tempfile(
            pattern = ".NicoSIMD_fused_",
            tmpdir = output_directory)
      write_connection <-  base::file(temporary_output_path, open = "wb")
      base::on.exit({
            base::try(base::close(write_connection), silent = TRUE)
            if (base::file.exists(temporary_output_path)) {
                  base::unlink(temporary_output_path)
            }
      }, add = TRUE)

      base::writeBin(output_bytes, write_connection)
      base::close(write_connection)

      if (!base::file.link(temporary_output_path, output_path)) {
            base::stop("The generated source could not be written without overwriting a file.")
      }

      base::message(base::paste0(
            "Prepared the two custom Weibull GLM calls; all other Stan calls remain for backend conversion: ",
            output_path))

      return(base::list(NicoSIMD_fused_Stan_model_file_path = output_path,
                        fused_calls = c("nico_weibull_log_glm_lupdf",
                                        "nico_weibull_log_glm_lccdf"),
                        source_for_remaining_backend_conversion = output_path))

}






















