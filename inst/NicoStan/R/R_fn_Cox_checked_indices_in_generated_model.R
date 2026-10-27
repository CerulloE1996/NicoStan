##
## ---- Checked data in the generated Cox model ---------------------------------------------------------------------------------------
##
R_fn_Cox_checked_indices_in_generated_model <-  function(
      code
) {

      if (!is.character(code) || length(code) != 1L || is.na(code))
            stop("Generated C++ must be one non-missing string.")

      ##
      ## ---- Match a closing delimiter -----------------------------------------------------------------------------------------------
      ##
      fn_close <-  function(
            text,
            opening,
            left,
            right
      ) {

            depth <-  0L

            ##
            ## ---- Read the delimited expression -------------------------------------------------------------------------------------
            ##
            for (position in seq.int(
                        opening,
                        nchar(text)
            )) {

                  value <-  substr(
                        text,
                        position,
                        position
                  )
                  if (identical(
                              value,
                              left
                  )) depth <-  depth + 1L
                  if (identical(
                              value,
                              right
                  )) depth <-  depth - 1L
                  if (depth == 0L) return(position)

            }

            stop("Unbalanced generated C++ delimiter.")

      }

      ##
      ## ---- Preserve positions when masking literals and comments -------------------------------------------------------------------
      ##
      fn_spaces <-  function(
            value
      ) {

            return(paste(
                        rep(
                              " ",
                              nchar(value)
                        ),
                        collapse = ""
            ))

      }

      visible <-  code
      literal <-  paste0(
            '"([^"\\\\]|\\\\.)*"',
            "|'([^'\\\\]|\\\\.)*'",
            "|//[^\\r\\n]*|/\\*[\\s\\S]*?\\*/"
      )
      masked <-  gregexpr(
            pattern = literal,
            text = visible,
            perl = TRUE
      )
      ##
      ## ---- Mask the C++ literals and comments --------------------------------------------------------------------------------------
      ##
      regmatches(
            visible,
            masked
      ) <-  lapply(
            regmatches(
                  visible,
                  masked
            ),
            function(values) {

                  return(vapply(
                              values,
                              fn_spaces,
                              character(1)
                  ))

            }
      )

      target <-  paste0(
            "nico_dot_products_of_affine_predictor_with_indexed_effects_and_",
            "log_cumulative_sum_exp_with_one_vector_of_N_values"
      )
      occurrences <-  gregexpr(
            pattern = paste0(
                  "\\b",
                  target,
                  "\\b"
            ),
            text = visible,
            perl = TRUE
      )[[1]]
      if (occurrences[1] < 0L) return(code)

      ##
      ## ---- Locate the generated model ----------------------------------------------------------------------------------------------
      ##
      class_pattern <-  paste0(
            "class\\s+([[:alnum:]_]+)\\s+final\\s*:\\s*public\\s+",
            "(?:stan::model::)?model_base_crtp\\s*<\\s*\\1\\s*>\\s*\\{"
      )
      classes <-  gregexpr(
            pattern = class_pattern,
            text = visible,
            perl = TRUE
      )[[1]]
      if (classes[1] < 0L || length(classes) != 1L)
            stop("Expected one generated model class for the Cox indices.")

      class_open <-  classes[1] + attr(
            classes,
            "match.length"
      )[1] - 1L
      class_end <-  fn_close(
            visible,
            class_open,
            "{",
            "}"
      )
      class_header <-  substr(
            visible,
            classes[1],
            class_open
      )
      class_name <-  sub(
            pattern = "^class\\s+([[:alnum:]_]+)[\\s\\S]*$",
            replacement = "\\1",
            x = class_header,
            perl = TRUE
      )

      ##
      ## ---- Read argument boundaries without changing their expressions -------------------------------------------------------------
      ##
      fn_arguments <-  function(
            opening,
            closing
      ) {

            round <-  0L
            square <-  0L
            curly <-  0L
            commas <-  integer(0)

            ##
            ## ---- Find only top-level argument separators ---------------------------------------------------------------------------
            ##
            for (position in seq.int(
                        opening + 1L,
                        closing - 1L
            )) {

                  value <-  substr(
                        visible,
                        position,
                        position
                  )
                  if (identical(
                              value,
                              "("
                  )) round <-  round + 1L
                  if (identical(
                              value,
                              ")"
                  )) round <-  round - 1L
                  if (identical(
                              value,
                              "["
                  )) square <-  square + 1L
                  if (identical(
                              value,
                              "]"
                  )) square <-  square - 1L
                  if (identical(
                              value,
                              "{"
                  )) curly <-  curly + 1L
                  if (identical(
                              value,
                              "}"
                  )) curly <-  curly - 1L
                  if (min(
                              round,
                              square,
                              curly
                        ) < 0L)
                        stop("Unbalanced generated Cox call arguments.")
                  if (identical(
                              value,
                              ","
                        ) && round == 0L && square == 0L && curly == 0L)
                        commas <-  c(
                        commas,
                        position
                  )

            }

            if (any(c(
                              round,
                              square,
                              curly
                        ) != 0L))
                  stop("Unbalanced generated Cox call arguments.")

            boundaries <-  c(
                  opening,
                  commas,
                  closing
            )
            first <-  head(
                  boundaries,
                  -1L
            ) + 1L
            last <-  tail(
                  boundaries,
                  -1L
            ) - 1L
            values <-  trimws(substring(
                        code,
                        first,
                        last
            ))
            return(list(
                        first = first,
                        last = last,
                        values = values
            ))

      }

      calls <-  list()

      ##
      ## ---- Locate the actual node calls --------------------------------------------------------------------------------------------
      ##
      for (position in occurrences) {

            after_name <-  position + nchar(target)
            suffix <-  substring(
                  visible,
                  after_name
            )
            opening <-  regexpr(
                  pattern = "^\\s*(?:<[^();{}]*>\\s*)?\\(",
                  text = suffix,
                  perl = TRUE
            )[1]
            if (opening < 0L) next

            span <-  attr(
                  regexpr(
                        pattern = "^\\s*(?:<[^();{}]*>\\s*)?\\(",
                        text = suffix,
                        perl = TRUE
                  ),
                  "match.length"
            )
            opening <-  after_name + span - 1L
            closing <-  fn_close(
                  visible,
                  opening,
                  "(",
                  ")"
            )
            if (position <= class_open || closing >= class_end)
                  stop("A Cox node call outside the model needs an explicit integration.")
            calls[[length(calls) + 1L]] <-  list(
                  first = position,
                  opening = opening,
                  closing = closing,
                  arguments = fn_arguments(
                        opening,
                        closing
                  )
            )

      }

      if (length(calls) == 0L) return(code)

      ##
      ## ---- Locate the data constructor and private fields --------------------------------------------------------------------------
      ##
      constructor_pattern <-  paste0(
            "\\b",
            class_name,
            "\\s*\\(\\s*stan::io::var_context\\s*&"
      )
      constructors <-  gregexpr(
            pattern = constructor_pattern,
            text = visible,
            perl = TRUE
      )[[1]]
      if (constructors[1] < 0L || length(constructors) != 1L)
            stop("Expected one generated data constructor for the Cox model.")

      signature_open <-  constructors[1] +
      regexpr(
            pattern = "\\(",
            text = substring(
                  visible,
                  constructors[1]
            )
      )[1] - 1L
      signature_end <-  fn_close(
            visible,
            signature_open,
            "(",
            ")"
      )
      initializer_pattern <-  paste0(
            "^\\s*:\\s*(?:stan::model::)?model_base_crtp",
            "\\s*\\(\\s*0\\s*\\)\\s*\\{"
      )
      initializer <-  regexpr(
            pattern = initializer_pattern,
            text = substring(
                  visible,
                  signature_end + 1L
            ),
            perl = TRUE
      )
      if (initializer[1] < 0L) stop("Unrecognised generated Cox constructor initializer.")

      constructor_open <-  signature_end + attr(
            initializer,
            "match.length"
      )
      constructor_end <-  fn_close(
            visible,
            constructor_open,
            "{",
            "}"
      )
      if (constructors[1] <= class_open || constructor_end >= class_end)
            stop("The Cox data constructor is outside the generated model.")

      private_prefix <-  substr(
            visible,
            class_open + 1L,
            constructors[1] - 1L
      )
      private_label <-  regexpr(
            pattern = "\\bprivate\\s*:",
            text = private_prefix,
            perl = TRUE
      )
      public_label <-  regexpr(
            pattern = "\\bpublic\\s*:",
            text = private_prefix,
            perl = TRUE
      )
      if (private_label[1] < 0L || public_label[1] < 0L || private_label[1] >= public_label[1])
            stop("Unrecognised generated Cox private fields.")

      private_end <-  class_open + public_label[1] - 1L
      private_fields <-  substr(
            visible,
            class_open + 1L,
            private_end
      )

      arrays <-  c(
            "group_decreasing_time",
            "rows_in_order_of_group",
            "last_position_of_each_group"
      )
      fields <-  c(
            paste0(
                  "\\bstd::vector\\s*<\\s*int\\s*>\\s+",
                  arrays,
                  "\\s*;"
            ),
            "\\bint\\s+N\\s*;",
            "\\bint\\s+J\\s*;"
      )

      ##
      ## ---- Require the exact data members used by the node -------------------------------------------------------------------------
      ##
      for (pattern in fields) {

            if (!grepl(
                        pattern = pattern,
                        x = private_fields,
                        perl = TRUE
            ))
                  stop("A required generated Cox index or dimension field is missing.")

      }

      member <-  "nico_checked_affine_indices_"
      context_type <-  "nico::math::checked_indices_of_affine_predictor_with_one_vector"
      member_pattern <-  paste0(
            "\\b",
            member,
            "\\b"
      )
      existing_member <-  grepl(
            pattern = member_pattern,
            x = visible,
            perl = TRUE
      )
      construction <-  paste0(
            "std::make_shared<const ",
            context_type,
            ">(",
            "this->",
            arrays[1],
            ", this->",
            arrays[2],
            ", this->",
            arrays[3],
            ", this->N, this->J)"
      )
      normalized_constructor <-  gsub(
            pattern = "\\s+",
            replacement = "",
            x = substr(
                  visible,
                  constructor_open + 1L,
                  constructor_end - 1L
            ),
            perl = TRUE
      )

      constructor_body <-  substr(
            visible,
            constructor_open + 1L,
            constructor_end - 1L
      )
      if (grepl(
                  pattern = "\\b(return|goto)\\b",
                  x = constructor_body,
                  perl = TRUE
      ))
            stop("The generated data constructor has unsupported early control flow.")

      ##
      ## ---- Keep node calls outside the data constructor ----------------------------------------------------------------------------
      ##
      for (call in calls) {

            if (call$first > constructors[1] && call$closing < constructor_end)
                  stop("The Cox node cannot use its checked context inside the data constructor.")

      }

      ##
      ## ---- Recognise only the exact completed transformation -----------------------------------------------------------------------
      ##
      if (existing_member) {

            expected_field <-  gsub(
                  "\\s+",
                  "",
                  paste0(
                        "std::shared_ptr<const ",
                        context_type,
                        "> ",
                        member,
                        ";"
                  )
            )
            expected_construction <-  gsub(
                  "\\s+",
                  "",
                  paste0(
                        member,
                        "=",
                        construction,
                        ";"
                  )
            )
            normalized_fields <-  gsub(
                  "\\s+",
                  "",
                  private_fields
            )
            if (!startsWith(
                        normalized_fields,
                        paste0(
                              "private:",
                              expected_field
                        )
                  ) ||
                  !endsWith(
                        normalized_constructor,
                        expected_construction
            ))
                  stop("The checked Cox context is not in its emitted declaration and constructor tail.")

            assignment <-  regexpr(
                  pattern = paste0(
                        member_pattern,
                        "\\s*="
                  ),
                  text = constructor_body,
                  perl = TRUE
            )[1]
            if (assignment < 0L) stop("The checked Cox context has no constructor assignment.")

            before_assignment <-  substr(
                  constructor_body,
                  1L,
                  assignment - 1L
            )
            delimiters <-  c(
                  "(",
                  ")",
                  "{",
                  "}",
                  "[",
                  "]"
            )

            ##
            ## ---- Require an outer constructor statement ----------------------------------------------------------------------------
            ##
            balance <-  vapply(
                  delimiters,
                  function(symbol) {

                        matches <-  gregexpr(
                              pattern = symbol,
                              text = before_assignment,
                              fixed = TRUE
                        )[[1]]
                        if (matches[1] < 0L) return(0L)
                        return(length(matches))

                  },
                  integer(1)
            )
            if (any(
                  balance[c(
                        1L,
                        3L,
                        5L
                  )] != balance[c(
                        2L,
                        4L,
                        6L
                  )]
            ))
                  stop("The checked Cox context is nested inside another expression or block.")

            prefix <-  trimws(before_assignment)
            if (nzchar(prefix) &&
                  !substr(
                        prefix,
                        nchar(prefix),
                        nchar(prefix)
                  ) %in% c(
                        ";",
                        "}"
            ))
                  stop("The checked Cox context does not follow a completed constructor statement.")

            uses <-  gregexpr(
                  pattern = member_pattern,
                  text = visible,
                  perl = TRUE
            )[[1]]
            if (length(uses) != length(calls) + 2L)
                  stop("Unexpected access to the generated Cox checked context.")

            ##
            ## ---- Verify every existing checked call --------------------------------------------------------------------------------
            ##
            for (call in calls) {

                  arguments <-  call$arguments$values
                  if (length(arguments) != 10L ||
                        !identical(
                              gsub(
                                    "\\s+",
                                    "",
                                    arguments[7]
                              ),
                              paste0(
                                    "*",
                                    member
                              )
                  ))
                        stop("An existing Cox node call is not the typed checked entry.")

            }

            return(code)

      }

      edits <-  list()
      private_insert <-  class_open + private_label[1] + attr(
            private_label,
            "match.length"
      )
      member_code <-  paste0(
            "\n  std::shared_ptr<const\n    ",
            context_type,
            "> ",
            member,
            ";"
      )
      edits[[length(edits) + 1L]] <-  list(
            first = private_insert,
            last = private_insert - 1L,
            value = member_code
      )
      cpp_constructor <-  c(
            paste0(
                  "    ",
                  member,
                  " ="
            ),
            "      std::make_shared<const",
            paste0(
                  "        ",
                  context_type,
                  ">("
            ),
            paste0(
                  "          this->",
                  arrays[1],
                  ","
            ),
            paste0(
                  "          this->",
                  arrays[2],
                  ","
            ),
            paste0(
                  "          this->",
                  arrays[3],
                  ","
            ),
            "          this->N,",
            "          this->J);"
      )
      construction_code <-  paste0(
            "\n",
            paste(
                  cpp_constructor,
                  collapse = "\n"
            ),
            "\n  "
      )
      edits[[length(edits) + 1L]] <-  list(
            first = constructor_end,
            last = constructor_end - 1L,
            value = construction_code
      )

      ##
      ## ---- Replace only the three raw index arguments ------------------------------------------------------------------------------
      ##
      for (call in calls) {

            arguments <-  call$arguments$values
            if (length(arguments) != 12L)
                  stop("Unrecognised generated Cox node signature.")

            supplied <-  gsub(
                  pattern = "\\s+",
                  replacement = "",
                  x = arguments[7:9],
                  perl = TRUE
            )
            if (!all(supplied == arrays | supplied == paste0(
                              "this->",
                              arrays
            )))
                  stop("The Cox node does not use the model's complete index arrays.")

            edits[[length(edits) + 1L]] <-  list(
                  first = call$arguments$first[7],
                  last = call$arguments$last[9],
                  value = paste0(
                        "*",
                        member
                  )
            )

      }

      ##
      ## ---- Include the owning pointer type explicitly ------------------------------------------------------------------------------
      ##
      if (!grepl(
                  pattern = "#include\\s*<memory>",
                  x = visible,
                  perl = TRUE
      )) {

            include <-  regexpr(
                  pattern = "#include\\s*<stan/model/model_header\\.hpp>",
                  text = visible,
                  perl = TRUE
            )
            if (include[1] < 0L) stop("Generated Stan model header include is missing.")

            include_end <-  include[1] + attr(
                  include,
                  "match.length"
            )
            edits[[length(edits) + 1L]] <-  list(
                  first = include_end,
                  last = include_end - 1L,
                  value = "\n#include <memory>"
            )

      }

      ##
      ## ---- Apply disjoint edits in reverse source order ----------------------------------------------------------------------------
      ##
      first <-  vapply(
            edits,
            function(edit) {

                  return(edit$first)

            },
            integer(1)
      )
      ##
      ## ---- Read the end of each edit -----------------------------------------------------------------------------------------------
      ##
      last <-  vapply(
            edits,
            function(edit) {

                  return(edit$last)

            },
            integer(1)
      )
      order <-  order(
            first,
            decreasing = TRUE
      )
      if (any(head(
                        first[order],
                        -1L
                  ) <= tail(
                        last[order],
                        -1L
      )))
            stop("Generated Cox edits overlap.")

      ##
      ## ---- Keep every other generated byte -----------------------------------------------------------------------------------------
      ##
      for (index in order) {

            edit <-  edits[[index]]
            code <-  paste0(
                  substr(
                        code,
                        1L,
                        edit$first - 1L
                  ),
                  edit$value,
                  substring(
                        code,
                        edit$last + 1L
                  )
            )

      }

      return(code)

}

