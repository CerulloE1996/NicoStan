##
## ==== The scalar type carried through helpers containing a vector map =======================================
##
R_fn_Stan_map_type_in_helper_functions <-  function(code) {

      ## The bracket, block and parameter readers follow the nico_reduce_sum model writer.
      visible <-  code
      protected_pattern <-  "\"(?:[^\"\\\\]|\\\\.)*\"|/\\*[\\s\\S]*?\\*/|//[^\\n]*"
      protected_matches <-  gregexpr(pattern = protected_pattern, text = code, perl = TRUE)[[1]]
      if (protected_matches[1] > 0) {
            lengths <-  attr(protected_matches, "match.length")
            for (index in seq_along(protected_matches)) {
                  first <-  protected_matches[index]
                  last <-  first + lengths[index] - 1
                  substr(visible, first, last) <-  strrep(" ", lengths[index])
            }
      }
      fn_closing_bracket <-  function(text, position) {
            opening <-  substr(text, position, position)
            closing <-  c("(" = ")", "[" = "]", "{" = "}", "<" = ">")[[opening]]
            characters <-  strsplit(substring(text, position), "", fixed = TRUE)[[1]]
            depth <-  0
            for (index in seq_along(characters)) {
                  if (characters[index] == opening) depth <-  depth + 1
                  if (characters[index] == closing) depth <-  depth - 1
                  if (depth == 0) return(position + index - 1)
            }
            stop("No closing bracket in the Stan map type arguments.")
      }
      fn_split <-  function(text, separator) {
            characters <-  strsplit(text, "", fixed = TRUE)[[1]]
            depth <-  0
            start <-  1
            pieces <-  character(0)
            for (index in seq_along(characters)) {
                  if (characters[index] %in% c("(", "[", "{")) depth <-  depth + 1
                  if (characters[index] %in% c(")", "]", "}")) depth <-  depth - 1
                  if (depth == 0 && characters[index] == separator) {
                        pieces <-  c(pieces, substr(text, start, index - 1))
                        start <-  index + 1
                  }
            }
            c(pieces, substr(text, start, nchar(text)))
      }
      blocks <-  list()
      position <-  1
      repeat {
            match <-  regexpr(pattern = paste0("^\\s*(functions|transformed data|data|",
                                               "transformed parameters|parameters|model|generated quantities)",
                                               "\\s*\\{"),
                              text = substring(visible, position), perl = TRUE)
            if (match[1] < 0) break
            matched <-  regmatches(substring(visible, position), match)
            name <-  gsub(pattern = "\\s+", replacement = " ",
                           x = trimws(sub(pattern = "\\{$", replacement = "", x = matched)))
            opening <-  position + attr(match, "match.length") - 1
            closing <-  fn_closing_bracket(visible, opening)
            blocks[[name]] <-  c(opening = opening, closing = closing)
            position <-  closing + 1
      }
      if (is.null(blocks$functions)) return(code)
      functions <-  list()
      position <-  blocks$functions[["opening"]] + 1
      last_function <-  blocks$functions[["closing"]]
      repeat {
            remaining <-  substr(visible, position, last_function - 1)
            match <-  regexpr(pattern = "\\b([A-Za-z_][A-Za-z0-9_]*)\\s*\\(",
                              text = remaining, perl = TRUE)
            if (match[1] < 0) break
            first <-  position + match[1] - 1
            opening <-  first + regexpr("\\(", regmatches(remaining, match))[1] - 1
            closing <-  fn_closing_bracket(visible, opening)
            after <-  closing + 1
            repeat {
                  if (after >= last_function || !grepl("\\s", substr(visible, after, after))) break
                  after <-  after + 1
            }
            following <-  substr(visible, after, after)
            if (!(following %in% c("{", ";"))) {
                  position <-  closing + 1
                  next
            }
            name <-  trimws(substr(visible, first, opening - 1))
            end <-  if (following == "{") fn_closing_bracket(visible, after) else after
            body <-  if (following == "{") substr(visible, after + 1, end - 1) else ""
            arguments <-  trimws(fn_split(substr(visible, opening + 1, closing - 1), ","))
            functions[[length(functions) + 1]] <-  list(name = name, first = first,
                  closing_arguments = closing, opening_body = after, closing_body = end,
                  body = body, arguments = arguments)
            position <-  end + 1
      }
      if (length(functions) == 0) return(code)
      map_calls <-  gregexpr(pattern = "\\bmap_rect\\s*\\(\\s*([A-Za-z_][A-Za-z0-9_]*)",
                             text = visible, perl = TRUE)[[1]]
      if (map_calls[1] < 0) return(code)
      map_jobs <-  if (map_calls[1] > 0) {
            sub(pattern = "^.*\\(\\s*", replacement = "", x = regmatches(visible, list(map_calls))[[1]])
      } else character(0)
      readers <-  c("nicostan_nuisance_value", "nicostan_nuisance_values", "nicostan_nuisance_values_at",
            "nicostan_nuisance_vector", "nicostan_nuisance_vector_at", "nicostan_nuisance_array_of_reals",
            "nicostan_nuisance_array_of_reals_at", "nicostan_nuisance_array_of_vectors",
            "nicostan_nuisance_array_of_row_vectors", "nicostan_nuisance_array_of_matrices",
            "nicostan_nuisance_matrix_of_terms", "nicostan_nuisance_matrix_block")
      reader_pattern <-  paste0("\\b(", paste(readers, collapse = "|"), ")\\s*\\(")
      helpers <-  unique(vapply(Filter(function(entry) {
            grepl(pattern = "\\bmap_rect\\s*\\(", x = entry$body, perl = TRUE) ||
                  grepl(pattern = reader_pattern, x = entry$body, perl = TRUE)
      }, functions), function(entry) entry$name, character(1)))
      helpers <-  setdiff(helpers, map_jobs)
      repeat {
            previous <-  helpers
            for (entry in functions) {
                  if (entry$name %in% map_jobs || length(helpers) == 0) next
                  sampled_names <-  sub("_lp[dm]f$", "", helpers[grepl("_lp[dm]f$", helpers)])
                  called_names <-  unique(c(helpers, sub("_lp([dm])f$", "_lup\\1f", helpers)))
                  pattern <-  paste0("(?:\\b(?:", paste(called_names, collapse = "|"), ")",
                                     if (length(sampled_names) > 0) {
                                           paste0("|~\\s*(?:", paste(sampled_names, collapse = "|"), ")\\b")
                                     } else "", ")\\s*\\(")
                  callback_pattern <-  paste0("\\breduce_sum(?:_static)?\\s*\\(\\s*(",
                                               paste(helpers, collapse = "|"), ")\\s*,")
                  if (grepl(pattern = pattern, x = entry$body, perl = TRUE) ||
                      grepl(pattern = callback_pattern, x = entry$body, perl = TRUE)) {
                        helpers <-  union(helpers, entry$name)
                  }
            }
            if (identical(previous, helpers)) break
      }
      ## This placeholder only keeps the source Stan call well-typed. The generated model's DUMMY_VAR__ replaces
      ## it after stanc, so neither a main parameter nor a map-data value supplies the automatic AD type.
      model_type <-  "0.0"
      type_name <-  "nicostan_type_for_vector_jobs"
      repeat {
            if (!grepl(paste0("\\b", type_name, "\\b"), visible, perl = TRUE)) break
            type_name <-  paste0(type_name, "_value")
      }
      insertions <-  list()
      for (entry in functions) {
            if (!(entry$name %in% helpers)) next
            separator <-  if (any(nzchar(entry$arguments))) ", " else ""
            insertions[[length(insertions) + 1]] <-  list(position = entry$closing_arguments,
                  text = paste0(separator, "real ", type_name))
      }
      calls <-  integer(0)
      if (length(helpers) > 0) {
            ## (a density or mass helper is also called as foo_lupdf(...) / foo_lupmf(...), and in a sampling
            ##  statement as y ~ foo(...), whose marker goes after foo's own arguments)
            called_names <-  unique(c(helpers, sub("_lp([dm])f$", "_lup\\1f", helpers)))
            sampled_names <-  sub("_lp[dm]f$", "", helpers[grepl("_lp[dm]f$", helpers)])
            calls <-  gregexpr(pattern = paste0("(?:\\b(?:", paste(called_names, collapse = "|"), ")",
                                                if (length(sampled_names) > 0) {
                                                      paste0("|~\\s*(?:", paste(sampled_names, collapse = "|"),
                                                             ")\\b")
                                                } else "", ")\\s*\\("),
                               text = visible, perl = TRUE)[[1]]
            callback_calls <-  gregexpr(pattern = paste0("\\breduce_sum(?:_static)?\\s*\\(\\s*(",
                                                         paste(unique(c(helpers,
                                                                        sub("_lp([dm])f$", "_lup\\1f",
                                                                            helpers))),
                                                               collapse = "|"), ")\\s*,"),
                                         text = visible, perl = TRUE)[[1]]
            calls <-  sort(unique(c(calls[calls > 0], callback_calls[callback_calls > 0])))
      }
      for (first in calls[calls > 0]) {
            if (any(vapply(functions, function(entry) entry$first == first, logical(1)))) next
            opening <-  first + regexpr("\\(", substring(visible, first))[1] - 1
            closing <-  fn_closing_bracket(visible, opening)
            owner <-  Filter(function(entry) first > entry$opening_body && first < entry$closing_body, functions)
            if (length(owner) > 0) {
                  entry <-  owner[[1]]
                  if (entry$name %in% helpers) {
                        marker <-  type_name
                  } else if (entry$name %in% map_jobs) {
                        argument <-  sub("^.*\\b([A-Za-z_][A-Za-z0-9_]*)$", "\\1", entry$arguments[1],
                                          perl = TRUE)
                        marker <-  paste0("sum(", argument, ")")
                  } else stop("No scalar type is available in a caller of a map helper.")
            } else {
                  in_value_block <-  any(vapply(blocks[intersect(names(blocks),
                        c("data", "transformed data", "generated quantities"))], function(block) {
                              first > block[["opening"]] && first < block[["closing"]]
                        }, logical(1)))
                  marker <-  if (in_value_block) "0.0" else model_type
            }
            separator <-  if (nzchar(trimws(substr(visible, opening + 1, closing - 1)))) ", " else ""
            insertions[[length(insertions) + 1]] <-  list(position = closing,
                  text = paste0(separator, marker))
      }
      reader_calls <-  gregexpr(pattern = reader_pattern, text = visible, perl = TRUE)[[1]]
      for (first in reader_calls[reader_calls > 0]) {
            if (any(vapply(functions, function(entry) entry$first == first, logical(1)))) next
            opening <-  first + regexpr("\\(", substring(visible, first))[1] - 1
            closing <-  fn_closing_bracket(visible, opening)
            owner <-  Filter(function(entry) first > entry$opening_body && first < entry$closing_body, functions)
            if (length(owner) > 0) {
                  entry <-  owner[[1]]
                  if (entry$name %in% helpers) {
                        marker <-  type_name
                  } else if (entry$name %in% map_jobs) {
                        argument <-  sub("^.*\\b([A-Za-z_][A-Za-z0-9_]*)$", "\\1", entry$arguments[1],
                                          perl = TRUE)
                        marker <-  paste0("sum(", argument, ")")
                  } else stop("No scalar type is available in a caller of a nuisance reader.")
            } else {
                  in_value_block <-  any(vapply(blocks[intersect(names(blocks),
                        c("data", "transformed data", "generated quantities"))], function(block) {
                              first > block[["opening"]] && first < block[["closing"]]
                        }, logical(1)))
                  marker <-  if (in_value_block) "0.0" else model_type
            }
            arguments <-  fn_split(substr(visible, opening + 1, closing - 1), ",")
            last_argument <-  arguments[length(arguments)]
            start <-  closing - nchar(last_argument)
            original_argument <-  substr(code, start, closing - 1)
            comments <-  gregexpr(pattern = "/\\*[\\s\\S]*?\\*/|//[^\\n]*",
                                  text = original_argument, perl = TRUE)
            saved_comments <-  regmatches(original_argument, comments)[[1]]
            if (length(saved_comments) > 0) {
                  marker <-  paste0(marker, " ", paste(saved_comments, collapse = "\n"), "\n")
            }
            insertions[[length(insertions) + 1]] <-  list(position = start,
                  last = closing - 1, text = marker)
      }
      replacements <-  Filter(function(entry) !is.null(entry$last), insertions)
      insertions <-  Filter(function(entry) {
            !any(vapply(replacements, function(replacement) {
                  entry$position > replacement$position && entry$position <= replacement$last
            }, logical(1)))
      }, insertions)
      order_of_insertions <-  order(vapply(insertions, function(entry) entry$position, numeric(1)),
                                    decreasing = TRUE)
      for (index in order_of_insertions) {
            insertion <-  insertions[[index]]
            last <-  if (is.null(insertion$last)) insertion$position - 1 else insertion$last
            code <-  paste0(substr(code, 1, insertion$position - 1), insertion$text,
                            substring(code, last + 1))
      }
      attr(code, "nicostan_map_type_carriers") <-  list(
            helper_functions = helpers,
            map_callbacks = map_jobs,
            nuisance_readers = readers)
      return(code)

}






















