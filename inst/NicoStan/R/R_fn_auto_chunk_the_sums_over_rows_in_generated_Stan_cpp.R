##
## ---- R_fn_auto_chunk_the_sums_over_rows_in_generated_Stan_cpp.R --------------------------------------------------------------------
##
## AUTO-CHUNKING of an unchanged Stan model: in the C++ that stanc writes (its default level), each
## stan::math::sum(E) in log_prob_impl whose argument E is row-wise (made only of the rows of the data,
## elementwise functions of them and of scalar parameters, and products of N-row data matrices with parameter
## vectors), and each density of the rows (a univariate density, or a GLM density with its coefficients shared),
## becomes nico::math::auto_chunking::sum_of_rows_in_the_chunks_of_reduce_sum_static(R,
## N_chunks_of_auto_chunking__, [&](first__, n_rows__, shared...) {...}, shared...): the same expression on each
## chunk of rows (exactly reduce_sum_static's chunks, chunk_size_of_auto_chunking = ceiling(R /
## N_chunks_of_auto_chunking), N_chunks_of_auto_chunking read from the model's data), its data read through
## rows_of(), and an N-row local used only inside that sum made inside the chunk. Anything not row-wise leaves the
## statement as stanc wrote it, and the report says which call stopped it.
## A density whose term of a row also reads the row before it (its contract in
## auto_chunking_contracts_of_row_sums_with_the_row_before.hpp) has the same chunks; each chunk whose first row
## does not start a group is also given the row before it (runtime: nico_auto_chunking_row_before.hpp).
## A block of the model whose rows are the rows 1 .. size(x) of a row array x, and whose only effect outside it is
## to add to the target, runs on each chunk of rows as its own statements, unchanged, with x and the data of N
## rows read at the chunk's rows; its first call is checked against the block on all its rows, and a block whose
## chunks do not add up to it is run as stanc wrote it (runtime: nico_auto_chunking_blocks_of_rows.hpp).
## A loop over groups of rows (e.g. the items, each a sum over the N persons), whose only effect outside it is to
## add to the target, runs whole on each chunk of rows, every group of the chunk in one tile, with N the chunk's
## number of rows, and is checked with the blocks.
## (spec: main_work_chat_1/AUTO_CHUNKING_converter_spec.md; runtime: nico_auto_chunking.hpp)
##
##
## ---- The tokens of a C++ expression ------------------------------------------------------------------------------------------------
##
fn_tokens_of_cpp_expression <-  function(expression_text) {

        pattern <-  paste0("[A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)*(?:<[^<>()]*>)?",
                           "|[0-9]+\\.?[0-9]*(?:[eE][-+]?[0-9]+)?",
                           "|\"[^\"]*\"",
                           "|[()+*/,-]",
                           "|\\S")
        tokens <-  regmatches(x = expression_text, m = gregexpr(pattern = pattern, text = expression_text,
                                                                perl = TRUE))[[1]]
        return(tokens)

}
##
## ---- The tree of a C++ expression (calls, + - * /, unary minus, brackets, names, numbers, strings) ---------------------------------
##
fn_tree_of_cpp_expression <-  function(tokens) {

        state <-  new.env()
        state$position <-  1
        ##
        ## ---- The next token, and moving past it ------------------------------------------------------------------------------------
        ##
        peek <-  function() {

                if (state$position > length(tokens)) return("")
                return(tokens[[state$position]])

        }
        take <-  function(expected) {

                token <-  peek()
                if (!is.null(expected) && (token != expected)) stop("expected ", expected, ", found ", token)
                state$position <-  state$position + 1
                return(token)

        }
        ##
        ## ---- expression := term (('+' | '-') term)* --------------------------------------------------------------------------------
        ##
        parse_expression <-  function() {

                node <-  parse_term()
                while (peek() %in% c("+", "-")) {
                        operator <-  take(expected = NULL)
                        node <-  list(type = "operator", operator = operator, args = list(node, parse_term()))
                }
                return(node)

        }
        ##
        ## ---- term := unary (('*' | '/') unary)* ------------------------------------------------------------------------------------
        ##
        parse_term <-  function() {

                node <-  parse_unary()
                while (peek() %in% c("*", "/")) {
                        operator <-  take(expected = NULL)
                        node <-  list(type = "operator", operator = operator, args = list(node, parse_unary()))
                }
                return(node)

        }
        ##
        ## ---- unary := '-' unary | primary ------------------------------------------------------------------------------------------
        ##
        parse_unary <-  function() {

                if (peek() == "-") {
                        take(expected = "-")
                        return(list(type = "minus", args = list(parse_unary())))
                }
                return(parse_primary())

        }
        ##
        ## ---- primary := number | string | name ['(' arguments ')'] | '(' expression ')' --------------------------------------------
        ##
        parse_primary <-  function() {

                token <-  take(expected = NULL)
                if (token == "(") {
                        node <-  parse_expression()
                        take(expected = ")")
                        return(list(type = "brackets", args = list(node)))
                }
                if (grepl(pattern = "^[0-9]", x = token)) return(list(type = "number", value = token))
                if (grepl(pattern = "^\"", x = token)) return(list(type = "string", value = token))
                if (!grepl(pattern = "^[A-Za-z_]", x = token)) stop("unexpected token ", token)
                if (peek() != "(") return(list(type = "name", value = token))
                take(expected = "(")
                arguments <-  list()
                if (peek() != ")") {
                        arguments[[1]] <-  parse_expression()
                        while (peek() == ",") {
                                take(expected = ",")
                                arguments[[length(arguments) + 1]] <-  parse_expression()
                        }
                }
                take(expected = ")")
                return(list(type = "call", name = token, args = arguments))

        }
        ##
        tree <-  parse_expression()
        if (state$position <= length(tokens)) stop("tokens left over after the expression: ", peek())
        return(tree)

}
##
## ---- The text of a tree ------------------------------------------------------------------------------------------------------------
##
fn_text_of_cpp_tree <-  function(node) {

        if (node$type %in% c("number", "string", "name")) return(node$value)
        if (node$type == "brackets") return(paste0("(", fn_text_of_cpp_tree(node = node$args[[1]]), ")"))
        if (node$type == "minus") return(paste0("-(", fn_text_of_cpp_tree(node = node$args[[1]]), ")"))
        if (node$type == "operator") {
                return(paste0("(", fn_text_of_cpp_tree(node = node$args[[1]]), " ", node$operator, " ",
                              fn_text_of_cpp_tree(node = node$args[[2]]), ")"))
        }
        arguments <-  vapply(X = node$args, FUN = function(argument) fn_text_of_cpp_tree(node = argument),
                             FUN.VALUE = character(1))
        return(paste0(node$name, "(", paste(arguments, collapse = ", "), ")"))

}
##
## ---- The data of the Stan model: each name with its first dimension (its row set) and its shape ------------------------------------
##
fn_data_of_Stan_model <-  function(Stan_model_lines) {

        text <-  paste(Stan_model_lines, collapse = "\n")
        text <-  gsub(pattern = "//[^\n]*", replacement = "", x = text)
        ##
        ## ---- the data and transformed data blocks ----------------------------------------------------------------------------------
        ##
        blocks_text <-  character(0)
        for (block_name in c("data", "transformed data")) {
                start <-  regexpr(pattern = paste0("(^|\n)\\s*", block_name, "\\s*\\{"), text = text)
                if (start < 0) next
                position <-  start + attr(start, "match.length")
                depth <-  1
                characters <-  strsplit(x = substring(text, position, nchar(text)), split = "")[[1]]
                end <-  0
                for (i in seq_along(characters)) {
                        if (characters[[i]] == "{") depth <-  depth + 1
                        if (characters[[i]] == "}") depth <-  depth - 1
                        if (depth == 0) {
                                end <-  i
                                break
                        }
                }
                blocks_text <-  c(blocks_text, substring(text, position, position + end - 2))
        }
        statements <-  trimws(unlist(strsplit(x = paste(blocks_text, collapse = "\n"), split = ";")))
        ##
        ## ---- each declaration: vectors, matrices and arrays of reals with a first dimension, and scalars ---------------------------
        ##
        data <-  list()
        pattern_of_a_container <-  paste0("^(vector|row_vector|matrix|array) ?\\[ ?([^],]+)",
                                          "(?:,[^]]*)?\\] ?(real |int )?([A-Za-z_][A-Za-z0-9_]*)")
        for (statement in statements) {
                statement <-  gsub(pattern = "\\s+", replacement = " ", x = statement)
                ## (a declaration right after the end of a block: the closing braces before it are not part of it)
                statement <-  sub(pattern = "^[} ]+", replacement = "", x = statement)
                statement <-  sub(pattern = "<[^>]*>", replacement = "", x = statement)
                matched <-  regmatches(x = statement,
                                       m = regexec(pattern = pattern_of_a_container, text = statement,
                                                   perl = TRUE))[[1]]
                if (length(matched) == 5) {
                        shape <-  if (matched[[2]] %in% c("matrix", "row_vector")) matched[[2]] else "vector"
                        if ((matched[[2]] == "array") && (trimws(matched[[4]]) == "int")) shape <-  "integers"
                        if ((matched[[2]] == "array") && (trimws(matched[[4]]) == "real")) shape <-  "reals"
                        data[[matched[[5]]]] <-  list(first_dimension = trimws(matched[[3]]), shape = shape)
                        next
                }
                matched <-  regmatches(x = statement,
                                       m = regexec(pattern = "^(real|int) ([A-Za-z_][A-Za-z0-9_]*)",
                                                   text = statement))[[1]]
                if (length(matched) == 3) data[[matched[[3]]]] <-  list(first_dimension = NA_character_,
                                                                       shape = "scalar")
        }
        return(data)

}
##
## ---- The kind of each name of log_prob_impl: parameters and locals, scalar or vector -----------------------------------------------
##
fn_parameters_and_locals_of_log_prob <-  function(body_text,
                                                  names_of_the_nuisance_blocks) {

        kinds <-  list()
        name_pattern <-  "([A-Za-z_][A-Za-z0-9_]*)"
        ##
        ## ---- the parameters read from in__ -----------------------------------------------------------------------------------------
        ##
        pattern_of_a_read <-  paste0("auto\\s+", name_pattern, "\\s*=\\s*in__\\.template\\s+",
                                     "read[A-Za-z_]*<\\s*([^,>]+(?:<[^>]*>)?)")
        reads <-  regmatches(x = body_text,
                             m = gregexpr(pattern = pattern_of_a_read, text = body_text, perl = TRUE))[[1]]
        for (read in reads) {
                name <-  sub(pattern = "^auto\\s+([A-Za-z_][A-Za-z0-9_]*).*$", replacement = "\\1", x = read)
                kinds[[name]] <-  if (grepl(pattern = "Eigen::Matrix<local_scalar_t__", x = read)) "var_vector"
                                  else "var_scalar"
        }
        ##
        ## ---- the locals of local_scalar_t__ ----------------------------------------------------------------------------------------
        ##
        pattern_of_a_scalar_local <-  paste0("local_scalar_t__\\s+", name_pattern, "\\s*=\\s*DUMMY_VAR__")
        scalar_locals <-  regmatches(x = body_text,
                                     m = gregexpr(pattern = pattern_of_a_scalar_local, text = body_text,
                                                  perl = TRUE))[[1]]
        for (local in scalar_locals) {
                kinds[[sub(pattern = paste0("^local_scalar_t__\\s+", name_pattern, ".*$"), replacement = "\\1",
                           x = local)]] <-  "var_scalar"
        }
        vector_of_locals <-  "Eigen::Matrix<local_scalar_t__,\\s*-1,\\s*1>"
        pattern_of_a_vector_local <-  paste0(vector_of_locals, "\\s+", name_pattern, "\\s*=\\s*",
                                             vector_of_locals, "::Constant\\(([^;]*),\\s*DUMMY_VAR__\\)")
        vector_locals <-  regmatches(x = body_text,
                                     m = gregexpr(pattern = pattern_of_a_vector_local, text = body_text,
                                                  perl = TRUE))[[1]]
        for (local in vector_locals) {
                pattern_of_its_size <-  paste0(">\\s+", name_pattern,
                                               "\\s*=.*Constant\\(\\s*([^;]*),\\s*DUMMY_VAR__\\)")
                matched <-  regmatches(x = local, m = regexec(pattern = pattern_of_its_size, text = local))[[1]]
                kinds[[matched[[2]]]] <-  list(kind = "var_vector_local", size = trimws(matched[[3]]))
        }
        ## (matrix locals of local_scalar_t__: shared, never rows)
        pattern_of_a_matrix_local <-  paste0("Eigen::Matrix<local_scalar_t__,\\s*-1,\\s*-1>\\s+", name_pattern,
                                             "\\s*=")
        matrix_locals <-  regmatches(x = body_text,
                                     m = gregexpr(pattern = pattern_of_a_matrix_local, text = body_text,
                                                  perl = TRUE))[[1]]
        for (local in matrix_locals) {
                kinds[[sub(pattern = paste0("^.*>\\s+", name_pattern, "\\s*=$"), replacement = "\\1",
                           x = local)]] <-  "var_matrix"
        }
        ##
        ## ---- the size of each parameter vector (an array of reals or a vector), and whether it is an Eigen vector ----------------
        ##
        ## (a parameter vector whose size is a row set is read by rows in a tile; an element of an Eigen vector
        ##  parameter, x[i], is a shared scalar)
        pattern_of_a_vector_read <-  paste0("auto\\s+", name_pattern, "\\s*=\\s*in__\\.template\\s+",
                                            "read[A-Za-z_]*<\\s*(std::vector<local_scalar_t__>|",
                                            "Eigen::Matrix<local_scalar_t__,\\s*-1,\\s*1>)[^;]*\\(([^;]*)\\);")
        vector_reads <-  regmatches(x = body_text,
                                    m = gregexpr(pattern = pattern_of_a_vector_read, text = body_text,
                                                 perl = TRUE))[[1]]
        sizes <-  character(0)
        is_Eigen_vector <-  logical(0)
        for (read in vector_reads) {
                matched <-  regmatches(x = read, m = regexec(pattern = pattern_of_a_vector_read, text = read,
                                                             perl = TRUE))[[1]]
                arguments <-  trimws(strsplit(x = matched[[4]], split = ",")[[1]])
                sizes[[matched[[2]]]] <-  arguments[[length(arguments)]]
                is_Eigen_vector[[matched[[2]]]] <-  startsWith(matched[[3]], "Eigen::")
        }
        ##
        ## ---- Parameter dimensions and declared nuisance membership ----------------------------------------------------------------
        ##
        dimensions <-  list()
        parameter_names <-  character(0)
        for (read in reads) {

                name <-  sub("^auto\\s+([A-Za-z_][A-Za-z0-9_]*).*$", "\\1", read)
                parameter_names <-  c(parameter_names, name)

        }
        starts <-  gregexpr("auto\\s+([A-Za-z_][A-Za-z0-9_]*)\\s*=\\s*in__\\.template\\s+read", body_text,
                            perl = TRUE)[[1]]
        for (start in starts[starts > 0L]) {

                remaining <-  substring(body_text, start)
                name <-  sub("^auto\\s+([A-Za-z_][A-Za-z0-9_]*).*$", "\\1", remaining)
                opening <-  regexpr("<", remaining)[[1]]
                depth <-  0L
                closing <-  NA_integer_
                characters <-  strsplit(remaining, "")[[1]]
                for (i in seq.int(opening, length(characters))) {

                        if (characters[[i]] == "<") depth <-  depth + 1L
                        if (characters[[i]] == ">") depth <-  depth - 1L
                        if (depth == 0L) {

                                closing <-  i
                                break

                        }

                }
                if (is.na(closing)) stop("Unclosed parameter read for ", name)
                arguments_open <-  closing + regexpr("\\(", substring(remaining, closing))[[1]] - 1L
                arguments_close <-  fn_end_of_bracket(remaining, arguments_open)
                arguments <-  fn_arguments_of_a_call_text(substring(remaining,
                                                                    arguments_open + 1L,
                                                                    arguments_close - 1L))
                type <-  substring(remaining, opening + 1L, closing - 1L)
                is_array <-  grepl("std::vector<", type, fixed = TRUE)
                is_matrix <-  grepl("Eigen::Matrix<", type, fixed = TRUE)
                if (!is_array && !is_matrix) next
                kinds[[name]] <-  if (is_matrix && !is_array &&
                                      grepl(",-1,-1>", gsub("\\s+", "", type))) "var_matrix" else "var_vector"
                n_dimensions <-  lengths(regmatches(type, gregexpr("std::vector<", type, fixed = TRUE)))
                if (is_matrix) n_dimensions <-  n_dimensions +
                        if (grepl(",-1,-1>", gsub("\\s+", "", type))) 2L else 1L
                if (length(arguments) < n_dimensions) stop("Missing parameter dimensions for ", name)
                dimensions[[name]] <-  tail(arguments, n_dimensions)
                if (!is_array && is_matrix && n_dimensions == 1L) sizes[[name]] <-  tail(arguments, 1L)

        }
        attr(kinds, "parameter_dimensions") <-  dimensions
        attr(kinds, "parameter_names") <-  unique(parameter_names)
        attr(kinds, "names_of_the_nuisance_blocks") <-  names_of_the_nuisance_blocks
        attr(kinds, "sizes_of_the_parameter_vectors") <-  sizes
        attr(kinds, "is_Eigen_vector_parameter") <-  is_Eigen_vector
        return(kinds)

}
##
## ---- The end of the bracket that opens at a position (no string holds a bracket in stanc's log_prob) -------------------------------
##
fn_end_of_bracket <-  function(text,
                               position_of_open_bracket) {

        characters <-  strsplit(x = substring(text, position_of_open_bracket, nchar(text)), split = "")[[1]]
        depth <-  0
        for (i in seq_along(characters)) {
                if (characters[[i]] == "(") depth <-  depth + 1
                if (characters[[i]] == ")") {
                        depth <-  depth - 1
                        if (depth == 0) return(position_of_open_bracket + i - 1)
                }
        }
        stop("no closing bracket")

}
##
## ---- The assignment of an N-row local (stan::model::assign(local, E2, "...")): its expression and where it is ----------------------
##
fn_assignment_of_local <-  function(body_text,
                                    local_name) {

        start <-  regexpr(pattern = paste0("stan::model::assign\\(\\s*", local_name, "\\s*,"), text = body_text)
        if (start < 0) return(NULL)
        open_bracket <-  start + nchar("stan::model::assign")
        close_bracket <-  fn_end_of_bracket(text = body_text, position_of_open_bracket = open_bracket)
        inner <-  substring(body_text, open_bracket + 1, close_bracket - 1)
        tokens <-  c("f", "(", fn_tokens_of_cpp_expression(expression_text = inner), ")")
        tree <-  fn_tree_of_cpp_expression(tokens = tokens)
        semicolon <-  regexpr(pattern = "^\\s*;",
                              text = substring(body_text, close_bracket + 1, nchar(body_text)))
        return(list(expression = tree$args[[2]], start = start,
                    end = close_bracket + attr(semicolon, "match.length")))

}
##
## ---- The N-row locals as stanc made them: each one's declaration and assignment, in stanc's order ----------------------------------
##
## (NULL if one of them is not found as stanc writes it; "" for none)
fn_statements_of_the_locals_as_stanc_made_them <-  function(body_text,
                                                            names_of_the_locals) {

        if (length(names_of_the_locals) == 0) return("")
        vector_of_locals <-  "Eigen::Matrix<local_scalar_t__,\\s*-1,\\s*1>"
        statements <-  character(0)
        starts <-  numeric(0)
        for (local_name in names_of_the_locals) {
                assignment <-  fn_assignment_of_local(body_text = body_text, local_name = local_name)
                declaration_pattern <-  paste0(vector_of_locals, "\\s+", local_name, "\\s*=\\s*",
                                               vector_of_locals, "::Constant\\(", "[^;]*\\);")
                declaration <-  regmatches(body_text, regexpr(pattern = declaration_pattern, text = body_text,
                                                              perl = TRUE))
                validation_pattern <-  paste0("stan::math::validate_non_negative_index\\(\\s*\"", local_name,
                                              "\"[^;]*\\);")
                validation <-  regmatches(body_text, regexpr(pattern = validation_pattern, text = body_text,
                                                             perl = TRUE))
                if (is.null(assignment) || (length(declaration) != 1)) return(NULL)
                ## (stanc's order: the size check, the declaration, the assignment)
                statements <-  c(statements, paste0(if (length(validation) == 1) paste0(validation, " ") else "",
                                                    declaration, " ",
                                                    substring(body_text, assignment$start, assignment$end)))
                starts <-  c(starts, assignment$start)
        }
        return(paste0(paste(statements[order(starts)], collapse = " "), " "))

}
##
## ---- A slice of the data, D[a:b] (stanc: rvalue(D, "D", index_min_max(a, b))): its name and bounds (NULL if not one) ---------------
##
fn_slice_of_the_data <-  function(node,
                                  data) {

        if ((node$type != "call") || (node$name != "stan::model::rvalue") || (length(node$args) != 3) ||
            (node$args[[1]]$type != "name") || is.null(data[[node$args[[1]]$value]]) ||
            !(data[[node$args[[1]]$value]]$shape %in% c("integers", "reals", "vector")) ||
            (node$args[[3]]$type != "call") || (node$args[[3]]$name != "stan::model::index_min_max") ||
            (length(node$args[[3]]$args) != 2)) {

                return(NULL)

        }
        return(list(name = node$args[[1]]$value, shape = data[[node$args[[1]]$value]]$shape,
                    lower = fn_text_of_cpp_tree(node = node$args[[3]]$args[[1]]),
                    upper = fn_text_of_cpp_tree(node = node$args[[3]]$args[[2]])))

}
##
## ---- The row set whose length is the slice's length (NA if none) ---------------------------------------------------------------------
##
## (a row set of the data, or an integer scalar of the data, equal to b - a + 1 at three draws of random integers for
##  the names in a, b and the candidate; the tiles check the length again when they run)
fn_row_set_of_the_slice <-  function(slice,
                                     data) {

        length_text <-  paste0("((", slice$upper, ") - (", slice$lower, ") + 1)")
        allowed <-  "^[A-Za-z0-9_ ()+*-]+$"
        if (!grepl(pattern = allowed, x = length_text)) return(NA_character_)
        first_dimensions <-  unique(na.omit(vapply(X = data, FUN = function(entry) entry$first_dimension,
                                                   FUN.VALUE = character(1))))
        integer_scalars <-  names(data)[vapply(X = data, FUN = function(entry) entry$shape == "scalar",
                                               FUN.VALUE = logical(1))]
        candidates <-  unique(c(first_dimensions, integer_scalars))
        candidates <-  candidates[grepl(pattern = allowed, x = candidates)]
        names_in <-  function(text) unique(unlist(regmatches(x = text, m = gregexpr(pattern = "[A-Za-z_][A-Za-z0-9_]*",
                                                                                     text = text))))
        for (candidate in candidates) {

                names_of_both <-  names_in(c(length_text, candidate))
                same <-  vapply(X = c(11L, 23L, 37L), FUN = function(seed) {
                        set.seed(seed)
                        values <-  as.list(setNames(sample(3:97, length(names_of_both), replace = TRUE), names_of_both))
                        isTRUE(eval(parse(text = length_text), envir = values) ==
                                       eval(parse(text = candidate), envir = values))
                }, FUN.VALUE = logical(1))
                if (all(same)) return(candidate)

        }
        return(NA_character_)

}
##
## ---- The row sets of the slices of the data in a candidate -----------------------------------------------------------------------------
##
fn_row_sets_of_the_slices_in <-  function(node,
                                          data) {

        slice <-  fn_slice_of_the_data(node = node, data = data)
        if (!is.null(slice)) return(na.omit(fn_row_set_of_the_slice(slice = slice, data = data)))
        if (is.null(node$args)) return(character(0))
        return(unique(unlist(lapply(X = node$args, FUN = function(argument) {
                fn_row_sets_of_the_slices_in(node = argument, data = data)
        }))))

}
##
## ---- The data with the row sets of a candidate's slices of the data (so that its vectors of that length are rows) ----------------
##
fn_data_with_the_row_sets_of_the_slices <-  function(node,
                                                     data) {

        row_sets <-  fn_row_sets_of_the_slices_in(node = node, data = data)
        if (length(row_sets) == 0) return(data)
        row_sets_of_the_slices <-  lapply(X = row_sets, FUN = function(row_set) {
                list(first_dimension = row_set, shape = "row set of a slice")
        })
        names(row_sets_of_the_slices) <-  paste0("__row_set_of_a_slice_", seq_along(row_sets))
        result <-  c(data, row_sets_of_the_slices)
        attr(result, "scalar_dimension_aliases") <-  attr(data, "scalar_dimension_aliases")
        return(result)

}
##
## ---- An index that reads no rows: only integer scalars of the data and loop variables -----------------------------------------------
##
fn_check_that_an_index_reads_no_rows <-  function(index_text,
                                                  data,
                                                  parameters_and_locals,
                                                  name) {

        names_in_the_index <-  unlist(regmatches(x = index_text, m = gregexpr(pattern = "[A-Za-z_][A-Za-z0-9_]*",
                                                                              text = index_text)))
        for (index_name in names_in_the_index) {

                is_data_scalar <-  !is.null(data[[index_name]]) && (data[[index_name]]$shape == "scalar")
                if (!is.null(data[[index_name]]) && !is_data_scalar) {
                        stop("the index of ", name, " reads the data ", index_name)
                }
                if (!is.null(parameters_and_locals[[index_name]])) stop("the index of ", name, " reads ", index_name)

        }
        return(invisible(TRUE))

}
##
## ---- Row-wise or not: the kind of a node, and its text inside a tile ---------------------------------------------------------------
##
## kind: "rows" (row set, var or data, vector or matrix), "scalar" or "vector" (not row-wise; var or data), or
## "stream"; names_of_vars: the var names it reads (the shared arguments); moved: the N-row locals made inside the
## tile.
## struct_of_arrays: TRUE inside a sum (the tile's var vectors as Stan's struct-of-arrays vectors, for the nico_
## elementwise overloads), FALSE inside a density (Stan's array-of-vars vectors, which every density takes).
fn_row_wise_kind_of_node <-  function(node,
                                      data,
                                      parameters_and_locals,
                                      body_text,
                                      struct_of_arrays,
                                      start_of_the_loop) {

        elementwise_functions <-  paste0("stan::math::",
                                         c("add", "subtract", "elt_multiply", "elt_divide", "divide", "minus",
                                           "square", "pow", "sqrt", "exp", "log", "log1p", "log1m", "inv_logit",
                                           "log_inv_logit", "log1p_exp", "Phi", "inv", "fabs", "abs", "multiply",
                                           "to_vector", "fma"))
        ## (the nico_ functions of nico_auto_chunking.hpp's struct-of-arrays overloads only; any other nico_ call
        ##  of the rows, e.g. a reduction like nico_log_sum_exp, leaves the sum as stanc wrote it)
        elementwise_nico_functions <-  paste0("nico_", c("exp", "log", "log1p", "log1m", "log1p_exp", "inv_logit",
                                                         "log_inv_logit", "Phi"))
        result <-  function(kind,
                            row_set,
                            is_var,
                            shape,
                            text,
                            names_of_vars,
                            moved) {

                return(list(kind = kind, row_set = row_set, is_var = is_var, shape = shape, text = text,
                            names_of_vars = names_of_vars, moved = moved))

        }
        first_dimension_of <-  function(entry) entry$first_dimension
        row_sets_of_the_data <-  unique(na.omit(vapply(X = data, FUN = first_dimension_of,
                                                       FUN.VALUE = character(1))))
        ## (a vector of vars read by rows: each tile is given a copy of its own rows only, and adds their adjoints
        ##  back at those rows, as reduce_sum's sliced argument; the tile's rows as Stan's struct-of-arrays var
        ##  vector inside a sum, and as an Eigen vector of vars inside a density)
        sliced_at_rows <-  function(name,
                                    row_set) {

                lambda_parameter <-  paste0(name, "_rows__")
                slice <-  if (struct_of_arrays) "sliced_at_rows_of(" else "sliced_at_rows_as_vars_of("
                return(result("rows", row_set, TRUE, "vector", lambda_parameter,
                              setNames(paste0("nico::math::auto_chunking::", slice, name, ")"), lambda_parameter),
                              list()))

        }
        ##
        ## ---- names: the rows of the data, parameters and locals, the stream, data scalars ------------------------------------------
        ##
        if (node$type == "number") return(result("scalar", NA, FALSE, "scalar", node$value, character(0), list()))
        if (node$type == "string") stop("a string inside the sum")
        if (node$type == "name") {
                name <-  node$value
                if (name == "pstream__") return(result("stream", NA, FALSE, NA, name, character(0), list()))
                if (!is.null(data[[name]])) {
                        entry <-  data[[name]]
                        if (entry$shape == "scalar") {
                                return(result("scalar", NA, FALSE, "scalar", name, character(0), list()))
                        }
                        ## (a row vector's product with a vector is a dot product, not a row set: never tiled)
                        if (entry$shape == "row_vector") stop("the row vector ", name, " inside the sum")
                        ## (an array of integers: its rows' values as doubles, as Stan's to_vector() of it)
                        return(result("rows", entry$first_dimension, FALSE,
                                      if (entry$shape %in% c("integers", "reals")) "vector" else entry$shape,
                                      paste0("nico::math::auto_chunking::rows_of(", name, ", first__, n_rows__)"),
                                      character(0), list()))
                }
                kind <-  parameters_and_locals[[name]]
                ## (a parameter vector whose size is a row set: sliced)
                size_of_the_parameter <-  attr(parameters_and_locals, "sizes_of_the_parameter_vectors")[name]
                if ((name %in% attr(parameters_and_locals, "names_of_the_nuisance_blocks")) &&
                    !is.na(size_of_the_parameter)) {
                        return(sliced_at_rows(name = name, row_set = size_of_the_parameter))
                }
                if (is.null(kind)) stop("unknown name ", name)
                ## (a shared argument: the name of the tile's lambda parameter, and the argument given for it)
                shared_name <-  setNames(name, name)
                if (identical(kind, "var_scalar")) {
                        return(result("scalar", NA, TRUE, "scalar", name, shared_name, list()))
                }
                if (identical(kind, "var_vector") || identical(kind, "var_matrix")) {
                        return(result("vector", NA, TRUE, "vector", name, shared_name, list()))
                }
                ## (a vector local whose size is not a row set: a shared vector, like a parameter vector)
                if (!(kind$size %in% row_sets_of_the_data)) {
                        return(result("vector", NA, TRUE, "vector", name, shared_name, list()))
                }
                ##
                ## ---- an N-row local: made inside the tile from its own assignment --------------------------------------------------
                ##
                assignment <-  fn_assignment_of_local(body_text = body_text, local_name = name)
                if (is.null(assignment)) stop("no assignment of the local ", name)
                ## (one made before the loop that holds the sum: made once, where stanc made it, and sliced)
                if (!is.na(start_of_the_loop) && (assignment$start < start_of_the_loop)) {
                        return(sliced_at_rows(name = name, row_set = kind$size))
                }
                inner <-fn_row_wise_kind_of_node(node = assignment$expression, data = data,
                                                   parameters_and_locals = parameters_and_locals,
                                                   body_text = body_text, struct_of_arrays = struct_of_arrays,
                                                   start_of_the_loop = start_of_the_loop)
                if ((inner$kind != "rows") || (inner$row_set != kind$size)) {
                        stop("the local ", name, " is not row-wise")
                }
                moved <-  c(inner$moved, setNames(list(inner$text), name))
                return(result("rows", kind$size, inner$is_var, "vector", name, inner$names_of_vars, moved))
        }
        ##
        ## ---- A nuisance reader used directly by its own elementwise prior ---------------------------------------------------------
        ##
        if ((node$type == "call") &&
            (node$name %in% c("nicostan_nuisance_vector", "nicostan_nuisance_array_of_reals")) &&
            (length(node$args) == 5L) && (node$args[[1]]$type == "number") &&
            (node$args[[1]]$value == "1") && (node$args[[2]]$type == "name") &&
            (node$args[[3]]$type == "name") && startsWith(node$args[[3]]$value, "nicostan_offset_of_")) {

                block <-  sub("^nicostan_offset_of_", "", node$args[[3]]$value)
                terms <-  paste0("nicostan_terms_of_", block)
                if (node$args[[2]]$value != terms) stop("The nuisance reader has inconsistent term metadata")
                parameter <-  paste0(block, "_prior_rows__")
                text <-  fn_text_of_cpp_tree(node)
                return(result("rows", terms, TRUE, "vector", parameter,
                              setNames(paste0("nico::math::auto_chunking::sliced_at_rows_as_vars_of(",
                                              text, ")"), parameter), list()))

        }
        ##
        ## ---- A flattened nuisance block or a local made directly from that block --------------------------------------------------
        ##
        if ((node$type == "call") && (node$name == "stan::math::to_vector") &&
            (length(node$args) == 1L) && (node$args[[1]]$type == "name")) {

                name <-  node$args[[1]]$value
                source_name <-  name
                dimensions <-  attr(parameters_and_locals, "parameter_dimensions")[[name]]
                assignment <-  fn_assignment_of_local(body_text, name)
                source <-  if (is.null(assignment)) NULL else assignment$expression
                reader <-  FALSE
                if (!is.null(source) && source$type == "call" && source$name == "stan::math::to_matrix" &&
                    length(source$args) == 1L && source$args[[1]]$type == "name") {

                        source_name <-  source$args[[1]]$value
                        dimensions <-  attr(parameters_and_locals, "parameter_dimensions")[[source_name]]

                }
                if (!is.null(source) && source$type == "call" &&
                    source$name == "nicostan_nuisance_matrix_of_terms" && length(source$args) == 6L &&
                    source$args[[1]]$type == "number" && source$args[[1]]$value == "1" &&
                    source$args[[2]]$type == "name" && source$args[[3]]$type == "name" &&
                    source$args[[4]]$type == "name") {

                        block <-  sub("^nicostan_offset_of_", "", source$args[[4]]$value)
                        reader <-  identical(source$args[[2]]$value, paste0("nicostan_terms_of_", block)) &&
                                   identical(source$args[[3]]$value, paste0("nicostan_values_per_term_of_", block))
                        if (reader) dimensions <-  c(source$args[[2]]$value, source$args[[3]]$value)

                }
                declared <-  source_name %in% attr(parameters_and_locals, "names_of_the_nuisance_blocks")
                if ((declared || reader) && length(dimensions) > 0L) {

                        row_set <-  paste0("(", paste(dimensions, collapse = " * "), ")")
                        parameter <-  paste0(name, "_prior_rows__")
                        return(result("rows", row_set, TRUE, "vector", parameter,
                                      setNames(paste0("nico::math::auto_chunking::sliced_at_rows_as_vars_of(",
                                                      fn_text_of_cpp_tree(node), ")"), parameter), list()))

                }

        }
        ##
        ## ---- a slice of the data, D[a:b] (stanc: rvalue(D, "D", index_min_max(a, b))): the rows of its row set --------------------
        ##
        slice <-  fn_slice_of_the_data(node = node, data = data)
        if (!is.null(slice)) {
                row_set <-  fn_row_set_of_the_slice(slice = slice, data = data)
                if (is.na(row_set)) stop("the length of the slice of ", slice$name, " is not a row set")
                return(result("rows", row_set, FALSE, "vector",
                              paste0("nico::math::auto_chunking::rows_of_slice(", slice$name, ", ", slice$lower, ", ",
                                     slice$upper, ", ", row_set, ", first__, n_rows__)"),
                              character(0), list()))
        }
        ##
        ## ---- a value of a var read at indices that read no rows, x[i], x[i:k] (stanc: rvalue(x, "x", index_...)): shared ---------
        ##
        ## (a scalar when x is an Eigen vector parameter read at one index, else a vector)
        is_read_of_a_var <-  (node$type == "call") && (node$name == "stan::model::rvalue") &&
                             (length(node$args) >= 3) && (node$args[[1]]$type == "name") &&
                             !is.null(parameters_and_locals[[node$args[[1]]$value]]) &&
                             all(vapply(X = node$args[-(1:2)], FUN = function(index) {
                                     (index$type == "call") &&
                                             (index$name %in% paste0("stan::model::index_",
                                                                     c("uni", "min_max", "min", "max", "omni")))
                             }, FUN.VALUE = logical(1)))
        if (is_read_of_a_var) {
                index_text <-  paste(vapply(X = node$args[-(1:2)], FUN = function(index) {
                        paste(vapply(X = index$args, FUN = function(argument) fn_text_of_cpp_tree(node = argument),
                                     FUN.VALUE = character(1)), collapse = " ")
                }, FUN.VALUE = character(1)), collapse = " ")
                fn_check_that_an_index_reads_no_rows(index_text = index_text, data = data,
                                                     parameters_and_locals = parameters_and_locals,
                                                     name = node$args[[1]]$value)
                is_scalar <-  isTRUE(attr(parameters_and_locals, "is_Eigen_vector_parameter")[node$args[[1]]$value]) &&
                              (length(node$args) == 3) && (node$args[[3]]$name == "stan::model::index_uni")
                name_of_the_index <-  gsub(pattern = "[^A-Za-z0-9]+", replacement = "_", x = trimws(index_text))
                lambda_parameter <-  paste0(node$args[[1]]$value, "_at_", name_of_the_index, "__")
                shape <-  if (is_scalar) "scalar" else "vector"
                return(result(shape, NA, TRUE, shape, lambda_parameter,
                              setNames(fn_text_of_cpp_tree(node = node), lambda_parameter), list()))
        }
        ##
        ## ---- a one-column matrix of a vector of a row set, to_matrix(E, R, 1): the rows of a matrix -----------------------------
        ##
        ## (E of the rows as it is; any other E made once, as stanc makes it, and sliced)
        if ((node$type == "call") && (node$name == "stan::math::to_matrix") && (length(node$args) == 3) &&
            (fn_text_of_cpp_tree(node = node$args[[2]]) %in% row_sets_of_the_data) &&
            (node$args[[3]]$type == "number") && (node$args[[3]]$value == "1")) {
                row_set <-  fn_text_of_cpp_tree(node = node$args[[2]])
                inner <-  tryCatch(fn_row_wise_kind_of_node(node = node$args[[1]], data = data,
                                                            parameters_and_locals = parameters_and_locals,
                                                            body_text = body_text, struct_of_arrays = FALSE,
                                                            start_of_the_loop = start_of_the_loop),
                                   error = function(e) NULL)
                if (!is.null(inner) && (inner$kind == "rows")) {
                        aliases <-  attr(data, "scalar_dimension_aliases")
                        inner_dimension <-  inner$row_set
                        seen <-  character(0)
                        while (inner_dimension %in% names(aliases) && !(inner_dimension %in% seen)) {

                                seen <-  c(seen, inner_dimension)
                                inner_dimension <-  aliases[[inner_dimension]]

                        }
                        if (identical(inner_dimension, row_set)) inner$row_set <-  row_set
                        if ((inner$row_set != row_set) || (inner$shape != "vector")) {
                                stop("to_matrix of rows of another row set")
                        }
                        return(result("rows", row_set, inner$is_var, "matrix",
                                      paste0("stan::math::to_matrix(", inner$text, ", n_rows__, 1)"),
                                      inner$names_of_vars, inner$moved))
                }
                text_of_E <-  fn_text_of_cpp_tree(node = node$args[[1]])
                lambda_parameter <-  paste0("one_column_of_", gsub(pattern = "[^A-Za-z0-9]+", replacement = "_",
                                                                   x = substr(text_of_E, 1, 24)), "_rows__")
                return(result("rows", row_set, TRUE, "matrix",
                              paste0("stan::math::to_matrix(", lambda_parameter, ", n_rows__, 1)"),
                              setNames(paste0("nico::math::auto_chunking::sliced_at_rows_as_vars_of(",
                                              fn_text_of_cpp_tree(node = node$args[[1]]), ")"), lambda_parameter),
                              list()))
        }
        ##
        ## ---- NicoStan's nuisance block at an index of the rows (nicostan_nuisance_vector_at(group, offset, type, pstream__)) -------
        ##
        ## (the whole block read once, by NicoStan's own reader, as a vector of vars on the main tape; each tile gathers
        ##  its own rows of it, as a parameter vector at an index of the rows)
        if ((node$type == "call") && (node$name == "nicostan_nuisance_vector_at") && (length(node$args) == 4) &&
            (node$args[[1]]$type == "name") && (node$args[[2]]$type == "name") &&
            grepl(pattern = "^nicostan_offset_of_", x = node$args[[2]]$value)) {
                index_name <-  node$args[[1]]$value
                index_entry <-  data[[index_name]]
                if (is.null(index_entry) || (index_entry$shape != "integers")) {
                        stop("the index ", index_name, " is not an integer array of data")
                }
                offset_name <-  node$args[[2]]$value
                terms_name <-  sub(pattern = "^nicostan_offset_of_", replacement = "nicostan_terms_of_", x = offset_name)
                if (!grepl(pattern = paste0("(?<![A-Za-z0-9_])", terms_name, "(?![A-Za-z0-9_])"), x = body_text,
                           perl = TRUE)) {
                        stop("no ", terms_name, " for the nuisance block of ", offset_name)
                }
                block_name <-  sub(pattern = "^nicostan_offset_of_", replacement = "", x = offset_name)
                lambda_parameter <-  paste0(block_name, "_at_", index_name, "__")
                whole_block <-  paste0("nicostan_nuisance_vector(1, ", terms_name, ", ", offset_name, ", ",
                                       fn_text_of_cpp_tree(node = node$args[[3]]), ", ",
                                       fn_text_of_cpp_tree(node = node$args[[4]]), ")")
                gather <-  if (struct_of_arrays) "gathered_at_rows_of(" else "gathered_at_rows_as_vars_of("
                return(result("rows", index_entry$first_dimension, TRUE, "vector", lambda_parameter,
                              setNames(paste0("nico::math::auto_chunking::", gather, whole_block, ", ", index_name,
                                              ")"), lambda_parameter),
                              list()))
        }
        ##
        ## ---- a parameter vector at an index of the rows, z[group] (stanc: rvalue(z, "z", index_multi(group))) ----------------------
        ##
        if ((node$type == "call") && (node$name == "stan::model::rvalue") && (length(node$args) == 3) &&
            (node$args[[1]]$type == "name") && (node$args[[3]]$type == "call") &&
            (node$args[[3]]$name == "stan::model::index_multi") && (length(node$args[[3]]$args) == 1) &&
            (node$args[[3]]$args[[1]]$type == "name")) {
                vector_name <-  node$args[[1]]$value
                index_name <-  node$args[[3]]$args[[1]]$value
                index_entry <-  data[[index_name]]
                if (is.null(index_entry) || (index_entry$shape != "integers")) {
                        stop("the index ", index_name, " is not an integer array of data")
                }
                ## (a parameter vector, or a vector local whose size is not a row set, e.g. the K group effects)
                kind_of_vector <-  parameters_and_locals[[vector_name]]
                is_shared_vector <-  identical(kind_of_vector, "var_vector") ||
                                     (is.list(kind_of_vector) && !(kind_of_vector$size %in% row_sets_of_the_data))
                if (!is_shared_vector) stop("the indexed vector ", vector_name, " is not a shared var vector")
                lambda_parameter <-  paste0(vector_name, "_at_", index_name, "__")
                gather <-  if (struct_of_arrays) "gathered_at_rows_of(" else "gathered_at_rows_as_vars_of("
                return(result("rows", index_entry$first_dimension, TRUE, "vector", lambda_parameter,
                              setNames(paste0("nico::math::auto_chunking::", gather, vector_name, ", ",
                                              index_name, ")"), lambda_parameter),
                              list()))
        }
        ##
        ## ---- a column of a parameter matrix at an index of the rows, M[group, c] ---------------------------------------------------
        ##      (stanc: rvalue(M, "M", index_multi(group), index_uni(c)))
        ##
        if ((node$type == "call") && (node$name == "stan::model::rvalue") && (length(node$args) == 4) &&
            (node$args[[1]]$type == "name") && (node$args[[3]]$type == "call") &&
            (node$args[[3]]$name == "stan::model::index_multi") && (node$args[[3]]$args[[1]]$type == "name") &&
            (node$args[[4]]$type == "call") && (node$args[[4]]$name == "stan::model::index_uni") &&
            (node$args[[4]]$args[[1]]$type == "number")) {
                matrix_name <-  node$args[[1]]$value
                index_name <-  node$args[[3]]$args[[1]]$value
                column <-  as.integer(node$args[[4]]$args[[1]]$value)
                index_entry <-  data[[index_name]]
                if (is.null(index_entry) || (index_entry$shape != "integers")) {
                        stop("the index ", index_name, " is not an integer array of data")
                }
                if (!identical(parameters_and_locals[[matrix_name]], "var_matrix")) {
                        stop("the indexed matrix ", matrix_name, " is not a var matrix")
                }
                lambda_parameter <-  paste0(matrix_name, "_column_", column, "_at_", index_name, "__")
                gather <-  if (struct_of_arrays) "gathered_at_rows_of(" else "gathered_at_rows_as_vars_of("
                return(result("rows", index_entry$first_dimension, TRUE, "vector", lambda_parameter,
                              setNames(paste0("nico::math::auto_chunking::", gather, matrix_name,
                                              ".col(", column - 1, "), ", index_name, ")"), lambda_parameter),
                              list()))
        }
        ##
        ## ---- a constant vector of the rows, rep_vector(c, R) with R a row set ------------------------------------------------------
        ##
        if ((node$type == "call") && (node$name == "stan::math::rep_vector") && (length(node$args) == 2) &&
            (node$args[[2]]$type == "name") && (node$args[[2]]$value %in% row_sets_of_the_data) &&
            (node$args[[1]]$type == "number")) {
                return(result("rows", node$args[[2]]$value, FALSE, "vector",
                              paste0("stan::math::rep_vector(", node$args[[1]]$value, ", n_rows__)"),
                              character(0), list()))
        }
        ##
        ## ---- brackets, unary minus and operators: scalars only ---------------------------------------------------------------------
        ##
        parts <-  lapply(X = node$args, FUN = function(argument) {
                fn_row_wise_kind_of_node(node = argument, data = data,
                                         parameters_and_locals = parameters_and_locals, body_text = body_text,
                                         struct_of_arrays = struct_of_arrays,
                                         start_of_the_loop = start_of_the_loop)
        })
        names_of_vars <-  unlist(lapply(X = parts, FUN = function(part) part$names_of_vars))
        names_of_vars <-  names_of_vars[!duplicated(names(names_of_vars))]
        moved <-  do.call(c, c(list(list()), lapply(X = parts, FUN = function(part) part$moved)))
        moved <-  moved[!duplicated(names(moved))]
        is_var <-  any(vapply(X = parts, FUN = function(part) isTRUE(part$is_var), FUN.VALUE = logical(1)))
        kinds <-  vapply(X = parts, FUN = function(part) part$kind, FUN.VALUE = character(1))
        texts <-  vapply(X = parts, FUN = function(part) part$text, FUN.VALUE = character(1))
        if (node$type %in% c("brackets", "minus", "operator")) {
                if (any(kinds != "scalar")) stop("an operator on a vector")
                text <-  if (node$type == "brackets") paste0("(", texts[[1]], ")")
                         else if (node$type == "minus") paste0("-(", texts[[1]], ")")
                         else paste0("(", texts[[1]], " ", node$operator, " ", texts[[2]], ")")
                return(result("scalar", NA, is_var, "scalar", text, names_of_vars, moved))
        }
        ##
        ## ---- calls -----------------------------------------------------------------------------------------------------------------
        ##
        name <-  node$name
        rows <-  which(kinds == "rows")
        if (length(rows) == 0) {
                ## (a call with no rows: kept as it is, a scalar or a vector)
                if (grepl(pattern = "_lp[dm]f|_lup[dm]f|_rng", x = name)) {
                        stop("a density or rng call inside the sum")
                }
                shape <-  if (all(kinds %in% c("scalar", "stream"))) "scalar" else "vector"
                return(result(shape, NA, is_var, shape, paste0(name, "(", paste(texts, collapse = ", "), ")"),
                              names_of_vars, moved))
        }
        row_sets <-  unique(vapply(X = parts[rows], FUN = function(part) part$row_set, FUN.VALUE = character(1)))
        if (length(row_sets) != 1) stop("rows of two row sets in one call: ", name)
        ##
        ## ---- the product of an N-row data matrix with a parameter vector -----------------------------------------------------------
        ##
        if ((name == "stan::math::multiply") && (length(parts) == 2) && (kinds[[1]] == "rows") &&
            (parts[[1]]$shape == "matrix")) {
                if (kinds[[2]] != "vector") stop("a product of the rows with something other than a vector")
                second <-  if (parts[[2]]$is_var && struct_of_arrays) {
                                   paste0("nico::math::auto_chunking::as_struct_of_arrays(", texts[[2]], ")")
                           } else {
                                   texts[[2]]
                           }
                return(result("rows", row_sets, is_var, "vector",
                              paste0(name, "(", texts[[1]], ", ", second, ")"), names_of_vars, moved))
        }
        ##
        ## ---- the columns of the rows side by side, append_col(A, B), of data only --------------------------------------------------
        ##
        if (name == "stan::math::append_col") {
                if ((length(parts) != 2) || any(kinds != "rows") || is_var) {
                        stop("append_col of other than data rows")
                }
                return(result("rows", row_sets, FALSE, "matrix",
                              paste0(name, "(", texts[[1]], ", ", texts[[2]], ")"), names_of_vars, moved))
        }
        ##
        ## ---- elementwise functions of the rows and of scalars ----------------------------------------------------------------------
        ##
        is_elementwise <-  (name %in% elementwise_functions) || (name %in% elementwise_nico_functions)
        if (!is_elementwise) stop("the call ", name, " of the rows is not elementwise")
        ## (a product or quotient is elementwise only with one row-wise argument, the others scalars)
        if ((name %in% c("stan::math::multiply", "stan::math::divide")) && (length(rows) != 1)) {
                stop("the rows of two arguments in ", name)
        }
        if (any(kinds[-rows] %in% c("vector"))) {
                stop("a vector that is not row-wise in the elementwise call ", name)
        }
        if (any(parts[[rows[[1]]]]$shape == "matrix")) stop("the rows of a matrix in the elementwise call ", name)
        is_var_of <-  function(part) isTRUE(part$is_var)
        rows_are_var <-  any(vapply(X = parts[rows], FUN = is_var_of, FUN.VALUE = logical(1)))
        if (is_var && !rows_are_var && struct_of_arrays) {
                ## (rows of data meeting scalar parameters only: read as struct-of-arrays, so the result is too)
                names_of_scalar_vars <-  unique(names(unlist(lapply(X = parts[-rows],
                                                                    FUN = function(part) part$names_of_vars))))
                for (i in rows) {
                        texts[[i]] <-  paste0("nico::math::auto_chunking::data_as_struct_of_arrays_if_var(",
                                              texts[[i]], ", ", paste(names_of_scalar_vars, collapse = ", "), ")")
                }
        }
        return(result("rows", row_sets, is_var, "vector",
                      paste0(name, "(", paste(texts, collapse = ", "), ")"), names_of_vars, moved))

}
##
## ---- A density of the rows: its value is the sum of its values over any split of the rows ------------------------------------------
##
## The univariate densities (each row's term added up) and the GLM densities (the rows of y and x, the
## coefficient vector shared) of Stan Math and NicoSIMD; any other density, or another argument that is not
## row-wise or a scalar (a coefficient vector anywhere but its slot, e.g. effects by group passed whole), leaves
## it as stanc wrote it.
fn_contracts_of_row_sum_densities <-  local({

        previous <-  NULL
        function(header_paths) {

                header_paths <-  sort(normalizePath(header_paths, mustWork = TRUE))
                hashes <-  tools::md5sum(header_paths)
                if (identical(hashes, previous$hashes)) return(previous$contracts)
                contracts <-  list()
                for (path in header_paths) {

                        lines <-  readLines(path, warn = FALSE)
                        starts <-  grep("^// AUTO_ROW_SUM [A-Za-z_][A-Za-z0-9_]*$", lines)
                        if (length(starts) == 0) next
                        declaration_text <-  paste(sub("^\\s*//\\s?", "", lines, perl = TRUE), collapse = " ")
                        declarations <-  regmatches(declaration_text,
                                gregexpr("real\\s+[A-Za-z_][A-Za-z0-9_]*\\s*\\([^;{}]*\\)\\s*;",
                                         declaration_text, perl = TRUE))[[1]]
                        for (start in starts) {

                                name <-  sub("^// AUTO_ROW_SUM ", "", lines[[start]])
                                if (start >= length(lines) ||
                                    !startsWith(lines[[start + 1L]], "// AUTO_ROW_SUM_ARGUMENTS ")) {

                                        stop("missing row-sum argument contract in ", path)

                                }
                                tokens <-  strsplit(sub("^// AUTO_ROW_SUM_ARGUMENTS ", "",
                                                        lines[[start + 1L]]), " +")[[1]]
                                pairs <-  strsplit(tokens, ":", fixed = TRUE)
                                if (any(lengths(pairs) != 2L)) stop("invalid row-sum contract in ", path)
                                arguments <-  vapply(pairs, `[[`, character(1), 1L)
                                roles <-  setNames(vapply(pairs, `[[`, character(1), 2L), arguments)
                                allowed <-  c("rows", "data_matrix_rows", "group_index",
                                               "group_vector", "group_matrix")
                                if (anyDuplicated(arguments) || any(!roles %in% allowed) ||
                                    sum(roles == "group_index") != 1L ||
                                    !any(roles %in% c("group_vector", "group_matrix"))) {

                                        stop("invalid grouped row-sum roles in ", path)

                                }
                                matching <-  grep(paste0("^real\\s+", name, "\\s*\\("),
                                                  declarations, value = TRUE, perl = TRUE)
                                formal_names <-  lapply(matching, function(declaration) {

                                        body <-  sub("^[^(]*\\(", "", declaration)
                                        body <-  sub("\\)\\s*;$", "", body)
                                        trimws(sub("^.*[[:space:]]", "",
                                                   trimws(strsplit(body, ",", fixed = TRUE)[[1]])))

                                })
                                if (!any(vapply(formal_names, identical, logical(1), arguments))) {

                                        stop("row-sum contract has no matching declaration in ", path)

                                }
                                key <-  paste(name, length(arguments), sep = "/")
                                if (!is.null(contracts[[key]])) stop("duplicate row-sum contract: ", key)
                                contracts[[key]] <-  list(name = name, roles = roles,
                                                         header = path, md5 = unname(hashes[[path]]))

                        }

                }
                previous <<-  list(hashes = hashes, contracts = contracts)
                return(contracts)

        }

})
##
## ---- A declared grouped row sum: gather only the rows read by the chunk -----------------------------------------------------------
##
fn_grouped_density_of_rows <-  function(node,
                                         contract,
                                         data,
                                         parameters_and_locals,
                                         body_text,
                                         start_of_the_loop) {

        roles <-  unname(contract$roles)
        arguments <-  node$args
        has_stream <-  length(arguments) == length(roles) + 1L
        if (has_stream) {

                stream <-  arguments[[length(arguments)]]
                if (stream$type != "name" || stream$value != "pstream__") stop("invalid density stream")
                arguments <-  arguments[-length(arguments)]

        }
        if (length(arguments) != length(roles)) stop("density arity differs from its header contract")
        index_slot <-  which(roles == "group_index")
        index <-  arguments[[index_slot]]
        if (index$type != "name" || is.null(data[[index$value]]) ||
            data[[index$value]]$shape != "integers") stop("group index must be an integer data array")
        index_name <-  index$value
        row_set <-  data[[index_name]]$first_dimension
        shared <-  character(0)
        texts <-  character(length(roles))
        moved <-  list()
        group_names <-  character(0)
        for (i in seq_along(roles)) {

                argument <-  arguments[[i]]
                role <-  roles[[i]]
                if (role == "group_index") {

                        texts[[i]] <-  ""
                        next

                }
                if (role %in% c("group_vector", "group_matrix")) {

                        if (argument$type != "name") stop("group operands must be named vectors or matrices")
                        name <-  argument$value
                        kind <-  parameters_and_locals[[name]]
                        is_matrix <-  identical(kind, "var_matrix") ||
                                      identical(data[[name]]$shape, "matrix")
                        is_vector <-  identical(kind, "var_vector") ||
                                      identical(data[[name]]$shape, "vector") ||
                                      (is.list(kind) && identical(kind$kind, "var_vector_local"))
                        if ((role == "group_matrix" && !is_matrix) ||
                            (role == "group_vector" && !is_vector)) stop("group operand shape: ", name)
                        parameter <-  paste0(name, "_at_", index_name, "__")
                        texts[[i]] <-  paste0("nico::math::auto_chunking::values_of_unique_groups(", parameter, ")")
                        shared[[parameter]] <-  paste0("nico::math::auto_chunking::",
                                "gathered_at_unique_groups_of(", name, ", ", index_name, ")")
                        group_names <-  c(group_names, name)
                        next

                }
                is_array <-  argument$type == "name" && !is.null(data[[argument$value]]) &&
                             data[[argument$value]]$shape %in% c("integers", "reals")
                if (is_array) {

                        part <-  list(kind = "rows", row_set = data[[argument$value]]$first_dimension,
                                      is_var = FALSE, shape = "vector", names_of_vars = character(0),
                                      moved = list(), text = paste0("nico::math::auto_chunking::array_rows_of(",
                                              argument$value, ", first__, n_rows__)"))

                } else {

                        matrix_from_vector <-  argument$type == "call" &&
                                               argument$name == "stan::math::to_matrix" &&
                                               length(argument$args) == 1L
                        part <-  fn_row_wise_kind_of_node(
                                node = if (matrix_from_vector) argument$args[[1]] else argument,
                                data = data,
                                parameters_and_locals = parameters_and_locals,
                                body_text = body_text,
                                struct_of_arrays = FALSE,
                                start_of_the_loop = start_of_the_loop)
                        if (matrix_from_vector) {

                                part$text <-  paste0("stan::math::to_matrix(", part$text, ")")
                                part$shape <-  "matrix"

                        }

                }
                if (part$kind != "rows" || part$row_set != row_set) stop("density row sets differ")
                if (role == "data_matrix_rows" && (part$is_var || part$shape != "matrix")) {

                        stop("density design matrix must be rows of data")

                }
                texts[[i]] <-  part$text
                shared <-  c(shared, part$names_of_vars)
                moved <-  c(moved, part$moved)

        }
        if (length(group_names) > 1L) {

                i <-  which(roles %in% c("group_vector", "group_matrix"))[[length(group_names)]]
                parameter <-  paste0(arguments[[i]]$value, "_at_", index_name, "__")
                checked <-  paste0("nico::math::auto_chunking::with_matching_group_rows(",
                                    paste(rev(group_names), collapse = ", "), ")")
                shared[[parameter]] <-  paste0("nico::math::auto_chunking::gathered_at_unique_group_values_of(",
                                               checked, ", ", index_name, ")")

        }
        first_group <-  paste0(group_names[[1]], "_at_", index_name, "__")
        texts[[index_slot]] <-  paste0("nico::math::auto_chunking::indices_of_unique_groups(",
                                       first_group, ")")
        if (has_stream) texts <-  c(texts, "pstream__")
        shared <-  shared[!duplicated(names(shared))]
        is_var <-  any(vapply(group_names, function(name) !is.null(parameters_and_locals[[name]]), logical(1)))
        return(list(kind = "rows", row_set = row_set, is_var = is_var, shape = "scalar",
                    text = paste0(node$name, "(", paste(texts, collapse = ", "), ")"),
                    names_of_vars = shared, moved = moved))

}
##
## ---- The row-sum contracts whose term of a row also reads the row before it (AUTO_ROW_SUM_WITH_THE_ROW_BEFORE) ---------------------
##
## (each: the function's name, the role of each argument (rows, rows_with_the_row_before, group_start or scalar),
##  and the value of each rows and group_start argument at a row whose own term is zero)
fn_contracts_of_row_sums_with_the_row_before <-  function(header_paths) {

        contracts <-  list()
        allowed <-  c("rows", "rows_with_the_row_before", "group_start", "scalar")
        for (path in sort(normalizePath(header_paths, mustWork = TRUE))) {

                lines <-  readLines(path, warn = FALSE)
                starts <-  grep("^// AUTO_ROW_SUM_WITH_THE_ROW_BEFORE [A-Za-z_][A-Za-z0-9_]*$", lines)
                if (length(starts) == 0) next
                declaration_text <-  paste(sub("^\\s*//\\s?", "", lines, perl = TRUE), collapse = " ")
                declarations <-  regmatches(declaration_text,
                                            gregexpr("real\\s+[A-Za-z_][A-Za-z0-9_]*\\s*\\([^;{}]*\\)\\s*;",
                                                     declaration_text, perl = TRUE))[[1]]
                ##
                ## ---- Each contract: its name, its roles and the values of a row whose own term is zero -----------------------------
                ##
                for (start in starts) {

                        name <-  sub("^// AUTO_ROW_SUM_WITH_THE_ROW_BEFORE ", "", lines[[start]])
                        if ((start + 2 > length(lines)) ||
                            !startsWith(lines[[start + 1]], "// AUTO_ROW_SUM_WITH_THE_ROW_BEFORE_ARGUMENTS ") ||
                            !startsWith(lines[[start + 2]], "// AUTO_ROW_SUM_WITH_THE_ROW_BEFORE_ZERO_TERM ")) {

                                stop("an incomplete row-sum contract with the row before in ", path)

                        }
                        text_of_the_roles <-  sub("^// AUTO_ROW_SUM_WITH_THE_ROW_BEFORE_ARGUMENTS ", "",
                                                  lines[[start + 1]])
                        text_of_the_zero_term <-  sub("^// AUTO_ROW_SUM_WITH_THE_ROW_BEFORE_ZERO_TERM ", "",
                                                      lines[[start + 2]])
                        pairs <-  strsplit(strsplit(text_of_the_roles, " +")[[1]], ":", fixed = TRUE)
                        pairs_of_the_zero_term <-  strsplit(strsplit(text_of_the_zero_term, " +")[[1]], ":",
                                                            fixed = TRUE)
                        if (any(lengths(pairs) != 2) || any(lengths(pairs_of_the_zero_term) != 2)) {

                                stop("an invalid row-sum contract with the row before in ", path)

                        }
                        roles <-  setNames(vapply(pairs, `[[`, character(1), 2),
                                           vapply(pairs, `[[`, character(1), 1))
                        values_at_a_zero_term <-  setNames(vapply(pairs_of_the_zero_term, `[[`, character(1), 2),
                                                           vapply(pairs_of_the_zero_term, `[[`, character(1), 1))
                        ## (every rows and group_start argument has its value at a row whose term is zero; the
                        ##  group start is 1 there, so that row reads no row before it)
                        names_with_a_value <-  names(roles)[roles %in% c("rows", "group_start")]
                        if (anyDuplicated(names(roles)) || any(!(roles %in% allowed)) ||
                            (sum(roles == "rows_with_the_row_before") < 1) ||
                            (sum(roles == "group_start") != 1) ||
                            !setequal(names(values_at_a_zero_term), names_with_a_value) ||
                            any(!grepl("^-?[0-9]+([.][0-9]+)?$", values_at_a_zero_term)) ||
                            (values_at_a_zero_term[[names(roles)[roles == "group_start"]]] != "1")) {

                                stop("invalid roles of a row-sum contract with the row before in ", path)

                        }
                        matching <-  grep(paste0("^real\\s+", name, "\\s*\\("), declarations, value = TRUE,
                                          perl = TRUE)
                        formal_names <-  lapply(matching, function(declaration) {

                                body <-  sub("\\)\\s*;$", "", sub("^[^(]*\\(", "", declaration))
                                trimws(sub("^.*[[:space:]]", "", trimws(strsplit(body, ",", fixed = TRUE)[[1]])))

                        })
                        if (!any(vapply(formal_names, identical, logical(1), names(roles)))) {

                                stop("a row-sum contract with the row before has no matching declaration in ",
                                     path)

                        }
                        key <-  paste(name, length(roles), sep = "/")
                        if (!is.null(contracts[[key]])) {
                                stop("duplicate row-sum contract with the row before: ", key)
                        }
                        contracts[[key]] <-  list(name = name, roles = roles,
                                                 values_at_a_zero_term = values_at_a_zero_term, header = path)

                }

        }
        return(contracts)

}
##
## ---- A declared row sum with the row before: the chunk's rows, and the row before them as a row whose term is zero -----------------
##
## (rows and group_start: arrays or vectors of the data, read by name, the row before set to its contract's value;
##  rows_with_the_row_before: any vector of the rows, read at the chunk's rows and at the row before them; scalar:
##  shared; the group start is given to the runtime, which gives a chunk the row before it only when its first
##  row is not row 1 and does not start a group)
fn_density_of_rows_with_the_row_before <-  function(node,
                                                    contract,
                                                    data,
                                                    parameters_and_locals,
                                                    body_text,
                                                    start_of_the_loop) {

        roles <-  unname(contract$roles)
        names_of_the_arguments <-  names(contract$roles)
        arguments <-  node$args
        has_stream <-  (length(arguments) == length(roles) + 1) &&
                       (arguments[[length(arguments)]]$type == "name") &&
                       (arguments[[length(arguments)]]$value == "pstream__")
        if (has_stream) arguments <-  arguments[-length(arguments)]
        if (length(arguments) != length(roles)) stop("the call of ", node$name, " differs from its contract")
        row_set <-  NA_character_
        group_start <-  NA_character_
        texts <-  character(length(roles))
        shared <-  character(0)
        moved <-  list()
        is_var <-  FALSE
        ##
        ## ---- Each argument, by its role --------------------------------------------------------------------------------------------
        ##
        for (i in seq_along(roles)) {

                argument <-  arguments[[i]]
                role <-  roles[[i]]
                name_of_the_argument <-  names_of_the_arguments[[i]]
                if (role %in% c("rows", "group_start")) {

                        entry <-  if (argument$type == "name") data[[argument$value]] else NULL
                        if (is.null(entry) || !(entry$shape %in% c("integers", "reals", "vector"))) {
                                stop("the ", role, " argument ", name_of_the_argument, " of ", node$name,
                                     " is not an array or vector of the data")
                        }
                        helper <-  if (entry$shape == "vector") "rows_with_the_row_before_set_to("
                                   else "array_rows_with_the_row_before_set_to("
                        texts[[i]] <-  paste0("nico::math::auto_chunking::", helper, argument$value,
                                              ", first__, n_rows__, row_before__, ",
                                              contract$values_at_a_zero_term[[name_of_the_argument]], ")")
                        row_set_of_the_argument <-  entry$first_dimension
                        if (role == "group_start") group_start <-  argument$value

                } else {

                        part <-  fn_row_wise_kind_of_node(node = argument, data = data,
                                                          parameters_and_locals = parameters_and_locals,
                                                          body_text = body_text, struct_of_arrays = FALSE,
                                                          start_of_the_loop = start_of_the_loop)
                        is_a_vector_of_the_rows <-  (part$kind == "rows") && (part$shape == "vector")
                        if ((role == "rows_with_the_row_before") && !is_a_vector_of_the_rows) {
                                stop("the argument ", name_of_the_argument, " of ", node$name,
                                     " is not a vector of the rows")
                        }
                        if ((role == "scalar") && (part$kind != "scalar")) {
                                stop("the argument ", name_of_the_argument, " of ", node$name, " is not a scalar")
                        }
                        texts[[i]] <-  part$text
                        shared <-  c(shared, part$names_of_vars)
                        moved <-  c(moved, part$moved)
                        is_var <-  is_var || isTRUE(part$is_var)
                        if (role == "scalar") next
                        row_set_of_the_argument <-  part$row_set

                }
                if (!is.na(row_set) && (row_set_of_the_argument != row_set)) {
                        stop("rows of two row sets in ", node$name)
                }
                row_set <-  row_set_of_the_argument

        }
        if (has_stream) texts <-  c(texts, "pstream__")
        shared <-  shared[!duplicated(names(shared))]
        moved <-  moved[!duplicated(names(moved))]
        return(list(kind = "rows", row_set = row_set, is_var = is_var, shape = "scalar",
                    text = paste0(node$name, "(", paste(texts, collapse = ", "), ")"),
                    names_of_vars = shared, moved = moved, group_start = group_start))

}
##
## ---- A density of the rows with an explicit declaration contract ------------------------------------------------------------------
##
fn_row_wise_kind_of_density <-  function(node,
                                         data,
                                         parameters_and_locals,
                                         body_text,
                                         start_of_the_loop) {

        univariate_densities <-  c(paste0("stan::math::",
                                          c("normal", "student_t", "cauchy", "logistic", "double_exponential",
                                            "exponential", "gamma", "inv_gamma", "weibull", "lognormal", "beta",
                                            "bernoulli", "bernoulli_logit", "binomial", "binomial_logit",
                                            "poisson", "poisson_log", "neg_binomial_2", "neg_binomial_2_log",
                                            "std_normal")),
                                   "nico_normal", "nico_exponential", "nico_lognormal")
        ## (the GLM densities: the slot of x, the N-row data matrix, and of the coefficient vector)
        GLM_densities <-  c("stan::math::ordered_logistic_glm", "nico_ordered_logistic_glm",
                            "stan::math::bernoulli_logit_glm", "stan::math::poisson_log_glm",
                            "stan::math::normal_id_glm", "stan::math::neg_binomial_2_log_glm",
                            "nico_student_t_id_glm", "nico_weibull_log_glm")
        slot_of_x <-  2
        slot_of_coefficients <-  4
        function_name <-  sub(pattern = "<.*$", replacement = "", x = node$name)
        family <-  sub(pattern = "_(lpdf|lpmf|lupdf|lupmf|lccdf|lcdf)$", replacement = "", x = function_name)
        ## (ordered_logistic_glm(y | x, beta, cuts): the coefficients and the cutpoints shared)
        slots_of_shared_vectors <-  slot_of_coefficients
        if (family %in% c("stan::math::ordered_logistic_glm", "nico_ordered_logistic_glm")) {
                slots_of_shared_vectors <-  c(3, 4)
        }
        is_GLM <-  family %in% GLM_densities
        is_without_a_contract <-  FALSE
        if (!(family %in% univariate_densities) && !is_GLM) {

                headers <-  getOption("NicoStan.auto_chunking_density_contract_headers",
                        list.files(system.file("include/NicoSIMD", package = "NicoSIMD"),
                                   pattern = "[.]hpp$", full.names = TRUE))
                arity <-  length(node$args)
                if (arity > 0L && node$args[[arity]]$type == "name" &&
                    node$args[[arity]]$value == "pstream__") arity <-  arity - 1L
                ## (a density whose term of a row also reads the row before it, by its contract)
                key <-  paste(function_name, arity, sep = "/")
                contract_with_the_row_before <-  fn_contracts_of_row_sums_with_the_row_before(
                                                         header_paths = headers)[[key]]
                if (!is.null(contract_with_the_row_before)) {

                        return(fn_density_of_rows_with_the_row_before(
                                       node = node,
                                       contract = contract_with_the_row_before,
                                       data = data,
                                       parameters_and_locals = parameters_and_locals,
                                       body_text = body_text,
                                       start_of_the_loop = start_of_the_loop))

                }
                contracts <-  fn_contracts_of_row_sum_densities(header_paths = headers)
                contract <-  contracts[[paste(function_name, arity, sep = "/")]]
                if (!is.null(contract)) {

                        return(fn_grouped_density_of_rows(node = node,
                                                          contract = contract,
                                                          data = data,
                                                          parameters_and_locals = parameters_and_locals,
                                                          body_text = body_text,
                                                          start_of_the_loop = start_of_the_loop))

                }
                ## (a density with no declared contract: summed over its rows as a univariate density is, when
                ##  every argument is its rows or a scalar; its first call checks the chunks' sum against stanc's
                ##  own statement)
                if (!grepl("_(lpdf|lpmf|lupdf|lupmf|lcdf|lccdf)$", function_name)) {

                        stop("no row-sum contract for ", function_name)

                }
                is_without_a_contract <-  TRUE

        }
        ##
        ## ---- each argument: rows, a scalar, the stream, or (a GLM's coefficients) a shared vector ----------------------------------
        ##
        parts <-  list()
        for (i in seq_along(node$args)) {

                argument <-  node$args[[i]]
                is_array <-  (argument$type == "name") && !is.null(data[[argument$value]]) &&
                             (data[[argument$value]]$shape %in% c("integers", "reals"))
                if (is_array) {
                        ## (an array of the data, given to the density as an array of the tile's own values)
                        parts[[i]] <-  list(kind = "rows", row_set = data[[argument$value]]$first_dimension,
                                            is_var = FALSE, shape = "vector",
                                            text = paste0("nico::math::auto_chunking::array_rows_of(",
                                                          argument$value, ", first__, n_rows__)"),
                                            names_of_vars = character(0), moved = list())
                } else if (!is.null(slice <-  fn_slice_of_the_data(node = argument, data = data)) &&
                           (slice$shape %in% c("integers", "reals"))) {
                        ## (a slice of an array of the data, D[a:b], given as an array of the tile's own values)
                        row_set <-  fn_row_set_of_the_slice(slice = slice, data = data)
                        if (is.na(row_set)) stop("the length of the slice of ", slice$name, " is not a row set")
                        parts[[i]] <-  list(kind = "rows", row_set = row_set, is_var = FALSE, shape = "vector",
                                            text = paste0("nico::math::auto_chunking::array_rows_of_slice(",
                                                          slice$name, ", ", slice$lower, ", ", slice$upper, ", ",
                                                          row_set, ", first__, n_rows__)"),
                                            names_of_vars = character(0), moved = list())
                } else {
                        parts[[i]] <-  fn_row_wise_kind_of_node(node = argument, data = data,
                                                                parameters_and_locals = parameters_and_locals,
                                                                body_text = body_text, struct_of_arrays = FALSE,
                                                                start_of_the_loop = start_of_the_loop)
                }
                kind <-  parts[[i]]$kind
                if (is_GLM && (i == slot_of_x)) {
                        if ((kind != "rows") || (parts[[i]]$shape != "matrix")) {
                                stop("the x of ", function_name, " is not the rows of a matrix")
                        }
                } else if (is_GLM && (i %in% slots_of_shared_vectors)) {
                        if (kind == "rows") stop("the coefficients of ", function_name, " are row-wise")
                } else {
                        if (kind == "vector") stop("a vector that is not row-wise in ", function_name)
                        if ((kind == "rows") && (parts[[i]]$shape == "matrix")) {
                                stop("the rows of a matrix in ", function_name)
                        }
                }

        }
        kinds <-  vapply(X = parts, FUN = function(part) part$kind, FUN.VALUE = character(1))
        rows <-  which(kinds == "rows")
        if (length(rows) == 0) stop("no rows in ", function_name)
        row_sets <-  unique(vapply(X = parts[rows], FUN = function(part) part$row_set, FUN.VALUE = character(1)))
        if (length(row_sets) != 1) stop("rows of two row sets in ", function_name)
        names_of_vars <-  unlist(lapply(X = parts, FUN = function(part) part$names_of_vars))
        names_of_vars <-  names_of_vars[!duplicated(names(names_of_vars))]
        moved <-  do.call(c, c(list(list()), lapply(X = parts, FUN = function(part) part$moved)))
        moved <-  moved[!duplicated(names(moved))]
        is_var <-  any(vapply(X = parts, FUN = function(part) isTRUE(part$is_var), FUN.VALUE = logical(1)))
        texts <-  vapply(X = parts, FUN = function(part) part$text, FUN.VALUE = character(1))
        return(list(kind = "rows", row_set = row_sets, is_var = is_var, shape = "scalar",
                    text = paste0(node$name, "(", paste(texts, collapse = ", "), ")"),
                    names_of_vars = names_of_vars, moved = moved,
                    name_of_a_density_without_a_contract = if (is_without_a_contract) function_name))

}
##
## ---- Why the N-row locals of a candidate cannot be made inside its tiles (NULL if they can) ----------------------------------------
##
## Each must be assigned exactly once, before the candidate, in a block that holds the candidate (so that the
## assignment has always run when the candidate runs), after the N-row locals it reads, and nothing else the tiles
## read may be assigned between that assignment and the candidate (so that the tile makes the local from the same
## values as stanc's code).
fn_why_the_locals_cannot_be_made_in_the_tile <-  function(body_text,
                                                          candidate_start,
                                                          moved,
                                                          names_of_vars) {

        ## (stanc's assignments: stan::model::assign(x, ...) for containers, and a statement x = ...; for scalars)
        pattern_of_an_assignment_of <-  function(name) {

                paste0("stan::model::assign\\(\\s*", name, "\\s*,|(?:^|[;{}])\\s*", name, "\\s*=(?!=)")

        }
        names_in <-  function(texts) unique(unlist(regmatches(x = texts,
                                                              m = gregexpr(pattern = "[A-Za-z_][A-Za-z0-9_]*",
                                                                           text = texts))))
        start_of_assignment <-  list()
        for (local_name in names(moved)) {

                starts <-  as.integer(gregexpr(pattern = pattern_of_an_assignment_of(local_name),
                                               text = body_text, perl = TRUE)[[1]])
                starts <-  starts[starts > 0]
                if (length(starts) != 1) {
                        return(paste0("the local ", local_name, " is assigned ", length(starts), " times"))
                }
                if (starts > candidate_start) {
                        return(paste0("the local ", local_name, " is assigned after its use"))
                }
                start_of_assignment[[local_name]] <-  starts
                assignment <-  fn_assignment_of_local(body_text = body_text, local_name = local_name)
                between <-  substring(body_text, assignment$end + 1, candidate_start - 1)
                characters <-  strsplit(x = between, split = "")[[1]]
                depth <-  cumsum((characters == "{") - (characters == "}"))
                if (any(depth < 0)) {
                        return(paste0("the local ", local_name,
                                      " is assigned in a block that ends before its use"))
                }
                ## (the names this local's own expression reads, through the shared arguments it reads too)
                names_in_the_local <-  names_in(moved[[local_name]])
                shared_read <-  names_of_vars[intersect(names(names_of_vars), names_in_the_local)]
                names_read_by_the_local <-  setdiff(names_in(c(names_in_the_local, unname(shared_read))),
                                                    names(moved))
                for (name in names_read_by_the_local) {

                        if (grepl(pattern = pattern_of_an_assignment_of(name), x = between, perl = TRUE)) {
                                return(paste0(name, " is assigned between the assignment of ", local_name,
                                              " and its use"))
                        }

                }

        }
        ## (an N-row local read by another is assigned before it)
        for (local_name in names(moved)) {

                for (read_local in intersect(names_in(moved[[local_name]]), names(moved))) {

                        if (start_of_assignment[[read_local]] > start_of_assignment[[local_name]]) {
                                return(paste0("the local ", read_local, " is assigned after ", local_name,
                                              ", which reads it"))
                        }

                }

        }
        return(NULL)

}
##
## ---- The integer locals that are slices of the data: data, read by rows -------------------------------------------------------------
##
## (a local std::vector<int> x = D[a:b], D an integer array of the data, assigned once: an integer array of the
##  data whose rows are its size, read by the tiles through rows_of() and as an index, like D itself)
fn_slices_of_the_data_in_log_prob <-  function(body_text,
                                               data) {

        slices <-  list()
        pattern_of_a_local <-  paste0("std::vector<int>\\s+([A-Za-z_][A-Za-z0-9_]*)\\s*=\\s*",
                                      "std::vector<int>\\(\\s*([^,;]+),")
        locals <-  regmatches(x = body_text, m = gregexpr(pattern = pattern_of_a_local, text = body_text,
                                                          perl = TRUE))[[1]]
        for (local in locals) {

                matched <-  regmatches(x = local, m = regexec(pattern = pattern_of_a_local, text = local,
                                                              perl = TRUE))[[1]]
                local_name <-  matched[[2]]
                pattern_of_its_assignments <-  paste0("stan::model::assign\\(\\s*", local_name,
                                                      "\\s*,|(?:^|[;{}])\\s*", local_name, "\\s*=(?!=)")
                assignments <-  as.integer(gregexpr(pattern = pattern_of_its_assignments, text = body_text,
                                                    perl = TRUE)[[1]])
                if (sum(assignments > 0) != 1) next
                assignment <-  fn_assignment_of_local(body_text = body_text, local_name = local_name)
                if (is.null(assignment)) next
                expression <-  assignment$expression
                is_slice <-  (expression$type == "call") && (expression$name == "stan::model::rvalue") &&
                             (length(expression$args) == 3) && (expression$args[[1]]$type == "name") &&
                             !is.null(data[[expression$args[[1]]$value]]) &&
                             (data[[expression$args[[1]]$value]]$shape == "integers") &&
                             (expression$args[[3]]$type == "call") &&
                             (expression$args[[3]]$name == "stan::model::index_min_max")
                if (is_slice) {
                        slices[[local_name]] <-  list(first_dimension = trimws(matched[[3]]), shape = "integers")
                }

        }
        return(slices)

}
##
## ---- The start of the innermost for loop that holds a position of log_prob_impl (NA if none) --------------------------------------
##
fn_start_of_the_innermost_loop <-  function(body_text,
                                            position) {

        starts <-  as.integer(gregexpr(pattern = "for \\(", text = substring(body_text, 1, position))[[1]])
        starts <-  starts[starts > 0]
        for (start in rev(starts)) {

                close_bracket <-  fn_end_of_bracket(text = body_text, position_of_open_bracket = start + 4)
                after <-  substring(body_text, close_bracket + 1, nchar(body_text))
                brace <-  regexpr(pattern = "^\\s*\\{", text = after)
                if (brace < 0) next
                open_brace <-  close_bracket + attr(brace, "match.length")
                characters <-  strsplit(x = substring(body_text, open_brace, nchar(body_text)), split = "")[[1]]
                depth <-  cumsum((characters == "{") - (characters == "}"))
                close_brace <-  open_brace + match(0, depth) - 1
                if (position < close_brace) return(start)

        }
        return(NA_integer_)

}
##
## ---- The running sums: user functions whose body makes a running sum (cumulative_sum) over the rows --------------------------------
##
## A user function of the Stan model is read from stanc's own C++ of its body. It is chunked when its statements are
## only these: integer and scalar locals; N-row vector locals made from the rows elementwise; an update of the first
## row of one of them; a running sum, cumulative_sum(v), of one of them; the largest of a row argument, max(x), in a
## scalar; one test of the first row whose failure leaves the function's other return; and a return that adds over
## the rows (sums and dot products of the rows, each times a scalar at most). In a chunk, each running sum starts
## from the total of the chunks before it (its carry), the first row is updated in the first chunk only, the test
## is made in the first chunk only (if it fails, the statement is done as stanc wrote it), and max(x) is the
## largest row of all the chunks, read on the main tape at its row.
##
##
## ---- The statements of a C++ block (an if with its braces is one statement) --------------------------------------------------------
##
fn_statements_of_cpp_block <-  function(block_text) {

        characters <-  strsplit(x = block_text, split = "")[[1]]
        statements <-  character(0)
        start <-  1
        depth_of_brackets <-  0
        depth_of_braces <-  0
        in_string <-  FALSE
        for (i in seq_along(characters)) {

                character <-  characters[[i]]
                if (character == "\"") in_string <-  !in_string
                if (in_string) next
                if (character == "(") depth_of_brackets <-  depth_of_brackets + 1
                if (character == ")") depth_of_brackets <-  depth_of_brackets - 1
                if (character == "{") depth_of_braces <-  depth_of_braces + 1
                if (character == "}") {
                        depth_of_braces <-  depth_of_braces - 1
                        if ((depth_of_braces == 0) && (depth_of_brackets == 0)) {
                                statements <-  c(statements, trimws(substring(block_text, start, i)))
                                start <-  i + 1
                        }
                }
                if ((character == ";") && (depth_of_brackets == 0) && (depth_of_braces == 0)) {
                        statements <-  c(statements, trimws(substring(block_text, start, i)))
                        start <-  i + 1
                }

        }
        statements <-  statements[nzchar(statements)]
        return(statements)

}
##
## ---- The text between a bracket at a position and the bracket that closes it ---------------------------------------------------------
##
fn_inside_of_the_bracket <-  function(text,
                                      position_of_open_bracket) {

        open <-  substring(text, position_of_open_bracket, position_of_open_bracket)
        if (open == "(") {
                close_bracket <-  fn_end_of_bracket(text = text, position_of_open_bracket = position_of_open_bracket)
                return(list(inside = substring(text, position_of_open_bracket + 1, close_bracket - 1),
                            close_bracket = close_bracket))
        }
        ## (a brace: matched past the strings in it)
        characters <-  strsplit(x = substring(text, position_of_open_bracket, nchar(text)), split = "")[[1]]
        depth <-  0
        in_string <-  FALSE
        for (i in seq_along(characters)) {

                if ((characters[[i]] == "\"") && ((i == 1) || (characters[[i - 1]] != "\\"))) in_string <-  !in_string
                if (in_string) next
                if (characters[[i]] == "{") depth <-  depth + 1
                if (characters[[i]] == "}") {
                        depth <-  depth - 1
                        if (depth == 0) {
                                close_bracket <-  position_of_open_bracket + i - 1
                                return(list(inside = substring(text, position_of_open_bracket + 1, close_bracket - 1),
                                            close_bracket = close_bracket))
                        }
                }

        }
        stop("no closing brace")

}
##
## ---- The arguments of a call, split at its top-level commas ---------------------------------------------------------------------------
##
fn_arguments_of_a_call_text <-  function(inside_text) {

        characters <-  strsplit(x = inside_text, split = "")[[1]]
        arguments <-  character(0)
        start <-  1
        depth <-  0
        for (i in seq_along(characters)) {

                if (characters[[i]] %in% c("(", "<", "{")) depth <-  depth + 1
                if (characters[[i]] %in% c(")", ">", "}")) depth <-  depth - 1
                if ((characters[[i]] == ",") && (depth == 0)) {
                        arguments <-  c(arguments, trimws(substring(inside_text, start, i - 1)))
                        start <-  i + 1
                }

        }
        arguments <-  c(arguments, trimws(substring(inside_text, start, nchar(inside_text))))
        return(arguments[nzchar(arguments)])

}
##
## ---- The user functions of the model that return a real and make a running sum over their rows ---------------------------------------
##
## (from the Stan source: the names of the functions block that return a real; from stanc's C++: each one's
##  definition, its arguments in order, the names its body reads them by, and the statements of its try block)
fn_running_sum_functions_of_generated_cpp <-  function(text,
                                                       Stan_model_lines) {

        functions <-  list()
        Stan_model_lines <-  unlist(strsplit(x = Stan_model_lines, split = "\n"))
        start_of_the_block <-  grep(pattern = "^\\s*functions\\s*\\{", x = Stan_model_lines)
        if (length(start_of_the_block) == 0) return(functions)
        lines_of_definitions <-  grep(pattern = "^\\s*real\\s+[A-Za-z_][A-Za-z0-9_]*\\s*\\(", x = Stan_model_lines,
                                      value = TRUE)
        names_of_real_functions <-  unique(sub(pattern = "^\\s*real\\s+([A-Za-z_][A-Za-z0-9_]*)\\s*\\(.*$",
                                               replacement = "\\1", x = lines_of_definitions))
        for (name in names_of_real_functions) {

                starts <-  as.integer(gregexpr(pattern = paste0("\n", name, "\\("), text = text)[[1]])
                starts <-  starts[starts > 0]
                for (start in starts) {

                        open_bracket <-  start + nchar(name) + 1
                        ## (a text that is not a definition with its body, e.g. a declaration: not a running sum)
                        parameters <-  tryCatch(fn_inside_of_the_bracket(text = text,
                                                                         position_of_open_bracket = open_bracket),
                                                error = function(e) NULL)
                        if (is.null(parameters)) next
                        after <-  substring(text, parameters$close_bracket + 1, nchar(text))
                        if (!grepl(pattern = "^\\s*\\{", x = after)) next
                        open_brace <-  parameters$close_bracket + regexpr(pattern = "\\{", text = after)[[1]]
                        body <-  tryCatch(fn_inside_of_the_bracket(text = text, position_of_open_bracket = open_brace)$inside,
                                          error = function(e) NULL)
                        if (is.null(body)) next
                        if (!grepl(pattern = "stan::math::cumulative_sum(", x = body, fixed = TRUE)) next
                        try_at <-  regexpr(pattern = "try\\s*\\{", text = body)[[1]]
                        if (try_at < 0) next
                        open_of_try <-  try_at + attr(regexpr(pattern = "try\\s*\\{", text = body), "match.length") - 1
                        try_block <-  fn_inside_of_the_bracket(text = body, position_of_open_bracket = open_of_try)$inside
                        arguments <-  fn_arguments_of_a_call_text(inside_text = parameters$inside)
                        argument_names <-  sub(pattern = "^.*[^A-Za-z0-9_]([A-Za-z_][A-Za-z0-9_]*)$", replacement = "\\1",
                                               x = arguments)
                        argument_names <-  argument_names[argument_names != "pstream__"]
                        ## (the name the body reads each argument by: const auto& x = stan::math::to_ref(x_arg__);)
                        names_in_the_body <-  vapply(X = argument_names, FUN = function(argument_name) {
                                reference <-  regmatches(x = body, m = regexec(pattern = paste0(
                                        "const auto& ([A-Za-z_][A-Za-z0-9_]*) = stan::math::to_ref\\(", argument_name,
                                        "\\);"), text = body))[[1]]
                                if (length(reference) == 2) reference[[2]] else argument_name
                        }, FUN.VALUE = character(1))
                        functions[[name]] <-  list(name = name, names_in_the_body = unname(names_in_the_body),
                                                   statements = fn_statements_of_cpp_block(block_text = try_block))

                }

        }
        return(functions)

}
##
## ---- One running-sum function, made for a chunk: its body as a lambda over its arguments ----------------------------------------------
##
## (row_names: the names in its body that read rows (its row arguments, and its N-row vector locals once made);
##  first_carry: the index of its first running sum among the statement's carries; largest_of: for each row
##  argument whose max(x) it reads, the name of the shared largest row it is given)
fn_running_sum_function_for_a_chunk <-  function(running_function,
                                                  row_arguments,
                                                  first_carry,
                                                  largest_of) {

        names_of_rows <-  row_arguments
        names_of_scalars <-  setdiff(running_function$names_in_the_body, row_arguments)
        lines <-  character(0)
        n_carries <-  0
        test_of_the_first_row <-  NULL
        main_return <-  NULL
        reads_rows <-  function(text_of_code,
                                names) {

                any(vapply(X = names, FUN = function(name) {
                        grepl(pattern = paste0("(?<![A-Za-z0-9_\"])", name, "(?![A-Za-z0-9_\"])"), x = text_of_code,
                              perl = TRUE)
                }, FUN.VALUE = logical(1)))

        }
        ## (the reads of the first row of the rows, rvalue(v, "v", index_uni(1)), taken out of a text)
        without_reads_of_the_first_row <-  function(text_of_code) {

                gsub(pattern = paste0("stan::model::rvalue\\(\\s*[A-Za-z_][A-Za-z0-9_]*\\s*,\\s*\"[^\"]*\"\\s*,\\s*",
                                      "stan::model::index_uni\\(1\\)\\)"), replacement = "FIRST_ROW", x = text_of_code)

        }
        for (statement in running_function$statements) {

                one_line <-  gsub(pattern = "\\s+", replacement = " ", x = statement)
                ##
                ## ---- the statement numbers, the size checks, and the declarations of integers and scalars --------------------------
                ##
                if (grepl(pattern = "^current_statement__ = [0-9]+;$", x = one_line) ||
                    grepl(pattern = "^stan::math::validate_non_negative_index\\(", x = one_line)) {
                        lines <-  c(lines, one_line)
                        next
                }
                declared_integer <-  regmatches(x = one_line, m = regexec(pattern = paste0(
                        "^int ([A-Za-z_][A-Za-z0-9_]*) = std::numeric_limits<int>::min\\(\\);$"), text = one_line))[[1]]
                declared_scalar <-  regmatches(x = one_line, m = regexec(pattern = paste0(
                        "^local_scalar_t__ ([A-Za-z_][A-Za-z0-9_]*) = DUMMY_VAR__;$"), text = one_line))[[1]]
                if ((length(declared_integer) == 2) || (length(declared_scalar) == 2)) {
                        lines <-  c(lines, one_line)
                        names_of_scalars <-  c(names_of_scalars, c(declared_integer, declared_scalar)[[2]])
                        next
                }
                ##
                ## ---- the declaration of a vector of the rows (its size an integer local, e.g. M = rows(x)) -------------------------
                ##
                declared_vector <-  regmatches(x = one_line, m = regexec(pattern = paste0(
                        "^Eigen::Matrix<local_scalar_t__, ?-1, ?1> ([A-Za-z_][A-Za-z0-9_]*) = ",
                        "Eigen::Matrix<local_scalar_t__, ?-1, ?1>::Constant\\(([A-Za-z_][A-Za-z0-9_]*), DUMMY_VAR__\\);$"),
                        text = one_line))[[1]]
                if (length(declared_vector) == 3) {
                        lines <-  c(lines, one_line)
                        names_of_rows <-  c(names_of_rows, declared_vector[[2]])
                        next
                }
                ##
                ## ---- an assignment of an integer or scalar local: no reads of the rows but rows(x) and max(x) ------------------------
                ##
                assigned_scalar <-  regmatches(x = one_line, m = regexec(pattern = "^([A-Za-z_][A-Za-z0-9_]*) = (.*);$",
                                                                         text = one_line))[[1]]
                if ((length(assigned_scalar) == 3) && (assigned_scalar[[2]] %in% names_of_scalars)) {
                        expression <-  assigned_scalar[[3]]
                        for (argument in names(largest_of)) {
                                expression <-  gsub(pattern = paste0("stan::math::max\\(", argument, "\\)"),
                                                    replacement = largest_of[[argument]], x = expression)
                        }
                        expression_without_sizes <-  gsub(pattern = "stan::math::rows\\([A-Za-z_][A-Za-z0-9_]*\\)",
                                                          replacement = "SIZE", x = expression)
                        if (reads_rows(text_of_code = expression_without_sizes, names = names_of_rows)) {
                                stop("the scalar ", assigned_scalar[[2]], " reads the rows other than by max()")
                        }
                        lines <-  c(lines, paste0(assigned_scalar[[2]], " = ", expression, ";"))
                        next
                }
                ##
                ## ---- an assignment of a vector of the rows: whole, or its first row ------------------------------------------------
                ##
                assigned_vector <-  regmatches(x = one_line, m = regexec(pattern = paste0(
                        "^stan::model::assign\\(([A-Za-z_][A-Za-z0-9_]*), (.*), (\"assigning variable [^\"]*\")",
                        "(, stan::model::index_uni\\(1\\))?\\);$"), text = one_line))[[1]]
                if ((length(assigned_vector) == 5) && (assigned_vector[[2]] %in% names_of_rows)) {
                        expression <-  assigned_vector[[3]]
                        if (nzchar(assigned_vector[[5]])) {
                                ## (the first row: only in the first chunk; it reads only first rows and scalars)
                                if (reads_rows(text_of_code = without_reads_of_the_first_row(expression),
                                               names = names_of_rows)) {
                                        stop("the update of the first row of ", assigned_vector[[2]], " reads other rows")
                                }
                                lines <-  c(lines, paste0("if (first__ == 0) { ", one_line, " }"))
                                next
                        }
                        ## (a running sum: from the chunk's carry; its total is the next chunk's carry)
                        at <-  as.integer(gregexpr(pattern = "stan::math::cumulative_sum(", text = expression,
                                                   fixed = TRUE)[[1]])
                        at <-  at[at > 0]
                        carries_out <-  character(0)
                        for (position in rev(at)) {
                                open_bracket <-  position + nchar("stan::math::cumulative_sum")
                                running <-  fn_inside_of_the_bracket(text = expression, position_of_open_bracket = open_bracket)
                                carry <-  first_carry + n_carries
                                n_carries <-  n_carries + 1
                                expression <-  paste0(substring(expression, 1, position - 1),
                                                      "stan::math::add(stan::math::cumulative_sum(", running$inside,
                                                      "), carries_in__.coeff(", carry, "))",
                                                      substring(expression, running$close_bracket + 1,
                                                                nchar(expression)))
                                carries_out <-  c(carries_out, paste0("carries_out__(", carry, ") = carries_in__.coeff(",
                                                                      carry, ") + stan::math::sum(", running$inside, ");"))
                        }
                        without_the_running_sums <-  gsub(pattern = "stan::math::cumulative_sum\\(", replacement = "(",
                                                          x = expression)
                        if (grepl(pattern = "stan::math::(max|min|sum|dot_product|mean|log_sum_exp)\\(",
                                  x = without_the_running_sums)) {
                                stop("the vector ", assigned_vector[[2]], " reads a sum or an extreme of the rows")
                        }
                        lines <-  c(lines, paste0("stan::model::assign(", assigned_vector[[2]], ", ", expression, ", ",
                                                  assigned_vector[[4]], ");"), rev(carries_out))
                        next
                }
                ##
                ## ---- the test of the first row: if it fails, the function's other return (the statement as stanc wrote it) --------
                ##
                tested <-  regmatches(x = one_line, m = regexec(pattern = paste0(
                        "^if \\((.*)\\) \\{ (current_statement__ = [0-9]+; )?return (.*); \\}$"), text = one_line))[[1]]
                if ((length(tested) == 4) && is.null(test_of_the_first_row) && is.null(main_return)) {
                        if (reads_rows(text_of_code = without_reads_of_the_first_row(tested[[2]]), names = names_of_rows)) {
                                stop("the test reads rows other than the first")
                        }
                        test_of_the_first_row <-  tested[[2]]
                        main_return <-  tested[[4]]
                        lines <-  c(lines, paste0("if ((first__ == 0) && !(", tested[[2]], ")) { main_path__ = false; }"))
                        next
                }
                ##
                ## ---- the return -----------------------------------------------------------------------------------------------------
                ##
                returned <-  regmatches(x = one_line, m = regexec(pattern = "^return (.*);$", text = one_line))[[1]]
                if (length(returned) == 2) {
                        if (is.null(main_return)) main_return <-  returned[[2]]
                        break
                }
                stop("a statement of ", running_function$name, " that is not chunked: ", one_line)

        }
        if (is.null(main_return)) stop("no return in ", running_function$name)
        if (n_carries == 0) stop("no running sum in ", running_function$name)
        ##
        ## ---- the return adds over the rows: each sum or dot product of the rows times a scalar at most --------------------------------
        ##
        is_additive <-  function(node) {

                text_of_node <-  fn_text_of_cpp_tree(node = node)
                if (!reads_rows(text_of_code = text_of_node, names = names_of_rows)) return(FALSE)
                if (node$type == "brackets") return(is_additive(node$args[[1]]))
                if (node$type == "minus") return(is_additive(node$args[[1]]))
                if ((node$type == "operator") && (node$operator %in% c("+", "-"))) {
                        return(is_additive(node$args[[1]]) && is_additive(node$args[[2]]))
                }
                if ((node$type == "operator") && (node$operator %in% c("*", "/"))) {
                        reads <-  vapply(X = node$args, FUN = function(argument) {
                                reads_rows(text_of_code = fn_text_of_cpp_tree(node = argument), names = names_of_rows)
                        }, FUN.VALUE = logical(1))
                        if (node$operator == "/") return(reads[[1]] && !reads[[2]] && is_additive(node$args[[1]]))
                        return((sum(reads) == 1) && is_additive(node$args[[which(reads)]]))
                }
                if ((node$type == "call") && (node$name %in% c("stan::math::sum", "stan::math::dot_product"))) {
                        return(TRUE)
                }
                return(FALSE)

        }
        tree_of_the_return <-  fn_tree_of_cpp_expression(tokens = fn_tokens_of_cpp_expression(
                                                                expression_text = main_return))
        if (!is_additive(tree_of_the_return)) stop("the return of ", running_function$name, " does not add over the rows")
        lines <-  c(lines, paste0("return ", main_return, ";"))
        parameters <-  paste0("const auto& ", running_function$names_in_the_body, collapse = ", ")
        prelude <-  c(paste0("using local_scalar_t__ = stan::return_type_t<",
                             paste0("std::decay_t<decltype(", running_function$names_in_the_body, ")>", collapse = ", "),
                             ", std::decay_t<decltype(carries_in__)>",
                             if (length(largest_of) > 0) {
                                     paste0(", std::decay_t<decltype(", unlist(largest_of), ")>", collapse = "")
                             } else {
                                     ""
                             }, ">;"),
                      "local_scalar_t__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());",
                      "(void) DUMMY_VAR__;")
        return(list(lambda = paste0("[&](", parameters, ") { ", paste(c(prelude, lines), collapse = " "), " }"),
                    n_carries = n_carries))

}
##
## ---- The terms of a sum: each with its sign ----------------------------------------------------------------------------------------
##
fn_terms_of_a_sum <-  function(node,
                               sign) {

        if (node$type == "brackets") return(fn_terms_of_a_sum(node = node$args[[1]], sign = sign))
        if (node$type == "minus") return(fn_terms_of_a_sum(node = node$args[[1]], sign = -sign))
        if ((node$type == "operator") && (node$operator %in% c("+", "-"))) {
                return(c(fn_terms_of_a_sum(node = node$args[[1]], sign = sign),
                         fn_terms_of_a_sum(node = node$args[[2]], sign = if (node$operator == "-") -sign else sign)))
        }
        if ((node$type == "call") && (node$name %in% c("stan::math::add", "stan::math::subtract")) &&
            (length(node$args) == 2)) {
                return(c(fn_terms_of_a_sum(node = node$args[[1]], sign = sign),
                         fn_terms_of_a_sum(node = node$args[[2]],
                                           sign = if (node$name == "stan::math::subtract") -sign else sign)))
        }
        return(list(list(sign = sign, node = node)))

}
##
## ---- The running-sum contracts of NicoSIMD's general functions (AUTO_RUNNING_SUM in its headers) ---------------------------------
##
## (each: the function's name, the role of each argument, rows or log_carry, and the expression of the carry after a chunk)
fn_contracts_of_running_sums <-  function(header_paths) {

        contracts <-  list()
        for (path in header_paths) {

                lines <-  readLines(path, warn = FALSE)
                starts <-  grep("^// AUTO_RUNNING_SUM [A-Za-z_][A-Za-z0-9_]*$", lines)
                for (start in starts) {

                        name <-  sub("^// AUTO_RUNNING_SUM ", "", lines[[start]])
                        if ((start + 2 > length(lines)) ||
                            !startsWith(lines[[start + 1]], "// AUTO_RUNNING_SUM_ARGUMENTS ") ||
                            !startsWith(lines[[start + 2]], "// AUTO_RUNNING_SUM_CARRY_OUT ")) {
                                stop("an incomplete running-sum contract in ", path)
                        }
                        pairs <-  strsplit(strsplit(sub("^// AUTO_RUNNING_SUM_ARGUMENTS ", "", lines[[start + 1]]),
                                                    " +")[[1]], ":", fixed = TRUE)
                        if (any(lengths(pairs) != 2)) stop("an invalid running-sum contract in ", path)
                        roles <-  setNames(vapply(pairs, `[[`, character(1), 2), vapply(pairs, `[[`, character(1), 1))
                        if (any(!(roles %in% c("rows", "log_carry"))) || (sum(roles == "log_carry") != 1) ||
                            !any(roles == "rows")) {
                                stop("invalid running-sum roles in ", path)
                        }
                        contracts[[name]] <-  list(name = name, roles = roles,
                                                   carry_out = sub("^// AUTO_RUNNING_SUM_CARRY_OUT ", "",
                                                                   lines[[start + 2]]))

                }

        }
        return(contracts)

}
##
## ---- Each statement lp_accum__.add(E) of a log_prob body whose E calls a running-sum function, chunked ---------------------------------
##
fn_auto_chunk_the_running_sums_in_one_log_prob_body <-  function(body_text,
                                                                 data,
                                                                 parameters_and_locals,
                                                                 N_chunks_of_auto_chunking_text,
                                                                 running_sum_functions) {

        report <-  character(0)
        locals_moved <-  character(0)
        headers <-  getOption("NicoStan.auto_chunking_density_contract_headers",
                              list.files(system.file("include/NicoSIMD", package = "NicoSIMD"), pattern = "[.]hpp$",
                                         full.names = TRUE))
        contracts <-  fn_contracts_of_running_sums(header_paths = headers)
        if ((length(running_sum_functions) == 0) && (length(contracts) == 0)) {
                return(list(body_text = body_text, report = report, locals_moved = locals_moved))
        }
        starts <-  as.integer(gregexpr(pattern = "lp_accum__.add(", text = body_text, fixed = TRUE)[[1]])
        starts <-  starts[starts > 0]
        for (start in rev(starts)) {

                open_bracket <-  start + nchar("lp_accum__.add")
                statement <-  tryCatch(fn_inside_of_the_bracket(text = body_text, position_of_open_bracket = open_bracket),
                                       error = function(e) NULL)
                if (is.null(statement)) next
                pattern_of_a_call <-  paste0("(?<![A-Za-z0-9_:])(",
                                             paste(c(names(running_sum_functions), names(contracts)), collapse = "|"),
                                             ")\\(")
                if (!grepl(pattern = pattern_of_a_call, x = statement$inside, perl = TRUE)) next
                if (grepl(pattern = "nico_chunked_sum_begin", x = statement$inside, fixed = TRUE)) next
                what <-  gsub(pattern = "\\s+", replacement = " ", x = statement$inside)
                outcome <-  tryCatch({
                        tree <-  fn_tree_of_cpp_expression(tokens = fn_tokens_of_cpp_expression(
                                                                        expression_text = statement$inside))
                        terms <-  fn_terms_of_a_sum(node = tree, sign = 1)
                        texts <-  character(0)
                        shared <-  character(0)
                        moved <-  list()
                        functions_made <-  character(0)
                        largest_rows <-  list()
                        n_carries <-  0
                        row_set <-  NA_character_
                        start_of_the_loop <-  fn_start_of_the_innermost_loop(body_text = body_text, position = start)
                        data_of_the_statement <-  fn_data_with_the_row_sets_of_the_slices(node = tree, data = data)
                        rows_of <-  function(node) {

                                part <-  fn_row_wise_kind_of_node(node = node, data = data_of_the_statement,
                                                                  parameters_and_locals = parameters_and_locals,
                                                                  body_text = body_text, struct_of_arrays = FALSE,
                                                                  start_of_the_loop = start_of_the_loop)
                                if (part$kind != "rows") stop("an argument is not row-wise: ", fn_text_of_cpp_tree(node))
                                if (!is.na(row_set) && (part$row_set != row_set)) stop("two row sets in the statement")
                                row_set <<-  part$row_set
                                shared <<-  c(shared, part$names_of_vars)
                                moved <<-  c(moved, part$moved)
                                return(part)

                        }
                        for (term in terms) {

                                node <-  term$node
                                sign <-  if (term$sign < 0) "-" else "+"
                                ## (a running-sum function of NicoSIMD, by its contract: each chunk from the carry before it)
                                if ((node$type == "call") && (node$name %in% names(contracts))) {
                                        contract <-  contracts[[node$name]]
                                        arguments <-  node$args
                                        if ((length(arguments) > 0) &&
                                            (fn_text_of_cpp_tree(arguments[[length(arguments)]]) == "pstream__")) {
                                                arguments <-  arguments[-length(arguments)]
                                        }
                                        if (length(arguments) != length(contract$roles)) {
                                                stop("the call of ", node$name, " has another number of arguments")
                                        }
                                        argument_texts <-  character(0)
                                        for (i in seq_along(arguments)) {
                                                if (contract$roles[[i]] == "rows") {
                                                        argument_texts <-  c(argument_texts, rows_of(arguments[[i]])$text)
                                                        next
                                                }
                                                text_of_the_carry <-  fn_text_of_cpp_tree(arguments[[i]])
                                                if (grepl(pattern = "[A-Za-z_][A-Za-z0-9_]*(?![A-Za-z0-9_(:])",
                                                          x = gsub(pattern = "[0-9.e+-]+", replacement = "",
                                                                   x = text_of_the_carry), perl = TRUE)) {
                                                        stop("the carry of ", node$name, " is not a constant")
                                                }
                                                argument_texts <-  c(argument_texts, text_of_the_carry)
                                        }
                                        parameters <-  names(contract$roles)
                                        carry_parameter <-  parameters[contract$roles == "log_carry"]
                                        carry_out <-  gsub(pattern = paste0("(?<![A-Za-z0-9_])", carry_parameter,
                                                                            "(?![A-Za-z0-9_])"),
                                                           replacement = "carry__", x = contract$carry_out, perl = TRUE)
                                        arguments_of_the_call <-  replace(parameters, parameters == carry_parameter,
                                                                          "carry__")
                                        lambda <-  paste0("[&](", paste0("const auto& ", parameters, collapse = ", "),
                                                          ") { using local_scalar_t__ = stan::return_type_t<",
                                                          paste0("std::decay_t<decltype(", parameters, ")>",
                                                                 collapse = ", "),
                                                          ", std::decay_t<decltype(carries_in__)>>; ",
                                                          "const local_scalar_t__ carry__ = (first__ == 0) ? ",
                                                          "local_scalar_t__(", carry_parameter, ") : ",
                                                          "local_scalar_t__(carries_in__.coeff(", n_carries, ")); ",
                                                          "carries_out__(", n_carries, ") = ", carry_out, "; ",
                                                          "return ", node$name, "(",
                                                          paste(arguments_of_the_call, collapse = ", "),
                                                          ", pstream__); }")
                                        n_carries <-  n_carries + 1
                                        value_name <-  paste0("running_sum_function__", length(functions_made) + 1)
                                        functions_made <-  c(functions_made, paste0("const auto ", value_name, " = ",
                                                                                   lambda, "(",
                                                                                   paste(argument_texts, collapse = ", "),
                                                                                   ");"))
                                        texts <-  c(texts, paste0(sign, " ", value_name))
                                        next
                                }
                                if ((node$type == "call") && (node$name %in% names(running_sum_functions))) {
                                        running_function <-  running_sum_functions[[node$name]]
                                        arguments <-  node$args
                                        if ((length(arguments) > 0) &&
                                            (fn_text_of_cpp_tree(arguments[[length(arguments)]]) == "pstream__")) {
                                                arguments <-  arguments[-length(arguments)]
                                        }
                                        if (length(arguments) != length(running_function$names_in_the_body)) {
                                                stop("the call of ", node$name, " has another number of arguments")
                                        }
                                        argument_texts <-  character(0)
                                        row_arguments <-  character(0)
                                        for (i in seq_along(arguments)) {
                                                ## (a constant, e.g. negative_infinity(), or a scalar of the model)
                                                text_of_the_argument <-  fn_text_of_cpp_tree(arguments[[i]])
                                                is_constant <-  !grepl(pattern = "[A-Za-z_][A-Za-z0-9_]*(?![A-Za-z0-9_(:])",
                                                                       x = gsub(pattern = "[0-9.e+-]+", replacement = "",
                                                                                x = text_of_the_argument), perl = TRUE)
                                                is_scalar <-  is_constant || tryCatch({
                                                        kind <-  fn_row_wise_kind_of_node(
                                                                node = arguments[[i]], data = data_of_the_statement,
                                                                parameters_and_locals = parameters_and_locals,
                                                                body_text = body_text, struct_of_arrays = FALSE,
                                                                start_of_the_loop = start_of_the_loop)
                                                        kind$kind == "scalar"
                                                }, error = function(e) FALSE)
                                                if (is_scalar) {
                                                        argument_texts <-  c(argument_texts,
                                                                             fn_text_of_cpp_tree(arguments[[i]]))
                                                        next
                                                }
                                                part <-  rows_of(arguments[[i]])
                                                argument_texts <-  c(argument_texts, part$text)
                                                row_arguments <-  c(row_arguments, running_function$names_in_the_body[[i]])
                                        }
                                        ## (the largest row of each row argument whose max() the body reads)
                                        largest_of <-  list()
                                        for (i in seq_along(running_function$names_in_the_body)) {
                                                name <-  running_function$names_in_the_body[[i]]
                                                reads_max <-  any(grepl(pattern = paste0("stan::math::max\\(", name, "\\)"),
                                                                        x = running_function$statements))
                                                if (reads_max && (name %in% row_arguments)) {
                                                        largest_name <-  paste0("largest_row__", length(largest_rows) + 1)
                                                        largest_rows[[largest_name]] <-  argument_texts[[i]]
                                                        largest_of[[name]] <-  largest_name
                                                }
                                        }
                                        made <-  fn_running_sum_function_for_a_chunk(running_function = running_function,
                                                                                     row_arguments = row_arguments,
                                                                                     first_carry = n_carries,
                                                                                     largest_of = largest_of)
                                        n_carries <-  n_carries + made$n_carries
                                        value_name <-  paste0("running_sum_function__", length(functions_made) + 1)
                                        functions_made <-  c(functions_made, paste0("const auto ", value_name, " = ",
                                                                                   made$lambda, "(",
                                                                                   paste(argument_texts, collapse = ", "),
                                                                                   ");"))
                                        texts <-  c(texts, paste0(sign, " ", value_name))
                                        next
                                }
                                if ((node$type == "call") && (node$name == "stan::math::dot_product") &&
                                    (length(node$args) == 2)) {
                                        first <-  rows_of(node$args[[1]])
                                        second <-  rows_of(node$args[[2]])
                                        if (!isTRUE(first$is_var) && !isTRUE(second$is_var)) stop("a dot product of data")
                                        texts <-  c(texts, paste0(sign, " stan::math::dot_product(", first$text, ", ",
                                                                  second$text, ")"))
                                        next
                                }
                                if ((node$type == "call") && (node$name == "stan::math::sum") && (length(node$args) == 1)) {
                                        part <-  rows_of(node$args[[1]])
                                        texts <-  c(texts, paste0(sign, " stan::math::sum(", part$text, ")"))
                                        next
                                }
                                stop("a term that does not add over the rows: ", fn_text_of_cpp_tree(node))

                        }
                        if (length(functions_made) == 0) stop("no running-sum function in the statement")
                        shared <-  shared[!duplicated(names(shared))]
                        moved <-  moved[!duplicated(names(moved))]
                        list(kind = "rows", texts = texts, shared = shared, moved = moved, functions_made = functions_made,
                             largest_rows = largest_rows, n_carries = n_carries, row_set = row_set)
                }, error = function(e) list(kind = "not", reason = conditionMessage(e)))
                if (outcome$kind != "rows") {
                        report <-  c(report, paste0("left as stanc wrote it: ", what, " (", outcome$reason, ")"))
                        next
                }
                reason <-  fn_why_the_locals_cannot_be_made_in_the_tile(body_text = body_text, candidate_start = start,
                                                                        moved = outcome$moved,
                                                                        names_of_vars = outcome$shared)
                if (!is.null(reason)) {
                        report <-  c(report, paste0("left as stanc wrote it: ", what, " (", reason, ")"))
                        next
                }
                ##
                ## ---- the replacement: the largest rows first, then the chunks with their carries ------------------------------------
                ##
                shared_names <-  names(outcome$shared)
                moved_lines <-  vapply(X = names(outcome$moved), FUN = function(local_name) {
                        paste0("const auto ", local_name, " = stan::math::eval(", outcome$moved[[local_name]], ");")
                }, FUN.VALUE = character(1))
                made_in_the_tile <-  paste(moved_lines, collapse = " ")
                shared_parameters <-  paste0("const auto& ", shared_names)
                locals_as_stanc_made_them <-  fn_statements_of_the_locals_as_stanc_made_them(
                                                    body_text = body_text, names_of_the_locals = names(outcome$moved))
                if (is.null(locals_as_stanc_made_them)) locals_as_stanc_made_them <-  ""
                as_stanc_wrote_it <-  paste0("/*nico_locals_of_stanc_begin*/ ", locals_as_stanc_made_them,
                                             "/*nico_locals_of_stanc_end*/ return ", statement$inside, ";")
                largest_lines <-  vapply(X = names(outcome$largest_rows), FUN = function(largest_name) {
                        paste0("const auto ", largest_name, " = nico::math::auto_chunking::",
                               "largest_row_in_the_chunks_of_reduce_sum_static(", outcome$row_set, ", ",
                               N_chunks_of_auto_chunking_text, ", [&](", paste(c("const int first__", "const int n_rows__",
                                                                                 shared_parameters), collapse = ", "),
                               ") { ", made_in_the_tile, " return stan::math::eval(",
                               outcome$largest_rows[[largest_name]], "); }, ",
                               paste(outcome$shared, collapse = ", "), ");")
                }, FUN.VALUE = character(1))
                chunk_parameters <-  c("const int first__", "const int n_rows__", "const auto& carries_in__",
                                       shared_parameters,
                                       if (length(outcome$largest_rows) > 0) {
                                               paste0("const auto& ", names(outcome$largest_rows))
                                       })
                value_of_the_chunk <-  paste(outcome$texts, collapse = " ")
                chunk <-  paste0("[&](", paste(chunk_parameters, collapse = ", "), ") { ", made_in_the_tile, " ",
                                 "using scalar_of_the_chunk__ = stan::return_type_t<std::decay_t<decltype(carries_in__)>",
                                 paste0(", std::decay_t<decltype(", c(shared_names, names(outcome$largest_rows)), ")>",
                                        collapse = ""), ">; ",
                                 "Eigen::Matrix<scalar_of_the_chunk__, -1, 1> carries_out__(", outcome$n_carries, "); ",
                                 "bool main_path__ = true; ", paste(outcome$functions_made, collapse = " "), " ",
                                 "return std::make_tuple(scalar_of_the_chunk__(", value_of_the_chunk, "), carries_out__, ",
                                 "main_path__); }")
                replacement <-  paste0("/*nico_chunked_sum_begin*/[&]() -> local_scalar_t__ { ",
                                       "if (nico::math::auto_chunking::is_one_chunk(", outcome$row_set, ", ",
                                       N_chunks_of_auto_chunking_text, ")) { ", as_stanc_wrote_it, " } ",
                                       paste(largest_lines, collapse = " "), " ",
                                       "return nico::math::auto_chunking::",
                                       "sum_of_rows_with_running_sums_in_the_chunks_of_reduce_sum_static_or_as_stanc_wrote_it(",
                                       outcome$row_set, ", ", N_chunks_of_auto_chunking_text, ", ", outcome$n_carries,
                                       ", [&]() { ", as_stanc_wrote_it, " }, ", chunk, ", ",
                                       paste(c(unname(outcome$shared), names(outcome$largest_rows)), collapse = ", "),
                                       "); }()/*nico_chunked_sum_end*/")
                body_text <-  paste0(substring(body_text, 1, open_bracket), replacement,
                                     substring(body_text, statement$close_bracket, nchar(body_text)))
                locals_moved <-  c(locals_moved, names(outcome$moved))
                report <-  c(report, paste0("tiled over ", outcome$row_set, " rows, with running sums: ", what,
                                            " (carries: ", outcome$n_carries, "; largest rows: ",
                                            length(outcome$largest_rows), "; shared: ",
                                            paste(shared_names, collapse = ", "), "; made in the chunk: ",
                                            paste(names(outcome$moved), collapse = ", "), ")"))

        }
        return(list(body_text = body_text, report = report, locals_moved = locals_moved))

}
##
## ---- The rewrite of one log_prob_impl body -----------------------------------------------------------------------------------------
##
fn_auto_chunk_one_log_prob_body <-  function(body_text,
                                             data,
                                             N_chunks_of_auto_chunking_text,
                                             running_sum_functions,
                                             names_of_the_nuisance_blocks) {

        parameters_and_locals <-  fn_parameters_and_locals_of_log_prob(
                body_text = body_text,
                names_of_the_nuisance_blocks = names_of_the_nuisance_blocks)
        aliases <-  attr(data, "scalar_dimension_aliases")
        data <-  c(data, fn_slices_of_the_data_in_log_prob(body_text = body_text, data = data))
        attr(data, "scalar_dimension_aliases") <-  aliases
        report <-  character(0)
        ##
        ## ---- the candidates: each stan::math::sum(E), and each density call (Stan Math's or a nico_ one) ---------------------------
        ##
        sum_starts <-  as.integer(gregexpr(pattern = "stan::math::sum\\(", text = body_text)[[1]])
        sum_starts <-  sum_starts[sum_starts > 0]
        pattern_of_a_density <-  paste0("(?<![A-Za-z0-9_:])(?:stan::math::)?[A-Za-z_][A-Za-z0-9_]*_",
                                        "(?:lpdf|lpmf|lupdf|lupmf|lccdf|lcdf)(?:<[^<>()]*>)?\\(")
        density_matches <-  gregexpr(pattern = pattern_of_a_density, text = body_text, perl = TRUE)[[1]]
        density_starts <-  as.integer(density_matches)
        density_lengths <-  attr(density_matches, "match.length")
        density_lengths <-  density_lengths[density_starts > 0]
        density_starts <-  density_starts[density_starts > 0]
        candidates <-  data.frame(start = c(sum_starts, density_starts),
                                  open_bracket = c(sum_starts + nchar("stan::math::sum"),
                                                   density_starts + density_lengths - 1),
                                  is_density = c(rep(FALSE, length(sum_starts)),
                                                 rep(TRUE, length(density_starts))))
        if ((nrow(candidates) == 0) && (length(running_sum_functions) == 0)) {
                return(list(body_text = body_text, report = "no sum() or density in log_prob"))
        }
        candidates <-  candidates[order(candidates$start, decreasing = TRUE), ]
        locals_moved <-  character(0)
        ##
        ## ---- each candidate, from the last to the first (so that the positions before it stay the same) ----------------------------
        ##
        for (k in seq_len(nrow(candidates))) {
                start <-  candidates$start[[k]]
                open_bracket <-  candidates$open_bracket[[k]]
                is_density <-  candidates$is_density[[k]]
                start_of_the_loop <-  fn_start_of_the_innermost_loop(body_text = body_text, position = start)
                close_bracket <-  fn_end_of_bracket(text = body_text, position_of_open_bracket = open_bracket)
                argument_text <-  substring(body_text, open_bracket + 1, close_bracket - 1)
                call_text <-  substring(body_text, start, close_bracket)
                outcome <-  tryCatch({
                        if (is_density) {
                                tree <-  fn_tree_of_cpp_expression(tokens = fn_tokens_of_cpp_expression(
                                                                                expression_text = call_text))
                                fn_row_wise_kind_of_density(node = tree,
                                                            data = fn_data_with_the_row_sets_of_the_slices(
                                                                    node = tree, data = data),
                                                            parameters_and_locals = parameters_and_locals,
                                                            body_text = body_text,
                                                            start_of_the_loop = start_of_the_loop)
                        } else {
                                tree <-  fn_tree_of_cpp_expression(tokens = fn_tokens_of_cpp_expression(
                                                                                expression_text = argument_text))
                                fn_row_wise_kind_of_node(node = tree,
                                                         data = fn_data_with_the_row_sets_of_the_slices(
                                                                 node = tree, data = data),
                                                         parameters_and_locals = parameters_and_locals,
                                                         body_text = body_text, struct_of_arrays = TRUE,
                                                         start_of_the_loop = start_of_the_loop)
                        }
                }, error = function(e) list(kind = "not", reason = conditionMessage(e)))
                one_line_argument <-  gsub(pattern = "\\s+", replacement = " ", x = argument_text)
                what <-  if (is_density) gsub(pattern = "\\s+", replacement = " ", x = call_text)
                         else paste0("sum(", one_line_argument, ")")
                if ((outcome$kind != "rows") || !isTRUE(outcome$is_var)) {
                        reason <-  if (!is.null(outcome$reason)) outcome$reason
                                   else "not a row-wise expression of vars"
                        report <-  c(report, paste0("left as stanc wrote it: ", what, " (", reason, ")"))
                        next
                }
                reason <-  fn_why_the_locals_cannot_be_made_in_the_tile(body_text = body_text,
                                                                        candidate_start = start,
                                                                        moved = outcome$moved,
                                                                        names_of_vars = outcome$names_of_vars)
                if (!is.null(reason)) {
                        report <-  c(report, paste0("left as stanc wrote it: ", what, " (", reason, ")"))
                        next
                }
                ##
                ## ---- the tiles: the N-row locals made first, then the sum of the expression ----------------------------------------
                ##
                shared <-  outcome$names_of_vars
                ## (each local made in the tile evaluated at once, as stanc's assign does: in log_prob<double> an
                ##  elementwise call of the rows is a lazy Eigen expression whose temporaries end with the statement)
                moved_lines <-  vapply(X = names(outcome$moved),
                                       FUN = function(local_name) paste0("const auto ", local_name,
                                                                         " = stan::math::eval(",
                                                                         outcome$moved[[local_name]], ");"),
                                       FUN.VALUE = character(1))
                ## (a density whose term of a row reads the row before: its runtime, given the group starts, and
                ##  each chunk told whether its rows begin with the row before its first row)
                with_the_row_before <-  !is.null(outcome$group_start)
                lambda_parameters <-  paste(c("const int first__", "const int n_rows__",
                                              if (with_the_row_before) "const bool row_before__",
                                              paste0("const auto& ", names(shared))), collapse = ", ")
                name_of_the_sum <-  if (with_the_row_before) {
                                            "sum_of_rows_with_the_row_before_in_the_chunks_of_reduce_sum_static"
                                    } else {
                                            "sum_of_rows_in_the_chunks_of_reduce_sum_static"
                                    }
                group_starts <-  if (with_the_row_before) paste0(outcome$group_start, ", ") else ""
                ## (a density of the tile's rows is its own value; a sum's expression is summed)
                value_of_the_tile <-  if (is_density) outcome$text
                                      else paste0("stan::math::sum(", outcome$text, ")")
                made_in_the_tile <-  paste(moved_lines, collapse = " ")
                if (length(moved_lines) > 0) made_in_the_tile <-  paste0(made_in_the_tile, " ")
                replacement <-  paste0("nico::math::auto_chunking::",
                                       name_of_the_sum, "(", outcome$row_set, ", ",
                                       N_chunks_of_auto_chunking_text, ", ", group_starts, "[&](",
                                       lambda_parameters, ") { ", made_in_the_tile, "return ", value_of_the_tile,
                                       "; }, ",
                                       paste(shared, collapse = ", "), ")")
                ## (one chunk runs stanc's own statement, with its N-row locals made as stanc made them, between
                ##  marks: those copies are dropped below if the locals stay outside the tiles anyway)
                locals_as_stanc_made_them <-  fn_statements_of_the_locals_as_stanc_made_them(
                                                    body_text = body_text,
                                                    names_of_the_locals = names(outcome$moved))
                density_without_a_contract <-  outcome$name_of_a_density_without_a_contract
                if (!is.null(density_without_a_contract) && is.null(locals_as_stanc_made_them)) {
                        report <-  c(report, paste0("left as stanc wrote it: ", what, " (a density with no ",
                                                    "declared row-sum contract whose statement cannot be ",
                                                    "run again for its check)"))
                        next
                }
                if (!is.null(locals_as_stanc_made_them)) {
                        replacement <-  paste0("nico::math::auto_chunking::",
                                               name_of_the_sum, "_",
                                               "or_as_stanc_wrote_it(",
                                               outcome$row_set, ", ", N_chunks_of_auto_chunking_text,
                                               ", ", group_starts, "[&]() { ",
                                               "/*nico_locals_of_stanc_begin*/ ", locals_as_stanc_made_them,
                                               "/*nico_locals_of_stanc_end*/ ",
                                               "return ", substring(body_text, start, close_bracket),
                                               "; }, [&](", lambda_parameters, ") { ", made_in_the_tile,
                                               "return ", value_of_the_tile, "; }, ",
                                               paste(shared, collapse = ", "), ")")
                }
                ## (a density with no declared row-sum contract: the runtime that checks, at its first call, the
                ##  chunks' sum against stanc's own statement)
                if (!is.null(density_without_a_contract)) {
                        replacement <-  sub(paste0("nico::math::auto_chunking::", name_of_the_sum,
                                                   "_or_as_stanc_wrote_it("),
                                            paste0("nico::math::auto_chunking::sum_of_rows_of_a_density_",
                                                   "without_a_contract_in_the_chunks_of_reduce_sum_static_",
                                                   "or_as_stanc_wrote_it(\"", density_without_a_contract,
                                                   "\", "),
                                            replacement, fixed = TRUE)
                }
                ## (a shared argument that reads NicoStan's nuisance block: read only with more than one chunk, as
                ##  one chunk runs stanc's own statement, which reads the block itself)
                if (!is.null(locals_as_stanc_made_them) &&
                    any(grepl(pattern = "nicostan_nuisance_", x = shared, fixed = TRUE))) {
                        call_with_more_than_one_chunk <-  if (!is.null(density_without_a_contract)) {
                                                                  replacement
                                                          } else {
                                                                  paste0("nico::math::auto_chunking::",
                                                                         name_of_the_sum, "(",
                                                                         outcome$row_set, ", ",
                                                                         N_chunks_of_auto_chunking_text,
                                                                         ", ", group_starts, "[&](",
                                                                         lambda_parameters, ") { ",
                                                                         made_in_the_tile, "return ",
                                                                         value_of_the_tile, "; }, ",
                                                                         paste(shared, collapse = ", "),
                                                                         ")")
                                                          }
                        replacement <-  paste0("[&]() -> local_scalar_t__ { ",
                                               "if (nico::math::auto_chunking::chunk_size_of_auto_chunking(",
                                               outcome$row_set, ", ", N_chunks_of_auto_chunking_text, ") >= ",
                                               outcome$row_set, ") { ",
                                               "/*nico_locals_of_stanc_begin*/ ", locals_as_stanc_made_them,
                                               "/*nico_locals_of_stanc_end*/ ",
                                               "return ", substring(body_text, start, close_bracket), "; } ",
                                               "return ", call_with_more_than_one_chunk, "; }()")
                }
                if (any(grepl("gathered_at_unique_groups_of(", shared, fixed = TRUE))) {
                        replacement <-  sub("sum_of_rows_in_the_chunks_of_reduce_sum_static",
                                             "sum_of_grouped_rows_in_the_chunks_of_reduce_sum_static",
                                             replacement, fixed = TRUE)
                }
                replacement <-  paste0("/*nico_chunked_sum_begin*/", replacement, "/*nico_chunked_sum_end*/")
                body_text <-  paste0(substring(body_text, 1, start - 1), replacement,
                                     substring(body_text, close_bracket + 1, nchar(body_text)))
                locals_moved <-  c(locals_moved, names(outcome$moved))
                names_made_in_the_tile <-  if (length(outcome$moved) > 0) {
                                                   paste(names(outcome$moved), collapse = ", ")
                                           } else {
                                                   "none"
                                           }
                report <-  c(report, paste0("tiled over ", outcome$row_set, " rows",
                                            if (with_the_row_before) {
                                                    paste0(", each chunk with the row before its first row ",
                                                           "unless ", outcome$group_start, " starts a group there")
                                            } else {
                                                    ""
                                            }, ": ", what,
                                            " (shared: ", paste(shared, collapse = ", "), "; made in the tile: ",
                                            names_made_in_the_tile, "; one chunk: ",
                                            if (!is.null(locals_as_stanc_made_them)) "stanc's own statement" else
                                                  "the tile of all the rows",
                                            if (!is.null(density_without_a_contract)) {
                                                    paste0("; a density with no declared row-sum ",
                                                           "contract: its first call checks the ",
                                                           "chunks' sum against stanc's own statement")
                                            }, ")"))
        }
        ##
        ## ---- each statement that adds a running-sum function of the rows: chunked with the carries of its running sums -----------
        ##
        running <-  fn_auto_chunk_the_running_sums_in_one_log_prob_body(
                          body_text                      = body_text,
                          data                           = data,
                          parameters_and_locals          = parameters_and_locals,
                          N_chunks_of_auto_chunking_text = N_chunks_of_auto_chunking_text,
                          running_sum_functions          = running_sum_functions)
        body_text <-  running$body_text
        report <-  c(report, running$report)
        locals_moved <-  c(locals_moved, running$locals_moved)
        ##
        ## ---- the N-row locals made in the tiles: declarations, size checks and assignments removed together ------------------------
        ##
        ## (only if none of them is used anywhere else; else all are kept, made both outside and inside the tiles)
        locals_moved <-  unique(locals_moved)
        body_without_all <-  body_text
        for (local_name in locals_moved) {
                assignment <-  fn_assignment_of_local(body_text = body_without_all, local_name = local_name)
                body_without_all <-  paste0(substring(body_without_all, 1, assignment$start - 1),
                                            substring(body_without_all, assignment$end + 1,
                                                      nchar(body_without_all)))
                vector_of_locals <-  "Eigen::Matrix<local_scalar_t__,\\s*-1,\\s*1>"
                declaration_pattern <-  paste0(vector_of_locals, "\\s+", local_name, "\\s*=\\s*",
                                               vector_of_locals, "::Constant\\(", "[^;]*\\);")
                validation_pattern <-  paste0("stan::math::validate_non_negative_index\\(\\s*\"", local_name,
                                              "\"[^;]*\\);")
                body_without_all <-  sub(pattern = declaration_pattern, replacement = "", x = body_without_all,
                                         perl = TRUE)
                body_without_all <-  sub(pattern = validation_pattern, replacement = "", x = body_without_all,
                                         perl = TRUE)
        }
        ## (uses left outside the tiles, where the locals are now made: each "const auto <local> =")
        pattern_of_a_chunked_sum <-  paste0("nico::math::auto_chunking::",
                                            "sum_of_rows_in_the_chunks_of_reduce_sum_static\\(.*?\\}, [^)]*\\)")
        ## (each call between its marks, and the locals made as stanc made them inside it, between theirs)
        pattern_of_a_marked_call <-  paste0("(?s)/\\*nico_chunked_sum_begin\\*/.*?",
                                            "/\\*nico_chunked_sum_end\\*/")
        pattern_of_the_marked_locals <-  paste0("(?s)/\\*nico_locals_of_stanc_begin\\*/.*?",
                                                "/\\*nico_locals_of_stanc_end\\*/ ")
        outside_text <-  gsub(pattern = pattern_of_a_chunked_sum,
                              replacement = "",
                              x = gsub(pattern = pattern_of_a_marked_call, replacement = "", x = body_without_all,
                                       perl = TRUE),
                              perl = TRUE)
        used_outside <-  vapply(X = locals_moved, FUN = function(local_name) {
                gregexpr(pattern = paste0("(?<![A-Za-z0-9_\"])", local_name, "(?![A-Za-z0-9_\"])"),
                         text = outside_text, perl = TRUE)[[1]][[1]] > 0
        }, FUN.VALUE = logical(1))
        if (length(locals_moved) > 0) {
                if (any(used_outside)) {
                        report <-  c(report, paste0("the locals ", paste(locals_moved, collapse = ", "),
                                                    " are kept outside the tiles too (used outside: ",
                                                    paste(locals_moved[used_outside], collapse = ", "), ")"))
                } else {
                        body_text <-  body_without_all
                        report <-  c(report, paste0("the locals ", paste(locals_moved, collapse = ", "),
                                                    ": their declarations, size checks and assignments removed",
                                                    " (made inside the tiles)"))
                }
        }
        ## (the locals kept outside the tiles: one chunk reads them there, so their marked copies are dropped)
        if ((length(locals_moved) > 0) && any(used_outside)) {
                body_text <-  gsub(pattern = pattern_of_the_marked_locals, replacement = "", x = body_text,
                                   perl = TRUE)
        }
        body_text <-  gsub(pattern = "/\\*nico_(chunked_sum|locals_of_stanc)_(begin|end)\\*/ ?", replacement = "",
                           x = body_text, perl = TRUE)
        return(list(body_text = body_text, report = report))

}
##
## ---- The data members of the model: the names stanc declares in its private section, before the constructor -----------------------
##
fn_data_members_of_generated_cpp <-  function(text) {

        start <-  regexpr(pattern = "\n private:\n", text = text, fixed = TRUE)
        end <-  regexpr(pattern = ": model_base_crtp(0) {", text = text, fixed = TRUE)
        if ((start < 0) || (end < start)) stop("no private section before the model constructor")
        section <-  substring(text, start, end)
        ## (each "  type name;" or "  type name{...};" line)
        declarations <-  regmatches(x = section,
                                    m = gregexpr(pattern = "(?m)^  [^\n;(]*?[A-Za-z_][A-Za-z0-9_]*\\s*(\\{[^;\n]*\\})?;$",
                                                 text = section, perl = TRUE))[[1]]
        names_of_members <-  sub(pattern = "^.*?([A-Za-z_][A-Za-z0-9_]*)\\s*(\\{[^;]*\\})?;$", replacement = "\\1",
                                 x = declarations, perl = TRUE)
        return(unique(names_of_members))

}
##
## ---- Each direct call of a reduce_sum partial-sum function on all the rows of its sliced argument: cut into tiles ------------------
##
## (a partial-sum function f given to reduce_sum / reduce_sum_static is, by its contract, the sum over the rows start ..
##  end of its sliced argument; the model's call f(x, 1, size(x), ...), its branch with one chunk, is cut into the
##  chunks reduce_sum_static makes, each tile given its own rows of x; data are read in place, every other argument
##  is copied into each tile's nested tape; a function that reads NicoStan's nuisance block itself is left as it is)
fn_auto_chunk_the_calls_of_partial_sum_functions <-  function(body_text,
                                                              text,
                                                              N_chunks_of_auto_chunking_text) {

        report <-  character(0)
        functors <-  regmatches(x = text,
                                m = gregexpr(pattern = "reduce_sum(?:_static)?<\\s*[A-Za-z_][A-Za-z0-9_]*?_rsfunctor__",
                                             text = text, perl = TRUE))[[1]]
        names_of_partial_sums <-  unique(sub(pattern = "^.*<\\s*([A-Za-z_][A-Za-z0-9_]*?)_rsfunctor__$",
                                             replacement = "\\1", x = functors, perl = TRUE))
        if (length(names_of_partial_sums) == 0) return(list(body_text = body_text, report = report))
        data_members <-  fn_data_members_of_generated_cpp(text = text)
        for (f in names_of_partial_sums) {

                pattern_of_a_call <-  paste0("(?<![A-Za-z0-9_:])", f, "(?:<[^<>()]*>)?\\(")
                matches <-  gregexpr(pattern = pattern_of_a_call, text = body_text, perl = TRUE)[[1]]
                if (matches[[1]] < 0) next
                starts <-  as.integer(matches)
                lengths <-  attr(matches, "match.length")
                ## (from the last call to the first, so that the positions before each stay the same)
                for (k in rev(seq_along(starts))) {

                        start <-  starts[[k]]
                        open_bracket <-  start + lengths[[k]] - 1
                        close_bracket <-  fn_end_of_bracket(text = body_text, position_of_open_bracket = open_bracket)
                        call_text <-  substring(body_text, start, close_bracket)
                        what <-  gsub(pattern = "\\s+", replacement = " ", x = substr(call_text, 1, 160))
                        name_with_template <-  substring(body_text, start, open_bracket - 1)
                        arguments <-  fn_arguments_of_a_call_text(
                                              inside_text = substring(body_text, open_bracket + 1, close_bracket - 1))
                        x <-  arguments[[1]]
                        is_all_the_rows <-  (length(arguments) >= 3) &&
                                            grepl(pattern = "^[A-Za-z_][A-Za-z0-9_]*$", x = x) &&
                                            (arguments[[2]] == "1") &&
                                            grepl(pattern = paste0("^stan::math::size\\(\\s*", x, "\\s*\\)$"),
                                                  x = gsub(pattern = "\\s+", replacement = "", x = arguments[[3]]))
                        if (!is_all_the_rows) {
                                report <-  c(report, paste0("left as stanc wrote it: ", what,
                                                            " (not a call on all the rows of its sliced argument)"))
                                next
                        }
                        ##
                        ## ---- a partial-sum function that reads NicoStan's nuisance block: each tile is one chunk of it -------------
                        ##
                        reads_the_nuisance_block <-  any(grepl(pattern = "^nicostan_", x = arguments))
                        name_of_the_sum <-  paste0("sum_of_rows_in_the_chunks_of_reduce_sum_static",
                                                   if (reads_the_nuisance_block) "_reading_the_nuisance_block"
                                                   else "")
                        ##
                        ## ---- the tile's call: its rows of x, their first and last row, the shared arguments ------------------------
                        ##
                        rest <-  arguments[-(1:3)]
                        shared <-  character(0)
                        arguments_in_the_tile <-  character(0)
                        for (j in seq_along(rest)) {

                                argument <-  rest[[j]]
                                is_name <-  grepl(pattern = "^[A-Za-z_][A-Za-z0-9_]*$", x = argument)
                                is_number <-  grepl(pattern = "^-?[0-9.]+$", x = argument)
                                if (is_number || (argument == "pstream__") || (is_name && (argument %in% data_members))) {
                                        arguments_in_the_tile <-  c(arguments_in_the_tile, argument)
                                        next
                                }
                                lambda_parameter <-  if (is_name) paste0(argument, "_of_the_tile__")
                                                     else paste0("argument_", j, "_of_the_tile__")
                                shared <-  c(shared, setNames(argument, lambda_parameter))
                                arguments_in_the_tile <-  c(arguments_in_the_tile, lambda_parameter)

                        }
                        lambda_parameters <-  paste(c("const int first__", "const int n_rows__",
                                                      "const auto& rows_of_the_tile__",
                                                      paste0("const auto& ", names(shared))), collapse = ", ")
                        call_of_the_tile <-  paste0(name_with_template, "(rows_of_the_tile__, first__ + 1, ",
                                                    "first__ + n_rows__, ",
                                                    paste(arguments_in_the_tile, collapse = ", "), ")")
                        replacement <-  paste0("nico::math::auto_chunking::", name_of_the_sum, "_or_as_stanc_wrote_it(",
                                               "stan::math::size(", x, "), ", N_chunks_of_auto_chunking_text,
                                               ", [&]() { return ", call_text, "; }, [&](", lambda_parameters,
                                               ") { return ", call_of_the_tile, "; }, ",
                                               paste(c(paste0("nico::math::auto_chunking::sliced_array_at_rows_of(",
                                                              x, ")"),
                                                       unname(shared)), collapse = ", "), ")")
                        ## (a shared argument that reads NicoStan's nuisance block: read only with more than one
                        ##  chunk, as one chunk runs stanc's own call, which reads the block itself)
                        if (any(grepl(pattern = "nicostan_nuisance_", x = shared, fixed = TRUE))) {
                                replacement <-  paste0("[&]() -> local_scalar_t__ { ",
                                                       "if (nico::math::auto_chunking::chunk_size_of_auto_chunking(",
                                                       "stan::math::size(", x, "), ", N_chunks_of_auto_chunking_text,
                                                       ") >= stan::math::size(", x, ")) { return ", call_text, "; } ",
                                                       "return nico::math::auto_chunking::", name_of_the_sum, "(",
                                                       "stan::math::size(", x, "), ", N_chunks_of_auto_chunking_text,
                                                       ", [&](", lambda_parameters,
                                                       ") { return ", call_of_the_tile, "; }, ",
                                                       paste(c(paste0("nico::math::auto_chunking::",
                                                                      "sliced_array_at_rows_of(", x, ")"),
                                                               unname(shared)), collapse = ", "), "); }()")
                        }
                        body_text <-  paste0(substring(body_text, 1, start - 1), replacement,
                                             substring(body_text, close_bracket + 1, nchar(body_text)))
                        report <-  c(report, paste0("tiled over stan::math::size(", x, ") rows: the partial-sum ",
                                                    "function ", f, " on all the rows of ", x, " (shared: ",
                                                    if (length(shared) > 0) paste(unname(shared), collapse = ", ")
                                                    else "none",
                                                    "; data read in place; one chunk: stanc's own call",
                                                    if (reads_the_nuisance_block)
                                                            "; each tile one chunk of NicoStan's nuisance block"
                                                    else "", ")"))

                }

        }
        return(list(body_text = body_text, report = report))

}
##
## ---- A C++ text with the inside of each string blanked, its length kept -----------------------------------------------------------
##
fn_cpp_text_without_strings <-  function(text) {

        strings <-  gregexpr(pattern = "\"([^\"\\\\]|\\\\.)*\"", text = text, perl = TRUE)[[1]]
        if (strings[[1]] < 0) return(text)
        lengths_of_strings <-  attr(strings, "match.length")
        regmatches(x = text, m = list(strings)) <-  list(strrep(" ", lengths_of_strings))
        return(text)

}
##
## ---- The matching closing brace of each opening brace of a C++ text (past its strings) ----------------------------------------------
##
fn_braces_of_cpp_text <-  function(text) {

        ## (the strings blanked, so that no brace in a string is counted)
        without_strings <-  fn_cpp_text_without_strings(text = text)
        stopifnot(nchar(without_strings) == nchar(text))
        positions <-  gregexpr(pattern = "[{}]", text = without_strings)[[1]]
        if (positions[[1]] < 0) return(data.frame(open = integer(0), close = integer(0)))
        kinds <-  substring(without_strings, positions, positions)
        opens <-  integer(0)
        closes <-  integer(0)
        stack <-  integer(0)
        for (i in seq_along(positions)) {

                if (kinds[[i]] == "{") {
                        stack <-  c(stack, positions[[i]])
                } else {
                        if (length(stack) == 0) stop("a closing brace with no opening brace")
                        opens <-  c(opens, stack[[length(stack)]])
                        closes <-  c(closes, positions[[i]])
                        stack <-  stack[-length(stack)]
                }

        }
        if (length(stack) > 0) stop("an opening brace with no closing brace")
        return(data.frame(open = opens, close = closes))

}
##
## ---- The names a C++ text declares (stanc's declarations of locals, loop indices and parameters) ----------------------------------
##
fn_names_declared_in_cpp_text <-  function(text) {

        pattern <-  paste0("(?:\\bint|\\bdouble|\\bauto|\\blocal_scalar_t__|>)\\s+([A-Za-z_][A-Za-z0-9_]*)",
                           "\\s*(?:=|;|\\{)")
        declared <-  regmatches(x = text, m = gregexpr(pattern = pattern, text = text, perl = TRUE))[[1]]
        return(unique(sub(pattern = pattern, replacement = "\\1", x = declared, perl = TRUE)))

}
##
## ---- The names a C++ text reads: its identifiers, past strings, namespaces and members ------------------------------------------
##
fn_names_read_in_cpp_text <-  function(text) {

        without_strings <-  gsub(pattern = "\"([^\"\\\\]|\\\\.)*\"", replacement = "\"\"", x = text, perl = TRUE)
        pattern <-  "(?<![A-Za-z0-9_:.>])([A-Za-z_][A-Za-z0-9_]*)\\b(?!\\s*::)"
        read <-  regmatches(x = without_strings,
                            m = gregexpr(pattern = pattern, text = without_strings, perl = TRUE))[[1]]
        return(unique(read))

}
##
## ---- The data members of the model and their C++ types --------------------------------------------------------------------------
##
fn_types_of_the_data_members_of_generated_cpp <-  function(text) {

        start <-  regexpr(pattern = "\n private:\n", text = text, fixed = TRUE)
        end <-  regexpr(pattern = ": model_base_crtp(0) {", text = text, fixed = TRUE)
        if ((start < 0) || (end < start)) stop("no private section before the model constructor")
        section <-  substring(text, start, end)
        pattern <-  "(?m)^  ([^\n;(]*?)\\s+([A-Za-z_][A-Za-z0-9_]*)\\s*(\\{[^;\n]*\\})?;$"
        declarations <-  regmatches(x = section,
                                    m = gregexpr(pattern = pattern, text = section, perl = TRUE))[[1]]
        types <-  sub(pattern = pattern, replacement = "\\1", x = declarations, perl = TRUE)
        names(types) <-  sub(pattern = pattern, replacement = "\\2", x = declarations, perl = TRUE)
        return(types)

}
##
## ---- The type and first dimension of a var declared before a block of log_prob ------------------------------------------------------
##
## (a parameter read from in__ (its type in read<...>, its dimensions as the read's arguments), or a local of
##  local_scalar_t__ (its type before its name, its first dimension the first argument of its constructor))
fn_type_and_first_dimension_of_a_var <-  function(text_before_the_block,
                                                  name) {

        read <-  regexpr(pattern = paste0("auto\\s+", name, "\\s*=\\s*in__\\.template\\s+read[A-Za-z_]*<"),
                         text = text_before_the_block, perl = TRUE)
        if (read > 0) {

                opening <-  read + attr(read, "match.length") - 1L
                characters <-  strsplit(x = substring(text_before_the_block, opening, opening + 400L),
                                        split = "")[[1]]
                depth <-  0L
                closing <-  NA_integer_
                for (i in seq_along(characters)) {
                        if (characters[[i]] == "<") depth <-  depth + 1L
                        if (characters[[i]] == ">") depth <-  depth - 1L
                        if (depth == 0L) {
                                closing <-  i
                                break
                        }
                }
                if (is.na(closing)) return(NULL)
                type <-  fn_arguments_of_a_call_text(inside_text = paste(characters[2:(closing - 1L)],
                                                                          collapse = ""))[[1]]
                after <-  substring(text_before_the_block, opening + closing, nchar(text_before_the_block))
                arguments_open <-  regexpr(pattern = "\\(", text = after)[[1]]
                arguments <-  fn_inside_of_the_bracket(text = after,
                                                       position_of_open_bracket = arguments_open)$inside
                arguments <-  fn_arguments_of_a_call_text(inside_text = arguments)
                ## (a constrained read gives its bounds first: the dimensions are the last arguments, one per
                ##  std::vector and one or two for an Eigen type)
                n_dimensions <-  lengths(regmatches(type, gregexpr("std::vector<", type, fixed = TRUE))) +
                        if (grepl("Eigen::Matrix<", type, fixed = TRUE))
                                (if (grepl(",-1,-1>", gsub("\\s+", "", type), fixed = TRUE)) 2L else 1L) else 0L
                first_dimension <-  if ((n_dimensions > 0L) && (length(arguments) >= n_dimensions))
                                            arguments[[length(arguments) - n_dimensions + 1L]] else NA_character_
                return(list(type = gsub("\\s+", "", type), first_dimension = first_dimension))

        }
        pattern <-  paste0("((?:std::vector<|Eigen::Matrix<)[^;=]*?local_scalar_t__[^;=]*?>|",
                           "local_scalar_t__)\\s+",
                           name, "\\s*=\\s*([^;]*);")
        declaration <-  regmatches(x = text_before_the_block,
                                   m = regexec(pattern = pattern, text = text_before_the_block, perl = TRUE))[[1]]
        if (length(declaration) != 3) return(NULL)
        type <-  gsub("\\s+", "", declaration[[2]])
        constructor <-  declaration[[3]]
        opening <-  regexpr(pattern = "\\(", text = constructor)[[1]]
        first_dimension <-  NA_character_
        if ((opening > 0) && (type != "local_scalar_t__")) {
                arguments <-  fn_arguments_of_a_call_text(
                        inside_text = fn_inside_of_the_bracket(text = constructor,
                                                               position_of_open_bracket = opening)$inside)
                if (length(arguments) > 0) first_dimension <-  arguments[[1]]
        }
        return(list(type = type, first_dimension = first_dimension))

}
##
## ---- Each block of the rows 1 .. size(x) of a row array x: run on each chunk of rows as its own statements ------------------------
##
## (a block of stanc's log_prob (a { ... } of the model) whose rows are the rows 1 .. size(x) of a row array x,
##  read as x[1:size(x)] of the data, and whose only effect outside it is to add to the target: each tile runs
##  the block's statements as stanc wrote them, with x the tile's copy of its own rows, every data object with N
##  entries in a dimension read at the tile's rows, and every other var copied into the tile's nested tape; its
##  first call is checked against the block on all its rows, and a block whose chunks do not add up to it is run
##  as stanc wrote it from then on (runtime: nico_auto_chunking_blocks_of_rows.hpp); the loops over groups of
##  rows tiled before it share its check)
fn_auto_chunk_the_blocks_of_rows_in_one_log_prob_body <-  function(body_text,
                                                                  text,
                                                                  N_chunks_of_auto_chunking_text,
                                                                  names_of_the_states_of_the_loops) {

        report <-  character(0)
        names_of_the_states <-  character(0)
        pattern_of_a_row_slice <-  "index_min_max\\(1,\\s*stan::math::size\\(([A-Za-z_][A-Za-z0-9_]*)\\)\\)"
        slices <-  regmatches(x = body_text,
                              m = gregexpr(pattern = pattern_of_a_row_slice, text = body_text))[[1]]
        row_arrays <-  unique(sub(pattern = pattern_of_a_row_slice, replacement = "\\1", x = slices))
        if ((length(row_arrays) == 0) && (length(names_of_the_states_of_the_loops) == 0)) {
                return(list(body_text = body_text, report = report))
        }
        types_of_the_data <-  fn_types_of_the_data_members_of_generated_cpp(text = text)
        containers_of_the_data <-  names(types_of_the_data)[grepl(pattern = "^(std::vector<|Eigen::)",
                                                                  x = types_of_the_data)]
        containers_of_the_data <-  containers_of_the_data[!grepl(pattern = "_data__$",
                                                                 x = containers_of_the_data)]
        ##
        ## ---- Each row array, its block found again in the text as rewritten so far -------------------------------------------------
        ##
        for (k in seq_along(row_arrays)) {

                row_array <-  row_arrays[[k]]
                size_text <-  paste0("stan::math::size(", row_array, ")")
                uses <-  gregexpr(pattern = size_text, text = body_text, fixed = TRUE)[[1]]
                braces <-  fn_braces_of_cpp_text(text = body_text)
                around <-  braces[(braces$open < min(uses)) & (braces$close > max(uses)), ]
                ## (the innermost block holding every use, never the body of log_prob itself)
                around <-  around[around$open > min(braces$open), ]
                if (nrow(around) == 0) {
                        report <-  c(report, paste0("block of the rows of ", row_array,
                                                    ": not tiled (its uses are not ",
                                                    "inside one block of the model)"))
                        next
                }
                open <-  max(around$open)
                close <-  around$close[around$open == open]
                block <-  substring(body_text, open + 1L, close - 1L)
                before <-  substring(body_text, 1L, open)
                ##
                ## ---- What the block must be: an addition to the target, and nothing else outside it ------------------------------------
                ##
                why_not <-  character(0)
                if (!grepl("lp_accum__.add(", block, fixed = TRUE)) {
                        why_not <-  c(why_not, "it adds nothing to the target")
                }
                if (grepl("\\blp__\\b", block, perl = TRUE)) why_not <-  c(why_not, "it reads target()")
                if (grepl("auto_chunking::", block, fixed = TRUE)) {
                        why_not <-  c(why_not, "it holds a tile already")
                }
                if (grepl("in__.", block, fixed = TRUE)) why_not <-  c(why_not, "it reads the parameters")
                if (grepl("\\breturn\\b", block, perl = TRUE)) why_not <-  c(why_not, "it returns")
                declared_inside <-  fn_names_declared_in_cpp_text(text = block)
                declared_before <-  fn_names_declared_in_cpp_text(text = before)
                assigned <-  c(regmatches(x = block,
                                          m = gregexpr(pattern = paste0("stan::model::assign\\(\\s*",
                                                                        "[A-Za-z_][A-Za-z0-9_]*"),
                                                       text = block, perl = TRUE))[[1]],
                               regmatches(x = block,
                                          m = gregexpr(pattern = paste0("(?m)^\\s*[A-Za-z_][A-Za-z0-9_]*",
                                                                        "\\s*(?:\\+|-|\\*|/)?=(?!=)"),
                                                       text = block, perl = TRUE))[[1]])
                assigned <-  unique(sub(pattern = "^.*?([A-Za-z_][A-Za-z0-9_]*)\\s*(?:\\+|-|\\*|/)?=?$",
                                        replacement = "\\1",
                                        x = sub(pattern = "^stan::model::assign\\(\\s*", replacement = "",
                                                x = assigned),
                                        perl = TRUE))
                assigned_outside <-  setdiff(intersect(assigned, c(declared_before, names(types_of_the_data))),
                                             c(declared_inside, "current_statement__"))
                if (length(assigned_outside) > 0) {
                        why_not <-  c(why_not, paste0("it assigns ", paste(assigned_outside, collapse = ", "),
                                                      ", declared outside it"))
                }
                if (length(why_not) > 0) {
                        report <-  c(report, paste0("block of the rows of ", row_array, ": not tiled (",
                                                    paste(why_not, collapse = "; "), ")"))
                        next
                }
                ##
                ## ---- The data it reads, and the vars declared before it: its rows, or copied whole into each tile ----------------------
                ##
                read <-  setdiff(fn_names_read_in_cpp_text(text = block), declared_inside)
                data_read <-  intersect(read, containers_of_the_data)
                vars_read <-  character(0)
                kinds_of_vars <-  character(0)
                row_array_info <-  NULL
                if (!(row_array %in% names(types_of_the_data))) {
                        row_array_info <-  fn_type_and_first_dimension_of_a_var(text_before_the_block = before,
                                                                                name = row_array)
                        if (is.null(row_array_info)) {
                                report <-  c(report, paste0("block of the rows of ", row_array, ": not tiled (",
                                                            row_array, " is neither data nor a var ",
                                                            "declared before ",
                                                            "it)"))
                                next
                        }
                }
                kind_of_a_row_var <-  function(info) {

                        if (startsWith(info$type, "std::vector<")) return("sliced_array")
                        if (grepl("^Eigen::Matrix<local_scalar_t__,-1,1>$", info$type)) return("sliced_vector")
                        return(NA_character_)

                }
                unsupported <-  character(0)
                for (name in setdiff(intersect(read, declared_before), c(names(types_of_the_data), "DUMMY_VAR__",
                                                                          "current_statement__", "lp_accum__",
                                                                          "pstream__"))) {

                        info <-  fn_type_and_first_dimension_of_a_var(text_before_the_block = before, name = name)
                        if (is.null(info)) next
                        is_a_row_var <-  (name == row_array) ||
                                         (!is.null(row_array_info) && !is.na(info$first_dimension) &&
                                          !is.na(row_array_info$first_dimension) &&
                                          (info$first_dimension == row_array_info$first_dimension))
                        kind <-  if (is_a_row_var) kind_of_a_row_var(info = info) else "whole"
                        if (is.na(kind)) unsupported <-  c(unsupported, name)
                        vars_read <-  c(vars_read, name)
                        kinds_of_vars <-  c(kinds_of_vars, kind)

                }
                if (length(unsupported) > 0) {
                        report <-  c(report, paste0("block of the rows of ", row_array,
                                                    ": not tiled (its row vars ",
                                                    paste(unsupported, collapse = ", "),
                                                    " are neither arrays nor ",
                                                    "vectors)"))
                        next
                }
                if (length(vars_read) == 0) {
                        report <-  c(report, paste0("block of the rows of ", row_array,
                                                    ": not tiled (it reads no ",
                                                    "var)"))
                        next
                }
                ##
                ## ---- The tile: the block's statements as stanc wrote them, on the tile's rows ------------------------------------------
                ##
                rows_name <-  paste0("N_rows_of_block_", k, "__")
                state_name <-  paste0("state_of_block_", k, "__")
                rebinding <-  unlist(lapply(data_read, function(name) {

                        c(paste0("const auto ", name, "_at_the_rows_of_the_tile__ = ",
                                 "nico::math::auto_chunking::data_at_the_rows_of_the_tile(this->", name, ", ",
                                 rows_name, ", first__, n_rows__);"),
                          paste0("const auto& ", name, " = nico::math::auto_chunking::the_data(", name,
                                 "_at_the_rows_of_the_tile__);"))

                }))
                ## (non-const references, as stanc's statement reads its own locals: the block's functions are the
                ##  ones stanc's statement calls)
                arguments_of_the_tile <-  paste0("auto", "& ", vars_read)
                shared <-  ifelse(kinds_of_vars == "sliced_array",
                                  paste0("nico::math::auto_chunking::rows_of_the_array_of_a_block_of(",
                                         vars_read, ")"),
                                  ifelse(kinds_of_vars == "sliced_vector",
                                         paste0("nico::math::auto_chunking::sliced_at_rows_as_vars_of(",
                                                vars_read,
                                                ")"),
                                         vars_read))
                name_of_the_block <-  paste0("the block of the rows 1 .. size(", row_array, ")")
                tiled <-  paste0(
                        "\n{\n",
                        "const int ", rows_name, " = static_cast<int>(", size_text, ");\n",
                        "lp_accum__.add(nico::math::auto_chunking::sum_of_a_block_of_rows_in_the_chunks_of_",
                        "reduce_sum_static(\n",
                        "\"", name_of_the_block, "\", ", rows_name, ", ", N_chunks_of_auto_chunking_text, ", ",
                        state_name, ",\n",
                        "[&](const int first__, const int n_rows__, ",
                        paste(arguments_of_the_tile, collapse = ", "),
                        ## (its body compiled once, not inlined at each place the runtime calls it)
                        ") __attribute__((noinline)) {\n",
                        "stan::math::accumulator<local_scalar_t__> lp_accum__;\n",
                        paste0(rebinding, collapse = "\n"), if (length(rebinding) > 0) "\n" else "",
                        "{\n", block, "\n}\n",
                        "return lp_accum__.sum();\n",
                        "},\n",
                        paste(shared, collapse = ", "), "));\n",
                        "}\n")
                body_text <-  paste0(substring(body_text, 1L, open), tiled,
                                     substring(body_text, close, nchar(body_text)))
                names_of_the_states <-  c(names_of_the_states, state_name)
                report <-  c(report, paste0(name_of_the_block, ": tiled over the rows of ", row_array,
                                            " (its own statements; data at the tile's rows: ",
                                            paste(data_read, collapse = ", "), "; its rows: ",
                                            paste(vars_read[kinds_of_vars != "whole"], collapse = ", "),
                                            "; copied whole into each tile: ",
                                            paste(vars_read[kinds_of_vars == "whole"], collapse = ", "), ")"))

        }
        names_of_the_states <-  c(names_of_the_states_of_the_loops, names_of_the_states)
        if (length(names_of_the_states) == 0) return(list(body_text = body_text, report = report))
        ##
        ## ---- The state of each block, at the top of log_prob, and the check of the first call with chunks at its end -------------
        ##
        ## (against stanc's own log_prob of the values, with no tape; if they differ, every block is run as
        ##  stanc wrote it from then on, and the call is made again)
        accumulator <-  "stan::math::accumulator<T__> lp_accum__;"
        at_the_top <-  regexpr(pattern = accumulator, text = body_text, fixed = TRUE)[[1]]
        ending <-  "return lp_accum__.sum();"
        at_the_end <-  max(gregexpr(pattern = ending, text = body_text, fixed = TRUE)[[1]])
        if ((at_the_top < 0) || (at_the_end < 0) || (at_the_end < at_the_top)) {
                stop("log_prob has no accumulator before its return: its blocks cannot be checked")
        }
        ## (each block's state is the model's, one for every instantiation of log_prob (propto, jacobian), so that
        ##  one check settles them all; the reference is stanc's log_prob<double> with every term (propto =
        ##  false), against the chunks' log density with every term: a call with propto = true evaluates it once,
        ##  on a nested tape freed at once, with no check of its own)
        states <-  paste(names_of_the_states, collapse = ", ")
        names_of_the_model_states <-  paste0(sub("__$", "", names_of_the_states), "_of_the_model__")
        declarations_before_the_class <-  paste0(
                "inline std::atomic<int>& ", names_of_the_model_states, "() {\n",
                "  static std::atomic<int> state{0};\n",
                "  return state;\n",
                "}\n", collapse = "")
        declarations <-  paste0("\n    std::atomic<int>& ", names_of_the_states, " = ",
                                names_of_the_model_states,
                                "();", collapse = "")
        check <-  paste0(
                "{\n",
                "auto lp_of_the_chunks__ = lp_accum__.sum();\n",
                "if (!nico::math::auto_chunking::the_chunks_of_the_blocks_are_being_checked\n",
                "    && nico::math::auto_chunking::a_block_of_rows_is_not_yet_checked(", states, ")) {\n",
                "auto values_of_the_parameters__ = stan::math::eval(stan::math::value_of(params_r__));\n",
                "const double lp_of_stanc__ = this->template log_prob_impl<false, jacobian__>(",
                "values_of_the_parameters__, params_i__, pstream__);\n",
                "double lp_of_the_chunks_with_every_term__ = stan::math::value_of(lp_of_the_chunks__);\n",
                "if constexpr (propto__) {\n",
                "const stan::math::nested_rev_autodiff nested__;\n",
                "const nico::math::auto_chunking::check_of_the_chunks_of_the_blocks check__;\n",
                "lp_of_the_chunks_with_every_term__ = stan::math::value_of(",
                "this->template log_prob_impl<false, jacobian__>(params_r__, params_i__, pstream__));\n",
                "}\n",
                "const int state__ = nico::math::auto_chunking::state_after_the_check(",
                "lp_of_the_chunks_with_every_term__, lp_of_stanc__);\n",
                "nico::math::auto_chunking::record_the_check_of_the_blocks_of_rows(state__, ", states, ");\n",
                "if (state__ == 2) {\n",
                "return this->template log_prob_impl<propto__, jacobian__>(params_r__, params_i__, pstream__);\n",
                "}\n",
                "}\n",
                "return lp_of_the_chunks__;\n",
                "}")
        body_text <-  paste0(substring(body_text, 1L, at_the_end - 1L), check,
                             substring(body_text, at_the_end + nchar(ending), nchar(body_text)))
        body_text <-  paste0(substring(body_text, 1L, at_the_top + nchar(accumulator) - 1L), declarations,
                             substring(body_text, at_the_top + nchar(accumulator), nchar(body_text)))
        return(list(body_text = body_text, report = report,
                    declarations_before_the_class = declarations_before_the_class))

}
##
## ---- Whether a loop reads a data object only in blocks of N entries, x[lower:upper] with both bounds made of N ----------------
##
## (e.g. y_by_item[((j - 1) * N + 1):(j * N)]: the rows of one group after another, each N long)
fn_data_read_only_in_blocks_of_N_entries <-  function(loop_text,
                                                      name,
                                                      N_text) {

        without_strings <-  fn_cpp_text_without_strings(text = loop_text)
        uses <-  gregexpr(pattern = paste0("(?<![A-Za-z0-9_:.>])", name, "\\b"), text = without_strings,
                          perl = TRUE)[[1]]
        if (uses[[1]] < 0) return(FALSE)
        head_of_a_read <-  "stan::model::rvalue("
        head_of_a_range <-  "stan::model::index_min_max("
        for (at in as.integer(uses)) {

                before_the_name <-  substring(without_strings, at - nchar(head_of_a_read), at - 1L)
                if (before_the_name != head_of_a_read) return(FALSE)
                arguments <-  fn_arguments_of_a_call_text(
                        inside_text = fn_inside_of_the_bracket(text = loop_text,
                                                               position_of_open_bracket = at - 1L)$inside)
                if (length(arguments) != 3) return(FALSE)
                index <-  trimws(arguments[[3]])
                if (!startsWith(index, head_of_a_range)) return(FALSE)
                open_of_the_range <-  nchar(head_of_a_range)
                inside_of_the_range <-  fn_inside_of_the_bracket(text = index,
                                                                 position_of_open_bracket = open_of_the_range)
                bounds <-  fn_arguments_of_a_call_text(inside_text = inside_of_the_range$inside)
                if (length(bounds) != 2) return(FALSE)
                for (bound in bounds) {
                        if (!(N_text %in% fn_names_read_in_cpp_text(text = bound))) return(FALSE)
                }

        }
        return(TRUE)

}
##
## ---- Each loop over groups of rows (items, classes, ...): all its groups run on each chunk of rows, as its own statements --------
##
## (a for loop of stanc's reverse-mode log_prob whose bounds read no rows, whose only effect outside it is to add
##  to the target, and which reads vars of N rows, N an integer of the data that its bounds do not read (e.g. a
##  loop over the items, each a sum over the N persons): each tile runs the whole loop as stanc wrote it, with N
##  the tile's number of rows, each var of N rows its tile's rows, each data object read only in blocks of N
##  entries (the rows of one group after another) the tile's rows of each block, every other data object with N
##  entries in a dimension at the tile's rows, a nuisance block NicoStan reads whole its tile's rows, and every
##  other var (fixed in size, or of the loop's own size) copied into the tile's nested tape: every group of a
##  chunk of rows in one tile, as in reduce_sum_static's partial sums; checked with the blocks of rows (runtime:
##  nico_auto_chunking_blocks_of_rows.hpp))
fn_auto_chunk_the_loops_over_groups_of_rows_in_one_log_prob_body <-  function(body_text,
                                                                             text,
                                                                             data,
                                                                             N_chunks_of_auto_chunking_text) {

        report <-  character(0)
        none <-  list(body_text = body_text, report = report, names_of_the_states = character(0),
                      tiled_loops = character(0), placeholders = character(0))
        without_strings <-  fn_cpp_text_without_strings(text = body_text)
        pattern_of_a_loop <-  paste0("for \\(int ([A-Za-z_][A-Za-z0-9_]*) = ([^;]+); \\1 <= ([^;]+); ",
                                     "\\+\\+\\1\\) \\{")
        matches <-  gregexpr(pattern = pattern_of_a_loop, text = without_strings, perl = TRUE)
        if (matches[[1]][[1]] < 0) return(none)
        headers <-  regmatches(x = without_strings, m = matches)[[1]]
        types_of_the_data <-  fn_types_of_the_data_members_of_generated_cpp(text = text)
        containers_of_the_data <-  names(types_of_the_data)[grepl(pattern = "^(std::vector<|Eigen::)",
                                                                  x = types_of_the_data)]
        containers_of_the_data <-  containers_of_the_data[!grepl(pattern = "_data__$",
                                                                 x = containers_of_the_data)]
        ## (a number of rows is an integer of the Stan model's own data, e.g. N; never one of stanc's dimensions
        ##  of a parameter (x_1dim__), which are the sizes of small parameters)
        scalars_of_the_Stan_data <-  names(data)[vapply(data, function(entry) identical(entry$shape, "scalar"),
                                                        logical(1))]
        integers_of_the_data <-  names(types_of_the_data)[gsub("\\s+", "", types_of_the_data) == "int"]
        integers_of_the_data <-  intersect(integers_of_the_data, scalars_of_the_Stan_data)
        integers_of_the_data <-  integers_of_the_data[!grepl(pattern = "__$", x = integers_of_the_data)]
        braces <-  fn_braces_of_cpp_text(text = body_text)
        starts <-  as.integer(matches[[1]])
        opens <-  starts + attr(matches[[1]], "match.length") - 1L
        closes <-  braces$close[match(opens, braces$open)]
        ##
        ## ---- Each loop: what it must be, its number of rows, and how each var and data object goes into a tile ---------------------
        ##
        candidates <-  list()
        for (i in seq_along(starts)) {

                header <-  regmatches(x = headers[[i]],
                                      m = regexec(pattern = pattern_of_a_loop, text = headers[[i]],
                                                  perl = TRUE))[[1]]
                index <-  header[[2]]
                names_in_the_bounds <-  fn_names_read_in_cpp_text(text = paste(header[[3]], header[[4]]))
                loop_text <-  substring(body_text, starts[[i]], closes[[i]])
                before <-  substring(body_text, 1L, starts[[i]] - 1L)
                where <-  paste0("the loop over ", index, " at character ", starts[[i]])
                why_not <-  character(0)
                if (!grepl("lp_accum__.add(", loop_text, fixed = TRUE)) {
                        why_not <-  c(why_not, "it adds nothing to the target")
                }
                if (grepl("\\blp__\\b", loop_text, perl = TRUE)) why_not <-  c(why_not, "it reads target()")
                if (grepl("auto_chunking::", loop_text, fixed = TRUE)) {
                        why_not <-  c(why_not, "it holds a tile already")
                }
                if (grepl("in__.", loop_text, fixed = TRUE)) why_not <-  c(why_not, "it reads the parameters")
                if (grepl("\\breturn\\b", loop_text, perl = TRUE)) why_not <-  c(why_not, "it returns")
                declared_inside <-  c(index, fn_names_declared_in_cpp_text(text = loop_text))
                declared_before <-  fn_names_declared_in_cpp_text(text = before)
                assigned <-  c(regmatches(x = loop_text,
                                          m = gregexpr(pattern = paste0("stan::model::assign\\(\\s*",
                                                                        "[A-Za-z_][A-Za-z0-9_]*"),
                                                       text = loop_text, perl = TRUE))[[1]],
                               regmatches(x = loop_text,
                                          m = gregexpr(pattern = paste0("(?m)^\\s*[A-Za-z_][A-Za-z0-9_]*",
                                                                        "\\s*(?:\\+|-|\\*|/)?=(?!=)"),
                                                       text = loop_text, perl = TRUE))[[1]])
                assigned <-  unique(sub(pattern = "^.*?([A-Za-z_][A-Za-z0-9_]*)\\s*(?:\\+|-|\\*|/)?=?$",
                                        replacement = "\\1",
                                        x = sub(pattern = "^stan::model::assign\\(\\s*", replacement = "",
                                                x = assigned),
                                        perl = TRUE))
                assigned_outside <-  setdiff(intersect(assigned, c(declared_before, names(types_of_the_data))),
                                             c(declared_inside, "current_statement__"))
                if (length(assigned_outside) > 0) {
                        why_not <-  c(why_not, paste0("it assigns ", paste(assigned_outside, collapse = ", "),
                                                      ", declared outside it"))
                }
                ## (the vars declared before the loop that it reads, each with its type and first dimension; an
                ##  integer or a double declared before it is read as it is)
                read <-  setdiff(fn_names_read_in_cpp_text(text = loop_text), declared_inside)
                infos <-  list()
                unknown <-  character(0)
                for (name in setdiff(intersect(read, declared_before),
                                     c(names(types_of_the_data), "DUMMY_VAR__", "current_statement__",
                                       "lp_accum__", "pstream__", "lp__"))) {

                        info <-  fn_type_and_first_dimension_of_a_var(text_before_the_block = before, name = name)
                        if (!is.null(info)) {
                                infos[[name]] <-  info
                        } else if (!grepl(pattern = paste0("(?:\\bint|\\bdouble|std::vector<int>|",
                                                           "std::vector<double>)\\s+", name, "\\s*(?:=|;|\\{)"),
                                          x = before, perl = TRUE)) {
                                unknown <-  c(unknown, name)
                        }

                }
                if (length(unknown) > 0) {
                        why_not <-  c(why_not, paste0("it reads ", paste(unknown, collapse = ", "),
                                                      ", whose type is not known"))
                }
                first_dimensions <-  unique(stats::na.omit(vapply(infos, function(info) info$first_dimension,
                                                                  character(1))))
                row_counts <-  setdiff(intersect(first_dimensions, integers_of_the_data), names_in_the_bounds)
                ## (a nuisance block NicoStan reads whole, nicostan_nuisance_array_of_reals(1, terms, offset,
                ##  type, pstream__): read once on the main tape before the tile, each tile given its own rows of
                ##  it, and the reader inside the tile returning them; its number of rows is the integer of the
                ##  data in the bounds of the loop's ranges, x[lower:upper])
                reader_name <-  "nicostan_nuisance_array_of_reals"
                reader_call <-  NULL
                loop_text_without_strings <-  fn_cpp_text_without_strings(text = loop_text)
                at_the_readers <-  gregexpr(pattern = paste0("(?<![A-Za-z0-9_:])", reader_name, "\\("),
                                            text = loop_text_without_strings, perl = TRUE)[[1]]
                if (grepl("\\bnicostan_nuisance_(?!array_of_reals\\()", loop_text_without_strings, perl = TRUE)) {
                        why_not <-  c(why_not, "it reads the nuisance block through a reader other than an array")
                }
                if (at_the_readers[[1]] > 0) {

                        calls <-  unique(vapply(as.integer(at_the_readers), function(position) {

                                open <-  position + nchar(reader_name)
                                gsub("\\s+", "", fn_inside_of_the_bracket(text = loop_text,
                                                                          position_of_open_bracket = open)$inside)

                        }, character(1)))
                        arguments <-  fn_arguments_of_a_call_text(inside_text = calls[[1]])
                        if ((length(calls) != 1) || (length(arguments) != 5) || (arguments[[1]] != "1") ||
                            (arguments[[5]] != "pstream__") || !(arguments[[2]] %in% names(types_of_the_data))) {
                                why_not <-  c(why_not,
                                              "it reads the nuisance block other than whole, in one call")
                        } else {
                                reader_call <-  paste0(reader_name, "(", paste(arguments, collapse = ", "), ")")
                                ranges <-  gregexpr(pattern = "index_min_max\\(",
                                                    text = loop_text_without_strings)[[1]]
                                names_in_the_ranges <-  unique(unlist(lapply(as.integer(ranges),
                                                                             function(position) {

                                        open <-  position + nchar("index_min_max")
                                        inside <-  fn_inside_of_the_bracket(text = loop_text,
                                                                            position_of_open_bracket = open)
                                        fn_names_read_in_cpp_text(text = inside$inside)

                                })))
                                if (ranges[[1]] < 0) names_in_the_ranges <-  character(0)
                                row_counts <-  unique(c(row_counts,
                                                        setdiff(intersect(names_in_the_ranges,
                                                                          integers_of_the_data),
                                                                names_in_the_bounds)))
                        }

                }
                row_counts <-  row_counts[!startsWith(row_counts, "nicostan_")]
                if (length(row_counts) == 0) why_not <-  c(why_not, "it reads no var of N rows")
                if (length(row_counts) > 1) {
                        why_not <-  c(why_not, paste0("its vars have more than one number of rows (",
                                                      paste(row_counts, collapse = ", "), ")"))
                }
                if (length(why_not) > 0) {
                        report <-  c(report, paste0(where, ": not tiled (", paste(why_not, collapse = "; "), ")"))
                        next
                }
                N_text <-  row_counts[[1]]
                kinds <-  vapply(names(infos), function(name) {

                        info <-  infos[[name]]
                        if (!is.na(info$first_dimension) && (info$first_dimension == N_text)) {
                                if (startsWith(info$type, "std::vector<")) return("sliced_array")
                                if (grepl("^Eigen::Matrix<local_scalar_t__,-1,1>$", info$type)) {
                                        return("sliced_vector")
                                }
                                return("rows of an unknown type")
                        }
                        ## (copied whole into each tile: fixed in size, or of the loop's own size)
                        if (is.na(info$first_dimension) || grepl("^[0-9]+$", info$first_dimension) ||
                            (info$first_dimension %in% names_in_the_bounds)) return("whole")
                        return(paste0("whole, of size ", info$first_dimension))

                }, character(1))
                not_into_a_tile <-  !(kinds %in% c("sliced_array", "sliced_vector", "whole"))
                if (any(not_into_a_tile)) {
                        report <-  c(report, paste0(where, ": not tiled (it reads ",
                                                    paste0(names(kinds)[not_into_a_tile], " (",
                                                           kinds[not_into_a_tile], ")", collapse = ", "), ")"))
                        next
                }
                data_read <-  intersect(read, containers_of_the_data)
                in_blocks <-  vapply(data_read, function(name) {

                        fn_data_read_only_in_blocks_of_N_entries(loop_text = loop_text,
                                                                 name = name,
                                                                 N_text = N_text)

                }, logical(1))
                candidates[[length(candidates) + 1L]] <-  list(start = starts[[i]], close = closes[[i]],
                                                               index = index, loop_text = loop_text,
                                                               N_text = N_text, kinds = kinds,
                                                               data_read = data_read, in_blocks = in_blocks,
                                                               reader_call = reader_call)

        }
        if (length(candidates) == 0) {
                none$report <-  report
                return(none)
        }
        ##
        ## ---- The outermost loops only (a loop inside a tiled loop is in its tile), each tiled --------------------------------------
        ##
        spans <-  t(vapply(candidates, function(candidate) c(candidate$start, candidate$close), numeric(2)))
        outermost <-  vapply(seq_along(candidates), function(i) {

                !any((spans[-i, 1] < spans[i, 1]) & (spans[-i, 2] > spans[i, 2]))

        }, logical(1))
        candidates <-  candidates[outermost]
        names_of_the_states <-  character(0)
        tiled_loops <-  character(0)
        placeholders <-  character(0)
        for (k in seq_along(candidates)) {

                candidate <-  candidates[[k]]
                rows_name <-  paste0("N_rows_of_loop_", k, "__")
                state_name <-  paste0("state_of_loop_", k, "__")
                rebinding <-  unlist(lapply(candidate$data_read, function(name) {

                        reader <-  if (candidate$in_blocks[[name]]) "data_in_blocks_at_the_rows_of_the_tile" else
                                           "data_at_the_rows_of_the_tile"
                        c(paste0("const auto ", name, "_at_the_rows_of_the_tile__ = ",
                                 "nico::math::auto_chunking::", reader, "(this->", name, ", ",
                                 rows_name, ", first__, n_rows__);"),
                          paste0("const auto& ", name, " = nico::math::auto_chunking::the_data(", name,
                                 "_at_the_rows_of_the_tile__);"))

                }))
                vars_read <-  names(candidate$kinds)
                kinds_of_the_tile <-  candidate$kinds
                nuisance_rows_name <-  paste0("nicostan_rows_of_the_nuisance_block_of_loop_", k, "__")
                read_before_the_tile <-  ""
                reader_in_the_tile <-  ""
                if (!is.null(candidate$reader_call)) {
                        read_before_the_tile <-  paste0("auto ", nuisance_rows_name, " = ", candidate$reader_call,
                                                        ";\n")
                        reader_in_the_tile <-  paste0("const auto nicostan_nuisance_array_of_reals = ",
                                                      "[&](const auto&...) { return ", nuisance_rows_name,
                                                      "; };\n")
                        vars_read <-  c(vars_read, nuisance_rows_name)
                        kinds_of_the_tile <-  c(kinds_of_the_tile, "sliced_array")
                }
                ## (non-const references, as stanc's statements read their own locals)
                arguments_of_the_tile <-  paste0("auto", "& ", vars_read)
                shared <-  ifelse(kinds_of_the_tile == "sliced_array",
                                  paste0("nico::math::auto_chunking::rows_of_the_array_of_a_block_of(",
                                         vars_read, ")"),
                                  ifelse(kinds_of_the_tile == "sliced_vector",
                                         paste0("nico::math::auto_chunking::sliced_at_rows_as_vars_of(",
                                                vars_read, ")"),
                                         vars_read))
                name_of_the_loop <-  paste0("the loop over ", candidate$index, " of the rows 1 .. ",
                                            candidate$N_text)
                tiled <-  paste0(
                        "{\n",
                        "const int ", rows_name, " = ", candidate$N_text, ";\n", read_before_the_tile,
                        "lp_accum__.add(nico::math::auto_chunking::sum_of_a_block_of_rows_in_the_chunks_of_",
                        "reduce_sum_static(\n",
                        "\"", name_of_the_loop, "\", ", rows_name, ", ", N_chunks_of_auto_chunking_text, ", ",
                        state_name, ",\n",
                        "[&](const int first__, const int n_rows__, ",
                        paste(arguments_of_the_tile, collapse = ", "),
                        ## (its body compiled once, not inlined at each place the runtime calls it)
                        ") __attribute__((noinline)) {\n",
                        "stan::math::accumulator<local_scalar_t__> lp_accum__;\n",
                        paste0(rebinding, collapse = "\n"), if (length(rebinding) > 0) "\n" else "",
                        reader_in_the_tile, "const int ", candidate$N_text, " = n_rows__;\n",
                        candidate$loop_text, "\n",
                        "return lp_accum__.sum();\n",
                        "},\n",
                        paste(shared, collapse = ", "), "));\n",
                        "}")
                names_of_the_states <-  c(names_of_the_states, state_name)
                tiled_loops <-  c(tiled_loops, tiled)
                ## (what the rules of sums see in its place: a statement that reads its vars)
                placeholders <-  c(placeholders, paste0("nico_auto_chunking_tiled_loop_", k, "__(",
                                                        paste(names(candidate$kinds), collapse = ", "), ");"))
                report <-  c(report, paste0(name_of_the_loop,
                                            ": tiled, every group of a chunk of rows in one tile ",
                                            "(its own statements, with ", candidate$N_text, " the tile's rows; ",
                                            "its rows: ",
                                            paste(vars_read[kinds_of_the_tile != "whole"], collapse = ", "),
                                            "; data in blocks of ", candidate$N_text, " rows: ",
                                            paste(candidate$data_read[candidate$in_blocks], collapse = ", "),
                                            "; data at the tile's rows: ",
                                            paste(candidate$data_read[!candidate$in_blocks], collapse = ", "),
                                            "; copied whole into each tile: ",
                                            paste(vars_read[kinds_of_the_tile == "whole"], collapse = ", "), ")"))

        }
        ## (the last loop first, so that the places of the ones before it hold)
        for (k in rev(seq_along(candidates))) {
                body_text <-  paste0(substring(body_text, 1L, candidates[[k]]$start - 1L), placeholders[[k]],
                                     substring(body_text, candidates[[k]]$close + 1L, nchar(body_text)))
        }
        return(list(body_text = body_text, report = report, names_of_the_states = names_of_the_states,
                    tiled_loops = tiled_loops, placeholders = placeholders))

}
##
## ---- The rewrite of the generated C++ of a Stan model ------------------------------------------------------------------------------
##
fn_auto_chunk_the_sums_over_rows_in_generated_Stan_cpp <-  function(generated_cpp_lines,
                                                                    Stan_model_lines,
                                                                    N_chunks_of_auto_chunking_text,
                                                                    names_of_the_nuisance_blocks) {

        data <-  fn_data_of_Stan_model(Stan_model_lines = Stan_model_lines)
        source_text <-  gsub("//[^\n]*", "", paste(Stan_model_lines, collapse = "\n"))
        start <-  regexpr("(^|\n)\\s*transformed data\\s*\\{", source_text, perl = TRUE)[[1]]
        declarations <-  character(0)
        if (start > 0L) {

                opening <-  start + regexpr("\\{", substring(source_text, start))[[1]] - 1L
                block <-  fn_inside_of_the_bracket(source_text, opening)$inside
                declarations <-  fn_statements_of_cpp_block(block)

        }
        pattern <-  "^int\\s+([A-Za-z_][A-Za-z0-9_]*)\\s*=\\s*([A-Za-z_][A-Za-z0-9_]*)\\s*;$"
        aliases <-  character(0)
        for (declaration in declarations) {

                match <-  regmatches(trimws(declaration), regexec(pattern, trimws(declaration), perl = TRUE))[[1]]
                if (length(match) != 3L) next
                if (!is.null(data[[match[[3]]]]) && data[[match[[3]]]]$shape == "scalar") {

                        aliases[[match[[2]]]] <-  match[[3]]
                        data[[match[[2]]]] <-  list(first_dimension = NA_character_, shape = "scalar")

                }

        }
        attr(data, "scalar_dimension_aliases") <-  aliases
        text <-  paste(generated_cpp_lines, collapse = "\n")
        report <-  character(0)
        declarations_before_the_class <-  character(0)
        running_sum_functions <-  fn_running_sum_functions_of_generated_cpp(text = text,
                                                                            Stan_model_lines = Stan_model_lines)
        ##
        ## ---- each log_prob_impl body (stanc writes one per container type of the parameters) ---------------------------------------
        ##
        starts <-  as.integer(gregexpr(pattern = "log_prob_impl\\(VecR& params_r__", text = text)[[1]])
        if (starts[[1]] < 0) stop("no log_prob_impl in the generated C++")
        for (start in rev(starts)) {
                body_open <-  start + regexpr(pattern = "\\{",
                                              text = substring(text, start, nchar(text)))[[1]] - 1
                characters <-  strsplit(x = substring(text, body_open, nchar(text)), split = "")[[1]]
                depth <-  0
                body_close <-  NA
                for (i in seq_along(characters)) {
                        if (characters[[i]] == "{") depth <-  depth + 1
                        if (characters[[i]] == "}") {
                                depth <-  depth - 1
                                if (depth == 0) {
                                        body_close <-  body_open + i - 1
                                        break
                                }
                        }
                }
                ## (the loops over groups of rows and the blocks of rows in the reverse-mode log_prob only, the one
                ##  whose template requires a var: log_prob<double> has no gradient to chunk)
                template_of_this_log_prob <-  tail(strsplit(x = substring(text, max(1, start - 2000), start),
                                                            split = "template <", fixed = TRUE)[[1]], 1)
                is_the_reverse_mode_log_prob <-  grepl("stan::require_st_var<VecR>", template_of_this_log_prob,
                                                       fixed = TRUE)
                loops <-  if (is_the_reverse_mode_log_prob) {
                        fn_auto_chunk_the_loops_over_groups_of_rows_in_one_log_prob_body(
                                body_text                      = substring(text, body_open, body_close),
                                text                           = text,
                                data                           = data,
                                N_chunks_of_auto_chunking_text = N_chunks_of_auto_chunking_text)
                } else {
                        list(body_text = substring(text, body_open, body_close), report = character(0),
                             names_of_the_states = character(0), tiled_loops = character(0),
                             placeholders = character(0))
                }
                rewritten <-  fn_auto_chunk_one_log_prob_body(body_text = loops$body_text,
                                                              data = data,
                                                              N_chunks_of_auto_chunking_text =
                                                                      N_chunks_of_auto_chunking_text,
                                                              running_sum_functions = running_sum_functions,
                                                              names_of_the_nuisance_blocks =
                                                                      names_of_the_nuisance_blocks)
                partial_sums <-  fn_auto_chunk_the_calls_of_partial_sum_functions(
                                         body_text                      = rewritten$body_text,
                                         text                           = text,
                                         N_chunks_of_auto_chunking_text = N_chunks_of_auto_chunking_text)
                ## (each tiled loop back in its place, where the rules of sums saw only a statement reading its vars)
                with_the_loops <-  partial_sums$body_text
                for (k in seq_along(loops$tiled_loops)) {

                        at <-  gregexpr(pattern = loops$placeholders[[k]], text = with_the_loops, fixed = TRUE)[[1]]
                        if (length(at) != 1 || at[[1]] < 0) stop("a tiled loop's place was not kept: ", k)
                        with_the_loops <-  paste0(substring(with_the_loops, 1L, at[[1]] - 1L), loops$tiled_loops[[k]],
                                                  substring(with_the_loops, at[[1]] + nchar(loops$placeholders[[k]]),
                                                            nchar(with_the_loops)))

                }
                blocks <-  if (is_the_reverse_mode_log_prob) {
                        fn_auto_chunk_the_blocks_of_rows_in_one_log_prob_body(
                                body_text                         = with_the_loops,
                                text                              = text,
                                N_chunks_of_auto_chunking_text    = N_chunks_of_auto_chunking_text,
                                names_of_the_states_of_the_loops  = loops$names_of_the_states)
                } else {
                        list(body_text = with_the_loops, report = character(0))
                }
                declarations_before_the_class <-  c(declarations_before_the_class,
                                                   blocks$declarations_before_the_class)
                text <-  paste0(substring(text, 1, body_open - 1), blocks$body_text,
                                substring(text, body_close + 1, nchar(text)))
                report <-  c(report, paste0("log_prob_impl at character ", start, ":"),
                             paste0("  ", c(loops$report, rewritten$report, partial_sums$report, blocks$report)))
        }
        ## (the state of each block of rows, the model's, before the model's class)
        if (length(declarations_before_the_class) > 0) {

                ## (stanc breaks a long class line after "public")
                class_line <-  regexpr(pattern = paste0("\nclass [A-Za-z_][A-Za-z0-9_]* final :\\s*",
                                                        "public\\s+model_base_crtp<"),
                                       text = text, perl = TRUE)[[1]]
                if (class_line < 0) stop("no model class in the generated C++")
                text <-  paste0(substring(text, 1, class_line),
                                paste(declarations_before_the_class, collapse = ""),
                                substring(text, class_line + 1, nchar(text)))

        }
        return(list(generated_cpp_lines = strsplit(x = text, split = "\n")[[1]], report = report))

}
##
## ---- The number of chunks of AUTO-CHUNKING, read from the model's data when the model is made --------------------------------------
##
## (a member of the model, N_chunks_of_auto_chunking__, set in its constructor from the data's integer
##  N_chunks_of_auto_chunking: one build per cell, the number of chunks searched like N_chunks)
fn_read_N_chunks_of_auto_chunking_from_the_data <-  function(generated_cpp_lines) {

        private_line <-  grep(pattern = "^ private:$", x = generated_cpp_lines)
        constructor_line <-  grep(pattern = ": model_base_crtp\\(0\\) \\{$", x = generated_cpp_lines)
        if ((length(private_line) != 1) || (length(constructor_line) != 1) ||
            (constructor_line < private_line)) {
                stop("the generated C++ has not exactly one private section followed by one model constructor")
        }
        ##
        ## ---- The constructor's signature, however stanc wraps it: its lines back to the one that names var_context -----------------
        ##
        first_line_of_the_signature <-  constructor_line
        while ((first_line_of_the_signature > private_line) &&
               !grepl(pattern = "var_context",
                      x = generated_cpp_lines[first_line_of_the_signature],
                      fixed = TRUE)) {

                first_line_of_the_signature <-  first_line_of_the_signature - 1

        }
        signature <-  gsub(pattern = "[[:space:]]*(::|&)[[:space:]]*",
                           replacement = "\\1",
                           x = paste(generated_cpp_lines[first_line_of_the_signature:constructor_line],
                                     collapse = " "))
        if (!grepl(pattern = "var_context&context__", x = signature, fixed = TRUE)) {
                stop("the model's constructor does not take its data as context__")
        }
        if (any(grepl(pattern = "int N_chunks_of_auto_chunking__;", x = generated_cpp_lines, fixed = TRUE))) {
                stop("the generated C++ already has a member N_chunks_of_auto_chunking__")
        }
        read_from_the_data <-  paste0("    N_chunks_of_auto_chunking__ = nico::math::auto_chunking::",
                                      "N_chunks_of_auto_chunking_from_the_data(context__);")
        lines <-  append(x = generated_cpp_lines, values = read_from_the_data, after = constructor_line)
        lines <-  append(x = lines, values = "  int N_chunks_of_auto_chunking__;", after = private_line)
        return(lines)

}
