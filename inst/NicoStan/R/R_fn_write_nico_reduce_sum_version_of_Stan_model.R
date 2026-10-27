
##
## ==== R_fn_write_nico_reduce_sum_version_of_Stan_model.R ===================================================
##
## ---- nico_reduce_sum: the version of any Stan model whose nuisance blocks (nuisance_params) are held by
##      NicoStan inside the chunks of its reduce_sum, written automatically from the model itself ----
##
## With nuisance_params = c("z", ...) (the names of the model's nuisance blocks, as declared in its parameters
## block), NicoStan writes the nico_reduce_sum version of the model next to it
## (<model>_nico_reduce_sum.stan) and its user header:
##
##   1. each nuisance block is taken out of the parameters block (its declaration is kept as a comment):
##      NicoStan samples its values (the blocks one after the other, each in Stan's own order of its values), and
##      the model reads them with NicoStan's readers (stan_external_nuisance_values_of_the_block_owned_by_
##      NicoStan.hpp), declared in the functions block;
##   2. a reduce_sum (or reduce_sum_static) whose sliced argument is a nuisance block gets the block's term
##      indices (data) as its sliced argument instead, and a copy of its partial-sum function (named
##      <function>_nicostan_chunked_of_<block>) reads the chunk's own terms at its start, under the name of its
##      sliced argument, so the rest of the user's function is unchanged: each chunk's values live on the
##      chunk's own tape only;
##   3. every other use of a nuisance block reads its values where it is used: one term (z[i]), several terms
##      (z[idx], z[a:b]) or the whole block (z), whichever the use is (for an index, stanc decides between the
##      readers of one term and of several terms); on the main tape these values are correct, but not chunked;
##   4. the transformed data get each block's number of terms, values per term, offset and term indices, and
##      the size of the whole nuisance block, which the generated quantities report
##      (nicostan_size_of_the_nuisance_block_held_by_NicoStan) so that NicoStan reads it after compiling.
##
## The "type" argument of the readers is the first value of the model's first main parameter (it only carries
## the scalar type: with autodiff the values become nodes, in the generated quantities numbers). Supported
## nuisance blocks: vector[E], row_vector[E], array[E] real, array[E] vector[K] and array[E] row_vector[K],
## matrix[R, C] (read whole where it is used), array[A] matrix[R, C] (A terms of R x C values), and tuples of
## these (e.g. tuple(array[N] real, vector[K]) z, also with tuples inside them): each component of a tuple
## becomes a nuisance block of its own (<block>_tuple_component_<i>, and <block>_tuple_component_<i>_<j> for a
## tuple inside it), in the tuple's own order, which is Stan's order of its values. Each of these may have a
## constraint (<lower = a>, <upper = b>, <lower = a, upper = b>, <offset = a>, <multiplier = b> or
## <offset = a, multiplier = b>, with bounds that are real numbers): NicoStan then holds the block's UNCONSTRAINED
## values, as Stan's own sampler does, every value the model reads is constrained with Stan math's own transform
## (nicostan_constrain()), and the log-Jacobian of the transform is added to the target once per value
## (nicostan_log_jacobian_of_constraint()): inside the chunks, in the copy of the partial-sum function of the
## reduce_sum that slices the block and adds its value to the target at the top level of the model block, else
## at the start of the model block. Any other block (an array of tuples, an array of more dimensions, bounds of
## one value per value of the block) stops with an error saying so. Any number of nuisance blocks may be given.
##
##
##
#' The nico_reduce_sum version of a Stan model (its nuisance blocks held by NicoStan inside the chunks)
#'
#' @param Stan_model_file_path The Stan model, with its nuisance blocks declared in its parameters block.
#' @param nuisance_params The names of its nuisance blocks (in the order NicoStan holds them).
#' @param stanc_file_path The stanc compiler (BridgeStan's, which compiles the model).
#' @param stanc_include_paths_arguments The --include-paths arguments of the model's stanc arguments (NULL:
#'   none).
#' @param Stan_cpp_user_header The model's own user header (NULL: none), included by the new user header.
#' @param nico_reduce_sum_Stan_model_file_path Where the version is written (default: next to the model).
#' @return A list: Stan_model_file_path and Stan_cpp_user_header of the version, and the data frame
#'   nuisance_blocks (name, shape, the Stan expressions of its terms, values per term and offset).
#' @noRd
R_fn_write_nico_reduce_sum_version_of_Stan_model <-  function(
                    Stan_model_file_path,
                    nuisance_params,
                    stanc_file_path,
                    stanc_include_paths_arguments              = NULL,
                    Stan_cpp_user_header                       = NULL,
                    nico_reduce_sum_Stan_model_file_path       =
                          fn_path_of_the_nico_reduce_sum_version_of_a_Stan_model(Stan_model_file_path)
) {

        stopifnot(is.character(nuisance_params), length(nuisance_params) >= 1, !anyDuplicated(nuisance_params))
        ##
        ## (a version already written under the former name, <model>_reduce_sum_chunked.stan, keeps that name and
        ##  the wording of its written comments, so that its compiled library is still the one it was built from)
        name_of_the_method_in_the_written_comments <-  if (grepl("_reduce_sum_chunked\\.stan$",
                                                                 nico_reduce_sum_Stan_model_file_path)) {
              "reduce_sum_chunked"
        } else {
              "nico_reduce_sum"
        }
        ##
        ## ---- a version already written from these very files, by this very NicoStan install: its record is
        ##      returned as it is, without the rewrite and the stanc probes (the key holds the md5 of the model,
        ##      of
        ##      every .stan file of its include folders, of the user header and of NicoStan's own code and
        ##      headers, with the nuisance names; the version and its user header must still be the files whose
        ##      md5 the record holds):
        ##
        include_folders <-  unlist(strsplit(x = sub(pattern = "^--include-paths=", replacement = "",
                                                    x = as.character(stanc_include_paths_arguments)),
                                            split = ","))
        files_of_the_key <-  c(Stan_model_file_path,
                               if (length(include_folders) > 0) {
                                     list.files(path = include_folders, pattern = "\\.stan$", full.names = TRUE,
                                                recursive = TRUE)
                               },
                               if (!is.null(Stan_cpp_user_header)) Stan_cpp_user_header,
                               file.path(system.file(package = "NicoStan"), "R", "NicoStan.rdb"),
                               fn_file_of_the_one_NicoStan_install(
                                     "include", "NicoStan", "runtime", "general_functions",
                                     "stan_external_nuisance_values_of_the_block_owned_by_NicoStan.hpp",
                                     package = "NicoStan", mustWork = TRUE),
                               fn_file_of_the_one_NicoStan_install(
                                     "include", "NicoStan", "runtime", "general_functions",
                                     "bridge_memory.hpp",
                                     package = "NicoStan", mustWork = TRUE))
        files_of_the_key <-  files_of_the_key[file.exists(files_of_the_key)]
        key_of_this_version <-  paste(c(nuisance_params,
                                        nico_reduce_sum_Stan_model_file_path,
                                        files_of_the_key,
                                        unname(tools::md5sum(files = files_of_the_key))), collapse = "|")
        file_of_the_record <-  sub(pattern = "\\.stan$", replacement = "_record_of_the_writer.RDS",
                                   x = nico_reduce_sum_Stan_model_file_path)
        user_header_of_the_version <-  sub(pattern = "\\.stan$", replacement = "_user_header.hpp",
                                           x = nico_reduce_sum_Stan_model_file_path)
        ##
        ## ---- the md5 of a version with its E5 cell label block left out (the lines from "// ---- E5 cell label"
        ##      to "// ---- end of the E5 cell label"), so that a version whose only change is its label is kept:
        ##
        fn_md5_of_the_version_without_its_E5_cell_label <-  function(file) {

              lines_of_the_file <-  readLines(con = file, warn = FALSE)
              where_first <-  which(startsWith(lines_of_the_file, "// ---- E5 cell label"))
              where_last <-  which(startsWith(lines_of_the_file, "// ---- end of the E5 cell label"))
              if ((length(where_first) == 1) && (length(where_last) == 1) && (where_first < where_last)) {
                    lines_of_the_file <-  lines_of_the_file[-(where_first:where_last)]
              }
              file_of_the_lines <-  tempfile(fileext = ".stan")
              writeLines(text = lines_of_the_file, con = file_of_the_lines)
              md5_of_the_lines <-  unname(tools::md5sum(files = file_of_the_lines))
              unlink(file_of_the_lines)
              return(md5_of_the_lines)

        }
        if (file.exists(file_of_the_record) && file.exists(nico_reduce_sum_Stan_model_file_path) &&
            file.exists(user_header_of_the_version)) {
              record_of_the_writer <-  tryCatch(readRDS(file_of_the_record), error = function(e) NULL)
              # if (!is.null(record_of_the_writer) && identical(record_of_the_writer$key, key_of_this_version) &&
              #     identical(record_of_the_writer$md5_of_the_version,
              #               unname(tools::md5sum(files = nico_reduce_sum_Stan_model_file_path))) &&
              #     identical(record_of_the_writer$md5_of_the_user_header,
              #               unname(tools::md5sum(files = user_header_of_the_version)))) {
              ## (the version as written, or the version as written with only its E5 cell label changed)
              if (!is.null(record_of_the_writer) && identical(record_of_the_writer$key, key_of_this_version) &&
                  (identical(record_of_the_writer$md5_of_the_version,
                             unname(tools::md5sum(files = nico_reduce_sum_Stan_model_file_path))) ||
                   (!is.null(record_of_the_writer$md5_of_the_version_without_its_E5_cell_label) &&
                    identical(record_of_the_writer$md5_of_the_version_without_its_E5_cell_label,
                              fn_md5_of_the_version_without_its_E5_cell_label(
                                    file = nico_reduce_sum_Stan_model_file_path)))) &&
                  identical(record_of_the_writer$md5_of_the_user_header,
                            unname(tools::md5sum(files = user_header_of_the_version)))) {
                    return(record_of_the_writer$result)
              }
        }
        text <-  R_fn_Stan_parallel_call_names(code = R_fn_read_Stan_source_with_includes(
              model_file = Stan_model_file_path, stanc_args = stanc_include_paths_arguments))
        ##
        ## ---- comments and strings are set aside (replaced by numbered markers) before the rewrite, as in
        ##      NicoStan's AVX writer:
        ##
        protected_pattern <-  "\"(?:[^\"\\\\]|\\\\.)*\"|/\\*[\\s\\S]*?\\*/|//[^\\n]*"
        protected <-  regmatches(x = text,
                                 m = gregexpr(pattern = protected_pattern, text = text, perl = TRUE))[[1]]
        for (i in seq_along(protected)) {
              text <-  sub(pattern = protected[i], replacement = paste0("\001", i, "\001"), x = text,
                           fixed = TRUE)
        }
        fn_restore <-  function(code) {

              markers <-  regmatches(x = code, m = gregexpr(pattern = "\001[0-9]+\001", text = code))[[1]]
              for (marker in markers) {
                    i <-  as.numeric(gsub(pattern = "\001", replacement = "", x = marker, fixed = TRUE))
                    code <-  sub(pattern = marker, replacement = protected[i], x = code, fixed = TRUE)
              }
              return(code)

        }
        fn_without_markers <-  function(code) gsub(pattern = "\001[0-9]+\001", replacement = "", x = code)
        ##
        ## ---- the position of the bracket closing the one at "position" ((, [, { or <), and the top-level
        ##      pieces of a text split at a separator (outside every bracket):
        ##
        fn_position_of_closing_bracket <-  function(code, position) {

              opening <-  substr(code, position, position)
              closing <-  c("(" = ")", "[" = "]", "{" = "}", "<" = ">")[[opening]]
              characters <-  strsplit(substring(code, position), "")[[1]]
              depth <-  0
              for (k in seq_along(characters)) {
                    if (characters[k] == opening) depth <-  depth + 1
                    if (characters[k] == closing) depth <-  depth - 1
                    if (depth == 0) return(position + k - 1)
              }
              stop(paste0("nico_reduce_sum: no bracket closes the ", opening, " at character ", position))

        }
        fn_split_at_top_level <-  function(code, separator) {

              characters <-  strsplit(code, "")[[1]]
              depth <-  0
              pieces <-  c()
              start <-  1
              for (k in seq_along(characters)) {
                    if (characters[k] %in% c("(", "[", "{")) depth <-  depth + 1
                    if (characters[k] %in% c(")", "]", "}")) depth <-  depth - 1
                    if (depth == 0 && characters[k] == separator) {
                          pieces <-  c(pieces, substr(code, start, k - 1))
                          start <-  k + 1
                    }
              }
              pieces <-  c(pieces, substr(code, start, nchar(code)))
              return(pieces)

        }
        ##
        ## ---- the program blocks: the positions of the { and } of each block present (at the top level):
        ##
        fn_blocks <-  function(code) {

              names_of_blocks <-  c("functions", "transformed data", "data", "transformed parameters",
                                    "parameters", "model", "generated quantities")
              blocks <-  list()
              position <-  1
              repeat {
                    rest <-  substring(code, position)
                    match <-  regexpr(pattern = paste0("^(\\s|\001[0-9]+\001)*(",
                                                       paste(names_of_blocks, collapse = "|"), ")\\s*\\{"),
                                      text = rest, perl = TRUE)
                    if (match == -1) break
                    matched <-  fn_without_markers(regmatches(rest, match))
                    name <-  gsub(pattern = "\\s+", replacement = " ",
                                  x = trimws(sub(pattern = "\\{$", replacement = "", x = matched)))
                    opening <-  position + attr(match, "match.length") - 1
                    closing <-  fn_position_of_closing_bracket(code, opening)
                    blocks[[name]] <-  c(opening = opening, closing = closing)
                    position <-  closing + 1
              }
              return(blocks)

        }
        ##
        ## ---- a declaration of the parameters block: its name, base type, array sizes, sizes and constraint:
        ##
        fn_declaration <-  function(statement) {

              statement <-  trimws(fn_without_markers(statement))
              array_sizes <-  NULL
              if (grepl(pattern = "^array\\s*\\[", x = statement)) {
                    opening <-  regexpr(pattern = "\\[", text = statement)
                    closing <-  fn_position_of_closing_bracket(statement, opening)
                    array_sizes <-  trimws(fn_split_at_top_level(substr(statement, opening + 1, closing - 1),
                                                                 ","))
                    statement <-  trimws(substring(statement, closing + 1))
              }
              base_type <-  sub(pattern = "^([A-Za-z_]+).*$", replacement = "\\1", x = statement)
              statement <-  trimws(substring(statement, nchar(base_type) + 1))
              constraint <-  ""
              if (startsWith(statement, "<")) {
                    closing <-  fn_position_of_closing_bracket(statement, 1)
                    constraint <-  substr(statement, 1, closing)
                    statement <-  trimws(substring(statement, closing + 1))
              }
              sizes <-  NULL
              if (startsWith(statement, "[")) {
                    closing <-  fn_position_of_closing_bracket(statement, 1)
                    sizes <-  trimws(fn_split_at_top_level(substr(statement, 2, closing - 1), ","))
                    statement <-  trimws(substring(statement, closing + 1))
              }
              return(list(name = statement, base_type = base_type, array_sizes = array_sizes, sizes = sizes,
                          constraint = constraint))

        }
        ##
        ## ---- the constraint of a nuisance block, from the "<...>" of its declaration: its kind, 0 (none), 1
        ##      (lower), 2 (upper), 3 (lower and upper) or 4 (offset and multiplier), as nicostan_constrain() and
        ##      nicostan_log_jacobian_of_constraint() take it, and its two bounds (Stan expressions): lower and
        ##      upper, or offset and multiplier (a bound the kind does not use is "0"; an offset not given is "0",
        ##      a multiplier not given "1", as in Stan):
        ##
        fn_constraint <-  function(constraint,
                                   name_of_block) {

              if (!nzchar(constraint)) {
                    return(list(kind = 0, name_of_kind = "", text = "", first_bound = "0", second_bound = "0",
                                first_bound_is_a_container = FALSE, second_bound_is_a_container = FALSE))
              }
              pieces <-  trimws(fn_split_at_top_level(substr(constraint, 2, nchar(constraint) - 1), ","))
              position_of_equals <-  regexpr(pattern = "=", text = pieces, fixed = TRUE)
              if (any(position_of_equals < 1)) {
                    stop(paste0("nico_reduce_sum: the constraint ", constraint, " of the nuisance block ",
                                name_of_block, " is not of the form <key = value, ...>"))
              }
              keys <-  trimws(substr(pieces, 1, position_of_equals - 1))
              bounds <-  trimws(substring(pieces, position_of_equals + 1))
              names(bounds) <-  keys
              keys_given <-  paste(sort(keys), collapse = ", ")
              kind <-  switch(keys_given,
                              "lower"               = 1,
                              "upper"               = 2,
                              "lower, upper"        = 3,
                              "offset"              = 4,
                              "multiplier"          = 4,
                              "multiplier, offset"  = 4,
                              NA)
              if (is.na(kind)) {
                    stop(paste0("nico_reduce_sum: the constraint ", constraint, " of the nuisance block ",
                                name_of_block, " is not one of <lower = a>, <upper = b>, ",
                                "<lower = a, upper = b>, <offset = a>, <multiplier = b> or ",
                                "<offset = a, multiplier = b>"))
              }
              first_bound <-  if (kind %in% c(1, 3)) bounds[["lower"]] else
                              if (kind == 4 && "offset" %in% keys) bounds[["offset"]] else "0"
              second_bound <-  if (kind %in% c(2, 3)) bounds[["upper"]] else
                               if (kind == 4 && "multiplier" %in% keys) bounds[["multiplier"]] else
                               if (kind == 4) "1" else "0"
              ## (a bound that uses a nuisance block cannot be read where the block's values are constrained)
              names_in_the_bounds <-  regmatches(paste(bounds, collapse = " "),
                                                 gregexpr(pattern = "[A-Za-z_][A-Za-z0-9_]*",
                                                          text = paste(bounds, collapse = " ")))[[1]]
              if (any(names_in_the_bounds %in% nuisance_params)) {
                    stop(paste0("nico_reduce_sum: the constraint ", constraint, " of the nuisance block ",
                                name_of_block, " uses the nuisance block(s) ",
                                paste(intersect(names_in_the_bounds, nuisance_params), collapse = ", "),
                                ", which is not supported"))
              }
              ## (whether a bound is one value per value of the block: set by stanc's check below)
              return(list( kind                         = kind,
                           name_of_kind                 = c("lower", "upper", "lower_upper",
                                                            "offset_multiplier")[kind],
                           text                         = constraint,
                           first_bound                  = first_bound,
                           second_bound                 = second_bound,
                           first_bound_is_a_container   = FALSE,
                           second_bound_is_a_container  = FALSE))

        }
        ##
        ## ---- statements put at the start of the model block (a model block is added after the parameters block,
        ##      or after the transformed parameters block when there is one, when the model has none):
        ##
        fn_with_statements_at_the_start_of_the_model_block <-  function(code,
                                                                       statements) {

              if (is.null(fn_blocks(code)[["model"]])) {
                    block_before <-  fn_blocks(code)[["transformed parameters"]]
                    if (is.null(block_before)) block_before <-  fn_blocks(code)[["parameters"]]
                    code <-  paste0(substr(code, 1, block_before[["closing"]]), "\nmodel {\n}",
                                    substring(code, block_before[["closing"]] + 1))
              }
              opening <-  fn_blocks(code)[["model"]][["opening"]]
              return(paste0(substr(code, 1, opening), statements, substring(code, opening + 1)))

        }
        ##
        ## ---- statements put at the start of the transformed parameters block (one is added after the parameters
        ##      block when the model has none):
        ##
        fn_with_statements_at_the_start_of_the_transformed_parameters_block <-  function(code,
                                                                                        statements) {

              if (is.null(fn_blocks(code)[["transformed parameters"]])) {
                    block_before <-  fn_blocks(code)[["parameters"]]
                    code <-  paste0(substr(code, 1, block_before[["closing"]]), "\ntransformed parameters {\n}",
                                    substring(code, block_before[["closing"]] + 1))
              }
              opening <-  fn_blocks(code)[["transformed parameters"]][["opening"]]
              return(paste0(substr(code, 1, opening), statements, substring(code, opening + 1)))

        }
        ##
        ## ---- stanc's check of a text of the model (comments and strings put back):
        ##
        candidate_file <-  tempfile(fileext = ".stan")
        stanc_output_file <-  tempfile(fileext = ".hpp")
        on.exit(unlink(c(candidate_file, stanc_output_file)), add = TRUE)
        last_message_of_stanc <-  ""
        fn_stanc_accepts <-  function(code) {

              writeLines(text = fn_restore(code), con = candidate_file)
              output <-  suppressWarnings(system2( command = stanc_file_path,
                                                   args    = c("--allow-undefined", stanc_include_paths_arguments,
                                                               paste0("--o=", shQuote(stanc_output_file)),
                                                               shQuote(candidate_file)),
                                                   stdout  = TRUE,
                                                   stderr  = TRUE))
              status <-  attr(output, "status")
              last_message_of_stanc <<-  paste(output, collapse = "\n")
              return(is.null(status) || identical(as.integer(status), 0L))

        }
        ##
        ## ---- the nuisance blocks and the main parameters, from the parameters block:
        ##
        ##
        ## ---- tuple nuisance blocks: each component of the tuple (and of every tuple inside it) becomes a
        ##      nuisance block of its own, named <block>_tuple_component_<i> (<block>_tuple_component_<i>_<j>
        ##      inside a tuple inside it), in the tuple's own order (Stan's order of its values). In the
        ##      parameters block, the tuple's declaration becomes one declaration per component; after it
        ##      (transformed parameters, model, generated quantities), every use of a component (z.1, z.2.1)
        ##      becomes the component's name, and every use of the whole tuple (z), or of a tuple inside it (z.2),
        ##      becomes the tuple of its components, e.g. (z_tuple_component_1, z_tuple_component_2). The model is
        ##      then rewritten as for any other nuisance blocks, with the components in place of the tuple in
        ##      nuisance_params:
        ##
        fn_split_type_at_top_level <-  function(code) {

              ## (a list of types split at its top-level commas: the commas inside (), [], {} and <> are kept)
              characters <-  strsplit(code, "")[[1]]
              depth <-  0
              pieces <-  c()
              start <-  1
              for (k in seq_along(characters)) {
                    if (characters[k] %in% c("(", "[", "{", "<")) depth <-  depth + 1
                    if (characters[k] %in% c(")", "]", "}", ">")) depth <-  depth - 1
                    if (depth == 0 && characters[k] == ",") {
                          pieces <-  c(pieces, substr(code, start, k - 1))
                          start <-  k + 1
                    }
              }
              pieces <-  c(pieces, substr(code, start, nchar(code)))
              return(trimws(pieces))

        }
        fn_tuple_tree <-  function(type, name_of_this_value, prefix_of_its_components) {

              ## (the tree of a tuple type: for each value, its name or, for a tuple, its components)
              type <-  trimws(type)
              if (!grepl(pattern = "^tuple\\s*\\(", x = type)) {
                    return(list( expression   = name_of_this_value,
                                 children     = NULL,
                                 leaf_names   = name_of_this_value,
                                 leaf_types   = type))
              }
              opening <-  regexpr(pattern = "\\(", text = type)
              closing <-  fn_position_of_closing_bracket(type, opening)
              component_types <-  fn_split_type_at_top_level(substr(type, opening + 1, closing - 1))
              children <-  lapply(seq_along(component_types), function(i) {

                    fn_tuple_tree( type                       = component_types[i],
                                   name_of_this_value         = paste0(prefix_of_its_components, "_", i),
                                   prefix_of_its_components   = paste0(prefix_of_its_components, "_", i))
              })

              return(list( expression   = paste0("(", paste(vapply(children, function(child) child$expression,
                                                                   character(1)), collapse = ", "), ")"),
                           children     = children,
                           leaf_names   = unlist(lapply(children, function(child) child$leaf_names)),
                           leaf_types   = unlist(lapply(children, function(child) child$leaf_types))))

        }
        blocks_before_the_tuples <-  fn_blocks(text)
        if (!is.null(blocks_before_the_tuples[["parameters"]])) {

              opening_of_parameters <-  blocks_before_the_tuples[["parameters"]][["opening"]]
              closing_of_parameters <-  blocks_before_the_tuples[["parameters"]][["closing"]]
              statements_of_parameters <-  fn_split_at_top_level(substr(text, opening_of_parameters + 1,
                                                                        closing_of_parameters - 1), ";")
              nuisance_params_with_the_tuple_components <-  c()
              text_after_the_parameters <-  substring(text, closing_of_parameters + 1)
              for (name in nuisance_params) {
                    k <-  which(vapply(statements_of_parameters, function(statement) {

                          statement <-  trimws(fn_without_markers(statement))
                          nzchar(statement) && grepl(pattern = paste0("(^|[^A-Za-z0-9_])", name, "$"),
                                                     x = statement)
                    }, logical(1)))

                    declaration_text <-  ""
                    if (length(k) == 1) {
                          declaration_text <-  trimws(fn_without_markers(statements_of_parameters[k]))
                    }
                    if (grepl(pattern = "^array\\s*\\[[\\s\\S]*\\]\\s*tuple\\s*\\(", x = declaration_text,
                              perl = TRUE)) {
                          stop(paste0("nico_reduce_sum: the nuisance block ", name, " is an array of tuples, ",
                                      "which is not supported; declare it as a tuple of arrays instead (e.g. ",
                                      "tuple(array[N] real, array[N] vector[K]) rather than ",
                                      "array[N] tuple(real, vector[K]))"))
                    }
                    if (!grepl(pattern = "^tuple\\s*\\(", x = declaration_text)) {
                          nuisance_params_with_the_tuple_components <-  c(
                                nuisance_params_with_the_tuple_components, name)
                          next
                    }
                    ## (the tuple's type is everything before its name)
                    tuple_type <-  trimws(substr(declaration_text, 1, nchar(declaration_text) - nchar(name)))
                    tree <-  fn_tuple_tree( type                       = tuple_type,
                                            name_of_this_value         = name,
                                            prefix_of_its_components   = paste0(name, "_tuple_component"))
                    names_already_used <-  regmatches(text, gregexpr(pattern = "[A-Za-z_][A-Za-z0-9_]*",
                                                                     text = text))[[1]]
                    if (any(tree$leaf_names %in% names_already_used)) {
                          stop(paste0("nico_reduce_sum: the model already uses one of the names given to the ",
                                      "components of the tuple ", name, ": ",
                                      paste(intersect(tree$leaf_names, names_already_used), collapse = ", ")))
                    }
                    ##
                    ## (the declaration: one per component, with the comments before it kept)
                    leading_part <-  regmatches(statements_of_parameters[k],
                                                regexpr(pattern = "^(\\s|\001[0-9]+\001)*",
                                                        text = statements_of_parameters[k]))
                    statements_of_parameters[k] <-  paste0(leading_part,
                                                           paste(paste0(tree$leaf_types, " ", tree$leaf_names),
                                                                 collapse = ";\n      "))
                    ##
                    ## (the uses after the parameters block: z, z.1, z.2.1, ...)
                    use_pattern <-  paste0("(?<![A-Za-z0-9_.])", name, "((?:\\.[0-9]+)*)(?![A-Za-z0-9_])")
                    uses <-  gregexpr(pattern = use_pattern, text = text_after_the_parameters, perl = TRUE)[[1]]
                    if (uses[1] != -1) {
                          replacements <-  vapply(regmatches(text_after_the_parameters, list(uses))[[1]],
                                                  function(use) {

                                path <-  as.numeric(strsplit(substring(use, nchar(name) + 2), ".",
                                                             fixed = TRUE)[[1]])
                                node <-  tree
                                for (index in path[!is.na(path)]) {
                                      if (is.null(node$children) || index > length(node$children)) {
                                            stop(paste0("nico_reduce_sum: ", use, " is not a component of ",
                                                        "the tuple ", name))
                                      }
                                      node <-  node$children[[index]]
                                }
                                return(node$expression)
                          }, character(1))

                          regmatches(text_after_the_parameters, list(uses)) <-  list(replacements)
                    }
                    nuisance_params_with_the_tuple_components <-  c(nuisance_params_with_the_tuple_components,
                                                                    tree$leaf_names)
                    message(paste0("\033[36mnico_reduce_sum: the tuple ", name, " is held as the nuisance ",
                                   "blocks ", paste(tree$leaf_names, collapse = ", "), "\033[0m"))
              }
              text <-  paste0(substr(text, 1, opening_of_parameters),
                              paste(statements_of_parameters, collapse = ";"),
                              substr(text, closing_of_parameters, closing_of_parameters),
                              text_after_the_parameters)
              nuisance_params <-  nuisance_params_with_the_tuple_components
              stopifnot(!anyDuplicated(nuisance_params))
        }


        blocks <-  fn_blocks(text)
        if (is.null(blocks[["parameters"]])) stop("nico_reduce_sum: the model has no parameters block")
        ##
        ## ---- the model's own data and transformed data (before the rewrite), from which the sizes of the
        ##      nuisance blocks are written in terms of the data in nuisance_blocks (see the end):
        ##
        data_body_of_the_model <-  if (is.null(blocks[["data"]])) "" else
                                        fn_without_markers(substr(text, blocks[["data"]][["opening"]] + 1,
                                                                  blocks[["data"]][["closing"]] - 1))
        transformed_data_body_of_the_model <-  if (is.null(blocks[["transformed data"]])) "" else
                                                    fn_without_markers(substr(text,
                                                        blocks[["transformed data"]][["opening"]] + 1,
                                                        blocks[["transformed data"]][["closing"]] - 1))
        parameters_body <-  substr(text, blocks[["parameters"]][["opening"]] + 1,
                                   blocks[["parameters"]][["closing"]] - 1)
        statements <-  fn_split_at_top_level(parameters_body, ";")
        statements <-  statements[nzchar(trimws(fn_without_markers(statements)))]
        declarations <-  lapply(statements, fn_declaration)
        names_declared <-  vapply(declarations, function(declaration) declaration$name, character(1))
        missing_blocks <-  setdiff(nuisance_params, names_declared)
        if (length(missing_blocks) > 0) {

              stop(paste0("nico_reduce_sum: nuisance_params not declared in the parameters block: ",
                          paste(missing_blocks, collapse = ", ")))
        }

        nuisance_blocks <-  data.frame()
        offset_expression <-  "0"
        ## (each block's constraint (fn_constraint()), and the rows and columns of each matrix of an array of
        ##  matrices)
        constraint_of_each_block <-  list()
        rows_and_columns_of_each_block <-  list()
        for (name in nuisance_params) {
              declaration <-  declarations[[match(name, names_declared)]]
              constraint_of_each_block[[name]] <-  fn_constraint(constraint     = declaration$constraint,
                                                                 name_of_block  = name)
              shape <-  NA_character_
              if (is.null(declaration$array_sizes) && declaration$base_type %in% c("vector", "row_vector") &&
                  length(declaration$sizes) == 1) {
                    shape <-  declaration$base_type
                    terms <-  declaration$sizes
                    values_per_term <-  "1"
              }
              if (length(declaration$array_sizes) == 1 && declaration$base_type == "real" &&
                  is.null(declaration$sizes)) {
                    shape <-  "array_of_reals"
                    terms <-  declaration$array_sizes
                    values_per_term <-  "1"
              }
              if (length(declaration$array_sizes) == 1 && declaration$base_type %in% c("vector", "row_vector") &&
                  length(declaration$sizes) == 1) {
                    shape <-  paste0("array_of_", declaration$base_type, "s")
                    terms <-  declaration$array_sizes
                    values_per_term <-  declaration$sizes
              }
              ## (matrix[R, C]: C terms (its columns) of R values each, Stan's order of its values being column
              ##  by column; every use reads the whole matrix and keeps its own indexing)
              if (is.null(declaration$array_sizes) && declaration$base_type == "matrix" &&
                  length(declaration$sizes) == 2) {
                    shape <-  "matrix"
                    terms <-  declaration$sizes[2]
                    values_per_term <-  declaration$sizes[1]
              }
              ## (array[A] matrix[R, C]: A terms (its matrices) of R x C values each, the values of each matrix in
              ##  Stan's order, column by column)
              if (length(declaration$array_sizes) == 1 && declaration$base_type == "matrix" &&
                  length(declaration$sizes) == 2) {
                    shape <-  "array_of_matrices"
                    terms <-  declaration$array_sizes
                    values_per_term <-  paste0("(", declaration$sizes[1], ") * (", declaration$sizes[2], ")")
                    rows_and_columns_of_each_block[[name]] <-  declaration$sizes
              }
              if (is.na(shape)) {
                    stop(paste0("nico_reduce_sum: the nuisance block ", name, " is not one of vector[E], ",
                                "row_vector[E], array[E] real, array[E] vector[K], array[E] row_vector[K], ",
                                "matrix[R, C] or array[A] matrix[R, C]"))
              }
              nuisance_blocks <-  rbind(nuisance_blocks,
                                        data.frame( name = name, shape = shape, terms = terms,
                                                    values_per_term = values_per_term,
                                                    offset = offset_expression, stringsAsFactors = FALSE))
              offset_expression <-  paste0("nicostan_offset_of_", name, " + nicostan_terms_of_", name,
                                           " * nicostan_values_per_term_of_", name)
        }
        ##
        ## ---- each bound of a constraint is a real number, or one value per value of the block (a container of
        ##      the block's own type and sizes, as Stan allows): stanc's check of the model with the bound given
        ##      to a real, then to the block's type, at the start of its model block (a number needs no check):
        ##
        for (name in names(constraint_of_each_block)) {
              constraint <-  constraint_of_each_block[[name]]
              if (constraint$kind == 0) next
              declaration <-  declarations[[match(name, names_declared)]]
              type_of_the_block <-  paste0(if (!is.null(declaration$array_sizes)) {
                                                 paste0("array[", paste(declaration$array_sizes, collapse = ", "),
                                                        "] ")
                                           } else "",
                                           declaration$base_type,
                                           if (!is.null(declaration$sizes)) {
                                                 paste0("[", paste(declaration$sizes, collapse = ", "), "]")
                                           } else "")
              for (which_bound in c("first", "second")) {
                    bound <-  constraint[[paste0(which_bound, "_bound")]]
                    if (grepl(pattern = "^[-+]?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?$", x = bound)) next
                    probe <-  fn_with_statements_at_the_start_of_the_model_block(
                                    code        = text,
                                    statements  = paste0("\n    real ",
                                                         "nicostan_check_that_the_bound_is_a_real_number = ",
                                                         bound, ";\n"))
                    if (fn_stanc_accepts(probe)) next
                    probe <-  fn_with_statements_at_the_start_of_the_model_block(
                                    code        = text,
                                    statements  = paste0("\n    ", type_of_the_block, " ",
                                                         "nicostan_check_that_the_bound_is_a_container = ",
                                                         bound, ";\n"))
                    if (!fn_stanc_accepts(probe)) {
                          stop(paste0("nico_reduce_sum: the bound ", bound, " of the constraint ",
                                      constraint$text, " of the nuisance block ", name, " is neither a real ",
                                      "number nor a ", type_of_the_block, ":\n", last_message_of_stanc))
                    }
                    constraint[[paste0(which_bound, "_bound_is_a_container")]] <-  TRUE
              }
              constraint_of_each_block[[name]] <-  constraint
              message(colourise(paste0("nico_reduce_sum: the nuisance block ", name, " has the constraint ",
                                       constraint$text, ": NicoStan holds its unconstrained values, and each ",
                                       "value read is constrained with Stan math's own transform"), "cyan"))
        }
        main_declarations <-  declarations[!(names_declared %in% nuisance_params)]
        if (length(main_declarations) == 0) {
              stop(paste0("nico_reduce_sum: the model needs at least one main parameter (besides its ",
                          "nuisance blocks)"))
        }
        ## (the first value of the first main parameter: the type of the readers)
        first_main <-  main_declarations[[1]]
        n_indices_of_first_main <-  length(first_main$array_sizes) + length(first_main$sizes)
        if (first_main$base_type %in% c("cholesky_factor_corr", "cholesky_factor_cov", "corr_matrix",
                                        "cov_matrix") && length(first_main$sizes) == 1) {
              n_indices_of_first_main <-  length(first_main$array_sizes) + 2
        }
        type_of_the_readers <-  if (n_indices_of_first_main == 0) first_main$name else
                                     paste0(first_main$name, "[", paste(rep("1", n_indices_of_first_main),
                                                                         collapse = ", "), "]")
        ##
        ## ---- the readers' calls for a block: the whole block, terms a..b, one term, several terms (index NULL
        ##      for the whole block); the values of a constrained block constrained with Stan math's own
        ##      transform:
        ##
        fn_read <-  function(block,
                             what,
                             index) {

              ## (what = "indexed": a read made by fn_candidates_of_a_use_in_a_chunk(), index = c(the use's own
              ##  brackets, the reader's call with them))
              call <-  if (what == "indexed") index[2] else fn_read_unconstrained(block = block, what = what,
                                                                                  index = index)
              constraint <-  constraint_of_each_block[[block$name]]
              if (constraint$kind == 0) return(call)
              ## (a bound of one value per value of the block is indexed as the use is)
              fn_bound_of_the_read <-  function(bound,
                                                bound_is_a_container) {

                    if (!bound_is_a_container || (what == "whole")) return(bound)
                    bound_to_index <-  if (grepl(pattern = "^[A-Za-z_][A-Za-z0-9_]*$", x = bound)) bound else
                                            paste0("(", bound, ")")
                    if (what == "indexed") return(paste0(bound_to_index, trimws(index[1])))
                    if (what == "range") return(paste0(bound_to_index, "[", index[1], ":", index[2], "]"))
                    return(paste0(bound_to_index, "[", index, "]"))
              }

              return(paste0("nicostan_constrain(", call, ", ", constraint$kind, ", ",
                            fn_bound_of_the_read(bound                 = constraint$first_bound,
                                                 bound_is_a_container  = constraint$first_bound_is_a_container),
                            ", ",
                            fn_bound_of_the_read(bound                 = constraint$second_bound,
                                                 bound_is_a_container  = constraint$second_bound_is_a_container),
                            ")"))

        }
        fn_read_unconstrained <-  function(block,
                                           what,
                                           index) {

              offset <-  paste0("nicostan_offset_of_", block$name)
              k <-  paste0("nicostan_values_per_term_of_", block$name)
              first <-  if (what == "whole") "1" else if (what == "range") index[1] else index
              last <-  if (what == "whole") paste0("nicostan_terms_of_", block$name) else
                       if (what == "range") index[2] else index
              type <-  type_of_the_readers
              if (block$shape == "matrix") {
                    return(paste0("to_matrix(nicostan_nuisance_vector(1, nicostan_terms_of_", block$name,
                                  " * ", k, ", ", offset, ", ", type, "), ", k, ", nicostan_terms_of_",
                                  block$name, ")"))
              }
              if (block$shape %in% c("vector", "row_vector")) {
                    if (what == "one") {
                          return(paste0("nicostan_nuisance_value(", index, ", ", offset, ", ", type, ")"))
                    }
                    call <-  if (what == "several") {
                          paste0("nicostan_nuisance_vector_at(", index, ", ", offset, ", ", type, ")")
                    } else {
                          paste0("nicostan_nuisance_vector(", first, ", ", last, ", ", offset, ", ", type, ")")
                    }
                    if (block$shape == "row_vector") call <-  paste0("(", call, ")'")
                    return(call)
              }
              if (block$shape == "array_of_matrices") {
                    rows_and_columns <-  paste0("nicostan_rows_per_term_of_", block$name,
                                                ", nicostan_columns_per_term_of_", block$name)
                    if (what == "one") {
                          return(paste0("nicostan_nuisance_array_of_matrices(", index, ", ", index, ", ",
                                        rows_and_columns, ", ", offset, ", ", type, ")[1]"))
                    }
                    return(paste0("nicostan_nuisance_array_of_matrices(", first, ", ", last, ", ",
                                  rows_and_columns, ", ", offset, ", ", type, ")"))
              }
              if (block$shape == "array_of_reals") {
                    if (what == "one") {
                          return(paste0("nicostan_nuisance_value(", index, ", ", offset, ", ", type, ")"))
                    }
                    if (what == "several") {
                          return(paste0("nicostan_nuisance_array_of_reals_at(", index, ", ", offset, ", ", type,
                                        ")"))
                    }
                    return(paste0("nicostan_nuisance_array_of_reals(", first, ", ", last, ", ", offset, ", ",
                                  type, ")"))
              }
              reader <-  paste0("nicostan_nuisance_", block$shape)
              if (what == "one") {
                    return(paste0(reader, "(", index, ", ", index, ", ", k, ", ", offset, ", ", type, ")[1]"))
              }
              return(paste0(reader, "(", first, ", ", last, ", ", k, ", ", offset, ", ", type, ")"))

        }
        ##
        ## ---- 2. each reduce_sum whose sliced argument is a nuisance block: the term indices as its sliced
        ##      argument, and a copy of its partial-sum function reading the chunk's own terms:
        ##
        ##
        ##      For a constrained block, the copy reads the chunk's unconstrained values, constrains them, and,
        ##      when its call says so (nicostan_with_the_log_jacobian = 1), adds the log-Jacobian of the chunk's
        ##      own values to every value it returns: the reduce_sum calls the copy once per chunk, its chunks
        ##      hold each term exactly once, so each value's log-Jacobian is added once. Only the first reduce_sum
        ##      that slices the block and adds its value to the target as it is, at the top level of the model
        ##      block (fn_reduce_sum_call_adds_its_value_to_the_target_once()), says so; for a constrained block
        ##      that no such reduce_sum slices, the log-Jacobian is added by jacobian += at the start of the
        ##      transformed parameters block (see below):
        ##
        fn_body_with_the_log_jacobian_added_to_every_return <-  function(body_of_definition) {

              positions <-  gregexpr(pattern = "(?<![A-Za-z0-9_])return(?![A-Za-z0-9_])",
                                     text = body_of_definition, perl = TRUE)[[1]]
              if (positions[1] == -1) {
                    stop("nico_reduce_sum: a partial-sum function with no return statement")
              }
              for (position in rev(positions)) {
                    start_of_value <-  position + nchar("return")
                    value <-  fn_split_at_top_level(substring(body_of_definition, start_of_value), ";")[1]
                    body_of_definition <-  paste0(substr(body_of_definition, 1, start_of_value - 1),
                                                  " nicostan_log_jacobian_of_the_slice + (", trimws(value), ")",
                                                  substring(body_of_definition, start_of_value + nchar(value)))
              }
              return(body_of_definition)

        }
        fn_reduce_sum_call_adds_its_value_to_the_target_once <-  function(code,
                                                                         start_of_the_call,
                                                                         end_of_the_call) {

              model_block <-  fn_blocks(code)[["model"]]
              if (is.null(model_block) || start_of_the_call < model_block[["opening"]] ||
                  end_of_the_call > model_block[["closing"]]) {
                    return(FALSE)
              }
              ## (at the top level of the model block: no { } open before it, so not inside a loop, a condition or
              ##  a block of its own)
              before <-  fn_without_markers(substr(code, model_block[["opening"]] + 1, start_of_the_call - 1))
              characters_before <-  strsplit(before, "")[[1]]
              if (sum(characters_before == "{") != sum(characters_before == "}")) return(FALSE)
              ## (its statement, from the end of the statement before it: target += <call>, with only + terms
              ##  before it and + or - terms after it, outside every bracket; a loop or a condition without { }
              ##  makes the statement start with for, while, if or else)
              ends_of_statements <-  gregexpr(pattern = "[;{}]", text = before)[[1]]
              start_of_statement <-  if (ends_of_statements[1] == -1) 0 else max(ends_of_statements)
              before_the_call <-  trimws(substring(before, start_of_statement + 1))
              if (!grepl(pattern = "^target\\s*\\+=", x = before_the_call)) return(FALSE)
              terms_before <-  trimws(sub(pattern = "^target\\s*\\+=", replacement = "", x = before_the_call))
              characters_of_terms_before <-  strsplit(terms_before, "")[[1]]
              if (nzchar(terms_before) && (!endsWith(terms_before, "+") ||
                                           sum(characters_of_terms_before %in% c("(", "[", "{")) !=
                                           sum(characters_of_terms_before %in% c(")", "]", "}")))) {
                    return(FALSE)
              }
              rest_of_the_code <-  fn_without_markers(substring(code, end_of_the_call + 1))
              after_the_call <-  trimws(fn_split_at_top_level(rest_of_the_code, ";")[1])
              if (nzchar(after_the_call) && !grepl(pattern = "^[-+]", x = after_the_call)) return(FALSE)
              return(TRUE)

        }
        blocks_whose_log_jacobian_is_added_inside_the_chunks <-  c()
        copies_of_partial_sum_functions <-  c()
        repeat {
              match <-  regexpr(pattern = paste0("(?<![A-Za-z0-9_])reduce_sum(_static)?\\s*\\(",
                                                 "\\s*[A-Za-z0-9_]+\\s*,",
                                                 "\\s*(", paste(nuisance_params, collapse = "|"), ")\\s*,"),
                                text = text, perl = TRUE)
              if (match == -1) break
              opening <-  match + regexpr(pattern = "\\(", text = regmatches(text, match)) - 1
              closing <-  fn_position_of_closing_bracket(text, opening)
              arguments <-  trimws(fn_split_at_top_level(substr(text, opening + 1, closing - 1), ","))
              block <-  nuisance_blocks[nuisance_blocks$name == arguments[2], ]
              if (block$shape %in% c("vector", "row_vector", "matrix")) {
                    stop(paste0("nico_reduce_sum: reduce_sum slices arrays only, not the ", block$shape, " ",
                                block$name))
              }
              partial_sum <-  arguments[1]
              suffix <-  regmatches(partial_sum, regexpr(pattern = "_(lpdf|lupdf|lpmf|lupmf)$",
                                                         text = partial_sum))
              if (length(suffix) == 0) suffix <-  ""
              base_name <-  sub(pattern = "_(lpdf|lupdf|lpmf|lupmf)$", replacement = "", x = partial_sum)
              definition_suffix <-  sub(pattern = "lupdf", replacement = "lpdf",
                                        x = sub(pattern = "lupmf", replacement = "lpmf", x = suffix))
              copy_name <-  paste0(base_name, "_nicostan_chunked_of_", block$name)
              ## (the copy's sliced argument is the term indices, integers: a density (_lpdf) becomes a mass
              ##  function (_lpmf), whose first argument is an integer array, with the same meaning)
              copy_suffix <-  sub(pattern = "pdf$", replacement = "pmf", x = suffix)
              copy_definition_suffix <-  sub(pattern = "pdf$", replacement = "pmf", x = definition_suffix)
              if (!(copy_name %in% copies_of_partial_sum_functions)) {
                    ## (the definition of the partial-sum function in the functions block: its return type at the
                    ##  start of its line, its arguments and its body)
                    definition <-  regexpr(pattern = paste0("(?<![A-Za-z0-9_])", base_name, definition_suffix,
                                                            "\\s*\\("), text = text, perl = TRUE)
                    if (definition == -1) {
                          stop(paste0("nico_reduce_sum: the partial-sum function ", partial_sum, " is not ",
                                      "defined in the model"))
                    }
                    start_of_line <-  definition
                    while (start_of_line > 1 && substr(text, start_of_line - 1, start_of_line - 1) != "\n") {
                          start_of_line <-  start_of_line - 1
                    }
                    header_of_definition <-  substr(text, start_of_line, definition - 1)
                    opening_of_arguments <-  definition + attr(definition, "match.length") - 1
                    closing_of_arguments <-  fn_position_of_closing_bracket(text, opening_of_arguments)
                    opening_of_body <-  regexpr(pattern = "\\{", text = substring(text, closing_of_arguments)) +
                                        closing_of_arguments - 1
                    closing_of_body <-  fn_position_of_closing_bracket(text, opening_of_body)
                    function_arguments <-  trimws(fn_split_at_top_level(
                                                  substr(text, opening_of_arguments + 1,
                                                         closing_of_arguments - 1), ","))
                    fn_name_of_argument <-  function(argument) {

                          sub(pattern = "^.*[^A-Za-z0-9_]([A-Za-z0-9_]+)$", replacement = "\\1",
                              x = trimws(fn_without_markers(argument)))
                    }

                    slice_name <-  fn_name_of_argument(function_arguments[1])
                    start_name <-  fn_name_of_argument(function_arguments[2])
                    end_name <-  fn_name_of_argument(function_arguments[3])
                    ## (an array of row_vectors that the body only ever reads as to_matrix(<slice>): read
                    ##  straight as the rows of one matrix, nicostan_nuisance_matrix_of_terms(); see
                    ##  fn_slice_of_a_nuisance_block_read_as_a_matrix() below)
                    slice <-  fn_slice_of_a_nuisance_block_read_as_a_matrix(
                                    body_of_definition = substr(text, opening_of_body + 1, closing_of_body),
                                    slice_name         = slice_name,
                                    shape_of_the_block = block$shape)
                    slice_read_as_a_matrix <-  slice$read_as_a_matrix
                    body_of_definition <-  slice$body_of_definition
                    # declaration_of_slice <-  if (block$shape == "array_of_reals") {
                    declaration_of_slice <-  if (slice_read_as_a_matrix) {
                          paste0("matrix[", end_name, " - ", start_name, " + 1, nicostan_values_per_term] ",
                                 slice_name, " = nicostan_nuisance_matrix_of_terms(", start_name, ", ", end_name,
                                 ", nicostan_values_per_term, nicostan_offset, nicostan_type);")
                    } else if (block$shape == "array_of_reals") {
                          paste0("array[", end_name, " - ", start_name, " + 1] real ", slice_name,
                                 " = nicostan_nuisance_array_of_reals(", start_name, ", ", end_name,
                                 ", nicostan_offset, nicostan_type);")
                    } else if (block$shape == "array_of_matrices") {
                          paste0("array[", end_name, " - ", start_name, " + 1] ",
                                 "matrix[nicostan_rows_per_term, nicostan_columns_per_term] ", slice_name,
                                 " = nicostan_nuisance_array_of_matrices(", start_name, ", ", end_name,
                                 ", nicostan_rows_per_term, nicostan_columns_per_term, nicostan_offset, ",
                                 "nicostan_type);")
                    } else {
                          paste0("array[", end_name, " - ", start_name, " + 1] ",
                                 sub(pattern = "^array_of_(.*)s$", replacement = "\\1", x = block$shape),
                                 "[nicostan_values_per_term] ", slice_name, " = nicostan_nuisance_", block$shape,
                                 "(", start_name, ", ", end_name, ", nicostan_values_per_term, nicostan_offset, ",
                                 "nicostan_type);")
                    }
                    ## (a constrained block: the chunk's unconstrained values, then its constrained values under
                    ##  the name of the sliced argument, and the log-Jacobian of the chunk's own values added to
                    ##  every value the copy returns when its call says so)
                    constraint <-  constraint_of_each_block[[block$name]]
                    if (constraint$kind > 0) {
                          end_of_type <-  regexpr(pattern = paste0(" ", slice_name, " = "),
                                                  text = declaration_of_slice, fixed = TRUE)
                          type_of_slice <-  substr(declaration_of_slice, 1, end_of_type - 1)
                          reader_of_slice <-  substring(declaration_of_slice,
                                                        end_of_type + nchar(paste0(" ", slice_name, " = ")))
                          unconstrained_slice_name <-  paste0(slice_name, "_nicostan_unconstrained")
                          ## (a bound of one value per value of the block: the chunk's own terms of it)
                          fn_bound_of_the_slice <-  function(name_of_the_bound,
                                                             bound_is_a_container) {

                                if (!bound_is_a_container) return(name_of_the_bound)
                                slice_of_the_bound <-  paste0(name_of_the_bound, "[", start_name, ":", end_name,
                                                              "]")
                                if (slice_read_as_a_matrix) slice_of_the_bound <-  paste0("to_matrix(",
                                                                                          slice_of_the_bound, ")")
                                return(slice_of_the_bound)
                          }

                          bounds_of_the_slice <-  paste0(
                                fn_bound_of_the_slice(
                                      name_of_the_bound     = "nicostan_first_bound",
                                      bound_is_a_container  = constraint$first_bound_is_a_container), ", ",
                                fn_bound_of_the_slice(
                                      name_of_the_bound     = "nicostan_second_bound",
                                      bound_is_a_container  = constraint$second_bound_is_a_container))
                          declaration_of_slice <-  paste0(
                                type_of_slice, " ", unconstrained_slice_name, " = ", reader_of_slice,
                                "\n        ",
                                type_of_slice, " ", slice_name, " = nicostan_constrain(",
                                unconstrained_slice_name, ", ", constraint$kind, ", ", bounds_of_the_slice,
                                ");\n        ",
                                "real nicostan_log_jacobian_of_the_slice = 0;\n        ",
                                "if (nicostan_with_the_log_jacobian == 1) {\n              ",
                                "nicostan_log_jacobian_of_the_slice = nicostan_log_jacobian_of_constraint(",
                                unconstrained_slice_name, ", ", constraint$kind, ", ", bounds_of_the_slice,
                                ");\n        }\n        ")
                          body_of_definition <-  fn_body_with_the_log_jacobian_added_to_every_return(
                                                       body_of_definition = body_of_definition)
                    }
                    other_arguments_of_the_copy <-  paste0(
                          if (block$shape == "array_of_matrices") {
                                "data int nicostan_rows_per_term, data int nicostan_columns_per_term, "
                          } else "",
                          if (constraint$kind > 0) {
                                ## (a bound of one value per value of the block: the whole block's type)
                                types_of_the_whole_blocks <-  c("vector" = "vector", "row_vector" = "row_vector",
                                                                "array_of_reals" = "array[] real",
                                                                "array_of_vectors" = "array[] vector",
                                                                "array_of_row_vectors" = "array[] row_vector",
                                                                "matrix" = "matrix",
                                                                "array_of_matrices" = "array[] matrix")
                                type_of_the_whole_block <-  types_of_the_whole_blocks[[block$shape]]
                                paste0(if (constraint$first_bound_is_a_container) type_of_the_whole_block else
                                             "real", " nicostan_first_bound, ",
                                       if (constraint$second_bound_is_a_container) type_of_the_whole_block else
                                             "real", " nicostan_second_bound, ",
                                       "data int nicostan_with_the_log_jacobian, ")
                          } else "")
                    copy <-  paste0("\n", header_of_definition, copy_name, copy_definition_suffix, "(",
                                    "data array[] int nicostan_terms_of_the_slice, ",
                                    paste(function_arguments[-1], collapse = ", "),
                                    ", data int nicostan_values_per_term, data int nicostan_offset, ",
                                    other_arguments_of_the_copy,
                                    "real nicostan_type) {\n        ",
                                    "// (", slice_name, ": the chunk's own terms, held by NicoStan)\n        ",
                                    # declaration_of_slice, substr(text, opening_of_body + 1, closing_of_body),
                                    declaration_of_slice, body_of_definition,
                                    "\n")
                    text <-  paste0(substr(text, 1, closing_of_body), copy, substring(text, closing_of_body + 1))
                    copies_of_partial_sum_functions <-  c(copies_of_partial_sum_functions, copy_name)
                    ## (the reduce_sum call moves by the length of the copy when it comes after it)
                    if (match > closing_of_body) {
                          match <-  match + nchar(copy)
                          opening <-  opening + nchar(copy)
                          closing <-  closing + nchar(copy)
                    }
              }
              ## (for a constrained block: its bounds, and whether this call adds the log-Jacobian)
              constraint <-  constraint_of_each_block[[block$name]]
              with_the_log_jacobian <-  0
              if (constraint$kind > 0 &&
                  !(block$name %in% blocks_whose_log_jacobian_is_added_inside_the_chunks) &&
                  fn_reduce_sum_call_adds_its_value_to_the_target_once(code               = text,
                                                                       start_of_the_call  = match,
                                                                       end_of_the_call    = closing)) {
                    with_the_log_jacobian <-  1
                    blocks_whose_log_jacobian_is_added_inside_the_chunks <-  c(
                          blocks_whose_log_jacobian_is_added_inside_the_chunks, block$name)
              }
              other_arguments_of_the_call <-  paste0(
                    if (block$shape == "array_of_matrices") {
                          paste0(", nicostan_rows_per_term_of_", block$name, ", nicostan_columns_per_term_of_",
                                 block$name)
                    } else "",
                    if (constraint$kind > 0) {
                          paste0(", ", constraint$first_bound, ", ", constraint$second_bound, ", ",
                                 with_the_log_jacobian)
                    } else "")
              new_call <-  paste0(substr(text, match, opening - 1), "(", copy_name, copy_suffix, ", ",
                                  "nicostan_term_indices_of_", block$name, ", ",
                                  paste(arguments[-(1:2)], collapse = ", "),
                                  ", nicostan_values_per_term_of_", block$name,
                                  ", nicostan_offset_of_", block$name, other_arguments_of_the_call, ", ",
                                  type_of_the_readers, ")")
              text <-  paste0(substr(text, 1, match - 1), new_call, substring(text, closing + 1))
        }
        ##
        ## ---- the readers' declarations (functions block) and the blocks' transformed data (at the end of the
        ##      transformed data block):
        ##
        declarations_of_the_readers <-  paste0(
              "\n    // (NicoStan's readers of its nuisance block: ",
              name_of_the_method_in_the_written_comments, ")\n",
              "    real nicostan_nuisance_value(int term, int values_before_the_block, real type);\n",
              "    vector nicostan_nuisance_vector(int first_term, int last_term, int values_before_the_block, ",
              "real type);\n",
              "    vector nicostan_nuisance_vector_at(array[] int terms, int values_before_the_block, ",
              "real type);\n",
              "    array[] real nicostan_nuisance_array_of_reals(int first_term, int last_term, ",
              "int values_before_the_block, ",
              "real type);\n",
              "    array[] real nicostan_nuisance_array_of_reals_at(array[] int terms, ",
              "int values_before_the_block, real type);\n",
              "    array[] vector nicostan_nuisance_array_of_vectors(int first_term, int last_term, ",
              "int values_per_term, int values_before_the_block, real type);\n",
              "    array[] row_vector nicostan_nuisance_array_of_row_vectors(int first_term, int last_term, ",
              "int values_per_term, int values_before_the_block, real type);\n",
              "    matrix nicostan_nuisance_matrix_of_terms(int first_term, int last_term, ",
              "int values_per_term, int values_before_the_block, real type);\n")
        ## (the reader of arrays of matrices, and the constraint functions for each type read, only in a model
        ##  that needs them, so that every other model's version is as before)
        if (any(nuisance_blocks$shape == "array_of_matrices")) {
              declarations_of_the_readers <-  paste0(declarations_of_the_readers,
                    "    array[] matrix nicostan_nuisance_array_of_matrices(int first_term, int last_term, ",
                    "int rows, int columns, int values_before_the_block, real type);\n")
        }
        kinds_of_the_constraints <-  vapply(constraint_of_each_block, function(constraint) constraint$kind,
                                            numeric(1))
        if (any(kinds_of_the_constraints > 0)) {

              types_read <-  c("real", "vector", "row_vector", "matrix", "array[] real", "array[] vector",
                               "array[] row_vector", "array[] matrix")
              declarations_of_the_readers <-  paste0(declarations_of_the_readers,
                    "    // (the constraints of the nuisance blocks: Stan math's own transforms, NicoStan ",
                    "holding the unconstrained values)\n",
                    paste0("    ", types_read, " nicostan_constrain(", types_read,
                           " x, int kind, real a, real b);\n", collapse = ""),
                    paste0("    real nicostan_log_jacobian_of_constraint(", types_read,
                           " x, int kind, real a, real b);\n", collapse = ""))
              ## (bounds of one value per value of the block: x and a bound of the same type)
              some_bound_is_a_container <-  any(vapply(constraint_of_each_block, function(constraint)
                    constraint$first_bound_is_a_container || constraint$second_bound_is_a_container, logical(1)))
              if (some_bound_is_a_container) {

                    ## (every type read but real, whose bound of one value is already a real)
                    types_read_of_the_containers <-  setdiff(types_read, "real")
                    types_of_the_bounds <-  list(c("T", "T"), c("T", "real"), c("real", "T"))
                    for (types in types_of_the_bounds) {
                          a <-  if (types[1] == "T") types_read_of_the_containers else "real"
                          b <-  if (types[2] == "T") types_read_of_the_containers else "real"
                          declarations_of_the_readers <-  paste0(declarations_of_the_readers,
                                paste0("    ", types_read_of_the_containers, " nicostan_constrain(",
                                       types_read_of_the_containers, " x, int kind, ", a, " a, ", b, " b);\n",
                                       collapse = ""),
                                paste0("    real nicostan_log_jacobian_of_constraint(",
                                       types_read_of_the_containers, " x, int kind, ", a, " a, ", b, " b);\n",
                                       collapse = ""))
                    }
              }

        }

        ## (the reader of a block of rows and columns of a matrix, for a model whose reduce_sum passes a nuisance
        ##  block as a shared argument: see 2b below)
        shared_arguments_of_the_reduce_sums <-  regmatches(text, gregexpr(
              pattern = "(?<![A-Za-z0-9_])reduce_sum(_static)?\\s*\\([^;]*", text = text, perl = TRUE))[[1]]
        names_in_the_reduce_sums <-  unique(unlist(regmatches(shared_arguments_of_the_reduce_sums, gregexpr(
              pattern = "[A-Za-z_][A-Za-z0-9_]*", text = shared_arguments_of_the_reduce_sums))))
        if (any(nuisance_params %in% names_in_the_reduce_sums)) {
              declarations_of_the_readers <-  paste0(declarations_of_the_readers,
                    "    matrix nicostan_nuisance_matrix_block(int first_row, int last_row, int first_column, ",
                    "int last_column, int rows, int values_before_the_block, real type);\n")
        }
        if (is.null(fn_blocks(text)[["functions"]])) text <-  paste0("functions {\n}\n", text)
        functions_block <-  fn_blocks(text)[["functions"]]
        text <-  paste0(substr(text, 1, functions_block[["opening"]]), declarations_of_the_readers,
                        substring(text, functions_block[["opening"]] + 1))
        transformed_data <-  c()
        for (i in seq_len(nrow(nuisance_blocks))) {
              block <-  nuisance_blocks[i, ]
              transformed_data <-  c(transformed_data,
                    paste0("    int nicostan_terms_of_", block$name, " = ", block$terms, ";"),
                    paste0("    int nicostan_values_per_term_of_", block$name, " = ", block$values_per_term, ";"),
                    paste0("    int nicostan_offset_of_", block$name, " = ", block$offset, ";"),
                    # paste0("    array[nicostan_terms_of_", block$name, "] int nicostan_term_indices_of_",
                    #        block$name, " = linspaced_int_array(nicostan_terms_of_", block$name, ", 1, ",
                    #        "nicostan_terms_of_", block$name, ");"))
                    ## (a block may have no terms, e.g. no study with 4 index tests in a network of 10 studies:
                    ##  Stan's linspaced_int_array(0, 1, 0) stops on high < low before it returns the empty array,
                    ##  so the high end is max(1, terms); the indices are the same for every block with terms)
                    paste0("    array[nicostan_terms_of_", block$name, "] int nicostan_term_indices_of_",
                           block$name, " = linspaced_int_array(nicostan_terms_of_", block$name, ", 1, ",
                           "max(1, nicostan_terms_of_", block$name, "));"))
              ## (an array of matrices: the rows and columns of each of its matrices)
              if (block$shape == "array_of_matrices") {
                    transformed_data <-  c(transformed_data,
                          paste0("    int nicostan_rows_per_term_of_", block$name, " = ",
                                 rows_and_columns_of_each_block[[block$name]][1], ";"),
                          paste0("    int nicostan_columns_per_term_of_", block$name, " = ",
                                 rows_and_columns_of_each_block[[block$name]][2], ";"))
              }
        }
        last_block <-  nuisance_blocks[nrow(nuisance_blocks), ]
        transformed_data <-  c(transformed_data,
              paste0("    int nicostan_size_of_the_nuisance_block = nicostan_offset_of_", last_block$name,
                     " + nicostan_terms_of_", last_block$name, " * nicostan_values_per_term_of_",
                     last_block$name, ";"))
        transformed_data <-  paste0("\n    // (the nuisance blocks held by NicoStan: ",
                                    name_of_the_method_in_the_written_comments, ")\n",
                                    paste(transformed_data, collapse = "\n"), "\n")
        if (is.null(fn_blocks(text)[["transformed data"]])) {
              data_block <-  fn_blocks(text)[["data"]]
              if (is.null(data_block)) stop("nico_reduce_sum: the model has no data block")
              text <-  paste0(substr(text, 1, data_block[["closing"]]), "\ntransformed data {\n}",
                              substring(text, data_block[["closing"]] + 1))
        }
        transformed_data_block <-  fn_blocks(text)[["transformed data"]]
        text <-  paste0(substr(text, 1, transformed_data_block[["closing"]] - 1), transformed_data,
                        substring(text, transformed_data_block[["closing"]]))
        ##
        ## ---- 3. every other use of a nuisance block (outside the parameters and functions blocks): read where
        ##      it is used. The probes keep the blocks declared, so that only the use probed changes:
        ##
        if (!fn_stanc_accepts(text)) {
              stop(paste0("nico_reduce_sum: stanc rejects the model with NicoStan's readers declared and its ",
                          "reduce_sum calls rewritten (before the other uses are read): ", Stan_model_file_path,
                          "\n",
                          last_message_of_stanc))
        }
        ##
        ## ---- 2b. each reduce_sum that passes a nuisance block WHOLE as one of its shared arguments (e.g.
        ##      reduce_sum(partial_sum_lpdf, y, chunk_size, alpha, theta), the units of y being the rows of
        ##      theta): reduce_sum would copy the whole block into every chunk, an N-sized copy outside the
        ##      chunks, so the block is no longer passed; a copy of the partial-sum function,
        ##      <function>_nicostan_reads_<block>, takes the block's sizes, offset, bounds and the readers' type
        ##      in its place, and every use of the block in it reads only the values that use indexes, on the
        ##      chunk's own tape (theta[start:end], theta[start:end, 1], C_raw[c][start:end, ], ...;
        ##      fn_candidates_of_a_use_in_a_chunk(); a use the readers cannot index reads the whole block, on the
        ##      chunk's own tape). The log-Jacobian of a constrained block passed so is added by a reduce_sum of
        ##      its own over the block's terms, with the same chunk_size, just before the statement of that
        ##      reduce_sum, when that statement adds its value to the target as it is, at the top level of the
        ##      model block (else by jacobian += in the transformed parameters block, see below):
        ##
        fn_range_of_an_index <-  function(index,
                                          size) {

              index <-  trimws(index)
              if (!nzchar(index) || (index == ":")) return(list(first = "1", last = size, drops = FALSE))
              sides <-  trimws(fn_split_at_top_level(index, ":"))
              if (length(sides) == 2) {
                    return(list(first  = if (nzchar(sides[1])) sides[1] else "1",
                                last   = if (nzchar(sides[2])) sides[2] else size,
                                drops  = FALSE))
              }
              return(list(first = index, last = index, drops = TRUE))

        }
        fn_candidates_of_a_use_in_a_chunk <-  function(code,
                                                       use,
                                                       name_of_the_use,
                                                       block) {

              after <-  use + nchar(name_of_the_use)
              whole <-  list(text = fn_read(block = block, what = "whole", index = NULL), end_of_use = after - 1)
              rest <-  substring(code, after)
              if (!grepl(pattern = "^\\s*\\[", x = rest)) return(list(whole))
              opening <-  after + regexpr(pattern = "\\[", text = rest) - 1
              closing <-  fn_position_of_closing_bracket(code, opening)
              index_text <-  substr(code, opening + 1, closing - 1)
              indices <-  trimws(fn_split_at_top_level(index_text, ","))
              ## (a second bracket right after the first, as in C_raw[c][start:end, ])
              second <-  NULL
              rest_after_the_first <-  substring(code, closing + 1)
              if (grepl(pattern = "^\\s*\\[", x = rest_after_the_first)) {
                    opening_of_the_second <-  closing + regexpr(pattern = "\\[", text = rest_after_the_first)
                    closing_of_the_second <-  fn_position_of_closing_bracket(code, opening_of_the_second)
                    second <-  trimws(fn_split_at_top_level(substr(code, opening_of_the_second + 1,
                                                                   closing_of_the_second - 1), ","))
              }
              b <-  block$name
              offset <-  paste0("nicostan_offset_of_", b)
              terms <-  paste0("nicostan_terms_of_", b)
              values_per_term <-  paste0("nicostan_values_per_term_of_", b)
              candidates <-  list()
              fn_indexed <-  function(call,
                                      original_index,
                                      end_of_use) {

                    return(list(text        = fn_read(block = block, what = "indexed",
                                                          index = c(original_index, call)),
                                end_of_use  = end_of_use))
              }

              ## (a block of rows and columns of one matrix: the matrix block, or one matrix of an array of
              ##  matrices indexed by its term and then its rows and columns)
              rows_and_columns <-  NULL
              if (block$shape == "matrix" && length(indices) %in% c(1, 2)) {
                    rows_and_columns <-  indices
                    rows <-  values_per_term
                    columns <-  terms
                    offset_of_the_matrix <-  offset
                    end_of_use <-  closing
              }
              if (block$shape == "array_of_matrices" && ((length(indices) == 3) ||
                                                         ((length(indices) == 1) && !is.null(second)))) {
                    rows_and_columns <-  if (length(indices) == 3) indices[2:3] else second
                    rows <-  paste0("nicostan_rows_per_term_of_", b)
                    columns <-  paste0("nicostan_columns_per_term_of_", b)
                    offset_of_the_matrix <-  paste0(offset, " + (", indices[1], " - 1) * ", values_per_term)
                    end_of_use <-  if (length(indices) == 3) closing else closing_of_the_second
              }
              if (length(rows_and_columns) %in% c(1, 2)) {
                    if (length(rows_and_columns) == 1) rows_and_columns <-  c(rows_and_columns, "")
                    range_of_rows <-  fn_range_of_an_index(index = rows_and_columns[1], size = rows)
                    range_of_columns <-  fn_range_of_an_index(index = rows_and_columns[2], size = columns)
                    call <-  paste0("nicostan_nuisance_matrix_block(", range_of_rows$first, ", ",
                                    range_of_rows$last, ", ", range_of_columns$first, ", ",
                                    range_of_columns$last, ", ", rows, ", ", offset_of_the_matrix, ", ",
                                    type_of_the_readers, ")")
                    call <-  paste0(call, if (range_of_rows$drops && range_of_columns$drops) "[1, 1]" else
                                          if (range_of_rows$drops) "[1]" else
                                          if (range_of_columns$drops) "[:, 1]" else "")
                    candidates <-  c(candidates, list(fn_indexed(
                                                          call            = call,
                                                          original_index  = substr(code, after, end_of_use),
                                                          end_of_use      = end_of_use)))
              }
              ## (an array of vectors or row_vectors indexed by its terms and then its values)
              if (block$shape %in% c("array_of_vectors", "array_of_row_vectors") && (length(indices) == 2)) {
                    range_of_terms <-  fn_range_of_an_index(index = indices[1], size = terms)
                    call <-  if (range_of_terms$drops) {
                          paste0(fn_read_unconstrained(block = block, what = "one", index = indices[1]), "[",
                                 indices[2], "]")
                    } else {
                          paste0(fn_read_unconstrained(block = block, what = "range",
                                                       index = c(range_of_terms$first, range_of_terms$last)),
                                 "[:, ", indices[2], "]")
                    }
                    candidates <-  c(candidates, list(fn_indexed(call            = call,
                                                                 original_index  = substr(code, after, closing),
                                                                 end_of_use      = closing)))
              }
              ## (one index of any other shape, as the uses outside the chunks: terms a..b, one term, several)
              if ((block$shape != "matrix") && (length(indices) == 1)) {
                    range <-  trimws(fn_split_at_top_level(indices, ":"))
                    if ((length(range) == 2) && all(nzchar(range))) {
                          candidates <-  c(candidates, list(list(
                                                text        = fn_read(block = block, what = "range",
                                                                      index = range),
                                                end_of_use  = closing)))
                    } else if (length(range) == 1) {
                          candidates <-  c(candidates, list(list(
                                                text        = fn_read(block = block, what = "one",
                                                                      index = indices),
                                                end_of_use  = closing)))
                          if (block$shape %in% c("vector", "row_vector", "array_of_reals")) {
                                candidates <-  c(candidates, list(list(
                                      text        = fn_read(block = block, what = "several", index = indices),
                                      end_of_use  = closing)))
                          }
                    }
              }
              return(c(candidates, list(whole)))

        }
        copies_reading_a_shared_nuisance_block <-  c()
        repeat {
              current_blocks <-  fn_blocks(text)
              calls <-  gregexpr(pattern = "(?<![A-Za-z0-9_])reduce_sum(_static)?\\s*\\(", text = text,
                                 perl = TRUE)[[1]]
              call_found <-  FALSE
              for (match in calls[calls > 0]) {
                    ## (a call inside the functions block passes its function's own arguments: not handled here)
                    if (!is.null(current_blocks[["functions"]]) &&
                        (match > current_blocks[["functions"]][["opening"]]) &&
                        (match < current_blocks[["functions"]][["closing"]])) next
                    opening <-  match + regexpr(pattern = "\\(", text = substring(text, match)) - 1
                    closing <-  fn_position_of_closing_bracket(text, opening)
                    arguments <-  trimws(fn_split_at_top_level(substr(text, opening + 1, closing - 1), ","))
                    names_of_the_arguments <-  trimws(fn_without_markers(arguments))
                    positions_of_shared_blocks <-  which((seq_along(arguments) >= 4) &
                                                         (names_of_the_arguments %in% nuisance_params))
                    if (length(positions_of_shared_blocks) == 0) next
                    call_found <-  TRUE
                    position_of_the_block <-  positions_of_shared_blocks[1]
                    break
              }
              if (!call_found) break
              block <-  nuisance_blocks[nuisance_blocks$name == names_of_the_arguments[position_of_the_block], ]
              b <-  block$name
              constraint <-  constraint_of_each_block[[b]]
              partial_sum <-  names_of_the_arguments[1]
              suffix <-  regmatches(partial_sum, regexpr(pattern = "_(lpdf|lupdf|lpmf|lupmf)$",
                                                         text = partial_sum))
              if (length(suffix) == 0) suffix <-  ""
              base_name <-  sub(pattern = "_(lpdf|lupdf|lpmf|lupmf)$", replacement = "", x = partial_sum)
              definition_suffix <-  sub(pattern = "lupdf", replacement = "lpdf",
                                        x = sub(pattern = "lupmf", replacement = "lpmf", x = suffix))
              copy_name <-  paste0(base_name, "_nicostan_reads_", b)
              ##
              ## (the arguments of the copy and of its call in place of the block: its terms, values per term,
              ##  offset, rows and columns of an array of matrices, bounds of a constrained block, and the type)
              types_of_the_whole_blocks <-  c("vector" = "vector", "row_vector" = "row_vector",
                                              "array_of_reals" = "array[] real",
                                              "array_of_vectors" = "array[] vector",
                                              "array_of_row_vectors" = "array[] row_vector", "matrix" = "matrix",
                                              "array_of_matrices" = "array[] matrix")
              type_of_the_whole_block <-  types_of_the_whole_blocks[[block$shape]]
              arguments_of_the_copy <-  paste0(
                    "data int nicostan_terms_of_", b, ", data int nicostan_values_per_term_of_", b,
                    ", data int nicostan_offset_of_", b,
                    if (block$shape == "array_of_matrices") {
                          paste0(", data int nicostan_rows_per_term_of_", b,
                                 ", data int nicostan_columns_per_term_of_", b)
                    } else "",
                    if (constraint$kind > 0) {
                          paste0(", ", if (constraint$first_bound_is_a_container) type_of_the_whole_block else
                                             "real", " nicostan_first_bound_of_", b,
                                 ", ", if (constraint$second_bound_is_a_container) type_of_the_whole_block else
                                             "real", " nicostan_second_bound_of_", b)
                    } else "",
                    ", real nicostan_type_of_", b)
              arguments_of_the_call <-  paste0(
                    "nicostan_terms_of_", b, ", nicostan_values_per_term_of_", b, ", nicostan_offset_of_", b,
                    if (block$shape == "array_of_matrices") {
                          paste0(", nicostan_rows_per_term_of_", b, ", nicostan_columns_per_term_of_", b)
                    } else "",
                    if (constraint$kind > 0) {
                          paste0(", ", constraint$first_bound, ", ", constraint$second_bound)
                    } else "",
                    ", ", type_of_the_readers)
              ##
              ## (the log-Jacobian of a constrained block with real bounds, by a reduce_sum of its own over the
              ##  block's terms, just before the statement of this reduce_sum when it adds its value to the target
              ##  as it is, at the top level of the model block)
              start_of_the_statement <-  NA
              if ((constraint$kind > 0) && !constraint$first_bound_is_a_container &&
                  !constraint$second_bound_is_a_container &&
                  !(b %in% blocks_whose_log_jacobian_is_added_inside_the_chunks) &&
                  fn_reduce_sum_call_adds_its_value_to_the_target_once(code               = text,
                                                                       start_of_the_call  = match,
                                                                       end_of_the_call    = closing)) {
                    ends_of_statements <-  gregexpr(pattern = "[;{}]", text = substr(text, 1, match - 1))[[1]]
                    start_of_the_statement <-  max(ends_of_statements) + 1
              }
              ##
              ## ---- the edits, from the end of the text to its start (so that each position before them stays):
##
              ##      the call, the log-Jacobian's statement, then the copy and the log-Jacobian's function
              ##
              new_call <-  paste0(substr(text, match, opening - 1), "(",
                                  paste(c(paste0(copy_name, suffix), arguments[2:(position_of_the_block - 1)],
                                          arguments_of_the_call,
                                          if (position_of_the_block < length(arguments)) {
                                                arguments[(position_of_the_block + 1):length(arguments)]
                                          }), collapse = ", "), ")")
              text <-  paste0(substr(text, 1, match - 1), new_call, substring(text, closing + 1))
              if (!is.na(start_of_the_statement)) {
                    text <-  paste0(substr(text, 1, start_of_the_statement - 1),
                                    "\n    // (the log-Jacobian of the constraint of the nuisance block ", b,
                                    ", whose unconstrained values NicoStan holds: once per value, inside the ",
                                    "chunks)",
                                    "\n    target += reduce_sum(nicostan_log_jacobian_of_", b, "_lpmf, ",
                                    "nicostan_term_indices_of_", b, ", ", arguments[3],
                                    ", nicostan_values_per_term_of_", b, ", nicostan_offset_of_", b, ", ",
                                    constraint$first_bound, ", ", constraint$second_bound, ", ",
                                    type_of_the_readers, ");",
                                    substring(text, start_of_the_statement))
                    blocks_whose_log_jacobian_is_added_inside_the_chunks <-  c(
                          blocks_whose_log_jacobian_is_added_inside_the_chunks, b)
                    function_of_the_log_jacobian <-  paste0(
                          "\n    real nicostan_log_jacobian_of_", b, "_lpmf(",
                          "data array[] int nicostan_terms_of_the_slice, int start, int end, ",
                          "data int nicostan_values_per_term, data int nicostan_offset, ",
                          "real nicostan_first_bound, real nicostan_second_bound, real nicostan_type) {\n",
                          "        return nicostan_log_jacobian_of_constraint(",
                          "nicostan_nuisance_array_of_vectors(start, end, nicostan_values_per_term, ",
                          "nicostan_offset, ",
                          "nicostan_type), ", constraint$kind,
                          ", nicostan_first_bound, nicostan_second_bound);\n    }\n")
                    closing_of_the_functions_block <-  fn_blocks(text)[["functions"]][["closing"]]
                    text <-  paste0(substr(text, 1, closing_of_the_functions_block - 1),
                                    function_of_the_log_jacobian, substring(text, closing_of_the_functions_block))
              }
              if (copy_name %in% copies_reading_a_shared_nuisance_block) next
              ##
              ## ---- the copy of the partial-sum function, its block's parameter first kept as a local of the
              ##      whole block (so that every use compiles whilst the uses are read one by one):
              ##
              definition <-  regexpr(pattern = paste0("(?<![A-Za-z0-9_])", base_name, definition_suffix,
                                                      "\\s*\\("), text = text, perl = TRUE)
              if (definition == -1) {
                    stop(paste0("nico_reduce_sum: the partial-sum function ", partial_sum, " is not defined ",
                                "in the model"))
              }
              start_of_line <-  definition
              while (start_of_line > 1 && substr(text, start_of_line - 1, start_of_line - 1) != "\n") {
                    start_of_line <-  start_of_line - 1
              }
              header_of_definition <-  substr(text, start_of_line, definition - 1)
              opening_of_arguments <-  definition + attr(definition, "match.length") - 1
              closing_of_arguments <-  fn_position_of_closing_bracket(text, opening_of_arguments)
              opening_of_body <-  regexpr(pattern = "\\{", text = substring(text, closing_of_arguments)) +
                                  closing_of_arguments - 1
              closing_of_body <-  fn_position_of_closing_bracket(text, opening_of_body)
              function_arguments <-  trimws(fn_split_at_top_level(substr(text, opening_of_arguments + 1,
                                                                         closing_of_arguments - 1), ","))
              ## (the call's argument k, after the function, the sliced argument and the chunk_size, is the
              ##  function's argument k, after the slice, start and end)
              parameter_name <-  sub(pattern = "^.*[^A-Za-z0-9_]([A-Za-z0-9_]+)$", replacement = "\\1",
                                     x = trimws(fn_without_markers(function_arguments[position_of_the_block])))
              ## (the reads inside the copy use the copy's own arguments: its type and bounds)
              type_of_the_readers_outside_the_copy <-  type_of_the_readers
              constraint_outside_the_copy <-  constraint_of_each_block[[b]]
              type_of_the_readers <-  paste0("nicostan_type_of_", b)
              if (constraint$kind > 0) {
                    constraint_of_each_block[[b]]$first_bound <-  paste0("nicostan_first_bound_of_", b)
                    constraint_of_each_block[[b]]$second_bound <-  paste0("nicostan_second_bound_of_", b)
              }
              type_of_the_local <-  switch(block$shape,
                    "vector"                = paste0("vector[nicostan_terms_of_", b, "]"),
                    "row_vector"            = paste0("row_vector[nicostan_terms_of_", b, "]"),
                    "array_of_reals"        = paste0("array[nicostan_terms_of_", b, "] real"),
                    "array_of_vectors"      = paste0("array[nicostan_terms_of_", b, "] ",
                                                     "vector[nicostan_values_per_term_of_", b, "]"),
                    "array_of_row_vectors"  = paste0("array[nicostan_terms_of_", b,
                                                     "] row_vector[nicostan_values_per_term_of_", b, "]"),
                    "matrix"                = paste0("matrix[nicostan_values_per_term_of_", b,
                                                     ", nicostan_terms_of_", b, "]"),
                    "array_of_matrices"     = paste0("array[nicostan_terms_of_", b, "] ",
                                                     "matrix[nicostan_rows_per_term_of_", b,
                                                     ", nicostan_columns_per_term_of_", b, "]"))
              local_of_the_whole_block <-  paste0("\n        ", type_of_the_local, " ", parameter_name, " = ",
                                                  fn_read(block = block, what = "whole", index = NULL), ";")
              copy <-  paste0("\n", header_of_definition, copy_name, definition_suffix, "(",
                              paste(c(function_arguments[seq_len(position_of_the_block - 1)],
                                      arguments_of_the_copy,
                                      if (position_of_the_block < length(function_arguments)) {
                                            function_arguments[(position_of_the_block + 1):
                                                               length(function_arguments)]
                                      }), collapse = ", "),
                              ") {\n        // (", parameter_name, ": the nuisance block ", b,
                              ", held by NicoStan, ",
                              "read where it is used, on the chunk's own tape)", local_of_the_whole_block,
                              substr(text, opening_of_body + 1, closing_of_body), "\n")
              text <-  paste0(substr(text, 1, closing_of_body), copy, substring(text, closing_of_body + 1))
              copies_reading_a_shared_nuisance_block <-  c(copies_reading_a_shared_nuisance_block, copy_name)
              if (!fn_stanc_accepts(text)) {
                    stop(paste0("nico_reduce_sum: stanc rejects the copy ", copy_name, " that reads the ",
                                "shared nuisance block ", b, ":\n", last_message_of_stanc))
              }
              ##
              ## ---- each use of the block's parameter in the copy (after its local) read where it is used:
              ##
              repeat {
                    start_of_the_copy <-  regexpr(pattern = paste0("(?<![A-Za-z0-9_])", copy_name,
                                                                   definition_suffix, "\\s*\\("),
                                                  text = text, perl = TRUE)
                    opening_of_the_body_of_the_copy <-  regexpr(pattern = "\\{",
                                                                text = substring(text, start_of_the_copy)) +
                                                        start_of_the_copy - 1
                    closing_of_the_body_of_the_copy <-  fn_position_of_closing_bracket(
                                                                text, opening_of_the_body_of_the_copy)
                    end_of_the_local <-  regexpr(pattern = local_of_the_whole_block,
                                                 text = substring(text, opening_of_the_body_of_the_copy),
                                                 fixed = TRUE) + opening_of_the_body_of_the_copy - 1 +
                                         nchar(local_of_the_whole_block) - 1
                    uses <-  gregexpr(pattern = paste0("(?<![A-Za-z0-9_.])", parameter_name, "(?![A-Za-z0-9_(])"),
                                      text = text, perl = TRUE)[[1]]
                    uses <-  uses[(uses > end_of_the_local) & (uses < closing_of_the_body_of_the_copy)]
                    if (length(uses) == 0) break
                    use <-  uses[1]
                    candidates <-  fn_candidates_of_a_use_in_a_chunk(code             = text,
                                                                     use              = use,
                                                                     name_of_the_use  = parameter_name,
                                                                     block            = block)
                    chosen <-  NULL
                    for (candidate in candidates) {
                          probe <-  paste0(substr(text, 1, use - 1), "(", candidate$text, ")",
                                           substring(text, candidate$end_of_use + 1))
                          if (fn_stanc_accepts(probe)) {
                                chosen <-  candidate
                                break
                          }
                    }
                    if (is.null(chosen)) {
                          stop(paste0("nico_reduce_sum: stanc accepts none of NicoStan's readers for the use ",
                                      "of ", parameter_name, " (the nuisance block ", b, ") in ", copy_name, ": ",
                                      fn_restore(substr(text, use, use + nchar(parameter_name) + 40))))
                    }
                    text <-  paste0(substr(text, 1, use - 1), "(", chosen$text, ")",
                                    substring(text, chosen$end_of_use + 1))
              }
              ## (the local of the whole block, no longer used, out of the copy)
              position_of_the_local <-  regexpr(pattern = local_of_the_whole_block, text = text, fixed = TRUE)
              text <-  paste0(substr(text, 1, position_of_the_local - 1),
                              substring(text, position_of_the_local + nchar(local_of_the_whole_block)))
              type_of_the_readers <-  type_of_the_readers_outside_the_copy
              constraint_of_each_block[[b]] <-  constraint_outside_the_copy
              message(colourise(paste0("nico_reduce_sum: the nuisance block ", b, ", a shared argument of ",
                                       partial_sum, ", is read inside the chunks by ", copy_name), "cyan"))
        }
        ## (a constrained block that no reduce_sum slices: its uses in the model block read one local of the
        ##  whole block, <block>_nicostan_constrained, read and constrained once at the start of the model block,
        ##  so that the transform is not made again at every use)
        blocks_sliced_by_a_reduce_sum <-  unique(sub(pattern = "^.*_nicostan_chunked_of_", replacement = "",
                                                     x = copies_of_partial_sum_functions))
        blocks_read_once_in_the_model_block <-  c()
        for (block_name in nuisance_params) {
              block <-  nuisance_blocks[nuisance_blocks$name == block_name, ]
              read_once_in_the_model_block <-  (constraint_of_each_block[[block_name]]$kind > 0) &&
                                               !(block_name %in% blocks_sliced_by_a_reduce_sum)
              repeat {
                    current_blocks <-  fn_blocks(text)
                    uses <-  gregexpr(pattern = paste0("(?<![A-Za-z0-9_.])", block_name, "(?![A-Za-z0-9_(])"),
                                      text = text, perl = TRUE)[[1]]
                    uses <-  uses[uses > 0]
                    fn_in_block <-  function(position, name) {

                          !is.null(current_blocks[[name]]) && position > current_blocks[[name]][["opening"]] &&
                          position < current_blocks[[name]][["closing"]]
                    }

                    uses <-  uses[!vapply(uses, function(position) fn_in_block(position, "parameters") ||
                                                                     fn_in_block(position, "functions"),
                                          logical(1))]
                    if (length(uses) == 0) break
                    use <-  uses[1]
                    if (read_once_in_the_model_block && fn_in_block(use, "model")) {

                          text <-  paste0(substr(text, 1, use - 1), block_name, "_nicostan_constrained",
                                          substring(text, use + nchar(block_name)))
                          blocks_read_once_in_the_model_block <-  union(blocks_read_once_in_the_model_block,
                                                                        block_name)
                          next
                    }

                    after <-  use + nchar(block_name)
                    rest <-  substring(text, after)
                    candidates <-  c()
                    end_of_use <-  after - 1
                    if (block$shape != "matrix" && grepl(pattern = "^\\s*\\[", x = rest)) {
                          opening <-  after + regexpr(pattern = "\\[", text = rest) - 1
                          closing <-  fn_position_of_closing_bracket(text, opening)
                          indices <-  trimws(fn_split_at_top_level(substr(text, opening + 1, closing - 1), ","))
                          if (length(indices) == 1) {
                                range <-  trimws(fn_split_at_top_level(indices, ":"))
                                if (length(range) == 2 && all(nzchar(range))) {
                                      candidates <-  fn_read(block, "range", range)
                                } else if (length(range) == 1) {
                                      candidates <-  c(fn_read(block, "one", indices),
                                                       if (block$shape %in% c("vector", "row_vector",
                                                                              "array_of_reals"))
                                                             fn_read(block, "several", indices))
                                }
                                if (length(candidates) > 0) end_of_use <-  closing
                          }
                    }
                    if (length(candidates) == 0) {
                          candidates <-  fn_read(block = block, what = "whole", index = NULL)
                    }
                    chosen <-  NA_character_
                    for (candidate in candidates) {
                          probe <-  paste0(substr(text, 1, use - 1), "(", candidate, ")",
                                           substring(text, end_of_use + 1))
                          if (length(candidates) == 1 || fn_stanc_accepts(probe)) {
                                chosen <-  candidate
                                break
                          }
                    }
                    if (is.na(chosen)) {
                          stop(paste0("nico_reduce_sum: stanc accepts none of NicoStan's readers for the use ",
                                      "of ",
                                      block_name, ": ", fn_restore(substr(text, use, end_of_use))))
                    }
                    text <-  paste0(substr(text, 1, use - 1), "(", chosen, ")", substring(text, end_of_use + 1))
              }
        }
        ##
        ## ---- the log-Jacobian of each constrained block that no reduce_sum adds inside the chunks: Stan's own
        ##      jacobian += at the start of the transformed parameters block (added only when the log density is
        ##      asked for with its Jacobian, as for Stan's own constrained parameters), once per value, from the
        ##      block's unconstrained values read in Stan's order (on the main tape):
        ##
        log_jacobians_in_the_transformed_parameters <-  c()
        for (i in seq_len(nrow(nuisance_blocks))) {
              block <-  nuisance_blocks[i, ]
              constraint <-  constraint_of_each_block[[block$name]]
              if (constraint$kind == 0) next
              if (block$name %in% blocks_whose_log_jacobian_is_added_inside_the_chunks) {
                    message(colourise(paste0("nico_reduce_sum: the log-Jacobian of the constraint of ",
                                             block$name, " is added inside the chunks, by the reduce_sum that ",
                                             "slices it"), "cyan"))
                    next
              }
              message(colourise(paste0("nico_reduce_sum: the log-Jacobian of the constraint of ", block$name,
                                       " is added by jacobian += at the start of the transformed parameters ",
                                       "block (no reduce_sum that slices it adds its value to the target as it ",
                                       "is, at the top level of the model block)"), "cyan"))
              log_jacobians_in_the_transformed_parameters <-  c(log_jacobians_in_the_transformed_parameters,
                                                                paste0(
                    "\n    // (the log-Jacobian of the constraint of the nuisance block ", block$name,
                    ", whose unconstrained values NicoStan holds: once per value)\n",
                    "    jacobian += nicostan_log_jacobian_of_constraint(\n",
                    "                    ",
                    ## (bounds of one value per value of the block: the block read whole, in its own type)
                    if (constraint$first_bound_is_a_container || constraint$second_bound_is_a_container) {
                          fn_read_unconstrained(block = block, what = "whole", index = NULL)
                    } else {
                          paste0("nicostan_nuisance_array_of_reals(1, nicostan_terms_of_", block$name,
                                 " * nicostan_values_per_term_of_", block$name, ",\n",
                                 "                                                     nicostan_offset_of_",
                                 block$name, ", ", type_of_the_readers, ")")
                    },
                    ",\n",
                    "                    ", constraint$kind, ", ", constraint$first_bound, ", ",
                    constraint$second_bound, ");\n"))
        }
        if (length(log_jacobians_in_the_transformed_parameters) > 0) {
              text <-  fn_with_statements_at_the_start_of_the_transformed_parameters_block(
                             code        = text,
                             statements  = paste(log_jacobians_in_the_transformed_parameters, collapse = ""))
        }
        ##
        ## ---- the local of each constrained block read once in the model block (see the uses above): the whole
        ##      block, read and constrained at the start of the model block:
        ##
        locals_of_the_model_block <-  c()
        for (block_name in blocks_read_once_in_the_model_block) {
              block <-  nuisance_blocks[nuisance_blocks$name == block_name, ]
              terms <-  paste0("nicostan_terms_of_", block_name)
              values_per_term <-  paste0("nicostan_values_per_term_of_", block_name)
              type_of_the_block <-  switch(block$shape,
                    "vector"                = paste0("vector[", terms, "]"),
                    "row_vector"            = paste0("row_vector[", terms, "]"),
                    "array_of_reals"        = paste0("array[", terms, "] real"),
                    "array_of_vectors"      = paste0("array[", terms, "] vector[", values_per_term, "]"),
                    "array_of_row_vectors"  = paste0("array[", terms, "] row_vector[", values_per_term, "]"),
                    "matrix"                = paste0("matrix[", values_per_term, ", ", terms, "]"),
                    "array_of_matrices"     = paste0("array[", terms, "] matrix[nicostan_rows_per_term_of_",
                                                     block_name, ", nicostan_columns_per_term_of_", block_name,
                                                     "]"))
              locals_of_the_model_block <-  c(locals_of_the_model_block, paste0(
                    "\n    // (the nuisance block ", block_name, ", held by NicoStan, read and constrained ",
                    "once)\n",
                    "    ", type_of_the_block, " ", block_name, "_nicostan_constrained = ",
                    fn_read(block = block, what = "whole", index = NULL), ";\n"))
        }
        if (length(locals_of_the_model_block) > 0) {
              text <-  fn_with_statements_at_the_start_of_the_model_block(
                             code        = text,
                             statements  = paste(locals_of_the_model_block, collapse = ""))
        }
        ##
        ## ---- 1. the nuisance blocks out of the parameters block (each declaration kept as a comment), and 4.
        ##      the size of the nuisance block in the generated quantities:
        ##
        parameters_block <-  fn_blocks(text)[["parameters"]]
        parameters_body <-  substr(text, parameters_block[["opening"]] + 1, parameters_block[["closing"]] - 1)
        statements <-  fn_split_at_top_level(parameters_body, ";")
        for (k in seq_len(length(statements) - 1)) {
              if (fn_declaration(statements[k])$name %in% nuisance_params) {
                    statements[k] <-  paste0(sub(pattern = "^(\\s*).*$", replacement = "\\1",
                                                 x = fn_without_markers(statements[k])),
                                             paste0("// (held by NicoStan, ",
                                                    name_of_the_method_in_the_written_comments, ":) "),
                                             gsub(pattern = "\\s+", replacement = " ",
                                                  x = trimws(fn_without_markers(statements[k]))), "\n    //")
              }
        }
        parameters_body <-  paste(statements, collapse = ";")
        text <-  paste0(substr(text, 1, parameters_block[["opening"]]), parameters_body,
                        substring(text, parameters_block[["closing"]]))
        if (is.null(fn_blocks(text)[["generated quantities"]])) {
              text <-  paste0(text, "\ngenerated quantities {\n}\n")
        }
        generated_quantities_block <-  fn_blocks(text)[["generated quantities"]]
        text <-  paste0(substr(text, 1, generated_quantities_block[["closing"]] - 1),
                        "    // (the size of the nuisance block held by NicoStan, read by NicoStan: reduce_sum_",
                        "chunked)\n    int nicostan_size_of_the_nuisance_block_held_by_NicoStan = ",
                        "nicostan_size_of_the_nuisance_block;\n",
                        substring(text, generated_quantities_block[["closing"]]))
        if (!fn_stanc_accepts(text)) {
              kept <-  sub("\\.stan$", "_rejected_by_stanc.stan", nico_reduce_sum_Stan_model_file_path)
              writeLines(text = fn_restore(text), con = kept)
              stop(paste0("nico_reduce_sum: stanc rejects the nico_reduce_sum version of ",
                          Stan_model_file_path, " (kept for inspection: ", kept, ")"))
        }
        ##
        ## ---- the version and its user header (NicoStan's readers, then the model's own header):
        ##
        ## (each file written only when its lines change, so that a library built from it stays newer than it)
        ## (the text compared byte for byte with the file as writeLines() writes it: each element and a newline)
        fn_write_lines_when_they_change <-  function(lines,
                                                     file) {

              text_of_the_file <-  paste0(paste(lines, collapse = "\n"), "\n")
              if (file.exists(file) && identical(readChar(con = file, nchars = file.info(file)$size,
                                                          useBytes = TRUE), text_of_the_file)) {
                    return(invisible(FALSE))
              }
              writeLines(text = lines, con = file)
              return(invisible(TRUE))
        }

        ## (each reduce_sum_static call this writer rewrote, i.e. whose partial sum is its _nicostan_chunked_of_
        ##  copy, calls NicoStan's name, nico_reduce_sum_static; every other call is left as the model has it;
        ##  R_fn_Stan_parallel_call_names gives stanc Stan's name back, so the C++ is the same)
        text_with_the_name_of_NicoStan <-  gsub(
              pattern = paste0("(?<![A-Za-z0-9_])reduce_sum_static",
                               "(\\s*\\(\\s*[A-Za-z_][A-Za-z0-9_]*_nicostan_chunked_of_)"),
              replacement = "nico_reduce_sum_static\\1",
              x = text,
              perl = TRUE)
        # fn_write_lines_when_they_change(lines  = c(paste0("// ", name_of_the_method_in_the_written_comments,
        #                                                   " version of ", basename(Stan_model_file_path),
        #                                                   ", written by NicoStan: its nuisance blocks (",
        #                                                   paste(nuisance_params, collapse = ", "),
        #                                                   ") are held by NicoStan inside the chunks"),
        #                                            fn_restore(text)),
        #                                 file   = nico_reduce_sum_Stan_model_file_path)
        fn_write_lines_when_they_change(lines  = c(paste0("// ", name_of_the_method_in_the_written_comments,
                                                          " version of ", basename(Stan_model_file_path),
                                                          ", written by NicoStan: its nuisance blocks (",
                                                          paste(nuisance_params, collapse = ", "),
                                                          ") are held by NicoStan inside the chunks"),
                                                   fn_restore(text_with_the_name_of_NicoStan)),
                                        file   = nico_reduce_sum_Stan_model_file_path)
        # readers_header <-  system.file("include", "NicoStan", "runtime", "general_functions",
        #                                "stan_external_nuisance_values_of_the_block_owned_by_NicoStan.hpp",
        #                                package = "NicoStan", mustWork = TRUE)
        ## (the readers' header of the one NicoStan install, in the R installation's own libraries - never a side
        readers_header <-  fn_file_of_the_one_NicoStan_install(
              "include", "NicoStan", "runtime", "general_functions",
              "stan_external_nuisance_values_of_the_block_owned_by_NicoStan.hpp",
              package = "NicoStan", mustWork = TRUE)
        user_header <-  sub("\\.stan$", "_user_header.hpp", nico_reduce_sum_Stan_model_file_path)
        fn_write_lines_when_they_change(lines  = c("#pragma once",
                                                   paste0("#include \"", readers_header, "\""),
                                                   if (!is.null(Stan_cpp_user_header)) {
                                                         paste0("#include \"",
                                                                normalizePath(Stan_cpp_user_header,
                                                                              mustWork = TRUE), "\"")
                                                   }),
                                        file   = user_header)
        ##
        ## ---- the sizes of the nuisance blocks in nuisance_blocks, written in terms of the data: every name in
        ##      a size that is an integer of the transformed data, declared with its value in one expression
        ##      (e.g. int M = M_per_dimension * M_per_dimension;) and never assigned again, is replaced by that
        ##      expression, in brackets, again and again until only the data's own names are left, so that
        ##      fn_size_of_the_nuisance_blocks_held_by_NicoStan_from_the_data() works the sizes out in the data
        ##      alone. The Stan code of the version keeps the model's own expressions (its transformed data
        ##      have them). A size that cannot be written in terms of the data stops the writer, naming it.
        ##
        names_of_the_data <-  vapply(Filter(function(statement) nzchar(trimws(statement)),
                                            fn_split_at_top_level(data_body_of_the_model, ";")),
                                     function(statement) fn_declaration(statement)$name, character(1))
        fn_in_terms_of_the_data <-  function(expression, name_of_block) {

              for (depth in 1:50) {
                    ## (the names in the expression, those of functions, e.g. max(N, 1), left out)
                    names_used <-  regmatches(expression,
                                              gregexpr(pattern = "\\b[A-Za-z][A-Za-z0-9_]*\\b(?!\\s*\\()",
                                                       text = expression, perl = TRUE))[[1]]
                    names_to_replace <-  setdiff(unique(names_used), names_of_the_data)
                    if (length(names_to_replace) == 0) return(expression)
                    for (name in names_to_replace) {
                          declaration_pattern <-  paste0("\\bint\\s*(<[^;]*?>)?\\s+", name, "\\s*=\\s*([^;]+);")
                          declaration <-  regmatches(transformed_data_body_of_the_model,
                                                     regexec(pattern = declaration_pattern,
                                                             text = transformed_data_body_of_the_model,
                                                             perl = TRUE))[[1]]
                          n_assignments <-  lengths(regmatches(transformed_data_body_of_the_model,
                                                gregexpr(pattern = paste0("\\b", name, "\\s*(\\[[^]]*\\])?\\s*",
                                                                          "([-+*/]?=)(?!=)"),
                                                         text = transformed_data_body_of_the_model,
                                                         perl = TRUE)))
                          # if (length(declaration) == 0 || n_assignments != 1) {
                          #   stop(paste0("nico_reduce_sum: the size of the nuisance block ", name_of_block,
                          #               " uses ", name, ", which is neither a variable of the data block ",
                          #               "nor an integer of the transformed data declared with its value in ",
                          #               "one expression and never assigned again; it cannot be written in ",
                          #               "terms of the data (size so far: ", expression, ")"))
                          # }
                          ## (a size that uses a transformed-data integer set in any other way, e.g. in a loop,
                          ##  cannot be written in terms of the data: NA, and the size of the block is then read
                          ##  from the compiled model, whose generated quantities report it:
                          ##  fn_size_of_the_nuisance_blocks_held_by_NicoStan_from_the_compiled_model())
                          if (length(declaration) == 0 || n_assignments != 1) {
                                message(colourise(paste0("nico_reduce_sum: the size of the nuisance block ",
                                                         name_of_block, " uses ", name, ", which is set in the ",
                                                         "transformed data in a way that cannot be written in ",
                                                         "terms of the data (size so far: ", expression, "); ",
                                                         "its size is read from the compiled model"),
                                                  "cyan"))
                                return(NA_character_)
                          }
                          expression <-  gsub(pattern = paste0("\\b", name, "\\b(?!\\s*\\()"),
                                              replacement = paste0("(", trimws(declaration[3]), ")"),
                                              x = expression, perl = TRUE)
                    }
              }
              stop(paste0("nico_reduce_sum: the size of the nuisance block ", name_of_block, " is not ",
                          "written in terms of the data after 50 replacements of transformed-data integers ",
                          "(size so far: ", expression, ")"))

        }
        # for (i in seq_len(nrow(nuisance_blocks))) {
        #       nuisance_blocks$terms[i] <-  fn_in_terms_of_the_data(nuisance_blocks$terms[i],
        #                                                            nuisance_blocks$name[i])
        #       nuisance_blocks$values_per_term[i] <-  fn_in_terms_of_the_data(nuisance_blocks$values_per_term[i],
        #                                                                      nuisance_blocks$name[i])
        # }
        ## (a block whose size cannot be written in terms of the data keeps the model's own expressions and is
        ##  marked size_read_from_the_compiled_model = TRUE; the column exists only when some block is marked,
        ##  so that every other version's nuisance_blocks is as before)
        for (i in seq_len(nrow(nuisance_blocks))) {
              terms_in_terms_of_the_data <-  fn_in_terms_of_the_data(nuisance_blocks$terms[i],
                                                                     nuisance_blocks$name[i])
              values_per_term_in_terms_of_the_data <-  fn_in_terms_of_the_data(nuisance_blocks$values_per_term[i],
                                                                               nuisance_blocks$name[i])
              if (is.na(terms_in_terms_of_the_data) || is.na(values_per_term_in_terms_of_the_data)) {
                    if (is.null(nuisance_blocks$size_read_from_the_compiled_model)) {
                          nuisance_blocks$size_read_from_the_compiled_model <-  FALSE
                    }
                    nuisance_blocks$size_read_from_the_compiled_model[i] <-  TRUE
                    next
              }
              nuisance_blocks$terms[i] <-  terms_in_terms_of_the_data
              nuisance_blocks$values_per_term[i] <-  values_per_term_in_terms_of_the_data
        }
        ## (the constraint of each block, and the rows and columns of each matrix of an array of matrices: columns
        ##  that exist only when some block has them, so that every other version's nuisance_blocks is as before)
        if (any(kinds_of_the_constraints > 0)) {
              nuisance_blocks$constraint <-  vapply(nuisance_blocks$name, function(name)
                                                    constraint_of_each_block[[name]]$name_of_kind, character(1),
                                                    USE.NAMES = FALSE)
              nuisance_blocks$first_bound <-  vapply(nuisance_blocks$name, function(name)
                                                     constraint_of_each_block[[name]]$first_bound, character(1),
                                                     USE.NAMES = FALSE)
              nuisance_blocks$second_bound <-  vapply(nuisance_blocks$name, function(name)
                                                      constraint_of_each_block[[name]]$second_bound, character(1),
                                                      USE.NAMES = FALSE)
        }
        if (any(nuisance_blocks$shape == "array_of_matrices")) {

              nuisance_blocks$rows_per_term <-  vapply(nuisance_blocks$name, function(name)
                    if (is.null(rows_and_columns_of_each_block[[name]])) NA_character_ else
                          rows_and_columns_of_each_block[[name]][1], character(1), USE.NAMES = FALSE)
              nuisance_blocks$columns_per_term <-  vapply(nuisance_blocks$name, function(name)
                    if (is.null(rows_and_columns_of_each_block[[name]])) NA_character_ else
                          rows_and_columns_of_each_block[[name]][2], character(1), USE.NAMES = FALSE)
        }

        result_of_the_writer <-  list(
              Stan_model_file_path  = nico_reduce_sum_Stan_model_file_path,
              Stan_cpp_user_header  = user_header,
              unit_upstream_proof   = R_fn_bridge_memory_proof_of_unit_upstream(
                                            model_file = nico_reduce_sum_Stan_model_file_path,
                                            make_args = paste0("USER_HEADER=", user_header)),
              nuisance_blocks       = nuisance_blocks)
        ##
        ## ---- the record of this writing (written to a temporary file and renamed, so that a fit reading it at
        ##      the same moment never reads half of it):
        ##
        temporary_file_of_the_record <-  tempfile(pattern = "record_of_the_writer_",
                                                  tmpdir = dirname(file_of_the_record),
                                                  fileext = ".RDS")
        saveRDS(object = list(
                      key                     = key_of_this_version,
                      md5_of_the_version      = unname(tools::md5sum(
                                                      files = nico_reduce_sum_Stan_model_file_path)),
                      md5_of_the_version_without_its_E5_cell_label =
                            fn_md5_of_the_version_without_its_E5_cell_label(
                                  file = nico_reduce_sum_Stan_model_file_path),
                      md5_of_the_user_header  = unname(tools::md5sum(files = user_header)),
                      result                  = result_of_the_writer),
                file = temporary_file_of_the_record)
        file.rename(from = temporary_file_of_the_record, to = file_of_the_record)
        ##
        return(result_of_the_writer)

}





#' A file of the one NicoStan install, in the R installation's own libraries
#'
#' As system.file(), but always in the R installation's own libraries (.Library.site, .Library and R_LIBS_USER),
#' never in a side library put first in .libPaths(), so that every header NicoStan writes includes the one
#'
#' @param ... The parts of the file's path inside the package, as for system.file().
#' @param package The package ("NicoStan").
#' @param mustWork TRUE: stop when the file does not exist.
#' @return The full path of the file.
#' @keywords internal
fn_file_of_the_one_NicoStan_install <-  function(..., package, mustWork) {

        libraries_of_the_R_installation <-  unique(c(.Library.site, .Library,
                                                     path.expand(strsplit(x = Sys.getenv("R_LIBS_USER"),
                                                                          split = .Platform$path.sep)[[1]])))
        the_one_install <-  find.package(package = package, lib.loc = libraries_of_the_R_installation)
        file <-  file.path(the_one_install, ...)
        if (isTRUE(mustWork) && !file.exists(file)) {
              stop(paste0("no file ", file, " in the one ", package, " install (", the_one_install, ")"))
        }
        return(file)

}





#' The number of nuisance values held by NicoStan for a nico_reduce_sum model
#'
#' Each nuisance block's number of terms times its values per term (both Stan expressions of the data, as
#' R_fn_write_nico_reduce_sum_version_of_Stan_model() returns them in nuisance_blocks), worked out in R from
#' Stan_data_list (the writer has already written every transformed-data integer of a size in terms of the data);
#' stops, naming the block, when an expression cannot be worked out from the data or is not a whole number, so
#' that the size is then given by hand (n_nuisance_override).
#'
#' @param nuisance_blocks The nuisance_blocks of R_fn_write_nico_reduce_sum_version_of_Stan_model().
#' @param Stan_data_list The model's data.
#' @return The number of nuisance values, summed over the blocks.
#' @keywords internal
fn_size_of_the_nuisance_blocks_held_by_NicoStan_from_the_data <-  function( nuisance_blocks,
                                                                            Stan_data_list) {

        ## (a block whose size the writer could not write in terms of the data: read from the compiled model)
        if (isTRUE(any(nuisance_blocks$size_read_from_the_compiled_model))) {
              stop(paste0("nico_reduce_sum: the size of the nuisance block(s) ",
                          paste(nuisance_blocks$name[nuisance_blocks$size_read_from_the_compiled_model],
                                collapse = ", "),
                          " is set in the transformed data in a way that cannot be written in terms of the ",
                          "data: read it from the compiled model with ",
                          "fn_size_of_the_nuisance_blocks_held_by_NicoStan_from_the_compiled_model()"))
        }
        size_of_every_block <-  0
        for (i in seq_len(nrow(nuisance_blocks))) {
              size_of_block <-  tryCatch(
                    eval(parse(text = nuisance_blocks$terms[i]), envir = Stan_data_list) *
                    eval(parse(text = nuisance_blocks$values_per_term[i]), envir = Stan_data_list),
                    error = function(e) NA_real_)
              # if (!is.numeric(size_of_block) || length(size_of_block) != 1 || is.na(size_of_block)) {
              ## (a whole number only: Stan's / between integers rounds towards zero, R's does not)
              if (!is.numeric(size_of_block) || length(size_of_block) != 1 || is.na(size_of_block) ||
                  size_of_block != round(size_of_block)) {

                    stop(paste0("nico_reduce_sum: the size of the nuisance block ", nuisance_blocks$name[i],
                                " (", nuisance_blocks$terms[i], " x ", nuisance_blocks$values_per_term[i],
                                ") could not be worked out from Stan_data_list; give the size of all the ",
                                "nuisance blocks with n_nuisance_override."))
              }

              size_of_every_block <-  size_of_every_block + size_of_block
        }
        return(size_of_every_block)

}





#' The number of nuisance values held by NicoStan for a nico_reduce_sum model, read from its compiled library
#'
#' For a model whose nuisance block size is set in its transformed data in a way the writer cannot write in
#' terms of the data (nuisance_blocks$size_read_from_the_compiled_model, e.g. a size from a flag set in a loop):
#' the compiled library of the nico_reduce_sum version, constructed with the data, reports the size of the
#' whole nuisance block in its generated quantities (nicostan_size_of_the_nuisance_block_held_by_NicoStan, written
#' by R_fn_write_nico_reduce_sum_version_of_Stan_model()), read with BridgeStan's param_constrain at the main
#' parameters 0 on the unconstrained scale. Transformed parameters or generated quantities that read the nuisance
#' block are given nuisance values of 0 (the values used outside NicoStan), as many as they ask for. Runs in a
#' fresh R process (a library loaded once in a process is never loaded again from a newer build there).
#'
#' @param model_so_file The compiled library (BridgeStan) of the nico_reduce_sum version.
#' @param Stan_data_list The model's data (a list, or the path of its JSON file).
#' @return The number of nuisance values held by NicoStan.
#' @keywords internal
fn_size_of_the_nuisance_blocks_held_by_NicoStan_from_the_compiled_model <-  function( model_so_file,
                                                                                      Stan_data_list) {

        ## (the data written as cmdstanr writes them, with no chunk_size asked for: a model whose nuisance blocks
        ##  NicoStan holds need not have a reduce_sum, e.g. MetaOrdDTA's random-cutpoint models with C_raw held,
        ##  and fn_JSON_text_of_Stan_data_list() stops on data without exactly one chunk_size)
        JSON_text_of_Stan_data <-  if (is.character(Stan_data_list) && length(Stan_data_list) == 1) {
              paste(readLines(Stan_data_list, warn = FALSE), collapse = "\n")
        } else {
              JSON_file_of_Stan_data <-  tempfile(fileext = ".json")
              on.exit(unlink(JSON_file_of_Stan_data), add = TRUE)
              cmdstanr::write_stan_json(data = Stan_data_list, file = JSON_file_of_Stan_data)
              paste(readLines(JSON_file_of_Stan_data, warn = FALSE), collapse = "\n")
        }
        fn_size_in_a_fresh_process <-  function(model_so_file, JSON_text_of_Stan_data) {

              library_name <-  tools::file_path_sans_ext(basename(model_so_file))
              dyn.load(model_so_file)
              construction <-  .C( "bs_model_construct_R", as.character(JSON_text_of_Stan_data), as.integer(1),
                                   ptr_out = raw(8), err_msg = as.character(""), err_ptr = raw(8),
                                   PACKAGE = library_name)
              if (all(construction$ptr_out == 0)) stop(paste0("BridgeStan: ", construction$err_msg))
              model_pointer <-  construction$ptr_out
              rng <-  .C( "bs_rng_construct_R", as.integer(1), ptr_out = raw(8), err_msg = as.character(""),
                          err_ptr = raw(8), PACKAGE = library_name)
              if (all(rng$ptr_out == 0)) stop(paste0("BridgeStan: ", rng$err_msg))
              names_with_the_generated_quantities <-  strsplit(.C( "bs_param_names_R", as.raw(model_pointer),
                                                                    as.logical(FALSE), as.logical(TRUE),
                                                                    names_out = as.character(""),
                                                                    PACKAGE = library_name)$names_out, ",")[[1]]
              position_of_the_size <-  match("nicostan_size_of_the_nuisance_block_held_by_NicoStan",
                                             names_with_the_generated_quantities)
              if (is.na(position_of_the_size)) {
                    stop(paste0("the library ", model_so_file, " does not report ",
                                "nicostan_size_of_the_nuisance_block_held_by_NicoStan: it is not a ",
                                "nico_reduce_sum version written by NicoStan"))
              }
              n_unconstrained <-  .C( "bs_param_unc_num_R", as.raw(model_pointer), num = as.integer(0),
                                      PACKAGE = library_name)$num
              n_values_out <-  .C( "bs_param_num_R", as.raw(model_pointer), as.logical(FALSE), as.logical(TRUE),
                                   num = as.integer(0), PACKAGE = library_name)$num
              ## (the nuisance values used outside NicoStan: none at first, then as many as the readers ask for,
              ##  doubled each time, all 0)
              n_nuisance_values_given <-  0
              for (attempt in 1:64) {
                    call <-  .C( "bs_param_constrain_R", as.raw(model_pointer), as.logical(FALSE),
                                 as.logical(TRUE), as.double(rep(0, n_unconstrained)),
                                 theta = double(n_values_out), rng = as.raw(rng$ptr_out),
                                 return_code = as.integer(0), err_msg = as.character(""), err_ptr = raw(8),
                                 NAOK = TRUE, PACKAGE = library_name)
                    if (call$return_code == 0) return(call$theta[position_of_the_size])
                    if (!grepl("no nuisance values|outside the nuisance values", call$err_msg)) {
                          stop(paste0("BridgeStan param_constrain: ", call$err_msg))
                    }
                    last_value_asked_for <-  suppressWarnings(as.numeric(sub("^.* to ([0-9]+) are outside.*$",
                                                                             "\\1", call$err_msg)))
                    if (is.na(last_value_asked_for)) last_value_asked_for <-  1
                    n_nuisance_values_given <-  max(2 * n_nuisance_values_given, last_value_asked_for)
                    ## (the routine of the model library named through do.call(), as everywhere else: a .C() call
                    ##  with the name written out is registered by Rcpp as a routine of NicoStan's own library,
                    ##  which does not define it, and NicoStan then fails to load)
                    do.call(".C", list( "nicostan_set_the_nuisance_values_used_outside_NicoStan_R",
                                        as.double(rep(0, n_nuisance_values_given)),
                                        as.integer(n_nuisance_values_given),
                                        PACKAGE = library_name))
              }
              stop("the size of the nuisance block could not be read from the compiled model")

        }
        size_of_every_block <-  callr::r(fn_size_in_a_fresh_process,
                                         args = list(model_so_file, JSON_text_of_Stan_data))
        if (!is.numeric(size_of_every_block) || length(size_of_every_block) != 1 ||
            is.na(size_of_every_block) || size_of_every_block != round(size_of_every_block)) {
              stop(paste0("the size of the nuisance block read from ", model_so_file, " is not a whole number"))
        }
        return(size_of_every_block)

}
##
##
## ---- The slice of a nuisance block held by NicoStan read as ONE matrix: in a copy of a partial-sum function, a
##      slice that is an array of row_vectors, and that the body only ever reads as to_matrix(<slice>), is read
##      straight as the rows of one matrix by nicostan_nuisance_matrix_of_terms() (runtime/general_functions/
##      stan_external_nuisance_values_of_the_block_owned_by_NicoStan.hpp): the same values and nodes, made in the
##      same order, without a row_vector of its own for every term (each a heap allocation and its free in every
##      gradient) nor the copy of to_matrix(); the body's to_matrix(<slice>) become <slice>. A slice that the body
##      reads in any other way (e.g. <slice>[k], or passed to a function) keeps its array reader. The body comes
##      with its comments and strings set aside as markers, as in the writer -
##
fn_slice_of_a_nuisance_block_read_as_a_matrix <-  function(body_of_definition,
                                                           slice_name,
                                                           shape_of_the_block) {

        pattern_of_a_use_of_the_slice <-  paste0("(?<![A-Za-z0-9_])", slice_name, "(?![A-Za-z0-9_])")
        pattern_of_to_matrix_of_the_slice <-  paste0("(?<![A-Za-z0-9_])to_matrix\\s*\\(\\s*", slice_name,
                                                     "\\s*\\)")
        n_uses_of_the_slice <-  sum(gregexpr(pattern = pattern_of_a_use_of_the_slice,
                                             text = body_of_definition, perl = TRUE)[[1]] > 0)
        n_to_matrix_of_the_slice <-  sum(gregexpr(pattern = pattern_of_to_matrix_of_the_slice,
                                                  text = body_of_definition, perl = TRUE)[[1]] > 0)
        read_as_a_matrix <-  (shape_of_the_block == "array_of_row_vectors") &&
                             (n_uses_of_the_slice >= 1) &&
                             (n_to_matrix_of_the_slice == n_uses_of_the_slice)
        if (read_as_a_matrix) {
              body_of_definition <-  gsub(pattern = pattern_of_to_matrix_of_the_slice, replacement = slice_name,
                                          x = body_of_definition, perl = TRUE)
        }
        return(list( read_as_a_matrix   = read_as_a_matrix,
                     body_of_definition = body_of_definition))

}
##
##
## ---- A nico_reduce_sum Stan model written before the matrix reader: each slice read by
##      nicostan_nuisance_array_of_row_vectors() that the body after it only reads as to_matrix(<slice>) is read
##      by nicostan_nuisance_matrix_of_terms() instead, and the matrix reader is declared in the functions block
##      after the array reader; the rest of the model (its comments and commented-out code included) is
##      unchanged. Writes the model to new_Stan_model_file_path (which may be Stan_model_file_path itself) and
##      returns the names of the slices read as matrices -
##
R_fn_read_the_nuisance_slices_of_a_nico_reduce_sum_Stan_model_as_matrices <-  function(
                    Stan_model_file_path,
                    new_Stan_model_file_path
) {

        text <-  paste(readLines(con = Stan_model_file_path, warn = FALSE), collapse = "\n")
        ##
        ## ---- comments and strings set aside (numbered markers), as in the writer:
        ##
        protected_pattern <-  "\"(?:[^\"\\\\]|\\\\.)*\"|/\\*[\\s\\S]*?\\*/|//[^\\n]*"
        protected <-  regmatches(x = text,
                                 m = gregexpr(pattern = protected_pattern, text = text, perl = TRUE))[[1]]
        for (i in seq_along(protected)) {
              text <-  sub(pattern = protected[i], replacement = paste0("\001", i, "\001"), x = text,
                           fixed = TRUE)
        }
        fn_restore <-  function(code) {

              for (i in rev(seq_along(protected))) {
                    code <-  sub(pattern = paste0("\001", i, "\001"), replacement = protected[i], x = code,
                                 fixed = TRUE)
              }
              return(code)
        }

        fn_position_of_closing_bracket <-  function(code, position) {

              depth <-  0
              characters <-  strsplit(code, "")[[1]]
              for (k in position:length(characters)) {
                    if (characters[k] == "{") depth <-  depth + 1
                    if (characters[k] == "}") {
                          depth <-  depth - 1
                          if (depth == 0) return(k)
                    }
              }
              stop("nico_reduce_sum: a bracket is not closed")
        }

        ##
        ## ---- each declaration of a slice read by the array reader, from the last to the first (so that the
        ##      positions of those before it stay the same):
        ##
        pattern_of_a_declaration <-  paste0("array\\[\\s*([A-Za-z0-9_]+)\\s*-\\s*([A-Za-z0-9_]+)\\s*\\+\\s*1\\s*",
                                            "\\]\\s*row_vector\\[nicostan_values_per_term\\]\\s+([A-Za-z0-9_]+)",
                                            "\\s*=\\s*nicostan_nuisance_array_of_row_vectors\\(([^;]*)\\);")
        declarations <-  gregexpr(pattern = pattern_of_a_declaration, text = text, perl = TRUE)[[1]]
        slices_read_as_matrices <-  c()
        if (declarations[1] > 0) {
              for (d in rev(seq_along(declarations))) {
                    start_of_declaration <-  declarations[d]
                    end_of_declaration <-  start_of_declaration + attr(declarations, "match.length")[d] - 1
                    declaration <-  substr(text, start_of_declaration, end_of_declaration)
                    parts <-  regmatches(declaration, regexec(pattern = pattern_of_a_declaration,
                                                              text = declaration, perl = TRUE))[[1]]
                    end_name <-  parts[2]
                    start_name <-  parts[3]
                    slice_name <-  parts[4]
                    arguments_of_the_reader <-  parts[5]
                    ## (the body of the copy of the partial-sum function, whose first statement is the
                    ##  declaration: up to the closing bracket of the last opening bracket before the declaration)
                    opening_of_body <-  max(gregexpr(pattern = "\\{",
                                                     text = substr(text, 1, start_of_declaration))[[1]])
                    closing_of_body <-  fn_position_of_closing_bracket(text, opening_of_body)
                    slice <-  fn_slice_of_a_nuisance_block_read_as_a_matrix(
                                    body_of_definition = substr(text, end_of_declaration + 1, closing_of_body),
                                    slice_name         = slice_name,
                                    shape_of_the_block = "array_of_row_vectors")
                    if (!slice$read_as_a_matrix) next
                    new_declaration <-  paste0("matrix[", end_name, " - ", start_name,
                                               " + 1, nicostan_values_per_term] ", slice_name,
                                               " = nicostan_nuisance_matrix_of_terms(", arguments_of_the_reader,
                                               ");")
                    # text <-  paste0(substr(text, 1, start_of_declaration - 1), new_declaration,
                    #                 slice$body_of_definition, substring(text, closing_of_body + 1))
                    ## (every line changed is kept above its new form as a comment: from the start of the line of
                    ##  the declaration to the end of the body, line by line)
                    start_of_line <-  start_of_declaration
                    while (start_of_line > 1 && substr(text, start_of_line - 1, start_of_line - 1) != "\n") {
                          start_of_line <-  start_of_line - 1
                    }
                    old_lines <-  strsplit(substr(text, start_of_line, closing_of_body), "\n",
                                           fixed = TRUE)[[1]]
                    new_lines <-  strsplit(paste0(substr(text, start_of_line, start_of_declaration - 1),
                                                  new_declaration, slice$body_of_definition),
                                           "\n", fixed = TRUE)[[1]]
                    if (length(old_lines) != length(new_lines)) {
                          stop(paste0("nico_reduce_sum: the lines of the slice ", slice_name,
                                      " changed in number"))
                    }
                    lines_with_the_old_ones_as_comments <-  c()
                    for (k in seq_along(old_lines)) {
                          if (old_lines[k] != new_lines[k]) {
                                indent <-  sub("^([ ]*).*$", "\\1", old_lines[k])
                                lines_with_the_old_ones_as_comments <-  c(lines_with_the_old_ones_as_comments,
                                      paste0(indent, "// ", sub("^[ ]*", "", old_lines[k])))
                          }
                          lines_with_the_old_ones_as_comments <-  c(lines_with_the_old_ones_as_comments,
                                                                    new_lines[k])
                    }
                    text <-  paste0(substr(text, 1, start_of_line - 1),
                                    paste(lines_with_the_old_ones_as_comments, collapse = "\n"),
                                    substring(text, closing_of_body + 1))
                    slices_read_as_matrices <-  c(slice_name, slices_read_as_matrices)
              }
        }
        ##
        ## ---- the matrix reader declared after the array reader (functions block), once:
        ##
        if (length(slices_read_as_matrices) >= 1 &&
            !grepl("matrix nicostan_nuisance_matrix_of_terms(", text, fixed = TRUE)) {
              array_reader <-  regexpr(pattern = paste0("array\\[\\]\\s*row_vector\\s+",
                                                        "nicostan_nuisance_array_of_row_vectors\\([^;]*\\);"),
                                       text = text, perl = TRUE)
              if (array_reader == -1) {
                    stop(paste0("nico_reduce_sum: ", Stan_model_file_path, " does not declare ",
                                "nicostan_nuisance_array_of_row_vectors()"))
              }
              end_of_the_array_reader <-  array_reader + attr(array_reader, "match.length") - 1
              text <-  paste0(substr(text, 1, end_of_the_array_reader),
                              "\n    matrix nicostan_nuisance_matrix_of_terms(int first_term, int last_term, ",
                              "int values_per_term, int values_before_the_block, real type);",
                              substring(text, end_of_the_array_reader + 1))
        }
        writeLines(text = fn_restore(text), con = new_Stan_model_file_path)
        return(slices_read_as_matrices)

}



#' The file of the nico_reduce_sum version of a Stan model
#'
#' The file written beside the model, <model>_nico_reduce_sum.stan; a version already written under the former
#' name, <model>_reduce_sum_chunked.stan, keeps that name, so that the library built from it is still found.
#'
#' @param Stan_model_file_path The Stan model.
#' @return The path of the nico_reduce_sum version of the model.
#' @noRd
fn_path_of_the_nico_reduce_sum_version_of_a_Stan_model <-  function(
                    Stan_model_file_path
) {

        file_under_the_former_name <-  sub("\\.stan$", "_reduce_sum_chunked.stan", Stan_model_file_path)
        if (file.exists(file_under_the_former_name)) {
              return(file_under_the_former_name)
        }
        return(sub("\\.stan$", "_nico_reduce_sum.stan", Stan_model_file_path))

}



#' The former name of the writer of the nico_reduce_sum version of a Stan model
#'
#' Calls R_fn_write_nico_reduce_sum_version_of_Stan_model() with the same arguments.
#'
#' @inheritParams R_fn_write_nico_reduce_sum_version_of_Stan_model
#' @noRd
R_fn_write_reduce_sum_chunked_version_of_Stan_model <-  function(
                    Stan_model_file_path,
                    nuisance_params,
                    stanc_file_path,
                    stanc_include_paths_arguments,
                    Stan_cpp_user_header,
                    nico_reduce_sum_Stan_model_file_path
) {

        return(R_fn_write_nico_reduce_sum_version_of_Stan_model(
                    Stan_model_file_path                  = Stan_model_file_path,
                    nuisance_params                       = nuisance_params,
                    stanc_file_path                       = stanc_file_path,
                    stanc_include_paths_arguments         = stanc_include_paths_arguments,
                    Stan_cpp_user_header                  = Stan_cpp_user_header,
                    nico_reduce_sum_Stan_model_file_path  = nico_reduce_sum_Stan_model_file_path))

}






















