##
## ==== R_fn_find_loops_of_Stan_model_whose_size_grows_with_N.R =================================================
##
## ---- Lists every loop (for / while) of a Stan model whose number of steps grows with N (subjects, observations,
##      time points), so that no such loop can stay in a model unseen -
##
## How it decides:
##   (1) the sizes that grow with N are found from the model's own data files: every whole-number setting of the
##       data (e.g. N, N_obs, K, T) whose value differs between data files at different N, plus the names that
##       always grow with N (N, N_obs, N_blocks, start, end, n_subjects, N_subjects);
##   (2) a loop is listed as "GROWS WITH N" when its range uses one of those sizes, and as "CHECK" when its range
##       uses the size of a container (size(), rows(), cols(), num_elements()) or it is a while loop, whose length
##       the line alone does not show; loops over fixed sizes (tests, classes, categories, items) are not listed.
##
##
#' The names of the whole-number data settings of a model that grow with N
#'
#' @param data_files Paths of the model's data files (.rds), at two or more different N; each holds the data list
#'   itself or a list with an element stan_data_list.
#' @return A character vector: the names that always grow with N, and every whole-number setting whose value
#'   differs between the data files.
#' @noRd
fn_names_of_data_sizes_that_grow_with_N <-  function(data_files) {

        ##
        names_always_growing_with_N <-  c("N", "N_obs", "N_blocks", "start", "end", "n_subjects", "N_subjects")
        if (length(data_files) < 2) return(names_always_growing_with_N)
        ##
        data_lists <-  lapply(data_files, function(data_file) {

              data_object <-  readRDS(data_file)
              if (!is.null(data_object$stan_data_list)) return(data_object$stan_data_list)
              return(data_object)
        })

        is_whole_number_setting <-  function(value) is.numeric(value) && (length(value) == 1)
        names_of_settings <-  Reduce(intersect, lapply(data_lists, function(data_list) {

              names(data_list)[sapply(data_list, is_whole_number_setting)]
        }))

        values <-  sapply(data_lists, function(data_list) unlist(data_list[names_of_settings]))
        if (is.null(dim(values))) values <-  matrix(values, nrow = length(names_of_settings))
        names_growing <-  names_of_settings[apply(values, 1, function(row) length(unique(row)) > 1)]
        ##
        return(unique(c(names_always_growing_with_N, names_growing)))
##
}

##
##
#' Every loop of a Stan model whose number of steps grows with N, or whose length the line alone does not show
#'
#' @param Stan_model_file_path Path of the .stan file.
#' @param data_files Paths of the model's data files at two or more different N (see
#'   fn_names_of_data_sizes_that_grow_with_N()).
#' @return NULL when there is no such loop; otherwise a data.frame with one row per loop: line, block (functions,
#'   transformed data, model, generated quantities, ...), verdict ("GROWS WITH N (<names>)", "CHECK (while loop)"
#'   or "CHECK (size of a container)") and the line's text.
#' @noRd
fn_find_loops_of_Stan_model_whose_size_grows_with_N <-  function( Stan_model_file_path,
                                                                   data_files) {

        ##
        lines <-  readLines(Stan_model_file_path, warn = FALSE)
        sizes_growing_with_N <-  fn_names_of_data_sizes_that_grow_with_N(data_files = data_files)
        names_of_blocks <-  c("functions", "data", "transformed data", "parameters", "transformed parameters",
                              "model", "generated quantities")
        block <-  "functions"
        rows <-  list()
        ##
        inside_block_comment <-  FALSE
        for (line_number in seq_along(lines)) {
              ## (the text of the line without its comments: // to the end of the line, and /* ... */ blocks,
              ##  which may span several lines)
              code <-  lines[line_number]
              if (inside_block_comment) {
                    if (!grepl("\\*/", code)) next
                    code <-  sub("^.*?\\*/", "", code, perl = TRUE)
                    inside_block_comment <-  FALSE
              }
              code <-  gsub("/\\*.*?\\*/", "", code, perl = TRUE)
              if (grepl("/\\*", code)) {
                    code <-  sub("/\\*.*$", "", code)
                    inside_block_comment <-  TRUE
              }
              code <-  sub("//.*$", "", code)
              header <-  regmatches(code, regexpr("^[a-z ]+(?=\\{)", code, perl = TRUE))
              if ((length(header) == 1) && (trimws(header) %in% names_of_blocks)) block <-  trimws(header)
              ##
              has_for <-  grepl("\\bfor\\s*\\(", code)
              has_while <-  grepl("\\bwhile\\s*\\(", code)
              if (!has_for && !has_while) next
              ##
              verdict <-  ""
              if (has_while) verdict <-  "CHECK (while loop)"
              if (has_for) {
                    ## (every for of the line, e.g. "for (i in 1:N) for (c in 1:n_class)", each up to the first
                    ##  closing bracket of its range)
                    ranges <-  regmatches(code,
                                          gregexpr("\\bfor\\s*\\(\\s*[A-Za-z_][A-Za-z0-9_]*\\s+in\\s+[^)]*",
                                                   code))[[1]]
                    ranges <-  sub("^.*\\bin\\s+", "", ranges)
                    for (range_text in ranges) {
                          names_in_range <-  regmatches(range_text, gregexpr("[A-Za-z_][A-Za-z0-9_]*",
                                                                             range_text))[[1]]
                          if (any(names_in_range %in% sizes_growing_with_N)) {
                                verdict <-  paste0("GROWS WITH N (",
                                                   paste(intersect(names_in_range, sizes_growing_with_N),
                                                         collapse = ", "), ")")
                                break
                          } else if (grepl("\\b(size|rows|cols|num_elements)\\s*\\(", range_text)) {
                                verdict <-  "CHECK (size of a container)"
                          }
                    }
              }
              if (verdict == "") next
              rows[[length(rows) + 1]] <-  data.frame( line     = line_number,
                                                       block    = block,
                                                       verdict  = verdict,
                                                       text     = substr(trimws(code), 1, 90))
        }
        ##
        if (length(rows) == 0) return(NULL)
        return(do.call(rbind, rows))
##
}






















