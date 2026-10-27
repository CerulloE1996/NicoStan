
##
## ---- R_fn_nuisance_blocks_first.R --------------------------------------------------------------------------------------------------
##
## ---- A Stan model that calls neither reduce_sum nor map_rect has nothing to chunk: Stan holds its nuisance
##      blocks in its own parameters block, declared first, as cmdstanr runs it, and NicoStan samples them as its
##      nuisance block
##

##
## ---- Does a Stan model call reduce_sum or map_rect (in its included files too)? ----------------------------------------------------
##
R_fn_Stan_model_calls_reduce_sum_or_map_rect <-  function(Stan_model_file_path,
                                                          stanc_args) {

        bridgestan_folder <-  utils::getFromNamespace("get_bridgestan_path", "bridgestan")(download = FALSE)
        stanc_file_path <-  file.path(bridgestan_folder, "bin", "stanc")
        ##
        ## ---- CmdStan's stanc when BridgeStan has none ------------------------------------------------------------------------------
        ##
        if (!file.exists(stanc_file_path)) {

              stanc_file_path <-  file.path(cmdstanr::cmdstan_path(), "bin", "stanc")

        }
        ## (stanc --info lists every function of Stan's library the model calls; its JSON is on stdout, its
        ##  warnings on stderr, kept apart so that a warning never breaks the JSON)
        stanc_args_as_text <-  as.character(unlist(stanc_args, use.names = FALSE))
        stderr_file_of_stanc <-  tempfile(fileext = ".txt")
        on.exit(unlink(stderr_file_of_stanc), add = TRUE)
        output_of_stanc <-  suppressWarnings(system2(command = stanc_file_path,
                                                     args    = c("--info",
                                                                 shQuote(stanc_args_as_text),
                                                                 shQuote(Stan_model_file_path)),
                                                     stdout  = TRUE,
                                                     stderr  = stderr_file_of_stanc))
        ##
        ## ---- A model stanc cannot read stops here ----------------------------------------------------------------------------------
        ##
        if (!is.null(attr(output_of_stanc, "status"))) {

              stop(paste0("R_fn_Stan_model_calls_reduce_sum_or_map_rect: stanc --info failed on ",
                          Stan_model_file_path, ":\n", paste(c(output_of_stanc,
                                                               readLines(stderr_file_of_stanc, warn = FALSE)),
                                                             collapse = "\n")))

        }
        functions_called <-  jsonlite::fromJSON(paste(output_of_stanc, collapse = "\n"))$functions
        return(any(c("reduce_sum", "reduce_sum_static", "map_rect") %in% functions_called))

}



##
## ---- The copy of a Stan model with its nuisance blocks declared first in its parameters block --------------------------------------
##
## (<model>_nuisance_first.stan beside the model: the declarations of the nuisance blocks move, in their own
##  order, to the start of the parameters block; every other line is the model's own, in its own order)
R_fn_write_Stan_model_with_the_nuisance_blocks_declared_first <-  function(Stan_model_file_path,
                                                                           nuisance_params) {

        lines_of_the_model <-  readLines(Stan_model_file_path, warn = FALSE)
        code_of_each_line <-  sub("//.*$", "", lines_of_the_model)
        first_line_of_the_block <-  grep("^\\s*parameters\\s*\\{\\s*$", code_of_each_line)
        ##
        ## ---- A parameters block whose opening brace ends a line of its own ---------------------------------------------------------
        ##
        if (length(first_line_of_the_block) != 1) {

              stop(paste0("nuisance blocks declared first: no line \"parameters {\" of its own in ",
                          Stan_model_file_path))

        }
        ## (its closing brace: the first line after it that is "}" alone)
        lines_after_the_opening <-  seq(from = first_line_of_the_block + 1, to = length(code_of_each_line))
        last_line_of_the_block <-  lines_after_the_opening[grepl("^\\s*\\}\\s*$",
                                                                 code_of_each_line[lines_after_the_opening])][1]
        lines_inside_the_block <-  seq(from = first_line_of_the_block + 1, to = last_line_of_the_block - 1)
        ##
        ## ---- Only // comments inside the block, and no other brace -----------------------------------------------------------------
        ##
        if (any(grepl("/\\*|\\{|\\}", code_of_each_line[lines_inside_the_block]))) {

              stop(paste0("nuisance blocks declared first: the parameters block of ", Stan_model_file_path,
                          " has a /* */ comment or a brace inside it"))

        }
        ## (each declaration: its lines, from its first line of code to the line that ends it with ";")
        declaration_of_each_line <-  rep(NA_integer_, length(lines_of_the_model))
        name_of_each_declaration <-  character(0)
        n_declarations <-  0L
        a_declaration_is_open <-  FALSE
        ##
        ## ---- The declarations of the block, one by one -----------------------------------------------------------------------------
        ##
        for (line in lines_inside_the_block) {

              ##
              ## ---- A line with no code outside a declaration stays where it is -----------------------------------------------------
              ##
              if (!a_declaration_is_open && !grepl("\\S", code_of_each_line[line])) {

                    next

              }
              ##
              ## ---- The first line of a declaration ---------------------------------------------------------------------------------
              ##
              if (!a_declaration_is_open) {

                    n_declarations <-  n_declarations + 1L
                    a_declaration_is_open <-  TRUE

              }
              declaration_of_each_line[line] <-  n_declarations
              ##
              ## ---- The line that ends it: one declaration per statement, its name before the ";" -----------------------------------
              ##
              if (grepl(";", code_of_each_line[line])) {

                    lines_of_the_declaration <-  which(declaration_of_each_line %in% n_declarations)
                    code_of_the_declaration <-  paste(code_of_each_line[lines_of_the_declaration],
                                                      collapse = " ")
                    n_semicolons <-  lengths(regmatches(code_of_the_declaration,
                                                        gregexpr(";", code_of_the_declaration)))
                    ##
                    ## ---- Two statements in one declaration stop here ---------------------------------------------------------------
                    ##
                    if ((n_semicolons > 1) || !grepl(";\\s*$", code_of_the_declaration)) {

                          stop(paste0("nuisance blocks declared first: more than one statement in \"",
                                      trimws(code_of_the_declaration), "\" of ", Stan_model_file_path))

                    }
                    name_of_each_declaration[n_declarations] <-  sub(
                                                                     "^.*?([A-Za-z_][A-Za-z0-9_]*)\\s*;\\s*$",
                                                                     "\\1",
                                                                     code_of_the_declaration,
                                                                     perl = TRUE)
                    a_declaration_is_open <-  FALSE

              }

        }
        declaration_of_each_nuisance_block <-  match(nuisance_params, name_of_each_declaration)
        ##
        ## ---- A nuisance block that is not declared in the parameters block stops here ----------------------------------------------
        ##
        if (anyNA(declaration_of_each_nuisance_block)) {

              stop(paste0("nuisance blocks declared first: ",
                          paste(nuisance_params[is.na(declaration_of_each_nuisance_block)], collapse = ", "),
                          " not declared in the parameters block of ", Stan_model_file_path))

        }
        ##
        ## ---- A name given twice in nuisance_params, or declared twice in the block, stops here -------------------------------------
        ##
        if ((anyDuplicated(nuisance_params) > 0) || (anyDuplicated(name_of_each_declaration) > 0)) {

              stop(paste0("nuisance blocks declared first: a name given twice in nuisance_params (",
                          paste(nuisance_params, collapse = ", "), ") or declared twice in the parameters ",
                          "block of ", Stan_model_file_path))

        }
        declarations_of_the_nuisance_blocks <-  sort(declaration_of_each_nuisance_block)
        ##
        ## ---- A copy already written, with its nuisance blocks first, is used as it is ----------------------------------------------
        ##
        if (grepl("_nuisance_first\\.stan$", Stan_model_file_path) &&
            identical(declarations_of_the_nuisance_blocks, seq_along(nuisance_params))) {

              return(Stan_model_file_path)

        }
        ##
        ## ---- A nuisance block whose declaration uses a parameter declared before it cannot move first ------------------------------
        ##
        for (declaration in declarations_of_the_nuisance_blocks) {

              names_declared_before <-  setdiff(name_of_each_declaration[seq_len(declaration - 1L)],
                                                nuisance_params)
              code_of_the_declaration <-  paste(code_of_each_line[declaration_of_each_line %in% declaration],
                                                collapse = " ")
              names_used <-  names_declared_before[vapply(names_declared_before,
                                                          function(name) grepl(paste0("\\b", name, "\\b"),
                                                                               code_of_the_declaration,
                                                                               perl = TRUE),
                                                          TRUE)]
              ##
              ## ---- Stop, naming the block and the parameters it uses ---------------------------------------------------------------
              ##
              if (length(names_used) > 0) {

                    stop(paste0("nuisance blocks declared first: the declaration of ",
                                name_of_each_declaration[declaration], " uses ",
                                paste(names_used, collapse = ", "), ", declared before it in ",
                                Stan_model_file_path, ", so it cannot move first"))

              }

        }
        lines_of_the_nuisance_blocks <-  which(declaration_of_each_line %in% declarations_of_the_nuisance_blocks)
        lines_of_the_rest_of_the_block <-  setdiff(lines_inside_the_block, lines_of_the_nuisance_blocks)
        lines_from_the_closing_brace <-  seq(from = last_line_of_the_block, to = length(lines_of_the_model))
        lines_of_the_copy <-  lines_of_the_model[c(seq_len(first_line_of_the_block),
                                                   lines_of_the_nuisance_blocks,
                                                   lines_of_the_rest_of_the_block,
                                                   lines_from_the_closing_brace)]
        path_of_the_copy <-  sub("\\.stan$", "_nuisance_first.stan", Stan_model_file_path)
        ##
        ## ---- The copy, written only when its lines change (as the nico_reduce_sum writer writes its version) -----------------------
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

        fn_write_lines_when_they_change(lines  = lines_of_the_copy,
                                        file   = path_of_the_copy)
        ## (the library of a copy written again is older than it: init_bs_model stops the fit, naming both files)
        return(path_of_the_copy)

}
