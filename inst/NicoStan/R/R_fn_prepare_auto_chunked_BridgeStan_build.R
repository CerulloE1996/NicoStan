##
## ---- The converter's functions, as loaded ----------------------------------------------------------------------------------------
##
R_fn_auto_chunking_converter <-  function() {

        functions <-  c("fn_tokens_of_cpp_expression", "fn_tree_of_cpp_expression", "fn_text_of_cpp_tree",
                        "fn_data_of_Stan_model", "fn_parameters_and_locals_of_log_prob", "fn_end_of_bracket",
                        "fn_assignment_of_local", "fn_statements_of_the_locals_as_stanc_made_them",
                        "fn_row_wise_kind_of_node", "fn_row_wise_kind_of_density",
                        "fn_why_the_locals_cannot_be_made_in_the_tile", "fn_auto_chunk_one_log_prob_body",
                        "fn_auto_chunk_the_sums_over_rows_in_generated_Stan_cpp",
                        "fn_read_N_chunks_of_auto_chunking_from_the_data", "fn_statements_of_cpp_block",
                        "fn_inside_of_the_bracket", "fn_arguments_of_a_call_text",
                        "fn_running_sum_functions_of_generated_cpp", "fn_running_sum_function_for_a_chunk",
                        "fn_terms_of_a_sum", "fn_auto_chunk_the_running_sums_in_one_log_prob_body",
                        "fn_contracts_of_running_sums", "fn_contracts_of_row_sum_densities",
                        "fn_grouped_density_of_rows", "fn_data_members_of_generated_cpp",
                        "fn_auto_chunk_the_calls_of_partial_sum_functions",
                        "fn_contracts_of_row_sums_with_the_row_before",
                        "fn_density_of_rows_with_the_row_before",
                        "fn_cpp_text_without_strings", "fn_braces_of_cpp_text",
                        "fn_names_declared_in_cpp_text", "fn_names_read_in_cpp_text",
                        "fn_types_of_the_data_members_of_generated_cpp",
                        "fn_type_and_first_dimension_of_a_var",
                        "fn_auto_chunk_the_blocks_of_rows_in_one_log_prob_body",
                        "fn_data_read_only_in_blocks_of_N_entries",
                        "fn_auto_chunk_the_loops_over_groups_of_rows_in_one_log_prob_body")
        ##
        ## ---- Bind the converter actually loaded rather than a different source file ------------------------------------------------
        ##
        converter <-  lapply(functions, function(name) {

                fun <-  utils::removeSource(get(name, envir = parent.env(environment()), inherits = TRUE))
                expression <-  parse(text = deparse(body(fun),
                                                      control = c("keepNA", "keepInteger", "niceNames", "hexNumeric")),
                                      keep.source = FALSE)[[1]]
                return(list(arguments = formals(fun), expression = expression))

        })
        names(converter) <-  functions
        return(converter)

}
##
## ---- The library of a Stan file, with AUTO-CHUNKING on or off --------------------------------------------------------------------
##
R_fn_library_of_Stan_model_file <-  function(model_file,
                                             auto_chunking) {

        if (!(isTRUE(auto_chunking) || isFALSE(auto_chunking))) stop("auto_chunking must be TRUE or FALSE.")
        stem <-  tools::file_path_sans_ext(normalizePath(model_file, mustWork = FALSE))
        if (!auto_chunking) return(paste0(stem, "_model.so"))
        report_file <-  paste0(stem, "_auto_chunked_auto_chunking_report.rds")
        ##
        ## ---- A model with no sum over rows to chunk: its own library, while its receipt still holds ------------------------------
        ##
        if (file.exists(report_file)) {

                record <-  readRDS(report_file)
                if (isFALSE(record$converted) && file.exists(model_file) &&
                    identical(record$source_md5, unname(tools::md5sum(model_file))) &&
                    identical(record$converter, R_fn_auto_chunking_converter())) {

                        return(paste0(stem, "_model.so"))

                }

        }
        return(paste0(stem, "_auto_chunked_model.so"))

}
##
## ---- An optional AUTO-CHUNKING runtime header: beside the runtime header, else in the first library that has it --------------------
##
fn_optional_runtime_header_of_auto_chunking <-  function(
        runtime_header,
        file_name) {

        ## (a frozen copy of NicoSIMD may be older than the converter: its runtime headers come first, the
        ##  ones it lacks from the installed NicoSIMD)
        folders <-  c(dirname(runtime_header), file.path(.libPaths(), "NicoSIMD", "include", "NicoSIMD"))
        paths <-  file.path(folders, file_name)
        found <-  paths[file.exists(paths)]
        if (length(found) == 0L) {

                stop("No AUTO-CHUNKING runtime header ", file_name, " in ", paste(folders, collapse = ", "))

        }
        return(normalizePath(found[1], mustWork = TRUE))

}
##
## ---- Record expanded source and local compilation dependencies ---------------------------------------------------------------------
##
R_fn_auto_chunking_dependencies <-  function(
        model_file,
        stanc_args,
        wrapper,
        runtime_header) {

        source <-  R_fn_read_Stan_source_with_includes(model_file = model_file,
                                                      stanc_args = stanc_args)
        headers <-  fn_local_header_dependency_records(
                header_paths = wrapper,
                NicoSIMD_headers_folder = dirname(runtime_header))
        headers <-  sub(" \\| [^|]+ \\| ([[:xdigit:]]+)$", " | \\1", headers)
        return(list(expanded_source = source, local_headers = headers, converter = R_fn_auto_chunking_converter()))

}
##
## ---- Build the AUTO-CHUNKING library of a Stan file: stanc's C++ of the file itself, converted, then compiled ----------------------
##
R_fn_build_auto_chunked_BridgeStan_library <-  function(
        model_file,
        stanc_args,
        make_args,
        names_of_the_nuisance_blocks) {

        model_file <-  normalizePath(model_file, mustWork = TRUE)
        if (tools::file_ext(model_file) != "stan") stop("AUTO-CHUNKING builds from a .stan file: ", model_file)
        stem <-  paste0(tools::file_path_sans_ext(model_file), "_auto_chunked")
        report_file <-  paste0(stem, "_auto_chunking_report.rds")
        library <-  paste0(stem, "_model.so")
        flags <-  as.character(unlist(stanc_args))
        arguments <-  as.character(unlist(make_args))
        ##
        ## ---- Keep the default translation and the caller's C++ compilation flags ---------------------------------------------------
        ##
        if (any(!grepl("^(--allow-undefined|--include-paths=.*)$", flags)) ||
            any(startsWith(arguments, "STANCFLAGS=")) || nzchar(Sys.getenv("STANCFLAGS"))) {

                stop("AUTO-CHUNKING accepts only default translation with undefined calls and include paths.")

        }
        if (file.exists(library)) stop("An existing library must not be rewritten: ", library)
        runtime_header <-  getOption("NicoStan.auto_chunking_header", NULL)
        ##
        ## ---- Use the installed runtime unless a development header was explicitly selected -----------------------------------------
        ##
        if (is.null(runtime_header)) {

                runtime_header <-  system.file("include",
                                               "NicoSIMD",
                                               "nico_auto_chunking.hpp",
                                               package = "NicoSIMD",
                                               mustWork = TRUE)

        }
        runtime_header <-  normalizePath(runtime_header, mustWork = TRUE)
        original_headers <-  grep("^USER_HEADER=", arguments)
        stopifnot(length(original_headers) <= 1L)
        wrapper <-  paste0(stem, "_auto_chunking_user_header.hpp")
        includes <-  c("#pragma once")
        ##
        ## ---- Keep the caller's header before the AUTO-CHUNKING runtime -------------------------------------------------------------
        ##
        if (length(original_headers) == 1L) {

                original_header <-  normalizePath(sub("^USER_HEADER=", "", arguments[original_headers]),
                                                  mustWork = TRUE)
                stopifnot(original_header != wrapper)
                includes <-  c(includes, paste0("#include ", encodeString(original_header, quote = '"')))
                arguments <-  arguments[-original_headers]

        }
        ##
        ## ---- The runtime from the NicoSIMD folder the caller's header already includes ---------------------------------------------
        ##
        ## (a caller's header may include NicoSIMD's headers from another copy of NicoSIMD, e.g. its source
        ##  folder: the runtime and its functions header are then taken from that same folder, so that no
        ##  NicoSIMD header is read twice)
        NicoSIMD_folders_of_the_caller <-  character(0)
        headers_to_read <-  if (length(original_headers) == 1L) original_header else character(0)
        headers_read <-  character(0)
        while (length(headers_to_read) > 0L) {

                header <-  headers_to_read[1]
                headers_to_read <-  headers_to_read[-1]
                headers_read <-  c(headers_read, header)
                included <-  sub(pattern = '^[[:space:]]*#[[:space:]]*include[[:space:]]*"([^"]+)".*$',
                                 replacement = "\\1",
                                 x = grep(pattern = '^[[:space:]]*#[[:space:]]*include[[:space:]]*"/',
                                          x = readLines(header, warn = FALSE),
                                          value = TRUE))
                included <-  normalizePath(included[file.exists(included)], mustWork = TRUE)
                ## (any header in a folder named NicoSIMD, e.g. the functions header of a frozen copy of NicoSIMD,
                ##  names the NicoSIMD folder the caller uses)
                is_a_NicoSIMD_header <-  basename(dirname(included)) == "NicoSIMD"
                NicoSIMD_folders_of_the_caller <-  unique(c(NicoSIMD_folders_of_the_caller,
                                                            dirname(included[is_a_NicoSIMD_header])))
                headers_to_read <-  unique(c(headers_to_read,
                                             setdiff(included[!is_a_NicoSIMD_header], headers_read)))

        }
        if (length(NicoSIMD_folders_of_the_caller) > 1L) {

                stop("The model's header includes NicoSIMD from more than one folder: ",
                     paste(NicoSIMD_folders_of_the_caller, collapse = ", "))

        }
        if (length(NicoSIMD_folders_of_the_caller) == 1L) {

                if (!is.null(getOption("NicoStan.auto_chunking_header", NULL)) &&
                    !identical(dirname(runtime_header), NicoSIMD_folders_of_the_caller)) {

                        stop("NicoStan.auto_chunking_header is not in the NicoSIMD folder the model's header ",
                             "includes: ",
                             NicoSIMD_folders_of_the_caller)

                }
                runtime_header <-  normalizePath(file.path(NicoSIMD_folders_of_the_caller,
                                                           "nico_auto_chunking.hpp"),
                                                 mustWork = TRUE)

        }
        ## (the runtime's elementwise calls use NicoSIMD's kernels: NicoSIMD's functions header, beside the runtime header,
        ##  comes before it whatever the caller's header holds)
        functions_header <-  normalizePath(file.path(dirname(runtime_header), "stan_external_functions.hpp"),
                                           mustWork = TRUE)
        includes <-  c(includes, paste0("#include ", encodeString(functions_header, quote = '"')),
                       paste0("#include ", encodeString(runtime_header, quote = '"')))
        writeLines(includes, wrapper)
        arguments <-  c(arguments, paste0("USER_HEADER=", wrapper))
        runtime <-  normalizePath(utils::getFromNamespace("get_bridgestan_path", "bridgestan")(download = FALSE),
                                  mustWork = TRUE)
        generated_file <-  paste0(stem, ".hpp")
        ## (the model named for this library, so that it and the model's own library can be loaded together)
        stancflags <-  paste("--include-paths=.", paste(flags, collapse = " "),
                             paste0("--name=", basename(stem), "_model"))
        make_command <-  get("MAKE", envir = asNamespace("bridgestan"))
        ##
        ## ---- stanc's C++ of the file itself: the runtime's own rule for <model>.hpp, given to <model>_auto_chunked.hpp ---------------
        ##
        makefile_lines <-  readLines(file.path(runtime, "Makefile"), warn = FALSE)
        rule <-  grep("^%[.]hpp : %[.]stan", makefile_lines)
        if (length(rule) != 1L) stop("No single rule from .stan to .hpp in ", file.path(runtime, "Makefile"))
        recipe <-  makefile_lines[-seq_len(rule)]
        recipe <-  recipe[seq_len(match(FALSE, startsWith(recipe, "\t"), nomatch = length(recipe) + 1L) - 1L)]
        rule_file <-  tempfile(pattern = "nicostan_auto_chunking_rule_", fileext = ".mk")
        writeLines(c("include Makefile",
                     sub("^%[.]hpp : %[.]stan", "%_auto_chunked.hpp : %.stan", makefile_lines[rule]),
                     recipe),
                   rule_file)
        log_file <-  paste0(stem, "_auto_chunking_stanc.log")
        status <-  system2(command = make_command,
                           args = c("-C", shQuote(runtime), "-f", shQuote(rule_file), arguments,
                                    paste0("STANCFLAGS=", shQuote(stancflags)),
                                    paste0("--assume-new=", shQuote(model_file)), shQuote(generated_file)),
                           stdout = log_file,
                           stderr = log_file)
        file.remove(rule_file)
        if (status != 0L) stop("AUTO-CHUNKING translation failed; inspect ", log_file)
        original <-  readLines(generated_file, warn = FALSE)
        source <-  R_fn_read_Stan_source_with_includes(model_file = model_file,
                                                      stanc_args = stanc_args)
        result <-  fn_auto_chunk_the_sums_over_rows_in_generated_Stan_cpp(
                generated_cpp_lines = original,
                Stan_model_lines = source,
                N_chunks_of_auto_chunking_text = "N_chunks_of_auto_chunking__",
                names_of_the_nuisance_blocks = names_of_the_nuisance_blocks)
        ##
        ## ---- A model with no sum over rows to chunk: its own library is used, as its receipt says ---------------------------------
        ##
        if (!any(grepl("tiled over ", result$report, fixed = TRUE))) {

                file.remove(generated_file, wrapper)
                saveRDS(list(source = model_file,
                             source_md5 = unname(tools::md5sum(model_file)),
                             converted = FALSE,
                             converter = R_fn_auto_chunking_converter(),
                             names_of_the_nuisance_blocks = names_of_the_nuisance_blocks,
                             report = result$report),
                        report_file)
                message("AUTO-CHUNKING: no sum over rows to chunk in ", model_file, "; its own library is used.")
                return(list(converted = FALSE, library = paste0(tools::file_path_sans_ext(model_file), "_model.so")))

        }
        ##
        ## ---- The number of chunks read from the model's data (searched per N, model and device, never fixed) -----------------------
        ##
        result$generated_cpp_lines <-  fn_read_N_chunks_of_auto_chunking_from_the_data(
                generated_cpp_lines = result$generated_cpp_lines)
        ## (a running sum chunked: its runtime, beside the AUTO-CHUNKING runtime, after it in the wrapper)
        if (any(grepl("sum_of_rows_with_running_sums_in_the_chunks_of_reduce_sum_static", result$generated_cpp_lines,
                      fixed = TRUE))) {

                runtime_of_the_running_sums <-  fn_optional_runtime_header_of_auto_chunking(
                        runtime_header = runtime_header,
                        file_name = "nico_auto_chunking_running_sums.hpp")
                write(paste0("#include ", encodeString(runtime_of_the_running_sums, quote = '"')), file = wrapper,
                      append = TRUE)

        }
        ## (unique groups gathered within each chunk: its optional runtime, after the base runtime)
        if (any(grepl("gathered_at_unique_groups_of(",
                      result$generated_cpp_lines,
                      fixed = TRUE))) {

                runtime_of_unique_groups <-  fn_optional_runtime_header_of_auto_chunking(
                        runtime_header = runtime_header,
                        file_name = "nico_auto_chunking_unique_groups.hpp")
                write(paste0("#include ", encodeString(runtime_of_unique_groups, quote = '"')),
                      file = wrapper,
                      append = TRUE)

        }
        ## (a partial-sum function's call on all its rows cut into tiles: its optional runtime, after the base runtime)
        if (any(grepl("sliced_array_at_rows_of(",
                      result$generated_cpp_lines,
                      fixed = TRUE))) {

                runtime_of_partial_sum_calls <-  fn_optional_runtime_header_of_auto_chunking(
                        runtime_header = runtime_header,
                        file_name = "nico_auto_chunking_partial_sum_calls.hpp")
                write(paste0("#include ", encodeString(runtime_of_partial_sum_calls, quote = '"')),
                      file = wrapper,
                      append = TRUE)

        }
        ## (a partial-sum function that reads NicoStan's nuisance block, each tile one chunk of it: its optional
        ##  runtime, after the runtime of the partial-sum calls)
        if (any(grepl("sum_of_rows_in_the_chunks_of_reduce_sum_static_reading_the_nuisance_block",
                      result$generated_cpp_lines,
                      fixed = TRUE))) {

                runtime_of_nuisance_block_calls <-  fn_optional_runtime_header_of_auto_chunking(
                        runtime_header = runtime_header,
                        file_name = "nico_auto_chunking_partial_sum_calls_reading_the_nuisance_block.hpp")
                write(paste0("#include ", encodeString(runtime_of_nuisance_block_calls, quote = '"')),
                      file = wrapper,
                      append = TRUE)

        }
        ## (a density whose term of a row reads the row before: its runtime, after the base runtime)
        if (any(grepl("sum_of_rows_with_the_row_before_in_the_chunks_of_reduce_sum_static",
                      result$generated_cpp_lines,
                      fixed = TRUE))) {

                runtime_of_the_row_before <-  fn_optional_runtime_header_of_auto_chunking(
                        runtime_header = runtime_header,
                        file_name = "nico_auto_chunking_row_before.hpp")
                write(paste0("#include ", encodeString(runtime_of_the_row_before, quote = '"')),
                      file = wrapper,
                      append = TRUE)

        }
        ## (a density with no declared row-sum contract, checked at its first call: its runtime, after the base runtime)
        if (any(grepl("sum_of_rows_of_a_density_without_a_contract_in_the_chunks_of_reduce_sum_static",
                      result$generated_cpp_lines,
                      fixed = TRUE))) {

                runtime_of_densities_without_a_contract <-  fn_optional_runtime_header_of_auto_chunking(
                        runtime_header = runtime_header,
                        file_name = "nico_auto_chunking_densities_without_a_contract.hpp")
                write(paste0("#include ", encodeString(runtime_of_densities_without_a_contract, quote = '"')),
                      file = wrapper,
                      append = TRUE)

        }
        ## (a block of the model run on each chunk of its rows as its own statements: its runtime, after the
        ##  runtime of the partial-sum calls, which it includes)
        if (any(grepl("sum_of_a_block_of_rows_in_the_chunks_of_reduce_sum_static",
                      result$generated_cpp_lines,
                      fixed = TRUE))) {

                runtime_of_the_blocks_of_rows <-  fn_optional_runtime_header_of_auto_chunking(
                        runtime_header = runtime_header,
                        file_name = "nico_auto_chunking_blocks_of_rows.hpp")
                write(paste0("#include ", encodeString(runtime_of_the_blocks_of_rows, quote = '"')),
                      file = wrapper,
                      append = TRUE)

        }
        original_md5 <-  unname(tools::md5sum(generated_file))
        writeLines(result$generated_cpp_lines, generated_file)
        writeLines(result$report, paste0(stem, "_auto_chunking_report.txt"))
        record <-  list(
                source = model_file,
                source_md5 = unname(tools::md5sum(model_file)),
                converted = TRUE,
                library = library,
                wrapper = wrapper,
                generated_cpp = generated_file,
                original_generated_cpp = original,
                original_generated_cpp_md5 = original_md5,
                converted_generated_cpp_md5 = unname(tools::md5sum(generated_file)),
                runtime_header = runtime_header,
                runtime_header_md5 = unname(tools::md5sum(runtime_header)),
                dependencies = R_fn_auto_chunking_dependencies(
                        model_file = model_file,
                        stanc_args = stanc_args,
                        wrapper = wrapper,
                        runtime_header = runtime_header),
                N_chunks_of_auto_chunking = "read from the model's data",
                names_of_the_nuisance_blocks = names_of_the_nuisance_blocks,
                stanc_args = stanc_args,
                make_args = arguments,
                report = result$report,
                gather_path_qualified = FALSE,
                library_md5 = NULL)
        saveRDS(record, report_file)
        ##
        ## ---- Compile and link the converted C++ with the caller's make arguments (make has no rule to remake it) --------------------
        ##
        make_log <-  paste0(stem, "_auto_chunking_make.log")
        status <-  system2(command = make_command,
                           args = c("-C", shQuote(runtime), arguments,
                                    paste0("STANCFLAGS=", shQuote(stancflags)),
                                    shQuote(library)),
                           stdout = make_log,
                           stderr = make_log)
        if (status != 0L || !file.exists(library)) stop("AUTO-CHUNKING compilation failed; inspect ", make_log)
        stopifnot(identical(record$converted_generated_cpp_md5, unname(tools::md5sum(generated_file))))
        record$library_md5 <-  unname(tools::md5sum(library))
        saveRDS(record, report_file)
        return(list(converted = TRUE, library = library))

}
##
## ---- Require matching provenance when an AUTO-CHUNKING library is reused -----------------------------------------------------------
##
R_fn_check_auto_chunked_BridgeStan_library <-  function(library) {

        report_file <-  sub("_model[.]so$", "_auto_chunking_report.rds", library)
        if (!file.exists(report_file)) stop("No AUTO-CHUNKING build receipt for ", library)
        record <-  readRDS(report_file)
        ##
        ## ---- Refuse a library built from a copy of its Stan file, or with fixed rows per tile ------------------------------------
        ##
        if (!isTRUE(record$converted) || is.null(record$wrapper) || grepl("_auto_chunked[.]stan$", record$source)) {

                stop("AUTO-CHUNKING library not built from its model's own Stan file with auto_chunking = TRUE: ",
                     library)

        }
        if (!is.null(record$rows_per_tile) ||
            !identical(record$N_chunks_of_auto_chunking, "read from the model's data")) {

                stop("AUTO-CHUNKING library built with fixed rows per tile, not the searched number of chunks: ",
                     library)

        }
        ##
        ## ---- Check source and library without rebuilding ---------------------------------------------------------------------------
        ##
        if (is.null(record$library_md5) ||
            !identical(record$library_md5, unname(tools::md5sum(library))) ||
            !identical(record$source_md5, unname(tools::md5sum(record$source)))) {

                stop("AUTO-CHUNKING source/library receipt does not match: ", library)

        }
        runtime_header <-  getOption("NicoStan.auto_chunking_header", record$runtime_header)
        ##
        ## ---- Refuse stale dependencies without rewriting the retained library ------------------------------------------------------
        ##
        if (is.null(record$dependencies) || is.null(record$runtime_header) ||
            !identical(normalizePath(runtime_header, mustWork = TRUE), record$runtime_header) ||
            !identical(record$converted_generated_cpp_md5,
                       unname(tools::md5sum(record$generated_cpp))) ||
            !identical(record$dependencies, R_fn_auto_chunking_dependencies(
                    model_file = record$source,
                    stanc_args = record$stanc_args,
                    wrapper = record$wrapper,
                    runtime_header = record$runtime_header))) {

                stop("AUTO-CHUNKING dependencies do not match their build receipt: ", library)

        }
        return(invisible(record))

}
##
## ---- Renew the receipt of a retained library when only the converter's code changed ------------------------------------------------
##
## (the current converter, run again on the receipt's own stanc C++ of the same Stan file, must give the converted
##  C++ the library was compiled from, byte for byte, with the same expanded Stan source and the same headers; the
##  library is not touched, and the renewal and the converter it replaced are written in the receipt)
R_fn_renew_receipt_of_auto_chunked_BridgeStan_library <-  function(library) {

        report_file <-  sub("_model[.]so$", "_auto_chunking_report.rds", library)
        record <-  readRDS(report_file)
        if (!isTRUE(record$converted) || is.null(record$original_generated_cpp) ||
            !identical(record$library_md5, unname(tools::md5sum(library))) ||
            !identical(record$source_md5, unname(tools::md5sum(record$source))) ||
            !identical(record$converted_generated_cpp_md5, unname(tools::md5sum(record$generated_cpp)))) {

                stop("The library, its Stan file or its converted C++ does not match its receipt: ", library)

        }
        ##
        ## ---- The current converter on the receipt's own stanc C++ ------------------------------------------------------------------
        ##
        result <-  fn_auto_chunk_the_sums_over_rows_in_generated_Stan_cpp(
                generated_cpp_lines = record$original_generated_cpp,
                Stan_model_lines = R_fn_read_Stan_source_with_includes(model_file = record$source,
                                                                      stanc_args = record$stanc_args),
                N_chunks_of_auto_chunking_text = "N_chunks_of_auto_chunking__",
                names_of_the_nuisance_blocks = record$names_of_the_nuisance_blocks)
        file_converted_again <-  tempfile(fileext = ".hpp")
        writeLines(text = fn_read_N_chunks_of_auto_chunking_from_the_data(
                           generated_cpp_lines = result$generated_cpp_lines),
                   con = file_converted_again)
        md5_converted_again <-  unname(tools::md5sum(file_converted_again))
        file.remove(file_converted_again)
        dependencies <-  R_fn_auto_chunking_dependencies(model_file = record$source,
                                                         stanc_args = record$stanc_args,
                                                         wrapper = record$wrapper,
                                                         runtime_header = record$runtime_header)
        if (!identical(md5_converted_again, record$converted_generated_cpp_md5) ||
            !identical(dependencies$expanded_source, record$dependencies$expanded_source) ||
            !identical(dependencies$local_headers, record$dependencies$local_headers)) {

                stop("The current converter, Stan source or headers do not give this library's C++; rebuild it: ",
                     library)

        }
        record$receipt_renewed <-  c(record$receipt_renewed,
                                     list(list(at = format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
                                               converter_before = record$dependencies$converter)))
        record$dependencies <-  dependencies
        saveRDS(record, report_file)
        return(invisible(R_fn_check_auto_chunked_BridgeStan_library(library)))

}
