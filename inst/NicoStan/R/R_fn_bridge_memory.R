## Model compilation support. This changes model artifacts, never result caches.
## the libomp shared library NicoStan.so is linked against (from ldd; NULL when not found), so that a model library
## compiled with the OpenMP reduce_sum links the same OpenMP runtime
R_fn_bridge_memory_libomp_of_NicoStan <-  function() {
      nicostan_so <-  system.file("libs", "NicoStan.so", package = "NicoStan")
      if (!nzchar(x = nicostan_so) || Sys.which("ldd") == "") return(NULL)
      ldd_lines <-  tryCatch(system2(command = "ldd", args = shQuote(string = nicostan_so), stdout = TRUE,
                                     stderr = FALSE), error = function(e) character(0))
      omp_line <-  grep(pattern = "libomp\\.so", x = ldd_lines, value = TRUE)
      if (length(x = omp_line) == 0) return(NULL)
      path <-  sub(pattern = "^.*=> *([^ ]+) .*$", replacement = "\\1", x = omp_line[1])
      if (!file.exists(path)) return(NULL)
      normalizePath(path = path, mustWork = TRUE)
}

R_fn_bridge_memory_compile_options <-  function(model_file, make_args = NULL,
                                               stanc_args = NULL,
                                               release_header = NULL) {
      if (is.null(x = release_header)) {
            # release_header <-  system.file("include", "NicoStan", "runtime",
            #                               "general_functions", "bridge_memory.hpp",
            #                               package = "NicoStan", mustWork = TRUE)
            ## (the file of the one NicoStan install, never a side library: fn_file_of_the_one_NicoStan_install())
            release_header <-  fn_file_of_the_one_NicoStan_install(
                  "include", "NicoStan", "runtime", "general_functions",
                  "bridge_memory.hpp",
                  package = "NicoStan", mustWork = TRUE)
      }
      release_header <-  normalizePath(path = release_header, mustWork = TRUE)
      wrapper_file <-  paste0(tools::file_path_sans_ext(x = model_file), "_nicostan_user_header.hpp")
      make_args <-  as.character(x = unlist(x = make_args))
      user_header_entries <-  grep(pattern = "^USER_HEADER=", x = make_args)
      if (length(user_header_entries) > 1) stop("Multiple USER_HEADER arguments are ambiguous.")
      wrapper_lines <-  c("#pragma once",
                         paste0("#include ", encodeString(x = release_header, quote = '"')))
      if (length(user_header_entries) == 1) {
            original_header <-  sub(pattern = "^USER_HEADER=", replacement = "",
                                   x = make_args[user_header_entries])
            original_header <-  normalizePath(path = original_header, mustWork = TRUE)
            if (identical(x = original_header, y = normalizePath(path = wrapper_file, mustWork = FALSE))) {
                  existing_lines <-  readLines(con = original_header, warn = FALSE)
                  if (length(existing_lines) < 2 || existing_lines[1] != "#pragma once" ||
                      !startsWith(x = existing_lines[2], prefix = "#include ")) {
                        stop("Cannot safely reuse an unrecognised generated user header: ", original_header)
                  }
                  ## Keep its original custom includes, replacing only the cleanup include.
                  wrapper_lines <-  c(wrapper_lines, existing_lines[-c(1, 2)])
            } else {
                  wrapper_lines <-  c(wrapper_lines,
                                     paste0("#include ", encodeString(x = original_header, quote = '"')))
            }
            make_args <-  make_args[-user_header_entries]
      }
      if (!file.exists(wrapper_file) ||
          !identical(x = readLines(con = wrapper_file, warn = FALSE), y = wrapper_lines)) {
            writeLines(text = wrapper_lines, con = wrapper_file)
      }
      ## (the OpenMP variant of NicoStan's reduce_sum patch, options(NicoStan.reduce_sum_parallel_bookkeeping =
      ##  TRUE): the model is compiled with -fopenmp and NICOSTAN_REDUCE_SUM_OPENMP and linked against the libomp
      ##  that NicoStan.so itself uses (one OpenMP runtime in the process); Linux only, like the page release)
      if (isTRUE(x = getOption("NicoStan.reduce_sum_parallel_bookkeeping", FALSE))) {
            libomp <-  R_fn_bridge_memory_libomp_of_NicoStan()
            ## (the value quoted inside the string, as CXXFLAGS_OPTIM='...' is: make receives one argument)
            make_args <-  c(make_args,
                            "CXXFLAGS_OS='-D_REENTRANT -fopenmp -DNICOSTAN_REDUCE_SUM_OPENMP'",
                            if (!is.null(x = libomp)) paste0("LDLIBS_OS='", libomp, "'") else "LDLIBS_OS='-lomp'")
      }
      list(make_args = as.list(c(make_args, paste0("USER_HEADER=", normalizePath(path = wrapper_file)))),
           stanc_args = as.list(unique(c(as.character(x = unlist(x = stanc_args)), "--allow-undefined"))),
           release_header_md5 = unname(obj = tools::md5sum(files = release_header)))
}

R_fn_bridge_memory_page_release_patch_lines <-  function(allocator_methods = NULL) {
      if (is.null(x = allocator_methods)) {
            # allocator_methods <-  system.file("include", "NicoStan", "runtime",
            #                                  "general_functions",
            #                                  "bridge_memory_allocator_methods.inc",
            #                                  package = "NicoStan", mustWork = TRUE)
            ## (the file of the one NicoStan install, never a side library: fn_file_of_the_one_NicoStan_install())
            allocator_methods <-  fn_file_of_the_one_NicoStan_install(
                  "include", "NicoStan", "runtime", "general_functions",
                  "bridge_memory_allocator_methods.inc",
                  package = "NicoStan", mustWork = TRUE)
      }
      allocator_methods <-  normalizePath(path = allocator_methods, mustWork = TRUE)
      method_lines <-  readLines(con = allocator_methods, warn = FALSE)
      if (length(x = method_lines) == 0 ||
          !identical(x = method_lines[1], y = "#ifdef __linux__") ||
          !identical(x = method_lines[length(x = method_lines)], y = "#endif")) {
            stop("Unexpected BridgeStan page-release method snippet: ", allocator_methods)
      }
      c("#ifdef __linux__", "#include <sys/mman.h>", "#include <unistd.h>",
        "#endif", method_lines)
}

R_fn_bridge_memory_patch_md5 <-  function(patch_lines) {
      patch_file <-  tempfile(pattern = "nicostan_page_release_patch_")
      on.exit(unlink(x = patch_file, force = TRUE), add = TRUE)
      writeLines(text = patch_lines, con = patch_file, useBytes = TRUE)
      unname(obj = tools::md5sum(files = patch_file))
}

R_fn_bridge_memory_patch_stack_alloc <-  function(stack_alloc_file, patch_lines) {
      stack_lines <-  readLines(con = stack_alloc_file, warn = FALSE)
      if (any(grepl(pattern = "nicostan_discard_pages", x = stack_lines, fixed = TRUE))) {
            stop("BridgeStan stack_alloc.hpp already contains page-release methods: ", stack_alloc_file)
      }
      include_anchor <-  which(stack_lines == "#include <vector>")
      public_anchor <-  which(stack_lines == " public:")
      if (length(x = include_anchor) != 1 || length(x = public_anchor) != 1) {
            stop("Unexpected BridgeStan stack_alloc.hpp patch anchors: ", stack_alloc_file)
      }
      stack_lines <-  append(x = stack_lines, values = patch_lines[1:4],
                            after = include_anchor)
      public_anchor <-  which(stack_lines == " public:")
      stack_lines <-  append(x = stack_lines, values = patch_lines[-(1:4)],
                            after = public_anchor)
      writeLines(text = stack_lines, con = stack_alloc_file, useBytes = TRUE)
      invisible(stack_alloc_file)
}

##      terms, the adjoint adds of precomputed_gradients_vari) done in parallel on the chain's TBB threads; the
##      C++ is reduce_sum_parallel_bookkeeping.inc (next to bridge_memory_allocator_methods.inc)
R_fn_bridge_memory_reduce_sum_patch_lines <-  function(patch_inc = NULL) {
      if (is.null(x = patch_inc)) {
            # patch_inc <-  system.file("include", "NicoStan", "runtime", "general_functions",
            #                          "reduce_sum_parallel_bookkeeping.inc",
            #                          package = "NicoStan", mustWork = TRUE)
            ## (the file of the one NicoStan install, never a side library: fn_file_of_the_one_NicoStan_install())
            patch_inc <-  fn_file_of_the_one_NicoStan_install(
                  "include", "NicoStan", "runtime", "general_functions",
                  "reduce_sum_parallel_bookkeeping.inc",
                  package = "NicoStan", mustWork = TRUE)
      }
      readLines(con = normalizePath(path = patch_inc, mustWork = TRUE), warn = FALSE)
}

R_fn_bridge_memory_patch_reduce_sum <-  function(reduce_sum_file, patch_lines) {
      lines <-  readLines(con = reduce_sum_file, warn = FALSE)
      if (any(grepl(pattern = "nicostan_parallel_precomputed_gradients_vari", x = lines, fixed = TRUE))) {
            stop("BridgeStan reduce_sum.hpp already carries NicoStan's parallel bookkeeping: ", reduce_sum_file)
      }
      namespace_anchor <-  which(lines == "namespace stan {")[1]
      save_anchor <-  which(lines == "    save_varis(varis, vmapped);")
      return_anchor <-  which(lines == "    return var(new precomputed_gradients_vari(")
      if (is.na(namespace_anchor) || length(x = save_anchor) != 1 || length(x = return_anchor) != 1 ||
          !grepl("worker.sum_, num_vars_sliced_terms \\+ num_vars_shared_terms, varis,", lines[return_anchor + 1]) ||
          !grepl("partials\\)\\);", lines[return_anchor + 2])) {
            stop("Unexpected BridgeStan reduce_sum.hpp patch anchors: ", reduce_sum_file)
      }
      ## (the parallel section: TBB's reduce kept, or the OpenMP team when NICOSTAN_REDUCE_SUM_OPENMP is defined)
      isolate_anchor <-  which(lines == "    tbb::this_task_arena::isolate([&] {")
      if (length(x = isolate_anchor) != 1) stop("Unexpected reduce_sum.hpp isolate anchor: ", reduce_sum_file)
      isolate_end <-  isolate_anchor + which(lines[(isolate_anchor + 1):length(x = lines)] == "    });")[1]
      if (is.na(isolate_end)) stop("Unexpected reduce_sum.hpp isolate end: ", reduce_sum_file)
      ## (the in-place replacements first, whilst every anchor still points at its line; then the insertions,
      ##  from the bottom of the file upwards, so that no insertion moves a line still to be edited)
      lines[save_anchor] <-  "    internal::nicostan_save_varis_of_sliced_terms(varis, vmapped, num_vars_per_term);"
      lines[return_anchor] <-  "    return var(new nicostan_parallel_precomputed_gradients_vari("
      lines[return_anchor + 1] <-  "        worker.sum_, num_vars_sliced_terms, num_vars_sliced_terms + num_vars_shared_terms, varis,"
      lines <-  append(x = lines, values = c("#else", "    internal::nicostan_reduce_sum_openmp(worker, num_terms, grainsize);", "#endif"),
                       after = isolate_end)
      lines <-  append(x = lines, values = "#if !defined(NICOSTAN_REDUCE_SUM_OPENMP)", after = isolate_anchor - 1)
      lines <-  append(x = lines, values = patch_lines, after = namespace_anchor - 1)
      writeLines(text = lines, con = reduce_sum_file, useBytes = TRUE)
      invisible(reduce_sum_file)
}

## ---- NicoStan's third runtime patch: reduce_sum's lean slicing (its bookkeeping of the sliced terms done inside
##      each chunk: each chunk writes the pointers of its own terms and zeroes its own partials, in place of the
##      passes over all N terms on the main thread; the same gradient); its description is
##      reduce_sum_lean_slicing.inc. Applied when options(NicoStan.reduce_sum_lean_slicing = TRUE); never together
##      with the parallel bookkeeping patch (each is measured on its own)
R_fn_bridge_memory_reduce_sum_lean_slicing_patch_lines <-  function(patch_inc = NULL) {
      if (is.null(x = patch_inc)) {
            # patch_inc <-  system.file("include", "NicoStan", "runtime", "general_functions",
            #                          "reduce_sum_lean_slicing.inc",
            #                          package = "NicoStan", mustWork = TRUE)
            ## (the file of the one NicoStan install, never a side library: fn_file_of_the_one_NicoStan_install())
            patch_inc <-  fn_file_of_the_one_NicoStan_install(
                  "include", "NicoStan", "runtime", "general_functions",
                  "reduce_sum_lean_slicing.inc",
                  package = "NicoStan", mustWork = TRUE)
      }
      readLines(con = normalizePath(path = patch_inc, mustWork = TRUE), warn = FALSE)
}

R_fn_bridge_memory_patch_reduce_sum_lean_slicing <-  function(reduce_sum_file, patch_lines) {
      lines <-  readLines(con = reduce_sum_file, warn = FALSE)
      if (any(grepl(pattern = "nicostan_sliced_varis_", x = lines, fixed = TRUE))) {
            stop("BridgeStan reduce_sum.hpp already carries NicoStan's lean slicing: ", reduce_sum_file)
      }
      if (any(grepl(pattern = "nicostan_parallel_precomputed_gradients_vari", x = lines, fixed = TRUE))) {
            stop("NicoStan's lean slicing is not combined with its parallel bookkeeping: ", reduce_sum_file)
      }
      namespace_anchor <-  which(lines == "namespace stan {")[1]
      member_anchor <-  which(lines == "    double sum_{0.0};")
      split_anchor <-  which(lines == "          args_tuple_(other.args_tuple_) {}")
      copy_loop_anchor <-  which(lines == "      for (size_t i = r.begin(); i < r.end(); ++i) {")
      accumulate_anchor <-  which(lines == paste0("      accumulate_adjoints(sliced_partials_ + r.begin() * ",
                                                  "num_vars_per_term_,"))
      save_anchor <-  which(lines == "    save_varis(varis, vmapped);")
      zero_anchor <-  which(lines == "    for (size_t i = 0; i < num_vars_sliced_terms; ++i) {")
      worker_anchor <-  which(lines == "                             std::forward<Args>(args)...);")
      if (is.na(namespace_anchor) || length(x = member_anchor) != 1 || length(x = split_anchor) != 1 ||
          length(x = copy_loop_anchor) != 1 || length(x = accumulate_anchor) != 1 ||
          length(x = save_anchor) != 1 || length(x = zero_anchor) != 1 || length(x = worker_anchor) != 1 ||
          lines[copy_loop_anchor + 1] != "        local_sub_slice.emplace_back(deep_copy_vars(vmapped_[i]));" ||
          lines[accumulate_anchor + 1] != "                          std::move(local_sub_slice));" ||
          lines[zero_anchor + 1] != "      partials[i] = 0.0;" || lines[zero_anchor + 2] != "    }" ||
          lines[worker_anchor - 2] != paste0("    recursive_reducer worker(num_vars_per_term, ",
                                             "num_vars_shared_terms, partials,")) {
            stop("Unexpected BridgeStan reduce_sum.hpp lean slicing anchors: ", reduce_sum_file)
      }
      ## (every replacement is in place, so that every anchor still points at its line; then the insertion)
      ##
      ## ---- the list of the sliced terms' nodes, written by the chunks
      lines[member_anchor] <-  paste0("    double sum_{0.0};\n",
                                      "    vari** nicostan_sliced_varis_{nullptr};  ",
                                      "// (NicoStan: each chunk writes its own terms' nodes here)")
      lines[split_anchor] <-  paste0("          args_tuple_(other.args_tuple_) {\n",
                                     "      nicostan_sliced_varis_ = other.nicostan_sliced_varis_;\n",
                                     "    }")
      ##
      ## ---- each chunk: the pointers of its own terms, from the terms it copies onto its own tape
      lines[copy_loop_anchor] <-  paste0("      vari** nicostan_next_vari = nicostan_sliced_varis_",
                                         " + r.begin() * num_vars_per_term_;\n",
                                         "      for (size_t i = r.begin(); i < r.end(); ++i) {")
      lines[copy_loop_anchor + 1] <-  paste0("        nicostan_next_vari = save_varis(nicostan_next_vari, ",
                                             "vmapped_[i]);\n",
                                             "        local_sub_slice.emplace_back(deep_copy_vars(vmapped_[i]));")
      ##
      ## ---- each chunk: zeroes its own partials just before it accumulates them
      lines[accumulate_anchor] <-  paste0("      std::fill(sliced_partials_ + r.begin() * num_vars_per_term_,\n",
                                          "                sliced_partials_ + r.end() * num_vars_per_term_,",
                                          " 0.0);\n",
                                          "      accumulate_adjoints(sliced_partials_ + r.begin() * ",
                                          "num_vars_per_term_,")
      ##
      ## ---- the main thread: no pass over all N terms (the shared arguments' pointers are still written here)
      lines[save_anchor] <-  "    //// (NicoStan lean slicing: each chunk writes the pointers of its own terms)"
      lines[zero_anchor] <-  "    //// (NicoStan lean slicing: each chunk zeroes its own partials)"
      lines[zero_anchor + 1] <-  "    ////"
      lines[zero_anchor + 2] <-  "    ////"
      lines[worker_anchor] <-  paste0("                             std::forward<Args>(args)...);\n",
                                      "    worker.nicostan_sliced_varis_ = varis;")
      lines <-  append(x = lines, values = patch_lines, after = namespace_anchor - 1)
      writeLines(text = lines, con = reduce_sum_file, useBytes = TRUE)
      invisible(reduce_sum_file)
}

## ---- part of the lean slicing patch: BridgeStan's gradient call without a copy of the parameters and without
##      a heap allocation per gradient (BridgeStan copies the N unconstrained values into a new vector and
##      stan::math::gradient makes a new vector of N nodes on every call); the nodes of the parameters live in a
##      per-thread vector reused across calls, with the same arithmetic as stan::math::gradient
R_fn_bridge_memory_patch_bridgestan_gradient_without_copies <-  function(model_rng_file) {
      lines <-  readLines(con = model_rng_file, warn = FALSE)
      if (any(grepl(pattern = "nicostan_parameters", x = lines, fixed = TRUE))) {
            stop("BridgeStan model_rng.cpp already carries NicoStan's gradient without copies: ", model_rng_file)
      }
      copy_anchor <-  which(lines == "  Eigen::VectorXd params_unc = Eigen::VectorXd::Map(theta_unc, N);")
      gradient_anchor <-  which(lines == "  stan::math::gradient(logp, params_unc, *val, grad, grad + N);")
      if (length(x = copy_anchor) != 1 || length(x = gradient_anchor) != 1 ||
          gradient_anchor != copy_anchor + 1 || lines[copy_anchor - 1] != "  int N = param_unc_num_;") {
            stop("Unexpected BridgeStan model_rng.cpp gradient anchors: ", model_rng_file)
      }
      lines[copy_anchor] <-  paste0("  //// (NicoStan lean slicing: no copy of the parameters and no heap ",
                                    "allocation per gradient)\n",
                                    "  stan::math::nested_rev_autodiff nicostan_nested;\n",
                                    "  static thread_local Eigen::Matrix<stan::math::var, -1, 1> ",
                                    "nicostan_parameters;\n",
                                    "  if (nicostan_parameters.size() != N) nicostan_parameters.resize(N);\n",
                                    "  for (int i = 0; i < N; ++i) {\n",
                                    "    nicostan_parameters.coeffRef(i) = stan::math::var(theta_unc[i]);\n",
                                    "  }")
      lines[gradient_anchor] <-  paste0("  stan::math::var nicostan_log_density = logp(nicostan_parameters);\n",
                                        "  *val = nicostan_log_density.val();\n",
                                        "  stan::math::grad(nicostan_log_density.vi_);\n",
                                        "  for (int i = 0; i < N; ++i) {\n",
                                        "    grad[i] = nicostan_parameters.coeff(i).adj();\n",
                                        "  }")
      writeLines(text = lines, con = model_rng_file, useBytes = TRUE)
      invisible(model_rng_file)
}

## ---- NicoStan's fourth runtime patch: the nuisance block owned by NicoStan (on top of the lean slicing, which
##      it needs): reduce_sum's chunks read the nuisance values straight from NicoStan's array and add their
##      adjoints straight into NicoStan's nuisance gradient, BridgeStan's gradient call adds the adjoints of the
##      values read on the main tape, and the model library exports
##      nicostan_bs_log_density_gradient_with_the_nuisance_block;
##      its description is reduce_sum_nuisance_block_owned_by_NicoStan.inc. Applied when
##      options(NicoStan.reduce_sum_nuisance_block = TRUE) (with options(NicoStan.reduce_sum_lean_slicing = TRUE))
## ---- private build-time proof: every reduction is the complete right-hand side of a model target addition.
##      Calls inside functions, included source, weights and nesting are deliberately not proved here.
R_fn_bridge_memory_build_controls_allow_unit_proof <-  function(make_args) {
      denied <-  function(reason) structure(FALSE, reason = reason)
      arguments <-  as.character(unlist(make_args))
      known_keys <-  c("STAN_THREADS", "PRECOMPILED_HEADERS", "CXXFLAGS_OPTIM", "CPPFLAGS_OPTIM",
                       "CXXFLAGS_OS", "CXXFLAGS", "LDLIBS_OS", "USER_HEADER")
      keys <-  sub(pattern = "=.*$", replacement = "", x = arguments)
      unknown <-  !(keys %in% known_keys) | !grepl(pattern = "=", x = arguments, fixed = TRUE)
      if (any(unknown)) return(denied(paste0("uninspected make assignment: ", arguments[which(unknown)[1]])))
      if (any(duplicated(keys))) return(denied("duplicate make assignments are not proved"))
      known_flags <-  c("-O0", "-O1", "-O2", "-O3", "-Os", "-Og", "-g", "-g0", "-g1", "-g2", "-g3",
                        "-march=native", "-mtune=native", "-mfma", "-mavx", "-mavx2", "-mavx512f", "-mavx512vl",
                        "-mavx512dq", "-fopenmp", "-pthread", "-fPIC", "-fpic", "-pipe",
                        "-std=c++14", "-std=c++17", "-std=c++20", "-Wall", "-Wextra",
                        "-Wno-deprecated-declarations", "-Wno-ignored-attributes", "-Wno-sign-compare",
                        "-fno-math-errno", "-fno-signed-zeros", "-fno-trapping-math")
      known_defines <-  c("BAYESMVP_FORCE_AVX2", "_REENTRANT", "NICOSTAN_REDUCE_SUM_OPENMP", "NDEBUG",
                          "BOOST_DISABLE_ASSERTS")
      known_flags <-  c(known_flags, paste0("-D", known_defines), paste0("-D", known_defines, "=1"))
      for (index in seq_along(arguments)) {
            value <-  sub(pattern = "^[^=]*=", replacement = "", x = arguments[index])
            value <-  sub(pattern = "^(['\"])(.*)\\1$", replacement = "\\2", x = value, perl = TRUE)
            if (keys[index] == "USER_HEADER") next
            if (keys[index] == "STAN_THREADS" && identical(value, "true")) next
            if (keys[index] == "PRECOMPILED_HEADERS" && identical(value, "false")) next
            if (keys[index] == "LDLIBS_OS" && identical(value, "-lomp")) next
            if (keys[index] == "LDLIBS_OS" && startsWith(value, "/") && file.exists(value)) {
                  libomp <-  R_fn_bridge_memory_libomp_of_NicoStan()
                  if (!is.null(libomp) && identical(normalizePath(value), normalizePath(libomp))) next
            }
            if (!(keys[index] %in% c("CXXFLAGS_OPTIM", "CPPFLAGS_OPTIM", "CXXFLAGS_OS", "CXXFLAGS"))) {
                  return(denied(paste0("uninspected make value: ", arguments[index])))
            }
            tokens <-  strsplit(trimws(value), "[[:space:]]+")[[1]]
            unknown_tokens <-  tokens[!(tokens %in% known_flags) & nzchar(tokens)]
            if (length(unknown_tokens) > 0) {
                  return(denied(paste0("uninspected compiler flag: ", unknown_tokens[1])))
            }
      }
      controls <-  c("CPATH", "CPLUS_INCLUDE_PATH", "C_INCLUDE_PATH", "OBJC_INCLUDE_PATH", "CPPFLAGS",
                     "CXXFLAGS", "CXX", "CXX14", "CXX17", "CXX20", "CC", "CPP", "STANCFLAGS",
                     "CPPFLAGS_OPTIM", "CPPFLAGS_OS", "CXXFLAGS_OPTIM", "CXXFLAGS_OS", "LDFLAGS", "LDLIBS_OS",
                     "USER_HEADER", "MAKEFLAGS", "MFLAGS", "GNUMAKEFLAGS", "MAKEFILES", "COMPILER_PATH",
                     "GCC_EXEC_PREFIX")
      inherited <-  controls[nzchar(Sys.getenv(controls))]
      if (length(inherited) > 0) {
            return(denied(paste0("uninspected inherited build control: ", inherited[1])))
      }
      original <-  tryCatch(utils::getFromNamespace("get_bridgestan_path", "bridgestan")(download = FALSE),
                            error = function(e) NULL)
      if (is.null(original)) return(denied("the original BridgeStan make/local could not be inspected"))
      local_file <-  file.path(original, "make", "local")
      if (file.exists(local_file)) {
            local_lines <-  trimws(sub(pattern = "#.*$", replacement = "",
                                       x = readLines(local_file, warn = FALSE)))
            for (line in local_lines[nzchar(local_lines)]) {
                  if (identical(line, "STAN_THREADS=true")) next
                  assignment <-  regmatches(line, regexec(
                        pattern = "^(CXXFLAGS|CPPFLAGS)\\s*\\+=\\s*(.*)$", text = line, perl = TRUE))[[1]]
                  if (length(assignment) != 3) {
                        return(denied(paste0("uninspected local build assignment: ", local_file)))
                  }
                  tokens <-  strsplit(trimws(assignment[3]), "[[:space:]]+")[[1]]
                  if (any(!(tokens %in% known_flags))) {
                        return(denied(paste0("uninspected local compiler flag: ", local_file)))
                  }
            }
      }
      return(TRUE)
}

R_fn_bridge_memory_external_functions_of_the_build <-  function(make_args) {
      result <-  list(closure_proven = FALSE, names = character(0), files = character(0),
                      reason = "no external function definition was proved")
      arguments <-  as.character(unlist(make_args))
      entries <-  grep(pattern = "^USER_HEADER=", x = arguments, value = TRUE)
      if (length(entries) != 1) return(result)
      header <-  sub(pattern = "^USER_HEADER=", replacement = "", x = entries)
      if (!file.exists(header)) return(result)
      controls <-  R_fn_bridge_memory_build_controls_allow_unit_proof(make_args = make_args)
      if (!isTRUE(controls)) {
            result$reason <-  attr(controls, "reason")
            return(result)
      }
      root <-  tryCatch(fn_file_of_the_one_NicoStan_install("include", "NicoSIMD", package = "NicoSIMD",
                                                            mustWork = FALSE), error = function(e) "")
      if (!nzchar(root) || !dir.exists(root)) return(result)
      root <-  normalizePath(root, mustWork = TRUE)
      roots <-  root
      parent <-  dirname(normalizePath(header, mustWork = TRUE))
      repeat {
            package_root <-  file.path(parent, "R_packages", "NicoStan", "inst", "NicoSIMD")
            description <-  file.path(package_root, "DESCRIPTION")
            development_root <-  file.path(package_root, "inst", "include", "NicoSIMD")
            if (file.exists(description) && dir.exists(development_root)) {
                  package <-  read.dcf(description, fields = "Package")
                  if (identical(unname(package[1, 1]), "NicoSIMD")) {
                        roots <-  c(roots, normalizePath(development_root, mustWork = TRUE))
                  }
            }
            if (identical(parent, dirname(parent))) break
            parent <-  dirname(parent)
      }
      runtime_root <-  fn_file_of_the_one_NicoStan_install("include", "NicoStan", "runtime",
                                                           "general_functions", package = "NicoStan",
                                                           mustWork = TRUE)
      runtime_files <-  normalizePath(file.path(runtime_root, c(
            "bridge_memory.hpp", "stan_external_nuisance_values_of_the_block_owned_by_NicoStan.hpp")),
            mustWork = TRUE)
      installed_runtime_root <-  system.file("include", "NicoStan", "runtime", "general_functions",
                                             package = "NicoStan", mustWork = TRUE)
      installed_runtime_files <-  normalizePath(file.path(installed_runtime_root, c(
            "bridge_memory.hpp", "stan_external_nuisance_values_of_the_block_owned_by_NicoStan.hpp")),
            mustWork = TRUE)
      runtime_files <-  unique(c(runtime_files, installed_runtime_files))
      pending <-  normalizePath(header, mustWork = TRUE)
      visited <-  character(0)
      definitions <-  character(0)
      while (length(pending) > 0) {
            path <-  pending[1]
            pending <-  pending[-1]
            if (path %in% visited || path %in% runtime_files) next
            visited <-  c(visited, path)
            lines <-  readLines(path, warn = FALSE)
            matching_runtime_files <-  runtime_files[basename(runtime_files) == basename(path)]
            if (length(matching_runtime_files) > 0 && any(vapply(matching_runtime_files,
                  function(known_file) identical(lines, readLines(known_file, warn = FALSE)), logical(1)))) {
                  next
            }
            code <-  paste(lines, collapse = "\n")
            code <-  gsub(pattern = "/\\*[\\s\\S]*?\\*/|//[^\\n]*", replacement = " ",
                           x = code, perl = TRUE)
            matched_roots <-  roots[startsWith(path, paste0(roots, "/"))]
            selected_root <-  if (length(matched_roots) > 0) matched_roots[which.max(nchar(matched_roots))] else
                              root
            relative <-  if (startsWith(path, paste0(selected_root, "/"))) {
                               substring(path, nchar(selected_root) + 2)
                         } else
                         sub(pattern = "^.*[/]NicoSIMD[/]", replacement = "", x = path)
            canonical <-  file.path(root, relative)
            is_canonical <-  length(matched_roots) > 0 ||
                             (file.exists(canonical) && identical(lines, readLines(canonical, warn = FALSE)))
            includes <-  regmatches(code, gregexpr(
                  pattern = "(?m)^\\s*#\\s*include\\s*[<\"][^>\"]+[>\"]", text = code,
                  perl = TRUE))[[1]]
            if (is_canonical) {
                  executable <-  gsub(pattern = "\"(?:[^\"\\\\]|\\\\.)*\"", replacement = " ",
                                       x = code, perl = TRUE)
                  if (grepl(pattern = paste0("\\breduce_sum(?:_static)?\\b|",
                                              "\\bnicostan_nuisance_[A-Za-z_]*\\b"),
                            x = executable, perl = TRUE)) {
                        result$reason <-  paste0("external include needs general upstream support: ", path)
                        return(result)
                  }
                  if (grepl(pattern = "(?m)^\\s*#\\s*include\\s+[^<\"\\s]", x = code, perl = TRUE)) {
                        result$reason <-  paste0("an external include is not literal: ", path)
                        return(result)
                  }
                  definitions <-  c(definitions, code)
                  result$files <-  c(result$files, path)
            } else {
                  remainder <-  gsub(pattern = "(?m)^\\s*#\\s*include[^\\n]*", replacement = "",
                                      x = code, perl = TRUE)
                  remainder <-  gsub(pattern = "(?m)^\\s*#\\s*pragma\\s+once[^\\n]*", replacement = "",
                                      x = remainder, perl = TRUE)
                  remainder <-  gsub(pattern = paste0("(?m)^\\s*#\\s*(?:if|elif)\\s+",
                                                       "__has_include\\([^\\n]*|",
                                                       "^\\s*#\\s*(?:else|endif)\\s*$"),
                                      replacement = "", x = remainder, perl = TRUE)
                  if (nzchar(trimws(remainder))) {
                        result$reason <-  paste0("external code is not a canonical NicoSIMD definition: ", path)
                        return(result)
                  }
            }
            for (include in includes) {
                  name <-  sub(pattern = "^[^<\"]*[<\"]([^>\"]+)[>\"]$", replacement = "\\1",
                                x = include)
                  ## (SIMDe supplies the same numeric intrinsics as immintrin.h, not Stan callbacks;
                  ##  this exact canonical header includes it on ARM, similarly to Eigen's numeric headers)
                  if (is_canonical && relative == "nico_simd_intrinsics.hpp" &&
                      name %in% c("../simde/x86/avx2.h", "../simde/x86/fma.h")) next
                  if (startsWith(name, "NicoSIMD/")) {
                        candidate <-  file.path(dirname(selected_root), name)
                  } else if (startsWith(name, "/")) {
                        candidate <-  name
                  } else if (grepl(pattern = "\"", x = include, fixed = TRUE)) {
                        candidate <-  file.path(dirname(path), name)
                  } else {
                        standard <-  c("algorithm", "array", "atomic", "cassert", "cfloat", "cmath",
                                       "cstddef", "cstdint", "cstdlib", "cstring", "dlfcn.h",
                                       "functional", "immintrin.h",
                                       "limits", "memory", "new", "numeric", "ostream", "stdexcept",
                                       "string", "type_traits", "utility", "vector", "omp.h")
                        if (name %in% standard || grepl(pattern = "^(stan/|Eigen/|boost/|tbb/|simde/)",
                                                       x = name)) next
                        result$reason <-  paste0("unresolved external dependency: ", name)
                        return(result)
                  }
                  if (!file.exists(candidate)) {
                        if (!is_canonical && grepl(pattern = "__has_include", x = code, fixed = TRUE)) next
                        result$reason <-  paste0("missing external include: ", candidate)
                        return(result)
                  }
                  pending <-  c(pending, normalizePath(candidate, mustWork = TRUE))
            }
      }
      code <-  paste(definitions, collapse = "\n")
      bodies <-  regmatches(code, gregexpr(
            pattern = "\\b(?:nico_[A-Za-z0-9_]+|bmvp_simd_lanes)\\s*\\([^;{}]*\\)\\s*\\{", text = code,
            perl = TRUE))[[1]]
      names <-  sub(pattern = "\\s*\\([\\s\\S]*$", replacement = "", x = bodies, perl = TRUE)
      macros <-  regmatches(code, gregexpr(
            pattern = "\\bBMVP_(?:STAN_EXTERNAL|FASTER_REV_EXTERNAL)\\s*\\(\\s*nico_[A-Za-z0-9_]+",
            text = code, perl = TRUE))[[1]]
      if (length(macros) > 0 &&
          grepl(pattern = "#define BMVP_STAN_EXTERNAL", x = code, fixed = TRUE) &&
          grepl(pattern = "inline VecVar NAME(", x = code, fixed = TRUE)) {
            names <-  c(names, sub(pattern = "^.*\\(\\s*", replacement = "", x = macros, perl = TRUE))
      }
      result$names <-  unique(names)
      result$closure_proven <-  TRUE
      result$reason <-  "external definitions and literal include dependencies contain no nested reduction"
      return(result)
}

R_fn_bridge_memory_proof_of_unit_upstream <-  function(model_file, make_args) {
      code <-  paste(readLines(con = model_file, warn = FALSE), collapse = "\n")
      code <-  R_fn_Stan_parallel_call_names(code = code)
      protected <-  "\"(?:[^\"\\\\]|\\\\.)*\"|/\\*[\\s\\S]*?\\*/|//[^\\n]*"
      code <-  gsub(pattern = protected, replacement = " ", x = code, perl = TRUE)
      calls <-  gregexpr(pattern = "\\breduce_sum(?:_static)?\\s*\\(", text = code, perl = TRUE)[[1]]
      calls <-  calls[calls > 0]
      proof <-  list(proven = FALSE, number_of_calls = length(calls),
                     reason = "a reduction is not proved to have unit upstream")
      proof$map_rect_chunked <-  grepl(pattern = "\\bmap_rect\\s*\\(", x = code, perl = TRUE)
      controls <-  R_fn_bridge_memory_build_controls_allow_unit_proof(make_args = make_args)
      if (!isTRUE(controls)) {
            proof$reason <-  attr(controls, "reason")
            return(proof)
      }
      if (grepl(pattern = "#", x = code, fixed = TRUE) || length(calls) == 0) return(proof)
      external <-  NULL
      if (any(startsWith(as.character(unlist(make_args)), "USER_HEADER="))) {
            external <-  R_fn_bridge_memory_external_functions_of_the_build(make_args = make_args)
            proof$external <-  external
            if (!isTRUE(external$closure_proven)) {
                  proof$reason <-  external$reason
                  return(proof)
            }
      }
      characters <-  strsplit(code, "", fixed = TRUE)[[1]]
      depth <-  0
      model_start <-  NA_real_
      model_end <-  NA_real_
      functions_start <-  NA_real_
      functions_end <-  NA_real_
      statement_start <-  NA_real_
      known_readers <-  c("nicostan_nuisance_value", "nicostan_nuisance_vector",
                         "nicostan_nuisance_vector_at", "nicostan_nuisance_array_of_reals",
                         "nicostan_nuisance_array_of_reals_at", "nicostan_nuisance_array_of_vectors",
                         "nicostan_nuisance_array_of_row_vectors", "nicostan_nuisance_matrix_of_terms")
      for (position in seq_along(characters)) {
            if (characters[position] == "{") {
                  if (depth == 0) {
                        prefix <-  substr(code, 1, position - 1)
                        if (grepl(pattern = "(^|})\\s*model\\s*$", x = prefix, perl = TRUE)) {
                              model_start <-  position
                        }
                        if (grepl(pattern = "(^|})\\s*functions\\s*$", x = prefix, perl = TRUE)) {
                              functions_start <-  position
                              statement_start <-  position + 1
                        }
                  }
                  depth <-  depth + 1
            }
            if (characters[position] == "}") {
                  depth <-  depth - 1
                  if (depth == 0 && !is.na(model_start) && is.na(model_end)) model_end <-  position
                  if (depth == 0 && !is.na(functions_start) && is.na(functions_end)) functions_end <-  position
                  if (depth == 1 && !is.na(functions_start) && is.na(functions_end)) {
                        statement_start <-  position + 1
                  }
            }
            if (characters[position] == ";" && depth == 1 &&
                !is.na(functions_start) && is.na(functions_end)) {
                  declaration <-  substr(code, statement_start, position)
                  name <-  regmatches(declaration, regexec(
                        pattern = "\\b([A-Za-z_][A-Za-z0-9_]*)\\s*\\(", text = declaration, perl = TRUE))[[1]]
                  if (length(name) == 2 && !(name[2] %in% known_readers)) {
                        identifier <-  name[2]
                        if (grepl(pattern = "_lpdf$", x = identifier)) {
                              identifier <-  paste0(sub("_lpdf$", "", identifier), "(?:_lpdf|_lupdf)?")
                        } else if (grepl(pattern = "_lpmf$", x = identifier)) {
                              identifier <-  paste0(sub("_lpmf$", "", identifier), "(?:_lpmf|_lupmf)?")
                        }
                        occurrences <-  gregexpr(pattern = paste0("\\b", identifier, "\\b"), text = code,
                                                 perl = TRUE)[[1]]
                        ## (a declaration alone cannot call a hidden reduction; density aliases count too)
                        if (sum(occurrences > 0) == 1) {
                              statement_start <-  position + 1
                              next
                        }
                  }
                  if (length(name) == 2 && !(name[2] %in% known_readers) && is.null(external)) {
                        external <-  R_fn_bridge_memory_external_functions_of_the_build(make_args = make_args)
                  }
                  if (length(name) != 2 || !(name[2] %in% c(known_readers, external$names))) {
                        proof$reason <-  "an external function body is unavailable to the unit-upstream proof"
                        if (!is.null(external)) proof$external <-  external
                        return(proof)
                  }
                  statement_start <-  position + 1
            }
      }
      if (is.na(model_start) || is.na(model_end) || depth != 0) return(proof)
      if (isTRUE(proof$map_rect_chunked) && !is.na(functions_start) && !is.na(functions_end)) {
            functions_code <-  substr(code, functions_start, functions_end)
            if (grepl(pattern = "\\bmap_rect\\s*\\(", x = functions_code, perl = TRUE)) {
                  proof$reason <-  paste0("map calls inside user functions retain general nuisance ",
                                          "derivatives for possible nested reductions")
                  return(proof)
            }
      }
      for (position in calls) {
            if (position <= model_start || position >= model_end) return(proof)
            prefix <-  substr(code, model_start, position - 1)
            if (!grepl(pattern = "[;{}]\\s*target\\s*\\+=\\s*$", x = prefix, perl = TRUE)) return(proof)
            opening <-  position + regexpr(pattern = "\\(", text = substring(code, position)) - 1
            nesting <-  0
            closing <-  NA_real_
            for (index in seq.int(opening, nchar(code))) {
                  if (characters[index] == "(") nesting <-  nesting + 1
                  if (characters[index] == ")") nesting <-  nesting - 1
                  if (nesting == 0) {
                        closing <-  index
                        break
                  }
            }
            if (is.na(closing) || !grepl(pattern = "^\\s*;", x = substring(code, closing + 1))) return(proof)
      }
      proof$proven <-  TRUE
      proof$reason <-  "all reductions directly add their complete result to the model target"
      if (!is.null(external)) proof$external <-  external
      return(proof)
}

R_fn_bridge_memory_patch_reduce_sum_with_proven_unit_upstream <-  function(reduce_sum_file, patch_lines) {
      lines <-  readLines(con = reduce_sum_file, warn = FALSE)
      if (any(grepl(pattern = "nicostan_nuisance_block_of_the_caller_", x = lines, fixed = TRUE))) {
            stop("BridgeStan reduce_sum.hpp already carries NicoStan's nuisance block: ", reduce_sum_file)
      }
      namespace_anchor <-  which(lines == "namespace stan {")[1]
      member_anchor <-  which(lines == paste0("    vari** nicostan_sliced_varis_{nullptr};  ",
                                              "// (NicoStan: each chunk writes its own terms' nodes here)"))
      split_anchor <-  which(lines == "      nicostan_sliced_varis_ = other.nicostan_sliced_varis_;")
      nest_anchor <-  which(lines == "      const nested_rev_autodiff begin_nest;")
      gradient_anchor <-  which(lines == "      sub_sum_v.grad();")
      return_anchor <-  which(lines == "    return var(new precomputed_gradients_vari(")
      if (is.na(namespace_anchor) || length(member_anchor) != 1 || length(split_anchor) != 1 ||
          length(nest_anchor) != 1 || length(gradient_anchor) != 1 || length(return_anchor) != 1 ||
          !grepl("worker.sum_, num_vars_sliced_terms \\+ num_vars_shared_terms, varis,",
                 lines[return_anchor + 1]) || !grepl("partials\\)\\);", lines[return_anchor + 2])) {
            stop("Unexpected BridgeStan reduce_sum.hpp proven-unit anchors: ", reduce_sum_file)
      }
      lines[member_anchor] <-  paste0(lines[member_anchor], "\n",
            "    nicostan_nuisance_block_t* nicostan_nuisance_block_of_the_caller_{nicostan_nuisance_block};")
      lines[split_anchor] <-  paste0(lines[split_anchor], "\n",
            "      nicostan_nuisance_block_of_the_caller_ = other.nicostan_nuisance_block_of_the_caller_;")
      lines[nest_anchor] <-  paste0(lines[nest_anchor], "\n",
            "      nicostan_scope_of_a_chunk_of_the_nuisance_block nicostan_scope(\n",
            "          nicostan_nuisance_block_of_the_caller_, static_cast<int>(r.begin()),\n",
            "          static_cast<int>(r.end()) - 1);")
      lines[gradient_anchor] <-  paste0(lines[gradient_anchor], "\n",
            "      nicostan_add_the_adjoints_of_the_chunk_of_the_nuisance_block();")
      lines[return_anchor] <-  "    return var(new nicostan_precomputed_gradients_vari_of_the_nuisance_block("
      lines[return_anchor + 2] <-  paste0("        partials, nicostan_nuisance_block != nullptr &&\n",
            "                  nicostan_nuisance_block->read_inside_a_reduce_sum));")
      lines <-  append(x = lines, values = patch_lines, after = namespace_anchor - 1)
      writeLines(text = lines, con = reduce_sum_file, useBytes = TRUE)
      invisible(reduce_sum_file)
}

R_fn_bridge_memory_reduce_sum_nuisance_block_patch_lines <-  function(patch_inc = NULL) {
      if (is.null(x = patch_inc)) {
            # patch_inc <-  system.file("include", "NicoStan", "runtime", "general_functions",
            #                          "reduce_sum_nuisance_block_owned_by_NicoStan.inc",
            #                          package = "NicoStan", mustWork = TRUE)
            ## (the file of the one NicoStan install, never a side library: fn_file_of_the_one_NicoStan_install())
            patch_inc <-  fn_file_of_the_one_NicoStan_install(
                  "include", "NicoStan", "runtime", "general_functions",
                  "reduce_sum_nuisance_block_owned_by_NicoStan.inc",
                  package = "NicoStan", mustWork = TRUE)
      }
      readLines(con = normalizePath(path = patch_inc, mustWork = TRUE), warn = FALSE)
}

R_fn_bridge_memory_patch_reduce_sum_nuisance_block <-  function(reduce_sum_file, patch_lines) {
      lines <-  readLines(con = reduce_sum_file, warn = FALSE)
      if (any(grepl(pattern = "nicostan_nuisance_block_of_the_caller_", x = lines, fixed = TRUE))) {
            stop("BridgeStan reduce_sum.hpp already carries NicoStan's nuisance block: ", reduce_sum_file)
      }
      namespace_anchor <-  which(lines == "namespace stan {")[1]
      member_anchor <-  which(lines == paste0("    vari** nicostan_sliced_varis_{nullptr};  ",
                                              "// (NicoStan: each chunk writes its own terms' nodes here)"))
      split_anchor <-  which(lines == "      nicostan_sliced_varis_ = other.nicostan_sliced_varis_;")
      nest_anchor <-  which(lines == "      const nested_rev_autodiff begin_nest;")
      gradient_anchor <-  which(lines == "      sub_sum_v.grad();")
      join_anchor <-  which(lines == "      sum_ += rhs.sum_;")
      return_anchor <-  which(lines == "    return var(new precomputed_gradients_vari(")
      if (is.na(namespace_anchor) || length(x = member_anchor) != 1 || length(x = split_anchor) != 1 ||
          length(x = nest_anchor) != 1 || length(x = gradient_anchor) != 1 || length(x = join_anchor) != 1 ||
          length(x = return_anchor) != 1 ||
          !grepl("worker.sum_, num_vars_sliced_terms \\+ num_vars_shared_terms, varis,",
                 lines[return_anchor + 1]) ||
          !grepl("partials\\)\\);", lines[return_anchor + 2])) {
            stop("Unexpected BridgeStan reduce_sum.hpp nuisance block anchors (the lean slicing patch must be ",
                 "applied first): ", reduce_sum_file)
      }
      ## (every replacement is in place, so that every anchor still points at its line; then the insertion)
      ##
      ## ---- the caller's nuisance block, taken by the reducer on the caller's thread and handed to every split
      lines[member_anchor] <-  paste0(lines[member_anchor], "\n",
                                      "    nicostan_nuisance_block_t* nicostan_nuisance_block_of_the_caller_",
                                      "{nicostan_nuisance_block};\n",
                                      "    std::vector<std::pair<int, double>> nicostan_nuisance_derivatives_;")
      lines[split_anchor] <-  paste0(lines[split_anchor], "\n",
                                     "      nicostan_nuisance_block_of_the_caller_ = ",
                                     "other.nicostan_nuisance_block_of_the_caller_;")
      ##
      ## ---- each chunk: the caller's nuisance block and its own units, for its duration; after its own gradient,
      ##      the adjoints of the values it read
      lines[nest_anchor] <-  paste0(lines[nest_anchor], "\n",
                                    "      nicostan_scope_of_a_chunk_of_the_nuisance_block nicostan_scope(\n",
                                    "          nicostan_nuisance_block_of_the_caller_, ",
                                    "static_cast<int>(r.begin()),\n",
                                    "          static_cast<int>(r.end()) - 1);\n",
                                    "      nicostan_scope_of_the_nuisance_derivative_collection ",
                                    "nicostan_derivative_scope(\n",
                                    "          nicostan_nuisance_derivatives_);")
      lines[gradient_anchor] <-  paste0(lines[gradient_anchor], "\n",
                                        "      nicostan_add_the_adjoints_of_the_chunk_of_the_nuisance_block();")
      lines[join_anchor] <-  paste0(lines[join_anchor], "\n",
            "      nicostan_nuisance_derivatives_.insert(nicostan_nuisance_derivatives_.end(),\n",
            "          rhs.nicostan_nuisance_derivatives_.begin(), rhs.nicostan_nuisance_derivatives_.end());")
      ##
      ## ---- the node of the sum: an error if a sum whose chunks read the nuisance block is not added to the
      ##      target as it is
      ##      The returned numeric payload now applies the actual multiplier in the reverse pass.
      lines[return_anchor] <-  "    return var(new nicostan_precomputed_gradients_vari_of_the_nuisance_block("
      lines[return_anchor + 2] <-  paste0("        partials, worker.nicostan_nuisance_block_of_the_caller_,\n",
                                          "        worker.nicostan_nuisance_derivatives_));")
      lines <-  append(x = lines, values = patch_lines, after = namespace_anchor - 1)
      writeLines(text = lines, con = reduce_sum_file, useBytes = TRUE)
      invisible(reduce_sum_file)
}

## ---- part of the nuisance block patch: BridgeStan's gradient call adds the adjoints of the nuisance values read
##      on the main tape (before the main tape is recovered), and the model library exports the gradient call
##      nicostan_bs_log_density_gradient_with_the_nuisance_block (zeroes NicoStan's nuisance gradient, sets the
##      nuisance block for the duration of the gradient, then adds the adjoints that chunks kept for other units)
R_fn_bridge_memory_patch_bridgestan_gradient_with_the_nuisance_block <-  function(model_rng_file,
                                                                                bridgestan_file) {
      lines <-  readLines(con = model_rng_file, warn = FALSE)
      gradient_anchor <-  which(lines == "  stan::math::grad(nicostan_log_density.vi_);")
      log_density_anchor <-  which(lines == "  stan::math::var nicostan_log_density = logp(nicostan_parameters);")
      if (length(x = gradient_anchor) != 1 || length(x = log_density_anchor) != 1 ||
          any(grepl(pattern = "nodes_on_the_main_tape", x = lines, fixed = TRUE))) {
            stop("Unexpected BridgeStan model_rng.cpp nuisance block anchor (the lean slicing patch must be ",
                 "applied first, once): ", model_rng_file)
      }
      lines[log_density_anchor] <-  paste0(
            "  if (stan::math::internal::nicostan_nuisance_block != nullptr) {\n",
            "    stan::math::internal::nicostan_nuisance_block->identity_of_the_main_tape =\n",
            "        stan::math::ChainableStack::instance_;\n",
            "    stan::math::internal::nicostan_nuisance_block->nested_depth_of_the_main_tape =\n",
            "        stan::math::ChainableStack::instance_->nested_var_stack_sizes_.size();\n",
            "  }\n", lines[log_density_anchor])
      lines[gradient_anchor] <-  paste0(lines[gradient_anchor], "\n",
            "  //// (NicoStan's nuisance block: the adjoints of the nuisance values read on the main tape)\n",
            "  if (stan::math::internal::nicostan_nuisance_block != nullptr) {\n",
            "    stan::math::internal::nicostan_nuisance_block_t* nicostan_block =\n",
            "        stan::math::internal::nicostan_nuisance_block;\n",
            "    for (const auto& unit_and_node : nicostan_block->nodes_on_the_main_tape) {\n",
            "      nicostan_block->gradient[unit_and_node.first] += unit_and_node.second->adj_;\n",
            "    }\n",
            "    nicostan_block->nodes_on_the_main_tape.clear();\n",
            "    nicostan_block->nodes_by_value_on_the_main_tape.clear();\n",
            "    nicostan_block->identity_of_the_main_tape = nullptr;\n",
            "    nicostan_block->nested_depth_of_the_main_tape =\n",
            "        std::numeric_limits<std::size_t>::max();\n",
            "  }")
      writeLines(text = lines, con = model_rng_file, useBytes = TRUE)
      ##
      lines <-  readLines(con = bridgestan_file, warn = FALSE)
      if (any(grepl(pattern = "nicostan_bs_log_density_gradient_with_the_nuisance_block", x = lines,
                    fixed = TRUE))) {
            stop("BridgeStan bridgestan.cpp already exports NicoStan's nuisance block gradient: ",
                 bridgestan_file)
      }
      lines <-  c(lines, "",
            paste0("//// NicoStan's nuisance block: the log density and gradient of a model in NicoStan's ",
                   "nuisance-block form,"),
            "//// with the nuisance values read from nuisance_values and their gradient written into",
            "//// nuisance_gradient (theta_unc and grad hold the main parameters only)",
            "extern \"C\" BS_PUBLIC int nicostan_bs_log_density_gradient_with_the_nuisance_block(",
            "    const bs_model* m, bool propto, bool jacobian, const double* theta_unc,",
            "    const double* nuisance_values, double* nuisance_gradient, int n_nuisance, double* val,",
            "    double* grad, char** error_msg) {",
            "  stan::math::internal::nicostan_nuisance_block_t block;",
            "  block.values = nuisance_values;",
            "  block.gradient = nuisance_gradient;",
            "  block.size = n_nuisance;",
            "  for (int i = 0; i < n_nuisance; ++i) nuisance_gradient[i] = 0.0;",
            "  stan::math::internal::nicostan_nuisance_block_t* previous_block =",
            "      stan::math::internal::nicostan_nuisance_block;",
            "  stan::math::internal::nicostan_nuisance_block = &block;",
            paste0("  const int result = bs_log_density_gradient(m, propto, jacobian, theta_unc, val, grad, ",
                   "error_msg);"),
            "  stan::math::internal::nicostan_nuisance_block = previous_block;",
            "  for (const auto& unit_and_adjoint : block.adjoints_from_other_chunks) {",
            "    nuisance_gradient[unit_and_adjoint.first] += unit_and_adjoint.second;",
            "  }",
            "  return result;",
            "}",
            "",
            "//// (the same for R's .C, as BridgeStan's bs_log_density_gradient_R)",
            "extern \"C\" BS_PUBLIC void nicostan_bs_log_density_gradient_with_the_nuisance_block_R(",
            "    bs_model** model, int* propto, int* jacobian, const double* theta_unc,",
            "    const double* nuisance_values, double* nuisance_gradient, int* n_nuisance, double* val,",
            "    double* grad, int* return_code, char** err_msg, void** err_ptr) {",
            "  *return_code = nicostan_bs_log_density_gradient_with_the_nuisance_block(",
            "      *model, (*propto != 0), (*jacobian != 0), theta_unc, nuisance_values, nuisance_gradient,",
            "      *n_nuisance, val, grad, err_msg);",
            "  *err_ptr = static_cast<void*>(*err_msg);",
            "}",
            "",
            "//// the nuisance values used when the model is evaluated outside NicoStan's gradient call (R's .C)",
            "extern \"C\" BS_PUBLIC void nicostan_set_the_nuisance_values_used_outside_NicoStan_R(",
            "    const double* nuisance_values, int* n_nuisance) {",
            "  stan::math::internal::nicostan_nuisance_values_used_outside_NicoStan.assign(",
            "      nuisance_values, nuisance_values + *n_nuisance);",
            "}")
      writeLines(text = lines, con = bridgestan_file, useBytes = TRUE)
      invisible(c(model_rng_file, bridgestan_file))
}

## ---- NicoStan's fifth runtime patch: the chunk-local leapfrog step (on top of the nuisance block owned by
##      NicoStan, which it needs): in the gradient call of a leapfrog step, each chunk of reduce_sum zeroes the
##      gradient of its own nuisance values, adds their adjoints and has NicoStan make their step (the closing
##      half kick, then the next step's half kick and rotation) whilst they are in the cache; the model library
##      exports nicostan_bs_log_density_gradient_with_the_nuisance_block_and_the_chunk_local_leapfrog_step. Every
##      other gradient call is made as before. Its description is reduce_sum_chunk_local_leapfrog_step.inc; the
##      description of the step shared with NicoStan's sampler is chunk_local_leapfrog_step_of_the_nuisance_
##      block.hpp. Applied when options(NicoStan.reduce_sum_chunk_local_leapfrog_step = TRUE) (with
##      options(NicoStan.reduce_sum_nuisance_block = TRUE) and options(NicoStan.reduce_sum_lean_slicing = TRUE))
R_fn_bridge_memory_reduce_sum_chunk_local_leapfrog_step_patch_lines <-  function(step_header = NULL,
                                                                               patch_inc = NULL) {
      if (is.null(x = step_header)) {
            # step_header <-  system.file("include", "NicoStan", "runtime", "general_functions",
            #                            "chunk_local_leapfrog_step_of_the_nuisance_block.hpp",
            #                            package = "NicoStan", mustWork = TRUE)
            ## (the file of the one NicoStan install, never a side library: fn_file_of_the_one_NicoStan_install())
            step_header <-  fn_file_of_the_one_NicoStan_install(
                  "include", "NicoStan", "runtime", "general_functions",
                  "chunk_local_leapfrog_step_of_the_nuisance_block.hpp",
                  package = "NicoStan", mustWork = TRUE)
      }
      if (is.null(x = patch_inc)) {
            # patch_inc <-  system.file("include", "NicoStan", "runtime", "general_functions",
            #                          "reduce_sum_chunk_local_leapfrog_step.inc",
            #                          package = "NicoStan", mustWork = TRUE)
            ## (the file of the one NicoStan install, never a side library: fn_file_of_the_one_NicoStan_install())
            patch_inc <-  fn_file_of_the_one_NicoStan_install(
                  "include", "NicoStan", "runtime", "general_functions",
                  "reduce_sum_chunk_local_leapfrog_step.inc",
                  package = "NicoStan", mustWork = TRUE)
      }
      ## (the description of the step shared with the sampler, without its part for NicoStan's own side, then the
      ##  patch itself)
      c("#define NICOSTAN_CHUNK_LOCAL_LEAPFROG_STEP_INSIDE_THE_MODEL_LIBRARY",
        readLines(con = normalizePath(path = step_header, mustWork = TRUE), warn = FALSE),
        readLines(con = normalizePath(path = patch_inc, mustWork = TRUE), warn = FALSE))
}

R_fn_bridge_memory_patch_reduce_sum_chunk_local_leapfrog_step <-  function(reduce_sum_file,
                                                                         patch_lines,
                                                                         nuisance_block_patch_lines) {
      lines <-  readLines(con = reduce_sum_file, warn = FALSE)
      if (any(grepl(pattern = "nicostan_chunk_local_leapfrog_step_of_the_caller_", x = lines, fixed = TRUE))) {
            stop("BridgeStan reduce_sum.hpp already carries NicoStan's chunk-local leapfrog step: ",
                 reduce_sum_file)
      }
      ## (the lines the nuisance block patch wrote, each now a line of its own)
      first_line_of_nuisance_block_patch <-  which(lines == nuisance_block_patch_lines[1])
      member_anchor <-  which(lines == paste0("    nicostan_nuisance_block_t* nicostan_nuisance_block_of_the_",
                                              "caller_{nicostan_nuisance_block};"))
      split_anchor <-  which(lines == paste0("      nicostan_nuisance_block_of_the_caller_ = ",
                                             "other.nicostan_nuisance_block_of_the_caller_;"))
      scope_anchor <-  which(lines == "      nicostan_scope_of_a_chunk_of_the_nuisance_block nicostan_scope(")
      adjoints_anchor <-  which(lines == "      nicostan_add_the_adjoints_of_the_chunk_of_the_nuisance_block();")
      caller_anchor <-  which(lines == "    worker.nicostan_sliced_varis_ = varis;")
      if (length(x = first_line_of_nuisance_block_patch) != 1 || length(x = member_anchor) != 1 ||
          length(x = split_anchor) != 1 || length(x = scope_anchor) != 1 || length(x = adjoints_anchor) != 1 ||
          length(x = caller_anchor) != 1 ||
          !identical(x = lines[first_line_of_nuisance_block_patch +
                               seq_along(along.with = nuisance_block_patch_lines) - 1],
                     y = nuisance_block_patch_lines) ||
          !grepl("static_cast<int>\\(r.end\\(\\)\\) - 1\\);$", lines[scope_anchor + 2])) {
            stop("Unexpected BridgeStan reduce_sum.hpp chunk-local leapfrog step anchors (the nuisance block ",
                 "patch must be applied first): ", reduce_sum_file)
      }
      ## (every replacement is in place, so that every anchor still points at its line; then the insertion)
      ##
      ## ---- the caller's chunk-local step and the number of its reduce_sum call, taken by the reducer on the
      ##      caller's thread and handed to every split
      lines[member_anchor] <-  paste0(lines[member_anchor], "\n",
            "    nicostan_chunk_local_leapfrog_step_in_the_model_t* nicostan_chunk_local_leapfrog_step_of_the_",
            "caller_{\n        nicostan_chunk_local_leapfrog_step};\n",
            "    int nicostan_reduce_sum_call_of_the_chunk_local_leapfrog_step_{0};")
      lines[split_anchor] <-  paste0(lines[split_anchor], "\n",
            "      nicostan_chunk_local_leapfrog_step_of_the_caller_ =\n",
            "          other.nicostan_chunk_local_leapfrog_step_of_the_caller_;\n",
            "      nicostan_reduce_sum_call_of_the_chunk_local_leapfrog_step_ =\n",
            "          other.nicostan_reduce_sum_call_of_the_chunk_local_leapfrog_step_;")
      ##
      ## ---- each chunk: the caller's chunk-local step on the chunk's thread for its duration; after its own
      ##      gradient, its adjoints and the step of its own values
      lines[scope_anchor + 2] <-  paste0(lines[scope_anchor + 2], "\n",
            "      nicostan_scope_of_a_chunk_of_the_chunk_local_leapfrog_step nicostan_step_scope(\n",
            "          nicostan_chunk_local_leapfrog_step_of_the_caller_);")
      lines[adjoints_anchor] <-  paste0(
            "      nicostan_add_the_adjoints_of_the_chunk_of_the_nuisance_block_and_make_its_leapfrog_step(\n",
            "          nicostan_chunk_local_leapfrog_step_of_the_caller_, ",
            "nicostan_reduce_sum_call_of_the_chunk_local_leapfrog_step_);")
      ##
      ## ---- the number of this reduce_sum call, on the caller's thread
      lines[caller_anchor] <-  paste0(lines[caller_anchor], "\n",
            "    worker.nicostan_reduce_sum_call_of_the_chunk_local_leapfrog_step_ =\n",
            "        nicostan_number_of_this_reduce_sum_call_of_the_chunk_local_leapfrog_step(\n",
            "            worker.nicostan_chunk_local_leapfrog_step_of_the_caller_);")
      ##
      ## ---- the patch's own text, right after that of the nuisance block patch (which it uses)
      lines <-  append(x = lines, values = patch_lines,
                       after = first_line_of_nuisance_block_patch + length(x = nuisance_block_patch_lines) - 1)
      writeLines(text = lines, con = reduce_sum_file, useBytes = TRUE)
      invisible(reduce_sum_file)
}

## ---- part of the chunk-local leapfrog step patch: BridgeStan's gradient zeroes the gradient of the values no
##      chunk owns, and keeps the values read on the main tape, before their adjoints are added; the model library
##      exports the gradient call of a leapfrog step,
##      nicostan_bs_log_density_gradient_with_the_nuisance_block_and_the_chunk_local_leapfrog_step (no pass
##      zeroing the whole nuisance gradient; sets the nuisance block and the step for the duration of the
##      gradient, then adds the adjoints that chunks kept for other units and hands NicoStan the runs of values
##      stepped inside the chunks and the values read outside their own chunk)
R_fn_bridge_memory_patch_bridgestan_gradient_with_the_chunk_local_leapfrog_step <-  function(model_rng_file,
                                                                                           bridgestan_file) {
      lines <-  readLines(con = model_rng_file, warn = FALSE)
      main_tape_anchor <-  which(lines ==
                                 "    for (const auto& unit_and_node : nicostan_block->nodes_on_the_main_tape) {")
      if (length(x = main_tape_anchor) != 1 ||
          any(grepl(pattern = "nicostan_before_the_adjoints_of_the_values_read_on_the_main_tape", x = lines,
                    fixed = TRUE))) {
            stop("Unexpected BridgeStan model_rng.cpp chunk-local leapfrog step anchor (the nuisance block ",
                 "patch must be applied first, once): ", model_rng_file)
      }
      lines[main_tape_anchor] <-  paste0(
            "    //// (NicoStan's chunk-local leapfrog step: the values no chunk owns zeroed first)\n",
            "    stan::math::internal::nicostan_before_the_adjoints_of_the_values_read_on_the_main_tape(\n",
            "        nicostan_block);\n",
            lines[main_tape_anchor])
      writeLines(text = lines, con = model_rng_file, useBytes = TRUE)
      ##
      lines <-  readLines(con = bridgestan_file, warn = FALSE)
      if (any(grepl(pattern = "nicostan_bs_log_density_gradient_with_the_nuisance_block_and_the_chunk_local_",
                    x = lines, fixed = TRUE))) {
            stop("BridgeStan bridgestan.cpp already exports NicoStan's chunk-local leapfrog step: ",
                 bridgestan_file)
      }
      lines <-  c(lines, "",
            "//// NicoStan's chunk-local leapfrog step: the gradient call of a leapfrog step of a model in",
            "//// NicoStan's nuisance-block form, which also makes the step of the nuisance values inside the",
            "//// chunks",
            paste0("extern \"C\" BS_PUBLIC int nicostan_bs_log_density_gradient_with_the_nuisance_block_and_the_",
                   "chunk_local_leapfrog_step("),
            "    const bs_model* m, bool propto, bool jacobian, const double* theta_unc,",
            "    const double* nuisance_values, double* nuisance_gradient, int n_nuisance, double* val,",
            "    double* grad, nicostan_chunk_local_leapfrog_step_t* step, char** error_msg) {",
            "  stan::math::internal::nicostan_nuisance_block_t block;",
            "  block.values = nuisance_values;",
            "  block.gradient = nuisance_gradient;",
            "  block.size = n_nuisance;",
            "  //// (no pass zeroing the whole nuisance gradient: each chunk zeroes its own values, and the",
            "  ////  values no chunk owns are zeroed after the chunks)",
            "  static thread_local stan::math::internal::nicostan_chunk_local_leapfrog_step_in_the_model_t",
            "      chunk_local_step;",
            "  chunk_local_step.step = step;",
            "  chunk_local_step.number_of_reduce_sum_calls.store(0);",
            "  chunk_local_step.reduce_sum_call_whose_chunks_own_the_values.store(0);",
            "  chunk_local_step.runs_of_values_owned_by_the_chunks.clear();",
            "  chunk_local_step.runs_of_values_stepped_inside_the_chunks.clear();",
            "  chunk_local_step.values_read_outside_their_own_chunk.clear();",
            "  chunk_local_step.first_and_end_of_the_runs_of_values_stepped_inside_the_chunks.clear();",
            "  chunk_local_step.gradient_of_the_values_no_chunk_owns_zeroed = false;",
            "  step->step_made_inside_the_chunks = 0;",
            "  step->first_and_end_of_the_runs_of_values_stepped_inside_the_chunks = nullptr;",
            "  step->n_runs_of_values_stepped_inside_the_chunks = 0;",
            "  step->values_read_outside_their_own_chunk = nullptr;",
            "  step->n_values_read_outside_their_own_chunk = 0;",
            "  stan::math::internal::nicostan_nuisance_block_t* previous_block =",
            "      stan::math::internal::nicostan_nuisance_block;",
            "  stan::math::internal::nicostan_chunk_local_leapfrog_step_in_the_model_t* previous_step =",
            "      stan::math::internal::nicostan_chunk_local_leapfrog_step;",
            "  stan::math::internal::nicostan_nuisance_block = &block;",
            "  stan::math::internal::nicostan_chunk_local_leapfrog_step = &chunk_local_step;",
            paste0("  const int result = bs_log_density_gradient(m, propto, jacobian, theta_unc, val, grad, ",
                   "error_msg);"),
            "  stan::math::internal::nicostan_nuisance_block = previous_block;",
            "  stan::math::internal::nicostan_chunk_local_leapfrog_step = previous_step;",
            "  //// (zeroed here only when BridgeStan's gradient stopped before it zeroed them)",
            "  stan::math::internal::nicostan_zero_the_gradient_of_the_values_no_chunk_owns(&chunk_local_step,",
            "                                                                               &block);",
            "  for (const auto& unit_and_adjoint : block.adjoints_from_other_chunks) {",
            "    nuisance_gradient[unit_and_adjoint.first] += unit_and_adjoint.second;",
            "    chunk_local_step.values_read_outside_their_own_chunk.push_back(unit_and_adjoint.first);",
            "  }",
            "  std::vector<int>& values_read_outside = chunk_local_step.values_read_outside_their_own_chunk;",
            "  std::sort(values_read_outside.begin(), values_read_outside.end());",
            "  values_read_outside.erase(std::unique(values_read_outside.begin(), values_read_outside.end()),",
            "                            values_read_outside.end());",
            "  std::vector<std::pair<int, int>>& runs =",
            "      chunk_local_step.runs_of_values_stepped_inside_the_chunks;",
            "  std::sort(runs.begin(), runs.end());",
            "  for (const auto& run : runs) {",
            "    std::vector<int>& first_and_end =",
            "        chunk_local_step.first_and_end_of_the_runs_of_values_stepped_inside_the_chunks;",
            "    first_and_end.push_back(run.first);",
            "    first_and_end.push_back(run.second);",
            "  }",
            "  step->first_and_end_of_the_runs_of_values_stepped_inside_the_chunks =",
            "      chunk_local_step.first_and_end_of_the_runs_of_values_stepped_inside_the_chunks.data();",
            "  step->n_runs_of_values_stepped_inside_the_chunks = static_cast<int>(runs.size());",
            "  step->values_read_outside_their_own_chunk = values_read_outside.data();",
            "  step->n_values_read_outside_their_own_chunk = static_cast<int>(values_read_outside.size());",
            "  step->step_made_inside_the_chunks = (result == 0) ? 1 : 0;",
            "  return result;",
            "}")
      writeLines(text = lines, con = bridgestan_file, useBytes = TRUE)
      invisible(c(model_rng_file, bridgestan_file))
}

R_fn_bridge_memory_patch_generated_map_calls <-  function(generated_file, automatic_header, Stan_source_code) {
      lines <-  readLines(con = generated_file, warn = FALSE)
      include <-  which(lines == "#include <stan/model/model_header.hpp>")
      if (length(include) != 1) stop("Unrecognised generated model header; no map change made.")
      code <-  paste(lines, collapse = "\n")
      visible <-  code
      protected <-  "\"(?:[^\"\\\\]|\\\\.)*\"|/\\*[\\s\\S]*?\\*/|//[^\\n]*"
      type_carriers <-  attr(Stan_source_code, "nicostan_map_type_carriers", exact = TRUE)
      Stan_visible <-  gsub(pattern = protected, replacement = " ",
                            x = paste(Stan_source_code, collapse = "\n"), perl = TRUE)
      Stan_compact <-  gsub(pattern = "[[:space:]]+", replacement = "", x = Stan_visible)
      required_block_construction <-  c(
            "intN_blocks=(N+chunk_size-1)%/%chunk_size;",
            "block_first=to_int(to_array_1d(linspaced_vector(N_blocks,0,N_blocks-1)*chunk_size+1));",
            "block_last=to_int(to_array_1d(linspaced_vector(N_blocks,1,N_blocks)*chunk_size));",
            "block_last[N_blocks]=N;",
            "K_and_chunk_size_of_block=rep_array({K,chunk_size},N_blocks-1);",
            "array[max(N_blocks-1,0),chunk_size*K]realX_of_block_by_column;",
            "array[N_blocks-1]vector[chunk_size]log_frailty_raw_of_blocks_before_the_last")
      ## Syntax recognition is bounded to this block layout; it is not a general proof against reassignment.
      ## Observation/job correspondence also requires review of the actual unchanged model at the call site.
      observation_blocks_match_the_inspected_source <-  all(vapply(
            required_block_construction, function(statement) {
            grepl(pattern = statement, x = Stan_compact, fixed = TRUE)
      }, logical(1)))
      mapped_jobs_read_the_held_nuisance_block <-  observation_blocks_match_the_inspected_source &&
            grepl(pattern = "nicostan_offset_of_log_frailty_raw", x = Stan_compact, fixed = TRUE)
      if (mapped_jobs_read_the_held_nuisance_block) {
            job_name <-  "log_frailty_raw_of_blocks_before_the_last"
            ## (the readers' type argument: beta[1] as the writer wrote it, or the marker that the type
            ##  transform of a map_rect model puts in its place (0.0, made DUMMY_VAR__ later in this function);
            ##  neither is used by the loaders below, which take their type from the map runtime and from beta(0))
            type_argument_of_the_reader <-  paste0("(?:stan::model::rvalue\\(\\s*beta,\\s*\"beta\",\\s*",
                                                   "stan::model::index_uni\\(1\\)\\)|0\\.0|DUMMY_VAR__)")
            vector_type <-  paste0("std::vector\\s*<\\s*Eigen::Matrix\\s*<\\s*local_scalar_t__",
                                    "\\s*,\\s*-1\\s*,\\s*1\\s*>\\s*>")
            eager_jobs <-  paste0(vector_type, "\\s+", job_name, "\\s*=\\s*", vector_type,
                  "\\([^;]*;\\s*current_statement__\\s*=\\s*[0-9]+;\\s*",
                  "stan::model::assign\\(\\s*", job_name, ",\\s*columns_as_array_of_vectors\\(",
                  "\\s*stan::math::to_matrix\\(\\s*nicostan_nuisance_vector_at\\(",
                  "\\s*group_of_observations_of_blocks_before_the_last,",
                  "\\s*nicostan_offset_of_log_frailty_raw,\\s*", type_argument_of_the_reader,
                  ",\\s*pstream__\\),",
                  "\\s*chunk_size,\\s*\\(N_blocks\\s*-\\s*1\\)\\),\\s*1,",
                  "\\s*\\(N_blocks\\s*-\\s*1\\),\\s*pstream__\\),",
                  "\\s*\"assigning variable ", job_name, "\"\\);")
            matches <-  gregexpr(pattern = eager_jobs, text = code, perl = TRUE)[[1]]
            if (matches[1] < 0) {
                  stop("Recognised map layout has no verified nuisance-job construction; no rewrite made.")
            }
            loader_name <-  "nicostan_load_nuisance_values_for_this_map_job"
            replacement <-  paste(c(
                  paste0("const auto ", loader_name, " = [&](const std::size_t job, const auto& type) {"),
                  "  using T_value = stan::return_type_t<std::decay_t<decltype(type)>>;",
                  "  namespace readers = nicostan_readers_of_the_nuisance_block;",
                  "  const auto nuisance_values = readers::fn_values_of_the_nuisance_block(",
                  "      \"nicostan_nuisance_vector_at\", 0, -1);",
                  "  Eigen::Matrix<T_value, -1, 1> values(chunk_size);",
                  "  const std::size_t first = job * static_cast<std::size_t>(chunk_size);",
                  "  for (int i = 0; i < chunk_size; ++i) {",
                  "    const int term = group_of_observations_of_blocks_before_the_last[first + i];",
                  "    const int value = nicostan_offset_of_log_frailty_raw + term - 1;",
                  "    if (value < 0 || value >= nuisance_values.size) {",
                  "      throw std::domain_error(\"nicostan_nuisance_vector_at: value \" +",
                  "          std::to_string(value + 1) + \" is outside the nuisance values (\" +",
                  "          std::to_string(nuisance_values.size) + \")\");",
                  "    }",
                  "    if constexpr (std::is_same<T_value, stan::math::var>::value) {",
                  "      values.coeffRef(i) = readers::fn_node_of_value(nuisance_values, value, term - 1);",
                  "    } else {",
                  "      values.coeffRef(i) = nuisance_values.values[value];",
                  "    }",
                  "  }",
                  "  return values;",
                  "};",
                  paste0("const stan::math::internal::nicostan_map_nuisance_job_parameters<decltype(",
                           loader_name, ")>"),
                  paste0("  ", job_name, "(N_blocks - 1, chunk_size, ", loader_name, ");")),
                  collapse = "\n")
            code <-  gsub(pattern = eager_jobs, replacement = replacement, x = code, perl = TRUE)
            packed_rows <-  paste0(
                  "std::vector<std::vector<local_scalar_t__>>\\s+packed_of_block\\s*=",
                  "\\s*std::vector<std::vector<local_scalar_t__>>\\(N_blocks,",
                  "\\s*std::vector<local_scalar_t__>\\(N_packed,\\s*DUMMY_VAR__\\)\\);",
                  "\\s*current_statement__\\s*=\\s*[0-9]+;\\s*stan::model::assign\\(",
                  "packed_of_block,\\s*stan::math::to_array_2d\\(\\s*stan::math::append_col\\(",
                  "\\s*stan::math::to_matrix\\(\\s*nicostan_nuisance_vector_at\\(",
                  "\\s*group_of_packed_value_by_column,\\s*nicostan_offset_of_log_frailty_raw,",
                  "\\s*", type_argument_of_the_reader, ",",
                  "\\s*pstream__\\),\\s*N_blocks,\\s*\\(N_packed\\s*-\\s*1\\)\\),",
                  "\\s*log_risk_before_block\\)\\),\\s*\"assigning variable packed_of_block\"\\);")
            if (!grepl(pattern = packed_rows, x = code, perl = TRUE)) {
                  stop("Recognised map layout has no verified downstream nuisance rows; no rewrite made.")
            }
            code <-  gsub(pattern = packed_rows,
                  replacement = paste0("const auto packed_of_block = ",
                                       "stan::math::to_array_1d(log_risk_before_block);"),
                  x = code, perl = TRUE)
            original_reducer <-  paste0(
                  "reduce_sum_static<partial_sum_packed_blocks_lpdf_rsfunctor__<propto__>>(")
            code <-  gsub(pattern = original_reducer, replacement = paste0(
                  "reduce_sum_static<nicostan_rows_loaded_inside_the_reducer<",
                  "partial_sum_packed_blocks_lpdf_rsfunctor__<propto__>>>("), x = code, fixed = TRUE)
            code <-  gsub(pattern = "packed_of_block, 1, pstream__, X_decreasing_time,",
                  replacement = paste0("packed_of_block, 1, pstream__, ",
                        "group_of_packed_value_by_column, N_blocks, N_packed, ",
                        "nicostan_offset_of_log_frailty_raw, beta(0), ",
                        "X_decreasing_time,"), x = code, fixed = TRUE)
            row_loader <-  paste(c(
                  "template <typename F> struct nicostan_rows_loaded_inside_the_reducer {",
                  "  template <typename T_last, typename T_type, typename... T_args>",
                  "  auto operator()(const std::vector<T_last>& offsets, int first, int last,",
                  "                  std::ostream* messages,",
                  "                  const std::vector<int>& indices, int rows, int columns, int offset,",
                  "                  const T_type& type,",
                  "                  const T_args&... arguments) const {",
                  "    using T_value = stan::return_type_t<T_last, T_type, T_args...>;",
                  "    const int number_of_rows = last - first + 1;",
                  "    namespace readers = nicostan_readers_of_the_nuisance_block;",
                  "    const auto nuisance_values = readers::fn_values_of_the_nuisance_block(",
                  "        \"nicostan_nuisance_vector_at\", 0, -1);",
                  "    std::vector<std::vector<T_value>> packed(number_of_rows);",
                  "    for (int row = 0; row < number_of_rows; ++row) {",
                  "      packed[row].resize(columns);",
                  "      for (int column = 0; column < columns - 1; ++column) {",
                  "        const int term = indices[first + row + static_cast<std::size_t>(column) * rows];",
                  "        const int value = offset + term - 1;",
                  "        if (value < 0 || value >= nuisance_values.size) {",
                  "          throw std::domain_error(\"nicostan_nuisance_vector_at: value \" +",
                  "              std::to_string(value + 1) + \" is outside the nuisance values (\" +",
                  "              std::to_string(nuisance_values.size) + \")\");",
                  "        }",
                  "        if constexpr (std::is_same<stan::return_type_t<T_type>, stan::math::var>::value) {",
                  "          packed[row][column] = readers::fn_node_of_value(nuisance_values, value, term - 1);",
                  "        } else {",
                  "          packed[row][column] = nuisance_values.values[value];",
                  "        }",
                  "      }",
                  "      packed[row][columns - 1] = offsets[row];",
                  "    }",
                  "    return F()(packed, first, last, messages, arguments...);",
                  "  }",
                  "};",
                  ""), collapse = "\n")
            class_start <-  regexpr(pattern = "class [[:alnum:]_]+ final :", text = code, perl = TRUE)[1]
            if (class_start < 1) stop("Cannot locate generated model class; no rewrite made.")
            code <-  paste0(substr(code, 1, class_start - 1), row_loader, substring(code, class_start))
            visible <-  code
      }
      mapped_block_call <-  paste0(
            "^map_rect\\s*<\\s*[0-9]+\\s*,\\s*",
            "log_sum_exp_eta_of_block_from_its_raw_log_frailties_functor__\\s*>\\s*\\(\\s*",
            "beta_and_sigma_frailty\\s*,\\s*log_frailty_raw_of_blocks_before_the_last\\s*,\\s*",
            "X_of_block_by_column\\s*,\\s*K_and_chunk_size_of_block\\s*,\\s*pstream__\\s*\\)")
      observation_calls <-  0
      protected_matches <-  gregexpr(pattern = protected, text = code, perl = TRUE)[[1]]
      if (protected_matches[1] > 0) {
            lengths <-  attr(protected_matches, "match.length")
            for (index in seq_along(protected_matches)) {
                  first <-  protected_matches[index]
                  last <-  first + lengths[index] - 1
                  substr(visible, first, last) <-  strrep(" ", lengths[index])
            }
      }
      model_class <-  regexpr(pattern = paste0("class\\s+[[:alnum:]_]+\\s+final\\s*:\\s*public\\s+",
                                               "(?:stan::model::)?model_base_crtp\\s*<"),
                              text = visible, perl = TRUE)[1]
      if (!is.null(type_carriers) && model_class > 0) {
            fn_closing_bracket <-  function(text,
                                             position) {
                  opening <-  substr(text, position, position)
                  matching <-  c("(" = ")", "[" = "]", "{" = "}")
                  characters <-  strsplit(substring(text, position), "", fixed = TRUE)[[1]]
                  stack <-  character(0)
                  for (index in seq_along(characters)) {
                        character <-  characters[index]
                        if (character %in% names(matching)) stack <-  c(stack, matching[[character]])
                        else if (character %in% matching) {
                              if (length(stack) == 0 || stack[length(stack)] != character) {
                                    stop("Unbalanced generated C++ delimiters; no type rewrite made.")
                              }
                              stack <-  stack[-length(stack)]
                              if (length(stack) == 0) return(position + index - 1)
                        }
                  }
                  stop("No matching generated C++ delimiter; no type rewrite made.")
            }
            fn_argument_ranges <-  function(text,
                                             opening,
                                             closing) {
                  if (closing == opening + 1) return(list())
                  matching <-  c("(" = ")", "[" = "]", "{" = "}")
                  stack <-  character(0)
                  commas <-  integer(0)
                  index <-  opening + 1
                  while (index < closing) {
                        character <-  substr(text, index, index)
                        if (character == "<") {
                              previous <-  index - 1
                              while (previous > opening && grepl("\\s", substr(text, previous, previous))) {
                                    previous <-  previous - 1
                              }
                              last <-  previous
                              while (previous > opening && grepl("[A-Za-z0-9_]",
                                                                  substr(text, previous, previous))) {
                                    previous <-  previous - 1
                              }
                              candidate <-  substr(text, previous + 1, last)
                              if (nzchar(candidate) &&
                                  !grepl("\\s", substr(text, index + 1, index + 1))) {
                                    template_close <-  fn_template_closing_angle(text, index, closing - 1)
                                    if (!is.na(template_close) &&
                                        !grepl("\\s", substr(text, template_close - 1,
                                                              template_close - 1))) {
                                          after_template <-  template_close + 1
                                          while (after_template < closing &&
                                                 grepl("\\s", substr(text, after_template,
                                                                      after_template))) {
                                                after_template <-  after_template + 1
                                          }
                                          if (substr(text, after_template, after_template) %in%
                                              c("(", ":", "{", ",", ")", "]")) {
                                                index <-  template_close + 1
                                                next
                                          }
                                    }
                              }
                        }
                        if (character %in% names(matching)) stack <-  c(stack, matching[[character]])
                        else if (character %in% matching) {
                              if (length(stack) == 0 || stack[length(stack)] != character) {
                                    stop("Unbalanced generated C++ arguments; no type rewrite made.")
                              }
                              stack <-  stack[-length(stack)]
                        } else if (character == "," && length(stack) == 0) {
                              commas <-  c(commas, index)
                        }
                        index <-  index + 1
                  }
                  starts <-  c(opening + 1, commas + 1)
                  ends <-  c(commas - 1, closing - 1)
                  lapply(seq_along(starts), function(index) c(starts[index], ends[index]))
            }
            fn_template_closing_angle <-  function(text,
                                                   opening,
                                                   limit) {
                  if (opening >= limit) return(NA_integer_)
                  characters <-  strsplit(substr(text, opening + 1, limit), "", fixed = TRUE)[[1]]
                  matching <-  c("(" = ")", "[" = "]", "{" = "}")
                  stack <-  character(0)
                  depth <-  1L
                  for (index in seq_along(characters)) {
                        character <-  characters[index]
                        if (character %in% names(matching)) stack <-  c(stack, matching[[character]])
                        else if (character %in% matching) {
                              if (length(stack) == 0 || stack[length(stack)] != character) {
                                    stop("Unbalanced generated C++ template arguments; no type rewrite made.")
                              }
                              stack <-  stack[-length(stack)]
                        } else if (length(stack) == 0 && character == "<" &&
                                   substr(text, opening + index + 1, opening + index + 1) != "=") {
                              depth <-  depth + 1L
                        } else if (length(stack) == 0 && character == ">" &&
                                   substr(text, opening + index + 1, opening + index + 1) != "=") {
                              depth <-  depth - 1L
                              if (depth == 0L) return(opening + index)
                        }
                  }
                  NA_integer_
            }
            fn_target_call_opening <-  function(text,
                                                 name_end) {
                  position <-  name_end + 1
                  while (position <= nchar(text) && grepl("\\s", substr(text, position, position))) {
                        position <-  position + 1
                  }
                  if (substr(text, position, position) == "<") {
                        template_close <-  fn_template_closing_angle(text, position, nchar(text))
                        if (is.na(template_close)) {
                              stop("Unrecognised generated template call; no type rewrite made.")
                        }
                        position <-  template_close + 1
                        while (position <= nchar(text) && grepl("\\s", substr(text, position, position))) {
                              position <-  position + 1
                        }
                  }
                  if (substr(text, position, position) != "(") return(NA_integer_)
                  position
            }
            fn_definition_ranges <-  function(function_names,
                                               text) {
                  if (length(function_names) == 0) return(list())
                  pattern <-  paste0("\\b(", paste(function_names, collapse = "|"), ")\\s*\\(")
                  matches <-  gregexpr(pattern = pattern, text = text, perl = TRUE)[[1]]
                  if (matches[1] < 0) return(list())
                  lengths <-  attr(matches, "match.length")
                  ranges <-  list()
                  for (index in seq_along(matches)) {
                        first <-  matches[index]
                        opening <-  first + regexpr("\\(", substring(text, first))[1] - 1
                        closing <-  fn_closing_bracket(text, opening)
                        after <-  closing + 1
                        while (after <= nchar(text) && grepl("\\s", substr(text, after, after))) {
                              after <-  after + 1
                        }
                        if (substr(text, after, after) == "{") {
                              ranges[[length(ranges) + 1]] <-  c(after + 1,
                                    fn_closing_bracket(text, after) - 1)
                        }
                  }
                  ranges
            }
            fn_token_replacements <-  function(function_names,
                                                scope,
                                                text,
                                                visible_text) {
                  if (length(function_names) == 0 || scope[1] > scope[2]) return(list())
                  pattern <-  paste0("\\b(", paste(function_names, collapse = "|"), ")\\b")
                  matches <-  gregexpr(pattern = pattern,
                        text = substring(visible_text, scope[1], scope[2]), perl = TRUE)[[1]]
                  if (matches[1] < 0) return(list())
                  lengths <-  attr(matches, "match.length")
                  replacements <-  list()
                  for (index in rev(seq_along(matches))) {
                        first <-  scope[1] + matches[index] - 1
                        name_end <-  first + lengths[index] - 1
                        opening <-  fn_target_call_opening(visible_text, name_end)
                        if (is.na(opening)) next
                        closing <-  fn_closing_bracket(visible_text, opening)
                        after_call <-  closing + 1
                        while (after_call <= nchar(visible_text) &&
                               grepl("\\s", substr(visible_text, after_call, after_call))) {
                              after_call <-  after_call + 1
                        }
                        following <-  substr(visible_text, after_call, after_call)
                        arguments <-  fn_argument_ranges(visible_text, opening, closing)
                        last_argument <-  if (length(arguments) == 0) "" else
                              trimws(substr(visible_text, arguments[[length(arguments)]][1],
                                            arguments[[length(arguments)]][2]))
                        if (following == "{" || (following == ";" &&
                            grepl("(?:std::)?ostream\\s*\\*", last_argument))) next
                        ## (stanc ends a call of an _lp function with lp__, lp_accum__, pstream__ and a call of an
                        ##  _rng function with base_rng__, pstream__; every other call with pstream__ only)
                        name <-  substr(visible_text, first, name_end)
                        arguments_after_the_marker <-  if (grepl("_lp$", name)) {
                              c("lp__", "lp_accum__", "pstream__")
                        } else if (grepl("_rng$", name)) {
                              c("base_rng__", "pstream__")
                        } else "pstream__"
                        if (length(arguments) < length(arguments_after_the_marker) + 1) {
                              stop("Generated map type-carrier call has no marker argument.")
                        }
                        last_arguments <-  vapply(utils::tail(arguments, length(arguments_after_the_marker)),
                              function(range) trimws(substr(visible_text, range[1], range[2])), character(1))
                        if (!identical(unname(last_arguments), arguments_after_the_marker)) {
                              stop("Generated map type-carrier call lacks its pstream argument.")
                        }
                        marker <-  arguments[[length(arguments) - length(arguments_after_the_marker)]]
                        source_marker <-  substr(text, marker[1], marker[2])
                        comments <-  gregexpr(pattern = "/\\*[\\s\\S]*?\\*/|//[^\\n]*",
                              text = source_marker, perl = TRUE)
                        saved_comments <-  regmatches(source_marker, comments)[[1]]
                        replacement <-  if (length(saved_comments) == 0) "DUMMY_VAR__" else
                              paste0("DUMMY_VAR__ ", paste(saved_comments, collapse = "\n"), "\n")
                        replacements[[length(replacements) + 1]] <-  list(
                              first = marker[1], last = marker[2], text = replacement)
                  }
                  replacements
            }
            ## (a helper given to reduce_sum is called through its functor, as stan::math::reduce_sum<
            ##  helper_lpdf_rsfunctor__<propto__>>(slice, grainsize, pstream__, ..., marker): the marker is its
            ##  last argument)
            fn_reduce_sum_callback_replacements <-  function(function_names,
                                                              scope,
                                                              text,
                                                              visible_text) {
                  if (length(function_names) == 0 || scope[1] > scope[2]) return(list())
                  pattern <-  paste0("\\breduce_sum(?:_static)?\\s*<\\s*(?:",
                                     paste(function_names, collapse = "|"), ")_rsfunctor__\\b")
                  matches <-  gregexpr(pattern = pattern,
                        text = substring(visible_text, scope[1], scope[2]), perl = TRUE)[[1]]
                  if (matches[1] < 0) return(list())
                  replacements <-  list()
                  for (index in rev(seq_along(matches))) {
                        first <-  scope[1] + matches[index] - 1
                        template_open <-  first + regexpr("<", substring(visible_text, first))[1] - 1
                        opening <-  fn_target_call_opening(visible_text, template_open - 1)
                        if (is.na(opening)) stop("Unrecognised generated reduce_sum call; no type rewrite made.")
                        closing <-  fn_closing_bracket(visible_text, opening)
                        arguments <-  fn_argument_ranges(visible_text, opening, closing)
                        if (length(arguments) < 4 ||
                            !identical(trimws(substr(visible_text, arguments[[3]][1], arguments[[3]][2])),
                                       "pstream__")) {
                              stop("Generated reduce_sum type-carrier call has no marker argument.")
                        }
                        marker <-  arguments[[length(arguments)]]
                        source_marker <-  substr(text, marker[1], marker[2])
                        comments <-  gregexpr(pattern = "/\\*[\\s\\S]*?\\*/|//[^\\n]*",
                              text = source_marker, perl = TRUE)
                        saved_comments <-  regmatches(source_marker, comments)[[1]]
                        replacement <-  if (length(saved_comments) == 0) "DUMMY_VAR__" else
                              paste0("DUMMY_VAR__ ", paste(saved_comments, collapse = "\n"), "\n")
                        replacements[[length(replacements) + 1]] <-  list(
                              first = marker[1], last = marker[2], text = replacement)
                  }
                  replacements
            }
            callback_ranges <-  fn_definition_ranges(type_carriers$map_callbacks, visible)
            if (length(type_carriers$map_callbacks) > 0 && length(callback_ranges) == 0) {
                  stop("Generated map callback type scopes are missing; no scalar-token rewrite made.")
            }
            replacement_scopes <-  lapply(callback_ranges, function(range) range)
            replacement_scopes[[length(replacement_scopes) + 1]] <-  c(model_class, nchar(visible))
            target_functions <-  unique(c(type_carriers$helper_functions,
                  sub(pattern = "_lp([dm])f$", replacement = "_lup\\1f",
                      x = type_carriers$helper_functions),
                                          type_carriers$nuisance_readers))
            replacements <-  unlist(lapply(replacement_scopes, function(scope) {
                  c(fn_token_replacements(target_functions, scope, code, visible),
                    fn_reduce_sum_callback_replacements(type_carriers$helper_functions, scope, code, visible))
            }), recursive = FALSE)
            if (length(replacements) > 0) {
                  order_of_replacements <-  order(vapply(replacements, function(entry) entry$first,
                        numeric(1)), decreasing = TRUE)
                  for (index in order_of_replacements) {
                        replacement <-  replacements[[index]]
                        code <-  paste0(substr(code, 1, replacement$first - 1), replacement$text,
                              substring(code, replacement$last + 1))
                  }
                  visible <-  code
                  protected_matches <-  gregexpr(pattern = protected, text = code, perl = TRUE)[[1]]
                  if (protected_matches[1] > 0) {
                        protected_lengths <-  attr(protected_matches, "match.length")
                        for (index in seq_along(protected_matches)) {
                              first <-  protected_matches[index]
                              last <-  first + protected_lengths[index] - 1
                              substr(visible, first, last) <-  strrep(" ", protected_lengths[index])
                        }
                  }
            }
      }
      calls <-  gregexpr(pattern = "\\bmap_rect\\s*<", text = visible, perl = TRUE)[[1]]
      ## (a model without map_rect still keeps the DUMMY_VAR__ type of its readers and helpers made above)
      if (calls[1] < 0) {
            if (!identical(code, paste(lines, collapse = "\n"))) {
                  writeLines(text = code, con = generated_file, useBytes = TRUE)
            }
            return(invisible(generated_file))
      }
      model_class <-  regexpr(pattern = paste0("class\\s+[[:alnum:]_]+\\s+final\\s*:\\s*public\\s+",
                                               "(?:stan::model::)?model_base_crtp\\s*<"),
                              text = visible, perl = TRUE)[1]
      lengths <-  attr(calls, "match.length")
      for (index in rev(seq_along(calls))) {
            first <-  calls[index]
            last <-  first + lengths[index] - 1
            if (observation_blocks_match_the_inspected_source && model_class > 0 && first > model_class &&
                grepl(pattern = mapped_block_call, x = substring(visible, first), perl = TRUE)) {
                  opening <-  regexpr(pattern = ">\\s*\\(", text = substring(visible, first),
                                       perl = TRUE)[1]
                  prefix <-  substring(code, first, first + opening - 2)
                  prefix <-  sub(pattern = "^map_rect\\s*<", replacement = paste0(
                        "nicostan_map_rect_chunked_in_observation_chunks_auto<local_scalar_t__, "),
                        x = prefix, perl = TRUE)
                  opening_length <-  attr(regexpr(pattern = ">\\s*\\(", text = substring(visible, first),
                                                  perl = TRUE), "match.length")
                  after_opening <-  first + opening - 1 + opening_length
                  code <-  paste0(substr(code, 1, first - 1), prefix,
                                    ">(chunk_size, block_first, block_last, ", substring(code, after_opening))
                  observation_calls <-  observation_calls + 1
                  next
            }
            code <-  paste0(substr(code, 1, first - 1),
                            "nicostan_map_rect_chunked_auto<local_scalar_t__, ", substring(code, last + 1))
      }
      lines <-  strsplit(code, "\n", fixed = TRUE)[[1]]
      lines <-  append(x = lines,
                        values = paste0("#include ", encodeString(x = automatic_header, quote = '"')),
                        after = include)
      writeLines(text = lines, con = generated_file, useBytes = TRUE)
      message(paste0("map_rect_chunked: ", length(calls),
                      " generated calls use their enclosing scalar type automatically: ", generated_file))
      message(paste0("map_rect_chunked: ", observation_calls,
                      " calls carry explicit observation boundaries and the model's exact chunk_size."))
      invisible(generated_file)
}

R_fn_bridge_memory_generate_Stan_model <-  function(arguments, automatic_header) {
      if (length(arguments) < 3) stop("Stan generation needs the source, output and compiler paths.")
      source_file <-  normalizePath(arguments[1], mustWork = TRUE)
      generated_file <-  arguments[2]
      compiler <-  arguments[3]
      flags <-  if (length(arguments) > 3) arguments[-seq_len(3)] else character(0)
      code <-  R_fn_Stan_parallel_call_names(code = R_fn_read_Stan_source_with_includes(
            model_file = source_file, stanc_args = flags))
      if (!is.null(automatic_header)) code <-  R_fn_Stan_map_type_in_helper_functions(code = code)
      private_directory <-  tempfile(pattern = "nicostan_Stan_source_", tmpdir = dirname(generated_file))
      if (!dir.create(private_directory)) stop("Cannot create the private Stan generation directory.")
      private_source <-  file.path(private_directory, basename(source_file))
      writeLines(code, private_source, useBytes = TRUE)
      status <-  system2(command = compiler, args = c(vapply(flags, shQuote, character(1)),
            shQuote(paste0("--o=", generated_file)), shQuote(private_source)))
      if (status != 0) stop(paste0("Stan generation failed with exit ", status, "."))
      if (!is.null(automatic_header)) {
            R_fn_bridge_memory_patch_generated_map_calls(generated_file, automatic_header, code)
      }
      invisible(generated_file)
}

R_fn_bridge_memory_patch_map_generation <-  function(runtime_path, automatic_header) {
      make_file <-  file.path(runtime_path, "Makefile")
      lines <-  readLines(con = make_file, warn = FALSE)
      recipe <-  which(startsWith(lines, "\t$(STANC) $(STANCFLAGS) --o="))
      if (length(recipe) != 1) stop("Unrecognised private Stan generation recipe; no map change made.")
      script_file <-  file.path(runtime_path, "src", "nicostan_map_rect_generated_calls.R")
      functions <-  list(R_fn_Stan_parallel_call_names = R_fn_Stan_parallel_call_names,
            R_fn_read_Stan_source_with_includes = R_fn_read_Stan_source_with_includes,
            R_fn_Stan_map_type_in_helper_functions = R_fn_Stan_map_type_in_helper_functions,
            R_fn_bridge_memory_patch_generated_map_calls = R_fn_bridge_memory_patch_generated_map_calls,
            R_fn_bridge_memory_generate_Stan_model = R_fn_bridge_memory_generate_Stan_model)
      script <-  unlist(lapply(names(functions), function(name) {
            lines <-  deparse(functions[[name]])
            lines[1] <-  paste0(name, " <- ", lines[1])
            c(lines, "")
      }), use.names = FALSE)
      header <-  if (is.null(automatic_header)) "NULL" else encodeString(automatic_header, quote = '"')
      script <-  c(script, paste0("R_fn_bridge_memory_generate_Stan_model(",
            "arguments = commandArgs(trailingOnly = TRUE), automatic_header = ", header, ")"))
      writeLines(text = script, con = script_file, useBytes = TRUE)
      Rscript <-  Sys.which("Rscript")
      if (!nzchar(Rscript)) stop("Rscript is required for automatic map scalar types.")
      command <-  paste0("\t", shQuote(Rscript), " --vanilla ",
            '"$(BS_ROOT)/src/nicostan_map_rect_generated_calls.R" ',
            '"$(subst  \\,/,$<)" "$(subst  \\,/,$@)" "$(STANC)" $(STANCFLAGS)')
      lines[recipe] <-  command
      writeLines(text = lines, con = make_file, useBytes = TRUE)
      invisible(make_file)
}

R_fn_bridge_memory_patch_map_rect_chunked <-  function(map_rect_file, runtime_header) {
      lines <-  readLines(con = map_rect_file, warn = FALSE)
      start <-  which(trimws(lines) == "using ReduceF")
      finish <-  which(trimws(lines) == "return combine(world_output, world_f_out);")
      include <-  which(lines == "#include <stan/math/rev/core/chainablestack.hpp>")
      if (length(start) != 1 || length(finish) != 1 || length(include) != 1 || finish <= start) {
            stop("Unrecognised private map_rect_concurrent.hpp; no map change made.")
      }
      original_body <-  lines[seq.int(start, finish)]
      replacement <-  c(
            "  if (nicostan_nuisance_block == nullptr) {",
            paste0("  ", original_body),
            "  }",
            "  throw std::invalid_argument(\"nico_map_rect: verified partition metadata is required; \"",
            "                              \"no fixed map grain is selected.\");")
      lines <-  c(lines[seq_len(start - 1)], replacement, lines[seq.int(finish + 1, length(lines))])
      include_lines <-  paste0("#include ", encodeString(x = runtime_header, quote = '"'))
      lines <-  append(x = lines, values = include_lines, after = include)
      writeLines(text = lines, con = map_rect_file, useBytes = TRUE)
      invisible(map_rect_file)
}

R_fn_bridge_memory_prepare_runtime <-  function(
      unit_upstream_proof,
      release_header = NULL,
      original_bridgestan_path = NULL,
      original_runtime_path = NULL,
      allocator_methods = NULL,
      runtime_cache_dir = tools::R_user_dir("NicoStan", "cache")) {
      if (!identical(x = Sys.info()[["sysname"]], y = "Linux")) {
            stop("The page-release BridgeStan runtime currently supports Linux only.")
      }
      if (!is.null(x = original_bridgestan_path) && !is.null(x = original_runtime_path)) {
            stop("Supply only one of original_bridgestan_path and original_runtime_path.")
      }
      if (is.null(x = original_bridgestan_path)) original_bridgestan_path <-  original_runtime_path
      if (is.null(x = original_bridgestan_path)) {
            if (!requireNamespace(package = "bridgestan", quietly = TRUE)) {
                  stop("The bridgestan package is required to locate BridgeStan.")
            }
            original_bridgestan_path <-  utils::getFromNamespace("get_bridgestan_path", "bridgestan")(
                  download = FALSE)
      }
      original_bridgestan_path <-  normalizePath(path = original_bridgestan_path, mustWork = TRUE)
      if (is.null(x = release_header)) {
            # release_header <-  system.file("include", "NicoStan", "runtime",
            #                               "general_functions", "bridge_memory.hpp",
            #                               package = "NicoStan", mustWork = TRUE)
            ## (the file of the one NicoStan install, never a side library: fn_file_of_the_one_NicoStan_install())
            release_header <-  fn_file_of_the_one_NicoStan_install(
                  "include", "NicoStan", "runtime", "general_functions",
                  "bridge_memory.hpp",
                  package = "NicoStan", mustWork = TRUE)
      }
      release_header <-  normalizePath(path = release_header, mustWork = TRUE)
      stack_alloc_file <-  file.path(original_bridgestan_path, "stan", "lib", "stan_math",
                                    "stan", "math", "memory", "stack_alloc.hpp")
      if (!file.exists(stack_alloc_file)) {
            stop("Configured BridgeStan source has no stack_alloc.hpp: ", stack_alloc_file)
      }
      original_stack_alloc_md5 <-  unname(obj = tools::md5sum(files = stack_alloc_file))
      patch_lines <-  R_fn_bridge_memory_page_release_patch_lines(
            allocator_methods = allocator_methods)
      ## (the reduce_sum patch is applied when options(NicoStan.reduce_sum_parallel_bookkeeping = TRUE); OFF by
      ##  default until its gradient checks (old runtime against new, PK and the aliased-terms model) and the
      apply_reduce_sum_patch <-  isTRUE(x = getOption("NicoStan.reduce_sum_parallel_bookkeeping", FALSE))
      reduce_sum_patch_lines <-  if (apply_reduce_sum_patch) R_fn_bridge_memory_reduce_sum_patch_lines() else
                                 character(0)
      ## (the lean slicing patch, options(NicoStan.reduce_sum_lean_slicing = TRUE): never with the bookkeeping
      ##  patch)
      apply_lean_slicing_patch <-  isTRUE(x = getOption("NicoStan.reduce_sum_lean_slicing", FALSE))
      if (apply_reduce_sum_patch && apply_lean_slicing_patch) {
            stop(paste0("NicoStan.reduce_sum_parallel_bookkeeping and NicoStan.reduce_sum_lean_slicing are not ",
                        "combined."))
      }
      ## (the nuisance block patch, options(NicoStan.reduce_sum_nuisance_block = TRUE): on top of the lean
      ##  slicing)
      apply_nuisance_block_patch <-  isTRUE(x = getOption("NicoStan.reduce_sum_nuisance_block", FALSE))
      if (apply_nuisance_block_patch && !apply_lean_slicing_patch) {
            stop("NicoStan.reduce_sum_nuisance_block needs NicoStan.reduce_sum_lean_slicing = TRUE.")
      }
      apply_map_rect_patch <-  apply_nuisance_block_patch && isTRUE(unit_upstream_proof$map_rect_chunked)
      map_rect_header <-  NULL
      automatic_map_header <-  NULL
      if (apply_nuisance_block_patch) {
            automatic_map_header <-  normalizePath(path = file.path(dirname(release_header),
                  "map_rect_with_the_automatic_scalar_type_of_the_model.hpp"), mustWork = TRUE)
      }
      if (apply_map_rect_patch) {
            map_rect_header <-  normalizePath(path = file.path(dirname(release_header),
                  "map_rect_with_the_nuisance_block_owned_by_NicoStan.hpp"), mustWork = TRUE)
            message(paste0("map_rect_chunked: GENERAL nuisance derivatives; shared/job AD types are ",
                            "preserved; automatic map calls require verified observation-partition metadata."))
      }
      ## (chunk-local bookkeeping on its own is switched off: its gain was within the noise; it is used only
      ##  inside nico_reduce_sum, i.e. together with NicoStan.reduce_sum_nuisance_block = TRUE)
      if (apply_lean_slicing_patch && !apply_nuisance_block_patch) {
            stop(paste0("NicoStan.reduce_sum_lean_slicing on its own is switched off (its gain was within the ",
                        "noise); it is used only with NicoStan.reduce_sum_nuisance_block = TRUE ",
                        "(nico_reduce_sum)."))
      }
      nuisance_block_patch_lines <-  if (apply_nuisance_block_patch) {
            if (isTRUE(unit_upstream_proof$proven)) {
                  readLines(con = fn_file_of_the_one_NicoStan_install(
                        "include", "NicoStan", "runtime", "general_functions",
                        "reduce_sum_with_proven_unit_upstream_owned_by_NicoStan.inc",
                        package = "NicoStan", mustWork = TRUE), warn = FALSE)
            } else {
                  R_fn_bridge_memory_reduce_sum_nuisance_block_patch_lines()
            }
      } else {
            character(0)
      }
      ## (the chunk-local leapfrog step patch, options(NicoStan.reduce_sum_chunk_local_leapfrog_step = TRUE): on
      ##  top of the nuisance block patch; off by default)
      apply_chunk_local_leapfrog_step_patch <-  isTRUE(x = getOption(
                                                       "NicoStan.reduce_sum_chunk_local_leapfrog_step", FALSE))
      if (apply_chunk_local_leapfrog_step_patch && !apply_nuisance_block_patch) {
            stop(paste0("NicoStan.reduce_sum_chunk_local_leapfrog_step needs ",
                        "NicoStan.reduce_sum_nuisance_block = TRUE."))
      }
      chunk_local_leapfrog_step_patch_lines <-  if (apply_chunk_local_leapfrog_step_patch) {
            R_fn_bridge_memory_reduce_sum_chunk_local_leapfrog_step_patch_lines()
      } else {
            character(0)
      }
      lean_slicing_patch_lines <-  if (apply_lean_slicing_patch) {
            R_fn_bridge_memory_reduce_sum_lean_slicing_patch_lines()
      } else {
            character(0)
      }
      ## (the md5 covers every patch applied, the C++ and the R that edits the lines of BridgeStan's files: a
      ##  change of any makes a new private runtime; with neither reduce_sum patch it is the page-release patch's
      ##  alone)
      patch_md5 <-  if (apply_reduce_sum_patch) {
            R_fn_bridge_memory_patch_md5(patch_lines = c(patch_lines, "---- reduce_sum ----",
                                                         reduce_sum_patch_lines))
      } else if (apply_lean_slicing_patch) {
            lean_slicing_R_functions <-  c(deparse(R_fn_bridge_memory_patch_reduce_sum_lean_slicing),
                                           deparse(R_fn_bridge_memory_patch_bridgestan_gradient_without_copies))
            nuisance_block_lines <-  if (apply_nuisance_block_patch) {
                  c("---- reduce_sum nuisance block ----", nuisance_block_patch_lines,
                    if (isTRUE(unit_upstream_proof$proven)) {
                          deparse(R_fn_bridge_memory_patch_reduce_sum_with_proven_unit_upstream)
                    } else {
                          deparse(R_fn_bridge_memory_patch_reduce_sum_nuisance_block)
                    },
                    deparse(R_fn_bridge_memory_patch_bridgestan_gradient_with_the_nuisance_block))
            } else {
                  character(0)
            }
            ## (with the option off, nothing is added, so that the md5, and with it every library already built,
            ##  stays as it was)
            chunk_local_leapfrog_step_lines <-  if (apply_chunk_local_leapfrog_step_patch) {
                  c("---- reduce_sum chunk-local leapfrog step ----", chunk_local_leapfrog_step_patch_lines,
                    deparse(R_fn_bridge_memory_patch_reduce_sum_chunk_local_leapfrog_step),
                    deparse(R_fn_bridge_memory_patch_bridgestan_gradient_with_the_chunk_local_leapfrog_step))
            } else {
                  character(0)
            }
            R_fn_bridge_memory_patch_md5(patch_lines = c(patch_lines, "---- reduce_sum lean slicing ----",
                                                         lean_slicing_patch_lines, lean_slicing_R_functions,
                                                         nuisance_block_lines, chunk_local_leapfrog_step_lines))
      } else {
            R_fn_bridge_memory_patch_md5(patch_lines = patch_lines)
      }
      {
            ## (the runtime's patched files include these headers by PATH: a header outside the one install, the
            ##  development tree's, must give the runtime a key of its own, or a build against the installed
            ##  headers would reuse it and read the same header twice as two files; with every header under the
            ##  one install nothing is added, so that the md5 of every library already built stays)
            folder_of_the_headers_of_the_one_install <-  normalizePath(path = fn_file_of_the_one_NicoStan_install(
                  "include", "NicoStan", "runtime", "general_functions",
                  package = "NicoStan", mustWork = TRUE), mustWork = TRUE)
            paths_of_the_headers_outside_the_one_install <-  c(release_header, map_rect_header, automatic_map_header)
            paths_of_the_headers_outside_the_one_install <-  paths_of_the_headers_outside_the_one_install[
                  !startsWith(x = paths_of_the_headers_outside_the_one_install,
                              prefix = folder_of_the_headers_of_the_one_install)]
            patch_md5 <-  R_fn_bridge_memory_patch_md5(patch_lines = c(
                  patch_md5, paths_of_the_headers_outside_the_one_install,
                  if (apply_map_rect_patch) readLines(con = map_rect_header, warn = FALSE),
                  if (apply_nuisance_block_patch) readLines(con = automatic_map_header, warn = FALSE),
                  deparse(R_fn_Stan_parallel_call_names), deparse(R_fn_read_Stan_source_with_includes),
                  deparse(R_fn_Stan_map_type_in_helper_functions),
                  deparse(R_fn_bridge_memory_generate_Stan_model),
                  deparse(R_fn_bridge_memory_patch_map_rect_chunked),
                  deparse(R_fn_bridge_memory_patch_generated_map_calls),
                  deparse(R_fn_bridge_memory_patch_map_generation)))
      }
      existing_marker_path <-  file.path(original_bridgestan_path,
                                        ".nicostan_page_release_ready")
      if (file.exists(existing_marker_path)) {
            existing_marker <-  readLines(con = existing_marker_path, warn = FALSE)
            if (any(existing_marker == paste0("patch_md5=", patch_md5))) {
                  message(paste0("Reusing private page-release BridgeStan runtime: ",
                                 original_bridgestan_path))
                  return(list(bridgestan_path = original_bridgestan_path,
                              original_bridgestan_path = original_bridgestan_path,
                              runtime_key = sub("^nicostan_page_release_bridgestan_", "",
                                                 basename(original_bridgestan_path)),
                              release_header = release_header,
                              patch_md5 = patch_md5))
            }
      }
      identity_file <-  tempfile(pattern = "nicostan_page_release_identity_")
      on.exit(unlink(x = identity_file, force = TRUE), add = TRUE)
      writeLines(text = c(original_bridgestan_path, original_stack_alloc_md5, patch_md5),
                 con = identity_file, useBytes = TRUE)
      runtime_key <-  unname(obj = tools::md5sum(files = identity_file))
      dir.create(path = runtime_cache_dir, recursive = TRUE, showWarnings = FALSE)
      runtime_path <-  file.path(runtime_cache_dir,
                                paste0("nicostan_page_release_bridgestan_", runtime_key))
      marker_path <-  file.path(runtime_path, ".nicostan_page_release_ready")
      marker_lines <-  c("nicostan_page_release_runtime_v1",
                        paste0("original_bridgestan_path=", original_bridgestan_path),
                        paste0("original_stack_alloc_md5=", original_stack_alloc_md5),
                        paste0("patch_md5=", patch_md5))
      if (file.exists(marker_path) &&
          identical(x = readLines(con = marker_path, warn = FALSE), y = marker_lines)) {
            message(paste0("Reusing private page-release BridgeStan runtime: ", runtime_path))
            return(list(bridgestan_path = normalizePath(path = runtime_path, mustWork = TRUE),
                        original_bridgestan_path = original_bridgestan_path,
                        runtime_key = runtime_key,
                        release_header = release_header,
                        patch_md5 = patch_md5))
      }
      temporary_path <-  tempfile(pattern = ".nicostan_page_release_", tmpdir = runtime_cache_dir)
      if (!dir.create(path = temporary_path, recursive = TRUE, showWarnings = FALSE)) {
            stop("Could not create temporary private BridgeStan runtime: ", temporary_path)
      }
      keep_temporary <-  TRUE
      on.exit(if (keep_temporary) unlink(x = temporary_path, recursive = TRUE, force = TRUE),
              add = TRUE)
      source_entries <-  list.files(path = original_bridgestan_path, all.files = TRUE,
                                   full.names = TRUE, no.. = TRUE)
      for (source_entry in source_entries) {
            if (!isTRUE(x = file.copy(from = source_entry, to = temporary_path,
                                      recursive = TRUE, copy.date = TRUE))) {
                  stop("Could not copy BridgeStan source entry: ", source_entry)
            }
      }
      copied_stack_alloc_file <-  file.path(temporary_path, "stan", "lib", "stan_math",
                                           "stan", "math", "memory", "stack_alloc.hpp")
      if (!file.exists(copied_stack_alloc_file)) {
            stop("Copied BridgeStan runtime has no stack_alloc.hpp: ", copied_stack_alloc_file)
      }
      R_fn_bridge_memory_patch_stack_alloc(stack_alloc_file = copied_stack_alloc_file,
                                           patch_lines = patch_lines)
      copied_reduce_sum_file <-  file.path(temporary_path, "stan", "lib", "stan_math",
                                          "stan", "math", "rev", "functor", "reduce_sum.hpp")
      if (!file.exists(copied_reduce_sum_file)) {
            stop("Copied BridgeStan runtime has no reduce_sum.hpp: ", copied_reduce_sum_file)
      }
      if (apply_reduce_sum_patch) {
            R_fn_bridge_memory_patch_reduce_sum(reduce_sum_file = copied_reduce_sum_file,
                                                patch_lines = reduce_sum_patch_lines)
      }
      if (apply_lean_slicing_patch) {
            R_fn_bridge_memory_patch_reduce_sum_lean_slicing(reduce_sum_file = copied_reduce_sum_file,
                                                             patch_lines = lean_slicing_patch_lines)
            copied_model_rng_file <-  file.path(temporary_path, "src", "model_rng.cpp")
            if (!file.exists(copied_model_rng_file)) {
                  stop("Copied BridgeStan runtime has no model_rng.cpp: ", copied_model_rng_file)
            }
            R_fn_bridge_memory_patch_bridgestan_gradient_without_copies(model_rng_file = copied_model_rng_file)
            if (apply_nuisance_block_patch) {
                  if (isTRUE(unit_upstream_proof$proven)) {
                        R_fn_bridge_memory_patch_reduce_sum_with_proven_unit_upstream(
                              reduce_sum_file = copied_reduce_sum_file, patch_lines = nuisance_block_patch_lines)
                  } else {
                        R_fn_bridge_memory_patch_reduce_sum_nuisance_block(
                              reduce_sum_file = copied_reduce_sum_file, patch_lines = nuisance_block_patch_lines)
                  }
                  R_fn_bridge_memory_patch_bridgestan_gradient_with_the_nuisance_block(
                        model_rng_file = copied_model_rng_file,
                        bridgestan_file = file.path(temporary_path, "src", "bridgestan.cpp"))
            }
            if (apply_chunk_local_leapfrog_step_patch) {
                  R_fn_bridge_memory_patch_reduce_sum_chunk_local_leapfrog_step(
                        reduce_sum_file             = copied_reduce_sum_file,
                        patch_lines                 = chunk_local_leapfrog_step_patch_lines,
                        nuisance_block_patch_lines  = nuisance_block_patch_lines)
                  R_fn_bridge_memory_patch_bridgestan_gradient_with_the_chunk_local_leapfrog_step(
                        model_rng_file = copied_model_rng_file,
                        bridgestan_file = file.path(temporary_path, "src", "bridgestan.cpp"))
            }
      }
      R_fn_bridge_memory_patch_map_generation(runtime_path = temporary_path,
                                               automatic_header = automatic_map_header)
      if (apply_map_rect_patch) {
            R_fn_bridge_memory_patch_map_rect_chunked(
                  map_rect_file = file.path(temporary_path, "stan", "lib", "stan_math", "stan", "math",
                                            "rev", "functor", "map_rect_concurrent.hpp"),
                  runtime_header = map_rect_header)
      }
      copied_objects <-  list.files(path = file.path(temporary_path, "src"),
                                   pattern = "\\.o$", full.names = TRUE,
                                   recursive = FALSE)
      if (length(x = copied_objects) > 0) unlink(x = copied_objects, force = TRUE)
      writeLines(text = marker_lines, con = file.path(temporary_path,
                                                       ".nicostan_page_release_ready"),
                 useBytes = TRUE)
      if (dir.exists(runtime_path)) unlink(x = runtime_path, recursive = TRUE, force = TRUE)
      if (!isTRUE(x = file.rename(from = temporary_path, to = runtime_path))) {
            stop("Could not publish private page-release BridgeStan runtime: ", runtime_path)
      }
      keep_temporary <-  FALSE
      message(paste0("Prepared private page-release BridgeStan runtime: ", runtime_path))
      list(bridgestan_path = normalizePath(path = runtime_path, mustWork = TRUE),
           original_bridgestan_path = original_bridgestan_path,
           runtime_key = runtime_key,
           release_header = release_header,
           patch_md5 = patch_md5)
}

## Inspect exports without loading a model DSO into the controlling R process.
## This staged implementation supports the current Linux host only.
## ---- TRUE when a compiled model library is in NicoStan's nuisance-block form (it exports
##      nicostan_bs_log_density_gradient_with_the_nuisance_block)
R_fn_model_library_is_in_NicoStan_nuisance_block_form <-  function(model_library) {
      if (!file.exists(model_library) || !nzchar(x = Sys.which(names = "nm"))) return(FALSE)
      exports <-  suppressWarnings(system2(command = Sys.which(names = "nm"),
                                           args = c("-D", "--defined-only", shQuote(string = model_library)),
                                           stdout = TRUE, stderr = FALSE))
      any(grepl(pattern = "[[:space:]]nicostan_bs_log_density_gradient_with_the_nuisance_block$", x = exports))
}

R_fn_bridge_memory_has_export <-  function(model_library) {
      if (!file.exists(model_library)) return(FALSE)
      if (!identical(x = Sys.info()[["sysname"]], y = "Linux")) {
            stop("The staged model export check currently supports Linux only.")
      }
      nm_path <-  Sys.which(names = "nm")
      if (!nzchar(x = nm_path)) stop("Cannot verify model cleanup capability: nm is unavailable.")
      exports <-  system2(command = nm_path,
                         args = c("-D", "--defined-only", shQuote(string = model_library)),
                         stdout = TRUE, stderr = TRUE)
      status <-  attr(x = exports, which = "status")
      if (!is.null(status) && status != 0) stop("Could not inspect compiled model exports: ", model_library)
      any(grepl(pattern = "[[:space:]]nicostan_bs_release_thread_memory_v1$", x = exports))
}






















