## Model compilation support. This changes model artifacts, never result caches.
R_fn_bridge_memory_compile_options <- function(model_file, make_args = NULL,
                                               stanc_args = NULL,
                                               release_header = NULL) {
      if (is.null(x = release_header)) {
            release_header <- system.file("include", "NicoStan", "runtime",
                                          "general_functions", "bridge_memory.hpp",
                                          package = "NicoStan", mustWork = TRUE)
      }
      release_header <- normalizePath(path = release_header, mustWork = TRUE)
      wrapper_file <- paste0(tools::file_path_sans_ext(x = model_file), "_nicostan_user_header.hpp")
      make_args <- as.character(x = unlist(x = make_args))
      user_header_entries <- grep(pattern = "^USER_HEADER=", x = make_args)
      if (length(user_header_entries) > 1L) stop("Multiple USER_HEADER arguments are ambiguous.")
      wrapper_lines <- c("#pragma once",
                         paste0("#include ", encodeString(x = release_header, quote = '"')))
      if (length(user_header_entries) == 1L) {
            original_header <- sub(pattern = "^USER_HEADER=", replacement = "",
                                   x = make_args[user_header_entries])
            original_header <- normalizePath(path = original_header, mustWork = TRUE)
            if (identical(x = original_header, y = normalizePath(path = wrapper_file, mustWork = FALSE))) {
                  existing_lines <- readLines(con = original_header, warn = FALSE)
                  if (length(existing_lines) < 2L || existing_lines[1] != "#pragma once" ||
                      !startsWith(x = existing_lines[2], prefix = "#include ")) {
                        stop("Cannot safely reuse an unrecognised generated user header: ", original_header)
                  }
                  ## Keep its original custom includes, replacing only the cleanup include.
                  wrapper_lines <- c(wrapper_lines, existing_lines[-c(1L, 2L)])
            } else {
                  wrapper_lines <- c(wrapper_lines,
                                     paste0("#include ", encodeString(x = original_header, quote = '"')))
            }
            make_args <- make_args[-user_header_entries]
      }
      if (!file.exists(wrapper_file) ||
          !identical(x = readLines(con = wrapper_file, warn = FALSE), y = wrapper_lines)) {
            writeLines(text = wrapper_lines, con = wrapper_file)
      }
      list(make_args = as.list(c(make_args, paste0("USER_HEADER=", normalizePath(path = wrapper_file)))),
           stanc_args = as.list(unique(c(as.character(x = unlist(x = stanc_args)), "--allow-undefined"))),
           release_header_md5 = unname(obj = tools::md5sum(files = release_header)))
}

R_fn_bridge_memory_page_release_patch_lines <- function(allocator_methods = NULL) {
      if (is.null(x = allocator_methods)) {
            allocator_methods <- system.file("include", "NicoStan", "runtime",
                                             "general_functions",
                                             "bridge_memory_allocator_methods.inc",
                                             package = "NicoStan", mustWork = TRUE)
      }
      allocator_methods <- normalizePath(path = allocator_methods, mustWork = TRUE)
      method_lines <- readLines(con = allocator_methods, warn = FALSE)
      if (length(x = method_lines) == 0L ||
          !identical(x = method_lines[1L], y = "#ifdef __linux__") ||
          !identical(x = method_lines[length(x = method_lines)], y = "#endif")) {
            stop("Unexpected BridgeStan page-release method snippet: ", allocator_methods)
      }
      c("#ifdef __linux__", "#include <sys/mman.h>", "#include <unistd.h>",
        "#endif", method_lines)
}

R_fn_bridge_memory_patch_md5 <- function(patch_lines) {
      patch_file <- tempfile(pattern = "nicostan_page_release_patch_")
      on.exit(unlink(x = patch_file, force = TRUE), add = TRUE)
      writeLines(text = patch_lines, con = patch_file, useBytes = TRUE)
      unname(obj = tools::md5sum(files = patch_file))
}

R_fn_bridge_memory_patch_stack_alloc <- function(stack_alloc_file, patch_lines) {
      stack_lines <- readLines(con = stack_alloc_file, warn = FALSE)
      if (any(grepl(pattern = "nicostan_discard_pages", x = stack_lines, fixed = TRUE))) {
            stop("BridgeStan stack_alloc.hpp already contains page-release methods: ", stack_alloc_file)
      }
      include_anchor <- which(stack_lines == "#include <vector>")
      public_anchor <- which(stack_lines == " public:")
      if (length(x = include_anchor) != 1L || length(x = public_anchor) != 1L) {
            stop("Unexpected BridgeStan stack_alloc.hpp patch anchors: ", stack_alloc_file)
      }
      stack_lines <- append(x = stack_lines, values = patch_lines[1:4],
                            after = include_anchor)
      public_anchor <- which(stack_lines == " public:")
      stack_lines <- append(x = stack_lines, values = patch_lines[-(1:4)],
                            after = public_anchor)
      writeLines(text = stack_lines, con = stack_alloc_file, useBytes = TRUE)
      invisible(stack_alloc_file)
}

R_fn_bridge_memory_prepare_runtime <- function(
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
      if (is.null(x = original_bridgestan_path)) original_bridgestan_path <- original_runtime_path
      if (is.null(x = original_bridgestan_path)) {
            if (!requireNamespace(package = "bridgestan", quietly = TRUE)) {
                  stop("The bridgestan package is required to locate BridgeStan.")
            }
            original_bridgestan_path <- getFromNamespace("get_bridgestan_path", "bridgestan")(
                  download = FALSE)
      }
      original_bridgestan_path <- normalizePath(path = original_bridgestan_path, mustWork = TRUE)
      if (is.null(x = release_header)) {
            release_header <- system.file("include", "NicoStan", "runtime",
                                          "general_functions", "bridge_memory.hpp",
                                          package = "NicoStan", mustWork = TRUE)
      }
      release_header <- normalizePath(path = release_header, mustWork = TRUE)
      stack_alloc_file <- file.path(original_bridgestan_path, "stan", "lib", "stan_math",
                                    "stan", "math", "memory", "stack_alloc.hpp")
      if (!file.exists(stack_alloc_file)) {
            stop("Configured BridgeStan source has no stack_alloc.hpp: ", stack_alloc_file)
      }
      original_stack_alloc_md5 <- unname(obj = tools::md5sum(files = stack_alloc_file))
      patch_lines <- R_fn_bridge_memory_page_release_patch_lines(
            allocator_methods = allocator_methods)
      patch_md5 <- R_fn_bridge_memory_patch_md5(patch_lines = patch_lines)
      existing_marker_path <- file.path(original_bridgestan_path,
                                        ".nicostan_page_release_ready")
      if (file.exists(existing_marker_path)) {
            existing_marker <- readLines(con = existing_marker_path, warn = FALSE)
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
      identity_file <- tempfile(pattern = "nicostan_page_release_identity_")
      on.exit(unlink(x = identity_file, force = TRUE), add = TRUE)
      writeLines(text = c(original_bridgestan_path, original_stack_alloc_md5, patch_md5),
                 con = identity_file, useBytes = TRUE)
      runtime_key <- unname(obj = tools::md5sum(files = identity_file))
      dir.create(path = runtime_cache_dir, recursive = TRUE, showWarnings = FALSE)
      runtime_path <- file.path(runtime_cache_dir,
                                paste0("nicostan_page_release_bridgestan_", runtime_key))
      marker_path <- file.path(runtime_path, ".nicostan_page_release_ready")
      marker_lines <- c("nicostan_page_release_runtime_v1",
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
      temporary_path <- tempfile(pattern = ".nicostan_page_release_", tmpdir = runtime_cache_dir)
      if (!dir.create(path = temporary_path, recursive = TRUE, showWarnings = FALSE)) {
            stop("Could not create temporary private BridgeStan runtime: ", temporary_path)
      }
      keep_temporary <- TRUE
      on.exit(if (keep_temporary) unlink(x = temporary_path, recursive = TRUE, force = TRUE),
              add = TRUE)
      source_entries <- list.files(path = original_bridgestan_path, all.files = TRUE,
                                   full.names = TRUE, no.. = TRUE)
      for (source_entry in source_entries) {
            if (!isTRUE(x = file.copy(from = source_entry, to = temporary_path,
                                      recursive = TRUE, copy.date = TRUE))) {
                  stop("Could not copy BridgeStan source entry: ", source_entry)
            }
      }
      copied_stack_alloc_file <- file.path(temporary_path, "stan", "lib", "stan_math",
                                           "stan", "math", "memory", "stack_alloc.hpp")
      if (!file.exists(copied_stack_alloc_file)) {
            stop("Copied BridgeStan runtime has no stack_alloc.hpp: ", copied_stack_alloc_file)
      }
      R_fn_bridge_memory_patch_stack_alloc(stack_alloc_file = copied_stack_alloc_file,
                                           patch_lines = patch_lines)
      copied_objects <- list.files(path = file.path(temporary_path, "src"),
                                   pattern = "\\.o$", full.names = TRUE,
                                   recursive = FALSE)
      if (length(x = copied_objects) > 0L) unlink(x = copied_objects, force = TRUE)
      writeLines(text = marker_lines, con = file.path(temporary_path,
                                                       ".nicostan_page_release_ready"),
                 useBytes = TRUE)
      if (dir.exists(runtime_path)) unlink(x = runtime_path, recursive = TRUE, force = TRUE)
      if (!isTRUE(x = file.rename(from = temporary_path, to = runtime_path))) {
            stop("Could not publish private page-release BridgeStan runtime: ", runtime_path)
      }
      keep_temporary <- FALSE
      message(paste0("Prepared private page-release BridgeStan runtime: ", runtime_path))
      list(bridgestan_path = normalizePath(path = runtime_path, mustWork = TRUE),
           original_bridgestan_path = original_bridgestan_path,
           runtime_key = runtime_key,
           release_header = release_header,
           patch_md5 = patch_md5)
}

## Inspect exports without loading a model DSO into the controlling R process.
## This staged implementation supports the current Linux host only.
R_fn_bridge_memory_has_export <- function(model_library) {
      if (!file.exists(model_library)) return(FALSE)
      if (!identical(x = Sys.info()[["sysname"]], y = "Linux")) {
            stop("The staged model export check currently supports Linux only.")
      }
      nm_path <- Sys.which(names = "nm")
      if (!nzchar(x = nm_path)) stop("Cannot verify model cleanup capability: nm is unavailable.")
      exports <- system2(command = nm_path,
                         args = c("-D", "--defined-only", shQuote(string = model_library)),
                         stdout = TRUE, stderr = TRUE)
      status <- attr(x = exports, which = "status")
      if (!is.null(status) && status != 0L) stop("Could not inspect compiled model exports: ", model_library)
      any(grepl(pattern = "[[:space:]]nicostan_bs_release_thread_memory_v1$", x = exports))
}






















