##
## ==== Stan source and its included functions ================================================================
##
R_fn_read_Stan_source_with_includes <-  function( model_file,
                                                  stanc_args) {

      arguments <-  as.character(unlist(stanc_args, use.names = FALSE))
      include_paths <-  arguments[startsWith(arguments, "--include-paths=")]
      include_paths <-  sub("^--include-paths=", "", include_paths)
      separate <-  which(arguments == "--include-paths")
      if (any(separate == length(arguments))) stop("Missing Stan include paths.")
      include_paths <-  c(include_paths, arguments[separate + 1])
      include_paths <-  unlist(strsplit(include_paths, ",", fixed = TRUE), use.names = FALSE)
      include_paths <-  unique(c(dirname(normalizePath(model_file, mustWork = TRUE)), include_paths, getwd()))
      fn_expand <-  function( path,
                              ancestors) {

            path <-  normalizePath(path, mustWork = TRUE)
            if (path %in% ancestors) stop(paste0("Circular Stan include: ", path))
            code <-  paste(readLines(path, warn = FALSE), collapse = "\n")
            visible <-  code
            protected <-  "\"(?:[^\"\\\\]|\\\\.)*\"|/\\*[\\s\\S]*?\\*/|//[^\\n]*"
            matches <-  gregexpr(protected, code, perl = TRUE)[[1]]
            if (matches[1] > 0) {
                  lengths <-  attr(matches, "match.length")
                  for (index in seq_along(matches)) {
                        first <-  matches[index]
                        last <-  first + lengths[index] - 1
                        original <-  substr(visible, first, last)
                        substr(visible, first, last) <-  gsub("[^\n]", " ", original, perl = TRUE)
                  }
            }
            matches <-  gregexpr("(?m)^[ \t]*#include[ \t]+[^\n]*", visible, perl = TRUE)[[1]]
            if (matches[1] < 0) return(code)
            lengths <-  attr(matches, "match.length")
            for (index in rev(seq_along(matches))) {
                  first <-  matches[index]
                  last <-  first + lengths[index] - 1
                  directive <-  substr(code, first, last)
                  name <-  trimws(sub("^[ \t]*#include[ \t]+", "", directive))
                  name <-  sub("[ \t]*//.*$", "", name)
                  name <-  trimws(name)
                  if (startsWith(name, '"') && endsWith(name, '"')) {
                        name <-  substr(name, 2, nchar(name) - 1)
                  }
                  name <-  sub("^<([^>]+)>$", "\\1", name)
                  candidates <-  unique(c(file.path(dirname(path), name), file.path(include_paths, name), name))
                  candidates <-  candidates[file.exists(candidates)]
                  if (length(candidates) == 0) stop(paste0("Stan include not found: ", name, " in ", path))
                  expanded <-  fn_expand(candidates[1], c(ancestors, path))
                  code <-  paste0(substr(code, 1, first - 1), expanded, substring(code, last + 1))
            }
            return(code)
      }

      return(fn_expand(model_file, character(0)))

}

##
## ==== NicoStan's names for Stan's parallel calls =============================================================
##
R_fn_Stan_parallel_call_names <-  function(code) {

      ## Comments and strings retain their text; only actual call names are translated before stanc.
      protected_pattern <-  "\"(?:[^\"\\\\]|\\\\.)*\"|/\\*[\\s\\S]*?\\*/|//[^\\n]*"
      protected_matches <-  gregexpr(pattern = protected_pattern, text = code, perl = TRUE)[[1]]
      visible <-  code
      if (protected_matches[1] > 0) {
            lengths <-  attr(protected_matches, "match.length")
            for (index in seq_along(protected_matches)) {
                  first <-  protected_matches[index]
                  last <-  first + lengths[index] - 1
                  substr(visible, first, last) <-  strrep(" ", lengths[index])
            }
      }
      calls <-  gregexpr(pattern = "\\bnico_(?:map_rect|reduce_sum_static)\\s*(?=\\()",
                         text = visible, perl = TRUE)[[1]]
      if (calls[1] < 0) return(code)
      lengths <-  attr(calls, "match.length")
      for (index in rev(seq_along(calls))) {
            first <-  calls[index]
            last <-  first + lengths[index] - 1
            name <-  sub(pattern = "^nico_", replacement = "", x = substr(code, first, last))
            code <-  paste0(substr(code, 1, first - 1), name, substring(code, last + 1))
      }
      return(code)

}





















