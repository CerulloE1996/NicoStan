##
## ==== R_fn_write_Stan_math_fused_version_of_Stan_model.R ====
##
## ---- The Stan::math-fused version of a Stan model, written from its nico::math-fused (AVX) version: every
##      nico_ copy of a Stan function goes back to the Stan function; every fused node with no Stan function
##      (e.g. nico_binomial_counts_above_successive_cutpoints_of_a_cumulative_link_lpmf) stays, and its library is
##      built with NICO_SIMD_SCALAR_KERNELS so that the node runs Stan::math's scalar functions.
##




#' The names of every Stan math function, from stanc's own list of signatures
#'
#' @param stanc_path The stanc of the CmdStan used for the builds.
#' @return The function names (e.g. "normal_lpdf", "exp", "bernoulli_logit_glm_lpmf").
#' @noRd
fn_names_of_Stan_math_functions <-  function( stanc_path
) {

      stopifnot(file.exists(stanc_path))
      signatures <-  system2(stanc_path,
                             "--dump-stan-math-signatures",
                             stdout = TRUE)
      lines_of_functions <-  signatures[grepl("^[A-Za-z_][A-Za-z0-9_]*\\(", signatures)]
      names_of_functions <-  unique(sub("\\(.*$", "", lines_of_functions))
      stopifnot(length(names_of_functions) > 100)
      return(names_of_functions)

}




#' Writes the Stan::math-fused Stan file of a model from its nico::math-fused (AVX) Stan file
#'
#' @param AVX_fused_Stan_file_path The nico::math-fused Stan file (it calls at least one fused node).
#' @param Stan_math_fused_Stan_file_path The Stan file to write.
#' @param names_of_Stan_math_functions From fn_names_of_Stan_math_functions().
#' @return A list: the file written, the fused nodes kept and the nico_ copies given back their Stan names.
#' @noRd
fn_write_Stan_math_fused_version_of_Stan_model <-  function( AVX_fused_Stan_file_path,
                                                             Stan_math_fused_Stan_file_path,
                                                             names_of_Stan_math_functions
) {

      stopifnot(file.exists(AVX_fused_Stan_file_path),
                AVX_fused_Stan_file_path != Stan_math_fused_Stan_file_path)
      lines <-  readLines(AVX_fused_Stan_file_path, warn = FALSE)
      ##
      ## ---- the nico_ names of the code (comments left out):
      ##
      code <-  sub("//.*$", "", lines)
      nico_names <-  unique(unlist(regmatches(code, gregexpr("\\bnico_[A-Za-z0-9_]+", code, perl = TRUE))))
      ##
      ## ---- a copy of a Stan function: without nico_ (_lupdf / _lupmf read as _lpdf / _lpmf) a Stan function,
      ##      or a distribution of a "~" statement (its _lpdf or _lpmf a Stan function):
      ##
      Stan_names <-  sub("_lup(df|mf)$", "_lp\\1", sub("^nico_", "", nico_names))
      is_a_copy <-  (Stan_names %in% names_of_Stan_math_functions) |
                    (paste0(Stan_names, "_lpdf") %in% names_of_Stan_math_functions) |
                    (paste0(Stan_names, "_lpmf") %in% names_of_Stan_math_functions)
      fused_nodes_kept <-  nico_names[!is_a_copy]
      if (length(fused_nodes_kept) == 0) {
            stop(paste0(basename(AVX_fused_Stan_file_path), " calls no fused node: its Stan::math version is ",
                        "the Stan::math version itself, not a Stan::math-fused one"))
      }
      copies <-  nico_names[is_a_copy]
      ##
      ## ---- declarations of the copies (functions block, up to their ";") are left out:
      ##
      ## (a declaration starts with a return type of Stan, never with "return" or a variable)
      types_of_Stan <-  paste0("(?:real|int|void|vector|row_vector|matrix|complex|complex_vector|",
                               "complex_row_vector|complex_matrix|",
                               "array\\[[ ,]*\\]\\s+[A-Za-z_]+(?:\\[[ ,]*\\])?|tuple\\(.*\\))")
      start_of_a_declaration <-  paste0("^\\s*", types_of_Stan, "\\s+(", paste(copies, collapse = "|"),
                                        ")\\s*\\(")
      keep <-  rep(TRUE, length(lines))
      i <-  1
      while (length(copies) > 0 && i <= length(lines)) {
            if (grepl(start_of_a_declaration, code[i], perl = TRUE)) {
                  j <-  i
                  while (j <= length(lines) && !grepl(";\\s*$", code[j]) && !grepl("\\{", code[j])) j <-  j + 1
                  if (j <= length(lines) && grepl(";\\s*$", code[j])) keep[i:j] <-  FALSE
                  i <-  j
            }
            i <-  i + 1
      }
      lines <-  lines[keep]
      ##
      ## ---- every call of a copy gets its Stan name (longest names first):
      ##
      for (k in order(nchar(copies), decreasing = TRUE)) {
            lines <-  gsub(paste0("\\b", copies[k], "\\b"), sub("^nico_", "", copies[k]), lines, perl = TRUE)
      }
      writeLines(lines, Stan_math_fused_Stan_file_path)
      return(list( Stan_math_fused_Stan_file_path  = Stan_math_fused_Stan_file_path,
                   fused_nodes_kept                = fused_nodes_kept,
                   copies_given_their_Stan_names   = copies))

}























