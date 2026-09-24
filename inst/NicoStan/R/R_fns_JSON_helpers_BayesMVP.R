






#### =====================================================================================================================================
##
## ---- Atomic file writes into the SHARED stan_data folder:
##
##      convert_stan_data_list_to_JSON() and friends write content-addressed files (data_<md5>.json, and the per-stream
##      copies data_<md5>_<stream>.json) into the installed package's stan_data folder, which every R process on the
##      machine shares. Writing them in place let two concurrent fits with the same data read a half-written file
##      ("Invalid JSON ... premature EOF" was observed). Every writer now writes a unique temporary file IN THE SAME
##      DIRECTORY (process id + tempfile's random suffix, so concurrent processes never share it) and file.rename()s it
##      over the target. rename() within one directory is atomic on POSIX file systems, so a reader sees either the old
##      complete file or the new complete file, never a partial one. File names and contents are unchanged, so cached
##      files keep working. tempfile() does not touch R's random-number stream, so the user's MCMC seed is unaffected.
##



#' fn_write_file_atomically
#'
#' Writes a file by calling fn_write_to_temporary_file_path(temporary_file_path) on a unique temporary file in the
#' target's directory, then renames it over target_file_path (atomic within one directory on POSIX file systems).
#' @param target_file_path the final file path.
#' @param fn_write_to_temporary_file_path a function of one argument (the temporary file path) that writes the content.
#' @return invisibly, target_file_path.
fn_write_file_atomically <- function( target_file_path,
                                      fn_write_to_temporary_file_path) {

        ##
        target_directory_path <- dirname(target_file_path)
        ##
        if (!dir.exists(target_directory_path)) {
              dir.create( path         = target_directory_path,
                          recursive    = TRUE,
                          showWarnings = FALSE)
        }
        ##
        target_file_extension <- sub( pattern     = "^.*(\\.[^.]*)$",
                                      replacement = "\\1",
                                      x           = basename(target_file_path))
        if (identical(x = target_file_extension, y = basename(target_file_path))) target_file_extension <- ""
        ##
        ## ---- unique per process (pid) AND per call (tempfile's random suffix, re-drawn until the name is free);
        ##      hidden (leading ".") and kept in the SAME directory so file.rename() is a same-file-system rename:
        ##
        temporary_file_path <- tempfile( pattern = paste0(".tmp_", Sys.getpid(), "_", sub("\\.[^.]*$", "", basename(target_file_path)), "_"),
                                         tmpdir  = target_directory_path,
                                         fileext = target_file_extension)
        ##
        on.exit(expr = if (file.exists(temporary_file_path)) unlink(temporary_file_path), add = TRUE)
        ##
        fn_write_to_temporary_file_path(temporary_file_path)
        ##
        if (!file.exists(temporary_file_path)) {
              stop(paste0("fn_write_file_atomically: the writer did not create the temporary file for ", target_file_path, "."))
        }
        ##
        rename_succeeded <- file.rename( from = temporary_file_path,
                                         to   = target_file_path)
        ##
        if (!isTRUE(rename_succeeded)) {
              stop(paste0("fn_write_file_atomically: could not rename the temporary file ", temporary_file_path, " to ", target_file_path, "."))
        }
        ##
        return(invisible(target_file_path))

}



#' fn_write_stan_json_atomically
#'
#' Writes a Stan data (or inits) list to json_file_path atomically: "{}" for an empty list (BridgeStan requires an
#' empty OBJECT, whereas cmdstanr::write_stan_json() would write an empty ARRAY), cmdstanr::write_stan_json() otherwise.
#' The content is identical to the previous in-place writes.
#' @param stan_data_list the list to write.
#' @param json_file_path the final file path.
#' @return invisibly, json_file_path.
fn_write_stan_json_atomically <- function( stan_data_list,
                                           json_file_path) {

        fn_write_file_atomically( target_file_path                = json_file_path,
                                  fn_write_to_temporary_file_path = function(temporary_file_path) {
                                        if (length(stan_data_list) == 0) {
                                              writeLines( text = "{}",
                                                          con  = temporary_file_path)
                                        } else {
                                              cmdstanr::write_stan_json( data = stan_data_list,
                                                                         file = temporary_file_path)
                                        }
                                  })

}



#' fn_copy_file_atomically
#'
#' Copies source_file_path to target_file_path atomically (copy to a unique temporary file in the target's directory,
#' then rename it over the target).
#' @param source_file_path the file to copy.
#' @param target_file_path the final file path.
#' @return invisibly, target_file_path.
fn_copy_file_atomically <- function( source_file_path,
                                     target_file_path) {

        fn_write_file_atomically( target_file_path                = target_file_path,
                                  fn_write_to_temporary_file_path = function(temporary_file_path) {
                                        copy_succeeded <- file.copy( from      = source_file_path,
                                                                     to        = temporary_file_path,
                                                                     overwrite = TRUE)
                                        if (!isTRUE(copy_succeeded)) {
                                              stop(paste0("fn_copy_file_atomically: could not copy ", source_file_path, " to ", temporary_file_path, "."))
                                        }
                                  })

}




#' convert_stan_data_list_to_JSON
#' @export
convert_stan_data_list_to_JSON <- function(stan_data_list, 
                                           pkg_data_dir = NULL) {
  
        if (is.null(pkg_data_dir)) { 
          pkg_root_dir <- system.file(package = "NicoStan")
          pkg_data_dir <- file.path(pkg_root_dir, "stan_data")  # directory to store data inc. JSON data files
        }
        ##
        ## Create data directory if it doesn't exist:
        ##
        if (!dir.exists(pkg_data_dir)) {
          dir.create(pkg_data_dir, recursive = TRUE)
        }
        ##
        ## make persistent (non-temp) JSON data file path with unique identifier:
        ##
        data_hash <- digest::digest(stan_data_list)  # Hash the data to create unique identifier
        json_filename <- paste0("data_", data_hash, ".json")
        json_file_path <- file.path(pkg_data_dir, json_filename)
        ##
        ## write JSON data using cmdstanr:
        ##
        ## ---- a model with NO data gets "{}" (cmdstanr::write_stan_json() would write an empty ARRAY ([]), but
        ##      BridgeStan requires an empty OBJECT ({})). Written ATOMICALLY (temporary file + rename), because this
        ##      content-addressed file sits in the shared stan_data folder and concurrent fits with the same data
        ##      read it (see fn_write_file_atomically):
        ##
        fn_write_stan_json_atomically( stan_data_list = stan_data_list,
                                       json_file_path = json_file_path)
        ##
        ## Output the JSON file path:
        ##
        return(json_file_path)
  
}









#' convert_JSON_string_to_R_vector
#' @export
convert_JSON_string_to_R_vector <- function(json_string) {
  
        require(jsonlite)
        
        ## Helper function to flatten a matrix/array row by row
        flatten_matrix <- function(m) {
          if(is.null(dim(m))) {
            return(as.numeric(m))
          }
          return(as.numeric(t(matrix(unlist(m), ncol=ncol(m), byrow=TRUE))))
        }
        
        ## Helper function to recursively process nested structures
        process_element <- function(x) {
          if(is.numeric(x) && length(x) == 1) {
            return(x)
          }
          if(is.list(x) || is.matrix(x) || is.array(x)) {
            if(is.matrix(x) || (is.list(x) && all(sapply(x, length) == length(x[[1]])))) {
              return(flatten_matrix(x))
            }
            # Recursively process nested elements
            return(unlist(lapply(x, process_element)))
          }
          return(as.numeric(x))
        }
        
        ## Parse JSON
        data <- jsonlite::fromJSON(json_string)
        
        ## Process each top-level element in order of appearance
        result <- c()
        for(name in names(data)) {
          result <- c(result, process_element(data[[name]]))
        }
        
        return(result)
  
}









#' convert_JSON_string_to_ordered_R_vector
#' @export
convert_JSON_string_to_ordered_R_vector <- function(json_string, 
                                                    param_names_main_correct_order) {
  
        require(jsonlite)
        
        ## Helper function to flatten a matrix/array row by row
        flatten_matrix <- function(m) {
          if(is.null(dim(m))) {
            return(as.numeric(m))
          }
          return(as.numeric(t(matrix(unlist(m), ncol=ncol(m), byrow=TRUE))))
        }
        
        ## Helper function to recursively process nested structures:
        process_element <- function(x) {
          if(is.numeric(x) && length(x) == 1) {
            return(x)
          }
          if(is.list(x) || is.matrix(x) || is.array(x)) {
            if(is.matrix(x) || (is.list(x) && all(sapply(x, length) == length(x[[1]])))) {
              return(flatten_matrix(x))
            }
            # Recursively process nested elements
            return(unlist(lapply(x, process_element)))
          }
          return(as.numeric(x))
        }
        
        ## Parse JSON
        data <- jsonlite::fromJSON(json_string)
        
        ## Extract parameter root names (without indices) from the reference list
        root_names <- unique(sapply(strsplit(param_names_main_correct_order, "\\[|_"), function(x) x[1]))
        
        ## Process each parameter in the order specified by param_names_main_correct_order
        result <- numeric()
        
        for (root_name in root_names) {
          ## Find matching parameters in the JSON data
          matching_params <- grep(paste0("^", root_name), names(data), value = TRUE)
          
          ## Skip if no matching parameters found
          if(length(matching_params) == 0) {
            warning(paste("Parameter", root_name, "not found in JSON data"))
            next
          }
          
          ## Process each matching parameter
          for(param in matching_params) {
            values <- process_element(data[[param]])
            result <- c(result, values)
          }
        }
        
        ## Check if the length of the result matches the expected length
        ## This is a basic validation assuming the JSON data contains all expected parameters
        if(length(result) != length(unlist(lapply(param_names_main_correct_order, function(name) {
          param_matches <- grep(name, names(data), fixed = TRUE)
          if(length(param_matches) > 0) {
            return(data[param_matches])
          } else {
            return(NULL)
          }
        })))) {
          warning("The length of the ordered vector may not match the expected parameters")
        }
        
        return(result)
  
}














