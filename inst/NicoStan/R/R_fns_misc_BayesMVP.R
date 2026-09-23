
#' Print a large, hard-to-miss warning banner
#'
#' Used by the 'reorder_cols_MVP' machinery so that refusals / caveats are not
#' lost in the (very verbose) sampler output.
#'
#' @param title single-line headline (printed in CAPS between rules)
#' @param body character vector of additional lines (wrapped, may be empty)
#' @param warn if TRUE, also raise an R warning() with the same headline so the
#'        message survives into warnings() / worker error handling
#' @export
big_warning_banner <- function(title,
                               body = character(0),
                               warn = TRUE) {

        rule <- paste(rep("!", 100), collapse = "")
        ##
        message("\n", rule)
        message("!!!!  ", toupper(title))
        message(rule)
        ##
        for (line in body) {
          message("!!!!  ", line)
        }
        ##
        if (length(body) > 0) {
          message(rule)
        }
        message("")
        ##
        if (isTRUE(warn)) {
          warning(title, call. = FALSE, immediate. = TRUE)
        }
        ##
        invisible(NULL)

}



#' @export
remove_duplicates_by_name <- function(my_list) {
  
      if (is.null(names(my_list))) return(my_list)
      my_list[!duplicated(names(my_list))]
  
}


#' is_NaN_or_Inf_vec
#' @export
is_NaN_or_Inf_vec <- function(x) { 
  
        if ((any(is.infinite(x))) || (any(is.nan(x)))) { 
          return(TRUE)
        } else { 
          return(FALSE)
        }
  
}





#' convert_bridgestan_par_names_to_stan
#' @export
convert_bridgestan_par_names_to_stan <- function(names) {
        
        # Handle the conversion more carefully
        result <- names
        
        # Find first dot and replace with [
        result <- sub("\\.", "[", result)
        
        # Replace all remaining dots with commas
        result <- gsub("\\.", ",", result)
        
        # Add closing bracket if we added an opening one
        needs_bracket <- grepl("\\[", result) & !grepl("\\]", result)
        result[needs_bracket] <- paste0(result[needs_bracket], "]")
        
        return(result)
  
}





#' transform_stan_path
#' @export
transform_stan_path <- function(stan_path) {
  
        # Remove any leading ~/ if present
        path_no_tilde <- sub("^~/", "", stan_path)
        
        # Extract directory and filename
        dir_path <- dirname(path_no_tilde)
        filename <- basename(path_no_tilde)
        
        # Remove .stan extension and add _model.so or _model.dll
        if (.Platform$OS.type == "windows") {
          
          model_name <- sub("\\.stan$", "_model.dll", filename)
          final_path <- file.path(dir_path, model_name)
          
        } else { 
          
          model_name <- sub("\\.stan$", "_model.so", filename)
          final_path <- if (startsWith(x = dir_path, prefix = "/")) {
                            file.path(dir_path, model_name)
                        } else {
                            file.path("/", dir_path, model_name)
                        }
          
        }
        
        # final_path <- file.path("~", dir_path, model_name)
        
        return(final_path)
  
}




#' copy_json_with_worker_id
#' @export
copy_json_with_worker_id <- function(original_path, 
                                     ii) {
  
        # Create new path with worker ID
        base_path <- sub("\\.json$", "", original_path)
        new_path <- paste0(base_path, "_", ii, ".json")
        
        # Copy the file to the new path - ATOMICALLY (temporary file + rename; see fn_write_file_atomically), since
        # data_<md5>_<stream>.json sits in the shared stan_data folder and a concurrent fit with the same data and
        # stream could otherwise read it half-copied:
        if (file.exists(original_path)) {
          fn_copy_file_atomically( source_file_path = original_path,
                                   target_file_path = new_path)
        } else {
          stop(paste("Original file doesn't exist:", original_path))
        }
        
        return(new_path)
  
}



#' copy_.so_with_worker_id
#' @export
copy_.so_with_worker_id <- function(original_path, 
                                    ii) {
  
        # Create new path with worker ID
        base_path <- sub("\\.so$", "", original_path)
        new_path <- paste0(base_path, "_", ii, ".so")
        
        # Copy the file to the new path
        if (file.exists(original_path)) {
          file.copy(original_path, new_path, overwrite = TRUE)
        } else {
          stop(paste("Original file doesn't exist:", original_path))
        }
        
        return(new_path)
  
}



#' copy_.so_with_worker_id
#' @export
copy_.stan_with_worker_id <- function( original_path, 
                                       ii) {
  
        # Create new path with worker ID
        base_path <- sub("\\.stan$", "", original_path)
        new_path <- paste0(base_path, "_", ii, ".stan")
        
        # Copy the file to the new path
        if (file.exists(original_path)) {
          file.copy(original_path, new_path, overwrite = TRUE)
        } else {
          stop(paste("Original file doesn't exist:", original_path))
        }
        
        return(new_path)
  
}



#' if_null_then_set_to
#' @export
if_null_then_set_to <- function(x, 
                                set_to_thif_null, 
                                debugging = FALSE
)  {
  
        if (is.null(x)) {
          y <- set_to_thif_null
          return(y)
        } else { 
          y <- x
          return(y)
        }
        
        # if (debugging) print(paste(y))
  
  
}





#' Detect n_nuisance from Stan model (UNCONSTRAINED coordinates of the FIRST
#' declared parameter block)
#'
#' BayesMVP's convention: the nuisance / high-dimensional latent parameter block is
#' the FIRST declaration in the `parameters` block of the Stan model. Everything
#' after it is "main". The dimension must be counted in UNCONSTRAINED coordinates
#' (the space the sampler integrates over): a 4-category simplex occupies four
#' constrained names but only three unconstrained coordinates, so counting
#' constrained names would overstate the nuisance dimension.
#'
#' The first DECLARED parameter is identified from stanc compiler metadata
#' (`stanc --info`), not from the first returned name: if the first declaration has
#' ZERO length (e.g. an empty array under a switched-off branch) it contributes no
#' names at all and the first returned name belongs to the SECOND declaration -
#' a name-based heuristic would wrongly treat that second block as nuisance.
#' Comments, `#include`-d files and data-dependent dimensions are all handled by
#' stanc / BridgeStan rather than by a handwritten regex parser.
#'
#' `sample_nuisance = FALSE` means the model has NO nuisance block: every declared
#' parameter belongs to the main block and 0 is returned (this is distinct from a
#' nuisance declaration that exists but evaluates to zero length - see
#' detect_nuisance_block_info_from_stan_model()).
#'
#' @param bs_model a BridgeStan StanModel (with its data already loaded)
#' @param sample_nuisance whether a nuisance block exists and should be sampled
#' @param Stan_model_file_path path to the .stan source (used for stanc metadata)
#' @param stanc_args optional stanc arguments (e.g. list(include_paths = ...))
#' @export
detect_n_nuisance_from_stan_model <- function(bs_model, 
                                              sample_nuisance = TRUE,
                                              Stan_model_file_path = NULL,
                                              stanc_args = NULL) {
  
        if (!isTRUE(sample_nuisance)) {
          return(as.integer(0))
        }
        ##
        as.integer(detect_nuisance_block_info_from_stan_model(bs_model = bs_model,
                                                              Stan_model_file_path = Stan_model_file_path,
                                                              stanc_args = stanc_args)$n_nuisance)
  
}



#' Full nuisance-block metadata for an external Stan model
#'
#' Returns the UNCONSTRAINED nuisance dimension (n_nuisance), the number of
#' CONSTRAINED names belonging to the nuisance block (n_nuisance_names_constrained -
#' used to locate the main block inside constrained draws), and the declared base
#' name of the nuisance block (nuisance_base_name). See
#' detect_n_nuisance_from_stan_model() for the conventions.
#' @export
detect_nuisance_block_info_from_stan_model <- function(bs_model,
                                                       Stan_model_file_path = NULL,
                                                       stanc_args = NULL) {
  
        bs_names <- bs_model$param_names()
        bs_unc_names <- bs_model$param_unc_names()
        ##
        if (length(bs_unc_names) == 0) {
          return(list( n_nuisance = as.integer(0),
                       n_nuisance_names_constrained = as.integer(0),
                       nuisance_base_name = NA_character_))
        }
        ##
        fn_param_base <- function(bridgestan_names) {
          
              sub( pattern = "\\..*$",
                   replacement = "",
                   x = bridgestan_names)
              
        }
        ##
        ## ---- Identify the FIRST DECLARED parameter from stanc compiler metadata
        ## (declaration order, with comments / includes / data-dependent dims handled
        ## by the compiler itself):
        ##
        first_declared_param <- NULL
        first_declared_param_used_stanc <- FALSE
        ##
        if (!is.null(Stan_model_file_path) && file.exists(Stan_model_file_path)) {
          
              stanc_bin <- file.path( cmdstanr::cmdstan_path(),
                                      "bin",
                                      "stanc")
              ##
              if (file.exists(stanc_bin)) {
                
                    ## External AVX functions are defined by the C++ user header.
                    ## Declaration metadata must accept those Stan prototypes too.
                    stanc_cmd_args <- c( "--info",
                                         "--allow-undefined",
                                         shQuote(Stan_model_file_path))
                    ##
                    ## Accept both named CmdStanR-style lists and BridgeStan's raw
                    ## character/list flags. Preserve include paths when inspecting metadata.
                    include_paths <- if (is.list(stanc_args)) stanc_args$include_paths else NULL
                    raw_stanc_args <- as.character(unlist(stanc_args, use.names = FALSE))
                    include_flags <- raw_stanc_args[startsWith(raw_stanc_args, "--include-paths=")]
                    if (length(include_flags) > 0L) {
                      include_paths <- c(include_paths, sub("^--include-paths=", "", include_flags))
                    }
                    include_positions <- which(raw_stanc_args == "--include-paths")
                    if (length(include_positions) > 0L) {
                      if (any(include_positions == length(raw_stanc_args))) stop("--include-paths requires a directory.")
                      include_paths <- c(include_paths, raw_stanc_args[include_positions + 1L])
                    }
                    ##
                    if (length(include_paths) > 0L) {
                      stanc_cmd_args <- c(stanc_cmd_args,
                                          shQuote(paste0("--include-paths=", paste(unique(include_paths), collapse = ","))))
                    }
                    ##
                    stanc_out <- tryCatch( expr = system2( command = stanc_bin,
                                                           args = stanc_cmd_args,
                                                           stdout = TRUE,
                                                           stderr = TRUE),
                                           error = function(e) NULL)
                    ##
                    stanc_json_text <- paste( stanc_out,
                                              collapse = "\n")
                    ##
                    if (!is.null(stanc_out) && length(stanc_out) > 0 &&
                        grepl( pattern = "\"parameters\"",
                               x = stanc_json_text,
                               fixed = TRUE)) {
                      
                          stanc_info <- jsonlite::fromJSON( txt = stanc_json_text,
                                                            simplifyVector = FALSE)
                          ##
                          if (!is.null(stanc_info$parameters) && length(stanc_info$parameters) > 0) {
                            first_declared_param <- names(stanc_info$parameters)[1]
                            first_declared_param_used_stanc <- TRUE
                          }
                          
                    }
              }
          
        }
        ##
        if (is.null(first_declared_param)) {
          ##
          ## fallback: first UNCONSTRAINED name (cannot distinguish a zero-length
          ## first declaration from "no first declaration", so the dimension of the
          ## first NON-EMPTY block is used):
          ##
          first_declared_param <- fn_param_base(bs_unc_names[1])
          ##
          warning( "detect_nuisance_block_info_from_stan_model: stanc metadata unavailable; ",
                   "the first declared parameter was inferred from the first unconstrained name ",
                   "('", first_declared_param, "'). A ZERO-LENGTH first declaration cannot be ",
                   "detected this way - supply a working cmdstan installation (stanc --info) or ",
                   "use n_nuisance_override.")
          
        }
        ##
        ## ---- count the coordinates belonging to that declaration:
        ##
        n_nuisance <- sum(fn_param_base(bs_unc_names) == first_declared_param)
        n_nuisance_names_constrained <- sum(fn_param_base(bs_names) == first_declared_param)
        ##
        list( n_nuisance = as.integer(n_nuisance),
              n_nuisance_names_constrained = as.integer(n_nuisance_names_constrained),
              nuisance_base_name = first_declared_param,
              first_declared_param_used_stanc = first_declared_param_used_stanc)
  
}





#' get_n_params_main_from_n_nuisance_using_Stan_bs_model
#' @export
get_n_params_main_from_n_nuisance_using_Stan_bs_model <- function(  n_nuisance, 
                                                                 Stan_data_list,
                                                                 Stan_model_file_path,
                                                                 stanc_args = NULL
) { 
  
  
        if (is.null(stanc_args)) { 
          outs <- init_bs_model_external(  Stan_data_list = Stan_data_list,
                                           Stan_model_file_path = Stan_model_file_path)
        } else { 
          outs <- init_bs_model_external(  Stan_data_list = Stan_data_list,
                                          Stan_model_file_path = Stan_model_file_path,
                                          stanc_args = stanc_args)
        }
        ##
        bs_model <- outs$bs_model
        ##
        bs_names  <- bs_model$param_names() ## bookmark_2 (previous code)
        bs_names_inc_tp <-  (bs_model$param_names(include_tp = TRUE))
        bs_names_inc_tp_and_gq <-  (bs_model$param_names(include_tp = TRUE, include_gq = TRUE))
        ##
        n_params <- length(bs_names)
        n_params_main <- n_params - n_nuisance
        ##
        return(list(n_params = n_params,
                    n_params_main = n_params_main,
                    n_nuisance = n_nuisance,
                    ##
                    bs_model = bs_model,
                    bs_names = bs_names,
                    bs_names_inc_tp = bs_names_inc_tp,
                    bs_names_inc_tp_and_gq = bs_names_inc_tp_and_gq))
  
}





#' check_stan_data
#' @export
check_stan_data <- function(stan_model,
                            stan_data_list
) {
  
        # Get the data variables from the Stan model
        vars <- stan_model$variables()
        data_vars <- vars$data
        
        # Extract variable names from the data block
        required_vars <- names(data_vars)
        
        # Check which variables are in the stan_data_list
        provided_vars <- names(stan_data_list)
        
        # Find missing variables
        missing_vars <- setdiff(required_vars, provided_vars)
        
        # Find extra variables (not in model but in data list)
        extra_vars <- setdiff(provided_vars, required_vars)
        
        # Report results
        if (length(missing_vars) > 0) {
          missing_details <- character(length(missing_vars))
          
          # Add details about each missing variable
          for (i in seq_along(missing_vars)) {
            var_name <- missing_vars[i]
            var_info <- data_vars[[var_name]]
            var_type <- var_info$type
            var_dims <- var_info$dimensions
            
            # Format dimension info
            if (length(var_dims) == 0 || all(var_dims == 0)) {
              dim_str <- "scalar"
            } else {
              dim_str <- paste0(var_dims, "D array/matrix")
            }
            
            missing_details[i] <- sprintf("  - %s (type: %s, dimensions: %s)", 
                                          var_name, var_type, dim_str)
          }
          
          error_msg <- paste0(
            "Missing required Stan data variables:\n",
            paste(missing_details, collapse = "\n"),
            "\n\nPlease add these variables to your Stan_data_list."
          )
          
          stop(error_msg, call. = FALSE)
        }
        
        # Optional: warn about extra variables
        if (length(extra_vars) > 0) {
          warning(sprintf(
            "The following variables in Stan_data_list are not in the model's data block:\n  %s\n",
            paste(extra_vars, collapse = ", ")
          ), call. = FALSE)
        }
        
        # If we get here, all required variables are present
        message("✓ All required Stan data variables are present!")
        
        # Return summary
        invisible(list(
          required = required_vars,
          provided = provided_vars,
          missing = missing_vars,
          extra = extra_vars,
          check_passed = length(missing_vars) == 0
        ))
        
}










#' Get model dimensions (n_params, n_params_main, n_nuisance)
#' @export
get_model_info <- function(  Model_type, 
                             ##
                             stream,
                             ##
                             Stan_model_name = NULL,
                             ##
                             Stan_data_list = NULL,
                             Stan_model_file_path = NULL,
                             stanc_args = NULL,
                             make_args = NULL,
                             ##
                             sample_nuisance = NULL,
                             n_nuisance_override = NULL,
                             ##
                             model_args_list
) {
  
        n_tests <- NULL
        n_class <- NULL
        N <- NULL
        n_covariates_total <- NULL
        ##
        n_cat_per_ord_test <- NULL
        n_thr_per_ord_test <- NULL
        
        if (Model_type == "Stan") {
                  
               outs_bs_model <- init_bs_model_external(   stream = stream,
                                                          Stan_data_list = Stan_data_list,
                                                          Stan_model_file_path = Stan_model_file_path,
                                                          stanc_args = stanc_args,
                                                          make_args = make_args)
          
                bs_model <- outs_bs_model$bs_model
                
                # Total number of parameters is the UNCONSTRAINED dimension - that is
                # what the sampler integrates over. Counting constrained names is wrong
                # whenever a parameter is constrained (a 4-category simplex has 4
                # constrained names but only 3 unconstrained coordinates).
                n_params <- bs_model$param_unc_num()
                
                nuisance_info <- NULL
                
                # Detect or use provided n_nuisance
                if (!is.null(n_nuisance_override)) {
                  n_nuisance <- n_nuisance_override
                  ## constrained-name count of the nuisance block cannot be inferred
                  ## from an override alone; fall back to the unc count (exact whenever
                  ## the nuisance block has no dimension-changing constraints):
                  n_nuisance_names_constrained <- n_nuisance
                } else {
                  nuisance_info <- detect_nuisance_block_info_from_stan_model( bs_model = bs_model,
                                                                               Stan_model_file_path = Stan_model_file_path,
                                                                               stanc_args = stanc_args)
                  if (!isTRUE(sample_nuisance)) {
                    ## ---- Models WITHOUT a nuisance block: every declared parameter is
                    ## main. (distinct from sample_nuisance = TRUE with a zero-length
                    ## first declaration, which detect_... resolves to 0 by itself)
                    n_nuisance <- 0L
                    n_nuisance_names_constrained <- 0L
                  } else {
                    n_nuisance <- nuisance_info$n_nuisance
                    n_nuisance_names_constrained <- nuisance_info$n_nuisance_names_constrained
                  }
                }
                
                if (n_nuisance < 0 || n_nuisance > n_params) {
                  stop( "get_model_info (Stan): n_nuisance = ", n_nuisance,
                        " is outside [0, n_params = ", n_params, "]. ",
                        "Check n_nuisance_override and the declaration order of the parameters block ",
                        "(the nuisance block must be the FIRST declaration).")
                }
                
                n_params_main <- n_params - n_nuisance
          
        } else {
          
                outs_bs_model <- init_bs_model_internal(   stream = stream,
                                                           Stan_data_list = Stan_data_list,
                                                           Stan_model_name = Stan_model_name)
                # ##
#                 # # make_Stan_data_list_for_internal_models(Model_type = "LC_MVP", model_args_list = list(y = y))
#                 # Stan_data_list <- make_Stan_data_list_for_internal_models( Model_type = "LC_MVOP",
#                 #                                                            model_args_list = model_args_list)
#                 #                                                            # model_args_list = list(y = y))
#                 ##
#                 Stan_data_list <- make_Stan_data_list_for_internal_models( Model_type = "LC_MVP",
#                                                                            model_args_list = model_args_list)
#                 #
#                 Stan_data_list$overflow_threshold  <- +5
#                 Stan_data_list$underflow_threshold <- -5
#                 ##
#                 Stan_data_list$prior_only <- 0
#                 Stan_data_list$Phi_type <- 1
#                 ##
#                 Stan_data_list$pop
#                 Stan_data_list$n_pops
#                 ##
#                 model_args_list$prior_prev_a
#                 model_args_list$prior_prev_b
#                 ##
#                 Stan_data_list$prior_prev_a
#                 Stan_data_list$prior_prev_b
#                 ## 
#                 Stan_data_list$baseline_case_nd <- baseline_case_nd
#                 Stan_data_list$baseline_case_d <- baseline_case_d
#                 ##
#                 # library(cmdstanr)
#                 # mod <- cmdstan_model(system.file("stan_models/LC_MVP_cpp_skeleton.stan", package = "BayesMVP"))
#                 # mod <- cmdstan_model(system.file("stan_models/LC_MVOP_cpp_skeleton.stan", package = "BayesMVP"))
#                 # mod <- cmdstan_model(system.file("stan_models/LC_MVP_bin_cpp_skeleton.stan", package = "BayesMVP"))
#                 mod <- cmdstan_model(system.file("stan_models/LC_MVP_bin_cpp_skeleton.stan", package = "BayesMVP"))
#                 ##
#                 fit <- mod$sample(
#                   data = Stan_data_list,  # your R list before JSON conversion
#                   chains = 1,
#                   iter_warmup = 10,
#                   iter_sampling = 10,
#                   seed = 123
#                 )
#                 # # #
#                 # # Stan_data_list
# #                 ##
#                 ## For hard-coded models:
                ##
                y <- model_args_list$y
                ##
                if (is.null(y)) { 
                  stop("input data y (inside 'model_args_list') is needed if using a built-in/hard-coded model
                 (i.e. for Model_type = 'LC_MVP', 'MVP', 'latent_trait')")
                }
                ##
                n_tests <- ncol(y)
                N <- nrow(y)
                n_class <- ifelse(Model_type %in% c("LC_MVP", "LC_MVOP", "latent_trait"), 2, 1)
                n_pops <- model_args_list$n_pops
                n_pops
                ##
                # Calculate n_covariates_total
                # if (is.null(model_args_list$n_covariates_per_outcome_mat)) {
                #   n_covariates_total <- n_tests  # Intercept-only
                # } else {
                  n_covariates_total <- sum(model_args_list$n_covariates_per_outcome_mat)
                # }
                
                # Calculate dimensions based on model type
                if (Model_type %in% c("MVP")) {
                    n_params_main <- n_covariates_total + choose(n_tests, 2)
                    n_nuisance <- n_tests * N
                } else if (Model_type %in% c("LC_MVP")) {
                    n_params_main <- n_pops + n_covariates_total + 2 * choose(n_tests, 2)
                    n_nuisance <- n_tests * N
                } else if (Model_type == "latent_trait") {
                    n_params_main <- n_pops + n_covariates_total + 2 * n_tests
                    n_nuisance <- n_tests * N
                }
                ##
##
                n_cat_per_ord_test <- 0
                n_thr_per_ord_test <- 0
                ##
                n_corrs <- choose(n_tests, 2)
                ##
                if (Model_type %in% c("LC_MVOP", "MVOP")) {
                  
                      ##
                      ## ---- n_cat_per_ord_test MUST be user-supplied (see get_dims_for_internal_models):
                      ##
                      n_cat_per_ord_test <- model_args_list$n_cat_per_ord_test
                      ##
                      if (is.null(n_cat_per_ord_test)) {
                        stop(paste0("get_model_info: model_args_list$n_cat_per_ord_test MUST be supplied for ",
                                    Model_type, ". It cannot be inferred from y (the top category is often ",
                                    "unobserved, so the observed max undercounts)."))
                      }
                      ##
                      n_cat_per_ord_test <- as.integer(n_cat_per_ord_test)
                      n_thr_per_ord_test <- n_cat_per_ord_test - 1L
                      ##
                      if (Model_type == "LC_MVOP") {
                        n_params_main <- n_pops + n_covariates_total + n_class*n_corrs + n_class*sum(n_thr_per_ord_test)
                      } else {  ## MVOP
                        n_params_main <- n_covariates_total + n_corrs + sum(n_thr_per_ord_test)   ## FIXED: was "n_thr_per_ord_t"
                      }
                      ##
                      n_nuisance <- n_tests * N
                  
                }
                ##
                n_params <- n_params_main + n_nuisance
                
        }

        return(list(
          outs_bs_model = outs_bs_model,
          ##
          n_params = n_params,
          n_params_main = n_params_main,
          n_nuisance = n_nuisance,
          ##
          n_nuisance_names_constrained = if (Model_type == "Stan") n_nuisance_names_constrained else n_nuisance,
          nuisance_base_name = if (Model_type == "Stan") (if (is.null(nuisance_info)) NA_character_ else nuisance_info$nuisance_base_name) else NA_character_,
          ##
          N = N,
          n_class = n_class,
          n_tests = n_tests,
          n_covariates_total = n_covariates_total,
          ##
          n_cat_per_ord_test = n_cat_per_ord_test,
          n_thr_per_ord_test = n_thr_per_ord_test
        ))
  
}


#' Check and validate Model_args_as_Rcpp_List
#' @export
validate_rcpp_list <- function(Model_args_as_Rcpp_List, 
                               Model_type) {
        
        required_fields <- c(
          "N", "n_nuisance", "n_params_main",
          "model_so_file", "json_file_path"
        )
        
        if (Model_type %in% c("MVP", "LC_MVP", "latent_trait", "MVOP", "LC_MVOP")) {
          required_fields <- c(required_fields,
                               "Model_args_bools", "Model_args_ints", "Model_args_doubles",
                               "Model_args_strings", "Model_args_mats_double", "Model_args_mats_int"
          )
        }
        
        missing <- setdiff(required_fields, names(Model_args_as_Rcpp_List))
        
        if (length(missing) > 0) {
          stop("Missing required fields in Model_args_as_Rcpp_List: ", 
               paste(missing, collapse = ", "))
        }
        
        # Check types
        if (!is.numeric(Model_args_as_Rcpp_List$N)) {
          stop("N must be numeric")
        }
        
        if (!is.numeric(Model_args_as_Rcpp_List$n_nuisance)) {
          stop("n_nuisance must be numeric")
        }
        
        if (!is.numeric(Model_args_as_Rcpp_List$n_params_main)) {
          stop("n_params_main must be numeric")
        }
        
        return(TRUE)
  
}



#' Debug helper to inspect model structure
#' @export
inspect_model <- function(model_obj) {
        
        cat("Model Type:", model_obj$Model_type, "\n")
        cat("N:", model_obj$init_object$N, "\n")
        cat("n_params:", model_obj$init_object$n_params, "\n")
        cat("n_params_main:", model_obj$init_object$n_params_main, "\n")
        cat("n_nuisance:", model_obj$init_object$n_nuisance, "\n")
        cat("sample_nuisance:", model_obj$init_object$sample_nuisance, "\n")
        
        if (model_obj$Model_type == "Stan") {
          
              cat("\nStan model info:\n")
              cat("  Model file:", model_obj$init_object$Stan_model_file_path, "\n")
              cat("  JSON file:", model_obj$init_object$json_file_path, "\n")
              cat("  SO file:", model_obj$init_object$model_so_file, "\n")
              
              if (!is.null(model_obj$init_object$bs_model)) {
                param_names <- model_obj$init_object$bs_model$param_names()
                cat("  Total parameters:", length(param_names), "\n")
                cat("  First 5 params:", paste(head(param_names, 5), collapse = ", "), "\n")
              }
          
        } else {
              
              cat("\nHard-coded model info:\n")
              cat("  n_tests:", model_obj$init_object$model_args_list$n_tests, "\n")
              cat("  n_class:", model_obj$init_object$model_args_list$n_class, "\n")
              cat("  n_covariates_total:", model_obj$init_object$model_args_list$n_covariates_total, "\n")
          
        }
        
        # Check Rcpp list
        if (!is.null(model_obj$init_object$Model_args_as_Rcpp_List)) {
          cat("\nModel_args_as_Rcpp_List fields:\n")
          cat("  ", paste(names(model_obj$init_object$Model_args_as_Rcpp_List), collapse = ", "), "\n")
        }
  
}




#' compute_equivalent_ESS
#' @export
compute_equivalent_ESS <- function(N, 
                                   N_new, 
                                   ESS_observed, 
                                   scaling_powers = c(0.33, 0.5, 1.0)
) {
  
        # For each scaling assumption, compute equivalent ESS
        # posterior_SD ∝ 1/N^power
        # So SD_new/SD_old = (N/N_new)^power
        
        equivalent_ESS <- numeric(length(scaling_powers))
        
        for (i in seq_along(scaling_powers)) {
          power <- scaling_powers[i]
          
          # Ratio of posterior SDs
          SD_ratio <- (N / N_new)^power
          
          # For equal MC error: ESS_new = ESS_old * SD_ratio^2
          equivalent_ESS[i] <- ESS_observed * SD_ratio^2
        }
        
        # Create nice output
        cat(sprintf("Original: N = %d studies, ESS = %.0f\n", N, ESS_observed))
        cat(sprintf("New:      N = %d studies\n", N_new))
        cat("\nEquivalent ESS for same MC error:\n")
        cat(sprintf("  Conservative (SD ∝ 1/N^0.33): ESS ≈ %.0f\n", equivalent_ESS[1]))
        cat(sprintf("  Typical      (SD ∝ 1/N^0.50): ESS ≈ %.0f\n", equivalent_ESS[2]))
        cat(sprintf("  Optimistic   (SD ∝ 1/N^1.00): ESS ≈ %.0f\n", equivalent_ESS[3]))
        cat(sprintf("\nRange: %.0f - %.0f\n", min(equivalent_ESS), max(equivalent_ESS)))
        
        # Return invisibly for programmatic use
        invisible(list(
          N = N,
          N_new = N_new,
          ESS_observed = ESS_observed,
          equivalent_ESS = equivalent_ESS,
          range = c(min = min(equivalent_ESS), max = max(equivalent_ESS))
        ))
  
}



#' validate_and_extrapolate_ESS
#' @export
validate_and_extrapolate_ESS <- function(N1, 
                                         ESS1,
                                         N2, 
                                         ESS2,
                                         N_target
) {
        # Estimate scaling power
        p <- log(ESS2/ESS1) / (2*log(N1/N2))
        
        # Extrapolate
        ESS_target <- ESS1 * (N1/N_target)^(2*p)
        
        cat(sprintf("Observed scaling: SD ∝ 1/N^%.2f\n", p))
        cat(sprintf("Target ESS for N=%d: %.0f\n", N_target, ESS_target))
        
        # Add safety margin
        cat(sprintf("With 20%% safety margin: %.0f\n", ESS_target * 0.8))
        
        return(list(p = p,
                    ESS_target = ESS_target))
  
}

# If SD ∝ 1/N^p, then from N=500 to N=2500:
# ESS should scale by (500/2500)^(2p) = (1/5)^(2p)
##
# So if you observe ESS_500 = X and ESS_2500 = Y:
# p = log(Y/X) / (2*log(5))
##
# Then for N=10,000:
# ESS_10000 = ESS_500 * (500/10000)^(2p)
##
# For N=25,000:
# ESS_25000 = ESS_500 * (500/25000)^(2p)



#' resize_init_list
#' @export
resize_init_list <- function(init_lists_per_chain, 
                             n_chains_new
) {
        
        n_chains_old <- length(init_lists_per_chain)
        
        if (n_chains_new == n_chains_old) {
          # No change needed
          return(init_lists_per_chain)
        } else if (n_chains_new < n_chains_old) {
          # Fewer chains - just take first n_chains_new
          return(init_lists_per_chain[1:n_chains_new])
        } else {
          # More chains - repeat cyclically
          indices <- rep(1:n_chains_old, length.out = n_chains_new)
          return(init_lists_per_chain[indices])
        }
  
}


#' @export
permute_3d_draws_array <- function(array,
                                   iter_index = 1,
                                   chain_index = 2,
                                   param_index = 3,
                                   new_iter_index = 2,
                                   new_chain_index = 3,
                                   new_param_index = 1
) {
  
        # Create the permutation vector
        # We need to map old positions to new positions
        perm_vector <- numeric(3)
        
        # Find which old index goes to position 1
        if (new_iter_index == 1) perm_vector[1] <- iter_index
        else if (new_chain_index == 1) perm_vector[1] <- chain_index
        else if (new_param_index == 1) perm_vector[1] <- param_index
        
        # Find which old index goes to position 2
        if (new_iter_index == 2) perm_vector[2] <- iter_index
        else if (new_chain_index == 2) perm_vector[2] <- chain_index
        else if (new_param_index == 2) perm_vector[2] <- param_index
        
        # Find which old index goes to position 3
        if (new_iter_index == 3) perm_vector[3] <- iter_index
        else if (new_chain_index == 3) perm_vector[3] <- chain_index
        else if (new_param_index == 3) perm_vector[3] <- param_index
        
        # Apply the permutation
        new_array <- aperm(array, perm = perm_vector)
        
        # Update dimension names if they exist
        if (!is.null(dimnames(array))) {
          old_names <- dimnames(array)
          new_names <- list(
            old_names[[perm_vector[1]]],
            old_names[[perm_vector[2]]],
            old_names[[perm_vector[3]]]
          )
          
          # Rename based on new positions
          names(new_names) <- c(
            if (new_iter_index == 1) "iteration" else if (new_chain_index == 1) "chain" else "parameter",
            if (new_iter_index == 2) "iteration" else if (new_chain_index == 2) "chain" else "parameter",
            if (new_iter_index == 3) "iteration" else if (new_chain_index == 3) "chain" else "parameter"
          )
          
          dimnames(new_array) <- new_names
        }
        
        return(new_array)
  
}










