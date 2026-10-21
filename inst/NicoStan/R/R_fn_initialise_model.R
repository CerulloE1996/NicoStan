

# 
# json$n_thr_per_ord_test
# json$n_cat_per_ord_test
# json$n_ordinal_tests
# 
# 


## R_fn_initialise_model.R               
                       
                       
#' initialise_model
#' @export
initialise_model <- function( Model_type,
                              ##
                              stream = NULL,
                              ##
                              sample_nuisance = NULL,
                              n_nuisance_override = NULL,
                              nuisance_params = NULL,
                              names_of_the_nuisance_blocks_of_a_regular_library = NULL,
                              ##
                              model_args_list = NULL, # For internal/hard-coded models 
                              ##
                              Stan_data_list = NULL, ## for user-supplied Stan models
                              ##
                              compile = TRUE,
                              force_recompile = FALSE,
                              ##
                              cmdstanr_model_fit_obj = NULL,
                              ##
                              Stan_model_file_path = NULL,
                              Stan_cpp_user_header = NULL,
                              Stan_cpp_flags = NULL,
                              stanc_args = NULL,
                              make_args = NULL,
                              ##
                              Stan_model_math_functions = "Stan_math",
                              BayesMVP_AVX_headers_folder = NULL
) {
  
        if (!Model_type %in% .nicostan_model_types) {
            stop("This model provider supports: ", paste(.nicostan_model_types, collapse = ", "), ".")
        }
        ##
        ## Generate stream ID if not provided
        ##
        if (is.null(stream)) {
          stream <- sample.int(1e6, 1)  # Random ID to avoid collisions
        }
        ##
        hard_coded_models_vec <- c("LC_MVP", 
                                   "MVP",
                                   "latent_trait",
                                   "LC_MVOP", 
                                   "MVOP")
        ##
        models_vec <- c("Stan", hard_coded_models_vec)
        ##
        if (!(Model_type %in% models_vec)) { 
          stop("Model_type must be set and be one of the following: 'Stan' 'LC_MVP', 'MVP', 'latent_trait', 'LC_MVOP', 'MVOP'")
        }
        ##
        y <- model_args_list$y
        if (Model_type  %in% hard_coded_models_vec) {
          if (is.null(y)) { 
            stop("input data y (inside 'model_args_list') is needed if using a built-in/hard-coded model
                 (i.e. for Model_type = 'LC_MVP', 'MVP', 'latent_trait', 'LC_MVOP', 'MVOP')")
          }
          N <- model_args_list$N
          if (is.null(N)) { 
            warning("N not inputted into 'model_args_list' - assuming N is the number of rows of the data (y)")
            N <- nrow(y)
            # print(paste("N = ", N))
          }
        }
        
        if (Model_type != "Stan") {
          if ((is.null(model_args_list$X))) { 
            warning("Assuming intercept-only model - supply X for a model with covariates")
          }
        }
        
        if (Model_type == "Stan") {
          
                if (is.null(Stan_model_file_path)) { 
                  stop("Stan_model_file_path must be supplied if Model_type is 'Stan'")
                }
                if (is.null(Stan_data_list)) { 
                  stop("Stan_data_list must be supplied if Model_type is 'Stan'")
                }
                Stan_model_name <- NULL ## not needed for external Stan models
              
        } else {
          
               ##
               ## ---- Get "Stan_model_name" for internal/hard-coded models:
               ##
               if (Model_type == "LC_MVP") { 
                   Stan_model_name <- "LC_MVP_bin_cpp_skeleton.stan"
               } else if (Model_type == "MVP") { 
                   Stan_model_name <- "MVP_bin_cpp_skeleton.stan"
               } else if (Model_type == "latent_trait") { 
                   Stan_model_name <- "latent_trait_bin_cpp_skeleton_10.stan"
               } else if (Model_type == "LC_MVOP") {
                   Stan_model_name <- "LC_MVOP_cpp_skeleton.stan"
               } else if (Model_type == "MVOP") {
                   Stan_model_name <- "MVOP_cpp_skeleton.stan"
               }
          
        }
        ##
        ## ---- Process internal models:
        ##
        if (Model_type != "Stan") {
                ##
                ## ---- Get final "model_args_list" and "Model_args_as_Rcpp_List" for internal/hard-coded models:
                ##
                outs <- init_hard_coded_model( Model_type = Model_type,
                                               model_args_list = model_args_list)
                ##
                model_args_list <- outs$model_args_list
                Model_args_as_Rcpp_List = outs$Model_args_as_Rcpp_List
                ##
                ## ---- Get "Stan_data_list" for internal/hard-coded models:
                ##
                Stan_data_list <- make_Stan_data_list_for_internal_models( Model_type = Model_type,
                                                                           model_args_list = model_args_list)
        }
        ##
        ## ---- nico_reduce_sum (nuisance_params = c("z", ...), the names of the model's nuisance blocks):
##
        ##      NicoStan writes the version of the Stan model whose nuisance blocks it holds inside the chunks of
        ##      its reduce_sum (R_fn_write_nico_reduce_sum_version_of_Stan_model.R), and its user header; the
        ##      version is compiled against NicoStan's private runtime with the chunk-local bookkeeping and the
        ##      nuisance block (options NicoStan.reduce_sum_lean_slicing and NicoStan.reduce_sum_nuisance_block,
        ##      set here for the session), and the size of its nuisance block is worked out from the data (or
        ##      given with n_nuisance_override). The AVX writer below, when asked for, then works on this version:
        ##
        ## (the writer's record of the nuisance blocks held by NicoStan, which gives their initial values their
        ##  order; NULL when there are none)
        nuisance_blocks_held_by_NicoStan <-  NULL
        if (Model_type == "Stan" && is.character(nuisance_params) && length(nuisance_params) == 0 &&
            is.null(names_of_the_nuisance_blocks_of_a_regular_library)) {
              names_of_the_nuisance_blocks_of_a_regular_library <-  character(0)
        }
        if (!is.null(nuisance_params) && (Model_type != "Stan" || !is.character(nuisance_params) ||
                                       length(nuisance_params) > 0)) {
              ##
              if (Model_type != "Stan") {
                    stop("nuisance_params is only for Model_type = \"Stan\".")
              }
              bridgestan_folder <-  utils::getFromNamespace("get_bridgestan_path", "bridgestan")(download = FALSE)
              stanc_file_path <-  file.path(bridgestan_folder, "bin", "stanc")
              if (!file.exists(stanc_file_path)) {
                    stanc_file_path <-  file.path(cmdstanr::cmdstan_path(), "bin", "stanc")
              }
              stanc_args_as_text <-  as.character(unlist(stanc_args, use.names = FALSE))
              stanc_include_paths_arguments <-  stanc_args_as_text[startsWith(x = stanc_args_as_text,
                                                                              prefix = "--include-paths")]
              nico_reduce_sum_version <-  R_fn_write_nico_reduce_sum_version_of_Stan_model(
                                                   Stan_model_file_path           = Stan_model_file_path,
                                                   nuisance_params                = nuisance_params,
                                                   stanc_file_path                = stanc_file_path,
                                                   stanc_include_paths_arguments  = stanc_include_paths_arguments,
                                                   Stan_cpp_user_header           = Stan_cpp_user_header)
              Stan_model_file_path <-  nico_reduce_sum_version$Stan_model_file_path
              Stan_cpp_user_header <-  nico_reduce_sum_version$Stan_cpp_user_header
              nuisance_blocks_held_by_NicoStan <-  nico_reduce_sum_version$nuisance_blocks
              options(NicoStan.reduce_sum_lean_slicing = TRUE, NicoStan.reduce_sum_nuisance_block = TRUE)
              ##
              ## (the size of the nuisance block: each block's number of terms times its values per term, both
              ##  Stan expressions of the data, worked out in R from Stan_data_list)
              # if (is.null(n_nuisance_override)) {
              #       n_nuisance_override <-  fn_size_of_the_nuisance_blocks_held_by_NicoStan_from_the_data(
              #                                     nuisance_blocks  = nico_reduce_sum_version$nuisance_blocks,
              #                                     Stan_data_list   = Stan_data_list)
              # }
              ## (a block whose size the writer could not write in terms of the data, e.g. a size from a flag set
              ##  in a loop of the transformed data: the size is read from the compiled model in get_model_info())
              size_read_from_the_compiled_model <-  isTRUE(any(
                    nico_reduce_sum_version$nuisance_blocks$size_read_from_the_compiled_model))
              if (is.null(n_nuisance_override) && !size_read_from_the_compiled_model) {
                    n_nuisance_override <-  fn_size_of_the_nuisance_blocks_held_by_NicoStan_from_the_data(
                                                  nuisance_blocks  = nico_reduce_sum_version$nuisance_blocks,
                                                  Stan_data_list   = Stan_data_list)
              }
              message(colourise(paste0("nico_reduce_sum: the nuisance blocks ", paste(nuisance_params,
                                                                                         collapse = ", "),
                                       " (", if (is.null(n_nuisance_override)) "their number of values read from "
                                       else n_nuisance_override,
                                       if (is.null(n_nuisance_override)) "the compiled model" else " values",
                                       ") are held by NicoStan inside the chunks: ", Stan_model_file_path),
                                "cyan"))
              ##
        }
        ##
        ## ---- The math functions of a Stan model (Stan_model_math_functions): "Stan_math" (the default)
        ##      compiles the Stan model as it is. "NicoSIMD_AVX2", "NicoSIMD_AVX512" or
        ##      "NicoSIMD_AVX512_if_supported_else_AVX2" (NicoSIMD's nico_ functions), or "BayesMVP_AVX2",
        ##      "BayesMVP_AVX512" or "BayesMVP_AVX512_if_supported_else_AVX2" (BayesMVP's fast_ functions: the
        ##      same C++), write the AVX version of the Stan model next to it
        ##      (every call of a Stan function which BayesMVP replaces, for the argument types it supports,
        ##      becomes BayesMVP's AVX function) and its user header, and the AVX version is then compiled, with
        ##      the instruction-set make arguments, and used as the Stan model, as any Stan model
        ##      (R_fn_Stan_model_math_functions_BayesMVP_AVX.R). The SIMD lanes of the compiled AVX version are
        ##      checked below, once its BridgeStan model exists:
        ##
        BayesMVP_AVX_version_of_Stan_model <-  NULL
        ##
        if (!identical(Stan_model_math_functions, "Stan_math")) {
              ##
              if (Model_type != "Stan") {
                    stop(paste0("Stan_model_math_functions = \"",
                                paste(Stan_model_math_functions, collapse = ", "),
                                "\" is only for Model_type = \"Stan\" (the built-in models set their SIMD ",
                                "functions with model_args_list$vect_type)."))
              }
              ##
              BayesMVP_AVX_version_of_Stan_model <-
                    R_fn_write_BayesMVP_AVX_version_of_Stan_model_and_user_header(
                          Stan_model_math_functions    = Stan_model_math_functions,
                          Stan_model_file_path         = Stan_model_file_path,
                          Stan_cpp_user_header         = Stan_cpp_user_header,
                          stanc_args                   = stanc_args,
                          make_args                    = make_args,
                          BayesMVP_AVX_headers_folder  = BayesMVP_AVX_headers_folder)
              ##
              Stan_model_file_path <-  BayesMVP_AVX_version_of_Stan_model$Stan_model_file_path
              Stan_cpp_user_header <-  BayesMVP_AVX_version_of_Stan_model$Stan_cpp_user_header
              make_args <-  BayesMVP_AVX_version_of_Stan_model$make_args
              ##
        }
        ##
        ## ---- A user C++ header (external Stan functions, e.g. the BayesMVP AVX kernels exposed
        ## to Stan) is honoured by BridgeStan through TWO things: stanc needs --allow-undefined
        ## (the functions are declared without bodies) and make needs USER_HEADER=<path>
        ## (BridgeStan's Makefile then adds -include <path>). Stan_cpp_user_header used to be
        ## stored on the object but never reached the compile call, so it was silently ignored.
        ##
        if (!is.null(Stan_cpp_user_header)) {
              ##
              if (!file.exists(Stan_cpp_user_header))
                    stop("Stan_cpp_user_header does not exist: ", Stan_cpp_user_header)
              ##
              stanc_args <-  unique(c( as.list(stanc_args),
                                      "--allow-undefined"))
              ##
              # make_args <- c( as.list(make_args),
              #                 paste0( "USER_HEADER=",
              #                         normalizePath(Stan_cpp_user_header)))
              ##
              ## ---- exactly one USER_HEADER: make_args taken from an init_object (e.g. when $sample() rebuilds
              ##      the model, as in the fresh R process) already hold it, and a second copy stopped the compile
              ##      ("Multiple USER_HEADER arguments are ambiguous"). A USER_HEADER for another file is an
              ##      error:
              ##
              make_args <-  as.list(as.character(unlist(make_args)))
              make_arg_is_user_header <-  grepl(pattern = "^USER_HEADER=", x = unlist(make_args))
              user_header_files_given <-  normalizePath(sub(pattern = "^USER_HEADER=", replacement = "",
                                                           x = unlist(make_args)[make_arg_is_user_header]),
                                                       mustWork = FALSE)
              if (any(user_header_files_given != normalizePath(Stan_cpp_user_header))) {
                    stop("make_args hold USER_HEADER=", paste(user_header_files_given, collapse = ", "),
                         ", which differs from Stan_cpp_user_header = ", normalizePath(Stan_cpp_user_header),
                         "; give the user header in one of the two only.")
              }
              make_args <-  c( make_args[!make_arg_is_user_header],
                              paste0( "USER_HEADER=",
                                      normalizePath(Stan_cpp_user_header)))
        }
        ##
        ## ---- Compile using BridgeStan (NEED to have "Stan_data_list" to do this - even for internal models!):
        ##
        outs_model_info <- get_model_info(  Model_type = Model_type,
                                            compile = compile,
                                            force_recompile = force_recompile,
                                            ##
                                            stream = stream,
                                            ##
                                            Stan_model_name = Stan_model_name,
                                            ##
                                            Stan_data_list = Stan_data_list,
                                            Stan_model_file_path = Stan_model_file_path,
                                            stanc_args = stanc_args,
                                            make_args = make_args,
                                            ##
                                            sample_nuisance = sample_nuisance,
                                            n_nuisance_override = n_nuisance_override,
                                            names_of_the_nuisance_blocks_of_a_regular_library =
                                                  names_of_the_nuisance_blocks_of_a_regular_library,
                                            ##
                                            model_args_list = model_args_list)
        ##
        bs_model <- outs_model_info$outs_bs_model$bs_model
        # print(paste("bs_model = "))
        # print(bs_model)
        ##
        ## ---- The SIMD lanes which the compiled AVX version of the Stan model reports (8 for AVX-512, 4 for
        ##      AVX2):
        ##
        if (!is.null(BayesMVP_AVX_version_of_Stan_model)) {
              ##
              AVX_version <-  BayesMVP_AVX_version_of_Stan_model
              ##
              BayesMVP_AVX_version_of_Stan_model$SIMD_lanes_reported_by_compiled_model <-
                    R_fn_check_SIMD_lanes_reported_by_compiled_BayesMVP_AVX_Stan_model(
                          bs_model                        = bs_model,
                          SIMD_lanes_expected             = AVX_version$SIMD_lanes_expected,
                          Stan_model_math_functions_used  = AVX_version$Stan_model_math_functions_used,
                          AVX_Stan_model_file_path        = AVX_version$Stan_model_file_path)
              ##
        }
        ##
        json_file_path <- outs_model_info$outs_bs_model$json_file_path
        json_file_path <- normalizePath(json_file_path)
        # json_file_path <- copy_json_with_worker_id(json_file_path, stream)
        # print(paste("json_file_path = "))  
        # print(json_file_path)
        ##
        model_so_file <- outs_model_info$outs_bs_model$model_so_file
        model_so_file <- normalizePath(model_so_file)
        # model_so_file <- copy_.so_with_worker_id(model_so_file, stream)
        # print(paste("model_so_file = "))
        # print(model_so_file)
        ##
        Stan_model_file_path <- outs_model_info$outs_bs_model$Stan_model_file_path
        Stan_model_file_path <- normalizePath(Stan_model_file_path)
        # print(paste("Stan_model_file_path = "))
        # print(Stan_model_file_path)
        ##
        ##
        if (Model_type == "Stan") { 
            dummy_json_file_path <- NULL
            dummy_model_so_file <- NULL
        } else { 
            dummy_json_file_path <- json_file_path
            ##
            dummy_model_so_file <- model_so_file
        }
        ##
        bs_main_param_names <- outs_model_info$outs_bs_model$bs_model$param_names()
        ## (the unconstrained names, one per coordinate of the sampler: a simplex[K] has K constrained values but
        ##  K - 1 coordinates, so only these names line up with the sampler's main block)
        bs_main_param_unc_names <- outs_model_info$outs_bs_model$bs_model$param_unc_names()
        bs_main_and_tp_param_names <- outs_model_info$outs_bs_model$bs_model$param_names(include_tp = TRUE)
        bs_main_and_tp_and_gq_param_names <- outs_model_info$outs_bs_model$bs_model$param_names(include_tp = TRUE, include_gq = TRUE)
        ##
        stan_main_param_names <- convert_bridgestan_par_names_to_stan(bs_main_param_names)
        stan_main_and_tp_param_names <- convert_bridgestan_par_names_to_stan(bs_main_and_tp_param_names)
        stan_main_and_tp_and_gq_param_names <- convert_bridgestan_par_names_to_stan(bs_main_and_tp_and_gq_param_names)
        ##
        n_nuisance <- outs_model_info$n_nuisance
        ## (the size of a nuisance block held by NicoStan, read from the compiled model: kept as
        ##  n_nuisance_override)
        if (is.null(n_nuisance_override) && !is.null(nuisance_blocks_held_by_NicoStan)) {
              n_nuisance_override <-  n_nuisance
        }
        n_params_main <- outs_model_info$n_params_main
        n_params <- outs_model_info$n_params
        n_nuisance_names_constrained <- outs_model_info$n_nuisance_names_constrained
        nuisance_base_name <- outs_model_info$nuisance_base_name
        ##
        # print(paste("n_params_main = ", n_params_main))
        # print(paste("n_nuisance = ", n_nuisance))
        # print(paste("n_params = ", n_params))
        ##
        ##
        ## ---- Make "Model_args_as_Rcpp_List":
        ##
        if (Model_type %in% hard_coded_models_vec) {
          
              # Already created above
              # Model_args_as_Rcpp_List already set from init_hard_coded_model
          
        } else if (Model_type == "Stan") { 
          
              Model_args_as_Rcpp_List <- list()
              ##
              Model_args_as_Rcpp_List$n_params_main <- n_params_main
              Model_args_as_Rcpp_List$n_nuisance <- n_nuisance
              ##
              Model_args_as_Rcpp_List$N <- 100 ## dummy
              Model_args_as_Rcpp_List$n_tests <- 5 ## dummy
              Model_args_as_Rcpp_List$n_class <- 2 ## dummy
              Model_args_as_Rcpp_List$y <- array(0, dim = c(100, 5)) ## dummy
              ##
              ## the C++ list->struct conversion reads these unconditionally; for
              ## external Stan models they are unused dummies:
              Model_args_as_Rcpp_List$n_binary_tests <- 0L
              Model_args_as_Rcpp_List$n_ordinal_tests <- 0L
              
        }
        ##
        Model_args_as_Rcpp_List$json_file_path <- json_file_path
        Model_args_as_Rcpp_List$model_so_file <- model_so_file
        # ##
        # ## ---- Process initial values:
        # ##
        # outs <- R_fn_init_initial_values( Model_type = Model_type,
        #                                   bs_model = bs_model,
        #                                   ##
        #                                   n_chains_burnin = n_chains_burnin,
        #                                   init_lists_per_chain = init_lists_per_chain,
        #                                   ##
        #                                   sample_nuisance = sample_nuisance,
        #                                   n_nuisance = n_nuisance,
        #                                   n_params_main = n_params_main)
        # ##
        # inits_unconstrained_vec_per_chain <- outs$inits_unconstrained_vec_per_chain
        # theta_us_vectors_all_chains_input_from_R <- outs$theta_us_vectors_all_chains_input_from_R
        # theta_main_vectors_all_chains_input_from_R <- outs$theta_main_vectors_all_chains_input_from_R
        # n_chains_burnin <- outs$n_chains_burnin
        # init_lists_per_chain <- outs$init_lists_per_chain
        
 
        
        return(list(   Model_type = Model_type,
                       ##
                       stream = stream,
                       ##
                       cmdstanr_model_fit_obj = NULL,
                       ##
                       sample_nuisance = sample_nuisance,
                       compile = compile,
                       force_recompile = force_recompile,
                       n_nuisance_override = n_nuisance_override,
                       nuisance_blocks_held_by_NicoStan = nuisance_blocks_held_by_NicoStan,
                       ##
                       n_nuisance = n_nuisance,
                       n_params_main = n_params_main,
                       n_params = n_params,
                       ##
                       n_nuisance_names_constrained = n_nuisance_names_constrained,
                       nuisance_base_name = nuisance_base_name,
                       ##
                       bs_model = bs_model,
                       ##
                       bs_main_param_names = bs_main_param_names,
                       bs_main_param_unc_names = bs_main_param_unc_names,
                       bs_main_and_tp_param_names = bs_main_and_tp_param_names,
                       bs_main_and_tp_and_gq_param_names = bs_main_and_tp_and_gq_param_names,
                       ##
                       stan_main_param_names = stan_main_param_names,
                       stan_main_and_tp_param_names = stan_main_and_tp_param_names,
                       stan_main_and_tp_and_gq_param_names = stan_main_and_tp_and_gq_param_names,
                       ##
                       model_args_list = model_args_list,
                       ##
                       Stan_data_list = Stan_data_list,
                       ##
                       Model_args_as_Rcpp_List = Model_args_as_Rcpp_List,
                       ##
                       Stan_model_file_path = Stan_model_file_path,
                       ##
                       # init_lists_per_chain = init_lists_per_chain,
                       # n_chains_burnin = n_chains_burnin,
                       ##
                       # inits_unconstrained_vec_per_chain = inits_unconstrained_vec_per_chain,
                       # theta_us_vectors_all_chains_input_from_R = theta_us_vectors_all_chains_input_from_R,
                       # theta_main_vectors_all_chains_input_from_R = theta_main_vectors_all_chains_input_from_R, 
                       ##
                       json_file_path = json_file_path,
                       model_so_file = model_so_file,
                       dummy_json_file_path = dummy_json_file_path,
                       dummy_model_so_file = dummy_model_so_file,
                       ##
                       Stan_cpp_user_header = Stan_cpp_user_header,
                       Stan_cpp_flags = Stan_cpp_flags,
                       stanc_args = stanc_args,
                       make_args = make_args,
                       ##
                       ## (NULL when this initialisation compiled the given Stan model as it is; with BayesMVP's
                       ##  AVX functions, the AVX version written and compiled by this initialisation)
                       BayesMVP_AVX_version_of_Stan_model = BayesMVP_AVX_version_of_Stan_model))
  
}






















