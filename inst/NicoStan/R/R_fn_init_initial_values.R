
## R_fn_init_initial_values.R

#' R_fn_init_initial_values
#' @keywords internal
#' @export
R_fn_init_initial_values <- function( Model_type,
                                      bs_model,
                                      ##
                                      n_chains_burnin,
                                      init_lists_per_chain,
                                      ##
                                      sample_nuisance,
                                      n_nuisance,
                                      n_params_main,
                                      ##
                                      ## built-in models: list(N, n_tests, n_chunks, vect_type) of the
                                      ## burn-in, the chunk layout of the nuisance vector the C++ reads:
                                      chunk_layout_of_nuisance_vector_in_burnin = NULL,
                                      ##
                                      ## Stan models whose nuisance blocks are held by NicoStan (nuisance_params,
                                      ## nico_reduce_sum): the writer's record of the blocks, and the data,
                                      ## which give each block's size:
                                      nuisance_blocks_held_by_NicoStan = NULL,
                                      Stan_data_list = NULL
) {
  
        if (is.null(init_lists_per_chain)) { 
          stop("initial values per chain (init_lists_per_chain) not supplied")
        }
        if (is.null(n_chains_burnin)) { 
          stop("n_chains_burnin not supplied")
        }
        if (length(init_lists_per_chain) != n_chains_burnin) { 
          stop("n_chains_burnin must be equal to the length of the initial values list ('init_lists_per_chain')")
        }
        theta_nuisance_vectors_all_chains_input_from_R <- NULL
        theta_main_vectors_all_chains_input_from_R <- NULL
        inits_unconstrained_vec_per_chain <- list()
        ##
        for (kk in 1:n_chains_burnin) {
            
            if (Model_type == "Stan") {
              
              ##
              ## ---- Stan models: preserve the COMPLETE external-model initialisation and
              ## unconstrain it, and ONLY THEN split nuisance and main coordinates. No
              ## parameter is removed by name (previously "u_raw" was deleted), so the
              ## nuisance block may be called anything and a nuisance block that exists
              ## but has zero length contributes zero coordinates naturally.
              ##
              ## BridgeStan's param_unconstrain_json() rejects PARTIAL inits, so missing
              ## entries are completed first with the model's own defaults (the
              ## constrained transform of the zero unconstrained vector) and merged with
              ## the supplied values.
              ##
              init_list_kk <- init_lists_per_chain[[kk]]
              ##
              ## ---- a Stan model whose nuisance blocks are held by NicoStan (nuisance_params,
              ##      nico_reduce_sum): the blocks are not parameters of the compiled model, so their initial
              ##      values are taken out of the init list here, and put in front of the main parameters'
              ##      unconstrained values below, as the split into nuisance and main coordinates expects:
              ##
              initial_values_of_the_nuisance_blocks_held_by_NicoStan_kk <-  NULL
              if (!is.null(nuisance_blocks_held_by_NicoStan)) {
                    if (!isTRUE(sample_nuisance)) {
                          stop(paste0("R_fn_init_initial_values: the nuisance blocks of nuisance_params are ",
                                      "sampled by NicoStan, so sample_nuisance must be TRUE."))
                    }
                    nuisance_blocks_and_main_parameters_kk <-
                          fn_initial_values_of_the_nuisance_blocks_held_by_NicoStan_from_the_init_list(
                                nuisance_blocks_held_by_NicoStan  = nuisance_blocks_held_by_NicoStan,
                                init_list                         = init_list_kk,
                                Stan_data_list                    = Stan_data_list)
                    initial_values_of_the_nuisance_blocks_held_by_NicoStan_kk <-
                          nuisance_blocks_and_main_parameters_kk$initial_values_of_the_nuisance_blocks
                    init_list_kk <-  nuisance_blocks_and_main_parameters_kk$init_list_of_the_main_parameters
              }
              ##
              json_string_for_inits_chain_kk <- convert_stan_data_list_to_JSON(init_list_kk)
              validated_json_string <- paste(readLines(json_string_for_inits_chain_kk), collapse="")
              ##
              inits_unconstrained_vec_kk <- tryCatch( expr = {
                bs_model$param_unconstrain_json(validated_json_string)
              }, error = function(error_condition) {

                NULL
              })

              ##
              if (is.null(inits_unconstrained_vec_kk)) {
                
                    ## ---- partial initialisation: complete missing parameters with the
                    ## model defaults and retry.
                    bs_names <- bs_model$param_names()
                    ##
                    default_theta_constrained <- tryCatch( expr = {
                      bs_model$param_constrain( rep( x = 0.0,
                                                     times = bs_model$param_unc_num()),
                                                include_tp = FALSE,
                                                include_gq = FALSE,
                                                rng = bs_model$new_rng(seed = 123))
                    }, error = function(error_condition) {

                      NULL
                    })

                    ##
                    if (is.null(default_theta_constrained)) {
                      stop( "R_fn_init_initial_values: the initial values for chain ", kk,
                            " could not be unconstrained, and model-default initial values could not be ",
                            "constructed either. BridgeStan error: ",
                            conditionMessage(error_condition))
                    }
                    ##
                    names(default_theta_constrained) <- bs_names
                    ##
                    for (param_name in names(init_list_kk)) {
                      matching_indices <- which(bs_names == param_name)
                      ##
                      if (length(matching_indices) == 0) {
                        matching_indices <- grep( pattern = paste0( "^",
                                                                    gsub( pattern = "\\[",
                                                                          replacement = "\\\\[",
                                                                          x = gsub( pattern = "\\]",
                                                                                    replacement = "\\\\]",
                                                                                    x = param_name)),
                                                                  "(\\[|$)"),
                                                  x = bs_names)
                      }
                      ##
                      if (length(matching_indices) > 0) {
                        default_theta_constrained[matching_indices] <- as.numeric(init_list_kk[[param_name]])
                      } else {
                        warning( "R_fn_init_initial_values: '", param_name,
                                 "' in the init list for chain ", kk,
                                 " does not match any parameter of the model and was ignored.")
                      }
                    }
                    ##
                    inits_unconstrained_vec_kk <- bs_model$param_unconstrain( default_theta_constrained )
                    ##
                    message("R_fn_init_initial_values: chain ", kk,
                            " inits were PARTIAL - missing parameters were completed with model defaults.")
                
              }
              ##
              ## (the nuisance blocks held by NicoStan in front of the main parameters; nothing otherwise)
              inits_unconstrained_vec_kk <-  c(initial_values_of_the_nuisance_blocks_held_by_NicoStan_kk,
                                               inits_unconstrained_vec_kk)
              ##
              inits_unconstrained_vec_per_chain[[kk]] <- inits_unconstrained_vec_kk
              
            } else {
              
##
              ## ---- built-in models: unchanged (u_raw dropped by name, as before):
##
              init_list_kk <- init_lists_per_chain[[kk]]
              init_list_kk$u_raw <- NULL  # Remove nuisance
              ##
              json_string_for_inits_chain_kk <- convert_stan_data_list_to_JSON(init_list_kk)
              validated_json_string <- paste(readLines(json_string_for_inits_chain_kk), collapse="")
              inits_unconstrained_vec_per_chain[[kk]] <- bs_model$param_unconstrain_json(validated_json_string)
              
            }
            ##
            cat("n_params_main =", n_params_main, "\n")
            cat("length of unconstrained vec =", length(inits_unconstrained_vec_per_chain[[kk]]), "\n")
            
        }
        ##
        if (Model_type == "Stan") {
          
              ## ---- Stan models: split the COMPLETE unconstrained vector by position -
              ## the first n_nuisance coordinates are nuisance, the rest main. This holds
              ## for n_nuisance = 0 (empty nuisance; everything is main) as well.
              if (isTRUE(sample_nuisance) && n_nuisance > 0) {
                    index_nuisance <- seq_len(n_nuisance)
                    index_main <- (1 + n_nuisance):(n_nuisance + n_params_main)
                    ##
                    theta_nuisance_vectors_all_chains_input_from_R <- array( 0,
                                                                              dim = c(n_nuisance, n_chains_burnin))
                    theta_main_vectors_all_chains_input_from_R      <- array( 0,
                                                                              dim = c(n_params_main, n_chains_burnin))
                    ##
                    for (kk in 1:n_chains_burnin) {
                      theta_nuisance_vectors_all_chains_input_from_R[, kk] <- inits_unconstrained_vec_per_chain[[kk]][index_nuisance]
                      theta_main_vectors_all_chains_input_from_R[, kk]      <- inits_unconstrained_vec_per_chain[[kk]][index_main]
                    }
              } else {
                    ## sample_nuisance = FALSE (no nuisance block) OR n_nuisance = 0
                    ## (zero-length nuisance declaration): the complete unconstrained
                    ## vector is the MAIN block.
                    theta_nuisance_vectors_all_chains_input_from_R <- array( 0,
                                                                              dim = c(0, n_chains_burnin))
                    theta_main_vectors_all_chains_input_from_R      <- array( 0,
                                                                              dim = c(n_params_main, n_chains_burnin))
                    ##
                    for (kk in 1:n_chains_burnin) {
                      theta_main_vectors_all_chains_input_from_R[, kk] <- inits_unconstrained_vec_per_chain[[kk]][1:n_params_main]
                    }
              }
          
        } else {
          
##
              ## ---- built-in models: unchanged split logic:
##
              if (sample_nuisance == FALSE) {
                
                    n_nuisance_dummy <- 1 ## bookmark
                    n_params <- n_params_main
                    ##
                    theta_nuisance_vectors_all_chains_input_from_R    <- array(0, dim = c(n_nuisance_dummy, n_chains_burnin))
                    theta_main_vectors_all_chains_input_from_R  <- array(0, dim = c(n_params_main, n_chains_burnin))
                    ##
                    for (kk in 1:n_chains_burnin) {
                        theta_main_vectors_all_chains_input_from_R[, kk] <-     inits_unconstrained_vec_per_chain[[kk]][1:n_params_main]
                    }
                
              } else if (sample_nuisance == TRUE) {
                    
                    # n_params <- n_params_main ## + n_nuisance
                    index_nuisance <- seq_len(n_nuisance)
                    # index_main <- (1 + n_nuisance):n_params
                    ##
                    theta_nuisance_vectors_all_chains_input_from_R    <- array(0, dim = c(n_nuisance, n_chains_burnin))
                    theta_main_vectors_all_chains_input_from_R  <- array(0, dim = c(n_params_main, n_chains_burnin))
                    ##
                    ##
                    ## ---- the built-in C++ log densities store the nuisance vector CHUNK BY CHUNK (chunk k
                    ##      is a column-major (chunk_size x n_tests) block, fn_nuisance_chunk_layout_positions),
                    ##      so each (observation, test) entry of u_raw (N x n_tests, or a vector read column by
                    ##      column) goes to its position in that layout, with the burn-in's number of chunks
                    ##      and vectorisation (with one chunk, the layout is column by column):
                    ##
                    chunk_layout_in_burnin <-  chunk_layout_of_nuisance_vector_in_burnin
                    if (is.null(chunk_layout_in_burnin)) {
                      stop(paste0("R_fn_init_initial_values: chunk_layout_of_nuisance_vector_in_burnin ",
                                  "(list(N, n_tests, n_chunks, vect_type)) is needed to lay out the u_raw ",
                                  "initial values of a built-in model."))
                    }
                    position_in_nuisance_vector <-
                          fn_nuisance_chunk_layout_positions( N = chunk_layout_in_burnin$N,
                                                              n_tests = chunk_layout_in_burnin$n_tests,
                                                              n_chunks = chunk_layout_in_burnin$n_chunks,
                                                              vect_type = chunk_layout_in_burnin$vect_type)
                    position_in_nuisance_vector <-  as.vector(position_in_nuisance_vector)
                    stopifnot(length(position_in_nuisance_vector) == n_nuisance)
                    ##
                    for (kk in 1:n_chains_burnin) {
                      u_raw_kk <- init_lists_per_chain[[kk]]$u_raw
                      stopifnot(length(u_raw_kk) == n_nuisance)
                      theta_main_vectors_all_chains_input_from_R[, kk] <-     inits_unconstrained_vec_per_chain[[kk]] ## [index_main]
                      # theta_nuisance_vectors_all_chains_input_from_R[, kk] <-   u_raw_kk ##
                      #   inits_unconstrained_vec_per_chain[[kk]][index_nuisance]
                      theta_nuisance_vectors_all_chains_input_from_R[position_in_nuisance_vector, kk] <-
                            as.vector(u_raw_kk)
                    }
                
              }
          
        }
        ##
        return(list(inits_unconstrained_vec_per_chain = inits_unconstrained_vec_per_chain,
                    theta_nuisance_vectors_all_chains_input_from_R = theta_nuisance_vectors_all_chains_input_from_R,
                    theta_main_vectors_all_chains_input_from_R = theta_main_vectors_all_chains_input_from_R,
                    n_chains_burnin = n_chains_burnin,
                    init_lists_per_chain = init_lists_per_chain))

}




##
## ---- The initial values of the nuisance blocks held by NicoStan (nuisance_params, nico_reduce_sum): --------
##
#' The initial values of the nuisance blocks held by NicoStan, taken out of one chain's init list
#'
#' The nuisance blocks of nuisance_params are not parameters of the compiled model (NicoStan holds them; see
#' R_fn_write_nico_reduce_sum_version_of_Stan_model.R), so their initial values are taken out of the init
#' list here, in the order NicoStan holds them: the blocks one after the other (in the order of
#' nuisance_params), each in Stan's own order of its values, i.e. term after term, with the values of each term
#' together (for array[E] vector[K] and array[E] row_vector[K]: z[1][1:K], then z[2][1:K], ...). Each block is
#' given as in Stan's JSON data: a vector (vector[E], row_vector[E] and array[E] real), or, for
#' array[E] vector[K] and array[E] row_vector[K], a matrix with one row per term (E x K) or a list of the E
#' terms' vectors. A component of a tuple block z (z_tuple_component_<i>, and z_tuple_component_<i>_<j> for a
#' tuple inside it) is the i-th element of z (and the j-th element of that); an array[A] matrix[R, C] block is an
#' A x R x C array, or a list of its A matrices. NicoStan holds every block on its UNCONSTRAINED scale: the
#' initial values of a constrained block (its record's constraint, first_bound and second_bound) are given on its
#' own, constrained scale, as for Stan, and are put on the unconstrained scale here with Stan's own inverse
#' transforms (lower: log(x - a); upper: log(b - x); lower and upper: logit((x - a) / (b - a)); offset and
#' multiplier: (x - a) / b), its bounds worked out from Stan_data_list. A block not in the init list starts at 0
#' on the unconstrained scale.
#'
#' @param nuisance_blocks_held_by_NicoStan The nuisance_blocks of
#'   R_fn_write_nico_reduce_sum_version_of_Stan_model() (name, shape, terms and values_per_term, the last two
#'   Stan expressions of the data).
#' @param init_list The initial values of one chain.
#' @param Stan_data_list The model's data, which give each block's size (its terms times its values per term).
#' @return A list: initial_values_of_the_nuisance_blocks (a vector) and init_list_of_the_main_parameters (the init
#'   list without the nuisance blocks, nor the tuples they come from).
#' @keywords internal
fn_initial_values_of_the_nuisance_blocks_held_by_NicoStan_from_the_init_list <-  function(
        nuisance_blocks_held_by_NicoStan,
        init_list,
        Stan_data_list
) {

        initial_values_of_the_nuisance_blocks <-  c()
        names_taken_out_of_the_init_list <-  c()
        ##
        for (i in seq_len(nrow(nuisance_blocks_held_by_NicoStan))) {
              ##
              block_name <-  nuisance_blocks_held_by_NicoStan$name[i]
              ##
              ## ---- the block's terms and values per term (Stan expressions of the data):
##
              n_terms <-  tryCatch(eval(parse(text = nuisance_blocks_held_by_NicoStan$terms[i]),
                                        envir = Stan_data_list),
                                   error = function(e) NA_real_)
              n_values_per_term <-  tryCatch(eval(parse(text =
                                                        nuisance_blocks_held_by_NicoStan$values_per_term[i]),
                                                  envir = Stan_data_list),
                                             error = function(e) NA_real_)
              size_of_block <-  n_terms * n_values_per_term
              ##
              ## ---- the block's initial values in the init list: by its name, or, for a component of a tuple
              ##      block, the element of that tuple:
              ##
              initial_values_of_block <-  init_list[[block_name]]
              if (!is.null(initial_values_of_block)) {

                    names_taken_out_of_the_init_list <-  c(names_taken_out_of_the_init_list, block_name)
              } else if (grepl(pattern = "_tuple_component_[0-9_]+$", x = block_name)) {

                    tuple_name <-  sub(pattern = "_tuple_component_[0-9_]+$", replacement = "", x = block_name)
                    components <-  as.numeric(strsplit(x = sub(pattern = "^.*_tuple_component_", replacement = "",
                                                               x = block_name),
                                                       split = "_")[[1]])
                    initial_values_of_block <-  init_list[[tuple_name]]
                    for (component in components) {
                          if (is.null(initial_values_of_block)) break
                          initial_values_of_block <-  initial_values_of_block[[component]]
                    }
                    names_taken_out_of_the_init_list <-  c(names_taken_out_of_the_init_list, tuple_name)
              }
              ##
              ## ---- in Stan's own order of the block's values (term after term); 0 when not given:
              ##
              if (is.null(initial_values_of_block)) {
                    if (is.na(size_of_block)) {
                          stop(paste0("R_fn_init_initial_values: the nuisance block ", block_name, " is not in ",
                                      "the init list, and its size could not be worked out from the data; give ",
                                      "its initial values in the init list."))
                    }
                    initial_values_of_block <-  rep(0, size_of_block)
              } else if (identical(nuisance_blocks_held_by_NicoStan$shape[i], "matrix") &&
                         is.matrix(initial_values_of_block)) {
                    ## (a matrix block: its own R x C matrix, whose values Stan orders column by column, one
                    ##  column per term)
                    if (!is.na(n_values_per_term) && (nrow(initial_values_of_block) != n_values_per_term)) {
                          stop(paste0("R_fn_init_initial_values: the initial values of the matrix nuisance block ",
                                      block_name, " have ", nrow(initial_values_of_block), " rows; give ",
                                      n_values_per_term, " rows (the rows of the matrix)."))
                    }
                    initial_values_of_block <-  as.vector(initial_values_of_block)
              } else if (identical(nuisance_blocks_held_by_NicoStan$shape[i], "array_of_matrices") &&
                         is.array(initial_values_of_block) && length(dim(initial_values_of_block)) == 3) {
                    ## (an array of matrices as Stan's JSON data, A x R x C: matrix after matrix, each column by
                    ##  column)
                    if (!is.na(n_terms) && (dim(initial_values_of_block)[1] != n_terms)) {
                          stop(paste0("R_fn_init_initial_values: the initial values of the nuisance block ",
                                      block_name, " are an array of ", dim(initial_values_of_block)[1],
                                      " matrices; give ", n_terms, "."))
                    }
                    initial_values_of_block <-  as.vector(aperm(initial_values_of_block, c(2, 3, 1)))
              } else if (is.matrix(initial_values_of_block)) {
                    ## (one row per term, as in Stan's JSON data)
                    if (!is.na(n_terms) && (nrow(initial_values_of_block) != n_terms)) {
                          stop(paste0("R_fn_init_initial_values: the initial values of the nuisance block ",
                                      block_name, " are a matrix with ", nrow(initial_values_of_block),
                                      " rows; give one row per term (", n_terms, " rows)."))
                    }
                    initial_values_of_block <-  as.vector(t(initial_values_of_block))
              } else if (is.list(initial_values_of_block)) {
                    ## (the terms' vectors, one after the other)
                    initial_values_of_block <-  as.numeric(unlist(initial_values_of_block))
              } else {
                    initial_values_of_block <-  as.numeric(initial_values_of_block)
              }
              if (!is.na(size_of_block) && (length(initial_values_of_block) != size_of_block)) {
                    stop(paste0("R_fn_init_initial_values: the nuisance block ", block_name, " has ",
                                length(initial_values_of_block), " initial values; it needs ", size_of_block,
                                " (", nuisance_blocks_held_by_NicoStan$terms[i], " x ",
                                nuisance_blocks_held_by_NicoStan$values_per_term[i], ")."))
              }
              ##
              ## ---- a constrained block given in the init list: its values on the unconstrained scale, with
              ##      Stan's own inverse transforms:
              ##
              name_of_the_constraint <-  if (is.null(nuisance_blocks_held_by_NicoStan$constraint)) "" else
                                              nuisance_blocks_held_by_NicoStan$constraint[i]
              if (nzchar(name_of_the_constraint) && (block_name %in% names_taken_out_of_the_init_list)) {
                    fn_bound <-  function(expression) {

                          bound <-  tryCatch(eval(parse(text = expression), envir = Stan_data_list),
                                             error = function(e) NA_real_)
                          if (!is.numeric(bound) || length(bound) != 1 || is.na(bound)) {

                                stop(paste0("R_fn_init_initial_values: the bound ", expression, " of the ",
                                            "constrained nuisance block ", block_name, " cannot be worked out ",
                                            "from Stan_data_list; leave the block out of the init list (it then ",
                                            "starts at 0 on the unconstrained scale)."))
                          }

                          return(bound)
                    }

                    a <-  fn_bound(nuisance_blocks_held_by_NicoStan$first_bound[i])
                    b <-  fn_bound(nuisance_blocks_held_by_NicoStan$second_bound[i])
                    x <-  initial_values_of_block
                    inside <-  switch(name_of_the_constraint,
                                      "lower"              = all(x > a),
                                      "upper"              = all(x < b),
                                      "lower_upper"        = all(x > a & x < b),
                                      "offset_multiplier"  = (b > 0))
                    if (!isTRUE(inside)) {
                          stop(paste0("R_fn_init_initial_values: the initial values of the nuisance block ",
                                      block_name, " are not strictly inside its constraint (",
                                      name_of_the_constraint, ", ", a, ", ", b, ")."))
                    }
                    initial_values_of_block <-  switch(name_of_the_constraint,
                                                       "lower"              = log(x - a),
                                                       "upper"              = log(b - x),
                                                       "lower_upper"        = stats::qlogis((x - a) / (b - a)),
                                                       "offset_multiplier"  = (x - a) / b)
              }
              ##
              initial_values_of_the_nuisance_blocks <-  c(initial_values_of_the_nuisance_blocks,
                                                          initial_values_of_block)
        }
        ##
        init_list_of_the_main_parameters <-  init_list[!(names(init_list) %in% names_taken_out_of_the_init_list)]
        ##
        return(list( initial_values_of_the_nuisance_blocks  = initial_values_of_the_nuisance_blocks,
                     init_list_of_the_main_parameters       = init_list_of_the_main_parameters))

}

















                  
                   
                  
                 
  




















