#### =============================================================================================================
## R_fn_name_burnin_record_by_sampler_coordinate.R
##
## ---- Names on every per-coordinate object of the burn-in record (burn-in draws, adapted metric,
##      theta_hat_us, snaper / eigen vectors, final burn-in states), so that each entry says which coordinate of
##      the sampler's own vector it belongs to. The sampler stores the main block in its own layout (arrays
##      element by element, i.e. class by class, and test-indexed parameters in the FITTED test order of
##      reorder_cols_MVP), which is not the order of the summaries (first index fastest; test-indexed parameters
##      in the original test order).
#### =============================================================================================================


#' Names of the sampler's main and nuisance coordinates, in the sampler's own layout
#'
#' Main block: the constrained name of each coordinate of the sampler's main vector, read off
#' param_constrain() of an index vector (exact when every main parameter of the model has the identity
#' transform, as in the built-in skeletons; otherwise positional names). For Model_type = "LC_MVP" the test
#' index of beta[c,k,s] (fitted slot s) is replaced by the original test test_perm[s], i.e. each name is the
#' row name of the summaries (create_summary_and_traces) holding the same draws. Omega_unconstrained_vec[c,k]
#' keeps its sampler name (it has no original-order counterpart); its description gives the class and the
#' fitted and original tests of the pair. Nuisance block of a built-in model (N x n_tests latent values):
#' u_raw[observation, ORIGINAL test] at the position of the chunk layout the burn-in used
#' (fn_nuisance_chunk_layout_positions of the model package). For Model_type = "LC_MVOP" beta and
#' Omega_unconstrained_vec are named as for LC_MVP, and C_unc_vec[c,position] by its position in the
#' concatenation of the threshold blocks in the original order of the ordinal tests (the summary row).
#' @return list(main_names, main_descriptions, nuisance_names, nuisance_layout_description,
#'   names_method_main, names_method_nuisance)
#' @keywords internal
fn_names_of_sampler_coordinates_in_sampler_layout <-  function( init_object,
                                                               Model_type,
                                                               test_perm,
                                                               num_chunks_of_burnin_nuisance_layout,
                                                               vect_type_of_burnin_nuisance_layout,
                                                               fn_nuisance_chunk_layout_positions = NULL) {

        n_params_main <-  init_object$n_params_main
        n_nuisance    <-  init_object$n_nuisance
        bs_model      <-  init_object$bs_model
        n_tests       <-  init_object$model_args_list$n_tests
        if (is.null(test_perm) && !is.null(n_tests)) test_perm <-  seq_len(n_tests)
        ##
        ## ---- main block: positions of the constrained names in the sampler's vector (the main block is the
        ##      whole unconstrained vector of a built-in skeleton, and the tail of an external model that declares
        ##      its nuisance block first):
        ##
        main_names <-  paste0("main_coordinate_", seq_len(n_params_main))
        main_descriptions <-  paste0("sampler main position ", seq_len(n_params_main),
                                     ": parameter name not identified (transformed parameters)")
        names_method_main <-  "positional"
        n_unconstrained <-  tryCatch(bs_model$param_unc_num(), error = function(error_object) NA)
        n_leading_coordinates <-  if (isTRUE(n_unconstrained == n_params_main)) 0 else
                                  if (isTRUE(n_unconstrained == n_params_main + n_nuisance)) n_nuisance else NA
        if (!is.na(n_leading_coordinates)) {
              index_vector <-  seq_len(n_unconstrained)
              constrained <-  tryCatch(bs_model$param_constrain(index_vector, include_tp = FALSE,
                                                                include_gq = FALSE),
                                       error = function(error_object) NULL)
              constrained_names <-  bs_model$param_names()
              if (!is.null(constrained) && (length(constrained) == n_unconstrained) &&
                  all(constrained == round(constrained)) && setequal(constrained, index_vector)) {
                    name_at_position <-  character(n_unconstrained)
                    name_at_position[constrained] <-  convert_bridgestan_par_names_to_stan(constrained_names)
                    main_names <-  name_at_position[n_leading_coordinates + seq_len(n_params_main)]
                    main_descriptions <-  paste0("sampler main position ", seq_len(n_params_main), ": ",
                                                 main_names)
                    names_method_main <-  "identity_transform_of_every_main_parameter"
              }
        }
        ##
        ## ---- test-indexed main parameters of LC_MVP: beta in the ORIGINAL test order (the summary row holding
        ##      the same draws), and the class and tests of each raw correlation parameter:
        ##      (LC_MVOP too: beta and the raw correlation parameters as for LC_MVP; its cutpoints below)
        ##
        # if ((names_method_main != "positional") && identical(Model_type, "LC_MVP") &&
        #     !is.null(test_perm)) {
        if ((names_method_main != "positional") && (Model_type %in% c("LC_MVP", "LC_MVOP")) &&
            !is.null(test_perm)) {
              fn_indices <-  function(name) {
                    return(as.integer(strsplit(sub("^[^\\[]*\\[(.*)\\]$", "\\1", name), ",")[[1]]))
              }
              pair_first_test  <-  unlist(lapply(2:n_tests, function(i) rep(i, i - 1)))
              pair_second_test <-  unlist(lapply(2:n_tests, function(i) seq_len(i - 1)))
              for (position in seq_len(n_params_main)) {
                    name <-  main_names[position]
                    if (grepl("^beta\\[[0-9]+,[0-9]+,[0-9]+\\]$", name)) {
                          index <-  fn_indices(name)
                          main_names[position] <-  paste0("beta[", index[1], ",", index[2], ",",
                                                          test_perm[index[3]], "]")
                          main_descriptions[position] <-  paste0("sampler main position ", position, ": ",
                                                                 main_names[position], " = coefficient ",
                                                                 index[2], " of class ", index[1],
                                                                 ", fitted test ", index[3],
                                                                 " = original test ", test_perm[index[3]])
                    } else if (grepl("^Omega_unconstrained_vec\\[[0-9]+,[0-9]+\\]$", name)) {
                          index <-  fn_indices(name)
                          i <-  pair_first_test[index[2]]
                          j <-  pair_second_test[index[2]]
                          main_descriptions[position] <-  paste0("sampler main position ", position, ": ",
                                                                 name, " = raw correlation parameter of class ",
                                                                 index[1], ", fitted tests (", i, ",", j,
                                                                 ") = original tests (", test_perm[i], ",",
                                                                 test_perm[j], ")")
                    }
              }
        }
        ##
        ## ---- cutpoints of LC_MVOP: C_unc_vec[c, position] of the sampler is the concatenation of the ordinal
        ##      tests' threshold blocks in the FITTED order of the ordinal tests; each name is the summary row
        ##      holding the same draws, i.e. the position in the concatenation in the ORIGINAL order of the
        ##      ordinal tests (as create_summary_and_traces re-indexes C_unc_vec, C_raw_vec and C_vec):
        ##
        n_cat_per_test_in_fitted_order     <-  init_object$model_args_list$n_cat_per_test
        n_thr_per_ord_test_in_fitted_order <-  init_object$model_args_list$n_thr_per_ord_test
        if ((names_method_main != "positional") && identical(Model_type, "LC_MVOP") && !is.null(test_perm) &&
            !is.null(n_cat_per_test_in_fitted_order) && !is.null(n_thr_per_ord_test_in_fitted_order) &&
            (length(n_thr_per_ord_test_in_fitted_order) > 0)) {
              original_test_of_fitted_ordinal_slot <-  test_perm[n_cat_per_test_in_fitted_order > 2]
              fitted_ordinal_slot_of_original_ordinal_test <-
                    match(sort(original_test_of_fitted_ordinal_slot), original_test_of_fitted_ordinal_slot)
              end_fitted   <-  cumsum(n_thr_per_ord_test_in_fitted_order)
              start_fitted <-  end_fitted - n_thr_per_ord_test_in_fitted_order + 1
              ## the fitted position of each original position, then its inverse:
              fitted_position_of_original_position <-
                    unlist(lapply(fitted_ordinal_slot_of_original_ordinal_test,
                                  function(k) start_fitted[k]:end_fitted[k]))
              original_position_of_fitted_position <-  order(fitted_position_of_original_position)
              fitted_ordinal_slot_of_fitted_position <-  rep(seq_along(n_thr_per_ord_test_in_fitted_order),
                                                             times = n_thr_per_ord_test_in_fitted_order)
              for (position in seq_len(n_params_main)) {
                    name <-  main_names[position]
                    if (grepl("^C_unc_vec\\[[0-9]+,[0-9]+\\]$", name)) {
                          index <-  as.integer(strsplit(sub("^[^\\[]*\\[(.*)\\]$", "\\1", name), ",")[[1]])
                          k <-  fitted_ordinal_slot_of_fitted_position[index[2]]
                          main_names[position] <-  paste0("C_unc_vec[", index[1], ",",
                                                          original_position_of_fitted_position[index[2]], "]")
                          main_descriptions[position] <-  paste0("sampler main position ", position, ": ",
                                                                 main_names[position], " = threshold ",
                                                                 index[2] - start_fitted[k] + 1, " of class ",
                                                                 index[1], ", fitted ordinal slot ", k,
                                                                 " = original test ",
                                                                 original_test_of_fitted_ordinal_slot[k])
                    }
              }
        }
        ##
        ## ---- nuisance block of a built-in model: u_raw[observation, ORIGINAL test] in the burn-in chunk layout:
        ##
        nuisance_names <-  if (n_nuisance > 0) paste0("nuisance_coordinate_", seq_len(n_nuisance)) else
                                                 character(0)
        nuisance_layout_description <-  "nuisance coordinates not identified (positional names)"
        names_method_nuisance <-  "positional"
        N <-  init_object$model_args_list$N
        if ((n_nuisance > 0) && !identical(Model_type, "Stan") &&
            is.function(fn_nuisance_chunk_layout_positions) &&
            !is.null(N) && !is.null(n_tests) && (N * n_tests == n_nuisance) &&
            !is.null(num_chunks_of_burnin_nuisance_layout) &&
            !is.null(vect_type_of_burnin_nuisance_layout)) {
              position_of_observation_and_fitted_test <-  fn_nuisance_chunk_layout_positions(
                    N        = N,
                    n_tests  = n_tests,
                    n_chunks = num_chunks_of_burnin_nuisance_layout,
                    vect_type = vect_type_of_burnin_nuisance_layout)
              nuisance_names[as.vector(position_of_observation_and_fitted_test)] <-
                    paste0("u_raw[", rep(seq_len(N), times = n_tests), ",", rep(test_perm, each = N), "]")
              nuisance_layout_description <-  paste0("u_raw[observation, ORIGINAL test]; the burn-in layout is ",
                                                     "chunk by chunk (num_chunks = ",
                                                     num_chunks_of_burnin_nuisance_layout, ", vect_type ",
                                                     vect_type_of_burnin_nuisance_layout,
                                                     "), each chunk a column-major (rows x fitted tests) block; ",
                                                     "fitted test j = original test test_perm[j] (test_perm = ",
                                                     paste(test_perm, collapse = " "), ")")
              names_method_nuisance <-  "chunk_layout_of_the_burnin"
        } else if ((n_nuisance > 0) &&
                   identical(names_method_main, "identity_transform_of_every_main_parameter") &&
                   isTRUE(n_leading_coordinates == n_nuisance)) {
              nuisance_names <-  name_at_position[seq_len(n_nuisance)]
              nuisance_layout_description <-  "the model's own nuisance parameters, in the sampler's order"
              names_method_nuisance <-  "identity_transform_of_every_nuisance_parameter"
        }
        ##
        return(list( main_names                  = main_names,
                     main_descriptions           = main_descriptions,
                     nuisance_names              = nuisance_names,
                     nuisance_layout_description = nuisance_layout_description,
                     names_method_main           = names_method_main,
                     names_method_nuisance       = names_method_nuisance))

}


#' Burn-in record with names on every per-coordinate object
#'
#' Adds names (vectors) or dimnames (matrices, arrays) from fn_names_of_sampler_coordinates_in_sampler_layout()
#' to the per-coordinate objects of the burn-in record; values are unchanged. Also records the names, the
#' descriptions of the main coordinates and the nuisance layout in the record. Called on the returned record
#' only (after sampling), so the objects handed to the sampler are untouched.
#' @keywords internal
fn_burnin_record_with_names_of_sampler_coordinates <-  function( burnin_object,
                                                                coordinate_names) {

        main_names     <-  coordinate_names$main_names
        nuisance_names <-  coordinate_names$nuisance_names
        n_main     <-  length(main_names)
        n_nuisance <-  length(nuisance_names)
        ##
        fn_name_vector <-  function(x, coordinate_names_for_x) {
              if (is.numeric(x) && is.null(dim(x)) && (length(x) == length(coordinate_names_for_x))) {
                    names(x) <-  coordinate_names_for_x
              }
              return(x)
        }
        fn_name_rows <-  function(x, coordinate_names_for_rows) {
              if (is.numeric(x) && (length(dim(x)) == 2) && (nrow(x) == length(coordinate_names_for_rows))) {
                    rownames(x) <-  coordinate_names_for_rows
              }
              return(x)
        }
        fn_name_rows_and_columns <-  function(x, coordinate_names_for_both) {
              if (is.numeric(x) && (length(dim(x)) == 2) && all(dim(x) == length(coordinate_names_for_both))) {
                    dimnames(x) <-  list(coordinate_names_for_both, coordinate_names_for_both)
              }
              return(x)
        }
        ##
        ## ---- burn-in draws and metric history (iteration x coordinate (x chain)), final states
        ##      (coordinate x chain):
        ##
        if (length(dim(burnin_object$burnin_trace_main_all_chains)) == 3 &&
            dim(burnin_object$burnin_trace_main_all_chains)[2] == n_main) {
              dimnames(burnin_object$burnin_trace_main_all_chains) <-  list(NULL, main_names, NULL)
        }
        if (length(dim(burnin_object$burnin_metric_main_variance_history)) == 2 &&
            ncol(burnin_object$burnin_metric_main_variance_history) == n_main) {
              colnames(burnin_object$burnin_metric_main_variance_history) <-  main_names
        }
        burnin_object$theta_main_vectors_all_chains_input_from_R <-
              fn_name_rows(burnin_object$theta_main_vectors_all_chains_input_from_R, main_names)
        for (field in c("theta_nuisance_vectors_all_chains_input_from_R",
                        "theta_us_vectors_all_chains_input_from_R")) {
              burnin_object[[field]] <-  fn_name_rows(burnin_object[[field]], nuisance_names)
        }
        ##
        ## ---- trajectory direction and metric factor (main; joint = main rows, then nuisance rows):
        ##
        for (field in c("snaper_direction_main", "trajectory_metric_factor_main")) {
              burnin_object[[field]] <-  fn_name_vector(burnin_object[[field]], main_names)
        }
        burnin_object$snaper_direction_joint <-  fn_name_vector(burnin_object$snaper_direction_joint,
                                                                c(main_names, nuisance_names))
        ##
        ## ---- adapted metric:
        ##
        for (field in c("M_inv_main_vec", "M_main_vec")) {
              burnin_object$EHMC_Metric_as_Rcpp_List[[field]] <-
                    fn_name_vector(burnin_object$EHMC_Metric_as_Rcpp_List[[field]], main_names)
        }
        for (field in c("M_dense_main", "M_inv_dense_main", "M_inv_dense_main_chol")) {
              burnin_object$EHMC_Metric_as_Rcpp_List[[field]] <-
                    fn_name_rows_and_columns(burnin_object$EHMC_Metric_as_Rcpp_List[[field]], main_names)
        }
        for (field in c("M_inv_us_vec", "M_us_vec")) {
              burnin_object$EHMC_Metric_as_Rcpp_List[[field]] <-
                    fn_name_vector(burnin_object$EHMC_Metric_as_Rcpp_List[[field]], nuisance_names)
        }
        burnin_object$EHMC_Metric_as_Rcpp_List$theta_hat_us_vec <-
              fn_name_rows(burnin_object$EHMC_Metric_as_Rcpp_List$theta_hat_us_vec, nuisance_names)
        ##
        ## ---- burn-in state of the adaptation (snaper centres / scales / directions, eigen vectors, metric
        ##      roots):
        ##
        for (field in c("snaper_m_vec_main", "snaper_w_vec_main", "eigen_vector_main",
                        "snaper_s_vec_main_empirical", "sqrt_M_main_vec")) {
              burnin_object$EHMC_burnin_as_Rcpp_List[[field]] <-
                    fn_name_vector(burnin_object$EHMC_burnin_as_Rcpp_List[[field]], main_names)
        }
        burnin_object$EHMC_burnin_as_Rcpp_List$M_dense_sqrt <-
              fn_name_rows_and_columns(burnin_object$EHMC_burnin_as_Rcpp_List$M_dense_sqrt, main_names)
        for (field in c("snaper_m_vec_us", "snaper_w_vec_us", "eigen_vector_us",
                        "snaper_s_vec_us_empirical", "sqrt_M_us_vec")) {
              burnin_object$EHMC_burnin_as_Rcpp_List[[field]] <-
                    fn_name_vector(burnin_object$EHMC_burnin_as_Rcpp_List[[field]], nuisance_names)
        }
        ##
        ## ---- the names themselves, the descriptions of the main coordinates and the nuisance layout:
        ##
        burnin_object$sampler_coordinate_names_main            <-  main_names
        burnin_object$sampler_coordinate_descriptions_main     <-  coordinate_names$main_descriptions
        burnin_object$sampler_coordinate_names_nuisance        <-  nuisance_names
        burnin_object$sampler_nuisance_layout_description      <-  coordinate_names$nuisance_layout_description
        burnin_object$sampler_coordinate_names_method_main     <-  coordinate_names$names_method_main
        burnin_object$sampler_coordinate_names_method_nuisance <-  coordinate_names$names_method_nuisance
        ##
        return(burnin_object)

}






















