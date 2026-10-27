##
## ---- R_fn_Stan_model_math_functions_BayesMVP_AVX.R ---------------------------------------------------------------------------------
##
## ---- The math backend used to write a Stan model -----------------------------------------------------------------------------------
##
## Stan_math keeps Stan's functions. Every accepted AVX label writes native NicoSIMD `nico_` calls and headers.
## Catalogued or unclassified calls that require NicoSIMD stop generation; listed basic and structural calls stay.
## Generated models are reused only when the source, declarations and writer implementation match.
##
##
##
##
##
## ---- BayesMVP's AVX functions which replace Stan's functions, and their Stan declarations ------------------------------------------
##
#' BayesMVP's AVX functions which replace Stan's functions in the AVX version of a Stan model
#'
#' The ONE place of the lists: the elementwise functions (each declared for a vector argument), the fused log
#' densities (each declared for every argument type it supports; the name of each list element is the Stan
#' distribution it replaces) and the BayesMVP header which defines each fused log density.
#' @return A list: elementwise_functions, fused_distribution_declarations and
#'   fused_distribution_BayesMVP_header_files.
#' @keywords internal
##
## ---- The function ------------------------------------------------------------------------------------------------------------------
##
fn_BayesMVP_AVX_functions_and_their_Stan_declarations <-  function() {

        ##
        ## ---- the elementwise functions (vector -> vector, stan_external_functions.hpp): --------------------------------------------
        ##
        # elementwise_functions <-  c("exp", "log", "log1p", "log1m", "log1p_exp", "log1m_exp", "inv_logit",
        #                             "log_inv_logit", "Phi", "Phi_approx", "inv_Phi", "tanh", "square", "sqrt",
        #                             "logit", "log1m_inv_logit", "expm1")
        ## (sqrt, square, abs, fmin, fmax and floor may remain on Stan math)
        elementwise_functions <-  c("exp", "log", "log1p", "log1m", "log1p_exp", "log1m_exp", "inv_logit",
                                    "log_inv_logit", "Phi", "Phi_approx", "inv_Phi", "tanh", "logit",
                                    "log1m_inv_logit", "expm1")
        ## (scalar functions return real; container functions return the same shape)
        elementwise_function_argument_types <-  c("real", "vector", "row_vector", "matrix")
        ##
        ## ---- the fused AVX log densities, and the Stan declarations of the argument types each supports: ---------------------------
        ##
        fused_distribution_declarations <-  list(
              bernoulli_logit_rng = c(
                    "    int nico_bernoulli_logit_rng(real x);",
                    "    array[] int nico_bernoulli_logit_rng(vector x);",
                    "    array[] int nico_bernoulli_logit_rng(row_vector x);",
                    "    array[] int nico_bernoulli_logit_rng(array[] real x);"),
              ## (the random generators with exp, log or tan inside: the same draws in the same order, those
              ##  calls on nico; uniform_rng and bernoulli_rng, a product and a compare, stay on Stan math)
              normal_rng = c(
                    "    real nico_normal_rng(real mu, real sigma);",
                    "    array[] real nico_normal_rng(vector mu, real sigma);",
                    "    array[] real nico_normal_rng(real mu, vector sigma);",
                    "    array[] real nico_normal_rng(vector mu, vector sigma);",
                    "    array[] real nico_normal_rng(row_vector mu, real sigma);",
                    "    array[] real nico_normal_rng(array[] real mu, real sigma);"),
              multi_normal_rng = c(
                    "    vector nico_multi_normal_rng(vector mu, matrix Sigma);",
                    "    array[] vector nico_multi_normal_rng(array[] vector mu, matrix Sigma);"),
              student_t_rng = c(
                    "    real nico_student_t_rng(real nu, real mu, real sigma);",
                    "    array[] real nico_student_t_rng(real nu, vector mu, real sigma);",
                    "    array[] real nico_student_t_rng(real nu, real mu, vector sigma);",
                    "    array[] real nico_student_t_rng(real nu, vector mu, vector sigma);",
                    "    array[] real nico_student_t_rng(vector nu, vector mu, vector sigma);"),
              weibull_rng = c(
                    "    real nico_weibull_rng(real alpha, real sigma);",
                    "    array[] real nico_weibull_rng(real alpha, vector sigma);",
                    "    array[] real nico_weibull_rng(vector alpha, real sigma);",
                    "    array[] real nico_weibull_rng(vector alpha, vector sigma);"),
              normal = c("    real nico_normal_lpdf(vector y, vector mu, vector sigma);",
                         "    real nico_normal_lpdf(vector y, vector mu, real sigma);",
                         "    real nico_normal_lpdf(vector y, real mu, vector sigma);",
                         "    real nico_normal_lpdf(vector y, real mu, real sigma);",
                         ## (a row_vector y, mu and sigma, as the NMA ord_biv model's priors of beta_mu's rows)
                         "    real nico_normal_lpdf(row_vector y, row_vector mu, row_vector sigma);",
                         "    real nico_normal_lpdf(array[] real y, vector mu, vector sigma);",
                         "    real nico_normal_lpdf(array[] real y, vector mu, real sigma);",
                         "    real nico_normal_lpdf(array[] real y, real mu, vector sigma);",
                         "    real nico_normal_lpdf(array[] real y, real mu, real sigma);",
                         "    real nico_normal_lpdf(real y, real mu, real sigma);"),
              student_t =
                    c("    real nico_student_t_lpdf(vector y, real nu, vector mu, vector sigma);",
                      "    real nico_student_t_lpdf(vector y, real nu, vector mu, real sigma);",
                      "    real nico_student_t_lpdf(vector y, real nu, real mu, vector sigma);",
                      "    real nico_student_t_lpdf(vector y, real nu, real mu, real sigma);",
                      "    real nico_student_t_lpdf(array[] real y, real nu, vector mu, vector sigma);",
                      "    real nico_student_t_lpdf(array[] real y, real nu, vector mu, real sigma);",
                      "    real nico_student_t_lpdf(array[] real y, real nu, real mu, vector sigma);",
                      "    real nico_student_t_lpdf(array[] real y, real nu, real mu, real sigma);",
                      ## (one value, e.g. log_kappa[c] ~ student_t(...): the kernel takes a scalar y)
                      "    real nico_student_t_lpdf(real y, real nu, real mu, real sigma);"),
              ## (Weibull: y vector or array[] real; alpha and sigma each real, vector or array[] real;
              ##  the log density and the log complementary CDF)
              weibull = as.vector(outer(
                    X = c("nico_weibull_lpdf", "nico_weibull_lccdf"),
                    Y = apply(expand.grid(y = c("vector y", "array[] real y"),
                                          alpha = c("real alpha", "vector alpha", "array[] real alpha"),
                                          sigma = c("real sigma", "vector sigma", "array[] real sigma"),
                                          stringsAsFactors = FALSE),
                              MARGIN = 1, FUN = paste, collapse = ", "),
                    ##
                    ## ---- The function ----------------------------------------------------------------------------------------------
                    ##
                    FUN = function(function_name, arguments) {

                          paste0("    real ", function_name, "(", arguments, ");")

                    })),

              ordered_logistic =
                    c("    real nico_ordered_logistic_lpmf(int k, real eta, vector c);",
                      "    real nico_ordered_logistic_lpmf(array[] int k, vector eta, vector c);",
                      "    real nico_ordered_logistic_lpmf(array[] int k, vector eta, array[] vector c);"),
              bernoulli_logit = c("    real nico_bernoulli_logit_lpmf(array[] int y, vector f);"),
              ## (the Bernoulli-logit GLM: y | x, alpha, beta, as Stan's bernoulli_logit_glm)
              bernoulli_logit_glm = paste0("    real nico_bernoulli_logit_glm_lpmf(",
                                           "array[] int y, matrix x, ",
                                           c("real alpha", "vector alpha"), ", vector beta);"),
              ## (the Poisson-log GLM: y | x, alpha, beta, as Stan's poisson_log_glm)
              poisson_log_glm = paste0("    real nico_poisson_log_glm_lpmf(",
                                       "array[] int y, matrix x, ",
                                       c("real alpha", "vector alpha"), ", vector beta);"),
              bernoulli_logit_glm_with_a_tensor_product_basis = paste0(
                    "    real nico_bernoulli_logit_glm_with_a_tensor_product_basis_lpmf(\n",
                    "          array[] int y,\n",
                    "          data matrix x,\n",
                    "          real alpha,\n",
                    "          vector beta,\n",
                    "          data matrix basis_1,\n",
                    "          data matrix basis_2,\n",
                    "          matrix weights);"),
              ## (the Bernoulli-logit GLM on x, with each group's intercept alpha[group] and slopes B[group, ])
              bernoulli_logit_glm_with_coefficients_by_group = paste0(
                    "    real nico_bernoulli_logit_glm_with_coefficients_by_group_lpmf(array[] int y, matrix x, ",
                    "array[] int group,\n",
                    "                                                                  vector alpha, matrix B);"),
              ordered_logistic_glm = c(
                    "    real nico_ordered_logistic_glm_lpmf(array[] int y, matrix x, vector beta, vector cuts);",
                    paste0("    real nico_ordered_logistic_glm_lpmf(array[] int y, row_vector x, ",
                           "vector beta, vector cuts);"),
                    "    real nico_ordered_logistic_glm_lpmf(int y, matrix x, vector beta, vector cuts);",
                    "    real nico_ordered_logistic_glm_lpmf(int y, row_vector x, vector beta, vector cuts);"),
              dot_product_exp_of_affine_predictor_at_quadrature_nodes_by_column = paste0(
                    "    real nico_dot_product_exp_of_affine_predictor_at_quadrature_nodes_by_column(\n",
                    "          data vector node_weights, data vector node_positions, data vector weight_scale,\n",
                    "          data vector position_scale, data matrix X, vector a, vector b, vector beta);"),
              normal_id_glm = paste0("    real nico_normal_id_glm_lpdf(vector y, matrix x, ",
                                     c("real alpha", "vector alpha"), ", vector beta, real sigma);"),
              student_t_id_glm = paste0(
                    "    real nico_student_t_id_glm_lpdf(array[] real y,\n",
                    "                                    matrix x,\n",
                    "                                    real alpha,\n",
                    "                                    vector beta,\n",
                    "                                    real nu,\n",
                    "                                    real sigma);"),
              weibull_log_glm = c(
                    paste0(
                          "    real nico_weibull_log_glm_lpdf(array[] real y,\n",
                          "                               matrix x,\n",
                          "                               real alpha,\n",
                          "                               vector beta,\n",
                          "                               real shape);"),
                    paste0(
                          "    real nico_weibull_log_glm_lccdf(vector y,\n",
                          "                                matrix x,\n",
                          "                                real alpha,\n",
                          "                                vector beta,\n",
                          "                                real shape);")),
              lognormal = c("    real nico_lognormal_lpdf(real y, real mu, real sigma);",
                            "    real nico_lognormal_lpdf(vector y, real mu, real sigma);"),
              exponential = paste0("    real nico_exponential_lpdf(real y,\n",
                                    "                               real rate);"),
              inv_gamma = "    real nico_inv_gamma_lpdf(real y, data real alpha, data real beta);",
              ## (the vector, data or parameters: value lgamma, partial digamma, from one SIMD pass)
              lgamma = c("    real nico_lgamma(data real x);",
                         "    vector nico_lgamma(vector x);"),
              ## (a vector y with vector shapes, data or parameters, as the Se/Sp prior's beta_lpdf)
              beta = c("    real nico_beta_lpdf(real y, data real alpha, data real beta);",
                       "    real nico_beta_lpdf(vector y, vector alpha, vector beta);"),
              ## (alpha data or parameters: a data alpha takes the data kernel, a parameter alpha the other)
              dirichlet = "    real nico_dirichlet_lpdf(vector theta, vector alpha);",
              lkj_corr_cholesky = "    real nico_lkj_corr_cholesky_lpdf(matrix L, data real eta);",
              erfc = c("    real nico_erfc(real x);",
                       "    vector nico_erfc(vector x);"),
              sin = c("    real nico_sin(data real x);",
                      "    vector nico_sin(data vector x);"),
              ## (sinh, cosh, asinh, acosh and atanh: real, vector, row_vector and matrix, values and gradients)
              sinh = paste0("    ", c("real", "vector", "row_vector", "matrix"), " nico_sinh(",
                            c("real", "vector", "row_vector", "matrix"), " x);"),
              cosh = paste0("    ", c("real", "vector", "row_vector", "matrix"), " nico_cosh(",
                            c("real", "vector", "row_vector", "matrix"), " x);"),
              asinh = paste0("    ", c("real", "vector", "row_vector", "matrix"), " nico_asinh(",
                             c("real", "vector", "row_vector", "matrix"), " x);"),
              acosh = paste0("    ", c("real", "vector", "row_vector", "matrix"), " nico_acosh(",
                             c("real", "vector", "row_vector", "matrix"), " x);"),
              atanh = paste0("    ", c("real", "vector", "row_vector", "matrix"), " nico_atanh(",
                             c("real", "vector", "row_vector", "matrix"), " x);"),
              ## (y real or vector; alpha and beta real)
              uniform = c("    real nico_uniform_lpdf(real y, real alpha, real beta);",
                          "    real nico_uniform_lpdf(vector y, real alpha, real beta);"),
              log_diff_exp = c("    real nico_log_diff_exp(real x, real y);",
                               "    vector nico_log_diff_exp(vector x, vector y);"),
              std_normal_log_qf = c(
                    "    real nico_std_normal_log_qf(real x);",
                    "    vector nico_std_normal_log_qf(vector x);"
              ),
              cumulative_sum = c("    vector nico_cumulative_sum(vector x);",
                                 "    array[] int nico_cumulative_sum(array[] int x);"),
              cholesky_decompose = "    matrix nico_cholesky_decompose(matrix x);",
              mdivide_left_tri_low = c("    vector nico_mdivide_left_tri_low(matrix x, vector y);",
                                       "    matrix nico_mdivide_left_tri_low(matrix x, matrix y);"),
              mdivide_right_tri_low = c("    row_vector nico_mdivide_right_tri_low(row_vector x, matrix y);",
                                        "    matrix nico_mdivide_right_tri_low(matrix x, matrix y);"),
              gp_normal_marginal = paste0(
                    "    real nico_gp_normal_marginal_lpdf(vector y, real mu, data array[] real x,\n",
                    "                                           data int kernel, real sigma,\n",
                    "                                           real length_scale, real diagonal);"),
              gp_cholesky_factor_of_covariance_plus_diagonal_times_vector = paste0(
                    "    vector nico_gp_cholesky_factor_of_covariance_plus_diagonal_times_vector(\n",
                    "          data array[] real x, data int kernel, real sigma,\n",
                    "          real length_scale, real diagonal, vector z);"),
              binomial = "    real nico_binomial_lpmf(array[] int n, array[] int N, vector theta);",
              ## (log_sum_exp reductions and elementwise pairs)
              log_sum_exp = c("    real nico_log_sum_exp(vector x);",
                              "    real nico_log_sum_exp(row_vector x);",
                              "    real nico_log_sum_exp(matrix x);",
                              "    real nico_log_sum_exp(array[] real x);",
                              "    real nico_log_sum_exp(real a, real b);",
                              "    vector nico_log_sum_exp(vector a, vector b);",
                              "    vector nico_log_sum_exp(real a, vector b);",
                              "    vector nico_log_sum_exp(vector a, real b);"),
              ## (the row-wise log_sum_exp of a matrix: the calls of a model's own function log_sum_exp_2d(), as
              ##  in the latent class models, whose definition keeps its name)
              log_sum_exp_2d = c("    vector nico_log_sum_exp_2d(matrix x);"),
              ## (the AR(1) path and the normal log density on the log scale, as in stochastic volatility)
              normal_log_scale = "    real nico_normal_log_scale_lpdf(vector y, real mu, vector log_scale);",
              AR1_recursion = "    vector nico_AR1_recursion(vector shocks, real carry, real a);",
              ## (the GHK log likelihood of a finite mixture of multivariate ordered probit models)
              finite_mixture_of_multivariate_ordered_probit_GHK_log_likelihood = paste0(
                    "    real nico_finite_mixture_of_multivariate_ordered_probit_GHK_log_likelihood(\n",
                    "          array[] row_vector u_raw, data int nuisance_transformation, matrix intercept,\n",
                    "          array[] matrix coefficients, ",
                    "data array[,] int coefficient_set_of_class_and_test,\n",
                    "          data array[,] matrix covariates, data array[,] int n_covariates, ",
                    "array[] vector cutpoints,\n",
                    "          data array[,] int cutpoint_set_of_class_and_test,\n",
                    "          data array[] int position_of_first_cutpoint_of_test,\n",
                    "          array[] matrix L_Omega, matrix log_class_probability, ",
                    "data array[,] int category,\n",
                    "          data array[,] int upper_category, data array[] int test_is_binary,\n",
                    "          data array[] int n_categories_of_test, data array[] int test_at_GHK_position, ",
                    "data int Phi_type,\n",
                    "          data real overflow_threshold, data real underflow_threshold);"),
              finite_mixture_of_multivariate_ordered_probit_GHK_log_likelihood_in_tiles_of_subjects = paste0(
                    "      real nico_finite_mixture_of_multivariate_ordered_probit_GHK_",
                    "log_likelihood_in_tiles_of_subjects(\n",
                    "            matrix u_raw_of_each_test_and_subject,\n",
                    "            data int first_subject,\n",
                    "            data int nuisance_transformation,\n",
                    "            matrix intercept,\n",
                    "            array[] matrix coefficients,\n",
                    "            data array[,] int coefficient_set_of_class_and_test,\n",
                    "            data array[,] matrix covariates,\n",
                    "            data array[,] int n_covariates,\n",
                    "            array[] vector cutpoints,\n",
                    "            data array[,] int cutpoint_set_of_class_and_test,\n",
                    "            data array[] int position_of_first_cutpoint_of_test,\n",
                    "            array[] matrix L_Omega,\n",
                    "            matrix log_class_probability,\n",
                    "            data array[,] int category,\n",
                    "            data array[,] int upper_category,\n",
                    "            data array[] int test_is_binary,\n",
                    "            data array[] int n_categories_of_test,\n",
                    "            data array[] int test_at_GHK_position,\n",
                    "            data int Phi_type,\n",
                    "            data real overflow_threshold,\n",
                    "            data real underflow_threshold);"),
              ## (the sum over k of A[group_i, k] exp(B[group_i, k] t_i), as in the PK model)
              sum_of_exponentials_by_group = paste0(
                    "    vector nico_sum_of_exponentials_by_group(data vector t, data array[] int group, ",
                    "matrix A, matrix B);"),
              ## (the exponents and coefficients of a linear two-compartment system with first-order input)
              two_compartment_first_order_input_exponents_and_coefficients = paste0(
                    "    matrix nico_two_compartment_first_order_input_exponents_and_coefficients(matrix rates, ",
                    "data real minimum_rate_separation);"),
              ## (exp(mean + z[i, ] transposed_loadings), or without exp(), for every row i of z)
              rows_of_mean_plus_random_effects_times_transposed_loadings = paste0(
                    "    matrix nico_rows_of_mean_plus_random_effects_times_transposed_loadings(matrix z, ",
                    "row_vector mean, matrix transposed_loadings, data int exponentiate);"),
              ## (the proportional-hazards log likelihood with a piecewise linear baseline hazard on a grid, as in
              ##  latent diffusion survival)
              proportional_hazards_log_likelihood_with_a_piecewise_linear_baseline_hazard_on_a_grid = paste0(
                    "    real nico_proportional_hazards_log_likelihood_with_a_piecewise_linear_baseline_hazard_",
                    "on_a_grid(\n",
                    "          data array[] int event, data matrix X, vector beta, data array[] int cell,\n",
                    "          data vector time_into_cell, data vector cell_widths, vector h);"),
              ## (the binomial likelihood of the counts above successive reported cutpoints of a cumulative
              ##  link, and the induced-Dirichlet prior of ordered cutpoints, as in the NMA ord_biv model)
              binomial_counts_above_successive_cutpoints_of_a_cumulative_link = paste0(
                    "    real nico_binomial_counts_above_successive_cutpoints_of_a_cumulative_link_lpmf(\n",
                    "          array[] int n, array[] int n_previous, vector latent, data vector is_first, ",
                    "data int link,\n",
                    "          data real bound);"),
              induced_dirichlet_prior_of_ordered_cutpoints = paste0(
                    "    real nico_induced_dirichlet_prior_of_ordered_cutpoints_lpdf(vector C, ",
                    "data vector alpha, data int link);"),
              ## (the log determinant of the Jacobian of ordered cutpoints built from raw differences, as
              ##  MetaOrdDTA's raw_C_to_C_log_det_J_lp in the NMA ord_biv model)
              log_det_jacobian_of_cutpoints_from_raw_differences = paste0(
                    "    real nico_log_det_jacobian_of_cutpoints_from_raw_differences(vector C_raw, ",
                    "data int softplus);"),
              ## (the squared exponential covariance of data points, as Stan's gp_exp_quad_cov)
              ## (the points must be data: BayesMVP's C++ does not compile with parameters there; not declared
              ##  data, so that a call in the model's own functions, whose arguments are not data, is accepted)
              gp_exp_quad_cov = c(paste0("    matrix nico_gp_exp_quad_cov(array[] real x, real sigma, ",
                                         "real length_scale);"),
                                  paste0("    matrix nico_gp_exp_quad_cov(array[] real x1, ",
                                         "array[] real x2, real sigma, real length_scale);")),
              ## (the fused nodes of the Cox partial likelihood, written into a model only by NicoStan's fused
              ##  Cox writer, never renamed from a model's own call: fused_functions_written_only_by_fused_writers)
              log_sum_exp_of_affine_predictor = paste0(
                    "    real nico_log_sum_exp_of_affine_predictor(data matrix x, vector beta, real scale, ",
                    "vector z);"),
              dot_products_of_affine_predictor_and_log_cumulative_sum_exp = paste0(
                    "    real nico_dot_products_of_affine_predictor_and_log_cumulative_sum_exp(\n",
                    "          vector linear_weights, vector prefix_weights, data matrix x, vector beta,\n",
                    "          real scale, vector z, real log_offset);"),
              log_cumulative_sum_exp = "    vector nico_log_cumulative_sum_exp(vector x, real log_offset);")
        ##
        ## ---- std_normal maps to nico_normal with mu = 0 and sigma = 1: -------------------------------------------------------------
        ##
        std_normal_as_fused_normal_declarations <-  c(
              "    real nico_normal_lpdf(real y, real mu, real sigma);",
              "    real nico_normal_lpdf(vector y, real mu, real sigma);",
              "    real nico_normal_lpdf(array[] real y, real mu, real sigma);")
        ##
        ## ---- the model's own functions which BayesMVP replaces: only where the model defines a function of this --------------------
        ##
        ##      name with exactly this signature; its calls (not its definitions) become the BayesMVP function:
        ##
        model_functions_replaced_by_BayesMVP <-  data.frame(
              model_function          = c("log_Phi",
                                          "log_Phi_approx",
                                          "inv_Phi_approx_from_logit_prob",
                                          "inv_Phi_approx_from_prob",
                                          "inv_Phi_from_log_p",
                                          "inv_Phi_approx_from_prob",
                                          "inv_Phi_approx_from_logit_prob"),
              model_signature         = c("vector log_Phi(vector)",
                                          "vector log_Phi_approx(vector)",
                                          "vector inv_Phi_approx_from_logit_prob(vector)",
                                          "vector inv_Phi_approx_from_prob(vector)",
                                          "vector inv_Phi_from_log_p(vector, vector)",
                                          "real inv_Phi_approx_from_prob(real)",
                                          "real inv_Phi_approx_from_logit_prob(real)"),
              BayesMVP_function       = c("nico_log_Phi",
                                          "nico_log_Phi_approx",
                                          "nico_inv_Phi_approx_from_logit_prob",
                                          "nico_inv_Phi_approx",
                                          "nico_inv_Phi_from_log_p",
                                          "nico_inv_Phi_approx",
                                          "nico_inv_Phi_approx_from_logit_prob"),
              BayesMVP_declaration    = c("    vector nico_log_Phi(vector x);",
                                          "    vector nico_log_Phi_approx(vector x);",
                                          "    vector nico_inv_Phi_approx_from_logit_prob(vector logit_p);",
                                          "    vector nico_inv_Phi_approx(vector p);",
                                          "    vector nico_inv_Phi_from_log_p(vector log_p, vector log_1m_p);",
                                          "    real nico_inv_Phi_approx(real p);",
                                          "    real nico_inv_Phi_approx_from_logit_prob(real logit_p);"),
              stringsAsFactors = FALSE)
        ##
        ## ---- the math calls reported when they remain on Stan math: ----------------------------------------------------------------
        ##
        math_functions_whose_calls_left_on_Stan_math_are_reported <-  list(
              exp = 1, log = 1, log1p = 1, log1m = 1, expm1 = 1, log1p_exp = 1, log1m_exp = 1, inv_logit = 1,
              logit = 1, log_inv_logit = 1, log1m_inv_logit = 1, Phi = 1, Phi_approx = 1, inv_Phi = 1,
              std_normal_log_qf = 1, std_normal_lcdf = 1, tanh = 1, sinh = 1, cosh = 1, asinh = 1, acosh = 1,
              atanh = 1, sin = 1,
              sqrt = 1, floor = 1, fma = 3,
              square = 1,
              pow = 2, abs = 1, fmin = 2, fmax = 2, cumulative_sum = 1, lgamma = 1, digamma = 1,
              log_sum_exp = 2, log_softmax = c(), softmax = c(), log_diff_exp = 2, log_mix = 3, erf = 1, erfc = 1,
              gp_exp_quad_cov = c(),
              bernoulli_logit_rng = 1, normal_rng = 2, multi_normal_rng = 2, student_t_rng = 3, weibull_rng = 2,
              log_Phi = 1, log_Phi_approx = 1, inv_Phi_approx_from_logit_prob = 1, inv_Phi_approx_from_prob = 1,
              inv_Phi_from_log_p = 2, diag_pre_multiply = 2,
              ## (a data sparse matrix times a vector: a matrix-vector product, as in latent diffusion survival)
              csr_matrix_times_vector = c(),
              ## (the dot product of each row of two matrices: products and sums, as in the NMA ord_biv model)
              rows_dot_product = 2)
        math_functions_which_may_remain_on_Stan_math <-  c("sqrt", "square", "abs", "fmin", "fmax", "floor",
                                                           "pow", "fma", "cumulative_sum", "uniform_rng",
                                                           "bernoulli_rng",
                                                           "diag_pre_multiply", "csr_matrix_times_vector",
                                                           "rows_dot_product")
        structural_functions_that_may_remain_on_Stan_math <-  c(
              "size", "num_elements", "dims", "get_base1", "get_base2", "rows", "cols",
              "to_vector", "to_row_vector", "to_matrix", "to_array_1d", "to_array_2d", "to_int",
              "rep_array", "rep_vector", "rep_row_vector", "rep_matrix", "append_array", "append_row",
              "append_col", "head", "tail", "segment", "block", "row", "col", "diagonal", "diag_matrix",
              "linspaced_int_array", "linspaced_vector", "linspaced_row_vector", "sort_indices_asc", "min", "max",
              "sort_asc", "sort_indices_desc", "sum", "dot_product", "diag_post_multiply", "is_inf", "ceil",
              "add_diag", "dot_self", "reverse", "mean", "sd", "rank", "multiply_lower_tri_self_transpose",
              "is_nan")
        stan_functions_dispatching_user_functions <-  c("reduce_sum", "reduce_sum_static", "map_rect",
                                                         "nico_reduce_sum_static", "nico_map_rect")
        distributions_whose_calls_left_on_Stan_math_are_reported <-  list(
              std_normal       = list(n_arguments = 1, declaration_suffix = "_lpdf"),
              normal           = list(n_arguments = 3, declaration_suffix = "_lpdf"),
              lognormal        = list(n_arguments = 3, declaration_suffix = "_lpdf"),
              exponential      = list(n_arguments = 2, declaration_suffix = "_lpdf"),
              inv_gamma        = list(n_arguments = 3, declaration_suffix = "_lpdf"),
              beta             = list(n_arguments = 3, declaration_suffix = "_lpdf"),
              dirichlet        = list(n_arguments = 2, declaration_suffix = "_lpdf"),
              lkj_corr_cholesky = list(n_arguments = 2, declaration_suffix = "_lpdf"),
              student_t        = list(n_arguments = 4, declaration_suffix = "_lpdf"),
              weibull          = list(n_arguments = 3, declaration_suffix = "_lpdf"),
              bernoulli_logit  = list(n_arguments = 2, declaration_suffix = "_lpmf"),
              bernoulli_logit_glm = list(n_arguments = 4, declaration_suffix = "_lpmf"),
              bernoulli_logit_glm_with_a_tensor_product_basis = list(n_arguments = 7,
                                                                    declaration_suffix = "_lpmf"),
              bernoulli_logit_glm_with_coefficients_by_group = list(n_arguments = 5,
                                                                   declaration_suffix = "_lpmf"),
              ordered_logistic = list(n_arguments = 3, declaration_suffix = "_lpmf"),
              ordered_logistic_glm = list(n_arguments = 4, declaration_suffix = "_lpmf"),
              normal_id_glm = list(n_arguments = 5, declaration_suffix = "_lpdf"),
              binomial = list(n_arguments = 3, declaration_suffix = "_lpmf"),
              poisson_log_glm = list(n_arguments = 4, declaration_suffix = "_lpmf"),
              uniform = list(n_arguments = 3, declaration_suffix = "_lpdf"))
        ##
        ## ---- the BayesMVP header which defines each fused log density (in BayesMVP's header folder): -------------------------------
        ##
        fused_distribution_BayesMVP_header_files <-  c(
              bernoulli_logit_rng = "stan_external_bernoulli_logit_rng.hpp",
              normal_rng           = "stan_external_rng_functions.hpp",
              multi_normal_rng     = "stan_external_rng_functions.hpp",
              student_t_rng        = "stan_external_rng_functions.hpp",
              weibull_rng          = "stan_external_rng_functions.hpp",
              normal               = "stan_external_fused_normal_lpdf.hpp",
              lognormal            = "stan_external_lognormal_lpdf.hpp",
              exponential          = "stan_external_exponential_lpdf.hpp",
              inv_gamma            = "stan_external_inv_gamma_lpdf.hpp",
              lgamma               = "stan_external_lgamma.hpp",
              beta                 = "stan_external_beta_lpdf.hpp",
              dirichlet            = "stan_external_dirichlet_lpdf.hpp",
              lkj_corr_cholesky     = "stan_external_lkj_corr_cholesky_lpdf.hpp",
              erfc                 = "stan_external_erfc.hpp",
              sin                  = "stan_external_sin.hpp",
              sinh                 = "stan_external_hyperbolic_functions.hpp",
              cosh                 = "stan_external_hyperbolic_functions.hpp",
              asinh                = "stan_external_hyperbolic_functions.hpp",
              acosh                = "stan_external_hyperbolic_functions.hpp",
              atanh                = "stan_external_hyperbolic_functions.hpp",
              uniform              = "stan_external_uniform_lpdf.hpp",
              log_diff_exp         = "stan_external_log_diff_exp.hpp",
              std_normal_log_qf    = "stan_external_std_normal_log_qf.hpp",
              cumulative_sum       = "stan_external_cumulative_sum.hpp",
              cholesky_decompose   = "stan_external_cholesky_decompose.hpp",
              mdivide_left_tri_low = "stan_external_mdivide_tri_low.hpp",
              mdivide_right_tri_low = "stan_external_mdivide_tri_low.hpp",
              gp_normal_marginal = paste0("stan_external_gp_normal_marginal_lpdf_and_",
                                          "cholesky_factor_times_vector_with_a_choice_of_kernel.hpp"),
              gp_cholesky_factor_of_covariance_plus_diagonal_times_vector = paste0(
                    "stan_external_gp_normal_marginal_lpdf_and_",
                    "cholesky_factor_times_vector_with_a_choice_of_kernel.hpp"),
              student_t            = "stan_external_fused_student_t_lpdf.hpp",
              weibull              = "stan_external_fused_weibull_lpdf.hpp",
              ordered_logistic     = "stan_external_fused_ordered_logistic_lpmf.hpp",
              bernoulli_logit      = "stan_external_fused_bernoulli_logit_lpmf.hpp",
              bernoulli_logit_glm  = "stan_external_fused_bernoulli_logit_and_probit_glm_lpmf.hpp",
              poisson_log_glm      = "stan_external_fused_poisson_log_glm_lpmf.hpp",
              bernoulli_logit_glm_with_a_tensor_product_basis =
                    "stan_external_fused_bernoulli_glm_with_a_tensor_product_basis_lpmf.hpp",
              ordered_logistic_glm = "stan_external_fused_ordered_logistic_glm_lpmf.hpp",
              dot_product_exp_of_affine_predictor_at_quadrature_nodes_by_column =
                    "stan_external_dot_product_exp_of_affine_predictor_by_column.hpp",
              normal_id_glm        = paste0("stan_external_fused_student_t_id_glm_weibull_log_glm_",
                                            "and_bernoulli_logit_glm_by_group.hpp"),
              student_t_id_glm     = paste0("stan_external_fused_student_t_id_glm_weibull_log_glm_",
                                            "and_bernoulli_logit_glm_by_group.hpp"),
              weibull_log_glm      = paste0("stan_external_fused_student_t_id_glm_weibull_log_glm_",
                                            "and_bernoulli_logit_glm_by_group.hpp"),
              bernoulli_logit_glm_with_coefficients_by_group = paste0(
                    "stan_external_fused_student_t_id_glm_weibull_log_glm_",
                    "and_bernoulli_logit_glm_by_group.hpp"),
              binomial             = "stan_external_binomial_lpmf.hpp",
              log_sum_exp          = "stan_external_log_sum_exp_with_nico_exponential.hpp",
              log_sum_exp_2d       = "stan_external_log_sum_exp_with_nico_exponential.hpp",
              normal_log_scale     = "stan_external_fused_normal_log_scale_lpdf.hpp",
              AR1_recursion        = "stan_external_AR1_path.hpp",
              finite_mixture_of_multivariate_ordered_probit_GHK_log_likelihood = paste0(
                    "stan_external_finite_mixture_of_multivariate_ordered_probit_GHK_log_likelihood.hpp"),
              finite_mixture_of_multivariate_ordered_probit_GHK_log_likelihood_in_tiles_of_subjects = paste0(
                    "stan_external_finite_mixture_of_multivariate_ordered_probit_GHK_",
                    "log_likelihood_in_tiles_of_subjects.hpp"),
              sum_of_exponentials_by_group = "stan_external_sum_of_exponentials_by_group.hpp",
              two_compartment_first_order_input_exponents_and_coefficients =
                    "stan_external_two_compartment_first_order_input_exponents_and_coefficients.hpp",
              rows_of_mean_plus_random_effects_times_transposed_loadings =
                    "stan_external_rows_of_mean_plus_random_effects_times_transposed_loadings.hpp",
              proportional_hazards_log_likelihood_with_a_piecewise_linear_baseline_hazard_on_a_grid = paste0(
                    "stan_external_proportional_hazards_log_likelihood_with_a_piecewise_linear_baseline_",
                    "hazard_on_a_grid.hpp"),
              binomial_counts_above_successive_cutpoints_of_a_cumulative_link =
                    "stan_external_binomial_counts_above_successive_cutpoints_of_a_cumulative_link.hpp",
              induced_dirichlet_prior_of_ordered_cutpoints =
                    "stan_external_induced_dirichlet_prior_of_ordered_cutpoints.hpp",
              log_det_jacobian_of_cutpoints_from_raw_differences =
                    "stan_external_log_det_jacobian_of_cutpoints_from_raw_differences.hpp",
              gp_exp_quad_cov      = "stan_external_gp_exp_quad_cov.hpp",
              log_sum_exp_of_affine_predictor = "stan_external_affine_block_reductions.hpp",
              dot_products_of_affine_predictor_and_log_cumulative_sum_exp =
                    "stan_external_fused_affine_compositions.hpp",
              log_cumulative_sum_exp = "stan_external_log_cumulative_sum_exp.hpp")
        ##
        return(list( elementwise_functions                     = elementwise_functions,
                     elementwise_function_argument_types       = elementwise_function_argument_types,
                     fused_distribution_declarations           = fused_distribution_declarations,
                     fused_distribution_BayesMVP_header_files  = fused_distribution_BayesMVP_header_files,
                     std_normal_as_fused_normal_declarations   = std_normal_as_fused_normal_declarations,
                     model_functions_replaced_by_BayesMVP      = model_functions_replaced_by_BayesMVP,
                     math_functions_whose_calls_left_on_Stan_math_are_reported =
                           math_functions_whose_calls_left_on_Stan_math_are_reported,
                     math_functions_which_may_remain_on_Stan_math =
                           math_functions_which_may_remain_on_Stan_math,
                     structural_functions_that_may_remain_on_Stan_math =
                           structural_functions_that_may_remain_on_Stan_math,
                     stan_functions_dispatching_user_functions =
                           stan_functions_dispatching_user_functions,
                     distributions_whose_calls_left_on_Stan_math_are_reported =
                           distributions_whose_calls_left_on_Stan_math_are_reported))

}




##
## ---- the instruction-set macros the compiler defines for this CPU ------------------------------------------------------------------
##
#' The instruction-set macros (__AVX2__, __FMA__, __AVX512F__, __AVX512VL__, __AVX512DQ__, __ARM_NEON) which the
#' C++ compiler R uses defines for this CPU with -march=native, i.e. the instruction sets every model built here
#' gets
#'
#' Works on Linux, macOS and Windows (Rtools), on x86-64 and on ARM, because it asks the compiler itself (the one
#' R CMD config CXX gives) rather than reading an operating-system file; when the compiler does not accept
#' -march=native (some clang versions on ARM), it is asked without it, which still reports __ARM_NEON on ARM.
#' @return A character vector of the macro names defined (possibly empty).
#' @keywords internal
##
## ---- The function ------------------------------------------------------------------------------------------------------------------
##
R_fn_instruction_set_macros_defined_by_the_compiler_for_this_CPU <-  function() {

        macros_asked <-  c("__AVX2__", "__FMA__", "__AVX512F__", "__AVX512VL__", "__AVX512DQ__", "__ARM_NEON")
        R_binary <-  file.path(R.home("bin"), "R")
        compiler <-  tryCatch(suppressWarnings(system2(command = R_binary,
                                                       args = c("CMD", "config", "CXX"),
                                                       stdout = TRUE,
                                                       stderr = FALSE)),
                              error = function(e) character(0))
        compiler <-  trimws(paste(compiler, collapse = " "))
        if (!nzchar(compiler)) return(character(0))
        compiler_words <-  strsplit(x = compiler, split = "[[:space:]]+")[[1]]
        empty_source_file <-  tempfile(fileext = ".cpp")
        writeLines(text = character(0), con = empty_source_file)
        on.exit(unlink(empty_source_file), add = TRUE)
        ##
        ## ---- The function ----------------------------------------------------------------------------------------------------------
        ##
        fn_macros_defined_with_flags <-  function(flags) {

              output <-  tryCatch(suppressWarnings(system2(command = compiler_words[1],
                                                           args = c(compiler_words[-1], flags, "-dM", "-E",
                                                                    shQuote(empty_source_file)),
                                                           stdout = TRUE,
                                                           stderr = FALSE)),
                                  error = function(e) character(0))
              if (!is.null(attr(output, "status")) && attr(output, "status") != 0) return(NULL)
              macros_defined <-  sub(pattern = "^#define[[:space:]]+([A-Za-z0-9_]+).*$", replacement = "\\1",
                                     x = grep(pattern = "^#define", x = output, value = TRUE))
              return(macros_asked[macros_asked %in% macros_defined])

        }
        macros_defined <-  fn_macros_defined_with_flags(flags = "-march=native")
        if (is.null(macros_defined)) macros_defined <-  fn_macros_defined_with_flags(flags = character(0))
        if (is.null(macros_defined)) return(character(0))
        return(macros_defined)

}




##
## ---- the instruction sets of the SIMD (nico_) functions which this CPU supports ----------------------------------------------------
##
#' The instruction sets of NicoSIMD's (and BayesMVP's) SIMD functions which this CPU supports
#'
#' Asked of the C++ compiler R uses (R_fn_instruction_set_macros_defined_by_the_compiler_for_this_CPU), so that
#' it works on Linux, macOS and Windows, x86-64 and ARM; when the compiler gives no answer, the operating
#' system is asked (Linux: the flags line of /proc/cpuinfo; macOS: sysctl machdep.cpu.features and
#' leaf7_features; Windows:
#' the PROCESSOR_ARCHITECTURE variable, which tells x86-64 from ARM only). AVX2 needs avx2 and fma; AVX-512 needs
#' avx2, fma, avx512f, avx512vl and avx512dq (the instruction-set flags the AVX version of a Stan model is
#' compiled with); on 64-bit ARM the AVX2 functions run as NEON through SIMDe, so ARM with NEON counts as AVX2.
#' @return A named logical vector: BayesMVP_AVX2 and BayesMVP_AVX512.
#' @keywords internal
##
## ---- The function ------------------------------------------------------------------------------------------------------------------
##
R_fn_BayesMVP_AVX_instruction_sets_supported_by_this_CPU <-  function() {

        macros_defined <-  R_fn_instruction_set_macros_defined_by_the_compiler_for_this_CPU()
        ##
        ## ---- Check the condition ---------------------------------------------------------------------------------------------------
        ##
        if (length(macros_defined) > 0) {

              ##
              ## ---- Check the condition ---------------------------------------------------------------------------------------------
              ##
              if ("__ARM_NEON" %in% macros_defined) {

                    return(c( BayesMVP_AVX2 = TRUE, BayesMVP_AVX512 = FALSE))

              }
              return(c( BayesMVP_AVX2    = all(c("__AVX2__", "__FMA__") %in% macros_defined),
                        BayesMVP_AVX512  = all(c("__AVX2__", "__FMA__", "__AVX512F__", "__AVX512VL__",
                                                 "__AVX512DQ__") %in% macros_defined)))

        }
        ##
        ## ---- the compiler gave no answer: the operating system is asked ------------------------------------------------------------
        ##
        CPU_flags <-  character(0)
        operating_system <-  Sys.info()[["sysname"]]
        ##
        ## ---- Check the condition ---------------------------------------------------------------------------------------------------
        ##
        if (identical(operating_system, "Linux") && file.exists("/proc/cpuinfo")) {

              cpuinfo_lines <-  readLines(con = "/proc/cpuinfo", warn = FALSE)
              first_flags_line <-  grep(pattern = "^(flags|Features)", x = cpuinfo_lines, value = TRUE)[1]
              ##
              ## ---- Check the condition ---------------------------------------------------------------------------------------------
              ##
              if (!is.na(first_flags_line)) {

                    CPU_flags <-  strsplit(x = sub(pattern = "^[A-Za-z]+[[:space:]]*:[[:space:]]*",
                                                   replacement = "", x = first_flags_line),
                                           split = "[[:space:]]+")[[1]]

              }

        ##
        ## ---- The alternative -------------------------------------------------------------------------------------------------------
        ##
        } else if (identical(operating_system, "Darwin")) {

              sysctl_output <-  tryCatch(suppressWarnings(system2(command = "sysctl",
                                                                  args = c("-n", "machdep.cpu.features",
                                                                           "machdep.cpu.leaf7_features",
                                                                           "hw.optional.neon"),
                                                                  stdout = TRUE,
                                                                  stderr = FALSE)),
                                         error = function(e) character(0))
              CPU_flags <-  tolower(strsplit(x = paste(sysctl_output, collapse = " "),
                                             split = "[[:space:]]+")[[1]])
              on_Apple_silicon <-  identical(Sys.info()[["machine"]], "arm64") ||
                                   identical(sysctl_output[length(sysctl_output)], "1")
              ##
              ## ---- Check the condition ---------------------------------------------------------------------------------------------
              ##
              if (on_Apple_silicon) {

                    CPU_flags <-  c(CPU_flags, "neon")

              }

        ##
        ## ---- The alternative -------------------------------------------------------------------------------------------------------
        ##
        } else if (identical(operating_system, "Windows")) {

              ##
              ## ---- Check the condition ---------------------------------------------------------------------------------------------
              ##
              if (grepl(pattern = "ARM", x = Sys.getenv("PROCESSOR_ARCHITECTURE"), ignore.case = TRUE)) {

                    CPU_flags <-  "neon"

              }

        }
        CPU_flags <-  tolower(CPU_flags)
        ##
        ## ---- Check the condition ---------------------------------------------------------------------------------------------------
        ##
        if (any(c("neon", "asimd") %in% CPU_flags)) {

              return(c( BayesMVP_AVX2 = TRUE, BayesMVP_AVX512 = FALSE))

        }
        ##
        ## ---- Check the condition ---------------------------------------------------------------------------------------------------
        ##
        if (length(CPU_flags) == 0) {

              stop(paste0("The instruction sets of this CPU could not be found: the C++ compiler R uses (R CMD ",
                          "config CXX) gave no answer to -march=native -dM -E, and the operating system gave no ",
                          "list of CPU flags. Set Stan_model_math_functions to \"NicoSIMD_AVX2\" or ",
                          "\"NicoSIMD_AVX512\" by hand."))

        }
        CPU_flags_needed_for_AVX2 <-    c("avx2", "fma")
        CPU_flags_needed_for_AVX512 <-  c("avx2", "fma", "avx512f", "avx512vl", "avx512dq")
        return(c( BayesMVP_AVX2    = all(CPU_flags_needed_for_AVX2 %in% CPU_flags),
                  BayesMVP_AVX512  = all(CPU_flags_needed_for_AVX512 %in% CPU_flags)))

}




#' Write a Stan model with NicoSIMD calls where stanc accepts their declarations
#'
#' Required and unclassified calls left on Stan math block writing. Listed basic and structural calls remain.
#' Comments and strings are left as they are. The file is written through a temporary file renamed over it, so
#' no reader sees a partly written model.
#' @param source_Stan_model_file_path The Stan model.
#' @param AVX_Stan_model_file_path The AVX version written.
#' @param first_line_of_AVX_Stan_model The first line of the AVX version (a Stan comment recording the md5 of the
#'   Stan model and of the declarations offered).
#' @param elementwise_functions The elementwise functions offered.
#' @param fused_distribution_declarations The Stan declarations of the fused log densities offered (a named list;
#'   each name is the Stan distribution replaced).
#' @param stanc_file_path The stanc compiler (BridgeStan's, which compiles the model).
#' @param math_functions_which_may_remain_on_Stan_math The listed basic functions that remain on Stan math.
#' @param structural_functions_that_may_remain_on_Stan_math The listed Stan container and shape functions.
#' @param stan_functions_dispatching_user_functions The Stan calls that dispatch user-defined functions.
#' @param stanc_include_paths_arguments The --include-paths arguments of the model's stanc arguments (NULL: none).
#' @param elementwise_function_argument_types The argument types for which each elementwise function is declared.
#' @param model_functions_replaced_by_BayesMVP The model's own functions which BayesMVP replaces (a data frame).
#' @param std_normal_as_fused_normal_declarations The declarations of nico_normal_lpdf for std_normal (NULL:
#'   none).
#' @param math_functions_whose_calls_left_on_Stan_math_are_reported The functions (a named list: the numbers of
#'   arguments of their calls on reals).
#' @param distributions_whose_calls_left_on_Stan_math_are_reported The distributions (a named list).
#' @param report_file_path_of_calls_left_on_Stan_math The report written (NULL: none).
#' @return The NicoSIMD functions used (the elementwise functions and fused distributions, by their Stan names).
#' @keywords internal
R_fn_write_AVX_version_of_Stan_model_with_BayesMVP_functions <-  function(
                    source_Stan_model_file_path,
                    AVX_Stan_model_file_path,
                    first_line_of_AVX_Stan_model,
                    elementwise_functions,
                    fused_distribution_declarations,
                    stanc_file_path,
                    math_functions_which_may_remain_on_Stan_math,
                    structural_functions_that_may_remain_on_Stan_math,
                    stan_functions_dispatching_user_functions,
                    stanc_include_paths_arguments                               = NULL,
                    elementwise_function_argument_types                         = "vector",
                    model_functions_replaced_by_BayesMVP                        = NULL,
                    std_normal_as_fused_normal_declarations                     = NULL,
                    math_functions_whose_calls_left_on_Stan_math_are_reported   = NULL,
                    distributions_whose_calls_left_on_Stan_math_are_reported    = NULL,
                    report_file_path_of_calls_left_on_Stan_math                 = NULL
##
## ---- The function ------------------------------------------------------------------------------------------------------------------
##
) {

        source_Stan_model_file_name <-  basename(source_Stan_model_file_path)
        fused_distributions <-  names(fused_distribution_declarations)
        ## (fused functions which only NicoStan's fused writers write into a model: declared and given their
        ##  headers, never renamed from a call of the model's own function of the same name, e.g. the Cox model's
        ##  log_cumulative_sum_exp, whose non-fused cells keep exp, cumulative_sum and log as separate calls)
        fused_functions_written_only_by_fused_writers <-  c(
              "log_sum_exp_of_affine_predictor",
              "dot_products_of_affine_predictor_and_log_cumulative_sum_exp",
              "log_cumulative_sum_exp")
        ## (cheap functions, additions only, which stay on Stan math in every nico::math version: never renamed
        ##  from a call, e.g. cumulative_sum)
        cheap_functions_never_renamed_from_calls <-  c("cumulative_sum")
        fused_distributions_renamed_from_calls <-  setdiff(fused_distributions,
                                                           c(fused_functions_written_only_by_fused_writers,
                                                             cheap_functions_never_renamed_from_calls))
        text <-  paste(readLines(con = source_Stan_model_file_path, warn = FALSE), collapse = "\n")
        ##
        ## ---- comments and strings are set aside (replaced by numbered markers) before the rewrite: ---------------------------------
        ##
        protected_pattern <-  "\"(?:[^\"\\\\]|\\\\.)*\"|/\\*[\\s\\S]*?\\*/|//[^\\n]*"
        protected <-  regmatches(x = text,
                                 m = gregexpr(pattern = protected_pattern, text = text, perl = TRUE))[[1]]
        ##
        ## ---- Each item -------------------------------------------------------------------------------------------------------------
        ##
        for (i in seq_along(protected)) {

              text <-  sub(pattern = protected[i], replacement = paste0("\001", i, "\001"), x = text,
                           fixed = TRUE)

        }
        ## (only the markers which are in the code are restored, so that each line of a long model is restored
        ##  quickly)
        ##
        ## ---- The function ----------------------------------------------------------------------------------------------------------
        ##
        fn_restore <-  function(code) {

              markers <-  regmatches(x = code, m = gregexpr(pattern = "\001[0-9]+\001", text = code))[[1]]
              ##
              ## ---- Each item -------------------------------------------------------------------------------------------------------
              ##
              for (marker in markers) {

                    i <-  as.numeric(gsub(pattern = "\001", replacement = "", x = marker, fixed = TRUE))
                    code <-  sub(pattern = marker, replacement = protected[i], x = code, fixed = TRUE)

              }
              return(code)

        }
        ##
        ## ---- the model's own functions which BayesMVP replaces: those which the model defines with exactly the ---------------------
        ##
        ##      signature of model_functions_replaced_by_BayesMVP (e.g. vector log_Phi(vector x)):
        ##
        ##
        ## ---- The function ----------------------------------------------------------------------------------------------------------
        ##
        fn_pattern_of_definition <-  function(model_signature) {

              ## (e.g. "vector inv_Phi_from_log_p(vector, vector)": the return type, the name and the argument
              ##  types; each argument may be data, and has a name)
              return_type <-  sub(pattern = "^([A-Za-z_]+)\\s.*$", replacement = "\\1", x = model_signature)
              function_name <-  sub(pattern = "^[A-Za-z_]+\\s+([A-Za-z0-9_]+)\\(.*$", replacement = "\\1",
                                    x = model_signature)
              argument_types <-  strsplit(x = sub(pattern = "^.*\\((.*)\\)$", replacement = "\\1",
                                                  x = model_signature),
                                          split = ",\\s*")[[1]]
              arguments_pattern <-  paste(paste0("(data\\s+)?", argument_types, "\\s+[A-Za-z][A-Za-z0-9_]*"),
                                          collapse = "\\s*,\\s*")
              return(paste0("(?<![A-Za-z0-9_.])", return_type, "\\s+", function_name, "\\s*\\(\\s*",
                            arguments_pattern, "\\s*\\)\\s*\\{"))

        }
        ##
        ## ---- Check the condition ---------------------------------------------------------------------------------------------------
        ##
        if (is.null(model_functions_replaced_by_BayesMVP)) {

              model_functions_replaced_by_BayesMVP <-  data.frame( model_function        = character(0),
                                                                   model_signature       = character(0),
                                                                   BayesMVP_function     = character(0),
                                                                   BayesMVP_declaration  = character(0),
                                                                   stringsAsFactors = FALSE)

        }
        model_function_is_defined_with_its_signature <-  vapply(
              X = model_functions_replaced_by_BayesMVP$model_signature,
              ##
              ## ---- The function ----------------------------------------------------------------------------------------------------
              ##
              FUN = function(model_signature) {

                    grepl(pattern = fn_pattern_of_definition(model_signature), x = text, perl = TRUE)

              },

              FUN.VALUE = logical(1))
        model_functions_offered <-  model_functions_replaced_by_BayesMVP[
                                          model_function_is_defined_with_its_signature, , drop = FALSE]
        ##
        ## ---- (4) the declarations (every one offered whilst stanc decides; only those used at the end): ----------------------------
        ##
        ##
        ## ---- The function ----------------------------------------------------------------------------------------------------------
        ##
        fn_declarations <-  function(used) {

              used_elementwise <-  intersect(used, elementwise_functions)
              used_fused <-  intersect(used, fused_distributions)
              used_model_functions <-  intersect(used, model_functions_offered$model_function)
              declarations <-  "    real bmvp_simd_lanes();"
              ##
              ## ---- Check the condition ---------------------------------------------------------------------------------------------
              ##
              if (length(used_elementwise) > 0) {

                    ## (each for every argument type offered, e.g. vector, row_vector and matrix)
                    ##
                    ## ---- Each item -------------------------------------------------------------------------------------------------
                    ##
                    for (argument_type in elementwise_function_argument_types) {

                          declarations <-  c(declarations,
                                             paste0("    ", argument_type, " nico_", used_elementwise, "(",
                                                    argument_type, " x);"))

                    }

              }
              declarations <-  c(declarations,
                                 unlist(fused_distribution_declarations[used_fused], use.names = FALSE))
              declarations <-  c(declarations,
                                 model_functions_offered$BayesMVP_declaration[
                                       model_functions_offered$model_function %in% used_model_functions])
              return(unique(declarations))

        }
        ## (extra_declarations: the declarations of the probes of (0) and (5), below)
        ##
        ## ---- The function ----------------------------------------------------------------------------------------------------------
        ##
        fn_assemble <-  function(code, used, extra_declarations = NULL) {

              registered_declarations <-  fn_declarations(used)
              code_without_whitespace <-  gsub(pattern = "[[:space:]]+",
                                                replacement = "",
                                                x = code,
                                                perl = TRUE)
              registered_declaration_is_present <-  vapply(
                    X = registered_declarations,
                    ##
                    ## ---- The function ----------------------------------------------------------------------------------------------
                    ##
                    FUN = function(declaration) {

                          declaration_without_whitespace <-  gsub(pattern = "[[:space:]]+",
                                                                   replacement = "",
                                                                   x = declaration,
                                                                   perl = TRUE)
                          return(grepl(pattern = declaration_without_whitespace,
                                       x = code_without_whitespace,
                                       fixed = TRUE))

                    },
                    FUN.VALUE = logical(1))
              registered_declarations <-  registered_declarations[!registered_declaration_is_present]
              declarations <-  paste(c(registered_declarations, extra_declarations), collapse = "\n")
              ##
              ## ---- Check the condition ---------------------------------------------------------------------------------------------
              ##
              if (grepl(pattern = "(^|\\n)\\s*functions\\s*\\{", x = code, perl = TRUE)) {

                    code <-  sub(pattern = "((^|\\n)\\s*functions\\s*\\{)",
                                 replacement = paste0("\\1\n", declarations),
                                 x = code, perl = TRUE)

              ##
              ## ---- The alternative -------------------------------------------------------------------------------------------------
              ##
              } else {

                    code <-  paste0("functions {\n", declarations, "\n}\n", code)

              }
              ##
              ## ---- Check the condition ---------------------------------------------------------------------------------------------
              ##
              if (grepl(pattern = "(^|\\n)\\s*generated\\s+quantities\\s*\\{", x = code, perl = TRUE)) {

                    code <-  sub(pattern = "((^|\\n)\\s*generated\\s+quantities\\s*\\{)",
                                 replacement = "\\1\n    real simd_lanes = bmvp_simd_lanes();",
                                 x = code, perl = TRUE)

              ##
              ## ---- The alternative -------------------------------------------------------------------------------------------------
              ##
              } else {

                    code <-  paste0(code, "\ngenerated quantities {\n",
                                    "    real simd_lanes = bmvp_simd_lanes();\n}\n")

              }
              return(code)

        }
        ##
        ## ---- The function ----------------------------------------------------------------------------------------------------------
        ##
        fn_used <-  function(code) {

              used <-  c()
              ##
              ## ---- Each item -------------------------------------------------------------------------------------------------------
              ##
              for (elementwise_function in elementwise_functions) {

                    if (grepl(pattern = paste0("(?<![A-Za-z0-9_.])nico_", elementwise_function, "(?=\\s*\\()"),
                              ##
                              ## ---- Check the condition -----------------------------------------------------------------------------
                              ##
                              x = code, perl = TRUE)) {

                          used <-  c(used, elementwise_function)

                    }

              }
              ##
              ## ---- Each item -------------------------------------------------------------------------------------------------------
              ##
              for (distribution in fused_distributions) {

                    if (grepl(pattern = paste0("(?<![A-Za-z0-9_.])nico_", distribution,
                                               "(_lpdf|_lupdf|_lpmf|_lupmf|_lccdf)?(?=\\s*\\()"),
                              ##
                              ## ---- Check the condition -----------------------------------------------------------------------------
                              ##
                              x = code, perl = TRUE)) {

                          used <-  c(used, distribution)

                    }

              }
              ##
              ## ---- Each item -------------------------------------------------------------------------------------------------------
              ##
              for (i in seq_len(nrow(model_functions_offered))) {

                    if (grepl(pattern = paste0("(?<![A-Za-z0-9_.])", model_functions_offered$BayesMVP_function[i],
                                               "(?=\\s*\\()"),
                              ##
                              ## ---- Check the condition -----------------------------------------------------------------------------
                              ##
                              x = code, perl = TRUE)) {

                          used <-  c(used, model_functions_offered$model_function[i])

                    }

              }
              return(used)

        }
        ##
        ## ---- stanc, on code with the declarations of the functions used (and of the probes of (0) and (5)); the --------------------
        ##
        ##      calls of named functions in the code; the lines of positions in the code; the end of a bracket:
        ##
        candidate_file <-  tempfile(pattern = "AVX_version_candidate_", fileext = ".stan")
        stanc_output_file <-  tempfile(pattern = "AVX_version_candidate_", fileext = ".hpp")
        ##
        ## ---- The function ----------------------------------------------------------------------------------------------------------
        ##
        fn_stanc_output <-  function(code) {

              writeLines(text = R_fn_Stan_parallel_call_names(code = fn_restore(code)), con = candidate_file)
              return(suppressWarnings(system2( command = stanc_file_path,
                                               args    = c("--allow-undefined",
                                                           stanc_include_paths_arguments,
                                                           paste0("--o=", shQuote(stanc_output_file)),
                                                           shQuote(candidate_file)),
                                               stdout  = TRUE,
                                               stderr  = TRUE)))

        }
        ## (the calls of the functions named, not their definitions (a name after a return type), each with its
        ##  position, name and suffix; with_distribution_suffixes: a distribution's name followed by _lpdf,
        ##  _lupdf, _lpmf, _lupmf, _lccdf, _lcdf or, in a ~ statement, nothing)
        ##
        ## ---- The function ----------------------------------------------------------------------------------------------------------
        ##
        fn_calls_of_functions <-  function(code, names_of_functions, with_distribution_suffixes) {

              ##
              ## ---- Check the condition ---------------------------------------------------------------------------------------------
              ##
              suffix_pattern <-  if (with_distribution_suffixes) {

                    "(_lpdf|_lupdf|_lpmf|_lupmf|_lccdf|_lcdf)?"

              } else "()"
              pattern <-  paste0("(?<![A-Za-z0-9_.])(", paste(names_of_functions, collapse = "|"), ")",
                                 suffix_pattern, "(?=\\s*\\()")
              matches <-  gregexpr(pattern = pattern, text = code, perl = TRUE)[[1]]
              ##
              ## ---- Check the condition ---------------------------------------------------------------------------------------------
              ##
              if (matches[1] == -1) {

                    return(data.frame( position = numeric(0), name = character(0), suffix = character(0),
                                       in_tilde_statement = logical(0), stringsAsFactors = FALSE))

              }
              starts_of_names <-  attr(matches, "capture.start")[, 1]
              starts_of_suffixes <-  attr(matches, "capture.start")[, 2]
              names_found <-  substring(text = code, first = starts_of_names,
                                        last = starts_of_names + attr(matches, "capture.length")[, 1] - 1)
              suffixes_found <-  substring(text = code, first = starts_of_suffixes,
                                           last = starts_of_suffixes + attr(matches, "capture.length")[, 2] - 1)
              text_before_call <-  substring(text = code, first = pmax(1, matches - 40), last = matches - 1)
              text_before_call_for_tilde <-  gsub(pattern = "\001[0-9]+\001", replacement = "",
                                                   x = text_before_call)
              is_definition <-  grepl(pattern = "(vector|row_vector|matrix|real|int|void)\\s+$",
                                      x = text_before_call)
              calls <-  data.frame( position            = as.numeric(matches),
                                    name                = names_found,
                                    suffix              = suffixes_found,
                                    in_tilde_statement  = grepl(pattern = "~\\s*$",
                                                                 x = text_before_call_for_tilde),
                                    stringsAsFactors = FALSE)
              return(calls[!is_definition, , drop = FALSE])

        }
        fn_calls_of_all_functions <-  function(
              code
        ##
        ## ---- The function ----------------------------------------------------------------------------------------------------------
        ##
        ) {

              matches <-  gregexpr(pattern = "(?<![A-Za-z0-9_.])([A-Za-z_][A-Za-z0-9_]*)(?=\\s*\\()",
                                   text = code, perl = TRUE)[[1]]
              ##
              ## ---- Check the condition ---------------------------------------------------------------------------------------------
              ##
              if (matches[1] == -1) {

                    return(data.frame(position = numeric(0), name = character(0), suffix = character(0),
                                      in_tilde_statement = logical(0), stringsAsFactors = FALSE))

              }
              starts <-  attr(matches, "capture.start")[, 1]
              lengths <-  attr(matches, "capture.length")[, 1]
              names_found <-  substring(code, starts, starts + lengths - 1)
              text_before_call <-  substring(code, pmax(1, matches - 80), matches - 1)
              text_before_call_for_tilde <-  gsub(pattern = "\001[0-9]+\001", replacement = "",
                                                   x = text_before_call)
              function_type_pattern <-  paste0(
                    "((array\\s*\\[[^]]*\\]\\s*)*(real|int|vector|row_vector|matrix|",
                    "complex|complex_vector|complex_row_vector|complex_matrix|void|tuple))\\s+$")
              is_definition <-  grepl(pattern = function_type_pattern, x = text_before_call, perl = TRUE)
              calls <-  data.frame(position = as.numeric(matches), name = names_found, suffix = "",
                                   in_tilde_statement = grepl(pattern = "~\\s*$",
                                                              x = text_before_call_for_tilde),
                                   stringsAsFactors = FALSE)
              return(calls[!is_definition, , drop = FALSE])

        }
        fn_names_of_stan_function_definitions <-  function(
              code
        ##
        ## ---- The function ----------------------------------------------------------------------------------------------------------
        ##
        ) {

              pattern <-  paste0("(?<![A-Za-z0-9_.])(?:array\\s*\\[[^]]*\\]\\s*)*",
                                 "(?:real|int|vector|row_vector|matrix|complex|complex_vector|",
                                 "complex_row_vector|complex_matrix|void|tuple)\\s+",
                                 "([A-Za-z_][A-Za-z0-9_]*)\\s*\\([^;{}]*\\)\\s*\\{")
              matches <-  gregexpr(pattern = pattern, text = code, perl = TRUE)[[1]]
              if (matches[1] == -1) return(character(0))
              starts <-  attr(matches, "capture.start")[, 1]
              lengths <-  attr(matches, "capture.length")[, 1]
              return(substring(code, starts, starts + lengths - 1))

        }
        ##
        ## ---- NicoStan's nuisance readers, only declared in the functions block: the model's own, not Stan math ---------------------
        ##
        ##      (any other declared function stays unclassified):
        fn_names_of_stan_function_declarations <-  function(
              code
        ##
        ## ---- The function ----------------------------------------------------------------------------------------------------------
        ##
        ) {

              pattern <-  paste0("(?<![A-Za-z0-9_.])(?:array\\s*\\[[^]]*\\]\\s*)*",
                                 "(?:real|int|vector|row_vector|matrix|complex|complex_vector|",
                                 "complex_row_vector|complex_matrix|void|tuple)\\s+",
                                 "([A-Za-z_][A-Za-z0-9_]*)\\s*\\([^;{}]*\\)\\s*;")
              matches <-  gregexpr(pattern = pattern, text = code, perl = TRUE)[[1]]
              if (matches[1] == -1) return(character(0))
              starts <-  attr(matches, "capture.start")[, 1]
              lengths <-  attr(matches, "capture.length")[, 1]
              declared_names <-  substring(code, starts, starts + lengths - 1)
              return(declared_names[startsWith(declared_names, "nicostan_nuisance_")])

        }
        ## (the line in the Stan model of each position in the code: the lines of the code before it, plus the
        ##  lines of the comments and strings set aside before it)
        ##
        ## ---- The function ----------------------------------------------------------------------------------------------------------
        ##
        fn_lines_in_Stan_model_of_positions <-  function(code, positions) {

              newline_positions <-  gregexpr(pattern = "\n", text = code, fixed = TRUE)[[1]]
              newline_positions <-  newline_positions[newline_positions > 0]
              lines <-  1 + findInterval(x = positions - 1, vec = newline_positions)
              markers <-  gregexpr(pattern = "\001[0-9]+\001", text = code)[[1]]
              ##
              ## ---- Check the condition ---------------------------------------------------------------------------------------------
              ##
              if (markers[1] != -1) {

                    marker_numbers <-  as.numeric(gsub(pattern = "\001", replacement = "",
                                                       x = regmatches(x = code, m = list(markers))[[1]],
                                                       fixed = TRUE))
                    lines_in_markers <-  nchar(gsub(pattern = "[^\n]", replacement = "",
                                                    x = protected[marker_numbers]))
                    markers_before_each_position <-  findInterval(x = positions - 1, vec = as.numeric(markers))
                    lines <-  lines + c(0, cumsum(lines_in_markers))[markers_before_each_position + 1]

              }
              return(lines)

        }
        fn_position_of_matching_closing_bracket <-  function( code,
                                                              position_of_opening_bracket,
                                                              opening,
                                                              closing
        ##
        ## ---- The function ----------------------------------------------------------------------------------------------------------
        ##
        ) {

              characters <-  strsplit(x = substr(code, position_of_opening_bracket, nchar(code)), split = "")[[1]]
              depth <-  cumsum((characters == opening) - (characters == closing))
              return(position_of_opening_bracket - 1 + which(depth == 0)[1])

        }
        ## (the probes: each call k is renamed bmvp_probe_<k>_<its name> and declared with the signatures of
        ##  probe_declarations_of_each_call[[k]]; stanc rejects one call at a time, naming its argument types, and
        ##  that call gets its own name back, until stanc accepts the code. Returns the argument types of each
        ##  call stanc rejected, as stanc names them (e.g. "vector", "matrix, real"), and "" for every call it
        ##  accepted)
        ##
        ## ---- The function ----------------------------------------------------------------------------------------------------------
        ##
        fn_code_with_probe_names <-  function(code, calls, call_is_probed) {

              pieces <-  c()
              previous_end <-  0
              ##
              ## ---- Each item -------------------------------------------------------------------------------------------------------
              ##
              for (k in which(call_is_probed)) {

                    pieces <-  c(pieces, substr(code, previous_end + 1, calls$position[k] - 1),
                                 paste0("bmvp_probe_", k, "_"))
                    previous_end <-  calls$position[k] - 1

              }
              return(paste0(paste(pieces, collapse = ""), substr(code, previous_end + 1, nchar(code))))

        }
        fn_argument_types_of_calls_which_stanc_rejects <-  function(code, calls, probe_declarations_of_each_call,
                                                                    ##
                                                                    ## ---- The function ----------------------------------------------
                                                                    ##
                                                                    used) {

              argument_types <-  rep("", nrow(calls))
              call_is_probed <-  rep(TRUE, nrow(calls))
              ##
              ## ---- Each item -------------------------------------------------------------------------------------------------------
              ##
              for (attempt in seq_len(nrow(calls) + 1)) {

                    stanc_output <-  fn_stanc_output(code = fn_assemble(
                          code                = fn_code_with_probe_names(code = code, calls = calls,
                                                                         call_is_probed = call_is_probed),
                          used                = used,
                          extra_declarations  = unlist(probe_declarations_of_each_call[call_is_probed],
                                                       use.names = FALSE)))
                    ##
                    ## ---- Check the condition ---------------------------------------------------------------------------------------
                    ##
                    if (is.null(attr(stanc_output, "status")) || attr(stanc_output, "status") == 0) {

                          return(argument_types)

                    }
                    stanc_message <-  paste(stanc_output, collapse = "\n")
                    rejected_call <-  regmatches(x = stanc_message,
                                                 m = regexec(pattern = paste0("Ill-typed arguments supplied to ",
                                                                              "function 'bmvp_probe_([0-9]+)_",
                                                                              "[A-Za-z0-9_]*':[[:space:]]*",
                                                                              "\\(([^)]*)\\)"),
                                                             text = stanc_message))[[1]]
                    ##
                    ## ---- Check the condition ---------------------------------------------------------------------------------------
                    ##
                    if (length(rejected_call) < 3) {

                          stop(paste0("stanc rejected the AVX version of ", source_Stan_model_file_name,
                                      " for another reason than the argument types of a call (whilst finding ",
                                      "the argument types of the calls):\n", stanc_message))

                    }
                    k <-  as.numeric(rejected_call[2])
                    argument_types[k] <-  rejected_call[3]
                    call_is_probed[k] <-  FALSE

              }
              stop(paste0("stanc did not accept the probes of ", source_Stan_model_file_name, "."))

        }
        ##
        ## ---- std_normal calls use nico_normal with mu = 0 and sigma = 1: -----------------------------------------------------------
        ##
        ##
        ## ---- Check the condition ---------------------------------------------------------------------------------------------------
        ##
        if (("normal" %in% fused_distributions) && (length(std_normal_as_fused_normal_declarations) > 0)) {

              std_normal_calls <-  fn_calls_of_functions(code = text, names_of_functions = "std_normal",
                                                         with_distribution_suffixes = TRUE)
              std_normal_calls <-  std_normal_calls[(std_normal_calls$suffix %in% c("_lpdf", "_lupdf")) |
                                                    (std_normal_calls$suffix == "" &
                                                     std_normal_calls$in_tilde_statement), , drop = FALSE]
              ##
              ## ---- Check the condition ---------------------------------------------------------------------------------------------
              ##
              if (nrow(std_normal_calls) > 0) {

                    probe_declarations_of_each_call <-  lapply(X = seq_len(nrow(std_normal_calls)),
                                                               ##
                                                               ## ---- The function ---------------------------------------------------
                                                               ##
                                                               FUN = function(k) {

                                                                     paste0("    real bmvp_probe_", k,
                                                                            "_std_normal_lpdf(",
                                                                     c("real y", "vector y", "array[] real y"),
                                                                     ");")

                                                               })

                    argument_types_of_std_normal_calls_rejected <-
                          fn_argument_types_of_calls_which_stanc_rejects(
                          code                             = text,
                          calls                            = std_normal_calls,
                          probe_declarations_of_each_call  = probe_declarations_of_each_call,
                          used                             = c())
                    ## (from the last call to the first, so that the positions of the calls before it stay valid)
                    ##
                    ## ---- Each item -------------------------------------------------------------------------------------------------
                    ##
                    for (k in rev(which(argument_types_of_std_normal_calls_rejected == ""))) {

                          position_of_call <-  std_normal_calls$position[k]
                          position_of_opening_parenthesis <-  position_of_call - 1 +
                                regexpr(pattern = "(", text = substr(text, position_of_call, nchar(text)),
                                        fixed = TRUE)
                          position_of_closing_parenthesis <-  fn_position_of_matching_closing_bracket(
                                code = text, position_of_opening_bracket = position_of_opening_parenthesis,
                                opening = "(", closing = ")")
                          arguments_of_call <-  substr(text, position_of_opening_parenthesis + 1,
                                                       position_of_closing_parenthesis - 1)
                          ##
                          ## ---- Check the condition ---------------------------------------------------------------------------------
                          ##
                          call_with_BayesMVP_normal <-  if (std_normal_calls$suffix[k] == "") {

                                "nico_normal(0, 1)"

                          ##
                          ## ---- The alternative -------------------------------------------------------------------------------------
                          ##
                          } else {

                                paste0("nico_normal", std_normal_calls$suffix[k], "(", arguments_of_call,
                                       " | 0, 1)")

                          }
                          text <-  paste0(substr(text, 1, position_of_call - 1), call_with_BayesMVP_normal,
                                          substr(text, position_of_closing_parenthesis + 1, nchar(text)))

                    }

              }

        }
        ##
        ## ---- (1) and (2): Stan's functions become BayesMVP's (a name followed by "(", not part of a longer -------------------------
        ##
        ##      name):
        ##
        ##
        ## ---- Check the condition ---------------------------------------------------------------------------------------------------
        ##
        if (length(elementwise_functions) > 0) {

              text <-  gsub(pattern = paste0("(?<![A-Za-z0-9_.])(", paste(elementwise_functions, collapse = "|"),
                                             ")(?=\\s*\\()"),
                            replacement = "nico_\\1",
                            x = text,
                            perl = TRUE)

        }
        ##
        ## ---- Check the condition ---------------------------------------------------------------------------------------------------
        ##
        if (length(fused_distributions) > 0) {

              text <-  gsub(pattern = paste0("(?<![A-Za-z0-9_.])(",
                                             paste(fused_distributions_renamed_from_calls, collapse = "|"),
                                             ")(_lpdf|_lupdf|_lpmf|_lupmf)?(?=\\s*\\()"),
                            replacement = "nico_\\1\\2",
                            x = text,
                            perl = TRUE)

        }
        ##
        ## ---- Check the condition ---------------------------------------------------------------------------------------------------
        ##
        if ("weibull" %in% fused_distributions) {

              text <-  gsub(pattern = "(?<![A-Za-z0-9_.])weibull_lccdf(?=\\s*\\()",
                            replacement = "nico_weibull_lccdf",
                            x = text,
                            perl = TRUE)

        }
        ## (a model's own function log_sum_exp_2d keeps its name in its definition and declarations, i.e. after
        ##  its return type vector: only its calls become BayesMVP's nico_log_sum_exp_2d)
        ##
        ## ---- Check the condition ---------------------------------------------------------------------------------------------------
        ##
        if ("log_sum_exp_2d" %in% fused_distributions) {

              text <-  gsub(pattern = "(?<![A-Za-z0-9_.])(vector\\s+)nico_log_sum_exp_2d(?=\\s*\\()",
                            replacement = "\\1log_sum_exp_2d",
                            x = text,
                            perl = TRUE)

        }
        ## (the model's own functions offered: their calls become BayesMVP's function, e.g.
        ##  inv_Phi_approx_from_prob -> nico_inv_Phi_approx; their definitions, after a return type, keep their
        ##  names)
        ##
        ## ---- Each item -------------------------------------------------------------------------------------------------------------
        ##
        for (i in seq_len(nrow(model_functions_offered))) {

              text <-  gsub(pattern = paste0("(?<![A-Za-z0-9_.])", model_functions_offered$model_function[i],
                                             "(?=\\s*\\()"),
                            replacement = model_functions_offered$BayesMVP_function[i],
                            x = text,
                            perl = TRUE)
              text <-  gsub(pattern = paste0("(?<![A-Za-z0-9_.])((vector|row_vector|matrix|real|int|void)\\s+)",
                                             model_functions_offered$BayesMVP_function[i], "(?=\\s*\\()"),
                            replacement = paste0("\\1", model_functions_offered$model_function[i]),
                            x = text,
                            perl = TRUE)

        }
        ##
        ## ---- (3) stanc decides: a call it rejects gets Stan's own function back, one call at a time: -------------------------------
        ##
        candidate_file <-  tempfile(pattern = "AVX_version_candidate_", fileext = ".stan")
        stanc_output_file <-  tempfile(pattern = "AVX_version_candidate_", fileext = ".hpp")
        stanc_accepted_the_AVX_version <-  FALSE
        # for (attempt in seq_len(500)) {
        ## (at most every call of a BayesMVP function is given back: one attempt more than there are calls)
        number_of_calls_of_BayesMVP_functions <-  length(gregexpr(
              pattern = "(?<![A-Za-z0-9_.])nico_[A-Za-z0-9_]+(?=\\s*\\()", text = text, perl = TRUE)[[1]])
        ##
        ## ---- Each item -------------------------------------------------------------------------------------------------------------
        ##
        for (attempt in seq_len(number_of_calls_of_BayesMVP_functions + 1)) {

              code <-  fn_assemble(code = text, used = c(elementwise_functions, fused_distributions,
                                                         model_functions_offered$model_function))
              writeLines(text = R_fn_Stan_parallel_call_names(code = fn_restore(code)), con = candidate_file)
              stanc_output <-  suppressWarnings(system2( command = stanc_file_path,
                                                         args    = c("--allow-undefined",
                                                                     stanc_include_paths_arguments,
                                                                     paste0("--o=", shQuote(stanc_output_file)),
                                                                     shQuote(candidate_file)),
                                                         stdout  = TRUE,
                                                         stderr  = TRUE))
              ##
              ## ---- Check the condition ---------------------------------------------------------------------------------------------
              ##
              if (is.null(attr(stanc_output, "status")) || attr(stanc_output, "status") == 0) {

                    stanc_accepted_the_AVX_version <-  TRUE
                    break

              }
              stanc_message <-  paste(stanc_output, collapse = " ")
              position <-  regmatches(x = stanc_message,
                                      m = regexec(pattern = "line ([0-9]+), column ([0-9]+)",
                                                  text = stanc_message))[[1]]
              ##
              ## ---- Check the condition ---------------------------------------------------------------------------------------------
              ##
              if (length(position) < 3 || !grepl(pattern = "nico_[A-Za-z0-9_]+", x = stanc_message)) {

                    stop(paste0("stanc rejected the AVX version of ", source_Stan_model_file_name,
                                " for another reason than a call of a BayesMVP function:\n",
                                paste(stanc_output, collapse = "\n")))

              }
              ## (the line of the rejected call in the rewritten text: the assembled code has the declarations
              ##  above it, so the line is found again by its content)
              code_lines <-  strsplit(x = fn_restore(code), split = "\n")[[1]]
              rejected_line <-  code_lines[as.numeric(position[2])]
              rejected_column <-  as.numeric(position[3])
              text_lines <-  strsplit(x = text, split = "\n")[[1]]
              matching_lines <-  which(vapply(X = text_lines,
                                              FUN = function(line) identical(fn_restore(line), rejected_line),
                                              FUN.VALUE = logical(1)))
              ##
              ## ---- Check the condition ---------------------------------------------------------------------------------------------
              ##
              if (length(matching_lines) == 0) {

                    stop(paste0("The line of ", source_Stan_model_file_name, " which stanc rejected could not ",
                                "be found in the AVX version (e.g. a line with the start of a comment over ",
                                "several lines): ", rejected_line))

              }

              # line_index <-  matching_lines[1]
              ## (several lines with the same content, e.g. the vector and real versions of a function: the line
              ##  at stanc's line, i.e. the same one, counted in order, of the identical lines of the candidate,
              ##  which has the declarations above them)
              candidate_lines_with_the_same_content <-  which(code_lines == rejected_line)
              order_of_rejected_line <-  match(as.numeric(position[2]), candidate_lines_with_the_same_content)
              if (length(candidate_lines_with_the_same_content) != length(matching_lines) ||
                  ##
                  ## ---- Check the condition -----------------------------------------------------------------------------------------
                  ##
                  is.na(order_of_rejected_line)) {

                    stop(paste0("The line of ", source_Stan_model_file_name, " which stanc rejected could not ",
                                "be told apart from other lines with the same content: ", rejected_line))

              }
              line_index <-  matching_lines[order_of_rejected_line]
              ## (the rejected call: the call of a BayesMVP function in the rejected line which starts nearest to
              ##  the column given by stanc; the name in stanc's message is not used, since the message also shows
              ##  the lines around the rejected one, which may call other BayesMVP functions; a ~ statement names
              ##  the distribution without its _lpdf / _lpmf suffix, as it is written in the line)
              calls <-  gregexpr(pattern = "nico_[A-Za-z0-9_]+(?=\\s*\\()", text = text_lines[line_index],
                                 perl = TRUE)[[1]]
              ##
              ## ---- Check the condition ---------------------------------------------------------------------------------------------
              ##
              if (calls[1] == -1) {

                    stop(paste0("The line of ", source_Stan_model_file_name, " which stanc rejected has no ",
                                "call of a BayesMVP function: ", rejected_line, "\n",
                                paste(stanc_output, collapse = "\n")))

              }
              closest_call <-  calls[which.min(abs(calls - (rejected_column + 1)))]
              text_before_call <-  substr(x = text_lines[line_index], start = 1, stop = closest_call - 1)
              text_from_call_without_prefix <-  substr(x = text_lines[line_index],
                                                     start = closest_call + nchar("nico_"),
                                                     stop = nchar(text_lines[line_index]))
              ## (a BayesMVP function which replaces a model's own function gives the model's function back, e.g.
              ##  nico_inv_Phi_approx -> inv_Phi_approx_from_prob)
              text_from_call <-  substr(x = text_lines[line_index], start = closest_call,
                                        stop = nchar(text_lines[line_index]))
              name_of_closest_call <-  regmatches(x = text_from_call,
                                                  m = regexpr(pattern = "^nico_[A-Za-z0-9_]+",
                                                              text = text_from_call))
              ##
              ## ---- Check the condition ---------------------------------------------------------------------------------------------
              ##
              if (name_of_closest_call %in% model_functions_offered$BayesMVP_function) {

                    model_function <-  model_functions_offered$model_function[
                                             match(name_of_closest_call,
                                                   model_functions_offered$BayesMVP_function)]
                    text_from_call_without_prefix <-  paste0(model_function,
                                                           substr(x = text_from_call,
                                                                  start = nchar(name_of_closest_call) + 1,
                                                                  stop = nchar(text_from_call)))

              }
              text_lines[line_index] <-  paste0(text_before_call, text_from_call_without_prefix)
              text <-  paste(text_lines, collapse = "\n")

        }
        ##
        ## ---- Check the condition ---------------------------------------------------------------------------------------------------
        ##
        if (!stanc_accepted_the_AVX_version) {

              # stop(paste0("stanc did not accept the AVX version of ", source_Stan_model_file_name,
              #             " after 500 calls of BayesMVP functions were given back to Stan."))
              stop(paste0("stanc did not accept the AVX version of ", source_Stan_model_file_name,
                          " after every call of a BayesMVP function was given back to Stan."))

        }
        ##
        ## ---- the calls left on Stan math and their argument types: -----------------------------------------------------------------
        ##
        used <-  fn_used(code = text)
        NicoSIMD_function_declarations <-  c(unlist(fused_distribution_declarations, use.names = FALSE),
                                             model_functions_offered$BayesMVP_declaration)
        NicoSIMD_function_names_from_declarations <-  regmatches(
              x = NicoSIMD_function_declarations,
              m = regexpr(pattern = "(?<![A-Za-z0-9_.])nico_[A-Za-z0-9_]+(?=\\s*\\()",
                          text = NicoSIMD_function_declarations, perl = TRUE))
        NicoSIMD_function_names_from_declarations <-  NicoSIMD_function_names_from_declarations[
              nzchar(NicoSIMD_function_names_from_declarations)]
        NicoSIMD_function_names <-  unique(c(paste0("nico_", elementwise_functions),
                                             NicoSIMD_function_names_from_declarations))
        NicoSIMD_function_names <-  unique(c(NicoSIMD_function_names,
                                             sub(pattern = "_lpdf$", replacement = "_lupdf",
                                                 x = NicoSIMD_function_names),
                                             sub(pattern = "_lpmf$", replacement = "_lupmf",
                                                 x = NicoSIMD_function_names)))
        NicoSIMD_tilde_distribution_names <-  sub(
              pattern = "_(lpdf|lupdf|lpmf|lupmf)$",
              replacement = "",
              x = NicoSIMD_function_names_from_declarations[
                    grepl(pattern = "_(lpdf|lupdf|lpmf|lupmf)$",
                          x = NicoSIMD_function_names_from_declarations)])
        function_calls_left <-  fn_calls_of_functions(
              code                        = text,
              names_of_functions          = names(math_functions_whose_calls_left_on_Stan_math_are_reported),
              with_distribution_suffixes  = FALSE)
        function_calls_left$kind <-  rep("function", nrow(function_calls_left))
        distribution_calls_left <-  fn_calls_of_functions(
              code                        = text,
              names_of_functions          = names(distributions_whose_calls_left_on_Stan_math_are_reported),
              with_distribution_suffixes  = TRUE)
        ## (a distribution's name without a suffix is a call of the distribution only in a ~ statement)
        distribution_calls_left <-  distribution_calls_left[(distribution_calls_left$suffix != "") |
                                                            distribution_calls_left$in_tilde_statement, ,
                                                            drop = FALSE]
        distribution_calls_left$kind <-  rep("distribution", nrow(distribution_calls_left))
        distribution_names_from_suffix_calls <-  regmatches(
              x = text,
              m = gregexpr(pattern = paste0("(?<![A-Za-z0-9_.])[A-Za-z_][A-Za-z0-9_]*",
                                            "(_lpdf|_lupdf|_lpmf|_lupmf|_lcdf|_lccdf)(?=\\s*\\()"),
                           text = text, perl = TRUE))[[1]]
        distribution_names_from_suffix_calls <-  unique(sub(
              pattern = "_(lpdf|lupdf|lpmf|lupmf|lcdf|lccdf)$",
              replacement = "",
              x = distribution_names_from_suffix_calls))
        text_without_markers <-  gsub(pattern = "\001[0-9]+\001", replacement = " ", x = text)
        distribution_calls_in_tilde_statements <-  regmatches(
              x = text_without_markers,
              m = gregexpr(pattern = "~\\s*[A-Za-z_][A-Za-z0-9_]*\\s*\\(",
                           text = text_without_markers, perl = TRUE))[[1]]
        distribution_names_in_tilde_statements <-  sub(
              pattern = "\\s*\\($",
              replacement = "",
              x = sub(pattern = "^~\\s*", replacement = "", x = distribution_calls_in_tilde_statements))
        unlisted_distribution_names <-  setdiff(
              unique(c(distribution_names_from_suffix_calls, distribution_names_in_tilde_statements)),
              names(distributions_whose_calls_left_on_Stan_math_are_reported))
        user_distribution_function_names <-  character(0)
        functions_code <-  ""
        functions_block <-  regexpr(pattern = "(^|\\n)\\s*functions\\s*\\{", text = text, perl = TRUE)
        ##
        ## ---- Check the condition ---------------------------------------------------------------------------------------------------
        ##
        if (functions_block != -1) {

              opening_brace <-  functions_block + attr(functions_block, "match.length") - 1
              closing_brace <-  fn_position_of_matching_closing_bracket(
                    code = text, position_of_opening_bracket = opening_brace, opening = "{", closing = "}")
              functions_code <-  substr(text, opening_brace + 1, closing_brace - 1)
              user_function_names <-  c(fn_names_of_stan_function_definitions(code = functions_code),
                                        fn_names_of_stan_function_declarations(code = functions_code))
              # distribution_definitions <-  gregexpr(
              #       pattern = paste0("(?<![A-Za-z0-9_.])real\\s+([A-Za-z_][A-Za-z0-9_]*",
              #                        "_(?:lpdf|lupdf|lpmf|lupmf|lcdf|lccdf))\\s*\\([^{}]*\\)\\s*\\{"),
              ## (no ; in the arguments: a declaration before a definition is not taken into its match)
              distribution_definitions <-  gregexpr(
                    pattern = paste0("(?<![A-Za-z0-9_.])real\\s+([A-Za-z_][A-Za-z0-9_]*",
                                     "_(?:lpdf|lupdf|lpmf|lupmf|lcdf|lccdf))\\s*\\([^{};]*\\)\\s*\\{"),
                    text = functions_code, perl = TRUE)[[1]]
              ##
              ## ---- Check the condition ---------------------------------------------------------------------------------------------
              ##
              if (distribution_definitions[1] != -1) {

                    definition_starts <-  attr(distribution_definitions, "capture.start")[, 1]
                    definition_lengths <-  attr(distribution_definitions, "capture.length")[, 1]
                    user_distribution_function_names <-  substring(
                          functions_code, definition_starts, definition_starts + definition_lengths - 1)

              }

        ##
        ## ---- The alternative -------------------------------------------------------------------------------------------------------
        ##
        } else {

              user_function_names <-  character(0)

        }
        unlisted_distribution_calls_left <-  data.frame(
              position = numeric(0),
              name = character(0),
              suffix = character(0),
              in_tilde_statement = logical(0),
              stringsAsFactors = FALSE)
        ##
        ## ---- Check the condition ---------------------------------------------------------------------------------------------------
        ##
        if (length(unlisted_distribution_names) > 0) {

              unlisted_distribution_calls_left <-  fn_calls_of_functions(
                    code                        = text,
                    names_of_functions          = unlisted_distribution_names,
                    with_distribution_suffixes  = TRUE)
              unlisted_distribution_calls_left <-  unlisted_distribution_calls_left[
                    (unlisted_distribution_calls_left$suffix != "") |
                    unlisted_distribution_calls_left$in_tilde_statement, , drop = FALSE]
              # custom_distribution_call <-  paste0(unlisted_distribution_calls_left$name,
              #                                      unlisted_distribution_calls_left$suffix) %in%
              #                               user_distribution_function_names
              ## (a _lupdf / _lupmf call of a user function defined as _lpdf / _lpmf is that user function)
              custom_distribution_call <-  paste0(unlisted_distribution_calls_left$name,
                                                   sub(pattern = "^_lup(df|mf)$", replacement = "_lp\\1",
                                                       x = unlisted_distribution_calls_left$suffix)) %in%
                                            user_distribution_function_names
              custom_distribution_call[unlisted_distribution_calls_left$in_tilde_statement] <-
                    paste0(unlisted_distribution_calls_left$name[
                                 unlisted_distribution_calls_left$in_tilde_statement], "_lpdf") %in%
                              user_distribution_function_names |
                          paste0(unlisted_distribution_calls_left$name[
                                 unlisted_distribution_calls_left$in_tilde_statement], "_lpmf") %in%
                              user_distribution_function_names
              custom_distribution_call <-  custom_distribution_call &
                    !startsWith(paste0(unlisted_distribution_calls_left$name,
                                       unlisted_distribution_calls_left$suffix), "nico_")
              unlisted_distribution_calls_left <-  unlisted_distribution_calls_left[
                    !custom_distribution_call, , drop = FALSE]
              registered_nico_distribution_call <-  paste0(unlisted_distribution_calls_left$name,
                                                           unlisted_distribution_calls_left$suffix) %in%
                                                    NicoSIMD_function_names
              registered_nico_distribution_call[unlisted_distribution_calls_left$in_tilde_statement] <-
                    unlisted_distribution_calls_left$name[
                          unlisted_distribution_calls_left$in_tilde_statement] %in%
                    NicoSIMD_tilde_distribution_names
              unlisted_distribution_calls_left <-  unlisted_distribution_calls_left[
                    !registered_nico_distribution_call, , drop = FALSE]
              unlisted_distribution_calls_left$kind <-  rep("distribution",
                                                             nrow(unlisted_distribution_calls_left))
              unlisted_distribution_calls_left$line_in_Stan_model <-  fn_lines_in_Stan_model_of_positions(
                    code = text, positions = unlisted_distribution_calls_left$position)
              unlisted_distribution_calls_left$argument_types <-  rep(
                    "not probed", nrow(unlisted_distribution_calls_left))
              unlisted_distribution_calls_left$in_definition_not_called <-  rep(
                    "", nrow(unlisted_distribution_calls_left))

        }
        calls_left <-  rbind(function_calls_left, distribution_calls_left)
        calls_left <-  calls_left[order(calls_left$position), , drop = FALSE]
        calls_left$line_in_Stan_model <-  fn_lines_in_Stan_model_of_positions(code = text,
                                                                              positions = calls_left$position)
        ## (the probes: real and int arguments, in every combination, for each number of arguments of the calls on
        ##  reals; a function with none (e.g. softmax) gets a signature which no call has, a 4-dimensional int
        ##  array; the variate of a distribution, its first argument, is real for a density and int for a mass
        ##  function)
        ##
        ## ---- The function ----------------------------------------------------------------------------------------------------------
        ##
        fn_signatures_on_reals <-  function(n_arguments, type_of_first_argument = c("real", "int")) {

              if (length(n_arguments) == 0) return("array[,,,] int a1")
              signatures <-  c()
              ##
              ## ---- Each item -------------------------------------------------------------------------------------------------------
              ##
              for (n in n_arguments) {

                    argument_types <-  expand.grid(c(list(type_of_first_argument),
                                                     rep(list(c("real", "int")), n - 1)),
                                                   stringsAsFactors = FALSE)
                    ##
                    ## ---- Each item -------------------------------------------------------------------------------------------------
                    ##
                    for (r in seq_len(nrow(argument_types))) {

                          signatures <-  c(signatures, paste(paste0(unlist(argument_types[r, ]), " a", 1:n),
                                                             collapse = ", "))

                    }

              }
              return(signatures)

        }
        ##
        ## ---- The function ----------------------------------------------------------------------------------------------------------
        ##
        probe_declarations_of_each_call <-  lapply(X = seq_len(nrow(calls_left)), FUN = function(k) {

              name <-  calls_left$name[k]
              ##
              ## ---- Check the condition ---------------------------------------------------------------------------------------------
              ##
              if (identical(calls_left$kind[k], "function")) {

                    numbers_of_arguments_on_reals <-  math_functions_whose_calls_left_on_Stan_math_are_reported[[
                                                            name]]
                    return(paste0("    real bmvp_probe_", k, "_", name, "(",
                                  fn_signatures_on_reals(numbers_of_arguments_on_reals), ");"))

              }
              distribution <-  distributions_whose_calls_left_on_Stan_math_are_reported[[name]]
              ##
              ## ---- Check the condition ---------------------------------------------------------------------------------------------
              ##
              declaration_suffix <-  if (calls_left$suffix[k] %in% c("_lccdf", "_lcdf")) {

                    calls_left$suffix[k]

              } else distribution$declaration_suffix
              type_of_variate <-  if (identical(declaration_suffix, "_lpmf")) "int" else "real"
              return(paste0("    real bmvp_probe_", k, "_", name, declaration_suffix, "(",
                            fn_signatures_on_reals(n_arguments = distribution$n_arguments,
                                                   type_of_first_argument = type_of_variate), ");"))

        })
        ##
        ## ---- Check the condition ---------------------------------------------------------------------------------------------------
        ##
        calls_left$argument_types <-  if (nrow(calls_left) > 0) {

              fn_argument_types_of_calls_which_stanc_rejects(code = text, calls = calls_left,
                                                             probe_declarations_of_each_call =
                                                                   probe_declarations_of_each_call,
                                                             used = used)

        } else character(0)
        ##
        ## (the definitions which the AVX version does not call: those of the model's functions offered, and
        ##  log_sum_exp_2d where nico_log_sum_exp_2d is used, unless a call left on Stan math has their argument
        ##  types)
        signatures_of_definitions_not_called <-  model_functions_offered$model_signature
        ##
        ## ---- Check the condition ---------------------------------------------------------------------------------------------------
        ##
        if ("log_sum_exp_2d" %in% used) {

              signatures_of_definitions_not_called <-  c(signatures_of_definitions_not_called,
                                                         "vector log_sum_exp_2d(matrix)")

        }
        calls_left$in_definition_not_called <-  rep("", nrow(calls_left))
        ## (each definition which the AVX version does not call: from its return type to its closing brace)
        definitions_commented_out <-  data.frame(start            = numeric(0),
                                                 end              = numeric(0),
                                                 model_signature  = character(0),
                                                 stringsAsFactors = FALSE)
        ##
        ## ---- Check the condition ---------------------------------------------------------------------------------------------------
        ##
        if (nrow(unlisted_distribution_calls_left) > 0) {

              calls_left <-  rbind(calls_left, unlisted_distribution_calls_left)
              calls_left <-  calls_left[order(calls_left$position), , drop = FALSE]

        }
        all_function_calls <-  fn_calls_of_all_functions(code = text)
        ## (only zero-argument constant getters; log2(x) remains a required mathematical call)
        constant_getters <-  c("pi", "log2", "positive_infinity", "negative_infinity")
        ##
        ## ---- The function ----------------------------------------------------------------------------------------------------------
        ##
        is_constant_getter <-  vapply(seq_len(nrow(all_function_calls)), function(i) {

              if (!all_function_calls$name[i] %in% constant_getters) return(FALSE)
              call_tail <-  substring(text, all_function_calls$position[i] + nchar(all_function_calls$name[i]))
              call_tail <-  gsub("\001[0-9]+\001", "", call_tail)
              return(grepl("^[[:space:]]*\\([[:space:]]*\\)", call_tail, perl = TRUE))

        }, logical(1))
        all_function_calls <-  all_function_calls[!is_constant_getter, , drop = FALSE]
        ## (only declared or generated NicoSIMD names are excluded from this scan)
        all_function_calls <-  all_function_calls[!all_function_calls$name %in% c(
              names(math_functions_whose_calls_left_on_Stan_math_are_reported),
              elementwise_functions,
              model_functions_offered$model_function,
              structural_functions_that_may_remain_on_Stan_math,
              stan_functions_dispatching_user_functions,
              user_function_names,
              "if", "for", "while", "return", "print", "reject") &
              ## (a random generator left on Stan math is reported: it is not a distribution's density)
              !grepl(pattern = "_(lpdf|lupdf|lpmf|lupmf|lcdf|lccdf)$",
                     x = all_function_calls$name) &
              !all_function_calls$in_tilde_statement &
              !all_function_calls$name %in% NicoSIMD_function_names, , drop = FALSE]
        all_function_calls$kind <-  rep("unclassified", nrow(all_function_calls))
        ## (uniform_rng and bernoulli_rng: listed basic functions which stay on Stan math)
        all_function_calls$kind[all_function_calls$name %in% math_functions_which_may_remain_on_Stan_math] <-
              "function"
        all_function_calls$line_in_Stan_model <-  fn_lines_in_Stan_model_of_positions(
              code = text, positions = all_function_calls$position)
        all_function_calls$argument_types <-  rep("not classified", nrow(all_function_calls))
        all_function_calls$in_definition_not_called <-  rep("", nrow(all_function_calls))
        ##
        ## ---- Check the condition ---------------------------------------------------------------------------------------------------
        ##
        if (nrow(all_function_calls) > 0) {

              calls_left <-  rbind(calls_left, all_function_calls)
              calls_left <-  calls_left[order(calls_left$position), , drop = FALSE]

        }
        ##
        ## ---- Each item -------------------------------------------------------------------------------------------------------------
        ##
        for (model_signature in signatures_of_definitions_not_called) {

              definition <-  regexpr(pattern = fn_pattern_of_definition(model_signature), text = text,
                                     perl = TRUE)
              if (definition == -1) next
              function_name <-  sub(pattern = "^[A-Za-z_]+\\s+([A-Za-z0-9_]+)\\(.*$", replacement = "\\1",
                                    x = model_signature)
              argument_types_of_signature <-  sub(pattern = "^.*\\((.*)\\)$", replacement = "\\1",
                                                  x = model_signature)
              argument_types_of_calls_left <-  gsub(pattern = "data ", replacement = "",
                                                    x = calls_left$argument_types)
              calls_of_this_definition_left <-  (calls_left$name == function_name) &
                                                (argument_types_of_calls_left == argument_types_of_signature)
              if (any(calls_of_this_definition_left)) next
              position_of_opening_brace <-  definition + attr(definition, "match.length") - 1
              position_of_closing_brace <-  fn_position_of_matching_closing_bracket(
                    code = text, position_of_opening_bracket = position_of_opening_brace, opening = "{",
                    closing = "}")
              call_is_in_definition <-  (calls_left$position > position_of_opening_brace) &
                                        (calls_left$position < position_of_closing_brace)
              calls_left$in_definition_not_called[call_is_in_definition] <-  model_signature
              definitions_commented_out <-  rbind(definitions_commented_out,
                                                  data.frame(start            = as.numeric(definition),
                                                             end              = position_of_closing_brace,
                                                             model_signature  = model_signature,
                                                             stringsAsFactors = FALSE))

        }
        ## (calls without an offered NicoSIMD replacement are reported; unsupported math calls stop writing)
        distributions_offered <-  setdiff(fused_distributions, fused_functions_written_only_by_fused_writers)
        ##
        ## ---- Check the condition ---------------------------------------------------------------------------------------------------
        ##
        if (("normal" %in% fused_distributions) && (length(std_normal_as_fused_normal_declarations) > 0)) {

              distributions_offered <-  c(distributions_offered, "std_normal")

        }
        NicoSIMD_has_a_version_of_the_call <-  (calls_left$name %in% c(elementwise_functions,
                                                                        model_functions_offered$model_function,
                                                                        fused_distributions) &
                                                 calls_left$suffix == "") |
                                                (calls_left$name %in% distributions_offered &
                                                 calls_left$suffix %in% c("_lpdf", "_lupdf", "_lpmf", "_lupmf")) |
                                                (calls_left$name == "weibull" &
                                                 "weibull" %in% fused_distributions &
                                                 calls_left$suffix == "_lccdf")
        ## (a distribution on the list above stays on Stan math, which computes it in full)
        calls_left$requires_NicoSIMD <-  (calls_left$kind == "unclassified") |
                                         (calls_left$kind == "distribution" &
                                          !(calls_left$name %in% math_functions_which_may_remain_on_Stan_math)) |
                                         (calls_left$name %in% c(elementwise_functions,
                                                                 model_functions_offered$model_function,
                                                                 fused_distributions) &
                                          !(calls_left$name %in% cheap_functions_never_renamed_from_calls)) |
                                         (calls_left$name %in% names(
                                               math_functions_whose_calls_left_on_Stan_math_are_reported) &
                                          !(calls_left$name %in% math_functions_which_may_remain_on_Stan_math))
        # calls_left$category <-  ifelse(calls_left$argument_types == "",
        #                                "real or int: stays on Stan math",
        #                                ifelse(calls_left$in_definition_not_called != "",
        #                                       paste0("in the definition of ",
        #                                              calls_left$in_definition_not_called,
        #                                              ", which the AVX version does not call"),
        #                                       "NOT SWAPPED: left on Stan math"))
        calls_left$category <-  ifelse(calls_left$in_definition_not_called != "",
                                       paste0("in the definition of ", calls_left$in_definition_not_called,
                                              ", which the AVX version does not call"),
                                ifelse(calls_left$kind == "unclassified",
                                       "UNCLASSIFIED: no structural or NicoSIMD route",
                                ifelse(!calls_left$requires_NicoSIMD,
                                       "verified basic function: stays on Stan math",
                                ifelse(NicoSIMD_has_a_version_of_the_call,
                                       "UNSUPPORTED: NicoSIMD call remains on Stan math",
                                       "UNSUPPORTED: no NicoSIMD function for this math call"))))
        calls_left_requiring_NicoSIMD <-  calls_left[calls_left$requires_NicoSIMD &
                                                      calls_left$in_definition_not_called == "", , drop = FALSE]
        ##
        ## (the report: NicoSIMD calls and every remaining Stan math call)
        ##
        NicoSIMD_calls <-  regmatches(x = text,
                                      m = gregexpr(pattern = "(?<![A-Za-z0-9_.])nico_[A-Za-z0-9_]+(?=\\s*\\()",
                                                   text = text, perl = TRUE))[[1]]
        NicoSIMD_calls_counted <-  table(NicoSIMD_calls)
        calls_left_as_text <-  paste0(formatC(calls_left$line_in_Stan_model, width = 6), "  ",
                                      formatC(paste0(calls_left$name, calls_left$suffix,
                                                     ifelse(calls_left$in_tilde_statement, " (~)", "")),
                                              width = -34),
                                      formatC(ifelse(calls_left$argument_types == "", "real / int",
                                                     calls_left$argument_types),
                                              width = -28),
                                      calls_left$category)
        report_lines <-  c(paste0("Calls left on Stan math in the AVX version of ", source_Stan_model_file_name,
                                  ": ", basename(AVX_Stan_model_file_path)),
                           "",
                           paste0("Calls of NicoSIMD functions: ", length(NicoSIMD_calls), " (",
                                  paste(paste0(names(NicoSIMD_calls_counted), " ",
                                               as.numeric(NicoSIMD_calls_counted)),
                                        collapse = ", "),
                                  ")"),
                           paste0("Calls left on Stan math with scalar arguments: ",
                                  sum(calls_left$argument_types == "")),
                           paste0("Calls left on Stan math in definitions which the AVX version does not call: ",
                                  sum(calls_left$in_definition_not_called != "")),
                           paste0("Calls left on Stan math requiring NicoSIMD or classification: ",
                                  nrow(calls_left_requiring_NicoSIMD)),
                           paste0("Calls left on Stan math allowed to stay: ",
                                  sum(!calls_left$requires_NicoSIMD &
                                      calls_left$in_definition_not_called == "")),
                           "",
                           paste0(formatC("line", width = 6), "  ", formatC("function", width = -34),
                                  formatC("argument types", width = -28), "category"),
                           calls_left_as_text)
        ##
        ## ---- Check the condition ---------------------------------------------------------------------------------------------------
        ##
        if (!is.null(report_file_path_of_calls_left_on_Stan_math)) {

              writeLines(text = report_lines, con = report_file_path_of_calls_left_on_Stan_math)

        }
        message(colourise(paste0(source_Stan_model_file_name, ": ", length(NicoSIMD_calls),
                                 " calls of NicoSIMD functions; ",
                                 "required or unclassified calls left on Stan math: ",
                                 nrow(calls_left_requiring_NicoSIMD),
                                 ##
                                 ## ---- Check the condition --------------------------------------------------------------------------
                                 ##
                                 if (!is.null(report_file_path_of_calls_left_on_Stan_math)) {

                                       paste0(" (", report_file_path_of_calls_left_on_Stan_math, ")")

                                 }),
                          "cyan"))
        ##
        ## ---- Check the condition ---------------------------------------------------------------------------------------------------
        ##
        if (nrow(calls_left_requiring_NicoSIMD) > 0) {

              stop(paste0("The AVX version of ", source_Stan_model_file_name, " would leave ",
                          nrow(calls_left_requiring_NicoSIMD),
                          " required or unclassified calls on Stan math, so it is not written:\n",
                          paste(paste0("  line ", calls_left_requiring_NicoSIMD$line_in_Stan_model, ": ",
                                       calls_left_requiring_NicoSIMD$name,
                                       calls_left_requiring_NicoSIMD$suffix, "(",
                                       calls_left_requiring_NicoSIMD$argument_types, ")"),
                                collapse = "\n")))

        }
        ##
        ## ---- the definitions which the AVX version does not call are commented out, so that no Stan math call is left in them ------
        ##
        definitions_commented_out <-  definitions_commented_out[!duplicated(definitions_commented_out$start), ,
                                                                drop = FALSE]
        ##
        ## ---- Each item -------------------------------------------------------------------------------------------------------------
        ##
        for (k in rev(order(definitions_commented_out$start))) {

              ## (from the definition's return type to its closing brace; the last
              ##  definition first, so that the positions of the others stay the same)
              definition_start <-  definitions_commented_out$start[k]
              definition_lines <-  strsplit(x = fn_restore(substr(text,
                                                                  definition_start,
                                                                  definitions_commented_out$end[k])),
                                            split = "\n")[[1]]
              protected <-  c(protected,
                              paste(c(paste0("// (commented out: NicoSIMD replaces every call of ",
                                             definitions_commented_out$model_signature[k], ")"),
                                      paste0("//", definition_lines),
                                      ""),
                                    collapse = "\n"))
              text <-  paste0(substr(text, 1, definition_start - 1), "\001", length(protected), "\001",
                              substr(text, definitions_commented_out$end[k] + 1, nchar(text)))

        }
        ##
        ## ---- Check the condition ---------------------------------------------------------------------------------------------------
        ##
        if (nrow(definitions_commented_out) > 0) {

              stanc_output <-  fn_stanc_output(code = fn_assemble(code = text, used = fn_used(code = text)))
              ##
              ## ---- Check the condition ---------------------------------------------------------------------------------------------
              ##
              if (!(is.null(attr(stanc_output, "status")) || attr(stanc_output, "status") == 0)) {

                    stop(paste0("stanc rejected the AVX version of ", source_Stan_model_file_name,
                                " once the definitions it does not call were commented out:\n",
                                paste(stanc_output, collapse = "\n")))

              }

        }
        ##
        ## ---- the AVX version: its first line, the BayesMVP functions used, and the model with only their ---------------------------
        ##
        ##      declarations:
        ##
        used <-  fn_used(code = text)
        code <-  fn_restore(fn_assemble(code = text, used = used))
        BayesMVP_functions_used_as_text <-  if (length(used) > 0) paste(used, collapse = ", ") else "none"
        header_lines <-  c(first_line_of_AVX_Stan_model,
                           paste0("// NicoSIMD functions used: ", BayesMVP_functions_used_as_text))
        temporary_file <-  tempfile(pattern = "AVX_version_being_written_",
                                    tmpdir = dirname(AVX_Stan_model_file_path),
                                    fileext = ".stan.tmp")
        writeLines(text = c(header_lines, code), con = temporary_file)
        ##
        ## ---- Check the condition ---------------------------------------------------------------------------------------------------
        ##
        if (!file.rename(from = temporary_file, to = AVX_Stan_model_file_path)) {

              stop(paste0("Could not write the AVX version of the Stan model: ", AVX_Stan_model_file_path))

        }
        ##
        return(used)

}




##
## ---- local headers used by a compiled model: ---------------------------------------------------------------------------------------
##
fn_local_header_dependency_records <-  function(header_paths,
                                                ##
                                                ## ---- The function ------------------------------------------------------------------
                                                ##
                                                NicoSIMD_headers_folder) {

        NicoSIMD_headers_folder <-  normalizePath(path = NicoSIMD_headers_folder, mustWork = TRUE)
        NicoSIMD_include_root <-  dirname(NicoSIMD_headers_folder)
        include_roots <-  unique(c(NicoSIMD_headers_folder, NicoSIMD_include_root))
        pending_paths <-  unique(normalizePath(path = header_paths, mustWork = TRUE))
        dependency_paths <-  character(0)
        ##
        ## ---- Repeat the search -----------------------------------------------------------------------------------------------------
        ##
        while (length(pending_paths) > 0) {

              current_path <-  pending_paths[1]
              pending_paths <-  pending_paths[-1]
              if (current_path %in% dependency_paths) next
              dependency_paths <-  c(dependency_paths, current_path)
              header_lines <-  readLines(con = current_path, warn = FALSE)
              inside_block_comment <-  FALSE
              ##
              ## ---- Each item -------------------------------------------------------------------------------------------------------
              ##
              for (line_index in seq_along(header_lines)) {

                    line <-  header_lines[line_index]
                    ##
                    ## ---- Check the condition ---------------------------------------------------------------------------------------
                    ##
                    if (inside_block_comment) {

                          block_end <-  regexpr(pattern = "\\*/", text = line, perl = TRUE)
                          if (block_end == -1) next
                          line <-  substring(line, block_end + 2)
                          inside_block_comment <-  FALSE

                    }
                    block_start <-  regexpr(pattern = "/\\*", text = line, perl = TRUE)
                    ##
                    ## ---- Repeat the search -----------------------------------------------------------------------------------------
                    ##
                    while (block_start != -1) {

                          block_end <-  regexpr(pattern = "\\*/", text = substring(line, block_start + 2),
                                                perl = TRUE)
                          ##
                          ## ---- Check the condition ---------------------------------------------------------------------------------
                          ##
                          if (block_end == -1) {

                                line <-  substring(line, 1, block_start - 1)
                                inside_block_comment <-  TRUE
                                break

                          }
                          line <-  paste0(substring(line, 1, block_start - 1), " ",
                                          substring(line, block_start + block_end + 3))
                          block_start <-  regexpr(pattern = "/\\*", text = line, perl = TRUE)

                    }
                    if (!grepl(pattern = "^\\s*#\\s*include\\b", x = line, perl = TRUE)) next
                    include_match <-  regmatches(
                          x = line,
                          m = regexec(pattern = '^\\s*#\\s*include\\s*(?:<([^>]+)>|"([^"]+)")',
                                      text = line, perl = TRUE))[[1]]
                    ##
                    ## ---- Check the condition ---------------------------------------------------------------------------------------
                    ##
                    if (length(include_match) < 3) {

                          stop(paste0("Could not read a local include in ", current_path, " at line ",
                                      line_index, "."))

                    }
                    include_is_quoted <-  nzchar(include_match[3])
                    include_name <-  if (include_is_quoted) include_match[3] else include_match[2]
                    ##
                    ## ---- Check the condition ---------------------------------------------------------------------------------------
                    ##
                    if (grepl(pattern = "\\\\[[:space:]]*$", x = line, perl = TRUE)) {

                          stop(paste0("A continued include cannot be resolved in ", current_path, " at line ",
                                      line_index, "."))

                    }
                    ##
                    ## ---- Check the condition ---------------------------------------------------------------------------------------
                    ##
                    candidate_paths <-  if (startsWith(include_name, "/")) {

                          include_name

                    ##
                    ## ---- The alternative -------------------------------------------------------------------------------------------
                    ##
                    } else if (include_is_quoted) {

                          c(file.path(dirname(current_path), include_name),
                            file.path(include_roots, include_name))

                    ##
                    ## ---- The alternative -------------------------------------------------------------------------------------------
                    ##
                    } else {

                          file.path(include_roots, include_name)

                    }
                    existing_paths <-  candidate_paths[file.exists(candidate_paths)]
                    ##
                    ## ---- Check the condition ---------------------------------------------------------------------------------------
                    ##
                    if (length(existing_paths) > 0) {

                          pending_paths <-  unique(c(pending_paths,
                                                     normalizePath(path = existing_paths[1], mustWork = TRUE)))

                    ##
                    ## ---- The alternative -------------------------------------------------------------------------------------------
                    ##
                    } else if (include_is_quoted || startsWith(include_name, "NicoSIMD/")) {

                          stop(paste0("Could not resolve local include ", include_name, " from ", current_path,
                                      " at line ", line_index, "."))

                    }

              }

        }
        dependency_info <-  file.info(dependency_paths)
        dependency_md5 <-  unname(tools::md5sum(files = dependency_paths))
        ##
        ## ---- Check the condition ---------------------------------------------------------------------------------------------------
        ##
        if (anyNA(dependency_info$mtime) || anyNA(dependency_md5)) {

              stop("A local header dependency could not be read for the compiled model.")

        }
        dependency_mtime <-  format(as.numeric(dependency_info$mtime), scientific = FALSE, trim = TRUE,
                                    digits = 17)
        return(paste0("// ", dependency_paths, " | ", dependency_mtime, " | ", dependency_md5))

}



##
## ---- the AVX version of a Stan model, its user header and its make arguments -------------------------------------------------------
##
#' Write the AVX version of a Stan model and its user header, and return them with the make arguments
#'
#' Accepts the established backend labels. Every AVX path uses NicoSIMD's native nico_ names.
#' The generated model is reused only when its source, declarations and writer implementation match.
#' Changed local header dependencies stop before existing compiled files are reused.
#' @param Stan_model_math_functions One of the existing NicoSIMD_ or BayesMVP_ AVX2 / AVX512 labels.
#' @param Stan_model_file_path The Stan model.
#' @param Stan_cpp_user_header The user's own C++ header (NULL: none); included by the AVX version's user header.
#' @param stanc_args The model's stanc arguments (only their --include-paths arguments are used here).
#' @param make_args The model's make arguments (a USER_HEADER is included by the AVX version's user header; the
#'   instruction-set make arguments CXXFLAGS_OPTIM, CPPFLAGS_OPTIM and PRECOMPILED_HEADERS must not be given).
#' @param BayesMVP_AVX_headers_folder The legacy header-folder argument; when used, it points to NicoSIMD headers.
#' @param NicoSIMD_headers_folder NicoSIMD's installed include folder; NULL (default) uses its installed path.
#' @return A list: Stan_model_math_functions (as given), Stan_model_math_functions_used ("NicoSIMD_AVX2",
#'   "NicoSIMD_AVX512", "BayesMVP_AVX2" or "BayesMVP_AVX512"), package_of_the_SIMD_functions ("NicoSIMD"),
#'   source_Stan_model_file_path, Stan_model_file_path, Stan_cpp_user_header, make_args,
#'   headers_folder_of_the_SIMD_functions, BayesMVP_AVX_headers_folder, BayesMVP_functions_used and
#'   SIMD_lanes_expected (8 for AVX-512, 4 for AVX2).
#' @keywords internal
R_fn_write_BayesMVP_AVX_version_of_Stan_model_and_user_header <-  function( Stan_model_math_functions,
                                                                           Stan_model_file_path,
                                                                           Stan_cpp_user_header = NULL,
                                                                           stanc_args = NULL,
                                                                           make_args = NULL,
                                                                           BayesMVP_AVX_headers_folder = NULL,
                                                                           NicoSIMD_headers_folder = NULL
##
## ---- The function ------------------------------------------------------------------------------------------------------------------
##
) {

        Stan_model_math_functions_with_BayesMVP_AVX <-  c("BayesMVP_AVX2",
                                                          "BayesMVP_AVX512",
                                                          "BayesMVP_AVX512_if_supported_else_AVX2")
        Stan_model_math_functions_with_NicoSIMD <-  c("NicoSIMD_AVX2",
                                                      "NicoSIMD_AVX512",
                                                      "NicoSIMD_AVX512_if_supported_else_AVX2")
        if (!(is.character(Stan_model_math_functions) && length(Stan_model_math_functions) == 1 &&
              Stan_model_math_functions %in% c("Stan_math", Stan_model_math_functions_with_NicoSIMD,
                                               ##
                                               ## ---- Check the condition ------------------------------------------------------------
                                               ##
                                               Stan_model_math_functions_with_BayesMVP_AVX))) {

              stop(paste0("Stan_model_math_functions must be one of: \"Stan_math\" (the default), \"",
                          paste(c(Stan_model_math_functions_with_NicoSIMD,
                                  Stan_model_math_functions_with_BayesMVP_AVX),
                                collapse = "\", \""), "\"."))

        }
        ##
        ## ---- Check the condition ---------------------------------------------------------------------------------------------------
        ##
        if (identical(Stan_model_math_functions, "Stan_math")) {

              stop(paste0("R_fn_write_BayesMVP_AVX_version_of_Stan_model_and_user_header: ",
                          "Stan_model_math_functions = \"Stan_math\" needs no AVX version."))

        }
        ##
        ## ---- all accepted labels use NicoSIMD's native nico_ functions: ------------------------------------------------------------
        ##
        package_of_the_SIMD_functions <-  "NicoSIMD"
        public_backend_prefix <-  sub(pattern = "_.*$", replacement = "", x = Stan_model_math_functions)
        ## ("AVX2", "AVX512" or "AVX512_if_supported_else_AVX2")
        instruction_set_asked <-  sub(pattern = "^(NicoSIMD|BayesMVP)_", replacement = "",
                                      x = Stan_model_math_functions)
        ##
        ## ---- NicoSIMD's headers: the installed NicoSIMD's, unless a folder is given (NicoSIMD is needed only -----------------------
        ##
        ##      here, i.e. only for its SIMD functions):
        ##
        ##
        ## ---- Check the condition ---------------------------------------------------------------------------------------------------
        ##
        if (is.null(NicoSIMD_headers_folder)) {

              NicoSIMD_headers_folder <-  BayesMVP_AVX_headers_folder

        }
        ##
        ## ---- Check the condition ---------------------------------------------------------------------------------------------------
        ##
        if (is.null(NicoSIMD_headers_folder)) {

              ##
              ## ---- Check the condition ---------------------------------------------------------------------------------------------
              ##
              if (!requireNamespace("NicoSIMD", quietly = TRUE)) {

                    stop(paste0("Stan_model_math_functions = \"", Stan_model_math_functions, "\" needs ",
                                "NicoSIMD's functions. Install NicoSIMD or give NicoSIMD_headers_folder."))

              }
              NicoSIMD_headers_folder <-  system.file("include", "NicoSIMD", package = "NicoSIMD")

        }
        headers_folder_of_the_SIMD_functions <-  normalizePath(path = NicoSIMD_headers_folder,
                                                               mustWork = FALSE)
        elementwise_functions_header_of_the_SIMD_functions <-  file.path(headers_folder_of_the_SIMD_functions,
                                                                         "stan_external_functions.hpp")
        scalar_functions_header_of_the_SIMD_functions <-  file.path(headers_folder_of_the_SIMD_functions,
                                                                    "stan_external_scalar_functions.hpp")
        if (!file.exists(elementwise_functions_header_of_the_SIMD_functions) ||
            !file.exists(scalar_functions_header_of_the_SIMD_functions) ||
            !file.exists(file.path(headers_folder_of_the_SIMD_functions,
                                   ##
                                   ## ---- Check the condition ------------------------------------------------------------------------
                                   ##
                                   "stan_external_functions_of_NicoSIMD.hpp"))) {

              stop(paste0("NicoSIMD's native headers are not in ", headers_folder_of_the_SIMD_functions, "."))

        }
        ##
        ## ---- the instruction set, which this CPU must support: ---------------------------------------------------------------------
        ##
        instruction_sets_supported <-  R_fn_BayesMVP_AVX_instruction_sets_supported_by_this_CPU()
        ## ("AVX2" or "AVX512")
        instruction_set_used <-  instruction_set_asked
        ##
        ## ---- The alternative -------------------------------------------------------------------------------------------------------
        ##
        if (identical(instruction_set_asked, "AVX512_if_supported_else_AVX2")) {

              instruction_set_used <-  if (instruction_sets_supported[["BayesMVP_AVX512"]]) "AVX512" else "AVX2"

        }
        Stan_model_math_functions_used <-  paste0(public_backend_prefix, "_", instruction_set_used)
        ##
        ## ---- Check the condition ---------------------------------------------------------------------------------------------------
        ##
        if (!instruction_sets_supported[[paste0("BayesMVP_", instruction_set_used)]]) {

              stop(paste0("This CPU does not support the instruction set of Stan_model_math_functions = \"",
                          Stan_model_math_functions_used, "\" (AVX2 needs the CPU flags avx2 and fma; ",
                          "AVX-512 also needs avx512f, avx512vl and avx512dq)."))

        }
        SIMD_lanes_expected <-  if (identical(instruction_set_used, "AVX512")) 8 else 4
        ##
        ## ---- the Stan model (the source of the AVX version; never an AVX version written by NicoStan itself): ----------------------
        ##
        source_Stan_model_file_path <-  normalizePath(path = Stan_model_file_path, mustWork = TRUE)
        source_Stan_model_file_name <-  basename(source_Stan_model_file_path)
        first_line_of_source_Stan_model <-  readLines(con = source_Stan_model_file_path, n = 1, warn = FALSE)
        if (length(first_line_of_source_Stan_model) == 1 &&
            (startsWith(x = first_line_of_source_Stan_model, prefix = "// BayesMVP_AVX") ||
             startsWith(x = first_line_of_source_Stan_model, prefix = "// NicoSIMD_AVX")) &&
            ##
            ## ---- Check the condition -----------------------------------------------------------------------------------------------
            ##
            grepl(pattern = "written by NicoStan", x = first_line_of_source_Stan_model, fixed = TRUE)) {

              stop(paste0(source_Stan_model_file_name, " is already an AVX version written by NicoStan; give ",
                          "the Stan model it was written from, with Stan_model_math_functions = \"",
                          Stan_model_math_functions, "\"."))

        }
        models_folder <-  dirname(source_Stan_model_file_path)
        ##
        ## ---- Check the condition ---------------------------------------------------------------------------------------------------
        ##
        if (file.access(names = models_folder, mode = 2) != 0) {

              stop(paste0("NicoStan writes the AVX version of the Stan model next to it (as BridgeStan writes ",
                          "the compiled model there), but its folder cannot be written to: ", models_folder))

        }
        ##
        ## ---- the declarations offered to stanc: every elementwise function, and each fused log density whose -----------------------
        ##
        ##      header is in BayesMVP's header folder:
        ##
        BayesMVP_AVX_functions <-  fn_BayesMVP_AVX_functions_and_their_Stan_declarations()
        elementwise_functions <-  BayesMVP_AVX_functions$elementwise_functions
        header_file_of_each_fused_distribution <-  BayesMVP_AVX_functions$fused_distribution_BayesMVP_header_files
        fused_distribution_header_is_in_folder <-  file.exists(file.path(headers_folder_of_the_SIMD_functions,
                                                                         header_file_of_each_fused_distribution))
        fused_distributions_offered <-  names(header_file_of_each_fused_distribution)[
                                              fused_distribution_header_is_in_folder]
        fused_distribution_declarations_offered <-  BayesMVP_AVX_functions$fused_distribution_declarations[
                                                          fused_distributions_offered]
        # Stan_declarations_offered <-  c("    real bmvp_simd_lanes();",
        #                                 paste0("    vector fast_", elementwise_functions, "(vector x);"),
        #                                 unlist(fused_distribution_declarations_offered, use.names = FALSE))
        ## (with the declarations of the elementwise functions for each argument type, of the model's own
        ##  functions replaced and of std_normal as the fused normal, and the functions whose calls left on Stan
        ##  math are reported: a change of any of them writes the AVX version again)
        elementwise_function_argument_types <-  BayesMVP_AVX_functions$elementwise_function_argument_types
        Stan_declarations_offered <-  c("    real bmvp_simd_lanes();",
                                        as.vector(outer(X = elementwise_function_argument_types,
                                                        Y = elementwise_functions,
                                                        ##
                                                        ## ---- The function ----------------------------------------------------------
                                                        ##
                                                        FUN = function(argument_type, elementwise_function) {

                                                              paste0("    ", argument_type, " nico_",
                                                                     elementwise_function, "(", argument_type,
                                                                     " x);")

                                                        })),

                                        unlist(fused_distribution_declarations_offered, use.names = FALSE),
                                        BayesMVP_AVX_functions$model_functions_replaced_by_BayesMVP$
                                              model_signature,
                                        BayesMVP_AVX_functions$model_functions_replaced_by_BayesMVP$
                                              BayesMVP_declaration,
                                        paste0("// std_normal as: ",
                                               BayesMVP_AVX_functions$std_normal_as_fused_normal_declarations),
                                        paste0("// calls left on Stan math reported: ",
                                               paste(names(BayesMVP_AVX_functions[[paste0(
                                                     "math_functions_whose_calls_left_on_Stan_math_",
                                                     "are_reported")]]),
                                                     collapse = ", "),
                                               "; ",
                                               paste(names(BayesMVP_AVX_functions[[paste0(
                                                     "distributions_whose_calls_left_on_Stan_math_",
                                                     "are_reported")]]),
                                                     collapse = ", ")))
        Stan_declarations_offered_md5 <-  digest::digest(object = paste(Stan_declarations_offered,
                                                                        collapse = "\n"),
                                                         algo = "md5",
                                                         serialize = FALSE)
        Stan_model_writer_source <-  paste(c(
              deparse(fn_BayesMVP_AVX_functions_and_their_Stan_declarations),
              deparse(R_fn_write_AVX_version_of_Stan_model_with_BayesMVP_functions),
              deparse(R_fn_write_BayesMVP_AVX_version_of_Stan_model_and_user_header),
              deparse(fn_local_header_dependency_records)),
              collapse = "\n")
        Stan_model_writer_md5 <-  digest::digest(object = Stan_model_writer_source,
                                                 algo = "md5",
                                                 serialize = FALSE)
        ##
        ## ---- the AVX version, written again only when the Stan model or the declarations offered change: ---------------------------
        ##
        source_Stan_model_md5 <-  unname(tools::md5sum(files = source_Stan_model_file_path))
        AVX_Stan_model_file_name <-  paste0(tools::file_path_sans_ext(source_Stan_model_file_name), "_",
                                            Stan_model_math_functions_used, ".stan")
        AVX_Stan_model_file_path <-  file.path(models_folder, AVX_Stan_model_file_name)
        first_line_of_AVX_Stan_model <-  paste0("// ", Stan_model_math_functions_used, " version of ",
                                                source_Stan_model_file_name, " written by NicoStan ",
                                                "(source model md5 ", source_Stan_model_md5,
                                                "; NicoSIMD declarations md5 ", Stan_declarations_offered_md5,
                                                "; writer implementation md5 ", Stan_model_writer_md5, ")")
        AVX_version_is_up_to_date <-  file.exists(AVX_Stan_model_file_path) &&
                                      identical(readLines(con = AVX_Stan_model_file_path, n = 1, warn = FALSE),
                                                first_line_of_AVX_Stan_model)
        fn_write_current_AVX_model <-  function(model_file_path,
                                                 ##
                                                 ## ---- The function -----------------------------------------------------------------
                                                 ##
                                                 report_file_path) {

              bridgestan_folder <-  utils::getFromNamespace("get_bridgestan_path", "bridgestan")(download = FALSE)
              stanc_file_path <-  file.path(bridgestan_folder, "bin", "stanc")
              ##
              ## ---- Check the condition ---------------------------------------------------------------------------------------------
              ##
              if (!file.exists(stanc_file_path)) {

                    stanc_file_path <-  file.path(cmdstanr::cmdstan_path(), "bin", "stanc")

              }
              stanc_args_as_text <-  as.character(unlist(stanc_args, use.names = FALSE))
              stanc_include_paths_arguments <-  stanc_args_as_text[startsWith(x = stanc_args_as_text,
                                                                              prefix = "--include-paths")]
              return(R_fn_write_AVX_version_of_Stan_model_with_BayesMVP_functions(
                    source_Stan_model_file_path = source_Stan_model_file_path,
                    AVX_Stan_model_file_path = model_file_path,
                    first_line_of_AVX_Stan_model = first_line_of_AVX_Stan_model,
                    elementwise_functions = elementwise_functions,
                    fused_distribution_declarations = fused_distribution_declarations_offered,
                    stanc_file_path = stanc_file_path,
                    math_functions_which_may_remain_on_Stan_math = BayesMVP_AVX_functions$
                          math_functions_which_may_remain_on_Stan_math,
                    structural_functions_that_may_remain_on_Stan_math = BayesMVP_AVX_functions$
                          structural_functions_that_may_remain_on_Stan_math,
                    stan_functions_dispatching_user_functions = BayesMVP_AVX_functions$
                          stan_functions_dispatching_user_functions,
                    stanc_include_paths_arguments = stanc_include_paths_arguments,
                    elementwise_function_argument_types = elementwise_function_argument_types,
                    model_functions_replaced_by_BayesMVP = BayesMVP_AVX_functions$
                          model_functions_replaced_by_BayesMVP,
                    std_normal_as_fused_normal_declarations = BayesMVP_AVX_functions$
                          std_normal_as_fused_normal_declarations,
                    math_functions_whose_calls_left_on_Stan_math_are_reported = BayesMVP_AVX_functions$
                          math_functions_whose_calls_left_on_Stan_math_are_reported,
                    distributions_whose_calls_left_on_Stan_math_are_reported = BayesMVP_AVX_functions$
                          distributions_whose_calls_left_on_Stan_math_are_reported,
                    report_file_path_of_calls_left_on_Stan_math = report_file_path))

        }
        ##
        ## ---- Check the condition ---------------------------------------------------------------------------------------------------
        ##
        if (!AVX_version_is_up_to_date) {

              AVX_model_so_file <-  paste0(tools::file_path_sans_ext(AVX_Stan_model_file_path), "_model.so")
              AVX_model_object_file <-  paste0(tools::file_path_sans_ext(AVX_Stan_model_file_path), ".o")
              compiled_model_files <-  c(AVX_model_so_file, AVX_model_object_file)
              compiled_model_files <-  compiled_model_files[file.exists(compiled_model_files)]
              ##
              ## ---- Check the condition ---------------------------------------------------------------------------------------------
              ##
              if (length(compiled_model_files) > 0) {

                    ##
                    ## ---- Check the condition ---------------------------------------------------------------------------------------
                    ##
                    if (!file.exists(AVX_Stan_model_file_path)) {

                          stop("The compiled model has no generated source; nothing was replaced.")

                    }
                    comparison_model <-  tempfile(pattern = "NicoSIMD_source_comparison_", fileext = ".stan")
                    comparison_report <-  tempfile(pattern = "NicoSIMD_calls_comparison_", fileext = ".txt")
                    on.exit(unlink(c(comparison_model, comparison_report)), add = TRUE)
                    fn_write_current_AVX_model(model_file_path = comparison_model,
                                                report_file_path = comparison_report)
                    current_body <-  readLines(comparison_model, warn = FALSE)[-1]
                    existing_body <-  readLines(AVX_Stan_model_file_path, warn = FALSE)[-1]
                    ##
                    ## ---- Check the condition ---------------------------------------------------------------------------------------
                    ##
                    if (!identical(current_body, existing_body)) {

                          stop(paste0("The current generated source differs for ", AVX_Stan_model_file_name,
                                      "; compiled files were preserved: ",
                                      paste(compiled_model_files, collapse = ", "),
                                      ". An approved preservation and rebuild step is required."))

                    }
                    AVX_version_is_up_to_date <-  TRUE
                    message("The generated source body is identical; its compiled files were preserved.")

              }

        }
        ##
        ## ---- Check the condition ---------------------------------------------------------------------------------------------------
        ##
        if (AVX_version_is_up_to_date) {

              ## (the BayesMVP functions it uses, from its second line)
              second_line_of_AVX_Stan_model <-  readLines(con = AVX_Stan_model_file_path, n = 2, warn = FALSE)[2]
              BayesMVP_functions_used_as_text <-  sub(pattern = "^// (BayesMVP|NicoSIMD) functions used: ",
                                                      replacement = "",
                                                      x = second_line_of_AVX_Stan_model)
              BayesMVP_functions_used <-  c()
              ##
              ## ---- Check the condition ---------------------------------------------------------------------------------------------
              ##
              if (!identical(BayesMVP_functions_used_as_text, "none")) {

                    BayesMVP_functions_used <-  strsplit(x = BayesMVP_functions_used_as_text, split = ", ",
                                                         fixed = TRUE)[[1]]

              }
              message(colourise(paste0("Reusing the ", Stan_model_math_functions_used, " version of ",
                                       source_Stan_model_file_name, ": ", AVX_Stan_model_file_path),
                                "cyan"))

        ##
        ## ---- The alternative -------------------------------------------------------------------------------------------------------
        ##
        } else {

              message(colourise(paste0("Writing the ", Stan_model_math_functions_used, " version of ",
                                       source_Stan_model_file_name, " (", package_of_the_SIMD_functions,
                                       "'s SIMD functions where stanc ",
                                       "accepts them): ", AVX_Stan_model_file_path),
                                "cyan"))
              BayesMVP_functions_used <-  fn_write_current_AVX_model(
                    model_file_path = AVX_Stan_model_file_path,
                    report_file_path = paste0(tools::file_path_sans_ext(AVX_Stan_model_file_path),
                                               "_calls_left_on_Stan_math.txt"))

        }
        message(colourise(paste0(package_of_the_SIMD_functions, " functions used: ",
                                 ##
                                 ## ---- Check the condition --------------------------------------------------------------------------
                                 ##
                                 if (length(BayesMVP_functions_used) > 0) {

                                       paste(BayesMVP_functions_used, collapse = ", ")

                                 } else "none (every call keeps Stan's own function)"),
                          "cyan"))
        ##
        ## ---- the user header of the AVX version: BayesMVP's elementwise functions, the header of each fused log --------------------
        ##
        ##      density used, and the user's own header (Stan_cpp_user_header or a USER_HEADER make argument):
        ##
        make_args_given <-  as.character(unlist(make_args, use.names = FALSE))
        make_arg_is_user_header <-  grepl(pattern = "^USER_HEADER=", x = make_args_given)
        user_headers_given <-  c(Stan_cpp_user_header,
                                 sub(pattern = "^USER_HEADER=", replacement = "",
                                     x = make_args_given[make_arg_is_user_header]))
        user_headers_given <-  unique(normalizePath(path = user_headers_given, mustWork = TRUE))
        ##
        ## ---- Check the condition ---------------------------------------------------------------------------------------------------
        ##
        if (length(user_headers_given) > 1) {

              stop(paste0("Stan_cpp_user_header and the USER_HEADER of make_args give different user headers (",
                          paste(user_headers_given, collapse = ", "), "); give the user header in one of the ",
                          "two only."))

        }
        fused_distributions_used <-  intersect(BayesMVP_functions_used,
                                               names(header_file_of_each_fused_distribution))
        fused_distribution_headers_used <-  unique(file.path(headers_folder_of_the_SIMD_functions,
                                                             header_file_of_each_fused_distribution[
                                                                   fused_distributions_used]))
        AVX_user_header_dependency_paths <-  c(elementwise_functions_header_of_the_SIMD_functions,
                                               scalar_functions_header_of_the_SIMD_functions,
                                               fused_distribution_headers_used,
                                               user_headers_given)
        AVX_user_header_dependency_records <-  fn_local_header_dependency_records(
              header_paths                 = AVX_user_header_dependency_paths,
              NicoSIMD_headers_folder      = headers_folder_of_the_SIMD_functions)
        AVX_user_header_lines <-  c("#pragma once",
                                    "// Local records cover NicoSIMD and user headers.",
                                    "// External compiler headers remain builder-managed.",
                                    paste0("// User header of ", AVX_Stan_model_file_name,
                                           ", written by NicoStan: ", package_of_the_SIMD_functions,
                                           "'s SIMD functions",
                                           if (length(user_headers_given) == 1) " and the user's own header"),
                                    AVX_user_header_dependency_records,
                                    paste0("#include ",
                                           encodeString(x = elementwise_functions_header_of_the_SIMD_functions,
                                                        quote = "\"")))
        AVX_user_header_lines <-  c(AVX_user_header_lines,
                                    paste0("#include ",
                                           encodeString(x = scalar_functions_header_of_the_SIMD_functions,
                                                        quote = "\"")))
        ##
        ## ---- Check the condition ---------------------------------------------------------------------------------------------------
        ##
        if (length(fused_distribution_headers_used) > 0) {

              AVX_user_header_lines <-  c(AVX_user_header_lines,
                                          paste0("#include ", encodeString(x = fused_distribution_headers_used,
                                                                           quote = "\"")))

        }
        ##
        ## ---- Check the condition ---------------------------------------------------------------------------------------------------
        ##
        if (length(user_headers_given) == 1) {

              AVX_user_header_lines <-  c(AVX_user_header_lines,
                                          paste0("#include ", encodeString(x = user_headers_given, quote = "\"")))

        }
        AVX_user_header_file <-  paste0(tools::file_path_sans_ext(AVX_Stan_model_file_path), "_user_header.hpp")
        AVX_user_header_is_up_to_date <-  file.exists(AVX_user_header_file) &&
                                          identical(readLines(con = AVX_user_header_file, warn = FALSE),
                                                    AVX_user_header_lines)
        ##
        ## ---- Check the condition ---------------------------------------------------------------------------------------------------
        ##
        if (!AVX_user_header_is_up_to_date) {

              AVX_model_so_file <-  paste0(tools::file_path_sans_ext(AVX_Stan_model_file_path), "_model.so")
              AVX_model_object_file <-  paste0(tools::file_path_sans_ext(AVX_Stan_model_file_path), ".o")
              compiled_model_files <-  c(AVX_model_so_file, AVX_model_object_file)
              compiled_model_files <-  compiled_model_files[file.exists(compiled_model_files)]
              ##
              ## ---- Check the condition ---------------------------------------------------------------------------------------------
              ##
              if (length(compiled_model_files) > 0) {

                    stop(paste0("The local header dependency record for ", AVX_Stan_model_file_name,
                                " is missing or changed, but compiled files already exist: ",
                                paste(compiled_model_files, collapse = ", "),
                                ". Stop before reuse; an approved preservation and rebuild step is required."))

              }
              temporary_file <-  tempfile(pattern = "AVX_user_header_being_written_", tmpdir = models_folder,
                                          fileext = ".hpp.tmp")
              writeLines(text = AVX_user_header_lines, con = temporary_file)
              ##
              ## ---- Check the condition ---------------------------------------------------------------------------------------------
              ##
              if (!file.rename(from = temporary_file, to = AVX_user_header_file)) {

                    stop(paste0("Could not write the user header of the AVX version: ", AVX_user_header_file))

              }

        }
        ##
        ## ---- the make arguments of the instruction set (those of NicoStan's examples): ---------------------------------------------
        ##
        make_args_without_user_header <-  make_args_given[!make_arg_is_user_header]
        names_of_make_args_given <-  sub(pattern = "=.*$", replacement = "", x = make_args_without_user_header)
        make_arg_names_set_for_instruction_set <-  c("CXXFLAGS_OPTIM", "CPPFLAGS_OPTIM", "PRECOMPILED_HEADERS")
        make_arg_is_set_for_instruction_set <-  (names_of_make_args_given %in%
                                                 make_arg_names_set_for_instruction_set)
        ##
        ## ---- Check the condition ---------------------------------------------------------------------------------------------------
        ##
        if (any(make_arg_is_set_for_instruction_set)) {

              stop(paste0("make_args hold ",
                          paste(make_args_without_user_header[make_arg_is_set_for_instruction_set],
                                collapse = ", "),
                          ", which Stan_model_math_functions = \"", Stan_model_math_functions, "\" sets ",
                          "itself (CXXFLAGS_OPTIM with the instruction-set flags, ",
                          "CPPFLAGS_OPTIM=-DBAYESMVP_FORCE_AVX2 for AVX2, PRECOMPILED_HEADERS=false); remove ",
                          "them from make_args."))

        }
        ##
        ## ---- Check the condition ---------------------------------------------------------------------------------------------------
        ##
        instruction_set_flags <-  if (identical(instruction_set_used, "AVX512")) {

              "-O3 -march=native -mtune=native -mfma -mavx2 -mavx512f -mavx512vl -mavx512dq"

        } else "-O3 -march=native -mtune=native -mfma -mavx2"
        make_args_of_AVX_version <-  make_args_without_user_header
        ##
        ## ---- Check the condition ---------------------------------------------------------------------------------------------------
        ##
        if (!("STAN_THREADS" %in% names_of_make_args_given)) {

              make_args_of_AVX_version <-  c(make_args_of_AVX_version, "STAN_THREADS=true")

        }
        make_args_of_AVX_version <-  c(make_args_of_AVX_version,
                                       "PRECOMPILED_HEADERS=false",
                                       paste0("CXXFLAGS_OPTIM=", shQuote(instruction_set_flags)))
        ##
        ## ---- Check the condition ---------------------------------------------------------------------------------------------------
        ##
        if (identical(instruction_set_used, "AVX2")) {

              make_args_of_AVX_version <-  c(make_args_of_AVX_version, "CPPFLAGS_OPTIM=-DBAYESMVP_FORCE_AVX2")

        }
        ##
        return(list( Stan_model_math_functions        = Stan_model_math_functions,
                     Stan_model_math_functions_used   = Stan_model_math_functions_used,
                     source_Stan_model_file_path      = source_Stan_model_file_path,
                     Stan_model_file_path             = AVX_Stan_model_file_path,
                     Stan_cpp_user_header             = AVX_user_header_file,
                     make_args                        = as.list(make_args_of_AVX_version),
                     package_of_the_SIMD_functions    = package_of_the_SIMD_functions,
                     headers_folder_of_the_SIMD_functions  = headers_folder_of_the_SIMD_functions,
                     BayesMVP_AVX_headers_folder      = BayesMVP_AVX_headers_folder,
                     BayesMVP_functions_used          = BayesMVP_functions_used,
                     SIMD_lanes_expected              = SIMD_lanes_expected))

}




##
## ---- NicoSIMD's names in the AVX version of a Stan model written with BayesMVP's names ---------------------------------------------
##
#' Put NicoSIMD's names (nico_) in place of BayesMVP's (fast_) in the AVX version of a Stan model
#'
#' NicoSIMD holds BayesMVP's Stan-facing SIMD functions, with the same C++ and the names changed from fast_ to
#' nico_; its file names_renamed_from_fast_to_nico.txt lists every name changed. In the AVX version written with
#' BayesMVP's names, each name of that list, as a whole name (never inside a longer name), becomes its nico_ name,
#' in every line but the first (which records the source model and the declarations offered); the second line then
#' names NicoSIMD's functions. The AVX version is then checked again with stanc.
#' @param AVX_Stan_model_file_path The AVX version, written with BayesMVP's names.
#' @param NicoSIMD_headers_folder The folder of NicoSIMD's headers (with names_renamed_from_fast_to_nico.txt).
#' @param stanc_file_path The stanc compiler.
#' @param stanc_include_paths_arguments The --include-paths arguments of the model's stanc arguments.
#' @return The lines of the AVX version with NicoSIMD's names (invisibly).
#' @keywords internal
R_fn_put_NicoSIMD_names_in_place_of_BayesMVP_names_in_AVX_version_of_Stan_model <-  function(
                                                                          AVX_Stan_model_file_path,
                                                                          NicoSIMD_headers_folder,
                                                                          stanc_file_path,
                                                                          stanc_include_paths_arguments
##
## ---- The function ------------------------------------------------------------------------------------------------------------------
##
) {

        file_of_the_names_renamed <-  file.path(NicoSIMD_headers_folder, "names_renamed_from_fast_to_nico.txt")
        ##
        ## ---- Check the condition ---------------------------------------------------------------------------------------------------
        ##
        if (!file.exists(file_of_the_names_renamed)) {

              stop(paste0("NicoSIMD's list of the names changed from fast_ to nico_ is not in ",
                          NicoSIMD_headers_folder, ": reinstall NicoSIMD."))

        }
        BayesMVP_names <-  readLines(con = file_of_the_names_renamed, warn = FALSE)
        BayesMVP_names <-  BayesMVP_names[nzchar(BayesMVP_names)]
        ## (the longest first, so that a name is never changed inside a longer one)
        BayesMVP_names <-  BayesMVP_names[order(nchar(BayesMVP_names), decreasing = TRUE)]
        ##
        lines_of_AVX_version <-  readLines(con = AVX_Stan_model_file_path, warn = FALSE)
        lines_after_the_first <-  lines_of_AVX_version[-1]
        ##
        ## ---- Each item -------------------------------------------------------------------------------------------------------------
        ##
        for (BayesMVP_name in BayesMVP_names) {

              NicoSIMD_name <-  sub(pattern = "^fast_", replacement = "nico_", x = BayesMVP_name)
              lines_after_the_first <-  gsub(pattern = paste0("(?<![A-Za-z0-9_.])", BayesMVP_name,
                                                              "(?![A-Za-z0-9_])"),
                                             replacement = NicoSIMD_name,
                                             x = lines_after_the_first,
                                             perl = TRUE)

        }
        lines_after_the_first[1] <-  sub(pattern = "^// BayesMVP functions used: ",
                                         replacement = "// NicoSIMD functions used: ",
                                         x = lines_after_the_first[1])
        lines_of_AVX_version <-  c(lines_of_AVX_version[1], lines_after_the_first)
        ##
        ## ---- written through a temporary file next to it, then checked with stanc: -------------------------------------------------
        ##
        temporary_file <-  tempfile(pattern = "AVX_version_with_NicoSIMD_names_",
                                    tmpdir = dirname(AVX_Stan_model_file_path), fileext = ".stan.tmp")
        writeLines(text = lines_of_AVX_version, con = temporary_file)
        ##
        ## ---- Check the condition ---------------------------------------------------------------------------------------------------
        ##
        if (!file.rename(from = temporary_file, to = AVX_Stan_model_file_path)) {

              stop(paste0("Could not write the AVX version with NicoSIMD's names: ", AVX_Stan_model_file_path))

        }
        stanc_output_file <-  tempfile(pattern = "AVX_version_with_NicoSIMD_names_", fileext = ".hpp")
        stanc_output <-  suppressWarnings(system2( command = stanc_file_path,
                                                   args    = c("--allow-undefined",
                                                               stanc_include_paths_arguments,
                                                               paste0("--o=", shQuote(stanc_output_file)),
                                                               shQuote(AVX_Stan_model_file_path)),
                                                   stdout  = TRUE,
                                                   stderr  = TRUE))
        ##
        ## ---- Check the condition ---------------------------------------------------------------------------------------------------
        ##
        if (!is.null(attr(stanc_output, "status")) || !file.exists(stanc_output_file)) {

              stop(paste0("stanc does not accept the AVX version with NicoSIMD's names (",
                          AVX_Stan_model_file_path, "):\n", paste(stanc_output, collapse = "\n")))

        }
        ##
        return(invisible(lines_of_AVX_version))

}




##
## ---- the SIMD lanes which the compiled AVX version of a Stan model reports ---------------------------------------------------------
##
#' Check that the compiled AVX version of a Stan model reports the SIMD lanes of its instruction set
#'
#' Reads the generated quantity simd_lanes (bmvp_simd_lanes(): 8 with BayesMVP's AVX-512 kernels, 4 with the AVX2
#' kernels) of the compiled model, with BridgeStan, and stops when it differs from the instruction set's. The
#' generated quantities are evaluated at 0 on the unconstrained scale, or, if they cannot be evaluated there, at
#' the points where every coordinate is 0.5, -0.5, 1 or -1.
#' @param bs_model The BridgeStan model of the compiled AVX version.
#' @param SIMD_lanes_expected 8 (AVX-512) or 4 (AVX2).
#' @param Stan_model_math_functions_used "NicoSIMD_AVX512", "NicoSIMD_AVX2", "BayesMVP_AVX512" or
#'   "BayesMVP_AVX2".
#' @param AVX_Stan_model_file_path The AVX version (for the messages).
#' @return The SIMD lanes reported.
#' @keywords internal
R_fn_check_SIMD_lanes_reported_by_compiled_BayesMVP_AVX_Stan_model <-  function( bs_model,
                                                                                SIMD_lanes_expected,
                                                                                Stan_model_math_functions_used,
                                                                                AVX_Stan_model_file_path
##
## ---- The function ------------------------------------------------------------------------------------------------------------------
##
) {

        AVX_Stan_model_file_name <-  basename(AVX_Stan_model_file_path)
        parameter_and_generated_quantity_names <-  bs_model$param_names(include_tp = FALSE, include_gq = TRUE)
        position_of_simd_lanes <-  which(parameter_and_generated_quantity_names == "simd_lanes")
        ##
        ## ---- Check the condition ---------------------------------------------------------------------------------------------------
        ##
        if (length(position_of_simd_lanes) != 1) {

              stop(paste0("The compiled model of ", AVX_Stan_model_file_name, " has no generated quantity ",
                          "simd_lanes, so its SIMD lanes cannot be checked."))

        }
        n_unconstrained <-  bs_model$param_unc_num()
        points_tried <-  list(rep(0, n_unconstrained),
                              rep(0.5, n_unconstrained),
                              rep(-0.5, n_unconstrained),
                              rep(1, n_unconstrained),
                              rep(-1, n_unconstrained))
        ##
        SIMD_lanes_reported <-  NA
        evaluation_errors <-  c()
        ##
        ## ---- Each item -------------------------------------------------------------------------------------------------------------
        ##
        for (point in points_tried) {

              constrained_values <-  tryCatch(expr = bs_model$param_constrain( theta_unc   = point,
                                                                               include_tp  = FALSE,
                                                                               include_gq  = TRUE,
                                                                               rng         = bs_model$new_rng(
                                                                                                   seed = 1)),
                                              error = function(condition) conditionMessage(condition))
              ##
              ## ---- Check the condition ---------------------------------------------------------------------------------------------
              ##
              if (is.numeric(constrained_values)) {

                    SIMD_lanes_reported <-  constrained_values[position_of_simd_lanes]
                    break

              }

              evaluation_errors <-  c(evaluation_errors, constrained_values)

        }
        ##
        ## ---- Check the condition ---------------------------------------------------------------------------------------------------
        ##
        if (is.na(SIMD_lanes_reported)) {

              stop(paste0("The generated quantities of ", AVX_Stan_model_file_name, " could not be evaluated at ",
                          "any of the points tried, so its SIMD lanes cannot be checked:\n",
                          paste(unique(evaluation_errors), collapse = "\n")))

        }
        ##
        ##
        ## ---- Check the condition ---------------------------------------------------------------------------------------------------
        ##
        if (SIMD_lanes_reported != SIMD_lanes_expected) {

              stop(paste0("The compiled model of ", AVX_Stan_model_file_name, " reports ", SIMD_lanes_reported,
                          " SIMD lanes, but ", Stan_model_math_functions_used, " needs ", SIMD_lanes_expected,
                          ": it was not built with the kernels of this instruction set (e.g. a model library ",
                          "built earlier with other flags). Remove ",
                          paste0(tools::file_path_sans_ext(AVX_Stan_model_file_path), "_model.so"),
                          " to build it again."))

        }
        message(colourise(paste0(AVX_Stan_model_file_name, ": the compiled model reports ", SIMD_lanes_reported,
                                 " SIMD lanes, as ", Stan_model_math_functions_used, " needs."),
                          "green"))
        ##
        return(SIMD_lanes_reported)

}




















