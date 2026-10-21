#pragma once
//
// -| --------- NicoSIMD: the AVX2 / AVX-512 math, fused log densities and fused general functions, exposed to Stan
// -| --------- as external functions (the one file a Stan model's user header includes) ------------------------
//
// Every function here is a nico_ function: the same C++ as the BayesMVP function of the same name with fast_ in
// place of nico_ (e.g. nico_normal_id_glm_lpdf is nico_normal_id_glm_lpdf), so that the log density and the
// gradient of a model are bit for bit those of its BayesMVP version. The list of every function, its header, kind
// and overloads: list_of_every_function_moved_from_BayesMVP_to_NicoSIMD.csv (in the outer package folder).
//
// Declared in the Stan functions block as e.g.   vector nico_Phi(vector x);   (no body), and compiled with
// stanc --allow-undefined and this file as the user header (NicoSIMD's
// fn_include_folder_Stan_user_header_and_build_arguments_of_NicoSIMD_for_this_CPU() gives the path, the compiler
// flags and the stanc argument).
//
// Kernel dispatch, at compile time (as in BayesMVP): AVX-512 (8 lanes) with -mavx512f -mavx512vl -mavx512dq, else
// AVX2 (4 lanes) with -mavx2; anything less is a compile error. -DBAYESMVP_FORCE_AVX2 selects the AVX2 kernels on
// an AVX-512 CPU (the flag keeps its BayesMVP name, so that the build arguments of both packages are the same);
// bmvp_simd_lanes() returns 4 or 8.
//
// One translation unit includes either BayesMVP's stan_external_functions.hpp or this file, never both: the
// internal C++ (namespace nico::math, the SIMD kernels fast_*_AVX2 / fast_*_AVX512 and the BMVP_* macros) is
// the same in both, and only the Stan-facing names differ.
//
// ---------------------------------------------------------------------------------------------------------------

// ---- the elementwise math, the fused log densities and the fused general functions that BayesMVP's
//      stan_external_functions.hpp includes (the same file, with the nico_ names)
#include "stan_external_functions.hpp"

// ---- the general functions written in NicoSIMD itself (every new general function is NicoSIMD's only)
//
// (the Bernoulli GLM with a tensor-product basis term, logit and probit links: e.g. a Hilbert space approximate
//  Gaussian process in two dimensions, a tensor-product spline, any separable basis of two variables)
#include "stan_external_fused_bernoulli_glm_with_a_tensor_product_basis_lpmf.hpp"
//
// (the binomial log probability of counts with their own probabilities: e.g. the counts above each threshold of
//  ordinal and continuation-ratio models, dose-response data, aggregated Bernoulli data)
#include "stan_external_binomial_lpmf.hpp"
//
// (the binomial counts above successive cutpoints of a cumulative link, probit or logit: e.g. ordinal models of
//  counts reported at some or all of the cutpoints of each group, ordinal meta-analysis of test accuracy)
#include "stan_external_binomial_counts_above_successive_cutpoints_of_a_cumulative_link.hpp"
//
// (the induced-Dirichlet prior of ordered cutpoints, probit or logit: any cumulative-link ordinal model with an
//  induced-Dirichlet prior on its cutpoints)
#include "stan_external_induced_dirichlet_prior_of_ordered_cutpoints.hpp"
//
// (the log determinant of the Jacobian of ordered cutpoints built from the first one plus the exp, or softplus,
//  of raw differences: any cumulative-link ordinal model with cutpoints parameterised this way)
#include "stan_external_log_det_jacobian_of_cutpoints_from_raw_differences.hpp"
//
// (the factorised binomial log-likelihood of ordinal test accuracy data and its latent locations, in double
//  precision for the generated quantities: MetaOrdDTA's compute_log_lik_binomial_fact_data and
//  map_latent_surv_to_random_hetero_C)
#include "stan_external_factorised_binomial_log_lik_of_ordinal_test_accuracy_data.hpp"

// ---- the headers that BayesMVP's stan_external_functions.hpp does not include: a model that uses one of their
//      functions includes it in its own user header, after this file, as its BayesMVP version does now
//
// (the affine predictor and affine recurrence reductions, e.g. any AR / state-space model)
// #include "stan_external_affine_block_reductions.hpp"
//
// (the Cholesky factor with the SIMD rank-k update, and with Eigen's own core)
// #include "stan_external_cholesky_decompose.hpp"
// #include "stan_external_cholesky_decompose_native_core.hpp"
//
// (the squared exponential covariance plus a diagonal term and its Cholesky factor, in one composition)
// #include "stan_external_gp_exp_quad_cov_cholesky.hpp"
//
// (the dot products of an affine predictor with indexed effects and its log cumulative sum of exponentials, with
//  its blocks of rows run in parallel on the threads of the chain: an overload with one more argument)
// #include "stan_external_dot_products_of_affine_predictor_with_indexed_effects_and_log_cumulative_sum_exp_in_parallel_blocks_of_rows.hpp"
//
// (the normal log density with an affine location and an affine log scale)
// #include "stan_external_fused_normal_affine_location_and_log_scale_lpdf.hpp"
//
// (the truncated-normal quantile and log mass, e.g. any probit / GHK / Tobit / interval-censored model)
// #include "stan_external_truncated_normal_quantile_and_log_mass.hpp"
//
// ---- the headers being written in BayesMVP now; their nico_ copies are made with the next run of
//      tools/copy_the_Stan_facing_headers_of_BayesMVP_into_NicoSIMD_with_nico_names.R once they are finished
//
// (the fused ordered logistic GLM, as Stan's ordered_logistic_glm)
// #include "stan_external_fused_ordered_logistic_glm_lpmf.hpp"
//
// (the fused marginal log likelihood of a Gaussian process with the squared exponential kernel and a normal
//  likelihood)
// #include "stan_external_gp_exp_quad_cov_normal_marginal_lpdf.hpp"























