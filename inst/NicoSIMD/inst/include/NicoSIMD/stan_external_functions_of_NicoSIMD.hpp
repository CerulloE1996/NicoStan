#pragma once
// ---- Native scalar and SIMD math, densities and general functions exposed to Stan:

#include "stan_external_functions.hpp"
#include "stan_external_scalar_functions.hpp"
#include "stan_external_lognormal_lpdf.hpp"
//
// (lub_constrain of a matrix of vars on the NicoSIMD kernels: only when NICOSIMD_SIMD_LUB_CONSTRAIN is defined)
#include "stan_external_lub_constrain.hpp"

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
#include "stan_external_dirichlet_cutpoint_median.hpp"
//
// (the Dirichlet, Beta and LKJ-Cholesky log densities: e.g. the Dirichlet and Beta priors on the category and
//  cutpoint probabilities of ordinal models, LKJ priors on correlation matrices)
#include "stan_external_dirichlet_lpdf.hpp"
#include "stan_external_beta_lpdf.hpp"
#include "stan_external_lkj_corr_cholesky_lpdf.hpp"
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























