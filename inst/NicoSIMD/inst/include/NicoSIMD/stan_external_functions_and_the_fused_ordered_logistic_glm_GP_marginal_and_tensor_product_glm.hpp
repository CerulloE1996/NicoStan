#pragma once
//
// -| --------- NicoSIMD's elementwise math, fused log densities and fused general functions, with the fused ordered
// -| --------- logistic GLM, the GP marginal and the Bernoulli GLM with a tensor-product basis -------------------
//
// The set of headers of stan_external_functions_with_the_fast_names_of_BayesMVP.hpp, without its layer of fast_
// names (every AVX function is called by its nico_ name, never by a fast_ name).
//
#include "stan_external_functions.hpp"
#include "stan_external_fused_ordered_logistic_glm_lpmf.hpp"
#include "stan_external_gp_exp_quad_cov_normal_marginal_lpdf.hpp"
#include "stan_external_fused_bernoulli_glm_with_a_tensor_product_basis_lpmf.hpp"






















