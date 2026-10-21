//// user_header_of_the_fused_variant_with_NicoSIMD.hpp
////
//// ---- The user header of models/four_class_avx_fused.stan (compile it with --allow-undefined and this header):
////      NicoSIMD's truncated-normal quantile and log mass (the GHK step of both its branches), whose header
////      includes NicoSIMD's stan_external_functions.hpp (nico_Phi, nico_log_Phi, nico_inv_Phi, nico_exp, ... and
////      bmvp_simd_lanes, the other functions the model declares). This header holds no SIMD code: every SIMD
////      THE ROOT"). It takes the place of the example's own copies in include/ (nico_normal_interval and
////      nico_normal_binary, and the copies of BayesMVP's headers).
////
#pragma once
#if __has_include(<NicoSIMD/stan_external_truncated_normal_quantile_and_log_mass.hpp>)
#include <NicoSIMD/stan_external_truncated_normal_quantile_and_log_mass.hpp>
#elif __has_include("../../../NicoSIMD/include/NicoSIMD/stan_external_truncated_normal_quantile_and_log_mass.hpp")
//// (no include path to NicoSIMD: the installed NicoSIMD beside the installed NicoStan in the same library)
#include "../../../NicoSIMD/include/NicoSIMD/stan_external_truncated_normal_quantile_and_log_mass.hpp"
#else
//// (the development tree: NicoSIMD in R_packages/NicoStan/inst/NicoSIMD/inst/include/)
#include "../../NicoSIMD/inst/include/NicoSIMD/stan_external_truncated_normal_quantile_and_log_mass.hpp"
#endif






















