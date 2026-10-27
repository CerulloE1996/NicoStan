#pragma once
//
// ---- Double-precision SIMD exponential, shared by the Gaussian process covariance functions.
//      The range reduction, polynomial and exceptional lanes are the existing general GP implementation.
//
#ifndef BMVP_LANES
#  error "Include stan_external_functions.hpp before this SIMD exponential helper"
#endif

namespace nico::math {

//// simd_t is the selected SIMD register: four doubles on AVX2, eight doubles on AVX-512.
inline simd_t nico_simd_exp_in_double_precision(const simd_t argument) {

#if defined(NICO_SIMD_SCALAR_KERNELS)
    return std::exp(argument);
#else
    return BMVP_KERNEL_NAME(nico_exp_1)(argument);
#endif

}

}  // namespace nico::math























