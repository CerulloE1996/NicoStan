#pragma once

#include <cmath>
#include <ostream>
#include <Eigen/Core>

namespace nico::math {

inline double nico_lgamma(const double x,
                          std::ostream* /*pstream__*/) {

    int sign = 1;
    return ::lgamma_r(x, &sign);

}

// (the vector overload, one lgamma_r per value, commented out: the SIMD one is in
//  stan_external_lgamma_and_digamma_simd.hpp, included below)
// inline Eigen::Matrix<double, Eigen::Dynamic, 1>
// nico_lgamma(const Eigen::Matrix<double, Eigen::Dynamic, 1>& x,
//              std::ostream* pstream__) {
//
//     Eigen::Matrix<double, Eigen::Dynamic, 1> result(x.size());
//     for (Eigen::Index i = 0; i < x.size(); ++i) {
//         result[i] = nico_lgamma(x[i], pstream__);
//     }
//     return result;
//
// }

}  // namespace nico::math

using nico::math::nico_lgamma;

#include "stan_external_lgamma_and_digamma_simd.hpp"






















