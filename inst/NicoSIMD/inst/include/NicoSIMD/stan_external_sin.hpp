#pragma once

#include "stan_external_functions.hpp"
#if BMVP_LANES == 8
#include "../simde/x86/avx512/loadu.h"
#include "../simde/x86/avx512/storeu.h"
#endif
#include "../simde/x86/svml.h"

namespace nico::math {

inline double nico_sin(const double x,
                       std::ostream* /*pstream__*/) {

    return std::sin(x);

}

inline Eigen::Matrix<double, Eigen::Dynamic, 1>
nico_sin(const Eigen::Matrix<double, Eigen::Dynamic, 1>& x,
         std::ostream* /*pstream__*/) {

    Eigen::Matrix<double, Eigen::Dynamic, 1> result(x.size());
    Eigen::Index i = 0;
#if BMVP_LANES == 8
    for (; i + BMVP_LANES <= x.size(); i += BMVP_LANES) {
        simde_mm512_storeu_pd(result.data() + i, simde_mm512_sin_pd(simde_mm512_loadu_pd(x.data() + i)));
    }
#elif BMVP_LANES == 4
    for (; i + BMVP_LANES <= x.size(); i += BMVP_LANES) {
        simde_mm256_storeu_pd(result.data() + i, simde_mm256_sin_pd(simde_mm256_loadu_pd(x.data() + i)));
    }
#endif
    for (; i < x.size(); ++i) {
        result[i] = std::sin(x[i]);
    }
    return result;

}

}  // namespace nico::math

using nico::math::nico_sin;






















