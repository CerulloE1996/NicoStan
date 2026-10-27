#pragma once

#include "stan_external_scalar_functions.hpp"
#include "stan_external_lgamma.hpp"

namespace nico::math {

template <bool propto__, typename T_y>
inline stan::return_type_t<T_y>
nico_inv_gamma_lpdf(const T_y& y,
                    const double alpha,
                    const double beta,
                    std::ostream* pstream__) {

    static_assert(std::is_arithmetic<T_y>::value || stan::is_var<T_y>::value,
                  "The outcome must be an arithmetic or reverse-mode scalar.");
    static constexpr const char* function = "inv_gamma_lpdf";
    const double y_value = stan::math::value_of(y);
    stan::math::check_not_nan(function, "Random variable", y_value);
    stan::math::check_positive_finite(function, "Shape parameter", alpha);
    stan::math::check_positive_finite(function, "Scale parameter", beta);
    if constexpr (!stan::math::include_summand<propto__, T_y>::value) {
        return 0.0;
    }
    if (y_value <= 0.0) {
        return stan::math::LOG_ZERO;
    }

    auto ops_partials = stan::math::make_partials_propagator(y);
    double log_prob = 0.0;
    if constexpr (!propto__) {
        log_prob -= nico_lgamma(alpha, pstream__);
        log_prob += alpha * nico_log(beta, pstream__);
    }
    log_prob -= (alpha + 1.0) * nico_log(y_value, pstream__);
    const double inv_y = 1.0 / y_value;
    log_prob -= beta * inv_y;
    if constexpr (!stan::is_constant_all<T_y>::value) {
        stan::math::partials<0>(ops_partials)[0] = (beta * inv_y - alpha - 1.0) * inv_y;
    }
    return ops_partials.build(log_prob);

}

}  // namespace nico::math

using nico::math::nico_inv_gamma_lpdf;






















