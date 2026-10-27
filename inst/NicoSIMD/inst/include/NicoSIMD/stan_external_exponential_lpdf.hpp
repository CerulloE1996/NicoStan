#pragma once

#include <ostream>
#include <type_traits>

#include "stan_external_scalar_functions.hpp"
#include "stan_external_fused_normal_lpdf.hpp"

namespace nico::math {

template <bool propto__, typename T_y, typename T_rate>
inline std::enable_if_t<
    fused_normal::is_supported_scalar<T_y>::value
        && fused_normal::is_supported_scalar<T_rate>::value,
    stan::return_type_t<T_y, T_rate>>
nico_exponential_lpdf(const T_y& y,
                      const T_rate& rate,
                      std::ostream* pstream__) {

    static constexpr const char* function = "exponential_lpdf";
    const double y_value = stan::math::value_of(y);
    const double rate_value = stan::math::value_of(rate);

    stan::math::check_nonnegative(function, "Random variable", y_value);
    stan::math::check_positive_finite(function, "Inverse scale parameter", rate_value);

    auto ops_partials = stan::math::make_partials_propagator(y, rate);
    double log_prob = 0.0;

    if constexpr (stan::math::include_summand<propto__, T_rate>::value) {
        log_prob = nico_log(rate_value, pstream__);
    }
    if constexpr (stan::math::include_summand<propto__, T_y, T_rate>::value) {
        log_prob -= rate_value * y_value;
    }

    if constexpr (!stan::is_constant_all<T_y>::value) {
        stan::math::partials<0>(ops_partials)[0] = -rate_value;
    }
    if constexpr (!stan::is_constant_all<T_rate>::value) {
        stan::math::partials<1>(ops_partials)[0] = 1.0 / rate_value - y_value;
    }

    return ops_partials.build(log_prob);

}

}  // namespace nico::math

using nico::math::nico_exponential_lpdf;






















