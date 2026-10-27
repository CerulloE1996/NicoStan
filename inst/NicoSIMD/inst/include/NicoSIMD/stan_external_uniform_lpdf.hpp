#pragma once

#include <limits>
#include <ostream>
#include <type_traits>

#include "stan_external_scalar_functions.hpp"
#include "stan_external_fused_normal_lpdf.hpp"

namespace nico::math {

// uniform(y | alpha, beta): -log(beta - alpha) for each y in [alpha, beta], -inf outside (as Stan's uniform_lpdf);
// y a real or a vector, alpha and beta reals; the log is NicoSIMD's
template <bool propto__, typename T_y, typename T_low, typename T_high>
inline std::enable_if_t<
    (fused_normal::is_supported_scalar<T_y>::value || stan::is_eigen_col_vector<T_y>::value)
        && fused_normal::is_supported_scalar<T_low>::value
        && fused_normal::is_supported_scalar<T_high>::value,
    stan::return_type_t<T_y, T_low, T_high>>
nico_uniform_lpdf(const T_y& y,
                  const T_low& alpha,
                  const T_high& beta,
                  std::ostream* pstream__) {

    static constexpr const char* function = "uniform_lpdf";
    const auto& y_ref = stan::math::to_ref(y);
    const double alpha_value = stan::math::value_of(alpha);
    const double beta_value = stan::math::value_of(beta);

    stan::math::check_not_nan(function, "Random variable", stan::math::value_of(y_ref));
    stan::math::check_finite(function, "Lower bound parameter", alpha_value);
    stan::math::check_finite(function, "Upper bound parameter", beta_value);
    stan::math::check_greater(function, "Upper bound parameter", beta_value, alpha_value);

    const double n = static_cast<double>(stan::math::size(y_ref));
    auto ops_partials = stan::math::make_partials_propagator(y_ref, alpha, beta);
    stan::scalar_seq_view<std::decay_t<decltype(y_ref)>> y_view(y_ref);
    for (size_t i = 0; i < stan::math::size(y_ref); ++i) {
        const double y_i = stan::math::value_of(y_view[i]);
        if (y_i < alpha_value || y_i > beta_value) return ops_partials.build(stan::math::LOG_ZERO);
    }

    double log_prob = 0.0;
    if constexpr (stan::math::include_summand<propto__, T_low, T_high>::value) {
        log_prob = -n * nico_log(beta_value - alpha_value, pstream__);
    }
    if constexpr (!stan::is_constant_all<T_low>::value) {
        stan::math::partials<1>(ops_partials)[0] = n / (beta_value - alpha_value);
    }
    if constexpr (!stan::is_constant_all<T_high>::value) {
        stan::math::partials<2>(ops_partials)[0] = -n / (beta_value - alpha_value);
    }

    return ops_partials.build(log_prob);

}

}  // namespace nico::math

using nico::math::nico_uniform_lpdf;
