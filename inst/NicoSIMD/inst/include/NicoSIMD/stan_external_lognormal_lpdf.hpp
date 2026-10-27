#pragma once

#include <ostream>
#include <type_traits>

#include "stan_external_scalar_functions.hpp"
#include "stan_external_fused_normal_lpdf.hpp"

namespace nico::math {

template <bool propto__, typename T_y, typename T_loc, typename T_scale>
inline std::enable_if_t<
    fused_normal::is_supported_scalar<T_y>::value
        && fused_normal::is_supported_scalar<T_loc>::value
        && fused_normal::is_supported_scalar<T_scale>::value,
    stan::return_type_t<T_y, T_loc, T_scale>>
nico_lognormal_lpdf(const T_y& y,
                    const T_loc& mu,
                    const T_scale& sigma,
                    std::ostream* pstream__) {

    static constexpr const char* function = "lognormal_lpdf";

    const auto y_value = stan::math::value_of(y);
    const auto mu_value = stan::math::value_of(mu);
    const auto sigma_value = stan::math::value_of(sigma);

    stan::math::check_nonnegative(function, "Random variable", y_value);
    stan::math::check_finite(function, "Location parameter", mu_value);
    stan::math::check_positive_finite(function, "Scale parameter", sigma_value);

    if (!stan::math::include_summand<propto__, T_y, T_loc, T_scale>::value) {
        return 0;
    }

    auto ops_partials = stan::math::make_partials_propagator(y, mu, sigma);

    if (y_value == 0) {
        return ops_partials.build(stan::math::LOG_ZERO);
    }

    const auto log_y = nico_log(y, pstream__);
    auto lognormal_value = nico_normal_lpdf<propto__>(log_y, mu, sigma, pstream__);

    if constexpr (propto__ && !stan::is_constant_all<T_y, T_loc, T_scale>::value) {
        lognormal_value += stan::math::NEG_LOG_SQRT_TWO_PI;
    }

    if (stan::math::include_summand<propto__, T_y>::value) {
        return lognormal_value - log_y;
    }

    return lognormal_value;

}

template <bool propto__, typename T_y, typename T_loc, typename T_scale>
inline std::enable_if_t<
    (fused_normal::is_supported_scalar<T_y>::value || fused_normal::is_supported_vector<T_y>::value)
        && (fused_normal::is_supported_scalar<T_loc>::value
            || fused_normal::is_supported_vector<T_loc>::value)
        && (fused_normal::is_supported_scalar<T_scale>::value
            || fused_normal::is_supported_vector<T_scale>::value)
        && (fused_normal::is_supported_vector<T_y>::value
            || fused_normal::is_supported_vector<T_loc>::value
            || fused_normal::is_supported_vector<T_scale>::value),
    stan::return_type_t<T_y, T_loc, T_scale>>
nico_lognormal_lpdf(const T_y& y,
                    const T_loc& mu,
                    const T_scale& sigma,
                    std::ostream* pstream__) {

    static constexpr const char* function = "lognormal_lpdf";
    using return_t = stan::return_type_t<T_y, T_loc, T_scale>;
    using y_ref_t = stan::ref_type_if_not_constant_t<T_y>;
    using mu_ref_t = stan::ref_type_if_not_constant_t<T_loc>;
    using sigma_ref_t = stan::ref_type_if_not_constant_t<T_scale>;

    stan::math::check_consistent_sizes(function, "Random variable", y, "Location parameter", mu,
                                       "Scale parameter", sigma);
    y_ref_t y_ref = y;
    mu_ref_t mu_ref = mu;
    sigma_ref_t sigma_ref = sigma;
    decltype(auto) y_value = stan::math::to_ref(stan::math::as_value_column_array_or_scalar(y_ref));
    decltype(auto) mu_value = stan::math::to_ref(stan::math::as_value_column_array_or_scalar(mu_ref));
    decltype(auto) sigma_value = stan::math::to_ref(stan::math::as_value_column_array_or_scalar(sigma_ref));

    stan::math::check_nonnegative(function, "Random variable", y_value);
    stan::math::check_finite(function, "Location parameter", mu_value);
    stan::math::check_positive_finite(function, "Scale parameter", sigma_value);

    if (stan::math::size_zero(y, mu, sigma)
        || !stan::math::include_summand<propto__, T_y, T_loc, T_scale>::value) {
        return return_t(0.0);
    }
    if (stan::math::sum(stan::math::promote_scalar<int>(y_value == 0))) {
        auto ops_partials = stan::math::make_partials_propagator(y_ref, mu_ref, sigma_ref);
        return ops_partials.build(stan::math::LOG_ZERO);
    }

    const auto log_y = [&]() {

        if constexpr (fused_normal::is_supported_vector<T_y>::value) {
            return nico_log(stan::math::to_vector(y_ref), pstream__);
        } else {
            return nico_log(y_ref, pstream__);
        }

    }();
    const auto n = stan::math::max_size(y, mu, sigma);
    return_t lognormal_value = nico_normal_lpdf<propto__>(log_y, mu_ref, sigma_ref, pstream__);

    if constexpr (propto__ && !stan::is_constant_all<T_y, T_loc, T_scale>::value) {
        lognormal_value += n * stan::math::NEG_LOG_SQRT_TWO_PI;
    }
    if constexpr (stan::math::include_summand<propto__, T_y>::value) {
        lognormal_value -= stan::math::sum(log_y) * n / stan::math::size(y);
    }

    return lognormal_value;

}

}  // namespace nico::math

using nico::math::nico_lognormal_lpdf;






















