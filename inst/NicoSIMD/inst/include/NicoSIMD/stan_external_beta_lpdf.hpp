#pragma once

#include "stan_external_scalar_functions.hpp"
#include "stan_external_lgamma.hpp"

namespace nico::math {

template <bool propto__, typename T_y>
inline stan::return_type_t<T_y>
nico_beta_lpdf(const T_y& y,
               const double alpha,
               const double beta,
               std::ostream* pstream__) {

    static_assert(std::is_arithmetic<T_y>::value || stan::is_var<T_y>::value,
                  "The outcome must be an arithmetic scalar or a reverse-mode scalar.");
    static constexpr const char* function = "beta_lpdf";
    const double y_value = stan::math::value_of(y);
    stan::math::check_positive_finite(function, "First shape parameter", alpha);
    stan::math::check_positive_finite(function, "Second shape parameter", beta);
    stan::math::check_bounded(function, "Random variable", y_value, 0.0, 1.0);

    if constexpr (!stan::math::include_summand<propto__, T_y>::value) {
        return 0.0;
    }

    auto ops_partials = stan::math::make_partials_propagator(y);
    const double log_y = nico_log(y_value, pstream__);
    const double log1m_y = nico_log1p_1_double_precision(-y_value);
    double log_prob = 0.0;
    if constexpr (!propto__) {
        log_prob -= nico_lgamma(alpha, pstream__);
        log_prob -= nico_lgamma(beta, pstream__);
    }
    log_prob += (alpha - 1.0) * log_y;
    log_prob += (beta - 1.0) * log1m_y;
    if constexpr (!propto__) {
        log_prob += nico_lgamma(alpha + beta, pstream__);
    }
    if constexpr (!stan::is_constant_all<T_y>::value) {
        stan::math::partials<0>(ops_partials)[0]
            = (alpha - 1.0) / y_value + (beta - 1.0) / (y_value - 1.0);
    }
    return ops_partials.build(log_prob);

}

}  // namespace nico::math

using nico::math::nico_beta_lpdf;

namespace nico::math {

template <bool propto__, typename T_y, typename T_alpha, typename T_beta,
          stan::require_all_eigen_col_vector_t<T_y, T_alpha, T_beta>* = nullptr>
inline stan::return_type_t<T_y, T_alpha, T_beta>
nico_beta_lpdf(const T_y& y,
               const T_alpha& alpha,
               const T_beta& beta,
               std::ostream* pstream__) {

    using stan::math::include_summand;
    static constexpr const char* function = "beta_lpdf";
    const auto& y_ref = stan::math::to_ref(y);
    const auto& alpha_ref = stan::math::to_ref(alpha);
    const auto& beta_ref = stan::math::to_ref(beta);
    stan::math::check_consistent_sizes(function, "Random variable", y_ref, "First shape parameter", alpha_ref,
                                       "Second shape parameter", beta_ref);
    const int N = static_cast<int>(y_ref.size());
    const int N_8 = ((N + 7) / 8) * 8;
    const int N3_8 = ((3 * N + 7) / 8) * 8;
    constexpr bool lgamma_or_digamma_needed = include_summand<propto__, T_alpha>::value
                                              || include_summand<propto__, T_beta>::value
                                              || include_summand<propto__, T_alpha, T_beta>::value
                                              || !stan::is_constant_all<T_alpha, T_beta>::value;
    double* scratch = nico_scratch_of_doubles(3 * static_cast<std::size_t>(N_8)
                                              + (lgamma_or_digamma_needed ? 3 : 1)
                                                * static_cast<std::size_t>(N3_8));
    Eigen::Map<Eigen::VectorXd> y_values(scratch, N);
    Eigen::Map<Eigen::VectorXd> log_y(scratch + N_8, N);
    Eigen::Map<Eigen::VectorXd> log1m_y(scratch + 2 * N_8, N);
    Eigen::Map<Eigen::VectorXd> shapes_and_sum(scratch + 3 * N_8, 3 * N);
    Eigen::Map<Eigen::VectorXd> lgamma_of_shapes(lgamma_or_digamma_needed ? scratch + 3 * N_8 + N3_8 : nullptr,
                                                 lgamma_or_digamma_needed ? 3 * N : 0);
    Eigen::Map<Eigen::VectorXd> digamma_of_shapes(lgamma_or_digamma_needed ? scratch + 3 * N_8 + 2 * N3_8
                                                                           : nullptr,
                                                  lgamma_or_digamma_needed ? 3 * N : 0);
    auto alpha_values = shapes_and_sum.segment(0, N);
    auto beta_values = shapes_and_sum.segment(N, N);
    y_values = stan::math::value_of(y_ref);
    alpha_values = stan::math::value_of(alpha_ref);
    beta_values = stan::math::value_of(beta_ref);
    stan::math::check_positive_finite(function, "First shape parameter", alpha_values);
    stan::math::check_positive_finite(function, "Second shape parameter", beta_values);
    stan::math::check_bounded(function, "Random variable", y_values, 0.0, 1.0);
    if (N == 0 || !include_summand<propto__, T_y, T_alpha, T_beta>::value) {
        return 0.0;
    }

    k_log_into(y_values.data(), N, log_y.data());
    k_log1m_into(y_values.data(), N, log1m_y.data());
    constexpr bool lgamma_needed = include_summand<propto__, T_alpha>::value
                                   || include_summand<propto__, T_beta>::value
                                   || include_summand<propto__, T_alpha, T_beta>::value;
    constexpr bool digamma_needed = !stan::is_constant_all<T_alpha, T_beta>::value;
    if constexpr (lgamma_needed || digamma_needed) {
        shapes_and_sum.segment(2 * N, N) = alpha_values + beta_values;
        k_lgamma_and_digamma<digamma_needed>(shapes_and_sum.data(), 3 * N, lgamma_of_shapes.data(),
                                             digamma_of_shapes.data());
    }
    double log_prob = 0.0;
    if constexpr (include_summand<propto__, T_alpha>::value) {
        log_prob -= lgamma_of_shapes.segment(0, N).sum();
    }
    if constexpr (include_summand<propto__, T_beta>::value) {
        log_prob -= lgamma_of_shapes.segment(N, N).sum();
    }
    if constexpr (include_summand<propto__, T_y, T_alpha>::value) {
        log_prob += ((alpha_values.array() - 1.0) * log_y.array()).sum();
    }
    if constexpr (include_summand<propto__, T_y, T_beta>::value) {
        log_prob += ((beta_values.array() - 1.0) * log1m_y.array()).sum();
    }
    if constexpr (include_summand<propto__, T_alpha, T_beta>::value) {
        log_prob += lgamma_of_shapes.segment(2 * N, N).sum();
    }

    auto ops_partials = stan::math::make_partials_propagator(y_ref, alpha_ref, beta_ref);
    if constexpr (!stan::is_constant_all<T_y>::value) {
        stan::math::partials<0>(ops_partials) = (alpha_values.array() - 1.0) / y_values.array()
                                                + (beta_values.array() - 1.0) / (y_values.array() - 1.0);
    }
    if constexpr (!stan::is_constant_all<T_alpha>::value) {
        stan::math::partials<1>(ops_partials) = log_y.array() + digamma_of_shapes.segment(2 * N, N).array()
                                                - digamma_of_shapes.segment(0, N).array();
    }
    if constexpr (!stan::is_constant_all<T_beta>::value) {
        stan::math::partials<2>(ops_partials) = log1m_y.array() + digamma_of_shapes.segment(2 * N, N).array()
                                                - digamma_of_shapes.segment(N, N).array();
    }
    return ops_partials.build(log_prob);

}

}  // namespace nico::math

using nico::math::nico_beta_lpdf;
