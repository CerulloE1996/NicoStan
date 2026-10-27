#pragma once



#include "stan_external_functions.hpp"

#include <stan/math/prim/err.hpp>
#include <stan/math/prim/fun/constants.hpp>
#include <stan/math/prim/fun/lgamma.hpp>
#include <stan/math/prim/functor/partials_propagator.hpp>
#include <stan/math/prim/meta.hpp>

#include <cmath>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace nico::math::student_t_regression_detail {

using simd_t = nico::math::simd_t;

inline simd_t simd_one() {

#if BMVP_LANES == 8
  return _mm512_set1_pd(1.0);
#else
  return _mm256_set1_pd(1.0);
#endif

}

inline simd_t simd_set1(const double value) {

#if BMVP_LANES == 8
  return _mm512_set1_pd(value);
#else
  return _mm256_set1_pd(value);
#endif

}

inline simd_t simd_add(const simd_t x, const simd_t y) {

#if BMVP_LANES == 8
  return _mm512_add_pd(x, y);
#else
  return _mm256_add_pd(x, y);
#endif

}

inline simd_t simd_subtract(const simd_t x, const simd_t y) {

#if BMVP_LANES == 8
  return _mm512_sub_pd(x, y);
#else
  return _mm256_sub_pd(x, y);
#endif

}

inline simd_t simd_multiply(const simd_t x, const simd_t y) {

#if BMVP_LANES == 8
  return _mm512_mul_pd(x, y);
#else
  return _mm256_mul_pd(x, y);
#endif

}

inline simd_t simd_divide(const simd_t x, const simd_t y) {

#if BMVP_LANES == 8
  return _mm512_div_pd(x, y);
#else
  return _mm256_div_pd(x, y);
#endif

}

inline simd_t simd_negate(const simd_t x) {

  return simd_subtract(simd_set1(0.0), x);

}

inline simd_t nico_log1p_simd(const simd_t x) {

  return BMVP_KERNEL_NAME(nico_log1p_1_double_precision)(x);

}

inline simd_t nico_log_simd(const simd_t x) {

  return BMVP_KERNEL_NAME(nico_log_1_double_precision)(x);

}

inline double simd_sum(const simd_t x) {

  alignas(64) double lanes[BMVP_LANES];
  nico::math::simd_storeu(lanes, x);
  double result = 0.0;
  for (int lane = 0; lane < BMVP_LANES; ++lane) {
    result += lanes[lane];
  }
  return result;

}

inline double stable_standardised_residual(const double y, const double mu,
                                           const double sigma) {

  const long double difference = static_cast<long double>(y)
                                - static_cast<long double>(mu);
  const long double scaled = difference / static_cast<long double>(sigma);
  const long double max_double
      = static_cast<long double>(std::numeric_limits<double>::max());
  if (scaled > max_double) {
    return std::numeric_limits<double>::infinity();
  }
  if (scaled < -max_double) {
    return -std::numeric_limits<double>::infinity();
  }
  return static_cast<double>(scaled);

}

inline double direct_standardised_residual(const double y, const double mu,
                                           const double sigma) {

  const double difference = y - mu;
  const double residual = difference / sigma;
  if (std::isfinite(difference) && std::isfinite(residual)) {
    return residual;
  }
  return stable_standardised_residual(y, mu, sigma);

}

inline double stable_log1p_from_log_q(const double log_q) {

  if (std::isinf(log_q) && log_q < 0.0) {
    return 0.0;
  }
  if (log_q > 0.0) {
    return log_q + std::log1p(std::exp(-log_q));
  }
  return std::log1p(std::exp(log_q));

}

inline double stable_fraction_from_log_q(const double log_q) {

  if (std::isinf(log_q) && log_q < 0.0) {
    return 0.0;
  }
  if (log_q > 0.0) {
    return 1.0 / (1.0 + std::exp(-log_q));
  }
  const double exp_log_q = std::exp(log_q);
  return exp_log_q / (1.0 + exp_log_q);

}

inline double stable_ratio_for_tail(const double residual,
                                    const double abs_residual,
                                    const double nu) {

  if (residual == 0.0) {
    return 0.0;
  }
  if (!std::isfinite(abs_residual)) {
    return std::copysign(0.0, residual);
  }
  const double log_nu_over_square
      = std::log(nu) - 2.0 * std::log(abs_residual);
  const double nu_over_square
      = log_nu_over_square < std::log(std::numeric_limits<double>::max())
            ? std::exp(log_nu_over_square)
            : std::numeric_limits<double>::infinity();
  return std::copysign(1.0 / (abs_residual * (1.0 + nu_over_square)), residual);

}

}  // namespace nico::math::student_t_regression_detail


namespace nico::math {

template <bool propto = false, typename T_y, typename T_X, typename T_start,
          typename T_end, typename T_dof, typename T_loc, typename T_beta,
          typename T_scale,
          std::enable_if_t<stan::is_vector<T_beta>::value
                               && stan::is_constant_all<T_y>::value
                               && stan::is_constant_all<T_X>::value
                               && stan::is_constant_all<T_dof>::value,
                           int> = 0>
inline stan::return_type_t<T_loc, T_beta, T_scale>
nico_student_t_regression_lpdf(
    const T_y& y_slice, const T_X& X, const T_start& start, const T_end& end,
    const T_dof& nu, const T_loc& alpha, const T_beta& beta,
    const T_scale& sigma, std::ostream* /*pstream__*/ = nullptr) {

  using T_partials_return = stan::partials_return_t<T_loc, T_beta, T_scale>;
  static constexpr const char* function = "nico_student_t_regression_lpdf";

  if (stan::math::size_zero(y_slice)) {
    return 0.0;
  }

  const int first_row = static_cast<int>(start) - 1;
  const int last_row = static_cast<int>(end) - 1;
  if (first_row < 0 || last_row < first_row
      || last_row >= static_cast<int>(X.rows())) {

    throw std::invalid_argument("nico_student_t_regression_lpdf: invalid slice");

  }

  const size_t n = static_cast<size_t>(last_row - first_row + 1);
  if (y_slice.size() != n || beta.size() != X.cols()) {
    throw std::invalid_argument(
        "nico_student_t_regression_lpdf: inconsistent dimensions");
  }
  if (!stan::math::include_summand<propto, T_y, T_dof, T_loc, T_beta,
                                  T_scale>::value) {

    return 0.0;

  }

  stan::math::check_not_nan(function, "Random variable", y_slice);
  stan::math::check_positive_finite(function, "Degrees of freedom parameter",
                                     nu);
  stan::math::check_positive_finite(function, "Scale parameter", sigma);

  const double nu_value = stan::math::value_of(nu);
  const double alpha_value = stan::math::value_of(alpha);
  const double sigma_value = stan::math::value_of(sigma);
  const size_t K = static_cast<size_t>(beta.size());
  std::vector<double> beta_values(K);
  std::vector<double> beta_partials(K, 0.0);
  for (size_t j = 0; j < K; ++j) {
    beta_values[j] = stan::math::value_of(beta.coeff(j));
  }

  const bool include_constant
      = stan::math::include_summand<propto>::value;
  const bool include_dof_normaliser
      = stan::math::include_summand<propto, T_dof>::value;
  const bool include_scale
      = stan::math::include_summand<propto, T_scale>::value;
  const double half_nu = 0.5 * nu_value;
  const double log_dof_normaliser
      = include_dof_normaliser
            ? std::lgamma(half_nu + 0.5) - std::lgamma(half_nu)
                  - 0.5 * std::log(nu_value)
            : 0.0;
  const double log_sigma = include_scale ? std::log(sigma_value) : 0.0;

  auto ops_partials = stan::math::make_partials_propagator(alpha, beta, sigma);
  const nico::math::student_t_regression_detail::simd_t one = nico::math::student_t_regression_detail::simd_one();
  const nico::math::student_t_regression_detail::simd_t zero = nico::math::student_t_regression_detail::simd_subtract(one, one);
  const nico::math::student_t_regression_detail::simd_t nu_simd = nico::math::student_t_regression_detail::simd_set1(nu_value);
  const nico::math::student_t_regression_detail::simd_t inverse_nu = nico::math::student_t_regression_detail::simd_divide(one, nu_simd);
  const nico::math::student_t_regression_detail::simd_t half_nu_plus_half
      = nico::math::student_t_regression_detail::simd_set1(half_nu + 0.5);
  const nico::math::student_t_regression_detail::simd_t scalar_log_sigma
      = nico::math::student_t_regression_detail::simd_set1(log_sigma);
  nico::math::student_t_regression_detail::simd_t logp_sum = zero;
  nico::math::student_t_regression_detail::simd_t alpha_partial_sum = zero;
  nico::math::student_t_regression_detail::simd_t sigma_partial_sum = zero;
  double scalar_logp_remainder = 0.0;
  double scalar_alpha_remainder = 0.0;
  double scalar_sigma_remainder = 0.0;

  stan::scalar_seq_view<T_y> y_values(y_slice);

  auto process_scalar = [&](const size_t local_index, const int row,
                            const double y_value, const double mu_value,
                            const double sigma_i) {

    const double residual = nico::math::student_t_regression_detail::direct_standardised_residual(
        y_value, mu_value, sigma_i);
    const double squared_residual = residual * residual;
    const double q = squared_residual / nu_value;
    const bool q_is_finite = std::isfinite(q);
    const double log_q = q_is_finite
                             ? 0.0
                             : (std::fabs(residual) == 0.0
                                    ? -std::numeric_limits<double>::infinity()
                                    : 2.0 * std::log(std::fabs(residual))
                                          - std::log(nu_value));
    const double log1p_q = q_is_finite
                               ? std::log1p(q)
                               : nico::math::student_t_regression_detail::stable_log1p_from_log_q(log_q);
    const double fraction = q_is_finite
                                ? q / (1.0 + q)
                                : nico::math::student_t_regression_detail::stable_fraction_from_log_q(log_q);
    scalar_logp_remainder -= (half_nu + 0.5) * log1p_q;
    if (include_scale) {
      scalar_logp_remainder -= log_sigma;
    }

    const double ratio = q_is_finite
                             ? residual * (1.0 / nu_value) / (1.0 + q)
                             : nico::math::student_t_regression_detail::stable_ratio_for_tail(
                                   residual, std::fabs(residual), nu_value);
    const double inverse_sigma = 1.0 / sigma_i;
    const double common = q_is_finite
                              ? (residual / (1.0 + q) + ratio) * inverse_sigma
                              : (nu_value * ratio + ratio) * inverse_sigma;
    if constexpr (!stan::is_constant_all<T_loc>::value) {
      if constexpr (stan::is_vector<T_loc>::value) {
        stan::math::partials<0>(ops_partials)[local_index] = common;
      } else {
        scalar_alpha_remainder += common;
      }
    }
    if constexpr (!stan::is_constant_all<T_beta>::value) {
      for (size_t j = 0; j < K; ++j) {
        beta_partials[j] += common * X.coeff(row, j);
      }
    }
    if constexpr (!stan::is_constant_all<T_scale>::value) {
      const double partial_sigma
          = (nu_value * fraction + fraction - 1.0) * inverse_sigma;
      scalar_sigma_remainder += partial_sigma;
    }
    return true;

  };

  size_t local_index = 0;
  for (; local_index + BMVP_LANES <= n; local_index += BMVP_LANES) {
    alignas(64) double y_lanes[BMVP_LANES];
    alignas(64) double mu_lanes[BMVP_LANES];
    alignas(64) double sigma_lanes[BMVP_LANES];
    alignas(64) double residual_lanes[BMVP_LANES];
    alignas(64) double q_lanes[BMVP_LANES];
    alignas(64) double log1p_lanes[BMVP_LANES];
    for (int lane = 0; lane < BMVP_LANES; ++lane) {
      const size_t index = local_index + static_cast<size_t>(lane);
      const int row = first_row + static_cast<int>(index);
      y_lanes[lane] = static_cast<double>(y_values.val(index));
      double mu = alpha_value;
      for (size_t j = 0; j < K; ++j) {
        mu += X.coeff(row, j) * beta_values[j];
      }
      if (!std::isfinite(mu)) {
        throw std::domain_error(
            "nico_student_t_regression_lpdf: non-finite linear predictor");
      }
      mu_lanes[lane] = mu;
      sigma_lanes[lane] = sigma_value;
    }
    const auto y_simd = nico::math::simd_load(y_lanes);
    const auto mu_simd = nico::math::simd_load(mu_lanes);
    const auto sigma_simd = nico::math::simd_load(sigma_lanes);
    const auto residual = nico::math::student_t_regression_detail::simd_divide(
        nico::math::student_t_regression_detail::simd_subtract(y_simd, mu_simd), sigma_simd);
    const auto q = nico::math::student_t_regression_detail::simd_multiply(
        nico::math::student_t_regression_detail::simd_multiply(residual, residual), inverse_nu);
    const auto log1p_q = nico::math::student_t_regression_detail::nico_log1p_simd(q);
    nico::math::simd_store(residual_lanes, residual);
    nico::math::simd_store(q_lanes, q);
    nico::math::simd_store(log1p_lanes, log1p_q);
    bool ordinary_block = true;
    for (int lane = 0; lane < BMVP_LANES; ++lane) {
      if (!std::isfinite(residual_lanes[lane])
          || !std::isfinite(q_lanes[lane])
          || !std::isfinite(log1p_lanes[lane])) {

        ordinary_block = false;
        break;

      }
    }
    if (!ordinary_block) {
      for (int lane = 0; lane < BMVP_LANES; ++lane) {
        const size_t index = local_index + static_cast<size_t>(lane);
        const int row = first_row + static_cast<int>(index);
        if (!process_scalar(index, row, y_lanes[lane], mu_lanes[lane],
                            sigma_lanes[lane])) {

          return 0.0;

        }
      }
      continue;
    }

    auto block_logp = nico::math::student_t_regression_detail::simd_multiply(
        nico::math::student_t_regression_detail::simd_negate(half_nu_plus_half), log1p_q);
    if (include_scale) {
      block_logp = nico::math::student_t_regression_detail::simd_subtract(block_logp, scalar_log_sigma);
    }
    logp_sum = nico::math::student_t_regression_detail::simd_add(logp_sum, block_logp);

    const auto inverse_denominator
        = nico::math::student_t_regression_detail::simd_divide(one, nico::math::student_t_regression_detail::simd_add(one, q));
    const auto fraction = nico::math::student_t_regression_detail::simd_multiply(q, inverse_denominator);
    const auto inverse_sigma = nico::math::student_t_regression_detail::simd_divide(one, sigma_simd);
    const auto ratio = nico::math::student_t_regression_detail::simd_multiply(
        nico::math::student_t_regression_detail::simd_multiply(residual, inverse_nu), inverse_denominator);
    const auto common = nico::math::student_t_regression_detail::simd_multiply(
        nico::math::student_t_regression_detail::simd_add(nico::math::student_t_regression_detail::simd_multiply(residual, inverse_denominator),
                         ratio),
        inverse_sigma);
    if constexpr (!stan::is_constant_all<T_loc>::value) {
      if constexpr (stan::is_vector<T_loc>::value) {
        nico::math::simd_store(residual_lanes, common);
        for (int lane = 0; lane < BMVP_LANES; ++lane) {
          stan::math::partials<0>(ops_partials)[local_index + lane]
              = residual_lanes[lane];
        }
      } else {
        alpha_partial_sum = nico::math::student_t_regression_detail::simd_add(alpha_partial_sum, common);
      }
    }
    if constexpr (!stan::is_constant_all<T_beta>::value) {
      nico::math::simd_store(residual_lanes, common);
      for (int lane = 0; lane < BMVP_LANES; ++lane) {
        const int row = first_row + static_cast<int>(local_index + lane);
        for (size_t j = 0; j < K; ++j) {
          beta_partials[j] += residual_lanes[lane] * X.coeff(row, j);
        }
      }
    }
    if constexpr (!stan::is_constant_all<T_scale>::value) {
      const auto partial_sigma = nico::math::student_t_regression_detail::simd_multiply(
          nico::math::student_t_regression_detail::simd_subtract(
              nico::math::student_t_regression_detail::simd_add(nico::math::student_t_regression_detail::simd_multiply(nu_simd, fraction),
                               fraction),
              one),
          inverse_sigma);
      sigma_partial_sum = nico::math::student_t_regression_detail::simd_add(sigma_partial_sum, partial_sigma);
    }
  }

  for (; local_index < n; ++local_index) {
    const int row = first_row + static_cast<int>(local_index);
    const double y_value = static_cast<double>(y_values.val(local_index));
    double mu_value = alpha_value;
    for (size_t j = 0; j < K; ++j) {
      mu_value += X.coeff(row, j) * beta_values[j];
    }
    if (!std::isfinite(mu_value)) {
      throw std::domain_error(
          "nico_student_t_regression_lpdf: non-finite linear predictor");
    }
    process_scalar(local_index, row, y_value, mu_value, sigma_value);
  }

  double logp = nico::math::student_t_regression_detail::simd_sum(logp_sum) + scalar_logp_remainder;
  if (include_constant) {
    logp -= stan::math::LOG_SQRT_PI * static_cast<double>(n);
  }
  if (include_dof_normaliser) {
    logp += log_dof_normaliser * static_cast<double>(n);
  }
  if constexpr (!stan::is_constant_all<T_loc>::value
                && !stan::is_vector<T_loc>::value) {

    stan::math::partials<0>(ops_partials)
        = nico::math::student_t_regression_detail::simd_sum(alpha_partial_sum) + scalar_alpha_remainder;

  }
  if constexpr (!stan::is_constant_all<T_beta>::value) {
    stan::math::partials<1>(ops_partials)
        = Eigen::Map<const Eigen::Matrix<double, -1, 1>>(beta_partials.data(), K);
  }
  if constexpr (!stan::is_constant_all<T_scale>::value) {
    stan::math::partials<2>(ops_partials)
        = nico::math::student_t_regression_detail::simd_sum(sigma_partial_sum) + scalar_sigma_remainder;
  }

  return ops_partials.build(static_cast<T_partials_return>(logp));

}

}  // namespace nico::math

using nico::math::nico_student_t_regression_lpdf;





















