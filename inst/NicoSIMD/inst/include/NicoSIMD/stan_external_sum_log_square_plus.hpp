#pragma once

#include "stan_external_functions.hpp"

// sum_i log(x[i]^2 + offset), without separate autodiff nodes for square/add.
// The existing full double-precision AVX log kernel preserves native Stan log accuracy;
// the installed nico_sum_log uses a different, single-precision-accuracy log polynomial.
// This is a general mathematical reduction; interpolation and model terms stay at the call site.
// Values at zero/nonpositive arguments use the existing log kernel, without clipping or jitter.
namespace nico::math {

template <typename T_x, typename T_offset,
          stan::require_eigen_col_vector_t<T_x>* = nullptr,
          stan::require_stan_scalar_t<T_offset>* = nullptr>
inline stan::return_type_t<T_x, T_offset> nico_sum_log_square_plus(
    const T_x& x, const T_offset& offset, std::ostream* /*pstream__*/ = nullptr) {

  const auto& x_ref = stan::math::to_ref(x);
  const Eigen::Index n = x_ref.size();
  if (n == 0) return 0.0;
  const Eigen::Matrix<double, -1, 1> x_values = stan::math::value_of(x_ref);
  const double offset_value = stan::math::value_of(offset);
  Eigen::Matrix<double, -1, 1> argument = x_values.array().square().matrix();
  argument.array() += offset_value;
  Eigen::Matrix<double, -1, 1> log_argument = argument;
  nico::math::apply_kernel(log_argument.data(), static_cast<int>(n),
      [](const nico::math::simd_t values) {

        return BMVP_KERNEL_NAME(nico_log_1_double_precision)(values);

      });
  const double value = log_argument.sum();
  if constexpr (stan::is_constant_all<T_x, T_offset>::value) {
    return value;
  } else {
    auto ops_partials = stan::math::make_partials_propagator(x_ref, offset);
    if constexpr (!stan::is_constant_all<T_x>::value) {
      stan::math::partials<0>(ops_partials) =
          (argument.array().inverse() * (2.0 * x_values.array())).matrix();
    }
    if constexpr (!stan::is_constant_all<T_offset>::value) {
      stan::math::partials<1>(ops_partials) = argument.cwiseInverse().sum();
    }
    return ops_partials.build(value);
  }

}

}  // namespace nico::math

using nico::math::nico_sum_log_square_plus;
























