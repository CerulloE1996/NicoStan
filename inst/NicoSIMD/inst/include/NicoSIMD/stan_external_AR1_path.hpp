#pragma once

// Exact stationary non-centred AR(1) path transformation for external Stan declarations.
// The reverse pass uses the analytic path adjoint rather than one reverse-mode operation per recursion.

#include "stan_external_functions.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace nico::math::stationary_ar1 {

template <typename T>
struct has_var_vector
    : std::bool_constant<
          stan::is_vector<std::decay_t<T>>::value
          && stan::is_var<stan::scalar_type_t<std::decay_t<T>>>::value> {};

template <typename T>
inline constexpr bool has_var =
    stan::is_var<std::decay_t<T>>::value || has_var_vector<T>::value;

template <typename Original, typename Values>
inline double value_at(const Values& values, std::size_t index) {

  if constexpr (stan::is_vector<std::decay_t<Original>>::value) {
    return values.coeff(static_cast<Eigen::Index>(index));
  } else {
    return stan::math::value_of(values);
  }

}

template <typename T>
inline stan::math::vari* parent_at(const T& values, std::size_t index) {

  return values.coeff(static_cast<Eigen::Index>(index)).vi_;

}

template <typename T_z, typename ZValues>
inline Eigen::Matrix<double, -1, 1> evaluate_path(const ZValues& z_values, double mu_value,
                                     double phi_value, double sigma_value,
                                     std::size_t n) {

  Eigen::Matrix<double, -1, 1> h(static_cast<Eigen::Index>(n));
  if (n == 0) {
    return h;
  }
  const double stationary_sd = std::sqrt(1.0 - phi_value * phi_value);
  h[0] = mu_value + sigma_value * value_at<T_z>(z_values, 0)
         / stationary_sd;
  for (std::size_t i = 1; i < n; ++i) {
    h[static_cast<Eigen::Index>(i)] =
        mu_value + phi_value
            * (h[static_cast<Eigen::Index>(i - 1)] - mu_value)
        + sigma_value * value_at<T_z>(z_values, i);
  }
  return h;

}

template <typename T_z, typename T_mu, typename T_phi, typename T_sigma,
          typename ZValues>
inline Eigen::Matrix<stan::math::var, -1, 1> reverse_path(
    const T_z& z, const T_mu& mu, const T_phi& phi, const T_sigma& sigma,
    const ZValues& z_values, double mu_value, double phi_value,
    double sigma_value, const Eigen::Matrix<double, -1, 1>& h) {

  using var = stan::math::var;
  using vari = stan::math::vari;
  const std::size_t n = static_cast<std::size_t>(h.size());
  Eigen::Matrix<var, -1, 1> result(static_cast<Eigen::Index>(n));
  if (n == 0) {
    return result;
  }
  auto& memory = stan::math::ChainableStack::instance_->memalloc_;
  vari** z_parent = nullptr;
  if constexpr (has_var<T_z>) {
    z_parent = memory.alloc_array<vari*>(n);
    for (std::size_t i = 0; i < n; ++i) {
      z_parent[i] = parent_at(z, i);
    }
  }
  vari* mu_parent = nullptr;
  vari* phi_parent = nullptr;
  vari* sigma_parent = nullptr;
  if constexpr (has_var<T_mu>) mu_parent = mu.vi_;
  if constexpr (has_var<T_phi>) phi_parent = phi.vi_;
  if constexpr (has_var<T_sigma>) sigma_parent = sigma.vi_;
  vari* result_nodes = memory.alloc_array<vari>(n);
  for (std::size_t i = 0; i < n; ++i) {
    ::new (static_cast<void*>(result_nodes + i))
        vari(h[static_cast<Eigen::Index>(i)], false);
    result.coeffRef(static_cast<Eigen::Index>(i)) = var(result_nodes + i);
  }
  double* z_value_copy = memory.alloc_array<double>(n);
  double* path_adjoint = memory.alloc_array<double>(n);
  for (std::size_t i = 0; i < n; ++i) {
    z_value_copy[i] = value_at<T_z>(z_values, i);
  }
  stan::math::reverse_pass_callback(
      [result_nodes, z_parent, mu_parent, phi_parent, sigma_parent,
       z_value_copy, path_adjoint, mu_value, phi_value, sigma_value, n]() {

        for (std::size_t i = 0; i < n; ++i) {
          path_adjoint[i] = result_nodes[i].adj_;
        }
        for (std::size_t i = n; i-- > 1;) {
          path_adjoint[i - 1] += phi_value * path_adjoint[i];
        }
        const double inv_stationary_sd =
            1.0 / std::sqrt(1.0 - phi_value * phi_value);
        double mu_partial = path_adjoint[0];
        double sigma_partial =
            z_value_copy[0] * inv_stationary_sd * path_adjoint[0];
        double phi_partial =
            sigma_value * z_value_copy[0] * phi_value
            * inv_stationary_sd * inv_stationary_sd * inv_stationary_sd
            * path_adjoint[0];
        if constexpr (has_var<T_z>) {
          z_parent[0]->adj_ += sigma_value * inv_stationary_sd
                               * path_adjoint[0];
        }
        for (std::size_t i = 1; i < n; ++i) {
          const double g = path_adjoint[i];
          if constexpr (has_var<T_z>) z_parent[i]->adj_ += sigma_value * g;
          mu_partial += (1.0 - phi_value) * g;
          sigma_partial += z_value_copy[i] * g;
          phi_partial +=
              (result_nodes[i - 1].val_ - mu_value) * g;
        }
        if constexpr (has_var<T_mu>) mu_parent->adj_ += mu_partial;
        if constexpr (has_var<T_phi>) phi_parent->adj_ += phi_partial;
        if constexpr (has_var<T_sigma>) sigma_parent->adj_ += sigma_partial;

      });
  return result;

}

}

namespace nico::math {

template <typename T_z, typename T_mu, typename T_phi, typename T_sigma,
          stan::require_all_not_nonscalar_prim_or_rev_kernel_expression_t<
              T_z, T_mu, T_phi, T_sigma>* = nullptr>
inline auto nico_stationary_noncentred_AR1_path(
    T_z&& z, T_mu&& mu, T_phi&& phi, T_sigma&& sigma,
    std::ostream* /* pstream__ */ = nullptr) {

  static constexpr const char* function =
      "nico_stationary_noncentred_AR1_path";
  stan::math::check_consistent_sizes(function, "Standard-normal innovations", z);
  decltype(auto) z_ref = std::forward<T_z>(z);
  decltype(auto) mu_ref = std::forward<T_mu>(mu);
  decltype(auto) phi_ref = std::forward<T_phi>(phi);
  decltype(auto) sigma_ref = std::forward<T_sigma>(sigma);
  decltype(auto) z_values = stan::math::to_ref(
      stan::math::as_value_column_array_or_scalar(z_ref));
  const double mu_value = stan::math::value_of(mu_ref);
  const double phi_value = stan::math::value_of(phi_ref);
  const double sigma_value = stan::math::value_of(sigma_ref);
  stan::math::check_not_nan(function, "Standard-normal innovations", z_values);
  stan::math::check_finite(function, "Location parameter", mu_value);
  stan::math::check_finite(function, "AR coefficient", phi_value);
  stan::math::check_finite(function, "Innovation scale", sigma_value);
  stan::math::check_positive(function, "Innovation scale", sigma_value);
  if (!(std::abs(phi_value) < 1.0)) {
    throw std::domain_error(
        "nico_stationary_noncentred_AR1_path: |phi| must be less than 1");
  }
  const std::size_t n = stan::math::size(z_ref);
  const auto h = nico::math::stationary_ar1::evaluate_path<T_z>(
      z_values, mu_value, phi_value, sigma_value, n);
  if constexpr (nico::math::stationary_ar1::has_var<T_z>
                || nico::math::stationary_ar1::has_var<T_mu>
                || nico::math::stationary_ar1::has_var<T_phi>
                || nico::math::stationary_ar1::has_var<T_sigma>) {

    return nico::math::stationary_ar1::reverse_path(
        z_ref, mu_ref, phi_ref, sigma_ref, z_values, mu_value, phi_value,
        sigma_value, h);

  } else {
    return h;
  }

}

}  // namespace nico::math

using nico::math::nico_stationary_noncentred_AR1_path;

namespace nico::math::ar1_recursion {

template <typename T_shocks, typename T_carry, typename T_a>
inline auto path(const T_shocks& shocks, const T_carry& carry, const T_a& a) {

  using stan::math::var;
  using stan::math::vari;
  using nico::math::stationary_ar1::has_var;
  const auto& shocks_ref = stan::math::to_ref(shocks);
  const auto& shock_values = stan::math::to_ref(stan::math::value_of(shocks_ref));
  const double carry_value = stan::math::value_of(carry);
  const double a_value = stan::math::value_of(a);
  const Eigen::Index n = shocks_ref.size();
  Eigen::Matrix<double, -1, 1> values(n);
  double previous = carry_value;
  for (Eigen::Index i = 0; i < n; ++i) {
    values.coeffRef(i) = a_value * previous + shock_values.coeff(i);
    previous = values.coeff(i);
  }
  if constexpr (!has_var<T_shocks> && !has_var<T_carry> && !has_var<T_a>) {
    return values;
  } else {
    Eigen::Matrix<var, Eigen::Dynamic, 1> result(n);
    if (n == 0) return result;
    auto& memory = stan::math::ChainableStack::instance_->memalloc_;
    vari** shock_parents = nullptr;
    if constexpr (has_var<T_shocks>) {
      shock_parents = memory.alloc_array<vari*>(n);
      for (Eigen::Index i = 0; i < n; ++i) {
        shock_parents[i] = shocks_ref.coeff(i).vi_;
      }
    }
    vari* carry_parent = nullptr;
    vari* a_parent = nullptr;
    if constexpr (has_var<T_carry>) carry_parent = carry.vi_;
    if constexpr (has_var<T_a>) a_parent = a.vi_;
    vari* result_nodes = memory.alloc_array<vari>(n);
    for (Eigen::Index i = 0; i < n; ++i) {
      ::new (static_cast<void*>(result_nodes + i)) vari(values.coeff(i), false);
      result.coeffRef(i) = var(result_nodes + i);
    }
    stan::math::reverse_pass_callback(
        [result_nodes, shock_parents, carry_parent, a_parent, carry_value, a_value, n]() {

          double path_adjoint = 0.0;
          double a_partial = 0.0;
          for (Eigen::Index i = n; i-- > 0;) {
            path_adjoint = result_nodes[i].adj_ + a_value * path_adjoint;
            if constexpr (has_var<T_shocks>) shock_parents[i]->adj_ += path_adjoint;
            if constexpr (has_var<T_a>) {
              const double prior_value = i == 0 ? carry_value : result_nodes[i - 1].val_;
              a_partial += prior_value * path_adjoint;
            }
          }
          if constexpr (has_var<T_carry>) carry_parent->adj_ += a_value * path_adjoint;
          if constexpr (has_var<T_a>) a_parent->adj_ += a_partial;

        });
    return result;
  }

}

}  // namespace nico::math::ar1_recursion

// The general affine recursion d[i] = a * d[i - 1] + shocks[i], with d[-1] = carry.
// Unlike the stationary path, a may be any finite real and carry is an independent argument.
// One analytic reverse pass covers the path. This is a path transform, not a model log density.
namespace nico::math {

template <typename T_shocks, typename T_carry, typename T_a,
          stan::require_eigen_col_vector_t<T_shocks>* = nullptr,
          stan::require_all_not_nonscalar_prim_or_rev_kernel_expression_t<
              T_shocks, T_carry, T_a>* = nullptr>
inline auto nico_AR1_recursion(const T_shocks& shocks, const T_carry& carry, const T_a& a,
                               std::ostream* /*pstream__*/ = nullptr) {

  static const char* function = "nico_AR1_recursion";
  stan::math::check_finite(function, "Carry", stan::math::value_of(carry));
  stan::math::check_finite(function, "AR coefficient", stan::math::value_of(a));
  return nico::math::ar1_recursion::path(shocks, carry, a);

}

}  // namespace nico::math

using nico::math::nico_AR1_recursion;























