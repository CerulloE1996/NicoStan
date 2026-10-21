#pragma once

// Fused normal log-density functions for external Stan declarations.
// The value and partial arrays are evaluated in one pass using the selected AVX2 or AVX-512 lane width.
// Scalar broadcast scales use one exact std::log or std::exp call; vector scales use guarded kernels.
// The log-scale form uses sigma = exp(log_scale) algebraically, with d/d(log_scale) = z^2 - 1.

#include "stan_external_functions.hpp"

#include <stan/math/prim/functor/partials_propagator.hpp>

#include <cmath>
#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

namespace nico::math::normal_detail {

using nico::math::simd_t;

template <typename Original, typename Values>
inline double value_at(const Values& values, std::size_t index) {

  if constexpr (stan::is_vector<std::decay_t<Original>>::value) {
    return values.coeff(static_cast<Eigen::Index>(index));
  } else {
    return stan::math::value_of(values);
  }

}

template <std::size_t Index, typename Original, typename OpsPartials>
inline void assign_partial(OpsPartials& ops_partials,
                           const Eigen::Matrix<double, -1, 1>& partial) {

  if constexpr (stan::is_vector<std::decay_t<Original>>::value) {
    stan::math::partials<Index>(ops_partials) = partial;
  } else {
    stan::math::partials<Index>(ops_partials) = partial.sum();
  }

}

struct workspace_t {
  std::vector<double> y;
  std::vector<double> mu;
  std::vector<double> scale;
  std::vector<double> inverse_scale;
  std::vector<double> z;
  Eigen::Matrix<double, -1, 1> partial_y;
  Eigen::Matrix<double, -1, 1> partial_mu;
  Eigen::Matrix<double, -1, 1> partial_scale;

  void ensure(std::size_t n, bool need_y, bool need_mu, bool need_scale,
              bool need_inverse_scale) {

    y.resize(n);
    mu.resize(n);
    scale.resize(n);
    z.resize(n);
    if (need_inverse_scale) {
      inverse_scale.resize(n);
    } else {
      inverse_scale.resize(0);
    }
    if (need_y) {
      partial_y.resize(static_cast<Eigen::Index>(n));
    } else {
      partial_y.resize(0);
    }
    if (need_mu) {
      partial_mu.resize(static_cast<Eigen::Index>(n));
    } else {
      partial_mu.resize(0);
    }
    if (need_scale) {
      partial_scale.resize(static_cast<Eigen::Index>(n));
    } else {
      partial_scale.resize(0);
    }

  }
};

inline workspace_t& workspace() {

  thread_local workspace_t buffers;
  return buffers;

}

template <typename Original, typename Values>
inline void copy_values(const Values& values, std::size_t n,
                        std::vector<double>& output) {

  for (std::size_t i = 0; i < n; ++i) {
    output[i] = value_at<Original>(values, i);
  }

}

inline simd_t simd_set1(double value) {

#if BMVP_LANES == 8
  return _mm512_set1_pd(value);
#else
  return _mm256_set1_pd(value);
#endif

}

inline simd_t simd_sub(simd_t left, simd_t right) {

#if BMVP_LANES == 8
  return _mm512_sub_pd(left, right);
#else
  return _mm256_sub_pd(left, right);
#endif

}

inline simd_t simd_mul(simd_t left, simd_t right) {

#if BMVP_LANES == 8
  return _mm512_mul_pd(left, right);
#else
  return _mm256_mul_pd(left, right);
#endif

}

inline simd_t simd_div(simd_t left, simd_t right) {

#if BMVP_LANES == 8
  return _mm512_div_pd(left, right);
#else
  return _mm256_div_pd(left, right);
#endif

}

inline simd_t simd_add(simd_t left, simd_t right) {

#if BMVP_LANES == 8
  return _mm512_add_pd(left, right);
#else
  return _mm256_add_pd(left, right);
#endif

}

inline double simd_sum(simd_t values) {

  alignas(64) double lanes[BMVP_LANES];
  nico::math::simd_storeu(lanes, values);
  double result = 0.0;
  for (int lane = 0; lane < BMVP_LANES; ++lane) {
    result += lanes[lane];
  }
  return result;

}

template <typename Original, typename Values>
inline simd_t load_block(const Values& values, std::size_t index,
                         double* strided_buffer) {

  // Eigen maps and ordinary column vectors stay on the direct packet path; only strided views gather.
  if constexpr (stan::is_vector<std::decay_t<Original>>::value) {
    if (values.innerStride() == 1) {
      return nico::math::simd_loadu(values.data() + index);
    }
    for (int lane = 0; lane < BMVP_LANES; ++lane) {
      strided_buffer[lane] = value_at<Original>(
          values, index + static_cast<std::size_t>(lane));
    }
    return nico::math::simd_loadu(strided_buffer);
  } else {
    return simd_set1(stan::math::value_of(values));
  }

}

template <bool need_y, bool need_mu, bool need_scale>
inline void calculate_normal_terms(workspace_t& buffers, std::size_t n) {

  std::size_t i = 0;
  for (; i + BMVP_LANES <= n; i += BMVP_LANES) {
    const simd_t y = nico::math::simd_loadu(buffers.y.data() + i);
    const simd_t mu = nico::math::simd_loadu(buffers.mu.data() + i);
    const simd_t scale = nico::math::simd_loadu(buffers.scale.data() + i);
    const simd_t inverse_scale = simd_div(simd_set1(1.0), scale);
    const simd_t z = simd_mul(simd_sub(y, mu), inverse_scale);
    nico::math::simd_storeu(buffers.z.data() + i, z);
    if constexpr (need_y || need_mu || need_scale) {
      const simd_t z_squared = simd_mul(z, z);
      const simd_t inverse_scale_z = simd_mul(inverse_scale, z);
      if constexpr (need_y) {
        nico::math::simd_storeu(
            buffers.partial_y.data() + i,
            simd_sub(simd_set1(0.0), inverse_scale_z));
      }
      if constexpr (need_mu) {
        nico::math::simd_storeu(buffers.partial_mu.data() + i,
                                   inverse_scale_z);
      }
      if constexpr (need_scale) {
        const simd_t scale_partial =
            simd_mul(inverse_scale, simd_sub(z_squared, simd_set1(1.0)));
        nico::math::simd_storeu(buffers.partial_scale.data() + i,
                                   scale_partial);
      }
    }
  }

  for (; i < n; ++i) {
    const double inverse_scale = 1.0 / buffers.scale[i];
    buffers.z[i] = (buffers.y[i] - buffers.mu[i]) * inverse_scale;
    if constexpr (need_y || need_mu || need_scale) {
      const double z_squared = buffers.z[i] * buffers.z[i];
      const double inverse_scale_z = inverse_scale * buffers.z[i];
      if constexpr (need_y) {
        buffers.partial_y[i] = -inverse_scale_z;
      }
      if constexpr (need_mu) {
        buffers.partial_mu[i] = inverse_scale_z;
      }
      if constexpr (need_scale) {
        buffers.partial_scale[i] = inverse_scale * (z_squared - 1.0);
      }
    }
  }

}

// The log-scale path uses the guarded SIMD exp kernel for vector scales and scalar exp for a
// broadcast scale. The remaining arithmetic is evaluated using the selected SIMD lane width.
template <bool need_y, bool need_mu, bool need_scale, bool scale_vector>
inline void calculate_log_scale_terms(workspace_t& buffers, std::size_t n) {

  if constexpr (scale_vector) {
    for (std::size_t i = 0; i < n; ++i) {
      buffers.inverse_scale[i] = -buffers.scale[i];
    }
    nico::math::k_exp(buffers.inverse_scale.data(),
                         static_cast<int>(n));
  } else {
    const double inverse_scale = std::exp(-buffers.scale[0]);
    for (std::size_t i = 0; i < n; ++i) {
      buffers.inverse_scale[i] = inverse_scale;
    }
  }

  std::size_t i = 0;
  for (; i + BMVP_LANES <= n; i += BMVP_LANES) {
    const simd_t y = nico::math::simd_loadu(buffers.y.data() + i);
    const simd_t mu = nico::math::simd_loadu(buffers.mu.data() + i);
    const simd_t inverse_scale =
        nico::math::simd_loadu(buffers.inverse_scale.data() + i);
    const simd_t z = simd_mul(simd_sub(y, mu), inverse_scale);
    nico::math::simd_storeu(buffers.z.data() + i, z);
    if constexpr (need_y || need_mu || need_scale) {
      const simd_t inverse_scale_z = simd_mul(inverse_scale, z);
      if constexpr (need_y) {
        nico::math::simd_storeu(
            buffers.partial_y.data() + i,
            simd_sub(simd_set1(0.0), inverse_scale_z));
      }
      if constexpr (need_mu) {
        nico::math::simd_storeu(buffers.partial_mu.data() + i,
                                   inverse_scale_z);
      }
      if constexpr (need_scale) {
        const simd_t z_squared = simd_mul(z, z);
        nico::math::simd_storeu(
            buffers.partial_scale.data() + i,
            simd_sub(z_squared, simd_set1(1.0)));
      }
    }
  }

  for (; i < n; ++i) {
    const double inverse_scale = buffers.inverse_scale[i];
    buffers.z[i] = (buffers.y[i] - buffers.mu[i]) * inverse_scale;
    if constexpr (need_y || need_mu || need_scale) {
      const double inverse_scale_z = inverse_scale * buffers.z[i];
      if constexpr (need_y) {
        buffers.partial_y[i] = -inverse_scale_z;
      }
      if constexpr (need_mu) {
        buffers.partial_mu[i] = inverse_scale_z;
      }
      if constexpr (need_scale) {
        const double z_squared = buffers.z[i] * buffers.z[i];
        buffers.partial_scale[i] = z_squared - 1.0;
      }
    }
  }

}

inline double sum_squared_z(const workspace_t& buffers, std::size_t n) {

  double result = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    result += buffers.z[i] * buffers.z[i];
  }
  return result;

}

inline double sum_values(const std::vector<double>& values, std::size_t n) {

  double result = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    result += values[i];
  }
  return result;

}

template <bool scale_vector>
inline double sum_log_scale(workspace_t& buffers, std::size_t n) {

  if constexpr (scale_vector) {
    for (std::size_t i = 0; i < n; ++i) {
      buffers.inverse_scale[i] = buffers.scale[i];
    }
    nico::math::k_log(buffers.inverse_scale.data(),
                         static_cast<int>(n));
    return sum_values(buffers.inverse_scale, n);
  } else {
    return std::log(buffers.scale[0]) * static_cast<double>(n);
  }

}

template <bool propto, bool log_scale, typename T_y, typename T_loc,
          typename T_scale, typename YValues, typename MuValues,
          typename ScaleValues>
inline double primitive_lpdf(const YValues& y_values, const MuValues& mu_values,
                             const ScaleValues& scale_values, std::size_t n) {

  constexpr bool scale_vector = stan::is_vector<std::decay_t<T_scale>>::value;
  alignas(64) double y_buffer[BMVP_LANES];
  alignas(64) double mu_buffer[BMVP_LANES];
  alignas(64) double scale_buffer[BMVP_LANES];
  simd_t sum_z_squared = simd_set1(0.0);
  simd_t sum_log_scale_vector = simd_set1(0.0);
  double sum_z_squared_tail = 0.0;
  double sum_log_scale_tail = 0.0;
  const double scalar_scale =
      scale_vector ? 1.0 : value_at<T_scale>(scale_values, 0);
  const double scalar_log_scale =
      scale_vector || log_scale ? 0.0 : std::log(scalar_scale);
  const std::size_t n_full = n - n % BMVP_LANES;

  for (std::size_t i = 0; i < n_full; i += BMVP_LANES) {
    const simd_t y = load_block<T_y>(y_values, i, y_buffer);
    const simd_t mu = load_block<T_loc>(mu_values, i, mu_buffer);
    simd_t inverse_scale;
    if constexpr (log_scale) {
      if constexpr (scale_vector) {
        const simd_t log_scale_values = load_block<T_scale>(
            scale_values, i, scale_buffer);
        sum_log_scale_vector = simd_add(sum_log_scale_vector,
                                        log_scale_values);
        inverse_scale = BMVP_KERNEL_NAME(nico_exp_1)(
            simd_sub(simd_set1(0.0), log_scale_values));
      } else {
        inverse_scale = simd_set1(std::exp(-scalar_scale));
      }
    } else if constexpr (scale_vector) {
      const simd_t scale = load_block<T_scale>(
          scale_values, i, scale_buffer);
      inverse_scale = simd_div(simd_set1(1.0), scale);
      sum_log_scale_vector = simd_add(
          sum_log_scale_vector, BMVP_KERNEL_NAME(nico_log_1)(scale));
    } else {
      inverse_scale = simd_set1(1.0 / scalar_scale);
    }
    const simd_t z = simd_mul(simd_sub(y, mu), inverse_scale);
    sum_z_squared = simd_add(sum_z_squared, simd_mul(z, z));
  }

  for (std::size_t i = n_full; i < n; ++i) {
    const double y = value_at<T_y>(y_values, i);
    const double mu = value_at<T_loc>(mu_values, i);
    double inverse_scale;
    if constexpr (log_scale) {
      if constexpr (scale_vector) {
        const double log_scale_value = value_at<T_scale>(scale_values, i);
        sum_log_scale_tail += log_scale_value;
        inverse_scale = std::exp(-log_scale_value);
      } else {
        inverse_scale = std::exp(-scalar_scale);
      }
    } else if constexpr (scale_vector) {
      const double scale = value_at<T_scale>(scale_values, i);
      sum_log_scale_tail += std::log(scale);
      inverse_scale = 1.0 / scale;
    } else {
      inverse_scale = 1.0 / scalar_scale;
    }
    const double z = (y - mu) * inverse_scale;
    sum_z_squared_tail += z * z;
  }

  const double sum_z_squared_total =
      simd_sum(sum_z_squared) + sum_z_squared_tail;
  const double sum_log_scale_total =
      simd_sum(sum_log_scale_vector) + sum_log_scale_tail;

  double logp = -0.5 * sum_z_squared_total;
  if (stan::math::include_summand<propto>::value) {
    logp += ::NEG_HALF_LOG_2PI * static_cast<double>(n);
  }
  if (stan::math::include_summand<propto, T_scale>::value) {
    if constexpr (log_scale || scale_vector) {
      if constexpr (log_scale && !scale_vector) {
        logp -= scalar_scale * static_cast<double>(n);
      } else {
        logp -= sum_log_scale_total;
      }
    } else {
      logp -= scalar_log_scale * static_cast<double>(n);
    }
  }
  return logp;

}

template <std::size_t Index, bool vector_operand, typename OpsPartials>
inline void store_partial_block(OpsPartials& ops_partials, std::size_t index,
                                simd_t partial, double& scalar_sum) {

  if constexpr (vector_operand) {
    nico::math::simd_storeu(
        stan::math::partials<Index>(ops_partials).data() + index, partial);
  } else {
    scalar_sum += simd_sum(partial);
  }

}

template <std::size_t Index, bool vector_operand, typename OpsPartials>
inline void store_partial_scalar(OpsPartials& ops_partials, std::size_t index,
                                 double partial, double& scalar_sum) {

  if constexpr (vector_operand) {
    stan::math::partials<Index>(ops_partials).coeffRef(
        static_cast<Eigen::Index>(index)) = partial;
  } else {
    scalar_sum += partial;
  }

}

template <typename T, bool is_vector = stan::is_vector<std::decay_t<T>>::value>
struct custom_reverse_operand_supported;

template <typename T>
struct custom_reverse_operand_supported<T, false>
    : std::bool_constant<stan::is_var<std::decay_t<T>>::value
                         || std::is_arithmetic<std::decay_t<T>>::value> {};

template <typename T>
struct custom_reverse_operand_supported<T, true>
    : std::bool_constant<
          stan::is_var<stan::scalar_type_t<std::decay_t<T>>>::value
          || std::is_arithmetic<stan::scalar_type_t<std::decay_t<T>>>::value> {};

struct custom_reverse_storage {
  using vari = stan::math::vari;
  vari** y_parent = nullptr;
  vari** mu_parent = nullptr;
  vari** scale_parent = nullptr;
  double* y_partial = nullptr;
  double* mu_partial = nullptr;
  double* scale_partial = nullptr;
  double y_scalar_partial = 0.0;
  double mu_scalar_partial = 0.0;
  double scale_scalar_partial = 0.0;
  std::size_t y_size = 0;
  std::size_t mu_size = 0;
  std::size_t scale_size = 0;
};

template <typename T>
inline stan::math::vari* custom_parent_at(const T& values,
                                          std::size_t index) {

  if constexpr (stan::is_vector<std::decay_t<T>>::value) {
    if constexpr (stan::is_eigen<std::decay_t<T>>::value) {
      return values.coeff(static_cast<Eigen::Index>(index)).vi_;
    } else {
      return values[index].vi_;
    }
  } else {
    return values.vi_;
  }

}

template <typename T>
inline void allocate_custom_operand(const T& values, std::size_t n,
                                    stan::math::vari**& parents,
                                    double*& partials) {

  auto& memory = stan::math::ChainableStack::instance_->memalloc_;
  if constexpr (stan::is_vector<std::decay_t<T>>::value) {
    parents = memory.alloc_array<stan::math::vari*>(n);
    partials = memory.alloc_array<double>(n);
    for (std::size_t i = 0; i < n; ++i) {
      parents[i] = custom_parent_at(values, i);
    }
  } else {
    parents = memory.alloc_array<stan::math::vari*>(1);
    partials = nullptr;
    parents[0] = custom_parent_at(values, 0);
  }

}

template <std::size_t Index, bool vector_operand>
inline void store_custom_block(custom_reverse_storage& storage,
                               std::size_t index, simd_t partial) {

  if constexpr (vector_operand) {
    if constexpr (Index == 0) {
      nico::math::simd_storeu(storage.y_partial + index, partial);
    } else if constexpr (Index == 1) {
      nico::math::simd_storeu(storage.mu_partial + index, partial);
    } else {
      nico::math::simd_storeu(storage.scale_partial + index, partial);
    }
  } else if constexpr (Index == 0) {
    storage.y_scalar_partial += simd_sum(partial);
  } else if constexpr (Index == 1) {
    storage.mu_scalar_partial += simd_sum(partial);
  } else {
    storage.scale_scalar_partial += simd_sum(partial);
  }

}

template <std::size_t Index, bool vector_operand>
inline void store_custom_scalar(custom_reverse_storage& storage,
                                std::size_t index, double partial) {

  if constexpr (vector_operand) {
    if constexpr (Index == 0) {
      storage.y_partial[index] = partial;
    } else if constexpr (Index == 1) {
      storage.mu_partial[index] = partial;
    } else {
      storage.scale_partial[index] = partial;
    }
  } else if constexpr (Index == 0) {
    storage.y_scalar_partial += partial;
  } else if constexpr (Index == 1) {
    storage.mu_scalar_partial += partial;
  } else {
    storage.scale_scalar_partial += partial;
  }

}

template <typename T_y, typename T_loc, typename T_scale>
struct custom_partial_sink {
  static constexpr bool need_y = !stan::is_constant_all<T_y>::value;
  static constexpr bool need_mu = !stan::is_constant_all<T_loc>::value;
  static constexpr bool need_scale = !stan::is_constant_all<T_scale>::value;
  static constexpr bool vector_y = stan::is_vector<std::decay_t<T_y>>::value;
  static constexpr bool vector_mu = stan::is_vector<std::decay_t<T_loc>>::value;
  static constexpr bool vector_scale =
      stan::is_vector<std::decay_t<T_scale>>::value;
  custom_reverse_storage storage;

  custom_partial_sink(const T_y& y, const T_loc& mu, const T_scale& scale,
                      std::size_t n) {

    if constexpr (need_y) {
      allocate_custom_operand(y, n, storage.y_parent, storage.y_partial);
      if constexpr (vector_y) {
        storage.y_size = n;
      }
    }
    if constexpr (need_mu) {
      allocate_custom_operand(mu, n, storage.mu_parent, storage.mu_partial);
      if constexpr (vector_mu) {
        storage.mu_size = n;
      }
    }
    if constexpr (need_scale) {
      allocate_custom_operand(scale, n, storage.scale_parent,
                              storage.scale_partial);
      if constexpr (vector_scale) {
        storage.scale_size = n;
      }
    }

  }

  template <std::size_t Index, bool vector_operand>
  inline void store_block(std::size_t index, simd_t partial) {

    store_custom_block<Index, vector_operand>(storage, index, partial);

  }

  template <std::size_t Index, bool vector_operand>
  inline void store_scalar(std::size_t index, double partial) {

    store_custom_scalar<Index, vector_operand>(storage, index, partial);

  }

  inline void finish() {}

  inline stan::math::var build(double logp) {

    stan::math::var result(logp);
    const custom_reverse_storage saved = storage;
    stan::math::reverse_pass_callback([result, saved]() {

      const double result_adjoint = result.adj();
      if constexpr (need_y) {
        if constexpr (vector_y) {
          for (std::size_t i = 0; i < saved.y_size; ++i) {
            saved.y_parent[i]->adj_ += result_adjoint * saved.y_partial[i];
          }
        } else {
          saved.y_parent[0]->adj_ += result_adjoint
                                      * saved.y_scalar_partial;
        }
      }
      if constexpr (need_mu) {
        if constexpr (vector_mu) {
          for (std::size_t i = 0; i < saved.mu_size; ++i) {
            saved.mu_parent[i]->adj_ += result_adjoint * saved.mu_partial[i];
          }
        } else {
          saved.mu_parent[0]->adj_ += result_adjoint
                                       * saved.mu_scalar_partial;
        }
      }
      if constexpr (need_scale) {
        if constexpr (vector_scale) {
          for (std::size_t i = 0; i < saved.scale_size; ++i) {
            saved.scale_parent[i]->adj_ += result_adjoint
                                           * saved.scale_partial[i];
          }
        } else {
          saved.scale_parent[0]->adj_ += result_adjoint
                                         * saved.scale_scalar_partial;
        }
      }

    });
    return result;

  }
};

template <bool propto, bool log_scale, typename T_y, typename T_loc,
          typename T_scale, typename YValues, typename MuValues,
          typename ScaleValues>
inline stan::return_type_t<T_y, T_loc, T_scale> reverse_lpdf_custom(
    const T_y& y, const T_loc& mu, const T_scale& scale,
    const YValues& y_values, const MuValues& mu_values,
    const ScaleValues& scale_values, std::size_t n) {

  constexpr bool need_y = !stan::is_constant_all<T_y>::value;
  constexpr bool need_mu = !stan::is_constant_all<T_loc>::value;
  constexpr bool need_scale = !stan::is_constant_all<T_scale>::value;
  constexpr bool vector_y = stan::is_vector<std::decay_t<T_y>>::value;
  constexpr bool vector_mu = stan::is_vector<std::decay_t<T_loc>>::value;
  constexpr bool vector_scale = stan::is_vector<std::decay_t<T_scale>>::value;
  custom_partial_sink<T_y, T_loc, T_scale> sink(y, mu, scale, n);
  alignas(64) double y_buffer[BMVP_LANES];
  alignas(64) double mu_buffer[BMVP_LANES];
  alignas(64) double scale_buffer[BMVP_LANES];
  simd_t sum_z_squared = simd_set1(0.0);
  simd_t sum_log_scale_vector = simd_set1(0.0);
  double sum_z_squared_tail = 0.0;
  double sum_log_scale_tail = 0.0;
  const double scalar_scale_value =
      vector_scale ? 1.0 : value_at<T_scale>(scale_values, 0);
  const double scalar_log_scale =
      vector_scale || log_scale ? 0.0 : std::log(scalar_scale_value);
  const std::size_t n_full = n - n % BMVP_LANES;

  for (std::size_t i = 0; i < n_full; i += BMVP_LANES) {
    const simd_t y_values_block = load_block<T_y>(y_values, i, y_buffer);
    const simd_t mu_values_block = load_block<T_loc>(mu_values, i, mu_buffer);
    simd_t inverse_scale;
    if constexpr (log_scale) {
      if constexpr (vector_scale) {
        const simd_t log_scale_values = load_block<T_scale>(
            scale_values, i, scale_buffer);
        sum_log_scale_vector = simd_add(sum_log_scale_vector,
                                        log_scale_values);
        inverse_scale = BMVP_KERNEL_NAME(nico_exp_1)(
            simd_sub(simd_set1(0.0), log_scale_values));
      } else {
        inverse_scale = simd_set1(std::exp(-scalar_scale_value));
      }
    } else if constexpr (vector_scale) {
      const simd_t scale_values_block = load_block<T_scale>(
          scale_values, i, scale_buffer);
      inverse_scale = simd_div(simd_set1(1.0), scale_values_block);
      sum_log_scale_vector = simd_add(
          sum_log_scale_vector, BMVP_KERNEL_NAME(nico_log_1)(
              scale_values_block));
    } else {
      inverse_scale = simd_set1(1.0 / scalar_scale_value);
    }
    const simd_t z = simd_mul(
        simd_sub(y_values_block, mu_values_block), inverse_scale);
    const simd_t z_squared = simd_mul(z, z);
    sum_z_squared = simd_add(sum_z_squared, z_squared);
    const simd_t inverse_scale_z = simd_mul(inverse_scale, z);
    if constexpr (need_y) {
      sink.template store_block<0, vector_y>(
          i, simd_sub(simd_set1(0.0), inverse_scale_z));
    }
    if constexpr (need_mu) {
      sink.template store_block<1, vector_mu>(i, inverse_scale_z);
    }
    if constexpr (need_scale) {
      const simd_t scale_partial = log_scale
          ? simd_sub(z_squared, simd_set1(1.0))
          : simd_mul(inverse_scale, simd_sub(z_squared, simd_set1(1.0)));
      sink.template store_block<2, vector_scale>(i, scale_partial);
    }
  }

  for (std::size_t i = n_full; i < n; ++i) {
    const double y_value = value_at<T_y>(y_values, i);
    const double mu_value = value_at<T_loc>(mu_values, i);
    double inverse_scale;
    if constexpr (log_scale) {
      if constexpr (vector_scale) {
        const double log_scale_value = value_at<T_scale>(scale_values, i);
        sum_log_scale_tail += log_scale_value;
        inverse_scale = std::exp(-log_scale_value);
      } else {
        inverse_scale = std::exp(-scalar_scale_value);
      }
    } else if constexpr (vector_scale) {
      const double scale_value = value_at<T_scale>(scale_values, i);
      sum_log_scale_tail += std::log(scale_value);
      inverse_scale = 1.0 / scale_value;
    } else {
      inverse_scale = 1.0 / scalar_scale_value;
    }
    const double z = (y_value - mu_value) * inverse_scale;
    const double z_squared = z * z;
    sum_z_squared_tail += z_squared;
    const double inverse_scale_z = inverse_scale * z;
    if constexpr (need_y) {
      sink.template store_scalar<0, vector_y>(
          i, -inverse_scale_z);
    }
    if constexpr (need_mu) {
      sink.template store_scalar<1, vector_mu>(i, inverse_scale_z);
    }
    if constexpr (need_scale) {
      const double scale_partial = log_scale
          ? z_squared - 1.0
          : inverse_scale * (z_squared - 1.0);
      sink.template store_scalar<2, vector_scale>(i, scale_partial);
    }
  }

  const double sum_z_squared_total =
      simd_sum(sum_z_squared) + sum_z_squared_tail;
  const double sum_log_scale_total =
      simd_sum(sum_log_scale_vector) + sum_log_scale_tail;
  double logp = -0.5 * sum_z_squared_total;
  if (stan::math::include_summand<propto>::value) {
    logp += ::NEG_HALF_LOG_2PI * static_cast<double>(n);
  }
  if (stan::math::include_summand<propto, T_scale>::value) {
    if constexpr (log_scale || vector_scale) {
      if constexpr (log_scale && !vector_scale) {
        logp -= scalar_scale_value * static_cast<double>(n);
      } else {
        logp -= sum_log_scale_total;
      }
    } else {
      logp -= scalar_log_scale * static_cast<double>(n);
    }
  }
  return sink.build(logp);

}

template <bool propto, bool log_scale, typename T_y, typename T_loc,
          typename T_scale, typename YValues, typename MuValues,
          typename ScaleValues>
inline stan::return_type_t<T_y, T_loc, T_scale> reverse_lpdf_legacy(
    const T_y& y, const T_loc& mu, const T_scale& scale,
    const YValues& y_values, const MuValues& mu_values,
    const ScaleValues& scale_values, std::size_t n) {

  constexpr bool need_y = !stan::is_constant_all<T_y>::value;
  constexpr bool need_mu = !stan::is_constant_all<T_loc>::value;
  constexpr bool need_scale = !stan::is_constant_all<T_scale>::value;
  constexpr bool vector_y = stan::is_vector<std::decay_t<T_y>>::value;
  constexpr bool vector_mu = stan::is_vector<std::decay_t<T_loc>>::value;
  constexpr bool vector_scale = stan::is_vector<std::decay_t<T_scale>>::value;
  constexpr bool custom_supported =
      custom_reverse_operand_supported<T_y>::value
      && custom_reverse_operand_supported<T_loc>::value
      && custom_reverse_operand_supported<T_scale>::value;
  if constexpr (custom_supported) {
    return reverse_lpdf_custom<propto, log_scale>(
        y, mu, scale, y_values, mu_values, scale_values, n);
  }
  auto ops_partials = stan::math::make_partials_propagator(y, mu, scale);
  alignas(64) double y_buffer[BMVP_LANES];
  alignas(64) double mu_buffer[BMVP_LANES];
  alignas(64) double scale_buffer[BMVP_LANES];
  simd_t sum_z_squared = simd_set1(0.0);
  simd_t sum_log_scale_vector = simd_set1(0.0);
  double sum_z_squared_tail = 0.0;
  double sum_log_scale_tail = 0.0;
  double scalar_y_partial = 0.0;
  double scalar_mu_partial = 0.0;
  double scalar_scale_partial = 0.0;
  const double scalar_scale_value =
      vector_scale ? 1.0 : value_at<T_scale>(scale_values, 0);
  const double scalar_log_scale =
      vector_scale || log_scale ? 0.0 : std::log(scalar_scale_value);
  const std::size_t n_full = n - n % BMVP_LANES;

  for (std::size_t i = 0; i < n_full; i += BMVP_LANES) {
    const simd_t y_values_block = load_block<T_y>(y_values, i, y_buffer);
    const simd_t mu_values_block = load_block<T_loc>(mu_values, i, mu_buffer);
    simd_t inverse_scale;
    if constexpr (log_scale) {
      if constexpr (vector_scale) {
        const simd_t log_scale_values = load_block<T_scale>(
            scale_values, i, scale_buffer);
        sum_log_scale_vector = simd_add(sum_log_scale_vector,
                                        log_scale_values);
        inverse_scale = BMVP_KERNEL_NAME(nico_exp_1)(
            simd_sub(simd_set1(0.0), log_scale_values));
      } else {
        inverse_scale = simd_set1(std::exp(-scalar_scale_value));
      }
    } else if constexpr (vector_scale) {
      const simd_t scale_values_block = load_block<T_scale>(
          scale_values, i, scale_buffer);
      inverse_scale = simd_div(simd_set1(1.0), scale_values_block);
      sum_log_scale_vector = simd_add(
          sum_log_scale_vector, BMVP_KERNEL_NAME(nico_log_1)(
              scale_values_block));
    } else {
      inverse_scale = simd_set1(1.0 / scalar_scale_value);
    }
    const simd_t z = simd_mul(
        simd_sub(y_values_block, mu_values_block), inverse_scale);
    const simd_t z_squared = simd_mul(z, z);
    sum_z_squared = simd_add(sum_z_squared, z_squared);
    const simd_t inverse_scale_z = simd_mul(inverse_scale, z);
    if constexpr (need_y) {
      store_partial_block<0, vector_y>(ops_partials, i,
                                       simd_sub(simd_set1(0.0),
                                                inverse_scale_z),
                                       scalar_y_partial);
    }
    if constexpr (need_mu) {
      store_partial_block<1, vector_mu>(ops_partials, i, inverse_scale_z,
                                        scalar_mu_partial);
    }
    if constexpr (need_scale) {
      const simd_t scale_partial = log_scale
          ? simd_sub(z_squared, simd_set1(1.0))
          : simd_mul(inverse_scale, simd_sub(z_squared, simd_set1(1.0)));
      store_partial_block<2, vector_scale>(ops_partials, i, scale_partial,
                                           scalar_scale_partial);
    }
  }

  for (std::size_t i = n_full; i < n; ++i) {
    const double y_value = value_at<T_y>(y_values, i);
    const double mu_value = value_at<T_loc>(mu_values, i);
    double inverse_scale;
    if constexpr (log_scale) {
      if constexpr (vector_scale) {
        const double log_scale_value = value_at<T_scale>(scale_values, i);
        sum_log_scale_tail += log_scale_value;
        inverse_scale = std::exp(-log_scale_value);
      } else {
        inverse_scale = std::exp(-scalar_scale_value);
      }
    } else if constexpr (vector_scale) {
      const double scale_value = value_at<T_scale>(scale_values, i);
      sum_log_scale_tail += std::log(scale_value);
      inverse_scale = 1.0 / scale_value;
    } else {
      inverse_scale = 1.0 / scalar_scale_value;
    }
    const double z = (y_value - mu_value) * inverse_scale;
    const double z_squared = z * z;
    sum_z_squared_tail += z_squared;
    const double inverse_scale_z = inverse_scale * z;
    if constexpr (need_y) {
      store_partial_scalar<0, vector_y>(
          ops_partials, i, -inverse_scale_z, scalar_y_partial);
    }
    if constexpr (need_mu) {
      store_partial_scalar<1, vector_mu>(
          ops_partials, i, inverse_scale_z, scalar_mu_partial);
    }
    if constexpr (need_scale) {
      const double scale_partial = log_scale
          ? z_squared - 1.0
          : inverse_scale * (z_squared - 1.0);
      store_partial_scalar<2, vector_scale>(
          ops_partials, i, scale_partial, scalar_scale_partial);
    }
  }

  const double sum_z_squared_total =
      simd_sum(sum_z_squared) + sum_z_squared_tail;
  const double sum_log_scale_total =
      simd_sum(sum_log_scale_vector) + sum_log_scale_tail;
  double logp = -0.5 * sum_z_squared_total;
  if (stan::math::include_summand<propto>::value) {
    logp += ::NEG_HALF_LOG_2PI * static_cast<double>(n);
  }
  if (stan::math::include_summand<propto, T_scale>::value) {
    if constexpr (log_scale || vector_scale) {
      if constexpr (log_scale && !vector_scale) {
        logp -= scalar_scale_value * static_cast<double>(n);
      } else {
        logp -= sum_log_scale_total;
      }
    } else {
      logp -= scalar_log_scale * static_cast<double>(n);
    }
  }
  if constexpr (need_y && !vector_y) {
    stan::math::partials<0>(ops_partials) = scalar_y_partial;
  }
  if constexpr (need_mu && !vector_mu) {
    stan::math::partials<1>(ops_partials) = scalar_mu_partial;
  }
  if constexpr (need_scale && !vector_scale) {
    stan::math::partials<2>(ops_partials) = scalar_scale_partial;
  }
  return ops_partials.build(logp);

}

template <bool propto, bool log_scale, typename T_y, typename T_loc,
          typename T_scale, typename YValues, typename MuValues,
          typename ScaleValues>
inline stan::return_type_t<T_y, T_loc, T_scale> reverse_lpdf(
    const T_y& y, const T_loc& mu, const T_scale& scale,
    const YValues& y_values, const MuValues& mu_values,
    const ScaleValues& scale_values, std::size_t n) {

  constexpr bool custom_supported =
      custom_reverse_operand_supported<T_y>::value
      && custom_reverse_operand_supported<T_loc>::value
      && custom_reverse_operand_supported<T_scale>::value;
  if constexpr (custom_supported) {
    return reverse_lpdf_custom<propto, log_scale>(
        y, mu, scale, y_values, mu_values, scale_values, n);
  } else if constexpr (log_scale) {
    return stan::math::normal_lpdf<propto>(
        y, mu, stan::math::exp(scale));
  } else {
    return stan::math::normal_lpdf<propto>(y, mu, scale);
  }

}

template <bool propto, bool log_scale, typename T_y, typename T_loc,
          typename T_scale>
inline stan::return_type_t<T_y, T_loc, T_scale> normal_lpdf_impl(
    T_y&& y, T_loc&& mu, T_scale&& scale) {

  using T_y_ref = stan::ref_type_if_not_constant_t<T_y>;
  using T_mu_ref = stan::ref_type_if_not_constant_t<T_loc>;
  using T_scale_ref = stan::ref_type_if_not_constant_t<T_scale>;

  static constexpr const char* function = "nico_normal_lpdf";
  stan::math::check_consistent_sizes(function, "Random variable", y,
                                     "Location parameter", mu,
                                     "Scale parameter", scale);
  T_y_ref y_ref = std::forward<T_y>(y);
  T_mu_ref mu_ref = std::forward<T_loc>(mu);
  T_scale_ref scale_ref = std::forward<T_scale>(scale);

  decltype(auto) y_val = stan::math::to_ref(
      stan::math::as_value_column_array_or_scalar(y_ref));
  decltype(auto) mu_val = stan::math::to_ref(
      stan::math::as_value_column_array_or_scalar(mu_ref));
  decltype(auto) scale_val = stan::math::to_ref(
      stan::math::as_value_column_array_or_scalar(scale_ref));

  stan::math::check_not_nan(function, "Random variable", y_val);
  stan::math::check_finite(function, "Location parameter", mu_val);
  if constexpr (log_scale) {
    stan::math::check_finite(function, "Log scale parameter", scale_val);
  } else {
    stan::math::check_positive(function, "Scale parameter", scale_val);
  }

  if (stan::math::size_zero(y_ref, mu_ref, scale_ref)) {
    return 0.0;
  }
  if (!stan::math::include_summand<propto, T_y, T_loc, T_scale>::value) {
    return 0.0;
  }

  const std::size_t n = stan::math::max_size(y_ref, mu_ref, scale_ref);
  constexpr bool need_y = !stan::is_constant_all<T_y>::value;
  constexpr bool need_mu = !stan::is_constant_all<T_loc>::value;
  constexpr bool need_scale = !stan::is_constant_all<T_scale>::value;
  constexpr bool need_partials = need_y || need_mu || need_scale;

  if constexpr (!need_partials) {
    return primitive_lpdf<propto, log_scale, T_y, T_loc, T_scale>(
        y_val, mu_val, scale_val, n);
  } else {
    return reverse_lpdf<propto, log_scale, T_y_ref, T_mu_ref, T_scale_ref>(
        y_ref, mu_ref, scale_ref, y_val, mu_val, scale_val, n);
  }

}

}  // namespace nico::math::normal_detail

// (the fused normal with a log scale, used here for its log-scale form; its nico_normal_lpdf is commented
//  out below so that it does not clash with BayesMVP's own nico_normal_lpdf)
// template <bool propto = false, typename T_y, typename T_loc, typename T_scale,
//           stan::require_all_not_nonscalar_prim_or_rev_kernel_expression_t<
//               T_y, T_loc, T_scale>* = nullptr>
// inline stan::return_type_t<T_y, T_loc, T_scale> nico_normal_lpdf(
//     T_y&& y, T_loc&& mu, T_scale&& sigma,
//     std::ostream* /* pstream__ */ = nullptr) {
//   return nico::math::normal_detail::normal_lpdf_impl<propto, false>(
//       std::forward<T_y>(y), std::forward<T_loc>(mu),
//       std::forward<T_scale>(sigma));
// }

namespace nico::math {

template <bool propto = false, typename T_y, typename T_loc, typename T_log_scale,
          stan::require_all_not_nonscalar_prim_or_rev_kernel_expression_t<
              T_y, T_loc, T_log_scale>* = nullptr>
inline stan::return_type_t<T_y, T_loc, T_log_scale>
nico_normal_log_scale_lpdf(T_y&& y, T_loc&& mu, T_log_scale&& log_scale,
                           std::ostream* /* pstream__ */ = nullptr) {

  return nico::math::normal_detail::normal_lpdf_impl<propto, true>(
      std::forward<T_y>(y), std::forward<T_loc>(mu),
      std::forward<T_log_scale>(log_scale));

}

}  // namespace nico::math

using nico::math::nico_normal_log_scale_lpdf;






















