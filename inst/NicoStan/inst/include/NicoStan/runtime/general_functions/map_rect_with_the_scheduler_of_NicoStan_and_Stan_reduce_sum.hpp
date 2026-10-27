#ifndef NICOSTAN_MAP_RECT_WITH_THE_SCHEDULER_OF_NICOSTAN_AND_STAN_REDUCE_SUM_HPP
#define NICOSTAN_MAP_RECT_WITH_THE_SCHEDULER_OF_NICOSTAN_AND_STAN_REDUCE_SUM_HPP

#ifdef NICOSTAN_MAP_RECT_WITH_THE_AUTOMATIC_SCALAR_TYPE_OF_THE_MODEL_HPP
#error "nico_map_rect with Stan's reduce_sum and the held nuisance block's map header are not combined."
#endif

#include <stan/math/rev.hpp>
#include "map_rect_with_the_nuisance_block_owned_by_NicoStan.hpp"

namespace stan {
namespace math {

//// nico_map_rect with Stan's reduce_sum: no nuisance block is held, so every job parameter is an ordinary node
//// and NicoStan's map scheduler runs on every call (no fallback to Stan's map_rect).
//// The two ordinary-job calls of map_rect_with_the_automatic_scalar_type_of_the_model.hpp, without its
//// nicostan_nuisance_block == nullptr branches.

template <typename T_model, int call_id, typename F, typename T_shared, typename T_job>
Eigen::Matrix<return_type_t<T_model, T_shared, T_job>, Eigen::Dynamic, 1>
nicostan_map_rect_chunked_auto(
      const T_shared& shared_parameters,
      const std::vector<Eigen::Matrix<T_job, Eigen::Dynamic, 1>>& job_parameters,
      const std::vector<std::vector<double>>& real_data,
      const std::vector<std::vector<int>>& integer_data,
      std::ostream* messages) {

  if (internal::nicostan_nuisance_block != nullptr) {
    throw std::domain_error("nico_map_rect with Stan's reduce_sum: a nuisance block is held.");
  }
  //// The native map scheduler does not require an observation partition.
  const T_model derivative_type(0.0);
  return internal::nicostan_map_rect_chunked_with_partition<F>(shared_parameters, job_parameters,
      real_data, integer_data, derivative_type, std::nullopt, nullptr, nullptr, messages);

}

template <typename T_model, int call_id, typename F, typename T_shared, typename T_job>
Eigen::Matrix<return_type_t<T_model, T_shared, T_job>, Eigen::Dynamic, 1>
nicostan_map_rect_chunked_in_observation_chunks_auto(
      const int chunk_size,
      const std::vector<int>& first_observation,
      const std::vector<int>& last_observation,
      const T_shared& shared_parameters,
      const std::vector<Eigen::Matrix<T_job, Eigen::Dynamic, 1>>& job_parameters,
      const std::vector<std::vector<double>>& real_data,
      const std::vector<std::vector<int>>& integer_data,
      std::ostream* messages) {

  if (internal::nicostan_nuisance_block != nullptr) {
    throw std::domain_error("nico_map_rect with Stan's reduce_sum: a nuisance block is held.");
  }
  const T_model derivative_type(0.0);
  return internal::nicostan_map_rect_chunked_with_partition<F>(shared_parameters, job_parameters,
      real_data, integer_data, derivative_type, chunk_size, &first_observation, &last_observation, messages);

}

}  // namespace math
}  // namespace stan

#endif
