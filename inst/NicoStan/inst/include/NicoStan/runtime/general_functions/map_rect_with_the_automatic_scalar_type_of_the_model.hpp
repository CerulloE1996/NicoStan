#ifndef NICOSTAN_MAP_RECT_WITH_THE_AUTOMATIC_SCALAR_TYPE_OF_THE_MODEL_HPP
#define NICOSTAN_MAP_RECT_WITH_THE_AUTOMATIC_SCALAR_TYPE_OF_THE_MODEL_HPP

#include <stan/math/rev.hpp>
#include "map_rect_with_the_nuisance_block_owned_by_NicoStan.hpp"

namespace stan {
namespace math {

//// The generated caller supplies its existing scalar type; no parameter value is used as a multiplier.
//// A model evaluated with AD therefore retains nuisance derivatives even with data-only map arguments.

template <typename T_model, int call_id, typename F, typename T_shared, typename T_job>
Eigen::Matrix<return_type_t<T_model, T_shared, T_job>, Eigen::Dynamic, 1>
nicostan_map_rect_chunked_auto(
      const T_shared& shared_parameters,
      const std::vector<Eigen::Matrix<T_job, Eigen::Dynamic, 1>>& job_parameters,
      const std::vector<std::vector<double>>& real_data,
      const std::vector<std::vector<int>>& integer_data,
      std::ostream* messages) {

  using T_result = return_type_t<T_model, T_shared, T_job>;
  if (internal::nicostan_nuisance_block == nullptr) {
    //// With no hidden nuisance parameters, retain the ordinary map implementation and its AD types.
    const auto ordinary_output = map_rect<call_id, F>(shared_parameters, job_parameters,
                                                     real_data, integer_data, messages);
    Eigen::Matrix<T_result, Eigen::Dynamic, 1> output = ordinary_output.template cast<T_result>();
    return output;
  }
  //// The native map scheduler does not require an observation partition.
  //// Keep nuisance reads inside jobs without inventing observation units or a fixed chunk_size.
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
  if (internal::nicostan_nuisance_block == nullptr) {
    return nicostan_map_rect_chunked_auto<T_model, call_id, F>(shared_parameters, job_parameters,
                                                              real_data, integer_data, messages);
  }
  const T_model derivative_type(0.0);
  return internal::nicostan_map_rect_chunked_with_partition<F>(shared_parameters, job_parameters,
      real_data, integer_data, derivative_type, chunk_size, &first_observation, &last_observation, messages);
}

template <typename T_model, int call_id, typename F, typename T_shared, typename T_loader>
Eigen::Matrix<return_type_t<T_model, T_shared>, Eigen::Dynamic, 1>
nicostan_map_rect_chunked_in_observation_chunks_auto(
      const int chunk_size,
      const std::vector<int>& first_observation,
      const std::vector<int>& last_observation,
      const T_shared& shared_parameters,
      const internal::nicostan_map_nuisance_job_parameters<T_loader>& job_parameters,
      const std::vector<std::vector<double>>& real_data,
      const std::vector<std::vector<int>>& integer_data,
      std::ostream* messages) {
  //// A deferred reader also loads configured numeric values outside NicoStan's own gradient call.
  //// An ordinary map cannot receive this data-only recipe as though it contained parameter values.
  const T_model derivative_type(0.0);
  return internal::nicostan_map_rect_chunked_from_job_source<F>(shared_parameters, job_parameters,
      real_data, integer_data, derivative_type, chunk_size, &first_observation, &last_observation, messages);
}

}  // namespace math
}  // namespace stan

#endif






















