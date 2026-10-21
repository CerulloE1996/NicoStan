#ifndef NICOSTAN_MAP_RECT_WITH_THE_NUISANCE_BLOCK_OWNED_BY_NICOSTAN_HPP
#define NICOSTAN_MAP_RECT_WITH_THE_NUISANCE_BLOCK_OWNED_BY_NICOSTAN_HPP

#include <stan/math/rev/core.hpp>
#include <stan/math/rev/fun/value_of.hpp>
#include <stan/math/rev/functor/reduce_sum.hpp>
#include <stan/math/prim/fun/to_ref.hpp>
#include "reuse_of_nuisance_read_nodes_within_one_map_job.hpp"
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/partitioner.h>
#include <tbb/task_arena.h>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace stan {
namespace math {
namespace internal {

//// ---------------------------------------------------------------------------------------------------------
//// Each output keeps numeric derivatives only; every job's AD nodes live in its nested tape scope.
//// Nuisance values are read by the job, not copied into a shared vector before the jobs start.
//// ---------------------------------------------------------------------------------------------------------

struct nicostan_map_rect_job_result {
  Eigen::VectorXd values;
  Eigen::MatrixXd ordinary_derivatives;
  std::vector<std::pair<int, double>> nuisance_derivatives;
  std::vector<std::size_t> nuisance_offsets;
  std::string messages;
};

//// The reducer's numeric derivative hand-off, with one callback for all vector outputs of a job.
//// Each output's adjoint remains separate; aliases add to their original node or nuisance index.
class nicostan_map_rect_derivatives_of_one_job : public vari {
  const Eigen::Index number_of_outputs_;
  const Eigen::Index number_of_ordinary_derivatives_;
  vari** ordinary_nodes_;
  vari** output_nodes_;
  double* ordinary_derivatives_;
  nicostan_nuisance_block_t* nuisance_block_;
  std::size_t* nuisance_offsets_;
  int* nuisance_indices_;
  double* nuisance_derivatives_;

 public:
  nicostan_map_rect_derivatives_of_one_job(const nicostan_map_rect_job_result& result,
                                          vari** ordinary_nodes,
                                          nicostan_nuisance_block_t* nuisance_block)
      : vari(0.0), number_of_outputs_(result.values.size()),
        number_of_ordinary_derivatives_(result.ordinary_derivatives.rows()),
        ordinary_nodes_(ordinary_nodes), output_nodes_(nullptr), ordinary_derivatives_(nullptr),
        nuisance_block_(nuisance_block), nuisance_offsets_(nullptr), nuisance_indices_(nullptr),
        nuisance_derivatives_(nullptr) {
    auto& arena = ChainableStack::instance_->memalloc_;
    output_nodes_ = arena.alloc_array<vari*>(number_of_outputs_);
    for (Eigen::Index element = 0; element < number_of_outputs_; ++element) {
      output_nodes_[element] = new vari(result.values(element), false);
    }
    if (result.ordinary_derivatives.size() > 0) {
      ordinary_derivatives_ = arena.alloc_array<double>(result.ordinary_derivatives.size());
      for (Eigen::Index index = 0; index < result.ordinary_derivatives.size(); ++index) {
        ordinary_derivatives_[index] = result.ordinary_derivatives.data()[index];
      }
    }
    nuisance_offsets_ = arena.alloc_array<std::size_t>(number_of_outputs_ + 1);
    for (Eigen::Index element = 0; element <= number_of_outputs_; ++element) {
      nuisance_offsets_[element] = result.nuisance_offsets[element];
    }
    const std::size_t number_of_nuisance_derivatives = result.nuisance_derivatives.size();
    if (number_of_nuisance_derivatives > 0) {
      nuisance_indices_ = arena.alloc_array<int>(number_of_nuisance_derivatives);
      nuisance_derivatives_ = arena.alloc_array<double>(number_of_nuisance_derivatives);
      for (std::size_t index = 0; index < number_of_nuisance_derivatives; ++index) {
        nuisance_indices_[index] = result.nuisance_derivatives[index].first;
        nuisance_derivatives_[index] = result.nuisance_derivatives[index].second;
      }
    }
  }

  vari* output_node(const Eigen::Index element) const {
    return output_nodes_[element];
  }

  void chain() override {
    for (Eigen::Index element = 0; element < number_of_outputs_; ++element) {
      const double multiplier = output_nodes_[element]->adj_;
      if (multiplier == 0.0) continue;
      const Eigen::Index first_ordinary_derivative = element * number_of_ordinary_derivatives_;
      for (Eigen::Index index = 0; index < number_of_ordinary_derivatives_; ++index) {
        const double derivative = ordinary_derivatives_[first_ordinary_derivative + index];
        ordinary_nodes_[index]->adj_ += multiplier == 1.0 ? derivative : multiplier * derivative;
      }
      for (std::size_t index = nuisance_offsets_[element];
           index < nuisance_offsets_[element + 1]; ++index) {
        const double derivative = multiplier == 1.0 ? nuisance_derivatives_[index] :
                                                    multiplier * nuisance_derivatives_[index];
        if (nicostan_nuisance_derivatives_of_the_chunk != nullptr) {
          nicostan_nuisance_derivatives_of_the_chunk->emplace_back(nuisance_indices_[index], derivative);
        } else if (nuisance_block_ != nullptr) {
          nuisance_block_->gradient[nuisance_indices_[index]] += derivative;
        }
      }
    }
  }
};

//// Existing observation blocks stay intact. The range extent and chunk_size count observations.
class nicostan_map_range_of_observation_chunks {
  std::size_t first_job_;
  std::size_t end_job_;
  const std::vector<int>* first_observation_;
  const std::vector<int>* last_observation_;
  std::size_t chunk_size_;

 public:
  nicostan_map_range_of_observation_chunks(const std::size_t first_job,
                                          const std::size_t end_job,
                                          const std::vector<int>& first_observation,
                                          const std::vector<int>& last_observation,
                                          const std::size_t chunk_size)
      : first_job_(first_job), end_job_(end_job), first_observation_(&first_observation),
        last_observation_(&last_observation), chunk_size_(chunk_size) {}

  nicostan_map_range_of_observation_chunks(nicostan_map_range_of_observation_chunks& other,
                                          tbb::split)
      : first_job_(other.first_job_ + (other.end_job_ - other.first_job_) / 2),
        end_job_(other.end_job_), first_observation_(other.first_observation_),
        last_observation_(other.last_observation_), chunk_size_(other.chunk_size_) {
    other.end_job_ = first_job_;
  }

  std::size_t begin() const { return first_job_; }
  std::size_t end() const { return end_job_; }
  std::size_t chunk_size() const { return chunk_size_; }
  bool empty() const { return first_job_ == end_job_; }
  std::size_t size() const {
    if (empty()) return 0;
    return static_cast<std::size_t>((*last_observation_)[end_job_ - 1])
           - static_cast<std::size_t>((*first_observation_)[first_job_]) + 1;
  }
  bool is_divisible() const { return end_job_ - first_job_ > 1 && size() > chunk_size_; }
};

//// Similarly to the static reducer, each range uses this thread's nested tape and shared copies once.
//// Each job has a further nested scope; vector-output reverse sweeps reset the shared leaf adjoints too.

template <typename T_job>
class nicostan_map_ordinary_job_parameters {
  const std::vector<Eigen::Matrix<T_job, Eigen::Dynamic, 1>>& parameters_;

 public:
  using scalar_type = T_job;

  explicit nicostan_map_ordinary_job_parameters(
        const std::vector<Eigen::Matrix<T_job, Eigen::Dynamic, 1>>& parameters)
      : parameters_(parameters) {}

  std::size_t size() const { return parameters_.size(); }
  Eigen::Index parameters_in_job(const std::size_t job) const { return parameters_[job].size(); }

  template <typename T_local>
  decltype(auto) load(const std::size_t job) const {
    if constexpr (std::is_same<T_job, double>::value && std::is_same<T_local, double>::value) {
      return (parameters_[job]);
    } else {
      Eigen::Matrix<T_local, Eigen::Dynamic, 1> values(parameters_[job].size());
      for (Eigen::Index index = 0; index < values.size(); ++index) {
        values(index) = value_of(parameters_[job](index));
      }
      return values;
    }
  }

  vari* node(const std::size_t job, const Eigen::Index index) const {
    if constexpr (is_var<T_job>::value) return parameters_[job](index).vi_;
    else return nullptr;
  }
};

//// A data-only recipe loads hidden parameters after the job's nested tape and nuisance scope begin.
//// No ordinary-parameter Jacobian columns are allocated for these hidden values.
template <typename T_loader>
class nicostan_map_nuisance_job_parameters {
  const std::size_t number_of_jobs_;
  const Eigen::Index parameters_per_job_;
  const T_loader& loader_;

 public:
  using scalar_type = double;

  nicostan_map_nuisance_job_parameters(const std::size_t number_of_jobs,
                                      const Eigen::Index parameters_per_job,
                                      const T_loader& loader)
      : number_of_jobs_(number_of_jobs), parameters_per_job_(parameters_per_job), loader_(loader) {
    if (parameters_per_job < 0) throw std::invalid_argument("nico_map_rect: negative job width.");
  }

  std::size_t size() const { return number_of_jobs_; }
  Eigen::Index parameters_in_job(const std::size_t) const { return parameters_per_job_; }

  template <typename T_local>
  Eigen::Matrix<T_local, Eigen::Dynamic, 1> load(const std::size_t job) const {
    Eigen::Matrix<T_local, Eigen::Dynamic, 1> values = loader_(job, T_local(0.0));
    if (values.size() != parameters_per_job_) {
      throw std::invalid_argument("nico_map_rect: loaded job width differs from its declared width.");
    }
    return values;
  }
};

template <typename F, typename T_shared, typename T_jobs, typename T_type>
Eigen::Matrix<return_type_t<T_shared, typename T_jobs::scalar_type, T_type>, Eigen::Dynamic, 1>
nicostan_map_rect_chunked_from_job_source(
      const T_shared& shared_parameters,
      const T_jobs& job_parameters,
      const std::vector<std::vector<double>>& real_data,
      const std::vector<std::vector<int>>& integer_data,
      const T_type& derivative_type,
      const std::optional<int> chunk_size,
      const std::vector<int>* first_observation,
      const std::vector<int>* last_observation,
      std::ostream* messages) {

  //// T_result is the scalar result type; derivative_type also carries an otherwise hidden nuisance AD type.
  using T_job = typename T_jobs::scalar_type;
  using T_result = return_type_t<T_shared, T_job, T_type>;
  using T_shared_scalar = scalar_type_t<T_shared>;
  using T_local_shared = std::conditional_t<is_var<T_shared_scalar>::value || is_var<T_type>::value,
                                          var, double>;
  using T_local_job = std::conditional_t<is_var<T_job>::value || is_var<T_type>::value, var, double>;
  using T_result_vector = Eigen::Matrix<T_result, Eigen::Dynamic, 1>;
  static_assert(std::is_same<T_result, double>::value || is_var<T_result>::value,
                "map_rect_chunked supports double and reverse-mode scalar inputs.");
  static_cast<void>(derivative_type);

  if (chunk_size.has_value() && *chunk_size < 1) {
    throw std::invalid_argument("map_rect_chunked: chunk_size must be positive.");
  }
  const std::size_t number_of_jobs = job_parameters.size();
  if ((first_observation == nullptr) != (last_observation == nullptr)) {
    throw std::invalid_argument("map_rect_chunked: both observation-boundary arrays are required.");
  }
  if (first_observation != nullptr && !chunk_size.has_value()) {
    throw std::invalid_argument("map_rect_chunked: observation chunks require an explicit chunk_size.");
  }
  if (first_observation != nullptr &&
      (first_observation->size() < number_of_jobs || last_observation->size() < number_of_jobs)) {
    throw std::invalid_argument("map_rect_chunked: observation boundaries do not cover every mapped job.");
  }
  if (real_data.size() != number_of_jobs || integer_data.size() != number_of_jobs) {
    throw std::invalid_argument("map_rect_chunked: job and data counts must agree.");
  }
  if (number_of_jobs == 0) return T_result_vector(0);
  for (std::size_t job = 0; job < number_of_jobs; ++job) {
    if (job_parameters.parameters_in_job(job) != job_parameters.parameters_in_job(0)
        || real_data[job].size() != real_data[0].size()
        || integer_data[job].size() != integer_data[0].size()) {
      throw std::invalid_argument("map_rect_chunked: job inputs must be rectangular.");
    }
    if (first_observation != nullptr) {
      const int first = (*first_observation)[job];
      const int last = (*last_observation)[job];
      if (first < 1 || last < first || static_cast<std::size_t>(last) - first + 1 >
          static_cast<std::size_t>(*chunk_size)) {
        throw std::invalid_argument("map_rect_chunked: a job exceeds its observation chunk_size.");
      }
      if (job > 0 && static_cast<std::size_t>(first) !=
          static_cast<std::size_t>((*last_observation)[job - 1]) + 1) {
        throw std::invalid_argument("map_rect_chunked: mapped observation blocks must be consecutive.");
      }
    }
  }

  nicostan_nuisance_block_t* const nuisance_block = nicostan_nuisance_block;
  const Eigen::VectorXd shared_values = value_of(shared_parameters);
  constexpr bool shared_has_derivatives = is_var<T_shared_scalar>::value;
  constexpr bool jobs_have_derivatives = is_var<T_job>::value;
  const Eigen::Index number_of_shared_derivatives = shared_has_derivatives ? shared_values.size() : 0;
  const Eigen::Index number_of_job_derivatives =
      jobs_have_derivatives ? job_parameters.parameters_in_job(0) : 0;
  const Eigen::Index number_of_ordinary_derivatives =
      number_of_shared_derivatives + number_of_job_derivatives;
  std::vector<nicostan_map_rect_job_result> results(number_of_jobs);

  auto execute_jobs = [&](const std::size_t first, const std::size_t last) {
    if constexpr (!is_var<T_result>::value) {
      for (std::size_t job = first; job < last; ++job) {
        nicostan_scope_of_a_chunk_of_the_nuisance_block chunk_scope(nuisance_block, 0, -1);
        std::optional<std::ostringstream> job_messages;
        if (messages != nullptr) job_messages.emplace();
        auto& result = results[job];
        decltype(auto) local_job = job_parameters.template load<double>(job);
        result.values = F()(shared_values, local_job, real_data[job], integer_data[job],
                            job_messages ? &*job_messages : nullptr);
        if (job_messages) result.messages = job_messages->str();
      }
      return;
    }
    const nested_rev_autodiff range_nested;
    Eigen::Matrix<T_local_shared, Eigen::Dynamic, 1> local_shared(shared_values.size());
    for (Eigen::Index index = 0; index < local_shared.size(); ++index) {
      local_shared(index) = shared_values(index);
    }
    for (std::size_t job = first; job < last; ++job) {
      nested_rev_autodiff nested;
      nicostan_scope_of_a_chunk_of_the_nuisance_block chunk_scope(nuisance_block, 0, -1);
      nicostan_scope_of_reused_nuisance_nodes_for_one_map_job reused_nodes(
          nuisance_block, static_cast<std::size_t>(job_parameters.parameters_in_job(job)));
      auto& result = results[job];
      const std::size_t first_own_node = nicostan_chunk_of_the_nuisance_block.nodes.size();
      const std::size_t first_other_node =
          nicostan_chunk_of_the_nuisance_block.nodes_of_the_values_of_other_terms.size();
      nicostan_scope_of_the_nuisance_derivative_collection collection_scope(result.nuisance_derivatives);
      decltype(auto) local_job = job_parameters.template load<T_local_job>(job);
      std::optional<std::ostringstream> job_messages;
      if (messages != nullptr) job_messages.emplace();
      auto output = to_ref(F()(local_shared, local_job, real_data[job], integer_data[job],
                               job_messages ? &*job_messages : nullptr));
      result.values = value_of(output);
      if (job_messages) result.messages = job_messages->str();
      if constexpr (is_var<T_result>::value) {
        static_assert(is_var<scalar_type_t<decltype(output)>>::value,
                      "map_rect_chunked: job must preserve the reverse-mode scalar type.");
        result.ordinary_derivatives.resize(number_of_ordinary_derivatives, output.size());
        result.nuisance_derivatives.clear();
        const std::size_t direct_read_count =
            nicostan_chunk_of_the_nuisance_block.nodes.size() - first_own_node +
            nicostan_chunk_of_the_nuisance_block.nodes_of_the_values_of_other_terms.size() - first_other_node;
        const std::size_t output_count = static_cast<std::size_t>(output.size());
        if (output_count > 0 && direct_read_count <= result.nuisance_derivatives.max_size() / output_count) {
          result.nuisance_derivatives.reserve(direct_read_count * output_count);
        }
        result.nuisance_offsets.resize(output.size() + 1);
        result.nuisance_offsets[0] = 0;
        for (Eigen::Index element = 0; element < output.size(); ++element) {
          nested.set_zero_all_adjoints();
          if constexpr (is_var<T_local_shared>::value) {
            for (Eigen::Index index = 0; index < local_shared.size(); ++index) {
              local_shared(index).adj() = 0.0;
            }
          }
          output(element).grad();
          if constexpr (shared_has_derivatives) {
            for (Eigen::Index index = 0; index < local_shared.size(); ++index) {
              result.ordinary_derivatives(index, element) = local_shared(index).adj();
            }
          }
          if constexpr (jobs_have_derivatives) {
            for (Eigen::Index index = 0; index < local_job.size(); ++index) {
              result.ordinary_derivatives(number_of_shared_derivatives + index, element) =
                  local_job(index).adj();
            }
          }
          //// Keep the read-node lists for every output's reverse sweep; do not clear them here.
          const auto& own_nodes = nicostan_chunk_of_the_nuisance_block.nodes;
          for (std::size_t index = first_own_node; index < own_nodes.size(); ++index) {
            const auto& value_and_node = own_nodes[index];
            result.nuisance_derivatives.emplace_back(value_and_node.first, value_and_node.second->adj_);
          }
          const auto& other_nodes = nicostan_chunk_of_the_nuisance_block.nodes_of_the_values_of_other_terms;
          for (std::size_t index = first_other_node; index < other_nodes.size(); ++index) {
            const auto& value_and_node = other_nodes[index];
            result.nuisance_derivatives.emplace_back(value_and_node.first, value_and_node.second->adj_);
          }
          result.nuisance_offsets[element + 1] = result.nuisance_derivatives.size();
        }
      }
    }
  };

#ifdef STAN_THREADS
  tbb::this_task_arena::isolate([&] {
    if (first_observation != nullptr) {
      tbb::parallel_for(nicostan_map_range_of_observation_chunks(
                            0, number_of_jobs, *first_observation, *last_observation, *chunk_size),
                        [&](const nicostan_map_range_of_observation_chunks& range) {
                          execute_jobs(range.begin(), range.end());
                        }, tbb::simple_partitioner());
    } else if (chunk_size.has_value()) {
      tbb::parallel_for(tbb::blocked_range<std::size_t>(0, number_of_jobs, *chunk_size),
                      [&](const tbb::blocked_range<std::size_t>& range) {
                        execute_jobs(range.begin(), range.end());
                      }, tbb::simple_partitioner());
    } else {
      //// Without observation boundaries, retain Stan's native map job scheduling.
      //// Job count is not treated as observation N; no observation chunk_size is inferred.
      tbb::parallel_for(tbb::blocked_range<std::size_t>(0, number_of_jobs),
                       [&](const tbb::blocked_range<std::size_t>& range) {
                         execute_jobs(range.begin(), range.end());
                       });
    }
  });
#else
  execute_jobs(0, number_of_jobs);
#endif

  Eigen::Index number_of_outputs = 0;
  for (const auto& result : results) number_of_outputs += result.values.size();
  T_result_vector output(number_of_outputs);
  Eigen::Index offset = 0;
  for (std::size_t job = 0; job < number_of_jobs; ++job) {
    const auto& result = results[job];
    if (messages != nullptr) *messages << result.messages;
    if constexpr (is_var<T_result>::value) {
      if (result.values.size() == 0) continue;
      vari** nodes = nullptr;
      if (number_of_ordinary_derivatives > 0) {
        nodes = ChainableStack::instance_->memalloc_.alloc_array<vari*>(number_of_ordinary_derivatives);
      }
      if constexpr (shared_has_derivatives) {
        for (Eigen::Index index = 0; index < number_of_shared_derivatives; ++index) {
          nodes[index] = shared_parameters(index).vi_;
        }
      }
      if constexpr (jobs_have_derivatives) {
        for (Eigen::Index index = 0; index < number_of_job_derivatives; ++index) {
          nodes[number_of_shared_derivatives + index] = job_parameters.node(job, index);
        }
      }
      auto* derivatives = new nicostan_map_rect_derivatives_of_one_job(result, nodes, nuisance_block);
      for (Eigen::Index element = 0; element < result.values.size(); ++element) {
        output(offset++) = var(derivatives->output_node(element));
      }
    } else {
      output.segment(offset, result.values.size()) = result.values;
      offset += result.values.size();
    }
  }
  return output;
}

template <typename F, typename T_shared, typename T_job, typename T_type>
Eigen::Matrix<return_type_t<T_shared, T_job, T_type>, Eigen::Dynamic, 1>
nicostan_map_rect_chunked_with_partition(
      const T_shared& shared_parameters,
      const std::vector<Eigen::Matrix<T_job, Eigen::Dynamic, 1>>& job_parameters,
      const std::vector<std::vector<double>>& real_data,
      const std::vector<std::vector<int>>& integer_data,
      const T_type& derivative_type,
      const std::optional<int> chunk_size,
      const std::vector<int>* first_observation,
      const std::vector<int>* last_observation,
      std::ostream* messages) {
  const nicostan_map_ordinary_job_parameters<T_job> jobs(job_parameters);
  return nicostan_map_rect_chunked_from_job_source<F>(shared_parameters, jobs, real_data,
      integer_data, derivative_type, chunk_size, first_observation, last_observation, messages);
}

template <typename F, typename T_shared, typename T_type, typename T_loader>
Eigen::Matrix<return_type_t<T_shared, T_type>, Eigen::Dynamic, 1>
nicostan_map_rect_chunked_with_nuisance_job_loader(
      const T_shared& shared_parameters,
      const std::vector<std::vector<double>>& real_data,
      const std::vector<std::vector<int>>& integer_data,
      const T_type& derivative_type,
      const int chunk_size,
      const std::vector<int>& first_observation,
      const std::vector<int>& last_observation,
      const Eigen::Index parameters_per_job,
      const T_loader& loader,
      std::ostream* messages) {
  const nicostan_map_nuisance_job_parameters<T_loader> jobs(real_data.size(), parameters_per_job, loader);
  return nicostan_map_rect_chunked_from_job_source<F>(shared_parameters, jobs, real_data,
      integer_data, derivative_type, chunk_size, &first_observation, &last_observation, messages);
}

//// The explicit job-domain interface retains its caller's chunk_size; no automatic constant is selected.
template <typename F, typename T_shared, typename T_job, typename T_type>
Eigen::Matrix<return_type_t<T_shared, T_job, T_type>, Eigen::Dynamic, 1>
nicostan_map_rect_chunked(
      const T_shared& shared_parameters,
      const std::vector<Eigen::Matrix<T_job, Eigen::Dynamic, 1>>& job_parameters,
      const std::vector<std::vector<double>>& real_data,
      const std::vector<std::vector<int>>& integer_data,
      const T_type& derivative_type,
      const int chunk_size,
      std::ostream* messages) {
  return nicostan_map_rect_chunked_with_partition<F>(shared_parameters, job_parameters,
      real_data, integer_data, derivative_type, chunk_size, nullptr, nullptr, messages);
}

}  // namespace internal
}  // namespace math
}  // namespace stan

#endif






















