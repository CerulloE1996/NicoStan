#pragma once

#include <stan/math/rev/core.hpp>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace stan {
namespace math {
namespace internal {

struct nicostan_nuisance_block_t;

//// One sparse table belongs to one map job and its exact nested tape, never to the whole nuisance block.
//// Repeated reads share a leaf; its accumulated adjoint is collected once for each output's reverse sweep.
struct nicostan_reused_nuisance_nodes_for_one_map_job {
  const nicostan_nuisance_block_t* block;
  const ChainableStack::AutodiffStackStorage* tape;
  const std::size_t nested_depth;
  std::vector<std::pair<int, vari*>> nodes;
  std::size_t number_of_values;

  nicostan_reused_nuisance_nodes_for_one_map_job(const nicostan_nuisance_block_t* owner,
                                                const std::size_t expected_reads)
      : block(owner), tape(ChainableStack::instance_),
        nested_depth(tape->nested_var_stack_sizes_.size()), number_of_values(0) {

    if (owner == nullptr || expected_reads == 0) return;
    if (expected_reads > nodes.max_size() / 2) {
      throw std::length_error("nico_map_rect: nuisance read count exceeds the job's storage range.");
    }
    std::size_t capacity = 2;
    for (; capacity / 2 < expected_reads; capacity *= 2) {
      if (capacity > nodes.max_size() / 2) {
        throw std::length_error("nico_map_rect: nuisance read count exceeds the job's storage range.");
      }
    }
    nodes.resize(capacity, {0, nullptr});

  }

  bool matches(const nicostan_nuisance_block_t* owner) const {

    return block == owner && tape == ChainableStack::instance_ &&
           nested_depth == tape->nested_var_stack_sizes_.size();

  }

  vari** position_of_value(const int value) {

    if (nodes.empty() || number_of_values == nodes.size() / 2) {
      if (nodes.size() > nodes.max_size() / 2) {
        throw std::length_error("nico_map_rect: too many distinct nuisance reads in one job.");
      }
      const std::size_t next_size = nodes.empty() ? 2 : 2 * nodes.size();
      std::vector<std::pair<int, vari*>> next(next_size, {0, nullptr});
      for (const auto& entry : nodes) {
        if (entry.second == nullptr) continue;
        std::size_t position = static_cast<std::size_t>(entry.first) & (next_size - 1);
        for (; next[position].second != nullptr; position = (position + 1) & (next_size - 1)) {}
        next[position] = entry;
      }
      nodes.swap(next);
    }
    std::size_t position = static_cast<std::size_t>(value) & (nodes.size() - 1);
    for (; nodes[position].second != nullptr; position = (position + 1) & (nodes.size() - 1)) {
      if (nodes[position].first == value) return &nodes[position].second;
    }
    nodes[position].first = value;
    ++number_of_values;
    return &nodes[position].second;

  }
};

inline thread_local nicostan_reused_nuisance_nodes_for_one_map_job*
    nicostan_reused_nuisance_nodes_of_this_map_job = nullptr;

class nicostan_scope_of_reused_nuisance_nodes_for_one_map_job {
  nicostan_reused_nuisance_nodes_for_one_map_job nodes_;
  nicostan_reused_nuisance_nodes_for_one_map_job* previous_;

 public:
  explicit nicostan_scope_of_reused_nuisance_nodes_for_one_map_job(
        const nicostan_nuisance_block_t* block, const std::size_t expected_reads)
      : nodes_(block, expected_reads), previous_(nicostan_reused_nuisance_nodes_of_this_map_job) {

    nicostan_reused_nuisance_nodes_of_this_map_job = &nodes_;

  }

  ~nicostan_scope_of_reused_nuisance_nodes_for_one_map_job() {

    nicostan_reused_nuisance_nodes_of_this_map_job = previous_;

  }

  nicostan_scope_of_reused_nuisance_nodes_for_one_map_job(
        const nicostan_scope_of_reused_nuisance_nodes_for_one_map_job&) = delete;
  nicostan_scope_of_reused_nuisance_nodes_for_one_map_job& operator=(
        const nicostan_scope_of_reused_nuisance_nodes_for_one_map_job&) = delete;
};

}  // namespace internal
}  // namespace math
}  // namespace stan






















