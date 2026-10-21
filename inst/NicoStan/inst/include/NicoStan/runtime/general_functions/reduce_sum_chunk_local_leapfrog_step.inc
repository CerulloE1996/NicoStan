// NicoStan patch of stan/math/rev/functor/reduce_sum.hpp, "the chunk-local leapfrog step" (applied by
// R_fn_bridge_memory_patch_reduce_sum_chunk_local_leapfrog_step to NicoStan's private BridgeStan runtime when
// options(NicoStan.reduce_sum_chunk_local_leapfrog_step = TRUE), on top of the nuisance block owned by NicoStan
// and the lean slicing; never to a user's own Stan). The third part of reduce_sum_chunked: the sampler's step of
// the nuisance values that follows their gradient (the closing half kick, then the next step's half kick and
// rotation, or the closing half kick alone after the last step) is made inside each chunk, on the chunk's own
// values, right after the chunk has added their adjoints, instead of in a pass over all N values after the
// gradient call (at large N, each such pass streams the values, their gradient, their velocities and the
// factors of the rotation from DRAM, since the N-sized vectors of a chain no longer fit in its share of the L3).
// Only in the gradient call of a leapfrog step: NicoStan gives the step through the exported gradient call
// nicostan_bs_log_density_gradient_with_the_nuisance_block_and_the_chunk_local_leapfrog_step, and every other
// gradient call (through nicostan_bs_log_density_gradient_with_the_nuisance_block) is made exactly as before.
//   - each chunk zeroes the gradient of its own values just before it adds their adjoints (each value belongs
//     to one chunk), so the pass zeroing the whole nuisance gradient before the gradient call is gone; the
//     values no chunk owns are zeroed after the chunks, before the adjoints of the values read on the main
//     tape and those kept by the chunks for the values of other terms are added, in the same order as before;
//   - each chunk then has NicoStan make the step of its own values (NicoStan's own code, through the function
//     in the step: the same compiled arithmetic as NicoStan's pass), except the values that were read outside
//     their own chunk in the previous gradient call (e.g. a value read by the next chunk of a recursion), whose
//     gradient is complete only after all the chunks; NicoStan makes their step after the gradient call, and
//     makes again the step of any value made inside a chunk whose gradient then had adjoints from outside it,
//     so every value is made from its complete gradient whatever the model reads where;
//   - the step writes the next values into NicoStan's arrays of the next values, never over the values the
//     chunks read, so no chunk ever sees another chunk's next values.
// A model whose nuisance values are the own values of more than one reduce_sum in a gradient call (e.g. a
// reduce_sum inside a chunk that also reads its own values) stops the gradient with an error: each value's
// gradient must come from one chunk.
#include <algorithm>
#include <atomic>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

namespace stan {
namespace math {
namespace internal {

struct nicostan_chunk_local_leapfrog_step_in_the_model_t {
  nicostan_chunk_local_leapfrog_step_t* step = nullptr;   // NicoStan's description of the step
  std::atomic<int> number_of_reduce_sum_calls{0};
  std::atomic<int> reduce_sum_call_whose_chunks_own_the_values{0};
  std::mutex mutex_of_the_runs;
  std::vector<std::pair<int, int>> runs_of_values_owned_by_the_chunks;         // [first, end)
  std::vector<std::pair<int, int>> runs_of_values_stepped_inside_the_chunks;   // [first, end)
  std::vector<int> values_read_outside_their_own_chunk;
  std::vector<int> first_and_end_of_the_runs_of_values_stepped_inside_the_chunks;
  bool gradient_of_the_values_no_chunk_owns_zeroed = false;
};
//// (set by the exported gradient call of a leapfrog step for its duration, on the chain's thread; set for each
////  chunk on the thread that runs it, so that a reduce_sum inside a chunk sees it too)
inline thread_local nicostan_chunk_local_leapfrog_step_in_the_model_t* nicostan_chunk_local_leapfrog_step =
    nullptr;

struct nicostan_scope_of_a_chunk_of_the_chunk_local_leapfrog_step {
  nicostan_chunk_local_leapfrog_step_in_the_model_t* previous_step;
  explicit nicostan_scope_of_a_chunk_of_the_chunk_local_leapfrog_step(
      nicostan_chunk_local_leapfrog_step_in_the_model_t* step_of_the_caller)
      : previous_step(nicostan_chunk_local_leapfrog_step) {
    nicostan_chunk_local_leapfrog_step = step_of_the_caller;
  }
  ~nicostan_scope_of_a_chunk_of_the_chunk_local_leapfrog_step() {
    nicostan_chunk_local_leapfrog_step = previous_step;
  }
};

//// the number of a reduce_sum call within the gradient call (1, 2, ...; 0 outside a leapfrog step), taken on
//// the caller's thread
inline int nicostan_number_of_this_reduce_sum_call_of_the_chunk_local_leapfrog_step(
    nicostan_chunk_local_leapfrog_step_in_the_model_t* chunk_local_step) {
  if (chunk_local_step == nullptr) return 0;
  return chunk_local_step->number_of_reduce_sum_calls.fetch_add(1) + 1;
}

//// after the chunk's own gradient (its nodes still hold their adjoints): outside a leapfrog step, exactly as
//// nicostan_add_the_adjoints_of_the_chunk_of_the_nuisance_block(); in a leapfrog step, the gradient of the
//// chunk's own values zeroed and their adjoints added, then the step of those not read outside their own chunk
inline void nicostan_add_the_adjoints_of_the_chunk_of_the_nuisance_block_and_make_its_leapfrog_step(
    nicostan_chunk_local_leapfrog_step_in_the_model_t* chunk_local_step, const int reduce_sum_call) {
  if (chunk_local_step == nullptr) {
    nicostan_add_the_adjoints_of_the_chunk_of_the_nuisance_block();
    return;
  }
  nicostan_chunk_of_the_nuisance_block_t& chunk = nicostan_chunk_of_the_nuisance_block;
  nicostan_nuisance_block_t* block = nicostan_nuisance_block;
  if (!chunk.active || block == nullptr) return;
  if (chunk.nodes.empty() && chunk.nodes_of_the_values_of_other_terms.empty()) return;
  if (!chunk.nodes.empty()) {
    //// (the values of one reduce_sum call only: a value owned by the chunks of two calls would be zeroed and
    ////  stepped twice)
    int reduce_sum_call_before = 0;
    if (!chunk_local_step->reduce_sum_call_whose_chunks_own_the_values.compare_exchange_strong(
            reduce_sum_call_before, reduce_sum_call) &&
        reduce_sum_call_before != reduce_sum_call) {
      throw std::domain_error("NicoStan's chunk-local leapfrog step: the nuisance values are the own values of "
                              "more than one reduce_sum in a gradient call (build the model without "
                              "NicoStan.reduce_sum_chunk_local_leapfrog_step)");
    }
    //// ---- the chunk's own values, ascending, each once (the readers give them ascending; any other order is
    ////      sorted here)
    static thread_local std::vector<int> own_values;
    own_values.clear();
    own_values.reserve(chunk.nodes.size());
    bool strictly_ascending = true;
    for (const auto& value_and_node : chunk.nodes) {
      if (!own_values.empty() && value_and_node.first <= own_values.back()) strictly_ascending = false;
      own_values.push_back(value_and_node.first);
    }
    if (!strictly_ascending) {
      std::sort(own_values.begin(), own_values.end());
      own_values.erase(std::unique(own_values.begin(), own_values.end()), own_values.end());
    }
    //// ---- their gradient: zeroed, then the adjoints in the order of the nodes (as before, when the whole
    ////      gradient was zeroed before the gradient call)
    for (const int value : own_values) block->gradient[value] = 0.0;
    for (const auto& value_and_node : chunk.nodes) {
      block->gradient[value_and_node.first] += value_and_node.second->adj_;
    }
    //// ---- the runs of the own values, and the runs of the own values not read outside their own chunk in the
    ////      previous gradient call, whose step is made here
    nicostan_chunk_local_leapfrog_step_t* step = chunk_local_step->step;
    const int* values_read_outside = step->values_read_outside_their_own_chunk_in_the_previous_gradient_call;
    const int* end_of_values_read_outside =
        values_read_outside + step->n_values_read_outside_their_own_chunk_in_the_previous_gradient_call;
    const int* next_value_read_outside =
        std::lower_bound(values_read_outside, end_of_values_read_outside, own_values.front());
    std::vector<std::pair<int, int>> runs_owned;
    std::vector<std::pair<int, int>> runs_stepped;
    for (const int value : own_values) {
      if (!runs_owned.empty() && runs_owned.back().second == value) {
        runs_owned.back().second = value + 1;
      } else {
        runs_owned.emplace_back(value, value + 1);
      }
      while (next_value_read_outside < end_of_values_read_outside && *next_value_read_outside < value) {
        ++next_value_read_outside;
      }
      const bool read_outside_its_own_chunk = (next_value_read_outside < end_of_values_read_outside) &&
                                              (*next_value_read_outside == value);
      if (read_outside_its_own_chunk) continue;
      if (!runs_stepped.empty() && runs_stepped.back().second == value) {
        runs_stepped.back().second = value + 1;
      } else {
        runs_stepped.emplace_back(value, value + 1);
      }
    }
    for (const auto& run : runs_stepped) {
      step->fn_make_the_step_of_the_values_first_to_end(step->NicoStan_pass_over_the_values, run.first,
                                                        run.second);
    }
    std::lock_guard<std::mutex> lock(chunk_local_step->mutex_of_the_runs);
    chunk_local_step->runs_of_values_owned_by_the_chunks.insert(
        chunk_local_step->runs_of_values_owned_by_the_chunks.end(), runs_owned.begin(), runs_owned.end());
    chunk_local_step->runs_of_values_stepped_inside_the_chunks.insert(
        chunk_local_step->runs_of_values_stepped_inside_the_chunks.end(), runs_stepped.begin(),
        runs_stepped.end());
  }
  chunk.nodes.clear();
  block->read_inside_a_reduce_sum.store(true, std::memory_order_relaxed);
  //// ---- the adjoints of the values of other terms: kept for after all the chunks, as before
  if (chunk.nodes_of_the_values_of_other_terms.empty()) return;
  std::vector<std::pair<int, double>> other_values;
  other_values.reserve(chunk.nodes_of_the_values_of_other_terms.size());
  for (const auto& value_and_node : chunk.nodes_of_the_values_of_other_terms) {
    other_values.emplace_back(value_and_node.first, value_and_node.second->adj_);
  }
  chunk.nodes_of_the_values_of_other_terms.clear();
  std::lock_guard<std::mutex> lock(block->mutex_of_the_adjoints_from_other_chunks);
  block->adjoints_from_other_chunks.insert(block->adjoints_from_other_chunks.end(), other_values.begin(),
                                           other_values.end());
}

//// after all the chunks: the gradient of the values no chunk owns zeroed (once per gradient call)
inline void nicostan_zero_the_gradient_of_the_values_no_chunk_owns(
    nicostan_chunk_local_leapfrog_step_in_the_model_t* chunk_local_step, nicostan_nuisance_block_t* block) {
  if (chunk_local_step == nullptr || block == nullptr) return;
  if (chunk_local_step->gradient_of_the_values_no_chunk_owns_zeroed) return;
  std::vector<std::pair<int, int>>& runs = chunk_local_step->runs_of_values_owned_by_the_chunks;
  std::sort(runs.begin(), runs.end());
  int first_value_not_owned = 0;
  for (const auto& run : runs) {
    for (int value = first_value_not_owned; value < run.first; ++value) block->gradient[value] = 0.0;
    first_value_not_owned = std::max(first_value_not_owned, run.second);
  }
  for (int value = first_value_not_owned; value < block->size; ++value) block->gradient[value] = 0.0;
  chunk_local_step->gradient_of_the_values_no_chunk_owns_zeroed = true;
}

//// in BridgeStan's gradient, before the adjoints of the values read on the main tape are added: the values no
//// chunk owns zeroed, and the values read on the main tape kept as values read outside their own chunk
inline void nicostan_before_the_adjoints_of_the_values_read_on_the_main_tape(nicostan_nuisance_block_t* block) {
  nicostan_chunk_local_leapfrog_step_in_the_model_t* chunk_local_step = nicostan_chunk_local_leapfrog_step;
  if (chunk_local_step == nullptr) return;
  nicostan_zero_the_gradient_of_the_values_no_chunk_owns(chunk_local_step, block);
  for (const auto& unit_and_node : block->nodes_on_the_main_tape) {
    chunk_local_step->values_read_outside_their_own_chunk.push_back(unit_and_node.first);
  }
}

}  // namespace internal
}  // namespace math
}  // namespace stan























