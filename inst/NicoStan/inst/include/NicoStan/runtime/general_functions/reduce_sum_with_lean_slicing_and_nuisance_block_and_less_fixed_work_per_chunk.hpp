#ifndef STAN_MATH_REV_FUNCTOR_REDUCE_SUM_HPP
#define STAN_MATH_REV_FUNCTOR_REDUCE_SUM_HPP

// ==== reduce_sum_with_lean_slicing_and_nuisance_block_and_less_fixed_work_per_chunk.hpp ========================
//
// ---- A whole replacement of stan/math/rev/functor/reduce_sum.hpp for a COPY of NicoStan's private BridgeStan
//      runtime (never for the runtime NicoStan prepares itself, never for a user's own Stan): the reduce_sum of
//      the runtime with NicoStan's lean slicing and its nuisance block (R_fn_bridge_memory.R, the options
//      NicoStan.reduce_sum_lean_slicing = TRUE and NicoStan.reduce_sum_nuisance_block = TRUE), with two switches
//      that must both be defined (0 or 1) at the top of the copy, before its include guard:
//
//   NICOSTAN_REDUCE_SUM_WITH_LESS_FIXED_WORK_PER_CHUNK
//      0: every chunk does exactly the work of the runtime of now (the code of the .inc files, line for line);
//      1: the same arithmetic, in the same order, with less work that does not depend on the chunk's size:
//         (a) reduce_sum_static: the shared arguments are copied onto each chunk's own nested tape, instead of
//             a new ScopedChainableStack per body (its own 64 KB block from malloc, its own lists of nodes) and
//             one copy there; with a deterministic partition every chunk has its own body (TBB splits the body
//             at every split of the range), so the scoped stack was made, filled once and freed for each chunk;
//         (b) no std::stringstream per body: a chunk writes its messages into one stream of its thread (one per
//             depth of nested reduce_sum calls), moved into a stream of its body only when it wrote something;
//         (c) the lists of the nodes of the nuisance values that a chunk reads keep their memory on the thread
//             from one chunk to the next (each chunk works at their end, and gives back what it added), instead
//             of two new lists per chunk that grow from empty;
//         (d) reduce_sum_static on a thread whose arena has 1 thread (NicoStan's arena of a chain with 1 thread
//             per chain, NICOSTAN_PER_CHAIN_TBB_ARENA=1): the chunks, the bodies and the joins in exactly the
//             tree of TBB's parallel_deterministic_reduce with its simple_partitioner (the range halved until a
//             piece has at most grainsize terms; the left half keeps the body, the right half gets a new body,
//             then left.join(right)), on the thread itself, without TBB's tasks;
//         every value and every adjoint is added in the same order as before, so the log density and the
//         gradient are the same to the last bit;
//
//   NICOSTAN_COUNT_THE_INSTRUCTIONS_OF_EACH_PART_OF_REDUCE_SUM
//      0: nothing counted;
//      1: the instructions (user space, the thread's own hardware counter, perf_event_open) of each part of
//         reduce_sum are added up per part, on every thread, and written when the process ends into the file
//         named by the environment variable NICOSTAN_FILE_OF_THE_INSTRUCTIONS_OF_EACH_PART_OF_REDUCE_SUM (one
//         line per part: part, instructions, number of times, the instructions of one reading of the counter,
//         to be taken off each time, and the number of threads). A measurement only: never for a timing (each
//         reading is a system call), and never beside pmc_stat (the core has 6 counters, all taken by pmc_stat:
//         a 7th event makes the kernel share the counters in turns, and both counts would be wrong).
//
// The two NicoStan descriptions below (the lean slicing and the nuisance block) are those of the runtime.
//
// ===============================================================================================================

#if !defined(NICOSTAN_REDUCE_SUM_WITH_LESS_FIXED_WORK_PER_CHUNK)
#error "NICOSTAN_REDUCE_SUM_WITH_LESS_FIXED_WORK_PER_CHUNK must be defined (0 or 1) at the top of this file"
#endif
#if !defined(NICOSTAN_COUNT_THE_INSTRUCTIONS_OF_EACH_PART_OF_REDUCE_SUM)
#error "NICOSTAN_COUNT_THE_INSTRUCTIONS_OF_EACH_PART_OF_REDUCE_SUM must be defined (0 or 1) at the top"
#endif

#include <stan/math/prim/meta.hpp>
#include <stan/math/prim/functor.hpp>
#include <stan/math/rev/core.hpp>

#include <tbb/task_arena.h>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>

#include <tuple>
#include <memory>
#include <utility>
#include <vector>
#include <sstream>

// NicoStan patch of stan/math/rev/functor/reduce_sum.hpp, "lean slicing" (applied by
// R_fn_bridge_memory_patch_reduce_sum_lean_slicing to NicoStan's private BridgeStan runtime when
// options(NicoStan.reduce_sum_lean_slicing = TRUE); never to a user's own Stan): reduce_sum's own bookkeeping of
// the sliced terms is done inside the chunks, in N_chunks blocks, instead of in passes over all N terms on the
// main thread. For every sliced term Stan's reduce_sum (1) writes a pointer to its node into a list on the main
// tape, (2) zero-fills its partial, both in passes over all N terms before the chunks, and (3) after the chunks
// adds the adjoints through the pointer list. Here:
//   (1) each chunk writes the pointers of its own terms into the list, from the same terms it is copying onto its
//       own tape (so they are already in cache), in place of the pass over all N on the main thread;
//   (2) each chunk zeroes its own partials just before it accumulates them (each term belongs to exactly one
//       chunk), so the separate zero-fill pass over all of them is gone;
//   (3) the adjoints are added by Stan's own precomputed_gradients_vari, through the same pointer list in the
//       same order, so the gradient is identical.
// General: any sliced argument (arrays of scalars, of vectors or row_vectors, of arrays; data or parameters) and
// any shared arguments. The same runtime for every math backend (Stan math, AVX2, AVX-512).
#include <algorithm>

// NicoStan patch of stan/math/rev/functor/reduce_sum.hpp, "the nuisance block owned by NicoStan" (applied by
// R_fn_bridge_memory_patch_reduce_sum_nuisance_block to NicoStan's private BridgeStan runtime when
// options(NicoStan.reduce_sum_nuisance_block = TRUE), on top of the lean slicing; never to a user's own Stan).
// A model written in this form declares only its main parameters in its parameters block; its N-sized nuisance
// block (one or more latent values per unit, e.g. the latent path of a state-space model) is NicoStan's, and the
// model reads its values with nicostan_nuisance_values(first, last, type) inside the chunks of a reduce_sum. So:
//   - no node of the main tape, no copy in the model and no entry of reduce_sum's lists is made for the nuisance
//     values: each chunk makes nodes of its own values on its own tape, from NicoStan's array of values;
//   - after the chunk's own gradient, the adjoints of its own values are added straight into NicoStan's array of
//     the nuisance gradient (each unit belongs to exactly one chunk); the adjoints of values a chunk read from
//     other units (e.g. the previous unit of a recursion), and of values read on the main tape, are added after
//     all the chunks;
//   - the sum returned by reduce_sum must be added to the target as it is (its adjoint is then exactly 1): any
//     other use stops the gradient with an error, since the chunks' adjoints were added with a factor of 1.
// The gradient is the same as that of the model with the nuisance block declared as parameters, up to the
// order of the additions of the few values read outside their own chunk.
#include <atomic>
#include <cstddef>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

#if NICOSTAN_COUNT_THE_INSTRUCTIONS_OF_EACH_PART_OF_REDUCE_SUM
#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#endif

namespace stan {
namespace math {
namespace internal {

struct nicostan_nuisance_block_t {
  const double* values = nullptr;   // NicoStan's array of the nuisance values (0-based)
  double* gradient = nullptr;       // NicoStan's array of their gradient, zeroed before the gradient call
  int size = 0;
  std::vector<std::pair<int, vari*>> nodes_on_the_main_tape;     // (unit, node) of values read on the main tape
  std::vector<std::pair<int, double>> adjoints_from_other_chunks;  // (unit, adjoint) of values read by the chunk
  std::mutex mutex_of_the_adjoints_from_other_chunks;              //  of another unit
  std::atomic<bool> read_inside_a_reduce_sum{false};
  //// Repeated main-tape reads share one leaf per nuisance value within this gradient only.
  //// No storage is allocated for a model which reads its nuisance values only inside chunks.
  std::vector<vari*> nodes_by_value_on_the_main_tape;
  const ChainableStack::AutodiffStackStorage* identity_of_the_main_tape = nullptr;
  std::size_t nested_depth_of_the_main_tape = std::numeric_limits<std::size_t>::max();
};
//// (set by NicoStan's gradient call for its duration, on the chain's thread; set for each chunk on the thread
////  that runs it)
inline thread_local nicostan_nuisance_block_t* nicostan_nuisance_block = nullptr;
//// (the nuisance values used when the model is evaluated outside NicoStan's gradient call, e.g. through
////  BridgeStan's own interface: set by nicostan_set_the_nuisance_values_used_outside_NicoStan_R; their gradient
////  is not kept)
inline std::vector<double> nicostan_nuisance_values_used_outside_NicoStan;

//// ---- Numeric nuisance derivatives of one reduce_sum call:
////
inline thread_local std::vector<std::pair<int, double>>* nicostan_nuisance_derivatives_of_the_chunk = nullptr;

struct nicostan_scope_of_the_nuisance_derivative_collection {
  std::vector<std::pair<int, double>>* previous_collection;

  explicit nicostan_scope_of_the_nuisance_derivative_collection(
        std::vector<std::pair<int, double>>& derivatives)
      : previous_collection(nicostan_nuisance_derivatives_of_the_chunk) {

    nicostan_nuisance_derivatives_of_the_chunk = &derivatives;

  }

  ~nicostan_scope_of_the_nuisance_derivative_collection() {

    nicostan_nuisance_derivatives_of_the_chunk = previous_collection;

  }
};

struct nicostan_chunk_of_the_nuisance_block_t {
  bool active = false;
  int first = 0;   // the chunk's own terms (0-based, inclusive): the terms of reduce_sum's sliced argument
  int last = -1;
  //// (value, node on the chunk's own tape) of every value it read: the values of its own terms (each term
  ////  belongs to exactly one chunk, so only this chunk adds into their gradient), and the values of other terms
  ////  (e.g. the previous term of a recursion), whose adjoints are added after all the chunks; the readers
  ////  decide which is which, from the term of each value
  std::vector<std::pair<int, vari*>> nodes;
  std::vector<std::pair<int, vari*>> nodes_of_the_values_of_other_terms;
  //// (used with NICOSTAN_REDUCE_SUM_WITH_LESS_FIXED_WORK_PER_CHUNK = 1: where the chunk's own entries of the two
  ////  lists begin; the entries before them belong to the chunk of an outer reduce_sum on the same thread)
  std::size_t first_node = 0;
  std::size_t first_node_of_the_values_of_other_terms = 0;
};
inline thread_local nicostan_chunk_of_the_nuisance_block_t nicostan_chunk_of_the_nuisance_block;

#if NICOSTAN_REDUCE_SUM_WITH_LESS_FIXED_WORK_PER_CHUNK

//// for the duration of one chunk: the caller's nuisance block and the chunk's own units; the two lists of the
//// thread keep their memory, and the chunk's own entries are those after the sizes they had when it began
struct nicostan_scope_of_a_chunk_of_the_nuisance_block {
  nicostan_chunk_of_the_nuisance_block_t* chunk_of_this_thread;
  nicostan_nuisance_block_t* previous_block;
  bool previous_active;
  int previous_first;
  int previous_last;
  std::size_t previous_first_node;
  std::size_t previous_first_node_of_the_values_of_other_terms;
  nicostan_scope_of_a_chunk_of_the_nuisance_block(nicostan_nuisance_block_t* block, int first, int last)
      : chunk_of_this_thread(&nicostan_chunk_of_the_nuisance_block),
        previous_block(nicostan_nuisance_block) {

    nicostan_chunk_of_the_nuisance_block_t& chunk = *chunk_of_this_thread;
    previous_active = chunk.active;
    previous_first = chunk.first;
    previous_last = chunk.last;
    previous_first_node = chunk.first_node;
    previous_first_node_of_the_values_of_other_terms = chunk.first_node_of_the_values_of_other_terms;
    nicostan_nuisance_block = block;
    chunk.active = (block != nullptr);
    chunk.first = first;
    chunk.last = last;
    chunk.first_node = chunk.nodes.size();
    chunk.first_node_of_the_values_of_other_terms = chunk.nodes_of_the_values_of_other_terms.size();

  }
  ~nicostan_scope_of_a_chunk_of_the_nuisance_block() {

    nicostan_chunk_of_the_nuisance_block_t& chunk = *chunk_of_this_thread;
    //// (the chunk's own entries are given back, also when the chunk stopped with an exception)
    chunk.nodes.resize(chunk.first_node);
    chunk.nodes_of_the_values_of_other_terms.resize(chunk.first_node_of_the_values_of_other_terms);
    chunk.active = previous_active;
    chunk.first = previous_first;
    chunk.last = previous_last;
    chunk.first_node = previous_first_node;
    chunk.first_node_of_the_values_of_other_terms = previous_first_node_of_the_values_of_other_terms;
    nicostan_nuisance_block = previous_block;

  }
};

//// after the chunk's own gradient (its nodes still hold their adjoints): the adjoints of the values of its own
//// terms into the nuisance gradient, those of the values of other terms kept for after all the chunks (the same
//// additions, in the same order, as the runtime of now)
inline void nicostan_add_the_adjoints_of_the_chunk_of_the_nuisance_block() {

  nicostan_chunk_of_the_nuisance_block_t& chunk = nicostan_chunk_of_the_nuisance_block;
  nicostan_nuisance_block_t* block = nicostan_nuisance_block;
  if (!chunk.active || block == nullptr) return;
  const std::size_t end_of_the_nodes = chunk.nodes.size();
  const std::size_t end_of_the_other_nodes = chunk.nodes_of_the_values_of_other_terms.size();
  if (end_of_the_nodes == chunk.first_node &&
      end_of_the_other_nodes == chunk.first_node_of_the_values_of_other_terms) return;
  if (nicostan_nuisance_derivatives_of_the_chunk != nullptr) {
    auto& derivatives = *nicostan_nuisance_derivatives_of_the_chunk;
    for (std::size_t i = chunk.first_node; i < end_of_the_nodes; ++i) {
      derivatives.emplace_back(chunk.nodes[i].first, chunk.nodes[i].second->adj_);
    }
    for (std::size_t i = chunk.first_node_of_the_values_of_other_terms; i < end_of_the_other_nodes; ++i) {
      derivatives.emplace_back(chunk.nodes_of_the_values_of_other_terms[i].first,
                               chunk.nodes_of_the_values_of_other_terms[i].second->adj_);
    }
    chunk.nodes.resize(chunk.first_node);
    chunk.nodes_of_the_values_of_other_terms.resize(chunk.first_node_of_the_values_of_other_terms);
    return;
  }
  for (std::size_t i = chunk.first_node; i < end_of_the_nodes; ++i) {
    block->gradient[chunk.nodes[i].first] += chunk.nodes[i].second->adj_;
  }
  chunk.nodes.resize(chunk.first_node);
  block->read_inside_a_reduce_sum.store(true, std::memory_order_relaxed);
  if (end_of_the_other_nodes == chunk.first_node_of_the_values_of_other_terms) return;
  std::vector<std::pair<int, double>> other_values;
  other_values.reserve(end_of_the_other_nodes - chunk.first_node_of_the_values_of_other_terms);
  for (std::size_t i = chunk.first_node_of_the_values_of_other_terms; i < end_of_the_other_nodes; ++i) {
    other_values.emplace_back(chunk.nodes_of_the_values_of_other_terms[i].first,
                              chunk.nodes_of_the_values_of_other_terms[i].second->adj_);
  }
  chunk.nodes_of_the_values_of_other_terms.resize(chunk.first_node_of_the_values_of_other_terms);
  std::lock_guard<std::mutex> lock(block->mutex_of_the_adjoints_from_other_chunks);
  block->adjoints_from_other_chunks.insert(block->adjoints_from_other_chunks.end(), other_values.begin(),
                                           other_values.end());

}

//// the streams into which the chunks of this thread write their messages: one per depth of nested reduce_sum
//// calls (a chunk may call reduce_sum itself), kept for the life of the thread
struct nicostan_message_streams_of_this_thread_t {
  std::vector<std::unique_ptr<std::ostringstream>> streams;
  std::size_t depth = 0;
};
inline thread_local nicostan_message_streams_of_this_thread_t nicostan_message_streams_of_this_thread;

#else

//// for the duration of one chunk: the caller's nuisance block and the chunk's own units
struct nicostan_scope_of_a_chunk_of_the_nuisance_block {
  nicostan_nuisance_block_t* previous_block;
  nicostan_chunk_of_the_nuisance_block_t previous_chunk;
  nicostan_scope_of_a_chunk_of_the_nuisance_block(nicostan_nuisance_block_t* block, int first, int last)
      : previous_block(nicostan_nuisance_block),
        previous_chunk(std::move(nicostan_chunk_of_the_nuisance_block)) {

    nicostan_nuisance_block = block;
    nicostan_chunk_of_the_nuisance_block = nicostan_chunk_of_the_nuisance_block_t();
    nicostan_chunk_of_the_nuisance_block.active = (block != nullptr);
    nicostan_chunk_of_the_nuisance_block.first = first;
    nicostan_chunk_of_the_nuisance_block.last = last;

  }
  ~nicostan_scope_of_a_chunk_of_the_nuisance_block() {

    nicostan_nuisance_block = previous_block;
    nicostan_chunk_of_the_nuisance_block = std::move(previous_chunk);

  }
};

//// after the chunk's own gradient (its nodes still hold their adjoints): the adjoints of the values of its own
//// terms into the nuisance gradient, those of the values of other terms kept for after all the chunks
inline void nicostan_add_the_adjoints_of_the_chunk_of_the_nuisance_block() {

  nicostan_chunk_of_the_nuisance_block_t& chunk = nicostan_chunk_of_the_nuisance_block;
  nicostan_nuisance_block_t* block = nicostan_nuisance_block;
  if (!chunk.active || block == nullptr) return;
  if (chunk.nodes.empty() && chunk.nodes_of_the_values_of_other_terms.empty()) return;
  if (nicostan_nuisance_derivatives_of_the_chunk != nullptr) {
    auto& derivatives = *nicostan_nuisance_derivatives_of_the_chunk;
    for (const auto& value_and_node : chunk.nodes) {
      derivatives.emplace_back(value_and_node.first, value_and_node.second->adj_);
    }
    for (const auto& value_and_node : chunk.nodes_of_the_values_of_other_terms) {
      derivatives.emplace_back(value_and_node.first, value_and_node.second->adj_);
    }
    chunk.nodes.clear();
    chunk.nodes_of_the_values_of_other_terms.clear();
    return;
  }
  for (const auto& value_and_node : chunk.nodes) {
    block->gradient[value_and_node.first] += value_and_node.second->adj_;
  }
  chunk.nodes.clear();
  block->read_inside_a_reduce_sum.store(true, std::memory_order_relaxed);
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

#endif

//// reduce_sum's node of the sum: the chunks added their adjoints with a factor of 1, so the sum must be added
//// to the target as it is
//// Active reducers instead collect numeric derivatives and defer every nuisance-gradient addition to this
//// returned node. Its reverse pass applies the actual multiplier; nested calls add to the outer collection.
class nicostan_precomputed_gradients_vari_of_the_nuisance_block : public precomputed_gradients_vari {
  nicostan_nuisance_block_t* block_;
  std::size_t number_of_nuisance_derivatives_;
  int* nuisance_indices_;
  double* nuisance_derivatives_;

 public:
  nicostan_precomputed_gradients_vari_of_the_nuisance_block(
        double val, std::size_t size, vari** varis, double* gradients, nicostan_nuisance_block_t* block,
        const std::vector<std::pair<int, double>>& derivatives)
      : precomputed_gradients_vari(val, size, varis, gradients), block_(block),
        number_of_nuisance_derivatives_(derivatives.size()), nuisance_indices_(nullptr),
        nuisance_derivatives_(nullptr) {

    if (number_of_nuisance_derivatives_ == 0) return;
    nuisance_indices_ = ChainableStack::instance_->memalloc_.alloc_array<int>(number_of_nuisance_derivatives_);
    nuisance_derivatives_ =
        ChainableStack::instance_->memalloc_.alloc_array<double>(number_of_nuisance_derivatives_);
    for (std::size_t i = 0; i < number_of_nuisance_derivatives_; ++i) {
      nuisance_indices_[i] = derivatives[i].first;
      nuisance_derivatives_[i] = derivatives[i].second;
    }

  }

  void chain() {

    if (adj_ == 0.0) return;
    precomputed_gradients_vari::chain();
    for (std::size_t i = 0; i < number_of_nuisance_derivatives_; ++i) {
      const double derivative = adj_ == 1.0 ? nuisance_derivatives_[i] : adj_ * nuisance_derivatives_[i];
      if (nicostan_nuisance_derivatives_of_the_chunk != nullptr) {
        nicostan_nuisance_derivatives_of_the_chunk->emplace_back(nuisance_indices_[i], derivative);
      } else if (block_ != nullptr) {
        block_->gradient[nuisance_indices_[i]] += derivative;
      }
    }

  }
};

}  // namespace internal
}  // namespace math
}  // namespace stan


// ---- The instructions of each part of reduce_sum
//      (NICOSTAN_COUNT_THE_INSTRUCTIONS_OF_EACH_PART_OF_REDUCE_SUM = 1)

#if NICOSTAN_COUNT_THE_INSTRUCTIONS_OF_EACH_PART_OF_REDUCE_SUM

namespace stan {
namespace math {
namespace internal {

enum nicostan_part_of_reduce_sum : int {
  nicostan_part_whole_reduce_sum_call = 0,
  nicostan_part_body_made_for_a_split_of_the_range,
  nicostan_part_body_destroyed,
  nicostan_part_join_of_two_bodies,
  nicostan_part_shared_arguments_copied_or_their_adjoints_zeroed,
  nicostan_part_nested_tape_started_and_sliced_terms_copied,
  nicostan_part_chunk_function_forward_pass,
  nicostan_part_chunk_function_reverse_pass,
  nicostan_part_nuisance_adjoints_added,
  nicostan_part_sliced_partials_and_shared_adjoints_accumulated,
  nicostan_part_nested_tape_recovered_and_chunk_scopes_closed,
  nicostan_number_of_parts_of_reduce_sum
};

inline const char* const nicostan_names_of_the_parts_of_reduce_sum[nicostan_number_of_parts_of_reduce_sum] = {
  "whole_reduce_sum_call",
  "body_made_for_a_split_of_the_range",
  "body_destroyed",
  "join_of_two_bodies",
  "shared_arguments_copied_or_their_adjoints_zeroed",
  "nested_tape_started_and_sliced_terms_copied",
  "chunk_function_forward_pass",
  "chunk_function_reverse_pass",
  "nuisance_adjoints_added",
  "sliced_partials_and_shared_adjoints_accumulated",
  "nested_tape_recovered_and_chunk_scopes_closed"
};

struct nicostan_instructions_of_the_parts_of_one_thread_t {
  int file_descriptor_of_the_counter = -1;
  double instructions_of_one_reading = 0.0;
  unsigned long long instructions[nicostan_number_of_parts_of_reduce_sum] = {0};
  unsigned long long number_of_times[nicostan_number_of_parts_of_reduce_sum] = {0};
};

//// every thread's counts (each thread's block is kept to the end of the process), written when it ends
struct nicostan_instructions_of_the_parts_of_every_thread_t {
  std::mutex mutex;
  std::vector<nicostan_instructions_of_the_parts_of_one_thread_t*> threads;
  ~nicostan_instructions_of_the_parts_of_every_thread_t() {

    const char* file_name = std::getenv("NICOSTAN_FILE_OF_THE_INSTRUCTIONS_OF_EACH_PART_OF_REDUCE_SUM");
    if (file_name == nullptr || threads.empty()) return;
    std::FILE* file = std::fopen(file_name, "a");
    if (file == nullptr) return;
    double sum_of_the_instructions_of_one_reading = 0.0;
    int number_of_threads_with_a_counter = 0;
    unsigned long long instructions[nicostan_number_of_parts_of_reduce_sum] = {0};
    unsigned long long number_of_times[nicostan_number_of_parts_of_reduce_sum] = {0};
    for (const auto* thread : threads) {
      if (thread->file_descriptor_of_the_counter < 0) continue;
      sum_of_the_instructions_of_one_reading += thread->instructions_of_one_reading;
      number_of_threads_with_a_counter += 1;
      for (int part = 0; part < nicostan_number_of_parts_of_reduce_sum; ++part) {
        instructions[part] += thread->instructions[part];
        number_of_times[part] += thread->number_of_times[part];
      }
    }
    std::fprintf(file, "part,instructions,number_of_times,instructions_of_one_reading,threads\n");
    for (int part = 0; part < nicostan_number_of_parts_of_reduce_sum; ++part) {
      std::fprintf(file, "%s,%llu,%llu,%.1f,%d\n", nicostan_names_of_the_parts_of_reduce_sum[part],
                   instructions[part], number_of_times[part],
                   (number_of_threads_with_a_counter > 0)
                         ? sum_of_the_instructions_of_one_reading / number_of_threads_with_a_counter : -1.0,
                   number_of_threads_with_a_counter);
    }
    std::fclose(file);

  }
};
inline nicostan_instructions_of_the_parts_of_every_thread_t nicostan_instructions_of_the_parts_of_every_thread;

inline unsigned long long nicostan_reading_of_the_counter(int file_descriptor) {

  unsigned long long value = 0;
  if (file_descriptor < 0 || ::read(file_descriptor, &value, sizeof(value)) != sizeof(value)) return 0;
  return value;

}

//// this thread's block: its own counter of the instructions retired in user space (made at its first use),
//// and the instructions of one reading of it, measured over 64 readings in a row
inline nicostan_instructions_of_the_parts_of_one_thread_t& nicostan_instructions_of_the_parts_of_this_thread() {

  thread_local nicostan_instructions_of_the_parts_of_one_thread_t* parts_of_this_thread = [] {
    auto* parts = new nicostan_instructions_of_the_parts_of_one_thread_t();
    perf_event_attr attributes;
    std::memset(&attributes, 0, sizeof(attributes));
    attributes.size = sizeof(attributes);
    attributes.type = PERF_TYPE_HARDWARE;
    attributes.config = PERF_COUNT_HW_INSTRUCTIONS;
    attributes.exclude_kernel = 1;
    attributes.exclude_hv = 1;
    parts->file_descriptor_of_the_counter = static_cast<int>(::syscall(__NR_perf_event_open, &attributes, 0,
                                                                       -1, -1, 0));
    if (parts->file_descriptor_of_the_counter >= 0) {
      const unsigned long long first = nicostan_reading_of_the_counter(parts->file_descriptor_of_the_counter);
      unsigned long long last = first;
      for (int i = 0; i < 64; ++i) last = nicostan_reading_of_the_counter(parts->file_descriptor_of_the_counter);
      parts->instructions_of_one_reading = static_cast<double>(last - first) / 64.0;
    }
    std::lock_guard<std::mutex> lock(nicostan_instructions_of_the_parts_of_every_thread.mutex);
    nicostan_instructions_of_the_parts_of_every_thread.threads.push_back(parts);
    return parts;
  }();
  return *parts_of_this_thread;

}

inline unsigned long long nicostan_instructions_now() {

  return nicostan_reading_of_the_counter(
        nicostan_instructions_of_the_parts_of_this_thread().file_descriptor_of_the_counter);

}

//// adds the instructions since "from" to the part, and returns the reading of now (the start of the next part)
inline unsigned long long nicostan_add_to_the_part(int part, unsigned long long from) {

  const unsigned long long now = nicostan_instructions_now();
  nicostan_instructions_of_the_parts_of_one_thread_t& parts = nicostan_instructions_of_the_parts_of_this_thread();
  parts.instructions[part] += now - from;
  parts.number_of_times[part] += 1;
  return now;

}

//// the first member of a body: its reading when the body begins to be made (its members are made in their
//// order, so this one first) and when its destructor begins (its members are destroyed in reverse order, so
//// this one last, and its destructor adds the whole destruction of the body)
struct nicostan_readings_of_a_body {
  unsigned long long when_made = nicostan_instructions_now();
  unsigned long long when_its_destruction_began = 0;
  ~nicostan_readings_of_a_body() {

    if (when_its_destruction_began != 0) {
      nicostan_add_to_the_part(nicostan_part_body_destroyed, when_its_destruction_began);
    }

  }
};

//// the last part of a chunk (made first in operator(), so destroyed last): from the end of the accumulation to
//// after the nested tape is recovered and every scope of the chunk is closed
struct nicostan_reading_at_the_end_of_a_chunk {
  unsigned long long from = 0;
  ~nicostan_reading_at_the_end_of_a_chunk() {

    if (from != 0) nicostan_add_to_the_part(nicostan_part_nested_tape_recovered_and_chunk_scopes_closed, from);

  }
};

}  // namespace internal
}  // namespace math
}  // namespace stan

#define NICOSTAN_INSTRUCTIONS_NOW() ::stan::math::internal::nicostan_instructions_now()
#define NICOSTAN_ADD_TO_THE_PART(part, from) ::stan::math::internal::nicostan_add_to_the_part( \
      ::stan::math::internal::part, from)

#else

#define NICOSTAN_INSTRUCTIONS_NOW() 0ULL
#define NICOSTAN_ADD_TO_THE_PART(part, from) 0ULL

#endif


namespace stan {
namespace math {
namespace internal {

/**
 * Var specialization of reduce_sum_impl
 *
 * @tparam ReduceFunction Type of reducer function
 * @tparam ReturnType Must be var
 * @tparam Vec Type of sliced argument
 * @tparam Args Types of shared arguments
 */
template <typename ReduceFunction, typename ReturnType, typename Vec,
          typename... Args>
struct reduce_sum_impl<ReduceFunction, require_var_t<ReturnType>, ReturnType,
                       Vec, Args...> {
  struct scoped_args_tuple {
    ScopedChainableStack stack_;
    using args_tuple_t
        = std::tuple<decltype(deep_copy_vars(std::declval<Args>()))...>;
    std::unique_ptr<args_tuple_t> args_tuple_holder_;

    scoped_args_tuple() : stack_(), args_tuple_holder_(nullptr) {}
  };

#if NICOSTAN_REDUCE_SUM_WITH_LESS_FIXED_WORK_PER_CHUNK

  /**
   * The body of tbb::parallel_reduce / parallel_deterministic_reduce (the Imperative form), with less work that
   *  does not depend on the size of a chunk (see the top of this file): the same values and adjoints, added in
   *  the same order.
   */
  struct recursive_reducer {
#if NICOSTAN_COUNT_THE_INSTRUCTIONS_OF_EACH_PART_OF_REDUCE_SUM
    nicostan_readings_of_a_body nicostan_readings_of_this_body_;
#endif
    const size_t num_vars_per_term_;
    const size_t num_vars_shared_terms_;  // Number of vars in shared arguments
    double* sliced_partials_;  // Points to adjoints of the partial calculations
    Vec vmapped_;
    std::tuple<Args...> args_tuple_;
    //// (reduce_sum_static, a deterministic partition: TRUE, each chunk copies the shared arguments onto its own
    ////  nested tape; reduce_sum, an automatic partition: FALSE, one copy per body in its own scoped stack, made
    ////  at its first chunk, as in Stan, since such a body may run many chunks)
    bool copy_the_shared_arguments_in_each_chunk_{true};
    std::unique_ptr<scoped_args_tuple> local_args_tuple_scope_;
    //// (made only when a chunk of this body wrote a message)
    std::unique_ptr<std::stringstream> msgs_;
    double sum_{0.0};
    vari** nicostan_sliced_varis_{nullptr};  // (NicoStan: each chunk writes its own terms' nodes here)
    nicostan_nuisance_block_t* nicostan_nuisance_block_of_the_caller_{nicostan_nuisance_block};
    std::vector<std::pair<int, double>> nicostan_nuisance_derivatives_;
    Eigen::Matrix<double, -1, 1> args_adjoints_{0};

    template <typename VecT, typename... ArgsT>
    recursive_reducer(size_t num_vars_per_term, size_t num_vars_shared_terms,
                      double* sliced_partials, VecT&& vmapped, ArgsT&&... args)
        : num_vars_per_term_(num_vars_per_term),
          num_vars_shared_terms_(num_vars_shared_terms),
          sliced_partials_(sliced_partials),
          vmapped_(std::forward<VecT>(vmapped)),
          args_tuple_(std::forward<ArgsT>(args)...) {}

    /*
     * The split constructor of tbb::parallel_reduce (Imperative form): a new body for the right half of a
     *   split range, with its own sum_ and args_adjoints_ (both empty).
     */
    recursive_reducer(recursive_reducer& other, tbb::split)
        : num_vars_per_term_(other.num_vars_per_term_),
          num_vars_shared_terms_(other.num_vars_shared_terms_),
          sliced_partials_(other.sliced_partials_),
          vmapped_(other.vmapped_),
          args_tuple_(other.args_tuple_) {

      copy_the_shared_arguments_in_each_chunk_ = other.copy_the_shared_arguments_in_each_chunk_;
      nicostan_sliced_varis_ = other.nicostan_sliced_varis_;
      nicostan_nuisance_block_of_the_caller_ = other.nicostan_nuisance_block_of_the_caller_;
#if NICOSTAN_COUNT_THE_INSTRUCTIONS_OF_EACH_PART_OF_REDUCE_SUM
      NICOSTAN_ADD_TO_THE_PART(nicostan_part_body_made_for_a_split_of_the_range,
                               nicostan_readings_of_this_body_.when_made);
#endif

    }

#if NICOSTAN_COUNT_THE_INSTRUCTIONS_OF_EACH_PART_OF_REDUCE_SUM
    ~recursive_reducer() {

      nicostan_readings_of_this_body_.when_its_destruction_began = NICOSTAN_INSTRUCTIONS_NOW();

    }
#endif

    /**
     * One chunk (the terms of r): the value and the Jacobian of `ReduceFunction` over r, on a nested tape of
     *  this thread, added to sum_ and args_adjoints_ (and the partials of the chunk's own sliced terms).
     */
    inline void operator()(const tbb::blocked_range<size_t>& r) {

      if (r.empty()) {
        return;
      }
#if NICOSTAN_COUNT_THE_INSTRUCTIONS_OF_EACH_PART_OF_REDUCE_SUM
      nicostan_reading_at_the_end_of_a_chunk nicostan_end_of_this_chunk;
#endif
      unsigned long long nicostan_reading = NICOSTAN_INSTRUCTIONS_NOW();
      (void) nicostan_reading;

      if (args_adjoints_.size() == 0) {
        args_adjoints_ = Eigen::Matrix<double, -1, 1>::Zero(num_vars_shared_terms_);
      }

      if (copy_the_shared_arguments_in_each_chunk_) {
        // Initialize nested autodiff stack
        const nested_rev_autodiff begin_nest;
        nicostan_scope_of_a_chunk_of_the_nuisance_block nicostan_scope(
            nicostan_nuisance_block_of_the_caller_, static_cast<int>(r.begin()),
            static_cast<int>(r.end()) - 1);
        nicostan_scope_of_the_nuisance_derivative_collection nicostan_derivative_scope(
            nicostan_nuisance_derivatives_);
        nicostan_reading = NICOSTAN_ADD_TO_THE_PART(nicostan_part_nested_tape_started_and_sliced_terms_copied,
                                                    nicostan_reading);
        //// (copies of the shared arguments on the chunk's own nested tape: the same copies, of the same
        ////  values, with adjoints of zero, that Stan's reduce_sum makes once per body in its own scoped stack)
        typename scoped_args_tuple::args_tuple_t args_tuple_local = math::apply(
            [&](auto&&... args) {

              return typename scoped_args_tuple::args_tuple_t(deep_copy_vars(args)...);

            },
            args_tuple_);
        nicostan_reading = NICOSTAN_ADD_TO_THE_PART(
              nicostan_part_shared_arguments_copied_or_their_adjoints_zeroed, nicostan_reading);
        nicostan_reading = nicostan_run_the_chunk_on_its_nested_tape(r, args_tuple_local, nicostan_reading);
#if NICOSTAN_COUNT_THE_INSTRUCTIONS_OF_EACH_PART_OF_REDUCE_SUM
        nicostan_end_of_this_chunk.from = nicostan_reading;
#endif
        return;
      }

      if (!local_args_tuple_scope_) {
        local_args_tuple_scope_ = std::make_unique<scoped_args_tuple>();
      }
      if (!local_args_tuple_scope_->args_tuple_holder_) {
        // shared arguments need to be copied to reducer-specific
        // scope. In this case no need for zeroing adjoints, since the
        // fresh copy has all adjoints set to zero.
        local_args_tuple_scope_->stack_.execute([&]() {

          math::apply(
              [&](auto&&... args) {

                local_args_tuple_scope_->args_tuple_holder_ = std::make_unique<
                    typename scoped_args_tuple::args_tuple_t>(
                    deep_copy_vars(args)...);

              },
              args_tuple_);

        });
      } else {
        // set adjoints of shared arguments to zero
        local_args_tuple_scope_->stack_.execute([] { set_zero_all_adjoints(); });
      }
      nicostan_reading = NICOSTAN_ADD_TO_THE_PART(nicostan_part_shared_arguments_copied_or_their_adjoints_zeroed,
                                                  nicostan_reading);

      auto& args_tuple_local = *(local_args_tuple_scope_->args_tuple_holder_);

      // Initialize nested autodiff stack
      const nested_rev_autodiff begin_nest;
      nicostan_scope_of_a_chunk_of_the_nuisance_block nicostan_scope(
          nicostan_nuisance_block_of_the_caller_, static_cast<int>(r.begin()),
          static_cast<int>(r.end()) - 1);
      nicostan_scope_of_the_nuisance_derivative_collection nicostan_derivative_scope(
          nicostan_nuisance_derivatives_);
      nicostan_reading = NICOSTAN_ADD_TO_THE_PART(nicostan_part_nested_tape_started_and_sliced_terms_copied,
                                                  nicostan_reading);
      nicostan_reading = nicostan_run_the_chunk_on_its_nested_tape(r, args_tuple_local, nicostan_reading);
#if NICOSTAN_COUNT_THE_INSTRUCTIONS_OF_EACH_PART_OF_REDUCE_SUM
      nicostan_end_of_this_chunk.from = nicostan_reading;
#endif

    }

    /**
     * The chunk on its nested tape (already started, with the scope of the chunk of the nuisance block): its
     *  sliced terms copied, the function called and differentiated, and every adjoint accumulated. Returns the
     *  last reading of the instruction counter (0 when nothing is counted).
     */
    template <typename ArgsTupleLocal>
    inline unsigned long long nicostan_run_the_chunk_on_its_nested_tape(const tbb::blocked_range<size_t>& r,
                                                                        ArgsTupleLocal& args_tuple_local,
                                                                        unsigned long long nicostan_reading) {

      // Create nested autodiff copies of sliced argument that do not point
      //   back to main autodiff stack
      std::decay_t<Vec> local_sub_slice;
      local_sub_slice.reserve(r.size());
      vari** nicostan_next_vari = nicostan_sliced_varis_ + r.begin() * num_vars_per_term_;
      for (size_t i = r.begin(); i < r.end(); ++i) {
        nicostan_next_vari = save_varis(nicostan_next_vari, vmapped_[i]);
        local_sub_slice.emplace_back(deep_copy_vars(vmapped_[i]));
      }

      //// (the stream of this thread at this depth of nested reduce_sum calls, emptied if an earlier chunk
      ////  stopped with an exception before its messages were taken)
      nicostan_message_streams_of_this_thread_t& message_streams = nicostan_message_streams_of_this_thread;
      if (message_streams.streams.size() <= message_streams.depth) {
        message_streams.streams.emplace_back(new std::ostringstream());
      }
      std::ostringstream& messages_of_this_chunk = *(message_streams.streams[message_streams.depth]);
      if (messages_of_this_chunk.tellp() > 0) {
        messages_of_this_chunk.str(std::string());
        messages_of_this_chunk.clear();
      }
      struct one_depth_deeper_for_the_duration_of_the_call {
        nicostan_message_streams_of_this_thread_t& streams;
        explicit one_depth_deeper_for_the_duration_of_the_call(nicostan_message_streams_of_this_thread_t& s)
            : streams(s) {

          streams.depth += 1;

        }
        ~one_depth_deeper_for_the_duration_of_the_call() { streams.depth -= 1; }
      };
      nicostan_reading = NICOSTAN_ADD_TO_THE_PART(nicostan_part_nested_tape_started_and_sliced_terms_copied,
                                                  nicostan_reading);

      // Perform calculation
      var sub_sum_v = [&]() {

        const one_depth_deeper_for_the_duration_of_the_call one_depth_deeper(message_streams);
        return math::apply(
            [&](auto&&... args) {

              return ReduceFunction()(local_sub_slice, r.begin(), r.end() - 1,
                                      &messages_of_this_chunk, args...);

            },
            args_tuple_local);

      }();
      if (messages_of_this_chunk.tellp() > 0) {
        if (!msgs_) {
          msgs_ = std::make_unique<std::stringstream>();
        }
        *msgs_ << messages_of_this_chunk.str();
        messages_of_this_chunk.str(std::string());
        messages_of_this_chunk.clear();
      }
      nicostan_reading = NICOSTAN_ADD_TO_THE_PART(nicostan_part_chunk_function_forward_pass, nicostan_reading);

      // Compute Jacobian
      sub_sum_v.grad();
      nicostan_reading = NICOSTAN_ADD_TO_THE_PART(nicostan_part_chunk_function_reverse_pass, nicostan_reading);
      nicostan_add_the_adjoints_of_the_chunk_of_the_nuisance_block();
      nicostan_reading = NICOSTAN_ADD_TO_THE_PART(nicostan_part_nuisance_adjoints_added, nicostan_reading);

      // Accumulate value of reduce_sum
      sum_ += sub_sum_v.val();

      // Accumulate adjoints of sliced_arguments
      std::fill(sliced_partials_ + r.begin() * num_vars_per_term_,
                sliced_partials_ + r.end() * num_vars_per_term_, 0.0);
      accumulate_adjoints(sliced_partials_ + r.begin() * num_vars_per_term_,
                          std::move(local_sub_slice));

      // Accumulate adjoints of shared_arguments
      math::apply(
          [&](auto&&... args) {

            accumulate_adjoints(args_adjoints_.data(), args...);

          },
          args_tuple_local);
      nicostan_reading = NICOSTAN_ADD_TO_THE_PART(nicostan_part_sliced_partials_and_shared_adjoints_accumulated,
                                                  nicostan_reading);
      return nicostan_reading;

    }

    /**
     * Join reducers. Accumuluate the value (sum_) and Jacobian (arg_adoints_)
     *   of the other reducer.
     *
     * @param rhs Another partial sum
     */
    inline void join(const recursive_reducer& rhs) {

      unsigned long long nicostan_reading = NICOSTAN_INSTRUCTIONS_NOW();
      (void) nicostan_reading;
      sum_ += rhs.sum_;
      nicostan_nuisance_derivatives_.insert(nicostan_nuisance_derivatives_.end(),
          rhs.nicostan_nuisance_derivatives_.begin(), rhs.nicostan_nuisance_derivatives_.end());
      if (args_adjoints_.size() != 0 && rhs.args_adjoints_.size() != 0) {
        args_adjoints_ += rhs.args_adjoints_;
      } else if (args_adjoints_.size() == 0 && rhs.args_adjoints_.size() != 0) {
        args_adjoints_ = rhs.args_adjoints_;
      }
      if (rhs.msgs_) {
        if (!msgs_) {
          msgs_ = std::make_unique<std::stringstream>();
        }
        *msgs_ << rhs.msgs_->str();
      }
      NICOSTAN_ADD_TO_THE_PART(nicostan_part_join_of_two_bodies, nicostan_reading);

    }
  };

  /**
   * reduce_sum_static on a thread whose arena has 1 thread: the chunks, the bodies and the joins of TBB's
   *  parallel_deterministic_reduce with its simple_partitioner, in its own tree, on this thread (TBB 2020:
   *  simple_partition_type::execute halves the range while it has more than grainsize terms; at each halving
   *  the left half keeps the body and the right half gets a new body split from it, and the finish task does
   *  left.join(right) after both halves are done). The sum and the adjoints are added in exactly the same order.
   */
  static inline void nicostan_reduce_in_the_tree_of_tbbs_deterministic_reduce_on_this_thread(
      tbb::blocked_range<std::size_t>& range, recursive_reducer& body) {

    if (range.is_divisible()) {
      //// (the range keeps the first half; right_range is the second half)
      tbb::blocked_range<std::size_t> right_range(range, tbb::split());
      recursive_reducer right_body(body, tbb::split());
      nicostan_reduce_in_the_tree_of_tbbs_deterministic_reduce_on_this_thread(range, body);
      nicostan_reduce_in_the_tree_of_tbbs_deterministic_reduce_on_this_thread(right_range, right_body);
      body.join(right_body);
      return;
    }
    body(range);

  }

#else

  /**
   * This struct is used by the TBB to accumulate partial
   *  sums over consecutive ranges of the input. To distribute the workload,
   *  the TBB can split larger partial sums into smaller ones in which
   *  case the splitting copy constructor is used. It is designed to
   *  meet the Imperative form requirements of `tbb::parallel_reduce`.
   *
   * @note see link [here](https://tinyurl.com/vp7xw2t) for requirements.
   */
  struct recursive_reducer {
#if NICOSTAN_COUNT_THE_INSTRUCTIONS_OF_EACH_PART_OF_REDUCE_SUM
    nicostan_readings_of_a_body nicostan_readings_of_this_body_;
#endif
    const size_t num_vars_per_term_;
    const size_t num_vars_shared_terms_;  // Number of vars in shared arguments
    double* sliced_partials_;  // Points to adjoints of the partial calculations
    Vec vmapped_;
    std::stringstream msgs_;
    std::tuple<Args...> args_tuple_;
    scoped_args_tuple local_args_tuple_scope_;
    double sum_{0.0};
    vari** nicostan_sliced_varis_{nullptr};  // (NicoStan: each chunk writes its own terms' nodes here)
    nicostan_nuisance_block_t* nicostan_nuisance_block_of_the_caller_{nicostan_nuisance_block};
    std::vector<std::pair<int, double>> nicostan_nuisance_derivatives_;
    Eigen::Matrix<double, -1, 1> args_adjoints_{0};

    template <typename VecT, typename... ArgsT>
    recursive_reducer(size_t num_vars_per_term, size_t num_vars_shared_terms,
                      double* sliced_partials, VecT&& vmapped, ArgsT&&... args)
        : num_vars_per_term_(num_vars_per_term),
          num_vars_shared_terms_(num_vars_shared_terms),
          sliced_partials_(sliced_partials),
          vmapped_(std::forward<VecT>(vmapped)),
          args_tuple_(std::forward<ArgsT>(args)...) {}

    /*
     * This is the copy operator as required for tbb::parallel_reduce
     *   Imperative form. This requires sum_ and arg_adjoints_ be reset
     *   to zero since the newly created reducer is used to accumulate
     *   an independent partial sum.
     */
    recursive_reducer(recursive_reducer& other, tbb::split)
        : num_vars_per_term_(other.num_vars_per_term_),
          num_vars_shared_terms_(other.num_vars_shared_terms_),
          sliced_partials_(other.sliced_partials_),
          vmapped_(other.vmapped_),
          args_tuple_(other.args_tuple_) {

      nicostan_sliced_varis_ = other.nicostan_sliced_varis_;
      nicostan_nuisance_block_of_the_caller_ = other.nicostan_nuisance_block_of_the_caller_;
#if NICOSTAN_COUNT_THE_INSTRUCTIONS_OF_EACH_PART_OF_REDUCE_SUM
      NICOSTAN_ADD_TO_THE_PART(nicostan_part_body_made_for_a_split_of_the_range,
                               nicostan_readings_of_this_body_.when_made);
#endif

    }

#if NICOSTAN_COUNT_THE_INSTRUCTIONS_OF_EACH_PART_OF_REDUCE_SUM
    ~recursive_reducer() {

      nicostan_readings_of_this_body_.when_its_destruction_began = NICOSTAN_INSTRUCTIONS_NOW();

    }
#endif

    /**
     * Compute, using nested autodiff, the value and Jacobian of
     *  `ReduceFunction` called over the range defined by r and accumulate those
     *  in member variable sum_ (for the value) and args_adjoints_ (for the
     *  Jacobian). The nested autodiff uses deep copies of the involved operands
     *  ensuring that no side effects are implied to the adjoints of the input
     *  operands which reside potentially on a autodiff tape stored in a
     *  different thread other than the current thread of execution. This
     * function may be called multiple times per object instantiation (so the
     * sum_ and args_adjoints_ must be accumulated, not just assigned).
     *
     * @param r Range over which to compute reduce_sum
     */
    inline void operator()(const tbb::blocked_range<size_t>& r) {

      if (r.empty()) {
        return;
      }
#if NICOSTAN_COUNT_THE_INSTRUCTIONS_OF_EACH_PART_OF_REDUCE_SUM
      nicostan_reading_at_the_end_of_a_chunk nicostan_end_of_this_chunk;
#endif
      unsigned long long nicostan_reading = NICOSTAN_INSTRUCTIONS_NOW();
      (void) nicostan_reading;

      if (args_adjoints_.size() == 0) {
        args_adjoints_ = Eigen::Matrix<double, -1, 1>::Zero(num_vars_shared_terms_);
      }

      // Obtain reference to a local copy of all shared arguments that do
      // not point
      //   back to main autodiff stack

      if (!local_args_tuple_scope_.args_tuple_holder_) {
        // shared arguments need to be copied to reducer-specific
        // scope. In this case no need for zeroing adjoints, since the
        // fresh copy has all adjoints set to zero.
        local_args_tuple_scope_.stack_.execute([&]() {

          math::apply(
              [&](auto&&... args) {

                local_args_tuple_scope_.args_tuple_holder_ = std::make_unique<
                    typename scoped_args_tuple::args_tuple_t>(
                    deep_copy_vars(args)...);

              },
              args_tuple_);

        });
      } else {
        // set adjoints of shared arguments to zero
        local_args_tuple_scope_.stack_.execute([] { set_zero_all_adjoints(); });
      }
      nicostan_reading = NICOSTAN_ADD_TO_THE_PART(nicostan_part_shared_arguments_copied_or_their_adjoints_zeroed,
                                                  nicostan_reading);

      auto& args_tuple_local = *(local_args_tuple_scope_.args_tuple_holder_);

      // Initialize nested autodiff stack
      const nested_rev_autodiff begin_nest;
      nicostan_scope_of_a_chunk_of_the_nuisance_block nicostan_scope(
          nicostan_nuisance_block_of_the_caller_, static_cast<int>(r.begin()),
          static_cast<int>(r.end()) - 1);
      nicostan_scope_of_the_nuisance_derivative_collection nicostan_derivative_scope(
          nicostan_nuisance_derivatives_);

      // Create nested autodiff copies of sliced argument that do not point
      //   back to main autodiff stack
      std::decay_t<Vec> local_sub_slice;
      local_sub_slice.reserve(r.size());
      vari** nicostan_next_vari = nicostan_sliced_varis_ + r.begin() * num_vars_per_term_;
      for (size_t i = r.begin(); i < r.end(); ++i) {
        nicostan_next_vari = save_varis(nicostan_next_vari, vmapped_[i]);
        local_sub_slice.emplace_back(deep_copy_vars(vmapped_[i]));
      }
      nicostan_reading = NICOSTAN_ADD_TO_THE_PART(nicostan_part_nested_tape_started_and_sliced_terms_copied,
                                                  nicostan_reading);

      // Perform calculation
      var sub_sum_v = math::apply(
          [&](auto&&... args) {

            return ReduceFunction()(local_sub_slice, r.begin(), r.end() - 1,
                                    &msgs_, args...);

          },
          args_tuple_local);
      nicostan_reading = NICOSTAN_ADD_TO_THE_PART(nicostan_part_chunk_function_forward_pass, nicostan_reading);

      // Compute Jacobian
      sub_sum_v.grad();
      nicostan_reading = NICOSTAN_ADD_TO_THE_PART(nicostan_part_chunk_function_reverse_pass, nicostan_reading);
      nicostan_add_the_adjoints_of_the_chunk_of_the_nuisance_block();
      nicostan_reading = NICOSTAN_ADD_TO_THE_PART(nicostan_part_nuisance_adjoints_added, nicostan_reading);

      // Accumulate value of reduce_sum
      sum_ += sub_sum_v.val();

      // Accumulate adjoints of sliced_arguments
      std::fill(sliced_partials_ + r.begin() * num_vars_per_term_,
                sliced_partials_ + r.end() * num_vars_per_term_, 0.0);
      accumulate_adjoints(sliced_partials_ + r.begin() * num_vars_per_term_,
                          std::move(local_sub_slice));

      // Accumulate adjoints of shared_arguments
      math::apply(
          [&](auto&&... args) {

            accumulate_adjoints(args_adjoints_.data(), args...);

          },
          args_tuple_local);
      nicostan_reading = NICOSTAN_ADD_TO_THE_PART(nicostan_part_sliced_partials_and_shared_adjoints_accumulated,
                                                  nicostan_reading);
#if NICOSTAN_COUNT_THE_INSTRUCTIONS_OF_EACH_PART_OF_REDUCE_SUM
      nicostan_end_of_this_chunk.from = nicostan_reading;
#endif

    }

    /**
     * Join reducers. Accumuluate the value (sum_) and Jacobian (arg_adoints_)
     *   of the other reducer.
     *
     * @param rhs Another partial sum
     */
    inline void join(const recursive_reducer& rhs) {

      unsigned long long nicostan_reading = NICOSTAN_INSTRUCTIONS_NOW();
      (void) nicostan_reading;
      sum_ += rhs.sum_;
      nicostan_nuisance_derivatives_.insert(nicostan_nuisance_derivatives_.end(),
          rhs.nicostan_nuisance_derivatives_.begin(), rhs.nicostan_nuisance_derivatives_.end());
      if (args_adjoints_.size() != 0 && rhs.args_adjoints_.size() != 0) {
        args_adjoints_ += rhs.args_adjoints_;
      } else if (args_adjoints_.size() == 0 && rhs.args_adjoints_.size() != 0) {
        args_adjoints_ = rhs.args_adjoints_;
      }
      msgs_ << rhs.msgs_.str();
      NICOSTAN_ADD_TO_THE_PART(nicostan_part_join_of_two_bodies, nicostan_reading);

    }
  };

#endif

  /**
   * Call an instance of the function `ReduceFunction` on every element
   *   of an input sequence and sum these terms.
   *
   * This specialization is parallelized using tbb and works for reverse
   *   mode autodiff.
   *
   * ReduceFunction must define an operator() with the same signature as:
   *   var f(Vec&& vmapped_subset, int start, int end, std::ostream* msgs,
   * Args&&... args)
   *
   * `ReduceFunction` must be default constructible without any arguments
   *
   * Each call to `ReduceFunction` is responsible for computing the
   *   start through end (inclusive) terms of the overall sum. All args are
   * passed from this function through to the `ReduceFunction` instances.
   *   However, only the start through end (inclusive) elements of the vmapped
   * argument are passed to the `ReduceFunction` instances (as the
   * `vmapped_subset` argument).
   *
   * This function distributes computation of the desired sum and the Jacobian
   * of that sum over multiple threads by coordinating calls to `ReduceFunction`
   * instances. Results are stored as precomputed varis in the autodiff tree.
   *
   * If auto partitioning is true, break work into pieces automatically,
   *  taking grainsize as a recommended work size. The partitioning is
   *  not deterministic nor is the order guaranteed in which partial
   *  sums are accumulated. Due to floating point imprecisions this will likely
   *  lead to slight differences in the accumulated results between
   *  multiple runs. If false, break work deterministically into pieces smaller
   *  than or equal to grainsize and accumulate all the partial sums
   *  in the same order. This still may not achieve bitwise reproducibility.
   *
   * @param vmapped Vector containing one element per term of sum
   * @param auto_partitioning Work partitioning style
   * @param grainsize Suggested grainsize for tbb
   * @param[in, out] msgs The print stream for warning messages
   * @param args Shared arguments used in every sum term
   * @return Summation of all terms
   */
  inline var operator()(Vec&& vmapped, bool auto_partitioning, int grainsize,
                        std::ostream* msgs, Args&&... args) const {

    if (vmapped.empty()) {
      return var(0.0);
    }
    unsigned long long nicostan_reading_at_the_start_of_the_call = NICOSTAN_INSTRUCTIONS_NOW();
    (void) nicostan_reading_at_the_start_of_the_call;

    const std::size_t num_terms = vmapped.size();
    const std::size_t num_vars_per_term = count_vars(vmapped[0]);
    const std::size_t num_vars_sliced_terms = num_terms * num_vars_per_term;
    const std::size_t num_vars_shared_terms = count_vars(args...);

    vari** varis = ChainableStack::instance_->memalloc_.alloc_array<vari*>(
        num_vars_sliced_terms + num_vars_shared_terms);
    double* partials = ChainableStack::instance_->memalloc_.alloc_array<double>(
        num_vars_sliced_terms + num_vars_shared_terms);

    //// (NicoStan lean slicing: each chunk writes the pointers of its own terms)
    save_varis(varis + num_vars_sliced_terms, args...);

    //// (NicoStan lean slicing: each chunk zeroes its own partials)
    ////
    ////

    //// (the worker in its own block, so that its destruction is inside the instructions of the call)
    var nicostan_sum_of_the_chunks;
    {
      recursive_reducer worker(num_vars_per_term, num_vars_shared_terms, partials,
                               std::forward<Vec>(vmapped),
                               std::forward<Args>(args)...);
      worker.nicostan_sliced_varis_ = varis;

#if NICOSTAN_REDUCE_SUM_WITH_LESS_FIXED_WORK_PER_CHUNK
      worker.copy_the_shared_arguments_in_each_chunk_ = !auto_partitioning;
      if (!auto_partitioning && tbb::this_task_arena::max_concurrency() == 1) {
        tbb::blocked_range<std::size_t> range(0, num_terms, grainsize);
        nicostan_reduce_in_the_tree_of_tbbs_deterministic_reduce_on_this_thread(range, worker);
      } else {
#endif
      // we must use task isolation as described here:
      // https://software.intel.com/content/www/us/en/develop/documentation/tbb-documentation/top/intel-threading-building-blocks-developer-guide/task-isolation.html
      // this is to ensure that the thread local AD tape ressource is
      // not being modified from a different task which may happen
      // whenever this function is being used itself in a parallel
      // context (like running multiple chains for Stan)
      tbb::this_task_arena::isolate([&] {
        if (auto_partitioning) {
          tbb::parallel_reduce(
              tbb::blocked_range<std::size_t>(0, num_terms, grainsize), worker);
        } else {
          tbb::simple_partitioner partitioner;
          tbb::parallel_deterministic_reduce(
              tbb::blocked_range<std::size_t>(0, num_terms, grainsize), worker,
              partitioner);
        }
      });
#if NICOSTAN_REDUCE_SUM_WITH_LESS_FIXED_WORK_PER_CHUNK
      }
#endif

      for (size_t i = 0; i < num_vars_shared_terms; ++i) {
        partials[num_vars_sliced_terms + i] = worker.args_adjoints_.coeff(i);
      }

#if NICOSTAN_REDUCE_SUM_WITH_LESS_FIXED_WORK_PER_CHUNK
      if (msgs && worker.msgs_) {
        *msgs << worker.msgs_->str();
      }
#else
      if (msgs) {
        *msgs << worker.msgs_.str();
      }
#endif

      nicostan_sum_of_the_chunks = var(new nicostan_precomputed_gradients_vari_of_the_nuisance_block(
          worker.sum_, num_vars_sliced_terms + num_vars_shared_terms, varis,
          partials, worker.nicostan_nuisance_block_of_the_caller_, worker.nicostan_nuisance_derivatives_));
    }
    NICOSTAN_ADD_TO_THE_PART(nicostan_part_whole_reduce_sum_call, nicostan_reading_at_the_start_of_the_call);
    return nicostan_sum_of_the_chunks;

  }
};
}  // namespace internal

}  // namespace math
}  // namespace stan

#endif























