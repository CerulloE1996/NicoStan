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
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace stan {
namespace math {
namespace internal {

//// the (value, node) of every value read on the main tape, kept as runs: one entry for each run of values
//// read one after another whose nodes were made at addresses one node apart (as nodes made one after another
//// are), never one entry per value; iterated as the (value, node) of every value, in the order they were read.
//// Nodes of the model's own tape are marked shared: a repeated read of their value returns the same node.
//// Values read out of order (more runs than number_of_runs_before_the_node_of_every_value): from then on one
//// (value, node) pair per value, as before the runs, iterated after the runs
class nicostan_nodes_on_the_main_tape_t {
 public:
  //// (the address of a run's node k: the address of its first node plus k node sizes, in integers; a run grows
  ////  only when the new node's own address is that address)
  static_assert(std::is_same<vari, vari_value<double>>::value, "the nodes of the values are vari_value<double>");
  static constexpr std::uintptr_t node_stride = sizeof(vari_value<double>);

  struct run_t {
    std::uintptr_t first_address;
    int first_value;
    int n;
    bool shared;
  };

  //// (indices into the runs and the pairs, the runs first)
  class iterator {
    const nicostan_nodes_on_the_main_tape_t* nodes_;
    std::size_t number_of_runs_;
    std::size_t run_;
    int offset_;
    std::size_t pair_;

   public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = std::pair<int, vari*>;
    using difference_type = std::ptrdiff_t;
    using pointer = void;
    using reference = value_type;
    iterator(const nicostan_nodes_on_the_main_tape_t* nodes, std::size_t run, int offset, std::size_t pair)
        : nodes_(nodes), number_of_runs_(nodes->runs_.size()), run_(run), offset_(offset), pair_(pair) {}
    value_type operator*() const {
      if (run_ < number_of_runs_) {
        const run_t& run = nodes_->runs_[run_];
        return value_type(run.first_value + offset_,
                          reinterpret_cast<vari*>(run.first_address +
                                                  static_cast<std::uintptr_t>(offset_) * node_stride));
      }
      return nodes_->pairs_[pair_];
    }
    iterator& operator++() {
      if (run_ < number_of_runs_) {
        if (++offset_ == nodes_->runs_[run_].n) {
          ++run_;
          offset_ = 0;
        }
      } else {
        ++pair_;
      }
      return *this;
    }
    bool operator==(const iterator& other) const {
      return run_ == other.run_ && offset_ == other.offset_ && pair_ == other.pair_;
    }
    bool operator!=(const iterator& other) const { return !(*this == other); }
  };

  iterator begin() const { return iterator(this, 0, 0, 0); }
  iterator end() const { return iterator(this, runs_.size(), 0, pairs_.size()); }
  bool empty() const { return runs_.empty() && pairs_.empty(); }
  std::size_t number_of_runs() const { return runs_.size(); }
  std::size_t number_of_pairs() const { return pairs_.size(); }

  //// above this many runs the values are read out of order: from then on one (value, node) pair per value, and
  //// a node pointer per value (nicostan_nuisance_block_t::nodes_by_value_on_the_main_tape), as before the runs
  static constexpr std::size_t number_of_runs_before_the_node_of_every_value = 32;
  bool values_read_out_of_order() const { return values_read_out_of_order_; }
  //// (the node pointers are written chunk by chunk)
  static constexpr int values_per_chunk_of_the_node_of_every_value = 512;
  //// the node of every shared value of the runs at its value, nullptr for the others
  void fill_the_node_of_every_value(std::vector<vari*>& node_of_every_value, int size) const {
    const int chunk = values_per_chunk_of_the_node_of_every_value;
    node_of_every_value.clear();
    node_of_every_value.reserve(size);
    for (int first = 0; first < size; first += chunk) {
      node_of_every_value.insert(node_of_every_value.end(), std::min(chunk, size - first), nullptr);
    }
    for (const run_t& run : runs_) {
      if (!run.shared) continue;
      for (int first = 0; first < run.n; first += chunk) {
        const int length = std::min(chunk, run.n - first);
        const std::size_t first_index = static_cast<std::size_t>(run.first_value) + first;
        const std::uintptr_t first_address = run.first_address + static_cast<std::uintptr_t>(first) * node_stride;
        for (int k = 0; k < length; ++k) {
          node_of_every_value[first_index + k] =
                reinterpret_cast<vari*>(first_address + static_cast<std::uintptr_t>(k) * node_stride);
        }
      }
    }
  }

  //// a node of a nested tape: listed, never shared
  void emplace_back(int value, vari* node) { add(value, node, false); }
  //// a node of the model's own tape: listed and shared
  void emplace_back_shared(int value, vari* node) { add(value, node, true); }

  //// the shared node of a value in the runs, or nullptr (the run of the last value found, then the next run,
  //// first: a repeated read goes through the values in the same order)
  vari* shared_node_of_value(int value) {
    if (value < first_shared_value_ || value >= end_of_the_shared_values_) return nullptr;
    const std::size_t n_runs = runs_.size();
    for (std::size_t k = run_of_the_last_value_found_; k < n_runs && k < run_of_the_last_value_found_ + 2;
         ++k) {
      const run_t& run = runs_[k];
      if (run.shared && value >= run.first_value && value < run.first_value + run.n) {
        run_of_the_last_value_found_ = k;
        return node_of_run(run, value);
      }
    }
    for (std::size_t k = 0; k < n_runs; ++k) {
      const run_t& run = runs_[k];
      if (run.shared && value >= run.first_value && value < run.first_value + run.n) {
        run_of_the_last_value_found_ = k;
        return node_of_run(run, value);
      }
    }
    return nullptr;
  }

  void clear() {
    runs_.clear();
    pairs_.clear();
    values_read_out_of_order_ = false;
    run_of_the_last_value_found_ = 0;
    first_shared_value_ = std::numeric_limits<int>::max();
    end_of_the_shared_values_ = 0;
  }

 private:
  std::vector<run_t> runs_;
  std::vector<std::pair<int, vari*>> pairs_;
  bool values_read_out_of_order_ = false;
  std::size_t run_of_the_last_value_found_ = 0;
  int first_shared_value_ = std::numeric_limits<int>::max();
  int end_of_the_shared_values_ = 0;

  static vari* node_of_run(const run_t& run, int value) {
    return reinterpret_cast<vari*>(run.first_address +
                                   static_cast<std::uintptr_t>(value - run.first_value) * node_stride);
  }

  void add(int value, vari* node, bool shared) {
    if (values_read_out_of_order_) {
      pairs_.emplace_back(value, node);
      return;
    }
    if (shared) {
      if (value < first_shared_value_) first_shared_value_ = value;
      if (value + 1 > end_of_the_shared_values_) end_of_the_shared_values_ = value + 1;
    }
    const std::uintptr_t address = reinterpret_cast<std::uintptr_t>(node);
    if (!runs_.empty()) {
      run_t& last = runs_.back();
      if (last.shared == shared && value == last.first_value + last.n &&
          address == last.first_address + static_cast<std::uintptr_t>(last.n) * node_stride) {
        ++last.n;
        return;
      }
    }
    runs_.push_back(run_t{address, value, 1, shared});
    if (runs_.size() > number_of_runs_before_the_node_of_every_value) values_read_out_of_order_ = true;
  }
};

struct nicostan_nuisance_block_t {
  const double* values = nullptr;   // NicoStan's array of the nuisance values (0-based)
  double* gradient = nullptr;       // NicoStan's array of their gradient, zeroed before the gradient call
  int size = 0;
  // std::vector<std::pair<int, vari*>> nodes_on_the_main_tape;   // (unit, node) of values read on the main tape
  nicostan_nodes_on_the_main_tape_t nodes_on_the_main_tape;        // (unit, node) of values read on the main tape
  std::vector<std::pair<int, double>> adjoints_from_other_chunks;  // (unit, adjoint) of values read by the chunk
  std::mutex mutex_of_the_adjoints_from_other_chunks;              //  of another unit
  std::atomic<bool> read_inside_a_reduce_sum{false};
  //// Repeated main-tape reads share one leaf per nuisance value within this gradient only.
  //// No storage is allocated for a model which reads its nuisance values only inside chunks.
  //// (filled only once the values are read out of order, i.e. in more runs than
  ////  nicostan_nodes_on_the_main_tape_t::number_of_runs_before_the_node_of_every_value; until then the shared
  ////  runs of nodes_on_the_main_tape find a value's node; cleared by BridgeStan's gradient)
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
};
inline thread_local nicostan_chunk_of_the_nuisance_block_t nicostan_chunk_of_the_nuisance_block;

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






















