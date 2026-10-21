// NicoStan patch of stan/math/rev/functor/reduce_sum.hpp (applied by R_fn_bridge_memory_patch_reduce_sum to the
// private BridgeStan runtime, next to the page-release patch): the serial bookkeeping of reduce_sum's reverse
// with 45 threads per chain both the Stan math and the AVX libraries sit on the same floor per gradient, set by
// the single-threaded work around the parallel section; two of its parts are reduce_sum's own:
//   (1) save_varis over every sliced var before the parallel section (one pointer write per var);
//   (2) precomputed_gradients_vari::chain(): one adjoint add per sliced var in the reverse pass.
// Both are done here in parallel over blocks of terms / vars (grainsize 2048), inside this_task_arena::isolate,
// as the parallel section itself. The adjoint adds are parallel ONLY when the saved vari addresses of the sliced
// part are strictly increasing (checked in O(n) once per call), which proves the sliced vars are distinct (no
// two threads ever add to the same adjoint); any repeated reference falls back to the serial loop, so the
// gradient is the same for every model. The shared arguments' part (num_vars_shared_terms) stays serial.
// The same runtime for every math backend (Stan math, AVX2, AVX-512); all of it inside the timed gradient.
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/task_arena.h>

namespace stan {
namespace math {
namespace internal {

class nicostan_parallel_precomputed_gradients_vari : public vari {
 protected:
  const size_t num_sliced_;
  const size_t size_;
  vari** varis_;
  double* gradients_;
  const bool sliced_varis_distinct_;

 public:
  nicostan_parallel_precomputed_gradients_vari(double val, size_t num_sliced, size_t size, vari** varis,
                                               double* gradients)
      : vari(val), num_sliced_(num_sliced), size_(size), varis_(varis), gradients_(gradients),
        sliced_varis_distinct_(varis_strictly_increasing(varis, num_sliced)) {}

  static bool varis_strictly_increasing(vari** varis, size_t n) {
    for (size_t i = 1; i < n; ++i) {
      if (!(varis[i - 1] < varis[i])) return false;
    }
    return true;
  }

  void chain() {
    const double adj = adj_;
    if (sliced_varis_distinct_ && num_sliced_ >= 8192) {
      tbb::this_task_arena::isolate([&] {
        tbb::parallel_for(tbb::blocked_range<size_t>(0, num_sliced_, 2048),
                          [&](const tbb::blocked_range<size_t>& r) {
                            for (size_t i = r.begin(); i < r.end(); ++i) {
                              varis_[i]->adj_ += adj * gradients_[i];
                            }
                          });
      });
    } else {
      for (size_t i = 0; i < num_sliced_; ++i) varis_[i]->adj_ += adj * gradients_[i];
    }
    for (size_t i = num_sliced_; i < size_; ++i) varis_[i]->adj_ += adj * gradients_[i];
  }
};

// save_varis of the sliced terms in parallel: term i's varis are written at varis + i * num_vars_per_term, so
// the blocks never overlap; below 8192 terms the serial save_varis is used (the parallel start costs more)
template <typename Vec>
inline void nicostan_save_varis_of_sliced_terms(vari** varis, const Vec& vmapped, size_t num_vars_per_term) {
  const size_t num_terms = vmapped.size();
  if (num_terms < 8192 || num_vars_per_term == 0) {
    save_varis(varis, vmapped);
    return;
  }
  tbb::this_task_arena::isolate([&] {
    tbb::parallel_for(tbb::blocked_range<size_t>(0, num_terms, 2048),
                      [&](const tbb::blocked_range<size_t>& r) {
                        for (size_t i = r.begin(); i < r.end(); ++i) {
                          save_varis(varis + i * num_vars_per_term, vmapped[i]);
                        }
                      });
  });
}

}  // namespace internal
}  // namespace math
}  // namespace stan

// ---- (3) the chunks of reduce_sum on the chain's OpenMP team instead of TBB's parallel reduce (compiled in with
//      -DNICOSTAN_REDUCE_SUM_OPENMP and -fopenmp; the model library links the libomp NicoStan itself uses):
//      the pieces are EXACTLY the leaves TBB's parallel_deterministic_reduce with the simple_partitioner makes
//      (recursive halving of [0, num_terms) at the middle until a piece has at most grainsize terms, in order),
//      so every model sees the same chunk boundaries as with TBB; the pieces are given to the threads up front
//      sleep / wake cycles per gradient call with 45 TBB threads, 49 with 8), and the team spins between the
//      gradient calls of a chain (KMP_BLOCKTIME, as NicoStan's own models). One thread, or a call from inside
//      another OpenMP region (the Hessian adaptation), runs the pieces serially on the calling thread, which
//      keeps today's chunking at 1 thread per chain. Each thread runs its own copy of Stan's recursive_reducer
//      (the split constructor) on its own thread-local autodiff stack, exactly as a TBB worker does; the copies
//      are joined in thread order (deterministic for a given thread count). An exception is caught INSIDE the
//      iteration (or the parallel block) that threw it, as the OpenMP specification requires, recorded, and
//      rethrown on the calling thread after the region; once one is recorded the remaining iterations do nothing.
#if defined(NICOSTAN_REDUCE_SUM_OPENMP)
#include <omp.h>
#include <atomic>
#include <exception>
#include <algorithm>
#include <memory>
#include <utility>
#include <vector>
namespace stan {
namespace math {
namespace internal {

// the leaves of TBB's blocked_range<size_t>(0, num_terms, grainsize) under the simple_partitioner, in order
inline void nicostan_tbb_leaves(std::size_t begin, std::size_t end, std::size_t grainsize,
                                std::vector<std::pair<std::size_t, std::size_t>>& leaves) {
  if (end - begin > grainsize) {
    const std::size_t middle = begin + (end - begin) / 2;
    nicostan_tbb_leaves(begin, middle, grainsize, leaves);
    nicostan_tbb_leaves(middle, end, grainsize, leaves);
  } else {
    leaves.emplace_back(begin, end);
  }
}

template <typename Reducer>
inline void nicostan_reduce_sum_openmp(Reducer& worker, std::size_t num_terms, std::size_t grainsize) {
  const std::size_t g = std::max<std::size_t>(1, grainsize);
  std::vector<std::pair<std::size_t, std::size_t>> leaves;
  leaves.reserve(num_terms / g + 2);
  nicostan_tbb_leaves(0, num_terms, g, leaves);
  const std::size_t n_leaves = leaves.size();
  int n_threads = omp_in_parallel() ? 1 : omp_get_max_threads();
  if (n_threads <= 1 || n_leaves <= 1) {
    for (std::size_t c = 0; c < n_leaves; ++c) {
      worker(tbb::blocked_range<std::size_t>(leaves[c].first, leaves[c].second));
    }
    return;
  }
  if (static_cast<std::size_t>(n_threads) > n_leaves) n_threads = static_cast<int>(n_leaves);
  std::vector<std::unique_ptr<Reducer>> reducers(n_threads);
  std::vector<std::exception_ptr> errors(n_threads);
  std::atomic<bool> failed(false);
#pragma omp parallel num_threads(n_threads)
  {
    // each OpenMP worker needs its own thread-local autodiff stack (a TBB worker gets it from Stan's scheduler
    // observer): made ONCE per thread and kept for the thread's lifetime (libomp keeps its workers), so the
    // worker's arena persists across gradient calls as a TBB worker's does (no rebuild, no page faults per
    // call); on the calling thread, which already has its stack, the constructor is a no-op (own_instance_
    // false) and the caller's stack is untouched
    static thread_local std::unique_ptr<stan::math::ChainableStack> ad_tape_of_this_thread;
    if (!ad_tape_of_this_thread) ad_tape_of_this_thread.reset(new stan::math::ChainableStack());
    const int t = omp_get_thread_num();
    bool thread_ready = false;
    try {
      reducers[t].reset(new Reducer(worker, tbb::split()));
      thread_ready = true;
    } catch (...) {
      errors[t] = std::current_exception();
      failed.store(true);
    }
#pragma omp for schedule(static)
    for (std::size_t c = 0; c < n_leaves; ++c) {
      if (!thread_ready || failed.load()) continue;
      try {
        (*reducers[t])(tbb::blocked_range<std::size_t>(leaves[c].first, leaves[c].second));
      } catch (...) {
        errors[t] = std::current_exception();
        failed.store(true);
      }
    }
  }
  for (auto& err : errors) {
    if (err) std::rethrow_exception(err);
  }
  for (auto& r : reducers) {
    if (r) worker.join(*r);
  }
}
}  // namespace internal
}  // namespace math
}  // namespace stan
#endif
