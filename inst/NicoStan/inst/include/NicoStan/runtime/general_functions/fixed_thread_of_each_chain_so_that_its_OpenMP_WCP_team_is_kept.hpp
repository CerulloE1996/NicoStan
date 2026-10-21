#pragma once

#include <sched.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <exception>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <stan/math/rev/core/chainablestack.hpp>

////
//// ---- (built-in models: LC-MVP, MVP, LC-MVOP, MVOP) each chain run on a fixed thread of its own, rather than
////      on whichever of RcppParallel's (TBB's) threads is free. The thread that runs a chain is the root of the
////      chain's within-chain parallel (WCP) OpenMP team, and libomp keeps a root's team (its "hot team") for as
////      long as that thread lives. With TBB, the chains of successive burn-in iterations land on different
////      threads, each of which then builds a new team of n_threads_WCP - 1 OpenMP threads, inside the timed
////      iterations, and keeps it: on the local-HPC (180 TBB threads; sampling layouts, then 4 x 45 burn-in) 1552
////      threads were alive, and libomp 14 then grew its table of threads and crashed (a segfault in the start-up
////      of a new OpenMP thread, which reads the old table of threads whilst another root frees it). On a fixed
////      thread, each chain's team is built once and reused, and it is given back to libomp's pool when the
////      thread ends. The results are the same: each chain's work does not depend on the thread that runs it.
////
////      The number of threads is RcppParallel's (RCPP_PARALLEL_NUM_THREADS, set by
////      RcppParallel::setThreadOptions(); the CPUs of the process when it is not set), at most one per chain;
////      with more chains than threads, thread k runs chain k first, then each next chain not yet started.
////
inline bool fn_is_built_in_model_whose_WCP_is_OpenMP(const std::string &Model_type) {

        return (Model_type == "LC_MVP") || (Model_type == "MVP") ||
               (Model_type == "LC_MVOP") || (Model_type == "MVOP");

}

//// (the switch: NICOSTAN_RUN_EACH_CHAIN_OF_A_BUILT_IN_MODEL_ON_A_FIXED_THREAD_OF_ITS_OWN = 0 runs the chains
////  of a built-in model on RcppParallel's threads, as before; unset or 1, on fixed threads of their own)
inline bool fn_run_each_chain_of_this_model_on_a_fixed_thread_of_its_own(const std::string &Model_type) {

        if (!fn_is_built_in_model_whose_WCP_is_OpenMP(Model_type)) return false;
        const char *value_of_the_switch =
              std::getenv("NICOSTAN_RUN_EACH_CHAIN_OF_A_BUILT_IN_MODEL_ON_A_FIXED_THREAD_OF_ITS_OWN");
        if (value_of_the_switch == nullptr) return true;
        return std::atoi(value_of_the_switch) != 0;

}

inline int fn_number_of_threads_of_RcppParallel() {

        int n_threads = -1;
        const char *value_of_RcppParallel_setting = std::getenv("RCPP_PARALLEL_NUM_THREADS");
        if (value_of_RcppParallel_setting != nullptr) n_threads = std::atoi(value_of_RcppParallel_setting);
        if (n_threads <= 0) {
              //// (as TBB: the CPUs the process may run on)
              cpu_set_t CPUs_of_the_process;
              CPU_ZERO(&CPUs_of_the_process);
              if (sched_getaffinity(0, sizeof(CPUs_of_the_process), &CPUs_of_the_process) == 0) {
                    n_threads = CPU_COUNT(&CPUs_of_the_process);
              }
        }
        if (n_threads <= 0) n_threads = static_cast<int>(std::thread::hardware_concurrency());
        if (n_threads <= 0) n_threads = 1;
        return n_threads;

}

class fixed_threads_of_the_chains {

public:

      fixed_threads_of_the_chains(const int n_chains_, const int n_threads_)
        : n_chains(n_chains_), n_threads(std::max(1, std::min(n_chains_, n_threads_))),
          exception_of_each_chain(n_chains_) {

            threads.reserve(n_threads);
            try {
                  for (int thread_index = 0; thread_index < n_threads; thread_index++) {
                        threads.emplace_back(&fixed_threads_of_the_chains::fn_loop_of_thread, this, thread_index);
                  }
            } catch (...) {
                  fn_end_the_threads();
                  throw;
            }

      }

      ~fixed_threads_of_the_chains() {

            fn_end_the_threads();

      }

      //// (held by the caller whilst the chains run on these threads, when the threads are kept for the whole
      ////  process: one caller at a time)
      std::mutex mutex_held_whilst_the_chains_run;

      fixed_threads_of_the_chains(const fixed_threads_of_the_chains &) = delete;
      fixed_threads_of_the_chains &operator=(const fixed_threads_of_the_chains &) = delete;

      //// every chain run once (run_chain(chain_index)), on the fixed threads; the first exception, in chain
      //// order, is thrown again on the calling thread, as RcppParallel's parallelFor does:
      void fn_run_every_chain(const std::function<void(int)> &run_chain_) {

            {
                  std::lock_guard<std::mutex> lock(mutex);
                  run_chain = &run_chain_;
                  n_threads_done = 0;
                  next_chain_not_yet_started.store(n_threads);
                  std::fill(exception_of_each_chain.begin(), exception_of_each_chain.end(), nullptr);
                  round_index++;
            }
            start_of_round.notify_all();
            {
                  std::unique_lock<std::mutex> lock(mutex);
                  end_of_round.wait(lock, [this]() { return n_threads_done == n_threads; });
                  run_chain = nullptr;
            }
            for (int chain_index = 0; chain_index < n_chains; chain_index++) {
                  if (exception_of_each_chain[chain_index]) {
                        std::rethrow_exception(exception_of_each_chain[chain_index]);
                  }
            }

      }

private:

      const int n_chains;
      const int n_threads;
      std::vector<std::thread> threads;
      std::mutex mutex;
      std::condition_variable start_of_round;
      std::condition_variable end_of_round;
      long long round_index = 0;
      int n_threads_done = 0;
      bool ending = false;
      const std::function<void(int)> *run_chain = nullptr;
      std::atomic<int> next_chain_not_yet_started{0};
      std::vector<std::exception_ptr> exception_of_each_chain;

      void fn_loop_of_thread(const int thread_index) {

            //// (Stan's autodiff tape of this thread, as Stan makes one for each TBB thread: the multi-attempt
            ////  fallback to autodiff may run on it)
            stan::math::ChainableStack autodiff_tape_of_this_thread;
            long long round_index_seen = 0;
            while (true) {
                  const std::function<void(int)> *run_chain_of_round = nullptr;
                  {
                        std::unique_lock<std::mutex> lock(mutex);
                        start_of_round.wait(lock, [&]() { return ending || round_index != round_index_seen; });
                        if (ending) return;
                        round_index_seen = round_index;
                        run_chain_of_round = run_chain;
                  }
                  int chain_index = thread_index;
                  while (chain_index < n_chains) {
                        try {
                              (*run_chain_of_round)(chain_index);
                        } catch (...) {
                              exception_of_each_chain[chain_index] = std::current_exception();
                        }
                        chain_index = next_chain_not_yet_started.fetch_add(1);
                  }
                  {
                        std::lock_guard<std::mutex> lock(mutex);
                        n_threads_done++;
                        if (n_threads_done == n_threads) end_of_round.notify_one();
                  }
            }

      }

      void fn_end_the_threads() {

            {
                  std::lock_guard<std::mutex> lock(mutex);
                  ending = true;
            }
            start_of_round.notify_all();
            for (std::thread &thread : threads) {
                  if (thread.joinable()) thread.join();
            }
            threads.clear();

      }

};

////
//// ---- (sampling) every chain of one call on fixed threads made for the call, ended at its end (their teams
////      then go back to libomp's pool, for the next call):
////
template <typename Worker>
inline void fn_run_each_chain_on_a_fixed_thread_for_this_call(const int n_chains, Worker &worker) {

        fixed_threads_of_the_chains threads_of_the_chains(n_chains, fn_number_of_threads_of_RcppParallel());
        threads_of_the_chains.fn_run_every_chain([&worker](const int chain_index) {

              worker(static_cast<std::size_t>(chain_index), static_cast<std::size_t>(chain_index) + 1);

        });

}

////
//// ---- (burn-in) the fixed threads of each persistent burn-in worker: made at its first iteration, kept for
////      its later iterations, ended when the worker is deleted. (The table is never destroyed: at the end of
////      the R process its threads simply stop with it.)
////
inline std::mutex &fn_mutex_of_the_fixed_threads_of_each_burnin_worker() {

        static std::mutex *mutex_of_the_table = new std::mutex();
        return *mutex_of_the_table;

}

inline std::map<const void *, std::unique_ptr<fixed_threads_of_the_chains>> &
fn_fixed_threads_of_each_burnin_worker() {

        static auto *table = new std::map<const void *, std::unique_ptr<fixed_threads_of_the_chains>>();
        return *table;

}

inline fixed_threads_of_the_chains &fn_fixed_threads_of_the_chains_of_burnin_worker(const void *burnin_worker,
                                                                                    const int n_chains) {

        std::lock_guard<std::mutex> lock(fn_mutex_of_the_fixed_threads_of_each_burnin_worker());
        std::unique_ptr<fixed_threads_of_the_chains> &threads_of_the_chains =
              fn_fixed_threads_of_each_burnin_worker()[burnin_worker];
        if (!threads_of_the_chains) {
              const int n_threads = fn_number_of_threads_of_RcppParallel();
              threads_of_the_chains.reset(new fixed_threads_of_the_chains(n_chains, n_threads));
        }
        return *threads_of_the_chains;

}

inline void fn_end_the_fixed_threads_of_the_chains_of_burnin_worker(const void *burnin_worker) {

        std::unique_ptr<fixed_threads_of_the_chains> threads_of_the_chains;
        {
              std::lock_guard<std::mutex> lock(fn_mutex_of_the_fixed_threads_of_each_burnin_worker());
              auto entry = fn_fixed_threads_of_each_burnin_worker().find(burnin_worker);
              if (entry == fn_fixed_threads_of_each_burnin_worker().end()) return;
              threads_of_the_chains = std::move(entry->second);
              fn_fixed_threads_of_each_burnin_worker().erase(entry);
        }
        //// (the threads end here, outside the lock)
        threads_of_the_chains.reset();

}

////
//// ---- (sampling and burn-in of the built-in models) the fixed threads of the chains, kept for the whole R
////      process: one set of threads for each number of chains (and number of threads), made the first time it
////      is needed and never ended. Ending them crashed TBB: with each chain's own TBB arena
////      (NICOSTAN_PER_CHAIN_TBB_ARENA=1, the_TBB_arena_of_each_chain.hpp) each fixed thread is a TBB master with
////      an arena of its own; once the fixed threads of a burn-in worker had ended, the first TBB call of a new
////      fixed thread (the master scheduler of its arena, tbb::internal::generic_scheduler::allocate_task)
////      segfaulted (Paper 1 redo, E1 burn-in, LC-MVP, N = 500, 4 x 8 / 8: run 1 finished, run 2 crashed;
////      with the per-chain TBB arena off, 73 runs and no crash). Kept threads also keep each chain's OpenMP team
////      and TBB arena from one call to the next.
////
inline std::mutex &fn_mutex_of_the_fixed_threads_kept_for_the_whole_process() {

        static std::mutex *mutex_of_the_table = new std::mutex();
        return *mutex_of_the_table;

}

inline std::map<std::pair<int, int>, std::unique_ptr<fixed_threads_of_the_chains>> &
fn_fixed_threads_kept_for_the_whole_process() {

        static auto *table = new std::map<std::pair<int, int>, std::unique_ptr<fixed_threads_of_the_chains>>();
        return *table;

}

inline fixed_threads_of_the_chains &
fn_fixed_threads_of_the_chains_kept_for_the_whole_process(const int n_chains) {

        const int n_threads = std::max(1, std::min(n_chains, fn_number_of_threads_of_RcppParallel()));
        std::lock_guard<std::mutex> lock(fn_mutex_of_the_fixed_threads_kept_for_the_whole_process());
        std::unique_ptr<fixed_threads_of_the_chains> &threads_of_the_chains =
              fn_fixed_threads_kept_for_the_whole_process()[std::make_pair(n_chains, n_threads)];
        if (!threads_of_the_chains) {
              threads_of_the_chains.reset(new fixed_threads_of_the_chains(n_chains, n_threads));
        }
        return *threads_of_the_chains;

}

//// every chain run once (run_chain(chain_index)) on the threads kept for this number of chains; if another
//// caller is running chains on them at that moment, on threads made for this call only (ended at its end):
inline void fn_run_every_chain_on_the_fixed_threads_kept_for_the_whole_process(
                  const int n_chains,
                  const std::function<void(int)> &run_chain) {

        fixed_threads_of_the_chains &threads_of_the_chains =
              fn_fixed_threads_of_the_chains_kept_for_the_whole_process(n_chains);
        std::unique_lock<std::mutex> lock(threads_of_the_chains.mutex_held_whilst_the_chains_run,
                                          std::try_to_lock);
        if (lock.owns_lock()) {
              threads_of_the_chains.fn_run_every_chain(run_chain);
              return;
        }
        fixed_threads_of_the_chains threads_of_this_call(n_chains, fn_number_of_threads_of_RcppParallel());
        threads_of_this_call.fn_run_every_chain(run_chain);

}
























