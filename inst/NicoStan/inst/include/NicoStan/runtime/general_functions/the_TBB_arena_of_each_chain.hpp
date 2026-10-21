#pragma once

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <string>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/partitioner.h>
#include <tbb/task_arena.h>

#include <NicoStan/runtime/general_functions/full_clock_of_the_compute_threads.hpp>

////
//// ---- Each chain's own TBB arena of exactly n_threads_WCP threads (one reserved for the chain's thread), in
////      which the chain's within-chain parallel (WCP) work runs: the gradient's reduce_sum
////      (lp_grad_model_selector.hpp) and, when the chain holds many nuisance values, the leapfrog's pass over
////      them (EHMC_dual_sampler_fns.hpp). Without it every chain's tasks go to the one global TBB pool of the
////      process, where the chains take each other's workers. Switched on with the environment variable
////      NICOSTAN_PER_CHAIN_TBB_ARENA=1 (read once per process; off by default: with the shared pool a chain in
////      its serial part lends its idle CPUs to the other chains' tasks, which made Stan math's burn-in faster on
////      PK at N = 25,000). One arena per chain thread (thread_local), made again only when n_threads_WCP
////      changes; the threads entering it ask for the full clock (full_clock_of_the_compute_threads.hpp).
////

inline bool fn_each_chain_runs_its_WCP_work_in_its_own_TBB_arena() {

    static const bool use_per_chain_TBB_arena = [] {
        const char *value = std::getenv("NICOSTAN_PER_CHAIN_TBB_ARENA");
        return (value != nullptr) && (std::string(value) == "1");
    }();
    return use_per_chain_TBB_arena;

}

inline tbb::task_arena &fn_TBB_arena_of_this_chain(const int n_threads_WCP) {

    static thread_local std::unique_ptr<tbb::task_arena> arena_of_this_chain;
    static thread_local int n_threads_of_arena_of_this_chain = -1;
    static thread_local std::unique_ptr<full_clock_of_the_threads_entering_an_arena>
          full_clock_of_the_arena_of_this_chain;
    if (n_threads_of_arena_of_this_chain != n_threads_WCP) {
        full_clock_of_the_arena_of_this_chain.reset();
        arena_of_this_chain.reset(new tbb::task_arena(std::max(1, n_threads_WCP), 1));
        arena_of_this_chain->initialize();
        full_clock_of_the_arena_of_this_chain.reset(
              new full_clock_of_the_threads_entering_an_arena(*arena_of_this_chain));
        n_threads_of_arena_of_this_chain = n_threads_WCP;
    }
    return *arena_of_this_chain;

}

////
//// ---- One pass over the nuisance values of a leapfrog step (each value on its own) run over ranges of the
////      values on the chain's own WCP threads, when the chain has more than one and its own arena, and holds at
////      least minimum_values_per_range values per thread; otherwise on the chain's thread.
////      pass_over_a_range(first, end) does the values first to end - 1. Every range starts at a multiple of 64
////      values, so that each value is computed by the same part of the compiled loop (its vectorised body or its
////      last few values) as in one pass over all of them, and the draws do not change (checked: identical draws
////      after 300 burn-in iterations at 4 x 16 / 32; the last-bit differences of about 1e-18 seen at other
////      layouts also appear between two runs of the same build, at 1 thread per chain too).
////
const int minimum_values_per_range_of_the_nuisance_pass_on_the_WCP_threads = 1024;
const int values_per_block_of_the_nuisance_pass_on_the_WCP_threads = 64;

template <typename Pass_over_a_range>
inline void fn_pass_over_the_nuisance_values_on_the_WCP_threads_of_this_chain(
                  const int n_nuisance,
                  const int n_threads_WCP,
                  Pass_over_a_range &&pass_over_a_range) {

    const int values_per_range =
          std::max(minimum_values_per_range_of_the_nuisance_pass_on_the_WCP_threads,
                   (n_nuisance + std::max(1, n_threads_WCP) - 1) / std::max(1, n_threads_WCP));
    if (n_threads_WCP <= 1 || !fn_each_chain_runs_its_WCP_work_in_its_own_TBB_arena() ||
        n_nuisance < 2 * values_per_range) {

        pass_over_a_range(0, n_nuisance);
        return;

    }
    //// (ranges of whole blocks of 64 values; the last block ends at n_nuisance)
    const int block = values_per_block_of_the_nuisance_pass_on_the_WCP_threads;
    const int n_blocks = (n_nuisance + block - 1) / block;
    const int blocks_per_range = std::max(1, values_per_range / block);
    fn_TBB_arena_of_this_chain(n_threads_WCP).execute([&]() {

        tbb::parallel_for(tbb::blocked_range<int>(0, n_blocks, blocks_per_range),
                          [&](const tbb::blocked_range<int> &range_of_blocks) {

                                pass_over_a_range(range_of_blocks.begin() * block,
                                                  std::min(n_nuisance, range_of_blocks.end() * block));

                          },
                          tbb::simple_partitioner());

    });

}























