#pragma once

#include <cstdint>
#include <cstdlib>
#include <string>

#include <dirent.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <tbb/task_arena.h>
#include <tbb/task_scheduler_observer.h>

////
//// ---- Every thread that runs NicoStan's chains or their within-chain parallel (WCP) work asks the clock
////      governor for the full clock whenever it runs: the thread's minimum utilisation clamp (uclamp.min) is
////      set to 1024. Without it the governor (schedutil) sets each CPU's clock from the utilisation of the
////      threads on it, measured at that CPU's current clock, so a thread that often sleeps and wakes - a WCP
////      thread between the parallel parts of a gradient, or a chain's thread whilst R adapts between burn-in
////      iterations - is seen as lightly used and its CPU stays at the lowest clock (1.36 to 1.9 GHz on the
////      local-HPC, against 3.7 GHz): the chains on those CPUs ran up to 2.4 times slower than the chain on R's
////      own thread (local-HPC, centred SV, AVX2, N = 25,000, burn-in 4 x 8 / 32: the slowest chain 8.2 ms per
////      iteration, 4.5 ms with the clamp). The clamp needs no privilege, keeps each thread's policy and nice
////      level (so every run keeps its lowest priority), changes no result and ends with the process. A kernel
////      without utilisation clamping refuses the request, and nothing changes. Switched off with the
////      environment variable NICOSTAN_DO_NOT_ASK_FOR_THE_FULL_CLOCK_OF_THE_COMPUTE_THREADS=1 (read once per
////      process).
////

//// (the kernel's struct sched_attr with its utilisation clamps)
struct nicostan_scheduling_attributes_with_utilisation_clamps_t {
    uint32_t size;
    uint32_t sched_policy;
    uint64_t sched_flags;
    int32_t  sched_nice;
    uint32_t sched_priority;
    uint64_t sched_runtime;
    uint64_t sched_deadline;
    uint64_t sched_period;
    uint32_t sched_util_min;
    uint32_t sched_util_max;
};

inline bool fn_compute_threads_ask_for_the_full_clock() {

    static const bool ask = [] {
        const char *value = std::getenv("NICOSTAN_DO_NOT_ASK_FOR_THE_FULL_CLOCK_OF_THE_COMPUTE_THREADS");
        return !((value != nullptr) && (std::string(value) == "1"));
    }();
    return ask;

}

//// (one thread, by its id; 0 is the calling thread)
inline void fn_ask_the_clock_governor_for_the_full_clock_of_a_thread(const long thread_id) {

#if defined(__linux__) && defined(SYS_sched_setattr)
    nicostan_scheduling_attributes_with_utilisation_clamps_t attributes{};
    attributes.size = sizeof(attributes);
    //// (SCHED_FLAG_KEEP_POLICY | SCHED_FLAG_KEEP_PARAMS | SCHED_FLAG_UTIL_CLAMP_MIN: only uclamp.min changes)
    attributes.sched_flags = 0x08 | 0x10 | 0x20;
    attributes.sched_util_min = 1024;
    syscall(SYS_sched_setattr, thread_id, &attributes, 0);
#endif

}

//// (the calling thread, once)
inline void fn_ask_the_clock_governor_for_the_full_clock_whenever_this_thread_runs() {

    static thread_local bool asked = false;
    if (asked) return;
    asked = true;
    if (!fn_compute_threads_ask_for_the_full_clock()) return;
    fn_ask_the_clock_governor_for_the_full_clock_of_a_thread(0);

}

//// (every thread of this process at the time of the call: e.g. TBB's workers made before NicoStan's chains ran,
////  which run a gradient's parallel parts without entering NicoStan's code first; threads made later by a thread
////  that asked inherit its clamp)
inline void fn_ask_the_clock_governor_for_the_full_clock_of_every_thread_of_this_process() {

#if defined(__linux__)
    if (!fn_compute_threads_ask_for_the_full_clock()) return;
    DIR *threads = opendir("/proc/self/task");
    if (threads == nullptr) return;
    while (struct dirent *thread = readdir(threads)) {
        if (thread->d_name[0] < '0' || thread->d_name[0] > '9') continue;
        fn_ask_the_clock_governor_for_the_full_clock_of_a_thread(std::strtol(thread->d_name, nullptr, 10));
    }
    closedir(threads);
#endif

}

//// (the TBB threads entering one arena, e.g. a chain's own arena of its WCP threads)
class full_clock_of_the_threads_entering_an_arena : public tbb::task_scheduler_observer {
public:
    explicit full_clock_of_the_threads_entering_an_arena(tbb::task_arena &arena)
      : tbb::task_scheduler_observer(arena) {

        observe(true);

    }
    ~full_clock_of_the_threads_entering_an_arena() override {
        observe(false);
    }
    void on_scheduler_entry(bool /*worker*/) override {
        fn_ask_the_clock_governor_for_the_full_clock_whenever_this_thread_runs();
    }
};























