#pragma once

#include <atomic>
#include <chrono>
#include <string>

////
//// ---- (burn-in) the within-chain parallel (WCP) OpenMP team of the calling thread put to sleep at once, after
////      the thread's chains of one burn-in iteration: libomp's workers otherwise spin on their CPUs for
////      KMP_BLOCKTIME (200 ms by default) after the team's last parallel region, waiting for the next one,
////      whilst R runs the tau update (e.g. SpESS-R's threads) and adaptation on those CPUs. The thread's spin
////      time is set to 0 for one short region of the same team size, after which the workers sleep, and is then
////      restored at once, so the next iteration's trajectories run with the usual spin time between their
////      regions (the team is woken once, at its next region). Spin time 0 for the whole burn-in instead puts the
////      workers to sleep between the leapfrog steps' regions too, which made burn-in slower. The region must do
////      something: an empty one does not put the team to sleep. Built-in models only (their WCP regions are
////      OpenMP; a Stan model's are not), and only when the OpenMP runtime is libomp (whose kmp_get_blocktime /
////      kmp_set_blocktime it uses, declared weak, as GCC's omp.h does not declare them; with another runtime
////      nothing is done). Nothing changes but the waiting: every result is the same.
////
extern "C" {
int kmp_get_blocktime(void) __attribute__((weak));
void kmp_set_blocktime(int) __attribute__((weak));
}

inline void fn_put_WCP_team_of_this_thread_to_sleep_after_burnin_iteration(const std::string &Model_type,
                                                                           const int n_threads_WCP) {
#if defined(_OPENMP)
        const bool built_in_model = (Model_type == "LC_MVP") || (Model_type == "MVP") ||
                                    (Model_type == "LC_MVOP") || (Model_type == "MVOP");
        if (!built_in_model || n_threads_WCP <= 1) return;
        if (kmp_get_blocktime == nullptr || kmp_set_blocktime == nullptr) return;   // (not libomp)
        static std::atomic<long> regions_that_put_a_team_to_sleep{0};
        const int spin_time_before = kmp_get_blocktime();
        kmp_set_blocktime(0);
        #pragma omp parallel num_threads(n_threads_WCP)
        {
                regions_that_put_a_team_to_sleep.fetch_add(1, std::memory_order_relaxed);
        }
        kmp_set_blocktime(spin_time_before);
#else
        (void) Model_type;
        (void) n_threads_WCP;
#endif
}

////
//// ---- the gap rule: a WCP team is put to sleep after a burn-in iteration only when R took at least
////      minimum_gap_ms_for_putting_WCP_team_to_sleep_after_burnin_iteration between the previous two iterations
////      (from the end of the last chain of one to the start of the next), i.e. when R's work between iterations
////      is long, as SpESS-R's update on the joint block is. After a short gap, waking the team and its cold
////      caches cost more than the spinning (binary LC-MVP, N = 10,000, b125, one CCD: about 13 ms gaps for ESJD,
////      CHESSR and SpESS-R on interest_only, whose burn-in was 2-10% slower when the team slept after every
////      iteration; about 41 ms for ESJD on the joint block, unchanged either way; about 62 ms for SpESS-R on the
////      joint block, 15-17% faster):
////
const double minimum_gap_ms_for_putting_WCP_team_to_sleep_after_burnin_iteration = 25.0;

inline long long fn_steady_clock_nanoseconds_now() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now().time_since_epoch()).count();
}

//// the time the last chain of the latest burn-in iteration finished (0 before the first):
inline std::atomic<long long> &fn_end_of_last_burnin_iteration_nanoseconds() {
        static std::atomic<long long> end_of_last_burnin_iteration_nanoseconds{0};
        return end_of_last_burnin_iteration_nanoseconds;
}

//// the gap since then, in milliseconds (0 before the first iteration), read when an iteration starts:
inline double fn_gap_since_last_burnin_iteration_ms() {
        const long long end_of_last = fn_end_of_last_burnin_iteration_nanoseconds().load();
        if (end_of_last <= 0) return 0.0;
        return 1e-6 * static_cast<double>(fn_steady_clock_nanoseconds_now() - end_of_last);
}

//// a chain's end recorded (the latest of the iteration's chains is kept):
inline void fn_record_end_of_burnin_iteration_chain() {
        const long long now = fn_steady_clock_nanoseconds_now();
        std::atomic<long long> &end_of_last = fn_end_of_last_burnin_iteration_nanoseconds();
        long long previous = end_of_last.load();
        while (previous < now && !end_of_last.compare_exchange_weak(previous, now)) {
        }
}






















