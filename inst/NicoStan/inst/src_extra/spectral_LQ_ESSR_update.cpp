//// spectral_LQ_ESSR_update.cpp
////
//// ---- The complete per-update computation of the spectral LQ_ESSR criterion (burnin_algorithm =
////      "LQ_ESSR_spectral", "LQ_ESSR_spectral_long_bin_memory" and "LQ_ESSR_spec_bins99_evid_expand";
////      R_fn_spectral_ESS_tau_criterion.R), in C++ (8 Oct 2026): the moving averages, the quadratic and flow length
////      bins, the cosine-mixture fit (NNLS on the normal equations), the objective on the grid of candidate tau,
////      its maximiser, the bootstrap test of the expansion rule "evidence", and the tau move. The arithmetic is
////      that of fn_spectral_ESS_soft_minimum_tau_update() with finite_run_draws = NULL, objective_mix = "none" and
////      jump_weighted_acceptance = FALSE, step by step (R's seq(), findInterval(), which.max() and sd()
////      conventions included), so that the tau path is the same as the R code's up to floating-point rounding
////      (8 Oct 2026: except for the bootstrap draws of the expansion rule "evidence", which come from the
////      bootstrap's own generator, BootstrapNormalGenerator, rather than from R's, for speed).
////      The state is the same nested list as the R code's, so the two can be swapped mid-run.
////      (8 Oct 2026) The cost per update is kept small: no heap allocations in the NNLS refits, the cosine
////      sums of the jitter by rotation, and the soft minimum's power 8 by repeated squaring.
////
////      Compiled at first use by Rcpp::sourceCpp() (no rebuild of NicoStan), as
////      spectral_ESS_nnls_normal_equations.cpp.
////
#include <Rcpp.h>
#include <R_ext/Random.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <cstdint>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <functional>

namespace {

const double PI_VALUE = 3.141592653589793238462643383279502884;
const double NA_DOUBLE = std::numeric_limits<double>::quiet_NaN();

//// ---- R's seq(from, to, by) for by > 0 (seq.default): n = as.integer((to - from) / by + 1e-10),
////      from + (0:n) * by, capped at to; a single value when to and from are (relatively) equal:
std::vector<double> fn_seq_by(const double from, const double to, const double by) {
        std::vector<double> x;
        const double del = to - from;
        if (del == 0.0 && to == 0.0) {
                x.push_back(to);
                return x;
        }
        const double dd = std::fabs(del) / std::max(std::fabs(to), std::fabs(from));
        if (dd < 100.0 * std::numeric_limits<double>::epsilon()) {
                x.push_back(from);
                return x;
        }
        const int n = static_cast<int>(del / by + 1e-10);
        for (int i = 0; i <= n; i++) x.push_back(std::min(from + i * by, to));
        return x;
}

//// ---- R's findInterval(x, edges) + 1, as a 0-based bin index (the number of edges <= x); -1 for NaN:
int fn_bin_index(const double x, const std::vector<double> &edges) {
        if (std::isnan(x)) return -1;
        return static_cast<int>(std::upper_bound(edges.begin(), edges.end(), x) - edges.begin());
}

//// ---- min / max that propagate NaN from either argument (R's pmin / pmax):
inline double fn_min_na(const double a, const double b) {
        if (std::isnan(a) || std::isnan(b)) return NA_DOUBLE;
        return std::min(a, b);
}
inline double fn_max_na(const double a, const double b) {
        if (std::isnan(a) || std::isnan(b)) return NA_DOUBLE;
        return std::max(a, b);
}

//// ---- NicoStan's jitter (EHMC "Compute L"): E[cos(omega L eps)] and E[L] (fn_spectral_ESS_expected_*):
double fn_sum_of_cosines(const double x, const double n) {
        const double half_sine = std::sin(x / 2.0);
        if (std::fabs(half_sine) < 1e-12) return n;
        return std::sin(n * x / 2.0) * std::cos((n + 1.0) * x / 2.0) / half_sine;
}
double fn_expected_cosine_under_jitter(const double omega, const double tau, const double eps) {
        const double a = 2.0 * tau / eps;
        if (a <= 1.0) return std::cos(omega * eps);
        const double n = std::ceil(a);
        const double x = omega * eps;
        return (fn_sum_of_cosines(x, n - 1.0) + (a - n + 1.0) * std::cos(n * x)) / a;
}
double fn_expected_leapfrog_steps_under_jitter(const double tau, const double eps) {
        const double a = 2.0 * tau / eps;
        if (a <= 1.0) return 1.0;
        const double n = std::ceil(a);
        return ((n - 1.0) * n / 2.0 + (a - n + 1.0) * n) / a;
}

//// ---- NNLS on the normal equations (spectral_ESS_nnls_normal_equations.cpp, unchanged): AtA (n x n, row-major),
////      one right-hand side Atb (n); Lawson-Hanson active set, Cholesky with a tiny ridge:
bool fn_cholesky_solve(std::vector<double> gram, const std::vector<double> &right_hand_side, const int n,
                       std::vector<double> &solution) {
        double largest_diagonal = 0.0;
        for (int i = 0; i < n; i++) largest_diagonal = std::max(largest_diagonal, gram[i * n + i]);
        const double ridge = 1e-12 * (largest_diagonal > 0.0 ? largest_diagonal : 1.0);
        for (int i = 0; i < n; i++) gram[i * n + i] += ridge;
        for (int j = 0; j < n; j++) {
                double s = gram[j * n + j];
                for (int k = 0; k < j; k++) s -= gram[j * n + k] * gram[j * n + k];
                if (!(s > 0.0)) return false;
                const double d = std::sqrt(s);
                gram[j * n + j] = d;
                for (int i = j + 1; i < n; i++) {
                        double t = gram[i * n + j];
                        for (int k = 0; k < j; k++) t -= gram[i * n + k] * gram[j * n + k];
                        gram[i * n + j] = t / d;
                }
        }
        std::vector<double> forward(n);
        for (int i = 0; i < n; i++) {
                double t = right_hand_side[i];
                for (int k = 0; k < i; k++) t -= gram[i * n + k] * forward[k];
                forward[i] = t / gram[i * n + i];
        }
        solution.assign(n, 0.0);
        for (int i = n - 1; i >= 0; i--) {
                double t = forward[i];
                for (int k = i + 1; k < n; k++) t -= gram[k * n + i] * solution[k];
                solution[i] = t / gram[i * n + i];
        }
        return true;
}
//// (warm_start: the support of a previous solution, as the initial passive set; the inner loop below makes it
////  feasible or empties it, so the result is a Lawson-Hanson solution either way: the same as a cold start
////  whenever the NNLS solution is unique)
std::vector<double> fn_nnls_one_column(const std::vector<double> &AtA, const std::vector<double> &Atb_column,
                                       const int n, const double tolerance,
                                       const std::vector<double> *warm_start = nullptr) {
        std::vector<double> x(n, 0.0), w(n), s(n, 0.0);
        std::vector<int> passive(n, 0), excluded(n, 0);
        for (int i = 0; i < n; i++) w[i] = Atb_column[i];
        bool warm = false;
        if (warm_start != nullptr) {
                for (int i = 0; i < n; i++) {
                        if ((*warm_start)[i] > 0.0) {
                                passive[i] = 1;
                                x[i] = (*warm_start)[i];   // a feasible start: the previous solution
                                warm = true;
                        }
                }
        }
        for (int outer = 0; outer < 3 * n; outer++) {
                int entering = -1;
                if (!warm) {
                        double largest = tolerance;
                        for (int i = 0; i < n; i++) {
                                if (!passive[i] && !excluded[i] && w[i] > largest) {
                                        largest = w[i];
                                        entering = i;
                                }
                        }
                        if (entering < 0) break;
                        passive[entering] = 1;
                }
                warm = false;
                for (int inner = 0; inner < 3 * n; inner++) {
                        std::vector<int> P;
                        for (int i = 0; i < n; i++) if (passive[i]) P.push_back(i);
                        const int np = static_cast<int>(P.size());
                        if (np == 0) break;
                        std::vector<double> gram(np * np), right_hand_side(np), sub;
                        for (int a = 0; a < np; a++) {
                                right_hand_side[a] = Atb_column[P[a]];
                                for (int b = 0; b < np; b++) gram[a * np + b] = AtA[P[a] * n + P[b]];
                        }
                        if (!fn_cholesky_solve(gram, right_hand_side, np, sub)) {
                                if (entering >= 0) {
                                        passive[entering] = 0;
                                        excluded[entering] = 1;
                                } else {
                                        std::fill(passive.begin(), passive.end(), 0);
                                        std::fill(x.begin(), x.end(), 0.0);
                                }
                                break;
                        }
                        std::fill(s.begin(), s.end(), 0.0);
                        bool all_positive = true;
                        for (int a = 0; a < np; a++) {
                                s[P[a]] = sub[a];
                                if (sub[a] <= 0.0) all_positive = false;
                        }
                        if (all_positive) {
                                x = s;
                                break;
                        }
                        double alpha = 1e300;
                        for (int a = 0; a < np; a++) {
                                const int i = P[a];
                                if (s[i] <= 0.0) {
                                        const double ratio = x[i] / (x[i] - s[i]);
                                        if (ratio < alpha) alpha = ratio;
                                }
                        }
                        for (int i = 0; i < n; i++) x[i] += alpha * (s[i] - x[i]);
                        for (int a = 0; a < np; a++) {
                                const int i = P[a];
                                if (x[i] <= 1e-15) {
                                        x[i] = 0.0;
                                        passive[i] = 0;
                                }
                        }
                }
                // (8 Oct 2026) w = Atb - AtA x over the non-zero x only: the zero terms subtract exactly zero, so
                // w is unchanged (up to the sign of a zero, which only enters the comparisons above)
                std::vector<int> support;
                for (int j = 0; j < n; j++) if (x[j] != 0.0) support.push_back(j);
                for (int i = 0; i < n; i++) {
                        double t = Atb_column[i];
                        for (int j : support) t -= AtA[i * n + j] * x[j];
                        w[i] = t;
                }
        }
        return x;
}

//// ---- (8 Oct 2026) the same Cholesky solve and NNLS without heap allocations: every buffer comes from a
////      workspace that is reused across rows, bootstrap draws and updates (the arithmetic is that of
////      fn_cholesky_solve() and fn_nnls_one_column() above, which are kept for reference):
struct NNLSWorkspace {
        std::vector<double> x, w, s, gram, right_hand_side, sub, forward;
        std::vector<int> passive, excluded, P, support;
        // the factors of A^T A on the passive sets met so far with one A^T A (fn_nnls_one_column_in_workspace()):
        const double *AtA_of_the_stored_factors = nullptr;
        int n_of_the_stored_factors = -1;
        // (the identity of the prepared fit whose A^T A the stored factors are of, and of the fit now being made:
        //  0 = not given, the address of A^T A decides; buffers kept between updates can reuse an address)
        uint64_t fit_identity_of_the_stored_factors = 0, fit_identity_now = 0;
        std::vector<uint64_t> stored_factor_passive_set;   // per slot: the passive set as a bit mask (0 = empty)
        std::vector<char> stored_factor_exists;            // per slot: 1 = factorised, 0 = factorisation failed
        std::vector<double> stored_factors;                // per slot: the factor, np x np (as gram after it)
        void resize(const int n) {
                x.resize(n);
                w.resize(n);
                s.resize(n);
                gram.resize(static_cast<size_t>(n) * n);
                right_hand_side.resize(n);
                sub.resize(n);
                forward.resize(n);
                passive.resize(n);
                excluded.resize(n);
                P.resize(n);
                support.resize(n);
        }
};

bool fn_cholesky_solve_in_place(double *gram, const double *right_hand_side, const int n, double *forward,
                                double *solution) {
        double largest_diagonal = 0.0;
        for (int i = 0; i < n; i++) largest_diagonal = std::max(largest_diagonal, gram[i * n + i]);
        const double ridge = 1e-12 * (largest_diagonal > 0.0 ? largest_diagonal : 1.0);
        for (int i = 0; i < n; i++) gram[i * n + i] += ridge;
        for (int j = 0; j < n; j++) {
                double s = gram[j * n + j];
                for (int k = 0; k < j; k++) s -= gram[j * n + k] * gram[j * n + k];
                if (!(s > 0.0)) return false;
                const double d = std::sqrt(s);
                gram[j * n + j] = d;
                for (int i = j + 1; i < n; i++) {
                        double t = gram[i * n + j];
                        for (int k = 0; k < j; k++) t -= gram[i * n + k] * gram[j * n + k];
                        gram[i * n + j] = t / d;
                }
        }
        for (int i = 0; i < n; i++) {
                double t = right_hand_side[i];
                for (int k = 0; k < i; k++) t -= gram[i * n + k] * forward[k];
                forward[i] = t / gram[i * n + i];
        }
        for (int i = n - 1; i >= 0; i--) {
                double t = forward[i];
                for (int k = i + 1; k < n; k++) t -= gram[k * n + i] * solution[k];
                solution[i] = t / gram[i * n + i];
        }
        return true;
}

//// ---- the factors of A^T A on the passive sets met within one update: A^T A is the same for every row and
////      every bootstrap draw of an update, so the factor of a passive set is the same numbers each time the
////      set is met; a row or draw that meets a stored set solves only the two triangular systems with it (the
////      loops of fn_cholesky_solve_in_place() after the factorisation). Sets of up to
////      largest_passive_set_with_stored_factor frequencies, in n_slots_for_stored_factors slots (a set whose
////      slot holds another set is factorised again and replaces it); with n > 64 frequencies nothing is stored:
const int n_slots_for_stored_factors = 4096;
const int largest_passive_set_with_stored_factor = 8;

void fn_cholesky_triangular_solves(const double *gram, const double *right_hand_side, const int n,
                                   double *forward, double *solution) {
        for (int i = 0; i < n; i++) {
                double t = right_hand_side[i];
                for (int k = 0; k < i; k++) t -= gram[i * n + k] * forward[k];
                forward[i] = t / gram[i * n + i];
        }
        for (int i = n - 1; i >= 0; i--) {
                double t = forward[i];
                for (int k = i + 1; k < n; k++) t -= gram[k * n + i] * solution[k];
                solution[i] = t / gram[i * n + i];
        }
}

void fn_prepare_stored_factors(NNLSWorkspace &ws, const double *AtA, const int n) {
        if (ws.stored_factor_passive_set.empty()) {
                ws.stored_factor_passive_set.assign(n_slots_for_stored_factors, 0);
                ws.stored_factor_exists.assign(n_slots_for_stored_factors, 0);
                ws.stored_factors.resize(static_cast<size_t>(n_slots_for_stored_factors) *
                                         largest_passive_set_with_stored_factor *
                                         largest_passive_set_with_stored_factor);
        }
        // if (AtA != ws.AtA_of_the_stored_factors || n != ws.n_of_the_stored_factors) {
        const bool other_fit = ws.fit_identity_now != 0 ?
                               ws.fit_identity_now != ws.fit_identity_of_the_stored_factors :
                               AtA != ws.AtA_of_the_stored_factors;
        if (other_fit || n != ws.n_of_the_stored_factors) {
                std::fill(ws.stored_factor_passive_set.begin(), ws.stored_factor_passive_set.end(), 0);
                ws.AtA_of_the_stored_factors = AtA;
                ws.n_of_the_stored_factors = n;
                ws.fit_identity_of_the_stored_factors = ws.fit_identity_now;
        }
}

void fn_nnls_one_column_in_workspace(const std::vector<double> &AtA, const double *Atb_column, const int n,
                                     const double tolerance, const double *warm_start, NNLSWorkspace &ws,
                                     double *x_out) {
        double *x = ws.x.data(), *w = ws.w.data(), *s = ws.s.data();
        int *passive = ws.passive.data(), *excluded = ws.excluded.data(), *P = ws.P.data();
        int *support = ws.support.data();
        const bool factors_can_be_stored = n <= 64;
        if (factors_can_be_stored) fn_prepare_stored_factors(ws, AtA.data(), n);
        for (int i = 0; i < n; i++) {
                x[i] = 0.0;
                s[i] = 0.0;
                passive[i] = 0;
                excluded[i] = 0;
                w[i] = Atb_column[i];
        }
        bool warm = false;
        if (warm_start != nullptr) {
                for (int i = 0; i < n; i++) {
                        if (warm_start[i] > 0.0) {
                                passive[i] = 1;
                                x[i] = warm_start[i];   // a feasible start: the previous solution
                                warm = true;
                        }
                }
        }
        for (int outer = 0; outer < 3 * n; outer++) {
                int entering = -1;
                if (!warm) {
                        double largest = tolerance;
                        for (int i = 0; i < n; i++) {
                                if (!passive[i] && !excluded[i] && w[i] > largest) {
                                        largest = w[i];
                                        entering = i;
                                }
                        }
                        if (entering < 0) break;
                        passive[entering] = 1;
                }
                warm = false;
                for (int inner = 0; inner < 3 * n; inner++) {
                        int np = 0;
                        for (int i = 0; i < n; i++) if (passive[i]) P[np++] = i;
                        if (np == 0) break;
                        double *gram = ws.gram.data(), *right_hand_side = ws.right_hand_side.data();
                        double *sub = ws.sub.data();
                        // for (int a = 0; a < np; a++) {
                        //         right_hand_side[a] = Atb_column[P[a]];
                        //         for (int b = 0; b < np; b++) gram[a * np + b] = AtA[P[a] * n + P[b]];
                        // }
                        // if (!fn_cholesky_solve_in_place(gram, right_hand_side, np, ws.forward.data(), sub)) {
                        for (int a = 0; a < np; a++) right_hand_side[a] = Atb_column[P[a]];
                        bool solved = false, factor_was_stored = false;
                        int slot = -1;
                        uint64_t passive_set = 0;
                        if (factors_can_be_stored && np <= largest_passive_set_with_stored_factor) {
                                for (int a = 0; a < np; a++) passive_set |= static_cast<uint64_t>(1) << P[a];
                                slot = static_cast<int>((passive_set * 0x9E3779B97F4A7C15ULL) >> 52);
                                if (ws.stored_factor_passive_set[slot] == passive_set) {
                                        factor_was_stored = true;
                                        solved = ws.stored_factor_exists[slot] != 0;
                                        if (solved) {
                                                fn_cholesky_triangular_solves(ws.stored_factors.data() +
                                                                              static_cast<size_t>(slot) * 64,
                                                                              right_hand_side, np,
                                                                              ws.forward.data(), sub);
                                        }
                                }
                        }
                        if (!factor_was_stored) {
                                for (int a = 0; a < np; a++) {
                                        for (int b = 0; b < np; b++) gram[a * np + b] = AtA[P[a] * n + P[b]];
                                }
                                solved = fn_cholesky_solve_in_place(gram, right_hand_side, np, ws.forward.data(),
                                                                    sub);
                                if (slot >= 0) {
                                        ws.stored_factor_passive_set[slot] = passive_set;
                                        ws.stored_factor_exists[slot] = solved ? 1 : 0;
                                        if (solved) {
                                                std::copy(gram, gram + np * np, ws.stored_factors.data() +
                                                                                static_cast<size_t>(slot) * 64);
                                        }
                                }
                        }
                        if (!solved) {
                                if (entering >= 0) {
                                        passive[entering] = 0;
                                        excluded[entering] = 1;
                                } else {
                                        for (int i = 0; i < n; i++) {
                                                passive[i] = 0;
                                                x[i] = 0.0;
                                        }
                                }
                                break;
                        }
                        for (int i = 0; i < n; i++) s[i] = 0.0;
                        bool all_positive = true;
                        for (int a = 0; a < np; a++) {
                                s[P[a]] = sub[a];
                                if (sub[a] <= 0.0) all_positive = false;
                        }
                        if (all_positive) {
                                for (int i = 0; i < n; i++) x[i] = s[i];
                                break;
                        }
                        double alpha = 1e300;
                        for (int a = 0; a < np; a++) {
                                const int i = P[a];
                                if (s[i] <= 0.0) {
                                        const double ratio = x[i] / (x[i] - s[i]);
                                        if (ratio < alpha) alpha = ratio;
                                }
                        }
                        for (int i = 0; i < n; i++) x[i] += alpha * (s[i] - x[i]);
                        for (int a = 0; a < np; a++) {
                                const int i = P[a];
                                if (x[i] <= 1e-15) {
                                        x[i] = 0.0;
                                        passive[i] = 0;
                                }
                        }
                }
                int n_support = 0;
                for (int j = 0; j < n; j++) if (x[j] != 0.0) support[n_support++] = j;
                for (int i = 0; i < n; i++) {
                        double t = Atb_column[i];
                        for (int k = 0; k < n_support; k++) t -= AtA[i * n + support[k]] * x[support[k]];
                        w[i] = t;
                }
        }
        for (int i = 0; i < n; i++) x_out[i] = x[i];
}

//// ---- a row's NNLS solution when it is the solve on a given support (the previous update's support of the
////      row): the solve on the support, by the arithmetic of fn_nnls_one_column_in_workspace() (the same factor
////      and triangular solves), is accepted when every entry is positive and no frequency outside the support
////      would enter (w = A^T b - A^T A x, summed as there, at most the tolerance); that is where the active-set
////      loop of fn_nnls_one_column_in_workspace() from a cold start stops, with the same solve, so the result is
////      the same numbers (the NNLS solution being unique). Near-ties are left to the cold start: the support is
////      only when each entry exceeds 1e-8 times the largest and each w outside it is below -1e4 times the
////      tolerance (a clear margin, so the cold start cannot stop on another support). Otherwise false, and the
////      row's NNLS runs from a cold start:
bool fn_nnls_on_support_if_solution_in_workspace(const std::vector<double> &AtA, const double *Atb_column,
                                                 const int n, const double tolerance, const char *support,
                                                 NNLSWorkspace &ws, double *x_out) {
        double *x = ws.x.data(), *w = ws.w.data();
        int *P = ws.P.data(), *passive = ws.passive.data();
        int np = 0;
        for (int i = 0; i < n; i++) {
                passive[i] = support[i] ? 1 : 0;
                if (support[i]) P[np++] = i;
        }
        if (np == 0) return false;
        const bool factors_can_be_stored = n <= 64;
        if (factors_can_be_stored) fn_prepare_stored_factors(ws, AtA.data(), n);
        double *gram = ws.gram.data(), *right_hand_side = ws.right_hand_side.data();
        double *sub = ws.sub.data();
        for (int a = 0; a < np; a++) right_hand_side[a] = Atb_column[P[a]];
        bool solved = false, factor_was_stored = false;
        int slot = -1;
        uint64_t passive_set = 0;
        if (factors_can_be_stored && np <= largest_passive_set_with_stored_factor) {
                for (int a = 0; a < np; a++) passive_set |= static_cast<uint64_t>(1) << P[a];
                slot = static_cast<int>((passive_set * 0x9E3779B97F4A7C15ULL) >> 52);
                if (ws.stored_factor_passive_set[slot] == passive_set) {
                        factor_was_stored = true;
                        solved = ws.stored_factor_exists[slot] != 0;
                        if (solved) {
                                fn_cholesky_triangular_solves(ws.stored_factors.data() +
                                                              static_cast<size_t>(slot) * 64,
                                                              right_hand_side, np, ws.forward.data(), sub);
                        }
                }
        }
        if (!factor_was_stored) {
                for (int a = 0; a < np; a++) {
                        for (int b = 0; b < np; b++) gram[a * np + b] = AtA[P[a] * n + P[b]];
                }
                solved = fn_cholesky_solve_in_place(gram, right_hand_side, np, ws.forward.data(), sub);
                if (slot >= 0) {
                        ws.stored_factor_passive_set[slot] = passive_set;
                        ws.stored_factor_exists[slot] = solved ? 1 : 0;
                        if (solved) {
                                std::copy(gram, gram + np * np,
                                          ws.stored_factors.data() + static_cast<size_t>(slot) * 64);
                        }
                }
        }
        if (!solved) return false;
        // for (int a = 0; a < np; a++) if (!(sub[a] > 0.0)) return false;
        double largest_entry = 0.0;
        for (int a = 0; a < np; a++) largest_entry = std::max(largest_entry, sub[a]);
        for (int a = 0; a < np; a++) if (!(sub[a] > 1e-8 * largest_entry)) return false;
        for (int i = 0; i < n; i++) x[i] = 0.0;
        for (int a = 0; a < np; a++) x[P[a]] = sub[a];
        // (w as fn_nnls_one_column_in_workspace() computes it, over the non-zero x in increasing order)
        for (int i = 0; i < n; i++) {
                double t = Atb_column[i];
                for (int a = 0; a < np; a++) t -= AtA[i * n + P[a]] * x[P[a]];
                w[i] = t;
        }
        // for (int i = 0; i < n; i++) if (!passive[i] && w[i] > tolerance) return false;
        for (int i = 0; i < n; i++) if (!passive[i] && !(w[i] < -1e4 * tolerance)) return false;
        for (int i = 0; i < n; i++) x_out[i] = x[i];
        return true;
}

//// ---- the perturbed inputs of a bootstrap draw, entries begin to end - 1 of its normals (the quadratic bins'
////      first, then the usable flow bins'), from the generator's pairs: the normal 2j is u f and 2j + 1 is v f
////      of the pair j (pair_f holding f), as BootstrapNormalGenerator::fn_normal() returns them; each entry by
////      the arithmetic of the draw loop of the update (fn_generate_draw_into_set(), commented out there):
void fn_perturbed_inputs_of_draw(const double *pair_u, const double *pair_v, const double *pair_f,
                                 const int64_t first_normal, const int64_t first_pair, const int64_t begin,
                                 const int64_t end, const int64_t n_quadratic_normals, const int n_rows,
                                 const double *quadratic_means, const double *quadratic_errors,
                                 const double *flow_means, const double *flow_errors, const int *flow_usable,
                                 const double *flow_counts, double *perturbed_quadratic,
                                 double *perturbed_flow_sums) {
        for (int64_t k = begin; k < end; k++) {
                const int64_t normal_index = first_normal + k;
                const int64_t j = normal_index / 2 - first_pair;
                const double normal = (normal_index % 2 == 0) ? pair_u[j] * pair_f[j] : pair_v[j] * pair_f[j];
                if (k < n_quadratic_normals) {
                        perturbed_quadratic[k] = fn_max_na(quadratic_means[k] + quadratic_errors[k] * normal,
                                                           0.0);
                } else {
                        const int64_t index = k - n_quadratic_normals;
                        const int bin = flow_usable[index / n_rows];
                        perturbed_flow_sums[index] = fn_max_na(flow_means[index] + flow_errors[index] * normal,
                                                               0.0) * flow_counts[bin];
                }
        }
}

//// ---- (8 Oct 2026) the bootstrap's own normal generator (xoshiro256++ seeded by splitmix64, Marsaglia's polar
////      method), in place of R's generator: about ten times faster per draw, it leaves R's random-number stream
////      untouched, and it is seeded from the update count as before (104729 + the quadratic updates so far):
struct BootstrapNormalGenerator {
        uint64_t state[4];
        bool has_spare = false;
        double spare = 0.0;
        static uint64_t fn_splitmix64(uint64_t &x) {
                uint64_t z = (x += 0x9E3779B97F4A7C15ULL);
                z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
                z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
                return z ^ (z >> 31);
        }
        static inline uint64_t fn_rotate_left(const uint64_t x, const int k) {
                return (x << k) | (x >> (64 - k));
        }
        explicit BootstrapNormalGenerator(uint64_t seed) {
                for (int i = 0; i < 4; i++) state[i] = fn_splitmix64(seed);
        }
        inline uint64_t fn_next() {
                const uint64_t result = fn_rotate_left(state[0] + state[3], 23) + state[0];
                const uint64_t t = state[1] << 17;
                state[2] ^= state[0];
                state[3] ^= state[1];
                state[1] ^= state[2];
                state[0] ^= state[3];
                state[2] ^= t;
                state[3] = fn_rotate_left(state[3], 45);
                return result;
        }
        inline double fn_uniform() {
                return static_cast<double>(fn_next() >> 11) * 0x1.0p-53;   // [0, 1)
        }
        inline double fn_normal() {
                if (has_spare) {
                        has_spare = false;
                        return spare;
                }
                double u, v, q;
                do {
                        u = 2.0 * fn_uniform() - 1.0;
                        v = 2.0 * fn_uniform() - 1.0;
                        q = u * u + v * v;
                } while (q >= 1.0 || q == 0.0);
                const double f = std::sqrt(-2.0 * std::log(q) / q);
                spare = v * f;
                has_spare = true;
                return u * f;
        }
};

//// ---- x^power, by repeated squaring for the soft minimum's default power (8), std::pow otherwise:
inline double fn_power(const double x, const double power) {
        if (power == 8.0) {
                const double x2 = x * x;
                const double x4 = x2 * x2;
                return x4 * x4;
        }
        return std::pow(x, power);
}

//// ---- the per-parameter (per-row) fits spread over threads: the rows are split into contiguous blocks, at most
////      one block per thread and at least minimum_rows_per_thread_for_per_parameter_fits rows per block (with
////      fewer rows: one block, i.e. the serial loop), and the calling thread fits the first block itself.
////      Each row's arithmetic is the serial loop's, and nothing is summed across rows inside a block, so the
////      results do not depend on the number of threads. The worker threads touch only C++ buffers prepared
////      beforehand (no R or Rcpp calls):
// const int minimum_rows_per_thread_for_per_parameter_fits = 1000;
const int minimum_rows_per_thread_for_per_parameter_fits = 250;

int fn_number_of_row_blocks(const int n_rows, const int n_threads_for_update_loops) {
        const int largest_number_of_blocks_by_rows = n_rows / minimum_rows_per_thread_for_per_parameter_fits;
        return std::max(1, std::min(n_threads_for_update_loops, largest_number_of_blocks_by_rows));
}

// template <typename BlockWork>
// void fn_for_blocks(const int64_t n_items, const int n_blocks, BlockWork work) {
//         auto block_start = [n_items, n_blocks](const int block) {
//                 return (n_items * block) / n_blocks;
//         };
//         if (n_blocks <= 1) {
//                 work(0, static_cast<int64_t>(0), n_items);
//                 return;
//         }
//         std::vector<std::thread> workers;
//         workers.reserve(n_blocks - 1);
//         int n_started = 0;
//         try {
//                 for (int block = 1; block < n_blocks; block++) {
//                         workers.emplace_back(work, block, block_start(block), block_start(block + 1));
//                         n_started++;
//                 }
//         } catch (...) {
//                 // (a thread that could not be started: its block, and the blocks after it, run on this thread)
//         }
//         work(0, block_start(0), block_start(1));
//         for (int block = 1 + n_started; block < n_blocks; block++) {
//                 work(block, block_start(block), block_start(block + 1));
//         }
//         for (std::thread &worker : workers) worker.join();
// }

//// ---- the threads that run the blocks, started once (at the first loop that needs them) and kept, waiting, for
////      the later loops of this R process, instead of starting new threads for every loop: a loop's blocks are
////      handed out one at a time to the waiting threads and to the calling thread, and the loop returns when
////      every block has finished. Each block writes only its own results, so which thread runs a block does not
////      change any result. The threads touch only C++ buffers (no R or Rcpp calls), and the pool is never
////      destroyed (its threads end with the process):
struct ThreadsForBlocks {
        std::mutex mutex;
        std::condition_variable work_ready, work_finished;
        std::vector<std::thread> threads;
        const std::function<void(int)> *work = nullptr;
        int n_blocks = 0, next_block = 0, n_unfinished = 0;
        uint64_t loop_number = 0;
        void fn_wait_for_loops() {
                uint64_t loop_seen = 0;
                std::unique_lock<std::mutex> lock(mutex);
                for (;;) {
                        work_ready.wait(lock, [&] { return loop_number != loop_seen; });
                        loop_seen = loop_number;
                        while (work != nullptr && next_block < n_blocks) {
                                const int block = next_block++;
                                const std::function<void(int)> *loop_work = work;
                                lock.unlock();
                                (*loop_work)(block);
                                lock.lock();
                                if (--n_unfinished == 0) work_finished.notify_all();
                        }
                }
        }
        // (at least n_threads_wanted threads waiting; false if none could be started)
        bool fn_start_threads(const int n_threads_wanted) {
                std::unique_lock<std::mutex> lock(mutex);
                while (static_cast<int>(threads.size()) < n_threads_wanted) {
                        try {
                                threads.emplace_back(&ThreadsForBlocks::fn_wait_for_loops, this);
                                threads.back().detach();
                        } catch (...) {
                                break;
                        }
                }
                return !threads.empty();
        }
        void fn_run(const int n_blocks_of_loop, const std::function<void(int)> &loop_work) {
                std::unique_lock<std::mutex> lock(mutex);
                work = &loop_work;
                n_blocks = n_blocks_of_loop;
                next_block = 0;
                n_unfinished = n_blocks_of_loop;
                loop_number++;
                work_ready.notify_all();
                while (next_block < n_blocks) {
                        const int block = next_block++;
                        lock.unlock();
                        loop_work(block);
                        lock.lock();
                        --n_unfinished;
                }
                work_finished.wait(lock, [&] { return n_unfinished == 0; });
                work = nullptr;
        }
};

ThreadsForBlocks &fn_threads_for_blocks() {
        static ThreadsForBlocks *threads_for_blocks = new ThreadsForBlocks();   // (never destroyed)
        return *threads_for_blocks;
}

template <typename BlockWork>
void fn_for_blocks(const int64_t n_items, const int n_blocks, BlockWork work) {
        auto block_start = [n_items, n_blocks](const int block) {
                return (n_items * block) / n_blocks;
        };
        if (n_blocks <= 1) {
                work(0, static_cast<int64_t>(0), n_items);
                return;
        }
        ThreadsForBlocks &threads_for_blocks = fn_threads_for_blocks();
        if (!threads_for_blocks.fn_start_threads(n_blocks - 1)) {
                for (int block = 0; block < n_blocks; block++) {
                        work(block, block_start(block), block_start(block + 1));
                }
                return;
        }
        const std::function<void(int)> loop_work = [&](const int block) {
                work(block, block_start(block), block_start(block + 1));
        };
        threads_for_blocks.fn_run(n_blocks, loop_work);
}

//// the copies and scalings of whole matrices (entry by entry, so the same in any split), at least
//// minimum_entries_per_thread_for_copies entries (2 MB) per thread:
const int64_t minimum_entries_per_thread_for_copies = 262144;

int fn_number_of_entry_blocks(const int64_t n_entries, const int n_threads_for_update_loops) {
        const int64_t largest_number_of_blocks_by_entries = n_entries / minimum_entries_per_thread_for_copies;
        return static_cast<int>(std::max(static_cast<int64_t>(1),
                                         std::min(static_cast<int64_t>(n_threads_for_update_loops),
                                                  largest_number_of_blocks_by_entries)));
}

//// the per-entry arithmetic of the bootstrap draws (a logarithm and a square root per pair, the perturbed
//// inputs), at least minimum_entries_per_thread_for_draw_arithmetic entries per thread:
const int64_t minimum_entries_per_thread_for_draw_arithmetic = 16384;

//// the bootstrap generates the next draw's normals on another thread, whilst the current draw is fitted,
//// only when a draw has at least minimum_normals_per_draw_for_generation_on_another_thread normals; below it,
//// starting and joining a thread for every draw costs more than generating the draw, so the next draw is
//// generated on the calling thread after the current draw's fit (the same normals in the same order):
const int64_t minimum_normals_per_draw_for_generation_on_another_thread = 16384;

int fn_number_of_draw_arithmetic_blocks(const int64_t n_entries, const int n_threads_for_update_loops) {
        const int64_t largest_number_of_blocks_by_entries = n_entries /
                                                            minimum_entries_per_thread_for_draw_arithmetic;
        return static_cast<int>(std::max(static_cast<int64_t>(1),
                                         std::min(static_cast<int64_t>(n_threads_for_update_loops),
                                                  largest_number_of_blocks_by_entries)));
}

void fn_copy_in_blocks(const double *source, double *destination, const int64_t n_entries,
                       const int n_threads_for_update_loops) {
        fn_for_blocks(n_entries, fn_number_of_entry_blocks(n_entries, n_threads_for_update_loops),
                      [&](const int, const int64_t begin, const int64_t end) {
                std::copy(source + begin, source + end, destination + begin);
        });
}

void fn_scale_in_blocks(std::vector<double> &values, const double factor, const int n_threads_for_update_loops) {
        double *data = values.data();
        const int64_t n_entries = static_cast<int64_t>(values.size());
        fn_for_blocks(n_entries, fn_number_of_entry_blocks(n_entries, n_threads_for_update_loops),
                      [&](const int, const int64_t begin, const int64_t end) {
                for (int64_t i = begin; i < end; i++) data[i] *= factor;
        });
}

//// the scaling of the columns of an n_rows x n_bins matrix, except those of bins that never had data (their
//// count is 0, and every entry of their column is 0, which the scaling leaves 0), entry by entry as
//// fn_scale_in_blocks(), the rows over threads:
void fn_scale_columns_of_bins_with_data(std::vector<double> &values, const int n_rows,
                                        const std::vector<double> &counts, const double factor,
                                        const int n_threads_for_update_loops) {
        double *data = values.data();
        const int n_bins = static_cast<int>(counts.size());
        fn_for_blocks(n_rows, fn_number_of_row_blocks(n_rows, n_threads_for_update_loops),
                      [&](const int, const int row_begin, const int row_end) {
                for (int bin = 0; bin < n_bins; bin++) {
                        if (counts[bin] == 0.0) continue;
                        double *column = data + static_cast<size_t>(bin) * n_rows;
                        for (int r = row_begin; r < row_end; r++) column[r] *= factor;
                }
        });
}

//// ---- the length-bin states (column-major matrices, as R stores them):
struct Bins {
        double reference_length = 0.0;
        std::vector<double> edges;                 // bin edges in doublings
        int n_rows = 0, n_bins = 0;
        std::vector<double> sums;                  // n_rows x n_bins, the forgotten sums of normalised squared jumps
        std::vector<double> counts;                // n_bins
        std::vector<double> length_sums;           // n_bins (trajectory or executed lengths)
        std::vector<double> largest_length;        // n_bins
        double n_updates = 0.0;
        std::vector<double> frequencies;           // flow only
        std::vector<double> cosine_sums;           // flow only: n_bins x n_frequencies
};

std::vector<double> fn_as_vector(SEXP x) {
        Rcpp::NumericVector v(x);
        return std::vector<double>(v.begin(), v.end());
}

Bins fn_read_bins(const Rcpp::List &bins_list, const bool flow, const int n_threads_for_update_loops = 1) {
        Bins b;
        b.reference_length = Rcpp::as<double>(bins_list["reference_length"]);
        b.edges = fn_as_vector(bins_list["bin_edges_in_doublings"]);
        Rcpp::NumericMatrix sums = bins_list["sum_of_normalised_squared_jumps"];
        b.n_rows = sums.nrow();
        b.n_bins = sums.ncol();
        // b.sums = std::vector<double>(sums.begin(), sums.end());
        b.sums.resize(sums.size());
        fn_copy_in_blocks(sums.begin(), b.sums.data(), static_cast<int64_t>(sums.size()),
                          n_threads_for_update_loops);
        b.counts = fn_as_vector(bins_list["sum_of_chain_counts"]);
        b.n_updates = Rcpp::as<double>(bins_list["n_updates"]);
        if (flow) {
                b.length_sums = fn_as_vector(bins_list["sum_of_executed_lengths"]);
                b.largest_length = fn_as_vector(bins_list["largest_executed_length_in_bin"]);
                b.frequencies = fn_as_vector(bins_list["frequencies"]);
                b.cosine_sums = fn_as_vector(bins_list["sum_of_cosines"]);
        } else {
                b.length_sums = fn_as_vector(bins_list["sum_of_trajectory_lengths"]);
                b.largest_length = fn_as_vector(bins_list["largest_trajectory_length_in_bin"]);
        }
        return b;
}

Rcpp::NumericMatrix fn_matrix(const std::vector<double> &values, const int n_rows, const int n_cols,
                              const int n_threads_for_update_loops = 1) {
        // Rcpp::NumericMatrix m(n_rows, n_cols);
        Rcpp::NumericMatrix m(Rcpp::no_init(n_rows, n_cols));   // (not zero-filled: every entry is written below)
        // std::copy(values.begin(), values.end(), m.begin());
        // (the R memory is taken here, on the R thread; the threads only copy into it)
        fn_copy_in_blocks(values.data(), m.begin(), static_cast<int64_t>(values.size()),
                          n_threads_for_update_loops);
        if (values.size() < static_cast<size_t>(m.size())) std::fill(m.begin() + values.size(), m.end(), 0.0);
        return m;
}

Rcpp::List fn_write_bins(const Bins &b, const bool flow, const int n_threads_for_update_loops = 1) {
        if (flow) {
                return Rcpp::List::create(
                      Rcpp::Named("reference_length")                = b.reference_length,
                      Rcpp::Named("bin_edges_in_doublings")          = Rcpp::wrap(b.edges),
                      Rcpp::Named("frequencies")                     = Rcpp::wrap(b.frequencies),
                      Rcpp::Named("sum_of_normalised_squared_jumps") = fn_matrix(b.sums, b.n_rows, b.n_bins,
                                                                                 n_threads_for_update_loops),
                      Rcpp::Named("sum_of_chain_counts")             = Rcpp::wrap(b.counts),
                      Rcpp::Named("sum_of_executed_lengths")         = Rcpp::wrap(b.length_sums),
                      Rcpp::Named("largest_executed_length_in_bin")  = Rcpp::wrap(b.largest_length),
                      Rcpp::Named("sum_of_cosines")                  = fn_matrix(b.cosine_sums, b.n_bins,
                                                                                 static_cast<int>(b.frequencies.size())),
                      Rcpp::Named("n_updates")                       = b.n_updates);
        }
        return Rcpp::List::create(
              Rcpp::Named("reference_length")                 = b.reference_length,
              Rcpp::Named("bin_edges_in_doublings")           = Rcpp::wrap(b.edges),
              Rcpp::Named("sum_of_normalised_squared_jumps")  = fn_matrix(b.sums, b.n_rows, b.n_bins,
                                                                          n_threads_for_update_loops),
              Rcpp::Named("sum_of_chain_counts")              = Rcpp::wrap(b.counts),
              Rcpp::Named("sum_of_trajectory_lengths")        = Rcpp::wrap(b.length_sums),
              Rcpp::Named("largest_trajectory_length_in_bin") = Rcpp::wrap(b.largest_length),
              Rcpp::Named("second_moment_moving_average")     = R_NilValue,
              Rcpp::Named("fourth_moment_moving_average")     = R_NilValue,
              Rcpp::Named("n_updates")                        = b.n_updates);
}

//// ---- fn_length_response_initialise_state / fn_spectral_ESS_initialise_flow_state (bins a quarter of a
////      doubling wide from -10 to 6 doublings; the flow's frequencies a quarter of a doubling apart):
Bins fn_initialise_bins(const int n_rows, const double reference_length, const bool flow, const double eps_at_start) {
        Bins b;
        b.reference_length = reference_length;
        b.edges = fn_seq_by(-10.0, 6.0, 0.25);
        b.n_rows = n_rows;
        b.n_bins = static_cast<int>(b.edges.size()) + 1;
        b.sums.assign(static_cast<size_t>(n_rows) * b.n_bins, 0.0);
        b.counts.assign(b.n_bins, 0.0);
        b.length_sums.assign(b.n_bins, 0.0);
        b.largest_length.assign(b.n_bins, 0.0);
        b.n_updates = 0.0;
        if (flow) {
                const double lowest_frequency = PI_VALUE / (2.0 * std::pow(2.0, 6.0) * reference_length);
                const double highest_frequency = 4.0 * PI_VALUE / eps_at_start;
                std::vector<double> exponents = fn_seq_by(std::log2(lowest_frequency), std::log2(highest_frequency), 0.25);
                for (double e : exponents) b.frequencies.push_back(std::pow(2.0, e));
                b.cosine_sums.assign(static_cast<size_t>(b.n_bins) * b.frequencies.size(), 0.0);
        }
        return b;
}

//// ---- forgotten sums of squares per row and bin (fn_spectral_ESS_add_bin_sum_of_squares):
void fn_add_sum_of_squares(std::vector<double> &sum_of_squares, const int n_rows, const std::vector<double> &values,
                           const std::vector<int> &bin_index, const std::vector<int> &chains,
                           const double forgetting_factor, const int n_threads_for_update_loops = 1,
                           const std::vector<double> *counts_before_update = nullptr) {
        // for (double &v : sum_of_squares) v *= forgetting_factor;
        // fn_scale_in_blocks(sum_of_squares, forgetting_factor, n_threads_for_update_loops);
        if (counts_before_update != nullptr) {
                fn_scale_columns_of_bins_with_data(sum_of_squares, n_rows, *counts_before_update,
                                                   forgetting_factor, n_threads_for_update_loops);
        } else {
                fn_scale_in_blocks(sum_of_squares, forgetting_factor, n_threads_for_update_loops);
        }
        // for (int chain : chains) {
        //         const int bin = bin_index[chain];
        //         for (int r = 0; r < n_rows; r++) {
        //                 const double v = values[r + static_cast<size_t>(chain) * n_rows];
        //                 sum_of_squares[r + static_cast<size_t>(bin) * n_rows] += v * v;
        //         }
        // }
        // (the rows in blocks over threads; each entry's adds stay in chain order)
        const int n_blocks = fn_number_of_row_blocks(n_rows, n_threads_for_update_loops);
        fn_for_blocks(n_rows, n_blocks, [&](const int, const int row_begin, const int row_end) {
                for (int chain : chains) {
                        const int bin = bin_index[chain];
                        for (int r = row_begin; r < row_end; r++) {
                                const double v = values[r + static_cast<size_t>(chain) * n_rows];
                                sum_of_squares[r + static_cast<size_t>(bin) * n_rows] += v * v;
                        }
                }
        });
}

//// ---- the objective's inputs that do not change within an update (the candidates and their integral weights):
struct ObjectiveInputs {
        int n_rows = 0;
        std::vector<double> candidate_tau, integral_weights, expected_leapfrog_steps;   // weights: n_nodes x n_candidates
        int n_nodes = 0;
        double eps_now = 0.0, acceptance_moving_average = 1.0, soft_minimum_power = 8.0, minimum_bin_count = 2.0;
};

//// ---- the cosine-mixture fit of the flow bins (fn_spectral_ESS_fit_flow_weights), in two parts: what depends
////      only on the bins' lengths and counts (the frequencies used, the design, A^T A and, per candidate, each
////      frequency's variance inflation), computed once per update, and the weights for given flow sums (one NNLS
////      per row), computed for the estimates and again for every bootstrap draw (whose perturbations change only
////      the flow sums, so everything prepared stays the same):
struct PreparedFit {
        bool usable = false;
        std::vector<int> usable_bins;
        int n_used = 0, n_usable = 0;
        std::vector<double> frequencies, counts, design, AtA, variance_inflation;   // VI: n_used x n_candidates
        double equality_weight = 0.0, lowest_resolvable_frequency = NA_DOUBLE;
        int first_frequency_index = 0;   // (of frequencies[0] among the flow state's frequencies)
        uint64_t identity = 0;           // (a number of its own, for the factors stored with its A^T A)
};

std::atomic<uint64_t> number_of_prepared_fits(0);

PreparedFit fn_prepare_fit(const Bins &flow, const ObjectiveInputs &in) {
        PreparedFit prep;
        for (int bin = 0; bin < flow.n_bins; bin++) if (flow.counts[bin] >= in.minimum_bin_count) prep.usable_bins.push_back(bin);
        if (prep.usable_bins.size() < 3) return prep;
        double longest = -std::numeric_limits<double>::infinity();
        for (int bin : prep.usable_bins) longest = std::max(longest, flow.largest_length[bin]);
        prep.lowest_resolvable_frequency = PI_VALUE / (2.0 * longest);
        const double highest_frequency_used = PI_VALUE / in.eps_now;
        const int n_frequencies = static_cast<int>(flow.frequencies.size());
        int first = -1;
        for (int k = 0; k < n_frequencies; k++) {
                if (flow.frequencies[k] >= prep.lowest_resolvable_frequency) {
                        first = k;
                        break;
                }
        }
        if (first < 0) first = n_frequencies - 1;
        int last = first;
        for (int k = 0; k < n_frequencies; k++) {
                if (flow.frequencies[k] <= highest_frequency_used * (1.0 + 1e-9)) last = std::max(last, k);
        }
        prep.n_used = last - first + 1;
        prep.n_usable = static_cast<int>(prep.usable_bins.size());
        prep.frequencies.assign(flow.frequencies.begin() + first, flow.frequencies.begin() + last + 1);
        prep.first_frequency_index = first;
        prep.counts.resize(prep.n_usable);
        prep.design.resize(static_cast<size_t>(prep.n_usable) * prep.n_used);
        double sum_of_counts = 0.0;
        for (int i = 0; i < prep.n_usable; i++) {
                prep.counts[i] = flow.counts[prep.usable_bins[i]];
                sum_of_counts += prep.counts[i];
                for (int u = 0; u < prep.n_used; u++) {
                        prep.design[i + static_cast<size_t>(u) * prep.n_usable] =
                              1.0 - flow.cosine_sums[prep.usable_bins[i] + static_cast<size_t>(first + u) * flow.n_bins] / prep.counts[i];
                }
        }
        prep.equality_weight = 1e4 * sum_of_counts;
        prep.AtA.resize(static_cast<size_t>(prep.n_used) * prep.n_used);
        for (int a = 0; a < prep.n_used; a++) {
                for (int b = 0; b < prep.n_used; b++) {
                        double t = 0.0;
                        for (int i = 0; i < prep.n_usable; i++) {
                                t += prep.design[i + static_cast<size_t>(a) * prep.n_usable] *
                                     (prep.counts[i] * prep.design[i + static_cast<size_t>(b) * prep.n_usable]);
                        }
                        prep.AtA[a * prep.n_used + b] = t + prep.equality_weight;
                }
        }
        const int n_candidates = static_cast<int>(in.candidate_tau.size());
        prep.variance_inflation.resize(static_cast<size_t>(prep.n_used) * n_candidates);
        // (8 Oct 2026) fn_expected_cosine_under_jitter() for every frequency and candidate, with cos(n x) and
        // the cosine sum over k = 1, ..., n - 1 advanced by rotation over n = 1, 2, ... (re-anchored on the exact
        // cosine and sine every 32 steps), in place of the closed forms; the candidates increase, so one pass
        // over n serves them all (beyond 4096 steps, the closed form):
        for (int u = 0; u < prep.n_used; u++) {
                const double omega = prep.frequencies[u];
                const double x = omega * in.eps_now;
                const double cosine_x = std::cos(x), sine_x = std::sin(x);
                double cosine_n_x = 1.0, sine_n_x = 0.0, sum_of_cosines = 0.0;   // at n = 0
                int n = 0;
                for (int c = 0; c < n_candidates; c++) {
                        const double a = 2.0 * in.candidate_tau[c] / in.eps_now;
                        double expected_cosine;
                        if (a <= 1.0) {
                                expected_cosine = cosine_x;
                        } else if (a > 4096.0) {
                                expected_cosine = fn_expected_cosine_under_jitter(omega, in.candidate_tau[c],
                                                                                  in.eps_now);
                        } else {
                                const int n_target = static_cast<int>(std::ceil(a));
                                while (n < n_target) {
                                        // (the sum, over k = 1, ..., n after this line)
                                        if (n >= 1) sum_of_cosines += cosine_n_x;
                                        n++;
                                        if (n % 32 == 0) {
                                                cosine_n_x = std::cos(n * x);
                                                sine_n_x = std::sin(n * x);
                                        } else {
                                                const double cosine_next = cosine_n_x * cosine_x -
                                                                           sine_n_x * sine_x;
                                                sine_n_x = sine_n_x * cosine_x + cosine_n_x * sine_x;
                                                cosine_n_x = cosine_next;
                                        }
                                }
                                expected_cosine = (sum_of_cosines + (a - n + 1.0) * cosine_n_x) / a;
                        }
                        double lag_one = 1.0 - in.acceptance_moving_average * (1.0 - expected_cosine);
                        lag_one = fn_min_na(lag_one, 1.0 - 1e-9);
                        prep.variance_inflation[u + static_cast<size_t>(c) * prep.n_used] = (1.0 + lag_one) / (1.0 - lag_one);
                }
        }
        prep.usable = true;
        prep.identity = ++number_of_prepared_fits;
        return prep;
}

//// the weights (n_used x n_rows, column-major, each column summing to 1) for given flow sums; warm_start = a
//// previous weights matrix whose supports start each row's NNLS:
void fn_fit_weights(const PreparedFit &prep, const std::vector<double> &flow_sums, const int n_rows,
                    std::vector<double> &weights, const std::vector<double> *warm_start) {
        const int n_used = prep.n_used;
        std::vector<double> Atb(static_cast<size_t>(n_used) * n_rows);
        double largest_absolute = 0.0;
        // (8 Oct 2026) each bin's count x mean once per row (the same values, in the same order, as computing
        // them inside the loop over the frequencies)
        std::vector<double> count_times_bin_mean(prep.n_usable);
        for (int r = 0; r < n_rows; r++) {
                for (int i = 0; i < prep.n_usable; i++) {
                        const double bin_mean = flow_sums[r + static_cast<size_t>(prep.usable_bins[i]) * n_rows] /
                                                prep.counts[i];
                        count_times_bin_mean[i] = prep.counts[i] * bin_mean;
                }
                for (int a = 0; a < n_used; a++) {
                        double t = 0.0;
                        const double *design_column = prep.design.data() + static_cast<size_t>(a) * prep.n_usable;
                        for (int i = 0; i < prep.n_usable; i++) {
                                t += design_column[i] * count_times_bin_mean[i];
                        }
                        Atb[a + static_cast<size_t>(r) * n_used] = t + prep.equality_weight;
                        largest_absolute = std::max(largest_absolute, std::fabs(Atb[a + static_cast<size_t>(r) * n_used]));
                }
        }
        const double tolerance = 1e-12 * largest_absolute;
        weights.assign(static_cast<size_t>(n_used) * n_rows, 0.0);
        for (int r = 0; r < n_rows; r++) {
                std::vector<double> column(Atb.begin() + static_cast<size_t>(r) * n_used,
                                           Atb.begin() + static_cast<size_t>(r + 1) * n_used);
                std::vector<double> start;
                if (warm_start != nullptr) {
                        start.assign(warm_start->begin() + static_cast<size_t>(r) * n_used,
                                     warm_start->begin() + static_cast<size_t>(r + 1) * n_used);
                }
                std::vector<double> w = fn_nnls_one_column(prep.AtA, column, n_used, tolerance,
                                                           warm_start != nullptr ? &start : nullptr);
                double column_sum = 0.0;
                for (double v : w) column_sum += v;
                column_sum = std::max(column_sum, 1e-300);
                for (int u = 0; u < n_used; u++) weights[u + static_cast<size_t>(r) * n_used] = w[u] / column_sum;
        }
}

/* (the serial version, before the rows were spread over threads:)
//// (8 Oct 2026) fn_fit_weights() with the workspace buffers (no allocations per row or per bootstrap draw):
struct FitWorkspace {
        std::vector<double> Atb, count_times_bin_mean, column_solution;
        NNLSWorkspace nnls;
};
void fn_fit_weights_in_workspace(const PreparedFit &prep, const std::vector<double> &flow_sums, const int n_rows,
                                 std::vector<double> &weights, const std::vector<double> *warm_start,
                                 FitWorkspace &fit_workspace) {
        const int n_used = prep.n_used;
        fit_workspace.Atb.resize(static_cast<size_t>(n_used) * n_rows);
        fit_workspace.count_times_bin_mean.resize(prep.n_usable);
        fit_workspace.column_solution.resize(n_used);
        fit_workspace.nnls.resize(n_used);
        double *Atb = fit_workspace.Atb.data();
        double *count_times_bin_mean = fit_workspace.count_times_bin_mean.data();
        double largest_absolute = 0.0;
        for (int r = 0; r < n_rows; r++) {
                for (int i = 0; i < prep.n_usable; i++) {
                        const double bin_mean = flow_sums[r + static_cast<size_t>(prep.usable_bins[i]) * n_rows] /
                                                prep.counts[i];
                        count_times_bin_mean[i] = prep.counts[i] * bin_mean;
                }
                for (int a = 0; a < n_used; a++) {
                        double t = 0.0;
                        const double *design_column = prep.design.data() + static_cast<size_t>(a) * prep.n_usable;
                        for (int i = 0; i < prep.n_usable; i++) t += design_column[i] * count_times_bin_mean[i];
                        const size_t index = a + static_cast<size_t>(r) * n_used;
                        Atb[index] = t + prep.equality_weight;
                        largest_absolute = std::max(largest_absolute, std::fabs(Atb[index]));
                }
        }
        const double tolerance = 1e-12 * largest_absolute;
        weights.assign(static_cast<size_t>(n_used) * n_rows, 0.0);
        double *column_solution = fit_workspace.column_solution.data();
        for (int r = 0; r < n_rows; r++) {
                const double *row_warm_start = warm_start != nullptr ?
                                               warm_start->data() + static_cast<size_t>(r) * n_used : nullptr;
                fn_nnls_one_column_in_workspace(prep.AtA, Atb + static_cast<size_t>(r) * n_used, n_used,
                                                tolerance, row_warm_start, fit_workspace.nnls, column_solution);
                double column_sum = 0.0;
                for (int u = 0; u < n_used; u++) column_sum += column_solution[u];
                column_sum = std::max(column_sum, 1e-300);
                for (int u = 0; u < n_used; u++) {
                        weights[u + static_cast<size_t>(r) * n_used] = column_solution[u] / column_sum;
                }
        }
}
*/

//// fn_fit_weights() with the workspace buffers (no allocations per row or per bootstrap draw), with the rows
//// spread over n_threads_for_update_loops threads (each block of rows with its own buffers):
struct FitWorkspace {
        std::vector<double> Atb;
        std::vector<std::vector<double>> count_times_bin_mean, column_solution;   // one per block of rows
        std::vector<std::vector<double>> column_solution_start;                    // one per block of rows
        std::vector<NNLSWorkspace> nnls;                                          // one per block of rows
        std::vector<double> largest_absolute;                                     // one per block of rows
};
void fn_fit_weights_in_workspace(const PreparedFit &prep, const std::vector<double> &flow_sums, const int n_rows,
                                 std::vector<double> &weights, const std::vector<double> *warm_start,
                                 FitWorkspace &fit_workspace, const int n_threads_for_update_loops,
                                 const std::vector<char> *support_to_try_first = nullptr) {
        const int n_used = prep.n_used;
        const int n_blocks = fn_number_of_row_blocks(n_rows, n_threads_for_update_loops);
        fit_workspace.Atb.resize(static_cast<size_t>(n_used) * n_rows);
        fit_workspace.count_times_bin_mean.resize(n_blocks);
        fit_workspace.column_solution.resize(n_blocks);
        fit_workspace.column_solution_start.resize(n_blocks);
        fit_workspace.nnls.resize(n_blocks);
        fit_workspace.largest_absolute.assign(n_blocks, 0.0);
        for (int block = 0; block < n_blocks; block++) {
                fit_workspace.count_times_bin_mean[block].resize(prep.n_usable);
                fit_workspace.column_solution[block].resize(n_used);
                fit_workspace.column_solution_start[block].resize(n_used);
                fit_workspace.nnls[block].resize(n_used);
                fit_workspace.nnls[block].fit_identity_now = prep.identity;
        }
        double *Atb = fit_workspace.Atb.data();
        // ---- A^T b of every row, and the largest |A^T b| of each block of rows:
        fn_for_blocks(n_rows, n_blocks, [&](const int block, const int row_begin, const int row_end) {
                double *count_times_bin_mean = fit_workspace.count_times_bin_mean[block].data();
                double largest_absolute = 0.0;
                for (int r = row_begin; r < row_end; r++) {
                        for (int i = 0; i < prep.n_usable; i++) {
                                const double bin_mean = flow_sums[r + static_cast<size_t>(prep.usable_bins[i]) *
                                                                      n_rows] / prep.counts[i];
                                count_times_bin_mean[i] = prep.counts[i] * bin_mean;
                        }
                        for (int a = 0; a < n_used; a++) {
                                double t = 0.0;
                                const double *design_column = prep.design.data() +
                                                              static_cast<size_t>(a) * prep.n_usable;
                                for (int i = 0; i < prep.n_usable; i++) {
                                        t += design_column[i] * count_times_bin_mean[i];
                                }
                                const size_t index = a + static_cast<size_t>(r) * n_used;
                                Atb[index] = t + prep.equality_weight;
                                largest_absolute = std::max(largest_absolute, std::fabs(Atb[index]));
                        }
                }
                fit_workspace.largest_absolute[block] = largest_absolute;
        });
        // ---- the tolerance shared by all rows (the largest over the blocks is the largest over all rows):
        double largest_absolute = 0.0;
        for (int block = 0; block < n_blocks; block++) {
                largest_absolute = std::max(largest_absolute, fit_workspace.largest_absolute[block]);
        }
        const double tolerance = 1e-12 * largest_absolute;
        // ---- each row's NNLS (every entry of the weights is written below, so they are not zero-filled first):
        // weights.assign(static_cast<size_t>(n_used) * n_rows, 0.0);
        weights.resize(static_cast<size_t>(n_used) * n_rows);
        fn_for_blocks(n_rows, n_blocks, [&](const int block, const int row_begin, const int row_end) {
                double *column_solution = fit_workspace.column_solution[block].data();
                for (int r = row_begin; r < row_end; r++) {
                        const double *row_warm_start = warm_start != nullptr ?
                                                       warm_start->data() + static_cast<size_t>(r) * n_used :
                                                       nullptr;
                        // fn_nnls_one_column_in_workspace(prep.AtA, Atb + static_cast<size_t>(r) * n_used,
                        //                                 n_used, tolerance, row_warm_start,
                        //                                 fit_workspace.nnls[block], column_solution);
                        // (the row's support of the previous update first, when given: see
                        //  fn_nnls_on_support_if_solution_in_workspace())
                        const double *row_Atb = Atb + static_cast<size_t>(r) * n_used;
                        NNLSWorkspace &nnls_workspace = fit_workspace.nnls[block];
                        const bool solution_on_support = support_to_try_first != nullptr &&
                              fn_nnls_on_support_if_solution_in_workspace(prep.AtA, row_Atb, n_used, tolerance,
                                                                          support_to_try_first->data() +
                                                                          static_cast<size_t>(r) * n_used,
                                                                          nnls_workspace, column_solution);
                        if (!solution_on_support) {
                                // // (the previous update's support as the warm start: positive entries on it, so
                                // //  the active-set loop starts from it; the NNLS solution is the same, being
                                // //  unique)
                                // const double *row_start_values = row_warm_start;
                                // double *previous_support_start =
                                //       fit_workspace.column_solution_start[block].data();
                                // if (row_start_values == nullptr && support_to_try_first != nullptr) {
                                //         const char *row_support = support_to_try_first->data() +
                                //                                   static_cast<size_t>(r) * n_used;
                                //         bool any = false;
                                //         for (int u = 0; u < n_used; u++) {
                                //                 previous_support_start[u] = row_support[u] ? 1.0 : 0.0;
                                //                 any = any || row_support[u];
                                //         }
                                //         if (any) row_start_values = previous_support_start;
                                // }
                                // fn_nnls_one_column_in_workspace(prep.AtA, row_Atb, n_used, tolerance,
                                //                                 row_start_values, nnls_workspace,
                                //                                 column_solution);
                                // (from the start the code without the previous support uses: the active-set
                                //  loop stops within its tolerance, so a start from the previous support could
                                //  stop at a slightly different solution, changing the criterion's later digits)
                                fn_nnls_one_column_in_workspace(prep.AtA, row_Atb, n_used, tolerance,
                                                                row_warm_start, nnls_workspace,
                                                                column_solution);
                        }
                        double column_sum = 0.0;
                        for (int u = 0; u < n_used; u++) column_sum += column_solution[u];
                        column_sum = std::max(column_sum, 1e-300);
                        for (int u = 0; u < n_used; u++) {
                                weights[u + static_cast<size_t>(r) * n_used] = column_solution[u] / column_sum;
                        }
                }
        });
}

//// ---- the objective (fn_objective of the R code with its defaults) at the candidates `which` (the others NaN),
////      for given quadratic bin means (n_rows x n_nodes) and flow weights:
/* (the serial version, before the candidates were spread over threads:)
void fn_objective_at(const ObjectiveInputs &in, const PreparedFit &prep, const std::vector<double> &quadratic_bin_means,
                     const std::vector<double> &weights, const std::vector<int> &which, std::vector<double> &objective) {
        const int n_rows = in.n_rows;
        const int n_used = prep.n_used;
        objective.assign(in.candidate_tau.size(), NA_DOUBLE);
        std::vector<double> scores(2 * static_cast<size_t>(n_rows));
        for (int c : which) {
                for (int r = 0; r < n_rows; r++) {
                        // the spectral linear score:
                        double integrated_autocorrelation_time = 0.0;
                        for (int u = 0; u < n_used; u++) {
                                integrated_autocorrelation_time += weights[u + static_cast<size_t>(r) * n_used] *
                                                                   prep.variance_inflation[u + static_cast<size_t>(c) * n_used];
                        }
                        scores[r] = 1.0 / integrated_autocorrelation_time;
                        // the quadratic lag-one score:
                        double integral = 0.0;
                        for (int k = 0; k < in.n_nodes; k++) {
                                integral += quadratic_bin_means[r + static_cast<size_t>(k) * n_rows] *
                                            in.integral_weights[k + static_cast<size_t>(c) * in.n_nodes];
                        }
                        double lag_one = 1.0 - integral / (2.0 * in.candidate_tau[c]);
                        lag_one = fn_min_na(fn_max_na(lag_one, -1.0 + 1e-6), 1.0 - 1e-6);
                        scores[n_rows + r] = (1.0 - lag_one) / (1.0 + lag_one);
                }
                // the soft minimum per expected gradient:
                double smallest = std::numeric_limits<double>::infinity();
                bool has_nan = false;
                for (double score : scores) {
                        if (std::isnan(score)) has_nan = true;
                        else smallest = std::min(smallest, score);
                }
                if (has_nan) continue;
                double sum_of_powers = 0.0;
                for (double score : scores) {
                        sum_of_powers += fn_power((1.0 / score) / (1.0 / smallest), in.soft_minimum_power);
                }
                const double soft_minimum = smallest * std::pow(sum_of_powers, -1.0 / in.soft_minimum_power);
                objective[c] = soft_minimum / in.expected_leapfrog_steps[c];
        }
}
*/

void fn_objective_at(const ObjectiveInputs &in, const PreparedFit &prep, const std::vector<double> &quadratic_bin_means,
                     const std::vector<double> &weights, const std::vector<int> &which,
                     std::vector<double> &objective, const int n_threads_for_update_loops = 1) {
        const int n_rows = in.n_rows;
        const int n_used = prep.n_used;
        objective.assign(in.candidate_tau.size(), NA_DOUBLE);
        // (the candidates in blocks over threads, with fewer than minimum_rows_per_thread_for_per_parameter_fits
        //  rows all on this thread; each candidate's scores, soft minimum and objective are computed as in the
        //  serial loop, with each thread's own scores, and each candidate writes only its own entry of the
        //  objective)
        const int n_which = static_cast<int>(which.size());
        const int n_blocks = n_rows < minimum_rows_per_thread_for_per_parameter_fits ? 1 :
                             std::max(1, std::min(n_threads_for_update_loops, n_which));
        fn_for_blocks(n_which, n_blocks, [&](const int, const int which_begin, const int which_end) {
                std::vector<double> scores(2 * static_cast<size_t>(n_rows));
                for (int j = which_begin; j < which_end; j++) {
                        const int c = which[j];
                        for (int r = 0; r < n_rows; r++) {
                                // the spectral linear score:
                                double integrated_autocorrelation_time = 0.0;
                                for (int u = 0; u < n_used; u++) {
                                        integrated_autocorrelation_time +=
                                              weights[u + static_cast<size_t>(r) * n_used] *
                                              prep.variance_inflation[u + static_cast<size_t>(c) * n_used];
                                }
                                scores[r] = 1.0 / integrated_autocorrelation_time;
                                // the quadratic lag-one score:
                                double integral = 0.0;
                                for (int k = 0; k < in.n_nodes; k++) {
                                        integral += quadratic_bin_means[r + static_cast<size_t>(k) * n_rows] *
                                                    in.integral_weights[k + static_cast<size_t>(c) * in.n_nodes];
                                }
                                double lag_one = 1.0 - integral / (2.0 * in.candidate_tau[c]);
                                lag_one = fn_min_na(fn_max_na(lag_one, -1.0 + 1e-6), 1.0 - 1e-6);
                                scores[n_rows + r] = (1.0 - lag_one) / (1.0 + lag_one);
                        }
                        // the soft minimum per expected gradient:
                        double smallest = std::numeric_limits<double>::infinity();
                        bool has_nan = false;
                        for (double score : scores) {
                                if (std::isnan(score)) has_nan = true;
                                else smallest = std::min(smallest, score);
                        }
                        if (has_nan) continue;
                        double sum_of_powers = 0.0;
                        for (double score : scores) {
                                sum_of_powers += fn_power((1.0 / score) / (1.0 / smallest),
                                                          in.soft_minimum_power);
                        }
                        const double soft_minimum = smallest * std::pow(sum_of_powers,
                                                                        -1.0 / in.soft_minimum_power);
                        objective[c] = soft_minimum / in.expected_leapfrog_steps[c];
                }
        });
}

//// ---- the objective at the candidates `which` in one pass over the rows, without storing the scores: the rows
////      in sub-blocks of rows_per_sub_block_of_objective (a fixed size, so the result does not depend on the
////      number of threads), over threads; for each sub-block and candidate, the two scores of its rows (each
////      row's two sums the same operations as in fn_objective_at(), across the rows of the sub-block at once,
////      the weights transposed so that the rows are contiguous), their smallest, and their soft-minimum sum
////      relative to it, the sum over the sub-block of ((1 / score) / (1 / smallest of the sub-block))^power;
////      then, for each candidate, the sub-blocks in order, each sum rescaled to the smallest score so far,
////      ((1 / smallest of the sub-block) / (1 / smallest so far))^power. The same soft minimum as summing over
////      all the scores in turn, up to the rounding of the extra products; a NaN score makes the candidate's
////      objective NaN:
const int rows_per_sub_block_of_objective = 256;

//// one sub-block's part at one candidate: its scores from the two sums of each row, the smallest score, the
//// soft-minimum sum relative to it, and 1 if a score is NaN, into parts[0], parts[1] and parts[2]. The loops
//// are branch-free, so the compiler can use vector instructions across the rows: std::max(x, low) and
//// std::min(x, high) return x when it is NaN, as fn_max_na() and fn_min_na() do with bounds that are never NaN;
//// the smallest score is over eight running minima (the minimum does not depend on the order), and the terms are
//// summed over eight running sums, then the sums in turn:
void fn_soft_minimum_part_of_sub_block(const double *linear_sum, const double *quadratic_sum,
                                       const int n_sub_rows, const double two_tau, const double power,
                                       double *scores, double *terms, double *parts) {
        for (int i = 0; i < n_sub_rows; i++) {
                // the spectral linear score:
                scores[i] = 1.0 / linear_sum[i];
        }
        for (int i = 0; i < n_sub_rows; i++) {
                // the quadratic lag-one score:
                double lag_one = 1.0 - quadratic_sum[i] / two_tau;
                lag_one = std::min(std::max(lag_one, -1.0 + 1e-6), 1.0 - 1e-6);
                scores[n_sub_rows + i] = (1.0 - lag_one) / (1.0 + lag_one);
        }
        const int n_scores = 2 * n_sub_rows;
        int n_nan = 0;
        for (int i = 0; i < n_scores; i++) n_nan += scores[i] != scores[i];
        const bool has_nan = n_nan > 0;
        double smallest_of_lane[8];
        for (int lane = 0; lane < 8; lane++) smallest_of_lane[lane] = std::numeric_limits<double>::infinity();
        int i_scores = 0;
        for (; i_scores + 8 <= n_scores; i_scores += 8) {
                for (int lane = 0; lane < 8; lane++) {
                        smallest_of_lane[lane] = std::min(smallest_of_lane[lane], scores[i_scores + lane]);
                }
        }
        double smallest = std::numeric_limits<double>::infinity();
        for (int lane = 0; lane < 8; lane++) smallest = std::min(smallest, smallest_of_lane[lane]);
        for (; i_scores < n_scores; i_scores++) smallest = std::min(smallest, scores[i_scores]);
        double sum_of_powers = 0.0;
        if (!has_nan) {
                const double inverse_smallest = 1.0 / smallest;
                if (power == 8.0) {
                        for (int i = 0; i < n_scores; i++) {
                                const double x = (1.0 / scores[i]) / inverse_smallest;
                                const double x2 = x * x;
                                const double x4 = x2 * x2;
                                terms[i] = x4 * x4;
                        }
                } else {
                        for (int i = 0; i < n_scores; i++) {
                                terms[i] = std::pow((1.0 / scores[i]) / inverse_smallest, power);
                        }
                }
                double sum_of_lane[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
                int i_terms = 0;
                for (; i_terms + 8 <= n_scores; i_terms += 8) {
                        for (int lane = 0; lane < 8; lane++) sum_of_lane[lane] += terms[i_terms + lane];
                }
                for (int lane = 0; lane < 8; lane++) sum_of_powers += sum_of_lane[lane];
                for (; i_terms < n_scores; i_terms++) sum_of_powers += terms[i_terms];
        }
        parts[0] = smallest;
        parts[1] = sum_of_powers;
        parts[2] = has_nan ? 1.0 : 0.0;
}

void fn_objective_at_in_one_pass(const ObjectiveInputs &in, const PreparedFit &prep,
                                 const std::vector<double> &quadratic_bin_means,
                                 const std::vector<double> &weights,
                                 const std::vector<int> &which, std::vector<double> &objective,
                                 const int n_threads_for_update_loops, std::vector<double> &transposed_weights,
                                 std::vector<double> &sub_block_parts) {
        const int n_rows = in.n_rows;
        const int n_used = prep.n_used;
        const int n_nodes = in.n_nodes;
        const int n_which = static_cast<int>(which.size());
        const double power = in.soft_minimum_power;
        objective.assign(in.candidate_tau.size(), NA_DOUBLE);
        transposed_weights.resize(static_cast<size_t>(n_used) * n_rows);
        const int n_sub_blocks = (n_rows + rows_per_sub_block_of_objective - 1) / rows_per_sub_block_of_objective;
        // (per sub-block and candidate: the smallest score, the sum relative to it, and 1 if a score is NaN)
        sub_block_parts.resize(static_cast<size_t>(n_sub_blocks) * n_which * 3);
        const int n_blocks = fn_number_of_row_blocks(n_rows, n_threads_for_update_loops);
        fn_for_blocks(n_sub_blocks, n_blocks, [&](const int, const int sub_block_begin, const int sub_block_end) {
                alignas(64) double linear_sum[rows_per_sub_block_of_objective];
                alignas(64) double quadratic_sum[rows_per_sub_block_of_objective];
                alignas(64) double scores[2 * rows_per_sub_block_of_objective];
                alignas(64) double terms[2 * rows_per_sub_block_of_objective];
                for (int sub_block = sub_block_begin; sub_block < sub_block_end; sub_block++) {
                        const int row_start = sub_block * rows_per_sub_block_of_objective;
                        const int n_sub_rows = std::min(rows_per_sub_block_of_objective, n_rows - row_start);
                        for (int r = row_start; r < row_start + n_sub_rows; r++) {
                                for (int u = 0; u < n_used; u++) {
                                        transposed_weights[r + static_cast<size_t>(u) * n_rows] =
                                              weights[u + static_cast<size_t>(r) * n_used];
                                }
                        }
                        for (int j = 0; j < n_which; j++) {
                                const int c = which[j];
                                const double *variance_inflation = prep.variance_inflation.data() +
                                                                   static_cast<size_t>(c) * n_used;
                                const double *integral_weights = in.integral_weights.data() +
                                                                 static_cast<size_t>(c) * n_nodes;
                                const double two_tau = 2.0 * in.candidate_tau[c];
                                for (int i = 0; i < n_sub_rows; i++) {
                                        linear_sum[i] = 0.0;
                                        quadratic_sum[i] = 0.0;
                                }
                                for (int u = 0; u < n_used; u++) {
                                        const double v = variance_inflation[u];
                                        const double *column = transposed_weights.data() + row_start +
                                                               static_cast<size_t>(u) * n_rows;
                                        for (int i = 0; i < n_sub_rows; i++) linear_sum[i] += column[i] * v;
                                }
                                for (int k = 0; k < n_nodes; k++) {
                                        const double v = integral_weights[k];
                                        const double *column = quadratic_bin_means.data() + row_start +
                                                               static_cast<size_t>(k) * n_rows;
                                        for (int i = 0; i < n_sub_rows; i++) quadratic_sum[i] += column[i] * v;
                                }
                                double *parts = sub_block_parts.data() +
                                                (static_cast<size_t>(sub_block) * n_which + j) * 3;
                                fn_soft_minimum_part_of_sub_block(linear_sum, quadratic_sum, n_sub_rows, two_tau,
                                                                  power, scores, terms, parts);
                        }
                }
        });
        for (int j = 0; j < n_which; j++) {
                const int c = which[j];
                double smallest = std::numeric_limits<double>::infinity(), sum_of_powers = 0.0;
                bool has_nan = false;
                for (int sub_block = 0; sub_block < n_sub_blocks; sub_block++) {
                        const double *parts = sub_block_parts.data() +
                                              (static_cast<size_t>(sub_block) * n_which + j) * 3;
                        if (parts[2] != 0.0) {
                                has_nan = true;
                                break;
                        }
                        if (parts[0] < smallest) {
                                if (std::isfinite(smallest)) {
                                        sum_of_powers *= fn_power((1.0 / smallest) / (1.0 / parts[0]), power);
                                }
                                sum_of_powers += parts[1];
                                smallest = parts[0];
                        } else {
                                sum_of_powers += parts[1] * fn_power((1.0 / parts[0]) / (1.0 / smallest), power);
                        }
                }
                if (has_nan) continue;
                const double soft_minimum = smallest * std::pow(sum_of_powers, -1.0 / power);
                objective[c] = soft_minimum / in.expected_leapfrog_steps[c];
        }
}

int fn_which_max(const std::vector<double> &x) {
        int best = -1;
        for (int i = 0; i < static_cast<int>(x.size()); i++) {
                if (std::isnan(x[i])) continue;
                if (best < 0 || x[i] > x[best]) best = i;
        }
        return best;
}

//// ---- the update's large buffers, kept with the state between updates (BinMatricesHeldInCpp) so that they are
////      not allocated, and their memory pages mapped, again at every update (each is resized and written in
////      full before it is read):
struct UpdateBuffers {
        FitWorkspace fit_workspace;
        std::vector<double> quadratic_values, linear_values, bin_means, flow_weights, transposed_weights,
                            sub_block_parts, draw_weights;
        std::vector<char> supports_to_try;
        std::vector<double> quadratic_standard_errors, flow_means, flow_standard_errors, perturbed_quadratic[2],
                            perturbed_flow_sums[2], pair_u[2], pair_v[2], pair_q[2], pair_f;
        std::vector<int> flow_usable;
};

//// ---- the four n_rows x n_bins matrices of the state, held in C++ memory between updates: the quadratic and
////      the flow bins' sums of normalised squared jumps and (with the expansion evidence) their sums of squares.
////      The state list returned to R holds NULL in their places, an external pointer to them
////      (bin_matrices_held_in_cpp_between_updates) and bin_matrices_write_count, and each update changes them in
////      place: they are not copied from R and back, and R does not allocate (and collect) them at every update.
////      The write count, in the list and here, increases at every update that changes them, so an older state
////      list of the same matrices, or one saved and read back (whose pointer is then empty), is refused, never
////      used.
////      fn_spectral_LQ_ESSR_state_with_bin_matrices_in_R() returns the state with the matrices in R (for the R
////      version of the criterion and the debug record); the update accepts that form too:
struct BinMatricesHeldInCpp {
        int n_rows = 0;
        double write_count = 0.0;
        bool update_in_progress = false;   // (set during an update; left set if the update stopped part-way)
        // (bins_decay_lazily: each array's true values are its stored values times its decay factor)
        bool stored_sums_decay_lazily = false;
        double decay_of_quadratic_stored_sums = 1.0, decay_of_flow_stored_sums = 1.0;
        // (bins_decay_lazily: each row's sum over the bins of its stored values, kept up to date as values are
        //  added, for choosing the slowest-moving rows without summing the bins at every update)
        std::vector<double> linear_movement_totals, quadratic_movement_totals;
        bool movement_totals_valid = false;
        // (the estimates' supports of the latest update, per row and frequency (n_used x n_rows, 1 = positive
        //  weight), and the index of their first frequency among the flow state's frequencies: each row's NNLS
        //  of the next update tries its support first)
        std::vector<char> estimate_supports;
        UpdateBuffers buffers;
        int estimate_supports_first_frequency_index = -1, estimate_supports_n_used = 0;
        std::vector<double> quadratic_sums, flow_sums, quadratic_sum_of_squares, flow_sum_of_squares;
};

//// (during an update the matrices are moved into its bins, never copied; at every exit from the update, an
////  error included, they are moved back into the held object)
struct MoveBinMatricesBackOnExit {
        BinMatricesHeldInCpp *&held;
        std::vector<double> &quadratic_sums, &flow_sums, &quadratic_sum_of_squares, &flow_sum_of_squares;
        MoveBinMatricesBackOnExit(BinMatricesHeldInCpp *&held_in, std::vector<double> &quadratic_sums_in,
                                  std::vector<double> &flow_sums_in,
                                  std::vector<double> &quadratic_sum_of_squares_in,
                                  std::vector<double> &flow_sum_of_squares_in)
                : held(held_in), quadratic_sums(quadratic_sums_in), flow_sums(flow_sums_in),
                  quadratic_sum_of_squares(quadratic_sum_of_squares_in),
                  flow_sum_of_squares(flow_sum_of_squares_in) {}
        ~MoveBinMatricesBackOnExit() {
                if (held == nullptr) return;
                held->quadratic_sums.swap(quadratic_sums);
                held->flow_sums.swap(flow_sums);
                held->quadratic_sum_of_squares.swap(quadratic_sum_of_squares);
                held->flow_sum_of_squares.swap(flow_sum_of_squares);
        }
};

//// the held object of a state list (NULL if the list holds its matrices in R), after the checks above:
BinMatricesHeldInCpp *fn_held_bin_matrices_of_state(const Rcpp::List &state, const int n_rows) {
        if (!state.containsElementNamed("bin_matrices_held_in_cpp_between_updates")) return nullptr;
        Rcpp::XPtr<BinMatricesHeldInCpp> pointer(
              Rcpp::as<SEXP>(state["bin_matrices_held_in_cpp_between_updates"]));
        BinMatricesHeldInCpp *held = pointer.get();
        if (held == nullptr) {
                Rcpp::stop(std::string("the SpESS-R state's bin matrices were held in the C++ memory of an R ") +
                           "session that has ended (e.g. a state saved and read back); use the state from " +
                           "fn_spectral_LQ_ESSR_state_with_bin_matrices_in_R() for that.");
        }
        if (held->update_in_progress) {
                Rcpp::stop(std::string("an earlier SpESS-R update of this state stopped part-way, so its bin ") +
                           "matrices are refused.");
        }
        if (!(Rcpp::as<double>(state["bin_matrices_write_count"]) == held->write_count)) {
                Rcpp::stop(std::string("this SpESS-R state list is not the latest one of its bin matrices ") +
                           "(held in C++ between updates, and changed since by a later update); use the " +
                           "latest state list.");
        }
        if (n_rows >= 0 && held->n_rows != n_rows) {
                Rcpp::stop(std::string("the SpESS-R state's bin matrices have a different number of rows from ") +
                           "this update's statistics.");
        }
        return held;
}

//// fn_read_bins() without the sums (held in C++), and fn_write_bins() with NULL in their place:
Bins fn_read_bins_without_sums(const Rcpp::List &bins_list, const bool flow, const int n_rows) {
        Bins b;
        b.reference_length = Rcpp::as<double>(bins_list["reference_length"]);
        b.edges = fn_as_vector(bins_list["bin_edges_in_doublings"]);
        b.n_rows = n_rows;
        b.counts = fn_as_vector(bins_list["sum_of_chain_counts"]);
        b.n_bins = static_cast<int>(b.counts.size());
        b.n_updates = Rcpp::as<double>(bins_list["n_updates"]);
        if (flow) {
                b.length_sums = fn_as_vector(bins_list["sum_of_executed_lengths"]);
                b.largest_length = fn_as_vector(bins_list["largest_executed_length_in_bin"]);
                b.frequencies = fn_as_vector(bins_list["frequencies"]);
                b.cosine_sums = fn_as_vector(bins_list["sum_of_cosines"]);
        } else {
                b.length_sums = fn_as_vector(bins_list["sum_of_trajectory_lengths"]);
                b.largest_length = fn_as_vector(bins_list["largest_trajectory_length_in_bin"]);
        }
        return b;
}

Rcpp::List fn_write_bins_without_sums(const Bins &b, const bool flow) {
        if (flow) {
                return Rcpp::List::create(
                      Rcpp::Named("reference_length")                = b.reference_length,
                      Rcpp::Named("bin_edges_in_doublings")          = Rcpp::wrap(b.edges),
                      Rcpp::Named("frequencies")                     = Rcpp::wrap(b.frequencies),
                      Rcpp::Named("sum_of_normalised_squared_jumps") = R_NilValue,
                      Rcpp::Named("sum_of_chain_counts")             = Rcpp::wrap(b.counts),
                      Rcpp::Named("sum_of_executed_lengths")         = Rcpp::wrap(b.length_sums),
                      Rcpp::Named("largest_executed_length_in_bin")  = Rcpp::wrap(b.largest_length),
                      Rcpp::Named("sum_of_cosines")                  = fn_matrix(b.cosine_sums, b.n_bins,
                                                                                 static_cast<int>(b.frequencies.size())),
                      Rcpp::Named("n_updates")                       = b.n_updates);
        }
        return Rcpp::List::create(
              Rcpp::Named("reference_length")                 = b.reference_length,
              Rcpp::Named("bin_edges_in_doublings")           = Rcpp::wrap(b.edges),
              Rcpp::Named("sum_of_normalised_squared_jumps")  = R_NilValue,
              Rcpp::Named("sum_of_chain_counts")              = Rcpp::wrap(b.counts),
              Rcpp::Named("sum_of_trajectory_lengths")        = Rcpp::wrap(b.length_sums),
              Rcpp::Named("largest_trajectory_length_in_bin") = Rcpp::wrap(b.largest_length),
              Rcpp::Named("second_moment_moving_average")     = R_NilValue,
              Rcpp::Named("fourth_moment_moving_average")     = R_NilValue,
              Rcpp::Named("n_updates")                        = b.n_updates);
}

//// (a thread joined at every exit from its scope, an error included)
struct JoinThreadOnExit {
        std::thread &thread;
        explicit JoinThreadOnExit(std::thread &thread_in) : thread(thread_in) {}
        ~JoinThreadOnExit() {
                if (thread.joinable()) thread.join();
        }
};

//// ---- the bins decaying lazily (bins_decay_lazily): instead of multiplying every stored value of a matrix by
////      the forgetting factor at every update, its decay factor is multiplied and the new values are added
////      divided by it (the true values are the stored values times the decay factor); the same arithmetic up to
////      rounding, with only the new values touched. A factor below 1e-150 is moved into the stored values:
void fn_add_sum_of_squares_lazily(std::vector<double> &sum_of_squares, const int n_rows,
                                  const std::vector<double> &values, const std::vector<int> &bin_index,
                                  const std::vector<int> &chains, const double decay) {
        for (int chain : chains) {
                const int bin = bin_index[chain];
                for (int r = 0; r < n_rows; r++) {
                        const double v = values[r + static_cast<size_t>(chain) * n_rows];
                        sum_of_squares[r + static_cast<size_t>(bin) * n_rows] += (v * v) / decay;
                }
        }
}

//// ---- the rows fitted and scored when there are more than largest_number_of_rows_fitted_per_update: half the
////      limit from the rows with the least linear movement (the sum of their normalised squared jumps over the
////      flow bins with data) and half from the rows with the least quadratic movement (the same over the
////      quadratic bins), as these set the soft minimum; a non-finite movement counts as the least. All the bins
////      share their counts and decay factor, so the sums rank the rows as their bin means do:
std::vector<int> fn_slowest_moving_rows_of_movements(const std::vector<double> &linear_movement,
                                                     const std::vector<double> &quadratic_movement,
                                                     const int n_rows,
                                                     const int largest_number_of_rows_fitted_per_update);

std::vector<int> fn_slowest_moving_rows(const Bins &quadratic, const Bins &flow, const int n_rows,
                                        const int largest_number_of_rows_fitted_per_update,
                                        const int n_threads_for_update_loops) {
        std::vector<double> linear_movement(n_rows, 0.0), quadratic_movement(n_rows, 0.0);
        const int n_blocks = fn_number_of_row_blocks(n_rows, n_threads_for_update_loops);
        fn_for_blocks(n_rows, n_blocks, [&](const int, const int row_begin, const int row_end) {
                for (int bin = 0; bin < flow.n_bins; bin++) {
                        if (!(flow.counts[bin] > 0.0)) continue;
                        const double *column = flow.sums.data() + static_cast<size_t>(bin) * n_rows;
                        for (int r = row_begin; r < row_end; r++) linear_movement[r] += column[r];
                }
                for (int bin = 0; bin < quadratic.n_bins; bin++) {
                        if (!(quadratic.counts[bin] > 0.0)) continue;
                        const double *column = quadratic.sums.data() + static_cast<size_t>(bin) * n_rows;
                        for (int r = row_begin; r < row_end; r++) quadratic_movement[r] += column[r];
                }
        });
        return fn_slowest_moving_rows_of_movements(linear_movement, quadratic_movement, n_rows,
                                                   largest_number_of_rows_fitted_per_update);
}

std::vector<int> fn_slowest_moving_rows_of_movements(const std::vector<double> &linear_movement,
                                                     const std::vector<double> &quadratic_movement,
                                                     const int n_rows,
                                                     const int largest_number_of_rows_fitted_per_update) {
        auto fn_least = [n_rows](const std::vector<double> &movement, const int n_wanted) {
                std::vector<int> rows(n_rows);
                for (int r = 0; r < n_rows; r++) rows[r] = r;
                // (a non-finite movement as minus infinity, so it counts as the least)
                std::vector<double> key(n_rows);
                for (int r = 0; r < n_rows; r++) {
                        key[r] = std::isfinite(movement[r]) ? movement[r] :
                                 -std::numeric_limits<double>::infinity();
                }
                auto less_movement = [&key](const int a, const int b) {
                        return key[a] < key[b] || (key[a] == key[b] && a < b);
                };
                const int n_kept = std::min(n_wanted, n_rows);
                std::nth_element(rows.begin(), rows.begin() + n_kept, rows.end(), less_movement);
                rows.resize(n_kept);
                return rows;
        };
        const int n_from_each = std::max(1, largest_number_of_rows_fitted_per_update / 2);
        std::vector<int> chosen = fn_least(linear_movement, n_from_each);
        const std::vector<int> chosen_quadratic = fn_least(quadratic_movement, n_from_each);
        chosen.insert(chosen.end(), chosen_quadratic.begin(), chosen_quadratic.end());
        std::sort(chosen.begin(), chosen.end());
        chosen.erase(std::unique(chosen.begin(), chosen.end()), chosen.end());
        return chosen;
}

//// the bins of the chosen rows only (the true values: the stored values times the decay factor):
Bins fn_bins_of_rows(const Bins &b, const std::vector<int> &rows, const int n_rows, const double decay) {
        Bins view;
        view.reference_length = b.reference_length;
        view.edges = b.edges;
        view.n_bins = b.n_bins;
        view.n_rows = static_cast<int>(rows.size());
        view.counts = b.counts;
        view.length_sums = b.length_sums;
        view.largest_length = b.largest_length;
        view.n_updates = b.n_updates;
        view.frequencies = b.frequencies;
        view.cosine_sums = b.cosine_sums;
        const int m = view.n_rows;
        view.sums.assign(static_cast<size_t>(m) * b.n_bins, 0.0);
        for (int bin = 0; bin < b.n_bins; bin++) {
                if (!(b.counts[bin] > 0.0)) continue;   // (a bin that never had data holds zeros)
                for (int i = 0; i < m; i++) {
                        const double stored = b.sums[rows[i] + static_cast<size_t>(bin) * n_rows];
                        view.sums[i + static_cast<size_t>(bin) * m] = decay == 1.0 ? stored : stored * decay;
                }
        }
        return view;
}

std::vector<double> fn_matrix_rows(const std::vector<double> &values, const std::vector<int> &rows,
                                   const int n_rows, const double decay,
                                   const std::vector<double> &column_counts) {
        std::vector<double> view;
        if (values.empty() || n_rows == 0) return view;
        const int m = static_cast<int>(rows.size());
        const int n_columns = static_cast<int>(values.size() / n_rows);
        view.assign(static_cast<size_t>(m) * n_columns, 0.0);
        for (int column = 0; column < n_columns; column++) {
                // (a bin that never had data holds zeros)
                if (column < static_cast<int>(column_counts.size()) && !(column_counts[column] > 0.0)) continue;
                for (int i = 0; i < m; i++) {
                        const double stored = values[rows[i] + static_cast<size_t>(column) * n_rows];
                        view[i + static_cast<size_t>(column) * m] = decay == 1.0 ? stored : stored * decay;
                }
        }
        return view;
}

//// ---- SpESS-R's per-coordinate statistics (the "LQ_ESSR_spec_bins99_evid_expand" branch of
////      fn_metric_position_criterion() in R_fn_metric_trajectory_adaptation.R) for a diagonal trajectory metric,
////      with the same operations in the same order as the R code: every product and difference rounded on its
////      own (no fused multiply-adds in this function), R's x^4 as pow(x, 4), and colSums() adding each column in
////      long double, so every value is the same number as R's:
#pragma GCC push_options
#pragma GCC optimize ("fp-contract=off")
void fn_spectral_LQ_ESSR_statistics_of_rows(const double *theta_initial, const double *theta_proposed,
                                            const double *mean_initial, const double *metric_vector,
                                            const double *velocity_proposed, const double *tau_values,
                                            const int n_rows, const int n_chains, const int row_begin,
                                            const int row_end, double *linear_jump_squared,
                                            double *quadratic_jump_squared, double *initial_second_moment,
                                            double *initial_fourth_moment, double *gradient_terms) {
        for (int c = 0; c < n_chains; c++) {
                const double t = tau_values[c];
                for (int r = row_begin; r < row_end; r++) {
                        const size_t i = r + static_cast<size_t>(c) * n_rows;
                        const double jump = metric_vector[r] * (theta_proposed[i] - theta_initial[i]);
                        const double proposed = metric_vector[r] * (theta_proposed[i] - mean_initial[r]);
                        // (the start and the velocity in metric coordinates, as fn_apply_trajectory_metric())
                        const double start = metric_vector[r] * (theta_initial[i] - mean_initial[r]);
                        const double velocity = metric_vector[r] * velocity_proposed[i];
                        const double proposed_squared = proposed * proposed;
                        const double start_squared = start * start;
                        const double change = proposed_squared - start_squared;
                        linear_jump_squared[i] = jump * jump;
                        quadratic_jump_squared[i] = change * change;
                        initial_second_moment[i] = start_squared;
                        initial_fourth_moment[i] = std::pow(start, 4.0);
                        gradient_terms[i] = 2.0 * jump * velocity * t;
                }
        }
}

double fn_column_sum_in_long_double(const double *column, const int n_rows) {
        long double sum = 0.0;
        for (int r = 0; r < n_rows; r++) sum += column[r];
        return static_cast<double>(sum);
}
#pragma GCC pop_options

}  // namespace

//// (largest_number_of_rows_fitted_per_update: 0 = every row; above it, the slowest-moving rows, see
////  fn_slowest_moving_rows(), are fitted and scored. bins_decay_lazily: TRUE = the four bin matrices decay
////  lazily, see fn_add_sum_of_squares_lazily(). Their defaults leave the update as it was.)
// [[Rcpp::export(rng = false)]]
Rcpp::List fn_spectral_LQ_ESSR_update_cpp(  SEXP state_in,
                                            Rcpp::NumericMatrix linear_jump_squared,
                                            Rcpp::NumericMatrix quadratic_jump_squared,
                                            Rcpp::NumericMatrix initial_second_moment,
                                            Rcpp::NumericMatrix initial_fourth_moment,
                                            const double tau,
                                            Rcpp::NumericVector tau_values,
                                            double eps_used_for_trajectories,
                                            const double eps_now,
                                            Rcpp::NumericVector probabilities,
                                            Rcpp::NumericVector divergences,
                                            const bool use_proposals,
                                            const double learning_rate,
                                            const double schedule_iteration,
                                            const double schedule_length,
                                            const double soft_minimum_power,
                                            const double minimum_bin_count,
                                            const double grid_step_in_doublings,
                                            const double upper_end_zone_in_doublings,
                                            const double bin_forgetting_factor,
                                            const bool expansion_evidence,
                                            const int n_bootstrap_draws,
                                            const int n_threads_for_update_loops = 1,
                                            const int largest_number_of_rows_fitted_per_update = 0,
                                            const bool bins_decay_lazily = false) {

        const int n_rows = linear_jump_squared.nrow();
        const int n_chains = static_cast<int>(probabilities.size());
        if (!(std::isfinite(eps_used_for_trajectories) && eps_used_for_trajectories > 0.0)) eps_used_for_trajectories = eps_now;
        auto at = [n_rows](const Rcpp::NumericMatrix &m, const int r, const int c) { return m[r + static_cast<size_t>(c) * n_rows]; };
        // ---- which chains contribute:
        std::vector<int> binned(n_chains), valid(n_chains);
        std::vector<double> weights(n_chains), executed_lengths(n_chains);
        bool any_binned = false, any_valid = false;
        // (whether each chain's statistics are all finite, the rows over threads)
        const int n_input_blocks = fn_number_of_row_blocks(n_rows, n_threads_for_update_loops);
        std::vector<char> all_finite_of_block_and_chain(static_cast<size_t>(n_input_blocks) * n_chains, 1);
        fn_for_blocks(n_rows, n_input_blocks, [&](const int block, const int row_begin, const int row_end) {
                for (int c = 0; c < n_chains; c++) {
                        for (int r = row_begin; r < row_end; r++) {
                                if (!std::isfinite(at(linear_jump_squared, r, c)) ||
                                    !std::isfinite(at(quadratic_jump_squared, r, c)) ||
                                    !std::isfinite(at(initial_fourth_moment, r, c))) {
                                        const size_t entry = block + static_cast<size_t>(c) * n_input_blocks;
                                        all_finite_of_block_and_chain[entry] = 0;
                                        break;
                                }
                        }
                }
        });
        for (int c = 0; c < n_chains; c++) {
                bool finite_statistics = true;
                // for (int r = 0; r < n_rows; r++) {
                //         if (!std::isfinite(at(linear_jump_squared, r, c)) ||
                //             !std::isfinite(at(quadratic_jump_squared, r, c)) ||
                //             !std::isfinite(at(initial_fourth_moment, r, c))) {
                //                 finite_statistics = false;
                //                 break;
                //         }
                // }
                for (int block = 0; block < n_input_blocks; block++) {
                        if (!all_finite_of_block_and_chain[block + static_cast<size_t>(c) * n_input_blocks]) {
                                finite_statistics = false;
                        }
                }
                binned[c] = std::isfinite(tau_values[c]) && tau_values[c] > 0.0;
                valid[c] = finite_statistics && binned[c] && std::isfinite(divergences[c]) && divergences[c] == 0.0;
                double w = use_proposals ? fn_min_na(1.0, fn_max_na(0.0, probabilities[c])) : 1.0;
                if (!std::isfinite(w) || !valid[c]) w = 0.0;
                weights[c] = w;
                any_binned = any_binned || binned[c];
                any_valid = any_valid || valid[c];
                executed_lengths[c] = eps_used_for_trajectories * std::max(1.0, std::ceil(tau_values[c] / eps_used_for_trajectories));
        }
        Rcpp::List result = Rcpp::List::create(Rcpp::Named("performed") = false, Rcpp::Named("state") = state_in);
        if (!any_binned) return result;
        // ---- the moving averages of the start moments and of the acceptance:
        std::vector<double> second_now(n_rows, 0.0), fourth_now(n_rows, 0.0);
        if (any_valid) {
                int n_valid = 0;
                for (int c = 0; c < n_chains; c++) if (valid[c]) n_valid++;
                fn_for_blocks(n_rows, n_input_blocks,
                      [&](const int, const int row_begin, const int row_end) {
                for (int r = row_begin; r < row_end; r++) {
                        double s2 = 0.0, s4 = 0.0;
                        for (int c = 0; c < n_chains; c++) {
                                if (!valid[c]) continue;
                                s2 += at(initial_second_moment, r, c);
                                s4 += at(initial_fourth_moment, r, c);
                        }
                        second_now[r] = s2 / n_valid;
                        fourth_now[r] = s4 / n_valid;
                }
                });
        }
        double acceptance_now = 1.0;
        if (use_proposals) {
                double s = 0.0;
                int n = 0;
                for (int c = 0; c < n_chains; c++) {
                        if (binned[c] && std::isfinite(probabilities[c])) {
                                s += std::min(1.0, std::max(0.0, static_cast<double>(probabilities[c])));
                                n++;
                        }
                }
                if (n > 0) acceptance_now = s / n;
        }
        // ---- the state (new, or the one carried between updates):
        Bins quadratic, flow;
        std::vector<double> second_ma, fourth_ma, quadratic_sum_of_squares, flow_sum_of_squares;
        double acceptance_ma = NA_DOUBLE;
        bool has_acceptance_ma = false;
        // (the bin matrices held in C++ between updates: moved into the bins below, and back at every exit)
        BinMatricesHeldInCpp *held = nullptr;
        Rcpp::RObject held_pointer;
        MoveBinMatricesBackOnExit move_bin_matrices_back_on_exit(held, quadratic.sums, flow.sums,
                                                                  quadratic_sum_of_squares, flow_sum_of_squares);
        if (Rf_isNull(state_in)) {
                if (!any_valid) return result;
                double mean_tau = 0.0, mean_executed = 0.0;
                int n = 0;
                for (int c = 0; c < n_chains; c++) {
                        if (!binned[c]) continue;
                        mean_tau += tau_values[c];
                        mean_executed += executed_lengths[c];
                        n++;
                }
                quadratic = fn_initialise_bins(n_rows, mean_tau / n, false, 0.0);
                flow = fn_initialise_bins(n_rows, mean_executed / n, true, eps_used_for_trajectories);
        } else {
                Rcpp::List state(state_in);
                BinMatricesHeldInCpp *held_by_the_state = fn_held_bin_matrices_of_state(state, n_rows);
                if (held_by_the_state != nullptr) {
                        // (the matrices held in C++: only the small fields are read from the list)
                        quadratic = fn_read_bins_without_sums(Rcpp::as<Rcpp::List>(state["quadratic_state"]),
                                                              false, n_rows);
                        flow = fn_read_bins_without_sums(Rcpp::as<Rcpp::List>(state["flow_state"]), true, n_rows);
                        if (held_by_the_state->quadratic_sums.size() !=
                                  static_cast<size_t>(n_rows) * quadratic.n_bins ||
                            held_by_the_state->flow_sums.size() != static_cast<size_t>(n_rows) * flow.n_bins) {
                                Rcpp::stop("the SpESS-R state's held bin matrices do not match its bins.");
                        }
                        held = held_by_the_state;
                        held_pointer = state["bin_matrices_held_in_cpp_between_updates"];
                        held->update_in_progress = true;
                        quadratic.sums.swap(held->quadratic_sums);
                        flow.sums.swap(held->flow_sums);
                        quadratic_sum_of_squares.swap(held->quadratic_sum_of_squares);
                        flow_sum_of_squares.swap(held->flow_sum_of_squares);
                } else {
                        // quadratic = fn_read_bins(Rcpp::as<Rcpp::List>(state["quadratic_state"]), false);
                        // flow = fn_read_bins(Rcpp::as<Rcpp::List>(state["flow_state"]), true);
                        quadratic = fn_read_bins(Rcpp::as<Rcpp::List>(state["quadratic_state"]), false,
                                                 n_threads_for_update_loops);
                        flow = fn_read_bins(Rcpp::as<Rcpp::List>(state["flow_state"]), true,
                                            n_threads_for_update_loops);
                }
                if (!Rf_isNull(state["second_moment_moving_average"])) second_ma = fn_as_vector(state["second_moment_moving_average"]);
                if (!Rf_isNull(state["fourth_moment_moving_average"])) fourth_ma = fn_as_vector(state["fourth_moment_moving_average"]);
                if (!Rf_isNull(state["acceptance_moving_average"])) {
                        acceptance_ma = Rcpp::as<double>(state["acceptance_moving_average"]);
                        has_acceptance_ma = true;
                }
                if (held == nullptr && state.containsElementNamed("quadratic_sum_of_squares") &&
                    !Rf_isNull(state["quadratic_sum_of_squares"])) {
                        // quadratic_sum_of_squares = fn_as_vector(state["quadratic_sum_of_squares"]);
                        Rcpp::NumericVector saved_sum_of_squares(state["quadratic_sum_of_squares"]);
                        quadratic_sum_of_squares.resize(saved_sum_of_squares.size());
                        fn_copy_in_blocks(saved_sum_of_squares.begin(), quadratic_sum_of_squares.data(),
                                          static_cast<int64_t>(saved_sum_of_squares.size()),
                                          n_threads_for_update_loops);
                }
                if (held == nullptr && state.containsElementNamed("flow_sum_of_squares") &&
                    !Rf_isNull(state["flow_sum_of_squares"])) {
                        // flow_sum_of_squares = fn_as_vector(state["flow_sum_of_squares"]);
                        Rcpp::NumericVector saved_sum_of_squares(state["flow_sum_of_squares"]);
                        flow_sum_of_squares.resize(saved_sum_of_squares.size());
                        fn_copy_in_blocks(saved_sum_of_squares.begin(), flow_sum_of_squares.data(),
                                          static_cast<int64_t>(saved_sum_of_squares.size()),
                                          n_threads_for_update_loops);
                }
        }
        // ---- the decay factors of the stored bin sums (1 when the stored values are the true ones):
        double decay_quadratic = 1.0, decay_flow = 1.0;
        if (held != nullptr && held->stored_sums_decay_lazily) {
                if (bins_decay_lazily) {
                        decay_quadratic = held->decay_of_quadratic_stored_sums;
                        decay_flow = held->decay_of_flow_stored_sums;
                } else {
                        // (no longer lazy: the decay factors are moved into the stored values)
                        fn_scale_in_blocks(quadratic.sums, held->decay_of_quadratic_stored_sums,
                                           n_threads_for_update_loops);
                        if (!quadratic_sum_of_squares.empty()) {
                                fn_scale_in_blocks(quadratic_sum_of_squares, held->decay_of_quadratic_stored_sums,
                                                   n_threads_for_update_loops);
                        }
                        fn_scale_in_blocks(flow.sums, held->decay_of_flow_stored_sums,
                                           n_threads_for_update_loops);
                        if (!flow_sum_of_squares.empty()) {
                                fn_scale_in_blocks(flow_sum_of_squares, held->decay_of_flow_stored_sums,
                                                   n_threads_for_update_loops);
                        }
                }
        }
        if (any_valid) {
                if (second_ma.empty()) {
                        second_ma = second_now;
                        fourth_ma = fourth_now;
                } else {
                        fn_for_blocks(n_rows, n_input_blocks,
                      [&](const int, const int row_begin, const int row_end) {
                        for (int r = row_begin; r < row_end; r++) {
                                second_ma[r] = 0.9 * second_ma[r] + 0.1 * second_now[r];
                                fourth_ma[r] = 0.9 * fourth_ma[r] + 0.1 * fourth_now[r];
                        }
                        });
                }
        }
        acceptance_ma = has_acceptance_ma ? 0.9 * acceptance_ma + 0.1 * acceptance_now : acceptance_now;
        std::vector<double> variance_linear(n_rows), variance_quadratic(n_rows);
        fn_for_blocks(n_rows, n_input_blocks,
                      [&](const int, const int row_begin, const int row_end) {
        for (int r = row_begin; r < row_end; r++) {
                variance_linear[r] = fn_max_na(second_ma[r], 1e-12);
                const double v = fourth_ma[r] - second_ma[r] * second_ma[r];
                variance_quadratic[r] = (std::isfinite(v) && v > 0.0) ? v : 2.0 * variance_linear[r] * variance_linear[r];
        }
        });
        // ---- the bins decaying lazily: one pass over the rows adds each chain's normalised values, divided by
        //      the decay factors, to the bins, their sums of squares and the rows' movement totals (the same
        //      arithmetic as the binning below up to rounding, without its intermediate arrays):
        std::vector<double> linear_movement_totals, quadratic_movement_totals;
        if (bins_decay_lazily) {
                std::vector<int> quadratic_bin_index(n_chains, -1), binned_chains, flow_bin_index(n_chains, -1),
                                 valid_chains;
                for (int c = 0; c < n_chains; c++) {
                        if (binned[c]) {
                                binned_chains.push_back(c);
                                quadratic_bin_index[c] = fn_bin_index(std::log2(tau_values[c] /
                                                                                quadratic.reference_length),
                                                                      quadratic.edges);
                        }
                        if (valid[c]) {
                                valid_chains.push_back(c);
                                flow_bin_index[c] = fn_bin_index(std::log2(executed_lengths[c] /
                                                                           flow.reference_length),
                                                                 flow.edges);
                        }
                }
                // (the rows' movement totals: carried in the held state, or summed from the stored bins)
                if (held != nullptr && held->movement_totals_valid &&
                    held->linear_movement_totals.size() == static_cast<size_t>(n_rows) &&
                    held->quadratic_movement_totals.size() == static_cast<size_t>(n_rows)) {
                        linear_movement_totals.swap(held->linear_movement_totals);
                        quadratic_movement_totals.swap(held->quadratic_movement_totals);
                        held->movement_totals_valid = false;
                } else {
                        linear_movement_totals.assign(n_rows, 0.0);
                        quadratic_movement_totals.assign(n_rows, 0.0);
                        for (int bin = 0; bin < flow.n_bins; bin++) {
                                const double *column = flow.sums.data() + static_cast<size_t>(bin) * n_rows;
                                for (int r = 0; r < n_rows; r++) linear_movement_totals[r] += column[r];
                        }
                        for (int bin = 0; bin < quadratic.n_bins; bin++) {
                                const double *column = quadratic.sums.data() + static_cast<size_t>(bin) * n_rows;
                                for (int r = 0; r < n_rows; r++) quadratic_movement_totals[r] += column[r];
                        }
                }
                decay_quadratic *= bin_forgetting_factor;
                for (double &v : quadratic.counts) v *= bin_forgetting_factor;
                for (double &v : quadratic.length_sums) v *= bin_forgetting_factor;
                for (int c : binned_chains) {
                        const int bin = quadratic_bin_index[c];
                        quadratic.counts[bin] += 1.0;
                        quadratic.length_sums[bin] += tau_values[c];
                        quadratic.largest_length[bin] = std::max(quadratic.largest_length[bin],
                                                                 static_cast<double>(tau_values[c]));
                }
                quadratic.n_updates += 1.0;
                if (any_valid) {
                        decay_flow *= bin_forgetting_factor;
                        for (double &v : flow.counts) v *= bin_forgetting_factor;
                        for (double &v : flow.length_sums) v *= bin_forgetting_factor;
                        for (double &v : flow.cosine_sums) v *= bin_forgetting_factor;
                        const int n_frequencies = static_cast<int>(flow.frequencies.size());
                        for (int c : valid_chains) {
                                const int bin = flow_bin_index[c];
                                flow.counts[bin] += 1.0;
                                flow.length_sums[bin] += executed_lengths[c];
                                flow.largest_length[bin] = std::max(flow.largest_length[bin],
                                                                    executed_lengths[c]);
                                for (int k = 0; k < n_frequencies; k++) {
                                        flow.cosine_sums[bin + static_cast<size_t>(k) * flow.n_bins] +=
                                              std::cos(flow.frequencies[k] * executed_lengths[c]);
                                }
                        }
                        flow.n_updates += 1.0;
                }
                if (expansion_evidence) {
                        if (quadratic_sum_of_squares.empty()) {
                                quadratic_sum_of_squares.assign(quadratic.sums.size(), 0.0);
                        }
                        if (flow_sum_of_squares.empty()) flow_sum_of_squares.assign(flow.sums.size(), 0.0);
                }
                const double inverse_decay_quadratic = 1.0 / decay_quadratic;
                const double inverse_decay_flow = 1.0 / decay_flow;
                const std::vector<int> no_chains;
                const std::vector<int> &flow_chains = any_valid ? valid_chains : no_chains;
                // (the rows in blocks over threads: each row writes only its own entries)
                fn_for_blocks(n_rows, fn_number_of_row_blocks(n_rows, n_threads_for_update_loops),
                              [&](const int, const int row_begin, const int row_end) {
                for (int r = row_begin; r < row_end; r++) {
                        const double quadratic_scale = 1.0 / (2.0 * variance_quadratic[r]);
                        const double linear_scale = 1.0 / (2.0 * variance_linear[r]);
                        for (int c : binned_chains) {
                                double q = at(quadratic_jump_squared, r, c);
                                if (!std::isfinite(q)) q = 0.0;
                                const double value = q * weights[c] * quadratic_scale;
                                const size_t index = r + static_cast<size_t>(quadratic_bin_index[c]) * n_rows;
                                quadratic.sums[index] += value * inverse_decay_quadratic;
                                quadratic_movement_totals[r] += value * inverse_decay_quadratic;
                                if (expansion_evidence) {
                                        quadratic_sum_of_squares[index] +=
                                              value * value * inverse_decay_quadratic;
                                }
                        }
                        for (int c : flow_chains) {
                                double l = at(linear_jump_squared, r, c);
                                if (!std::isfinite(l)) l = 0.0;
                                const double value = l * linear_scale;
                                const size_t index = r + static_cast<size_t>(flow_bin_index[c]) * n_rows;
                                flow.sums[index] += value * inverse_decay_flow;
                                linear_movement_totals[r] += value * inverse_decay_flow;
                                if (expansion_evidence) {
                                        flow_sum_of_squares[index] += value * value * inverse_decay_flow;
                                }
                        }
                }
                });
        } else {
        // ---- the quadratic bins (acceptance-weighted, by the jittered length):
        // std::vector<double> quadratic_values(static_cast<size_t>(n_rows) * n_chains);
        std::vector<double> quadratic_values_of_this_update;
        std::vector<double> &quadratic_values = held != nullptr ? held->buffers.quadratic_values :
                                                                  quadratic_values_of_this_update;
        quadratic_values.resize(static_cast<size_t>(n_rows) * n_chains);
        fn_for_blocks(n_rows, n_input_blocks,
                      [&](const int, const int row_begin, const int row_end) {
        for (int c = 0; c < n_chains; c++) {
                for (int r = row_begin; r < row_end; r++) {
                        double q = at(quadratic_jump_squared, r, c);
                        if (!std::isfinite(q)) q = 0.0;
                        quadratic_values[r + static_cast<size_t>(c) * n_rows] = (q * weights[c]) / (2.0 * variance_quadratic[r]);
                }
        }
        });
        std::vector<int> quadratic_bin_index(n_chains, -1), binned_chains, flow_bin_index(n_chains, -1), valid_chains;
        for (int c = 0; c < n_chains; c++) {
                if (binned[c]) {
                        binned_chains.push_back(c);
                        quadratic_bin_index[c] = fn_bin_index(std::log2(tau_values[c] / quadratic.reference_length), quadratic.edges);
                }
                if (valid[c]) {
                        valid_chains.push_back(c);
                        flow_bin_index[c] = fn_bin_index(std::log2(executed_lengths[c] / flow.reference_length), flow.edges);
                }
        }
        // for (double &v : quadratic.sums) v *= bin_forgetting_factor;
        if (bins_decay_lazily) {
                decay_quadratic *= bin_forgetting_factor;
        } else {
                // fn_scale_in_blocks(quadratic.sums, bin_forgetting_factor, n_threads_for_update_loops);
                fn_scale_columns_of_bins_with_data(quadratic.sums, n_rows, quadratic.counts,
                                                   bin_forgetting_factor, n_threads_for_update_loops);
        }
        // (the counts before this update: which columns of the sums of squares have data)
        const std::vector<double> quadratic_counts_before_update = quadratic.counts;
        for (double &v : quadratic.counts) v *= bin_forgetting_factor;
        for (double &v : quadratic.length_sums) v *= bin_forgetting_factor;
        for (int c : binned_chains) {
                const int bin = quadratic_bin_index[c];
                if (bins_decay_lazily) {
                        for (int r = 0; r < n_rows; r++) {
                                quadratic.sums[r + static_cast<size_t>(bin) * n_rows] +=
                                      quadratic_values[r + static_cast<size_t>(c) * n_rows] / decay_quadratic;
                        }
                } else {
                // for (int r = 0; r < n_rows; r++) {
                //         quadratic.sums[r + static_cast<size_t>(bin) * n_rows] +=
                //               quadratic_values[r + static_cast<size_t>(c) * n_rows];
                // }
                fn_for_blocks(n_rows, n_input_blocks,
                      [&](const int, const int row_begin, const int row_end) {
                        for (int r = row_begin; r < row_end; r++) {
                                quadratic.sums[r + static_cast<size_t>(bin) * n_rows] +=
                                      quadratic_values[r + static_cast<size_t>(c) * n_rows];
                        }
                });
                }
                quadratic.counts[bin] += 1.0;
                quadratic.length_sums[bin] += tau_values[c];
                quadratic.largest_length[bin] = std::max(quadratic.largest_length[bin], static_cast<double>(tau_values[c]));
        }
        quadratic.n_updates += 1.0;
        if (expansion_evidence) {
                if (quadratic_sum_of_squares.empty()) quadratic_sum_of_squares.assign(quadratic.sums.size(), 0.0);
                if (bins_decay_lazily) {
                        fn_add_sum_of_squares_lazily(quadratic_sum_of_squares, n_rows, quadratic_values,
                                                     quadratic_bin_index, binned_chains, decay_quadratic);
                } else {
                fn_add_sum_of_squares(quadratic_sum_of_squares, n_rows, quadratic_values, quadratic_bin_index, binned_chains,
                                      bin_forgetting_factor, n_threads_for_update_loops,
                                      &quadratic_counts_before_update);
                }
        }
        // ---- the flow bins (unweighted proposal jumps of the valid chains, by the executed length):
        // std::vector<double> linear_values(static_cast<size_t>(n_rows) * n_chains);
        std::vector<double> linear_values_of_this_update;
        std::vector<double> &linear_values = held != nullptr ? held->buffers.linear_values :
                                                               linear_values_of_this_update;
        linear_values.resize(static_cast<size_t>(n_rows) * n_chains);
        fn_for_blocks(n_rows, n_input_blocks,
                      [&](const int, const int row_begin, const int row_end) {
        for (int c = 0; c < n_chains; c++) {
                for (int r = row_begin; r < row_end; r++) {
                        double l = at(linear_jump_squared, r, c);
                        if (!std::isfinite(l)) l = 0.0;
                        linear_values[r + static_cast<size_t>(c) * n_rows] = l / (2.0 * variance_linear[r]);
                }
        }
        });
        // (the flow counts before this update: which columns of the flow sums of squares have data)
        const std::vector<double> flow_counts_before_update = flow.counts;
        if (any_valid) {
                // for (double &v : flow.sums) v *= bin_forgetting_factor;
                if (bins_decay_lazily) {
                        decay_flow *= bin_forgetting_factor;
                } else {
                        // fn_scale_in_blocks(flow.sums, bin_forgetting_factor, n_threads_for_update_loops);
                        fn_scale_columns_of_bins_with_data(flow.sums, n_rows, flow.counts, bin_forgetting_factor,
                                                           n_threads_for_update_loops);
                }
                for (double &v : flow.counts) v *= bin_forgetting_factor;
                for (double &v : flow.length_sums) v *= bin_forgetting_factor;
                for (double &v : flow.cosine_sums) v *= bin_forgetting_factor;
                const int n_frequencies = static_cast<int>(flow.frequencies.size());
                for (int c : valid_chains) {
                        const int bin = flow_bin_index[c];
                        if (bins_decay_lazily) {
                                for (int r = 0; r < n_rows; r++) {
                                        flow.sums[r + static_cast<size_t>(bin) * n_rows] +=
                                              linear_values[r + static_cast<size_t>(c) * n_rows] / decay_flow;
                                }
                        } else {
                        // for (int r = 0; r < n_rows; r++) {
                        //         flow.sums[r + static_cast<size_t>(bin) * n_rows] +=
                        //               linear_values[r + static_cast<size_t>(c) * n_rows];
                        // }
                        fn_for_blocks(n_rows, n_input_blocks,
                      [&](const int, const int row_begin, const int row_end) {
                                for (int r = row_begin; r < row_end; r++) {
                                        flow.sums[r + static_cast<size_t>(bin) * n_rows] +=
                                              linear_values[r + static_cast<size_t>(c) * n_rows];
                                }
                        });
                        }
                        flow.counts[bin] += 1.0;
                        flow.length_sums[bin] += executed_lengths[c];
                        flow.largest_length[bin] = std::max(flow.largest_length[bin], executed_lengths[c]);
                        for (int k = 0; k < n_frequencies; k++) {
                                flow.cosine_sums[bin + static_cast<size_t>(k) * flow.n_bins] += std::cos(flow.frequencies[k] * executed_lengths[c]);
                        }
                }
                flow.n_updates += 1.0;
        }
        if (expansion_evidence) {
                if (flow_sum_of_squares.empty()) flow_sum_of_squares.assign(flow.sums.size(), 0.0);
                if (any_valid && bins_decay_lazily) {
                        fn_add_sum_of_squares_lazily(flow_sum_of_squares, n_rows, linear_values, flow_bin_index,
                                                     valid_chains, decay_flow);
                } else if (any_valid) {
                        fn_add_sum_of_squares(flow_sum_of_squares, n_rows, linear_values, flow_bin_index, valid_chains,
                                              bin_forgetting_factor, n_threads_for_update_loops,
                                              &flow_counts_before_update);
                }
        }
        }
        // (a decay factor below 1e-150 is moved into its stored values)
        if (bins_decay_lazily && decay_quadratic < 1e-150) {
                for (double &v : quadratic_movement_totals) v *= decay_quadratic;
                fn_scale_in_blocks(quadratic.sums, decay_quadratic, n_threads_for_update_loops);
                if (!quadratic_sum_of_squares.empty()) {
                        fn_scale_in_blocks(quadratic_sum_of_squares, decay_quadratic, n_threads_for_update_loops);
                }
                decay_quadratic = 1.0;
        }
        if (bins_decay_lazily && decay_flow < 1e-150) {
                for (double &v : linear_movement_totals) v *= decay_flow;
                fn_scale_in_blocks(flow.sums, decay_flow, n_threads_for_update_loops);
                if (!flow_sum_of_squares.empty()) {
                        fn_scale_in_blocks(flow_sum_of_squares, decay_flow, n_threads_for_update_loops);
                }
                decay_flow = 1.0;
        }
        // ---- the state to return (the R code's nested list):
        // Rcpp::List state_out = Rcpp::List::create(
        //       // Rcpp::Named("quadratic_state")              = fn_write_bins(quadratic, false),
        //       // Rcpp::Named("flow_state")                   = fn_write_bins(flow, true),
        //       Rcpp::Named("quadratic_state")              = fn_write_bins(quadratic, false,
        //                                                                   n_threads_for_update_loops),
        //       Rcpp::Named("flow_state")                   = fn_write_bins(flow, true,
        //                                                                   n_threads_for_update_loops),
        //       Rcpp::Named("second_moment_moving_average") = Rcpp::wrap(second_ma),
        //       Rcpp::Named("fourth_moment_moving_average") = Rcpp::wrap(fourth_ma),
        //       Rcpp::Named("acceptance_moving_average")    = acceptance_ma);
        // if (expansion_evidence) {
        //         // state_out.push_back(fn_matrix(quadratic_sum_of_squares, n_rows, quadratic.n_bins),
        //         //                     "quadratic_sum_of_squares");
        //         // state_out.push_back(fn_matrix(flow_sum_of_squares, n_rows, flow.n_bins),
        //         //                     "flow_sum_of_squares");
        //         state_out.push_back(fn_matrix(quadratic_sum_of_squares, n_rows, quadratic.n_bins,
        //                                       n_threads_for_update_loops), "quadratic_sum_of_squares");
        //         state_out.push_back(fn_matrix(flow_sum_of_squares, n_rows, flow.n_bins,
        //                                       n_threads_for_update_loops), "flow_sum_of_squares");
        // }
        // (the four bin matrices stay in C++, NULL in their places in the list; a new state, or one with its
        //  matrices in R, gets its held object here)
        if (held == nullptr) {
                Rcpp::XPtr<BinMatricesHeldInCpp> new_held_pointer(new BinMatricesHeldInCpp(), true);
                held = new_held_pointer.get();
                held->n_rows = n_rows;
                held->update_in_progress = true;
                held_pointer = new_held_pointer;
        }
        held->write_count += 1.0;
        held->update_in_progress = false;
        held->stored_sums_decay_lazily = bins_decay_lazily;
        held->decay_of_quadratic_stored_sums = decay_quadratic;
        held->decay_of_flow_stored_sums = decay_flow;
        // (the movement totals: a copy, since the choice of rows below reads them)
        held->linear_movement_totals = linear_movement_totals;
        held->quadratic_movement_totals = quadratic_movement_totals;
        held->movement_totals_valid = bins_decay_lazily;
        Rcpp::List state_out = Rcpp::List::create(
              Rcpp::Named("quadratic_state")              = fn_write_bins_without_sums(quadratic, false),
              Rcpp::Named("flow_state")                   = fn_write_bins_without_sums(flow, true),
              Rcpp::Named("second_moment_moving_average") = Rcpp::wrap(second_ma),
              Rcpp::Named("fourth_moment_moving_average") = Rcpp::wrap(fourth_ma),
              Rcpp::Named("acceptance_moving_average")    = acceptance_ma);
        if (expansion_evidence) {
                state_out.push_back(R_NilValue, "quadratic_sum_of_squares");
                state_out.push_back(R_NilValue, "flow_sum_of_squares");
        }
        state_out.push_back(held_pointer, "bin_matrices_held_in_cpp_between_updates");
        state_out.push_back(held->write_count, "bin_matrices_write_count");
        result["state"] = state_out;
        // ---- the choice of tau from the bins: of every row, or of the rows fitted (below); the same code for
        //      both:
        auto fn_choose_tau = [&](const Bins &quadratic, const Bins &flow, const int n_rows,
                                 const std::vector<double> &quadratic_sum_of_squares,
                                 const std::vector<double> &flow_sum_of_squares) -> Rcpp::List {
        // ---- the candidates and the quadratic bin means:
        std::vector<int> usable;
        for (int bin = 0; bin < quadratic.n_bins; bin++) if (quadratic.counts[bin] >= minimum_bin_count) usable.push_back(bin);
        if (usable.size() < 3) return result;
        std::vector<double> bin_centres;
        for (int bin : usable) bin_centres.push_back(quadratic.length_sums[bin] / quadratic.counts[bin]);
        // std::vector<double> bin_means;
        UpdateBuffers &buffers = held->buffers;
        std::vector<double> &bin_means = buffers.bin_means;
        // for (int bin : usable) {
        //         for (int r = 0; r < n_rows; r++) {
        //                 bin_means.push_back(quadratic.sums[r + static_cast<size_t>(bin) * n_rows] /
        //                                     quadratic.counts[bin]);
        //         }
        // }
        const double longest_length_with_data = quadratic.largest_length[usable.back()];
        const bool extra_node = longest_length_with_data > bin_centres.back();
        const int n_usable_bins = static_cast<int>(usable.size());
        bin_means.resize(static_cast<size_t>(n_usable_bins + (extra_node ? 1 : 0)) * n_rows);
        fn_for_blocks(n_rows, fn_number_of_row_blocks(n_rows, n_threads_for_update_loops),
                      [&](const int, const int row_begin, const int row_end) {
                for (int position = 0; position < n_usable_bins; position++) {
                        const int bin = usable[position];
                        const double *sums_of_bin = quadratic.sums.data() + static_cast<size_t>(bin) * n_rows;
                        double *means_of_bin = bin_means.data() + static_cast<size_t>(position) * n_rows;
                        for (int r = row_begin; r < row_end; r++) {
                                means_of_bin[r] = sums_of_bin[r] / quadratic.counts[bin];
                        }
                }
                if (extra_node) {
                        for (int r = row_begin; r < row_end; r++) {
                                bin_means[r + static_cast<size_t>(n_usable_bins) * n_rows] =
                                      bin_means[r + static_cast<size_t>(n_usable_bins - 1) * n_rows];
                        }
                }
        });
        if (extra_node) {
                bin_centres.push_back(longest_length_with_data);
                // for (int r = 0; r < n_rows; r++) bin_means.push_back(bin_means[bin_means.size() - n_rows]);
        }
        const double smallest_candidate = bin_centres.front();
        const double largest_candidate = bin_centres.back() / 2.0;
        if (!(largest_candidate > smallest_candidate)) return result;
        ObjectiveInputs in;
        in.n_rows = n_rows;
        in.eps_now = eps_now;
        in.acceptance_moving_average = acceptance_ma;
        in.soft_minimum_power = soft_minimum_power;
        in.minimum_bin_count = minimum_bin_count;
        for (double e : fn_seq_by(std::log2(smallest_candidate), std::log2(largest_candidate), grid_step_in_doublings)) {
                in.candidate_tau.push_back(std::pow(2.0, e));
        }
        if (in.candidate_tau.back() < largest_candidate) in.candidate_tau.push_back(largest_candidate);
        const int n_candidates = static_cast<int>(in.candidate_tau.size());
        in.n_nodes = static_cast<int>(bin_centres.size());
        // (fn_length_response_piecewise_linear_integral_weights, without the node at 0):
        std::vector<double> nodes(1, 0.0);
        nodes.insert(nodes.end(), bin_centres.begin(), bin_centres.end());
        std::vector<double> all_weights(static_cast<size_t>(nodes.size()) * n_candidates, 0.0);
        for (int c = 0; c < n_candidates; c++) {
                const double upper = 2.0 * in.candidate_tau[c];
                for (int segment = 0; segment + 1 < static_cast<int>(nodes.size()); segment++) {
                        const double left = nodes[segment], right = nodes[segment + 1];
                        if (upper <= left) break;
                        const double width = right - left;
                        const double covered = std::min(right, upper) - left;
                        all_weights[segment + static_cast<size_t>(c) * nodes.size()] += covered - covered * covered / (2.0 * width);
                        all_weights[segment + 1 + static_cast<size_t>(c) * nodes.size()] += covered * covered / (2.0 * width);
                }
        }
        in.integral_weights.resize(static_cast<size_t>(in.n_nodes) * n_candidates);
        for (int c = 0; c < n_candidates; c++) {
                for (int k = 0; k < in.n_nodes; k++) {
                        in.integral_weights[k + static_cast<size_t>(c) * in.n_nodes] = all_weights[k + 1 + static_cast<size_t>(c) * nodes.size()];
                }
        }
        for (double candidate : in.candidate_tau) in.expected_leapfrog_steps.push_back(fn_expected_leapfrog_steps_under_jitter(candidate, eps_now));
        // ---- the objective, its maximiser and the tau move:
        std::vector<double> objective;
        const PreparedFit prep = fn_prepare_fit(flow, in);
        if (!prep.usable) return result;
        // std::vector<double> flow_weights;
        // FitWorkspace fit_workspace;
        std::vector<double> &flow_weights = buffers.flow_weights;
        FitWorkspace &fit_workspace = buffers.fit_workspace;
        // fn_fit_weights_in_workspace(prep, flow.sums, n_rows, flow_weights, nullptr, fit_workspace,
        //                             n_threads_for_update_loops);
        // (each row's support of the previous update, mapped to this update's frequencies, is tried first)
        // std::vector<char> supports_to_try;
        std::vector<char> &supports_to_try = buffers.supports_to_try;
        const size_t n_previous_support_entries = held != nullptr ?
              static_cast<size_t>(held->estimate_supports_n_used) * n_rows : 0;
        const bool supports_of_previous_update = held != nullptr && held->n_rows == n_rows &&
                                                 held->estimate_supports_n_used > 0 &&
                                                 held->estimate_supports.size() == n_previous_support_entries;
        if (supports_of_previous_update) {
                supports_to_try.resize(static_cast<size_t>(prep.n_used) * n_rows);
                const int shift = prep.first_frequency_index - held->estimate_supports_first_frequency_index;
                const int previous_n_used = held->estimate_supports_n_used;
                const std::vector<char> &previous = held->estimate_supports;
                fn_for_blocks(n_rows, fn_number_of_row_blocks(n_rows, n_threads_for_update_loops),
                              [&](const int, const int row_begin, const int row_end) {
                        for (int r = row_begin; r < row_end; r++) {
                                for (int u = 0; u < prep.n_used; u++) {
                                        const int previous_u = u + shift;
                                        supports_to_try[u + static_cast<size_t>(r) * prep.n_used] =
                                              (previous_u >= 0 && previous_u < previous_n_used) ?
                                              previous[previous_u + static_cast<size_t>(r) * previous_n_used] : 0;
                                }
                        }
                });
        }
        fn_fit_weights_in_workspace(prep, flow.sums, n_rows, flow_weights, nullptr, fit_workspace,
                                    n_threads_for_update_loops,
                                    supports_of_previous_update ? &supports_to_try : nullptr);
        if (held != nullptr && held->n_rows == n_rows) {
                held->estimate_supports.resize(flow_weights.size());
                fn_for_blocks(n_rows, fn_number_of_row_blocks(n_rows, n_threads_for_update_loops),
                              [&](const int, const int row_begin, const int row_end) {
                        for (size_t i = static_cast<size_t>(row_begin) * prep.n_used;
                             i < static_cast<size_t>(row_end) * prep.n_used; i++) {
                                held->estimate_supports[i] = flow_weights[i] > 0.0 ? 1 : 0;
                        }
                });
                held->estimate_supports_first_frequency_index = prep.first_frequency_index;
                held->estimate_supports_n_used = prep.n_used;
        }
        std::vector<int> all_candidates(n_candidates);
        for (int c = 0; c < n_candidates; c++) all_candidates[c] = c;
        // fn_objective_at(in, prep, bin_means, flow_weights, all_candidates, objective,
        //                 n_threads_for_update_loops);
        std::vector<double> &transposed_weights = buffers.transposed_weights;
        std::vector<double> &sub_block_parts = buffers.sub_block_parts;
        fn_objective_at_in_one_pass(in, prep, bin_means, flow_weights, all_candidates, objective,
                                    n_threads_for_update_loops, transposed_weights, sub_block_parts);
        const int best = fn_which_max(objective);
        if (best < 0) return result;
        const bool maximum_is_at_upper_end = in.candidate_tau[best] >= largest_candidate * std::pow(2.0, -upper_end_zone_in_doublings);
        const double progress = schedule_length <= 1.0 ? 0.0 :
                                std::min(1.0, std::max(0.0, (schedule_iteration - 1.0) / (schedule_length - 1.0)));
        const double learning_rate_now = learning_rate * (1.0 - (1.0 - learning_rate) * progress);
        bool expand = maximum_is_at_upper_end;
        double evidence_z = NA_DOUBLE;
        bool bootstrap_ran = false;
        if (expand && expansion_evidence) {
                const int top = n_candidates - 1;
                int reference = 0;
                double closest = std::numeric_limits<double>::infinity();
                const double reference_position = std::log2(in.candidate_tau[top]) - 0.25;
                for (int c = 0; c < n_candidates; c++) {
                        const double distance = std::fabs(std::log2(in.candidate_tau[c]) - reference_position);
                        if (distance < closest) {
                                closest = distance;
                                reference = c;
                        }
                }
                // // ---- the bootstrap's standard errors (quadratic bins with the extra node; flow bins):
                // std::vector<double> quadratic_standard_errors;
                // for (int bin : usable) {
                //         for (int r = 0; r < n_rows; r++) {
                //                 const double count = quadratic.counts[bin];
                //                 const double mean = quadratic.sums[r + static_cast<size_t>(bin) * n_rows] /
                //                                     count;
                //                 const double variance = fn_max_na(quadratic_sum_of_squares[r +
                //                                                   static_cast<size_t>(bin) * n_rows] / count -
                //                                                   mean * mean, 0.0);
                //                 quadratic_standard_errors.push_back(std::sqrt(variance / count));
                //         }
                // }
                // if (bin_means.size() > quadratic_standard_errors.size()) {
                //         for (int r = 0; r < n_rows; r++) {
                //                 quadratic_standard_errors.push_back(
                //                       quadratic_standard_errors[quadratic_standard_errors.size() - n_rows]);
                //         }
                // }
                // std::vector<int> flow_usable;
                // for (int bin = 0; bin < flow.n_bins; bin++) if (flow.counts[bin] >= minimum_bin_count)
                //         flow_usable.push_back(bin);
                // std::vector<double> flow_means, flow_standard_errors;
                // for (int bin : flow_usable) {
                //         for (int r = 0; r < n_rows; r++) {
                //                 const double count = flow.counts[bin];
                //                 const double mean = flow.sums[r + static_cast<size_t>(bin) * n_rows] / count;
                //                 flow_means.push_back(mean);
                //                 flow_standard_errors.push_back(std::sqrt(fn_max_na(flow_sum_of_squares[r +
                //                                                static_cast<size_t>(bin) * n_rows] /
                //                                                count - mean * mean, 0.0) / count));
                //         }
                // }
                // (the standard errors, the flow means and the usable flow bins: computed below, over threads,
                //  once the draws are known to be needed)
                // std::vector<double> quadratic_standard_errors, flow_means, flow_standard_errors;
                // std::vector<int> flow_usable;
                std::vector<double> &quadratic_standard_errors = buffers.quadratic_standard_errors;
                std::vector<double> &flow_means = buffers.flow_means;
                std::vector<double> &flow_standard_errors = buffers.flow_standard_errors;
                std::vector<int> &flow_usable = buffers.flow_usable;
                flow_usable.clear();
                // (8 Oct 2026) the test with two exact shortcuts (the block before them is below, commented
                // out): with reference == top, or a difference estimate <= 0 (or NaN), z <= 0 whatever the draws,
                // so there is no expansion and no draw is needed; and once the finite draws so far have a sum of
                // squared deviations SS_k with SS_k / (B - 1) >= (difference / 1.645)^2, the final standard
                // deviation is at least difference / 1.645 (the sum of squares about the final mean is at least
                // SS_k, and B - 1 is the largest divisor), so z <= 1.645 and there is no expansion either. The
                // decision is the same as with all B draws; evidence_z is then the bound from the draws so far
                // (or NaN, without draws):
                const double difference_estimate = objective[top] - objective[reference];
                const double z_threshold = R::qnorm(0.95, 0.0, 1.0, 1, 0);
                if (!(reference < top) || !(difference_estimate > 0.0)) {
                        evidence_z = NA_DOUBLE;
                        expand = false;
                } else {
                        // ---- the bootstrap's standard errors (quadratic bins with the extra node; flow bins),
                        //      each entry by the arithmetic of the loops above (commented out), the rows over
                        //      threads:
                        const int n_usable_quadratic = static_cast<int>(usable.size());
                        quadratic_standard_errors.resize(bin_means.size());
                        for (int bin = 0; bin < flow.n_bins; bin++) {
                                if (flow.counts[bin] >= minimum_bin_count) flow_usable.push_back(bin);
                        }
                        const int n_flow_usable = static_cast<int>(flow_usable.size());
                        flow_means.resize(static_cast<size_t>(n_flow_usable) * n_rows);
                        flow_standard_errors.resize(static_cast<size_t>(n_flow_usable) * n_rows);
                        fn_for_blocks(n_rows, fn_number_of_row_blocks(n_rows, n_threads_for_update_loops),
                                      [&](const int, const int row_begin, const int row_end) {
                                for (int position = 0; position < n_usable_quadratic; position++) {
                                        const int bin = usable[position];
                                        const double count = quadratic.counts[bin];
                                        const size_t bin_start = static_cast<size_t>(bin) * n_rows;
                                        const double *sums = quadratic.sums.data() + bin_start;
                                        const double *squares = quadratic_sum_of_squares.data() + bin_start;
                                        double *errors = quadratic_standard_errors.data() +
                                                         static_cast<size_t>(position) * n_rows;
                                        for (int r = row_begin; r < row_end; r++) {
                                                const double mean = sums[r] / count;
                                                const double variance = fn_max_na(squares[r] / count -
                                                                                  mean * mean, 0.0);
                                                errors[r] = std::sqrt(variance / count);
                                        }
                                }
                                if (bin_means.size() > static_cast<size_t>(n_usable_quadratic) * n_rows) {
                                        double *extra = quadratic_standard_errors.data() +
                                                        static_cast<size_t>(n_usable_quadratic) * n_rows;
                                        const double *last = extra - n_rows;
                                        for (int r = row_begin; r < row_end; r++) extra[r] = last[r];
                                }
                                for (int position = 0; position < n_flow_usable; position++) {
                                        const int bin = flow_usable[position];
                                        const double count = flow.counts[bin];
                                        const size_t bin_start = static_cast<size_t>(bin) * n_rows;
                                        const double *sums = flow.sums.data() + bin_start;
                                        const double *squares = flow_sum_of_squares.data() + bin_start;
                                        double *means = flow_means.data() +
                                                        static_cast<size_t>(position) * n_rows;
                                        double *errors = flow_standard_errors.data() +
                                                         static_cast<size_t>(position) * n_rows;
                                        for (int r = row_begin; r < row_end; r++) {
                                                const double mean = sums[r] / count;
                                                means[r] = mean;
                                                errors[r] = std::sqrt(fn_max_na(squares[r] /
                                                                                count - mean * mean, 0.0) /
                                                                      count);
                                        }
                                }
                        });
                        // (the draws' flow sums hold only the usable flow bins, the only ones the fit reads, in
                        //  their order: the fit of the draws reads them through prep_of_draws, whose usable bins
                        //  are 0, 1, ... in place of the bins' numbers; the same values in the same order)
                        PreparedFit prep_of_draws = prep;
                        for (int i = 0; i < prep_of_draws.n_usable; i++) prep_of_draws.usable_bins[i] = i;
                        // ---- the draws, from the bootstrap's own generator, seeded from the update count as R's
                        //      generator was (set.seed(104729 + the quadratic updates so far)); R's stream is not
                        //      touched:
                        BootstrapNormalGenerator generator(static_cast<uint64_t>(104729.0 + quadratic.n_updates));
                        std::vector<double> difference_draws;
                        // std::vector<double> perturbed_quadratic(bin_means.size());
                        // std::vector<double> perturbed_flow_sums = flow.sums;
                        // (two sets of perturbed inputs: each draw's set is generated, from the same generator
                        //  and in the same order, on another thread whilst the previous draw's set is fitted, so
                        //  the draws are the same numbers as when they are generated in turn)
                        // std::vector<double> perturbed_quadratic_of_set[2] = {
                        //       std::vector<double>(bin_means.size()), std::vector<double>(bin_means.size())};
                        std::vector<double> *perturbed_quadratic_of_set = buffers.perturbed_quadratic;
                        for (int set = 0; set < 2; set++) {
                                perturbed_quadratic_of_set[set].resize(bin_means.size());
                        }
                        // std::vector<double> perturbed_flow_sums_of_set[2] = {flow.sums, flow.sums};
                        // std::vector<double> perturbed_flow_sums_of_set[2] = {
                        //       std::vector<double>(flow_means.size()), std::vector<double>(flow_means.size())};
                        std::vector<double> *perturbed_flow_sums_of_set = buffers.perturbed_flow_sums;
                        for (int set = 0; set < 2; set++) {
                                perturbed_flow_sums_of_set[set].resize(flow_means.size());
                        }
                        std::atomic<bool> stop_generating_draws(false);
                        // auto fn_generate_draw_into_set = [&](const int set) {
                        //         std::vector<double> &perturbed_quadratic = perturbed_quadratic_of_set[set];
                        //         std::vector<double> &perturbed_flow_sums = perturbed_flow_sums_of_set[set];
                        //         for (size_t i = 0; i < bin_means.size(); i++) {
                        //                 if ((i & 65535) == 0 && stop_generating_draws.load()) return;
                        //                 perturbed_quadratic[i] = fn_max_na(bin_means[i] +
                        //                                                    quadratic_standard_errors[i] *
                        //                                                    generator.fn_normal(), 0.0);
                        //         }
                        //         size_t index = 0;
                        //         for (int bin : flow_usable) {
                        //                 if (stop_generating_draws.load()) return;
                        //                 for (int r = 0; r < n_rows; r++) {
                        //                         perturbed_flow_sums[r + static_cast<size_t>(bin) * n_rows] =
                        //                               fn_max_na(flow_means[index] +
                        //                                         flow_standard_errors[index] *
                        //                                         generator.fn_normal(), 0.0) *
                        //                               flow.counts[bin];
                        //                         index++;
                        //                 }
                        //         }
                        // };
                        // (the draws' normals, in the order fn_generate_draw_into_set() above used them: the
                        //  generator's pair j, (u, v, q) with q = u^2 + v^2 in (0, 1), gives the normals 2j = u f
                        //  and 2j + 1 = v f with f = sqrt(-2 log(q) / q), as
                        //  BootstrapNormalGenerator::fn_normal(); draw d uses the normals from
                        //  d x n_normals_per_draw on, the quadratic bins' first. The
                        //  pairs are drawn in turn, on the generator thread, and f, the normals and the perturbed
                        //  inputs are computed over the update's threads, so the perturbed inputs are the same
                        //  numbers)
                        const int64_t n_quadratic_normals = static_cast<int64_t>(bin_means.size());
                        const int64_t n_normals_per_draw = n_quadratic_normals +
                                                           static_cast<int64_t>(flow_usable.size()) * n_rows;
                        // std::vector<double> pair_u_of_set[2], pair_v_of_set[2], pair_q_of_set[2], pair_f;
                        std::vector<double> *pair_u_of_set = buffers.pair_u, *pair_v_of_set = buffers.pair_v,
                                            *pair_q_of_set = buffers.pair_q;
                        std::vector<double> &pair_f = buffers.pair_f;
                        int64_t first_pair_of_set[2] = {0, 0};
                        auto fn_draw_pairs_into_set = [&](const int set, const int draw) {
                                const int64_t first_normal = static_cast<int64_t>(draw) * n_normals_per_draw;
                                const int64_t first_pair = first_normal / 2;
                                const int64_t last_pair = (first_normal + n_normals_per_draw - 1) / 2;
                                std::vector<double> &pair_u = pair_u_of_set[set];
                                std::vector<double> &pair_v = pair_v_of_set[set];
                                std::vector<double> &pair_q = pair_q_of_set[set];
                                pair_u.resize(last_pair - first_pair + 1);
                                pair_v.resize(last_pair - first_pair + 1);
                                pair_q.resize(last_pair - first_pair + 1);
                                first_pair_of_set[set] = first_pair;
                                int64_t pair = first_pair;
                                // (the previous draw's last pair, when this draw starts with its second normal)
                                if (draw > 0) {
                                        const std::vector<double> &previous_u = pair_u_of_set[1 - set];
                                        const int64_t n_previous_pairs = static_cast<int64_t>(previous_u.size());
                                        const int64_t previous_last_pair = first_pair_of_set[1 - set] +
                                                                           n_previous_pairs - 1;
                                        if (first_pair == previous_last_pair) {
                                                pair_u[0] = previous_u.back();
                                                pair_v[0] = pair_v_of_set[1 - set].back();
                                                pair_q[0] = pair_q_of_set[1 - set].back();
                                                pair++;
                                        }
                                }
                                for (; pair <= last_pair; pair++) {
                                        if (((pair - first_pair) & 65535) == 0 && stop_generating_draws.load()) {
                                                return;
                                        }
                                        double u, v, q;
                                        do {
                                                u = 2.0 * generator.fn_uniform() - 1.0;
                                                v = 2.0 * generator.fn_uniform() - 1.0;
                                                q = u * u + v * v;
                                        } while (q >= 1.0 || q == 0.0);
                                        pair_u[pair - first_pair] = u;
                                        pair_v[pair - first_pair] = v;
                                        pair_q[pair - first_pair] = q;
                                }
                        };
                        auto fn_perturbed_inputs_of_set = [&](const int set, const int draw) {
                                const std::vector<double> &pair_u = pair_u_of_set[set];
                                const std::vector<double> &pair_v = pair_v_of_set[set];
                                const std::vector<double> &pair_q = pair_q_of_set[set];
                                const int64_t n_pairs = static_cast<int64_t>(pair_q.size());
                                pair_f.resize(n_pairs);
                                const int n_threads = n_threads_for_update_loops;
                                const int n_pair_blocks = fn_number_of_draw_arithmetic_blocks(n_pairs, n_threads);
                                fn_for_blocks(n_pairs, n_pair_blocks, [&](const int, const int64_t begin,
                                                                          const int64_t end) {
                                        for (int64_t j = begin; j < end; j++) {
                                                pair_f[j] = std::sqrt(-2.0 * std::log(pair_q[j]) / pair_q[j]);
                                        }
                                });
                                const int64_t first_normal = static_cast<int64_t>(draw) * n_normals_per_draw;
                                const int64_t first_pair = first_pair_of_set[set];
                                double *perturbed_quadratic = perturbed_quadratic_of_set[set].data();
                                double *perturbed_flow_sums = perturbed_flow_sums_of_set[set].data();
                                const int n_normal_blocks =
                                      fn_number_of_draw_arithmetic_blocks(n_normals_per_draw, n_threads);
                                const double *quadratic_errors = quadratic_standard_errors.data();
                                const double *flow_errors = flow_standard_errors.data();
                                fn_for_blocks(n_normals_per_draw, n_normal_blocks,
                                              [&](const int, const int64_t begin, const int64_t end) {
                                        fn_perturbed_inputs_of_draw(pair_u.data(), pair_v.data(), pair_f.data(),
                                                                    first_normal, first_pair, begin, end,
                                                                    n_quadratic_normals, n_rows, bin_means.data(),
                                                                    quadratic_errors, flow_means.data(),
                                                                    flow_errors, flow_usable.data(),
                                                                    flow.counts.data(), perturbed_quadratic,
                                                                    perturbed_flow_sums);
                                });
                        };
                        // (the fits and the objective leave one of the update's threads to the generation)
                        const int n_threads_for_draw_loops = std::max(1, n_threads_for_update_loops - 1);
                        // std::vector<double> draw_objective, draw_weights;
                        std::vector<double> draw_objective;
                        std::vector<double> &draw_weights = buffers.draw_weights;
                        const std::vector<int> top_and_reference = {top, reference};
                        const double variance_bound_for_no_expansion = (difference_estimate / z_threshold) *
                                                                       (difference_estimate / z_threshold) *
                                                                       (1.0 + 1e-12);
                        double running_mean = 0.0, running_sum_of_squares = 0.0;
                        int n_finite = 0;
                        bool stopped_early = false;
                        // for (int draw = 0; draw < n_bootstrap_draws; draw++) {
                        //         for (size_t i = 0; i < bin_means.size(); i++) {
                        //                 perturbed_quadratic[i] = fn_max_na(bin_means[i] +
                        //                                                    quadratic_standard_errors[i] *
                        //                                                    generator.fn_normal(), 0.0);
                        //         }
                        //         size_t index = 0;
                        //         for (int bin : flow_usable) {
                        //                 for (int r = 0; r < n_rows; r++) {
                        //                         perturbed_flow_sums[r + static_cast<size_t>(bin) * n_rows] =
                        //                               fn_max_na(flow_means[index] +
                        //                                         flow_standard_errors[index] *
                        //                                         generator.fn_normal(), 0.0) *
                        //                               flow.counts[bin];
                        //                         index++;
                        //                 }
                        //         }
                        // fn_generate_draw_into_set(0);
                        fn_draw_pairs_into_set(0, 0);
                        fn_perturbed_inputs_of_set(0, 0);
                        for (int draw = 0; draw < n_bootstrap_draws; draw++) {
                                const int set = draw % 2;
                                // ---- the next draw's set, generated on another thread (or after this draw's
                                //      fit, on this thread, if no thread can be started):
                                std::thread generator_of_next_draw;
                                bool next_draw_is_being_generated = false;
                                // if (draw + 1 < n_bootstrap_draws) {
                                if (draw + 1 < n_bootstrap_draws && n_normals_per_draw >=
                                    minimum_normals_per_draw_for_generation_on_another_thread) {
                                        try {
                                                // generator_of_next_draw = std::thread(fn_generate_draw_into_set,
                                                //                                      1 - set);
                                                generator_of_next_draw = std::thread(fn_draw_pairs_into_set,
                                                                                     1 - set, draw + 1);
                                                next_draw_is_being_generated = true;
                                        } catch (...) {
                                        }
                                }
                                JoinThreadOnExit join_generator_of_next_draw(generator_of_next_draw);
                                // (only the two candidates that the test compares; each row's NNLS starts from
                                //  the estimates' support)
                                // fn_fit_weights_in_workspace(prep, perturbed_flow_sums_of_set[set], n_rows,
                                //                             draw_weights, &flow_weights, fit_workspace,
                                //                             n_threads_for_draw_loops);
                                fn_fit_weights_in_workspace(prep_of_draws, perturbed_flow_sums_of_set[set],
                                                            n_rows, draw_weights, &flow_weights, fit_workspace,
                                                            n_threads_for_draw_loops);
                                // fn_objective_at(in, prep, perturbed_quadratic_of_set[set], draw_weights,
                                //                 top_and_reference, draw_objective, n_threads_for_draw_loops);
                                fn_objective_at_in_one_pass(in, prep, perturbed_quadratic_of_set[set],
                                                            draw_weights, top_and_reference, draw_objective,
                                                            n_threads_for_draw_loops, transposed_weights,
                                                            sub_block_parts);
                                const double difference_draw = draw_objective[top] - draw_objective[reference];
                                difference_draws.push_back(difference_draw);
                                if (!std::isnan(difference_draw)) {
                                        n_finite++;
                                        const double deviation = difference_draw - running_mean;
                                        running_mean += deviation / n_finite;
                                        running_sum_of_squares += deviation * (difference_draw - running_mean);
                                        if (n_finite >= 2 && running_sum_of_squares / (n_bootstrap_draws - 1) >=
                                                             variance_bound_for_no_expansion) {
                                                stopped_early = true;
                                                stop_generating_draws.store(true);   // (no next set needed)
                                                break;
                                        }
                                }
                                // if (!next_draw_is_being_generated && draw + 1 < n_bootstrap_draws) {
                                //         fn_generate_draw_into_set(1 - set);
                                // }
                                if (draw + 1 < n_bootstrap_draws) {
                                        if (!next_draw_is_being_generated) {
                                                fn_draw_pairs_into_set(1 - set, draw + 1);
                                        } else if (generator_of_next_draw.joinable()) {
                                                generator_of_next_draw.join();
                                        }
                                        fn_perturbed_inputs_of_set(1 - set, draw + 1);
                                }
                        }
                        bootstrap_ran = true;
                        if (stopped_early) {
                                evidence_z = difference_estimate /
                                             std::sqrt(running_sum_of_squares / (n_bootstrap_draws - 1));
                                expand = false;
                        } else {
                                // ---- sd(difference_draws, na.rm = TRUE), and the one-sided 5% test:
                                std::vector<double> finite_draws;
                                for (double d : difference_draws) if (!std::isnan(d)) finite_draws.push_back(d);
                                double spread = NA_DOUBLE;
                                if (finite_draws.size() >= 2) {
                                        double mean = 0.0;
                                        for (double d : finite_draws) mean += d;
                                        mean /= finite_draws.size();
                                        double sum_of_squares = 0.0;
                                        for (double d : finite_draws) sum_of_squares += (d - mean) * (d - mean);
                                        spread = std::sqrt(sum_of_squares / (finite_draws.size() - 1));
                                }
                                evidence_z = (std::isfinite(spread) && spread > 0.0) ?
                                             difference_estimate / spread :
                                             -std::numeric_limits<double>::infinity();
                                expand = reference < top && evidence_z > z_threshold;
                        }
                }
                // // ---- the draws: (8 Oct 2026) from the bootstrap's own generator, seeded from the update
                // //      count as R's generator was (set.seed(104729 + the quadratic updates so far)); R's
                // //      stream is not
                // //      touched:
                // // Rcpp::Function set_seed("set.seed");
                // // set_seed(104729.0 + quadratic.n_updates);
                // // GetRNGstate();
                // BootstrapNormalGenerator generator(static_cast<uint64_t>(104729.0 + quadratic.n_updates));
                // std::vector<double> difference_draws;
                // std::vector<double> perturbed_quadratic(bin_means.size()), perturbed_flow_sums = flow.sums,
                //                     draw_objective;
                // const std::vector<int> top_and_reference = {top, reference};
                // std::vector<double> draw_weights;
                // for (int draw = 0; draw < n_bootstrap_draws; draw++) {
                //         for (size_t i = 0; i < bin_means.size(); i++) {
                //                 perturbed_quadratic[i] = fn_max_na(bin_means[i] +
                //                                                    quadratic_standard_errors[i] *
                //                                                                   generator.fn_normal(), 0.0);
                //         }
                //         size_t index = 0;
                //         for (int bin : flow_usable) {
                //                 for (int r = 0; r < n_rows; r++) {
                //                         perturbed_flow_sums[r + static_cast<size_t>(bin) * n_rows] =
                //                               fn_max_na(flow_means[index] + flow_standard_errors[index] *
                //                                                             generator.fn_normal(), 0.0) *
                //                               flow.counts[bin];
                //                         index++;
                //                 }
                //         }
                //         // (only the two candidates that the test compares; each row's NNLS starts from the
                //         //  estimates' support)
                //         fn_fit_weights_in_workspace(prep, perturbed_flow_sums, n_rows, draw_weights,
                //                                     &flow_weights, fit_workspace);
                //         fn_objective_at(in, prep, perturbed_quadratic, draw_weights, top_and_reference,
                //                         draw_objective);
                //         difference_draws.push_back(draw_objective[top] - draw_objective[reference]);
                // }
                // // PutRNGstate();
                // bootstrap_ran = true;
                // // ---- sd(difference_draws, na.rm = TRUE), and the one-sided 5% test:
                // std::vector<double> finite_draws;
                // for (double d : difference_draws) if (!std::isnan(d)) finite_draws.push_back(d);
                // double spread = NA_DOUBLE;
                // if (finite_draws.size() >= 2) {
                //         double mean = 0.0;
                //         for (double d : finite_draws) mean += d;
                //         mean /= finite_draws.size();
                //         double sum_of_squares = 0.0;
                //         for (double d : finite_draws) sum_of_squares += (d - mean) * (d - mean);
                //         spread = std::sqrt(sum_of_squares / (finite_draws.size() - 1));
                // }
                // const double difference_estimate = objective[top] - objective[reference];
                // evidence_z = (std::isfinite(spread) && spread > 0.0) ? difference_estimate / spread :
                //              -std::numeric_limits<double>::infinity();
                // expand = reference < top && evidence_z > R::qnorm(0.95, 0.0, 1.0, 1, 0);
        }
        const double target_tau = expand ? 2.0 * std::max(tau, largest_candidate) : in.candidate_tau[best];
        const double new_tau = std::exp(std::log(tau) + learning_rate_now * (std::log(target_tau) - std::log(tau)));
        result["performed"] = true;
        result.push_back(new_tau, "new_tau");
        result.push_back(std::log(target_tau) - std::log(tau), "gradient");
        result.push_back(objective[best], "criterion");
        result.push_back(in.candidate_tau[best], "tau_maximising_criterion");
        result.push_back(maximum_is_at_upper_end, "maximum_is_at_upper_end");
        result.push_back(largest_candidate, "largest_candidate");
        result.push_back(learning_rate_now, "learning_rate_now");
        result.push_back(prep.lowest_resolvable_frequency, "lowest_resolvable_frequency");
        result.push_back(acceptance_ma, "acceptance_moving_average");
        result.push_back(expand, "expanded_towards_doubling");
        result.push_back(evidence_z, "expansion_evidence_z");
        result.push_back(bootstrap_ran, "bootstrap_ran");
        return result;
        };
        const bool rows_bounded = largest_number_of_rows_fitted_per_update > 0 &&
                                  n_rows > largest_number_of_rows_fitted_per_update;
        if (!rows_bounded && !bins_decay_lazily) {
                return fn_choose_tau(quadratic, flow, n_rows, quadratic_sum_of_squares, flow_sum_of_squares);
        }
        // ---- the rows fitted and scored (the slowest-moving ones above the limit, otherwise every row), with
        //      their true bin values (the stored values times the decay factors):
        std::vector<int> rows_fitted;
        if (rows_bounded && bins_decay_lazily) {
                rows_fitted = fn_slowest_moving_rows_of_movements(linear_movement_totals,
                                                                  quadratic_movement_totals, n_rows,
                                                                  largest_number_of_rows_fitted_per_update);
        } else if (rows_bounded) {
                rows_fitted = fn_slowest_moving_rows(quadratic, flow, n_rows,
                                                     largest_number_of_rows_fitted_per_update,
                                                     n_threads_for_update_loops);
        } else {
                rows_fitted.resize(n_rows);
                for (int r = 0; r < n_rows; r++) rows_fitted[r] = r;
        }
        const Bins quadratic_of_rows_fitted = fn_bins_of_rows(quadratic, rows_fitted, n_rows, decay_quadratic);
        const Bins flow_of_rows_fitted = fn_bins_of_rows(flow, rows_fitted, n_rows, decay_flow);
        const std::vector<double> quadratic_sum_of_squares_of_rows_fitted =
              fn_matrix_rows(quadratic_sum_of_squares, rows_fitted, n_rows, decay_quadratic, quadratic.counts);
        const std::vector<double> flow_sum_of_squares_of_rows_fitted =
              fn_matrix_rows(flow_sum_of_squares, rows_fitted, n_rows, decay_flow, flow.counts);
        Rcpp::List chosen = fn_choose_tau(quadratic_of_rows_fitted, flow_of_rows_fitted,
                                          static_cast<int>(rows_fitted.size()),
                                          quadratic_sum_of_squares_of_rows_fitted,
                                          flow_sum_of_squares_of_rows_fitted);
        chosen.push_back(static_cast<double>(rows_fitted.size()), "n_rows_fitted");
        return chosen;

}


//// ---- the state with its four bin matrices in R (copies of the matrices held in C++), as the update returned
////      it before they were held in C++; a NULL state, or one with its matrices in R already, is returned as
////      it is:
// [[Rcpp::export(rng = false)]]
SEXP fn_spectral_LQ_ESSR_state_with_bin_matrices_in_R(SEXP state_in) {
        if (Rf_isNull(state_in)) return state_in;
        Rcpp::List state(state_in);
        BinMatricesHeldInCpp *held = fn_held_bin_matrices_of_state(state, -1);
        if (held == nullptr) return state_in;
        const int n_rows = held->n_rows;
        Rcpp::CharacterVector names = state.names();
        Rcpp::List out;
        for (int i = 0; i < state.size(); i++) {
                const std::string name = Rcpp::as<std::string>(names[i]);
                if (name == "bin_matrices_held_in_cpp_between_updates" || name == "bin_matrices_write_count") {
                        continue;
                }
                if (name == "quadratic_state" || name == "flow_state") {
                        Rcpp::List bins = Rcpp::clone(Rcpp::as<Rcpp::List>(state[i]));
                        const int n_bins = Rf_length(bins["sum_of_chain_counts"]);
                        const std::vector<double> &sums = name == "quadratic_state" ? held->quadratic_sums :
                                                          held->flow_sums;
                        const double decay = !held->stored_sums_decay_lazily ? 1.0 :
                                             name == "quadratic_state" ? held->decay_of_quadratic_stored_sums :
                                             held->decay_of_flow_stored_sums;
                        std::vector<double> true_sums = sums;
                        if (decay != 1.0) for (double &v : true_sums) v *= decay;
                        bins["sum_of_normalised_squared_jumps"] = fn_matrix(true_sums, n_rows, n_bins);
                        out.push_back(bins, name);
                } else if (name == "quadratic_sum_of_squares" || name == "flow_sum_of_squares") {
                        const std::vector<double> &values = name == "quadratic_sum_of_squares" ?
                                                            held->quadratic_sum_of_squares :
                                                            held->flow_sum_of_squares;
                        const int n_bins = static_cast<int>(values.size() / std::max(1, n_rows));
                        const double decay = !held->stored_sums_decay_lazily ? 1.0 :
                                             name == "quadratic_sum_of_squares" ?
                                             held->decay_of_quadratic_stored_sums :
                                             held->decay_of_flow_stored_sums;
                        std::vector<double> true_values = values;
                        if (decay != 1.0) for (double &v : true_values) v *= decay;
                        out.push_back(fn_matrix(true_values, n_rows, n_bins), name);
                } else {
                        out.push_back(state[i], name);
                }
        }
        return out;
}

//// ---- the export of SpESS-R's statistics (fn_spectral_LQ_ESSR_statistics_of_rows(); the rows in blocks over
////      n_threads threads, each column's sum on one thread), returned as fn_metric_position_criterion() returns
////      them:
// [[Rcpp::export(rng = false)]]
Rcpp::List fn_spectral_LQ_ESSR_statistics_cpp(Rcpp::NumericMatrix theta_initial,
                                             Rcpp::NumericMatrix theta_proposed,
                                             Rcpp::NumericMatrix velocity_proposed,
                                             Rcpp::NumericVector mean_initial, Rcpp::NumericVector metric_vector,
                                             Rcpp::NumericVector tau_values, const int n_threads) {
        const int n_rows = theta_initial.nrow(), n_chains = theta_initial.ncol();
        Rcpp::NumericMatrix linear_jump_squared(Rcpp::no_init(n_rows, n_chains));
        Rcpp::NumericMatrix quadratic_jump_squared(Rcpp::no_init(n_rows, n_chains));
        Rcpp::NumericMatrix initial_second_moment(Rcpp::no_init(n_rows, n_chains));
        Rcpp::NumericMatrix initial_fourth_moment(Rcpp::no_init(n_rows, n_chains));
        std::vector<double> gradient_terms(static_cast<size_t>(n_rows) * n_chains);
        // (the R memory is taken here, on the R thread; the threads only write into it)
        const double *theta_initial_values = theta_initial.begin();
        const double *theta_proposed_values = theta_proposed.begin();
        const double *mean_initial_values = mean_initial.begin(), *metric_values = metric_vector.begin();
        const double *velocity_values = velocity_proposed.begin();
        const double *tau = tau_values.begin();
        double *linear = linear_jump_squared.begin(), *quadratic = quadratic_jump_squared.begin();
        double *second = initial_second_moment.begin(), *fourth = initial_fourth_moment.begin();
        fn_for_blocks(n_rows, fn_number_of_row_blocks(n_rows, n_threads),
                      [&](const int, const int row_begin, const int row_end) {
                fn_spectral_LQ_ESSR_statistics_of_rows(theta_initial_values, theta_proposed_values,
                                                       mean_initial_values, metric_values, velocity_values, tau,
                                                       n_rows, n_chains, row_begin, row_end, linear, quadratic,
                                                       second, fourth, gradient_terms.data());
        });
        std::vector<double> numerator(n_chains), numerator_gradient(n_chains);
        fn_for_blocks(n_chains, std::max(1, std::min(n_threads, n_chains)),
                      [&](const int, const int chain_begin, const int chain_end) {
                for (int c = chain_begin; c < chain_end; c++) {
                        const size_t start = static_cast<size_t>(c) * n_rows;
                        numerator[c] = fn_column_sum_in_long_double(linear + start, n_rows);
                        numerator_gradient[c] = fn_column_sum_in_long_double(gradient_terms.data() + start,
                                                                             n_rows);
                }
        });
        return Rcpp::List::create(Rcpp::Named("linear_jump_squared")    = linear_jump_squared,
                                  Rcpp::Named("quadratic_jump_squared") = quadratic_jump_squared,
                                  Rcpp::Named("initial_second_moment")  = initial_second_moment,
                                  Rcpp::Named("initial_fourth_moment")  = initial_fourth_moment,
                                  Rcpp::Named("numerator")              = Rcpp::wrap(numerator),
                                  Rcpp::Named("numerator_gradient")     = Rcpp::wrap(numerator_gradient),
                                  Rcpp::Named("tau_cost_exponent")      = 1.0);
}























