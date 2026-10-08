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

void fn_nnls_one_column_in_workspace(const std::vector<double> &AtA, const double *Atb_column, const int n,
                                     const double tolerance, const double *warm_start, NNLSWorkspace &ws,
                                     double *x_out) {
        double *x = ws.x.data(), *w = ws.w.data(), *s = ws.s.data();
        int *passive = ws.passive.data(), *excluded = ws.excluded.data(), *P = ws.P.data();
        int *support = ws.support.data();
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
                        for (int a = 0; a < np; a++) {
                                right_hand_side[a] = Atb_column[P[a]];
                                for (int b = 0; b < np; b++) gram[a * np + b] = AtA[P[a] * n + P[b]];
                        }
                        if (!fn_cholesky_solve_in_place(gram, right_hand_side, np, ws.forward.data(), sub)) {
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

Bins fn_read_bins(const Rcpp::List &bins_list, const bool flow) {
        Bins b;
        b.reference_length = Rcpp::as<double>(bins_list["reference_length"]);
        b.edges = fn_as_vector(bins_list["bin_edges_in_doublings"]);
        Rcpp::NumericMatrix sums = bins_list["sum_of_normalised_squared_jumps"];
        b.n_rows = sums.nrow();
        b.n_bins = sums.ncol();
        b.sums = std::vector<double>(sums.begin(), sums.end());
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

Rcpp::NumericMatrix fn_matrix(const std::vector<double> &values, const int n_rows, const int n_cols) {
        Rcpp::NumericMatrix m(n_rows, n_cols);
        std::copy(values.begin(), values.end(), m.begin());
        return m;
}

Rcpp::List fn_write_bins(const Bins &b, const bool flow) {
        if (flow) {
                return Rcpp::List::create(
                      Rcpp::Named("reference_length")                = b.reference_length,
                      Rcpp::Named("bin_edges_in_doublings")          = Rcpp::wrap(b.edges),
                      Rcpp::Named("frequencies")                     = Rcpp::wrap(b.frequencies),
                      Rcpp::Named("sum_of_normalised_squared_jumps") = fn_matrix(b.sums, b.n_rows, b.n_bins),
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
              Rcpp::Named("sum_of_normalised_squared_jumps")  = fn_matrix(b.sums, b.n_rows, b.n_bins),
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
                           const double forgetting_factor) {
        for (double &v : sum_of_squares) v *= forgetting_factor;
        for (int chain : chains) {
                const int bin = bin_index[chain];
                for (int r = 0; r < n_rows; r++) {
                        const double v = values[r + static_cast<size_t>(chain) * n_rows];
                        sum_of_squares[r + static_cast<size_t>(bin) * n_rows] += v * v;
                }
        }
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
};

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

//// ---- the objective (fn_objective of the R code with its defaults) at the candidates `which` (the others NaN),
////      for given quadratic bin means (n_rows x n_nodes) and flow weights:
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

int fn_which_max(const std::vector<double> &x) {
        int best = -1;
        for (int i = 0; i < static_cast<int>(x.size()); i++) {
                if (std::isnan(x[i])) continue;
                if (best < 0 || x[i] > x[best]) best = i;
        }
        return best;
}

}  // namespace

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
                                            const int n_bootstrap_draws) {

        const int n_rows = linear_jump_squared.nrow();
        const int n_chains = static_cast<int>(probabilities.size());
        if (!(std::isfinite(eps_used_for_trajectories) && eps_used_for_trajectories > 0.0)) eps_used_for_trajectories = eps_now;
        auto at = [n_rows](const Rcpp::NumericMatrix &m, const int r, const int c) { return m[r + static_cast<size_t>(c) * n_rows]; };
        // ---- which chains contribute:
        std::vector<int> binned(n_chains), valid(n_chains);
        std::vector<double> weights(n_chains), executed_lengths(n_chains);
        bool any_binned = false, any_valid = false;
        for (int c = 0; c < n_chains; c++) {
                bool finite_statistics = true;
                for (int r = 0; r < n_rows; r++) {
                        if (!std::isfinite(at(linear_jump_squared, r, c)) || !std::isfinite(at(quadratic_jump_squared, r, c)) ||
                            !std::isfinite(at(initial_fourth_moment, r, c))) {
                                finite_statistics = false;
                                break;
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
                for (int r = 0; r < n_rows; r++) {
                        double s2 = 0.0, s4 = 0.0;
                        for (int c = 0; c < n_chains; c++) {
                                if (!valid[c]) continue;
                                s2 += at(initial_second_moment, r, c);
                                s4 += at(initial_fourth_moment, r, c);
                        }
                        second_now[r] = s2 / n_valid;
                        fourth_now[r] = s4 / n_valid;
                }
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
                quadratic = fn_read_bins(Rcpp::as<Rcpp::List>(state["quadratic_state"]), false);
                flow = fn_read_bins(Rcpp::as<Rcpp::List>(state["flow_state"]), true);
                if (!Rf_isNull(state["second_moment_moving_average"])) second_ma = fn_as_vector(state["second_moment_moving_average"]);
                if (!Rf_isNull(state["fourth_moment_moving_average"])) fourth_ma = fn_as_vector(state["fourth_moment_moving_average"]);
                if (!Rf_isNull(state["acceptance_moving_average"])) {
                        acceptance_ma = Rcpp::as<double>(state["acceptance_moving_average"]);
                        has_acceptance_ma = true;
                }
                if (state.containsElementNamed("quadratic_sum_of_squares") && !Rf_isNull(state["quadratic_sum_of_squares"])) {
                        quadratic_sum_of_squares = fn_as_vector(state["quadratic_sum_of_squares"]);
                }
                if (state.containsElementNamed("flow_sum_of_squares") && !Rf_isNull(state["flow_sum_of_squares"])) {
                        flow_sum_of_squares = fn_as_vector(state["flow_sum_of_squares"]);
                }
        }
        if (any_valid) {
                if (second_ma.empty()) {
                        second_ma = second_now;
                        fourth_ma = fourth_now;
                } else {
                        for (int r = 0; r < n_rows; r++) {
                                second_ma[r] = 0.9 * second_ma[r] + 0.1 * second_now[r];
                                fourth_ma[r] = 0.9 * fourth_ma[r] + 0.1 * fourth_now[r];
                        }
                }
        }
        acceptance_ma = has_acceptance_ma ? 0.9 * acceptance_ma + 0.1 * acceptance_now : acceptance_now;
        std::vector<double> variance_linear(n_rows), variance_quadratic(n_rows);
        for (int r = 0; r < n_rows; r++) {
                variance_linear[r] = fn_max_na(second_ma[r], 1e-12);
                const double v = fourth_ma[r] - second_ma[r] * second_ma[r];
                variance_quadratic[r] = (std::isfinite(v) && v > 0.0) ? v : 2.0 * variance_linear[r] * variance_linear[r];
        }
        // ---- the quadratic bins (acceptance-weighted, by the jittered length):
        std::vector<double> quadratic_values(static_cast<size_t>(n_rows) * n_chains);
        for (int c = 0; c < n_chains; c++) {
                for (int r = 0; r < n_rows; r++) {
                        double q = at(quadratic_jump_squared, r, c);
                        if (!std::isfinite(q)) q = 0.0;
                        quadratic_values[r + static_cast<size_t>(c) * n_rows] = (q * weights[c]) / (2.0 * variance_quadratic[r]);
                }
        }
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
        for (double &v : quadratic.sums) v *= bin_forgetting_factor;
        for (double &v : quadratic.counts) v *= bin_forgetting_factor;
        for (double &v : quadratic.length_sums) v *= bin_forgetting_factor;
        for (int c : binned_chains) {
                const int bin = quadratic_bin_index[c];
                for (int r = 0; r < n_rows; r++) {
                        quadratic.sums[r + static_cast<size_t>(bin) * n_rows] += quadratic_values[r + static_cast<size_t>(c) * n_rows];
                }
                quadratic.counts[bin] += 1.0;
                quadratic.length_sums[bin] += tau_values[c];
                quadratic.largest_length[bin] = std::max(quadratic.largest_length[bin], static_cast<double>(tau_values[c]));
        }
        quadratic.n_updates += 1.0;
        if (expansion_evidence) {
                if (quadratic_sum_of_squares.empty()) quadratic_sum_of_squares.assign(quadratic.sums.size(), 0.0);
                fn_add_sum_of_squares(quadratic_sum_of_squares, n_rows, quadratic_values, quadratic_bin_index, binned_chains,
                                      bin_forgetting_factor);
        }
        // ---- the flow bins (unweighted proposal jumps of the valid chains, by the executed length):
        std::vector<double> linear_values(static_cast<size_t>(n_rows) * n_chains);
        for (int c = 0; c < n_chains; c++) {
                for (int r = 0; r < n_rows; r++) {
                        double l = at(linear_jump_squared, r, c);
                        if (!std::isfinite(l)) l = 0.0;
                        linear_values[r + static_cast<size_t>(c) * n_rows] = l / (2.0 * variance_linear[r]);
                }
        }
        if (any_valid) {
                for (double &v : flow.sums) v *= bin_forgetting_factor;
                for (double &v : flow.counts) v *= bin_forgetting_factor;
                for (double &v : flow.length_sums) v *= bin_forgetting_factor;
                for (double &v : flow.cosine_sums) v *= bin_forgetting_factor;
                const int n_frequencies = static_cast<int>(flow.frequencies.size());
                for (int c : valid_chains) {
                        const int bin = flow_bin_index[c];
                        for (int r = 0; r < n_rows; r++) {
                                flow.sums[r + static_cast<size_t>(bin) * n_rows] += linear_values[r + static_cast<size_t>(c) * n_rows];
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
                if (any_valid) {
                        fn_add_sum_of_squares(flow_sum_of_squares, n_rows, linear_values, flow_bin_index, valid_chains,
                                              bin_forgetting_factor);
                }
        }
        // ---- the state to return (the R code's nested list):
        Rcpp::List state_out = Rcpp::List::create(
              Rcpp::Named("quadratic_state")              = fn_write_bins(quadratic, false),
              Rcpp::Named("flow_state")                   = fn_write_bins(flow, true),
              Rcpp::Named("second_moment_moving_average") = Rcpp::wrap(second_ma),
              Rcpp::Named("fourth_moment_moving_average") = Rcpp::wrap(fourth_ma),
              Rcpp::Named("acceptance_moving_average")    = acceptance_ma);
        if (expansion_evidence) {
                state_out.push_back(fn_matrix(quadratic_sum_of_squares, n_rows, quadratic.n_bins), "quadratic_sum_of_squares");
                state_out.push_back(fn_matrix(flow_sum_of_squares, n_rows, flow.n_bins), "flow_sum_of_squares");
        }
        result["state"] = state_out;
        // ---- the candidates and the quadratic bin means:
        std::vector<int> usable;
        for (int bin = 0; bin < quadratic.n_bins; bin++) if (quadratic.counts[bin] >= minimum_bin_count) usable.push_back(bin);
        if (usable.size() < 3) return result;
        std::vector<double> bin_centres;
        for (int bin : usable) bin_centres.push_back(quadratic.length_sums[bin] / quadratic.counts[bin]);
        std::vector<double> bin_means;
        for (int bin : usable) {
                for (int r = 0; r < n_rows; r++) {
                        bin_means.push_back(quadratic.sums[r + static_cast<size_t>(bin) * n_rows] / quadratic.counts[bin]);
                }
        }
        const double longest_length_with_data = quadratic.largest_length[usable.back()];
        if (longest_length_with_data > bin_centres.back()) {
                bin_centres.push_back(longest_length_with_data);
                for (int r = 0; r < n_rows; r++) bin_means.push_back(bin_means[bin_means.size() - n_rows]);
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
        std::vector<double> flow_weights;
        FitWorkspace fit_workspace;
        fn_fit_weights_in_workspace(prep, flow.sums, n_rows, flow_weights, nullptr, fit_workspace);
        std::vector<int> all_candidates(n_candidates);
        for (int c = 0; c < n_candidates; c++) all_candidates[c] = c;
        fn_objective_at(in, prep, bin_means, flow_weights, all_candidates, objective);
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
                // ---- the bootstrap's standard errors (quadratic bins with the extra node; flow bins):
                std::vector<double> quadratic_standard_errors;
                for (int bin : usable) {
                        for (int r = 0; r < n_rows; r++) {
                                const double count = quadratic.counts[bin];
                                const double mean = quadratic.sums[r + static_cast<size_t>(bin) * n_rows] / count;
                                const double variance = fn_max_na(quadratic_sum_of_squares[r + static_cast<size_t>(bin) * n_rows] / count -
                                                                  mean * mean, 0.0);
                                quadratic_standard_errors.push_back(std::sqrt(variance / count));
                        }
                }
                if (bin_means.size() > quadratic_standard_errors.size()) {
                        for (int r = 0; r < n_rows; r++) {
                                quadratic_standard_errors.push_back(quadratic_standard_errors[quadratic_standard_errors.size() - n_rows]);
                        }
                }
                std::vector<int> flow_usable;
                for (int bin = 0; bin < flow.n_bins; bin++) if (flow.counts[bin] >= minimum_bin_count) flow_usable.push_back(bin);
                std::vector<double> flow_means, flow_standard_errors;
                for (int bin : flow_usable) {
                        for (int r = 0; r < n_rows; r++) {
                                const double count = flow.counts[bin];
                                const double mean = flow.sums[r + static_cast<size_t>(bin) * n_rows] / count;
                                flow_means.push_back(mean);
                                flow_standard_errors.push_back(std::sqrt(fn_max_na(flow_sum_of_squares[r + static_cast<size_t>(bin) * n_rows] /
                                                                                   count - mean * mean, 0.0) / count));
                        }
                }
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
                        // ---- the draws, from the bootstrap's own generator, seeded from the update count as R's
                        //      generator was (set.seed(104729 + the quadratic updates so far)); R's stream is not
                        //      touched:
                        BootstrapNormalGenerator generator(static_cast<uint64_t>(104729.0 + quadratic.n_updates));
                        std::vector<double> difference_draws;
                        std::vector<double> perturbed_quadratic(bin_means.size());
                        std::vector<double> perturbed_flow_sums = flow.sums;
                        std::vector<double> draw_objective, draw_weights;
                        const std::vector<int> top_and_reference = {top, reference};
                        const double variance_bound_for_no_expansion = (difference_estimate / z_threshold) *
                                                                       (difference_estimate / z_threshold) *
                                                                       (1.0 + 1e-12);
                        double running_mean = 0.0, running_sum_of_squares = 0.0;
                        int n_finite = 0;
                        bool stopped_early = false;
                        for (int draw = 0; draw < n_bootstrap_draws; draw++) {
                                for (size_t i = 0; i < bin_means.size(); i++) {
                                        perturbed_quadratic[i] = fn_max_na(bin_means[i] +
                                                                           quadratic_standard_errors[i] *
                                                                           generator.fn_normal(), 0.0);
                                }
                                size_t index = 0;
                                for (int bin : flow_usable) {
                                        for (int r = 0; r < n_rows; r++) {
                                                perturbed_flow_sums[r + static_cast<size_t>(bin) * n_rows] =
                                                      fn_max_na(flow_means[index] + flow_standard_errors[index] *
                                                                                    generator.fn_normal(), 0.0) *
                                                      flow.counts[bin];
                                                index++;
                                        }
                                }
                                // (only the two candidates that the test compares; each row's NNLS starts from
                                //  the estimates' support)
                                fn_fit_weights_in_workspace(prep, perturbed_flow_sums, n_rows, draw_weights,
                                                            &flow_weights, fit_workspace);
                                fn_objective_at(in, prep, perturbed_quadratic, draw_weights, top_and_reference,
                                                draw_objective);
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
                                                break;
                                        }
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

}
