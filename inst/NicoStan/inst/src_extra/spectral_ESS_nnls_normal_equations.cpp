//// spectral_ESS_nnls_normal_equations.cpp
////
//// ---- burnin_algorithm = "LQ_ESSR_spectral" (R_fn_spectral_ESS_tau_criterion.R, 6 Oct 2026): non-negative least
////      squares on the normal equations, one right-hand side per column of Atb (one column per monitored coordinate):
////        minimise || A w - b ||^2 subject to w >= 0, given AtA = A^T A (n x n) and Atb = A^T b (n x m).
////      Lawson-Hanson active set on the normal equations (as Bro and De Jong 1997); the passive-set system is solved by
////      Cholesky with a tiny ridge (1e-12 x the largest diagonal entry). A column whose passive-set system is not
////      positive definite drops the entering frequency and does not try it again. R imposes sum(w) = 1 by adding a
////      heavily weighted row of ones to A (i.e. a large constant to every entry of AtA and Atb) and divides each column
////      by its sum afterwards.
////
////      Compiled at first use by Rcpp::sourceCpp() (no rebuild of NicoStan or BayesMVP), as
////      tau_initial_nuisance_moments.cpp.
////

#include <Rcpp.h>
#include <vector>
#include <cmath>
#include <algorithm>


static bool fn_spectral_ESS_cholesky_solve(  std::vector<double> gram,
                                             const std::vector<double> &right_hand_side,
                                             const int n,
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


// [[Rcpp::export]]
Rcpp::NumericMatrix fn_spectral_ESS_nnls_normal_equations(  const Rcpp::NumericMatrix &AtA,
                                                            const Rcpp::NumericMatrix &Atb,
                                                            const double tolerance) {

        const int n = AtA.nrow();
        const int m = Atb.ncol();
        if (AtA.ncol() != n || Atb.nrow() != n) {
          Rcpp::stop("fn_spectral_ESS_nnls_normal_equations: AtA must be n x n and Atb n x m.");
        }
        Rcpp::NumericMatrix X(n, m);
        for (int column = 0; column < m; column++) {
                std::vector<double> x(n, 0.0), w(n), s(n, 0.0);
                std::vector<int> passive(n, 0), excluded(n, 0);
                for (int i = 0; i < n; i++) w[i] = Atb(i, column);
                for (int outer = 0; outer < 3 * n; outer++) {
                        int entering = -1;
                        double largest = tolerance;
                        for (int i = 0; i < n; i++) {
                                if (!passive[i] && !excluded[i] && w[i] > largest) {
                                        largest = w[i];
                                        entering = i;
                                }
                        }
                        if (entering < 0) break;
                        passive[entering] = 1;
                        for (int inner = 0; inner < 3 * n; inner++) {
                                std::vector<int> P;
                                for (int i = 0; i < n; i++) if (passive[i]) P.push_back(i);
                                const int np = static_cast<int>(P.size());
                                if (np == 0) break;
                                std::vector<double> gram(np * np), right_hand_side(np), sub;
                                for (int a = 0; a < np; a++) {
                                        right_hand_side[a] = Atb(P[a], column);
                                        for (int b = 0; b < np; b++) gram[a * np + b] = AtA(P[a], P[b]);
                                }
                                if (!fn_spectral_ESS_cholesky_solve(gram, right_hand_side, np, sub)) {
                                        passive[entering] = 0;
                                        excluded[entering] = 1;
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
                        for (int i = 0; i < n; i++) {
                                double t = Atb(i, column);
                                for (int k = 0; k < n; k++) t -= AtA(i, k) * x[k];
                                w[i] = t;
                        }
                }
                for (int i = 0; i < n; i++) X(i, column) = x[i];
        }
        return X;

}
























