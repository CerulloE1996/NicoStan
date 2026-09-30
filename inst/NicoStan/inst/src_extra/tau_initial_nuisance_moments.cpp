

//// tau_initial_nuisance_moments.cpp
////
//// ---- tau_initial = "adaptive" (init_and_run_burnin_ChESSR): the pooled within-individual second moment of the standardised
////      nuisance coordinates at one burn-in iteration.
////
////      For individual i (row i of positions), chain k and test t, y(i, t) = (u(p, k) - centre(p)) / scale(p), p = positions(i, t).
////      Returns the n_tests x n_tests sum over i and k of y_i y_i^T, where n_tests = ncol(positions). R keeps the running sum and the
////      number of (individual, chain, iteration) terms, and takes the largest eigenvalue of sum / count at the handover.
////
////      u_values:  n_nuisance x n_chains matrix of the nuisance coordinates (one chain per column);
////      centre:    length n_nuisance (the running centre, snaper_m_vec_us);
////      scale:     length n_nuisance (sqrt(M_inv_us_vec), the nuisance metric's own scale);
////      positions: N x n_tests matrix of 1-based positions in the nuisance vector (the model's chunk-major storage order, from
////                 fn_nuisance_chunk_layout_positions; one column when every coordinate is its own block).
////
////      Compiled at first use by Rcpp::sourceCpp() (no rebuild of NicoStan or BayesMVP). Non-finite inputs give a non-finite sum,
////      which R treats as a missing estimate (tau falls back to pi for that block).
////

#include <Rcpp.h>
#include <vector>


// [[Rcpp::export]]
Rcpp::NumericMatrix fn_tau_initial_nuisance_second_moment_sum(  const Rcpp::NumericMatrix &u_values,
                                                                const Rcpp::NumericVector &centre,
                                                                const Rcpp::NumericVector &scale,
                                                                const Rcpp::IntegerMatrix &positions) {

        const int n_nuisance = u_values.nrow();
        const int n_chains   = u_values.ncol();
        const int N          = positions.nrow();
        const int n_tests    = positions.ncol();

        if (centre.size() != n_nuisance || scale.size() != n_nuisance) {
          Rcpp::stop("fn_tau_initial_nuisance_second_moment_sum: centre and scale must have one entry per nuisance coordinate.");
        }
        if (static_cast<long long>(N) * static_cast<long long>(n_tests) != static_cast<long long>(n_nuisance)) {
          Rcpp::stop("fn_tau_initial_nuisance_second_moment_sum: positions must hold N x n_tests = n_nuisance entries.");
        }

        //// 0-based positions (column-major, as R stores the matrix), checked once:
        std::vector<int> position_index(static_cast<size_t>(N) * n_tests);
        for (int t = 0; t < n_tests; ++t) {
          for (int i = 0; i < N; ++i) {
            const int p = positions(i, t) - 1;
            if (p < 0 || p >= n_nuisance) Rcpp::stop("fn_tau_initial_nuisance_second_moment_sum: position out of range.");
            position_index[static_cast<size_t>(t) * N + i] = p;
          }
        }

        //// reciprocal scales, once per call:
        std::vector<double> scale_recip(n_nuisance);
        for (int p = 0; p < n_nuisance; ++p) scale_recip[p] = 1.0 / scale[p];

        //// upper triangle of the sum, then mirrored:
        std::vector<double> sum_upper(static_cast<size_t>(n_tests) * n_tests, 0.0);
        std::vector<double> y(n_tests);

        for (int k = 0; k < n_chains; ++k) {

          const double *u_k = &u_values(0, k);

          for (int i = 0; i < N; ++i) {

            for (int t = 0; t < n_tests; ++t) {
              const int p = position_index[static_cast<size_t>(t) * N + i];
              y[t] = (u_k[p] - centre[p]) * scale_recip[p];
            }

            for (int s = 0; s < n_tests; ++s) {
              const double y_s = y[s];
              double *sum_col_s = &sum_upper[static_cast<size_t>(s) * n_tests];
              for (int t = 0; t <= s; ++t) sum_col_s[t] += y[t] * y_s;
            }

          }

        }

        Rcpp::NumericMatrix second_moment_sum(n_tests, n_tests);
        for (int s = 0; s < n_tests; ++s) {
          for (int t = 0; t <= s; ++t) {
            const double value = sum_upper[static_cast<size_t>(s) * n_tests + t];
            second_moment_sum(t, s) = value;
            second_moment_sum(s, t) = value;
          }
        }

        return second_moment_sum;

}
























