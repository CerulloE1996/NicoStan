#pragma once




#include <cmath>
#include <cfloat>
#include <cstddef>
#include <cstring>
#include <vector>

#include <R_ext/BLAS.h>




//// ---------------------------------------------------------------------------------------------------
//// EXACT R ARITHMETIC FOR THE RESIDENT BURN-IN STATISTICS
////
//// Nuisance-sized reductions which the R burn-in loop (init_and_run_burnin_ChESSR) performs on the
//// matrices returned by the persistent burn-in worker. Each kernel below reproduces the operation order
//// and the intermediate precision of the corresponding R code, so its result is bitwise identical to R's:
////
////   - rowMeans() and rowSums() accumulate each row over the columns in long double (R's LDOUBLE),
////     starting from zero and adding the columns in order; rowMeans() then divides by the number of
////     columns in long double before rounding to double (R 4.3.3, src/main/array.c, do_colsum);
////   - sum() of a double vector accumulates in long double (src/main/summary.c, rsum);
////   - x^2 is computed as x * x (src/main/arithmetic.c, R_pow);
////   - matrix %*% vector calls the BLAS routine dgemv, unless the imprecise non-finite check
////     mayHaveNaNOrInf() fires, in which case R uses its own loop instead (src/main/array.c, matprod);
////   - every other R vector operation is one rounded double operation per element.
////
//// R evaluates each vector operation separately, so a multiply is never fused with an add. The package
//// is compiled with -mfma and GCC's default -ffp-contract=fast, which would fuse them, so contraction is
//// switched off for this block only, and each kernel is kept out of line so that it cannot be inlined
//// into (and re-optimised with) a caller compiled with contraction switched on.
//// Long double accumulations use the x87 unit, which has no fused multiply-add.
//// The package is also compiled with -fno-signed-zeros, which lets GCC drop the leading "0 + x" of an
//// accumulation (R's accumulators start at +0, so a row of -0 entries sums to +0 in R); with GCC, signed
//// zeros are therefore switched back on for this block as well.
//// ---------------------------------------------------------------------------------------------------




#if defined(__clang__)
  #define NICOSTAN_EXACT_R_ARITHMETIC_SCOPE _Pragma("clang fp contract(off)")
#else
  #define NICOSTAN_EXACT_R_ARITHMETIC_SCOPE
#endif

#if defined(__GNUC__) || defined(__clang__)
  #define NICOSTAN_EXACT_R_ARITHMETIC_NOINLINE __attribute__((noinline))
#else
  #define NICOSTAN_EXACT_R_ARITHMETIC_NOINLINE
#endif

#if defined(__GNUC__) && !defined(__clang__)
  #pragma GCC push_options
  #pragma GCC optimize ("fp-contract=off", "signed-zeros")
#endif




//// ---------------------------------------------------------------- rowMeans(x), x given column by column:
////
//// R: rowMeans(x) for an n_rows x n_columns double matrix, na.rm = FALSE.
//// columns[j] points to column j (length n_rows).
////
NICOSTAN_EXACT_R_ARITHMETIC_NOINLINE
inline void fn_exact_R_rowMeans_of_columns(   const double * const *columns,
                                              const int n_columns,
                                              const std::size_t n_rows,
                                              double *row_means_out) {

        NICOSTAN_EXACT_R_ARITHMETIC_SCOPE

        const long double n_columns_ld = static_cast<long double>(n_columns);

        for (std::size_t i = 0; i < n_rows; ++i) {

              long double row_sum = 0.0L;
              for (int j = 0; j < n_columns; ++j) {
                    row_sum += columns[j][i];
              }
              row_sum /= n_columns_ld;
              row_means_out[i] = static_cast<double>(row_sum);

        }

}




//// ---------------------------------------------------------------- pooled Welford update (rows of one block):
////
//// R (metric_estimator == "pooled"), restricted to the rows of one block. Every operation below is
//// row-local, so the rows of the nuisance block give the same numbers as the full R computation:
////
////     K     <- ncol(X_all)
////     m_b   <- rowMeans(X_all)
////     D_b   <- X_all - m_b
////     delta <- m_b - wf_m
////     n_new <- wf_n + K
////     wf_M2 <- wf_M2 + rowSums(D_b^2) + delta^2 * (wf_n * K / n_new)
////     wf_m  <- wf_m + delta * (K / n_new)
////     wf_n  <- n_new
////     if (wf_n >= wf_min_draws) var_draws_all <- wf_M2 / (wf_n - 1)
////
//// wf_n is the count BEFORE this update (after any window reset). Returns the updated count n_new.
//// var_draws is only overwritten when n_new >= wf_min_draws, exactly as in R (otherwise it keeps the
//// value from the last update that reached wf_min_draws).
////
NICOSTAN_EXACT_R_ARITHMETIC_NOINLINE
inline double fn_exact_R_pooled_Welford_rows(   const double * const *columns,
                                                const int n_columns,
                                                const std::size_t n_rows,
                                                const double wf_n,
                                                const double wf_min_draws,
                                                double *wf_m,
                                                double *wf_M2,
                                                double *var_draws) {

        NICOSTAN_EXACT_R_ARITHMETIC_SCOPE

        const double K_dbl = static_cast<double>(n_columns);
        const long double K_ld = static_cast<long double>(n_columns);
        ////
        const double n_new = wf_n + K_dbl;
        const double M2_cross_factor = (wf_n * K_dbl) / n_new;   //// wf_n * K / n_new
        const double mean_step_factor = K_dbl / n_new;           //// K / n_new

        for (std::size_t i = 0; i < n_rows; ++i) {

              //// m_b = rowMeans(X_all):
              long double row_sum = 0.0L;
              for (int j = 0; j < n_columns; ++j) {
                    row_sum += columns[j][i];
              }
              row_sum /= K_ld;
              const double m_b = static_cast<double>(row_sum);

              //// rowSums(D_b^2), with D_b = X_all - m_b and D_b^2 = D_b * D_b:
              long double row_sum_sq = 0.0L;
              for (int j = 0; j < n_columns; ++j) {
                    const double D_b = columns[j][i] - m_b;
                    const double D_b_sq = D_b * D_b;
                    row_sum_sq += D_b_sq;
              }
              const double rowSums_D_b_sq = static_cast<double>(row_sum_sq);

              const double delta = m_b - wf_m[i];

              //// wf_M2 <- (wf_M2 + rowSums(D_b^2)) + ((delta^2) * (wf_n * K / n_new)):
              const double M2_plus_row_sums = wf_M2[i] + rowSums_D_b_sq;
              const double delta_sq = delta * delta;
              const double delta_sq_term = delta_sq * M2_cross_factor;
              wf_M2[i] = M2_plus_row_sums + delta_sq_term;

              //// wf_m <- wf_m + (delta * (K / n_new)):
              const double mean_step = delta * mean_step_factor;
              wf_m[i] = wf_m[i] + mean_step;

        }

        if (n_new >= wf_min_draws) {
              const double n_new_minus_one = n_new - 1.0;
              for (std::size_t i = 0; i < n_rows; ++i) {
                    var_draws[i] = wf_M2[i] / n_new_minus_one;
              }
        }

        return n_new;

}




//// ---------------------------------------------------------------- fn_update_snaper_mean():
////
//// R (R_fns_SNAPER_HMC.R, kappa = 8.0):
////
////     eta_mean <- 1.0 / (ceiling(ii / kappa) + 1.0)
////     if (ii < 2) return(c(theta_mean))
////     return((1.0 - eta_mean) * c(snaper_mean) + eta_mean * c(theta_mean))
////
//// updated_mean may be the same array as snaper_mean (each element is read before it is written).
////
NICOSTAN_EXACT_R_ARITHMETIC_NOINLINE
inline void fn_exact_R_update_snaper_mean(   const double *snaper_mean,
                                             const double *theta_mean,
                                             const std::size_t n,
                                             const double ii,
                                             double *updated_mean) {

        NICOSTAN_EXACT_R_ARITHMETIC_SCOPE

        const double kappa = 8.0;
        const double eta_mean = 1.0 / (std::ceil(ii / kappa) + 1.0);

        if (ii < 2.0) {
              for (std::size_t i = 0; i < n; ++i) updated_mean[i] = theta_mean[i];
              return;
        }

        const double weight_previous = 1.0 - eta_mean;
        for (std::size_t i = 0; i < n; ++i) {
              const double previous_term = weight_previous * snaper_mean[i];
              const double theta_term = eta_mean * theta_mean[i];
              updated_mean[i] = previous_term + theta_term;
        }

}




//// ---------------------------------------------------------------- sum(x), na.rm = FALSE:
NICOSTAN_EXACT_R_ARITHMETIC_NOINLINE
inline double fn_exact_R_sum(   const double *x,
                                const std::size_t n) {

        NICOSTAN_EXACT_R_ARITHMETIC_SCOPE

        long double s = 0.0L;
        for (std::size_t i = 0; i < n; ++i) s += x[i];

        if (s > DBL_MAX)       return R_PosInf;
        else if (s < -DBL_MAX) return R_NegInf;
        return static_cast<double>(s);

}




//// ---------------------------------------------------------------- matprod(): R's imprecise non-finite check:
////
//// TRUE means R's %*% would NOT call the BLAS for this operand (the sum of two adjacent entries is not finite),
//// so the BLAS result below would not be the number R computes.
////
NICOSTAN_EXACT_R_ARITHMETIC_NOINLINE
inline bool fn_exact_R_matprod_may_have_NaN_or_Inf(   const double *x,
                                                      const std::size_t n) {

        NICOSTAN_EXACT_R_ARITHMETIC_SCOPE

        if (((n & 1) != 0) && !std::isfinite(x[0])) return true;
        for (std::size_t i = (n & 1); i < n; i += 2) {
              const double pair_sum = x[i] + x[i + 1];
              if (!std::isfinite(pair_sum)) return true;
        }
        return false;

}




//// ---------------------------------------------------------------- fn_weighted_proposal_mean():
////
//// R (R_fns_SNAPER_HMC.R):
////
////     valid <- is.finite(p) & p > 0 & is.finite(d) & d == 0 & colSums(!is.finite(proposals)) == 0
////     if (!any(valid)) return(c(previous_mean))
////     weights <- p[valid]
////     weights <- weights / sum(weights)
////     return(c(proposals[, valid, drop = FALSE] %*% weights))
////
//// proposal_columns[k] points to the (n_rows) proposal of chain k, in the row order R uses
//// (rbind(nuisance, main)); the caller passes a staging matrix, so the columns can come from separate
//// nuisance and main vectors. The dimension check (length(p) == ncol(proposals), ...) is the caller's.
////
//// Returns:
////   0 - the weighted mean was computed through dgemv, exactly as R's default matprod does;
////   1 - no chain is valid: previous_mean was returned (R: return(c(previous_mean)));
////   2 - R's non-finite check would have sent %*% to its non-BLAS loop: nothing was computed and the
////       caller must use the R code for this call.
////
//// The matprod option is the caller's responsibility: R only uses this path for options(matprod = "default").
////
NICOSTAN_EXACT_R_ARITHMETIC_NOINLINE
inline int fn_exact_R_weighted_proposal_mean(   const std::vector<const double *> &proposal_columns_nuisance,
                                                const std::vector<const double *> &proposal_columns_main,
                                                const std::size_t n_rows_nuisance,
                                                const std::size_t n_rows_main,
                                                const double *acceptance_probabilities,
                                                const double *divergences,
                                                const int n_chains,
                                                const double *previous_mean,
                                                std::vector<double> &staging_matrix,
                                                std::vector<double> &staging_weights,
                                                double *weighted_mean_out) {

        NICOSTAN_EXACT_R_ARITHMETIC_SCOPE

        const std::size_t n_rows = n_rows_nuisance + n_rows_main;

        //// ---- which chains are valid (the colSums(!is.finite(proposals)) == 0 term scans every row):
        std::vector<int> valid_chains;
        valid_chains.reserve(static_cast<std::size_t>(n_chains));
        for (int k = 0; k < n_chains; ++k) {

              const double p_k = acceptance_probabilities[k];
              const double d_k = divergences[k];
              bool is_valid = std::isfinite(p_k) && (p_k > 0.0) && std::isfinite(d_k) && (d_k == 0.0);
              if (is_valid) {
                    const double *column_nuisance = proposal_columns_nuisance[k];
                    for (std::size_t i = 0; i < n_rows_nuisance; ++i) {
                          if (!std::isfinite(column_nuisance[i])) { is_valid = false; break; }
                    }
              }
              if (is_valid) {
                    const double *column_main = proposal_columns_main[k];
                    for (std::size_t i = 0; i < n_rows_main; ++i) {
                          if (!std::isfinite(column_main[i])) { is_valid = false; break; }
                    }
              }
              if (is_valid) valid_chains.push_back(k);

        }

        if (valid_chains.empty()) {
              for (std::size_t i = 0; i < n_rows; ++i) weighted_mean_out[i] = previous_mean[i];
              return 1;
        }

        //// ---- weights <- p[valid]; weights <- weights / sum(weights):
        const int n_valid = static_cast<int>(valid_chains.size());
        staging_weights.resize(static_cast<std::size_t>(n_valid));
        for (int v = 0; v < n_valid; ++v) staging_weights[v] = acceptance_probabilities[valid_chains[v]];
        const double sum_of_weights = fn_exact_R_sum(staging_weights.data(), static_cast<std::size_t>(n_valid));
        for (int v = 0; v < n_valid; ++v) staging_weights[v] = staging_weights[v] / sum_of_weights;

        //// ---- proposals[, valid, drop = FALSE]: one contiguous n_rows x n_valid column-major matrix, as R builds it:
        staging_matrix.resize(n_rows * static_cast<std::size_t>(n_valid));
        for (int v = 0; v < n_valid; ++v) {
              double *destination = staging_matrix.data() + static_cast<std::size_t>(v) * n_rows;
              const double *column_nuisance = proposal_columns_nuisance[valid_chains[v]];
              const double *column_main = proposal_columns_main[valid_chains[v]];
              for (std::size_t i = 0; i < n_rows_nuisance; ++i) destination[i] = column_nuisance[i];
              for (std::size_t i = 0; i < n_rows_main; ++i)     destination[n_rows_nuisance + i] = column_main[i];
        }

        //// ---- %*%: R's default matprod checks both operands before trusting the BLAS:
        if (fn_exact_R_matprod_may_have_NaN_or_Inf(staging_matrix.data(), staging_matrix.size()) ||
            fn_exact_R_matprod_may_have_NaN_or_Inf(staging_weights.data(), staging_weights.size())) {
              return 2;
        }

        if (n_rows == 0) return 0;

        const char trans_N = 'N';
        const int n_rows_int = static_cast<int>(n_rows);
        const int n_valid_int = n_valid;
        const double one = 1.0;
        const double zero = 0.0;
        const int increment_one = 1;
        F77_CALL(dgemv)(&trans_N, &n_rows_int, &n_valid_int, &one, staging_matrix.data(),
                        &n_rows_int, staging_weights.data(), &increment_one, &zero, weighted_mean_out, &increment_one FCONE);

        return 0;

}




//// ---------------------------------------------------------------------------------------------------
//// JOINT (MAIN + NUISANCE) TRAJECTORY BLOCK (tau_adaptation_block = "joint")
////
//// The kernels below reproduce the nuisance-sized parts of the R trajectory-length code for the joint block
//// (R_fn_metric_trajectory_adaptation.R, R_fns_SNAPER_HMC.R and R_fns_ChESSR_HMC.R). A joint position or
//// velocity has the main rows first, then the nuisance rows. The main rows (which may involve a dense
//// metric factor) are computed in R, by the same R code on the same numbers, and passed in; the nuisance
//// rows are formed here, one rounded double operation per element, in R's order. Every reduction over the
//// joint rows then runs over the main rows first and the nuisance rows after them, as R's does.
//// ---------------------------------------------------------------------------------------------------




//// ---------------------------------------------------------------- sum(x^2):
////
//// R: sum(x^2): x^2 is the double x * x of every element, then sum() accumulates in long double.
////
NICOSTAN_EXACT_R_ARITHMETIC_NOINLINE
inline double fn_exact_R_sum_of_squares(   const double *x,
                                           const std::size_t n) {

        NICOSTAN_EXACT_R_ARITHMETIC_SCOPE

        long double s = 0.0L;
        for (std::size_t i = 0; i < n; ++i) {
              const double x_sq = x[i] * x[i];
              s += x_sq;
        }

        if (s > DBL_MAX)       return R_PosInf;
        else if (s < -DBL_MAX) return R_NegInf;
        return static_cast<double>(s);

}




//// ---------------------------------------------------------------- sum(v^2 * mass):
////
//// R (R_fn_kinetic_energy_change, diagonal mass): sum(velocity^2 * c(mass_matrix)), i.e. the double
//// (v * v) * mass of every element, accumulated in long double by sum().
////
NICOSTAN_EXACT_R_ARITHMETIC_NOINLINE
inline double fn_exact_R_sum_of_squares_times_mass(   const double *velocity,
                                                      const double *mass,
                                                      const std::size_t n) {

        NICOSTAN_EXACT_R_ARITHMETIC_SCOPE

        long double s = 0.0L;
        for (std::size_t i = 0; i < n; ++i) {
              const double velocity_sq = velocity[i] * velocity[i];
              const double term = velocity_sq * mass[i];
              s += term;
        }

        if (s > DBL_MAX)       return R_PosInf;
        else if (s < -DBL_MAX) return R_NegInf;
        return static_cast<double>(s);

}




//// ---------------------------------------------------------------- nuisance rows in metric coordinates:
////
//// R (fn_apply_trajectory_metric with a joint factor, nuisance rows): metric_factor$us * (x - m), i.e. the
//// double difference first, then the double product; or metric_factor$us * x when centre is a null pointer.
////
NICOSTAN_EXACT_R_ARITHMETIC_NOINLINE
inline void fn_exact_R_nuisance_rows_in_metric_coordinates(   const double *x,
                                                              const double *centre,
                                                              const double *factor_us,
                                                              const std::size_t n_us,
                                                              double *out) {

        NICOSTAN_EXACT_R_ARITHMETIC_SCOPE

        if (centre == nullptr) {
              for (std::size_t i = 0; i < n_us; ++i) out[i] = factor_us[i] * x[i];
        } else {
              for (std::size_t i = 0; i < n_us; ++i) {
                    const double difference = x[i] - centre[i];
                    out[i] = factor_us[i] * difference;
              }
        }

}




//// ---------------------------------------------------------------- ChEES / ChEES-rate column sums, joint block:
////
//// R (fn_metric_position_criterion, algorithm != "SNAPER"), for the joint block:
////
////     initial  <- fn_apply_trajectory_metric(metric_factor, theta_initial - mean_initial)
////     proposed <- fn_apply_trajectory_metric(metric_factor, theta_proposed - mean_proposed)
////     velocity <- fn_apply_trajectory_metric(metric_factor, velocity_proposed)
////     colSums(proposed^2), colSums(initial^2), colSums(proposed * velocity)
////
//// colSums() accumulates each column in long double over the rows in order (R 4.3.3, src/main/array.c,
//// do_colsum), here the main rows (initial_main etc., n_main x n_columns, from R) and then the nuisance rows,
//// formed from the resident state as in fn_exact_R_nuisance_rows_in_metric_coordinates().
////
NICOSTAN_EXACT_R_ARITHMETIC_NOINLINE
inline void fn_exact_R_joint_position_column_sums(   const double *initial_main,
                                                     const double *proposed_main,
                                                     const double *velocity_main,
                                                     const std::size_t n_main,
                                                     const std::vector<const double *> &theta_initial_us_columns,
                                                     const std::vector<const double *> &theta_proposed_us_columns,
                                                     const std::vector<const double *> &velocity_proposed_us_columns,
                                                     const double *mean_initial_us,
                                                     const double *mean_proposed_us,
                                                     const double *factor_us,
                                                     const std::size_t n_us,
                                                     const int n_columns,
                                                     double *column_sums_proposed_sq,
                                                     double *column_sums_initial_sq,
                                                     double *column_sums_proposed_velocity) {

        NICOSTAN_EXACT_R_ARITHMETIC_SCOPE

        for (int k = 0; k < n_columns; ++k) {

              const std::size_t offset_main = static_cast<std::size_t>(k) * n_main;
              const double *theta_initial_us = theta_initial_us_columns[k];
              const double *theta_proposed_us = theta_proposed_us_columns[k];
              const double *velocity_proposed_us = velocity_proposed_us_columns[k];

              long double sum_proposed_sq = 0.0L;
              long double sum_initial_sq = 0.0L;
              long double sum_proposed_velocity = 0.0L;

              for (std::size_t i = 0; i < n_main; ++i) {
                    const double proposed = proposed_main[offset_main + i];
                    const double initial = initial_main[offset_main + i];
                    const double velocity = velocity_main[offset_main + i];
                    const double proposed_sq = proposed * proposed;
                    const double initial_sq = initial * initial;
                    const double proposed_velocity = proposed * velocity;
                    sum_proposed_sq += proposed_sq;
                    sum_initial_sq += initial_sq;
                    sum_proposed_velocity += proposed_velocity;
              }

              for (std::size_t i = 0; i < n_us; ++i) {
                    const double proposed_difference = theta_proposed_us[i] - mean_proposed_us[i];
                    const double proposed = factor_us[i] * proposed_difference;
                    const double initial_difference = theta_initial_us[i] - mean_initial_us[i];
                    const double initial = factor_us[i] * initial_difference;
                    const double velocity = factor_us[i] * velocity_proposed_us[i];
                    const double proposed_sq = proposed * proposed;
                    const double initial_sq = initial * initial;
                    const double proposed_velocity = proposed * velocity;
                    sum_proposed_sq += proposed_sq;
                    sum_initial_sq += initial_sq;
                    sum_proposed_velocity += proposed_velocity;
              }

              column_sums_proposed_sq[k] = static_cast<double>(sum_proposed_sq);
              column_sums_initial_sq[k] = static_cast<double>(sum_initial_sq);
              column_sums_proposed_velocity[k] = static_cast<double>(sum_proposed_velocity);

        }

}




//// ---------------------------------------------------------------- c(crossprod(v, Y)) for a vector v:
////
//// R: crossprod(v, Y) with v a vector of length n_rows (a column vector) and Y an n_rows x n_columns matrix.
//// R's crossprod() (src/main/array.c) checks both operands with mayHaveNaNOrInf() and, when neither fires,
//// calls dgemv("T", n_rows, n_columns, 1, Y, n_rows, v, 1, 0, z, 1).
////
//// Returns 0 (z written through the BLAS, as R does), or 2 (R would have used its own loop instead: nothing
//// was computed, and the caller must use the R code).
////
NICOSTAN_EXACT_R_ARITHMETIC_NOINLINE
inline int fn_exact_R_crossprod_vector_matrix(   const double *v,
                                                 const double *Y,
                                                 const std::size_t n_rows,
                                                 const int n_columns,
                                                 double *z) {

        NICOSTAN_EXACT_R_ARITHMETIC_SCOPE

        if (fn_exact_R_matprod_may_have_NaN_or_Inf(v, n_rows) ||
            fn_exact_R_matprod_may_have_NaN_or_Inf(Y, n_rows * static_cast<std::size_t>(n_columns))) {
              return 2;
        }

        const char trans_T = 'T';
        const int n_rows_int = static_cast<int>(n_rows);
        const int n_columns_int = n_columns;
        const double one = 1.0;
        const double zero = 0.0;
        const int increment_one = 1;
        F77_CALL(dgemv)(&trans_T, &n_rows_int, &n_columns_int, &one, Y,
                        &n_rows_int, v, &increment_one, &zero, z, &increment_one FCONE);

        return 0;

}




//// ---------------------------------------------------------------- fn_update_snaper_w_minibatch(), after the metric:
////
//// R (R_fns_SNAPER_HMC.R), with X already in metric coordinates (n_rows x n_columns, column-major):
////
////     direction <- c(snaper_w_vec)
////     norm <- sqrt(sum(direction^2))
////     if (!is.finite(norm) || norm == 0) direction <- fn_initialise_snaper_direction(nrow(X)) else direction <- direction / norm
////     candidate <- c(X %*% crossprod(X, direction))
////     magnitude <- sqrt(sum(candidate^2))
////     if (!is.finite(magnitude)) return(direction)
////     if (magnitude == 0) {
////         column_norms <- colSums(X^2)
////         if (max(column_norms) == 0) return(direction)
////         candidate <- X[, which.max(column_norms)]
////         magnitude <- sqrt(sum(candidate^2))
////     }
////     updated <- direction + eta_w * candidate / magnitude
////     norm <- sqrt(sum(updated^2))
////     if (!is.finite(norm) || norm == 0) return(direction)
////     return(c(updated) / norm)
////
//// crossprod(X, direction) is dgemv("T", ...) and X %*% (that n_columns x 1 matrix) is dgemv("N", ...), each only
//// when R's mayHaveNaNOrInf() check does not fire (R 4.3.3, src/main/array.c, crossprod() and matprod()).
////
//// Returns 0 (direction_out written; it may be the same array as snaper_w_vec, which is only read before
//// direction_out is written), or 2 (one of R's two products would not have used the BLAS: nothing written,
//// and the caller must use the R code).
////
NICOSTAN_EXACT_R_ARITHMETIC_NOINLINE
inline int fn_exact_R_update_snaper_w_minibatch(   const double *X,
                                                   const std::size_t n_rows,
                                                   const int n_columns,
                                                   const double *snaper_w_vec,
                                                   const double eta_w,
                                                   std::vector<double> &work_direction,
                                                   std::vector<double> &work_candidate,
                                                   double *direction_out) {

        NICOSTAN_EXACT_R_ARITHMETIC_SCOPE

        const std::size_t n_columns_size = static_cast<std::size_t>(n_columns);

        //// direction <- c(snaper_w_vec); norm <- sqrt(sum(direction^2)):
        work_direction.resize(n_rows);
        double *direction = work_direction.data();
        const double norm_initial = std::sqrt(fn_exact_R_sum_of_squares(snaper_w_vec, n_rows));
        if (!std::isfinite(norm_initial) || norm_initial == 0.0) {
              const double initial_entry = 1.0 / std::sqrt(static_cast<double>(n_rows));
              for (std::size_t i = 0; i < n_rows; ++i) direction[i] = initial_entry;
        } else {
              for (std::size_t i = 0; i < n_rows; ++i) direction[i] = snaper_w_vec[i] / norm_initial;
        }

        //// crossprod(X, direction):
        if (fn_exact_R_matprod_may_have_NaN_or_Inf(X, n_rows * n_columns_size) ||
            fn_exact_R_matprod_may_have_NaN_or_Inf(direction, n_rows)) {
              return 2;
        }
        std::vector<double> projection(n_columns_size);
        {
              const char trans_T = 'T';
              const int n_rows_int = static_cast<int>(n_rows);
              const int n_columns_int = n_columns;
              const double one = 1.0;
              const double zero = 0.0;
              const int increment_one = 1;
              F77_CALL(dgemv)(&trans_T, &n_rows_int, &n_columns_int, &one, X,
                              &n_rows_int, direction, &increment_one, &zero, projection.data(), &increment_one FCONE);
        }

        //// X %*% (crossprod result, an n_columns x 1 matrix):
        if (fn_exact_R_matprod_may_have_NaN_or_Inf(projection.data(), n_columns_size)) {
              return 2;
        }
        work_candidate.resize(n_rows);
        {
              const char trans_N = 'N';
              const int n_rows_int = static_cast<int>(n_rows);
              const int n_columns_int = n_columns;
              const double one = 1.0;
              const double zero = 0.0;
              const int increment_one = 1;
              F77_CALL(dgemv)(&trans_N, &n_rows_int, &n_columns_int, &one, X,
                              &n_rows_int, projection.data(), &increment_one, &zero, work_candidate.data(), &increment_one FCONE);
        }
        const double *candidate = work_candidate.data();

        double magnitude = std::sqrt(fn_exact_R_sum_of_squares(candidate, n_rows));
        if (!std::isfinite(magnitude)) {
              for (std::size_t i = 0; i < n_rows; ++i) direction_out[i] = direction[i];
              return 0;
        }
        if (magnitude == 0.0) {
              //// column_norms <- colSums(X^2); max(); which.max() (the first maximum):
              double max_column_norm = 0.0;
              int which_max_column = -1;
              for (int k = 0; k < n_columns; ++k) {
                    long double column_sum = 0.0L;
                    const double *column = X + static_cast<std::size_t>(k) * n_rows;
                    for (std::size_t i = 0; i < n_rows; ++i) {
                          const double x_sq = column[i] * column[i];
                          column_sum += x_sq;
                    }
                    const double column_norm = static_cast<double>(column_sum);
                    if (!std::isnan(column_norm) && ((which_max_column < 0) || (column_norm > max_column_norm))) {
                          max_column_norm = column_norm;
                          which_max_column = k;
                    }
              }
              if ((which_max_column < 0) || (max_column_norm == 0.0)) {
                    for (std::size_t i = 0; i < n_rows; ++i) direction_out[i] = direction[i];
                    return 0;
              }
              candidate = X + static_cast<std::size_t>(which_max_column) * n_rows;
              magnitude = std::sqrt(fn_exact_R_sum_of_squares(candidate, n_rows));
        }

        //// updated <- direction + ((eta_w * candidate) / magnitude) (candidate may point into X):
        std::vector<double> updated(n_rows);
        for (std::size_t i = 0; i < n_rows; ++i) {
              const double scaled = eta_w * candidate[i];
              const double step = scaled / magnitude;
              updated[i] = direction[i] + step;
        }
        const double norm_updated = std::sqrt(fn_exact_R_sum_of_squares(updated.data(), n_rows));
        if (!std::isfinite(norm_updated) || norm_updated == 0.0) {
              for (std::size_t i = 0; i < n_rows; ++i) direction_out[i] = direction[i];
              return 0;
        }
        for (std::size_t i = 0; i < n_rows; ++i) direction_out[i] = updated[i] / norm_updated;
        return 0;

}




//// ---------------------------------------------------------------- fn_transport_snaper_direction(), joint factor:
////
//// R (R_fn_metric_trajectory_adaptation.R), for a joint (list) metric factor that differs from the previous one:
////
////     transported <- c(fn_transport_snaper_direction(direction[main_rows], previous_factor$main, metric_factor$main) *
////                          sqrt(sum(direction[main_rows]^2)),
////                      fn_transport_snaper_direction(direction[-main_rows], previous_factor$us, metric_factor$us) *
////                          sqrt(sum(direction[-main_rows]^2)))
////     magnitude <- sqrt(sum(transported^2))
////     if (!is.finite(magnitude) || magnitude == 0) return(fn_initialise_snaper_direction(length(direction)))
////     return(transported / magnitude)
////
//// transported_main (the first element of c(), main-sized) is computed in R. The nuisance call, for the
//// diagonal factors previous_factor_us and new_factor_us, is
////
////     if (identical(previous_factor, metric_factor)) return(direction)          (us_factor_unchanged = true)
////     transported <- metric_factor * (direction / previous_factor)
////     magnitude <- sqrt(sum(transported^2))
////     if (!is.finite(magnitude) || magnitude == 0) return(fn_initialise_snaper_direction(length(direction)))
////     return(transported / magnitude)
////
//// direction_out may be the same array as direction (it is written last).
////
NICOSTAN_EXACT_R_ARITHMETIC_NOINLINE
inline void fn_exact_R_transport_joint_snaper_direction(   const double *direction,
                                                           const std::size_t n_main,
                                                           const std::size_t n_us,
                                                           const double *transported_main,
                                                           const double *previous_factor_us,
                                                           const double *new_factor_us,
                                                           const bool us_factor_unchanged,
                                                           std::vector<double> &work,
                                                           double *direction_out) {

        NICOSTAN_EXACT_R_ARITHMETIC_SCOPE

        const std::size_t n_all = n_main + n_us;
        work.resize(n_all);
        for (std::size_t i = 0; i < n_main; ++i) work[i] = transported_main[i];

        const double *direction_us = direction + n_main;
        double *transported_us = work.data() + n_main;

        //// the nuisance call:
        if (us_factor_unchanged) {
              for (std::size_t i = 0; i < n_us; ++i) transported_us[i] = direction_us[i];
        } else {
              for (std::size_t i = 0; i < n_us; ++i) {
                    const double theta_space_direction = direction_us[i] / previous_factor_us[i];
                    transported_us[i] = new_factor_us[i] * theta_space_direction;
              }
              const double magnitude_us = std::sqrt(fn_exact_R_sum_of_squares(transported_us, n_us));
              if (!std::isfinite(magnitude_us) || magnitude_us == 0.0) {
                    const double initial_entry = 1.0 / std::sqrt(static_cast<double>(n_us));
                    for (std::size_t i = 0; i < n_us; ++i) transported_us[i] = initial_entry;
              } else {
                    for (std::size_t i = 0; i < n_us; ++i) transported_us[i] = transported_us[i] / magnitude_us;
              }
        }
        //// ... * sqrt(sum(direction[-main_rows]^2)):
        const double scale_us = std::sqrt(fn_exact_R_sum_of_squares(direction_us, n_us));
        for (std::size_t i = 0; i < n_us; ++i) transported_us[i] = transported_us[i] * scale_us;

        //// the joint normalisation:
        const double magnitude = std::sqrt(fn_exact_R_sum_of_squares(work.data(), n_all));
        if (!std::isfinite(magnitude) || magnitude == 0.0) {
              const double initial_entry = 1.0 / std::sqrt(static_cast<double>(n_all));
              for (std::size_t i = 0; i < n_all; ++i) direction_out[i] = initial_entry;
        } else {
              for (std::size_t i = 0; i < n_all; ++i) direction_out[i] = work[i] / magnitude;
        }

}




#if defined(__GNUC__) && !defined(__clang__)
  #pragma GCC pop_options
#endif
























