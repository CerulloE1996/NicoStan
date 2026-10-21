#pragma once
//
// -| --------- BayesMVP GHK log likelihood of a finite mixture of multivariate (ordered) probit models, ---------
// -| --------- computed in tiles of subjects (Stan external function) -------------------------------------------
//
//     nico_finite_mixture_of_multivariate_ordered_probit_GHK_log_likelihood_in_tiles_of_subjects(
//           u_raw_of_each_test_and_subject, first_subject, nuisance_transformation, intercept, coefficients,
//           coefficient_set_of_class_and_test, covariates, n_covariates, cutpoints,
//           cutpoint_set_of_class_and_test,
//           position_of_first_cutpoint_of_test, L_Omega, log_class_probability, category, upper_category,
//           test_is_binary, n_categories_of_test, test_at_GHK_position, Phi_type, overflow_threshold,
//           underflow_threshold)
//
// The same model, the same value and the same gradient as
// nico_finite_mixture_of_multivariate_ordered_probit_GHK_log_likelihood
// (stan_external_finite_mixture_of_multivariate_ordered_probit_GHK_log_likelihood.hpp, whose SIMD kernels and
// rows of each branch are called here unchanged), computed in three different ways:
//   (1) the M subjects of the slice are taken in tiles of at most 128 subjects: the forward pass of every class,
//       the mixture over the classes and the reverse sweep of one tile are all made before the next tile starts,
//       so that the buffers of a tile (about 1.5 KB per subject for C = 2 and T = 5, i.e. about 0.2 MB for a
//       tile) stay in each core's L2 cache whatever M is. The function above makes 20-30 passes over buffers of
//       all M subjects, which leave the L2 cache once M is large (1 MB per core on the local-HPC, 512 KB on the
//       laptop);
//   (2) the nuisance values are one T x M matrix (one column per subject), so that their values are read, and
//       their partial derivatives written, as one contiguous block of T x 128 numbers per tile;
//   (3) category, upper_category and covariates are the full data arrays (every subject of the data), read at
//       the rows first_subject, ..., first_subject + M - 1: no copy of the slice's data is made at each call.
// The gradients with respect to the intercepts, the coefficients, the cutpoints and L_Omega are summed over the
// tiles; the partial derivatives with respect to the nuisance values and to log_class_probability are written
// tile by tile, straight into the node's partials.
//
// The arguments are those of the function above, except:
//   u_raw_of_each_test_and_subject   T x M matrix (column m: the nuisance values of the slice's subject m, in
//                                    GHK order)
//   first_subject                    the row (1, ..., N) of category, upper_category and covariates of the
//                                    slice's first subject (reduce_sum's start)
//   covariates, category, upper_category   the full data arrays (at least first_subject + M - 1 rows)
//
// Declared in the Stan functions block (no body; stanc --allow-undefined):
//
//     real nico_finite_mixture_of_multivariate_ordered_probit_GHK_log_likelihood_in_tiles_of_subjects(
//           matrix u_raw_of_each_test_and_subject, data int first_subject, data int nuisance_transformation,
//           matrix intercept, array[] matrix coefficients, data array[,] int coefficient_set_of_class_and_test,
//           data array[,] matrix covariates, data array[,] int n_covariates, array[] vector cutpoints,
//           data array[,] int cutpoint_set_of_class_and_test,
//           data array[] int position_of_first_cutpoint_of_test,
//           array[] matrix L_Omega, matrix log_class_probability, data array[,] int category,
//           data array[,] int upper_category, data array[] int test_is_binary,
//           data array[] int n_categories_of_test, data array[] int test_at_GHK_position, data int Phi_type,
//           data real overflow_threshold, data real underflow_threshold);
//
// This header is the user header of the versions that call it (it includes stan_external_functions.hpp); it is
// not included by stan_external_functions.hpp, so that no library built with that header changes.
//
// -----------------------------------------------------------------------------------------------------------

#include "stan_external_functions.hpp"
#include "stan_external_finite_mixture_of_multivariate_ordered_probit_GHK_log_likelihood.hpp"

#include <stan/math/rev.hpp>
#include <stan/math/prim.hpp>

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace nico::math {

//// ---- the largest number of subjects in a tile
static constexpr int LARGEST_NUMBER_OF_SUBJECTS_IN_A_TILE = 128;

//// ---- the thread-local buffers of one tile (each grows to the largest tile seen, then is reused); apart from
////      those of the function above, so that the two never share a buffer
struct buffers_of_mixture_of_multivariate_ordered_probit_GHK_in_tiles_of_subjects_t {
    std::vector<double> values_of_each_subject_and_position;
    std::vector<double> values_of_each_class_subject_and_position;
    std::vector<double> values_of_each_subject_and_class;
    std::vector<double> values_of_each_subject;
    std::vector<double> values_of_each_set_test_and_subject;
    std::vector<int>    codes_of_each_subject_and_test;
    std::vector<int>    rows_of_each_branch;
};

inline buffers_of_mixture_of_multivariate_ordered_probit_GHK_in_tiles_of_subjects_t&
buffers_of_mixture_of_multivariate_ordered_probit_GHK_in_tiles_of_subjects() {

    static thread_local buffers_of_mixture_of_multivariate_ordered_probit_GHK_in_tiles_of_subjects_t buffers;
    return buffers;

}

inline void stop_with_a_message_about_the_arguments_of_the_function_in_tiles(const std::string& message) {

    throw std::domain_error(std::string("nico_finite_mixture_of_multivariate_ordered_probit_GHK_log_likelihood_")
                            + "in_tiles_of_subjects: " + message);

}

//// ---- the value of one tile of n subjects (the slice's subjects first_column, ..., first_column + n - 1, the
////      data's rows first_data_row, ..., first_data_row + n - 1), and, with the gradient: the partial
////      derivatives of its nuisance values ([i * T + j] from gradient_of_u_raw_of_tile, the tile's columns of
////      the T x M partials) and of its log class probabilities ([c * M + first_column + i] from
////      partials_of_log_class_probability, the M x C partials), each written only when its pointer is not
////      nullptr, and the gradients of the intercepts, coefficients, cutpoints and L_Omega added to the sums of
////      the tiles before it. The steps are those of the function above, on n subjects in place of M.
template <bool gradient_is_needed, typename T_u_raw_ref, typename T_log_class_probability_ref>
inline double GHK_mixture_log_likelihood_of_one_tile_of_subjects(
        const int n,
        const int first_column,
        const int first_data_row,
        const int M,
        const T_u_raw_ref& u_raw_ref,
        const int nuisance_transformation,
        const Eigen::Matrix<double, -1, -1>& intercept_values,
        const std::vector<Eigen::Matrix<double, -1, -1>>& coefficient_values,
        const std::vector<std::vector<int>>& coefficient_set_of_class_and_test,
        const std::vector<std::vector<Eigen::Matrix<double, -1, -1>>>& covariates,
        const std::vector<std::vector<int>>& n_covariates,
        const bool there_are_covariates,
        const std::vector<Eigen::Matrix<double, -1, 1>>& cutpoint_values,
        const std::vector<std::vector<int>>& cutpoint_set_of_class_and_test,
        const std::vector<int>& position_of_first_cutpoint_of_test,
        const std::vector<Eigen::Matrix<double, -1, -1>>& L_values,
        const T_log_class_probability_ref& log_class_probability_ref,
        const std::vector<std::vector<int>>& category,
        const std::vector<std::vector<int>>& upper_category,
        const std::vector<int>& test_is_binary,
        const std::vector<int>& n_categories_of_test,
        const std::vector<int>& test_at_GHK_position,
        const int Phi_type,
        const double overflow_threshold,
        const double underflow_threshold,
        double* gradient_of_u_raw_of_tile,
        double* partials_of_log_class_probability,
        Eigen::Matrix<double, -1, -1>& gradient_of_intercept,
        std::vector<Eigen::Matrix<double, -1, -1>>& gradient_of_coefficients,
        std::vector<Eigen::Matrix<double, -1, 1>>& gradient_of_cutpoints,
        std::vector<Eigen::Matrix<double, -1, -1>>& gradient_of_L) {

    using stan::math::value_of;
    const double infinity = std::numeric_limits<double>::infinity();
    const int C = static_cast<int>(L_values.size());
    const int T = static_cast<int>(intercept_values.cols());
    const int n_coefficient_sets = static_cast<int>(coefficient_values.size());
    const int n_covariate_sets = static_cast<int>(covariates.size());

    //// ---- the buffers of the tile:
    buffers_of_mixture_of_multivariate_ordered_probit_GHK_in_tiles_of_subjects_t& buffers =
          buffers_of_mixture_of_multivariate_ordered_probit_GHK_in_tiles_of_subjects();
    const std::size_t nT = static_cast<std::size_t>(n) * T;
    const std::size_t nC = static_cast<std::size_t>(n) * C;
    double* u_raw_values = buffer_of_at_least(buffers.values_of_each_subject_and_position, 9 * nT);
    double* u_values = u_raw_values + nT;                           // [j * n + i], GHK positions
    double* one_minus_u_values = u_values + nT;
    double* log_u_values = one_minus_u_values + nT;
    double* log_1m_u_values = log_u_values + nT;
    double* log_derivative_of_u = log_1m_u_values + nT;
    double* derivative_of_log_Jacobian = log_derivative_of_u + nT;
    double* Z_of_missing_results = derivative_of_log_Jacobian + nT;
    double* gradient_of_u_raw = Z_of_missing_results + nT;
    double* values_of_each_class_subject_and_position = buffer_of_at_least(
          buffers.values_of_each_class_subject_and_position, 8 * C * nT);
    double* log_mass_sum_of_each_class = buffer_of_at_least(buffers.values_of_each_subject_and_class, 2 * nC);
    double* weight_of_each_class = log_mass_sum_of_each_class + nC;   // [c * n + i]
    double* eta = buffer_of_at_least(buffers.values_of_each_subject,
                                     static_cast<std::size_t>(36 + T) * static_cast<std::size_t>(n));
    double* a = eta + n;
    double* b = a + n;
    double* log_mass = b + n;
    double* gathered = log_mass + n;            // 8 n: the inputs and outputs of the rows of one branch
    double* work = gathered + 8 * n;            // 12 n
    double* exponent = work + 12 * n;           // 5 n: the arguments of the partials' exponentials
    double* log_density = exponent + 5 * n;     // 3 n
    double* derivative_wrt_a = log_density + 3 * n;
    double* derivative_wrt_b = derivative_wrt_a + n;
    double* derivative_wrt_eta = derivative_wrt_b + n;
    double* increment = derivative_wrt_eta + n;
    double* adjoint_of_Z = increment + n;       // T n (one class at a time)
    const std::size_t n_linear_predictors = there_are_covariates
                                             ? static_cast<std::size_t>(n_coefficient_sets) * T * n : 0;
    double* linear_predictor_of_each_set = buffer_of_at_least(buffers.values_of_each_set_test_and_subject,
                                                              2 * n_linear_predictors + 1);
    double* gradient_of_linear_predictor = linear_predictor_of_each_set + n_linear_predictors;
    int* code_of_lower_bound = buffer_of_at_least(buffers.codes_of_each_subject_and_test, 2 * nT);
    int* code_of_upper_bound = code_of_lower_bound + nT;                // [t * n + i], tests in data order
    int* rows_of_each_branch = buffer_of_at_least(buffers.rows_of_each_branch, 4 * static_cast<std::size_t>(n));

    //// ---- the nuisance values of the tile (its T x n block of the T x M matrix, contiguous): u, 1 - u, log(u),
    ////      log(1 - u), log |du/du_raw| and the derivative of the log Jacobian (Phi: -u_raw; tanh,
    ////      u = inv_logit(2 u_raw): 2 (1 - 2 u))
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < T; ++j) {
            u_raw_values[static_cast<std::size_t>(j) * n + i] = value_of(u_raw_ref.coeff(j, first_column + i));
        }
    }
    if (nuisance_transformation == 1) {
        for (std::size_t k = 0; k < nT; ++k) {
            u_values[k] = u_raw_values[k];
            one_minus_u_values[k] = -u_raw_values[k];
            log_u_values[k] = u_raw_values[k];
            log_1m_u_values[k] = -u_raw_values[k];
            log_derivative_of_u[k] = -0.5 * u_raw_values[k] * u_raw_values[k] + NEG_HALF_LOG_2PI;
            derivative_of_log_Jacobian[k] = -u_raw_values[k];
        }
        k_Phi(u_values, static_cast<int>(nT));
        k_Phi(one_minus_u_values, static_cast<int>(nT));
        k_log_Phi_in_double_precision(log_u_values, static_cast<int>(nT));
        k_log_Phi_in_double_precision(log_1m_u_values, static_cast<int>(nT));
    } else {
        for (std::size_t k = 0; k < nT; ++k) {
            u_values[k] = 2.0 * u_raw_values[k];
            one_minus_u_values[k] = -2.0 * u_raw_values[k];
            log_u_values[k] = 2.0 * u_raw_values[k];
            log_1m_u_values[k] = -2.0 * u_raw_values[k];
        }
        k_inv_logit(u_values, static_cast<int>(nT));
        k_inv_logit(one_minus_u_values, static_cast<int>(nT));
        k_log_inv_logit_in_double_precision(log_u_values, static_cast<int>(nT));
        k_log_inv_logit_in_double_precision(log_1m_u_values, static_cast<int>(nT));
        for (std::size_t k = 0; k < nT; ++k) {
            log_derivative_of_u[k] = LN2 + log_u_values[k] + log_1m_u_values[k];
            derivative_of_log_Jacobian[k] = 2.0 * (one_minus_u_values[k] - u_values[k]);
        }
    }
    double log_Jacobian = 0.0;
    for (std::size_t k = 0; k < nT; ++k) log_Jacobian += log_derivative_of_u[k];
    //// (the untruncated draws of the missing results, Z = F^{-1}(u): with u = Phi(u_raw) and Phi_type 1,
    ////  Z = u_raw itself)
    if (nuisance_transformation == 1 && Phi_type == 1) {
        for (std::size_t k = 0; k < nT; ++k) Z_of_missing_results[k] = u_raw_values[k];
    } else {
        inverse_of_F_from_the_log_scale(log_u_values, log_1m_u_values, static_cast<int>(nT), Phi_type,
                                        Z_of_missing_results);
    }
    for (std::size_t k = 0; k < nT; ++k) gradient_of_u_raw[k] = 0.0;

    //// ---- the codes of the bounds of each subject and test of the tile, from the data's rows (and the checks
    ////      of the categories); a missing result (category < 0) has both bounds infinite
    for (int i = 0; i < n; ++i) {
        const int row = first_data_row + i;
        if (static_cast<int>(category[row].size()) != T || static_cast<int>(upper_category[row].size()) != T) {
            stop_with_a_message_about_the_arguments_of_the_function_in_tiles(
                  "every row of category and upper_category must have T values");
        }
    }
    for (int t = 0; t < T; ++t) {
        const int K = n_categories_of_test[t];
        const int first = position_of_first_cutpoint_of_test[t] - 1;
        for (int i = 0; i < n; ++i) {
            const int k = category[first_data_row + i][t];
            int lower_code = CODE_OF_AN_INFINITE_BOUND;
            int upper_code = CODE_OF_AN_INFINITE_BOUND;
            if (k >= 0) {
                if (test_is_binary[t] == 1) {
                    if (k > 1) {
                        stop_with_a_message_about_the_arguments_of_the_function_in_tiles(
                              "a binary result must be 0 or 1");
                    }
                    lower_code = (k == 1) ? CODE_OF_THE_CUTPOINT_ZERO_OF_A_BINARY_TEST
                                          : CODE_OF_AN_INFINITE_BOUND;
                    upper_code = (k == 1) ? CODE_OF_AN_INFINITE_BOUND
                                          : CODE_OF_THE_CUTPOINT_ZERO_OF_A_BINARY_TEST;
                } else {
                    const int k_upper = upper_category[first_data_row + i][t];
                    if (k < 1 || k_upper < k || k_upper > K) {
                        stop_with_a_message_about_the_arguments_of_the_function_in_tiles(
                              "an ordinal category is out of range");
                    }
                    lower_code = (k == 1) ? CODE_OF_AN_INFINITE_BOUND : first + k - 2;
                    upper_code = (k_upper == K) ? CODE_OF_AN_INFINITE_BOUND : first + k_upper - 1;
                }
            }
            code_of_lower_bound[static_cast<std::size_t>(t) * n + i] = lower_code;
            code_of_upper_bound[static_cast<std::size_t>(t) * n + i] = upper_code;
        }
    }

    //// ---- the linear predictors of the covariates of each coefficient set and test, on the tile's rows of the
    ////      data (read in place, no copy)
    if (there_are_covariates) {
        for (int s = 0; s < n_coefficient_sets; ++s) {
            const int s_X = (n_covariate_sets == 1) ? 0 : s;
            for (int t = 0; t < T; ++t) {
                Eigen::Map<Eigen::Matrix<double, -1, 1>> linear_predictor(linear_predictor_of_each_set
                                                             + (static_cast<std::size_t>(s) * T + t) * n, n);
                const int P = n_covariates[s_X][t];
                if (P > 0) {
                    linear_predictor.noalias() = covariates[s_X][t].block(first_data_row, 0, n, P)
                                                 * coefficient_values[s].col(t).head(P);
                } else {
                    linear_predictor.setZero();
                }
            }
        }
        for (std::size_t k = 0; k < n_linear_predictors; ++k) gradient_of_linear_predictor[k] = 0.0;
    }

    //// ---- the forward pass of the GHK recursion of each class: Z, the log masses and the partial derivatives
    ////      of each step, in the buffers of the class
    for (int c = 0; c < C; ++c) {
        double* Z_of_class = values_of_each_class_subject_and_position + 8 * c * nT;   // [j * n + i]
        double* dZ_da_of_class = Z_of_class + nT;
        double* dZ_db_of_class = dZ_da_of_class + nT;
        double* dZ_du_raw_of_class = dZ_db_of_class + nT;
        double* dlog_mass_da_of_class = dZ_du_raw_of_class + nT;
        double* dlog_mass_db_of_class = dlog_mass_da_of_class + nT;
        double* finite_a_of_class = dlog_mass_db_of_class + nT;     // a where finite, 0 where infinite
        double* finite_b_of_class = finite_a_of_class + nT;
        double* log_mass_sum = log_mass_sum_of_each_class + static_cast<std::size_t>(c) * n;
        for (int i = 0; i < n; ++i) {
            log_mass_sum[i] = value_of(log_class_probability_ref.coeff(first_column + i, c));
            increment[i] = 0.0;
        }
        const Eigen::Map<const Eigen::Matrix<double, -1, -1>> Z_matrix(Z_of_class, n, T);

        for (int j = 0; j < T; ++j) {
            const int t = test_at_GHK_position[j] - 1;
            const double s = 1.0 / L_values[c](j, j);
            const std::size_t offset = static_cast<std::size_t>(j) * n;
            const double* cutpoints_of_test = (test_is_binary[t] == 1) ? nullptr
                                              : cutpoint_values[cutpoint_set_of_class_and_test[c][t] - 1].data();
            const double* linear_predictor = there_are_covariates
                  ? linear_predictor_of_each_set
                    + (static_cast<std::size_t>(coefficient_set_of_class_and_test[c][t] - 1) * T + t) * n
                  : nullptr;
            const int* lower_codes = code_of_lower_bound + static_cast<std::size_t>(t) * n;
            const int* upper_codes = code_of_upper_bound + static_cast<std::size_t>(t) * n;
            const double* u_of_position = u_values + offset;
            const double* one_minus_u_of_position = one_minus_u_values + offset;
            const double* log_u_of_position = log_u_values + offset;
            const double* log_1m_u_of_position = log_1m_u_values + offset;
            const double* log_derivative_of_u_of_position = log_derivative_of_u + offset;
            double* Z_of_position = Z_of_class + offset;

            //// (the linear predictor and the bounds of every subject of the tile)
            const double intercept_of_class_and_test = intercept_values(c, t);
            for (int i = 0; i < n; ++i) {
                eta[i] = intercept_of_class_and_test + increment[i]
                         + (there_are_covariates ? linear_predictor[i] : 0.0);
            }
            for (int i = 0; i < n; ++i) {
                const int lower_code = lower_codes[i];
                const int upper_code = upper_codes[i];
                const double lower_cutpoint = (lower_code >= 0) ? cutpoints_of_test[lower_code] : 0.0;
                const double upper_cutpoint = (upper_code >= 0) ? cutpoints_of_test[upper_code] : 0.0;
                a[i] = (lower_code == CODE_OF_AN_INFINITE_BOUND) ? -infinity : (lower_cutpoint - eta[i]) * s;
                b[i] = (upper_code == CODE_OF_AN_INFINITE_BOUND) ? infinity : (upper_cutpoint - eta[i]) * s;
            }

            //// (the rows of each branch: narrow intervals first, then the lower and the upper tails, as the
            ////  function above; the missing results are given their untruncated draws)
            int n_rows_of_branch[4] = { 0, 0, 0, 0 };
            int* rows_of_branch[4] = { rows_of_each_branch, rows_of_each_branch + n, rows_of_each_branch + 2 * n,
                                       rows_of_each_branch + 3 * n };
            for (int i = 0; i < n; ++i) {
                if (category[first_data_row + i][t] < 0) {
                    Z_of_position[i] = Z_of_missing_results[offset + i];
                    log_mass[i] = 0.0;
                    continue;
                }
                int branch = ROWS_DIRECTLY;
                const bool both_bounds_are_finite = (a[i] != -infinity) && (b[i] != infinity);
                if (both_bounds_are_finite && (b[i] - a[i]) * (1.0 + std::fabs(0.5 * (a[i] + b[i])))
                                              < LARGEST_WIDTH_OF_A_NARROW_INTERVAL) {

                    branch = ROWS_OF_A_NARROW_INTERVAL;

                } else if (b[i] < underflow_threshold) {
                    branch = ROWS_IN_THE_LOWER_TAIL;
                } else if (a[i] > overflow_threshold) {
                    branch = ROWS_IN_THE_UPPER_TAIL;
                }
                rows_of_branch[branch][n_rows_of_branch[branch]] = i;
                n_rows_of_branch[branch] += 1;
            }
            for (int branch = 0; branch < 4; ++branch) {
                const int n_rows = n_rows_of_branch[branch];
                if (n_rows == 0) continue;
                const int* rows = rows_of_branch[branch];
                double* gathered_a = gathered;
                double* gathered_b = gathered + n_rows;
                double* gathered_u = gathered + 2 * n_rows;
                double* gathered_one_minus_u = gathered + 3 * n_rows;
                double* gathered_log_u = gathered + 4 * n_rows;
                double* gathered_log_1m_u = gathered + 5 * n_rows;
                double* gathered_Z = gathered + 6 * n_rows;
                double* gathered_log_mass = gathered + 7 * n_rows;
                for (int r = 0; r < n_rows; ++r) {
                    const int i = rows[r];
                    gathered_a[r] = a[i];
                    gathered_b[r] = b[i];
                    gathered_u[r] = u_of_position[i];
                    gathered_one_minus_u[r] = one_minus_u_of_position[i];
                    gathered_log_u[r] = log_u_of_position[i];
                    gathered_log_1m_u[r] = log_1m_u_of_position[i];
                }
                if (branch == ROWS_DIRECTLY) {
                    if (Phi_type == 1) {
                        GHK_rows_directly_with_Phi(n_rows, gathered_a, gathered_b, gathered_u,
                                                   gathered_one_minus_u, gathered_Z, gathered_log_mass, work);
                    } else {
                        GHK_rows_directly_with_Phi_approx(n_rows, gathered_a, gathered_b, gathered_log_u,
                                                          gathered_log_1m_u, gathered_Z, gathered_log_mass, work);
                    }
                } else if (branch == ROWS_IN_THE_LOWER_TAIL) {
                    GHK_rows_in_the_lower_tail(n_rows, Phi_type, gathered_a, gathered_b, gathered_log_u,
                                               gathered_log_1m_u, gathered_Z, gathered_log_mass, work);
                } else if (branch == ROWS_IN_THE_UPPER_TAIL) {
                    GHK_rows_in_the_upper_tail(n_rows, Phi_type, gathered_a, gathered_b, gathered_log_u,
                                               gathered_log_1m_u, gathered_Z, gathered_log_mass, work);
                } else {
                    GHK_rows_of_a_narrow_interval(n_rows, Phi_type, gathered_a, gathered_b, gathered_u,
                                                  gathered_one_minus_u, gathered_Z, gathered_log_mass, work);
                }
                for (int r = 0; r < n_rows; ++r) {
                    Z_of_position[rows[r]] = gathered_Z[r];
                    log_mass[rows[r]] = gathered_log_mass[r];
                }
            }
            for (int i = 0; i < n; ++i) log_mass_sum[i] += log_mass[i];

            //// (the partial derivatives of the step, every subject of the tile at once; an infinite bound takes
            ////  exponent 0 and partial 0)
            if constexpr (gradient_is_needed) {
                double* log_density_of_a = log_density;
                double* log_density_of_b = log_density + n;
                double* log_density_of_Z = log_density + 2 * n;
                for (int i = 0; i < n; ++i) {
                    log_density_of_a[i] = (a[i] == -infinity) ? 0.0 : a[i];
                    log_density_of_b[i] = (b[i] == infinity) ? 0.0 : b[i];
                    log_density_of_Z[i] = Z_of_position[i];
                }
                k_log_density_of_F(log_density_of_a, n, Phi_type, work);
                k_log_density_of_F(log_density_of_b, n, Phi_type, work);
                k_log_density_of_F(log_density_of_Z, n, Phi_type, work);
                double* exponent_dZ_da = exponent;
                double* exponent_dZ_db = exponent + n;
                double* exponent_dZ_du_raw = exponent + 2 * n;
                double* exponent_dlog_mass_da = exponent + 3 * n;
                double* exponent_dlog_mass_db = exponent + 4 * n;
                for (int i = 0; i < n; ++i) {
                    const bool a_is_finite = (a[i] != -infinity);
                    const bool b_is_finite = (b[i] != infinity);
                    exponent_dZ_da[i] = a_is_finite ? log_1m_u_of_position[i] + log_density_of_a[i]
                                                      - log_density_of_Z[i] : 0.0;
                    exponent_dZ_db[i] = b_is_finite ? log_u_of_position[i] + log_density_of_b[i]
                                                      - log_density_of_Z[i] : 0.0;
                    exponent_dZ_du_raw[i] = log_mass[i] + log_derivative_of_u_of_position[i]
                                            - log_density_of_Z[i];
                    exponent_dlog_mass_da[i] = a_is_finite ? log_density_of_a[i] - log_mass[i] : 0.0;
                    exponent_dlog_mass_db[i] = b_is_finite ? log_density_of_b[i] - log_mass[i] : 0.0;
                }
                k_exp(exponent, 5 * n);
                for (int i = 0; i < n; ++i) {
                    const bool a_is_finite = (a[i] != -infinity);
                    const bool b_is_finite = (b[i] != infinity);
                    dZ_da_of_class[offset + i] = a_is_finite ? exponent_dZ_da[i] : 0.0;
                    dZ_db_of_class[offset + i] = b_is_finite ? exponent_dZ_db[i] : 0.0;
                    dZ_du_raw_of_class[offset + i] = exponent_dZ_du_raw[i];
                    dlog_mass_da_of_class[offset + i] = a_is_finite ? -exponent_dlog_mass_da[i] : 0.0;
                    dlog_mass_db_of_class[offset + i] = b_is_finite ? exponent_dlog_mass_db[i] : 0.0;
                    finite_a_of_class[offset + i] = a_is_finite ? a[i] : 0.0;
                    finite_b_of_class[offset + i] = b_is_finite ? b[i] : 0.0;
                }
            }

            //// (the increment of the next GHK position: sum_{k <= j} L[j + 1, k] Z[k])
            if (j + 1 < T) {
                Eigen::Map<Eigen::Matrix<double, -1, 1>> increment_vector(increment, n);
                increment_vector.noalias() = Z_matrix.leftCols(j + 1)
                                             * L_values[c].row(j + 1).head(j + 1).transpose();
            }
        }
    }

    //// ---- the mixture over the classes: log sum_c exp(log_class_probability + the class's log masses) of each
    ////      subject of the tile, and the weight of each class (the derivative of the value with respect to its
    ////      terms)
    double value = log_Jacobian;
    {
        double* largest_term = derivative_wrt_a;
        double* sum_of_exponentials = derivative_wrt_b;
        for (int i = 0; i < n; ++i) largest_term[i] = log_mass_sum_of_each_class[i];
        for (int c = 1; c < C; ++c) {
            const double* log_mass_sum = log_mass_sum_of_each_class + static_cast<std::size_t>(c) * n;
            for (int i = 0; i < n; ++i) largest_term[i] = std::max(largest_term[i], log_mass_sum[i]);
        }
        for (int c = 0; c < C; ++c) {
            const double* log_mass_sum = log_mass_sum_of_each_class + static_cast<std::size_t>(c) * n;
            double* weight = weight_of_each_class + static_cast<std::size_t>(c) * n;
            for (int i = 0; i < n; ++i) weight[i] = log_mass_sum[i] - largest_term[i];
        }
        k_exp(weight_of_each_class, static_cast<int>(nC));
        for (int i = 0; i < n; ++i) sum_of_exponentials[i] = 0.0;
        for (int c = 0; c < C; ++c) {
            const double* weight = weight_of_each_class + static_cast<std::size_t>(c) * n;
            for (int i = 0; i < n; ++i) sum_of_exponentials[i] += weight[i];
        }
        for (int c = 0; c < C; ++c) {
            double* weight = weight_of_each_class + static_cast<std::size_t>(c) * n;
            for (int i = 0; i < n; ++i) weight[i] /= sum_of_exponentials[i];
        }
        k_log(sum_of_exponentials, n);
        for (int i = 0; i < n; ++i) value += largest_term[i] + sum_of_exponentials[i];
    }

    if constexpr (!gradient_is_needed) {
        return value;
    } else {

        //// ---- the reverse sweep of the GHK recursion of each class, seeded by the class weights: the gradient
        ////      with respect to the tile's nuisance values and linear predictors, and the tile's part of the
        ////      gradients with respect to the intercepts, the cutpoints and L_Omega
        for (int c = 0; c < C; ++c) {
            const double* Z_of_class = values_of_each_class_subject_and_position + 8 * c * nT;
            const double* dZ_da_of_class = Z_of_class + nT;
            const double* dZ_db_of_class = dZ_da_of_class + nT;
            const double* dZ_du_raw_of_class = dZ_db_of_class + nT;
            const double* dlog_mass_da_of_class = dZ_du_raw_of_class + nT;
            const double* dlog_mass_db_of_class = dlog_mass_da_of_class + nT;
            const double* finite_a_of_class = dlog_mass_db_of_class + nT;
            const double* finite_b_of_class = finite_a_of_class + nT;
            const double* weight = weight_of_each_class + static_cast<std::size_t>(c) * n;
            const Eigen::Map<const Eigen::Matrix<double, -1, -1>> Z_matrix(Z_of_class, n, T);
            Eigen::Map<Eigen::Matrix<double, -1, -1>> adjoint_of_Z_matrix(adjoint_of_Z, n, T);
            adjoint_of_Z_matrix.setZero();
            for (int j = T - 1; j >= 0; --j) {
                const int t = test_at_GHK_position[j] - 1;
                const double s = 1.0 / L_values[c](j, j);
                const std::size_t offset = static_cast<std::size_t>(j) * n;
                const double* adjoint_of_Z_of_position = adjoint_of_Z + offset;
                double sum_of_bounds_times_their_derivatives = 0.0;
                double sum_of_derivatives_wrt_eta = 0.0;
                for (int i = 0; i < n; ++i) {
                    derivative_wrt_a[i] = adjoint_of_Z_of_position[i] * dZ_da_of_class[offset + i]
                                          + weight[i] * dlog_mass_da_of_class[offset + i];
                    derivative_wrt_b[i] = adjoint_of_Z_of_position[i] * dZ_db_of_class[offset + i]
                                          + weight[i] * dlog_mass_db_of_class[offset + i];
                    gradient_of_u_raw[offset + i] += adjoint_of_Z_of_position[i] * dZ_du_raw_of_class[offset + i];
                    derivative_wrt_eta[i] = -s * (derivative_wrt_a[i] + derivative_wrt_b[i]);
                    sum_of_bounds_times_their_derivatives +=
                          finite_a_of_class[offset + i] * derivative_wrt_a[i]
                          + finite_b_of_class[offset + i] * derivative_wrt_b[i];
                    sum_of_derivatives_wrt_eta += derivative_wrt_eta[i];
                }
                //// (a = (cutpoint - eta - inc) s: d/dcutpoint = s; d/dL[j, j] = (a / s) (-s^2) = -s a)
                if (test_is_binary[t] != 1) {
                    double* gradient_of_set =
                          gradient_of_cutpoints[cutpoint_set_of_class_and_test[c][t] - 1].data();
                    const int* lower_codes = code_of_lower_bound + static_cast<std::size_t>(t) * n;
                    const int* upper_codes = code_of_upper_bound + static_cast<std::size_t>(t) * n;
                    for (int i = 0; i < n; ++i) {
                        if (lower_codes[i] >= 0) gradient_of_set[lower_codes[i]] += s * derivative_wrt_a[i];
                        if (upper_codes[i] >= 0) gradient_of_set[upper_codes[i]] += s * derivative_wrt_b[i];
                    }
                }
                gradient_of_L[c](j, j) += -s * sum_of_bounds_times_their_derivatives;
                gradient_of_intercept(c, t) += sum_of_derivatives_wrt_eta;
                if (there_are_covariates) {
                    double* gradient_of_linear_predictor_of_set = gradient_of_linear_predictor
                          + (static_cast<std::size_t>(coefficient_set_of_class_and_test[c][t] - 1) * T + t) * n;
                    for (int i = 0; i < n; ++i) gradient_of_linear_predictor_of_set[i] += derivative_wrt_eta[i];
                }
                //// (inc = sum_{k < j} L[j, k] Z[k])
                if (j > 0) {
                    const Eigen::Map<const Eigen::Matrix<double, -1,
                                                         1>> derivative_wrt_eta_vector(derivative_wrt_eta, n);
                    adjoint_of_Z_matrix.leftCols(j).noalias() += derivative_wrt_eta_vector
                                                                 * L_values[c].row(j).head(j);
                    gradient_of_L[c].row(j).head(j).noalias() += derivative_wrt_eta_vector.transpose()
                                                                 * Z_matrix.leftCols(j);
                }
            }
        }

        //// ---- the tile's partial derivatives of its nuisance values, written as its contiguous T x n block of
        ////      the T x M partials
        if (gradient_of_u_raw_of_tile != nullptr) {
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < T; ++j) {
                    const std::size_t k = static_cast<std::size_t>(j) * n + i;
                    gradient_of_u_raw_of_tile[static_cast<std::size_t>(i) * T + j] =
                          gradient_of_u_raw[k] + derivative_of_log_Jacobian[k];
                }
            }
        }
        //// ---- the tile's part of the gradients of the coefficients (its rows of the covariates, read in place)
        if (there_are_covariates) {
            for (int s = 0; s < n_coefficient_sets; ++s) {
                const int s_X = (n_covariate_sets == 1) ? 0 : s;
                for (int t = 0; t < T; ++t) {
                    const int P = n_covariates[s_X][t];
                    if (P == 0) continue;
                    const Eigen::Map<const Eigen::Matrix<double, -1, 1>> gradient_of_linear_predictor_of_set(
                          gradient_of_linear_predictor + (static_cast<std::size_t>(s) * T + t) * n, n);
                    gradient_of_coefficients[s].col(t).head(P).noalias() +=
                          covariates[s_X][t].block(first_data_row, 0, n, P).transpose()
                          * gradient_of_linear_predictor_of_set;
                }
            }
        }
        //// ---- the tile's partial derivatives of its log class probabilities (the weights of the classes)
        if (partials_of_log_class_probability != nullptr) {
            for (int c = 0; c < C; ++c) {
                const double* weight = weight_of_each_class + static_cast<std::size_t>(c) * n;
                double* partials_of_class = partials_of_log_class_probability
                                            + static_cast<std::size_t>(c) * M + first_column;
                for (int i = 0; i < n; ++i) partials_of_class[i] = weight[i];
            }
        }
        return value;
    }

}

}  // namespace nico::math

namespace nico::math {

template <typename T_u_raw, typename T_intercept, typename T_coefficients, typename T_cutpoints,
          typename T_L_Omega, typename T_log_class_probability>
inline stan::return_type_t<T_u_raw, T_intercept, T_coefficients, T_cutpoints, T_L_Omega, T_log_class_probability>
nico_finite_mixture_of_multivariate_ordered_probit_GHK_log_likelihood_in_tiles_of_subjects(
        const T_u_raw& u_raw_of_each_test_and_subject,
        const int first_subject,
        const int nuisance_transformation,
        const T_intercept& intercept,
        const T_coefficients& coefficients,
        const std::vector<std::vector<int>>& coefficient_set_of_class_and_test,
        const std::vector<std::vector<Eigen::Matrix<double, -1, -1>>>& covariates,
        const std::vector<std::vector<int>>& n_covariates,
        const T_cutpoints& cutpoints,
        const std::vector<std::vector<int>>& cutpoint_set_of_class_and_test,
        const std::vector<int>& position_of_first_cutpoint_of_test,
        const T_L_Omega& L_Omega,
        const T_log_class_probability& log_class_probability,
        const std::vector<std::vector<int>>& category,
        const std::vector<std::vector<int>>& upper_category,
        const std::vector<int>& test_is_binary,
        const std::vector<int>& n_categories_of_test,
        const std::vector<int>& test_at_GHK_position,
        const int Phi_type,
        const double overflow_threshold,
        const double underflow_threshold,
        std::ostream* /*pstream__*/) {

    using namespace nico::math;
    using stan::math::value_of;
    constexpr bool gradient_is_needed = !stan::is_constant_all<T_u_raw, T_intercept, T_coefficients, T_cutpoints,
                                                               T_L_Omega, T_log_class_probability>::value;

    const auto& u_raw_ref = stan::math::to_ref(u_raw_of_each_test_and_subject);
    const auto& intercept_ref = stan::math::to_ref(intercept);
    const auto& log_class_probability_ref = stan::math::to_ref(log_class_probability);

    //// ---- the sizes, and the checks of the arguments (each O(1) per test, class or set, except the rows of the
    ////      data, checked tile by tile where their codes are made):
    const int M = static_cast<int>(u_raw_ref.cols());
    const int C = static_cast<int>(L_Omega.size());
    const int T = (C > 0) ? static_cast<int>(L_Omega[0].rows()) : 0;
    const int first_data_row = first_subject - 1;
    const int n_coefficient_sets = static_cast<int>(coefficients.size());
    const int n_cutpoint_sets = static_cast<int>(cutpoints.size());
    const int n_covariate_sets = static_cast<int>(covariates.size());
    if (C < 1) {
        stop_with_a_message_about_the_arguments_of_the_function_in_tiles("L_Omega must have one class");
    }
    if (Phi_type != 1 && Phi_type != 2) {
        stop_with_a_message_about_the_arguments_of_the_function_in_tiles("Phi_type must be 1 or 2");
    }
    if (nuisance_transformation != 1 && nuisance_transformation != 2) {
        stop_with_a_message_about_the_arguments_of_the_function_in_tiles(
              "nuisance_transformation must be 1 (Phi) or 2 (tanh)");
    }
    if (static_cast<int>(u_raw_ref.rows()) != T) {
        stop_with_a_message_about_the_arguments_of_the_function_in_tiles(
              "u_raw_of_each_test_and_subject must be T x M");
    }
    if (intercept_ref.rows() != C || intercept_ref.cols() != T) {
        stop_with_a_message_about_the_arguments_of_the_function_in_tiles("intercept must be C x T");
    }
    if (log_class_probability_ref.rows() != M || log_class_probability_ref.cols() != C) {
        stop_with_a_message_about_the_arguments_of_the_function_in_tiles("log_class_probability must be M x C");
    }
    if (first_subject < 1 || static_cast<int>(category.size()) < first_data_row + M
        || static_cast<int>(upper_category.size()) < first_data_row + M) {

        stop_with_a_message_about_the_arguments_of_the_function_in_tiles(
              "category and upper_category must have the rows first_subject, ..., first_subject + M - 1");

    }
    if (static_cast<int>(test_is_binary.size()) != T || static_cast<int>(n_categories_of_test.size()) != T
        || static_cast<int>(test_at_GHK_position.size()) != T
        || static_cast<int>(position_of_first_cutpoint_of_test.size()) != T) {

        stop_with_a_message_about_the_arguments_of_the_function_in_tiles(
              "the arrays of the tests must have T elements");

    }
    if (static_cast<int>(coefficient_set_of_class_and_test.size()) != C
        || static_cast<int>(cutpoint_set_of_class_and_test.size()) != C) {

        stop_with_a_message_about_the_arguments_of_the_function_in_tiles(
              "the sets of each class and test must be C x T");

    }
    for (int c = 0; c < C; ++c) {
        if (L_Omega[c].rows() != T || L_Omega[c].cols() != T) {
            stop_with_a_message_about_the_arguments_of_the_function_in_tiles("every L_Omega must be T x T");
        }
        if (static_cast<int>(coefficient_set_of_class_and_test[c].size()) != T
            || static_cast<int>(cutpoint_set_of_class_and_test[c].size()) != T) {

            stop_with_a_message_about_the_arguments_of_the_function_in_tiles(
                  "the sets of each class and test must be C x T");

        }
    }
    //// (covariates: none when there are no covariate sets or no covariates in any of them)
    int largest_number_of_covariates = 0;
    if (n_covariate_sets > 0) {
        if (n_covariate_sets != 1 && n_covariate_sets != n_coefficient_sets) {
            stop_with_a_message_about_the_arguments_of_the_function_in_tiles(
                  "covariates must have 1 or S_B sets");
        }
        if (static_cast<int>(n_covariates.size()) != n_covariate_sets) {
            stop_with_a_message_about_the_arguments_of_the_function_in_tiles(
                  "n_covariates must have the sets of covariates");
        }
        for (int s = 0; s < n_covariate_sets; ++s) {
            if (static_cast<int>(covariates[s].size()) != T || static_cast<int>(n_covariates[s].size()) != T) {
                stop_with_a_message_about_the_arguments_of_the_function_in_tiles(
                      "every set of covariates must have T matrices");
            }
            for (int t = 0; t < T; ++t) {
                const int P = n_covariates[s][t];
                if (P < 0 || P > covariates[s][t].cols()
                    || (P > 0 && covariates[s][t].rows() < first_data_row + M)) {

                    stop_with_a_message_about_the_arguments_of_the_function_in_tiles(
                          "covariates must have the rows first_subject, ..., first_subject + M - 1 and at least "
                          "n_covariates columns");

                }
                largest_number_of_covariates = std::max(largest_number_of_covariates, P);
            }
        }
    }
    const bool there_are_covariates = (largest_number_of_covariates > 0);
    if (there_are_covariates) {
        for (int s = 0; s < n_coefficient_sets; ++s) {
            if (coefficients[s].cols() != T || coefficients[s].rows() < largest_number_of_covariates) {
                stop_with_a_message_about_the_arguments_of_the_function_in_tiles(
                      "every coefficient set must be P_max x T");
            }
        }
    }
    for (int c = 0; c < C; ++c) {
        for (int t = 0; t < T; ++t) {
            if (there_are_covariates && (coefficient_set_of_class_and_test[c][t] < 1
                                         || coefficient_set_of_class_and_test[c][t] > n_coefficient_sets)) {

                stop_with_a_message_about_the_arguments_of_the_function_in_tiles(
                      "a coefficient set is out of range");

            }
            if (test_is_binary[t] != 1 && (cutpoint_set_of_class_and_test[c][t] < 1
                                           || cutpoint_set_of_class_and_test[c][t] > n_cutpoint_sets)) {

                stop_with_a_message_about_the_arguments_of_the_function_in_tiles(
                      "a cutpoint set is out of range");

            }
        }
    }
    std::vector<int> position_of_test_in_GHK_order(T, -1);
    for (int j = 0; j < T; ++j) {
        const int t = test_at_GHK_position[j] - 1;
        if (t < 0 || t >= T || position_of_test_in_GHK_order[t] != -1) {
            stop_with_a_message_about_the_arguments_of_the_function_in_tiles(
                  "test_at_GHK_position must be a permutation of 1:T");
        }
        position_of_test_in_GHK_order[t] = j;
    }
    for (int t = 0; t < T; ++t) {
        if (test_is_binary[t] == 1) continue;
        const int first = position_of_first_cutpoint_of_test[t] - 1;
        for (int s = 0; s < n_cutpoint_sets; ++s) {
            if (first < 0 || first + n_categories_of_test[t] - 1 > cutpoints[s].size()) {
                stop_with_a_message_about_the_arguments_of_the_function_in_tiles(
                      "the cutpoints of an ordinal test are out of range");
            }
        }
    }

    //// ---- the values of the parameters, once for every tile:
    const Eigen::Matrix<double, -1, -1> intercept_values = value_of(intercept_ref);
    std::vector<Eigen::Matrix<double, -1, -1>> L_values(C);
    for (int c = 0; c < C; ++c) L_values[c] = value_of(L_Omega[c]);
    std::vector<Eigen::Matrix<double, -1, 1>> cutpoint_values(n_cutpoint_sets);
    for (int s = 0; s < n_cutpoint_sets; ++s) cutpoint_values[s] = value_of(cutpoints[s]);
    std::vector<Eigen::Matrix<double, -1, -1>> coefficient_values(there_are_covariates ? n_coefficient_sets : 0);
    for (int s = 0; s < static_cast<int>(coefficient_values.size()); ++s) {
        coefficient_values[s] = value_of(coefficients[s]);
    }

    //// ---- the sums of the gradients over the tiles:
    Eigen::Matrix<double, -1, -1> gradient_of_intercept = Eigen::Matrix<double, -1, -1>::Zero(C, T);
    std::vector<Eigen::Matrix<double, -1, -1>> gradient_of_L(C, Eigen::Matrix<double, -1, -1>::Zero(T, T));
    std::vector<Eigen::Matrix<double, -1, 1>> gradient_of_cutpoints(n_cutpoint_sets);
    for (int s = 0; s < n_cutpoint_sets; ++s) {
        gradient_of_cutpoints[s] = Eigen::Matrix<double, -1, 1>::Zero(cutpoints[s].size());
    }
    std::vector<Eigen::Matrix<double, -1, -1>> gradient_of_coefficients(n_coefficient_sets);
    for (int s = 0; s < n_coefficient_sets; ++s) {
        gradient_of_coefficients[s] = Eigen::Matrix<double, -1, -1>::Zero(coefficients[s].rows(),
                                                                          coefficients[s].cols());
    }

    if constexpr (!gradient_is_needed) {
        double value = 0.0;
        for (int first_column = 0; first_column < M; first_column += LARGEST_NUMBER_OF_SUBJECTS_IN_A_TILE) {
            const int n = std::min(LARGEST_NUMBER_OF_SUBJECTS_IN_A_TILE, M - first_column);
            value += GHK_mixture_log_likelihood_of_one_tile_of_subjects<false>(
                           n, first_column, first_data_row + first_column, M, u_raw_ref, nuisance_transformation,
                           intercept_values, coefficient_values, coefficient_set_of_class_and_test, covariates,
                           n_covariates, there_are_covariates, cutpoint_values, cutpoint_set_of_class_and_test,
                           position_of_first_cutpoint_of_test, L_values, log_class_probability_ref, category,
                           upper_category, test_is_binary, n_categories_of_test, test_at_GHK_position, Phi_type,
                           overflow_threshold, underflow_threshold, nullptr, nullptr, gradient_of_intercept,
                           gradient_of_coefficients, gradient_of_cutpoints, gradient_of_L);
        }
        return value;
    } else {

        //// ---- the node, and its partials of the nuisance values (T x M) and of the log class probabilities
        ////      (M x C), written tile by tile
        auto partials = stan::math::make_partials_propagator(u_raw_ref, intercept_ref, coefficients, cutpoints,
                                                             L_Omega, log_class_probability_ref);
        double* partials_of_u_raw = nullptr;
        if constexpr (!stan::is_constant_all<T_u_raw>::value) {
            partials_of_u_raw = stan::math::partials<0>(partials).data();
        }
        double* partials_of_log_class_probability = nullptr;
        if constexpr (!stan::is_constant_all<T_log_class_probability>::value) {
            partials_of_log_class_probability = stan::math::partials<5>(partials).data();
        }
        double value = 0.0;
        for (int first_column = 0; first_column < M; first_column += LARGEST_NUMBER_OF_SUBJECTS_IN_A_TILE) {
            const int n = std::min(LARGEST_NUMBER_OF_SUBJECTS_IN_A_TILE, M - first_column);
            double* partials_of_u_raw_of_tile = (partials_of_u_raw == nullptr) ? nullptr
                  : partials_of_u_raw + static_cast<std::size_t>(first_column) * T;
            value += GHK_mixture_log_likelihood_of_one_tile_of_subjects<true>(
                           n, first_column, first_data_row + first_column, M, u_raw_ref, nuisance_transformation,
                           intercept_values, coefficient_values, coefficient_set_of_class_and_test, covariates,
                           n_covariates, there_are_covariates, cutpoint_values, cutpoint_set_of_class_and_test,
                           position_of_first_cutpoint_of_test, L_values, log_class_probability_ref, category,
                           upper_category, test_is_binary, n_categories_of_test, test_at_GHK_position, Phi_type,
                           overflow_threshold, underflow_threshold, partials_of_u_raw_of_tile,
                           partials_of_log_class_probability, gradient_of_intercept, gradient_of_coefficients,
                           gradient_of_cutpoints, gradient_of_L);
        }
        if constexpr (!stan::is_constant_all<T_intercept>::value) {
            stan::math::partials<1>(partials) = gradient_of_intercept;
        }
        if constexpr (!stan::is_constant_all<T_coefficients>::value) {
            for (int s = 0; s < n_coefficient_sets; ++s) {
                stan::math::partials_vec<2>(partials)[s] = gradient_of_coefficients[s];
            }
        }
        if constexpr (!stan::is_constant_all<T_cutpoints>::value) {
            for (int s = 0; s < n_cutpoint_sets; ++s) {
                stan::math::partials_vec<3>(partials)[s] = gradient_of_cutpoints[s];
            }
        }
        if constexpr (!stan::is_constant_all<T_L_Omega>::value) {
            for (int c = 0; c < C; ++c) stan::math::partials_vec<4>(partials)[c] = gradient_of_L[c];
        }
        return partials.build(value);
    }

}

}  // namespace nico::math

using nico::math::nico_finite_mixture_of_multivariate_ordered_probit_GHK_log_likelihood_in_tiles_of_subjects;























