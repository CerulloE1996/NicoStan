#pragma once
//
// -| --------- BayesMVP proportional-hazards log likelihood with a piecewise linear baseline hazard on a grid ---
//
//     nico_proportional_hazards_log_likelihood_with_a_piecewise_linear_baseline_hazard_on_a_grid(
//           event, X, beta, cell, time_into_cell, cell_widths, h)
//           = sum_i [ event_i (X_i beta + log h_0(t_i)) - exp(X_i beta) H_0(t_i) ]
//
// the log likelihood of right-censored survival times under proportional hazards, with the baseline hazard h_0
// linear between the M + 1 points of a grid (its values h at the points; the cells of widths cell_widths, any
// spacing), so that, for a time t in cell c (1-based) at time_into_cell tau from the cell's first point, with
// f = tau / width_c,
//     h_0(t) = (1 - f) h_c + f h_(c+1),
//     H_0(t) = H_c + (tau - tau^2 / (2 width_c)) h_c + (tau^2 / (2 width_c)) h_(c+1),
// where H_c = sum_(k < c) 0.5 width_k (h_k + h_(k+1)) is the cumulative baseline hazard at the grid point c (the
// exact integral of the piecewise linear h_0). A general function: any proportional-hazards model whose baseline
// hazard is given at the points of a grid, e.g. a latent diffusion of the log baseline hazard, a piecewise linear
// baseline on a grid, the baseline of a spline evaluated at grid points, and the time-to-event part of a joint
// model.
//
// The subjects must be ordered by their cell (cell nondecreasing; e.g. ordered by time once, in the model's
// transformed data: the log likelihood is a sum, so the order changes nothing else), so that the subjects of each
// cell are one contiguous run of every slice of the subjects (each chunk of reduce_sum): within a run the grid
// values are the same numbers, used in whole-run vector work, and the gradient into the grid is one sum per run,
// so there is no gather and no scatter of one subject at a time.
//
// ONE autodiff node for the whole log likelihood (Stan's version: the nodes of X beta, of exp, of the
// interpolations of the two grids, of log and of the sums), with BayesMVP's exp and log kernels on whole
// vectors, X beta and X' r as Eigen products, every other step whole-run vector work (one pass per cell present),
// and the cumulative sums over the grid (H, and its adjoint) as BayesMVP's SIMD scans
// (stan_external_simd_affine_recurrence_scans.hpp, with the coefficient 1). The partial derivatives are made in
// the forward pass (the result is one number). With e_i = exp(X_i beta), S_c the sum of e_i over the subjects of
// cell c (the adjoint of H_c is -S_c), and the adjoint g of the result:
//     adjoint of beta   += g X' (event - e .* H_0(t))
//     adjoint of h_c    += g [ sum over the subjects of cell c of
//                                  ( -e_i (tau_i - tau_i^2 / (2 width_c)) + event_i (1 - f_i) / h_0(t_i) )
//                              + sum over the subjects of cell c - 1 of
//                                  ( -e_i tau_i^2 / (2 width_(c-1)) + event_i f_i / h_0(t_i) )
//                              - 0.5 width_c sum_(k > c) S_k - 0.5 width_(c-1) sum_(k > c - 1) S_k ]
//
// Declared in the Stan functions block (no body; stanc --allow-undefined):
//
//     real nico_proportional_hazards_log_likelihood_with_a_piecewise_linear_baseline_hazard_on_a_grid(
//           data array[] int event, data matrix X, vector beta, data array[] int cell,
//           data vector time_into_cell, data vector cell_widths, vector h);
//
// Checks: the sizes (X: one row per subject, as many columns as beta; cell and time_into_cell one per subject;
// h one more than cell_widths), event 0 or 1, every cell in 1, ..., M and nondecreasing, time_into_cell in
// [0, the width of its cell], the widths and every h_c positive and finite.
//
// Included at the end of stan_external_functions.hpp.
//
// -----------------------------------------------------------------------------------------------------------

#include "stan_external_functions.hpp"
#include "stan_external_simd_affine_recurrence_scans.hpp"

#include <stan/math/rev.hpp>
#include <stan/math/prim.hpp>

#include <Eigen/Dense>
#include <algorithm>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace nico::math {
namespace proportional_hazards_on_a_grid {

//// ---- the inclusive sums of x[0], ..., x[i] (prefix = true) or of x[i], ..., x[n - 1] (prefix = false), with
////      BayesMVP's SIMD scan of the recurrence y[i] = y[i -/+ 1] + x[i] (the coefficient 1)
inline Eigen::Matrix<double, -1, 1> cumulative_sums_in_SIMD(const Eigen::Matrix<double, -1, 1>& x,
                                                            const bool prefix) {

    const Eigen::Index n = x.size();
    const Eigen::Index n_padded = padded_to_lanes(n);
    std::vector<double> padded_x(static_cast<std::size_t>(n_padded), 0.0);
    std::vector<double> sums(static_cast<std::size_t>(n_padded), 0.0);
    std::copy(x.data(), x.data() + n, padded_x.begin());
    const affine_scan_coefficients_t coefficients_of_the_sums = affine_scan_coefficients(1.0);
    if (prefix) {
        forward_affine_scan(padded_x.data(), n_padded, 0.0, sums.data(), coefficients_of_the_sums);
    } else {
        backward_affine_scan(padded_x.data(), n_padded, 0.0, sums.data(), coefficients_of_the_sums);
    }
    return Eigen::Map<const Eigen::Matrix<double, -1, 1>>(sums.data(), n);

}

//// ---- the end (one past the last subject) of the run of subjects of the cell of subject first
inline int end_of_the_run_of_one_cell(const std::vector<int>& cell, const int first) {

    return static_cast<int>(std::upper_bound(cell.begin() + first, cell.end(), cell[first]) - cell.begin());

}

}  // namespace proportional_hazards_on_a_grid
}  // namespace nico::math


// ---- the Stan-facing function (global scope; stanc calls it with pstream__ last)
namespace nico::math {

template <typename T_X, typename T_beta, typename T_tau, typename T_widths, typename T_h,
          stan::require_eigen_matrix_dynamic_t<T_X>* = nullptr,
          stan::require_eigen_col_vector_t<T_beta>* = nullptr,
          stan::require_eigen_col_vector_t<T_tau>* = nullptr,
          stan::require_eigen_col_vector_t<T_widths>* = nullptr,
          stan::require_eigen_col_vector_t<T_h>* = nullptr>
inline stan::return_type_t<T_beta, T_h>
nico_proportional_hazards_log_likelihood_with_a_piecewise_linear_baseline_hazard_on_a_grid(
        const std::vector<int>& event, const T_X& X, const T_beta& beta, const std::vector<int>& cell,
        const T_tau& time_into_cell, const T_widths& cell_widths, const T_h& h,
        std::ostream* /*pstream__*/ = nullptr) {

    namespace on_a_grid = nico::math::proportional_hazards_on_a_grid;
    static_assert(std::is_arithmetic<stan::value_type_t<T_X>>::value, "X must be data");
    static_assert(std::is_arithmetic<stan::value_type_t<T_tau>>::value, "time_into_cell must be data");
    static_assert(std::is_arithmetic<stan::value_type_t<T_widths>>::value, "cell_widths must be data");
    static const char* function_name
        = "nico_proportional_hazards_log_likelihood_with_a_piecewise_linear_baseline_hazard_on_a_grid";
    const auto& beta_ref = stan::math::to_ref(beta);
    const auto& h_ref = stan::math::to_ref(h);
    const int n = static_cast<int>(event.size());
    const int M = static_cast<int>(cell_widths.size());
    stan::math::check_size_match(function_name, "rows of X", X.rows(), "size of event", n);
    stan::math::check_size_match(function_name, "columns of X", X.cols(), "size of beta", beta_ref.size());
    stan::math::check_size_match(function_name, "size of cell", cell.size(), "size of event", n);
    stan::math::check_size_match(function_name, "size of time_into_cell", time_into_cell.size(),
                                 "size of event", n);
    stan::math::check_size_match(function_name, "size of h", h_ref.size(), "size of cell_widths plus 1", M + 1);
    const Eigen::Matrix<double, -1, 1> widths = stan::math::value_of(cell_widths);
    stan::math::check_positive_finite(function_name, "cell_widths", widths);
    const Eigen::Matrix<double, -1, 1> h_values = stan::math::value_of(h_ref);
    stan::math::check_positive_finite(function_name, "h", h_values);
    if (n == 0) return 0.0;
    stan::math::check_bounded(function_name, "event", event, 0, 1);
    stan::math::check_bounded(function_name, "cell", cell, 1, M);
    const Eigen::Map<const Eigen::Matrix<int, -1, 1>> cell_of_each_subject(cell.data(), n);
    if (n > 1 && (cell_of_each_subject.tail(n - 1) - cell_of_each_subject.head(n - 1)).minCoeff() < 0) {
        throw std::domain_error(std::string(function_name)
                                + ": the subjects must be ordered by their cell (cell nondecreasing)");
    }
    const Eigen::Matrix<double, -1, 1> beta_values = stan::math::value_of(beta_ref);
    //// (the data as they are, without copies: X may be a block of rows of the model's data matrix)
    const Eigen::Ref<const Eigen::Matrix<double, -1, -1>, 0, Eigen::OuterStride<>> X_values(X);
    const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> tau(time_into_cell);
    const Eigen::Matrix<double, -1, 1> event_values = Eigen::Map<const Eigen::Matrix<int, -1, 1>>(event.data(),
        n).cast<double>();

    //// ---- H_c, the cumulative baseline hazard at the grid points 1, ..., M (H_1 = 0): the prefix sums of the
    ////      trapezoids 0.5 width_k (h_k + h_(k+1)), in SIMD
    const Eigen::Matrix<double, -1,
                        1> trapezoids = 0.5 * widths.cwiseProduct(h_values.head(M) + h_values.tail(M));
    Eigen::Matrix<double, -1, 1> H_at_the_grid_points(M);
    H_at_the_grid_points(0) = 0.0;
    if (M > 1) {
        H_at_the_grid_points.tail(M - 1) = on_a_grid::cumulative_sums_in_SIMD(trapezoids.head(M - 1), true);
    }

    //// ---- the linear predictor, its exp (BayesMVP's kernel, in place)
    const Eigen::Matrix<double, -1, 1> linear_predictor = X_values * beta_values;
    Eigen::Matrix<double, -1, 1> exp_of_the_linear_predictor = linear_predictor;
    nico::math::k_exp(exp_of_the_linear_predictor.data(), n);

    //// ---- the baseline cumulative hazard and hazard at each subject's time, run of subjects by run (one run
    ////      per cell present), with the run's grid values the same numbers for the whole run; the sum of exp of
    ////      each run for the gradient (the hazard's log comes after, on the whole vector)
    Eigen::Matrix<double, -1, 1> fraction_of_the_cell(n);
    Eigen::Matrix<double, -1, 1> weight_of_the_lower_point(n);
    Eigen::Matrix<double, -1, 1> weight_of_the_upper_point(n);
    Eigen::Matrix<double, -1, 1> H_0_at_t(n);
    Eigen::Matrix<double, -1, 1> h_0_at_t(n);
    Eigen::Matrix<double, -1, 1> sum_of_exp_of_each_cell = Eigen::Matrix<double, -1, 1>::Zero(M);
    for (int first = 0; first < n; ) {
        const int end = on_a_grid::end_of_the_run_of_one_cell(cell, first);
        const int length = end - first;
        const int c = cell[first] - 1;
        const double width = widths(c);
        const auto tau_of_the_run = tau.segment(first, length).array();
        if ((tau_of_the_run < 0.0).any() || (tau_of_the_run > width).any()) {
            throw std::domain_error(std::string(function_name) + ": time_into_cell outside [0, the width of its "
                                    "cell] in cell " + std::to_string(c + 1));
        }
        fraction_of_the_cell.segment(first, length).array() = tau_of_the_run / width;
        weight_of_the_upper_point.segment(first, length).array() = 0.5 * tau_of_the_run.square() / width;
        weight_of_the_lower_point.segment(first, length).array()
            = tau_of_the_run - weight_of_the_upper_point.segment(first, length).array();
        H_0_at_t.segment(first, length).array()
            = H_at_the_grid_points(c) + h_values(c) * weight_of_the_lower_point.segment(first, length).array()
              + h_values(c + 1) * weight_of_the_upper_point.segment(first, length).array();
        h_0_at_t.segment(first, length).array()
            = h_values(c) + (h_values(c + 1) - h_values(c)) * fraction_of_the_cell.segment(first, length).array();
        sum_of_exp_of_each_cell(c) = exp_of_the_linear_predictor.segment(first, length).sum();
        first = end;
    }
    Eigen::Matrix<double, -1, 1> log_h_0_at_t = h_0_at_t;
    nico::math::k_log(log_h_0_at_t.data(), n);
    const double value = event_values.dot(linear_predictor + log_h_0_at_t)
                         - exp_of_the_linear_predictor.dot(H_0_at_t);

    if constexpr (stan::is_constant_all<T_beta, T_h>::value) {
        return value;
    } else {
        //// (the result is one number: its partial derivatives are made here, in the forward pass, and the
        ////  reverse pass only scales them by the result's adjoint; nothing of size n is kept on the tape)
        auto ops_partials = stan::math::make_partials_propagator(beta_ref, h_ref);
        if constexpr (!stan::is_constant_all<T_beta>::value) {
            stan::math::partials<0>(ops_partials)
                = X_values.transpose() * (event_values - exp_of_the_linear_predictor.cwiseProduct(H_0_at_t));
        }
        if constexpr (!stan::is_constant_all<T_h>::value) {
            //// (each subject's derivatives with respect to the lower and upper grid values of its cell)
            const Eigen::Matrix<double, -1, 1> event_over_h_0 = event_values.cwiseQuotient(h_0_at_t);
            const Eigen::Matrix<double, -1, 1> derivative_by_the_lower_point
                = event_over_h_0.cwiseProduct(Eigen::Matrix<double, -1, 1>::Ones(n) - fraction_of_the_cell)
                  - exp_of_the_linear_predictor.cwiseProduct(weight_of_the_lower_point);
            const Eigen::Matrix<double, -1, 1> derivative_by_the_upper_point
                = event_over_h_0.cwiseProduct(fraction_of_the_cell)
                  - exp_of_the_linear_predictor.cwiseProduct(weight_of_the_upper_point);
            Eigen::Matrix<double, -1, 1> partials_of_h = Eigen::Matrix<double, -1, 1>::Zero(M + 1);
            for (int first = 0; first < n; ) {
                const int end = on_a_grid::end_of_the_run_of_one_cell(cell, first);
                const int c = cell[first] - 1;
                partials_of_h(c) += derivative_by_the_lower_point.segment(first, end - first).sum();
                partials_of_h(c + 1) += derivative_by_the_upper_point.segment(first, end - first).sum();
                first = end;
            }
            //// (through H: H_c is the sum of the trapezoids before c, so the adjoint of trapezoid k
            ////  (0-based) is minus the sum of S over the cells after k + 1, i.e. minus the suffix sums of S
            ////  shifted by one, in SIMD; trapezoid k is 0.5 width_k (h_k + h_(k+1)))
            if (M > 1) {
                const Eigen::Matrix<double, -1, 1> sum_of_exp_of_the_cells_after_each_trapezoid
                    = on_a_grid::cumulative_sums_in_SIMD(sum_of_exp_of_each_cell.tail(M - 1), false);
                const Eigen::Matrix<double, -1, 1> half_widths_times_the_adjoints
                    = -0.5 * widths.head(M - 1).cwiseProduct(sum_of_exp_of_the_cells_after_each_trapezoid);
                partials_of_h.head(M - 1) += half_widths_times_the_adjoints;
                partials_of_h.segment(1, M - 1) += half_widths_times_the_adjoints;
            }
            stan::math::partials<1>(ops_partials) = partials_of_h;
        }
        return ops_partials.build(value);
    }

}

}  // namespace nico::math

using nico::math::nico_proportional_hazards_log_likelihood_with_a_piecewise_linear_baseline_hazard_on_a_grid;























