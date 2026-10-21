#pragma once
//
// -| --------- BayesMVP weighted sum of exponentials of an affine predictor by column (Stan external function) -
//
//     nico_dot_product_exp_of_affine_predictor_by_column(w, t, X, a, b, beta)
//           = sum_{q, i} w[q, i] exp(a[i] + b[i] t[q, i] + X[q, i] beta)
//
// with the points stored column by column, Q points per column i = 1, ..., n (the element (q, i) of w and t is
// element (i - 1) Q + q of the vectors w and t, and row (i - 1) Q + q of X), Q = size(w) / n: each column has its
// own intercept a[i] and its own slope b[i] on the covariate t, and every point the common linear part X beta. A
// general function: the cumulative hazard by quadrature of any proportional-hazards model with group (subject)
// effects and time-varying covariates (Q quadrature nodes per subject; w the quadrature weights times the
// half-lengths, t the nodes' times), joint longitudinal-survival models, and the expected counts of any Poisson
// log-linear model with group intercepts and slopes and Q exposures per group.
//
// ONE autodiff node for the whole sum (Stan's version: one node per element of X beta, of the broadcast of a, of
// b .* t, of the sums and of exp), with BayesMVP's exp kernel over all the points; everything else is
// whole-matrix Eigen work (no loop over the points or the columns), and the partial derivatives are made in the
// forward pass (the result is one number). With e[q, i] = exp(...) and g the adjoint of the result:
//
//     adjoint of a[i]   += g sum_q w[q, i] e[q, i]
//     adjoint of b[i]   += g sum_q w[q, i] e[q, i] t[q, i]
//     adjoint of beta   += g X' (w .* e)
//
// Declared in the Stan functions block (no body; stanc --allow-undefined):
//
//     real nico_dot_product_exp_of_affine_predictor_by_column(data vector w, data vector t, data matrix X,
//                                                             vector a, vector b, vector beta);
//
// Included at the end of stan_external_functions.hpp.
//
// -----------------------------------------------------------------------------------------------------------

#include "stan_external_functions.hpp"

#include <stan/math/rev.hpp>
#include <stan/math/prim.hpp>

#include <Eigen/Dense>
#include <ostream>
#include <vector>

// ---- the Stan-facing function (global scope; stanc calls it as fast_...(w, t, X, a, b, beta, pstream__))
namespace nico::math {

template <typename T_w, typename T_t, typename T_X, typename T_a, typename T_b, typename T_beta,
          stan::require_eigen_col_vector_t<T_w>* = nullptr,
          stan::require_eigen_col_vector_t<T_t>* = nullptr,
          stan::require_eigen_matrix_dynamic_t<T_X>* = nullptr,
          stan::require_eigen_col_vector_t<T_a>* = nullptr,
          stan::require_eigen_col_vector_t<T_b>* = nullptr,
          stan::require_eigen_col_vector_t<T_beta>* = nullptr>
inline stan::return_type_t<T_a, T_b, T_beta> nico_dot_product_exp_of_affine_predictor_by_column(
        const T_w& w, const T_t& t, const T_X& X, const T_a& a, const T_b& b, const T_beta& beta,
        std::ostream* /*pstream__*/ = nullptr) {

    using stan::arena_t;
    using stan::math::var;
    static_assert(std::is_arithmetic<stan::value_type_t<T_w>>::value, "w must be data");
    static_assert(std::is_arithmetic<stan::value_type_t<T_t>>::value, "t must be data");
    static_assert(std::is_arithmetic<stan::value_type_t<T_X>>::value, "X must be data");
    static const char* function_name = "nico_dot_product_exp_of_affine_predictor_by_column";
    const auto& a_ref = stan::math::to_ref(a);
    const auto& b_ref = stan::math::to_ref(b);
    const auto& beta_ref = stan::math::to_ref(beta);
    const int n = static_cast<int>(a_ref.size());
    const int M = static_cast<int>(w.size());
    stan::math::check_size_match(function_name, "size of b", b_ref.size(), "size of a", n);
    stan::math::check_size_match(function_name, "size of t", t.size(), "size of w", M);
    stan::math::check_size_match(function_name, "rows of X", X.rows(), "size of w", M);
    stan::math::check_size_match(function_name, "columns of X", X.cols(), "size of beta", beta_ref.size());
    if (n == 0 || M == 0) return 0.0;
    if (M % n != 0) {
        throw std::domain_error(std::string(function_name) + ": size of w (" + std::to_string(M) +
                                ") is not a multiple of the size of a (" + std::to_string(n) + ")");
    }
    const int Q = M / n;
    const Eigen::Matrix<double, -1, 1> a_values = stan::math::value_of(a_ref);
    const Eigen::Matrix<double, -1, 1> b_values = stan::math::value_of(b_ref);
    const Eigen::Matrix<double, -1, 1> beta_values = stan::math::value_of(beta_ref);
    //// (the data as they are, without copies: X may be a block of rows of the model's data matrix)
    const Eigen::Ref<const Eigen::Matrix<double, -1, -1>, 0, Eigen::OuterStride<>> X_values(X);
    const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> w_values(w);
    const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> t_values(t);
    //// (the predictor at every point: the common linear part, then each column's intercept and slope; then exp,
    ////  with BayesMVP's kernel, in place; then the weights)
    Eigen::Matrix<double, -1, -1> weighted_exponentials(Q, n);
    Eigen::Map<Eigen::Matrix<double, -1, 1>>(weighted_exponentials.data(), M).noalias() = X_values * beta_values;
    const Eigen::Map<const Eigen::Matrix<double, -1, -1>> t_by_column(t_values.data(), Q, n);
    weighted_exponentials.array() += t_by_column.array().rowwise() * b_values.transpose().array();
    weighted_exponentials.rowwise() += a_values.transpose();
    nico::math::k_exp(weighted_exponentials.data(), M);
    weighted_exponentials.array() *= Eigen::Map<const Eigen::Matrix<double, -1, -1>>(w_values.data(), Q,
                                                                                     n).array();
    const double value = weighted_exponentials.sum();
    if constexpr (stan::is_constant_all<T_a, T_b, T_beta>::value) {
        return value;
    } else {
        //// (the result is one number: its partial derivatives are made here, in the forward pass, and the
        ////  reverse pass only scales them by the result's adjoint; nothing of size M is kept on the tape)
        auto ops_partials = stan::math::make_partials_propagator(a_ref, b_ref, beta_ref);
        if constexpr (!stan::is_constant_all<T_a>::value) {
            stan::math::partials<0>(ops_partials) = weighted_exponentials.colwise().sum().transpose();
        }
        if constexpr (!stan::is_constant_all<T_b>::value) {
            stan::math::partials<1>(ops_partials) =
                  (weighted_exponentials.array() * t_by_column.array()).matrix().colwise().sum().transpose();
        }
        if constexpr (!stan::is_constant_all<T_beta>::value) {
            stan::math::partials<2>(ops_partials) =
                  X_values.transpose() * Eigen::Map<const Eigen::Matrix<double, -1,
                                                                        1>>(weighted_exponentials.data(), M);
        }
        return ops_partials.build(value);
    }

}

}  // namespace nico::math

using nico::math::nico_dot_product_exp_of_affine_predictor_by_column;

// -| --------- the same sum at the nodes of a quadrature rule, the points' weights and positions made in the node
//
//     nico_dot_product_exp_of_affine_predictor_at_quadrature_nodes_by_column(node_weights, node_positions,
//                                                                           weight_scale, position_scale,
//                                                                           X, a, b, beta)
//           = sum_{q, i} w[q, i] exp(a[i] + b[i] t[q, i] + X[q, i] beta),
//
//     w[q, i] = node_weights[q] weight_scale[i],     t[q, i] = node_positions[q] position_scale[i],
//
// with Q = size(node_weights) points per column and n = size(a) columns: any quadrature whose points are the
// rule's Q nodes scaled to each column's interval, e.g. Gauss-Legendre on [0, T_i] (node_weights the rule's
// weights on [-1, 1], weight_scale T / 2, node_positions (x + 1) / 2, position_scale T). w and t are the same
// products as a model's transformed data makes for nico_dot_product_exp_of_affine_predictor_by_column (e.g.
// to_vector(quad_weights * (T / 2)') and to_vector(quad_time_fraction * T')), so the value and the partials are
// the same to the last bit; only X is read per point, and w and t (2 Q n values) are made per call in this
// thread's own buffers instead of read from memory.
//
// Declared in the Stan functions block (no body; stanc --allow-undefined):
//
//     real nico_dot_product_exp_of_affine_predictor_at_quadrature_nodes_by_column(
//           data vector node_weights, data vector node_positions, data vector weight_scale,
//           data vector position_scale, data matrix X, vector a, vector b, vector beta);
//
namespace nico::math {

template <typename T_node_weights, typename T_node_positions, typename T_weight_scale, typename T_position_scale,
          typename T_X, typename T_a, typename T_b, typename T_beta,
          stan::require_all_eigen_col_vector_t<T_node_weights, T_node_positions, T_weight_scale,
                                               T_position_scale>* = nullptr,
          stan::require_eigen_matrix_dynamic_t<T_X>* = nullptr,
          stan::require_all_eigen_col_vector_t<T_a, T_b, T_beta>* = nullptr>
inline stan::return_type_t<T_a, T_b, T_beta>
nico_dot_product_exp_of_affine_predictor_at_quadrature_nodes_by_column(
        const T_node_weights& node_weights, const T_node_positions& node_positions,
        const T_weight_scale& weight_scale, const T_position_scale& position_scale, const T_X& X, const T_a& a,
        const T_b& b, const T_beta& beta, std::ostream* pstream__ = nullptr) {

    static_assert(stan::is_constant_all<T_node_weights, T_node_positions, T_weight_scale,
                                        T_position_scale>::value,
                  "The quadrature rule and the scales must be data.");
    static const char* function_name = "nico_dot_product_exp_of_affine_predictor_at_quadrature_nodes_by_column";
    const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> node_weight_values(node_weights);
    const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> node_position_values(node_positions);
    const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> weight_scale_values(weight_scale);
    const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> position_scale_values(position_scale);
    const int Q = static_cast<int>(node_weight_values.size());
    const int n = static_cast<int>(weight_scale_values.size());
    stan::math::check_size_match(function_name, "size of node_positions", node_position_values.size(),
                                 "size of node_weights", Q);
    stan::math::check_size_match(function_name, "size of position_scale", position_scale_values.size(),
                                 "size of weight_scale", n);
    const int M = Q * n;
    //// (this thread's buffers of the points' weights and positions, kept between calls)
    static thread_local std::vector<double> weights_of_the_points;
    static thread_local std::vector<double> positions_of_the_points;
    if (static_cast<int>(weights_of_the_points.size()) < M) {
        weights_of_the_points.resize(M);
        positions_of_the_points.resize(M);
    }
    Eigen::Map<Eigen::Matrix<double, -1, -1>> w_by_column(weights_of_the_points.data(), Q, n);
    Eigen::Map<Eigen::Matrix<double, -1, -1>> t_by_column(positions_of_the_points.data(), Q, n);
    w_by_column.noalias() = node_weight_values * weight_scale_values.transpose();
    t_by_column.noalias() = node_position_values * position_scale_values.transpose();
    return nico_dot_product_exp_of_affine_predictor_by_column(
          Eigen::Map<const Eigen::Matrix<double, -1, 1>>(weights_of_the_points.data(), M),
          Eigen::Map<const Eigen::Matrix<double, -1, 1>>(positions_of_the_points.data(),
                                                         M), X, a, b, beta, pstream__);

}

}  // namespace nico::math

using nico::math::nico_dot_product_exp_of_affine_predictor_at_quadrature_nodes_by_column;























