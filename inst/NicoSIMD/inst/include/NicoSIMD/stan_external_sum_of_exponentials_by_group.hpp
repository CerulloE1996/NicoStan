#pragma once
//
// -| --------- BayesMVP sum of exponentials by group, a Stan external function -------------------------------------
//
//     nico_sum_of_exponentials_by_group(t, group, A, B)[i] = sum_k A[group_i, k] exp(B[group_i, k] t_i)
//
// with t a vector of data (e.g. times), group an array of ints in 1, ..., G (one per element of t), and A and B
// G x K matrices (each group's K coefficients and K exponents). A general function: any multi-exponential curve
// with each group's own coefficients and rates (decay with B < 0, growth with B > 0), e.g. the concentrations of
// any compartmental model, bi-exponential curves, mixtures of exponential terms.
//
// ONE autodiff step for the whole vector (Stan's version: one node per element of B[group] .* t, of exp, of
// A[group] .* exp(...) and of each sum), with BayesMVP's exp kernel over all the elements of each term. With
// e_ik = exp(B[g_i, k] t_i) and the adjoints of the result, g_i:
//
//     adjoint of A[g, k]   += sum_{i: group_i = g} g_i e_ik
//     adjoint of B[g, k]   += sum_{i: group_i = g} g_i A[g, k] t_i e_ik
//
// Declared in the Stan functions block (no body; stanc --allow-undefined):
//
//     vector nico_sum_of_exponentials_by_group(data vector t, data array[] int group, matrix A, matrix B);
//
// Included at the end of stan_external_functions.hpp; its helpers are in their own namespace,
// nico::math::sum_of_exponentials_by_group.
//
// -----------------------------------------------------------------------------------------------------------

#include "stan_external_functions.hpp"

#include <stan/math/rev.hpp>
#include <stan/math/prim.hpp>

#include <Eigen/Dense>
#include <algorithm>
#include <ostream>
#include <vector>

namespace nico::math {
namespace sum_of_exponentials_by_group {

// ---- the values y_i and every e_ik (exponentials[k * M + i]), with the values of A and B (column-major, G rows)
inline void values_and_exponentials(const double* t, const int* group, const int M, const int G, const int K,
                                    const double* A, const double* B, double* y, double* exponentials) {

    std::fill(y, y + M, 0.0);
    for (int k = 0; k < K; ++k) {
        double* e = exponentials + static_cast<std::ptrdiff_t>(k) * M;
        const double* B_of_term = B + static_cast<std::ptrdiff_t>(k) * G;
        const double* A_of_term = A + static_cast<std::ptrdiff_t>(k) * G;
        for (int i = 0; i < M; ++i) e[i] = B_of_term[group[i] - 1] * t[i];
        nico::math::k_exp(e, M);
        for (int i = 0; i < M; ++i) y[i] += A_of_term[group[i] - 1] * e[i];
    }

}

}  // namespace sum_of_exponentials_by_group
}  // namespace nico::math




// ---- the Stan-facing function (global scope; stanc calls it as fast_...(t, group, A, B, pstream__))
namespace nico::math {

template <typename T_t, typename T_group, typename T_A, typename T_B,
          stan::require_eigen_col_vector_t<T_t>* = nullptr,
          stan::require_std_vector_vt<std::is_integral, T_group>* = nullptr,
          stan::require_eigen_matrix_dynamic_t<T_A>* = nullptr,
          stan::require_eigen_matrix_dynamic_t<T_B>* = nullptr>
inline Eigen::Matrix<stan::return_type_t<T_A, T_B>, Eigen::Dynamic, 1> nico_sum_of_exponentials_by_group(
        const T_t& t, const T_group& group, const T_A& A, const T_B& B, std::ostream* /*pstream__*/ = nullptr) {

    namespace soe = nico::math::sum_of_exponentials_by_group;
    using stan::arena_t;
    using stan::math::var;
    static_assert(std::is_arithmetic<stan::value_type_t<T_t>>::value, "t must be data");
    static const char* function_name = "nico_sum_of_exponentials_by_group";
    const auto& A_ref = stan::math::to_ref(A);
    const auto& B_ref = stan::math::to_ref(B);
    const int M = static_cast<int>(t.size());
    const int G = static_cast<int>(A_ref.rows());
    const int K = static_cast<int>(A_ref.cols());
    stan::math::check_consistent_size(function_name, "group", group, M);
    stan::math::check_size_match(function_name, "rows of A", G, "rows of B", B_ref.rows());
    stan::math::check_size_match(function_name, "columns of A", K, "columns of B", B_ref.cols());
    stan::math::check_bounded(function_name, "group", group, 1, G);
    const Eigen::Matrix<double, -1, 1> t_values = stan::math::value_of(t);
    const Eigen::Matrix<double, -1, -1> A_values = stan::math::value_of(A_ref);
    const Eigen::Matrix<double, -1, -1> B_values = stan::math::value_of(B_ref);
    if constexpr (stan::is_constant_all<T_A, T_B>::value) {
        Eigen::Matrix<double, -1, 1> y(M);
        std::vector<double> exponentials(static_cast<std::size_t>(M) * K);
        soe::values_and_exponentials(t_values.data(), group.data(), M, G, K, A_values.data(), B_values.data(),
                                     y.data(), exponentials.data());
        return y;
    } else {
        arena_t<Eigen::Matrix<double, -1, 1>> y(M);
        arena_t<Eigen::Matrix<double, -1, 1>> exponentials(static_cast<Eigen::Index>(M) * K);
        arena_t<Eigen::Matrix<double, -1, 1>> t_arena = t_values;
        arena_t<Eigen::Matrix<double, -1, -1>> A_arena = A_values;
        arena_t<std::vector<int>> group_arena(group.begin(), group.end());
        soe::values_and_exponentials(t_values.data(), group.data(), M, G, K, A_values.data(), B_values.data(),
                                     y.data(), exponentials.data());
        arena_t<Eigen::Matrix<var, Eigen::Dynamic, Eigen::Dynamic>> A_vars = A_ref;
        arena_t<Eigen::Matrix<var, Eigen::Dynamic, Eigen::Dynamic>> B_vars = B_ref;
        arena_t<Eigen::Matrix<var, Eigen::Dynamic, 1>> result = y;
        stan::math::reverse_pass_callback([result, A_vars, B_vars, A_arena, t_arena, exponentials, group_arena,
                                           M, K]() mutable {
            for (int k = 0; k < K; ++k) {
                const double* e = exponentials.data() + static_cast<std::ptrdiff_t>(k) * M;
                // (a run of elements of the same group summed before it is added)
                int i = 0;
                while (i < M) {
                    const int g = group_arena[i] - 1;
                    double sum_for_A = 0.0;
                    double sum_for_B = 0.0;
                    int j = i;
                    for (; j < M && group_arena[j] - 1 == g; ++j) {
                        const double adjoint_times_e = result.coeff(j).adj() * e[j];
                        sum_for_A += adjoint_times_e;
                        sum_for_B += adjoint_times_e * t_arena.coeff(j);
                    }
                    A_vars.coeffRef(g, k).adj() += sum_for_A;
                    B_vars.coeffRef(g, k).adj() += A_arena.coeff(g, k) * sum_for_B;
                    i = j;
                }
            }
        });
        return Eigen::Matrix<var, Eigen::Dynamic, 1>(result);
    }

}

}  // namespace nico::math

using nico::math::nico_sum_of_exponentials_by_group;

// ---- dot_product(w, exp(x)) = sum_i w_i exp(x_i), with BayesMVP's exp kernel and one autodiff node (operands: w and
//      x, those that are parameters; d / d w_i = exp(x_i), d / d x_i = w_i exp(x_i)), in place of Stan's one node
//      per element of exp(x) and the dot product's node. A general function, e.g. any quadrature of a hazard or
//      any weighted sum of exponentials.
//
//     real nico_dot_product_exp(vector w, vector x);
//
namespace nico::math {

template <typename T_w, typename T_x,
          stan::require_eigen_col_vector_t<T_w>* = nullptr,
          stan::require_eigen_col_vector_t<T_x>* = nullptr>
inline stan::return_type_t<T_w, T_x> nico_dot_product_exp(const T_w& w, const T_x& x,
                                                           std::ostream* /*pstream__*/ = nullptr) {

    static const char* function_name = "nico_dot_product_exp";
    const auto& w_ref = stan::math::to_ref(w);
    const auto& x_ref = stan::math::to_ref(x);
    const int M = static_cast<int>(x_ref.size());
    stan::math::check_consistent_sizes(function_name, "w", w_ref, "x", x_ref);
    if (M == 0) return 0.0;
    const Eigen::Matrix<double, -1, 1> w_values = stan::math::value_of(w_ref);
    Eigen::Matrix<double, -1, 1> exp_x = stan::math::value_of(x_ref);
    nico::math::k_exp(exp_x.data(), M);
    const double value = w_values.dot(exp_x);
    if constexpr (stan::is_constant_all<T_w, T_x>::value) {
        return value;
    } else {
        auto ops_partials = stan::math::make_partials_propagator(w_ref, x_ref);
        if constexpr (!stan::is_constant_all<T_w>::value) stan::math::partials<0>(ops_partials) = exp_x;
        if constexpr (!stan::is_constant_all<T_x>::value) {
            stan::math::partials<1>(ops_partials) = w_values.cwiseProduct(exp_x);
        }
        return ops_partials.build(value);
    }

}

}  // namespace nico::math

using nico::math::nico_dot_product_exp;

// ---- sum(log(x)) = sum_i log(x_i), with BayesMVP's log kernel and one autodiff node (d / d x_i = 1 / x_i), in place
//      of Stan's one node per element of log(x) and the sum's node. A general function, e.g. the log of a product
//      of hazards or of any positive terms.
//
//     real nico_sum_log(vector x);
//
namespace nico::math {

template <typename T_x, stan::require_eigen_col_vector_t<T_x>* = nullptr>
inline stan::return_type_t<T_x> nico_sum_log(const T_x& x, std::ostream* /*pstream__*/ = nullptr) {

    const auto& x_ref = stan::math::to_ref(x);
    const int M = static_cast<int>(x_ref.size());
    if (M == 0) return 0.0;
    const Eigen::Matrix<double, -1, 1> x_values = stan::math::value_of(x_ref);
    Eigen::Matrix<double, -1, 1> log_x = x_values;
    nico::math::k_log(log_x.data(), M);
    const double value = log_x.sum();
    if constexpr (stan::is_constant_all<T_x>::value) {
        return value;
    } else {
        auto ops_partials = stan::math::make_partials_propagator(x_ref);
        stan::math::partials<0>(ops_partials) = x_values.cwiseInverse();
        return ops_partials.build(value);
    }

}

}  // namespace nico::math

using nico::math::nico_sum_log;






















