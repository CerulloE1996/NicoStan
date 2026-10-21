#pragma once
//
// -| --------- BayesMVP dot products of an affine predictor with indexed effects and its log cumulative sum of
// -| --------- exponentials (Stan external function) ------------------------------------------------------------
//
//     nico_dot_products_of_affine_predictor_with_indexed_effects_and_log_cumulative_sum_exp(
//           linear_weights, prefix_weights, X, beta, scale, z, index, rows_in_order_of_index,
//           last_position_of_each_index, offset)
//       = sum_i linear_weights[i] eta[i] + sum_i prefix_weights[i] L[i],
//
//     eta[i] = X[i] beta + scale z[index[i]],     L[i] = log(exp(offset) + sum_{j <= i} exp(eta[j])),
//
// with z the effects of the groups (one per group, e.g. the log frailties of the clusters), index[i] the group of
// row i, and, given in the transformed data, rows_in_order_of_index (the rows sorted by their group, e.g.
// sort_indices_asc(index)) and last_position_of_each_index[g] (the number of rows of the groups 1, ..., g). A
// general function: the partial likelihood of any proportional-hazards model with group effects (rows in order of
// decreasing time; linear_weights the event indicators, prefix_weights minus the events of each risk set), and
// any running total on the log scale of an affine predictor with group effects. The same value as
// nico_dot_products_of_affine_predictor_and_log_cumulative_sum_exp(linear_weights, prefix_weights, X, beta,
// scale, z[index], offset), but the effects are read by their index inside the node: no vector of N effects on
// the tape and none in reduce_sum's sliced argument, one partial per group.
//
// ONE autodiff node, its partial derivatives made in the forward pass (the result is one number). Every
// per-element operation is a whole-vector Eigen operation, BayesMVP's exp and log kernels, or BayesMVP's SIMD
// scans (stan_external_simd_affine_recurrence_scans.hpp, coefficient 1): no element-by-element loop over the rows
// or the groups. With t[j] = exp(eta[j] - shift), C[i] = exp(offset - shift) + sum_{j <= i} t[j] (a forward scan)
// and q[j] = sum_{i >= j} prefix_weights[i] / C[i] (a backward scan), the derivative of the result with respect
// to eta[j] is linear_weights[j] + t[j] q[j] (the backward recursion of the log cumulative sum written as a
// running sum: every term of it is at most the sum of the prefix weights); the partials of the groups are the
// differences, at the last position of each group, of the running sum of these derivatives in the order of the
// groups. When C[1] is under 1e-280 (eta[1] far below max(eta)), the values and factors of
// nico::math::log_cumulative_sum_exp are used instead, with SIMD scans for values and reverse derivatives.
// Four passes over the rows, each block by block (1024 rows; the scans carry their last value, or their first
// one backwards, from one block to the next, with the same arithmetic as one scan of every row), so that a
// block's values stay in the core's own cache from one step of a pass to the next: (1) eta, its dot product with
// the linear weights and its maximum; (2) t, C, the dot product of the prefix weights with log C,
// prefix_weights / C; (3) q, the derivatives and their product with x; (4) the sums of the derivatives of each
// group. Two vectors of n values per thread.
//
// Declared in the Stan functions block (no body; stanc --allow-undefined):
//
//     real nico_dot_products_of_affine_predictor_with_indexed_effects_and_log_cumulative_sum_exp(
//           data vector linear_weights, data vector prefix_weights, data matrix X, vector beta, real scale,
//           array[] real z, data array[] int index, data array[] int rows_in_order_of_index,
//           data array[] int last_position_of_each_index, real log_offset);
//
// Included at the end of stan_external_functions.hpp.
//
// -----------------------------------------------------------------------------------------------------------

#include "stan_external_functions.hpp"
#include "stan_external_simd_affine_recurrence_scans.hpp"
#include "stan_external_log_cumulative_sum_exp.hpp"

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

//// (the buffers of the function, kept by each thread between calls: none is allocated per call once they have
////  the size of the largest call. Two vectors of n values per thread, each reused as soon as its contents are no
////  longer needed, and three of one block of rows; every pass over the rows goes block by block, so that the
////  block's values stay in the core's own cache from one step to the next: with 5 vectors of n values and a
////  separate pass for every step, the node's cost per row doubled from n = 10,000 to n = 25,000 on the laptop's
////  16 MB shared cache, 16 chains)
struct buffers_of_affine_predictor_with_indexed_effects_t {
    //// (a leading zero, then: eta, then t = exp(eta - shift); padded with zeros; at the end, the running sums of
    ////  the derivatives in the order of the groups)
    Eigen::Matrix<double, -1, 1> predictor_then_exponentials_then_running_sums_in_order_of_index;
    //// (a leading zero, then: prefix_weights / C, then, in place, the derivative of the result at each row)
    Eigen::Matrix<double, -1, 1> prefix_weights_over_running_sums_then_derivatives;
    Eigen::Matrix<double, -1, 1> running_sums_of_the_block;      // C of one block of rows
    Eigen::Matrix<double, -1, 1> logs_of_the_block;              // log C of one block of rows
    Eigen::Matrix<double, -1, 1> backward_running_sums_of_the_block;  // q of one block of rows
    Eigen::Matrix<double, -1, 1> effects_with_a_leading_zero;    // 0, then z[1], ..., z[J]
    void ensure(const Eigen::Index n_padded, const Eigen::Index J, const Eigen::Index block_size) {

        if (predictor_then_exponentials_then_running_sums_in_order_of_index.size() < n_padded + 1) {
            predictor_then_exponentials_then_running_sums_in_order_of_index.resize(n_padded + 1);
            prefix_weights_over_running_sums_then_derivatives.resize(n_padded + 1);
        }
        if (running_sums_of_the_block.size() < block_size) {
            running_sums_of_the_block.resize(block_size);
            logs_of_the_block.resize(block_size);
            backward_running_sums_of_the_block.resize(block_size);
        }
        if (effects_with_a_leading_zero.size() < J + 1) effects_with_a_leading_zero.resize(J + 1);

    }
};
inline buffers_of_affine_predictor_with_indexed_effects_t& buffers_of_affine_predictor_with_indexed_effects() {

    static thread_local buffers_of_affine_predictor_with_indexed_effects_t buffers;
    return buffers;

}
//// (the rows of one block: a multiple of BMVP_LANES, so that only the last block is padded; 1024 rows, 8 KB per
////  vector of the block)
inline constexpr Eigen::Index rows_per_block_of_affine_predictor_with_indexed_effects = 1024;

}  // namespace nico::math

// ---- the Stan-facing function (global scope; stanc calls it as fast_...(..., pstream__))
namespace nico::math {

template <typename T_linear_weights, typename T_prefix_weights, typename T_x, typename T_beta, typename T_scale,
          typename T_z, typename T_offset>
inline stan::return_type_t<T_beta, T_scale, T_z, T_offset>
nico_dot_products_of_affine_predictor_with_indexed_effects_and_log_cumulative_sum_exp(
        const T_linear_weights& linear_weights, const T_prefix_weights& prefix_weights, const T_x& x,
        const T_beta& beta, const T_scale& scale, const std::vector<T_z>& z, const std::vector<int>& index,
        const std::vector<int>& rows_in_order_of_index, const std::vector<int>& last_position_of_each_index,
        const T_offset& offset, std::ostream* /*pstream__*/ = nullptr) {

    static_assert(stan::is_constant_all<T_linear_weights, T_prefix_weights, T_x>::value,
                  "The weights and the predictor matrix must be data.");
    static const char* function =
          "nico_dot_products_of_affine_predictor_with_indexed_effects_and_log_cumulative_sum_exp";
    namespace lcse = nico::math::log_cumulative_sum_exp;
    const auto& x_ref = stan::math::to_ref(x);
    const auto& beta_ref = stan::math::to_ref(beta);
    const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> linear_values(linear_weights);
    const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> prefix_values(prefix_weights);
    const Eigen::Index n = x_ref.rows();
    const Eigen::Index J = static_cast<Eigen::Index>(z.size());
    stan::math::check_size_match(function, "columns of x", x_ref.cols(), "size of beta", beta_ref.size());
    stan::math::check_size_match(function, "rows of x", n, "size of index", index.size());
    stan::math::check_size_match(function, "rows of x", n, "linear weights", linear_values.size());
    stan::math::check_size_match(function, "rows of x", n, "prefix weights", prefix_values.size());
    stan::math::check_size_match(function, "rows of x", n, "rows in order of index",
                                 rows_in_order_of_index.size());
    stan::math::check_size_match(function, "size of z", J, "last position of each index",
                                 last_position_of_each_index.size());
    if (n == 0) return 0.0;
    //// (the indices, checked as whole vectors)
    const Eigen::Map<const Eigen::Matrix<int, -1, 1>> index_of_each_row(index.data(), n);
    const Eigen::Map<const Eigen::Matrix<int, -1, 1>> rows_in_order(rows_in_order_of_index.data(), n);
    const Eigen::Map<const Eigen::Matrix<int, -1, 1>> last_positions(last_position_of_each_index.data(), J);
    if (index_of_each_row.minCoeff() < 1 || index_of_each_row.maxCoeff() > J || rows_in_order.minCoeff() < 1 ||
        rows_in_order.maxCoeff() > n || last_positions.minCoeff() < 0 || last_positions[J - 1] != n) {

        throw std::domain_error(std::string(function) + ": an index, a row in order of index or a last position "
                                "is outside its range");

    }
    const Eigen::Index n_padded = nico::math::padded_to_lanes(n);
    const Eigen::Index block_size = nico::math::rows_per_block_of_affine_predictor_with_indexed_effects;
    auto& buffers = nico::math::buffers_of_affine_predictor_with_indexed_effects();
    buffers.ensure(n_padded, J, block_size);
    Eigen::Matrix<double, -1,
                  1>& first_vector = buffers.predictor_then_exponentials_then_running_sums_in_order_of_index;
    Eigen::Matrix<double, -1, 1>& second_vector = buffers.prefix_weights_over_running_sums_then_derivatives;
    //// (the effects with a leading zero, so that the 1-based indices read them directly)
    buffers.effects_with_a_leading_zero[0] = 0.0;
    buffers.effects_with_a_leading_zero.segment(1, J) =
          stan::math::value_of(Eigen::Map<const Eigen::Matrix<T_z, -1, 1>>(z.data(), J));
    const Eigen::Matrix<double, -1, 1> beta_values = stan::math::value_of(beta_ref);
    const double scale_value = stan::math::value_of(scale);
    const double offset_value = stan::math::value_of(offset);
    //// (row i, 0-based, of eta and then t at position i + 1 of the first vector; of prefix_weights / C and then
    ////  of the derivatives at position i + 1 of the second; position 0 of each: the leading zero of the gathers)
    double* t = first_vector.data() + 1;
    double* derivatives_of_the_rows = second_vector.data() + 1;
    //// ---- pass 1, block by block: the predictor (the linear part, then each row's group effect), its dot
    ////      product with the linear weights and its largest value
    double linear_part = 0.0;
    double largest_eta = -std::numeric_limits<double>::infinity();
    bool every_eta_is_finite = true;
    for (Eigen::Index first_row = 0; first_row < n; first_row += block_size) {
          const Eigen::Index m = std::min(block_size, n - first_row);
          Eigen::Map<Eigen::Matrix<double, -1, 1>> eta_of_the_block(t + first_row, m);
          eta_of_the_block.noalias() = x_ref.middleRows(first_row, m) * beta_values;
          const auto effects_with_a_leading_zero = buffers.effects_with_a_leading_zero.head(J + 1);
          eta_of_the_block += scale_value * effects_with_a_leading_zero(index_of_each_row.segment(first_row, m));
          every_eta_is_finite = every_eta_is_finite && eta_of_the_block.allFinite();
          linear_part += linear_values.segment(first_row, m).dot(eta_of_the_block);
          largest_eta = std::max(largest_eta, eta_of_the_block.maxCoeff());
    }
    const Eigen::Map<const Eigen::Matrix<double, -1, 1>> eta(t, n);
    if (!every_eta_is_finite) stan::math::check_finite(function, "affine predictor", eta);
    const double shift = std::max(largest_eta, offset_value);
    const double term_of_offset = std::exp(offset_value - shift);
    const bool C_1_is_representable = (std::exp(eta[0] - shift) + term_of_offset >= 1e-280);
    const nico::math::affine_scan_coefficients_t coefficients_of_one =
          nico::math::affine_scan_coefficients(1.0);
    double value = linear_part;
    //// (the partials of beta, of the first row's running derivative and of the offset's factor, made below)
    Eigen::Matrix<double, -1, 1> partials_of_beta = Eigen::Matrix<double, -1, 1>::Zero(beta_values.size());
    double running_derivative_of_the_first_row = 0.0;
    double factor_of_offset = term_of_offset;
    if (!C_1_is_representable) {
          //// ---- C[1] under 1e-280: the shared SIMD prefix values/factors and variable-coefficient reverse scan
          Eigen::Matrix<double, -1, 1> L_of_the_rare_case(n);
          Eigen::Matrix<double, -1, 1> factors_of_the_recursion(n);
          Eigen::Matrix<double, -1, 1> shares_of_the_recursion(n);
          factor_of_offset = lcse::values_and_factors(t, static_cast<int>(n), offset_value,
                                                      L_of_the_rare_case.data(), factors_of_the_recursion.data(),
                                                      shares_of_the_recursion.data());
          value += prefix_values.dot(L_of_the_rare_case);
          if constexpr (stan::is_constant_all<T_beta, T_scale, T_z, T_offset>::value) return value;
          running_derivative_of_the_first_row = lcse::nico_backward_prefix_scan(
                prefix_values.data(), factors_of_the_recursion.data(),
                static_cast<int>(n), derivatives_of_the_rows);
          Eigen::Map<Eigen::Matrix<double, -1, 1>> derivatives(derivatives_of_the_rows, n);
          derivatives.array() = linear_values.array() + shares_of_the_recursion.array() * derivatives.array();
          if constexpr (!stan::is_constant_all<T_beta>::value) {
                partials_of_beta.noalias() = x_ref.transpose() * Eigen::Map<const Eigen::Matrix<double, -1, 1>>(
                                                                       derivatives_of_the_rows, n);
          }
    } else {
          //// ---- pass 2, block by block: t = exp(eta - shift) (in place of eta), the forward scan of t
          ////      (C = exp(offset - shift) + the running sum, the last C of each block carried to the next), the
          ////      dot product of the prefix weights with log C, and prefix_weights / C (for pass 3)
          first_vector.segment(1 + n, n_padded - n).setZero();
          double sum_of_prefix_weights_times_logs = 0.0;
          double running_sum_before_the_block = term_of_offset;
          for (Eigen::Index first_row = 0; first_row < n; first_row += block_size) {
                const Eigen::Index m = std::min(block_size, n - first_row);
                const Eigen::Index m_padded = nico::math::padded_to_lanes(m);
                Eigen::Map<Eigen::Matrix<double, -1, 1>>(t + first_row, m).array() -= shift;
                lcse::nico_exp_of_nonpositive_prefix_arguments(t + first_row, static_cast<int>(m));
                double* C_of_the_block = buffers.running_sums_of_the_block.data();
                nico::math::forward_affine_scan(t + first_row, m_padded, running_sum_before_the_block,
                                                   C_of_the_block, coefficients_of_one);
                running_sum_before_the_block = C_of_the_block[m - 1];
                buffers.logs_of_the_block.head(m) = buffers.running_sums_of_the_block.head(m);
                lcse::nico_log_of_positive_prefix_sums(buffers.logs_of_the_block.data(), static_cast<int>(m));
                sum_of_prefix_weights_times_logs += prefix_values.segment(first_row, m).dot(
                                                          buffers.logs_of_the_block.head(m));
                Eigen::Map<Eigen::Matrix<double, -1, 1>>(derivatives_of_the_rows + first_row, m).array() =
                      prefix_values.segment(first_row, m).array() /
                      buffers.running_sums_of_the_block.head(m).array();
          }
          //// (sum_i prefix_weights[i] L[i], with L = log C + shift)
          value += sum_of_prefix_weights_times_logs + shift * prefix_values.sum();
          if constexpr (stan::is_constant_all<T_beta, T_scale, T_z, T_offset>::value) return value;
          //// ---- pass 3, block by block from the last: q = the backward scan of prefix_weights / C (the first q
          ////      of each block carried to the block before), the derivative of the result at each row,
          ////      linear_weights + t q (in place of prefix_weights / C), and its product with the block of x
          second_vector.segment(1 + n, n_padded - n).setZero();
          double backward_running_sum_after_the_block = 0.0;
          const Eigen::Index first_row_of_the_last_block = ((n - 1) / block_size) * block_size;
          for (Eigen::Index first_row = first_row_of_the_last_block; first_row >= 0; first_row -= block_size) {
                const Eigen::Index m = std::min(block_size, n - first_row);
                const Eigen::Index m_padded = nico::math::padded_to_lanes(m);
                double* q_of_the_block = buffers.backward_running_sums_of_the_block.data();
                nico::math::backward_affine_scan(derivatives_of_the_rows + first_row, m_padded,
                                                    backward_running_sum_after_the_block, q_of_the_block,
                                                    coefficients_of_one);
                backward_running_sum_after_the_block = q_of_the_block[0];
                Eigen::Map<Eigen::Matrix<double, -1,
                                         1>> derivatives_of_the_block(derivatives_of_the_rows + first_row, m);
                derivatives_of_the_block.array() =
                      linear_values.segment(first_row, m).array() +
                      Eigen::Map<const Eigen::Matrix<double, -1, 1>>(t + first_row, m).array() *
                      buffers.backward_running_sums_of_the_block.head(m).array();
                if constexpr (!stan::is_constant_all<T_beta>::value) {
                      partials_of_beta.noalias() += x_ref.middleRows(first_row, m).transpose() *
                                                    derivatives_of_the_block;
                }
          }
          running_derivative_of_the_first_row = backward_running_sum_after_the_block;
    }
    //// ---- pass 4: the sums of the derivatives over the rows of each group: the derivatives in the order of the
    ////      groups (in t's place, no longer needed), their running sum (in place), and its differences at the
    ////      last position of each group
    Eigen::Matrix<double, -1, 1> sums_of_each_group(J);
    if constexpr (!stan::is_constant_all<T_scale, T_z>::value) {
          second_vector[0] = 0.0;
          first_vector[0] = 0.0;
          first_vector.segment(1, n) = second_vector.head(n + 1)(rows_in_order);
          first_vector.segment(n + 1, n_padded - n).setZero();
          nico::math::forward_affine_scan(first_vector.data() + 1, n_padded, 0.0, first_vector.data() + 1,
                                             coefficients_of_one);
          const Eigen::Matrix<double, -1,
                              1> running_sums_at_the_last_positions = first_vector.head(n + 1)(last_positions);
          sums_of_each_group[0] = running_sums_at_the_last_positions[0];
          sums_of_each_group.tail(J - 1) = running_sums_at_the_last_positions.tail(J - 1) -
                                           running_sums_at_the_last_positions.head(J - 1);
    }
    auto partials = stan::math::make_partials_propagator(beta_ref, scale, z, offset);
    if constexpr (!stan::is_constant_all<T_beta>::value) {
          stan::math::partials<0>(partials) = partials_of_beta;
    }
    if constexpr (!stan::is_constant_all<T_scale>::value) {
          //// (sum_i z[index[i]] d eta[i] = sum_g z[g] (the sum of d eta over the rows of group g))
          stan::math::partials<1>(partials) = buffers.effects_with_a_leading_zero.segment(1, J).dot(
                                                    sums_of_each_group);
    }
    if constexpr (!stan::is_constant_all<T_z>::value) {
          stan::math::partials<2>(partials) = scale_value * sums_of_each_group;
    }
    if constexpr (!stan::is_constant_all<T_offset>::value) {
          stan::math::partials<3>(partials) = factor_of_offset * running_derivative_of_the_first_row;
    }
    return partials.build(value);

}

}  // namespace nico::math

using nico::math::nico_dot_products_of_affine_predictor_with_indexed_effects_and_log_cumulative_sum_exp;
























