#pragma once
//
// -| --------- NicoSIMD dot products of an affine predictor with indexed effects and its log cumulative sum of
// -| --------- exponentials, with one vector of N values per thread (Stan external function) -------------------
//
//     nico_dot_products_of_affine_predictor_with_indexed_effects_and_log_cumulative_sum_exp_with_one_vector_of_
//     N_values(linear_weights, prefix_weights, X, beta, scale, z, index, rows_in_order_of_index,
//              last_position_of_each_index, offset, rows_per_block_run_in_parallel)
//       = sum_i linear_weights[i] eta[i] + sum_i prefix_weights[i] L[i],
//
//     eta[i] = X[i] beta + scale z[index[i]],     L[i] = log(exp(offset) + sum_{j <= i} exp(eta[j])),
//
// the same value and gradient (up to floating-point rounding) as
// nico_dot_products_of_affine_predictor_with_indexed_effects_and_log_cumulative_sum_exp with the same arguments
// (stan_external_dot_products_of_affine_predictor_with_indexed_effects_and_log_cumulative_sum_exp.hpp and its
// overload whose blocks of rows run in parallel, ..._in_parallel_blocks_of_rows.hpp): the partial likelihood of
// any proportional-hazards model with group effects, and any running total on the log scale of an affine
// predictor with group effects. The same passes, but ONE vector of N values per thread in place of two, so that
// each chain's own state per row is 8 bytes rather than 16 (the data, X, the weights and the indices, are shared
// by the chains of a fit):
//   - the one vector holds eta, then the running sums C (in place of eta, in the pass that makes them), then the
//     derivative of the result at each row (in place of C, in the backward pass);
//   - in the backward pass, prefix_weights / C is made again from C, one block of 1024 rows at a time, and
//     exp(eta - s) is made again from X, beta and the effects (the same arithmetic as the forward pass), in place
//     of reading them from two vectors of N values;
//   - the sums of the derivatives over the rows of each group are made one block of 1024 positions at a time (the
//     derivatives in the order of the groups, their running sum carried from one block to the next, and its
//     values at the last positions of the groups within the block), in place of a second vector of N values.
// Every other step, and every per-element operation (whole-vector Eigen operations, NicoSIMD's exp and log
// kernels and SIMD scans, coefficient 1), is that of the function without _with_one_vector_of_N_values; the loops
// are over the blocks, never over the rows or the groups. The rare case of a first running sum under 1e-280
// uses the shared SIMD prefix and reverse scans, retaining each prefix maximum and its scaled sum.
//
// ONE autodiff node, its partial derivatives made in the forward pass; nothing on the tape but the node.
// With rows_per_block_run_in_parallel at least N (one block, e.g. N_chunks = 1), every pass runs on the calling
// thread; otherwise the blocks of rows_per_block_run_in_parallel rows (rounded up to a multiple of 8) run in
// parallel on the threads of the calling chain, one TBB task per block, in an isolated region of the calling
// arena, as in the overload ..._in_parallel_blocks_of_rows.hpp.
//
// Declared in the Stan functions block (no body; stanc --allow-undefined):
//
//     real nico_dot_products_of_affine_predictor_with_indexed_effects_and_log_cumulative_sum_exp_with_one_vector_
//          of_N_values(data vector linear_weights, data vector prefix_weights, data matrix X, vector beta,
//                      real scale, array[] real z, data array[] int index,
//                      data array[] int rows_in_order_of_index, data array[] int last_position_of_each_index,
//                      real log_offset, data int rows_per_block_run_in_parallel);
//
// -----------------------------------------------------------------------------------------------------------

#include "stan_external_functions.hpp"
#include "stan_external_simd_affine_recurrence_scans.hpp"
#include "stan_external_log_cumulative_sum_exp.hpp"
#include "stan_external_dot_products_of_affine_predictor_with_indexed_effects_and_log_cumulative_sum_exp_in_parallel_blocks_of_rows.hpp"

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

//// (the one vector of N values of the calling thread, kept between calls: a leading zero (for the gathers in the
////  order of the groups), then eta, then C, then the derivatives; padded with zeros)
struct one_vector_of_N_values_of_affine_predictor_with_indexed_effects_t {
    Eigen::Matrix<double, -1, 1> values_of_the_rows;
    Eigen::Matrix<double, -1, 1> effects_with_a_leading_zero;    // 0, then z[1], ..., z[J]
    void ensure(const Eigen::Index n_padded, const Eigen::Index J) {

        if (values_of_the_rows.size() < n_padded + 1) values_of_the_rows.resize(n_padded + 1);
        if (effects_with_a_leading_zero.size() < J + 1) effects_with_a_leading_zero.resize(J + 1);

    }
};
inline one_vector_of_N_values_of_affine_predictor_with_indexed_effects_t&
one_vector_of_N_values_of_affine_predictor_with_indexed_effects() {

    static thread_local one_vector_of_N_values_of_affine_predictor_with_indexed_effects_t buffers;
    return buffers;

}

//// (the vectors of one block of 1024 rows of the thread that runs a pass over a block: exp(eta - s),
////  prefix_weights / C, log C, q, and the derivatives in the order of the groups; padded with zeros to a whole
////  number of registers where a scan reads them)
struct vectors_of_one_block_of_affine_predictor_with_indexed_effects_t {
    Eigen::Matrix<double, -1, 1> exponentials;
    Eigen::Matrix<double, -1, 1> prefix_weights_over_running_sums;
    Eigen::Matrix<double, -1, 1> logs;
    Eigen::Matrix<double, -1, 1> backward_running_sums;
    Eigen::Matrix<double, -1, 1> derivatives_in_order_of_index;
    void ensure(const Eigen::Index block_size) {

        if (exponentials.size() < block_size) {
            exponentials.resize(block_size);
            prefix_weights_over_running_sums.resize(block_size);
            logs.resize(block_size);
            backward_running_sums.resize(block_size);
            derivatives_in_order_of_index.resize(block_size);
        }

    }
};
inline vectors_of_one_block_of_affine_predictor_with_indexed_effects_t&
vectors_of_one_block_of_affine_predictor_with_indexed_effects() {

    static thread_local vectors_of_one_block_of_affine_predictor_with_indexed_effects_t buffers;
    return buffers;

}

//// the running sums, at the last position of each of n_groups groups from first_group, of the derivatives in
//// the order of the groups (from the vector of the derivatives with its leading zero), one block of 1024
//// positions at a time, the running sum carried from one block to the next (a group whose last position is
//// first_position: 0); the loop is over the blocks of positions
inline void running_sums_of_derivatives_at_the_last_positions_one_block_of_positions_at_a_time(
        const double* derivatives_with_a_leading_zero, const Eigen::Index n,
        const Eigen::Map<const Eigen::Matrix<int, -1, 1>>& rows_in_order,
        const Eigen::Map<const Eigen::Matrix<int, -1, 1>>& last_positions, const Eigen::Index first_group,
        const Eigen::Index n_groups, const Eigen::Index first_position, const Eigen::Index end_position,
        Eigen::Matrix<double, -1, 1>& running_sums_at_the_last_positions) {

    const Eigen::Index block_size = rows_per_block_of_affine_predictor_with_indexed_effects;
    auto& block = vectors_of_one_block_of_affine_predictor_with_indexed_effects();
    block.ensure(block_size);
    const affine_scan_coefficients_t coefficients_of_one = affine_scan_coefficients(1.0);
    const Eigen::Map<const Eigen::Matrix<double, -1, 1>> derivatives(derivatives_with_a_leading_zero, n + 1);
    const int* last_position_of_the_groups = last_positions.data() + first_group;
    running_sums_at_the_last_positions.setZero();
    double running_sum_before_the_block = 0.0;
    for (Eigen::Index first = first_position; first < end_position; first += block_size) {
          const Eigen::Index m = std::min(block_size, end_position - first);
          const Eigen::Index m_padded = padded_to_lanes(m);
          block.derivatives_in_order_of_index.head(m) = derivatives(rows_in_order.segment(first, m));
          block.derivatives_in_order_of_index.segment(m, m_padded - m).setZero();
          forward_affine_scan(block.derivatives_in_order_of_index.data(), m_padded, running_sum_before_the_block,
                              block.derivatives_in_order_of_index.data(), coefficients_of_one);
          running_sum_before_the_block = block.derivatives_in_order_of_index[m - 1];
          //// (the groups whose last position is in (first, first + m])
          const Eigen::Index g_begin = std::upper_bound(last_position_of_the_groups,
                                                        last_position_of_the_groups + n_groups,
                                                        static_cast<int>(first)) - last_position_of_the_groups;
          const Eigen::Index g_end = std::upper_bound(last_position_of_the_groups,
                                                      last_position_of_the_groups + n_groups,
                                                      static_cast<int>(first + m)) - last_position_of_the_groups;
          if (g_end > g_begin) {
                const Eigen::Matrix<int, -1, 1> positions_in_the_block =
                      last_positions.segment(first_group + g_begin, g_end - g_begin).array() -
                      static_cast<int>(first + 1);
                running_sums_at_the_last_positions.segment(g_begin, g_end - g_begin) =
                      block.derivatives_in_order_of_index.head(m)(positions_in_the_block);
          }
    }

}

}  // namespace nico::math

// ---- the Stan-facing function (global scope; stanc calls it as nico_...(..., pstream__))
namespace nico::math {

template <typename T_linear_weights, typename T_prefix_weights, typename T_x, typename T_beta, typename T_scale,
          typename T_z, typename T_offset>
inline stan::return_type_t<T_beta, T_scale, T_z, T_offset>
nico_dot_products_of_affine_predictor_with_indexed_effects_and_log_cumulative_sum_exp_with_one_vector_of_N_values(
        const T_linear_weights& linear_weights, const T_prefix_weights& prefix_weights, const T_x& x,
        const T_beta& beta, const T_scale& scale, const std::vector<T_z>& z, const std::vector<int>& index,
        const std::vector<int>& rows_in_order_of_index, const std::vector<int>& last_position_of_each_index,
        const T_offset& offset, const int& rows_per_block_run_in_parallel,
        std::ostream* /*pstream__*/ = nullptr) {

    static_assert(stan::is_constant_all<T_linear_weights, T_prefix_weights, T_x>::value,
                  "The weights and the predictor matrix must be data.");
    static const char* function =
          "nico_dot_products_of_affine_predictor_with_indexed_effects_and_log_cumulative_sum_exp_with_one_vector_"
          "of_N_values";
    namespace lcse = nico::math::log_cumulative_sum_exp;
    const auto& x_ref = stan::math::to_ref(x);
    const auto& beta_ref = stan::math::to_ref(beta);
    const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> linear_values(linear_weights);
    const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> prefix_values(prefix_weights);
    const Eigen::Index n = x_ref.rows();
    const Eigen::Index J = static_cast<Eigen::Index>(z.size());
    const Eigen::Index K = beta_ref.size();
    stan::math::check_size_match(function, "columns of x", x_ref.cols(), "size of beta", K);
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
    //// (the size of the blocks run in parallel: a multiple of 8 rows; at least n: one block)
    const Eigen::Index multiple = nico::math::rows_of_parallel_blocks_are_a_multiple_of;
    const Eigen::Index rows_per_block =
          ((std::max<Eigen::Index>(1, rows_per_block_run_in_parallel) + multiple - 1) / multiple) * multiple;
    const Eigen::Index n_blocks = (rows_per_block >= n) ? 1 : (n + rows_per_block - 1) / rows_per_block;
    const Eigen::Index n_padded = nico::math::padded_to_lanes(n);
    const Eigen::Index cache_block_size = nico::math::rows_per_block_of_affine_predictor_with_indexed_effects;
    auto& buffers = nico::math::one_vector_of_N_values_of_affine_predictor_with_indexed_effects();
    buffers.ensure(n_padded, J);
    //// (row i, 0-based, at position i + 1 of the one vector; position 0: the leading zero of the gathers; the
    ////  padding after row n: zero)
    buffers.values_of_the_rows[0] = 0.0;
    double* values = buffers.values_of_the_rows.data() + 1;
    buffers.values_of_the_rows.segment(1 + n, n_padded - n).setZero();
    //// (the effects with a leading zero, so that the 1-based indices read them directly)
    buffers.effects_with_a_leading_zero[0] = 0.0;
    buffers.effects_with_a_leading_zero.segment(1, J) =
          stan::math::value_of(Eigen::Map<const Eigen::Matrix<T_z, -1, 1>>(z.data(), J));
    const auto effects_with_a_leading_zero = buffers.effects_with_a_leading_zero.head(J + 1);
    const Eigen::Matrix<double, -1, 1> beta_values = stan::math::value_of(beta_ref);
    const double scale_value = stan::math::value_of(scale);
    const double offset_value = stan::math::value_of(offset);
    const nico::math::affine_scan_coefficients_t coefficients_of_one =
          nico::math::affine_scan_coefficients(1.0);
    auto first_row_of_block = [&](const Eigen::Index b) { return b * rows_per_block; };
    auto end_row_of_block = [&](const Eigen::Index b) { return std::min(n, (b + 1) * rows_per_block); };
    //// (eta of m rows from first_row into the m doubles at destination: the same arithmetic in every pass)
    auto eta_of_the_rows = [&](const Eigen::Index first_row, const Eigen::Index m, double* destination) {

          Eigen::Map<Eigen::Matrix<double, -1, 1>> eta(destination, m);
          eta.noalias() = x_ref.middleRows(first_row, m) * beta_values;
          eta += scale_value * effects_with_a_leading_zero(index_of_each_row.segment(first_row, m));

    };
    //// (task(b) for every block: on the calling thread with one block, else one TBB task per block)
    auto for_every_block = [&](const auto& task) {

          if (n_blocks == 1) {
                task(0);
          } else {
                nico::math::run_one_task_per_block_in_parallel(n_blocks, task);
          }

    };
    //// ---- (1) each block's eta (kept in the one vector), its dot product with the linear weights, its largest
    ////      value m_b and, with more than one block, S_b = the sum of exp(eta - m_b) over its rows
    std::vector<double> linear_part_of_block(n_blocks, 0.0);
    std::vector<double> largest_eta_of_block(n_blocks, -std::numeric_limits<double>::infinity());
    std::vector<double> sum_of_exponentials_of_block(n_blocks, 0.0);
    std::vector<char> eta_of_block_is_finite(n_blocks, 1);
    for_every_block([&](const Eigen::Index b) {

          auto& block = nico::math::vectors_of_one_block_of_affine_predictor_with_indexed_effects();
          block.ensure(cache_block_size);
          const Eigen::Index first_row_of_the_block = first_row_of_block(b);
          const Eigen::Index end_row_of_the_block = end_row_of_block(b);
          double linear_part = 0.0;
          double largest_eta = -std::numeric_limits<double>::infinity();
          bool every_eta_is_finite = true;
          for (Eigen::Index first_row = first_row_of_the_block; first_row < end_row_of_the_block;
               first_row += cache_block_size) {

                const Eigen::Index m = std::min(cache_block_size, end_row_of_the_block - first_row);
                eta_of_the_rows(first_row, m, values + first_row);
                const Eigen::Map<const Eigen::Matrix<double, -1, 1>> eta(values + first_row, m);
                every_eta_is_finite = every_eta_is_finite && eta.allFinite();
                linear_part += linear_values.segment(first_row, m).dot(eta);
                largest_eta = std::max(largest_eta, eta.maxCoeff());

          }
          linear_part_of_block[b] = linear_part;
          largest_eta_of_block[b] = largest_eta;
          eta_of_block_is_finite[b] = every_eta_is_finite ? 1 : 0;
          if (!every_eta_is_finite || n_blocks == 1) return;
          double sum_of_exponentials = 0.0;
          for (Eigen::Index first_row = first_row_of_the_block; first_row < end_row_of_the_block;
               first_row += cache_block_size) {

                const Eigen::Index m = std::min(cache_block_size, end_row_of_the_block - first_row);
                block.exponentials.head(m) = Eigen::Map<const Eigen::Matrix<double, -1, 1>>(values + first_row,
                                                                                            m).array() -
                                             largest_eta;
                lcse::nico_exp_of_nonpositive_prefix_arguments(block.exponentials.data(), static_cast<int>(m));
                sum_of_exponentials += block.exponentials.head(m).sum();

          }
          sum_of_exponentials_of_block[b] = sum_of_exponentials;

    });
    if (std::find(eta_of_block_is_finite.begin(), eta_of_block_is_finite.end(), 0) !=
        eta_of_block_is_finite.end()) {

          stan::math::check_finite(function, "affine predictor", Eigen::Map<const Eigen::Matrix<double, -1,
                                   1>>(values, n));

    }
    //// ---- (2) on the calling thread: each block's offset o_b (the log of exp(offset) plus the sum of exp(eta)
    ////      over every row before the block) and its units s_b = max(m_b, o_b); with one block, s_1 is the
    ////      larger of the largest eta and the offset, as in the function without _with_one_vector_of_N_values
    std::vector<double> offset_of_block(n_blocks);
    std::vector<double> shift_of_block(n_blocks);
    bool every_first_running_sum_is_representable = true;
    if (n_blocks == 1) {
          offset_of_block[0] = offset_value;
          shift_of_block[0] = std::max(largest_eta_of_block[0], offset_value);
          every_first_running_sum_is_representable = (std::exp(values[0] - shift_of_block[0]) +
                                                      std::exp(offset_value - shift_of_block[0]) >= 1e-280);
    } else {
          double log_total_before_the_block = offset_value;
          for (Eigen::Index b = 0; b < n_blocks; ++b) {
                offset_of_block[b] = log_total_before_the_block;
                shift_of_block[b] = std::max(largest_eta_of_block[b], log_total_before_the_block);
                const double first_running_sum = std::exp(log_total_before_the_block - shift_of_block[b]) +
                                                 std::exp(values[first_row_of_block(b)] - shift_of_block[b]);
                every_first_running_sum_is_representable = every_first_running_sum_is_representable &&
                                                           (first_running_sum >= 1e-280);
                log_total_before_the_block = nico::math::log_sum_exp_of_two_numbers(
                      log_total_before_the_block,
                      largest_eta_of_block[b] + std::log(sum_of_exponentials_of_block[b]));
          }
    }
    double value = 0.0;
    for (Eigen::Index b = 0; b < n_blocks; ++b) value += linear_part_of_block[b];
    Eigen::Matrix<double, -1, 1> partials_of_beta = Eigen::Matrix<double, -1, 1>::Zero(K);
    double running_derivative_of_the_first_row = 0.0;
    double factor_of_offset = std::exp(offset_value - shift_of_block[0]);
    if (!every_first_running_sum_is_representable) {
          //// ---- the rare case of a first running sum under 1e-280: the values and factors of
          ////      the shared SIMD prefix scan on the calling thread; SIMD reverse derivatives replace eta
          Eigen::Matrix<double, -1, 1> L_of_the_rare_case(n);
          Eigen::Matrix<double, -1, 1> factors_of_the_recursion(n);
          Eigen::Matrix<double, -1, 1> shares_of_the_recursion(n);
          factor_of_offset = lcse::values_and_factors(values, static_cast<int>(n), offset_value,
                                                      L_of_the_rare_case.data(), factors_of_the_recursion.data(),
                                                      shares_of_the_recursion.data());
          value += prefix_values.dot(L_of_the_rare_case);
          if constexpr (stan::is_constant_all<T_beta, T_scale, T_z, T_offset>::value) return value;
          running_derivative_of_the_first_row = lcse::nico_backward_prefix_scan(
                prefix_values.data(), factors_of_the_recursion.data(), static_cast<int>(n), values);
          Eigen::Map<Eigen::Matrix<double, -1, 1>> derivatives(values, n);
          derivatives.array() = linear_values.array() + shares_of_the_recursion.array() * derivatives.array();
          if constexpr (!stan::is_constant_all<T_beta>::value) {
                partials_of_beta.noalias() = x_ref.transpose() * Eigen::Map<const Eigen::Matrix<double, -1,
                                                                                                1>>(values, n);
          }
    } else {
          //// ---- (3) each block in units of exp(s_b): exp(eta - s_b) (one block of 1024 rows at a time),
          ////      its running sums C from exp(o_b - s_b) (the last C of each 1024 rows carried to the next;
          ////      C in place of eta), the dot product of the prefix weights with log C, and the sum R_b of
          ////      prefix_weights / C over the block
          std::vector<double> prefix_part_of_block(n_blocks, 0.0);
          std::vector<double> sum_of_prefix_weights_over_running_sums_of_block(n_blocks, 0.0);
          for_every_block([&](const Eigen::Index b) {

                auto& block = nico::math::vectors_of_one_block_of_affine_predictor_with_indexed_effects();
                block.ensure(cache_block_size);
                const Eigen::Index first_row_of_the_block = first_row_of_block(b);
                const Eigen::Index end_row_of_the_block = end_row_of_block(b);
                double running_sum_before_the_rows = std::exp(offset_of_block[b] - shift_of_block[b]);
                double sum_of_prefix_weights_times_logs = 0.0;
                double sum_of_prefix_weights_over_running_sums = 0.0;
                for (Eigen::Index first_row = first_row_of_the_block; first_row < end_row_of_the_block;
                     first_row += cache_block_size) {

                      const Eigen::Index m = std::min(cache_block_size, end_row_of_the_block - first_row);
                      const Eigen::Index m_padded = nico::math::padded_to_lanes(m);
                      block.exponentials.head(m) =
                            Eigen::Map<const Eigen::Matrix<double, -1, 1>>(values + first_row,
                                                                           m).array() - shift_of_block[b];
                      lcse::nico_exp_of_nonpositive_prefix_arguments(
                            block.exponentials.data(), static_cast<int>(m));
                      block.exponentials.segment(m, m_padded - m).setZero();
                      //// (C in place of eta; the scan writes m_padded values: within the rows of the block but
                      ////  for the padding after row n, whose eta is no longer needed)
                      if (first_row + m_padded <= end_row_of_the_block || end_row_of_the_block == n) {
                            nico::math::forward_affine_scan(block.exponentials.data(), m_padded,
                                                               running_sum_before_the_rows, values + first_row,
                                                               coefficients_of_one);
                      } else {
                            nico::math::forward_affine_scan(block.exponentials.data(), m_padded,
                                                               running_sum_before_the_rows,
                                                               block.logs.data(), coefficients_of_one);
                            Eigen::Map<Eigen::Matrix<double, -1, 1>>(values + first_row, m) = block.logs.head(m);
                      }
                      running_sum_before_the_rows = values[first_row + m - 1];
                      const Eigen::Map<const Eigen::Matrix<double, -1, 1>> C_of_the_rows(values + first_row, m);
                      block.logs.head(m) = C_of_the_rows;
                      lcse::nico_log_of_positive_prefix_sums(block.logs.data(), static_cast<int>(m));
                      sum_of_prefix_weights_times_logs += prefix_values.segment(first_row, m).dot(
                                                                block.logs.head(m));
                      sum_of_prefix_weights_over_running_sums += (prefix_values.segment(first_row, m).array() /
                                                                  C_of_the_rows.array()).sum();

                }
                //// (sum_i prefix_weights[i] L[i] over the block, with L = log C + s_b)
                prefix_part_of_block[b] = sum_of_prefix_weights_times_logs +
                                          shift_of_block[b] * prefix_values.segment(
                                                first_row_of_the_block,
                                                end_row_of_the_block - first_row_of_the_block).sum();
                sum_of_prefix_weights_over_running_sums_of_block[b] = sum_of_prefix_weights_over_running_sums;

          });
          for (Eigen::Index b = 0; b < n_blocks; ++b) value += prefix_part_of_block[b];
          if constexpr (stan::is_constant_all<T_beta, T_scale, T_z, T_offset>::value) return value;
          //// ---- (4) on the calling thread: each block's sum of prefix_weights / C over every row after it, in
          ////      its own units (the last block: none)
          std::vector<double> backward_sum_after_the_block(n_blocks, 0.0);
          for (Eigen::Index b = n_blocks - 1; b-- > 0;) {
                backward_sum_after_the_block[b] = std::exp(shift_of_block[b] - shift_of_block[b + 1]) *
                                                  (sum_of_prefix_weights_over_running_sums_of_block[b + 1] +
                                                   backward_sum_after_the_block[b + 1]);
          }
          //// ---- (5) each block from its last 1024 rows: prefix_weights / C made again from C, its backward
          ////      scan q from that sum (the first q of each 1024 rows carried to the 1024 before),
          ////      exp(eta - s_b) made again from X, beta and the effects, the derivative of the result at each
          ////      row, linear_weights + exp(eta - s_b) q (in place of C), and its product with the rows of x
          Eigen::Matrix<double, -1, -1> partials_of_beta_of_block = Eigen::Matrix<double, -1, -1>::Zero(K,
                                                                                                        n_blocks);
          for_every_block([&](const Eigen::Index b) {

                auto& block = nico::math::vectors_of_one_block_of_affine_predictor_with_indexed_effects();
                block.ensure(cache_block_size);
                const Eigen::Index first_row_of_the_block = first_row_of_block(b);
                const Eigen::Index end_row_of_the_block = end_row_of_block(b);
                double backward_running_sum_after_the_rows = backward_sum_after_the_block[b];
                const Eigen::Index first_row_of_the_last_rows =
                      first_row_of_the_block +
                      ((end_row_of_the_block - 1 - first_row_of_the_block) / cache_block_size) * cache_block_size;
                for (Eigen::Index first_row = first_row_of_the_last_rows; first_row >= first_row_of_the_block;
                     first_row -= cache_block_size) {

                      const Eigen::Index m = std::min(cache_block_size, end_row_of_the_block - first_row);
                      const Eigen::Index m_padded = nico::math::padded_to_lanes(m);
                      Eigen::Map<Eigen::Matrix<double, -1, 1>> values_of_these_rows(values + first_row, m);
                      block.prefix_weights_over_running_sums.head(m) =
                            prefix_values.segment(first_row, m).array() / values_of_these_rows.array();
                      block.prefix_weights_over_running_sums.segment(m, m_padded - m).setZero();
                      nico::math::backward_affine_scan(block.prefix_weights_over_running_sums.data(), m_padded,
                                                          backward_running_sum_after_the_rows,
                                                          block.backward_running_sums.data(),
                                                          coefficients_of_one);
                      backward_running_sum_after_the_rows = block.backward_running_sums[0];
                      eta_of_the_rows(first_row, m, block.exponentials.data());
                      block.exponentials.head(m).array() -= shift_of_block[b];
                      lcse::nico_exp_of_nonpositive_prefix_arguments(
                            block.exponentials.data(), static_cast<int>(m));
                      values_of_these_rows.array() = linear_values.segment(first_row, m).array() +
                                                     block.exponentials.head(m).array() *
                                                     block.backward_running_sums.head(m).array();
                      if constexpr (!stan::is_constant_all<T_beta>::value) {
                            partials_of_beta_of_block.col(b).noalias() +=
                                  x_ref.middleRows(first_row, m).transpose() * values_of_these_rows;
                      }

                }
                if (b == 0) running_derivative_of_the_first_row = backward_running_sum_after_the_rows;

          });
          for (Eigen::Index b = 0; b < n_blocks; ++b) partials_of_beta += partials_of_beta_of_block.col(b);
    }
    //// ---- (6) the sums of the derivatives over the rows of each group, one block of 1024 positions at a time
    ////      (over ranges of the groups in parallel, with more than one block)
    Eigen::Matrix<double, -1, 1> sums_of_each_group = Eigen::Matrix<double, -1, 1>::Zero(J);
    if constexpr (!stan::is_constant_all<T_scale, T_z>::value) {
          const double* derivatives_with_a_leading_zero = buffers.values_of_the_rows.data();
          const Eigen::Index n_ranges_wanted = std::max<Eigen::Index>(1, std::min<Eigen::Index>(n_blocks, J));
          const Eigen::Index groups_per_range = (J + n_ranges_wanted - 1) / n_ranges_wanted;
          const Eigen::Index n_ranges = (J + groups_per_range - 1) / groups_per_range;
          auto sums_of_one_range = [&](const Eigen::Index range) {

                const Eigen::Index first_group = range * groups_per_range;
                const Eigen::Index n_groups = std::min(groups_per_range, J - first_group);
                const Eigen::Index first_position = (first_group == 0) ? 0 : last_positions[first_group - 1];
                const Eigen::Index end_position = last_positions[first_group + n_groups - 1];
                Eigen::Matrix<double, -1, 1> running_sums_at_the_last_positions(n_groups);
                nico::math::running_sums_of_derivatives_at_the_last_positions_one_block_of_positions_at_a_time(
                      derivatives_with_a_leading_zero, n, rows_in_order, last_positions, first_group, n_groups,
                      first_position, end_position, running_sums_at_the_last_positions);
                sums_of_each_group[first_group] = running_sums_at_the_last_positions[0];
                sums_of_each_group.segment(first_group + 1, n_groups - 1) =
                      running_sums_at_the_last_positions.tail(n_groups - 1) -
                      running_sums_at_the_last_positions.head(n_groups - 1);

          };
          if (n_ranges == 1) {
                sums_of_one_range(0);
          } else {
                nico::math::run_one_task_per_block_in_parallel(n_ranges, sums_of_one_range);
          }
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

using nico::math::nico_dot_products_of_affine_predictor_with_indexed_effects_and_log_cumulative_sum_exp_with_one_vector_of_N_values;






















