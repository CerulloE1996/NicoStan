#pragma once
//
// -| --------- NicoSIMD dot products of an affine predictor with indexed effects and its log cumulative sum of
// -| --------- exponentials, with no vector of N values per chain at large N (Stan external function) ----------
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
// predictor with group effects.
//
// Up to rows_up_to_which_the_vectors_of_N_values_are_kept rows: one vector holds eta, then running sums C, then
// the row derivatives, and two more thread-local vectors retain exp(eta - s) and prefix_weights / C for the
// backward pass, which reuses them without another predictor, indexed-effect gather, exponential or division.
// The sums of the row derivatives over each group: with one block, and in the rare case, each derivative is
// added to the sum of its group where it is made (on the calling thread); with more blocks, the existing
// blockwise gather and scan.
// Above that many rows, no vector of N values per chain: each pass reads only the data (X, the weights and the
// indices, shared by the chains of a fit) and makes every value of a row again from it, one block of 1024 rows
// at a time, where the pass needs it, so that each chain's own values stay in its cache at any N:
//   - one block (e.g. N_chunks = 1): one forward pass over the blocks of 1024 rows, in order, each in units of
//     its own s_c = max(its largest eta, o_c) (o_c: the log of exp(offset) plus the sum of exp(eta) over every
//     row before the block, carried from block to block), keeping only o_c and s_c; then one backward pass from
//     the last block, eta, exp(eta - s_c) and C made again, the backward running sums of prefix_weights / C
//     carried into the block before in its units, each derivative added to the sum of its group where it is
//     made;
//   - more blocks, run in parallel: the passes (1), (3) and (5) below, eta and exp(eta - s_b) made again in each,
//     the first C of each 1024 rows kept from (3) for (5), and only the row derivatives kept in the one vector,
//     for the sums over the groups (6).
// There, an eta that is not finite or a first running sum under 1e-280 runs the passes up to that many rows
// instead (with their check and their rare case).
// Every per-element operation (whole-vector Eigen operations, NicoSIMD's exp and log kernels and SIMD scans,
// coefficient 1) is that of the function without _with_one_vector_of_N_values; the loops are over the blocks,
// never over the rows or the groups, but for the additions of the derivatives to the sums of their groups (one
// per row, with one block and in the rare case). The rare case of a first running sum under 1e-280 uses the
// shared SIMD prefix and reverse scans, retaining each prefix maximum and its scaled sum.
// The ranges of the indices are checked at each call with the three arrays of indices; the overload with
// checked_indices_of_affine_predictor_with_one_vector (made once per dataset, its constructor checking the same
// ranges) checks them no more.
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
// nico_dot_products_of_affine_predictor_with_standard_normal_indexed_effects_and_log_cumulative_sum_exp_with_
// one_vector_of_N_values, with the same arguments: the same value plus sum_g -z[g]^2 / 2, the standard normal
// log density of the effects without its constant (as normal_lupdf(z | 0, 1)), its derivatives -z added to the
// partials of z, from the same read of z: one pass over z for both terms instead of two.
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

//// (immutable index data, validated before the model retains them)
class checked_indices_of_affine_predictor_with_one_vector {
    const std::vector<int> index_;
    const std::vector<int> rows_;
    const std::vector<int> last_;

    static std::vector<int> checked_index_copy(
            const std::vector<int>& index,
            const std::vector<int>& rows,
            const std::vector<int>& last,
            const Eigen::Index n,
            const Eigen::Index J) {

        static const char* function =
              "nico_dot_products_of_affine_predictor_with_indexed_effects_and_log_cumulative_sum_exp_"
              "with_one_vector_of_N_values";
        stan::math::check_size_match(function, "rows of x", n, "size of index", index.size());
        stan::math::check_size_match(function, "rows of x", n, "rows in order of index", rows.size());
        stan::math::check_size_match(function, "size of z", J, "last position of each index", last.size());
        if (n == 0) return index;
        const Eigen::Map<const Eigen::Matrix<int, -1, 1>> index_values(index.data(), n);
        const Eigen::Map<const Eigen::Matrix<int, -1, 1>> rows_values(rows.data(), n);
        const Eigen::Map<const Eigen::Matrix<int, -1, 1>> last_values(last.data(), J);
        if (index_values.minCoeff() < 1 || index_values.maxCoeff() > J ||
            rows_values.minCoeff() < 1 || rows_values.maxCoeff() > n ||
            last_values.minCoeff() < 0 || last_values[J - 1] != n) {

            throw std::domain_error(std::string(function) +
                                    ": an index, a row in order of index or a last position "
                                    "is outside its range");

        }
        return index;

    }

public:
    checked_indices_of_affine_predictor_with_one_vector(
            const std::vector<int>& index,
            const std::vector<int>& rows,
            const std::vector<int>& last,
            const Eigen::Index n,
            const Eigen::Index J)
        : index_(checked_index_copy(index, rows, last, n, J)), rows_(rows), last_(last) {

    }

    const std::vector<int>& index() const {

        return index_;

    }

    const std::vector<int>& rows() const {

        return rows_;

    }

    const std::vector<int>& last() const {

        return last_;

    }
};

//// (only the typed checked entry may bypass range validation)
class affine_prefix_with_checked_indices {
    template <bool check_index_ranges, bool with_standard_normal_effects, typename T_linear_weights,
          typename T_prefix_weights, typename T_x, typename T_beta, typename T_scale, typename T_z,
          typename T_offset>
    static stan::return_type_t<T_beta, T_scale, T_z, T_offset>
    evaluate_impl(
        const T_linear_weights& linear_weights,
        const T_prefix_weights& prefix_weights,
        const T_x& x,
        const T_beta& beta,
        const T_scale& scale,
        const std::vector<T_z>& z,
        const std::vector<int>& index,
        const std::vector<int>& rows_in_order_of_index,
        const std::vector<int>& last_position_of_each_index,
        const T_offset& offset,
        const int& rows_per_block_run_in_parallel,
        std::ostream* pstream__);

public:
    template <bool with_standard_normal_effects, typename T_linear_weights, typename T_prefix_weights,
              typename T_x, typename T_beta, typename T_scale, typename T_z, typename T_offset>
    static stan::return_type_t<T_beta, T_scale, T_z, T_offset>
    raw(
        const T_linear_weights& linear_weights,
        const T_prefix_weights& prefix_weights,
        const T_x& x,
        const T_beta& beta,
        const T_scale& scale,
        const std::vector<T_z>& z,
        const std::vector<int>& index,
        const std::vector<int>& rows_in_order_of_index,
        const std::vector<int>& last_position_of_each_index,
        const T_offset& offset,
        const int& rows_per_block_run_in_parallel,
        std::ostream* pstream__);

    template <bool with_standard_normal_effects, typename T_linear_weights, typename T_prefix_weights,
              typename T_x, typename T_beta, typename T_scale, typename T_z, typename T_offset>
    static stan::return_type_t<T_beta, T_scale, T_z, T_offset>
    checked(
        const T_linear_weights& linear_weights,
        const T_prefix_weights& prefix_weights,
        const T_x& x,
        const T_beta& beta,
        const T_scale& scale,
        const std::vector<T_z>& z,
        const checked_indices_of_affine_predictor_with_one_vector& indices,
        const T_offset& offset,
        const int& rows_per_block_run_in_parallel,
        std::ostream* pstream__);
};

//// (the calling thread keeps the row workspace and the middle-pass values between calls)
struct one_vector_of_N_values_of_affine_predictor_with_indexed_effects_t {
    Eigen::Matrix<double, -1, 1> values_of_the_rows;
    Eigen::Matrix<double, -1, 1> exponentials_of_the_rows;
    Eigen::Matrix<double, -1, 1> prefix_weights_over_running_sums_of_the_rows;
    Eigen::Matrix<double, -1, 1> effects_with_a_leading_zero;    // 0, then z[1], ..., z[J]
    void ensure(const Eigen::Index n_padded, const Eigen::Index J) {

        if (values_of_the_rows.size() < n_padded + 1) values_of_the_rows.resize(n_padded + 1);
        if (effects_with_a_leading_zero.size() < J + 1) effects_with_a_leading_zero.resize(J + 1);

    }
    void ensure_middle_pass(const Eigen::Index n) {

        if (exponentials_of_the_rows.size() < n) exponentials_of_the_rows.resize(n);
        if (prefix_weights_over_running_sums_of_the_rows.size() < n) {
            prefix_weights_over_running_sums_of_the_rows.resize(n);
        }

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

//// (above this many rows, no vector of N values per chain: see the head of this file)
#ifndef NICO_ROWS_UP_TO_WHICH_THE_VECTORS_OF_N_VALUES_ARE_KEPT
#define NICO_ROWS_UP_TO_WHICH_THE_VECTORS_OF_N_VALUES_ARE_KEPT 20000
#endif
inline constexpr Eigen::Index rows_up_to_which_the_vectors_of_N_values_are_kept =
      NICO_ROWS_UP_TO_WHICH_THE_VECTORS_OF_N_VALUES_ARE_KEPT;

//// (the vectors of one block of 1024 rows of the thread that makes it again with no vector of N values: eta
////  (then the row derivatives, with one block), exp(eta - s), C, log C (then prefix_weights / C) and q; padded
////  with zeros to a whole number of registers where a scan reads them)
struct vectors_of_one_block_with_no_vector_of_N_values_t {
    Eigen::Matrix<double, -1, 1> eta;
    Eigen::Matrix<double, -1, 1> exponentials;
    Eigen::Matrix<double, -1, 1> running_sums;
    Eigen::Matrix<double, -1, 1> logs;
    Eigen::Matrix<double, -1, 1> backward_running_sums;
    void ensure(const Eigen::Index block_size) {

        if (eta.size() < block_size) {
            eta.resize(block_size);
            exponentials.resize(block_size);
            running_sums.resize(block_size);
            logs.resize(block_size);
            backward_running_sums.resize(block_size);
        }

    }
};
inline vectors_of_one_block_with_no_vector_of_N_values_t& vectors_of_one_block_with_no_vector_of_N_values() {

    static thread_local vectors_of_one_block_with_no_vector_of_N_values_t buffers;
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

//// the sums of the derivatives over the rows of each of n_groups groups from first_group (the differences of
//// their running sums at the last positions of the groups, from the function above, over the positions of these
//// groups)
inline void sums_of_derivatives_over_the_groups_of_one_range(
        const double* derivatives_with_a_leading_zero, const Eigen::Index n,
        const Eigen::Map<const Eigen::Matrix<int, -1, 1>>& rows_in_order,
        const Eigen::Map<const Eigen::Matrix<int, -1, 1>>& last_positions, const Eigen::Index first_group,
        const Eigen::Index n_groups, Eigen::Matrix<double, -1, 1>& sums_of_each_group) {

    const Eigen::Index first_position = (first_group == 0) ? 0 : last_positions[first_group - 1];
    const Eigen::Index end_position = last_positions[first_group + n_groups - 1];
    Eigen::Matrix<double, -1, 1> running_sums_at_the_last_positions(n_groups);
    running_sums_of_derivatives_at_the_last_positions_one_block_of_positions_at_a_time(
          derivatives_with_a_leading_zero, n, rows_in_order, last_positions, first_group, n_groups,
          first_position, end_position, running_sums_at_the_last_positions);
    sums_of_each_group[first_group] = running_sums_at_the_last_positions[0];
    sums_of_each_group.segment(first_group + 1, n_groups - 1) =
          running_sums_at_the_last_positions.tail(n_groups - 1) -
          running_sums_at_the_last_positions.head(n_groups - 1);

}

}  // namespace nico::math

// ---- the Stan-facing function (global scope; stanc calls it as nico_...(..., pstream__))
namespace nico::math {

template <bool check_index_ranges, bool with_standard_normal_effects, typename T_linear_weights,
          typename T_prefix_weights, typename T_x, typename T_beta, typename T_scale, typename T_z,
          typename T_offset>
inline stan::return_type_t<T_beta, T_scale, T_z, T_offset>
affine_prefix_with_checked_indices::evaluate_impl(
        const T_linear_weights& linear_weights, const T_prefix_weights& prefix_weights, const T_x& x,
        const T_beta& beta, const T_scale& scale, const std::vector<T_z>& z, const std::vector<int>& index,
        const std::vector<int>& rows_in_order_of_index, const std::vector<int>& last_position_of_each_index,
        const T_offset& offset, const int& rows_per_block_run_in_parallel,
        std::ostream* /*pstream__*/) {

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
    //// (no rows: only the standard normal log density of the effects, -z'z / 2, and its derivatives -z)
    if (n == 0) {

        if constexpr (!with_standard_normal_effects) {
              return 0.0;
        } else {
              const Eigen::Matrix<double, -1, 1> effect_values =
                    stan::math::value_of(Eigen::Map<const Eigen::Matrix<T_z, -1, 1>>(z.data(), J));
              auto partials = stan::math::make_partials_propagator(beta_ref, scale, z, offset);
              if constexpr (!stan::is_constant_all<T_z>::value) {
                    stan::math::partials<2>(partials) = -effect_values;
              }
              return partials.build(-0.5 * effect_values.squaredNorm());
        }

    }
    //// (the indices, checked as whole vectors)
    const Eigen::Map<const Eigen::Matrix<int, -1, 1>> index_of_each_row(index.data(), n);
    const Eigen::Map<const Eigen::Matrix<int, -1, 1>> rows_in_order(rows_in_order_of_index.data(), n);
    const Eigen::Map<const Eigen::Matrix<int, -1, 1>> last_positions(last_position_of_each_index.data(), J);
    if constexpr (check_index_ranges) {

        if (index_of_each_row.minCoeff() < 1 || index_of_each_row.maxCoeff() > J ||
            rows_in_order.minCoeff() < 1 || rows_in_order.maxCoeff() > n ||
            last_positions.minCoeff() < 0 || last_positions[J - 1] != n) {
    
            throw std::domain_error(std::string(function) +
                                    ": an index, a row in order of index or a last position "
                                    "is outside its range");
    
        }

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
    //// (with standard normal effects: their log density -z'z / 2 (without its constant), from the effects just
    ////  read, added to every value returned below; their derivatives -z added to those of z)
    double log_density_of_the_effects = 0.0;
    if constexpr (with_standard_normal_effects) {
          log_density_of_the_effects = -0.5 * buffers.effects_with_a_leading_zero.segment(1, J).squaredNorm();
    }
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
    //// ---- above rows_up_to_which_the_vectors_of_N_values_are_kept rows: no vector of N values per chain
    ////      (but the row derivatives, with more than one block); an eta that is not finite, or a first running
    ////      sum under 1e-280, leaves this branch for the passes (1) to (6) below
    if (n > nico::math::rows_up_to_which_the_vectors_of_N_values_are_kept) {

          bool swept = true;
          double value = 0.0;
          Eigen::Matrix<double, -1, 1> partials_of_beta = Eigen::Matrix<double, -1, 1>::Zero(K);
          Eigen::Matrix<double, -1, 1> sums_of_each_group = Eigen::Matrix<double, -1, 1>::Zero(J);
          double running_derivative_of_the_first_row = 0.0;
          double factor_of_offset = 0.0;
          if (n_blocks == 1) {

                auto& sweep = nico::math::vectors_of_one_block_with_no_vector_of_N_values();
                sweep.ensure(cache_block_size);
                const Eigen::Index n_cache_blocks = (n + cache_block_size - 1) / cache_block_size;
                //// ---- forward, from the first block of 1024 rows: eta, its dot product with the linear
                ////      weights, s_c = max(its largest eta, o_c), exp(eta - s_c), its running sums C from
                ////      exp(o_c - s_c), the dot product of the prefix weights with log C + s_c, and o_c of the
                ////      next block, s_c + log of the last C; o_c and s_c kept for the backward pass
                std::vector<double> offset_of_cache_block(n_cache_blocks);
                std::vector<double> shift_of_cache_block(n_cache_blocks);
                double log_total_before_the_rows = offset_value;
                for (Eigen::Index c = 0; c < n_cache_blocks; ++c) {

                      const Eigen::Index first_row = c * cache_block_size;
                      const Eigen::Index m = std::min(cache_block_size, n - first_row);
                      const Eigen::Index m_padded = nico::math::padded_to_lanes(m);
                      eta_of_the_rows(first_row, m, sweep.eta.data());
                      const auto eta = sweep.eta.head(m);
                      if (!eta.allFinite()) {

                            swept = false;
                            break;

                      }
                      const double shift = std::max(eta.maxCoeff(), log_total_before_the_rows);
                      sweep.exponentials.head(m) = eta.array() - shift;
                      lcse::nico_exp_of_nonpositive_prefix_arguments(sweep.exponentials.data(),
                                                                     static_cast<int>(m));
                      sweep.exponentials.segment(m, m_padded - m).setZero();
                      nico::math::forward_affine_scan(sweep.exponentials.data(), m_padded,
                                                      std::exp(log_total_before_the_rows - shift),
                                                      sweep.running_sums.data(), coefficients_of_one);
                      if (!(sweep.running_sums[0] >= 1e-280)) {

                            swept = false;
                            break;

                      }
                      sweep.logs.head(m) = sweep.running_sums.head(m);
                      lcse::nico_log_of_positive_prefix_sums(sweep.logs.data(), static_cast<int>(m));
                      value += linear_values.segment(first_row, m).dot(eta) +
                               prefix_values.segment(first_row, m).dot(sweep.logs.head(m)) +
                               shift * prefix_values.segment(first_row, m).sum();
                      offset_of_cache_block[c] = log_total_before_the_rows;
                      shift_of_cache_block[c] = shift;
                      log_total_before_the_rows = shift + sweep.logs[m - 1];

                }
                if (swept) {

                      if constexpr (stan::is_constant_all<T_beta, T_scale, T_z, T_offset>::value) {
                            return value + log_density_of_the_effects;
                      }
                      //// ---- backward, from the last block of 1024 rows: eta, exp(eta - s_c) and C made
                      ////      again, prefix_weights / C, its backward running sums q (the first q of the block
                      ////      carried to the block before, times exp(s_{c-1} - s_c)), the row derivatives
                      ////      linear_weights + exp(eta - s_c) q (in place of eta), their products with the
                      ////      rows of x, and each added to the sum of its group
                      double backward_running_sum_after_the_rows = 0.0;
                      for (Eigen::Index c = n_cache_blocks; c-- > 0;) {

                            const Eigen::Index first_row = c * cache_block_size;
                            const Eigen::Index m = std::min(cache_block_size, n - first_row);
                            const Eigen::Index m_padded = nico::math::padded_to_lanes(m);
                            eta_of_the_rows(first_row, m, sweep.eta.data());
                            sweep.exponentials.head(m) = sweep.eta.head(m).array() - shift_of_cache_block[c];
                            lcse::nico_exp_of_nonpositive_prefix_arguments(sweep.exponentials.data(),
                                                                           static_cast<int>(m));
                            sweep.exponentials.segment(m, m_padded - m).setZero();
                            nico::math::forward_affine_scan(sweep.exponentials.data(), m_padded,
                                                            std::exp(offset_of_cache_block[c] -
                                                                     shift_of_cache_block[c]),
                                                            sweep.running_sums.data(), coefficients_of_one);
                            sweep.logs.head(m) = prefix_values.segment(first_row, m).array() /
                                                 sweep.running_sums.head(m).array();
                            sweep.logs.segment(m, m_padded - m).setZero();
                            nico::math::backward_affine_scan(sweep.logs.data(), m_padded,
                                                             backward_running_sum_after_the_rows,
                                                             sweep.backward_running_sums.data(),
                                                             coefficients_of_one);
                            Eigen::Map<Eigen::Matrix<double, -1, 1>> derivatives(sweep.eta.data(), m);
                            derivatives.array() = linear_values.segment(first_row, m).array() +
                                                  sweep.exponentials.head(m).array() *
                                                  sweep.backward_running_sums.head(m).array();
                            if constexpr (!stan::is_constant_all<T_scale, T_z>::value) {
                                  for (Eigen::Index j = 0; j < m; ++j) {
                                        sums_of_each_group[index_of_each_row[first_row + j] - 1] +=
                                              derivatives[j];
                                  }
                            }
                            if constexpr (!stan::is_constant_all<T_beta>::value) {
                                  partials_of_beta.noalias() += x_ref.middleRows(first_row, m).transpose() *
                                                                derivatives;
                            }
                            backward_running_sum_after_the_rows =
                                  (c > 0) ? std::exp(shift_of_cache_block[c - 1] - shift_of_cache_block[c]) *
                                            sweep.backward_running_sums[0]
                                          : sweep.backward_running_sums[0];

                      }
                      running_derivative_of_the_first_row = backward_running_sum_after_the_rows;
                      factor_of_offset = std::exp(offset_value - shift_of_cache_block[0]);

                }

          } else {

                //// ---- (1) each block, one block of 1024 rows at a time: eta, its dot product with the
                ////      linear weights, its first eta, its largest value m_b and S_b = the sum of
                ////      exp(eta - m_b) over its rows (the sum so far carried in the units of the largest eta
                ////      so far)
                std::vector<double> linear_part_of_block(n_blocks, 0.0);
                std::vector<double> first_eta_of_block(n_blocks, 0.0);
                std::vector<double> largest_eta_of_block(n_blocks, -std::numeric_limits<double>::infinity());
                std::vector<double> sum_of_exponentials_of_block(n_blocks, 0.0);
                std::vector<char> eta_of_block_is_finite(n_blocks, 1);
                for_every_block([&](const Eigen::Index b) {

                      auto& sweep = nico::math::vectors_of_one_block_with_no_vector_of_N_values();
                      sweep.ensure(cache_block_size);
                      const Eigen::Index first_row_of_the_block = first_row_of_block(b);
                      const Eigen::Index end_row_of_the_block = end_row_of_block(b);
                      double linear_part = 0.0;
                      double largest_eta = -std::numeric_limits<double>::infinity();
                      double sum_of_exponentials = 0.0;
                      for (Eigen::Index first_row = first_row_of_the_block; first_row < end_row_of_the_block;
                           first_row += cache_block_size) {

                            const Eigen::Index m = std::min(cache_block_size, end_row_of_the_block - first_row);
                            eta_of_the_rows(first_row, m, sweep.eta.data());
                            const auto eta = sweep.eta.head(m);
                            if (!eta.allFinite()) {

                                  eta_of_block_is_finite[b] = 0;
                                  return;

                            }
                            if (first_row == first_row_of_the_block) first_eta_of_block[b] = sweep.eta[0];
                            linear_part += linear_values.segment(first_row, m).dot(eta);
                            const double largest_eta_so_far = std::max(largest_eta, eta.maxCoeff());
                            sweep.exponentials.head(m) = eta.array() - largest_eta_so_far;
                            lcse::nico_exp_of_nonpositive_prefix_arguments(sweep.exponentials.data(),
                                                                           static_cast<int>(m));
                            sum_of_exponentials = sum_of_exponentials *
                                                  std::exp(largest_eta - largest_eta_so_far) +
                                                  sweep.exponentials.head(m).sum();
                            largest_eta = largest_eta_so_far;

                      }
                      linear_part_of_block[b] = linear_part;
                      largest_eta_of_block[b] = largest_eta;
                      sum_of_exponentials_of_block[b] = sum_of_exponentials;

                });
                swept = (std::find(eta_of_block_is_finite.begin(), eta_of_block_is_finite.end(), 0) ==
                         eta_of_block_is_finite.end());
                //// ---- (2) on the calling thread: each block's offset o_b and units s_b = max(m_b, o_b), as
                ////      in (2) below
                std::vector<double> offset_of_block(n_blocks);
                std::vector<double> shift_of_block(n_blocks);
                double log_total_before_the_block = offset_value;
                for (Eigen::Index b = 0; b < n_blocks && swept; ++b) {

                      offset_of_block[b] = log_total_before_the_block;
                      shift_of_block[b] = std::max(largest_eta_of_block[b], log_total_before_the_block);
                      const double first_running_sum = std::exp(log_total_before_the_block - shift_of_block[b]) +
                                                       std::exp(first_eta_of_block[b] - shift_of_block[b]);
                      swept = (first_running_sum >= 1e-280);
                      log_total_before_the_block = nico::math::log_sum_exp_of_two_numbers(
                            log_total_before_the_block,
                            largest_eta_of_block[b] + std::log(sum_of_exponentials_of_block[b]));

                }
                if (swept) {

                      for (Eigen::Index b = 0; b < n_blocks; ++b) value += linear_part_of_block[b];
                      //// ---- (3) each block in units of exp(s_b), one block of 1024 rows at a time: eta and
                      ////      exp(eta - s_b) made again, its running sums C from exp(o_b - s_b) (the C before
                      ////      each 1024 rows kept for (5)), the dot product of the prefix weights with log C,
                      ////      and the sum R_b of prefix_weights / C over the block
                      const Eigen::Index cache_blocks_per_block =
                            (rows_per_block + cache_block_size - 1) / cache_block_size;
                      std::vector<double> running_sum_before_the_cache_block(n_blocks * cache_blocks_per_block,
                                                                             0.0);
                      std::vector<double> prefix_part_of_block(n_blocks, 0.0);
                      std::vector<double> sum_of_prefix_weights_over_running_sums_of_block(n_blocks, 0.0);
                      for_every_block([&](const Eigen::Index b) {

                            auto& sweep = nico::math::vectors_of_one_block_with_no_vector_of_N_values();
                            sweep.ensure(cache_block_size);
                            const Eigen::Index first_row_of_the_block = first_row_of_block(b);
                            const Eigen::Index end_row_of_the_block = end_row_of_block(b);
                            double running_sum_before_the_rows = std::exp(offset_of_block[b] - shift_of_block[b]);
                            double sum_of_prefix_weights_times_logs = 0.0;
                            double sum_of_prefix_weights_over_running_sums = 0.0;
                            Eigen::Index c = b * cache_blocks_per_block;
                            for (Eigen::Index first_row = first_row_of_the_block;
                                 first_row < end_row_of_the_block; first_row += cache_block_size, ++c) {

                                  const Eigen::Index m = std::min(cache_block_size,
                                                                  end_row_of_the_block - first_row);
                                  const Eigen::Index m_padded = nico::math::padded_to_lanes(m);
                                  eta_of_the_rows(first_row, m, sweep.eta.data());
                                  sweep.exponentials.head(m) = sweep.eta.head(m).array() - shift_of_block[b];
                                  lcse::nico_exp_of_nonpositive_prefix_arguments(sweep.exponentials.data(),
                                                                                 static_cast<int>(m));
                                  sweep.exponentials.segment(m, m_padded - m).setZero();
                                  running_sum_before_the_cache_block[c] = running_sum_before_the_rows;
                                  nico::math::forward_affine_scan(sweep.exponentials.data(), m_padded,
                                                                  running_sum_before_the_rows,
                                                                  sweep.running_sums.data(), coefficients_of_one);
                                  running_sum_before_the_rows = sweep.running_sums[m - 1];
                                  sweep.logs.head(m) = sweep.running_sums.head(m);
                                  lcse::nico_log_of_positive_prefix_sums(sweep.logs.data(), static_cast<int>(m));
                                  sum_of_prefix_weights_times_logs += prefix_values.segment(first_row, m).dot(
                                                                            sweep.logs.head(m));
                                  sum_of_prefix_weights_over_running_sums +=
                                        (prefix_values.segment(first_row, m).array() /
                                         sweep.running_sums.head(m).array()).sum();

                            }
                            //// (sum_i prefix_weights[i] L[i] over the block, with L = log C + s_b)
                            prefix_part_of_block[b] = sum_of_prefix_weights_times_logs +
                                                      shift_of_block[b] * prefix_values.segment(
                                                            first_row_of_the_block,
                                                            end_row_of_the_block - first_row_of_the_block).sum();
                            sum_of_prefix_weights_over_running_sums_of_block[b] =
                                  sum_of_prefix_weights_over_running_sums;

                      });
                      for (Eigen::Index b = 0; b < n_blocks; ++b) value += prefix_part_of_block[b];
                      if constexpr (stan::is_constant_all<T_beta, T_scale, T_z, T_offset>::value) {
                            return value + log_density_of_the_effects;
                      }
                      //// ---- (4) on the calling thread: each block's sum of prefix_weights / C over every row
                      ////      after it, in its own units (the last block: none)
                      std::vector<double> backward_sum_after_the_block(n_blocks, 0.0);
                      for (Eigen::Index b = n_blocks - 1; b-- > 0;) {
                            backward_sum_after_the_block[b] =
                                  std::exp(shift_of_block[b] - shift_of_block[b + 1]) *
                                  (sum_of_prefix_weights_over_running_sums_of_block[b + 1] +
                                   backward_sum_after_the_block[b + 1]);
                      }
                      //// ---- (5) each block from its last 1024 rows: eta, exp(eta - s_b) and C (from the C kept
                      ////      before each 1024 rows in (3)) made again, prefix_weights / C, its backward scan q
                      ////      (the first q of each 1024 rows carried to the 1024 before), the row derivatives
                      ////      linear_weights + exp(eta - s_b) q (kept in the one vector, for (6)), and their
                      ////      products with the rows of x
                      Eigen::Matrix<double, -1, -1> partials_of_beta_of_block =
                            Eigen::Matrix<double, -1, -1>::Zero(K, n_blocks);
                      for_every_block([&](const Eigen::Index b) {

                            auto& sweep = nico::math::vectors_of_one_block_with_no_vector_of_N_values();
                            sweep.ensure(cache_block_size);
                            const Eigen::Index first_row_of_the_block = first_row_of_block(b);
                            const Eigen::Index end_row_of_the_block = end_row_of_block(b);
                            double backward_running_sum_after_the_rows = backward_sum_after_the_block[b];
                            const Eigen::Index first_row_of_the_last_rows =
                                  first_row_of_the_block +
                                  ((end_row_of_the_block - 1 - first_row_of_the_block) / cache_block_size) *
                                  cache_block_size;
                            Eigen::Index c = b * cache_blocks_per_block +
                                             (first_row_of_the_last_rows - first_row_of_the_block) /
                                             cache_block_size;
                            for (Eigen::Index first_row = first_row_of_the_last_rows;
                                 first_row >= first_row_of_the_block; first_row -= cache_block_size, --c) {

                                  const Eigen::Index m = std::min(cache_block_size,
                                                                  end_row_of_the_block - first_row);
                                  const Eigen::Index m_padded = nico::math::padded_to_lanes(m);
                                  eta_of_the_rows(first_row, m, sweep.eta.data());
                                  sweep.exponentials.head(m) = sweep.eta.head(m).array() - shift_of_block[b];
                                  lcse::nico_exp_of_nonpositive_prefix_arguments(sweep.exponentials.data(),
                                                                                 static_cast<int>(m));
                                  sweep.exponentials.segment(m, m_padded - m).setZero();
                                  nico::math::forward_affine_scan(sweep.exponentials.data(), m_padded,
                                                                  running_sum_before_the_cache_block[c],
                                                                  sweep.running_sums.data(), coefficients_of_one);
                                  sweep.logs.head(m) = prefix_values.segment(first_row, m).array() /
                                                       sweep.running_sums.head(m).array();
                                  sweep.logs.segment(m, m_padded - m).setZero();
                                  nico::math::backward_affine_scan(sweep.logs.data(), m_padded,
                                                                   backward_running_sum_after_the_rows,
                                                                   sweep.backward_running_sums.data(),
                                                                   coefficients_of_one);
                                  backward_running_sum_after_the_rows = sweep.backward_running_sums[0];
                                  Eigen::Map<Eigen::Matrix<double, -1, 1>> derivatives(values + first_row, m);
                                  derivatives.array() = linear_values.segment(first_row, m).array() +
                                                        sweep.exponentials.head(m).array() *
                                                        sweep.backward_running_sums.head(m).array();
                                  if constexpr (!stan::is_constant_all<T_beta>::value) {
                                        partials_of_beta_of_block.col(b).noalias() +=
                                              x_ref.middleRows(first_row, m).transpose() * derivatives;
                                  }

                            }
                            if (b == 0) running_derivative_of_the_first_row = backward_running_sum_after_the_rows;

                      });
                      for (Eigen::Index b = 0; b < n_blocks; ++b) {
                            partials_of_beta += partials_of_beta_of_block.col(b);
                      }
                      factor_of_offset = std::exp(offset_value - shift_of_block[0]);
                      //// ---- (6) the sums of the derivatives over the rows of each group, one block of 1024
                      ////      positions at a time (over ranges of the groups in parallel), as in (6) below
                      if constexpr (!stan::is_constant_all<T_scale, T_z>::value) {
                            const double* derivatives_with_a_leading_zero = buffers.values_of_the_rows.data();
                            const Eigen::Index n_ranges_wanted =
                                  std::max<Eigen::Index>(1, std::min<Eigen::Index>(n_blocks, J));
                            const Eigen::Index groups_per_range = (J + n_ranges_wanted - 1) / n_ranges_wanted;
                            const Eigen::Index n_ranges = (J + groups_per_range - 1) / groups_per_range;
                            auto sums_of_one_range = [&](const Eigen::Index range) {

                                  const Eigen::Index first_group = range * groups_per_range;
                                  nico::math::sums_of_derivatives_over_the_groups_of_one_range(
                                        derivatives_with_a_leading_zero, n, rows_in_order, last_positions,
                                        first_group, std::min(groups_per_range, J - first_group),
                                        sums_of_each_group);

                            };
                            if (n_ranges == 1) {
                                  sums_of_one_range(0);
                            } else {
                                  nico::math::run_one_task_per_block_in_parallel(n_ranges, sums_of_one_range);
                            }
                      }

                }

          }
          if (swept) {

                auto partials = stan::math::make_partials_propagator(beta_ref, scale, z, offset);
                if constexpr (!stan::is_constant_all<T_beta>::value) {
                      stan::math::partials<0>(partials) = partials_of_beta;
                }
                if constexpr (!stan::is_constant_all<T_scale>::value) {
                      stan::math::partials<1>(partials) = buffers.effects_with_a_leading_zero.segment(1, J).dot(
                                                                sums_of_each_group);
                }
                if constexpr (!stan::is_constant_all<T_z>::value) {
                      if constexpr (with_standard_normal_effects) {
                            stan::math::partials<2>(partials) =
                                  scale_value * sums_of_each_group -
                                  buffers.effects_with_a_leading_zero.segment(1, J);
                      } else {
                            stan::math::partials<2>(partials) = scale_value * sums_of_each_group;
                      }
                }
                if constexpr (!stan::is_constant_all<T_offset>::value) {
                      stan::math::partials<3>(partials) = factor_of_offset * running_derivative_of_the_first_row;
                }
                return partials.build(value + log_density_of_the_effects);

          }

    }
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
    //// (the sums of the derivatives over the rows of each group: each derivative added to the sum of its group
    ////  where it is made, on the calling thread, with one block and in the rare case; with more blocks, in (6))
    Eigen::Matrix<double, -1, 1> sums_of_each_group = Eigen::Matrix<double, -1, 1>::Zero(J);
    bool sums_of_each_group_are_added_where_the_derivatives_are_made = (n_blocks == 1);
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
          if constexpr (stan::is_constant_all<T_beta, T_scale, T_z, T_offset>::value) {
                return value + log_density_of_the_effects;
          }
          running_derivative_of_the_first_row = lcse::nico_backward_prefix_scan(
                prefix_values.data(), factors_of_the_recursion.data(), static_cast<int>(n), values);
          Eigen::Map<Eigen::Matrix<double, -1, 1>> derivatives(values, n);
          derivatives.array() = linear_values.array() + shares_of_the_recursion.array() * derivatives.array();
          if constexpr (!stan::is_constant_all<T_scale, T_z>::value) {
                for (Eigen::Index i = 0; i < n; ++i) sums_of_each_group[index_of_each_row[i] - 1] += values[i];
          }
          sums_of_each_group_are_added_where_the_derivatives_are_made = true;
          if constexpr (!stan::is_constant_all<T_beta>::value) {
                partials_of_beta.noalias() = x_ref.transpose() * Eigen::Map<const Eigen::Matrix<double, -1,
                                                                                                1>>(values, n);
          }
    } else {
          buffers.ensure_middle_pass(n);
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
                      buffers.exponentials_of_the_rows.segment(first_row, m) = block.exponentials.head(m);
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
                      block.prefix_weights_over_running_sums.head(m) =
                            prefix_values.segment(first_row, m).array() / C_of_the_rows.array();
                      buffers.prefix_weights_over_running_sums_of_the_rows.segment(first_row, m) =
                            block.prefix_weights_over_running_sums.head(m);
                      sum_of_prefix_weights_over_running_sums +=
                            block.prefix_weights_over_running_sums.head(m).sum();

                }
                //// (sum_i prefix_weights[i] L[i] over the block, with L = log C + s_b)
                prefix_part_of_block[b] = sum_of_prefix_weights_times_logs +
                                          shift_of_block[b] * prefix_values.segment(
                                                first_row_of_the_block,
                                                end_row_of_the_block - first_row_of_the_block).sum();
                sum_of_prefix_weights_over_running_sums_of_block[b] = sum_of_prefix_weights_over_running_sums;

          });
          for (Eigen::Index b = 0; b < n_blocks; ++b) value += prefix_part_of_block[b];
          if constexpr (stan::is_constant_all<T_beta, T_scale, T_z, T_offset>::value) {
                return value + log_density_of_the_effects;
          }
          //// ---- (4) on the calling thread: each block's sum of prefix_weights / C over every row after it, in
          ////      its own units (the last block: none)
          std::vector<double> backward_sum_after_the_block(n_blocks, 0.0);
          for (Eigen::Index b = n_blocks - 1; b-- > 0;) {
                backward_sum_after_the_block[b] = std::exp(shift_of_block[b] - shift_of_block[b + 1]) *
                                                  (sum_of_prefix_weights_over_running_sums_of_block[b + 1] +
                                                   backward_sum_after_the_block[b + 1]);
          }
          //// ---- (5) each block from its last rows: reuse prefix_weights / C for its backward scan q,
          ////      then reuse exp(eta - s_b) for each row derivative and its product with the rows of x
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
                            buffers.prefix_weights_over_running_sums_of_the_rows.segment(first_row, m);
                      block.prefix_weights_over_running_sums.segment(m, m_padded - m).setZero();
                      nico::math::backward_affine_scan(block.prefix_weights_over_running_sums.data(), m_padded,
                                                          backward_running_sum_after_the_rows,
                                                          block.backward_running_sums.data(),
                                                          coefficients_of_one);
                      backward_running_sum_after_the_rows = block.backward_running_sums[0];
                      values_of_these_rows.array() = linear_values.segment(first_row, m).array() +
                            buffers.exponentials_of_the_rows.segment(first_row, m).array() *
                            block.backward_running_sums.head(m).array();
                      //// (one block, run on the calling thread: each derivative added to the sum of its group)
                      if constexpr (!stan::is_constant_all<T_scale, T_z>::value) {
                            if (n_blocks == 1) {
                                  for (Eigen::Index j = 0; j < m; ++j) {
                                        sums_of_each_group[index_of_each_row[first_row + j] - 1] +=
                                              values_of_these_rows[j];
                                  }
                            }
                      }
                      if constexpr (!stan::is_constant_all<T_beta>::value) {
                            partials_of_beta_of_block.col(b).noalias() +=
                                  x_ref.middleRows(first_row, m).transpose() * values_of_these_rows;
                      }

                }
                if (b == 0) running_derivative_of_the_first_row = backward_running_sum_after_the_rows;

          });
          for (Eigen::Index b = 0; b < n_blocks; ++b) partials_of_beta += partials_of_beta_of_block.col(b);
    }
    //// ---- (6) with more than one block: the sums of the derivatives over the rows of each group, one block of
    ////      1024 positions at a time (over ranges of the groups in parallel)
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
          if (sums_of_each_group_are_added_where_the_derivatives_are_made) {
                //// (nothing more: each derivative was added to the sum of its group where it was made)
          } else if (n_ranges == 1) {
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
          if constexpr (with_standard_normal_effects) {
                stan::math::partials<2>(partials) =
                      scale_value * sums_of_each_group - buffers.effects_with_a_leading_zero.segment(1, J);
          } else {
                stan::math::partials<2>(partials) = scale_value * sums_of_each_group;
          }
    }
    if constexpr (!stan::is_constant_all<T_offset>::value) {
          stan::math::partials<3>(partials) = factor_of_offset * running_derivative_of_the_first_row;
    }
    return partials.build(value + log_density_of_the_effects);

}

template <bool with_standard_normal_effects, typename T_linear_weights, typename T_prefix_weights, typename T_x,
          typename T_beta, typename T_scale, typename T_z, typename T_offset>
inline stan::return_type_t<T_beta, T_scale, T_z, T_offset>
affine_prefix_with_checked_indices::raw(
        const T_linear_weights& linear_weights,
        const T_prefix_weights& prefix_weights,
        const T_x& x,
        const T_beta& beta,
        const T_scale& scale,
        const std::vector<T_z>& z,
        const std::vector<int>& index,
        const std::vector<int>& rows_in_order_of_index,
        const std::vector<int>& last_position_of_each_index,
        const T_offset& offset,
        const int& rows_per_block_run_in_parallel,
        std::ostream* pstream__) {

    return evaluate_impl<true, with_standard_normal_effects>(
          linear_weights, prefix_weights, x, beta, scale, z, index, rows_in_order_of_index,
          last_position_of_each_index, offset, rows_per_block_run_in_parallel, pstream__);

}

template <bool with_standard_normal_effects, typename T_linear_weights, typename T_prefix_weights, typename T_x,
          typename T_beta, typename T_scale, typename T_z, typename T_offset>
inline stan::return_type_t<T_beta, T_scale, T_z, T_offset>
affine_prefix_with_checked_indices::checked(
        const T_linear_weights& linear_weights,
        const T_prefix_weights& prefix_weights,
        const T_x& x,
        const T_beta& beta,
        const T_scale& scale,
        const std::vector<T_z>& z,
        const checked_indices_of_affine_predictor_with_one_vector& indices,
        const T_offset& offset,
        const int& rows_per_block_run_in_parallel,
        std::ostream* pstream__) {

    return evaluate_impl<false, with_standard_normal_effects>(
          linear_weights, prefix_weights, x, beta, scale, z, indices.index(), indices.rows(), indices.last(),
          offset, rows_per_block_run_in_parallel, pstream__);

}

//// (legacy callers retain the original validation and default stream)
template <typename T_linear_weights, typename T_prefix_weights, typename T_x, typename T_beta, typename T_scale,
          typename T_z, typename T_offset>
inline stan::return_type_t<T_beta, T_scale, T_z, T_offset>
nico_dot_products_of_affine_predictor_with_indexed_effects_and_log_cumulative_sum_exp_with_one_vector_of_N_values(
        const T_linear_weights& linear_weights,
        const T_prefix_weights& prefix_weights,
        const T_x& x,
        const T_beta& beta,
        const T_scale& scale,
        const std::vector<T_z>& z,
        const std::vector<int>& index,
        const std::vector<int>& rows_in_order_of_index,
        const std::vector<int>& last_position_of_each_index,
        const T_offset& offset,
        const int& rows_per_block_run_in_parallel,
        std::ostream* pstream__ = nullptr) {

    return affine_prefix_with_checked_indices::raw<false>(
          linear_weights, prefix_weights, x, beta, scale, z, index, rows_in_order_of_index,
          last_position_of_each_index, offset, rows_per_block_run_in_parallel, pstream__);

}

//// (the generated model passes its owned checked data, not raw arrays or a validation flag)
template <typename T_linear_weights, typename T_prefix_weights, typename T_x, typename T_beta, typename T_scale,
          typename T_z, typename T_offset>
inline stan::return_type_t<T_beta, T_scale, T_z, T_offset>
nico_dot_products_of_affine_predictor_with_indexed_effects_and_log_cumulative_sum_exp_with_one_vector_of_N_values(
        const T_linear_weights& linear_weights,
        const T_prefix_weights& prefix_weights,
        const T_x& x,
        const T_beta& beta,
        const T_scale& scale,
        const std::vector<T_z>& z,
        const checked_indices_of_affine_predictor_with_one_vector& indices,
        const T_offset& offset,
        const int& rows_per_block_run_in_parallel,
        std::ostream* pstream__) {

    return affine_prefix_with_checked_indices::checked<false>(
          linear_weights, prefix_weights, x, beta, scale, z, indices, offset,
          rows_per_block_run_in_parallel, pstream__);

}

//// (the same plus the standard normal log density of the effects, -z'z / 2 without its constant: one pass
////  over z for the value and the derivatives of both; the raw arrays, validated at each call)
template <typename T_linear_weights, typename T_prefix_weights, typename T_x, typename T_beta, typename T_scale,
          typename T_z, typename T_offset>
inline stan::return_type_t<T_beta, T_scale, T_z, T_offset>
nico_dot_products_of_affine_predictor_with_standard_normal_indexed_effects_and_log_cumulative_sum_exp_with_one_vector_of_N_values(
        const T_linear_weights& linear_weights,
        const T_prefix_weights& prefix_weights,
        const T_x& x,
        const T_beta& beta,
        const T_scale& scale,
        const std::vector<T_z>& z,
        const std::vector<int>& index,
        const std::vector<int>& rows_in_order_of_index,
        const std::vector<int>& last_position_of_each_index,
        const T_offset& offset,
        const int& rows_per_block_run_in_parallel,
        std::ostream* pstream__ = nullptr) {

    return affine_prefix_with_checked_indices::raw<true>(
          linear_weights, prefix_weights, x, beta, scale, z, index, rows_in_order_of_index,
          last_position_of_each_index, offset, rows_per_block_run_in_parallel, pstream__);

}

//// (the same plus the standard normal log density of the effects, with the owned checked data)
template <typename T_linear_weights, typename T_prefix_weights, typename T_x, typename T_beta, typename T_scale,
          typename T_z, typename T_offset>
inline stan::return_type_t<T_beta, T_scale, T_z, T_offset>
nico_dot_products_of_affine_predictor_with_standard_normal_indexed_effects_and_log_cumulative_sum_exp_with_one_vector_of_N_values(
        const T_linear_weights& linear_weights,
        const T_prefix_weights& prefix_weights,
        const T_x& x,
        const T_beta& beta,
        const T_scale& scale,
        const std::vector<T_z>& z,
        const checked_indices_of_affine_predictor_with_one_vector& indices,
        const T_offset& offset,
        const int& rows_per_block_run_in_parallel,
        std::ostream* pstream__) {

    return affine_prefix_with_checked_indices::checked<true>(
          linear_weights, prefix_weights, x, beta, scale, z, indices, offset,
          rows_per_block_run_in_parallel, pstream__);

}

}  // namespace nico::math

using nico::math::nico_dot_products_of_affine_predictor_with_indexed_effects_and_log_cumulative_sum_exp_with_one_vector_of_N_values;
using nico::math::nico_dot_products_of_affine_predictor_with_standard_normal_indexed_effects_and_log_cumulative_sum_exp_with_one_vector_of_N_values;
