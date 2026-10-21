#pragma once
//
// -| --------- BayesMVP log_cumulative_sum_exp, a Stan external function -------------------------------------------
//
//     nico_log_cumulative_sum_exp(x, log_offset)[i] = log(exp(log_offset) + sum_{j <= i} exp(x_j))
//
// with x a vector and log_offset a real (negative infinity: no offset). A general function, the running version of
// log_sum_exp (nico_log_cumulative_sum_exp(x, negative_infinity())[n] = log_sum_exp(x)), as Stan's
// log(cumulative_sum(exp(x))) but without its overflow and with one autodiff step for the whole vector (Stan's
// version: one node per element of x - shift, of exp, of cumulative_sum and of log), e.g. for the risk sets of
// any survival model (rows in order of decreasing time) and for any running total on the log scale.
//
// With S_i = exp(log_offset) + sum_{j <= i} exp(x_j) and L_i = log(S_i), the adjoints of the result, g_i, give
//
//     adjoint of x_j          += (exp(x_j) / S_j) r_j,        r_j = g_j + (S_j / S_{j+1}) r_{j+1}
//     adjoint of log_offset   += (exp(log_offset) / S_1) r_1
//
// (every recurrence factor is at most 1). The values: shift = max(max(x),
// log_offset), t_j = exp(x_j - shift) (BayesMVP's exp kernel), C_i = exp(log_offset - shift) + sum_{j <= i} t_j,
// L_i = shift + log(C_i) (BayesMVP's log kernel), S_j / S_{j+1} = C_j / C_{j+1}, exp(x_j) / S_j = t_j / C_j; when
// C_1 is under 1e-280 (x_1 far below max(x)), a SIMD scan retains each prefix maximum and scaled sum.
//
// Declared in the Stan functions block (no body; stanc --allow-undefined):
//
//     vector nico_log_cumulative_sum_exp(vector x, real log_offset);
//
// Included at the end of stan_external_functions.hpp; its helpers are in their own namespace,
// nico::math::log_cumulative_sum_exp.
//
// -----------------------------------------------------------------------------------------------------------

#include "stan_external_functions.hpp"
#include "stan_external_simd_log_cumulative_sum_exp_scans.hpp"

#include <stan/math/rev.hpp>
#include <stan/math/prim.hpp>

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <limits>
#include <ostream>

namespace nico::math {
namespace log_cumulative_sum_exp {

// ---- the values L_i, and the factors of the backward recursion: ratio_j = S_j / S_{j+1} (j < M - 1) and
//      share_j = exp(x_j) / S_j; returns exp(log_offset) / S_1
// ---- the values L_i only (every input data: no factors of the backward recursion are needed): the same shift
//      and running sum as values_and_factors, nothing else (the doubles only: it computes no derivatives)
inline void values_only(const double* x, const int M, const double log_offset, double* L) {

    if (M == 0) return;
    const Eigen::Map<const Eigen::Matrix<double, -1, 1>> input(x, M);
    Eigen::Map<Eigen::Matrix<double, -1, 1>> output(L, M);
    if (std::isnan(log_offset) || log_offset == std::numeric_limits<double>::infinity()) {
        output.setConstant(log_offset);
        return;
    }
    const double shift = std::max(log_offset, input.maxCoeff());
    output.array() = input.array() - shift;
    nico_exp_of_nonpositive_prefix_arguments(L, M);
    nico_sum_prefix_scan(L, M, std::exp(log_offset - shift));
    if (L[0] >= 1e-280) {
        nico_log_of_positive_prefix_sums(L, M);
        output.array() += shift;
        return;
    }
    //// The extreme-input path is a log-domain SIMD prefix scan, not a scalar row recurrence.
    nico_log_prefix_scan(x, M, log_offset, L);

}

inline double values_and_factors(const double* x, const int M, const double log_offset, double* L,
                                 double* ratio, double* share) {

    if (M == 0) return 0.0;
    const Eigen::Map<const Eigen::Matrix<double, -1, 1>> input(x, M);
    Eigen::Map<Eigen::Matrix<double, -1, 1>> output(L, M);
    Eigen::Map<Eigen::Matrix<double, -1, 1>> shares(share, M);
    if (std::isnan(log_offset) || log_offset == std::numeric_limits<double>::infinity()) {
        output.setConstant(log_offset);
        shares.setConstant(std::isnan(log_offset) ? log_offset : 0.0);
        if (M > 1) Eigen::Map<Eigen::Matrix<double, -1, 1>>(ratio, M - 1).setConstant(
                       std::isnan(log_offset) ? log_offset : 1.0);
        return std::isnan(log_offset) ? log_offset : 1.0;
    }
    const double shift = std::max(log_offset, input.maxCoeff());
    shares.array() = input.array() - shift;
    nico_exp_of_nonpositive_prefix_arguments(share, M);
    const double term_of_offset = std::exp(log_offset - shift);
    output = shares;
    nico_sum_prefix_scan(L, M, term_of_offset);
    if (L[0] >= 1e-280) {
        //// Form factors before the scaled sums are replaced by their logarithms.
        if (M > 1) Eigen::Map<Eigen::Matrix<double, -1, 1>>(ratio, M - 1).array() =
                       output.head(M - 1).array() / output.tail(M - 1).array();
        const double share_of_offset = term_of_offset / L[0];
        shares.array() /= output.array();
        nico_log_of_positive_prefix_sums(L, M);
        output.array() += shift;
        return share_of_offset;
    }
    //// Temporarily hold the prefix maxima in L and their scaled sums in share. Form every ratio before
    //// those buffers become log values and shares; subtracting rounded log values would lose large prefixes.
    nico_log_prefix_pairs(x, M, log_offset, L, share);
    if (M > 1) {
        Eigen::Map<Eigen::Matrix<double, -1, 1>> ratios(ratio, M - 1);
        ratios = output.head(M - 1) - output.tail(M - 1);
        nico_exp_of_nonpositive_prefix_arguments(ratio, M - 1);
        ratios.array() *= shares.head(M - 1).array() / shares.tail(M - 1).array();
    }
    const double share_of_offset = std::exp(log_offset - L[0]) / share[0];
    for (int first = 0; first < M; first += BMVP_LANES) {
        const int lanes = std::min(BMVP_LANES, M - first);
        const simd_t maximum = nico_load_prefix_packet(L + first, lanes, 0.0);
        const simd_t sum = nico_load_prefix_packet(share + first, lanes, 1.0);
        const simd_t argument = nico_load_prefix_packet(x + first, lanes, 0.0);
        const simd_t log_sum = BMVP_KERNEL_NAME(nico_log_1_double_precision_wo_checks)(sum);
#if BMVP_LANES == 8
        const simd_t value = _mm512_add_pd(maximum, log_sum);
        const simd_t share_value = _mm512_div_pd(
              nico_exp_of_nonpositive_prefix_argument(_mm512_sub_pd(argument, maximum)), sum);
#else
        const simd_t value = _mm256_add_pd(maximum, log_sum);
        const simd_t share_value = _mm256_div_pd(
              nico_exp_of_nonpositive_prefix_argument(_mm256_sub_pd(argument, maximum)), sum);
#endif
        nico_store_prefix_packet(L + first, lanes, value);
        nico_store_prefix_packet(share + first, lanes, share_value);
    }
    return share_of_offset;

}

}  // namespace log_cumulative_sum_exp
}  // namespace nico::math




// ---- the Stan-facing function (stanc finds the nico_ name through its global using declaration)
namespace nico::math {

template <typename T_x, typename T_offset,
          stan::require_eigen_col_vector_t<T_x>* = nullptr,
          stan::require_stan_scalar_t<T_offset>* = nullptr>
inline Eigen::Matrix<stan::return_type_t<T_x, T_offset>, Eigen::Dynamic, 1> nico_log_cumulative_sum_exp(
        const T_x& x, const T_offset& log_offset, std::ostream* /*pstream__*/ = nullptr) {

    namespace lcse = nico::math::log_cumulative_sum_exp;
    using stan::arena_t;
    using stan::math::var;
    const auto& x_ref = stan::math::to_ref(x);
    const int M = static_cast<int>(x_ref.size());
    const Eigen::Matrix<double, -1, 1> x_values = stan::math::value_of(x_ref);
    const double log_offset_value = stan::math::value_of(log_offset);
    if (!x_values.allFinite()) stan::math::check_finite("nico_log_cumulative_sum_exp", "x", x_values);
    if constexpr (stan::is_constant_all<T_x, T_offset>::value) {
        Eigen::Matrix<double, -1, 1> L(M);
        if (M > 0) lcse::values_only(x_values.data(), M, log_offset_value, L.data());
        return L;
    } else {
        arena_t<Eigen::Matrix<double, -1, 1>> L(M);
        arena_t<Eigen::Matrix<double, -1, 1>> ratio(std::max(M - 1, 1));
        arena_t<Eigen::Matrix<double, -1, 1>> share(M);
        double share_of_offset = 0.0;
        if (M > 0) share_of_offset = lcse::values_and_factors(x_values.data(), M, log_offset_value, L.data(),
                                                              ratio.data(), share.data());
        arena_t<Eigen::Matrix<var, Eigen::Dynamic, 1>> x_arena = x_ref;
        var log_offset_var = log_offset;
        arena_t<Eigen::Matrix<var, Eigen::Dynamic, 1>> result = L;
        stan::math::reverse_pass_callback([x_arena, log_offset_var, result, ratio, share, share_of_offset,
                                           M]() mutable {
            Eigen::Matrix<double, -1, 1> running(M);
            const Eigen::Matrix<double, -1, 1> weights = result.adj();
            const double first_running = lcse::nico_backward_prefix_scan(
                  weights.data(), ratio.data(), M, running.data());
            for (int first = 0; first < M; first += BMVP_LANES) {
                const int lanes = std::min(BMVP_LANES, M - first);
                double* adjoints[BMVP_LANES] = {};
                for (int lane = 0; lane < lanes; ++lane) {
                    adjoints[lane] = &x_arena.coeffRef(first + lane).adj();
                }
                lcse::nico_add_prefix_adjoint_packet(adjoints, lanes,
                      lcse::nico_load_prefix_packet(share.data() + first, lanes, 0.0),
                      lcse::nico_load_prefix_packet(running.data() + first, lanes, 0.0));
            }
            log_offset_var.adj() += share_of_offset * first_running;
        });
        return Eigen::Matrix<var, Eigen::Dynamic, 1>(result);
    }

}

}  // namespace nico::math

using nico::math::nico_log_cumulative_sum_exp;

// ---- dot_product(w, log_cumulative_sum_exp(x, log_offset)): the same values and factors, one autodiff node
//      (operands: w, x and log_offset, those that are parameters) and no vector of results (the backward recursion
//      with g = w; d / d w_i = L_i)
//
//     real nico_dot_product_log_cumulative_sum_exp(vector w, vector x, real log_offset);
//
namespace nico::math {

template <typename T_w, typename T_x, typename T_offset,
          stan::require_eigen_col_vector_t<T_w>* = nullptr,
          stan::require_eigen_col_vector_t<T_x>* = nullptr,
          stan::require_stan_scalar_t<T_offset>* = nullptr>
inline stan::return_type_t<T_w, T_x, T_offset> nico_dot_product_log_cumulative_sum_exp(
        const T_w& w, const T_x& x, const T_offset& log_offset, std::ostream* /*pstream__*/ = nullptr) {

    namespace lcse = nico::math::log_cumulative_sum_exp;
    static const char* function_name = "nico_dot_product_log_cumulative_sum_exp";
    const auto& w_ref = stan::math::to_ref(w);
    const auto& x_ref = stan::math::to_ref(x);
    const int M = static_cast<int>(x_ref.size());
    stan::math::check_consistent_sizes(function_name, "w", w_ref, "x", x_ref);
    if (M == 0) return 0.0;
    const Eigen::Matrix<double, -1, 1> w_values = stan::math::value_of(w_ref);
    const Eigen::Matrix<double, -1, 1> x_values = stan::math::value_of(x_ref);
    if (!x_values.allFinite()) stan::math::check_finite(function_name, "x", x_values);
    auto& buffers = nico::math::scratch();
    buffers.ensure(M);
    double* L = buffers.a.data();
    double* ratio = buffers.b.data();
    double* share = buffers.c.data();
    if constexpr (stan::is_constant_all<T_w, T_x, T_offset>::value) {
        // (every input data: the values only, no factors, no backward recursion)
        lcse::values_only(x_values.data(), M, stan::math::value_of(log_offset), L);
        return w_values.dot(Eigen::Map<const Eigen::Matrix<double, -1, 1>>(L, M));
    } else {
        const double share_of_offset = lcse::values_and_factors(
              x_values.data(), M, stan::math::value_of(log_offset), L, ratio, share);
        const double value = w_values.dot(Eigen::Map<const Eigen::Matrix<double, -1, 1>>(L, M));
        Eigen::Matrix<double, -1, 1> partials_of_x(M);
        const double r = lcse::nico_backward_prefix_scan(w_values.data(), ratio, M, partials_of_x.data());
        partials_of_x.array() *= Eigen::Map<const Eigen::Matrix<double, -1, 1>>(share, M).array();
        auto ops_partials = stan::math::make_partials_propagator(w_ref, x_ref, log_offset);
        if constexpr (!stan::is_constant_all<T_w>::value) {
            stan::math::partials<0>(ops_partials) = Eigen::Map<const Eigen::Matrix<double, -1, 1>>(L, M);
        }
        if constexpr (!stan::is_constant_all<T_x>::value) stan::math::partials<1>(ops_partials) = partials_of_x;
        if constexpr (!stan::is_constant_all<T_offset>::value) {
            stan::math::partials<2>(ops_partials) = share_of_offset * r;
        }
        return ops_partials.build(value);
    }

}

}  // namespace nico::math

using nico::math::nico_dot_product_log_cumulative_sum_exp;






















