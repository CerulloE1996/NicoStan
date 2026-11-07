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

//// (the same values and factors, with the logs of the shifted-prefix path, C_1 at least 1e-280, left to
//// logs_of_the_sums(shift): L holds the scaled sums C_i when it is called, every ratio and share formed)
template <typename F>
inline double values_and_factors_then(const double* x, const int M, const double log_offset, double* L,
                                      double* ratio, double* share, F&& logs_of_the_sums) {

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
        // nico_log_of_positive_prefix_sums(L, M);
        // output.array() += shift;
        logs_of_the_sums(shift);
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
#if defined(NICO_SIMD_SCALAR_KERNELS)
        const simd_t value = maximum + std::log(sum);
        const simd_t share_value = nico_exp_of_nonpositive_prefix_argument(argument - maximum) / sum;
#else
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
#endif
        nico_store_prefix_packet(L + first, lanes, value);
        nico_store_prefix_packet(share + first, lanes, share_value);
    }
    return share_of_offset;

}

inline double values_and_factors(const double* x, const int M, const double log_offset, double* L,
                                 double* ratio, double* share) {

    return values_and_factors_then(x, M, log_offset, L, ratio, share, [L, M](const double shift) {
        nico_log_of_positive_prefix_sums(L, M);
        Eigen::Map<Eigen::Matrix<double, -1, 1>>(L, M).array() += shift;
    });

}

//// thread-local buffers of the dot products: the doubles of x and the weights (no allocation per call)
struct scratch_of_the_dot_products_t {
    std::vector<double> x, w;
    void ensure(int n) { if ((int)x.size() < n) { x.resize(n); w.resize(n); } }
};
inline scratch_of_the_dot_products_t& scratch_of_the_dot_products() {
    thread_local scratch_of_the_dot_products_t s;
    return s;
}

//// the scaled sums C_i of rows 1 to M - 1 whose weight is not 0, then row M, gathered in order to the front
//// of C (in place) with their weights in kept_weights; returns their number and the sum of every weight
inline int gather_the_sums_of_nonzero_weights(double* C, const double* w, const int M, double* kept_weights,
                                              double* sum_of_weights) {

    int kept = 0;
    double sum = 0.0;
#if !defined(NICO_SIMD_SCALAR_KERNELS) && BMVP_LANES == 8
    __m512d sums = _mm512_setzero_pd();
    for (int first = 0; first < M - 1; first += 8) {
        const int lanes = std::min(8, M - 1 - first);
        const __mmask8 valid = static_cast<__mmask8>((1u << lanes) - 1u);
        const __m512d sums_of_rows = _mm512_maskz_loadu_pd(valid, C + first);
        const __m512d weights = _mm512_maskz_loadu_pd(valid, w + first);
        const __mmask8 nonzero = _mm512_mask_cmp_pd_mask(valid, weights, _mm512_setzero_pd(), _CMP_NEQ_UQ);
        const int count = __builtin_popcount(static_cast<unsigned>(nonzero));
        const __mmask8 front = static_cast<__mmask8>((1u << count) - 1u);
        //// (the compress in the register, then a masked store: never past row first + lanes - 1)
        _mm512_mask_storeu_pd(C + kept, front, _mm512_maskz_compress_pd(nonzero, sums_of_rows));
        _mm512_mask_storeu_pd(kept_weights + kept, front, _mm512_maskz_compress_pd(nonzero, weights));
        sums = _mm512_add_pd(sums, weights);
        kept += count;
    }
    sum = _mm512_reduce_add_pd(sums);
#else
    for (int i = 0; i < M - 1; ++i) {
        C[kept] = C[i];
        kept_weights[kept] = w[i];
        sum += w[i];
        kept += (w[i] != 0.0);
    }
#endif
    C[kept] = C[M - 1];
    kept_weights[kept] = w[M - 1];
    *sum_of_weights = sum + w[M - 1];
    return kept + 1;

}

//// values_and_factors with data weights w, giving value = sum_i w_i L_i and last = L_M: on the shifted-prefix
//// path L_i = shift + log(C_i) and C_i is positive and finite, so w_i L_i is shift w_i wherever w_i is 0, and
//// the log is taken only of the C_i with w_i not 0 (and of C_M); on the other paths L holds every L_i
inline double values_and_factors_of_data_weights(const double* x, const double* w, const int M,
                                                 const double log_offset, double* L, double* ratio,
                                                 double* share, double* kept_weights, double* value,
                                                 double* last) {

    if (M == 0) {
        *value = 0.0;
        *last = log_offset;
        return 0.0;
    }
    bool only_the_kept_sums = false;
    const double share_of_offset = values_and_factors_then(x, M, log_offset, L, ratio, share,
          [&](const double shift) {
              double sum_of_weights = 0.0;
              const int kept = gather_the_sums_of_nonzero_weights(L, w, M, kept_weights, &sum_of_weights);
              nico_log_of_positive_prefix_sums(L, kept);
              *value = shift * sum_of_weights + Eigen::Map<const Eigen::Matrix<double, -1, 1>>(kept_weights,
                                                      kept).dot(Eigen::Map<const Eigen::Matrix<double, -1, 1>>(L,
                                                                                                         kept));
              *last = shift + L[kept - 1];
              only_the_kept_sums = true;
          });
    if (!only_the_kept_sums) {
        *value = Eigen::Map<const Eigen::Matrix<double, -1, 1>>(w, M).dot(
              Eigen::Map<const Eigen::Matrix<double, -1, 1>>(L, M));
        *last = L[M - 1];
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
//      with g = w; d / d w_i = L_i); with data w, the log is taken only of the prefix sums whose weight is not 0
//
//     real nico_dot_prod_log_cumul_sum_exp(vector w, vector x, real log_offset);
//
namespace nico::math {

template <typename T_w, typename T_x, typename T_offset,
          stan::require_eigen_col_vector_t<T_w>* = nullptr,
          stan::require_eigen_col_vector_t<T_x>* = nullptr,
          stan::require_stan_scalar_t<T_offset>* = nullptr>
inline stan::return_type_t<T_w, T_x, T_offset> nico_dot_prod_log_cumul_sum_exp(
        const T_w& w, const T_x& x, const T_offset& log_offset, std::ostream* /*pstream__*/ = nullptr) {

    namespace lcse = nico::math::log_cumulative_sum_exp;
    static const char* function_name = "nico_dot_prod_log_cumul_sum_exp";
    const auto& w_ref = stan::math::to_ref(w);
    const auto& x_ref = stan::math::to_ref(x);
    const int M = static_cast<int>(x_ref.size());
    stan::math::check_consistent_sizes(function_name, "w", w_ref, "x", x_ref);
    if (M == 0) return 0.0;
    // const Eigen::Matrix<double, -1, 1> w_values = stan::math::value_of(w_ref);
    // const Eigen::Matrix<double, -1, 1> x_values = stan::math::value_of(x_ref);
    //// (the doubles of data w and x read where they are; those of a var into thread-local buffers)
    auto& own = lcse::scratch_of_the_dot_products();
    own.ensure(M);
    const double* w_values = nullptr;
    const double* x_values = nullptr;
    if constexpr (stan::is_constant_all<T_w>::value) {
        w_values = w_ref.data();
    } else {
        for (int i = 0; i < M; ++i) own.w[i] = stan::math::value_of(w_ref.coeff(i));
        w_values = own.w.data();
    }
    if constexpr (stan::is_constant_all<T_x>::value) {
        x_values = x_ref.data();
    } else {
        for (int i = 0; i < M; ++i) own.x[i] = stan::math::value_of(x_ref.coeff(i));
        x_values = own.x.data();
    }
    const Eigen::Map<const Eigen::Matrix<double, -1, 1>> x_map(x_values, M);
    if (!x_map.allFinite()) stan::math::check_finite(function_name, "x", x_map);
    auto& buffers = nico::math::scratch();
    buffers.ensure(M);
    double* L = buffers.a.data();
    double* ratio = buffers.b.data();
    double* share = buffers.c.data();
    if constexpr (stan::is_constant_all<T_w, T_x, T_offset>::value) {
        // (every input data: the values only, no factors, no backward recursion)
        lcse::values_only(x_values, M, stan::math::value_of(log_offset), L);
        return Eigen::Map<const Eigen::Matrix<double, -1, 1>>(w_values, M).dot(
              Eigen::Map<const Eigen::Matrix<double, -1, 1>>(L, M));
    } else {
        // const double share_of_offset = lcse::values_and_factors(
        //       x_values.data(), M, stan::math::value_of(log_offset), L, ratio, share);
        // const double value = w_values.dot(Eigen::Map<const Eigen::Matrix<double, -1, 1>>(L, M));
        // Eigen::Matrix<double, -1, 1> partials_of_x(M);
        // const double r = lcse::nico_backward_prefix_scan(w_values.data(), ratio, M, partials_of_x.data());
        // partials_of_x.array() *= Eigen::Map<const Eigen::Matrix<double, -1, 1>>(share, M).array();
        //// (data w: the log only where w is not 0; a var w needs every L_i, its partials)
        double share_of_offset = 0.0;
        double value = 0.0;
        if constexpr (stan::is_constant_all<T_w>::value) {
            double last = 0.0;
            share_of_offset = lcse::values_and_factors_of_data_weights(
                  x_values, w_values, M, stan::math::value_of(log_offset), L, ratio, share, own.w.data(), &value,
                  &last);
        } else {
            share_of_offset = lcse::values_and_factors(x_values, M, stan::math::value_of(log_offset), L, ratio,
                                                       share);
            value = Eigen::Map<const Eigen::Matrix<double, -1, 1>>(w_values, M).dot(
                  Eigen::Map<const Eigen::Matrix<double, -1, 1>>(L, M));
        }
        auto ops_partials = stan::math::make_partials_propagator(w_ref, x_ref, log_offset);
        //// (the backward recursion straight into the partials of x, on the tape; into own.x when x is data)
        double r = 0.0;
        if constexpr (!stan::is_constant_all<T_x>::value) {
            double* partials_of_x = stan::math::partials<1>(ops_partials).data();
            r = lcse::nico_backward_prefix_scan(w_values, ratio, M, partials_of_x);
            Eigen::Map<Eigen::Matrix<double, -1, 1>>(partials_of_x, M).array() *=
                  Eigen::Map<const Eigen::Matrix<double, -1, 1>>(share, M).array();
        } else if constexpr (!stan::is_constant_all<T_offset>::value) {
            r = lcse::nico_backward_prefix_scan(w_values, ratio, M, own.x.data());
        }
        if constexpr (!stan::is_constant_all<T_w>::value) {
            stan::math::partials<0>(ops_partials) = Eigen::Map<const Eigen::Matrix<double, -1, 1>>(L, M);
        }
        // if constexpr (!stan::is_constant_all<T_x>::value)
        //     stan::math::partials<1>(ops_partials) = partials_of_x;
        if constexpr (!stan::is_constant_all<T_offset>::value) {
            stan::math::partials<2>(ops_partials) = share_of_offset * r;
        }
        return ops_partials.build(value);
    }

}

}  // namespace nico::math

using nico::math::nico_dot_prod_log_cumul_sum_exp;






















