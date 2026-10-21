#pragma once
//
// -| --------- NicoSIMD binomial counts above successive cutpoints of a cumulative link, a Stan external fn ----
//
//     nico_binomial_counts_above_successive_cutpoints_of_a_cumulative_link_lpmf(
//           n | n_previous, latent, is_first, link, bound)
//       = sum_k binomial_lpmf(n_k | n_previous_k, p_k),
//
//     S_k  = F(latent_k)       (F = Phi when link = 1, inv_logit otherwise: the probability above cutpoint k),
//     S0_k = 1 when is_first_k = 1 (the first reported cutpoint of its group), S_(k-1) otherwise,
//     p_k  = fmax(bound, fmin(1 - bound, S_k / S0_k))      (Stan's fmax and fmin, NaN included),
//
// with n and n_previous arrays of int (data: the count above cutpoint k, and the count above the group's previous
// reported cutpoint or the group's total), latent a vector (the location of the group minus cutpoint k) and
// is_first a vector of 0 and 1 (data). The probabilities above successive cutpoints of one group and its counts
// above them: the conditional binomial (continuation) form of the multinomial over its categories. A general
// function: any cumulative-link (probit or logit) ordinal model of counts reported at some or all of the
// cutpoints of each group (ordinal meta-analysis of test accuracy, interval-censored grouped data, ordinal
// regression with aggregated counts per category). The same values and derivatives as the Stan statements
//
//     vector[K] S  = (link == 1) ? Phi(latent) : inv_logit(latent);
//     vector[K] S0 = append_row(1.0, S[1:(K - 1)]) .* (1 - is_first) + is_first;
//     target += binomial_lpmf(n | n_previous, fmax(bound, fmin(1 - bound, S ./ S0)));
//
// with every constant (log choose(n_previous_k, n_k)) unless propto, as binomial_lpmf, in ONE autodiff node in
// place of about eight: the forward pass and the reverse pass over all K elements, with BayesMVP's SIMD kernels
// for Phi, inv_logit, exp, log and log1m (the same kernels as nico_Phi, nico_inv_logit and nico_binomial_lpmf,
// so the same values), and the element-wise arithmetic in branch-free loops over the elements (each element
// reads only its own values and its predecessor's S, so there is no recurrence and the compiler vectorises them).
// The counts are checked as in nico_binomial_lpmf: Stan's checks run only when the pass over the values finds a
// bad one.
//
// Declared in the Stan functions block (no body; stanc --allow-undefined) as
//
//     real nico_binomial_counts_above_successive_cutpoints_of_a_cumulative_link_lpmf(
//           array[] int n, array[] int n_previous, vector latent, data vector is_first, data int link,
//           data real bound);
//
// It uses the namespace nico::math, the kernels of stan_external_functions.hpp and the table of log k! of
// stan_external_binomial_lpmf.hpp, which it includes (#pragma once). Its helpers are in their own namespace,
// nico::math::binomial_counts_above_successive_cutpoints.
//
// -----------------------------------------------------------------------------------------------------------

#include "stan_external_functions.hpp"
#include "stan_external_binomial_lpmf.hpp"

#include <stan/math/rev.hpp>
#include <stan/math/prim.hpp>

#include <Eigen/Dense>
#include <cmath>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace nico::math {
namespace binomial_counts_above_successive_cutpoints {

static constexpr const char* function_name =
    "nico_binomial_counts_above_successive_cutpoints_of_a_cumulative_link_lpmf";




// ---- the forward pass over K elements, into buffers of K doubles each:
//        S: F(latent); p: the bounded conditional probabilities; log_p, log1m_p: their logs;
//        p_passes: 1 where p_k is S_k / S0_k (the bounds of fmax and fmin not taken, not NaN), 0 otherwise.
//      Returns false, and nothing else, when a count is out of range (the caller then runs Stan's checks).
template <bool include_log_choose>
inline bool forward_pass(const int* n, const int* n_previous, const double* latent, const double* is_first,
                         const int link, const double bound, const int K, double* S, double* p,
                         double* log_p, double* log1m_p, double* p_passes, double& sum) {

    int n_bad = 0;
    for (int k = 0; k < K; ++k) {
        n_bad += (n[k] < 0) | (n[k] > n_previous[k]);
    }
    if (n_bad > 0) return false;
    copy_n(latent, K, S);
    if (link == 1) {
        k_Phi(S, K);
    } else {
        k_inv_logit(S, K);
    }
    const double upper = 1.0 - bound;
    for (int k = 0; k < K; ++k) {
        //// (S0: 1 at the first reported cutpoint of its group, the previous S otherwise, as
        ////  append_row(1.0, S[1:(K - 1)]) .* (1 - is_first) + is_first)
        const double S_previous = (k == 0) ? 1.0 : S[k - 1];
        const double S0 = S_previous * (1.0 - is_first[k]) + is_first[k];
        const double ratio = S[k] / S0;
        //// (Stan's fmin(1 - bound, ratio): ratio when ratio < 1 - bound, else the bound, NaN giving the bound;
        ////  then fmax(bound, .): the bound when bound > value, else the value)
        const double after_fmin = (ratio < upper) ? ratio : upper;
        const double after_fmax = (bound > after_fmin) ? bound : after_fmin;
        p[k] = after_fmax;
        p_passes[k] = ((ratio < upper) && !(bound > ratio)) ? 1.0 : 0.0;
        log_p[k] = after_fmax;
        log1m_p[k] = after_fmax;
    }
    k_log(log_p, K);
    k_log1m(log1m_p, K);
    double total = 0.0;
    for (int k = 0; k < K; ++k) {
        const double successes = n[k];
        const double failures = n_previous[k] - n[k];
        total += ((successes > 0.0) ? successes * log_p[k] : 0.0)
                 + ((failures > 0.0) ? failures * log1m_p[k] : 0.0);
    }
    if constexpr (include_log_choose) {
        const double* table = binomial::table_of_log_factorials();
        for (int k = 0; k < K; ++k) {
            total += binomial::log_factorial(n_previous[k], table) - binomial::log_factorial(n[k], table)
                     - binomial::log_factorial(n_previous[k] - n[k], table);
        }
    }
    sum = total;
    return true;

}




// ---- the reverse pass: d / d latent_k, from the forward pass's buffers (adjoint: the node's adjoint). The
//      chain of the Stan statements above: d lp / d p_k = n_k / p_k - (n_previous_k - n_k) / (1 - p_k) where p_k
//      passes the bounds (0 otherwise); p_k = S_k / S0_k gives S_k the share 1 / S0_k and, where S0_k is S_(k-1),
//      S_(k-1) the share - S_k / S0_k^2 (as Stan's division: - adjoint * (S_k / S0_k) / S0_k); then
//      d S_k / d latent_k = phi(latent_k) (probit) or S_k (1 - S_k) (logit).
inline void reverse_pass(const int* n, const int* n_previous, const double* latent, const double* is_first,
                         const int link, const int K, const double* S, const double* p, const double* p_passes,
                         const double adjoint, double* d_S, double* d_latent) {

    for (int k = 0; k < K; ++k) d_S[k] = 0.0;
    for (int k = 0; k < K; ++k) {
        const double successes = n[k];
        const double failures = n_previous[k] - n[k];
        const double d_p = adjoint * p_passes[k]
                           * (((successes > 0.0) ? successes / p[k] : 0.0)
                              - ((failures > 0.0) ? failures / (1.0 - p[k]) : 0.0));
        const double S_previous = (k == 0) ? 1.0 : S[k - 1];
        const double S0 = S_previous * (1.0 - is_first[k]) + is_first[k];
        d_S[k] += d_p / S0;
        if (k > 0) d_S[k - 1] -= d_p * (S[k] / S0) / S0 * (1.0 - is_first[k]);
    }
    if (link == 1) {
        //// (phi(latent) = exp(-latent^2 / 2) / sqrt(2 pi), with the exp kernel, as nico_Phi's derivative)
        for (int k = 0; k < K; ++k) d_latent[k] = -0.5 * latent[k] * latent[k];
        k_exp(d_latent, K);
        for (int k = 0; k < K; ++k) d_latent[k] *= INV_SQRT_2PI * d_S[k];
    } else {
        for (int k = 0; k < K; ++k) d_latent[k] = d_S[k] * S[k] * (1.0 - S[k]);
    }

}




// ---- Stan's checks of the counts (they throw; reached only when the pass over the values found a bad one)
template <typename T_n, typename T_n_previous>
inline void run_the_checks_of_Stan(const T_n& n, const T_n_previous& n_previous) {

    stan::math::check_nonnegative(function_name, "Population size parameter", n_previous);
    stan::math::check_bounded(function_name, "Successes variable", n, 0, n_previous);
    throw std::domain_error(std::string(function_name) + ": an argument is out of its range");

}

}  // namespace binomial_counts_above_successive_cutpoints
}  // namespace nico::math




// ---- the Stan-facing function (global scope; stanc calls it as
//      nico_binomial_counts_above_successive_cutpoints_of_a_cumulative_link_lpmf<propto__>(n, n_previous,
//      latent, is_first, link, bound, pstream__))
namespace nico::math {

template <bool propto, typename T_n, typename T_n_previous, typename T_latent, typename T_is_first,
          stan::require_all_std_vector_vt<std::is_integral, T_n, T_n_previous>* = nullptr,
          stan::require_eigen_col_vector_t<T_latent>* = nullptr,
          stan::require_eigen_col_vector_vt<std::is_arithmetic, T_is_first>* = nullptr>
inline stan::return_type_t<T_latent> nico_binomial_counts_above_successive_cutpoints_of_a_cumulative_link_lpmf(
        const T_n& n, const T_n_previous& n_previous, const T_latent& latent, const T_is_first& is_first,
        const int link, const double bound, std::ostream* /*pstream__*/) {

    namespace counts = nico::math::binomial_counts_above_successive_cutpoints;
    using stan::math::var;
    stan::math::check_consistent_sizes(counts::function_name, "Successes variable", n,
                                       "Population size parameter", n_previous, "Latent values", latent,
                                       "Indicators of the first cutpoints", is_first);
    const int K = static_cast<int>(n.size());
    if (K == 0) return 0.0;
    const auto& latent_ref = stan::math::to_ref(latent);
    const Eigen::Matrix<double, -1, 1> latent_values = stan::math::value_of(latent_ref);
    const Eigen::Matrix<double, -1, 1> is_first_values = stan::math::value_of(is_first);
    if constexpr (stan::is_constant_all<T_latent>::value) {
        std::vector<double> buffers(5 * static_cast<std::size_t>(K));
        double sum = 0.0;
        if (!counts::forward_pass<true>(n.data(), n_previous.data(), latent_values.data(), is_first_values.data(),
                                        link, bound, K, buffers.data(), buffers.data() + K,
                                        buffers.data() + 2 * K, buffers.data() + 3 * K, buffers.data() + 4 * K,
                                        sum)) {

            counts::run_the_checks_of_Stan(n, n_previous);

        }
        return propto ? 0.0 : sum;
    } else {
        auto& arena = stan::math::ChainableStack::instance_->memalloc_;
        //// (on the arena, kept for the reverse pass: the data, latent's values, S, p and where p passes)
        int* n_kept = arena.alloc_array<int>(K);
        int* n_previous_kept = arena.alloc_array<int>(K);
        double* latent_kept = arena.alloc_array<double>(K);
        double* is_first_kept = arena.alloc_array<double>(K);
        double* S = arena.alloc_array<double>(K);
        double* p = arena.alloc_array<double>(K);
        double* p_passes = arena.alloc_array<double>(K);
        double* log_p = arena.alloc_array<double>(K);
        double* log1m_p = arena.alloc_array<double>(K);
        for (int k = 0; k < K; ++k) {
            n_kept[k] = n[k];
            n_previous_kept[k] = n_previous[k];
            latent_kept[k] = latent_values(k);
            is_first_kept[k] = is_first_values(k);
        }
        double sum = 0.0;
        if (!counts::forward_pass<!propto>(n_kept, n_previous_kept, latent_kept, is_first_kept, link, bound, K,
                                           S, p, log_p, log1m_p, p_passes, sum)) {

            counts::run_the_checks_of_Stan(n, n_previous);

        }
        stan::arena_t<Eigen::Matrix<var, Eigen::Dynamic, 1>> latent_vars = latent_ref;
        //// (log_p and log1m_p are not needed after the forward pass: their arena space holds d_S and d_latent)
        return stan::math::make_callback_var(sum, [latent_vars, n_kept, n_previous_kept, latent_kept,
                                                   is_first_kept, link, K, S, p, p_passes, log_p,
                                                   log1m_p](auto& vi) mutable {
            double* d_S = log_p;
            double* d_latent = log1m_p;
            counts::reverse_pass(n_kept, n_previous_kept, latent_kept, is_first_kept, link, K, S, p, p_passes,
                                 vi.adj(), d_S, d_latent);
            for (int k = 0; k < K; ++k) latent_vars.coeffRef(k).adj() += d_latent[k];
        });
    }

}

}  // namespace nico::math

using nico::math::nico_binomial_counts_above_successive_cutpoints_of_a_cumulative_link_lpmf;
























