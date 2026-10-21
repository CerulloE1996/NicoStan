#pragma once
//
// -| --------- NicoSIMD induced-Dirichlet prior of ordered cutpoints, a Stan external function ----------------
//
//     nico_induced_dirichlet_prior_of_ordered_cutpoints_lpdf(C | alpha, link)
//       = J(C) + dirichlet_lpdf(p | alpha),
//     nico_induced_dirichlet_prior_of_ordered_cutpoints_lpdf(C | alpha, link, location)
//       = the same at the latent location: sum over the rows r of C of J(C_r - location) +
//         dirichlet_lpdf(p_r | alpha), with p_r the category probabilities of C_r - location,
//
// with, for each set of K ordered cutpoints c (a vector C, or each row of a matrix C: one set per group, e.g.
// the random cutpoints of each study),
//
//     F_k = Phi(c_k) when link = 1, inv_logit(c_k) otherwise          (k = 1, ..., K),
//     p   = (F_1, F_2 - F_1, ..., F_K - F_(K-1), 1 - F_K)             (the K + 1 category probabilities),
//     J(c) = sum_k log f(c_k): std_normal_lpdf(c) when link = 1 (its -0.5 log(2 pi) per value left out with
//            propto), sum(log_inv_logit(c) + log1m_inv_logit(c)) otherwise (always kept),
//
// and alpha a vector of K + 1 Dirichlet weights, data (a prior) or parameters (a between-group model of the
// cutpoints, e.g. alpha = 0.01 + phi kappa). The induced Dirichlet prior (or between-group model) of the
// cutpoints of an ordinal model (Betancourt's): a Dirichlet on the category probabilities at a latent location
// (0 in the first form), with the log Jacobian of the map from the cutpoints to the cumulative probabilities,
// J. A general function: any cumulative-link (probit or logit) ordinal model with an induced Dirichlet on its
// cutpoints, fixed or random over groups (ordinal regression, ordinal meta-analysis and network meta-analysis of
// test accuracy, ordinal latent class models). The same values and derivatives as the Stan statements, for each
// set c of cutpoints,
//
//     vector[K] F = (link == 1) ? Phi(c - location) : inv_logit(c - location);
//     vector[K + 1] p = cumul_probs_to_ord_probs(F);
//     target += (link == 1) ? std_normal_lupdf(c - location)
//                           : sum(log_inv_logit(c - location) + log1m_inv_logit(c - location));
//     target += dirichlet_lupdf(p | alpha);
//
// (with propto; every constant, lgamma(sum(alpha)) - sum(lgamma(alpha)) of a data alpha and -0.5 log(2 pi) per
// value, without it, as Stan's _lpdf; the lgamma terms of a parameter alpha are always kept, as Stan's), in ONE
// autodiff node for all the sets in place of about K + 6 per set: BayesMVP's SIMD kernels for Phi, inv_logit,
// exp, log, log_inv_logit and log1m_inv_logit (the same kernels as nico_Phi, nico_inv_logit,
// nico_log_inv_logit and nico_log1m_inv_logit), and the element-wise arithmetic in branch-free loops (each
// category reads only its own cumulative probability and its predecessor's, so there is no recurrence).
// d / d c_k = (d lp / d F_k) f(c_k) + d J / d c_k, with d lp / d F_k = (alpha_k - 1) / p_k - (alpha_(k+1) - 1)
// / p_(k+1), f = phi or F (1 - F), and d J / d c_k = - c_k (probit) or 1 - 2 F_k (logit) (c at the location);
// d / d alpha_j = sum over the sets of log p_j, plus (number of sets) (digamma(sum(alpha)) - digamma(alpha_j))
// (the K + 1 digamma with Stan's scalar function, once per call).
//
// Declared in the Stan functions block (no body; stanc --allow-undefined) as any of
//
//     real nico_induced_dirichlet_prior_of_ordered_cutpoints_lpdf(vector C, data vector alpha, data int link);
//     real nico_induced_dirichlet_prior_of_ordered_cutpoints_lpdf(vector C, vector alpha, data int link,
//                                                                 data real location);
//     real nico_induced_dirichlet_prior_of_ordered_cutpoints_lpdf(row_vector C, vector alpha, data int link,
//                                                                 data real location);
//     real nico_induced_dirichlet_prior_of_ordered_cutpoints_lpdf(matrix C, vector alpha, data int link,
//                                                                 data real location);
//
// It uses the namespace nico::math and the kernels of stan_external_functions.hpp, which it includes
// (#pragma once). Its helpers are in their own namespace, nico::math::induced_dirichlet_prior.
//
// -----------------------------------------------------------------------------------------------------------

#include "stan_external_functions.hpp"

#include <stan/math/rev.hpp>
#include <stan/math/prim.hpp>

#include <Eigen/Dense>
#include <cmath>
#include <ostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace nico::math {
namespace induced_dirichlet_prior {

static constexpr const char* function_name = "nico_induced_dirichlet_prior_of_ordered_cutpoints_lpdf";
static constexpr double half_log_of_two_pi = 0.91893853320467274178;




// ---- the forward pass of one set of K cutpoints x (the cutpoints minus the location), into buffers: lF (K: log F(x),
//      log Phi(x) or log_inv_logit(x)), lS (K + 1, the first K used: log(1 - F(x)), from the same kernel at -x),
//      log_p (2 K + 1: the log of the K + 1 category probabilities, then K values of work space), J_terms (K: log
//      f(x_k) without the constant); its log probability, without the lgamma terms of alpha (added once by the
//      caller), into sum. Every category probability is computed on the LOG scale, with BayesMVP's SIMD kernels
//      (nico_log_Phi with its log-scale tails, log_inv_logit, log1m_exp), no clamps:
//          log p_1     = log F(x_1),      log p_(K+1) = log(1 - F(x_K)),
//          log p_k     = log F(x_k) + log1m_exp(log F(x_(k-1)) - log F(x_k))          (x_(k-1) < 0),
//                      = log S(x_(k-1)) + log1m_exp(log S(x_k) - log S(x_(k-1)))      (x_(k-1) >= 0, S = 1 - F),
//      so that a category probability far in a tail, or between two close cutpoints, is never rounded to 0 (as
//      Phi(x_k) - Phi(x_(k-1)) in double precision was, which threw and left the chain without a gradient); two
//      equal cutpoints give log p = -inf, i.e. a log density of -inf (a rejected proposal), never an error. A term
//      with alpha_j = 1 is 0 and is left out (0 x log p). Returns false, and nothing else, only when the cutpoints
//      are not in increasing order or a value is not a number (the caller then runs Stan's checks).
template <bool include_the_constant_of_the_normal>
inline bool forward_pass(const double* x, const double* alpha, const int link, const int K, double* lF,
                         double* lS, double* log_p, double* J_terms, double& sum) {

    copy_n(x, K, lF);
    for (int k = 0; k < K; ++k) lS[k] = -x[k];
    if (link == 1) {
        k_log_Phi(lF, K);
        k_log_Phi(lS, K);
    } else {
        k_log_inv_logit(lF, K);
        k_log_inv_logit(lS, K);
    }
    log_p[0] = lF[0];
    log_p[K] = lS[K - 1];
    double* work = log_p + (K + 1);
    for (int k = 1; k < K; ++k) {
        work[k - 1] = (x[k - 1] >= 0.0) ? (lS[k] - lS[k - 1]) : (lF[k - 1] - lF[k]);
        if (!(work[k - 1] <= 0.0)) return false;
    }
    if (K > 1) k_log1m_exp(work, K - 1);
    for (int k = 1; k < K; ++k) log_p[k] = ((x[k - 1] >= 0.0) ? lS[k - 1] : lF[k]) + work[k - 1];
    double total = 0.0;
    for (int j = 0; j <= K; ++j) {
        if (std::isnan(log_p[j])) return false;
        if (alpha[j] != 1.0) total += (alpha[j] - 1.0) * log_p[j];
    }
    //// (J: the log density of F at each cutpoint: -x^2 / 2 (probit), log F + log(1 - F) (logit))
    if (link == 1) {
        for (int k = 0; k < K; ++k) J_terms[k] = -0.5 * x[k] * x[k];
        if constexpr (include_the_constant_of_the_normal) total -= K * half_log_of_two_pi;
    } else {
        for (int k = 0; k < K; ++k) J_terms[k] = lF[k] + lS[k];
    }
    for (int k = 0; k < K; ++k) total += J_terms[k];
    sum = total;
    return true;

}




// ---- the reverse pass of one set: d / d x_k (adjoint: the node's adjoint), from the forward pass's buffers, with
//      f(x_k) / p_j = exp(log f(x_k) - log p_j) (the SIMD exp kernel), so that a tiny p_j never divides 0 by 0:
//          d / d x_k = (alpha_k - 1) f(x_k) / p_k - (alpha_(k+1) - 1) f(x_k) / p_(k+1) + d / d x_k log f(x_k).
inline void reverse_pass(const double* x, const double* alpha, const int link, const int K, const double* lF,
                         const double* lS, const double* log_p, const double adjoint, double* d_x) {

    std::vector<double> ratio_to_own_category(K), ratio_to_next_category(K);
    for (int k = 0; k < K; ++k) {
        const double log_f = (link == 1) ? (-0.5 * x[k] * x[k] - half_log_of_two_pi) : (lF[k] + lS[k]);
        ratio_to_own_category[k] = log_f - log_p[k];
        ratio_to_next_category[k] = log_f - log_p[k + 1];
    }
    k_exp(ratio_to_own_category.data(), K);
    k_exp(ratio_to_next_category.data(), K);
    for (int k = 0; k < K; ++k) {
        double d_F = 0.0;
        if (alpha[k] != 1.0) d_F += (alpha[k] - 1.0) * ratio_to_own_category[k];
        if (alpha[k + 1] != 1.0) d_F -= (alpha[k + 1] - 1.0) * ratio_to_next_category[k];
        const double d_J = (link == 1) ? -x[k] : (std::exp(lS[k]) - std::exp(lF[k]));
        d_x[k] = adjoint * (d_F + d_J);
    }

}




// ---- Stan's checks (they throw; reached only when a category probability was not in (0, 1])
template <typename T_p, typename T_alpha>
inline void run_the_checks_of_Stan(const T_p& p_values, const T_alpha& alpha) {

    stan::math::check_positive(function_name, "prior size parameter", alpha);
    stan::math::check_simplex(function_name, "probabilities", p_values);
    throw std::domain_error(std::string(function_name) + ": a category probability is not positive");

}

}  // namespace induced_dirichlet_prior
}  // namespace nico::math




// ---- the Stan-facing function, at a latent location (global scope; stanc calls it as
//      nico_induced_dirichlet_prior_of_ordered_cutpoints_lpdf<propto__>(C, alpha, link, location, pstream__)):
//      C a vector or row_vector (one set of cutpoints) or a matrix (one set per row); alpha data or parameters
namespace nico::math {

template <bool propto, typename T_C, typename T_alpha, stan::require_eigen_t<T_C>* = nullptr,
          stan::require_eigen_col_vector_t<T_alpha>* = nullptr>
inline stan::return_type_t<T_C, T_alpha> nico_induced_dirichlet_prior_of_ordered_cutpoints_lpdf(
        const T_C& C, const T_alpha& alpha, const int link, const double location, std::ostream* /*pstream__*/) {

    namespace prior = nico::math::induced_dirichlet_prior;
    using stan::math::var;
    constexpr bool C_is_one_set = stan::is_eigen_vector<T_C>::value;
    constexpr bool C_is_constant = stan::is_constant_all<T_C>::value;
    constexpr bool alpha_is_constant = stan::is_constant_all<T_alpha>::value;
    //// (the lgamma terms of alpha: left out with propto only when alpha is data, as Stan's dirichlet_lupdf)
    constexpr bool include_the_lgamma_terms = !propto || !alpha_is_constant;
    constexpr bool include_the_constant_of_the_normal = !propto;
    const int R = C_is_one_set ? 1 : static_cast<int>(C.rows());
    const int K = C_is_one_set ? static_cast<int>(C.size()) : static_cast<int>(C.cols());
    stan::math::check_size_match(prior::function_name, "Number of prior weights", alpha.size(),
                                 "Number of cutpoints plus 1", K + 1);
    if ((K == 0) || (R == 0)) return 0.0;
    if constexpr (propto && C_is_constant && alpha_is_constant) return 0.0;
    const auto& C_ref = stan::math::to_ref(C);
    const auto& alpha_ref = stan::math::to_ref(alpha);
    //// (the buffers: x = C - location (row by row), alpha, F (log F), p (log(1 - F)) and log p of every set, kept for the reverse
    ////  pass on the arena when there is one (J_terms for the forward pass only, its space reused for d_x); in a
    ////  std::vector when nothing is differentiated, so that no evaluation of the log density alone leaves
    ////  memory on the arena)
    const std::size_t RK = static_cast<std::size_t>(R) * K;
    const std::size_t n_buffer_values = 3 * RK + (K + 1) + static_cast<std::size_t>(R) * (K + 1) +
                                        static_cast<std::size_t>(R) * (2 * K + 1);
    std::vector<double> buffer_without_autodiff;
    double* buffers = nullptr;
    if constexpr (C_is_constant && alpha_is_constant) {
        buffer_without_autodiff.resize(n_buffer_values);
        buffers = buffer_without_autodiff.data();
    } else {
        buffers = stan::math::ChainableStack::instance_->memalloc_.alloc_array<double>(n_buffer_values);
    }
    double* x = buffers;
    double* alpha_kept = x + RK;
    double* F = alpha_kept + (K + 1);
    double* p = F + RK;
    double* log_p = p + static_cast<std::size_t>(R) * (K + 1);
    double* J_terms = log_p + static_cast<std::size_t>(R) * (2 * K + 1);
    for (int r = 0; r < R; ++r) {
        for (int k = 0; k < K; ++k) {
            const double value_of_C = C_is_one_set ? stan::math::value_of(C_ref.coeff(k))
                                                   : stan::math::value_of(C_ref.coeff(r, k));
            x[static_cast<std::size_t>(r) * K + k] = value_of_C - location;
        }
    }
    for (int j = 0; j <= K; ++j) alpha_kept[j] = stan::math::value_of(alpha_ref.coeff(j));
    double sum = 0.0;
    for (int r = 0; r < R; ++r) {
        double sum_of_set = 0.0;
        double* p_r = p + static_cast<std::size_t>(r) * (K + 1);
        const std::size_t start_of_set = static_cast<std::size_t>(r) * K;
        if (!prior::forward_pass<include_the_constant_of_the_normal>(
                    x + start_of_set, alpha_kept, link, K, F + start_of_set, p_r,
                    log_p + static_cast<std::size_t>(r) * (2 * K + 1), J_terms + start_of_set, sum_of_set)) {

            Eigen::Matrix<double, -1, 1> p_of_the_set(K + 1);
            const double* log_p_r = log_p + static_cast<std::size_t>(r) * (2 * K + 1);
            for (int j = 0; j <= K; ++j) p_of_the_set(j) = std::exp(log_p_r[j]);
            prior::run_the_checks_of_Stan(p_of_the_set, Eigen::Map<const Eigen::Matrix<double, -1, 1>>(alpha_kept,
                                                                                                       K + 1));

        }
        sum += sum_of_set;
    }
    if constexpr (include_the_lgamma_terms) {
        double sum_of_alpha = 0.0;
        double sum_of_lgamma_of_alpha = 0.0;
        for (int j = 0; j <= K; ++j) {
            sum_of_alpha += alpha_kept[j];
            sum_of_lgamma_of_alpha += std::lgamma(alpha_kept[j]);
        }
        sum += R * (std::lgamma(sum_of_alpha) - sum_of_lgamma_of_alpha);
    }
    if constexpr (C_is_constant && alpha_is_constant) {
        return sum;
    } else {
        using C_vars_type = std::conditional_t<C_is_constant,
                                               Eigen::Matrix<var, Eigen::Dynamic, Eigen::Dynamic>,
                                               stan::plain_type_t<stan::math::promote_scalar_t<var, T_C>>>;
        using alpha_vars_type = Eigen::Matrix<var, Eigen::Dynamic, 1>;
        stan::arena_t<C_vars_type> C_vars;
        stan::arena_t<alpha_vars_type> alpha_vars;
        if constexpr (!C_is_constant) C_vars = C_ref;
        if constexpr (!alpha_is_constant) alpha_vars = alpha_ref;
        return stan::math::make_callback_var(sum, [C_vars, alpha_vars, x, alpha_kept, link, R, K, F, p, log_p,
                                                   J_terms](auto& vi) mutable {
            if constexpr (!C_is_constant) {
                for (int r = 0; r < R; ++r) {
                    double* d_x = J_terms + static_cast<std::size_t>(r) * K;
                    prior::reverse_pass(x + static_cast<std::size_t>(r) * K, alpha_kept, link, K,
                                        F + static_cast<std::size_t>(r) * K,
                                        p + static_cast<std::size_t>(r) * (K + 1),
                                        log_p + static_cast<std::size_t>(r) * (2 * K + 1), vi.adj(), d_x);
                    for (int k = 0; k < K; ++k) {
                        if constexpr (C_is_one_set) {
                            C_vars.coeffRef(k).adj() += d_x[k];
                        } else {
                            C_vars.coeffRef(r, k).adj() += d_x[k];
                        }
                    }
                }
            }
            if constexpr (!alpha_is_constant) {
                double sum_of_alpha = 0.0;
                for (int j = 0; j <= K; ++j) sum_of_alpha += alpha_kept[j];
                const double digamma_of_the_sum = stan::math::digamma(sum_of_alpha);
                for (int j = 0; j <= K; ++j) {
                    double sum_of_log_p = 0.0;
                    for (int r = 0; r < R; ++r) {
                        sum_of_log_p += log_p[static_cast<std::size_t>(r) * (2 * K + 1) + j];
                    }
                    const double d_alpha = sum_of_log_p +
                                           R * (digamma_of_the_sum - stan::math::digamma(alpha_kept[j]));
                    alpha_vars.coeffRef(j).adj() += vi.adj() * d_alpha;
                }
            }
        });
    }

}

}  // namespace nico::math

using nico::math::nico_induced_dirichlet_prior_of_ordered_cutpoints_lpdf;




// ---- the Stan-facing function at the latent location 0 (stanc calls it as
//      nico_induced_dirichlet_prior_of_ordered_cutpoints_lpdf<propto__>(C, alpha, link, pstream__))
namespace nico::math {

template <bool propto, typename T_C, typename T_alpha, stan::require_eigen_t<T_C>* = nullptr,
          stan::require_eigen_col_vector_t<T_alpha>* = nullptr>
inline stan::return_type_t<T_C, T_alpha> nico_induced_dirichlet_prior_of_ordered_cutpoints_lpdf(
        const T_C& C, const T_alpha& alpha, const int link, std::ostream* pstream__) {

    return nico_induced_dirichlet_prior_of_ordered_cutpoints_lpdf<propto>(C, alpha, link, 0.0, pstream__);

}

}  // namespace nico::math

using nico::math::nico_induced_dirichlet_prior_of_ordered_cutpoints_lpdf;























