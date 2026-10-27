#pragma once
//
// -| --------- NicoSIMD factorised binomial log-likelihood of ordinal test accuracy data (generated quantities)
//
// The nico_ versions of MetaOrdDTA's functions of the generated quantities of its ordinal models
// (Stan_fns_log_lik.stan, Stan_fns_ordinal.stan), with the same arguments and results, in double precision
// (the model blocks use the one autodiff node of
// nico_binomial_counts_above_successive_cutpoints_of_a_cumulative_link_lpmf for the same likelihood). For each
// class c (1, 2), study s and reported cutpoint cut_i (1, ..., n_obs_cutpoints[s]), with
// k = cutpoint_index[c, s, cut_i + 1], latent_surv[c][s, cut_i] is (+infinity everywhere else):
//
//   * nico_map_latent_surv_to_random_hetero_C(C, n_thr, locations, [scales,] n_studies, n_obs_cutpoints,
//                                             cutpoint_index)          (Xu, random cutpoints: C[c] a matrix)
//       (locations[s, c] - C[c][s, k]) / scales[s, c], or locations[s, c] - C[c][s, k], when k != -1;
//   * nico_map_latent_surv_to_fixed_hetero_C(C, locations, [scales,] n_studies, n_obs_cutpoints, cutpoint_index)
//                                                                       (Xu, fixed cutpoints: C[c] a vector)
//       -(C[c][k] - locations[s, c]) / scales[s, c], or locations[s, c] - C[c][k], when k != -1;
//   * nico_map_latent_surv_prob_to_fixed_C_HSROC(C, locations, scales, n_studies, n_obs_cutpoints,
//                                                cutpoint_index)
//                                                                       (R&G, fixed cutpoints: C a vector)
//       -(C[k] + locations[s, c]) * scales[s, c], when k != 0 and k != -1;
//   * nico_map_latent_surv_prob_to_random_C_HSROC(C, n_thr, locations, scales, n_studies, n_obs_cutpoints,
//                                                 cutpoint_index)       (R&G, random cutpoints: C a matrix)
//       -(C[s, k] + locations[s, c]) * scales[s, c], when k != 0 and k != -1;
//
// and the factorised binomial log-likelihood, { log_lik, cond_prob, surv_prob } (each an array of 2 matrices
// [n_studies, n_thr]):
//
//   * nico_compute_log_lik_binomial_fact_data(latent_surv, use_probit_link, n_thr, n_studies, x_2, n, N_total,
//                                             n_obs_cutpoints, K_fold_CV_indicator)
//   * nico_compute_NMA_log_lik_binomial_fact_data(latent_surv, use_probit_link, n_thr, n_studies, x_2, n,
//                                                 N_total, n_obs_cutpoints, indicator_index_test_t_in_study,
//                                                 K_fold_CV_indicator)
//       surv_prob[c] = Phi(latent_surv[c]) (use_probit_link = 1; 0 below -37.5, 1 from 8.25, as safe_Phi) or
//                      inv_logit(latent_surv[c]), on every element (1 where latent_surv is +infinity);
//       cond_prob[c][s, 1] = surv_prob[c][s, 1],
//       cond_prob[c][s, k] = surv_prob[c][s, k] / surv_prob[c][s, k - 1] (k = 2, ..., n_obs_cutpoints[s]),
//                      1 elsewhere;
//       log_lik[c][s, k] = safe_binomial_lpmf(x_2[c, s, k] | n[c, s, k], cond_prob[c][s, k]) (k = 1, ...,
//                      n_obs_cutpoints[s]) in the studies with K_fold_CV_indicator[s] = 1 (and, in the network
//                      meta-analysis, indicator_index_test_t_in_study[s] = 1), 0 elsewhere: the binomial log
//                      probability at the probability kept within [1e-10, 1 - 1e-10], every constant included
//                      (log choose(n, x)), as safe_binomial_lpmf.
//
// Phi and inv_logit with BayesMVP's SIMD kernels over every element at once (the same kernels as nico_Phi and
// nico_inv_logit), the logs of the probabilities of all the terms with the log kernel and the double-precision
// log1m (as nico_binomial_lpmf), and log choose(n, x) from nico_binomial_lpmf's table of log factorials. The
// latent values are computed with the same arithmetic, in the same order, as the Stan functions (so they are the
// same doubles).
//
// Declared in the Stan functions block (no body; stanc --allow-undefined) as any of
//
//     array[] matrix nico_map_latent_surv_to_random_hetero_C(data array[] matrix C, data int n_thr,
//                                                            data matrix locations, data matrix scales,
//                                                            data int n_studies,
//                                                            data array[] int n_obs_cutpoints,
//                                                            data array[,,] int cutpoint_index);
//     array[] matrix nico_map_latent_surv_to_random_hetero_C(data array[] matrix C, data int n_thr,
//                                                            data matrix locations,
//                                                            data int n_studies,
//                                                            data array[] int n_obs_cutpoints,
//                                                            data array[,,] int cutpoint_index);
//     array[] matrix nico_map_latent_surv_to_fixed_hetero_C(data array[] vector C,
//                                                           data matrix locations, data matrix scales,
//                                                           data int n_studies,
//                                                           data array[] int n_obs_cutpoints,
//                                                           data array[,,] int cutpoint_index);
//     array[] matrix nico_map_latent_surv_to_fixed_hetero_C(data array[] vector C,
//                                                           data matrix locations,
//                                                           data int n_studies,
//                                                           data array[] int n_obs_cutpoints,
//                                                           data array[,,] int cutpoint_index);
//     array[] matrix nico_map_latent_surv_prob_to_fixed_C_HSROC(data vector C,
//                                                               data matrix locations, data matrix scales,
//                                                               data int n_studies,
//                                                               data array[] int n_obs_cutpoints,
//                                                               data array[,,] int cutpoint_index);
//     array[] matrix nico_map_latent_surv_prob_to_random_C_HSROC(data matrix C, data int n_thr,
//                                                                data matrix locations, data matrix scales,
//                                                                data int n_studies,
//                                                                data array[] int n_obs_cutpoints,
//                                                                data array[,,] int cutpoint_index);
//     array[,] matrix nico_compute_log_lik_binomial_fact_data(data array[] matrix latent_surv,
//                                                             data int use_probit_link, data int n_thr,
//                                                             data int n_studies, data array[,,] int x_2,
//                                                             data array[,,] int n, data array[,] int N_total,
//                                                             data array[] int n_obs_cutpoints,
//                                                             data array[] int K_fold_CV_indicator);
//     array[,] matrix nico_compute_NMA_log_lik_binomial_fact_data(data array[] matrix latent_surv,
//                                                                 data int use_probit_link, data int n_thr,
//                                                                 data int n_studies, data array[,,] int x_2,
//                                                                 data array[,,] int n,
//                                                                 data array[,] int N_total,
//                                                                 data array[] int n_obs_cutpoints,
//                                                                 data array[] int
//                                                                       indicator_index_test_t_in_study,
//                                                                 data array[] int K_fold_CV_indicator);
//
// (data: the values of the generated quantities, as MetaOrdDTA's compute_log_lik_binomial_fact_data). It uses
// the namespace nico::math, the kernels of stan_external_functions.hpp and the table of log factorials of
// stan_external_binomial_lpmf.hpp, which it includes (#pragma once).
//
// -----------------------------------------------------------------------------------------------------------

#include "stan_external_functions.hpp"
#include "stan_external_binomial_lpmf.hpp"

#include <stan/math/prim.hpp>

#include <Eigen/Dense>
#include <cmath>
#include <limits>
#include <ostream>
#include <vector>

namespace nico::math {
namespace factorised_binomial_of_ordinal_test_accuracy_data {

// ---- latent_surv of both classes: latent_value(c, s, k) at each reported cutpoint whose index k passes
//      index_is_used(k), +infinity elsewhere
template <typename F_latent_value, typename F_index_is_used>
inline std::vector<Eigen::Matrix<double, -1, -1>> latent_surv_of_both_classes(const int n_thr,
                                                                              const int n_studies,
                                                                const std::vector<int>& n_obs_cutpoints,
                                                                const std::vector<std::vector<std::vector<int>>>&
                                                                        cutpoint_index,
                                                                const F_latent_value& latent_value,
                                                                const F_index_is_used& index_is_used) {

    std::vector<Eigen::Matrix<double, -1, -1>> latent_surv(2);
    for (int c = 0; c < 2; ++c) {
        latent_surv[c] = Eigen::Matrix<double, -1, -1>::Constant(n_studies, n_thr,
                                                                 std::numeric_limits<double>::infinity());
        for (int s = 0; s < n_studies; ++s) {
            for (int cut_i = 0; cut_i < n_obs_cutpoints[s]; ++cut_i) {
                const int k = cutpoint_index[c][s][cut_i + 1];
                if (index_is_used(k)) latent_surv[c](s, cut_i) = latent_value(c, s, k - 1);
            }
        }
    }
    return latent_surv;

}
//
inline bool index_is_not_minus_1(const int k) { return k != -1; }
inline bool index_is_not_0_or_minus_1(const int k) { return (k != 0) && (k != -1); }

// ---- the factorised binomial log-likelihood, conditional and survival probabilities; study_is_included(s) says
//      which studies have log-likelihood terms
template <typename T_latent_surv, typename F_study_is_included>
inline std::vector<std::vector<Eigen::Matrix<double, -1, -1>>> factorised_binomial(
        const std::vector<T_latent_surv>& latent_surv, const int use_probit_link, const int n_thr,
        const int n_studies, const std::vector<std::vector<std::vector<int>>>& x_2,
        const std::vector<std::vector<std::vector<int>>>& n, const std::vector<int>& n_obs_cutpoints,
        const F_study_is_included& study_is_included) {

    namespace binomial = nico::math::binomial;
    const int n_values = n_studies * n_thr;
    std::vector<Eigen::Matrix<double, -1, -1>> surv_prob(2), cond_prob(2), log_lik(2);
    //// (the terms of the likelihood, both classes: their probability, kept within [1e-10, 1 - 1e-10], counts,
    ////  and where their log probability goes)
    std::vector<double> p_of_each_term, log_p_of_each_term, log1m_p_of_each_term;
    std::vector<int> x_of_each_term, n_of_each_term, class_of_each_term, index_of_each_term;
    p_of_each_term.reserve(2 * n_values);
    for (int c = 0; c < 2; ++c) {
        //// (the survival probabilities: Phi or inv_logit of every element, with the SIMD kernels; 1 at +infinity
        ////  and 0 at -infinity, the values of safe_Phi and inv_logit there)
        const Eigen::Matrix<double, -1, -1> latent_c = latent_surv[c];
        surv_prob[c] = latent_c;
        double* values = surv_prob[c].data();
        if (use_probit_link == 1) {
            nico::math::k_Phi(values, n_values);
        } else {
            nico::math::k_inv_logit(values, n_values);
        }
        for (int i = 0; i < n_values; ++i) {
            const double x = latent_c.data()[i];
            if (std::isinf(x)) values[i] = (x > 0.0) ? 1.0 : 0.0;
        }
        //// (the conditional probabilities, 1 elsewhere)
        cond_prob[c] = Eigen::Matrix<double, -1, -1>::Ones(n_studies, n_thr);
        log_lik[c] = Eigen::Matrix<double, -1, -1>::Zero(n_studies, n_thr);
        for (int s = 0; s < n_studies; ++s) {
            if (n_thr > 0) cond_prob[c](s, 0) = surv_prob[c](s, 0);
            for (int k = 1; k < n_obs_cutpoints[s]; ++k) {
                cond_prob[c](s, k) = surv_prob[c](s, k) / surv_prob[c](s, k - 1);
            }
            if (!study_is_included(s)) continue;
            for (int k = 0; k < n_obs_cutpoints[s]; ++k) {
                p_of_each_term.push_back(std::fmax(1e-10, std::fmin(1.0 - 1e-10, cond_prob[c](s, k))));
                x_of_each_term.push_back(x_2[c][s][k]);
                n_of_each_term.push_back(n[c][s][k]);
                class_of_each_term.push_back(c);
                index_of_each_term.push_back(s + k * n_studies);
            }
        }
    }
    //// (the log probabilities of all the terms at once, with the log kernel and the double-precision log1m)
    const int n_terms = static_cast<int>(p_of_each_term.size());
    log_p_of_each_term = p_of_each_term;
    log1m_p_of_each_term = p_of_each_term;
    if (n_terms > 0) {
        nico::math::k_log(log_p_of_each_term.data(), n_terms);
        binomial::log1m_in_double_precision(log1m_p_of_each_term.data(), n_terms);
    }
    const double* table = binomial::table_of_log_factorials();
    for (int i = 0; i < n_terms; ++i) {
        const int x = x_of_each_term[i];
        const int N = n_of_each_term[i];
        double lp = binomial::log_factorial(N, table) - binomial::log_factorial(x, table) -
                    binomial::log_factorial(N - x, table);
        if (x > 0) lp += x * log_p_of_each_term[i];
        if (N - x > 0) lp += (N - x) * log1m_p_of_each_term[i];
        log_lik[class_of_each_term[i]].data()[index_of_each_term[i]] = lp;
    }
    std::vector<std::vector<Eigen::Matrix<double, -1, -1>>> out(3);
    out[0] = log_lik;
    out[1] = cond_prob;
    out[2] = surv_prob;
    return out;

}

}  // namespace factorised_binomial_of_ordinal_test_accuracy_data
}  // namespace nico::math

// ---- the Stan-facing functions (global scope; stanc calls them with pstream__ last)
//
// (Xu, random cutpoints, with scales)
namespace nico::math {

template <typename T_C, typename T_locations, typename T_scales>
inline std::vector<Eigen::Matrix<double, -1, -1>> nico_map_latent_surv_to_random_hetero_C(
        const std::vector<T_C>& C, const int n_thr, const T_locations& locations, const T_scales& scales,
        const int n_studies, const std::vector<int>& n_obs_cutpoints,
        const std::vector<std::vector<std::vector<int>>>& cutpoint_index, std::ostream* /*pstream__*/) {

    namespace fact = nico::math::factorised_binomial_of_ordinal_test_accuracy_data;
    const Eigen::Matrix<double, -1,
                        -1> C_1 = C[0], C_2 = C[1], locations_values = locations, scales_values = scales;
    return fact::latent_surv_of_both_classes(n_thr, n_studies, n_obs_cutpoints, cutpoint_index,
        [&](const int c, const int s, const int k) {

            return (locations_values(s, c) - ((c == 0) ? C_1 : C_2)(s, k)) / scales_values(s, c);

        }, fact::index_is_not_minus_1);

}

}  // namespace nico::math

using nico::math::nico_map_latent_surv_to_random_hetero_C;
//
// (Xu, random cutpoints, without scales: scale 1)
namespace nico::math {

template <typename T_C, typename T_locations>
inline std::vector<Eigen::Matrix<double, -1, -1>> nico_map_latent_surv_to_random_hetero_C(
        const std::vector<T_C>& C, const int n_thr, const T_locations& locations, const int n_studies,
        const std::vector<int>& n_obs_cutpoints, const std::vector<std::vector<std::vector<int>>>& cutpoint_index,
        std::ostream* /*pstream__*/) {

    namespace fact = nico::math::factorised_binomial_of_ordinal_test_accuracy_data;
    const Eigen::Matrix<double, -1, -1> C_1 = C[0], C_2 = C[1], locations_values = locations;
    return fact::latent_surv_of_both_classes(n_thr, n_studies, n_obs_cutpoints, cutpoint_index,
        [&](const int c, const int s, const int k) {

            return locations_values(s, c) - ((c == 0) ? C_1 : C_2)(s, k);

        }, fact::index_is_not_minus_1);

}

}  // namespace nico::math

using nico::math::nico_map_latent_surv_to_random_hetero_C;
//
// (Xu, fixed cutpoints, with scales)
namespace nico::math {

template <typename T_C, typename T_locations, typename T_scales>
inline std::vector<Eigen::Matrix<double, -1, -1>> nico_map_latent_surv_to_fixed_hetero_C(
        const std::vector<T_C>& C, const T_locations& locations, const T_scales& scales, const int n_studies,
        const std::vector<int>& n_obs_cutpoints, const std::vector<std::vector<std::vector<int>>>& cutpoint_index,
        std::ostream* /*pstream__*/) {

    namespace fact = nico::math::factorised_binomial_of_ordinal_test_accuracy_data;
    const Eigen::Matrix<double, -1, 1> C_1 = C[0], C_2 = C[1];
    const Eigen::Matrix<double, -1, -1> locations_values = locations, scales_values = scales;
    return fact::latent_surv_of_both_classes(static_cast<int>(C_1.size()), n_studies, n_obs_cutpoints,
                                             cutpoint_index,
        [&](const int c, const int s, const int k) {

            return -(((c == 0) ? C_1 : C_2)(k) - locations_values(s, c)) / scales_values(s, c);

        }, fact::index_is_not_minus_1);

}

}  // namespace nico::math

using nico::math::nico_map_latent_surv_to_fixed_hetero_C;
//
// (Xu, fixed cutpoints, without scales)
namespace nico::math {

template <typename T_C, typename T_locations>
inline std::vector<Eigen::Matrix<double, -1, -1>> nico_map_latent_surv_to_fixed_hetero_C(
        const std::vector<T_C>& C, const T_locations& locations, const int n_studies,
        const std::vector<int>& n_obs_cutpoints, const std::vector<std::vector<std::vector<int>>>& cutpoint_index,
        std::ostream* /*pstream__*/) {

    namespace fact = nico::math::factorised_binomial_of_ordinal_test_accuracy_data;
    const Eigen::Matrix<double, -1, 1> C_1 = C[0], C_2 = C[1];
    const Eigen::Matrix<double, -1, -1> locations_values = locations;
    return fact::latent_surv_of_both_classes(static_cast<int>(C_1.size()), n_studies, n_obs_cutpoints,
                                             cutpoint_index,
        [&](const int c, const int s, const int k) {

            return locations_values(s, c) - ((c == 0) ? C_1 : C_2)(k);

        }, fact::index_is_not_minus_1);

}

}  // namespace nico::math

using nico::math::nico_map_latent_surv_to_fixed_hetero_C;
//
// (R&G, fixed cutpoints: one vector C for both classes)
namespace nico::math {

template <typename T_C, typename T_locations, typename T_scales, stan::require_eigen_vector_t<T_C>* = nullptr>
inline std::vector<Eigen::Matrix<double, -1, -1>> nico_map_latent_surv_prob_to_fixed_C_HSROC(
        const T_C& C, const T_locations& locations, const T_scales& scales, const int n_studies,
        const std::vector<int>& n_obs_cutpoints, const std::vector<std::vector<std::vector<int>>>& cutpoint_index,
        std::ostream* /*pstream__*/) {

    namespace fact = nico::math::factorised_binomial_of_ordinal_test_accuracy_data;
    const Eigen::Matrix<double, -1, 1> C_values = C;
    const Eigen::Matrix<double, -1, -1> locations_values = locations, scales_values = scales;
    return fact::latent_surv_of_both_classes(static_cast<int>(C_values.size()), n_studies, n_obs_cutpoints,
                                             cutpoint_index,
        [&](const int c, const int s, const int k) {

            return -(C_values(k) + locations_values(s, c)) * scales_values(s, c);

        }, fact::index_is_not_0_or_minus_1);

}

}  // namespace nico::math

using nico::math::nico_map_latent_surv_prob_to_fixed_C_HSROC;
//
// (R&G, random cutpoints: one matrix C [n_studies, n_thr] for both classes)
namespace nico::math {

template <typename T_C, typename T_locations, typename T_scales,
          stan::require_eigen_matrix_dynamic_t<T_C>* = nullptr>
inline std::vector<Eigen::Matrix<double, -1, -1>> nico_map_latent_surv_prob_to_random_C_HSROC(
        const T_C& C, const int n_thr, const T_locations& locations, const T_scales& scales, const int n_studies,
        const std::vector<int>& n_obs_cutpoints, const std::vector<std::vector<std::vector<int>>>& cutpoint_index,
        std::ostream* /*pstream__*/) {

    namespace fact = nico::math::factorised_binomial_of_ordinal_test_accuracy_data;
    const Eigen::Matrix<double, -1, -1> C_values = C, locations_values = locations, scales_values = scales;
    return fact::latent_surv_of_both_classes(n_thr, n_studies, n_obs_cutpoints, cutpoint_index,
        [&](const int c, const int s, const int k) {

            return -(C_values(s, k) + locations_values(s, c)) * scales_values(s, c);

        }, fact::index_is_not_0_or_minus_1);

}

}  // namespace nico::math

using nico::math::nico_map_latent_surv_prob_to_random_C_HSROC;
//
// (the factorised binomial log-likelihood of a meta-analysis)
namespace nico::math {

template <typename T_latent_surv>
inline std::vector<std::vector<Eigen::Matrix<double, -1, -1>>> nico_compute_log_lik_binomial_fact_data(
        const std::vector<T_latent_surv>& latent_surv, const int use_probit_link, const int n_thr,
        const int n_studies, const std::vector<std::vector<std::vector<int>>>& x_2,
        const std::vector<std::vector<std::vector<int>>>& n, const std::vector<std::vector<int>>& /*N_total*/,
        const std::vector<int>& n_obs_cutpoints, const std::vector<int>& K_fold_CV_indicator,
        std::ostream* /*pstream__*/) {

    namespace fact = nico::math::factorised_binomial_of_ordinal_test_accuracy_data;
    return fact::factorised_binomial(latent_surv, use_probit_link, n_thr, n_studies, x_2, n, n_obs_cutpoints,
                                     [&](const int s) { return K_fold_CV_indicator[s] == 1; });

}

}  // namespace nico::math

using nico::math::nico_compute_log_lik_binomial_fact_data;
//
// (the factorised binomial log-likelihood of one test of a network meta-analysis: the studies with the test)
namespace nico::math {

template <typename T_latent_surv>
inline std::vector<std::vector<Eigen::Matrix<double, -1, -1>>> nico_compute_NMA_log_lik_binomial_fact_data(
        const std::vector<T_latent_surv>& latent_surv, const int use_probit_link, const int n_thr,
        const int n_studies, const std::vector<std::vector<std::vector<int>>>& x_2,
        const std::vector<std::vector<std::vector<int>>>& n, const std::vector<std::vector<int>>& /*N_total*/,
        const std::vector<int>& n_obs_cutpoints, const std::vector<int>& indicator_index_test_t_in_study,
        const std::vector<int>& K_fold_CV_indicator, std::ostream* /*pstream__*/) {

    namespace fact = nico::math::factorised_binomial_of_ordinal_test_accuracy_data;
    return fact::factorised_binomial(latent_surv, use_probit_link, n_thr, n_studies, x_2, n, n_obs_cutpoints,
                                     [&](const int s) {

                                         return (indicator_index_test_t_in_study[s] == 1) &&
                                                (K_fold_CV_indicator[s] == 1);

                                     });

}

}  // namespace nico::math

using nico::math::nico_compute_NMA_log_lik_binomial_fact_data;






















