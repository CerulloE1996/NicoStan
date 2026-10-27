#pragma once

#include "stan_external_scalar_functions.hpp"
#include "stan_external_lgamma.hpp"

namespace nico::math {

// (the data-alpha overload, commented out: its log of theta made one node per element and it was slower
//  than Stan's; the overload below takes a data alpha with one node for theta)
// template <bool propto__, typename T_theta>
// inline stan::return_type_t<T_theta>
// nico_dirichlet_lpdf(const Eigen::Matrix<T_theta, Eigen::Dynamic, 1>& theta,
//                     const Eigen::Matrix<double, -1, 1>& alpha,
//                     std::ostream* pstream__) {
//
//     static_assert(std::is_arithmetic<T_theta>::value || stan::is_var<T_theta>::value,
//                   "The probabilities must be arithmetic or reverse-mode scalars.");
//     static constexpr const char* function = "dirichlet_lpdf";
//     stan::math::check_consistent_sizes(function, "probabilities", theta, "prior sample sizes", alpha);
//     stan::math::check_positive(function, "prior sample sizes", alpha);
//     stan::math::check_simplex(function, "probabilities", theta);
//
//     stan::return_type_t<T_theta> log_prob = 0.0;
//     if constexpr (!propto__) {
//         // Eigen::Matrix<double, -1, 1> log_gamma_alpha(alpha.size());
//         // for (Eigen::Index i = 0; i < alpha.size(); ++i) {
//         //     log_gamma_alpha[i] = nico_lgamma(alpha[i], pstream__);
//         // }
//         // log_prob += nico_lgamma(alpha.sum(), pstream__) - log_gamma_alpha.sum();
//         //// (lgamma of alpha_1, ..., alpha_K and of sum(alpha) in one SIMD pass)
//         const Eigen::Index K = alpha.size();
//         Eigen::Matrix<double, -1, 1> log_gamma_of_alpha_and_sum(K + 1);
//         log_gamma_of_alpha_and_sum.head(K) = alpha;
//         log_gamma_of_alpha_and_sum[K] = alpha.sum();
//         k_lgamma(log_gamma_of_alpha_and_sum.data(), static_cast<int>(K + 1));
//         log_prob += log_gamma_of_alpha_and_sum[K] - log_gamma_of_alpha_and_sum.head(K).sum();
//     }
//     if constexpr (stan::math::include_summand<propto__, T_theta>::value) {
//         const auto log_theta = nico_log(theta, pstream__);
//         for (Eigen::Index i = 0; i < theta.size(); ++i) {
//             log_prob += (alpha[i] - 1.0) * log_theta[i];
//         }
//     }
//     return log_prob;
//
// }

}  // namespace nico::math

// using nico::math::nico_dirichlet_lpdf;   (declared after the overload below)




// ---- alpha data or parameters (e.g. alpha = 0.01 + phi kappa): one node for theta and alpha,
//      d / d theta_i = (alpha_i - 1) / theta_i,
//      d / d alpha_i = log(theta_i) + digamma(sum(alpha)) - digamma(alpha_i),
//      log(theta) with k_log_into; lgamma (and digamma for a parameter alpha) of alpha_1, ..., alpha_K and of
//      sum(alpha) in one pass (k_lgamma_and_digamma); each term kept as Stan's dirichlet_lpdf keeps it (the
//      lgamma terms not in the lupdf of a data alpha). The values of a call sit in the thread's scratch buffer.
//      Declared as   real nico_dirichlet_lpdf(vector theta, vector alpha);
#include "stan_external_functions.hpp"

namespace nico::math {

// ---- K <= 7 (e.g. q of the m + 1 segments): no buffer and no Eigen temporaries; log(theta) in one SIMD vector
//      (k_log_and_log1m_short), lgamma (and digamma) of alpha and sum(alpha) in one more (k_lgamma_and_digamma);
//      the same terms and partials as the general path below
static constexpr int NICO_DIRICHLET_SHORT_PATH_UP_TO_K = 7;

template <bool propto__, typename T_theta, typename T_alpha, typename T_theta_ref, typename T_alpha_ref>
inline stan::return_type_t<T_theta, T_alpha>
nico_dirichlet_lpdf_short(const T_theta_ref& theta_ref,
                          const T_alpha_ref& alpha_ref) {

    using stan::math::include_summand;
    using stan::math::value_of;
    static constexpr const char* function = "dirichlet_lpdf";
    const int K = static_cast<int>(alpha_ref.size());
    double theta_values[NICO_DIRICHLET_SHORT_PATH_UP_TO_K];
    double alpha_and_sum[NICO_DIRICHLET_SHORT_PATH_UP_TO_K + 1];
    for (int i = 0; i < K; ++i) {
        theta_values[i] = value_of(theta_ref.coeff(i));
        alpha_and_sum[i] = value_of(alpha_ref.coeff(i));
    }
    // stan::math::check_positive(function, "prior sample sizes",
    //                            Eigen::Map<const Eigen::VectorXd>(alpha_and_sum, K));
    // stan::math::check_simplex(function, "probabilities", Eigen::Map<const Eigen::VectorXd>(theta_values, K));
    //// (one pass over the values; Stan's checks only when one fails)
    double sum_of_theta = 0.0;
    double sum_of_alpha = 0.0;
    bool all_valid = K > 0;
    for (int i = 0; i < K; ++i) {
        all_valid = all_valid && (alpha_and_sum[i] > 0.0) && (theta_values[i] >= 0.0);
        sum_of_theta += theta_values[i];
        sum_of_alpha += alpha_and_sum[i];
    }
    all_valid = all_valid && (std::fabs(1.0 - sum_of_theta) <= stan::math::CONSTRAINT_TOLERANCE);
    if (!all_valid) {
        stan::math::check_positive(function, "prior sample sizes",
                                   Eigen::Map<const Eigen::VectorXd>(alpha_and_sum, K));
        stan::math::check_simplex(function, "probabilities",
                                  Eigen::Map<const Eigen::VectorXd>(theta_values, K));
    }
    if constexpr (!include_summand<propto__, T_theta, T_alpha>::value) {
        return 0.0;
    }

    double log_theta[NICO_DIRICHLET_SHORT_PATH_UP_TO_K];
    k_log_and_log1m_short(theta_values, K, nullptr, 0, log_theta, nullptr);
    double log_prob = 0.0;
    for (int i = 0; i < K; ++i) log_prob += (alpha_and_sum[i] - 1.0) * log_theta[i];
    constexpr bool lgamma_needed = include_summand<propto__, T_alpha>::value;
    constexpr bool digamma_needed = !stan::is_constant_all<T_alpha>::value;
    double lgamma_of_alpha_and_sum[NICO_DIRICHLET_SHORT_PATH_UP_TO_K + 1];
    double digamma_of_alpha_and_sum[NICO_DIRICHLET_SHORT_PATH_UP_TO_K + 1];
    if constexpr (lgamma_needed || digamma_needed) {
        // double sum_of_alpha = 0.0;
        // for (int i = 0; i < K; ++i) sum_of_alpha += alpha_and_sum[i];
        //// (sum(alpha) from the check pass)
        alpha_and_sum[K] = sum_of_alpha;
        k_lgamma_and_digamma<digamma_needed>(alpha_and_sum, K + 1, lgamma_of_alpha_and_sum,
                                             digamma_of_alpha_and_sum);
    }
    if constexpr (lgamma_needed) {
        log_prob += lgamma_of_alpha_and_sum[K];
        for (int i = 0; i < K; ++i) log_prob -= lgamma_of_alpha_and_sum[i];
    }

    auto ops_partials = stan::math::make_partials_propagator(theta_ref, alpha_ref);
    for (int i = 0; i < K; ++i) {
        if constexpr (!stan::is_constant_all<T_theta>::value) {
            stan::math::partials<0>(ops_partials)[i] = (alpha_and_sum[i] - 1.0) / theta_values[i];
        }
        if constexpr (digamma_needed) {
            stan::math::partials<1>(ops_partials)[i] = log_theta[i] + digamma_of_alpha_and_sum[K]
                                                       - digamma_of_alpha_and_sum[i];
        }
    }
    return ops_partials.build(log_prob);

}

template <bool propto__, typename T_theta, typename T_alpha,
          stan::require_all_eigen_col_vector_t<T_theta, T_alpha>* = nullptr>
inline stan::return_type_t<T_theta, T_alpha>
nico_dirichlet_lpdf(const T_theta& theta,
                    const T_alpha& alpha,
                    std::ostream* pstream__) {

    using stan::math::include_summand;
    static constexpr const char* function = "dirichlet_lpdf";
    const auto& theta_ref = stan::math::to_ref(theta);
    const auto& alpha_ref = stan::math::to_ref(alpha);
    stan::math::check_consistent_sizes(function, "probabilities", theta_ref, "prior sample sizes", alpha_ref);
    const int K = static_cast<int>(alpha_ref.size());
    if (K <= NICO_DIRICHLET_SHORT_PATH_UP_TO_K) {
        return nico_dirichlet_lpdf_short<propto__, T_theta, T_alpha>(theta_ref, alpha_ref);
    }
    // (scratch: theta K, alpha and sum(alpha) K + 1, log(theta) K, lgamma K + 1, digamma K + 1)
    double* scratch = nico_scratch_of_doubles(5 * static_cast<std::size_t>(K) + 3);
    Eigen::Map<Eigen::VectorXd> theta_values(scratch, K);
    Eigen::Map<Eigen::VectorXd> alpha_and_sum(scratch + K, K + 1);
    Eigen::Map<Eigen::VectorXd> log_theta(scratch + 2 * K + 1, K);
    Eigen::Map<Eigen::VectorXd> lgamma_of_alpha_and_sum(scratch + 3 * K + 1, K + 1);
    Eigen::Map<Eigen::VectorXd> digamma_of_alpha_and_sum(scratch + 4 * K + 2, K + 1);
    theta_values = stan::math::value_of(theta_ref);
    alpha_and_sum.head(K) = stan::math::value_of(alpha_ref);
    stan::math::check_positive(function, "prior sample sizes", alpha_and_sum.head(K));
    stan::math::check_simplex(function, "probabilities", theta_values);
    if constexpr (!include_summand<propto__, T_theta, T_alpha>::value) {
        return 0.0;
    }

    k_log_into(theta_values.data(), K, log_theta.data());
    double log_prob = 0.0;
    for (int i = 0; i < K; ++i) log_prob += (alpha_and_sum[i] - 1.0) * log_theta[i];
    constexpr bool lgamma_needed = include_summand<propto__, T_alpha>::value;
    constexpr bool digamma_needed = !stan::is_constant_all<T_alpha>::value;
    if constexpr (lgamma_needed || digamma_needed) {
        alpha_and_sum[K] = alpha_and_sum.head(K).sum();
        k_lgamma_and_digamma<digamma_needed>(alpha_and_sum.data(), K + 1, lgamma_of_alpha_and_sum.data(),
                                             digamma_of_alpha_and_sum.data());
    }
    if constexpr (lgamma_needed) {
        log_prob += lgamma_of_alpha_and_sum[K] - lgamma_of_alpha_and_sum.head(K).sum();
    }

    auto ops_partials = stan::math::make_partials_propagator(theta_ref, alpha_ref);
    if constexpr (!stan::is_constant_all<T_theta>::value) {
        stan::math::partials<0>(ops_partials) = (alpha_and_sum.head(K).array() - 1.0) / theta_values.array();
    }
    if constexpr (digamma_needed) {
        stan::math::partials<1>(ops_partials) = log_theta.array() + digamma_of_alpha_and_sum[K]
                                                - digamma_of_alpha_and_sum.head(K).array();
    }
    return ops_partials.build(log_prob);

}

}  // namespace nico::math

using nico::math::nico_dirichlet_lpdf;
