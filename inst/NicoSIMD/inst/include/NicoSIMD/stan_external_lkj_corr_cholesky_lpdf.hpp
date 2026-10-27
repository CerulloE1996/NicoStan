#pragma once

#include "stan_external_scalar_functions.hpp"
#include "stan_external_lgamma.hpp"

namespace nico::math {

inline double
nico_lkj_constant_from_data(const double eta,
                            const unsigned int K,
                            std::ostream* pstream__) {

    const int Km1 = K - 1;
    double constant;
    if (eta == 1.0) {
        Eigen::Matrix<double, -1, 1> denominator(Km1 / 2);
        for (Eigen::Index k = 1; k <= denominator.rows(); ++k) {
            denominator[k - 1] = nico_lgamma(2.0 * k, pstream__);
        }
        constant = -denominator.sum();
        if (K % 2 == 1) {
            constant -= 0.25 * (K * K - 1) * stan::math::LOG_PI
                        - 0.25 * (Km1 * Km1) * stan::math::LOG_TWO
                        - Km1 * nico_lgamma(0.5 * (K + 1), pstream__);
        } else {
            constant -= 0.25 * K * (K - 2) * stan::math::LOG_PI
                        + 0.25 * (3 * K * K - 4 * K) * stan::math::LOG_TWO
                        + K * nico_lgamma(0.5 * K, pstream__)
                        - Km1 * nico_lgamma(static_cast<double>(K), pstream__);
        }
    } else {
        constant = Km1 * nico_lgamma(eta + 0.5 * Km1, pstream__);
        for (int k = 1; k <= Km1; ++k) {
            constant -= 0.5 * k * stan::math::LOG_PI
                        + nico_lgamma(eta + 0.5 * (Km1 - k), pstream__);
        }
    }
    return constant;

}

template <bool propto__, typename T_L>
inline stan::return_type_t<T_L>
nico_lkj_corr_cholesky_lpdf(const Eigen::Matrix<T_L, Eigen::Dynamic, Eigen::Dynamic>& L,
                           const double eta,
                           std::ostream* pstream__) {

    static_assert(std::is_arithmetic<T_L>::value || stan::is_var<T_L>::value,
                  "The factor must contain arithmetic or reverse-mode scalars.");
    static constexpr const char* function = "lkj_corr_cholesky_lpdf";
    stan::math::check_positive(function, "Shape parameter", eta);
    stan::math::check_cholesky_factor(function, "Random variable", L);
    const unsigned int K = L.rows();
    if (K == 0) return 0.0;

    stan::return_type_t<T_L> log_prob = 0.0;
    if constexpr (!propto__) {
        log_prob += nico_lkj_constant_from_data(eta, K, pstream__);
    }
    if constexpr (stan::math::include_summand<propto__, T_L>::value) {
        const int Km1 = K - 1;
        const Eigen::Matrix<T_L, Eigen::Dynamic, 1> diagonals = L.diagonal().tail(Km1);
        const auto log_diagonals = nico_log(diagonals, pstream__);
        Eigen::Matrix<stan::return_type_t<T_L>, Eigen::Dynamic, 1> values(Km1);
        for (int k = 0; k < Km1; ++k) {
            values[k] = (Km1 - k - 1) * log_diagonals[k];
        }
        if (eta != 1.0) {
            for (int k = 0; k < Km1; ++k) {
                values[k] += (2.0 * eta - 2.0) * log_diagonals[k];
            }
        }
        log_prob += stan::math::sum(values);
    }
    return log_prob;

}

}  // namespace nico::math

using nico::math::nico_lkj_corr_cholesky_lpdf;






















