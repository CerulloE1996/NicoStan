#pragma once

#include <stan/math/mix.hpp>
#include "math/nico_double_probability_reference.hpp"
#include "stan_external_scalar_functions.hpp"
#include "stan_external_std_normal_log_qf.hpp"
#include <cmath>
#include <limits>
#include <ostream>

namespace nico::math {

static constexpr double CUTPOINT_HALF_LOG_TWO_PI = 0.918938533204672741780329736406;
static constexpr double CUTPOINT_LOG_HALF = -0.693147180559945309417232121458;

// ---- lgamma, digamma and trigamma of x > 0: recurrence up to x >= 10, then the asymptotic series

inline void cutpoint_lgamma_digamma_trigamma(const double x,
                                             double* lgamma_of_x,
                                             double* digamma_of_x,
                                             double* trigamma_of_x) {


        double shifted = x;
        double product = 1.0;
        double digamma_shift = 0.0;
        double trigamma_shift = 0.0;
        while (shifted < 10.0) {
                const double inverse = 1.0 / shifted;
                product *= shifted;
                digamma_shift += inverse;
                trigamma_shift += inverse * inverse;
                shifted += 1.0;
        }
        const double inverse = 1.0 / shifted;
        const double inverse_2 = inverse * inverse;
        const double log_shifted = nico_log(shifted, nullptr);

        *lgamma_of_x = (shifted - 0.5) * log_shifted - shifted + CUTPOINT_HALF_LOG_TWO_PI
                       + inverse * (1.0 / 12.0 - inverse_2 * (1.0 / 360.0 - inverse_2 * (1.0 / 1260.0
                       - inverse_2 * (1.0 / 1680.0 - inverse_2 * (1.0 / 1188.0 - inverse_2 * (691.0 / 360360.0
                       - inverse_2 / 156.0))))))
                       - nico_log(product, nullptr);
        *digamma_of_x = log_shifted - 0.5 * inverse
                        - inverse_2 * (1.0 / 12.0 - inverse_2 * (1.0 / 120.0 - inverse_2 * (1.0 / 252.0
                        - inverse_2 * (1.0 / 240.0 - inverse_2 * (1.0 / 132.0 - inverse_2 * (691.0 / 32760.0
                        - inverse_2 / 12.0))))))
                        - digamma_shift;
        if (trigamma_of_x != nullptr) {
                *trigamma_of_x = inverse + 0.5 * inverse_2
                                 + inverse * inverse_2 * (1.0 / 6.0 - inverse_2 * (1.0 / 30.0 - inverse_2 * (1.0 / 42.0
                                 - inverse_2 * (1.0 / 30.0 - inverse_2 * (5.0 / 66.0 - inverse_2 * (691.0 / 2730.0
                                 - inverse_2 * 7.0 / 6.0))))))
                                 + trigamma_shift;
        }

}

// ---- F = 2F1(1, s; r; x) = sum over n of (s)_n / (r)_n x^n and F_r, and (with_derivatives) the other derivatives the
//      median and its Jacobian need, in one pass over the terms:
//      series = (F, F_s, F_r, F_rr, F_sr, x F_x, x F_rx)

template <bool with_derivatives>
inline void cutpoint_hypergeometric_series(const double s,
                                           const double r,
                                           const double x,
                                           double* series) {


        double term = 1.0;
        double sum_of_inverse_s = 0.0;
        double sum_of_inverse_r = 0.0;
        double sum_of_inverse_r_squared = 0.0;
        double F = 0.0;
        double F_s = 0.0;
        double F_r = 0.0;
        double F_rr = 0.0;
        double F_sr = 0.0;
        double x_F_x = 0.0;
        double x_F_rx = 0.0;
        for (int n = 0; n < 1000000; ++n) {
                F += term;
                F_r -= term * sum_of_inverse_r;
                if constexpr (with_derivatives) {
                        F_s += term * sum_of_inverse_s;
                        F_rr += term * (sum_of_inverse_r * sum_of_inverse_r + sum_of_inverse_r_squared);
                        F_sr -= term * sum_of_inverse_s * sum_of_inverse_r;
                        x_F_x += term * n;
                        x_F_rx -= term * sum_of_inverse_r * n;
                }
                const double s_n = s + n;
                const double r_n = r + n;
                const double weight = with_derivatives
                                      ? 1.0 + n * (1.0 + sum_of_inverse_r) + sum_of_inverse_s * sum_of_inverse_r
                                        + sum_of_inverse_r * sum_of_inverse_r + sum_of_inverse_r_squared
                                      : 1.0 + sum_of_inverse_r;
                if (s_n * x < r_n && term * weight < 1e-17 * F) {
                        break;
                }
                const double inverse_r_n = 1.0 / r_n;
                sum_of_inverse_s += 1.0 / s_n;
                sum_of_inverse_r += inverse_r_n;
                sum_of_inverse_r_squared += inverse_r_n * inverse_r_n;
                term *= s_n * inverse_r_n * x;
        }

        series[0] = F;
        series[2] = F_r;
        if constexpr (with_derivatives) {
                series[1] = F_s;
                series[3] = F_rr;
                series[4] = F_sr;
                series[5] = x_F_x;
                series[6] = x_F_rx;
        }

}

// ---- log median u of Beta(a, b), a <= b (so u <= log(1 / 2)): Newton on
//      g(u) = log I_{exp(u)}(a, b) + log(2) = a u + b log(1 - exp(u)) - log(a) - lbeta(a, b) + log(F) + log(2),
//      g'(u) = a / ((1 - exp(u)) F), kept inside the bracket of the root

inline double cutpoint_beta_median_log(const double a,
                                       const double b,
                                       const double log_a,
                                       const double lbeta) {


        double upper_u = CUTPOINT_LOG_HALF;
        double lower_u = -std::numeric_limits<double>::infinity();
        double u = a >= 1.0 ? nico_log((a - 1.0 / 3.0) / (a + b - 2.0 / 3.0), nullptr)
                            : (CUTPOINT_LOG_HALF + log_a + lbeta) / a;
        if (!(u < upper_u)) {
                u = upper_u;
        }
        double series[7];
        for (int iteration = 0; iteration < 200; ++iteration) {
                const double x = nico_exp(u, nullptr);
                const double log_1m_x = nico_log1m_exp(u, nullptr);
                cutpoint_hypergeometric_series<false>(a + b, a + 1.0, x, series);
                const double g = a * u + b * log_1m_x - log_a - lbeta + nico_log(series[0], nullptr)
                                 - CUTPOINT_LOG_HALF;
                if (g == 0.0) {
                        return u;
                }
                if (g > 0.0) {
                        upper_u = u;
                } else {
                        lower_u = u;
                }
                double next_u = u - g * nico_exp(log_1m_x, nullptr) * series[0] / a;
                if (!(next_u < upper_u) || !(next_u > lower_u)) {
                        next_u = std::isinf(lower_u) ? upper_u - 2.0 * (upper_u - u) - 1.0 : 0.5 * (lower_u + upper_u);
                }
                if (std::abs(next_u - u) <= 4e-16 * (1.0 + std::abs(u))) {
                        return next_u;
                }
                u = next_u;
        }

        return u;

}

}

// ---- Median of the Dirichlet cutpoint (the median of Beta(kappa P_lower, kappa P_upper) on the link scale) and
//      the log of its Jacobian, with every partial derivative, in one pass:
//      values = (cutpoint, log J); partials(output, input), inputs (P_lower, P_upper, kappa)

template <bool with_partials>
inline void nico_dirichlet_cutpoint_median_and_log_J_values_and_partials(const double P_lower,
                                                                         const double P_upper,
                                                                         const double kappa,
                                                                         const int use_probit_link,
                                                                         Eigen::Matrix<double, 2, 1>& values,
                                                                         Eigen::Matrix<double, 2, 3, Eigen::DontAlign>& partials) {


        const bool reflect = P_lower > P_upper;
        const double P_a = reflect ? P_upper : P_lower;
        const double P_b = reflect ? P_lower : P_upper;
        const double a = kappa * P_a;
        const double b = kappa * P_b;
        const double log_kappa = nico::math::nico_log(kappa, nullptr);
        const double log_a = nico::math::nico_log(a, nullptr);

        double lgamma_a_plus_1, digamma_a_plus_1, trigamma_a_plus_1;
        double lgamma_b, digamma_b, trigamma_b;
        double lgamma_total, digamma_total;
        nico::math::cutpoint_lgamma_digamma_trigamma(a + 1.0, &lgamma_a_plus_1, &digamma_a_plus_1, &trigamma_a_plus_1);
        nico::math::cutpoint_lgamma_digamma_trigamma(b, &lgamma_b, &digamma_b, &trigamma_b);
        nico::math::cutpoint_lgamma_digamma_trigamma(a + b, &lgamma_total, &digamma_total, nullptr);
        const double lbeta = lgamma_a_plus_1 - log_a + lgamma_b - lgamma_total;

        const double u = nico::math::cutpoint_beta_median_log(a, b, log_a, lbeta);
        const double x = nico::math::nico_exp(u, nullptr);
        const double log_1m_x = nico::math::nico_log1m_exp(u, nullptr);

        double series[7];
        nico::math::cutpoint_hypergeometric_series<with_partials>(a + b, a + 1.0, x, series);
        const double F = series[0];
        const double score_difference = log_1m_x - u + digamma_a_plus_1 - digamma_b - series[2] / F;

        double cutpoint;
        double log_link_derivative;
        if (use_probit_link == 1) {
                cutpoint = nico::math::nico_std_normal_log_qf_value(u, nullptr);
                if (cutpoint < -nico::math::reference::NORMAL_TAIL_THRESH) {
                        double fraction = -cutpoint;
                        for (int k = nico::math::reference::MILLS_CF_DEPTH; k >= 1; --k) {
                                fraction = -cutpoint + static_cast<double>(k) / fraction;
                        }
                        log_link_derivative = -nico::math::nico_log(fraction, nullptr);
                } else {
                        log_link_derivative = u + 0.5 * cutpoint * cutpoint + nico::math::CUTPOINT_HALF_LOG_TWO_PI;
                }
        } else {
                cutpoint = u - log_1m_x;
                log_link_derivative = -log_1m_x;
        }

        values[0] = reflect ? -cutpoint : cutpoint;
        values[1] = log_kappa + nico::math::nico_log(score_difference, nullptr) + nico::math::CUTPOINT_LOG_HALF + lbeta
                    - a * u - (b - 1.0) * log_1m_x + log_link_derivative;

        if constexpr (with_partials) {
                const double F_s = series[1];
                const double F_r = series[2];
                const double F_rr = series[3];
                const double F_sr = series[4];
                const double x_F_x = series[5];
                const double x_F_rx = series[6];
                const double inverse_F_2 = 1.0 / (F * F);
                const double x_over_1m_x = nico::math::nico_exp(u - log_1m_x, nullptr);

                const double score_a = u - digamma_a_plus_1 + digamma_total + (F_s + F_r) / F;
                const double multiplier = nico::math::nico_exp(nico::math::CUTPOINT_LOG_HALF + lbeta - a * u
                                                               - (b - 1.0) * log_1m_x, nullptr);
                const double du_da = -score_a * multiplier;
                const double du_db = -(score_a + score_difference) * multiplier;

                const double link_derivative = nico::math::nico_exp(log_link_derivative, nullptr);
                const double d_log_link_derivative_du = use_probit_link == 1 ? 1.0 + cutpoint * link_derivative
                                                                             : x_over_1m_x;
                const double sign = reflect ? -1.0 : 1.0;

                const double d_difference_da = trigamma_a_plus_1 - ((F_sr + F_rr) * F - F_r * (F_s + F_r)) * inverse_F_2;
                const double d_difference_db = -trigamma_b - (F_sr * F - F_r * F_s) * inverse_F_2;
                const double d_difference_du = -x_over_1m_x - 1.0 - (x_F_rx * F - F_r * x_F_x) * inverse_F_2;

                const double dH_da = d_difference_da / score_difference + digamma_a_plus_1 - 1.0 / a - digamma_total - u;
                const double dH_db = d_difference_db / score_difference + digamma_b - digamma_total - log_1m_x;
                const double dH_du = d_difference_du / score_difference - a + (b - 1.0) * x_over_1m_x
                                     + d_log_link_derivative_du;

                const double d_cutpoint_da = sign * link_derivative * du_da;
                const double d_cutpoint_db = sign * link_derivative * du_db;
                const double d_log_J_da = dH_da + dH_du * du_da;
                const double d_log_J_db = dH_db + dH_du * du_db;

                const int column_a = reflect ? 1 : 0;
                const int column_b = reflect ? 0 : 1;
                partials(0, column_a) = d_cutpoint_da * kappa;
                partials(0, column_b) = d_cutpoint_db * kappa;
                partials(0, 2) = d_cutpoint_da * P_a + d_cutpoint_db * P_b;
                partials(1, column_a) = d_log_J_da * kappa;
                partials(1, column_b) = d_log_J_db * kappa;
                partials(1, 2) = d_log_J_da * P_a + d_log_J_db * P_b + 1.0 / kappa;
        }

}

template <typename T_lower,
          typename T_upper,
          typename T_kappa>
inline Eigen::Matrix<stan::return_type_t<T_lower, T_upper, T_kappa>, -1, 1>
nico_dirichlet_cutpoint_median_and_log_J(const T_lower& P_lower,
                                      const T_upper& P_upper,
                                      const T_kappa& kappa,
                                      const int use_probit_link,
                                      std::ostream* pstream__) {

        using T = stan::return_type_t<T_lower, T_upper, T_kappa>;
        Eigen::Matrix<double, 2, 1> values;
        Eigen::Matrix<double, 2, 3, Eigen::DontAlign> partials;
        (void) pstream__;
        if constexpr (!stan::is_var<T>::value) {
                static_assert(std::is_same<T, double>::value, "double or var arguments only");
                nico_dirichlet_cutpoint_median_and_log_J_values_and_partials<false>(
                    P_lower, P_upper, kappa, use_probit_link, values, partials);
                Eigen::Matrix<double, -1, 1> result(2);
                result[0] = values[0];
                result[1] = values[1];
                return result;
        } else {
                nico_dirichlet_cutpoint_median_and_log_J_values_and_partials<true>(
                    stan::math::value_of(P_lower), stan::math::value_of(P_upper), stan::math::value_of(kappa),
                    use_probit_link, values, partials);
                Eigen::Matrix<T, -1, 1> result(2);
                result[0] = stan::math::var(new stan::math::vari(values[0], false));
                result[1] = stan::math::var(new stan::math::vari(values[1], false));
                const stan::math::var median = result[0];
                const stan::math::var log_J = result[1];
                stan::math::reverse_pass_callback([P_lower, P_upper, kappa, median, log_J, partials]() {

                        if constexpr (stan::is_var<T_lower>::value)
                                P_lower.adj() += median.adj() * partials(0, 0) + log_J.adj() * partials(1, 0);
                        if constexpr (stan::is_var<T_upper>::value)
                                P_upper.adj() += median.adj() * partials(0, 1) + log_J.adj() * partials(1, 1);
                        if constexpr (stan::is_var<T_kappa>::value)
                                kappa.adj() += median.adj() * partials(0, 2) + log_J.adj() * partials(1, 2);

                });
                return result;
        }

}
