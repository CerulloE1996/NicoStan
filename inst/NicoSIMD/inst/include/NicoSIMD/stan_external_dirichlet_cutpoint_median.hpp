#pragma once

#include <stan/math/mix.hpp>
#include "math/nico_double_probability_reference.hpp"
#include "stan_external_scalar_functions.hpp"
#include "stan_external_std_normal_log_qf.hpp"
#include <cmath>
#include <ostream>

namespace nico::math {

template <typename T>
inline auto cutpoint_log(const T& x) {

        return nico_log(x, nullptr);

}

template <typename T>
inline stan::math::fvar<T> cutpoint_log(const stan::math::fvar<T>& x) {

        return {cutpoint_log(x.val_), x.d_ / x.val_};

}

template <typename T>
inline auto cutpoint_exp(const T& x) {

        return nico_exp(x, nullptr);

}

template <typename T>
inline stan::math::fvar<T> cutpoint_exp(const stan::math::fvar<T>& x) {

        const T value = cutpoint_exp(x.val_);
        return {value, x.d_ * value};

}

template <typename T>
inline auto cutpoint_log1m_exp(const T& x) {

        return nico_log1m_exp(x, nullptr);

}

template <typename T>
inline stan::math::fvar<T> cutpoint_log1m_exp(const stan::math::fvar<T>& x) {

        const T value = cutpoint_log1m_exp(x.val_);
        return {value, -x.d_ * cutpoint_exp(x.val_ - value)};

}

}

// ---- Dirichlet cutpoint median and log derivative with respect to the cumulative probability.

template <typename T>
inline Eigen::Matrix<T, 2, 1>
nico_beta_median_log_p_scores(const T& a,
                              const T& b,
                              const T& log_p) {


        const T total = a + b;
        const T probability = nico::math::cutpoint_exp(log_p);
        const T series = stan::math::hypergeometric_2F1(1.0, total, a + 1.0, probability);
        const auto derivatives = stan::math::grad_2F1<true>(1.0,
                                                           total,
                                                           a + 1.0,
                                                           probability,
                                                           1e-14,
                                                           1000000);
        const T score_a = log_p - stan::math::digamma(a + 1.0) + stan::math::digamma(total)
                          + (std::get<1>(derivatives) + std::get<2>(derivatives)) / series;
        const T difference = nico::math::cutpoint_log1m_exp(log_p) - log_p
                             + stan::math::digamma(a + 1.0) - stan::math::digamma(b)
                             - std::get<2>(derivatives) / series;

        Eigen::Matrix<T, 2, 1> result;
        result[0] = score_a;
        result[1] = difference;

        return result;

}

inline double nico_beta_median_log_p(const double a,
                                     const double b) {


        const double probability = stan::math::inv_inc_beta(a, b, 0.5);
        if (probability > 0.0) {
                return nico::math::cutpoint_log(probability);
        }

        const long double extended_probability = boost::math::ibeta_inv(static_cast<long double>(a),
                                                                         static_cast<long double>(b),
                                                                         0.5L,
                                                                         stan::math::boost_policy_t<0>());
        if (extended_probability > 0.0L) {
                return static_cast<double>(std::log(extended_probability));
        }

        return (stan::math::LOG_HALF + nico::math::cutpoint_log(a) + stan::math::lbeta(a, b)) / a;

}

inline double nico_beta_median_log_p_from_log_shapes(const double log_a,
                                                     const double log_b) {


        return nico_beta_median_log_p(nico::math::cutpoint_exp(log_a), nico::math::cutpoint_exp(log_b));

}

inline stan::math::var nico_beta_median_log_p_from_log_shapes(const stan::math::var& log_a,
                                                              const stan::math::var& log_b) {


        const double a = nico::math::cutpoint_exp(log_a.val());
        const double b = nico::math::cutpoint_exp(log_b.val());
        const double log_p = nico_beta_median_log_p(a, b);
        const auto scores = nico_beta_median_log_p_scores(a, b, log_p);
        const double log_multiplier = stan::math::LOG_HALF + stan::math::lbeta(a, b)
                                      - a * log_p - (b - 1.0) * nico::math::cutpoint_log1m_exp(log_p);
        const double derivative_a = -scores[0] * nico::math::cutpoint_exp(log_a.val() + log_multiplier);
        const double derivative_b = -(scores[0] + scores[1]) * nico::math::cutpoint_exp(log_b.val() + log_multiplier);

        return stan::math::make_callback_var(log_p,
                                             [log_a, log_b, derivative_a, derivative_b](auto& result) mutable {

                log_a.adj() += result.adj() * derivative_a;
                log_b.adj() += result.adj() * derivative_b;

        });

}

template <typename T>
inline stan::math::fvar<T>
nico_beta_median_log_p_from_log_shapes(const stan::math::fvar<T>& log_a,
                                       const stan::math::fvar<T>& log_b) {


        const T a = nico::math::cutpoint_exp(log_a.val_);
        const T b = nico::math::cutpoint_exp(log_b.val_);
        const T log_p = nico_beta_median_log_p_from_log_shapes(log_a.val_, log_b.val_);
        const auto scores = nico_beta_median_log_p_scores(a, b, log_p);
        const T log_multiplier = stan::math::LOG_HALF + stan::math::lbeta(a, b)
                                 - a * log_p - (b - 1.0) * nico::math::cutpoint_log1m_exp(log_p);
        const T derivative = -log_a.d_ * scores[0] * nico::math::cutpoint_exp(log_a.val_ + log_multiplier)
                             - log_b.d_ * (scores[0] + scores[1])
                               * nico::math::cutpoint_exp(log_b.val_ + log_multiplier);

        return stan::math::fvar<T>(log_p, derivative);

}

template <typename T>
inline T nico_normal_lower_log_Mills_ratio(const T& cutpoint,
                                           const T& log_p) {


        if (stan::math::value_of_rec(cutpoint) < -nico::math::reference::NORMAL_TAIL_THRESH) {
                T fraction = -cutpoint;
                for (int k = nico::math::reference::MILLS_CF_DEPTH; k >= 1; --k) {
                        fraction = -cutpoint + static_cast<double>(k) / fraction;
                }
                return -nico::math::cutpoint_log(fraction);
        }

        return log_p - stan::math::std_normal_lpdf(cutpoint);

}

inline double nico_normal_lower_quantile_from_log_p(const double log_p) {


        return nico::math::nico_std_normal_log_qf_value(log_p, nullptr);

}

inline stan::math::var nico_normal_lower_quantile_from_log_p(const stan::math::var& log_p) {


        const double cutpoint = nico::math::nico_std_normal_log_qf_value(log_p.val(), nullptr);
        const double derivative = nico::math::cutpoint_exp(nico_normal_lower_log_Mills_ratio(cutpoint, log_p.val()));

        return stan::math::make_callback_var(cutpoint, [log_p, derivative](auto& result) mutable {

                log_p.adj() += result.adj() * derivative;

        });

}

template <typename T>
inline stan::math::fvar<T>
nico_normal_lower_quantile_from_log_p(const stan::math::fvar<T>& log_p) {


        const T cutpoint = nico_normal_lower_quantile_from_log_p(log_p.val_);
        const T derivative = nico::math::cutpoint_exp(nico_normal_lower_log_Mills_ratio(cutpoint, log_p.val_));

        return stan::math::fvar<T>(cutpoint, log_p.d_ * derivative);

}

template <typename T_lower,
          typename T_upper,
          typename T_kappa>
inline Eigen::Matrix<stan::return_type_t<T_lower, T_upper, T_kappa>, -1, 1>
nico_dirichlet_cutpoint_median_and_log_J_evaluate(const T_lower& P_lower,
                                        const T_upper& P_upper,
                                        const T_kappa& kappa,
                                        const int use_probit_link,
                                        std::ostream* pstream__) {


        using T = stan::return_type_t<T_lower, T_upper, T_kappa>;

        const T lower(P_lower);
        const T upper(P_upper);
        const T concentration(kappa);
        const bool reflect = lower > upper;

        const T log_shape_a = nico::math::cutpoint_log(concentration) + nico::math::cutpoint_log(reflect ? upper : lower);
        const T log_shape_b = nico::math::cutpoint_log(concentration) + nico::math::cutpoint_log(reflect ? lower : upper);
        const T shape_a = nico::math::cutpoint_exp(log_shape_a);
        const T shape_b = nico::math::cutpoint_exp(log_shape_b);
        const T log_p = nico_beta_median_log_p_from_log_shapes(log_shape_a, log_shape_b);
        const T log_1m_p = nico::math::cutpoint_log1m_exp(log_p);
        const auto scores = nico_beta_median_log_p_scores(shape_a, shape_b, log_p);

        T cutpoint;
        T log_link_derivative;
        if (use_probit_link == 1) {
                cutpoint = nico_normal_lower_quantile_from_log_p(log_p);
                log_link_derivative = nico_normal_lower_log_Mills_ratio(cutpoint, log_p);
        } else {
                cutpoint = log_p - log_1m_p;
                log_link_derivative = -log_1m_p;
        }

        if (reflect) {
                cutpoint = -cutpoint;
        }

        Eigen::Matrix<T, -1, 1> result(2);
        result[0] = cutpoint;
        result[1] = nico::math::cutpoint_log(concentration) + nico::math::cutpoint_log(scores[1]) + stan::math::LOG_HALF
                    + stan::math::lbeta(shape_a, shape_b) - shape_a * log_p
                    - (shape_b - 1.0) * log_1m_p + log_link_derivative;
        (void) pstream__;

        return result;

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
        if constexpr (!stan::is_var<T>::value) {
                return nico_dirichlet_cutpoint_median_and_log_J_evaluate(
                    P_lower, P_upper, kappa, use_probit_link, pstream__);
        } else {
                const double lower = stan::math::value_of(P_lower);
                const double upper = stan::math::value_of(P_upper);
                const double concentration = stan::math::value_of(kappa);
                Eigen::Matrix<double, 2, 3, Eigen::DontAlign> partials;
                Eigen::Matrix<double, 2, 1> values;
                for (int direction = 0; direction < 3; ++direction) {
                        const auto evaluated = nico_dirichlet_cutpoint_median_and_log_J_evaluate(
                            stan::math::fvar<double>(lower, direction == 0 ? 1.0 : 0.0),
                            stan::math::fvar<double>(upper, direction == 1 ? 1.0 : 0.0),
                            stan::math::fvar<double>(concentration, direction == 2 ? 1.0 : 0.0),
                            use_probit_link, pstream__);
                        for (int output = 0; output < 2; ++output) {
                                values[output] = evaluated[output].val_;
                                partials(output, direction) = evaluated[output].d_;
                        }
                }
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























