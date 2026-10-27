#pragma once

#include "stan_external_scalar_functions.hpp"

namespace nico::math {

inline double nico_log_diff_exp_value(const double x,
                                      const double y,
                                      std::ostream* pstream__) {

    if (x <= y) {
        return (x < stan::math::INFTY && x == y) ? stan::math::NEGATIVE_INFTY : stan::math::NOT_A_NUMBER;
    }
    return x + nico_log1m_exp(y - x, pstream__);

}

template <typename T_x, typename T_y>
inline stan::return_type_t<T_x, T_y>
nico_log_diff_exp_from_value(const T_x& x,
                              const T_y& y,
                              const double value,
                              std::ostream* pstream__) {

    static_assert(std::is_arithmetic<T_x>::value || stan::is_var<T_x>::value,
                  "The first argument must be an arithmetic or reverse-mode scalar.");
    static_assert(std::is_arithmetic<T_y>::value || stan::is_var<T_y>::value,
                  "The second argument must be an arithmetic or reverse-mode scalar.");
    if constexpr (stan::is_constant_all<T_x, T_y>::value) {
        return value;
    } else {
        const double x_value = stan::math::value_of(x);
        const double y_value = stan::math::value_of(y);
        auto ops_partials = stan::math::make_partials_propagator(x, y);
        if constexpr (!stan::is_constant_all<T_x>::value) {
            if constexpr (stan::is_constant_all<T_y>::value) {
                if (value == stan::math::NEGATIVE_INFTY) {
                    stan::math::partials<0>(ops_partials)[0]
                        = (y_value == stan::math::NEGATIVE_INFTY) ? 1.0 : stan::math::INFTY;
                } else {
                    stan::math::partials<0>(ops_partials)[0] = -1.0 / nico_expm1(y_value - x_value, pstream__);
                }
            } else {
                stan::math::partials<0>(ops_partials)[0] = -1.0 / nico_expm1(y_value - x_value, pstream__);
            }
        }
        if constexpr (!stan::is_constant_all<T_y>::value) {
            if constexpr (stan::is_constant_all<T_x>::value) {
                if (value == stan::math::NEGATIVE_INFTY) {
                    stan::math::partials<1>(ops_partials)[0] = -stan::math::INFTY;
                } else {
                    stan::math::partials<1>(ops_partials)[0] = -1.0 / nico_expm1(x_value - y_value, pstream__);
                }
            } else {
                stan::math::partials<1>(ops_partials)[0] = -1.0 / nico_expm1(x_value - y_value, pstream__);
            }
        }
        return ops_partials.build(value);
    }

}

inline double nico_log_diff_exp(const double x,
                                const double y,
                                std::ostream* pstream__) {

    return nico_log_diff_exp_value(x, y, pstream__);

}

inline stan::math::var nico_log_diff_exp(const stan::math::var& x,
                                        const stan::math::var& y,
                                        std::ostream* pstream__) {

    return nico_log_diff_exp_from_value(x, y, nico_log_diff_exp_value(x.val(), y.val(), pstream__), pstream__);

}

inline stan::math::var nico_log_diff_exp(const stan::math::var& x,
                                        const double y,
                                        std::ostream* pstream__) {

    return nico_log_diff_exp_from_value(x, y, nico_log_diff_exp_value(x.val(), y, pstream__), pstream__);

}

inline stan::math::var nico_log_diff_exp(const double x,
                                        const stan::math::var& y,
                                        std::ostream* pstream__) {

    return nico_log_diff_exp_from_value(x, y, nico_log_diff_exp_value(x, y.val(), pstream__), pstream__);

}

template <typename T_x, typename T_y>
inline Eigen::Matrix<stan::return_type_t<T_x, T_y>, Eigen::Dynamic, 1>
nico_log_diff_exp(const Eigen::Matrix<T_x, Eigen::Dynamic, 1>& x,
                   const Eigen::Matrix<T_y, Eigen::Dynamic, 1>& y,
                   std::ostream* pstream__) {

    stan::math::check_matching_dims("Binary function", "x", x, "y", y);
    const Eigen::Matrix<double, Eigen::Dynamic, 1> x_values = stan::math::value_of(x);
    const Eigen::Matrix<double, Eigen::Dynamic, 1> y_values = stan::math::value_of(y);
    const Eigen::Matrix<double, Eigen::Dynamic, 1> differences = y_values - x_values;
    const Eigen::Matrix<double, Eigen::Dynamic, 1> log_mass = nico_log1m_exp(differences, pstream__);
    Eigen::Matrix<stan::return_type_t<T_x, T_y>, Eigen::Dynamic, 1> result(x.size());
    for (Eigen::Index i = 0; i < x.size(); ++i) {
        const double value = x_values[i] <= y_values[i]
            ? nico_log_diff_exp_value(x_values[i], y_values[i], pstream__)
            : x_values[i] + log_mass[i];
        result[i] = nico_log_diff_exp_from_value(x[i], y[i], value, pstream__);
    }
    return result;

}

}  // namespace nico::math

using nico::math::nico_log_diff_exp;






















