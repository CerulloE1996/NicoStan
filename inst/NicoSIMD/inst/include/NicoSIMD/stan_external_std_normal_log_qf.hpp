#pragma once

// Normal quantiles from log probabilities, with the existing log-space polynomial.

#include "stan_external_log_diff_exp.hpp"
#include "stan_external_log_sum_exp_with_nico_exponential.hpp"
#include "stan_external_fused_normal_lpdf.hpp"
#include <stan/math/prim/functor/apply_scalar_unary.hpp>

namespace nico::math {

inline double nico_std_normal_log_qf_log_complement(
    const double x,
    std::ostream* pstream__) {

    constexpr double ln2 = 0.693147180559945309417232121458;
    if (x < -ln2) {

        return nico_log1p(
            -nico_exp(
                x,
                pstream__),
            pstream__);

    }
    const double u = x * x;
    double e = -3617.0 / 170729965486080000.0;
    e = std::fma(e, u, 1.0 / 1046139494400.0);
    e = std::fma(e, u, -691.0 / 15692092416000.0);
    e = std::fma(e, u, 1.0 / 479001600.0);
    e = std::fma(e, u, -1.0 / 9676800.0);
    e = std::fma(e, u, 1.0 / 181440.0);
    e = std::fma(e, u, -1.0 / 2880.0);
    e = std::fma(e, u, 1.0 / 24.0);
    const double h = std::fma(u, e, 0.5 * x);

    return nico_log(
        -x,
        pstream__) + h;

}

inline double nico_std_normal_log_qf_log_difference(
    const double x,
    const double y,
    std::ostream* pstream__) {

    if (x <= y) {

        return (x < stan::math::INFTY && x == y) ? stan::math::NEGATIVE_INFTY : stan::math::NOT_A_NUMBER;

    }
    return x + nico_std_normal_log_qf_log_complement(
        y - x,
        pstream__);

}


inline double nico_std_normal_log_qf_value(
    const double log_p,
    std::ostream* pstream__) {

    using stan::math::INFTY;
    using stan::math::LOG_HALF;
    using stan::math::LOG_TEN;
    using stan::math::LOG_TWO;
    using stan::math::NEGATIVE_INFTY;

    stan::math::check_not_nan(
        "std_normal_log_qf",
        "Log probability variable",
        log_p);
    stan::math::check_less_or_equal(
        "std_normal_log_qf",
        "Probability variable",
        log_p,
        0);

    if (log_p == NEGATIVE_INFTY) {

        return NEGATIVE_INFTY;

    }
    if (log_p == 0) {

        return INFTY;

    }

    static constexpr double log_a[8]
        = {1.2199838032983212, 4.8914137334471356, 7.5865960847956080,
           9.5274618535358388, 10.734698580862359, 11.116406781896242,
           10.417226196842595, 7.8276718012189362};
    static constexpr double log_b[8] = {0.,
                                       3.7451021830139207,
                                       6.5326064640478618,
                                       8.5930788436817044,
                                       9.9624069236663077,
                                       10.579180688621286,
                                       10.265665328832871,
                                       8.5614962136628454};
    static constexpr double log_c[8]
        = {0.3530744474482423, 1.5326298343683388, 1.7525849400614634,
           1.2941374937060454, 0.2393776640901312, -1.419724057885092,
           -3.784340465764968, -7.163234779359426};
    static constexpr double log_d[8] = {0.,
                                       0.7193954734947205,
                                       0.5166395879845317,
                                       -0.371400933927844,
                                       -1.909840708457214,
                                       -4.186547581055928,
                                       -7.509976771225415,
                                       -20.67376157385924};
    static constexpr double log_e[8]
        = {1.8958048169567149, 1.6981417567726154, 0.5793212339927351,
           -1.215503791936417, -3.629396584023968, -6.690500273261249,
           -10.51540298415323, -15.41979457491781};
    static constexpr double log_f[8] = {0.,
                                       -0.511105318617135,
                                       -1.988286302259815,
                                       -4.208049039384857,
                                       -7.147448611626374,
                                       -10.89973190740069,
                                       -15.76637472711685,
                                       -33.82373901099482};

    double log_q = log_p <= LOG_HALF
        ? nico_std_normal_log_qf_log_difference(
            LOG_HALF,
            log_p,
            pstream__)
        : nico_std_normal_log_qf_log_difference(
            log_p,
            LOG_HALF,
            pstream__);
    int log_q_sign = log_p <= LOG_HALF ? -1 : 1;
    double log_r = log_q_sign == -1 ? log_p : nico_std_normal_log_qf_log_complement(
        log_p,
        pstream__);

    if (stan::math::is_inf(log_r)) {

        return 0;

    }

    double log_inner_r;
    double log_pre_mult;
    const double* num_ptr;
    const double* den_ptr;

    static constexpr double LOG_FIVE = LOG_TEN - LOG_TWO;
    static constexpr double LOG_16 = LOG_TWO * 4;
    static constexpr double LOG_425 = 6.0520891689244171729;
    static constexpr double LOG_425_OVER_1000 = LOG_425 - LOG_TEN * 3;

    if (log_q <= LOG_425_OVER_1000) {

        log_inner_r = nico_std_normal_log_qf_log_difference(
            LOG_425_OVER_1000 * 2,
            log_q * 2,
            pstream__);
        log_pre_mult = log_q;
        num_ptr = &log_a[0];
        den_ptr = &log_b[0];

    } else {

        double log_temp_r = nico_log(
            -log_r,
            pstream__) / 2.0;
        if (log_temp_r <= LOG_FIVE) {

            log_inner_r = nico_std_normal_log_qf_log_difference(
                log_temp_r,
                LOG_16 - LOG_TEN,
                pstream__);
            num_ptr = &log_c[0];
            den_ptr = &log_d[0];

        } else {

            log_inner_r = nico_std_normal_log_qf_log_difference(
                log_temp_r,
                LOG_FIVE,
                pstream__);
            num_ptr = &log_e[0];
            den_ptr = &log_f[0];

        }
        log_pre_mult = 0.0;

    }

    Eigen::VectorXd log_r_pow
        = Eigen::ArrayXd::LinSpaced(
            8,
            0,
            7) * log_inner_r - LOG_TEN;
    Eigen::Map<const Eigen::VectorXd> num_map(
        num_ptr,
        8);
    Eigen::Map<const Eigen::VectorXd> den_map(
        den_ptr,
        8);
    double log_result
        = nico_log_sum_exp(
            log_r_pow + num_map,
            pstream__)
          - nico_log_sum_exp(
            log_r_pow + den_map,
            pstream__);
    return log_q_sign * nico_exp(
        log_pre_mult + log_result,
        pstream__);

}

template <typename T, stan::require_stan_scalar_t<T>* = nullptr>
inline auto nico_std_normal_log_qf(
    const T& log_p,
    std::ostream* pstream__) {

    static_assert(stan::is_var_or_arithmetic<T>::value,
                  "The log quantile accepts arithmetic and reverse-mode scalars.");
    if constexpr (stan::is_var<T>::value) {

        const double value = nico_std_normal_log_qf_value(
            log_p.val(),
            pstream__);
        const double derivative = nico_exp(
            log_p.val() - nico_normal_lpdf<false>(
                value,
                0.0,
                1.0,
                pstream__),
            pstream__);
        return stan::math::make_callback_var(
            value,
            [log_p, derivative](auto& result_node) mutable {

                log_p.adj() += result_node.adj() * derivative;

            });

    } else {

        return nico_std_normal_log_qf_value(
            static_cast<double>(log_p),
            pstream__);

    }

}

struct nico_std_normal_log_qf_fun {

    template <typename T>
    static inline auto fun(T&& x) {

        return nico_std_normal_log_qf(
            std::forward<T>(x),
            nullptr);

    }

};

template <
    typename T,
    stan::require_all_not_nonscalar_prim_or_rev_kernel_expression_t<T>* = nullptr,
    stan::require_container_t<T>* = nullptr,
    stan::require_not_var_matrix_t<T>* = nullptr>
inline auto nico_std_normal_log_qf(
    T&& x,
    std::ostream* /*pstream__*/) {

    static_assert(stan::is_var_or_arithmetic<stan::scalar_type_t<T>>::value,
                  "The log quantile accepts arithmetic and reverse-mode containers.");
    return stan::math::apply_scalar_unary<nico_std_normal_log_qf_fun, T>::apply(std::forward<T>(x));

}

}  // namespace nico::math

using nico::math::nico_std_normal_log_qf;

