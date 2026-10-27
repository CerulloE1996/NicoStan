#pragma once

#include "stan_external_functions.hpp"

#include <cmath>
#include <ostream>
#include <type_traits>

#include "math/nico_double_exp_log.hpp"
#include "math/nico_double_probability.hpp"

namespace nico::math {

template <typename Value_kernel, typename Derivative>
inline var nico_apply_scalar_kernel(const var& x,
                                    Value_kernel value_kernel,
                                    Derivative derivative) {

    const double operand = x.val();
    const double result = value_kernel(operand);
    double partial = 0.0;
    derivative(&operand, &result, 1, &partial);

    return stan::math::make_callback_var(result, [x, partial](auto& result_node) {

        x.adj() += result_node.adj() * partial;

    });

}

inline double nico_log1m_inv_logit_value(const double x) {

    return nico_log_inv_logit(-x);

}

inline double nico_exp(const double x,
                       std::ostream* /*pstream__*/) {

    return nico::math::nico_exp_1(x);

}

inline var nico_exp(const var& x,
                    std::ostream* /*pstream__*/) {

    return nico_apply_scalar_kernel(
        x,
        &nico::math::nico_exp_1,
        &nico::math::d_exp);

}

inline double nico_log(const double x,
                       std::ostream* /*pstream__*/) {

    return nico::math::nico_log_1_double_precision(x);

}

inline var nico_log(const var& x,
                    std::ostream* /*pstream__*/) {

    return nico_apply_scalar_kernel(
        x,
        &nico::math::nico_log_1_double_precision,
        &nico::math::d_log);

}

inline double nico_log1m(const double x,
                         std::ostream* /*pstream__*/) {

    return nico::math::nico_log1m_1(x);

}

inline var nico_log1m(const var& x,
                      std::ostream* /*pstream__*/) {

    return nico_apply_scalar_kernel(
        x,
        &nico::math::nico_log1m_1,
        &nico::math::d_log1m);

}

inline double nico_log1p(const double x,
                         std::ostream* /*pstream__*/) {

    return nico::math::nico_log1p_1_double_precision(x);

}

inline var nico_log1p(const var& x,
                      std::ostream* /*pstream__*/) {

    return nico_apply_scalar_kernel(
        x,
        &nico::math::nico_log1p_1_double_precision,
        &nico::math::d_log1p);

}

inline double nico_log1p_exp(const double x,
                             std::ostream* /*pstream__*/) {

    return nico::math::nico_log1p_exp_1(x);

}

inline var nico_log1p_exp(const var& x,
                          std::ostream* /*pstream__*/) {

    return nico_apply_scalar_kernel(
        x,
        &nico::math::nico_log1p_exp_1,
        &nico::math::d_log1p_exp);

}

inline double nico_log1m_exp(const double x,
                             std::ostream* /*pstream__*/) {

    return nico::math::nico_log1m_exp_1(x);

}

inline var nico_log1m_exp(const var& x,
                          std::ostream* /*pstream__*/) {

    return nico_apply_scalar_kernel(
        x,
        &nico::math::nico_log1m_exp_1,
        &nico::math::d_log1m_exp);

}

inline double nico_inv_logit(const double x,
                             std::ostream* /*pstream__*/) {

    return nico::math::nico_inv_logit(x);

}

inline var nico_inv_logit(const var& x,
                          std::ostream* /*pstream__*/) {

    return nico_apply_scalar_kernel(
        x,
        static_cast<double (*)(double)>(&nico::math::nico_inv_logit),
        &nico::math::d_inv_logit);

}

inline double nico_log_inv_logit(const double x,
                                 std::ostream* /*pstream__*/) {

    return nico::math::nico_log_inv_logit(x);

}

inline var nico_log_inv_logit(const var& x,
                              std::ostream* /*pstream__*/) {

    return nico_apply_scalar_kernel(
        x,
        static_cast<double (*)(double)>(&nico::math::nico_log_inv_logit),
        &nico::math::d_log_inv_logit);

}

inline double nico_log1m_inv_logit(const double x,
                                   std::ostream* /*pstream__*/) {

    return nico::math::nico_log1m_inv_logit_value(x);

}

inline var nico_log1m_inv_logit(const var& x,
                                std::ostream* /*pstream__*/) {

    return nico_apply_scalar_kernel(
        x,
        &nico::math::nico_log1m_inv_logit_value,
        &nico::math::d_log1m_inv_logit);

}

inline double nico_Phi(const double x,
                       std::ostream* /*pstream__*/) {

    return nico::math::nico_Phi(x);

}

inline var nico_Phi(const var& x,
                    std::ostream* /*pstream__*/) {

    return nico_apply_scalar_kernel(
        x,
        static_cast<double (*)(double)>(&nico::math::nico_Phi),
        &nico::math::d_Phi);

}

inline double nico_Phi_approx(const double x,
                              std::ostream* /*pstream__*/) {

    return nico::math::nico_Phi_approx(x);

}

inline var nico_Phi_approx(const var& x,
                           std::ostream* /*pstream__*/) {

    return nico_apply_scalar_kernel(
        x,
        static_cast<double (*)(double)>(&nico::math::nico_Phi_approx),
        &nico::math::d_Phi_approx);

}

inline double nico_log_Phi(const double x,
                           std::ostream* /*pstream__*/) {

    return nico::math::nico_log_Phi(x);

}

inline var nico_log_Phi(const var& x,
                        std::ostream* /*pstream__*/) {

    return nico_apply_scalar_kernel(
        x,
        static_cast<double (*)(double)>(&nico::math::nico_log_Phi),
        &nico::math::d_log_Phi);

}

inline double nico_log_Phi_approx(const double x,
                                  std::ostream* /*pstream__*/) {

    return nico::math::nico_log_Phi_approx(x);

}

inline var nico_log_Phi_approx(const var& x,
                               std::ostream* /*pstream__*/) {

    return nico_apply_scalar_kernel(
        x,
        static_cast<double (*)(double)>(&nico::math::nico_log_Phi_approx),
        &nico::math::d_log_Phi_approx);

}

inline double nico_inv_Phi(const double x,
                           std::ostream* /*pstream__*/) {

    return nico::math::nico_inv_Phi_wo_checks(x);

}

inline var nico_inv_Phi(const var& x,
                        std::ostream* /*pstream__*/) {

    return nico_apply_scalar_kernel(
        x,
        &nico::math::nico_inv_Phi_wo_checks,
        &nico::math::d_inv_Phi);

}

inline double nico_inv_Phi_approx(const double x,
                                  std::ostream* /*pstream__*/) {

    return nico::math::nico_inv_Phi_approx(x);

}

inline var nico_inv_Phi_approx(const var& x,
                               std::ostream* /*pstream__*/) {

    return nico_apply_scalar_kernel(
        x,
        static_cast<double (*)(double)>(&nico::math::nico_inv_Phi_approx),
        &nico::math::faster_rev::d_inv_Phi_approx_from_z_and_p);

}

inline double nico_inv_Phi_approx_from_logit_prob(const double x,
                                                  std::ostream* /*pstream__*/) {

    return nico::math::nico_inv_Phi_approx_from_logit_prob_public(x);

}

inline var nico_inv_Phi_approx_from_logit_prob(const var& x,
                                               std::ostream* /*pstream__*/) {

    return nico_apply_scalar_kernel(
        x,
        &nico::math::nico_inv_Phi_approx_from_logit_prob_public,
        &nico::math::faster_rev::d_inv_Phi_approx_from_logit_prob_from_z);

}

inline double nico_logit(const double x,
                         std::ostream* /*pstream__*/) {

    return nico::math::nico_logit(x);

}

inline var nico_logit(const var& x,
                      std::ostream* /*pstream__*/) {

    return nico_apply_scalar_kernel(
        x,
        static_cast<double (*)(double)>(&nico::math::nico_logit),
        &nico::math::d_logit);

}

inline double nico_square(const double x,
                          std::ostream* /*pstream__*/) {

    return nico::math::nico_square(x);

}

inline var nico_square(const var& x,
                       std::ostream* /*pstream__*/) {

    return nico_apply_scalar_kernel(
        x,
        static_cast<double (*)(double)>(&nico::math::nico_square),
        &nico::math::d_square);

}

inline double nico_sqrt(const double x,
                        std::ostream* /*pstream__*/) {

    return nico::math::nico_sqrt(x);

}

inline var nico_sqrt(const var& x,
                     std::ostream* /*pstream__*/) {

    return nico_apply_scalar_kernel(
        x,
        static_cast<double (*)(double)>(&nico::math::nico_sqrt),
        &nico::math::d_sqrt);

}

inline double nico_expm1(const double x,
                         std::ostream* /*pstream__*/) {

    return nico::math::nico_expm1_value(x);

}

inline var nico_expm1(const var& x,
                      std::ostream* /*pstream__*/) {

    return nico_apply_scalar_kernel(
        x,
        &nico::math::nico_expm1_value,
        &nico::math::d_expm1);

}

inline double nico_tanh(const double x,
                        std::ostream* /*pstream__*/) {

    return nico::math::nico_tanh_value(x);

}

inline var nico_tanh(const var& x,
                     std::ostream* /*pstream__*/) {

    return nico_apply_scalar_kernel(
        x,
        &nico::math::nico_tanh_value,
        &nico::math::d_tanh);

}

template <typename Log_p, typename Log_1m_p>
inline var nico_apply_scalar_inv_Phi_from_log_p(const Log_p& log_p,
                                                const Log_1m_p& log_1m_p,
                                                const double value) {

    const double log_p_value = stan::math::value_of(log_p);
    const double log_1m_p_value = stan::math::value_of(log_1m_p);
    const double probability = nico::math::nico_exp_1(log_p_value);
    const bool uses_log_p = (std::abs(probability - 0.5) <= 0.425) ||
                            (log_p_value < log_1m_p_value);
    const double log_inv_phi = std::fma(
        0.5,
        value * value,
        -NICO_DOUBLE_PROBABILITY_NEG_HALF_LOG_2PI);
    const double log_side = uses_log_p ? log_p_value : log_1m_p_value;
    const double magnitude = nico::math::nico_exp_1(log_side + log_inv_phi);
    const double derivative_of_log_p = uses_log_p ? magnitude : 0.0;
    const double derivative_of_log_1m_p = uses_log_p ? 0.0 : -magnitude;

    return stan::math::make_callback_var(
        value,
        [log_p, log_1m_p, derivative_of_log_p, derivative_of_log_1m_p](auto& result_node) mutable {

            const double adjoint = result_node.adj();

            if constexpr (stan::is_var<std::decay_t<Log_p>>::value) {
                log_p.adj() += adjoint * derivative_of_log_p;
            }

            if constexpr (stan::is_var<std::decay_t<Log_1m_p>>::value) {
                log_1m_p.adj() += adjoint * derivative_of_log_1m_p;
            }

        });

}

inline double nico_inv_Phi_from_log_p(const double log_p,
                                      const double log_1m_p,
                                      std::ostream* /*pstream__*/) {

    return nico::math::nico_inv_Phi_from_log_p_wo_checks(log_p, log_1m_p);

}

inline var nico_inv_Phi_from_log_p(const var& log_p,
                                   const double log_1m_p,
                                   std::ostream* /*pstream__*/) {

    const double value = nico::math::nico_inv_Phi_from_log_p_wo_checks(log_p.val(), log_1m_p);
    return nico_apply_scalar_inv_Phi_from_log_p(log_p, log_1m_p, value);

}

inline var nico_inv_Phi_from_log_p(const double log_p,
                                   const var& log_1m_p,
                                   std::ostream* /*pstream__*/) {

    const double value = nico::math::nico_inv_Phi_from_log_p_wo_checks(log_p, log_1m_p.val());
    return nico_apply_scalar_inv_Phi_from_log_p(log_p, log_1m_p, value);

}

inline var nico_inv_Phi_from_log_p(const var& log_p,
                                   const var& log_1m_p,
                                   std::ostream* /*pstream__*/) {

    const double value = nico::math::nico_inv_Phi_from_log_p_wo_checks(log_p.val(), log_1m_p.val());
    return nico_apply_scalar_inv_Phi_from_log_p(log_p, log_1m_p, value);

}

}  // namespace nico::math

using nico::math::nico_Phi;
using nico::math::nico_Phi_approx;
using nico::math::nico_exp;
using nico::math::nico_expm1;
using nico::math::nico_inv_Phi;
using nico::math::nico_inv_Phi_approx;
using nico::math::nico_inv_Phi_approx_from_logit_prob;
using nico::math::nico_inv_Phi_from_log_p;
using nico::math::nico_inv_logit;
using nico::math::nico_log;
using nico::math::nico_log1m;
using nico::math::nico_log1m_exp;
using nico::math::nico_log1m_inv_logit;
using nico::math::nico_log1p;
using nico::math::nico_log1p_exp;
using nico::math::nico_log_Phi;
using nico::math::nico_log_Phi_approx;
using nico::math::nico_log_inv_logit;
using nico::math::nico_logit;
using nico::math::nico_square;
using nico::math::nico_sqrt;
using nico::math::nico_tanh;






















