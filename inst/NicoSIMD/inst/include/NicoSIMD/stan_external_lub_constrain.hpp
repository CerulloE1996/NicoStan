#pragma once
////
//// ---- lub_constrain of a matrix of vars with arithmetic bounds, with the NicoSIMD kernels ------------------------------------------
////
//// Only when NICOSIMD_SIMD_LUB_CONSTRAIN is defined. The same transform, Jacobian and adjoints as Stan math's
//// rev/constraint/lub_constrain.hpp (matrix, scalar bounds): inv_logit and log1p_exp of all the values go
//// through nico::math::k_inv_logit and k_log1p_exp, not one scalar exp / log1p per value. Stan's deserializer
//// reads a matrix<lower=a, upper=b> parameter through lub_constrain(x, a, b, lp), which finds these by ADL
//// (they are not templates, so they are taken over Stan's template for these argument types).
////
#ifdef NICOSIMD_SIMD_LUB_CONSTRAIN
#include <stan/math/rev.hpp>
#include <Eigen/Dense>
#include "stan_external_functions.hpp"

namespace stan {
namespace math {
namespace nico_lub_constrain_internal {

////
//// ---- The transform, and its log-Jacobian when lp is given ------------------------------------------------------------------------
////
inline Eigen::Matrix<var, -1, -1> fn_lub_constrain(const Eigen::Matrix<var, -1, -1>& x,
                                                   const double lb_val,
                                                   const double ub_val,
                                                   var* lp) {

        check_less("lub_constrain", "lb", lb_val, ub_val);
        arena_t<Eigen::Matrix<var, -1, -1>> arena_x = x;
        const int n = static_cast<int>(arena_x.size());
        const double diff = ub_val - lb_val;
        arena_t<Eigen::Matrix<double, -1, -1>> inv_logit_x = value_of(arena_x);
        ////
        //// ---- log(diff) + (-|x| - 2 log1p_exp(-|x|)), summed (the same terms and order as Stan's) ----------------------------------
        ////
        if (lp != nullptr) {

                Eigen::ArrayXd neg_abs_x = -(inv_logit_x.reshaped().array().abs());
                Eigen::ArrayXd log1p_exp_of_neg_abs_x = neg_abs_x;
                nico::math::k_log1p_exp(log1p_exp_of_neg_abs_x.data(), n);
                *lp += (std::log(diff) + (neg_abs_x - (2.0 * log1p_exp_of_neg_abs_x))).sum();

        }
        nico::math::k_inv_logit(inv_logit_x.data(), n);
        arena_t<Eigen::Matrix<var, -1, -1>> ret = (diff * inv_logit_x.array() + lb_val).matrix();
        ////
        //// ---- Adjoints ------------------------------------------------------------------------------------------------------------
        ////
        if (lp != nullptr) {

                var lp_after = *lp;
                reverse_pass_callback([arena_x, ret, lp_after, diff, inv_logit_x]() mutable {

                        arena_x.adj().array() += ret.adj().array() * diff * inv_logit_x.array()
                                                   * (1.0 - inv_logit_x.array())
                                                 + lp_after.adj() * (1.0 - 2.0 * inv_logit_x.array());

                });

        } else {

                reverse_pass_callback([arena_x, ret, diff, inv_logit_x]() mutable {

                        arena_x.adj().array() += ret.adj().array() * diff * inv_logit_x.array()
                                                 * (1.0 - inv_logit_x.array());

                });

        }
        return Eigen::Matrix<var, -1, -1>(ret);

}

////
//// ---- Infinite bounds: Stan's own template ----------------------------------------------------------------------------------------
////
inline bool fn_both_bounds_finite(const double lb_val,
                                  const double ub_val) {

        return lb_val != NEGATIVE_INFTY && ub_val != INFTY;

}

}  // namespace nico_lub_constrain_internal

////
//// ---- The overloads Stan's deserializer finds: int or double bounds, with and without lp ------------------------------------------
////
#define NICOSIMD_LUB_CONSTRAIN_OVERLOADS(T_lb, T_ub)                                                             \
inline Eigen::Matrix<var, -1, -1> lub_constrain(const Eigen::Matrix<var, -1, -1>& x,                             \
                                                const T_lb lb,                                                   \
                                                const T_ub ub,                                                   \
                                                var& lp) {                                                       \
        if (!nico_lub_constrain_internal::fn_both_bounds_finite(lb, ub))                                         \
                return lub_constrain<Eigen::Matrix<var, -1, -1>, T_lb, T_ub>(x, lb, ub, lp);                     \
        return nico_lub_constrain_internal::fn_lub_constrain(x, lb, ub, &lp);                                    \
}                                                                                                                \
inline Eigen::Matrix<var, -1, -1> lub_constrain(const Eigen::Matrix<var, -1, -1>& x,                             \
                                                const T_lb lb,                                                   \
                                                const T_ub ub) {                                                 \
        if (!nico_lub_constrain_internal::fn_both_bounds_finite(lb, ub))                                         \
                return lub_constrain<Eigen::Matrix<var, -1, -1>, T_lb, T_ub>(x, lb, ub);                         \
        return nico_lub_constrain_internal::fn_lub_constrain(x, lb, ub, nullptr);                                \
}
NICOSIMD_LUB_CONSTRAIN_OVERLOADS(int, int)
NICOSIMD_LUB_CONSTRAIN_OVERLOADS(int, double)
NICOSIMD_LUB_CONSTRAIN_OVERLOADS(double, int)
NICOSIMD_LUB_CONSTRAIN_OVERLOADS(double, double)
#undef NICOSIMD_LUB_CONSTRAIN_OVERLOADS

}  // namespace math
}  // namespace stan
#endif
