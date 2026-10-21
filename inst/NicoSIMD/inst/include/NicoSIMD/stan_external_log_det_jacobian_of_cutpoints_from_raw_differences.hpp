#pragma once
//
// -| --------- NicoSIMD log determinant of the Jacobian of ordered cutpoints built from raw differences ---------
//
//     nico_log_det_jacobian_of_cutpoints_from_raw_differences(C_raw, softplus)
//
// For a set of K ordered cutpoints c built from K unconstrained values r (a vector or row_vector C_raw, or each
// row of a matrix C_raw: one set per group, e.g. the random cutpoints of each study) as
//
//     c_1 = r_1,   c_k = c_(k-1) + exp(r_k)         (softplus = 0),
//     c_1 = r_1,   c_k = c_(k-1) + log1p_exp(r_k)   (softplus = 1),          k = 2, ..., K,
//
// the log determinant of the Jacobian of the map from r to c (lower triangular, so the sum of the logs of its
// diagonal: d c_k / d r_k), summed over every set:
//
//     softplus = 0:   sum over the sets of sum_(k = 2..K) r_k,
//     softplus = 1:   sum over the sets of sum_(k = 2..K) log_inv_logit(r_k).
//
// A general function: any ordinal model whose ordered cutpoints are the first one plus the exp (or softplus) of
// unconstrained raw differences (ordinal regression, ordinal meta-analysis and network meta-analysis of test
// accuracy, ordinal latent class models), with fixed cutpoints (one set) or random cutpoints (one set per group).
// The same value and derivatives as MetaOrdDTA's raw_C_to_C_log_det_J_lp (Stan_fns_Jacobian.stan, with the
// Jacobians of construct_C(): softplus -> log_inv_logit(r), exp -> r),
//
//     if (softplus == 1)  log_det_J = sum(log_inv_logit(C_raw[2:K]));     (C_raw[, 2:K] for a matrix)
//     else                log_det_J = sum(C_raw[2:K]);
//
// in ONE autodiff node for all the sets, in place of one node per element plus a sum per set: the sum with
// Eigen's SIMD reduction (AVX2 / AVX-512 packets), and in the softplus case BayesMVP's SIMD kernels for
// log_inv_logit (the value, the same kernel as nico_log_inv_logit) and inv_logit (the derivative,
// d / d r log_inv_logit(r) = inv_logit(-r), the same kernel as nico_inv_logit), computed in the forward pass and
// kept on the arena, so that the reverse pass is the adjoint update alone. d / d r_1 = 0 (the first cutpoint is
// r_1 itself); d / d r_k = 1 (softplus = 0) or inv_logit(-r_k) (softplus = 1), k = 2, ..., K.
//
// Declared in the Stan functions block (no body; stanc --allow-undefined) as any of
//
//     real nico_log_det_jacobian_of_cutpoints_from_raw_differences(vector C_raw, data int softplus);
//     real nico_log_det_jacobian_of_cutpoints_from_raw_differences(row_vector C_raw, data int softplus);
//     real nico_log_det_jacobian_of_cutpoints_from_raw_differences(matrix C_raw, data int softplus);
//
// and used as   target += nico_log_det_jacobian_of_cutpoints_from_raw_differences(C_raw, softplus);
// The same function is also given under the name of MetaOrdDTA's Stan function with nico_ in front,
//
//     real nico_raw_C_to_C_log_det_J_lp(vector C_raw, data int softplus);   (also row_vector and matrix)
//
// (an _lp function, so stanc passes it lp__ and lp_accum__, which it does not use).
// It uses the namespace nico::math and the kernels of stan_external_functions.hpp, which it includes
// (#pragma once).
//
// -----------------------------------------------------------------------------------------------------------

#include "stan_external_functions.hpp"

#include <stan/math/rev.hpp>
#include <stan/math/prim.hpp>

#include <Eigen/Dense>
#include <cstddef>
#include <ostream>
#include <type_traits>
#include <vector>

// ---- the Stan-facing function (global scope; stanc calls it as
//      nico_log_det_jacobian_of_cutpoints_from_raw_differences(C_raw, softplus, pstream__)): C_raw a vector or
//      row_vector (one set of cutpoints) or a matrix (one set per row)
namespace nico::math {

template <typename T_C_raw, stan::require_eigen_t<T_C_raw>* = nullptr>
inline stan::return_type_t<T_C_raw> nico_log_det_jacobian_of_cutpoints_from_raw_differences(
        const T_C_raw& C_raw, const int softplus, std::ostream* /*pstream__*/) {

    using stan::math::var;
    constexpr bool C_raw_is_one_set  = stan::is_eigen_vector<T_C_raw>::value;
    constexpr bool C_raw_is_constant = stan::is_constant_all<T_C_raw>::value;
    const int R = C_raw_is_one_set ? 1 : static_cast<int>(C_raw.rows());
    const int K = C_raw_is_one_set ? static_cast<int>(C_raw.size()) : static_cast<int>(C_raw.cols());
    if ((K < 2) || (R == 0)) return 0.0;
    const auto& C_raw_ref = stan::math::to_ref(C_raw);
    //// (the K - 1 raw differences of every set, set after set: r_2, ..., r_K of set 1, then of set 2, ...)
    const int n_differences_per_set = K - 1;
    const std::size_t n_differences = static_cast<std::size_t>(R) * n_differences_per_set;
    //// (the buffers: the raw differences, then their derivatives in the softplus case; on the arena when the
    ////  derivatives are kept for the reverse pass, in a std::vector otherwise, so that no evaluation of the log
    ////  density alone, and no exp case, leaves memory on the arena)
    const bool keep_the_derivatives = !C_raw_is_constant && (softplus == 1);
    std::vector<double> buffer_without_autodiff;
    double* r = nullptr;
    if (keep_the_derivatives) {
        r = stan::math::ChainableStack::instance_->memalloc_.alloc_array<double>(2 * n_differences);
    } else {
        buffer_without_autodiff.resize(n_differences);
        r = buffer_without_autodiff.data();
    }
    for (int s = 0; s < R; ++s) {
        double* r_s = r + static_cast<std::size_t>(s) * n_differences_per_set;
        for (int k = 1; k < K; ++k) {
            r_s[k - 1] = C_raw_is_one_set ? stan::math::value_of(C_raw_ref.coeff(k))
                                          : stan::math::value_of(C_raw_ref.coeff(s, k));
        }
    }
    double log_det_J = 0.0;
    double* d_r = keep_the_derivatives ? r + n_differences : nullptr;
    if (softplus == 1) {
        //// (log_inv_logit(r) and, for the reverse pass, inv_logit(-r), with BayesMVP's SIMD kernels)
        std::vector<double> log_inv_logit_of_r(r, r + n_differences);
        nico::math::k_log_inv_logit(log_inv_logit_of_r.data(), static_cast<int>(n_differences));
        log_det_J = Eigen::Map<const Eigen::Matrix<double, -1, 1>>(log_inv_logit_of_r.data(),
                                                                   n_differences).sum();
        if (keep_the_derivatives) {
            for (std::size_t i = 0; i < n_differences; ++i) d_r[i] = -r[i];
            nico::math::k_inv_logit(d_r, static_cast<int>(n_differences));
        }
    } else {
        log_det_J = Eigen::Map<const Eigen::Matrix<double, -1, 1>>(r, n_differences).sum();
    }
    if constexpr (C_raw_is_constant) {
        return log_det_J;
    } else {
        using C_raw_vars_type = stan::plain_type_t<stan::math::promote_scalar_t<var, T_C_raw>>;
        stan::arena_t<C_raw_vars_type> C_raw_vars = C_raw_ref;
        const bool softplus_case = (softplus == 1);
        return stan::math::make_callback_var(log_det_J, [C_raw_vars, d_r, softplus_case, R, K,
                                                         n_differences_per_set](auto& vi) mutable {
            const double adjoint = vi.adj();
            for (int s = 0; s < R; ++s) {
                const double* d_r_s = softplus_case ? d_r + static_cast<std::size_t>(s) * n_differences_per_set
                                                    : nullptr;
                for (int k = 1; k < K; ++k) {
                    const double d = softplus_case ? adjoint * d_r_s[k - 1] : adjoint;
                    if constexpr (C_raw_is_one_set) {
                        C_raw_vars.coeffRef(k).adj() += d;
                    } else {
                        C_raw_vars.coeffRef(s, k).adj() += d;
                    }
                }
            }
        });
    }

}

}  // namespace nico::math

using nico::math::nico_log_det_jacobian_of_cutpoints_from_raw_differences;
//
// ---- the same function under the name of MetaOrdDTA's Stan function with nico_ in front (an _lp function: stanc
//      calls it as nico_raw_C_to_C_log_det_J_lp<propto__>(C_raw, softplus, lp__, lp_accum__, pstream__))
namespace nico::math {

template <bool propto__, typename T_C_raw, typename T_lp, typename T_lp_accum,
          stan::require_eigen_t<T_C_raw>* = nullptr>
inline stan::return_type_t<T_C_raw> nico_raw_C_to_C_log_det_J_lp(const T_C_raw& C_raw, const int softplus,
                                                                 T_lp& /*lp__*/, T_lp_accum& /*lp_accum__*/,
                                                                 std::ostream* pstream__) {

    return nico_log_det_jacobian_of_cutpoints_from_raw_differences(C_raw, softplus, pstream__);

}

}  // namespace nico::math

using nico::math::nico_raw_C_to_C_log_det_J_lp;























