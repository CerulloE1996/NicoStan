#pragma once

// A normal distribution with location mu and scale 1 / inv_sigma, truncated to [lower, upper], at a uniform u:
//     a = (lower - mu) * inv_sigma,  b = (upper - mu) * inv_sigma   (an infinite bound stays infinite)
//     column 1:  Z = Phi^{-1}((1 - u) Phi(a) + u Phi(b))   (the standardised quantile at u)
//     column 2:  log(Phi(b) - Phi(a))                      (the log probability of [lower, upper])
// The step of the GHK algorithm of multivariate probit models (binary outcomes: one bound infinite; ordinal
// outcomes: two cutpoints), and of Tobit and truncated regressions. Exact Phi (Phi_type 1).
// Rows with b < underflow_threshold (both bounds in the lower tail) and rows with a > overflow_threshold (both in
// the upper tail) are computed on the log scale; the other rows directly, reflected (Phi(x) = 1 - Phi(-x)) when
// a > 0 or b = +inf, so that the smaller tail mass is the one formed. One node for the whole vector; its partial
// derivatives are those of the function itself (every branch computes the same function):
//     dZ/da = (1 - u) phi(a) / phi(Z),  dZ/db = u phi(b) / phi(Z),  dZ/du = (Phi(b) - Phi(a)) / phi(Z),
//     dlog_mass/da = -phi(a) / (Phi(b) - Phi(a)),  dlog_mass/db = phi(b) / (Phi(b) - Phi(a)),
// each formed on the log scale, and 0 for an infinite bound.
// Declared in the Stan functions block as
//     matrix nico_truncated_normal_quantile_and_log_mass(vector lower, vector upper, vector mu, real inv_sigma,
//                                                        vector u, data real overflow_threshold,
//                                                        data real underflow_threshold);

#include "stan_external_functions.hpp"

namespace nico::math {

inline std::vector<double>& truncated_normal_scratch() {

    static thread_local std::vector<double> values;
    return values;

}

inline double log_phi_or_minus_infinity(const double x) {

    if (std::isinf(x)) return -std::numeric_limits<double>::infinity();
    return -0.5 * x * x + NEG_HALF_LOG_2PI;

}

//// Z = Phi^{-1}(p) from log(p) and log(1 - p), from whichever side is the smaller:
inline double inv_Phi_from_log_p_scalar(const double log_p, const double log_1m_p) {

    if (log_p < log_1m_p) return stan::math::std_normal_log_qf(log_p);
    return -stan::math::std_normal_log_qf(log_1m_p);

}

}  // namespace nico::math

namespace nico::math {

template <typename T_lower, typename T_upper, typename T_mu, typename T_inv_sigma, typename T_u>
inline Eigen::Matrix<stan::return_type_t<T_lower, T_upper, T_mu, T_inv_sigma, T_u>, -1, -1>
nico_truncated_normal_quantile_and_log_mass(const T_lower& lower, const T_upper& upper, const T_mu& mu,
                                            const T_inv_sigma& inv_sigma, const T_u& u,
                                            const double overflow_threshold, const double underflow_threshold,
                                            std::ostream* /*pstream__*/ = nullptr) {

    using scalar = stan::return_type_t<T_lower, T_upper, T_mu, T_inv_sigma, T_u>;
    static const char* function = "nico_truncated_normal_quantile_and_log_mass";
    const int M = mu.size();
    stan::math::check_size_match(function, "lower", lower.size(), "mu", M);
    stan::math::check_size_match(function, "upper", upper.size(), "mu", M);
    stan::math::check_size_match(function, "u", u.size(), "mu", M);
    const double inf = std::numeric_limits<double>::infinity();

    const Eigen::Matrix<double, -1, 1> lower_values = stan::math::value_of(stan::math::to_ref(lower));
    const Eigen::Matrix<double, -1, 1> upper_values = stan::math::value_of(stan::math::to_ref(upper));
    const Eigen::Matrix<double, -1, 1> mu_values = stan::math::value_of(stan::math::to_ref(mu));
    const Eigen::Matrix<double, -1, 1> u_values = stan::math::value_of(stan::math::to_ref(u));
    const double s = stan::math::value_of(inv_sigma);

    //// ---- the standardised bounds, and the rows of each branch (0 central, 1 lower tail, 2 upper tail):
    std::vector<double>& scratch = nico::math::truncated_normal_scratch();
    if (static_cast<int>(scratch.size()) < 8 * M) scratch.resize(8 * M);
    double* a = scratch.data();
    double* b = a + M;
    double* Phi_a = b + M;
    double* Phi_b = Phi_a + M;
    double* q = Phi_b + M;
    double* log_mass_central = q + M;
    double* reflect = log_mass_central + M;
    double* branch = reflect + M;
    for (int i = 0; i < M; ++i) {
        a[i] = std::isinf(lower_values[i]) ? lower_values[i] : (lower_values[i] - mu_values[i]) * s;
        b[i] = std::isinf(upper_values[i]) ? upper_values[i] : (upper_values[i] - mu_values[i]) * s;
        branch[i] = (b[i] < underflow_threshold) ? 1.0 : ((a[i] > overflow_threshold) ? 2.0 : 0.0);
        reflect[i] = ((a[i] > 0.0) || (b[i] == inf)) ? 1.0 : 0.0;
        //// (the arguments of the two Phi calls, finite: Phi(-40) and Phi(40) are 0 and 1 in double precision)
        const double x_1 = (reflect[i] == 1.0) ? -b[i] : a[i];
        const double x_2 = (reflect[i] == 1.0) ? -a[i] : b[i];
        Phi_a[i] = std::min(40.0, std::max(-40.0, x_1));
        Phi_b[i] = std::min(40.0, std::max(-40.0, x_2));
    }

    //// ---- the central rows, all at once in SIMD (the tail rows are written over below):
    nico::math::k_Phi(Phi_a, M);
    nico::math::k_Phi(Phi_b, M);
    for (int i = 0; i < M; ++i) {
        const double prob = Phi_b[i] - Phi_a[i];
        const double u_or_1m_u = (reflect[i] == 1.0) ? 1.0 - u_values[i] : u_values[i];
        q[i] = Phi_a[i] + u_or_1m_u * prob;
        log_mass_central[i] = prob;
    }
    nico::math::k_inv_Phi(q, M);
    nico::math::k_log(log_mass_central, M);
    Eigen::Matrix<double, -1, 1> Z_values(M);
    Eigen::Matrix<double, -1, 1> log_mass_values(M);
    for (int i = 0; i < M; ++i) {
        Z_values[i] = (reflect[i] == 1.0) ? -q[i] : q[i];
        log_mass_values[i] = log_mass_central[i];
    }

    //// ---- the tail rows, on the log scale:
    for (int i = 0; i < M; ++i) {
        if (branch[i] == 0.0) continue;
        const double log_u = std::log(u_values[i]);
        const double log_1m_u = stan::math::log1m(u_values[i]);
        if (branch[i] == 1.0) {
            const double log_Phi_lo = (a[i] == -inf) ? -inf : stan::math::std_normal_lcdf(a[i]);
            const double log_Phi_hi = stan::math::std_normal_lcdf(b[i]);
            const double log_Phi_Z = stan::math::log_sum_exp(log_1m_u + log_Phi_lo, log_u + log_Phi_hi);
            const double log_1m_Phi_Z = stan::math::log1m_exp(log_Phi_Z);
            Z_values[i] = nico::math::inv_Phi_from_log_p_scalar(log_Phi_Z, log_1m_Phi_Z);
            log_mass_values[i] = log_Phi_hi + stan::math::log1m_exp(log_Phi_lo - log_Phi_hi);
        } else {
            const double log_1m_Phi_lo = stan::math::std_normal_lcdf(-a[i]);
            const double log_1m_Phi_hi = (b[i] == inf) ? -inf : stan::math::std_normal_lcdf(-b[i]);
            const double log_1m_Phi_Z = stan::math::log_sum_exp(log_1m_u + log_1m_Phi_lo, log_u + log_1m_Phi_hi);
            const double log_Phi_Z = stan::math::log1m_exp(log_1m_Phi_Z);
            Z_values[i] = nico::math::inv_Phi_from_log_p_scalar(log_Phi_Z, log_1m_Phi_Z);
            log_mass_values[i] = log_1m_Phi_lo + stan::math::log1m_exp(log_1m_Phi_hi - log_1m_Phi_lo);
        }
    }

    if constexpr (stan::is_constant_all<T_lower, T_upper, T_mu, T_inv_sigma, T_u>::value) {
        Eigen::Matrix<double, -1, -1> result(M, 2);
        result.col(0) = Z_values;
        result.col(1) = log_mass_values;
        return result;
    } else {
        //// ---- the partial derivatives with respect to a, b and u, on the arena:
        stan::arena_t<Eigen::Matrix<double, -1, -1>> partials(5, M);
        // dZ/da, dZ/db, dZ/du, dlog_mass/da, dlog_mass/db
        for (int i = 0; i < M; ++i) {
            const double log_phi_Z = nico::math::log_phi_or_minus_infinity(Z_values[i]);
            const double log_phi_a = nico::math::log_phi_or_minus_infinity(a[i]);
            const double log_phi_b = nico::math::log_phi_or_minus_infinity(b[i]);
            const double log_u = std::log(u_values[i]);
            const double log_1m_u = stan::math::log1m(u_values[i]);
            partials(0, i) = std::exp(log_1m_u + log_phi_a - log_phi_Z);
            partials(1, i) = std::exp(log_u + log_phi_b - log_phi_Z);
            partials(2, i) = std::exp(log_mass_values[i] - log_phi_Z);
            partials(3, i) = -std::exp(log_phi_a - log_mass_values[i]);
            partials(4, i) = std::exp(log_phi_b - log_mass_values[i]);
        }
        //// (lower - mu and upper - mu, for the derivative with respect to inv_sigma; 0 for an infinite bound,
        ////  whose partials are 0)
        stan::arena_t<Eigen::Matrix<double, -1, -1>> distances(2, M);
        for (int i = 0; i < M; ++i) {
            distances(0, i) = std::isinf(a[i]) ? 0.0 : lower_values[i] - mu_values[i];
            distances(1, i) = std::isinf(b[i]) ? 0.0 : upper_values[i] - mu_values[i];
        }
        stan::arena_t<T_lower> lower_arena = lower;
        stan::arena_t<T_upper> upper_arena = upper;
        stan::arena_t<T_mu> mu_arena = mu;
        stan::arena_t<T_u> u_arena = u;
        using stan::math::vari;
        auto& arena = stan::math::ChainableStack::instance_->memalloc_;
        vari* result_nodes = arena.alloc_array<vari>(2 * M);
        Eigen::Matrix<scalar, -1, -1> result(M, 2);
        for (int i = 0; i < M; ++i) {
            ::new (static_cast<void*>(result_nodes + i)) vari(Z_values[i], false);
            ::new (static_cast<void*>(result_nodes + M + i)) vari(log_mass_values[i], false);
            result(i, 0) = stan::math::var(result_nodes + i);
            result(i, 1) = stan::math::var(result_nodes + M + i);
        }
        vari* inv_sigma_parent = nullptr;
        if constexpr (!stan::is_constant_all<T_inv_sigma>::value) inv_sigma_parent = inv_sigma.vi_;
        stan::math::reverse_pass_callback([result_nodes, partials, distances, lower_arena, upper_arena, mu_arena,
                                           u_arena, inv_sigma_parent, s, M]() mutable {
            for (int i = 0; i < M; ++i) {
                const double adjoint_of_Z = result_nodes[i].adj_;
                const double adjoint_of_log_mass = result_nodes[M + i].adj_;
                const double adjoint_of_a = adjoint_of_Z * partials(0, i) + adjoint_of_log_mass * partials(3, i);
                const double adjoint_of_b = adjoint_of_Z * partials(1, i) + adjoint_of_log_mass * partials(4, i);
                if constexpr (!stan::is_constant_all<T_u>::value) {
                    u_arena.coeffRef(i).adj() += adjoint_of_Z * partials(2, i);
                }
                if constexpr (!stan::is_constant_all<T_lower>::value) {
                    lower_arena.coeffRef(i).adj() += s * adjoint_of_a;
                }
                if constexpr (!stan::is_constant_all<T_upper>::value) {
                    upper_arena.coeffRef(i).adj() += s * adjoint_of_b;
                }
                if constexpr (!stan::is_constant_all<T_mu>::value) {
                    mu_arena.coeffRef(i).adj() -= s * (adjoint_of_a + adjoint_of_b);
                }
                if (inv_sigma_parent) {
                    inv_sigma_parent->adj_ += distances(0, i) * adjoint_of_a + distances(1, i) * adjoint_of_b;
                }
            }
        });
        return result;
    }

}

}  // namespace nico::math

using nico::math::nico_truncated_normal_quantile_and_log_mass;























