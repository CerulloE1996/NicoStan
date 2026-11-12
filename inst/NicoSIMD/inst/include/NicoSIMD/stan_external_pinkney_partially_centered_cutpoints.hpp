#pragma once
// ---- NicoSIMD fused Sean Pinkney partially-centered study cutpoints and total log-Jacobian ----
#if __has_include("stan_external_functions_of_NicoSIMD.hpp")
#include "stan_external_functions_of_NicoSIMD.hpp"
#include "stan_external_dirichlet_cutpoint_median.hpp"
#elif __has_include("NicoSIMD/stan_external_functions_of_NicoSIMD.hpp")
#include "NicoSIMD/stan_external_functions_of_NicoSIMD.hpp"
#include "NicoSIMD/stan_external_dirichlet_cutpoint_median.hpp"
#else
#include "/home/enzocerullo/Documents/Work/PhD_work/R_packages/R_library_for_the_sim_pilot_on_the_local_HPC/NicoSIMD/include/NicoSIMD/stan_external_functions_of_NicoSIMD.hpp"
#include "/home/enzocerullo/Documents/Work/PhD_work/R_packages/R_library_for_the_sim_pilot_on_the_local_HPC/NicoSIMD/include/NicoSIMD/stan_external_dirichlet_cutpoint_median.hpp"
#endif

#include <stan/math/rev.hpp>
#include <stan/math/prim.hpp>
#include <Eigen/Dense>
#include <cmath>
#include <ostream>
#include <vector>

template <typename T_mu,
          typename T_tau,
          typename T_z,
          typename T_w,
          typename T_s,
          typename T_l,
          typename T_wd>
inline Eigen::Matrix<stan::return_type_t<T_mu, T_tau, T_z>, -1, 1>
nico_pinkney_C_and_log_J_from_pairs(const T_mu& logit_centre_of_pair,
                                    const T_tau& spread_free_of_pair,
                                    const T_z& C_raw_c,
                                    const T_w& Pinkney_w_c,
                                    const T_s& logit_start,
                                    const T_l& C_raw_lower_c,
                                    const T_wd& C_raw_width,
                                    const double sum_log_width,
                                    const std::vector<int>& pair_of_each_C,
                                    const std::vector<int>& is_increment_of_each_C,
                                    const int softplus,
                                    std::ostream* pstream__) {
    (void) pstream__;
    using T = stan::return_type_t<T_mu, T_tau, T_z>;
    using T_mu_scalar  = stan::value_type_t<T_mu>;
    using T_tau_scalar = stan::value_type_t<T_tau>;
    using T_z_scalar   = stan::value_type_t<T_z>;

    const Eigen::Index n_C = C_raw_c.size();
    const Eigen::Index n_pairs = logit_centre_of_pair.size();
    if (n_C == 0) {
        Eigen::Matrix<T, -1, 1> empty_res(1);
        empty_res[0] = 0.0;
        return empty_res;
    }

    const auto& mu_ref  = stan::math::to_ref(logit_centre_of_pair);
    const auto& tau_ref = stan::math::to_ref(spread_free_of_pair);
    const auto& z_ref   = stan::math::to_ref(C_raw_c);
    const auto& w_ref   = stan::math::to_ref(Pinkney_w_c);
    const auto& s_ref   = stan::math::to_ref(logit_start);
    const auto& l_ref   = stan::math::to_ref(C_raw_lower_c);
    const auto& wd_ref  = stan::math::to_ref(C_raw_width);

    Eigen::VectorXd mu_val(n_pairs);
    Eigen::VectorXd tau_val(n_pairs);
    Eigen::VectorXd log_tau_val(n_pairs);
    for (Eigen::Index u = 0; u < n_pairs; ++u) {
        mu_val[u]  = stan::math::value_of(mu_ref.coeff(u));
        tau_val[u] = stan::math::value_of(tau_ref.coeff(u));
    }
    log_tau_val = tau_val;
    nico::math::k_log(log_tau_val.data(), static_cast<int>(n_pairs));

    if constexpr (!stan::is_var<T>::value) {
        Eigen::VectorXd theta(n_C);
        Eigen::VectorXd abs_theta(n_C);
        Eigen::VectorXd D_vec(n_C);
        double sum_log_tau_over_D = 0.0;
        for (Eigen::Index p = 0; p < n_C; ++p) {
            const int u = pair_of_each_C[p] - 1;
            const double w_p = stan::math::value_of(w_ref.coeff(p));
            const double tau_u = tau_val[u];
            const double D_p = tau_u - w_p * (tau_u - 1.0);
            const double eta_sh = (1.0 - w_p) * stan::math::value_of(s_ref.coeff(p))
                                  + stan::math::value_of(z_ref.coeff(p));
            const double th = (w_p * mu_val[u] + tau_u * eta_sh) / D_p;
            theta[p] = th;
            abs_theta[p] = std::abs(th);
            D_vec[p] = D_p;
            sum_log_tau_over_D += log_tau_val[u];
        }
        Eigen::VectorXd log_D = D_vec;
        nico::math::k_log(log_D.data(), static_cast<int>(n_C));
        for (Eigen::Index p = 0; p < n_C; ++p) sum_log_tau_over_D -= log_D[p];

        Eigen::VectorXd q = theta;
        nico::math::k_inv_logit(q.data(), static_cast<int>(n_C));
        Eigen::VectorXd log_inv_abs = abs_theta;
        nico::math::k_log_inv_logit(log_inv_abs.data(), static_cast<int>(n_C));

        Eigen::VectorXd r_vec(n_C);
        double log_J = sum_log_width + sum_log_tau_over_D;
        for (Eigen::Index p = 0; p < n_C; ++p) {
            r_vec[p] = stan::math::value_of(l_ref.coeff(p))
                       + stan::math::value_of(wd_ref.coeff(p)) * q[p];
            log_J += 2.0 * log_inv_abs[p] - abs_theta[p];
        }

        Eigen::VectorXd step_val = r_vec;
        if (softplus == 1) {
            nico::math::k_log1p_exp(step_val.data(), static_cast<int>(n_C));
            Eigen::VectorXd lil_r = r_vec;
            nico::math::k_log_inv_logit(lil_r.data(), static_cast<int>(n_C));
            for (Eigen::Index p = 0; p < n_C; ++p) {
                if (is_increment_of_each_C[p] == 1) log_J += lil_r[p];
            }
        } else {
            nico::math::k_exp(step_val.data(), static_cast<int>(n_C));
            for (Eigen::Index p = 0; p < n_C; ++p) {
                if (is_increment_of_each_C[p] == 1) log_J += r_vec[p];
            }
        }

        Eigen::Matrix<double, -1, 1> result(n_C + 1);
        for (Eigen::Index p = 0; p < n_C; ++p) {
            result[p] = (is_increment_of_each_C[p] == 1) ? (result[p - 1] + step_val[p]) : r_vec[p];
        }
        result[n_C] = log_J;
        return result;
    } else {
        stan::arena_t<Eigen::Matrix<T_mu_scalar,  -1, 1>> arena_mu  = mu_ref;
        stan::arena_t<Eigen::Matrix<T_tau_scalar, -1, 1>> arena_tau = tau_ref;
        stan::arena_t<Eigen::Matrix<T_z_scalar,   -1, 1>> arena_z   = z_ref;
        stan::arena_t<Eigen::Matrix<stan::math::var, -1, 1>> arena_result(n_C + 1);
        stan::arena_t<Eigen::Matrix<double, 8, -1>> arena_buf(8, n_C);
        int* arena_pair = stan::math::ChainableStack::instance_->memalloc_.alloc_array<int>(n_C);
        int* arena_inc  = stan::math::ChainableStack::instance_->memalloc_.alloc_array<int>(n_C);

        Eigen::VectorXd theta(n_C);
        Eigen::VectorXd abs_theta(n_C);
        Eigen::VectorXd D_vec(n_C);
        double sum_log_tau_over_D = 0.0;

        for (Eigen::Index p = 0; p < n_C; ++p) {
            const int u = pair_of_each_C[p] - 1;
            arena_pair[p] = u;
            arena_inc[p]  = is_increment_of_each_C[p];
            const double w_p = stan::math::value_of(w_ref.coeff(p));
            const double tau_u = tau_val[u];
            const double D_p = tau_u - w_p * (tau_u - 1.0);
            const double inv_D_p = 1.0 / D_p;
            const double eta_sh = (1.0 - w_p) * stan::math::value_of(s_ref.coeff(p))
                                  + stan::math::value_of(arena_z.coeff(p));
            const double th = (w_p * mu_val[u] + tau_u * eta_sh) * inv_D_p;
            theta[p] = th;
            abs_theta[p] = std::abs(th);
            D_vec[p] = D_p;
            sum_log_tau_over_D += log_tau_val[u];
            arena_buf.coeffRef(0, p) = inv_D_p;
            arena_buf.coeffRef(1, p) = w_p;
            arena_buf.coeffRef(2, p) = tau_u;
            arena_buf.coeffRef(3, p) = eta_sh - (1.0 - w_p) * th;
            arena_buf.coeffRef(7, p) = 1.0 / tau_u;
        }

        Eigen::VectorXd log_D = D_vec;
        nico::math::k_log(log_D.data(), static_cast<int>(n_C));
        for (Eigen::Index p = 0; p < n_C; ++p) sum_log_tau_over_D -= log_D[p];

        Eigen::VectorXd q = theta;
        nico::math::k_inv_logit(q.data(), static_cast<int>(n_C));
        Eigen::VectorXd one_m_q = -theta;
        nico::math::k_inv_logit(one_m_q.data(), static_cast<int>(n_C));
        Eigen::VectorXd log_inv_abs = abs_theta;
        nico::math::k_log_inv_logit(log_inv_abs.data(), static_cast<int>(n_C));

        Eigen::VectorXd r_vec(n_C);
        double log_J = sum_log_width + sum_log_tau_over_D;
        for (Eigen::Index p = 0; p < n_C; ++p) {
            const double wd_p = stan::math::value_of(wd_ref.coeff(p));
            const double q_p = q[p];
            const double omq_p = one_m_q[p];
            r_vec[p] = stan::math::value_of(l_ref.coeff(p)) + wd_p * q_p;
            arena_buf.coeffRef(4, p) = q_p - omq_p;
            arena_buf.coeffRef(5, p) = wd_p * q_p * omq_p;
            log_J += 2.0 * log_inv_abs[p] - abs_theta[p];
        }

        Eigen::VectorXd step_val = r_vec;
        if (softplus == 1) {
            nico::math::k_log1p_exp(step_val.data(), static_cast<int>(n_C));
            Eigen::VectorXd s_r = r_vec;
            nico::math::k_inv_logit(s_r.data(), static_cast<int>(n_C));
            Eigen::VectorXd lil_r = r_vec;
            nico::math::k_log_inv_logit(lil_r.data(), static_cast<int>(n_C));
            for (Eigen::Index p = 0; p < n_C; ++p) {
                arena_buf.coeffRef(6, p) = s_r[p];
                if (arena_inc[p] == 1) log_J += lil_r[p];
            }
        } else {
            nico::math::k_exp(step_val.data(), static_cast<int>(n_C));
            for (Eigen::Index p = 0; p < n_C; ++p) {
                arena_buf.coeffRef(6, p) = step_val[p];
                if (arena_inc[p] == 1) log_J += r_vec[p];
            }
        }

        double prev_C = 0.0;
        for (Eigen::Index p = 0; p < n_C; ++p) {
            const double C_p = (arena_inc[p] == 1) ? (prev_C + step_val[p]) : r_vec[p];
            prev_C = C_p;
            arena_result.coeffRef(p) = stan::math::var(new stan::math::vari(C_p, false));
        }
        arena_result.coeffRef(n_C) = stan::math::var(new stan::math::vari(log_J, false));

        const bool softplus_flag = (softplus == 1);
        stan::math::reverse_pass_callback(
            [arena_mu, arena_tau, arena_z, arena_result, arena_buf, arena_pair, arena_inc, n_C, softplus_flag]() mutable {
                const double adj_J = arena_result.coeff(n_C).adj();
                double running_adj_C = 0.0;
                for (Eigen::Index p = n_C - 1; p >= 0; --p) {
                    const double adj_C_here = arena_result.coeff(p).adj() + running_adj_C;
                    const bool is_inc = (arena_inc[p] == 1);
                    running_adj_C = is_inc ? adj_C_here : 0.0;
                    double adj_r;
                    if (!is_inc) {
                        adj_r = adj_C_here;
                    } else if (!softplus_flag) {
                        adj_r = adj_C_here * arena_buf.coeff(6, p) + adj_J;
                    } else {
                        const double s_p = arena_buf.coeff(6, p);
                        adj_r = adj_C_here * s_p + adj_J * (1.0 - s_p);
                    }
                    const double adj_theta = adj_r * arena_buf.coeff(5, p) - adj_J * arena_buf.coeff(4, p);
                    const double inv_D_p = arena_buf.coeff(0, p);
                    const double w_p     = arena_buf.coeff(1, p);
                    const double tau_u   = arena_buf.coeff(2, p);
                    const int u          = arena_pair[p];
                    if constexpr (stan::is_var<T_z_scalar>::value) {
                        arena_z.coeffRef(p).adj() += adj_theta * (tau_u * inv_D_p);
                    }
                    if constexpr (stan::is_var<T_mu_scalar>::value) {
                        arena_mu.coeffRef(u).adj() += adj_theta * (w_p * inv_D_p);
                    }
                    if constexpr (stan::is_var<T_tau_scalar>::value) {
                        arena_tau.coeffRef(u).adj() += adj_theta * (arena_buf.coeff(3, p) * inv_D_p)
                                                       + adj_J * (w_p * arena_buf.coeff(7, p) * inv_D_p);
                    }
                }
            });
        return Eigen::Matrix<T, -1, 1>(arena_result);
    }
}
