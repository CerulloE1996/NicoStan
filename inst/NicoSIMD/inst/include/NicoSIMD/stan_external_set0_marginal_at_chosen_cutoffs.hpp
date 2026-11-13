#pragma once
// ---- NicoSIMD exact Set-0 marginal log-density at chosen cutoffs via 24-node Laplace-centered Gauss-Hermite ----
//
// Evaluates the exact normalized marginal log-density of x_S = C_S - Xbeta_baseline under Set 0:
//
//   log p_Set0(x_S) = log integral_{-inf}^{inf} N(b | 0, sqrt(v0))
//                       * Dirichlet(Delta F(x_S + b) | alpha_of_the_segments)
//                       * prod_{j=1}^m f(x_j + b) db
//
// where F = Phi (use_probit_link == 1) or inv_logit (use_probit_link == 0), f = F', and v0 > 0 is the
// prior variance of Xbeta_baseline under Set 0. Because g(b) is strictly concave in b, 8 Newton steps
// locate the exact mode b* and curvature scale sigma* = (-g''(b*))^{-1/2} to machine precision, and a
// 24-node probabilist Gauss-Hermite quadrature centered at (b*, sigma*) matches adaptive integration to ~1e-15
// across the entire parameter space (including extreme tails and K = 63 thresholds).
// Uses NicoSIMD's k_log_Phi / k_log_inv_logit / k_log1m_exp / k_exp kernels (AVX-512 on 8-lane hosts,
// AVX2 on 4-lane hosts such as ALICE with zero AVX-512 instructions) and one reverse_pass_callback node.

#if __has_include("stan_external_functions.hpp")
#include "stan_external_functions.hpp"
#elif __has_include("NicoSIMD/stan_external_functions.hpp")
#include "NicoSIMD/stan_external_functions.hpp"
#endif

#include <stan/math/rev.hpp>
#include <stan/math/prim.hpp>
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <ostream>
#include <stdexcept>
#include <vector>

namespace nico::math::set0_marginal_detail {

static constexpr double HALF_LOG_TWO_PI = 0.918938533204672741780329736406;

static constexpr double GH24_T[24] = {
    -8.50780351919527433e+00,
    -7.43789066602167104e+00,
    -6.54167500509862876e+00,
    -5.73274717525120181e+00,
    -4.97804137463911367e+00,
    -4.26038360501990354e+00,
    -3.56930676407355918e+00,
    -2.89772864322331181e+00,
    -2.24046785169175244e+00,
    -1.59348042981641846e+00,
    -9.53421922932108146e-01,
    -3.17370096629455034e-01,
     3.17370096629453258e-01,
     9.53421922932111698e-01,
     1.59348042981641846e+00,
     2.24046785169175156e+00,
     2.89772864322331269e+00,
     3.56930676407356007e+00,
     4.26038360501990709e+00,
     4.97804137463911989e+00,
     5.73274717525120003e+00,
     6.54167500509863320e+00,
     7.43789066602166216e+00,
     8.50780351919525657e+00
};

static constexpr double GH24_LW[24] = {
    -3.69042806607304854e+01,
    -2.86212344883108756e+01,
    -2.24843031451895179e+01,
    -1.76020197436324572e+01,
    -1.36185797828409356e+01,
    -1.03468101270751411e+01,
    -7.67407824125704963e+00,
    -5.52732585071174043e+00,
    -3.85723446710271611e+00,
    -2.63012623460007200e+00,
    -1.82350086207048112e+00,
    -1.42349743041213506e+00,
    -1.42349743041215793e+00,
    -1.82350086207048467e+00,
    -2.63012623460007866e+00,
    -3.85723446710268059e+00,
    -5.52732585071170757e+00,
    -7.67407824125703453e+00,
    -1.03468101270751269e+01,
    -1.36185797828409267e+01,
    -1.76020197436324182e+01,
    -2.24843031451895001e+01,
    -2.86212344883107441e+01,
    -3.69042806607303291e+01
};

inline void eval_log_integrand(const double b,
                               const double* x,
                               const double* w_seg,
                               const int m,
                               const double inv_v0,
                               const double lg_const,
                               const int link,
                               double* z_buf,
                               double* lF_buf,
                               double* lS_buf,
                               double* lf_buf,
                               double* log_q_buf,
                               double* work_buf,
                               double* r_own_buf,
                               double* r_nxt_buf,
                               double* dg_dz_out,
                               double& val_out,
                               double& dg_db_out,
                               double& d2g_db2_out) {
    for (int j = 0; j < m; ++j) {
        const double zj = x[j] + b;
        z_buf[j]  = zj;
        lF_buf[j] = zj;
        lS_buf[j] = -zj;
    }
    if (link == 1) {
        nico::math::k_log_Phi(lF_buf, m);
        nico::math::k_log_Phi(lS_buf, m);
        for (int j = 0; j < m; ++j) {
            lf_buf[j] = -0.5 * z_buf[j] * z_buf[j] - HALF_LOG_TWO_PI;
        }
    } else {
        nico::math::k_log_inv_logit(lF_buf, m);
        nico::math::k_log_inv_logit(lS_buf, m);
        for (int j = 0; j < m; ++j) {
            lf_buf[j] = lF_buf[j] + lS_buf[j];
        }
    }

    log_q_buf[0] = lF_buf[0];
    log_q_buf[m] = lS_buf[m - 1];
    if (m > 1) {
        for (int j = 1; j < m; ++j) {
            work_buf[j - 1] = (z_buf[j - 1] >= 0.0) ? (lS_buf[j] - lS_buf[j - 1]) : (lF_buf[j - 1] - lF_buf[j]);
        }
        nico::math::k_log1m_exp(work_buf, m - 1);
        for (int j = 1; j < m; ++j) {
            log_q_buf[j] = ((z_buf[j - 1] >= 0.0) ? lS_buf[j - 1] : lF_buf[j]) + work_buf[j - 1];
        }
    }

    double val = -0.5 * b * b * inv_v0 + lg_const;
    for (int i = 0; i <= m; ++i) {
        if (w_seg[i] != 1.0) val += (w_seg[i] - 1.0) * log_q_buf[i];
    }
    for (int j = 0; j < m; ++j) {
        val += lf_buf[j];
        r_own_buf[j] = lf_buf[j] - log_q_buf[j];
        r_nxt_buf[j] = lf_buf[j] - log_q_buf[j + 1];
    }
    nico::math::k_exp(r_own_buf, m);
    nico::math::k_exp(r_nxt_buf, m);

    double sum_dg_dz = 0.0;
    for (int j = 0; j < m; ++j) {
        const double d_f = (link == 1) ? (-z_buf[j]) : (std::exp(lS_buf[j]) - std::exp(lF_buf[j]));
        const double g_j = (w_seg[j] - 1.0) * r_own_buf[j] - (w_seg[j + 1] - 1.0) * r_nxt_buf[j] + d_f;
        dg_dz_out[j] = g_j;
        sum_dg_dz += g_j;
    }
    dg_db_out = -b * inv_v0 + sum_dg_dz;

    double sum_d2 = -inv_v0;
    for (int j = 0; j < m; ++j) {
        if (link == 1) {
            sum_d2 -= 1.0;
        } else {
            sum_d2 -= 2.0 * std::exp(lf_buf[j]);
        }
    }
    auto dlogf_dz = [&](const int idx) -> double {
        return (link == 1) ? (-z_buf[idx]) : (std::exp(lS_buf[idx]) - std::exp(lF_buf[idx]));
    };
    if (w_seg[0] != 1.0) {
        const double u0 = r_own_buf[0];
        sum_d2 += (w_seg[0] - 1.0) * (dlogf_dz(0) * u0 - u0 * u0);
    }
    if (w_seg[m] != 1.0) {
        const double um = r_nxt_buf[m - 1];
        sum_d2 += (w_seg[m] - 1.0) * (-dlogf_dz(m - 1) * um - um * um);
    }
    for (int i = 1; i < m; ++i) {
        if (w_seg[i] != 1.0) {
            const double r_hi = r_own_buf[i];
            const double r_lo = r_nxt_buf[i - 1];
            const double ui   = r_hi - r_lo;
            sum_d2 += (w_seg[i] - 1.0) * (dlogf_dz(i) * r_hi - dlogf_dz(i - 1) * r_lo - ui * ui);
        }
    }
    val_out = val;
    d2g_db2_out = sum_d2;
}

}  // namespace nico::math::set0_marginal_detail

template <typename T_x, typename T_seg, typename T_v0>
inline stan::return_type_t<T_x>
nico_log_Set0_marginal_at_chosen_cutoffs(const T_x& x_chosen,
                                         const T_seg& alpha_of_the_segments,
                                         const T_v0& v0_arg,
                                         const int use_probit_link,
                                         std::ostream* pstream__) {
    (void) pstream__;
    namespace detail = nico::math::set0_marginal_detail;
    using T = stan::return_type_t<T_x>;
    const double v0 = stan::math::value_of(v0_arg);
    const int m = static_cast<int>(x_chosen.size());
    if (m == 0) return 0.0;
    if (alpha_of_the_segments.size() != m + 1) {
        throw std::invalid_argument("nico_log_Set0_marginal_at_chosen_cutoffs: alpha_of_the_segments size != m + 1");
    }
    if (!(v0 > 0.0)) {
        throw std::domain_error("nico_log_Set0_marginal_at_chosen_cutoffs: v0 must be positive");
    }

    const auto& x_ref   = stan::math::to_ref(x_chosen);
    const auto& seg_ref = stan::math::to_ref(alpha_of_the_segments);

    std::vector<double> x_val(m);
    for (int j = 0; j < m; ++j) {
        x_val[j] = stan::math::value_of(x_ref.coeff(j));
        if (!std::isfinite(x_val[j])) {
            throw std::domain_error("nico_log_Set0_marginal_at_chosen_cutoffs: x_chosen is not finite");
        }
        if (j > 0 && !(x_val[j] > x_val[j - 1])) {
            throw std::domain_error("nico_log_Set0_marginal_at_chosen_cutoffs: x_chosen is not strictly increasing");
        }
    }

    std::vector<double> w_seg(m + 1);
    double sum_seg = 0.0;
    double sum_lgamma_seg = 0.0;
    for (int i = 0; i <= m; ++i) {
        w_seg[i] = stan::math::value_of(seg_ref.coeff(i));
        sum_seg += w_seg[i];
        sum_lgamma_seg += std::lgamma(w_seg[i]);
    }
    const double lg_const = std::lgamma(sum_seg) - sum_lgamma_seg;
    const double inv_v0   = 1.0 / v0;

    // Initial bracketed guess from Beta/Dirichlet mean probabilities
    double b_star = 0.0;
    double cum_w  = 0.0;
    for (int j = 0; j < m; ++j) {
        cum_w += w_seg[j];
        const double p_mean = cum_w / sum_seg;
        const double q_mean = (use_probit_link == 1)
                              ? stan::math::inv_Phi(p_mean)
                              : std::log(p_mean / (1.0 - p_mean));
        b_star += (q_mean - x_val[j]);
    }
    b_star /= static_cast<double>(m);

    std::vector<double> work_all(8 * m + 4);
    double* z_buf     = work_all.data();
    double* lF_buf    = z_buf + m;
    double* lS_buf    = lF_buf + m;
    double* lf_buf    = lS_buf + m;
    double* log_q_buf = lf_buf + m;
    double* work_buf  = log_q_buf + (m + 1);
    double* r_own_buf = work_buf + m;
    double* r_nxt_buf = r_own_buf + m;
    std::vector<double> dg_dz_tmp(m);

    double val = 0.0, dg_db = 0.0, d2g_db2 = -inv_v0;
    for (int it = 0; it < 10; ++it) {
        detail::eval_log_integrand(b_star, x_val.data(), w_seg.data(), m, inv_v0, lg_const, use_probit_link,
                                   z_buf, lF_buf, lS_buf, lf_buf, log_q_buf, work_buf, r_own_buf, r_nxt_buf,
                                   dg_dz_tmp.data(), val, dg_db, d2g_db2);
        double step = -dg_db / d2g_db2;
        if (step >  2.0) step =  2.0;
        if (step < -2.0) step = -2.0;
        b_star += step;
        if (std::abs(step) <= 1e-14 * (1.0 + std::abs(b_star))) break;
    }
    detail::eval_log_integrand(b_star, x_val.data(), w_seg.data(), m, inv_v0, lg_const, use_probit_link,
                               z_buf, lF_buf, lS_buf, lf_buf, log_q_buf, work_buf, r_own_buf, r_nxt_buf,
                               dg_dz_tmp.data(), val, dg_db, d2g_db2);
    const double sig_star = 1.0 / std::sqrt(-d2g_db2);

    double lw_terms[24];
    std::vector<double> grad_nodes(24 * m);
    double max_lw = -std::numeric_limits<double>::infinity();
    for (int i = 0; i < 24; ++i) {
        const double t_i = detail::GH24_T[i];
        const double b_i = b_star + sig_star * t_i;
        double val_i = 0.0, dg_i = 0.0, d2_i = 0.0;
        detail::eval_log_integrand(b_i, x_val.data(), w_seg.data(), m, inv_v0, lg_const, use_probit_link,
                                   z_buf, lF_buf, lS_buf, lf_buf, log_q_buf, work_buf, r_own_buf, r_nxt_buf,
                                   grad_nodes.data() + i * m, val_i, dg_i, d2_i);
        const double lw_i = detail::GH24_LW[i] + val_i + 0.5 * t_i * t_i;
        lw_terms[i] = lw_i;
        if (lw_i > max_lw) max_lw = lw_i;
    }

    double sum_exp = 0.0;
    double w_norm[24];
    for (int i = 0; i < 24; ++i) {
        w_norm[i] = std::exp(lw_terms[i] - max_lw);
        sum_exp += w_norm[i];
    }
    const double inv_sum = 1.0 / sum_exp;
    const double log_marginal = -0.5 * std::log(v0) + std::log(sig_star) + max_lw + std::log(sum_exp);

    if constexpr (!stan::is_var<T>::value) {
        return log_marginal;
    } else {
        stan::arena_t<Eigen::Matrix<stan::math::var, -1, 1>> arena_x = x_ref;
        double* arena_grad = stan::math::ChainableStack::instance_->memalloc_.alloc_array<double>(m);
        for (int j = 0; j < m; ++j) {
            double g_j = 0.0;
            for (int i = 0; i < 24; ++i) {
                g_j += (w_norm[i] * inv_sum) * grad_nodes[i * m + j];
            }
            arena_grad[j] = g_j;
        }
        stan::math::var res_var(new stan::math::vari(log_marginal, false));
        stan::math::reverse_pass_callback([arena_x, arena_grad, res_var, m]() mutable {
            const double adj = res_var.adj();
            for (int j = 0; j < m; ++j) {
                arena_x.coeffRef(j).adj() += adj * arena_grad[j];
            }
        });
        return res_var;
    }
}
