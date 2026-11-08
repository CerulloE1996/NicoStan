#pragma once
//
// -| --------- NicoSIMD induced Dirichlet of ordered cutpoints with a Se / Sp prior at chosen cutoffs, and the
// -| --------- Dirichlet of a simplex with the same Se / Sp prior, Stan external functions -------------------
//
//     nico_induced_dirichlet_with_Se_Sp_prior_lpdf(C | alpha, link, location)
//       = nico_induced_dirichlet_prior_of_ordered_cutpoints_lpdf(C | alpha, link, location) (no Se / Sp prior,
//         e.g. the between-study model of random cutpoints),
//     nico_induced_dirichlet_with_Se_Sp_prior_lpdf(C | alpha, link, location, alpha_of_the_segments,
//                                                  chosen_cutoffs, Se_Sp_prior_type, prior_Se_Sp_dirichlet_A,
//                                                  prior_Se_Sp_beta, prior_Se_Sp_link_mean_SD)
//       = for each set c of K cutpoints (a vector C, or each row of a matrix C), at x = c - location: the induced
//         Dirichlet of c, - dirichlet_lpdf(q | alpha_of_the_segments) + H(x[chosen_cutoffs]),
//     nico_dirichlet_with_Se_Sp_prior_lpdf(phi | log_J, C_baseline, alpha, alpha_of_the_segments, chosen_cutoffs,
//                                          Se_Sp_prior_type, prior_Se_Sp_dirichlet_A, prior_Se_Sp_beta,
//                                          prior_Se_Sp_link_mean_SD, link)
//       = dirichlet_lpdf(phi | alpha) - dirichlet_lpdf(q_phi | alpha_of_the_segments) + sum(log_J)
//         + sum(log f(C_baseline)) + H(C_baseline),
//     nico_Se_Sp_prior_at_chosen_cutoffs_lpdf(x | Se_Sp_prior_type, prior_Se_Sp_dirichlet_A, prior_Se_Sp_beta,
//                                                prior_Se_Sp_link_mean_SD, link)
//       = sum(log f(x)) + H(x) of type 1 or 2 when Se_Sp_prior_type is 1 or 2, of type 4 otherwise (a Dirichlet on
//         the segment probabilities, a Beta on F at each value, or a normal on the link; the m values need not be
//         ordered except for type 1),
//
// with m the number of chosen cutoffs, F = Phi (link = 1) or inv_logit, f its density, q the m + 1 segment
// probabilities of F at the chosen cutoffs (F_1, F_2 - F_1, ..., 1 - F_m), q_phi the sums of phi over the same
// segments, log_J and C_baseline given at the chosen cutoffs (m values each), and H the Se / Sp prior on F at the
// chosen cutoffs:
//
//     Se_Sp_prior_type 1: dirichlet_lpdf(q | prior_Se_Sp_dirichlet_A),
//     Se_Sp_prior_type 2: sum_j beta_lpdf(F_j | prior_Se_Sp_beta[j, 1], prior_Se_Sp_beta[j, 2]),
//     Se_Sp_prior_type 4: sum_j normal_lpdf(x_j | prior_Se_Sp_link_mean_SD[j, 1], prior_Se_Sp_link_mean_SD[j, 2])
//                         - sum_j log f(x_j),
//     any other:          sum_j beta_lpdf(q_j / (1 - F_(j-1)) | prior_Se_Sp_beta[j, ]) - sum_j log1m(F_(j-1)),
//
// F_0 = 0. The Se / Sp prior of ordinal models of test accuracy with a prior on Sp or 1 - Se at chosen cutoffs:
// on fixed cutpoints (the first form) or on the population cutpoints of random cutpoints (the second). Each is ONE
// autodiff node, the expensive functions with BayesMVP's SIMD kernels (log Phi, log_inv_logit, log1m_exp, exp)
// on the log scale, as nico_induced_dirichlet_prior_of_ordered_cutpoints_lpdf, whose forward and reverse passes
// it uses. Every argument may be data or parameters except link, location, chosen_cutoffs and Se_Sp_prior_type;
// the constants are kept as Stan's _lpdf, and left out with propto only when their arguments are data.
//
// Declared in the Stan functions block (no body; stanc --allow-undefined) as any of
//
//     real nico_induced_dirichlet_with_Se_Sp_prior_lpdf(vector C, vector alpha, data int link,
//                                                       data real location);
//     real nico_induced_dirichlet_with_Se_Sp_prior_lpdf(matrix C, vector alpha, data int link,
//                                                       data real location);
//     real nico_induced_dirichlet_with_Se_Sp_prior_lpdf(vector C, vector alpha, data int link,
//                                                       data real location, vector alpha_of_the_segments,
//                                                       data array[] int chosen_cutoffs,
//                                                       data int Se_Sp_prior_type,
//                                                       vector prior_Se_Sp_dirichlet_A, matrix prior_Se_Sp_beta,
//                                                       matrix prior_Se_Sp_link_mean_SD);
//     real nico_dirichlet_with_Se_Sp_prior_lpdf(vector phi, vector log_J, vector C_baseline, vector alpha,
//                                               vector alpha_of_the_segments, data array[] int chosen_cutoffs,
//                                               data int Se_Sp_prior_type, vector prior_Se_Sp_dirichlet_A,
//                                               matrix prior_Se_Sp_beta, matrix prior_Se_Sp_link_mean_SD,
//                                               data int link);
//     real nico_Se_Sp_prior_at_chosen_cutoffs_lpdf(vector x, data int Se_Sp_prior_type,
//                                                  vector prior_Se_Sp_dirichlet_A, matrix prior_Se_Sp_beta,
//                                                  matrix prior_Se_Sp_link_mean_SD, data int link);
//
// -----------------------------------------------------------------------------------------------------------

#include "stan_external_induced_dirichlet_prior_of_ordered_cutpoints.hpp"

#include <stan/math/rev.hpp>
#include <stan/math/prim.hpp>

#include <Eigen/Dense>
#include <cmath>
#include <ostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace nico::math {
namespace Se_Sp_prior_at_chosen_cutoffs {

static constexpr double half_log_of_two_pi = 0.91893853320467274178;




// ---- the arena buffers of the Se / Sp prior at m chosen cutoffs: the values of the hyperparameters, the
//      log-scale values of one set and the partials of H (d_x of one set; the hyperparameters' summed over sets)
struct buffers_of_the_Se_Sp_prior {

    int m = 0;
    double* ones = nullptr;           // m + 1
    double* segments = nullptr;       // m + 1: alpha_of_the_segments
    double* A = nullptr;              // m + 1
    double* a = nullptr;              // m
    double* b = nullptr;              // m
    double* mu = nullptr;             // m
    double* sd = nullptr;             // m
    double* lF = nullptr;             // m
    double* lS = nullptr;             // m + 1
    double* log_q = nullptr;          // 2 m + 1
    double* log_f = nullptr;          // m
    double* d_segments = nullptr;     // m + 1
    double* d_A = nullptr;            // m + 1
    double* d_a = nullptr;            // m
    double* d_b = nullptr;            // m
    double* d_mu = nullptr;           // m
    double* d_sd = nullptr;           // m

    inline static std::size_t n_values(const int m) {

        return static_cast<std::size_t>(18) * m + 7;

    }

    inline void place(double* start, const int m_of_the_prior) {

        m = m_of_the_prior;
        double* next = start;
        auto take = [&next](const int n) { double* out = next; next += n; return out; };
        ones = take(m + 1);
        segments = take(m + 1);
        A = take(m + 1);
        a = take(m);
        b = take(m);
        mu = take(m);
        sd = take(m);
        lF = take(m);
        lS = take(m + 1);
        log_q = take(2 * m + 1);
        log_f = take(m);
        d_segments = take(m + 1);
        d_A = take(m + 1);
        d_a = take(m);
        d_b = take(m);
        d_mu = take(m);
        d_sd = take(m);
        for (int j = 0; j <= m; ++j) {
            ones[j] = 1.0;
            d_segments[j] = 0.0;
            d_A[j] = 0.0;
        }
        for (int j = 0; j < m; ++j) {
            d_a[j] = 0.0;
            d_b[j] = 0.0;
            d_mu[j] = 0.0;
            d_sd[j] = 0.0;
        }

    }

};




// ---- sum_i (w_i - 1) log q_i + (lgamma(sum w) - sum lgamma(w)) [include_the_constant] of the m + 1 segment
//      probabilities, its partials d / d x_j (times sign, added to d_x) and d / d w_i (times sign, added to d_w)
inline double dirichlet_of_the_segments(const double* w, const double* log_q, const double* log_f, const int m,
                                        const bool include_the_constant, const double sign, double* d_x,
                                        double* d_w) {

    double total = 0.0;
    double sum_of_w = 0.0;
    for (int i = 0; i <= m; ++i) {
        if (w[i] != 1.0) total += (w[i] - 1.0) * log_q[i];
        sum_of_w += w[i];
    }
    if (include_the_constant) {
        double sum_of_lgamma = 0.0;
        for (int i = 0; i <= m; ++i) sum_of_lgamma += std::lgamma(w[i]);
        total += std::lgamma(sum_of_w) - sum_of_lgamma;
    }
    if (d_x != nullptr) {
        for (int j = 0; j < m; ++j) {
            double d_F = 0.0;
            if (w[j] != 1.0) d_F += (w[j] - 1.0) * std::exp(log_f[j] - log_q[j]);
            if (w[j + 1] != 1.0) d_F -= (w[j + 1] - 1.0) * std::exp(log_f[j] - log_q[j + 1]);
            d_x[j] += sign * d_F;
        }
    }
    if (d_w != nullptr) {
        const double digamma_of_the_sum = stan::math::digamma(sum_of_w);
        for (int i = 0; i <= m; ++i) {
            d_w[i] += sign * (log_q[i] + digamma_of_the_sum - stan::math::digamma(w[i]));
        }
    }
    return sign * total;

}




// ---- H(x) of one set of m link values x (the header), from the buffers' lF, lS, log_q and log_f; its partials d / d x_j added to d_x, those of the hyperparameters to the
//      buffers' d_ arrays when differentiate_the_hyperparameters
inline double Se_Sp_prior(const double* x, const int link, const int Se_Sp_prior_type,
                          buffers_of_the_Se_Sp_prior& buffers, const bool include_the_constants,
                          const bool differentiate_the_hyperparameters, double* d_x) {

    const int m = buffers.m;
    const double* lF = buffers.lF;
    const double* lS = buffers.lS;
    const double* log_q = buffers.log_q;
    const double* log_f = buffers.log_f;
    double total = 0.0;
    if (Se_Sp_prior_type == 1) {
        //// (Dirichlet on the segment probabilities)
        total += dirichlet_of_the_segments(buffers.A, log_q, log_f, m, include_the_constants, 1.0, d_x,
                                           differentiate_the_hyperparameters ? buffers.d_A : nullptr);
    } else if (Se_Sp_prior_type == 2) {
        //// (Beta on each F_j)
        for (int j = 0; j < m; ++j) {
            const double a = buffers.a[j];
            const double b = buffers.b[j];
            total += (a - 1.0) * lF[j] + (b - 1.0) * lS[j];
            if (include_the_constants) total -= stan::math::lbeta(a, b);
            if (d_x != nullptr) {
                d_x[j] += (a - 1.0) * std::exp(log_f[j] - lF[j]) - (b - 1.0) * std::exp(log_f[j] - lS[j]);
            }
            if (differentiate_the_hyperparameters) {
                const double digamma_of_the_sum = stan::math::digamma(a + b);
                buffers.d_a[j] += lF[j] - stan::math::digamma(a) + digamma_of_the_sum;
                buffers.d_b[j] += lS[j] - stan::math::digamma(b) + digamma_of_the_sum;
            }
        }
    } else if (Se_Sp_prior_type == 4) {
        //// (normal on the link values, minus log f: the Jacobian from F to the link)
        for (int j = 0; j < m; ++j) {
            const double mu = buffers.mu[j];
            const double sd = buffers.sd[j];
            const double z = (x[j] - mu) / sd;
            total += -0.5 * z * z - log_f[j];
            if (include_the_constants) total -= std::log(sd) + half_log_of_two_pi;
            const double d_log_f = (link == 1) ? -x[j] : (std::exp(lS[j]) - std::exp(lF[j]));
            if (d_x != nullptr) d_x[j] += -z / sd - d_log_f;
            if (differentiate_the_hyperparameters) {
                buffers.d_mu[j] += z / sd;
                buffers.d_sd[j] += (z * z - 1.0) / sd;
            }
        }
    } else {
        //// (Beta on each (F_j - F_(j-1)) / (1 - F_(j-1)), minus each log(1 - F_(j-1)); F_0 = 0)
        for (int j = 0; j < m; ++j) {
            const double a = buffers.a[j];
            const double b = buffers.b[j];
            const double lS_previous = (j == 0) ? 0.0 : lS[j - 1];
            const double log_u = log_q[j] - lS_previous;
            const double log1m_u = lS[j] - lS_previous;
            total += (a - 1.0) * log_u + (b - 1.0) * log1m_u - lS_previous;
            if (include_the_constants) total -= stan::math::lbeta(a, b);
            if (d_x != nullptr) {
                d_x[j] += (a - 1.0) * std::exp(log_f[j] - log_q[j]) - (b - 1.0) * std::exp(log_f[j] - lS[j]);
                if (j + 1 < m) {
                    const double a_next = buffers.a[j + 1];
                    const double b_next = buffers.b[j + 1];
                    d_x[j] += -(a_next - 1.0) * std::exp(log_f[j] - log_q[j + 1]) +
                              (a_next + b_next - 1.0) * std::exp(log_f[j] - lS[j]);
                }
            }
            if (differentiate_the_hyperparameters) {
                const double digamma_of_the_sum = stan::math::digamma(a + b);
                buffers.d_a[j] += log_u - stan::math::digamma(a) + digamma_of_the_sum;
                buffers.d_b[j] += log1m_u - stan::math::digamma(b) + digamma_of_the_sum;
            }
        }
    }
    return total;

}




// ---- the log-scale values of one set of m increasing link values x into the buffers (lF, lS, log_q, and log_f the
//      log density of F, with its constant); false when x is not increasing or not a number
inline bool values_of_one_set(const double* x, const int link, buffers_of_the_Se_Sp_prior& buffers) {

    double unused = 0.0;
    const bool ok = nico::math::induced_dirichlet_prior::forward_pass<false>(x, buffers.ones, link, buffers.m,
                                                                            buffers.lF, buffers.lS, buffers.log_q,
                                                                            buffers.log_f, unused);
    if (link == 1) {
        for (int j = 0; j < buffers.m; ++j) buffers.log_f[j] -= half_log_of_two_pi;
    }
    return ok;

}




// ---- lF, lS and log_f of m link values in any order (no segment probabilities) into the buffers
inline void values_of_one_set_in_any_order(const double* x, const int link, buffers_of_the_Se_Sp_prior& buffers) {

    const int m = buffers.m;
    copy_n(x, m, buffers.lF);
    for (int j = 0; j < m; ++j) buffers.lS[j] = -x[j];
    if (link == 1) {
        k_log_Phi(buffers.lF, m);
        k_log_Phi(buffers.lS, m);
        for (int j = 0; j < m; ++j) buffers.log_f[j] = -0.5 * x[j] * x[j] - half_log_of_two_pi;
    } else {
        k_log_inv_logit(buffers.lF, m);
        k_log_inv_logit(buffers.lS, m);
        for (int j = 0; j < m; ++j) buffers.log_f[j] = buffers.lF[j] + buffers.lS[j];
    }

}




// ---- the values of the hyperparameters into the buffers, with Stan's checks of their sizes
template <typename T_segments, typename T_A, typename T_beta, typename T_link_mean_SD>
inline void values_of_the_hyperparameters(const char* function_name, const T_segments& alpha_of_the_segments,
                                          const int Se_Sp_prior_type, const T_A& prior_Se_Sp_dirichlet_A,
                                          const T_beta& prior_Se_Sp_beta,
                                          const T_link_mean_SD& prior_Se_Sp_link_mean_SD,
                                          buffers_of_the_Se_Sp_prior& buffers) {

    const int m = buffers.m;
    stan::math::check_size_match(function_name, "Number of segment weights", alpha_of_the_segments.size(),
                                 "Number of chosen cutoffs plus 1", m + 1);
    for (int i = 0; i <= m; ++i) buffers.segments[i] = stan::math::value_of(alpha_of_the_segments.coeff(i));
    if (Se_Sp_prior_type == 1) {
        stan::math::check_size_match(function_name, "Number of Se / Sp Dirichlet weights",
                                     prior_Se_Sp_dirichlet_A.size(), "Number of chosen cutoffs plus 1", m + 1);
        for (int i = 0; i <= m; ++i) buffers.A[i] = stan::math::value_of(prior_Se_Sp_dirichlet_A.coeff(i));
    } else if (Se_Sp_prior_type == 4) {
        stan::math::check_size_match(function_name, "Rows of the link mean and SD", prior_Se_Sp_link_mean_SD.rows(),
                                     "Number of chosen cutoffs", m);
        for (int j = 0; j < m; ++j) {
            buffers.mu[j] = stan::math::value_of(prior_Se_Sp_link_mean_SD.coeff(j, 0));
            buffers.sd[j] = stan::math::value_of(prior_Se_Sp_link_mean_SD.coeff(j, 1));
        }
    } else {
        stan::math::check_size_match(function_name, "Rows of the Beta parameters", prior_Se_Sp_beta.rows(),
                                     "Number of chosen cutoffs", m);
        for (int j = 0; j < m; ++j) {
            buffers.a[j] = stan::math::value_of(prior_Se_Sp_beta.coeff(j, 0));
            buffers.b[j] = stan::math::value_of(prior_Se_Sp_beta.coeff(j, 1));
        }
    }

}




// ---- the hyperparameters' partials (times the adjoint) into their vars, in the reverse pass
template <typename T_segments_vars, typename T_A_vars, typename T_beta_vars, typename T_link_mean_SD_vars>
inline void add_the_partials_of_the_hyperparameters(const int Se_Sp_prior_type,
                                                    const buffers_of_the_Se_Sp_prior& buffers, const double adjoint,
                                                    T_segments_vars& segments_vars, T_A_vars& A_vars,
                                                    T_beta_vars& beta_vars,
                                                    T_link_mean_SD_vars& link_mean_SD_vars) {

    const int m = buffers.m;
    if (segments_vars.size() > 0) {
        for (int i = 0; i <= m; ++i) segments_vars.coeffRef(i).adj() += adjoint * buffers.d_segments[i];
    }
    if ((Se_Sp_prior_type == 1) && (A_vars.size() > 0)) {
        for (int i = 0; i <= m; ++i) A_vars.coeffRef(i).adj() += adjoint * buffers.d_A[i];
    } else if ((Se_Sp_prior_type == 4) && (link_mean_SD_vars.size() > 0)) {
        for (int j = 0; j < m; ++j) {
            link_mean_SD_vars.coeffRef(j, 0).adj() += adjoint * buffers.d_mu[j];
            link_mean_SD_vars.coeffRef(j, 1).adj() += adjoint * buffers.d_sd[j];
        }
    } else if ((Se_Sp_prior_type != 1) && (Se_Sp_prior_type != 4) && (beta_vars.size() > 0)) {
        for (int j = 0; j < m; ++j) {
            beta_vars.coeffRef(j, 0).adj() += adjoint * buffers.d_a[j];
            beta_vars.coeffRef(j, 1).adj() += adjoint * buffers.d_b[j];
        }
    }

}




// ---- the chosen cutoffs (1-based in Stan) checked: increasing, within 1, ..., K
inline void check_the_chosen_cutoffs(const char* function_name, const std::vector<int>& chosen_cutoffs,
                                     const int K) {

    for (std::size_t j = 0; j < chosen_cutoffs.size(); ++j) {
        if ((chosen_cutoffs[j] < 1) || (chosen_cutoffs[j] > K) ||
            ((j > 0) && (chosen_cutoffs[j] <= chosen_cutoffs[j - 1]))) {
            throw std::domain_error(std::string(function_name) + ": chosen_cutoffs must rise within 1, ..., " +
                                    std::to_string(K));
        }
    }

}

}  // namespace Se_Sp_prior_at_chosen_cutoffs
}  // namespace nico::math




// ---- the Stan-facing functions (global scope; stanc calls them with propto first and pstream__ last)
namespace nico::math {

//// (no Se / Sp prior: the induced Dirichlet itself)
template <bool propto, typename T_C, typename T_alpha, stan::require_eigen_t<T_C>* = nullptr,
          stan::require_eigen_col_vector_t<T_alpha>* = nullptr>
inline stan::return_type_t<T_C, T_alpha> nico_induced_dirichlet_with_Se_Sp_prior_lpdf(
        const T_C& C, const T_alpha& alpha, const int link, const double location, std::ostream* pstream__) {

    return nico_induced_dirichlet_prior_of_ordered_cutpoints_lpdf<propto>(C, alpha, link, location, pstream__);

}




//// (with the Se / Sp prior at the chosen cutoffs of every set)
template <bool propto, typename T_C, typename T_alpha, typename T_segments, typename T_A, typename T_beta,
          typename T_link_mean_SD, stan::require_eigen_t<T_C>* = nullptr,
          stan::require_eigen_col_vector_t<T_alpha>* = nullptr>
inline stan::return_type_t<T_C, T_alpha, T_segments, T_A, T_beta, T_link_mean_SD>
nico_induced_dirichlet_with_Se_Sp_prior_lpdf(const T_C& C, const T_alpha& alpha, const int link,
                                             const double location, const T_segments& alpha_of_the_segments,
                                             const std::vector<int>& chosen_cutoffs, const int Se_Sp_prior_type,
                                             const T_A& prior_Se_Sp_dirichlet_A, const T_beta& prior_Se_Sp_beta,
                                             const T_link_mean_SD& prior_Se_Sp_link_mean_SD,
                                             std::ostream* pstream__) {

    namespace prior = nico::math::induced_dirichlet_prior;
    namespace Se_Sp = nico::math::Se_Sp_prior_at_chosen_cutoffs;
    using stan::math::var;
    static constexpr const char* function_name = "nico_induced_dirichlet_with_Se_Sp_prior_lpdf";
    const int m = static_cast<int>(chosen_cutoffs.size());
    if (m == 0) {
        return nico_induced_dirichlet_prior_of_ordered_cutpoints_lpdf<propto>(C, alpha, link, location, pstream__);
    }
    constexpr bool C_is_one_set = stan::is_eigen_vector<T_C>::value;
    constexpr bool C_is_constant = stan::is_constant_all<T_C>::value;
    constexpr bool alpha_is_constant = stan::is_constant_all<T_alpha>::value;
    constexpr bool hyperparameters_are_constant =
            stan::is_constant_all<T_segments, T_A, T_beta, T_link_mean_SD>::value;
    constexpr bool everything_is_constant = C_is_constant && alpha_is_constant && hyperparameters_are_constant;
    constexpr bool include_the_lgamma_terms = !propto || !alpha_is_constant;
    constexpr bool include_the_constant_of_the_normal = !propto;
    constexpr bool include_the_constants_of_the_Se_Sp_prior = !propto || !hyperparameters_are_constant;
    const int R = C_is_one_set ? 1 : static_cast<int>(C.rows());
    const int K = C_is_one_set ? static_cast<int>(C.size()) : static_cast<int>(C.cols());
    stan::math::check_size_match(function_name, "Number of prior weights", alpha.size(),
                                 "Number of cutpoints plus 1", K + 1);
    Se_Sp::check_the_chosen_cutoffs(function_name, chosen_cutoffs, K);
    if (R == 0) return 0.0;
    if constexpr (propto && everything_is_constant) return 0.0;
    const auto& C_ref = stan::math::to_ref(C);
    const auto& alpha_ref = stan::math::to_ref(alpha);
    //// (the buffers, as nico_induced_dirichlet_prior_of_ordered_cutpoints_lpdf's, then those of the Se / Sp prior:
    ////  the chosen link values of every set, the partials of H with respect to them (R m), and one set's work
    ////  space; on the arena when anything is differentiated, in a std::vector otherwise)
    const std::size_t RK = static_cast<std::size_t>(R) * K;
    const std::size_t Rm = static_cast<std::size_t>(R) * m;
    const std::size_t n_buffer_values = 3 * RK + (K + 1) + static_cast<std::size_t>(R) * (K + 1) +
                                        static_cast<std::size_t>(R) * (2 * K + 1) +
                                        2 * static_cast<std::size_t>(K) + 2 * Rm +
                                        Se_Sp::buffers_of_the_Se_Sp_prior::n_values(m);
    std::vector<double> buffer_without_autodiff;
    double* buffers = nullptr;
    if constexpr (everything_is_constant) {
        buffer_without_autodiff.resize(n_buffer_values);
        buffers = buffer_without_autodiff.data();
    } else {
        buffers = stan::math::ChainableStack::instance_->memalloc_.alloc_array<double>(n_buffer_values);
    }
    double* x = buffers;
    double* alpha_kept = x + RK;
    double* F = alpha_kept + (K + 1);
    double* p = F + RK;
    double* log_p = p + static_cast<std::size_t>(R) * (K + 1);
    double* J_terms = log_p + static_cast<std::size_t>(R) * (2 * K + 1);
    double* ratios = J_terms + RK;
    double* x_chosen = ratios + 2 * static_cast<std::size_t>(K);
    double* d_x_chosen = x_chosen + Rm;
    Se_Sp::buffers_of_the_Se_Sp_prior Se_Sp_buffers;
    Se_Sp_buffers.place(d_x_chosen + Rm, m);
    for (int r = 0; r < R; ++r) {
        for (int k = 0; k < K; ++k) {
            const double value_of_C = C_is_one_set ? stan::math::value_of(C_ref.coeff(k))
                                                   : stan::math::value_of(C_ref.coeff(r, k));
            x[static_cast<std::size_t>(r) * K + k] = value_of_C - location;
        }
        for (int j = 0; j < m; ++j) {
            x_chosen[static_cast<std::size_t>(r) * m + j] =
                    x[static_cast<std::size_t>(r) * K + (chosen_cutoffs[j] - 1)];
            d_x_chosen[static_cast<std::size_t>(r) * m + j] = 0.0;
        }
    }
    for (int j = 0; j <= K; ++j) alpha_kept[j] = stan::math::value_of(alpha_ref.coeff(j));
    Se_Sp::values_of_the_hyperparameters(function_name, alpha_of_the_segments, Se_Sp_prior_type,
                                         prior_Se_Sp_dirichlet_A, prior_Se_Sp_beta, prior_Se_Sp_link_mean_SD,
                                         Se_Sp_buffers);
    double sum = 0.0;
    for (int r = 0; r < R; ++r) {
        //// (the induced Dirichlet of the set)
        double sum_of_set = 0.0;
        double* p_r = p + static_cast<std::size_t>(r) * (K + 1);
        const std::size_t start_of_set = static_cast<std::size_t>(r) * K;
        if (!prior::forward_pass<include_the_constant_of_the_normal>(
                    x + start_of_set, alpha_kept, link, K, F + start_of_set, p_r,
                    log_p + static_cast<std::size_t>(r) * (2 * K + 1), J_terms + start_of_set, sum_of_set)) {

            Eigen::Matrix<double, -1, 1> p_of_the_set(K + 1);
            const double* log_p_r = log_p + static_cast<std::size_t>(r) * (2 * K + 1);
            for (int j = 0; j <= K; ++j) p_of_the_set(j) = std::exp(log_p_r[j]);
            prior::run_the_checks_of_Stan(p_of_the_set, Eigen::Map<const Eigen::Matrix<double, -1, 1>>(alpha_kept,
                                                                                                       K + 1));

        }
        sum += sum_of_set;
        //// (minus the Dirichlet of its segment probabilities, plus the Se / Sp prior, at the chosen cutoffs)
        const double* x_r = x_chosen + static_cast<std::size_t>(r) * m;
        double* d_x_r = d_x_chosen + static_cast<std::size_t>(r) * m;
        if (!Se_Sp::values_of_one_set(x_r, link, Se_Sp_buffers)) {
            throw std::domain_error(std::string(function_name) + ": the cutpoints at the chosen cutoffs are not "
                                    "increasing");
        }
        sum += Se_Sp::dirichlet_of_the_segments(Se_Sp_buffers.segments, Se_Sp_buffers.log_q,
                                                Se_Sp_buffers.log_f, m,
                                                include_the_constants_of_the_Se_Sp_prior, -1.0, d_x_r,
                                                stan::is_constant_all<T_segments>::value
                                                        ? nullptr : Se_Sp_buffers.d_segments);
        sum += Se_Sp::Se_Sp_prior(x_r, link, Se_Sp_prior_type, Se_Sp_buffers,
                                  include_the_constants_of_the_Se_Sp_prior, !hyperparameters_are_constant, d_x_r);
    }
    if constexpr (include_the_lgamma_terms) {
        double sum_of_alpha = 0.0;
        double sum_of_lgamma_of_alpha = 0.0;
        for (int j = 0; j <= K; ++j) {
            sum_of_alpha += alpha_kept[j];
            sum_of_lgamma_of_alpha += std::lgamma(alpha_kept[j]);
        }
        sum += R * (std::lgamma(sum_of_alpha) - sum_of_lgamma_of_alpha);
    }
    if constexpr (everything_is_constant) {
        return sum;
    } else {
        using C_vars_type = std::conditional_t<C_is_constant, Eigen::Matrix<var, -1, -1>,
                                               stan::plain_type_t<stan::math::promote_scalar_t<var, T_C>>>;
        stan::arena_t<C_vars_type> C_vars;
        stan::arena_t<Eigen::Matrix<var, -1, 1>> alpha_vars;
        stan::arena_t<Eigen::Matrix<var, -1, 1>> segments_vars;
        stan::arena_t<Eigen::Matrix<var, -1, 1>> A_vars;
        stan::arena_t<Eigen::Matrix<var, -1, -1>> beta_vars;
        stan::arena_t<Eigen::Matrix<var, -1, -1>> link_mean_SD_vars;
        if constexpr (!C_is_constant) C_vars = C_ref;
        if constexpr (!alpha_is_constant) alpha_vars = alpha_ref;
        if constexpr (!stan::is_constant_all<T_segments>::value) segments_vars = alpha_of_the_segments;
        if constexpr (!stan::is_constant_all<T_A>::value) A_vars = prior_Se_Sp_dirichlet_A;
        if constexpr (!stan::is_constant_all<T_beta>::value) beta_vars = prior_Se_Sp_beta;
        if constexpr (!stan::is_constant_all<T_link_mean_SD>::value) link_mean_SD_vars = prior_Se_Sp_link_mean_SD;
        int* chosen_kept = stan::math::ChainableStack::instance_->memalloc_.alloc_array<int>(m);
        for (int j = 0; j < m; ++j) chosen_kept[j] = chosen_cutoffs[j] - 1;
        return stan::math::make_callback_var(sum, [C_vars, alpha_vars, segments_vars, A_vars, beta_vars,
                                                   link_mean_SD_vars, x, alpha_kept, link, R, K, m, F, p, log_p,
                                                   J_terms, ratios, d_x_chosen, chosen_kept, Se_Sp_buffers,
                                                   Se_Sp_prior_type](auto& vi) mutable {
            if constexpr (!C_is_constant) {
                for (int r = 0; r < R; ++r) {
                    double* d_x = J_terms + static_cast<std::size_t>(r) * K;
                    prior::reverse_pass(x + static_cast<std::size_t>(r) * K, alpha_kept, link, K,
                                        F + static_cast<std::size_t>(r) * K,
                                        p + static_cast<std::size_t>(r) * (K + 1),
                                        log_p + static_cast<std::size_t>(r) * (2 * K + 1), vi.adj(), d_x,
                                        ratios, ratios + K);
                    for (int j = 0; j < m; ++j) {
                        d_x[chosen_kept[j]] += vi.adj() * d_x_chosen[static_cast<std::size_t>(r) * m + j];
                    }
                    for (int k = 0; k < K; ++k) {
                        if constexpr (C_is_one_set) {
                            C_vars.coeffRef(k).adj() += d_x[k];
                        } else {
                            C_vars.coeffRef(r, k).adj() += d_x[k];
                        }
                    }
                }
            }
            if constexpr (!alpha_is_constant) {
                double sum_of_alpha = 0.0;
                for (int j = 0; j <= K; ++j) sum_of_alpha += alpha_kept[j];
                const double digamma_of_the_sum = stan::math::digamma(sum_of_alpha);
                for (int j = 0; j <= K; ++j) {
                    double sum_of_log_p = 0.0;
                    for (int r = 0; r < R; ++r) {
                        sum_of_log_p += log_p[static_cast<std::size_t>(r) * (2 * K + 1) + j];
                    }
                    const double d_alpha = sum_of_log_p +
                                           R * (digamma_of_the_sum - stan::math::digamma(alpha_kept[j]));
                    alpha_vars.coeffRef(j).adj() += vi.adj() * d_alpha;
                }
            }
            Se_Sp::add_the_partials_of_the_hyperparameters(Se_Sp_prior_type, Se_Sp_buffers, vi.adj(),
                                                           segments_vars, A_vars, beta_vars, link_mean_SD_vars);
        });
    }

}

}  // namespace nico::math

using nico::math::nico_induced_dirichlet_with_Se_Sp_prior_lpdf;




// ---- the Dirichlet of a simplex phi with the Se / Sp prior at the chosen cutoffs (global scope)
namespace nico::math {

template <bool propto, typename T_phi, typename T_log_J, typename T_x, typename T_alpha, typename T_segments,
          typename T_A, typename T_beta, typename T_link_mean_SD,
          stan::require_eigen_col_vector_t<T_phi>* = nullptr>
inline stan::return_type_t<T_phi, T_log_J, T_x, T_alpha, T_segments, T_A, T_beta, T_link_mean_SD>
nico_dirichlet_with_Se_Sp_prior_lpdf(const T_phi& phi, const T_log_J& log_J, const T_x& C_baseline,
                                     const T_alpha& alpha, const T_segments& alpha_of_the_segments,
                                     const std::vector<int>& chosen_cutoffs, const int Se_Sp_prior_type,
                                     const T_A& prior_Se_Sp_dirichlet_A, const T_beta& prior_Se_Sp_beta,
                                     const T_link_mean_SD& prior_Se_Sp_link_mean_SD, const int link,
                                     std::ostream* /*pstream__*/) {

    namespace Se_Sp = nico::math::Se_Sp_prior_at_chosen_cutoffs;
    using stan::math::var;
    static constexpr const char* function_name = "nico_dirichlet_with_Se_Sp_prior_lpdf";
    const int n_cat = static_cast<int>(phi.size());
    const int m = static_cast<int>(chosen_cutoffs.size());
    constexpr bool phi_is_constant = stan::is_constant_all<T_phi>::value;
    constexpr bool log_J_is_constant = stan::is_constant_all<T_log_J>::value;
    constexpr bool x_is_constant = stan::is_constant_all<T_x>::value;
    constexpr bool alpha_is_constant = stan::is_constant_all<T_alpha>::value;
    constexpr bool segments_are_constant = stan::is_constant_all<T_segments>::value;
    constexpr bool hyperparameters_are_constant =
            stan::is_constant_all<T_segments, T_A, T_beta, T_link_mean_SD>::value;
    constexpr bool everything_is_constant = phi_is_constant && log_J_is_constant && x_is_constant &&
                                            alpha_is_constant && hyperparameters_are_constant;
    constexpr bool include_the_lgamma_terms = !propto || !alpha_is_constant;
    constexpr bool include_the_constants_of_the_Se_Sp_prior = !propto || !hyperparameters_are_constant;
    stan::math::check_size_match(function_name, "Number of prior weights", alpha.size(), "Number of categories",
                                 n_cat);
    stan::math::check_size_match(function_name, "Number of log_J", log_J.size(), "Number of chosen cutoffs", m);
    stan::math::check_size_match(function_name, "Number of C_baseline", C_baseline.size(),
                                 "Number of chosen cutoffs", m);
    Se_Sp::check_the_chosen_cutoffs(function_name, chosen_cutoffs, n_cat - 1);
    if constexpr (propto && everything_is_constant) return 0.0;
    const auto& phi_ref = stan::math::to_ref(phi);
    const auto& log_J_ref = stan::math::to_ref(log_J);
    const auto& x_ref = stan::math::to_ref(C_baseline);
    const auto& alpha_ref = stan::math::to_ref(alpha);
    //// (the buffers: phi, alpha, the log of the m + 1 segment sums of phi, x, d_x (m), then the Se / Sp prior's)
    const std::size_t n_buffer_values = 2 * static_cast<std::size_t>(n_cat) + (m + 1) + 2 * m +
                                        Se_Sp::buffers_of_the_Se_Sp_prior::n_values(m);
    std::vector<double> buffer_without_autodiff;
    double* buffers = nullptr;
    if constexpr (everything_is_constant) {
        buffer_without_autodiff.resize(n_buffer_values);
        buffers = buffer_without_autodiff.data();
    } else {
        buffers = stan::math::ChainableStack::instance_->memalloc_.alloc_array<double>(n_buffer_values);
    }
    double* phi_kept = buffers;
    double* alpha_kept = phi_kept + n_cat;
    double* log_q_phi = alpha_kept + n_cat;
    double* x = log_q_phi + (m + 1);
    double* d_x = x + m;
    Se_Sp::buffers_of_the_Se_Sp_prior Se_Sp_buffers;
    Se_Sp_buffers.place(d_x + m, m);
    for (int n = 0; n < n_cat; ++n) {
        phi_kept[n] = stan::math::value_of(phi_ref.coeff(n));
        alpha_kept[n] = stan::math::value_of(alpha_ref.coeff(n));
    }
    for (int j = 0; j < m; ++j) {
        x[j] = stan::math::value_of(x_ref.coeff(j));
        d_x[j] = 0.0;
    }
    Se_Sp::values_of_the_hyperparameters(function_name, alpha_of_the_segments, Se_Sp_prior_type,
                                         prior_Se_Sp_dirichlet_A, prior_Se_Sp_beta, prior_Se_Sp_link_mean_SD,
                                         Se_Sp_buffers);
    //// (the Dirichlet of phi)
    double sum = 0.0;
    double sum_of_alpha = 0.0;
    for (int n = 0; n < n_cat; ++n) {
        if (alpha_kept[n] != 1.0) sum += (alpha_kept[n] - 1.0) * std::log(phi_kept[n]);
        sum_of_alpha += alpha_kept[n];
    }
    if constexpr (include_the_lgamma_terms) {
        double sum_of_lgamma_of_alpha = 0.0;
        for (int n = 0; n < n_cat; ++n) sum_of_lgamma_of_alpha += std::lgamma(alpha_kept[n]);
        sum += std::lgamma(sum_of_alpha) - sum_of_lgamma_of_alpha;
    }
    //// (minus the Dirichlet of the segment sums of phi: segment s ends at the chosen cutoff s, the last at n_cat)
    {
        int start = 0;
        for (int s = 0; s <= m; ++s) {
            const int end = (s < m) ? chosen_cutoffs[s] : n_cat;
            double q = 0.0;
            for (int n = start; n < end; ++n) q += phi_kept[n];
            log_q_phi[s] = std::log(q);
            start = end;
        }
        sum += Se_Sp::dirichlet_of_the_segments(Se_Sp_buffers.segments, log_q_phi, nullptr, m,
                                                include_the_constants_of_the_Se_Sp_prior, -1.0, nullptr,
                                                segments_are_constant ? nullptr : Se_Sp_buffers.d_segments);
    }
    //// (sum(log_J), then sum(log f(x)) and the Se / Sp prior at x = C_baseline)
    for (int j = 0; j < m; ++j) sum += stan::math::value_of(log_J_ref.coeff(j));
    if (!Se_Sp::values_of_one_set(x, link, Se_Sp_buffers)) {
        throw std::domain_error(std::string(function_name) + ": C_baseline is not increasing");
    }
    for (int j = 0; j < m; ++j) {
        sum += Se_Sp_buffers.log_f[j];
        d_x[j] += (link == 1) ? -x[j] : (std::exp(Se_Sp_buffers.lS[j]) - std::exp(Se_Sp_buffers.lF[j]));
    }
    sum += Se_Sp::Se_Sp_prior(x, link, Se_Sp_prior_type, Se_Sp_buffers, include_the_constants_of_the_Se_Sp_prior,
                              !hyperparameters_are_constant, d_x);
    if constexpr (everything_is_constant) {
        return sum;
    } else {
        stan::arena_t<Eigen::Matrix<var, -1, 1>> phi_vars;
        stan::arena_t<Eigen::Matrix<var, -1, 1>> log_J_vars;
        stan::arena_t<Eigen::Matrix<var, -1, 1>> x_vars;
        stan::arena_t<Eigen::Matrix<var, -1, 1>> alpha_vars;
        stan::arena_t<Eigen::Matrix<var, -1, 1>> segments_vars;
        stan::arena_t<Eigen::Matrix<var, -1, 1>> A_vars;
        stan::arena_t<Eigen::Matrix<var, -1, -1>> beta_vars;
        stan::arena_t<Eigen::Matrix<var, -1, -1>> link_mean_SD_vars;
        if constexpr (!phi_is_constant) phi_vars = phi_ref;
        if constexpr (!log_J_is_constant) log_J_vars = log_J_ref;
        if constexpr (!x_is_constant) x_vars = x_ref;
        if constexpr (!alpha_is_constant) alpha_vars = alpha_ref;
        if constexpr (!segments_are_constant) segments_vars = alpha_of_the_segments;
        if constexpr (!stan::is_constant_all<T_A>::value) A_vars = prior_Se_Sp_dirichlet_A;
        if constexpr (!stan::is_constant_all<T_beta>::value) beta_vars = prior_Se_Sp_beta;
        if constexpr (!stan::is_constant_all<T_link_mean_SD>::value) link_mean_SD_vars = prior_Se_Sp_link_mean_SD;
        int* segment_ends = stan::math::ChainableStack::instance_->memalloc_.alloc_array<int>(m + 1);
        for (int s = 0; s < m; ++s) segment_ends[s] = chosen_cutoffs[s];
        segment_ends[m] = n_cat;
        return stan::math::make_callback_var(sum, [phi_vars, log_J_vars, x_vars, alpha_vars, segments_vars, A_vars,
                                                   beta_vars, link_mean_SD_vars, phi_kept, alpha_kept, log_q_phi,
                                                   d_x, segment_ends, n_cat, m, sum_of_alpha, Se_Sp_buffers,
                                                   Se_Sp_prior_type](auto& vi) mutable {
            const double adjoint = vi.adj();
            if constexpr (!phi_is_constant) {
                int start = 0;
                for (int s = 0; s <= m; ++s) {
                    const double w = Se_Sp_buffers.segments[s];
                    const double d_segment = (w != 1.0) ? (w - 1.0) * std::exp(-log_q_phi[s]) : 0.0;
                    for (int n = start; n < segment_ends[s]; ++n) {
                        const double d_dirichlet = (alpha_kept[n] != 1.0) ? (alpha_kept[n] - 1.0) / phi_kept[n]
                                                                          : 0.0;
                        phi_vars.coeffRef(n).adj() += adjoint * (d_dirichlet - d_segment);
                    }
                    start = segment_ends[s];
                }
            }
            if constexpr (!log_J_is_constant) {
                for (int j = 0; j < m; ++j) log_J_vars.coeffRef(j).adj() += adjoint;
            }
            if constexpr (!x_is_constant) {
                for (int j = 0; j < m; ++j) x_vars.coeffRef(j).adj() += adjoint * d_x[j];
            }
            if constexpr (!alpha_is_constant) {
                const double digamma_of_the_sum = stan::math::digamma(sum_of_alpha);
                for (int n = 0; n < n_cat; ++n) {
                    alpha_vars.coeffRef(n).adj() += adjoint * (std::log(phi_kept[n]) + digamma_of_the_sum -
                                                               stan::math::digamma(alpha_kept[n]));
                }
            }
            Se_Sp::add_the_partials_of_the_hyperparameters(Se_Sp_prior_type, Se_Sp_buffers, adjoint,
                                                           segments_vars, A_vars, beta_vars, link_mean_SD_vars);
        });
    }

}

}  // namespace nico::math

using nico::math::nico_dirichlet_with_Se_Sp_prior_lpdf;




// ---- the Se / Sp prior at m link values in any order (global scope)
namespace nico::math {

template <bool propto, typename T_x, typename T_A, typename T_beta, typename T_link_mean_SD,
          stan::require_eigen_col_vector_t<T_x>* = nullptr>
inline stan::return_type_t<T_x, T_A, T_beta, T_link_mean_SD>
nico_Se_Sp_prior_at_chosen_cutoffs_lpdf(const T_x& x, const int Se_Sp_prior_type, const T_A& prior_Se_Sp_dirichlet_A,
                                        const T_beta& prior_Se_Sp_beta,
                                        const T_link_mean_SD& prior_Se_Sp_link_mean_SD, const int link,
                                        std::ostream* /*pstream__*/) {

    namespace Se_Sp = nico::math::Se_Sp_prior_at_chosen_cutoffs;
    using stan::math::var;
    static constexpr const char* function_name = "nico_Se_Sp_prior_at_chosen_cutoffs_lpdf";
    const int m = static_cast<int>(x.size());
    const int type_of_H = (Se_Sp_prior_type == 1 || Se_Sp_prior_type == 2) ? Se_Sp_prior_type : 4;
    constexpr bool x_is_constant = stan::is_constant_all<T_x>::value;
    constexpr bool hyperparameters_are_constant = stan::is_constant_all<T_A, T_beta, T_link_mean_SD>::value;
    constexpr bool everything_is_constant = x_is_constant && hyperparameters_are_constant;
    constexpr bool include_the_constants = !propto || !hyperparameters_are_constant;
    if (m == 0) return 0.0;
    if constexpr (propto && everything_is_constant) return 0.0;
    const auto& x_ref = stan::math::to_ref(x);
    //// (the buffers: x, d_x (m), then the Se / Sp prior's; the segment weights are not used)
    const std::size_t n_buffer_values = 2 * static_cast<std::size_t>(m) +
                                        Se_Sp::buffers_of_the_Se_Sp_prior::n_values(m);
    std::vector<double> buffer_without_autodiff;
    double* buffers = nullptr;
    if constexpr (everything_is_constant) {
        buffer_without_autodiff.resize(n_buffer_values);
        buffers = buffer_without_autodiff.data();
    } else {
        buffers = stan::math::ChainableStack::instance_->memalloc_.alloc_array<double>(n_buffer_values);
    }
    double* x_kept = buffers;
    double* d_x = x_kept + m;
    Se_Sp::buffers_of_the_Se_Sp_prior Se_Sp_buffers;
    Se_Sp_buffers.place(d_x + m, m);
    for (int j = 0; j < m; ++j) {
        x_kept[j] = stan::math::value_of(x_ref.coeff(j));
        d_x[j] = 0.0;
    }
    const auto& segments_unused = Eigen::Matrix<double, -1, 1>::Ones(m + 1);
    Se_Sp::values_of_the_hyperparameters(function_name, segments_unused, type_of_H, prior_Se_Sp_dirichlet_A,
                                         prior_Se_Sp_beta, prior_Se_Sp_link_mean_SD, Se_Sp_buffers);
    if (type_of_H == 1) {
        if (!Se_Sp::values_of_one_set(x_kept, link, Se_Sp_buffers)) {
            throw std::domain_error(std::string(function_name) + ": x is not increasing");
        }
    } else {
        Se_Sp::values_of_one_set_in_any_order(x_kept, link, Se_Sp_buffers);
    }
    double sum = 0.0;
    for (int j = 0; j < m; ++j) {
        sum += Se_Sp_buffers.log_f[j];
        d_x[j] += (link == 1) ? -x_kept[j] : (std::exp(Se_Sp_buffers.lS[j]) - std::exp(Se_Sp_buffers.lF[j]));
    }
    sum += Se_Sp::Se_Sp_prior(x_kept, link, type_of_H, Se_Sp_buffers, include_the_constants,
                              !hyperparameters_are_constant, d_x);
    if constexpr (everything_is_constant) {
        return sum;
    } else {
        stan::arena_t<Eigen::Matrix<var, -1, 1>> x_vars;
        stan::arena_t<Eigen::Matrix<var, -1, 1>> segments_vars;
        stan::arena_t<Eigen::Matrix<var, -1, 1>> A_vars;
        stan::arena_t<Eigen::Matrix<var, -1, -1>> beta_vars;
        stan::arena_t<Eigen::Matrix<var, -1, -1>> link_mean_SD_vars;
        if constexpr (!x_is_constant) x_vars = x_ref;
        if constexpr (!stan::is_constant_all<T_A>::value) A_vars = prior_Se_Sp_dirichlet_A;
        if constexpr (!stan::is_constant_all<T_beta>::value) beta_vars = prior_Se_Sp_beta;
        if constexpr (!stan::is_constant_all<T_link_mean_SD>::value) link_mean_SD_vars = prior_Se_Sp_link_mean_SD;
        return stan::math::make_callback_var(sum, [x_vars, segments_vars, A_vars, beta_vars, link_mean_SD_vars, d_x,
                                                   m, Se_Sp_buffers, type_of_H](auto& vi) mutable {
            if constexpr (!x_is_constant) {
                for (int j = 0; j < m; ++j) x_vars.coeffRef(j).adj() += vi.adj() * d_x[j];
            }
            Se_Sp::add_the_partials_of_the_hyperparameters(type_of_H, Se_Sp_buffers, vi.adj(), segments_vars, A_vars,
                                                           beta_vars, link_mean_SD_vars);
        });
    }

}

}  // namespace nico::math

using nico::math::nico_Se_Sp_prior_at_chosen_cutoffs_lpdf;
