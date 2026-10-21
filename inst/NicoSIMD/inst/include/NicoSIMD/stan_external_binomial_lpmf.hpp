#pragma once
//
// -| --------- NicoSIMD binomial log probability, a Stan external function -------------------------------------
//
//     nico_binomial_lpmf(n | N, theta) = sum_i [ n_i log(theta_i) + (N_i - n_i) log1m(theta_i)
//                                               + log choose(N_i, n_i) ],
//
// the log probability of Stan's binomial_lpmf(n | N, theta), with n and N arrays of int (data) and theta a vector
// of probabilities. A general function: any binomial counts with their own probabilities (e.g. the counts above
// each threshold of ordinal and continuation-ratio models, dose-response data, aggregated Bernoulli data).
// ONE autodiff node for the whole vector, with the BayesMVP log and log1m kernels over all its elements (Stan's
// version: one node too, but std::log, log1m and three lgamma() per element, at every call), and:
//   - log choose(N_i, n_i) = log N_i! - log n_i! - log (N_i - n_i)!, from a table of log k! for k < 4096 (made
//     once, with std::lgamma(k + 1); std::lgamma above it), left out with propto, as in Stan;
//   - n_i log(theta_i) is 0 when n_i = 0, and (N_i - n_i) log1m(theta_i) is 0 when n_i = N_i, as in Stan (so
//     theta_i = 0 or 1 is allowed where its counts allow it);
//   - d / d theta_i = n_i / theta_i - (N_i - n_i) / (1 - theta_i), with the same two cases.
// The arguments are checked with Stan's own check functions (and so the same exceptions) only when the pass over
// the values finds a bad one (n_i outside 0, ..., N_i, or theta_i outside [0, 1] or NaN), so that valid
// arguments cost no extra pass.
//
// Declared in the Stan functions block (no body; stanc --allow-undefined) as
//
//     real nico_binomial_lpmf(array[] int n, array[] int N, vector theta);
//
// which also gives  nico_binomial_lupmf(n | N, theta)  and  n ~ nico_binomial(N, theta).
//
// It uses the namespace nico::math, BMVP_LANES and the kernels k_log() and k_log1m() of
// stan_external_functions.hpp, which it includes (#pragma once), so that a user header may include this file
// alone. Its helpers are in their own namespace, nico::math::binomial.
//
// -----------------------------------------------------------------------------------------------------------

#include "stan_external_functions.hpp"

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
namespace binomial {

static constexpr const char* function_name = "nico_binomial_lpmf";
static constexpr int size_of_the_table_of_log_factorials = 4096;




// ---- log k! for k = 0, ..., 4095 (a function-local static: made once, by the first call, thread-safe)
inline const double* table_of_log_factorials() {

    static const std::vector<double> table = [] {
        std::vector<double> log_factorials(size_of_the_table_of_log_factorials);
        for (int k = 0; k < size_of_the_table_of_log_factorials; ++k) {
            log_factorials[k] = std::lgamma(k + 1.0);
        }
        return log_factorials;
    }();
    return table.data();

}

inline double log_factorial(const int k, const double* table) {

    return (k < size_of_the_table_of_log_factorials) ? table[k] : std::lgamma(k + 1.0);

}




// ---- the sum of the log probabilities, from the values of theta (M elements). log_theta and log1m_theta:
//      buffers of M doubles; with compute_partials, d / d theta_i is written to partials[i]. Returns false,
//      and nothing else, when the pass over the values finds a bad argument (the caller then runs Stan's
//      checks).
template <bool include_log_choose, bool compute_partials>
inline bool sum_of_log_probabilities(const int* n, const int* N, const double* theta, const int M,
                                     double* log_theta, double* log1m_theta, double* partials, double& sum) {

    int n_bad = 0;
    for (int i = 0; i < M; ++i) {
        const double theta_i = theta[i];
        n_bad += (n[i] < 0) | (n[i] > N[i]) | !(theta_i >= 0.0 && theta_i <= 1.0);
        log_theta[i] = theta_i;
        log1m_theta[i] = theta_i;
    }
    if (n_bad > 0) return false;
    k_log(log_theta, M);
    k_log1m(log1m_theta, M);
    double total = 0.0;
    for (int i = 0; i < M; ++i) {
        const double successes = n[i];
        const double failures = N[i] - n[i];
        total += ((successes > 0.0) ? successes * log_theta[i] : 0.0)
                 + ((failures > 0.0) ? failures * log1m_theta[i] : 0.0);
        if constexpr (compute_partials) {
            partials[i] = ((successes > 0.0) ? successes / theta[i] : 0.0)
                          - ((failures > 0.0) ? failures / (1.0 - theta[i]) : 0.0);
        }
    }
    if constexpr (include_log_choose) {
        const double* table = table_of_log_factorials();
        for (int i = 0; i < M; ++i) {
            total += log_factorial(N[i], table) - log_factorial(n[i], table) - log_factorial(N[i] - n[i], table);
        }
    }
    sum = total;
    return true;

}




// ---- Stan's checks of the arguments (they throw; reached only when the pass over the values found a bad one)
template <typename T_n, typename T_N, typename T_theta_values>
inline void run_the_checks_of_Stan(const T_n& n, const T_N& N, const T_theta_values& theta_values) {

    stan::math::check_nonnegative(function_name, "Population size parameter", N);
    stan::math::check_bounded(function_name, "Successes variable", n, 0, N);
    stan::math::check_bounded(function_name, "Probability parameter", theta_values, 0.0, 1.0);
    throw std::domain_error(std::string(function_name) + ": an argument is out of its range");

}

}  // namespace binomial
}  // namespace nico::math




// ---- the Stan-facing function (global scope; stanc calls it as  nico_binomial_lpmf<propto__>(n, N, theta,
//      pstream__), with n and N arrays of int and theta a vector or vector expression, of double or of var)
namespace nico::math {

template <bool propto, typename T_n, typename T_N, typename T_theta,
          stan::require_all_std_vector_vt<std::is_integral, T_n, T_N>* = nullptr,
          stan::require_eigen_col_vector_t<T_theta>* = nullptr>
inline stan::return_type_t<T_theta> nico_binomial_lpmf(const T_n& n, const T_N& N, const T_theta& theta,
                                                       std::ostream* /*pstream__*/) {

    namespace binomial = nico::math::binomial;
    using stan::math::var;
    stan::math::check_consistent_sizes(binomial::function_name, "Successes variable", n,
                                       "Population size parameter", N, "Probability parameter", theta);
    const int M = static_cast<int>(n.size());
    if (M == 0) return 0.0;
    const auto& theta_ref = stan::math::to_ref(theta);
    const Eigen::Matrix<double, -1, 1> theta_values = stan::math::value_of(theta_ref);
    if constexpr (stan::is_constant_all<T_theta>::value) {
        // (nothing depends on a parameter: with propto, Stan checks the arguments, then returns 0)
        std::vector<double> log_theta(M);
        std::vector<double> log1m_theta(M);
        double sum = 0.0;
        if (!binomial::sum_of_log_probabilities<true, false>(n.data(), N.data(), theta_values.data(), M,
                                                             log_theta.data(), log1m_theta.data(), nullptr,
                                                             sum)) {

            binomial::run_the_checks_of_Stan(n, N, theta_values);

        }
        return propto ? 0.0 : sum;
    } else {
        auto& arena = stan::math::ChainableStack::instance_->memalloc_;
        double* log_theta = arena.alloc_array<double>(M);
        double* log1m_theta = arena.alloc_array<double>(M);
        double* partials = arena.alloc_array<double>(M);
        double sum = 0.0;
        if (!binomial::sum_of_log_probabilities<!propto, true>(n.data(), N.data(), theta_values.data(), M,
                                                               log_theta, log1m_theta, partials, sum)) {

            binomial::run_the_checks_of_Stan(n, N, theta_values);

        }
        stan::arena_t<Eigen::Matrix<var, Eigen::Dynamic, 1>> theta_vars = theta_ref;
        return stan::math::make_callback_var(sum, [theta_vars, partials, M](auto& vi) mutable {
            const double adjoint = vi.adj();
            for (int i = 0; i < M; ++i) theta_vars.coeffRef(i).adj() += adjoint * partials[i];
        });
    }

}

}  // namespace nico::math

using nico::math::nico_binomial_lpmf;























