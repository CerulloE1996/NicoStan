#pragma once
//
// -| --------- NicoSIMD fused Poisson-log GLM log probability, Stan external function
//
//     nico_poisson_log_glm_lpmf(y | x, alpha, beta) = Stan's poisson_log_glm_lpmf(y | x, alpha, beta)
//                                                   = sum_i (y_i eta_i - exp(eta_i) - lgamma(y_i + 1))
//
// with eta = x * beta + alpha (x: N x K matrix; alpha: real, or vector of length N; beta: vector of length K).
// ONE pass over the rows, in chunks of rows_per_chunk rows: eta of the chunk (Eigen's matrix-vector product),
// exp(eta) once with NicoSIMD's exp kernel (Stan computes it twice), the chunk's sum and partials
// d_i = y_i - exp(eta_i), then the chunk's x^T d (Eigen), so that the chunk's rows of x are read again from the
// cache (Stan's version reads x twice from memory, for x * beta and for x^T d). lgamma(y_i + 1) with NicoSIMD's
// lgamma kernel, only when propto is false (_lpmf; _lupmf leaves it out, as Stan does). Partials: beta: x^T d;
// alpha: sum_i d_i (real) or d (vector); x (when it is a parameter): d_i beta_k, formed in the reverse pass. ONE
// autodiff node, whose operands are the parameters among x, alpha and beta.
//
// Checks as in Stan's poisson_log_glm_lpmf, with Stan's check functions (and so the same exceptions): consistent
// sizes, y nonnegative; when sum_i d_i is not finite: beta, alpha and eta finite. With every argument of double
// and propto = true the result is 0, as in Stan.
//
// Declared in the Stan functions block (no body; stanc --allow-undefined) as
//
//     real nico_poisson_log_glm_lpmf(array[] int y, matrix x, real alpha, vector beta);
//     real nico_poisson_log_glm_lpmf(array[] int y, matrix x, vector alpha, vector beta);
//
// e.g.  y ~ nico_poisson_log_glm(x, alpha, beta);  in place of  y ~ poisson_log_glm(x, alpha, beta);
// (Stan's other forms, int y or row_vector x, are not provided: a call with them keeps Stan's function.)
//
// -----------------------------------------------------------------------------------------------------------

#include "stan_external_functions.hpp"
#include "stan_external_lgamma.hpp"

#include <stan/math/rev.hpp>
#include <stan/math/prim.hpp>

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <ostream>
#include <type_traits>
#include <vector>

namespace nico::math {
namespace fused_poisson_log_glm {

// ---- the rows of one chunk: the chunk's rows of x (512 rows x K columns of double) stay in L2 between x * beta
//      and x^T d
static constexpr int rows_per_chunk = 512;

using x_values_t = Eigen::Ref<const Eigen::Matrix<double, -1, -1>, 0, Eigen::OuterStride<>>;

// ---- the inputs and outputs of one pass (where the partials go: nullptr when that partial is not needed)
struct poisson_glm_pass {
    const int* y = nullptr;
    int N = 0;
    int K = 0;
    const double* beta = nullptr;               // K values
    const double* alpha_vector = nullptr;       // N values, or nullptr when alpha is real
    double alpha_scalar = 0.0;
    double* partials_of_eta = nullptr;          // N values of y_i - exp(eta_i)
    double* partials_of_beta = nullptr;         // K values
    double* partial_of_alpha_scalar = nullptr;  // 1 value: sum_i (y_i - exp(eta_i))
    double sum_of_partials_of_eta = 0.0;
};

// ---- the one pass: returns sum_i (y_i eta_i - exp(eta_i)), minus sum_i lgamma(y_i + 1) when propto is false,
//      and fills the partials asked for
template <bool propto, bool compute_partials>
inline double sum_of_poisson_glm_log_probabilities(poisson_glm_pass& pass,
                                                   const x_values_t& x) {

    alignas(64) double eta_of_chunk[rows_per_chunk];
    alignas(64) double exp_eta_of_chunk[rows_per_chunk];
    alignas(64) double partials_of_eta_of_chunk[rows_per_chunk];
    const Eigen::Map<const Eigen::Matrix<double, -1, 1>> beta(pass.beta, pass.K);
    double sum = 0.0;
    double sum_of_partials = 0.0;
    if constexpr (compute_partials) {
        if (pass.partials_of_beta != nullptr) {
            for (int k = 0; k < pass.K; ++k) pass.partials_of_beta[k] = 0.0;
        }
    }
    for (int chunk_first = 0; chunk_first < pass.N; chunk_first += rows_per_chunk) {
        const int chunk_rows = std::min(rows_per_chunk, pass.N - chunk_first);
        const int* y_of_chunk = pass.y + chunk_first;
        // ---- eta = x * beta + alpha (Eigen), then exp(eta) (NicoSIMD):
        Eigen::Map<Eigen::Matrix<double, -1, 1>> eta(eta_of_chunk, chunk_rows);
        eta.noalias() = x.middleRows(chunk_first, chunk_rows) * beta;
        if (pass.alpha_vector != nullptr) {
            eta += Eigen::Map<const Eigen::Matrix<double, -1, 1>>(pass.alpha_vector + chunk_first, chunk_rows);
        } else {
            eta.array() += pass.alpha_scalar;
        }
        std::copy(eta_of_chunk, eta_of_chunk + chunk_rows, exp_eta_of_chunk);
        nico::math::k_exp(exp_eta_of_chunk, chunk_rows);
        // ---- the chunk's sum and partials d_i = y_i - exp(eta_i):
        for (int i = 0; i < chunk_rows; ++i) {
            const double y_i = static_cast<double>(y_of_chunk[i]);
            sum += y_i * eta_of_chunk[i] - exp_eta_of_chunk[i];
            const double d_i = y_i - exp_eta_of_chunk[i];
            partials_of_eta_of_chunk[i] = d_i;
            sum_of_partials += d_i;
        }
        // ---- minus lgamma(y_i + 1) (NicoSIMD), only for _lpmf:
        if constexpr (!propto) {
            for (int i = 0; i < chunk_rows; ++i) exp_eta_of_chunk[i] = static_cast<double>(y_of_chunk[i]) + 1.0;
            nico::math::k_lgamma(exp_eta_of_chunk, chunk_rows);
            for (int i = 0; i < chunk_rows; ++i) sum -= exp_eta_of_chunk[i];
        }
        // ---- the chunk's partials: d kept where asked for, and x^T d (Eigen) added to the partials of beta:
        if constexpr (compute_partials) {
            if (pass.partials_of_eta != nullptr) {
                std::copy(partials_of_eta_of_chunk, partials_of_eta_of_chunk + chunk_rows,
                          pass.partials_of_eta + chunk_first);
            }
            if (pass.partials_of_beta != nullptr) {
                Eigen::Map<Eigen::Matrix<double, -1, 1>>(pass.partials_of_beta, pass.K).noalias()
                    += x.middleRows(chunk_first, chunk_rows).transpose()
                       * Eigen::Map<const Eigen::Matrix<double, -1, 1>>(partials_of_eta_of_chunk, chunk_rows);
            }
        }
    }
    if constexpr (compute_partials) {
        if (pass.partial_of_alpha_scalar != nullptr) *pass.partial_of_alpha_scalar = sum_of_partials;
    }
    pass.sum_of_partials_of_eta = sum_of_partials;
    return sum;

}




// ---- the autodiff node: the operands among beta and alpha with their partials, and, when x is a parameter, x's
//      varis (column-major), whose partials d_i beta_k are formed in the reverse pass
class vari_fused_poisson_log_glm_lpmf final : public stan::math::vari {
 public:
    stan::math::vari** operands_;
    double* partials_;
    int n_operands_;
    stan::math::vari** x_operands_;
    const double* partials_of_eta_;
    const double* beta_values_;
    int N_;
    int K_;

    vari_fused_poisson_log_glm_lpmf(const double value, stan::math::vari** operands, double* partials,
                                    const int n_operands, stan::math::vari** x_operands,
                                    const double* partials_of_eta, const double* beta_values, const int N,
                                    const int K)
        : stan::math::vari(value), operands_(operands), partials_(partials), n_operands_(n_operands),
          x_operands_(x_operands), partials_of_eta_(partials_of_eta), beta_values_(beta_values), N_(N), K_(K) {}

    void chain() final {
        const double adjoint = adj_;
        for (int j = 0; j < n_operands_; ++j) operands_[j]->adj_ += adjoint * partials_[j];
        if (x_operands_ != nullptr) {
            for (int k = 0; k < K_; ++k) {
                const double adjoint_times_beta_k = adjoint * beta_values_[k];
                stan::math::vari** x_operands_of_column = x_operands_ + static_cast<std::ptrdiff_t>(k) * N_;
                for (int i = 0; i < N_; ++i) {
                    x_operands_of_column[i]->adj_ += adjoint_times_beta_k * partials_of_eta_[i];
                }
            }
        }
    }
};




// ---- the checks, the values, the pass and the node
template <bool propto, typename T_y, typename T_x, typename T_alpha, typename T_beta>
inline stan::return_type_t<T_x, T_alpha, T_beta> fused_poisson_log_glm_lpmf(const T_y& y, const T_x& x,
                                                                            const T_alpha& alpha,
                                                                            const T_beta& beta) {

    static_assert(std::is_same<stan::value_type_t<T_y>, int>::value, "y must be an array of int");
    constexpr bool x_is_var        = stan::is_var<stan::value_type_t<T_x>>::value;
    constexpr bool alpha_is_var    = stan::is_var<stan::scalar_type_t<T_alpha>>::value;
    constexpr bool beta_is_var     = stan::is_var<stan::value_type_t<T_beta>>::value;
    constexpr bool alpha_is_vector = stan::is_eigen<T_alpha>::value;
    constexpr bool some_parameter  = x_is_var || alpha_is_var || beta_is_var;
    static constexpr const char* function_name = "nico_poisson_log_glm_lpmf";
    const int N = static_cast<int>(x.rows());
    const int K = static_cast<int>(x.cols());
    stan::math::check_consistent_size(function_name, "Vector of dependent variables", y, N);
    stan::math::check_consistent_size(function_name, "Weight vector", beta, K);
    stan::math::check_consistent_size(function_name, "Vector of intercepts", alpha, N);
    stan::math::check_nonnegative(function_name, "Vector of dependent variables", y);
    if (y.empty()) return 0.0;
    if constexpr (!some_parameter && propto) return 0.0;
    // ---- the values of x (in place when its columns are contiguous), beta and alpha:
    const x_values_t x_values(stan::math::value_of(x));
    const Eigen::Matrix<double, -1, 1> beta_values = stan::math::value_of(beta);
    Eigen::Matrix<double, -1, 1> alpha_vector_values;
    poisson_glm_pass pass;
    pass.y = y.data();
    pass.N = N;
    pass.K = K;
    pass.beta = beta_values.data();
    if constexpr (alpha_is_vector) {
        alpha_vector_values = stan::math::value_of(alpha);
        pass.alpha_vector = alpha_vector_values.data();
    } else {
        pass.alpha_scalar = stan::math::value_of(alpha);
    }
    // ---- the checks of Stan's poisson_log_glm_lpmf after the pass (before any autodiff node exists):
    auto fn_checks_after_the_pass = [&]() {

        if (!std::isfinite(pass.sum_of_partials_of_eta)) {
            stan::math::check_finite(function_name, "Weight vector", beta);
            stan::math::check_finite(function_name, "Intercept", alpha);
            Eigen::Matrix<double, -1, 1> eta = x_values * beta_values;
            for (int i = 0; i < N; ++i) eta[i] += alpha_is_vector ? pass.alpha_vector[i] : pass.alpha_scalar;
            stan::math::check_finite(function_name, "Matrix of independent variables", eta);
        }

    };
    if constexpr (!some_parameter) {
        const double sum = sum_of_poisson_glm_log_probabilities<propto, false>(pass, x_values);
        fn_checks_after_the_pass();
        return sum;
    } else {
        auto& arena = stan::math::ChainableStack::instance_->memalloc_;
        // ---- the partials of the operands among beta (K) and alpha (1 or N), in this order; y - exp(eta) is
        //      kept for a parameter vector alpha (it is alpha's partials) or a parameter x
        const int n_beta_operands  = beta_is_var ? K : 0;
        const int n_alpha_operands = alpha_is_var ? (alpha_is_vector ? N : 1) : 0;
        const int n_operands = n_beta_operands + n_alpha_operands;
        double* partials = arena.alloc_array<double>(n_operands);
        if (beta_is_var) pass.partials_of_beta = partials;
        if (alpha_is_var && !alpha_is_vector) pass.partial_of_alpha_scalar = partials + n_beta_operands;
        if (alpha_is_var && alpha_is_vector) {
            pass.partials_of_eta = partials + n_beta_operands;
        } else if (x_is_var) {
            pass.partials_of_eta = arena.alloc_array<double>(N);
        }
        const double sum = sum_of_poisson_glm_log_probabilities<propto, true>(pass, x_values);
        fn_checks_after_the_pass();
        stan::math::vari** operands = arena.alloc_array<stan::math::vari*>(n_operands);
        int operand = 0;
        if constexpr (beta_is_var) {
            for (int k = 0; k < K; ++k) operands[operand++] = beta.coeff(k).vi_;
        }
        if constexpr (alpha_is_var) {
            if constexpr (alpha_is_vector) {
                for (int i = 0; i < N; ++i) operands[operand++] = alpha.coeff(i).vi_;
            } else {
                operands[operand++] = alpha.vi_;
            }
        }
        stan::math::vari** x_operands = nullptr;
        double* beta_values_on_arena = nullptr;
        if constexpr (x_is_var) {
            x_operands = arena.alloc_array<stan::math::vari*>(static_cast<std::size_t>(N) * K);
            for (int k = 0; k < K; ++k) {
                for (int i = 0; i < N; ++i) {
                    x_operands[static_cast<std::ptrdiff_t>(k) * N + i] = x.coeff(i, k).vi_;
                }
            }
            beta_values_on_arena = arena.alloc_array<double>(K);
            for (int k = 0; k < K; ++k) beta_values_on_arena[k] = beta_values[k];
        }
        return stan::math::var(new vari_fused_poisson_log_glm_lpmf(sum, operands, partials, n_operands,
                                                                   x_operands, pass.partials_of_eta,
                                                                   beta_values_on_arena, N, K));
    }

}

}  // namespace fused_poisson_log_glm
}  // namespace nico::math




// ---- the Stan-facing function (global scope; stanc calls it as
//      nico_poisson_log_glm_lpmf<propto__>(y, x, alpha, beta, pstream__), with propto__ = true for ~ and
//      _lupmf): y an array of int, x a matrix, alpha a real or a vector, beta a vector; x, alpha and beta of
//      double or of var
namespace nico::math {

template <bool propto, typename T_y, typename T_x, typename T_alpha, typename T_beta,
          stan::require_std_vector_vt<std::is_integral, T_y>* = nullptr,
          stan::require_eigen_matrix_dynamic_t<T_x>* = nullptr,
          stan::require_any_t<stan::is_stan_scalar<T_alpha>, stan::is_eigen_col_vector<T_alpha>>* = nullptr,
          stan::require_eigen_col_vector_t<T_beta>* = nullptr>
inline stan::return_type_t<T_x, T_alpha, T_beta> nico_poisson_log_glm_lpmf(const T_y& y, const T_x& x,
                                                                           const T_alpha& alpha,
                                                                           const T_beta& beta,
                                                                           std::ostream* /*pstream__*/) {

    return nico::math::fused_poisson_log_glm::fused_poisson_log_glm_lpmf<propto>(y, x, alpha, beta);

}

}  // namespace nico::math

using nico::math::nico_poisson_log_glm_lpmf;
