#pragma once
//
// -| --------- NicoSIMD fused Bernoulli GLM log probabilities with a tensor-product basis term (logit and probit
//              links), Stan external functions
//
//     nico_bernoulli_logit_glm_with_a_tensor_product_basis_lpmf(y | x, alpha, beta, basis_1, basis_2, weights)
//         = bernoulli_logit_lpmf(y | alpha + x * beta + rows_dot_product(basis_1 * weights, basis_2))
//     nico_bernoulli_probit_glm_with_a_tensor_product_basis_lpmf(...)   (the same, with the probit link)
//
// with x an N x K matrix, alpha a real or a vector of length N, beta a vector of length K, basis_1 an N x P
// matrix, basis_2 an N x Q matrix and weights a P x Q matrix: the linear predictor of person i is
//
//     f_i = alpha_i + x_i' beta + sum_{a, b} basis_1(i, a) weights(a, b) basis_2(i, b),
//
// i.e. the GLM whose design matrix has, besides x, the P Q columns basis_1(, a) .* basis_2(, b) (the tensor
// product of two bases: a Hilbert space approximate Gaussian process in two dimensions, a tensor-product spline,
// any separable basis of two variables), without those P Q columns ever being formed: each person's row is read
// as K + P + Q values instead of K + P Q (e.g. 33 instead of 228 with K = 3 and P = Q = 15), so that a model
// whose time goes in reading its design matrix reads about P Q / (P + Q) times less.
//
// ONE pass over the rows, in the chunks of rows of stan_external_fused_bernoulli_logit_and_probit_glm_lpmf.hpp
// (its helpers: f of the chunk from x, then the link's log probabilities and d_i = d log p_i / d f_i in SIMD
// blocks, then the chunk's x^T d), with, for each chunk, the tensor term added to f (V = basis_1 * weights of
// the chunk's rows, then f += sum_b V(, b) .* basis_2(, b)) and its partials added to those of the weights
// (d weights += basis_1^T (d .* basis_2) of the chunk's rows): Eigen's matrix products and column operations,
// vectorised with the build's AVX2 or AVX-512 flags, on the chunk's rows while they are in the cache. ONE
// autodiff node, whose operands are the parameters among beta, alpha and the weights.
//
// x, basis_1 and basis_2 are data (matrices of double). Checks as nico_bernoulli_logit_glm_lpmf (Stan's checks,
// with Stan's check functions), with the sizes of the bases and of the weights, and, when the sum is not finite
// or some f is NaN, the weights finite as well.
//
// Declared in the Stan functions block (no body; stanc --allow-undefined) as
//
//     real nico_bernoulli_logit_glm_with_a_tensor_product_basis_lpmf(array[] int y, matrix x, real alpha,
//                                                                    vector beta, matrix basis_1,
//                                                                    matrix basis_2, matrix weights);
//
// (alpha also a vector; the probit link with nico_bernoulli_probit_glm_with_a_tensor_product_basis_lpmf).
//
// Included by stan_external_functions_of_NicoSIMD.hpp, after stan_external_functions.hpp and its
// stan_external_fused_bernoulli_logit_and_probit_glm_lpmf.hpp, whose helpers it uses. Its own helpers are in the
// namespace nico::math::fused_bernoulli_glm_with_a_tensor_product_basis (the internal C++ of NicoSIMD keeps
// BayesMVP's namespace).
//
// -----------------------------------------------------------------------------------------------------------

#include "stan_external_functions.hpp"
#include "stan_external_fused_bernoulli_logit_and_probit_glm_lpmf.hpp"

#include <stan/math/rev.hpp>
#include <stan/math/prim.hpp>

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <ostream>
#include <type_traits>
#include <vector>

namespace nico::math {
namespace fused_bernoulli_glm_with_a_tensor_product_basis {

using stan::math::var;
using stan::math::vari;
namespace glm         = nico::math::fused_bernoulli_glm;
namespace fused_logit = nico::math::fused_bernoulli_logit;

using const_view_of_rows = Eigen::Map<const Eigen::Matrix<double, -1, -1>, 0, Eigen::OuterStride<>>;

// ---- every SIMD block variable declared aligned to its own width (ALIGN64 for AVX-512, ALIGN32 for AVX2)
#if BMVP_LANES == 8
#  define ALIGN_OF_A_SIMD_BLOCK ALIGN64
#else
#  define ALIGN_OF_A_SIMD_BLOCK ALIGN32
#endif




// ---- the tensor term: the two bases (column-major, each column contiguous), the weights (P x Q, column-major)
//      and where the partials of the weights go (nullptr when the weights are data)
struct tensor_term {
    const double* basis_1 = nullptr;              // basis_1(i, a) at basis_1[a * basis_1_column_stride + i]
    int basis_1_column_stride = 0;
    int P = 0;
    const double* basis_2 = nullptr;              // basis_2(i, b) at basis_2[b * basis_2_column_stride + i]
    int basis_2_column_stride = 0;
    int Q = 0;
    const double* weights = nullptr;              // weights(a, b) at weights[b * P + a]
    double* partials_of_weights = nullptr;        // P Q values, column-major
};

// ---- f of the chunk's rows += sum_b (basis_1 * weights)(, b) .* basis_2(, b); buffer: rows_per_chunk Q values
inline void add_tensor_term_to_f_of_chunk(const tensor_term& term, const int chunk_first, const int chunk_rows,
                                          double* f_of_chunk, double* buffer) {

    const const_view_of_rows basis_1_of_chunk(term.basis_1 + chunk_first, chunk_rows, term.P,
                                              Eigen::OuterStride<>(term.basis_1_column_stride));
    const const_view_of_rows basis_2_of_chunk(term.basis_2 + chunk_first, chunk_rows, term.Q,
                                              Eigen::OuterStride<>(term.basis_2_column_stride));
    const Eigen::Map<const Eigen::Matrix<double, -1, -1>> weights(term.weights, term.P, term.Q);
    Eigen::Map<Eigen::Matrix<double, -1, -1>> basis_1_times_weights(buffer, chunk_rows, term.Q);
    basis_1_times_weights.noalias() = basis_1_of_chunk * weights;
    Eigen::Map<Eigen::Matrix<double, -1, 1>> f(f_of_chunk, chunk_rows);
    for (int b = 0; b < term.Q; ++b) {
        f += basis_1_times_weights.col(b).cwiseProduct(basis_2_of_chunk.col(b));
    }

}

// ---- the partials of the weights += basis_1^T (d .* basis_2) of the chunk's rows; buffer: rows_per_chunk Q
//      values
inline void add_partials_of_weights_of_chunk(const tensor_term& term, const int chunk_first, const int chunk_rows,
                                             const double* partials_of_f_of_chunk, double* buffer) {

    const const_view_of_rows basis_1_of_chunk(term.basis_1 + chunk_first, chunk_rows, term.P,
                                              Eigen::OuterStride<>(term.basis_1_column_stride));
    const const_view_of_rows basis_2_of_chunk(term.basis_2 + chunk_first, chunk_rows, term.Q,
                                              Eigen::OuterStride<>(term.basis_2_column_stride));
    const Eigen::Map<const Eigen::Matrix<double, -1, 1>> d(partials_of_f_of_chunk, chunk_rows);
    Eigen::Map<Eigen::Matrix<double, -1, -1>> d_times_basis_2(buffer, chunk_rows, term.Q);
    for (int b = 0; b < term.Q; ++b) {
        d_times_basis_2.col(b) = basis_2_of_chunk.col(b).cwiseProduct(d);
    }
    Eigen::Map<Eigen::Matrix<double, -1, -1>> partials_of_weights(term.partials_of_weights, term.P, term.Q);
    partials_of_weights.noalias() += basis_1_of_chunk.transpose() * d_times_basis_2;

}

// ---- the one pass (as glm::sum_of_glm_log_probabilities, with the tensor term in each chunk): returns
//      sum_i log p_i, and fills the partials asked for
template <typename link, bool compute_partials>
inline double sum_of_glm_log_probabilities_with_a_tensor_product_basis(glm::glm_pass& pass,
                                                                       const tensor_term& term) {

    constexpr int rows_per_chunk = glm::rows_per_chunk;
    fused_logit::bad_y_bits_t bad_y = fused_logit::no_bad_y_bits();
    fused_logit::nan_lanes_t nan_lanes = fused_logit::no_nan_lanes();
    ALIGN_OF_A_SIMD_BLOCK simd_t sum = fused_logit::zero_block();
    ALIGN_OF_A_SIMD_BLOCK simd_t alpha_accumulator = fused_logit::zero_block();
    ALIGN_OF_A_SIMD_BLOCK simd_t d = fused_logit::zero_block();
    alignas(64) double partials_of_f_of_chunk[rows_per_chunk];
    alignas(64) double f_of_chunk[rows_per_chunk];
    // (the buffer of the chunk's basis_1 * weights, then of its d .* basis_2: kept by each thread between calls)
    thread_local std::vector<double> buffer;
    buffer.resize(static_cast<std::size_t>(rows_per_chunk) * term.Q);
    if constexpr (compute_partials) {
        if (pass.partials_of_beta != nullptr) {
            for (int k = 0; k < pass.K; ++k) pass.partials_of_beta[k] = 0.0;
        }
        if (term.partials_of_weights != nullptr) {
            for (int j = 0; j < term.P * term.Q; ++j) term.partials_of_weights[j] = 0.0;
        }
    }
    for (int chunk_first = 0; chunk_first < pass.N; chunk_first += rows_per_chunk) {
        const int chunk_end = std::min(chunk_first + rows_per_chunk, pass.N);
        const int chunk_rows = chunk_end - chunk_first;
        glm::compute_f_of_chunk_into_buffer(pass, chunk_first, chunk_rows, f_of_chunk);
        add_tensor_term_to_f_of_chunk(term, chunk_first, chunk_rows, f_of_chunk, buffer.data());
        int first = chunk_first;
        for (; first + BMVP_LANES <= chunk_end; first += BMVP_LANES) {
            ALIGN_OF_A_SIMD_BLOCK const simd_t f = simd_load(f_of_chunk + (first - chunk_first));
            ALIGN_OF_A_SIMD_BLOCK const simd_t s = fused_logit::sign_from_y(pass.y + first, bad_y);
            fused_logit::collect_nan_lanes(nan_lanes, f);
            sum = glm::add_blocks(sum, link::template log_probability_and_partial<compute_partials>(f, s, d));
            if constexpr (compute_partials) {
                glm::keep_partials_of_block<false>(pass, first, BMVP_LANES, d,
                                                   partials_of_f_of_chunk + (first - chunk_first),
                                                   alpha_accumulator);
            }
        }
        if (first < chunk_end) {
            // ---- the final partial block (of the last chunk): spare lanes with y = 0 and f = 0, left out of the
            //      sum and of the partials
            const int count = chunk_end - first;
            alignas(64) int y_buffer[BMVP_LANES] = {0};
            for (int lane = 0; lane < count; ++lane) y_buffer[lane] = pass.y[first + lane];
            ALIGN_OF_A_SIMD_BLOCK const simd_t f
                = glm::keep_first_lanes(simd_load(f_of_chunk + (first - chunk_first)), count);
            ALIGN_OF_A_SIMD_BLOCK const simd_t s = fused_logit::sign_from_y(y_buffer, bad_y);
            fused_logit::collect_nan_lanes(nan_lanes, f);
            ALIGN_OF_A_SIMD_BLOCK const simd_t log_probabilities =
                link::template log_probability_and_partial<compute_partials>(f, s, d);
            sum = glm::add_blocks(sum, glm::keep_first_lanes(log_probabilities, count));
            if constexpr (compute_partials) {
                glm::keep_partials_of_block<true>(pass, first, count, glm::keep_first_lanes(d, count),
                                                  partials_of_f_of_chunk + (first - chunk_first),
                                                  alpha_accumulator);
            }
        }
        if constexpr (compute_partials) {
            if (pass.partials_of_beta != nullptr) {
                glm::add_partials_of_beta_of_chunk(pass, chunk_first, chunk_rows, partials_of_f_of_chunk);
            }
            if (term.partials_of_weights != nullptr) {
                add_partials_of_weights_of_chunk(term, chunk_first, chunk_rows, partials_of_f_of_chunk,
                                                 buffer.data());
            }
        }
    }
    if constexpr (compute_partials) {
        if (pass.partial_of_alpha_scalar != nullptr) {
            *pass.partial_of_alpha_scalar = fused_logit::sum_of_lanes(alpha_accumulator);
        }
    }
    pass.some_y_not_binary = fused_logit::any_bad_y(bad_y);
    pass.some_f_nan = fused_logit::any_nan(nan_lanes);
    return fused_logit::sum_of_lanes(sum);

}




// ---- a data matrix's values, in place when its columns are contiguous, else copied into storage
template <typename T_matrix>
inline const double* values_of_data_matrix(const T_matrix& m, Eigen::Matrix<double, -1, -1>& storage,
                                           int& column_stride) {

    const double* data = glm::column_major_data_or_nullptr(m, column_stride);
    if (data == nullptr) {
        storage = m;
        data = storage.data();
        column_stride = static_cast<int>(m.rows());
    }
    return data;

}

// ---- both links: the checks, the values, the pass and the node
template <bool propto, typename link, typename T_y, typename T_x, typename T_alpha, typename T_beta,
          typename T_basis_1, typename T_basis_2, typename T_weights>
inline stan::return_type_t<T_alpha, T_beta, T_weights> fused_bernoulli_glm_with_a_tensor_product_basis_lpmf(
        const T_y& y, const T_x& x, const T_alpha& alpha, const T_beta& beta, const T_basis_1& basis_1,
        const T_basis_2& basis_2, const T_weights& weights, const char* function_name) {

    static_assert(std::is_same<stan::value_type_t<T_y>, int>::value, "y must be an array of int");
    static_assert(std::is_arithmetic<stan::value_type_t<T_x>>::value
                  && std::is_arithmetic<stan::value_type_t<T_basis_1>>::value
                  && std::is_arithmetic<stan::value_type_t<T_basis_2>>::value,
                  "x, basis_1 and basis_2 must be data");
    constexpr bool alpha_is_var     = stan::is_var<stan::scalar_type_t<T_alpha>>::value;
    constexpr bool beta_is_var      = stan::is_var<stan::value_type_t<T_beta>>::value;
    constexpr bool weights_are_var  = stan::is_var<stan::value_type_t<T_weights>>::value;
    constexpr bool alpha_is_vector  = stan::is_eigen<T_alpha>::value;
    constexpr bool some_parameter   = alpha_is_var || beta_is_var || weights_are_var;
    const int N = static_cast<int>(x.rows());
    const int K = static_cast<int>(x.cols());
    const int P = static_cast<int>(weights.rows());
    const int Q = static_cast<int>(weights.cols());
    stan::math::check_consistent_size(function_name, "Vector of dependent variables", y, N);
    stan::math::check_consistent_size(function_name, "Weight vector", beta, K);
    stan::math::check_consistent_size(function_name, "Vector of intercepts", alpha, N);
    stan::math::check_size_match(function_name, "Rows of basis_1", basis_1.rows(), "rows of x", N);
    stan::math::check_size_match(function_name, "Rows of basis_2", basis_2.rows(), "rows of x", N);
    stan::math::check_size_match(function_name, "Columns of basis_1", basis_1.cols(), "rows of weights", P);
    stan::math::check_size_match(function_name, "Columns of basis_2", basis_2.cols(), "columns of weights", Q);
    if (y.empty()) return 0.0;
    if constexpr (!some_parameter && propto) {
        stan::math::check_bounded(function_name, "Vector of dependent variables", y, 0, 1);
        return 0.0;
    }
    // ---- the values of x, the bases (in place when their columns are contiguous), beta, alpha and the weights:
    Eigen::Matrix<double, -1, -1> x_storage;
    Eigen::Matrix<double, -1, -1> basis_1_storage;
    Eigen::Matrix<double, -1, -1> basis_2_storage;
    glm::glm_pass pass;
    pass.y = y.data();
    pass.N = N;
    pass.K = K;
    pass.x = values_of_data_matrix(x, x_storage, pass.x_column_stride);
    const Eigen::Matrix<double, -1, 1> beta_values = stan::math::value_of(beta);
    pass.beta = beta_values.data();
    Eigen::Matrix<double, -1, 1> alpha_vector_values;
    if constexpr (alpha_is_vector) {
        alpha_vector_values = stan::math::value_of(alpha);
        pass.alpha_vector = alpha_vector_values.data();
    } else {
        pass.alpha_scalar = stan::math::value_of(alpha);
    }
    const Eigen::Matrix<double, -1, -1> weights_values = stan::math::value_of(weights);
    tensor_term term;
    term.basis_1 = values_of_data_matrix(basis_1, basis_1_storage, term.basis_1_column_stride);
    term.basis_2 = values_of_data_matrix(basis_2, basis_2_storage, term.basis_2_column_stride);
    term.P = P;
    term.Q = Q;
    term.weights = weights_values.data();
    // ---- the checks of Stan's bernoulli_logit_glm_lpmf after the pass (before any autodiff node exists):
    auto fn_checks_after_the_pass = [&](const double sum) {

        if (pass.some_y_not_binary) {
            stan::math::check_bounded(function_name, "Vector of dependent variables", y, 0, 1);
        }
        if (!std::isfinite(sum) || pass.some_f_nan) {
            stan::math::check_finite(function_name, "Weight vector", beta);
            stan::math::check_finite(function_name, "Intercept", alpha);
            stan::math::check_finite(function_name, "Weights of the tensor-product basis", weights_values);
            stan::math::check_finite(function_name, "Matrix of independent variables",
                                     Eigen::Map<const Eigen::Matrix<double, -1, -1>, 0, Eigen::OuterStride<>>(
                                             pass.x, N, K, Eigen::OuterStride<>(pass.x_column_stride)));
        }

    };
    if constexpr (!some_parameter) {
        const double sum = sum_of_glm_log_probabilities_with_a_tensor_product_basis<link, false>(pass, term);
        fn_checks_after_the_pass(sum);
        return sum;
    } else {
        auto& arena = stan::math::ChainableStack::instance_->memalloc_;
        // ---- the partials of the operands among beta (K), alpha (1 or N) and the weights (P Q), in this order
        const int n_beta_operands    = beta_is_var ? K : 0;
        const int n_alpha_operands   = alpha_is_var ? (alpha_is_vector ? N : 1) : 0;
        const int n_weights_operands = weights_are_var ? P * Q : 0;
        const int n_operands = n_beta_operands + n_alpha_operands + n_weights_operands;
        double* partials = arena.alloc_array<double>(n_operands);
        if (beta_is_var) pass.partials_of_beta = partials;
        if (alpha_is_var && !alpha_is_vector) pass.partial_of_alpha_scalar = partials + n_beta_operands;
        if (alpha_is_var && alpha_is_vector) pass.partials_of_f = partials + n_beta_operands;
        if (weights_are_var) term.partials_of_weights = partials + n_beta_operands + n_alpha_operands;
        const double sum = sum_of_glm_log_probabilities_with_a_tensor_product_basis<link, true>(pass, term);
        fn_checks_after_the_pass(sum);
        vari** operands = arena.alloc_array<vari*>(n_operands);
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
        if constexpr (weights_are_var) {
            // (column-major, as the partials of the weights)
            for (int b = 0; b < Q; ++b) {
                for (int a = 0; a < P; ++a) operands[operand++] = weights.coeff(a, b).vi_;
            }
        }
        return stan::math::var(new glm::vari_fused_bernoulli_glm_lpmf(sum, operands, partials, n_operands,
                                                                      nullptr, nullptr, nullptr, N, K));
    }

}

}  // namespace fused_bernoulli_glm_with_a_tensor_product_basis
}  // namespace nico::math




// ---- the Stan-facing functions (global scope; stanc calls them as
//      nico_bernoulli_logit_glm_with_a_tensor_product_basis_lpmf<propto__>(y, x, alpha, beta, basis_1, basis_2,
//      weights, pstream__)): y an array of int; x, basis_1 and basis_2 data matrices; alpha a real or a vector;
//      beta a vector; weights a matrix; alpha, beta and the weights of double or of var
namespace nico::math {

template <bool propto, typename T_y, typename T_x, typename T_alpha, typename T_beta, typename T_basis_1,
          typename T_basis_2, typename T_weights,
          stan::require_std_vector_vt<std::is_integral, T_y>* = nullptr,
          stan::require_eigen_matrix_dynamic_t<T_x>* = nullptr,
          stan::require_any_t<stan::is_stan_scalar<T_alpha>, stan::is_eigen_col_vector<T_alpha>>* = nullptr,
          stan::require_eigen_col_vector_t<T_beta>* = nullptr,
          stan::require_eigen_matrix_dynamic_t<T_basis_1>* = nullptr,
          stan::require_eigen_matrix_dynamic_t<T_basis_2>* = nullptr,
          stan::require_eigen_matrix_dynamic_t<T_weights>* = nullptr>
inline stan::return_type_t<T_alpha, T_beta, T_weights> nico_bernoulli_logit_glm_with_a_tensor_product_basis_lpmf(
        const T_y& y, const T_x& x, const T_alpha& alpha, const T_beta& beta, const T_basis_1& basis_1,
        const T_basis_2& basis_2, const T_weights& weights, std::ostream* /*pstream__*/) {

    namespace fused = nico::math::fused_bernoulli_glm_with_a_tensor_product_basis;
    namespace glm   = nico::math::fused_bernoulli_glm;
    return fused::fused_bernoulli_glm_with_a_tensor_product_basis_lpmf<propto, glm::logit_link>(
               y, x, alpha, beta, basis_1, basis_2, weights,
               "nico_bernoulli_logit_glm_with_a_tensor_product_basis_lpmf");

}

}  // namespace nico::math

using nico::math::nico_bernoulli_logit_glm_with_a_tensor_product_basis_lpmf;

#if !defined(NICO_SIMD_SCALAR_KERNELS)
namespace nico::math {

template <bool propto, typename T_y, typename T_x, typename T_alpha, typename T_beta, typename T_basis_1,
          typename T_basis_2, typename T_weights,
          stan::require_std_vector_vt<std::is_integral, T_y>* = nullptr,
          stan::require_eigen_matrix_dynamic_t<T_x>* = nullptr,
          stan::require_any_t<stan::is_stan_scalar<T_alpha>, stan::is_eigen_col_vector<T_alpha>>* = nullptr,
          stan::require_eigen_col_vector_t<T_beta>* = nullptr,
          stan::require_eigen_matrix_dynamic_t<T_basis_1>* = nullptr,
          stan::require_eigen_matrix_dynamic_t<T_basis_2>* = nullptr,
          stan::require_eigen_matrix_dynamic_t<T_weights>* = nullptr>
inline stan::return_type_t<T_alpha, T_beta, T_weights> nico_bernoulli_probit_glm_with_a_tensor_product_basis_lpmf(
        const T_y& y, const T_x& x, const T_alpha& alpha, const T_beta& beta, const T_basis_1& basis_1,
        const T_basis_2& basis_2, const T_weights& weights, std::ostream* /*pstream__*/) {

    namespace fused = nico::math::fused_bernoulli_glm_with_a_tensor_product_basis;
    namespace glm   = nico::math::fused_bernoulli_glm;
    return fused::fused_bernoulli_glm_with_a_tensor_product_basis_lpmf<propto, glm::probit_link>(
               y, x, alpha, beta, basis_1, basis_2, weights,
               "nico_bernoulli_probit_glm_with_a_tensor_product_basis_lpmf");

}

}  // namespace nico::math

using nico::math::nico_bernoulli_probit_glm_with_a_tensor_product_basis_lpmf;
#endif






















