#pragma once
//
// -| --------- BayesMVP fused GLM log densities: Student-t (identity link), Weibull (log link on the scale) and
//              Bernoulli-logit with each group's own coefficients, Stan external functions ---------------------
//
//     nico_student_t_id_glm_lpdf(y | x, alpha, beta, nu, sigma)
//                       = student_t_lpdf(y | nu, alpha + x * beta, sigma)
//     nico_weibull_log_glm_lpdf(y | x, alpha, beta, shape)
//                       = weibull_lpdf(y | shape, exp(alpha + x * beta))
//     nico_weibull_log_glm_lccdf(y | x, alpha, beta, shape)
//                       = weibull_lccdf(y | shape, exp(alpha + x * beta))
//     nico_bernoulli_logit_glm_with_coefficients_by_group_lpmf(y | x, group, alpha, B)
//                       = bernoulli_logit_lpmf(y | f),   f_i = alpha(group_i) + sum_k x(i, k) B(group_i, k)
//     nico_bernoulli_logit_glm_with_coefficients_by_group_lpmf(y | x, group, B)     (the same without alpha)
//
// with x an N x K matrix of data; alpha a real or a vector of length N; beta a vector of length K; nu, sigma and
// shape reals; group an array of N ints in 1, ..., G; B a G x K matrix (row g: the coefficients of group g, e.g.
// x = [1, z] gives each group its own intercept and slope). The same arguments, in the same order, as Stan's
// GLMs (normal_id_glm_lpdf, poisson_log_glm_lpmf, ...), with the distribution's own parameters last.
//
// ONE pass over the rows, as in nico_bernoulli_logit_glm_lpmf (stan_external_fused_bernoulli_logit_and_probit_
// glm_lpmf.hpp, whose code for the linear predictor and for x^T d is used here): in chunks of rows_per_chunk rows,
// f = alpha + x * beta of the chunk, column by column (each column's rows of the chunk read as one contiguous
// run), then, for each block of BMVP_LANES rows, the log density and d_i = d log p_i / d f_i with the BayesMVP
// SIMD kernels, then the chunk's x^T d, added to the partials of beta (x is read from memory once, then from the
// cache). ONE autodiff node, whose operands are the parameters among beta, alpha and the distribution's own
// parameters (Stan's version: one node per element of x * beta and of alpha + x * beta, then the density's node).
//
// ---- Student-t, with z_i = (y_i - f_i) / sigma, r_i = z_i^2 / nu and w_i = 1 / (1 + r_i) (the partials of
//      nico_student_t_lpdf, stan_external_fused_student_t_lpdf.hpp, whose log1p kernel is used):
//
//     log p_i     = lgamma((nu + 1) / 2) - lgamma(nu / 2) - 0.5 log(nu) - log(sqrt(pi)) - log(sigma)
//                   - ((nu + 1) / 2) log1p(r_i)
//     d / d f_i   = (nu + 1) z_i w_i / (nu sigma)
//     d / d sigma = sum_i (nu - (nu + 1) w_i) / sigma
//     d / d nu    = sum_i 0.5 (digamma((nu + 1) / 2) - digamma(nu / 2) + 1 - log1p(r_i) - (nu + 1) w_i / nu)
//
// ---- Weibull, with L_i = log(y_i), u_i = L_i - f_i and w_i = exp(shape u_i) = (y_i / exp(f_i))^shape (the log
//      and exp kernels of nico_weibull_lpdf, stan_external_fused_weibull_lpdf.hpp):
//
//     log p_i         = log(shape) + (shape - 1) L_i - shape f_i - w_i
//     d / d f_i       = shape (w_i - 1)
//     d / d shape     = sum_i (1 / shape + u_i (1 - w_i))
//     log S_i         = -w_i,   d / d f_i = shape w_i,   d / d shape = -sum_i u_i w_i
//
// ---- Bernoulli-logit by group (alpha a vector of length G: each group's intercept), in the way of
//      nico_bernoulli_logit_grouped_regression_lpmf, for any K: f row by row from the values of alpha
//      and B (read once), BayesMVP's exp and log1p kernels over all the rows, and d / d alpha(g) = sum_{i: group_i
//      = g} d_i, d / d B(g, k) = sum_{i: group_i = g} x(i, k) d_i, each run of rows of the same group (as in data
//      sorted by group) summed in registers before it is added.
//
// propto: the terms of Stan's student_t_lpdf, weibull_lpdf and bernoulli_logit_lpmf, i.e. propto = true drops
// log(sqrt(pi)), the terms in nu alone when nu is data, log(sigma) when sigma is data, log(shape) when shape is
// data, and (shape - 1) L_i when shape is data (y is data); with no parameter argument and propto = true the
// result is 0, as in Stan. The lccdf has no propto.
//
// Checks as in Stan, with Stan's own check functions (and so the same exceptions): consistent sizes; nu, sigma and
// shape positive and finite; for the Bernoulli-logit, y in {0, 1} and every group in 1, ..., G (checked before
// it is used as an index). When the sum is not finite: y, beta, alpha (and B) finite; for the Weibull lpdf, as
// in weibull_lpdf, a negative y gives a log density of negative infinity; the Weibull lccdf checks y >= 0.
//
// x must be data (a matrix of double, a Map of one, or a block of its rows; as in every benchmark model), and y
// is data (array[] real or vector for the Student-t and Weibull; array[] int for the Bernoulli-logit).
//
// Declared in the Stan functions block (no body; stanc --allow-undefined), e.g.
//
//     real nico_student_t_id_glm_lpdf(array[] real y, matrix x, real alpha, vector beta, real nu, real sigma);
//     real nico_weibull_log_glm_lpdf(array[] real y, matrix x, real alpha, vector beta, real shape);
//     real nico_weibull_log_glm_lccdf(vector y, matrix x, real alpha, vector beta, real shape);
//     real nico_bernoulli_logit_glm_with_coefficients_by_group_lpmf(array[] int y, matrix x, array[] int group,
//                                                                   vector alpha, matrix B);
//
// Included at the end of stan_external_functions.hpp, after stan_external_fused_bernoulli_logit_and_probit_glm_
// lpmf.hpp; its helpers are in their own namespace, nico::math::fused_glms_by_family.
//
// -----------------------------------------------------------------------------------------------------------

#include "stan_external_functions.hpp"
#include "stan_external_fused_bernoulli_logit_lpmf.hpp"
#include "stan_external_fused_student_t_lpdf.hpp"
#include "stan_external_fused_weibull_lpdf.hpp"
#include "stan_external_fused_bernoulli_logit_and_probit_glm_lpmf.hpp"
// (the kernels of nico_bernoulli_logit_grouped_regression_lpmf, used by the GLM by group)
#include "stan_external_fused_bernoulli_logit_grouped_regression_lpmf_of_13_Oct.hpp"

#include <stan/math/rev.hpp>
#include <stan/math/prim.hpp>

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include "nico_simd_intrinsics.hpp"   //// <immintrin.h> on x86-64; SIMDe (NEON) on ARM
#include <limits>
#include <ostream>
#include <type_traits>
#include <vector>

namespace nico::math {
namespace fused_glms_by_family {

using stan::math::var;
using stan::math::vari;
namespace glm         = nico::math::fused_bernoulli_glm;
namespace fused_logit = nico::math::fused_bernoulli_logit;
namespace fused_t     = nico::math::fused_student_t;
namespace fused_w     = nico::math::fused_weibull;




// ---- the gather of one block of BMVP_LANES values, base[indices[0]], ..., base[indices[BMVP_LANES - 1]]
#if BMVP_LANES == 8
ALWAYS_INLINE simd_t gather_of_block(const double* base, const int* indices) {

    ALIGN32  __m256i const index_block = _mm256_load_si256(reinterpret_cast<const __m256i*>(indices));
    return _mm512_i32gather_pd(index_block, base, 8);

}
#else
ALWAYS_INLINE simd_t gather_of_block(const double* base, const int* indices) {

    const __m128i index_block = _mm_load_si128(reinterpret_cast<const __m128i*>(indices));
    return _mm256_i32gather_pd(base, index_block, 8);

}
#endif

// ---- y of one block of rows: BMVP_LANES values, or (count < BMVP_LANES: a final partial block) the first count
//      values and 1.0 in the spare lanes (a value with a finite log density, left out of every sum)
ALWAYS_INLINE simd_t y_of_block(const double* y_of_block_start, const int count) {

    if (count == BMVP_LANES) return simd_loadu(y_of_block_start);
    alignas(64) double y_buffer[BMVP_LANES];
    for (int lane = 0; lane < BMVP_LANES; ++lane) y_buffer[lane] = (lane < count) ? y_of_block_start[lane] : 1.0;
    return simd_load(y_buffer);

}

// ---- the address of y's values (array[] real, or a vector of double with contiguous elements), else a copy
template <typename T_y>
inline const double* data_of_y(const T_y& y, std::vector<double>& copy_of_y) {

    if constexpr (stan::is_std_vector<T_y>::value) {
        return y.data();
    } else {
        using plain_t = std::decay_t<T_y>;
        if constexpr (static_cast<bool>(Eigen::internal::traits<plain_t>::Flags & Eigen::DirectAccessBit)) {
            if (y.innerStride() == 1) return y.data();
        }
        copy_of_y.resize(y.size());
        for (Eigen::Index i = 0; i < y.size(); ++i) copy_of_y[i] = y.coeff(i);
        return copy_of_y.data();
    }

}




// ---- the linear predictor alpha + x * beta: the values of x (in place when its columns are contiguous), alpha and
//      beta, in the GLM pass of stan_external_fused_bernoulli_logit_and_probit_glm_lpmf.hpp (built in place: the
//      pass points into this object's own vectors, so it is never copied)
struct values_of_linear_predictor {
    Eigen::Matrix<double, -1, -1> x_values;
    Eigen::Matrix<double, -1, 1> beta_values;
    Eigen::Matrix<double, -1, 1> alpha_vector_values;
    glm::glm_pass pass;

    template <typename T_x, typename T_alpha, typename T_beta>
    values_of_linear_predictor(const T_x& x, const T_alpha& alpha, const T_beta& beta) {

        const int N = static_cast<int>(x.rows());
        const int K = static_cast<int>(x.cols());
        int x_column_stride = 0;
        const double* x_data = glm::column_major_data_or_nullptr(x, x_column_stride);
        if (x_data == nullptr) {
            x_values = stan::math::value_of(x);
            x_data = x_values.data();
            x_column_stride = N;
        }
        beta_values = stan::math::value_of(beta);
        pass.N = N;
        pass.K = K;
        pass.x = x_data;
        pass.x_column_stride = x_column_stride;
        pass.beta = beta_values.data();
        if constexpr (stan::is_eigen<T_alpha>::value) {
            alpha_vector_values = stan::math::value_of(alpha);
            pass.alpha_vector = alpha_vector_values.data();
        } else {
            pass.alpha_scalar = stan::math::value_of(alpha);
        }

    }
    values_of_linear_predictor(const values_of_linear_predictor&) = delete;
    values_of_linear_predictor& operator=(const values_of_linear_predictor&) = delete;

    // ---- f_i = alpha_i + x(i, ) beta, one row at a time (for the checks after the pass only)
    double f_of_row(const int i) const {

        double f_i = (pass.alpha_vector != nullptr) ? pass.alpha_vector[i] : pass.alpha_scalar;
        for (int k = 0; k < pass.K; ++k) {
            f_i += pass.x[static_cast<std::ptrdiff_t>(k) * pass.x_column_stride + i] * pass.beta[k];
        }
        return f_i;

    }
};

// ---- the partials of f of one block (zero in the spare lanes of a final partial block): kept in the chunk's
//      buffer for x^T d, and where a parameter vector alpha takes them; added up for a parameter real alpha
ALWAYS_INLINE void keep_partials_of_f_of_block(glm::glm_pass& pass, const int first, const int count,
                                               const simd_t d, double* partials_of_f_of_chunk,
                                               simd_t& alpha_accumulator) {

    simd_store(partials_of_f_of_chunk, d);
    if (pass.partials_of_f != nullptr) {
        if (count == BMVP_LANES) {
            simd_storeu(pass.partials_of_f + first, d);
        } else {
            for (int lane = 0; lane < count; ++lane) pass.partials_of_f[first + lane] = partials_of_f_of_chunk[lane];
        }
    }
    if (pass.partial_of_alpha_scalar != nullptr) alpha_accumulator = glm::add_blocks(alpha_accumulator, d);

}




// ---- Student-t: the one pass, which returns sum_i log1p(r_i) and sum_i w_i, and fills the partials of beta and
//      alpha (through d_i) asked for
struct sums_of_student_t_glm {
    double sum_of_log1p_r = 0.0;
    double sum_of_w = 0.0;
};

template <bool compute_partials>
inline sums_of_student_t_glm student_t_glm_pass(glm::glm_pass& pass, const double* y, const double nu,
                                                const double sigma) {

    const simd_t one = fused_t::simd_set1(1.0);
    const simd_t inverse_of_sigma = fused_t::simd_set1(1.0 / sigma);
    const simd_t inverse_of_nu = fused_t::simd_set1(1.0 / nu);
    const simd_t factor_of_partial_of_f = fused_t::simd_set1((nu + 1.0) / (nu * sigma));
    simd_t sum_of_log1p_r = fused_t::simd_zero();
    simd_t sum_of_w = fused_t::simd_zero();
    simd_t alpha_accumulator = fused_t::simd_zero();
    alignas(64) double f_of_chunk[glm::rows_per_chunk];
    alignas(64) double partials_of_f_of_chunk[glm::rows_per_chunk];
    if constexpr (compute_partials) {
        if (pass.partials_of_beta != nullptr) {
            for (int k = 0; k < pass.K; ++k) pass.partials_of_beta[k] = 0.0;
        }
    }
    for (int chunk_first = 0; chunk_first < pass.N; chunk_first += glm::rows_per_chunk) {
        const int chunk_rows = std::min(glm::rows_per_chunk, pass.N - chunk_first);
        glm::compute_f_of_chunk_into_buffer(pass, chunk_first, chunk_rows, f_of_chunk);
        for (int row = 0; row < chunk_rows; row += BMVP_LANES) {
            const int count = std::min(BMVP_LANES, chunk_rows - row);
            const simd_t y_block = y_of_block(y + chunk_first + row, count);
            const simd_t f = simd_load(f_of_chunk + row);
            const simd_t z = fused_t::simd_mul(fused_t::simd_sub(y_block, f), inverse_of_sigma);
            const simd_t r = fused_t::simd_mul(fused_t::simd_mul(z, z), inverse_of_nu);
            simd_t w = fused_t::simd_div(one, fused_t::simd_add(one, r));
            simd_t log1p_r = fused_t::simd_log1p(r);
            if (count < BMVP_LANES) {
                w = glm::keep_first_lanes(w, count);
                log1p_r = glm::keep_first_lanes(log1p_r, count);
            }
            sum_of_log1p_r = fused_t::simd_add(sum_of_log1p_r, log1p_r);
            sum_of_w = fused_t::simd_add(sum_of_w, w);
            if constexpr (compute_partials) {
                const simd_t d = fused_t::simd_mul(fused_t::simd_mul(factor_of_partial_of_f, z), w);
                keep_partials_of_f_of_block(pass, chunk_first + row, count, d, partials_of_f_of_chunk + row,
                                            alpha_accumulator);
            }
        }
        if constexpr (compute_partials) {
            if (pass.partials_of_beta != nullptr) {
                glm::add_partials_of_beta_of_chunk(pass, chunk_first, chunk_rows, partials_of_f_of_chunk);
            }
        }
    }
    if constexpr (compute_partials) {
        if (pass.partial_of_alpha_scalar != nullptr) {
            *pass.partial_of_alpha_scalar = fused_t::sum_of_lanes(alpha_accumulator);
        }
    }
    sums_of_student_t_glm sums;
    sums.sum_of_log1p_r = fused_t::sum_of_lanes(sum_of_log1p_r);
    sums.sum_of_w = fused_t::sum_of_lanes(sum_of_w);
    return sums;

}




// ---- Weibull: the one pass, which returns sum_i L_i, sum_i f_i, sum_i w_i and sum_i u_i w_i, and fills the
//      partials of beta and alpha (through d_i) asked for; is_lccdf: d_i of log S_i, else of log p_i
struct sums_of_weibull_glm {
    double sum_of_log_y = 0.0;
    double sum_of_f = 0.0;
    double sum_of_w = 0.0;
    double sum_of_u_times_w = 0.0;
};

template <bool is_lccdf, bool compute_partials>
inline sums_of_weibull_glm weibull_glm_pass(glm::glm_pass& pass, const double* y, const double shape) {

    const simd_t one = fused_t::simd_set1(1.0);
    const simd_t shape_block = fused_t::simd_set1(shape);
    simd_t sum_of_log_y = fused_t::simd_zero();
    simd_t sum_of_f = fused_t::simd_zero();
    simd_t sum_of_w = fused_t::simd_zero();
    simd_t sum_of_u_times_w = fused_t::simd_zero();
    simd_t alpha_accumulator = fused_t::simd_zero();
    alignas(64) double f_of_chunk[glm::rows_per_chunk];
    alignas(64) double partials_of_f_of_chunk[glm::rows_per_chunk];
    if constexpr (compute_partials) {
        if (pass.partials_of_beta != nullptr) {
            for (int k = 0; k < pass.K; ++k) pass.partials_of_beta[k] = 0.0;
        }
    }
    for (int chunk_first = 0; chunk_first < pass.N; chunk_first += glm::rows_per_chunk) {
        const int chunk_rows = std::min(glm::rows_per_chunk, pass.N - chunk_first);
        glm::compute_f_of_chunk_into_buffer(pass, chunk_first, chunk_rows, f_of_chunk);
        for (int row = 0; row < chunk_rows; row += BMVP_LANES) {
            const int count = std::min(BMVP_LANES, chunk_rows - row);
            const simd_t y_block = y_of_block(y + chunk_first + row, count);
            simd_t f = simd_load(f_of_chunk + row);
            simd_t log_y = fused_w::simd_log(y_block);
            const simd_t u = fused_t::simd_sub(log_y, f);
            simd_t w = fused_w::simd_exp(fused_t::simd_mul(shape_block, u));
            if (count < BMVP_LANES) {
                f = glm::keep_first_lanes(f, count);
                log_y = glm::keep_first_lanes(log_y, count);
                w = glm::keep_first_lanes(w, count);
            }
            sum_of_log_y = fused_t::simd_add(sum_of_log_y, log_y);
            sum_of_f = fused_t::simd_add(sum_of_f, f);
            sum_of_w = fused_t::simd_add(sum_of_w, w);
            sum_of_u_times_w = fused_t::simd_add(sum_of_u_times_w, fused_t::simd_mul(u, w));
            if constexpr (compute_partials) {
                // (log p: shape (w - 1); log S: shape w; zero in the spare lanes)
                simd_t d = is_lccdf ? fused_t::simd_mul(shape_block, w)
                                    : fused_t::simd_mul(shape_block, fused_t::simd_sub(w, one));
                if (count < BMVP_LANES) d = glm::keep_first_lanes(d, count);
                keep_partials_of_f_of_block(pass, chunk_first + row, count, d, partials_of_f_of_chunk + row,
                                            alpha_accumulator);
            }
        }
        if constexpr (compute_partials) {
            if (pass.partials_of_beta != nullptr) {
                glm::add_partials_of_beta_of_chunk(pass, chunk_first, chunk_rows, partials_of_f_of_chunk);
            }
        }
    }
    if constexpr (compute_partials) {
        if (pass.partial_of_alpha_scalar != nullptr) {
            *pass.partial_of_alpha_scalar = fused_t::sum_of_lanes(alpha_accumulator);
        }
    }
    sums_of_weibull_glm sums;
    sums.sum_of_log_y = fused_t::sum_of_lanes(sum_of_log_y);
    sums.sum_of_f = fused_t::sum_of_lanes(sum_of_f);
    sums.sum_of_w = fused_t::sum_of_lanes(sum_of_w);
    sums.sum_of_u_times_w = fused_t::sum_of_lanes(sum_of_u_times_w);
    return sums;

}




// ---- the operands and partials of the node of a GLM with a linear predictor: beta (K), alpha (1 or N), then the
//      distribution's own real parameters (in the order given, each only when it is a parameter); the partials of
//      beta and alpha are filled by the pass, those of the real parameters after it
template <typename T_alpha, typename T_beta>
struct operands_of_glm_node {
    vari** operands = nullptr;
    double* partials = nullptr;
    int n_operands = 0;
    int n_beta_and_alpha_operands = 0;

    operands_of_glm_node(glm::glm_pass& pass, const T_alpha& alpha, const T_beta& beta,
                         const int n_real_parameters) {

        constexpr bool alpha_is_var    = stan::is_var<stan::scalar_type_t<T_alpha>>::value;
        constexpr bool beta_is_var     = stan::is_var<stan::value_type_t<T_beta>>::value;
        constexpr bool alpha_is_vector = stan::is_eigen<T_alpha>::value;
        auto& arena = stan::math::ChainableStack::instance_->memalloc_;
        const int n_beta_operands  = beta_is_var ? pass.K : 0;
        const int n_alpha_operands = alpha_is_var ? (alpha_is_vector ? pass.N : 1) : 0;
        n_beta_and_alpha_operands = n_beta_operands + n_alpha_operands;
        n_operands = n_beta_and_alpha_operands + n_real_parameters;
        partials = arena.alloc_array<double>(n_operands);
        operands = arena.alloc_array<vari*>(n_operands);
        if constexpr (beta_is_var) {
            pass.partials_of_beta = partials;
            for (int k = 0; k < pass.K; ++k) operands[k] = beta.coeff(k).vi_;
        }
        if constexpr (alpha_is_var) {
            if constexpr (alpha_is_vector) {
                pass.partials_of_f = partials + n_beta_operands;
                for (int i = 0; i < pass.N; ++i) operands[n_beta_operands + i] = alpha.coeff(i).vi_;
            } else {
                pass.partial_of_alpha_scalar = partials + n_beta_operands;
                operands[n_beta_operands] = alpha.vi_;
            }
        }

    }
};




// ---- Student-t identity GLM: the checks, the values, the pass and the node
template <bool propto, typename T_y, typename T_x, typename T_alpha, typename T_beta, typename T_nu,
          typename T_sigma>
inline stan::return_type_t<T_x, T_alpha, T_beta, T_nu, T_sigma> fused_student_t_id_glm_lpdf(
        const T_y& y, const T_x& x, const T_alpha& alpha, const T_beta& beta, const T_nu& nu,
        const T_sigma& sigma) {

    static_assert(std::is_arithmetic<stan::value_type_t<T_y>>::value, "y must be data");
    static_assert(std::is_arithmetic<stan::value_type_t<T_x>>::value, "x must be data");
    constexpr bool alpha_is_var = stan::is_var<stan::scalar_type_t<T_alpha>>::value;
    constexpr bool beta_is_var  = stan::is_var<stan::value_type_t<T_beta>>::value;
    constexpr bool nu_is_var    = stan::is_var<T_nu>::value;
    constexpr bool sigma_is_var = stan::is_var<T_sigma>::value;
    constexpr bool some_parameter = alpha_is_var || beta_is_var || nu_is_var || sigma_is_var;
    static const char* function_name = "nico_student_t_id_glm_lpdf";
    const int N = static_cast<int>(x.rows());
    const int K = static_cast<int>(x.cols());
    stan::math::check_consistent_size(function_name, "Vector of dependent variables", y, N);
    stan::math::check_consistent_size(function_name, "Weight vector", beta, K);
    stan::math::check_consistent_size(function_name, "Vector of intercepts", alpha, N);
    const double nu_value = stan::math::value_of(nu);
    const double sigma_value = stan::math::value_of(sigma);
    stan::math::check_positive_finite(function_name, "Degrees of freedom parameter", nu_value);
    stan::math::check_positive_finite(function_name, "Scale parameter", sigma_value);
    if (N == 0) return 0.0;
    if constexpr (!some_parameter && propto) return 0.0;
    std::vector<double> copy_of_y;
    const double* y_data = data_of_y(y, copy_of_y);
    values_of_linear_predictor values(x, alpha, beta);
    // ---- the value, from the pass's sums: the terms of Stan's student_t_lpdf<propto>
    const double half_nu = 0.5 * nu_value;
    auto fn_value = [&](const sums_of_student_t_glm& sums) {

        double value = -(half_nu + 0.5) * sums.sum_of_log1p_r;
        if (!propto) value -= N * 0.5723649429247001;   // N log(sqrt(pi))
        if (!propto || nu_is_var) {
            value += N * (std::lgamma(half_nu + 0.5) - std::lgamma(half_nu) - 0.5 * std::log(nu_value));
        }
        if (!propto || sigma_is_var) value -= N * std::log(sigma_value);
        return value;

    };
    // ---- the checks of Stan's student_t_lpdf after the pass (before any autodiff node exists):
    auto fn_checks_after_the_pass = [&](const double value) {

        if (!std::isfinite(value)) {
            for (int i = 0; i < N; ++i) {
                stan::math::check_not_nan(function_name, "Random variable", y_data[i]);
            }
            stan::math::check_finite(function_name, "Weight vector", beta);
            stan::math::check_finite(function_name, "Intercept", alpha);
            for (int i = 0; i < N; ++i) {
                stan::math::check_finite(function_name, "Location parameter", values.f_of_row(i));
            }
        }

    };
    if constexpr (!some_parameter) {
        const double value = fn_value(student_t_glm_pass<false>(values.pass, y_data, nu_value, sigma_value));
        fn_checks_after_the_pass(value);
        return value;
    } else {
        constexpr int n_real_parameters = static_cast<int>(nu_is_var) + static_cast<int>(sigma_is_var);
        operands_of_glm_node<T_alpha, T_beta> node(values.pass, alpha, beta, n_real_parameters);
        const sums_of_student_t_glm sums = student_t_glm_pass<true>(values.pass, y_data, nu_value, sigma_value);
        const double value = fn_value(sums);
        fn_checks_after_the_pass(value);
        int operand = node.n_beta_and_alpha_operands;
        if constexpr (nu_is_var) {
            node.operands[operand] = nu.vi_;
            node.partials[operand++] = 0.5 * (N * (stan::math::digamma(half_nu + 0.5) - stan::math::digamma(half_nu)
                                                   + 1.0)
                                              - sums.sum_of_log1p_r - (nu_value + 1.0) * sums.sum_of_w / nu_value);
        }
        if constexpr (sigma_is_var) {
            node.operands[operand] = sigma.vi_;
            node.partials[operand++] = (N * nu_value - (nu_value + 1.0) * sums.sum_of_w) / sigma_value;
        }
        return var(new glm::vari_fused_bernoulli_glm_lpmf(value, node.operands, node.partials, node.n_operands,
                                                          nullptr, nullptr, nullptr, N, K));
    }

}




// ---- Normal identity GLM (Stan's normal_id_glm_lpdf), with z_i = (y_i - f_i) / sigma:
//
//     log p_i     = -0.5 z_i^2 - log(sigma) - 0.5 log(2 pi)
//     d / d f_i   = z_i / sigma
//     d / d sigma = sum_i (z_i^2 - 1) / sigma
//
//      the one pass (the SIMD blocks of the Student-t pass, without its log1p), which returns sum_i z_i^2 and fills
//      the partials of beta and alpha (through d_i) asked for
template <bool compute_partials>
inline double normal_glm_pass(glm::glm_pass& pass, const double* y, const double sigma) {

    const simd_t inverse_of_sigma = fused_t::simd_set1(1.0 / sigma);
    simd_t sum_of_z_squared = fused_t::simd_zero();
    simd_t alpha_accumulator = fused_t::simd_zero();
    alignas(64) double f_of_chunk[glm::rows_per_chunk];
    alignas(64) double partials_of_f_of_chunk[glm::rows_per_chunk];
    if constexpr (compute_partials) {
        if (pass.partials_of_beta != nullptr) {
            for (int k = 0; k < pass.K; ++k) pass.partials_of_beta[k] = 0.0;
        }
    }
    for (int chunk_first = 0; chunk_first < pass.N; chunk_first += glm::rows_per_chunk) {
        const int chunk_rows = std::min(glm::rows_per_chunk, pass.N - chunk_first);
        glm::compute_f_of_chunk_into_buffer(pass, chunk_first, chunk_rows, f_of_chunk);
        for (int row = 0; row < chunk_rows; row += BMVP_LANES) {
            const int count = std::min(BMVP_LANES, chunk_rows - row);
            const simd_t y_block = y_of_block(y + chunk_first + row, count);
            const simd_t f = simd_load(f_of_chunk + row);
            simd_t z = fused_t::simd_mul(fused_t::simd_sub(y_block, f), inverse_of_sigma);
            if (count < BMVP_LANES) z = glm::keep_first_lanes(z, count);
            sum_of_z_squared = fused_t::simd_add(sum_of_z_squared, fused_t::simd_mul(z, z));
            if constexpr (compute_partials) {
                const simd_t d = fused_t::simd_mul(z, inverse_of_sigma);
                keep_partials_of_f_of_block(pass, chunk_first + row, count, d, partials_of_f_of_chunk + row,
                                            alpha_accumulator);
            }
        }
        if constexpr (compute_partials) {
            if (pass.partials_of_beta != nullptr) {
                glm::add_partials_of_beta_of_chunk(pass, chunk_first, chunk_rows, partials_of_f_of_chunk);
            }
        }
    }
    if constexpr (compute_partials) {
        if (pass.partial_of_alpha_scalar != nullptr) {
            *pass.partial_of_alpha_scalar = fused_t::sum_of_lanes(alpha_accumulator);
        }
    }
    return fused_t::sum_of_lanes(sum_of_z_squared);

}

// ---- Normal identity GLM: the checks, the values, the pass and the node
template <bool propto, typename T_y, typename T_x, typename T_alpha, typename T_beta, typename T_sigma>
inline stan::return_type_t<T_x, T_alpha, T_beta, T_sigma> fused_normal_id_glm_lpdf(
        const T_y& y, const T_x& x, const T_alpha& alpha, const T_beta& beta, const T_sigma& sigma) {

    static_assert(std::is_arithmetic<stan::value_type_t<T_y>>::value, "y must be data");
    static_assert(std::is_arithmetic<stan::value_type_t<T_x>>::value, "x must be data");
    constexpr bool alpha_is_var = stan::is_var<stan::scalar_type_t<T_alpha>>::value;
    constexpr bool beta_is_var  = stan::is_var<stan::value_type_t<T_beta>>::value;
    constexpr bool sigma_is_var = stan::is_var<T_sigma>::value;
    constexpr bool some_parameter = alpha_is_var || beta_is_var || sigma_is_var;
    static const char* function_name = "nico_normal_id_glm_lpdf";
    const int N = static_cast<int>(x.rows());
    const int K = static_cast<int>(x.cols());
    stan::math::check_consistent_size(function_name, "Vector of dependent variables", y, N);
    stan::math::check_consistent_size(function_name, "Weight vector", beta, K);
    stan::math::check_consistent_size(function_name, "Vector of intercepts", alpha, N);
    const double sigma_value = stan::math::value_of(sigma);
    stan::math::check_positive_finite(function_name, "Scale vector", sigma_value);
    if (N == 0) return 0.0;
    if constexpr (!some_parameter && propto) return 0.0;
    std::vector<double> copy_of_y;
    const double* y_data = data_of_y(y, copy_of_y);
    values_of_linear_predictor values(x, alpha, beta);
    // ---- the value: the terms of Stan's normal_id_glm_lpdf<propto>
    auto fn_value = [&](const double sum_of_z_squared) {

        double value = -0.5 * sum_of_z_squared;
        if (!propto) value -= N * 0.9189385332046727;   // N log(sqrt(2 pi))
        if (!propto || sigma_is_var) value -= N * std::log(sigma_value);
        return value;

    };
    auto fn_checks_after_the_pass = [&](const double value) {

        if (!std::isfinite(value)) {
            for (int i = 0; i < N; ++i) {
                stan::math::check_finite(function_name, "Vector of dependent variables", y_data[i]);
            }
            stan::math::check_finite(function_name, "Weight vector", beta);
            stan::math::check_finite(function_name, "Intercept", alpha);
        }

    };
    if constexpr (!some_parameter) {
        const double value = fn_value(normal_glm_pass<false>(values.pass, y_data, sigma_value));
        fn_checks_after_the_pass(value);
        return value;
    } else {
        constexpr int n_real_parameters = static_cast<int>(sigma_is_var);
        operands_of_glm_node<T_alpha, T_beta> node(values.pass, alpha, beta, n_real_parameters);
        const double sum_of_z_squared = normal_glm_pass<true>(values.pass, y_data, sigma_value);
        const double value = fn_value(sum_of_z_squared);
        fn_checks_after_the_pass(value);
        if constexpr (sigma_is_var) {
            const int operand = node.n_beta_and_alpha_operands;
            node.operands[operand] = sigma.vi_;
            node.partials[operand] = (sum_of_z_squared - N) / sigma_value;
        }
        return var(new glm::vari_fused_bernoulli_glm_lpmf(value, node.operands, node.partials, node.n_operands,
                                                          nullptr, nullptr, nullptr, N, K));
    }

}




// ---- Weibull log GLM (lpdf and lccdf): the checks, the values, the pass and the node
template <bool propto, bool is_lccdf, typename T_y, typename T_x, typename T_alpha, typename T_beta,
          typename T_shape>
inline stan::return_type_t<T_x, T_alpha, T_beta, T_shape> fused_weibull_log_glm(const T_y& y, const T_x& x,
                                                                                const T_alpha& alpha,
                                                                                const T_beta& beta,
                                                                                const T_shape& shape) {

    static_assert(std::is_arithmetic<stan::value_type_t<T_y>>::value, "y must be data");
    static_assert(std::is_arithmetic<stan::value_type_t<T_x>>::value, "x must be data");
    constexpr bool alpha_is_var = stan::is_var<stan::scalar_type_t<T_alpha>>::value;
    constexpr bool beta_is_var  = stan::is_var<stan::value_type_t<T_beta>>::value;
    constexpr bool shape_is_var = stan::is_var<T_shape>::value;
    constexpr bool some_parameter = alpha_is_var || beta_is_var || shape_is_var;
    static const char* function_name = is_lccdf ? "nico_weibull_log_glm_lccdf" : "nico_weibull_log_glm_lpdf";
    const int N = static_cast<int>(x.rows());
    const int K = static_cast<int>(x.cols());
    stan::math::check_consistent_size(function_name, "Random variable", y, N);
    stan::math::check_consistent_size(function_name, "Weight vector", beta, K);
    stan::math::check_consistent_size(function_name, "Vector of intercepts", alpha, N);
    const double shape_value = stan::math::value_of(shape);
    stan::math::check_positive_finite(function_name, "Shape parameter", shape_value);
    if (N == 0) return 0.0;
    if constexpr (!is_lccdf && !some_parameter && propto) return 0.0;
    std::vector<double> copy_of_y;
    const double* y_data = data_of_y(y, copy_of_y);
    values_of_linear_predictor values(x, alpha, beta);
    // ---- the value, from the pass's sums: the terms of Stan's weibull_lpdf<propto> (log S: -sum_i w_i)
    auto fn_value = [&](const sums_of_weibull_glm& sums) {

        if constexpr (is_lccdf) {
            return -sums.sum_of_w;
        } else {
            double value = -shape_value * sums.sum_of_f - sums.sum_of_w;
            if (!propto || shape_is_var) value += N * std::log(shape_value);
            if (!propto || shape_is_var) value += (shape_value - 1.0) * sums.sum_of_log_y;
            return value;
        }

    };
    // ---- the checks of Stan's weibull_lpdf / weibull_lccdf after the pass (before any autodiff node exists):
    //      a negative y gives negative infinity for the lpdf, as in weibull_lpdf
    bool some_y_negative = false;
    auto fn_checks_after_the_pass = [&](const double value) {

        if (!std::isfinite(value)) {
            for (int i = 0; i < N; ++i) {
                if constexpr (is_lccdf) {
                    stan::math::check_nonnegative(function_name, "Random variable", y_data[i]);
                } else {
                    stan::math::check_finite(function_name, "Random variable", y_data[i]);
                }
                if (y_data[i] < 0.0) some_y_negative = true;
            }
            stan::math::check_finite(function_name, "Weight vector", beta);
            stan::math::check_finite(function_name, "Intercept", alpha);
            for (int i = 0; i < N; ++i) {
                stan::math::check_positive_finite(function_name, "Scale parameter", std::exp(values.f_of_row(i)));
            }
        }

    };
    if constexpr (!some_parameter) {
        const double value = fn_value(weibull_glm_pass<is_lccdf, false>(values.pass, y_data, shape_value));
        fn_checks_after_the_pass(value);
        if (some_y_negative) return stan::math::NEGATIVE_INFTY;
        return value;
    } else {
        constexpr int n_real_parameters = static_cast<int>(shape_is_var);
        operands_of_glm_node<T_alpha, T_beta> node(values.pass, alpha, beta, n_real_parameters);
        const sums_of_weibull_glm sums = weibull_glm_pass<is_lccdf, true>(values.pass, y_data, shape_value);
        const double value = fn_value(sums);
        fn_checks_after_the_pass(value);
        if (some_y_negative) return var(stan::math::NEGATIVE_INFTY);
        if constexpr (shape_is_var) {
            const int operand = node.n_beta_and_alpha_operands;
            node.operands[operand] = shape.vi_;
            if constexpr (is_lccdf) {
                node.partials[operand] = -sums.sum_of_u_times_w;
            } else {
                // (sum_i u_i = sum_i L_i - sum_i f_i)
                node.partials[operand] = N / shape_value + (sums.sum_of_log_y - sums.sum_of_f)
                                         - sums.sum_of_u_times_w;
            }
        }
        return var(new glm::vari_fused_bernoulli_glm_lpmf(value, node.operands, node.partials, node.n_operands,
                                                          nullptr, nullptr, nullptr, N, K));
    }

}




// ---- Bernoulli-logit by group: the inputs and outputs of one pass
struct pass_of_glm_by_group {
    const int* y = nullptr;
    int N = 0;
    int K = 0;
    int G = 0;
    const double* x = nullptr;                  // x(i, k) at x[k * x_column_stride + i]
    int x_column_stride = 0;
    const int* group = nullptr;                 // N values in 1, ..., G
    const double* B = nullptr;                  // B(g, k) at B[k * G + g] (g from 0)
    double* partials_of_B = nullptr;            // G K values, as B, or nullptr
    bool some_y_not_binary = false;
    bool some_f_nan = false;
    bool some_group_out_of_range = false;
};

// ---- the one pass: returns sum_i log p_i, and fills the partials of B asked for
template <bool compute_partials>
inline double sum_of_glm_by_group_log_probabilities(pass_of_glm_by_group& pass) {

    fused_logit::bad_y_bits_t bad_y = fused_logit::no_bad_y_bits();
    fused_logit::nan_lanes_t nan_lanes = fused_logit::no_nan_lanes();
    simd_t sum = fused_logit::zero_block();
    simd_t d = fused_logit::zero_block();
    alignas(64) int group_index_of_chunk[glm::rows_per_chunk];
    alignas(64) double partials_of_f_of_chunk[glm::rows_per_chunk];
    if constexpr (compute_partials) {
        for (int j = 0; j < pass.G * pass.K; ++j) pass.partials_of_B[j] = 0.0;
    }
    for (int chunk_first = 0; chunk_first < pass.N; chunk_first += glm::rows_per_chunk) {
        const int chunk_rows = std::min(glm::rows_per_chunk, pass.N - chunk_first);
        // ---- each row's group, from 0 (a group out of range is flagged, and replaced by 0 so that the gathers
        //      stay inside B; the spare lanes of a final partial block take group 0)
        for (int i = 0; i < chunk_rows; ++i) {
            int index = pass.group[chunk_first + i] - 1;
            if (index < 0 || index >= pass.G) {
                pass.some_group_out_of_range = true;
                index = 0;
            }
            group_index_of_chunk[i] = index;
        }
        const int chunk_rows_rounded_up = ((chunk_rows + BMVP_LANES - 1) / BMVP_LANES) * BMVP_LANES;
        for (int i = chunk_rows; i < chunk_rows_rounded_up; ++i) group_index_of_chunk[i] = 0;
        for (int row = 0; row < chunk_rows; row += BMVP_LANES) {
            const int count = std::min(BMVP_LANES, chunk_rows - row);
            simd_t f = fused_logit::zero_block();
            for (int k = 0; k < pass.K; ++k) {
                const double* x_of_block_start = pass.x + static_cast<std::ptrdiff_t>(k) * pass.x_column_stride
                                                 + chunk_first + row;
                const simd_t x_block = (count == BMVP_LANES) ? simd_loadu(x_of_block_start)
                                                             : glm::load_first_lanes(x_of_block_start, count);
                f = glm::multiply_and_add_blocks(x_block,
                                                 gather_of_block(pass.B + static_cast<std::ptrdiff_t>(k) * pass.G,
                                                                 group_index_of_chunk + row),
                                                 f);
            }
            simd_t s;
            if (count == BMVP_LANES) {
                s = fused_logit::sign_from_y(pass.y + chunk_first + row, bad_y);
            } else {
                alignas(64) int y_buffer[BMVP_LANES] = {0};
                for (int lane = 0; lane < count; ++lane) y_buffer[lane] = pass.y[chunk_first + row + lane];
                s = fused_logit::sign_from_y(y_buffer, bad_y);
            }
            fused_logit::collect_nan_lanes(nan_lanes, f);
            simd_t log_probabilities = fused_logit::log_probability_and_partial_of_block<compute_partials>(f,
                                                                                                           s,
                                                                                                           d);
            if (count < BMVP_LANES) {
                log_probabilities = glm::keep_first_lanes(log_probabilities, count);
                d = glm::keep_first_lanes(d, count);
            }
            sum = glm::add_blocks(sum, log_probabilities);
            if constexpr (compute_partials) simd_store(partials_of_f_of_chunk + row, d);
        }
        if constexpr (compute_partials) {
            // ---- d / d B(g, k) += x(i, k) d_i, one column of B at a time; a run of rows of the same group is
            //      summed before it is added
            for (int k = 0; k < pass.K; ++k) {
                const double* x_of_column = pass.x + static_cast<std::ptrdiff_t>(k) * pass.x_column_stride
                                            + chunk_first;
                double* partials_of_B_of_column = pass.partials_of_B + static_cast<std::ptrdiff_t>(k) * pass.G;
                int group_of_run = group_index_of_chunk[0];
                double sum_of_run = 0.0;
                for (int i = 0; i < chunk_rows; ++i) {
                    const int index = group_index_of_chunk[i];
                    if (index != group_of_run) {
                        partials_of_B_of_column[group_of_run] += sum_of_run;
                        group_of_run = index;
                        sum_of_run = 0.0;
                    }
                    sum_of_run += x_of_column[i] * partials_of_f_of_chunk[i];
                }
                partials_of_B_of_column[group_of_run] += sum_of_run;
            }
        }
    }
    pass.some_y_not_binary = fused_logit::any_bad_y(bad_y);
    pass.some_f_nan = fused_logit::any_nan(nan_lanes);
    return fused_logit::sum_of_lanes(sum);

}

// ---- Bernoulli-logit by group: the checks, the values, the pass and the node (the first version, with the
//      intercepts as a column of ones in x and gathers; kept, no longer called: 3.7% slower than that function
//      of 13 Oct at full load, so replaced by the version with intercepts below)
template <bool propto, typename T_y, typename T_x, typename T_group, typename T_B>
inline stan::return_type_t<T_x, T_B> fused_bernoulli_logit_glm_by_group_with_gathers_lpmf(const T_y& y,
                                                                                          const T_x& x,
                                                                             const T_group& group,
                                                                             const T_B& B) {

    static_assert(std::is_same<stan::value_type_t<T_y>, int>::value, "y must be an array of int");
    static_assert(std::is_same<stan::value_type_t<T_group>, int>::value, "group must be an array of int");
    static_assert(std::is_arithmetic<stan::value_type_t<T_x>>::value, "x must be data");
    constexpr bool B_is_var = stan::is_var<stan::value_type_t<T_B>>::value;
    static const char* function_name = "nico_bernoulli_logit_glm_with_coefficients_by_group_lpmf";
    const int N = static_cast<int>(x.rows());
    const int K = static_cast<int>(x.cols());
    const int G = static_cast<int>(B.rows());
    stan::math::check_consistent_size(function_name, "Vector of dependent variables", y, N);
    stan::math::check_consistent_size(function_name, "Group indices", group, N);
    stan::math::check_size_match(function_name, "Columns of the matrix of coefficients", B.cols(),
                                 "columns of the matrix of independent variables", K);
    if (N == 0) return 0.0;
    if constexpr (!B_is_var && propto) {
        // (nothing depends on a parameter: Stan checks y, then returns 0)
        stan::math::check_bounded(function_name, "Vector of dependent variables", y, 0, 1);
        return 0.0;
    }
    // ---- the values of x (in place when its columns are contiguous) and B:
    int x_column_stride = 0;
    const double* x_data = glm::column_major_data_or_nullptr(x, x_column_stride);
    Eigen::Matrix<double, -1, -1> x_values;
    if (x_data == nullptr) {
        x_values = stan::math::value_of(x);
        x_data = x_values.data();
        x_column_stride = N;
    }
    const Eigen::Matrix<double, -1, -1> B_values = stan::math::value_of(B);
    pass_of_glm_by_group pass;
    pass.y = y.data();
    pass.N = N;
    pass.K = K;
    pass.G = G;
    pass.x = x_data;
    pass.x_column_stride = x_column_stride;
    pass.group = group.data();
    pass.B = B_values.data();
    // ---- the checks after the pass (before any autodiff node exists):
    auto fn_checks_after_the_pass = [&](const double sum) {

        if (pass.some_group_out_of_range) {
            stan::math::check_bounded(function_name, "Group indices", group, 1, G);
        }
        if (pass.some_y_not_binary) {
            stan::math::check_bounded(function_name, "Vector of dependent variables", y, 0, 1);
        }
        if (!std::isfinite(sum) || pass.some_f_nan) {
            stan::math::check_finite(function_name, "Matrix of coefficients", B_values);
            stan::math::check_finite(function_name, "Matrix of independent variables", x);
        }

    };
    if constexpr (!B_is_var) {
        const double sum = sum_of_glm_by_group_log_probabilities<false>(pass);
        fn_checks_after_the_pass(sum);
        return sum;
    } else {
        auto& arena = stan::math::ChainableStack::instance_->memalloc_;
        const int n_operands = G * K;
        double* partials = arena.alloc_array<double>(n_operands);
        pass.partials_of_B = partials;
        const double sum = sum_of_glm_by_group_log_probabilities<true>(pass);
        fn_checks_after_the_pass(sum);
        vari** operands = arena.alloc_array<vari*>(n_operands);
        for (int k = 0; k < K; ++k) {
            for (int g = 0; g < G; ++g) operands[static_cast<std::ptrdiff_t>(k) * G + g] = B.coeff(g, k).vi_;
        }
        return var(new glm::vari_fused_bernoulli_glm_lpmf(sum, operands, partials, n_operands, nullptr, nullptr,
                                                          nullptr, N, K));
    }

}




// ---- Bernoulli-logit by group with each group's own intercept alpha(g) and coefficients B(g, ) (any number of
//      columns K of x, as Stan's GLMs), in the way of nico_bernoulli_logit_grouped_regression_lpmf
//      (K = 1 only): f row by row (no gathers), the values of alpha and B read once (not once per row),
//      BayesMVP's exp and log1p kernels over all the rows, then the partials, each run of rows of the same group
//      summed in registers before it is added (data sorted by group, as in the benchmark models; any order of the
//      rows gives the same result). The inputs of one pass:
struct pass_of_glm_by_group_with_intercepts {
    const int* y = nullptr;
    int N = 0;
    int K = 0;
    int G = 0;
    const double* x = nullptr;                  // x(i, k) at x[k * x_column_stride + i]
    int x_column_stride = 0;
    const int* group = nullptr;                 // N values in 1, ..., G
    const double* alpha = nullptr;              // alpha(g) at alpha[g] (g from 0)
    const double* B = nullptr;                  // B(g, k) at B[k * G + g] (g from 0)
    bool some_y_not_binary = false;
    bool some_group_out_of_range = false;
};

// ---- each row's -s_i f_i and s_i, with s_i = 2 y_i - 1 and f_i = alpha(g) + sum_k x(i, k) B(g, k), g = group_i
//      (K_known: the number of columns when it is 0, 1 or 2, so that the sum over k is unrolled; else -1)
template <int K_known>
inline void minus_signed_f_and_signs_of_rows(pass_of_glm_by_group_with_intercepts& pass, double* minus_signed_f,
                                             double* signs) {

    const int K = (K_known >= 0) ? K_known : pass.K;
    bool some_y_not_binary = false;
    bool some_group_out_of_range = false;
    for (int i = 0; i < pass.N; ++i) {
        int g = pass.group[i] - 1;
        if (g < 0 || g >= pass.G) {
            some_group_out_of_range = true;
            g = 0;
        }
        const int y_i = pass.y[i];
        if (y_i != 0 && y_i != 1) some_y_not_binary = true;
        double f = pass.alpha[g];
        for (int k = 0; k < K; ++k) {
            f += pass.x[static_cast<std::ptrdiff_t>(k) * pass.x_column_stride + i]
                 * pass.B[static_cast<std::ptrdiff_t>(k) * pass.G + g];
        }
        const double s = static_cast<double>(2 * y_i - 1);
        minus_signed_f[i] = -s * f;
        signs[i] = s;
    }
    pass.some_y_not_binary = some_y_not_binary;
    pass.some_group_out_of_range = some_group_out_of_range;

}

// ---- out[g - 1] += the sum over each run of rows of the same group g of partials_of_f[i] (times x_column[i]
//      when x_column is given): one addition to out per run
inline void add_sums_of_runs_of_the_same_group(const int* group, const int N, const double* partials_of_f,
                                               const double* x_column, double* out) {

    int i = 0;
    while (i < N) {
        const int g = group[i];
        double sum_of_run = 0.0;
        int row = i;
        if (x_column == nullptr) {
            for (; row < N && group[row] == g; ++row) sum_of_run += partials_of_f[row];
        } else {
            for (; row < N && group[row] == g; ++row) sum_of_run += x_column[row] * partials_of_f[row];
        }
        out[g - 1] += sum_of_run;
        i = row;
    }

}

// ---- the partials of alpha and B: with K known (0, 1 or 2), one pass over the runs with every sum in a register;
//      else one pass for alpha and one for each column of B
template <int K_known>
inline void add_partials_of_alpha_and_B(const pass_of_glm_by_group_with_intercepts& pass,
                                        const double* partials_of_f, double* partials_of_alpha,
                                        double* partials_of_B) {

    const int N = pass.N;
    const int G = pass.G;
    const int K = (K_known >= 0) ? K_known : pass.K;
    std::fill(partials_of_alpha, partials_of_alpha + G, 0.0);
    std::fill(partials_of_B, partials_of_B + static_cast<std::ptrdiff_t>(G) * K, 0.0);
    if constexpr (K_known >= 0) {
        int i = 0;
        while (i < N) {
            const int g = pass.group[i];
            double sum_of_run = 0.0;
            double sums_of_run_times_x[(K_known > 0) ? K_known : 1] = {};
            int row = i;
            for (; row < N && pass.group[row] == g; ++row) {
                const double partial = partials_of_f[row];
                sum_of_run += partial;
                for (int k = 0; k < K_known; ++k) {
                    sums_of_run_times_x[k] += pass.x[static_cast<std::ptrdiff_t>(k) * pass.x_column_stride + row]
                                              * partial;
                }
            }
            partials_of_alpha[g - 1] += sum_of_run;
            for (int k = 0; k < K_known; ++k) {
                partials_of_B[static_cast<std::ptrdiff_t>(k) * G + g - 1] += sums_of_run_times_x[k];
            }
            i = row;
        }
    } else {
        add_sums_of_runs_of_the_same_group(pass.group, N, partials_of_f, nullptr, partials_of_alpha);
        for (int k = 0; k < K; ++k) {
            add_sums_of_runs_of_the_same_group(pass.group, N, partials_of_f,
                                               pass.x + static_cast<std::ptrdiff_t>(k) * pass.x_column_stride,
                                               partials_of_B + static_cast<std::ptrdiff_t>(k) * G);
        }
    }

}

// ---- the one pass: returns sum_i log p_i and (compute_partials) fills the partials of alpha and B, with the
//      kernels of that function (nico::math::grouped_bernoulli: -|z|, the partials of f and the
//      sum) and BayesMVP's exp and log1p kernels
template <int K_known, bool compute_partials>
inline double sum_of_glm_by_group_with_intercepts_log_probabilities(pass_of_glm_by_group_with_intercepts& pass,
                                                                    double* partials_of_alpha,
                                                                    double* partials_of_B) {

    namespace grouped = ::nico::math::grouped_bernoulli;
    const int N = pass.N;
    auto& buffers = nico::math::scratch();
    buffers.ensure(N);
    double* minus_signed_f = buffers.a.data();
    double* log_probabilities = buffers.b.data();
    double* partials_of_f = buffers.c.data();
    minus_signed_f_and_signs_of_rows<K_known>(pass, minus_signed_f, partials_of_f);
    // (a group out of range or a y not in {0, 1}: the caller's checks throw, with Stan's messages)
    if (pass.some_group_out_of_range || pass.some_y_not_binary) return 0.0;
    grouped::negative_abs(minus_signed_f, N, log_probabilities);
    nico::math::k_exp(log_probabilities, N);
    if constexpr (compute_partials) {
        grouped::derivatives(minus_signed_f, log_probabilities, partials_of_f, N, partials_of_f);
    }
    nico::math::k_log1p(log_probabilities, N);
    const double sum = grouped::sum_log_probability(minus_signed_f, log_probabilities, N);
    if constexpr (compute_partials) {
        add_partials_of_alpha_and_B<K_known>(pass, partials_of_f, partials_of_alpha, partials_of_B);
    }
    return sum;

}

template <bool compute_partials>
inline double sum_of_glm_by_group_with_intercepts_for_any_K(pass_of_glm_by_group_with_intercepts& pass,
                                                           double* partials_of_alpha, double* partials_of_B) {

    switch (pass.K) {
        case 0:
            return sum_of_glm_by_group_with_intercepts_log_probabilities<0, compute_partials>(
                       pass, partials_of_alpha, partials_of_B);
        case 1:
            return sum_of_glm_by_group_with_intercepts_log_probabilities<1, compute_partials>(
                       pass, partials_of_alpha, partials_of_B);
        case 2:
            return sum_of_glm_by_group_with_intercepts_log_probabilities<2, compute_partials>(
                       pass, partials_of_alpha, partials_of_B);
        default:
            return sum_of_glm_by_group_with_intercepts_log_probabilities<-1, compute_partials>(
                       pass, partials_of_alpha, partials_of_B);
    }

}

// ---- Bernoulli-logit by group with each group's own intercept: the checks, the values, the pass and the node
template <bool propto, typename T_y, typename T_x, typename T_group, typename T_alpha, typename T_B>
inline stan::return_type_t<T_x, T_alpha, T_B> fused_bernoulli_logit_glm_by_group_lpmf(const T_y& y,
                                                                                      const T_x& x,
                                                                                      const T_group& group,
                                                                                      const T_alpha& alpha,
                                                                                      const T_B& B) {

    static_assert(std::is_same<stan::value_type_t<T_y>, int>::value, "y must be an array of int");
    static_assert(std::is_same<stan::value_type_t<T_group>, int>::value, "group must be an array of int");
    static_assert(std::is_arithmetic<stan::value_type_t<T_x>>::value, "x must be data");
    constexpr bool alpha_is_var = stan::is_var<stan::value_type_t<T_alpha>>::value;
    constexpr bool B_is_var = stan::is_var<stan::value_type_t<T_B>>::value;
    static const char* function_name = "nico_bernoulli_logit_glm_with_coefficients_by_group_lpmf";
    const auto& alpha_ref = stan::math::to_ref(alpha);
    const auto& B_ref = stan::math::to_ref(B);
    const int N = static_cast<int>(x.rows());
    const int K = static_cast<int>(x.cols());
    const int G = static_cast<int>(alpha_ref.size());
    stan::math::check_consistent_size(function_name, "Vector of dependent variables", y, N);
    stan::math::check_consistent_size(function_name, "Group indices", group, N);
    stan::math::check_size_match(function_name, "Rows of the matrix of coefficients", B_ref.rows(),
                                 "size of the vector of intercepts", G);
    stan::math::check_size_match(function_name, "Columns of the matrix of coefficients", B_ref.cols(),
                                 "columns of the matrix of independent variables", K);
    if (N == 0) return 0.0;
    // (no groups: every group index is out of range)
    if (G == 0) stan::math::check_bounded(function_name, "Group indices", group, 1, G);
    if constexpr (!alpha_is_var && !B_is_var && propto) {
        // (nothing depends on a parameter: Stan checks y, then returns 0)
        stan::math::check_bounded(function_name, "Vector of dependent variables", y, 0, 1);
        return 0.0;
    }
    // ---- the values of x (in place when its columns are contiguous), alpha and B:
    int x_column_stride = 0;
    const double* x_data = glm::column_major_data_or_nullptr(x, x_column_stride);
    Eigen::Matrix<double, -1, -1> x_values;
    if (x_data == nullptr) {
        x_values = stan::math::value_of(x);
        x_data = x_values.data();
        x_column_stride = N;
    }
    const Eigen::Matrix<double, -1, 1> alpha_values = stan::math::value_of(alpha_ref);
    const Eigen::Matrix<double, -1, -1> B_values = stan::math::value_of(B_ref);
    pass_of_glm_by_group_with_intercepts pass;
    pass.y = y.data();
    pass.N = N;
    pass.K = K;
    pass.G = G;
    pass.x = x_data;
    pass.x_column_stride = x_column_stride;
    pass.group = group.data();
    pass.alpha = alpha_values.data();
    pass.B = B_values.data();
    // ---- the checks after the pass (before any autodiff node exists):
    auto fn_checks_after_the_pass = [&](const double sum) {

        if (pass.some_group_out_of_range) {
            stan::math::check_bounded(function_name, "Group indices", group, 1, G);
        }
        if (pass.some_y_not_binary) {
            stan::math::check_bounded(function_name, "Vector of dependent variables", y, 0, 1);
        }
        if (!std::isfinite(sum)) {
            stan::math::check_finite(function_name, "Vector of intercepts", alpha_values);
            stan::math::check_finite(function_name, "Matrix of coefficients", B_values);
            stan::math::check_finite(function_name, "Matrix of independent variables", x);
        }

    };
    if constexpr (!alpha_is_var && !B_is_var) {
        const double sum = sum_of_glm_by_group_with_intercepts_for_any_K<false>(pass, nullptr, nullptr);
        fn_checks_after_the_pass(sum);
        return sum;
    } else {
        auto& arena = stan::math::ChainableStack::instance_->memalloc_;
        const std::ptrdiff_t n_of_B = static_cast<std::ptrdiff_t>(G) * K;
        // (the partials of alpha, then of B, column by column)
        double* partials = arena.alloc_array<double>(G + n_of_B);
        const double sum = sum_of_glm_by_group_with_intercepts_for_any_K<true>(pass, partials, partials + G);
        fn_checks_after_the_pass(sum);
        // ---- the operands: alpha's (when a parameter), then B's (when a parameter), in the order of the partials
        const int n_operands = (alpha_is_var ? G : 0) + (B_is_var ? static_cast<int>(n_of_B) : 0);
        vari** operands = arena.alloc_array<vari*>(n_operands);
        int operand = 0;
        if constexpr (alpha_is_var) {
            for (int g = 0; g < G; ++g) operands[operand++] = alpha_ref.coeff(g).vi_;
        }
        if constexpr (B_is_var) {
            for (int k = 0; k < K; ++k) {
                for (int g = 0; g < G; ++g) operands[operand++] = B_ref.coeff(g, k).vi_;
            }
        }
        double* partials_of_operands = alpha_is_var ? partials : partials + G;
        return var(new glm::vari_fused_bernoulli_glm_lpmf(sum, operands, partials_of_operands, n_operands, nullptr,
                                                          nullptr, nullptr, N, K));
    }

}

// ---- Bernoulli-logit by group without intercepts (B alone; e.g. x with a column of ones): the same pass, with
//      intercepts of 0 (data)
template <bool propto, typename T_y, typename T_x, typename T_group, typename T_B>
inline stan::return_type_t<T_x, T_B> fused_bernoulli_logit_glm_by_group_lpmf(const T_y& y, const T_x& x,
                                                                             const T_group& group,
                                                                             const T_B& B) {

    const Eigen::Matrix<double, -1, 1> intercepts_of_zero = Eigen::Matrix<double, -1, 1>::Zero(B.rows());
    return fused_bernoulli_logit_glm_by_group_lpmf<propto>(y, x, group, intercepts_of_zero, B);

}

}  // namespace fused_glms_by_family
}  // namespace nico::math




// ---- the Stan-facing functions (global scope; stanc calls an lpdf / lpmf as
//      fast_..._lpdf<propto__>(y, ..., pstream__) and an lccdf as fast_..._lccdf(y, ..., pstream__))
namespace nico::math {

template <bool propto, typename T_y, typename T_x, typename T_alpha, typename T_beta, typename T_nu,
          typename T_sigma,
          stan::require_any_t<stan::is_std_vector<T_y>, stan::is_eigen_col_vector<T_y>>* = nullptr,
          stan::require_eigen_matrix_dynamic_t<T_x>* = nullptr,
          stan::require_any_t<stan::is_stan_scalar<T_alpha>, stan::is_eigen_col_vector<T_alpha>>* = nullptr,
          stan::require_eigen_col_vector_t<T_beta>* = nullptr,
          stan::require_all_stan_scalar_t<T_nu, T_sigma>* = nullptr>
inline stan::return_type_t<T_x, T_alpha, T_beta, T_nu, T_sigma> nico_student_t_id_glm_lpdf(
        const T_y& y, const T_x& x, const T_alpha& alpha, const T_beta& beta, const T_nu& nu,
        const T_sigma& sigma, std::ostream* /*pstream__*/) {

    namespace fused = nico::math::fused_glms_by_family;
    return fused::fused_student_t_id_glm_lpdf<propto>(y, x, alpha, beta, nu, sigma);

}

}  // namespace nico::math

using nico::math::nico_student_t_id_glm_lpdf;

//     real nico_normal_id_glm_lpdf(vector y, matrix x, vector alpha, vector beta, real sigma);   (alpha: real too)
namespace nico::math {

template <bool propto, typename T_y, typename T_x, typename T_alpha, typename T_beta, typename T_sigma,
          stan::require_any_t<stan::is_std_vector<T_y>, stan::is_eigen_col_vector<T_y>>* = nullptr,
          stan::require_eigen_matrix_dynamic_t<T_x>* = nullptr,
          stan::require_any_t<stan::is_stan_scalar<T_alpha>, stan::is_eigen_col_vector<T_alpha>>* = nullptr,
          stan::require_eigen_col_vector_t<T_beta>* = nullptr,
          stan::require_stan_scalar_t<T_sigma>* = nullptr>
inline stan::return_type_t<T_x, T_alpha, T_beta, T_sigma> nico_normal_id_glm_lpdf(
        const T_y& y, const T_x& x, const T_alpha& alpha, const T_beta& beta, const T_sigma& sigma,
        std::ostream* /*pstream__*/) {

    namespace fused = nico::math::fused_glms_by_family;
    const auto& alpha_ref = stan::math::to_ref(alpha);
    const auto& beta_ref = stan::math::to_ref(beta);
    return fused::fused_normal_id_glm_lpdf<propto>(y, x, alpha_ref, beta_ref, sigma);

}

}  // namespace nico::math

using nico::math::nico_normal_id_glm_lpdf;

namespace nico::math {

template <bool propto, typename T_y, typename T_x, typename T_alpha, typename T_beta, typename T_shape,
          stan::require_any_t<stan::is_std_vector<T_y>, stan::is_eigen_col_vector<T_y>>* = nullptr,
          stan::require_eigen_matrix_dynamic_t<T_x>* = nullptr,
          stan::require_any_t<stan::is_stan_scalar<T_alpha>, stan::is_eigen_col_vector<T_alpha>>* = nullptr,
          stan::require_eigen_col_vector_t<T_beta>* = nullptr,
          stan::require_stan_scalar_t<T_shape>* = nullptr>
inline stan::return_type_t<T_x, T_alpha, T_beta, T_shape> nico_weibull_log_glm_lpdf(
        const T_y& y, const T_x& x, const T_alpha& alpha, const T_beta& beta, const T_shape& shape,
        std::ostream* /*pstream__*/) {

    namespace fused = nico::math::fused_glms_by_family;
    return fused::fused_weibull_log_glm<propto, false>(y, x, alpha, beta, shape);

}

}  // namespace nico::math

using nico::math::nico_weibull_log_glm_lpdf;

namespace nico::math {

template <typename T_y, typename T_x, typename T_alpha, typename T_beta, typename T_shape,
          stan::require_any_t<stan::is_std_vector<T_y>, stan::is_eigen_col_vector<T_y>>* = nullptr,
          stan::require_eigen_matrix_dynamic_t<T_x>* = nullptr,
          stan::require_any_t<stan::is_stan_scalar<T_alpha>, stan::is_eigen_col_vector<T_alpha>>* = nullptr,
          stan::require_eigen_col_vector_t<T_beta>* = nullptr,
          stan::require_stan_scalar_t<T_shape>* = nullptr>
inline stan::return_type_t<T_x, T_alpha, T_beta, T_shape> nico_weibull_log_glm_lccdf(
        const T_y& y, const T_x& x, const T_alpha& alpha, const T_beta& beta, const T_shape& shape,
        std::ostream* /*pstream__*/) {

    namespace fused = nico::math::fused_glms_by_family;
    return fused::fused_weibull_log_glm<false, true>(y, x, alpha, beta, shape);

}

}  // namespace nico::math

using nico::math::nico_weibull_log_glm_lccdf;

namespace nico::math {

template <bool propto, typename T_y, typename T_x, typename T_group, typename T_B,
          stan::require_std_vector_vt<std::is_integral, T_y>* = nullptr,
          stan::require_eigen_matrix_dynamic_t<T_x>* = nullptr,
          stan::require_std_vector_vt<std::is_integral, T_group>* = nullptr,
          stan::require_eigen_matrix_dynamic_t<T_B>* = nullptr>
inline stan::return_type_t<T_x, T_B> nico_bernoulli_logit_glm_with_coefficients_by_group_lpmf(
        const T_y& y, const T_x& x, const T_group& group, const T_B& B, std::ostream* /*pstream__*/) {

    namespace fused = nico::math::fused_glms_by_family;
    return fused::fused_bernoulli_logit_glm_by_group_lpmf<propto>(y, x, group, B);

}

}  // namespace nico::math

using nico::math::nico_bernoulli_logit_glm_with_coefficients_by_group_lpmf;

namespace nico::math {

template <bool propto, typename T_y, typename T_x, typename T_group, typename T_alpha, typename T_B,
          stan::require_std_vector_vt<std::is_integral, T_y>* = nullptr,
          stan::require_eigen_matrix_dynamic_t<T_x>* = nullptr,
          stan::require_std_vector_vt<std::is_integral, T_group>* = nullptr,
          stan::require_eigen_col_vector_t<T_alpha>* = nullptr,
          stan::require_eigen_matrix_dynamic_t<T_B>* = nullptr>
inline stan::return_type_t<T_x, T_alpha, T_B> nico_bernoulli_logit_glm_with_coefficients_by_group_lpmf(
        const T_y& y, const T_x& x, const T_group& group, const T_alpha& alpha, const T_B& B,
        std::ostream* /*pstream__*/) {

    namespace fused = nico::math::fused_glms_by_family;
    return fused::fused_bernoulli_logit_glm_by_group_lpmf<propto>(y, x, group, alpha, B);

}

}  // namespace nico::math

using nico::math::nico_bernoulli_logit_glm_with_coefficients_by_group_lpmf;






















