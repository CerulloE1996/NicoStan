#pragma once
//
// -| --------- BayesMVP squared exponential covariance plus a diagonal term, and its Cholesky factor, in one ----
//              composition, exposed to Stan as an external function
//
// nico_gp_exp_quad_cov_cholesky(x, sigma, length_scale, diagonal): the same value as
//
//     cholesky_decompose(add_diag(gp_exp_quad_cov(x, sigma, length_scale), diagonal))
//
// for array[] real data points x and real sigma, length_scale and diagonal (each data or a parameter). General:
// any model with a squared exponential Gaussian process (regression, classification, latent fields) whose
// covariance has a diagonal term (a noise variance, a nugget or a jitter).
//
// The covariance K is computed by BayesMVP's nico_gp_exp_quad_cov arithmetic (its exp kernel, one column segment
// at a time; stan_external_gp_exp_quad_cov.hpp) as plain doubles, the diagonal term is added once to each
// diagonal entry (as add_diag), and K is held as ONE var_value matrix: none of its N x N entries is a scalar var.
// The factorisation is BayesMVP's nico_cholesky_decompose (stan_external_cholesky_decompose.hpp: Eigen's blocked
// LLT schedule with its rank-k update computed by explicit SIMD packets, in Eigen's own order, so the same factor
// as Stan's), and its reverse pass is Stan's own (Giles' algorithm up to N = 35, Murray's blocked algorithm
// above). Only the returned lower triangle is made of scalar vars (nochain stack), as the Stan external API needs
// a matrix of vars.
//
// Reverse mode: Stan's Cholesky reverse pass writes the adjoint of the lower triangle of K (the adjoint of each
// distinct symmetric entry). ONE callback then adds
//
//     adj(sigma)        += 2 / sigma (sum_{i > j} adj(K_ij) K_ij + sigma^2 sum_i adj(K_ii))
//     adj(length_scale) += sum_{i > j} adj(K_ij) K_ij (x_i - x_j)^2 / length_scale^3
//     adj(diagonal)     += sum_i adj(K_ii)
//
// with both sums below the diagonal in SIMD lanes (AVX-512: 8; AVX2, including -DBAYESMVP_FORCE_AVX2: 4), column
// by column, the last partial packet of each column with masked loads (no scalar loop over the entries).
//
// Checks, as Stan: those of gp_exp_quad_cov (sigma positive, length_scale positive and finite, no point NaN) and
// of cholesky_decompose (symmetric, not NaN, positive definite).
//
// Declared in the Stan functions block (no body) as
//
//     matrix nico_gp_exp_quad_cov_cholesky(data array[] real x, real sigma, real length_scale, real diagonal);
//
// ---------------------------------------------------------------------------------------------------------------

#include "stan_external_functions.hpp"
#include "stan_external_cholesky_decompose.hpp"

namespace nico::math {
namespace gp_exp_quad_cov_cholesky {

// ---- below the diagonal of the column-major n x n matrices of the entries K and of their adjoints:
//      sum_{i > j} K_ij adj(K_ij)  and  sum_{i > j} K_ij adj(K_ij) (x_i - x_j)^2
inline void sums_of_entries_times_adjoints_below_the_diagonal(const double* entries, const double* adjoints,
                                                              const double* x, const int n,
                                                              double& sum_of_products,
                                                              double& sum_of_products_times_squared_distances) {

#if defined(NICO_SIMD_SCALAR_KERNELS)
    sum_of_products = 0.0;
    sum_of_products_times_squared_distances = 0.0;
    for (int j = 0; j < n; ++j) {
        const double* entries_of_column  = entries  + static_cast<std::ptrdiff_t>(j) * n;
        const double* adjoints_of_column = adjoints + static_cast<std::ptrdiff_t>(j) * n;
        for (int i = j + 1; i < n; ++i) {
            const double product = entries_of_column[i] * adjoints_of_column[i];
            const double distance = x[i] - x[j];
            sum_of_products += product;
            sum_of_products_times_squared_distances = std::fma(product * distance, distance,
                                                               sum_of_products_times_squared_distances);
        }
    }
#elif BMVP_LANES == 8
    ALIGN64  __m512d sums_of_products = _mm512_setzero_pd();
    ALIGN64  __m512d sums_of_products_times_squared_distances = _mm512_setzero_pd();
    for (int j = 0; j < n; ++j) {
        const double* entries_of_column  = entries  + static_cast<std::ptrdiff_t>(j) * n;
        const double* adjoints_of_column = adjoints + static_cast<std::ptrdiff_t>(j) * n;
        ALIGN64  __m512d const x_of_column = _mm512_set1_pd(x[j]);
        for (int i = j + 1; i < n; i += 8) {
            const int count = n - i;
            const __mmask8 lanes = count >= 8 ? static_cast<__mmask8>(0xFF)
                                              : static_cast<__mmask8>((1u << count) - 1u);
            ALIGN64  __m512d const product = _mm512_mul_pd(_mm512_maskz_loadu_pd(lanes, entries_of_column + i),
                                                  _mm512_maskz_loadu_pd(lanes, adjoints_of_column + i));
            ALIGN64  __m512d const distance = _mm512_maskz_sub_pd(lanes, _mm512_maskz_loadu_pd(lanes, x + i),
                                                         x_of_column);
            sums_of_products = _mm512_add_pd(sums_of_products, product);
            sums_of_products_times_squared_distances
                = _mm512_fmadd_pd(_mm512_mul_pd(product, distance), distance,
                                  sums_of_products_times_squared_distances);
        }
    }
    sum_of_products = _mm512_reduce_add_pd(sums_of_products);
    sum_of_products_times_squared_distances = _mm512_reduce_add_pd(sums_of_products_times_squared_distances);
#else
    // (the lanes of a partial packet: the first count of the 4 lanes, as a load mask)
    alignas(32) static const long long lane_masks[8] = {-1, -1, -1, -1, 0, 0, 0, 0};
    ALIGN32  __m256d sums_of_products = _mm256_setzero_pd();
    ALIGN32  __m256d sums_of_products_times_squared_distances = _mm256_setzero_pd();
    for (int j = 0; j < n; ++j) {
        const double* entries_of_column  = entries  + static_cast<std::ptrdiff_t>(j) * n;
        const double* adjoints_of_column = adjoints + static_cast<std::ptrdiff_t>(j) * n;
        ALIGN32  __m256d const x_of_column = _mm256_set1_pd(x[j]);
        for (int i = j + 1; i < n; i += 4) {
            const int count = n - i;
            ALIGN32  __m256i const lanes = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                                        lane_masks + (count >= 4 ? 0 : 4 - count)));
            ALIGN32  __m256d const product = _mm256_mul_pd(_mm256_maskload_pd(entries_of_column + i, lanes),
                                                  _mm256_maskload_pd(adjoints_of_column + i, lanes));
            ALIGN32  __m256d const distance = _mm256_and_pd(_mm256_sub_pd(_mm256_maskload_pd(x + i, lanes),
                                                                 x_of_column),
                                                   _mm256_castsi256_pd(lanes));
            sums_of_products = _mm256_add_pd(sums_of_products, product);
            sums_of_products_times_squared_distances
                = _mm256_fmadd_pd(_mm256_mul_pd(product, distance), distance,
                                  sums_of_products_times_squared_distances);
        }
    }
    alignas(32) double lanes_of_sums[4];
    _mm256_store_pd(lanes_of_sums, sums_of_products);
    sum_of_products = (lanes_of_sums[0] + lanes_of_sums[1]) + (lanes_of_sums[2] + lanes_of_sums[3]);
    _mm256_store_pd(lanes_of_sums, sums_of_products_times_squared_distances);
    sum_of_products_times_squared_distances = (lanes_of_sums[0] + lanes_of_sums[1])
                                              + (lanes_of_sums[2] + lanes_of_sums[3]);
#endif

}

}  // namespace gp_exp_quad_cov_cholesky
}  // namespace nico::math


// ---- the Stan-facing function (global scope; stanc calls it as  nico_gp_exp_quad_cov_cholesky(x, sigma,
//      length_scale, diagonal, pstream__))
namespace nico::math {

template <typename T_x, typename T_sigma, typename T_length_scale, typename T_diagonal,
          stan::require_all_stan_scalar_t<T_sigma, T_length_scale, T_diagonal>* = nullptr>
inline Eigen::Matrix<stan::return_type_t<T_sigma, T_length_scale, T_diagonal>, -1, -1>
nico_gp_exp_quad_cov_cholesky(const std::vector<T_x>& x, const T_sigma& sigma,
                              const T_length_scale& length_scale, const T_diagonal& diagonal,
                              std::ostream* /*pstream__*/) {

    static_assert(std::is_arithmetic<T_x>::value, "nico_gp_exp_quad_cov_cholesky: x must be data");
    namespace gp = nico::math::gp_exp_quad_cov;
    using stan::math::var;
    using stan::math::vari;
    using return_t = stan::return_type_t<T_sigma, T_length_scale, T_diagonal>;
    const std::vector<double> x_values(x.begin(), x.end());
    gp::check_arguments(x_values, sigma, length_scale);
    const int n = static_cast<int>(x.size());
    if (n == 0) return Eigen::Matrix<return_t, -1, -1>(0, 0);
    const double sigma_value    = stan::math::value_of(sigma);
    const double length_value   = stan::math::value_of(length_scale);
    const double diagonal_value = stan::math::value_of(diagonal);
    Eigen::Matrix<double, -1, -1> values = gp::covariance(x_values, x_values, true, sigma_value, length_value);
    // (add_diag's single addition, with its rounding: the diagonal term is not combined with sigma^2 beforehand)
    values.diagonal().array() += diagonal_value;
    if constexpr (!stan::is_var<return_t>::value) {
        // return stan::math::cholesky_decompose(values);
        return Eigen::Matrix<double, -1, -1>(nico_cholesky_decompose(values, nullptr));
    } else {
        auto& arena = stan::math::ChainableStack::instance_->memalloc_;
        double* x_on_arena = arena.alloc_array<double>(n);
        std::copy(x_values.begin(), x_values.end(), x_on_arena);
        vari* sigma_node    = nullptr;
        vari* length_node   = nullptr;
        vari* diagonal_node = nullptr;
        if constexpr (stan::is_var<T_sigma>::value)        sigma_node    = sigma.vi_;
        if constexpr (stan::is_var<T_length_scale>::value) length_node   = length_scale.vi_;
        if constexpr (stan::is_var<T_diagonal>::value)     diagonal_node = diagonal.vi_;
        stan::math::var_value<Eigen::Matrix<double, -1, -1>> covariance_node(values);
        // (registered before the factorisation: its reverse pass runs after the Cholesky reverse pass, which
        //  writes the adjoint of K's lower triangle; no factor of two on the entries below the diagonal)
        stan::math::reverse_pass_callback([covariance_node, x_on_arena, n, sigma_node, length_node,
                                           diagonal_node, sigma_value, length_value]() mutable {
            double sum_of_products = 0.0;
            double sum_of_products_times_squared_distances = 0.0;
            if (sigma_node != nullptr || length_node != nullptr) {
                nico::math::gp_exp_quad_cov_cholesky::sums_of_entries_times_adjoints_below_the_diagonal(
                        covariance_node.val().data(), covariance_node.adj().data(), x_on_arena, n,
                        sum_of_products, sum_of_products_times_squared_distances);
            }
            const double sum_of_diagonal_adjoints = covariance_node.adj().diagonal().sum();
            if (sigma_node != nullptr) {
                sum_of_products += sigma_value * sigma_value * sum_of_diagonal_adjoints;
                sigma_node->adj_ += sum_of_products * 2.0 / sigma_value;
            }
            if (length_node != nullptr) {
                length_node->adj_ += sum_of_products_times_squared_distances
                                     / (length_value * length_value * length_value);
            }
            if (diagonal_node != nullptr) diagonal_node->adj_ += sum_of_diagonal_adjoints;
        });
        // (BayesMVP's factorisation, then Stan's own Giles / Murray reverse pass, with its small / large
        //  dispatch)
        // stan::math::var_value<Eigen::Matrix<double, -1, -1>> lower
        //     = stan::math::cholesky_decompose(covariance_node);
        stan::math::var_value<Eigen::Matrix<double, -1, -1>> lower = nico_cholesky_decompose(covariance_node,
                                                                                             nullptr);
        stan::arena_t<Eigen::Matrix<var, -1, -1>> result(n, n);
        const std::ptrdiff_t count = static_cast<std::ptrdiff_t>(n) * (n + 1) / 2;
        vari* nodes = arena.alloc_array<vari>(count);
        vari* upper_zero = new vari(0.0, false);
        std::ptrdiff_t position = 0;
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < j; ++i) result(i, j) = var(upper_zero);
            for (int i = j; i < n; ++i, ++position) {
                ::new (static_cast<void*>(nodes + position)) vari(lower.val()(i, j), false);   // nochain stack
                result(i, j) = var(nodes + position);
            }
        }
        stan::math::reverse_pass_callback([result, lower]() mutable {
            lower.adj().template triangularView<Eigen::Lower>() += result.adj();
        });
        return Eigen::Matrix<var, -1, -1>(result);
    }

}

}  // namespace nico::math

using nico::math::nico_gp_exp_quad_cov_cholesky;






















