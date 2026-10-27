#pragma once

#include "stan_external_functions.hpp"

#include <stdexcept>

namespace nico::math::grouped_bernoulli {

inline void negative_abs(const double* x, int n, double* out) {

    int i = 0;
#if defined(NICO_SIMD_SCALAR_KERNELS)
    //// (one lane: the loop below takes every element)
#elif BMVP_LANES == 8
    const nico::math::simd_t zero = _mm512_setzero_pd();
    for (; i + BMVP_LANES <= n; i += BMVP_LANES) {
        const nico::math::simd_t values = nico::math::simd_loadu(x + i);
        nico::math::simd_storeu(out + i, _mm512_sub_pd(zero, _mm512_abs_pd(values)));
    }
#else
    const nico::math::simd_t zero = _mm256_setzero_pd();
    for (; i + BMVP_LANES <= n; i += BMVP_LANES) {
        const nico::math::simd_t values = nico::math::simd_loadu(x + i);
        nico::math::simd_storeu(out + i, _mm256_sub_pd(zero, _mm256_andnot_pd(
            _mm256_set1_pd(-0.0), values)));
    }
#endif
    for (; i < n; ++i) {
        out[i] = -std::fabs(x[i]);
    }

}

inline void derivatives(const double* x, const double* exp_neg_abs_x, const double* signs,
                        int n, double* out) {

    int i = 0;
#if defined(NICO_SIMD_SCALAR_KERNELS)
    //// (one lane: the loop below takes every element)
#elif BMVP_LANES == 8
    const nico::math::simd_t zero = _mm512_setzero_pd();
    const nico::math::simd_t one = _mm512_set1_pd(1.0);
    for (; i + BMVP_LANES <= n; i += BMVP_LANES) {
        const nico::math::simd_t x_values = nico::math::simd_loadu(x + i);
        const nico::math::simd_t r_values = nico::math::simd_loadu(exp_neg_abs_x + i);
        const nico::math::simd_t denominator = _mm512_add_pd(one, r_values);
        const __mmask8 nonnegative = _mm512_cmp_pd_mask(x_values, zero, _CMP_GE_OQ);
        const nico::math::simd_t numerator = _mm512_mask_blend_pd(nonnegative, r_values, one);
        const nico::math::simd_t fractions = _mm512_div_pd(numerator, denominator);
        nico::math::simd_storeu(out + i,
                                   _mm512_mul_pd(nico::math::simd_loadu(signs + i), fractions));
    }
#else
    const nico::math::simd_t zero = _mm256_setzero_pd();
    const nico::math::simd_t one = _mm256_set1_pd(1.0);
    for (; i + BMVP_LANES <= n; i += BMVP_LANES) {
        const nico::math::simd_t x_values = nico::math::simd_loadu(x + i);
        const nico::math::simd_t r_values = nico::math::simd_loadu(exp_neg_abs_x + i);
        const nico::math::simd_t denominator = _mm256_add_pd(one, r_values);
        const nico::math::simd_t nonnegative = _mm256_cmp_pd(x_values, zero, _CMP_GE_OQ);
        const nico::math::simd_t numerator = _mm256_blendv_pd(r_values, one, nonnegative);
        const nico::math::simd_t fractions = _mm256_div_pd(numerator, denominator);
        nico::math::simd_storeu(out + i,
                                   _mm256_mul_pd(nico::math::simd_loadu(signs + i), fractions));
    }
#endif
    for (; i < n; ++i) {
        const double denominator = 1.0 + exp_neg_abs_x[i];
        const double numerator = x[i] >= 0.0 ? 1.0 : exp_neg_abs_x[i];
        out[i] = signs[i] * numerator / denominator;
    }

}

inline double sum_log_probability(const double* x, double* log1p_exp_neg_abs_x, int n) {

    int i = 0;
#if defined(NICO_SIMD_SCALAR_KERNELS)
    //// (one lane: the loop below takes every element)
#elif BMVP_LANES == 8
    const nico::math::simd_t zero = _mm512_setzero_pd();
    for (; i + BMVP_LANES <= n; i += BMVP_LANES) {
        const nico::math::simd_t x_values = nico::math::simd_loadu(x + i);
        const nico::math::simd_t positive_x = _mm512_max_pd(x_values, zero);
        const nico::math::simd_t log_values = nico::math::simd_loadu(
            log1p_exp_neg_abs_x + i);
        nico::math::simd_storeu(log1p_exp_neg_abs_x + i,
                                   _mm512_sub_pd(zero, _mm512_add_pd(positive_x, log_values)));
    }
#else
    const nico::math::simd_t zero = _mm256_setzero_pd();
    for (; i + BMVP_LANES <= n; i += BMVP_LANES) {
        const nico::math::simd_t x_values = nico::math::simd_loadu(x + i);
        const nico::math::simd_t positive_x = _mm256_max_pd(x_values, zero);
        const nico::math::simd_t log_values = nico::math::simd_loadu(
            log1p_exp_neg_abs_x + i);
        nico::math::simd_storeu(log1p_exp_neg_abs_x + i,
                                   _mm256_sub_pd(zero, _mm256_add_pd(positive_x, log_values)));
    }
#endif

    double logp = 0.0;
    for (int j = 0; j < i; ++j) {
        logp += log1p_exp_neg_abs_x[j];
    }
    for (; i < n; ++i) {
        logp += -std::max(x[i], 0.0) - log1p_exp_neg_abs_x[i];
    }
    return logp;

}

template <typename T_y, typename T_group, typename T_x, typename T_beta>
inline stan::partials_return_t<T_x, T_beta> value_and_partials(
    const T_y& y_slice, int start, const T_group& group, const T_x& x, const T_beta& beta,
    Eigen::Matrix<double, -1, -1>* beta_partials) {

    using T_partials_return = stan::partials_return_t<T_x, T_beta>;

    const int n = static_cast<int>(y_slice.size());
    const int beta_rows = static_cast<int>(beta.rows());
    auto& buffers = nico::math::scratch();
    buffers.ensure(n);
    double* signed_eta = buffers.a.data();
    double* logp_values = buffers.b.data();
    double* eta_derivatives = buffers.c.data();

    stan::scalar_seq_view<T_x> x_vec(x);
    for (int i = 0; i < n; ++i) {
        const int observation = start - 1 + i;
        const int group_i = group[observation];
        const double x_i = static_cast<double>(x_vec.val(observation));
        const double intercept = stan::math::value_of(beta.coeff(group_i - 1, 0));
        const double slope = stan::math::value_of(beta.coeff(group_i - 1, 1));
        const double eta_i = intercept + slope * x_i;
        if (std::isnan(eta_i)) {
            throw std::domain_error("nico_bernoulli_logit_grouped_regression_lpmf");
        }
        const double sign = static_cast<double>(2 * y_slice[i] - 1);
        signed_eta[i] = -sign * eta_i;
        eta_derivatives[i] = sign;
    }

    negative_abs(signed_eta, n, logp_values);
    nico::math::k_exp(logp_values, n);
    if (beta_partials != nullptr) {
        derivatives(signed_eta, logp_values, eta_derivatives, n, eta_derivatives);
    }

    nico::math::k_log1p(logp_values, n);
    T_partials_return logp = sum_log_probability(signed_eta, logp_values, n);

    if (beta_partials != nullptr) {
        beta_partials->setZero(beta_rows, 2);
        for (int i = 0; i < n; ++i) {
            const int observation = start - 1 + i;
            const int group_i = group[observation] - 1;
            const double x_i = static_cast<double>(x_vec.val(observation));
            (*beta_partials)(group_i, 0) += eta_derivatives[i];
            (*beta_partials)(group_i, 1) += eta_derivatives[i] * x_i;
        }
    }

    return logp;

}

}  // namespace nico::math::grouped_bernoulli

namespace nico::math {

template <bool propto = false, typename T_y, typename T_start, typename T_end, typename T_group,
          typename T_x, typename T_beta,
          stan::require_all_t<
              stan::is_std_vector<T_y>,
              std::is_integral<std::decay_t<stan::value_type_t<T_y>>>,
              std::is_integral<std::decay_t<T_start>>,
              std::is_integral<std::decay_t<T_end>>,
              stan::is_std_vector<T_group>,
              std::is_integral<std::decay_t<stan::value_type_t<T_group>>>,
              stan::is_col_vector<T_x>,
              stan::is_vt_not_complex<T_x>,
              stan::is_constant_all<T_x>,
              stan::is_eigen_matrix_dynamic<T_beta>,
              stan::is_vt_not_complex<T_beta>>* = nullptr>
inline stan::return_type_t<stan::base_type_t<T_beta>>
nico_bernoulli_logit_grouped_regression_lpmf(
    const T_y& y_slice, const T_start& start, const T_end& end, const T_group& group,
    const T_x& x, const T_beta& beta, std::ostream* /*pstream__*/ = nullptr) {

    static constexpr const char* function = "nico_bernoulli_logit_grouped_regression_lpmf";
    const int start_i = static_cast<int>(start);
    const int end_i = static_cast<int>(end);
    const int n = static_cast<int>(y_slice.size());

    if (n == 0) {
        return 0.0;
    }
    if (start_i < 1 || end_i < start_i || end_i - start_i + 1 != n) {
        throw std::invalid_argument(function);
    }
    if (static_cast<int>(stan::math::size(group)) < end_i
        || static_cast<int>(stan::math::size(x)) < end_i) {

        throw std::invalid_argument(function);

    }
    if (beta.cols() != 2) {
        throw std::invalid_argument(function);
    }

    stan::math::check_bounded(function, "y", y_slice, 0, 1);
    for (int i = 0; i < n; ++i) {
        const int group_i = group[start_i - 1 + i];
        if (group_i < 1 || group_i > beta.rows()) {
            throw std::domain_error(function);
        }
    }

    if (!stan::math::include_summand<propto, T_beta>::value) {
        return 0.0;
    }

    using T_beta_ref = stan::ref_type_if_not_constant_t<T_beta>;
    T_beta_ref beta_ref = beta;
    Eigen::Matrix<double, -1, -1> beta_partials;
    Eigen::Matrix<double, -1, -1>* beta_partials_ptr = nullptr;
    if (!stan::is_constant_all<T_beta>::value) {
        beta_partials_ptr = &beta_partials;
    }

    const auto logp = nico::math::grouped_bernoulli::value_and_partials(
        y_slice, start_i, group, x, beta_ref, beta_partials_ptr);
    if (beta_partials_ptr == nullptr) {
        return logp;
    }

    auto ops_partials = stan::math::make_partials_propagator(beta_ref);
    stan::math::partials<0>(ops_partials) = beta_partials;
    return ops_partials.build(logp);

}

}  // namespace nico::math

using nico::math::nico_bernoulli_logit_grouped_regression_lpmf;































