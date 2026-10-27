#pragma once

#include "stan_external_scalar_functions.hpp"
#include "stan_external_functions.hpp"

namespace nico::math {

// inline double nico_erfc_value(const double x) {
//
//     if (std::isnan(x)) return x;
//     if (std::isinf(x)) return x < 0.0 ? 2.0 : 0.0;
//
//     const double z = std::fabs(x) * std::sqrt(2.0);
//     double tail;
//     if (z > NICO_DOUBLE_PROBABILITY_NORMAL_TAIL_THRESH) {
//         const double scale = 2.0 * NICO_DOUBLE_PROBABILITY_INV_SQRT_2PI /
//             nico_inverse_mills_ratio_tail(z);
//         tail = scale * nico_exp_1(-0.5 * z * z);
//     } else {
//         tail = 2.0 * nico_Phi_wo_checks(-z);
//     }
//
//     return x < 0.0 ? 2.0 - tail : tail;
//
// }
//// (2 Phi(-x sqrt(2)) for every x: nico_Phi_wo_checks's rational holds on the whole range, no branch; NaN stays
////  NaN, -inf gives 2 and inf 0)
inline double nico_erfc_value(const double x) {

    return 2.0 * nico_Phi_wo_checks(-x * std::sqrt(2.0));

}

// ---- erfc(x) and exp(-x^2) from one exp: nico_Phi_wo_checks's rational for every z = |x| sqrt(2) (capped at
//      its z cap), scaled by exp(-z^2 / 2) = exp(-x^2); the var nico_erfc takes its partial, 2 / sqrt(pi)
//      exp(-x^2), from the same exp
inline double nico_erfc_value_and_exp_of_minus_x_squared(const double x,
                                                         double* exp_of_minus_x_squared) {

    if (std::isnan(x) || std::isinf(x)) {
        *exp_of_minus_x_squared = std::isnan(x) ? x : 0.0;
        return nico_erfc_value(x);
    }

    // const double z = std::fabs(x) * std::sqrt(2.0);
    // const double e = nico_exp_1(-0.5 * z * z);
    // *exp_of_minus_x_squared = e;
    // double tail;
    // if (z > NICO_DOUBLE_PROBABILITY_NORMAL_TAIL_THRESH) {
    //     tail = 2.0 * NICO_DOUBLE_PROBABILITY_INV_SQRT_2PI / nico_inverse_mills_ratio_tail(z) * e;
    // } else {
    //     double numerator = NICO_DOUBLE_PROBABILITY_FAST_PHI_MILLS_NUMERATOR_COEFFICIENTS[6];
    //     for (int k = 5; k >= 0; --k) {
    //         numerator = std::fma(numerator, z,
    //                              NICO_DOUBLE_PROBABILITY_FAST_PHI_MILLS_NUMERATOR_COEFFICIENTS[k]);
    //     }
    //     double denominator = NICO_DOUBLE_PROBABILITY_FAST_PHI_MILLS_DENOMINATOR_COEFFICIENTS[7];
    //     for (int k = 6; k >= 0; --k) {
    //         denominator = std::fma(denominator, z,
    //                                NICO_DOUBLE_PROBABILITY_FAST_PHI_MILLS_DENOMINATOR_COEFFICIENTS[k]);
    //     }
    //     tail = 2.0 * e * (numerator / denominator);
    // }
    //// (the rational for every z, its argument capped as in nico_Phi_wo_checks; no branch)
    const double z = std::fabs(x) * std::sqrt(2.0);
    const double e = nico_exp_1(-0.5 * (z * z));
    *exp_of_minus_x_squared = e;
    const double z_rational = (z < NICO_DOUBLE_PROBABILITY_FAST_PHI_MILLS_Z_CAP)
        ? z
        : NICO_DOUBLE_PROBABILITY_FAST_PHI_MILLS_Z_CAP;
    //// (Horner written out, as in nico_Phi_wo_checks)
    double numerator = NICO_DOUBLE_PROBABILITY_FAST_PHI_MILLS_NUMERATOR_COEFFICIENTS[6];
    numerator = std::fma(numerator, z_rational, NICO_DOUBLE_PROBABILITY_FAST_PHI_MILLS_NUMERATOR_COEFFICIENTS[5]);
    numerator = std::fma(numerator, z_rational, NICO_DOUBLE_PROBABILITY_FAST_PHI_MILLS_NUMERATOR_COEFFICIENTS[4]);
    numerator = std::fma(numerator, z_rational, NICO_DOUBLE_PROBABILITY_FAST_PHI_MILLS_NUMERATOR_COEFFICIENTS[3]);
    numerator = std::fma(numerator, z_rational, NICO_DOUBLE_PROBABILITY_FAST_PHI_MILLS_NUMERATOR_COEFFICIENTS[2]);
    numerator = std::fma(numerator, z_rational, NICO_DOUBLE_PROBABILITY_FAST_PHI_MILLS_NUMERATOR_COEFFICIENTS[1]);
    numerator = std::fma(numerator, z_rational, NICO_DOUBLE_PROBABILITY_FAST_PHI_MILLS_NUMERATOR_COEFFICIENTS[0]);
    double denominator = NICO_DOUBLE_PROBABILITY_FAST_PHI_MILLS_DENOMINATOR_COEFFICIENTS[7];
    denominator = std::fma(denominator, z_rational,
                           NICO_DOUBLE_PROBABILITY_FAST_PHI_MILLS_DENOMINATOR_COEFFICIENTS[6]);
    denominator = std::fma(denominator, z_rational,
                           NICO_DOUBLE_PROBABILITY_FAST_PHI_MILLS_DENOMINATOR_COEFFICIENTS[5]);
    denominator = std::fma(denominator, z_rational,
                           NICO_DOUBLE_PROBABILITY_FAST_PHI_MILLS_DENOMINATOR_COEFFICIENTS[4]);
    denominator = std::fma(denominator, z_rational,
                           NICO_DOUBLE_PROBABILITY_FAST_PHI_MILLS_DENOMINATOR_COEFFICIENTS[3]);
    denominator = std::fma(denominator, z_rational,
                           NICO_DOUBLE_PROBABILITY_FAST_PHI_MILLS_DENOMINATOR_COEFFICIENTS[2]);
    denominator = std::fma(denominator, z_rational,
                           NICO_DOUBLE_PROBABILITY_FAST_PHI_MILLS_DENOMINATOR_COEFFICIENTS[1]);
    denominator = std::fma(denominator, z_rational,
                           NICO_DOUBLE_PROBABILITY_FAST_PHI_MILLS_DENOMINATOR_COEFFICIENTS[0]);
    const double tail = 2.0 * (e * (numerator / denominator));

    return x < 0.0 ? 2.0 - tail : tail;

}

inline double nico_erfc(const double x,
                        std::ostream* /*pstream__*/) {

    return nico_erfc_value(x);

}

inline stan::math::var nico_erfc(const stan::math::var& x,
                                 std::ostream* pstream__) {

    // const double value = nico_erfc_value(x.val());
    // const double partial = stan::math::TWO_OVER_SQRT_PI * nico_exp(-x.val() * x.val(), pstream__);
    double exp_of_minus_x_squared;
    const double value = nico_erfc_value_and_exp_of_minus_x_squared(x.val(), &exp_of_minus_x_squared);
    const double partial = stan::math::TWO_OVER_SQRT_PI * exp_of_minus_x_squared;
    return stan::math::make_callback_var(value, [x, partial](auto& result_node) mutable {

        x.adj() -= result_node.adj() * partial;

    });

}

inline Eigen::Matrix<double, Eigen::Dynamic, 1>
nico_erfc_values(const Eigen::Matrix<double, Eigen::Dynamic, 1>& x) {

    Eigen::Matrix<double, Eigen::Dynamic, 1> result(x.size());
    Eigen::Index i = 0;
    //// (BMVP_LANES values at a time through the SIMD nico_Phi_wo_checks kernel, the same rational and exp)
#if BMVP_LANES == 8
    const __m512d minus_sqrt_2 = _mm512_set1_pd(-std::sqrt(2.0));
    for (; i + 8 <= x.size(); i += 8) {
        const __m512d phi = nico_Phi_wo_checks_AVX512(_mm512_mul_pd(_mm512_loadu_pd(x.data() + i), minus_sqrt_2));
        _mm512_storeu_pd(result.data() + i, _mm512_add_pd(phi, phi));
    }
#elif BMVP_LANES == 4
    const __m256d minus_sqrt_2 = _mm256_set1_pd(-std::sqrt(2.0));
    for (; i + 4 <= x.size(); i += 4) {
        const __m256d phi = nico_Phi_wo_checks_AVX2(_mm256_mul_pd(_mm256_loadu_pd(x.data() + i), minus_sqrt_2));
        _mm256_storeu_pd(result.data() + i, _mm256_add_pd(phi, phi));
    }
#endif
    for (; i < x.size(); ++i) {
        result[i] = nico_erfc_value(x[i]);
    }
    return result;

}

template <typename T_x>
inline Eigen::Matrix<stan::return_type_t<T_x>, Eigen::Dynamic, 1>
nico_erfc(const Eigen::Matrix<T_x, Eigen::Dynamic, 1>& x,
          std::ostream* pstream__) {

    static_assert(std::is_arithmetic<T_x>::value || stan::is_var<T_x>::value,
                  "The vector must contain arithmetic or reverse-mode scalars.");
    const Eigen::Matrix<double, Eigen::Dynamic, 1> values = stan::math::value_of(x);
    // const Eigen::Matrix<double, Eigen::Dynamic, 1> result_values = nico_erfc_values(values);
    Eigen::Matrix<stan::return_type_t<T_x>, Eigen::Dynamic, 1> result(x.size());
    if constexpr (stan::is_constant_all<T_x>::value) {
        // result = result_values;
        result = nico_erfc_values(values);
    } else {
        stan::arena_t<Eigen::Matrix<stan::math::var, Eigen::Dynamic, 1>> arena_x = x;
        // stan::arena_t<Eigen::Matrix<double, Eigen::Dynamic, 1>> partials
        //     = -values.array().square().matrix();
        // partials = stan::math::TWO_OVER_SQRT_PI * nico_exp(Eigen::Matrix<double, Eigen::Dynamic, 1>(partials),
        //                                                   pstream__);
        //// (erfc and its partial from one exp per value)
        Eigen::Matrix<double, Eigen::Dynamic, 1> result_values(x.size());
        stan::arena_t<Eigen::Matrix<double, Eigen::Dynamic, 1>> partials(x.size());
        for (Eigen::Index i = 0; i < x.size(); ++i) {
            double exp_of_minus_x_squared;
            result_values[i] = nico_erfc_value_and_exp_of_minus_x_squared(values[i], &exp_of_minus_x_squared);
            partials[i] = stan::math::TWO_OVER_SQRT_PI * exp_of_minus_x_squared;
        }
        stan::arena_t<Eigen::Matrix<stan::math::var, Eigen::Dynamic, 1>> arena_result = result_values;
        stan::math::reverse_pass_callback([arena_x, arena_result, partials]() mutable {

            arena_x.adj().array() -= arena_result.adj().array() * partials.array();

        });
        result = arena_result;
    }
    return result;

}

}  // namespace nico::math

using nico::math::nico_erfc;

namespace nico::math {

template <typename T_x,
          stan::require_eigen_col_vector_t<T_x>* = nullptr>
inline auto nico_erfc(const T_x& x,
                      std::ostream* pstream__) {

    using scalar = typename T_x::Scalar;
    const Eigen::Matrix<scalar, Eigen::Dynamic, 1> evaluated = x;
    return nico_erfc(evaluated, pstream__);

}

}  // namespace nico::math

using nico::math::nico_erfc;
