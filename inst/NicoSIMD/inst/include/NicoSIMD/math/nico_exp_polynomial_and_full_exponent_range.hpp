#pragma once

#include "../nico_simd_intrinsics.hpp"

#include <limits>

/*
namespace nico::math {

//// The existing degree-13 exponential polynomial, evaluated in each SIMD lane after range reduction.
//// A __m256d register holds four doubles; a __m512d register holds eight doubles.
inline constexpr double nico_inverse_factorials_for_double_precision_exp[14] = {
    1.0 / 6227020800.0, 1.0 / 479001600.0, 1.0 / 39916800.0, 1.0 / 3628800.0, 1.0 / 362880.0,
    1.0 / 40320.0, 1.0 / 5040.0, 1.0 / 720.0, 1.0 / 120.0, 1.0 / 24.0,
    1.0 / 6.0, 1.0 / 2.0, 1.0, 1.0
};

//// Conservative representability bounds, not approximations to the actual exp overflow/underflow cutoffs.
//// If a < q = min_exponent - digits - 1 < 0, exp(a) < 2^a < 2^q, half the smallest subnormal.
//// If a > max_exponent > 0, exp(a) > 2^a, beyond binary64's finite range.
inline constexpr double nico_exp_certainly_rounds_to_zero_below =
    static_cast<double>(std::numeric_limits<double>::min_exponent - std::numeric_limits<double>::digits - 1);
inline constexpr double nico_exp_certainly_overflows_above =
    static_cast<double>(std::numeric_limits<double>::max_exponent);

#if defined(NICO_SIMD_AVX2_KERNELS)

//// The input contains exact integral doubles of magnitude below 2048. The 1.5 * 2^52 bias converts all
//// four lanes to signed int64 without AVX-512 instructions or scalar conversions, at round-to-nearest.
inline __m256i nico_integral_exp_exponents_to_int64_AVX2(const __m256d exponents) {
    const __m256d bias = _mm256_set1_pd(6755399441055744.0);
    return _mm256_sub_epi64(_mm256_castpd_si256(_mm256_add_pd(exponents, bias)),
                            _mm256_castpd_si256(bias));
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((always_inline))
#endif
inline __m256d nico_exp_in_double_precision_AVX2(const __m256d argument) {
    const __m256d in_range = _mm256_and_pd(
        _mm256_cmp_pd(argument, _mm256_set1_pd(nico_exp_certainly_rounds_to_zero_below), _CMP_GE_OQ),
        _mm256_cmp_pd(argument, _mm256_set1_pd(nico_exp_certainly_overflows_above), _CMP_LE_OQ));
    const __m256d finite_argument = _mm256_and_pd(argument, in_range);
    const __m256d exponent = _mm256_round_pd(
        _mm256_mul_pd(finite_argument, _mm256_set1_pd(1.442695040888963387)),
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256d remainder = _mm256_fmadd_pd(exponent, _mm256_set1_pd(-0.693145751953125), finite_argument);
    remainder = _mm256_fmadd_pd(exponent, _mm256_set1_pd(-1.42860682030941723212e-6), remainder);
    __m256d polynomial = _mm256_set1_pd(nico_inverse_factorials_for_double_precision_exp[0]);
    for (int degree = 1; degree < 14; ++degree) {
        polynomial = _mm256_fmadd_pd(polynomial, remainder,
                                     _mm256_set1_pd(nico_inverse_factorials_for_double_precision_exp[degree]));
    }
    const __m256i integer_exponent = nico_integral_exp_exponents_to_int64_AVX2(exponent);
    const __m256d ordinary_result = _mm256_castsi256_pd(_mm256_add_epi64(
        _mm256_castpd_si256(polynomial), _mm256_slli_epi64(integer_exponent, 52)));
    const __m256d absolute_argument = _mm256_andnot_pd(_mm256_set1_pd(-0.0), argument);
    const __m256d ordinary = _mm256_and_pd(
        _mm256_cmp_pd(absolute_argument, _mm256_set1_pd(708.4), _CMP_LT_OQ),
        _mm256_cmp_pd(ordinary_result, _mm256_set1_pd(std::numeric_limits<double>::min()), _CMP_GE_OQ));
    if (_mm256_movemask_pd(ordinary) == 0xf) return ordinary_result;

    //// Both powers of two are normal. Their first multiplication stays normal, so the final
    //// multiplication alone rounds a subnormal result or overflows; no exponent bits wrap around.
    const __m256d first_exponent = _mm256_round_pd(_mm256_mul_pd(exponent, _mm256_set1_pd(0.5)),
                                                   _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
    const __m256d second_exponent = _mm256_sub_pd(exponent, first_exponent);
    const __m256i first_bits = _mm256_add_epi64(nico_integral_exp_exponents_to_int64_AVX2(first_exponent),
                                                _mm256_set1_epi64x(1023));
    const __m256i second_bits = _mm256_add_epi64(nico_integral_exp_exponents_to_int64_AVX2(second_exponent),
                                                 _mm256_set1_epi64x(1023));
    const __m256d first_power = _mm256_castsi256_pd(_mm256_slli_epi64(first_bits, 52));
    const __m256d second_power = _mm256_castsi256_pd(_mm256_slli_epi64(second_bits, 52));
    __m256d scaled = _mm256_mul_pd(_mm256_mul_pd(polynomial, first_power), second_power);
    scaled = _mm256_and_pd(scaled, in_range);
    const __m256d overflow = _mm256_cmp_pd(argument,
        _mm256_set1_pd(nico_exp_certainly_overflows_above), _CMP_GT_OQ);
    scaled = _mm256_blendv_pd(scaled, _mm256_set1_pd(std::numeric_limits<double>::infinity()), overflow);
    const __m256d nan = _mm256_cmp_pd(argument, argument, _CMP_UNORD_Q);
    scaled = _mm256_blendv_pd(scaled, argument, nan);
    return _mm256_blendv_pd(scaled, ordinary_result, ordinary);
}

#endif

#if defined(__AVX512VL__) && defined(__AVX512F__) && defined(__AVX512DQ__)

inline __m512d nico_exp_in_double_precision_AVX512(const __m512d argument) {
    const __mmask8 in_range = _mm512_cmp_pd_mask(argument,
        _mm512_set1_pd(nico_exp_certainly_rounds_to_zero_below), _CMP_GE_OQ)
        & _mm512_cmp_pd_mask(argument, _mm512_set1_pd(nico_exp_certainly_overflows_above), _CMP_LE_OQ);
    const __m512d finite_argument = _mm512_maskz_mov_pd(in_range, argument);
    const __m512d exponent = _mm512_roundscale_pd(
        _mm512_mul_pd(finite_argument, _mm512_set1_pd(1.442695040888963387)),
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m512d remainder = _mm512_fmadd_pd(exponent, _mm512_set1_pd(-0.693145751953125), finite_argument);
    remainder = _mm512_fmadd_pd(exponent, _mm512_set1_pd(-1.42860682030941723212e-6), remainder);
    __m512d polynomial = _mm512_set1_pd(nico_inverse_factorials_for_double_precision_exp[0]);
    for (int degree = 1; degree < 14; ++degree) {
        polynomial = _mm512_fmadd_pd(polynomial, remainder,
                                     _mm512_set1_pd(nico_inverse_factorials_for_double_precision_exp[degree]));
    }
    __m512d result = _mm512_maskz_mov_pd(in_range, _mm512_scalef_pd(polynomial, exponent));
    const __mmask8 overflow = _mm512_cmp_pd_mask(argument,
        _mm512_set1_pd(nico_exp_certainly_overflows_above), _CMP_GT_OQ);
    result = _mm512_mask_mov_pd(result, overflow, _mm512_set1_pd(std::numeric_limits<double>::infinity()));
    const __mmask8 nan = _mm512_cmp_pd_mask(argument, argument, _CMP_UNORD_Q);
    return _mm512_mask_mov_pd(result, nan, argument);
}

#endif

}  // namespace nico::math
*/






















