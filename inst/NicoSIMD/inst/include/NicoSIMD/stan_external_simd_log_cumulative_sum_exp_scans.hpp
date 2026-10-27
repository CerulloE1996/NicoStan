#pragma once

#include "stan_external_simd_affine_recurrence_scans.hpp"
#include "nico_simd_exp_in_double_precision.hpp"

#include <algorithm>
#include <limits>

namespace nico::math::log_cumulative_sum_exp {

//// Masked packet transfers never read or write past a partial final register.
inline simd_t nico_load_prefix_packet(const double* source, const int count, const double padding) {

#if defined(NICO_SIMD_SCALAR_KERNELS)
    return (count > 0) ? source[0] : padding;
#elif BMVP_LANES == 8
    const __mmask8 mask = static_cast<__mmask8>((1u << count) - 1u);
    return _mm512_mask_loadu_pd(scan_set1(padding), mask, source);
#else
    const __m256i mask = _mm256_cmpgt_epi64(_mm256_set1_epi64x(count), _mm256_set_epi64x(3, 2, 1, 0));
    return _mm256_blendv_pd(scan_set1(padding), _mm256_maskload_pd(source, mask), _mm256_castsi256_pd(mask));
#endif

}

inline void nico_store_prefix_packet(double* target, const int count, const simd_t value) {

#if defined(NICO_SIMD_SCALAR_KERNELS)
    if (count > 0) target[0] = value;
#elif BMVP_LANES == 8
    _mm512_mask_storeu_pd(target, static_cast<__mmask8>((1u << count) - 1u), value);
#else
    const __m256i mask = _mm256_cmpgt_epi64(_mm256_set1_epi64x(count), _mm256_set_epi64x(3, 2, 1, 0));
    _mm256_maskstore_pd(target, mask, value);
#endif

}

//// Inputs are nonpositive. Squaring exp(a / 2) keeps the exponential kernel in its normal domain,
//// including when the final answer is subnormal. Let q = min_exponent - digits - 1; 2^q is half the
//// smallest positive subnormal. For a < q < 0, exp(a) < 2^a < 2^q, so round-to-nearest gives zero.
//// This conservative bound leaves the actual zero/subnormal boundary to the final SIMD multiplication.
inline simd_t nico_exp_of_nonpositive_prefix_argument(const simd_t argument) {

    constexpr double certainly_zero_below = static_cast<double>(std::numeric_limits<double>::min_exponent
                                                               - std::numeric_limits<double>::digits - 1);
#if defined(NICO_SIMD_SCALAR_KERNELS)
    if (std::isnan(argument)) return argument;
    if (!(argument >= certainly_zero_below)) return 0.0;
    const simd_t root = nico_simd_exp_in_double_precision(argument * 0.5);
    return root * root;
#elif BMVP_LANES == 8
    const __mmask8 represented = _mm512_cmp_pd_mask(argument, scan_set1(certainly_zero_below), _CMP_GE_OQ);
    const simd_t half = _mm512_maskz_mul_pd(represented, argument, scan_set1(0.5));
    const simd_t root = nico_simd_exp_in_double_precision(half);
    const simd_t value = _mm512_maskz_mul_pd(represented, root, root);
    const __mmask8 nan = _mm512_cmp_pd_mask(argument, argument, _CMP_UNORD_Q);
    return _mm512_mask_mov_pd(value, nan, argument);
#else
    const simd_t represented = _mm256_cmp_pd(argument, scan_set1(certainly_zero_below), _CMP_GE_OQ);
    const simd_t half = _mm256_and_pd(represented, _mm256_mul_pd(argument, scan_set1(0.5)));
    const simd_t root = nico_simd_exp_in_double_precision(half);
    const simd_t value = _mm256_and_pd(represented, _mm256_mul_pd(root, root));
    const simd_t nan = _mm256_cmp_pd(argument, argument, _CMP_UNORD_Q);
    return _mm256_blendv_pd(value, argument, nan);
#endif

}

inline void nico_exp_of_nonpositive_prefix_arguments(double* values, const int count) {

    for (int first = 0; first < count; first += BMVP_LANES) {
        const int lanes = std::min(BMVP_LANES, count - first);
        const simd_t argument = nico_load_prefix_packet(values + first, lanes, 0.0);
        nico_store_prefix_packet(values + first, lanes, nico_exp_of_nonpositive_prefix_argument(argument));
    }

}

//// Positive normal inputs are proved by the shifted-prefix path (at least 1e-280).
inline void nico_log_of_positive_prefix_sums(double* values, const int count) {

    for (int first = 0; first < count; first += BMVP_LANES) {
        const int lanes = std::min(BMVP_LANES, count - first);
        const simd_t argument = nico_load_prefix_packet(values + first, lanes, 1.0);
#if defined(NICO_SIMD_SCALAR_KERNELS)
        const simd_t value = std::log(argument);
#else
        const simd_t value = BMVP_KERNEL_NAME(nico_log_1_double_precision_wo_checks)(argument);
#endif
        nico_store_prefix_packet(values + first, lanes, value);
    }

}

//// exp(component_maximum - combined_maximum), with the empty (-infinity, zero) identity.
inline simd_t nico_scale_log_prefix_packet(const simd_t component, const simd_t combined) {

#if defined(NICO_SIMD_SCALAR_KERNELS)
    simd_t difference = component - combined;
    if (std::isnan(difference)) difference = -std::numeric_limits<double>::infinity();
#elif BMVP_LANES == 8
    simd_t difference = _mm512_sub_pd(component, combined);
    const __mmask8 empty = _mm512_cmp_pd_mask(difference, difference, _CMP_UNORD_Q);
    difference = _mm512_mask_mov_pd(difference, empty,
                                    scan_set1(-std::numeric_limits<double>::infinity()));
#else
    simd_t difference = _mm256_sub_pd(component, combined);
    const simd_t empty = _mm256_cmp_pd(difference, difference, _CMP_UNORD_Q);
    difference = _mm256_blendv_pd(difference, scan_set1(-std::numeric_limits<double>::infinity()), empty);
#endif
    return nico_exp_of_nonpositive_prefix_argument(difference);

}

//// A prefix is (maximum, scaled sum), not just a rounded log value. Retaining both is necessary for
//// gradients when two large log prefixes round to the same double even though their sums differ.
inline void nico_compose_log_prefix_packets(simd_t& maximum, simd_t& sum,
                                           const simd_t other_maximum, const simd_t other_sum) {

#if defined(NICO_SIMD_SCALAR_KERNELS)
    const simd_t combined = (maximum > other_maximum) ? maximum : other_maximum;
#elif BMVP_LANES == 8
    const simd_t combined = _mm512_max_pd(maximum, other_maximum);
#else
    const simd_t combined = _mm256_max_pd(maximum, other_maximum);
#endif
    sum = scan_fmadd(sum, nico_scale_log_prefix_packet(maximum, combined),
                     scan_mul(other_sum, nico_scale_log_prefix_packet(other_maximum, combined)));
    maximum = combined;

}

template <int distance>
inline simd_t nico_shift_log_prefix_packet(const simd_t value) {

    const simd_t shifted = shift_up<distance>(value);
    const simd_t identity = scan_set1(-std::numeric_limits<double>::infinity());
#if defined(NICO_SIMD_SCALAR_KERNELS)
    return identity;
#elif BMVP_LANES == 8
    return _mm512_mask_mov_pd(shifted, static_cast<__mmask8>((1u << distance) - 1u), identity);
#else
    return _mm256_blend_pd(shifted, identity, (1 << distance) - 1);
#endif

}

inline void nico_log_prefix_pairs(const double* input, const int count, const double offset,
                                  double* maxima_or_logs, double* scaled_sums) {

    simd_t carry_maximum = scan_set1(offset);
    simd_t carry_sum = scan_set1(offset == -std::numeric_limits<double>::infinity() ? 0.0 : 1.0);
    for (int first = 0; first < count; first += BMVP_LANES) {
        const int lanes = std::min(BMVP_LANES, count - first);
        simd_t maximum = nico_load_prefix_packet(input + first, lanes,
                                                 -std::numeric_limits<double>::infinity());
#if defined(NICO_SIMD_SCALAR_KERNELS)
        simd_t sum = (lanes > 0) ? 1.0 : 0.0;
#elif BMVP_LANES == 8
        simd_t sum = _mm512_maskz_mov_pd(static_cast<__mmask8>((1u << lanes) - 1u), scan_set1(1.0));
#else
        const __m256i valid = _mm256_cmpgt_epi64(_mm256_set1_epi64x(lanes), _mm256_set_epi64x(3, 2, 1, 0));
        simd_t sum = _mm256_and_pd(_mm256_castsi256_pd(valid), scan_set1(1.0));
#endif
        nico_compose_log_prefix_packets(maximum, sum, nico_shift_log_prefix_packet<1>(maximum),
                                        shift_up<1>(sum));
        nico_compose_log_prefix_packets(maximum, sum, nico_shift_log_prefix_packet<2>(maximum),
                                        shift_up<2>(sum));
#if BMVP_LANES == 8
        nico_compose_log_prefix_packets(maximum, sum, nico_shift_log_prefix_packet<4>(maximum),
                                        shift_up<4>(sum));
#endif
        nico_compose_log_prefix_packets(maximum, sum, carry_maximum, carry_sum);
        carry_maximum = broadcast_last_lane(maximum);
        carry_sum = broadcast_last_lane(sum);
        if (scaled_sums != nullptr) {
            nico_store_prefix_packet(maxima_or_logs + first, lanes, maximum);
            nico_store_prefix_packet(scaled_sums + first, lanes, sum);
        } else {
#if defined(NICO_SIMD_SCALAR_KERNELS)
            const simd_t value = maximum + std::log(sum);
#else
            const simd_t logarithm = BMVP_KERNEL_NAME(nico_log_1_double_precision_wo_checks)(sum);
#if BMVP_LANES == 8
            const simd_t value = _mm512_add_pd(maximum, logarithm);
#else
            const simd_t value = _mm256_add_pd(maximum, logarithm);
#endif
#endif
            nico_store_prefix_packet(maxima_or_logs + first, lanes, value);
        }
    }

}

inline void nico_log_prefix_scan(const double* input, const int count, const double offset, double* output) {

    nico_log_prefix_pairs(input, count, offset, output, nullptr);

}

//// Stan operands can share one adjoint address. Unique packets use one multiply-add; repeated addresses
//// use masked groups and SIMD sums before writing each address once. Scalar work is address/memory only.
inline void nico_add_prefix_adjoint_packet(double* const* targets, const int count,
                                          const simd_t share, const simd_t running) {

    bool repeated = false;
    alignas(64) double adjoints[BMVP_LANES] = {};
    for (int lane = 0; lane < count; ++lane) {
        adjoints[lane] = *targets[lane];
        for (int previous = 0; previous < lane; ++previous) {
            repeated = repeated || targets[previous] == targets[lane];
        }
    }
    if (!repeated) {
        simd_store(adjoints, scan_fmadd(share, running, simd_load(adjoints)));
        for (int lane = 0; lane < count; ++lane) *targets[lane] = adjoints[lane];
        return;
    }
    alignas(64) double contributions[BMVP_LANES] = {};
    simd_store(contributions, scan_mul(share, running));
    const simd_t one = scan_set1(1.0);
    const affine_scan_coefficients_t coefficients{one, one, one, one, one};
    for (int lane = 0; lane < count; ++lane) {
        bool already_written = false;
        for (int previous = 0; previous < lane; ++previous) {
            already_written = already_written || targets[previous] == targets[lane];
        }
        if (already_written) continue;
        alignas(64) double members[BMVP_LANES] = {};
        for (int member = 0; member < count; ++member) {
            if (targets[member] == targets[lane]) members[member] = contributions[member];
        }
        const simd_t sums = forward_scan_in_register(simd_load(members), coefficients);
        const simd_t updated = scan_fmadd(scan_set1(1.0), broadcast_last_lane(sums),
                                          scan_set1(*targets[lane]));
        simd_store(adjoints, updated);
        *targets[lane] = adjoints[0];
    }

}

//// Reuse the existing constant-coefficient SIMD scan; only the final packet needs masking.
inline void nico_sum_prefix_scan(double* values, const int count, const double offset) {

    const int full = (count / BMVP_LANES) * BMVP_LANES;
    const simd_t one = scan_set1(1.0);
    const affine_scan_coefficients_t coefficients{one, one, one, one, one};
    forward_affine_scan(values, full, offset, values, coefficients);
    if (full < count) {
        const int lanes = count - full;
        const simd_t carry = scan_set1(full == 0 ? offset : values[full - 1]);
        const simd_t input = nico_load_prefix_packet(values + full, lanes, 0.0);
        const simd_t output = scan_fmadd(coefficients.weights_of_left_carry, carry,
                                        forward_scan_in_register(input, coefficients));
        nico_store_prefix_packet(values + full, lanes, output);
    }

}

//// Compose (a, b) pairs for r[i] = b[i] + a[i] r[i + 1]. Coefficients vary by lane.
template <int distance>
inline void nico_compose_backward_prefix_packets(simd_t& coefficient, simd_t& value) {

    value = scan_fmadd(coefficient, shift_down<distance>(value), value);
    simd_t shifted = shift_down<distance>(coefficient);
#if defined(NICO_SIMD_SCALAR_KERNELS)
    shifted = 1.0;
#elif BMVP_LANES == 8
    constexpr unsigned high_lanes = 0xffu ^ ((1u << (BMVP_LANES - distance)) - 1u);
    shifted = _mm512_mask_mov_pd(shifted, static_cast<__mmask8>(high_lanes), scan_set1(1.0));
#else
    shifted = _mm256_blend_pd(shifted, scan_set1(1.0), 0xf ^ ((1 << (BMVP_LANES - distance)) - 1));
#endif
    coefficient = scan_mul(coefficient, shifted);

}

//// ratio has count - 1 entries; its final implicit coefficient is zero. In-place weights/output is safe.
inline double nico_backward_prefix_scan(const double* weights, const double* ratio, const int count,
                                        double* output) {

    if (count == 0) return 0.0;
    simd_t carry = scan_zero();
    const int last = ((count - 1) / BMVP_LANES) * BMVP_LANES;
    for (int first = last; first >= 0; first -= BMVP_LANES) {
        const int lanes = std::min(BMVP_LANES, count - first);
        const int coefficients = std::min(BMVP_LANES, count - 1 - first);
        simd_t a = nico_load_prefix_packet(ratio + first, coefficients, 0.0);
        simd_t b = nico_load_prefix_packet(weights + first, lanes, 0.0);
        nico_compose_backward_prefix_packets<1>(a, b);
        nico_compose_backward_prefix_packets<2>(a, b);
#if BMVP_LANES == 8
        nico_compose_backward_prefix_packets<4>(a, b);
#endif
        b = scan_fmadd(a, carry, b);
        nico_store_prefix_packet(output + first, lanes, b);
        carry = broadcast_first_lane(b);
    }
    return output[0];

}

}  // namespace nico::math::log_cumulative_sum_exp























