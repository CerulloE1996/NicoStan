#pragma once

// SIMD scans of the affine recurrence x[i] = a x[i - 1] + u[i] (forward) and of its adjoint recurrence
// lambda[i] = a lambda[i + 1] + w[i] (backward), on BMVP_LANES consecutive elements per register:
//   - within one register, the scan is done in log2(BMVP_LANES) shift-and-multiply-add steps (shifts by 1, 2
//     and 4 lanes with the coefficients a, a^2 and a^4), so the lanes do not wait for one another;
//   - the carry from the previous register (the last lane of the forward scan, the first lane of the backward
//     scan) is then added with the weights a^1, ..., a^W (a^W, ..., a^1 backwards), one multiply-add per
//     register.
// So the only serial dependence is one multiply-add per register of W elements (W = 4 with AVX2, 8 with AVX-512),
// where an element-by-element loop has one per element. General: any AR(1) / state-space / exponential-smoothing
// recurrence, and the reverse pass of any of them. Uses the SIMD type and the load / store helpers of
// stan_external_functions.hpp.

#include "stan_external_functions.hpp"
#include <cmath>
#include <vector>

namespace nico::math {

#if defined(NICO_SIMD_SCALAR_KERNELS)
//// one lane: a shift brings in zero, a broadcast is the value itself
inline simd_t scan_set1(const double x) {

    return x;

}
inline simd_t scan_zero() {

    return 0.0;

}
inline simd_t scan_fmadd(const simd_t a,
                         const simd_t b,
                         const simd_t c) {

    return std::fma(a, b, c);

}
inline simd_t scan_mul(const simd_t a,
                       const simd_t b) {

    return a * b;

}
template <int k> inline simd_t shift_up(const simd_t /*v*/) {

    return 0.0;

}
template <int k> inline simd_t shift_down(const simd_t /*v*/) {

    return 0.0;

}
inline simd_t shift_up_one_with(const simd_t /*v*/,
                                const simd_t first) {

    return first;

}
inline simd_t broadcast_last_lane(const simd_t v) {

    return v;

}
inline simd_t broadcast_first_lane(const simd_t v) {

    return v;

}
#elif BMVP_LANES == 8
inline simd_t scan_set1(const double x)                                  { return _mm512_set1_pd(x); }
inline simd_t scan_zero()                                                { return _mm512_setzero_pd(); }
inline simd_t scan_fmadd(const simd_t a, const simd_t b, const simd_t c) { return _mm512_fmadd_pd(a, b, c); }
inline simd_t scan_mul(const simd_t a, const simd_t b)                   { return _mm512_mul_pd(a, b); }
template <int k> inline simd_t shift_up(const simd_t v) {         // lane i takes lane i - k, 0 below lane k

    return _mm512_castsi512_pd(_mm512_alignr_epi64(_mm512_castpd_si512(v), _mm512_setzero_si512(), 8 - k));

}
template <int k> inline simd_t shift_down(const simd_t v) {       // lane i takes lane i + k, 0 above lane 7 - k

    return _mm512_castsi512_pd(_mm512_alignr_epi64(_mm512_setzero_si512(), _mm512_castpd_si512(v), k));

}
inline simd_t shift_up_one_with(const simd_t v, const simd_t first) {   // lane 0 takes lane 7 of first

    return _mm512_castsi512_pd(_mm512_alignr_epi64(_mm512_castpd_si512(v), _mm512_castpd_si512(first), 7));

}
inline simd_t broadcast_last_lane(const simd_t v)  { return _mm512_permutexvar_pd(_mm512_set1_epi64(7), v); }
inline simd_t broadcast_first_lane(const simd_t v) { return _mm512_permutexvar_pd(_mm512_setzero_si512(), v); }
#else
inline simd_t scan_set1(const double x)                                  { return _mm256_set1_pd(x); }
inline simd_t scan_zero()                                                { return _mm256_setzero_pd(); }
inline simd_t scan_fmadd(const simd_t a, const simd_t b, const simd_t c) { return _mm256_fmadd_pd(a, b, c); }
inline simd_t scan_mul(const simd_t a, const simd_t b)                   { return _mm256_mul_pd(a, b); }
template <int k> inline simd_t shift_up(const simd_t v) {

    if constexpr (k == 1) {
        return _mm256_blend_pd(_mm256_permute4x64_pd(v, _MM_SHUFFLE(2, 1, 0, 0)), _mm256_setzero_pd(), 0x1);
    } else {
        return _mm256_blend_pd(_mm256_permute4x64_pd(v, _MM_SHUFFLE(1, 0, 0, 0)), _mm256_setzero_pd(), 0x3);
    }

}
template <int k> inline simd_t shift_down(const simd_t v) {

    if constexpr (k == 1) {
        return _mm256_blend_pd(_mm256_permute4x64_pd(v, _MM_SHUFFLE(3, 3, 2, 1)), _mm256_setzero_pd(), 0x8);
    } else {
        return _mm256_blend_pd(_mm256_permute4x64_pd(v, _MM_SHUFFLE(3, 3, 3, 2)), _mm256_setzero_pd(), 0xC);
    }

}
inline simd_t shift_up_one_with(const simd_t v, const simd_t first) {

    return _mm256_blend_pd(_mm256_permute4x64_pd(v, _MM_SHUFFLE(2, 1, 0, 0)), first, 0x1);

}
inline simd_t broadcast_last_lane(const simd_t v)  { return _mm256_permute4x64_pd(v, 0xFF); }
inline simd_t broadcast_first_lane(const simd_t v) { return _mm256_permute4x64_pd(v, 0x00); }
#endif

//// ---- the coefficients of the scans of one coefficient a
struct affine_scan_coefficients_t {
    simd_t a_1, a_2, a_4;           // a, a^2 and a^4 in every lane (the in-register steps)
    simd_t weights_of_left_carry;   // lane k: a^(k + 1)
    simd_t weights_of_right_carry;  // lane k: a^(W - k)
};
inline affine_scan_coefficients_t affine_scan_coefficients(const double a) {

    alignas(64) double left[BMVP_LANES];
    alignas(64) double right[BMVP_LANES];
    double power = 1.0;
    for (int k = 0; k < BMVP_LANES; ++k) {
        power *= a;
        left[k] = power;
        right[BMVP_LANES - 1 - k] = power;
    }
    affine_scan_coefficients_t c;
    c.a_1 = scan_set1(a);
    c.a_2 = scan_set1(a * a);
    c.a_4 = scan_set1(a * a * a * a);
    c.weights_of_left_carry = simd_load(left);
    c.weights_of_right_carry = simd_load(right);
    return c;

}

//// one register of the forward scan, from zero: x[k] = a x[k - 1] + u[k] within the register
inline simd_t forward_scan_in_register(simd_t u, const affine_scan_coefficients_t& c) {

    u = scan_fmadd(c.a_1, shift_up<1>(u), u);
    u = scan_fmadd(c.a_2, shift_up<2>(u), u);
#if BMVP_LANES == 8
    u = scan_fmadd(c.a_4, shift_up<4>(u), u);
#endif
    return u;

}
//// one register of the backward scan, from zero: lambda[k] = a lambda[k + 1] + w[k] within the register
inline simd_t backward_scan_in_register(simd_t w, const affine_scan_coefficients_t& c) {

    w = scan_fmadd(c.a_1, shift_down<1>(w), w);
    w = scan_fmadd(c.a_2, shift_down<2>(w), w);
#if BMVP_LANES == 8
    w = scan_fmadd(c.a_4, shift_down<4>(w), w);
#endif
    return w;

}

//// ---- x[i] = a x[i - 1] + u[i], x[-1] = start, for i < n_padded (a multiple of BMVP_LANES; u zero-padded)
inline void forward_affine_scan(const double* u, const Eigen::Index n_padded, const double start, double* x,
                                const affine_scan_coefficients_t& c) {

    simd_t carry = scan_set1(start);
    for (Eigen::Index t = 0; t < n_padded; t += BMVP_LANES) {
        const simd_t v = scan_fmadd(c.weights_of_left_carry, carry,
                                    forward_scan_in_register(simd_loadu(u + t), c));
        simd_storeu(x + t, v);
        carry = broadcast_last_lane(v);
    }

}
//// ---- lambda[i] = a lambda[i + 1] + w[i], lambda[n_padded] = end, for i < n_padded (w zero-padded)
inline void backward_affine_scan(const double* w, const Eigen::Index n_padded, const double end, double* lambda,
                                 const affine_scan_coefficients_t& c) {

    simd_t carry = scan_set1(end);
    for (Eigen::Index t = n_padded - BMVP_LANES; t >= 0; t -= BMVP_LANES) {
        const simd_t v = scan_fmadd(c.weights_of_right_carry, carry,
                                    backward_scan_in_register(simd_loadu(w + t), c));
        simd_storeu(lambda + t, v);
        carry = broadcast_first_lane(v);
    }

}

//// the length rounded up to a whole number of registers
inline Eigen::Index padded_to_lanes(const Eigen::Index n) {

    return ((n + BMVP_LANES - 1) / BMVP_LANES) * BMVP_LANES;

}

//// ---- s[i] = a s[i - 1] + c[i] z[i] (c[0] = first_scale, c[i] = scale for i >= 1), s[-1] = initial_value, at
////      the given positions (1-based, strictly increasing, the last at most length): its values and, when
////      with_derivatives, the derivatives of each value with respect to (initial_value, coefficient, scale,
////      first_scale), one column per position. The innovations z[0 .. length - 1] are read in place from memory,
////      a register at a time; only the last, partial register is copied, zero-padded, into a register-sized
////      array. One pass over the registers: the recurrence and its two derivatives, each a scan with its own
////      carry. The forward part of NicoStan's nico_affine_recurrence_at_positions_of_the_nuisance_block, moved
////      here unchanged, because every SIMD function lives in NicoSIMD; it knows nothing
////      of NicoStan: any AR(1) / state-space model whose innovations are in memory can call it.
inline void nico_forward_affine_recurrence_at_positions_from_innovations_in_memory(
        const double* innovations, const int length, const std::vector<int>& positions, const double coefficient,
        const double scale, const double first_scale, const double initial_value, const bool with_derivatives,
        Eigen::Matrix<double, -1, 1>& values, Eigen::Matrix<double, -1, -1>& derivatives) {

    const double* z = innovations;
    const double a = coefficient;
    const double b = scale;
    const double b_first = first_scale;
    const double start = initial_value;
    const Eigen::Index count = positions.size();
    const Eigen::Index n_padded = padded_to_lanes(length);
    alignas(64) double last_partial_register[BMVP_LANES];
    const Eigen::Index start_of_the_last_partial_register = (length / BMVP_LANES) * BMVP_LANES;
    for (int lane = 0; lane < BMVP_LANES; ++lane) {
        const Eigen::Index t_of_lane = start_of_the_last_partial_register + lane;
        last_partial_register[lane] = (t_of_lane < length) ? z[t_of_lane] : 0.0;
    }
    const affine_scan_coefficients_t c = affine_scan_coefficients(a);
    const simd_t scale_in_lanes = scan_set1(b);
    values.resize(count);
    if (with_derivatives) derivatives.resize(4, count);
    simd_t carry_of_state = scan_set1(start);
    simd_t carry_of_d_a = scan_zero();
    simd_t carry_of_d_scale = scan_zero();
    alignas(64) double lanes_of_state[BMVP_LANES];
    alignas(64) double lanes_of_d_a[BMVP_LANES];
    alignas(64) double lanes_of_d_scale[BMVP_LANES];
    alignas(64) double first_lanes[BMVP_LANES];
    Eigen::Index next = 0;
    for (Eigen::Index t = 0; t < n_padded && next < count; t += BMVP_LANES) {
        const simd_t z_lanes = (t + BMVP_LANES <= length) ? simd_loadu(z + t) : simd_load(last_partial_register);
        simd_t u = scan_mul(scale_in_lanes, z_lanes);
        simd_t z_after_the_first = z_lanes;
        if (t == 0) {
            //// (the first innovation has its own scale, and no part in the derivative with respect to scale)
            simd_store(first_lanes, u);
            first_lanes[0] = b_first * z[0];
            u = simd_load(first_lanes);
            simd_store(first_lanes, z_lanes);
            first_lanes[0] = 0.0;
            z_after_the_first = simd_load(first_lanes);
        }
        const simd_t state = scan_fmadd(c.weights_of_left_carry, carry_of_state, forward_scan_in_register(u, c));
        simd_t d_a = state;
        simd_t d_scale = state;
        if (with_derivatives) {
            const simd_t previous_state = shift_up_one_with(state, carry_of_state);
            d_a = scan_fmadd(c.weights_of_left_carry, carry_of_d_a, forward_scan_in_register(previous_state, c));
            d_scale = scan_fmadd(c.weights_of_left_carry, carry_of_d_scale,
                                 forward_scan_in_register(z_after_the_first, c));
            carry_of_d_a = broadcast_last_lane(d_a);
            carry_of_d_scale = broadcast_last_lane(d_scale);
        }
        carry_of_state = broadcast_last_lane(state);
        if (positions[next] - 1 < t + BMVP_LANES) {
            simd_store(lanes_of_state, state);
            if (with_derivatives) {
                simd_store(lanes_of_d_a, d_a);
                simd_store(lanes_of_d_scale, d_scale);
            }
            while (next < count && positions[next] - 1 < t + BMVP_LANES) {
                const int lane = static_cast<int>(positions[next] - 1 - t);
                values[next] = lanes_of_state[lane];
                if (with_derivatives) {
                    const double power = std::pow(a, positions[next] - 1);
                    derivatives.col(next) << power * a, lanes_of_d_a[lane], lanes_of_d_scale[lane], power * z[0];
                }
                ++next;
            }
        }
    }

}

//// ---- The reverse pass of the function above: the adjoints of its values at the positions enter one backward
////      scan over the registers as impulses, and the adjoints of the innovations z[0 .. length - 1] are added
////      into gradient_of_the_innovations[0 .. length - 1], a register at a time (the last, partial register lane
////      by lane). The reverse part of NicoStan's nico_affine_recurrence_at_positions_of_the_nuisance_block,
//// moved here unchanged; it knows nothing of NicoStan.
inline void nico_add_the_adjoints_of_the_innovations_of_an_affine_recurrence_at_positions(
        const int* positions, const double* adjoints_of_the_values, const Eigen::Index count,
        const double coefficient, const double scale, const double first_scale, const int length,
        double* gradient_of_the_innovations) {

    const double a = coefficient;
    const double b = scale;
    const double b_first = first_scale;
    const Eigen::Index n_padded = padded_to_lanes(length);
    const affine_scan_coefficients_t c = affine_scan_coefficients(a);
    const simd_t scale_in_lanes = scan_set1(b);
    simd_t carry = scan_zero();
    alignas(64) double lanes[BMVP_LANES];
    Eigen::Index next = count - 1;
    for (Eigen::Index t = n_padded - BMVP_LANES; t >= 0; t -= BMVP_LANES) {
        simd_store(lanes, scan_zero());
        while (next >= 0 && positions[next] - 1 >= t) {
            lanes[positions[next] - 1 - t] += adjoints_of_the_values[next];
            --next;
        }
        const simd_t lambda = scan_fmadd(c.weights_of_right_carry, carry,
                                         backward_scan_in_register(simd_load(lanes), c));
        carry = broadcast_first_lane(lambda);
        simd_t adjoints_of_innovations = scan_mul(scale_in_lanes, lambda);
        if (t == 0) {
            //// (the first innovation has its own scale)
            alignas(64) double lanes_of_lambda[BMVP_LANES];
            simd_store(lanes_of_lambda, lambda);
            simd_store(lanes, adjoints_of_innovations);
            lanes[0] = b_first * lanes_of_lambda[0];
            adjoints_of_innovations = simd_load(lanes);
        }
        if (t + BMVP_LANES <= length) {
            double* target = gradient_of_the_innovations + t;
            const simd_t sum = scan_fmadd(scan_set1(1.0), adjoints_of_innovations, simd_loadu(target));
            simd_storeu(target, sum);
        } else {
            simd_store(lanes, adjoints_of_innovations);
            for (Eigen::Index lane = 0; t + lane < length; ++lane) {
                gradient_of_the_innovations[t + lane] += lanes[lane];
            }
        }
    }

}

}  // namespace nico::math






















