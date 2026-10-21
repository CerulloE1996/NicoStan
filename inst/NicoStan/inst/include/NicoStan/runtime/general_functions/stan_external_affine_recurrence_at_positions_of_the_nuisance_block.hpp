#pragma once

// nico_affine_recurrence_at_positions_of_the_nuisance_block(positions, initial_value, coefficient, scale,
//                                                           first_scale):
// BayesMVP's nico_affine_recurrence_at_positions with NicoStan's nuisance block as its innovations (a model in
// NicoStan's nuisance-block form, e.g. the boundary pass of any AR / state-space model whose latent block is
// its innovations): the values of
//     s[i] = a s[i - 1] + c[i] z[i]  (c[0] = first_scale, c[i] = scale), s[-1] = initial_value,
// at the given positions, with z the nuisance values read straight from NicoStan's array (no node and no copy on
// the main tape for them), and, in the reverse pass, their adjoints added straight into NicoStan's array of the
// nuisance gradient (SIMD, a register at a time). Outside NicoStan's gradient call the values set for use
// outside NicoStan are read and their gradient is not kept. The same SIMD scans as BayesMVP's function
// (include after BayesMVP's stan_external_simd_affine_recurrence_scans.hpp). Declared in the Stan functions
// block (no body):
//     vector nico_affine_recurrence_at_positions_of_the_nuisance_block(array[] int positions, real initial_value,
//                                                                      real coefficient, real scale,
//                                                                      real first_scale);

//// (its two SIMD loops, forward and reverse, are NicoSIMD's, in its
////  stan_external_simd_affine_recurrence_scans.hpp:
////  nico_forward_affine_recurrence_at_positions_from_innovations_in_memory and
////  nico_add_the_adjoints_of_the_innovations_of_an_affine_recurrence_at_positions;
////  this file keeps NicoStan's part: the nuisance block, the checks and the nodes)
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace nico::math {

inline std::vector<double>& buffer_of_the_nuisance_block_of_affine_recurrence() {

    static thread_local std::vector<double> values;
    return values;

}

}  // namespace nico::math

template <typename T_initial, typename T_coefficient, typename T_scale, typename T_first_scale>
inline auto nico_affine_recurrence_at_positions_of_the_nuisance_block(
        const std::vector<int>& positions, const T_initial& initial_value, const T_coefficient& coefficient,
        const T_scale& scale, const T_first_scale& first_scale, std::ostream* /*pstream__*/ = nullptr) {

    using scalar = stan::return_type_t<T_initial, T_coefficient, T_scale, T_first_scale>;
    using nico::math::simd_t;
    static const char* function = "nico_affine_recurrence_at_positions_of_the_nuisance_block";
    //// ---- the nuisance values: NicoStan's block inside its gradient call, otherwise the values set for use
    ////      outside NicoStan
    stan::math::internal::nicostan_nuisance_block_t* block = stan::math::internal::nicostan_nuisance_block;
    const std::vector<double>& values_outside =
          stan::math::internal::nicostan_nuisance_values_used_outside_NicoStan;
    const double* nuisance_values = (block != nullptr) ? block->values : values_outside.data();
    const int size_of_the_block = (block != nullptr) ? block->size : static_cast<int>(values_outside.size());
    double* nuisance_gradient = (block != nullptr) ? block->gradient : nullptr;
    const Eigen::Index count = positions.size();
    const double a = stan::math::value_of(coefficient);
    const double b = stan::math::value_of(scale);
    const double b_first = stan::math::value_of(first_scale);
    const double start = stan::math::value_of(initial_value);
    stan::math::check_finite(function, "coefficient", a);
    stan::math::check_finite(function, "scale", b);
    stan::math::check_finite(function, "first innovation scale", b_first);
    stan::math::check_finite(function, "initial value", start);
    int previous_position = 0;
    for (int position : positions) {
        if (position <= previous_position || position > size_of_the_block) {
            throw std::invalid_argument("Recurrence positions must increase strictly within the nuisance "
                                        "block.");
        }
        previous_position = position;
    }
    const int length = count > 0 ? positions.back() : 0;
//    const Eigen::Index n_padded = nico::math::padded_to_lanes(length);
    //// (n_padded is now formed inside NicoSIMD's two functions below)
#ifdef NICOSTAN_CHUNK_LOCAL_LEAPFROG_STEP_INSIDE_THE_MODEL_LIBRARY
    //// (a model library with the chunk-local leapfrog step, inside the gradient call of a leapfrog step: the
    ////  reverse pass below adds the adjoints of the innovations 0 to length - 1 straight into NicoStan's
    ////  nuisance gradient, after every chunk has added its own, with no node on the main tape that would mark
    ////  these values as read outside their own chunk; hence they are kept as read outside their own chunk here,
    ////  so that NicoStan makes their step again from their complete gradient after the gradient call, and makes
    ////  it after the chunks in the next gradient call)
    stan::math::internal::nicostan_chunk_local_leapfrog_step_in_the_model_t* chunk_local_step =
          stan::math::internal::nicostan_chunk_local_leapfrog_step;
    if ((block != nullptr) && (chunk_local_step != nullptr) && (length > 0)) {
        std::vector<int>& values_read_outside = chunk_local_step->values_read_outside_their_own_chunk;
        const std::size_t size_before = values_read_outside.size();
        values_read_outside.resize(size_before + static_cast<std::size_t>(length));
        std::iota(values_read_outside.begin() + size_before, values_read_outside.end(), 0);
    }
#endif

    //// ---- the innovations' values, zero-padded to whole registers
    // std::vector<double>& buffer = nico::math::buffer_of_the_nuisance_block_of_affine_recurrence();
    // if (static_cast<Eigen::Index>(buffer.size()) < n_padded) buffer.resize(n_padded);
    // double* z = buffer.data();
    // Eigen::Map<Eigen::Array<double, -1, 1>>(z, length) = Eigen::Map<const Eigen::Array<double, -1,
    // 1>>(nuisance_values, length);
    // Eigen::Map<Eigen::Array<double, -1, 1>>(z + length, n_padded - length).setZero();
    // if (!Eigen::Map<const Eigen::Array<double, -1, 1>>(z, length).allFinite()) {
    // stan::math::check_finite(function, "innovation", Eigen::Map<const Eigen::Matrix<double, -1, 1>>(z,
    // length));
    // }
    //// (the innovations are read in place from NicoStan's array, a register at a time; only the last, partial
    ////  register is copied, zero-padded, into a register-sized array, so that no copy of the whole block is
    ////  written and read again on every gradient)
    const double* z = nuisance_values;
//    alignas(64) double last_partial_register[BMVP_LANES];
//    const Eigen::Index start_of_the_last_partial_register = (length / BMVP_LANES) * BMVP_LANES;
//    for (int lane = 0; lane < BMVP_LANES; ++lane) {
//        const Eigen::Index t_of_lane = start_of_the_last_partial_register + lane;
//        last_partial_register[lane] = (t_of_lane < length) ? z[t_of_lane] : 0.0;
//    }
    if (!Eigen::Map<const Eigen::Array<double, -1, 1>>(z, length).allFinite()) {
        stan::math::check_finite(function, "innovation", Eigen::Map<const Eigen::Matrix<double, -1, 1>>(z,
                                                                                                        length));
    }

    //// ---- one pass over the registers: the recurrence and its derivatives, each a scan with its own carry
    constexpr bool need_derivatives = !stan::is_constant_all<T_initial, T_coefficient, T_scale,
                                                             T_first_scale>::value;
//    const nico::math::affine_scan_coefficients_t c = nico::math::affine_scan_coefficients(a);
//    const simd_t scale_in_lanes = nico::math::scan_set1(b);
//    Eigen::Matrix<double, -1, 1> values(count);
//    Eigen::Matrix<double, -1, -1> derivatives;
//    if constexpr (need_derivatives) derivatives.resize(4, count);
//    simd_t carry_of_state = nico::math::scan_set1(start);
//    simd_t carry_of_d_a = nico::math::scan_zero();
//    simd_t carry_of_d_scale = nico::math::scan_zero();
//    alignas(64) double lanes_of_state[BMVP_LANES];
//    alignas(64) double lanes_of_d_a[BMVP_LANES];
//    alignas(64) double lanes_of_d_scale[BMVP_LANES];
//    alignas(64) double first_lanes[BMVP_LANES];
//    Eigen::Index next = 0;
//    for (Eigen::Index t = 0; t < n_padded && next < count; t += BMVP_LANES) {
//        // const simd_t z_lanes = nico::math::simd_loadu(z + t);
//        const simd_t z_lanes = (t + BMVP_LANES <= length) ? nico::math::simd_loadu(z + t)
//                                                          : nico::math::simd_load(last_partial_register);
//        simd_t u = nico::math::scan_mul(scale_in_lanes, z_lanes);
//        simd_t z_after_the_first = z_lanes;
//        if (t == 0) {
//            //// (the first innovation has its own scale, and no part in the derivative with respect to scale)
//            nico::math::simd_store(first_lanes, u);
//            first_lanes[0] = b_first * z[0];
//            u = nico::math::simd_load(first_lanes);
//            nico::math::simd_store(first_lanes, z_lanes);
//            first_lanes[0] = 0.0;
//            z_after_the_first = nico::math::simd_load(first_lanes);
//        }
//        const simd_t state = nico::math::scan_fmadd(c.weights_of_left_carry, carry_of_state,
//                                                       nico::math::forward_scan_in_register(u, c));
//        simd_t d_a = state;
//        simd_t d_scale = state;
//        if constexpr (need_derivatives) {
//            const simd_t previous_state = nico::math::shift_up_one_with(state, carry_of_state);
//            d_a = nico::math::scan_fmadd(c.weights_of_left_carry, carry_of_d_a,
//                                            nico::math::forward_scan_in_register(previous_state, c));
//            d_scale = nico::math::scan_fmadd(c.weights_of_left_carry, carry_of_d_scale,
//                                                nico::math::forward_scan_in_register(z_after_the_first, c));
//            carry_of_d_a = nico::math::broadcast_last_lane(d_a);
//            carry_of_d_scale = nico::math::broadcast_last_lane(d_scale);
//        }
//        carry_of_state = nico::math::broadcast_last_lane(state);
//        if (positions[next] - 1 < t + BMVP_LANES) {
//            nico::math::simd_store(lanes_of_state, state);
//            if constexpr (need_derivatives) {
//                nico::math::simd_store(lanes_of_d_a, d_a);
//                nico::math::simd_store(lanes_of_d_scale, d_scale);
//            }
//            while (next < count && positions[next] - 1 < t + BMVP_LANES) {
//                const int lane = static_cast<int>(positions[next] - 1 - t);
//                values[next] = lanes_of_state[lane];
//                if constexpr (need_derivatives) {
//                    const double power = std::pow(a, positions[next] - 1);
//                    derivatives.col(next) << power * a, lanes_of_d_a[lane], lanes_of_d_scale[lane],
//                                             power * z[0];
//                }
//                ++next;
//            }
//        }
//    }
    //// (the recurrence and its derivatives, one pass over the registers: NicoSIMD's, the same code moved there
    Eigen::Matrix<double, -1, 1> values(count);
    Eigen::Matrix<double, -1, -1> derivatives;
    nico::math::nico_forward_affine_recurrence_at_positions_from_innovations_in_memory(
          z, length, positions, a, b, b_first, start, need_derivatives, values, derivatives);
    if constexpr (!need_derivatives) {
        return values;
    } else {
        Eigen::Matrix<scalar, Eigen::Dynamic, 1> result(count);
        if (count == 0) return result;
        using stan::math::vari;
        auto& arena = stan::math::ChainableStack::instance_->memalloc_;
        int* selected_positions = arena.alloc_array<int>(count);
        std::copy(positions.begin(), positions.end(), selected_positions);
        stan::arena_t<Eigen::Matrix<double, -1, -1>> saved_derivatives = derivatives;
        vari* result_nodes = arena.alloc_array<vari>(count);
        for (Eigen::Index i = 0; i < count; ++i) {
            ::new (static_cast<void*>(result_nodes + i)) vari(values[i], false);
            result[i] = stan::math::var(result_nodes + i);
        }
        vari* initial_parent = nullptr;
        vari* a_parent = nullptr;
        vari* b_parent = nullptr;
        vari* first_scale_parent = nullptr;
        if constexpr (!stan::is_constant_all<T_initial>::value) initial_parent = initial_value.vi_;
        if constexpr (!stan::is_constant_all<T_coefficient>::value) a_parent = coefficient.vi_;
        if constexpr (!stan::is_constant_all<T_scale>::value) b_parent = scale.vi_;
        if constexpr (!stan::is_constant_all<T_first_scale>::value) first_scale_parent = first_scale.vi_;
        stan::math::reverse_pass_callback([nuisance_gradient, result_nodes, selected_positions,
            saved_derivatives,
//            initial_parent, a_parent, b_parent, first_scale_parent, a, b, b_first, length, n_padded,
            initial_parent, a_parent, b_parent, first_scale_parent, a, b, b_first, length,
            count]() mutable {
            for (Eigen::Index i = 0; i < count; ++i) {
                const double adjoint = result_nodes[i].adj_;
                if (initial_parent) initial_parent->adj_ += saved_derivatives(0, i) * adjoint;
                if (a_parent) a_parent->adj_ += saved_derivatives(1, i) * adjoint;
                if (b_parent) b_parent->adj_ += saved_derivatives(2, i) * adjoint;
                if (first_scale_parent) first_scale_parent->adj_ += saved_derivatives(3, i) * adjoint;
            }
            if (nuisance_gradient == nullptr) return;
//            //// one backward scan over the registers, the adjoints of the selected values entering as impulses;
//            //// the innovations' adjoints are added into NicoStan's nuisance gradient a register at a time
//            const nico::math::affine_scan_coefficients_t c = nico::math::affine_scan_coefficients(a);
//            const simd_t scale_in_lanes = nico::math::scan_set1(b);
//            simd_t carry = nico::math::scan_zero();
//            alignas(64) double lanes[BMVP_LANES];
//            Eigen::Index next = count - 1;
//            for (Eigen::Index t = n_padded - BMVP_LANES; t >= 0; t -= BMVP_LANES) {
//                nico::math::simd_store(lanes, nico::math::scan_zero());
//                while (next >= 0 && selected_positions[next] - 1 >= t) {
//                    lanes[selected_positions[next] - 1 - t] += result_nodes[next].adj_;
//                    --next;
//                }
//                const simd_t lambda = nico::math::scan_fmadd(c.weights_of_right_carry, carry,
//                                          nico::math::backward_scan_in_register(
//                                              nico::math::simd_load(lanes), c));
//                carry = nico::math::broadcast_first_lane(lambda);
//                simd_t adjoints_of_innovations = nico::math::scan_mul(scale_in_lanes, lambda);
//                if (t == 0) {
//                    //// (the first innovation has its own scale)
//                    alignas(64) double lanes_of_lambda[BMVP_LANES];
//                    nico::math::simd_store(lanes_of_lambda, lambda);
//                    nico::math::simd_store(lanes, adjoints_of_innovations);
//                    lanes[0] = b_first * lanes_of_lambda[0];
//                    adjoints_of_innovations = nico::math::simd_load(lanes);
//                }
//                if (t + BMVP_LANES <= length) {
//                    double* target = nuisance_gradient + t;
//                    const simd_t sum = nico::math::scan_fmadd(nico::math::scan_set1(1.0),
//                                                                 adjoints_of_innovations,
//                                                                 nico::math::simd_loadu(target));
//                    nico::math::simd_storeu(target, sum);
//                } else {
//                    nico::math::simd_store(lanes, adjoints_of_innovations);
//                    for (Eigen::Index lane = 0; t + lane < length; ++lane) {
//                        nuisance_gradient[t + lane] += lanes[lane];
//                    }
//                }
//            }
            //// (the adjoints of the innovations, one backward scan over the registers with the adjoints of the
            ////  selected values as impulses, added into NicoStan's nuisance gradient: NicoSIMD's, the same
            ////  code moved there)
            std::vector<double> adjoints_of_the_values(count);
            for (Eigen::Index i = 0; i < count; ++i) adjoints_of_the_values[i] = result_nodes[i].adj_;
            nico::math::nico_add_the_adjoints_of_the_innovations_of_an_affine_recurrence_at_positions(
                  selected_positions, adjoints_of_the_values.data(), count, a, b, b_first, length,
                  nuisance_gradient);
        });
        return result;
    }

}






















