#pragma once

// General scalar reductions of an affine predictor and an affine recurrence.
// The matrix is data; the vector, coefficients and initial value may be parameters.

#include "stan_external_functions.hpp"
#include "stan_external_simd_affine_recurrence_scans.hpp"

#include <cmath>
#include <limits>
#include <vector>

namespace nico::math {

template <typename T_x, typename T_beta, typename T_scale, typename T_z>
inline stan::return_type_t<T_beta, T_scale, T_z> nico_log_sum_exp_of_affine_predictor(
        const T_x& x, const T_beta& beta, const T_scale& scale, const T_z& z,
        std::ostream* /*pstream__*/ = nullptr) {

    static_assert(stan::is_constant_all<T_x>::value, "The predictor matrix must be data.");
    static const char* function = "nico_log_sum_exp_of_affine_predictor";
    const auto& x_ref = stan::math::to_ref(x);
    const auto& beta_ref = stan::math::to_ref(beta);
    const auto& z_ref = stan::math::to_ref(z);
    stan::math::check_size_match(function, "columns of x", x_ref.cols(), "size of beta", beta_ref.size());
    stan::math::check_size_match(function, "rows of x", x_ref.rows(), "size of z", z_ref.size());
    const Eigen::Index n = z_ref.size();
    if (n == 0) return -std::numeric_limits<double>::infinity();

    const Eigen::Matrix<double, -1, 1> beta_values = stan::math::value_of(beta_ref);
    const Eigen::Matrix<double, -1, 1> z_values = stan::math::value_of(z_ref);
    const double scale_value = stan::math::value_of(scale);
    Eigen::Matrix<double, -1, 1> eta = x_ref * beta_values;
    eta.noalias() += scale_value * z_values;
    stan::math::check_finite(function, "affine predictor", eta);
    const double shift = eta.maxCoeff();
    Eigen::Matrix<double, -1, 1> probabilities = eta.array() - shift;
    nico::math::k_exp(probabilities.data(), static_cast<int>(n));
    const double total = probabilities.sum();
    const double value = shift + std::log(total);

    if constexpr (stan::is_constant_all<T_beta, T_scale, T_z>::value) {
        return value;
    } else {
        probabilities /= total;
        auto partials = stan::math::make_partials_propagator(beta_ref, scale, z_ref);
        if constexpr (!stan::is_constant_all<T_beta>::value) {
            stan::math::partials<0>(partials) = x_ref.transpose() * probabilities;
        }
        if constexpr (!stan::is_constant_all<T_scale>::value) {
            stan::math::partials<1>(partials) = z_values.dot(probabilities);
        }
        if constexpr (!stan::is_constant_all<T_z>::value) {
            stan::math::partials<2>(partials) = scale_value * probabilities;
        }
        return partials.build(value);
    }

}

}  // namespace nico::math

using nico::math::nico_log_sum_exp_of_affine_predictor;


// The last value of s[i] = coefficient * s[i - 1] + scale * innovations[i], s[-1] = initial_value.
// No intermediate path nodes are needed when the caller only uses the last value.
namespace nico::math {

template <typename T_innovations, typename T_initial, typename T_coefficient, typename T_scale>
inline stan::return_type_t<T_innovations, T_initial, T_coefficient, T_scale>
nico_last_value_of_affine_recurrence(
        const T_innovations& innovations, const T_initial& initial_value,
        const T_coefficient& coefficient, const T_scale& scale, std::ostream* /*pstream__*/ = nullptr) {

    static const char* function = "nico_last_value_of_affine_recurrence";
    const auto& innovations_ref = stan::math::to_ref(innovations);
    const Eigen::Matrix<double, -1, 1> innovation_values = stan::math::value_of(innovations_ref);
    const double a = stan::math::value_of(coefficient);
    const double b = stan::math::value_of(scale);
    const double start = stan::math::value_of(initial_value);
    stan::math::check_finite(function, "innovations", innovation_values);
    stan::math::check_finite(function, "initial value", start);
    stan::math::check_finite(function, "coefficient", a);
    stan::math::check_finite(function, "scale", b);
    const Eigen::Index n = innovations_ref.size();
    double value = start;
    double partial_of_coefficient = 0.0;
    double partial_of_scale = 0.0;
    double partial_of_initial = 1.0;
    for (Eigen::Index i = 0; i < n; ++i) {
        if constexpr (!stan::is_constant_all<T_coefficient>::value) {
            partial_of_coefficient = value + a * partial_of_coefficient;
        }
        if constexpr (!stan::is_constant_all<T_scale>::value) {
            partial_of_scale = innovation_values[i] + a * partial_of_scale;
        }
        if constexpr (!stan::is_constant_all<T_initial>::value) partial_of_initial *= a;
        value = a * value + b * innovation_values[i];
    }
    stan::math::check_finite(function, "last value", value);

    if constexpr (stan::is_constant_all<T_innovations, T_initial, T_coefficient, T_scale>::value) {
        return value;
    } else {
        auto partials = stan::math::make_partials_propagator(innovations_ref, initial_value, coefficient, scale);
        if constexpr (!stan::is_constant_all<T_innovations>::value) {
            Eigen::Matrix<double, -1, 1> partial_of_innovations(n);
            double weight = b;
            for (Eigen::Index i = n; i-- > 0;) {
                partial_of_innovations[i] = weight;
                weight *= a;
            }
            stan::math::partials<0>(partials) = partial_of_innovations;
        }
        if constexpr (!stan::is_constant_all<T_initial>::value) {
            stan::math::partials<1>(partials) = partial_of_initial;
        }
        if constexpr (!stan::is_constant_all<T_coefficient>::value) {
            stan::math::partials<2>(partials) = partial_of_coefficient;
        }
        if constexpr (!stan::is_constant_all<T_scale>::value) {
            stan::math::partials<3>(partials) = partial_of_scale;
        }
        return partials.build(value);
    }

}

}  // namespace nico::math

using nico::math::nico_last_value_of_affine_recurrence;


// Selected values of the affine recurrence, with one reverse callback for the whole boundary pass.
// The first innovation scale is explicit; no stationary initialisation is assumed.
// Input coordinates are read directly from the sliced parameter array, without a vector of path nodes.
// All of its arithmetic is SIMD (stan_external_simd_affine_recurrence_scans.hpp), in ONE pass over the registers:
//     s[i] = a s[i - 1] + c[i] z[i]  (c[0] = first_scale, c[i] = scale), s[-1] = initial_value;
//     d s[i] / d a = s[i - 1] + a (d s[i - 1] / d a);
//     d s[i] / d scale = a (d s[i - 1] / d scale) + z[i] (i > 0);
//     d s[i] / d first_scale = a^i z[0];  d s[i] / d initial_value = a^(i + 1);
// each a forward scan of its own register, with its own carry; only the selected positions are stored. The
// reverse pass is one backward scan, lambda[i] = a lambda[i + 1] + (adjoint of s[i] where i is selected), whose
// lanes give the innovations' adjoints c[i] lambda[i]. The only element-by-element work is reading the
// innovations' values from their nodes and adding their adjoints to them (in order, so innovations that are the
// same parameter are summed correctly).
namespace nico::math {

inline std::vector<double>& buffer_of_innovations_of_affine_recurrence() {

    static thread_local std::vector<double> values;
    return values;

}

}  // namespace nico::math

namespace nico::math {

template <typename T_z, typename T_initial, typename T_coefficient, typename T_scale, typename T_first_scale>
inline auto nico_affine_recurrence_at_positions(
        const std::vector<T_z>& innovations, const std::vector<int>& positions,
        const T_initial& initial_value, const T_coefficient& coefficient,
        const T_scale& scale, const T_first_scale& first_scale, std::ostream* /*pstream__*/ = nullptr) {

    using scalar = stan::return_type_t<T_z, T_initial, T_coefficient, T_scale, T_first_scale>;
    using nico::math::simd_t;
    static const char* function = "nico_affine_recurrence_at_positions";
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
        if (position <= previous_position || position > static_cast<int>(innovations.size())) {
            throw std::invalid_argument("Recurrence positions must increase strictly within the input array.");
        }
        previous_position = position;
    }
    const int length = count > 0 ? positions.back() : 0;
    const Eigen::Index n_padded = nico::math::padded_to_lanes(length);

    //// ---- the innovations' values, zero-padded to whole registers
    std::vector<double>& buffer = nico::math::buffer_of_innovations_of_affine_recurrence();
    if (static_cast<Eigen::Index>(buffer.size()) < n_padded) buffer.resize(n_padded);
    double* z = buffer.data();
    for (int i = 0; i < length; ++i) z[i] = stan::math::value_of(innovations[i]);
    Eigen::Map<Eigen::Array<double, -1, 1>>(z + length, n_padded - length).setZero();
    if (!Eigen::Map<const Eigen::Array<double, -1, 1>>(z, length).allFinite()) {
        stan::math::check_finite(function, "innovation", Eigen::Map<const Eigen::Matrix<double, -1, 1>>(z,
                                                                                                        length));
    }

    //// ---- one pass over the registers: the recurrence and its derivatives, each a scan with its own carry
    constexpr bool need_derivatives = !stan::is_constant_all<T_z, T_initial, T_coefficient, T_scale,
                                                             T_first_scale>::value;
    const nico::math::affine_scan_coefficients_t c = nico::math::affine_scan_coefficients(a);
    const simd_t scale_in_lanes = nico::math::scan_set1(b);
    Eigen::Matrix<double, -1, 1> values(count);
    Eigen::Matrix<double, -1, -1> derivatives;
    if constexpr (need_derivatives) derivatives.resize(4, count);
    simd_t carry_of_state = nico::math::scan_set1(start);
    simd_t carry_of_d_a = nico::math::scan_zero();
    simd_t carry_of_d_scale = nico::math::scan_zero();
    alignas(64) double lanes_of_state[BMVP_LANES];
    alignas(64) double lanes_of_d_a[BMVP_LANES];
    alignas(64) double lanes_of_d_scale[BMVP_LANES];
    alignas(64) double first_lanes[BMVP_LANES];
    Eigen::Index next = 0;
    for (Eigen::Index t = 0; t < n_padded && next < count; t += BMVP_LANES) {
        const simd_t z_lanes = nico::math::simd_loadu(z + t);
        simd_t u = nico::math::scan_mul(scale_in_lanes, z_lanes);
        simd_t z_after_the_first = z_lanes;
        if (t == 0) {
            //// (the first innovation has its own scale, and no part in the derivative with respect to scale)
            nico::math::simd_store(first_lanes, u);
            first_lanes[0] = b_first * z[0];
            u = nico::math::simd_load(first_lanes);
            nico::math::simd_store(first_lanes, z_lanes);
            first_lanes[0] = 0.0;
            z_after_the_first = nico::math::simd_load(first_lanes);
        }
        const simd_t state = nico::math::scan_fmadd(c.weights_of_left_carry, carry_of_state,
                                                       nico::math::forward_scan_in_register(u, c));
        simd_t d_a = state;
        simd_t d_scale = state;
        if constexpr (need_derivatives) {
            const simd_t previous_state = nico::math::shift_up_one_with(state, carry_of_state);
            d_a = nico::math::scan_fmadd(c.weights_of_left_carry, carry_of_d_a,
                                            nico::math::forward_scan_in_register(previous_state, c));
            d_scale = nico::math::scan_fmadd(c.weights_of_left_carry, carry_of_d_scale,
                                                nico::math::forward_scan_in_register(z_after_the_first, c));
            carry_of_d_a = nico::math::broadcast_last_lane(d_a);
            carry_of_d_scale = nico::math::broadcast_last_lane(d_scale);
        }
        carry_of_state = nico::math::broadcast_last_lane(state);
        if (positions[next] - 1 < t + BMVP_LANES) {
            nico::math::simd_store(lanes_of_state, state);
            if constexpr (need_derivatives) {
                nico::math::simd_store(lanes_of_d_a, d_a);
                nico::math::simd_store(lanes_of_d_scale, d_scale);
            }
            while (next < count && positions[next] - 1 < t + BMVP_LANES) {
                const int lane = static_cast<int>(positions[next] - 1 - t);
                values[next] = lanes_of_state[lane];
                if constexpr (need_derivatives) {
                    const double power = std::pow(a, positions[next] - 1);
                    derivatives.col(next) << power * a, lanes_of_d_a[lane], lanes_of_d_scale[lane],
                                             power * z[0];
                }
                ++next;
            }
        }
    }
    if constexpr (!need_derivatives) {
        return values;
    } else {
        Eigen::Matrix<scalar, Eigen::Dynamic, 1> result(count);
        if (count == 0) return result;
        using stan::math::vari;
        auto& arena = stan::math::ChainableStack::instance_->memalloc_;
        vari** input_nodes = nullptr;
        if constexpr (!stan::is_constant_all<T_z>::value) {
            input_nodes = arena.alloc_array<vari*>(length);
            for (int i = 0; i < length; ++i) input_nodes[i] = innovations[i].vi_;
        }
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
        stan::math::reverse_pass_callback([input_nodes, result_nodes, selected_positions, saved_derivatives,
            initial_parent, a_parent, b_parent, first_scale_parent, a, b, b_first, length, n_padded,
            count]() mutable {
            for (Eigen::Index i = 0; i < count; ++i) {
                const double adjoint = result_nodes[i].adj_;
                if (initial_parent) initial_parent->adj_ += saved_derivatives(0, i) * adjoint;
                if (a_parent) a_parent->adj_ += saved_derivatives(1, i) * adjoint;
                if (b_parent) b_parent->adj_ += saved_derivatives(2, i) * adjoint;
                if (first_scale_parent) first_scale_parent->adj_ += saved_derivatives(3, i) * adjoint;
            }
            if (!input_nodes) return;
            //// one backward scan over the registers, the adjoints of the selected values entering as impulses
            const nico::math::affine_scan_coefficients_t c = nico::math::affine_scan_coefficients(a);
            const simd_t scale_in_lanes = nico::math::scan_set1(b);
            simd_t carry = nico::math::scan_zero();
            alignas(64) double lanes[BMVP_LANES];
            Eigen::Index next = count - 1;
            for (Eigen::Index t = n_padded - BMVP_LANES; t >= 0; t -= BMVP_LANES) {
                nico::math::simd_store(lanes, nico::math::scan_zero());
                while (next >= 0 && selected_positions[next] - 1 >= t) {
                    lanes[selected_positions[next] - 1 - t] += result_nodes[next].adj_;
                    --next;
                }
                const simd_t lambda = nico::math::scan_fmadd(c.weights_of_right_carry, carry,
                                          nico::math::backward_scan_in_register(
                                              nico::math::simd_load(lanes), c));
                carry = nico::math::broadcast_first_lane(lambda);
                nico::math::simd_store(lanes, nico::math::scan_mul(scale_in_lanes, lambda));
                if (t == 0) {
                    //// (the first innovation has its own scale)
                    alignas(64) double lanes_of_lambda[BMVP_LANES];
                    nico::math::simd_store(lanes_of_lambda, lambda);
                    lanes[0] = b_first * lanes_of_lambda[0];
                }
                const int last_lane = static_cast<int>(std::min<Eigen::Index>(BMVP_LANES, length - t));
                for (int lane = 0; lane < last_lane; ++lane) input_nodes[t + lane]->adj_ += lanes[lane];
            }
        });
        return result;
    }

}

}  // namespace nico::math

using nico::math::nico_affine_recurrence_at_positions;


// Log sums of an indexed affine predictor in arbitrary contiguous segments.
// eta[i] = X[i, ] * beta + scale * values[index[i]], without intermediate predictor or job parameter nodes.
namespace nico::math {

template <typename T_x, typename T_beta, typename T_scale, typename T_z>
inline auto nico_log_sum_exp_of_indexed_affine_predictor_in_segments(
        const T_x& x, const T_beta& beta, const T_scale& scale, const std::vector<T_z>& z,
        const std::vector<int>& index, const std::vector<int>& first, const std::vector<int>& last,
        std::ostream* /*pstream__*/ = nullptr) {

    static_assert(stan::is_constant_all<T_x>::value, "The predictor matrix must be data.");
    using scalar = stan::return_type_t<T_beta, T_scale, T_z>;
    static const char* function = "nico_log_sum_exp_of_indexed_affine_predictor_in_segments";
    const auto& x_ref = stan::math::to_ref(x);
    const auto& beta_ref = stan::math::to_ref(beta);
    const Eigen::Matrix<double, -1, 1> beta_values = stan::math::value_of(beta_ref);
    const double scale_value = stan::math::value_of(scale);
    stan::math::check_size_match(function, "columns of x", x_ref.cols(), "size of beta", beta_ref.size());
    stan::math::check_size_match(function, "rows of x", x_ref.rows(), "size of index", index.size());
    stan::math::check_size_match(function, "segment starts", first.size(), "segment ends", last.size());
    stan::math::check_finite(function, "beta", beta_values);
    stan::math::check_finite(function, "scale", scale_value);
    const Eigen::Index count = first.size();
    const Eigen::Index k = beta_ref.size();
    std::vector<int> offsets(count + 1, 0);
    for (Eigen::Index b = 0; b < count; ++b) {
        if (first[b] < 1 || last[b] < first[b] || last[b] > x_ref.rows()) {
            throw std::invalid_argument("Every affine segment must lie within the predictor matrix.");
        }
        offsets[b + 1] = offsets[b] + last[b] - first[b] + 1;
    }
    Eigen::Matrix<double, -1, 1> results(count);
    Eigen::Matrix<double, -1, -1> partial_of_beta;
    Eigen::Matrix<double, -1, 1> partial_of_scale;
    if constexpr (!stan::is_constant_all<T_beta>::value) partial_of_beta.resize(k, count);
    if constexpr (!stan::is_constant_all<T_scale>::value) partial_of_scale.resize(count);
    std::vector<stan::math::vari*> input_nodes;
    Eigen::Matrix<double, -1, 1> partial_of_z;
    if constexpr (!stan::is_constant_all<T_z>::value) {
        input_nodes.resize(offsets.back());
        partial_of_z.resize(offsets.back());
    }
    for (Eigen::Index b = 0; b < count; ++b) {
        const int start = first[b] - 1;
        const int length = last[b] - first[b] + 1;
        Eigen::Matrix<double, -1, 1> values(length);
        for (int i = 0; i < length; ++i) {
            const int position = index[start + i] - 1;
            if (position < 0 || position >= static_cast<int>(z.size())) {
                throw std::invalid_argument("An affine predictor index is outside the parameter array.");
            }
            values[i] = stan::math::value_of(z[position]);
            if constexpr (!stan::is_constant_all<T_z>::value) {
                input_nodes[offsets[b] + i] = z[position].vi_;
            }
        }
        Eigen::Matrix<double, -1, 1> probabilities = x_ref.middleRows(start, length) * beta_values;
        probabilities.noalias() += scale_value * values;
        stan::math::check_finite(function, "affine predictor", probabilities);
        const double shift = probabilities.maxCoeff();
        probabilities.array() -= shift;
        nico::math::k_exp(probabilities.data(), length);
        const double total = probabilities.sum();
        results[b] = shift + std::log(total);
        if constexpr (!stan::is_constant_all<T_beta, T_scale, T_z>::value) probabilities /= total;
        if constexpr (!stan::is_constant_all<T_beta>::value) {
            partial_of_beta.col(b) = x_ref.middleRows(start, length).transpose() * probabilities;
        }
        if constexpr (!stan::is_constant_all<T_scale>::value) {
            partial_of_scale[b] = values.dot(probabilities);
        }
        if constexpr (!stan::is_constant_all<T_z>::value) {
            partial_of_z.segment(offsets[b], length) = scale_value * probabilities;
        }
    }
    if constexpr (stan::is_constant_all<T_beta, T_scale, T_z>::value) {
        return results;
    } else {
        Eigen::Matrix<scalar, Eigen::Dynamic, 1> output(count);
        if (count == 0) return output;
        using stan::math::vari;
        auto& arena = stan::math::ChainableStack::instance_->memalloc_;
        vari* result_nodes = arena.alloc_array<vari>(count);
        for (Eigen::Index b = 0; b < count; ++b) {
            ::new (static_cast<void*>(result_nodes + b)) vari(results[b], false);
            output[b] = stan::math::var(result_nodes + b);
        }
        vari** z_parents = nullptr;
        vari** beta_parents = nullptr;
        vari* scale_parent = nullptr;
        int* segment_offsets = arena.alloc_array<int>(count + 1);
        std::copy(offsets.begin(), offsets.end(), segment_offsets);
        if constexpr (!stan::is_constant_all<T_z>::value) {
            z_parents = arena.alloc_array<vari*>(offsets.back());
            std::copy(input_nodes.begin(), input_nodes.end(), z_parents);
        }
        if constexpr (!stan::is_constant_all<T_beta>::value) {
            beta_parents = arena.alloc_array<vari*>(k);
            for (Eigen::Index i = 0; i < k; ++i) beta_parents[i] = beta_ref[i].vi_;
        }
        if constexpr (!stan::is_constant_all<T_scale>::value) scale_parent = scale.vi_;
        stan::arena_t<Eigen::Matrix<double, -1, -1>> beta_partials = partial_of_beta;
        stan::arena_t<Eigen::Matrix<double, -1, 1>> scale_partials = partial_of_scale;
        stan::arena_t<Eigen::Matrix<double, -1, 1>> z_partials = partial_of_z;
        stan::math::reverse_pass_callback([result_nodes, segment_offsets, z_parents, beta_parents,
            scale_parent, beta_partials, scale_partials, z_partials, count, k]() mutable {
            for (Eigen::Index b = 0; b < count; ++b) {
                const double adjoint = result_nodes[b].adj_;
                if (beta_parents) {
                    for (Eigen::Index i = 0; i < k; ++i) beta_parents[i]->adj_ += beta_partials(i, b) * adjoint;
                }
                if (scale_parent) scale_parent->adj_ += scale_partials[b] * adjoint;
                if (z_parents) {
                    for (int i = segment_offsets[b]; i < segment_offsets[b + 1]; ++i) {
                        z_parents[i]->adj_ += z_partials[i] * adjoint;
                    }
                }
            }
        });
        return output;
    }

}

}  // namespace nico::math

using nico::math::nico_log_sum_exp_of_indexed_affine_predictor_in_segments;


// A normal log density whose log scale is intercept + slope * s[i], with
// s[i] = coefficient * s[i - 1] + scale[i] * innovations[i], s[-1] = initial_value.
// The scale is scalar or a vector; a different initial innovation scale is supplied by the caller.
// The joint density also includes innovations ~ normal(innovation_mean, innovation_sd).
// No stationary scaling or model-specific observation convention is assumed.
// All of its arithmetic is SIMD: the recurrence and its adjoint are the forward and backward scans of
// stan_external_simd_affine_recurrence_scans.hpp, the exponentials BayesMVP's exp kernel, and every other
// element-wise term an Eigen array expression (vectorised); its working vectors are per-thread buffers reused
// across calls, and the partials of vector arguments are written straight into the nodes' arena storage.
namespace nico::math {

struct buffers_of_normal_affine_recurrence_t {
    std::vector<double> innovations, scaled_innovations, states, inverse_scale, residual_terms,
                        adjoint_of_states;
    void ensure(const Eigen::Index n) {

        if (static_cast<Eigen::Index>(innovations.size()) < n) {
            innovations.resize(n);
            scaled_innovations.resize(n);
            states.resize(n);
            inverse_scale.resize(n);
            residual_terms.resize(n);
            adjoint_of_states.resize(n);
        }

    }
};
inline buffers_of_normal_affine_recurrence_t& buffers_of_normal_affine_recurrence() {

    thread_local buffers_of_normal_affine_recurrence_t buffers;
    return buffers;

}

}  // namespace nico::math

namespace nico::math {

template <bool propto = false, typename T_y, typename T_mu, typename T_innovations,
          typename T_initial, typename T_coefficient, typename T_scale,
          typename T_intercept, typename T_slope, typename T_innovation_mean, typename T_innovation_sd>
inline stan::return_type_t<T_y, T_mu, T_innovations, T_initial, T_coefficient,
                           T_scale, T_intercept, T_slope, T_innovation_mean, T_innovation_sd>
nico_normal_affine_recurrence_log_scale_joint_lpdf(
        const T_y& y, const T_mu& mu, const T_innovations& innovations, const T_initial& initial_value,
        const T_coefficient& coefficient, const T_scale& scale, const T_intercept& intercept,
        const T_slope& slope, const T_innovation_mean& innovation_mean,
        const T_innovation_sd& innovation_sd, std::ostream* /*pstream__*/ = nullptr) {

    using Array = Eigen::Map<Eigen::Array<double, -1, 1>>;
    static const char* function = "nico_normal_affine_recurrence_log_scale_joint_lpdf";
    static_assert(stan::is_eigen<T_y>::value && stan::is_eigen<T_innovations>::value,
                  "y and the innovations are vectors");
    const auto& y_ref = stan::math::to_ref(y);
    const auto& mu_ref = stan::math::to_ref(mu);
    const auto& z_ref = stan::math::to_ref(innovations);
    const auto& scale_ref = stan::math::to_ref(scale);
    stan::math::check_consistent_sizes(function, "y", y_ref, "mu", mu_ref,
                                      "innovations", z_ref, "scale", scale_ref);
    const Eigen::Index n = z_ref.size();
    const Eigen::Index n_padded = nico::math::padded_to_lanes(n);
    nico::math::buffers_of_normal_affine_recurrence_t& buffers =
          nico::math::buffers_of_normal_affine_recurrence();
    buffers.ensure(n_padded);
    Array z(buffers.innovations.data(), n);
    z = stan::math::value_of(z_ref).array();
    const double a = stan::math::value_of(coefficient);
    const double start = stan::math::value_of(initial_value);
    const double c = stan::math::value_of(intercept);
    const double d = stan::math::value_of(slope);
    const double prior_mean = stan::math::value_of(innovation_mean);
    const double prior_sd = stan::math::value_of(innovation_sd);
    stan::math::check_finite(function, "y", stan::math::value_of(y_ref));
    stan::math::check_finite(function, "mu", stan::math::value_of(mu_ref));
    stan::math::check_finite(function, "innovations", z.matrix());
    stan::math::check_finite(function, "initial value", start);
    stan::math::check_finite(function, "coefficient", a);
    stan::math::check_finite(function, "scale", stan::math::value_of(scale_ref));
    stan::math::check_finite(function, "intercept", c);
    stan::math::check_finite(function, "slope", d);
    stan::math::check_finite(function, "innovation mean", prior_mean);
    stan::math::check_positive_finite(function, "innovation SD", prior_sd);
    if (n == 0) return 0.0;

    constexpr bool include_quadratic = stan::math::include_summand<propto, T_y, T_mu, T_innovations,
                         T_initial, T_coefficient, T_scale, T_intercept, T_slope>::value;
    constexpr bool include_log_scale = stan::math::include_summand<propto, T_innovations,
                         T_initial, T_coefficient, T_scale, T_intercept, T_slope>::value;
    constexpr bool include_prior_quadratic = stan::math::include_summand<propto, T_innovations,
                                                         T_innovation_mean, T_innovation_sd>::value;
    constexpr bool include_prior_log_scale = stan::math::include_summand<propto, T_innovation_sd>::value;

    //// ---- the recurrence: one forward scan of scale[i] * z[i] (zero-padded to whole registers)
    Array u(buffers.scaled_innovations.data(), n_padded);
    if constexpr (stan::is_eigen<T_scale>::value) {
        u.head(n) = stan::math::value_of(scale_ref).array() * z;
    } else {
        u.head(n) = stan::math::value_of(scale_ref) * z;
    }
    u.tail(n_padded - n).setZero();
    const nico::math::affine_scan_coefficients_t coefficients = nico::math::affine_scan_coefficients(a);
    nico::math::forward_affine_scan(u.data(), n_padded, start, buffers.states.data(), coefficients);
    const Array s(buffers.states.data(), n);

    //// ---- the log scale, its exponential and the normal terms
    Array inverse_scale(buffers.inverse_scale.data(), n);
    inverse_scale = -(c + d * s);
    stan::math::check_finite(function, "log scale", inverse_scale.matrix());
    double value = include_log_scale ? inverse_scale.sum() : 0.0;
    if constexpr (!propto) value -= n * 0.91893853320467274178;
    nico::math::k_exp(inverse_scale.data(), static_cast<int>(n));
    Array residual(buffers.residual_terms.data(), n);
    if constexpr (stan::is_eigen<T_mu>::value) {
        residual = (stan::math::value_of(y_ref).array() - stan::math::value_of(mu_ref).array()) * inverse_scale;
    } else {
        residual = (stan::math::value_of(y_ref).array() - stan::math::value_of(mu_ref)) * inverse_scale;
    }
    if constexpr (include_quadratic) value -= 0.5 * residual.square().sum();
    double prior_sum_of_squares = 0.0;
    if constexpr (include_prior_quadratic) {
        prior_sum_of_squares = ((z - prior_mean) / prior_sd).square().sum();
        value -= 0.5 * prior_sum_of_squares;
        if constexpr (include_prior_log_scale) value -= n * std::log(prior_sd);
        if constexpr (!propto) value -= n * 0.91893853320467274178;
    }
    if constexpr (stan::is_constant_all<T_y, T_mu, T_innovations, T_initial, T_coefficient,
                                      T_scale, T_intercept, T_slope, T_innovation_mean, T_innovation_sd>::value) {

        return value;

    } else {
        auto partials = stan::math::make_partials_propagator(y_ref, mu_ref, z_ref, initial_value,
                                   coefficient, scale_ref, intercept, slope, innovation_mean, innovation_sd);
        if constexpr (!stan::is_constant_all<T_y, T_mu>::value) {
            //// d / d y = -residual / scale, d / d mu = +residual / scale (before residual is overwritten)
            if constexpr (!stan::is_constant_all<T_y>::value) {
                stan::math::partials<0>(partials) = (-residual * inverse_scale).matrix();
            }
            if constexpr (!stan::is_constant_all<T_mu>::value) {
                if constexpr (stan::is_eigen<T_mu>::value) {
                    stan::math::partials<1>(partials) = (residual * inverse_scale).matrix();
                } else {
                    stan::math::partials<1>(partials) = (residual * inverse_scale).sum();
                }
            }
        }
        //// d / d log scale[i] = residual[i]^2 - 1 (the -1 only when the log scale term is kept), in place
        if constexpr (include_quadratic) {
            residual = residual.square() - (include_log_scale ? 1.0 : 0.0);
        } else {
            residual = -(include_log_scale ? 1.0 : 0.0);
        }
        const Array& partial_of_log_scale = residual;
        if constexpr (!stan::is_constant_all<T_intercept>::value) {
            stan::math::partials<6>(partials) = partial_of_log_scale.sum();
        }
        if constexpr (!stan::is_constant_all<T_slope>::value) {
            stan::math::partials<7>(partials) = (s * partial_of_log_scale).sum();
        }
        if constexpr (!stan::is_constant_all<T_innovation_mean>::value) {
            stan::math::partials<8>(partials) = ((z - prior_mean) / prior_sd).sum() / prior_sd;
        }
        if constexpr (!stan::is_constant_all<T_innovation_sd>::value) {
            stan::math::partials<9>(partials) = (prior_sum_of_squares -
                                                   (include_prior_log_scale ? n : 0.0)) / prior_sd;
        }
        if constexpr (!stan::is_constant_all<T_innovations, T_initial, T_coefficient, T_scale>::value) {
            //// the adjoint of the recurrence: one backward scan of slope * d / d log scale (zero-padded)
            Array w(buffers.scaled_innovations.data(), n_padded);
            w.head(n) = d * partial_of_log_scale;
            w.tail(n_padded - n).setZero();
            nico::math::backward_affine_scan(w.data(), n_padded, 0.0, buffers.adjoint_of_states.data(),
                                                coefficients);
            const Array lambda(buffers.adjoint_of_states.data(), n);
            if constexpr (!stan::is_constant_all<T_innovations>::value) {
                if constexpr (stan::is_eigen<T_scale>::value) {
                    stan::math::partials<2>(partials) = (stan::math::value_of(scale_ref).array() * lambda
                                                         - (z - prior_mean) / (prior_sd * prior_sd)).matrix();
                } else {
                    stan::math::partials<2>(partials) = (stan::math::value_of(scale_ref) * lambda
                                                         - (z - prior_mean) / (prior_sd * prior_sd)).matrix();
                }
            }
            if constexpr (!stan::is_constant_all<T_initial>::value) {
                stan::math::partials<3>(partials) = a * lambda[0];
            }
            if constexpr (!stan::is_constant_all<T_coefficient>::value) {
                stan::math::partials<4>(partials) = start * lambda[0]
                                                    + (s.head(n - 1) * lambda.tail(n - 1)).sum();
            }
            if constexpr (!stan::is_constant_all<T_scale>::value) {
                if constexpr (stan::is_eigen<T_scale>::value) {
                    stan::math::partials<5>(partials) = (z * lambda).matrix();
                } else {
                    stan::math::partials<5>(partials) = (z * lambda).sum();
                }
            }
        }
        return partials.build(value);
    }

}

}  // namespace nico::math

using nico::math::nico_normal_affine_recurrence_log_scale_joint_lpdf;















