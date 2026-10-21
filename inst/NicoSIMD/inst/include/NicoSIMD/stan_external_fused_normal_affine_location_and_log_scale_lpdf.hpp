#pragma once

// nico_normal_affine_location_and_log_scale_lpdf(y | a0, a1, x, b0, b1, w):
//     y[i] ~ normal(a0 + a1 x[i], exp(b0 + b1 w[i]))
// A normal linear model whose log scale is affine too: any AR / random-walk transition (y = a lagged copy of x),
// any heteroscedastic normal regression, any normal observation with a latent log variance (a1 = 0, w latent).
// y, x and w are vectors (data or parameters, and they may be the same parameters, e.g. overlapping segments of
// one array); a0, a1, b0 and b1 are scalars (data or parameters). ONE node, whatever the length: no Stan math
// node is made per element for the location or the scale. All of its arithmetic is SIMD: Eigen array expressions
// (vectorised) and BayesMVP's exp kernel, on per-thread buffers reused across calls; the partials of vector
// arguments are written straight into the nodes' arena storage. With r = (y - a0 - a1 x) exp(-(b0 + b1 w)):
//     log p = sum(-0.5 r^2 - (b0 + b1 w)) - n log(sqrt(2 pi)),
//     d/dy = -r exp(-ls), d/dx = a1 r exp(-ls), d/da0 = sum(r exp(-ls)), d/da1 = sum(r exp(-ls) x),
//     d/dw = b1 (r^2 - 1), d/db0 = sum(r^2 - 1), d/db1 = sum((r^2 - 1) w).
// Declared in the Stan functions block (no body), e.g.
//     real nico_normal_affine_location_and_log_scale_lpdf(vector y, real a0, real a1, vector x, real b0,
//                                                         real b1, vector w);

#include "stan_external_functions.hpp"

#include <cmath>
#include <vector>

namespace nico::math {

struct buffers_of_normal_affine_location_and_log_scale_t {
    std::vector<double> inverse_scale, standardised_residual;
    void ensure(const Eigen::Index n) {

        if (static_cast<Eigen::Index>(inverse_scale.size()) < n) {
            inverse_scale.resize(n);
            standardised_residual.resize(n);
        }

    }
};
inline buffers_of_normal_affine_location_and_log_scale_t& buffers_of_normal_affine_location_and_log_scale() {

    thread_local buffers_of_normal_affine_location_and_log_scale_t buffers;
    return buffers;

}

}  // namespace nico::math

namespace nico::math {

template <bool propto = false, typename T_y, typename T_a0, typename T_a1, typename T_x, typename T_b0,
          typename T_b1, typename T_w>
inline stan::return_type_t<T_y, T_a0, T_a1, T_x, T_b0, T_b1, T_w>
nico_normal_affine_location_and_log_scale_lpdf(const T_y& y, const T_a0& a0, const T_a1& a1, const T_x& x,
                                                const T_b0& b0, const T_b1& b1, const T_w& w,
                                                std::ostream* /*pstream__*/ = nullptr) {

    using Array = Eigen::Map<Eigen::Array<double, -1, 1>>;
    static const char* function = "nico_normal_affine_location_and_log_scale_lpdf";
    static_assert(stan::is_eigen<T_y>::value && stan::is_eigen<T_x>::value && stan::is_eigen<T_w>::value,
                  "y, x and w are vectors");
    const auto& y_ref = stan::math::to_ref(y);
    const auto& x_ref = stan::math::to_ref(x);
    const auto& w_ref = stan::math::to_ref(w);
    stan::math::check_consistent_sizes(function, "y", y_ref, "x", x_ref, "w", w_ref);
    const Eigen::Index n = y_ref.size();
    const double a0_value = stan::math::value_of(a0);
    const double a1_value = stan::math::value_of(a1);
    const double b0_value = stan::math::value_of(b0);
    const double b1_value = stan::math::value_of(b1);
    const auto y_values = stan::math::value_of(y_ref).array();
    const auto x_values = stan::math::value_of(x_ref).array();
    const auto w_values = stan::math::value_of(w_ref).array();
    stan::math::check_not_nan(function, "y", y_values.matrix());
    stan::math::check_finite(function, "intercept of the location", a0_value);
    stan::math::check_finite(function, "slope of the location", a1_value);
    stan::math::check_finite(function, "x", x_values.matrix());
    stan::math::check_finite(function, "intercept of the log scale", b0_value);
    stan::math::check_finite(function, "slope of the log scale", b1_value);
    stan::math::check_finite(function, "w", w_values.matrix());
    if (n == 0) return 0.0;

    constexpr bool include_quadratic = stan::math::include_summand<propto, T_y, T_a0, T_a1, T_x, T_b0, T_b1,
                                                                   T_w>::value;
    constexpr bool include_log_scale = stan::math::include_summand<propto, T_b0, T_b1, T_w>::value;

    nico::math::buffers_of_normal_affine_location_and_log_scale_t& buffers =
          nico::math::buffers_of_normal_affine_location_and_log_scale();
    buffers.ensure(n);
    //// ---- minus the log scale, its exponential (the inverse scale) and the standardised residuals
    Array inverse_scale(buffers.inverse_scale.data(), n);
    inverse_scale = -(b0_value + b1_value * w_values);
    double value = include_log_scale ? inverse_scale.sum() : 0.0;
    if constexpr (!propto) value -= n * 0.91893853320467274178;
    nico::math::k_exp(inverse_scale.data(), static_cast<int>(n));
    Array residual(buffers.standardised_residual.data(), n);
    residual = (y_values - (a0_value + a1_value * x_values)) * inverse_scale;
    if constexpr (include_quadratic) value -= 0.5 * residual.square().sum();
    if constexpr (stan::is_constant_all<T_y, T_a0, T_a1, T_x, T_b0, T_b1, T_w>::value) {
        return value;
    } else {
        auto partials = stan::math::make_partials_propagator(y_ref, a0, a1, x_ref, b0, b1, w_ref);
        //// d / d location[i] = r[i] / scale[i]
        if constexpr (!stan::is_constant_all<T_y, T_a0, T_a1, T_x>::value) {
            const auto d_location = residual * inverse_scale;
            if constexpr (!stan::is_constant_all<T_y>::value) {
                stan::math::partials<0>(partials) = (-d_location).matrix();
            }
            if constexpr (!stan::is_constant_all<T_a0>::value) {
                stan::math::partials<1>(partials) = d_location.sum();
            }
            if constexpr (!stan::is_constant_all<T_a1>::value) {
                stan::math::partials<2>(partials) = (d_location * x_values).sum();
            }
            if constexpr (!stan::is_constant_all<T_x>::value) {
                stan::math::partials<3>(partials) = (a1_value * d_location).matrix();
            }
        }
        //// d / d log scale[i] = r[i]^2 - 1 (the -1 only when the log scale term is kept)
        if constexpr (!stan::is_constant_all<T_b0, T_b1, T_w>::value) {
            residual = residual.square() - (include_log_scale ? 1.0 : 0.0);
            if constexpr (!stan::is_constant_all<T_b0>::value) {
                stan::math::partials<4>(partials) = residual.sum();
            }
            if constexpr (!stan::is_constant_all<T_b1>::value) {
                stan::math::partials<5>(partials) = (residual * w_values).sum();
            }
            if constexpr (!stan::is_constant_all<T_w>::value) {
                stan::math::partials<6>(partials) = (b1_value * residual).matrix();
            }
        }
        return partials.build(value);
    }

}

}  // namespace nico::math

using nico::math::nico_normal_affine_location_and_log_scale_lpdf;






















