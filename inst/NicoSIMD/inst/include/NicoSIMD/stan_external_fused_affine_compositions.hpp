#pragma once

// General affine compositions. The matrix is data; every other operand may be a parameter.
// eta = X * beta + scale * z; L[i] = log(exp(offset) + sum(exp(eta[1:i]))).
// The scalar result is dot_product(linear_weights, eta) + dot_product(prefix_weights, L).
// Neither weights nor observations have a model-specific interpretation here.

#include "stan_external_functions.hpp"
#include "stan_external_log_cumulative_sum_exp.hpp"

namespace nico::math {

inline std::vector<double>& affine_prefix_values_only_scratch() {

    static thread_local std::vector<double> values;
    return values;

}

}

namespace nico::math {

template <typename T_linear_weights, typename T_prefix_weights, typename T_x, typename T_beta,
          typename T_scale, typename T_z, typename T_offset>
inline stan::return_type_t<T_linear_weights, T_prefix_weights, T_beta, T_scale, T_z, T_offset>
nico_dot_products_of_affine_predictor_and_log_cumulative_sum_exp(
        const T_linear_weights& linear_weights, const T_prefix_weights& prefix_weights,
        const T_x& x, const T_beta& beta, const T_scale& scale, const T_z& z,
        const T_offset& offset, std::ostream* /*pstream__*/ = nullptr) {

    static_assert(stan::is_constant_all<T_x>::value, "The predictor matrix must be data.");
    static const char* function = "nico_dot_products_of_affine_predictor_and_log_cumulative_sum_exp";
    const auto& x_ref = stan::math::to_ref(x);
    const auto& beta_ref = stan::math::to_ref(beta);
    const auto& z_ref = stan::math::to_ref(z);
    const auto& linear_ref = stan::math::to_ref(linear_weights);
    const auto& prefix_ref = stan::math::to_ref(prefix_weights);
    const int n = static_cast<int>(x_ref.rows());
    stan::math::check_size_match(function, "columns of x", x_ref.cols(), "size of beta", beta_ref.size());
    stan::math::check_size_match(function, "rows of x", n, "size of z", z_ref.size());
    stan::math::check_size_match(function, "rows of x", n, "linear weights", linear_ref.size());
    stan::math::check_size_match(function, "rows of x", n, "prefix weights", prefix_ref.size());
    if (n == 0) return 0.0;

    const Eigen::Matrix<double, -1, 1> beta_values = stan::math::value_of(beta_ref);
    const Eigen::Matrix<double, -1, 1> z_values = stan::math::value_of(z_ref);
    const Eigen::Matrix<double, -1, 1> linear_values = stan::math::value_of(linear_ref);
    const Eigen::Matrix<double, -1, 1> prefix_values = stan::math::value_of(prefix_ref);
    const double scale_value = stan::math::value_of(scale);
    Eigen::Matrix<double, -1, 1> eta = x_ref * beta_values;
    eta.noalias() += scale_value * z_values;
    stan::math::check_finite(function, "affine predictor", eta);

    if constexpr (stan::is_constant_all<T_linear_weights, T_prefix_weights, T_beta,
                                        T_scale, T_z, T_offset>::value) {

        auto& values_buffer = nico::math::affine_prefix_values_only_scratch();
        if (static_cast<int>(values_buffer.size()) < n) values_buffer.resize(n);
        double* log_prefix = values_buffer.data();
        nico::math::log_cumulative_sum_exp::values_only(
            eta.data(), n, stan::math::value_of(offset), log_prefix);
        double value = linear_values.dot(eta);
        value += prefix_values.dot(Eigen::Map<const Eigen::Matrix<double, -1, 1>>(log_prefix, n));
        return value;

    } else {
        auto& buffers = nico::math::scratch();
        buffers.ensure(n);
        double* log_prefix = buffers.a.data();
        double* ratio = buffers.b.data();
        double* share = buffers.c.data();
        const double offset_share = nico::math::log_cumulative_sum_exp::values_and_factors(
                eta.data(), n, stan::math::value_of(offset), log_prefix, ratio, share);
        double value = linear_values.dot(eta);
        value += prefix_values.dot(Eigen::Map<const Eigen::Matrix<double, -1, 1>>(log_prefix, n));

        Eigen::Matrix<double, -1, 1> partial_of_eta(n);
        const double running_adjoint = nico::math::log_cumulative_sum_exp::nico_backward_prefix_scan(
              prefix_values.data(), ratio, n, partial_of_eta.data());
        partial_of_eta.array() = linear_values.array() +
                                 Eigen::Map<const Eigen::Matrix<double, -1, 1>>(share,
                                                                                n).array() *
                                                                                    partial_of_eta.array();
        auto partials = stan::math::make_partials_propagator(linear_ref, prefix_ref, beta_ref,
                                                            scale, z_ref, offset);
        if constexpr (!stan::is_constant_all<T_linear_weights>::value) {
            stan::math::partials<0>(partials) = eta;
        }
        if constexpr (!stan::is_constant_all<T_prefix_weights>::value) {
            stan::math::partials<1>(partials) = Eigen::Map<const Eigen::Matrix<double, -1, 1>>(log_prefix, n);
        }
        if constexpr (!stan::is_constant_all<T_beta>::value) {
            stan::math::partials<2>(partials) = x_ref.transpose() * partial_of_eta;
        }
        if constexpr (!stan::is_constant_all<T_scale>::value) {
            stan::math::partials<3>(partials) = z_values.dot(partial_of_eta);
        }
        if constexpr (!stan::is_constant_all<T_z>::value) {
            stan::math::partials<4>(partials) = scale_value * partial_of_eta;
        }
        if constexpr (!stan::is_constant_all<T_offset>::value) {
            stan::math::partials<5>(partials) = offset_share * running_adjoint;
        }
        return partials.build(value);
    }

}

}  // namespace nico::math

using nico::math::nico_dot_products_of_affine_predictor_and_log_cumulative_sum_exp;



// Normal density with log(sigma[i]) = intercept + slope * predictor[i].
// This fuses an affine log-scale transformation, not a state-space or volatility model.
namespace nico::math {

template <bool propto = false, typename T_y, typename T_mu, typename T_intercept,
          typename T_slope, typename T_predictor>
inline stan::return_type_t<T_y, T_mu, T_intercept, T_slope, T_predictor>
nico_normal_affine_log_scale_lpdf(
        const T_y& y, const T_mu& mu, const T_intercept& intercept,
        const T_slope& slope, const T_predictor& predictor, std::ostream* /*pstream__*/ = nullptr) {

    static const char* function = "nico_normal_affine_log_scale_lpdf";
    const auto& y_ref = stan::math::to_ref(y);
    const auto& mu_ref = stan::math::to_ref(mu);
    const auto& predictor_ref = stan::math::to_ref(predictor);
    stan::math::check_consistent_sizes(function, "y", y_ref, "mu", mu_ref, "predictor", predictor_ref);
    stan::math::check_finite(function, "y", stan::math::value_of(y_ref));
    stan::math::check_finite(function, "mu", stan::math::value_of(mu_ref));
    stan::math::check_finite(function, "intercept", stan::math::value_of(intercept));
    stan::math::check_finite(function, "slope", stan::math::value_of(slope));
    const int n = static_cast<int>(predictor_ref.size());
    if (n == 0) return 0.0;
    constexpr bool include_quadratic = stan::math::include_summand<propto, T_y, T_mu,
                                                   T_intercept, T_slope, T_predictor>::value;
    constexpr bool include_log_scale = stan::math::include_summand<propto, T_intercept,
                                                               T_slope, T_predictor>::value;
    if constexpr (!include_quadratic) return 0.0;

    namespace lcse = nico::math::log_cumulative_sum_exp;
    //// simd_t holds BMVP_LANES doubles: four with AVX2, eight with AVX-512.
#if BMVP_LANES == 8
#  define NICOSIMD_AFFINE_ALIGN ALIGN64
#else
#  define NICOSIMD_AFFINE_ALIGN ALIGN32
#endif
    const auto add = [](const simd_t left, const simd_t right) {

        NICOSIMD_AFFINE_ALIGN const simd_t left_aligned = left;
        NICOSIMD_AFFINE_ALIGN const simd_t right_aligned = right;
#if BMVP_LANES == 8
        return _mm512_add_pd(left_aligned, right_aligned);
#else
        return _mm256_add_pd(left_aligned, right_aligned);
#endif

    };
    const auto subtract = [](const simd_t left, const simd_t right) {

        NICOSIMD_AFFINE_ALIGN const simd_t left_aligned = left;
        NICOSIMD_AFFINE_ALIGN const simd_t right_aligned = right;
#if BMVP_LANES == 8
        return _mm512_sub_pd(left_aligned, right_aligned);
#else
        return _mm256_sub_pd(left_aligned, right_aligned);
#endif

    };
    const auto sum_packet = [](const simd_t packet) {

        NICOSIMD_AFFINE_ALIGN const simd_t packet_aligned = packet;
#if BMVP_LANES == 8
        return _mm512_reduce_add_pd(packet_aligned);
#else
        const __m128d pair = _mm_add_pd(_mm256_castpd256_pd128(packet_aligned),
                                       _mm256_extractf128_pd(packet_aligned, 1));
        return _mm_cvtsd_f64(_mm_add_sd(pair, _mm_unpackhi_pd(pair, pair)));
#endif

    };
    const auto remove_padding = [](const simd_t packet, const int lanes) {

        if (lanes == BMVP_LANES) return packet;
        alignas(64) double values[BMVP_LANES];
        simd_store(values, packet);
        return lcse::nico_load_prefix_packet(values, lanes, 0.0);

    };
    const auto load_values = [](const auto& values, const int first, const int lanes) {

        using values_type = std::decay_t<decltype(values)>;
        if constexpr (!stan::is_eigen<values_type>::value) {
            return scan_set1(values);
        } else {
            if constexpr ((values_type::Flags & Eigen::DirectAccessBit) != 0) {
                if (values.innerStride() == 1) {
                    return lcse::nico_load_prefix_packet(values.data() + first, lanes, 0.0);
                }
            }
            alignas(64) double packet_values[BMVP_LANES] = {};
            for (int lane = 0; lane < lanes; ++lane) packet_values[lane] = values.coeff(first + lane);
            return simd_load(packet_values);
        }

    };
    const auto y_values = stan::math::as_value_column_array_or_scalar(y_ref);
    const auto mu_values = stan::math::as_value_column_array_or_scalar(mu_ref);
    decltype(auto) predictor_values = stan::math::value_of(predictor_ref);
    stan::math::check_finite(function, "predictor", predictor_values);
    const double intercept_value = stan::math::value_of(intercept);
    const double slope_value = stan::math::value_of(slope);
    Eigen::Matrix<double, -1, 1> partial_of_y;
    Eigen::Matrix<double, -1, 1> partial_of_predictor;
    constexpr bool vector_y_partials =
        (!stan::is_constant_all<T_y>::value && !stan::is_stan_scalar<T_y>::value)
        || (!stan::is_constant_all<T_mu>::value && !stan::is_stan_scalar<T_mu>::value);
    constexpr bool scalar_y_partials =
        (!stan::is_constant_all<T_y>::value && stan::is_stan_scalar<T_y>::value)
        || (!stan::is_constant_all<T_mu>::value && stan::is_stan_scalar<T_mu>::value);
    if constexpr (vector_y_partials) partial_of_y.resize(n);
    if constexpr (!stan::is_constant_all<T_predictor>::value) partial_of_predictor.resize(n);
    NICOSIMD_AFFINE_ALIGN simd_t value_sum = scan_zero();
    NICOSIMD_AFFINE_ALIGN simd_t partial_y_sum = scan_zero();
    NICOSIMD_AFFINE_ALIGN simd_t partial_intercept_sum = scan_zero();
    NICOSIMD_AFFINE_ALIGN simd_t partial_slope_sum = scan_zero();
    for (int first = 0; first < n; first += BMVP_LANES) {
        const int lanes = std::min(BMVP_LANES, n - first);
        NICOSIMD_AFFINE_ALIGN const simd_t predictor_packet = load_values(predictor_values, first, lanes);
        NICOSIMD_AFFINE_ALIGN const simd_t log_scale = scan_fmadd(scan_set1(slope_value), predictor_packet,
                                           scan_set1(intercept_value));
        NICOSIMD_AFFINE_ALIGN const simd_t inverse_scale =
            BMVP_KERNEL_NAME(nico_exp_1)(scan_mul(scan_set1(-1.0), log_scale));
        NICOSIMD_AFFINE_ALIGN const simd_t y_packet = load_values(y_values, first, lanes);
        NICOSIMD_AFFINE_ALIGN const simd_t mu_packet = load_values(mu_values, first, lanes);
        NICOSIMD_AFFINE_ALIGN const simd_t residual = scan_mul(subtract(y_packet, mu_packet), inverse_scale);
        NICOSIMD_AFFINE_ALIGN const simd_t squared_residual = scan_mul(residual, residual);
        NICOSIMD_AFFINE_ALIGN simd_t contribution = scan_mul(scan_set1(-0.5), squared_residual);
        if constexpr (include_log_scale) contribution = subtract(contribution, log_scale);
        value_sum = add(value_sum, remove_padding(contribution, lanes));
        NICOSIMD_AFFINE_ALIGN const simd_t partial_log_scale = subtract(squared_residual,
                                                  scan_set1(include_log_scale ? 1.0 : 0.0));
        if constexpr (!stan::is_constant_all<T_y, T_mu>::value) {
            NICOSIMD_AFFINE_ALIGN const simd_t partial_y =
                scan_mul(scan_set1(-1.0), scan_mul(residual, inverse_scale));
            if constexpr (vector_y_partials) {
                lcse::nico_store_prefix_packet(partial_of_y.data() + first, lanes, partial_y);
            }
            if constexpr (scalar_y_partials) {
                partial_y_sum = add(partial_y_sum, remove_padding(partial_y, lanes));
            }
        }
        if constexpr (!stan::is_constant_all<T_intercept>::value) {
            partial_intercept_sum = add(partial_intercept_sum, remove_padding(partial_log_scale, lanes));
        }
        if constexpr (!stan::is_constant_all<T_slope>::value) {
            partial_slope_sum = add(partial_slope_sum,
                                    remove_padding(scan_mul(predictor_packet, partial_log_scale), lanes));
        }
        if constexpr (!stan::is_constant_all<T_predictor>::value) {
            lcse::nico_store_prefix_packet(partial_of_predictor.data() + first, lanes,
                                           scan_mul(scan_set1(slope_value), partial_log_scale));
        }
    }
    double value = sum_packet(value_sum);
    if constexpr (!propto) value -= n * 0.91893853320467274178;
    if constexpr (stan::is_constant_all<T_y, T_mu, T_intercept, T_slope, T_predictor>::value) {
        return value;
    } else {
        const double summed_partial_of_y = sum_packet(partial_y_sum);
        const double partial_of_intercept = sum_packet(partial_intercept_sum);
        const double partial_of_slope = sum_packet(partial_slope_sum);
        if constexpr (stan::is_var<stan::return_type_t<T_y, T_mu, T_intercept,
                                                      T_slope, T_predictor>>::value) {

            auto& arena = stan::math::ChainableStack::instance_->memalloc_;
            //// Only adjoint addresses are gathered element by element; all derivative arithmetic is SIMD.
            const auto collect_adjoints = [&arena](const auto& operand) -> double** {
                using operand_type = std::decay_t<decltype(operand)>;
                if constexpr (stan::is_constant_all<operand_type>::value) {
                    return nullptr;
                } else {
                    const int count = static_cast<int>(stan::math::size(operand));
                    double** targets = arena.template alloc_array<double*>(count);
                    if constexpr (stan::is_var_matrix<operand_type>::value) {
                        for (int i = 0; i < count; ++i) targets[i] = &operand.adj().coeffRef(i);
                    } else {
                        stan::scalar_seq_view<operand_type> elements(operand);
                        for (int i = 0; i < count; ++i) targets[i] = &elements[i].adj();
                    }
                    return targets;
                }
            };
            double** y_targets = collect_adjoints(y_ref);
            double** mu_targets = collect_adjoints(mu_ref);
            double** intercept_targets = collect_adjoints(intercept);
            double** slope_targets = collect_adjoints(slope);
            double** predictor_targets = collect_adjoints(predictor_ref);
            stan::arena_t<Eigen::Matrix<double, -1, 1>> y_partials = partial_of_y;
            stan::arena_t<Eigen::Matrix<double, -1, 1>> predictor_partials = partial_of_predictor;
            return stan::math::make_callback_var(value,
                [y_targets, mu_targets, intercept_targets, slope_targets, predictor_targets, y_partials,
                 predictor_partials, summed_partial_of_y, partial_of_intercept, partial_of_slope, n]
                (auto& result_node) {

                    NICOSIMD_AFFINE_ALIGN const simd_t upstream = scan_set1(result_node.adj());
                    const auto add_vector_partials = [&](double** targets, const double* partials,
                                                          const simd_t multiplier) {

                        for (int first = 0; first < n; first += BMVP_LANES) {
                            const int lanes = std::min(BMVP_LANES, n - first);
                            lcse::nico_add_prefix_adjoint_packet(targets + first, lanes,
                                lcse::nico_load_prefix_packet(partials + first, lanes, 0.0), multiplier);
                        }

                    };
                    if constexpr (!stan::is_constant_all<T_y>::value) {
                        if constexpr (stan::is_stan_scalar<T_y>::value) {
                            *y_targets[0] += result_node.adj() * summed_partial_of_y;
                        } else add_vector_partials(y_targets, y_partials.data(), upstream);
                    }
                    if constexpr (!stan::is_constant_all<T_mu>::value) {
                        if constexpr (stan::is_stan_scalar<T_mu>::value) {
                            *mu_targets[0] -= result_node.adj() * summed_partial_of_y;
                        } else add_vector_partials(mu_targets, y_partials.data(),
                                                  scan_mul(scan_set1(-1.0), upstream));
                    }
                    if constexpr (!stan::is_constant_all<T_intercept>::value) {
                        *intercept_targets[0] += result_node.adj() * partial_of_intercept;
                    }
                    if constexpr (!stan::is_constant_all<T_slope>::value) {
                        *slope_targets[0] += result_node.adj() * partial_of_slope;
                    }
                    if constexpr (!stan::is_constant_all<T_predictor>::value) {
                        add_vector_partials(predictor_targets, predictor_partials.data(), upstream);
                    }

                });

        } else {
            //// Other autodiff scalar types retain their existing partials-propagator contract.
            auto partials = stan::math::make_partials_propagator(y_ref, mu_ref, intercept, slope, predictor_ref);
            if constexpr (!stan::is_constant_all<T_y>::value) {
                if constexpr (stan::is_eigen<T_y>::value) stan::math::partials<0>(partials) = partial_of_y;
                else stan::math::partials<0>(partials) = summed_partial_of_y;
            }
            if constexpr (!stan::is_constant_all<T_mu>::value) {
                if constexpr (stan::is_eigen<T_mu>::value) stan::math::partials<1>(partials) = -partial_of_y;
                else stan::math::partials<1>(partials) = -summed_partial_of_y;
            }
            if constexpr (!stan::is_constant_all<T_intercept>::value) {
                stan::math::partials<2>(partials) = partial_of_intercept;
            }
            if constexpr (!stan::is_constant_all<T_slope>::value) {
                stan::math::partials<3>(partials) = partial_of_slope;
            }
            if constexpr (!stan::is_constant_all<T_predictor>::value) {
                stan::math::partials<4>(partials) = partial_of_predictor;
            }
            return partials.build(value);
        }
    }

#undef NICOSIMD_AFFINE_ALIGN

}

}  // namespace nico::math

using nico::math::nico_normal_affine_log_scale_lpdf;























