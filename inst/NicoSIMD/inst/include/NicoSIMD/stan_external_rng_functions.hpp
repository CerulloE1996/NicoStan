#pragma once

#include "stan_external_scalar_functions.hpp"
#include "math/nico_double_trigonometric.hpp"
#include <stan/math/prim/meta/VectorBuilder.hpp>
#include <stan/math/prim/err.hpp>
#include <stan/math/prim/fun/max_size.hpp>
#include <stan/math/prim/fun/scalar_seq_view.hpp>
#include <stan/math/prim/meta/StdVectorBuilder.hpp>
#include <stan/math/prim/fun/size_mvt.hpp>
#include <stan/math/prim/fun/as_column_vector_or_scalar.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/exponential_distribution.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/random/detail/int_float_pair.hpp>
#include <algorithm>
#include <cmath>
#include <utility>

namespace nico::math {

// The random generators of Stan's normal_rng, multi_normal_rng, student_t_rng and weibull_rng (boost's ziggurat
// normal and exponential, gamma by rejection, weibull by inversion), with the same draws in the same order; exp,
// log and tan go to the nico kernels, pow and sqrt stay on the standard library.

// one exponential(1) draw: the ziggurat of boost's unit_exponential_distribution, its density exp(-x) on nico_exp
template <class RNG>
inline double nico_unit_exponential_draw(RNG& rng) {

    const double* const table_x = boost::random::detail::exponential_table<double>::table_x;
    const double* const table_y = boost::random::detail::exponential_table<double>::table_y;
    double shift(0);
    for (;;) {

        const std::pair<double, int> vals = boost::random::detail::generate_int_float_pair<double, 8>(rng);
        const int i = vals.second;
        const double x = vals.first * double(table_x[i]);
        if (x < double(table_x[i + 1])) return shift + x;
        if (i == 0) {

            shift += double(table_x[1]);

        } else {

            const double y01 = boost::random::uniform_01<double>()(rng);
            const double y = double(table_y[i]) + y01 * double(table_y[i + 1] - table_y[i]);
            const double y_above_ubound = double(table_x[i] - table_x[i + 1]) * y01 - (double(table_x[i]) - x);
            const double y_above_lbound = y - (double(table_y[i + 1])
                                               + (double(table_x[i + 1]) - x) * double(table_y[i + 1]));
            if (y_above_ubound < 0 && (y_above_lbound < 0 || y < nico_exp_1(-x))) return x + shift;

        }

    }

}

// the tail of the normal ziggurat: exponential(x_1) draws shifted by x_1, accepted when 2 y > x^2
template <class RNG>
inline double nico_normal_tail_draw(RNG& rng) {

    const double tail_start = double(boost::random::detail::normal_table<double>::table_x[1]);
    for (;;) {

        const double x = nico_unit_exponential_draw(rng) / tail_start;
        const double y = nico_unit_exponential_draw(rng);
        if (2 * y > x * x) return x + tail_start;

    }

}

// one standard normal draw: the ziggurat of boost's unit_normal_distribution, its density exp(-x^2 / 2) on nico_exp
template <class RNG>
inline double nico_unit_normal_draw(RNG& rng) {

    const double* const table_x = boost::random::detail::normal_table<double>::table_x;
    const double* const table_y = boost::random::detail::normal_table<double>::table_y;
    for (;;) {

        const std::pair<double, int> vals = boost::random::detail::generate_int_float_pair<double, 8>(rng);
        int i = vals.second;
        const int sign = (i & 1) * 2 - 1;
        i = i >> 1;
        const double x = vals.first * double(table_x[i]);
        if (x < table_x[i + 1]) return x * sign;
        if (i == 0) return nico_normal_tail_draw(rng) * sign;
        const double y01 = boost::random::uniform_01<double>()(rng);
        const double y = double(table_y[i]) + y01 * double(table_y[i + 1] - table_y[i]);
        double y_above_ubound, y_above_lbound;
        if (table_x[i] >= 1) {

            y_above_ubound = double(table_x[i] - table_x[i + 1]) * y01 - (double(table_x[i]) - x);
            y_above_lbound = y - (double(table_y[i])
                                  + (double(table_x[i]) - x) * double(table_y[i]) * double(table_x[i]));

        } else {

            y_above_lbound = double(table_x[i] - table_x[i + 1]) * y01 - (double(table_x[i]) - x);
            y_above_ubound = y - (double(table_y[i])
                                  + (double(table_x[i]) - x) * double(table_y[i]) * double(table_x[i]));

        }
        if (y_above_ubound < 0 && (y_above_lbound < 0 || y < nico_exp_1(-(x * x / 2)))) return x * sign;

    }

}

// one gamma(alpha, beta) draw (beta a scale): boost's gamma_distribution, with tan, exp and log on nico
template <class RNG>
inline double nico_gamma_draw(const double alpha,
                              const double beta,
                              RNG& rng) {

    if (alpha == 1.0) return nico_unit_exponential_draw(rng) * beta;
    if (alpha > 1.0) {

        const double pi = 3.14159265358979323846;
        for (;;) {

            const double y = nico_tan_1(pi * boost::random::uniform_01<double>()(rng));
            const double x = std::sqrt(2.0 * alpha - 1.0) * y + alpha - 1.0;
            if (x <= 0.0) continue;
            if (boost::random::uniform_01<double>()(rng)
                > (1.0 + y * y) * nico_exp_1((alpha - 1.0) * nico_log_1_double_precision(x / (alpha - 1.0))
                                             - std::sqrt(2.0 * alpha - 1.0) * y)) continue;
            return x * beta;

        }

    }
    // e / (alpha + e), e = exp(1) as a constant
    const double e = 2.71828182845904523536;
    const double p = e / (alpha + e);
    for (;;) {

        const double u = boost::random::uniform_01<double>()(rng);
        const double y = nico_unit_exponential_draw(rng);
        double x, q;
        if (u < p) {

            x = nico_exp_1(-y / alpha);
            q = p * nico_exp_1(-x);

        } else {

            x = 1.0 + y;
            q = p + (1.0 - p) * std::pow(x, alpha - 1.0);

        }
        if (u >= q) continue;
        return x * beta;

    }

}

template <typename T_loc, typename T_scale, class RNG>
inline typename stan::VectorBuilder<true, double, T_loc, T_scale>::type nico_normal_rng(
    const T_loc& mu,
    const T_scale& sigma,
    RNG& rng,
    std::ostream* /*pstream__*/) {

    static constexpr const char* function = "nico_normal_rng";
    stan::math::check_consistent_sizes(function, "Location parameter", mu, "Scale Parameter", sigma);
    stan::ref_type_t<T_loc> mu_ref = mu;
    stan::ref_type_t<T_scale> sigma_ref = sigma;
    stan::math::check_finite(function, "Location parameter", mu_ref);
    stan::math::check_positive_finite(function, "Scale parameter", sigma_ref);
    stan::scalar_seq_view<stan::ref_type_t<T_loc>> mu_vec(mu_ref);
    stan::scalar_seq_view<stan::ref_type_t<T_scale>> sigma_vec(sigma_ref);
    const size_t N = stan::math::max_size(mu, sigma);
    stan::VectorBuilder<true, double, T_loc, T_scale> output(N);
    constexpr size_t chunk_size = 4096;
    for (size_t first = 0; first < N; first += chunk_size) {

        const size_t end = std::min(N, first + chunk_size);
        for (size_t n = first; n < end; ++n) {

            output[n] = nico_unit_normal_draw(rng) * double(sigma_vec[n]) + double(mu_vec[n]);

        }

    }
    return output.data();

}

template <typename T_loc, class RNG>
inline typename stan::StdVectorBuilder<true, Eigen::VectorXd, T_loc>::type nico_multi_normal_rng(
    const T_loc& mu,
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& S,
    RNG& rng,
    std::ostream* /*pstream__*/) {

    static constexpr const char* function = "nico_multi_normal_rng";
    stan::math::check_positive(function, "Covariance matrix rows", S.rows());
    stan::vector_seq_view<T_loc> mu_vec(mu);
    const size_t size_mu = mu_vec[0].size();
    const size_t N = stan::math::size_mvt(mu);
    constexpr size_t chunk_size = 4096;
    for (size_t first = 1; first < N; first += chunk_size) {

        const size_t end = std::min(N, first + chunk_size);
        for (size_t i = first; i < end; ++i) {

            stan::math::check_size_match(function, "Size of one of the vectors of the location variable",
                                         mu_vec[i].size(), "Size of the first vector of the location variable",
                                         size_mu);

        }

    }
    for (size_t first = 0; first < N; first += chunk_size) {

        const size_t end = std::min(N, first + chunk_size);
        for (size_t i = first; i < end; ++i) {

            stan::math::check_finite(function, "Location parameter", mu_vec[i]);

        }

    }
    const auto& S_ref = stan::math::to_ref(S);
    stan::math::check_not_nan(function, "Covariance matrix", S_ref);
    stan::math::check_symmetric(function, "Covariance matrix", S_ref);
    Eigen::LLT<Eigen::MatrixXd> llt_of_S = S_ref.llt();
    stan::math::check_pos_definite(function, "covariance matrix argument", llt_of_S);
    stan::StdVectorBuilder<true, Eigen::VectorXd, T_loc> output(N);
    for (size_t first = 0; first < N; first += chunk_size) {

        const size_t end = std::min(N, first + chunk_size);
        for (size_t n = first; n < end; ++n) {

            Eigen::VectorXd z(S.cols());
            for (Eigen::Index first_row = 0; first_row < S.cols(); first_row += chunk_size) {

                const Eigen::Index end_row = std::min<Eigen::Index>(S.cols(), first_row + chunk_size);
                for (Eigen::Index i = first_row; i < end_row; ++i) {

                    z(i) = nico_unit_normal_draw(rng) * 1.0 + 0.0;

                }

            }
            output[n] = stan::math::as_column_vector_or_scalar(mu_vec[n]) + llt_of_S.matrixL() * z;

        }

    }
    return output.data();

}

template <typename T_deg, typename T_loc, typename T_scale, class RNG>
inline typename stan::VectorBuilder<true, double, T_deg, T_loc, T_scale>::type nico_student_t_rng(
    const T_deg& nu,
    const T_loc& mu,
    const T_scale& sigma,
    RNG& rng,
    std::ostream* /*pstream__*/) {

    static constexpr const char* function = "nico_student_t_rng";
    stan::math::check_consistent_sizes(function, "Degrees of freedom parameter", nu, "Location parameter", mu,
                                       "Scale Parameter", sigma);
    stan::ref_type_t<T_deg> nu_ref = nu;
    stan::ref_type_t<T_loc> mu_ref = mu;
    stan::ref_type_t<T_scale> sigma_ref = sigma;
    stan::math::check_positive_finite(function, "Degrees of freedom parameter", nu_ref);
    stan::math::check_finite(function, "Location parameter", mu_ref);
    stan::math::check_positive_finite(function, "Scale parameter", sigma_ref);
    stan::scalar_seq_view<stan::ref_type_t<T_deg>> nu_vec(nu_ref);
    stan::scalar_seq_view<stan::ref_type_t<T_loc>> mu_vec(mu_ref);
    stan::scalar_seq_view<stan::ref_type_t<T_scale>> sigma_vec(sigma_ref);
    const size_t N = stan::math::max_size(nu, mu, sigma);
    stan::VectorBuilder<true, double, T_deg, T_loc, T_scale> output(N);
    constexpr size_t chunk_size = 4096;
    for (size_t first = 0; first < N; first += chunk_size) {

        const size_t end = std::min(N, first + chunk_size);
        for (size_t n = first; n < end; ++n) {

            // A standard normal over sqrt(chi-squared(nu) / nu), with chi-squared(nu) = 2 gamma(nu / 2, 1).
            const double nu_n = double(nu_vec[n]);
            const double numerator = nico_unit_normal_draw(rng) * 1.0 + 0.0;
            const double chi_squared = 2 * nico_gamma_draw(nu_n / 2, 1.0, rng);
            output[n] = double(mu_vec[n]) + double(sigma_vec[n]) * (numerator / std::sqrt(chi_squared / nu_n));

        }

    }
    return output.data();

}

template <typename T_shape, typename T_scale, class RNG>
inline typename stan::VectorBuilder<true, double, T_shape, T_scale>::type nico_weibull_rng(
    const T_shape& alpha,
    const T_scale& sigma,
    RNG& rng,
    std::ostream* /*pstream__*/) {

    static constexpr const char* function = "nico_weibull_rng";
    stan::math::check_consistent_sizes(function, "Shape parameter", alpha, "Scale Parameter", sigma);
    stan::ref_type_t<T_shape> alpha_ref = alpha;
    stan::ref_type_t<T_scale> sigma_ref = sigma;
    stan::math::check_positive_finite(function, "Shape parameter", alpha_ref);
    stan::math::check_positive_finite(function, "Scale parameter", sigma_ref);
    stan::scalar_seq_view<stan::ref_type_t<T_shape>> alpha_vec(alpha_ref);
    stan::scalar_seq_view<stan::ref_type_t<T_scale>> sigma_vec(sigma_ref);
    const size_t N = stan::math::max_size(alpha, sigma);
    stan::VectorBuilder<true, double, T_shape, T_scale> output(N);
    // Draw uniforms in order and transform one chunk at a time.
    constexpr size_t chunk_size = 4096;
    Eigen::VectorXd one_minus_u(std::min(N, chunk_size));
    for (size_t first = 0; first < N; first += chunk_size) {

        const size_t count = std::min(chunk_size, N - first);
        one_minus_u.resize(count);
        for (size_t row = 0; row < count; ++row) {

            one_minus_u.coeffRef(row) = 1 - boost::random::uniform_01<double>()(rng);

        }
        const Eigen::VectorXd log_one_minus_u = nico_log(one_minus_u, nullptr);
        for (size_t row = 0; row < count; ++row) {

            const size_t n = first + row;
            output[n] = double(sigma_vec[n]) * std::pow(-log_one_minus_u.coeff(row), 1 / double(alpha_vec[n]));

        }

    }
    return output.data();

}

}  // namespace nico::math

using nico::math::nico_normal_rng;
using nico::math::nico_multi_normal_rng;
using nico::math::nico_student_t_rng;
using nico::math::nico_weibull_rng;
