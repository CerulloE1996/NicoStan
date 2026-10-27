#pragma once
//
// -| --------- BayesMVP fused Student-t log density, exposed to Stan as an external function ------------------
//
// nico_student_t_lpdf(y | nu, mu, sigma): a drop-in replacement for Stan's student_t_lpdf(y | nu, mu, sigma),
// with the same arguments and the same log density: y, nu, mu and sigma each a scalar or a vector (vector,
// row_vector or array[] real; the vectors of one length), of data (double or int) or of parameters (var).
// ONE pass over the observations, with the BayesMVP SIMD kernels (nico_log1p_1 and
// nico_log_1 of nico_and_approx_AVX512_fns.hpp / nico_and_approx_AVX2_fns.hpp, as selected in
// stan_external_functions.hpp), computes the log density and every partial derivative which the argument
// types need:
//
//     * all arguments data          -> double (the value only)
//     * any argument a parameter    -> var (ONE autodiff node)
//
// With a parameter argument, the operand vari pointers of each vector parameter are copied to the arena and
// their values gathered (one loop, before the pass); the pass stores on the arena the partial derivatives with
// respect to the vector parameters and sums those with respect to the scalar parameters. The reverse pass is
// one callback, which adds adj(log density) * partial derivative to the adjoint of every operand (scalar and
// in order, since elements of a vector may share one vari). Stan's
// own student_t_lpdf also builds one node, but evaluates the value and the partial derivatives as separate
// elementwise passes with temporaries (with up to five divisions per observation), and log1p in scalar code.
//
// With z_i = (y_i - mu_i) / sigma_i, r_i = z_i^2 / nu_i and w_i = 1 / (1 + r_i) (one division per observation):
//
//     log p_i       = lgamma((nu_i + 1) / 2) - lgamma(nu_i / 2) - 0.5 * log(nu_i) - log(sqrt(pi))
//                     - log(sigma_i) - ((nu_i + 1) / 2) * log1p(r_i)
//     d / d mu_i    = (nu_i + 1) * z_i * w_i / (nu_i * sigma_i)                   ( = - d / d y_i)
//     d / d sigma_i = (nu_i - (nu_i + 1) * w_i) / sigma_i
//     d / d nu_i    = 0.5 * (digamma((nu_i + 1) / 2) - digamma(nu_i / 2) + 1 - log1p(r_i)
//                            - (nu_i + 1) * w_i / nu_i)
//
// i.e., the partial derivatives of stan::math::student_t_lpdf, with (nu + 1) * r / (1 + r) - 1
// = nu - (nu + 1) * w. A scalar argument's partial derivative is the sum over the observations, e.g., with nu
// and sigma scalars, d / d sigma = (N * nu - (nu + 1) * sum_i w_i) / sigma. The terms in nu alone (lgamma,
// log(nu) and digamma) are computed once for a scalar nu and once per element for a vector nu (after the pass).
// Scalar values use the native kernels; derivatives with respect to nu keep their separate path.
//
// propto = false: every term of stan::math::student_t_lpdf<false>. propto = true: as in Stan, the terms
// without a parameter are dropped (log(sqrt(pi)); the terms in nu alone when nu is data; log(sigma) when sigma
// is data), and the result is 0 when no argument is a parameter.
//
// Checks as in Stan, with Stan's own check functions (and so the same exceptions, under the name
// nico_student_t_lpdf): consistent sizes, y not NaN, nu positive finite, mu finite, sigma positive finite. The
// pass flags every element of a vector nu or sigma which is not positive finite, and a NaN y or a non-finite mu
// always makes the log density non-finite; only then are the check functions run, in Stan's order (so that the
// exception is the one Stan raises).
//
// Accuracy: that of the log1p and log kernels, in the log density and in d / d nu; the partial derivatives
// with respect to y, mu and sigma do not use the kernels.
//
// Declared in the Stan functions block (no body; stanc --allow-undefined), once per combination of argument
// types used, e.g.
//
//     real nico_student_t_lpdf(vector y, real nu, vector mu, real sigma);
//     real nico_student_t_lpdf(array[] real y, real nu, vector mu, real sigma);
//
// which also gives  nico_student_t_lupdf(y | nu, mu, sigma)  and  y ~ nico_student_t(nu, mu, sigma) (Stan's
// student_t_lpdf takes reals for each argument: real, array[] real, vector or row_vector).
//
// Included at the end of stan_external_functions.hpp (it uses that file's namespace nico::math, simd_t,
// BMVP_LANES, BMVP_KERNEL_NAME and its kernels). It also includes stan_external_functions.hpp itself
// (#pragma once), so a user header may include this file alone. Its helpers are in their own namespace,
// nico::math::fused_student_t, so that other fused headers included the same way cannot clash.
//
// -----------------------------------------------------------------------------------------------------------

#include "stan_external_functions.hpp"
#include "stan_external_lgamma.hpp"
#include "math/nico_double_exp_log.hpp"

#include <stan/math/rev.hpp>
#include <stan/math/prim.hpp>

#include <Eigen/Dense>
#include <cmath>
#include "nico_simd_intrinsics.hpp"   //// <immintrin.h> on x86-64; SIMDe (NEON) on ARM
#include <limits>
#include <ostream>
#include <type_traits>
#include <vector>

#if !defined(BMVP_LANES) || (!defined(BMVP_KERNEL_NAME) && !defined(NICO_SIMD_SCALAR_KERNELS))
#  error "stan_external_fused_student_t_lpdf.hpp: stan_external_functions.hpp must be included first"
#endif

namespace nico::math {
namespace fused_student_t {




//// ---- the SIMD operations of the pass, for the 8-lane (AVX-512) and the 4-lane (AVX2) kernels. lane_mask_t
////      holds the lanes in use: all of them, except in the last block of a vector whose length is not a
////      multiple of BMVP_LANES (its spare lanes are loaded as 1.0, never stored, and left out of every sum).
#if defined(NICO_SIMD_SCALAR_KERNELS)

using lane_mask_t = bool;

inline simd_t simd_set1(const double x) {

    return x;

}
inline simd_t simd_zero() {

    return 0.0;

}
inline simd_t simd_add(const simd_t a,
                       const simd_t b) {

    return a + b;

}
inline simd_t simd_sub(const simd_t a,
                       const simd_t b) {

    return a - b;

}
inline simd_t simd_mul(const simd_t a,
                       const simd_t b) {

    return a * b;

}
inline simd_t simd_div(const simd_t a,
                       const simd_t b) {

    return a / b;

}
inline simd_t simd_fmadd(const simd_t a,
                         const simd_t b,
                         const simd_t c) {

    return std::fma(a, b, c);

}
inline simd_t simd_fnmadd(const simd_t a,
                          const simd_t b,
                          const simd_t c) {

    return std::fma(-a, b, c);

}
inline double sum_of_lanes(const simd_t x) {

    return x;

}
inline lane_mask_t first_lanes(const int n_lanes) {

    return n_lanes > 0;

}
inline lane_mask_t no_lanes() {

    return false;

}
inline bool any_lane(const lane_mask_t m) {

    return m;

}
inline lane_mask_t lanes_or(const lane_mask_t a,
                            const lane_mask_t b) {

    return a || b;

}
inline lane_mask_t lanes_not_positive_finite(const simd_t x) {

        return !(x > 0.0) || !(x < std::numeric_limits<double>::infinity());

}
inline simd_t simd_add_in_lanes(const simd_t x,
                                const simd_t a,
                                const lane_mask_t lanes) {

        return x + (lanes ? a : 0.0);

}
inline simd_t simd_load_lanes(const double* p,
                              const lane_mask_t lanes) {

    return lanes ? p[0] : 1.0;

}
inline void simd_store_lanes(double* p,
                             const simd_t x,
                             const lane_mask_t lanes) {

    if (lanes) p[0] = x;

}

#elif BMVP_LANES == 8

using lane_mask_t = __mmask8;

inline simd_t simd_set1(const double x)                    { return _mm512_set1_pd(x); }
inline simd_t simd_zero()                                  { return _mm512_setzero_pd(); }
inline simd_t simd_add(const simd_t a, const simd_t b)     { return _mm512_add_pd(a, b); }
inline simd_t simd_sub(const simd_t a, const simd_t b)     { return _mm512_sub_pd(a, b); }
inline simd_t simd_mul(const simd_t a, const simd_t b)     { return _mm512_mul_pd(a, b); }
inline simd_t simd_div(const simd_t a, const simd_t b)     { return _mm512_div_pd(a, b); }
//// a * b + c, and c - a * b
inline simd_t simd_fmadd(const simd_t a, const simd_t b, const simd_t c)  { return _mm512_fmadd_pd(a, b, c); }
inline simd_t simd_fnmadd(const simd_t a, const simd_t b, const simd_t c) { return _mm512_fnmadd_pd(a, b, c); }
inline double sum_of_lanes(const simd_t x)                 { return _mm512_reduce_add_pd(x); }

inline lane_mask_t first_lanes(const int n_lanes)          { return static_cast<__mmask8>((1u << n_lanes) - 1u); }
inline lane_mask_t no_lanes()                              { return 0; }
inline bool        any_lane(const lane_mask_t m)           { return m != 0; }
inline lane_mask_t lanes_or(const lane_mask_t a,
                            const lane_mask_t b)           { return a | b; }

//// the lanes of x which are not positive finite (zero, negative, infinite or NaN)
inline lane_mask_t lanes_not_positive_finite(const simd_t x) {

        const simd_t infinity = _mm512_set1_pd(std::numeric_limits<double>::infinity());
        return _mm512_cmp_pd_mask(x, _mm512_setzero_pd(), _CMP_NGT_UQ) |
               _mm512_cmp_pd_mask(x, infinity, _CMP_NLT_UQ);

}

//// x + a in the lanes of the mask; x in the others
inline simd_t simd_add_in_lanes(const simd_t x, const simd_t a, const lane_mask_t lanes) {

        return _mm512_mask_add_pd(x, lanes, x, a);

}

//// load / store of the lanes of the mask only (spare lanes loaded as 1.0)
inline simd_t simd_load_lanes(const double* p, const lane_mask_t lanes) {

        return _mm512_mask_loadu_pd(_mm512_set1_pd(1.0), lanes, p);

}
inline void simd_store_lanes(double* p, const simd_t x, const lane_mask_t lanes) {

        _mm512_mask_storeu_pd(p, lanes, x);

}

#else

using lane_mask_t = __m256d;

inline simd_t simd_set1(const double x)                    { return _mm256_set1_pd(x); }
inline simd_t simd_zero()                                  { return _mm256_setzero_pd(); }
inline simd_t simd_add(const simd_t a, const simd_t b)     { return _mm256_add_pd(a, b); }
inline simd_t simd_sub(const simd_t a, const simd_t b)     { return _mm256_sub_pd(a, b); }
inline simd_t simd_mul(const simd_t a, const simd_t b)     { return _mm256_mul_pd(a, b); }
inline simd_t simd_div(const simd_t a, const simd_t b)     { return _mm256_div_pd(a, b); }
//// a * b + c, and c - a * b
inline simd_t simd_fmadd(const simd_t a, const simd_t b, const simd_t c)  { return _mm256_fmadd_pd(a, b, c); }
inline simd_t simd_fnmadd(const simd_t a, const simd_t b, const simd_t c) { return _mm256_fnmadd_pd(a, b, c); }
inline double sum_of_lanes(const simd_t x) {

        const __m128d pairs = _mm_add_pd(_mm256_castpd256_pd128(x), _mm256_extractf128_pd(x, 1));
        return _mm_cvtsd_f64(_mm_add_sd(pairs, _mm_unpackhi_pd(pairs, pairs)));

}

inline lane_mask_t first_lanes(const int n_lanes) {

        ALIGN32  __m256i const lane_index = _mm256_setr_epi64x(0, 1, 2, 3);
        return _mm256_castsi256_pd(_mm256_cmpgt_epi64(_mm256_set1_epi64x(n_lanes), lane_index));

}
inline lane_mask_t no_lanes()                              { return _mm256_setzero_pd(); }
inline bool        any_lane(const lane_mask_t m)           { return _mm256_movemask_pd(m) != 0; }
inline lane_mask_t lanes_or(const lane_mask_t a,
                            const lane_mask_t b)           { return _mm256_or_pd(a, b); }

//// the lanes of x which are not positive finite (zero, negative, infinite or NaN)
inline lane_mask_t lanes_not_positive_finite(const simd_t x) {

        const simd_t infinity = _mm256_set1_pd(std::numeric_limits<double>::infinity());
        return _mm256_or_pd(_mm256_cmp_pd(x, _mm256_setzero_pd(), _CMP_NGT_UQ),
                            _mm256_cmp_pd(x, infinity, _CMP_NLT_UQ));

}

//// x + a in the lanes of the mask; x in the others
inline simd_t simd_add_in_lanes(const simd_t x, const simd_t a, const lane_mask_t lanes) {

        return _mm256_add_pd(x, _mm256_and_pd(a, lanes));

}

//// load / store of the lanes of the mask only (spare lanes loaded as 1.0)
inline simd_t simd_load_lanes(const double* p, const lane_mask_t lanes) {

        const simd_t loaded = _mm256_maskload_pd(p, _mm256_castpd_si256(lanes));
        return _mm256_blendv_pd(_mm256_set1_pd(1.0), loaded, lanes);

}
inline void simd_store_lanes(double* p, const simd_t x, const lane_mask_t lanes) {

        _mm256_maskstore_pd(p, _mm256_castpd_si256(lanes), x);

}

#endif

#if defined(NICO_SIMD_SCALAR_KERNELS)
inline simd_t simd_log1p(const simd_t x) {

    return nico_log1p_1_double_precision(x);

}
inline simd_t simd_log(const simd_t x) {

    return nico_log_1_double_precision(x);

}
#else
inline simd_t simd_log1p(const simd_t x) { return BMVP_KERNEL_NAME(nico_log1p_1_double_precision)(x); }
inline simd_t simd_log(const simd_t x)   { return BMVP_KERNEL_NAME(nico_log_1_double_precision)(x); }
#endif




//// ---- the arguments: each a vector (Eigen vector or std::vector) or a scalar, of data (double or int) or of
////      parameters (var)
template <typename T>
struct argument_type {

        using T_plain  = std::decay_t<T>;
        using T_scalar = stan::scalar_type_t<T_plain>;

        static constexpr bool is_vector    = stan::is_vector<T_plain>::value;
        static constexpr bool is_parameter = stan::is_var<T_scalar>::value;
        static constexpr bool is_supported = (is_vector || stan::is_stan_scalar<T_plain>::value) &&
                                             (std::is_arithmetic<T_scalar>::value || is_parameter);

};


//// one argument as the pass reads it
struct argument_in_pass_t {

        const double* values = nullptr;            //// a vector: its values (data in place or copied; parameters
                                                   ////   gathered into a buffer)
        stan::math::vari** nodes = nullptr;        //// a parameter vector: its vari pointers, on the arena
        double scalar_value = 0.0;                 //// a scalar: its value,
        stan::math::vari* scalar_node = nullptr;   ////   and, for a parameter, its vari

};




//// ---- thread-local buffers for the values of the parameter vectors and of the data vectors which are not
////      stored contiguously as doubles (reused across calls, so no allocation after warm-up); each grows to the
////      longest vector it has held. Only the forward pass uses them: the reverse pass reads the arena only.
struct value_buffers_t {

        std::vector<double> y, nu, mu, sigma;

};

inline value_buffers_t& value_buffers() { thread_local value_buffers_t buffers; return buffers; }

inline double* buffer_of_length( std::vector<double>& buffer,
                                 const int N
) {

        if (static_cast<int>(buffer.size()) < N) buffer.resize(N);
        return buffer.data();

}




//// ---- the N values of a data vector (double or int): read in place when they are stored contiguously as
//// doubles (Eigen::Matrix<double, -1, 1>, a Map or a column block of one, std::vector<double>), otherwise copied
//// into
////      the buffer (expressions such as rep_vector(nu, N), strided blocks, int vectors)
template <typename T>
inline const double* values_of_data_vector( const T& x,
                                            const int N,
                                            std::vector<double>& buffer
) {

        using T_plain = std::decay_t<T>;

        if constexpr (stan::is_std_vector<T_plain>::value) {

                if constexpr (std::is_same<typename T_plain::value_type, double>::value) {
                        return x.data();
                } else {
                        double* values = buffer_of_length(buffer, N);
                        for (int i = 0; i < N; ++i) values[i] = static_cast<double>(x[i]);
                        return values;
                }

        } else {

                if constexpr (std::is_same<typename T_plain::Scalar, double>::value &&
                              static_cast<bool>(T_plain::Flags & Eigen::DirectAccessBit)) {

                        if (x.innerStride() == 1) return x.data();

                }
                Eigen::Map<Eigen::Matrix<double, -1, 1>> values(buffer_of_length(buffer, N), N);
                if constexpr (stan::is_eigen_col_vector<T_plain>::value) {
                        values = x.template cast<double>();
                } else {
                        values = x.transpose().template cast<double>();
                }
                return values.data();

        }

}




//// ---- the argument which the pass reads: a parameter vector given as an Eigen expression is evaluated once
////      (stan::math::to_ref); any other argument as it is
template <typename T>
inline decltype(auto) argument_for_pass(const T& x) {

        if constexpr (argument_type<T>::is_vector && argument_type<T>::is_parameter) {
                return stan::math::to_ref(x);
        } else {
                return (x);
        }

}




//// ---- one argument as the pass reads it: a data vector's values (in place or copied); a parameter vector's
////      operand vari pointers (copied to the arena) and values (gathered into the buffer); a scalar's value
////      (and vari)
template <typename T>
inline argument_in_pass_t argument_in_pass( const T& x,
                                            const int N,
                                            std::vector<double>& buffer
) {

        argument_in_pass_t argument;
        if constexpr (argument_type<T>::is_vector && argument_type<T>::is_parameter) {
                auto& arena = stan::math::ChainableStack::instance_->memalloc_;
                stan::math::vari** nodes = arena.template alloc_array<stan::math::vari*>(N);
                double* values = buffer_of_length(buffer, N);
                for (int i = 0; i < N; ++i) {
                        stan::math::vari* node;
                        if constexpr (stan::is_std_vector<std::decay_t<T>>::value) node = x[i].vi_;
                        else                                                       node = x.coeff(i).vi_;
                        nodes[i]  = node;
                        values[i] = node->val_;
                }
                argument.nodes  = nodes;
                argument.values = values;
        } else if constexpr (argument_type<T>::is_vector) {
                argument.values = values_of_data_vector(x, N, buffer);
        } else {
                argument.scalar_value = stan::math::value_of(x);
                if constexpr (argument_type<T>::is_parameter) argument.scalar_node = x.vi_;
        }
        return argument;

}




//// ---- the lanes from observation `first` of a vector argument (in_last_block: the lanes of the mask only)
template <bool in_last_block>
inline simd_t lanes_of_vector_argument( const argument_in_pass_t& argument,
                                        const int first,
                                        const lane_mask_t lanes
) {

        simd_t x;
        if constexpr (in_last_block) x = simd_load_lanes(argument.values + first, lanes);
        else                         x = simd_loadu(argument.values + first);
        return x;

}




//// ---- the log density, and (with any parameter argument) its single autodiff node: see the header comment
template <bool propto, typename T_y, typename T_nu, typename T_mu, typename T_sigma>
inline stan::return_type_t<T_y, T_nu, T_mu, T_sigma> student_t_lpdf( const T_y& y,
                                                                     const T_nu& nu,
                                                                     const T_mu& mu,
                                                                     const T_sigma& sigma
) {

        using return_t = stan::return_type_t<T_y, T_nu, T_mu, T_sigma>;
        static constexpr const char* function = "nico_student_t_lpdf";
        static constexpr double infinity = std::numeric_limits<double>::infinity();

        //// ---- the argument types:
        constexpr bool y_is_parameter     = argument_type<T_y>::is_parameter;
        constexpr bool nu_is_parameter    = argument_type<T_nu>::is_parameter;
        constexpr bool mu_is_parameter    = argument_type<T_mu>::is_parameter;
        constexpr bool sigma_is_parameter = argument_type<T_sigma>::is_parameter;
        constexpr bool y_is_vector        = argument_type<T_y>::is_vector;
        constexpr bool nu_is_vector       = argument_type<T_nu>::is_vector;
        constexpr bool mu_is_vector       = argument_type<T_mu>::is_vector;
        constexpr bool sigma_is_vector    = argument_type<T_sigma>::is_vector;
        constexpr bool any_parameter      = y_is_parameter || nu_is_parameter ||
                                            mu_is_parameter || sigma_is_parameter;

        //// ---- the terms of the log density (as in Stan's student_t_lpdf):
        using stan::math::include_summand;
        constexpr bool include_log_sqrt_pi_term  = include_summand<propto>::value;
        constexpr bool include_terms_in_nu_alone = include_summand<propto, T_nu>::value;
        constexpr bool include_log_sigma_term    = include_summand<propto, T_sigma>::value;
        constexpr bool include_log1p_term        = include_summand<propto, T_y, T_nu, T_mu, T_sigma>::value;

        //// ---- what the pass stores per observation (on the arena) and what it sums:
        ////      d_location = d / d mu_i (= - d / d y_i), for a parameter vector y or mu (stored), or a scalar
        ////      parameter y or mu (summed); sum_w = sum_i w_i, for d / d nu (scalar parameter nu), and for
        ////      d / d sigma (scalar nu and scalar parameter sigma)
        constexpr bool store_d_location = (y_is_parameter && y_is_vector) || (mu_is_parameter && mu_is_vector);
        constexpr bool sum_d_location   = (y_is_parameter && !y_is_vector) || (mu_is_parameter && !mu_is_vector);
        constexpr bool store_d_sigma    = sigma_is_parameter && sigma_is_vector;
        constexpr bool sum_sigma_terms  = sigma_is_parameter && !sigma_is_vector && nu_is_vector;
        constexpr bool store_d_nu       = nu_is_parameter && nu_is_vector;
        constexpr bool sum_w            = !nu_is_vector &&
                                          (nu_is_parameter || (sigma_is_parameter && !sigma_is_vector));

        stan::math::check_consistent_sizes(function, "Random variable", y, "Degrees of freedom parameter", nu,
                                           "Location parameter", mu, "Scale parameter", sigma);
        const int N = static_cast<int>(stan::math::max_size(y, nu, mu, sigma));

        //// ---- an argument of size 0, or no term (propto = true and no parameter): 0, after Stan's checks
        if (stan::math::size_zero(y, nu, mu, sigma) || !include_log1p_term) {
                stan::math::check_not_nan(function, "Random variable", y);
                stan::math::check_positive_finite(function, "Degrees of freedom parameter", nu);
                stan::math::check_finite(function, "Location parameter", mu);
                stan::math::check_positive_finite(function, "Scale parameter", sigma);
                return return_t(0.0);
        }

        //// ---- the arguments as the pass reads them (see argument_in_pass)
        const auto& y_for_pass     = argument_for_pass(y);
        const auto& nu_for_pass    = argument_for_pass(nu);
        const auto& mu_for_pass    = argument_for_pass(mu);
        const auto& sigma_for_pass = argument_for_pass(sigma);
        value_buffers_t& buffers = value_buffers();
        const argument_in_pass_t y_in_pass     = argument_in_pass(y_for_pass, N, buffers.y);
        const argument_in_pass_t nu_in_pass    = argument_in_pass(nu_for_pass, N, buffers.nu);
        const argument_in_pass_t mu_in_pass    = argument_in_pass(mu_for_pass, N, buffers.mu);
        const argument_in_pass_t sigma_in_pass = argument_in_pass(sigma_for_pass, N, buffers.sigma);
        const double y_scalar     = y_in_pass.scalar_value;
        const double nu_scalar    = nu_in_pass.scalar_value;
        const double mu_scalar    = mu_in_pass.scalar_value;
        const double sigma_scalar = sigma_in_pass.scalar_value;

        //// ---- Stan's checks (run only when an argument may be invalid: they raise Stan's exception for the
        ////      first invalid argument, in Stan's order)
        auto check_arguments_as_Stan = [&]() {

                stan::math::check_not_nan(function, "Random variable", y_for_pass);
                stan::math::check_positive_finite(function, "Degrees of freedom parameter", nu_for_pass);
                stan::math::check_finite(function, "Location parameter", mu_for_pass);
                stan::math::check_positive_finite(function, "Scale parameter", sigma_for_pass);

        };

        bool scalar_arguments_are_valid = true;
        if constexpr (!y_is_vector)     scalar_arguments_are_valid &= !std::isnan(y_scalar);
        if constexpr (!nu_is_vector)    scalar_arguments_are_valid &= (nu_scalar > 0.0 && nu_scalar < infinity);
        if constexpr (!mu_is_vector)    scalar_arguments_are_valid &= std::isfinite(mu_scalar);
        if constexpr (!sigma_is_vector) scalar_arguments_are_valid &= (sigma_scalar > 0.0 &&
                                                                      sigma_scalar < infinity);

        //// ---- the partial derivatives stored per observation, on the arena
        auto& arena = stan::math::ChainableStack::instance_->memalloc_;
        double* d_location_values = nullptr;
        double* d_sigma_values    = nullptr;
        double* d_nu_values       = nullptr;
        if constexpr (store_d_location) d_location_values = arena.template alloc_array<double>(N);
        if constexpr (store_d_sigma)    d_sigma_values    = arena.template alloc_array<double>(N);
        if constexpr (store_d_nu)       d_nu_values       = arena.template alloc_array<double>(N);

        //// ---- the SIMD pass
        const simd_t one  = simd_set1(1.0);
        const simd_t half = simd_set1(0.5);
        const simd_t y_scalar_lanes         = simd_set1(y_scalar);
        const simd_t mu_scalar_lanes        = simd_set1(mu_scalar);
        const simd_t sigma_scalar_lanes     = simd_set1(sigma_scalar);
        const simd_t inv_sigma_scalar_lanes = simd_set1(1.0 / sigma_scalar);
        const simd_t nu_scalar_lanes        = simd_set1(nu_scalar);
        const simd_t inv_nu_scalar_lanes    = simd_set1(1.0 / nu_scalar);
        //// (nu + 1) / (nu * sigma): the factor of d / d mu_i when nu and sigma are scalars
        const simd_t d_location_factor_scalar_lanes = simd_set1((nu_scalar + 1.0) / (nu_scalar * sigma_scalar));

        //// the sums over the observations, per lane:
        simd_t sum_log1p_lanes          = simd_zero();   //// log1p(r_i)                  (scalar nu)
        simd_t sum_weighted_log1p_lanes = simd_zero();   //// (nu_i + 1) / 2 * log1p(r_i) (vector nu)
        simd_t sum_log_sigma_lanes      = simd_zero();   //// log(sigma_i)                (vector sigma)
        simd_t sum_d_location_lanes     = simd_zero();   //// d / d mu_i                  (scalar parameter y, mu)
        simd_t sum_w_lanes              = simd_zero();   //// w_i
        simd_t sum_sigma_terms_lanes    = simd_zero();   //// nu_i - (nu_i + 1) * w_i     (scalar parameter sigma)
        lane_mask_t invalid_lanes       = no_lanes();    //// elements of a vector nu or sigma not positive finite

        //// one block of BMVP_LANES observations, from observation `first` (last_block: a shorter last block, in
        //// the lanes of the mask `lanes` only)
        auto one_block = [&](auto last_block, const int first, const lane_mask_t lanes) {

                constexpr bool is_last_block = decltype(last_block)::value;

                auto store = [&](double* values, const simd_t x) {

                        if constexpr (is_last_block) simd_store_lanes(values + first, x, lanes);
                        else                         simd_storeu(values + first, x);

                };
                auto add_to_sum = [&](simd_t& sum, const simd_t x) {

                        if constexpr (is_last_block) sum = simd_add_in_lanes(sum, x, lanes);
                        else                         sum = simd_add(sum, x);

                };

                simd_t y_lanes = y_scalar_lanes;
                if constexpr (y_is_vector) {
                        y_lanes = lanes_of_vector_argument<is_last_block>(y_in_pass, first, lanes);
                }
                simd_t mu_lanes = mu_scalar_lanes;
                if constexpr (mu_is_vector) {
                        mu_lanes = lanes_of_vector_argument<is_last_block>(mu_in_pass, first, lanes);
                }
                simd_t sigma_lanes     = sigma_scalar_lanes;
                simd_t inv_sigma_lanes = inv_sigma_scalar_lanes;
                if constexpr (sigma_is_vector) {
                        sigma_lanes     = lanes_of_vector_argument<is_last_block>(sigma_in_pass, first, lanes);
                        inv_sigma_lanes = simd_div(one, sigma_lanes);
                        invalid_lanes   = lanes_or(invalid_lanes, lanes_not_positive_finite(sigma_lanes));
                }
                simd_t nu_lanes     = nu_scalar_lanes;
                simd_t inv_nu_lanes = inv_nu_scalar_lanes;
                if constexpr (nu_is_vector) {
                        nu_lanes      = lanes_of_vector_argument<is_last_block>(nu_in_pass, first, lanes);
                        inv_nu_lanes  = simd_div(one, nu_lanes);
                        invalid_lanes = lanes_or(invalid_lanes, lanes_not_positive_finite(nu_lanes));
                }

                const simd_t z       = simd_mul(simd_sub(y_lanes, mu_lanes), inv_sigma_lanes);
                const simd_t r       = simd_mul(simd_mul(z, z), inv_nu_lanes);
                const simd_t log1p_r = simd_log1p(r);

                if constexpr (nu_is_vector) {
                        add_to_sum(sum_weighted_log1p_lanes, simd_mul(simd_fmadd(half, nu_lanes, half), log1p_r));
                } else {
                        add_to_sum(sum_log1p_lanes, log1p_r);
                }
                if constexpr (sigma_is_vector && include_log_sigma_term) {
                        add_to_sum(sum_log_sigma_lanes, simd_log(sigma_lanes));
                }

                if constexpr (any_parameter) {

                        const simd_t w = simd_div(one, simd_add(one, r));
                        const simd_t nu_plus_one = simd_add(nu_lanes, one);

                        if constexpr (store_d_location || sum_d_location) {
                                simd_t d_location_factor = d_location_factor_scalar_lanes;
                                if constexpr (nu_is_vector || sigma_is_vector) {
                                        d_location_factor = simd_mul(simd_mul(nu_plus_one, inv_nu_lanes),
                                                                     inv_sigma_lanes);
                                }
                                const simd_t d_location = simd_mul(simd_mul(d_location_factor, z), w);
                                if constexpr (store_d_location) store(d_location_values, d_location);
                                if constexpr (sum_d_location)   add_to_sum(sum_d_location_lanes, d_location);
                        }
                        if constexpr (sum_w) add_to_sum(sum_w_lanes, w);
                        if constexpr (store_d_sigma) {
                                const simd_t sigma_terms = simd_fnmadd(nu_plus_one, w, nu_lanes);
                                store(d_sigma_values, simd_mul(sigma_terms, inv_sigma_lanes));
                        }
                        if constexpr (sum_sigma_terms) {
                                add_to_sum(sum_sigma_terms_lanes, simd_fnmadd(nu_plus_one, w, nu_lanes));
                        }
                        if constexpr (store_d_nu) {
                                //// 0.5 * (1 - log1p(r_i) - (nu_i + 1) * w_i / nu_i); the digamma terms are added
                                //// after the pass
                                const simd_t w_terms = simd_mul(simd_mul(nu_plus_one, inv_nu_lanes), w);
                                store(d_nu_values, simd_fnmadd(half, simd_add(log1p_r, w_terms), half));
                        }

                }

        };

        int first = 0;
        for (; first + BMVP_LANES <= N; first += BMVP_LANES) one_block(std::false_type(), first, no_lanes());
        if (first < N) one_block(std::true_type(), first, first_lanes(N - first));

        //// ---- the log density (the terms of the pass)
        double log_density;
        if constexpr (nu_is_vector) log_density = -sum_of_lanes(sum_weighted_log1p_lanes);
        else                        log_density = -(0.5 * nu_scalar + 0.5) * sum_of_lanes(sum_log1p_lanes);
        if constexpr (include_log_sigma_term) {
                if constexpr (sigma_is_vector) log_density -= sum_of_lanes(sum_log_sigma_lanes);
                else                           log_density -= N * nico_log_1_double_precision(sigma_scalar);
        }

        //// ---- an invalid argument: Stan's exception (a non-finite log density with valid arguments, e.g., at
        ////      an infinite y, is returned as it is, as in Stan)
        if (!scalar_arguments_are_valid || any_lane(invalid_lanes) || !std::isfinite(log_density)) {
                check_arguments_as_Stan();
        }

        //// ---- the terms in nu alone, once for a scalar nu and once per element for a vector nu
        if constexpr (include_log_sqrt_pi_term) log_density -= stan::math::LOG_SQRT_PI * N;
        if constexpr (!nu_is_vector && include_terms_in_nu_alone) {
                const double half_nu = 0.5 * nu_scalar;
                log_density += N * (nico_lgamma(half_nu + 0.5, nullptr) - nico_lgamma(half_nu, nullptr)
                                    - 0.5 * nico_log_1_double_precision(nu_scalar));
        }
        if constexpr (nu_is_vector && (include_terms_in_nu_alone || nu_is_parameter)) {
                for (int i = 0; i < N; ++i) {
                        const double nu_i = nu_in_pass.values[i];
                        const double half_nu_i = 0.5 * nu_i;
                        if constexpr (include_terms_in_nu_alone) {
                                log_density += stan::math::lgamma(half_nu_i + 0.5) - stan::math::lgamma(half_nu_i)
                                               - 0.5 * std::log(nu_i);
                        }
                        if constexpr (nu_is_parameter) {
                                d_nu_values[i] += 0.5 * (stan::math::digamma(half_nu_i + 0.5)
                                                         - stan::math::digamma(half_nu_i));
                        }
                }
        }

        if constexpr (!any_parameter) {

                return log_density;

        } else {

                //// ---- the partial derivatives with respect to the scalar parameters (sums over observations;
                ////      d / d y = - d / d mu)
                double d_mu_total    = 0.0;
                double d_sigma_total = 0.0;
                double d_nu_total    = 0.0;
                if constexpr (sum_d_location) d_mu_total = sum_of_lanes(sum_d_location_lanes);
                if constexpr (sigma_is_parameter && !sigma_is_vector) {
                        if constexpr (nu_is_vector) {
                                d_sigma_total = sum_of_lanes(sum_sigma_terms_lanes) / sigma_scalar;
                        } else {
                                const double sum_w_total = sum_of_lanes(sum_w_lanes);
                                d_sigma_total = (N * nu_scalar - (nu_scalar + 1.0) * sum_w_total) / sigma_scalar;
                        }
                }
                if constexpr (nu_is_parameter && !nu_is_vector) {
                        const double half_nu = 0.5 * nu_scalar;
                        const double digamma_terms = stan::math::digamma(half_nu + 0.5)
                                                     - stan::math::digamma(half_nu) + 1.0;
                        d_nu_total = 0.5 * (N * digamma_terms - sum_of_lanes(sum_log1p_lanes)
                                            - (nu_scalar + 1.0) / nu_scalar * sum_of_lanes(sum_w_lanes));
                }

                //// ---- the single autodiff node: its reverse pass adds adj * partial derivative to each operand
                stan::math::vari** y_nodes     = y_in_pass.nodes;
                stan::math::vari** nu_nodes    = nu_in_pass.nodes;
                stan::math::vari** mu_nodes    = mu_in_pass.nodes;
                stan::math::vari** sigma_nodes = sigma_in_pass.nodes;
                stan::math::vari* y_node       = y_in_pass.scalar_node;
                stan::math::vari* nu_node      = nu_in_pass.scalar_node;
                stan::math::vari* mu_node      = mu_in_pass.scalar_node;
                stan::math::vari* sigma_node   = sigma_in_pass.scalar_node;
                return stan::math::make_callback_var(log_density,
                                                     [y_nodes, nu_nodes, mu_nodes, sigma_nodes,
                                                      y_node, nu_node, mu_node, sigma_node,
                                                      d_location_values, d_sigma_values, d_nu_values,
                                                      d_mu_total, d_sigma_total, d_nu_total, N](auto& result) {

                        const double adjoint = result.adj_;
                        if constexpr (y_is_parameter && y_is_vector) {
                                for (int i = 0; i < N; ++i) y_nodes[i]->adj_ -= adjoint * d_location_values[i];
                        }
                        if constexpr (y_is_parameter && !y_is_vector) y_node->adj_ -= adjoint * d_mu_total;
                        if constexpr (mu_is_parameter && mu_is_vector) {
                                for (int i = 0; i < N; ++i) mu_nodes[i]->adj_ += adjoint * d_location_values[i];
                        }
                        if constexpr (mu_is_parameter && !mu_is_vector) mu_node->adj_ += adjoint * d_mu_total;
                        if constexpr (sigma_is_parameter && sigma_is_vector) {
                                for (int i = 0; i < N; ++i) sigma_nodes[i]->adj_ += adjoint * d_sigma_values[i];
                        }
                        if constexpr (sigma_is_parameter && !sigma_is_vector) {
                                sigma_node->adj_ += adjoint * d_sigma_total;
                        }
                        if constexpr (nu_is_parameter && nu_is_vector) {
                                for (int i = 0; i < N; ++i) nu_nodes[i]->adj_ += adjoint * d_nu_values[i];
                        }
                        if constexpr (nu_is_parameter && !nu_is_vector) nu_node->adj_ += adjoint * d_nu_total;

                });

        }

}




}  // namespace fused_student_t
}  // namespace nico::math




//// ---- the Stan-facing function (global scope; stanc calls it as nico_student_t_lpdf<propto__>(y, nu, mu,
////      sigma, pstream__)): y, nu, mu and sigma each a vector (Eigen vector or std::vector) or a scalar, of
////      double, int or var
namespace nico::math {

template <bool propto, typename T_y, typename T_nu, typename T_mu, typename T_sigma,
          std::enable_if_t<nico::math::fused_student_t::argument_type<T_y>::is_supported &&
                           nico::math::fused_student_t::argument_type<T_nu>::is_supported &&
                           nico::math::fused_student_t::argument_type<T_mu>::is_supported &&
                           nico::math::fused_student_t::argument_type<T_sigma>::is_supported>* = nullptr>
inline stan::return_type_t<T_y, T_nu, T_mu, T_sigma> nico_student_t_lpdf( const T_y& y,
                                                                          const T_nu& nu,
                                                                          const T_mu& mu,
                                                                          const T_sigma& sigma,
                                                                          std::ostream* /*pstream__*/
) {

        return nico::math::fused_student_t::student_t_lpdf<propto>(y, nu, mu, sigma);

}

}  // namespace nico::math

using nico::math::nico_student_t_lpdf;
























