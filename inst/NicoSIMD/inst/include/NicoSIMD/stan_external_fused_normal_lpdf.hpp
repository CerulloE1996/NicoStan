#pragma once
//
// -| --------- BayesMVP AVX-512 / AVX2 fused normal log density, exposed to Stan as an external function -------
//
// nico_normal_lpdf(y | mu, sigma): a drop-in replacement for Stan's normal_lpdf(y | mu, sigma), with the same
// arguments and the same density, as ONE fused function. Its value and every partial derivative are computed in
// ONE SIMD pass over the elements, with the kernel selection of stan_external_functions.hpp (8 lanes with the
// AVX-512 kernels, 4 lanes with the AVX2 kernels, including -DBAYESMVP_FORCE_AVX2), and the result is ONE
// autodiff node (a callback vari), whatever the number of elements.
//
// Arguments, as Stan's normal_lpdf: y, mu and sigma may each be a scalar (int, double or var) or a vector (an
// Eigen vector, or any Eigen vector expression, of double or var; or std::vector of int, double or var, i.e.,
// array[] real). Every vector must have the same length. The value is a double when every argument is data,
// a var otherwise. With propto = false the value is the density of Stan's normal_lpdf, including
// -0.5 log(2 pi) per element; with propto = true the terms which do not depend on a parameter are dropped, as
// in Stan (the constant always; sum log(sigma) when sigma is data; every term when every argument is data).
//
// Forward pass: one pass over the elements, in blocks of 256. A vector of double with contiguous storage (a
// vector, a segment y[a:b], a map, std::vector<double>) is read in place. A vector of var with contiguous
// storage (a vector, a segment, a map, std::vector<var>) is read one SIMD vector at a time: its node pointers are
// loaded as one vector and compared with a constant spacing in memory, and its values gathered from the nodes
// (vgatherqpd) into a block buffer, which stays in L1; its node pointers are stored on the arena (for the reverse
// pass) only when its nodes are not evenly spaced. An expression of var (e.g. exp(h / 2)) is read once per
// element, through Eigen's evaluator, so each of its nodes is created exactly once, and its node pointers are
// stored. Any other vector of double or int (an expression of data, a strided block, array[] int) is evaluated
// once into a buffer of its own (Eigen's vectorised assignment).
// Per SIMD vector: 1 / sigma (a division, for a vector sigma), the standardised residuals z = (y - mu) / sigma,
// z^2, the terms of sum log(sigma) and every partial derivative, in registers; the partials of the var vector
// arguments are stored on the arena and the partials of the var scalar arguments are summed. Arena per element:
// 8 bytes (partial) per var vector argument, + 8 bytes (node pointer) when its nodes are not evenly spaced (or
// it is an expression of var), and nothing else. (Stan's normal_lpdf with a var vector argument copies the node
// pointers and zero-fills, then fills, the partials on the arena, evaluates the values of a var argument and its
// intermediate vectors into heap temporaries, one pass each, and registers one callback per argument.)
//
// Reverse pass: ONE callback, which adds adj * partial to the node of each element of each var vector
// argument (one element at a time, in order: different elements can share a node; evenly spaced nodes are walked
// from the first node, without reading their pointers back), and adj * (summed partial) to each var scalar
// argument.
//
// Sum of log(sigma) over a vector sigma, with no logarithm per element: each sigma = m * 2^e with m in [1, 2),
// so sum log(sigma) = log(2) * sum e + log(prod m). The exponents are summed exactly and the mantissas
// multiplied (renormalised once per block), and log() is applied once per call. AVX-512: getexp / getmant,
// which also handle subnormal and infinite sigma; AVX2: the exponent and mantissa bits, with a subnormal or
// infinite sigma sending the whole sum to std::log. Exact to rounding (relative error ~1e-15), where the
// nico_log_1 kernel has an absolute error of up to ~4e-8 per element (and Eigen's vectorised log, used by
// Stan's normal_lpdf, clamps a subnormal sigma to the smallest normal double, 2.2e-308).
//
// Checks, as Stan's normal_lpdf: y must not be NaN, mu must be finite and sigma positive, and the vectors must
// have the same length. The checks are made in the same SIMD pass; an argument which fails them raises Stan's
// own error (std::domain_error / std::invalid_argument from stan::math::check_*, with the argument's name and
// index), before any node is created.
//
// Declared in the Stan functions block, one declaration per argument shape used (no body), e.g.
//
//     real nico_normal_lpdf(vector y, real mu, vector sigma);
//     real nico_normal_lpdf(vector y, vector mu, real sigma);
//     real nico_normal_lpdf(array[] real y, real mu, vector sigma);
//
// and called as  y ~ nico_normal(0, sigma);  or  target += nico_normal_lupdf(y | 0, sigma);  (one C++ template
// serves every declared shape). Included at the end of stan_external_functions.hpp: it uses that file's
// kernels, SIMD types and macros.
//
// --------------------------------------------------------------------------------------------------------------

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#ifndef BMVP_LANES
#  error "stan_external_fused_normal_lpdf.hpp is included at the end of stan_external_functions.hpp"
#endif

namespace nico::math {
namespace fused_normal {

using stan::math::var;
using stan::math::vari;

static constexpr const char* function_name = "nico_normal_lpdf";

// ---- elements per block: each block buffer (2 KB) stays in L1 between its gather and its SIMD pass
static constexpr int block_size = 256;
static_assert(block_size % BMVP_LANES == 0, "block_size must be a multiple of the SIMD lanes");

static constexpr double positive_infinity = std::numeric_limits<double>::infinity();

// ---- the SIMD operations of the fused pass (simd_t, simd_loadu, simd_storeu, simd_load and simd_store are
// stan_external_functions.hpp's). A lane mask selects lanes: __mmask8 (AVX-512), all bits set (AVX2).
#if BMVP_LANES == 8
using lane_mask_t = __mmask8;
inline simd_t simd_set1(const double value)                               { return _mm512_set1_pd(value); }
inline simd_t simd_add(const simd_t a, const simd_t b)                     { return _mm512_add_pd(a, b); }
inline simd_t simd_sub(const simd_t a, const simd_t b)                     { return _mm512_sub_pd(a, b); }
inline simd_t simd_mul(const simd_t a, const simd_t b)                     { return _mm512_mul_pd(a, b); }
inline simd_t simd_div(const simd_t a, const simd_t b)                     { return _mm512_div_pd(a, b); }
inline simd_t simd_fmsub(const simd_t a, const simd_t b, const simd_t c)   { return _mm512_fmsub_pd(a, b, c); }
inline simd_t simd_select(const lane_mask_t mask, const simd_t if_true, const simd_t if_false) {

    return _mm512_mask_blend_pd(mask, if_false, if_true);

}
inline lane_mask_t mask_of_first_lanes(const int n_lanes) {

    return static_cast<lane_mask_t>((1u << n_lanes) - 1u);

}
inline lane_mask_t mask_with_no_lanes()                              { return 0; }
inline lane_mask_t mask_or(const lane_mask_t a, const lane_mask_t b) { return a | b; }
inline bool        mask_has_any_lane(const lane_mask_t mask)         { return mask != 0; }
inline lane_mask_t lanes_where_nan(const simd_t x) {

    return _mm512_cmp_pd_mask(x, x, _CMP_UNORD_Q);

}
inline lane_mask_t lanes_where_not_finite(const simd_t x) {                        // NaN or +-inf

    return _mm512_cmp_pd_mask(_mm512_abs_pd(x), _mm512_set1_pd(positive_infinity), _CMP_NLT_UQ);

}
inline lane_mask_t lanes_where_not_greater_than(const simd_t x, const double bound) {  // NaN included

    return _mm512_cmp_pd_mask(x, _mm512_set1_pd(bound), _CMP_NGT_UQ);

}
#else
using lane_mask_t = __m256d;
inline simd_t simd_set1(const double value)                               { return _mm256_set1_pd(value); }
inline simd_t simd_add(const simd_t a, const simd_t b)                     { return _mm256_add_pd(a, b); }
inline simd_t simd_sub(const simd_t a, const simd_t b)                     { return _mm256_sub_pd(a, b); }
inline simd_t simd_mul(const simd_t a, const simd_t b)                     { return _mm256_mul_pd(a, b); }
inline simd_t simd_div(const simd_t a, const simd_t b)                     { return _mm256_div_pd(a, b); }
inline simd_t simd_fmsub(const simd_t a, const simd_t b, const simd_t c)   { return _mm256_fmsub_pd(a, b, c); }
inline simd_t simd_select(const lane_mask_t mask, const simd_t if_true, const simd_t if_false) {

    return _mm256_blendv_pd(if_false, if_true, mask);

}
inline lane_mask_t mask_of_first_lanes(const int n_lanes) {

    return _mm256_cmp_pd(_mm256_set_pd(3.0, 2.0, 1.0, 0.0), _mm256_set1_pd(static_cast<double>(n_lanes)),
                         _CMP_LT_OQ);

}
inline lane_mask_t mask_with_no_lanes()                              { return _mm256_setzero_pd(); }
inline lane_mask_t mask_or(const lane_mask_t a, const lane_mask_t b) { return _mm256_or_pd(a, b); }
inline bool        mask_has_any_lane(const lane_mask_t mask)         { return _mm256_movemask_pd(mask) != 0; }
inline lane_mask_t lanes_where_nan(const simd_t x)                   { return _mm256_cmp_pd(x, x, _CMP_UNORD_Q); }
inline lane_mask_t lanes_where_not_finite(const simd_t x) {                        // NaN or +-inf

    const simd_t absolute_x = _mm256_andnot_pd(_mm256_set1_pd(-0.0), x);
    return _mm256_cmp_pd(absolute_x, _mm256_set1_pd(positive_infinity), _CMP_NLT_UQ);

}
inline lane_mask_t lanes_where_not_greater_than(const simd_t x, const double bound) {  // NaN included

    return _mm256_cmp_pd(x, _mm256_set1_pd(bound), _CMP_NGT_UQ);

}
#endif

inline double simd_sum_of_lanes(const simd_t x) {

    alignas(64) double lanes[BMVP_LANES];
    simd_store(lanes, x);
    double sum = 0.0;
    for (int lane = 0; lane < BMVP_LANES; ++lane) sum += lanes[lane];
    return sum;

}

// ---- node pointers in SIMD registers (8 with AVX-512, 4 with AVX2): the values of a var vector in contiguous
// memory are gathered from its nodes (vgatherqpd), and the nodes' addresses are compared with a constant spacing
#if BMVP_LANES == 8
using simd_pointers_t = __m512i;
inline simd_pointers_t simd_load_pointers(const void* p)                    { return _mm512_loadu_si512(p); }
inline void            simd_store_pointers(void* p, const simd_pointers_t v) { _mm512_storeu_si512(p, v); }
inline simd_pointers_t simd_add_bytes(const simd_pointers_t addresses, const std::int64_t bytes) {

    return _mm512_add_epi64(addresses, _mm512_set1_epi64(bytes));

}
inline bool simd_pointers_are_equal(const simd_pointers_t a, const simd_pointers_t b) {

    return _mm512_cmpeq_epi64_mask(a, b) == 0xFF;

}
inline simd_t simd_gather_doubles(const simd_pointers_t addresses) {

    return _mm512_i64gather_pd(addresses, nullptr, 1);

}
// first, first + spacing, ..., first + 7 * spacing (bytes)
inline simd_pointers_t simd_addresses_at_constant_spacing(const std::int64_t first, const std::int64_t spacing) {

    return _mm512_set_epi64(first + 7 * spacing, first + 6 * spacing, first + 5 * spacing, first + 4 * spacing,
                            first + 3 * spacing, first + 2 * spacing, first + spacing, first);

}
#else
using simd_pointers_t = __m256i;
inline simd_pointers_t simd_load_pointers(const void* p) {

    return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));

}
inline void simd_store_pointers(void* p, const simd_pointers_t v) {

    _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), v);

}
inline simd_pointers_t simd_add_bytes(const simd_pointers_t addresses, const std::int64_t bytes) {

    return _mm256_add_epi64(addresses, _mm256_set1_epi64x(bytes));

}
inline bool simd_pointers_are_equal(const simd_pointers_t a, const simd_pointers_t b) {

    return _mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(a, b))) == 0xF;

}
inline simd_t simd_gather_doubles(const simd_pointers_t addresses) {

    return _mm256_i64gather_pd(static_cast<const double*>(nullptr), addresses, 1);

}
// first, first + spacing, ..., first + 3 * spacing (bytes)
inline simd_pointers_t simd_addresses_at_constant_spacing(const std::int64_t first, const std::int64_t spacing) {

    return _mm256_set_epi64x(first + 3 * spacing, first + 2 * spacing, first + spacing, first);

}
#endif

// ---- sum of log(sigma) over positive sigma, with no logarithm per element (see the header comment)
struct sum_of_logs_accumulator {
#if BMVP_LANES == 8
    simd_t mantissa_product = _mm512_set1_pd(1.0);
    simd_t exponent_sum     = _mm512_setzero_pd();

    static simd_t mantissa_of(const simd_t x) {

        return _mm512_getmant_pd(x, _MM_MANT_NORM_1_2, _MM_MANT_SIGN_zero);

    }
    void add(const simd_t x) {

        exponent_sum     = _mm512_add_pd(exponent_sum, _mm512_getexp_pd(x));
        mantissa_product = _mm512_mul_pd(mantissa_product, mantissa_of(x));

    }
    void renormalise() {

        exponent_sum     = _mm512_add_pd(exponent_sum, _mm512_getexp_pd(mantissa_product));
        mantissa_product = mantissa_of(mantissa_product);

    }
    // lanes of sigma which are invalid (NaN, zero or negative); getexp / getmant handle every other value
    static lane_mask_t lanes_where_invalid_or_exceptional(const simd_t sigma) {

        return lanes_where_not_greater_than(sigma, 0.0);

    }
    double total() {

        renormalise();
        const double exponent_total = simd_sum_of_lanes(exponent_sum);
        if (exponent_total == positive_infinity) return positive_infinity;          // an infinite sigma
        alignas(64) double mantissas[BMVP_LANES];
        simd_store(mantissas, mantissa_product);
        double mantissa_total = 1.0;                                                // each in [1, 2): < 2^8
        for (int lane = 0; lane < BMVP_LANES; ++lane) mantissa_total *= mantissas[lane];
        return exponent_total * LN2 + std::log(mantissa_total);

    }
#else
    simd_t  mantissa_product    = _mm256_set1_pd(1.0);
    ALIGN32  __m256i biased_exponent_sum = _mm256_setzero_si256();   // each term carries the exponent bias 1023
    int64_t n_terms_per_lane    = 0;

    static simd_t mantissa_of(const simd_t x) {               // x positive, normal and finite

        const simd_t mantissa_bits = _mm256_castsi256_pd(_mm256_set1_epi64x(0x000FFFFFFFFFFFFFLL));
        return _mm256_or_pd(_mm256_and_pd(x, mantissa_bits), _mm256_set1_pd(1.0));

    }
    static __m256i biased_exponent_of(const simd_t x) { return _mm256_srli_epi64(_mm256_castpd_si256(x), 52); }
    void add(const simd_t x) {

        biased_exponent_sum = _mm256_add_epi64(biased_exponent_sum, biased_exponent_of(x));
        mantissa_product    = _mm256_mul_pd(mantissa_product, mantissa_of(x));
        n_terms_per_lane   += 1;

    }
    void renormalise() {

        biased_exponent_sum = _mm256_add_epi64(biased_exponent_sum, biased_exponent_of(mantissa_product));
        mantissa_product    = mantissa_of(mantissa_product);
        n_terms_per_lane   += 1;

    }
    // lanes of sigma which are invalid (NaN, zero or negative), subnormal or infinite: the exponent and mantissa
    // bits hold m * 2^e with m in [1, 2) for positive normal finite values only
    static lane_mask_t lanes_where_invalid_or_exceptional(const simd_t sigma) {

        const lane_mask_t below_smallest_normal = _mm256_cmp_pd(sigma, _mm256_set1_pd(2.2250738585072014e-308),
                                                                _CMP_NGE_UQ);
        const lane_mask_t infinite = _mm256_cmp_pd(sigma, _mm256_set1_pd(1.7976931348623157e+308), _CMP_GT_OQ);
        return _mm256_or_pd(below_smallest_normal, infinite);

    }
    double total() {

        renormalise();
        alignas(32) int64_t biased_exponents[4];
        _mm256_store_si256(reinterpret_cast<__m256i*>(biased_exponents), biased_exponent_sum);
        double exponent_total = 0.0;
        for (int lane = 0; lane < 4; ++lane) {
            exponent_total += static_cast<double>(biased_exponents[lane] - 1023 * n_terms_per_lane);
        }
        alignas(32) double mantissas[4];
        _mm256_store_pd(mantissas, mantissa_product);
        double mantissa_total = 1.0;                                                // each in [1, 2): < 2^4
        for (int lane = 0; lane < 4; ++lane) mantissa_total *= mantissas[lane];
        return exponent_total * LN2 + std::log(mantissa_total);

    }
#endif
};

// ---- the arguments: a scalar (int, double or var) or a vector (Eigen vector or std::vector) of them
template <typename T>
struct is_var_scalar : std::integral_constant<bool, std::is_same<std::decay_t<T>, var>::value> {};

template <typename T>
struct is_supported_scalar
    : std::integral_constant<bool, std::is_arithmetic<std::decay_t<T>>::value || is_var_scalar<T>::value> {};

template <typename T, typename = void>
struct is_supported_vector : std::false_type {};
template <typename T>
struct is_supported_vector<T, std::enable_if_t<stan::is_eigen_vector<std::decay_t<T>>::value
                                               || stan::is_std_vector<std::decay_t<T>>::value>>
    : std::integral_constant<bool, is_supported_scalar<stan::value_type_t<std::decay_t<T>>>::value> {};

// a vector of double whose elements are contiguous in memory (read in place)
template <typename T, typename = void>
struct is_contiguous_double_vector : std::false_type {};
template <typename T>
struct is_contiguous_double_vector<T, std::enable_if_t<stan::is_eigen_vector<std::decay_t<T>>::value>>
    : std::integral_constant<bool, std::is_same<typename std::decay_t<T>::Scalar, double>::value
                                   && (int(Eigen::internal::traits<std::decay_t<T>>::Flags)
                                       & Eigen::DirectAccessBit)
                                   && int(std::decay_t<T>::InnerStrideAtCompileTime) == 1> {};
template <typename T>
struct is_contiguous_double_vector<T, std::enable_if_t<stan::is_std_vector<std::decay_t<T>>::value>>
    : std::integral_constant<bool, std::is_same<stan::value_type_t<std::decay_t<T>>, double>::value> {};

// a vector of var whose elements (node pointers) are contiguous in memory: an Eigen vector, block, map or ref
// of var with unit inner stride, std::vector<var>
template <typename T, typename = void>
struct is_contiguous_var_vector : std::false_type {};
template <typename T>
struct is_contiguous_var_vector<T, std::enable_if_t<stan::is_eigen_vector<std::decay_t<T>>::value>>
    : std::integral_constant<bool, std::is_same<typename std::decay_t<T>::Scalar, var>::value
                                   && (int(Eigen::internal::traits<std::decay_t<T>>::Flags)
                                       & Eigen::DirectAccessBit)
                                   && int(std::decay_t<T>::InnerStrideAtCompileTime) == 1> {};
template <typename T>
struct is_contiguous_var_vector<T, std::enable_if_t<stan::is_std_vector<std::decay_t<T>>::value>>
    : std::integral_constant<bool, std::is_same<stan::value_type_t<std::decay_t<T>>, var>::value> {};

enum class argument_kind { arithmetic_scalar, var_scalar, contiguous_double_vector, other_arithmetic_vector,
                           contiguous_var_vector, var_vector_expression };

template <typename T>
constexpr argument_kind kind_of_argument() {

    using plain_t = std::decay_t<T>;
    if constexpr (std::is_arithmetic<plain_t>::value) {
        return argument_kind::arithmetic_scalar;
    } else if constexpr (is_var_scalar<plain_t>::value) {
        return argument_kind::var_scalar;
    } else if constexpr (is_contiguous_var_vector<plain_t>::value) {
        return argument_kind::contiguous_var_vector;
    } else if constexpr (is_var_scalar<stan::value_type_t<plain_t>>::value) {
        return argument_kind::var_vector_expression;
    } else if constexpr (is_contiguous_double_vector<plain_t>::value) {
        return argument_kind::contiguous_double_vector;
    } else {
        return argument_kind::other_arithmetic_vector;
    }

}

// i-th element of a vector of var: an Eigen vector with direct access (a vector, a block or segment, a map)
// through its own coeff(); any other Eigen vector expression through Eigen's evaluator (an expression is
// evaluated one element at a time, a product once into a temporary); a std::vector by its index
template <typename T, typename = void>
struct has_direct_access : std::false_type {};
template <typename T>
struct has_direct_access<T, std::enable_if_t<stan::is_eigen<T>::value>>
    : std::integral_constant<bool, (int(Eigen::internal::traits<T>::Flags) & Eigen::DirectAccessBit) != 0> {};

template <typename T, typename = void>
struct element_reader;
template <typename T>
struct element_reader<T, std::enable_if_t<stan::is_eigen<T>::value && has_direct_access<T>::value>> {
    const T& x;
    explicit element_reader(const T& x_in) : x(x_in) {}
    decltype(auto) operator()(const int i) const { return x.coeff(i); }
};
template <typename T>
struct element_reader<T, std::enable_if_t<stan::is_eigen<T>::value && !has_direct_access<T>::value>> {
    Eigen::internal::evaluator<T> evaluator;
    explicit element_reader(const T& x) : evaluator(x) {}
    decltype(auto) operator()(const int i) const { return evaluator.coeff(i); }
};
template <typename T>
struct element_reader<T, std::enable_if_t<stan::is_std_vector<T>::value>> {
    const T& x;
    explicit element_reader(const T& x_in) : x(x_in) {}
    decltype(auto) operator()(const int i) const { return x[i]; }
};

// a buffer per argument position (0: y, 1: mu, 2: sigma) and thread, reused from call to call
template <int position>
inline std::vector<double>& evaluation_buffer_of_argument() {

    thread_local std::vector<double> buffer;
    return buffer;

}

// ---- one argument, as the fused pass reads it (position: 0 for y, 1 for mu, 2 for sigma). Vector arguments give
// their values block by block:
//     block_values(first, count)                 values of elements first, ..., first + count - 1
//     tail_values(first, count, fill, padded)    the final count < BMVP_LANES values, padded with fill
//     value_at(i)                                one value (error path; var: after the forward pass)
template <int position, typename T, argument_kind kind = kind_of_argument<T>()>
struct argument;

template <int position, typename T>
struct argument<position, T, argument_kind::arithmetic_scalar> {
    static constexpr bool is_vector = false;
    static constexpr bool is_var    = false;
    double value;
    argument(const T& x, stan::math::stack_alloc* /*arena*/) : value(static_cast<double>(x)) {}
};

template <int position, typename T>
struct argument<position, T, argument_kind::var_scalar> {
    static constexpr bool is_vector = false;
    static constexpr bool is_var    = true;
    vari*  node;
    double value;
    argument(const T& x, stan::math::stack_alloc* /*arena*/) : node(x.vi_), value(x.vi_->val_) {}
};

template <int position, typename T>
struct argument<position, T, argument_kind::contiguous_double_vector> {
    static constexpr bool is_vector = true;
    static constexpr bool is_var    = false;
    const double* values;
    int n;
    argument(const T& x, stan::math::stack_alloc* /*arena*/) : values(x.data()), n(static_cast<int>(x.size())) {}
    int size() const { return n; }
    const double* block_values(const int first, const int /*count*/) { return values + first; }
    void tail_values(const int first, const int count, const double fill, double* padded) {

        for (int lane = 0; lane < BMVP_LANES; ++lane) padded[lane] = (lane < count) ? values[first + lane] : fill;

    }
    double value_at(const int i) const { return values[i]; }
};

// any other vector of int or double (an Eigen expression, e.g. exp(h / 2) or X * b of data, a strided block,
// std::vector<int>): evaluated once, into this argument's buffer (an Eigen expression by Eigen's vectorised
// assignment), then read in place
template <int position, typename T>
struct argument<position, T, argument_kind::other_arithmetic_vector> {
    static constexpr bool is_vector = true;
    static constexpr bool is_var    = false;
    const double* values;
    int n;
    argument(const T& x, stan::math::stack_alloc* /*arena*/) : n(static_cast<int>(x.size())) {

        std::vector<double>& buffer = evaluation_buffer_of_argument<position>();
        if (static_cast<int>(buffer.size()) < n) buffer.resize(n);
        if constexpr (stan::is_eigen<std::decay_t<T>>::value) {
            Eigen::Map<Eigen::Matrix<double, -1, 1>>(buffer.data(), n) = x.template cast<double>();
        } else {
            for (int i = 0; i < n; ++i) buffer[i] = static_cast<double>(x[i]);
        }
        values = buffer.data();

    }
    int size() const { return n; }
    const double* block_values(const int first, const int /*count*/) { return values + first; }
    void tail_values(const int first, const int count, const double fill, double* padded) {

        for (int lane = 0; lane < BMVP_LANES; ++lane) padded[lane] = (lane < count) ? values[first + lane] : fill;

    }
    double value_at(const int i) const { return values[i]; }
};

// ---- the nodes of a var vector argument, for the reverse pass. When the nodes lie at a constant spacing in
// memory (e.g. the parameters, or the result of a vectorised function: nodes allocated one after another; or one
// node repeated, spacing 0), the first node and the spacing, so that the reverse pass walks the nodes without
// reading their pointers back; otherwise their pointers, stored on the arena.
inline std::int64_t address_of(const vari* node) { return reinterpret_cast<std::int64_t>(node); }

struct nodes_of_var_vector_argument {
    bool         is_var_vector       = false;
    vari**       pointers            = nullptr;   // on the arena (a vector in contiguous memory: once needed)
    vari*        first_node          = nullptr;
    std::int64_t spacing_in_bytes    = 0;
    bool         spacing_is_constant = false;

    vari* node(const int i) const {

        if (spacing_is_constant) {
            return reinterpret_cast<vari*>(reinterpret_cast<char*>(first_node) + i * spacing_in_bytes);
        }
        return pointers[i];

    }
    // element i's node (an expression's nodes, read once, in order): its pointer, and whether the spacing is
    // still constant
    void note_node(const int i, vari* node) {

        pointers[i] = node;
        if (i == 0) {
            first_node          = node;
            spacing_is_constant = true;
            return;
        }
        if (i == 1) spacing_in_bytes = address_of(node) - address_of(first_node);
        spacing_is_constant = spacing_is_constant
                              && (address_of(node) == address_of(first_node) + i * spacing_in_bytes);

    }
    // adds scaled_adjoint * partials[i] to the adjoint of element i's node, one element at a time, in order
    void add_to_adjoints(const double scaled_adjoint, const double* partials, const int n) const {

        if (!is_var_vector) return;
        if (spacing_is_constant) {
            char* node_address = reinterpret_cast<char*>(first_node);
            for (int i = 0; i < n; ++i) {
                reinterpret_cast<vari*>(node_address)->adj_ += scaled_adjoint * partials[i];
                node_address += spacing_in_bytes;
            }
        } else {
            for (int i = 0; i < n; ++i) pointers[i]->adj_ += scaled_adjoint * partials[i];
        }

    }
};

// a vector of var in contiguous memory: per SIMD vector, the node pointers are loaded as one vector and compared
// with the constant spacing, and the values gathered from the nodes (vgatherqpd). The pointers are stored (on
// the arena) only from the first element which breaks the constant spacing (the earlier ones are read again
// from the vector, which is still there during the forward pass).
template <int position, typename T>
struct argument<position, T, argument_kind::contiguous_var_vector> {
    static constexpr bool is_vector = true;
    static constexpr bool is_var    = true;
    const T& x;
    const vari* const* node_pointers;                // the vector's own elements: var holds one vari*
    int n;
    stan::math::stack_alloc* arena;
    nodes_of_var_vector_argument nodes;
    std::int64_t offset_of_value_in_node = 0;
    simd_pointers_t addresses_of_first_nodes_at_constant_spacing;
    alignas(64) double buffer[block_size];
    argument(const T& x_in, stan::math::stack_alloc* arena_in)
        : x(x_in), node_pointers(reinterpret_cast<const vari* const*>(x_in.data())),
          n(static_cast<int>(x_in.size())), arena(arena_in) {

        static_assert(sizeof(var) == sizeof(vari*), "a var must hold exactly one vari pointer");
        nodes.is_var_vector = true;
        if (n > 0) {
            nodes.first_node          = x.data()[0].vi_;
            nodes.spacing_is_constant = true;
            offset_of_value_in_node   = reinterpret_cast<const char*>(&nodes.first_node->val_)
                                        - reinterpret_cast<const char*>(nodes.first_node);
        }
        if (n > 1) nodes.spacing_in_bytes = address_of(x.data()[1].vi_) - address_of(nodes.first_node);
        addresses_of_first_nodes_at_constant_spacing
            = simd_addresses_at_constant_spacing(address_of(nodes.first_node), nodes.spacing_in_bytes);

    }
    // the spacing is not constant at element i: the pointers of elements 0, ..., i - 1 are stored now, and every
    // pointer from element i on as it is read
    void store_pointers_from_element(const int i) {

        nodes.pointers = arena->template alloc_array<vari*>(n);
        for (int k = 0; k < i; ++k) nodes.pointers[k] = x.data()[k].vi_;
        nodes.spacing_is_constant = false;

    }
    int size() const { return n; }
    const double* block_values(const int first, const int count) {      // count: a multiple of BMVP_LANES

        for (int j = 0; j < count; j += BMVP_LANES) {
            const simd_pointers_t pointers = simd_load_pointers(node_pointers + first + j);
            if (nodes.spacing_is_constant) {
                const simd_pointers_t expected = simd_add_bytes(addresses_of_first_nodes_at_constant_spacing,
                                                                (first + j) * nodes.spacing_in_bytes);
                if (!simd_pointers_are_equal(pointers, expected)) store_pointers_from_element(first + j);
            }
            if (!nodes.spacing_is_constant) simd_store_pointers(nodes.pointers + first + j, pointers);
            simd_store(buffer + j, simd_gather_doubles(simd_add_bytes(pointers, offset_of_value_in_node)));
        }
        return buffer;

    }
    void tail_values(const int first, const int count, const double fill, double* padded) {

        for (int lane = 0; lane < BMVP_LANES; ++lane) {
            if (lane < count) {
                const int i = first + lane;
                vari* node = x.data()[i].vi_;
                if (nodes.spacing_is_constant
                    && address_of(node) != address_of(nodes.first_node) + i * nodes.spacing_in_bytes) {

                    store_pointers_from_element(i);

                }
                if (!nodes.spacing_is_constant) nodes.pointers[i] = node;
                padded[lane] = node->val_;
            } else {
                padded[lane] = fill;
            }
        }

    }
    double value_at(const int i) const { return nodes.node(i)->val_; }
};

// any other vector of var (an expression of var, e.g. exp(h / 2), whose nodes are made as it is read): each
// element is read once
template <int position, typename T>
struct argument<position, T, argument_kind::var_vector_expression> {
    static constexpr bool is_vector = true;
    static constexpr bool is_var    = true;
    element_reader<std::decay_t<T>> element;
    int n;
    nodes_of_var_vector_argument nodes;
    alignas(64) double buffer[block_size];
    argument(const T& x, stan::math::stack_alloc* arena) : element(x), n(static_cast<int>(x.size())) {

        nodes.is_var_vector = true;
        nodes.pointers      = arena->template alloc_array<vari*>(n);

    }
    int size() const { return n; }
    const double* block_values(const int first, const int count) {

        for (int j = 0; j < count; ++j) {
            vari* node = element(first + j).vi_;
            nodes.note_node(first + j, node);
            buffer[j] = node->val_;
        }
        return buffer;

    }
    void tail_values(const int first, const int count, const double fill, double* padded) {

        for (int lane = 0; lane < BMVP_LANES; ++lane) {
            if (lane < count) {
                vari* node = element(first + lane).vi_;
                nodes.note_node(first + lane, node);
                padded[lane] = node->val_;
            } else {
                padded[lane] = fill;
            }
        }

    }
    double value_at(const int i) const { return nodes.pointers[i]->val_; }
};

// the values of an argument as Stan's check functions take them (error path only)
template <typename argument_t>
inline auto values_of_argument(const argument_t& x) {

    if constexpr (argument_t::is_vector) {
        Eigen::Matrix<double, -1, 1> values(x.size());
        for (int i = 0; i < x.size(); ++i) values[i] = x.value_at(i);
        return values;
    } else {
        return x.value;
    }

}

template <typename argument_t>
inline const double* block_values_of(argument_t& x, const int first, const int count) {

    if constexpr (argument_t::is_vector) return x.block_values(first, count);
    else                                 return nullptr;

}

template <typename argument_t>
inline void tail_values_of(argument_t& x, const int first, const int count, const double fill, double* padded) {

    if constexpr (argument_t::is_vector) x.tail_values(first, count, fill, padded);

}

template <typename argument_t>
inline simd_t lanes_of(const argument_t& /*x*/, const double* values, const int offset, const simd_t broadcast) {

    if constexpr (argument_t::is_vector) return simd_loadu(values + offset);
    else                                 return broadcast;

}

template <typename argument_t>
inline nodes_of_var_vector_argument nodes_of_var_vector(const argument_t& x) {

    if constexpr (argument_t::is_var && argument_t::is_vector) return x.nodes;
    else                                                         return nodes_of_var_vector_argument();

}

template <typename argument_t>
inline vari* node_of_var_scalar(const argument_t& x) {

    if constexpr (argument_t::is_var && !argument_t::is_vector) return x.node;
    else                                                          return nullptr;

}

// Stan's checks, in Stan's order, on the values of the arguments: they throw Stan's own error for the first
// argument which fails them (called only when the fused pass found a lane needing attention)
template <typename y_argument_t, typename mu_argument_t, typename sigma_argument_t>
inline void throw_error_of_first_invalid_argument(const y_argument_t& y, const mu_argument_t& mu,
                                                  const sigma_argument_t& sigma) {

    const auto y_values     = values_of_argument(y);
    const auto mu_values    = values_of_argument(mu);
    const auto sigma_values = values_of_argument(sigma);
    stan::math::check_not_nan(function_name, "Random variable", y_values);
    stan::math::check_finite(function_name, "Location parameter", mu_values);
    stan::math::check_positive(function_name, "Scale parameter", sigma_values);

}

// ---- the reverse pass: the single callback of the result's node. Nodes and pointers are empty / nullptr for the
// arguments they do not apply to (location partials: (y - mu) / sigma^2 per element, minus for y, plus for mu).
struct reverse_pass_of_fused_normal {
    int                          n                        = 0;
    nodes_of_var_vector_argument y_nodes;                       // y a var vector
    nodes_of_var_vector_argument mu_nodes;                      // mu a var vector
    nodes_of_var_vector_argument sigma_nodes;                   // sigma a var vector
    const double*                location_partials        = nullptr;
    const double*                sigma_partials           = nullptr;
    vari*                        y_node                   = nullptr;     // y a var scalar
    vari*                        mu_node                  = nullptr;     // mu a var scalar
    vari*                        sigma_node               = nullptr;     // sigma a var scalar
    double                       summed_location_partial  = 0.0;
    double                       summed_sigma_partial     = 0.0;

    template <typename result_vari_t>
    void operator()(const result_vari_t& result) const {

        const double adjoint = result.adj_;
        y_nodes.add_to_adjoints(-adjoint, location_partials, n);
        mu_nodes.add_to_adjoints(adjoint, location_partials, n);
        sigma_nodes.add_to_adjoints(adjoint, sigma_partials, n);
        if (y_node != nullptr)     y_node->adj_     -= adjoint * summed_location_partial;
        if (mu_node != nullptr)    mu_node->adj_    += adjoint * summed_location_partial;
        if (sigma_node != nullptr) sigma_node->adj_ += adjoint * summed_sigma_partial;

    }
};

// ---- the fused pass: per SIMD vector of elements, the value terms, the checks and the partials
template <bool propto, typename y_argument_t, typename mu_argument_t, typename sigma_argument_t>
struct fused_pass {
    static constexpr bool any_var                 = y_argument_t::is_var || mu_argument_t::is_var
                                                    || sigma_argument_t::is_var;
    static constexpr bool include_log_sigma_term  = !propto || sigma_argument_t::is_var;
    static constexpr bool sigma_is_vector         = sigma_argument_t::is_vector;
    static constexpr bool store_location_partials = (y_argument_t::is_var && y_argument_t::is_vector)
                                                    || (mu_argument_t::is_var && mu_argument_t::is_vector);
    static constexpr bool sum_location_partials   = (y_argument_t::is_var && !y_argument_t::is_vector)
                                                    || (mu_argument_t::is_var && !mu_argument_t::is_vector);
    static constexpr bool store_sigma_partials    = sigma_argument_t::is_var && sigma_argument_t::is_vector;
    static constexpr bool sum_logs_of_sigma       = sigma_is_vector && include_log_sigma_term;

    simd_t                  sum_of_squared_standardised_residuals = simd_set1(0.0);
    simd_t                  sum_of_location_partials              = simd_set1(0.0);
    sum_of_logs_accumulator sum_of_logs_of_sigma;
    lane_mask_t             lanes_needing_attention               = mask_with_no_lanes();
    simd_t                  inverse_of_scalar_sigma               = simd_set1(1.0);
    double*                 location_partials                     = nullptr;          // on the arena
    double*                 sigma_partials                        = nullptr;          // on the arena

    // the elements element, ..., element + BMVP_LANES - 1; in the tail, only the lanes in valid_lanes are
    // elements (the others hold valid fill values, and their terms are set to zero)
    template <bool tail>
    void process(const simd_t y, const simd_t mu, const simd_t sigma, const int element,
                 const lane_mask_t valid_lanes) {

        const simd_t zero          = simd_set1(0.0);
        simd_t       inverse_sigma = inverse_of_scalar_sigma;
        if constexpr (sigma_is_vector) inverse_sigma = simd_div(simd_set1(1.0), sigma);
        const simd_t standardised_residual         = simd_mul(simd_sub(y, mu), inverse_sigma);   // z
        simd_t       squared_standardised_residual = simd_mul(standardised_residual, standardised_residual);
        simd_t       location_partial              = simd_mul(standardised_residual, inverse_sigma);
        if constexpr (tail) {
            squared_standardised_residual = simd_select(valid_lanes, squared_standardised_residual, zero);
            location_partial              = simd_select(valid_lanes, location_partial, zero);
        }
        sum_of_squared_standardised_residuals = simd_add(sum_of_squared_standardised_residuals,
                                                         squared_standardised_residual);
        if constexpr (store_location_partials) simd_storeu(location_partials + element, location_partial);
        if constexpr (sum_location_partials) {
            sum_of_location_partials = simd_add(sum_of_location_partials, location_partial);
        }
        // d/d sigma = (z^2 - 1) / sigma
        if constexpr (store_sigma_partials) {
            simd_storeu(sigma_partials + element,
                        simd_fmsub(squared_standardised_residual, inverse_sigma, inverse_sigma));
        }
        if constexpr (sum_logs_of_sigma) sum_of_logs_of_sigma.add(sigma);             // tail fill: 1
        // checks of the vector arguments (the scalar ones are checked once, before the pass)
        if constexpr (y_argument_t::is_vector) {
            lanes_needing_attention = mask_or(lanes_needing_attention, lanes_where_nan(y));
        }
        if constexpr (mu_argument_t::is_vector) {
            lanes_needing_attention = mask_or(lanes_needing_attention, lanes_where_not_finite(mu));
        }
        if constexpr (sigma_is_vector) {
            lanes_needing_attention = mask_or(lanes_needing_attention,
                                              sum_of_logs_accumulator::lanes_where_invalid_or_exceptional(sigma));
        }

    }
};

// ---- the fused normal log density: value and partials in one pass, one node (see the header comment)
template <bool propto, typename T_y, typename T_loc, typename T_scale>
inline stan::return_type_t<T_y, T_loc, T_scale> fused_normal_log_density(const T_y& y, const T_loc& mu,
                                                                          const T_scale& sigma) {

    static_assert(is_supported_scalar<T_y>::value || is_supported_vector<T_y>::value,
                  "y must be an int, double or var, or a vector (Eigen vector or std::vector) of them");
    static_assert(is_supported_scalar<T_loc>::value || is_supported_vector<T_loc>::value,
                  "mu must be an int, double or var, or a vector (Eigen vector or std::vector) of them");
    static_assert(is_supported_scalar<T_scale>::value || is_supported_vector<T_scale>::value,
                  "sigma must be an int, double or var, or a vector (Eigen vector or std::vector) of them");
    using return_t         = stan::return_type_t<T_y, T_loc, T_scale>;
    using y_argument_t     = argument<0, T_y>;
    using mu_argument_t    = argument<1, T_loc>;
    using sigma_argument_t = argument<2, T_scale>;
    using pass_t           = fused_pass<propto, y_argument_t, mu_argument_t, sigma_argument_t>;
    constexpr bool include_quadratic_term = !propto || pass_t::any_var;
    constexpr bool include_constant_term  = !propto;

    stan::math::check_consistent_sizes(function_name, "Random variable", y, "Location parameter", mu,
                                       "Scale parameter", sigma);

    stan::math::stack_alloc* arena = nullptr;                     // touched only when a node is made
    if constexpr (pass_t::any_var) arena = &stan::math::ChainableStack::instance_->memalloc_;

    // ---- the scalar arguments, checked once
    bool   scalar_argument_is_invalid = false;
    double y_scalar = 0.0, mu_scalar = 0.0, sigma_scalar = 1.0;
    if constexpr (!y_argument_t::is_vector) {
        y_scalar = static_cast<double>(stan::math::value_of(y));
        scalar_argument_is_invalid |= std::isnan(y_scalar);
    }
    if constexpr (!mu_argument_t::is_vector) {
        mu_scalar = static_cast<double>(stan::math::value_of(mu));
        scalar_argument_is_invalid |= !std::isfinite(mu_scalar);
    }
    if constexpr (!sigma_argument_t::is_vector) {
        sigma_scalar = static_cast<double>(stan::math::value_of(sigma));
        scalar_argument_is_invalid |= !(sigma_scalar > 0.0);
    }
    if (stan::math::size_zero(y, mu, sigma)) {
        if (scalar_argument_is_invalid) {
            const y_argument_t     y_argument(y, arena);
            const mu_argument_t    mu_argument(mu, arena);
            const sigma_argument_t sigma_argument(sigma, arena);
            throw_error_of_first_invalid_argument(y_argument, mu_argument, sigma_argument);
        }
        return return_t(0.0);
    }
    const int n = static_cast<int>(stan::math::max_size(y, mu, sigma));

    y_argument_t     y_argument(y, arena);
    mu_argument_t    mu_argument(mu, arena);
    sigma_argument_t sigma_argument(sigma, arena);

    pass_t pass;
    const int n_rounded_up_to_lanes = ((n + BMVP_LANES - 1) / BMVP_LANES) * BMVP_LANES;
    if constexpr (pass_t::store_location_partials) {
        pass.location_partials = arena->template alloc_array<double>(n_rounded_up_to_lanes);
    }
    if constexpr (pass_t::store_sigma_partials) {
        pass.sigma_partials = arena->template alloc_array<double>(n_rounded_up_to_lanes);
    }
    pass.inverse_of_scalar_sigma = simd_set1(1.0 / sigma_scalar);
    const simd_t y_broadcast     = simd_set1(y_scalar);
    const simd_t mu_broadcast    = simd_set1(mu_scalar);
    const simd_t sigma_broadcast = simd_set1(sigma_scalar);

    // ---- the fused pass: blocks of full SIMD vectors, then the final partial SIMD vector (fill values y = 0,
    // mu = 0, sigma = 1: valid values, whose terms are set to zero)
    const lane_mask_t all_lanes         = mask_of_first_lanes(BMVP_LANES);
    const int         n_in_full_vectors = n - n % BMVP_LANES;
    for (int first = 0; first < n_in_full_vectors; first += block_size) {
        const int count = std::min(block_size, n_in_full_vectors - first);
        const double* y_values     = block_values_of(y_argument, first, count);
        const double* mu_values    = block_values_of(mu_argument, first, count);
        const double* sigma_values = block_values_of(sigma_argument, first, count);
        for (int offset = 0; offset < count; offset += BMVP_LANES) {
            pass.template process<false>(lanes_of(y_argument, y_values, offset, y_broadcast),
                                         lanes_of(mu_argument, mu_values, offset, mu_broadcast),
                                         lanes_of(sigma_argument, sigma_values, offset, sigma_broadcast),
                                         first + offset, all_lanes);
        }
        if constexpr (pass_t::sum_logs_of_sigma) pass.sum_of_logs_of_sigma.renormalise();
    }
    const int n_in_tail = n - n_in_full_vectors;
    if (n_in_tail > 0) {
        alignas(64) double y_tail[BMVP_LANES];
        alignas(64) double mu_tail[BMVP_LANES];
        alignas(64) double sigma_tail[BMVP_LANES];
        tail_values_of(y_argument, n_in_full_vectors, n_in_tail, 0.0, y_tail);
        tail_values_of(mu_argument, n_in_full_vectors, n_in_tail, 0.0, mu_tail);
        tail_values_of(sigma_argument, n_in_full_vectors, n_in_tail, 1.0, sigma_tail);
        pass.template process<true>(lanes_of(y_argument, y_tail, 0, y_broadcast),
                                    lanes_of(mu_argument, mu_tail, 0, mu_broadcast),
                                    lanes_of(sigma_argument, sigma_tail, 0, sigma_broadcast),
                                    n_in_full_vectors, mask_of_first_lanes(n_in_tail));
    }

    // ---- an invalid argument raises Stan's error (no node has been made); otherwise, on the AVX2 path, a
    // subnormal or infinite sigma needs the sum of log(sigma) from std::log
    bool sum_of_logs_of_sigma_from_std_log = false;
    if (scalar_argument_is_invalid || mask_has_any_lane(pass.lanes_needing_attention)) {
        throw_error_of_first_invalid_argument(y_argument, mu_argument, sigma_argument);
        sum_of_logs_of_sigma_from_std_log = true;
    }

    // ---- the value
    const double sum_of_squared_standardised_residuals
        = simd_sum_of_lanes(pass.sum_of_squared_standardised_residuals);
    double log_density = 0.0;
    if constexpr (include_quadratic_term) log_density -= 0.5 * sum_of_squared_standardised_residuals;
    if constexpr (pass_t::include_log_sigma_term) {
        double sum_of_log_sigma = 0.0;
        if constexpr (pass_t::sum_logs_of_sigma) {
            if (sum_of_logs_of_sigma_from_std_log) {
                for (int i = 0; i < n; ++i) sum_of_log_sigma += std::log(sigma_argument.value_at(i));
            } else {
                sum_of_log_sigma = pass.sum_of_logs_of_sigma.total();
            }
        } else {
            sum_of_log_sigma = n * std::log(sigma_scalar);
        }
        log_density -= sum_of_log_sigma;
    }
    if constexpr (include_constant_term) log_density += n * stan::math::NEG_LOG_SQRT_TWO_PI;

    if constexpr (!pass_t::any_var) {
        return log_density;
    } else {
        // ---- the single node, and its reverse pass
        reverse_pass_of_fused_normal reverse_pass;
        reverse_pass.n                 = n;
        reverse_pass.y_nodes           = nodes_of_var_vector(y_argument);
        reverse_pass.mu_nodes          = nodes_of_var_vector(mu_argument);
        reverse_pass.sigma_nodes       = nodes_of_var_vector(sigma_argument);
        reverse_pass.location_partials = pass.location_partials;
        reverse_pass.sigma_partials    = pass.sigma_partials;
        reverse_pass.y_node            = node_of_var_scalar(y_argument);
        reverse_pass.mu_node           = node_of_var_scalar(mu_argument);
        reverse_pass.sigma_node        = node_of_var_scalar(sigma_argument);
        if constexpr (pass_t::sum_location_partials) {
            reverse_pass.summed_location_partial = simd_sum_of_lanes(pass.sum_of_location_partials);
        }
        if constexpr (sigma_argument_t::is_var && !sigma_argument_t::is_vector) {
            // d/d sigma = sum (z^2 - 1) / sigma
            reverse_pass.summed_sigma_partial = (sum_of_squared_standardised_residuals - n) / sigma_scalar;
        }
        // (moved: callback_vari stores the functor by value only when it is passed as an rvalue)
        return stan::math::make_callback_var(log_density, std::move(reverse_pass));
    }

}

}  // namespace fused_normal
}  // namespace nico::math

// ---- the Stan-facing function (global scope; stanc calls it as  nico_normal_lpdf<propto__>(y, mu, sigma,
// pstream__), with no forward declaration in the generated code)
namespace nico::math {

template <bool propto__, typename T_y, typename T_loc, typename T_scale>
inline stan::return_type_t<T_y, T_loc, T_scale> nico_normal_lpdf(const T_y& y, const T_loc& mu,
                                                                  const T_scale& sigma,
                                                                  std::ostream* /*pstream__*/) {

    return nico::math::fused_normal::fused_normal_log_density<propto__>(y, mu, sigma);

}

}  // namespace nico::math

using nico::math::nico_normal_lpdf;
























