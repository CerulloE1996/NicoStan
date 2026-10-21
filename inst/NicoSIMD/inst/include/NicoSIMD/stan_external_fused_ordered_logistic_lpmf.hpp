#pragma once
//
// -| --------- BayesMVP fused ordered logistic log probability (AVX-512 / AVX2), an external Stan lpmf --------
//
// nico_ordered_logistic_lpmf(k | eta, c): a drop-in replacement for Stan's ordered_logistic_lpmf (the same
// arguments, values, argument checks and errors), i.e. the log probability
//
//     sum_i log( inv_logit(eta_i - c_{k_i - 1}) - inv_logit(eta_i - c_{k_i}) ),    c_0 = -inf,  c_K = +inf,
//
// of categories k_i in 1, ..., K with ordered cutpoints c_1 < ... < c_{K-1} (one cutpoint vector for every
// response, or one per response), computed in ONE fused SIMD pass with the BayesMVP kernels (nico_exp_1,
// nico_log_1), which gives the value and, for var arguments, every partial (eta and the cutpoints) on ONE
// autodiff node (one callback vari, one reverse-pass loop).
//
// Formulation, per response of category k, with u = eta - c_{k-1} and v = eta - c_k (so u > v):
//
//     log P       =  min(u, 0) - max(v, 0) - log1p(s) + L_k,    s = t_u + t_v + t_u t_v,
//                                                               t_u = exp(-|u|), t_v = exp(-|v|),
//     L_k         =  log1m_exp(c_{k-1} - c_k)                   (0 for k = 1 and k = K),
//     d / d eta   =  F(-u) - F(v)                               (F = inv_logit),
//     d / d c_k-1 = -(A_k + F(-u)),  d / d c_k = A_k + F(v),    A_k = 1 / expm1(c_k - c_{k-1})
//                                                               (0 for k = 1 and k = K).
//
//   * log1p(t_u) + log1p(t_v) = log1p(s): ONE log per response (nico_log_1 of w = 1 + s, plus the rounding
//     correction (w - 1) - s, so that log1p(s) keeps its relative precision when s is small).
//   * F(-u) = t_u / (1 + t_u) (u > 0) or 1 / (1 + t_u) (u <= 0); F(v) = t_v / (1 + t_v) (v < 0) or
//     1 / (1 + t_v) (v >= 0); 1 / (1 + t_u) and 1 / (1 + t_v) come from ONE division, 1 / w.
//   * Every term of log P is <= 0 (no cancellation), and every partial is a sum of positive terms, apart from
//     d / d eta, a difference of two inv_logit values; t_u and t_v never overflow (arguments <= 0).
//   * L_k and A_k depend on the cutpoints only, and are computed with Stan's log1m_exp and std::expm1 from the
//     cutpoint gap itself (so that near-equal cutpoints keep full precision): with one cutpoint vector, once
//     per category with responses per call (the responses enter through the number of responses in each
//     category); with one cutpoint vector per response, once per distinct (lower, upper) cutpoint pair,
//     through a thread-local table keyed by the two values (e.g. c[item]: once per item and category).
//   * The first and last categories have u = +inf and v = -inf; the exp arguments are clamped at -708 (the
//     range of nico_exp_1_wo_checks) and t is set to 0 below it, so that these terms are exactly 0.
//
// Precision: nico_exp_1 has a relative error of about 2e-14 and nico_log_1 (the log of BayesMVP's kernel
// files) about 1.6e-8, so the log1p(s) term of each response has an absolute error of at most about 5e-9; the
// other terms of the value, and all of the partials, are computed in double precision.
//
// Stan declarations (the signatures of Stan's ordered_logistic_lpmf; in the functions block, no body, and
// compiled with --allow-undefined):
//
//     real nico_ordered_logistic_lpmf(int k, real eta, vector c);
//     real nico_ordered_logistic_lpmf(array[] int k, vector eta, vector c);
//     real nico_ordered_logistic_lpmf(array[] int k, vector eta, array[] vector c);
//
// with data or parameters for eta and c; stanc calls them as nico_ordered_logistic_lpmf<propto__>(k, eta, c,
// pstream__), e.g. for  y ~ nico_ordered_logistic(eta, c)  or  nico_ordered_logistic_lupmf(y | eta, c).
// The argument checks, errors and values are those of Stan's ordered_logistic_lpmf (the checks of eta and k
// run inside the fused pass and throw Stan's own errors afterwards); with propto = true and data eta and c the
// result is 0, as Stan's. An array of cutpoint vectors holds one vector per response (or ONE vector, used for
// every response).
//
// Memory (forward pass, on the arena): one cutpoint vector: per response 8 (eta vari pointer) + 8 (eta
// partial) bytes, plus K - 1 cutpoint vari pointers and partials; one cutpoint vector per response: also 32
// bytes per response for its two cutpoints (adjoint pointers and partials). The u, v of the responses go
// through BayesMVP's thread-local scratch. Reverse pass: one callback, adj += adj(result) * partial.
//
// Include: at the end of stan_external_functions.hpp (it uses its namespace, SIMD type, lane count, kernels
// and scratch); the include below makes it self-sufficient when it is included on its own.
//
// ------------------------------------------------------------------------------------------------------------

#include "stan_external_functions.hpp"

#include <cstdint>
#include <cstring>
#include <limits>
#include <ostream>
#include <type_traits>
#include <vector>

namespace nico::math {
namespace fused_ordered_logistic {

using stan::math::var;
using stan::math::vari;

static constexpr int    lanes                      = BMVP_LANES;
static constexpr double smallest_fast_exp_argument = -708.0;   // nico_exp_1_wo_checks is valid down to -708.39
static constexpr double positive_infinity          = std::numeric_limits<double>::infinity();
static constexpr const char* function_name         = "nico_ordered_logistic";
// one cutpoint vector: up to this K, the per-cutpoint sums and the category counts are accumulated in the SIMD
// pass with category masks (O(K) per block of lanes); above it, in a scalar pass over the responses (O(1) each)
static constexpr int    largest_K_with_category_masks = 16;

// ---- SIMD primitives: 8 lanes (AVX-512) or 4 lanes (AVX2). Categories are loaded as int32 lanes.
#if BMVP_LANES == 8
using simd_mask_t     = __mmask8;
using simd_category_t = __m256i;
ALWAYS_INLINE simd_t broadcast(const double x)                { return _mm512_set1_pd(x); }
ALWAYS_INLINE simd_t add(const simd_t a, const simd_t b)      { return _mm512_add_pd(a, b); }
ALWAYS_INLINE simd_t subtract(const simd_t a, const simd_t b) { return _mm512_sub_pd(a, b); }
ALWAYS_INLINE simd_t multiply(const simd_t a, const simd_t b) { return _mm512_mul_pd(a, b); }
ALWAYS_INLINE simd_t divide(const simd_t a, const simd_t b)   { return _mm512_div_pd(a, b); }
ALWAYS_INLINE simd_t minimum(const simd_t a, const simd_t b)  { return _mm512_min_pd(a, b); }
ALWAYS_INLINE simd_t maximum(const simd_t a, const simd_t b)  { return _mm512_max_pd(a, b); }
ALWAYS_INLINE simd_t multiply_add(const simd_t a, const simd_t b, const simd_t c) {

    return _mm512_fmadd_pd(a, b, c);

}
ALWAYS_INLINE simd_t negative_absolute_value(const simd_t a) { return _mm512_or_pd(a, _mm512_set1_pd(-0.0)); }
ALWAYS_INLINE simd_mask_t greater_than(const simd_t a, const simd_t b) {

    return _mm512_cmp_pd_mask(a, b, _CMP_GT_OQ);

}
ALWAYS_INLINE simd_mask_t less_than(const simd_t a, const simd_t b) {

    return _mm512_cmp_pd_mask(a, b, _CMP_LT_OQ);

}
ALWAYS_INLINE simd_t zero_where_false(const simd_mask_t m, const simd_t a) { return _mm512_maskz_mov_pd(m, a); }
// m ? factor * a : a
ALWAYS_INLINE simd_t times_factor_where_true(const simd_mask_t m, const simd_t factor, const simd_t a) {

    return _mm512_mask_mul_pd(a, m, factor, a);

}
ALWAYS_INLINE simd_t add_where_true(const simd_t sum, const simd_mask_t m, const simd_t a) {

    return _mm512_mask_add_pd(sum, m, sum, a);

}
ALWAYS_INLINE simd_t subtract_where_true(const simd_t sum, const simd_mask_t m, const simd_t a) {

    return _mm512_mask_sub_pd(sum, m, sum, a);

}
ALWAYS_INLINE double horizontal_sum(const simd_t a) { return _mm512_reduce_add_pd(a); }
ALWAYS_INLINE simd_category_t load_categories(const int* p) {

    return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));

}
ALWAYS_INLINE simd_mask_t category_equals(const simd_category_t k, const int category) {

    return _mm256_cmpeq_epi32_mask(k, _mm256_set1_epi32(category));

}
ALWAYS_INLINE int count_true(const simd_mask_t m) { return __builtin_popcount(static_cast<unsigned>(m)); }
ALWAYS_INLINE simd_t first_lane_x_other_lanes(const double x, const double other) {

    return _mm512_mask_blend_pd(0x01, _mm512_set1_pd(other), _mm512_set1_pd(x));

}
ALWAYS_INLINE double first_lane(const simd_t a) { return _mm512_cvtsd_f64(a); }
#else
using simd_mask_t     = __m256d;
using simd_category_t = __m128i;
ALWAYS_INLINE simd_t broadcast(const double x)                { return _mm256_set1_pd(x); }
ALWAYS_INLINE simd_t add(const simd_t a, const simd_t b)      { return _mm256_add_pd(a, b); }
ALWAYS_INLINE simd_t subtract(const simd_t a, const simd_t b) { return _mm256_sub_pd(a, b); }
ALWAYS_INLINE simd_t multiply(const simd_t a, const simd_t b) { return _mm256_mul_pd(a, b); }
ALWAYS_INLINE simd_t divide(const simd_t a, const simd_t b)   { return _mm256_div_pd(a, b); }
ALWAYS_INLINE simd_t minimum(const simd_t a, const simd_t b)  { return _mm256_min_pd(a, b); }
ALWAYS_INLINE simd_t maximum(const simd_t a, const simd_t b)  { return _mm256_max_pd(a, b); }
ALWAYS_INLINE simd_t multiply_add(const simd_t a, const simd_t b, const simd_t c) {

    return _mm256_fmadd_pd(a, b, c);

}
ALWAYS_INLINE simd_t negative_absolute_value(const simd_t a) { return _mm256_or_pd(a, _mm256_set1_pd(-0.0)); }
ALWAYS_INLINE simd_mask_t greater_than(const simd_t a, const simd_t b) { return _mm256_cmp_pd(a, b, _CMP_GT_OQ); }
ALWAYS_INLINE simd_mask_t less_than(const simd_t a, const simd_t b)    { return _mm256_cmp_pd(a, b, _CMP_LT_OQ); }
ALWAYS_INLINE simd_t zero_where_false(const simd_mask_t m, const simd_t a) { return _mm256_and_pd(m, a); }
// m ? factor * a : a
ALWAYS_INLINE simd_t times_factor_where_true(const simd_mask_t m, const simd_t factor, const simd_t a) {

    return _mm256_blendv_pd(a, _mm256_mul_pd(factor, a), m);

}
ALWAYS_INLINE simd_t add_where_true(const simd_t sum, const simd_mask_t m, const simd_t a) {

    return _mm256_add_pd(sum, _mm256_and_pd(m, a));

}
ALWAYS_INLINE simd_t subtract_where_true(const simd_t sum, const simd_mask_t m, const simd_t a) {

    return _mm256_sub_pd(sum, _mm256_and_pd(m, a));

}
ALWAYS_INLINE double horizontal_sum(const simd_t a) {

    const __m128d pair_sums = _mm_add_pd(_mm256_castpd256_pd128(a), _mm256_extractf128_pd(a, 1));
    return _mm_cvtsd_f64(_mm_add_sd(pair_sums, _mm_unpackhi_pd(pair_sums, pair_sums)));

}
ALWAYS_INLINE simd_category_t load_categories(const int* p) {

    return _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));

}
ALWAYS_INLINE simd_mask_t category_equals(const simd_category_t k, const int category) {

    return _mm256_castsi256_pd(_mm256_cvtepi32_epi64(_mm_cmpeq_epi32(k, _mm_set1_epi32(category))));

}
ALWAYS_INLINE int count_true(const simd_mask_t m) { return __builtin_popcount(_mm256_movemask_pd(m)); }
ALWAYS_INLINE simd_t first_lane_x_other_lanes(const double x, const double other) {

    return _mm256_blend_pd(_mm256_set1_pd(other), _mm256_set1_pd(x), 0x1);

}
ALWAYS_INLINE double first_lane(const simd_t a) { return _mm256_cvtsd_f64(a); }
#endif

ALWAYS_INLINE simd_t nico_exp(const simd_t a) { return BMVP_KERNEL_NAME(nico_exp_1_wo_checks)(a); }
ALWAYS_INLINE simd_t nico_log(const simd_t a) { return BMVP_KERNEL_NAME(nico_log_1_wo_checks)(a); }

// ---- exp(-|x|) for any x (including +-inf): nico_exp_1 on [-708, 0], and 0 below (exp(-708) = 3.3e-308)
ALWAYS_INLINE simd_t exp_of_negative_absolute_value(const simd_t x) {

    const simd_t argument = negative_absolute_value(x);
    const simd_t smallest = broadcast(smallest_fast_exp_argument);
    return zero_where_false(greater_than(argument, smallest), nico_exp(maximum(argument, smallest)));

}

// ---- one block of LANES responses, from u = eta - c_{k-1} and v = eta - c_k: the log probability without its
// cutpoint term L_k, and (with_partials) F(-u) = inv_logit(c_{k-1} - eta) and F(v) = inv_logit(eta - c_k)
struct block_result_t {
    simd_t log_probability_without_L;
    simd_t inv_logit_of_minus_u;
    simd_t inv_logit_of_v;
};

template <bool with_partials>
ALWAYS_INLINE block_result_t evaluate_block(const simd_t u, const simd_t v) {

    const simd_t one  = broadcast(1.0);
    const simd_t zero = broadcast(0.0);
    const simd_t t_u  = exp_of_negative_absolute_value(u);
    const simd_t t_v  = exp_of_negative_absolute_value(v);
    const simd_t s    = multiply_add(t_u, t_v, add(t_u, t_v));                 // (1 + t_u)(1 + t_v) - 1
    const simd_t w    = add(one, s);
    const simd_t log1p_s = subtract(nico_log(w), subtract(subtract(w, one), s));
    block_result_t result;
    result.log_probability_without_L = subtract(subtract(minimum(u, zero), maximum(v, zero)), log1p_s);
    if constexpr (with_partials) {
        const simd_t inverse_of_w            = divide(one, w);
        const simd_t inverse_of_one_plus_t_u = multiply(add(one, t_v), inverse_of_w);
        const simd_t inverse_of_one_plus_t_v = multiply(add(one, t_u), inverse_of_w);
        result.inv_logit_of_minus_u = times_factor_where_true(greater_than(u, zero), t_u,
                                                              inverse_of_one_plus_t_u);
        result.inv_logit_of_v       = times_factor_where_true(less_than(v, zero), t_v,
                                                              inverse_of_one_plus_t_v);
    } else {
        result.inv_logit_of_minus_u = zero;
        result.inv_logit_of_v       = zero;
    }
    return result;

}

// ---- per-call tables of the K categories (index = category; index 0: padding lanes, which contribute 0)
struct category_tables_t {
    std::vector<double> lower_cutpoint;                 // c_{k-1} (-inf for k = 0, 1)
    std::vector<double> upper_cutpoint;                 // c_k     (+inf for k = 0, K)
    std::vector<double> log1m_exp_of_cutpoint_gap;      // L_k (0 for k = 0, 1, K)
    std::vector<double> inverse_expm1_of_cutpoint_gap;  // A_k (0 for k = 0, 1, K)
    std::vector<long long> n_responses;                 // responses of each interior category
    std::vector<double> cutpoint_sums;  // K - 1 SIMD accumulators: sum F(v) (category m) - F(-u) (category m+1)
    std::vector<double> inv_logit_of_minus_u_of_each_response;  // K > largest_K_with_category_masks only
    std::vector<double> inv_logit_of_v_of_each_response;
    void resize(const int K) {

        lower_cutpoint.assign(K + 1, -positive_infinity);
        upper_cutpoint.assign(K + 1, positive_infinity);
        log1m_exp_of_cutpoint_gap.assign(K + 1, 0.0);
        inverse_expm1_of_cutpoint_gap.assign(K + 1, 0.0);
        n_responses.assign(K + 1, 0);
        cutpoint_sums.assign(static_cast<size_t>(K > 1 ? K - 1 : 1) * lanes, 0.0);

    }
};
inline category_tables_t& category_tables() { thread_local category_tables_t tables; return tables; }

// ---- thread-local buffers: the categories of a single int k (repeated for every response), and the A of each
// response (one cutpoint vector per response)
inline std::vector<int>& repeated_category() { thread_local std::vector<int> buffer; return buffer; }
inline std::vector<double>& inverse_expm1_of_gap_of_each_response() {

    thread_local std::vector<double> buffer;
    return buffer;

}

// ---- one cutpoint vector per response, and one response: L = log1m_exp(c_lower - c_upper) and
// A = 1 / expm1(c_upper - c_lower) of each (c_lower, c_upper) pair, kept in a thread-local direct-mapped table
// keyed by the two values (exact comparison; a pure function of the key, so a stored entry is valid in every
// later call). When many responses share a few cutpoint vectors (e.g. c[item], or one call per response with
// the same cutpoints), every pair is computed once instead of once per response.
struct cutpoint_gap_constants_cache_t {
    static constexpr int n_entries = 1024;
    struct entry_t {
        double lower;
        double upper;
        double log1m_exp_of_gap;
        double inverse_expm1_of_gap;
    };
    entry_t entries[n_entries];
    cutpoint_gap_constants_cache_t() {

        for (entry_t& e : entries) {
            e.lower = std::numeric_limits<double>::quiet_NaN();   // (a NaN key never compares equal: empty)
            e.upper = std::numeric_limits<double>::quiet_NaN();
        }

    }
    ALWAYS_INLINE const entry_t& constants_of(const double lower, const double upper) {

        std::uint64_t lower_bits, upper_bits;
        std::memcpy(&lower_bits, &lower, sizeof(double));
        std::memcpy(&upper_bits, &upper, sizeof(double));
        std::uint64_t h = lower_bits * 0x9E3779B97F4A7C15ULL ^ (upper_bits + 0x632BE59BD9B4E019ULL);
        h ^= h >> 31;
        h *= 0xBF58476D1CE4E5B9ULL;
        h ^= h >> 29;
        entry_t& e = entries[h & (n_entries - 1)];
        if (!(e.lower == lower && e.upper == upper)) {
            e.lower = lower;
            e.upper = upper;
            e.log1m_exp_of_gap = stan::math::log1m_exp(lower - upper);
            e.inverse_expm1_of_gap = 1.0 / std::expm1(upper - lower);
        }
        return e;

    }
};
inline cutpoint_gap_constants_cache_t& cutpoint_gap_constants_cache() {

    thread_local cutpoint_gap_constants_cache_t cache;
    return cache;

}

inline void fill_category_tables(const double* cutpoints, const int K, category_tables_t& tables) {

    tables.resize(K);
    for (int k = 1; k <= K; ++k) {
        if (k >= 2) tables.lower_cutpoint[k] = cutpoints[k - 2];
        if (k <= K - 1) tables.upper_cutpoint[k] = cutpoints[k - 1];
    }

}

// L_k and A_k of the interior categories with at least one response (counted by the fused pass); the others
// stay 0 and are never used (they enter the value and the partials multiplied by the category's responses)
inline void fill_cutpoint_gap_constants_of_categories_with_responses(const double* cutpoints, const int K,
                                                                     category_tables_t& tables) {

    for (int k = 2; k <= K - 1; ++k) {
        if (tables.n_responses[k] == 0) continue;
        tables.log1m_exp_of_cutpoint_gap[k]     = stan::math::log1m_exp(cutpoints[k - 2] - cutpoints[k - 1]);
        tables.inverse_expm1_of_cutpoint_gap[k] = 1.0 / std::expm1(cutpoints[k - 1] - cutpoints[k - 2]);
    }

}

// ---- the fused pass over the n responses (u, v, k contiguous), with ONE cutpoint vector. Returns the sum of
// the log probabilities without the L_k terms; writes F(-u) - F(v) to eta_partials (write_eta_partials);
// accumulates the cutpoint sums (sum_cutpoint_terms) and counts the responses of the interior categories
// (K >= 3).
template <bool write_eta_partials, bool sum_cutpoint_terms>
inline double fused_pass(const double* u, const double* v, const int* k, const int n, const int K,
                         double* eta_partials, category_tables_t& tables) {

    constexpr bool with_partials = write_eta_partials || sum_cutpoint_terms;
    const bool count_interior = (K >= 3);
    const bool use_category_masks = (K <= largest_K_with_category_masks);
    double* cutpoint_sums = tables.cutpoint_sums.data();
    long long* n_responses = tables.n_responses.data();
    double* inv_logit_of_minus_u_of_each_response = nullptr;
    double* inv_logit_of_v_of_each_response = nullptr;
    if (sum_cutpoint_terms && !use_category_masks) {
        if (static_cast<int>(tables.inv_logit_of_v_of_each_response.size()) < n + lanes) {
            tables.inv_logit_of_minus_u_of_each_response.resize(n + lanes);
            tables.inv_logit_of_v_of_each_response.resize(n + lanes);
        }
        inv_logit_of_minus_u_of_each_response = tables.inv_logit_of_minus_u_of_each_response.data();
        inv_logit_of_v_of_each_response = tables.inv_logit_of_v_of_each_response.data();
    }
    simd_t log_probability_sum = broadcast(0.0);

    // first: the index of the block's first response (the F values of each response, without category masks)
    const auto process_block = [&](const double* u_block, const double* v_block, const int* k_block,
                                   double* eta_partials_block, const int first) {

        const block_result_t r = evaluate_block<with_partials>(simd_loadu(u_block), simd_loadu(v_block));
        log_probability_sum = add(log_probability_sum, r.log_probability_without_L);
        if constexpr (write_eta_partials) {
            simd_storeu(eta_partials_block, subtract(r.inv_logit_of_minus_u, r.inv_logit_of_v));
        }
        if (!use_category_masks) {
            if constexpr (sum_cutpoint_terms) {
                simd_storeu(inv_logit_of_minus_u_of_each_response + first, r.inv_logit_of_minus_u);
                simd_storeu(inv_logit_of_v_of_each_response + first, r.inv_logit_of_v);
            }
        } else if (sum_cutpoint_terms || count_interior) {
            const simd_category_t categories = load_categories(k_block);
            if constexpr (sum_cutpoint_terms) {
                // cutpoint m: the upper cutpoint of category m (+F(v)), the lower one of category m + 1 (-F(-u))
                simd_mask_t is_category_m = category_equals(categories, 1);
                for (int m = 1; m <= K - 1; ++m) {
                    const simd_mask_t is_category_m_plus_1 = category_equals(categories, m + 1);
                    simd_t sum = simd_loadu(cutpoint_sums + (m - 1) * lanes);
                    sum = add_where_true(sum, is_category_m, r.inv_logit_of_v);
                    sum = subtract_where_true(sum, is_category_m_plus_1, r.inv_logit_of_minus_u);
                    simd_storeu(cutpoint_sums + (m - 1) * lanes, sum);
                    if (m >= 2) n_responses[m] += count_true(is_category_m);
                    is_category_m = is_category_m_plus_1;
                }
            } else {
                for (int m = 2; m <= K - 1; ++m) n_responses[m] += count_true(category_equals(categories, m));
            }
        }

    };

    int i = 0;
    for (; i + lanes <= n; i += lanes) {
        process_block(u + i, v + i, k + i, write_eta_partials ? eta_partials + i : nullptr, i);
    }
    if (i < n) {
        // a partial final block: padding lanes have u = +inf, v = -inf and category 0, and contribute exactly 0
        alignas(64) double u_padded[lanes], v_padded[lanes], eta_partials_padded[lanes];
        alignas(64) int k_padded[lanes];
        const int n_valid = n - i;
        for (int j = 0; j < lanes; ++j) {
            const bool valid = (j < n_valid);
            u_padded[j] = valid ? u[i + j] : positive_infinity;
            v_padded[j] = valid ? v[i + j] : -positive_infinity;
            k_padded[j] = valid ? k[i + j] : 0;
        }
        process_block(u_padded, v_padded, k_padded, eta_partials_padded, i);
        if constexpr (write_eta_partials) {
            for (int j = 0; j < n_valid; ++j) eta_partials[i + j] = eta_partials_padded[j];
        }
    }
    // without category masks (K > largest_K_with_category_masks): the counts and the per-cutpoint sums in a
    // scalar pass over the responses (the categories are in 1, ..., K here: checked before the fused pass); the
    // sums go to the first lane of each cutpoint's accumulator
    if (!use_category_masks) {
        if (count_interior) {
            for (int r = 0; r < n; ++r) n_responses[k[r]] += 1;
        }
        if constexpr (sum_cutpoint_terms) {
            for (int r = 0; r < n; ++r) {
                const int category = k[r];
                if (category <= K - 1) {
                    cutpoint_sums[(category - 1) * lanes] += inv_logit_of_v_of_each_response[r];
                }
                if (category >= 2) {
                    cutpoint_sums[(category - 2) * lanes] -= inv_logit_of_minus_u_of_each_response[r];
                }
            }
        }
    }
    return horizontal_sum(log_probability_sum);

}

// ---- the fused pass with ONE CUTPOINT VECTOR PER RESPONSE (u, v and the A of each response contiguous).
// Returns the sum of the log probabilities without the L terms; writes F(-u) - F(v) to eta_partials
// (write_eta_partials) and, for the cutpoints (write_cutpoint_partials), A + F(-u) (the minus partial of the
// lower cutpoint) and A + F(v) (the partial of the upper cutpoint) of each response.
template <bool write_eta_partials, bool write_cutpoint_partials>
inline double fused_pass_cutpoint_vector_per_response(const double* u, const double* v,
                                                      const double* inverse_expm1_of_gap, const int n,
                                                      double* eta_partials, double* minus_lower_partials,
                                                      double* upper_partials) {

    constexpr bool with_partials = write_eta_partials || write_cutpoint_partials;
    simd_t log_probability_sum = broadcast(0.0);
    const auto process_block = [&](const double* u_block, const double* v_block, const double* A_block,
                                   double* eta_partials_block, double* minus_lower_block, double* upper_block) {

        const block_result_t r = evaluate_block<with_partials>(simd_loadu(u_block), simd_loadu(v_block));
        log_probability_sum = add(log_probability_sum, r.log_probability_without_L);
        if constexpr (write_eta_partials) {
            simd_storeu(eta_partials_block, subtract(r.inv_logit_of_minus_u, r.inv_logit_of_v));
        }
        if constexpr (write_cutpoint_partials) {
            const simd_t A = simd_loadu(A_block);
            simd_storeu(minus_lower_block, add(A, r.inv_logit_of_minus_u));
            simd_storeu(upper_block, add(A, r.inv_logit_of_v));
        }

    };
    int i = 0;
    for (; i + lanes <= n; i += lanes) {
        process_block(u + i, v + i, inverse_expm1_of_gap + i, write_eta_partials ? eta_partials + i : nullptr,
                      write_cutpoint_partials ? minus_lower_partials + i : nullptr,
                      write_cutpoint_partials ? upper_partials + i : nullptr);
    }
    if (i < n) {
        alignas(64) double u_padded[lanes], v_padded[lanes], A_padded[lanes];
        alignas(64) double eta_partials_padded[lanes], minus_lower_padded[lanes], upper_padded[lanes];
        const int n_valid = n - i;
        for (int j = 0; j < lanes; ++j) {
            const bool valid = (j < n_valid);
            u_padded[j] = valid ? u[i + j] : positive_infinity;
            v_padded[j] = valid ? v[i + j] : -positive_infinity;
            A_padded[j] = valid ? inverse_expm1_of_gap[i + j] : 0.0;
        }
        process_block(u_padded, v_padded, A_padded, eta_partials_padded, minus_lower_padded, upper_padded);
        for (int j = 0; j < n_valid; ++j) {
            if constexpr (write_eta_partials) eta_partials[i + j] = eta_partials_padded[j];
            if constexpr (write_cutpoint_partials) {
                minus_lower_partials[i + j] = minus_lower_padded[j];
                upper_partials[i + j] = upper_padded[j];
            }
        }
    }
    return horizontal_sum(log_probability_sum);

}

// ---- eta: an Eigen column vector (or expression) of double or var, or a scalar
template <typename T>
struct is_eta_argument
    : std::integral_constant<bool, stan::is_eigen_col_vector<T>::value
                                   || std::is_arithmetic<std::decay_t<T>>::value
                                   || stan::is_var<std::decay_t<T>>::value> {};

template <typename T_eta>
inline int size_of_eta(const T_eta& eta) {

    if constexpr (stan::is_eigen<T_eta>::value) return static_cast<int>(eta.size());
    else return 1;

}

// the value of response i's eta (and, for var, its vari)
template <typename T_eta>
ALWAYS_INLINE double value_of_eta(const T_eta& eta, const int i) {

    if constexpr (stan::is_eigen<T_eta>::value) return stan::math::value_of(eta.coeff(i));
    else return stan::math::value_of(eta);

}
template <typename T_eta>
ALWAYS_INLINE vari* vari_of_eta(const T_eta& eta, const int i) {

    if constexpr (stan::is_eigen<T_eta>::value) return eta.coeff(i).vi_;
    else return eta.vi_;

}

// the categories as a contiguous int array (a single int k: repeated for every response)
template <typename T_k>
inline const int* contiguous_categories(const T_k& k, const int n) {

    if constexpr (std::is_integral<std::decay_t<T_k>>::value) {
        std::vector<int>& repeated = repeated_category();
        repeated.assign(n, static_cast<int>(k));
        return repeated.data();
    } else {
        return k.data();
    }

}

// a scalar eta is one response: Stan's (int k | real eta, vector c)
template <typename T_k, typename T_eta>
constexpr void static_check_that_a_scalar_eta_has_a_single_k() {

    static_assert(stan::is_eigen<T_eta>::value || std::is_integral<std::decay_t<T_k>>::value,
                  "nico_ordered_logistic_lpmf: a real eta takes a single int k "
                  "(Stan's int k | real eta, vector c)");

}

// ---- ONE response with one cutpoint vector (Stan's (int k | real eta, vector c), and any call with one
// response): the same block computation on registers (lane 0; the other lanes are padding), without the tables
// and buffers of the general path, and with L and A of the response's category from the thread-local table of
// cutpoint pairs. The checks, errors and value are those of the general path.
template <bool propto, typename T_k, typename T_eta, typename T_c>
inline stan::return_type_t<T_eta, T_c> ordered_logistic_one_response(const T_k& k, const T_eta& eta,
                                                                     const T_c& c) {

    using return_t = stan::return_type_t<T_eta, T_c>;
    constexpr bool eta_is_var = stan::is_var<stan::scalar_type_t<T_eta>>::value;
    constexpr bool c_is_var   = stan::is_var<stan::scalar_type_t<T_c>>::value;
    const int K = static_cast<int>(c.size()) + 1;
    int category;
    if constexpr (std::is_integral<std::decay_t<T_k>>::value) category = static_cast<int>(k);
    else category = k[0];
    const double eta_value = value_of_eta(eta, 0);
    // Stan's checks, in Stan's order
    stan::math::check_finite(function_name, "Location parameter", eta);
    stan::math::check_bounded(function_name, "Random variable", k, 1, K);
    stan::math::check_ordered(function_name, "Cut-points", c);
    if (K > 1) {
        if (K > 2) stan::math::check_finite(function_name, "Final cut-point", c.coeff(K - 2));
        stan::math::check_finite(function_name, "First cut-point", c.coeff(0));
    }
    if (!stan::math::include_summand<propto, T_eta, T_c>::value) return return_t(0.0);
    const bool has_lower = (category >= 2);
    const bool has_upper = (category <= K - 1);
    const double lower = has_lower ? stan::math::value_of(c.coeff(category - 2)) : -positive_infinity;
    const double upper = has_upper ? stan::math::value_of(c.coeff(category - 1)) : positive_infinity;
    constexpr bool with_partials = eta_is_var || c_is_var;
    const block_result_t r = evaluate_block<with_partials>(
                                   first_lane_x_other_lanes(eta_value - lower, positive_infinity),
                                   first_lane_x_other_lanes(eta_value - upper, -positive_infinity));
    double log_probability = first_lane(r.log_probability_without_L);
    double A = 0.0;
    if (has_lower && has_upper) {
        const cutpoint_gap_constants_cache_t::entry_t& constants =
                                                        cutpoint_gap_constants_cache().constants_of(lower, upper);
        log_probability += constants.log1m_exp_of_gap;
        A = constants.inverse_expm1_of_gap;
    }
    if constexpr (!with_partials) {
        return log_probability;
    } else {
        const double inv_logit_of_minus_u = first_lane(r.inv_logit_of_minus_u);
        const double inv_logit_of_v = first_lane(r.inv_logit_of_v);
        const double eta_partial = inv_logit_of_minus_u - inv_logit_of_v;
        const double lower_partial = -(A + inv_logit_of_minus_u);
        const double upper_partial = A + inv_logit_of_v;
        vari* eta_vari = nullptr;
        vari* lower_vari = nullptr;
        vari* upper_vari = nullptr;
        if constexpr (eta_is_var) eta_vari = vari_of_eta(eta, 0);
        if constexpr (c_is_var) {
            if (has_lower) lower_vari = c.coeff(category - 2).vi_;
            if (has_upper) upper_vari = c.coeff(category - 1).vi_;
        }
        return stan::math::make_callback_var(
            log_probability,
            [eta_vari, eta_partial, lower_vari, lower_partial, upper_vari, upper_partial](auto& result_vari) {

                const double adjoint = result_vari.adj_;
                if constexpr (eta_is_var) eta_vari->adj_ += adjoint * eta_partial;
                if constexpr (c_is_var) {
                    if (lower_vari != nullptr) lower_vari->adj_ += adjoint * lower_partial;
                    if (upper_vari != nullptr) upper_vari->adj_ += adjoint * upper_partial;
                }

            });
    }

}

// ---- the whole function for ONE cutpoint vector (Stan's (int | real, vector) and
// (array[] int | vector, vector))
template <bool propto, typename T_k, typename T_eta, typename T_c>
inline stan::return_type_t<T_eta, T_c> ordered_logistic_one_cutpoint_vector(const T_k& k, const T_eta& eta_in,
                                                                            const T_c& c_in) {

    using return_t = stan::return_type_t<T_eta, T_c>;
    constexpr bool eta_is_var = stan::is_var<stan::scalar_type_t<T_eta>>::value;
    constexpr bool c_is_var   = stan::is_var<stan::scalar_type_t<T_c>>::value;
    static_check_that_a_scalar_eta_has_a_single_k<T_k, T_eta>();

    stan::math::check_consistent_sizes(function_name, "Integers", k, "Locations", eta_in);
    const auto& eta = stan::math::to_ref(eta_in);
    const auto& c   = stan::math::to_ref(c_in);
    const int n = size_of_eta(eta);
    if (n == 0) return return_t(0.0);
    if (n == 1) return ordered_logistic_one_response<propto>(k, eta, c);
    const int K = static_cast<int>(c.size()) + 1;
    const int* categories = contiguous_categories(k, n);

    // the cutpoints: Stan's checks (ordered; first and last finite), then the per-category tables
    scratch_t& sc = scratch();
    sc.ensure(n > K ? n : K);
    double* cutpoint_values = sc.c.data();
    for (int m = 0; m < K - 1; ++m) cutpoint_values[m] = stan::math::value_of(c.coeff(m));
    const auto check_cutpoints = [&]() {

        stan::math::check_ordered(function_name, "Cut-points", c);
        if (K > 1) {
            if (K > 2) stan::math::check_finite(function_name, "Final cut-point", cutpoint_values[K - 2]);
            stan::math::check_finite(function_name, "First cut-point", cutpoint_values[0]);
        }

    };
    if (!stan::math::include_summand<propto, T_eta, T_c>::value) {
        // (data eta and c with propto: only the checks, then 0, as Stan's ordered_logistic_lpmf)
        stan::math::check_finite(function_name, "Location parameter", eta);
        stan::math::check_bounded(function_name, "Random variable", k, 1, K);
        check_cutpoints();
        return return_t(0.0);
    }
    category_tables_t& tables = category_tables();
    fill_category_tables(cutpoint_values, K, tables);

    // the u, v of every response (thread-local scratch), and the eta vari pointers (arena) for var eta
    double* u = sc.a.data();
    double* v = sc.b.data();
    vari** eta_varis = nullptr;
    if constexpr (eta_is_var) {
        eta_varis = stan::math::ChainableStack::instance_->memalloc_.alloc_array<vari*>(n);
    }
    const double* lower_cutpoint = tables.lower_cutpoint.data();
    const double* upper_cutpoint = tables.upper_cutpoint.data();
    unsigned any_category_out_of_range = 0;
    unsigned any_eta_not_finite = 0;
    for (int i = 0; i < n; ++i) {
        const unsigned category = static_cast<unsigned>(categories[i]);
        const unsigned out_of_range = (category - 1u >= static_cast<unsigned>(K));
        const unsigned table_index = out_of_range ? 0u : category;
        double eta_value;
        if constexpr (eta_is_var) {
            vari* eta_vari = vari_of_eta(eta, i);
            eta_varis[i] = eta_vari;
            eta_value = eta_vari->val_;
        } else {
            eta_value = value_of_eta(eta, i);
        }
        any_category_out_of_range |= out_of_range;
        any_eta_not_finite |= static_cast<unsigned>(!(std::fabs(eta_value)
                                                      <= std::numeric_limits<double>::max()));
        u[i] = eta_value - lower_cutpoint[table_index];
        v[i] = eta_value - upper_cutpoint[table_index];
    }
    // Stan's checks, in Stan's order, with Stan's messages (only when the fused pass found a problem)
    if (any_eta_not_finite) stan::math::check_finite(function_name, "Location parameter", eta);
    if (any_category_out_of_range) stan::math::check_bounded(function_name, "Random variable", k, 1, K);
    check_cutpoints();

    // the fused pass
    double* eta_partials = nullptr;
    if constexpr (eta_is_var) {
        eta_partials = stan::math::ChainableStack::instance_->memalloc_.alloc_array<double>(n);
    }
    const double sum_without_L = fused_pass<eta_is_var, c_is_var>(u, v, categories, n, K, eta_partials, tables);
    fill_cutpoint_gap_constants_of_categories_with_responses(cutpoint_values, K, tables);
    double log_probability = sum_without_L;
    for (int m = 2; m <= K - 1; ++m) {
        log_probability += static_cast<double>(tables.n_responses[m]) * tables.log1m_exp_of_cutpoint_gap[m];
    }
    if constexpr (!eta_is_var && !c_is_var) {
        return log_probability;
    } else {
        // the cutpoint partials: d / d c_m = sum F(v) (category m) - sum F(-u) (category m + 1)
        //                                    + n_m A_m - n_{m+1} A_{m+1}
        vari** cutpoint_varis = nullptr;
        double* cutpoint_partials = nullptr;
        const int n_cutpoints = K - 1;
        if constexpr (c_is_var) {
            auto& arena = stan::math::ChainableStack::instance_->memalloc_;
            cutpoint_varis = arena.alloc_array<vari*>(n_cutpoints);
            cutpoint_partials = arena.alloc_array<double>(n_cutpoints);
            const double* A = tables.inverse_expm1_of_cutpoint_gap.data();
            const long long* n_responses = tables.n_responses.data();
            for (int m = 1; m <= n_cutpoints; ++m) {
                alignas(64) double lane_sums[lanes];
                simd_store(lane_sums, simd_loadu(tables.cutpoint_sums.data() + (m - 1) * lanes));
                double partial = 0.0;
                for (int j = 0; j < lanes; ++j) partial += lane_sums[j];
                partial += static_cast<double>(n_responses[m]) * A[m];
                if (m + 1 <= K - 1) partial -= static_cast<double>(n_responses[m + 1]) * A[m + 1];
                cutpoint_partials[m - 1] = partial;
                cutpoint_varis[m - 1] = c.coeff(m - 1).vi_;
            }
        }
        return stan::math::make_callback_var(
            log_probability,
            [eta_varis, eta_partials, n, cutpoint_varis, cutpoint_partials, n_cutpoints](auto& result_vari) {

                const double adjoint = result_vari.adj_;
                if constexpr (eta_is_var) {
                    for (int i = 0; i < n; ++i) eta_varis[i]->adj_ += adjoint * eta_partials[i];
                }
                if constexpr (c_is_var) {
                    for (int m = 0; m < n_cutpoints; ++m) {
                        cutpoint_varis[m]->adj_ += adjoint * cutpoint_partials[m];
                    }
                }

            });
    }

}

// ---- the whole function for ONE CUTPOINT VECTOR PER RESPONSE (Stan's (array[] int | vector, array[] vector));
// an array of one vector is that vector for every response, an empty array gives 0 (as Stan)
template <bool propto, typename T_k, typename T_eta, typename T_c_array>
inline stan::return_type_t<T_eta, T_c_array> ordered_logistic_cutpoint_vector_per_response(
                                                  const T_k& k, const T_eta& eta_in, const T_c_array& c) {

    using return_t = stan::return_type_t<T_eta, T_c_array>;
    constexpr bool eta_is_var = stan::is_var<stan::scalar_type_t<T_eta>>::value;
    constexpr bool c_is_var   = stan::is_var<stan::scalar_type_t<T_c_array>>::value;
    static_check_that_a_scalar_eta_has_a_single_k<T_k, T_eta>();

    stan::math::check_consistent_sizes(function_name, "Integers", k, "Locations", eta_in);
    const int n_cutpoint_vectors = static_cast<int>(c.size());
    const int n = size_of_eta(eta_in);
    if (n_cutpoint_vectors > 1) {
        stan::math::check_size_match(function_name, "Length of location variables ", n,
                                     "Number of cutpoint vectors ", n_cutpoint_vectors);
    }
    if (n_cutpoint_vectors == 1) return ordered_logistic_one_cutpoint_vector<propto>(k, eta_in, c[0]);
    const auto& eta = stan::math::to_ref(eta_in);
    if (n_cutpoint_vectors == 0 || n == 0) {
        stan::math::check_finite(function_name, "Location parameter", eta);
        return return_t(0.0);
    }
    const int K = static_cast<int>(c[0].size()) + 1;
    for (int i = 0; i < n; ++i) {
        stan::math::check_size_match(function_name, "One cutpoint set", c[i].size(), "First cutpoint set", K - 1);
    }
    const int* categories = contiguous_categories(k, n);
    const auto check_cutpoints = [&]() {

        for (int i = 0; i < n; ++i) {
            stan::math::check_ordered(function_name, "Cut-points", c[i]);
            if (K > 1) {
                if (K > 2) stan::math::check_finite(function_name, "Final cut-point", c[i].coeff(K - 2));
                stan::math::check_finite(function_name, "First cut-point", c[i].coeff(0));
            }
        }

    };
    if (!stan::math::include_summand<propto, T_eta, T_c_array>::value) {
        stan::math::check_finite(function_name, "Location parameter", eta);
        stan::math::check_bounded(function_name, "Random variable", k, 1, K);
        check_cutpoints();
        return return_t(0.0);
    }

    // per response: u, v (scratch), A (thread-local), L (summed), and the adjoint addresses of its two cutpoints
    // (arena; the first / last category: a sink on the arena)
    scratch_t& sc = scratch();
    sc.ensure(n);
    double* u = sc.a.data();
    double* v = sc.b.data();
    std::vector<double>& A_buffer = inverse_expm1_of_gap_of_each_response();
    if (static_cast<int>(A_buffer.size()) < n) A_buffer.resize(n);
    double* A = A_buffer.data();
    auto& arena = stan::math::ChainableStack::instance_->memalloc_;
    vari** eta_varis = nullptr;
    if constexpr (eta_is_var) eta_varis = arena.alloc_array<vari*>(n);
    double** lower_cutpoint_adjoints = nullptr;
    double** upper_cutpoint_adjoints = nullptr;
    double* adjoint_sink = nullptr;
    if constexpr (c_is_var) {
        lower_cutpoint_adjoints = arena.alloc_array<double*>(n);
        upper_cutpoint_adjoints = arena.alloc_array<double*>(n);
        adjoint_sink = arena.alloc_array<double>(1);
    }
    cutpoint_gap_constants_cache_t& gap_constants = cutpoint_gap_constants_cache();
    double sum_of_L = 0.0;
    unsigned any_category_out_of_range = 0;
    unsigned any_eta_not_finite = 0;
    for (int i = 0; i < n; ++i) {
        const int category = categories[i];
        const unsigned out_of_range = (static_cast<unsigned>(category) - 1u >= static_cast<unsigned>(K));
        const bool has_lower = !out_of_range && (category >= 2);
        const bool has_upper = !out_of_range && (category <= K - 1);
        double eta_value;
        if constexpr (eta_is_var) {
            vari* eta_vari = vari_of_eta(eta, i);
            eta_varis[i] = eta_vari;
            eta_value = eta_vari->val_;
        } else {
            eta_value = value_of_eta(eta, i);
        }
        const double lower = has_lower ? stan::math::value_of(c[i].coeff(category - 2)) : -positive_infinity;
        const double upper = has_upper ? stan::math::value_of(c[i].coeff(category - 1)) : positive_infinity;
        u[i] = eta_value - lower;
        v[i] = eta_value - upper;
        A[i] = 0.0;
        if (has_lower && has_upper) {
            const cutpoint_gap_constants_cache_t::entry_t& constants = gap_constants.constants_of(lower, upper);
            sum_of_L += constants.log1m_exp_of_gap;
            A[i] = constants.inverse_expm1_of_gap;
        }
        if constexpr (c_is_var) {
            lower_cutpoint_adjoints[i] = has_lower ? &(c[i].coeff(category - 2).vi_->adj_) : adjoint_sink;
            upper_cutpoint_adjoints[i] = has_upper ? &(c[i].coeff(category - 1).vi_->adj_) : adjoint_sink;
        }
        any_category_out_of_range |= out_of_range;
        any_eta_not_finite |= static_cast<unsigned>(!(std::fabs(eta_value)
                                                      <= std::numeric_limits<double>::max()));
    }
    if (any_eta_not_finite) stan::math::check_finite(function_name, "Location parameter", eta);
    if (any_category_out_of_range) stan::math::check_bounded(function_name, "Random variable", k, 1, K);
    check_cutpoints();

    double* eta_partials = nullptr;
    double* minus_lower_partials = nullptr;
    double* upper_partials = nullptr;
    if constexpr (eta_is_var) eta_partials = arena.alloc_array<double>(n);
    if constexpr (c_is_var) {
        minus_lower_partials = arena.alloc_array<double>(n);
        upper_partials = arena.alloc_array<double>(n);
    }
    const double log_probability = sum_of_L
                                   + fused_pass_cutpoint_vector_per_response<eta_is_var, c_is_var>(
                                         u, v, A, n, eta_partials, minus_lower_partials, upper_partials);
    if constexpr (!eta_is_var && !c_is_var) {
        return log_probability;
    } else {
        return stan::math::make_callback_var(
            log_probability,
            [eta_varis, eta_partials, lower_cutpoint_adjoints, upper_cutpoint_adjoints, minus_lower_partials,
             upper_partials, n](auto& result_vari) {

                const double adjoint = result_vari.adj_;
                if constexpr (eta_is_var) {
                    for (int i = 0; i < n; ++i) eta_varis[i]->adj_ += adjoint * eta_partials[i];
                }
                if constexpr (c_is_var) {
                    for (int i = 0; i < n; ++i) {
                        *lower_cutpoint_adjoints[i] -= adjoint * minus_lower_partials[i];
                        *upper_cutpoint_adjoints[i] += adjoint * upper_partials[i];
                    }
                }

            });
    }

}

}  // namespace fused_ordered_logistic
}  // namespace nico::math

// ---- the Stan-facing functions (global scope; stanc calls nico_ordered_logistic_lpmf<propto__>(k, eta, c,
// pstream__)): one cutpoint vector, and one cutpoint vector per response
namespace nico::math {

template <bool propto, typename T_k, typename T_eta, typename T_c,
          stan::require_t<std::is_integral<stan::scalar_type_t<T_k>>>* = nullptr,
          stan::require_t<nico::math::fused_ordered_logistic::is_eta_argument<T_eta>>* = nullptr,
          stan::require_eigen_col_vector_t<T_c>* = nullptr>
inline stan::return_type_t<T_eta, T_c> nico_ordered_logistic_lpmf(const T_k& k, const T_eta& eta, const T_c& c,
                                                                   std::ostream* /*pstream__*/) {

    return nico::math::fused_ordered_logistic::ordered_logistic_one_cutpoint_vector<propto>(k, eta, c);

}

}  // namespace nico::math

using nico::math::nico_ordered_logistic_lpmf;

namespace nico::math {

template <bool propto, typename T_k, typename T_eta, typename T_c_array,
          stan::require_t<std::is_integral<stan::scalar_type_t<T_k>>>* = nullptr,
          stan::require_t<nico::math::fused_ordered_logistic::is_eta_argument<T_eta>>* = nullptr,
          stan::require_std_vector_vt<stan::is_eigen_col_vector, T_c_array>* = nullptr>
inline stan::return_type_t<T_eta, T_c_array> nico_ordered_logistic_lpmf(const T_k& k, const T_eta& eta,
                                                                         const T_c_array& c,
                                                                         std::ostream* /*pstream__*/) {

    return nico::math::fused_ordered_logistic::ordered_logistic_cutpoint_vector_per_response<propto>(k, eta,
                                                                                                       c);

}

}  // namespace nico::math

using nico::math::nico_ordered_logistic_lpmf;
























