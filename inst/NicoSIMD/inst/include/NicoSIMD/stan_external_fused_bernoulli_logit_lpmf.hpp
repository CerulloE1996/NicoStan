#pragma once
//
// -| --------- BayesMVP fused Bernoulli-logit log probability, exposed to Stan as an external function ----------
//
// nico_bernoulli_logit_lpmf(y | f) = sum_i [ y_i f_i - log1p_exp(f_i) ], the log probability of Stan's
// bernoulli_logit_lpmf(y | f), computed in ONE pass over the observations with the BayesMVP SIMD kernels
// (fast_and_approx_AVX512_fns.hpp / fast_and_approx_AVX2_fns.hpp, as selected in stan_external_functions.hpp):
//
//     * PRIM overload (f: Eigen column vector or expression of double)  -> double (the value only)
//     * REV  overload (f: Eigen column vector or expression of var)     -> var (ONE autodiff node)
//
// Per block of BMVP_LANES observations, with s_i = 2 y_i - 1 and ntheta_i = s_i f_i (as in Stan):
//
//     e_i            = exp(-|f_i|)                                    one exp, argument in [-708, 0]
//     log p_i        = min(ntheta_i, 0) - log1p(e_i)                  one log1p (also exact for |f_i| = inf)
//     d log p_i/df_i = s_i * inv_logit(-ntheta_i)  (= y_i - inv_logit(f_i))
//                    = s_i * (ntheta_i >= 0 ? e_i : 1) / (1 + e_i)    one division, no cancellation
//
// The REV overload computes the value and the partials in the same pass and keeps, per observation, only the
// operand vari pointer and the partial on the arena (16 bytes); its reverse pass is one loop,
// adj(f_i) += adj(log p) * d log p_i/df_i (scalar and in order, since elements of f may share one vari).
// Stan's own bernoulli_logit_lpmf with var f also builds one node, but evaluates exp, log1p and the partials
// as separate elementwise passes with temporaries, and log1p in scalar code.
//
// Checks as in Stan, with Stan's own check functions (and so the same exceptions): consistent sizes,
// y in {0, 1}, f not NaN. With f of double and propto = true the result is 0, as in Stan.
//
// The partials are exact (y_i - inv_logit(f_i)) for every f_i. Stan's partials differ for y_i = 1 and
// f_i > 20 only: Stan returns -exp(-f_i) there, i.e. the sign is flipped (|difference| < 4.2e-9).
//
// Declared in the Stan functions block (no body; stanc --allow-undefined) as
//
//     real nico_bernoulli_logit_lpmf(array[] int y, vector f);
//
// which also gives  nico_bernoulli_logit_lupmf(y | f)  and  y ~ nico_bernoulli_logit(f).
//
// Included at the end of stan_external_functions.hpp (it uses that file's namespace nico::math, simd_t,
// BMVP_LANES and BMVP_KERNEL_NAME). It also includes stan_external_functions.hpp itself (#pragma once), so a
// user header may include this file alone. Its helpers are in their own namespace,
// nico::math::fused_bernoulli_logit, so that other fused headers included the same way cannot clash.
//
// -----------------------------------------------------------------------------------------------------------

// while this file is open, stan_external_functions.hpp (included next) leaves out the GLM header, which needs this
// file's namespace nico::math::fused_bernoulli_logit (see the end of stan_external_functions.hpp)
#define BMVP_FUSED_BERNOULLI_LOGIT_LPMF_BEING_INCLUDED
#include "stan_external_functions.hpp"

#include <stan/math/rev.hpp>
#include <stan/math/prim.hpp>

#include <Eigen/Dense>
#include "nico_simd_intrinsics.hpp"   //// <immintrin.h> on x86-64; SIMDe (NEON) on ARM
#include <ostream>
#include <type_traits>
#include <vector>

#if !defined(BMVP_LANES) || !defined(BMVP_KERNEL_NAME)
#  error "stan_external_fused_bernoulli_logit_lpmf.hpp: stan_external_functions.hpp must be included first"
#endif

namespace nico::math {
namespace fused_bernoulli_logit {

using stan::math::var;
using stan::math::vari;

static constexpr const char* function_name = "nico_bernoulli_logit_lpmf";

// ---- exp() is evaluated at max(-|f|, -708): the exponent-bit shortcut of nico_exp_1_wo_checks is exact down to
//      -708.4, and exp(-708) = 3.3e-308 is already zero next to the other terms wherever |f| >= 708
static constexpr double min_argument_of_exp = -708.0;




// ---- one block of BMVP_LANES observations. f: the values of f; s: 2 y - 1 (+1 or -1). Returns the log
//      probabilities min(s f, 0) - log1p(exp(-|f|)); with compute_partials, also writes y - inv_logit(f) to
//      partial. Kernels: nico_exp_1_wo_checks and nico_log1p_1_wo_checks of the BayesMVP kernel files.
#if BMVP_LANES == 8

template <bool compute_partials>
ALWAYS_INLINE __m512d log_probability_and_partial_of_block(const __m512d f, const __m512d s, __m512d& partial) {

    ALIGN64  __m512d const f_aligned = f;
    ALIGN64  __m512d const s_aligned = s;
    ALIGN64  __m512d const zero        = _mm512_setzero_pd();
    ALIGN64  __m512d const one         = _mm512_set1_pd(1.0);
    ALIGN64  __m512d const minus_abs_f = _mm512_or_pd(f_aligned, _mm512_set1_pd(-0.0));
    ALIGN64  __m512d const e           = BMVP_KERNEL_NAME(nico_exp_1_wo_checks)(
                                    _mm512_max_pd(minus_abs_f, _mm512_set1_pd(min_argument_of_exp)));
    ALIGN64  __m512d const log1p_e     = BMVP_KERNEL_NAME(nico_log1p_1_wo_checks)(e);
    ALIGN64  __m512d const ntheta      = _mm512_mul_pd(s_aligned, f_aligned);
    if constexpr (compute_partials) {
        ALIGN64  __m512d const one_over_1p_e = _mm512_div_pd(one, _mm512_add_pd(one, e));
        const __mmask8 ntheta_ge_0  = _mm512_cmp_pd_mask(ntheta, zero, _CMP_GE_OQ);
        ALIGN64  __m512d const inv_logit_of_minus_ntheta = _mm512_mask_blend_pd(ntheta_ge_0, one_over_1p_e,
                                                                       _mm512_mul_pd(e, one_over_1p_e));
        partial = _mm512_mul_pd(s_aligned, inv_logit_of_minus_ntheta);
    }
    return _mm512_sub_pd(_mm512_min_pd(ntheta, zero), log1p_e);

}

// ---- s = 2 y - 1 from 8 int y; bad_y collects the bits of y other than bit 0 (nonzero: some y not in {0, 1})
ALWAYS_INLINE __m512d sign_from_y(const int* y, __m256i& bad_y) {

    ALIGN32  __m256i const y_int = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(y));
    bad_y = _mm256_or_si256(bad_y, _mm256_andnot_si256(_mm256_set1_epi32(1), y_int));
    return _mm512_fmadd_pd(_mm512_cvtepi32_pd(y_int), _mm512_set1_pd(2.0), _mm512_set1_pd(-1.0));

}

using bad_y_bits_t   = __m256i;
using nan_lanes_t    = __mmask8;
inline bad_y_bits_t no_bad_y_bits()                  { return _mm256_setzero_si256(); }
inline bool         any_bad_y(const bad_y_bits_t b)  { return !_mm256_testz_si256(b, b); }
inline nan_lanes_t  no_nan_lanes()                   { return 0; }
ALWAYS_INLINE void  collect_nan_lanes(nan_lanes_t& lanes, const __m512d f) {

    ALIGN64  __m512d const f_aligned = f;
    lanes = static_cast<__mmask8>(lanes | _mm512_cmp_pd_mask(f_aligned, f_aligned, _CMP_UNORD_Q));

}
inline bool         any_nan(const nan_lanes_t lanes) { return lanes != 0; }
inline __m512d      zero_block()                     { return _mm512_setzero_pd(); }
ALWAYS_INLINE __m512d add_blocks(const __m512d a, const __m512d b) {

      ALIGN64  __m512d const a_aligned = a;

      ALIGN64  __m512d const b_aligned = b; return _mm512_add_pd(a_aligned, b_aligned); }
inline double       sum_of_lanes(const __m512d x)    {

      ALIGN64  __m512d const x_aligned = x; return _mm512_reduce_add_pd(x_aligned); }

// ---- a block from the values of its 8 lanes, built in registers (8 scalar stores followed by one vector load
//      would stall: a load cannot be forwarded from several smaller stores)
template <typename V>
ALWAYS_INLINE __m512d block_from_lanes(const V& value_of_lane) {

    return _mm512_set_pd(value_of_lane(7), value_of_lane(6), value_of_lane(5), value_of_lane(4),
                         value_of_lane(3), value_of_lane(2), value_of_lane(1), value_of_lane(0));

}

// ---- the 8 vari pointers of a block of a plain vector of var, copied in one 64-byte move
ALWAYS_INLINE void copy_pointers_of_block(const var* f, vari** operands) {

    _mm512_storeu_si512(static_cast<void*>(operands), _mm512_loadu_si512(static_cast<const void*>(f)));

}

#else

template <bool compute_partials>
ALWAYS_INLINE __m256d log_probability_and_partial_of_block(const __m256d f, const __m256d s, __m256d& partial) {

    ALIGN32  __m256d const f_aligned = f;
    ALIGN32  __m256d const s_aligned = s;
    ALIGN32  __m256d const zero        = _mm256_setzero_pd();
    ALIGN32  __m256d const one         = _mm256_set1_pd(1.0);
    ALIGN32  __m256d const minus_abs_f = _mm256_or_pd(f_aligned, _mm256_set1_pd(-0.0));
    ALIGN32  __m256d const e           = BMVP_KERNEL_NAME(nico_exp_1_wo_checks)(
                                    _mm256_max_pd(minus_abs_f, _mm256_set1_pd(min_argument_of_exp)));
    ALIGN32  __m256d const log1p_e     = BMVP_KERNEL_NAME(nico_log1p_1_wo_checks)(e);
    ALIGN32  __m256d const ntheta      = _mm256_mul_pd(s_aligned, f_aligned);
    if constexpr (compute_partials) {
        ALIGN32  __m256d const one_over_1p_e = _mm256_div_pd(one, _mm256_add_pd(one, e));
        ALIGN32  __m256d const ntheta_ge_0   = _mm256_cmp_pd(ntheta, zero, _CMP_GE_OQ);
        ALIGN32  __m256d const inv_logit_of_minus_ntheta = _mm256_blendv_pd(one_over_1p_e,
                                                                   _mm256_mul_pd(e, one_over_1p_e), ntheta_ge_0);
        partial = _mm256_mul_pd(s_aligned, inv_logit_of_minus_ntheta);
    }
    return _mm256_sub_pd(_mm256_min_pd(ntheta, zero), log1p_e);

}

// ---- s = 2 y - 1 from 4 int y; bad_y collects the bits of y other than bit 0 (nonzero: some y not in {0, 1})
ALWAYS_INLINE __m256d sign_from_y(const int* y, __m128i& bad_y) {

    const __m128i y_int = _mm_loadu_si128(reinterpret_cast<const __m128i*>(y));
    bad_y = _mm_or_si128(bad_y, _mm_andnot_si128(_mm_set1_epi32(1), y_int));
    return _mm256_fmadd_pd(_mm256_cvtepi32_pd(y_int), _mm256_set1_pd(2.0), _mm256_set1_pd(-1.0));

}

using bad_y_bits_t   = __m128i;
using nan_lanes_t    = __m256d;
inline bad_y_bits_t no_bad_y_bits()                  { return _mm_setzero_si128(); }
inline bool         any_bad_y(const bad_y_bits_t b)  { return !_mm_testz_si128(b, b); }
inline nan_lanes_t  no_nan_lanes()                   { return _mm256_setzero_pd(); }
ALWAYS_INLINE void  collect_nan_lanes(nan_lanes_t& lanes, const __m256d f) {

    ALIGN32  __m256d const f_aligned = f;
    lanes = _mm256_or_pd(lanes, _mm256_cmp_pd(f_aligned, f_aligned, _CMP_UNORD_Q));

}
inline bool         any_nan(const nan_lanes_t lanes) { return _mm256_movemask_pd(lanes) != 0; }
inline __m256d      zero_block()                     { return _mm256_setzero_pd(); }
ALWAYS_INLINE __m256d add_blocks(const __m256d a, const __m256d b) {

      ALIGN32  __m256d const a_aligned = a;

      ALIGN32  __m256d const b_aligned = b; return _mm256_add_pd(a_aligned, b_aligned); }
inline double       sum_of_lanes(const __m256d x) {

    ALIGN32  __m256d const x_aligned = x;
    const __m128d pair = _mm_add_pd(_mm256_castpd256_pd128(x_aligned), _mm256_extractf128_pd(x_aligned, 1));
    return _mm_cvtsd_f64(_mm_add_sd(pair, _mm_unpackhi_pd(pair, pair)));

}

// ---- a block from the values of its 4 lanes, built in registers (4 scalar stores followed by one vector load
//      would stall: a load cannot be forwarded from several smaller stores)
template <typename V>
ALWAYS_INLINE __m256d block_from_lanes(const V& value_of_lane) {

    return _mm256_set_pd(value_of_lane(3), value_of_lane(2), value_of_lane(1), value_of_lane(0));

}

// ---- the 4 vari pointers of a block of a plain vector of var, copied in one 32-byte move
ALWAYS_INLINE void copy_pointers_of_block(const var* f, vari** operands) {

    _mm256_storeu_si256(reinterpret_cast<__m256i*>(operands),
                        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(f)));

}

#endif

static_assert(sizeof(var) == sizeof(vari*), "a var is one vari pointer (copy_pointers_of_block)");




// ---- the values of f for one block: read in place when f is a plain vector of double; else built from the
//      values of the lanes (an expression is evaluated once per element; a var's value is read through its
//      vari, whose pointer is kept in operands for the reverse pass)
template <typename T_f>
inline constexpr bool is_plain_vector_of_double = stan::is_plain_type<T_f>::value
                                                  && std::is_arithmetic<stan::value_type_t<T_f>>::value;
template <typename T_f>
inline constexpr bool is_plain_vector_of_var = stan::is_plain_type<T_f>::value
                                               && stan::is_var<stan::value_type_t<T_f>>::value;

template <typename T_f>
ALWAYS_INLINE simd_t values_of_block(const T_f& f, const int first, vari** operands) {

    if constexpr (is_plain_vector_of_double<T_f>) {
        return simd_loadu(f.data() + first);
    } else if constexpr (std::is_arithmetic<stan::value_type_t<T_f>>::value) {
        return block_from_lanes([&f, first](const int lane) {

            return static_cast<double>(f.coeff(first + lane));

        });
    } else if constexpr (is_plain_vector_of_var<T_f>) {
        const var* f_of_block = f.data() + first;
        copy_pointers_of_block(f_of_block, operands + first);
        return block_from_lanes([f_of_block](const int lane) { return f_of_block[lane].vi_->val_; });
    } else {
        return block_from_lanes([&f, first, operands](const int lane) {

            vari* operand = f.coeff(first + lane).vi_;
            operands[first + lane] = operand;
            return operand->val_;

        });
    }

}

// ---- the same for a final partial block of count < BMVP_LANES observations; the spare lanes get f = 0 and
//      y = 0, and are left out of the sum (and of the partials) by the caller
template <typename T_f>
inline simd_t values_of_partial_block(const T_f& f, const int first, const int count, double* block_buffer,
                                      vari** operands) {

    for (int lane = 0; lane < BMVP_LANES; ++lane) block_buffer[lane] = 0.0;
    if constexpr (std::is_arithmetic<stan::value_type_t<T_f>>::value) {
        for (int lane = 0; lane < count; ++lane) block_buffer[lane] = f.coeff(first + lane);
    } else {
        for (int lane = 0; lane < count; ++lane) {
            vari* operand = f.coeff(first + lane).vi_;
            operands[first + lane] = operand;
            block_buffer[lane] = operand->val_;
        }
    }
    return simd_load(block_buffer);

}




// ---- the one pass over the observations: returns sum_i log p_i; with compute_partials, writes
//      y_i - inv_logit(f_i) to partials[i] and the vari pointer of f_i to operands[i]. Throws (with Stan's
//      check functions, before any autodiff node exists) when some y is not in {0, 1} or some f is NaN.
template <bool compute_partials, typename T_y, typename T_f>
inline double sum_of_log_probabilities(const T_y& y, const T_f& f, double* partials, vari** operands) {

    static_assert(std::is_same<stan::value_type_t<T_y>, int>::value, "y must be an array of int");
    const int n = static_cast<int>(f.size());
    const int* y_data = y.data();
    alignas(64) double block_buffer[BMVP_LANES];
    bad_y_bits_t bad_y = no_bad_y_bits();
    nan_lanes_t nan_lanes = no_nan_lanes();
    simd_t sum = zero_block();
    simd_t partial_block = zero_block();
    int first = 0;
    for (; first + BMVP_LANES <= n; first += BMVP_LANES) {
        const simd_t f_block = values_of_block(f, first, operands);
        const simd_t s_block = sign_from_y(y_data + first, bad_y);
        collect_nan_lanes(nan_lanes, f_block);
        sum = add_blocks(sum, log_probability_and_partial_of_block<compute_partials>(f_block, s_block,
                                                                                     partial_block));
        if constexpr (compute_partials) simd_storeu(partials + first, partial_block);
    }
    double sum_of_final_block = 0.0;
    if (first < n) {
        const int count = n - first;
        alignas(64) int y_buffer[BMVP_LANES] = {0};
        for (int lane = 0; lane < count; ++lane) y_buffer[lane] = y_data[first + lane];
        const simd_t f_block = values_of_partial_block(f, first, count, block_buffer, operands);
        const simd_t s_block = sign_from_y(y_buffer, bad_y);
        collect_nan_lanes(nan_lanes, f_block);
        alignas(64) double log_probabilities[BMVP_LANES];
        simd_store(log_probabilities,
                   log_probability_and_partial_of_block<compute_partials>(f_block, s_block, partial_block));
        for (int lane = 0; lane < count; ++lane) sum_of_final_block += log_probabilities[lane];
        if constexpr (compute_partials) {
            simd_store(block_buffer, partial_block);
            for (int lane = 0; lane < count; ++lane) partials[first + lane] = block_buffer[lane];
        }
    }
    if (any_bad_y(bad_y)) stan::math::check_bounded(function_name, "n", y, 0, 1);
    if (any_nan(nan_lanes)) {
        stan::math::check_not_nan(function_name, "Logit transformed probability parameter",
                                  stan::math::value_of(f));
    }
    return sum_of_lanes(sum) + sum_of_final_block;

}




// ---- the autodiff node: the value, the operand varis and the partials (both arrays on the arena)
class vari_fused_bernoulli_logit_lpmf final : public vari {
 public:
    vari** operands_;
    double* partials_;
    int n_;

    vari_fused_bernoulli_logit_lpmf(const double value, vari** operands, double* partials, const int n)
        : vari(value), operands_(operands), partials_(partials), n_(n) {}

    // (four at a time: the four operand pointers and products are loaded first, so that their loads are in
    //  flight at once; the four updates stay in order, since elements of f may share one vari)
    void chain() final {
        const double adjoint = adj_;
        int i = 0;
        for (; i + 4 <= n_; i += 4) {
            vari* operand_0 = operands_[i];
            vari* operand_1 = operands_[i + 1];
            vari* operand_2 = operands_[i + 2];
            vari* operand_3 = operands_[i + 3];
            const double adjoint_0 = adjoint * partials_[i];
            const double adjoint_1 = adjoint * partials_[i + 1];
            const double adjoint_2 = adjoint * partials_[i + 2];
            const double adjoint_3 = adjoint * partials_[i + 3];
            operand_0->adj_ += adjoint_0;
            operand_1->adj_ += adjoint_1;
            operand_2->adj_ += adjoint_2;
            operand_3->adj_ += adjoint_3;
        }
        for (; i < n_; ++i) operands_[i]->adj_ += adjoint * partials_[i];
    }
};

}  // namespace fused_bernoulli_logit
}  // namespace nico::math




// ---- the Stan-facing functions (global scope; stanc calls them as  nico_bernoulli_logit_lpmf<propto__>(y, f,
//      pstream__), with y an array of int and f a vector or vector expression, of double or of var)
namespace nico::math {

template <bool propto, typename T_y, typename T_f,
          stan::require_std_vector_vt<std::is_integral, T_y>* = nullptr,
          stan::require_eigen_col_vector_vt<std::is_arithmetic, T_f>* = nullptr>
inline double nico_bernoulli_logit_lpmf(const T_y& y, const T_f& f, std::ostream* /*pstream__*/) {

    namespace fused = nico::math::fused_bernoulli_logit;
    stan::math::check_consistent_sizes(fused::function_name, "Random variable", y, "Probability parameter", f);
    if (f.size() == 0) return 0.0;
    if constexpr (propto) {
        // (nothing depends on a parameter: Stan checks the arguments, then returns 0)
        stan::math::check_bounded(fused::function_name, "n", y, 0, 1);
        stan::math::check_not_nan(fused::function_name, "Logit transformed probability parameter", f);
        return 0.0;
    }
    return fused::sum_of_log_probabilities<false>(y, f, nullptr, nullptr);

}

}  // namespace nico::math

using nico::math::nico_bernoulli_logit_lpmf;

namespace nico::math {

template <bool propto, typename T_y, typename T_f,
          stan::require_std_vector_vt<std::is_integral, T_y>* = nullptr,
          stan::require_eigen_col_vector_vt<stan::is_var, T_f>* = nullptr>
inline stan::math::var nico_bernoulli_logit_lpmf(const T_y& y, const T_f& f, std::ostream* /*pstream__*/) {

    namespace fused = nico::math::fused_bernoulli_logit;
    using stan::math::vari;
    stan::math::check_consistent_sizes(fused::function_name, "Random variable", y, "Probability parameter", f);
    const int n = static_cast<int>(f.size());
    if (n == 0) return stan::math::var(0.0);
    auto& arena = stan::math::ChainableStack::instance_->memalloc_;
    vari** operands = arena.alloc_array<vari*>(n);
    double* partials = arena.alloc_array<double>(n);
    const double value = fused::sum_of_log_probabilities<true>(y, f, partials, operands);
    return stan::math::var(new fused::vari_fused_bernoulli_logit_lpmf(value, operands, partials, n));

}

}  // namespace nico::math

using nico::math::nico_bernoulli_logit_lpmf;

#undef BMVP_FUSED_BERNOULLI_LOGIT_LPMF_BEING_INCLUDED

























