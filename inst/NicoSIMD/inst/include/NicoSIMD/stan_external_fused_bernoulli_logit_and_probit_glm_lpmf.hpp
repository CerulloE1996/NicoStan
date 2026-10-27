#pragma once
//
// -| --------- BayesMVP fused Bernoulli GLM log probabilities (logit and probit links), Stan external functions
//
//     nico_bernoulli_logit_glm_lpmf(y | x, alpha, beta)    = Stan's bernoulli_logit_glm_lpmf(y | x, alpha, beta)
//                                                           = sum_i log inv_logit(s_i f_i)
//     nico_bernoulli_probit_glm_lpmf(y | x, alpha, beta)   = sum_i log Phi(s_i f_i)   (Stan has no probit GLM:
//                                                             the same arguments, with the probit link)
//
// with f = alpha + x * beta (x: N x K matrix; alpha: real, or vector of length N; beta: vector of length K) and
// s_i = 2 y_i - 1. ONE pass over the rows, in chunks of rows_per_chunk rows: f of the chunk, column by column
// (each column's rows of the chunk read as one contiguous run), then, for each block of BMVP_LANES rows, the
// link's log probabilities and partials d_i = d log p_i / d f_i with the BayesMVP SIMD kernels; then, for each
// column k, the chunk's
// sum_i x_ik d_i, added to the partial of beta_k (x is read from memory once, then from the cache: Stan's
// version reads it twice, for x * beta and for x^T d). Partials: beta: x^T d; alpha: sum_i d_i (real) or d
// (vector); x (when it is a parameter): d_i beta_k, formed in the reverse pass. ONE autodiff node, whose
// operands are the parameters among x, alpha and beta.
//
// Logit link: the kernel of stan_external_fused_bernoulli_logit_lpmf.hpp (one exp, one log1p, one division
// per row; exact partials). Probit link, with z = s f, a = |z| and BayesMVP's Mills-ratio rational of nico_Phi,
// M(a) = N(a) / Q(a) (Phi(-a) = exp(-a^2/2) M(a), relative error <= 1.4e-11 for a <= 38.5; beyond, the
// continued fraction nico_mills_ratio_tail):
//
//     log Phi(z)       = log M(a) - a^2/2              (z < 0: no underflow, relative-accurate lower tail)
//                      = log(1 - exp(-a^2/2) M(a))     (z >= 0)
//     d log Phi(z)/dz  = phi(z) / Phi(z) = (z < 0 ? 1 : exp(-a^2/2)) / (sqrt(2 pi) * the argument of the log)
//
// i.e. one rational, one exp, one log and one division per row.
//
// Checks as in Stan's bernoulli_logit_glm_lpmf, with Stan's check functions (and so the same exceptions):
// consistent sizes, y in {0, 1}; when the sum is not finite or some f is NaN: beta, alpha and s .* f finite.
// With every argument of double and propto = true the result is 0, as in Stan.
//
// Declared in the Stan functions block (no body; stanc --allow-undefined) as
//
//     real nico_bernoulli_logit_glm_lpmf(array[] int y, matrix x, real alpha, vector beta);
//     real nico_bernoulli_logit_glm_lpmf(array[] int y, matrix x, vector alpha, vector beta);
//     real nico_bernoulli_probit_glm_lpmf(array[] int y, matrix x, real alpha, vector beta);
//     real nico_bernoulli_probit_glm_lpmf(array[] int y, matrix x, vector alpha, vector beta);
//
// e.g.  y ~ nico_bernoulli_logit_glm(x, alpha, beta);  in place of  y ~ bernoulli_logit_glm(x, alpha, beta);
// (Stan's other forms, int y or row_vector x, are not provided: a call with them keeps Stan's function.)
//
// Included at the end of stan_external_functions.hpp, after stan_external_fused_bernoulli_logit_lpmf.hpp (whose
// logit kernel and helpers it uses); it includes both itself (#pragma once). Its helpers are in their own
// namespace, nico::math::fused_bernoulli_glm.
//
// -----------------------------------------------------------------------------------------------------------

#include "stan_external_functions.hpp"
#include "stan_external_fused_bernoulli_logit_lpmf.hpp"

#include <stan/math/rev.hpp>
#include <stan/math/prim.hpp>

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include "nico_simd_intrinsics.hpp"   //// <immintrin.h> on x86-64; SIMDe (NEON) on ARM
#include <ostream>
#include <type_traits>
#include <vector>

namespace nico::math {
namespace fused_bernoulli_glm {

using stan::math::var;
using stan::math::vari;
namespace fused_logit = nico::math::fused_bernoulli_logit;




// ---- SIMD helpers of this header (the block width and the masked loads of a final partial block)
#if defined(NICO_SIMD_SCALAR_KERNELS)

inline double set_all_lanes(const double a) {

    return a;

}
ALWAYS_INLINE double multiply_and_add_blocks(const double a,
                                             const double b,
                                             const double c) {

    return std::fma(a, b, c);

}
ALWAYS_INLINE double add_blocks(const double a,
                                const double b) {

    return a + b;

}
ALWAYS_INLINE double load_first_lanes(const double* p,
                                      const int count) {

    return (count > 0) ? p[0] : 0.0;

}
ALWAYS_INLINE double keep_first_lanes(const double x,
                                      const int count) {

    return (count > 0) ? x : 0.0;

}

#elif BMVP_LANES == 8

inline __m512d       set_all_lanes(const double a) { return _mm512_set1_pd(a); }
ALWAYS_INLINE __m512d multiply_and_add_blocks(const __m512d a, const __m512d b, const __m512d c) {

    ALIGN64  __m512d const a_aligned = a;
    ALIGN64  __m512d const b_aligned = b;
    ALIGN64  __m512d const c_aligned = c;
    return _mm512_fmadd_pd(a_aligned, b_aligned, c_aligned);

}
ALWAYS_INLINE __m512d add_blocks(const __m512d a, const __m512d b) {

      ALIGN64  __m512d const a_aligned = a;

      ALIGN64  __m512d const b_aligned = b; return _mm512_add_pd(a_aligned, b_aligned); }

// ---- the first count (< 8) values from p, zero in the other lanes; and a block with lanes count, ... set to 0
ALWAYS_INLINE __m512d load_first_lanes(const double* p, const int count) {

    return _mm512_maskz_loadu_pd(static_cast<__mmask8>((1u << count) - 1u), p);

}
ALWAYS_INLINE __m512d keep_first_lanes(const __m512d x, const int count) {

    ALIGN64  __m512d const x_aligned = x;
    return _mm512_maskz_mov_pd(static_cast<__mmask8>((1u << count) - 1u), x_aligned);

}

// ---- the probit link for one block: returns log Phi(s f); with compute_partials, writes s phi(s f) / Phi(s f)
template <bool compute_partials>
ALWAYS_INLINE __m512d probit_log_probability_and_partial_of_block(const __m512d f, const __m512d s,
                                                                  __m512d& partial) {

    ALIGN64  __m512d const f_aligned = f;
    ALIGN64  __m512d const s_aligned = s;
    ALIGN64  __m512d const zero       = _mm512_setzero_pd();
    ALIGN64  __m512d const one        = _mm512_set1_pd(1.0);
    ALIGN64  __m512d const z          = _mm512_mul_pd(s_aligned, f_aligned);
    ALIGN64  __m512d const a          = _mm512_abs_pd(z);
    ALIGN64  __m512d const a_rational = _mm512_min_pd(a, _mm512_set1_pd(FAST_PHI_MILLS_Z_CAP));
    ALIGN64  __m512d numerator = _mm512_set1_pd(FAST_PHI_MILLS_NUMERATOR_COEFFICIENTS[6]);
    numerator = _mm512_fmadd_pd(numerator, a_rational, _mm512_set1_pd(FAST_PHI_MILLS_NUMERATOR_COEFFICIENTS[5]));
    numerator = _mm512_fmadd_pd(numerator, a_rational, _mm512_set1_pd(FAST_PHI_MILLS_NUMERATOR_COEFFICIENTS[4]));
    numerator = _mm512_fmadd_pd(numerator, a_rational, _mm512_set1_pd(FAST_PHI_MILLS_NUMERATOR_COEFFICIENTS[3]));
    numerator = _mm512_fmadd_pd(numerator, a_rational, _mm512_set1_pd(FAST_PHI_MILLS_NUMERATOR_COEFFICIENTS[2]));
    numerator = _mm512_fmadd_pd(numerator, a_rational, _mm512_set1_pd(FAST_PHI_MILLS_NUMERATOR_COEFFICIENTS[1]));
    numerator = _mm512_fmadd_pd(numerator, a_rational, _mm512_set1_pd(FAST_PHI_MILLS_NUMERATOR_COEFFICIENTS[0]));
    ALIGN64  __m512d denominator = _mm512_set1_pd(FAST_PHI_MILLS_DENOMINATOR_COEFFICIENTS[7]);
    denominator = _mm512_fmadd_pd(denominator, a_rational,
                                  _mm512_set1_pd(FAST_PHI_MILLS_DENOMINATOR_COEFFICIENTS[6]));
    denominator = _mm512_fmadd_pd(denominator, a_rational,
                                  _mm512_set1_pd(FAST_PHI_MILLS_DENOMINATOR_COEFFICIENTS[5]));
    denominator = _mm512_fmadd_pd(denominator, a_rational,
                                  _mm512_set1_pd(FAST_PHI_MILLS_DENOMINATOR_COEFFICIENTS[4]));
    denominator = _mm512_fmadd_pd(denominator, a_rational,
                                  _mm512_set1_pd(FAST_PHI_MILLS_DENOMINATOR_COEFFICIENTS[3]));
    denominator = _mm512_fmadd_pd(denominator, a_rational,
                                  _mm512_set1_pd(FAST_PHI_MILLS_DENOMINATOR_COEFFICIENTS[2]));
    denominator = _mm512_fmadd_pd(denominator, a_rational,
                                  _mm512_set1_pd(FAST_PHI_MILLS_DENOMINATOR_COEFFICIENTS[1]));
    denominator = _mm512_fmadd_pd(denominator, a_rational,
                                  _mm512_set1_pd(FAST_PHI_MILLS_DENOMINATOR_COEFFICIENTS[0]));
    ALIGN64  __m512d M = _mm512_div_pd(numerator, denominator);
    // (beyond the rational's range, a > 38.5: the continued fraction, R(a) / sqrt(2 pi); a rare, cold path)
    const __mmask8 beyond_rational = _mm512_cmp_pd_mask(a, _mm512_set1_pd(FAST_PHI_MILLS_Z_CAP), _CMP_GT_OQ);
    if (__builtin_expect(beyond_rational != 0, 0)) {
        ALIGN64  __m512d const R =
                nico_mills_ratio_tail_AVX512(_mm512_max_pd(a, _mm512_set1_pd(NORMAL_TAIL_THRESH)));
        M = _mm512_mask_blend_pd(beyond_rational, M, _mm512_mul_pd(R, _mm512_set1_pd(INV_SQRT_2PI)));
    }
    ALIGN64  __m512d const half_a_squared = _mm512_mul_pd(_mm512_set1_pd(0.5), _mm512_mul_pd(a, a));
    ALIGN64  __m512d const e = nico_exp_1_wo_checks_AVX512(
                          _mm512_max_pd(_mm512_sub_pd(zero, half_a_squared),
                                        _mm512_set1_pd(fused_logit::min_argument_of_exp)));
    const __mmask8 z_negative = _mm512_cmp_pd_mask(z, zero, _CMP_LT_OQ);
    // (the argument of the log: M(a) where z < 0, 1 - exp(-a^2/2) M(a) = Phi(z) where z >= 0)
    ALIGN64  __m512d const log_argument = _mm512_mask_blend_pd(z_negative, _mm512_fnmadd_pd(e, M, one), M);
    if constexpr (compute_partials) {
        ALIGN64  __m512d const numerator_of_partial = _mm512_mask_blend_pd(z_negative, e, one);
        partial = _mm512_mul_pd(s_aligned, _mm512_div_pd(_mm512_mul_pd(_mm512_set1_pd(INV_SQRT_2PI),
                                                               numerator_of_partial),
                                                 log_argument));
    }
    return _mm512_sub_pd(nico_log_1_double_precision_AVX512(log_argument), _mm512_maskz_mov_pd(z_negative, half_a_squared));

}

#else

inline __m256d       set_all_lanes(const double a) { return _mm256_set1_pd(a); }
ALWAYS_INLINE __m256d multiply_and_add_blocks(const __m256d a, const __m256d b, const __m256d c) {

    ALIGN32  __m256d const a_aligned = a;
    ALIGN32  __m256d const b_aligned = b;
    ALIGN32  __m256d const c_aligned = c;
    return _mm256_fmadd_pd(a_aligned, b_aligned, c_aligned);

}
ALWAYS_INLINE __m256d add_blocks(const __m256d a, const __m256d b) {

      ALIGN32  __m256d const a_aligned = a;

      ALIGN32  __m256d const b_aligned = b; return _mm256_add_pd(a_aligned, b_aligned); }

// ---- the lanes before count (< 4) as a mask, for the masked loads and the zeroing of a final partial block
ALWAYS_INLINE __m256i mask_of_first_lanes(const int count) {

    return _mm256_cmpgt_epi64(_mm256_set1_epi64x(count), _mm256_setr_epi64x(0, 1, 2, 3));

}
ALWAYS_INLINE __m256d load_first_lanes(const double* p, const int count) {

    return _mm256_maskload_pd(p, mask_of_first_lanes(count));

}
ALWAYS_INLINE __m256d keep_first_lanes(const __m256d x, const int count) {

    ALIGN32  __m256d const x_aligned = x;
    return _mm256_and_pd(x_aligned, _mm256_castsi256_pd(mask_of_first_lanes(count)));

}

// ---- the probit link for one block: returns log Phi(s f); with compute_partials, writes s phi(s f) / Phi(s f)
template <bool compute_partials>
ALWAYS_INLINE __m256d probit_log_probability_and_partial_of_block(const __m256d f, const __m256d s,
                                                                  __m256d& partial) {

    ALIGN32  __m256d const f_aligned = f;
    ALIGN32  __m256d const s_aligned = s;
    ALIGN32  __m256d const zero       = _mm256_setzero_pd();
    ALIGN32  __m256d const one        = _mm256_set1_pd(1.0);
    ALIGN32  __m256d const z          = _mm256_mul_pd(s_aligned, f_aligned);
    ALIGN32  __m256d const a          = _mm256_andnot_pd(_mm256_set1_pd(-0.0), z);
    ALIGN32  __m256d const a_rational = _mm256_min_pd(a, _mm256_set1_pd(FAST_PHI_MILLS_Z_CAP));
    ALIGN32  __m256d numerator = _mm256_set1_pd(FAST_PHI_MILLS_NUMERATOR_COEFFICIENTS[6]);
    numerator = _mm256_fmadd_pd(numerator, a_rational, _mm256_set1_pd(FAST_PHI_MILLS_NUMERATOR_COEFFICIENTS[5]));
    numerator = _mm256_fmadd_pd(numerator, a_rational, _mm256_set1_pd(FAST_PHI_MILLS_NUMERATOR_COEFFICIENTS[4]));
    numerator = _mm256_fmadd_pd(numerator, a_rational, _mm256_set1_pd(FAST_PHI_MILLS_NUMERATOR_COEFFICIENTS[3]));
    numerator = _mm256_fmadd_pd(numerator, a_rational, _mm256_set1_pd(FAST_PHI_MILLS_NUMERATOR_COEFFICIENTS[2]));
    numerator = _mm256_fmadd_pd(numerator, a_rational, _mm256_set1_pd(FAST_PHI_MILLS_NUMERATOR_COEFFICIENTS[1]));
    numerator = _mm256_fmadd_pd(numerator, a_rational, _mm256_set1_pd(FAST_PHI_MILLS_NUMERATOR_COEFFICIENTS[0]));
    ALIGN32  __m256d denominator = _mm256_set1_pd(FAST_PHI_MILLS_DENOMINATOR_COEFFICIENTS[7]);
    denominator = _mm256_fmadd_pd(denominator, a_rational,
                                  _mm256_set1_pd(FAST_PHI_MILLS_DENOMINATOR_COEFFICIENTS[6]));
    denominator = _mm256_fmadd_pd(denominator, a_rational,
                                  _mm256_set1_pd(FAST_PHI_MILLS_DENOMINATOR_COEFFICIENTS[5]));
    denominator = _mm256_fmadd_pd(denominator, a_rational,
                                  _mm256_set1_pd(FAST_PHI_MILLS_DENOMINATOR_COEFFICIENTS[4]));
    denominator = _mm256_fmadd_pd(denominator, a_rational,
                                  _mm256_set1_pd(FAST_PHI_MILLS_DENOMINATOR_COEFFICIENTS[3]));
    denominator = _mm256_fmadd_pd(denominator, a_rational,
                                  _mm256_set1_pd(FAST_PHI_MILLS_DENOMINATOR_COEFFICIENTS[2]));
    denominator = _mm256_fmadd_pd(denominator, a_rational,
                                  _mm256_set1_pd(FAST_PHI_MILLS_DENOMINATOR_COEFFICIENTS[1]));
    denominator = _mm256_fmadd_pd(denominator, a_rational,
                                  _mm256_set1_pd(FAST_PHI_MILLS_DENOMINATOR_COEFFICIENTS[0]));
    ALIGN32  __m256d M = _mm256_div_pd(numerator, denominator);
    // (beyond the rational's range, a > 38.5: the continued fraction, R(a) / sqrt(2 pi); a rare, cold path)
    ALIGN32  __m256d const beyond_rational = _mm256_cmp_pd(a, _mm256_set1_pd(FAST_PHI_MILLS_Z_CAP), _CMP_GT_OQ);
    if (__builtin_expect(_mm256_movemask_pd(beyond_rational) != 0, 0)) {
        ALIGN32  __m256d const R =
                nico_mills_ratio_tail_AVX2(_mm256_max_pd(a, _mm256_set1_pd(NORMAL_TAIL_THRESH)));
        M = _mm256_blendv_pd(M, _mm256_mul_pd(R, _mm256_set1_pd(INV_SQRT_2PI)), beyond_rational);
    }
    ALIGN32  __m256d const half_a_squared = _mm256_mul_pd(_mm256_set1_pd(0.5), _mm256_mul_pd(a, a));
    ALIGN32  __m256d const e = nico_exp_1_wo_checks_AVX2(
                          _mm256_max_pd(_mm256_sub_pd(zero, half_a_squared),
                                        _mm256_set1_pd(fused_logit::min_argument_of_exp)));
    ALIGN32  __m256d const z_negative = _mm256_cmp_pd(z, zero, _CMP_LT_OQ);
    // (the argument of the log: M(a) where z < 0, 1 - exp(-a^2/2) M(a) = Phi(z) where z >= 0)
    ALIGN32  __m256d const log_argument = _mm256_blendv_pd(_mm256_fnmadd_pd(e, M, one), M, z_negative);
    if constexpr (compute_partials) {
        ALIGN32  __m256d const numerator_of_partial = _mm256_blendv_pd(e, one, z_negative);
        partial = _mm256_mul_pd(s_aligned, _mm256_div_pd(_mm256_mul_pd(_mm256_set1_pd(INV_SQRT_2PI),
                                                               numerator_of_partial),
                                                 log_argument));
    }
    return _mm256_sub_pd(nico_log_1_double_precision_AVX2(log_argument), _mm256_and_pd(half_a_squared, z_negative));

}

#endif




// ---- the two links: the log probabilities of one block and, with compute_partials, d log p / d f
struct logit_link {
    static constexpr const char* function_name = "nico_bernoulli_logit_glm_lpmf";
    template <bool compute_partials>
    static ALWAYS_INLINE simd_t log_probability_and_partial(const simd_t f, const simd_t s, simd_t& partial) {

        return fused_logit::log_probability_and_partial_of_block<compute_partials>(f, s, partial);

    }
};

#if !defined(NICO_SIMD_SCALAR_KERNELS)
struct probit_link {
    static constexpr const char* function_name = "nico_bernoulli_probit_glm_lpmf";
    template <bool compute_partials>
    static ALWAYS_INLINE simd_t log_probability_and_partial(const simd_t f, const simd_t s, simd_t& partial) {

        return probit_log_probability_and_partial_of_block<compute_partials>(f, s, partial);

    }
};
#endif




// ---- the inputs and outputs of one pass (the values of x, alpha and beta; where the partials go: nullptr when
//      that partial is not needed)
struct glm_pass {
    const int* y = nullptr;
    int N = 0;
    int K = 0;
    const double* x = nullptr;                  // x(i, k) at x[k * x_column_stride + i]
    int x_column_stride = 0;
    const double* beta = nullptr;               // K values
    const double* alpha_vector = nullptr;       // N values, or nullptr when alpha is real
    double alpha_scalar = 0.0;
    double* partials_of_f = nullptr;            // N values of d log p_i / d f_i
    double* partials_of_beta = nullptr;         // K values
    double* partial_of_alpha_scalar = nullptr;  // 1 value: sum_i d log p_i / d f_i
    bool some_y_not_binary = false;
    bool some_f_nan = false;
};

// ---- x(first, ..., first + BMVP_LANES - 1; k) (count < BMVP_LANES: the first count rows, zero in the others)
template <bool partial_block>
ALWAYS_INLINE simd_t x_of_block(const glm_pass& pass, const int first, const int count, const int k) {

    const double* x_of_block_start = pass.x + static_cast<std::ptrdiff_t>(k) * pass.x_column_stride + first;
    return partial_block ? load_first_lanes(x_of_block_start, count) : simd_loadu(x_of_block_start);

}

// ---- f of one block of rows (count < BMVP_LANES: a final partial block, zero in its spare lanes), summed over
//      the columns in four independent sums (one chain of K multiply-adds would wait on each one in turn)
template <bool partial_block>
ALWAYS_INLINE simd_t f_of_block(const glm_pass& pass, const int first, const int count) {

    simd_t f_0 = (pass.alpha_vector != nullptr)
                     ? (partial_block ? load_first_lanes(pass.alpha_vector + first, count)
                                      : simd_loadu(pass.alpha_vector + first))
                     : set_all_lanes(pass.alpha_scalar);
    simd_t f_1 = fused_logit::zero_block();
    simd_t f_2 = fused_logit::zero_block();
    simd_t f_3 = fused_logit::zero_block();
    int k = 0;
    for (; k + 4 <= pass.K; k += 4) {
        f_0 = multiply_and_add_blocks(x_of_block<partial_block>(pass, first, count, k),
                                      set_all_lanes(pass.beta[k]), f_0);
        f_1 = multiply_and_add_blocks(x_of_block<partial_block>(pass, first, count, k + 1),
                                      set_all_lanes(pass.beta[k + 1]), f_1);
        f_2 = multiply_and_add_blocks(x_of_block<partial_block>(pass, first, count, k + 2),
                                      set_all_lanes(pass.beta[k + 2]), f_2);
        f_3 = multiply_and_add_blocks(x_of_block<partial_block>(pass, first, count, k + 3),
                                      set_all_lanes(pass.beta[k + 3]), f_3);
    }
    for (; k < pass.K; ++k) {
        f_0 = multiply_and_add_blocks(x_of_block<partial_block>(pass, first, count, k),
                                      set_all_lanes(pass.beta[k]), f_0);
    }
    return add_blocks(add_blocks(f_0, f_1), add_blocks(f_2, f_3));

}

// ---- the rows of one chunk (a multiple of BMVP_LANES): f, then d log p / d f of the chunk in a buffer of the
//      stack, then the chunk's x^T d, so that the chunk's rows of x are read again from the cache
// static constexpr int rows_per_chunk = 256;
// (512 rows: each column's run of x per chunk is 4 KB, a whole page, for the prefetchers; the chunk's rows of x,
//  read again for x^T d, stay in L2)
static constexpr int rows_per_chunk = 512;
static_assert(rows_per_chunk % BMVP_LANES == 0, "rows_per_chunk must be a multiple of the SIMD lanes");

// ---- f of the rows of one chunk, into a buffer of the stack, column by column: each column's rows of the chunk
//      are read as one contiguous run, four columns at a time (as Eigen's matrix-vector product reads x), rather
//      than one SIMD block of each of the K columns in turn (f_of_block: K far-apart cache lines per block, which
//      the prefetchers do not follow once x no longer fits in the cache); a final partial block has zero x in its
//      spare lanes, as in f_of_block
ALWAYS_INLINE void add_column_times_beta_to_f_of_chunk(const double* x_of_column, const simd_t beta_k,
                                                       const int n_full_blocks, const int count_of_final_block,
                                                       double* f_of_chunk) {

    for (int block = 0; block < n_full_blocks; ++block) {
        const int row = block * BMVP_LANES;
        simd_store(f_of_chunk + row,
                   multiply_and_add_blocks(simd_loadu(x_of_column + row), beta_k, simd_load(f_of_chunk + row)));
    }
    if (count_of_final_block > 0) {
        const int row = n_full_blocks * BMVP_LANES;
        simd_store(f_of_chunk + row,
                   multiply_and_add_blocks(load_first_lanes(x_of_column + row, count_of_final_block), beta_k,
                                           simd_load(f_of_chunk + row)));
    }

}

inline void compute_f_of_chunk_into_buffer(const glm_pass& pass, const int chunk_first, const int chunk_rows,
                                           double* f_of_chunk) {

    const int n_full_blocks = chunk_rows / BMVP_LANES;
    const int count_of_final_block = chunk_rows - n_full_blocks * BMVP_LANES;
    // ---- f = alpha:
    for (int block = 0; block < n_full_blocks; ++block) {
        const int row = block * BMVP_LANES;
        simd_store(f_of_chunk + row, (pass.alpha_vector != nullptr)
                                         ? simd_loadu(pass.alpha_vector + chunk_first + row)
                                         : set_all_lanes(pass.alpha_scalar));
    }
    if (count_of_final_block > 0) {
        const int row = n_full_blocks * BMVP_LANES;
        simd_store(f_of_chunk + row,
                   (pass.alpha_vector != nullptr)
                       ? load_first_lanes(pass.alpha_vector + chunk_first + row, count_of_final_block)
                       : set_all_lanes(pass.alpha_scalar));
    }
    // ---- f += x(chunk rows, k) beta_k, four columns at a time (four multiply-adds per block of f):
    int k = 0;
    for (; k + 4 <= pass.K; k += 4) {
        const double* x_of_column_0 = pass.x + static_cast<std::ptrdiff_t>(k) * pass.x_column_stride
                                      + chunk_first;
        const double* x_of_column_1 = x_of_column_0 + pass.x_column_stride;
        const double* x_of_column_2 = x_of_column_1 + pass.x_column_stride;
        const double* x_of_column_3 = x_of_column_2 + pass.x_column_stride;
        const simd_t beta_0 = set_all_lanes(pass.beta[k]);
        const simd_t beta_1 = set_all_lanes(pass.beta[k + 1]);
        const simd_t beta_2 = set_all_lanes(pass.beta[k + 2]);
        const simd_t beta_3 = set_all_lanes(pass.beta[k + 3]);
        for (int block = 0; block < n_full_blocks; ++block) {
            const int row = block * BMVP_LANES;
            simd_t f = simd_load(f_of_chunk + row);
            f = multiply_and_add_blocks(simd_loadu(x_of_column_0 + row), beta_0, f);
            f = multiply_and_add_blocks(simd_loadu(x_of_column_1 + row), beta_1, f);
            f = multiply_and_add_blocks(simd_loadu(x_of_column_2 + row), beta_2, f);
            f = multiply_and_add_blocks(simd_loadu(x_of_column_3 + row), beta_3, f);
            simd_store(f_of_chunk + row, f);
        }
        if (count_of_final_block > 0) {
            const int row = n_full_blocks * BMVP_LANES;
            const int count = count_of_final_block;
            simd_t f = simd_load(f_of_chunk + row);
            f = multiply_and_add_blocks(load_first_lanes(x_of_column_0 + row, count), beta_0, f);
            f = multiply_and_add_blocks(load_first_lanes(x_of_column_1 + row, count), beta_1, f);
            f = multiply_and_add_blocks(load_first_lanes(x_of_column_2 + row, count), beta_2, f);
            f = multiply_and_add_blocks(load_first_lanes(x_of_column_3 + row, count), beta_3, f);
            simd_store(f_of_chunk + row, f);
        }
    }
    for (; k < pass.K; ++k) {
        add_column_times_beta_to_f_of_chunk(pass.x + static_cast<std::ptrdiff_t>(k) * pass.x_column_stride
                                                + chunk_first,
                                            set_all_lanes(pass.beta[k]), n_full_blocks, count_of_final_block,
                                            f_of_chunk);
    }

}

// ---- d log p / d f of one block (zero in the spare lanes of a final partial block): kept in the chunk's buffer,
//      and where the partials of f or of a real alpha are needed
template <bool partial_block>
ALWAYS_INLINE void keep_partials_of_block(glm_pass& pass, const int first, const int count, const simd_t d,
                                          double* partials_of_f_of_chunk, simd_t& alpha_accumulator) {

    simd_store(partials_of_f_of_chunk, d);
    if (pass.partials_of_f != nullptr) {
        if constexpr (partial_block) {
            for (int lane = 0; lane < count; ++lane) {
                pass.partials_of_f[first + lane] = partials_of_f_of_chunk[lane];
            }
        } else {
            simd_storeu(pass.partials_of_f + first, d);
        }
    }
    if (pass.partial_of_alpha_scalar != nullptr) alpha_accumulator = add_blocks(alpha_accumulator, d);

}

// ---- the chunk's x^T d, added to the partials of beta (four accumulators per column: four independent chains of
//      multiply-adds)
inline void add_partials_of_beta_of_chunk(glm_pass& pass, const int chunk_first, const int chunk_rows,
                                          const double* partials_of_f_of_chunk) {

    const int n_full_blocks = chunk_rows / BMVP_LANES;
    const int count_of_final_block = chunk_rows - n_full_blocks * BMVP_LANES;
    for (int k = 0; k < pass.K; ++k) {
        const double* x_of_column = pass.x + static_cast<std::ptrdiff_t>(k) * pass.x_column_stride + chunk_first;
        simd_t accumulator_0 = fused_logit::zero_block();
        simd_t accumulator_1 = fused_logit::zero_block();
        simd_t accumulator_2 = fused_logit::zero_block();
        simd_t accumulator_3 = fused_logit::zero_block();
        int block = 0;
        for (; block + 4 <= n_full_blocks; block += 4) {
            const int row = block * BMVP_LANES;
            accumulator_0 = multiply_and_add_blocks(simd_loadu(x_of_column + row),
                                                    simd_load(partials_of_f_of_chunk + row), accumulator_0);
            accumulator_1 = multiply_and_add_blocks(simd_loadu(x_of_column + row + BMVP_LANES),
                                                    simd_load(partials_of_f_of_chunk + row + BMVP_LANES),
                                                    accumulator_1);
            accumulator_2 = multiply_and_add_blocks(simd_loadu(x_of_column + row + 2 * BMVP_LANES),
                                                    simd_load(partials_of_f_of_chunk + row + 2 * BMVP_LANES),
                                                    accumulator_2);
            accumulator_3 = multiply_and_add_blocks(simd_loadu(x_of_column + row + 3 * BMVP_LANES),
                                                    simd_load(partials_of_f_of_chunk + row + 3 * BMVP_LANES),
                                                    accumulator_3);
        }
        for (; block < n_full_blocks; ++block) {
            accumulator_0 = multiply_and_add_blocks(simd_loadu(x_of_column + block * BMVP_LANES),
                                                    simd_load(partials_of_f_of_chunk + block * BMVP_LANES),
                                                    accumulator_0);
        }
        if (count_of_final_block > 0) {
            accumulator_0 = multiply_and_add_blocks(load_first_lanes(x_of_column + block * BMVP_LANES,
                                                                     count_of_final_block),
                                                    simd_load(partials_of_f_of_chunk + block * BMVP_LANES),
                                                    accumulator_0);
        }
        const simd_t accumulator = add_blocks(add_blocks(accumulator_0, accumulator_1),
                                              add_blocks(accumulator_2, accumulator_3));
        pass.partials_of_beta[k] += fused_logit::sum_of_lanes(accumulator);
    }

}

// ---- the one pass: returns sum_i log p_i, and fills the partials asked for
template <typename link, bool compute_partials>
inline double sum_of_glm_log_probabilities(glm_pass& pass) {

    fused_logit::bad_y_bits_t bad_y = fused_logit::no_bad_y_bits();
    fused_logit::nan_lanes_t nan_lanes = fused_logit::no_nan_lanes();
    simd_t sum = fused_logit::zero_block();
    simd_t alpha_accumulator = fused_logit::zero_block();
    simd_t d = fused_logit::zero_block();
    alignas(64) double partials_of_f_of_chunk[rows_per_chunk];
    alignas(64) double f_of_chunk[rows_per_chunk];
    if constexpr (compute_partials) {
        if (pass.partials_of_beta != nullptr) {
            for (int k = 0; k < pass.K; ++k) pass.partials_of_beta[k] = 0.0;
        }
    }
    for (int chunk_first = 0; chunk_first < pass.N; chunk_first += rows_per_chunk) {
        const int chunk_end = std::min(chunk_first + rows_per_chunk, pass.N);
        compute_f_of_chunk_into_buffer(pass, chunk_first, chunk_end - chunk_first, f_of_chunk);
        int first = chunk_first;
        for (; first + BMVP_LANES <= chunk_end; first += BMVP_LANES) {
            // const simd_t f = f_of_block<false>(pass, first, BMVP_LANES);
            const simd_t f = simd_load(f_of_chunk + (first - chunk_first));
            const simd_t s = fused_logit::sign_from_y(pass.y + first, bad_y);
            fused_logit::collect_nan_lanes(nan_lanes, f);
            sum = add_blocks(sum, link::template log_probability_and_partial<compute_partials>(f, s, d));
            if constexpr (compute_partials) {
                keep_partials_of_block<false>(pass, first, BMVP_LANES, d,
                                              partials_of_f_of_chunk + (first - chunk_first), alpha_accumulator);
            }
        }
        if (first < chunk_end) {
            // ---- the final partial block (of the last chunk): spare lanes with y = 0 and x = 0, left out of
            //      the sum and of the partials
            const int count = chunk_end - first;
            alignas(64) int y_buffer[BMVP_LANES] = {0};
            for (int lane = 0; lane < count; ++lane) y_buffer[lane] = pass.y[first + lane];
            // const simd_t f = f_of_block<true>(pass, first, count);
            const simd_t f = simd_load(f_of_chunk + (first - chunk_first));
            const simd_t s = fused_logit::sign_from_y(y_buffer, bad_y);
            fused_logit::collect_nan_lanes(nan_lanes, f);
            const simd_t log_probabilities =
                link::template log_probability_and_partial<compute_partials>(f, s, d);
            sum = add_blocks(sum, keep_first_lanes(log_probabilities, count));
            if constexpr (compute_partials) {
                keep_partials_of_block<true>(pass, first, count, keep_first_lanes(d, count),
                                             partials_of_f_of_chunk + (first - chunk_first), alpha_accumulator);
            }
        }
        if constexpr (compute_partials) {
            if (pass.partials_of_beta != nullptr) {
                add_partials_of_beta_of_chunk(pass, chunk_first, chunk_end - chunk_first, partials_of_f_of_chunk);
            }
        }
    }
    if constexpr (compute_partials) {
        if (pass.partial_of_alpha_scalar != nullptr) {
            *pass.partial_of_alpha_scalar = fused_logit::sum_of_lanes(alpha_accumulator);
        }
    }
    pass.some_y_not_binary = fused_logit::any_bad_y(bad_y);
    pass.some_f_nan = fused_logit::any_nan(nan_lanes);
    return fused_logit::sum_of_lanes(sum);

}




// ---- the address of x(0, 0) and the distance between its columns, when each column of x is contiguous in
//      memory (a column-major matrix of double, a Map of one, or a block of its rows), else nullptr
template <typename T_x>
inline const double* column_major_data_or_nullptr(const T_x& x, int& column_stride) {

    using plain_t = std::decay_t<T_x>;
    if constexpr (std::is_arithmetic<stan::value_type_t<T_x>>::value
                  && static_cast<bool>(Eigen::internal::traits<plain_t>::Flags & Eigen::DirectAccessBit)
                  && !static_cast<bool>(Eigen::internal::traits<plain_t>::Flags & Eigen::RowMajorBit)) {

        if (x.innerStride() == 1) {
            column_stride = static_cast<int>(x.outerStride());
            return x.data();
        }

    }
    return nullptr;

}

// ---- the autodiff node: the operands among beta and alpha with their partials, and, when x is a parameter, x's
//      varis (column-major), whose partials d_i beta_k are formed in the reverse pass
class vari_fused_bernoulli_glm_lpmf final : public vari {
 public:
    vari** operands_;
    double* partials_;
    int n_operands_;
    vari** x_operands_;
    const double* partials_of_f_;
    const double* beta_values_;
    int N_;
    int K_;

    vari_fused_bernoulli_glm_lpmf(const double value, vari** operands, double* partials, const int n_operands,
                                  vari** x_operands, const double* partials_of_f, const double* beta_values,
                                  const int N, const int K)
        : vari(value), operands_(operands), partials_(partials), n_operands_(n_operands),
          x_operands_(x_operands), partials_of_f_(partials_of_f), beta_values_(beta_values), N_(N), K_(K) {}

    void chain() final {
        const double adjoint = adj_;
        for (int first = 0; first < n_operands_; first += rows_per_chunk) {
            const int count = std::min(rows_per_chunk, n_operands_ - first);
            for (int i = 0; i < count; ++i) {
                const int j = first + i;
                operands_[j]->adj_ += adjoint * partials_[j];
            }
        }
        if (x_operands_ != nullptr) {
            for (int k = 0; k < K_; ++k) {
                const double adjoint_times_beta_k = adjoint * beta_values_[k];
                vari** x_operands_of_column = x_operands_ + static_cast<std::ptrdiff_t>(k) * N_;
                for (int first = 0; first < N_; first += rows_per_chunk) {
                    const int count = std::min(rows_per_chunk, N_ - first);
                    for (int j = 0; j < count; ++j) {
                        const int i = first + j;
                        x_operands_of_column[i]->adj_ += adjoint_times_beta_k * partials_of_f_[i];
                    }
                }
            }
        }
    }
};




// ---- both GLMs: the checks, the values, the pass and the node
template <bool propto, typename link, typename T_y, typename T_x, typename T_alpha, typename T_beta>
inline stan::return_type_t<T_x, T_alpha, T_beta> fused_bernoulli_glm_lpmf(const T_y& y, const T_x& x,
                                                                          const T_alpha& alpha,
                                                                          const T_beta& beta) {

    static_assert(std::is_same<stan::value_type_t<T_y>, int>::value, "y must be an array of int");
    constexpr bool x_is_var        = stan::is_var<stan::value_type_t<T_x>>::value;
    constexpr bool alpha_is_var    = stan::is_var<stan::scalar_type_t<T_alpha>>::value;
    constexpr bool beta_is_var     = stan::is_var<stan::value_type_t<T_beta>>::value;
    constexpr bool alpha_is_vector = stan::is_eigen<T_alpha>::value;
    constexpr bool some_parameter  = x_is_var || alpha_is_var || beta_is_var;
    const char* function_name = link::function_name;
    const int N = static_cast<int>(x.rows());
    const int K = static_cast<int>(x.cols());
    stan::math::check_consistent_size(function_name, "Vector of dependent variables", y, N);
    stan::math::check_consistent_size(function_name, "Weight vector", beta, K);
    stan::math::check_consistent_size(function_name, "Vector of intercepts", alpha, N);
    if (y.empty()) return 0.0;
    if constexpr (!some_parameter && propto) {
        // (nothing depends on a parameter: Stan checks y, then returns 0)
        stan::math::check_bounded(function_name, "Vector of dependent variables", y, 0, 1);
        return 0.0;
    }
    // ---- the values of x (in place when its columns are contiguous), beta and alpha:
    int x_column_stride = 0;
    const double* x_data = column_major_data_or_nullptr(x, x_column_stride);
    Eigen::Matrix<double, -1, -1> x_values;
    if (x_data == nullptr) {
        x_values = stan::math::value_of(x);
        x_data = x_values.data();
        x_column_stride = N;
    }
    const Eigen::Matrix<double, -1, 1> beta_values = stan::math::value_of(beta);
    Eigen::Matrix<double, -1, 1> alpha_vector_values;
    glm_pass pass;
    pass.y = y.data();
    pass.N = N;
    pass.K = K;
    pass.x = x_data;
    pass.x_column_stride = x_column_stride;
    pass.beta = beta_values.data();
    if constexpr (alpha_is_vector) {
        alpha_vector_values = stan::math::value_of(alpha);
        pass.alpha_vector = alpha_vector_values.data();
    } else {
        pass.alpha_scalar = stan::math::value_of(alpha);
    }
    // ---- the checks of Stan's bernoulli_logit_glm_lpmf after the pass (before any autodiff node exists):
    auto fn_checks_after_the_pass = [&](const double sum) {

        if (pass.some_y_not_binary) {
            stan::math::check_bounded(function_name, "Vector of dependent variables", y, 0, 1);
        }
        if (!std::isfinite(sum) || pass.some_f_nan) {
            stan::math::check_finite(function_name, "Weight vector", beta);
            stan::math::check_finite(function_name, "Intercept", alpha);
            Eigen::Matrix<double, -1, 1> s_times_f(N);
            for (int i = 0; i < N; ++i) {
                double f_i = alpha_is_vector ? pass.alpha_vector[i] : pass.alpha_scalar;
                for (int k = 0; k < K; ++k) {
                    f_i += x_data[static_cast<std::ptrdiff_t>(k) * x_column_stride + i] * beta_values[k];
                }
                s_times_f[i] = (2 * y[i] - 1) * f_i;
            }
            stan::math::check_finite(function_name, "Matrix of independent variables", s_times_f);
        }

    };
    if constexpr (!some_parameter) {
        const double sum = sum_of_glm_log_probabilities<link, false>(pass);
        fn_checks_after_the_pass(sum);
        return sum;
    } else {
        auto& arena = stan::math::ChainableStack::instance_->memalloc_;
        // ---- the partials of the operands among beta (K) and alpha (1 or N), in this order; d log p / d f is
        //      kept for a parameter vector alpha (it is alpha's partials) or a parameter x
        const int n_beta_operands  = beta_is_var ? K : 0;
        const int n_alpha_operands = alpha_is_var ? (alpha_is_vector ? N : 1) : 0;
        const int n_operands = n_beta_operands + n_alpha_operands;
        double* partials = arena.alloc_array<double>(n_operands);
        if (beta_is_var) pass.partials_of_beta = partials;
        if (alpha_is_var && !alpha_is_vector) pass.partial_of_alpha_scalar = partials + n_beta_operands;
        if (alpha_is_var && alpha_is_vector) {
            pass.partials_of_f = partials + n_beta_operands;
        } else if (x_is_var) {
            pass.partials_of_f = arena.alloc_array<double>(N);
        }
        const double sum = sum_of_glm_log_probabilities<link, true>(pass);
        fn_checks_after_the_pass(sum);
        vari** operands = arena.alloc_array<vari*>(n_operands);
        int operand = 0;
        if constexpr (beta_is_var) {
            for (int k = 0; k < K; ++k) operands[operand++] = beta.coeff(k).vi_;
        }
        if constexpr (alpha_is_var) {
            if constexpr (alpha_is_vector) {
                for (int i = 0; i < N; ++i) operands[operand++] = alpha.coeff(i).vi_;
            } else {
                operands[operand++] = alpha.vi_;
            }
        }
        vari** x_operands = nullptr;
        double* beta_values_on_arena = nullptr;
        if constexpr (x_is_var) {
            x_operands = arena.alloc_array<vari*>(static_cast<std::size_t>(N) * K);
            for (int k = 0; k < K; ++k) {
                for (int i = 0; i < N; ++i) {
                    x_operands[static_cast<std::ptrdiff_t>(k) * N + i] = x.coeff(i, k).vi_;
                }
            }
            beta_values_on_arena = arena.alloc_array<double>(K);
            for (int k = 0; k < K; ++k) beta_values_on_arena[k] = beta_values[k];
        }
        return stan::math::var(new vari_fused_bernoulli_glm_lpmf(sum, operands, partials, n_operands, x_operands,
                                                                 pass.partials_of_f, beta_values_on_arena, N, K));
    }

}

}  // namespace fused_bernoulli_glm
}  // namespace nico::math




// ---- the Stan-facing functions (global scope; stanc calls them as
//      nico_bernoulli_logit_glm_lpmf<propto__>(y, x, alpha, beta, pstream__)): y an array of int, x a matrix,
//      alpha a real or a vector, beta a vector; x, alpha and beta of double or of var
namespace nico::math {

template <bool propto, typename T_y, typename T_x, typename T_alpha, typename T_beta,
          stan::require_std_vector_vt<std::is_integral, T_y>* = nullptr,
          stan::require_eigen_matrix_dynamic_t<T_x>* = nullptr,
          stan::require_any_t<stan::is_stan_scalar<T_alpha>, stan::is_eigen_col_vector<T_alpha>>* = nullptr,
          stan::require_eigen_col_vector_t<T_beta>* = nullptr>
inline stan::return_type_t<T_x, T_alpha, T_beta> nico_bernoulli_logit_glm_lpmf(const T_y& y, const T_x& x,
                                                                               const T_alpha& alpha,
                                                                               const T_beta& beta,
                                                                               std::ostream* /*pstream__*/) {

    namespace fused = nico::math::fused_bernoulli_glm;
    return fused::fused_bernoulli_glm_lpmf<propto, fused::logit_link>(y, x, alpha, beta);

}

}  // namespace nico::math

using nico::math::nico_bernoulli_logit_glm_lpmf;

#if !defined(NICO_SIMD_SCALAR_KERNELS)
namespace nico::math {

template <bool propto, typename T_y, typename T_x, typename T_alpha, typename T_beta,
          stan::require_std_vector_vt<std::is_integral, T_y>* = nullptr,
          stan::require_eigen_matrix_dynamic_t<T_x>* = nullptr,
          stan::require_any_t<stan::is_stan_scalar<T_alpha>, stan::is_eigen_col_vector<T_alpha>>* = nullptr,
          stan::require_eigen_col_vector_t<T_beta>* = nullptr>
inline stan::return_type_t<T_x, T_alpha, T_beta> nico_bernoulli_probit_glm_lpmf(const T_y& y, const T_x& x,
                                                                                const T_alpha& alpha,
                                                                                const T_beta& beta,
                                                                                std::ostream* /*pstream__*/) {

    namespace fused = nico::math::fused_bernoulli_glm;
    return fused::fused_bernoulli_glm_lpmf<propto, fused::probit_link>(y, x, alpha, beta);

}

}  // namespace nico::math

using nico::math::nico_bernoulli_probit_glm_lpmf;
#endif
























