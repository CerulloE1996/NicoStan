#pragma once
//
// -| --------- NicoSIMD Gaussian process functions with a choice of kernel, each ONE autodiff node with its
//              gradient computed by hand, exposed to Stan as external functions
//
// (1) nico_gp_normal_marginal_lpdf(y | mu, x, kernel, sigma, length_scale, diagonal): the same value as
//
//         multi_normal_lpdf(y | mu, add_diag(K, diagonal)),
//
//     the marginal log likelihood of a Gaussian process with a normal likelihood (the diagonal term: a noise
//     variance, a nugget or a jitter; one for every point, or one per point).
//
// (2) nico_gp_cholesky_factor_of_covariance_plus_diagonal_times_vector(x, kernel, sigma, length_scale, diagonal,
//     z): the same value as
//
//         cholesky_decompose(add_diag(K, diagonal)) * z,
//
//     the latent values f = L z of a Gaussian process in its non-centred form (z standard normal), for a model
//     with any likelihood of f (e.g. bernoulli_logit for classification, poisson_log for counts).
//
// The covariance K of the points x, by the kernel argument (data int), with d_ij the distance between the points
// x_i and x_j, each coordinate divided by its length scale, d_ij = sqrt(sum_k ((x_ik - x_jk) / l_k)^2) (as
// Stan's gp_exp_quad_cov, gp_exponential_cov, gp_matern32_cov and gp_matern52_cov):
//     kernel = 1, squared exponential:  K_ij = sigma^2 exp(-d^2 / 2);
//     kernel = 2, Matern 1/2:           K_ij = sigma^2 exp(-d);
//     kernel = 3, Matern 3/2:           K_ij = sigma^2 (1 + sqrt(3) d) exp(-sqrt(3) d);
//     kernel = 4, Matern 5/2:           K_ij = sigma^2 (1 + sqrt(5) d + 5 d^2 / 3) exp(-sqrt(5) d);
//     kernel = 5, periodic:             K_ij = sigma^2 exp(-2 sin^2(pi |x_i - x_j| / period) / length_scale^2)
//                                       (Stan's gp_periodic_cov; one length scale; points of one dimension, e.g.
//                                       times, as in more dimensions it is not a covariance in general; the
//                                       overloads with the period argument, a real, data or a parameter).
// The points x: data, array[] real (one dimension) or array[] vector (D dimensions); length_scale: real (the
// same for every dimension) or array[] real of size D (one per dimension, "ARD"); sigma: real; diagonal: real or
// vector of size N; mu: real or vector; y and z: vectors. Every argument but x and kernel may be data or a
// parameter.
//
// What they are general for: any Gaussian process with one of these five stationary kernels (the ones used in
// most applied work, and all of Stan's stationary ones), inputs of any dimension, with or without ARD, any
// mean (mu, or added to f outside the node), homoscedastic or heteroscedastic normal noise in (1), and any
// likelihood of the latent values in (2) (the likelihood is left to Stan or to BayesMVP's fused densities).
// (3), (4): the same two functions for a SUM of these kernels (nico_gp_sum_of_kernels_normal_marginal_lpdf and
// nico_gp_sum_of_kernels_cholesky_factor_of_covariance_plus_diagonal_times_vector; up to 8 terms, each with its
// own kernel, sigma, length scale and period), e.g. a smooth trend plus a seasonal pattern.
// What they are NOT general for: products of kernels, other kernels (rational
// quadratic, dot product), non-stationary kernels, a mean function with its own parameters inside the node,
// sparse or approximate Gaussian processes (inducing points, Hilbert space basis functions: these never form the
// N x N covariance), and in (1) a non-normal likelihood.
//
// Forward pass (doubles only; nothing on the autodiff tape): the lower triangle of add_diag(K, diagonal), written
// into the one N x N matrix of the node one column segment at a time in SIMD (the distances in SIMD, BayesMVP's
// AVX exp kernel, masked stores; AVX-512: 8 lanes, AVX2 including -DBAYESMVP_FORCE_AVX2: 4 lanes); L its lower
// Cholesky factor, in place, in tiles (stan_external_gp_exp_quad_cov_normal_marginal_lpdf.hpp: the same tiles,
// kernels of the tiles, OpenBLAS option and within-chain threads). Then
//   (1) z = L \ (y - mu), alpha = L' \ z, log density = -0.5 z' z - sum(log(diagonal(L))) - 0.5 N log(2 pi) (with
//       propto = true the terms which do not depend on a parameter are dropped, as in Stan);
//   (2) f = L z.
//
// Reverse pass (ONE callback; the O(N^3) work only when sigma, length_scale or diagonal is a parameter). With a
// symmetric weight matrix Omega (below), the derivative of the log density with respect to each covariance
// parameter theta is  sum_{i > j} Omega_ij dK_ij / d theta + 0.5 sum_i Omega_ii dK_ii / d theta,  i.e.
//   d / d sigma           = (2 sum_{i > j} Omega_ij K_ij + sigma^2 sum_i Omega_ii) / sigma,
//   d / d length_scale    = sum_{i > j} Omega_ij g_ij d_ij^2 / l             (one length scale),
//   d / d length_scale_k  = sum_{i > j} Omega_ij g_ij ((x_ik - x_jk) / l_k)^2 / l_k   (one per dimension),
//   d / d diagonal        = 0.5 sum_i Omega_ii   (one diagonal term),   0.5 Omega_ii   (one per point),
// with g_ij the kernel's weight of its derivative (dK_ij / d l_k = g_ij (x_ik - x_jk)^2 / l_k^3): squared
// exponential g = K; Matern 1/2 g = K / d (0 at d = 0); Matern 3/2 g = 3 sigma^2 exp(-sqrt(3) d); Matern 5/2
// g = (5 / 3) sigma^2 (1 + sqrt(5) d) exp(-sqrt(5) d).
//   (1) Omega = W = alpha alpha' - inverse(K) (times the adjoint of the log density);
//       d / d mu = alpha (sum(alpha) for a real mu), d / d y = -alpha.
//   (2) with v the adjoint of f, a = L' v (d / d z = a), M = inverse(L) and Q_ij = a_max(i,j) z_min(i,j):
//       Omega = G = M' Q M  (from d f = d L z, d L = L Phi(M d K M'), Phi the lower triangle with half its
//       diagonal: v' d f = 0.5 sum_ij G_ij dK_ij). R = Q M is made column by column in O(N^2): for column j of M
//       (rows i >= j), R_ij = a_i sum_{j <= k <= i} z_k M_kj + z_i sum_{k > i} a_k M_kj, and R_ij = z_i
//       sum_{k >= j} a_k M_kj for i < j, with the two running sums as SIMD prefix and suffix scans (in-register
//       scans of each packet and a carry between packets).
// Omega is made tile by tile of its lower triangle (64 x 64 tiles, in each core's L2), never as an N x N matrix:
//   (1) the tile of inverse(K) = M' M (as stan_external_gp_exp_quad_cov_normal_marginal_lpdf.hpp), then
//       W = alpha_i alpha_j - inverse(K)_ij in SIMD;
//   (2) the tile of G = M_II' R_IJ + M(after I, I)' R(after I, J);
// then, in the same pass over each column segment of the tile, in SIMD: the distances, K_ij and g_ij recomputed,
// and the sums above (masked loads for the last partial packet). The sums of the tiles are added in the order of
// the tiles, so the result is the same whatever the number of threads.
//
// Checks, as Stan: sigma positive, the length scales positive and finite, no coordinate of a point NaN, the
// kernel 1 to 4, the sizes (y, mu, z, diagonal: N; length_scale: 1 or D), y not NaN, mu finite, the diagonal not
// NaN, and cholesky_decompose's (positive definite, Stan's message).
//
// Declared in the Stan functions block (no body), one declaration per shape of the arguments used, e.g.
//
//     real nico_gp_normal_marginal_lpdf(vector y, real mu, data array[] real x, data int kernel, real sigma,
//                                       real length_scale, real diagonal);
//     real nico_gp_normal_marginal_lpdf(vector y, vector mu, data array[] vector x, data int kernel, real sigma,
//                                       array[] real length_scale, vector diagonal);
//     vector nico_gp_cholesky_factor_of_covariance_plus_diagonal_times_vector(data array[] real x,
//                                                                            data int kernel, real sigma,
//                                                                            real length_scale,
//                                                                            real diagonal, vector z);
//
// and called as  y ~ nico_gp_normal_marginal(mu, x, kernel, sigma, length_scale, diagonal);  and
// vector[N] f = mu + nico_gp_cholesky_factor_of_covariance_plus_diagonal_times_vector(x, kernel, sigma,
//                                                                                     length_scale, 1e-8, z);
//
// ---------------------------------------------------------------------------------------------------------------

// ---- the SIMD kernels and the tiles: NicoSIMD's own, unless BayesMVP's header of the same name has included
//      BayesMVP's (the same internal C++, namespace nico::math) and set
//      NICOSIMD_GP_WITH_A_CHOICE_OF_KERNEL_USES_THE_KERNELS_AND_TILES_ALREADY_INCLUDED, so that one translation
//      unit never holds both copies; BayesMVP's fast_ names are thin wrappers of the nico_ functions here
#ifndef NICOSIMD_GP_WITH_A_CHOICE_OF_KERNEL_USES_THE_KERNELS_AND_TILES_ALREADY_INCLUDED
#  include "stan_external_functions.hpp"
#  include "stan_external_gp_exp_quad_cov_normal_marginal_lpdf.hpp"
#endif

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <ostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#include "nico_simd_exp_in_double_precision.hpp"

#ifndef BMVP_LANES
#  error "the GP functions with a choice of kernel need the kernels of stan_external_functions.hpp"
#endif

// ---- the alignment of every SIMD variable (the macros: ALIGN64 for AVX-512, ALIGN32 for AVX2)
#if BMVP_LANES == 8
#  define BMVP_ALIGN_OF_A_SIMD_VARIABLE ALIGN64
#else
#  define BMVP_ALIGN_OF_A_SIMD_VARIABLE ALIGN32
#endif

namespace nico::math {
namespace gp_with_a_choice_of_kernel {

namespace tiles = nico::math::gp_exp_quad_cov_normal_marginal;

static constexpr const char* name_of_the_marginal_lpdf = "nico_gp_normal_marginal_lpdf";
static constexpr const char* name_of_the_latent_values
    = "nico_gp_cholesky_factor_of_covariance_plus_diagonal_times_vector";

static constexpr int squared_exponential_kernel = 1;
static constexpr int Matern_1_2_kernel          = 2;
static constexpr int Matern_3_2_kernel          = 3;
static constexpr int Matern_5_2_kernel          = 4;
static constexpr int periodic_kernel            = 5;

static constexpr int lanes             = BMVP_LANES;
static constexpr int size_of_the_tiles = tiles::size_of_the_tiles;

static constexpr double square_root_of_3 = 1.7320508075688772935274463415059;
static constexpr double square_root_of_5 = 2.2360679774997896964091736687313;
static constexpr double two_pi           = 6.2831853071795864769252867665590;


// ---- the SIMD operations (AVX-512: 8 lanes; AVX2: 4 lanes)
#if BMVP_LANES == 8
inline simd_t simd_all_lanes(const double value) { return _mm512_set1_pd(value); }
inline simd_t simd_zero() { return _mm512_setzero_pd(); }
inline simd_t simd_add(const simd_t a, const simd_t b) { return _mm512_add_pd(a, b); }
inline simd_t simd_subtract(const simd_t a, const simd_t b) { return _mm512_sub_pd(a, b); }
inline simd_t simd_multiply(const simd_t a, const simd_t b) { return _mm512_mul_pd(a, b); }
//  (a b + c, and a b - c)
inline simd_t simd_multiply_add(const simd_t a, const simd_t b, const simd_t c) {

    return _mm512_fmadd_pd(a, b, c);

}
inline simd_t simd_multiply_subtract(const simd_t a, const simd_t b, const simd_t c) {

    return _mm512_fmsub_pd(a, b, c);

}
inline simd_t simd_square_root(const simd_t a) { return _mm512_sqrt_pd(a); }
inline double simd_sum_of_the_lanes(const simd_t a) { return _mm512_reduce_add_pd(a); }

inline __mmask8 mask_of_the_first_lanes(const int count) {

    return count >= 8 ? static_cast<__mmask8>(0xFF) : static_cast<__mmask8>((1u << count) - 1u);

}
//  (the first count lanes from p, 0 in the others; nothing read beyond them)
inline simd_t simd_load_first_lanes(const double* p, const int count) {

    return _mm512_maskz_loadu_pd(mask_of_the_first_lanes(count), p);

}
//  (the first count lanes into p; nothing written beyond them)
inline void simd_store_first_lanes(double* p, const simd_t a, const int count) {

    _mm512_mask_storeu_pd(p, mask_of_the_first_lanes(count), a);

}
//  (numerator / denominator where the denominator is positive, 0 elsewhere)
inline simd_t simd_divide_where_positive(const simd_t numerator, const simd_t denominator) {

    const __mmask8 positive = _mm512_cmp_pd_mask(denominator, _mm512_setzero_pd(), _CMP_GT_OQ);
    return _mm512_maskz_div_pd(positive, numerator, denominator);

}
//  (lane l: a_0 + ... + a_l)
inline simd_t simd_inclusive_prefix_sums(const simd_t a) {

    ALIGN64 __m512d sums = a;
    sums = _mm512_add_pd(sums, _mm512_maskz_permutexvar_pd(0xFE, _mm512_set_epi64(6, 5, 4, 3, 2, 1, 0, 0), sums));
    sums = _mm512_add_pd(sums, _mm512_maskz_permutexvar_pd(0xFC, _mm512_set_epi64(5, 4, 3, 2, 1, 0, 0, 0), sums));
    sums = _mm512_add_pd(sums, _mm512_maskz_permutexvar_pd(0xF0, _mm512_set_epi64(3, 2, 1, 0, 0, 0, 0, 0), sums));
    return sums;

}
//  (lane l: a_l + ... + a_7)
inline simd_t simd_inclusive_suffix_sums(const simd_t a) {

    ALIGN64 __m512d sums = a;
    sums = _mm512_add_pd(sums, _mm512_maskz_permutexvar_pd(0x7F, _mm512_set_epi64(7, 7, 6, 5, 4, 3, 2, 1), sums));
    sums = _mm512_add_pd(sums, _mm512_maskz_permutexvar_pd(0x3F, _mm512_set_epi64(7, 7, 7, 6, 5, 4, 3, 2), sums));
    sums = _mm512_add_pd(sums, _mm512_maskz_permutexvar_pd(0x0F, _mm512_set_epi64(7, 7, 7, 7, 7, 6, 5, 4), sums));
    return sums;

}
//  (lane l: a_(l + 1); the last lane 0)
inline simd_t simd_shift_down_by_one_lane(const simd_t a) {

    return _mm512_maskz_permutexvar_pd(0x7F, _mm512_set_epi64(7, 7, 6, 5, 4, 3, 2, 1), a);

}
inline simd_t simd_first_lane_in_every_lane(const simd_t a) {

    return _mm512_permutexvar_pd(_mm512_set1_epi64(0), a);

}
inline simd_t simd_last_lane_in_every_lane(const simd_t a) {

    return _mm512_permutexvar_pd(_mm512_set1_epi64(7), a);

}
inline simd_t simd_round_to_nearest_integer(const simd_t a) {

    return _mm512_roundscale_pd(a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

}
inline simd_t simd_floor(const simd_t a) {

    return _mm512_roundscale_pd(a, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);

}
//  (the sine and cosine of the reduced argument swapped in the lanes where the quadrant is odd, and negated where
//   it says so: sine in quadrants 2 and 3, cosine in quadrants 1 and 2)
inline void simd_place_by_quadrant(const simd_t quadrant, const simd_t sine_of_reduced,
                                   const simd_t cosine_of_reduced, simd_t& sine, simd_t& cosine) {

    const __mmask8 quadrant_1 = _mm512_cmp_pd_mask(quadrant, _mm512_set1_pd(1.0), _CMP_EQ_OQ);
    const __mmask8 quadrant_2 = _mm512_cmp_pd_mask(quadrant, _mm512_set1_pd(2.0), _CMP_EQ_OQ);
    const __mmask8 quadrant_3 = _mm512_cmp_pd_mask(quadrant, _mm512_set1_pd(3.0), _CMP_EQ_OQ);
    const __mmask8 odd = quadrant_1 | quadrant_3;
    sine = _mm512_mask_blend_pd(odd, sine_of_reduced, cosine_of_reduced);
    cosine = _mm512_mask_blend_pd(odd, cosine_of_reduced, sine_of_reduced);
    sine = _mm512_mask_sub_pd(sine, quadrant_2 | quadrant_3, _mm512_setzero_pd(), sine);
    cosine = _mm512_mask_sub_pd(cosine, quadrant_1 | quadrant_2, _mm512_setzero_pd(), cosine);

}
#else
inline simd_t simd_all_lanes(const double value) { return _mm256_set1_pd(value); }
inline simd_t simd_zero() { return _mm256_setzero_pd(); }
inline simd_t simd_add(const simd_t a, const simd_t b) { return _mm256_add_pd(a, b); }
inline simd_t simd_subtract(const simd_t a, const simd_t b) { return _mm256_sub_pd(a, b); }
inline simd_t simd_multiply(const simd_t a, const simd_t b) { return _mm256_mul_pd(a, b); }
//  (a b + c, and a b - c)
inline simd_t simd_multiply_add(const simd_t a, const simd_t b, const simd_t c) {

    return _mm256_fmadd_pd(a, b, c);

}
inline simd_t simd_multiply_subtract(const simd_t a, const simd_t b, const simd_t c) {

    return _mm256_fmsub_pd(a, b, c);

}
inline simd_t simd_square_root(const simd_t a) { return _mm256_sqrt_pd(a); }
inline double simd_sum_of_the_lanes(const simd_t a) {

    ALIGN32 double values_of_the_lanes[4];
    _mm256_store_pd(values_of_the_lanes, a);
    return (values_of_the_lanes[0] + values_of_the_lanes[1]) + (values_of_the_lanes[2] + values_of_the_lanes[3]);

}

inline __m256i mask_of_the_first_lanes(const int count) {

    ALIGN32 static const long long lane_masks[8] = {-1, -1, -1, -1, 0, 0, 0, 0};
    return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(lane_masks + (count >= 4 ? 0 : 4 - count)));

}
//  (the first count lanes from p, 0 in the others; nothing read beyond them)
inline simd_t simd_load_first_lanes(const double* p, const int count) {

    return _mm256_maskload_pd(p, mask_of_the_first_lanes(count));

}
//  (the first count lanes into p; nothing written beyond them)
inline void simd_store_first_lanes(double* p, const simd_t a, const int count) {

    _mm256_maskstore_pd(p, mask_of_the_first_lanes(count), a);

}
//  (numerator / denominator where the denominator is positive, 0 elsewhere)
inline simd_t simd_divide_where_positive(const simd_t numerator, const simd_t denominator) {

    ALIGN32 __m256d const positive = _mm256_cmp_pd(denominator, _mm256_setzero_pd(), _CMP_GT_OQ);
    return _mm256_and_pd(_mm256_div_pd(numerator, denominator), positive);

}
//  (lane l: a_0 + ... + a_l)
inline simd_t simd_inclusive_prefix_sums(const simd_t a) {

    ALIGN32 __m256d sums = a;
    sums = _mm256_add_pd(sums, _mm256_blend_pd(_mm256_permute4x64_pd(sums, _MM_SHUFFLE(2, 1, 0, 0)),
                                               _mm256_setzero_pd(), 0x1));
    sums = _mm256_add_pd(sums, _mm256_permute2f128_pd(sums, sums, 0x08));
    return sums;

}
//  (lane l: a_l + ... + a_3)
inline simd_t simd_inclusive_suffix_sums(const simd_t a) {

    ALIGN32 __m256d sums = a;
    sums = _mm256_add_pd(sums, _mm256_blend_pd(_mm256_permute4x64_pd(sums, _MM_SHUFFLE(3, 3, 2, 1)),
                                               _mm256_setzero_pd(), 0x8));
    sums = _mm256_add_pd(sums, _mm256_permute2f128_pd(sums, sums, 0x81));
    return sums;

}
//  (lane l: a_(l + 1); the last lane 0)
inline simd_t simd_shift_down_by_one_lane(const simd_t a) {

    return _mm256_blend_pd(_mm256_permute4x64_pd(a, _MM_SHUFFLE(3, 3, 2, 1)), _mm256_setzero_pd(), 0x8);

}
inline simd_t simd_first_lane_in_every_lane(const simd_t a) { return _mm256_permute4x64_pd(a, 0x00); }
inline simd_t simd_last_lane_in_every_lane(const simd_t a) { return _mm256_permute4x64_pd(a, 0xFF); }
inline simd_t simd_round_to_nearest_integer(const simd_t a) {

    return _mm256_round_pd(a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

}
inline simd_t simd_floor(const simd_t a) { return _mm256_floor_pd(a); }
//  (the sine and cosine of the reduced argument swapped in the lanes where the quadrant is odd, and negated where
//   it says so: sine in quadrants 2 and 3, cosine in quadrants 1 and 2)
inline void simd_place_by_quadrant(const simd_t quadrant, const simd_t sine_of_reduced,
                                   const simd_t cosine_of_reduced, simd_t& sine, simd_t& cosine) {

    ALIGN32 __m256d const quadrant_1 = _mm256_cmp_pd(quadrant, _mm256_set1_pd(1.0), _CMP_EQ_OQ);
    ALIGN32 __m256d const quadrant_2 = _mm256_cmp_pd(quadrant, _mm256_set1_pd(2.0), _CMP_EQ_OQ);
    ALIGN32 __m256d const quadrant_3 = _mm256_cmp_pd(quadrant, _mm256_set1_pd(3.0), _CMP_EQ_OQ);
    ALIGN32 __m256d const odd = _mm256_or_pd(quadrant_1, quadrant_3);
    ALIGN32 __m256d const sign_bit = _mm256_set1_pd(-0.0);
    sine = _mm256_blendv_pd(sine_of_reduced, cosine_of_reduced, odd);
    cosine = _mm256_blendv_pd(cosine_of_reduced, sine_of_reduced, odd);
    sine = _mm256_xor_pd(sine, _mm256_and_pd(_mm256_or_pd(quadrant_2, quadrant_3), sign_bit));
    cosine = _mm256_xor_pd(cosine, _mm256_and_pd(_mm256_or_pd(quadrant_1, quadrant_2), sign_bit));

}
#endif

// inline simd_t simd_exp(const simd_t a) { return BMVP_KERNEL_NAME(nico_exp_1)(a); }
// ---- exp of a packet in double precision (relative error about 2e-16): the covariance K + diagonal of a GP has
//      condition numbers of 1e9 to 1e10, so the relative error of nico_exp_1 (about 2e-14: the constant term of
//      its polynomial is 1.00000000000001332) grew into differences of the log density and its gradient of up to
//      1e-5 against Stan math, about 300 times the 1e-9 to 5e-8 between two correct double precision Cholesky
//      factors of the same K. The same reduction as nico_exp_1 (exp(a) = 2^k e^f, k the nearest integer to
//      a log2(e), f = a - k log(2) in two parts, |f| <= log(2) / 2), then e^f by its Taylor series to f^13 / 13!
//      (the first term left out, f^14 / 14!, is below 4e-18 of the value), and nico_exp_1's own handling of the
//      lanes with |a| >= 708.4
inline simd_t simd_exp(const simd_t a) {

    return nico_simd_exp_in_double_precision(a);

}

// ---- the sine and cosine of a packet (general; for the periodic kernel): v = q pi / 2 + r with q the nearest
//      integer to 2 v / pi and r = v - q pi / 2 in [-pi / 4, pi / 4] (pi / 2 in three parts, Cody and Waite, so
//      that r keeps full precision for |v| up to about 1e6); sin(r) and cos(r) by the minimax polynomials of
//      Cephes on [-pi / 4, pi / 4] (sin(r) = r + r z P(z), cos(r) = 1 - z / 2 + z^2 Q(z), z = r^2; relative error
//      about 1e-16); then the quadrant q mod 4: (sin(v), cos(v)) = (sin r, cos r), (cos r, -sin r),
//      (-sin r, -cos r), (-cos r, sin r)
ALWAYS_INLINE void simd_sine_and_cosine(const simd_t v, simd_t& sine, simd_t& cosine) {

    BMVP_ALIGN_OF_A_SIMD_VARIABLE simd_t const q
        = simd_round_to_nearest_integer(simd_multiply(v, simd_all_lanes(0.63661977236758134307553505349006)));
    BMVP_ALIGN_OF_A_SIMD_VARIABLE simd_t r
        = simd_subtract(v, simd_multiply(q, simd_all_lanes(1.57079625129699707031)));
    r = simd_subtract(r, simd_multiply(q, simd_all_lanes(7.54978941586159635336e-8)));
    r = simd_subtract(r, simd_multiply(q, simd_all_lanes(5.39030285815811905290e-15)));
    BMVP_ALIGN_OF_A_SIMD_VARIABLE simd_t const z = simd_multiply(r, r);
    BMVP_ALIGN_OF_A_SIMD_VARIABLE simd_t P = simd_all_lanes(1.58962301576546568060e-10);
    P = simd_multiply_add(P, z, simd_all_lanes(-2.50507477628578072866e-8));
    P = simd_multiply_add(P, z, simd_all_lanes(2.75573136213857245213e-6));
    P = simd_multiply_add(P, z, simd_all_lanes(-1.98412698295895385996e-4));
    P = simd_multiply_add(P, z, simd_all_lanes(8.33333333332211858878e-3));
    P = simd_multiply_add(P, z, simd_all_lanes(-1.66666666666666307295e-1));
    BMVP_ALIGN_OF_A_SIMD_VARIABLE simd_t Q = simd_all_lanes(-1.13585365213876817300e-11);
    Q = simd_multiply_add(Q, z, simd_all_lanes(2.08757008419747316778e-9));
    Q = simd_multiply_add(Q, z, simd_all_lanes(-2.75573141792967388112e-7));
    Q = simd_multiply_add(Q, z, simd_all_lanes(2.48015872888517045348e-5));
    Q = simd_multiply_add(Q, z, simd_all_lanes(-1.38888888888730564116e-3));
    Q = simd_multiply_add(Q, z, simd_all_lanes(4.16666666666665929218e-2));
    BMVP_ALIGN_OF_A_SIMD_VARIABLE simd_t const sine_of_reduced = simd_multiply_add(simd_multiply(r, z), P, r);
    BMVP_ALIGN_OF_A_SIMD_VARIABLE simd_t const cosine_of_reduced
        = simd_multiply_add(simd_multiply(z, z), Q,
                            simd_multiply_add(simd_all_lanes(-0.5), z, simd_all_lanes(1.0)));
    BMVP_ALIGN_OF_A_SIMD_VARIABLE simd_t const quadrant
        = simd_subtract(q, simd_multiply(simd_all_lanes(4.0),
                                         simd_floor(simd_multiply(q, simd_all_lanes(0.25)))));
    simd_place_by_quadrant(quadrant, sine_of_reduced, cosine_of_reduced, sine, cosine);

}

// ---- the constants of the periodic kernel (Stan's gp_periodic_cov: K_ij = sigma^2 exp(-2 sin^2(pi r_ij / p)
//      / l^2), r_ij the distance between the points, one length scale l, the period p); unused by the other
//      kernels
struct constants_of_the_periodic_kernel {
    double length_scale = 1.0;
    double period = 1.0;
};


// ---- the kernel, as a compile-time constant (one copy of each loop per kernel; no branch inside the loops)
template <typename Task>
inline void with_the_kernel(const int kernel, const Task& task) {

    switch (kernel) {
        case squared_exponential_kernel: task(std::integral_constant<int, squared_exponential_kernel>()); break;
        case Matern_1_2_kernel:          task(std::integral_constant<int, Matern_1_2_kernel>());          break;
        case Matern_3_2_kernel:          task(std::integral_constant<int, Matern_3_2_kernel>());          break;
        case Matern_5_2_kernel:          task(std::integral_constant<int, Matern_5_2_kernel>());          break;
        case periodic_kernel:            task(std::integral_constant<int, periodic_kernel>());            break;
        default: break;
    }

}

// ---- one packet of K_ij and of its derivative weight g_ij (dK_ij / d l_k = g_ij (x_ik - x_jk)^2 / l_k^3), from
//      the squared distances of the packet (each coordinate divided by its length scale); for the periodic kernel
//      (one length scale l, period p, r = l d the distance): derivative_weight = K 2 (1 - cos(2 pi r / p)) / l^2,
//      i.e. dK / dl times l, and period_term = K sin(2 pi r / p) r, i.e. dK / dp times l^2 p^2 / (2 pi)
template <int kernel, bool derivative_weight_wanted>
ALWAYS_INLINE void kernel_value_and_derivative_weight_of_a_packet(
        const simd_t squared_distance, const simd_t squared_sigma,
        const constants_of_the_periodic_kernel& periodic, simd_t& kernel_value, simd_t& derivative_weight,
        simd_t& period_term) {

    if constexpr (kernel == squared_exponential_kernel) {
        kernel_value = simd_multiply(squared_sigma,
                                     simd_exp(simd_multiply(simd_all_lanes(-0.5), squared_distance)));
        if constexpr (derivative_weight_wanted) derivative_weight = kernel_value;
    } else if constexpr (kernel == Matern_1_2_kernel) {
        BMVP_ALIGN_OF_A_SIMD_VARIABLE simd_t const distance = simd_square_root(squared_distance);
        kernel_value = simd_multiply(squared_sigma, simd_exp(simd_subtract(simd_zero(), distance)));
        if constexpr (derivative_weight_wanted) {
            derivative_weight = simd_divide_where_positive(kernel_value, distance);
        }
    } else if constexpr (kernel == Matern_3_2_kernel) {
        BMVP_ALIGN_OF_A_SIMD_VARIABLE simd_t const scaled_distance
            = simd_multiply(simd_all_lanes(square_root_of_3), simd_square_root(squared_distance));
        BMVP_ALIGN_OF_A_SIMD_VARIABLE simd_t const squared_sigma_times_exp
            = simd_multiply(squared_sigma, simd_exp(simd_subtract(simd_zero(), scaled_distance)));
        kernel_value = simd_multiply(squared_sigma_times_exp, simd_add(simd_all_lanes(1.0), scaled_distance));
        if constexpr (derivative_weight_wanted) {
            derivative_weight = simd_multiply(simd_all_lanes(3.0), squared_sigma_times_exp);
        }
    } else if constexpr (kernel == periodic_kernel) {
        BMVP_ALIGN_OF_A_SIMD_VARIABLE simd_t const distance
            = simd_multiply(simd_all_lanes(periodic.length_scale), simd_square_root(squared_distance));
        BMVP_ALIGN_OF_A_SIMD_VARIABLE simd_t const inverse_squared_length_scale
            = simd_all_lanes(1.0 / (periodic.length_scale * periodic.length_scale));
        BMVP_ALIGN_OF_A_SIMD_VARIABLE simd_t sine;
        BMVP_ALIGN_OF_A_SIMD_VARIABLE simd_t cosine;
        simd_sine_and_cosine(simd_multiply(simd_all_lanes(two_pi / periodic.period), distance), sine, cosine);
        BMVP_ALIGN_OF_A_SIMD_VARIABLE simd_t const one_minus_cosine = simd_subtract(simd_all_lanes(1.0), cosine);
        kernel_value = simd_multiply(squared_sigma,
                                     simd_exp(simd_multiply(simd_subtract(simd_zero(), one_minus_cosine),
                                                            inverse_squared_length_scale)));
        if constexpr (derivative_weight_wanted) {
            derivative_weight = simd_multiply(simd_multiply(simd_all_lanes(2.0), kernel_value),
                                              simd_multiply(one_minus_cosine, inverse_squared_length_scale));
            period_term = simd_multiply(simd_multiply(kernel_value, sine), distance);
        }
    } else {
        BMVP_ALIGN_OF_A_SIMD_VARIABLE simd_t const scaled_distance
            = simd_multiply(simd_all_lanes(square_root_of_5), simd_square_root(squared_distance));
        BMVP_ALIGN_OF_A_SIMD_VARIABLE simd_t const squared_sigma_times_exp
            = simd_multiply(squared_sigma, simd_exp(simd_subtract(simd_zero(), scaled_distance)));
        BMVP_ALIGN_OF_A_SIMD_VARIABLE simd_t const one_plus_scaled_distance
            = simd_add(simd_all_lanes(1.0), scaled_distance);
        kernel_value = simd_multiply(squared_sigma_times_exp,
                                     simd_multiply_add(simd_all_lanes(5.0 / 3.0), squared_distance,
                                                       one_plus_scaled_distance));
        if constexpr (derivative_weight_wanted) {
            derivative_weight = simd_multiply(simd_all_lanes(5.0 / 3.0),
                                              simd_multiply(squared_sigma_times_exp, one_plus_scaled_distance));
        }
    }

}


// ---- the points, each coordinate divided by the length scale of its dimension, dimension by dimension (column
//      k at values + k stride); each dimension padded with zeros to stride, so that a packet of rows starting
//      before n_points never reads beyond the storage
struct points_divided_by_their_length_scales {
    int     n_points     = 0;
    int     n_dimensions = 0;
    int     stride       = 0;
    double* values       = nullptr;
};

inline int stride_of_each_dimension(const int n_points) {

    return ((n_points + lanes - 1) / lanes) * lanes + lanes;

}

//  (from the coordinates dimension by dimension, x_by_dimension[k n + i], and the n_dimensions length scales;
//   storage of n_dimensions stride doubles)
inline points_divided_by_their_length_scales divide_the_points_by_their_length_scales(
        const std::vector<double>& x_by_dimension, const int n_points, const int n_dimensions,
        const std::vector<double>& length_scales, double* storage) {

    points_divided_by_their_length_scales points;
    points.n_points     = n_points;
    points.n_dimensions = n_dimensions;
    points.stride       = stride_of_each_dimension(n_points);
    points.values       = storage;
    for (int k = 0; k < n_dimensions; ++k) {
        const double* coordinates = x_by_dimension.data() + static_cast<std::ptrdiff_t>(k) * n_points;
        double* divided = storage + static_cast<std::ptrdiff_t>(k) * points.stride;
        BMVP_ALIGN_OF_A_SIMD_VARIABLE simd_t const inverse_length_scale = simd_all_lanes(1.0 / length_scales[k]);
        for (int i = 0; i < points.stride; i += lanes) {
            const int count = std::max(0, std::min(lanes, n_points - i));
            simd_storeu(divided + i, simd_multiply(simd_load_first_lanes(coordinates + i, count),
                                                   inverse_length_scale));
        }
    }
    return points;

}

//  (the squared distances between the points first_row, ..., first_row + lanes - 1 and the point column)
ALWAYS_INLINE simd_t squared_distances_of_a_packet(const points_divided_by_their_length_scales& points,
                                                   const int first_row, const int column) {

    BMVP_ALIGN_OF_A_SIMD_VARIABLE simd_t sums = simd_zero();
    for (int k = 0; k < points.n_dimensions; ++k) {
        const double* dimension = points.values + static_cast<std::ptrdiff_t>(k) * points.stride;
        BMVP_ALIGN_OF_A_SIMD_VARIABLE simd_t const difference
            = simd_subtract(simd_loadu(dimension + first_row), simd_all_lanes(dimension[column]));
        sums = simd_multiply_add(difference, difference, sums);
    }
    return sums;

}


// ---- the lower triangle and the diagonal of add_diag(K, diagonal) into column-major n x n storage (its strict
//      upper triangle is not written); the diagonal term of point j is diagonal_values[j diagonal_stride]
//      (diagonal_stride 0: one term for every point)
template <int kernel>
inline void lower_triangle_of_covariance_plus_diagonal(const points_divided_by_their_length_scales& points,
                                                       const double squared_sigma_value,
                                                       const constants_of_the_periodic_kernel& periodic,
                                                       const double* diagonal_values, const int diagonal_stride,
                                                       double* values) {

    const int n = points.n_points;
    const int n_panels = (n + size_of_the_tiles - 1) / size_of_the_tiles;
    tiles::for_each_tile(n_panels, [&](const int panel) {

        BMVP_ALIGN_OF_A_SIMD_VARIABLE simd_t const squared_sigma = simd_all_lanes(squared_sigma_value);
        BMVP_ALIGN_OF_A_SIMD_VARIABLE simd_t kernel_value;
        BMVP_ALIGN_OF_A_SIMD_VARIABLE simd_t derivative_weight_not_used;
        BMVP_ALIGN_OF_A_SIMD_VARIABLE simd_t period_term_not_used;
        const int first_column = panel * size_of_the_tiles;
        const int end_column = std::min(n, first_column + size_of_the_tiles);
        for (int j = first_column; j < end_column; ++j) {
            double* column = values + static_cast<std::ptrdiff_t>(j) * n;
            for (int r = j + 1; r < n; r += lanes) {
                kernel_value_and_derivative_weight_of_a_packet<kernel, false>(
                        squared_distances_of_a_packet(points, r, j), squared_sigma, periodic, kernel_value,
                        derivative_weight_not_used, period_term_not_used);
                simd_store_first_lanes(column + r, kernel_value, n - r);
            }
            // (add_diag's single addition, with its rounding)
            column[j] = squared_sigma_value;
            column[j] += diagonal_values[static_cast<std::ptrdiff_t>(j) * diagonal_stride];
        }

    });

}


// ---- the sums over the lower triangle of the weight matrix Omega that the gradient needs (see the header)
struct sums_of_the_weights_times_the_derivatives_of_the_covariance {
    double weights_times_kernel_values_below_the_diagonal = 0.0;
    //  (one length scale: sum_{i > j} Omega_ij g_ij d_ij^2)
    double weights_times_derivative_weights_times_squared_distances_below_the_diagonal = 0.0;
    //  (one length scale per dimension: sum_{i > j} Omega_ij g_ij ((x_ik - x_jk) / l_k)^2, for each k)
    std::vector<double> weights_times_derivative_weights_times_squared_differences_of_each_dimension;
    //  (the periodic kernel: sum_{i > j} Omega_ij K_ij sin(2 pi r_ij / p) r_ij)
    double weights_times_period_terms_below_the_diagonal = 0.0;
    double weights_on_the_diagonal = 0.0;
};

//  The tiles of the lower triangle of Omega:
//  fill_the_tile_of_weights(first_row, rows, first_column, columns, tile) writes the tile's Omega into tile
//  (leading dimension rows; on a diagonal tile, its lower triangle at least); then each column segment below the
//  diagonal in SIMD. With weights_on_the_diagonal_of_each_point (not nullptr), Omega_ii is also written there,
//  for each point.
template <int kernel, typename Fill_the_tile_of_weights>
inline sums_of_the_weights_times_the_derivatives_of_the_covariance sums_over_the_tiles_of_the_weights(
        const points_divided_by_their_length_scales& points, const double squared_sigma_value,
        const constants_of_the_periodic_kernel& periodic,
        const bool one_length_scale_per_dimension, const Fill_the_tile_of_weights& fill_the_tile_of_weights,
        double* weights_on_the_diagonal_of_each_point) {

    const int n = points.n_points;
    const int n_dimensions = points.n_dimensions;
    const int b = size_of_the_tiles;
    const int n_panels = (n + b - 1) / b;
    std::vector<int> row_panel_of_each_tile;
    std::vector<int> column_panel_of_each_tile;
    for (int column_panel = 0; column_panel < n_panels; ++column_panel) {
        for (int row_panel = column_panel; row_panel < n_panels; ++row_panel) {
            row_panel_of_each_tile.push_back(row_panel);
            column_panel_of_each_tile.push_back(column_panel);
        }
    }
    const int n_tiles = static_cast<int>(row_panel_of_each_tile.size());
    const int n_sums_of_the_dimensions = one_length_scale_per_dimension ? n_dimensions : 0;
    std::vector<sums_of_the_weights_times_the_derivatives_of_the_covariance> sums_of_each_tile(n_tiles);
    tiles::for_each_tile(n_tiles, [&](const int tile) {

        const int first_row = row_panel_of_each_tile[tile] * b;
        const int rows = std::min(b, n - first_row);
        const int first_column = column_panel_of_each_tile[tile] * b;
        const int columns = std::min(b, n - first_column);
        const bool on_the_diagonal = first_row == first_column;
        thread_local std::vector<double> buffer_of_the_tile;
        thread_local std::vector<double> sums_of_each_dimension_in_lanes;
        buffer_of_the_tile.resize(static_cast<std::size_t>(b) * b);
        sums_of_each_dimension_in_lanes.assign(static_cast<std::size_t>(n_sums_of_the_dimensions) * lanes, 0.0);
        double* weights = buffer_of_the_tile.data();
        fill_the_tile_of_weights(first_row, rows, first_column, columns, weights);
        sums_of_the_weights_times_the_derivatives_of_the_covariance sums;
        sums.weights_times_derivative_weights_times_squared_differences_of_each_dimension.assign(
                n_sums_of_the_dimensions, 0.0);
        BMVP_ALIGN_OF_A_SIMD_VARIABLE simd_t const squared_sigma = simd_all_lanes(squared_sigma_value);
        BMVP_ALIGN_OF_A_SIMD_VARIABLE simd_t sums_of_weights_times_kernel_values = simd_zero();
        BMVP_ALIGN_OF_A_SIMD_VARIABLE simd_t sums_of_weights_times_derivative_weights_times_squared_distances
            = simd_zero();
        BMVP_ALIGN_OF_A_SIMD_VARIABLE simd_t sums_of_weights_times_period_terms = simd_zero();
        BMVP_ALIGN_OF_A_SIMD_VARIABLE simd_t kernel_value;
        BMVP_ALIGN_OF_A_SIMD_VARIABLE simd_t derivative_weight;
        BMVP_ALIGN_OF_A_SIMD_VARIABLE simd_t period_term;
        for (int column = 0; column < columns; ++column) {
            const int column_of_K = first_column + column;
            const int first_row_in_the_tile = on_the_diagonal ? column + 1 : 0;
            const int count = rows - first_row_in_the_tile;
            const double* weights_of_the_column = weights + static_cast<std::ptrdiff_t>(column) * rows;
            if (on_the_diagonal) {
                sums.weights_on_the_diagonal += weights_of_the_column[column];
                if (weights_on_the_diagonal_of_each_point != nullptr) {
                    weights_on_the_diagonal_of_each_point[column_of_K] = weights_of_the_column[column];
                }
            }
            const int first_row_of_K = first_row + first_row_in_the_tile;
            for (int r = 0; r < count; r += lanes) {
                BMVP_ALIGN_OF_A_SIMD_VARIABLE simd_t const weight
                    = simd_load_first_lanes(weights_of_the_column + first_row_in_the_tile + r, count - r);
                BMVP_ALIGN_OF_A_SIMD_VARIABLE simd_t const squared_distance
                    = squared_distances_of_a_packet(points, first_row_of_K + r, column_of_K);
                kernel_value_and_derivative_weight_of_a_packet<kernel, true>(squared_distance, squared_sigma,
                                                                             periodic, kernel_value,
                                                                             derivative_weight, period_term);
                sums_of_weights_times_kernel_values
                    = simd_multiply_add(weight, kernel_value, sums_of_weights_times_kernel_values);
                BMVP_ALIGN_OF_A_SIMD_VARIABLE simd_t const weight_times_derivative_weight
                    = simd_multiply(weight, derivative_weight);
                if constexpr (kernel == periodic_kernel) {
                    sums_of_weights_times_derivative_weights_times_squared_distances
                        = simd_add(weight_times_derivative_weight,
                                   sums_of_weights_times_derivative_weights_times_squared_distances);
                    sums_of_weights_times_period_terms
                        = simd_multiply_add(weight, period_term, sums_of_weights_times_period_terms);
                } else if (!one_length_scale_per_dimension) {
                    sums_of_weights_times_derivative_weights_times_squared_distances
                        = simd_multiply_add(weight_times_derivative_weight, squared_distance,
                                            sums_of_weights_times_derivative_weights_times_squared_distances);
                } else {
                    for (int k = 0; k < n_dimensions; ++k) {
                        const double* dimension = points.values + static_cast<std::ptrdiff_t>(k) * points.stride;
                        double* sums_of_dimension_k = sums_of_each_dimension_in_lanes.data()
                                                      + static_cast<std::ptrdiff_t>(k) * lanes;
                        BMVP_ALIGN_OF_A_SIMD_VARIABLE simd_t const difference
                            = simd_subtract(simd_loadu(dimension + first_row_of_K + r),
                                            simd_all_lanes(dimension[column_of_K]));
                        simd_storeu(sums_of_dimension_k,
                                    simd_multiply_add(simd_multiply(weight_times_derivative_weight, difference),
                                                      difference, simd_loadu(sums_of_dimension_k)));
                    }
                }
            }
        }
        sums.weights_times_kernel_values_below_the_diagonal
            = simd_sum_of_the_lanes(sums_of_weights_times_kernel_values);
        sums.weights_times_derivative_weights_times_squared_distances_below_the_diagonal
            = simd_sum_of_the_lanes(sums_of_weights_times_derivative_weights_times_squared_distances);
        sums.weights_times_period_terms_below_the_diagonal
            = simd_sum_of_the_lanes(sums_of_weights_times_period_terms);
        for (int k = 0; k < n_sums_of_the_dimensions; ++k) {
            sums.weights_times_derivative_weights_times_squared_differences_of_each_dimension[k]
                = simd_sum_of_the_lanes(simd_loadu(sums_of_each_dimension_in_lanes.data()
                                                   + static_cast<std::ptrdiff_t>(k) * lanes));
        }
        sums_of_each_tile[tile] = sums;

    });
    sums_of_the_weights_times_the_derivatives_of_the_covariance sums;
    sums.weights_times_derivative_weights_times_squared_differences_of_each_dimension.assign(
            n_sums_of_the_dimensions, 0.0);
    for (int tile = 0; tile < n_tiles; ++tile) {
        const auto& sums_of_the_tile = sums_of_each_tile[tile];
        sums.weights_times_kernel_values_below_the_diagonal
            += sums_of_the_tile.weights_times_kernel_values_below_the_diagonal;
        sums.weights_times_derivative_weights_times_squared_distances_below_the_diagonal
            += sums_of_the_tile.weights_times_derivative_weights_times_squared_distances_below_the_diagonal;
        for (int k = 0; k < n_sums_of_the_dimensions; ++k) {
            sums.weights_times_derivative_weights_times_squared_differences_of_each_dimension[k]
                += sums_of_the_tile
                       .weights_times_derivative_weights_times_squared_differences_of_each_dimension[k];
        }
        sums.weights_times_period_terms_below_the_diagonal
            += sums_of_the_tile.weights_times_period_terms_below_the_diagonal;
        sums.weights_on_the_diagonal += sums_of_the_tile.weights_on_the_diagonal;
    }
    return sums;

}


// ---- (1) the tile of W = alpha alpha' - inverse(K), with M = inverse(L) (lower, leading dimension n): the tile
//      of inverse(K) = M' M from the rows of the tile's row panel on (as tiles::sums_over_the_tiles_of_W), then W
//      in SIMD, column segment by column segment (on a diagonal tile, from the diagonal down)
inline void tile_of_alpha_alpha_transpose_minus_inverse_of_K(const double* M, const int n, const double* alpha,
                                                             const int first_row, const int rows,
                                                             const int first_column, const int columns,
                                                             double* tile) {

    const bool on_the_diagonal = first_row == first_column;
    const int first_row_after = first_row + rows;
    const int rows_after = n - first_row_after;
    tiles::view_of_a_block tile_of_inverse_K(tile, rows, columns, Eigen::OuterStride<>(rows));
    const double* M_II = M + first_row + static_cast<std::ptrdiff_t>(first_row) * n;
    const double* M_after_I_I = M + first_row_after + static_cast<std::ptrdiff_t>(first_row) * n;
    if (on_the_diagonal) {
        tile_of_inverse_K.triangularView<Eigen::Lower>()
            = tiles::const_view_of_a_block(M_II, rows, rows, Eigen::OuterStride<>(n));
        if (tiles::lower_triangle_of_transpose_times_itself_of_block(tile, rows, rows) != 0) {
            tiles::throw_not_positive_definite();
        }
        if (rows_after > 0) {
            tiles::add_rank_k_update_of_lower_triangle(1.0, true, M_after_I_I, n, rows, rows_after, tile, rows);
        }
    } else {
        const double* M_IJ = M + first_row + static_cast<std::ptrdiff_t>(first_column) * n;
        const double* M_after_I_J = M + first_row_after + static_cast<std::ptrdiff_t>(first_column) * n;
        tile_of_inverse_K = tiles::const_view_of_a_block(M_IJ, rows, columns, Eigen::OuterStride<>(n));
        tiles::multiply_left_by_lower_triangular(true, M_II, rows, n, tile, columns, rows);
        if (rows_after > 0) {
            tiles::add_product(1.0, true, false, M_after_I_I, n, M_after_I_J, n, rows, columns, rows_after,
                               tile, rows);
        }
    }
    for (int column = 0; column < columns; ++column) {
        const int first_row_in_the_tile = on_the_diagonal ? column : 0;
        double* column_of_the_tile = tile + static_cast<std::ptrdiff_t>(column) * rows;
        BMVP_ALIGN_OF_A_SIMD_VARIABLE simd_t const alpha_of_the_column
            = simd_all_lanes(alpha[first_column + column]);
        for (int r = first_row_in_the_tile; r < rows; r += lanes) {
            const int count = rows - r;
            simd_store_first_lanes(column_of_the_tile + r,
                                   simd_multiply_subtract(simd_load_first_lanes(alpha + first_row + r, count),
                                                          alpha_of_the_column,
                                                          simd_load_first_lanes(column_of_the_tile + r, count)),
                                   count);
        }
    }

}


// ---- (2) the columns first_column, ..., end_column - 1 of R = Q M, Q_ij = a_max(i,j) z_min(i,j), M = inverse(L)
//      (lower, leading dimension n), into R (leading dimension n), each from the first row of its tile down:
//      the suffix sums of a_k M_kj (from the last packet up: an in-register suffix scan of each packet, shifted
//      down one lane, plus the sum of the packets below it), then the prefix sums of z_k M_kj (from the first
//      packet down: an in-register prefix scan plus the sum of the packets above it)
inline void columns_of_Q_times_inverse_of_L(const double* M, const int n, const double* a, const double* z,
                                            const int first_column, const int end_column, double* R) {

    for (int j = first_column; j < end_column; ++j) {
        const double* column_of_M = M + static_cast<std::ptrdiff_t>(j) * n;
        double* column_of_R = R + static_cast<std::ptrdiff_t>(j) * n;
        const int count_from_j = n - j;
        const int n_packets = (count_from_j + lanes - 1) / lanes;
        // (rows j, ..., n - 1: R_ij = z_i sum_{k > i} a_k M_kj)
        BMVP_ALIGN_OF_A_SIMD_VARIABLE simd_t sum_of_the_packets_below = simd_zero();
        for (int packet = n_packets - 1; packet >= 0; --packet) {
            const int r = j + packet * lanes;
            const int count = n - r;
            BMVP_ALIGN_OF_A_SIMD_VARIABLE simd_t const suffix_sums
                = simd_inclusive_suffix_sums(simd_multiply(simd_load_first_lanes(a + r, count),
                                                           simd_load_first_lanes(column_of_M + r, count)));
            BMVP_ALIGN_OF_A_SIMD_VARIABLE simd_t const sums_after_each_row
                = simd_add(simd_shift_down_by_one_lane(suffix_sums), sum_of_the_packets_below);
            simd_store_first_lanes(column_of_R + r,
                                   simd_multiply(simd_load_first_lanes(z + r, count), sums_after_each_row),
                                   count);
            sum_of_the_packets_below = simd_add(sum_of_the_packets_below,
                                                simd_first_lane_in_every_lane(suffix_sums));
        }
        // (rows of j's tile before j: R_ij = z_i sum_{k >= j} a_k M_kj)
        const int first_row_of_the_tile = (j / size_of_the_tiles) * size_of_the_tiles;
        for (int r = first_row_of_the_tile; r < j; r += lanes) {
            const int count = j - r;
            simd_store_first_lanes(column_of_R + r,
                                   simd_multiply(simd_load_first_lanes(z + r, count), sum_of_the_packets_below),
                                   count);
        }
        // (rows j, ..., n - 1: R_ij += a_i sum_{j <= k <= i} z_k M_kj)
        BMVP_ALIGN_OF_A_SIMD_VARIABLE simd_t sum_of_the_packets_above = simd_zero();
        for (int packet = 0; packet < n_packets; ++packet) {
            const int r = j + packet * lanes;
            const int count = n - r;
            BMVP_ALIGN_OF_A_SIMD_VARIABLE simd_t const sums_up_to_each_row
                = simd_add(simd_inclusive_prefix_sums(simd_multiply(simd_load_first_lanes(z + r, count),
                                                                    simd_load_first_lanes(column_of_M + r,
                                                                                          count))),
                           sum_of_the_packets_above);
            simd_store_first_lanes(column_of_R + r,
                                   simd_multiply_add(simd_load_first_lanes(a + r, count), sums_up_to_each_row,
                                                     simd_load_first_lanes(column_of_R + r, count)),
                                   count);
            sum_of_the_packets_above = simd_last_lane_in_every_lane(sums_up_to_each_row);
        }
    }

}

//      the tile of G = M' R: M_II' R_IJ + M(after I, I)' R(after I, J)
inline void tile_of_G(const double* M, const double* R, const int n, const int first_row, const int rows,
                      const int first_column, const int columns, double* tile) {

    const int first_row_after = first_row + rows;
    const int rows_after = n - first_row_after;
    const double* M_II = M + first_row + static_cast<std::ptrdiff_t>(first_row) * n;
    const double* M_after_I_I = M + first_row_after + static_cast<std::ptrdiff_t>(first_row) * n;
    const double* R_IJ = R + first_row + static_cast<std::ptrdiff_t>(first_column) * n;
    const double* R_after_I_J = R + first_row_after + static_cast<std::ptrdiff_t>(first_column) * n;
    tiles::view_of_a_block(tile, rows, columns, Eigen::OuterStride<>(rows))
        = tiles::const_view_of_a_block(R_IJ, rows, columns, Eigen::OuterStride<>(n));
    tiles::multiply_left_by_lower_triangular(true, M_II, rows, n, tile, columns, rows);
    if (rows_after > 0) {
        tiles::add_product(1.0, true, false, M_after_I_I, n, R_after_I_J, n, rows, columns, rows_after, tile,
                           rows);
    }

}


// ---- the adjoint of K + diagonal from the adjoint of its lower Cholesky factor L, the lower triangle of v z'
//      (f = L z, v the adjoint of f), by the blocked Cholesky adjoint of Stan math (Murray 2016, the algorithm
//      of Stan's own cholesky_decompose), IN PLACE: L is in the lower triangle of factor, the adjoint below the
//      diagonal is kept transposed in factor's strict upper triangle (adjoint_ij, i > j, at factor[j + i n]),
//      and its diagonal in adjoint_on_the_diagonal; no other N x N matrix (no copy of L, no N x N adjoint, no
//      v z' made whole). Blocks of Stan's own size, N / 8 rows, at least 8 and at most 128 (as the reverse pass
//      below without the macro): with blocks of the forward tiles (256 rows) the adjoint of the near-singular
//      K + 1e-8 I of GP classification was up to 2.7e-3 off the long double one at N = 500 (two blocks), against
//      at most 6.2e-6 with Stan's. The same operations
//      as Stan's, on the transposed blocks: with R, D, B, C the blocks of L of a step and X' the adjoint of block
//      X kept transposed, C_adj' = D^-T \ C_adj', B_adj' -= R' C_adj', D_adj -= C_adj' C, ..., R_adj' -= B'
//      C_adj' ' and R_adj' -= R' sym(D_adj)
inline void adjoint_of_covariance_from_adjoint_of_cholesky_factor_times_vector_in_place(
        double* factor, const int n, const double* v, const double* z, double* adjoint_on_the_diagonal) {

    using Eigen::Lower;
    using Eigen::StrictlyLower;
    using Eigen::StrictlyUpper;
    using Eigen::Upper;
    Eigen::Map<Eigen::Matrix<double, -1, -1>> A(factor, n, n);
    // (the adjoint of L, v_i z_j for i > j, into column i of the strict upper triangle; v_i z_i on the diagonal)
    for (int i = 1; i < n; ++i) {
        Eigen::Map<Eigen::Matrix<double, -1, 1>>(factor + static_cast<std::ptrdiff_t>(i) * n, i)
            = v[i] * Eigen::Map<const Eigen::Matrix<double, -1, 1>>(z, i);
    }
    for (int i = 0; i < n; ++i) adjoint_on_the_diagonal[i] = v[i] * z[i];
    // const int block_size = std::max(8, std::min(n, size_of_the_tiles));
    const int block_size = std::min(std::max(n / 8, 8), 128);
    thread_local Eigen::Matrix<double, -1, -1> D;
    thread_local Eigen::Matrix<double, -1, -1> D_adjoint;
    for (int k = n; k > 0; k -= block_size) {
        const int j = std::max(0, k - block_size);
        const int b = k - j;
        const int rows_below = n - k;
        auto R = A.block(j, 0, b, j);
        auto B = A.block(k, 0, rows_below, j);
        auto C = A.block(k, j, rows_below, b);
        auto R_adjoint_transposed = A.block(0, j, j, b);
        auto B_adjoint_transposed = A.block(0, k, j, rows_below);
        auto C_adjoint_transposed = A.block(j, k, b, rows_below);
        // (D: the diagonal block of L, transposed; D_adjoint: the lower triangle of its adjoint)
        D.setZero(b, b);
        D.triangularView<Eigen::Upper>() = A.block(j, j, b, b).transpose();
        D_adjoint.setZero(b, b);
        D_adjoint.triangularView<Eigen::StrictlyLower>() = A.block(j, j, b, b).transpose();
        D_adjoint.diagonal() = Eigen::Map<const Eigen::Matrix<double, -1, 1>>(adjoint_on_the_diagonal + j, b);
        if (rows_below > 0) {
            D.triangularView<Eigen::Upper>().solveInPlace(C_adjoint_transposed);
            B_adjoint_transposed.noalias() -= R.transpose() * C_adjoint_transposed;
            D_adjoint.noalias() -= C_adjoint_transposed * C;
        }
        D_adjoint = (D * D_adjoint.triangularView<Eigen::Lower>()).eval();
        D_adjoint.triangularView<Eigen::StrictlyUpper>() =
            D_adjoint.adjoint().triangularView<Eigen::StrictlyUpper>();
        D.triangularView<Eigen::Upper>().solveInPlace(D_adjoint);
        D.triangularView<Eigen::Upper>().solveInPlace(D_adjoint.transpose());
        if (rows_below > 0) {
            R_adjoint_transposed.noalias() -= B.transpose() * C_adjoint_transposed.transpose();
        }
        R_adjoint_transposed.noalias() -= R.transpose() * D_adjoint.selfadjointView<Eigen::Lower>();
        D_adjoint.diagonal() *= 0.5;
        // (back: the strict lower triangle transposed, and the diagonal)
        A.block(j, j, b, b).triangularView<Eigen::StrictlyUpper>() = D_adjoint.transpose();
        Eigen::Map<Eigen::Matrix<double, -1, 1>>(adjoint_on_the_diagonal + j, b) = D_adjoint.diagonal();
    }

}

//      the tile of Omega from the adjoint kept in place by the function above: adjoint_ij below the diagonal (at
//      factor[j + i n], read along row i of the adjoint), 2 adjoint_ii on it (Omega counts half of the
//      diagonal), 0 above
inline void tile_of_the_adjoint_kept_in_place(const double* factor, const double* adjoint_on_the_diagonal,
                                              const int n, const int first_row, const int rows,
                                              const int first_column, const int columns, double* tile) {

    for (int row = 0; row < rows; ++row) {
        const int i = first_row + row;
        const double* row_of_the_adjoint = factor + static_cast<std::ptrdiff_t>(i) * n;
        for (int column = 0; column < columns; ++column) {
            const int j = first_column + column;
            tile[row + static_cast<std::ptrdiff_t>(column) * rows]
                = i > j ? row_of_the_adjoint[j] : (i == j ? 2.0 * adjoint_on_the_diagonal[i] : 0.0);
        }
    }

}

// ---- the arguments: the points dimension by dimension, the length scales, the diagonal terms
//      (array[] real: one dimension)
template <typename T_x, std::enable_if_t<std::is_arithmetic<T_x>::value>* = nullptr>
inline std::vector<double> points_dimension_by_dimension(const std::vector<T_x>& x, int& n_dimensions) {

    n_dimensions = 1;
    return std::vector<double>(x.begin(), x.end());

}

//      (array[] vector: every point with the same number of dimensions)
template <typename T_x, stan::require_eigen_col_vector_t<T_x>* = nullptr>
inline std::vector<double> points_dimension_by_dimension(const std::vector<T_x>& x, int& n_dimensions) {

    static_assert(std::is_arithmetic<stan::value_type_t<T_x>>::value, "the points x must be data");
    const int n_points = static_cast<int>(x.size());
    n_dimensions = n_points > 0 ? static_cast<int>(x[0].size()) : 1;
    std::vector<double> by_dimension(static_cast<std::size_t>(n_dimensions) * n_points);
    for (int i = 0; i < n_points; ++i) {
        stan::math::check_size_match("nico_gp", "dimensions of each point", x[i].size(), "dimensions of point 1",
                                     n_dimensions);
        for (int k = 0; k < n_dimensions; ++k) {
            by_dimension[static_cast<std::size_t>(k) * n_points + i] = x[i].coeff(k);
        }
    }
    return by_dimension;

}

template <typename T_length_scale>
inline std::vector<double> length_scale_of_each_dimension(const char* function_name,
                                                          const T_length_scale& length_scale,
                                                          const int n_dimensions) {

    if constexpr (stan::is_std_vector<T_length_scale>::value) {
        stan::math::check_size_match(function_name, "size of length_scale", length_scale.size(),
                                     "dimensions of the points", n_dimensions);
        const auto values = stan::math::value_of(length_scale);
        return std::vector<double>(values.begin(), values.end());
    } else {
        return std::vector<double>(n_dimensions, stan::math::value_of(length_scale));
    }

}

template <typename T_diagonal>
inline std::vector<double> diagonal_terms(const char* function_name, const T_diagonal& diagonal, const int n,
                                          int& diagonal_stride) {

    if constexpr (stan::is_eigen<T_diagonal>::value) {
        stan::math::check_size_match(function_name, "size of diagonal", diagonal.size(), "number of points", n);
        diagonal_stride = 1;
        const Eigen::Matrix<double, -1, 1> values = stan::math::value_of(diagonal);
        return std::vector<double>(values.data(), values.data() + values.size());
    } else {
        diagonal_stride = 0;
        return std::vector<double>(1, stan::math::value_of(diagonal));
    }

}

template <typename T_sigma, typename T_length_scale, typename T_period, typename T_diagonal>
inline void check_arguments(const char* function_name, const std::vector<double>& x_by_dimension,
                            const int kernel, const T_sigma& sigma, const T_length_scale& length_scale,
                            const T_period& period, const bool period_given, const T_diagonal& diagonal) {

    stan::math::check_bounded(function_name, "kernel", kernel, squared_exponential_kernel, periodic_kernel);
    if (kernel == periodic_kernel) {
        if (!period_given) {
            throw std::domain_error(std::string(function_name)
                                    + ": the periodic kernel (5) needs the period argument");
        }
        if (stan::is_std_vector<T_length_scale>::value) {
            throw std::domain_error(std::string(function_name)
                                    + ": the periodic kernel (5) has one length scale (as Stan's "
                                    + "gp_periodic_cov)");
        }
        stan::math::check_positive_finite(function_name, "period", period);
    }
    stan::math::check_positive(function_name, "magnitude", sigma);
    stan::math::check_positive_finite(function_name, "length scale", length_scale);
    stan::math::check_not_nan(function_name, "x", x_by_dimension);
    stan::math::check_not_nan(function_name, "diagonal term", diagonal);

}

//  (the periodic kernel of a distance in two or more dimensions is not positive definite in general; it is a
//   covariance for one-dimensional points, e.g. times)
inline void check_that_the_periodic_kernel_has_points_of_one_dimension(const char* function_name,
                                                                       const int kernel, const int n_dimensions) {

    if (kernel == periodic_kernel && n_dimensions != 1) {
        throw std::domain_error(std::string(function_name) + ": the periodic kernel (5) needs points of one "
                                + "dimension (in more, sin^2 of the distance is not a covariance in general)");
    }

}


// ---- the adjoints of sigma, the length scales and the diagonal terms from the sums over Omega (times factor)
struct nodes_of_the_covariance_parameters {
    vari*  sigma_node = nullptr;                       // nullptr: data
    vari** length_scale_nodes = nullptr;               // one, or one per dimension; nullptr: data
    int    n_length_scale_nodes = 0;
    vari*  diagonal_node = nullptr;                    // one diagonal term, a parameter
    vari** diagonal_nodes_of_each_point = nullptr;     // one per point, parameters
    vari*  period_node = nullptr;                      // the periodic kernel's period, a parameter
};

inline void add_the_adjoints_of_the_covariance_parameters(
        const nodes_of_the_covariance_parameters& nodes,
        const sums_of_the_weights_times_the_derivatives_of_the_covariance& sums,
        const double* weights_on_the_diagonal_of_each_point, const int n, const double sigma_value,
        const double* length_scales, const constants_of_the_periodic_kernel& periodic, const double factor) {

    if (nodes.sigma_node != nullptr) {
        nodes.sigma_node->adj_ += factor * (2.0 * sums.weights_times_kernel_values_below_the_diagonal
                                            + sigma_value * sigma_value * sums.weights_on_the_diagonal)
                                  / sigma_value;
    }
    if (nodes.length_scale_nodes != nullptr) {
        if (nodes.n_length_scale_nodes == 1) {
            nodes.length_scale_nodes[0]->adj_
                += factor * sums.weights_times_derivative_weights_times_squared_distances_below_the_diagonal
                   / length_scales[0];
        } else {
            for (int k = 0; k < nodes.n_length_scale_nodes; ++k) {
                nodes.length_scale_nodes[k]->adj_
                    += factor
                       * sums.weights_times_derivative_weights_times_squared_differences_of_each_dimension[k]
                       / length_scales[k];
            }
        }
    }
    if (nodes.period_node != nullptr) {
        nodes.period_node->adj_ += factor * two_pi * sums.weights_times_period_terms_below_the_diagonal
                                   / (periodic.length_scale * periodic.length_scale * periodic.period
                                      * periodic.period);
    }
    if (nodes.diagonal_node != nullptr) nodes.diagonal_node->adj_ += factor * 0.5 * sums.weights_on_the_diagonal;
    if (nodes.diagonal_nodes_of_each_point != nullptr) {
        for (int i = 0; i < n; ++i) {
            nodes.diagonal_nodes_of_each_point[i]->adj_
                += factor * 0.5 * weights_on_the_diagonal_of_each_point[i];
        }
    }

}

//  (the nodes of the covariance parameters that are parameters, on the arena)
template <typename T_sigma, typename T_length_scale, typename T_period, typename T_diagonal>
inline nodes_of_the_covariance_parameters nodes_of(const T_sigma& sigma, const T_length_scale& length_scale,
                                                   const T_period& period, const T_diagonal& diagonal,
                                                   const int n_dimensions) {

    auto& arena = stan::math::ChainableStack::instance_->memalloc_;
    nodes_of_the_covariance_parameters nodes;
    if constexpr (stan::is_var<T_sigma>::value) nodes.sigma_node = sigma.vi_;
    if constexpr (stan::is_var<T_period>::value) nodes.period_node = period.vi_;
    if constexpr (stan::is_var<stan::scalar_type_t<T_length_scale>>::value) {
        if constexpr (stan::is_std_vector<T_length_scale>::value) {
            nodes.n_length_scale_nodes = n_dimensions;
            nodes.length_scale_nodes = arena.alloc_array<vari*>(n_dimensions);
            for (int k = 0; k < n_dimensions; ++k) nodes.length_scale_nodes[k] = length_scale[k].vi_;
        } else {
            nodes.n_length_scale_nodes = 1;
            nodes.length_scale_nodes = arena.alloc_array<vari*>(1);
            nodes.length_scale_nodes[0] = length_scale.vi_;
        }
    }
    if constexpr (stan::is_var<stan::scalar_type_t<T_diagonal>>::value) {
        if constexpr (stan::is_eigen<T_diagonal>::value) {
            const int n = static_cast<int>(diagonal.size());
            nodes.diagonal_nodes_of_each_point = arena.alloc_array<vari*>(n);
            for (int i = 0; i < n; ++i) nodes.diagonal_nodes_of_each_point[i] = diagonal.coeff(i).vi_;
        } else {
            nodes.diagonal_node = diagonal.vi_;
        }
    }
    return nodes;

}

// ---- a SUM of kernels, K_ij = sum_t K_t(x_i, x_j), e.g. a smooth trend plus a seasonal pattern (the squared
//      exponential plus the periodic kernel): each term t its own kernel (1 to 5), sigma_t, one length scale l_t
//      and, for the periodic kernel, its period p_t (each data or a parameter); at most maximum_number_of_terms
//      terms. The points are kept undivided (each term divides the squared distance by its own l_t^2), and the
//      gradient of each term's parameters is made, as for one kernel, from the sums over the tiles of Omega:
//      d / d sigma_t = (2 sum_{i > j} Omega_ij K_t,ij + sigma_t^2 sum_i Omega_ii) / sigma_t,
//      d / d l_t = (sum_{i > j} Omega_ij g_t,ij d_t,ij^2) / l_t (the periodic kernel: its derivative weight
//      alone), d / d p_t = 2 pi (sum_{i > j} Omega_ij K_t,ij sin(2 pi r_ij / p_t) r_ij) / (l_t^2 p_t^2)
static constexpr int maximum_number_of_terms = 8;

struct terms_of_a_sum_of_kernels {
    int    n_terms = 0;
    int    kernel_of_each_term[maximum_number_of_terms] = {};
    double sigma_of_each_term[maximum_number_of_terms] = {};
    double length_scale_of_each_term[maximum_number_of_terms] = {};
    double period_of_each_term[maximum_number_of_terms] = {};
};

//  (one packet of one term: its kernel chosen at run time, the same packet functions as one kernel)
template <bool derivative_weight_wanted>
ALWAYS_INLINE void kernel_value_and_derivative_weight_of_a_term(const int kernel, const simd_t squared_distance,
                                                                const simd_t squared_sigma,
                                                                const constants_of_the_periodic_kernel& periodic,
                                                                simd_t& kernel_value, simd_t& derivative_weight,
                                                                simd_t& period_term) {

    switch (kernel) {
        case squared_exponential_kernel:
            kernel_value_and_derivative_weight_of_a_packet<squared_exponential_kernel, derivative_weight_wanted>(
                    squared_distance, squared_sigma, periodic, kernel_value, derivative_weight, period_term);
            break;
        case Matern_1_2_kernel:
            kernel_value_and_derivative_weight_of_a_packet<Matern_1_2_kernel, derivative_weight_wanted>(
                    squared_distance, squared_sigma, periodic, kernel_value, derivative_weight, period_term);
            break;
        case Matern_3_2_kernel:
            kernel_value_and_derivative_weight_of_a_packet<Matern_3_2_kernel, derivative_weight_wanted>(
                    squared_distance, squared_sigma, periodic, kernel_value, derivative_weight, period_term);
            break;
        case Matern_5_2_kernel:
            kernel_value_and_derivative_weight_of_a_packet<Matern_5_2_kernel, derivative_weight_wanted>(
                    squared_distance, squared_sigma, periodic, kernel_value, derivative_weight, period_term);
            break;
        default:
            kernel_value_and_derivative_weight_of_a_packet<periodic_kernel, derivative_weight_wanted>(
                    squared_distance, squared_sigma, periodic, kernel_value, derivative_weight, period_term);
            break;
    }

}

inline constants_of_the_periodic_kernel periodic_constants_of_a_term(const terms_of_a_sum_of_kernels& terms,
                                                                     const int t) {

    constants_of_the_periodic_kernel periodic;
    periodic.length_scale = terms.length_scale_of_each_term[t];
    periodic.period = terms.period_of_each_term[t];
    return periodic;

}

//  (the lower triangle and the diagonal of add_diag(sum_t K_t, diagonal), as lower_triangle_of_covariance_plus_
//   diagonal; points undivided)
inline void lower_triangle_of_covariance_of_a_sum_of_kernels_plus_diagonal(
        const points_divided_by_their_length_scales& points, const terms_of_a_sum_of_kernels& terms,
        const double* diagonal_values, const int diagonal_stride, double* values) {

    const int n = points.n_points;
    const int n_panels = (n + size_of_the_tiles - 1) / size_of_the_tiles;
    double sum_of_the_squared_sigmas = 0.0;
    for (int t = 0; t < terms.n_terms; ++t) {
        sum_of_the_squared_sigmas += terms.sigma_of_each_term[t] * terms.sigma_of_each_term[t];
    }
    tiles::for_each_tile(n_panels, [&](const int panel) {

        BMVP_ALIGN_OF_A_SIMD_VARIABLE simd_t kernel_value;
        BMVP_ALIGN_OF_A_SIMD_VARIABLE simd_t derivative_weight_not_used;
        BMVP_ALIGN_OF_A_SIMD_VARIABLE simd_t period_term_not_used;
        const int first_column = panel * size_of_the_tiles;
        const int end_column = std::min(n, first_column + size_of_the_tiles);
        for (int j = first_column; j < end_column; ++j) {
            double* column = values + static_cast<std::ptrdiff_t>(j) * n;
            for (int r = j + 1; r < n; r += lanes) {
                BMVP_ALIGN_OF_A_SIMD_VARIABLE simd_t const squared_distance
                    = squared_distances_of_a_packet(points, r, j);
                BMVP_ALIGN_OF_A_SIMD_VARIABLE simd_t sum_of_the_terms = simd_zero();
                for (int t = 0; t < terms.n_terms; ++t) {
                    const double length_scale = terms.length_scale_of_each_term[t];
                    const double sigma = terms.sigma_of_each_term[t];
                    kernel_value_and_derivative_weight_of_a_term<false>(
                            terms.kernel_of_each_term[t],
                            simd_multiply(squared_distance, simd_all_lanes(1.0 / (length_scale * length_scale))),
                            simd_all_lanes(sigma * sigma), periodic_constants_of_a_term(terms, t), kernel_value,
                            derivative_weight_not_used, period_term_not_used);
                    sum_of_the_terms = simd_add(sum_of_the_terms, kernel_value);
                }
                simd_store_first_lanes(column + r, sum_of_the_terms, n - r);
            }
            // (add_diag's single addition, with its rounding)
            column[j] = sum_of_the_squared_sigmas;
            column[j] += diagonal_values[static_cast<std::ptrdiff_t>(j) * diagonal_stride];
        }

    });

}

//  (the sums over the lower triangle of Omega, for each term)
struct sums_of_the_weights_for_a_sum_of_kernels {
    double weights_times_kernel_values_below_the_diagonal[maximum_number_of_terms] = {};
    double weights_times_length_scale_terms_below_the_diagonal[maximum_number_of_terms] = {};
    double weights_times_period_terms_below_the_diagonal[maximum_number_of_terms] = {};
    double weights_on_the_diagonal = 0.0;
};

template <typename Fill_the_tile_of_weights>
inline sums_of_the_weights_for_a_sum_of_kernels sums_over_the_tiles_of_the_weights_for_a_sum_of_kernels(
        const points_divided_by_their_length_scales& points, const terms_of_a_sum_of_kernels& terms,
        const Fill_the_tile_of_weights& fill_the_tile_of_weights, double* weights_on_the_diagonal_of_each_point) {

    const int n = points.n_points;
    const int b = size_of_the_tiles;
    const int n_panels = (n + b - 1) / b;
    const int n_terms = terms.n_terms;
    std::vector<int> row_panel_of_each_tile;
    std::vector<int> column_panel_of_each_tile;
    for (int column_panel = 0; column_panel < n_panels; ++column_panel) {
        for (int row_panel = column_panel; row_panel < n_panels; ++row_panel) {
            row_panel_of_each_tile.push_back(row_panel);
            column_panel_of_each_tile.push_back(column_panel);
        }
    }
    const int n_tiles = static_cast<int>(row_panel_of_each_tile.size());
    std::vector<sums_of_the_weights_for_a_sum_of_kernels> sums_of_each_tile(n_tiles);
    tiles::for_each_tile(n_tiles, [&](const int tile) {

        const int first_row = row_panel_of_each_tile[tile] * b;
        const int rows = std::min(b, n - first_row);
        const int first_column = column_panel_of_each_tile[tile] * b;
        const int columns = std::min(b, n - first_column);
        const bool on_the_diagonal = first_row == first_column;
        thread_local std::vector<double> buffer_of_the_tile;
        buffer_of_the_tile.resize(static_cast<std::size_t>(b) * b);
        double* weights = buffer_of_the_tile.data();
        fill_the_tile_of_weights(first_row, rows, first_column, columns, weights);
        // (the lanes of each sum of each term: kernel values, length-scale terms, period terms)
        BMVP_ALIGN_OF_A_SIMD_VARIABLE double lanes_of_the_sums[3 * maximum_number_of_terms * BMVP_LANES] = {};
        BMVP_ALIGN_OF_A_SIMD_VARIABLE simd_t kernel_value;
        BMVP_ALIGN_OF_A_SIMD_VARIABLE simd_t derivative_weight;
        BMVP_ALIGN_OF_A_SIMD_VARIABLE simd_t period_term;
        sums_of_the_weights_for_a_sum_of_kernels sums;
        for (int column = 0; column < columns; ++column) {
            const int column_of_K = first_column + column;
            const int first_row_in_the_tile = on_the_diagonal ? column + 1 : 0;
            const int count = rows - first_row_in_the_tile;
            const double* weights_of_the_column = weights + static_cast<std::ptrdiff_t>(column) * rows;
            if (on_the_diagonal) {
                sums.weights_on_the_diagonal += weights_of_the_column[column];
                if (weights_on_the_diagonal_of_each_point != nullptr) {
                    weights_on_the_diagonal_of_each_point[column_of_K] = weights_of_the_column[column];
                }
            }
            const int first_row_of_K = first_row + first_row_in_the_tile;
            for (int r = 0; r < count; r += lanes) {
                BMVP_ALIGN_OF_A_SIMD_VARIABLE simd_t const weight
                    = simd_load_first_lanes(weights_of_the_column + first_row_in_the_tile + r, count - r);
                BMVP_ALIGN_OF_A_SIMD_VARIABLE simd_t const squared_distance
                    = squared_distances_of_a_packet(points, first_row_of_K + r, column_of_K);
                for (int t = 0; t < n_terms; ++t) {
                    const double length_scale = terms.length_scale_of_each_term[t];
                    const double sigma = terms.sigma_of_each_term[t];
                    BMVP_ALIGN_OF_A_SIMD_VARIABLE simd_t const squared_distance_of_the_term
                        = simd_multiply(squared_distance, simd_all_lanes(1.0 / (length_scale * length_scale)));
                    kernel_value_and_derivative_weight_of_a_term<true>(
                            terms.kernel_of_each_term[t], squared_distance_of_the_term,
                            simd_all_lanes(sigma * sigma), periodic_constants_of_a_term(terms, t), kernel_value,
                            derivative_weight, period_term);
                    double* sums_of_kernel_values = lanes_of_the_sums + (3 * t) * lanes;
                    double* sums_of_length_scale_terms = lanes_of_the_sums + (3 * t + 1) * lanes;
                    double* sums_of_period_terms = lanes_of_the_sums + (3 * t + 2) * lanes;
                    simd_storeu(sums_of_kernel_values,
                                simd_multiply_add(weight, kernel_value, simd_loadu(sums_of_kernel_values)));
                    if (terms.kernel_of_each_term[t] == periodic_kernel) {
                        simd_storeu(sums_of_length_scale_terms,
                                    simd_multiply_add(weight, derivative_weight,
                                                      simd_loadu(sums_of_length_scale_terms)));
                        simd_storeu(sums_of_period_terms,
                                    simd_multiply_add(weight, period_term, simd_loadu(sums_of_period_terms)));
                    } else {
                        simd_storeu(sums_of_length_scale_terms,
                                    simd_multiply_add(simd_multiply(weight, derivative_weight),
                                                      squared_distance_of_the_term,
                                                      simd_loadu(sums_of_length_scale_terms)));
                    }
                }
            }
        }
        for (int t = 0; t < n_terms; ++t) {
            sums.weights_times_kernel_values_below_the_diagonal[t]
                = simd_sum_of_the_lanes(simd_loadu(lanes_of_the_sums + (3 * t) * lanes));
            sums.weights_times_length_scale_terms_below_the_diagonal[t]
                = simd_sum_of_the_lanes(simd_loadu(lanes_of_the_sums + (3 * t + 1) * lanes));
            sums.weights_times_period_terms_below_the_diagonal[t]
                = simd_sum_of_the_lanes(simd_loadu(lanes_of_the_sums + (3 * t + 2) * lanes));
        }
        sums_of_each_tile[tile] = sums;

    });
    sums_of_the_weights_for_a_sum_of_kernels sums;
    for (int tile = 0; tile < n_tiles; ++tile) {
        for (int t = 0; t < n_terms; ++t) {
            sums.weights_times_kernel_values_below_the_diagonal[t]
                += sums_of_each_tile[tile].weights_times_kernel_values_below_the_diagonal[t];
            sums.weights_times_length_scale_terms_below_the_diagonal[t]
                += sums_of_each_tile[tile].weights_times_length_scale_terms_below_the_diagonal[t];
            sums.weights_times_period_terms_below_the_diagonal[t]
                += sums_of_each_tile[tile].weights_times_period_terms_below_the_diagonal[t];
        }
        sums.weights_on_the_diagonal += sums_of_each_tile[tile].weights_on_the_diagonal;
    }
    return sums;

}

//  (the nodes of the parameters of the terms and of the diagonal, on the arena; nullptr: data)
struct nodes_of_the_parameters_of_a_sum_of_kernels {
    vari** sigma_nodes = nullptr;
    vari** length_scale_nodes = nullptr;
    vari** period_nodes = nullptr;
    vari*  diagonal_node = nullptr;
    vari** diagonal_nodes_of_each_point = nullptr;
};

template <typename T_vector>
inline vari** nodes_of_a_vector(const T_vector& vector_of_values) {

    if constexpr (stan::is_var<stan::scalar_type_t<T_vector>>::value) {
        auto& arena = stan::math::ChainableStack::instance_->memalloc_;
        const int size = static_cast<int>(vector_of_values.size());
        vari** nodes = arena.alloc_array<vari*>(size > 0 ? size : 1);
        for (int k = 0; k < size; ++k) nodes[k] = vector_of_values.coeff(k).vi_;
        return nodes;
    } else {
        return nullptr;
    }

}

template <typename T_sigmas, typename T_length_scales, typename T_periods, typename T_diagonal>
inline nodes_of_the_parameters_of_a_sum_of_kernels nodes_of_a_sum_of_kernels(const T_sigmas& sigmas,
                                                                             const T_length_scales& length_scales,
                                                                             const T_periods& periods,
                                                                             const T_diagonal& diagonal) {

    nodes_of_the_parameters_of_a_sum_of_kernels nodes;
    nodes.sigma_nodes = nodes_of_a_vector(sigmas);
    nodes.length_scale_nodes = nodes_of_a_vector(length_scales);
    nodes.period_nodes = nodes_of_a_vector(periods);
    if constexpr (stan::is_var<stan::scalar_type_t<T_diagonal>>::value) {
        if constexpr (stan::is_eigen<T_diagonal>::value) {
            nodes.diagonal_nodes_of_each_point = nodes_of_a_vector(diagonal);
        } else {
            nodes.diagonal_node = diagonal.vi_;
        }
    }
    return nodes;

}

inline void add_the_adjoints_of_the_parameters_of_a_sum_of_kernels(
        const nodes_of_the_parameters_of_a_sum_of_kernels& nodes,
        const sums_of_the_weights_for_a_sum_of_kernels& sums,
        const double* weights_on_the_diagonal_of_each_point, const int n, const terms_of_a_sum_of_kernels& terms,
        const double factor) {

    for (int t = 0; t < terms.n_terms; ++t) {
        const double sigma = terms.sigma_of_each_term[t];
        const double length_scale = terms.length_scale_of_each_term[t];
        const double period = terms.period_of_each_term[t];
        if (nodes.sigma_nodes != nullptr) {
            nodes.sigma_nodes[t]->adj_ += factor * (2.0 * sums.weights_times_kernel_values_below_the_diagonal[t]
                                                    + sigma * sigma * sums.weights_on_the_diagonal)
                                          / sigma;
        }
        if (nodes.length_scale_nodes != nullptr) {
            nodes.length_scale_nodes[t]->adj_
                += factor * sums.weights_times_length_scale_terms_below_the_diagonal[t] / length_scale;
        }
        if (nodes.period_nodes != nullptr && terms.kernel_of_each_term[t] == periodic_kernel) {
            nodes.period_nodes[t]->adj_ += factor * two_pi * sums.weights_times_period_terms_below_the_diagonal[t]
                                           / (length_scale * length_scale * period * period);
        }
    }
    if (nodes.diagonal_node != nullptr) nodes.diagonal_node->adj_ += factor * 0.5 * sums.weights_on_the_diagonal;
    if (nodes.diagonal_nodes_of_each_point != nullptr) {
        for (int i = 0; i < n; ++i) {
            nodes.diagonal_nodes_of_each_point[i]->adj_
                += factor * 0.5 * weights_on_the_diagonal_of_each_point[i];
        }
    }

}

//  (the terms from the arguments, with Stan's checks: 1 to maximum_number_of_terms terms, each kernel 1 to 5,
//   sigma positive, length scale positive and finite, the period positive and finite for a periodic term; the
//   periodic kernel only with points of one dimension)
template <typename T_sigmas, typename T_length_scales, typename T_periods>
inline terms_of_a_sum_of_kernels terms_of_the_arguments(const char* function_name,
                                                        const std::vector<int>& kernels, const T_sigmas& sigmas,
                                                        const T_length_scales& length_scales,
                                                        const T_periods& periods, const int n_dimensions) {

    const int n_terms = static_cast<int>(kernels.size());
    stan::math::check_bounded(function_name, "number of kernels", n_terms, 1, maximum_number_of_terms);
    stan::math::check_size_match(function_name, "size of sigmas", sigmas.size(), "number of kernels", n_terms);
    stan::math::check_size_match(function_name, "size of length_scales", length_scales.size(),
                                 "number of kernels", n_terms);
    stan::math::check_size_match(function_name, "size of periods", periods.size(), "number of kernels", n_terms);
    terms_of_a_sum_of_kernels terms;
    terms.n_terms = n_terms;
    const Eigen::Matrix<double, -1, 1> sigma_values = stan::math::value_of(sigmas);
    const Eigen::Matrix<double, -1, 1> length_scale_values = stan::math::value_of(length_scales);
    const Eigen::Matrix<double, -1, 1> period_values = stan::math::value_of(periods);
    for (int t = 0; t < n_terms; ++t) {
        stan::math::check_bounded(function_name, "kernel", kernels[t], squared_exponential_kernel,
                                  periodic_kernel);
        check_that_the_periodic_kernel_has_points_of_one_dimension(function_name, kernels[t], n_dimensions);
        stan::math::check_positive(function_name, "magnitude", sigma_values(t));
        stan::math::check_positive_finite(function_name, "length scale", length_scale_values(t));
        if (kernels[t] == periodic_kernel) {
            stan::math::check_positive_finite(function_name, "period", period_values(t));
        }
        terms.kernel_of_each_term[t] = kernels[t];
        terms.sigma_of_each_term[t] = sigma_values(t);
        terms.length_scale_of_each_term[t] = length_scale_values(t);
        terms.period_of_each_term[t] = period_values(t);
    }
    return terms;

}

}  // namespace gp_with_a_choice_of_kernel
}  // namespace nico::math


// ---- (1) the marginal log likelihood: its implementation, with the period of the periodic kernel (period_given
//      false: the call without a period, for the kernels 1 to 4); the two Stan-facing overloads follow
namespace nico::math {

template <bool propto__, typename T_y, typename T_mu, typename T_x, typename T_sigma, typename T_length_scale,
          typename T_period, typename T_diagonal>
inline stan::return_type_t<T_y, T_mu, T_sigma, T_length_scale, T_period, T_diagonal>
nico_gp_normal_marginal_lpdf_of_any_kernel_with_its_period(
        const T_y& y, const T_mu& mu, const std::vector<T_x>& x, const int kernel, const T_sigma& sigma,
        const T_length_scale& length_scale, const T_period& period, const bool period_given,
        const T_diagonal& diagonal) {

    namespace gp    = nico::math::gp_with_a_choice_of_kernel;
    namespace tiles = nico::math::gp_exp_quad_cov_normal_marginal;
    using stan::math::var;
    using stan::math::vari;
    using return_t = stan::return_type_t<T_y, T_mu, T_sigma, T_length_scale, T_period, T_diagonal>;
    constexpr bool mu_is_a_vector = stan::is_eigen<T_mu>::value;
    constexpr bool y_is_a_parameter = stan::is_var<stan::value_type_t<T_y>>::value;
    constexpr bool mu_is_a_vector_parameter = mu_is_a_vector && stan::is_var<stan::value_type_t<T_mu>>::value;
    constexpr bool covariance_has_a_parameter = stan::is_var<T_sigma>::value
                                                || stan::is_var<stan::scalar_type_t<T_length_scale>>::value
                                                || stan::is_var<T_period>::value
                                                || stan::is_var<stan::scalar_type_t<T_diagonal>>::value;
    constexpr bool include_constant = stan::math::include_summand<propto__>::value;
    constexpr bool include_log_determinant
        = stan::math::include_summand<propto__, T_sigma, T_length_scale, T_period, T_diagonal>::value;
    const char* function_name = gp::name_of_the_marginal_lpdf;

    const int n = static_cast<int>(x.size());
    int n_dimensions = 1;
    const std::vector<double> x_by_dimension = gp::points_dimension_by_dimension(x, n_dimensions);
    gp::check_that_the_periodic_kernel_has_points_of_one_dimension(function_name, kernel, n_dimensions);
    gp::check_arguments(function_name, x_by_dimension, kernel, sigma, length_scale, period, period_given,
                        diagonal);
    const auto& y_ref  = stan::math::to_ref(y);
    const auto& mu_ref = stan::math::to_ref(mu);
    stan::math::check_size_match(function_name, "Size of random variable", y_ref.size(), "number of points", n);
    if constexpr (mu_is_a_vector) {
        stan::math::check_size_match(function_name, "Size of location parameter", mu_ref.size(),
                                     "number of points", n);
    }
    stan::math::check_not_nan(function_name, "Random variable", stan::math::value_of(y_ref));
    stan::math::check_finite(function_name, "Location parameter", stan::math::value_of(mu_ref));
    const std::vector<double> length_scales = gp::length_scale_of_each_dimension(function_name, length_scale,
                                                                                 n_dimensions);
    int diagonal_stride = 0;
    const std::vector<double> diagonal_values = gp::diagonal_terms(function_name, diagonal, n, diagonal_stride);
    if (!stan::math::include_summand<propto__, T_y, T_mu, T_sigma, T_length_scale, T_period, T_diagonal>::value) {
        return return_t(0.0);
    }
    if (n == 0) return return_t(0.0);
    const double sigma_value = stan::math::value_of(sigma);
    const bool one_length_scale_per_dimension = stan::is_std_vector<T_length_scale>::value;
    gp::constants_of_the_periodic_kernel periodic;
    periodic.length_scale = length_scales[0];
    periodic.period = stan::math::value_of(period);

    if constexpr (!stan::is_var<return_t>::value) {
        // ---- every argument data: the value only, on the heap (nothing on the arena)
        std::vector<double> storage_of_the_points(static_cast<std::size_t>(n_dimensions)
                                                  * gp::stride_of_each_dimension(n));
        const gp::points_divided_by_their_length_scales points
            = gp::divide_the_points_by_their_length_scales(x_by_dimension, n, n_dimensions, length_scales,
                                                           storage_of_the_points.data());
        Eigen::Matrix<double, -1, -1> factor(n, n);
        gp::with_the_kernel(kernel, [&](auto kernel_constant) {

            gp::lower_triangle_of_covariance_plus_diagonal<decltype(kernel_constant)::value>(
                    points, sigma_value * sigma_value, periodic, diagonal_values.data(), diagonal_stride,
                    factor.data());

        });
        tiles::lower_cholesky_factor_in_tiles(factor.data(), n);
        Eigen::Matrix<double, -1, 1> residual(n);
        if constexpr (mu_is_a_vector) {
            residual = stan::math::value_of(y_ref) - stan::math::value_of(mu_ref);
        } else {
            residual = (stan::math::value_of(y_ref).array() - static_cast<double>(mu_ref)).matrix();
        }
        double sum_of_squares_of_z = 0.0;
        double sum_of_log_of_diagonal_of_factor = 0.0;
        tiles::solves_and_log_determinant_from_the_factor(factor.data(), n, residual, false, sum_of_squares_of_z,
                                                          sum_of_log_of_diagonal_of_factor);
        double log_density = -0.5 * sum_of_squares_of_z;
        if (include_log_determinant) log_density -= sum_of_log_of_diagonal_of_factor;
        if (include_constant) log_density -= 0.5 * n * stan::math::LOG_TWO_PI;
        return log_density;
    } else {
        // ---- the one N x N matrix of the node (K, then L, then inverse(L)), the points and alpha, on the arena
        auto& arena = stan::math::ChainableStack::instance_->memalloc_;
        double* storage_of_the_points
            = arena.alloc_array<double>(static_cast<std::size_t>(n_dimensions) * gp::stride_of_each_dimension(n));
        const gp::points_divided_by_their_length_scales points
            = gp::divide_the_points_by_their_length_scales(x_by_dimension, n, n_dimensions, length_scales,
                                                           storage_of_the_points);
        double* factor = arena.alloc_array<double>(static_cast<std::size_t>(n) * n);
        gp::with_the_kernel(kernel, [&](auto kernel_constant) {

            gp::lower_triangle_of_covariance_plus_diagonal<decltype(kernel_constant)::value>(
                    points, sigma_value * sigma_value, periodic, diagonal_values.data(), diagonal_stride, factor);

        });
        tiles::lower_cholesky_factor_in_tiles(factor, n);
        stan::arena_t<Eigen::Matrix<double, -1, 1>> alpha(n);
        if constexpr (mu_is_a_vector) {
            alpha = stan::math::value_of(y_ref) - stan::math::value_of(mu_ref);
        } else {
            alpha = (stan::math::value_of(y_ref).array()
                     - static_cast<double>(stan::math::value_of(mu_ref))).matrix();
        }
        double sum_of_squares_of_z = 0.0;
        double sum_of_log_of_diagonal_of_factor = 0.0;
        tiles::solves_and_log_determinant_from_the_factor(factor, n, alpha, true, sum_of_squares_of_z,
                                                          sum_of_log_of_diagonal_of_factor);
        double log_density = -0.5 * sum_of_squares_of_z;
        if (include_log_determinant) log_density -= sum_of_log_of_diagonal_of_factor;
        if (include_constant) log_density -= 0.5 * n * stan::math::LOG_TWO_PI;

        const gp::nodes_of_the_covariance_parameters nodes = gp::nodes_of(sigma, length_scale, period, diagonal,
                                                                          n_dimensions);
        double* weights_on_the_diagonal_of_each_point
            = nodes.diagonal_nodes_of_each_point != nullptr ? arena.alloc_array<double>(n) : nullptr;
        vari* mu_node = nullptr;
        if constexpr (!mu_is_a_vector && stan::is_var<T_mu>::value) mu_node = mu_ref.vi_;
        // (the var vectors y and mu on the arena, for their adjoints; data: an empty vector, never used)
        using y_on_arena_t  = std::conditional_t<y_is_a_parameter, stan::arena_t<Eigen::Matrix<var, -1, 1>>,
                                                 stan::arena_t<Eigen::Matrix<double, -1, 1>>>;
        using mu_on_arena_t = std::conditional_t<mu_is_a_vector_parameter,
                                                 stan::arena_t<Eigen::Matrix<var, -1, 1>>,
                                                 stan::arena_t<Eigen::Matrix<double, -1, 1>>>;
        y_on_arena_t y_on_arena;
        mu_on_arena_t mu_on_arena;
        if constexpr (y_is_a_parameter) y_on_arena = y_ref;
        if constexpr (mu_is_a_vector_parameter) mu_on_arena = mu_ref;
        const int kernel_value = kernel;
        // (the length scales on the arena: the callback is never destroyed, so it holds no std::vector)
        double* length_scales_on_arena = arena.alloc_array<double>(n_dimensions);
        std::copy(length_scales.begin(), length_scales.end(), length_scales_on_arena);

        return stan::math::make_callback_var(log_density, [factor, alpha, points, n, kernel_value, nodes,
                                                           weights_on_the_diagonal_of_each_point, mu_node,
                                                           y_on_arena, mu_on_arena, sigma_value,
                                                           length_scales_on_arena, periodic,
                                                           one_length_scale_per_dimension](auto& result_node)
                                                           mutable {
            const double adjoint = result_node.adj();
            if constexpr (covariance_has_a_parameter) {
                // ---- inverse(L) in place, then the sums over the tiles of W = alpha alpha' - inverse(K)
                tiles::inverse_of_lower_triangular_matrix_in_tiles(factor, n);
                gp::with_the_kernel(kernel_value, [&](auto kernel_constant) {

                    const gp::sums_of_the_weights_times_the_derivatives_of_the_covariance sums
                        = gp::sums_over_the_tiles_of_the_weights<decltype(kernel_constant)::value>(
                                points, sigma_value * sigma_value, periodic, one_length_scale_per_dimension,
                                [&](const int first_row, const int rows, const int first_column,
                                    const int columns, double* tile) {

                                    gp::tile_of_alpha_alpha_transpose_minus_inverse_of_K(
                                            factor, n, alpha.data(), first_row, rows, first_column, columns,
                                            tile);

                                },
                                weights_on_the_diagonal_of_each_point);
                    gp::add_the_adjoints_of_the_covariance_parameters(nodes, sums,
                                                                      weights_on_the_diagonal_of_each_point, n,
                                                                      sigma_value, length_scales_on_arena,
                                                                      periodic, adjoint);

                });
            }
            if constexpr (y_is_a_parameter) y_on_arena.adj() -= adjoint * alpha;
            if constexpr (mu_is_a_vector_parameter) mu_on_arena.adj() += adjoint * alpha;
            if (mu_node != nullptr) mu_node->adj_ += adjoint * alpha.sum();
        });
    }

}

}  // namespace nico::math

using nico::math::nico_gp_normal_marginal_lpdf_of_any_kernel_with_its_period;


//      the Stan-facing overloads (global scope; stanc calls them as
//      nico_gp_normal_marginal_lpdf<propto__>(y, mu, x, kernel, sigma, length_scale, diagonal, pstream__) and,
//      with the period of the periodic kernel, (y, mu, x, kernel, sigma, length_scale, period, diagonal,
//      pstream__))
namespace nico::math {

template <bool propto__, typename T_y, typename T_mu, typename T_x, typename T_sigma, typename T_length_scale,
          typename T_diagonal, stan::require_eigen_col_vector_t<T_y>* = nullptr,
          stan::require_stan_scalar_t<T_sigma>* = nullptr>
inline stan::return_type_t<T_y, T_mu, T_sigma, T_length_scale, T_diagonal>
nico_gp_normal_marginal_lpdf(const T_y& y, const T_mu& mu, const std::vector<T_x>& x, const int& kernel,
                             const T_sigma& sigma, const T_length_scale& length_scale, const T_diagonal& diagonal,
                             std::ostream* /*pstream__*/) {

    return nico_gp_normal_marginal_lpdf_of_any_kernel_with_its_period<propto__>(y, mu, x, kernel, sigma,
                                                                                length_scale, 1.0, false,
                                                                                diagonal);

}

}  // namespace nico::math

using nico::math::nico_gp_normal_marginal_lpdf;

namespace nico::math {

template <bool propto__, typename T_y, typename T_mu, typename T_x, typename T_sigma, typename T_length_scale,
          typename T_period, typename T_diagonal, stan::require_eigen_col_vector_t<T_y>* = nullptr,
          stan::require_all_stan_scalar_t<T_sigma, T_period>* = nullptr>
inline stan::return_type_t<T_y, T_mu, T_sigma, T_length_scale, T_period, T_diagonal>
nico_gp_normal_marginal_lpdf(const T_y& y, const T_mu& mu, const std::vector<T_x>& x, const int& kernel,
                             const T_sigma& sigma, const T_length_scale& length_scale, const T_period& period,
                             const T_diagonal& diagonal, std::ostream* /*pstream__*/) {

    return nico_gp_normal_marginal_lpdf_of_any_kernel_with_its_period<propto__>(y, mu, x, kernel, sigma,
                                                                                length_scale, period, true,
                                                                                diagonal);

}

}  // namespace nico::math

using nico::math::nico_gp_normal_marginal_lpdf;


// ---- (2) the latent values f = L z: their implementation, with the period of the periodic kernel (period_given
//      false: the call without a period, for the kernels 1 to 4); the two Stan-facing overloads follow
namespace nico::math {

template <typename T_x, typename T_sigma, typename T_length_scale, typename T_period, typename T_diagonal,
          typename T_z>
inline Eigen::Matrix<stan::return_type_t<T_sigma, T_length_scale, T_period, T_diagonal, T_z>, -1, 1>
nico_gp_cholesky_factor_of_covariance_plus_diagonal_times_vector_of_any_kernel_with_its_period(
        const std::vector<T_x>& x, const int kernel, const T_sigma& sigma, const T_length_scale& length_scale,
        const T_period& period, const bool period_given, const T_diagonal& diagonal, const T_z& z) {

    namespace gp    = nico::math::gp_with_a_choice_of_kernel;
    namespace tiles = nico::math::gp_exp_quad_cov_normal_marginal;
    using stan::math::var;
    using return_t = stan::return_type_t<T_sigma, T_length_scale, T_period, T_diagonal, T_z>;
    constexpr bool z_is_a_parameter = stan::is_var<stan::value_type_t<T_z>>::value;
    constexpr bool covariance_has_a_parameter = stan::is_var<T_sigma>::value
                                                || stan::is_var<stan::scalar_type_t<T_length_scale>>::value
                                                || stan::is_var<T_period>::value
                                                || stan::is_var<stan::scalar_type_t<T_diagonal>>::value;
    const char* function_name = gp::name_of_the_latent_values;

    const int n = static_cast<int>(x.size());
    int n_dimensions = 1;
    const std::vector<double> x_by_dimension = gp::points_dimension_by_dimension(x, n_dimensions);
    gp::check_that_the_periodic_kernel_has_points_of_one_dimension(function_name, kernel, n_dimensions);
    gp::check_arguments(function_name, x_by_dimension, kernel, sigma, length_scale, period, period_given,
                        diagonal);
    const auto& z_ref = stan::math::to_ref(z);
    stan::math::check_size_match(function_name, "Size of z", z_ref.size(), "number of points", n);
    stan::math::check_not_nan(function_name, "z", stan::math::value_of(z_ref));
    const std::vector<double> length_scales = gp::length_scale_of_each_dimension(function_name, length_scale,
                                                                                 n_dimensions);
    int diagonal_stride = 0;
    const std::vector<double> diagonal_values = gp::diagonal_terms(function_name, diagonal, n, diagonal_stride);
    if (n == 0) return Eigen::Matrix<return_t, -1, 1>(0);
    const double sigma_value = stan::math::value_of(sigma);
    const bool one_length_scale_per_dimension = stan::is_std_vector<T_length_scale>::value;
    gp::constants_of_the_periodic_kernel periodic;
    periodic.length_scale = length_scales[0];
    periodic.period = stan::math::value_of(period);

    if constexpr (!stan::is_var<return_t>::value) {
        // ---- every argument data: the values only, on the heap
        std::vector<double> storage_of_the_points(static_cast<std::size_t>(n_dimensions)
                                                  * gp::stride_of_each_dimension(n));
        const gp::points_divided_by_their_length_scales points
            = gp::divide_the_points_by_their_length_scales(x_by_dimension, n, n_dimensions, length_scales,
                                                           storage_of_the_points.data());
        Eigen::Matrix<double, -1, -1> factor(n, n);
        gp::with_the_kernel(kernel, [&](auto kernel_constant) {

            gp::lower_triangle_of_covariance_plus_diagonal<decltype(kernel_constant)::value>(
                    points, sigma_value * sigma_value, periodic, diagonal_values.data(), diagonal_stride,
                    factor.data());

        });
        tiles::lower_cholesky_factor_in_tiles(factor.data(), n);
        const Eigen::Matrix<double, -1, 1> z_values = stan::math::value_of(z_ref);
        Eigen::Matrix<double, -1, 1> f_values = factor.triangularView<Eigen::Lower>() * z_values;
        return f_values;
    } else {
        // ---- the one N x N matrix of the node (K, then L, then inverse(L)), the points and z, on the arena
        auto& arena = stan::math::ChainableStack::instance_->memalloc_;
        double* storage_of_the_points
            = arena.alloc_array<double>(static_cast<std::size_t>(n_dimensions) * gp::stride_of_each_dimension(n));
        const gp::points_divided_by_their_length_scales points
            = gp::divide_the_points_by_their_length_scales(x_by_dimension, n, n_dimensions, length_scales,
                                                           storage_of_the_points);
        double* factor = arena.alloc_array<double>(static_cast<std::size_t>(n) * n);
        gp::with_the_kernel(kernel, [&](auto kernel_constant) {

            gp::lower_triangle_of_covariance_plus_diagonal<decltype(kernel_constant)::value>(
                    points, sigma_value * sigma_value, periodic, diagonal_values.data(), diagonal_stride, factor);

        });
        tiles::lower_cholesky_factor_in_tiles(factor, n);
        stan::arena_t<Eigen::Matrix<double, -1, 1>> z_values = stan::math::value_of(z_ref);
        const Eigen::Map<const Eigen::Matrix<double, -1, -1>> L_of_the_forward_pass(factor, n, n);
        const Eigen::Matrix<double, -1,
                            1> f_values = L_of_the_forward_pass.triangularView<Eigen::Lower>() * z_values;
        stan::arena_t<Eigen::Matrix<var, -1, 1>> f_on_arena = f_values;

        const gp::nodes_of_the_covariance_parameters nodes = gp::nodes_of(sigma, length_scale, period, diagonal,
                                                                          n_dimensions);
        double* weights_on_the_diagonal_of_each_point
            = nodes.diagonal_nodes_of_each_point != nullptr ? arena.alloc_array<double>(n) : nullptr;
        using z_on_arena_t = std::conditional_t<z_is_a_parameter, stan::arena_t<Eigen::Matrix<var, -1, 1>>,
                                                stan::arena_t<Eigen::Matrix<double, -1, 1>>>;
        z_on_arena_t z_on_arena;
        if constexpr (z_is_a_parameter) z_on_arena = z_ref;
        const int kernel_value = kernel;
        // (the length scales on the arena: the callback is never destroyed, so it holds no std::vector)
        double* length_scales_on_arena = arena.alloc_array<double>(n_dimensions);
        std::copy(length_scales.begin(), length_scales.end(), length_scales_on_arena);

        stan::math::reverse_pass_callback([factor, f_on_arena, z_values, z_on_arena, points, n, kernel_value,
                                           nodes, weights_on_the_diagonal_of_each_point, sigma_value,
                                           length_scales_on_arena, periodic,
                                           one_length_scale_per_dimension]() mutable {
            // ---- a = L' v (v the adjoint of f): the adjoint of z
            const Eigen::Map<const Eigen::Matrix<double, -1, -1>> L(factor, n, n);
            const Eigen::Matrix<double, -1, 1> v = f_on_arena.adj();
            const Eigen::Matrix<double, -1, 1> a = L.triangularView<Eigen::Lower>().transpose() * v;
            if constexpr (z_is_a_parameter) z_on_arena.adj() += a;
            if constexpr (covariance_has_a_parameter) {
                // // ---- inverse(L) in place, R = Q inverse(L) (column panels on the chain's WCP threads), then
                // //      the sums over the tiles of G = inverse(L)' R
                // tiles::inverse_of_lower_triangular_matrix_in_tiles(factor, n);
                // thread_local std::vector<double> storage_of_R;
                // storage_of_R.resize(static_cast<std::size_t>(n) * n);
                // double* R = storage_of_R.data();
                // const int n_panels = (n + gp::size_of_the_tiles - 1) / gp::size_of_the_tiles;
                // tiles::for_each_tile(n_panels, [&](const int panel) {
                //     const int first_column = panel * gp::size_of_the_tiles;
                //     gp::columns_of_Q_times_inverse_of_L(factor, n, a.data(), z_values.data(), first_column,
                //                                         std::min(n, first_column + gp::size_of_the_tiles), R);
                // });
                // ---- the adjoint of K + diagonal by the blocked Cholesky adjoint of Stan math (Murray 2016, the
                //      algorithm of Stan's own cholesky_decompose), from the adjoint of L, the lower triangle of
                //      v z': about the O(N^3) work of Stan math's reverse pass, never inverse(L), R or G (the
                //      three O(N^3) passes this replaces); Omega_ij = adjoint_ij below the diagonal and
                //      2 adjoint_ii on it (Omega counts the diagonal half, the adjoint of the lower triangle
                //      counts it once)
#ifdef BAYESMVP_GP_CHOLESKY_ADJOINT_IN_PLACE
                // ---- the same adjoint IN PLACE (adjoint_of_covariance_from_adjoint_of_cholesky_factor_times_
                //      vector_in_place): L stays in factor's lower triangle, the adjoint in its strict upper
                //      triangle and on the diagonal vector below; no other N x N matrix
                thread_local std::vector<double> adjoint_on_the_diagonal;
                adjoint_on_the_diagonal.resize(n);
                gp::adjoint_of_covariance_from_adjoint_of_cholesky_factor_times_vector_in_place(
                        factor, n, v.data(), z_values.data(), adjoint_on_the_diagonal.data());
                // (the tiles below may run on the other threads of the chain, where adjoint_on_the_diagonal would
                //  name their own, empty, vector: they read this thread's through a pointer)
                const double* diagonal_of_the_adjoint_on_this_thread = adjoint_on_the_diagonal.data();
#else
                // Eigen::Matrix<double, -1, -1> L_without_its_upper_triangle = Eigen::Matrix<double, -1,
                // -1>::Zero(n, n);
                // L_without_its_upper_triangle.triangularView<Eigen::Lower>()
                //     = L.triangularView<Eigen::Lower>();
                // Eigen::Matrix<double, -1, -1> adjoint_of_K = Eigen::Matrix<double, -1, -1>::Zero(n, n);
                // ---- the two N x N work matrices kept on the thread from one gradient to the next: made new on
                //      the heap at every gradient (2 x 18 MB at N = 1500, 2 x 32 MB at N = 2000), each was mapped
                //      and unmapped by the kernel, and the chains of the process waited on each other's page
                //      faults (22-24% of the CPU time in the kernel at N = 1500 and 2000 on the local-HPC, 0-2%
                //      at N <= 1000); the same values and order of operations. Reached through references, so
                //      that the tiles below, which may run on the other threads of the chain, read this thread's.
                //      Memory: 2 x N^2 doubles per thread that runs this reverse pass, at most 2 x 32 MB at
                //      N = 2000 (180 threads: 11.5 GB on the local-HPC; 16 threads: 1 GB on the laptop);
                //      setZero(n, n) allocates exactly n x n whenever the size changes, so a smaller N after a
                //      larger one in the same thread gives the larger pair back at its first gradient
                thread_local Eigen::Matrix<double, -1, -1> L_without_its_upper_triangle_kept_on_the_thread;
                thread_local Eigen::Matrix<double, -1, -1> adjoint_of_K_kept_on_the_thread;
                Eigen::Matrix<double, -1, -1>& L_without_its_upper_triangle
                    = L_without_its_upper_triangle_kept_on_the_thread;
                Eigen::Matrix<double, -1, -1>& adjoint_of_K = adjoint_of_K_kept_on_the_thread;
                L_without_its_upper_triangle.setZero(n, n);
                L_without_its_upper_triangle.triangularView<Eigen::Lower>()
                    = L.triangularView<Eigen::Lower>();
                adjoint_of_K.setZero(n, n);
                adjoint_of_K.triangularView<Eigen::Lower>() = (v * z_values.transpose()).eval();
                {
                    using Eigen::Lower;
                    using Eigen::StrictlyUpper;
                    using Eigen::Upper;
                    const Eigen::Matrix<double, -1, -1>& L_A = L_without_its_upper_triangle;
                    Eigen::Matrix<double, -1, -1>& L_adj = adjoint_of_K;
                    const int M_ = n;
                    int block_size_ = std::max(M_ / 8, 8);
                    block_size_ = std::min(block_size_, 128);
                    for (int k = M_; k > 0; k -= block_size_) {
                        int j = std::max(0, k - block_size_);
                        auto R = L_A.block(j, 0, k - j, j);
                        auto D = L_A.block(j, j, k - j, k - j).eval();
                        auto B = L_A.block(k, 0, M_ - k, j);
                        auto C = L_A.block(k, j, M_ - k, k - j);
                        auto R_adj = L_adj.block(j, 0, k - j, j);
                        auto D_adj = L_adj.block(j, j, k - j, k - j);
                        auto B_adj = L_adj.block(k, 0, M_ - k, j);
                        auto C_adj = L_adj.block(k, j, M_ - k, k - j);
                        D.transposeInPlace();
                        if (C_adj.size() > 0) {
                            C_adj = D.template
                                triangularView<Eigen::Upper>().solve(C_adj.transpose()).transpose();
                            B_adj.noalias() -= C_adj * R;
                            D_adj.noalias() -= C_adj.transpose() * C;
                        }
                        D_adj = (D * D_adj.template triangularView<Eigen::Lower>()).eval();
                        D_adj.template triangularView<Eigen::StrictlyUpper>()
                            = D_adj.adjoint().template triangularView<Eigen::StrictlyUpper>();
                        D.template triangularView<Eigen::Upper>().solveInPlace(D_adj);
                        D.template triangularView<Eigen::Upper>().solveInPlace(D_adj.transpose());
                        R_adj.noalias() -= C_adj.transpose() * B;
                        R_adj.noalias() -= D_adj.template selfadjointView<Eigen::Lower>() * R;
                        D_adj.diagonal() *= 0.5;
                    }
                }
#endif
                gp::with_the_kernel(kernel_value, [&](auto kernel_constant) {

                    const gp::sums_of_the_weights_times_the_derivatives_of_the_covariance sums
                        = gp::sums_over_the_tiles_of_the_weights<decltype(kernel_constant)::value>(
                                points, sigma_value * sigma_value, periodic, one_length_scale_per_dimension,
                                [&](const int first_row, const int rows, const int first_column,
                                    const int columns, double* tile) {

                                    // gp::tile_of_G(factor, R, n, first_row, rows, first_column, columns, tile);
#ifdef BAYESMVP_GP_CHOLESKY_ADJOINT_IN_PLACE
                                    // gp::tile_of_the_adjoint_kept_in_place(factor,
                                    //                                       adjoint_on_the_diagonal.data(),
                                    //                                       n, first_row, rows, first_column,
                                    //                                       columns, tile);
                                    gp::tile_of_the_adjoint_kept_in_place(factor,
                                                                          diagonal_of_the_adjoint_on_this_thread,
                                                                          n, first_row, rows, first_column,
                                                                          columns, tile);
#else
                                    for (int column = 0; column < columns; ++column) {
                                        for (int row = 0; row < rows; ++row) {
                                            const int i = first_row + row;
                                            const int j = first_column + column;
                                            tile[row + static_cast<std::ptrdiff_t>(column) * rows]
                                                = i > j ? adjoint_of_K(i, j)
                                                        : (i == j ? 2.0 * adjoint_of_K(i, i) : 0.0);
                                        }
                                    }
#endif

                                },
                                weights_on_the_diagonal_of_each_point);
                    gp::add_the_adjoints_of_the_covariance_parameters(nodes, sums,
                                                                      weights_on_the_diagonal_of_each_point, n,
                                                                      sigma_value, length_scales_on_arena,
                                                                      periodic, 1.0);

                });
            }
        });
        return Eigen::Matrix<var, -1, 1>(f_on_arena);
    }

}

}  // namespace nico::math

using nico::math::nico_gp_cholesky_factor_of_covariance_plus_diagonal_times_vector_of_any_kernel_with_its_period;


//      the Stan-facing overloads (global scope; stanc calls them as
//      nico_gp_cholesky_factor_of_covariance_plus_diagonal_times_vector(x, kernel, sigma, length_scale, diagonal,
//      z, pstream__) and, with the period of the periodic kernel, (x, kernel, sigma, length_scale, period,
//      diagonal, z, pstream__))
namespace nico::math {

template <typename T_x, typename T_sigma, typename T_length_scale, typename T_diagonal, typename T_z,
          stan::require_eigen_col_vector_t<T_z>* = nullptr, stan::require_stan_scalar_t<T_sigma>* = nullptr>
inline Eigen::Matrix<stan::return_type_t<T_sigma, T_length_scale, T_diagonal, T_z>, -1, 1>
nico_gp_cholesky_factor_of_covariance_plus_diagonal_times_vector(const std::vector<T_x>& x, const int& kernel,
                                                                 const T_sigma& sigma,
                                                                 const T_length_scale& length_scale,
                                                                 const T_diagonal& diagonal, const T_z& z,
                                                                 std::ostream* /*pstream__*/) {

    return nico_gp_cholesky_factor_of_covariance_plus_diagonal_times_vector_of_any_kernel_with_its_period(
            x, kernel, sigma, length_scale, 1.0, false, diagonal, z);

}

}  // namespace nico::math

using nico::math::nico_gp_cholesky_factor_of_covariance_plus_diagonal_times_vector;

namespace nico::math {

template <typename T_x, typename T_sigma, typename T_length_scale, typename T_period, typename T_diagonal,
          typename T_z, stan::require_eigen_col_vector_t<T_z>* = nullptr,
          stan::require_all_stan_scalar_t<T_sigma, T_period>* = nullptr>
inline Eigen::Matrix<stan::return_type_t<T_sigma, T_length_scale, T_period, T_diagonal, T_z>, -1, 1>
nico_gp_cholesky_factor_of_covariance_plus_diagonal_times_vector(const std::vector<T_x>& x, const int& kernel,
                                                                 const T_sigma& sigma,
                                                                 const T_length_scale& length_scale,
                                                                 const T_period& period,
                                                                 const T_diagonal& diagonal, const T_z& z,
                                                                 std::ostream* /*pstream__*/) {

    return nico_gp_cholesky_factor_of_covariance_plus_diagonal_times_vector_of_any_kernel_with_its_period(
            x, kernel, sigma, length_scale, period, true, diagonal, z);

}

}  // namespace nico::math

using nico::math::nico_gp_cholesky_factor_of_covariance_plus_diagonal_times_vector;


// ---- (3) a SUM of kernels: the marginal log likelihood (stanc calls it as
//      nico_gp_sum_of_kernels_normal_marginal_lpdf<propto__>(y, mu, x, kernels, sigmas, length_scales, periods,
//      diagonal, pstream__)): the same value as multi_normal_lpdf(y | mu, add_diag(sum_t K_t, diagonal)), with
//      kernels an array of kernel numbers (1 to 5), sigmas, length_scales and periods vectors with one value per
//      kernel (the periods read for the periodic kernels only); one node, the gradient by hand
namespace nico::math {

template <bool propto__, typename T_y, typename T_mu, typename T_x, typename T_sigmas, typename T_length_scales,
          typename T_periods, typename T_diagonal, stan::require_eigen_col_vector_t<T_y>* = nullptr,
          stan::require_all_eigen_col_vector_t<T_sigmas, T_length_scales, T_periods>* = nullptr>
inline stan::return_type_t<T_y, T_mu, T_sigmas, T_length_scales, T_periods, T_diagonal>
nico_gp_sum_of_kernels_normal_marginal_lpdf(const T_y& y, const T_mu& mu, const std::vector<T_x>& x,
                                            const std::vector<int>& kernels, const T_sigmas& sigmas,
                                            const T_length_scales& length_scales, const T_periods& periods,
                                            const T_diagonal& diagonal, std::ostream* /*pstream__*/) {

    namespace gp    = nico::math::gp_with_a_choice_of_kernel;
    namespace tiles = nico::math::gp_exp_quad_cov_normal_marginal;
    using stan::math::var;
    using stan::math::vari;
    using return_t = stan::return_type_t<T_y, T_mu, T_sigmas, T_length_scales, T_periods, T_diagonal>;
    constexpr bool mu_is_a_vector = stan::is_eigen<T_mu>::value;
    constexpr bool y_is_a_parameter = stan::is_var<stan::value_type_t<T_y>>::value;
    constexpr bool mu_is_a_vector_parameter = mu_is_a_vector && stan::is_var<stan::value_type_t<T_mu>>::value;
    constexpr bool covariance_has_a_parameter = stan::is_var<stan::scalar_type_t<T_sigmas>>::value
                                                || stan::is_var<stan::scalar_type_t<T_length_scales>>::value
                                                || stan::is_var<stan::scalar_type_t<T_periods>>::value
                                                || stan::is_var<stan::scalar_type_t<T_diagonal>>::value;
    constexpr bool include_constant = stan::math::include_summand<propto__>::value;
    constexpr bool include_log_determinant
        = stan::math::include_summand<propto__, T_sigmas, T_length_scales, T_periods, T_diagonal>::value;
    const char* function_name = "nico_gp_sum_of_kernels_normal_marginal_lpdf";

    const int n = static_cast<int>(x.size());
    int n_dimensions = 1;
    const std::vector<double> x_by_dimension = gp::points_dimension_by_dimension(x, n_dimensions);
    stan::math::check_not_nan(function_name, "x", x_by_dimension);
    stan::math::check_not_nan(function_name, "diagonal term", diagonal);
    const gp::terms_of_a_sum_of_kernels terms = gp::terms_of_the_arguments(function_name, kernels, sigmas,
                                                                           length_scales, periods, n_dimensions);
    const auto& y_ref  = stan::math::to_ref(y);
    const auto& mu_ref = stan::math::to_ref(mu);
    stan::math::check_size_match(function_name, "Size of random variable", y_ref.size(), "number of points", n);
    if constexpr (mu_is_a_vector) {
        stan::math::check_size_match(function_name, "Size of location parameter", mu_ref.size(),
                                     "number of points", n);
    }
    stan::math::check_not_nan(function_name, "Random variable", stan::math::value_of(y_ref));
    stan::math::check_finite(function_name, "Location parameter", stan::math::value_of(mu_ref));
    int diagonal_stride = 0;
    const std::vector<double> diagonal_values = gp::diagonal_terms(function_name, diagonal, n, diagonal_stride);
    if (!stan::math::include_summand<propto__, T_y, T_mu, T_sigmas, T_length_scales, T_periods,
                                     T_diagonal>::value) {

        return return_t(0.0);

    }
    if (n == 0) return return_t(0.0);
    // (the points undivided: each term divides the squared distances by its own length scale)
    const std::vector<double> length_scales_of_one(n_dimensions, 1.0);

    if constexpr (!stan::is_var<return_t>::value) {
        std::vector<double> storage_of_the_points(static_cast<std::size_t>(n_dimensions)
                                                  * gp::stride_of_each_dimension(n));
        const gp::points_divided_by_their_length_scales points
            = gp::divide_the_points_by_their_length_scales(x_by_dimension, n, n_dimensions, length_scales_of_one,
                                                           storage_of_the_points.data());
        Eigen::Matrix<double, -1, -1> factor(n, n);
        gp::lower_triangle_of_covariance_of_a_sum_of_kernels_plus_diagonal(points, terms, diagonal_values.data(),
                                                                           diagonal_stride, factor.data());
        tiles::lower_cholesky_factor_in_tiles(factor.data(), n);
        Eigen::Matrix<double, -1, 1> residual(n);
        if constexpr (mu_is_a_vector) {
            residual = stan::math::value_of(y_ref) - stan::math::value_of(mu_ref);
        } else {
            residual = (stan::math::value_of(y_ref).array() - static_cast<double>(mu_ref)).matrix();
        }
        double sum_of_squares_of_z = 0.0;
        double sum_of_log_of_diagonal_of_factor = 0.0;
        tiles::solves_and_log_determinant_from_the_factor(factor.data(), n, residual, false, sum_of_squares_of_z,
                                                          sum_of_log_of_diagonal_of_factor);
        double log_density = -0.5 * sum_of_squares_of_z;
        if (include_log_determinant) log_density -= sum_of_log_of_diagonal_of_factor;
        if (include_constant) log_density -= 0.5 * n * stan::math::LOG_TWO_PI;
        return log_density;
    } else {
        auto& arena = stan::math::ChainableStack::instance_->memalloc_;
        double* storage_of_the_points
            = arena.alloc_array<double>(static_cast<std::size_t>(n_dimensions) * gp::stride_of_each_dimension(n));
        const gp::points_divided_by_their_length_scales points
            = gp::divide_the_points_by_their_length_scales(x_by_dimension, n, n_dimensions, length_scales_of_one,
                                                           storage_of_the_points);
        double* factor = arena.alloc_array<double>(static_cast<std::size_t>(n) * n);
        gp::lower_triangle_of_covariance_of_a_sum_of_kernels_plus_diagonal(points, terms, diagonal_values.data(),
                                                                           diagonal_stride, factor);
        tiles::lower_cholesky_factor_in_tiles(factor, n);
        stan::arena_t<Eigen::Matrix<double, -1, 1>> alpha(n);
        if constexpr (mu_is_a_vector) {
            alpha = stan::math::value_of(y_ref) - stan::math::value_of(mu_ref);
        } else {
            alpha = (stan::math::value_of(y_ref).array()
                     - static_cast<double>(stan::math::value_of(mu_ref))).matrix();
        }
        double sum_of_squares_of_z = 0.0;
        double sum_of_log_of_diagonal_of_factor = 0.0;
        tiles::solves_and_log_determinant_from_the_factor(factor, n, alpha, true, sum_of_squares_of_z,
                                                          sum_of_log_of_diagonal_of_factor);
        double log_density = -0.5 * sum_of_squares_of_z;
        if (include_log_determinant) log_density -= sum_of_log_of_diagonal_of_factor;
        if (include_constant) log_density -= 0.5 * n * stan::math::LOG_TWO_PI;

        const gp::nodes_of_the_parameters_of_a_sum_of_kernels nodes
            = gp::nodes_of_a_sum_of_kernels(sigmas, length_scales, periods, diagonal);
        double* weights_on_the_diagonal_of_each_point
            = nodes.diagonal_nodes_of_each_point != nullptr ? arena.alloc_array<double>(n) : nullptr;
        vari* mu_node = nullptr;
        if constexpr (!mu_is_a_vector && stan::is_var<T_mu>::value) mu_node = mu_ref.vi_;
        using y_on_arena_t  = std::conditional_t<y_is_a_parameter, stan::arena_t<Eigen::Matrix<var, -1, 1>>,
                                                 stan::arena_t<Eigen::Matrix<double, -1, 1>>>;
        using mu_on_arena_t = std::conditional_t<mu_is_a_vector_parameter,
                                                 stan::arena_t<Eigen::Matrix<var, -1, 1>>,
                                                 stan::arena_t<Eigen::Matrix<double, -1, 1>>>;
        y_on_arena_t y_on_arena;
        mu_on_arena_t mu_on_arena;
        if constexpr (y_is_a_parameter) y_on_arena = y_ref;
        if constexpr (mu_is_a_vector_parameter) mu_on_arena = mu_ref;

        return stan::math::make_callback_var(log_density, [factor, alpha, points, n, terms, nodes,
                                                           weights_on_the_diagonal_of_each_point, mu_node,
                                                           y_on_arena, mu_on_arena](auto& result_node) mutable {
            const double adjoint = result_node.adj();
            if constexpr (covariance_has_a_parameter) {
                // ---- inverse(L) in place, then the sums over the tiles of W = alpha alpha' - inverse(K)
                tiles::inverse_of_lower_triangular_matrix_in_tiles(factor, n);
                const gp::sums_of_the_weights_for_a_sum_of_kernels sums
                    = gp::sums_over_the_tiles_of_the_weights_for_a_sum_of_kernels(
                            points, terms,
                            [&](const int first_row, const int rows, const int first_column, const int columns,
                                double* tile) {

                                gp::tile_of_alpha_alpha_transpose_minus_inverse_of_K(
                                        factor, n, alpha.data(), first_row, rows, first_column, columns, tile);

                            },
                            weights_on_the_diagonal_of_each_point);
                gp::add_the_adjoints_of_the_parameters_of_a_sum_of_kernels(
                        nodes, sums, weights_on_the_diagonal_of_each_point, n, terms, adjoint);
            }
            if constexpr (y_is_a_parameter) y_on_arena.adj() -= adjoint * alpha;
            if constexpr (mu_is_a_vector_parameter) mu_on_arena.adj() += adjoint * alpha;
            if (mu_node != nullptr) mu_node->adj_ += adjoint * alpha.sum();
        });
    }

}

}  // namespace nico::math

using nico::math::nico_gp_sum_of_kernels_normal_marginal_lpdf;


// ---- (4) a SUM of kernels: the latent values f = L z (stanc calls it as
//      nico_gp_sum_of_kernels_cholesky_factor_of_covariance_plus_diagonal_times_vector(x, kernels, sigmas,
//      length_scales, periods, diagonal, z, pstream__)): the same value as
//      cholesky_decompose(add_diag(sum_t K_t, diagonal)) * z; one node, the gradient by hand
namespace nico::math {

template <typename T_x, typename T_sigmas, typename T_length_scales, typename T_periods, typename T_diagonal,
          typename T_z, stan::require_eigen_col_vector_t<T_z>* = nullptr,
          stan::require_all_eigen_col_vector_t<T_sigmas, T_length_scales, T_periods>* = nullptr>
inline Eigen::Matrix<stan::return_type_t<T_sigmas, T_length_scales, T_periods, T_diagonal, T_z>, -1, 1>
nico_gp_sum_of_kernels_cholesky_factor_of_covariance_plus_diagonal_times_vector(
        const std::vector<T_x>& x, const std::vector<int>& kernels, const T_sigmas& sigmas,
        const T_length_scales& length_scales, const T_periods& periods, const T_diagonal& diagonal, const T_z& z,
        std::ostream* /*pstream__*/) {

    namespace gp    = nico::math::gp_with_a_choice_of_kernel;
    namespace tiles = nico::math::gp_exp_quad_cov_normal_marginal;
    using stan::math::var;
    using return_t = stan::return_type_t<T_sigmas, T_length_scales, T_periods, T_diagonal, T_z>;
    constexpr bool z_is_a_parameter = stan::is_var<stan::value_type_t<T_z>>::value;
    constexpr bool covariance_has_a_parameter = stan::is_var<stan::scalar_type_t<T_sigmas>>::value
                                                || stan::is_var<stan::scalar_type_t<T_length_scales>>::value
                                                || stan::is_var<stan::scalar_type_t<T_periods>>::value
                                                || stan::is_var<stan::scalar_type_t<T_diagonal>>::value;
    const char* function_name = "nico_gp_sum_of_kernels_cholesky_factor_of_covariance_plus_diagonal_times_vector";

    const int n = static_cast<int>(x.size());
    int n_dimensions = 1;
    const std::vector<double> x_by_dimension = gp::points_dimension_by_dimension(x, n_dimensions);
    stan::math::check_not_nan(function_name, "x", x_by_dimension);
    stan::math::check_not_nan(function_name, "diagonal term", diagonal);
    const gp::terms_of_a_sum_of_kernels terms = gp::terms_of_the_arguments(function_name, kernels, sigmas,
                                                                           length_scales, periods, n_dimensions);
    const auto& z_ref = stan::math::to_ref(z);
    stan::math::check_size_match(function_name, "Size of z", z_ref.size(), "number of points", n);
    stan::math::check_not_nan(function_name, "z", stan::math::value_of(z_ref));
    int diagonal_stride = 0;
    const std::vector<double> diagonal_values = gp::diagonal_terms(function_name, diagonal, n, diagonal_stride);
    if (n == 0) return Eigen::Matrix<return_t, -1, 1>(0);
    const std::vector<double> length_scales_of_one(n_dimensions, 1.0);

    if constexpr (!stan::is_var<return_t>::value) {
        std::vector<double> storage_of_the_points(static_cast<std::size_t>(n_dimensions)
                                                  * gp::stride_of_each_dimension(n));
        const gp::points_divided_by_their_length_scales points
            = gp::divide_the_points_by_their_length_scales(x_by_dimension, n, n_dimensions, length_scales_of_one,
                                                           storage_of_the_points.data());
        Eigen::Matrix<double, -1, -1> factor(n, n);
        gp::lower_triangle_of_covariance_of_a_sum_of_kernels_plus_diagonal(points, terms, diagonal_values.data(),
                                                                           diagonal_stride, factor.data());
        tiles::lower_cholesky_factor_in_tiles(factor.data(), n);
        const Eigen::Matrix<double, -1, 1> z_values = stan::math::value_of(z_ref);
        Eigen::Matrix<double, -1, 1> f_values = factor.triangularView<Eigen::Lower>() * z_values;
        return f_values;
    } else {
        auto& arena = stan::math::ChainableStack::instance_->memalloc_;
        double* storage_of_the_points
            = arena.alloc_array<double>(static_cast<std::size_t>(n_dimensions) * gp::stride_of_each_dimension(n));
        const gp::points_divided_by_their_length_scales points
            = gp::divide_the_points_by_their_length_scales(x_by_dimension, n, n_dimensions, length_scales_of_one,
                                                           storage_of_the_points);
        double* factor = arena.alloc_array<double>(static_cast<std::size_t>(n) * n);
        gp::lower_triangle_of_covariance_of_a_sum_of_kernels_plus_diagonal(points, terms, diagonal_values.data(),
                                                                           diagonal_stride, factor);
        tiles::lower_cholesky_factor_in_tiles(factor, n);
        stan::arena_t<Eigen::Matrix<double, -1, 1>> z_values = stan::math::value_of(z_ref);
        const Eigen::Map<const Eigen::Matrix<double, -1, -1>> L_of_the_forward_pass(factor, n, n);
        const Eigen::Matrix<double, -1,
                            1> f_values = L_of_the_forward_pass.triangularView<Eigen::Lower>() * z_values;
        stan::arena_t<Eigen::Matrix<var, -1, 1>> f_on_arena = f_values;

        const gp::nodes_of_the_parameters_of_a_sum_of_kernels nodes
            = gp::nodes_of_a_sum_of_kernels(sigmas, length_scales, periods, diagonal);
        double* weights_on_the_diagonal_of_each_point
            = nodes.diagonal_nodes_of_each_point != nullptr ? arena.alloc_array<double>(n) : nullptr;
        using z_on_arena_t = std::conditional_t<z_is_a_parameter, stan::arena_t<Eigen::Matrix<var, -1, 1>>,
                                                stan::arena_t<Eigen::Matrix<double, -1, 1>>>;
        z_on_arena_t z_on_arena;
        if constexpr (z_is_a_parameter) z_on_arena = z_ref;

        stan::math::reverse_pass_callback([factor, f_on_arena, z_values, z_on_arena, points, n, terms, nodes,
                                           weights_on_the_diagonal_of_each_point]() mutable {
            // ---- a = L' v (v the adjoint of f): the adjoint of z
            const Eigen::Map<const Eigen::Matrix<double, -1, -1>> L(factor, n, n);
            const Eigen::Matrix<double, -1, 1> v = f_on_arena.adj();
            const Eigen::Matrix<double, -1, 1> a = L.triangularView<Eigen::Lower>().transpose() * v;
            if constexpr (z_is_a_parameter) z_on_arena.adj() += a;
            if constexpr (covariance_has_a_parameter) {
#ifdef BAYESMVP_GP_CHOLESKY_ADJOINT_IN_PLACE
                // ---- the adjoint of K + diagonal in place, as for one kernel (no inverse(L), R or G), then the
                //      sums over the tiles of Omega read from it
                thread_local std::vector<double> adjoint_on_the_diagonal;
                adjoint_on_the_diagonal.resize(n);
                gp::adjoint_of_covariance_from_adjoint_of_cholesky_factor_times_vector_in_place(
                        factor, n, v.data(), z_values.data(), adjoint_on_the_diagonal.data());
                // (the tiles below may run on the other threads of the chain: they read this thread's diagonal
                //  through a pointer)
                const double* diagonal_of_the_adjoint_on_this_thread = adjoint_on_the_diagonal.data();
                const gp::sums_of_the_weights_for_a_sum_of_kernels sums
                    = gp::sums_over_the_tiles_of_the_weights_for_a_sum_of_kernels(
                            points, terms,
                            [&](const int first_row, const int rows, const int first_column, const int columns,
                                double* tile) {

                                // gp::tile_of_the_adjoint_kept_in_place(factor, adjoint_on_the_diagonal.data(),
                                //                                       n, first_row, rows, first_column,
                                //                                       columns, tile);
                                gp::tile_of_the_adjoint_kept_in_place(factor,
                                                                      diagonal_of_the_adjoint_on_this_thread, n,
                                                                      first_row, rows, first_column, columns,
                                                                      tile);

                            },
                            weights_on_the_diagonal_of_each_point);
                gp::add_the_adjoints_of_the_parameters_of_a_sum_of_kernels(
                        nodes, sums, weights_on_the_diagonal_of_each_point, n, terms, 1.0);
#else
                // ---- inverse(L) in place, R = Q inverse(L), then the sums over the tiles of G = inverse(L)' R
                tiles::inverse_of_lower_triangular_matrix_in_tiles(factor, n);
                thread_local std::vector<double> storage_of_R;
                storage_of_R.resize(static_cast<std::size_t>(n) * n);
                double* R = storage_of_R.data();
                const int n_panels = (n + gp::size_of_the_tiles - 1) / gp::size_of_the_tiles;
                tiles::for_each_tile(n_panels, [&](const int panel) {

                    const int first_column = panel * gp::size_of_the_tiles;
                    gp::columns_of_Q_times_inverse_of_L(factor, n, a.data(), z_values.data(), first_column,
                                                        std::min(n, first_column + gp::size_of_the_tiles), R);

                });
                const gp::sums_of_the_weights_for_a_sum_of_kernels sums
                    = gp::sums_over_the_tiles_of_the_weights_for_a_sum_of_kernels(
                            points, terms,
                            [&](const int first_row, const int rows, const int first_column, const int columns,
                                double* tile) {

                                gp::tile_of_G(factor, R, n, first_row, rows, first_column, columns, tile);

                            },
                            weights_on_the_diagonal_of_each_point);
                gp::add_the_adjoints_of_the_parameters_of_a_sum_of_kernels(
                        nodes, sums, weights_on_the_diagonal_of_each_point, n, terms, 1.0);
#endif
            }
        });
        return Eigen::Matrix<var, -1, 1>(f_on_arena);
    }

}

}  // namespace nico::math

using nico::math::nico_gp_sum_of_kernels_cholesky_factor_of_covariance_plus_diagonal_times_vector;






















