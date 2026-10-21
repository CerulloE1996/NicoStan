#pragma once

//
// -| --------- BayesMVP AVX-512 / AVX2 special functions, exposed to Stan as external functions ------------------------------
//
// Vector -> vector versions of the special functions that dominate the LC-MVOP likelihood, computed
// with the BayesMVP SIMD kernels (fast_and_approx_AVX512_fns.hpp / fast_and_approx_AVX2_fns.hpp)
// instead of Stan's scalar implementations. Each function has
//
//     * a PRIM overload  (Eigen column vector of double)  -> Eigen::VectorXd
//     * a REV  overload  (Eigen column vector of var)     -> Eigen::Matrix<var, -1, 1>
//
// v4 (this file):
//
//   * SELF-CONTAINED. Only the two kernel files are included, with the three macros they need
//     defined here. v1-v3 included BayesMVP's initial_config / SIMD_config / eigen_config /
//     double_fns.hpp, and double_fns.hpp puts `using namespace Eigen; using namespace stan::math;`
//     and Eigen macro redefinitions into the WHOLE model translation unit (the user header is
//     force-included before the generated model code). Nothing of BayesMVP's leaks into the model now.
//
//   * LEAN TAPE. Forward pass: operand values gathered into thread-local scratch, kernel applied,
//     result varis placement-new'ed as ONE contiguous block on the arena and registered on the
//     nochain stack (their chain() is empty). Only the operand vari pointers are stored. Per element:
//     8 (operand ptr) + 24 (vari) bytes on the arena; nothing else. Stan's own path for
//     Eigen::Matrix<var> input (apply_scalar_unary -> one callback vari per element) is 32 bytes
//     per element plus one virtual call per element in the reverse pass.
//     Reverse pass: one callback for the whole vector; it re-reads operand and result values,
//     recomputes F'(x) with the kernels into scratch, and scatters adj(x) += adj(f) * F'(x).
//
//   In isolation (stan_ext/microbench, 96 threads, length-200 vectors) this is 1.1x (log) to 12x
//   (inv_Phi_approx_from_logit_prob) faster than Stan's built-ins, and 10-30% faster than v3.
//
// Declared in the Stan functions block as e.g.   vector nico_Phi(vector x);   (no body) and compiled
// with --allow-undefined and this file as the user header (cmdstanr: user_header = ...;
// BridgeStan / BayesMVP: stanc_args --allow-undefined + make_args USER_HEADER=...).
//
// Functions:  nico_Phi, nico_log_Phi, nico_inv_Phi, nico_exp, nico_log, nico_log1m, nico_log1p,
//             nico_log1p_exp, nico_log1m_exp, nico_inv_logit, nico_log_inv_logit, nico_Phi_approx,
//             nico_log_Phi_approx, nico_inv_Phi_approx, nico_inv_Phi_approx_from_logit_prob,
//             bmvp_simd_lanes (diagnostic: 8 = AVX-512 kernels, 4 = AVX2 kernels)
//
// Kernel dispatch: AVX-512 (8 lanes) when compiled with -mavx512f -mavx512vl -mavx512dq, else AVX2
// (4 lanes) with -mavx2; anything less is a compile error (a silently scalar build is what we do NOT
// want to benchmark). Compiling with -DBAYESMVP_FORCE_AVX2 selects the 4-lane AVX2 kernels
// even when AVX-512 is enabled (e.g. BridgeStan make argument CPPFLAGS_OPTIM=-DBAYESMVP_FORCE_AVX2), so an
// AVX-512 machine can run the AVX2 arm with every other compile flag unchanged; bmvp_simd_lanes() then
// returns 4, which callers check against the requested backend. Tail handling: a partial final block goes through a padded stack buffer whose
// spare lanes repeat the last valid element; only the valid lanes are written back.
//
// -----------------------------------------------------------------------------------------------------------------------------
// ---- the three macros the kernel files use (same definitions as BayesMVP's initial_config / SIMD_config)
// ---- the four global constants the kernel files read (same values as BayesMVP's double_fns.hpp)
// -0.5*log(2*pi)
//  1/sqrt(2*pi)
// ---- the SIMD kernels (self-contained: <immintrin.h>, <cmath>, the macros and constants above). Absolute paths: no -I needed.
// ---- constants
// Stan's Phi_approx: inv_logit(0.07056 x^3 + 1.5976 x)
// exact inverse of Phi_approx's cubic-logit map
// ---- generic elementwise application on a raw buffer, in place. Kernels are passed as callables
// (the BayesMVP kernels are always_inline and cannot be called through function pointers).
// ---- in-place kernels on a buffer of n doubles
// ---- VALUE functions:      f <- F(x)            (x read-only; f written; both length n; f != x)
// ---- DERIVATIVE functions: d <- F'(x)           given the operand values x AND the result values f,
//                                                  so the reverse pass can recompute cheaply.
// phi(x)
// 1 / phi(z)
// log1m_exp(x) = log(1 - exp(x)), x <= 0: log1p(-exp(x)) via the kernels for x < -ln 2; nearer 0
// that loses digits, so those lanes use the scalar log(-expm1(x)).
// -exp(x) / (1 - exp(x))
// 1 - inv_logit(x)
// p = exp(f)
// z = C2 * sinh(asinh(C1 * L) / 3): dz/dL = (C2 C1 / 3) * sqrt(1 + (z / C2)^2) / sqrt(1 + (C1 L)^2)
// ---- thread-local scratch: reused across calls, so no malloc after warm-up and it stays in cache
// ---- the rev-mode wrapper (see the header comment). Arena per element: 8 (operand vari*) + 24 (result vari).
// Select dependencies at compile time: never reread values which the derivative does not use.
// Keep 'both' as the default for existing callers and the matched pre-optimisation benchmark.
// Unused derivative arguments are nullptr; only the declared buffers are populated.
// nochain stack: chain() is empty
// ---- Specialised log / inv_Phi wrappers: same kernels, derivatives and arena ownership as rev_apply.
// Gather once and transform in place. log's reverse pass needs only operand values; inv_Phi's needs
// only result values. Keep adjoint accumulation scalar and ordered: different inputs can alias one vari.
// A constant full-block size lets the compiler remove padding/index tests from the hot loop.
// These two value-only paths can apply the kernel directly to the returned Eigen vector.
// ---- the prim (double) path: one output allocation, same as Stan's own vectorised functions.
// namespace bmvp_stan_ext
// ---- the Stan-facing functions (global scope; stanc calls them as  nico_Phi(x, pstream__)).
/*pstream__*/
/*pstream__*/
// log1m_exp's near-zero repair still needs the original x: do not overwrite it with exp(x).
// Keep the copy path here: in-place evaluation regressed complete-call timings in the matched benchmark.
/*pstream__*/
/*pstream__*/
/*pstream__*/
/*pstream__*/
// ---- diagnostic: which kernel path this build took. Declare in Stan as   real bmvp_simd_lanes();
/*pstream__*/

#include <NicoSIMD/stan_external_functions.hpp>






















