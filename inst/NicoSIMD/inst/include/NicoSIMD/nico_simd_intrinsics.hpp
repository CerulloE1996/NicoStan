
#pragma once


#ifndef NICO_SIMD_INTRINSICS_HPP
#define NICO_SIMD_INTRINSICS_HPP


////
//// ---- The SIMD intrinsics of NicoSIMD's kernels, on every CPU -
////
//// - x86-64 CPUs (Intel and AMD): the compiler's own <immintrin.h>. The AVX2 kernels are compiled when the model
////   is compiled with -mavx2 (and -mfma), and the AVX-512 kernels when it is compiled with -mavx512f -mavx512vl
////   -mavx512dq (as before: nothing changes on these CPUs).
//// - 64-bit ARM CPUs (e.g., Apple Silicon, Qualcomm Snapdragon, AWS Graviton), which have NEON but no AVX: the
////   AVX2 kernels, through SIMDe ("SIMD Everywhere"; header-only, MIT licence, in inst/include/simde with its
////   COPYING), which implements each AVX2 / FMA intrinsic with NEON instructions (i.e., two 128-bit NEON
////   operations for each 256-bit AVX2 operation). SIMDE_ENABLE_NATIVE_ALIASES makes the kernels' own _mm256_*
////   names and __m256d / __m256i types refer to SIMDe's, so the kernels' code is the same on every CPU.
//// - Any other CPU: no kernels (NICO_SIMD_AVX2_KERNELS stays undefined), and models are built on Stan math.
////
//// NICO_SIMD_AVX2_KERNELS:       the AVX2 kernels are compiled (natively on x86-64, through SIMDe on ARM);
//// NICO_SIMD_NEON_THROUGH_SIMDE: they are compiled through SIMDe, with NEON instructions.
////


#if defined(__x86_64__) || defined(_M_X64)

      #include <immintrin.h>

      #if defined(__AVX2__)
            #define NICO_SIMD_AVX2_KERNELS
      #endif

#elif defined(__aarch64__) || defined(_M_ARM64)

      #ifndef SIMDE_ENABLE_NATIVE_ALIASES
            #define SIMDE_ENABLE_NATIVE_ALIASES
      #endif

      #include "../simde/x86/avx2.h"
      #include "../simde/x86/fma.h"

      //// (SIMDe 0.8.2 gives every _MM_FROUND_* name of the kernels but this one)
      #ifndef _MM_FROUND_NO_EXC
            #define _MM_FROUND_NO_EXC SIMDE_MM_FROUND_NO_EXC
      #endif

      #define NICO_SIMD_AVX2_KERNELS
      #define NICO_SIMD_NEON_THROUGH_SIMDE

#endif


//// Macros to force byte alignment of SIMD local variables on Windows (not needed on Linux); the same definitions
//// as NicoStan's runtime/SIMD_config.hpp, each guarded so that an earlier definition is kept.
#ifdef _MSC_VER
      #ifndef ALIGN64
            #define ALIGN64 __declspec(align(64))
      #endif
      #ifndef ALIGN32
            #define ALIGN32 __declspec(align(32))
      #endif
#else
      #ifndef ALIGN64
            #define ALIGN64 alignas(64)
      #endif
      #ifndef ALIGN32
            #define ALIGN32 alignas(32)
      #endif
#endif


#endif






















