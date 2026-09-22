#pragma once 

#ifndef EIGEN_CONFIG_H
#define EIGEN_CONFIG_H

// Basic configuration
#define EIGEN_NO_DEBUG
#define EIGEN_DONT_PARALLELIZE

// Define EIGEN_MAX_ALIGN_BYTES if it's not already defined
#ifndef EIGEN_MAX_ALIGN_BYTES
#define EIGEN_MAX_ALIGN_BYTES 16  // Default value
#endif

// Modify alignment based on architecture
#ifdef _WIN32
        #if defined(USE_AVX_512)
            #undef EIGEN_MAX_ALIGN_BYTES
            #define EIGEN_VECTORIZE_AVX512
            #define EIGEN_MAX_ALIGN_BYTES 64
        #elif defined(USE_AVX2)
            #undef EIGEN_MAX_ALIGN_BYTES
            #define EIGEN_VECTORIZE_AVX2
            #define EIGEN_MAX_ALIGN_BYTES 32
        #elif defined(USE_AVX)
            #undef EIGEN_MAX_ALIGN_BYTES
            #define EIGEN_VECTORIZE_AVX
            #define EIGEN_MAX_ALIGN_BYTES 16
        #endif
#else  // Non-Windows (Linux, MacOS, etc.)
        #if defined(USE_AVX_512)
            #undef EIGEN_MAX_ALIGN_BYTES
            #define EIGEN_VECTORIZE_AVX512
            #define EIGEN_MAX_ALIGN_BYTES 64
        #elif defined(USE_AVX2)
            #undef EIGEN_MAX_ALIGN_BYTES
            #define EIGEN_VECTORIZE_AVX2
            #define EIGEN_MAX_ALIGN_BYTES 32
        #elif defined(USE_AVX)
            #undef EIGEN_MAX_ALIGN_BYTES
            #define EIGEN_VECTORIZE_AVX
            #define EIGEN_MAX_ALIGN_BYTES 16
        #endif
#endif



#endif // end of EIGEN_CONFIG_H