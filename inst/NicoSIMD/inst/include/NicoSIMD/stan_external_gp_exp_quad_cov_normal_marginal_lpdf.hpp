#pragma once
//
// -| --------- BayesMVP marginal log likelihood of a Gaussian process with the squared exponential kernel and a -
//              normal likelihood (a diagonal term), with its gradient computed by hand (ONE autodiff node),
//              exposed to Stan as an external function
//
// nico_gp_exp_quad_cov_normal_marginal_lpdf(y | mu, x, sigma, length_scale, diagonal): the same value as
//
//     multi_normal_lpdf(y | mu, add_diag(gp_exp_quad_cov(x, sigma, length_scale), diagonal))
//
// for a vector y (data or a parameter), mu a real or a vector (data or a parameter), array[] real data points x,
// and real sigma, length_scale and diagonal (each data or a parameter). General: the marginal likelihood of any
// Gaussian process with the squared exponential kernel and a normal likelihood or nugget (the diagonal term: a
// noise variance, a nugget or a jitter), and any multivariate normal with this covariance.
//
// Forward pass (doubles only; nothing on the autodiff tape):
//   K = sigma^2 exp(-0.5 (x_i - x_j)^2 / length_scale^2) + diagonal I, its lower triangle only, written straight
//       into the one N x N matrix of the node, one column segment at a time with BayesMVP's AVX exp kernel
//       (gp_exp_quad_cov::column_segment, stan_external_gp_exp_quad_cov.hpp);
//   L = the Cholesky factor of K, in place, in tiles (below);
//   z = L \ (y - mu),  alpha = L' \ z  (two triangular solves),  log density = -0.5 z' z - sum(log(diagonal(L)))
//       - 0.5 N log(2 pi), the log of the diagonal of L with BayesMVP's AVX log kernel (k_log); with propto =
//       true the terms which do not depend on a parameter are dropped, as in Stan (the constant always;
//       sum(log(diagonal(L))) when sigma, length_scale and diagonal are data).
//
// Reverse pass (ONE callback; the O(N^3) work only when a gradient is asked for, and only when sigma,
// length_scale or diagonal is a parameter), with W = alpha alpha' - inverse(K), all by hand:
//   d lp / d sigma        = 0.5 sum_ij W_ij dK_ij / d sigma        = (2 sum_{i > j} W_ij K_ij
//                                                                     + sigma^2 sum_i W_ii) / sigma,
//   d lp / d length_scale = 0.5 sum_ij W_ij dK_ij / d length_scale = sum_{i > j} W_ij K_ij (x_i - x_j)^2
//                                                                     / length_scale^3,
//   d lp / d diagonal     = 0.5 sum_i W_ii,
//   d lp / d mu = alpha (sum(alpha) for a real mu),   d lp / d y = -alpha;
//   inverse(L) in place over L, in tiles (below); then, for each tile of the lower triangle, its tile of
//   inverse(K) = inverse(L)' inverse(L) made into a tile buffer of its own (in the core's L2), and in the same
//   pass over that tile, column segment by column segment in SIMD: the kernel values K_ij recomputed (BayesMVP's
//   AVX exp), W_ij = alpha_i alpha_j - inverse(K)_ij, and the sums of W_ij K_ij and W_ij K_ij (x_i - x_j)^2
//   (AVX-512: 8 lanes; AVX2, including -DBAYESMVP_FORCE_AVX2: 4 lanes; the last partial packet with masked
//   loads); so inverse(K) is never held as an N x N matrix, and the node holds ONE N x N matrix (K, then L,
//   then inverse(L)).
//
// Tiles: of size_of_the_tiles = 64 rows and columns unless the build sets
// -DBAYESMVP_SIZE_OF_THE_TILES_OF_THE_GP_NODES (three tiles of 64 x 64 doubles, 96 KB, stay in each core's L2:
// 1 MB on the local-HPC, 512 KB on the laptop; of 128 x 128, 384 KB).
// The Cholesky factor (right-looking: the diagonal tile, the tiles below
// it, then the trailing column panels), the inverse of L (from the last column panel to the first: each row tile
// of a panel from the inverted trailing triangle, then the diagonal tile) and the pass above, each with the same
// tiles, in the same order of floating point operations, whatever the number of threads. The kernels of the
// tiles: with -DBAYESMVP_CHOLESKY_FACTOR_AND_INVERSE_WITH_LAPACK_FROM_OPENBLAS, OpenBLAS's dpotrf, dtrtri,
// dlauum, dtrsm, dtrmm, dgemm and dsyrk (one thread per call, below); otherwise Eigen's blocked LLT, the Eigen
// blocks of the inverse below, and Eigen's products, solves and rank-k updates.
//
// Within-chain parallel (WCP) threads: when the chain runs in its own TBB arena of n_threads_WCP threads
// (NicoStan's NICOSTAN_PER_CHAIN_TBB_ARENA=1: the gradient is computed inside the arena,
// the_TBB_arena_of_each_chain.hpp) and the arena has more than one thread, the independent tiles of each step
// (the tiles below a diagonal tile, the trailing panels, the row tiles of a panel of the inverse, the tiles of
// the last pass) are spread over the arena's threads with tbb::parallel_for; otherwise they are done one after
// the other on the chain's thread. Every OpenBLAS call is single-threaded, so the chain never uses more threads
// than its arena: n_threads_WCP in all (1 in the sampling layouts with one thread per chain), never OpenBLAS's
// own.
//
// OpenBLAS (with -DBAYESMVP_CHOLESKY_FACTOR_AND_INVERSE_WITH_LAPACK_FROM_OPENBLAS; 0.3.20 on the local-HPC, built
// without LAPACKE): opened once per process with dlopen(BAYESMVP_OPENBLAS_LIBRARY_FILE, default
// "libopenblas.so.0", RTLD_NOW | RTLD_LOCAL | RTLD_DEEPBIND), and its routines looked up in it with dlsym: an R
// process has R's own reference BLAS (libRblas.so, a dependency of libR.so) among its global symbols, which a
// plain call of dgemm_ from a model library would reach first; RTLD_DEEPBIND makes OpenBLAS's own routines call
// its own BLAS and LAPACK. OPENBLAS_NUM_THREADS is set to 1 before the dlopen when it is not set already (an
// OpenBLAS built with NUM_THREADS 128 starts one thread pool per process at its load and, with 180 chains,
// stops with "too many memory regions"; its openblas_set_num_threads(1) after the load comes too late), and
// openblas_set_num_threads(1) is called once, before the first call (the scripts also set
// OPENBLAS_NUM_THREADS=1, so that OpenBLAS starts no thread pool at all). With OpenBLAS the factor is
// not Stan's bit for bit (another order of the sums): the gradient differs from Stan's by the rounding amplified
// by the conditioning of K.
//
// Floating point operations per gradient (multiply and add counted as two; N = 500; Stan's blocks of
// b = min(max(N / 8, 8), 128) = 62):
//   Stan:  the forward factorisation N^3 / 3 (4.2e7), the Cholesky reverse pass (Murray's blocked algorithm)
//          2 N^3 / 3 + 2.5 b N^2 (8.3e7 + 3.9e7), and O(N^2) for the covariance, the triangular solve and the
//          likelihood terms, with N (N + 1) / 2 scalar nodes for L: about 1.64e8;
//   here:  the factorisation N^3 / 3, the inverse of L N^3 / 3, the tiles of inverse(K) N^3 / 3: about 1.25e8,
//          and O(N^2) (the covariance twice, two triangular solves, the SIMD sums), with no node for K or L.
//
// Checks, as Stan: those of gp_exp_quad_cov (sigma positive, length_scale positive and finite, no point NaN),
// of multi_normal (y not NaN, mu finite, matching sizes), the diagonal term not NaN and of cholesky_decompose
// (positive definite, Stan's message). K is symmetric by construction: its upper triangle is never formed.
//
// Declared in the Stan functions block (no body), one declaration per shape of mu used, as
//
//     real nico_gp_exp_quad_cov_normal_marginal_lpdf(vector y, real mu, data array[] real x, real sigma,
//                                                    real length_scale, real diagonal);
//     real nico_gp_exp_quad_cov_normal_marginal_lpdf(vector y, vector mu, data array[] real x, real sigma,
//                                                    real length_scale, real diagonal);
//
// and called as  y ~ nico_gp_exp_quad_cov_normal_marginal(mu, x, sigma, length_scale, diagonal);  or
// target += nico_gp_exp_quad_cov_normal_marginal_lupdf(y | mu, x, sigma, length_scale, diagonal);
//
// ---------------------------------------------------------------------------------------------------------------

#include "stan_external_functions.hpp"

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/partitioner.h>
#include <tbb/task_arena.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <ostream>
#include <string>
#include <type_traits>
#include <vector>

#ifdef BAYESMVP_CHOLESKY_FACTOR_AND_INVERSE_WITH_LAPACK_FROM_OPENBLAS
#include <dlfcn.h>
#include <stdexcept>
#ifndef BAYESMVP_OPENBLAS_LIBRARY_FILE
#  define BAYESMVP_OPENBLAS_LIBRARY_FILE "libopenblas.so.0"
#endif
#endif

#ifndef BMVP_LANES
#  error "stan_external_gp_exp_quad_cov_normal_marginal_lpdf.hpp needs the kernels of stan_external_functions.hpp"
#endif

namespace nico::math {
namespace gp_exp_quad_cov_normal_marginal {

static constexpr const char* function_name = "nico_gp_exp_quad_cov_normal_marginal_lpdf";

// ---- the rows and columns of a tile: 64 unless the build sets -DBAYESMVP_SIZE_OF_THE_TILES_OF_THE_GP_NODES=<b>
//      (e.g. 128: three tiles of 128 KB still stay in each core's L2, and every pass over the panels of L,
//      inverse(L) and R, whose reads grow with N^3 / b, reads half as much from memory)
#ifndef BAYESMVP_SIZE_OF_THE_TILES_OF_THE_GP_NODES
#  define BAYESMVP_SIZE_OF_THE_TILES_OF_THE_GP_NODES 64
#endif
static constexpr int size_of_the_tiles = BAYESMVP_SIZE_OF_THE_TILES_OF_THE_GP_NODES;
static_assert(size_of_the_tiles >= 16 && size_of_the_tiles % 16 == 0,
              "the size of the tiles of the GP nodes must be a multiple of 16 (whole SIMD packets)");

using view_of_a_block       = Eigen::Map<Eigen::Matrix<double, -1, -1>, 0, Eigen::OuterStride<>>;
using const_view_of_a_block = Eigen::Map<const Eigen::Matrix<double, -1, -1>, 0, Eigen::OuterStride<>>;
using vector_of_a_tile      = Eigen::Matrix<double, Eigen::Dynamic, 1, Eigen::ColMajor, size_of_the_tiles, 1>;


#ifdef BAYESMVP_CHOLESKY_FACTOR_AND_INVERSE_WITH_LAPACK_FROM_OPENBLAS
// ---- OpenBLAS's Fortran routines (the trailing arguments are the hidden lengths of the character arguments, as
//      gfortran passes them), looked up once per process (a thread-safe static), one thread per call
struct routines_of_openblas {
    void (*dpotrf)(const char*, const int*, double*, const int*, int*, std::size_t) = nullptr;
    void (*dtrtri)(const char*, const char*, const int*, double*, const int*, int*, std::size_t,
                   std::size_t) = nullptr;
    void (*dlauum)(const char*, const int*, double*, const int*, int*, std::size_t) = nullptr;
    void (*dtrsm)(const char*, const char*, const char*, const char*, const int*, const int*, const double*,
                  const double*, const int*, double*, const int*, std::size_t, std::size_t, std::size_t,
                  std::size_t) = nullptr;
    void (*dtrmm)(const char*, const char*, const char*, const char*, const int*, const int*, const double*,
                  const double*, const int*, double*, const int*, std::size_t, std::size_t, std::size_t,
                  std::size_t) = nullptr;
    void (*dgemm)(const char*, const char*, const int*, const int*, const int*, const double*, const double*,
                  const int*, const double*, const int*, const double*, double*, const int*, std::size_t,
                  std::size_t) = nullptr;
    void (*dsyrk)(const char*, const char*, const int*, const int*, const double*, const double*, const int*,
                  const double*, double*, const int*, std::size_t, std::size_t) = nullptr;
};

template <typename Routine>
inline void look_up_routine(void* handle, const char* name, Routine& routine) {

    routine = reinterpret_cast<Routine>(dlsym(handle, name));
    if (routine == nullptr) {
        throw std::runtime_error(std::string("BayesMVP: ") + name + " not found in "
                                 + BAYESMVP_OPENBLAS_LIBRARY_FILE);
    }

}

inline const routines_of_openblas& openblas() {

    static const routines_of_openblas routines = [] {
        routines_of_openblas looked_up;
        setenv("OPENBLAS_NUM_THREADS", "1", 0);
        void* handle = dlopen(BAYESMVP_OPENBLAS_LIBRARY_FILE, RTLD_NOW | RTLD_LOCAL | RTLD_DEEPBIND);
        if (handle == nullptr) {
            throw std::runtime_error(std::string("BayesMVP: OpenBLAS not opened: ") + dlerror());
        }
        auto set_number_of_threads = reinterpret_cast<void (*)(int)>(dlsym(handle, "openblas_set_num_threads"));
        if (set_number_of_threads != nullptr) set_number_of_threads(1);
        look_up_routine(handle, "dpotrf_", looked_up.dpotrf);
        look_up_routine(handle, "dtrtri_", looked_up.dtrtri);
        look_up_routine(handle, "dlauum_", looked_up.dlauum);
        look_up_routine(handle, "dtrsm_", looked_up.dtrsm);
        look_up_routine(handle, "dtrmm_", looked_up.dtrmm);
        look_up_routine(handle, "dgemm_", looked_up.dgemm);
        look_up_routine(handle, "dsyrk_", looked_up.dsyrk);
        return looked_up;
    }();
    return routines;

}
#endif


// ---- the Eigen blocks of the inverse of a lower triangular matrix and of the lower triangle of M' M (the
//      kernels of the tiles without OpenBLAS); in place over the lower triangle, the strict upper triangle
//      neither read nor written
//      (the inverse, column by column from the last; LAPACK's trti2, lower, non-unit diagonal:
//       inverse(L)_jj = 1 / L_jj,  inverse(L)_{j+1:n, j} = -inverse(L)_{j+1:n, j+1:n} L_{j+1:n, j} / L_jj)
inline void inverse_of_lower_triangular_matrix_column_by_column(Eigen::Ref<Eigen::Matrix<double, -1, -1>> A) {

    const int n = static_cast<int>(A.rows());
    for (int j = n - 1; j >= 0; --j) {
        A(j, j) = 1.0 / A(j, j);
        const int m = n - 1 - j;
        if (m > 0) {
            const vector_of_a_tile column
                = A.block(j + 1, j + 1, m, m).triangularView<Eigen::Lower>() * A.col(j).tail(m);
            A.col(j).tail(m) = -A(j, j) * column;
        }
    }

}

//      (the lower triangle of M' M, row by row; LAPACK's lauu2, lower:
//       (M' M)_ij = M_ii M_ij + sum_{k > i} M_ki M_kj  for j <= i)
inline void lower_triangle_of_transpose_times_itself_row_by_row(Eigen::Ref<Eigen::Matrix<double, -1, -1>> A) {

    const int n = static_cast<int>(A.rows());
    for (int i = 0; i < n; ++i) {
        const double entry_on_the_diagonal = A(i, i);
        const int m = n - 1 - i;
        if (m > 0) {
            A(i, i) = A.col(i).tail(n - i).squaredNorm();
            const vector_of_a_tile sums_below_row_i = A.block(i + 1, 0, m, i).transpose() * A.col(i).tail(m);
            A.row(i).head(i) = entry_on_the_diagonal * A.row(i).head(i) + sums_below_row_i.transpose();
        } else {
            A.row(i).head(i + 1) *= entry_on_the_diagonal;
        }
    }

}


// ---- the kernels of the tiles (column-major blocks, each with its own leading dimension)
//      (the lower Cholesky factor of an n x n block, in place; 0 on success)
inline int factor_lower_cholesky_of_block(double* a, const int n, const int lda) {

#ifdef BAYESMVP_CHOLESKY_FACTOR_AND_INVERSE_WITH_LAPACK_FROM_OPENBLAS
    int info = 0;
    openblas().dpotrf("L", &n, a, &lda, &info, 1);
    return info;
#else
    view_of_a_block A(a, n, n, Eigen::OuterStride<>(lda));
    return Eigen::internal::llt_inplace<double, Eigen::Lower>::blocked(A) >= 0 ? 1 : 0;
#endif

}

//      (the inverse of a lower triangular n x n block, in place; 0 on success)
inline int invert_lower_triangular_block(double* a, const int n, const int lda) {

#ifdef BAYESMVP_CHOLESKY_FACTOR_AND_INVERSE_WITH_LAPACK_FROM_OPENBLAS
    int info = 0;
    openblas().dtrtri("L", "N", &n, a, &lda, &info, 1, 1);
    return info;
#else
    view_of_a_block A(a, n, n, Eigen::OuterStride<>(lda));
    inverse_of_lower_triangular_matrix_column_by_column(A);
    return 0;
#endif

}

//      (the lower triangle of M' M for a lower triangular n x n block M, in place; 0 on success)
inline int lower_triangle_of_transpose_times_itself_of_block(double* a, const int n, const int lda) {

#ifdef BAYESMVP_CHOLESKY_FACTOR_AND_INVERSE_WITH_LAPACK_FROM_OPENBLAS
    int info = 0;
    openblas().dlauum("L", &n, a, &lda, &info, 1);
    return info;
#else
    view_of_a_block A(a, n, n, Eigen::OuterStride<>(lda));
    lower_triangle_of_transpose_times_itself_row_by_row(A);
    return 0;
#endif

}

//      (B <- B inverse(A)', A lower triangular n_a x n_a, B rows_b x n_a)
inline void solve_right_with_transpose_of_lower_triangular(const double* a, const int n_a, const int lda,
                                                           double* b, const int rows_b, const int ldb) {

#ifdef BAYESMVP_CHOLESKY_FACTOR_AND_INVERSE_WITH_LAPACK_FROM_OPENBLAS
    const double one = 1.0;
    openblas().dtrsm("R", "L", "T", "N", &rows_b, &n_a, &one, a, &lda, b, &ldb, 1, 1, 1, 1);
#else
    const const_view_of_a_block A(a, n_a, n_a, Eigen::OuterStride<>(lda));
    view_of_a_block B(b, rows_b, n_a, Eigen::OuterStride<>(ldb));
    // A.triangularView<Eigen::Lower>().transpose().solveInPlace<Eigen::OnTheRight>(B);
    //// (the transpose of the lower triangle of A is the upper triangle of A': the same solve, written so that
    ////  Eigen takes A, a read-only view, as read-only)
    A.transpose().triangularView<Eigen::Upper>().solveInPlace<Eigen::OnTheRight>(B);
#endif

}

//      (B <- factor B inverse(A), A lower triangular n_a x n_a, B rows_b x n_a)
inline void solve_right_with_lower_triangular_times_factor(const double factor, const double* a, const int n_a,
                                                           const int lda, double* b, const int rows_b,
                                                           const int ldb) {

#ifdef BAYESMVP_CHOLESKY_FACTOR_AND_INVERSE_WITH_LAPACK_FROM_OPENBLAS
    openblas().dtrsm("R", "L", "N", "N", &rows_b, &n_a, &factor, a, &lda, b, &ldb, 1, 1, 1, 1);
#else
    const const_view_of_a_block A(a, n_a, n_a, Eigen::OuterStride<>(lda));
    view_of_a_block B(b, rows_b, n_a, Eigen::OuterStride<>(ldb));
    A.triangularView<Eigen::Lower>().solveInPlace<Eigen::OnTheRight>(B);
    B *= factor;
#endif

}

//      (B <- A B, or with a_transposed B <- A' B; A lower triangular n_a x n_a, B n_a x cols_b)
inline void multiply_left_by_lower_triangular(const bool a_transposed, const double* a, const int n_a,
                                              const int lda, double* b, const int cols_b, const int ldb) {

#ifdef BAYESMVP_CHOLESKY_FACTOR_AND_INVERSE_WITH_LAPACK_FROM_OPENBLAS
    const double one = 1.0;
    openblas().dtrmm("L", "L", a_transposed ? "T" : "N", "N", &n_a, &cols_b, &one, a, &lda, b, &ldb,
                     1, 1, 1, 1);
#else
    const const_view_of_a_block A(a, n_a, n_a, Eigen::OuterStride<>(lda));
    view_of_a_block B(b, n_a, cols_b, Eigen::OuterStride<>(ldb));
    Eigen::Matrix<double, -1, -1> product(n_a, cols_b);
    if (a_transposed) {
        // product = A.triangularView<Eigen::Lower>().transpose() * B;
        product = A.transpose().triangularView<Eigen::Upper>() * B;
    } else {
        product = A.triangularView<Eigen::Lower>() * B;
    }
    B = product;
#endif

}

//      (C <- C + factor op(A) op(B): op(A) m x k, op(B) k x n; op = the transpose when its flag is set)
inline void add_product(const double factor, const bool a_transposed, const bool b_transposed, const double* a,
                        const int lda, const double* b, const int ldb, const int m, const int n, const int k,
                        double* c, const int ldc) {

#ifdef BAYESMVP_CHOLESKY_FACTOR_AND_INVERSE_WITH_LAPACK_FROM_OPENBLAS
    const double one = 1.0;
    openblas().dgemm(a_transposed ? "T" : "N", b_transposed ? "T" : "N", &m, &n, &k, &factor, a, &lda, b, &ldb,
                     &one, c, &ldc, 1, 1);
#else
    view_of_a_block C(c, m, n, Eigen::OuterStride<>(ldc));
    const const_view_of_a_block A(a, a_transposed ? k : m, a_transposed ? m : k, Eigen::OuterStride<>(lda));
    const_view_of_a_block B(b, b_transposed ? n : k, b_transposed ? k : n, Eigen::OuterStride<>(ldb));
    if (a_transposed && b_transposed) {
        C.noalias() += factor * (A.transpose() * B.transpose());
    } else if (a_transposed) {
        C.noalias() += factor * (A.transpose() * B);
    } else if (b_transposed) {
        C.noalias() += factor * (A * B.transpose());
    } else {
        C.noalias() += factor * (A * B);
    }
#endif

}

//      (the lower triangle of C <- C + factor op(A) op(A)': op(A) n x k; op = the transpose when a_transposed)
inline void add_rank_k_update_of_lower_triangle(const double factor, const bool a_transposed, const double* a,
                                                const int lda, const int n, const int k, double* c,
                                                const int ldc) {

#ifdef BAYESMVP_CHOLESKY_FACTOR_AND_INVERSE_WITH_LAPACK_FROM_OPENBLAS
    const double one = 1.0;
    openblas().dsyrk("L", a_transposed ? "T" : "N", &n, &k, &factor, a, &lda, &one, c, &ldc, 1, 1);
#else
    view_of_a_block C(c, n, n, Eigen::OuterStride<>(ldc));
    const const_view_of_a_block A(a, a_transposed ? k : n, a_transposed ? n : k, Eigen::OuterStride<>(lda));
    if (a_transposed) {
        C.selfadjointView<Eigen::Lower>().rankUpdate(A.transpose(), factor);
    } else {
        C.selfadjointView<Eigen::Lower>().rankUpdate(A, factor);
    }
#endif

}


// ---- the independent tiles of one step: on the chain's own WCP threads (its own TBB arena of n_threads_WCP
//      threads, NICOSTAN_PER_CHAIN_TBB_ARENA=1, more than one thread), otherwise one after the other
inline bool the_chain_has_its_own_TBB_arena() {

    static const bool own_arena = [] {
        const char* value = std::getenv("NICOSTAN_PER_CHAIN_TBB_ARENA");
        return (value != nullptr) && (std::string(value) == "1");
    }();
    return own_arena;

}

template <typename Task>
inline void for_each_tile(const int n_tiles, const Task& task_of_a_tile) {

    if (n_tiles > 1 && the_chain_has_its_own_TBB_arena() && tbb::this_task_arena::max_concurrency() > 1) {
        tbb::parallel_for(tbb::blocked_range<int>(0, n_tiles, 1),
                          [&task_of_a_tile](const tbb::blocked_range<int>& range) {

                              for (int tile = range.begin(); tile < range.end(); ++tile) task_of_a_tile(tile);

                          },
                          tbb::simple_partitioner());
    } else {
        for (int tile = 0; tile < n_tiles; ++tile) task_of_a_tile(tile);
    }

}

inline void throw_not_positive_definite() {

    stan::math::throw_domain_error("cholesky_decompose", "Matrix", " is not positive definite", "m");

}


// ---- the lower Cholesky factor L of the n x n matrix a (column-major, leading dimension n), in place over its
//      lower triangle, in tiles: for each column panel k, its diagonal tile, the tiles below it
//      (L_ik = A_ik inverse(L_kk)'), then each trailing column panel j (its diagonal tile by a rank-k update,
//      the rows below it by a product: A(j:, j) -= L(j:, k) L(j, k)')
inline void lower_cholesky_factor_in_tiles(double* a, const int n) {

    const int b = size_of_the_tiles;
    const int n_panels = (n + b - 1) / b;
    for (int k = 0; k < n_panels; ++k) {
        const int first_column = k * b;
        const int columns = std::min(b, n - first_column);
        double* diagonal_tile = a + first_column + static_cast<std::ptrdiff_t>(first_column) * n;
        if (factor_lower_cholesky_of_block(diagonal_tile, columns, n) != 0) throw_not_positive_definite();
        const_view_of_a_block factor_of_the_tile(diagonal_tile, columns, columns, Eigen::OuterStride<>(n));
        if (!(factor_of_the_tile.diagonal().array() > 0.0).all()) throw_not_positive_definite();
        const int first_row_below = first_column + columns;
        const int n_tiles_below = n_panels - k - 1;
        if (n_tiles_below == 0) break;
        for_each_tile(n_tiles_below, [&](const int tile) {

            const int first_row = first_row_below + tile * b;
            const int rows = std::min(b, n - first_row);
            double* tile_below = a + first_row + static_cast<std::ptrdiff_t>(first_column) * n;
            solve_right_with_transpose_of_lower_triangular(diagonal_tile, columns, n, tile_below, rows, n);

        });
        for_each_tile(n_tiles_below, [&](const int tile) {

            const int first_column_of_panel = first_row_below + tile * b;
            const int columns_of_panel = std::min(b, n - first_column_of_panel);
            const double* rows_of_panel_in_panel_k
                = a + first_column_of_panel + static_cast<std::ptrdiff_t>(first_column) * n;
            double* diagonal_tile_of_panel
                = a + first_column_of_panel + static_cast<std::ptrdiff_t>(first_column_of_panel) * n;
            add_rank_k_update_of_lower_triangle(-1.0, false, rows_of_panel_in_panel_k, n, columns_of_panel,
                                                columns, diagonal_tile_of_panel, n);
            const int first_row_after = first_column_of_panel + columns_of_panel;
            const int rows_after = n - first_row_after;
            if (rows_after > 0) {
                const double* rows_after_in_panel_k
                    = a + first_row_after + static_cast<std::ptrdiff_t>(first_column) * n;
                double* rows_after_in_panel
                    = a + first_row_after + static_cast<std::ptrdiff_t>(first_column_of_panel) * n;
                add_product(-1.0, false, true, rows_after_in_panel_k, n, rows_of_panel_in_panel_k, n, rows_after,
                            columns_of_panel, columns, rows_after_in_panel, n);
            }

        });
    }

}


// ---- the inverse of the lower triangular n x n matrix a (leading dimension n), in place over its lower
//      triangle, in tiles: from the last column panel j to the first, with the trailing triangle R already
//      inverted, inverse(L)_Rj = -inverse(L_RR) L_Rj inverse(L_jj): each row tile i of the panel (from a copy Q
//      of the panel) P_i = inverse(L)_ii Q_i + inverse(L)(i, R before i) Q(R before i), then P_i = -P_i
//      inverse(L_jj) (with L_jj not yet inverted); then the diagonal tile L_jj itself
inline void inverse_of_lower_triangular_matrix_in_tiles(double* a, const int n) {

    const int b = size_of_the_tiles;
    const int n_panels = (n + b - 1) / b;
    std::vector<double> copy_of_the_panel;
    for (int j = n_panels - 1; j >= 0; --j) {
        const int first_column = j * b;
        const int columns = std::min(b, n - first_column);
        double* diagonal_tile = a + first_column + static_cast<std::ptrdiff_t>(first_column) * n;
        const int first_row_below = first_column + columns;
        const int rows_below = n - first_row_below;
        if (rows_below > 0) {
            const double* panel = a + first_row_below + static_cast<std::ptrdiff_t>(first_column) * n;
            copy_of_the_panel.resize(static_cast<std::size_t>(rows_below) * columns);
            view_of_a_block(copy_of_the_panel.data(), rows_below, columns, Eigen::OuterStride<>(rows_below))
                = const_view_of_a_block(panel, rows_below, columns, Eigen::OuterStride<>(n));
            const double* Q = copy_of_the_panel.data();
            const int n_tiles_below = n_panels - j - 1;
            for_each_tile(n_tiles_below, [&](const int tile) {

                const int first_row = first_row_below + tile * b;
                const int rows = std::min(b, n - first_row);
                const int rows_before = first_row - first_row_below;
                double* P_i = a + first_row + static_cast<std::ptrdiff_t>(first_column) * n;
                const double* inverse_of_diagonal_tile_i
                    = a + first_row + static_cast<std::ptrdiff_t>(first_row) * n;
                view_of_a_block(P_i, rows, columns, Eigen::OuterStride<>(n))
                    = const_view_of_a_block(Q + rows_before, rows, columns, Eigen::OuterStride<>(rows_below));
                multiply_left_by_lower_triangular(false, inverse_of_diagonal_tile_i, rows, n, P_i, columns, n);
                if (rows_before > 0) {
                    const double* inverse_before_tile_i
                        = a + first_row + static_cast<std::ptrdiff_t>(first_row_below) * n;
                    add_product(1.0, false, false, inverse_before_tile_i, n, Q, rows_below, rows, columns,
                                rows_before, P_i, n);
                }
                solve_right_with_lower_triangular_times_factor(-1.0, diagonal_tile, columns, n, P_i, rows, n);

            });
        }
        if (invert_lower_triangular_block(diagonal_tile, columns, n) != 0) throw_not_positive_definite();
    }

}


// ---- one column segment of a tile, in SIMD: with W_r = alpha_rows[r] alpha_of_column - inverse_K[r] and the
//      kernel values K_r, adds sum_r W_r K_r and sum_r W_r K_r (x_rows[r] - x_of_column)^2
inline void add_sums_of_W_times_kernel_values_of_a_column_segment(
        const double* kernel_values, const double* inverse_K, const double* alpha_rows,
        const double alpha_of_column, const double* x_rows, const double x_of_column, const int count,
        double& sum_of_products, double& sum_of_products_times_squared_distances) {

#if BMVP_LANES == 8
    ALIGN64  __m512d sums_of_products = _mm512_setzero_pd();
    ALIGN64  __m512d sums_of_products_times_squared_distances = _mm512_setzero_pd();
    ALIGN64  __m512d const alpha_column = _mm512_set1_pd(alpha_of_column);
    ALIGN64  __m512d const x_column = _mm512_set1_pd(x_of_column);
    for (int r = 0; r < count; r += 8) {
        const int remaining = count - r;
        const __mmask8 lanes = remaining >= 8 ? static_cast<__mmask8>(0xFF)
                                              : static_cast<__mmask8>((1u << remaining) - 1u);
        ALIGN64  __m512d const W = _mm512_fmsub_pd(_mm512_maskz_loadu_pd(lanes, alpha_rows + r), alpha_column,
                                          _mm512_maskz_loadu_pd(lanes, inverse_K + r));
        ALIGN64  __m512d const product = _mm512_mul_pd(W, _mm512_maskz_loadu_pd(lanes, kernel_values + r));
        ALIGN64  __m512d const distance =
                _mm512_maskz_sub_pd(lanes, _mm512_maskz_loadu_pd(lanes, x_rows + r), x_column);
        sums_of_products = _mm512_add_pd(sums_of_products, product);
        sums_of_products_times_squared_distances
            = _mm512_fmadd_pd(_mm512_mul_pd(product, distance), distance,
                              sums_of_products_times_squared_distances);
    }
    sum_of_products += _mm512_reduce_add_pd(sums_of_products);
    sum_of_products_times_squared_distances += _mm512_reduce_add_pd(sums_of_products_times_squared_distances);
#else
    // (the lanes of a partial packet: the first count of the 4 lanes, as a load mask)
    alignas(32) static const long long lane_masks[8] = {-1, -1, -1, -1, 0, 0, 0, 0};
    ALIGN32  __m256d sums_of_products = _mm256_setzero_pd();
    ALIGN32  __m256d sums_of_products_times_squared_distances = _mm256_setzero_pd();
    ALIGN32  __m256d const alpha_column = _mm256_set1_pd(alpha_of_column);
    ALIGN32  __m256d const x_column = _mm256_set1_pd(x_of_column);
    for (int r = 0; r < count; r += 4) {
        const int remaining = count - r;
        ALIGN32  __m256i const lanes = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                                    lane_masks + (remaining >= 4 ? 0 : 4 - remaining)));
        ALIGN32  __m256d const W = _mm256_fmsub_pd(_mm256_maskload_pd(alpha_rows + r, lanes), alpha_column,
                                          _mm256_maskload_pd(inverse_K + r, lanes));
        ALIGN32  __m256d const product = _mm256_mul_pd(W, _mm256_maskload_pd(kernel_values + r, lanes));
        ALIGN32  __m256d const distance =
                _mm256_and_pd(_mm256_sub_pd(_mm256_maskload_pd(x_rows + r, lanes), x_column),
                                               _mm256_castsi256_pd(lanes));
        sums_of_products = _mm256_add_pd(sums_of_products, product);
        sums_of_products_times_squared_distances
            = _mm256_fmadd_pd(_mm256_mul_pd(product, distance), distance,
                              sums_of_products_times_squared_distances);
    }
    alignas(32) double lanes_of_sums[4];
    _mm256_store_pd(lanes_of_sums, sums_of_products);
    sum_of_products += (lanes_of_sums[0] + lanes_of_sums[1]) + (lanes_of_sums[2] + lanes_of_sums[3]);
    _mm256_store_pd(lanes_of_sums, sums_of_products_times_squared_distances);
    sum_of_products_times_squared_distances += (lanes_of_sums[0] + lanes_of_sums[1])
                                               + (lanes_of_sums[2] + lanes_of_sums[3]);
#endif

}


// ---- from M = inverse(L) (lower, leading dimension n): over the lower triangle of W = alpha alpha' -
//      inverse(K), the sums sum_{i > j} W_ij K_ij, sum_{i > j} W_ij K_ij (x_i - x_j)^2 and sum_i W_ii, tile by
//      tile: each tile of inverse(K) = M' M made into a tile buffer (in L2), from the rows of the tile's row
//      panel on only (inverse(K)_IJ = M_II' M_IJ + M(after I, I)' M(after I, J); a diagonal tile: the lower
//      triangle of M_II' M_II + M(after I, I)' M(after I, I)), then the SIMD pass over its column segments with
//      the kernel values recomputed; the sums of the tiles added in the order of the tiles (the same result
//      whatever the number of threads)
struct sums_over_the_lower_triangle_of_W {
    double sum_of_products_below_the_diagonal = 0.0;
    double sum_of_products_times_squared_distances_below_the_diagonal = 0.0;
    double sum_of_the_diagonal = 0.0;
};

inline sums_over_the_lower_triangle_of_W sums_over_the_tiles_of_W(const double* M, const int n,
                                                                  const double* alpha, const double* x,
                                                                  const double sigma, const double length_scale) {

    const int b = size_of_the_tiles;
    const int n_panels = (n + b - 1) / b;
    const double squared_sigma = sigma * sigma;
    const double minus_half_inverse_squared_length_scale = -0.5 / (length_scale * length_scale);
    std::vector<int> row_panel_of_each_tile;
    std::vector<int> column_panel_of_each_tile;
    for (int column_panel = 0; column_panel < n_panels; ++column_panel) {
        for (int row_panel = column_panel; row_panel < n_panels; ++row_panel) {
            row_panel_of_each_tile.push_back(row_panel);
            column_panel_of_each_tile.push_back(column_panel);
        }
    }
    const int n_tiles = static_cast<int>(row_panel_of_each_tile.size());
    std::vector<sums_over_the_lower_triangle_of_W> sums_of_each_tile(n_tiles);
    for_each_tile(n_tiles, [&](const int tile) {

        const int first_row = row_panel_of_each_tile[tile] * b;
        const int rows = std::min(b, n - first_row);
        const int first_column = column_panel_of_each_tile[tile] * b;
        const int columns = std::min(b, n - first_column);
        const bool on_the_diagonal = first_row == first_column;
        const int first_row_after = first_row + rows;
        const int rows_after = n - first_row_after;
        thread_local std::vector<double> buffer_of_the_tile;
        buffer_of_the_tile.resize(static_cast<std::size_t>(b) * b);
        double* inverse_K = buffer_of_the_tile.data();
        view_of_a_block tile_of_inverse_K(inverse_K, rows, columns, Eigen::OuterStride<>(rows));
        const double* M_II = M + first_row + static_cast<std::ptrdiff_t>(first_row) * n;
        const double* M_after_I_I = M + first_row_after + static_cast<std::ptrdiff_t>(first_row) * n;
        if (on_the_diagonal) {
            tile_of_inverse_K.triangularView<Eigen::Lower>()
                = const_view_of_a_block(M_II, rows, rows, Eigen::OuterStride<>(n));
            if (lower_triangle_of_transpose_times_itself_of_block(inverse_K, rows, rows) != 0) {
                throw_not_positive_definite();
            }
            if (rows_after > 0) {
                add_rank_k_update_of_lower_triangle(1.0, true, M_after_I_I, n, rows, rows_after, inverse_K, rows);
            }
        } else {
            const double* M_IJ = M + first_row + static_cast<std::ptrdiff_t>(first_column) * n;
            const double* M_after_I_J = M + first_row_after + static_cast<std::ptrdiff_t>(first_column) * n;
            tile_of_inverse_K = const_view_of_a_block(M_IJ, rows, columns, Eigen::OuterStride<>(n));
            multiply_left_by_lower_triangular(true, M_II, rows, n, inverse_K, columns, rows);
            if (rows_after > 0) {
                add_product(1.0, true, false, M_after_I_I, n, M_after_I_J, n, rows, columns, rows_after,
                            inverse_K, rows);
            }
        }
        sums_over_the_lower_triangle_of_W sums;
        alignas(64) double kernel_values[size_of_the_tiles];
        for (int column = 0; column < columns; ++column) {
            const int column_of_K = first_column + column;
            const int first_row_in_the_tile = on_the_diagonal ? column + 1 : 0;
            const int count = rows - first_row_in_the_tile;
            const double* column_of_inverse_K = inverse_K + static_cast<std::ptrdiff_t>(column) * rows;
            if (on_the_diagonal) {
                sums.sum_of_the_diagonal += alpha[column_of_K] * alpha[column_of_K] - column_of_inverse_K[column];
            }
            if (count > 0) {
                const int first_row_of_K = first_row + first_row_in_the_tile;
                nico::math::gp_exp_quad_cov::column_segment(x + first_row_of_K, count, x[column_of_K],
                                                               minus_half_inverse_squared_length_scale,
                                                               squared_sigma, kernel_values);
                add_sums_of_W_times_kernel_values_of_a_column_segment(
                        kernel_values, column_of_inverse_K + first_row_in_the_tile, alpha + first_row_of_K,
                        alpha[column_of_K], x + first_row_of_K, x[column_of_K], count,
                        sums.sum_of_products_below_the_diagonal,
                        sums.sum_of_products_times_squared_distances_below_the_diagonal);
            }
        }
        sums_of_each_tile[tile] = sums;

    });
    sums_over_the_lower_triangle_of_W sums;
    for (int tile = 0; tile < n_tiles; ++tile) {
        sums.sum_of_products_below_the_diagonal += sums_of_each_tile[tile].sum_of_products_below_the_diagonal;
        sums.sum_of_products_times_squared_distances_below_the_diagonal
            += sums_of_each_tile[tile].sum_of_products_times_squared_distances_below_the_diagonal;
        sums.sum_of_the_diagonal += sums_of_each_tile[tile].sum_of_the_diagonal;
    }
    return sums;

}


// ---- the lower triangle and the diagonal of K = gp_exp_quad_cov(x, sigma, length_scale) + diagonal I, into
//      column-major n x n storage (its strict upper triangle is not written)
inline void lower_triangle_of_covariance_plus_diagonal(const std::vector<double>& x, const double sigma,
                                                       const double length_scale, const double diagonal,
                                                       double* values) {

    const int n = static_cast<int>(x.size());
    const double squared_sigma = sigma * sigma;
    const double minus_half_inverse_squared_length_scale = -0.5 / (length_scale * length_scale);
    for (int j = 0; j < n; ++j) {
        double* column = values + static_cast<std::ptrdiff_t>(j) * n;
        nico::math::gp_exp_quad_cov::column_segment(x.data() + j + 1, n - j - 1, x[j],
                                                       minus_half_inverse_squared_length_scale, squared_sigma,
                                                       column + j + 1);
        // (add_diag's single addition, with its rounding: the diagonal term is not combined with sigma^2
        //  beforehand)
        column[j] = squared_sigma;
        column[j] += diagonal;
    }

}

// ---- from the lower Cholesky factor L (leading dimension n): z = L \ residual (in place), its sum of squares,
//      and the sum of the log of the diagonal of L (BayesMVP's log kernel); with alpha_wanted, the residual then
//      becomes alpha = L' \ z (in place)
inline void solves_and_log_determinant_from_the_factor(const double* factor, const int n,
                                                       Eigen::Ref<Eigen::Matrix<double, -1, 1>> residual,
                                                       const bool alpha_wanted, double& sum_of_squares_of_z,
                                                       double& sum_of_log_of_diagonal_of_factor) {

    const Eigen::Map<const Eigen::Matrix<double, -1, -1>> L(factor, n, n);
    L.triangularView<Eigen::Lower>().solveInPlace(residual);
    sum_of_squares_of_z = residual.squaredNorm();
    Eigen::Matrix<double, -1, 1> log_of_diagonal = L.diagonal();
    k_log(log_of_diagonal.data(), n);
    sum_of_log_of_diagonal_of_factor = log_of_diagonal.sum();
    if (alpha_wanted) L.triangularView<Eigen::Lower>().transpose().solveInPlace(residual);

}

}  // namespace gp_exp_quad_cov_normal_marginal
}  // namespace nico::math


// ---- the Stan-facing function (global scope; stanc calls it as
//      nico_gp_exp_quad_cov_normal_marginal_lpdf<propto__>(y, mu, x, sigma, length_scale, diagonal, pstream__))
namespace nico::math {

template <bool propto__, typename T_y, typename T_mu, typename T_x, typename T_sigma, typename T_length_scale,
          typename T_diagonal, stan::require_eigen_col_vector_t<T_y>* = nullptr,
          stan::require_all_stan_scalar_t<T_sigma, T_length_scale, T_diagonal>* = nullptr>
inline stan::return_type_t<T_y, T_mu, T_sigma, T_length_scale, T_diagonal>
nico_gp_exp_quad_cov_normal_marginal_lpdf(const T_y& y, const T_mu& mu, const std::vector<T_x>& x,
                                          const T_sigma& sigma, const T_length_scale& length_scale,
                                          const T_diagonal& diagonal, std::ostream* /*pstream__*/) {

    static_assert(std::is_arithmetic<T_x>::value, "nico_gp_exp_quad_cov_normal_marginal_lpdf: x must be data");
    namespace gp    = nico::math::gp_exp_quad_cov;
    namespace fused = nico::math::gp_exp_quad_cov_normal_marginal;
    using stan::math::var;
    using stan::math::vari;
    using return_t = stan::return_type_t<T_y, T_mu, T_sigma, T_length_scale, T_diagonal>;
    constexpr bool mu_is_a_vector = stan::is_eigen<T_mu>::value;
    constexpr bool y_is_a_parameter = stan::is_var<stan::value_type_t<T_y>>::value;
    constexpr bool mu_is_a_vector_parameter = mu_is_a_vector && stan::is_var<stan::value_type_t<T_mu>>::value;
    constexpr bool covariance_has_a_parameter = stan::is_var<T_sigma>::value
                                                || stan::is_var<T_length_scale>::value
                                                || stan::is_var<T_diagonal>::value;
    constexpr bool include_constant = stan::math::include_summand<propto__>::value;
    constexpr bool include_log_determinant
        = stan::math::include_summand<propto__, T_sigma, T_length_scale, T_diagonal>::value;

    const int n = static_cast<int>(x.size());
    const auto& y_ref  = stan::math::to_ref(y);
    const auto& mu_ref = stan::math::to_ref(mu);
    stan::math::check_size_match(fused::function_name, "Size of random variable", y_ref.size(),
                                 "number of points", n);
    if constexpr (mu_is_a_vector) {
        stan::math::check_size_match(fused::function_name, "Size of location parameter", mu_ref.size(),
                                     "number of points", n);
    }
    const std::vector<double> x_values(x.begin(), x.end());
    gp::check_arguments(x_values, sigma, length_scale);
    stan::math::check_not_nan(fused::function_name, "Random variable", stan::math::value_of(y_ref));
    stan::math::check_finite(fused::function_name, "Location parameter", stan::math::value_of(mu_ref));
    stan::math::check_not_nan(fused::function_name, "Diagonal term", diagonal);
    if (!stan::math::include_summand<propto__, T_y, T_mu, T_sigma, T_length_scale, T_diagonal>::value) {
        return return_t(0.0);
    }
    if (n == 0) return return_t(0.0);
    const double sigma_value    = stan::math::value_of(sigma);
    const double length_value   = stan::math::value_of(length_scale);
    const double diagonal_value = stan::math::value_of(diagonal);

    if constexpr (!stan::is_var<return_t>::value) {
        // ---- every argument data: the value only, on the heap (nothing on the arena)
        Eigen::Matrix<double, -1, -1> factor(n, n);
        fused::lower_triangle_of_covariance_plus_diagonal(x_values, sigma_value, length_value, diagonal_value,
                                                          factor.data());
        fused::lower_cholesky_factor_in_tiles(factor.data(), n);
        Eigen::Matrix<double, -1, 1> residual(n);
        if constexpr (mu_is_a_vector) {
            residual = stan::math::value_of(y_ref) - stan::math::value_of(mu_ref);
        } else {
            residual = (stan::math::value_of(y_ref).array() - static_cast<double>(mu_ref)).matrix();
        }
        double sum_of_squares_of_z = 0.0;
        double sum_of_log_of_diagonal_of_factor = 0.0;
        fused::solves_and_log_determinant_from_the_factor(factor.data(), n, residual, false, sum_of_squares_of_z,
                                                          sum_of_log_of_diagonal_of_factor);
        double log_density = -0.5 * sum_of_squares_of_z;
        if (include_log_determinant) log_density -= sum_of_log_of_diagonal_of_factor;
        if (include_constant) log_density -= 0.5 * n * stan::math::LOG_TWO_PI;
        return log_density;
    } else {
        // ---- the one N x N matrix of the node (K, then L, then inverse(L)) and alpha, on the arena
        auto& arena = stan::math::ChainableStack::instance_->memalloc_;
        double* factor = arena.alloc_array<double>(static_cast<std::size_t>(n) * n);
        fused::lower_triangle_of_covariance_plus_diagonal(x_values, sigma_value, length_value, diagonal_value,
                                                          factor);
        fused::lower_cholesky_factor_in_tiles(factor, n);
        stan::arena_t<Eigen::Matrix<double, -1, 1>> alpha(n);
        if constexpr (mu_is_a_vector) {
            alpha = stan::math::value_of(y_ref) - stan::math::value_of(mu_ref);
        } else {
            alpha = (stan::math::value_of(y_ref).array()
                     - static_cast<double>(stan::math::value_of(mu_ref))).matrix();
        }
        double sum_of_squares_of_z = 0.0;
        double sum_of_log_of_diagonal_of_factor = 0.0;
        fused::solves_and_log_determinant_from_the_factor(factor, n, alpha, true, sum_of_squares_of_z,
                                                          sum_of_log_of_diagonal_of_factor);
        double log_density = -0.5 * sum_of_squares_of_z;
        if (include_log_determinant) log_density -= sum_of_log_of_diagonal_of_factor;
        if (include_constant) log_density -= 0.5 * n * stan::math::LOG_TWO_PI;

        double* x_on_arena = arena.alloc_array<double>(n);
        std::copy(x_values.begin(), x_values.end(), x_on_arena);
        vari* sigma_node    = nullptr;
        vari* length_node   = nullptr;
        vari* diagonal_node = nullptr;
        vari* mu_node       = nullptr;
        if constexpr (stan::is_var<T_sigma>::value)        sigma_node    = sigma.vi_;
        if constexpr (stan::is_var<T_length_scale>::value) length_node   = length_scale.vi_;
        if constexpr (stan::is_var<T_diagonal>::value)     diagonal_node = diagonal.vi_;
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

        return stan::math::make_callback_var(log_density, [factor, alpha, x_on_arena, n, sigma_node, length_node,
                                                           diagonal_node, mu_node, y_on_arena, mu_on_arena,
                                                           sigma_value, length_value](auto& result_node) mutable {
            const double adjoint = result_node.adj();
            if constexpr (covariance_has_a_parameter) {
                // ---- inverse(L) in place, then the sums over the tiles of W = alpha alpha' - inverse(K)
                fused::inverse_of_lower_triangular_matrix_in_tiles(factor, n);
                const fused::sums_over_the_lower_triangle_of_W sums
                    = fused::sums_over_the_tiles_of_W(factor, n, alpha.data(), x_on_arena, sigma_value,
                                                      length_value);
                if (sigma_node != nullptr) {
                    sigma_node->adj_ += adjoint * (2.0 * sums.sum_of_products_below_the_diagonal
                                                   + sigma_value * sigma_value * sums.sum_of_the_diagonal)
                                        / sigma_value;
                }
                if (length_node != nullptr) {
                    length_node->adj_ += adjoint * sums.sum_of_products_times_squared_distances_below_the_diagonal
                                         / (length_value * length_value * length_value);
                }
                if (diagonal_node != nullptr) diagonal_node->adj_ += adjoint * 0.5 * sums.sum_of_the_diagonal;
            }
            if constexpr (y_is_a_parameter) y_on_arena.adj() -= adjoint * alpha;
            if constexpr (mu_is_a_vector_parameter) mu_on_arena.adj() += adjoint * alpha;
            if (mu_node != nullptr) mu_node->adj_ += adjoint * alpha.sum();
        });
    }

}

}  // namespace nico::math

using nico::math::nico_gp_exp_quad_cov_normal_marginal_lpdf;























