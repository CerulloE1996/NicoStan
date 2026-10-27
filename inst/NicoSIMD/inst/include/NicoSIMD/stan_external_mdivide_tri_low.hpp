#pragma once
//
// Lower-triangular solves: x = A \ b and x = b / A, with A lower triangular (its upper part not read).
// The same values as Stan's mdivide_left_tri_low / mdivide_right_tri_low (the same Eigen triangular solves) and
// Stan's reverse pass for matrix<var> A and data b: adj_A -= lower(A^{-T} (adj_x x^T)).
//
// Stan declarations:
//     vector nico_mdivide_left_tri_low(matrix A, vector b);
//     matrix nico_mdivide_left_tri_low(matrix A, matrix b);
//     row_vector nico_mdivide_right_tri_low(row_vector b, matrix A);
//     matrix nico_mdivide_right_tri_low(matrix b, matrix A);
// Compile with --allow-undefined. Supported: double A and b; matrix<var> A with a double vector b.
// Every other scalar combination does not compile.

#include <stan/math/rev.hpp>

#include <ostream>
#include <type_traits>

namespace nico::math {

//// ---- x = A \ b, double A and b:
template <typename EigMat1,
          typename EigMat2,
          stan::require_all_eigen_vt<std::is_arithmetic, EigMat1, EigMat2>* = nullptr>
inline Eigen::Matrix<double, EigMat1::RowsAtCompileTime, EigMat2::ColsAtCompileTime>
nico_mdivide_left_tri_low(const EigMat1& A,
                          const EigMat2& b,
                          std::ostream* /*pstream__*/) {

    stan::math::check_square("mdivide_left_tri_low", "A", A);
    stan::math::check_multiplicable("mdivide_left_tri_low", "A", A, "b", b);
    if (A.rows() == 0) {
        return {0, b.cols()};
    }
    using matrix_t = Eigen::Matrix<double, -1, -1>;
    return matrix_t(A).template triangularView<Eigen::Lower>().solve(matrix_t(b)).eval();

}

}  // namespace nico::math

using nico::math::nico_mdivide_left_tri_low;

namespace nico::math {

//// ---- x = A \ b, matrix<var> A and a double vector b:
template <typename EigMat,
          typename EigVec,
          stan::require_eigen_vt<stan::is_var, EigMat>* = nullptr,
          stan::require_eigen_col_vector_vt<std::is_arithmetic, EigVec>* = nullptr>
inline Eigen::Matrix<stan::math::var, EigMat::RowsAtCompileTime, 1>
nico_mdivide_left_tri_low(const EigMat& A,
                          const EigVec& b,
                          std::ostream* /*pstream__*/) {

    using stan::math::var;
    using stan::math::vari;
    stan::math::check_square("mdivide_left_tri_low", "A", A);
    stan::math::check_multiplicable("mdivide_left_tri_low", "A", A, "b", b);
    if (A.rows() == 0) {
        return {0, 1};
    }

    //// ---- the values, as Stan's: b and x held as N x 1 matrices, so Eigen takes the same (matrix) solve:
    const Eigen::Index size = A.rows();
    stan::arena_t<Eigen::Matrix<var, -1, -1>> A_on_arena = A;
    stan::arena_t<Eigen::Matrix<double, -1, -1>> A_values = A_on_arena.val();
    const Eigen::Matrix<double, -1, -1> b_values = b;
    stan::arena_t<Eigen::Matrix<double, -1, -1>> x_values = A_values.template triangularView<Eigen::Lower>()
                                                                    .solve(b_values);

    stan::arena_t<Eigen::Matrix<var, -1, 1>> x(size);
    for (Eigen::Index i = 0; i < size; ++i) {
        x(i) = var(new vari(x_values(i, 0), false));
    }

    //// ---- the reverse pass, as Stan's for data b: adj_A = -A^{-T} (adj_x x^T), added to the lower entries:
    stan::math::reverse_pass_callback([A_on_arena, A_values, x_values, x]() mutable {

        const Eigen::Matrix<double, -1, -1> adj_x = x.adj();
        Eigen::Matrix<double, -1, -1> adj_A(A_values.rows(), A_values.rows());
        adj_A.noalias() = -A_values.template triangularView<Eigen::Lower>()
                                   .transpose()
                                   .solve(adj_x * x_values.transpose());
        const Eigen::Index n = A_values.rows();
        for (Eigen::Index j = 0; j < n; ++j) {
            for (Eigen::Index i = j; i < n; ++i) {
                A_on_arena(i, j).vi_->adj_ += adj_A(i, j);
            }
        }

    });

    return Eigen::Matrix<var, EigMat::RowsAtCompileTime, 1>(x);

}

}  // namespace nico::math

using nico::math::nico_mdivide_left_tri_low;

namespace nico::math {

//// ---- x = b / A, double b and A:
template <typename EigMat1,
          typename EigMat2,
          stan::require_all_eigen_vt<std::is_arithmetic, EigMat1, EigMat2>* = nullptr>
inline Eigen::Matrix<double, EigMat1::RowsAtCompileTime, EigMat2::ColsAtCompileTime>
nico_mdivide_right_tri_low(const EigMat1& b,
                           const EigMat2& A,
                           std::ostream* /*pstream__*/) {

    stan::math::check_square("mdivide_right_tri", "A", A);
    stan::math::check_multiplicable("mdivide_right_tri", "b", b, "A", A);
    using matrix_t = Eigen::Matrix<double, -1, -1>;
    if (A.rows() == 0) {
        return matrix_t(b.rows(), 0);
    }
    return matrix_t(A).template triangularView<Eigen::Lower>()
                      .transpose()
                      .solve(matrix_t(b).transpose())
                      .transpose()
                      .eval();

}

}  // namespace nico::math

using nico::math::nico_mdivide_right_tri_low;






















