#pragma once
//
// -| --------- BayesMVP Cholesky factor with Eigen's own factorisation core and Stan's reverse pass, exposed to -
//              Stan as an external function
//
// nico_cholesky_decompose_native_core(A): the same value and gradient as Stan's cholesky_decompose(A), for any
// symmetric positive definite matrix A (data, a matrix of vars or a var_value matrix). General: any model with
// a Cholesky factor of a matrix built in the model.
//
// The factorisation is Eigen's own blocked in-place LLT core (Eigen::internal::llt_inplace<double,
// Eigen::Lower>::
// blocked, the routine that Stan's Eigen::LLT calls), on one contiguous copy of the values; the only work left
// out is LLT::compute's L1 norm of the input, which Stan's cholesky_decompose never uses. The reverse pass is
// Stan's own (Giles' algorithm up to N = 35, Murray's blocked algorithm above), on the lower-triangle adjoint of
// the input, as Stan. With a matrix of vars, the N (N + 1) / 2 nodes of the lower triangle are placed in ONE
// arena array, and the upper triangle shares one zero node. No condition-number test, fallback, jitter or
// tolerance.
//
// Checks, as Stan: square, symmetric (Stan's tolerance), not NaN, positive definite (Stan's message).
//
// Declared in the Stan functions block (no body) as
//
//     matrix nico_cholesky_decompose_native_core(matrix A);
//
// ---------------------------------------------------------------------------------------------------------------

#include <stan/math/rev.hpp>

#include <cstddef>
#include <new>
#include <ostream>
#include <type_traits>

namespace nico::math {
namespace cholesky_decompose_native_core {

// ---- the lower factor in place, on contiguous column-major storage (also Stan's arena matrix); the failure
//      test and message of Stan's check_pos_definite on Eigen's LLT; the strict upper triangle set to zero
template <typename Matrix>
inline void factor_lower_in_place(Matrix& storage) {

    Eigen::Map<Eigen::Matrix<double, -1, -1>> matrix(storage.data(), storage.rows(), storage.cols());
    const Eigen::Index failure = Eigen::internal::llt_inplace<double, Eigen::Lower>::blocked(matrix);
    if (failure >= 0 || !(matrix.diagonal().array() > 0.0).all()) {
        stan::math::throw_domain_error("cholesky_decompose", "Matrix", " is not positive definite", "m");
    }
    matrix.template triangularView<Eigen::StrictlyUpper>().setZero();

}

// ---- Stan's reverse pass, with its small / large dispatch
template <typename Input, typename Lower, typename Values>
inline void register_reverse_pass_of_Stan(Input& input, Lower& lower, Values& values) {

    if (values.rows() <= 35) {
        stan::math::reverse_pass_callback(stan::math::internal::unblocked_cholesky_lambda(values, lower, input));
    } else {
        stan::math::reverse_pass_callback(stan::math::internal::cholesky_lambda(values, lower, input));
    }

}

}  // namespace cholesky_decompose_native_core
}  // namespace nico::math


// ---- the Stan-facing functions (global scope; stanc calls them as  nico_cholesky_decompose_native_core(A,
//      pstream__))
namespace nico::math {

template <typename EigMat, stan::require_eigen_t<EigMat>* = nullptr,
          stan::require_not_eigen_vt<stan::is_var, EigMat>* = nullptr>
inline auto nico_cholesky_decompose_native_core(const EigMat& input, std::ostream* /*pstream__*/) {

    if constexpr (!std::is_same<stan::value_type_t<EigMat>, double>::value) {
        return stan::math::cholesky_decompose(input);
    } else {
        const stan::eval_return_type_t<EigMat>& evaluated = input.eval();
        stan::math::check_symmetric("cholesky_decompose", "m", evaluated);
        stan::math::check_not_nan("cholesky_decompose", "m", evaluated);
        Eigen::Matrix<double, -1, -1> lower = evaluated;
        nico::math::cholesky_decompose_native_core::factor_lower_in_place(lower);
        using return_t = Eigen::Matrix<double, EigMat::RowsAtCompileTime, EigMat::ColsAtCompileTime>;
        return return_t(lower);
    }

}

}  // namespace nico::math

using nico::math::nico_cholesky_decompose_native_core;

namespace nico::math {

template <typename EigMat, stan::require_eigen_vt<stan::is_var, EigMat>* = nullptr>
inline auto nico_cholesky_decompose_native_core(const EigMat& input, std::ostream* /*pstream__*/) {

    using stan::math::var;
    using stan::math::vari;
    stan::math::check_square("cholesky_decompose", "A", input);
    stan::arena_t<EigMat> input_on_arena = input;
    stan::arena_t<Eigen::Matrix<double, -1, -1>> values(input_on_arena.val());
    stan::math::check_symmetric("cholesky_decompose", "A", input);
    nico::math::cholesky_decompose_native_core::factor_lower_in_place(values);

    const Eigen::Index size = values.rows();
    stan::arena_t<EigMat> lower(size, size);
    auto& arena = stan::math::ChainableStack::instance_->memalloc_;
    const std::size_t n = static_cast<std::size_t>(size);
    const std::size_t count = n * (n + 1) / 2;
    vari* entries = count == 0 ? nullptr : arena.alloc_array<vari>(count);
    vari* upper_zero = new vari(0.0, false);
    std::size_t position = 0;
    for (Eigen::Index j = 0; j < size; ++j) {
        for (Eigen::Index i = 0; i < j; ++i) lower(i, j) = var(upper_zero);
        for (Eigen::Index i = j; i < size; ++i, ++position) {
            ::new (static_cast<void*>(entries + position)) vari(values(i, j), false);   // nochain stack
            lower(i, j) = var(entries + position);
        }
    }
    nico::math::cholesky_decompose_native_core::register_reverse_pass_of_Stan(input_on_arena, lower, values);
    return stan::plain_type_t<EigMat>(lower);

}

}  // namespace nico::math

using nico::math::nico_cholesky_decompose_native_core;

namespace nico::math {

template <typename T, stan::require_var_matrix_t<T>* = nullptr>
inline auto nico_cholesky_decompose_native_core(const T& input, std::ostream* /*pstream__*/) {

    stan::math::check_symmetric("cholesky_decompose", "A", input.val());
    stan::plain_type_t<T> lower = nico_cholesky_decompose_native_core(input.val(), nullptr);
    auto values = lower.val();
    auto input_handle = input;
    nico::math::cholesky_decompose_native_core::register_reverse_pass_of_Stan(input_handle, lower, values);
    return lower;

}

}  // namespace nico::math

using nico::math::nico_cholesky_decompose_native_core;






















