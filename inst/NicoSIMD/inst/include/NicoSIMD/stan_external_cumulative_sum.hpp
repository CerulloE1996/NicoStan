#pragma once
//
// -| --------- nico_cumulative_sum, a Stan external function ----------------------------------------------------
//
//     nico_cumulative_sum(x)[i] = sum_{j <= i} x_j
//
// The values by NicoSIMD's prefix sum scan; the adjoints by its backward scan with every ratio 1:
//
//     adjoint of x_j += sum_{k >= j} adjoint of result_k
//
// as Stan's cumulative_sum. An integer array is summed exactly, in order.
//
// Declared in the Stan functions block (no body; stanc --allow-undefined):
//
//     vector nico_cumulative_sum(vector x);
//     array[] int nico_cumulative_sum(array[] int x);
//
// -----------------------------------------------------------------------------------------------------------

#include "stan_external_functions.hpp"
#include "stan_external_simd_log_cumulative_sum_exp_scans.hpp"

#include <stan/math/rev.hpp>
#include <stan/math/prim.hpp>

#include <Eigen/Dense>
#include <algorithm>
#include <numeric>
#include <ostream>
#include <vector>

namespace nico::math {

template <typename T_x,
          stan::require_eigen_col_vector_t<T_x>* = nullptr>
inline Eigen::Matrix<stan::value_type_t<T_x>, -1, 1> nico_cumulative_sum(const T_x& x,
                                                                        std::ostream* /*pstream__*/) {


    namespace lcse = nico::math::log_cumulative_sum_exp;
    using stan::arena_t;
    using stan::math::var;

    const auto& x_ref = stan::math::to_ref(x);
    const int M = static_cast<int>(x_ref.size());
    Eigen::Matrix<double, -1, 1> values = stan::math::value_of(x_ref);
    if (M > 0) lcse::nico_sum_prefix_scan(values.data(), M, 0.0);

    if constexpr (stan::is_constant_all<T_x>::value) {
        return values;
    } else {
        arena_t<Eigen::Matrix<var, -1, 1>> x_arena = x_ref;
        arena_t<Eigen::Matrix<var, -1, 1>> result = values;
        stan::math::reverse_pass_callback([x_arena, result, M]() mutable {

            if (M == 0) return;
            const Eigen::Matrix<double, -1, 1> weights = result.adj();
            const Eigen::Matrix<double, -1, 1> ratio = Eigen::Matrix<double, -1, 1>::Ones(std::max(M - 1, 1));
            Eigen::Matrix<double, -1, 1> running(M);
            lcse::nico_backward_prefix_scan(weights.data(), ratio.data(), M, running.data());
            x_arena.adj() += running;

        });
        return Eigen::Matrix<var, -1, 1>(result);
    }


}

inline std::vector<int> nico_cumulative_sum(const std::vector<int>& x,
                                            std::ostream* /*pstream__*/) {


    std::vector<int> result(x.size());
    std::partial_sum(x.begin(), x.end(), result.begin());
    return result;


}

}  // namespace nico::math

using nico::math::nico_cumulative_sum;
























