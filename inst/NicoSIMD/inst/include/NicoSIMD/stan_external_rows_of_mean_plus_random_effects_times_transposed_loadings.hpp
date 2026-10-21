#pragma once
//
// -| --------- NicoSIMD rows of a mean plus random effects times transposed loadings, exponentiated if asked ----
//
//     nico_rows_of_mean_plus_random_effects_times_transposed_loadings(z, mean, transposed_loadings,
//                                                                     exponentiate)
//
// z: an n x K matrix, one row per unit (e.g. patient, subject, group), of random effects (e.g. standard normal);
// mean: a row vector of size K; transposed_loadings: a K x K matrix, the transpose of the loadings L (e.g.
// L = diag(omega) L_Omega for correlated random effects with standard deviations omega and the Cholesky factor
// L_Omega of their correlation matrix, or any K x K linear map, e.g. one that also turns the K parameters into
// other K quantities); exponentiate: 1 or 0 (data). The result is the n x K matrix
//
//     row i = exp(mean + z[i, ] transposed_loadings)    (exponentiate = 1: lognormal unit-level parameters)
//     row i =     mean + z[i, ] transposed_loadings     (exponentiate = 0: normal unit-level parameters)
//
// i.e. exp(mu + L z_i) (or mu + L z_i) for every unit i, as column vectors: the unit-level parameters of any model
// with correlated normal or lognormal random effects through an affine map (population PK and other nonlinear
// mixed-effects models, correlated person parameters of IRT and growth-curve models, multilevel regressions with
// correlated intercepts and slopes).
//
// ONE autodiff step for the whole matrix (Stan's version: rep_matrix, the n x K by K x K product, to_vector, exp
// and the copies between them), the product and its vector-Jacobian product made as whole-matrix Eigen work (no
// loop over the units), the exponential with BayesMVP's exp kernel over all n K values at once. With g the
// adjoint of the result and w = g .* result (exponentiate = 1) or w = g (exponentiate = 0):
//
//     adjoint of z                   += w transposed_loadings'
//     adjoint of mean                += the column sums of w
//     adjoint of transposed_loadings += z' w
//
// Declared in the Stan functions block (no body; stanc --allow-undefined):
//
//     matrix nico_rows_of_mean_plus_random_effects_times_transposed_loadings(matrix z, row_vector mean,
//                                                                            matrix transposed_loadings,
//                                                                            data int exponentiate);
//
// Included after stan_external_functions.hpp of the same folder (its exp kernel, nico::math::k_exp).
//
// -----------------------------------------------------------------------------------------------------------

#include "stan_external_functions.hpp"

#include <stan/math/rev.hpp>
#include <stan/math/prim.hpp>

#include <Eigen/Dense>
#include <ostream>
#include <stdexcept>
#include <string>




// ---- the Stan-facing function (global scope; stanc calls it with pstream__ last)
namespace nico::math {

template <typename T_z, typename T_mean, typename T_transposed_loadings,
          stan::require_eigen_matrix_dynamic_t<T_z>* = nullptr,
          stan::require_eigen_row_vector_t<T_mean>* = nullptr,
          stan::require_eigen_matrix_dynamic_t<T_transposed_loadings>* = nullptr>
inline Eigen::Matrix<stan::return_type_t<T_z, T_mean, T_transposed_loadings>, Eigen::Dynamic, Eigen::Dynamic>
nico_rows_of_mean_plus_random_effects_times_transposed_loadings(const T_z& z, const T_mean& mean,
                                                                const T_transposed_loadings& transposed_loadings,
                                                                const int exponentiate,
                                                                std::ostream* /*pstream__*/) {

    using stan::arena_t;
    using stan::math::var;
    static const char* function_name = "nico_rows_of_mean_plus_random_effects_times_transposed_loadings";
    const auto& z_ref = stan::math::to_ref(z);
    const auto& mean_ref = stan::math::to_ref(mean);
    const auto& transposed_loadings_ref = stan::math::to_ref(transposed_loadings);
    const Eigen::Index K = z_ref.cols();
    stan::math::check_size_match(function_name, "size of mean", mean_ref.size(), "columns of z", K);
    stan::math::check_size_match(function_name, "rows of transposed_loadings", transposed_loadings_ref.rows(),
                                 "columns of z", K);
    stan::math::check_size_match(function_name, "columns of transposed_loadings",
                                 transposed_loadings_ref.cols(), "columns of z", K);
    if (exponentiate != 0 && exponentiate != 1) {
        throw std::domain_error(std::string(function_name) + ": exponentiate is " + std::to_string(exponentiate)
                                + ", but must be 0 or 1");
    }
    //
    // the values: one product for the whole matrix, the mean added to every row, then (exponentiate = 1) the
    // exponential of all n K values in one call of the exp kernel
    Eigen::Matrix<double, -1,
                  -1> result_values = stan::math::value_of(z_ref) * stan::math::value_of(transposed_loadings_ref);
    result_values.rowwise() += stan::math::value_of(mean_ref);
    if (exponentiate == 1 && result_values.size() > 0) {
        nico::math::k_exp(result_values.data(), static_cast<int>(result_values.size()));
    }
    if constexpr (!stan::is_var<stan::return_type_t<T_z, T_mean, T_transposed_loadings>>::value) {
        return result_values;
    } else {
        //
        // the inputs on the arena (their values give the vector-Jacobian product; the result's own values give
        // exp() back for w)
        // (plain matrices of each input's own scalar, var or double: an input may come as an Eigen expression)
        arena_t<Eigen::Matrix<stan::value_type_t<T_z>, Eigen::Dynamic, Eigen::Dynamic>> arena_z = z_ref;
        arena_t<Eigen::Matrix<stan::value_type_t<T_mean>, 1, Eigen::Dynamic>> arena_mean = mean_ref;
        arena_t<Eigen::Matrix<stan::value_type_t<T_transposed_loadings>, Eigen::Dynamic, Eigen::Dynamic>>
              arena_transposed_loadings = transposed_loadings_ref;
        arena_t<Eigen::Matrix<var, Eigen::Dynamic, Eigen::Dynamic>> result = result_values;
        stan::math::reverse_pass_callback([arena_z, arena_mean, arena_transposed_loadings, result,
                                           exponentiate]() mutable {
            Eigen::Matrix<double, -1, -1> w = result.adj();
            if (exponentiate == 1) {
                w.array() *= result.val().array();
            }
            if constexpr (stan::is_var<stan::value_type_t<T_z>>::value) {
                arena_z.adj() += w * stan::math::value_of(arena_transposed_loadings).transpose();
            }
            if constexpr (stan::is_var<stan::value_type_t<T_mean>>::value) {
                arena_mean.adj() += w.colwise().sum();
            }
            if constexpr (stan::is_var<stan::value_type_t<T_transposed_loadings>>::value) {
                arena_transposed_loadings.adj() += stan::math::value_of(arena_z).transpose() * w;
            }
        });
        return Eigen::Matrix<var, Eigen::Dynamic, Eigen::Dynamic>(result);
    }

}

}  // namespace nico::math

using nico::math::nico_rows_of_mean_plus_random_effects_times_transposed_loadings;























