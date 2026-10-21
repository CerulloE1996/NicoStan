#pragma once
//
// -| --------- NicoSIMD exponents and coefficients of a linear two-compartment system, first-order input ---------
//
//     nico_two_compartment_first_order_input_exponents_and_coefficients(rates, minimum_rate_separation)
//
// rates: an N x 5 matrix, one row per unit (e.g. patient, subject, tracer experiment), with the columns k10 (the
// elimination rate from the central compartment), k12 and k21 (the transfer rates between the central and the
// peripheral compartment), ka (the first-order input rate) and D (the scale of the response, e.g. dose ka / V1).
// The closed-form solution of every linear two-compartment system with first-order input (oral or
// extravascular PK, PK/PD models with an effect driven by the central amount, tracer kinetics, any linear
// two-compartment ODE with a first-order source) is
//
//     C(t) = A exp(-alpha t) + B exp(-beta t) + K exp(-ka t),
//
// with the roots alpha > beta of s^2 - (k10 + k12 + k21) s + k10 k21 and the partial-fraction coefficients
// A = D (alpha - k21) / ((ka - alpha) R), B = D (k21 - beta) / ((ka - beta) R) and K = -(A + B), with
// R = alpha - beta.
// The result is an N x 7 matrix with the columns A, B, K, alpha, beta, k21 - beta and the smallest rate
// separation min(|ka - alpha|, |ka - beta|, R) (a value only, for the choice of the stable form: its adjoint is
// not used).
//
// Cancellation-free for both signs of a = k10 + k12 - k21, with R = sqrt(a^2 + 4 k12 k21):
//     a >= 0: alpha - k21 = (R + a) / 2 and k21 - beta = k12 k21 / (alpha - k21);
//     a <  0: k21 - beta = (R - a) / 2 and alpha - k21 = k12 k21 / (k21 - beta);
// alpha = k21 + (alpha - k21) and beta = k10 k21 / alpha. The units whose smallest rate separation is below
// minimum_rate_separation (the stable form, computed by the caller) get A = B = K = 0 and no adjoint through
// them, as in the Stan models that use it.
//
// ONE autodiff step for the whole matrix (Stan's version: about 30 nodes per unit), with every per-unit
// calculation of the values and of the vector-Jacobian product made on whole Eigen arrays (SIMD, no loop over
// the units), both branches computed for every unit and chosen with select().
//
// Declared in the Stan functions block (no body; stanc --allow-undefined):
//
//     matrix nico_two_compartment_first_order_input_exponents_and_coefficients(
//         matrix rates, data real minimum_rate_separation);
//
// -----------------------------------------------------------------------------------------------------------

#include <stan/math/rev.hpp>
#include <stan/math/prim.hpp>

#include <Eigen/Dense>
#include <ostream>




// ---- the Stan-facing function (global scope; stanc calls it with pstream__ last)
namespace nico::math {

template <typename T_rates, stan::require_eigen_matrix_dynamic_t<T_rates>* = nullptr>
inline Eigen::Matrix<stan::return_type_t<T_rates>, Eigen::Dynamic, Eigen::Dynamic>
nico_two_compartment_first_order_input_exponents_and_coefficients(const T_rates& rates,
                                                                   const double minimum_rate_separation,
                                                                   std::ostream* /*pstream__*/ = nullptr) {

    using stan::arena_t;
    using stan::math::var;
    using Array = Eigen::Array<double, -1, 1>;
    static const char* function_name = "nico_two_compartment_first_order_input_exponents_and_coefficients";
    stan::math::check_size_match(function_name, "columns of rates", rates.cols(), "", 5);
    const auto& rates_ref = stan::math::to_ref(rates);
    const Eigen::Index N = rates_ref.rows();
    const Eigen::Matrix<double, -1, -1> rates_values = stan::math::value_of(rates_ref);
    //
    // the values, one whole array per quantity
    const Array k10 = rates_values.col(0).array();
    const Array k12 = rates_values.col(1).array();
    const Array k21 = rates_values.col(2).array();
    const Array ka = rates_values.col(3).array();
    const Array D = rates_values.col(4).array();
    const Array k12_times_k21 = k12 * k21;
    const Array a = k10 + k12 - k21;
    const Array alpha_minus_beta = (a.square() + 4.0 * k12_times_k21).sqrt();
    const auto a_negative = (a < 0.0);
    const Array alpha_minus_k21_if_a_positive = 0.5 * (alpha_minus_beta + a);
    const Array k21_minus_beta_if_a_negative = 0.5 * (alpha_minus_beta - a);
    const Array alpha_minus_k21 = a_negative.select(k12_times_k21 / k21_minus_beta_if_a_negative,
                                                    alpha_minus_k21_if_a_positive);
    const Array k21_minus_beta = a_negative.select(k21_minus_beta_if_a_negative,
                                                   k12_times_k21 / alpha_minus_k21_if_a_positive);
    const Array alpha = k21 + alpha_minus_k21;
    const Array beta = k10 * k21 / alpha;
    const Array ka_minus_alpha = ka - alpha;
    const Array ka_minus_beta = ka - beta;
    const Array smallest_rate_separation = ka_minus_alpha.abs().min(ka_minus_beta.abs()).min(alpha_minus_beta);
    const auto stable_form = (smallest_rate_separation < minimum_rate_separation);
    // (the stable-form units' differences set to 1 in the divisions, and their coefficients then set to 0)
    const Array ka_minus_alpha_in_divisions = stable_form.select(Array::Ones(N), ka_minus_alpha);
    const Array ka_minus_beta_in_divisions = stable_form.select(Array::Ones(N), ka_minus_beta);
    const Array alpha_minus_beta_in_divisions = stable_form.select(Array::Ones(N), alpha_minus_beta);
    const Array D_over_alpha_minus_beta = D / alpha_minus_beta_in_divisions;
    const Array coefficient_alpha = stable_form.select(Array::Zero(N), D_over_alpha_minus_beta * alpha_minus_k21
                                                                       / ka_minus_alpha_in_divisions);
    const Array coefficient_beta = stable_form.select(Array::Zero(N), D_over_alpha_minus_beta * k21_minus_beta
                                                                      / ka_minus_beta_in_divisions);
    Eigen::Matrix<double, -1, -1> result_values(N, 7);
    result_values.col(0) = coefficient_alpha.matrix();
    result_values.col(1) = coefficient_beta.matrix();
    result_values.col(2) = (-coefficient_alpha - coefficient_beta).matrix();
    result_values.col(3) = alpha.matrix();
    result_values.col(4) = beta.matrix();
    result_values.col(5) = k21_minus_beta.matrix();
    result_values.col(6) = smallest_rate_separation.matrix();
    if constexpr (!stan::is_var<stan::value_type_t<T_rates>>::value) {
        return result_values;
    } else {
        //
        // the values the vector-Jacobian product needs, kept on the arena (one N x 13 block)
        arena_t<Eigen::Matrix<var, Eigen::Dynamic, Eigen::Dynamic>> arena_rates = rates_ref;
        arena_t<Eigen::Matrix<double, -1, -1>> kept(N, 13);
        kept.col(0) = k10.matrix();
        kept.col(1) = k12.matrix();
        kept.col(2) = k21.matrix();
        kept.col(3) = D.matrix();
        kept.col(4) = a.matrix();
        kept.col(5) = alpha_minus_beta.matrix();
        kept.col(6) = alpha_minus_k21.matrix();
        kept.col(7) = k21_minus_beta.matrix();
        kept.col(8) = alpha.matrix();
        kept.col(9) = beta.matrix();
        kept.col(10) = D_over_alpha_minus_beta.matrix();
        kept.col(11) = ka_minus_alpha_in_divisions.matrix();
        kept.col(12) = ka_minus_beta_in_divisions.matrix();
        arena_t<Eigen::Matrix<bool, Eigen::Dynamic, 1>> kept_a_negative = a_negative.matrix();
        arena_t<Eigen::Matrix<bool, Eigen::Dynamic, 1>> kept_stable_form = stable_form.matrix();
        arena_t<Eigen::Matrix<var, Eigen::Dynamic, Eigen::Dynamic>> result = result_values;
        stan::math::reverse_pass_callback([arena_rates, kept, kept_a_negative, kept_stable_form, result,
                                           N]() mutable {
            const Array k10 = kept.col(0).array();
            const Array k12 = kept.col(1).array();
            const Array k21 = kept.col(2).array();
            const Array D = kept.col(3).array();
            const Array a = kept.col(4).array();
            const Array alpha_minus_beta = kept.col(5).array();
            const Array alpha_minus_k21 = kept.col(6).array();
            const Array k21_minus_beta = kept.col(7).array();
            const Array alpha = kept.col(8).array();
            const Array beta = kept.col(9).array();
            const Array D_over_alpha_minus_beta = kept.col(10).array();
            const Array ka_minus_alpha_in_divisions = kept.col(11).array();
            const Array ka_minus_beta_in_divisions = kept.col(12).array();
            const auto a_negative = kept_a_negative.array();
            const auto stable_form = kept_stable_form.array();
            const Array adjoint_K = result.col(2).adj().array();
            // (K = -(A + B))
            const Array adjoint_A = stable_form.select(Array::Zero(N), result.col(0).adj().array() - adjoint_K);
            const Array adjoint_B = stable_form.select(Array::Zero(N), result.col(1).adj().array() - adjoint_K);
            // A = q (alpha - k21) / (ka - alpha), B = q (k21 - beta) / (ka - beta), q = D / R
            const Array A_over_q_by_A = alpha_minus_k21 / ka_minus_alpha_in_divisions;
            const Array B_over_q_by_B = k21_minus_beta / ka_minus_beta_in_divisions;
            const Array adjoint_q = adjoint_A * A_over_q_by_A + adjoint_B * B_over_q_by_B;
            const Array adjoint_ka_minus_alpha = -adjoint_A * D_over_alpha_minus_beta * A_over_q_by_A
                                                 / ka_minus_alpha_in_divisions;
            const Array adjoint_ka_minus_beta = -adjoint_B * D_over_alpha_minus_beta * B_over_q_by_B
                                                / ka_minus_beta_in_divisions;
            const Array adjoint_D = adjoint_q / alpha_minus_beta;
            Array adjoint_alpha_minus_beta = -adjoint_q * D_over_alpha_minus_beta / alpha_minus_beta;
            Array adjoint_alpha_minus_k21 = adjoint_A * D_over_alpha_minus_beta / ka_minus_alpha_in_divisions;
            Array adjoint_k21_minus_beta = result.col(5).adj().array()
                                           + adjoint_B * D_over_alpha_minus_beta / ka_minus_beta_in_divisions;
            // (the stable-form units: no adjoint through their coefficients; the products above are 0 there,
            //  and their divisions use 1)
            adjoint_alpha_minus_beta = stable_form.select(Array::Zero(N), adjoint_alpha_minus_beta);
            const Array adjoint_ka = adjoint_ka_minus_alpha + adjoint_ka_minus_beta;
            // beta = k10 k21 / alpha, alpha = k21 + (alpha - k21)
            const Array adjoint_beta = result.col(4).adj().array() - adjoint_ka_minus_beta;
            const Array adjoint_alpha = result.col(3).adj().array() - adjoint_ka_minus_alpha
                                        - adjoint_beta * beta / alpha;
            Array adjoint_k10 = adjoint_beta * k21 / alpha;
            Array adjoint_k21 = adjoint_beta * k10 / alpha + adjoint_alpha;
            adjoint_alpha_minus_k21 += adjoint_alpha;
            // the two branches of the sign of a (k12 k21 = (alpha - k21) (k21 - beta))
            //     a >= 0: alpha - k21 = (R + a) / 2, k21 - beta = k12 k21 / (alpha - k21)
            const Array adjoint_alpha_minus_k21_if_a_positive = adjoint_alpha_minus_k21
                                                                - adjoint_k21_minus_beta * k21_minus_beta
                                                                  / alpha_minus_k21;
            const Array adjoint_k12_times_k21_if_a_positive = adjoint_k21_minus_beta / alpha_minus_k21;
            //     a < 0: k21 - beta = (R - a) / 2, alpha - k21 = k12 k21 / (k21 - beta)
            const Array adjoint_k21_minus_beta_if_a_negative = adjoint_k21_minus_beta
                                                               - adjoint_alpha_minus_k21 * alpha_minus_k21
                                                                 / k21_minus_beta;
            const Array adjoint_k12_times_k21_if_a_negative = adjoint_alpha_minus_k21 / k21_minus_beta;
            Array adjoint_k12_times_k21 = a_negative.select(adjoint_k12_times_k21_if_a_negative,
                                                            adjoint_k12_times_k21_if_a_positive);
            adjoint_alpha_minus_beta += a_negative.select(0.5 * adjoint_k21_minus_beta_if_a_negative,
                                                          0.5 * adjoint_alpha_minus_k21_if_a_positive);
            Array adjoint_a = a_negative.select(-0.5 * adjoint_k21_minus_beta_if_a_negative,
                                                0.5 * adjoint_alpha_minus_k21_if_a_positive);
            // R = sqrt(a^2 + 4 k12 k21)
            adjoint_a += adjoint_alpha_minus_beta * a / alpha_minus_beta;
            adjoint_k12_times_k21 += 2.0 * adjoint_alpha_minus_beta / alpha_minus_beta;
            // a = k10 + k12 - k21, k12 k21
            adjoint_k10 += adjoint_a;
            const Array adjoint_k12 = adjoint_a + adjoint_k12_times_k21 * k21;
            adjoint_k21 += -adjoint_a + adjoint_k12_times_k21 * k12;
            arena_rates.col(0).adj().array() += adjoint_k10;
            arena_rates.col(1).adj().array() += adjoint_k12;
            arena_rates.col(2).adj().array() += adjoint_k21;
            arena_rates.col(3).adj().array() += adjoint_ka;
            arena_rates.col(4).adj().array() += adjoint_D;
        });
        return Eigen::Matrix<var, Eigen::Dynamic, Eigen::Dynamic>(result);
    }

}

}  // namespace nico::math

using nico::math::nico_two_compartment_first_order_input_exponents_and_coefficients;























