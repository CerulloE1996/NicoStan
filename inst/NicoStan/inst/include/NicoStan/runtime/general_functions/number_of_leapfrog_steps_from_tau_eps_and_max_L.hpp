#pragma once


#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>


inline int fn_max_L_from_positive_whole_number(const double max_L) {

    if (!std::isfinite(max_L) || max_L < 1 || std::floor(max_L) != max_L ||
        max_L > static_cast<double>(std::numeric_limits<int>::max())) {

        throw std::invalid_argument("max_L must be a positive whole number representable as an int.");

    }

    return static_cast<int>(max_L);

}


inline int fn_number_of_leapfrog_steps_from_tau_eps_and_max_L(
    const double tau,
    const double eps,
    const std::optional<int> &max_L
) {

    if (!std::isfinite(tau) || tau < 0 || !std::isfinite(eps) || eps <= 0) {
        throw std::invalid_argument("Trajectory length must be finite and nonnegative; eps must be positive.");
    }

    double number_of_leapfrog_steps = std::max(1.0, std::ceil(tau / eps));

    if (max_L.has_value()) {
        if (*max_L < 1) {
            throw std::invalid_argument("max_L must be positive.");
        }
        number_of_leapfrog_steps = std::min(number_of_leapfrog_steps, static_cast<double>(*max_L));
    }

    if (!std::isfinite(number_of_leapfrog_steps) ||
        number_of_leapfrog_steps > static_cast<double>(std::numeric_limits<int>::max())) {

        throw std::overflow_error("The uncapped leapfrog count cannot be represented as an int.");

    }

    return static_cast<int>(number_of_leapfrog_steps);

}






















