#pragma once

#include <cmath>
#include <cstdint>
#include <limits>

namespace nico::math {

// tan for |x| <= 2^29: x reduced to [-pi/4, pi/4] in three parts (pi/4 = DP1 + DP2 + DP3, DP1 with 24 bits, so
// k * DP1 is exact), then tan(z) = z + z^3 P(z^2) / Q(z^2) (Cephes tan.c coefficients) and, in the odd
// quadrants, -1 / tan(z) = -Q / (z Q + z^3 P); one division, no branches
inline double nico_tan_1_wo_checks(const double x) {

    constexpr double DP1 = 7.853981554508209228515625E-1;
    constexpr double DP2 = 7.94662735614792836714E-9;
    constexpr double DP3 = 3.06161699786838294307E-17;
    constexpr double four_over_pi = 1.27323954473516268615;
    constexpr double P0 = -1.30936939181383777646E4;
    constexpr double P1 = 1.15351664838587416140E6;
    constexpr double P2 = -1.79565251976484877988E7;
    constexpr double Q0 = 1.36812963470692954678E4;
    constexpr double Q1 = -1.32089234440210967447E6;
    constexpr double Q2 = 2.50083801823357915839E7;
    constexpr double Q3 = -5.38695755929454629881E7;

    const double magnitude = std::fabs(x);
    const std::int64_t octant = static_cast<std::int64_t>(magnitude * four_over_pi);
    // odd octants move up one, so that z is in [-pi/4, pi/4]
    const std::int64_t k_integer = octant + (octant & 1);
    const double k = static_cast<double>(k_integer);
    const bool odd_quadrant = (k_integer & 2) != 0;
    const double z = ((magnitude - k * DP1) - k * DP2) - k * DP3;
    const double zz = z * z;
    const double P = (P0 * zz + P1) * zz + P2;
    const double Q = (((zz + Q0) * zz + Q1) * zz + Q2) * zz + Q3;
    const double z_cubed_P = z * (zz * P);
    const double numerator = odd_quadrant ? -Q : z_cubed_P;
    const double denominator = odd_quadrant ? z * Q + z_cubed_P : Q;
    const double y = (odd_quadrant ? 0.0 : z) + numerator / denominator;
    return std::signbit(x) ? -y : y;

}

inline double nico_tan_1(const double x) {

    if (!(std::fabs(x) <= 536870912.0)) return std::numeric_limits<double>::quiet_NaN();
    return nico_tan_1_wo_checks(x);

}

}  // namespace nico::math
