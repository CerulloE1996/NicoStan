#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#if defined(__SSE4_1__)
#include <immintrin.h>
#endif

namespace nico::math {

inline std::uint64_t nico_double_to_bits(const double value) {

    std::uint64_t bits;
    std::memcpy(&bits, &value, sizeof(value));
    return bits;

}

inline double nico_bits_to_double(const std::uint64_t bits) {

    double value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;

}

inline double nico_copy_sign(const double source_sign,
                             const double source_value) {

    const std::uint64_t sign_mask = std::uint64_t{1} << 63;
    const std::uint64_t sign = nico_double_to_bits(source_sign) & sign_mask;
    const std::uint64_t magnitude = nico_double_to_bits(source_value) & ~sign_mask;

    return nico_bits_to_double(magnitude | sign);

}

inline double nico_ldexp(const double a,
                         const int i) {

    const std::uint64_t exponent_bits = static_cast<std::uint64_t>(i) << 52;
    return nico_bits_to_double(exponent_bits + nico_double_to_bits(a));

}

inline double nico_round_to_nearest_even(const double value) {

#if defined(__SSE4_1__)
    // one rounding instruction, nearest with ties to even whatever the rounding mode
    const __m128d v = _mm_set_sd(value);
    return _mm_cvtsd_f64(_mm_round_sd(v, v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
#else
    const double lower = std::floor(value);
    const double fraction = value - lower;

    if (fraction < 0.5) return lower;
    if (fraction > 0.5) return lower + 1.0;
    return std::fmod(lower, 2.0) == 0.0 ? lower : lower + 1.0;
#endif

}

inline double nico_exp_power_of_two(const int exponent) {

    const std::uint64_t exponent_bits = static_cast<std::uint64_t>(exponent + 1023) << 52;
    return nico_bits_to_double(exponent_bits);

}

inline double nico_exp_1_wo_checks(const double x) {

    const double t = x * 1.442695040888963387;
    const int exponent = static_cast<int>(nico_round_to_nearest_even(t));
    const double integer_exponent = static_cast<double>(exponent);

    double remainder = std::fma(integer_exponent, -0.693145751953125, x);
    remainder = std::fma(integer_exponent, -1.42860682030941723212e-6, remainder);

    double polynomial = 0.00000276479776161191821278;
    polynomial = std::fma(polynomial, remainder, 0.0000248844480527491290235);
    polynomial = std::fma(polynomial, remainder, 0.000198411488032534342194);
    polynomial = std::fma(polynomial, remainder, 0.00138888017711994078175);
    polynomial = std::fma(polynomial, remainder, 0.00833333340524595143906);
    polynomial = std::fma(polynomial, remainder, 0.0416666670404215802592);
    polynomial = std::fma(polynomial, remainder, 0.166666666664891632843);
    polynomial = std::fma(polynomial, remainder, 0.499999999994389376923);
    polynomial = std::fma(polynomial, remainder, 1.00000000000001221245);
    polynomial = std::fma(polynomial, remainder, 1.00000000000001332268);

    return nico_ldexp(polynomial, exponent);

}

inline constexpr double nico_exp_full_range_coefficients[10] = {
    0.00000276479776161191821278, 0.0000248844480527491290235,
    0.000198411488032534342194, 0.00138888017711994078175,
    0.00833333340524595143906, 0.0416666670404215802592,
    0.166666666664891632843, 0.499999999994389376923,
    1.00000000000001221245, 1.00000000000001332268
};

inline double nico_exp_over_the_full_double_range(const double argument) {

    const double lower = static_cast<double>(
        std::numeric_limits<double>::min_exponent - std::numeric_limits<double>::digits - 1);
    const double upper = static_cast<double>(std::numeric_limits<double>::max_exponent);
    const bool in_range = argument >= lower && argument <= upper;
    const double finite_argument = in_range ? argument : 0.0;
    const double exponent = nico_round_to_nearest_even(finite_argument * 1.442695040888963387);

    double remainder = std::fma(exponent, -0.693145751953125, finite_argument);
    remainder = std::fma(exponent, -1.42860682030941723212e-6, remainder);

    double polynomial = nico_exp_full_range_coefficients[0];
    for (int degree = 1; degree < 10; ++degree) {
        polynomial = std::fma(polynomial, remainder,
                              nico_exp_full_range_coefficients[degree]);
    }

    const bool candidate = std::fabs(argument) < 708.4;
    double ordinary_result = 0.0;
    if (candidate) {
        ordinary_result = polynomial * nico_exp_power_of_two(static_cast<int>(exponent));
    }
    const bool ordinary = candidate &&
        ordinary_result >= std::numeric_limits<double>::min();
    if (ordinary) return ordinary_result;

    const int first_exponent = static_cast<int>(std::floor(exponent * 0.5));
    const int second_exponent = static_cast<int>(exponent) - first_exponent;
    double scaled = (polynomial * nico_exp_power_of_two(first_exponent)) *
        nico_exp_power_of_two(second_exponent);

    if (!in_range) scaled = 0.0;
    if (argument > upper) scaled = std::numeric_limits<double>::infinity();
    if (std::isnan(argument)) scaled = argument;

    return scaled;

}

inline double nico_exp_1(const double x) {

    const double absolute_x = std::fabs(x);
    if (absolute_x < 708.4) {
        const double result = nico_exp_1_wo_checks(x);
        if (result >= std::numeric_limits<double>::min()) return result;
    }

    return nico_exp_over_the_full_double_range(x);

}

// single-precision log, commented out: every caller uses the double-precision kernel
// inline double nico_log_1_wo_checks(const double a) {
//
//     const std::uint64_t bits = nico_double_to_bits(a);
//     const std::uint64_t exponent_bits =
//         (bits - UINT64_C(0x3fe5555555555555)) & UINT64_C(0xfff0000000000000);
//     const double exponent = std::fma(
//         static_cast<double>(static_cast<std::int64_t>(exponent_bits)),
//         0.000000000000000222044604925031308085, 0.0);
//     const double m = nico_bits_to_double(bits - exponent_bits) - 1.0;
//     const double square = m * m;
//
//     double r = -0.13031005859375;
//     double t = 0.140869140625;
//     r = std::fma(r, square, -0.121483512222766876221);
//     t = std::fma(t, square, 0.139814853668212890625);
//     r = std::fma(r, square, -0.166846126317977905273);
//     t = std::fma(t, square, 0.200120344758033752441);
//     r = std::fma(r, square, -0.249996200203895568848);
//     r = std::fma(t, m, r);
//     r = std::fma(r, m, 0.333331972360610961914);
//     r = std::fma(r, m, -0.5);
//     r = std::fma(r, square, m);
//
//     return std::fma(exponent, 0.693147182464599609375, r);
//
// }

inline void nico_log_reduce(const double a,
                             double& f,
                             double& k) {

    const std::uint64_t bits = nico_double_to_bits(a);
    const std::uint64_t exponent_bits =
        (bits - UINT64_C(0x3fe6a09e667f3bcd)) & UINT64_C(0xfff0000000000000);
    k = static_cast<double>(static_cast<std::int64_t>(exponent_bits)) *
        0.000000000000000222044604925031308085;
    f = nico_bits_to_double(bits - exponent_bits) - 1.0;

}

inline double nico_log_fdlibm_core(const double f,
                                  const double k) {

    constexpr double ln2_hi = 6.93147180369123816490e-01;
    constexpr double ln2_lo = 1.90821492927058770002e-10;
    constexpr double Lg1 = 6.666666666666735130e-01;
    constexpr double Lg2 = 3.999999999940941908e-01;
    constexpr double Lg3 = 2.857142874366239149e-01;
    constexpr double Lg4 = 2.222219843214978396e-01;
    constexpr double Lg5 = 1.818357216161805012e-01;
    constexpr double Lg6 = 1.531383769920937332e-01;
    constexpr double Lg7 = 1.479819860511658591e-01;

    const double s = f / (2.0 + f);
    const double s2 = s * s;
    const double s4 = s2 * s2;
    const double t_odd = s2 * std::fma(s4, std::fma(s4,
        std::fma(s4, Lg7, Lg5), Lg3), Lg1);
    const double t_even = s4 * std::fma(s4, std::fma(s4, Lg6, Lg4), Lg2);
    const double R = t_odd + t_even;
    const double hfsq = 0.5 * (f * f);
    const double inner = std::fma(s, hfsq + R, k * ln2_lo);

    return k * ln2_hi - ((hfsq - inner) - f);

}

inline double nico_log_1_double_precision_wo_checks(const double a) {

    double f;
    double k;
    nico_log_reduce(a, f, k);
    return nico_log_fdlibm_core(f, k);

}

// single-precision log, commented out: every caller uses the double-precision kernel
// inline double nico_log_1(const double a) {
//
//     if (a >= std::numeric_limits<double>::min() &&
//         a < std::numeric_limits<double>::infinity()) {
//         return nico_log_1_wo_checks(a);
//     }
//
//     return std::log(a);
//
// }

inline double nico_log_1_double_precision(const double a) {

    if (a >= std::numeric_limits<double>::min() &&
        a < std::numeric_limits<double>::infinity()) {
        return nico_log_1_double_precision_wo_checks(a);
    }

    return std::log(a);

}

// single-precision log, commented out: every caller uses the double-precision kernel
// inline double nico_log1p_1_wo_checks(const double x) {
//
//     if (std::fabs(x) > 1e-4) return nico_log_1_wo_checks(1.0 + x);
//     return std::fma(-0.5, x, 1.0) * x;
//
// }

inline constexpr double nico_log1p_core_lower = -0.29289321881345247560;
inline constexpr double nico_log1p_core_upper = 0.41421356237309504880;

inline double nico_log1p_1_double_precision_wo_checks(const double x) {

    const bool in_core = x >= nico_log1p_core_lower && x < nico_log1p_core_upper;
    if (in_core) return nico_log_fdlibm_core(x, 0.0);

    const double u = 1.0 + x;
    double f_u;
    double k_u;
    nico_log_reduce(u, f_u, k_u);
    const double log_value = nico_log_fdlibm_core(f_u, k_u);
    const double reciprocal = 1.0 / u;
    const double correction = (x - (u - 1.0)) * reciprocal;

    return log_value + correction;

}

// single-precision log, commented out: every caller uses the double-precision kernel
// inline double nico_log1p_1(const double x) {
//
//     if (x > -1.0 && x < std::numeric_limits<double>::infinity()) {
//         return nico_log1p_1_wo_checks(x);
//     }
//
//     return std::log1p(x);
//
// }

inline double nico_log1p_1_double_precision(const double x) {

    if (x > -1.0 && x < std::numeric_limits<double>::infinity()) {
        return nico_log1p_1_double_precision_wo_checks(x);
    }

    return std::log1p(x);

}

inline double nico_log1m_1_wo_checks(const double x) {

    return nico_log1p_1_double_precision_wo_checks(-x);

}

inline double nico_log1m_1(const double x) {

    return nico_log1p_1_double_precision(-x);

}

inline double nico_logit_wo_checks(const double x) {

    const double log_x = nico_log_1_double_precision(x);
    const double log_1m_x = nico_log1m_1(x);
    return log_x - log_1m_x;

}

inline double nico_logit(const double x) {

    const double log_x = nico_log_1_double_precision(x);
    const double log_1m_x = nico_log1m_1(x);
    return log_x - log_1m_x;

}

inline double nico_log1p_exp_1_wo_checks(const double x) {

    if (x > 0.0) return x + nico_log1p_1_double_precision_wo_checks(nico_exp_1_wo_checks(-x));
    return nico_log1p_1_double_precision_wo_checks(nico_exp_1_wo_checks(x));

}

inline double nico_log1p_exp_1(const double x) {

    if (x > 0.0) return x + nico_log1p_1_double_precision(nico_exp_1(-x));
    return nico_log1p_1_double_precision(nico_exp_1(x));

}

inline double nico_log1m_exp_1(const double x) {

    constexpr double ln2 = 0.693147180559945309417232121458;
    if (x < -ln2) return nico_log1p_1_double_precision(-nico_exp_1(x));

    const double u = x * x;
    double e = -3617.0 / 170729965486080000.0;
    e = std::fma(e, u, 1.0 / 1046139494400.0);
    e = std::fma(e, u, -691.0 / 15692092416000.0);
    e = std::fma(e, u, 1.0 / 479001600.0);
    e = std::fma(e, u, -1.0 / 9676800.0);
    e = std::fma(e, u, 1.0 / 181440.0);
    e = std::fma(e, u, -1.0 / 2880.0);
    e = std::fma(e, u, 1.0 / 24.0);
    const double h = std::fma(u, e, 0.5 * x);

    return nico_log_1_double_precision(-x) + h;

}

inline double nico_inv_logit_for_x_pos_wo_checks(const double x) {

    const double exp_m_x = nico_exp_1_wo_checks(-x);
    return 1.0 / (1.0 + exp_m_x);

}

inline double nico_inv_logit_for_x_neg_wo_checks(const double x) {

    constexpr double log_eps = -18.420680743952367;
    const double exp_x = nico_exp_1_wo_checks(x);
    if (x > log_eps) return exp_x / (1.0 + exp_x);
    return exp_x;

}

inline double nico_inv_logit_wo_checks(const double x) {

    constexpr double log_eps = -18.420680743952367;
    const bool positive = x > 0.0;
    const double argument = positive ? -x : x;
    const double exp_of_argument = nico_exp_1_wo_checks(argument);
    const bool use_quotient = positive || x > log_eps;
    double result = use_quotient ?
        (positive ? 1.0 : exp_of_argument) / (1.0 + exp_of_argument) : exp_of_argument;

    if (std::isnan(result)) return result;
    if (result < 0.0) result = 0.0;
    if (result > 1.0) result = 1.0;
    return result;

}

inline double nico_inv_logit_for_x_pos(const double x) {

    const double exp_m_x = nico_exp_1(-x);
    return 1.0 / (1.0 + exp_m_x);

}

inline double nico_inv_logit_for_x_neg(const double x) {

    constexpr double log_eps = -18.420680743952367;
    const double exp_x = nico_exp_1(x);
    if (x > log_eps) return exp_x / (1.0 + exp_x);
    return exp_x;

}

inline double nico_inv_logit(const double x) {

    constexpr double log_eps = -18.420680743952367;
    const bool positive = x > 0.0;
    const double argument = positive ? -x : x;
    const double exp_of_argument = nico_exp_1(argument);
    const bool use_quotient = positive || x > log_eps;
    double result = use_quotient ?
        (positive ? 1.0 : exp_of_argument) / (1.0 + exp_of_argument) : exp_of_argument;

    if (std::isnan(result)) return result;
    if (result < 0.0) result = 0.0;
    if (result > 1.0) result = 1.0;
    return result;

}

inline double nico_log_inv_logit_for_x_pos_wo_checks(const double x) {

    return -nico_log1p_exp_1_wo_checks(-x);

}

inline double nico_log_inv_logit_for_x_neg_wo_checks(const double x) {

    constexpr double log_eps = -18.420680743952367;
    if (x > log_eps) return x - nico_log1p_1_double_precision_wo_checks(nico_exp_1_wo_checks(x));
    return x;

}

inline double nico_log_inv_logit_wo_checks(const double x) {

    constexpr double log_eps = -18.420680743952367;
    const bool nonnegative = x >= 0.0;
    const double argument = nonnegative ? -x : x;
    const double log1p_exp = nico_log1p_1_double_precision_wo_checks(nico_exp_1_wo_checks(argument));
    if (nonnegative) return -log1p_exp;
    if (x > log_eps) return x - log1p_exp;
    return x;

}

inline double nico_log_inv_logit_for_x_pos(const double x) {

    return -nico_log1p_1_double_precision(nico_exp_1(-x));

}

inline double nico_log_inv_logit_for_x_neg(const double x) {

    constexpr double log_eps = -18.420680743952367;
    if (x > log_eps) return x - nico_log1p_1_double_precision(nico_exp_1(x));
    return x;

}

inline double nico_log_inv_logit(const double x) {

    constexpr double log_eps = -18.420680743952367;
    if (x >= 0.0) return -nico_log1p_1_double_precision(nico_exp_1(-x));
    if (x > log_eps) return x - nico_log1p_1_double_precision(nico_exp_1(x));
    return x;

}

inline double nico_tanh_wo_checks(const double x) {

    return (2.0 / (1.0 + nico_exp_1_wo_checks(-2.0 * x))) - 1.0;

}

inline double nico_tanh(const double x) {

    return (2.0 / (1.0 + nico_exp_1(-2.0 * x))) - 1.0;

}

inline double nico_square(const double x) {

    return x * x;

}

inline double nico_sqrt(const double x) {

    return std::sqrt(x);

}

inline double nico_expm1_value(const double x) {

    static constexpr double nico_expm1_inverse_factorials[15] = {
        1.0 / 1307674368000.0,
        1.0 / 87178291200.0,
        1.0 / 6227020800.0,
        1.0 / 479001600.0,
        1.0 / 39916800.0,
        1.0 / 3628800.0,
        1.0 / 362880.0,
        1.0 / 40320.0,
        1.0 / 5040.0,
        1.0 / 720.0,
        1.0 / 120.0,
        1.0 / 24.0,
        1.0 / 6.0,
        1.0 / 2.0,
        1.0
    };

    if (std::abs(x) <= 0.5) {

        double series = nico_expm1_inverse_factorials[0];
        for (int k = 1; k < 15; ++k) {
            series = std::fma(series, x, nico_expm1_inverse_factorials[k]);
        }
        return series * x;

    }

    return nico_exp_1(x) - 1.0;

}

inline double nico_tanh_value(const double x) {

    static constexpr double nico_tanh_coefficients[8] = {
        -929569.0 / 638512875.0,
        21844.0 / 6081075.0,
        -1382.0 / 155925.0,
        62.0 / 2835.0,
        -17.0 / 315.0,
        2.0 / 15.0,
        -1.0 / 3.0,
        1.0
    };

    if (std::abs(x) <= 0.125) {

        const double squared = x * x;
        double series = nico_tanh_coefficients[0];
        for (int k = 1; k < 8; ++k) {
            series = std::fma(series, squared, nico_tanh_coefficients[k]);
        }
        return series * x;

    }

    return nico_tanh(x);

}

}






















