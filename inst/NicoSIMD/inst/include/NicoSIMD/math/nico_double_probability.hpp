#pragma once

#include <cmath>
#include <limits>

#include "nico_double_exp_log.hpp"

namespace nico::math {

//// ---- constants:

static constexpr double NICO_DOUBLE_PROBABILITY_NORMAL_TAIL_THRESH = 2.5;
static constexpr int NICO_DOUBLE_PROBABILITY_MILLS_CF_DEPTH = 40;
static constexpr double NICO_DOUBLE_PROBABILITY_NEG_HALF_LOG_2PI =
    -0.91893853320467274178;
static constexpr double NICO_DOUBLE_PROBABILITY_INV_SQRT_2PI =
    0.39894228040143267794;
static constexpr double NICO_DOUBLE_PROBABILITY_FAST_PHI_MILLS_Z_CAP = 38.5;

static constexpr double NICO_DOUBLE_PROBABILITY_FAST_PHI_MILLS_NUMERATOR_COEFFICIENTS[7] = {
    0.5,
    0.5968126365007661,
    0.34755100079419643,
    0.12199983678527791,
    0.027010741025906854,
    0.0035954377698120227,
    0.0002302422023077616
};

static constexpr double NICO_DOUBLE_PROBABILITY_FAST_PHI_MILLS_DENOMINATOR_COEFFICIENTS[8] = {
    1.0,
    1.9915098343653488,
    1.7840969388752417,
    0.9377097714656843,
    0.314821789667328,
    0.06828298351478591,
    0.009012426603260822,
    0.0005771316095331759
};

static constexpr double NICO_DOUBLE_PROBABILITY_LOG_PHI_TAIL_MILLS_NUMERATOR_COEFFICIENTS[9] = {
    1.0,
    71.1021192876737,
    1886.6541015845198,
    23805.12629965518,
    150884.97847730128,
    466855.69935666997,
    630792.3515363043,
    284804.99468690663,
    18191.359378269553
};

static constexpr double NICO_DOUBLE_PROBABILITY_LOG_PHI_TAIL_MILLS_NUMERATOR_MINUS_DENOMINATOR_COEFFICIENTS[9] = {
    0.0,
    -1.0,
    -69.10211928769195,
    -1754.4498629989466,
    -20668.839292316028,
    -117585.7095446167,
    -305560.76122112013,
    -306398.0298078199,
    -76229.08437753914
};


//// ---- Mills ratios:

inline double nico_mills_ratio_tail(const double x) {

    double f = x;

    for (int k = NICO_DOUBLE_PROBABILITY_MILLS_CF_DEPTH; k >= 1; --k) {
        f = x + static_cast<double>(k) / f;
    }

    return 1.0 / f;

}


inline double nico_inverse_mills_ratio_tail(const double z) {

    const double t = 1.0 / (z * z);

    double d = NICO_DOUBLE_PROBABILITY_LOG_PHI_TAIL_MILLS_NUMERATOR_MINUS_DENOMINATOR_COEFFICIENTS[8];
    double p = NICO_DOUBLE_PROBABILITY_LOG_PHI_TAIL_MILLS_NUMERATOR_COEFFICIENTS[8];

    d = std::fma(d, t,
                 NICO_DOUBLE_PROBABILITY_LOG_PHI_TAIL_MILLS_NUMERATOR_MINUS_DENOMINATOR_COEFFICIENTS[7]);
    p = std::fma(p, t,
                 NICO_DOUBLE_PROBABILITY_LOG_PHI_TAIL_MILLS_NUMERATOR_COEFFICIENTS[7]);
    d = std::fma(d, t,
                 NICO_DOUBLE_PROBABILITY_LOG_PHI_TAIL_MILLS_NUMERATOR_MINUS_DENOMINATOR_COEFFICIENTS[6]);
    p = std::fma(p, t,
                 NICO_DOUBLE_PROBABILITY_LOG_PHI_TAIL_MILLS_NUMERATOR_COEFFICIENTS[6]);
    d = std::fma(d, t,
                 NICO_DOUBLE_PROBABILITY_LOG_PHI_TAIL_MILLS_NUMERATOR_MINUS_DENOMINATOR_COEFFICIENTS[5]);
    p = std::fma(p, t,
                 NICO_DOUBLE_PROBABILITY_LOG_PHI_TAIL_MILLS_NUMERATOR_COEFFICIENTS[5]);
    d = std::fma(d, t,
                 NICO_DOUBLE_PROBABILITY_LOG_PHI_TAIL_MILLS_NUMERATOR_MINUS_DENOMINATOR_COEFFICIENTS[4]);
    p = std::fma(p, t,
                 NICO_DOUBLE_PROBABILITY_LOG_PHI_TAIL_MILLS_NUMERATOR_COEFFICIENTS[4]);
    d = std::fma(d, t,
                 NICO_DOUBLE_PROBABILITY_LOG_PHI_TAIL_MILLS_NUMERATOR_MINUS_DENOMINATOR_COEFFICIENTS[3]);
    p = std::fma(p, t,
                 NICO_DOUBLE_PROBABILITY_LOG_PHI_TAIL_MILLS_NUMERATOR_COEFFICIENTS[3]);
    d = std::fma(d, t,
                 NICO_DOUBLE_PROBABILITY_LOG_PHI_TAIL_MILLS_NUMERATOR_MINUS_DENOMINATOR_COEFFICIENTS[2]);
    p = std::fma(p, t,
                 NICO_DOUBLE_PROBABILITY_LOG_PHI_TAIL_MILLS_NUMERATOR_COEFFICIENTS[2]);
    d = std::fma(d, t,
                 NICO_DOUBLE_PROBABILITY_LOG_PHI_TAIL_MILLS_NUMERATOR_MINUS_DENOMINATOR_COEFFICIENTS[1]);
    p = std::fma(p, t,
                 NICO_DOUBLE_PROBABILITY_LOG_PHI_TAIL_MILLS_NUMERATOR_COEFFICIENTS[1]);

    d *= t;
    p = std::fma(p, t,
                 NICO_DOUBLE_PROBABILITY_LOG_PHI_TAIL_MILLS_NUMERATOR_COEFFICIENTS[0]);

    const double z_minus_z_w = std::fma(-z, d / p, z);

    if (z == std::numeric_limits<double>::infinity()) {
        return z;
    }

    return z_minus_z_w;

}


//// ---- approximate normal probabilities:

inline double nico_Phi_approx_wo_checks(const double x) {

    const double x_sq = x * x;
    const double a_x_sq_plus_b = std::fma(0.07056, x_sq, 1.5976);
    const double stuff_to_inv_logit = x * a_x_sq_plus_b;

    return nico_inv_logit_wo_checks(stuff_to_inv_logit);

}


inline double nico_Phi_approx(const double x) {

    const double x_sq = x * x;
    const double a_x_sq_plus_b = std::fma(0.07056, x_sq, 1.5976);
    const double stuff_to_inv_logit = x * a_x_sq_plus_b;

    return nico_inv_logit(stuff_to_inv_logit);

}


inline double nico_log_Phi_approx_wo_checks(const double x) {

    const double x_sq = x * x;
    const double a_x_sq_plus_b = std::fma(0.07056, x_sq, 1.5976);
    const double stuff_to_inv_logit = x * a_x_sq_plus_b;

    return nico_log_inv_logit_wo_checks(stuff_to_inv_logit);

}


inline double nico_log_Phi_approx(const double x) {

    const double x_sq = x * x;
    const double x_cubed = x_sq * x;
    const double result = std::fma(0.07056, x_cubed, 1.5976 * x);

    return nico_log_inv_logit(result);

}


//// ---- approximate inverse normal probabilities:

inline double nico_inv_Phi_approx_wo_checks(const double p) {

    const double m_logit_p = nico_log_1_double_precision_wo_checks(1.0 / p - 1.0);
    const double x_i = -0.34176618822627863 * m_logit_p;
    const double asinh_stuff_div_3 = 0.33333333333333331483 *
        nico_log_1_double_precision_wo_checks(x_i + std::sqrt(std::fma(x_i, x_i, 1.0)));
    const double exp_x_i = nico_exp_1_wo_checks(asinh_stuff_div_3);

    return 2.7472242570765295 * ((exp_x_i * exp_x_i - 1.0) / exp_x_i);

}


inline double nico_inv_Phi_approx(const double p) {

    if (p >= std::numeric_limits<double>::min() && p < 1.0) {
        const double m_logit_p = nico_log_1_double_precision((1.0 - p) / p);
        const double x_i = -0.34176618822627863 * m_logit_p;
        const double asinh_stuff_div_3 = 0.33333333333333331483 *
            nico_log_1_double_precision(x_i + std::sqrt(std::fma(x_i, x_i, 1.0)));
        const double exp_x_i = nico_exp_1(asinh_stuff_div_3);

        return 2.7472242570765295 * ((exp_x_i * exp_x_i - 1.0) / exp_x_i);
    }

    const double logit_probability = std::log(p) - std::log1p(-p);

    return 5.494448514153059 *
        std::sinh(std::asinh(0.34176618822627863 * logit_probability) / 3.0);

}


inline double nico_inv_Phi_approx_from_logit_prob_wo_checks(const double logit_p) {

    const double x_i = logit_p * 0.34176618822627863;
    const double asinh_stuff_div_3 = 0.33333333333333331483 *
        nico_log_1_double_precision_wo_checks(x_i + std::sqrt(std::fma(x_i, x_i, 1.0)));
    const double exp_x_i = nico_exp_1_wo_checks(asinh_stuff_div_3);

    return 2.7472242570765295 * ((exp_x_i * exp_x_i - 1.0) / exp_x_i);

}


inline double nico_inv_Phi_approx_from_logit_prob(const double logit_p) {

    const double x_i = logit_p * 0.34176618822627863;
    const double asinh_stuff_div_3 = 0.33333333333333331483 *
        nico_log_1_double_precision(x_i + std::sqrt(std::fma(x_i, x_i, 1.0)));
    const double exp_x_i = nico_exp_1(asinh_stuff_div_3);

    return 2.7472242570765295 * ((exp_x_i * exp_x_i - 1.0) / exp_x_i);

}


inline double nico_inv_Phi_approx_from_logit_prob_public(const double logit_p) {

    static constexpr double odd_inverse_factorials[8] = {
        1.0 / 1307674368000.0,
        1.0 / 6227020800.0,
        1.0 / 39916800.0,
        1.0 / 362880.0,
        1.0 / 5040.0,
        1.0 / 120.0,
        1.0 / 6.0,
        1.0
    };

    const double one = 1.0;
    const double s = std::fabs(0.34176618822627863 * logit_p);
    const double s_squared = s * s;
    const double log1p_argument = s + s_squared / (one + std::sqrt(one + s_squared));
    const double two_s_minus_one = std::fma(2.0, s, -one);
    const bool large = s > 1e8;
    const double asinh_argument = large ? two_s_minus_one : log1p_argument;
    const double y = 0.33333333333333331483 * nico_log1p_1_double_precision(asinh_argument);
    const double y_squared = y * y;

    double series = odd_inverse_factorials[0];
    for (int k = 1; k < 8; ++k) {
        series = std::fma(series, y_squared, odd_inverse_factorials[k]);
    }
    series = series * y;

    const double exp_y = nico_exp_1(y);
    const double from_exp = 0.5 * (exp_y - one / exp_y);
    const double sinh_y = y <= 0.5 ? series : from_exp;
    const double result = 5.494448514153059 * sinh_y;
    const std::uint64_t sign_bit = nico_double_to_bits(logit_p) & (std::uint64_t{1} << 63);

    return nico_bits_to_double(nico_double_to_bits(result) ^ sign_bit);

}


//// ---- normal probabilities:

inline double nico_Phi_wo_checks(const double x) {

    const double z = std::fabs(x);
    const double z_rational = (z < NICO_DOUBLE_PROBABILITY_FAST_PHI_MILLS_Z_CAP)
        ? z
        : NICO_DOUBLE_PROBABILITY_FAST_PHI_MILLS_Z_CAP;

    double numerator = NICO_DOUBLE_PROBABILITY_FAST_PHI_MILLS_NUMERATOR_COEFFICIENTS[6];
    numerator = std::fma(numerator, z_rational,
                         NICO_DOUBLE_PROBABILITY_FAST_PHI_MILLS_NUMERATOR_COEFFICIENTS[5]);
    numerator = std::fma(numerator, z_rational,
                         NICO_DOUBLE_PROBABILITY_FAST_PHI_MILLS_NUMERATOR_COEFFICIENTS[4]);
    numerator = std::fma(numerator, z_rational,
                         NICO_DOUBLE_PROBABILITY_FAST_PHI_MILLS_NUMERATOR_COEFFICIENTS[3]);
    numerator = std::fma(numerator, z_rational,
                         NICO_DOUBLE_PROBABILITY_FAST_PHI_MILLS_NUMERATOR_COEFFICIENTS[2]);
    numerator = std::fma(numerator, z_rational,
                         NICO_DOUBLE_PROBABILITY_FAST_PHI_MILLS_NUMERATOR_COEFFICIENTS[1]);
    numerator = std::fma(numerator, z_rational,
                         NICO_DOUBLE_PROBABILITY_FAST_PHI_MILLS_NUMERATOR_COEFFICIENTS[0]);

    double denominator = NICO_DOUBLE_PROBABILITY_FAST_PHI_MILLS_DENOMINATOR_COEFFICIENTS[7];
    denominator = std::fma(denominator, z_rational,
                           NICO_DOUBLE_PROBABILITY_FAST_PHI_MILLS_DENOMINATOR_COEFFICIENTS[6]);
    denominator = std::fma(denominator, z_rational,
                           NICO_DOUBLE_PROBABILITY_FAST_PHI_MILLS_DENOMINATOR_COEFFICIENTS[5]);
    denominator = std::fma(denominator, z_rational,
                           NICO_DOUBLE_PROBABILITY_FAST_PHI_MILLS_DENOMINATOR_COEFFICIENTS[4]);
    denominator = std::fma(denominator, z_rational,
                           NICO_DOUBLE_PROBABILITY_FAST_PHI_MILLS_DENOMINATOR_COEFFICIENTS[3]);
    denominator = std::fma(denominator, z_rational,
                           NICO_DOUBLE_PROBABILITY_FAST_PHI_MILLS_DENOMINATOR_COEFFICIENTS[2]);
    denominator = std::fma(denominator, z_rational,
                           NICO_DOUBLE_PROBABILITY_FAST_PHI_MILLS_DENOMINATOR_COEFFICIENTS[1]);
    denominator = std::fma(denominator, z_rational,
                           NICO_DOUBLE_PROBABILITY_FAST_PHI_MILLS_DENOMINATOR_COEFFICIENTS[0]);

    const double lower_tail_probability =
        nico_exp_1(-0.5 * (z * z)) * (numerator / denominator);

    if (x > 0.0) {
        return 1.0 - lower_tail_probability;
    }

    return lower_tail_probability;

}


inline double nico_Phi(const double x) {

    if (std::isnan(x)) {
        return x;
    }

    double probability;

    if (x >= -37.5 && x < 8.25) {
        probability = nico_Phi_wo_checks(x);
    } else if (x < -37.5) {
        probability = 0.0;
    } else {
        probability = 1.0;
    }

    if (probability < 0.0) {
        return 0.0;
    }

    if (probability > 1.0) {
        return 1.0;
    }

    return probability;

}


//// ---- inverse normal probabilities:

inline double nico_inv_Phi_wo_checks_case_1(const double,
                                            const double q) {

    const double q_sq = q * q;
    const double r = 0.180625 - q_sq;

    double numerator = std::fma(r, 2509.0809287301226727, 33430.575583588128105);
    numerator = std::fma(numerator, r, 67265.770927008700853);
    numerator = std::fma(numerator, r, 45921.953931549871457);
    numerator = std::fma(numerator, r, 13731.693765509461125);
    numerator = std::fma(numerator, r, 1971.5909503065514427);
    numerator = std::fma(numerator, r, 133.14166789178437745);
    numerator = std::fma(numerator, r, 3.387132872796366608);

    double denominator = std::fma(r, 5226.495278852854561, 28729.085735721942674);
    denominator = std::fma(denominator, r, 39307.89580009271061);
    denominator = std::fma(denominator, r, 21213.794301586595867);
    denominator = std::fma(denominator, r, 5394.1960214247511077);
    denominator = std::fma(denominator, r, 687.1870074920579083);
    denominator = std::fma(denominator, r, 42.313330701600911252);
    denominator = std::fma(denominator, r, 1.0);

    return q * (numerator / denominator);

}


inline double nico_inv_Phi_wo_checks_case_2a(const double,
                                             const double r) {

    const double r_shifted = r - 1.60;

    double numerator = 0.00077454501427834140764;
    numerator = std::fma(r_shifted, numerator, 0.0227238449892691845833);
    numerator = std::fma(r_shifted, numerator, 0.24178072517745061177);
    numerator = std::fma(r_shifted, numerator, 1.27045825245236838258);
    numerator = std::fma(r_shifted, numerator, 3.64784832476320460504);
    numerator = std::fma(r_shifted, numerator, 5.7694972214606914055);
    numerator = std::fma(r_shifted, numerator, 4.6303378461565452959);
    numerator = std::fma(r_shifted, numerator, 1.42343711074968357734);

    double denominator = 0.00000000105075007164441684324;
    denominator = std::fma(r_shifted, denominator, 0.0005475938084995344946);
    denominator = std::fma(r_shifted, denominator, 0.0151986665636164571966);
    denominator = std::fma(r_shifted, denominator, 0.14810397642748007459);
    denominator = std::fma(r_shifted, denominator, 0.68976733498510000455);
    denominator = std::fma(r_shifted, denominator, 1.6763848301838038494);
    denominator = std::fma(r_shifted, denominator, 2.05319162663775882187);
    denominator = std::fma(r_shifted, denominator, 1.0);

    return numerator / denominator;

}


inline double nico_inv_Phi_wo_checks_case_2b(const double,
                                             const double r) {

    const double r_shifted = r - 5.0;

    double numerator = 0.000000201033439929228813265;
    numerator = std::fma(r_shifted, numerator, 0.0000271155556874348757815);
    numerator = std::fma(r_shifted, numerator, 0.0012426609473880784386);
    numerator = std::fma(r_shifted, numerator, 0.026532189526576123093);
    numerator = std::fma(r_shifted, numerator, 0.29656057182850489123);
    numerator = std::fma(r_shifted, numerator, 1.7848265399172913358);
    numerator = std::fma(r_shifted, numerator, 5.4637849111641143699);
    numerator = std::fma(r_shifted, numerator, 6.6579046435011037772);

    double denominator = 0.00000000000000204426310338993978564;
    denominator = std::fma(r_shifted, denominator, 0.00000014215117583164458887);
    denominator = std::fma(r_shifted, denominator, 0.000018463183175100546818);
    denominator = std::fma(r_shifted, denominator, 0.0007868691311456132591);
    denominator = std::fma(r_shifted, denominator, 0.0148753612908506148525);
    denominator = std::fma(r_shifted, denominator, 0.13692988092273580531);
    denominator = std::fma(r_shifted, denominator, 0.59983220655588793769);
    denominator = std::fma(r_shifted, denominator, 1.0);

    return numerator / denominator;

}


inline double nico_inv_Phi_wo_checks_case_2(const double p,
                                            const double q) {

    const double tail_probability = q > 0.0 ? 1.0 - p : p;
    const double r = std::sqrt(-nico_log_1_double_precision(tail_probability));
    const double magnitude = r > 5.0
        ? nico_inv_Phi_wo_checks_case_2b(p, r)
        : nico_inv_Phi_wo_checks_case_2a(p, r);

    return q < 0.0 ? -magnitude : magnitude;

}


inline double nico_inv_Phi_wo_checks(const double p) {

    const double q = p - 0.5;
    const double central = std::fabs(q);

    if (central <= 0.425) {
        return nico_inv_Phi_wo_checks_case_1(p, q);
    }

    return nico_inv_Phi_wo_checks_case_2(p, q);

}


inline double nico_inv_Phi_from_log_p_wo_checks(const double log_p,
                                                const double log_1m_p) {

    const double p = nico_exp_1(log_p);
    const double q = p - 0.5;
    const bool central = std::fabs(q) <= 0.425;
    const double log_min = log_p < log_1m_p ? log_p : log_1m_p;
    const double r = std::sqrt(-log_min);
    const double magnitude = r > 5.0
        ? nico_inv_Phi_wo_checks_case_2b(p, r)
        : nico_inv_Phi_wo_checks_case_2a(p, r);
    const double tail = log_p < log_1m_p ? -magnitude : magnitude;
    const double center = nico_inv_Phi_wo_checks_case_1(p, q);

    return central ? center : tail;

}


//// ---- normal log probabilities and derivatives:

inline double nico_log_Phi(const double x) {

    if (x < -NICO_DOUBLE_PROBABILITY_NORMAL_TAIL_THRESH) {
        const double inverse_mills_ratio = nico_inverse_mills_ratio_tail(-x);
        const double mills_ratio = 1.0 / inverse_mills_ratio;

        return std::fma(-0.5, x * x,
                        NICO_DOUBLE_PROBABILITY_NEG_HALF_LOG_2PI) + nico_log_1_double_precision(mills_ratio);
    }

    return nico_log_1_double_precision(nico_Phi(x));

}


inline double nico_log_1m_Phi(const double x) {

    return nico_log_Phi(-x);

}


inline double nico_dlog_Phi_dx(const double x) {

    if (x < -NICO_DOUBLE_PROBABILITY_NORMAL_TAIL_THRESH) {
        return nico_inverse_mills_ratio_tail(-x);
    }

    const double phi = NICO_DOUBLE_PROBABILITY_INV_SQRT_2PI *
        nico_exp_1(-0.5 * x * x);

    return phi / nico_Phi(x);

}


inline double nico_dZ_dv_from_log_p(const double Z,
                                    const double log_p,
                                    const double log_1m_p,
                                    const double dlog_p_dv,
                                    const double dlog_1m_p_dv) {

    const double log_inv_phi = std::fma(0.5, Z * Z,
                                        -NICO_DOUBLE_PROBABILITY_NEG_HALF_LOG_2PI);
    const double e_lo = std::fmin(log_p + log_inv_phi, 700.0);
    const double e_hi = std::fmin(log_1m_p + log_inv_phi, 700.0);
    const double via_lo = nico_exp_1(e_lo) * dlog_p_dv;
    const double via_hi = -nico_exp_1(e_hi) * dlog_1m_p_dv;

    return Z < 0.0 ? via_lo : via_hi;

}


}






















