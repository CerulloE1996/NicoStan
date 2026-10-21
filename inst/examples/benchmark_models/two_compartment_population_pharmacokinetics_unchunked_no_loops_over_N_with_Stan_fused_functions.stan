// two_compartment_population_pharmacokinetics_unchunked_no_loops_over_N_with_Stan_fused_functions.stan
// Unchunked Stan::math version of the population PK model of the tape-chunking paper's layout timings
// (two_compartment_population_pharmacokinetics_chunked_no_loops_over_N_v2_automatic_reduce_sum_chunked, from its
// source two_compartment_population_pharmacokinetics_chunked_no_loops_over_N_v2): the same log density as that
// source at N_chunks = 1, and as two_compartment_population_pharmacokinetics.stan, as one pass over all N
// patients, without reduce_sum_static. The same data (chunk_size, when it is in the data, is not read) and the
// same unconstrained parameters, in the same order and with the same names (z, mu_CL, mu_Q, mu_V1, mu_V2,
// mu_ka, omega, L_Omega, sigma), as both files; z is the 5 x N matrix of
// two_compartment_population_pharmacokinetics.stan (column i: patient i's 5 values), whose unconstrained
// coordinates are in the same order as those of the source's array[N] row_vector[5] z (which exists only so that
// reduce_sum_static can slice the patients); with the matrix, the patients' log CL, log Q, log V1, log V2 and
// log ka are one 5 x 5 by 5 x N product, with no transposed copy of z.
// The functions are those of the source, unchanged: two_compartment_oral_concentrations() and the vector
// versions of the stable form, whole-vector calculations over all samples (the only loops left are over the
// terms of their Taylor series, 8 and 12 terms, each a whole-vector step, and the scalar functions of mu_ka's
// lower bound). Stan has no fused function for this likelihood: log(observed concentration) ~ normal(log C(t),
// sigma), Stan's vectorised normal on the whole vector of samples (lognormal on the concentrations would add
// the data term -sum(log(concentration)), a different log density with propto = FALSE).
// No loop over N anywhere.
//
// The model, as in two_compartment_population_pharmacokinetics.stan:
//
// Two-compartment population pharmacokinetic (PK) model with first-order oral absorption: N patients, each given
// one oral dose (dose, mg) at time 0, with log-normal residual error on the plasma concentrations sampled at
// times t > 0 (h), and correlated between-patient random effects on all five PK parameters, non-centred.
//
// Structural model (amounts in mg; depot A_d, central A_c, peripheral A_p; A_d(0) = dose, A_c(0) = A_p(0) = 0):
//   dA_d/dt = -ka A_d,
//   dA_c/dt = ka A_d - (k10 + k12) A_c + k21 A_p,
//   dA_p/dt = k12 A_c - k21 A_p,
// with the micro-constants k10 = CL / V1, k12 = Q / V1 and k21 = Q / V2 (CL, Q in L/h; V1, V2 in L; ka in 1/h),
// and the predicted concentration C(t) = A_c(t) / V1 (mg/L). The disposition rates alpha > beta are the roots of
// lambda^2 - (k10 + k12 + k21) lambda + k10 k21 = 0:
//   alpha = (k10 + k12 + k21 + R) / 2,  beta = k10 k21 / alpha,
//   R = alpha - beta = sqrt((k10 + k12 - k21)^2 + 4 k12 k21)
// (the discriminant written as a sum of non-negative terms, and beta from alpha, so that neither cancels), and
// beta < k21 < alpha always. The closed-form solution (tri-exponential) is
//   C(t) = dose ka / V1 [ A exp(-alpha t) + B exp(-beta t) + K exp(-ka t) ],
//   A = (alpha - k21) / ((ka - alpha) (alpha - beta)),  B = (k21 - beta) / ((ka - beta) (alpha - beta)),
//   K = (k21 - ka) / ((ka - alpha) (ka - beta)),
// i.e. dose ka / V1 times the second divided difference of f(lambda) = (k21 - lambda) exp(-lambda t) over the
// three rates {alpha, beta, ka}. Torsten's pmx_solve_twocpt is not used: the solution is written here, in
// Stan::math.
//
// Numerical robustness: the tri-exponential divides by the differences of the three rates, so when ka is close
// to alpha or beta (or alpha close to beta) its terms grow like 1 / (difference) and cancel: for a difference d
// at time t it loses about log10(1 / (d t)) digits in the concentration and twice that in the gradient, and it is
// undefined at d = 0. It is used for a patient only when every difference is at least minimum_rate_separation =
// 0.01 / (the earliest sampling time) (0.04 per h for the first sample at 0.25 h), i.e. d t >= 0.01 at every
// sample. Every other patient (rates closer than that, including exactly equal rates) gets the same solution
// written without any such division (the Leibniz rule for divided differences, with the rates in the order
// beta, alpha, ka):
//   C(t) = dose ka / V1 [ (k21 - beta) E2(alpha, beta, ka; t) + D1(alpha, ka; t) ],
// with D1(a, b; t) = (exp(-b t) - exp(-a t)) / (a - b) (minus the first divided difference of exp(-lambda t))
// and E2 the second divided difference of exp(-lambda t), both non-negative and computed stably (see the
// functions below), so the concentration is a sum of two non-negative terms.
// check_population_pharmacokinetics_closed_form.R checks both forms against numerical ODE solutions and a matrix
// exponential, also at nearly and exactly equal rates.
//
// Patient parameters (log scale; patient i = column i): log_theta_i = (log CL_i, log Q_i, log V1_i, log V2_i,
// log ka_i)' = mu + diag(omega) L_Omega z_i, z_i ~ N(0, I_5) (non-centred), i.e. log_theta_i ~ N(mu, Sigma)
// with Sigma = diag(omega) L_Omega L_Omega' diag(omega). z (5 x N) is the nuisance block, declared first
// (NicoStan samples the first declared block, with its standard normal prior, by diffusion-pathspace HMC); the
// main block is mu = (mu_CL, mu_Q, mu_V1, mu_V2, mu_ka), omega, L_Omega and sigma (5 + 5 + 10 + 1 = 21
// unconstrained coordinates).
//
// Flip-flop (identifiability): from oral data, the absorption rate ka can be exchanged with the faster
// disposition rate alpha, which gives a second, flip-flop solution. As the Torsten paper (Margossian, Zhang and
// Gillespie 2022, CPT: Pharmacometrics & Systems Pharmacology, section 3.2: "ka_pop > lambda_1") and Metrum's
// Torsten effect-compartment population example ("kaHat; // ka > lambda_1") do, the POPULATION absorption rate
// is constrained above the POPULATION alpha: exp(mu_ka) > alpha(exp(mu_CL), exp(mu_Q), exp(mu_V1), exp(mu_V2)),
// i.e. mu_ka has the lower bound two_compartment_log_alpha(mu_CL, mu_Q, mu_V1, mu_V2). This constrains only the
// population value: each patient's ka_i is only positive, so an individual patient may still have
// ka_i < alpha_i. As in both references, the prior of mu_ka is not renormalised for this bound.
//
// Priors: the Torsten paper's population priors (section 3.2) on the log scale: mu_CL ~ N(log 10, 0.25^2),
// mu_Q ~ N(log 15, 0.5^2), mu_V1 ~ N(log 35, 0.25^2), mu_V2 ~ N(log 105, 0.5^2), mu_ka ~ N(log 2.5, 1); its
// residual SD prior sigma ~ N+(0, 1); omega ~ N+(0, 0.5^2) (each between-patient SD; up to about 100% CV);
// L_Omega ~ LKJ-Cholesky(2) (mildly towards no correlation: 10 correlations from N patients); z ~ N(0, 1).
// Likelihood: log(observed concentration) ~ N(log C(t), sigma^2).
//
functions {
    //
    // log alpha, the larger disposition rate (1/h), from log CL, log Q, log V1 and log V2 (see the header):
    //
    real two_compartment_log_alpha(real log_CL, real log_Q, real log_V1, real log_V2) {
        real k10 = exp(log_CL - log_V1);
        real k12 = exp(log_Q - log_V1);
        real k21 = exp(log_Q - log_V2);
        real a = k10 + k12 - k21;
        return log(0.5 * (k10 + k12 + k21 + sqrt(square(a) + 4 * k12 * k21)));
    }
    //
    // psi1(z) = (1 - exp(-z)) / z for z >= 0 (psi1(0) = 1): below z = 0.05 its Taylor series sum_{k = 0}^{8}
    // (-z)^k / (k + 1)! in Horner form (the next term is below 1e-18; the quotient's derivative would lose
    // accuracy as z -> 0), and above it the quotient, with expm1():
    //
    real one_minus_exp_of_minus_z_over_z(real z) {
        if (z < 0.05) {
            real series = 1;
            for (j in 1:8) {
                series = 1 - z * series / (10 - j);
            }
            return series;
        }
        return -expm1(-z) / z;
    }
    //
    // D1(a, b; t) = (exp(-b t) - exp(-a t)) / (a - b) >= 0, minus the first divided difference of exp(-lambda t)
    // over the rates a, b (1/h), at time t (h), written as t exp(-min(a, b) t) psi1(|a - b| t): accurate for any
    // a, b, also a = b (where it is t exp(-a t)); the two branches give the right derivative also at a = b:
    //
    real minus_first_divided_difference_of_decay(real a, real b, real t) {
        if (a <= b) return t * exp(-a * t) * one_minus_exp_of_minus_z_over_z((b - a) * t);
        return t * exp(-b * t) * one_minus_exp_of_minus_z_over_z((a - b) * t);
    }
    //
    // E2(a, b, c; t) >= 0, the second divided difference of exp(-lambda t) over the rates a, b, c (1/h), at time
    // t (h), with the rates sorted (l0 <= l1 <= l2) and s = (l2 - l0) t:
    //   s >= 0.2: (D1(l0, l1; t) - D1(l1, l2; t)) / (l2 - l0) (both terms non-negative; the difference loses at
    //             most about one digit);
    //   s < 0.2:  its Taylor series about l0, t^2 exp(-l0 t) sum_{k = 0}^{12} (-1)^k h_k(z1, z2) / (k + 2)!, with
    //             z1 = (l1 - l0) t, z2 = (l2 - l0) t and h_k(z1, z2) = sum_{i = 0}^{k} z1^i z2^(k - i) (the next
    //             term is below 1e-18 of the sum); exact also when two or all three rates are equal:
    //
    real second_divided_difference_of_decay(real a, real b, real c, real t) {
        vector[3] rates = sort_asc([a, b, c]');
        if ((rates[3] - rates[1]) * t >= 0.2) {
            return (minus_first_divided_difference_of_decay(rates[1], rates[2], t)
                    - minus_first_divided_difference_of_decay(rates[2], rates[3], t)) / (rates[3] - rates[1]);
        }
        real z1 = (rates[2] - rates[1]) * t;
        real z2 = (rates[3] - rates[1]) * t;
        real h = 1;                     // h_0
        real z1_power = 1;              // z1^0
        real factorial = 2;             // (0 + 2)!
        real sign = 1;
        real series = 0.5;
        for (k in 1:12) {
            z1_power *= z1;
            h = z2 * h + z1_power;      // h_k = z2 h_{k - 1} + z1^k
            factorial *= k + 2;
            sign = -sign;
            series += sign * h / factorial;
        }
        return square(t) * exp(-rates[1] * t) * series;
    }
    //
    // The number of elements of x below threshold (x[i] < threshold), without a loop over the elements: rank() of
    // the threshold appended to x (the number of elements less than it):
    //
    int number_of_values_below(vector x, real threshold) {
        int n = num_elements(x);
        return rank(append_row(x, threshold), n + 1);
    }
    //
    // The vector version of one_minus_exp_of_minus_z_over_z(): psi1 of every element of z (z >= 0), the Taylor
    // series for the elements below 0.05 and the quotient for the others, each branch one whole-vector
    // calculation over its index set (sort_indices_asc(z): its first n_series entries are the elements below
    // 0.05):
    //
    vector one_minus_exp_of_minus_z_over_z(vector z) {
        int n = num_elements(z);
        int n_series = number_of_values_below(z, 0.05);
        if (n_series == 0) {
            return -expm1(-z) ./ z;
        }
        array[n] int by_z = sort_indices_asc(z);
        array[n_series] int on_series = by_z[1:n_series];
        array[n - n_series] int on_quotient = by_z[(n_series + 1):n];
        vector[n] psi1;
        vector[n_series] z_on_series = z[on_series];
        vector[n_series] series = rep_vector(1, n_series);
        for (j in 1:8) {
            series = 1 - z_on_series .* series / (10 - j);
        }
        psi1[on_series] = series;
        psi1[on_quotient] = -expm1(-z[on_quotient]) ./ z[on_quotient];
        return psi1;
    }
    //
    // The vector version of minus_first_divided_difference_of_decay(): D1(a, b; t) of every element (rates a, b
    // in 1/h, times t in h), t exp(-min(a, b) t) psi1(|a - b| t), with fmin() and abs() in place of the branch on
    // a <= b (the same values: fmin() returns the smaller rate itself, and |a - b| is b - a or a - b exactly):
    //
    vector minus_first_divided_difference_of_decay(vector a, vector b, data vector t) {
        return t .* exp(-fmin(a, b) .* t) .* one_minus_exp_of_minus_z_over_z(abs(a - b) .* t);
    }
    //
    // The vector version of second_divided_difference_of_decay(): E2(a, b, c; t) of every element (rates a, b, c
    // in 1/h, times t in h), with each element's rates sorted (l0 <= l1 <= l2) by fmin() and fmax() (the middle
    // one is max(min(a, b), min(max(a, b), c)); each is one of the rates itself, as with sort_asc()) and
    // s = (l2 - l0) t; the difference of the two D1 for the elements with s >= 0.2 and the Taylor series for the
    // others, each branch one whole-vector calculation over its index set (sort_indices_asc(s): its first
    // n_series entries are the elements with s < 0.2):
    //
    vector second_divided_difference_of_decay(vector a, vector b, vector c, data vector t) {
        int n = num_elements(t);
        vector[n] l0 = fmin(fmin(a, b), c);
        vector[n] l1 = fmax(fmin(a, b), fmin(fmax(a, b), c));
        vector[n] l2 = fmax(fmax(a, b), c);
        vector[n] s = (l2 - l0) .* t;
        int n_series = number_of_values_below(s, 0.2);
        vector[n] E2;
        array[n] int by_s = linspaced_int_array(n, 1, n);
        if (n_series > 0 && n_series < n) {
            by_s = sort_indices_asc(s);
        }
        if (n_series < n) {
            array[n - n_series] int on_difference = by_s[(n_series + 1):n];
            E2[on_difference] = (minus_first_divided_difference_of_decay(l0[on_difference], l1[on_difference],
                                                                         t[on_difference])
                                 - minus_first_divided_difference_of_decay(l1[on_difference], l2[on_difference],
                                                                           t[on_difference]))
                                ./ (l2[on_difference] - l0[on_difference]);
        }
        if (n_series > 0) {
            array[n_series] int on_series = by_s[1:n_series];
            vector[n_series] z1 = (l1[on_series] - l0[on_series]) .* t[on_series];
            vector[n_series] z2 = s[on_series];
            vector[n_series] h = rep_vector(1, n_series);            // h_0
            vector[n_series] z1_power = rep_vector(1, n_series);     // z1^0
            real factorial = 2;                                      // (0 + 2)!
            real sign = 1;
            vector[n_series] series = rep_vector(0.5, n_series);
            for (k in 1:12) {
                z1_power = z1_power .* z1;
                h = z2 .* h + z1_power;                              // h_k = z2 h_{k - 1} + z1^k
                factorial *= k + 2;
                sign = -sign;
                series += sign * h / factorial;
            }
            E2[on_series] = square(t[on_series]) .* exp(-l0[on_series] .* t[on_series]) .* series;
        }
        return E2;
    }
    //
    // The samples of the patients given (patient i's samples are the contiguous run first_sample[i] to
    // last_sample[i]), with the runs laid end to end in the order of the patients given: row 1, each sample;
    // row 2, its patient (1 to the number of patients given). No loop over the patients or the samples: the
    // patient of each sample is the cumulative sum of a 0 / 1 marker of the first sample of each run, and each
    // sample is its patient's first sample plus its place in the run (whole-array integer differences, with
    // to_int()):
    //
    array[,] int samples_of_patients(data array[] int first_sample, data array[] int last_sample) {
        int n_patients = size(first_sample);
        array[n_patients] int n_samples_of_patient = to_int(to_array_1d(to_vector(last_sample)
                                                                        - to_vector(first_sample) + 1));
        int n_samples = sum(n_samples_of_patient);
        array[n_patients] int end_of_run = cumulative_sum(n_samples_of_patient);
        array[n_patients] int start_of_run = to_int(to_array_1d(to_vector(end_of_run)
                                                                - to_vector(n_samples_of_patient) + 1));
        array[n_samples] int marker_of_start_of_run = rep_array(0, n_samples);
        array[2, n_samples] int samples_and_their_patients;
        marker_of_start_of_run[start_of_run] = rep_array(1, n_patients);
        samples_and_their_patients[2] = cumulative_sum(marker_of_start_of_run);
        samples_and_their_patients[1] =
              to_int(to_array_1d(linspaced_vector(n_samples, 1, n_samples)
                                 + to_vector(first_sample[samples_and_their_patients[2]])
                                 - to_vector(start_of_run[samples_and_their_patients[2]])));
        return samples_and_their_patients;
    }
    //
    // The predicted concentrations (mg/L) of all samples (sample j: patient patient[j], time time[j] (h); dose in
    // mg), from every patient's log CL, log Q, log V1, log V2 and log ka (the rows of log_theta; one column per
    // patient): the tri-exponential for every patient whose rates ka, alpha, beta all differ by at least
    // minimum_rate_separation, with every exp() taken over all samples at once, and the stable form (header) for
    // the other patients (their tri-exponential coefficients are set to 0, so that no division by a small
    // difference is ever made, and their samples are then replaced):
    //
    //
//  vector two_compartment_oral_concentrations(matrix log_theta, data real dose, data array[] int patient,
//                                             data vector time, data array[] int first_sample,
//                                             data array[] int last_sample, data real minimum_rate_separation) {
//      int N = cols(log_theta);
//      vector[N] k10 = exp((log_theta[1] - log_theta[3])');
//      vector[N] k12 = exp((log_theta[2] - log_theta[3])');
//      vector[N] k21 = exp((log_theta[2] - log_theta[4])');
//      vector[N] ka = exp(log_theta[5]');
//      vector[N] dose_ka_over_V1 = dose * exp((log_theta[5] - log_theta[3])');
//      vector[N] a = k10 + k12 - k21;
//      vector[N] alpha_minus_beta = sqrt(square(a) + 4 * k12 .* k21);
//      vector[N] alpha = 0.5 * (k10 + k12 + k21 + alpha_minus_beta);
//      vector[N] beta = k10 .* k21 ./ alpha;
//      vector[N] k21_minus_beta;
//      vector[N] coefficient_alpha;
//      vector[N] coefficient_beta;
//      vector[N] coefficient_ka;
//      array[N] int uses_stable_form;
//      // for (i in 1:N) {
//      //     // k21 - beta = (R - a) / 2, written without cancellation for either sign of a (R = alpha - beta):
//      //     k21_minus_beta[i] = a[i] > 0 ? 2 * k12[i] * k21[i] / (alpha_minus_beta[i] + a[i])
//      //                                  : 0.5 * (alpha_minus_beta[i] - a[i]);
//      //     uses_stable_form[i] = fmin(fmin(abs(ka[i] - alpha[i]), abs(ka[i] - beta[i])), alpha_minus_beta[i])
//      //                           < minimum_rate_separation;
//      //     if (uses_stable_form[i]) {
//      //         coefficient_alpha[i] = 0;
//      //         coefficient_beta[i] = 0;
//      //         coefficient_ka[i] = 0;
//      //     } else {
//      //         // alpha - k21 = k12 k21 / (k21 - beta) (Vieta: (alpha - k21) (k21 - beta) = k12 k21)
//      //         coefficient_alpha[i] = k12[i] * k21[i] / k21_minus_beta[i]
//      //                                / ((ka[i] - alpha[i]) * alpha_minus_beta[i]);
//      //         coefficient_beta[i] = k21_minus_beta[i] / ((ka[i] - beta[i]) * alpha_minus_beta[i]);
//      //         coefficient_ka[i] = (k21[i] - ka[i]) / ((ka[i] - alpha[i]) * (ka[i] - beta[i]));
//      //     }
//      // }
//      // (vectorised over the patients, with the same values as the loop above: each branch is one vectorised
//      //  operation over the index set of its patients (the sign of a; the tri-exponential or the stable form),
//      //  ka - alpha and ka - beta are computed once, and the stable-form test compares the differences with
//      //  +/- minimum_rate_separation directly, so that the test adds no autodiff node)
//      vector[N] ka_minus_alpha = ka - alpha;
//      vector[N] ka_minus_beta = ka - beta;
//      int n_a_positive = 0;
//      int n_tri_exponential = 0;
//      for (i in 1:N) {
//          // min(|ka - alpha|, |ka - beta|, alpha - beta) < minimum_rate_separation, as with fmin() and abs()
//          uses_stable_form[i] = (ka_minus_alpha[i] < minimum_rate_separation
//                                 && ka_minus_alpha[i] > -minimum_rate_separation)
//                                || (ka_minus_beta[i] < minimum_rate_separation
//                                    && ka_minus_beta[i] > -minimum_rate_separation)
//                                || alpha_minus_beta[i] < minimum_rate_separation;
//          n_a_positive += a[i] > 0;
//          n_tri_exponential += 1 - uses_stable_form[i];
//      }
//      array[n_a_positive] int a_positive;
//      array[N - n_a_positive] int a_not_positive;
//      array[n_tri_exponential] int tri_exponential;
//      {
//          int k_a_positive = 0;
//          int k_a_not_positive = 0;
//          int k_tri_exponential = 0;
//          for (i in 1:N) {
//              if (a[i] > 0) {
//                  k_a_positive += 1;
//                  a_positive[k_a_positive] = i;
//              } else {
//                  k_a_not_positive += 1;
//                  a_not_positive[k_a_not_positive] = i;
//              }
//              if (uses_stable_form[i] == 0) {
//                  k_tri_exponential += 1;
//                  tri_exponential[k_tri_exponential] = i;
//              }
//          }
//      }
//      // k21 - beta = (R - a) / 2, written without cancellation for either sign of a (R = alpha - beta):
//      k21_minus_beta[a_positive] = (2 * k12[a_positive]) .* k21[a_positive]
//                                   ./ (alpha_minus_beta[a_positive] + a[a_positive]);
//      k21_minus_beta[a_not_positive] = 0.5 * (alpha_minus_beta[a_not_positive] - a[a_not_positive]);
//      coefficient_alpha = rep_vector(0, N);
//      coefficient_beta = rep_vector(0, N);
//      coefficient_ka = rep_vector(0, N);
//      // alpha - k21 = k12 k21 / (k21 - beta) (Vieta: (alpha - k21) (k21 - beta) = k12 k21)
//      coefficient_alpha[tri_exponential] = k12[tri_exponential] .* k21[tri_exponential]
//                                           ./ k21_minus_beta[tri_exponential]
//                                           ./ (ka_minus_alpha[tri_exponential]
//                                               .* alpha_minus_beta[tri_exponential]);
//      coefficient_beta[tri_exponential] = k21_minus_beta[tri_exponential]
//                                          ./ (ka_minus_beta[tri_exponential]
//                                              .* alpha_minus_beta[tri_exponential]);
//      coefficient_ka[tri_exponential] = (k21[tri_exponential] - ka[tri_exponential])
//                                        ./ (ka_minus_alpha[tri_exponential] .* ka_minus_beta[tri_exponential]);
//      // vector[num_elements(time)] concentration =
//      //       dose_ka_over_V1[patient] .* (coefficient_alpha[patient] .* exp(-alpha[patient] .* time)
//      //                                    + coefficient_beta[patient] .* exp(-beta[patient] .* time)
//      //                                    + coefficient_ka[patient] .* exp(-ka[patient] .* time));
//      // (the minus sign on the data, -time, in place of on the rates: the same products, bit for bit, without
//      //  three autodiff negations of length n_samples)
//      vector[num_elements(time)] concentration =
//            dose_ka_over_V1[patient] .* (coefficient_alpha[patient] .* exp(-time .* alpha[patient])
//                                         + coefficient_beta[patient] .* exp(-time .* beta[patient])
//                                         + coefficient_ka[patient] .* exp(-time .* ka[patient]));
//      for (i in 1:N) {
//          if (uses_stable_form[i]) {
//              for (j in first_sample[i]:last_sample[i]) {
//                  concentration[j] = dose_ka_over_V1[i]
//                                     * (k21_minus_beta[i]
//                                        * second_divided_difference_of_decay(alpha[i], beta[i], ka[i], time[j])
//                                        + minus_first_divided_difference_of_decay(alpha[i], ka[i], time[j]));
//              }
//          }
//      }
//      return concentration;
//  }
    //
    // (the version above, with its loops over the patients and their samples, commented out; the same function
    //  without them: the tri-exponential coefficients of every patient are one whole-vector calculation in the
    //  patients' order, with the rate differences of the stable-form patients set to 1 before the divisions (so
    //  that no division by a small difference is ever made) and their coefficients then set to 0, and so is the
    //  first form of k21 - beta (R + a set to 1 for the patients with a < 0, whose k21 - beta is then the second
    //  form); the index sets of the stable-form patients and of the patients with a < 0 come from
    //  number_of_values_below() and, only when there are values on both sides of the threshold,
    //  sort_indices_asc(); and the stable form is computed for all the samples of the stable-form patients at
    //  once)
    vector two_compartment_oral_concentrations(matrix log_theta, data real dose, data array[] int patient,
                                               data vector time, data array[] int first_sample,
                                               data array[] int last_sample, data real minimum_rate_separation) {
        int N = cols(log_theta);
        vector[N] k10 = exp((log_theta[1] - log_theta[3])');
        vector[N] k12 = exp((log_theta[2] - log_theta[3])');
        vector[N] k21 = exp((log_theta[2] - log_theta[4])');
        vector[N] ka = exp(log_theta[5]');
        vector[N] dose_ka_over_V1 = dose * exp((log_theta[5] - log_theta[3])');
        vector[N] a = k10 + k12 - k21;
        vector[N] alpha_minus_beta = sqrt(square(a) + 4 * k12 .* k21);
        vector[N] alpha = 0.5 * (k10 + k12 + k21 + alpha_minus_beta);
        vector[N] beta = k10 .* k21 ./ alpha;
        vector[N] k21_minus_beta;
        vector[N] coefficient_alpha;
        vector[N] coefficient_beta;
        vector[N] coefficient_ka;
        vector[N] ka_minus_alpha = ka - alpha;
        vector[N] ka_minus_beta = ka - beta;
        // min(|ka - alpha|, |ka - beta|, alpha - beta): the stable form for the patients with this below
        // minimum_rate_separation (fmin() and abs() return the rates' differences themselves, or their negations)
        vector[N] smallest_rate_separation = fmin(fmin(abs(ka_minus_alpha), abs(ka_minus_beta)),
                                                  alpha_minus_beta);
        int n_stable_form = number_of_values_below(smallest_rate_separation, minimum_rate_separation);
        int n_a_negative = number_of_values_below(a, 0);
        array[N] int by_smallest_rate_separation = linspaced_int_array(N, 1, N);
        array[N] int by_a = linspaced_int_array(N, 1, N);
        if (n_stable_form > 0 && n_stable_form < N) {
            by_smallest_rate_separation = sort_indices_asc(smallest_rate_separation);
        }
        if (n_a_negative > 0 && n_a_negative < N) {
            by_a = sort_indices_asc(a);
        }
        array[n_stable_form] int stable_form = by_smallest_rate_separation[1:n_stable_form];
        array[n_a_negative] int a_negative = by_a[1:n_a_negative];
        // array[N - n_stable_form] int tri_exponential = by_smallest_rate_separation[(n_stable_form + 1):N];
        // array[N - n_a_negative] int a_not_negative = by_a[(n_a_negative + 1):N];
        // // k21 - beta = (R - a) / 2, written without cancellation for either sign of a (R = alpha - beta; at
        // // a = 0 both forms are R / 2):
        // k21_minus_beta[a_not_negative] = (2 * k12[a_not_negative]) .* k21[a_not_negative]
        //                                  ./ (alpha_minus_beta[a_not_negative] + a[a_not_negative]);
        // k21_minus_beta[a_negative] = 0.5 * (alpha_minus_beta[a_negative] - a[a_negative]);
        // coefficient_alpha = rep_vector(0, N);
        // coefficient_beta = rep_vector(0, N);
        // coefficient_ka = rep_vector(0, N);
        // // alpha - k21 = k12 k21 / (k21 - beta) (Vieta: (alpha - k21) (k21 - beta) = k12 k21)
        // coefficient_alpha[tri_exponential] = k12[tri_exponential] .* k21[tri_exponential]
        //                                      ./ k21_minus_beta[tri_exponential]
        //                                      ./ (ka_minus_alpha[tri_exponential]
        //                                          .* alpha_minus_beta[tri_exponential]);
        // coefficient_beta[tri_exponential] = k21_minus_beta[tri_exponential]
        //                                     ./ (ka_minus_beta[tri_exponential]
        //                                         .* alpha_minus_beta[tri_exponential]);
        // coefficient_ka[tri_exponential] = (k21[tri_exponential] - ka[tri_exponential])
        //                                   ./ (ka_minus_alpha[tri_exponential]
        //                                       .* ka_minus_beta[tri_exponential]);
        // (the same values for every patient in the patients' order, without gathering the tri-exponential
        //  patients, whose index set came out of sort_indices_asc() in the order of their rates' separations:
        //  the denominators of the stable-form patients, and R + a of the patients with a < 0, set to 1, so that
        //  their values, overwritten below, are finite and their autodiff nodes (with adjoints 0) give no NaN)
        vector[N] ka_minus_alpha_in_divisions = ka_minus_alpha;
        vector[N] ka_minus_beta_in_divisions = ka_minus_beta;
        vector[N] alpha_minus_beta_in_divisions = alpha_minus_beta;
        vector[N] alpha_minus_beta_plus_a_in_divisions = alpha_minus_beta + a;
        ka_minus_alpha_in_divisions[stable_form] = rep_vector(1, n_stable_form);
        ka_minus_beta_in_divisions[stable_form] = rep_vector(1, n_stable_form);
        alpha_minus_beta_in_divisions[stable_form] = rep_vector(1, n_stable_form);
        alpha_minus_beta_plus_a_in_divisions[a_negative] = rep_vector(1, n_a_negative);
        // k21 - beta = (R - a) / 2, written without cancellation for either sign of a (R = alpha - beta; at a = 0
        // both forms are R / 2):
        k21_minus_beta = (2 * k12) .* k21 ./ alpha_minus_beta_plus_a_in_divisions;
        k21_minus_beta[a_negative] = 0.5 * (alpha_minus_beta[a_negative] - a[a_negative]);
        // alpha - k21 = k12 k21 / (k21 - beta) (Vieta: (alpha - k21) (k21 - beta) = k12 k21)
        coefficient_alpha = k12 .* k21 ./ k21_minus_beta
                            ./ (ka_minus_alpha_in_divisions .* alpha_minus_beta_in_divisions);
        coefficient_beta = k21_minus_beta ./ (ka_minus_beta_in_divisions .* alpha_minus_beta_in_divisions);
        coefficient_ka = (k21 - ka) ./ (ka_minus_alpha_in_divisions .* ka_minus_beta_in_divisions);
        coefficient_alpha[stable_form] = rep_vector(0, n_stable_form);
        coefficient_beta[stable_form] = rep_vector(0, n_stable_form);
        coefficient_ka[stable_form] = rep_vector(0, n_stable_form);
        vector[num_elements(time)] concentration =
              dose_ka_over_V1[patient] .* (coefficient_alpha[patient] .* exp(-time .* alpha[patient])
                                           + coefficient_beta[patient] .* exp(-time .* beta[patient])
                                           + coefficient_ka[patient] .* exp(-time .* ka[patient]));
        // the stable form (header), for all the samples of the stable-form patients at once (their
        // tri-exponential coefficients are 0, so their samples above are 0 and are replaced here):
        if (n_stable_form > 0) {
            int n_stable_form_samples = sum(last_sample[stable_form]) - sum(first_sample[stable_form])
                                        + n_stable_form;
            array[2, n_stable_form_samples] int samples_and_their_patients =
                  samples_of_patients(first_sample[stable_form], last_sample[stable_form]);
            array[n_stable_form_samples] int sample_on_stable_form = samples_and_their_patients[1];
            array[n_stable_form_samples] int patient_on_stable_form = stable_form[samples_and_their_patients[2]];
            concentration[sample_on_stable_form] =
                  dose_ka_over_V1[patient_on_stable_form]
                  .* (k21_minus_beta[patient_on_stable_form]
                      .* second_divided_difference_of_decay(alpha[patient_on_stable_form],
                                                            beta[patient_on_stable_form],
                                                            ka[patient_on_stable_form],
                                                            time[sample_on_stable_form])
                      + minus_first_divided_difference_of_decay(alpha[patient_on_stable_form],
                                                                ka[patient_on_stable_form],
                                                                time[sample_on_stable_form]));
        }
        return concentration;
    }
}
data {
    int<lower=1> N;                                    // number of patients
    int<lower=1> n_samples;                            // number of concentration samples (all patients)
    array[n_samples] int<lower=1, upper=N> patient;    // patient of each sample (samples sorted by patient)
    vector<lower=0>[n_samples] time;                   // sampling time after the dose (h), > 0
    vector<lower=0>[n_samples] concentration_observed; // observed plasma concentration (mg/L), > 0
    real<lower=0> dose;                                // the oral dose of every patient, at time 0 (mg)
}
transformed data {
    vector[n_samples] log_concentration_observed = log(concentration_observed);
    // each patient's samples: first_sample[i] to last_sample[i]
    array[N] int first_sample = rep_array(0, N);
    array[N] int last_sample = rep_array(0, N);
    // the smallest rate difference of the tri-exponential (see the header): d t >= 0.01 at every sample
    real minimum_rate_separation = 0.01 / min(time);
    if (min(time) <= 0) reject("every sampling time must be > 0");
    // for (j in 1:n_samples) {
    //     if (j > 1 && patient[j] < patient[j - 1]) reject("the samples must be sorted by patient");
    //     if (first_sample[patient[j]] == 0) first_sample[patient[j]] = j;
    //     last_sample[patient[j]] = j;
    // }
    // for (i in 1:N) {
    //     if (first_sample[i] == 0) reject("patient ", i, " has no samples");
    // }
    // (the same checks and the same first_sample and last_sample without a loop: the steps of the samples'
    //  patients from one sample to the next must be 0 or 1, from patient 1 to patient N, i.e. sorted and every
    //  patient with samples; then a patient's first sample is a sample whose step from the one before is 1 (or
    //  the first sample), and its last sample one whose step to the next is 1 (or the last sample): the indices
    //  of the N ones of these 0 / 1 vectors, from sort_indices_desc(), sorted)
    vector[n_samples - 1] patient_step = to_vector(patient[2:n_samples]) - to_vector(patient[1:(n_samples - 1)]);
    if (n_samples > 1 && min(patient_step) < 0) reject("the samples must be sorted by patient");
    if (patient[1] != 1) reject("patient ", 1, " has no samples");
    if (patient[n_samples] != N) reject("patient ", N, " has no samples");
    if (n_samples > 1 && max(patient_step) > 1) {
        reject("a patient between ", patient[sort_indices_desc(patient_step)[1]], " and ",
               patient[sort_indices_desc(patient_step)[1] + 1], " has no samples");
    }
    first_sample = sort_asc(sort_indices_desc(append_row(1, patient_step))[1:N]);
    last_sample = sort_asc(sort_indices_desc(append_row(patient_step, 1))[1:N]);
}
parameters {
    matrix[5, N] z;                                    // standard normal nuisance block, declared first: patient
                                                       // i's 5 values in column i
    real mu_CL;                                        // population mean of log CL (CL in L/h)
    real mu_Q;                                         // population mean of log Q (L/h)
    real mu_V1;                                        // population mean of log V1 (L)
    real mu_V2;                                        // population mean of log V2 (L)
    real<lower=two_compartment_log_alpha(mu_CL, mu_Q, mu_V1, mu_V2)> mu_ka; // population mean of log ka (1/h):
                                                       // ka_pop > alpha_pop (flip-flop, see the header)
    vector<lower=0>[5] omega;                          // between-patient SDs of log CL, log Q, log V1, log V2,
                                                       // log ka
    cholesky_factor_corr[5] L_Omega;                   // Cholesky factor of their correlation matrix
    real<lower=0> sigma;                               // residual SD of the log concentrations
}
model {
    vector[5] mu = [mu_CL, mu_Q, mu_V1, mu_V2, mu_ka]';
    // the patients' log CL, log Q, log V1, log V2, log ka (one column per patient), non-centred
    matrix[5, N] log_theta = rep_matrix(mu, N) + diag_pre_multiply(omega, L_Omega) * z;

    // Priors
    mu_CL ~ normal(log(10), 0.25);
    mu_Q ~ normal(log(15), 0.5);
    mu_V1 ~ normal(log(35), 0.25);
    mu_V2 ~ normal(log(105), 0.5);
    mu_ka ~ normal(log(2.5), 1);
    omega ~ normal(0, 0.5);
    L_Omega ~ lkj_corr_cholesky(2);
    sigma ~ normal(0, 1);

    // Equivalent non-centred form of log_theta[, i] ~ multi_normal(mu, Sigma).
    to_vector(z) ~ std_normal();

    // Likelihood, on the whole vector of samples
    log_concentration_observed ~ normal(log(two_compartment_oral_concentrations(log_theta, dose, patient, time,
                                                                                first_sample, last_sample,
                                                                                minimum_rate_separation)),
                                        sigma);
}






















