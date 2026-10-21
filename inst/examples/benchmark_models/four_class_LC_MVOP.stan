// LC_MVOP_4class_joint_v1_reduce_sum.stan
//
// WITHIN-CHAIN PARALLELISM / CHUNKED-TAPE variant (reduce_sum_static) of LC_MVOP_4class_joint_v1.
// Same model, same log-density. Structural differences only:
//   (1) u_raw is array[N] row_vector[n_tests] so reduce_sum_static slices the nuisance parameters;
//   (2) the likelihood is a plain partial-sum function; the u tanh-Jacobian is RETURNED, not jacobian +=;
//   (3) no log_lik in transformed parameters - target is incremented once, inside reduce_sum_static;
//   (4) observed / missing index sets are built per slice inside the partial function;
//   (5) chunk_size (data) bounds the slice size; R defaults to ceil(N / num_chunks). With threads_per_chain = 1 this is "chunking":
//       the tape is sliced without parallelism, which is what relieves autodiff's memory
//       bandwidth. Compile with cpp_options = list(stan_threads = TRUE).
//   (6) inside the chunks (reduce_sum_static over the subjects, slices of at most chunk_size): each slice's log
//       class probabilities (log_prev_for_slice, from its own rows of the prevalence covariates), the u
//       tanh-Jacobian and the GHK likelihood, vectorised over the slice's subjects at every step of the
//       recursion over the tests. The four sums over the standardisation profiles of the use_cond_prior = 1
//       prior are tape-chunked the same way (reduce_sum_static over the profiles, in transformed parameters,
//       partial_sum_of_standardisation_profiles_for_conditional_prior), and so is the conditional-independence
//       shortcut (all correlations zero, no test covariates: partial_log_lik_conditional_independence_shortcut),
//       so that outside the chunks the main tape holds only the transformed parameters and priors whose size
//       does not grow with N, and one node per reduce_sum_static call.
//
//
// Four-class Latent Class Multivariate Ordinal Probit for JOINT diagnostic accuracy
// across TWO conditions (A and B).
//
// Extends LC_MVOP_PartialLog_v2.stan from 2 classes to a 2x2 factorial latent:
//     c = 1 -> (D_A = 0, D_B = 0)
//     c = 2 -> (D_A = 1, D_B = 0)
//     c = 3 -> (D_A = 0, D_B = 1)
//     c = 4 -> (D_A = 1, D_B = 1)
// The class layout is passed as data (class_D) so it is explicit rather than implied.
//
// Primary estimand: pi_11 = Pr(D_A = 1, D_B = 1), reported directly as prev[pop][4].
//
// ---------------------------------------------------------------------------------
// FOUR DIFFERENCES FROM THE 2-CLASS FILE. Read these before debugging anything.
// ---------------------------------------------------------------------------------
//
// (1) beta[c, t] IS A matrix[n_class, n_tests], AS IN THE 2-CLASS FILE - built from:
//     beta[c, t] = beta_own[s_own(c,t), t] + delta_cross[t] * D_other(c, t)
//     where s_own(c,t) in {1,2} is the status (absent/present) of the condition test t is
//     meant to measure in class c, and D_other is the status of the other condition.
//       - beta_own[2, t] > beta_own[1, t] by construction (declared lower=). This is the
//         2-class file's truncation trick applied per test to the OWN-condition pair, and
//         it PINS THE CLASS LABELS for BINARY tests. (For ORDINAL tests beta_own is 0 and
//         the labels are pinned by the cutpoint ordering - see (2).)
//       - delta_cross[t] is the CROSS-LOADING (item overlap): an A-test responding to B.
//         With four free beta[c, t] per test you would be estimating it whether you wanted
//         to or not; writing it as delta_cross lets it be switched OFF, which is the
//         assumption models M2-M4b make and M4c relaxes. allow_cross_loading[t] = 0 forces
//         it to zero.
//     The likelihood, the GQ and the C++ port all see beta[c, t] and nothing else.
//
// (2) CUTPOINTS ARE INDEXED BY THE TEST'S OWN-CONDITION STATUS, C_{t,k}^{[d_{c(t)}]}.
//     Two cutpoint sets per ordinal test: one for subjects WITHOUT the test's target
//     condition (s = 1), one for those WITH it (s = 2). A test for autism uses the set
//     chosen by the subject's autism status; a test for BPD by their BPD status. The
//     cutpoints do NOT vary with the OTHER condition - so the cross-loading has nowhere
//     to hide except delta_cross, which is the point. This is the direct generalisation of
//     the 2-class file (cutpoints by disease status) and preserves its class-specific
//     shape; delta_own is prior-identified against the own-status cutpoint shift exactly
//     as the 2-class beta is, via the normal prior and the fixed induced-Dirichlet anchor.
//     C_vec is therefore array[2], as in the 2-class file.
//
//     IDENTIFICATION (v5, after the first smoke test): with ALL tests ordinal and
//     fully free own-status cutpoints, delta_own > 0 does NOT pin the class labels - the present-
//     status cutpoints can shift up by more than delta_own and reverse the direction, and on the
//     first smoke test they did (Se < 1 - Sp on the ADOS). The 2-class file does not hit this
//     because its BINARY reference has a threshold fixed at 0 shared across classes, which pins
//     the class; here nothing does.
//     Fix (v6): STOCHASTIC ORDERING of the two cutpoint sets. For an ORDINAL test the own-
//     status effect lives ENTIRELY in the cutpoints - alpha and delta_own are NOT in its mean
//     (they would be flat directions against the cutpoints, which is exactly what let the
//     labels swap) - and the present set is a non-negative, ordering-preserving shift BELOW
//     the absent set:
//            C_{t,k}^{[1]} = C_{t,k}^{[0]} - d_{t,k},   d_{t,k} >= 0,   C^{[1]} still increasing in k,
//     so that, with the OTHER condition absent and covariates at the zero anchor, present scores higher
//     at every threshold. Own-status-specific cross-loadings can change this when the other condition
//     is present; marginal standardised ROC curves need not obey the same ordering. That is the standard ordinal-DTA assumption, it is what delta_own > 0
//     provides for free when cutpoints are shared, and it is what a binary reference provides in
//     the 2-class file. The shape (spacing) of the two sets is otherwise free. alpha and
//     delta_own remain for BINARY tests only, where there is no cutpoint to absorb them.
//     Both induced-Dirichlet priors are anchored at 0 for ordinal tests.
//
// (3) PREVALENCE IS A SIMPLEX OVER 4 CLASSES, with a Dirichlet prior, replacing the
//     single Beta-distributed prevalence. pi_11, the marginal prevalences and the
//     comorbidity odds ratio are all derived in generated quantities.
//
// (4) Covariates are subject properties. Slopes can be shared across classes or indexed
// by own-condition status. Cutpoint ordering pins the X=0, other-absent anchor;
// different slopes can change effective ordering at other covariate profiles.
// Prevalence slopes are shared across groups. Population summaries use the
// exact standardisation profiles and their supplied phase-one weights.
//
// ---------------------------------------------------------------------------------
// HOW TO GET MODELS 1-4 OUT OF THIS ONE FILE (all by data, no code changes)
// ---------------------------------------------------------------------------------
//   Model 2 (perfect refs, conditional independence):
//       is_perfect_ref = 1 for the two reference tests, 0 for the screens
//       known_values_indicator_list[c][i, j] = 1 and known_values_list[c][i, j] = 0
//         for every off-diagonal (i, j) -> Omega fixed at the identity
//       allow_cross_loading = 0 for all tests
//   Model 3 (perfect refs, screen-screen dependence):
//       as model 2, but leave the screen-screen entry of Omega free
//   Model 4 (imperfect refs, dependence):
//       is_perfect_ref = 0 for all tests; free the three substantive correlations
//         (screen-screen, and each screen with its own reference); fix the rest at 0
//   Model 1 (the naive estimator) is NOT fitted here - it is computed directly from the
//       observed 2x2 tables in R. See the simulation spec.
//
// ---------------------------------------------------------------------------------
// v4: OBSERVED-ONLY COMPUTATION. For each test the partition-and-branch machinery runs only over
// the subjects who have that test (index sets built per reduce_sum_static slice in this variant);
// missing subjects get only the cheap untruncated draw inv_Phi(u) that the conditioning chain
// needs, and y1 = 0. The old "compute everything for all N then override" path is gone. Each
// partition group now carries its own Bound_Z values alongside its subject ids, so the branch
// bodies are unchanged in substance but never touch a missing cell.
//
// v3: PARTIAL VERIFICATION / MISSING TEST RESULTS via a NEGATIVE SENTINEL in y
// ---------------------------------------------------------------------------------
// y[n, t] < 0 (use -1) means test t was not administered to subject n. The observation mask is
// DERIVED from this in transformed data; there is no separate mask to pass or keep in sync. Such
// a subject still contributes everything else they have - which is the entire point of the
// two-phase arm, since the unverified majority carry information through their screens alone.
//
// The correct treatment is NOT merely to zero the likelihood contribution. In this sequential
// (GHK-style) parameterisation, Z_std_norm[n, t] also feeds the conditional mean adjustment
// "inc" for every subsequent test, so an unobserved test must contribute an UNTRUNCATED draw:
//        Phi_lo = 0, Phi_hi = 1  ->  Phi_Z = u  ->  Z_std_norm = inv_Phi(u),  y1 = 0
// i.e. exactly the limit of an ordinal response whose observed category spans the whole scale.
// Zeroing y1 alone would leave the conditioning wrong for the tests that follow.
//
// VALIDITY: masking integrates the missing reference out correctly provided selection into
// phase two depends only on quantities that are themselves in the likelihood (here, the screen
// scores). That is missing-at-random given the observed data. If selection depended on anything
// unobserved, this is not valid and no amount of modelling recovers it. State the assumption.
//
// A perfect reference must be BINARY: the latent status is binary, so a reference that equals it
// cannot be ordinal without first choosing a dichotomisation. is_perfect_ref[t] = 1 is therefore
// only meaningful for t <= n_binary_tests. Ordinal references are supported - just set
// n_binary_tests = 0, n_ordinal_tests = 4 and is_perfect_ref = 0 throughout.
//
// Integrated with the APMS application revisions. See APMS_INTEGRATION.md for checks and scope.

functions {

      //////////////////////////////////////////////////////////////////////////
      // ---- Unchanged from the 2-class file:
      //////////////////////////////////////////////////////////////////////////
      vector lb_ub_jacobian( vector y,
                             real lb,
                             real ub) {
            int N = num_elements(y);
            vector[N] tanh_y;
            tanh_y = tanh(y);
            jacobian += log(ub - lb) - log2() + 2.0 * (log2() - y - log1p_exp(-2.0 * y));
            return lb + (ub - lb) * 0.5 * (1.0 + tanh(y));
      }
      real lb_ub_jacobian( real y,
                           real lb,
                           real ub) {
            jacobian += log(ub - lb) - log2() + 2.0 * (log2() - y - log1p_exp(-2.0 * y));
            return lb + (ub - lb) * 0.5 * (1.0 + tanh(y));
      }
      vector lb_ub_jacobian( vector y,
                             vector lb,
                             vector ub) {
            int N = num_elements(y);
            vector[N] tanh_y;
            tanh_y = tanh(y);
            jacobian += sum(log(ub - lb)) - num_elements(y) * log2() + 2.0 * sum(log2() - y - log1p_exp(-2.0 * y));
            return lb + (ub - lb) .* (0.5 * (1.0 + tanh(y)));
      }
      real lb_ub(real y, real lb, real ub) {
            return lb + (ub - lb) * 0.5 * (1.0 + tanh(y));
      }
      vector lb_ub(vector y, real lb, real ub) {
            return lb + (ub - lb) * 0.5 * (1.0 + tanh(y));
      }

      ////
      //// Sean Pinkney's LDL-with-bounds construction (unchanged):
      ////
      matrix Pinkney_LDL_bounds_opt_jacobian(  vector Omega_raw_vec,
                                               matrix lb_corr,
                                               matrix ub_corr,
                                               matrix known_values_indicator,
                                               matrix known_values) {

            int dim = rows(lb_corr);
            int counter = 1;
            matrix[dim, dim] L = diag_matrix(rep_vector(1.0, dim));
            vector[dim] D;
            D[1] = 1.0;
            ////
            //// Only FREE entries have a raw parameter. Order: lower triangle, row by row.
            //// A fixed entry therefore has neither an unused parameter nor a transform Jacobian.
            ////
            for (i in 2:dim) {
                  real remainder = 1.0;
                  for (j in 1:(i - 1)) {
                        real b1 = 0.0;
                        if (j > 1)
                              b1 = dot_product(L[j, 1:(j - 1)], D[1:(j - 1)]' .* L[i, 1:(j - 1)]);
                        //// x = Omega[i,j] - b1 = D[j] * L[i,j]. Its PD bound is sqrt(remainder * D[j]).
                        real bound = sqrt(remainder * D[j]);
                        real x;
                        if (known_values_indicator[i, j] == 1) {
                              x = known_values[i, j] - b1;
                              if (abs(x) >= bound)
                                    reject("fixed correlation is incompatible with positive definiteness at (", i, ",", j, ")");
                        } else {
                              real low = fmax(-bound, lb_corr[i, j] - b1);
                              real up  = fmin( bound, ub_corr[i, j] - b1);
                              if (low >= up) reject("empty correlation interval at (", i, ",", j, ")");
                              x = lb_ub_jacobian(Omega_raw_vec[counter], low, up);
                              counter += 1;
                              //// raw -> x -> Cholesky entry, L_chol[i,j] = x / sqrt(D[j]).
                              jacobian += -0.5 * log(D[j]);
                        }
                        L[i, j] = x / D[j];
                        remainder -= square(x) / D[j];
                  }
                  D[i] = remainder;
            }
            return diag_post_multiply(L, sqrt(D));
      }

      real inv_Phi_approx_from_prob(real p) {
            return 5.494448514153059 *  sinh(0.33333333333333331483 * asinh( 0.34176618822627863 * logit(p)  )) ;
      }
      vector inv_Phi_approx_from_prob(vector p) {
            return 5.494448514153059 *  sinh(0.33333333333333331483 * asinh( 0.34176618822627863 * logit(p)  )) ;
      }
      real inv_Phi_approx_from_logit_prob(real logit_p) {
            return 5.494448514153059 *  sinh(0.33333333333333331483 * asinh( 0.34176618822627863 * logit_p  )) ;
      }
      vector inv_Phi_approx_from_logit_prob(vector logit_p) {
            return 5.494448514153059 *  sinh(0.33333333333333331483 * asinh( 0.34176618822627863 *logit_p  )) ;
      }

      vector rowwise_sum(matrix M) {
            return M * rep_vector(1.0, cols(M));
      }
      vector rowwise_max(matrix M) {
            int N = rows(M);
            vector[N] rowwise_maxes;
//          for (n in 1:N) rowwise_maxes[n] = max(M[n, ]);
            //// (no loop over the rows: fmax of whole columns, in a loop over the columns)
            rowwise_maxes = M[, 1];
            for (j in 2:cols(M)) rowwise_maxes = fmax(rowwise_maxes, M[, j]);
            return rowwise_maxes;
      }
      ////
      //// Kept for the 2-column uses inside the numerical-handling branches:
      ////
      //// (any number of columns: also the 4-class mixtures of partial_log_lik() and ci_shared_log_lik(); in the
      ////  AVX2 and AVX-512 versions its calls become nico_log_sum_exp_2d)
      vector log_sum_exp_2d(matrix array_2d_to_lse) {
            int N = rows(array_2d_to_lse);
//          matrix[N, 2] rowwise_maxes_2d_array;
//          rowwise_maxes_2d_array[, 1] = rowwise_max(array_2d_to_lse);
//          rowwise_maxes_2d_array[, 2] = rowwise_maxes_2d_array[, 1];
            int n_cols = cols(array_2d_to_lse);
            matrix[N, n_cols] rowwise_maxes_2d_array = rep_matrix(rowwise_max(array_2d_to_lse), n_cols);
            return rowwise_maxes_2d_array[, 1] + log(rowwise_sum(exp((array_2d_to_lse - rowwise_maxes_2d_array))));
      }
      ////
      //// NEW: arbitrary number of columns, for the 4-class mixture:
      ////
      vector log_sum_exp_matrix(matrix M) {
            int N = rows(M);
            int C = cols(M);
            vector[N] maxes = rowwise_max(M);
            matrix[N, C] shifted = M - rep_matrix(maxes, C);
            return maxes + log(rowwise_sum(exp(shifted)));
      }
      ////
      //// ---- Index sets without a loop over the subjects. The subjects of a group (e.g. the rows below a
      //// threshold) are found by sorting: sort_indices_asc(x) lists the positions of x in increasing order of x,
      //// so the first number_of_elements_below(x, threshold) of them are the positions of the elements below the
      //// threshold. number_of_elements_below() is rank() of the threshold appended to x (rank(v, s) = the number
      //// of elements of v smaller than v[s]).
      ////
      //// sort_indices_asc() is always given a vector variable, never an indexed expression such as x[idx]: in
      //// Stan 2.36 its comparator keeps a reference to a temporary copy of such an expression, which is freed
      //// before the sort (wrong order or a segmentation fault).
      ////
      int number_of_elements_below(vector x, real threshold) {
            return rank(append_row(x, threshold), num_elements(x) + 1);
      }
      ////
      //// ---- (k - 1) * n_cat + k_upper for each element of the integer arrays k and k_upper (the position of
      //// the category pair (k, k_upper) in a vector of n_cat * n_cat values), without a loop over the elements:
      //// Stan has no arithmetic on integer arrays, so the arithmetic is on their copies as vectors (exact for
      //// these small integers), converted back with to_int().
      ////
      array[] int position_of_category_pair(array[] int k, array[] int k_upper, int n_cat) {
            return to_int(to_array_1d((to_vector(k) - 1.0) * n_cat + to_vector(k_upper)));
      }
      ////
      //// ---- The functions of the log-scale tails, as BayesMVP's C++ (MVP_log_scale_grad_calc_fns.hpp and the
      //// MVOP log-scale tails); each Phi_type uses its own pair. In the AVX2 and AVX-512 versions NicoStan's
      //// writer swaps their calls to BayesMVP's fast_ functions of the same names.
      ////
      //// Phi_type = 2: log Phi_approx(x) = log inv_logit(0.07056 x^3 + 1.5976 x) (with
      //// inv_Phi_approx_from_logit_prob() above for the inverse).
      ////
      vector log_Phi_approx(vector x) {
            return log_inv_logit(0.07056 * square(x) .* x + 1.5976 * x);
      }
      ////
      //// Phi_type = 1: log Phi(x), the exact log normal CDF, as the scalar nico_log_Phi() of BayesMVP's C++
      //// (frozen for Paper 1, general_functions/double_fns.hpp): for x < -2.5 (NORMAL_TAIL_THRESH), the
      //// Mills ratio Phi(x) / phi(x) by the backward recurrence of its continued fraction, of depth 40
      //// (MILLS_CF_DEPTH; t = -x, f = t, then f = t + k / f for k = 40, ..., 1, Mills ratio = 1 / f):
      ////      log Phi(x) = -x^2 / 2 - log(2 pi) / 2 + log(1 / f);
      //// otherwise log(erfc(-x / sqrt(2)) / 2). (no loop over the elements: the elements below -2.5 are found
      //// by sorting; the loop is over the 40 terms of the continued fraction only)
      ////
      vector log_Phi(vector x) {
            int n = num_elements(x);
            vector[n] x_copy = x;
            vector[n] log_Phi_x;
            int n_tail = number_of_elements_below(x_copy, -2.5);
//          array[n] int elements_in_order_of_x = sort_indices_asc(x_copy);
            //// (sorted only when some, but not all, of the elements are below -2.5)
            array[n] int elements_in_order_of_x = linspaced_int_array(n, 1, n);
            if (n_tail > 0 && n_tail < n) elements_in_order_of_x = sort_indices_asc(x_copy);
            array[n_tail] int elements_tail = elements_in_order_of_x[1:n_tail];
            array[n - n_tail] int elements_body = elements_in_order_of_x[(n_tail + 1):n];
            if (n_tail < n) {
                  log_Phi_x[elements_body] = log(0.5 * erfc(-x_copy[elements_body] * 0.70710678118654752440));
            }
            if (n_tail > 0) {
                  vector[n_tail] x_tail = x_copy[elements_tail];
                  vector[n_tail] t = -x_tail;
                  vector[n_tail] f = t;
                  for (k_reversed in 1:40) f = t + (41 - k_reversed) ./ f;
                  log_Phi_x[elements_tail] = -0.5 * x_tail .* x_tail + (-0.91893853320467274178) + log(1.0 ./ f);
            }
            return log_Phi_x;
      }
      ////
      //// ---- The three rational approximations of Phi^{-1} of AS241 (Wichura 1988), as BayesMVP's C++
      //// nico_inv_Phi_case_1, _case_2a and _case_2b (frozen for Paper 1, double_fns.hpp), on whole vectors:
      //// the central case of q = p - 1/2 (|q| <= 0.425), and the tail cases of r = sqrt(-log(min(p, 1 - p)))
      //// for r <= 5 and r > 5 (each the positive value, Phi^{-1}(1 - min(p, 1 - p))).
      ////
      vector inv_Phi_AS241_central_case(vector q) {
            int n = num_elements(q);
            vector[n] r = 0.180625 - q .* q;
            vector[n] num = 2509.0809287301226727 * r + 33430.575583588128105;
            num = num .* r + 67265.770927008700853;  num = num .* r + 45921.953931549871457;
            num = num .* r + 13731.693765509461125;  num = num .* r + 1971.5909503065514427;
            num = num .* r + 133.14166789178437745;  num = num .* r + 3.387132872796366608;
            vector[n] den = 5226.495278852854561 * r + 28729.085735721942674;
            den = den .* r + 39307.89580009271061;   den = den .* r + 21213.794301586595867;
            den = den .* r + 5394.1960214247511077;  den = den .* r + 687.1870074920579083;
            den = den .* r + 42.313330701600911252;  den = den .* r + 1.0;
            return q .* num ./ den;
      }
      vector inv_Phi_AS241_tail_case_for_r_up_to_5(vector r_in) {
            int n = num_elements(r_in);
            vector[n] r = r_in - 1.60;
            vector[n] num = rep_vector(0.00077454501427834140764, n);
            num = r .* num + 0.0227238449892691845833; num = r .* num + 0.24178072517745061177;
            num = r .* num + 1.27045825245236838258;   num = r .* num + 3.64784832476320460504;
            num = r .* num + 5.7694972214606914055;    num = r .* num + 4.6303378461565452959;
            num = r .* num + 1.42343711074968357734;
            vector[n] den = rep_vector(0.00000000105075007164441684324, n);
            den = r .* den + 0.0005475938084995344946; den = r .* den + 0.0151986665636164571966;
            den = r .* den + 0.14810397642748007459;   den = r .* den + 0.68976733498510000455;
            den = r .* den + 1.6763848301838038494;    den = r .* den + 2.05319162663775882187;
            den = r .* den + 1.0;
            return num ./ den;
      }
      vector inv_Phi_AS241_tail_case_for_r_above_5(vector r_in) {
            int n = num_elements(r_in);
            vector[n] r = r_in - 5.0;
            vector[n] num = rep_vector(0.000000201033439929228813265, n);
            num = r .* num + 0.0000271155556874348757815; num = r .* num + 0.0012426609473880784386;
            num = r .* num + 0.026532189526576123093;     num = r .* num + 0.29656057182850489123;
            num = r .* num + 1.7848265399172913358;       num = r .* num + 5.4637849111641143699;
            num = r .* num + 6.6579046435011037772;
            vector[n] den = rep_vector(0.00000000000000204426310338993978564, n);
            den = r .* den + 0.00000014215117583164458887; den = r .* den + 0.000018463183175100546818;
            den = r .* den + 0.0007868691311456132591;     den = r .* den + 0.0148753612908506148525;
            den = r .* den + 0.13692988092273580531;       den = r .* den + 0.59983220655588793769;
            den = r .* den + 1.0;
            return num ./ den;
      }
      ////
      //// Phi_type = 1: Z = Phi^{-1}(p) from log p and log(1 - p), as the scalar nico_inv_Phi_from_log_p() of
      //// BayesMVP's C++ (frozen for Paper 1): with q = exp(log p) - 1/2, the central case of AS241 for
      //// |q| <= 0.425; otherwise r = sqrt(-min(log p, log(1 - p))), the tail case for r <= 5 or r > 5, and
      //// Z = -value for p < 1/2 (log p < log(1 - p)), +value otherwise; no refinement. (no loop over the
      //// elements: the elements with q < -0.425, |q| <= 0.425 and q > 0.425, and then those with r <= 5, are
      //// found by sorting; an element exactly at q = 0.425 or r = 5, of probability zero, takes the other
      //// case, with the same value up to rounding)
      ////
      vector inv_Phi_from_log_p(vector log_p, vector log_1m_p) {
            int n = num_elements(log_p);
            vector[n] log_p_copy = log_p;
            vector[n] log_1m_p_copy = log_1m_p;
            vector[n] q = exp(log_p_copy) - 0.5;
            vector[n] Z;
            int n_lower = number_of_elements_below(q, -0.425);
            int n_not_upper = number_of_elements_below(q, 0.425);
            int n_upper = n - n_not_upper;
            int n_central = n_not_upper - n_lower;
//          array[n] int elements_in_order_of_q = sort_indices_asc(q);
            //// (sorted only when an element is not in the central case)
            array[n] int elements_in_order_of_q = linspaced_int_array(n, 1, n);
            if (n_central < n) elements_in_order_of_q = sort_indices_asc(q);
            array[n_central] int elements_central = elements_in_order_of_q[(n_lower + 1):n_not_upper];
            if (n_central > 0) Z[elements_central] = inv_Phi_AS241_central_case(q[elements_central]);
            if (n_central < n) {
                  //// (the tail elements: those with q < -0.425, value -v, then those with q > 0.425, value +v)
                  array[n - n_central] int elements_tail
                        = append_array(elements_in_order_of_q[1:n_lower],
                                       elements_in_order_of_q[(n_not_upper + 1):n]);
                  vector[n - n_central] sign_of_Z = append_row(rep_vector(-1.0, n_lower),
                                                               rep_vector(1.0, n_upper));
                  vector[n - n_central] r = sqrt(-fmin(log_p_copy[elements_tail], log_1m_p_copy[elements_tail]));
                  vector[n - n_central] v;
                  int n_r_up_to_5 = number_of_elements_below(r, 5.0);
//                array[n - n_central] int positions_in_order_of_r = sort_indices_asc(r);
                  //// (sorted only when some, but not all, of the tail elements have r <= 5)
                  array[n - n_central] int positions_in_order_of_r
                        = linspaced_int_array(n - n_central, 1, n - n_central);
                  if (n_r_up_to_5 > 0 && n_r_up_to_5 < n - n_central) {
                        positions_in_order_of_r = sort_indices_asc(r);
                  }
                  array[n_r_up_to_5] int positions_r_up_to_5 = positions_in_order_of_r[1:n_r_up_to_5];
                  array[n - n_central - n_r_up_to_5] int positions_r_above_5
                        = positions_in_order_of_r[(n_r_up_to_5 + 1):(n - n_central)];
                  if (n_r_up_to_5 > 0) {
                        v[positions_r_up_to_5] = inv_Phi_AS241_tail_case_for_r_up_to_5(r[positions_r_up_to_5]);
                  }
                  if (n_r_up_to_5 < n - n_central) {
                        v[positions_r_above_5] = inv_Phi_AS241_tail_case_for_r_above_5(r[positions_r_above_5]);
                  }
                  Z[elements_tail] = sign_of_Z .* v;
            }
            return Z;
      }

      //////////////////////////////////////////////////////////////////////////
      // ---- Ordinal helpers (unchanged):
      //////////////////////////////////////////////////////////////////////////
      vector construct_C(vector C_raw_vec, int softplus) {
            int n_total_cutpoints = num_elements(C_raw_vec);
            vector[n_total_cutpoints] C_vec;
            C_vec[1] = C_raw_vec[1];
            if (softplus == 1) {
                vector[n_total_cutpoints - 1] sp = log1p_exp(C_raw_vec[2:n_total_cutpoints]);
                for (k in 2:n_total_cutpoints) C_vec[k] = C_vec[k - 1] + sp[k - 1];
            } else {
                vector[n_total_cutpoints - 1] exp_C = exp(C_raw_vec[2:n_total_cutpoints]);
                for (k in 2:n_total_cutpoints) C_vec[k] = C_vec[k - 1] + exp_C[k - 1];
            }
            return C_vec;
      }
      array[] int calculate_start_indices(array[] int n_thr, int n_tests) {
            array[n_tests] int start_index;
            start_index[1] = 1;
            for (t in 2:n_tests) start_index[t] = start_index[t - 1] + n_thr[t - 1];
            return start_index;
      }
      array[] int calculate_end_indices(array[] int n_thr, int n_tests, array[] int start_index) {
            array[n_tests] int end_index;
            for (t in 1:n_tests) end_index[t] = start_index[t] + n_thr[t] - 1;
            return end_index;
      }
      vector get_test_values(vector flat_values, data array[] int start_index, data array[] int end_index, data int test_index) {
            int n_elements = end_index[test_index] - start_index[test_index] + 1;
            vector[n_elements] result;
            for (i in 1:n_elements) result[i] = flat_values[start_index[test_index] + i - 1];
            return result;
      }
      vector update_test_values(vector flat_values_to_update, vector new_values, data array[] int start_index, data array[] int end_index, data int test_index) {
            int n_elements = end_index[test_index] - start_index[test_index] + 1;
            vector[num_elements(flat_values_to_update)] result = flat_values_to_update;
            for (i in 1:n_elements) result[start_index[test_index] + i - 1] = new_values[i];
            return result;
      }
      vector cumul_probs_to_ord_probs( vector cumul_probs) {
                int K_minus_1 = num_elements(cumul_probs);
                int K = K_minus_1 + 1;
                vector[K] ord_probs;
                ord_probs[1] = cumul_probs[1];
                for (k in 2:K_minus_1) ord_probs[k] = cumul_probs[k] - cumul_probs[k - 1];
                ord_probs[K] = 1.0 - cumul_probs[K_minus_1];
                return ord_probs;
      }
      //// Narrow intervals must retain their log-width from the raw parameters. Subtracting
      //// rounded cutpoints (or two rounded CDFs) loses that width and can produce infinite gradients.
      //// Four-point Gauss-Legendre integration is used only when width * (1 + |mid|) is tiny.
      real log_Phi_stable(real x) {
                //// Differentiate the SAME expression used for the log-CDF value. This avoids
                //// the separate approximate derivative in the bundled std_normal_lcdf routine.
                if (x < -10.0) {
                    //// Laplace continued fraction for the Mills ratio Phi(-t) / phi(t).
                    //// Twenty terms agree to double precision in this t > 10 branch.
                    real t = -x;
                    real r = 0.0;
                    for (k in 1:20) r = (21.0 - k) / (t + r);
                    return -0.5 * square(x) - 0.5 * log(2.0 * pi()) - log(t + r);
                }
                if (x <= 0.0) return log(erfc(-x / sqrt(2.0))) - log(2.0);
                return log1m(0.5 * erfc(x / sqrt(2.0)));
      }
      real normal_interval_logprob(real lo, real hi, real log_width) {
                //// The raw parameterisation retains the gap even when hi - lo rounds to zero.
                real width = exp(log_width);
                real mid = lo + 0.5 * width;
                if (log_width + log1p(abs(mid)) < log(1e-3)) {
                    real w2 = square(width);
                    real m2 = square(mid);
                    real correction = (m2 - 1.0) * w2 / 24.0
                                    + (square(m2) - 6.0 * m2 + 3.0) * square(w2) / 1920.0;
                    return log_width + std_normal_lpdf(mid) + log1p(correction);
                }
                if (lo > 0.0) return log_diff_exp(log_Phi_stable(-lo), log_Phi_stable(-hi));
                return log_diff_exp(log_Phi_stable(hi), log_Phi_stable(lo));
      }
      real Phi_stable(real x) {
                //// Ordinary-range CDF without 1 + erf cancellation on the negative side.
                return 0.5 * erfc(-x / sqrt(2.0));
      }
      real inv_Phi_from_log_lower(real log_p) {
                //// Called only for a probability <= 0.5. Exponentiating is safe down to -700.
                // if (log_p > -700.0) return inv_Phi(exp(log_p));
                //// (no -700 threshold: the Newton steps below find the quantile at every log_p)
                real z = -sqrt(-2.0 * log_p);
                for (i in 1:5) {
                    real lp = log_Phi_stable(z);
                    z -= (lp - log_p) * exp(lp - std_normal_lpdf(z));
                }
                return z;
      }
      real normal_from_log_uniform(real log_u, real log1m_u) {
                if (log_u <= -log(2.0)) return inv_Phi_from_log_lower(log_u);
                return -inv_Phi_from_log_lower(log1m_u);
      }
      vector normal_interval_step(real lo, real hi, real log_width, real log_u, real log1m_u) {
                vector[2] out;            //// conditional normal draw, log interval probability
                real u = exp(log_u);
                int left_open = is_inf(lo);
                int right_open = is_inf(hi);
                if (!left_open && !right_open) {
                    real width = exp(log_width);
                    real mid = lo + 0.5 * width;
                    if (log_width + log1p(abs(mid)) < log(1e-5)) {
                        //// Third-order inverse expansion; omitted terms are below roundoff here.
                        real uv = u * exp(log1m_u);
                        out[1] = lo + width * (u - 0.5 * lo * width * uv
                               + square(width) * uv * (square(lo) * (1.0 - 2.0 * u) - 1.0 - u) / 6.0);
                        out[2] = normal_interval_logprob(lo, hi, log_width);
                        return out;
                    }
                }
                //// Ordinary cases use Phi / inv_Phi directly, reflecting positive intervals.
                //// Keep extreme uniforms in log space so the tanh map cannot round them to 0 / 1.
                if (log_u > log(1e-8) && log1m_u > log(1e-8)) {
                    if (left_open && hi >= -5.0 && hi <= 5.0) {
                        real p = Phi_stable(hi);
                        real q = u * p;
                        out[1] = q <= 0.5 ? inv_Phi(q) : -inv_Phi(exp(log1m_u) + u * Phi_stable(-hi));
                        out[2] = hi <= 0.0 ? log(p) : log1m(Phi_stable(-hi));
                        return out;
                    }
                    if (right_open && lo >= -5.0 && lo <= 5.0) {
                        real p = Phi_stable(-lo);
                        real q = exp(log1m_u) * p;
                        out[1] = q <= 0.5 ? -inv_Phi(q) : inv_Phi(u + exp(log1m_u) * Phi_stable(lo));
                        out[2] = lo >= 0.0 ? log(p) : log1m(Phi_stable(lo));
                        return out;
                    }
                    if (!left_open && !right_open && lo >= -5.0 && hi <= 5.0
                        && log_width + log1p(abs(0.5 * (lo + hi))) >= log(1e-3)) {
                        if (lo >= 0.0) {
                            real a = Phi_stable(-hi);
                            real p = Phi_stable(-lo) - a;
                            out[1] = -inv_Phi(a + exp(log1m_u) * p);
                            out[2] = log(p);
                        } else {
                            real a = Phi_stable(lo);
                            real p = Phi_stable(hi) - a;
                            real q = a + u * p;
                            out[1] = q <= 0.5 ? inv_Phi(q)
                                   : -inv_Phi(exp(log1m_u) * Phi_stable(-lo) + u * Phi_stable(-hi));
                            out[2] = log(p);
                        }
                        return out;
                    }
                }
                //// One normal distribution throughout: no cubic-logistic tail splice or clipping.
                {
                    real lp_lo = left_open ? negative_infinity() : log_Phi_stable(lo);
                    real lp_hi = right_open ? 0.0 : log_Phi_stable(hi);
                    real lq_lo = left_open ? 0.0 : log_Phi_stable(-lo);
                    real lq_hi = right_open ? negative_infinity() : log_Phi_stable(-hi);
                    real log_q = log_sum_exp(log1m_u + lp_lo, log_u + lp_hi);
                    real log1m_q = log_sum_exp(log1m_u + lq_lo, log_u + lq_hi);
                    out[1] = normal_from_log_uniform(log_q, log1m_q);
                    if (left_open) out[2] = lp_hi;
                    else if (right_open) out[2] = lq_lo;
                    else out[2] = normal_interval_logprob(lo, hi, log_width);
                }
                return out;
      }
      real normal_interval_mass(real lo, real hi, real log_width) {
                if (is_inf(lo)) return log_Phi_stable(hi);
                if (is_inf(hi)) return log_Phi_stable(-lo);
                return normal_interval_logprob(lo, hi, log_width);
      }
      real narrow_interval_logprob(real mid, real log_width, int approx_cdf) {
            vector[4] nodes = [-0.8611363115940526, -0.3399810435848563,
                               0.3399810435848563,  0.8611363115940526]';
            vector[4] weights = [0.1739274225687269, 0.3260725774312731,
                                 0.3260725774312731, 0.1739274225687269]';
            vector[4] terms;
            real width = exp(log_width);
            for (q in 1:4) {
                  real z = mid + 0.5 * width * nodes[q];
                  real ld;
                  if (approx_cdf == 1) {
                        real poly = 0.07056 * z * square(z) + 1.5976 * z;
                        ld = log(0.21168 * square(z) + 1.5976)
                             + log_inv_logit(poly) + log1m_inv_logit(poly);
                  } else ld = std_normal_lpdf(z);
                  terms[q] = log(weights[q]) + ld;
            }
            return log_width + log_sum_exp(terms);
      }
      real narrow_interval_z(real mid, real log_width, real u, int approx_cdf) {
            real width = exp(log_width);
            real score = -mid;
            if (approx_cdf == 1) {
                  real deriv = 0.21168 * square(mid) + 1.5976;
                  real poly = 0.07056 * mid * square(mid) + 1.5976 * mid;
                  score = 0.42336 * mid / deriv + deriv * (1 - 2 * inv_logit(poly));
            }
            //// Second-order inverse-CDF expansion. This branch requires width*(1+|mid|)<1e-5;
            //// omitted terms are of order width^3, at approximately floating-point accuracy.
            return mid + width * (u - 0.5) + 0.5 * score * square(width) * u * (1 - u);
      }
      ////
      //// ---- The same two functions for a vector of narrow intervals (the narrow rows of one ordinal test),
      //// with the same operations, in the same order, on whole vectors: one column of terms per Gauss-Legendre
      //// node (a loop over the four nodes only), and the log density of the standard normal, std_normal_lpdf(z)
      //// of each element, written out as -z^2 / 2 - log(2 pi) / 2.
      ////
      vector narrow_interval_logprob(vector mid, vector log_width, int approx_cdf) {
            int n = num_elements(mid);
            vector[4] nodes = [-0.8611363115940526, -0.3399810435848563,
                               0.3399810435848563,  0.8611363115940526]';
            vector[4] weights = [0.1739274225687269, 0.3260725774312731,
                                 0.3260725774312731, 0.1739274225687269]';
            matrix[n, 4] terms;
            vector[n] width = exp(log_width);
            for (q in 1:4) {
                  vector[n] z = mid + 0.5 * width * nodes[q];
                  vector[n] ld;
                  if (approx_cdf == 1) {
                        vector[n] poly = 0.07056 * z .* square(z) + 1.5976 * z;
                        //// (log1m_inv_logit(poly) written as log_inv_logit(-poly), the same formula in Stan
                        ////  math, so that the AVX writer swaps it)
                        ld = log(0.21168 * square(z) + 1.5976)
                             + log_inv_logit(poly) + log_inv_logit(-poly);
                  } else ld = -0.5 * square(z) - 0.5 * log(2.0 * pi());
                  terms[, q] = log(weights[q]) + ld;
            }
            return log_width + log_sum_exp_2d(terms);
      }
      vector narrow_interval_z(vector mid, vector log_width, vector u, int approx_cdf) {
            int n = num_elements(mid);
            vector[n] width = exp(log_width);
            vector[n] score = -mid;
            if (approx_cdf == 1) {
                  vector[n] deriv = 0.21168 * square(mid) + 1.5976;
                  vector[n] poly = 0.07056 * mid .* square(mid) + 1.5976 * mid;
                  score = 0.42336 * mid ./ deriv + deriv .* (1 - 2 * inv_logit(poly));
            }
            return mid + width .* (u - 0.5) + 0.5 * score .* square(width) .* u .* (1 - u);
      }

      real induced_dirichlet_lpdf(vector C_vec, vector log_spacing, vector alpha) {
                int n_thr = num_elements(C_vec);
                int K = n_thr + 1;
                real lp = lgamma(sum(alpha)) - sum(lgamma(alpha));
                //// A unit alpha contributes nothing. Do not build an unused log(0) AD node.
                for (k in 1:K) if (alpha[k] != 1.0) {
                    real log_p;
                    if (k == 1) log_p = log_Phi_stable(C_vec[1]);
                    else if (k == K) log_p = log_Phi_stable(-C_vec[n_thr]);
                    else log_p = normal_interval_logprob(C_vec[k - 1], C_vec[k], log_spacing[k]);
                    lp += (alpha[k] - 1.0) * log_p;
                }
                //// Jacobian from ordered cutpoints to the first K-1 cumulative probabilities.
                lp += std_normal_lpdf(C_vec);
                return lp;
      }


      //////////////////////////////////////////////////////////////////////////
      // ---- reduce_sum_static partial-sum function (4-class). Slices u_raw by OBSERVATION;
      //// valid because the only sequential dependence (the "inc" recursion) runs
      //// across TESTS within a subject, never across subjects.
      //// Body = the serial file's observed-only likelihood with N -> M and slice-local
      //// obs / mis index sets. Returns (tanh Jacobian of the u slice) + sum log-lik.
      //////////////////////////////////////////////////////////////////////////
      //// Shared measurement + conditional independence: each class/category probability is
      //// identical across respondents. Cache the observed categories once per evaluation.
      //// The probability formulas (including tail and narrow-interval branches) match the GHK
      //// likelihood below; missing tests contribute zero, and class mixtures remain per group.
      vector ci_shared_log_lik(data matrix y, data array[] int pop,
                            data array[,] int class_D, data array[] int test_condition,
                            matrix beta, matrix log_prev_subject,
                            array[] vector C_vec_tp, array[] vector log_C_gap_tp,
                            data array[] int ord_start_index, data array[] int ord_end_index,
                            data array[] int n_thr_per_ord_test, data array[] int n_cat_per_ord_test,
                            data int n_binary_tests, data int Phi_type, data real C_sentinel,
                            data real overflow_threshold, data real underflow_threshold) {
            int N = rows(y);
            int T = cols(y);
            int C = rows(beta);
            int Kmax = max(n_cat_per_ord_test) + 2;
//          array[N,T] int category;
            array[T,Kmax] int seen = rep_array(0,T,Kmax);
            matrix[C,T*Kmax] cache = rep_matrix(0.0,C,T*Kmax);
            vector[N] out;
//          for (n in 1:N) for (t in 1:T) {
//                category[n,t] = y[n,t] < 0 ? 0 : to_int(y[n,t]) + (t <= n_binary_tests ? 1 : 0);
//                if (category[n,t] > 0) seen[t,category[n,t]] = 1;
//          }
            //// (no loop over the subjects: for each test, the subjects in increasing order of their score, and
            ////  the positions in that order of the first and the last subject of each category k, i.e. of the
            ////  score k - 1 for a binary test and k for an ordinal one; the missing scores, y < 0, come
            ////  first, in no category)
            array[T, N] int subjects_in_order_of_score;
            array[T, Kmax] int first_position_of_category = rep_array(1, T, Kmax);
            array[T, Kmax] int last_position_of_category = rep_array(0, T, Kmax);
            for (t in 1:T) {
                  int K = t <= n_binary_tests ? 2 : n_cat_per_ord_test[t-n_binary_tests];
                  real score_of_category_1 = t <= n_binary_tests ? 0.0 : 1.0;
                  vector[N] score_of_test_t = y[, t];
                  subjects_in_order_of_score[t] = sort_indices_asc(score_of_test_t);
                  for (k in 1:K) {
                        real score_of_category_k = score_of_category_1 + k - 1;
                        first_position_of_category[t, k]
                              = number_of_elements_below(score_of_test_t, score_of_category_k - 0.5) + 1;
                        last_position_of_category[t, k]
                              = number_of_elements_below(score_of_test_t, score_of_category_k + 0.5);
                        if (last_position_of_category[t, k] >= first_position_of_category[t, k]) seen[t, k] = 1;
                  }
            }
            for (c in 1:C) for (t in 1:T) {
                  int K = t <= n_binary_tests ? 2 : n_cat_per_ord_test[t-n_binary_tests];
                  for (k in 1:K) if (seen[t,k] == 1) {
                        real lp;
                        //// Shared measurement: compute each class/category probability once.
                        //// Phi_type = 1 uses the exact-normal interval routine (log-space tails), exact
                        //// everywhere as the GHK path; only Phi_type = 2 takes the else branch below.
                        if (Phi_type == 1) {
                              if (t <= n_binary_tests)
                                    lp = log_Phi_stable(k == 1 ? -beta[c,t] : beta[c,t]);
                              else {
                                    int to = t - n_binary_tests;
                                    int own = class_D[c,test_condition[t]] + 1;
                                    int first = ord_start_index[to];
                                    real lo = k == 1 ? negative_infinity() : C_vec_tp[own][first+k-2] - beta[c,t];
                                    real hi = k == K ? positive_infinity() : C_vec_tp[own][first+k-1] - beta[c,t];
                                    real log_width = (k > 1 && k < K) ? log_C_gap_tp[own][first+k-1] : positive_infinity();
                                    lp = normal_interval_mass(lo,hi,log_width);
                              }
                        } else {
                        if (t <= n_binary_tests) {
                              real b = -beta[c,t];
                              real poly = 0.07056 * b * square(b) + 1.5976 * b;
                              if (k == 1 && b < underflow_threshold) lp = log_inv_logit(poly);
                              else if (k == 2 && b > overflow_threshold) lp = log_inv_logit(-poly);
                              //// Phi_type = 2: log Phi_approx(b) = log_inv_logit(poly), log(1 - Phi_approx(b)) = log_inv_logit(-poly),
                              //// exact and stable for every b (log1m of Phi_approx at b was log(0) for b > ~7.11).
                              else if (Phi_type == 2) lp = log_inv_logit(k == 1 ? poly : -poly);
                              else {
                                    real p = Phi(b);
                                    lp = k == 1 ? log(p) : log1m(p);
                              }
                        } else {
                              int to = t - n_binary_tests;
                              int own = class_D[c,test_condition[t]] + 1;
                              int first = ord_start_index[to];
                              real lo = (k == 1 ? -C_sentinel : C_vec_tp[own][first+k-2]) - beta[c,t];
                              real hi = (k == K ? C_sentinel : C_vec_tp[own][first+k-1]) - beta[c,t];
                              real mid = 0.5 * (lo+hi);
                              real log_width = (k > 1 && k < K) ? log_C_gap_tp[own][first+k-1] : 0.0;
                              if (k > 1 && k < K && log_width < log(1e-5) - log1p(abs(mid))) {
                                    int approx_cdf = (Phi_type == 2);
                                    lp = narrow_interval_logprob(mid,log_width,approx_cdf);
                              } else if (hi < underflow_threshold) {
                                    real llo = log_inv_logit(0.07056 * lo * square(lo) + 1.5976 * lo);
                                    real lhi = log_inv_logit(0.07056 * hi * square(hi) + 1.5976 * hi);
                                    lp = lhi + log1m_exp(llo-lhi);
                              } else if (lo > overflow_threshold) {
                                    real llo = log_inv_logit(-0.07056 * lo * square(lo) - 1.5976 * lo);
                                    real lhi = log_inv_logit(-0.07056 * hi * square(hi) - 1.5976 * hi);
                                    lp = llo + log1m_exp(lhi-llo);
                              } else if (Phi_type == 2) {
                                    //// Same stable Phi_approx interval identity as in partial_log_lik (no tail cancellation).
                                    real poly_lo = 0.07056 * lo * square(lo) + 1.5976 * lo;
                                    real poly_hi = 0.07056 * hi * square(hi) + 1.5976 * hi;
                                    real poly_gap = (hi - lo) * (0.07056 * (square(lo) + lo * hi + square(hi)) + 1.5976);
                                    lp = log_inv_logit(poly_hi) + log_inv_logit(-poly_lo) + log1m_exp(-poly_gap);
                              }
                              else lp = log(Phi(hi)-Phi(lo));
                        }
                        }
                        cache[c,(t-1)*Kmax+k] = lp;
                  }
            }
//          for (n in 1:N) {
//                vector[C] lp = to_vector(log_prev_subject[n]);
//                for (c in 1:C) for (t in 1:T) if (category[n,t] > 0)
//                      lp[c] += cache[c,(t-1)*Kmax+category[n,t]];
//                out[n] = log_sum_exp(lp);
//          }
            //// (vectorised over the subjects: in a loop over the classes, tests and categories, each class /
            ////  category term is added to the subjects with that category, in the same order of the tests; the
            ////  missing scores add nothing; then the log_sum_exp of each row)
            matrix[N, C] log_lik_by_class = log_prev_subject;
            for (c in 1:C) for (t in 1:T) {
                  int K = t <= n_binary_tests ? 2 : n_cat_per_ord_test[t-n_binary_tests];
                  for (k in 1:K) if (seen[t,k] == 1) {
                        int first_position = first_position_of_category[t, k];
                        int last_position = last_position_of_category[t, k];
                        array[last_position - first_position + 1] int subjects
                              = subjects_in_order_of_score[t, first_position:last_position];
                        log_lik_by_class[subjects, c] = log_lik_by_class[subjects, c] + cache[c,(t-1)*Kmax+k];
                  }
            }
            out = log_sum_exp_2d(log_lik_by_class);
            return out;
      }

      //// ---- log-prevalence rows for ONE reduce_sum slice, rebuilt from the SMALL prevalence
      //// parameters and the slice's own data rows. This exists so that log_prev_subject (an
      //// N x n_class PARAMETER matrix) is no longer a SHARED argument of reduce_sum_static:
      //// Stan copies every shared parameter argument into each chunk's local tape, so sharing
      //// it cost a 4N-var copy (29,612 vars at N = 7403) PER CHUNK and made more chunks slower.
      matrix log_prev_for_slice(data matrix X_prevalence_s, data array[] int pop_s, data int n_class,
                                data int use_OR_prior, array[] vector prev_simplex,
                                array[] real mu_A, array[] real sigma_A, array[] real z_A, vector beta_prevalence_A,
                                array[] real mu_B, array[] real sigma_B, array[] real z_B, vector beta_prevalence_B,
                                array[] real log_OR_par, data int log_OR_shared, data int use_cond_prior) {
            int M = size(pop_s);
            if (use_OR_prior == 0) {
                  matrix[M, n_class] out;
                  //// A population's simplex is shared, so evaluate its logarithm once per slice.
                  array[size(prev_simplex)] vector[n_class] log_prev_population;
                  for (g in 1:size(prev_simplex)) log_prev_population[g] = log(prev_simplex[g]);
//                for (m in 1:M) out[m] = log_prev_population[pop_s[m]]';
                  //// (no loop over the subjects: the rows of the populations' log class probabilities, one row
                  ////  per population, indexed by the subjects' populations)
                  matrix[size(prev_simplex), n_class] log_prev_population_rows;
                  for (g in 1:size(prev_simplex)) log_prev_population_rows[g] = log_prev_population[g]';
                  out = log_prev_population_rows[pop_s];
                  return out;
            }
            return log_prevalence_by_subject(X_prevalence_s, pop_s, n_class,
                                             mu_A[1], sigma_A[1], z_A, beta_prevalence_A,
                                             mu_B[1], sigma_B[1], z_B, beta_prevalence_B,
                                             log_OR_par, log_OR_shared, use_cond_prior);
      }

      //// ---- reduce_sum_static partial-sum function of the conditional-independence shortcut (all correlations
      //// zero and no test covariates; model block), over the subjects start to end (pop_slice: their
      //// populations): the slice's log class probabilities (log_prev_for_slice, as in partial_log_lik), then
      //// ci_shared_log_lik() over the slice's rows of y, unchanged (its class / category cache is built for the
      //// categories seen in the slice, with the same values), so that the main autodiff tape holds no
      //// per-subject term.
      real partial_log_lik_conditional_independence_shortcut(
                            array[] int pop_slice,
                            int start,
                            int end,
                            data matrix y,
                            data matrix X_prevalence, data int use_OR_prior, array[] vector prev_simplex,
                            array[] real mu_logit_A, array[] real sigma_logit_A, array[] real z_logit_A,
                            vector beta_prevalence_A,
                            array[] real mu_logit_B, array[] real sigma_logit_B, array[] real z_logit_B,
                            vector beta_prevalence_B,
                            array[] real log_OR_par, data int log_OR_shared, data int use_cond_prior,
                            data int n_class,
                            data array[,] int class_D, data array[] int test_condition,
                            matrix beta,
                            array[] vector C_vec_tp, array[] vector log_C_gap_tp,
                            data array[] int ord_start_index, data array[] int ord_end_index,
                            data array[] int n_thr_per_ord_test, data array[] int n_cat_per_ord_test,
                            data int n_binary_tests, data int Phi_type, data real C_sentinel,
                            data real overflow_threshold, data real underflow_threshold) {

            matrix[end - start + 1, n_class] log_prev
                  = log_prev_for_slice(X_prevalence[start:end, ], pop_slice, n_class, use_OR_prior, prev_simplex,
                                       mu_logit_A, sigma_logit_A, z_logit_A, beta_prevalence_A,
                                       mu_logit_B, sigma_logit_B, z_logit_B, beta_prevalence_B,
                                       log_OR_par, log_OR_shared, use_cond_prior);
            return sum(ci_shared_log_lik(y[start:end, ], pop_slice, class_D, test_condition, beta, log_prev,
                                         C_vec_tp, log_C_gap_tp, ord_start_index, ord_end_index,
                                         n_thr_per_ord_test, n_cat_per_ord_test, n_binary_tests,
                                         Phi_type, C_sentinel, overflow_threshold, underflow_threshold));

      }

      //// ---- reduce_sum_static partial-sum function of the use_cond_prior = 1 prior (transformed parameters):
      //// one of its four sums over the standardisation profiles, for the profiles start to end (pop_slice:
      //// their populations), with a_i = inv_logit(eta_A_i), q_i = inv_logit(eta_q_i),
      //// 1 - q_i = inv_logit(-eta_q_i) and the profile weights w_i (standardisation_w):
      ////      standardisation_sum_number = 1: sum_i w_i a_i
      ////      standardisation_sum_number = 2: sum_i w_i a_i q_i
      ////      standardisation_sum_number = 3: sum_i w_i a_i (1 - q_i)
      ////      standardisation_sum_number = 4: sum_i w_i a_i q_i (1 - q_i)
      //// Each term is positive and at most w_i, and 1 - q_i is never formed by subtracting from one, so the sums
      //// are accurate on the natural scale; their logs are the log_sum_exp()s of the serial version.
      real partial_sum_of_standardisation_profiles_for_conditional_prior(
                            array[] int pop_slice,
                            int start,
                            int end,
                            data int standardisation_sum_number,
                            data matrix X_prevalence_standardisation,
                            data vector standardisation_w,
                            array[] real mu_logit_A, array[] real sigma_logit_A, array[] real z_logit_A,
                            vector beta_prevalence_A,
                            array[] real mu_logit_B, array[] real sigma_logit_B, array[] real z_logit_B,
                            vector beta_prevalence_B,
                            array[] real log_OR_par) {

            int M = end - start + 1;
            vector[M] eta_A = mu_logit_A[1] + sigma_logit_A[1] * to_vector(z_logit_A[pop_slice]);
            if (cols(X_prevalence_standardisation) > 0)
                  eta_A += X_prevalence_standardisation[start:end, ] * beta_prevalence_A;
            vector[M] w_times_a = standardisation_w[start:end] .* inv_logit(eta_A);
            if (standardisation_sum_number == 1) return sum(w_times_a);
            vector[M] eta_q = mu_logit_B[1] + sigma_logit_B[1] * to_vector(z_logit_B[pop_slice]);
            if (cols(X_prevalence_standardisation) > 0)
                  eta_q += X_prevalence_standardisation[start:end, ] * beta_prevalence_B;
            eta_q += log_OR_par[1];
            if (standardisation_sum_number == 2) return dot_product(w_times_a, inv_logit(eta_q));
            if (standardisation_sum_number == 3) return dot_product(w_times_a, inv_logit(-eta_q));
            return dot_product(w_times_a, inv_logit(eta_q) .* inv_logit(-eta_q));

      }

      real partial_log_lik( array[] row_vector u_raw_slice,
                            int start,
                            int end,
                            data matrix y_use,
                            data array[,] int obs_mask,
                            data array[,] int y_ord,
                            data array[,] int y_ord_upper,
                            data array[] matrix X,
                            data array[] int n_covs_per_outcome,
                            data array[] int pop,
                            data array[,] int class_D,
                            data array[] int test_condition,
                            matrix beta,
                            array[] matrix beta_cov,
                            array[] matrix L_Omega,
                            matrix L_Omega_diag_recip,
                            data matrix X_prevalence, data int use_OR_prior, array[] vector prev_simplex,
                            array[] real mu_logit_A, array[] real sigma_logit_A, array[] real z_logit_A, vector beta_prevalence_A,
                            array[] real mu_logit_B, array[] real sigma_logit_B, array[] real z_logit_B, vector beta_prevalence_B,
                            array[] real log_OR_par, data int log_OR_shared, data int use_cond_prior,
                            array[] vector C_vec_tp, array[] vector log_C_gap_tp,
                            data array[] int ord_start_index,
                            data array[] int ord_end_index,
                            data array[] int n_thr_per_ord_test,
                            data array[] int n_cat_per_ord_test,
                            data int n_binary_tests,
                            data array[] int latent_test_order,
                            data array[] int latent_position_of_test,
                            data real C_sentinel,
                            data int n_tests,
                            data int n_class,
                            data int n_covariates_max,
                            data int Phi_type,
                            data real overflow_threshold,
                            data real underflow_threshold) {

            int M = end - start + 1;
            matrix[M, n_class] log_prev = log_prev_for_slice(X_prevalence[start:end, ], pop[start:end], n_class,
                                                             use_OR_prior, prev_simplex,
                                                             mu_logit_A, sigma_logit_A, z_logit_A, beta_prevalence_A,
                                                             mu_logit_B, sigma_logit_B, z_logit_B, beta_prevalence_B,
                                                             log_OR_par, log_OR_shared, use_cond_prior);
            real out = 0.0;

            ////
            //// ---- u transform + Jacobian (plain-function equivalent of lb_ub_jacobian(u_raw, 0, 1)):
            ////
            //// (no loop over the subjects: the slice's u_raw as one matrix, and as one vector for the Jacobian)
            matrix[M, n_tests] u_raw_s = to_matrix(u_raw_slice);
            vector[M * n_tests] u_raw_s_vector = to_vector(u_raw_s);
            matrix[M, n_tests] tanh_u;
//          for (m in 1:M)  tanh_u[m, ] = tanh(u_raw_slice[m]);
            tanh_u = tanh(u_raw_s);
            matrix[M, n_tests] u = 0.5 * (1.0 + tanh_u);
//          for (m in 1:M)
//                out += sum(log2() - 2.0 * u_raw_slice[m] - 2.0 * log1p_exp(-2.0 * u_raw_slice[m]));
            out += sum(log2() - 2.0 * u_raw_s_vector - 2.0 * log1p_exp(-2.0 * u_raw_s_vector));

            ////
            //// ---- slice-local copies of data:
            ////
            matrix[M, n_tests] y_use_s = y_use[start:end, ];
            array[M, n_tests] int obs_mask_s = obs_mask[start:end, ];
            array[size(y_ord), M] int y_ord_s;
            for (t_ord in 1:size(y_ord)) y_ord_s[t_ord] = y_ord[t_ord, start:end];
            array[size(y_ord_upper), M] int y_ord_upper_s;
            for (t_ord in 1:size(y_ord_upper)) y_ord_upper_s[t_ord] = y_ord_upper[t_ord, start:end];
            array[M] int pop_s = pop[start:end];
            array[M] int data_row_of_slice_row = linspaced_int_array(M, start, end);   //// = start + m - 1


      matrix[M, n_tests] Z_std_norm;
      matrix[M, n_tests] y1;
      matrix[M, n_class] lp;
      vector[M] inc;

      ////
      //// ---- slice-local observed / missing index sets of each test, once for all the classes (they do not
      //// depend on the class): sorted by obs_mask, the missing subjects come first and the observed ones last;
      //// when no subject of the slice misses the test, nothing is sorted (every subject is observed, in order).
      ////
      array[n_tests] int n_observed_in_slice_of_test;
      array[n_tests, M] int subjects_of_slice_in_order_of_obs_mask;
      for (t in 1:n_tests) {
            array[M] int obs_mask_of_test = obs_mask_s[, t];
            n_observed_in_slice_of_test[t] = sum(obs_mask_of_test);
            if (n_observed_in_slice_of_test[t] == M) {
                  subjects_of_slice_in_order_of_obs_mask[t] = linspaced_int_array(M, 1, M);
            } else {
                  subjects_of_slice_in_order_of_obs_mask[t] = sort_indices_asc(obs_mask_of_test);
            }
      }
      ////
      //// ---- once for all the classes (neither depends on the class): the untruncated draws Z = Phi^{-1}(u) of
      //// the missing subjects of each test, and the covariate effects of the observed subjects of each test for
      //// each own status (two, whereas there are four classes)
      ////
      matrix[M, n_tests] Z_of_missing_subjects;
      array[2] matrix[M, n_tests] Xbeta_of_own_status;
      for (j_latent in 1:n_tests) {
            int t = latent_test_order[j_latent];
            int n_o = n_observed_in_slice_of_test[t];
            int n_m = M - n_o;
            array[M] int subjects_in_order_of_obs_mask = subjects_of_slice_in_order_of_obs_mask[t];
            if (n_m > 0) {
                  array[n_m] int mis = subjects_in_order_of_obs_mask[1:n_m];
                  if (Phi_type == 2) {
                        vector[n_m] u_mis = u[mis, j_latent];
                        Z_of_missing_subjects[mis, j_latent]
                              = inv_Phi_approx_from_logit_prob(log(u_mis) - log1m(u_mis));
                  }
                  else Z_of_missing_subjects[mis, j_latent] = inv_Phi(u[mis, j_latent]);
            }
            if (n_o > 0 && n_covariates_max > 0 && n_covs_per_outcome[t] > 0) {
                  array[n_o] int idx = subjects_in_order_of_obs_mask[(n_m + 1):M];
                  array[n_o] int idx_g = data_row_of_slice_row[idx];
                  for (s in 1:2) {
                        Xbeta_of_own_status[s][idx, t] = X[t][idx_g, 1:n_covs_per_outcome[t]]
                                                         * to_vector(beta_cov[s][1:n_covs_per_outcome[t], t]);
                  }
            }
      }

      for (c in 1:n_class) {

            inc = rep_vector(0.0, M);

            // for (t in 1:n_tests) {
            for (j_latent in 1:n_tests) {
                  int t = latent_test_order[j_latent];

                  // real L_recip = L_Omega_diag_recip[c, t];
                  real L_recip = L_Omega_diag_recip[c, j_latent];
                  real beta_ct = beta[c, t];
                  ////
                  //// ---- slice-local observed / missing index sets for test t:
                  ////
//                int n_o = 0;
//                int n_m = 0;
//                for (m in 1:M) { if (obs_mask_s[m, t] == 1) n_o += 1; else n_m += 1; }
//                int n_o = sum(obs_mask_s[, t]);
                  int n_o = n_observed_in_slice_of_test[t];
                  int n_m = M - n_o;
                  array[max(n_o, 1)] int idx_o;
                  array[max(n_m, 1)] int idx_m;
                  {
//                      int ko = 1; int km = 1;
//                      for (m in 1:M) {
//                          if (obs_mask_s[m, t] == 1) { idx_o[ko] = m; ko += 1; }
//                          else                        { idx_m[km] = m; km += 1; }
//                      }
                        //// (no loop over the subjects: sorted by obs_mask, the missing subjects come first and
                        ////  the observed ones last; sorted once for all the classes, above the class loop)
//                      array[M] int subjects_in_order_of_obs_mask = sort_indices_asc(obs_mask_s[, t]);
                        array[M] int subjects_in_order_of_obs_mask = subjects_of_slice_in_order_of_obs_mask[t];
                        if (n_m > 0) idx_m[1:n_m] = subjects_in_order_of_obs_mask[1:n_m];
                        if (n_o > 0) idx_o[1:n_o] = subjects_in_order_of_obs_mask[(n_m + 1):M];
                  }

                  ////
                  //// ---- MISSING subjects: the untruncated draw the conditioning chain needs,
                  //// and a zero likelihood contribution. Nothing else is computed for them.
                  ////
                  if (n_m > 0) {
                        array[n_m] int mis = idx_m[1:n_m];
//                      if (Phi_type == 2) Z_std_norm[mis, j_latent] = inv_Phi_approx_from_prob(u[mis, j_latent]);
                        //// (Phi_approx^{-1}(u) from logit(u) = log(u) - log(1 - u), with the functions the AVX
                        ////  writer swaps)
//                      if (Phi_type == 2) {
//                            vector[n_m] u_mis = u[mis, j_latent];
//                            Z_std_norm[mis, j_latent]
//                                  = inv_Phi_approx_from_logit_prob(log(u_mis) - log1m(u_mis));
//                      }
//                      else               Z_std_norm[mis, j_latent] = inv_Phi(u[mis, j_latent]);
                        //// (the draws of the missing subjects, computed once for all the classes above)
                        Z_std_norm[mis, j_latent] = Z_of_missing_subjects[mis, j_latent];
                        y1[mis, j_latent] = rep_vector(0.0, n_m);
                  }

                  if (n_o > 0) {

                        array[n_o] int idx = idx_o[1:n_o];

                        ////
                        //// ---- Linear predictor for the OBSERVED subjects only:
                        ////
                        vector[n_o] Xbeta_o = rep_vector(beta_ct, n_o);
                        if (n_covariates_max > 0 && n_covs_per_outcome[t] > 0) {
//                            array[n_o] int idx_g;
//                            for (ii in 1:n_o) idx_g[ii] = start + idx[ii] - 1;
                              //// (the data rows of the observed subjects, without a loop over them)
//                            array[n_o] int idx_g = data_row_of_slice_row[idx];
//                            Xbeta_o += X[t][idx_g, 1:n_covs_per_outcome[t]]
//                                  * to_vector(beta_cov[class_D[c, test_condition[t]] + 1][1:n_covs_per_outcome[t], t]);
                              //// (the covariate effects of the class's own status, computed once for all the
                              ////  classes above)
                              Xbeta_o += Xbeta_of_own_status[class_D[c, test_condition[t]] + 1][idx, t];
                        }
                        vector[n_o] eta_o = Xbeta_o + inc[idx];

                        if (t <= n_binary_tests) {

                              ////////////////////////////////////////////////////////////
                              //// ---- BINARY test t, observed subjects only.
                              //// Partition idx into OK / (underflow & y==0) / (overflow & y==1),
                              //// carrying each group's Bound_Z alongside its subject ids.
                              ////////////////////////////////////////////////////////////
                              vector[n_o] Bz = - eta_o * L_recip;

//                            int n_ok = 0; int n_uf = 0; int n_of = 0;
//                            for (ii in 1:n_o) {
//                                  real yv = y_use_s[idx[ii], t];
//                                  if      ((Bz[ii] > overflow_threshold)  && (yv == 1)) n_of += 1;
//                                  else if ((Bz[ii] < underflow_threshold) && (yv == 0)) n_uf += 1;
//                                  else n_ok += 1;
//                            }
                              //// (no loop over the subjects: the observed subjects are sorted by y, and those
                              ////  with y == 0 and with y == 1 each by Bz; number_of_elements_below() then
                              ////  counts the underflows (y == 0, Bz < underflow_threshold) and the rows that
                              ////  are not overflows (y == 1, Bz < overflow_threshold; a row exactly at
                              ////  overflow_threshold, of probability zero, is taken as an overflow).
                              ////  positions_*: positions in idx.)
                              //// (when no Bz is below underflow_threshold and none is at or above
                              ////  overflow_threshold, whatever its y, no row is in a tail: every observed
                              ////  subject is OK, in its order, and nothing is sorted (as the LC-MVP's
                              ////  num_OK_index == M branch). positions_in_order_of_partition: the n_ok OK rows,
                              ////  then the n_uf underflows, then the n_of overflows.)
                              int n_uf = 0;
                              int n_of = 0;
                              array[n_o] int positions_in_order_of_partition = linspaced_int_array(n_o, 1, n_o);
                              if (number_of_elements_below(Bz, underflow_threshold) > 0
                                  || number_of_elements_below(Bz, overflow_threshold) < n_o) {
                                    vector[n_o] y_o = y_use_s[idx, t];
                                    int n_y0 = number_of_elements_below(y_o, 0.5);
                                    int n_y1 = n_o - n_y0;
                                    array[n_o] int positions_in_order_of_y = sort_indices_asc(y_o);
                                    array[n_y0] int positions_y0 = positions_in_order_of_y[1:n_y0];
                                    array[n_y1] int positions_y1 = positions_in_order_of_y[(n_y0 + 1):n_o];
                                    vector[n_y0] Bz_y0 = Bz[positions_y0];
                                    vector[n_y1] Bz_y1 = Bz[positions_y1];
                                    array[n_y0] int positions_y0_in_order_of_Bz
                                          = positions_y0[sort_indices_asc(Bz_y0)];
                                    array[n_y1] int positions_y1_in_order_of_Bz
                                          = positions_y1[sort_indices_asc(Bz_y1)];
                                    n_uf = number_of_elements_below(Bz_y0, underflow_threshold);
                                    n_of = n_y1 - number_of_elements_below(Bz_y1, overflow_threshold);
                                    array[n_y0 - n_uf + n_y1 - n_of] int positions_of_ok_rows
                                          = append_array(positions_y0_in_order_of_Bz[(n_uf + 1):n_y0],
                                                         positions_y1_in_order_of_Bz[1:(n_y1 - n_of)]);
                                    array[n_uf + n_of] int positions_of_tail_rows
                                          = append_array(positions_y0_in_order_of_Bz[1:n_uf],
                                                         positions_y1_in_order_of_Bz[(n_y1 - n_of + 1):n_y1]);
                                    positions_in_order_of_partition = append_array(positions_of_ok_rows,
                                                                                   positions_of_tail_rows);
                              }
                              int n_ok = n_o - n_uf - n_of;
                              array[n_uf] int positions_uf
                                    = positions_in_order_of_partition[(n_ok + 1):(n_ok + n_uf)];
                              array[n_of] int positions_of
                                    = positions_in_order_of_partition[(n_ok + n_uf + 1):n_o];
                              array[n_ok] int positions_ok = positions_in_order_of_partition[1:n_ok];
                              array[max(n_ok, 1)] int ok_idx;  vector[max(n_ok, 1)] ok_Bz;
                              array[max(n_uf, 1)] int uf_idx;  vector[max(n_uf, 1)] uf_Bz;
                              array[max(n_of, 1)] int of_idx;  vector[max(n_of, 1)] of_Bz;
                              {
//                                  int k1 = 1; int k2 = 1; int k3 = 1;
//                                  for (ii in 1:n_o) {
//                                        real yv = y_use_s[idx[ii], t];
//                                        if      ((Bz[ii] > overflow_threshold)  && (yv == 1)) { of_idx[k3] = idx[ii]; of_Bz[k3] = Bz[ii]; k3 += 1; }
//                                        else if ((Bz[ii] < underflow_threshold) && (yv == 0)) { uf_idx[k2] = idx[ii]; uf_Bz[k2] = Bz[ii]; k2 += 1; }
//                                        else                                                  { ok_idx[k1] = idx[ii]; ok_Bz[k1] = Bz[ii]; k1 += 1; }
//                                  }
                                    if (n_ok > 0) {
                                          ok_idx[1:n_ok] = idx[positions_ok];
                                          ok_Bz[1:n_ok] = Bz[positions_ok];
                                    }
                                    if (n_uf > 0) {
                                          uf_idx[1:n_uf] = idx[positions_uf];
                                          uf_Bz[1:n_uf] = Bz[positions_uf];
                                    }
                                    if (n_of > 0) {
                                          of_idx[1:n_of] = idx[positions_of];
                                          of_Bz[1:n_of] = Bz[positions_of];
                                    }
                              }

                              if (n_ok > 0) {
                                    array[n_ok] int index = ok_idx[1:n_ok];
                                    vector[n_ok] Bound_Z = ok_Bz[1:n_ok];
                                    vector[n_ok] yy = y_use_s[index, t];
                                    if (Phi_type == 2) {
                                          //// Stable Phi_approx binary branch: SAME target as the log of [1 - Phi_approx(Bz)] (y == 1) or [Phi_approx(Bz)] (y == 0),
                                          //// computed on the log scale. With poly(x) = 0.07056 x^3 + 1.5976 x: log Phi_approx(Bz) = log_inv_logit(poly) and
                                          //// log(1 - Phi_approx(Bz)) = log_inv_logit(-poly), exact and stable for every Bz. The old probability-scale 1 - Phi_approx(Bz)
                                          //// rounded to 0 for y == 1 and Bz > ~7.11 whenever overflow_threshold lets such Bz into this branch (log(0), NaN adjoint, Z = +Inf).
                                          //// sign_y = +1 (y == 1) or -1 (y == 0) and poly_signed = sign_y * poly, so y1 = log_inv_logit(-poly_signed).
                                          //// Z = Phi_approx^{-1}(q) via logit(q). For y == 1: q = Phi + (1 - Phi) u and 1 - q = (1 - Phi)(1 - u); y == 0 is the mirror
                                          //// image (poly -> -poly, u -> 1 - u, logit q -> -logit q), so with v = u (y == 1) or v = 1 - u (y == 0):
                                          ////     logit(q) = sign_y * [ log_sum_exp(log_inv_logit(poly_signed), log_inv_logit(-poly_signed) + log(v)) - (log(1 - v) + log_inv_logit(-poly_signed)) ].
                                          //// log(v) and log(1 - v) are taken from log(u) and log1m(u) directly (1 - u is never formed).
                                          vector[n_ok] sign_y = yy + (yy - 1.0);
                                          vector[n_ok] poly_signed = sign_y .* (0.07056 * square(Bound_Z) .* Bound_Z + 1.5976 * Bound_Z);
                                          vector[n_ok] log_Phi_signed = log_inv_logit(poly_signed);
                                          vector[n_ok] log_1m_Phi_signed = log_inv_logit(-poly_signed);
                                          vector[n_ok] log_u = log(u[index, j_latent]);
                                          vector[n_ok] log_1m_u = log1m(u[index, j_latent]);
                                          vector[n_ok] log_v;
                                          vector[n_ok] log_1m_v;
//                                        for (ii in 1:n_ok) {
//                                              if (yy[ii] == 1) { log_v[ii] = log_u[ii];    log_1m_v[ii] = log_1m_u[ii]; }
//                                              else             { log_v[ii] = log_1m_u[ii]; log_1m_v[ii] = log_u[ii];    }
//                                        }
                                          {
                                                //// (no loop over the subjects: sorted by yy, the rows with
                                                ////  yy == 0 come first)
                                                int n_ok_y0 = number_of_elements_below(yy, 0.5);
                                                array[n_ok] int rows_in_order_of_yy = sort_indices_asc(yy);
                                                array[n_ok_y0] int rows_y0 = rows_in_order_of_yy[1:n_ok_y0];
                                                array[n_ok - n_ok_y0] int rows_y1
                                                      = rows_in_order_of_yy[(n_ok_y0 + 1):n_ok];
                                                log_v[rows_y1] = log_u[rows_y1];
                                                log_1m_v[rows_y1] = log_1m_u[rows_y1];
                                                log_v[rows_y0] = log_1m_u[rows_y0];
                                                log_1m_v[rows_y0] = log_u[rows_y0];
                                          }
                                          matrix[n_ok, 2] tmp;
                                          tmp[, 1] = log_Phi_signed;
                                          tmp[, 2] = log_1m_Phi_signed + log_v;
                                          Z_std_norm[index, j_latent] = inv_Phi_approx_from_logit_prob(sign_y .* (log_sum_exp_2d(tmp) - (log_1m_v + log_1m_Phi_signed)));
                                          y1[index, j_latent] = log_1m_Phi_signed;
                                    } else {
                                          //// Phi_type = 1: exact Phi, with the upper side by reflection, as
                                          //// BayesMVP's fn_MVP_compute_lp_GHK_cols_T (exact Phi), with
                                          //// s = 1 - 2y:
                                          ////      prob = Phi(s Bz),   Z = s Phi^{-1}((y - (2y - 1) u) prob),
                                          ////      y1 = log(prob).
                                          vector[n_ok] s = 1.0 - 2.0 * yy;
                                          vector[n_ok] prob = Phi(s .* Bound_Z);
                                          vector[n_ok] w = yy - (2.0 * yy - 1.0) .* u[index, j_latent];
                                          Z_std_norm[index, j_latent] = s .* inv_Phi(w .* prob);
                                          y1[index, j_latent] = log(prob);
                                    }
                              }
                              if (n_uf > 0) {   //// underflow + y == 0
                                    array[n_uf] int index = uf_idx[1:n_uf];
                                    vector[n_uf] Bound_Z = uf_Bz[1:n_uf];
                                    //// the log-scale tail as BayesMVP's C++
                                    //// (fn_MVP_compute_lp_GHK_cols_log_scale_underflow), with each Phi_type's
                                    //// own functions (Phi_type = 2: log_Phi_approx() and
                                    //// inv_Phi_approx_from_logit_prob(); Phi_type = 1: log_Phi() and
                                    //// inv_Phi_from_log_p()):
                                    ////      y1 = log Phi(Bz),   log Phi_Z = log(u) + log Phi(Bz),
                                    ////      log(1 - Phi_Z) = log1m(u Phi(Bz)),
                                    ////      Z = Phi^{-1}(Phi_Z) from log Phi_Z and log(1 - Phi_Z).
                                    //// (before, for Phi_type = 2:)
//                                        vector[n_uf] log_Bound_U_Phi_Bound_Z = log_inv_logit(0.07056 * square(Bound_Z) .* Bound_Z + 1.5976 * Bound_Z);
//                                        vector[n_uf] log_Phi_Z    = log(u[index, j_latent]) + log_Bound_U_Phi_Bound_Z;
//                                        vector[n_uf] log_1m_Phi_Z = log1m_exp(log(u[index, j_latent]) + log_Bound_U_Phi_Bound_Z);
//                                        Z_std_norm[index, j_latent] = inv_Phi_approx_from_logit_prob(log_Phi_Z - log_1m_Phi_Z);
//                                        y1[index, j_latent] = log_Bound_U_Phi_Bound_Z;
                                    vector[n_uf] log_Bound_U_Phi_Bound_Z;
                                    if (Phi_type == 2) log_Bound_U_Phi_Bound_Z = log_Phi_approx(Bound_Z);
                                    else               log_Bound_U_Phi_Bound_Z = log_Phi(Bound_Z);
                                    vector[n_uf] Bound_U_Phi_Bound_Z = exp(log_Bound_U_Phi_Bound_Z);
                                    vector[n_uf] log_Phi_Z    = log(u[index, j_latent]) + log_Bound_U_Phi_Bound_Z;
                                    vector[n_uf] log_1m_Phi_Z = log1m(Bound_U_Phi_Bound_Z .* u[index, j_latent]);
                                    if (Phi_type == 2) {
                                          Z_std_norm[index, j_latent]
                                                = inv_Phi_approx_from_logit_prob(log_Phi_Z - log_1m_Phi_Z);
                                    } else {
                                          Z_std_norm[index, j_latent]
                                                = inv_Phi_from_log_p(log_Phi_Z, log_1m_Phi_Z);
                                    }
                                    y1[index, j_latent] = log_Bound_U_Phi_Bound_Z;
                              }
                              if (n_of > 0) {   //// overflow + y == 1
                                    array[n_of] int index = of_idx[1:n_of];
                                    vector[n_of] Bound_Z = of_Bz[1:n_of];
                                    //// the log-scale tail as BayesMVP's C++
                                    //// (fn_MVP_compute_lp_GHK_cols_log_scale_overflow), with each Phi_type's own
                                    //// functions, by reflection (1 - Phi(Bz) = Phi(-Bz)):
                                    ////      y1 = log(1 - Phi(Bz)) = log Phi(-Bz),
                                    ////      log Phi(Bz) = log1m(1 - Phi(Bz)),
                                    ////      log Phi_Z = log_sum_exp(log Phi(-Bz) + log(u), log Phi(Bz)),
                                    ////      log(1 - Phi_Z) = log(1 - u) + log Phi(-Bz),
                                    ////      Z = Phi^{-1}(Phi_Z) from log Phi_Z and log(1 - Phi_Z).
                                    //// (before, for Phi_type = 2:)
//                                        vector[n_of] log_Bound_U_Phi_Bound_Z_1m = log_inv_logit(- 0.07056 * square(Bound_Z) .* Bound_Z - 1.5976 * Bound_Z);
//                                        matrix[n_of, 2] tmp;
//                                        tmp[, 1] = log_Bound_U_Phi_Bound_Z_1m + log(u[index, j_latent]);
//                                        tmp[, 2] = log1m_exp(log_Bound_U_Phi_Bound_Z_1m);
//                                        vector[n_of] log_Phi_Z    = log_sum_exp_2d(tmp);
//                                        vector[n_of] log_1m_Phi_Z = log1m(u[index, j_latent]) + log_Bound_U_Phi_Bound_Z_1m;
//                                        Z_std_norm[index, j_latent] = inv_Phi_approx_from_logit_prob(log_Phi_Z - log_1m_Phi_Z);
//                                        y1[index, j_latent] = log_Bound_U_Phi_Bound_Z_1m;
                                    vector[n_of] log_Bound_U_Phi_Bound_Z_1m;
                                    if (Phi_type == 2) log_Bound_U_Phi_Bound_Z_1m = log_Phi_approx(-Bound_Z);
                                    else               log_Bound_U_Phi_Bound_Z_1m = log_Phi(-Bound_Z);
                                    vector[n_of] Bound_U_Phi_Bound_Z_1m = exp(log_Bound_U_Phi_Bound_Z_1m);
                                    vector[n_of] log_Bound_U_Phi_Bound_Z = log1m(Bound_U_Phi_Bound_Z_1m);
                                    matrix[n_of, 2] tmp;
                                    tmp[, 1] = log_Bound_U_Phi_Bound_Z_1m + log(u[index, j_latent]);
                                    tmp[, 2] = log_Bound_U_Phi_Bound_Z;
                                    vector[n_of] log_Phi_Z    = log_sum_exp_2d(tmp);
                                    vector[n_of] log_1m_Phi_Z
                                          = log1m(u[index, j_latent]) + log_Bound_U_Phi_Bound_Z_1m;
                                    if (Phi_type == 2) {
                                          Z_std_norm[index, j_latent]
                                                = inv_Phi_approx_from_logit_prob(log_Phi_Z - log_1m_Phi_Z);
                                    } else {
                                          Z_std_norm[index, j_latent]
                                                = inv_Phi_from_log_p(log_Phi_Z, log_1m_Phi_Z);
                                    }
                                    y1[index, j_latent] = log_Bound_U_Phi_Bound_Z_1m;
                              }

                        } else {

                              ////////////////////////////////////////////////////////////
                              //// ---- ORDINAL test t, observed subjects only.
                              ////////////////////////////////////////////////////////////
                              int t_ord = t - n_binary_tests;
                              int n_thr_t = n_thr_per_ord_test[t_ord];
                              int n_cat_t = n_cat_per_ord_test[t_ord];
                              int s_own = class_D[c, test_condition[t]] + 1;    //// 1 = target condition absent, 2 = present
                              vector[n_thr_t] C_t = get_test_values(C_vec_tp[s_own], ord_start_index, ord_end_index, t_ord);

//                            vector[n_o] C_lo;
//                            vector[n_o] C_hi;
//                            for (ii in 1:n_o) {
//                                  int k_obs = y_ord_s[t_ord, idx[ii]];
//                                  C_lo[ii] = (k_obs == 1)       ? -C_sentinel : C_t[k_obs - 1];
//                                  C_hi[ii] = (k_obs == n_cat_t) ?  C_sentinel : C_t[k_obs];
//                                  int k_obs = y_ord_s[t_ord, idx[ii]];
//                                  int k_obs_upper = y_ord_upper_s[t_ord, idx[ii]];   //// = k_obs unless interval-censored
//                                  C_lo[ii] = (k_obs == 1)       ? -C_sentinel : C_t[k_obs - 1];
//                                  C_hi[ii] = (k_obs_upper == n_cat_t) ?  C_sentinel : C_t[k_obs_upper];
//                            }
                              //// (no loop over the subjects: each subject's bounds are taken, by its category,
                              ////  from the cutpoints with -C_sentinel / C_sentinel at the open ends: C_lo of
                              ////  category k is element k of [-C_sentinel, C_t], and C_hi of category k is
                              ////  element k of [C_t, C_sentinel])
                              array[n_o] int k_obs = y_ord_s[t_ord, idx];
                              //// (k_obs_upper = k_obs unless the score is interval-censored)
                              array[n_o] int k_obs_upper = y_ord_upper_s[t_ord, idx];
                              vector[n_o] C_lo = append_row(-C_sentinel, C_t)[k_obs];
                              vector[n_o] C_hi = append_row(C_t, C_sentinel)[k_obs_upper];
                              vector[n_o] Bz_lo = (C_lo - eta_o) * L_recip;
                              vector[n_o] Bz_hi = (C_hi - eta_o) * L_recip;
//                            array[n_o] int is_narrow = rep_array(0, n_o);
                              vector[n_thr_t] log_gap_t = get_test_values(log_C_gap_tp[s_own], ord_start_index, ord_end_index, t_ord);
//                            for (ii in 1:n_o) {
//                                  int k_obs = y_ord_s[t_ord, idx[ii]];
//                                  if (k_obs > 1 && k_obs < n_cat_t) {
//                                        real log_width = log_gap_t[k_obs] + log(L_recip);
//                                  int k_obs = y_ord_s[t_ord, idx[ii]];
//                                  int k_obs_upper = y_ord_upper_s[t_ord, idx[ii]];
//                                  if (k_obs > 1 && k_obs_upper < n_cat_t) {
//                                        //// categories k_obs..k_obs_upper: the log of the sum of their gaps
//                                        real log_width = (k_obs_upper == k_obs ? log_gap_t[k_obs]
//                                                          : log_sum_exp(log_gap_t[k_obs:k_obs_upper]))
//                                                         + log(L_recip);
//                                        real mid = 0.5 * (Bz_lo[ii] + Bz_hi[ii]);
//                                        if (log_width < log(1e-5) - log1p(abs(mid))) {
//                                              int approx_cdf = (Phi_type == 2);
//                                              is_narrow[ii] = 1;
//                                              y1[idx[ii], j_latent] = narrow_interval_logprob(mid, log_width, approx_cdf);
//                                              Z_std_norm[idx[ii], j_latent] = narrow_interval_z(mid, log_width, u[idx[ii], j_latent], approx_cdf);
//                                        }
//                                  }
//                            }
                              //// (no loop over the subjects: the log of the summed gaps of categories k..k_upper
                              ////  is computed once for each pair (k, k_upper) of interior categories, +Inf for
                              ////  the other pairs (never narrow), and taken for each subject by its pair; the
                              ////  narrow rows, with log_width < log(1e-5) - log1p(abs(mid)), i.e.
                              ////  narrow_criterion < 0, are found by sorting)
                              //// (the table of the pairs only when some observed score of the slice is
                              ////  interval-censored; otherwise k_obs_upper = k_obs for every subject, and
                              ////  the log of the summed gaps is the log gap of the subject's own category,
                              ////  the diagonal of the table: log_gap_t[k] for the interior categories
                              ////  k = 2, ..., n_cat_t - 1, and +Inf for the open categories 1 and n_cat_t)
                              int n_interval_censored = n_o - number_of_elements_below(to_vector(k_obs_upper)
                                                                                       - to_vector(k_obs), 0.5);
                              vector[n_o] log_width;
                              if (n_interval_censored == 0) {
                                    vector[n_cat_t] log_gap_of_category
                                          = append_row(append_row(positive_infinity(), log_gap_t[2:n_thr_t]),
                                                       positive_infinity());
                                    log_width = log_gap_of_category[k_obs] + log(L_recip);
                              } else {
                                    vector[n_cat_t * n_cat_t] log_gap_sum_of_category_pair
                                          = rep_vector(positive_infinity(), n_cat_t * n_cat_t);
                                    for (k in 2:(n_cat_t - 1)) {
                                          for (k_upper in k:(n_cat_t - 1)) {
                                                //// categories k..k_upper: the log of the sum of their gaps
                                                log_gap_sum_of_category_pair[(k - 1) * n_cat_t + k_upper]
                                                      = (k_upper == k ? log_gap_t[k]
                                                         : log_sum_exp(log_gap_t[k:k_upper]));
                                          }
                                    }
                                    array[n_o] int category_pair
                                          = position_of_category_pair(k_obs, k_obs_upper, n_cat_t);
                                    log_width = log_gap_sum_of_category_pair[category_pair] + log(L_recip);
                              }
//                            vector[n_o] mid = 0.5 * (Bz_lo + Bz_hi);
//                            vector[n_o] narrow_criterion = log_width - (log(1e-5) - log1p(abs(mid)));
//                            int n_narrow = number_of_elements_below(narrow_criterion, 0.0);
                              //// (a row can be narrow only if its log_width < log(1e-5), since
                              ////  log1p(abs(mid)) >= 0: mid and the narrow criterion are computed only when
                              ////  some row has such a width; otherwise no row is narrow, as before)
                              int n_narrow = 0;
                              vector[n_o] mid;
                              vector[n_o] narrow_criterion;
                              if (number_of_elements_below(log_width, log(1e-5)) > 0) {
                                    mid = 0.5 * (Bz_lo + Bz_hi);
                                    narrow_criterion = log_width - (log(1e-5) - log1p(abs(mid)));
                                    n_narrow = number_of_elements_below(narrow_criterion, 0.0);
                              }
//                            array[n_o] int positions_in_order_of_narrow_criterion
//                                  = sort_indices_asc(narrow_criterion);
                              //// (sorted only when a row is narrow; otherwise every row is not narrow, in its
                              ////  order)
                              array[n_o] int positions_in_order_of_narrow_criterion
                                    = linspaced_int_array(n_o, 1, n_o);
                              if (n_narrow > 0) {
                                    positions_in_order_of_narrow_criterion = sort_indices_asc(narrow_criterion);
                              }
                              array[n_narrow] int positions_narrow
                                    = positions_in_order_of_narrow_criterion[1:n_narrow];
                              array[n_o - n_narrow] int positions_not_narrow
                                    = positions_in_order_of_narrow_criterion[(n_narrow + 1):n_o];
                              if (n_narrow > 0) {
                                    int approx_cdf = (Phi_type == 2);
                                    array[n_narrow] int index_narrow = idx[positions_narrow];
                                    vector[n_narrow] mid_narrow = mid[positions_narrow];
                                    vector[n_narrow] log_width_narrow = log_width[positions_narrow];
                                    y1[index_narrow, j_latent]
                                          = narrow_interval_logprob(mid_narrow, log_width_narrow, approx_cdf);
                                    Z_std_norm[index_narrow, j_latent]
                                          = narrow_interval_z(mid_narrow, log_width_narrow,
                                                              u[index_narrow, j_latent], approx_cdf);
                              }


                              //// partition: left = both bounds < UF (Bz_hi < UF); right = both > OF (Bz_lo > OF); else OK
//                            int n_ok = 0; int n_lt = 0; int n_rt = 0;
//                            for (ii in 1:n_o) {
//                                  if      (is_narrow[ii] == 1) continue;
//                                  else if (Bz_hi[ii] < underflow_threshold) n_lt += 1;
//                                  else if (Bz_lo[ii] > overflow_threshold)  n_rt += 1;
//                                  else n_ok += 1;
//                            }
                              //// (no loop over the subjects: the rows that are not narrow, sorted by Bz_hi,
                              ////  start with the n_lt rows with Bz_hi < UF; the others, sorted by Bz_lo, start
                              ////  with the n_ok rows with Bz_lo < OF, and the rest are the right-tail rows (a
                              ////  row exactly at overflow_threshold, of probability zero, is taken as a
                              ////  right-tail row). positions_*: positions in idx.)
                              int n_not_narrow = n_o - n_narrow;
                              vector[n_not_narrow] Bz_hi_not_narrow = Bz_hi[positions_not_narrow];
                              int n_lt = number_of_elements_below(Bz_hi_not_narrow, underflow_threshold);
//                            array[n_not_narrow] int positions_not_narrow_in_order_of_Bz_hi
//                                  = positions_not_narrow[sort_indices_asc(Bz_hi_not_narrow)];
                              //// (sorted by Bz_hi only when a row is in the left tail)
                              array[n_not_narrow] int positions_not_narrow_in_order_of_Bz_hi
                                    = positions_not_narrow;
                              if (n_lt > 0) {
                                    positions_not_narrow_in_order_of_Bz_hi
                                          = positions_not_narrow[sort_indices_asc(Bz_hi_not_narrow)];
                              }
                              array[n_lt] int positions_lt = positions_not_narrow_in_order_of_Bz_hi[1:n_lt];
                              array[n_not_narrow - n_lt] int positions_not_lt
                                    = positions_not_narrow_in_order_of_Bz_hi[(n_lt + 1):n_not_narrow];
                              vector[n_not_narrow - n_lt] Bz_lo_not_lt = Bz_lo[positions_not_lt];
                              int n_ok = number_of_elements_below(Bz_lo_not_lt, overflow_threshold);
                              int n_rt = n_not_narrow - n_lt - n_ok;
//                            array[n_not_narrow - n_lt] int positions_not_lt_in_order_of_Bz_lo
//                                  = positions_not_lt[sort_indices_asc(Bz_lo_not_lt)];
                              //// (sorted by Bz_lo only when a row is in the right tail)
                              array[n_not_narrow - n_lt] int positions_not_lt_in_order_of_Bz_lo
                                    = positions_not_lt;
                              if (n_rt > 0) {
                                    positions_not_lt_in_order_of_Bz_lo
                                          = positions_not_lt[sort_indices_asc(Bz_lo_not_lt)];
                              }
                              array[n_ok] int positions_ok = positions_not_lt_in_order_of_Bz_lo[1:n_ok];
                              array[n_rt] int positions_rt
                                    = positions_not_lt_in_order_of_Bz_lo[(n_ok + 1):(n_ok + n_rt)];
                              array[max(n_ok, 1)] int ok_idx;  vector[max(n_ok, 1)] ok_lo;  vector[max(n_ok, 1)] ok_hi;
                              array[max(n_lt, 1)] int lt_idx;  vector[max(n_lt, 1)] lt_lo;  vector[max(n_lt, 1)] lt_hi;
                              array[max(n_rt, 1)] int rt_idx;  vector[max(n_rt, 1)] rt_lo;  vector[max(n_rt, 1)] rt_hi;
                              {
//                                  int k1 = 1; int k2 = 1; int k3 = 1;
//                                  for (ii in 1:n_o) {
//                                        if      (is_narrow[ii] == 1) continue;
//                                        else if (Bz_hi[ii] < underflow_threshold) { lt_idx[k2] = idx[ii]; lt_lo[k2] = Bz_lo[ii]; lt_hi[k2] = Bz_hi[ii]; k2 += 1; }
//                                        else if (Bz_lo[ii] > overflow_threshold)  { rt_idx[k3] = idx[ii]; rt_lo[k3] = Bz_lo[ii]; rt_hi[k3] = Bz_hi[ii]; k3 += 1; }
//                                        else                                      { ok_idx[k1] = idx[ii]; ok_lo[k1] = Bz_lo[ii]; ok_hi[k1] = Bz_hi[ii]; k1 += 1; }
//                                  }
                                    if (n_ok > 0) {
                                          ok_idx[1:n_ok] = idx[positions_ok];
                                          ok_lo[1:n_ok] = Bz_lo[positions_ok];
                                          ok_hi[1:n_ok] = Bz_hi[positions_ok];
                                    }
                                    if (n_lt > 0) {
                                          lt_idx[1:n_lt] = idx[positions_lt];
                                          lt_lo[1:n_lt] = Bz_lo[positions_lt];
                                          lt_hi[1:n_lt] = Bz_hi[positions_lt];
                                    }
                                    if (n_rt > 0) {
                                          rt_idx[1:n_rt] = idx[positions_rt];
                                          rt_lo[1:n_rt] = Bz_lo[positions_rt];
                                          rt_hi[1:n_rt] = Bz_hi[positions_rt];
                                    }
                              }

                              if (n_ok > 0) {
                                    array[n_ok] int index = ok_idx[1:n_ok];
                                    vector[n_ok] Bound_Z_lo = ok_lo[1:n_ok];
                                    vector[n_ok] Bound_Z_hi = ok_hi[1:n_ok];
                                    if (Phi_type == 2) {
                                          //// Stable Phi_approx interval: SAME target as the log of [Phi_approx at hi minus Phi_approx at lo], computed
                                          //// without cancellation in either tail. With a = poly(lo), b = poly(hi), poly(x) = 0.07056 x^3 + 1.5976 x:
                                          ////     inv_logit(b) - inv_logit(a) = inv_logit(b) * inv_logit(-a) * (1 - exp(-(b - a))),
                                          ////     b - a = (hi - lo) * (0.07056 * (lo^2 + lo * hi + hi^2) + 1.5976)   (exact factorisation).
                                          //// The old probability-scale difference rounded to 0 for Bound_Z_lo > ~7.11 (log(0), NaN adjoint,
                                          //// Z = +Inf) and lost digits from ~5.5 upwards, because 1 minus Phi_approx at 7.5 is 7.4e-19, below machine epsilon.
                                          vector[n_ok] poly_lo = 0.07056 * square(Bound_Z_lo) .* Bound_Z_lo + 1.5976 * Bound_Z_lo;
                                          vector[n_ok] poly_hi = 0.07056 * square(Bound_Z_hi) .* Bound_Z_hi + 1.5976 * Bound_Z_hi;
                                          vector[n_ok] poly_gap = (Bound_Z_hi - Bound_Z_lo) .* (0.07056 * (square(Bound_Z_lo) + Bound_Z_lo .* Bound_Z_hi + square(Bound_Z_hi)) + 1.5976);
                                          vector[n_ok] log_Phi_lo = log_inv_logit(poly_lo);
                                          vector[n_ok] log_Phi_hi = log_inv_logit(poly_hi);
                                          vector[n_ok] log_1m_Phi_lo = log_inv_logit(-poly_lo);
                                          vector[n_ok] log_1m_Phi_hi = log_inv_logit(-poly_hi);
                                          y1[index, j_latent] = log_Phi_hi + log_1m_Phi_lo + log1m_exp(-poly_gap);
                                          //// Z = Phi_approx^{-1}(q), q = Phi_lo + u * (Phi_hi - Phi_lo), via logit(q) = log(q) - log(1 - q) with
                                          //// q = (1 - u) Phi_lo + u Phi_hi and 1 - q = (1 - u)(1 - Phi_lo) + u (1 - Phi_hi): both sums of positive terms.
                                          matrix[n_ok, 2] tmp_lower;
                                          matrix[n_ok, 2] tmp_upper;
                                          tmp_lower[, 1] = log_Phi_lo + log1m(u[index, j_latent]);
                                          tmp_lower[, 2] = log_Phi_hi + log(u[index, j_latent]);
                                          tmp_upper[, 1] = log_1m_Phi_lo + log1m(u[index, j_latent]);
                                          tmp_upper[, 2] = log_1m_Phi_hi + log(u[index, j_latent]);
                                          Z_std_norm[index, j_latent] = inv_Phi_approx_from_logit_prob(log_sum_exp_2d(tmp_lower) - log_sum_exp_2d(tmp_upper));
                                    } else {
                                          //// Phi_type = 1: exact Phi, as BayesMVP's
                                          //// fn_MVOP_compute_lp_GHK_cols_T (exact Phi): the rows with
                                          //// Bound_Z_lo > 0 are reflected, (lo, hi, u) -> (-hi, -lo, 1 - u), so
                                          //// that Phi(hi) - Phi(lo) is never a difference of two numbers close
                                          //// to 1:
                                          ////      a = lo or -hi,  b = hi or -lo,  w = u or 1 - u,  r = +1 or -1,
                                          ////      prob = Phi(b) - Phi(a),   Z = r Phi^{-1}(Phi(a) + w prob),
                                          ////      y1 = log(prob).
                                          //// (no loop over the subjects: the reflected rows are found by
                                          ////  sorting Bound_Z_lo; a row exactly at Bound_Z_lo = 0 is reflected,
                                          ////  with the same values up to rounding)
                                          int n_not_reflected = number_of_elements_below(Bound_Z_lo, 0.0);
                                          int n_reflected = n_ok - n_not_reflected;
//                                        array[n_ok] int rows_in_order_of_Bound_Z_lo
//                                              = sort_indices_asc(Bound_Z_lo);
                                          //// (sorted only when a row is reflected)
                                          array[n_ok] int rows_in_order_of_Bound_Z_lo
                                                = linspaced_int_array(n_ok, 1, n_ok);
                                          if (n_reflected > 0) {
                                                rows_in_order_of_Bound_Z_lo = sort_indices_asc(Bound_Z_lo);
                                          }
                                          array[n_reflected] int rows_reflected
                                                = rows_in_order_of_Bound_Z_lo[(n_not_reflected + 1):n_ok];
                                          vector[n_ok] u_ok = u[index, j_latent];
                                          vector[n_ok] a = Bound_Z_lo;
                                          vector[n_ok] b = Bound_Z_hi;
                                          vector[n_ok] w = u_ok;
                                          vector[n_ok] r = rep_vector(1.0, n_ok);
                                          if (n_reflected > 0) {
                                                a[rows_reflected] = -Bound_Z_hi[rows_reflected];
                                                b[rows_reflected] = -Bound_Z_lo[rows_reflected];
                                                w[rows_reflected] = 1.0 - u_ok[rows_reflected];
                                                r[rows_reflected] = rep_vector(-1.0, n_reflected);
                                          }
                                          vector[n_ok] Phi_a = Phi(a);
                                          vector[n_ok] prob_t = Phi(b) - Phi_a;
                                          Z_std_norm[index, j_latent] = r .* inv_Phi(Phi_a + w .* prob_t);
                                          y1[index, j_latent] = log(prob_t);
                                    }
                              }
                              if (n_lt > 0) {   //// BOTH bounds in the left tail
                                    array[n_lt] int index = lt_idx[1:n_lt];
                                    vector[n_lt] Bound_Z_lo = lt_lo[1:n_lt];
                                    vector[n_lt] Bound_Z_hi = lt_hi[1:n_lt];
                                    //// the log-scale tail as the lower-tail case of BayesMVP's C++ MVOP
                                    //// log-scale tails, with each Phi_type's own functions (Phi_type = 2:
                                    //// log_Phi_approx() and inv_Phi_approx_from_logit_prob(); Phi_type = 1:
                                    //// log_Phi() and inv_Phi_from_log_p()):
                                    ////      y1 = log Phi(hi) + log1m_exp(log Phi(lo) - log Phi(hi)),
                                    ////      log Phi_Z = log_sum_exp(log Phi(lo) + log(1 - u),
                                    ////                              log Phi(hi) + log(u)),
                                    ////      Z = Phi^{-1}(Phi_Z) from log Phi_Z and
                                    ////      log(1 - Phi_Z) = log1m_exp(log Phi_Z).
                                    //// (before, for Phi_type = 2:)
//                                        vector[n_lt] log_Phi_lo = log_inv_logit(0.07056 * square(Bound_Z_lo) .* Bound_Z_lo + 1.5976 * Bound_Z_lo);
//                                        vector[n_lt] log_Phi_hi = log_inv_logit(0.07056 * square(Bound_Z_hi) .* Bound_Z_hi + 1.5976 * Bound_Z_hi);
//                                        y1[index, j_latent] = log_Phi_hi + log1m_exp(log_Phi_lo - log_Phi_hi);
//                                        matrix[n_lt, 2] tmp;
//                                        tmp[, 1] = log_Phi_lo + log1m(u[index, j_latent]);
//                                        tmp[, 2] = log_Phi_hi + log(u[index, j_latent]);
//                                        vector[n_lt] log_Phi_Z = log_sum_exp_2d(tmp);
//                                        Z_std_norm[index, j_latent] = inv_Phi_approx_from_logit_prob(log_Phi_Z - log1m_exp(log_Phi_Z));
                                    vector[n_lt] log_Phi_lo;
                                    vector[n_lt] log_Phi_hi;
                                    if (Phi_type == 2) {
                                          log_Phi_lo = log_Phi_approx(Bound_Z_lo);
                                          log_Phi_hi = log_Phi_approx(Bound_Z_hi);
                                    } else {
                                          log_Phi_lo = log_Phi(Bound_Z_lo);
                                          log_Phi_hi = log_Phi(Bound_Z_hi);
                                    }
                                    y1[index, j_latent] = log_Phi_hi + log1m_exp(log_Phi_lo - log_Phi_hi);
                                    matrix[n_lt, 2] tmp;
                                    tmp[, 1] = log_Phi_lo + log1m(u[index, j_latent]);
                                    tmp[, 2] = log_Phi_hi + log(u[index, j_latent]);
                                    vector[n_lt] log_Phi_Z = log_sum_exp_2d(tmp);
                                    vector[n_lt] log_1m_Phi_Z = log1m_exp(log_Phi_Z);
                                    if (Phi_type == 2) {
                                          Z_std_norm[index, j_latent]
                                                = inv_Phi_approx_from_logit_prob(log_Phi_Z - log_1m_Phi_Z);
                                    } else {
                                          Z_std_norm[index, j_latent]
                                                = inv_Phi_from_log_p(log_Phi_Z, log_1m_Phi_Z);
                                    }
                              }
                              if (n_rt > 0) {   //// BOTH bounds in the right tail
                                    array[n_rt] int index = rt_idx[1:n_rt];
                                    vector[n_rt] Bound_Z_lo = rt_lo[1:n_rt];
                                    vector[n_rt] Bound_Z_hi = rt_hi[1:n_rt];
                                    //// the log-scale tail as the upper-tail case of BayesMVP's C++ MVOP
                                    //// log-scale tails, with each Phi_type's own functions, and the survival
                                    //// function 1 - Phi(x) = Phi(-x):
                                    ////      y1 = log Phi(-lo) + log1m_exp(log Phi(-hi) - log Phi(-lo)),
                                    ////      log(1 - Phi_Z) = log_sum_exp(log Phi(-lo) + log(1 - u),
                                    ////                                   log Phi(-hi) + log(u)),
                                    ////      Z = Phi^{-1}(Phi_Z) from log Phi_Z = log1m_exp(log(1 - Phi_Z)) and
                                    ////      log(1 - Phi_Z).
                                    //// (before, for Phi_type = 2:)
//                                        vector[n_rt] log_1m_Phi_lo = log_inv_logit(- 0.07056 * square(Bound_Z_lo) .* Bound_Z_lo - 1.5976 * Bound_Z_lo);
//                                        vector[n_rt] log_1m_Phi_hi = log_inv_logit(- 0.07056 * square(Bound_Z_hi) .* Bound_Z_hi - 1.5976 * Bound_Z_hi);
//                                        y1[index, j_latent] = log_1m_Phi_lo + log1m_exp(log_1m_Phi_hi - log_1m_Phi_lo);
//                                        matrix[n_rt, 2] tmp;
//                                        tmp[, 1] = log_1m_Phi_lo + log1m(u[index, j_latent]);
//                                        tmp[, 2] = log_1m_Phi_hi + log(u[index, j_latent]);
//                                        vector[n_rt] log_1m_Phi_Z = log_sum_exp_2d(tmp);
//                                        vector[n_rt] log_Phi_Z = log1m_exp(log_1m_Phi_Z);
//                                        Z_std_norm[index, j_latent] = inv_Phi_approx_from_logit_prob(log_Phi_Z - log_1m_Phi_Z);
                                    vector[n_rt] log_1m_Phi_lo;
                                    vector[n_rt] log_1m_Phi_hi;
                                    if (Phi_type == 2) {
                                          log_1m_Phi_lo = log_Phi_approx(-Bound_Z_lo);
                                          log_1m_Phi_hi = log_Phi_approx(-Bound_Z_hi);
                                    } else {
                                          log_1m_Phi_lo = log_Phi(-Bound_Z_lo);
                                          log_1m_Phi_hi = log_Phi(-Bound_Z_hi);
                                    }
                                    y1[index, j_latent] = log_1m_Phi_lo
                                                          + log1m_exp(log_1m_Phi_hi - log_1m_Phi_lo);
                                    matrix[n_rt, 2] tmp;
                                    tmp[, 1] = log_1m_Phi_lo + log1m(u[index, j_latent]);
                                    tmp[, 2] = log_1m_Phi_hi + log(u[index, j_latent]);
                                    vector[n_rt] log_1m_Phi_Z = log_sum_exp_2d(tmp);
                                    vector[n_rt] log_Phi_Z = log1m_exp(log_1m_Phi_Z);
                                    if (Phi_type == 2) {
                                          Z_std_norm[index, j_latent]
                                                = inv_Phi_approx_from_logit_prob(log_Phi_Z - log_1m_Phi_Z);
                                    } else {
                                          Z_std_norm[index, j_latent]
                                                = inv_Phi_from_log_p(log_Phi_Z, log_1m_Phi_Z);
                                    }
                              }

                        } //// end binary / ordinal

                  } //// end n_o > 0

                  // if (t < n_tests)   inc = block(Z_std_norm, 1, 1, M, t) * to_vector(head(L_Omega[c, t + 1, ], t));
                  if (j_latent < n_tests)
                        inc = block(Z_std_norm, 1, 1, M, j_latent) * to_vector(head(L_Omega[c, j_latent + 1, ], j_latent));

            }  //// end t loop

            lp[, c] = rowwise_sum(y1[, 1:n_tests])  +  to_vector(log_prev[, c]);

      } //// end c loop


//          out += sum(log_sum_exp_matrix(lp));
            out += sum(log_sum_exp_2d(lp));   //// (nico_log_sum_exp_2d in the AVX2 and AVX-512 versions)
            return out;

      }


      //// ---- (commented out: the marginal log-likelihood for LOO below, log_lik_obs_ghk_rng() and
      ////  log_lik_obs_ghk_of_every_subject_rng(), integrates the missing tests out over the observed
      ////  tests only - by the Cholesky factor of their correlation matrix - not by the GHK method of the
      ////  missing tests used everywhere else in this model (their untruncated draws in the conditioning
      ////  chain); kept as comments; n_loo_draws > 0 is rejected in transformed data)
      //////////////////////////////////////////////////////////////////////////
      // ---- MARGINAL per-observation log-likelihood for LOO, by GHK Monte Carlo.
      //// log p(y_n | theta) = log E_u[ prod_t p(y_nt | z_<t, theta) ] estimated with R fresh
      //// uniform draws. This is what PSIS-LOO needs. The augmented log p(y_n | theta, u_n)
      //// (one draw, the sampled u_n) is NOT usable for LOO: leaving y_n out leaves u_n
      //// unconstrained and the importance weights degenerate. Uses std_normal_lcdf / lccdf
      //// for tail stability; GQ only, so no autodiff cost.
      //////////////////////////////////////////////////////////////////////////
//    vector log_lik_obs_ghk_rng( int n, int R,
//                              data matrix y_use, data array[,] int obs_mask, data array[,] int y_ord,
//                              data array[,] int y_ord_upper,
//                              data array[] matrix X, data array[] int n_covs_per_outcome,
//                              data array[] int pop, data array[,] int class_D, data array[] int test_condition,
//                              matrix beta, array[] matrix beta_cov, array[] matrix L_Omega,
//                              matrix log_prev_subject, array[] vector C_vec_tp, array[] vector log_spacing_tp,
//                              data array[] int ord_start_index, data array[] int ord_end_index,
//                              data array[] int n_thr_per_ord_test, data array[] int n_cat_per_ord_test,
//                              data int n_binary_tests, data array[] int latent_test_order,
//                              data array[] int latent_position_of_test, data real C_sentinel,
//                              data int n_tests, data int n_class, data int n_covariates_max) {

//          vector[n_class] lp_class;
//          int P = 0;
//          array[n_tests] int test_buffer;
            // for (t in 1:n_tests) if (obs_mask[n, t] == 1) { P += 1; test_buffer[P] = t; }
//          array[n_tests] int latent_buffer;
//          for (j_latent in 1:n_tests) {
//                int t_latent = latent_test_order[j_latent];
//                if (obs_mask[n, t_latent] == 1) { P += 1; test_buffer[P] = t_latent; latent_buffer[P] = j_latent; }
//          }
//          if (P == 0) return to_vector(log_prev_subject[n]);
//          array[P] int tests = test_buffer[1:P];
//          array[P] int latent_positions = latent_buffer[1:P];
//          for (c in 1:n_class) {
//                matrix[P, P] L;
//                vector[R] lp_r;
//                if (P == n_tests) L = L_Omega[c];
//                else {
//                    matrix[n_tests, n_tests] Sigma = multiply_lower_tri_self_transpose(L_Omega[c]);
//                    L = cholesky_decompose(Sigma[latent_positions, latent_positions]);
//                }
//                for (r in 1:R) {
//                      real lp = 0.0;
//                      vector[P] z = rep_vector(0.0, P);
//                      for (j in 1:P) {
//                            int t = tests[j];
//                            int nc = n_covs_per_outcome[t];
//                            real mu = beta[c, t];
//                            real lo;
//                            real hi;
//                            real log_width = positive_infinity();
//                            if (nc > 0) mu += X[t][n, 1:nc] * beta_cov[class_D[c, test_condition[t]] + 1][1:nc, t];
//                            if (j > 1) mu += dot_product(L[j, 1:(j - 1)], z[1:(j - 1)]);
//                            if (t <= n_binary_tests) {
//                                  real b = -mu / L[j, j];
//                                  lo = y_use[n, t] == 1 ? b : negative_infinity();
//                                  hi = y_use[n, t] == 1 ? positive_infinity() : b;
//                            } else {
//                                  int to = t - n_binary_tests;
//                                  int k = y_ord[to, n];
//                                  int ss = class_D[c, test_condition[t]] + 1;
//                                  int first = ord_start_index[to];
//                                  lo = k == 1 ? negative_infinity() : (C_vec_tp[ss][first + k - 2] - mu) / L[j, j];
//                                  hi = k == n_cat_per_ord_test[to] ? positive_infinity()
//                                       : (C_vec_tp[ss][first + k - 1] - mu) / L[j, j];
//                                  if (k > 1 && k < n_cat_per_ord_test[to])
//                                      log_width = log_spacing_tp[ss][first + k - 1] - log(L[j, j]);
//                                  int k = y_ord[to, n];
//                                  int k_upper = y_ord_upper[to, n];   //// = k unless interval-censored
//                                  int ss = class_D[c, test_condition[t]] + 1;
//                                  int first = ord_start_index[to];
//                                  lo = k == 1 ? negative_infinity() : (C_vec_tp[ss][first + k - 2] - mu) / L[j, j];
//                                  hi = k_upper == n_cat_per_ord_test[to] ? positive_infinity()
//                                       : (C_vec_tp[ss][first + k_upper - 1] - mu) / L[j, j];
                                    //// the log width of categories k..k_upper = log of the sum of their spacings:
//                                  if (k > 1 && k_upper < n_cat_per_ord_test[to]) {
//                                      if (k_upper == k)
//                                          log_width = log_spacing_tp[ss][first + k - 1] - log(L[j, j]);
//                                      else
//                                          log_width = log_sum_exp(log_spacing_tp[ss][(first + k - 1):(first + k_upper - 1)])
//                                                      - log(L[j, j]);
//                                  }
//                            }
//                            if (j < P) {
//                                real u = uniform_rng(0.0, 1.0);
//                                while (u <= 0.0 || u >= 1.0) u = uniform_rng(0.0, 1.0);
//                                vector[2] step = normal_interval_step(lo, hi, log_width, log(u), log1m(u));
//                                z[j] = step[1]; lp += step[2];
//                            } else lp += normal_interval_mass(lo, hi, log_width);
//                      }
//                      lp_r[r] = lp;
//                }
//                lp_class[c] = log_sum_exp(lp_r) - log(R) + log_prev_subject[n, c];
//          }
            //// returns the JOINT log p(y_n, class = c) for each class. log_sum_exp of it is the
            //// marginal log-likelihood for LOO; softmax of it is the posterior class probability
            //// for that subject, which is the diagnostic quantity.
//          return lp_class;

//    }


      //////////////////////////////////////////////////////////////////////////
      // ---- The same marginal log-likelihood for LOO for every subject at once, without a loop over the
      // ---- subjects.
      //// The functions below are for generated quantities only (no autodiff).
      //////////////////////////////////////////////////////////////////////////
      ////
      //// ---- The log probabilities of the intervals (lo, hi) of the standard normal (as
      //// normal_interval_mass()), for a vector of intervals, with the exact log normal CDF log_Phi() of the
      //// likelihood's tails: lo = -Inf: log Phi(hi); hi = +Inf: log Phi(-lo); otherwise the expansion of
      //// normal_interval_logprob() for the narrow intervals (log_width + log1p(|mid|) < log(1e-3)),
      //// log(Phi(-lo) - Phi(-hi)) (reflected) for lo >= 0 and log(Phi(hi) - Phi(lo)) for the others. (no loop
      //// over the intervals: the open, narrow and reflected intervals are found by sorting; an interval exactly
      //// at lo = 0, of probability zero, is reflected, with the same value up to rounding)
      ////
      vector log_probability_of_normal_intervals(vector lo, vector hi, vector log_width) {
            int n = num_elements(lo);
            vector[n] lo_copy = lo;
            vector[n] hi_copy = hi;
            vector[n] log_probability;
            //// (sorted by lo, the left-open intervals come first; the others, sorted by hi, end with the
            ////  right-open ones; -1.7976931348623157e308 and 1.7976931348623157e308 are the largest finite
            ////  numbers)
            int n_left_open = number_of_elements_below(lo_copy, -1.7976931348623157e308);
            array[n] int intervals_in_order_of_lo = sort_indices_asc(lo_copy);
            array[n_left_open] int left_open = intervals_in_order_of_lo[1:n_left_open];
            array[n - n_left_open] int not_left_open = intervals_in_order_of_lo[(n_left_open + 1):n];
            vector[n - n_left_open] hi_not_left_open = hi_copy[not_left_open];
            int n_closed = number_of_elements_below(hi_not_left_open, 1.7976931348623157e308);
            int n_right_open = n - n_left_open - n_closed;
            array[n - n_left_open] int not_left_open_in_order_of_hi
                  = not_left_open[sort_indices_asc(hi_not_left_open)];
            array[n_closed] int closed = not_left_open_in_order_of_hi[1:n_closed];
            array[n_right_open] int right_open = not_left_open_in_order_of_hi[(n_closed + 1):(n - n_left_open)];
            if (n_left_open > 0) log_probability[left_open] = log_Phi(hi_copy[left_open]);
            if (n_right_open > 0) log_probability[right_open] = log_Phi(-lo_copy[right_open]);
            if (n_closed > 0) {
                  vector[n_closed] lo_closed = lo_copy[closed];
                  vector[n_closed] hi_closed = hi_copy[closed];
                  vector[n_closed] log_width_closed = log_width[closed];
                  vector[n_closed] width_closed = exp(log_width_closed);
                  vector[n_closed] mid_closed = lo_closed + 0.5 * width_closed;
                  vector[n_closed] narrow_criterion = log_width_closed + log1p(abs(mid_closed)) - log(1e-3);
                  int n_narrow = number_of_elements_below(narrow_criterion, 0.0);
                  array[n_closed] int positions_in_order_of_narrow_criterion = sort_indices_asc(narrow_criterion);
                  array[n_narrow] int positions_narrow = positions_in_order_of_narrow_criterion[1:n_narrow];
                  array[n_closed - n_narrow] int positions_not_narrow
                        = positions_in_order_of_narrow_criterion[(n_narrow + 1):n_closed];
                  if (n_narrow > 0) {
                        vector[n_narrow] w2 = square(width_closed[positions_narrow]);
                        vector[n_narrow] m2 = square(mid_closed[positions_narrow]);
                        vector[n_narrow] correction = (m2 - 1.0) .* w2 / 24.0
                                                      + (square(m2) - 6.0 * m2 + 3.0) .* square(w2) / 1920.0;
                        log_probability[closed[positions_narrow]] = log_width_closed[positions_narrow]
                                                                    - 0.5 * m2 - 0.5 * log(2.0 * pi())
                                                                    + log1p(correction);
                  }
                  if (n_narrow < n_closed) {
                        int n_not_narrow = n_closed - n_narrow;
                        vector[n_not_narrow] lo_not_narrow = lo_closed[positions_not_narrow];
                        int n_not_reflected = number_of_elements_below(lo_not_narrow, 0.0);
                        array[n_not_narrow] int positions_not_narrow_in_order_of_lo
                              = positions_not_narrow[sort_indices_asc(lo_not_narrow)];
                        array[n_not_reflected] int positions_not_reflected
                              = positions_not_narrow_in_order_of_lo[1:n_not_reflected];
                        array[n_not_narrow - n_not_reflected] int positions_reflected
                              = positions_not_narrow_in_order_of_lo[(n_not_reflected + 1):n_not_narrow];
                        if (n_not_reflected > 0) {
                              log_probability[closed[positions_not_reflected]]
                                    = log_diff_exp(log_Phi(hi_closed[positions_not_reflected]),
                                                   log_Phi(lo_closed[positions_not_reflected]));
                        }
                        if (n_not_reflected < n_not_narrow) {
                              log_probability[closed[positions_reflected]]
                                    = log_diff_exp(log_Phi(-lo_closed[positions_reflected]),
                                                   log_Phi(-hi_closed[positions_reflected]));
                        }
                  }
            }
            return log_probability;
      }
      ////
      //// ---- One step of the GHK recursion for a vector of intervals (as normal_interval_step()): the draw
      //// z = Phi^{-1}(q) of the standard normal truncated to (lo, hi), q = (1 - u) Phi(lo) + u Phi(hi), from
      //// log q = log_sum_exp(log(1 - u) + log Phi(lo), log(u) + log Phi(hi)) and
      //// log(1 - q) = log_sum_exp(log(1 - u) + log Phi(-lo), log(u) + log Phi(-hi)) with inv_Phi_from_log_p(),
      //// and the log probability of the interval (column 1: z; column 2: the log probability).
      ////
      matrix truncated_normal_draws_and_log_probabilities(vector lo, vector hi, vector log_width,
                                                          vector log_u, vector log1m_u) {
            int n = num_elements(lo);
            matrix[n, 2] out;
            matrix[n, 2] terms_of_log_q;
            matrix[n, 2] terms_of_log_1m_q;
            terms_of_log_q[, 1] = log1m_u + log_Phi(lo);
            terms_of_log_q[, 2] = log_u + log_Phi(hi);
            terms_of_log_1m_q[, 1] = log1m_u + log_Phi(-lo);
            terms_of_log_1m_q[, 2] = log_u + log_Phi(-hi);
            out[, 1] = inv_Phi_from_log_p(log_sum_exp_2d(terms_of_log_q), log_sum_exp_2d(terms_of_log_1m_q));
            out[, 2] = log_probability_of_normal_intervals(lo, hi, log_width);
            return out;
      }
      ////
      //// ---- n uniform draws strictly between 0 and 1 (uniform_rng(0, 1) can return 0: such a draw is drawn
      //// again, as in log_lik_obs_ghk_rng(); the while loop runs only while a draw is 0 or 1)
      ////
      vector uniform_draws_strictly_between_0_and_1_rng(int n) {
            vector[n] u = to_vector(uniform_rng(rep_vector(0.0, n), 1.0));
            while (min(u) <= 0.0 || max(u) >= 1.0) {
                  vector[n] u_copy = u;
                  //// (4.9406564584124654e-324 is the smallest positive number)
                  int n_at_0 = number_of_elements_below(u_copy, 4.9406564584124654e-324);
                  int n_below_1 = number_of_elements_below(u_copy, 1.0);
                  array[n] int draws_in_order_of_u = sort_indices_asc(u_copy);
                  array[n - n_below_1 + n_at_0] int draws_again
                        = append_array(draws_in_order_of_u[1:n_at_0], draws_in_order_of_u[(n_below_1 + 1):n]);
                  u[draws_again] = to_vector(uniform_rng(rep_vector(0.0, n - n_below_1 + n_at_0), 1.0));
            }
            return u;
      }
      //// (commented out, as log_lik_obs_ghk_rng() above: the missing tests integrated out over the
      ////  observed tests only, not by the GHK method)
      ////
      //// ---- log_lik_obs_ghk_rng() for every subject at once: the joint log p(y_n, class = c) of every subject
      //// (rows) and class (columns). The subjects are grouped by the pattern of their observed tests
      //// (subjects_in_order_of_observed_test_pattern and n_subjects_of_observed_test_pattern, transformed data;
      //// pattern code = the sum of 2^(j - 1) over the latent positions j of the observed tests); for each
      //// pattern and class, the GHK recursion over the P observed tests, with the Cholesky factor of their
      //// correlation matrix, runs on the vector of the pattern's subjects, R times (a loop over the patterns,
      //// classes, draws and tests only). A binary test is an ordinal test with one cutpoint, 0 (category y + 1).
      ////
//    matrix log_lik_obs_ghk_of_every_subject_rng(int R,
//                              data matrix y_use, data array[,] int y_ord, data array[,] int y_ord_upper,
//                              data array[] matrix X, data array[] int n_covs_per_outcome,
//                              data array[,] int class_D, data array[] int test_condition,
//                              matrix beta, array[] matrix beta_cov, array[] matrix L_Omega,
//                              matrix log_prev_subject, array[] vector C_vec_tp, array[] vector log_spacing_tp,
//                              data array[] int ord_start_index, data array[] int ord_end_index,
//                              data array[] int n_thr_per_ord_test, data array[] int n_cat_per_ord_test,
//                              data int n_binary_tests, data array[] int latent_test_order,
//                              data array[] int subjects_in_order_of_observed_test_pattern,
//                              data array[] int n_subjects_of_observed_test_pattern,
//                              data int n_tests, data int n_class) {

//          int N = rows(y_use);
//          int n_patterns = size(n_subjects_of_observed_test_pattern);
//          matrix[N, n_class] lp_class;
//          int n_subjects_of_previous_patterns = 0;
//          for (pattern in 0:(n_patterns - 1)) {
//                int M = n_subjects_of_observed_test_pattern[pattern + 1];
//                if (M == 0) continue;
//                int first = n_subjects_of_previous_patterns + 1;
//                int last = n_subjects_of_previous_patterns + M;
//                array[M] int subjects = subjects_in_order_of_observed_test_pattern[first:last];
//                n_subjects_of_previous_patterns += M;
                  ////
                  //// ---- the pattern's observed tests, in the latent order:
                  ////
//                int P = 0;
//                array[n_tests] int test_buffer;
//                array[n_tests] int latent_buffer;
//                int remaining_code = pattern;
//                for (j_latent in 1:n_tests) {
//                      if (remaining_code % 2 == 1) {
//                            P += 1;
//                            test_buffer[P] = latent_test_order[j_latent];
//                            latent_buffer[P] = j_latent;
//                      }
//                      remaining_code = remaining_code %/% 2;
//                }
//                if (P == 0) {
//                      lp_class[subjects] = log_prev_subject[subjects];
//                      continue;
//                }
//                array[P] int tests = test_buffer[1:P];
//                array[P] int latent_positions = latent_buffer[1:P];
//                for (c in 1:n_class) {
//                      matrix[P, P] L;
//                      if (P == n_tests) L = L_Omega[c];
//                      else {
//                          matrix[n_tests, n_tests] Sigma = multiply_lower_tri_self_transpose(L_Omega[c]);
//                          L = cholesky_decompose(Sigma[latent_positions, latent_positions]);
//                      }
                        ////
                        //// ---- for each observed test: the mean without the conditioning on the earlier tests,
                        //// the cutpoints of each subject's lower and upper categories (-Inf / +Inf at the open
                        //// ends) and the log of the summed gaps of its categories (+Inf where an end is open):
                        ////
//                      matrix[M, P] mean_of_test;
//                      matrix[M, P] C_lo;
//                      matrix[M, P] C_hi;
//                      matrix[M, P] log_gap_sum;
//                      for (j in 1:P) {
//                            int t = tests[j];
//                            int nc = n_covs_per_outcome[t];
//                            int ss = class_D[c, test_condition[t]] + 1;
//                            mean_of_test[, j] = rep_vector(beta[c, t], M);
//                            if (nc > 0) mean_of_test[, j] += X[t][subjects, 1:nc] * beta_cov[ss][1:nc, t];
//                            if (t <= n_binary_tests) {
//                                  array[M] int category = to_int(to_array_1d(y_use[subjects, t] + 1.0));
//                                  vector[2] C_lo_of_category = [negative_infinity(), 0.0]';
//                                  vector[2] C_hi_of_category = [0.0, positive_infinity()]';
//                                  C_lo[, j] = C_lo_of_category[category];
//                                  C_hi[, j] = C_hi_of_category[category];
//                                  log_gap_sum[, j] = rep_vector(positive_infinity(), M);
//                            } else {
//                                  int to = t - n_binary_tests;
//                                  int n_cat_t = n_cat_per_ord_test[to];
//                                  vector[n_thr_per_ord_test[to]] C_t
//                                        = get_test_values(C_vec_tp[ss], ord_start_index, ord_end_index, to);
//                                  vector[n_thr_per_ord_test[to]] log_gap_t
//                                        = get_test_values(log_spacing_tp[ss], ord_start_index, ord_end_index,
//                                                          to);
//                                  array[M] int k = y_ord[to, subjects];
//                                  array[M] int k_upper = y_ord_upper[to, subjects];   //// = k unless censored
//                                  C_lo[, j] = append_row(negative_infinity(), C_t)[k];
//                                  C_hi[, j] = append_row(C_t, positive_infinity())[k_upper];
                                    //// (the log of the summed gaps of categories k..k_upper, for each pair of
                                    ////  interior categories, as in partial_log_lik())
//                                  vector[n_cat_t * n_cat_t] log_gap_sum_of_category_pair
//                                        = rep_vector(positive_infinity(), n_cat_t * n_cat_t);
//                                  for (k_low in 2:(n_cat_t - 1)) {
//                                        for (k_up in k_low:(n_cat_t - 1)) {
//                                              log_gap_sum_of_category_pair[(k_low - 1) * n_cat_t + k_up]
//                                                    = (k_up == k_low ? log_gap_t[k_low]
//                                                                     : log_sum_exp(log_gap_t[k_low:k_up]));
//                                        }
//                                  }
//                                  array[M] int category_pair = position_of_category_pair(k, k_upper, n_cat_t);
//                                  log_gap_sum[, j] = log_gap_sum_of_category_pair[category_pair];
//                            }
//                      }
                        ////
                        //// ---- the GHK recursion, R times:
                        ////
//                      matrix[M, R] lp_r = rep_matrix(0.0, M, R);
//                      for (r in 1:R) {
//                            matrix[M, P] z;
//                            for (j in 1:P) {
//                                  vector[M] mu = mean_of_test[, j];
//                                  if (j > 1) mu += z[, 1:(j - 1)] * L[j, 1:(j - 1)]';
//                                  vector[M] lo = (C_lo[, j] - mu) / L[j, j];
//                                  vector[M] hi = (C_hi[, j] - mu) / L[j, j];
//                                  vector[M] log_width = log_gap_sum[, j] - log(L[j, j]);
//                                  if (j < P) {
//                                        vector[M] u = uniform_draws_strictly_between_0_and_1_rng(M);
//                                        matrix[M, 2] step
//                                              = truncated_normal_draws_and_log_probabilities(lo, hi, log_width,
//                                                                                             log(u), log1m(u));
//                                        z[, j] = step[, 1];
//                                        lp_r[, r] += step[, 2];
//                                  } else lp_r[, r] += log_probability_of_normal_intervals(lo, hi, log_width);
//                            }
//                      }
//                      lp_class[subjects, c] = log_sum_exp_2d(lp_r) - log(R) + log_prev_subject[subjects, c];
//                }
//          }
//          return lp_class;

//    }


      //////////////////////////////////////////////////////////////////////////
      // ---- P(Z1 > a, Z2 > b) for a standard bivariate normal with correlation rho.
      //// Needed for the screen-threshold predictive values: the joint probability that BOTH
      //// screens exceed a cutoff, within a latent class, is exactly this orthant.
      ////
      //// Integrates over the outer variable on the PROBABILITY scale, u = Phi(z), so the limits
      //// are exact (u from Phi(a) to 1) and the integrand - a normal CDF in z - is smooth, which
      //// makes the midpoint rule converge fast. GQ only, so the node count is free.
      //////////////////////////////////////////////////////////////////////////
//    real biv_norm_upper(real a, real b, real rho, data int n_node) {
//
//          real rho_c = fmin(0.999, fmax(-0.999, rho));
//          real s     = sqrt(1.0 - square(rho_c));
//          real Pa    = Phi(a);
//          if (Pa >= 1.0 - 1e-12) return 0.0;
//          real acc = 0.0;
//          for (i in 1:n_node) {
//                real u = Pa + (1.0 - Pa) * (i - 0.5) / n_node;
//                real z = inv_Phi(fmin(1.0 - 1e-12, fmax(1e-12, u)));
//                acc += Phi(-(b - rho_c * z) / s);
//          }
//          return (1.0 - Pa) * acc / n_node;
//
//    }
//    ////
//    //// (the same, for vectors of standardisation profiles: the same operations, in the same order, on whole
//    ////  vectors; is_below_the_upper_limit is 1 where Pa < 1 - 1e-12 and 0 elsewhere, where the version above
//    ////  returns 0)
//    vector biv_norm_upper(vector a, vector b, real rho, data int n_node) {
//
//          int n = num_elements(a);
//          real rho_c = fmin(0.999, fmax(-0.999, rho));
//          real s     = sqrt(1.0 - square(rho_c));
//          vector[n] Pa = Phi(a);
//          vector[n] is_below_the_upper_limit = fmin(1.0, ceil(fmax(0.0, (1.0 - 1e-12) - Pa)));
//          vector[n] acc = rep_vector(0.0, n);
//          for (i in 1:n_node) {
//                vector[n] u = Pa + (1.0 - Pa) * (i - 0.5) / n_node;
//                vector[n] z = inv_Phi(fmin(1.0 - 1e-12, fmax(1e-12, u)));
//                acc += Phi(-(b - rho_c * z) / s);
//          }
//          return is_below_the_upper_limit .* (1.0 - Pa) .* acc / n_node;
//
//    }
      ////
      //// (the same quadrature, for vectors of standardisation profiles, with no bound and no cut-off: the outer
      ////  variable on the scale of its UPPER-TAIL probability t = P(Z1 > z), from 0 to P(Z1 > a), whose log is
      ////  exact (log_Phi), so that no node is 1 minus something small and P(Z1 > a) can be as small as the log
      ////  scale allows; the node z_i = -std_normal_log_qf(log P(Z1 > a) + log((n_node - i + 0.5) / n_node)) is
      ////  the node inv_Phi(Phi(a) + (1 - Phi(a)) (i - 0.5) / n_node) of the version above, and sqrt(1 - rho^2)
      ////  is written as sqrt((1 - rho) (1 + rho)). Returns, for each element, log P(Z1 > a) and the conditional
      ////  probabilities P(Z2 > b | Z1 > a) and P(Z2 <= b | Z1 > a), both from the same nodes, with no
      ////  subtraction: the orthants P(Z1 > a, Z2 > b) and P(Z1 > a, Z2 <= b) are exp(column 1) times column 2
      ////  and column 3)
      matrix biv_norm_log_Pr_Z1_above_a_and_Pr_Z2_given_Z1_above_a(vector a, vector b, real rho,
                                                                   data int n_node) {

            int n = num_elements(a);
            real s = sqrt((1.0 - rho) * (1.0 + rho));
            vector[n] log_Pr_Z1_above_a = log_Phi(-a);
            vector[n] sum_of_Pr_Z2_above_b = rep_vector(0.0, n);
            vector[n] sum_of_Pr_Z2_not_above_b = rep_vector(0.0, n);
            for (i in 1:n_node) {
                  vector[n] z = -std_normal_log_qf(log_Pr_Z1_above_a + log((n_node - i + 0.5) / n_node));
                  sum_of_Pr_Z2_above_b += Phi(-(b - rho * z) / s);
                  sum_of_Pr_Z2_not_above_b += Phi((b - rho * z) / s);
            }
            matrix[n, 3] log_Pr_Z1_above_a_and_Pr_Z2_given_Z1_above_a;
            log_Pr_Z1_above_a_and_Pr_Z2_given_Z1_above_a[, 1] = log_Pr_Z1_above_a;
            log_Pr_Z1_above_a_and_Pr_Z2_given_Z1_above_a[, 2] = sum_of_Pr_Z2_above_b / n_node;
            log_Pr_Z1_above_a_and_Pr_Z2_given_Z1_above_a[, 3] = sum_of_Pr_Z2_not_above_b / n_node;
            return log_Pr_Z1_above_a_and_Pr_Z2_given_Z1_above_a;

      }


      //////////////////////////////////////////////////////////////////////////
      // ---- The 2x2 cell probability P(D_A = 1, D_B = 1) implied by the two MARGINAL
      //// prevalences and the ODDS RATIO (Plackett's construction).
      ////
      //// WHY THIS EXISTS. The Dirichlet couples beliefs about the marginal prevalences and the
      //// association. The OR of its mean table is not the mean / median of the prior OR, and the
      //// log-OR of that table is not E[log_OR]. Work directly with margins and log_OR when those
      //// are the quantities on which beliefs are specified, and inspect their induced prior draws.
      ////
      //// Parameterising by (prev_A, prev_B, log_OR) instead decouples them. The map is a BIJECTION
      //// onto the interior of the simplex, so nothing is lost: a 2x2 table with positive cells is
      //// uniquely determined by its margins and its odds ratio.
      ////
      //// NUMERICS. Solving psi = x(1-pA-pB+x) / ((pA-x)(pB-x)) for x gives a quadratic whose
      //// relevant root is x = (-b - sqrt(D)) / (2a) with a = psi - 1 - which is 0/0 at psi = 1. The
      //// algebraically identical "citardauq" form x = 2c / (-b + sqrt(D)) has no 1/(psi-1) and is
      //// smooth THROUGH psi = 1, returning exactly pA * pB there, so no special case is needed.
      //////////////////////////////////////////////////////////////////////////
//    real plackett_p11(real pA, real pB, real psi) {
//
//          real a  = psi - 1.0;
//          real nb = 1.0 + (pA + pB) * a;            //// = -b
//          real c  = psi * pA * pB;
//          real D  = fmax(nb * nb - 4.0 * a * c, 0.0);
//          return 2.0 * c / (nb + sqrt(D));
//
//    }
//    ////
//    //// (the same, for vectors of subjects: the same operations, in the same order, on whole vectors)
//    vector plackett_p11(vector pA, vector pB, vector psi) {
//
//          int N = num_elements(pA);
//          vector[N] a  = psi - 1.0;
//          vector[N] nb = 1.0 + (pA + pB) .* a;            //// = -b
//          vector[N] c  = psi .* pA .* pB;
//          vector[N] D  = fmax(nb .* nb - 4.0 * a .* c, 0.0);
//          return 2.0 * c ./ (nb + sqrt(D));
//
//    }
      ////
      //// ALL FOUR CELLS (class order 00, 10, 01, 11), with no floor. The cells 10, 01 and 00 are not taken as
      //// differences (pA - p11, pB - p11 and 1 - pA - pB + p11 can lose all their digits, or go below 0, at
      //// extreme margins or odds ratios): each cell is the Plackett root above for its own pair of margins and
      //// its own odds ratio (psi for the cells 11 and 00, 1 / psi for the cells 10 and 01), with pA, 1 - pA, pB
      //// and 1 - pB each from its own inv_logit. With delta = 1 - pA - pB (= p00 - p11) and epsilon = pB - pA
      //// (= p01 - p10), the discriminant of all four roots is the same sum of non-negative terms,
      ////      D = delta^2 + 2 psi (pA (1 - pA) + pB (1 - pB)) + psi^2 epsilon^2,
      //// the smaller cell of each pair (11 / 00, 10 / 01) is its root 2c / (-b + sqrt(D)), whose denominator is
      //// then a sum of positive terms, and the larger cell of each pair is the smaller one plus |delta| (or
      //// |epsilon|). Each cell is hence a ratio of positive terms, or such a ratio plus a positive term: it
      //// cannot go below 0 and it loses no digits by cancellation.
      //// (the 0 / 1 indicators of the smaller cell of each pair carry no gradient, and |delta| and |epsilon|
      ////  come from the same indicators, so that every choice is the same one; where delta = 0 or epsilon = 0
      ////  both choices give the same cells and the same gradient)
      matrix plackett_cells(vector logit_pA, vector logit_pB, vector psi) {

            int N = num_elements(logit_pA);
            vector[N] pA = inv_logit(logit_pA);
            vector[N] qA = inv_logit(-logit_pA);                 //// = 1 - pA
            vector[N] pB = inv_logit(logit_pB);
            vector[N] qB = inv_logit(-logit_pB);                 //// = 1 - pB
//          vector[N] delta = qA - pB;                           //// = 1 - pA - pB = p00 - p11
//          vector[N] epsilon = pB - pA;                         //// = p01 - p10
//          vector[N] cell_11_is_smaller = ceil(fmax(0.0, delta));     //// 1 where p11 < p00, else 0
//          vector[N] cell_10_is_smaller = ceil(fmax(0.0, epsilon));   //// 1 where p10 < p01, else 0
//          vector[N] abs_delta = (2.0 * cell_11_is_smaller - 1.0) .* delta;
//          vector[N] abs_epsilon = (2.0 * cell_10_is_smaller - 1.0) .* epsilon;
//          vector[N] sqrt_D = sqrt(square(delta) + 2.0 * psi .* (pA .* qA + pB .* qB) + square(psi .* epsilon));
            //// (|delta| and |epsilon| from the logits, not as differences of probabilities, which lose all their
            ////  digits when both probabilities are near 1 or both near 0: with s = logit_pA + logit_pB and
            ////  d = logit_pB - logit_pA, delta = pA pB expm1(-s) = -(1 - pA)(1 - pB) expm1(s) and
            ////  epsilon = -(1 - pA) pB expm1(-d) = pA (1 - pB) expm1(d), each in the form whose expm1 has a
            ////  non-positive argument, so that it cannot overflow)
            vector[N] sum_of_logits = logit_pA + logit_pB;
            vector[N] difference_of_logits = logit_pB - logit_pA;
            //// (1 where p11 < p00, i.e. where logit_pA + logit_pB < 0, and 1 where p10 < p01, i.e. where
            ////  logit_pB > logit_pA; else 0)
            vector[N] cell_11_is_smaller = fmin(1.0, ceil(fmax(0.0, -sum_of_logits)));
            vector[N] cell_10_is_smaller = fmin(1.0, ceil(fmax(0.0, difference_of_logits)));
            vector[N] abs_delta = (cell_11_is_smaller .* qA .* qB + (1.0 - cell_11_is_smaller) .* pA .* pB)
                                  .* (-expm1((2.0 * cell_11_is_smaller - 1.0) .* sum_of_logits));
            vector[N] abs_epsilon = (cell_10_is_smaller .* qA .* pB + (1.0 - cell_10_is_smaller) .* pA .* qB)
                                    .* (-expm1((1.0 - 2.0 * cell_10_is_smaller) .* difference_of_logits));
            vector[N] sqrt_D = sqrt(square(abs_delta) + 2.0 * psi .* (pA .* qA + pB .* qB)
                                    + square(psi .* abs_epsilon));
            ////
            //// (the smaller diagonal cell: p11 = R(pA, pB, psi) or p00 = R(1 - pA, 1 - pB, psi))
            vector[N] margin_product_of_smaller_diagonal_cell = cell_11_is_smaller .* pA .* pB
                                                               + (1.0 - cell_11_is_smaller) .* qA .* qB;
            vector[N] margin_sum_of_smaller_diagonal_cell = cell_11_is_smaller .* (pA + pB)
                                                           + (1.0 - cell_11_is_smaller) .* (qA + qB);
            vector[N] smaller_diagonal_cell = 2.0 * psi .* margin_product_of_smaller_diagonal_cell
                                              ./ (abs_delta + psi .* margin_sum_of_smaller_diagonal_cell
                                                  + sqrt_D);
            ////
            //// (the smaller off-diagonal cell: p10 = R(pA, 1 - pB, 1 / psi) or p01 = R(1 - pA, pB, 1 / psi),
            ////  with its numerator and its denominator multiplied by psi)
            vector[N] margin_product_of_smaller_off_diagonal_cell = cell_10_is_smaller .* pA .* qB
                                                                   + (1.0 - cell_10_is_smaller) .* qA .* pB;
            vector[N] margin_sum_of_smaller_off_diagonal_cell = cell_10_is_smaller .* (pA + qB)
                                                               + (1.0 - cell_10_is_smaller) .* (qA + pB);
            vector[N] smaller_off_diagonal_cell = 2.0 * margin_product_of_smaller_off_diagonal_cell
                                                  ./ (psi .* abs_epsilon + margin_sum_of_smaller_off_diagonal_cell
                                                      + sqrt_D);
            ////
            matrix[N, 4] cells;
            cells[, 1] = smaller_diagonal_cell + cell_11_is_smaller .* abs_delta;                    //// p00
            cells[, 2] = smaller_off_diagonal_cell + (1.0 - cell_10_is_smaller) .* abs_epsilon;      //// p10
            cells[, 3] = smaller_off_diagonal_cell + cell_10_is_smaller .* abs_epsilon;              //// p01
            cells[, 4] = smaller_diagonal_cell + (1.0 - cell_11_is_smaller) .* abs_delta;            //// p11
            return cells;

      }


      //////////////////////////////////////////////////////////////////////////
      // ---- PER-SUBJECT class probabilities from the stratum hierarchy plus the PREVALENCE
      //// REGRESSION. Each subject's two logit-prevalences are its stratum's random intercept plus
      //// a shared covariate contribution, so a sex (or age, IQ, ...) effect on latent prevalence is
      //// ONE coefficient per condition, partially pooled across strata by the hierarchy rather than
      //// re-estimated in every stratum-by-covariate cell. The class probabilities are then built
      //// exactly as the per-stratum version did - background-B parameterisation under
      //// use_cond_prior = 1, Plackett under 0 - so with no covariates this reproduces the old
      //// prev[pop[n]] entry for entry. Returns matrix[N, n_class]; class order 00, 10, 01, 11.
      //////////////////////////////////////////////////////////////////////////
      matrix log_prevalence_by_subject(data matrix X_prevalence, data array[] int pop, data int n_class,
                                   real mu_A, real sigma_A, array[] real z_A, vector beta_prevalence_A,
                                   real mu_B, real sigma_B, array[] real z_B, vector beta_prevalence_B,
                                   array[] real log_OR_par, data int log_OR_shared, data int use_cond_prior) {

            int N = rows(X_prevalence);
            int n_prevalence_covariates = cols(X_prevalence);
            matrix[N, n_class] log_prev_subject;
            //// Reuse population intercepts within this call (one chunk in reduce_sum).
            //// Covariate effects remain SUBJECT-SPECIFIC; the two matrix-vector products
            //// replace repeated row products, without sharing a subject's probabilities.
            vector[size(z_A)] intercept_A = mu_A + sigma_A * to_vector(z_A);
            vector[size(z_B)] intercept_B = mu_B + sigma_B * to_vector(z_B);
            vector[N] linear_predictor_A = intercept_A[pop];
            vector[N] linear_predictor_B = intercept_B[pop];
            if (n_prevalence_covariates > 0) {
                  linear_predictor_A += X_prevalence * beta_prevalence_A;
                  linear_predictor_B += X_prevalence * beta_prevalence_B;
            }
            if (use_cond_prior == 1) {
                  vector[N] eta_q = linear_predictor_B;
                  //// Each A log-probability is used by two classes; build it only once.
                  vector[N] log_A = log_inv_logit(linear_predictor_A);
//                vector[N] log1m_A = log1m_inv_logit(linear_predictor_A);
                  //// (log1m_inv_logit(x) as log_inv_logit(-x), the same formula in Stan math, so that the
                  ////  AVX writer swaps it)
                  vector[N] log1m_A = log_inv_logit(-linear_predictor_A);
                  if (log_OR_shared == 1) eta_q += log_OR_par[1];
                  else eta_q += to_vector(log_OR_par[pop]);
//                log_prev_subject[, 1] = log1m_A + log1m_inv_logit(linear_predictor_B);
//                log_prev_subject[, 2] = log_A + log1m_inv_logit(eta_q);
                  log_prev_subject[, 1] = log1m_A + log_inv_logit(-linear_predictor_B);
                  log_prev_subject[, 2] = log_A + log_inv_logit(-eta_q);
                  log_prev_subject[, 3] = log1m_A + log_inv_logit(linear_predictor_B);
                  log_prev_subject[, 4] = log_A + log_inv_logit(eta_q);
            } else {
                  //// The odds ratio is population-level even when the margins vary by subject.
                  vector[size(log_OR_par)] odds_ratio = exp(to_vector(log_OR_par));
//                for (n in 1:N) {
//                      real pA = inv_logit(linear_predictor_A[n]);
//                      real pB = inv_logit(linear_predictor_B[n]);
//                      real p11 = plackett_p11(pA, pB, odds_ratio[log_OR_shared == 1 ? 1 : pop[n]]);
//                      vector[4] p_raw = [1-pA-pB+p11, pA-p11, pB-p11, p11]';
//                      p_raw = fmax(p_raw, 1e-12);  //// retain the original OR-route rounding convention
//                      log_prev_subject[n] = log(p_raw / sum(p_raw))';
//                }
                  //// (no loop over the subjects: the same operations on whole vectors, one column per class)
//                vector[N] pA = inv_logit(linear_predictor_A);
//                vector[N] pB = inv_logit(linear_predictor_B);
//                vector[N] p11 = plackett_p11(pA, pB, log_OR_shared == 1 ? rep_vector(odds_ratio[1], N)
//                                                                        : odds_ratio[pop]);
//                matrix[N, 4] p_raw;
//                p_raw[, 1] = 1 - pA - pB + p11;
//                p_raw[, 2] = pA - p11;
//                p_raw[, 3] = pB - p11;
//                p_raw[, 4] = p11;
//                p_raw = fmax(p_raw, 1e-12);  //// retain the original OR-route rounding convention
//                vector[N] sum_of_p_raw = p_raw[, 1] + p_raw[, 2] + p_raw[, 3] + p_raw[, 4];
//                for (c in 1:4) log_prev_subject[, c] = log(p_raw[, c] ./ sum_of_p_raw);
                  //// (every cell by its own root, so no cell is a difference and none needs a floor)
                  matrix[N, 4] p_raw = plackett_cells(linear_predictor_A, linear_predictor_B,
                                                      log_OR_shared == 1 ? rep_vector(odds_ratio[1], N)
                                                                         : odds_ratio[pop]);
                  vector[N] sum_of_p_raw = p_raw[, 1] + p_raw[, 2] + p_raw[, 3] + p_raw[, 4];
                  for (c in 1:4) log_prev_subject[, c] = log(p_raw[, c] ./ sum_of_p_raw);
            }
            return log_prev_subject;

      }



}

data {
      int<lower=1> N;
      int<lower=2> n_tests;
      ////
      //// ---- FOUR-CLASS FACTORIAL STRUCTURE:
      ////
      int<lower=4, upper=4> n_class;                      //// fixed at 4 here
      array[n_class, 2] int<lower=0, upper=1> class_D;    //// class_D[c, 1] = D_A, class_D[c, 2] = D_B
      array[n_tests] int<lower=1, upper=2> test_condition;//// which condition test t measures: 1 = A, 2 = B
      array[n_tests] int<lower=0, upper=1> allow_cross_loading;  //// 1 -> free delta_cross[1:2, t] (own-status absent / present)
      array[n_tests] int<lower=0, upper=1> is_perfect_ref;       //// 1 -> pin accuracy to (near) perfect
      //// the order in which the tests' latent variables are conditioned on each other (the GHK sequence,
      //// the LDL order of each correlation matrix and the u_raw columns): latent_test_order[j] = the test at
      //// latent position j; 1:n_tests = the data's own order (the model as before):
      array[n_tests] int<lower=1, upper=n_tests> latent_test_order;
      real<lower=0> perfect_ref_M;                        //// magnitude for the pinned means, e.g. 8.0
      ////
      //// ---- Ordinal data fields:
      ////
      int<lower=0> n_binary_tests;
      int<lower=0> n_ordinal_tests;
      array[n_ordinal_tests] int<lower=2> n_cat_per_ord_test;
      array[n_ordinal_tests] int<lower=1> n_thr_per_ord_test;
      array[2] matrix<lower=0>[n_ordinal_tests > 0 ? max(n_cat_per_ord_test) : 1, max(n_ordinal_tests, 1)] prior_dirichlet_alpha;  //// [own-status][cat, ord test]
      ////
      matrix[N, n_tests] y;      //// binary: 0/1; ordinal: 1..K; MISSING: any negative value (use -1)
      //// the highest score consistent with the answers given, coded as y; equal to y wherever the score is
      //// exactly observed (binary and missing scores always). An ordinal score with y < y_upper is
      //// interval-censored, i.e. it lies in categories y..y_upper (e.g. a questionnaire total with
      //// unanswered items: y = the sum of the answered items, y_upper = that sum + the number of unanswered
      //// items, capped at the top category). A score censored over every category is missing: code it as y < 0.
      matrix[N, n_tests] y_upper_when_interval_censored;
      ////
      int<lower=1> n_pops;
      array[N] int pop;
      ////
      //// ---- Population share of each stratum, for the OVERALL (population-level) estimands in GQ.
      //// The bridge supplies the modelling groups; prev[g] may differ
      //// between them and the overall figure is the size-weighted average, NOT prev[1]. Need not sum
      //// to 1 exactly (raw counts are fine); normalised in transformed data.
      ////
      vector<lower=0>[n_pops] pop_weight;
      ////
      //// ---- Covariates (shared across classes; see header note 4):
      ////
      int n_covariates_max;
      array[n_tests] matrix[N, n_covariates_max] X;
      array[n_tests] int n_covs_per_outcome;
      ////
      //// ---- 0 = one covariate effect per test shared across classes (the old behaviour). 1 = separate
      //// effects for own-condition absent and present, so a covariate can move SENSITIVITY without
      //// moving specificity by the same amount. Label identification is unaffected: it comes from
      //// the ordering constraint at the anchor profile, not from every covariate profile.
      int<lower=0, upper=1> covariate_effects_by_own_status;
      array[2, n_covariates_max, n_tests] int<lower=0, upper=1> covariate_active;
      ////
      //// ---- PREVALENCE REGRESSION. Subject-level covariates on the two latent logit-prevalences,
      //// on top of the stratum hierarchy. n_prevalence_covariates = 0 reproduces the per-stratum model.
      //// Requires use_OR_prior = 1 (the Dirichlet route has no linear predictor to add to).
      int<lower=0> n_prevalence_covariates;
      matrix[N, n_prevalence_covariates] X_prevalence;
      vector[n_prevalence_covariates]          prior_beta_prevalence_A_mean;
      vector<lower=0>[n_prevalence_covariates] prior_beta_prevalence_A_sd;
      vector[n_prevalence_covariates]          prior_beta_prevalence_B_mean;
      vector<lower=0>[n_prevalence_covariates] prior_beta_prevalence_B_sd;
      ////
      int corr_force_positive;
      ////
      //// ---- CORRELATION POOLING across classes (Cerullo et al 2022, section 3.3.1, with class in
      //// place of study). 0 = one free correlation matrix per class, as before. 1 = each class is a
      //// convex combination of ONE global matrix and its own departure matrix,
      ////      Omega[c] = (1 - lambda[c]) * Omega_global + lambda[c] * Omega_departure[c],
      //// which is positive definite with unit diagonal by construction, carries the entrywise bounds
      //// through, and lets the ~7,300-person class inform the three classes with a few dozen people.
      //// lambda[c] ~ beta(a, b): small a / large b favours pooling.
      int<lower=0, upper=1> pool_correlations_across_classes;
      real<lower=0> prior_LKJ_global;
      real<lower=0> prior_LKJ_departure;
      real<lower=0> prior_lambda_shape_a;
      real<lower=0> prior_lambda_shape_b;
      ////
      array[n_class] matrix[n_tests, n_tests] known_values_list;
      array[n_class] matrix[n_tests, n_tests] known_values_indicator_list;
      array[n_class] matrix[n_tests, n_tests] lb_corr;
      array[n_class] matrix[n_tests, n_tests] ub_corr;
      ////
      real overflow_threshold;
      real underflow_threshold;
      ////
      real C_raw_lower;
      real C_raw_upper;
      ////
      int prior_only;
      ////
      //// ---- Priors:
      ////
      matrix[2, n_tests] prior_beta_mean;                 //// [own-status absent / present, test] - as prior_beta_mean[c, t] in the 2-class file
      matrix<lower=0>[2, n_tests] prior_beta_sd;
      vector<lower=0>[n_tests] prior_delta_cross_sd;      //// centred at 0
      ////
      //// ---- Covariate priors, [own-status absent / present][covariate, test]. With
      //// covariate_effects_by_own_status = 0 only slice 1 is read. A directional belief (e.g. lower
      //// ADOS sensitivity in women) goes in the PRESENT-slice mean; a zero-centred prior does not
      //// encode a direction, it only regularises the size.
      array[2] matrix[n_covariates_max, n_tests] prior_beta_cov_mean;
      array[2] matrix<lower=0>[n_covariates_max, n_tests] prior_beta_cov_sd;
      matrix<lower=0>[n_class, 1] prior_LKJ;
      vector<lower=0>[n_class] prior_prev_dirichlet;      //// Dirichlet over the 4 classes
      ////
      int<lower=1, upper=2> Phi_type;      //// 1 = Phi (the exact normal CDF), 2 = Phi_approx
      ////
      //// ---- Baseline case for GQ:
      ////
      array[n_tests] vector[n_covariates_max] baseline_case;
      ////
      //// ---- STANDARDISATION SET. Every population-level quantity in GQ (prev_*_overall, Se_ord,
      //// Sp_ord, the PPV grid, the diagnosed-co-occurrence block) is a weighted average over these
      //// subjects, each at its OWN covariate profile and its OWN class probabilities. The weight is
      //// the phase-one survey weight, so the target stays "phase_one_weighted". Passing every row
      //// (index = 1:N, weight = wt_ints1) is exact; a weighted subsample trades exactness for GQ
      //// time on the bivariate-normal grid. Under prior_only the bridge passes one row per stratum
      //// at X = 0 with weight = pop_weight, which reproduces the old per-stratum standardisation.
      int<lower=1> n_standardisation;
      array[n_standardisation] int<lower=1, upper=N> standardisation_index;
      vector<lower=0>[n_standardisation] standardisation_weight;
      ////
      //// ---- The PPV GRID's own standardisation set, given as POSITIONS within the set above. The
      //// grid calls biv_norm_upper once per profile x threshold pair x class, so on the full APMS
      //// covariate set (about 7,400 distinct profiles) it is roughly 94% of all generated-quantities
      //// time. Every quantity the paper reports stays on the full set; only this diagnostic display
      //// uses the subsample. Pass 1:n_standardisation to make it exact too.
      int<lower=1> n_grid_standardisation;
      array[n_grid_standardisation] int<lower=1, upper=n_standardisation> grid_standardisation_position;
      vector<lower=0>[n_grid_standardisation] grid_standardisation_weight;
      ////
      //// ---- reduce_sum_static chunk_size: maximum subjects per slice; R defaults to ceil(N / num_chunks).
      ////
      int<lower=1> chunk_size;
      ////
      //// ---- LOO: number of fresh GHK draws per observation for the MARGINAL log_lik in GQ.
      //// 0 skips it (log_lik = 0). Check Monte Carlo stability before using PSIS-LOO; leave it
      //// at 0 for the simulation runs (it is the dominant GQ cost).
      ////
      int<lower=0> n_loo_draws;
      ////
      //// ---- DIAGNOSTIC quantities in GQ (Section: posterior class probabilities and predictive
      //// values). n_grid_A / n_grid_B = 0 switches the threshold grid off.
      ////
      //// grid_thr_A / grid_thr_B are THRESHOLDS on the two screens in the model's own indexing:
      //// "screen positive" means category > k, i.e. raw score >= k. Keep the grids small - the
      //// cost is n_grid_A * n_grid_B * n_class * n_quad_nodes per draw.
      ////
      int<lower=0> n_grid_A;
      int<lower=0> n_grid_B;
      array[n_grid_A] int grid_thr_A;
      array[n_grid_B] int grid_thr_B;
      ////
      //// ---- fixed clinical thresholds for the REFERENCE tests, as CATEGORY indices (score + 1).
      //// Needed for the diagnosed-co-occurrence quantities below: unlike the screen grid, these are
      //// two fixed numbers, always ADOS >= 10 (category 11) and BPD criteria >= 5 (category 6), so
      //// they are scalars rather than a grid. Used only when tests 1,2 are ORDINAL (ignored, but
      //// still required as data, when they are binary - see the branch below).
      int<lower=2> ref_cutoff_category_A;
      int<lower=2> ref_cutoff_category_B;
      int<lower=1> n_quad_nodes;                      //// check quadrature stability for the fitted correlations
      ////
      //// per-subject posterior class probabilities. N x n_class per draw, so OFF by default -
      //// turn on for one fit when you want the individual-level output.
      ////
      int<lower=0, upper=1> save_subject_probs;
      ////
      //// ---- CLASS-PREVALENCE PRIOR. 0 = Dirichlet; 1 = hierarchical logit-normal margins
      //// plus a normal prior on log_OR. These encode different beliefs, so compare their actual
      //// prior distributions. log_OR mean 0 centres the association at independence on the log scale;
      //// it is a regularising prior, not an absence of prior information.
      //// The marginal hierarchy shares information across strata, including strata with no reference
      //// observations. In those strata the result depends on the measurement and pooling assumptions.
      ////
      int<lower=0, upper=1> use_OR_prior;
      ////
      //// log_OR_shared = 1: one within-stratum association. 0: independent stratum associations.
      //// All observed response patterns inform association under the model, including zero joint-
      //// positive counts. Neither a zero cell nor a small cell proves that a posterior is pure prior.
      //// A hierarchical alternative is a separate modelling choice; sparse data need not force its
      //// heterogeneity to zero.
      ////
      int<lower=0, upper=1> log_OR_shared;
      ////
      real          prior_mu_logit_A_mean;         //// e.g. logit(0.010) = -4.60
      real<lower=0> prior_mu_logit_A_sd;
      real          prior_mu_logit_B_mean;         //// e.g. logit(0.005) = -5.29
      real<lower=0> prior_mu_logit_B_sd;
      real<lower=0> prior_sigma_logit_sd;          //// half-normal sd on the between-stratum spread
      real          prior_log_OR_mean;
      real<lower=0> prior_log_OR_sd;
      ////
      //// ---- use_cond_prior = 1 replaces the normal prior on log_OR with a prior placed DIRECTLY on
      //// the population conditional probability Pr(B | A) = pi_11_overall / prev_A_overall, i.e. on
      //// the quantity already reported as Pr_B_given_A_overall. It requires use_OR_prior = 1 and
      //// log_OR_shared = 1. Under the odds-ratio parameterisation a belief about Pr(B | A) cannot be
      //// expressed by any choice of prior_log_OR_mean / prior_log_OR_sd, because the map from log_OR
      //// to Pr(B | A) depends on the prevalences, and a share of prior draws has a Frechet ceiling
      //// min(prev_A, prev_B) / prev_A below the intended lower endpoint - those draws cannot reach it
      //// at ANY odds ratio.
      ////
      //// In this mode the SAME declared parameters are reused with a different meaning, so that no
      //// parameter is left out of the target density:
      ////      mu_logit_B / sigma_logit_B / z_logit_B  ->  BACKGROUND Pr(B | not A, stratum), not Pr(B)
      ////      log_OR_par[1]                           ->  the shared conditional log odds ratio
      //// The marginal prev_B is then DERIVED as Pr(B) = Pr(B|A) Pr(A) + Pr(B|not A) Pr(not A),
      //// so prior_mu_logit_B_mean must be re-centred: the same centre applied to the background
      //// implies a LARGER marginal. Inspect the induced prior before fitting.
      ////
      int<lower=0, upper=1> use_cond_prior;
      real          prior_logit_Q_mean;            //// centre of the Pr(B | A) belief, on the logit scale
      real<lower=0> prior_logit_Q_sd;
}

transformed data {
      //// Independent tests need no GHK augmentation. With every off-diagonal fixed to zero,
      //// the interval probabilities do not depend on u; integrating those uniforms gives one.
      //// This removes N * n_tests redundant parameters in M3 without changing its model.
      int all_correlations_zero = 1;
      for (c in 1:n_class) for (i in 2:n_tests) for (j in 1:(i - 1)) {
          if (known_values_indicator_list[c][i,j] == 0 || known_values_list[c][i,j] != 0)
              all_correlations_zero = 0;
      }
      int n_u_rows = (prior_only == 1 || all_correlations_zero == 1) ? 0 : N;

      //// Only the CI likelihood path reads this, and prior_only bypasses every likelihood path,
      //// so build it only when it will actually be used.
      int n_u_CI_rows = (prior_only == 0 && all_correlations_zero == 1) ? N : 0;
      array[n_u_CI_rows] row_vector[n_tests] u_raw_CI =
          rep_array(rep_row_vector(0.0, n_tests), n_u_CI_rows);
      //// ---- Input checks shared by the serial and reduce_sum_static versions:
      if (save_subject_probs == 1 && n_loo_draws == 0)
          reject("save_subject_probs requires n_loo_draws > 0");
      //// (n_loo_draws > 0: the LOO log-likelihood of the generated quantities, which integrated the
      ////  missing tests out over the observed tests only, not by the GHK method, is commented out)
      if (n_loo_draws > 0)
          reject("n_loo_draws > 0 is not available: the LOO log-likelihood (log_lik_obs_ghk_rng) is ",
                 "commented out, since it integrated the missing tests out over the observed tests only, ",
                 "not by the GHK method; set n_loo_draws = 0");
      if (n_tests != 4 || n_binary_tests + n_ordinal_tests != n_tests)
          reject("this model requires four tests, binary tests first");
      if (n_binary_tests != 0 && n_binary_tests != 2)
          reject("supported reference views: zero or two binary references");
      if (sum(pop_weight) <= 0) reject("pop_weight must have a positive total");
      if (use_cond_prior == 1 && (use_OR_prior != 1 || log_OR_shared != 1))
          reject("use_cond_prior = 1 requires use_OR_prior = 1 and log_OR_shared = 1");
      if (n_prevalence_covariates > 0 && use_OR_prior != 1)
          reject("the prevalence regression requires use_OR_prior = 1");
      if (n_prevalence_covariates > 0 && use_OR_prior != 1)
          reject("prevalence covariates require use_OR_prior = 1");
      if (covariate_effects_by_own_status == 0)
          for (t in 1:n_tests) for (k in 1:n_covariates_max)
              if (covariate_active[1,k,t] != covariate_active[2,k,t])
                  reject("shared covariates require the same active mask in both statuses");
      if (sum(standardisation_weight) <= 0) reject("standardisation_weight must have a positive total");
      vector[n_standardisation] standardisation_w = standardisation_weight / sum(standardisation_weight);
      //// (the standardisation profiles' prevalence covariates and populations, for the tape-chunked
      ////  use_cond_prior = 1 sums in transformed parameters)
      matrix[n_standardisation, n_prevalence_covariates] X_prevalence_standardisation
            = X_prevalence[standardisation_index];
      array[n_standardisation] int pop_standardisation = pop[standardisation_index];
      //// (the standardisation profiles grouped by population, population 1 first, and the number of them in each
      ////  population, for the per-stratum class probabilities of generated quantities, without a loop over the
      ////  profiles)
      array[n_standardisation] int standardisation_profiles_by_pop;
      array[n_pops] int n_standardisation_profiles_in_pop;
      {
          vector[n_standardisation] pop_of_profile = to_vector(pop_standardisation);
          standardisation_profiles_by_pop = sort_indices_asc(pop_of_profile);
          for (g in 1:n_pops) {
              n_standardisation_profiles_in_pop[g] = number_of_elements_below(pop_of_profile, g + 0.5)
                                                     - number_of_elements_below(pop_of_profile, g - 0.5);
          }
      }
      if (sum(grid_standardisation_weight) <= 0) reject("grid_standardisation_weight must have a positive total");
      vector[n_grid_standardisation] grid_standardisation_w =
            grid_standardisation_weight / sum(grid_standardisation_weight);
      for (c in 1:4) {
          if (class_D[c, 1] != ((c == 2 || c == 4) ? 1 : 0) ||
              class_D[c, 2] != ((c == 3 || c == 4) ? 1 : 0))
              reject("class_D order must be 00, 10, 01, 11");
      }
      for (t in 1:n_tests) {
          if (n_covs_per_outcome[t] < 0 || n_covs_per_outcome[t] > n_covariates_max)
              reject("invalid n_covs_per_outcome");
          if (is_perfect_ref[t] == 1 && t > n_binary_tests)
              reject("perfect references must be binary");
      }
      for (t in 1:n_ordinal_tests) {
          if (n_thr_per_ord_test[t] != n_cat_per_ord_test[t] - 1)
              reject("ordinal category / threshold counts disagree");
          for (st in 1:2) for (k in 1:n_cat_per_ord_test[t])
              if (prior_dirichlet_alpha[st][k, t] <= 0)
                  reject("active Dirichlet concentrations must be strictly positive");
      }
      ////
      //// ---- Each test's subjects sorted by their score y, so that its n_missing_of_test[t] missing subjects
      //// (y < 0) come first and its observed ones last: the checks below, the observation mask, y_use and
      //// y_ord_upper, and the screen counts of generated quantities use these instead of a loop over the
      //// subjects.
      ////
      array[n_tests] int n_missing_of_test;
      array[n_tests, N] int subjects_in_order_of_y;
      for (t in 1:n_tests) {
          vector[N] y_t = y[, t];
          n_missing_of_test[t] = number_of_elements_below(y_t, 0.0);
          subjects_in_order_of_y[t] = sort_indices_asc(y_t);
      }
//    for (n in 1:N) {
//        if (pop[n] < 1 || pop[n] > n_pops) reject("pop is outside 1:n_pops");
//        for (t in 1:n_tests) if (y[n,t] >= 0) {
//            if (y[n,t] != floor(y[n,t])) reject("observed scores must be integers");
//            if (t <= n_binary_tests) {
//                if (y[n,t] > 1) reject("binary scores must be 0 or 1");
//            } else {
//                if (y[n,t] < 1 || y[n,t] > n_cat_per_ord_test[t - n_binary_tests])
//                    reject("ordinal score outside its category range");
//            }
//        }
//    }
      //// (no loop over the subjects: the same checks on the observed scores of each test as one vector)
      if (min(pop) < 1 || max(pop) > n_pops) reject("pop is outside 1:n_pops");
      for (t in 1:n_tests) {
          int n_observed_t = N - n_missing_of_test[t];
          if (n_observed_t > 0) {
              array[n_observed_t] int observed_t = subjects_in_order_of_y[t, (n_missing_of_test[t] + 1):N];
              vector[n_observed_t] y_observed_t = y[observed_t, t];
              if (max(abs(y_observed_t - floor(y_observed_t))) > 0) reject("observed scores must be integers");
              if (t <= n_binary_tests) {
                  if (max(y_observed_t) > 1) reject("binary scores must be 0 or 1");
              } else {
                  if (min(y_observed_t) < 1 || max(y_observed_t) > n_cat_per_ord_test[t - n_binary_tests])
                      reject("ordinal score outside its category range");
              }
          }
      }
      ////
      //// ---- Interval-censored ordinal scores (y_upper_when_interval_censored):
      ////
      int any_interval_censored_score = 0;
//    for (n in 1:N) {
//        for (t in 1:n_tests) {
//            real y_up = y_upper_when_interval_censored[n, t];
//            if (y[n, t] < 0 || t <= n_binary_tests) {
//                if (y_up != y[n, t])
//                    reject("y_upper_when_interval_censored must equal y for missing and binary scores");
//            } else {
//                if (y_up != floor(y_up) || y_up < y[n, t] || y_up > n_cat_per_ord_test[t - n_binary_tests])
//                    reject("y_upper_when_interval_censored must be an integer from y to the top category");
//                if (y[n, t] == 1 && y_up == n_cat_per_ord_test[t - n_binary_tests] && y_up > 1)
//                    reject("a score censored over every category is missing: code it as y < 0");
//                if (y_up > y[n, t]) any_interval_censored_score = 1;
//            }
//        }
//    }
      //// (no loop over the subjects: the same checks on the missing and the observed scores of each test as
      ////  vectors. With the observed ordinal scores whole numbers from 1 to n_cat (checked above) and y_up
      ////  from y to n_cat, y == 1 and y_up == n_cat, with n_cat >= 2, is y_up - y == n_cat - 1, its largest
      ////  value)
      for (t in 1:n_tests) {
          int n_missing_t = n_missing_of_test[t];
          int n_observed_t = N - n_missing_t;
          if (t <= n_binary_tests) {
              if (max(abs(y_upper_when_interval_censored[, t] - y[, t])) > 0)
                  reject("y_upper_when_interval_censored must equal y for missing and binary scores");
          } else {
              int n_cat_t = n_cat_per_ord_test[t - n_binary_tests];
              if (n_missing_t > 0) {
                  array[n_missing_t] int missing_t = subjects_in_order_of_y[t, 1:n_missing_t];
                  if (max(abs(y_upper_when_interval_censored[missing_t, t] - y[missing_t, t])) > 0)
                      reject("y_upper_when_interval_censored must equal y for missing and binary scores");
              }
              if (n_observed_t > 0) {
                  array[n_observed_t] int observed_t = subjects_in_order_of_y[t, (n_missing_t + 1):N];
                  vector[n_observed_t] y_up_o = y_upper_when_interval_censored[observed_t, t];
                  vector[n_observed_t] y_o = y[observed_t, t];
                  if (max(abs(y_up_o - floor(y_up_o))) > 0 || min(y_up_o - y_o) < 0 || max(y_up_o) > n_cat_t)
                      reject("y_upper_when_interval_censored must be an integer from y to the top category");
                  if (max(y_up_o - y_o) == n_cat_t - 1)
                      reject("a score censored over every category is missing: code it as y < 0");
                  if (max(y_up_o - y_o) > 0) any_interval_censored_score = 1;
              }
          }
      }
      if (any_interval_censored_score == 1 && prior_only == 0
          && all_correlations_zero == 1 && max(n_covs_per_outcome) == 0)
          reject("interval-censored scores are not implemented in the conditional-independence shortcut");
      int n_thr_max_gq = (n_ordinal_tests > 0) ? max(n_thr_per_ord_test) : 1;
      int n_total_C = (n_ordinal_tests > 0) ? sum(n_thr_per_ord_test) : 0;
      ////
      //// ---- Which tests carry their own-condition location in beta: binary tests only. For ordinal
      //// tests the own-status effect is in the cutpoints (see header note 2, v6).
      ////
      array[n_tests] int mean_has_own;
      for (t in 1:n_tests) mean_has_own[t] = (t <= n_binary_tests) ? 1 : 0;
      array[max(n_ordinal_tests, 1)] int ord_start_index;
      array[max(n_ordinal_tests, 1)] int ord_end_index;
      if (n_ordinal_tests > 0) {
          ord_start_index = calculate_start_indices(n_thr_per_ord_test, n_ordinal_tests);
          ord_end_index   = calculate_end_indices(n_thr_per_ord_test, n_ordinal_tests, ord_start_index);
      }
      ////
      ////
      //// ---- Observation mask, DERIVED from the sentinel, and a filled copy of y for the
      //// vectorised arithmetic (0 for a masked binary entry, 1 for a masked ordinal one - the
      //// value never reaches the likelihood, but the partition code must not see a negative):
      ////
      array[N, n_tests] int<lower=0, upper=1> obs_mask;
      matrix[N, n_tests] y_use = y;
//    for (n in 1:N) {
//        for (t in 1:n_tests) {
//            obs_mask[n, t] = (y[n, t] >= 0) ? 1 : 0;
//            if (obs_mask[n, t] == 0) y_use[n, t] = (t <= n_binary_tests) ? 0.0 : 1.0;
//        }
//    }
      //// (no loop over the subjects: obs_mask is 1, and then 0 for the missing subjects of each test, the first
      ////  n_missing_of_test[t] of subjects_in_order_of_y[t], whose y_use is 0 (binary) or 1 (ordinal))
      for (t in 1:n_tests) {
          int n_missing_t = n_missing_of_test[t];
          obs_mask[, t] = rep_array(1, N);
          if (n_missing_t > 0) {
              array[n_missing_t] int missing_t = subjects_in_order_of_y[t, 1:n_missing_t];
              obs_mask[missing_t, t] = rep_array(0, n_missing_t);
              y_use[missing_t, t] = rep_vector((t <= n_binary_tests) ? 0.0 : 1.0, n_missing_t);
          }
      }
      ////
      ////
      array[max(n_ordinal_tests, 1), N] int y_ord;
//    for (n in 1:N) y_ord[1, n] = 1;
      y_ord[1] = rep_array(1, N);
      if (n_ordinal_tests > 0) {
          for (t_ord in 1:n_ordinal_tests) {
//            for (n in 1:N) {
//                y_ord[t_ord, n] = to_int(y_use[n, n_binary_tests + t_ord]);
//            }
              y_ord[t_ord] = to_int(to_array_1d(y_use[, n_binary_tests + t_ord]));
          }
      }
      //// the upper category of each ordinal score (= y_ord unless the score is interval-censored):
      array[max(n_ordinal_tests, 1), N] int y_ord_upper = y_ord;
      if (n_ordinal_tests > 0) {
          for (t_ord in 1:n_ordinal_tests) {
//            for (n in 1:N) {
//                if (obs_mask[n, n_binary_tests + t_ord] == 1)
//                    y_ord_upper[t_ord, n] = to_int(y_upper_when_interval_censored[n, n_binary_tests + t_ord]);
//            }
              //// (no loop over the subjects: the observed subjects of the test, the last ones of
              ////  subjects_in_order_of_y, take their upper category from y_upper_when_interval_censored)
              int t = n_binary_tests + t_ord;
              int n_observed_t = N - n_missing_of_test[t];
              if (n_observed_t > 0) {
                  array[n_observed_t] int observed_t = subjects_in_order_of_y[t, (n_missing_of_test[t] + 1):N];
                  y_ord_upper[t_ord, observed_t]
                        = to_int(to_array_1d(y_upper_when_interval_censored[observed_t, t]));
              }
          }
      }
      ////
      real C_sentinel = 1000.0;
      ////
      array[n_class] matrix[n_tests, n_tests] lb_corr_actual = lb_corr;
      if (corr_force_positive == 1) {
         for (c in 1:n_class) {
           for (i in 2:n_tests) {
             for (j in 1:(i - 1)) {
               lb_corr_actual[c][i, j] = 0.0;
             }
           }
         }
      }
      ////
      //// ---- The latent order (latent_test_order): each test's latent position, and the correlation
      //// bounds and known values in the latent order for the LDL of each correlation matrix (its lower
      //// triangle, latent row j1 > latent column j2, is the data's lower-triangle entry of that pair):
      ////
      array[n_tests] int latent_position_of_test = rep_array(0, n_tests);
      for (j in 1:n_tests) {
            if (latent_position_of_test[latent_test_order[j]] != 0)
                  reject("latent_test_order must be a permutation of 1:n_tests");
            latent_position_of_test[latent_test_order[j]] = j;
      }
      array[n_class] matrix[n_tests, n_tests] lb_corr_latent_order = lb_corr_actual;
      array[n_class] matrix[n_tests, n_tests] ub_corr_latent_order = ub_corr;
      array[n_class] matrix[n_tests, n_tests] known_values_indicator_latent_order = known_values_indicator_list;
      array[n_class] matrix[n_tests, n_tests] known_values_latent_order = known_values_list;
      for (c in 1:n_class) {
            for (j1 in 2:n_tests) {
                  for (j2 in 1:(j1 - 1)) {
                        int a = max(latent_test_order[j1], latent_test_order[j2]);
                        int b = min(latent_test_order[j1], latent_test_order[j2]);
                        lb_corr_latent_order[c][j1, j2] = lb_corr_actual[c][a, b];
                        ub_corr_latent_order[c][j1, j2] = ub_corr[c][a, b];
                        known_values_indicator_latent_order[c][j1, j2] = known_values_indicator_list[c][a, b];
                        known_values_latent_order[c][j1, j2] = known_values_list[c][a, b];
                  }
            }
      }
      ////
      ////
      //// ---- Normalised population shares:
      ////
      vector[n_pops] pop_w = pop_weight / sum(pop_weight);
      ////
      //// ---- validate the DIAGNOSTIC grid. The grid entries are THRESHOLDS on the two screens, and
      //// the screens' ordinal index is (global index) - n_binary_tests: 1 and 2 in the binary-
      //// reference view, 3 and 4 in the all-ordinal view. Getting that mapping backwards indexes the
      //// clinician references instead of the screens, which is silent in the all-ordinal view because
      //// 1 and 2 are valid indices - so it is checked here, at data-load, rather than discovered in
      //// the middle of warmup or not at all.
      ////
      if (n_grid_A > 0 || n_grid_B > 0) {
          int tA_chk = 3 - n_binary_tests;
          int tB_chk = 4 - n_binary_tests;
          if (tA_chk < 1 || tB_chk > n_ordinal_tests) {
              reject("diagnostic grid: the screens (global tests 3 and 4) are not both ordinal. ",
                     "n_binary_tests = ", n_binary_tests, ", n_ordinal_tests = ", n_ordinal_tests);
          }
          for (i in 1:n_grid_A) {
              if (grid_thr_A[i] < 1 || grid_thr_A[i] > n_thr_per_ord_test[tA_chk]) {
                  reject("grid_thr_A[", i, "] = ", grid_thr_A[i], " is not a valid threshold for the ",
                         "autism screen, which has ", n_thr_per_ord_test[tA_chk], " thresholds.");
              }
          }
          for (j in 1:n_grid_B) {
              if (grid_thr_B[j] < 1 || grid_thr_B[j] > n_thr_per_ord_test[tB_chk]) {
                  reject("grid_thr_B[", j, "] = ", grid_thr_B[j], " is not a valid threshold for the ",
                         "BPD screen, which has ", n_thr_per_ord_test[tB_chk], " thresholds.");
              }
          }
      }
      int k_choose_2 = (n_tests * (n_tests - 1)) %/% 2;
      array[n_class] int n_corrs_per_class = rep_array(0, n_class);
      array[n_class] int corr_start_index;
      int n_corrs_total = 0;
      for (c in 1:n_class) {
          corr_start_index[c] = n_corrs_total + 1;
          for (i in 2:n_tests) for (j in 1:(i - 1)) {
              if (known_values_indicator_list[c][i, j] == 0) n_corrs_per_class[c] += 1;
          }
          n_corrs_total += n_corrs_per_class[c];
      }
      int n_corrs_global = pool_correlations_across_classes == 1 ? n_corrs_per_class[1] : 0;
      if (pool_correlations_across_classes == 1) {
          for (c in 2:n_class) {
              if (n_corrs_per_class[c] != n_corrs_per_class[1])
                  reject("correlation pooling requires the same free-entry pattern in every class");
              for (i in 2:n_tests) for (j in 1:(i - 1)) {
                  if (lb_corr_actual[c][i,j] != lb_corr_actual[1][i,j] || ub_corr[c][i,j] != ub_corr[1][i,j])
                      reject("correlation pooling requires identical bounds in every class");
                  if (known_values_indicator_list[c][i, j] != known_values_indicator_list[1][i, j] ||
                      known_values_list[c][i, j] != known_values_list[1][i, j])
                      reject("correlation pooling requires identical fixed entries in every class");
              }
          }
      }
      ////
      //// ---- Index of the "both conditions present" class, for reporting pi_11:
      ////
      int class_11 = 1;
      for (c in 1:n_class) {
          if ((class_D[c, 1] == 1) && (class_D[c, 2] == 1)) class_11 = c;
      }
      int class_00 = 1;
      for (c in 1:n_class) {
          if ((class_D[c, 1] == 0) && (class_D[c, 2] == 0)) class_00 = c;
      }
      int class_10 = 1;
      for (c in 1:n_class) {
          if ((class_D[c, 1] == 1) && (class_D[c, 2] == 0)) class_10 = c;
      }
      int class_01 = 1;
      for (c in 1:n_class) {
          if ((class_D[c, 1] == 0) && (class_D[c, 2] == 1)) class_01 = c;
      }
      ////
      //// ---- How many cross-loadings are actually free:
      ////
      int n_cross_free = sum(allow_cross_loading);

      ////
      //// ---- Padded ordinal metadata (zero-length arrays are not valid reduce_sum_static shared args):
      ////
      array[max(n_ordinal_tests, 1)] int n_thr_per_ord_test_pad = rep_array(1, max(n_ordinal_tests, 1));
      array[max(n_ordinal_tests, 1)] int n_cat_per_ord_test_pad = rep_array(2, max(n_ordinal_tests, 1));
      if (n_ordinal_tests > 0) {
          for (t_ord in 1:n_ordinal_tests) {
              n_thr_per_ord_test_pad[t_ord] = n_thr_per_ord_test[t_ord];
              n_cat_per_ord_test_pad[t_ord] = n_cat_per_ord_test[t_ord];
          }
      }
      ////
      //// ---- For the screen counts of generated quantities: each test's observed subjects grouped by population
      //// (population 1 first: the first N - n_missing_of_test[t] entries of observed_subjects_of_test_by_pop[t])
      //// and the number of them in each population, without a loop over the subjects:
      ////
      array[n_tests, N] int observed_subjects_of_test_by_pop = rep_array(1, n_tests, N);
      array[n_tests, n_pops] int n_observed_of_test_in_pop = rep_array(0, n_tests, n_pops);
      for (t in 1:n_tests) {
          int n_observed_t = N - n_missing_of_test[t];
          if (n_observed_t > 0) {
              array[n_observed_t] int observed_t = subjects_in_order_of_y[t, (n_missing_of_test[t] + 1):N];
              vector[n_observed_t] pop_of_observed_t = to_vector(pop[observed_t]);
              array[n_observed_t] int positions_in_order_of_pop = sort_indices_asc(pop_of_observed_t);
              observed_subjects_of_test_by_pop[t, 1:n_observed_t] = observed_t[positions_in_order_of_pop];
              for (g in 1:n_pops) {
                  n_observed_of_test_in_pop[t, g] = number_of_elements_below(pop_of_observed_t, g + 0.5)
                                                    - number_of_elements_below(pop_of_observed_t, g - 0.5);
              }
          }
      }
      ////
      //// ---- For the marginal log-likelihood of generated quantities (n_loo_draws > 0): the subjects grouped by
      //// the pattern of their observed tests, coded as the sum of 2^(j - 1) over the latent positions j of the
      //// observed tests (0 = no test observed, 2^n_tests - 1 = every test observed), and the number of subjects
      //// with each pattern (at pattern code + 1), without a loop over the subjects:
      ////
      int n_observed_test_patterns = 1;
      for (j in 1:n_tests) n_observed_test_patterns *= 2;
      array[n_loo_draws > 0 ? N : 0] int subjects_in_order_of_observed_test_pattern;
      array[n_observed_test_patterns] int n_subjects_of_observed_test_pattern
            = rep_array(0, n_observed_test_patterns);
      if (n_loo_draws > 0) {
          vector[N] observed_test_pattern = rep_vector(0.0, N);
          real power_of_two = 1.0;
          for (j in 1:n_tests) {
              observed_test_pattern += power_of_two * to_vector(obs_mask[, latent_test_order[j]]);
              power_of_two *= 2.0;
          }
          subjects_in_order_of_observed_test_pattern = sort_indices_asc(observed_test_pattern);
          for (pattern in 0:(n_observed_test_patterns - 1)) {
              n_subjects_of_observed_test_pattern[pattern + 1]
                    = number_of_elements_below(observed_test_pattern, pattern + 0.5)
                      - number_of_elements_below(observed_test_pattern, pattern - 0.5);
          }
      }
}

parameters {
      array[n_u_rows] row_vector[n_tests] u_raw;   //// array-of-rows so reduce_sum_static slices the nuisance parameters directly
      vector[n_corrs_total] Omega_unconstrained_vec;       //// FREE entries only, class then row then column
      ////
      //// ---- Factorial mean structure:
      ////
      vector[n_tests] beta_own_absent_free;                                     //// beta for own-condition ABSENT
      vector<lower=beta_own_absent_free>[n_tests] beta_own_present_free;        //// beta for own-condition PRESENT: > absent (the truncation trick)
      matrix[2, n_tests] delta_cross_free;                                      //// [own-status absent / present, test]; zeroed below unless allowed
      ////
      //// [own-status][covariate, test]; one slice unless covariate_effects_by_own_status = 1
      array[covariate_effects_by_own_status == 1 ? 2 : 1] matrix[n_covariates_max, n_tests] beta_cov_raw;
      ////
      //// ---- prevalence regression coefficients (zero-length when there are no prevalence covariates):
      vector[n_prevalence_covariates] beta_prevalence_A;
      vector[n_prevalence_covariates] beta_prevalence_B;
      ////
      //// ---- correlation pooling: the global matrix's free entries, and each class's pooling weight
      vector[n_corrs_global] Omega_global_unconstrained_vec;
      array[pool_correlations_across_classes == 1 ? n_class : 0] real<lower=0, upper=1> lambda_class;
      ////
      ////
      //// ---- class probabilities, under whichever parameterisation use_OR_prior selects. Exactly one
      //// of these two has nonzero length; the other is declared empty and costs nothing.
      ////
      array[use_OR_prior == 0 ? n_pops : 0] simplex[n_class] prev_simplex;
      ////
      array[use_OR_prior == 1 ? n_pops : 0] real z_logit_A;            //// non-centred stratum deviations
      array[use_OR_prior == 1 ? n_pops : 0] real z_logit_B;
      array[use_OR_prior == 1 ? 1 : 0]      real mu_logit_A;            //// population-typical logit prevalence
      array[use_OR_prior == 1 ? 1 : 0]      real mu_logit_B;
      array[use_OR_prior == 1 ? 1 : 0]      real<lower=0> sigma_logit_A; //// between-stratum spread
      array[use_OR_prior == 1 ? 1 : 0]      real<lower=0> sigma_logit_B;
      array[use_OR_prior == 1 ? (log_OR_shared == 1 ? 1 : n_pops) : 0] real log_OR_par;
      ////
      vector<lower=C_raw_lower, upper=C_raw_upper>[n_total_C] C_raw_vec;   //// own-status ABSENT : first cutpoint + log-increments, per test
      vector[n_total_C] d_raw_present;                                       //// own-status PRESENT: raw shift-down parameters, one per threshold (see TP)
}


transformed parameters {
      ////
      //// ---- CLASS PROBABILITIES are now PER SUBJECT (prevalence_by_subject(), functions block):
      //// the stratum random intercept plus the prevalence regression. An N x 4 matrix would be
      //// written to the output CSV every draw if declared here, so it is built LOCALLY in the model
      //// block (for the likelihood) and again in generated quantities (for the standardised
      //// estimands), each time from the same function. What this block computes is the handful of
      //// SCALARS the use_cond_prior = 1 prior needs.
      ////
      //// NO JACOBIAN IS NEEDED for the hierarchy itself: the priors below sit on mu_logit_*,
      //// sigma_logit_*, z_logit_*, beta_prevalence_* and log_OR_par - all DECLARED PARAMETERS whose
      //// own constraint transforms Stan already accounts for.
      ////
      //// use_cond_prior = 1 is the ONE exception. There the prior IS imposed on a transformed
      //// quantity - the population Pr(B | A), i.e. the standardisation-weighted
      ////      Q = sum_i w_i a_i q_i / sum_i w_i a_i
      //// - while sampling in log_OR_par[1], so a change of variables is required. Its derivative is
      //// analytic: dQ / d log_OR = sum_i w_i a_i q_i (1 - q_i) / sum_i w_i a_i. Because that
      //// derivative is evaluated at whatever prevalence draw is current, the conditional prior on
      //// Pr(B | A) is the stated logit-normal for EVERY prevalence draw, and so is the marginal.
      ////
      real Pr_B_given_A_target       = 0.0;   //// the POPULATION Pr(B | A) that carries the prior
      real log_Pr_B_given_A_target   = 0.0;   //// its log and log-complement, accumulated in log space so that
      real log1m_Pr_B_given_A_target = 0.0;   //// neither term is formed by subtracting from one
      real log_dQ_d_log_OR           = 0.0;   //// log derivative wrt log_OR_par[1], for the change of variables
      if (use_OR_prior == 1 && use_cond_prior == 1) {
//        vector[n_standardisation] l_wa;
//        vector[n_standardisation] l_waq;
//        vector[n_standardisation] l_wa1mq;
//        vector[n_standardisation] l_slope;
//        for (i in 1:n_standardisation) {
//            int n = standardisation_index[i];
//            real eta_A = mu_logit_A[1] + sigma_logit_A[1] * z_logit_A[pop[n]];
//            real eta_r = mu_logit_B[1] + sigma_logit_B[1] * z_logit_B[pop[n]];
//            if (n_prevalence_covariates > 0) {
//                eta_A += X_prevalence[n] * beta_prevalence_A;
//                eta_r += X_prevalence[n] * beta_prevalence_B;
//            }
//            real eta_q = eta_r + log_OR_par[1];
//            l_wa[i] = log(standardisation_w[i]) + log_inv_logit(eta_A);
//            l_waq[i] = l_wa[i] + log_inv_logit(eta_q);
//            l_wa1mq[i] = l_wa[i] + log1m_inv_logit(eta_q);
//            l_slope[i] = l_waq[i] + log1m_inv_logit(eta_q);
//        }
//        log_Pr_B_given_A_target = log_sum_exp(l_waq) - log_sum_exp(l_wa);
//        log1m_Pr_B_given_A_target = log_sum_exp(l_wa1mq) - log_sum_exp(l_wa);
//        log_dQ_d_log_OR = log_sum_exp(l_slope) - log_sum_exp(l_wa);
          //// (tape-chunked and vectorised: the four sums over the n_standardisation profiles are each computed
          ////  by reduce_sum_static, in chunks of chunk_size profiles, as the likelihood is over the subjects;
          ////  each chunk computes its profiles' linear predictors as one matrix-vector product and returns its
          ////  part of the sum (partial_sum_of_standardisation_profiles_for_conditional_prior), so that the main
          ////  tape holds one node per sum and none per profile. The logs of the four sums are the log_sum_exp()s
          ////  of the loop above: log_sum_exp(l_wa) = log(sum_w_a), log_sum_exp(l_waq) = log(sum_w_a_q), etc.)
          real sum_w_a        = reduce_sum_static(partial_sum_of_standardisation_profiles_for_conditional_prior,
                                                  pop_standardisation, chunk_size, 1,
                                                  X_prevalence_standardisation, standardisation_w,
                                                  mu_logit_A, sigma_logit_A, z_logit_A, beta_prevalence_A,
                                                  mu_logit_B, sigma_logit_B, z_logit_B, beta_prevalence_B,
                                                  log_OR_par);
          real sum_w_a_q      = reduce_sum_static(partial_sum_of_standardisation_profiles_for_conditional_prior,
                                                  pop_standardisation, chunk_size, 2,
                                                  X_prevalence_standardisation, standardisation_w,
                                                  mu_logit_A, sigma_logit_A, z_logit_A, beta_prevalence_A,
                                                  mu_logit_B, sigma_logit_B, z_logit_B, beta_prevalence_B,
                                                  log_OR_par);
          real sum_w_a_1m_q   = reduce_sum_static(partial_sum_of_standardisation_profiles_for_conditional_prior,
                                                  pop_standardisation, chunk_size, 3,
                                                  X_prevalence_standardisation, standardisation_w,
                                                  mu_logit_A, sigma_logit_A, z_logit_A, beta_prevalence_A,
                                                  mu_logit_B, sigma_logit_B, z_logit_B, beta_prevalence_B,
                                                  log_OR_par);
          real sum_w_a_q_1m_q = reduce_sum_static(partial_sum_of_standardisation_profiles_for_conditional_prior,
                                                  pop_standardisation, chunk_size, 4,
                                                  X_prevalence_standardisation, standardisation_w,
                                                  mu_logit_A, sigma_logit_A, z_logit_A, beta_prevalence_A,
                                                  mu_logit_B, sigma_logit_B, z_logit_B, beta_prevalence_B,
                                                  log_OR_par);
          real log_sum_w_a = log(sum_w_a);
          log_Pr_B_given_A_target = log(sum_w_a_q) - log_sum_w_a;
          log1m_Pr_B_given_A_target = log(sum_w_a_1m_q) - log_sum_w_a;
          log_dQ_d_log_OR = log(sum_w_a_q_1m_q) - log_sum_w_a;
          Pr_B_given_A_target = exp(log_Pr_B_given_A_target);
      }
      ////
      //// ---- Covariate effects indexed by own-condition status. With the switch off both slices are
      //// the same matrix, so the log-density is unchanged from the shared-effect model.
      ////
      array[2] matrix[n_covariates_max, n_tests] beta_cov_by_own_status
            = { beta_cov_raw[1], beta_cov_raw[covariate_effects_by_own_status == 1 ? 2 : 1] };
      for (st in 1:2) for (t in 1:n_tests) for (k in 1:n_covariates_max) {
          if (is_perfect_ref[t] == 1 || covariate_active[st, k, t] == 0)
              beta_cov_by_own_status[st][k, t] = 0.0;
      }
      ////
      array[n_class] matrix[n_tests, n_tests] Omega;
      array[n_class] matrix[n_tests, n_tests] L_Omega;
      matrix[n_class, n_tests] L_Omega_diag_recip;
      //// NOTE: the likelihood is evaluated once, inside reduce_sum_static, in the model block (no log_lik here).
      ////
      //// ---- beta_own[s, t]: the own-condition pair (absent / present) per test, after pinning
      //// perfect references and zeroing ordinal tests (whose location is in the cutpoints):
      ////
      matrix[2, n_tests] beta_own;
      matrix[2, n_tests] delta_cross;           //// [own-status, test]: the cross-loading may differ by own-status
      for (t in 1:n_tests) {
          if (is_perfect_ref[t] == 1) {
              beta_own[1, t] = -perfect_ref_M;
              beta_own[2, t] = +perfect_ref_M;
              delta_cross[1, t] = 0.0;       //// a perfect reference cannot cross-load
              delta_cross[2, t] = 0.0;
          } else if (mean_has_own[t] == 1) {  //// binary test: location in beta
              beta_own[1, t] = beta_own_absent_free[t];
              beta_own[2, t] = beta_own_present_free[t];
              delta_cross[1, t] = (allow_cross_loading[t] == 1) ? delta_cross_free[1, t] : 0.0;
              delta_cross[2, t] = (allow_cross_loading[t] == 1) ? delta_cross_free[2, t] : 0.0;
          } else {                            //// ordinal test: location in the cutpoints, NOT in beta
              beta_own[1, t] = 0.0;
              beta_own[2, t] = 0.0;
              delta_cross[1, t] = (allow_cross_loading[t] == 1) ? delta_cross_free[1, t] : 0.0;
              delta_cross[2, t] = (allow_cross_loading[t] == 1) ? delta_cross_free[2, t] : 0.0;
          }
      }
      ////
      //// ---- beta[c, t]: the class-by-test intercept matrix, as in the 2-class file. With the
      //// cross-loading own-status-specific this is the fully saturated 4 x T matrix (four values
      //// per test), subject to beta_own[2,t] > beta_own[1,t] and to the zeroing above:
      ////      beta[00, t] = beta_own[1, t]
      ////      beta[10, t] = beta_own[2, t]                              (for an A-test)
      ////      beta[01, t] = beta_own[1, t] + delta_cross[1, t]
      ////      beta[11, t] = beta_own[2, t] + delta_cross[2, t]
      ////
      matrix[n_class, n_tests] beta;
      for (c in 1:n_class) {
          for (t in 1:n_tests) {
              int s_own = class_D[c, test_condition[t]] + 1;
              int other = class_D[c, 3 - test_condition[t]];
              beta[c, t] = beta_own[s_own, t] + delta_cross[s_own, t] * other;
          }
      }
      ////
      ////
      //// ---- Correlations. Without pooling each class's matrix comes straight from its own free
      //// entries through the bounded LDL. With pooling, the same LDL builds ONE global matrix and a
      //// DEPARTURE matrix per class, and the class matrix is their convex combination. The LKJ
      //// priors and the LDL Jacobians attach to the COMPONENTS (global, departures); the combined
      //// matrix is a deterministic function of them and carries no prior of its own.
      ////
      matrix[n_tests, n_tests] L_Omega_global    = diag_matrix(rep_vector(1.0, n_tests));
      matrix[n_tests, n_tests] Omega_global      = diag_matrix(rep_vector(1.0, n_tests));
      array[n_class] matrix[n_tests, n_tests] L_Omega_departure;
//    if (pool_correlations_across_classes == 1) {
//          L_Omega_global = Pinkney_LDL_bounds_opt_jacobian(  Omega_global_unconstrained_vec,
//                                                             lb_corr_actual[1],
//                                                             ub_corr[1],
//                                                             known_values_indicator_list[1],
//                                                             known_values_list[1]);
//          Omega_global = multiply_lower_tri_self_transpose(L_Omega_global);
//    }
//    for (c in 1:n_class) {
//              vector[n_corrs_per_class[c]] Omega_raw_c = rep_vector(0.0, n_corrs_per_class[c]);
//              if (n_corrs_per_class[c] > 0)
//                  Omega_raw_c = segment(Omega_unconstrained_vec, corr_start_index[c], n_corrs_per_class[c]);
//              L_Omega_departure[c] = Pinkney_LDL_bounds_opt_jacobian(  Omega_raw_c,
//                                                                       lb_corr_actual[c],
//                                                                       ub_corr[c],
//                                                                       known_values_indicator_list[c],
//                                                                       known_values_list[c]);
//              if (pool_correlations_across_classes == 1) {
//                    matrix[n_tests, n_tests] Omega_departure_c = multiply_lower_tri_self_transpose(L_Omega_departure[c]);
//                    Omega[c]   = (1.0 - lambda_class[c]) * Omega_global + lambda_class[c] * Omega_departure_c;
//                    L_Omega[c] = cholesky_decompose(Omega[c]);
//              } else {
//                    L_Omega[c] = L_Omega_departure[c];
//                    Omega[c]   = multiply_lower_tri_self_transpose(L_Omega[c]);
//              }
//              L_Omega_diag_recip[c, ] = to_row_vector(1.0 ./ diagonal(L_Omega[c]));
//    }
      //// (latent_test_order: L_Omega_global, L_Omega_departure and L_Omega are built in the latent order,
      ////  which the likelihood's GHK sequence uses; Omega_global and Omega are returned in the data's
      ////  test order. With latent_test_order = 1:n_tests every value is as before.)
      {
            matrix[n_tests, n_tests] Omega_global_latent_order = diag_matrix(rep_vector(1.0, n_tests));
            if (pool_correlations_across_classes == 1) {
                  L_Omega_global = Pinkney_LDL_bounds_opt_jacobian(  Omega_global_unconstrained_vec,
                                                                     lb_corr_latent_order[1],
                                                                     ub_corr_latent_order[1],
                                                                     known_values_indicator_latent_order[1],
                                                                     known_values_latent_order[1]);
                  Omega_global_latent_order = multiply_lower_tri_self_transpose(L_Omega_global);
                  Omega_global = Omega_global_latent_order[latent_position_of_test, latent_position_of_test];
            }
            for (c in 1:n_class) {
                  vector[n_corrs_per_class[c]] Omega_raw_c = rep_vector(0.0, n_corrs_per_class[c]);
                  if (n_corrs_per_class[c] > 0)
                      Omega_raw_c = segment(Omega_unconstrained_vec, corr_start_index[c], n_corrs_per_class[c]);
                  L_Omega_departure[c] = Pinkney_LDL_bounds_opt_jacobian(  Omega_raw_c,
                                                                           lb_corr_latent_order[c],
                                                                           ub_corr_latent_order[c],
                                                                           known_values_indicator_latent_order[c],
                                                                           known_values_latent_order[c]);
                  matrix[n_tests, n_tests] Omega_c_latent_order;
                  if (pool_correlations_across_classes == 1) {
                        matrix[n_tests, n_tests] Omega_departure_c = multiply_lower_tri_self_transpose(L_Omega_departure[c]);
                        Omega_c_latent_order = (1.0 - lambda_class[c]) * Omega_global_latent_order
                                               + lambda_class[c] * Omega_departure_c;
                        L_Omega[c] = cholesky_decompose(Omega_c_latent_order);
                  } else {
                        L_Omega[c] = L_Omega_departure[c];
                        Omega_c_latent_order = multiply_lower_tri_self_transpose(L_Omega[c]);
                  }
                  Omega[c] = Omega_c_latent_order[latent_position_of_test, latent_position_of_test];
                  L_Omega_diag_recip[c, ] = to_row_vector(1.0 ./ diagonal(L_Omega[c]));
            }
      }
      ////
      //// ---- Cutpoints, one set per OWN-CONDITION STATUS (see header note 2):
      ////      C_vec_tp[1] : subject does NOT have the test's target condition
      ////      C_vec_tp[2] : subject DOES have it
      ////
      array[2] vector[n_total_C > 0 ? n_total_C : 1] C_vec_tp;
      array[2] vector[n_total_C > 0 ? n_total_C : 1] log_C_gap_tp; //// first element per test is unused
      vector[n_total_C > 0 ? n_total_C : 1] d_present;          //// the shift-down at each threshold, >= 0
      vector[n_total_C > 0 ? n_total_C : 1] log_J_d_present;    //// log-Jacobian of d_raw -> d, per threshold
      for (s in 1:2) C_vec_tp[s] = rep_vector(0.0, n_total_C > 0 ? n_total_C : 1);
      for (s in 1:2) log_C_gap_tp[s] = rep_vector(0.0, n_total_C > 0 ? n_total_C : 1);
      d_present       = rep_vector(0.0, n_total_C > 0 ? n_total_C : 1);
      log_J_d_present = rep_vector(0.0, n_total_C > 0 ? n_total_C : 1);
      if (n_ordinal_tests > 0) {
          for (t_ord in 1:n_ordinal_tests) {
                int n_thr_t = n_thr_per_ord_test[t_ord];
                int softplus = 0;
                ////
                //// ---- absent set: first cutpoint + log-increments, from C_raw_vec (as always)
                ////
                vector[n_thr_t] C_raw_t1 = get_test_values(C_raw_vec, ord_start_index, ord_end_index, t_ord);
                vector[n_thr_t] C_t1 = construct_C(C_raw_t1, softplus);
                C_vec_tp[1] = update_test_values(C_vec_tp[1], C_t1, ord_start_index, ord_end_index, t_ord);
                ////
                //// ---- present set: C^{[1]}_k = C^{[0]}_k - d_k with d_k >= 0 AND C^{[1]} increasing.
                //// C^{[1]}_k > C^{[1]}_{k-1}  <=>  d_k < d_{k-1} + (C^{[0]}_k - C^{[0]}_{k-1}), so:
                ////      d_1 = exp(r_1)                              in (0, inf)
                ////      d_k = (d_{k-1} + inc_k) * inv_logit(r_k)    in (0, d_{k-1} + inc_k),  k >= 2
                //// where inc_k = C^{[0]}_k - C^{[0]}_{k-1} > 0. Log-Jacobian |d d_k / d r_k| recorded
                //// per threshold and added to target in the model block.
                ////
                vector[n_thr_t] r_t = get_test_values(d_raw_present, ord_start_index, ord_end_index, t_ord);
                vector[n_thr_t] d_t;
                vector[n_thr_t] lJ_t;
                vector[n_thr_t] log_gap_t = rep_vector(0.0, n_thr_t);
                vector[n_thr_t] C_t2;
                d_t[1]  = exp(r_t[1]);
                lJ_t[1] = r_t[1];
                C_t2[1] = C_t1[1] - d_t[1];
                if (n_thr_t > 1) {
                    for (k in 2:n_thr_t) {
                        real cap_k = d_t[k - 1] + exp(C_raw_t1[k]);
                        real w_k   = inv_logit(r_t[k]);
                        d_t[k]  = cap_k * w_k;
                        log_gap_t[k] = log(cap_k) + log1m_inv_logit(r_t[k]);
                        C_t2[k] = C_t2[k - 1] + exp(log_gap_t[k]);
                        lJ_t[k] = log(cap_k) + log_inv_logit(r_t[k]) + log1m_inv_logit(r_t[k]);
                    }
                }
                //// Keep the mathematically positive width even when the displayed C_t2 values round equal.
                log_C_gap_tp[1] = update_test_values(log_C_gap_tp[1], C_raw_t1, ord_start_index, ord_end_index, t_ord);
                log_C_gap_tp[2] = update_test_values(log_C_gap_tp[2], log_gap_t, ord_start_index, ord_end_index, t_ord);
                C_vec_tp[2]     = update_test_values(C_vec_tp[2],     C_t2, ord_start_index, ord_end_index, t_ord);
                d_present       = update_test_values(d_present,       d_t,  ord_start_index, ord_end_index, t_ord);
                log_J_d_present = update_test_values(log_J_d_present, lJ_t, ord_start_index, ord_end_index, t_ord);
          }
      }
      ////
      //// ---- Likelihood:
      ////
}


model {
      ////
      //// ---- Factorial mean priors:
      ////
      for (t in 1:n_tests) {
          if (is_perfect_ref[t] == 0 && mean_has_own[t] == 1) {      //// binary, accuracy estimated
              beta_own_absent_free[t]  ~ normal(prior_beta_mean[1, t], prior_beta_sd[1, t]);
              beta_own_present_free[t] ~ normal(prior_beta_mean[2, t], prior_beta_sd[2, t]);  //// truncated below at beta_own_absent_free[t] by declaration
          }
          if (allow_cross_loading[t] == 1) {
              delta_cross_free[1, t] ~ normal(0.0, prior_delta_cross_sd[t]);
              delta_cross_free[2, t] ~ normal(0.0, prior_delta_cross_sd[t]);
          }
      }
      ////
      //// ---- Keep the unused parameters proper so the sampler does not wander:
      //// (perfect references, and ORDINAL tests whose location is in the cutpoints)
      ////
      for (t in 1:n_tests) {
          if (is_perfect_ref[t] == 1 || mean_has_own[t] == 0) {
              beta_own_absent_free[t]  ~ std_normal();
              beta_own_present_free[t] ~ std_normal();
          }
          if (allow_cross_loading[t] == 0) {
              delta_cross_free[1, t] ~ std_normal();
              delta_cross_free[2, t] ~ std_normal();
          }
      }
      ////
      //// ---- Covariate effects:
      ////
      //// EVERY entry gets its prior, including rows a test does not use (n_covs_per_outcome[t] <
      //// n_covariates_max): an entry with neither prior nor likelihood is an improper free parameter
      //// and wrecks R-hat without touching any quantity of interest.
      if (n_covariates_max > 0) {
          for (t in 1:n_tests) {
              for (k in 1:n_covariates_max) {
                  for (s in 1:(covariate_effects_by_own_status == 1 ? 2 : 1))
                        beta_cov_raw[s][k, t] ~ normal(prior_beta_cov_mean[s][k, t], prior_beta_cov_sd[s][k, t]);
              }
          }
      }
      ////
      //// ---- Prevalence regression coefficients:
      ////
      if (n_prevalence_covariates > 0) {
          beta_prevalence_A ~ normal(prior_beta_prevalence_A_mean, prior_beta_prevalence_A_sd);
          beta_prevalence_B ~ normal(prior_beta_prevalence_B_mean, prior_beta_prevalence_B_sd);
      }
      ////
      //// ---- Correlations:
      ////
      if (pool_correlations_across_classes == 1) {
          target += lkj_corr_cholesky_lpdf(L_Omega_global | prior_LKJ_global);
          for (c in 1:n_class) {
              target += lkj_corr_cholesky_lpdf(L_Omega_departure[c] | prior_LKJ_departure);
              lambda_class[c] ~ beta(prior_lambda_shape_a, prior_lambda_shape_b);
          }
      } else {
          for (c in 1:n_class) {
              target += lkj_corr_cholesky_lpdf(L_Omega[c] | prior_LKJ[c, 1]);
          }
      }
      ////
      //// ---- Class probabilities (replaces the Beta prior on a scalar prevalence):
      ////
      for (g in 1:n_pops) {
          if (use_OR_prior == 0) prev_simplex[g] ~ dirichlet(prior_prev_dirichlet);
      }
      ////
      //// ---- margins-and-association prior. Every statement here is on a declared parameter, so
      //// Stan's own constraint Jacobians are all that is required (see the transformed parameters
      //// block for why no user Jacobian belongs here).
      ////
      if (use_OR_prior == 1) {
          //// every statement is on a DECLARED parameter; prev_A_par / prev_B_par are transformed
          //// quantities with no distribution statement, so no user Jacobian is involved.
          z_logit_A ~ std_normal();
          z_logit_B ~ std_normal();
          mu_logit_A ~ normal(prior_mu_logit_A_mean, prior_mu_logit_A_sd);
          mu_logit_B ~ normal(prior_mu_logit_B_mean, prior_mu_logit_B_sd);
          sigma_logit_A ~ normal(0.0, prior_sigma_logit_sd);     //// half-normal via <lower=0>
          sigma_logit_B ~ normal(0.0, prior_sigma_logit_sd);
          //// ---- ASSOCIATION. use_cond_prior == 0 keeps the regularising normal prior on the shared
          //// log odds ratio. use_cond_prior == 1 instead places a logit-normal prior on the
          //// POPULATION Pr(B | A) and converts it to a density on log_OR_par[1] with the analytic
          //// derivative computed in transformed parameters. Pr_B_given_A_target is identically the
          //// Pr_B_given_A_overall reported in generated quantities, so the prior sits on the reported
          //// estimand itself rather than on a parameter that only implies it.
          if (use_cond_prior == 0) {
                log_OR_par ~ normal(prior_log_OR_mean, prior_log_OR_sd);
          } else {
                real logit_Pr_B_given_A_target = log_Pr_B_given_A_target - log1m_Pr_B_given_A_target;
                target += normal_lpdf(logit_Pr_B_given_A_target | prior_logit_Q_mean, prior_logit_Q_sd)
                          - log_Pr_B_given_A_target - log1m_Pr_B_given_A_target
                          + log_dQ_d_log_OR;
          }
      }
      ////
      //// ---- Induced-Dirichlet factors on the cutpoints + Jacobian:
      //// Their PRODUCT is restricted to C_present < C_absent at every threshold, and to the
      //// declared C_raw bounds. Thus the actual marginal accuracy priors are NOT generally the
      //// nominal Betas obtained by aggregating alpha. Inspect prior_only draws of Se_anchor_ord /
      //// Sp_anchor_ord and the population-standardised Se_ord / Sp_ord before interpreting them.
      ////
      if (n_ordinal_tests > 0) {
          for (t_ord in 1:n_ordinal_tests) {
                int n_thr_t = n_thr_per_ord_test[t_ord];
                int n_cat_t = n_cat_per_ord_test[t_ord];
                int t = t_ord + n_binary_tests;
                ////
                //// Anchor: 0 for both sets (the ordinal tests have no alpha in the mean - their
                //// location IS the cutpoints). Each set gets its own induced-Dirichlet prior on the
                //// category probabilities it implies at latent mean 0.
                ////
                real anchor_t = 0.0;
                ////
                for (s in 1:2) {
                      vector[n_thr_t] C_t = get_test_values(C_vec_tp[s], ord_start_index, ord_end_index, t_ord);
                      target += induced_dirichlet_lpdf(C_t | get_test_values(log_C_gap_tp[s], ord_start_index, ord_end_index, t_ord),
                                                       to_vector(prior_dirichlet_alpha[s][1:n_cat_t, t_ord]));
                }
                ////
                //// Jacobians: absent set - log-increments (as always); present set - the r -> d map
                //// (C^{[1]} = C^{[0]} - d is a shift, |det| = 1, so that is the whole of it).
                ////
                {
                    vector[n_thr_t] C_raw_t1 = get_test_values(C_raw_vec, ord_start_index, ord_end_index, t_ord);
                    if (n_thr_t > 1) target += sum(C_raw_t1[2:n_thr_t]);
                    target += sum(get_test_values(log_J_d_present, ord_start_index, ord_end_index, t_ord));
                }
          }
      }
      ////
      //// ---- NOTE: label switching is handled by construction - beta_own_present_free is declared
      //// lower=beta_own_absent_free (binary tests), and the present-status cutpoints are a
      //// non-negative shift below the absent ones (ordinal tests). See header notes 1 and 2.
      ////
      ////
      //// No nuisance parameters are declared under prior_only: the likelihood is bypassed.
      //// For data fits, only models with residual dependence need GHK nuisance parameters.
      ////
      if (prior_only == 0) {
          //// reduce_sum_static deterministically partitions into slices of at most chunk_size observations
          //// when compiled with STAN_THREADS=true, including when evaluated on one thread.
          if (all_correlations_zero == 1 && max(n_covs_per_outcome) == 0) {
          ////
          //// ---- per-subject class probabilities, LOCAL so they are not written out every draw:
          //// Only this CI shortcut needs the full matrix. Both reduce_sum paths below compute
          //// it inside partial_log_lik, once per chunk, including the prevalence covariates.
          ////
//        matrix[N, n_class] log_prev_subject;
//        if (use_OR_prior == 0) {
//              for (n in 1:N) log_prev_subject[n] = log(prev_simplex[pop[n]])';
//        } else {
//              log_prev_subject = log_prevalence_by_subject(X_prevalence, pop, n_class,
//                                        mu_logit_A[1], sigma_logit_A[1], z_logit_A, beta_prevalence_A,
//                                        mu_logit_B[1], sigma_logit_B[1], z_logit_B, beta_prevalence_B,
//                                        log_OR_par, log_OR_shared, use_cond_prior);
//        }
//        target += sum(ci_shared_log_lik(y, pop, class_D, test_condition, beta, log_prev_subject,
//                                   C_vec_tp, log_C_gap_tp, ord_start_index, ord_end_index,
//                                   n_thr_per_ord_test, n_cat_per_ord_test, n_binary_tests,
//                                   Phi_type, C_sentinel, overflow_threshold, underflow_threshold));
          //// (tape-chunked: the same log class probabilities and ci_shared_log_lik() terms, computed per slice
          ////  of at most chunk_size subjects by partial_log_lik_conditional_independence_shortcut, in place of
          ////  the N x n_class matrix and the per-subject loop on the main autodiff tape. The padded ordinal
          ////  metadata are passed, as zero-length arrays are not valid shared arguments of reduce_sum_static;
          ////  they equal n_thr_per_ord_test and n_cat_per_ord_test whenever there are ordinal tests.)
          target += reduce_sum_static(partial_log_lik_conditional_independence_shortcut,
                               pop,
                               chunk_size,
                               y, X_prevalence, use_OR_prior, prev_simplex,
                               mu_logit_A, sigma_logit_A, z_logit_A, beta_prevalence_A,
                               mu_logit_B, sigma_logit_B, z_logit_B, beta_prevalence_B,
                               log_OR_par, log_OR_shared, use_cond_prior,
                               n_class, class_D, test_condition,
                               beta,
                               C_vec_tp, log_C_gap_tp,
                               ord_start_index, ord_end_index,
                               n_thr_per_ord_test_pad, n_cat_per_ord_test_pad,
                               n_binary_tests, Phi_type, C_sentinel, overflow_threshold, underflow_threshold);
          } else if (all_correlations_zero == 1) {
          target += reduce_sum_static(partial_log_lik,
                               u_raw_CI,
                               chunk_size,
                               y_use, obs_mask, y_ord, y_ord_upper, X, n_covs_per_outcome, pop,
                               class_D, test_condition,
                               beta, beta_cov_by_own_status, L_Omega, L_Omega_diag_recip,
                               X_prevalence, use_OR_prior, prev_simplex,
                               mu_logit_A, sigma_logit_A, z_logit_A, beta_prevalence_A,
                               mu_logit_B, sigma_logit_B, z_logit_B, beta_prevalence_B,
                               log_OR_par, log_OR_shared, use_cond_prior,
                               C_vec_tp, log_C_gap_tp,
                               ord_start_index, ord_end_index,
                               n_thr_per_ord_test_pad, n_cat_per_ord_test_pad,
                               n_binary_tests, latent_test_order, latent_position_of_test,
                               C_sentinel, n_tests, n_class, n_covariates_max,
                               Phi_type, overflow_threshold, underflow_threshold);
              //// partial_log_lik includes the tanh Jacobian; at fixed zero it is constant.
              target += N * n_tests * log2();
          } else {
          target += reduce_sum_static(partial_log_lik,
                               u_raw,
                               chunk_size,
                               y_use, obs_mask, y_ord, y_ord_upper, X, n_covs_per_outcome, pop,
                               class_D, test_condition,
                               beta, beta_cov_by_own_status, L_Omega, L_Omega_diag_recip,
                               X_prevalence, use_OR_prior, prev_simplex,
                               mu_logit_A, sigma_logit_A, z_logit_A, beta_prevalence_A,
                               mu_logit_B, sigma_logit_B, z_logit_B, beta_prevalence_B,
                               log_OR_par, log_OR_shared, use_cond_prior,
                               C_vec_tp, log_C_gap_tp,
                               ord_start_index, ord_end_index,
                               n_thr_per_ord_test_pad, n_cat_per_ord_test_pad,
                               n_binary_tests, latent_test_order, latent_position_of_test,
                               C_sentinel, n_tests, n_class, n_covariates_max,
                               Phi_type, overflow_threshold, underflow_threshold);
          }
      }
}


generated quantities {
      array[n_pops] vector[n_class] prev;
      vector[n_pops] pop_w_std;
      vector[n_pops] pi_11;
      vector[n_pops] pi_10;
      vector[n_pops] pi_01;
      vector[n_pops] pi_00;
      vector[n_pops] prev_A;
      vector[n_pops] prev_B;
      vector[n_pops] OR_comorbid;
      vector[n_pops] log_OR_comorbid;
      vector[n_pops] Pr_B_given_A;
      vector[n_pops] Pr_A_given_B;
      vector[n_pops] pi_11_under_independence;
      array[use_OR_prior == 1 ? n_pops : 0] real prev_A_par;
      array[use_OR_prior == 1 ? n_pops : 0] real prev_B_par;
      array[n_pops] int group_in_standardisation;
      vector[n_class] prev_overall;
      real pi_11_overall;
      real pi_10_overall;
      real pi_01_overall;
      real pi_00_overall;
      real prev_A_overall;
      real prev_B_overall;
      real Pr_B_given_A_overall;
      real Pr_A_given_B_overall;
      real log_OR_comorbid_overall;
      real OR_comorbid_overall;
      real pi_11_under_independence_overall;
      vector[n_prevalence_covariates] beta_prevalence_A_out;
      vector[n_prevalence_covariates] beta_prevalence_B_out;
      matrix[2, n_tests] delta_cross_out;
      matrix[2, n_tests] beta_own_out;
      matrix[n_class, n_tests] beta_out;
      matrix[n_covariates_max, n_tests] beta_cov;
      matrix[n_covariates_max, n_tests] beta_cov_present;
      array[2] vector[n_tests] Xbeta_baseline;
      vector[n_binary_tests > 0 ? n_binary_tests : 1] Se_bin;
      vector[n_binary_tests > 0 ? n_binary_tests : 1] Sp_bin;
      array[max(n_ordinal_tests, 1)] vector[n_thr_max_gq] Se_ord;
      array[max(n_ordinal_tests, 1)] vector[n_thr_max_gq] Sp_ord;
      array[max(n_ordinal_tests, 1)] vector[n_thr_max_gq] Se_anchor_ord;
      array[max(n_ordinal_tests, 1)] vector[n_thr_max_gq] Sp_anchor_ord;
      array[2] vector[n_total_C > 0 ? n_total_C : 1] C_vec_out;
      vector[n_total_C > 0 ? n_total_C : 1] d_present_out;
      real p_refA_pos_overall;
      real p_refB_pos_overall;
      real p_ref_both_pos_overall;
      real Pr_diagnosed_B_given_diagnosed_A_overall;
      real Pr_diagnosed_A_given_diagnosed_B_overall;
      matrix[n_grid_A, n_grid_B] p_screen_both_pos;
      matrix[n_grid_A, n_grid_B] ppv_A_given_Bpos;
      matrix[n_grid_A, n_grid_B] ppv_A_given_Bneg;
      matrix[n_grid_A, n_grid_B] ppv_B_given_Apos;
      matrix[n_grid_A, n_grid_B] ppv_B_given_Aneg;
      matrix[n_grid_A, n_grid_B] ppv_both_given_bothpos;
      matrix[n_grid_A, n_grid_B] ppv_A_ratio_B;
      vector[N] log_lik;
      vector[n_class] expected_class_count;
      matrix[save_subject_probs == 1 ? N : 0, n_class] class_prob;
      matrix[2, n_tests] min_effective_separation;
      matrix[2, n_pops] screen_count_expected;
      array[2,n_pops] int screen_count_rep;
      matrix[2,n_pops] screen_present_mass;
      matrix[2,n_pops] screen_positive_present;
      matrix[2,n_pops] screen_positive_absent;
      {

      ////
      //// ---- PER-SUBJECT class probabilities, recomputed here from the same function the model
      //// block used. Local, so the N x 4 matrix is never written out. Everything below that used to
      //// read prev[g] now reads a standardisation-weighted average of prev_subject.
      ////
      matrix[N, n_class] log_prev_subject_gq;
      matrix[N, n_class] prev_subject;
      if (use_OR_prior == 0) {
//          for (n in 1:N) log_prev_subject_gq[n] = log(prev_simplex[pop[n]])';
            //// (no loop over the subjects: log_prev_for_slice() of every subject, as in the model block - the
            ////  rows of the populations' log class probabilities, indexed by the subjects' populations)
            log_prev_subject_gq = log_prev_for_slice(X_prevalence, pop, n_class, use_OR_prior, prev_simplex,
                                                     mu_logit_A, sigma_logit_A, z_logit_A, beta_prevalence_A,
                                                     mu_logit_B, sigma_logit_B, z_logit_B, beta_prevalence_B,
                                                     log_OR_par, log_OR_shared, use_cond_prior);
      } else {
            log_prev_subject_gq = log_prevalence_by_subject(X_prevalence, pop, n_class,
                                 mu_logit_A[1], sigma_logit_A[1], z_logit_A, beta_prevalence_A,
                                 mu_logit_B[1], sigma_logit_B[1], z_logit_B, beta_prevalence_B,
                                 log_OR_par, log_OR_shared, use_cond_prior);
      }
      prev_subject = exp(log_prev_subject_gq);
      ////
      //// ---- Per-stratum class probabilities: the weighted average of prev_subject over the
      //// standardisation subjects IN that stratum. With no prevalence covariates every subject in a
      //// stratum has the same prev_subject row, so this is exactly the old prev[g].
      ////
      
      pop_w_std = rep_vector(0.0, n_pops);      //// share of standardisation weight per stratum
      for (g in 1:n_pops) prev[g] = rep_vector(0.0, n_class);
//    for (i in 1:n_standardisation) {
//        int n = standardisation_index[i];
//        prev[pop[n]]   += standardisation_w[i] * prev_subject[n]';
//        pop_w_std[pop[n]] += standardisation_w[i];
//    }
      //// (no loop over the standardisation profiles: each population's profiles, a segment of
      ////  standardisation_profiles_by_pop (transformed data), as vectors)
      {
          int n_profiles_of_previous_pops = 0;
          for (g in 1:n_pops) {
              int n_profiles_in_pop = n_standardisation_profiles_in_pop[g];
              if (n_profiles_in_pop > 0) {
                  int first_position = n_profiles_of_previous_pops + 1;
                  int last_position = n_profiles_of_previous_pops + n_profiles_in_pop;
                  array[n_profiles_in_pop] int profiles
                        = standardisation_profiles_by_pop[first_position:last_position];
                  vector[n_profiles_in_pop] w = standardisation_w[profiles];
                  prev[g] = (w' * prev_subject[standardisation_index[profiles]])';
                  pop_w_std[g] = sum(w);
              }
              n_profiles_of_previous_pops += n_profiles_in_pop;
          }
      }
      for (g in 1:n_pops) {
          if (pop_w_std[g] > 0) prev[g] = prev[g] / pop_w_std[g];
          else {
              if (use_OR_prior == 0) prev[g] = prev_simplex[g];
              else {
                  matrix[1, n_prevalence_covariates] X_anchor = rep_matrix(0.0, 1, n_prevalence_covariates);
                  array[1] int group_anchor = {g};
                  matrix[1, n_class] lp_anchor = log_prevalence_by_subject(X_anchor, group_anchor, n_class,
                      mu_logit_A[1], sigma_logit_A[1], z_logit_A, beta_prevalence_A,
                      mu_logit_B[1], sigma_logit_B[1], z_logit_B, beta_prevalence_B,
                      log_OR_par, log_OR_shared, use_cond_prior);
                  prev[g] = exp(lp_anchor[1])';
              }
          } //// no target weight: report the explicitly flagged X=0 anchor, never invented 25% cells
      }
      ////
      //// ---- The primary estimand and its companions, per stratum:
      ////
      
      
      
      
                //// marginal Pr(D_A = 1)
                //// marginal Pr(D_B = 1)
           //// (pi_11 * pi_00) / (pi_10 * pi_01)
      
      
      
         //// prev_A * prev_B, for the contrast plot
      ////
      for (g in 1:n_pops) {
          pi_11[g] = prev[g][class_11];
          pi_10[g] = prev[g][class_10];
          pi_01[g] = prev[g][class_01];
          pi_00[g] = prev[g][class_00];
          prev_A[g] = pi_11[g] + pi_10[g];
          prev_B[g] = pi_11[g] + pi_01[g];
          OR_comorbid[g] = (pi_11[g] * pi_00[g]) / (pi_10[g] * pi_01[g]);
          log_OR_comorbid[g] = log(pi_11[g]) + log(pi_00[g]) - log(pi_10[g]) - log(pi_01[g]);
          Pr_B_given_A[g] = pi_11[g] / prev_A[g];
          Pr_A_given_B[g] = pi_11[g] / prev_B[g];
          pi_11_under_independence[g] = prev_A[g] * prev_B[g];
      }
      ////
      
      
      
      for (g in 1:n_pops) {
          group_in_standardisation[g] = pop_w_std[g] > 0;
          if (use_OR_prior == 1) { prev_A_par[g] = prev_A[g]; prev_B_par[g] = prev_B[g]; }
      }
      //// ---- OVERALL population-level estimands: the standardisation-weighted average over SUBJECTS.
      //// These are the numbers the paper reports. Weighting the CLASS PROBABILITIES (not the derived
      //// ratios) is what makes this a population estimand rather than an average of ratios.
      ////
      prev_overall = rep_vector(0.0, n_class);
//    for (i in 1:n_standardisation) prev_overall += standardisation_w[i] * prev_subject[standardisation_index[i]]';
      //// (no loop over the standardisation profiles)
      prev_overall = (standardisation_w' * prev_subject[standardisation_index])';
      pi_11_overall = prev_overall[class_11];
      pi_10_overall = prev_overall[class_10];
      pi_01_overall = prev_overall[class_01];
      pi_00_overall = prev_overall[class_00];
      prev_A_overall = pi_11_overall + pi_10_overall;
      prev_B_overall = pi_11_overall + pi_01_overall;
      Pr_B_given_A_overall = pi_11_overall / prev_A_overall;   //// == Pr_B_given_A_target when use_cond_prior = 1
      Pr_A_given_B_overall = pi_11_overall / prev_B_overall;
      log_OR_comorbid_overall = log(pi_11_overall) + log(pi_00_overall) - log(pi_10_overall) - log(pi_01_overall);
      OR_comorbid_overall = exp(log_OR_comorbid_overall);
      pi_11_under_independence_overall = prev_A_overall * prev_B_overall;
      ////
      //// ---- Prevalence regression coefficients, reported under their own names:
      ////
      beta_prevalence_A_out = beta_prevalence_A;
      beta_prevalence_B_out = beta_prevalence_B;
      ////
      //// ---- Cross-loading, reported explicitly (0 when not allowed):
      ////
      delta_cross_out = delta_cross;   //// [own-status absent / present, test]
      beta_own_out = beta_own;     //// [absent / present, test]
      beta_out = beta;       //// [class, test] - the 2-class file's beta, with 4 rows
      beta_cov = beta_cov_by_own_status[1];   //// own-status ABSENT  slice (== the only slice when the switch is off)
      beta_cov_present = beta_cov_by_own_status[2];   //// own-status PRESENT slice
      ////
      //// ---- Baseline covariate contribution, by own status:
      ////
      
      for (s in 1:2) {
          Xbeta_baseline[s] = rep_vector(0.0, n_tests);
          if (n_covariates_max > 0) for (t in 1:n_tests) if (n_covs_per_outcome[t] > 0)
              Xbeta_baseline[s][t] = dot_product(baseline_case[t][1:n_covs_per_outcome[t]],
                                                 to_vector(beta_cov_by_own_status[s][1:n_covs_per_outcome[t], t]));
      }
      ////
      //// ---- Covariate contribution for every STANDARDISATION subject, by own status: [status][i, t].
      //// Local; used by every population summary below.
      array[2] matrix[n_standardisation, n_tests] Xbeta_std;
      for (s in 1:2) {
          Xbeta_std[s] = rep_matrix(0.0, n_standardisation, n_tests);
          for (t in 1:n_tests) if (n_covs_per_outcome[t] > 0) {
              int nc = n_covs_per_outcome[t];
//            for (i in 1:n_standardisation)
//                Xbeta_std[s][i, t] = X[t][standardisation_index[i], 1:nc] * beta_cov_by_own_status[s][1:nc, t];
              //// (no loop over the standardisation profiles)
              Xbeta_std[s][, t] = X[t][standardisation_index, 1:nc] * beta_cov_by_own_status[s][1:nc, t];
          }
      }
      ////
      //// ---- Se/Sp for each test WITH RESPECT TO ITS OWN CONDITION, marginalising over
      //// the other condition. Note that when delta_cross[s, t] != 0 these depend on the
      //// mix of the other condition within each stratum - which is the point.
      ////
      Se_bin = rep_vector(-1.0, n_binary_tests > 0 ? n_binary_tests : 1);
      Sp_bin = rep_vector(-1.0, n_binary_tests > 0 ? n_binary_tests : 1);
      ////
      //// (the weight of every standardisation profile in every class, standardisation_w[i] times
      ////  prev_subject[standardisation_index[i], c], as one matrix; used by every sum over the profiles below)
      matrix[n_standardisation, n_class] w_of_each_profile_and_class
            = rep_matrix(standardisation_w, n_class) .* prev_subject[standardisation_index];
      ////
      if (n_binary_tests > 0) {
          for (t in 1:n_binary_tests) {
              real num_pos = 0.0;   real den_pos = 0.0;
              real num_neg = 0.0;   real den_neg = 0.0;
//            for (i in 1:n_standardisation) for (c in 1:n_class) {
//                int s_own = class_D[c, test_condition[t]] + 1;
//                real w = standardisation_w[i] * prev_subject[standardisation_index[i], c];
//                real p = Phi(beta[c, t] + Xbeta_std[s_own][i, t]);
//                if (class_D[c, test_condition[t]] == 1) { num_pos += w * p; den_pos += w; }
//                else                                    { num_neg += w * p; den_neg += w; }
//            }
              //// (no loop over the standardisation profiles: for each class, the weights and probabilities of
              ////  every profile as vectors)
              for (c in 1:n_class) {
                  int s_own = class_D[c, test_condition[t]] + 1;
                  vector[n_standardisation] w = w_of_each_profile_and_class[, c];
                  vector[n_standardisation] p = Phi(beta[c, t] + Xbeta_std[s_own][, t]);
                  if (class_D[c, test_condition[t]] == 1) { num_pos += dot_product(w, p); den_pos += sum(w); }
                  else                                    { num_neg += dot_product(w, p); den_neg += sum(w); }
              }
              Se_bin[t] = num_pos / den_pos;
              Sp_bin[t] = 1.0 - (num_neg / den_neg);
          }
      }
      ////
      
      
      ////
      if (n_ordinal_tests > 0) {
          for (t_ord in 1:n_ordinal_tests) {
              int t = n_binary_tests + t_ord;
              int n_thr_t = n_thr_per_ord_test[t_ord];
              array[2] vector[n_thr_t] C_ts;
              for (s in 1:2) C_ts[s] = get_test_values(C_vec_tp[s], ord_start_index, ord_end_index, t_ord);
              Se_ord[t_ord] = rep_vector(-1.0, n_thr_max_gq);
              Sp_ord[t_ord] = rep_vector(-1.0, n_thr_max_gq);
              for (k in 1:n_thr_t) {
                  real num_pos = 0.0;   real den_pos = 0.0;
                  real num_neg = 0.0;   real den_neg = 0.0;
//                for (i in 1:n_standardisation) for (c in 1:n_class) {
//                    int s_own = class_D[c, test_condition[t]] + 1;
//                    real w = standardisation_w[i] * prev_subject[standardisation_index[i], c];
//                    real p = Phi(-(C_ts[s_own][k] - (beta[c, t] + Xbeta_std[s_own][i, t])));   //// Pr(Y > k | class c)
//                    if (class_D[c, test_condition[t]] == 1) { num_pos += w * p; den_pos += w; }
//                    else                                    { num_neg += w * p; den_neg += w; }
//                }
                  //// (no loop over the standardisation profiles: for each class, the weights and probabilities
                  ////  Pr(Y > k | class c) of every profile as vectors)
                  for (c in 1:n_class) {
                      int s_own = class_D[c, test_condition[t]] + 1;
                      vector[n_standardisation] w = w_of_each_profile_and_class[, c];
                      vector[n_standardisation] p = Phi(-(C_ts[s_own][k] - (beta[c, t] + Xbeta_std[s_own][, t])));
                      if (class_D[c, test_condition[t]] == 1) { num_pos += dot_product(w, p); den_pos += sum(w); }
                      else                                    { num_neg += dot_product(w, p); den_neg += sum(w); }
                  }
                  Se_ord[t_ord][k] = num_pos / den_pos;
                  Sp_ord[t_ord][k] = 1.0 - (num_neg / den_neg);
              }
          }
      }
      ////
      //// ---- Accuracy at mean 0, with the OTHER condition absent. These expose the actual
      //// ordered-cutpoint prior in a prior_only fit; nominal Beta calculations alone do not.
      
      
      for (t_ord in 1:max(n_ordinal_tests, 1)) {
          Se_anchor_ord[t_ord] = rep_vector(-1.0, n_thr_max_gq);
          Sp_anchor_ord[t_ord] = rep_vector(-1.0, n_thr_max_gq);
      }
      for (t_ord in 1:n_ordinal_tests) {
          vector[n_thr_per_ord_test[t_ord]] C_abs = get_test_values(C_vec_tp[1], ord_start_index, ord_end_index, t_ord);
          vector[n_thr_per_ord_test[t_ord]] C_pre = get_test_values(C_vec_tp[2], ord_start_index, ord_end_index, t_ord);
          for (k in 1:n_thr_per_ord_test[t_ord]) {
              Se_anchor_ord[t_ord][k] = Phi(-C_pre[k]);
              Sp_anchor_ord[t_ord][k] = Phi( C_abs[k]);
          }
      }
      ////
      //// ---- Cutpoints out:
      ////
      C_vec_out = C_vec_tp;   //// [own-status][flat over tests]
      d_present_out = d_present;       //// C^{[0]} - C^{[1]} >= 0 at each threshold


      ////
      //// =====================================================================================
      //// DIAGNOSED co-occurrence, at the REFERENCE tests' clinical thresholds - as opposed to the
      //// TRUE-status co-occurrence above (Pr_B_given_A_overall, Pr_A_given_B_overall, which use the
      //// LATENT class and are what the paper's headline association is about).
      ////
      //// Here "diagnosed" is the variable-name shorthand for REFERENCE-THRESHOLD POSITIVE:
      //// ADOS >= 10 / SCID-II criteria >= 5. This need not equal an independently adjudicated
      //// clinical diagnosis. The difference from latent co-occurrence can have either sign;
      //// reference error does not necessarily attenuate the association.
      //// These are population-standardised predictions. A comparison with verified respondents
      //// must condition on their selection / scores; their observed positive rate is not a target
      //// neighbourhood for the unconditional population rate.
      ////
      
      
      
      
      
      {
            real sum_A = 0.0;
            real sum_B = 0.0;
            real sum_AB = 0.0;
            //// (for the conditional probabilities: the log weights w P(ref positive), and the probabilities of
            ////  refB positive given refA positive, of every profile and class)
            matrix[n_standardisation, n_class] log_weight_of_A_positive;
            matrix[n_standardisation, n_class] Pr_B_positive_given_A_positive;
            matrix[n_standardisation, n_class] log_weight_of_B_positive;
            //// pre-fetch the reference cutpoints ONCE per own-status, outside the class loop, exactly
            //// as the Se_ord block above does - avoids repeating the get_test_values lookup 4x.
            array[2] real C_A_ref;
            array[2] real C_B_ref;
            if (n_binary_tests < 2) {
                  for (s in 1:2) {
                        vector[n_thr_per_ord_test[1]] C_A_full = get_test_values(C_vec_tp[s], ord_start_index, ord_end_index, 1);
                        vector[n_thr_per_ord_test[2]] C_B_full = get_test_values(C_vec_tp[s], ord_start_index, ord_end_index, 2);
                        C_A_ref[s] = C_A_full[ref_cutoff_category_A - 1];   //// threshold index = category - 1
                        C_B_ref[s] = C_B_full[ref_cutoff_category_B - 1];
                  }
            }
//          for (i in 1:n_standardisation) for (c in 1:n_class) {
//                real w = standardisation_w[i] * prev_subject[standardisation_index[i], c];
//                int sA = class_D[c, test_condition[1]] + 1;
//                int sB = class_D[c, test_condition[2]] + 1;
//                real a;   //// P(Z_refA > a) = P(refA diagnosed positive | class c)
//                real b;
//                if (n_binary_tests >= 2) {
//                      //// binary view (M3/M4/M5): "positive" is Z > 0, location is beta[c,t] alone.
//                      a = -(beta[c, 1] + Xbeta_std[sA][i, 1]);
//                      b = -(beta[c, 2] + Xbeta_std[sB][i, 2]);
//                } else {
//                      //// ordinal view (M6-M10): own-status location lives in the cutpoints
//                      //// (beta_own = 0 for ordinal tests), so beta[c,t] contributes only any
//                      //// cross-loading / covariate shift, exactly as elsewhere in this file.
//                      a = C_A_ref[sA] - (beta[c, 1] + Xbeta_std[sA][i, 1]);
//                      b = C_B_ref[sB] - (beta[c, 2] + Xbeta_std[sB][i, 2]);
//                }
//                sum_A  += w * Phi(-a);
//                sum_B  += w * Phi(-b);
//                sum_AB += w * biv_norm_upper(a, b, Omega[c, 1, 2], n_quad_nodes);
//          }
            //// (no loop over the standardisation profiles: for each class, a and b of every profile as vectors,
            ////  and the orthant probabilities from biv_norm_log_Pr_Z1_above_a_and_Pr_Z2_given_Z1_above_a())
            for (c in 1:n_class) {
                  vector[n_standardisation] w = w_of_each_profile_and_class[, c];
                  int sA = class_D[c, test_condition[1]] + 1;
                  int sB = class_D[c, test_condition[2]] + 1;
                  vector[n_standardisation] a;   //// P(Z_refA > a) = P(refA diagnosed positive | class c)
                  vector[n_standardisation] b;
                  if (n_binary_tests >= 2) {
                        //// binary view (M3/M4/M5): "positive" is Z > 0, location is beta[c,t] alone.
                        a = -(beta[c, 1] + Xbeta_std[sA][, 1]);
                        b = -(beta[c, 2] + Xbeta_std[sB][, 2]);
                  } else {
                        //// ordinal view (M6-M10): own-status location lives in the cutpoints
                        //// (beta_own = 0 for ordinal tests), so beta[c,t] contributes only any
                        //// cross-loading / covariate shift, exactly as elsewhere in this file.
                        a = C_A_ref[sA] - (beta[c, 1] + Xbeta_std[sA][, 1]);
                        b = C_B_ref[sB] - (beta[c, 2] + Xbeta_std[sB][, 2]);
                  }
                  ////
                  //// (log P(refA positive) and the probability of refB positive given refA positive, over the
                  ////  nodes of the upper tail of Z_refA: the same P(both positive) for sum_AB and for both
                  ////  conditional probabilities)
                  matrix[n_standardisation, 3] quadrature_over_A_positive
                        = biv_norm_log_Pr_Z1_above_a_and_Pr_Z2_given_Z1_above_a(a, b, Omega[c, 1, 2],
                                                                                n_quad_nodes);
                  sum_A  += dot_product(w, Phi(-a));
                  sum_B  += dot_product(w, Phi(-b));
//                sum_AB += dot_product(w, biv_norm_upper(a, b, Omega[c, 1, 2], n_quad_nodes));
                  sum_AB += dot_product(w, exp(quadrature_over_A_positive[, 1])
                                           .* quadrature_over_A_positive[, 2]);
                  log_weight_of_A_positive[, c] = log(w) + quadrature_over_A_positive[, 1];
                  Pr_B_positive_given_A_positive[, c] = quadrature_over_A_positive[, 2];
                  log_weight_of_B_positive[, c] = log(w) + log_Phi(-b);
            }
            p_refA_pos_overall = sum_A;
            p_refB_pos_overall = sum_B;
            p_ref_both_pos_overall = sum_AB;
//          Pr_diagnosed_B_given_diagnosed_A_overall = sum_A > 1e-12 ? sum_AB / sum_A : 0.0;
//          Pr_diagnosed_A_given_diagnosed_B_overall = sum_B > 1e-12 ? sum_AB / sum_B : 0.0;
            //// (each conditional probability as the sum of w P(both positive) over the sum of w P(ref positive),
            ////  both from the log scale and scaled by the largest w P(ref positive): the sum in the denominator
            ////  is then at least 1, so there is no division by a sum that can be 0, and no bound; both use the
            ////  same P(both positive) as p_ref_both_pos_overall, so that each one times P(ref positive) is
            ////  P(both positive))
            matrix[n_standardisation, n_class] scaled_weight_of_A_positive
                  = exp(log_weight_of_A_positive - max(log_weight_of_A_positive));
            Pr_diagnosed_B_given_diagnosed_A_overall
                  = sum(scaled_weight_of_A_positive .* Pr_B_positive_given_A_positive)
                    / sum(scaled_weight_of_A_positive);
            Pr_diagnosed_A_given_diagnosed_B_overall
                  = sum(exp(log_weight_of_A_positive + log(Pr_B_positive_given_A_positive)
                            - max(log_weight_of_B_positive)))
                    / sum(exp(log_weight_of_B_positive - max(log_weight_of_B_positive)));
      }


      ////
      //// =====================================================================================
      //// DIAGNOSTIC quantities. Everything above is population epidemiology - how common the
      //// conditions are and how often they co-occur. These are the quantities a clinician faces:
      //// given what this person SCORED, what is the probability they have each condition?
      ////
      //// The distinction matters for this paper. Pr_B_given_A_overall is "of everyone with
      //// autism, what fraction have BPD" - a conditional prevalence. ppv_* below is "of everyone
      //// scoring this way, what fraction have autism" - a predictive value, which depends on
      //// the screens' accuracy and on prevalence, and is what misdiagnosis actually turns on.
      //// =====================================================================================
      ////
      //// ---- (a) THRESHOLD-GRID predictive values.
      ////
      //// For each pair of screen cutoffs (tA on the autism screen, tB on the BPD screen), the
      //// four cells of the 2x2 screen table are computed per latent class and mixed over classes
      //// with the population weights. Within a class the two screens are jointly normal with
      //// correlation Omega[c, tA_idx, tB_idx], so the both-positive cell is a bivariate orthant;
      //// the one-positive-one-negative cells follow by subtraction from the marginals.
      ////
      //// The DIFFERENTIAL-DIAGNOSIS contrast is the pair
      ////      ppv_A_given_Bneg[i, j]  =  P(autism | autism screen +, BPD screen -)
      ////      ppv_A_given_Bpos[i, j]  =  P(autism | autism screen +, BPD screen +)
      //// Does a positive BPD screen make autism MORE or LESS likely, in someone who has already
      //// screened positive for autism? Under conditional independence the answer is forced; with
      //// the screen-screen correlation and the cross-loading freed, it is not. Differences between
      //// model configurations quantify sensitivity to assumptions; clinical interpretation needs validation.
      ////
            //// P(both screens +)
             //// P(D_A | A screen +, B screen +)
             //// P(D_A | A screen +, B screen -)
             //// P(D_B | B screen +, A screen +)
             //// P(D_B | B screen +, A screen -)
       //// P(D_A and D_B | both screens +)
                //// ppv_A_given_Bpos / ppv_A_given_Bneg
      ////
      if (n_grid_A > 0 && n_grid_B > 0 && n_binary_tests <= 2 && n_ordinal_tests >= 2) {

            //// The screens are GLOBAL tests 3 and 4. Their ORDINAL index is the global index minus
            //// the number of binary tests, so it is 1,2 in the binary-reference view (M3-M5, where
            //// the only ordinal tests ARE the screens) and 3,4 in the all-ordinal view (M6-M10).
            //// Getting this backwards indexes the clinician references instead of the screens - and
            //// in the all-ordinal view that is silent, because 1 and 2 are valid indices.
            int tA   = 3 - n_binary_tests;               //// ordinal index of the autism screen
            int tB   = 4 - n_binary_tests;               //// ordinal index of the BPD screen
            int tA_g = 3;                                //// global index, for Omega / beta / class_D
            int tB_g = 4;
            ////
            //// (the same for every pair of cutoffs, so once only: each class's screen locations and weights of
            ////  every profile of the grid's standardisation set, as vectors)
            array[n_class] vector[n_grid_standardisation] muA_of_class;
            array[n_class] vector[n_grid_standardisation] muB_of_class;
            array[n_class] vector[n_grid_standardisation] w_of_class;
            {
                  array[n_grid_standardisation] int subjects_of_grid
                        = standardisation_index[grid_standardisation_position];
                  for (c in 1:n_class) {
                        int sA = class_D[c, test_condition[tA_g]] + 1;
                        int sB = class_D[c, test_condition[tB_g]] + 1;
                        muA_of_class[c] = beta[c, tA_g] + Xbeta_std[sA][grid_standardisation_position, tA_g];
                        muB_of_class[c] = beta[c, tB_g] + Xbeta_std[sB][grid_standardisation_position, tB_g];
                        w_of_class[c] = grid_standardisation_w .* prev_subject[subjects_of_grid, c];
                  }
            }

            for (i in 1:n_grid_A) {
              for (j in 1:n_grid_B) {

                  real joint_pp = 0.0;                   //// P(A+ , B+)
                  real joint_pn = 0.0;                   //// P(A+ , B-)
                  real joint_np = 0.0;                   //// P(A- , B+)
                  real num_A_pp = 0.0;  real num_A_pn = 0.0;
                  real num_B_pp = 0.0;  real num_B_np = 0.0;
                  real num_AB_pp = 0.0;

//                for (gi in 1:n_grid_standardisation) for (c in 1:n_class) {
//
//                      int ii = grid_standardisation_position[gi];
//                      int sA = class_D[c, test_condition[tA_g]] + 1;
//                      int sB = class_D[c, test_condition[tB_g]] + 1;
//                      real muA = beta[c, tA_g] + Xbeta_std[sA][ii, tA_g];
//                      real muB = beta[c, tB_g] + Xbeta_std[sB][ii, tB_g];
//                      ////
//                      vector[n_thr_per_ord_test[tA]] C_A = get_test_values(C_vec_tp[sA], ord_start_index, ord_end_index, tA);
//                      vector[n_thr_per_ord_test[tB]] C_B = get_test_values(C_vec_tp[sB], ord_start_index, ord_end_index, tB);
//                      ////
//                      real aA = C_A[grid_thr_A[i]] - muA;    //// screen A positive  <=>  Z_A > aA
//                      real aB = C_B[grid_thr_B[j]] - muB;
//                      ////
//                      real mA = Phi(-aA);                    //// P(A+ | c), marginal
//                      real mB = Phi(-aB);
//                      real pp = biv_norm_upper(aA, aB, Omega[c, tA_g, tB_g], n_quad_nodes);
//                      real pn = fmax(0.0, mA - pp);
//                      real np = fmax(0.0, mB - pp);
//                      real w  = grid_standardisation_w[gi] * prev_subject[standardisation_index[ii], c];
//                      ////
//                      joint_pp += w * pp;  joint_pn += w * pn;  joint_np += w * np;
//                      if (class_D[c, 1] == 1) { num_A_pp += w * pp;  num_A_pn += w * pn; }
//                      if (class_D[c, 2] == 1) { num_B_pp += w * pp;  num_B_np += w * np; }
//                      if (class_D[c, 1] == 1 && class_D[c, 2] == 1) num_AB_pp += w * pp;
//
//                }
                  //// (no loop over the grid's standardisation profiles: for each class, every profile's
                  ////  probabilities as vectors, and the orthant probabilities from
                  ////  biv_norm_log_Pr_Z1_above_a_and_Pr_Z2_given_Z1_above_a())
                  for (c in 1:n_class) {

                        int sA = class_D[c, test_condition[tA_g]] + 1;
                        int sB = class_D[c, test_condition[tB_g]] + 1;
                        ////
                        vector[n_thr_per_ord_test[tA]] C_A = get_test_values(C_vec_tp[sA], ord_start_index,
                                                                             ord_end_index, tA);
                        vector[n_thr_per_ord_test[tB]] C_B = get_test_values(C_vec_tp[sB], ord_start_index,
                                                                             ord_end_index, tB);
                        ////
                        //// screen A positive  <=>  Z_A > aA:
                        vector[n_grid_standardisation] aA = C_A[grid_thr_A[i]] - muA_of_class[c];
                        vector[n_grid_standardisation] aB = C_B[grid_thr_B[j]] - muB_of_class[c];
                        ////
//                      vector[n_grid_standardisation] mA = Phi(-aA);       //// P(A+ | c), marginal
//                      vector[n_grid_standardisation] mB = Phi(-aB);
//                      vector[n_grid_standardisation] pp = biv_norm_upper(aA, aB, Omega[c, tA_g, tB_g],
//                                                                         n_quad_nodes);
//                      vector[n_grid_standardisation] pn = fmax(0.0, mA - pp);
//                      vector[n_grid_standardisation] np = fmax(0.0, mB - pp);
                        //// (pp and pn from the same nodes, over the upper tail of Z_A; np, whose positive screen
                        ////  is B, from the nodes over the upper tail of Z_B: no subtraction and no bound)
                        matrix[n_grid_standardisation, 3] quadrature_over_A_positive
                              = biv_norm_log_Pr_Z1_above_a_and_Pr_Z2_given_Z1_above_a(aA, aB,
                                                                                      Omega[c, tA_g, tB_g],
                                                                                      n_quad_nodes);
                        matrix[n_grid_standardisation, 3] quadrature_over_B_positive
                              = biv_norm_log_Pr_Z1_above_a_and_Pr_Z2_given_Z1_above_a(aB, aA,
                                                                                      Omega[c, tA_g, tB_g],
                                                                                      n_quad_nodes);
                        vector[n_grid_standardisation] pp = exp(quadrature_over_A_positive[, 1])
                                                            .* quadrature_over_A_positive[, 2];
                        vector[n_grid_standardisation] pn = exp(quadrature_over_A_positive[, 1])
                                                            .* quadrature_over_A_positive[, 3];
                        vector[n_grid_standardisation] np = exp(quadrature_over_B_positive[, 1])
                                                            .* quadrature_over_B_positive[, 3];
                        vector[n_grid_standardisation] w = w_of_class[c];
                        ////
                        joint_pp += dot_product(w, pp);  joint_pn += dot_product(w, pn);
                        joint_np += dot_product(w, np);
                        if (class_D[c, 1] == 1) {
                              num_A_pp += dot_product(w, pp);  num_A_pn += dot_product(w, pn);
                        }
                        if (class_D[c, 2] == 1) {
                              num_B_pp += dot_product(w, pp);  num_B_np += dot_product(w, np);
                        }
                        if (class_D[c, 1] == 1 && class_D[c, 2] == 1) num_AB_pp += dot_product(w, pp);

                  }

                  p_screen_both_pos[i, j]      = joint_pp;
                  ppv_A_given_Bpos[i, j]       = joint_pp > 0 ? num_A_pp  / joint_pp : 0.0;
                  ppv_A_given_Bneg[i, j]       = joint_pn > 0 ? num_A_pn  / joint_pn : 0.0;
                  ppv_B_given_Apos[i, j]       = joint_pp > 0 ? num_B_pp  / joint_pp : 0.0;
                  ppv_B_given_Aneg[i, j]       = joint_np > 0 ? num_B_np  / joint_np : 0.0;
                  ppv_both_given_bothpos[i, j] = joint_pp > 0 ? num_AB_pp / joint_pp : 0.0;
                  ppv_A_ratio_B[i, j]          = ppv_A_given_Bneg[i, j] > 0
                                                 ? ppv_A_given_Bpos[i, j] / ppv_A_given_Bneg[i, j] : 0.0;

              }
            }

      } else {

            p_screen_both_pos      = rep_matrix(0.0, n_grid_A, n_grid_B);
            ppv_A_given_Bpos       = rep_matrix(0.0, n_grid_A, n_grid_B);
            ppv_A_given_Bneg       = rep_matrix(0.0, n_grid_A, n_grid_B);
            ppv_B_given_Apos       = rep_matrix(0.0, n_grid_A, n_grid_B);
            ppv_B_given_Aneg       = rep_matrix(0.0, n_grid_A, n_grid_B);
            ppv_both_given_bothpos = rep_matrix(0.0, n_grid_A, n_grid_B);
            ppv_A_ratio_B          = rep_matrix(0.0, n_grid_A, n_grid_B);

      }

      ////
      //// ---- log_lik for LOO (marginal, GHK Monte Carlo with n_loo_draws draws; 0 = skip), and
      //// ---- (b) PER-SUBJECT posterior class probabilities, which fall out of the same pass.
      ////
      //// class_prob[n, c] = P(class = c | this subject's observed scores). For a subject with
      //// both screens but no reference - 92% of the sample - this is the model's actual
      //// diagnostic statement about them, and it is what the design-weighted analysis cannot
      //// produce at all, since it uses only the verified.
      ////
      //// expected_class_count[c] = sum_n class_prob[n, c] is the model-implied number of people
      //// in each class, including the (1,1) cell: the expected number of comorbid individuals in
      //// this sample, as opposed to the 3 the references happened to catch.
      ////
      log_lik = rep_vector(0.0, N);
      expected_class_count = rep_vector(0.0, n_class);
      
      ////
      //// (commented out: the LOO log-likelihood over the observed tests only, log_lik_obs_ghk_of_every_
      ////  subject_rng(), which integrates the missing tests out instead of the GHK method; n_loo_draws > 0
      ////  is rejected in transformed data, so log_lik and expected_class_count stay 0)
//    if (n_loo_draws > 0) {
//        for (n in 1:N) {
//            vector[n_class] lp_c = log_lik_obs_ghk_rng(n, n_loo_draws,
//                                             y_use, obs_mask, y_ord, y_ord_upper, X, n_covs_per_outcome, pop,
//                                             class_D, test_condition,
//                                             beta, beta_cov_by_own_status, L_Omega, log_prev_subject_gq, C_vec_tp, log_C_gap_tp,
//                                             ord_start_index, ord_end_index,
//                                             n_thr_per_ord_test, n_cat_per_ord_test,
//                                             n_binary_tests, latent_test_order, latent_position_of_test,
//                                             C_sentinel, n_tests, n_class, n_covariates_max);
//            log_lik[n] = log_sum_exp(lp_c);
//            vector[n_class] pc = softmax(lp_c);
//            expected_class_count += pc;
//            if (save_subject_probs == 1) class_prob[n] = to_row_vector(pc);
//        }
          //// (no loop over the subjects: the joint log p(y_n, class = c) of every subject at once, from
          ////  log_lik_obs_ghk_of_every_subject_rng(); log_lik is the log_sum_exp of each row, the class
          ////  probabilities its softmax, exp(row - log_lik), and expected_class_count their sum over the
          ////  subjects)
//        matrix[N, n_class] lp_class_of_every_subject
//              = log_lik_obs_ghk_of_every_subject_rng(n_loo_draws,
//                                                     y_use, y_ord, y_ord_upper, X, n_covs_per_outcome,
//                                                     class_D, test_condition,
//                                                     beta, beta_cov_by_own_status, L_Omega, log_prev_subject_gq,
//                                                     C_vec_tp, log_C_gap_tp,
//                                                     ord_start_index, ord_end_index,
//                                                     n_thr_per_ord_test, n_cat_per_ord_test,
//                                                     n_binary_tests, latent_test_order,
//                                                     subjects_in_order_of_observed_test_pattern,
//                                                     n_subjects_of_observed_test_pattern,
//                                                     n_tests, n_class);
//        log_lik = log_sum_exp_2d(lp_class_of_every_subject);
//        matrix[N, n_class] class_prob_of_every_subject
//              = exp(lp_class_of_every_subject - rep_matrix(log_lik, n_class));
//        expected_class_count = (rep_row_vector(1.0, N) * class_prob_of_every_subject)';
//        if (save_subject_probs == 1) class_prob = class_prob_of_every_subject;
//    }
      
      for (other in 0:1) for (t in 1:n_tests) {
          real gap = t <= n_binary_tests ? beta_own[2,t] - beta_own[1,t]
              : C_vec_tp[1][ord_start_index[t-n_binary_tests] + min(n_thr_per_ord_test[t-n_binary_tests], (t == 1 ? ref_cutoff_category_A-1 : t == 2 ? ref_cutoff_category_B-1 : t == 3 ? 10 : 5)) - 1]
                - C_vec_tp[2][ord_start_index[t-n_binary_tests] + min(n_thr_per_ord_test[t-n_binary_tests], (t == 1 ? ref_cutoff_category_A-1 : t == 2 ? ref_cutoff_category_B-1 : t == 3 ? 10 : 5)) - 1];
          real smallest = positive_infinity();
//        for (n in 1:N) {
//            real contrast = gap + other * (delta_cross[2,t] - delta_cross[1,t]);
//            int nc = n_covs_per_outcome[t];
//            if (nc > 0) contrast += X[t][n,1:nc] * (beta_cov_by_own_status[2][1:nc,t] - beta_cov_by_own_status[1][1:nc,t]);
//            smallest = fmin(smallest, contrast);
//        }
          //// (no loop over the subjects: the contrast of every subject as one vector, and its smallest value)
          {
              real contrast_without_covariates = gap + other * (delta_cross[2,t] - delta_cross[1,t]);
              int nc = n_covs_per_outcome[t];
              if (nc > 0) {
                  vector[nc] beta_cov_difference = beta_cov_by_own_status[2][1:nc,t]
                                                   - beta_cov_by_own_status[1][1:nc,t];
                  smallest = fmin(smallest,
                                  min(contrast_without_covariates + X[t][, 1:nc] * beta_cov_difference));
              } else smallest = fmin(smallest, contrast_without_covariates);
          }
          min_effective_separation[other+1,t] = smallest;
      }

      screen_count_expected = rep_matrix(0.0, 2, n_pops);
      screen_count_rep = rep_array(0,2,n_pops);
      screen_present_mass = rep_matrix(0.0,2,n_pops);
      screen_positive_present = rep_matrix(0.0,2,n_pops);
      screen_positive_absent = rep_matrix(0.0,2,n_pops);
      for (t in 3:4) {
          int to = t - n_binary_tests;
          int cut = min(n_thr_per_ord_test[to], t == 3 ? 10 : 5);
          int at = ord_start_index[to] + cut - 1;
//        for (n in 1:N) if (y[n,t] >= 0) {
//            real pn = 0.0;
//            for (c in 1:n_class) {
//                int own = class_D[c,test_condition[t]] + 1;
//                int nc = n_covs_per_outcome[t];
//                real mu = beta[c,t];
//                real wc = prev_subject[n,c];
//                if (nc > 0) mu += X[t][n,1:nc] * beta_cov_by_own_status[own][1:nc,t];
//                real pc = Phi(mu - C_vec_tp[own][at]);
//                pn += wc * pc;
//                if (own == 2) {
//                    screen_present_mass[t-2,pop[n]] += wc;
//                    screen_positive_present[t-2,pop[n]] += wc * pc;
//                } else screen_positive_absent[t-2,pop[n]] += wc * pc;
//            }
//            screen_count_expected[t-2,pop[n]] += pn;
//            screen_count_rep[t-2,pop[n]] += bernoulli_rng(fmin(1.0,fmax(0.0,pn)));
//        }
          //// (no loop over the subjects: the subjects with test t observed, grouped by population
          ////  (observed_subjects_of_test_by_pop, transformed data), as vectors; each population's sums are the
          ////  sums of its segment, of n_observed_of_test_in_pop[t, g] subjects)
          int n_observed_t = N - n_missing_of_test[t];
          if (n_observed_t > 0) {
              array[n_observed_t] int observed_t = observed_subjects_of_test_by_pop[t, 1:n_observed_t];
              int nc = n_covs_per_outcome[t];
              vector[n_observed_t] pn = rep_vector(0.0, n_observed_t);
              vector[n_observed_t] present_mass = rep_vector(0.0, n_observed_t);
              vector[n_observed_t] positive_present = rep_vector(0.0, n_observed_t);
              vector[n_observed_t] positive_absent = rep_vector(0.0, n_observed_t);
              //// (log of each class's share of pn and of 1 - pn, for the replicated screens below)
              matrix[n_observed_t, n_class] log_pn_terms;
              matrix[n_observed_t, n_class] log_1m_pn_terms;
              for (c in 1:n_class) {
                  int own = class_D[c,test_condition[t]] + 1;
                  vector[n_observed_t] mu = rep_vector(beta[c,t], n_observed_t);
                  vector[n_observed_t] wc = prev_subject[observed_t, c];
                  if (nc > 0) mu += X[t][observed_t, 1:nc] * beta_cov_by_own_status[own][1:nc,t];
                  vector[n_observed_t] pc = Phi(mu - C_vec_tp[own][at]);
                  pn += wc .* pc;
                  log_pn_terms[, c] = log_prev_subject_gq[observed_t, c] + log_Phi(mu - C_vec_tp[own][at]);
                  log_1m_pn_terms[, c] = log_prev_subject_gq[observed_t, c] + log_Phi(C_vec_tp[own][at] - mu);
                  if (own == 2) {
                      present_mass += wc;
                      positive_present += wc .* pc;
                  } else positive_absent += wc .* pc;
              }
//            array[n_observed_t] int screen_positive_rep = bernoulli_rng(fmin(1.0, fmax(0.0, pn)));
              //// (the replicated screens from the log odds of pn, from the log class probabilities and log_Phi
              ////  of both tails: no probability above 1 or below 0 by rounding, so no bound; bernoulli_logit_rng
              ////  draws with inv_logit of its argument, i.e. pn, as bernoulli_rng(pn) did)
              vector[n_observed_t] log_odds_of_pn = log_sum_exp_2d(log_pn_terms)
                                                    - log_sum_exp_2d(log_1m_pn_terms);
              array[n_observed_t] int screen_positive_rep = bernoulli_logit_rng(log_odds_of_pn);
              int n_subjects_of_previous_pops = 0;
              for (g in 1:n_pops) {
                  int n_observed_in_pop = n_observed_of_test_in_pop[t, g];
                  if (n_observed_in_pop > 0) {
                      int first_position = n_subjects_of_previous_pops + 1;
                      int last_position = n_subjects_of_previous_pops + n_observed_in_pop;
                      screen_present_mass[t-2, g] = sum(present_mass[first_position:last_position]);
                      screen_positive_present[t-2, g] = sum(positive_present[first_position:last_position]);
                      screen_positive_absent[t-2, g] = sum(positive_absent[first_position:last_position]);
                      screen_count_expected[t-2, g] = sum(pn[first_position:last_position]);
                      screen_count_rep[t-2, g] = sum(screen_positive_rep[first_position:last_position]);
                  }
                  n_subjects_of_previous_pops += n_observed_in_pop;
              }
          }
      }


      }
}

