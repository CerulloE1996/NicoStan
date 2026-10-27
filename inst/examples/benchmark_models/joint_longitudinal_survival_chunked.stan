

// joint_longitudinal_survival_chunked_no_loops_over_N.stan
// joint_longitudinal_survival_chunked.stan without any loop over the subjects or the observations: the same log
// density, with every calculation whose size grows with the subjects vectorised and inside the chunks of
// reduce_sum_static. More specifically:
//   - b_raw is declared as array[N_subj] row_vector[2] (subject i's b_raw[i, 1] and b_raw[i, 2]; the same
//     unconstrained parameters, in the same order, as matrix[2, N_subj]), as u_raw in
//     LC_MVP_bin_PartialLog_v5_reduce_sum_static.stan, so that the sliced argument of reduce_sum_static is the
//     parameter itself: one single name holding everything per subject, nothing built on the main tape, in place
//     of b_raw_by_subject and the loop over the subjects which filled it (the sliced argument is never an
//     expression: Stan math's reduce_sum holds the result of an expression by value and copies all of it at every
//     split of the subjects into chunks, i.e. O(N x N_chunks));
//   - in the chunks, the log hazards at the quadrature points are a Q x n_subjects matrix (subject i's Q points
//     in column i): the subject-level part repeated over the Q rows (rep_matrix), the longitudinal fixed effects
//     at the points, and the outer product of (quad_points + 1) / 2 with each subject's slope times T_surv, in
//     place of the loop over the chunk's quadrature points which gave each point its subject;
//   - the position in the chunk of each observation's subject is subj_long - (start - 1), as one vector
//     operation (to_int of a real vector), in place of the loop over the chunk's observations;
//   - transformed data: the counts of observations before each subject (observations_before) from one sort of
//     subj_long together with the points 0.5, 1.5, ..., N_subj + 0.5, and the stacked quadrature data as outer
//     products (loops over the Q quadrature points and the K_long covariates only), in place of the loops over
//     the observations and the subjects;
//   - generated quantities: the same quantities with whole-vector operations, in place of the loops over the
//     observations, the subjects and their quadrature points.
// The description of joint_longitudinal_survival_chunked.stan follows.
//
// joint_longitudinal_survival_chunked.stan
// Tape-chunked version of joint_longitudinal_survival.stan: the same log density, with both
// likelihoods summed over the subjects by reduce_sum_static, in chunks of chunk_size subjects (data), so that
// each chunk's autodiff tape is built, differentiated and released before the next one (the
// chunking paper's chunking;
// N_chunks from NicoStan::fn_compute_initial_N_chunks()). Each chunk computes its subjects' terms with the
// whole-vector operations of joint_longitudinal_survival.stan (its subjects' observations and quadrature points
// are contiguous); eta_long and hazard_subj, transformed parameters there, are reported as generated quantities
// (the same values).
// The chunking data are NicoStan's for Stan models: N (the subjects, the units of the chunks; it must equal
// N_subj) and chunk_size, which NicoStan sets to ceiling(N / N_chunks) in each phase (num_chunks_burnin,
// num_chunks_sampling).
//
// Inside the chunks (everything whose cost grows with the subjects): the standard normal prior of the chunk's
// subjects' b_raw; the chunk's random effects b0 and b1, from b_raw and sigma_b0, sigma_b1 and rho_b; the
// longitudinal linear predictor and likelihood of the chunk's observations; the log hazards at the chunk's
// quadrature points and their weighted sum (the chunk's cumulative hazards); and the random-effect and
// survival-covariate part of the chunk's events' log hazards. Outside the chunks, only terms of fixed size: the
// priors of the main parameters, N_events * log(lambda) and the events' longitudinal fixed-effect term, a dot
// product of X_long_event summed over the events (data) with alpha_assoc * beta_long. The sliced argument is
// b_raw_by_subject, the subjects' columns of b_raw (the same autodiff variables, so no new nodes on the main
// tape), so that each chunk copies only its own subjects' b_raw rather than all of it; b0 and b1, transformed
// parameters before (computed outside the chunks and copied whole into every chunk), are reported as generated
// quantities (the same values).

functions {
    // real partial_sum_subjects_lpmf(array[] int subject_slice, int start, int end,
    //                                matrix X_long, vector Y_long, vector time_long, array[] int subj_long,
    //                                array[] int observations_before,
    //                                matrix X_surv, vector T_surv, vector event_indicator,
    //                                vector quad_weights,
    //                                matrix X_long_event, matrix X_long_quad_stacked, matrix X_surv_stacked,
    //                                vector t_quad, array[] int subject_of_quad,
    //                                vector beta_long, real sigma_long, vector b0, vector b1,
    //                                vector beta_surv, real lambda, real alpha_assoc) {
    //     int Q = rows(quad_weights);
    //     int n_subjects = end - start + 1;
    //     real lp = 0;
    //     // Longitudinal likelihood: the chunk's observations (subj_long is sorted, so they are contiguous)
    //     int n1 = observations_before[start] + 1;
    //     int n2 = observations_before[end + 1];
    //     if (n2 >= n1) {
    //       vector[n2 - n1 + 1] eta_long_chunk = X_long[n1:n2] * beta_long + b0[subj_long[n1:n2]]
    //                                            + b1[subj_long[n1:n2]] .* time_long[n1:n2];
    //       lp += normal_lupdf(Y_long[n1:n2] | eta_long_chunk, sigma_long);
    //     }
    //     // Survival likelihood: the chunk's quadrature points (rows (start - 1) * Q + 1 to end * Q;
    //     // subject i's Q points are column i of the Q x n_subjects matrix), then each event's log hazard
    //     // at its time
    //     {
    //       int r1 = (start - 1) * Q + 1;
    //       int r2 = end * Q;
    //       vector[r2 - r1 + 1] hazard_quad = lambda * exp(X_surv_stacked[r1:r2] * beta_surv
    //                                                      + alpha_assoc * (X_long_quad_stacked[r1:r2]
    //                                                                       * beta_long
    //                                                                       + b0[subject_of_quad[r1:r2]]
    //                                                                       + b1[subject_of_quad[r1:r2]]
    //                                                                         .* t_quad[r1:r2]));
    //       vector[n_subjects] cum_hazard = (to_matrix(hazard_quad, Q, n_subjects)' * quad_weights)
    //                                       .* T_surv[start:end] / 2;
    //       vector[n_subjects] log_hazard_subj = log(lambda) + X_surv[start:end] * beta_surv
    //                                            + alpha_assoc * (X_long_event[start:end] * beta_long
    //                                                             + b0[start:end]
    //                                                             + b1[start:end] .* T_surv[start:end]);
    //       lp += dot_product(event_indicator[start:end], log_hazard_subj) - sum(cum_hazard);
    //     }
    //     return lp;
    // }
    // b_raw_slice: the chunk's subjects' columns of b_raw (subject i's b_raw[1, i] and b_raw[2, i]), sliced by
    // reduce_sum_static, so that each chunk copies only its own subjects' b_raw
    // real partial_sum_subjects_lpdf(array[] row_vector b_raw_slice, int start, int end, data int Q,
    //                                data matrix X_long, data vector Y_long, data vector time_long,
    //                                data array[] int subj_long, data array[] int observations_before,
    //                                data matrix X_surv, data vector event_indicator, data vector event_time,
    //                                data matrix X_long_quad_stacked, data vector t_quad,
    //                                data vector quad_weight_times_half_T,
    //                                vector beta_long, real sigma_long, real sigma_b0,
    //                                real sigma_b1, real rho_b, vector beta_surv, real lambda,
    //                                real alpha_assoc) {
    // (b_raw_slice: the chunk's subjects' elements of b_raw, i.e. subject i's b_raw[i, 1] and b_raw[i, 2];
    //  quad_time_fraction: (quad_points + 1) / 2, the fraction of T_surv[i] at each quadrature point)
    real partial_sum_subjects_lpdf(array[] row_vector b_raw_slice, int start, int end, data int Q,
                                   data matrix X_long, data vector Y_long, data vector time_long,
                                   data array[] int subj_long, data array[] int observations_before,
                                   data matrix X_surv, data vector T_surv, data vector event_indicator,
                                   data vector event_time,
                                   data matrix X_long_quad_stacked, data vector quad_time_fraction,
                                   data vector quad_weight_times_half_T,
                                   vector beta_long, real sigma_long, real sigma_b0,
                                   real sigma_b1, real rho_b, vector beta_surv, real lambda,
                                   real alpha_assoc) {
        int n_subjects = end - start + 1;
        real lp = 0;
        matrix[n_subjects, 2] b_raw_chunk = to_matrix(b_raw_slice);
        // the chunk's observations (subj_long is sorted, so they are contiguous) and the chunk's quadrature
        // points (rows (start - 1) * Q + 1 to end * Q, Q per subject)
        int n1 = observations_before[start] + 1;
        int n2 = observations_before[end + 1];
        int r1 = (start - 1) * Q + 1;
        int r2 = end * Q;
        // the position in the chunk of each quadrature point's subject
        // array[r2 - r1 + 1] int subject_in_chunk_of_quad;
        // for (i in 1:n_subjects) {
        //   for (q in 1:Q) {
        //     subject_in_chunk_of_quad[(i - 1) * Q + q] = i;
        //   }
        // }
        // (not needed: the log hazards at the chunk's quadrature points are a Q x n_subjects matrix, below)
        // the chunk's random effects: the non-centred prior of its subjects' b_raw, then b0 and b1
        lp += std_normal_lupdf(to_vector(b_raw_chunk));
        vector[n_subjects] b0_chunk = sigma_b0 * b_raw_chunk[:, 1];
        vector[n_subjects] b1_chunk = sigma_b1 * (rho_b * b_raw_chunk[:, 1]
                                                  + sqrt(1 - square(rho_b)) * b_raw_chunk[:, 2]);
        // Longitudinal likelihood: the chunk's observations
        if (n2 >= n1) {
          // array[n2 - n1 + 1] int subject_in_chunk_of_observation;
          // for (k in 1:(n2 - n1 + 1)) {
          //   subject_in_chunk_of_observation[k] = subj_long[n1 + k - 1] - start + 1;
          // }
          // (the position in the chunk of each observation's subject, subj_long - (start - 1), as one vector
          //  operation: Stan has no arithmetic on integer arrays, hence through a real vector and to_int)
          array[n2 - n1 + 1] int subject_in_chunk_of_observation
                = to_int(to_array_1d(to_vector(subj_long[n1:n2]) - (start - 1)));
          vector[n2 - n1 + 1] eta_long_chunk = X_long[n1:n2] * beta_long
                                               + b0_chunk[subject_in_chunk_of_observation]
                                               + b1_chunk[subject_in_chunk_of_observation] .* time_long[n1:n2];
          lp += normal_lupdf(Y_long[n1:n2] | eta_long_chunk, sigma_long);
        }
        // Survival likelihood: the subject-level part of the log hazard, X_surv[i] * beta_surv
        // + alpha_assoc * b0[i], and its slope in time, alpha_assoc * b1[i], once per subject; the log hazards at
        // the chunk's quadrature points minus log(lambda); the events' random-effect and survival-covariate
        // terms; and minus the chunk's cumulative hazards, lambda times one dot product of the exp of those log
        // hazards with quad_weight_times_half_T (as in joint_longitudinal_survival.stan)
        {
          vector[n_subjects] subject_part = X_surv[start:end] * beta_surv + alpha_assoc * b0_chunk;
          vector[n_subjects] subject_slope = alpha_assoc * b1_chunk;
          // vector[r2 - r1 + 1] log_hazard_quad_minus_log_lambda = subject_part[subject_in_chunk_of_quad]
          //                                                        + X_long_quad_stacked[r1:r2]
          //                                                          * (alpha_assoc * beta_long)
          //                                                        + subject_slope[subject_in_chunk_of_quad]
          //                                                          .* t_quad[r1:r2];
          // (the log hazards minus log(lambda) at the chunk's quadrature points, as a Q x n_subjects matrix,
          //  subject i's Q points in column i: its subject-level part in every row; the longitudinal fixed
          //  effects at its points; and its slope times the points' times, T_surv[i] * (quad_points + 1) / 2,
          //  the outer product of quad_time_fraction with the subjects' slopes times T_surv)
          matrix[Q, n_subjects] log_hazard_quad_minus_log_lambda
                = rep_matrix(subject_part', Q)
                  + to_matrix(X_long_quad_stacked[r1:r2] * (alpha_assoc * beta_long), Q, n_subjects)
                  + quad_time_fraction * (T_surv[start:end] .* subject_slope)';
          lp += dot_product(event_indicator[start:end], subject_part)
                + dot_product(event_time[start:end], subject_slope);
          // lp += -lambda * dot_product(quad_weight_times_half_T[r1:r2], exp(log_hazard_quad_minus_log_lambda));
          // (exp of the matrix as a vector, column by column, i.e. in the order of the stacked points)
          lp += -lambda * dot_product(quad_weight_times_half_T[r1:r2],
                                      exp(to_vector(log_hazard_quad_minus_log_lambda)));
        }
        return lp;
    }
}


data {
  // Longitudinal data
  int<lower=1> N_long;              // number of longitudinal observations
  int<lower=1> N_subj;              // number of subjects
  int<lower=1> K_long;              // number of longitudinal covariates
  matrix[N_long, K_long] X_long;    // longitudinal covariate matrix
  vector[N_long] Y_long;            // longitudinal outcomes
  array[N_long] int<lower=1,upper=N_subj> subj_long; // subject IDs for longitudinal data
  vector[N_long] time_long;         // measurement times

  // Survival data
  int<lower=1> K_surv;              // number of survival covariates
  matrix[N_subj, K_surv] X_surv;    // survival covariate matrix
  vector<lower=0>[N_subj] T_surv;   // survival times
  array[N_subj] int<lower=0,upper=1> event; // event indicators

  // Quadrature for integration
  int<lower=1> Q;                   // number of quadrature points
  vector[Q] quad_points;            // quadrature points
  vector[Q] quad_weights;           // quadrature weights
  matrix[N_subj, K_long] X_long_event; // longitudinal fixed-effect design at T_surv
  array[N_subj] matrix[Q, K_long] X_long_quad; // same design at each subject's quadrature times

  // int<lower=1> chunk_size;           // subjects per chunk (reduce_sum_static)
  int<lower=1> N;                   // the units of the chunks: the subjects (must equal N_subj)
  int<lower=1> chunk_size;          // subjects per chunk (reduce_sum_static)
}


transformed data {
  // N, the units of the chunks (NicoStan's chunking data), must be the subjects
  if (N != N_subj) {
    reject("N (the units of the chunks) must equal N_subj, the number of subjects; found N = ", N,
           " and N_subj = ", N_subj);
  }
  // the observations of subjects 1, ..., i - 1 are the first observations_before[i] (subj_long must be sorted, so
  // that each subject's observations are contiguous)
  // array[N_subj + 1] int observations_before = rep_array(0, N_subj + 1);
  // array[N_subj] int subject_index = linspaced_int_array(N_subj, 1, N_subj);
  // for (n in 1:N_long) {
  //   if (n > 1 && subj_long[n] < subj_long[n - 1]) {
  //     reject("subj_long must be sorted (the tape-chunked model needs each subject's observations contiguous)");
  //   }
  //   observations_before[subj_long[n] + 1] += 1;
  // }
  // for (i in 1:N_subj) {
  //   observations_before[i + 1] += observations_before[i];
  // }
  // (subj_long sorted: no difference between consecutive subject IDs is negative)
  if (N_long > 1) {
    if (min(to_vector(subj_long[2:N_long]) - to_vector(subj_long[1:(N_long - 1)])) < 0) {
      reject("subj_long must be sorted (the tape-chunked model needs each subject's observations contiguous)");
    }
  }
  // (observations_before[i], the number of subj_long below i - 0.5, without a loop over the observations: the
  //  points 0.5, 1.5, ..., N_subj + 0.5 are sorted together with subj_long; the point i - 0.5 then has
  //  observations_before[i] observations and i - 1 points before it, i.e. it is at position
  //  observations_before[i] + i of the sorted order; the position of each element in the sorted order is the
  //  inverse of the sorting permutation, i.e. sort_indices_asc() of sort_indices_asc())
  array[N_long + N_subj + 1] int position_in_sorted_order_of_subj_long_and_half_points
        = sort_indices_asc(sort_indices_asc(append_row(to_vector(subj_long),
                                                       to_vector(linspaced_int_array(N_subj + 1, 0, N_subj))
                                                       + 0.5)));
  array[N_subj + 1] int observations_before
        = to_int(to_array_1d(to_vector(position_in_sorted_order_of_subj_long_and_half_points[
                                             (N_long + 1):(N_long + N_subj + 1)])
                             - to_vector(linspaced_int_array(N_subj + 1, 1, N_subj + 1))));
  // every subject's Q quadrature points, stacked (subject i in rows (i - 1) * Q + 1 to i * Q), as in
  // joint_longitudinal_survival.stan, with the subject's survival covariates on each row; the events as a vector
  int N_quad = N_subj * Q;
  // vector[N_quad] t_quad;
  matrix[N_quad, K_long] X_long_quad_stacked;
  // matrix[N_quad, K_surv] X_surv_stacked;
  // array[N_quad] int subject_of_quad;
  // for (i in 1:N_subj) {
  //   for (q in 1:Q) {
  //     int r = (i - 1) * Q + q;
  //     t_quad[r] = T_surv[i] * (quad_points[q] + 1) / 2;
  //     X_long_quad_stacked[r] = X_long_quad[i][q];
  //     X_surv_stacked[r] = X_surv[i];
  //     subject_of_quad[r] = i;
  //   }
  // }
  // (t_quad, X_surv_stacked and subject_of_quad are not needed: the chunks use the times' fraction of T_surv at
  //  each quadrature point, quad_time_fraction, and their subjects' X_surv rows. Column k of
  //  X_long_quad_stacked is covariate k at every subject's Q points, i.e. the Q x N_subj matrix of covariate k,
  //  stacked column by column (to_vector); a loop over the covariates and the quadrature points only)
  vector[Q] quad_time_fraction = (quad_points + 1) / 2;
  for (k in 1:K_long) {
    matrix[Q, N_subj] X_long_quad_of_covariate_k;
    for (q in 1:Q) {
      X_long_quad_of_covariate_k[q] = to_row_vector(X_long_quad[:, q, k]);
    }
    X_long_quad_stacked[:, k] = to_vector(X_long_quad_of_covariate_k);
  }
  vector[N_subj] event_indicator = to_vector(event);
  // as in joint_longitudinal_survival.stan: the quadrature weight of each stacked point times half its subject's
  // survival time (the sum of the cumulative hazards is one dot product with the stacked hazards), and the event
  // data summed over the subjects with an event
  // vector[N_quad] quad_weight_times_half_T;
  // for (i in 1:N_subj) {
  //   for (q in 1:Q) {
  //     quad_weight_times_half_T[(i - 1) * Q + q] = quad_weights[q] * T_surv[i] / 2;
  //   }
  // }
  // (the Q x N_subj outer product of quad_weights with T_surv / 2, stacked column by column)
  vector[N_quad] quad_weight_times_half_T = to_vector(quad_weights * (T_surv / 2)');
  int N_events = sum(event);
  vector[N_subj] event_time = event_indicator .* T_surv;
  vector[K_long] X_long_event_summed_over_events = X_long_event' * event_indicator;
  // (the subjects with an event, in increasing order, for the generated quantities: the first N_events indices
  //  of event sorted in decreasing order, then sorted)
  array[N_events] int event_index = sort_asc(sort_indices_desc(event)[1:N_events]);
}


parameters {
  // matrix[2, N_subj] b_raw;          // standard normal nuisance block, declared first
  // (subject i's b_raw[i, 1] and b_raw[i, 2]: the same unconstrained parameters, in the same order, as
  //  matrix[2, N_subj] b_raw, whose column i they were; reduce_sum_static slices them by subject)
  array[N_subj] row_vector[2] b_raw;  // standard normal nuisance block, declared first
  // Longitudinal submodel parameters
  vector[K_long] beta_long;         // fixed effects
  real<lower=0> sigma_long;         // residual error SD
  real<lower=0> sigma_b0;           // random intercept SD
  real<lower=0> sigma_b1;           // random slope SD
  real<lower=-1,upper=1> rho_b;     // correlation between random effects

  // Survival submodel parameters
  vector[K_surv] beta_surv;         // survival covariates effects
  real<lower=0> lambda;             // baseline hazard rate
  real alpha_assoc;                 // association parameter
}


// b0 and b1 are computed inside the chunks (from b_raw), and reported as generated quantities
// transformed parameters {
//   vector[N_subj] b0 = sigma_b0 * b_raw[1]';
//   vector[N_subj] b1 = sigma_b1 * (rho_b * b_raw[1]' + sqrt(1 - square(rho_b)) * b_raw[2]');
// }


model {
  // Priors
  beta_long ~ normal(0, 2);
  beta_surv ~ normal(0, 2);
  sigma_long ~ exponential(1);
  sigma_b0 ~ exponential(1);
  sigma_b1 ~ exponential(1);
  rho_b ~ uniform(-1, 1);
  // Baseline hazard prior on the example's time scale (median 0.2 per time unit).
  // lambda ~ lognormal(log(0.2), 1);
  lambda ~ lognormal(log(0.1), 1.5);
  alpha_assoc ~ normal(0, 1);

  // Equivalent non-centred form of the original correlated Gaussian effects.
  // to_vector(b_raw) ~ std_normal();
  // (computed inside the chunks, std_normal_lupdf of each chunk's columns of b_raw)

  // Longitudinal and survival likelihoods (reduce_sum_static over the subjects, in chunks of chunk_size)
  // target += reduce_sum_static(partial_sum_subjects_lupmf, subject_index, chunk_size,
  // target += reduce_sum_static(partial_sum_subjects_lupmf, subject_index, chunk_size,
  //                             X_long, Y_long, time_long, subj_long, observations_before,
  //                             X_surv, T_surv, event_indicator, quad_weights, X_long_event,
  //                             X_long_quad_stacked, X_surv_stacked, t_quad, subject_of_quad,
  //                             beta_long, sigma_long, b0, b1, beta_surv, lambda, alpha_assoc);
  // the subjects' columns of b_raw, which reduce_sum_static slices (the same autodiff variables as b_raw: no new
  // nodes on the main tape)
  // array[N_subj] row_vector[2] b_raw_by_subject;
  // for (i in 1:N_subj) {
  //   b_raw_by_subject[i] = b_raw[:, i]';
  // }
  // target += reduce_sum_static(partial_sum_subjects_lupdf, b_raw_by_subject, chunk_size,
  //                             Q, X_long, Y_long, time_long, subj_long, observations_before,
  //                             X_surv, event_indicator, event_time,
  //                             X_long_quad_stacked, t_quad, quad_weight_times_half_T,
  //                             beta_long, sigma_long, sigma_b0, sigma_b1, rho_b,
  //                             beta_surv, lambda, alpha_assoc);
  // (the sliced argument is the parameter b_raw itself, array[N_subj] row_vector[2], as u_raw in
  //  LC_MVP_bin_PartialLog_v5_reduce_sum_static.stan: nothing is built on the main tape, and reduce_sum holds a
  //  named variable by reference)
  target += reduce_sum_static(partial_sum_subjects_lupdf, b_raw, chunk_size,
                              Q, X_long, Y_long, time_long, subj_long, observations_before,
                              X_surv, T_surv, event_indicator, event_time,
                              X_long_quad_stacked, quad_time_fraction, quad_weight_times_half_T,
                              beta_long, sigma_long, sigma_b0, sigma_b1, rho_b,
                              beta_surv, lambda, alpha_assoc);
  // the events' log hazards, terms of fixed size: N_events * log(lambda) and the longitudinal fixed effects at
  // the events' times, alpha_assoc * X_long_event[i] * beta_long summed over the events
  target += N_events * log(lambda) + dot_product(X_long_event_summed_over_events, alpha_assoc * beta_long);
}


generated quantities {
  // vector[N_subj] b0 = sigma_b0 * b_raw[1]';
  // vector[N_subj] b1 = sigma_b1 * (rho_b * b_raw[1]' + sqrt(1 - square(rho_b)) * b_raw[2]');
  // (b_raw as an N_subj x 2 matrix, subject i in row i)
  vector[N_subj] b0 = sigma_b0 * to_matrix(b_raw)[:, 1];
  vector[N_subj] b1 = sigma_b1 * (rho_b * to_matrix(b_raw)[:, 1]
                                  + sqrt(1 - square(rho_b)) * to_matrix(b_raw)[:, 2]);
  vector[N_long] eta_long;          // longitudinal linear predictor
  vector[N_subj] hazard_subj;       // subject-specific hazards
  vector[N_long] y_pred;            // posterior predictions
  vector[N_subj] log_lik_surv;      // survival log-likelihood

  // Longitudinal linear predictor
  // for (n in 1:N_long) {
  //   eta_long[n] = X_long[n] * beta_long +
  //                 b0[subj_long[n]] +
  //                 b1[subj_long[n]] * time_long[n];
  // }
  eta_long = X_long * beta_long + b0[subj_long] + b1[subj_long] .* time_long;

  // Subject-specific hazards at event/censoring time
  // for (i in 1:N_subj) {
  //   real long_value = X_long_event[i] * beta_long + b0[i] + b1[i] * T_surv[i];
  //   hazard_subj[i] = lambda * exp(X_surv[i] * beta_surv + alpha_assoc * long_value);
  // }
  hazard_subj = lambda * exp(X_surv * beta_surv + alpha_assoc * (X_long_event * beta_long + b0 + b1 .* T_surv));

  // Posterior predictions for longitudinal data
  // for (n in 1:N_long) {
  //   y_pred[n] = normal_rng(eta_long[n], sigma_long);
  // }
  y_pred = to_vector(normal_rng(eta_long, sigma_long));

  // Log-likelihood for survival data
  // for (i in 1:N_subj) {
  //   real cum_hazard = 0;
  //   for (q in 1:Q) {
  //     real t_q = T_surv[i] * (quad_points[q] + 1) / 2;
  //     real long_value_q = X_long_quad[i][q] * beta_long + b0[i] + b1[i] * t_q;
  //     real hazard_q = lambda * exp(X_surv[i] * beta_surv + alpha_assoc * long_value_q);
  //     cum_hazard += quad_weights[q] * hazard_q;
  //   }
  //   cum_hazard *= T_surv[i] / 2;
  //
  //   if (event[i] == 1) {
  //     log_lik_surv[i] = log(hazard_subj[i]) - cum_hazard;
  //   } else {
  //     log_lik_surv[i] = -cum_hazard;
  //   }
  // }
  // (all subjects at once: the hazards at every subject's Q quadrature points, column i of the Q x N_subj matrix
  //  for subject i, and the cumulative hazards by quadrature; then the log hazards of the subjects with an event)
  {
    vector[N_quad] hazard_quad
          = lambda * exp(to_vector(rep_matrix((X_surv * beta_surv)', Q))
                         + alpha_assoc * (X_long_quad_stacked * beta_long
                                          + to_vector(rep_matrix(b0', Q))
                                          + to_vector(quad_time_fraction * (T_surv .* b1)')));
    vector[N_subj] cum_hazard = (to_matrix(hazard_quad, Q, N_subj)' * quad_weights) .* T_surv / 2;
    log_lik_surv = -cum_hazard;
    log_lik_surv[event_index] = log(hazard_subj[event_index]) - cum_hazard[event_index];
  }
}
