// joint_longitudinal_survival_unchunked_no_loops_over_N_with_Stan_fused_functions.stan
// Unchunked Stan::math version of the joint longitudinal-survival model of the chunking paper's layout timings
// (joint_longitudinal_survival_chunked_no_loops_over_N_v5_automatic_reduce_sum_chunked, from its source
// joint_longitudinal_survival_chunked_no_loops_over_N_v5): the same log density as that source at N_chunks = 1,
// and as joint_longitudinal_survival.stan, as one pass over all N_subj subjects, without reduce_sum_static. The
// same data (N and chunk_size, when they are in the data, are not read; subj_long need not be sorted, as in
// joint_longitudinal_survival.stan) and the same unconstrained parameters, in the same order and with the same
// names, as joint_longitudinal_survival.stan: b_raw is its 2 x N_subj matrix, whose unconstrained coordinates
// are in the same order as those of the source's array[N_subj] row_vector[2] b_raw (subject i's 2 values, then
// subject i + 1's), which exists only so that reduce_sum_static can slice the subjects.
// Stan's fused functions:
//   - the longitudinal likelihood is Stan's fused normal GLM, Y_long ~ normal_id_glm(X_long, the random effects
//     b0[subj] + b1[subj] time as the intercept of each observation, beta_long, sigma_long), as in the source;
//   - the cumulative hazards, -lambda sum over the subjects and their Q quadrature points of the weight
//     quad_weights[q] T_surv[i] / 2 times exp(eta), are Stan's fused Poisson-log GLM with every outcome 0:
//     poisson_log_glm_lpmf(0 | theta) = -exp(theta) (lgamma(0 + 1) = 0), with
//     theta = log(lambda) + log(weight) + eta, the covariates X_long_quad_stacked and the coefficients
//     alpha_assoc beta_long, i.e. -sum(lambda weight exp(eta)), the same term; the log of the weights is data,
//     made once in transformed data (the quadrature weights and the survival times must hence be positive);
//   - the events' log hazards, as in the source: the data summed over the events (X_long_event summed, the
//     number of events, the event times) times the parameters, a few dot products.
// No loop over N anywhere (the loops left in transformed data are over the K_long covariates and the Q
// quadrature points).


data {
  int<lower=1> N_long;              // number of longitudinal observations
  int<lower=1> N_subj;              // number of subjects
  int<lower=1> K_long;              // number of longitudinal covariates
  matrix[N_long, K_long] X_long;    // longitudinal covariate matrix
  vector[N_long] Y_long;            // longitudinal outcomes
  array[N_long] int<lower=1,upper=N_subj> subj_long; // subject IDs for longitudinal data
  vector[N_long] time_long;         // measurement times

  int<lower=1> K_surv;              // number of survival covariates
  matrix[N_subj, K_surv] X_surv;    // survival covariate matrix
  vector<lower=0>[N_subj] T_surv;   // survival times
  array[N_subj] int<lower=0,upper=1> event; // event indicators

  int<lower=1> Q;                   // number of quadrature points
  vector[Q] quad_points;            // quadrature points
  vector[Q] quad_weights;           // quadrature weights
  matrix[N_subj, K_long] X_long_event; // longitudinal fixed-effect design at T_surv
  array[N_subj] matrix[Q, K_long] X_long_quad; // same design at each subject's quadrature times
}


transformed data {
  // the log of the quadrature weights below needs positive weights and survival times
  if (min(quad_weights) <= 0 || min(T_surv) <= 0) {
    reject("the quadrature weights and the survival times must be positive (the cumulative hazards are a ",
           "Poisson-log GLM with the log of the weights as data)");
  }
  // every subject's Q quadrature points, stacked (subject i in rows (i - 1) * Q + 1 to i * Q), as in the source:
  // column k of X_long_quad_stacked is covariate k at every subject's Q points, i.e. the Q x N_subj matrix of
  // covariate k, stacked column by column (to_vector); a loop over the covariates and the quadrature points only
  int N_quad = N_subj * Q;
  matrix[N_quad, K_long] X_long_quad_stacked;
  vector[Q] quad_time_fraction = (quad_points + 1) / 2;
  for (k in 1:K_long) {
    matrix[Q, N_subj] X_long_quad_of_covariate_k;
    for (q in 1:Q) {
      X_long_quad_of_covariate_k[q] = to_row_vector(X_long_quad[:, q, k]);
    }
    X_long_quad_stacked[:, k] = to_vector(X_long_quad_of_covariate_k);
  }
  // the log of the quadrature weight of each stacked point times half its subject's survival time (the Q x N_subj
  // outer product of quad_weights with T_surv / 2, stacked column by column), and the outcomes 0 of the GLM
  vector[N_quad] log_quad_weight_times_half_T = log(to_vector(quad_weights * (T_surv / 2)'));
  array[N_quad] int zeros_of_quad = rep_array(0, N_quad);
  // the event data summed over the subjects with an event, as in the source
  vector[N_subj] event_indicator = to_vector(event);
  int N_events = sum(event);
  vector[N_subj] event_time = event_indicator .* T_surv;
  vector[K_long] X_long_event_summed_over_events = X_long_event' * event_indicator;
  // (the subjects with an event, in increasing order, for the generated quantities: the first N_events indices
  //  of event sorted in decreasing order, then sorted)
  array[N_events] int event_index = sort_asc(sort_indices_desc(event)[1:N_events]);
}


parameters {
  matrix[2, N_subj] b_raw;          // standard normal nuisance block, declared first
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
  to_vector(b_raw) ~ std_normal();

  {
    vector[N_subj] b0 = sigma_b0 * b_raw[1]';
    vector[N_subj] b1 = sigma_b1 * (rho_b * b_raw[1]' + sqrt(1 - square(rho_b)) * b_raw[2]');

    // Longitudinal likelihood (Stan's fused normal GLM, the random effects as each observation's intercept)
    Y_long ~ normal_id_glm(X_long, b0[subj_long] + b1[subj_long] .* time_long, beta_long, sigma_long);

    // Survival likelihood: each subject's log hazard at time u is
    // log(lambda) + subject_part + X_long(u) alpha_assoc beta_long + u subject_slope
    vector[N_subj] subject_part = X_surv * beta_surv + alpha_assoc * b0;
    vector[N_subj] subject_slope = alpha_assoc * b1;
    // the events' log hazards (the data summed over the events times the parameters)
    target += N_events * log(lambda) + dot_product(X_long_event_summed_over_events, alpha_assoc * beta_long)
              + dot_product(event_indicator, subject_part) + dot_product(event_time, subject_slope);
    // minus the cumulative hazards by quadrature, all subjects at once (Stan's fused Poisson-log GLM; see the
    // header; subject i's Q points are column i of the Q x N_subj matrix, stacked)
    target += poisson_log_glm_lupmf(zeros_of_quad | X_long_quad_stacked,
                                    log(lambda) + log_quad_weight_times_half_T
                                    + to_vector(rep_matrix(subject_part', Q)
                                                + quad_time_fraction * (T_surv .* subject_slope)'),
                                    alpha_assoc * beta_long);
  }
}


generated quantities {
  vector[N_subj] b0 = sigma_b0 * b_raw[1]';
  vector[N_subj] b1 = sigma_b1 * (rho_b * b_raw[1]' + sqrt(1 - square(rho_b)) * b_raw[2]');
  vector[N_long] eta_long;          // longitudinal linear predictor
  vector[N_subj] hazard_subj;       // subject-specific hazards
  vector[N_long] y_pred;            // posterior predictions
  vector[N_subj] log_lik_surv;      // survival log-likelihood

  // (on the whole vectors, as in the source)
  eta_long = X_long * beta_long + b0[subj_long] + b1[subj_long] .* time_long;
  hazard_subj = lambda * exp(X_surv * beta_surv + alpha_assoc * (X_long_event * beta_long + b0 + b1 .* T_surv));
  y_pred = to_vector(normal_rng(eta_long, sigma_long));
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






















