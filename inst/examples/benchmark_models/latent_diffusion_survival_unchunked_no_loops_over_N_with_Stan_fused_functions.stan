// latent_diffusion_survival_unchunked_no_loops_over_N_with_Stan_fused_functions.stan
// Proportional hazards model of survival after diagnosis (e.g. a cancer registry) whose log baseline hazard is a
// latent Ornstein-Uhlenbeck diffusion, with covariates. For subject n at time t (years):
//
//     log h_n(t) = x(t) + covariates[n] * beta,
//     dx(t) = kappa (mu_x - x(t)) dt + diffusion_scale dW(t),   x(0) = x_0,
//
// i.e. a log baseline hazard that varies smoothly around its long-run level mu_x and returns to it with rate
// kappa per year (as kappa -> 0, the random walk used for the Cox baseline hazard in practice, e.g. INLA). On the
// M-step Euler grid of [0, t_max] (dt = t_max / M): x at the M + 1 grid points from the standard normal
// innovations w (non-centred; w is the nuisance block, declared first), and subject n's hazard linear in t within
// each grid cell, equal to exp(x_k + covariates[n] * beta) at the grid points. With g_k = exp(x_k) the baseline
// hazard at grid point k and C_k its cumulative (trapezoid) up to grid point k, for t[n] in cell k (lower grid
// point k, fraction f_n of the cell, time tau_n into it):
//
//     h_n(t[n]) = exp(covariates[n] * beta) ((1 - f_n) g_k + f_n g_(k+1)),
//     H_n(t[n]) = exp(covariates[n] * beta) (C_k + a_n g_k + b_n g_(k+1)),
//     a_n = tau_n (2 - f_n) / 2,   b_n = tau_n f_n / 2,
//
// so that the log likelihood is sum over the events of (covariates[n] * beta + log h_0(t[n])) minus
// sum over all subjects of exp(covariates[n] * beta) times the baseline cumulative hazard at t[n]: whole-vector
// operations on the subjects (one exp per subject, one log per event), the data weights a_n and b_n and the cells
// made once in transformed data. (Stan's poisson_log_glm would give the same value with log(H_0(t[n])) as its
// intercepts, at the cost of one more log per subject, so the direct form is used.)
// The Stan::math unchunked model, the reference of every other version: one pass over all N
// subjects. The one loop
// left is the Euler-Maruyama recursion of the path over the M grid steps (M = 50, the same for any N).
// No loop over N anywhere.


data {
    int<lower=1> N;                          // number of subjects
    int<lower=1> K;                          // number of covariates (standardised)
    int<lower=2> M;                          // number of Euler-Maruyama steps on [0, t_max]
    real<lower=0> t_max;                     // end of follow-up (years)
    matrix[N, K] covariates;                 // the subjects' covariates
    vector<lower=0, upper=t_max>[N] t;       // event or censoring time (years after diagnosis)
    array[N] int<lower=0, upper=1> event;    // 1 = event observed, 0 = right censored at t
}


transformed data {
    real dt = t_max / M;
    real sqrt_dt = sqrt(dt);
    int N_event = sum(event);
    vector[N] event_as_vector = to_vector(event);
    array[N_event] int event_index;          // subjects with an observed event
    array[N] int cell_index;                 // lower grid point of the cell containing t[n]
    array[N] int cell_index_upper;           // upper grid point of that cell
    vector[N] weight_of_lower_grid_point;    // a_n: the weight of g_k in the baseline cumulative hazard at t[n]
    vector[N] weight_of_upper_grid_point;    // b_n: the weight of g_(k+1)
    vector[N] cell_fraction;                 // f_n: the fraction of the cell at t[n], in [0, 1]
    if (min(t) <= 0) reject("Event and censoring times must be strictly positive.");
    {
      // grid point k (1-based) sits at time (k - 1) dt; the cell of t[n] is the largest k <= M with
      // (k - 1) dt <= t[n], i.e. floor(t[n] / dt) + 1, at most M
      vector[N] cell_as_real = fmin(M, floor(t / dt) + 1);
      vector[N] time_into_cell = t - (cell_as_real - 1) * dt;
      cell_index = to_int(to_array_1d(cell_as_real));
      cell_index_upper = to_int(to_array_1d(cell_as_real + 1));
      cell_fraction = fmin(1, time_into_cell / dt);
      weight_of_lower_grid_point = 0.5 * time_into_cell .* (2 - cell_fraction);
      weight_of_upper_grid_point = 0.5 * time_into_cell .* cell_fraction;
      // the subjects with an observed event, in the subjects' order: the first N_event subjects sorted by the
      // distinct keys n + (N + 1) (1 - event[n]), which put every event (key n) before every censored subject
      vector[N] event_first_keys = linspaced_vector(N, 1, N) + (N + 1) * (1 - event_as_vector);
      event_index = sort_indices_asc(event_first_keys)[1:N_event];
    }
    // the events' cells and cell fractions (the baseline hazard at an event time, linear within its cell)
    array[N_event] int event_cell_index = cell_index[event_index];
    array[N_event] int event_cell_index_upper = cell_index_upper[event_index];
    vector[N_event] event_cell_fraction = cell_fraction[event_index];
    vector[N_event] one_minus_event_cell_fraction = 1 - event_cell_fraction;
}


parameters {
    vector[M] w;                             // standard normal innovations of the path, declared first
    real x_0;                                // log baseline hazard at diagnosis (per year)
    real mu_x;                               // long-run log baseline hazard (per year)
    real<lower=0> kappa;                     // rate of return to mu_x (per year)
    real<lower=0> diffusion_scale;           // diffusion of the log baseline hazard (per square root of a year)
    vector[K] beta;                          // log hazard ratios of the covariates
}


transformed parameters {
    vector[M + 1] x_path;                    // log baseline hazard at the M + 1 grid points
    // (the Euler-Maruyama recursion x_(k+1) = x_k + kappa (mu_x - x_k) dt + diffusion_scale sqrt(dt) w_k, with the
    //  terms that do not depend on the path computed once, as one vector; M steps, the same for any N)
    {
      vector[M] increment_without_the_path = (diffusion_scale * sqrt_dt) * w + kappa * mu_x * dt;
      real one_minus_kappa_dt = 1 - kappa * dt;
      x_path[1] = x_0;
      for (k in 1:M) {
        x_path[k + 1] = one_minus_kappa_dt * x_path[k] + increment_without_the_path[k];
      }
    }
}


model {
    // priors: weakly informative on the scales of survival after a cancer diagnosis (hazards per year)
    x_0 ~ normal(log(0.2), 1.5);
    mu_x ~ normal(log(0.1), 1.5);
    kappa ~ lognormal(log(0.5), 1);
    diffusion_scale ~ normal(0, 1);
    beta ~ normal(0, 1);
    w ~ std_normal();

    // likelihood: the baseline hazard and its cumulative on the grid, which every subject shares; then, on the
    // whole vectors of subjects and events, the events' log hazards and every subject's cumulative hazard
    {
      vector[M + 1] baseline_hazard_grid = exp(x_path);
      vector[M + 1] cumulative_baseline_hazard_grid;
      cumulative_baseline_hazard_grid[1] = 0;
      cumulative_baseline_hazard_grid[2:(M + 1)] = cumulative_sum(0.5 * dt * (baseline_hazard_grid[1:M]
                                                                              + baseline_hazard_grid[2:(M + 1)]));
      vector[N] linear_predictor = covariates * beta;
      vector[N] cumulative_baseline_hazard_at_t = cumulative_baseline_hazard_grid[cell_index]
                                                  + weight_of_lower_grid_point .* baseline_hazard_grid[cell_index]
                                                  + weight_of_upper_grid_point
                                                    .* baseline_hazard_grid[cell_index_upper];
      target += dot_product(event_as_vector, linear_predictor);
      if (N_event > 0) {
        target += sum(log(one_minus_event_cell_fraction .* baseline_hazard_grid[event_cell_index]
                          + event_cell_fraction .* baseline_hazard_grid[event_cell_index_upper]));
      }
      target += -dot_product(exp(linear_predictor), cumulative_baseline_hazard_at_t);
    }
}


generated quantities {
    vector[M + 1] baseline_hazard_path = exp(x_path);
    vector[M + 1] baseline_survival_path;    // S_0(u) = exp(-H_0(u)) at the grid points (covariates at 0)
    baseline_survival_path = exp(-append_row(0, cumulative_sum(0.5 * dt * (baseline_hazard_path[1:M]
                                                                            + baseline_hazard_path[2:(M + 1)]))));
}






















