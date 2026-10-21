

// latent_diffusion_survival_chunked_no_loops_over_N_hand_written_BayesMVP_AVX.stan
// Hand-written BayesMVP AVX version of latent_diffusion_survival_chunked_no_loops_over_N.stan:
// every elementwise call on a vector written as BayesMVP's fast_ function (declared in the functions block as
// NicoStan's writer declares them), every scalar call on Stan math; one text for both AVX2 and AVX-512
// (compiled with each backend's flags and user header).
//
// The tape-chunked model latent_diffusion_survival_chunked.stan (below) without a loop over the subjects: its
// log density and its chunks are unchanged (each chunk's terms were already whole-vector operations over the
// chunk's events, and -H(t) was already gathered by grid cell into dot products of length M), and its loops over
// the subjects in transformed data are whole-vector operations: each subject's cell is floor(t[n] / dt) + 1
// (at most M; the same cells as the loop of latent_diffusion_survival.stan on every data case), the events are
// the first N_event subjects in the order of the distinct keys n + (N + 1) (1 - event[n]) (sort_indices_asc), the
// events before each subject are a cumulative sum, and the sums over the subjects of each grid cell are the
// product of the M x N matrix of the indicators of the subjects' cells with the vectors of the subjects' terms.
// In generated quantities, the survival path is exp(-H) at the grid points, with H the cumulative sum of the
// trapezoids (the same values up to rounding).
// The only loop left is the Euler-Maruyama recursion of the path x over the M grid points (transformed
// parameters): its step is not linear in x (sin(x)), so it cannot be written as a linear system or a cumulative
// sum, and its length is M (the same for every N), not the subjects.
//
// latent_diffusion_survival_chunked.stan
// Tape-chunked version of latent_diffusion_survival.stan: the same log density, with the subjects' likelihood
// terms (log h(t) at each observed event time, and -H(t) for every subject) summed over the subjects by
// reduce_sum_static, in chunks of chunk_size subjects (data), so that each chunk's autodiff tape is built,
// differentiated and released before the next one (the tape-chunking paper's chunking; N_chunks from
// NicoStan::fn_compute_initial_N_chunks()). The latent path x_path and the hazard and integrated hazard on the
// grid, which every subject shares, are computed once, outside reduce_sum_static, and passed to every chunk.
// Each chunk computes its subjects' terms with the whole-vector operations of latent_diffusion_survival.stan
// (event_index lists the events in the subjects' order, so a chunk's events are contiguous in it).
// The chunking data are NicoStan's for Stan models: N (the subjects, the units of the chunks) and chunk_size,
// which NicoStan sets to ceiling(N / N_chunks) in each phase (num_chunks_burnin, num_chunks_sampling).
//
// Inside the chunks (everything whose cost grows with the subjects): the path interpolated at the chunk's
// event times and the log hazards there, log(x(t)^2 + x_squared_hazard_offset), one vectorised operation over
// the chunk's events. Outside the chunks, only terms whose size does not grow with N: the path recursion over the
// M + 1 grid points (sequential in the grid, the same for any N), the hazard and integrated hazard on the grid,
// N_event * log(hazard_scale), and -H(t) summed over all subjects, which is gathered by grid cell into dot
// products of length M with data computed in transformed data (as in latent_diffusion_survival.stan). x_path,
// of length M + 1, is the only autodiff argument passed to the chunks.
//
// The model, as in latent_diffusion_survival.stan:
//
// Latent diffusion survival model (Roberts and Sangalli, 2010), in the form used by
// Beskos, Kalogeropoulos and Pazos (2013, Sections 5 and 6.3). All subjects share one
// hazard trajectory h(u) = hazard_scale * (x(u)^2 + x_squared_hazard_offset), driven by the latent diffusion
//     dX_u = -(drift_sin_coefficient * sin(X_u) + drift_constant) du + sigma_x dB_u,  X_0 = x_0,
// on the observation window [0, t_max]. Beskos et al. use hazard_scale = 1, drift
// coefficients (1.4, 1), sigma_x = 1 (unit diffusion coefficient), x_0 = 2 and t_max = 1,
// and treat these as known. Here the drift coefficients and the hazard scale are estimated
// jointly with the path; sigma_x and x_0 are fixed data.
//
// Hazard offset (not in Beskos et al.): Beskos et al. use
// h(x) = x^2 exactly (x_squared_hazard_offset = 0). Then log h at an event time is 2 log|x(t)|, which is -Inf
// at x(t) = 0, and h(x) = h(-x). The drift pulls X towards its stable point -0.795 (the drift-only path from
// x_0 = 2 reaches 0 at t = 1.07, and about 60% of simulated paths cross 0 before t = 1). Once the path gets
// near zero, the posterior splits into sign patterns of x at the event times,
// separated by those -Inf walls. Hamiltonian trajectories cannot cross the walls and diverge near them:
// cmdstanr at adapt_delta 0.9 gave 120 divergences and x_path R-hat 1.11 with the old model, and still 43
// divergences and x_path R-hat 1.53 with every parameter fixed at the Beskos et al. values. A small offset
// (default 0.01 in the simulator, i.e. the hazard never falls below 1% of hazard_scale) makes log h finite
// and smooth, so the sign patterns are connected and the sampler mixes over them. The hazard shape is x^2
// wherever |x| is not tiny. Setting the offset to 0 gives the exact Beskos et al. form (and its divergences).
//
// Discretisation: Euler-Maruyama on M equal steps of length dt = t_max / M, non-centred.
// The nuisance block w holds the M standard normal Brownian increments, and
//     x[k + 1] = x[k] + drift(x[k]) * dt + sigma_x * sqrt(dt) * w[k].
// The path is taken as linear between grid points. The hazard at an event time uses the
// linearly interpolated path value (not a snap to the nearest grid point). The integrated
// hazard is the trapezoid rule over the whole grid cells before t[n], plus a trapezoid over
// the partial cell from its lower grid point to t[n]. Right-censored times contribute only
// -H(t[n]); administrative censoring at t_max is the usual case.

functions {
    real bmvp_simd_lanes();
    vector nico_exp(vector x);
    vector nico_log(vector x);
    real nico_normal_lpdf(vector y, real mu, real sigma);
    // real partial_sum_subjects_lpmf(array[] int subject_slice, int start, int end,
    //                                array[] int cell_index, array[] int cell_index_upper,
    //                                vector time_into_cell, vector cell_fraction,
    //                                array[] int event_index, array[] int events_before,
    //                                real x_squared_hazard_offset, vector x_path, vector hazard_grid,
    //                                vector cumulative_hazard_grid, real hazard_scale) {
    //     int n_subjects = end - start + 1;
    //     real lp = 0;
    //     // the chunk's subjects: the path at t[n] (linearly interpolated) and the hazard there
    //     vector[n_subjects] x_at_t = (1 - cell_fraction[start:end]) .* x_path[cell_index[start:end]]
    //                                 + cell_fraction[start:end] .* x_path[cell_index_upper[start:end]];
    //     vector[n_subjects] hazard_at_t = hazard_scale * (square(x_at_t) + x_squared_hazard_offset);
    //     // the chunk's events: event_index[e1:e2] (the events of subjects start to end)
    //     int e1 = events_before[start] + 1;
    //     int e2 = events_before[end + 1];
    //     // log h(t) at the chunk's observed event times; finite at x(t) = 0 whenever
    //     // x_squared_hazard_offset > 0.
    //     if (e2 >= e1) {
    //       array[e2 - e1 + 1] int event_position_in_chunk;
    //       for (k in 1:(e2 - e1 + 1)) {
    //         event_position_in_chunk[k] = event_index[e1 + k - 1] - start + 1;
    //       }
    //       lp += (e2 - e1 + 1) * log(hazard_scale)
    //             + sum(log(square(x_at_t[event_position_in_chunk]) + x_squared_hazard_offset));
    //     }
    //     // - H(t) for every subject of the chunk, events and censored alike.
    //     lp += -sum(cumulative_hazard_grid[cell_index[start:end]]
    //                + 0.5 * time_into_cell[start:end] .* (hazard_grid[cell_index[start:end]] + hazard_at_t));
    //     return lp;
    // }
    // the chunk's events (event_index[e1:e2], the events of subjects start to end): the path at their times,
    // linearly interpolated, and log(x(t)^2 + x_squared_hazard_offset), i.e. log h(t) minus log(hazard_scale);
    // finite at x(t) = 0 whenever x_squared_hazard_offset > 0
    real partial_sum_subjects_lpmf(array[] int subject_slice, int start, int end,
                                   data array[] int events_before, data array[] int event_cell_index,
                                   data array[] int event_cell_index_upper,
                                   data vector one_minus_event_cell_fraction, data vector event_cell_fraction,
                                   data real x_squared_hazard_offset, vector x_path) {
        real lp = 0;
        int e1 = events_before[start] + 1;
        int e2 = events_before[end + 1];
        if (e2 >= e1) {
          vector[e2 - e1 + 1] x_at_event_time = one_minus_event_cell_fraction[e1:e2]
                                                .* x_path[event_cell_index[e1:e2]]
                                                + event_cell_fraction[e1:e2]
                                                  .* x_path[event_cell_index_upper[e1:e2]];
          lp += sum(nico_log(square(x_at_event_time) + x_squared_hazard_offset));
        }
        return lp;
    }
}


data {
    int<lower=1> N;                          // number of subjects
    int<lower=2> M;                          // number of Euler-Maruyama steps on [0, t_max]
    real<lower=0> t_max;                     // end of the observation window
    real x_0;                                // fixed initial value of the latent diffusion
    real<lower=0> sigma_x;                   // fixed diffusion coefficient (1 = unit diffusion)
    real<lower=0> x_squared_hazard_offset;   // h(u) = hazard_scale * (x(u)^2 + offset);
                                             // 0 = exact Beskos et al. form
    vector<lower=0, upper=t_max>[N] t;       // event or censoring time
    array[N] int<lower=0, upper=1> event;    // 1 = event observed, 0 = right censored at t
    int<lower=1> chunk_size;                 // subjects per chunk (reduce_sum_static)
}


transformed data {
    real dt = t_max / M;
    real sqrt_dt = sqrt(dt);
    int N_event = sum(event);
    array[N_event] int event_index;          // subjects with an observed event
    array[N] int cell_index;                 // lower grid point of the cell containing t[n]
    array[N] int cell_index_upper;           // upper grid point of that cell
    vector[N] time_into_cell;                // t[n] minus the time of that lower grid point
    vector[N] cell_fraction;                 // time_into_cell / dt, in [0, 1]
    // {
    //   int position = 1;
    //   for (n in 1:N) {
    //     int cell = 1;
    //     if (t[n] <= 0) reject("Event and censoring times must be strictly positive.");
    //     // Grid point k (1-based) sits at time (k - 1) * dt; the last cell is [(M - 1) dt, t_max].
    //     while (cell < M && t[n] >= cell * dt) cell += 1;
    //     cell_index[n] = cell;
    //     cell_index_upper[n] = cell + 1;
    //     time_into_cell[n] = t[n] - (cell - 1) * dt;
    //     cell_fraction[n] = fmin(1, time_into_cell[n] / dt);
    //     if (event[n] == 1) {
    //       event_index[position] = n;
    //       position += 1;
    //     }
    //   }
    // }
    // (the same, without a loop over the subjects)
    if (min(t) <= 0) reject("Event and censoring times must be strictly positive.");
    {
      // Grid point k (1-based) sits at time (k - 1) * dt; the last cell is [(M - 1) dt, t_max]: the cell of t[n]
      // is the largest k <= M with (k - 1) dt <= t[n], i.e. floor(t[n] / dt) + 1, at most M
      vector[N] cell_as_real = fmin(M, floor(t / dt) + 1);
      cell_index = to_int(to_array_1d(cell_as_real));
      cell_index_upper = to_int(to_array_1d(cell_as_real + 1));
      time_into_cell = t - (cell_as_real - 1) * dt;
      cell_fraction = fmin(1, time_into_cell / dt);
      // the subjects with an observed event, in the subjects' order: the first N_event subjects sorted by the
      // distinct keys n + (N + 1) (1 - event[n]), which put every event (key n) before every censored subject
      // (key n + N + 1)
      // event_index = sort_indices_asc(linspaced_vector(N, 1, N) + (N + 1) * (1 - to_vector(event)))[1:N_event];
      // (the keys as a vector of their own: sort_indices_asc() of an unnamed vector expression sorts by a copy
      //  of it that Stan math frees before the sort starts, so that the sort reads freed memory and event_index
      //  can be wrong)
      vector[N] event_first_keys = linspaced_vector(N, 1, N) + (N + 1) * (1 - to_vector(event));
      event_index = sort_indices_asc(event_first_keys)[1:N_event];
    }
    // the subjects, which reduce_sum_static slices into chunks, and the events of subjects 1, ..., i - 1, which
    // are the first events_before[i] entries of event_index (event_index lists the events in the subjects' order)
    array[N] int subject_index = linspaced_int_array(N, 1, N);
    // array[N + 1] int events_before = rep_array(0, N + 1);
    // for (n in 1:N) {
    //   events_before[n + 1] = events_before[n] + event[n];
    // }
    // (the same, without a loop over the subjects)
    array[N + 1] int events_before = append_array({0}, cumulative_sum(event));
    // the subjects' terms of -H(t[n]) gathered by grid cell (cell k = [(k - 1) dt, k dt], lower grid point k),
    // so that their sum over the subjects is a few dot products with vectors of length M: the subjects in cell k,
    // half their summed time_into_cell, and the weights of x_path[k]^2, x_path[k] * x_path[k + 1] and
    // x_path[k + 1]^2 in half their summed time_into_cell * x(t[n])^2, where
    // x(t[n]) = (1 - cell_fraction[n]) * x_path[k] + cell_fraction[n] * x_path[k + 1]
    vector[M] subjects_in_cell = rep_vector(0, M);
    vector[M] half_time_into_cell_summed_by_cell = rep_vector(0, M);
    vector[M] weight_of_x_lower_squared_by_cell = rep_vector(0, M);
    vector[M] weight_of_x_lower_times_x_upper_by_cell = rep_vector(0, M);
    vector[M] weight_of_x_upper_squared_by_cell = rep_vector(0, M);
    // for (n in 1:N) {
    //   int k = cell_index[n];
    //   subjects_in_cell[k] += 1;
    //   half_time_into_cell_summed_by_cell[k] += 0.5 * time_into_cell[n];
    //   weight_of_x_lower_squared_by_cell[k] += 0.5 * time_into_cell[n] * square(1 - cell_fraction[n]);
    //   weight_of_x_lower_times_x_upper_by_cell[k] += time_into_cell[n] * (1 - cell_fraction[n])
    //                                                 * cell_fraction[n];
    //   weight_of_x_upper_squared_by_cell[k] += 0.5 * time_into_cell[n] * square(cell_fraction[n]);
    // }
    // (the same sums, without a loop over the subjects: the M x N matrix of the indicators of the subjects'
    //  cells, 1 where cell_index[n] = k and 0 elsewhere, i.e. 1 - |cell_index[n] - k| floored at 0, times the
    //  vectors of the subjects' terms; a local of this block, so that it is not kept after transformed data)
    {
      matrix[M, N] subject_is_in_cell = fmax(0, 1 - abs(rep_matrix(to_row_vector(cell_index), M)
                                                        - rep_matrix(linspaced_vector(M, 1, M), N)));
      subjects_in_cell = subject_is_in_cell * rep_vector(1, N);
      half_time_into_cell_summed_by_cell = subject_is_in_cell * (0.5 * time_into_cell);
      weight_of_x_lower_squared_by_cell = subject_is_in_cell * (0.5 * time_into_cell
                                                                .* square(1 - cell_fraction));
      weight_of_x_lower_times_x_upper_by_cell = subject_is_in_cell * (time_into_cell .* (1 - cell_fraction)
                                                                      .* cell_fraction);
      weight_of_x_upper_squared_by_cell = subject_is_in_cell * (0.5 * time_into_cell
                                                                .* square(cell_fraction));
    }
    real half_time_into_cell_summed_over_subjects = sum(half_time_into_cell_summed_by_cell);
    // the events' cells and cell fractions (the path at an event time is interpolated within its cell)
    array[N_event] int event_cell_index = cell_index[event_index];
    array[N_event] int event_cell_index_upper = cell_index_upper[event_index];
    vector[N_event] event_cell_fraction = cell_fraction[event_index];
    vector[N_event] one_minus_event_cell_fraction = 1 - event_cell_fraction;
}


parameters {
    vector[M] w;                             // standard normal Brownian increments, declared first
    real drift_sin_coefficient;              // drift: -(drift_sin_coefficient * sin(x) + drift_constant)
    real drift_constant;
    real<lower=0> hazard_scale;              // h(u) = hazard_scale * (x(u)^2 + x_squared_hazard_offset)
}


transformed parameters {
    vector[M + 1] x_path;                    // latent diffusion at the M + 1 grid points
    // x_path[1] = x_0;
    // for (k in 1:M) {
    //   x_path[k + 1] = x_path[k] - (drift_sin_coefficient * sin(x_path[k]) + drift_constant) * dt
    //                   + sigma_x * sqrt_dt * w[k];
    // }
    // The same recursion, with the terms that do not depend on the path computed once, as one vector:
    // sigma_x * sqrt(dt) * w[k] - drift_constant * dt. Only the sin(x_path[k]) term is sequential over the grid
    // (M steps, the same for any N).
    {
      vector[M] increment_without_sin_term = (sigma_x * sqrt_dt) * w - drift_constant * dt;
      real drift_sin_coefficient_times_dt = drift_sin_coefficient * dt;
      x_path[1] = x_0;
      for (k in 1:M) {
        x_path[k + 1] = x_path[k] - drift_sin_coefficient_times_dt * sin(x_path[k])
                        + increment_without_sin_term[k];
      }
    }
}


model {
    // priors
    // Drift priors: Beskos et al. treat the drift as known,
    // (1.4, 1). One path on [0, 1] carries little information about the drift (with the earlier normal(0, 2)
    // priors the posterior SDs were about 1.5, close to the prior), so the prior sets how far the sampler goes
    // along the drift_sin_coefficient / drift_constant ridge (posterior correlation about -0.7). With
    // normal(0, 2) and the hazard offset, cmdstanr (adapt_delta 0.8, N = 800) still gave 49 divergences, 30 of
    // them from the 5% of draws with drift_sin_coefficient <= -1.5 (drift_constant about 3 there), i.e. the far
    // tail of that ridge, where the Euler recursion is expanding. These priors relax "known drift" to
    // "drift known to within about 1"; with them that run had 0 divergences.
    // The hazard-scale prior (median 1) is unchanged.
    drift_sin_coefficient ~ normal(1.4, 1);
    drift_constant ~ normal(1, 1);
    hazard_scale ~ lognormal(0, 1);
    w ~ nico_normal(0, 1);

    // // likelihood: the hazard and the integrated hazard on the grid, which every subject shares, then the
    // // subjects' terms (reduce_sum_static over the subjects, in chunks of chunk_size)
    // {
    //   vector[M + 1] hazard_grid = hazard_scale * (square(x_path) + x_squared_hazard_offset);
    //   vector[M + 1] cumulative_hazard_grid;
    //   cumulative_hazard_grid[1] = 0;
    //   cumulative_hazard_grid[2:(M + 1)] = cumulative_sum(0.5 * dt * (hazard_grid[1:M]
    //                                                                  + hazard_grid[2:(M + 1)]));
    //   target += reduce_sum_static(partial_sum_subjects_lupmf, subject_index, chunk_size,
    //                               cell_index, cell_index_upper, time_into_cell, cell_fraction,
    //                               event_index, events_before, x_squared_hazard_offset,
    //                               x_path, hazard_grid, cumulative_hazard_grid, hazard_scale);
    // }
    // likelihood: the hazard and the integrated hazard on the grid, which every subject shares; log h(t) at the
    // observed event times, N_event * log(hazard_scale) and the subjects' terms log(x(t)^2 + offset)
    // (reduce_sum_static over the subjects, in chunks of chunk_size); then -H(t) summed over the subjects by grid
    // cell, as in latent_diffusion_survival.stan
    {
      vector[M + 1] hazard_grid = hazard_scale * (square(x_path) + x_squared_hazard_offset);
      vector[M + 1] cumulative_hazard_grid;
      cumulative_hazard_grid[1] = 0;
      cumulative_hazard_grid[2:(M + 1)] = cumulative_sum(0.5 * dt * (hazard_grid[1:M] + hazard_grid[2:(M + 1)]));
      target += N_event * log(hazard_scale);
      target += reduce_sum_static(partial_sum_subjects_lupmf, subject_index, chunk_size,
                                  events_before, event_cell_index, event_cell_index_upper,
                                  one_minus_event_cell_fraction, event_cell_fraction, x_squared_hazard_offset,
                                  x_path);
      // - H(t) for every subject, events and censored alike, summed by grid cell: the subjects' integrated
      // hazards up to their cell's lower grid point, their trapezoids' lower sides (hazard_grid[k]) and upper
      // sides (the hazard at t[n], hazard_scale * (x(t[n])^2 + x_squared_hazard_offset))
      target += -(dot_product(subjects_in_cell, cumulative_hazard_grid[1:M])
                  + dot_product(half_time_into_cell_summed_by_cell, hazard_grid[1:M])
                  + hazard_scale * (dot_product(weight_of_x_lower_squared_by_cell, square(x_path[1:M]))
                                    + dot_product(weight_of_x_lower_times_x_upper_by_cell,
                                                  x_path[1:M] .* x_path[2:(M + 1)])
                                    + dot_product(weight_of_x_upper_squared_by_cell,
                                                  square(x_path[2:(M + 1)]))
                                    + x_squared_hazard_offset * half_time_into_cell_summed_over_subjects));
    }
}


generated quantities {
    real simd_lanes = bmvp_simd_lanes();
    vector[M + 1] hazard_path = hazard_scale * (square(x_path) + x_squared_hazard_offset);
    vector[M + 1] survival_path;             // S(u) = exp(-H(u)) at the grid points
    // survival_path[1] = 1;
    // for (k in 1:M) {
    //   survival_path[k + 1] = survival_path[k] * exp(-0.5 * dt * (hazard_path[k] + hazard_path[k + 1]));
    // }
    // (the same values up to rounding, without a loop: exp(-H) with H the cumulative sum of the trapezoids)
    survival_path = nico_exp(-append_row(0, cumulative_sum(0.5 * dt * (hazard_path[1:M]
                                                                       + hazard_path[2:(M + 1)]))));
}






















