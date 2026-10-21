// shared_frailty_Cox_survival_unchunked_no_loops_over_N_with_Stan_fused_functions.stan
// Unchunked plain Stan version of shared_frailty_Cox_survival_direct_sliced_shared_offsets_one_fused_node_with_
// indexed_frailties_when_unchunked (the Stan math file of Paper 1's layout timings): the same log density as that
// file at N_chunks = 1 (one block, no map_rect job), and as shared_frailty_Cox_survival.stan, as one pass over
// all N observations, without reduce_sum_static or map_rect. The same data (chunk_size, when it is in the data,
// is not read) and the same unconstrained parameters, in the same order and with the same names
// (log_frailty_raw, beta, sigma_frailty), as both files; log_frailty_raw is a vector, as in
// shared_frailty_Cox_survival.stan (the chunked file's array[J] real, the same J values in the same order, exists
// only so that reduce_sum_static can slice it).
// The model, as in shared_frailty_Cox_survival.stan: the Cox partial likelihood with shared log-normal frailties
// (Breslow's handling of ties), eta = X beta + sigma_frailty log_frailty_raw[group]. With the observations sorted
// by decreasing time once (transformed data), each event's denominator is the log of the cumulative sum of
// exp(eta) up to the last position of its risk set (every j with t[j] >= t[n], ties included), so that
//     log partial likelihood = sum(event .* eta) - dot_product(w, log_cumulative_sum_exp(eta)),
// with w[k] the number of events whose risk set ends at position k (made once in transformed data, with no loop).
// Stan has no fused function for the log of a cumulative sum of exponentials: it is written with Stan's whole-
// vector exp, cumulative_sum and log, the shift outside the log (as in the chunked file; its rare underflow case
// recomputes only the leading positions). eta takes one fused multiply-add over the observations (fma), as in the
// chunked file.
// log_frailty and frailty, transformed parameters of shared_frailty_Cox_survival.stan, are reported as generated
// quantities (the same values).
// No loop over N anywhere.


functions {
    // the log of the cumulative sum of exp(x) after exp(log_offset), at every position k = 1, ..., rows(x):
    // log(exp(log_offset) + sum(exp(x[1:k]))), vectorised over the positions:
    // shift + log(cumulative_sum(exp(x - shift))), with shift the largest of x and log_offset (so that no term
    // overflows) and exp(log_offset - shift) added at the first position (log_offset = negative_infinity(): no
    // offset). The cumulative sums never decrease; if the first one is below 1e-280 (x[1] and log_offset more
    // than about 640 below the largest of x, far outside the posterior), the leading positions whose cumulative
    // sum is below 1e-280 (their number: the rank of 1e-280 among the cumulative sums) are computed again from
    // their own values by this function (with their own, smaller shift), and the others as above (the terms lost
    // to underflow are below 1e-39 of their sums); the number of leading positions computed again falls at each
    // call, so that the calls stop. The values are those of the scalar log_sum_exp recursion up to
    // floating-point rounding. (The function of the chunked file, unchanged.)
    vector log_cumulative_sum_exp(vector x, real log_offset) {
        int M = rows(x);
        real shift = fmax(max(x), log_offset);
        vector[M] terms_shifted = exp(x - shift);
        terms_shifted[1] += exp(log_offset - shift);
        vector[M] cumulative_sum_shifted = cumulative_sum(terms_shifted);
        if (cumulative_sum_shifted[1] >= 1e-280) {
            return shift + log(cumulative_sum_shifted);
        }
        int M_small = rank(append_row(cumulative_sum_shifted, 1e-280), M + 1);
        vector[M] log_cumulative_sum;
        log_cumulative_sum[1:M_small] = log_cumulative_sum_exp(x[1:M_small], log_offset);
        if (M_small < M) {
            log_cumulative_sum[(M_small + 1):M] = shift + log(cumulative_sum_shifted[(M_small + 1):M]);
        }
        return log_cumulative_sum;
    }
    // dot_product(w, log_cumulative_sum_exp(x, log_offset)), written with the shift outside the log:
    // shift * sum(w) + dot_product(w, log(cumulative_sum_shifted)) (as log_cumulative_sum_exp otherwise, and
    // through it in its rare case) (the function of the chunked file, unchanged)
    real weighted_sum_of_log_cumulative_sum_exp(data vector w, vector x, real log_offset) {
        int M = rows(x);
        real shift = fmax(max(x), log_offset);
        vector[M] terms_shifted = exp(x - shift);
        terms_shifted[1] += exp(log_offset - shift);
        vector[M] cumulative_sum_shifted = cumulative_sum(terms_shifted);
        if (cumulative_sum_shifted[1] >= 1e-280) {
            return shift * sum(w) + dot_product(w, log(cumulative_sum_shifted));
        }
        return dot_product(w, log_cumulative_sum_exp(x, log_offset));
    }
}


data {
    int<lower=0> N;                    // number of observations
    int<lower=0> K;                    // number of covariates
    int<lower=0> J;                    // number of groups/clusters
    matrix[N, K] X;                    // covariate matrix
    vector[N] t;                       // survival times
    array[N] int<lower=0,upper=1> event;     // event indicator
    array[N] int<lower=1,upper=J> group;     // group assignment
    array[N] int<lower=0> at_risk_count;     // number at risk at each event time
}


transformed data {
    // the observations in decreasing order of time, and the size of each observation's risk set (the number of
    // observations j with t[j] >= t[n], ties included: its position in that order, moved to the last of its ties)
    array[N] int order_decreasing_time = sort_indices_desc(t);
    array[N] int risk_set_size;
    // the observations in decreasing order of time (their rows of X, groups and event indicators), and
    // w_decreasing_time[k], the number of events whose risk set ends at position k (each event's denominator is
    // the log of the sum of exp(eta) at position risk_set_size[n]), as in the chunked file
    matrix[N, K] X_decreasing_time = X[order_decreasing_time];
    array[N] int group_decreasing_time = group[order_decreasing_time];
    vector[N] event_decreasing_time = to_vector(event[order_decreasing_time]);
    vector[N] t_decreasing_time = t[order_decreasing_time];
    vector[N] w_decreasing_time;
    // (without ties, i.e. the times strictly decreasing in that order: each observation's risk set ends at its
    //  own position, and w is the event indicators)
    if (N < 2 || min(t_decreasing_time[1:(N - 1)] - t_decreasing_time[2:N]) > 0) {
      risk_set_size[order_decreasing_time] = linspaced_int_array(N, 1, N);
      w_decreasing_time = event_decreasing_time;
    } else {
      // (with ties, without a loop: the last positions of the tie groups are the positions after which the
      //  sorted times decrease (a positive difference to the next time: rank() counts them, sort_indices_desc()
      //  of the differences puts them first and sort_asc() puts them in order), and position N; each position's
      //  tie group is the cumulative sum of the groups' starts; and each group's events, taken at its last
      //  position, are the differences of the cumulative sum of the events at the groups' last positions)
      vector[N - 1] decrease_to_next_time = t_decreasing_time[1:(N - 1)] - t_decreasing_time[2:N];
      int N_positions_before_a_decrease = rank(append_row(-decrease_to_next_time, 0), N);
      int N_tie_groups = N_positions_before_a_decrease + 1;
      array[N_tie_groups] int last_position_of_tie_group;
      last_position_of_tie_group[1:N_positions_before_a_decrease]
        = sort_asc(sort_indices_desc(decrease_to_next_time)[1:N_positions_before_a_decrease]);
      last_position_of_tie_group[N_tie_groups] = N;
      array[N_tie_groups] int first_position_of_tie_group
        = append_array({1}, to_int(to_array_1d(
                                to_vector(last_position_of_tie_group[1:N_positions_before_a_decrease]) + 1)));
      array[N] int start_of_tie_group_at_position = rep_array(0, N);
      start_of_tie_group_at_position[first_position_of_tie_group] = rep_array(1, N_tie_groups);
      array[N] int tie_group_of_position = cumulative_sum(start_of_tie_group_at_position);
      risk_set_size[order_decreasing_time] = last_position_of_tie_group[tie_group_of_position];
      vector[N_tie_groups] events_up_to_last_position_of_tie_group
        = cumulative_sum(event_decreasing_time)[last_position_of_tie_group];
      w_decreasing_time = rep_vector(0, N);
      w_decreasing_time[last_position_of_tie_group]
        = events_up_to_last_position_of_tie_group
          - append_row(0, events_up_to_last_position_of_tie_group[1:N_positions_before_a_decrease]);
    }
    // the event indicators in the original order (generated quantities)
    vector[N] event_vector = to_vector(event);
}


parameters {
    vector[J] log_frailty_raw;         // raw log frailties (nuisance parameters)
    vector[K] beta;                    // regression coefficients
    real<lower=0> sigma_frailty;       // frailty standard deviation
}


model {
    // priors
    beta ~ normal(0, 2);
    sigma_frailty ~ exponential(1);
    log_frailty_raw ~ std_normal();

    // partial likelihood for the Cox model with frailties, in decreasing order of time (see the header)
    if (N > 0) {
      vector[N] eta_decreasing_time = fma(sigma_frailty, log_frailty_raw[group_decreasing_time],
                                          X_decreasing_time * beta);
      target += dot_product(event_decreasing_time, eta_decreasing_time)
                - weighted_sum_of_log_cumulative_sum_exp(w_decreasing_time, eta_decreasing_time,
                                                         negative_infinity());
    }
}


generated quantities {
    vector[J] log_frailty = sigma_frailty * log_frailty_raw;
    vector[N] frailty = log_frailty[group];
    vector[N] log_lik;
    vector[J] frailty_exp = exp(log_frailty);
    // each observation's log likelihood, event[n] (eta[n] - the log of the cumulative sum of exp(eta) at the
    // last position of its risk set), as in shared_frailty_Cox_survival.stan, on the whole vectors
    if (N > 0) {
        vector[N] log_cumulative_risk = log_cumulative_sum_exp(X_decreasing_time * beta
                                                               + log_frailty[group_decreasing_time],
                                                               negative_infinity());
        log_lik = event_vector .* (X * beta + frailty - log_cumulative_risk[risk_set_size]);
    }
}






















