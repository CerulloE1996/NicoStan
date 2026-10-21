

// hierarchical_logistic_regression_chunked_no_loops_over_N.stan
// Tape-chunked version of hierarchical_logistic_regression.stan with no loop over the observations or the groups:
// the same log density, with everything whose size grows with N computed inside the chunks of
// reduce_sum_static, on whole vectors. The chunks are the groups (K of them, N / 12 at every data case), each
// with its observations, which are sorted by group once (transformed data, by sorting, with no loop). The
// nuisance block beta_raw is declared as an array of the K groups' rows, so that it is the sliced argument of
// reduce_sum_static itself (no copy of all K rows on the main tape, and each chunk copies only its own groups'
// rows); each chunk adds its groups' share of the nuisance block's standard normal prior, computes its groups'
// coefficients beta = mu + sigma * beta_raw (the transformed parameter of hierarchical_logistic_regression.stan,
// reported here as a generated quantity, the same values) and the likelihood of its groups' observations; only
// mu and sigma (4 values) are shared. The generated quantities are computed on whole vectors too.
// The unconstrained parameters are those of hierarchical_logistic_regression.stan in another order: beta_raw
// row by row (beta_raw[1, 1], beta_raw[1, 2], beta_raw[2, 1], ...) in place of column by column; the names are
// the same.
// The chunks: N_chunks = ceiling(N / chunk_size), from NicoStan's chunk_size of the observations (N, the units
// of the chunks), each of groups_per_chunk = ceiling(K / N_chunks) groups (the grainsize of reduce_sum_static
// over the groups).
//
// The header of the tape-chunked model this one comes from (the next 9 lines):
// hierarchical_logistic_regression_chunked.stan
// Tape-chunked version of hierarchical_logistic_regression.stan: the same log density, with the likelihood
// summed over the groups by reduce_sum_static. The observations are sorted by group once (transformed data), so
// that each group's observations are consecutive, and each group's coefficients, beta[k] (a row of the
// transformed parameter beta), are the sliced argument, so that each chunk copies only its own groups'
// coefficients (and not all K of them); each chunk's term is the original likelihood statement restricted to its
// groups' observations, vectorised over them. The chunks: N_chunks = ceiling(N / chunk_size), from NicoStan's
// chunk_size of the observations (N, the units of the chunks), each of groups_per_chunk = ceiling(K / N_chunks)
// groups (the grainsize of reduce_sum_static over the groups).
//
// The previous tape-chunked model (beta, all K rows, shared by every chunk) is kept below, commented out; its
// header was (the next 9 lines):
// hierarchical_logistic_regression_chunked.stan
// Tape-chunked version of hierarchical_logistic_regression.stan: the same log density, with the
// likelihood summed over the observations by reduce_sum_static, in chunks of chunk_size observations (data), so
// that each chunk's autodiff tape is built, differentiated and released before the next one (Paper 1's tape
// chunking; N_chunks from NicoStan::fn_compute_initial_N_chunks()). Each chunk's term is the original
// likelihood statement restricted to the chunk's observations.
// The chunking data are NicoStan's for Stan models: N (the observations, the units of the chunks) and
// chunk_size, which NicoStan sets to ceiling(N / N_chunks) in each phase (num_chunks_burnin,
// num_chunks_sampling).

functions {
    // real partial_sum_observations_lpmf(array[] int y_slice, int start, int end,
    //                                    array[] int group, vector x, matrix beta) {
    //     return bernoulli_logit_lupmf(y_slice | beta[group[start:end], 1]
    //                                            + beta[group[start:end], 2] .* x[start:end]);
    // }
    // // (the chunk's slice of the data copied once, as data, and indexed from the copy)
    // real partial_sum_observations_lpmf(array[] int y_slice, int start, int end,
    //                                    data array[] int group, data vector x, matrix beta) {
    //     array[end - start + 1] int group_slice = group[start:end];
    //     return bernoulli_logit_lupmf(y_slice | beta[group_slice, 1] + beta[group_slice, 2] .* x[start:end]);
    // }
    // the likelihood of the groups start to end, whose observations are first_of_group[start] to
    // last_of_group[end] in the order sorted by group: each observation's coefficients from beta_slice, by its
    // group's position in the slice
    // real partial_sum_groups(array[] row_vector beta_slice, int start, int end,
    //                         data array[] int y_sorted,
    //                         data vector x_sorted,
    //                         data array[] int group_sorted,
    //                         data array[] int first_of_group,
    //                         data array[] int last_of_group) {
    //     int first = first_of_group[start];
    //     int last = last_of_group[end];
    //     matrix[end - start + 1, 2] beta_of_slice = to_matrix(beta_slice);
    //     array[last - first + 1] int position_in_slice;
    //     for (n in first:last) {
    //         position_in_slice[n - first + 1] = group_sorted[n] - start + 1;
    //     }
    //     return bernoulli_logit_lpmf(y_sorted[first:last] | beta_of_slice[position_in_slice, 1]
    //                                                         + beta_of_slice[position_in_slice, 2]
    //                                                           .* x_sorted[first:last]);
    // }
    //
    // The log density of the groups start to end (one chunk of reduce_sum_static), with no loop: the groups'
    // share of the nuisance block's standard normal prior (to_vector(beta_raw) ~ std_normal()), their
    // coefficients beta = mu + sigma * beta_raw, and the likelihood of their observations, first_of_group[start]
    // to last_of_group[end] in the order sorted by group, each observation's coefficients taken by its group's
    // position in the slice (group_sorted - start + 1, computed on the whole vector of the slice's observations)
    //
    real partial_sum_groups_lpdf(array[] row_vector beta_raw_slice, int start, int end,
                                 vector mu,
                                 vector sigma,
                                 data array[] int y_sorted,
                                 data vector x_sorted,
                                 data array[] int group_sorted,
                                 data array[] int first_of_group,
                                 data array[] int last_of_group) {
        int first = first_of_group[start];
        int last = last_of_group[end];
        matrix[end - start + 1, 2] beta_raw_of_slice = to_matrix(beta_raw_slice);
        matrix[end - start + 1, 2] beta_of_slice;
        for (i in 1:2) {
            beta_of_slice[, i] = mu[i] + sigma[i] * beta_raw_of_slice[, i];
        }
        array[last - first + 1] int position_in_slice = to_int(to_array_1d(to_vector(group_sorted[first:last])
                                                                           - (start - 1)));
        return std_normal_lupdf(to_vector(beta_raw_of_slice))
               + bernoulli_logit_lupmf(y_sorted[first:last] | beta_of_slice[position_in_slice, 1]
                                                              + beta_of_slice[position_in_slice, 2]
                                                                .* x_sorted[first:last]);
    }
}


data {
    int<lower=0> N;                    // number of observations
    int<lower=1> K;                    // number of groups
    array[N] int<lower=1, upper=K> group; // group indicator for each observation
    vector[N] x;                       // predictor variable
    array[N] int<lower=0, upper=1> y;  // binary outcome
    // int<lower=1> grainsize;            // observations per chunk (reduce_sum_static)
    int<lower=1> chunk_size;           // observations per chunk (reduce_sum_static)
}


transformed data {
    // the observations sorted by group (each group's observations consecutive), and each group's first and last
    // observation in that order (a group without observations has last = first - 1)
    array[N] int order_by_group = sort_indices_asc(group);
    array[N] int group_sorted = group[order_by_group];
    array[N] int y_sorted = y[order_by_group];
    vector[N] x_sorted = x[order_by_group];
    // array[K] int n_of_group = rep_array(0, K);
    // array[K] int first_of_group;
    // array[K] int last_of_group;
    // for (n in 1:N) {
    //   n_of_group[group[n]] += 1;
    // }
    // {
    //   int position = 0;
    //   for (k in 1:K) {
    //     first_of_group[k] = position + 1;
    //     position += n_of_group[k];
    //     last_of_group[k] = position;
    //   }
    // }
    // (with no loop: group k's last observation is the number of observations of groups 1 to k, i.e. the
    //  position of k + 0.5 when the N groups of the observations and the K values 1.5, 2.5, ..., K + 0.5 are
    //  sorted together, minus the k values 1.5 to k + 0.5 up to it; its first observation is one after the
    //  last of group k - 1. sort_indices_asc(sort_indices_asc(v)) is the position of each element of v in v
    //  sorted.)
    array[N + K] real groups_and_half_way_values = append_array(to_array_1d(to_vector(group)),
                                                                to_array_1d(linspaced_vector(K, 1.5, K + 0.5)));
    array[N + K] int position_when_sorted = sort_indices_asc(sort_indices_asc(groups_and_half_way_values));
    array[K] int last_of_group = to_int(to_array_1d(to_vector(position_when_sorted[(N + 1):(N + K)])
                                                    - linspaced_vector(K, 1, K)));
    array[K] int first_of_group = append_array({1}, to_int(to_array_1d(to_vector(last_of_group[1:(K - 1)])
                                                                       + 1)));
    // the chunks: N_chunks from NicoStan's chunk_size of the observations, each of groups_per_chunk groups
    int N_chunks = (N + chunk_size - 1) %/% chunk_size;
    int groups_per_chunk = (K + N_chunks - 1) %/% N_chunks;
}


parameters {
    // matrix[K, 2] beta_raw;             // standard normal nuisance block, declared first
    // (the same block as an array of the K groups' rows, the sliced argument of reduce_sum_static)
    array[K] row_vector[2] beta_raw;   // standard normal nuisance block, declared first
    vector[2] mu;                      // population means for [intercept, slope]
    vector<lower=0>[2] sigma;          // population standard deviations
}

// transformed parameters {
//     matrix[K, 2] beta;                 // group-specific intercepts and slopes
//     for (i in 1:2) {
//       beta[, i] = mu[i] + sigma[i] * beta_raw[, i];
//     }
// }
// (beta: computed inside the chunks for the log density, and reported as a generated quantity)

model {
    // Priors
    mu ~ normal(0, 2);
    sigma ~ normal(0, 2);

    // Equivalent non-centred form of beta[, i] ~ normal(mu[i], sigma[i]).
    // to_vector(beta_raw) ~ std_normal();
    // (added inside the chunks, group by group: see partial_sum_groups_lpdf)

    // Likelihood (reduce_sum_static over the observations, in chunks of chunk_size)
    // target += reduce_sum_static(partial_sum_observations_lupmf, y, grainsize, group, x, beta);
    // target += reduce_sum_static(partial_sum_observations_lupmf, y, chunk_size, group, x, beta);

    // Likelihood (reduce_sum_static over the groups, in chunks of groups_per_chunk groups, each group's
    // coefficients sliced)
    // array[K] row_vector[2] beta_by_group;
    // for (k in 1:K) {
    //   beta_by_group[k] = beta[k];
    // }
    // target += reduce_sum_static(partial_sum_groups, beta_by_group, groups_per_chunk,
    //                             y_sorted, x_sorted, group_sorted, first_of_group, last_of_group);

    // The nuisance block's prior and the likelihood (reduce_sum_static over the groups, in chunks of
    // groups_per_chunk groups, each group's row of beta_raw sliced; mu and sigma shared)
    target += reduce_sum_static(partial_sum_groups_lupdf, beta_raw, groups_per_chunk, mu, sigma,
                                y_sorted, x_sorted, group_sorted, first_of_group, last_of_group);
}


generated quantities {
    // (beta: the transformed parameter of hierarchical_logistic_regression.stan, the same values)
    matrix[K, 2] beta;                 // group-specific intercepts and slopes
    vector[N] log_lik;
    vector[N] y_rep;

    {
      matrix[K, 2] beta_raw_matrix = to_matrix(beta_raw);
      for (i in 1:2) {
        beta[, i] = mu[i] + sigma[i] * beta_raw_matrix[, i];
      }
    }

    // for (n in 1:N) {
    //   real eta = beta[group[n], 1] + beta[group[n], 2] * x[n];
    //   log_lik[n] = bernoulli_logit_lpmf(y[n] | eta);
    //   y_rep[n] = bernoulli_rng(inv_logit(eta));
    // }
    // (on the whole vectors: each observation's log likelihood, bernoulli_logit_lpmf(y[n] | eta[n]), is
    //  log_inv_logit((2 y[n] - 1) eta[n]), and bernoulli_logit_rng(eta) draws the replicates in the same order,
    //  each as bernoulli_rng(inv_logit(eta[n])))
    {
      vector[N] eta = beta[group, 1] + beta[group, 2] .* x;
      log_lik = log_inv_logit((2 * to_vector(y) - 1) .* eta);
      y_rep = to_vector(bernoulli_logit_rng(eta));
    }
}
