// Stochastic volatility, CENTRED: the log volatilities h are the parameters (the same posterior over
// (mu, phi, sigma, h) as the non-centred model with h_std; the sampler moves in other coordinates). Every term
// depends on h[t] and h[t - 1] only, so each chunk of reduce_sum_static holds its own observations and
// transitions; outside the chunks there are only h[1]'s stationary term and the transitions across chunk
// boundaries (one per boundary). The deterministic binary partitions match Stan Math's simple partitioner for
// reduce_sum_static. Parameter count and order do not depend on chunk_size.
// AVX version: each chunk's two normal terms are BayesMVP's nico_normal_affine_location_and_log_scale_lpdf
// (one node each, all SIMD): y[t] ~ normal(0, exp(0 + 0.5 h[t])) and h[t] ~ normal(mu (1 - phi) + phi h[t - 1],
// exp(log(sigma) + 0 h[t - 1])).

functions {
    real bmvp_simd_lanes();
    real nico_normal_affine_location_and_log_scale_lpdf(vector y, real a0, real a1, vector x, real b0, real b1,
                                                        vector w);
    array[] int fn_deterministic_leaf_starts(data int first, data int last, data int chunk_size) {
        int length_of_range = last - first + 1;
        if (length_of_range <= chunk_size) return { first };
        int middle = first + length_of_range %/% 2;
        return append_array(fn_deterministic_leaf_starts(first, middle - 1, chunk_size),
                            fn_deterministic_leaf_starts(middle, last, chunk_size));
    }

    real fn_density_of_latent_path_slice_lpdf(array[] real h_slice, int start, int end, data array[] real y,
                                              data array[] int leaf_first, data array[] int leaf_of_time,
                                              real mu, real phi, real sigma) {
        int n = end - start + 1;
        if (start != leaf_first[leaf_of_time[start]]) {
            reject("Unexpected deterministic reduce_sum_static boundary: ", start);
        }
        vector[n] h_chunk = to_vector(h_slice);
        real log_density = nico_normal_affine_location_and_log_scale_lupdf(to_vector(y[start:end]) | 0, 0,
                                                                            h_chunk, 0, 0.5, h_chunk);
        if (n > 1) {
            log_density += nico_normal_affine_location_and_log_scale_lupdf(h_chunk[2:n] | mu * (1 - phi), phi,
                                                                            h_chunk[1:(n - 1)], log(sigma), 0,
                                                                            h_chunk[1:(n - 1)]);
        }
        return log_density;
    }
}

data {
    int<lower=1> T;
    array[T] real y;
    int<lower=1> N;
    int<lower=1> chunk_size;
}

transformed data {
    if (N != T) reject("N must equal the number of time points T.");
    int n_leaves = size(fn_deterministic_leaf_starts(1, T, chunk_size));
    array[n_leaves] int leaf_first = fn_deterministic_leaf_starts(1, T, chunk_size);
    array[T] int leaf_of_time;
    for (b in 1:n_leaves) {
        int leaf_last = b < n_leaves ? leaf_first[b + 1] - 1 : T;
        leaf_of_time[leaf_first[b]:leaf_last] = rep_array(b, leaf_last - leaf_first[b] + 1);
    }
    //// (the first time point of every chunk but the first, and the time point before it: the transitions that
    ////  cross a chunk boundary, the only latent terms outside the chunks)
    array[n_leaves - 1] int first_of_later_leaves = leaf_first[2:n_leaves];
    array[n_leaves - 1] int last_of_earlier_leaves;
    for (b in 2:n_leaves) last_of_earlier_leaves[b - 1] = leaf_first[b] - 1;
}

parameters {
    array[T] real h;
    real mu;
    real<lower=-1, upper=1> phi;
    real<lower=0> sigma;
}

model {
    phi ~ uniform(-1, 1);
    sigma ~ normal(0, 0.5);
    mu ~ normal(0, 2);
    //// the stationary first state and the transitions that cross a chunk boundary (n_leaves - 1 terms)
    h[1] ~ normal(mu, sigma / sqrt(1 - square(phi)));
    if (n_leaves > 1) {
        target += nico_normal_affine_location_and_log_scale_lupdf(to_vector(h[first_of_later_leaves]) |
                              mu * (1 - phi), phi, to_vector(h[last_of_earlier_leaves]), log(sigma), 0,
                              to_vector(h[last_of_earlier_leaves]));
    }
    target += reduce_sum_static(fn_density_of_latent_path_slice_lupdf, h, chunk_size,
                                 y, leaf_first, leaf_of_time, mu, phi, sigma);
}

generated quantities {
    real simd_lanes = bmvp_simd_lanes();
}






















