// Direct slicing of the standard-normal innovations, with no full-sized transformed working copy.
// The deterministic binary partitions match Stan Math's simple partitioner for reduce_sum_static.
// Block-end messages are evaluated on nested tapes, then only their carries link the density chunks.
// Parameter count, coordinate order, stationary initial scaling and priors do not depend on chunk_size.

functions {

    real bmvp_simd_lanes();
    vector nico_exp(vector x);
    real nico_normal_lpdf(vector y, real mu, real sigma);
    vector nico_AR1_recursion(vector shocks, real carry, real a);
    vector nico_affine_recurrence_at_positions(array[] real innovations, data array[] int positions,
                                               real initial_value, real coefficient,
                                               real scale, real first_scale);
    real nico_last_value_of_affine_recurrence(vector innovations, real initial_value,
                                              real coefficient, real scale);
    real nico_normal_affine_log_scale_lpdf(vector y, real mu, real intercept, real slope, vector predictor);
    real nico_normal_affine_recurrence_log_scale_joint_lpdf(
        vector y, real mu, vector innovations, real initial_value, real coefficient, real scale,
        real intercept, real slope, real innovation_mean, real innovation_sd);
    real nico_normal_affine_recurrence_log_scale_joint_lpdf(
        vector y, real mu, vector innovations, real initial_value, real coefficient, vector scale,
        real intercept, real slope, real innovation_mean, real innovation_sd);
    vector fn_path(vector shocks, real carry, real phi, data array[] int recursion_index) {
        return nico_AR1_recursion(shocks, carry, phi);
    }

    array[] int fn_deterministic_leaf_starts(data int first, data int last, data int chunk_size) {
        int length_of_range = last - first + 1;
        if (length_of_range <= chunk_size) return { first };
        int middle = first + length_of_range %/% 2;
        return append_array(fn_deterministic_leaf_starts(first, middle - 1, chunk_size),
                            fn_deterministic_leaf_starts(middle, last, chunk_size));
    }

    vector fn_end_of_innovation_block(vector phi_and_sigma, vector innovations,
                                       data array[] real unused_real_data, data array[] int block_data) {
        int length_of_block = block_data[1];
        // vector[length_of_block] unit_innovation = append_row(1, rep_vector(0, length_of_block - 1));
        // vector[length_of_block] powers_of_phi = fn_path(unit_innovation, 0, phi_and_sigma[1],
        //                                                 block_data[3:66]);
        vector[length_of_block] values = innovations[1:length_of_block];
        if (block_data[2] == 1) values[1] /= sqrt(1 - square(phi_and_sigma[1]));
        // return [phi_and_sigma[2] * dot_product(values, reverse(powers_of_phi))]';
        return [nico_last_value_of_affine_recurrence(values, 0, phi_and_sigma[1], phi_and_sigma[2])]';
    }

    real fn_density_of_innovation_slice_lpdf(array[] real innovations_slice, int start, int end,
                                              data array[] real y, array[] real carries,
                                              data array[] int leaf_first, data array[] int leaf_of_time,
                                              real mu, real phi, real sigma,
                                              data array[] int index_of_recursion_matrix) {
        int length_of_slice = end - start + 1;
        int leaf = leaf_of_time[start];
        if (start != leaf_first[leaf]) reject("Unexpected deterministic reduce_sum_static boundary: ", start);
        vector[length_of_slice] innovations = to_vector(innovations_slice);
        // vector[length_of_slice] shocks = sigma * innovations;
        real carry = carries[leaf];
        // if (start == 1) shocks[1] /= sqrt(1 - square(phi));
        // vector[length_of_slice] d = fn_path(shocks, carry, phi, index_of_recursion_matrix);
        // return nico_normal_lupdf(innovations | 0, 1)
        //          + nico_normal_affine_log_scale_lupdf(to_vector(y[start:end]) | 0, mu / 2, 0.5, d);
        if (start == 1) {
            vector[length_of_slice] scales = rep_vector(sigma, length_of_slice);
            scales[1] /= sqrt(1 - square(phi));
            return nico_normal_affine_recurrence_log_scale_joint_lupdf(
                to_vector(y[start:end]) | 0, innovations, carry, phi, scales, mu / 2, 0.5, 0, 1);
        }
        return nico_normal_affine_recurrence_log_scale_joint_lupdf(
            to_vector(y[start:end]) | 0, innovations, carry, phi, sigma, mu / 2, 0.5, 0, 1);
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
    int n_jobs = n_leaves - 1;
    array[n_leaves] int leaf_last;
    array[n_leaves] int leaf_length;
    array[T] int leaf_of_time;
    array[n_jobs, 66] int block_data;
    array[n_jobs, 0] real unused_real_data;
    array[64] int index_of_recursion_matrix;
    int largest_leaf_length;
    {
        matrix[8, 8] lag = rep_matrix(linspaced_vector(8, 1, 8), 8)
                             - rep_matrix(linspaced_row_vector(8, 1, 8), 8);
        matrix[8, 8] is_above_diagonal = fmin(1, fmax(0, -lag));
        index_of_recursion_matrix = to_int(to_array_1d(lag + 1 + is_above_diagonal .* (8 - lag)));
    }
    for (b in 1:n_leaves) {
        leaf_last[b] = b < n_leaves ? leaf_first[b + 1] - 1 : T;
        leaf_length[b] = leaf_last[b] - leaf_first[b] + 1;
        leaf_of_time[leaf_first[b]:leaf_last[b]] = rep_array(b, leaf_length[b]);
    }
    largest_leaf_length = max(leaf_length);
    if (n_jobs > 0) {
        for (b in 1:n_jobs) {
            block_data[b, 1] = leaf_length[b];
            block_data[b, 2] = b == 1 ? 1 : 0;
            block_data[b, 3:66] = index_of_recursion_matrix;
        }
    }
}

parameters {
    array[T] real h_std;
    real mu;
    real<lower=-1, upper=1> phi;
    real<lower=0> sigma;
}

model {
    phi ~ uniform(-1, 1);
    sigma ~ normal(0, 0.5);
    mu ~ normal(0, 2);

    array[n_leaves] real carries;
    carries[1] = 0;
    if (n_jobs > 0) {
        // array[n_jobs] vector[largest_leaf_length] innovations_of_blocks;
        // vector[2] phi_and_sigma = [phi, sigma]';
        // for (b in 1:n_jobs) {
        //     innovations_of_blocks[b, 1:leaf_length[b]] = to_vector(h_std[leaf_first[b]:leaf_last[b]]);
        //     if (leaf_length[b] < largest_leaf_length) {
        //         innovations_of_blocks[b, (leaf_length[b] + 1):largest_leaf_length]
        //             = rep_vector(0, largest_leaf_length - leaf_length[b]);
        //     }
        // }
        // vector[n_jobs] ends_from_zero = map_rect(fn_end_of_innovation_block, phi_and_sigma,
        //                                           innovations_of_blocks, unused_real_data, block_data);
        // for (b in 1:n_jobs) {
        //     carries[b + 1] = fma(pow(phi, leaf_length[b]), carries[b], ends_from_zero[b]);
        // }
        vector[n_jobs] boundaries = nico_affine_recurrence_at_positions(
            h_std, leaf_last[1:n_jobs], 0, phi, sigma, sigma / sqrt(1 - square(phi)));
        carries[2:n_leaves] = to_array_1d(boundaries);
    }
    target += reduce_sum_static(fn_density_of_innovation_slice_lupdf, h_std, chunk_size,
                                 y, carries, leaf_first, leaf_of_time, mu, phi, sigma,
                                 index_of_recursion_matrix);
}

generated quantities {
    real simd_lanes = bmvp_simd_lanes();
    array[T] real h;
    array[T] real y_rep;
    array[T] real volatility;
    for (b in 1:n_leaves) {
        int first = leaf_first[b];
        int last = leaf_last[b];
        int length_of_block = leaf_length[b];
        real carry = first == 1 ? 0 : h[first - 1] - mu;
        vector[length_of_block] shocks = sigma * to_vector(h_std[first:last]);
        if (first == 1) shocks[1] /= sqrt(1 - square(phi));
        vector[length_of_block] h_of_block = mu + fn_path(shocks, carry, phi, index_of_recursion_matrix);
        vector[length_of_block] scales = nico_exp(h_of_block / 2);
        h[first:last] = to_array_1d(h_of_block);
        volatility[first:last] = to_array_1d(scales);
        y_rep[first:last] = normal_rng(0, scales);
    }
}







































