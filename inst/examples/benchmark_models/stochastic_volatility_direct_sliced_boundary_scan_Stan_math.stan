// Direct slicing of the standard-normal innovations, with no full-sized transformed working copy.
// The deterministic binary partitions match Stan Math's simple partitioner for reduce_sum_static.
// Block-end messages are evaluated on nested tapes, then only their carries link the density chunks.
// Parameter count, coordinate order, stationary initial scaling and priors do not depend on chunk_size.

functions {
    vector fn_AR1_recursion_in_sub_blocks_without_a_loop(vector shocks, real carry, real a, int sub_block_length,
                                                        data array[] int index_of_recursion_matrix) {
        int L = num_elements(shocks);
        int P = sub_block_length;
        int n_sub = (L + P - 1) %/% P;                         // the number of sub-blocks
        vector[P + 1] a_power = append_row(1, append_row(a, pow(a, linspaced_vector(P - 1, 2, P))));   // a^0..a^P
        matrix[P, P] recursion_matrix = to_matrix(append_row(a_power[1:P], 0)[index_of_recursion_matrix], P, P);
        matrix[P, n_sub] shocks_by_sub_block = to_matrix(append_row(shocks, rep_vector(0, n_sub * P - L)),
                                                         P, n_sub);
        row_vector[n_sub] carry_into_sub_block;                // d before each sub-block's first value
        vector[n_sub * P] d_padded;
        carry_into_sub_block[1] = carry;
        if (n_sub > 1) {
            // each sub-block's recursion from 0, at its last value: the product of a^(P - p), p = 1, ..., P, with
            // its shocks
            row_vector[n_sub - 1] end_of_sub_block_from_zero = reverse(a_power[1:P])'
                                                               * shocks_by_sub_block[, 1:(n_sub - 1)];
            carry_into_sub_block[2:n_sub] = fn_AR1_recursion_in_sub_blocks_without_a_loop(
                                                  end_of_sub_block_from_zero', carry, a_power[P + 1], P,
                                                  index_of_recursion_matrix)';
        }
        // (each sub-block's first shock takes its carry: d[1] = a carry + shocks[1])
        shocks_by_sub_block[1] = shocks_by_sub_block[1] + a * carry_into_sub_block;
        d_padded = to_vector(recursion_matrix * shocks_by_sub_block);
        return d_padded[1:L];
    }
    vector fn_path(vector shocks, real carry, real phi, data array[] int recursion_index) {
        return fn_AR1_recursion_in_sub_blocks_without_a_loop(shocks, carry, phi, 8, recursion_index);
    }

    real fn_last_value_of_affine_recurrence(vector shocks, real carry, real coefficient) {
        int length_of_block = rows(shocks);
        int width = 8;
        if (length_of_block == 0) return carry;
        vector[width] powers = append_row(1, append_row(coefficient,
                                             pow(coefficient, linspaced_vector(width - 2, 2, width - 1))));
        if (length_of_block <= width) {
            return dot_product(shocks, reverse(powers[1:length_of_block]))
                     + pow(coefficient, length_of_block) * carry;
        }
        int n_sub_blocks = (length_of_block + width - 1) %/% width;
        matrix[width, n_sub_blocks] shocks_of_sub_blocks
            = to_matrix(append_row(rep_vector(0, n_sub_blocks * width - length_of_block), shocks),
                          width, n_sub_blocks);
        vector[n_sub_blocks] ends_from_zero = (reverse(powers)' * shocks_of_sub_blocks)';
        return fn_last_value_of_affine_recurrence(ends_from_zero, 0, pow(coefficient, width))
                 + pow(coefficient, length_of_block) * carry;
    }

    array[] int fn_deterministic_leaf_starts(data int first, data int last, data int grainsize) {
        int length_of_range = last - first + 1;
        if (length_of_range <= grainsize) return { first };
        int middle = first + length_of_range %/% 2;
        return append_array(fn_deterministic_leaf_starts(first, middle - 1, grainsize),
                            fn_deterministic_leaf_starts(middle, last, grainsize));
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
        return [phi_and_sigma[2] * fn_last_value_of_affine_recurrence(values, 0, phi_and_sigma[1])]';
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
        vector[length_of_slice] shocks = sigma * innovations;
        real carry = carries[leaf];
        if (start == 1) shocks[1] /= sqrt(1 - square(phi));
        vector[length_of_slice] d = fn_path(shocks, carry, phi, index_of_recursion_matrix);
        return std_normal_lupdf(innovations)
                 + normal_lupdf(to_vector(y[start:end]) | 0, exp((mu + d) / 2));
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
        array[n_jobs] vector[largest_leaf_length] innovations_of_blocks;
        vector[2] phi_and_sigma = [phi, sigma]';
        for (b in 1:n_jobs) {
            innovations_of_blocks[b, 1:leaf_length[b]] = to_vector(h_std[leaf_first[b]:leaf_last[b]]);
            if (leaf_length[b] < largest_leaf_length) {
                innovations_of_blocks[b, (leaf_length[b] + 1):largest_leaf_length]
                    = rep_vector(0, largest_leaf_length - leaf_length[b]);
            }
        }
        vector[n_jobs] ends_from_zero = map_rect(fn_end_of_innovation_block, phi_and_sigma,
                                                  innovations_of_blocks, unused_real_data, block_data);
        for (b in 1:n_jobs) {
            carries[b + 1] = fma(pow(phi, leaf_length[b]), carries[b], ends_from_zero[b]);
        }
    }
    target += reduce_sum_static(fn_density_of_innovation_slice_lupdf, h_std, chunk_size,
                                 y, carries, leaf_first, leaf_of_time, mu, phi, sigma,
                                 index_of_recursion_matrix);
}

generated quantities {

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
        vector[length_of_block] scales = exp(h_of_block / 2);
        h[first:last] = to_array_1d(h_of_block);
        volatility[first:last] = to_array_1d(scales);
        y_rep[first:last] = normal_rng(0, scales);
    }
}










































