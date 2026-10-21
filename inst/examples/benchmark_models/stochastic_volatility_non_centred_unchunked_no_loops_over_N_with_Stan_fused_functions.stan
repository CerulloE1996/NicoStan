// stochastic_volatility_non_centred_unchunked_no_loops_over_N_with_Stan_fused_functions.stan
// Unchunked plain Stan version of the NON-CENTRED stochastic volatility model of Paper 1's layout timings
// (stochastic_volatility_direct_sliced_boundary_scan_nuisance_block_owned_by_NicoStan, whose nuisance block h_std
// is held by NicoStan, written by hand from the plain Stan model stochastic_volatility_direct_sliced_boundary_
// scan): the same log density as that plain Stan model at N_chunks = 1 (one leaf: no map_rect job, every carry
// 0), and as stochastic_volatility.stan, as one pass over all T time points, without reduce_sum_static or
// map_rect. The version held by NicoStan is not a plain Stan model (its h_std are not Stan parameters); this
// file is the plain Stan model of the same likelihood, with h_std as its first parameters.
// The same data (N and chunk_size, when they are in the data, are not read; y is read as a vector, as in
// stochastic_volatility.stan) and the same unconstrained parameters, in the same order and with the same names
// (h_std, mu, phi, sigma), as both plain Stan files; h_std is a vector, as in stochastic_volatility.stan (the
// chunked file's array[T] real, the same T values in the same order, exists only so that reduce_sum_static can
// slice it).
// The latent path h = mu + d, d[t] = phi d[t - 1] + shocks[t] (d[1] = shocks[1]), with the shocks sigma h_std
// and the first one rescaled by 1 / sqrt(1 - phi^2) (stationary start), is the AR(1) recursion of
// stochastic_volatility.stan written without a loop over the time points, as in the chunked file: in sub-blocks
// of 8 values, each one 8 x 8 lower-triangular matrix of powers of phi times its shocks, with the carries into
// the sub-blocks from the same recursion on the sub-blocks' last values (a recursive call on T / 8 values, then
// T / 64, ...). Stan has no fused function for a normal density whose log scale varies by observation: the
// likelihood is Stan's vectorised normal on the whole vector of returns.
// No loop over N (T) anywhere.


functions {
    // the AR(1) recursion d[t] = a d[t - 1] + shocks[t] from d[0] = carry, without a loop over t: in sub-blocks
    // of sub_block_length values (the function of the chunked file, unchanged)
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
}


data {
    int<lower=1> T;               // # time points (equally spaced)
    vector[T] y;                  // mean corrected return at time t
}


transformed data {
    // the positions in the 8 x 8 lower-triangular matrix of the powers of phi (column by column): phi^(i - j)
    // at row i and column j, 0 above the diagonal (the ninth value), as in the chunked file
    array[64] int index_of_recursion_matrix;
    {
        matrix[8, 8] lag = rep_matrix(linspaced_vector(8, 1, 8), 8)
                             - rep_matrix(linspaced_row_vector(8, 1, 8), 8);
        matrix[8, 8] is_above_diagonal = fmin(1, fmax(0, -lag));
        index_of_recursion_matrix = to_int(to_array_1d(lag + 1 + is_above_diagonal .* (8 - lag)));
    }
}


parameters {
    vector[T] h_std;              // standard normal nuisance block, declared first
    real mu;                      // mean log volatility
    real<lower=-1, upper=1> phi;  // persistence of volatility
    real<lower=0> sigma;          // white noise shock scale
}


model {
    // priors
    phi ~ uniform(-1, 1);
    // Example-scale priors for mean-corrected returns measured in standardised units.
    sigma ~ normal(0, 0.5);
    mu ~ normal(0, 2);
    h_std ~ std_normal();

    // likelihood: the latent path from the shocks (see the header), then the returns
    {
      vector[T] shocks = sigma * h_std;
      shocks[1] /= sqrt(1 - square(phi));
      y ~ normal(0, exp((mu + fn_path(shocks, 0, phi, index_of_recursion_matrix)) / 2));
    }
}


generated quantities {
    vector[T] h;                  // log volatility at time t
    vector[T] y_rep;
    vector[T] volatility;
    {
      vector[T] shocks = sigma * h_std;
      shocks[1] /= sqrt(1 - square(phi));
      h = mu + fn_path(shocks, 0, phi, index_of_recursion_matrix);
    }
    volatility = exp(h / 2);
    y_rep = to_vector(normal_rng(0, volatility));
}






















