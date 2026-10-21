// graded_response_IRT_chunked_no_loops_over_N_hand_written_BayesMVP_AVX.stan
// Hand-written AVX version of models_staging_no_loops_over_N_hierarchical_robust_t_IRT/graded_response_IRT_chunked_no_loops_over_N.stan: every log density and elementwise function on a
// vector or matrix is BayesMVP's fast_ function, declared in the functions block; calls on single
// reals stay on Stan math. One text for both instruction sets (NicoStan's AVX2 or AVX-512 make
// arguments).
//
// graded_response_IRT_chunked_no_loops_over_N.stan
// Tape-chunked version of graded_response_IRT.stan with no loop over the persons or the responses: the same log
// density, with everything whose size grows with N computed inside the chunks of reduce_sum_static, on whole
// vectors. The abilities theta are declared as an array, so that they are the sliced argument of
// reduce_sum_static itself (no copy of all N abilities on the main tape, and each chunk copies only its own
// persons' theta); the item parameters a and c (J + J (K - 1) values) are shared. Each chunk adds its persons'
// share of the abilities' standard normal prior (theta ~ std_normal()) and, item by item (J = 20 items, a
// number that does not grow with N), one ordered logistic statement on the item's responses of the chunk's
// persons. The model is defined for complete data (every person answers every item once; see below), so these
// responses are one contiguous block of the responses sorted by item, and then by person, in the order of the
// chunk's abilities: each statement takes the chunk's abilities as they are, with no index of the responses'
// persons. The responses are sorted by item once (transformed data, by sorting, with no loop), and the data are
// checked to be complete. The abilities theta are the same unconstrained parameters, in the same order and with
// the same names, as in graded_response_IRT.stan.
// The chunking data are NicoStan's for Stan models: N (the persons, the units of the chunks; already in the data)
// and chunk_size, which NicoStan sets to ceiling(N / N_chunks) in each phase (num_chunks_burnin,
// num_chunks_sampling).
//
// The header of the tape-chunked model this one comes from (the next 14 lines):
// Tape-chunked version of graded_response_IRT.stan: the same log density, with the likelihood of the responses
// summed over the persons by reduce_sum_static, in chunks of chunk_size persons (data), so that each chunk's
// autodiff tape is built, differentiated and released before the next one (the chunking paper's
// chunking; N_chunks
// from NicoStan::fn_compute_initial_N_chunks()). Each chunk's term is the original likelihood statement
// restricted to the responses of the chunk's persons (the responses are sorted by person, so the chunk's
// responses are contiguous: first_response[start] to last_response[end]), with the responses' persons counted
// from the chunk's first person. The abilities theta are the sliced argument of reduce_sum_static
// (to_array_1d(theta)), so that each chunk copies only its own persons' theta, and not the whole person block,
// onto its tape; the item parameters a and c (J + J (K - 1) values) are shared. Each chunk also adds its
// persons' share of the abilities' standard normal prior (theta ~ std_normal()), so that nothing per person is
// computed outside reduce_sum_static.
// The chunking data are NicoStan's for Stan models: N (the persons, the units of the chunks; already in the data)
// and chunk_size, which NicoStan sets to ceiling(N / N_chunks) in each phase (num_chunks_burnin,
// num_chunks_sampling).
//
// The model, as in graded_response_IRT.stan:
//
// Samejima's graded response model (GRM; Samejima 1969, Psychometrika Monograph Supplement 17) for ordinal item
// responses, in its slope-intercept form on the logistic metric: N persons each answer every one of J items
// with one of K ordered categories 1, ..., K (complete data; the responses are sorted by person, so that each
// person's J responses are contiguous). For person n and item j,
//   P(Y_nj >= k) = inv_logit(a_j theta_n - c_j,k-1),  k = 2, ..., K,
// with the person's ability theta_n, the item's discrimination a_j > 0 and its ordered cutpoints
// c_j,1 < ... < c_j,K-1 (logit scale; the thresholds on the ability scale are b_jk = c_jk / a_j), i.e.
//   y ~ ordered_logistic(a[item] .* theta[person], c[item])
// (Stan's vectorised ordered logistic, with the cutpoint vector of each response's item).
//
// The abilities theta (N) are the nuisance block, declared first (NicoStan samples the first declared block,
// with its standard normal prior, by diffusion-pathspace HMC); their N(0, 1) prior fixes the location and the
// scale of the latent trait, and a_j > 0 its sign. The main block is a (J) and c (J x (K - 1)), i.e. J K
// unconstrained coordinates (100 for J = 20 items with K = 5 categories).
//
// Priors: a_j ~ lognormal(0, 0.5) (median 1, 95% prior interval 0.38 to 2.66: weakly informative over the usual
// range of discriminations on the logistic metric, about 0.5 to 2.5); every cutpoint c_jk ~ N(0, 5^2), on the
// ordered vector of each item (weak for an intercept on the logit scale: c_jk = a_j b_jk lies within about -6 to
// 6 for discriminations up to 2.5 and thresholds within -2.5 to 2.5); theta_n ~ N(0, 1).
//
functions {
    real bmvp_simd_lanes();
    real nico_normal_lpdf(vector y, real mu, real sigma);
    real nico_ordered_logistic_lpmf(array[] int k, vector eta, vector c);
    //
    // The log likelihood of the responses of persons start to end (one chunk of reduce_sum_static): the chunk's
    // abilities (theta_slice), the responses' persons counted from the chunk's first person, and the original
    // likelihood statement on the chunk's responses (first_response[start] to last_response[end]):
    //
    // real partial_sum_persons_lpdf(array[] real theta_slice, int start, int end, vector a, array[] vector c,
    //                               data array[] int person, data array[] int item, data array[] int y,
    //                               data array[] int first_response, data array[] int last_response) {
    //     int r1 = first_response[start];
    //     int r2 = last_response[end];
    //     vector[end - start + 1] theta_chunk = to_vector(theta_slice);
    //     array[r2 - r1 + 1] int person_in_chunk;
    //     for (r in r1:r2) {
    //         person_in_chunk[r - r1 + 1] = person[r] - start + 1;
    //     }
    //     return ordered_logistic_lupmf(y[r1:r2] | a[item[r1:r2]] .* theta_chunk[person_in_chunk],
    //                                              c[item[r1:r2]]);
    // }
    //
    // The same log likelihood with the chunk's responses grouped by item: one ordered logistic statement per
    // item, on the item's responses of persons start to end (y_by_item, person_by_item: every item's responses,
    // in the order of their persons, item after item), with the item's own discrimination and its one cutpoint
    // vector shared by all of them, in place of a copy of the item's cutpoints for every response:
    //
    // (the chunk's persons' share of the nuisance block's prior, theta ~ std_normal(), is also added here, so
    //  that nothing per person is computed outside reduce_sum_static)
    //
    // real partial_sum_persons_by_item_lpdf(array[] real theta_slice, int start, int end, vector a,
    //                                       array[] vector c, data array[] int y_by_item,
    //                                       data array[] int person_by_item, data array[] int item_offset,
    //                                       data array[,] int item_responses_before_person) {
    //     int J = num_elements(a);
    //     vector[end - start + 1] theta_chunk = to_vector(theta_slice);
    //     real log_likelihood = 0;
    //     for (j in 1:J) {
    //         int first = item_offset[j] + item_responses_before_person[j, start] + 1;
    //         int last = item_offset[j] + item_responses_before_person[j, end + 1];
    //         if (last >= first) {
    //             array[last - first + 1] int person_in_chunk;
    //             for (r in first:last) {
    //                 person_in_chunk[r - first + 1] = person_by_item[r] - start + 1;
    //             }
    //             log_likelihood += ordered_logistic_lupmf(y_by_item[first:last] |
    //                                                      a[j] * theta_chunk[person_in_chunk], c[j]);
    //         }
    //     }
    //     // return log_likelihood;
    //     // (with the chunk's persons' share of the abilities' standard normal prior, theta ~ std_normal())
    //     return std_normal_lupdf(theta_chunk) + log_likelihood;
    // }
    //
    // The same log density with no loop over the responses, for complete data (every person answers every item
    // once): item j's responses of persons start to end are y_by_item[(j - 1) N + start] to
    // y_by_item[(j - 1) N + end] (the responses sorted by item, and then by person), in the order of the chunk's
    // abilities, so that the item's statement takes the chunk's abilities as they are, a[j] * theta_chunk, with
    // no index of the responses' persons; the loop is over the J items, whose number does not grow with N
    // (the chunk's persons' share of the abilities' standard normal prior, theta ~ std_normal(), is added here)
    //
    real partial_sum_persons_by_item_of_complete_data_lpdf(array[] real theta_slice, int start, int end,
                                                           vector a, array[] vector c,
                                                           data array[] int y_by_item, int N) {
        int J = num_elements(a);
        vector[end - start + 1] theta_chunk = to_vector(theta_slice);
        real log_likelihood = 0;
        for (j in 1:J) {
            log_likelihood += nico_ordered_logistic_lupmf(y_by_item[((j - 1) * N + start):((j - 1) * N + end)] |
                                                     a[j] * theta_chunk, c[j]);
        }
        return nico_normal_lupdf(theta_chunk | 0, 1) + log_likelihood;
    }
}
data {
    int<lower=1> N;                                  // number of persons
    int<lower=1> J;                                  // number of items
    int<lower=2> K;                                  // number of ordered response categories
    int<lower=1> n_responses;                        // number of responses (N J: complete data)
    array[n_responses] int<lower=1, upper=N> person; // person of each response (responses sorted by person)
    array[n_responses] int<lower=1, upper=J> item;   // item of each response
    array[n_responses] int<lower=1, upper=K> y;      // response category of each response
    int<lower=1> chunk_size;                         // persons per chunk (reduce_sum_static)
}
transformed data {
    // // each person's responses: first_response[i] to last_response[i]
    // array[N] int first_response = rep_array(0, N);
    // array[N] int last_response = rep_array(0, N);
    // for (r in 1:n_responses) {
    //     if (r > 1 && person[r] < person[r - 1]) reject("the responses must be sorted by person");
    //     if (first_response[person[r]] == 0) first_response[person[r]] = r;
    //     last_response[person[r]] = r;
    // }
    // for (i in 1:N) {
    //     if (first_response[i] == 0) reject("person ", i, " has no responses");
    // }
    // // the responses grouped by item (each item's in the order of its persons): item j's responses are
    // // item_offset[j] + 1 to item_offset[j + 1] of y_by_item and person_by_item, and
    // // item_responses_before_person[j, i] of them belong to persons before person i (i = 1, ..., N + 1)
    // array[J] int n_responses_of_item = rep_array(0, J);
    // for (r in 1:n_responses) {
    //     n_responses_of_item[item[r]] += 1;
    // }
    // array[J + 1] int item_offset;
    // item_offset[1] = 0;
    // for (j in 1:J) {
    //     item_offset[j + 1] = item_offset[j] + n_responses_of_item[j];
    // }
    // array[n_responses] int y_by_item;
    // array[n_responses] int person_by_item;
    // array[J, N + 1] int item_responses_before_person;
    // {
    //     array[J] int filled = rep_array(0, J);
    //     for (i in 1:N) {
    //         for (j in 1:J) {
    //             item_responses_before_person[j, i] = filled[j];
    //         }
    //         for (r in first_response[i]:last_response[i]) {
    //             filled[item[r]] += 1;
    //             y_by_item[item_offset[item[r]] + filled[item[r]]] = y[r];
    //             person_by_item[item_offset[item[r]] + filled[item[r]]] = person[r];
    //         }
    //     }
    //     for (j in 1:J) {
    //         item_responses_before_person[j, N + 1] = filled[j];
    //     }
    // }
    //
    // (with no loop:) the responses sorted by item, and then by person, by the key item (N + 1) + person of each
    // response; for complete data (every person answers every item once, as the model is defined), the keys are
    // distinct and, sorted, are those of (item 1, person 1), ..., (item 1, person N), (item 2, person 1), ...,
    // (item J, person N), i.e. j (N + 1) + i in column j and row i of an N x J matrix, read column by column:
    // item j's responses are y_by_item[(j - 1) N + 1] to y_by_item[j N], those of persons 1 to N in turn
    vector[n_responses] key_of_response = to_vector(item) * (N + 1) + to_vector(person);
    array[n_responses] int order_by_item = sort_indices_asc(key_of_response);
    array[n_responses] int y_by_item = y[order_by_item];
    if (n_responses != N * J) {
        reject("the model is defined for complete data: n_responses must be N J = ", N * J,
               "; got n_responses = ", n_responses);
    }
    {
        matrix[N, J] key_of_each_person_and_item = rep_matrix(linspaced_row_vector(J, 1, J) * (N + 1), N)
                                                   + rep_matrix(linspaced_vector(N, 1, N), J);
        vector[n_responses] key_of_complete_data = to_vector(key_of_each_person_and_item);
        if (max(abs(key_of_response[order_by_item] - key_of_complete_data)) > 0) {
            reject("the model is defined for complete data: every person must answer every item once");
        }
    }
}
parameters {
    // vector[N] theta;                               // abilities: standard normal nuisance block, declared first
    // (the same block as an array, the sliced argument of reduce_sum_static)
    array[N] real theta;                             // abilities: standard normal nuisance block, declared first
    vector<lower=0>[J] a;                            // item discriminations
    array[J] ordered[K - 1] c;                       // item cutpoints (logit scale), ordered within each item
}
model {
    // Priors
    a ~ lognormal(0, 0.5);
    for (j in 1:J) {
        c[j] ~ nico_normal(0, 5);
    }

    // The abilities: the nuisance block's standard normal prior
    // theta ~ std_normal();
    // (added inside the chunks, person by person: see partial_sum_persons_by_item_lpdf)

    // Likelihood (reduce_sum_static over the persons, in chunks of chunk_size; the sliced argument is the
    // abilities theta, so that each chunk copies only its own persons' theta)
    // target += reduce_sum_static(partial_sum_persons_lupdf, to_array_1d(theta), chunk_size, a, c, person, item,
    //                             y, first_response, last_response);
    // (each chunk's responses grouped by item: see partial_sum_persons_by_item_lpdf)
    // target += reduce_sum_static(partial_sum_persons_by_item_lupdf, to_array_1d(theta), chunk_size, a, c,
    //                             y_by_item, person_by_item, item_offset, item_responses_before_person);
    // (theta sliced as it is declared, and each chunk's responses of each item one contiguous block of the
    //  complete data: see partial_sum_persons_by_item_of_complete_data_lpdf)
    target += reduce_sum_static(partial_sum_persons_by_item_of_complete_data_lupdf, theta, chunk_size, a, c,
                                y_by_item, N);
}


generated quantities {
    real simd_lanes = bmvp_simd_lanes();
}
