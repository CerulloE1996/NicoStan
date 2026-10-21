// graded_response_IRT_unchunked_no_loops_over_N_with_Stan_fused_functions.stan
// Unchunked plain Stan version of the graded response model of Paper 1's layout timings
// (graded_response_IRT_chunked_no_loops_over_N_automatic_reduce_sum_chunked, from its source
// graded_response_IRT_chunked_no_loops_over_N): the same log density as that source at N_chunks = 1, and as
// graded_response_IRT.stan, as one pass over all N persons, without reduce_sum_static. The same data
// (chunk_size, when it is in the data, is not read) and the same unconstrained parameters, in the same order and
// with the same names (theta, a, c), as both files; theta is a vector, as in graded_response_IRT.stan (the
// source's array[N] real theta, the same N values in the same order, exists only so that reduce_sum_static can
// slice the persons).
// The model is defined for complete data (every person answers every item once), as in the source: the
// responses are sorted by item, and then by person, once in transformed data (by sorting, with no loop), and the
// data are checked to be complete, so that item j's responses are the N responses y_by_item[(j - 1) N + 1] to
// y_by_item[j N], those of persons 1 to N in turn.
// The likelihood is Stan's fused ordered-logistic GLM, item by item: the abilities theta as the one column of x
// (an N x 1 matrix of the same values), the item's discrimination as its coefficient and the item's own
// cut-points; the loop is over the J items, whose number does not grow with N (one GLM per item, since each item
// has its own cut-points, and the GLM takes one cut-point vector).
// No loop over N anywhere.
//
// The model, as in graded_response_IRT.stan:
// Samejima's graded response model (GRM) for ordinal item responses, in its slope-intercept form on the
// logistic metric: N persons each answer every one of J items with one of K ordered categories 1, ..., K. For
// person n and item j, P(Y_nj >= k) = inv_logit(a_j theta_n - c_j,k-1), k = 2, ..., K, with the person's
// ability theta_n, the item's discrimination a_j > 0 and its ordered cutpoints c_j,1 < ... < c_j,K-1.
// Priors: a_j ~ lognormal(0, 0.5); every cutpoint c_jk ~ N(0, 5^2); theta_n ~ N(0, 1).


data {
    int<lower=1> N;                                  // number of persons
    int<lower=1> J;                                  // number of items
    int<lower=2> K;                                  // number of ordered response categories
    int<lower=1> n_responses;                        // number of responses (N J: complete data)
    array[n_responses] int<lower=1, upper=N> person; // person of each response (responses sorted by person)
    array[n_responses] int<lower=1, upper=J> item;   // item of each response
    array[n_responses] int<lower=1, upper=K> y;      // response category of each response
}


transformed data {
    // the responses sorted by item, and then by person, by the key item (N + 1) + person of each response; for
    // complete data the keys are distinct and, sorted, are those of (item 1, person 1), ..., (item 1, person N),
    // (item 2, person 1), ..., (item J, person N) (as in the chunked source)
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
    vector[N] theta;                                 // abilities: standard normal nuisance block, declared first
    vector<lower=0>[J] a;                            // item discriminations
    array[J] ordered[K - 1] c;                       // item cutpoints (logit scale), ordered within each item
}


model {
    // Priors
    a ~ lognormal(0, 0.5);
    for (j in 1:J) {
        c[j] ~ normal(0, 5);
    }

    // The abilities: the nuisance block's standard normal prior
    theta ~ std_normal();

    // Likelihood (Stan's fused ordered-logistic GLM, item by item; see the header)
    {
        matrix[N, 1] theta_as_matrix = to_matrix(theta);
        for (j in 1:J) {
            y_by_item[((j - 1) * N + 1):(j * N)] ~ ordered_logistic_glm(theta_as_matrix, a[j:j], c[j]);
        }
    }
}






















