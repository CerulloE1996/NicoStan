// shared_frailty_Cox_survival_one_fused_node_with_indexed_frailties_and_its_blocks_of_rows_run_in_parallel: the
// hand-written BayesMVP AVX version of
// shared_frailty_Cox_survival_direct_sliced_shared_offsets_one_fused_node_with_indexed_frailties_when_unchunked
// (the same model, the same log density), with the whole partial likelihood ONE node at EVERY number of chunks:
// BayesMVP's general nico_dot_products_of_affine_predictor_with_indexed_effects_and_log_cumulative_sum_exp, with
// its last argument rows_per_block_run_in_parallel = chunk_size
// (stan_external_dot_products_of_affine_predictor_with_indexed_effects_and_log_cumulative_sum_exp_in_parallel_
// blocks_of_rows.hpp): the observations in decreasing order of time are split into N_blocks blocks of chunk_size
// consecutive positions (rounded up to a multiple of 8 inside the node), which run in parallel on the chain's
// WCP threads, one task per block, as the chunks of reduce_sum_static; with one block, the node of the version it
// comes from. The two-level scan of the risk sets is made inside the node: (1) in parallel, each block's eta
// (the raw log frailties read by their group inside the node), its largest value and its sum of exponentials;
// (2) on the calling thread, a scan over the N_blocks block totals gives each block its offset (N_blocks numbers,
// never on Stan's tape); (3) in parallel, each block's running sums from its offset, the dot product of the
// events with log of them, and its sum of events / running sum; (4) on the calling thread, the backward scan of
// these N_blocks sums; (5) in parallel, each block's derivatives and their product with its rows of X; (6) in
// parallel over ranges of the groups, the derivative of each group's raw log frailty. Hence, against the version
// it comes from, with more than one block: no gather of the N raw log frailties on the main tape (raw_of_blocks),
// no pass over all N observations on the main tape for the offsets
// (nico_log_sum_exp_of_indexed_affine_predictor_in_segments), no reduce_sum with its copies of the N sliced
// values and of the data of each block (X_decreasing_time[first:last], -w_decreasing_time[first:last], ...) at
// every gradient: nothing on the main tape but the node, whose only O(N)-sized part is one partial per group (as
// with one block). The prior of the raw log frailties: one node (nico_normal_lupdf) with one block, and the
// chunks of reduce_sum_static over the parameter array itself (each group's prior in exactly one chunk) with
// more blocks, as before. Its Stan math library is
// shared_frailty_Cox_survival_one_fused_node_with_indexed_frailties_and_its_blocks_of_rows_run_in_parallel_Stan_
// math_for_layout_timings.stan.
//
// shared_frailty_Cox_survival_direct_sliced_shared_offsets_one_fused_node_with_indexed_frailties_when_unchunked:
// the hand-written BayesMVP AVX version of shared_frailty_Cox_survival_direct_sliced_shared_offsets (the same
// model, the same log density), but with one block (N_chunks = 1) the whole partial likelihood is ONE node,
// BayesMVP's general nico_dot_products_of_affine_predictor_with_indexed_effects_and_log_cumulative_sum_exp: the
// raw log frailties are read by their group inside the node (no vector of N raw log frailties on the main tape,
// no reduce_sum, one partial per group), with BayesMVP's exp and log kernels and SIMD scans. With more blocks,
// the version it comes from, unchanged. Its Stan math library is that of
// shared_frailty_Cox_survival_direct_sliced_shared_offsets.
//
// shared_frailty_Cox_survival_chunked_no_loops_over_N_v2_hand_written_BayesMVP_AVX.stan: the hand-written
// BayesMVP AVX version of shared_frailty_Cox_survival_chunked_no_loops_over_N_v2.stan (the same model): every
// element-wise call on a vector written by hand as BayesMVP's nico_exp and nico_log (exp and log of vectors),
// nico_log_sum_exp (log_sum_exp of a vector) and BayesMVP's fused nico_normal_lpdf (the priors on vectors); every
// scalar call stays on Stan's math library. BayesMVP's functions are declared in the functions block as
// NicoStan's AVX writer declares them; one text for AVX2 and AVX-512 (the instruction set comes from the compiler
// flags, as for NicoStan's AVX versions), and simd_lanes (generated quantities) reports the SIMD lanes compiled
// (4 for AVX2, 8 for AVX-512).
//
// Cox partial likelihood with lognormal shared frailty. Ties use the Breslow form.
// Tape-chunked version of shared_frailty_Cox_survival.stan, the O(N) model, with no loop over the observations
// in the log density, in the pattern of LC_MVP_bin_PartialLog_v5_reduce_sum_static.stan: the same log density,
// with every computation whose cost or tape grows with N vectorised and inside the chunks, and nothing whose
// size grows with N on the main tape or shared by the chunks. The observations, sorted by decreasing time once
// (transformed data), are split into N_blocks = ceiling(N / chunk_size) blocks of chunk_size consecutive
// positions (the last one possibly shorter): the units of reduce_sum_static, with grainsize 1 (one block per
// chunk). The units are the blocks, not the observations, so that each chunk starts at a block, whose offset
// from (2) below is known (with grainsize chunk_size over the observations, reduce_sum_static's chunks need not
// start at the blocks).
// As in LC_MVP_bin_PartialLog_v5_reduce_sum_static.stan: the sliced argument is one single name holding
// everything per unit, packed_of_block (one row per block, packed once on the main tape: copies of the
// parameters' pointers, no autodiff nodes), of N_packed = chunk_size + N_prior_slots + 1 values:
//   - the raw log frailties of the block's observations, log_frailty_raw[group] in decreasing order of time (the
//     last block padded with log_frailty_raw[1], not used);
//   - the raw log frailties of the block's share of the J groups for their prior, normal(0, 1): the groups
//     (b - 1) * N_prior_slots + 1 to min(b * N_prior_slots, J), with N_prior_slots = ceiling(J / N_blocks), so
//     that the prior of every group (with observations or not) is in exactly one chunk (padded with
//     log_frailty_raw[1], not used);
//   - the block's offset, the log of the sum of exp(eta) over every observation before it (block 1: none);
// every shared argument is a single name, data or O(1) parameters (beta and sigma_frailty); nothing is built in
// the calls of reduce_sum_static and map_rect; and inside the chunks, whole-vector calls only, with no loop over
// the observations or the blocks: each chunk computes its observations' log frailties,
// sigma_frailty * log_frailty_raw, itself. The main tape holds only the priors on beta and sigma_frailty
// (K + 1 values), the packing, the N_blocks - 1 block totals and their scan, and the one node of
// reduce_sum_static. The scan over the risk sets is chunked as a two-level scan:
//   (1) map_rect computes the log of the sum of exp(eta) over each block before the last one, from beta and
//       sigma_frailty (shared, a named vector) and the raw log frailties of the block's observations (its job
//       parameters, a named array);
//   (2) a scan over these N_blocks - 1 block totals, vectorised (log_cumulative_sum_exp), gives each block its
//       offset (O(N_blocks) on the main tape), packed into its row;
//   (3) reduce_sum_static, over packed_of_block, computes its blocks' eta again, their numerators, the scan from
//       the first block's offset, vectorised over the observations (log_cumulative_sum_exp:
//       shift + log(cumulative_sum(exp(eta - shift))), shift the largest of the eta and the offset), the
//       denominators, and the prior of the blocks' share of the groups.
// eta = X * beta + sigma_frailty * log_frailty_raw is computed with one fused multiply-add over the
// observations, fma(sigma_frailty, raw, X * beta), in map_rect's jobs and in the chunks (one node per observation
// in place of two, a product and a sum); and the chunks' denominators are written with the shift outside the log,
// shift * sum(w) + dot_product(w, log(cumulative sum)) (weighted_sum_of_log_cumulative_sum_exp), so that the
// shift is not added at every position.
// log_frailty and frailty, transformed parameters before, are generated quantities (the same values), so that
// neither is computed on the main tape at each log density; the generated quantities are vectorised too.
// The one loop over the observations left is in transformed data, and only when some survival times are tied:
// the tie groups (each observation's risk set ends at the last observation tied with it) need a comparison of
// neighbouring sorted times and a running "last position of the group", for which Stan has no vectorised
// function (no element-wise comparison of vectors, cumulative maximum or "which"); it runs once, on the data,
// when the model is constructed, and adds nothing to any log density or gradient. Without ties (every data set
// of the benchmark) it is not run.
// (Now, in this second version: no loop over the observations or the blocks is left, in transformed data or in
//  the model block, with the same values: the tie groups come from the positions after which the sorted times
//  decrease (sort_indices_desc() of the differences of neighbouring sorted times, counted with rank()), each
//  position's tie group is a cumulative sum of the groups' starts, and each group's events are differences of
//  the cumulative sum of the events; the blocks' first and last positions, their shares of the groups and
//  map_rect's data are whole-vector calculations on the data; the sliced argument of reduce_sum_static is a
//  two-dimensional array (one row per block), made with to_array_2d() from the matrix of the packed values; and
//  map_rect's job parameters, an array of vectors, come from the columns of a matrix by
//  columns_as_array_of_vectors(), which splits it by halves, as Stan has no function that makes an array of
//  vectors from a matrix.)
// The chunking data are NicoStan's for Stan models: N (the observations, the units of the chunks; already in the
// data) and chunk_size, which NicoStan sets to ceiling(N / N_chunks) in each phase (num_chunks_burnin,
// num_chunks_sampling), so that N_blocks is N_chunks (or fewer).
//
// The previous tape-chunked model (log_frailty and frailty transformed parameters, and the prior of
// log_frailty_raw, on the main tape; each block's offset, all N_blocks of them, a shared argument copied into
// every chunk; and the scalar log_sum_exp recursion within each block) is kept below, commented out; its header
// was (the next 17 lines):
// Tape-chunked version of shared_frailty_Cox_survival.stan, the O(N) model: the same log density, with every
// computation whose cost or tape grows with N inside the chunks, and nothing whose size grows with N shared by
// the chunks. The observations, sorted by decreasing time once (transformed data), are split into
// N_blocks = ceiling(N / chunk_size) blocks of chunk_size consecutive positions (the last one possibly shorter);
// each block's observations' log frailties, log_frailty[group] in that order (log_frailty_of_block[b], the last
// block padded with log_frailty[1], not used), are the sliced argument of reduce_sum_static and map_rect's job
// parameters, so that each chunk copies only its own block's log frailties (and not all J of them), and each
// map_rect job's Jacobian holds only beta and its own block's log frailties. The scan over the risk sets is
// chunked as a two-level scan:
//   (1) map_rect computes the log of the sum of exp(eta) over each block before the last one, from beta (shared)
//       and the block's log frailties (its job parameters);
//   (2) a scan over these N_blocks - 1 block totals gives each block its offset (O(N_blocks) on the main tape);
//   (3) reduce_sum_static, over log_frailty_of_block with grainsize 1 (one block per chunk), computes each
//       block's eta again, its numerators, and its own part of the scan from its offset with the scalar
//       log_sum_exp recursion (one node per observation, as in shared_frailty_Cox_survival.stan), and its
//       denominators.
//
// The previous tape-chunked model (log_frailty, all J values, shared by every chunk and every map_rect job; the
// scan within each block vectorised) is kept below, commented out; its header was (the next 19 lines):
// Tape-chunked version of shared_frailty_Cox_survival.stan, the O(N) model: the same log density, with every
// computation whose cost or tape grows with N inside the chunks. The observations, sorted by decreasing time
// once (transformed data), are split into N_blocks = ceiling(N / chunk_size) blocks of chunk_size consecutive
// positions (the last one possibly shorter), and the scan over the risk sets (the log of the cumulative sum of
// exp(eta) = exp(X * beta + log_frailty[group]) in that order) is chunked as a two-level scan:
//   (1) map_rect computes the log of the sum of exp(eta) over each block before the last one (each block's eta
//       and log_sum_exp on its own nested autodiff tape; one node per block on the main tape);
//   (2) a scan over these N_blocks - 1 block totals gives each block its offset, the log of the sum of exp(eta)
//       over every observation before it (O(N_blocks) on the main tape);
//   (3) reduce_sum_static, over the block indices with grainsize 1 (one block per chunk), computes each block's
//       eta again, its numerators, its own part of the scan started from its offset, and its denominators.
// The partial likelihood is written as a weighted sum over the positions in decreasing order of time,
// sum(event * eta) - sum(w * log_cumulative_risk), with w[k] the number of events whose risk set ends at
// position k (transformed data), so that no block needs values from another block beyond its offset. Within each
// block the scan is vectorised over the block's observations: shift + log(cumulative_sum(exp(eta - shift))), with
// shift the largest of the block's eta and its offset (so that no term overflows); if its first value underflows
// (eta spread over more than about 690 within a block, far outside the posterior), the block uses the scalar
// log_sum_exp recursion of the previous model instead. Both give the same values up to floating-point rounding.
// The chunking data are NicoStan's for Stan models: N (the observations, the units of the chunks; already in the
// data) and chunk_size, which NicoStan sets to ceiling(N / N_chunks) in each phase (num_chunks_burnin,
// num_chunks_sampling), so that N_blocks is N_chunks (or fewer).
//
// The previous tape-chunked model (the scan outside reduce_sum_static) is kept below, commented out; its header
// was (the next 28 lines):
// Tape-chunked version of shared_frailty_Cox_survival.stan, the O(N) model: the same log density. Its sequential
// part, the scan over the risk sets (the log of the cumulative sum of exp(X * beta + frailty) in decreasing order
// of time, each value from the previous one), stays outside reduce_sum_static, exactly as in
// shared_frailty_Cox_survival.stan; each observation's term of the partial likelihood after the scan,
// event[n] * (eta[n] - the log of the sum of exp(eta) over its risk set), is summed over the observations by
// reduce_sum_static, in chunks of chunk_size observations (data), so that each chunk's autodiff tape is built,
// differentiated and released before the next one (the tape-chunking paper's chunking; N_chunks from
// NicoStan::fn_compute_initial_N_chunks()). The observations' values eta[n] - (the log of the sum over the risk
// set) are the sliced argument of reduce_sum_static, so that each chunk copies only its own observations' values,
// and not whole vectors, onto its tape.
// The chunking data are NicoStan's for Stan models: N (the observations, the units of the chunks; already in the
// data) and chunk_size, which NicoStan sets to ceiling(N / N_chunks) in each phase (num_chunks_burnin,
// num_chunks_sampling).
//
// The model, as in shared_frailty_Cox_survival.stan: the O(N) version of NicoStan's example
// (inst/examples/models/Frailty_surv_shared_frailty_Cox.stan), whose denominator sums over all N observations for
// every event (O(N^2) per log density and gradient): the observations are sorted by decreasing time once
// (transformed data), the log of each risk set's sum of exp(X * beta + frailty) is a cumulative log_sum_exp in
// that order, and each observation takes the value at the last observation of its risk set (every j with
// t[j] >= t[n], ties included).
//
// The tape-chunked model of NicoStan's O(N^2) example is kept below, commented out; its header was (the next five
// lines):
// Tape-chunked version of Frailty_surv_shared_frailty_Cox.stan: the same log density, with the
// partial likelihood summed over the events by reduce_sum_static, in chunks of grainsize events (data), so that
// each chunk's autodiff tape is built, differentiated and released before the next one (the
// chunking paper's chunking;
// N_chunks from NicoStan::fn_compute_initial_N_chunks()). Each event's term is computed exactly as in the
// original model (the numerator, then log_sum_exp over its risk set, in the order j = 1, ..., N).

functions {
    real nico_dot_products_of_affine_predictor_with_indexed_effects_and_log_cumulative_sum_exp(
        data vector linear_weights, data vector prefix_weights, data matrix X, vector beta, real scale,
        array[] real z, data array[] int index, data array[] int rows_in_order_of_index,
        data array[] int last_position_of_each_index, real log_offset);
    // (the same, with its blocks of rows_per_block_run_in_parallel rows run in parallel on the chain's threads)
    real nico_dot_products_of_affine_predictor_with_indexed_effects_and_log_cumulative_sum_exp(
        data vector linear_weights, data vector prefix_weights, data matrix X, vector beta, real scale,
        array[] real z, data array[] int index, data array[] int rows_in_order_of_index,
        data array[] int last_position_of_each_index, real log_offset,
        data int rows_per_block_run_in_parallel);
    vector nico_log_sum_exp_of_indexed_affine_predictor_in_segments(
        data matrix x, vector beta, real scale, array[] real values,
        data array[] int index, data array[] int first, data array[] int last);
    real bmvp_simd_lanes();
    vector nico_exp(vector x);
    vector nico_log(vector x);
    real nico_normal_lpdf(vector y, vector mu, vector sigma);
    real nico_normal_lpdf(vector y, vector mu, real sigma);
    real nico_normal_lpdf(vector y, real mu, vector sigma);
    real nico_normal_lpdf(vector y, real mu, real sigma);
    real nico_normal_lpdf(array[] real y, vector mu, vector sigma);
    real nico_normal_lpdf(array[] real y, vector mu, real sigma);
    real nico_normal_lpdf(array[] real y, real mu, vector sigma);
    real nico_normal_lpdf(array[] real y, real mu, real sigma);
    real nico_normal_lpdf(real y, real mu, real sigma);
    real nico_log_sum_exp(vector x);
    real nico_log_sum_exp(row_vector x);
    real nico_log_sum_exp(matrix x);
    real nico_log_sum_exp(array[] real x);
    real nico_log_sum_exp(real a, real b);
    vector nico_log_sum_exp(vector a, vector b);
    vector nico_log_sum_exp(real a, vector b);
    vector nico_log_sum_exp(vector a, real b);
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
    // floating-point rounding.
    vector nico_log_cumulative_sum_exp(vector x, real log_offset);
    real nico_dot_product_log_cumulative_sum_exp(vector w, vector x, real log_offset);
    real nico_dot_products_of_affine_predictor_and_log_cumulative_sum_exp(
        vector linear_weights, vector prefix_weights, data matrix x, vector beta,
        real scale, vector z, real log_offset);
    real nico_log_sum_exp_of_affine_predictor(data matrix x, vector beta, real scale, vector z);
    vector log_cumulative_sum_exp(vector x, real log_offset) {
        int M = rows(x);
        real shift = fmax(max(x), log_offset);
        vector[M] terms_shifted = nico_exp(x - shift);
        terms_shifted[1] += exp(log_offset - shift);
        vector[M] cumulative_sum_shifted = cumulative_sum(terms_shifted);
        if (cumulative_sum_shifted[1] >= 1e-280) {
            return shift + nico_log(cumulative_sum_shifted);
        }
        int M_small = rank(append_row(cumulative_sum_shifted, 1e-280), M + 1);
        vector[M] log_cumulative_sum;
        log_cumulative_sum[1:M_small] = log_cumulative_sum_exp(x[1:M_small], log_offset);
        if (M_small < M) {
            log_cumulative_sum[(M_small + 1):M] = shift + nico_log(cumulative_sum_shifted[(M_small + 1):M]);
        }
        return log_cumulative_sum;
    }
    // dot_product(w, log_cumulative_sum_exp(x, log_offset)), written with the shift outside the log:
    // shift * sum(w) + dot_product(w, log(cumulative_sum_shifted)) (as log_cumulative_sum_exp otherwise, and
    // through it in its rare case)
    real weighted_sum_of_log_cumulative_sum_exp(data vector w, vector x, real log_offset) {
        int M = rows(x);
        real shift = fmax(max(x), log_offset);
        vector[M] terms_shifted = nico_exp(x - shift);
        terms_shifted[1] += exp(log_offset - shift);
        vector[M] cumulative_sum_shifted = cumulative_sum(terms_shifted);
        if (cumulative_sum_shifted[1] >= 1e-280) {
            return shift * sum(w) + dot_product(w, nico_log(cumulative_sum_shifted));
        }
        return dot_product(w, log_cumulative_sum_exp(x, log_offset));
    }
    // (1) the log of the sum of exp(eta) over one block of chunk_size observations (a map_rect job), with
    // eta = X_block * beta + sigma_frailty * the block's observations' raw log frailties (the job's own
    // parameters): beta and sigma_frailty (shared), the block's rows of X in decreasing order of time, packed
    // column by column (X_block_by_column), and K and chunk_size
    vector log_sum_exp_eta_of_block_from_its_raw_log_frailties(vector beta_and_sigma_frailty,
                                                               vector log_frailty_raw_of_block,
                                                               data array[] real X_block_by_column,
                                                               data array[] int K_and_chunk_size) {
        int K = K_and_chunk_size[1];
        int M = K_and_chunk_size[2];
        // vector[M] eta = to_matrix(X_block_by_column, M, K) * beta_and_sigma_frailty[1:K]
        //                 + beta_and_sigma_frailty[K + 1] * log_frailty_raw_of_block;
        // (one fused multiply-add over the block, in place of a product and a sum: two nodes fewer per
        //  observation)
        // vector[M] eta = fma(beta_and_sigma_frailty[K + 1], log_frailty_raw_of_block,
        //                     to_matrix(X_block_by_column, M, K) * beta_and_sigma_frailty[1:K]);
        // return [nico_log_sum_exp(eta)]';
        return [nico_log_sum_exp_of_affine_predictor(to_matrix(X_block_by_column, M, K),
                                                     beta_and_sigma_frailty[1:K],
                                                     beta_and_sigma_frailty[K + 1],
                                                     log_frailty_raw_of_block)]';
    }
    // the columns first to last of a matrix as an array of vectors (map_rect's job parameters), split by halves,
    // so that no loop over the columns is needed (Stan has no function that makes an array of vectors from a
    // matrix); the values are copies, with no autodiff node
    array[] vector columns_as_array_of_vectors(matrix m, int first, int last) {
        if (first == last) {
            return { col(m, first) };
        }
        int middle = (first + last) %/% 2;
        return append_array(columns_as_array_of_vectors(m, first, middle),
                            columns_as_array_of_vectors(m, middle + 1, last));
    }
    // (3) the terms of the blocks start to end (one block per chunk with grainsize 1; any range of consecutive
    // blocks gives the same sum), with whole-vector calls over their observations, which are consecutive in
    // decreasing order of time: the blocks' packed rows as the rows of one matrix; the observations' eta (the
    // rows' first chunk_size values, row after row, without the last block's padding, times sigma_frailty); the
    // numerators sum(event * eta); the scan from the first block's offset (the last value of its row; none for
    // block 1), vectorised (weighted_sum_of_log_cumulative_sum_exp), weighted by w; and the prior of the blocks'
    // share of the
    // groups (the rows' values chunk_size + 1 to N_packed - 1, row after row: every block holds N_prior_slots
    // groups until the groups run out, so the groups held are the first sum(N_prior_groups_of_block[start:end]))
    // real partial_sum_packed_blocks_lpdf(array[] row_vector packed_of_block_slice, int start, int end,
    // (the sliced argument as a two-dimensional array, one row per block, which the model block makes from a
    //  matrix without a loop; Stan has no function that makes an array of row vectors from a matrix)
    real partial_sum_packed_blocks_lpdf(array[,] real packed_of_block_slice, int start, int end,
                                        data matrix X_decreasing_time,
                                        data vector event_decreasing_time,
                                        data vector w_decreasing_time,
                                        data array[] int block_first,
                                        data array[] int block_last,
                                        data array[] int N_prior_groups_of_block,
                                        data int chunk_size_of_blocks,
                                        vector beta,
                                        real sigma_frailty) {
        int first = block_first[start];
        int last = block_last[end];
        int M = last - first + 1;
        int N_blocks_of_slice = end - start + 1;
        // int N_packed = cols(packed_of_block_slice[1]);
        int N_packed = size(packed_of_block_slice[1]);
        int N_prior_groups_of_slice = sum(N_prior_groups_of_block[start:end]);
        matrix[N_blocks_of_slice, N_packed] packed_of_slice = to_matrix(packed_of_block_slice);

        // the observations' eta and the numerators
        vector[M] log_frailty_raw_of_slice = to_vector(packed_of_slice[:, 1:chunk_size_of_blocks]')[1:M];
        // vector[M] eta = X_decreasing_time[first:last] * beta + sigma_frailty * log_frailty_raw_of_slice;
        // vector[M] eta = fma(sigma_frailty, log_frailty_raw_of_slice, X_decreasing_time[first:last] * beta);
        // real lp = dot_product(event_decreasing_time[first:last], eta);
        real lp;

        // the scan from the first block's offset, and the denominators
        real log_offset = negative_infinity();
        if (start > 1) {
            log_offset = packed_of_slice[1, N_packed];
        }
        // lp -= dot_product(w_decreasing_time[first:last], log_cumulative_sum_exp(eta, log_offset));
        // lp -= weighted_sum_of_log_cumulative_sum_exp(w_decreasing_time[first:last], eta, log_offset);
        // (the same values, with BayesMVP's log_cumulative_sum_exp: one autodiff step for the whole vector)
        // lp -= dot_product(w_decreasing_time[first:last], nico_log_cumulative_sum_exp(eta, log_offset));
        // lp -= nico_dot_product_log_cumulative_sum_exp(w_decreasing_time[first:last], eta, log_offset);
        lp = nico_dot_products_of_affine_predictor_and_log_cumulative_sum_exp(
            event_decreasing_time[first:last], -w_decreasing_time[first:last],
            X_decreasing_time[first:last], beta, sigma_frailty, log_frailty_raw_of_slice, log_offset);

        // the prior of the blocks' share of the groups
        if (N_prior_groups_of_slice > 0) {
            matrix[N_blocks_of_slice, N_packed - chunk_size_of_blocks - 1] prior_values_of_slice
                = packed_of_slice[:, (chunk_size_of_blocks + 1):(N_packed - 1)];
            vector[N_prior_groups_of_slice] log_frailty_raw_of_prior_groups
                = to_vector(prior_values_of_slice')[1:N_prior_groups_of_slice];
            lp += nico_normal_lupdf(log_frailty_raw_of_prior_groups | 0, 1);
        }
        return lp;
    }
    // (1) the log of the sum of exp(eta) over one block of chunk_size observations (a map_rect job): the block's
    // rows of X in decreasing order of time, packed column by column (X_block_by_column), and K followed by the
    // positions of the block's observations' log frailties in
    // beta_and_log_frailty = append_row(beta, log_frailty)
    // vector log_sum_exp_eta_of_block(vector beta_and_log_frailty, vector no_job_parameters,
    //                                 data array[] real X_block_by_column,
    //                                 data array[] int K_and_log_frailty_index) {
    //     int K = K_and_log_frailty_index[1];
    //     int M = size(K_and_log_frailty_index) - 1;
    //     vector[M] eta = to_matrix(X_block_by_column, M, K) * beta_and_log_frailty[1:K]
    //                     + beta_and_log_frailty[K_and_log_frailty_index[2:(M + 1)]];
    //     return [log_sum_exp(eta)]';
    // }
    // (1) the log of the sum of exp(eta) over one block of chunk_size observations (a map_rect job), with
    // eta = X_block * beta + the block's observations' log frailties (the job's own parameters): the block's rows
    // of X in decreasing order of time, packed column by column (X_block_by_column), and K
    // vector log_sum_exp_eta_of_block(vector beta, vector log_frailty_of_block,
    //                                 data array[] real X_block_by_column,
    //                                 data array[] int K_of_block) {
    //     int K = K_of_block[1];
    //     int M = rows(log_frailty_of_block);
    //     vector[M] eta = to_matrix(X_block_by_column, M, K) * beta + log_frailty_of_block;
    //     return [log_sum_exp(eta)]';
    // }
    // // (3) the terms of the partial likelihood of the blocks in block_slice (one block per chunk): each block's
    // // eta, its numerators sum(event * eta), and its own part of the scan from its offset
    // // log_risk_before_block[b]
    // // (the log of the sum of exp(eta) over every observation before the block), weighted by w
    // real partial_sum_blocks(array[] int block_slice, int start, int end,
    //                         data matrix X_decreasing_time,
    //                         data array[] int group_decreasing_time,
    //                         data vector event_decreasing_time,
    //                         data vector w_decreasing_time,
    //                         data array[] int block_first,
    //                         data array[] int block_last,
    //                         data vector w_sum_of_block,
    //                         vector log_risk_before_block,
    //                         vector beta,
    //                         vector log_frailty) {
    //     real lp = 0;
    //     for (i in 1:size(block_slice)) {
    //         int b = block_slice[i];
    //         int first = block_first[b];
    //         int last = block_last[b];
    //         int M = last - first + 1;
//
    //         // the block's eta and its numerators
    //         vector[M] eta = X_decreasing_time[first:last] * beta
    //                         + log_frailty[group_decreasing_time[first:last]];
    //         lp += dot_product(event_decreasing_time[first:last], eta);
//
    //         // the block's part of the scan, vectorised over its observations: the log of the sum of
    //         // exp(eta) over every observation up to each position, shift + log(cumulative_risk_shifted), with
    //         // the blocks before it (exp(log_risk_before_block[b] - shift)) added at its first position
    //         real shift = max(eta);
    //         if (b > 1) {
    //             shift = fmax(shift, log_risk_before_block[b]);
    //         }
    //         vector[M] risk_shifted = exp(eta - shift);
    //         if (b > 1) {
    //             risk_shifted[1] += exp(log_risk_before_block[b] - shift);
    //         }
    //         vector[M] cumulative_risk_shifted = cumulative_sum(risk_shifted);
//
    //         // the denominators: sum(w * (shift + log(cumulative_risk_shifted))) over the block's positions;
    //         // the scalar log_sum_exp recursion instead if the first (smallest) value of the cumulative sum
    //         // underflows
    //         if (cumulative_risk_shifted[1] > 1e-300) {
    //             lp -= shift * w_sum_of_block[b]
    //                   + dot_product(w_decreasing_time[first:last], log(cumulative_risk_shifted));
    //         } else {
    //             real log_cumulative_risk = eta[1];
    //             if (b > 1) {
    //                 log_cumulative_risk = log_sum_exp(log_risk_before_block[b], eta[1]);
    //             }
    //             lp -= w_decreasing_time[first] * log_cumulative_risk;
    //             for (k in 2:M) {
    //                 log_cumulative_risk = log_sum_exp(log_cumulative_risk, eta[k]);
    //                 lp -= w_decreasing_time[first + k - 1] * log_cumulative_risk;
    //             }
    //         }
    //     }
    //     return lp;
    // }
    // (3) the terms of the partial likelihood of the blocks start to end (one block per chunk): each block's eta
    // (its observations' log frailties from the sliced argument), its numerators sum(event * eta), and its own
    // part of the scan from its offset log_risk_before_block[b] (the log of the sum of exp(eta) over every
    // observation before the block), with the scalar log_sum_exp recursion, weighted by w
    // real partial_sum_blocks(array[] vector log_frailty_of_block_slice, int start, int end,
    //                         data matrix X_decreasing_time,
    //                         data vector event_decreasing_time,
    //                         data vector w_decreasing_time,
    //                         data array[] int block_first,
    //                         data array[] int block_last,
    //                         vector log_risk_before_block,
    //                         vector beta) {
    //     real lp = 0;
    //     for (b in start:end) {
    //         int first = block_first[b];
    //         int last = block_last[b];
    //         int M = last - first + 1;
    //
    //         // the block's eta and its numerators
    //         vector[M] eta = X_decreasing_time[first:last] * beta
    //                         + log_frailty_of_block_slice[b - start + 1][1:M];
    //         lp += dot_product(event_decreasing_time[first:last], eta);
    //
    //         // the block's part of the scan: the log of the sum of exp(eta) over every observation up to each
    //         // position, from the blocks before it (log_risk_before_block[b]), and the denominators
    //         vector[M] log_cumulative_risk;
    //         if (b > 1) {
    //             log_cumulative_risk[1] = log_sum_exp(log_risk_before_block[b], eta[1]);
    //         } else {
    //             log_cumulative_risk[1] = eta[1];
    //         }
    //         for (k in 2:M) {
    //             log_cumulative_risk[k] = log_sum_exp(log_cumulative_risk[k - 1], eta[k]);
    //         }
    //         lp -= dot_product(w_decreasing_time[first:last], log_cumulative_risk);
    //     }
    //     return lp;
    // }
    // real partial_sum_observations(array[] real eta_minus_log_risk_set_sum_slice, int start, int end,
    //                               vector event_indicator) {
    //     return dot_product(event_indicator[start:end], to_vector(eta_minus_log_risk_set_sum_slice));
    // }
    // real partial_sum_events(array[] int events_slice, int start, int end,
    //                         matrix X, vector beta, vector frailty, vector t) {
    //     int N = rows(X);
    //     real lp = 0;
    //     for (i in 1:size(events_slice)) {
    //       int n = events_slice[i];
    //       // numerator
    //       lp += X[n] * beta + frailty[n];
    //
    //       // denominator
    //       real log_sum_exp_denom = negative_infinity();
    //       for (j in 1:N) {
    //         if (t[j] >= t[n]) {  // j is at risk at time t[n]
    //           log_sum_exp_denom = log_sum_exp(log_sum_exp_denom,
    //                                         X[j] * beta + frailty[j]);
    //         }
    //       }
    //       lp += -log_sum_exp_denom;
    //     }
    //     return lp;
    // }
    // The parameter array itself is sliced for its independent standard-normal prior.
    real fn_density_of_frailty_blocks_lpdf(array[] vector raw_of_block_slice, int start, int end,
                                            data matrix X_decreasing_time,
                                            data vector event_decreasing_time,
                                            data vector w_decreasing_time,
                                            data array[] int block_first, data array[] int block_last,
                                            vector beta, real sigma_frailty, vector offsets) {
        real lp = 0;
        for (b in start:end) {
            int first = block_first[b];
            int last = block_last[b];
            int length_of_block = last - first + 1;
            vector[length_of_block] raw_of_block = raw_of_block_slice[b - start + 1][1:length_of_block];
            real log_offset_of_block = b == 1 ? negative_infinity() : offsets[b];
            lp += nico_dot_products_of_affine_predictor_and_log_cumulative_sum_exp(
                event_decreasing_time[first:last], -w_decreasing_time[first:last],
                X_decreasing_time[first:last], beta, sigma_frailty, raw_of_block, log_offset_of_block);
        }
        return lp;
    }

    real fn_prior_of_frailty_slice_lpdf(array[] real frailty_slice, int start, int end) {
        return nico_normal_lupdf(to_vector(frailty_slice) | 0, 1);
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
    // int<lower=1> grainsize;            // events per chunk (reduce_sum_static)
    int<lower=1> chunk_size;           // observations per chunk (reduce_sum_static)
}


transformed data {
    // the observations in decreasing order of time, and the size of each observation's risk set (the number of
    // observations j with t[j] >= t[n], ties included: its position in that order, moved to the last of its ties)
    array[N] int order_decreasing_time = sort_indices_desc(t);
    array[N] int risk_set_size;
    // {
    //   int k = 1;
    //   while (k <= N) {
    //     int k_last = k;
    //     while (k_last < N && t[order_decreasing_time[k_last + 1]] == t[order_decreasing_time[k]]) {
    //       k_last += 1;
    //     }
    //     for (m in k:k_last) {
    //       risk_set_size[order_decreasing_time[m]] = k_last;
    //     }
    //     k = k_last + 1;
    //   }
    // }
    // // the observations with an event, in their original order
    // int N_events = sum(event);
    // array[N_events] int event_index;
    // {
    //   int k = 1;
    //   for (n in 1:N) {
    //     if (event[n] == 1) {
    //       event_index[k] = n;
    //       k += 1;
    //     }
    //   }
    // }
    // // the event indicators as a vector (each observation's weight in its chunk's sum)
    // vector[N] event_indicator = to_vector(event);

    // the observations in decreasing order of time (their rows of X, groups and event indicators), and
    // w_decreasing_time[k], the number of events whose risk set ends at position k (each event's denominator is
    // the log of the sum of exp(eta) at position risk_set_size[n])
    // (and their times in that order, t_decreasing_time, for the tie groups)
    matrix[N, K] X_decreasing_time = X[order_decreasing_time];
    array[N] int group_decreasing_time = group[order_decreasing_time];
    vector[N] event_decreasing_time = to_vector(event[order_decreasing_time]);
    vector[N] t_decreasing_time = t[order_decreasing_time];
    vector[N] w_decreasing_time;
    // vector[N] w_decreasing_time = rep_vector(0, N);
    // for (i in 1:N_events) {
    //   w_decreasing_time[risk_set_size[event_index[i]]] += 1;
    // }
    // (without ties, i.e. the times strictly decreasing in that order: each observation's risk set ends at its
    //  own position, and w is the event indicators)
    if (N < 2 || min(t_decreasing_time[1:(N - 1)] - t_decreasing_time[2:N]) > 0) {
      risk_set_size[order_decreasing_time] = linspaced_int_array(N, 1, N);
      w_decreasing_time = event_decreasing_time;
    } else {
      // (with ties: the risk sets of each tie group's observations end at the group's last position, which takes
      //  the group's events; the one loop over the observations left, see the header)
      // w_decreasing_time = rep_vector(0, N);
      // int k = 1;
      // while (k <= N) {
      //   int k_last = k;
      //   while (k_last < N && t_decreasing_time[k_last + 1] == t_decreasing_time[k]) {
      //     k_last += 1;
      //   }
      //   risk_set_size[order_decreasing_time[k:k_last]] = rep_array(k_last, k_last - k + 1);
      //   w_decreasing_time[k_last] = sum(event_decreasing_time[k:k_last]);
      //   k = k_last + 1;
      // }
      // (the same without a loop: the last positions of the tie groups are the positions after which the sorted
      //  times decrease (a positive difference to the next time: rank() counts them, sort_indices_desc() of the
      //  differences puts them first and sort_asc() puts them in order), and position N; each position's tie
      //  group is the cumulative sum of the groups' starts; and each group's events, taken at its last position,
      //  are the differences of the cumulative sum of the events at the groups' last positions)
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
    // (one block: the rows in order of decreasing time sorted by their group, the number of rows of the groups
    //  1, ..., g (from the ranks of g + 0.5 among the groups and the half points, as observations_before in
    //  joint_longitudinal_survival_chunked_no_loops_over_N.stan), and minus the events of each risk set)
    array[N] int rows_in_order_of_group = sort_indices_asc(group_decreasing_time);
    array[N + J + 1] int position_in_sorted_order_of_groups_and_half_points
          = sort_indices_asc(sort_indices_asc(append_row(to_vector(group_decreasing_time),
                                                         linspaced_vector(J + 1, 0, J) + 0.5)));
    array[J] int last_position_of_each_group
          = to_int(to_array_1d(to_vector(position_in_sorted_order_of_groups_and_half_points[(N + 2):(N + J + 1)])
                               - linspaced_vector(J, 2, J + 1)));
    vector[N] minus_w_decreasing_time = -w_decreasing_time;

    // the blocks: N_blocks = ceiling(N / chunk_size) blocks of chunk_size consecutive positions in decreasing
    // order of time (the last one possibly shorter), each one chunk of reduce_sum_static (whose sliced argument
    // is block_index, with grainsize 1), and the sum of w over each block
    // (now: the sliced argument is packed_of_block, below, with grainsize 1; block_index and w_sum_of_block are
    //  no longer used, kept commented out)
    int N_blocks = (N + chunk_size - 1) %/% chunk_size;
    // array[N_blocks] int block_index;
    array[N_blocks] int block_first;
    array[N_blocks] int block_last;
    // vector[N_blocks] w_sum_of_block;
    // for (b in 1:N_blocks) {
    //   // block_index[b] = b;
    //   block_first[b] = (b - 1) * chunk_size + 1;
    //   block_last[b] = min(b * chunk_size, N);
    //   // w_sum_of_block[b] = sum(w_decreasing_time[block_first[b]:block_last[b]]);
    // }
    // (the same without a loop: the first positions 1, chunk_size + 1, 2 * chunk_size + 1, ..., and the last
    //  positions chunk_size, 2 * chunk_size, ..., the last block's being N)
    if (N_blocks > 0) {
      block_first = to_int(to_array_1d(linspaced_vector(N_blocks, 0, N_blocks - 1) * chunk_size + 1));
      block_last = to_int(to_array_1d(linspaced_vector(N_blocks, 1, N_blocks) * chunk_size));
      block_last[N_blocks] = N;
    }

    // each block's share of the J groups for their prior, normal(0, 1): the groups (b - 1) * N_prior_slots + 1 to
    // min(b * N_prior_slots, J), with N_prior_slots = ceiling(J / N_blocks), so that every group (with
    // observations or not) is in exactly one block (the last blocks may hold fewer, or none)
    // the groups of the values of each block's packed row, its sliced argument: its observations' groups in
    // decreasing order of time (the last block padded with group 1, not used), then its share of the groups for
    // their prior (padded with group 1, not used); the packed row holds the block's offset after these
    // array[N_blocks, chunk_size + N_prior_slots] int group_of_packed_value;
    // for (b in 1:N_blocks) {
    //   int M = block_last[b] - block_first[b] + 1;
    //   int first_prior_group = (b - 1) * N_prior_slots + 1;
    //   N_prior_groups_of_block[b] = max(0, min(b * N_prior_slots, J) - first_prior_group + 1);
    //   group_of_packed_value[b] = rep_array(1, chunk_size + N_prior_slots);
    //   group_of_packed_value[b, 1:M] = group_decreasing_time[block_first[b]:block_last[b]];
    //   if (N_prior_groups_of_block[b] > 0) {
    //     group_of_packed_value[b, (chunk_size + 1):(chunk_size + N_prior_groups_of_block[b])]
    //       = linspaced_int_array(N_prior_groups_of_block[b], first_prior_group,
    //                             first_prior_group + N_prior_groups_of_block[b] - 1);
    //   }
    // }
    // (the same without a loop: each block's number of groups for their prior, J - (b - 1) * N_prior_slots kept
    //  between 0 and N_prior_slots; and the groups of the packed values as one matrix, one row per block: the
    //  observations' groups in decreasing order of time (padded with group 1) in rows of chunk_size, next to the
    //  groups 1 to J (padded with group 1) in rows of N_prior_slots; the model block indexes the raw log
    //  frailties with its values in the order of its columns, group_of_packed_value_by_column, and map_rect's
    //  job parameters with the groups of the observations of the blocks before the last)
    int N_prior_slots = 0;
    array[N_blocks] int N_prior_groups_of_block = rep_array(0, N_blocks);
    int N_packed = chunk_size + 1;
    int prior_chunk_size = max(1, (J + max(N_blocks, 1) - 1) %/% max(N_blocks, 1));
    array[N_blocks, chunk_size] int group_of_packed_value;
    if (N_blocks > 0) {
        for (b in 1:N_blocks) {
            int length_of_block = block_last[b] - block_first[b] + 1;
            group_of_packed_value[b, 1:length_of_block]
                = group_decreasing_time[block_first[b]:block_last[b]];
            if (length_of_block < chunk_size) {
                group_of_packed_value[b, (length_of_block + 1):chunk_size]
                    = rep_array(1, chunk_size - length_of_block);
            }
        }
    }

    // map_rect's data of the blocks before the last one (each of chunk_size positions): their rows of
    // X_decreasing_time packed column by column, and K followed by the positions of their observations' log
    // frailties in append_row(beta, log_frailty); no job-specific parameters
    // array[max(N_blocks - 1, 0), chunk_size * K] real X_of_block_by_column;
    // // array[max(N_blocks - 1, 0), chunk_size + 1] int K_and_log_frailty_index_of_block;
    // // array[max(N_blocks - 1, 0)] vector[0] no_job_parameters;
    // array[max(N_blocks - 1, 0), 1] int K_of_block;
    // for (b in 1:(N_blocks - 1)) {
    //   // K_and_log_frailty_index_of_block[b, 1] = K;
    //   K_of_block[b, 1] = K;
    //   for (m in 1:chunk_size) {
    //     // K_and_log_frailty_index_of_block[b, m + 1] = K + group_decreasing_time[block_first[b] + m - 1];
    //     for (k in 1:K) {
    //       X_of_block_by_column[b, (k - 1) * chunk_size + m] = X_decreasing_time[block_first[b] + m - 1, k];
    //     }
    //   }
    // }
    // (now: their rows of X_decreasing_time packed column by column, to_array_1d() of the block's rows, and K and
    //  chunk_size; the job parameters are the blocks' packed vectors)
    array[max(N_blocks - 1, 0), chunk_size * K] real X_of_block_by_column;
    array[max(N_blocks - 1, 0), 2] int K_and_chunk_size_of_block;
    // for (b in 1:(N_blocks - 1)) {
    //   X_of_block_by_column[b] = to_array_1d(X_decreasing_time[block_first[b]:block_last[b]]);
    //   K_and_chunk_size_of_block[b] = {K, chunk_size};
    // }
    // (the same without a loop: row b holds X_decreasing_time[(b - 1) * chunk_size + m, k] at column
    //  (k - 1) * chunk_size + m, the value of to_vector(X_decreasing_time) at
    //  (k - 1) * N + (b - 1) * chunk_size + m, i.e. at the block's offset (b - 1) * chunk_size (in every column
    //  of its row) plus (k - 1) * N + m (in every row of the column))
    if (N_blocks > 1) {
      K_and_chunk_size_of_block = rep_array({K, chunk_size}, N_blocks - 1);
    }
    if (N_blocks > 1 && K > 0) {
      matrix[N_blocks - 1, chunk_size * K] position_in_X_of_block_by_column
        = rep_matrix(linspaced_vector(N_blocks - 1, 0, N_blocks - 2) * chunk_size, chunk_size * K)
          + rep_matrix(to_row_vector(rep_matrix(linspaced_vector(chunk_size, 1, chunk_size), K)
                                     + rep_matrix(linspaced_row_vector(K, 0, K - 1) * N, chunk_size)),
                       N_blocks - 1);
      array[(N_blocks - 1) * chunk_size * K] int position_in_X_in_order_of_columns
        = to_int(to_array_1d(position_in_X_of_block_by_column));
      X_of_block_by_column
        = to_array_2d(to_matrix(to_vector(X_decreasing_time)[position_in_X_in_order_of_columns],
                                N_blocks - 1, chunk_size * K));
    }

    // // each block's observations' groups in decreasing order of time (the last block padded with group 1, whose
    // // values are not used), for log_frailty_of_block
    // array[N_blocks, chunk_size] int group_of_block;
    // for (b in 1:N_blocks) {
    //   for (m in 1:chunk_size) {
    //     int position = block_first[b] + m - 1;
    //     group_of_block[b, m] = position <= N ? group_decreasing_time[position] : 1;
    //   }
    // }
}


parameters {
    array[J] real log_frailty_raw;         // raw log frailties (nuisance parameters)
    vector[K] beta;                    // regression coefficients
    real<lower=0> sigma_frailty;       // frailty standard deviation
}


// (log_frailty and frailty: generated quantities below, the same values; see the header)
// transformed parameters {
//     vector[J] log_frailty;
//     vector[N] frailty;
//
//     // non-centered parameterization for frailties
//     log_frailty = sigma_frailty * log_frailty_raw;
//
//     // assign frailties to observations
//     // for (n in 1:N) {
//     //   frailty[n] = log_frailty[group[n]];
//     // }
//     frailty = log_frailty[group];
// }


model {
    // priors
    // (the prior of log_frailty_raw: in the chunks, each block's share of the groups; see the header)
    beta ~ nico_normal(0, 2);
    // log_frailty_raw ~ normal(0, 1);
    sigma_frailty ~ exponential(1);

    // // partial likelihood for Cox model with frailties (O(N); see the header): the scan over the risk sets
    // // outside reduce_sum_static, then each observation's term, summed by reduce_sum_static in chunks of
    // // chunk_size observations
    // if (N > 0) {
    //   vector[N] eta = X * beta + frailty;
    //   vector[N] eta_decreasing_time = eta[order_decreasing_time];
    //   // the log of the sum of exp(eta) over the first k observations in decreasing order of time
    //   vector[N] log_cumulative_risk;
    //   log_cumulative_risk[1] = eta_decreasing_time[1];
    //   for (k in 2:N) {
    //     log_cumulative_risk[k] = log_sum_exp(log_cumulative_risk[k - 1], eta_decreasing_time[k]);
    //   }
    //   // each observation's eta[n] minus the log of the sum over its risk set (the sliced argument),
    //   // weighted by its event indicator in the chunks (sum(eta[event_index]) -
    //   // sum(log_cumulative_risk[risk_set_size[event_index]]) of shared_frailty_Cox_survival.stan, summed
    //   // observation by observation)
    //   target += reduce_sum_static(partial_sum_observations,
    //                               to_array_1d(eta - log_cumulative_risk[risk_set_size]), chunk_size,
    //                               event_indicator);
    // }

    // // partial likelihood for Cox model with frailties (O(N); see the header): every term inside the chunks,
    // // with
    // // the scan over the risk sets chunked as a two-level scan
    // if (N > 0) {
    //   // (1) and (2): each block's offset, the log of the sum of exp(eta) over every observation before it
    //   // (block 1 has none, and does not use its value)
    //   vector[N_blocks] log_risk_before_block = rep_vector(negative_infinity(), N_blocks);
    //   if (N_blocks > 1) {
    //     vector[N_blocks - 1] log_risk_of_block = map_rect(log_sum_exp_eta_of_block,
    //                                                       append_row(beta, log_frailty),
    //                                                       no_job_parameters,
    //                                                       X_of_block_by_column,
    //                                                       K_and_log_frailty_index_of_block);
    //     log_risk_before_block[2] = log_risk_of_block[1];
    //     for (b in 3:N_blocks) {
    //       log_risk_before_block[b] = log_sum_exp(log_risk_before_block[b - 1], log_risk_of_block[b - 1]);
    //     }
    //   }
    //   // (3): each block's terms, one block per chunk of reduce_sum_static
    //   target += reduce_sum_static(partial_sum_blocks,
    //                               block_index,
    //                               1,
    //                               X_decreasing_time,
    //                               group_decreasing_time,
    //                               event_decreasing_time,
    //                               w_decreasing_time,
    //                               block_first,
    //                               block_last,
    //                               w_sum_of_block,
    //                               log_risk_before_block,
    //                               beta,
    //                               log_frailty);
    // }

    // // partial likelihood for Cox model with frailties (O(N); see the header): every term inside the chunks,
    // // with the scan over the risk sets chunked as a two-level scan, and each block's log frailties sliced
    // if (N > 0) {
    //   array[N_blocks] vector[chunk_size] log_frailty_of_block;
    //   for (b in 1:N_blocks) {
    //     log_frailty_of_block[b] = log_frailty[group_of_block[b]];
    //   }
    //   // (1) and (2): each block's offset, the log of the sum of exp(eta) over every observation before it
    //   // (block 1 has none, and does not use its value)
    //   vector[N_blocks] log_risk_before_block = rep_vector(negative_infinity(), N_blocks);
    //   if (N_blocks > 1) {
    //     vector[N_blocks - 1] log_risk_of_block = map_rect(log_sum_exp_eta_of_block,
    //                                                       beta,
    //                                                       log_frailty_of_block[1:(N_blocks - 1)],
    //                                                       X_of_block_by_column,
    //                                                       K_of_block);
    //     log_risk_before_block[2] = log_risk_of_block[1];
    //     for (b in 3:N_blocks) {
    //       log_risk_before_block[b] = log_sum_exp(log_risk_before_block[b - 1], log_risk_of_block[b - 1]);
    //     }
    //   }
    //   // (3): each block's terms, one block per chunk of reduce_sum_static
    //   target += reduce_sum_static(partial_sum_blocks,
    //                               log_frailty_of_block,
    //                               1,
    //                               X_decreasing_time,
    //                               event_decreasing_time,
    //                               w_decreasing_time,
    //                               block_first,
    //                               block_last,
    //                               log_risk_before_block,
    //                               beta);
    // }

    // partial likelihood for Cox model with frailties (O(N); see the header): every term, and the prior of
    // log_frailty_raw, inside the chunks, vectorised, with the scan over the risk sets chunked as a two-level
    // scan, in the pattern of LC_MVP_bin_PartialLog_v5_reduce_sum_static.stan (one single name as the sliced
    // argument, single names as the shared arguments, nothing built in the calls)
    // (the previous version, with more than one block: the raw log frailties of every block gathered on the
    //  main tape, the offsets from a pass over all N observations on the main tape, and reduce_sum_static over
    //  the blocks; kept below, commented out)
    // if (N > 0 && N_blocks == 1) {
    //   // (one block: the whole partial likelihood in ONE node, the raw log frailties read by their group inside
    //   //  it)
    //   target += nico_dot_products_of_affine_predictor_with_indexed_effects_and_log_cumulative_sum_exp(
    //       event_decreasing_time, minus_w_decreasing_time, X_decreasing_time, beta, sigma_frailty,
    //       log_frailty_raw, group_decreasing_time, rows_in_order_of_group, last_position_of_each_group,
    //       negative_infinity());
    // } else if (N > 0) {
    //   // (1) and (2): each block's offset, the log of the sum of exp(eta) over every observation before it
    //   // (block 1 has none, and does not use its value)
    //   vector[N_blocks] log_risk_before_block = rep_vector(negative_infinity(), N_blocks);
    //   array[N_blocks] vector[chunk_size] raw_of_blocks;
    //   for (b in 1:N_blocks) {
    //       raw_of_blocks[b] = to_vector(log_frailty_raw[group_of_packed_value[b]]);
    //   }
    //   if (N_blocks > 1) {
    //     // (map_rect's arguments as named variables: the shared parameters, and the raw log frailties of the
    //     //  observations of each block before the last one)
    //     // vector[K + 1] beta_and_sigma_frailty = append_row(beta, sigma_frailty);
    //     // array[N_blocks - 1] vector[chunk_size] log_frailty_raw_of_blocks_before_the_last;
    //     // for (b in 1:(N_blocks - 1)) {
    //     //   log_frailty_raw_of_blocks_before_the_last[b]
    //     //     = log_frailty_raw[group_of_packed_value[b, 1:chunk_size]];
    //     // }
    //     // (the same without a loop: the raw log frailties of the observations of the blocks before the last,
    //     //   in
    //     //  decreasing order of time, as the columns of a matrix, one column per block, split into an array of
    //     //  vectors by columns_as_array_of_vectors())
    //     // array[N_blocks - 1] vector[chunk_size] log_frailty_raw_of_blocks_before_the_last;
    //     // for (b in 1:(N_blocks - 1)) {
    //     //     log_frailty_raw_of_blocks_before_the_last[b] =
    //     //       to_vector(log_frailty_raw[group_of_packed_value[b]]);
    //     // }
    //     // array[N_blocks - 1] vector[chunk_size] log_frailty_raw_of_blocks_before_the_last
    //     //     = raw_of_blocks[1:(N_blocks - 1)];
    //     // vector[N_blocks - 1] log_risk_of_block =
    //     //   map_rect(log_sum_exp_eta_of_block_from_its_raw_log_frailties,
    //     //                                                   beta_and_sigma_frailty,
    //     //                                                   log_frailty_raw_of_blocks_before_the_last,
    //     //                                                   X_of_block_by_column,
    //     //                                                   K_and_chunk_size_of_block);
    //     vector[N_blocks - 1] log_risk_of_block = nico_log_sum_exp_of_indexed_affine_predictor_in_segments(
    //         X_decreasing_time, beta, sigma_frailty, log_frailty_raw, group_decreasing_time,
    //         block_first[1:(N_blocks - 1)], block_last[1:(N_blocks - 1)]);
    //     log_risk_before_block[2:N_blocks] = log_cumulative_sum_exp(log_risk_of_block, negative_infinity());
    //   }
    //   // (3): each block's packed row (its observations' raw log frailties, its share of the groups' raw log
    //   // frailties, and its offset), the sliced argument, and each block's terms, one block per chunk
    //   // array[N_blocks] row_vector[N_packed] packed_of_block;
    //   // for (b in 1:N_blocks) {
    //   //   packed_of_block[b] = append_col(log_frailty_raw[group_of_packed_value[b]]',
    //   //     log_risk_before_block[b]);
    //   // }
    //   // (the same values without a loop, as a two-dimensional array, one row per block: the raw log frailties
    //   //   of
    //   //  the groups of the packed values, in the order of the columns of their matrix, next to the offsets)
    //   // array[N_blocks, N_packed] real packed_of_block;
    //   // for (b in 1:N_blocks) {
    //   //     packed_of_block[b, 1:chunk_size] = log_frailty_raw[group_of_packed_value[b]];
    //   //     packed_of_block[b, N_packed] = log_risk_before_block[b];
    //   // }
    //   target += reduce_sum_static(fn_density_of_frailty_blocks_lupdf,
    //                               raw_of_blocks,
    //                               1,
    //                               X_decreasing_time,
    //                               event_decreasing_time,
    //                               w_decreasing_time,
    //                               block_first,
    //                               block_last,
    //                               beta,
    //                               sigma_frailty,
    //                               log_risk_before_block);
    // }
    // // Every group is included exactly once, including groups with no observations.
    // if (J > 0) {
    //     target += reduce_sum_static(fn_prior_of_frailty_slice_lupdf, log_frailty_raw, prior_chunk_size);
    // }

    // (now: the whole partial likelihood ONE node at every number of blocks, its blocks of chunk_size positions
    //  run in parallel on the chain's threads, the two-level scan of the risk sets inside it; see the header)
    if (N > 0) {
      target += nico_dot_products_of_affine_predictor_with_indexed_effects_and_log_cumulative_sum_exp(
          event_decreasing_time, minus_w_decreasing_time, X_decreasing_time, beta, sigma_frailty,
          log_frailty_raw, group_decreasing_time, rows_in_order_of_group, last_position_of_each_group,
          negative_infinity(), chunk_size);
    }
    // Every group is included exactly once, including groups with no observations (one block: one node; more
    // blocks: in the chunks of reduce_sum_static, sliced by group)
    if (J > 0 && N_blocks == 1) {
        target += nico_normal_lupdf(log_frailty_raw | 0, 1);
    } else if (J > 0) {
        target += reduce_sum_static(fn_prior_of_frailty_slice_lupdf, log_frailty_raw, prior_chunk_size);
    }

    // // partial likelihood for Cox model with frailties (summed over the events, in chunks)
    // target += reduce_sum_static(partial_sum_events, event_index, grainsize, X, beta, frailty, t);
}


generated quantities {
    real simd_lanes = bmvp_simd_lanes();
    array[J] real log_frailty;
    array[N] real frailty;
    array[N] real log_lik;
    array[J] real frailty_exp;
    if (J > 0) {
        int n_group_blocks = (J + chunk_size - 1) %/% chunk_size;
        for (b in 1:n_group_blocks) {
            int first = (b - 1) * chunk_size + 1;
            int last = min(b * chunk_size, J);
            vector[last - first + 1] values = sigma_frailty * to_vector(log_frailty_raw[first:last]);
            log_frailty[first:last] = to_array_1d(values);
            frailty_exp[first:last] = to_array_1d(nico_exp(values));
        }
    }
    if (N > 0) {
        array[N] real log_cumulative_risk;
        real log_offset = negative_infinity();
        for (b in 1:N_blocks) {
            int first = block_first[b];
            int last = block_last[b];
            vector[last - first + 1] eta = X_decreasing_time[first:last] * beta
                                          + sigma_frailty * to_vector(
                                                log_frailty_raw[group_decreasing_time[first:last]]);
            vector[last - first + 1] prefix = nico_log_cumulative_sum_exp(eta, log_offset);
            log_cumulative_risk[first:last] = to_array_1d(prefix);
            log_offset = prefix[last - first + 1];
        }
        for (b in 1:N_blocks) {
            int first = block_first[b];
            int last = block_last[b];
            frailty[first:last] = log_frailty[group[first:last]];
            vector[last - first + 1] eta = X[first:last] * beta + to_vector(frailty[first:last]);
            log_lik[first:last] = to_array_1d(event_vector[first:last] .*
                (eta - to_vector(log_cumulative_risk[risk_set_size[first:last]])));
        }
    }
}
















