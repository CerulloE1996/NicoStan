#pragma once
//
// AUTO-CHUNKING: the row sums of NicoSIMD's general functions whose term of a row also reads the row before it,
// read by NicoStan's converter (no code here, and no model includes this file).
//
// f(arguments...) = the sum over the rows k of a term that reads row k of every argument and, unless row k starts
// a group, row k - 1 of its rows_with_the_row_before arguments. It is chunked as reduce_sum_static cuts the rows
// (the cuts are never moved); each chunk whose first row is not row 1 and does not start a group is given the row
// before its first row too, put first, as a row whose own term is zero: its rows_with_the_row_before arguments at
// the row before, its rows and group_start arguments at the values of AUTO_ROW_SUM_WITH_THE_ROW_BEFORE_ZERO_TERM.
// The chunk's first term then reads the row before, and the adjoints of the row before go back to its own
// parameters.
//
// Roles: rows (an array or vector of the data, by rows), rows_with_the_row_before (a vector of the rows, read at
// row k and at row k - 1), group_start (an array or vector of the data, 1 where a row starts a group: that row's
// term reads no row before it), scalar (the same for every row).
//
// nico_binomial_counts_above_successive_cutpoints_of_a_cumulative_link_lpmf(n | n_previous, latent, is_first,
// link, bound) = sum_k binomial_lpmf(n_k | n_previous_k, p_k), p_k = F(latent_k) / F(latent_(k - 1)) bounded,
// with F(latent_(k - 1)) = 1 where is_first_k = 1: a row with n = 0 and n_previous = 0 has the term
// binomial_lpmf(0 | 0, p) = 0 for every p (log choose(0, 0) = 0 too), and is_first = 1 there.
//
// (the arguments of a contract on one line, as the converter reads them)
// AUTO_ROW_SUM_WITH_THE_ROW_BEFORE nico_binomial_counts_above_successive_cutpoints_of_a_cumulative_link_lpmf
// AUTO_ROW_SUM_WITH_THE_ROW_BEFORE_ARGUMENTS n:rows n_previous:rows latent:rows_with_the_row_before is_first:group_start link:scalar bound:scalar
// AUTO_ROW_SUM_WITH_THE_ROW_BEFORE_ZERO_TERM n:0 n_previous:0 is_first:1
//
//     real nico_binomial_counts_above_successive_cutpoints_of_a_cumulative_link_lpmf(
//           array[] int n, array[] int n_previous, vector latent, data vector is_first, data int link,
//           data real bound);
