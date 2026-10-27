#pragma once
//
// AUTO-CHUNKING: the running sums of NicoSIMD's general functions, read by NicoStan's converter (no code here, and no
// model includes this file).
//
// A running-sum function f(rows..., carry) of the rows is chunked as reduce_sum_static cuts the rows: in each chunk,
// f(the chunk's rows..., the carry of the chunks before it); the carry of the first chunk is f's own argument, and the
// carry after each chunk is the expression of AUTO_RUNNING_SUM_CARRY_OUT (the argument names stand for the chunk's).
//
// nico_dot_product_log_cumulative_sum_exp(w, x, log_offset) = sum_i w_i log(exp(log_offset) + sum_{j <= i} exp(x_j)):
// the carry is the log of the running sum, log(exp(log_offset) + sum of exp(x) over the rows before).
//
// AUTO_RUNNING_SUM nico_dot_product_log_cumulative_sum_exp
// AUTO_RUNNING_SUM_ARGUMENTS w:rows x:rows log_offset:log_carry
// AUTO_RUNNING_SUM_CARRY_OUT nico_log_sum_exp(log_offset, nico_log_sum_exp(x, pstream__), pstream__)
