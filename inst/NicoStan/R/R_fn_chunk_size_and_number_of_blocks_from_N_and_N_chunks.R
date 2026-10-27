##
## ---- R_fn_chunk_size_and_number_of_blocks_from_N_and_N_chunks.R --------------------------------------------------------------------
##
## The one chunk_size formula of NicoStan, chunk_size = ceiling(N / N_chunks), and the chunks reduce_sum_static
## runs at that chunk_size: TBB's simple_partitioner halves a piece [first, last] at first + size %/% 2 while it
## has more than chunk_size units (stan/math/rev/functor/reduce_sum.hpp, tbb/blocked_range.h). Every chunk_size,
## N_blocks, block_first and block_last given to a Stan model comes from this function, so that every block is
## one of reduce_sum_static's own chunks.
##
#' The chunks reduce_sum_static makes of N units at chunk_size = ceiling(N / N_chunks)
#'
#' chunk_size = ceiling(N / N_chunks). From the one piece 1, ..., N, every piece with more than chunk_size units
#' is halved, into first, ..., first + size %/% 2 - 1 and first + size %/% 2, ..., last, until no piece has more
#' than chunk_size units: the chunks of Stan's reduce_sum_static (TBB's simple_partitioner) at that chunk_size,
#' in order. N_blocks, the number of chunks that run, can differ from N_chunks (e.g. N = 200,000 and
#' N_chunks = 24 give chunk_size = 8334 and 32 chunks of 6250 units). N_chunks larger than N is refused: a layout
#' with more chunks than units.
#'
#' @param N The number of units chunked (rows, groups, studies or time points), one positive whole number.
#' @param N_chunks The number of chunks asked for, one positive whole number, at most N.
#' @return A list: chunk_size, N_blocks (the number of chunks that run), and block_first and block_last, the
#'   first and the last unit of each chunk (integer vectors of length N_blocks).
#' @export
fn_chunk_size_and_number_of_blocks_from_N_and_N_chunks <-  function(
        N,
        N_chunks ) {

        ##
        ## ---- N: one positive whole number ------------------------------------------------------------------------------------------
        ##
        if (!is.numeric(N) || (length(N) != 1) || !is.finite(N) || (N < 1) || (N != round(N)) ||
            (N > .Machine$integer.max)) {
              stop(paste0("fn_chunk_size_and_number_of_blocks_from_N_and_N_chunks: N must be one positive whole ",
                          "number, not ", paste(N, collapse = ", "), "."))
        }
        ##
        ## ---- N_chunks: one positive whole number, at most N ------------------------------------------------------------------------
        ##
        if (!is.numeric(N_chunks) || (length(N_chunks) != 1) || !is.finite(N_chunks) ||
            (N_chunks < 1) || (N_chunks != round(N_chunks))) {
              stop(paste0("fn_chunk_size_and_number_of_blocks_from_N_and_N_chunks: N_chunks must be one ",
                          "positive whole number, not ", paste(N_chunks, collapse = ", "), "."))
        }
        if (N_chunks > N) {
              stop(paste0("fn_chunk_size_and_number_of_blocks_from_N_and_N_chunks: N_chunks = ", N_chunks,
                          " is larger than N = ", N, ": more chunks than units."))
        }
        ##
        ## ---- chunk_size = ceiling(N / N_chunks) ------------------------------------------------------------------------------------
        ##
        chunk_size <-  as.integer(ceiling(N / N_chunks))
        block_first <-  1L
        block_last <-  as.integer(N)
        size_of_each_block <-  block_last - block_first + 1L
        ##
        ## ---- Each piece with more than chunk_size units halved at first + size %/% 2, in order -------------------------------------
        ##
        while (any(size_of_each_block > chunk_size)) {

              is_halved <-  size_of_each_block > chunk_size
              first_of_right_half <-  block_first + size_of_each_block %/% 2L
              ## (each piece, then its right half if it is halved: the pieces stay in order)
              is_kept <-  c(rbind(TRUE, is_halved))
              block_last <-  c(rbind(ifelse(is_halved, first_of_right_half - 1L, block_last), block_last))[is_kept]
              block_first <-  c(rbind(block_first, first_of_right_half))[is_kept]
              size_of_each_block <-  block_last - block_first + 1L

        }
        return(list( chunk_size   = chunk_size,
                     N_blocks     = length(block_first),
                     block_first  = as.integer(block_first),
                     block_last   = as.integer(block_last)))

}




##
## ---- The Stan data with the chunks of N_chunks -------------------------------------------------------------------------------------
##
#' The Stan data with chunk_size, N_blocks, block_first and block_last of N_chunks chunks
#'
#' Sets the four fields from fn_chunk_size_and_number_of_blocks_from_N_and_N_chunks(), with block_first and
#' block_last as one-dimensional arrays, so that cmdstanr::write_stan_json() writes them in brackets even for one
#' chunk.
#'
#' @param Stan_data_list The Stan data, with its N.
#' @param N_chunks The number of chunks asked for.
#' @return The Stan data with the four fields set.
#' @export
fn_Stan_data_list_with_the_chunks_of_N_chunks <-  function(
        Stan_data_list,
        N_chunks ) {

        chunks <-  fn_chunk_size_and_number_of_blocks_from_N_and_N_chunks(N         = Stan_data_list$N,
                                                                           N_chunks  = N_chunks)
        Stan_data_list$chunk_size <-  chunks$chunk_size
        Stan_data_list$N_blocks <-  chunks$N_blocks
        Stan_data_list$block_first <-  array(chunks$block_first, dim = chunks$N_blocks)
        Stan_data_list$block_last <-  array(chunks$block_last, dim = chunks$N_blocks)
        return(Stan_data_list)

}
