##
## ==== R_fn_check_the_value_and_every_derivative_of_an_auto_chunked_library.R ===================================
##
## ---- The check mode of the E5_DEBUG runner: the value and EVERY derivative of an auto-chunked library, with
##      auto-chunking off (one chunk, Stan's own result) against auto-chunking on at 2, 4, 8, 16, 32 and 64
##      pieces, each twice (the second call reuses what the first made), plus the memory of each call
##      (allocations after the first gradient, container bytes outside a piece) -
##
## Nothing is timed here. The library is the one the runner times, called through BridgeStan's C entry points
## (bs_model_construct_R, bs_param_unc_num_R, bs_log_density_gradient_R) and the switches the library exports:
##   nicostan_set_auto_chunking(int*)                                  0 = off, 1 = on;
##   nicostan_set_chunk_budget_bytes(double*)                          the bytes of one piece;
##   nicostan_last_piece_counts(double*, double*)                      fewest and most pieces of the last call;
##   nicostan_recycled_allocations_after_the_first_gradient(long*, long*);
##   nicostan_container_bytes_outside_a_piece(double*, double*).
## The pieces come from bytes only: the bytes of k pieces are the measured memory of one gradient of this model
## and data (M_bytes_per_chain, from fn_memory_of_one_gradient_measured_in_this_R_process()) over k, k a power
## of 2. A call the library stops (a named stop) is a row with its message, never a silent pass. A row passes
## when the worst relative difference of the value and of every derivative is at most the tolerance (1e-4), no
## allocation was made after the first gradient and no container byte lay outside a piece.
##
##
#' The value and every derivative of an auto-chunked library against auto-chunking off, at 2 to 64 pieces
#'
#' @param model_so_file The auto-chunked BridgeStan library (.so), the one the runner times.
#' @param JSON_text_of_Stan_data The Stan data as JSON text.
#' @param bytes_of_one_gradient The measured memory of one gradient of this model and data (M_bytes_per_chain).
#' @param numbers_of_pieces The piece counts (default 2, 4, 8, 16, 32, 64; powers of 2 only).
#' @param tolerance The largest relative difference that passes (default 1e-4).
#' @param propto,jacobian As in NicoStan's sampler (default TRUE and TRUE).
#' @param seed The seed of the model's construction and of the point (default 123).
#' @param max_stops_listed A library built to list its stops (NICOSTAN_CHECK_LISTS_EVERY_STOP) fires each stop once and
#'   then goes on as stock; the same gradient is run again until it passes or a stop comes back, at most this many
#'   times (default 5000). Every stop seen is in stop_message, joined by " || ".
#' @return A data.frame, one row per call: library, pieces_asked, run, worst_relative_difference,
#'   fewest_pieces, most_pieces, allocations_after_the_first_gradient,
#'   big_allocations_after_the_first_gradient, container_bytes_outside_a_piece,
#'   most_container_bytes_outside_a_piece, stop_message, pass.
#' @noRd
fn_check_the_value_and_every_derivative_of_an_auto_chunked_library <-  function( model_so_file,
                                                                                  JSON_text_of_Stan_data,
                                                                                  bytes_of_one_gradient,
                                                                                  numbers_of_pieces  = 2^(1:6),
                                                                                  tolerance          = 1e-4,
                                                                                  propto             = TRUE,
                                                                                  jacobian           = TRUE,
                                                                                  seed               = 123,
                                                                                  max_stops_listed   = 5000
) {

        stopifnot(all(numbers_of_pieces >= 2), all(log2(numbers_of_pieces) == round(log2(numbers_of_pieces))))
        stopifnot(is.finite(bytes_of_one_gradient), bytes_of_one_gradient > 0)
        model_so_file <-  normalizePath(model_so_file, mustWork = TRUE)
        library_name <-  tools::file_path_sans_ext(basename(model_so_file))
        if (!is.loaded("bs_model_construct_R", PACKAGE = library_name)) dyn.load(model_so_file)
        for (symbol in c("nicostan_set_auto_chunking", "nicostan_set_chunk_budget_bytes")) {
              if (!is.loaded(symbol, PACKAGE = library_name)) {
                    stop(paste0("The library ", library_name, " has no ", symbol, ": it is not an auto-chunked ",
                                "library, so it cannot be checked."))
              }
        }
        ## (every call of this library's routines goes through fn_call_C, never a direct call of R.s C interface: the package's
        ##  build registers every routine its R code names in such a call as its OWN routine, and these belong to
        ##  the checked BridgeStan library, not to NicoStan)
        fn_call_C <-  get(".C", envir = baseenv(), mode = "function")
        ##
        ## ---- the model, constructed with the data:
        ##
        construction <-  fn_call_C( "bs_model_construct_R",
                             as.character(JSON_text_of_Stan_data),
                             as.integer(seed),
                             ptr_out  = raw(8),
                             err_msg  = as.character(""),
                             err_ptr  = raw(8),
                             PACKAGE  = library_name)
        if (all(construction$ptr_out == 0)) {
              stop(paste0("BridgeStan could not construct the model ", library_name, ": ", construction$err_msg))
        }
        model_pointer <-  construction$ptr_out
        n_unconstrained_parameters <-  fn_call_C( "bs_param_unc_num_R",
                                           as.raw(model_pointer),
                                           num      = as.integer(0),
                                           PACKAGE  = library_name)$num
        ##
        ## ---- one call of the log density and every derivative, and the library's own switches and counts:
        ##
        fn_value_and_every_derivative <-  function(theta) {

              fn_call_C( "bs_log_density_gradient_R",
                  as.raw(model_pointer),
                  as.logical(propto),
                  as.logical(jacobian),
                  as.double(theta),
                  val          = double(1),
                  gradient     = double(n_unconstrained_parameters),
                  return_code  = as.integer(0),
                  err_msg      = as.character(""),
                  err_ptr      = raw(8),
                  NAOK         = TRUE,
                  PACKAGE      = library_name)

        }
        fn_switch_auto_chunking <-  function(on) {
              invisible(fn_call_C("nicostan_set_auto_chunking", as.integer(on), PACKAGE = library_name))
        }
        fn_set_the_bytes_of_a_piece <-  function(bytes) {
              invisible(fn_call_C("nicostan_set_chunk_budget_bytes", as.double(bytes), PACKAGE = library_name))
        }
        fn_two_doubles_if_exported <-  function(symbol) {
              if (!is.loaded(symbol, PACKAGE = library_name)) return(c(NA_real_, NA_real_))
              out <-  fn_call_C(symbol, first = double(1), second = double(1), PACKAGE = library_name)
              return(c(out$first, out$second))
        }
        fn_two_longs_if_exported <-  function(symbol) {
              ## (two C longs of 8 bytes each: passed as raw bytes, never as R integers of 4 bytes)
              if (!is.loaded(symbol, PACKAGE = library_name)) return(c(NA_real_, NA_real_))
              out <-  fn_call_C(symbol, first = raw(8), second = raw(8), PACKAGE = library_name)
              return(c(readBin(out$first, what = "integer", size = 8, n = 1, endian = .Platform$endian),
                       readBin(out$second, what = "integer", size = 8, n = 1, endian = .Platform$endian)))
        }
        ##
        ## ---- the point: values in (-1, 1) from the seed, the same for every call:
        ##
        set.seed(seed)
        theta <-  stats::runif(n_unconstrained_parameters, min = -1, max = 1)
        ##
        ## ---- the reference: auto-chunking off, Stan's own value and every derivative:
        ##
        fn_switch_auto_chunking(0L)
        reference <-  fn_value_and_every_derivative(theta)
        if (reference$return_code != 0) {
              stop(paste0("The reference call (auto-chunking off) of ", library_name, " failed: ", reference$err_msg))
        }
        ##
        ## ---- auto-chunking on, at each number of pieces, twice:
        ##
        fn_switch_auto_chunking(1L)
        rows <-  list()
        for (pieces_asked in numbers_of_pieces) {
              fn_set_the_bytes_of_a_piece(bytes_of_one_gradient / pieces_asked)
              for (run in 1:2) {
                    call <-  fn_value_and_every_derivative(theta)
                    ## (a library built to list its stops fires each stop once and then goes on as stock, so the same
                    ##  gradient is run again until it passes or a stop comes back a second time: one run lists them all)
                    stop_messages <-  character(0)
                    while (call$return_code != 0 && !(call$err_msg %in% stop_messages) &&
                           length(stop_messages) < max_stops_listed) {
                          stop_messages <-  c(stop_messages, call$err_msg)
                          call <-  fn_value_and_every_derivative(theta)
                    }
                    stop_message <-  paste(stop_messages, collapse = " || ")
                    worst <-  NA_real_
                    if (call$return_code == 0) {
                          worst <-  max(abs(call$val - reference$val) / max(1, abs(reference$val)),
                                        abs(call$gradient - reference$gradient) / pmax(1, abs(reference$gradient)))
                    }
                    piece_counts <-  fn_two_doubles_if_exported("nicostan_last_piece_counts")
                    allocations <-  fn_two_longs_if_exported("nicostan_recycled_allocations_after_the_first_gradient")
                    bytes_outside <-  fn_two_doubles_if_exported("nicostan_container_bytes_outside_a_piece")
                    pass <-  (length(stop_messages) == 0) && (call$return_code == 0) && is.finite(worst) &&
                             (worst <= tolerance) &&
                             (is.na(allocations[1]) || allocations[1] <= 0) &&
                             (is.na(bytes_outside[1]) || bytes_outside[1] <= 0)
                    rows[[length(rows) + 1]] <-  data.frame(
                          library                                     = library_name,
                          pieces_asked                                = pieces_asked,
                          run                                         = run,
                          worst_relative_difference                   = worst,
                          fewest_pieces                               = piece_counts[1],
                          most_pieces                                 = piece_counts[2],
                          allocations_after_the_first_gradient        = allocations[1],
                          big_allocations_after_the_first_gradient    = allocations[2],
                          container_bytes_outside_a_piece             = bytes_outside[1],
                          most_container_bytes_outside_a_piece        = bytes_outside[2],
                          stop_message                                = stop_message,
                          pass                                        = pass,
                          stringsAsFactors                            = FALSE)
              }
        }
        fn_switch_auto_chunking(0L)
        invisible(fn_call_C("bs_model_destruct_R", as.raw(model_pointer), PACKAGE = library_name))
        return(do.call(rbind, rows))

}
