##
## ==== R_fn_compute_burnin_n_threads_WCP_and_num_chunks.R =====================================================
##
## ---- Burn-in within-chain threads (n_threads_WCP_burnin) and chunks (num_chunks_burnin) of a Stan model whose
##      likelihood is summed by reduce_sum_static, from M_bytes/chain and the CPUs available to the R process ----
##
## The rule, fitted to the burn-in timings of Paper 1 (its Stan LC-MVP model run through NicoStan with 4 burn-in
## chains, N = 500 to 50,000, on the 96-core local-HPC: 12 L3 caches of 32 MiB, 16 hardware threads each, and on
## the 8-core laptop: one L3 cache of 16 MiB, 16 hardware threads) and of Paper 3's benchmark models
## (hierarchical logistic regression, joint longitudinal-survival, latent diffusion survival and the four-class
## LC-MVOP model, on one 16-thread L3 cache of the local-HPC):
##
##   H, N_L3, H_L3         = the hardware threads available to the process, the number of L3 caches they belong
##                           to, and the largest number of them that share one L3 cache
##   T                     = the largest power of two at most min( floor(H / N_chains), H_L3 ), at least 1
##   N_active/L3           = min( H_L3, ceil(N_chains * T / N_L3) )
##   S_target              = S_L3 / N_active/L3
##   num_chunks_burnin     = 2^ceil( log2( max( ceil(M_bytes/chain / S_target), T * j ) ) )
##   n_threads_WCP_burnin  = T
##
## i.e. each chain gets the hardware threads available per chain, but no more than one L3 cache holds (on the
## local-HPC, 16 threads per chain were faster than 24, 32 and 44), whatever the size of the model: with each
## burn-in iteration timed for at least 0.2 s, 4 threads per chain on one 16-thread L3 cache were the fastest for
## every model timed, down to a tape of 0.07 MiB (Paper 3's hierarchical logistic regression at N = 500). Each
## piece of a chain's tape fits in its thread's share of the L3 cache (Paper 1's chunk rule,
## fn_compute_initial_N_chunks(), with the burn-in layout), and each thread gets j pieces, so that the threads'
## work is balanced: j = 2 by default; for a Stan model with its init_object,
## fn_compute_burnin_n_threads_WCP_and_num_chunks_for_Stan_model() times the rule's configurations for j = 1, 2
## and 4, at T, T / 2 and 1 threads per chain, in NicoStan's own burn-in worker, and uses the fastest. T is a
## power of two, so that it divides the number of pieces and every thread gets the same number of pieces.
##
## The number of chunks is a power of two because reduce_sum_static (TBB's simple partitioner on a blocked range)
## halves each range of units, the left half rounded down, until each piece holds at most chunk_size units: with
## chunk_size = ceiling(N_units / 2^k) and N_units well above 2^k, every range is halved exactly k times, so
## num_chunks = 2^k gives 2^k pieces of equal size (to within one unit). Other chunk sizes can give other numbers
## of pieces, of unequal sizes (e.g. 10 nominal chunks at N = 500 ran as 16 pieces in Paper 1).
##
##
##
##
#### ---- the CPUs available to this R process and their L3 caches -----------------------------------------------

#' The hardware threads available to this R process and the L3 caches they belong to
#'
#' Reads the CPU affinity mask of the R process (Cpus_allowed_list of /proc/self/status; e.g. the CPUs given to
#' taskset), and, for each of those CPUs, the L3 cache it belongs to (Linux sysfs, cache/index*/level,
#' shared_cpu_list and size).
#'
#' @param CPU_ids_available The CPU ids available (default: the affinity mask of this R process).
#' @param sysfs_CPU_directory The sysfs CPU directory (default "/sys/devices/system/cpu").
#' @return A list: CPU_ids_available, N_hardware_threads_available, N_L3_caches_available (the L3 caches that hold
#'   at least one available CPU), N_hardware_threads_per_L3_available (the largest number of available CPUs in one
#'   L3 cache), L3_cache_size_bytes (the smallest of those L3 caches) and N_available_CPUs_in_each_L3.
#' @keywords internal
## (read once per R process for each set of CPUs: each read walks the cache directories of every CPU given, and
##  Paper 1's pickers read it several times per layout; the copy read first is returned again for the same CPUs
##  and sysfs directory, the affinity mask being read each time)
environment_of_the_L3_cache_topology_of_available_CPUs_read_in_this_R_process <-  new.env()
fn_read_L3_cache_topology_of_available_CPUs <-  function( CPU_ids_available    = NULL,
                                                          sysfs_CPU_directory  = "/sys/devices/system/cpu"
) {

        if (is.null(x = CPU_ids_available) && file.exists("/proc/self/status")) {
              status_lines <-  readLines(con = "/proc/self/status", warn = FALSE)
              CPU_mask_line <-  status_lines[startsWith(x = status_lines, prefix = "Cpus_allowed_list:")]
              if (length(x = CPU_mask_line) == 1) {
                    CPU_ids_available <-  fn_parse_sysfs_CPU_list(
                          CPU_list_string = sub(pattern = "^Cpus_allowed_list:[[:space:]]*", replacement = "",
                                                x = CPU_mask_line))
              }
        }
        name_of_saved_copy <-  paste0(sysfs_CPU_directory, ": ", paste(sort(CPU_ids_available), collapse = ","))
        saved_copy <-  get0(x = name_of_saved_copy, inherits = FALSE,
                            envir = environment_of_the_L3_cache_topology_of_available_CPUs_read_in_this_R_process)
        if (!is.null(saved_copy)) return(saved_copy)
        L3_cache_topology <-  fn_read_L3_cache_topology_of_available_CPUs_each_time(
                                    CPU_ids_available    = CPU_ids_available,
                                    sysfs_CPU_directory  = sysfs_CPU_directory)
        assign(x = name_of_saved_copy, value = L3_cache_topology,
               envir = environment_of_the_L3_cache_topology_of_available_CPUs_read_in_this_R_process)
        return(L3_cache_topology)

}
## (the reading itself, from sysfs and the affinity mask, each time it is called)
fn_read_L3_cache_topology_of_available_CPUs_each_time <-  function(
        CPU_ids_available    = NULL,
        sysfs_CPU_directory  = "/sys/devices/system/cpu"
) {

        if (!identical(x = Sys.info()[["sysname"]], y = "Linux") || !dir.exists(paths = sysfs_CPU_directory)) {
              stop(paste0("fn_read_L3_cache_topology_of_available_CPUs: the CPU topology is read from Linux ",
                          "sysfs (", sysfs_CPU_directory, "), which is not available on this system. Supply ",
                          "N_hardware_threads_available, N_L3_caches_available, ",
                          "N_hardware_threads_per_L3_available and L3_cache_size_bytes as arguments instead."))
        }
        ##
        if (is.null(x = CPU_ids_available)) {
              status_lines <-  readLines(con = "/proc/self/status", warn = FALSE)
              CPU_mask_line <-  status_lines[startsWith(x = status_lines, prefix = "Cpus_allowed_list:")]
              if (length(x = CPU_mask_line) != 1) {
                    stop(paste0("fn_read_L3_cache_topology_of_available_CPUs: Cpus_allowed_list not found ",
                                "in /proc/self/status."))
              }
              CPU_ids_available <-  fn_parse_sysfs_CPU_list(
                    CPU_list_string = sub(pattern = "^Cpus_allowed_list:[[:space:]]*", replacement = "",
                                          x = CPU_mask_line))
        }
        ##
        online_CPU_list_file <-  file.path(sysfs_CPU_directory, "online")
        if (file.exists(online_CPU_list_file)) {
              online_CPU_ids <-  fn_parse_sysfs_CPU_list(CPU_list_string = readLines(con = online_CPU_list_file,
                                                                                    n = 1, warn = FALSE))
              CPU_ids_available <-  CPU_ids_available[CPU_ids_available %in% online_CPU_ids]
        }
        if (length(x = CPU_ids_available) == 0) {
              stop("fn_read_L3_cache_topology_of_available_CPUs: no online CPU is available to this process.")
        }
        ##
        ## ---- the L3 cache of each available CPU (identified by its shared_cpu_list) and its size:
        ##
        L3_cache_of_each_CPU <-  rep(x = NA_character_, times = length(x = CPU_ids_available))
        L3_cache_size_bytes_of_each_CPU <-  rep(x = NA_real_, times = length(x = CPU_ids_available))
        for (CPU_index in seq_along(CPU_ids_available)) {
              cache_index_directories <-  list.files(path = file.path(sysfs_CPU_directory,
                                                                      paste0("cpu", CPU_ids_available[CPU_index]),
                                                                      "cache"),
                                                     pattern = "^index[0-9]+$", full.names = TRUE)
              for (cache_index_directory in cache_index_directories) {
                    cache_level <-  suppressWarnings(as.numeric(
                          readLines(con = file.path(cache_index_directory, "level"), n = 1, warn = FALSE)))
                    if (!isTRUE(cache_level == 3)) next
                    L3_cache_of_each_CPU[CPU_index] <-  readLines(con = file.path(cache_index_directory,
                                                                                  "shared_cpu_list"),
                                                                  n = 1, warn = FALSE)
                    L3_cache_size_bytes_of_each_CPU[CPU_index] <-  fn_parse_sysfs_cache_size_bytes(
                          cache_size_string = readLines(con = file.path(cache_index_directory, "size"),
                                                        n = 1, warn = FALSE))
              }
        }
        if (anyNA(x = L3_cache_of_each_CPU) || anyNA(x = L3_cache_size_bytes_of_each_CPU)) {
              stop(paste0("fn_read_L3_cache_topology_of_available_CPUs: no L3 cache information under ",
                          sysfs_CPU_directory, " for CPU(s) ",
                          paste(CPU_ids_available[is.na(L3_cache_of_each_CPU)], collapse = ", "),
                          "; supply the topology as arguments instead."))
        }
        ##
        N_available_CPUs_in_each_L3 <-  table(L3_cache_of_each_CPU)
        ##
        return(list( CPU_ids_available                    = sort(x = CPU_ids_available),
                     N_hardware_threads_available         = length(x = CPU_ids_available),
                     N_L3_caches_available                = length(x = N_available_CPUs_in_each_L3),
                     N_hardware_threads_per_L3_available  = max(N_available_CPUs_in_each_L3),
                     L3_cache_size_bytes                  = min(L3_cache_size_bytes_of_each_CPU),
                     N_available_CPUs_in_each_L3          = c(N_available_CPUs_in_each_L3)))

}




#### ---- the rule -----------------------------------------------------------------------------------------------

#' Burn-in within-chain threads and chunks of a Stan model with reduce_sum_static
#'
#' Chooses n_threads_WCP_burnin (the within-chain threads of each burn-in chain) and num_chunks_burnin (the chunks
#' of the burn-in phase, i.e. chunk_size = ceiling(N_units / num_chunks_burnin)) for a Stan model whose likelihood
#' is summed over N_units units by reduce_sum_static, from M_bytes/chain (the memory of one full-data
#' log-density-and-gradient evaluation of one chain, e.g. from [fn_estimate_initial_M_bytes_per_chain()]) and the
#' CPUs available to the R process, with the rule
#'
#' \preformatted{
#'   T                     = largest power of two <= min( floor(H / N_chains), H_L3 ), at least 1
#'   N_active/L3           = min( H_L3, ceil(N_chains * T / N_L3) )
#'   S_target              = S_L3 / N_active/L3
#'   num_chunks_burnin     = 2^ceil( log2( max( ceil(M_bytes/chain / S_target), T * j ) ) ),
#'                           at most 2^floor(log2(N_units))
#'   n_threads_WCP_burnin  = T
#' }
#'
#' where H is the number of hardware threads available, N_L3 the number of L3 caches they belong to, H_L3 the
#' largest number of them in one L3 cache, S_L3 the L3 cache size and j the pieces per thread
#' (pieces_per_thread). It was fitted to the burn-in timings of Paper 1's Stan LC-MVP model and of Paper 3's
#' benchmark models run through NicoStan with 4 burn-in chains. The number of chunks is a power of two, because
#' reduce_sum_static halves the range of units until each piece holds at most chunk_size units: with chunk_size
#' = ceiling(N_units / num_chunks_burnin) and N_units well above num_chunks_burnin, the pieces are exactly
#' num_chunks_burnin, of equal size to within one unit.
#'
#' @param M_bytes_per_chain Memory, in bytes, of one full-data log-density-and-gradient evaluation of one chain
#'   (the model evaluated as one piece), or the list returned by [fn_estimate_initial_M_bytes_per_chain()].
#' @param N_units The number of units reduce_sum_static sums over (the N of the Stan data).
#' @param n_chains_burnin Number of burn-in chains run in parallel (default 4).
#' @param pieces_per_thread j, the pieces of tape per thread (default 2; the best j depends on the model, see
#'   [fn_compute_burnin_n_threads_WCP_and_num_chunks_for_Stan_model()], which times j = 1, 2 and 4).
#' @param n_threads_WCP_burnin The threads per chain (default NULL: T of the rule; a value given is used, with
#'   the rule's chunks for it).
#' @param CPU_ids_available The CPU ids available (default: the affinity mask of this R process).
#' @param N_hardware_threads_available The hardware threads available (with the next three, the topology of the
#'   available CPUs; when all four are given, sysfs is not read).
#' @param N_L3_caches_available The number of L3 caches the available CPUs belong to.
#' @param N_hardware_threads_per_L3_available The largest number of available CPUs in one L3 cache.
#' @param L3_cache_size_bytes The size of one L3 cache in bytes.
#' @param sysfs_CPU_directory The sysfs CPU directory (default "/sys/devices/system/cpu").
#' @param seconds_per_lp_grad_call,fewest_seconds_of_gradient_per_WCP_thread With both given (default NULL: not
#'   used, as before): T, when not given, is T of the rule only when one thread's share of a gradient,
#'   seconds_per_lp_grad_call / T, is at least fewest_seconds_of_gradient_per_WCP_thread, and 1 otherwise, so that
#'   WCP is used only when each thread's share is well above the per-piece cost of reduce_sum_static (scored on
#'   the ground truth of Paper 3's models, in its benchmark folder:
#'   score_changes_to_Paper_1_burnin_rule_on_the_ground_truth_of_every_model.R).
#' @param one_piece_per_thread_when_one_thread_per_chain TRUE: one piece per thread when T = 1 (pieces for load
#'   balance are of no use without other threads); FALSE (default): pieces_per_thread whatever T, as before.
#' @param S_target_of_the_L2_cache_per_core_when_every_burnin_thread_has_a_core TRUE: S_target is the L2 cache of
#'   one core when the burn-in's n_chains_burnin x T threads are at most the cores available (each thread then has
#'   a core of its own), the L3 share otherwise; FALSE (default): the L3 share, as before.
#' @param L2_cache_size_bytes_per_core,N_threads_per_core The L2 cache of one core and the hardware threads per
#'   core (default NULL: read from sysfs when needed).
#' @return A list with n_threads_WCP_burnin, num_chunks_burnin, chunk_size_burnin and every intermediate quantity.
#' @examples
#' ## The Stan LC-MVP model of Paper 1 at N = 10,000 (19,152 bytes of tape per individual), 4 burn-in chains on
#' ## the whole 96-core local-HPC (gives 16 threads per chain and 64 chunks):
#' fn_compute_burnin_n_threads_WCP_and_num_chunks( M_bytes_per_chain                    = 19152 * 10000,
#'                                                 N_units                              = 10000,
#'                                                 n_chains_burnin                      = 4,
#'                                                 N_hardware_threads_available         = 192,
#'                                                 N_L3_caches_available                = 12,
#'                                                 N_hardware_threads_per_L3_available  = 16,
#'                                                 L3_cache_size_bytes                  = 32 * 1024^2)
#' @export
fn_compute_burnin_n_threads_WCP_and_num_chunks <-  function(
        M_bytes_per_chain,
        N_units,
        n_chains_burnin                      = 4,
        pieces_per_thread                    = 2,
        n_threads_WCP_burnin                 = NULL,
        CPU_ids_available                    = NULL,
        N_hardware_threads_available         = NULL,
        N_L3_caches_available                = NULL,
        N_hardware_threads_per_L3_available  = NULL,
        L3_cache_size_bytes                  = NULL,
        sysfs_CPU_directory                  = "/sys/devices/system/cpu",
        seconds_per_lp_grad_call                                               = NULL,
        fewest_seconds_of_gradient_per_WCP_thread                              = NULL,
        one_piece_per_thread_when_one_thread_per_chain                         = FALSE,
        S_target_of_the_L2_cache_per_core_when_every_burnin_thread_has_a_core  = FALSE,
        L2_cache_size_bytes_per_core                                           = NULL,
        N_threads_per_core                                                     = NULL
) {

        ##
        ## ---- the list returned by fn_estimate_initial_M_bytes_per_chain() is accepted in place of the number:
        ##
        if (is.list(x = M_bytes_per_chain) && !is.null(x = M_bytes_per_chain$M_bytes_per_chain)) {
              M_bytes_per_chain <-  M_bytes_per_chain$M_bytes_per_chain
        }
        ##
        fn_check_positive_number <-  function( value,
                                               value_name,
                                               whole_number = FALSE) {

              if (!is.numeric(value) || length(x = value) != 1 || !is.finite(x = value) || value <= 0 ||
                  (whole_number && value != round(x = value))) {
                    stop(paste0("fn_compute_burnin_n_threads_WCP_and_num_chunks: ", value_name,
                                " must be a single positive ", if (whole_number) "whole number" else "number",
                                "; got: ", paste(as.character(value), collapse = ", ")))
              }
              invisible(TRUE)
        }

        fn_check_positive_number(value = M_bytes_per_chain, value_name = "M_bytes_per_chain")
        fn_check_positive_number(value = N_units,           value_name = "N_units",           whole_number = TRUE)
        fn_check_positive_number(value = n_chains_burnin,   value_name = "n_chains_burnin",   whole_number = TRUE)
        fn_check_positive_number(value = pieces_per_thread, value_name = "pieces_per_thread", whole_number = TRUE)
        if (!is.null(x = n_threads_WCP_burnin)) {
              fn_check_positive_number(value = n_threads_WCP_burnin, value_name = "n_threads_WCP_burnin",
                                       whole_number = TRUE)
        }
        ##
        ## ---- the topology of the available CPUs: arguments, or sysfs and the affinity mask when any is missing:
        ##
        topology <-  list( N_hardware_threads_available         = N_hardware_threads_available,
                           N_L3_caches_available                = N_L3_caches_available,
                           N_hardware_threads_per_L3_available  = N_hardware_threads_per_L3_available,
                           L3_cache_size_bytes                  = L3_cache_size_bytes)
        topology_source <-  "argument"
        if (any(vapply(X = topology, FUN = is.null, FUN.VALUE = FALSE))) {
              topology_read <-  fn_read_L3_cache_topology_of_available_CPUs(
                                      CPU_ids_available   = CPU_ids_available,
                                      sysfs_CPU_directory = sysfs_CPU_directory)
              for (topology_value_name in names(x = topology)) {
                    if (is.null(x = topology[[topology_value_name]])) {
                          topology[[topology_value_name]] <-  topology_read[[topology_value_name]]
                    }
              }
              topology_source <-  paste0("sysfs and the CPU affinity mask (CPUs ",
                                         paste(range(topology_read$CPU_ids_available), collapse = " to "), ", ",
                                         topology_read$N_hardware_threads_available, " available)")
        }
        for (topology_value_name in names(x = topology)) {
              fn_check_positive_number(value        = topology[[topology_value_name]],
                                       value_name   = topology_value_name,
                                       whole_number = !grepl(pattern = "_bytes$", x = topology_value_name))
        }
        ##
        ## ---- threads per chain: the hardware threads per chain, at most one L3 cache's worth, as a power of
        ##      two, so that it divides the number of pieces (also a power of two); or the value given:
        ##
        n_threads_per_chain_from_hardware <-  max(1, floor(topology$N_hardware_threads_available /
                                                           n_chains_burnin))
        n_threads_per_chain_from_L3_cache <-  topology$N_hardware_threads_per_L3_available
        WCP_threads_limited_by_the_gradient <-  FALSE
        if (is.null(x = n_threads_WCP_burnin)) {
              n_threads_WCP_burnin <-  2^floor(log2(min(n_threads_per_chain_from_hardware,
                                                        n_threads_per_chain_from_L3_cache)))
              ## (with the lp_grad time: 1 thread per chain when one thread's share of a gradient is below
              ##  fewest_seconds_of_gradient_per_WCP_thread)
              if (!is.null(seconds_per_lp_grad_call) && !is.null(fewest_seconds_of_gradient_per_WCP_thread) &&
                  (seconds_per_lp_grad_call / n_threads_WCP_burnin < fewest_seconds_of_gradient_per_WCP_thread)) {
                    n_threads_WCP_burnin <-  1
                    WCP_threads_limited_by_the_gradient <-  TRUE
              }
        }
        ##
        ## ---- the pieces: each fits in its thread's share of the L3 cache, and pieces_per_thread per thread:
        ##
        N_active_threads_per_L3 <-  min(topology$N_hardware_threads_per_L3_available,
                                        ceiling(n_chains_burnin * n_threads_WCP_burnin /
                                                topology$N_L3_caches_available))
        S_target_bytes <-  topology$L3_cache_size_bytes / N_active_threads_per_L3
        ## (the L2 cache of one core when every burn-in thread has a core of its own)
        S_target_is_the_L2_cache_per_core <-  FALSE
        if (isTRUE(S_target_of_the_L2_cache_per_core_when_every_burnin_thread_has_a_core)) {
              if (is.null(L2_cache_size_bytes_per_core) || is.null(N_threads_per_core)) {
                    first_CPU <-  fn_read_threads_per_core_and_L2_cache_size_of_first_CPU_of_this_process(
                                        sysfs_CPU_directory = sysfs_CPU_directory)
                    if (is.null(L2_cache_size_bytes_per_core)) {
                          L2_cache_size_bytes_per_core <-  first_CPU$L2_cache_size_bytes
                    }
                    if (is.null(N_threads_per_core)) N_threads_per_core <-  first_CPU$N_threads_per_core
              }
              N_cores_available <-  floor(topology$N_hardware_threads_available / N_threads_per_core)
              if (!is.null(L2_cache_size_bytes_per_core) &&
                  (n_chains_burnin * n_threads_WCP_burnin <= N_cores_available)) {
                    S_target_bytes <-  L2_cache_size_bytes_per_core
                    S_target_is_the_L2_cache_per_core <-  TRUE
              }
        }
        N_pieces_for_cache <-  ceiling(M_bytes_per_chain / S_target_bytes)
        ## (one piece per thread at one thread per chain, when asked)
        if (isTRUE(one_piece_per_thread_when_one_thread_per_chain) && (n_threads_WCP_burnin == 1)) {
              pieces_per_thread <-  1
        }
        N_pieces_for_load_balance <-  n_threads_WCP_burnin * pieces_per_thread
        ##
        num_chunks_burnin <-  2^ceiling(log2(max(N_pieces_for_cache, N_pieces_for_load_balance)))
        chunk_size_burnin <-  ceiling(N_units / num_chunks_burnin)
        ##
        return(list( n_threads_WCP_burnin                 = n_threads_WCP_burnin,
                     num_chunks_burnin                    = num_chunks_burnin,
                     chunk_size_burnin                    = chunk_size_burnin,
                     tape_bytes_per_piece                 = M_bytes_per_chain / num_chunks_burnin,
                     M_bytes_per_chain                    = M_bytes_per_chain,
                     N_units                              = N_units,
                     n_chains_burnin                      = n_chains_burnin,
                     pieces_per_thread                    = pieces_per_thread,
                     n_threads_per_chain_from_hardware    = n_threads_per_chain_from_hardware,
                     n_threads_per_chain_from_L3_cache    = n_threads_per_chain_from_L3_cache,
                     N_active_threads_per_L3              = N_active_threads_per_L3,
                     S_target_bytes                       = S_target_bytes,
                     N_pieces_for_cache                   = N_pieces_for_cache,
                     N_pieces_for_load_balance            = N_pieces_for_load_balance,
                     WCP_threads_limited_by_the_gradient  = WCP_threads_limited_by_the_gradient,
                     S_target_is_the_L2_cache_per_core    = S_target_is_the_L2_cache_per_core,
                     topology                             = topology,
                     topology_source                      = topology_source))

}




#### ---- the burn-in iteration timed for a few chunk counts, in NicoStan's own burn-in worker -------------------

#' Seconds per burn-in iteration of a Stan model for each of a few chunk counts, in NicoStan's burn-in worker
#'
#' Times NicoStan's persistent burn-in worker (all burn-in chains advanced one diffusion-HMC iteration in
#' parallel, as in NicoStan's burn-in, with the shared TBB pool of n_chains_burnin * n_threads_WCP_burnin
#' threads) for each chunk count in num_chunks_candidates, at a tiny step size (eps = 1e-5, L = 20 leapfrog
#' steps, so that every chunk count does the same work), in n_rounds rounds that each time every candidate once
#' (so that a drift of the machine affects every candidate alike), every iteration with the same trajectory
#' length, L leapfrog steps (not the burn-in's random length between 0 and 2 L, which would make a timing of
#' one iteration and a timing of several time different numbers of steps). This is how Paper 1's burn-in
#' benchmark timed its configurations. Each candidate's chunk_size = ceiling(N / num_chunks) is written to a
#' temporary JSON data file.
#'
#' @param init_object The model's initialisation object (from NicoStan's initialise_model(): it holds
#'   Model_args_as_Rcpp_List, n_params_main and n_nuisance).
#' @param Stan_data_list The Stan data (with integer fields N and chunk_size).
#' @param n_chains_burnin,n_threads_WCP_burnin The burn-in chains and the within-chain threads of each.
#' @param num_chunks_candidates The chunk counts to time.
#' @param n_rounds Rounds of timings (default 3; the median over the rounds is used).
#' @param n_timed_iterations The fewest burn-in iterations timed per candidate and round (default 3, after one
#'   untimed); more when they take less than minimum_seconds_per_timing.
#' @param minimum_seconds_per_timing The shortest time of each candidate's timing in each round (default 0.2 s),
#'   so that fast models are not timed at the resolution of the clock.
#' @param L_main,eps_main The leapfrog steps and step size (default 20 and 1e-5).
#' @param environment_of_workers_kept_for_reuse An environment in which each layout's worker is kept between
#'   calls, so that a layout timed again reuses its worker, without its data file or untimed iteration (default
#'   NULL: a new worker for every timing, removed after it); see fn_remove_workers_kept_for_reuse().
#' @param n_untimed_iterations_before_timing Iterations of each new worker run before its timed iterations, so
#'   that the timing does not include its first iteration (default 0: the first iteration is timed).
#' @param pass_the_data_to_the_worker_as_JSON_text_instead_of_a_file TRUE: each new worker is given the data's
#'   JSON text itself, with the layout's chunk_size (BridgeStan reads a string that does not end in .json as
#'   JSON), so that no data file is written per layout (about 0.02 s per layout at N = 25,000); FALSE (default): a
#'   data file per layout, as before. Needs JSON_text_of_Stan_data_list.
#' @param main_sized_outputs_only_as_in_the_resident_burn_in TRUE (default): each iteration returns only its
#'   main-sized quantities to R, the nuisance values staying in the worker, as NicoStan's burn-in does
#'   (resident_burnin = TRUE in R_fn_init_and_run_burnin_CHESS.R); FALSE: each iteration also copies every
#'   chain's nuisance values, velocities and their start and proposed values into new R matrices (the burn-in's
#'   older interface, which made every burn-in timing pay for six n_nuisance x n_chains copies per iteration).
#' @return A data frame: num_chunks and median_seconds_per_burnin_iteration.
#' @keywords internal
## fn_time_burnin_iteration_for_each_num_chunks <-  function( init_object,
##                                                            Stan_data_list,
##                                                            n_chains_burnin,
##                                                            n_threads_WCP_burnin,
##                                                            num_chunks_candidates,
##                                                            n_rounds            = 3,
##                                                            n_timed_iterations  = 3,
##                                                            minimum_seconds_per_timing = 0.2,
##                                                            L_main              = 20,
##                                                            eps_main            = 1e-5
## ) {
##
##         n_params_main <-  init_object$n_params_main
##         n_nuisance <-  init_object$n_nuisance
##         ##
##         ## ---- one JSON data file per candidate:
##         ##
##         json_file_paths <-  vapply(X = num_chunks_candidates, FUN = function(num_chunks) {
##               Stan_data_for_candidate <-  Stan_data_list
##               Stan_data_for_candidate$chunk_size <-  as.integer(ceiling(Stan_data_list$N / num_chunks))
##            json_file_path <-  tempfile(pattern = paste0("burnin_chunks_", num_chunks, "_"), fileext = ".json")
##               cmdstanr::write_stan_json(data = Stan_data_for_candidate, file = json_file_path)
##               normalizePath(json_file_path, mustWork = TRUE)
##         }, FUN.VALUE = "")
##         on.exit(unlink(x = json_file_paths), add = TRUE)
##         ##
##      ## ---- tiny step size and L leapfrog steps, the default metric, and initial values close to zero on the
##         ##      unconstrained scale (deterministic, so that R's random number stream is not touched):
##         ##
##         EHMC_args_as_Rcpp_List <-  init_EHMC_args_as_Rcpp_List( diffusion_HMC            = TRUE,
##                                                                 diffusion_HMC_integrator = "kick_flow_kick")
##         EHMC_args_as_Rcpp_List$share_tau_ii_across_chains <-  TRUE
##         EHMC_args_as_Rcpp_List$eps_main <-  eps_main
##         EHMC_args_as_Rcpp_List$tau_main <-  L_main * eps_main
##         EHMC_args_as_Rcpp_List$eps_us <-  eps_main
##         EHMC_args_as_Rcpp_List$tau_us <-  L_main * eps_main
##         EHMC_Metric_as_Rcpp_List <-  init_EHMC_Metric_as_Rcpp_List( n_params_main     = n_params_main,
##                                                                     n_nuisance        = n_nuisance,
##                                                                     metric_shape_main = "dense")
##         fn_small_values <-  function(n_values) 0.005 * (((seq_len(n_values) * 7) %% 5) - 2)
##         ## initial_main <-  matrix(fn_small_values(n_params_main * n_chains_burnin), nrow = n_params_main)
##         ## initial_nuisance <-  matrix(fn_small_values(n_nuisance * n_chains_burnin), nrow = n_nuisance)
##      ## (one column per chain also for a model without nuisance parameters, i.e. a 0 x n_chains_burnin matrix)
##         initial_main <-  matrix(fn_small_values(n_params_main * n_chains_burnin), nrow = n_params_main,
##                                 ncol = n_chains_burnin)
##         initial_nuisance <-  matrix(fn_small_values(n_nuisance * n_chains_burnin), nrow = n_nuisance,
##                                     ncol = n_chains_burnin)
##         ##
##         RcppParallel::setThreadOptions(numThreads = n_chains_burnin * n_threads_WCP_burnin)
##         seconds <-  matrix(NA_real_, nrow = n_rounds, ncol = length(num_chunks_candidates))
##         for (round_index in seq_len(n_rounds)) {
##               for (candidate_index in seq_along(num_chunks_candidates)) {
##
##                     Model_args_as_Rcpp_List <-  init_object$Model_args_as_Rcpp_List
##                     Model_args_as_Rcpp_List$json_file_path <-  json_file_paths[candidate_index]
##                     worker <-  fn_create_persistent_burnin_worker(
##                                      n_threads_R = n_chains_burnin, partitioned_HMC_R = FALSE,
##                                      diffusion_HMC_R = TRUE, Model_type_R = "Stan", sample_nuisance_R = TRUE,
##                                      force_autodiff_R = TRUE, force_PartialLog_R = FALSE,
##                                      multi_attempts_R = FALSE,
##                                      y_Eigen_R = matrix(0, nrow = 1, ncol = 1),
##                                      Model_args_as_Rcpp_List = Model_args_as_Rcpp_List,
##                                      EHMC_args_as_Rcpp_List = EHMC_args_as_Rcpp_List,
##                                      EHMC_Metric_as_Rcpp_List = EHMC_Metric_as_Rcpp_List,
##                                      n_threads_WCP = n_threads_WCP_burnin)
##                     fn_persistent_burnin_update_adaptation(worker, EHMC_args_as_Rcpp_List,
##                                                            EHMC_Metric_as_Rcpp_List)
##                     fn_persistent_burnin_set_theta(worker, initial_main, initial_nuisance)
##                     ## (one untimed iteration, then at least n_timed_iterations and at least
##                     ##  minimum_seconds_per_timing of timed iterations)
##                     fn_persistent_burnin_run_one_iter(worker, 1000 * round_index, 1)
##                     start_time <-  proc.time()[["elapsed"]]
##                     n_iterations_timed <-  0
##                     while (n_iterations_timed < n_timed_iterations ||
##                            proc.time()[["elapsed"]] - start_time < minimum_seconds_per_timing) {
##                           n_iterations_timed <-  n_iterations_timed + 1
##                        fn_persistent_burnin_run_one_iter(worker, 1000 * round_index + 1 + n_iterations_timed,
##                                                             1 + n_iterations_timed)
##                     }
##                     seconds_per_iteration <-  (proc.time()[["elapsed"]] - start_time) / n_iterations_timed
##                     rm(worker)
##                     invisible(gc(verbose = FALSE))
##                     seconds[round_index, candidate_index] <-  mean(seconds_per_iteration)
##
##               }
##         }
##         return(data.frame( num_chunks                          = num_chunks_candidates,
##                            median_seconds_per_burnin_iteration = apply(X = seconds, MARGIN = 2,
##                                                                        FUN = stats::median)))
##
## }


fn_time_burnin_iteration_for_each_num_chunks <-  function( init_object,
                                                           Stan_data_list,
                                                           n_chains_burnin,
                                                           n_threads_WCP_burnin,
                                                           num_chunks_candidates,
                                                           n_rounds            = 3,
                                                           n_timed_iterations  = 3,
                                                           minimum_seconds_per_timing = 0.2,
                                                           L_main              = 20,
                                                           eps_main            = 1e-5,
                                                           environment_of_workers_kept_for_reuse  = NULL,
                                                           JSON_text_of_Stan_data_list            = NULL,
                                                           snapshot_hardware_counters_around_timed_iterations =
                                                                 FALSE,
                                                           n_untimed_iterations_before_timing     = 0,
                                                           initial_values_of_the_chains           = NULL,
                                                           main_sized_outputs_only_as_in_the_resident_burn_in =
                                                                 TRUE,
                                                      pass_the_data_to_the_worker_as_JSON_text_instead_of_a_file =
                                                                 FALSE,
                                                           ## (each untimed iteration's leapfrog steps; NULL:
                                                           L_of_the_untimed_iterations_before_timing = NULL
) {

        n_params_main <-  init_object$n_params_main
        ## (a gradient of the model's library runs in this R process: recorded, see
        ##  fn_a_gradient_of_the_library_ran_in_this_R_process())
        fn_record_that_a_gradient_of_the_library_ran_in_this_R_process(model_so_file = init_object$model_so_file)
        n_nuisance <-  init_object$n_nuisance
        ##
        ## ---- one JSON data file per candidate (written only for a worker which is created, i.e. not for a
        ##      worker kept for reuse):
        ##
        fn_write_JSON_data_file_of_candidate <-  function(num_chunks) {

              chunk_size <-  as.integer(ceiling(Stan_data_list$N / num_chunks))
              ## (the data's JSON text itself, with the layout's chunk_size, given to the worker in place of a
              ##  file)
              if (isTRUE(pass_the_data_to_the_worker_as_JSON_text_instead_of_a_file) &&
                  !is.null(JSON_text_of_Stan_data_list)) {
                    return(fn_JSON_text_of_Stan_data_list_with_chunk_size(
                                 JSON_text_of_Stan_data_list  = JSON_text_of_Stan_data_list,
                                 chunk_size                   = chunk_size))
              }
              json_file_path <-  tempfile(pattern = paste0("burnin_chunks_", num_chunks, "_"), fileext = ".json")
              if (is.null(JSON_text_of_Stan_data_list)) {
                    ## (the whole data written as JSON, about 0.7 s for N = 5000)
                    Stan_data_for_candidate <-  Stan_data_list
                    Stan_data_for_candidate$chunk_size <-  chunk_size
                    cmdstanr::write_stan_json(data = Stan_data_for_candidate, file = json_file_path)
              } else {
                    ## (the data's JSON text, written once by fn_JSON_text_of_Stan_data_list(), with only its
                    ##  chunk_size changed: milliseconds)
                    # JSON_text_of_candidate <-  sub(pattern = "\"chunk_size\":[[:space:]]*[0-9]+",
                    #                               replacement = paste0("\"chunk_size\": ", chunk_size),
                    #                               x = JSON_text_of_Stan_data_list)
                    ## (sub() on the whole text re-encodes and scans all of it: 1.7 s for 37 MB; the pieces: ms)
                    JSON_text_of_candidate <-  fn_JSON_text_of_Stan_data_list_with_chunk_size(
                                                     JSON_text_of_Stan_data_list  = JSON_text_of_Stan_data_list,
                                                     chunk_size                   = chunk_size)
                    writeChar(object = JSON_text_of_candidate, con = json_file_path, eos = NULL, useBytes = TRUE)
              }
              return(normalizePath(json_file_path, mustWork = TRUE))
        }

        ## json_file_paths <-  vapply(X = num_chunks_candidates, FUN = function(num_chunks) {
        ##       Stan_data_for_candidate <-  Stan_data_list
        ##       Stan_data_for_candidate$chunk_size <-  as.integer(ceiling(Stan_data_list$N / num_chunks))
##             json_file_path <-  tempfile(pattern = paste0("burnin_chunks_", num_chunks, "_"), fileext = ".json")
        ##       cmdstanr::write_stan_json(data = Stan_data_for_candidate, file = json_file_path)
        ##       normalizePath(json_file_path, mustWork = TRUE)
        ## }, FUN.VALUE = "")
        ## on.exit(unlink(x = json_file_paths), add = TRUE)
        json_file_paths_of_workers_not_kept <-  character(0)
        ## on.exit(unlink(x = json_file_paths_of_workers_not_kept), add = TRUE)
        ## (only data files are removed: the data given as JSON text is not a file)
        on.exit(unlink(x = json_file_paths_of_workers_not_kept[grepl(pattern = "\\.json$",
                                                                     x = json_file_paths_of_workers_not_kept)]),
                add = TRUE)
        ##
        ## ---- tiny step size and L leapfrog steps, the default metric, and initial values close to zero on the
        ##      unconstrained scale (deterministic, so that R's random number stream is not touched):
        ##
        EHMC_args_as_Rcpp_List <-  init_EHMC_args_as_Rcpp_List( diffusion_HMC            = TRUE,
                                                                diffusion_HMC_integrator = "kick_flow_kick")
        EHMC_args_as_Rcpp_List$share_tau_ii_across_chains <-  TRUE
        EHMC_args_as_Rcpp_List$eps_main <-  eps_main
        EHMC_args_as_Rcpp_List$tau_main <-  L_main * eps_main
        EHMC_args_as_Rcpp_List$eps_us <-  eps_main
        EHMC_args_as_Rcpp_List$tau_us <-  L_main * eps_main
        EHMC_Metric_as_Rcpp_List <-  init_EHMC_Metric_as_Rcpp_List( n_params_main     = n_params_main,
                                                                    n_nuisance        = n_nuisance,
                                                                    # metric_shape_main = "dense")
                                                                    ##  "EVERYWHERE": the main blocks are small now that every nuisance block is named)
                                                                    # metric_shape_main = "diag")
                                                                    metric_shape_main = "dense")
        fn_small_values <-  function(n_values) 0.005 * (((seq_len(n_values) * 7) %% 5) - 2)
        ## initial_main <-  matrix(fn_small_values(n_params_main * n_chains_burnin), nrow = n_params_main)
        ## initial_nuisance <-  matrix(fn_small_values(n_nuisance * n_chains_burnin), nrow = n_nuisance)
        ## (one column per chain also for a model without nuisance parameters, i.e. a 0 x n_chains_burnin matrix)
        initial_main <-  matrix(fn_small_values(n_params_main * n_chains_burnin), nrow = n_params_main,
                                ncol = n_chains_burnin)
        initial_nuisance <-  matrix(fn_small_values(n_nuisance * n_chains_burnin), nrow = n_nuisance,
                                    ncol = n_chains_burnin)
        ## (given starting values instead, e.g. those of a model whose constraint rejects the default: the first
        ##  n_chains_burnin chains)
        if (!is.null(initial_values_of_the_chains)) {

              stopifnot(ncol(initial_values_of_the_chains$main) >= n_chains_burnin,
                        nrow(initial_values_of_the_chains$main) == n_params_main,
                        nrow(initial_values_of_the_chains$nuisance) == n_nuisance)
              initial_main <-  initial_values_of_the_chains$main[, seq_len(n_chains_burnin), drop = FALSE]
              initial_nuisance <-  initial_values_of_the_chains$nuisance[, seq_len(n_chains_burnin),
                                                                         drop = FALSE]
        }

        ##
        ## (every iteration with the same trajectory length, L_main leapfrog steps, in every chain: the burn-in
        ##  draws each iteration's trajectory length uniformly between 0 and twice tau, so that a timing of one
        ##  iteration and a timing of several would otherwise time different numbers of leapfrog steps)
        tau_of_every_chain <-  rep(L_main * eps_main, n_chains_burnin)
        ##
        RcppParallel::setThreadOptions(numThreads = n_chains_burnin * n_threads_WCP_burnin)
        seconds <-  matrix(NA_real_, nrow = n_rounds, ncol = length(num_chunks_candidates))
        ## (the transitions of the timed iterations, and how many of them were divergent: always 0 when the
        ##  timings are returned, since any divergent transition stops the timing; saved with every timing)
        n_divergent_transitions <-  matrix(0, nrow = n_rounds, ncol = length(num_chunks_candidates))
        n_transitions <-  matrix(0, nrow = n_rounds, ncol = length(num_chunks_candidates))
        for (round_index in seq_len(n_rounds)) {
              for (candidate_index in seq_along(num_chunks_candidates)) {

                    ##
                    ## ---- the candidate's worker: kept from an earlier timing of the same layout (then already
                    ##      past its untimed iteration), or created, given its data, values and one untimed
                    ##      iteration:
                    ##
                    name_of_layout <-  paste0(n_chains_burnin, "_chains_x_", n_threads_WCP_burnin, "_threads_x_",
                                              num_chunks_candidates[candidate_index], "_chunks_L_", L_main)
                    ## (the start of the layout's timing, for its start-up seconds: its data file, worker and
                    ##  untimed iterations)
                    time_of_layout_start <-  proc.time()[["elapsed"]]
                    seconds_of_data_file_of_layout <-  0
                    seconds_of_data_file_and_worker_of_layout <-  0
                    kept <-  if (is.null(environment_of_workers_kept_for_reuse)) NULL else
                                   environment_of_workers_kept_for_reuse[[name_of_layout]]
                    if (!is.null(kept)) {
                          worker <-  kept$worker
                          n_iterations_run <-  kept$n_iterations_run
                    } else {
                          json_file_path <-  fn_write_JSON_data_file_of_candidate(
                                                   num_chunks_candidates[candidate_index])
                          seconds_of_data_file_of_layout <-  proc.time()[["elapsed"]] - time_of_layout_start
                          Model_args_as_Rcpp_List <-  init_object$Model_args_as_Rcpp_List
                          ## Model_args_as_Rcpp_List$json_file_path <-  json_file_paths[candidate_index]
                          Model_args_as_Rcpp_List$json_file_path <-  json_file_path
                          worker <-  fn_create_persistent_burnin_worker(
                                           n_threads_R = n_chains_burnin, partitioned_HMC_R = FALSE,
                                           diffusion_HMC_R = TRUE, Model_type_R = "Stan",
                                           sample_nuisance_R = TRUE,
                                           force_autodiff_R = TRUE, force_PartialLog_R = FALSE,
                                           multi_attempts_R = FALSE,
                                           y_Eigen_R = matrix(0, nrow = 1, ncol = 1),
                                           Model_args_as_Rcpp_List = Model_args_as_Rcpp_List,
                                           EHMC_args_as_Rcpp_List = EHMC_args_as_Rcpp_List,
                                           EHMC_Metric_as_Rcpp_List = EHMC_Metric_as_Rcpp_List,
                                           n_threads_WCP = n_threads_WCP_burnin)
                          fn_persistent_burnin_update_adaptation(worker, EHMC_args_as_Rcpp_List,
                                                                 EHMC_Metric_as_Rcpp_List)
                          fn_persistent_burnin_set_theta(worker, initial_main, initial_nuisance)
                          seconds_of_data_file_and_worker_of_layout <-  proc.time()[["elapsed"]] -
                                                                        time_of_layout_start
##                        ## (one untimed iteration)
##                        ## fn_persistent_burnin_run_one_iter(worker, 1000 * round_index, 1)
##                        fn_persistent_burnin_run_one_iter_tau_jitter(worker, 1000 * round_index, 1,
##                                                                     tau_of_every_chain, tau_of_every_chain,
##                                                                     FALSE, FALSE)
##                        n_iterations_run <-  1
                          ## (no untimed iteration: the timing starts with the worker's first iteration)
                          n_iterations_run <-  0
                          ## (n_untimed_iterations_before_timing untimed iterations first, so that the timing does
                          ##  not include the first iteration of the new worker)
                          for (untimed_iteration in seq_len(n_untimed_iterations_before_timing)) {
                                seed_of_untimed_iteration <-  1000 * round_index + untimed_iteration
                                ## (the untimed iteration at its own leapfrog steps, when given)
                                tau_of_every_chain_of_the_untimed_iteration <-  tau_of_every_chain
                                if (!is.null(L_of_the_untimed_iterations_before_timing)) {
                                      tau_of_every_chain_of_the_untimed_iteration <-
                                            rep(L_of_the_untimed_iterations_before_timing * eps_main,
                                                n_chains_burnin)
                                }
                                untimed_iter_outputs <-  fn_persistent_burnin_run_one_iter_tau_jitter(
                                                               worker, seed_of_untimed_iteration,
                                                               untimed_iteration,
                                                               tau_of_every_chain_of_the_untimed_iteration,
                                                               tau_of_every_chain_of_the_untimed_iteration,
                                                               main_sized_outputs_only_as_in_the_resident_burn_in,
                                                               FALSE)
                                ## (no divergent transition is allowed in the untimed iterations either)
                                n_divs_untimed_iter <-
                                      sum(untimed_iter_outputs$other_main_out_vector_all_chains_output_to_R[2, ])
                                if (n_divs_untimed_iter > 0) {
                                      stop("Burn-in timing encountered ", n_divs_untimed_iter,
                                           " divergent transition(s) in untimed iter ", untimed_iteration,
                                           ". The configuration is invalid for a timing comparison.")
                                }
                                n_iterations_run <-  n_iterations_run + 1
                          }
                          if (is.null(environment_of_workers_kept_for_reuse)) {
                                json_file_paths_of_workers_not_kept <-  c(json_file_paths_of_workers_not_kept,
                                                                          json_file_path)
                          }
                    }
                    ## ## (one untimed iteration, then at least n_timed_iterations and at least
                    ## ##  minimum_seconds_per_timing of timed iterations)
                    ## fn_persistent_burnin_run_one_iter(worker, 1000 * round_index, 1)
                    ## (at least n_timed_iterations and at least minimum_seconds_per_timing of timed iterations,
                    ##  between two snapshots of the hardware counters when they are taken)
                    if (isTRUE(snapshot_hardware_counters_around_timed_iterations)) {
                          fn_snapshot_hardware_counters()
                    }
                    start_time <-  proc.time()[["elapsed"]]
                    seconds_of_start_up_of_layout <-  start_time - time_of_layout_start
                    n_iterations_timed <-  0
                    while (n_iterations_timed < n_timed_iterations ||
                           proc.time()[["elapsed"]] - start_time < minimum_seconds_per_timing) {
                          n_iterations_timed <-  n_iterations_timed + 1
##                        fn_persistent_burnin_run_one_iter(worker, 1000 * round_index + 1 + n_iterations_timed,
##                                                          1 + n_iterations_timed)
##                        fn_persistent_burnin_run_one_iter(worker, 1000 * round_index + n_iterations_run +
##                                                                  n_iterations_timed,
##                                                          n_iterations_run + n_iterations_timed)
                          seed_of_iteration <-  1000 * round_index + n_iterations_run + n_iterations_timed
                          # fn_persistent_burnin_run_one_iter_tau_jitter(worker, seed_of_iteration,
                          #                                              n_iterations_run + n_iterations_timed,
                          #                                              tau_of_every_chain, tau_of_every_chain,
                          #                                              FALSE, FALSE)
                          burnin_iter_outputs <-  fn_persistent_burnin_run_one_iter_tau_jitter(
                                                        worker, seed_of_iteration,
                                                        n_iterations_run + n_iterations_timed,
                                                        tau_of_every_chain, tau_of_every_chain,
                                                        main_sized_outputs_only_as_in_the_resident_burn_in, FALSE)
                          ## (as Paper 1's burn-in timings, R_fn_run_ps_1_burnin_benchmark.R: eps is tiny and L
                          ##  fixed, so nothing should diverge; a divergent transition makes the configuration
                          ##  invalid)
                          n_divs_this_iter <-
                                sum(burnin_iter_outputs$other_main_out_vector_all_chains_output_to_R[2, ])
                          if (n_divs_this_iter > 0) {
                                stop("Burn-in timing encountered ", n_divs_this_iter,
                                     " divergent transition(s) in timed iter ", n_iterations_timed,
                                     ". The configuration is invalid for a timing comparison.")
                          }
                          n_divergent_transitions[round_index, candidate_index] <-
                                n_divergent_transitions[round_index, candidate_index] + n_divs_this_iter
                          n_transitions[round_index, candidate_index] <-
                                n_transitions[round_index, candidate_index] +
                                length(burnin_iter_outputs$other_main_out_vector_all_chains_output_to_R[2, ])
                    }
                    seconds_per_iteration <-  (proc.time()[["elapsed"]] - start_time) / n_iterations_timed
                    if (isTRUE(snapshot_hardware_counters_around_timed_iterations)) {
                          fn_snapshot_hardware_counters()
                    }
                    ## rm(worker)
                    ## invisible(gc(verbose = FALSE))
                    ## (the worker kept for reuse, or else removed)
                    if (!is.null(environment_of_workers_kept_for_reuse)) {
                          if (is.null(kept)) {
                                kept <-  list(worker = worker, json_file_path = json_file_path)
                          }
                          kept$n_iterations_run <-  n_iterations_run + n_iterations_timed
                          environment_of_workers_kept_for_reuse[[name_of_layout]] <-  kept
                    } else {
                          rm(worker)
                          ## (no gc() per layout: the worker is freed at R's next collection)
                          ## invisible(gc(verbose = FALSE))
                    }
                    seconds[round_index, candidate_index] <-  mean(seconds_per_iteration)
                    ## (the layout's start-up and clean-up seconds next to its timed seconds: anything above a
                    ##  few hundredths of a second outside the timed iterations is to be removed)
                    message(paste0("  burn-in layout ", name_of_layout, ": start-up ",
                                   round(seconds_of_start_up_of_layout, 3), " s (data file ",
                                   round(seconds_of_data_file_of_layout, 3), " s, worker ",
                                   round(seconds_of_data_file_and_worker_of_layout -
                                         seconds_of_data_file_of_layout, 3), " s, untimed iterations ",
                                   round(seconds_of_start_up_of_layout -
                                         seconds_of_data_file_and_worker_of_layout, 3), " s), ",
                                   n_iterations_timed,
                                   " timed iterations in ", round(seconds_per_iteration * n_iterations_timed, 3),
                                   " s, then ", round(proc.time()[["elapsed"]] - start_time -
                                                      seconds_per_iteration * n_iterations_timed, 3),
                                   " s to the next layout"))

              }
        }
        return(data.frame( num_chunks                                   = num_chunks_candidates,
                           median_seconds_per_burnin_iteration          = apply(X = seconds, MARGIN = 2,
                                                                                FUN = stats::median),
                           n_divergent_transitions_of_timed_iterations  = colSums(n_divergent_transitions),
                           n_transitions_of_timed_iterations            = colSums(n_transitions)))

}




#### ---- the workers kept for reuse by fn_time_burnin_iteration_for_each_num_chunks(), removed ---------------

#' Removes the workers kept for reuse by fn_time_burnin_iteration_for_each_num_chunks(), and their data files
#'
#' @param environment_of_workers_kept_for_reuse The environment in which the workers were kept.
#' @return NULL (invisibly).
#' @keywords internal
fn_remove_workers_kept_for_reuse <-  function(environment_of_workers_kept_for_reuse) {

        ## (nothing to remove: no collection, which costs about 0.1 s)
        if (length(ls(envir = environment_of_workers_kept_for_reuse)) == 0) return(invisible(NULL))
        for (name_of_layout in ls(envir = environment_of_workers_kept_for_reuse)) {
              ## unlink(x = environment_of_workers_kept_for_reuse[[name_of_layout]]$json_file_path)
              ## (only data files are removed: the data given as JSON text is not a file)
              json_file_path <-  environment_of_workers_kept_for_reuse[[name_of_layout]]$json_file_path
              if (is.character(json_file_path) && grepl(pattern = "\\.json$", x = json_file_path)) {
                    unlink(x = json_file_path)
              }
        }
        rm(list = ls(envir = environment_of_workers_kept_for_reuse),
           envir = environment_of_workers_kept_for_reuse)
        invisible(gc(verbose = FALSE))
        return(invisible(NULL))

}




#### ---- the JSON text of Stan data, written once, so that each layout's data file only changes chunk_size ---

#' The JSON text of Stan data, written once, so that each layout's data file only changes chunk_size
#'
#' Writes the Stan data as cmdstanr::write_stan_json() does, once, and returns the text, so that
#' fn_time_burnin_iteration_for_each_num_chunks() writes each layout's data file by changing only its chunk_size
#' (milliseconds) rather than writing the whole data again (about 0.7 s per layout for N = 5000).
#'
#' @param Stan_data_list The Stan data (with integer fields N and chunk_size).
#' @return The JSON text, one string with exactly one chunk_size field.
#' @export
fn_JSON_text_of_Stan_data_list <-  function(Stan_data_list) {

        json_file_path <-  tempfile(fileext = ".json")
        on.exit(unlink(x = json_file_path), add = TRUE)
        cmdstanr::write_stan_json(data = Stan_data_list, file = json_file_path)
        JSON_text <-  readChar(con = json_file_path, nchars = file.info(json_file_path)$size, useBytes = TRUE)
        positions_of_chunk_size <-  gregexpr(pattern = "\"chunk_size\":", text = JSON_text, fixed = TRUE)[[1]]
        if ((positions_of_chunk_size[1] < 0) || (length(positions_of_chunk_size) != 1)) {
              stop("fn_JSON_text_of_Stan_data_list: the Stan data must have exactly one chunk_size field.")
        }
        return(JSON_text)

}




#### ---- the JSON text of Stan data with another chunk_size, without a regular expression ----------------------

#' The JSON text of Stan data with another chunk_size, without a regular expression
#'
#' The same text as sub("\"chunk_size\":[[:space:]]*[0-9]+", ...) gives, found with a fixed search: a regular
#' expression over the JSON text of large data took about 0.02 s per layout at N = 25,000.
#'
#' @param JSON_text_of_Stan_data_list The data's JSON text from fn_JSON_text_of_Stan_data_list().
#' @param chunk_size The chunk_size written into it.
#' @return The JSON text with chunk_size replaced.
#' @keywords internal
## (the two pieces of a JSON text around its chunk_size value, found ONCE per text and kept: every layout then
##  costs one paste0 of the pieces (one copy of the text, tens of milliseconds for 37 MB), where finding the
##  field and taking the two substrings cost three copies and about 0.6 s per layout at N = 50,000 of the
##  four-class LC-MVOP; the text is known by its length in bytes and its first 400 bytes)
pieces_of_the_JSON_texts_around_chunk_size <-  new.env(parent = emptyenv())
fn_JSON_text_of_Stan_data_list_with_chunk_size <-  function( JSON_text_of_Stan_data_list,
                                                             chunk_size
) {

        key_of_text <-  paste0(nchar(JSON_text_of_Stan_data_list, type = "bytes"), "_",
                               substr(x = JSON_text_of_Stan_data_list, start = 1, stop = 400))
        pieces <-  pieces_of_the_JSON_texts_around_chunk_size[[key_of_text]]
        if (is.null(pieces)) {
              position_of_field <-  regexpr(pattern = "\"chunk_size\":", text = JSON_text_of_Stan_data_list,
                                            fixed = TRUE, useBytes = TRUE)
              if (position_of_field < 0) {
                    stop("fn_JSON_text_of_Stan_data_list_with_chunk_size: the JSON text has no chunk_size field.")
              }
              position_after_field <-  position_of_field + attr(position_of_field, "match.length")
              ## (the spaces and the digits of the old value, which a JSON writer puts after the field's colon)
              text_after_field <-  substr(x = JSON_text_of_Stan_data_list, start = position_after_field,
                                          stop = position_after_field + 30)
              length_of_old_value <-  attr(regexpr(pattern = "^[[:space:]]*[0-9]+", text = text_after_field),
                                           "match.length")
              if (length_of_old_value < 1) {
                    stop(paste0("fn_JSON_text_of_Stan_data_list_with_chunk_size: the chunk_size field has no ",
                                "whole number."))
              }
              pieces <-  list(before = substr(x = JSON_text_of_Stan_data_list, start = 1,
                                              stop = position_after_field - 1),
                              after  = substr(x = JSON_text_of_Stan_data_list,
                                              start = position_after_field + length_of_old_value,
                                              stop = nchar(JSON_text_of_Stan_data_list, type = "bytes")))
              pieces_of_the_JSON_texts_around_chunk_size[[key_of_text]] <-  pieces
        }
        return(paste0(pieces$before, " ", as.integer(chunk_size), pieces$after))
        # position_of_field <-  regexpr(pattern = "\"chunk_size\":", text = JSON_text_of_Stan_data_list,
        #                               fixed = TRUE)
        # if (position_of_field < 0) {
        #       stop("fn_JSON_text_of_Stan_data_list_with_chunk_size: the JSON text has no chunk_size field.")
        # }
        # position_after_field <-  position_of_field + attr(position_of_field, "match.length")
        # ## (the spaces and the digits of the old value, which a JSON writer puts after the field's colon)
        # text_after_field <-  substr(x = JSON_text_of_Stan_data_list, start = position_after_field,
        #                             stop = position_after_field + 30)
        # length_of_old_value <-  attr(regexpr(pattern = "^[[:space:]]*[0-9]+", text = text_after_field),
        #                              "match.length")
        # if (length_of_old_value < 1) {
        #       stop("fn_JSON_text_of_Stan_data_list_with_chunk_size: the chunk_size field has no whole number.")
        # }
        # return(paste0(substr(x = JSON_text_of_Stan_data_list, start = 1, stop = position_after_field - 1),
        #               " ", as.integer(chunk_size),
        #               substr(x = JSON_text_of_Stan_data_list, start = position_after_field + length_of_old_value,
        #                      stop = nchar(JSON_text_of_Stan_data_list, type = "bytes"))))

}




#### ---- a snapshot of the hardware counters of pmc_stat and umc_stat (Paper 1's E4 counter tools) ----------

#' A snapshot of the hardware counters of pmc_stat and umc_stat, when this process runs under them
#'
#' Sends SIGUSR1 to pmc_stat and umc_stat (Paper 1's E4 counter tools; their process ids are in the environment
#' variables PMC_STAT_PID and UMC_STAT_PID), so that each appends a line with its running totals to its output
#' file (argument 1 of its own command line), then waits until each of those files has grown, usually well under
#' a millisecond and at most 1 s, so that the snapshot is taken before the next work starts and the next snapshot
#' is never merged into this one. Without the tools, it does nothing. (It replaces a fixed pause of 0.25 s per
#' snapshot.)
#'
#' @return NULL (invisibly).
#' @keywords internal
fn_snapshot_hardware_counters <-  function() {

        # for (variable_name in c("PMC_STAT_PID", "UMC_STAT_PID")) {
        #       process_id_of_tool <-  as.integer(Sys.getenv(variable_name, unset = "0"))
        #       if (process_id_of_tool > 0) tools::pskill(pid = process_id_of_tool, signal = tools::SIGUSR1)
        # }
        # Sys.sleep(0.25)
        output_files_of_the_tools <-  character(0)
        sizes_before_the_snapshot <-  numeric(0)
        for (variable_name in c("PMC_STAT_PID", "UMC_STAT_PID")) {
              process_id_of_tool <-  as.integer(Sys.getenv(variable_name, unset = "0"))
              if (is.na(process_id_of_tool) || (process_id_of_tool <= 0)) next
              arguments_of_tool <-  strsplit(paste(system2("ps", c("-o", "args=", "-p", process_id_of_tool),
                                                           stdout = TRUE), collapse = " "),
                                             " ", fixed = TRUE)[[1]]
              output_file_of_tool <-  arguments_of_tool[2]
              if (!is.na(output_file_of_tool) && file.exists(output_file_of_tool)) {
                    output_files_of_the_tools <-  c(output_files_of_the_tools, output_file_of_tool)
                    sizes_before_the_snapshot <-  c(sizes_before_the_snapshot, file.size(output_file_of_tool))
              }
              tools::pskill(pid = process_id_of_tool, signal = tools::SIGUSR1)
        }
        seconds_waited <-  0
        while (length(output_files_of_the_tools) > 0 &&
               any(file.size(output_files_of_the_tools) <= sizes_before_the_snapshot) &&
               (seconds_waited < 1)) {
              Sys.sleep(0.0005)
              seconds_waited <-  seconds_waited + 0.0005
        }
        return(invisible(NULL))

}




#### ---- the hardware threads per core of the CPUs of this process, read from sysfs ---------------------------

#' The hardware threads per core (e.g. 2 with SMT) of the first CPU available to this process, read from sysfs
#'
#' @param sysfs_CPU_directory The sysfs CPU directory (default "/sys/devices/system/cpu").
#' @return The number of hardware threads of that CPU's core (1 when sysfs does not give it).
#' @keywords internal
fn_read_hardware_threads_per_core <-  function(sysfs_CPU_directory = "/sys/devices/system/cpu") {

        CPU_id <-  fn_read_L3_cache_topology_of_available_CPUs()$CPU_ids_available[1]
        siblings_file <-  file.path(sysfs_CPU_directory, paste0("cpu", CPU_id), "topology",
                                    "thread_siblings_list")
        if (!file.exists(siblings_file)) return(1)
        return(length(fn_parse_sysfs_CPU_list(CPU_list_string = readLines(con = siblings_file, n = 1))))

}




#### ---- the burn-in choice saved for reuse: same model, same N, same burn-in chains, same CPUs ---------------

#' The file of NicoStan's saved burn-in choices (n_threads_WCP_burnin and num_chunks_burnin)
#'
#' One row per model file, N, number of burn-in chains and CPU list, in NicoStan's cache folder
#' (tools::R_user_dir("NicoStan", "cache")).
#' @return The path of the file.
#' @keywords internal
fn_saved_burnin_choices_file <-  function() {

        return(file.path(tools::R_user_dir(package = "NicoStan", which = "cache"),
                         "burnin_n_threads_WCP_and_num_chunks_chosen_per_model_N_chains_and_CPUs.rds"))

}

#' CPU ids written as a sysfs-style list, e.g. "16-31,112-127"
#' @param CPU_ids The CPU ids.
#' @return The list as a string.
#' @keywords internal
fn_format_CPU_list <-  function( CPU_ids ) {

        CPU_ids <-  sort(x = unique(x = CPU_ids))
        run_starts <-  CPU_ids[c(TRUE, diff(CPU_ids) != 1)]
        run_ends <-  CPU_ids[c(diff(CPU_ids) != 1, TRUE)]
        runs <-  ifelse(run_starts == run_ends, run_starts, paste0(run_starts, "-", run_ends))
        return(paste(runs, collapse = ","))

}

#' The saved burn-in choice for one model file, N, number of burn-in chains and CPU list (NULL if there is none)
#' @param model_so_file Path of the compiled BridgeStan model library (.so).
#' @param N_units The N of the Stan data.
#' @param n_chains_burnin The number of burn-in chains.
#' @param CPU_list The CPUs available, as fn_format_CPU_list() writes them.
#' @return The saved row, or NULL.
#' @keywords internal
fn_read_saved_burnin_choice <-  function( model_so_file,
                                          N_units,
                                          n_chains_burnin,
                                          CPU_list
) {

        saved_choices_file <-  fn_saved_burnin_choices_file()
        if (!file.exists(saved_choices_file)) return(NULL)
        saved_choices <-  tryCatch(expr = readRDS(file = saved_choices_file), error = function(condition) NULL)
        if (!is.data.frame(saved_choices) || nrow(saved_choices) == 0) return(NULL)
        same_model_N_chains_and_CPUs <-  saved_choices$model_so_file == normalizePath(model_so_file) &
                                         saved_choices$N_units == N_units &
                                         saved_choices$n_chains_burnin == n_chains_burnin &
                                         saved_choices$CPU_list == CPU_list
        if (!any(same_model_N_chains_and_CPUs)) return(NULL)
        return(saved_choices[which(same_model_N_chains_and_CPUs)[1], , drop = FALSE])

}

#' Save a burn-in choice (replacing the saved one for the same model file, N, burn-in chains and CPU list)
#' @param model_so_file,N_units,n_chains_burnin,CPU_list As for fn_read_saved_burnin_choice().
#' @param rule The chosen rule (n_threads_WCP_burnin, num_chunks_burnin, M_bytes_per_chain).
#' @param seconds_spent_choosing The time the choice took (memory measurement and timings).
#' @return The path of the file, invisibly.
#' @keywords internal
fn_save_burnin_choice <-  function( model_so_file,
                                    N_units,
                                    n_chains_burnin,
                                    CPU_list,
                                    rule,
                                    seconds_spent_choosing
) {

        saved_choices_file <-  fn_saved_burnin_choices_file()
        dir.create(path = dirname(saved_choices_file), recursive = TRUE, showWarnings = FALSE)
        new_choice <-  data.frame( model_so_file           = normalizePath(model_so_file),
                                   N_units                 = N_units,
                                   n_chains_burnin         = n_chains_burnin,
                                   CPU_list                = CPU_list,
                                   n_threads_WCP_burnin    = rule$n_threads_WCP_burnin,
                                   num_chunks_burnin       = rule$num_chunks_burnin,
                                   M_bytes_per_chain       = rule$M_bytes_per_chain,
                                   seconds_spent_choosing  = seconds_spent_choosing,
                                   stringsAsFactors        = FALSE)
        saved_choices <-  if (file.exists(saved_choices_file)) {
              tryCatch(expr = readRDS(file = saved_choices_file), error = function(condition) NULL)
        } else NULL
        if (is.data.frame(saved_choices) && nrow(saved_choices) > 0) {

              same_model_N_chains_and_CPUs <-  saved_choices$model_so_file == new_choice$model_so_file &
                                               saved_choices$N_units == N_units &
                                               saved_choices$n_chains_burnin == n_chains_burnin &
                                               saved_choices$CPU_list == CPU_list
              saved_choices <-  rbind(saved_choices[!same_model_N_chains_and_CPUs, , drop = FALSE], new_choice)
        } else {

              saved_choices <-  new_choice
        }
        ## (written through a temporary file renamed over it, as several fits can save at once)
        temporary_file <-  tempfile(pattern = "saved_burnin_choices_", tmpdir = dirname(saved_choices_file),
                                    fileext = ".rds.tmp")
        saveRDS(object = saved_choices, file = temporary_file)
        file.rename(from = temporary_file, to = saved_choices_file)
        invisible(saved_choices_file)

}




#### ---- the rule for a Stan model with NicoStan's N / chunk_size data contract, with its measurements ----------

#' Burn-in within-chain threads and chunks of a Stan model, with M_bytes/chain measured and the chunks timed
#'
#' For a Stan model with NicoStan's N / chunk_size data contract (integer data fields N, the units summed by
#' reduce_sum_static, and chunk_size, the most units per piece): measures M_bytes/chain with
#' [fn_estimate_initial_M_bytes_per_chain()] on the full data evaluated as one piece (chunk_size = N) and applies
#' [fn_compute_burnin_n_threads_WCP_and_num_chunks()]. When the model's init_object is given, the rule's
#' configurations for 1, 2 and 4 pieces per thread, at T, T / 2 and 1 threads per chain, are timed in NicoStan's
#' burn-in worker (fn_time_burnin_iteration_for_each_num_chunks()) and the fastest is used: the best number of
#' pieces per thread depends on the model, T / 2 guards the layouts the rule was not timed on, and 1 thread per
#' chain is the fastest for a model whose likelihood has a long sequential part outside reduce_sum_static (in
#' Paper 3's checks, the stochastic volatility model at T = 5,000 and the shared-frailty Cox model at
#' N = 5,000). The chunk count of the fastest configuration is then checked as in Paper 1's algorithm for
#' choosing N_chunks ([fn_check_num_chunks_by_doubling_and_halving()], at its threads per chain): twice it
#' and
#' half it, then the next doubling (or halving) for as long as it is faster. Without init_object (or if the
#' timing fails), T threads and two pieces per thread.
#'
#' The choice is saved (fn_save_burnin_choice(), in NicoStan's cache folder) and reused, without measuring or
#' timing again, only by a fit of the same model file with the same N, the same number of burn-in chains and the
#' same CPUs; any other fit chooses again. The seconds the choice took are returned (seconds_spent_choosing; 0
#' when reused) and are not part of the fit's burn-in time.
#'
#' @param model_so_file Path of the compiled BridgeStan model library (.so).
#' @param Stan_data_list The Stan data (with integer fields N and chunk_size).
#' @param n_chains_burnin Number of burn-in chains run in parallel (default 4).
#' @param init_list Initial values of one chain (the memory measurement's point; default: zero on the
#'   unconstrained scale).
#' @param initial_values_of_the_chains A list with the matrices main (n_params_main rows) and nuisance
#'   (n_nuisance rows), one column per chain (at least n_chains_burnin), on the unconstrained scale:
#'   the starting values of the burn-in layout timings (default NULL: small values near 0).
#' @param init_object The model's initialisation object (default NULL: the chunk counts are not timed).
#' @param reuse_saved_choice TRUE (default): reuse the choice saved for the same model file, N, number of
#'   burn-in chains and CPUs, when there is one.
#' @param ... Further arguments of [fn_compute_burnin_n_threads_WCP_and_num_chunks()].
#' @param M_bytes_per_chain M_bytes/chain of this model and data (default NULL: measured with
#'   [fn_estimate_initial_M_bytes_per_chain()] on the full data evaluated as one piece); given, it is used
#'   as given (e.g. measured once for this model and N, and reused for other CPUs).
#' @param n_rounds Not used by Paper 1's check (each layout is timed once, and the check's last round times
#'   the layout found and its neighbours once more; default 3).
#' @param n_timed_iterations,minimum_seconds_per_timing Each timing of Paper 1's check: at least
#'   n_timed_iterations iterations and at least minimum_seconds_per_timing seconds of them (see
#'   [fn_time_burnin_iteration_for_each_num_chunks()]; defaults 1 and 0.25 s).
#' @param n_untimed_iterations_before_timing Iterations of each new worker of Paper 1's check run before its timed
#'   iterations (default 1), so that the timing does not count the first touch of its memory.
#' @param n_nuisance_owned_by_NicoStan The size of the nuisance block of a library in NicoStan's nuisance-block
#'   form (default NULL: init_object's n_nuisance_override when the option NicoStan.reduce_sum_nuisance_block is
#'   set, otherwise 0), whose values the memory measurement sets (see [fn_estimate_initial_M_bytes_per_chain()]).
#' @param n_repeats_of_memory_measurement The fresh-process repeats of the memory measurement (default 1, as in
#'   the ground truth's timings, fn_time_burnin_and_sampling_layouts_of_Stan_model_on_each_math_backend()).
#' @param leapfrog_steps_per_timed_iteration The leapfrog steps of each timed iteration, i.e. a fixed trajectory
#'   length (default 10).
#' @param time_Paper_1_check_of_num_chunks TRUE (default): the hardware candidates of Paper 1's burn-in equation
#'   (1, the cores per L3 cache and the hardware threads per L3 cache per chain, each with the equations'
#'   N_chunks) are timed, and Paper 1's check of the threads per chain and N_chunks together
#'   ([fn_search_threads_per_chain_and_num_chunks_by_doubling_and_halving()]) follows from the fastest of them
#'   (with init_object given); FALSE: Paper 1's equations alone (T threads and two pieces per thread), nothing
#'   timed.
#' @param seconds_per_lp_grad_call The seconds of one lp_grad call of this model and data (default NULL: measured
#'   here, with [fn_seconds_per_lp_grad_call_of_compiled_Stan_model()]), e.g. measured once by
#'   [fn_compute_burnin_and_sampling_layouts_of_Stan_model_within_a_budget_of_lp_grad_calls()] for both phases.
#' @param JSON_text_of_Stan_data_list The data's JSON text (fn_JSON_text_of_Stan_data_list()), when already
#'   written (default NULL: written here).
#' @param most_seconds_spent_choosing_the_burnin_layout The budget of this choice, in seconds, from the start of
#'   this function (default NULL: no budget, as before). With it: the leapfrog steps of the timings stay at
#'   leapfrog_steps_per_timed_iteration (never lowered); each timed call lasts the seconds left divided by the
#'   timed calls still planned (n_timed_calls_planned_within_the_budget), less the start-up of a call (measured
#'   from the calls made); a layout is timed only when one more call fits in the seconds left, and the search
#'   ends, with the fastest layout timed, when none does; and when not even two of the shortest calls fit, Paper
#'   1's equations alone choose the layout, nothing timed.
#' @param n_timed_calls_planned_within_the_budget The timed calls the budget is shared out over (default 10: the
#'   starting candidates, the neighbours and the last round of Paper 1's check of a typical model).
#' @param seconds_of_start_up_of_a_timed_call_before_one_is_measured The start-up of a timed call (data, worker
#'   and untimed iteration) assumed until the first call has measured it (default 0.05 s).
#' @param pass_the_data_to_the_workers_as_JSON_text_instead_of_files TRUE: each layout's worker is given the
#'   data's JSON text, without a data file (see fn_time_burnin_iteration_for_each_num_chunks()); FALSE (default):
#'   a data file per layout, as before.
#' @param fraction_slower_within_which_a_neighbour_is_timed_again_in_the_last_round Passed to
#'   [fn_search_threads_per_chain_and_num_chunks_by_doubling_and_halving()] (default NULL: every neighbour timed
#'   again).
#' @param time_the_doubling_after_a_doubling_of_num_chunks_within_the_move_threshold Passed to
#'   [fn_search_threads_per_chain_and_num_chunks_by_doubling_and_halving()] (default TRUE).
#' @param seconds_kept_from_the_budget_for_removing_the_workers_kept_for_reuse Under a budget, with
#'   keep_workers_for_reuse_in_check = TRUE: the seconds of the budget kept for removing the workers at the end
#'   (a full collection of R's memory, about 0.1 s; default 0.1), so that the choice ends within its budget.
#'   decision): Paper 1's check runs only when its cheapest form, two timed calls, costs at most this fraction of
#'   a typical fit without chunking, its seconds from the lp_grad time; otherwise any search would cost more than
#'   it can save, and Paper 1's equations alone choose the layout (default 0.02; NULL: no such rule, as before).
#'   The result's rule records seconds_of_a_typical_fit_without_chunking, seconds_of_the_cheapest_search and
#'   search_skipped_because_it_would_cost_more_than_its_share_of_a_typical_fit.
#' @param iterations_of_a_typical_fit_without_chunking The iterations of a typical fit without chunking, each of
#'   sampling iterations).
#' @param fewest_seconds_of_gradient_per_WCP_thread With seconds_per_lp_grad_call given (default NULL: not used,
#'   as before): Paper 1's equations give 1 thread per chain when one thread's share of a gradient is below it
#'   (see [fn_compute_burnin_n_threads_WCP_and_num_chunks()]), and the hardware candidates timed first are only
#'   those whose threads per chain get at least this share (1 thread per chain always).
#' @return The list of [fn_compute_burnin_n_threads_WCP_and_num_chunks()] for the chosen configuration (its
#'   num_chunks_burnin from Paper 1's check), with num_chunks_check (the list of
#'   [fn_check_num_chunks_by_doubling_and_halving()]; NULL when not checked), M_bytes_measurement and
#'   configuration_timings (NULL when not timed).
#' @export
## fn_compute_burnin_n_threads_WCP_and_num_chunks_for_Stan_model <-  function(
##                                                               model_so_file,
##                                                                  Stan_data_list,
##                                                                  n_chains_burnin  = 4,
##                                                                  init_list        = NULL,
##                                                                  init_object      = NULL,
##                                                                  reuse_saved_choice = TRUE,
##                                                                  M_bytes_per_chain = NULL,
##                                                                  n_rounds = 3,
##                                                                  n_timed_iterations = 1,
##                                                                  minimum_seconds_per_timing = 0.25,
##                                                                  time_Paper_1_check_of_num_chunks =
##                                                                        TRUE,
##                                                                  leapfrog_steps_per_timed_iteration =
##                                                                        10,
##                                                                  keep_workers_for_reuse_in_check =
##                                                                        FALSE,
##                                                                  fraction_faster_needed_to_move =
##                                                                        0.03,
##                                                                  n_untimed_iterations_before_timing =
##                                                                        1,
##                                                                  n_nuisance_owned_by_NicoStan = NULL,
##                                                                  n_repeats_of_memory_measurement = 1,
##                                                                  ...
## ) {
## (the arguments one per line, as the names of the budget's arguments are long)
fn_compute_burnin_n_threads_WCP_and_num_chunks_for_Stan_model <-  function(
        model_so_file,
        Stan_data_list,
        ##  Stan::math library, "nico_math" for a nico::math (AVX) library; no default, every caller passes it)
        kind_of_tape,
        n_chains_burnin                                                            = 4,
        init_list                                                                  = NULL,
        initial_values_of_the_chains                                               = NULL,
        init_object                                                                = NULL,
        reuse_saved_choice                                                         = TRUE,
        M_bytes_per_chain                                                          = NULL,
        n_rounds                                                                   = 3,
        n_timed_iterations                                                         = 1,
        minimum_seconds_per_timing                                                 = 0.25,
        time_Paper_1_check_of_num_chunks                                           = TRUE,
        leapfrog_steps_per_timed_iteration                                         = 10,
        keep_workers_for_reuse_in_check                                            = FALSE,
        fraction_faster_needed_to_move                                             = 0.03,
##        n_untimed_iterations_before_timing                                         = 1,
        n_untimed_iterations_before_timing                                         = 0,
        n_nuisance_owned_by_NicoStan                                               = NULL,
        n_repeats_of_memory_measurement                                            = 1,
        seconds_per_lp_grad_call                                                   = NULL,
        JSON_text_of_Stan_data_list                                                = NULL,
        most_seconds_spent_choosing_the_burnin_layout                              = NULL,
        n_timed_calls_planned_within_the_budget                                    = 10,
        seconds_of_start_up_of_a_timed_call_before_one_is_measured                 = 0.05,
        pass_the_data_to_the_workers_as_JSON_text_instead_of_files                 = FALSE,
        fraction_slower_within_which_a_neighbour_is_timed_again_in_the_last_round  = NULL,
        time_the_doubling_after_a_doubling_of_num_chunks_within_the_move_threshold = TRUE,
        fewest_seconds_of_gradient_per_WCP_thread                                  = NULL,
        seconds_kept_from_the_budget_for_removing_the_workers_kept_for_reuse       = 0.1,
      # largest_fraction_of_a_typical_fit_without_chunking_spent_searching         = 0.02,
      ##  the 2% skip left the search out at N = 500, where the equations' layout was 57-186% slower)
        largest_fraction_of_a_typical_fit_without_chunking_spent_searching         = NULL,
        iterations_of_a_typical_fit_without_chunking                               = 1500,
        ...
) {

        if (!fn_Stan_data_has_N_and_chunk_size(Stan_data_list = Stan_data_list)) {
              stop(paste0("fn_compute_burnin_n_threads_WCP_and_num_chunks_for_Stan_model: the Stan data need ",
                          "integer fields N and chunk_size (NicoStan's N / chunk_size data contract for ",
                          "reduce_sum_static)."))
        }
        ##
        ## ---- the choice saved for the same model file, N, burn-in chains and CPUs, reused as it is:
        ##
        CPU_list <-  fn_format_CPU_list(CPU_ids = fn_read_L3_cache_topology_of_available_CPUs()$CPU_ids_available)
        if (isTRUE(reuse_saved_choice)) {
              saved_choice <-  fn_read_saved_burnin_choice(model_so_file   = model_so_file,
                                                           N_units         = Stan_data_list$N,
                                                           n_chains_burnin = n_chains_burnin,
                                                           CPU_list        = CPU_list)
              if (!is.null(saved_choice)) {
                    return(list( n_threads_WCP_burnin    = saved_choice$n_threads_WCP_burnin,
                                 num_chunks_burnin       = saved_choice$num_chunks_burnin,
                                 chunk_size_burnin       = ceiling(Stan_data_list$N /
                                                                   saved_choice$num_chunks_burnin),
                                 M_bytes_per_chain       = saved_choice$M_bytes_per_chain,
                                 N_units                 = Stan_data_list$N,
                                 n_chains_burnin         = n_chains_burnin,
                                 CPU_list                = CPU_list,
                                 reused_saved_choice     = TRUE,
                                 saved_choice            = saved_choice,
                                 seconds_spent_choosing  = 0))
              }
        }
        time_choice_start <-  proc.time()[["elapsed"]]
        ## (the seconds of each step of this choice, printed at its end: the memory measurement, the data's JSON
        ##  text and the lp_grad timing, each timed call with the gap before it, and the rest)
        seconds_of_the_lp_grad_step <-  0
        data_file_of_the_fit_used <-  "not needed"
        model_of_the_fit_used <-  "not needed"
        repeated_calls_of_the_lp_grad_timing_made <-  "not needed"
        time_at_the_end_of_the_last_step <-  NA_real_
        seconds_of_each_timed_call <-  numeric(0)
        seconds_of_the_timed_iterations_of_each_call <-  numeric(0)
        seconds_of_the_gap_before_each_call <-  numeric(0)
        ##
        ## ---- a library in NicoStan's nuisance-block form: the size of its nuisance block (given, or that of
        ##      init_object when the option NicoStan.reduce_sum_nuisance_block is set), whose values the memory
        ##      measurement sets before its evaluation; 0 for every other library:
        ##
        if (is.null(n_nuisance_owned_by_NicoStan)) {
              ## (the held-block record identifies ownership, independently of the current session option)
              n_nuisance_owned_by_NicoStan <-  0
              if (!is.null(init_object) && length(init_object$nuisance_blocks_held_by_NicoStan) > 0 &&
                  !is.null(init_object$n_nuisance_override)) {
                    n_nuisance_owned_by_NicoStan <-  init_object$n_nuisance_override
              }
        }
        ##
        ## ---- the full data as one piece, so that the measured memory is that of the full-data gradient:
        ##
        ## (M_bytes/chain given, e.g. measured once for this model and N and reused for other CPUs: used as given)
        time_of_memory_measurement_start <-  proc.time()[["elapsed"]]
        if (is.null(M_bytes_per_chain)) {
              Stan_data_list_one_piece <-  Stan_data_list
              Stan_data_list_one_piece$chunk_size <-  as.integer(Stan_data_list$N)
              M_bytes_measurement_as_returned <-  fn_estimate_initial_M_bytes_per_chain(
                                            model_so_file                 = model_so_file,
                                            Stan_data_list                = Stan_data_list_one_piece,
                                            init_list                     = init_list,
                                            n_repeats                     = n_repeats_of_memory_measurement,
                                            n_nuisance_owned_by_NicoStan  = n_nuisance_owned_by_NicoStan)
        } else {
              M_bytes_measurement_as_returned <-  M_bytes_per_chain
        }
        ##  the number of bytes is what every calculation below and Paper 1's equations use; the list as returned
        ##  is kept in the rule)
        M_bytes_measurement <-  if (is.list(M_bytes_measurement_as_returned)) {
              M_bytes_measurement_as_returned$M_bytes_per_chain
        } else M_bytes_measurement_as_returned
        M_bytes_per_chain_of_the_equations <-  M_bytes_measurement
        seconds_of_memory_measurement <-  proc.time()[["elapsed"]] - time_of_memory_measurement_start
##         ##
##         ## ---- the rule (T threads, 2 pieces per thread), and its configurations for 1, 2 and 4 pieces per
##         ##      thread
##         ##      at T and T / 2 threads per chain:
##         ##
##         rule <-  fn_compute_burnin_n_threads_WCP_and_num_chunks( M_bytes_per_chain  = M_bytes_measurement,
##                                                                  N_units            = Stan_data_list$N,
##                                                                  n_chains_burnin    = n_chains_burnin,
##                                                                  pieces_per_thread  = 2,
##                                                                  ...)
##         ## (1 thread per chain too: for a model whose likelihood has a long sequential part outside
##         ##  reduce_sum_static, e.g. a time series' recursion or a survival model's risk-set scan, no
##         ##  within-chain
##         ##  threads can be the fastest)
##         thread_candidates <-  unique(c(rule$n_threads_WCP_burnin, max(1, rule$n_threads_WCP_burnin / 2), 1))
##         rules <-  list()
##         for (n_threads_candidate in thread_candidates) {
##               for (pieces_per_thread in c(1, 2, 4)) {
##                     rules[[length(rules) + 1]] <-  fn_compute_burnin_n_threads_WCP_and_num_chunks(
##                                                          M_bytes_per_chain     = M_bytes_measurement,
##                                                          N_units               = Stan_data_list$N,
##                                                          n_chains_burnin       = n_chains_burnin,
##                                                          pieces_per_thread     = pieces_per_thread,
##                                                          n_threads_WCP_burnin  = n_threads_candidate,
##                                                          ...)
##               }
##         }
##         configurations <-  unique(data.frame(
##               n_threads_WCP_burnin = vapply(X = rules, FUN = function(x) x$n_threads_WCP_burnin,
##                                             FUN.VALUE = 0),
##               num_chunks_burnin = vapply(X = rules, FUN = function(x) x$num_chunks_burnin, FUN.VALUE = 0)))
##         ##
##         ## ---- the fastest of the rule's chunk counts, timed in NicoStan's burn-in worker:
##         ##
##         configuration_timings <-  NULL
##         if (isTRUE(time_Paper_1_check_of_num_chunks) && !is.null(init_object) && nrow(configurations) > 1) {
##               configuration_timings <-  tryCatch(
##                     expr  = do.call(rbind, lapply(X = unique(configurations$n_threads_WCP_burnin),
##                                                   FUN = function(n_threads_candidate) {
##                           timings <-  fn_time_burnin_iteration_for_each_num_chunks(
##                                             init_object           = init_object,
##                                             Stan_data_list        = Stan_data_list,
##                                             n_chains_burnin       = n_chains_burnin,
##                                             n_threads_WCP_burnin  = n_threads_candidate,
##                                             num_chunks_candidates = configurations$num_chunks_burnin[
##                                                   configurations$n_threads_WCP_burnin == n_threads_candidate],
##                                             n_rounds                    = n_rounds,
##                                             n_timed_iterations          = n_timed_iterations,
##                                             minimum_seconds_per_timing  = minimum_seconds_per_timing)
##                           cbind(n_threads_WCP_burnin = n_threads_candidate, timings)
##                     })),
##                     error = function(condition) {
##                           message(paste0("Burn-in configurations not timed (", conditionMessage(condition),
##                                          "); the rule's threads and two pieces per thread are used."))
##                           NULL
##                     })
##               if (!is.null(configuration_timings)) {
##                     fastest_index <-  which.min(configuration_timings$median_seconds_per_burnin_iteration)
##                     fastest <-  configuration_timings[fastest_index, ]
##                     rule <-  rules[[which(vapply(X = rules, FUN = function(x) {
##                           x$n_threads_WCP_burnin == fastest$n_threads_WCP_burnin &&
##                                 x$num_chunks_burnin == fastest$num_chunks
##                     }, FUN.VALUE = FALSE))[1]]]
##               }
##         }
##         ##
##         ## ---- Paper 1's check of the chosen configuration's chunk count, at its threads per chain: twice it
##         ##      and
##         ##      half it, then the next doubling (or halving) for as long as it is faster (powers of two, the
##         ##      partial sums of reduce_sum_static, at least one per thread):
##         ##
##         num_chunks_check <-  NULL
##         if (isTRUE(time_Paper_1_check_of_num_chunks) && !is.null(init_object) &&
##             (!is.null(configuration_timings) || nrow(configurations) == 1)) {
##               n_threads_of_choice <-  rule$n_threads_WCP_burnin
##               num_chunks_check <-  tryCatch(
##                     expr  = fn_check_num_chunks_by_doubling_and_halving(
##                                   num_chunks_start        = rule$num_chunks_burnin,
##                                   fn_median_seconds_per_iteration_of_each_num_chunks =
##                                         function(num_chunks_candidates) {
##                                               fn_time_burnin_iteration_for_each_num_chunks(
##                                                     init_object                 = init_object,
##                                                     Stan_data_list              = Stan_data_list,
##                                                     n_chains_burnin             = n_chains_burnin,
##                                                     n_threads_WCP_burnin        = n_threads_of_choice,
##                                                     num_chunks_candidates       = num_chunks_candidates,
##                                                     n_rounds                    = n_rounds,
##                                                     n_timed_iterations          = n_timed_iterations,
##                                                     minimum_seconds_per_timing  = minimum_seconds_per_timing
##                                               )$median_seconds_per_burnin_iteration
##                                         },
##                                   fewest_num_chunks       = n_threads_of_choice,
##                                   most_num_chunks         = max(n_threads_of_choice,
##                                                                 2^floor(log2(Stan_data_list$N))),
##                                   multiple_of_num_chunks  = n_threads_of_choice,
##                                   powers_of_two           = TRUE),
##                     error = function(condition) {
##                           message(paste0("Burn-in chunk count not checked (", conditionMessage(condition),
##                                          "); the fastest of the rule's configurations is used."))
##                           NULL
##                     })
##               if (!is.null(num_chunks_check)) {
##                     rule$num_chunks_burnin <-  num_chunks_check$num_chunks
##                     rule$chunk_size_burnin <-  ceiling(Stan_data_list$N / num_chunks_check$num_chunks)
##                     rule$tape_bytes_per_piece <-  rule$M_bytes_per_chain / num_chunks_check$num_chunks
##               }
##         }
##         rule$num_chunks_check <-  num_chunks_check
        ##
        ## ---- the rule (T threads per chain, 2 pieces per thread), i.e. Paper 1's equations for the burn-in
        ##      layout:
        ##
        rule <-  fn_compute_burnin_n_threads_WCP_and_num_chunks( M_bytes_per_chain  = M_bytes_measurement,
                                                                 N_units            = Stan_data_list$N,
                                                                 n_chains_burnin    = n_chains_burnin,
                                                                 pieces_per_thread  = 2,
                                                                 seconds_per_lp_grad_call =
                                                                       seconds_per_lp_grad_call,
                                                                 fewest_seconds_of_gradient_per_WCP_thread =
                                                                       fewest_seconds_of_gradient_per_WCP_thread,
                                                                 ...)
        ##
        ## ---- whether the model's tape fits in its share of the cache at full load (Paper 1's sampling
        ##      equations, one chain per CPU, give N_chunks = 1); the burn-in's threads per chain are not a
        ##      question of the cache (with 4 chains, most CPUs are free, and a model whose gradient needs much
        ##      computing on little memory, e.g. an ODE model, gains from WCP), so every model's burn-in layout is
        ##      checked:
        ##
        n_CPUs_of_process <-  length(fn_read_L3_cache_topology_of_available_CPUs()$CPU_ids_available)
        # sampling_rule_at_full_load <-  fn_compute_sampling_num_chunks_for_CPUs_of_this_process(
        #                                      M_bytes_per_chain       = M_bytes_measurement,
        #                                      n_chains_sampling       = n_CPUs_of_process,
        #                                      n_threads_WCP_sampling  = 1,
        #                                      gradient_type           = "autodiff")
        # tape_fits_in_cache <-  (sampling_rule_at_full_load$N_chunks <= 1)
        ##  threads per core; the sampling equation at one chain per CPU says whether the tape fits in the cache)
        topology_of_the_equations <-  fn_read_L3_cache_topology_of_available_CPUs()
        L2_cache_and_threads_per_core <-
              fn_read_threads_per_core_and_L2_cache_size_of_first_CPU_of_this_process()
        N_cores_of_the_equations <-  max(1, floor(topology_of_the_equations$N_hardware_threads_available /
                                                  L2_cache_and_threads_per_core$N_threads_per_core))
        sampling_equation_at_full_load <-  fn_sampling_num_chunks_of_Paper_1_equation(
                                                 M_bytes_per_chain             =
                                                       M_bytes_per_chain_of_the_equations,
                                                 N_units                       = Stan_data_list$N,
                                                 kind_of_tape                  = kind_of_tape,
                                                 L2_cache_size_bytes_per_core  =
                                                       L2_cache_and_threads_per_core$L2_cache_size_bytes,
                                                 N_threads_per_core            =
                                                       L2_cache_and_threads_per_core$N_threads_per_core)
        tape_fits_in_cache <-  (sampling_equation_at_full_load$num_chunks_sampling <= 1)
##      if (tape_fits_in_cache) {
##            rule$n_threads_WCP_burnin <-  1
##            rule$num_chunks_burnin <-  1
##            rule$chunk_size_burnin <-  Stan_data_list$N
##            rule$tape_bytes_per_piece <-  rule$M_bytes_per_chain
##      }
        n_threads_WCP_burnin_from_equations <-  rule$n_threads_WCP_burnin
        num_chunks_burnin_from_equations <-  rule$num_chunks_burnin
##         ##
##         ## ---- Paper 1's check of the whole layout, starting from the equations' layout:
##         ##        - at each threads per chain, Paper 1's check of the chunk count
##         ##          (fn_check_num_chunks_by_doubling_and_halving(), from the rule's chunk count for those
##         ##          threads);
##         ##        - the threads per chain doubled (or halved) for as long as the fastest layout of the next
##         ##          threads per chain is faster.
##         ##      The threads per chain are the powers of two up to the most which the CPUs allow
##      ##      (floor(CPUs / burn-in chains)), and that most; 1 thread per chain is included, for a model whose
##         ##      likelihood has a long sequential part outside reduce_sum_static (e.g. a time series' recursion,
##         ##      or a survival model's risk-set scan):
##         ##
##         most_threads_per_chain <-  rule$n_threads_per_chain_from_hardware
##         threads_per_chain_values <-  sort(unique(c(2^(0:floor(log2(most_threads_per_chain))),
##                                                    most_threads_per_chain,
##                                                    n_threads_WCP_burnin_from_equations)))
##         configurations <-  data.frame( n_threads_WCP_burnin  = n_threads_WCP_burnin_from_equations,
##                                        num_chunks_burnin     = num_chunks_burnin_from_equations)
##         configuration_timings <-  NULL
##         num_chunks_check <-  NULL
##         layouts_checked <-  list()
##         L_main_of_timings <-  NA
##         if (isTRUE(time_Paper_1_check_of_num_chunks) && !is.null(init_object)) {
## ##            ##
## ##            ## (the leapfrog steps per timed iteration, the same for every timing of this model and N: 20, or
## ##            ##  fewer for a slow model, so that each timing lasts about minimum_seconds_per_timing; from the
## ##            ##  equations' layout)
## ##            L_main_of_timings <-
## ##                  fn_leapfrog_steps_per_timed_iteration_for_timings_of_at_least_minimum_seconds(
## ##                        init_object                 = init_object,
## ##                        Stan_data_list              = Stan_data_list,
## ##                        n_chains                    = n_chains_burnin,
## ##                        n_threads_WCP               = n_threads_WCP_burnin_from_equations,
## ##                        num_chunks                  = num_chunks_burnin_from_equations,
## ##                        n_timed_iterations          = n_timed_iterations,
## ##                        minimum_seconds_per_timing  = minimum_seconds_per_timing)
##               ## (Paper 1's L = 20 leapfrog steps for every timing by default; a slow model runs fewer timed
##            ##  iterations instead, down to n_timed_iterations, i.e. the iterations fall with N, as in Paper 1;
##               ##  leapfrog_steps_per_timed_iteration = NULL: fewer steps for a slow model too, so that each
##               ##  timing lasts about minimum_seconds_per_timing)
##               ## L_main_of_timings <-  20
##               L_main_of_timings <-  leapfrog_steps_per_timed_iteration
##               if (is.null(L_main_of_timings)) {
##                     L_main_of_timings <-
##                           fn_leapfrog_steps_per_timed_iteration_for_timings_of_at_least_minimum_seconds(
##                                 init_object                 = init_object,
##                                 Stan_data_list              = Stan_data_list,
##                                 n_chains                    = n_chains_burnin,
##                                 n_threads_WCP               = n_threads_WCP_burnin_from_equations,
##                                 num_chunks                  = num_chunks_burnin_from_equations,
##                                 n_timed_iterations          = n_timed_iterations,
##                                 minimum_seconds_per_timing  = minimum_seconds_per_timing)
##               }
##               ##
##            ## (the fastest layout at one threads per chain: Paper 1's check of the chunk count, from the rule's
##               ##  chunk count for those threads)
##               fn_check_layout_at_threads_per_chain <-  function(n_threads_per_chain) {
##                     rule_at_threads <-  fn_compute_burnin_n_threads_WCP_and_num_chunks(
##                                               M_bytes_per_chain     = M_bytes_measurement,
##                                               N_units               = Stan_data_list$N,
##                                               n_chains_burnin       = n_chains_burnin,
##                                               pieces_per_thread     = 2,
##                                               n_threads_WCP_burnin  = n_threads_per_chain,
##                                               ...)
##                     check <-  fn_check_num_chunks_by_doubling_and_halving(
##                                     num_chunks_start        = rule_at_threads$num_chunks_burnin,
##                                     fn_median_seconds_per_iteration_of_each_num_chunks =
##                                           function(num_chunks_candidates) {
##                                                 fn_time_burnin_iteration_for_each_num_chunks(
##                                                       init_object                 = init_object,
##                                                       Stan_data_list              = Stan_data_list,
##                                                       n_chains_burnin             = n_chains_burnin,
##                                                       n_threads_WCP_burnin        = n_threads_per_chain,
##                                                       num_chunks_candidates       = num_chunks_candidates,
##                                                       n_rounds                    = n_rounds,
##                                                       n_timed_iterations          = n_timed_iterations,
##                                                       minimum_seconds_per_timing  = minimum_seconds_per_timing,
##                                                       L_main                      = L_main_of_timings
##                                                 )$median_seconds_per_burnin_iteration
##                                           },
##                                     fewest_num_chunks       = n_threads_per_chain,
##                                     most_num_chunks         = max(n_threads_per_chain,
##                                                                   2^floor(log2(Stan_data_list$N))),
##                                     multiple_of_num_chunks  = n_threads_per_chain,
##                                     powers_of_two           = TRUE)
##                     return(list( n_threads_WCP_burnin  = n_threads_per_chain,
##                                  rule                  = rule_at_threads,
##                                  check                 = check,
##                                  seconds_of_fastest    = min(check$timings$median_seconds_per_iteration)))
##               }
##               ##
##               ## (the fastest seconds per burn-in iteration of the threads per chain at a position of
##               ##  threads_per_chain_values, among the layouts checked so far)
##               fn_seconds_at_position <-  function(position) {
##                  return(layouts_checked[[as.character(threads_per_chain_values[position])]]$seconds_of_fastest)
##               }
##               ##
##               checked <-  tryCatch(
##                     expr  = {
##                           ##
##                           ## the equations' threads per chain, and the threads per chain below and above it:
##                           position_current <-  match(n_threads_WCP_burnin_from_equations,
##                                                      threads_per_chain_values)
##                           positions_first <-  unique(c(position_current,
##                                                        max(1, position_current - 1),
##                                                        min(length(threads_per_chain_values),
##                                                            position_current + 1)))
##                           for (position in positions_first) {
##                                 layouts_checked[[as.character(threads_per_chain_values[position])]] <-
##                                       fn_check_layout_at_threads_per_chain(threads_per_chain_values[position])
##                           }
##                           ##
##                        ## the direction: towards the faster of the two, when it is faster than the equations':
##                           seconds_first <-  vapply(X          = positions_first,
##                                                    FUN        = fn_seconds_at_position,
##                                                    FUN.VALUE  = 0)
##                           position_fastest_first <-  positions_first[which.min(seconds_first)]
##                           step <-  0
##                           if (fn_seconds_at_position(position_fastest_first) <
##                               fn_seconds_at_position(position_current)) {
##                                 step <-  if (position_fastest_first > position_current) 1 else -1
##                                 position_current <-  position_fastest_first
##                           }
##                           ##
##                           ## the next threads per chain, for as long as its fastest layout is faster:
##                           while (step != 0) {
##                                 position_next <-  position_current + step
##                              if ((position_next < 1) || (position_next > length(threads_per_chain_values))) {
##                                       break
##                                 }
##                                 layouts_checked[[as.character(threads_per_chain_values[position_next])]] <-
##                                       fn_check_layout_at_threads_per_chain(
##                                             threads_per_chain_values[position_next])
##                                 if (fn_seconds_at_position(position_next) <
##                                     fn_seconds_at_position(position_current)) {
##                                       position_current <-  position_next
##                                 } else {
##                                       step <-  0
##                                 }
##                           }
##                           TRUE
##                     },
##                     error = function(condition) {
##                           message(paste0("Burn-in layout not checked (", conditionMessage(condition),
##                                          "); Paper 1's equations are used."))
##                           FALSE
##                     })
##               ##
##               if (isTRUE(checked)) {
##                     fastest <-  layouts_checked[[as.character(threads_per_chain_values[position_current])]]
##                     rule <-  fastest$rule
##                     num_chunks_check <-  fastest$check
##                     rule$num_chunks_burnin <-  fastest$check$num_chunks
##                     rule$chunk_size_burnin <-  ceiling(Stan_data_list$N / fastest$check$num_chunks)
##                     rule$tape_bytes_per_piece <-  rule$M_bytes_per_chain / fastest$check$num_chunks
##                  ## (every layout timed: its threads per chain, chunk count and seconds per burn-in iteration)
##                  configuration_timings <-  do.call(rbind, lapply(X = layouts_checked, FUN = function(layout) {
##                           data.frame( n_threads_WCP_burnin                 = layout$n_threads_WCP_burnin,
##                                       num_chunks                           = layout$check$timings$num_chunks,
##                                       median_seconds_per_burnin_iteration  =
##                                             layout$check$timings$median_seconds_per_iteration)
##                     }))
##                     rownames(configuration_timings) <-  NULL
##               }
##         }
        ##
        ## ---- Paper 1's check of the whole layout (unless the tape fits in the cache, see above): the threads
        ##      per chain and N_chunks doubled and halved together from the equations' layout, while a neighbour
        ##      is faster (fn_search_threads_per_chain_and_num_chunks_by_doubling_and_halving()); the threads per
        ##      chain at most floor(CPUs / burn-in chains), each layout's worker kept for reuse during the check:
        ##
        configuration_timings <-  NULL
        num_chunks_check <-  NULL
        L_main_of_timings <-  NA
        lp_grad <-  NULL
##      if (isTRUE(time_Paper_1_check_of_num_chunks) && !is.null(init_object) && !tape_fits_in_cache) {
        starting_candidates <-  NULL
        ##
        ## ---- the budget of this choice (when given): the seconds left, the start-up of the calls made, and
        ##      whether one more timed call fits:
        ##
        budget_given <-  !is.null(most_seconds_spent_choosing_the_burnin_layout)
        equations_alone_because_of_the_budget <-  FALSE
        ## (the budget rule's record, see below)
        seconds_of_a_typical_fit_without_chunking <-  NA_real_
        seconds_of_the_cheapest_search <-  NA_real_
        search_skipped_because_it_would_cost_more_than_its_share_of_a_typical_fit <-  FALSE
        n_timed_calls_made <-  0
        seconds_of_start_up_of_each_call_made <-  numeric(0)
        seconds_per_iteration_of_each_call_made <-  numeric(0)
        fn_seconds_left_of_the_budget <-  function() {

              seconds_spent_so_far <-  proc.time()[["elapsed"]] - time_choice_start
              ## (less the seconds kept for removing the workers kept for reuse, at the end)
              seconds_kept_for_the_end <-  0
              if (isTRUE(keep_workers_for_reuse_in_check)) {
                    seconds_kept_for_the_end <-
                          seconds_kept_from_the_budget_for_removing_the_workers_kept_for_reuse
              }
              return(most_seconds_spent_choosing_the_burnin_layout - seconds_spent_so_far -
                     seconds_kept_for_the_end)
        }

        fn_seconds_of_start_up_of_a_call <-  function() {

              if (length(seconds_of_start_up_of_each_call_made) == 0) {
                    return(seconds_of_start_up_of_a_timed_call_before_one_is_measured)
              }
              return(mean(seconds_of_start_up_of_each_call_made))
        }

        ## (the shortest call: n_timed_iterations iterations at the fastest seconds per iteration seen, or, before
        ##  any call, at L leapfrog steps of the lp_grad time)
        fn_seconds_of_the_shortest_call <-  function( L_of_the_calls,
                                                      seconds_per_lp_grad_call_of_model) {

              seconds_per_iteration <-  if (length(seconds_per_iteration_of_each_call_made) == 0) {
                    L_of_the_calls * seconds_per_lp_grad_call_of_model
              } else min(seconds_per_iteration_of_each_call_made)
              return(fn_seconds_of_start_up_of_a_call() + n_timed_iterations * seconds_per_iteration)
        }

        if (isTRUE(time_Paper_1_check_of_num_chunks) && !is.null(init_object)) {
              L_main_of_timings <-  leapfrog_steps_per_timed_iteration
              ## (the leapfrog steps of the timed iterations from the model's own lp_grad time, see
              ##  R_fn_seconds_per_lp_grad_call_and_L_and_N_iter_of_each_timed_call.R: lowered only when even
              ##  the fewest timed iterations at leapfrog_steps_per_timed_iteration would last longer than
              ##  minimum_seconds_per_timing)
              if (!is.null(L_main_of_timings)) {
                    time_of_lp_grad_timing_start <-  proc.time()[["elapsed"]]
                    if (!is.null(seconds_per_lp_grad_call) && is.finite(seconds_per_lp_grad_call)) {
                          ## (the lp_grad time given, e.g. measured once for both phases)
                          lp_grad <-  list(seconds_per_lp_grad_call = seconds_per_lp_grad_call)
                    } else {
                          ## (measured at the equations' burn-in N_chunks, the layout the search starts from,
                          ##  never unchunked: at a large N the unchunked gradient (one tape of GBs) can be
                          ##  many times slower than the chunked one the timings run, and L and N_iter would
                          Stan_data_list_of_lp_grad_timing <-  Stan_data_list
                          # Stan_data_list_of_lp_grad_timing$chunk_size <-  as.integer(Stan_data_list$N)
                          Stan_data_list_of_lp_grad_timing$chunk_size <-
                                as.integer(ceiling(Stan_data_list$N / num_chunks_burnin_from_equations))
                          ## JSON_text_of_lp_grad_timing <-  fn_JSON_text_of_Stan_data_list(
                          ##                                 Stan_data_list = Stan_data_list_of_lp_grad_timing)
                          ## (the data's JSON text written once, here, and the lp_grad timing's text made from it
                          ##  by changing its chunk_size only: writing the JSON text twice, through a temporary
                          ##  order)
                          # if (is.null(JSON_text_of_Stan_data_list)) {
                          #       JSON_text_of_Stan_data_list <-  fn_JSON_text_of_Stan_data_list(
                          #                                             Stan_data_list = Stan_data_list)
                          # }
                          ##  written, init_object's data file, read instead of written again, when it is of the
                          ##  Stan data given; (b) the fit's own BridgeStan model, init_object$bs_model, used for
                          ##  the lp_grad timing when it was built from that same text with the same chunk_size,
                          ##  never a second model; (c) the repeated calls of the lp_grad timing made only when
                          ##  the search will run, the budget rule being decided from the timing's own estimate
                          ##  first)
                          data_file_of_init_object <-
                                fn_JSON_text_of_the_data_file_of_init_object_when_it_is_of_the_Stan_data_given(
                                      init_object     = init_object,
                                      Stan_data_list  = Stan_data_list)
                          if (is.null(JSON_text_of_Stan_data_list)) {
                                JSON_text_of_Stan_data_list <-  if (!is.null(data_file_of_init_object))
                                      data_file_of_init_object$JSON_text else
                                      fn_JSON_text_of_Stan_data_list(Stan_data_list = Stan_data_list)
                          }
                          JSON_text_of_lp_grad_timing <-  fn_JSON_text_of_Stan_data_list_with_chunk_size(
                                                                JSON_text_of_Stan_data_list =
                                                                      JSON_text_of_Stan_data_list,
                                                                chunk_size                  =
                                                                      Stan_data_list_of_lp_grad_timing$chunk_size)
                          model_of_the_fit <-  NULL
                          if (!is.null(data_file_of_init_object) &&
                              identical(as.integer(data_file_of_init_object$chunk_size),
                                        as.integer(Stan_data_list_of_lp_grad_timing$chunk_size)) &&
                              inherits(x = init_object$bs_model, what = "StanModel") &&
                              identical(normalizePath(init_object$bs_model$.__enclos_env__$private$lib,
                                                      mustWork = FALSE),
                                        normalizePath(model_so_file, mustWork = FALSE))) {
                                model_of_the_fit <-  init_object$bs_model
                          }
                          ## (whether the search will run at an lp_grad time: the budget rule below, with the
                          ##  timed calls' L from that lp_grad time)
                          fn_the_search_will_run_at_this_lp_grad_time <-  function(
                                                                    seconds_per_lp_grad_call_estimated) {

                                return(TRUE)
                                if (is.null(largest_fraction_of_a_typical_fit_without_chunking_spent_searching)) {
                                      return(TRUE)
                                }
                                L_estimated <-  fn_L_and_N_iter_of_each_timed_call_from_seconds_per_lp_grad_call(
                                      seconds_per_lp_grad_call     = seconds_per_lp_grad_call_estimated,
                                      target_seconds_of_long_call  = minimum_seconds_per_timing,
                                      L_preferred                  = leapfrog_steps_per_timed_iteration,
                                      L_may_be_lowered_when_the_fewest_iterations_last_longer_than_the_target =
                                            !budget_given)$L
                                seconds_of_a_typical_fit <-  iterations_of_a_typical_fit_without_chunking *
                                                             leapfrog_steps_per_timed_iteration *
                                                             seconds_per_lp_grad_call_estimated
                                seconds_of_a_timed_call <-  if (budget_given) {
                                      fn_seconds_of_the_shortest_call(L_of_the_calls = L_estimated,
                                                                      seconds_per_lp_grad_call_of_model =
                                                                            seconds_per_lp_grad_call_estimated)
                                } else {
                                      fn_seconds_of_start_up_of_a_call() +
                                      max(minimum_seconds_per_timing,
                                          n_timed_iterations * L_estimated * seconds_per_lp_grad_call_estimated)
                                }
                                return(2 * seconds_of_a_timed_call <=
                                       largest_fraction_of_a_typical_fit_without_chunking_spent_searching *
                                       seconds_of_a_typical_fit)
                          }

                          lp_grad <-  fn_seconds_per_lp_grad_call_of_compiled_Stan_model(
                                            model_so_file                   = model_so_file,
                                            JSON_text_of_Stan_data          = JSON_text_of_lp_grad_timing,
                                            stop_if_library_already_loaded  = FALSE,
                                            n_nuisance_owned_by_NicoStan    = n_nuisance_owned_by_NicoStan,
                                            BridgeStan_model_already_built_from_the_same_JSON_text =
                                                  model_of_the_fit,
                                            fn_whether_the_repeated_calls_are_needed =
                                                  fn_the_search_will_run_at_this_lp_grad_time)
                          data_file_of_the_fit_used <-  if (is.null(data_file_of_init_object)) "no" else "yes"
                          model_of_the_fit_used <-  if (is.null(model_of_the_fit)) "no" else "yes"
                          repeated_calls_of_the_lp_grad_timing_made <-  if (isTRUE(lp_grad$repeated_calls_made))
                                                                             "yes" else "no"
                    }
                    ##  and N_chunks the search starts from, in place of the rule's)
                    burnin_equations_of_18_Oct <-
                          fn_burnin_threads_per_chain_and_num_chunks_of_Paper_1_equations(
                          M_bytes_per_chain                    = M_bytes_per_chain_of_the_equations,
                          N_units                              = Stan_data_list$N,
                          n_chains_burnin                      = n_chains_burnin,
                          seconds_per_lp_grad_call             = lp_grad$seconds_per_lp_grad_call,
                          kind_of_tape                         = kind_of_tape,
                          threads_per_chain_candidates         =
                                fn_threads_per_chain_candidates_of_Paper_1_burnin_equation(
                                      N_hardware_threads_available         =
                                            topology_of_the_equations$N_hardware_threads_available,
                                      N_hardware_threads_per_L3_available  =
                                            topology_of_the_equations$N_hardware_threads_per_L3_available,
                                      n_chains_burnin                      = n_chains_burnin),
                          N_hardware_threads_per_L3_available  =
                                topology_of_the_equations$N_hardware_threads_per_L3_available,
                          N_cores_available                    = N_cores_of_the_equations,
                          L2_cache_size_bytes_per_core         =
                                L2_cache_and_threads_per_core$L2_cache_size_bytes)
                    n_threads_WCP_burnin_from_equations <-  burnin_equations_of_18_Oct$n_threads_WCP_burnin
                    num_chunks_burnin_from_equations <-  burnin_equations_of_18_Oct$num_chunks_burnin
                    rule$n_threads_WCP_burnin <-  n_threads_WCP_burnin_from_equations
                    rule$num_chunks_burnin <-  num_chunks_burnin_from_equations
                    rule$chunk_size_burnin <-  ceiling(Stan_data_list$N / num_chunks_burnin_from_equations)
                    rule$tape_bytes_per_piece <-  M_bytes_measurement / num_chunks_burnin_from_equations
                    rule$burnin_equations_of_18_Oct <-  burnin_equations_of_18_Oct
                    ## (under a budget, L is never lowered: a model too slow for the budget gets the equations'
                    ##  layout instead, see below)
                    L_and_N_iter <-  fn_L_and_N_iter_of_each_timed_call_from_seconds_per_lp_grad_call(
                                           seconds_per_lp_grad_call     = lp_grad$seconds_per_lp_grad_call,
                                           target_seconds_of_long_call  = minimum_seconds_per_timing,
                                           L_preferred                  = leapfrog_steps_per_timed_iteration,
                                       L_may_be_lowered_when_the_fewest_iterations_last_longer_than_the_target =
                                                 !budget_given)
                    L_main_of_timings <-  L_and_N_iter$L
                    message(paste0("  the leapfrog steps of the burn-in timings from the lp_grad time: L = ",
                                   L_main_of_timings, " (", signif(lp_grad$seconds_per_lp_grad_call, 3),
                                   " s per lp_grad call; ", round(proc.time()[["elapsed"]] -
                                                                 time_of_lp_grad_timing_start, 3), " s)"))
                    seconds_of_the_lp_grad_step <-  proc.time()[["elapsed"]] - time_of_lp_grad_timing_start
                    time_at_the_end_of_the_last_step <-  proc.time()[["elapsed"]]
              }
              if (is.null(L_main_of_timings)) {
                    L_main_of_timings <-
                          fn_leapfrog_steps_per_timed_iteration_for_timings_of_at_least_minimum_seconds(
                                init_object                 = init_object,
                                Stan_data_list              = Stan_data_list,
                                n_chains                    = n_chains_burnin,
                                n_threads_WCP               = n_threads_WCP_burnin_from_equations,
                                num_chunks                  = num_chunks_burnin_from_equations,
                                n_timed_iterations          = n_timed_iterations,
                                minimum_seconds_per_timing  = minimum_seconds_per_timing)
              }
              workers_kept_for_reuse <-  NULL
              if (isTRUE(keep_workers_for_reuse_in_check)) workers_kept_for_reuse <-  new.env()
              ## (the data's JSON text, written once: each layout's data file only changes its chunk_size)
              ## JSON_text_of_Stan_data_list <-  fn_JSON_text_of_Stan_data_list(Stan_data_list = Stan_data_list)
              ## (the data's JSON text, given, or written once here: each layout's data only change their
              ##  chunk_size)
              ## if (is.null(JSON_text_of_Stan_data_list)) {
              ##       JSON_text_of_Stan_data_list <-  fn_JSON_text_of_Stan_data_list(
              ##                                             Stan_data_list = Stan_data_list)
              ## }
              ## (written below, after the budget rule, and only when the search runs: a skipped search never
              ##  needs it)
              ## (the seconds per burn-in iteration of each layout, each timed once; under a budget, each call
              ##  lasts the seconds left divided by the calls still planned, less the start-up of a call, and the
              ##  start-up of each call made, i.e. its wall seconds less its timed seconds, is kept)
              fn_seconds_per_burnin_iteration_of_each_layout <-  function(layouts) {

                    seconds_per_iteration <-  numeric(nrow(layouts))
                    ## (the details of each timing: the call's wall seconds, the timed iterations' seconds and the
                    ##  divergent transitions)
                    details_of_each_timing <-  list()
                    for (index in seq_len(nrow(layouts))) {
                          minimum_seconds_of_this_call <-  minimum_seconds_per_timing
                          if (budget_given) {
                                n_timed_calls_left <-  max(1, n_timed_calls_planned_within_the_budget -
                                                              n_timed_calls_made)
                                minimum_seconds_of_this_call <-  max(0, fn_seconds_left_of_the_budget() /
                                                                        n_timed_calls_left -
                                                                        fn_seconds_of_start_up_of_a_call())
                          }
                          time_of_call_start <-  proc.time()[["elapsed"]]
                          seconds_of_the_gap_before_each_call <<-  c(seconds_of_the_gap_before_each_call,
                                                                     if (is.na(time_at_the_end_of_the_last_step))
                                                                           0 else time_of_call_start -
                                                                     time_at_the_end_of_the_last_step)
                          ## seconds_per_iteration[index] <-  fn_time_burnin_iteration_for_each_num_chunks(
                          timing_of_the_call <-  fn_time_burnin_iteration_for_each_num_chunks(
                                init_object                            = init_object,
                                Stan_data_list                         = Stan_data_list,
                                initial_values_of_the_chains           =
                                      initial_values_of_the_chains,
                                n_chains_burnin                        = n_chains_burnin,
                                n_threads_WCP_burnin                   = layouts$threads_per_chain[index],
                                num_chunks_candidates                  = layouts$num_chunks[index],
                                n_rounds                               = 1,
                                n_timed_iterations                     = n_timed_iterations,
                                minimum_seconds_per_timing             = minimum_seconds_of_this_call,
                                L_main                                 = L_main_of_timings,
                                environment_of_workers_kept_for_reuse  = workers_kept_for_reuse,
                                JSON_text_of_Stan_data_list            = JSON_text_of_Stan_data_list,
                                n_untimed_iterations_before_timing     = n_untimed_iterations_before_timing,
                                pass_the_data_to_the_worker_as_JSON_text_instead_of_a_file =
                                      pass_the_data_to_the_workers_as_JSON_text_instead_of_files
                          ## )$median_seconds_per_burnin_iteration[1]
                          )
                          seconds_per_iteration[index] <-
                                timing_of_the_call$median_seconds_per_burnin_iteration[1]
                          seconds_of_call <-  proc.time()[["elapsed"]] - time_of_call_start
                          time_at_the_end_of_the_last_step <<-  proc.time()[["elapsed"]]
                          seconds_of_each_timed_call <<-  c(seconds_of_each_timed_call, seconds_of_call)
                          seconds_of_the_timed_iterations_of_each_call <<-
                                c(seconds_of_the_timed_iterations_of_each_call,
                                  n_timed_iterations * seconds_per_iteration[index])
                          details_of_each_timing[[index]] <-  data.frame(
                                n_chains                                    = n_chains_burnin,
                                seconds_of_the_call                         = seconds_of_call,
                                seconds_per_burnin_iteration                = seconds_per_iteration[index],
                                n_timed_iterations                          = n_timed_iterations,
                                n_divergent_transitions_of_timed_iterations =
                                      timing_of_the_call$n_divergent_transitions_of_timed_iterations[1],
                                n_transitions_of_timed_iterations           =
                                      timing_of_the_call$n_transitions_of_timed_iterations[1])
                          n_timed_calls_made <<-  n_timed_calls_made + 1
                          seconds_per_iteration_of_each_call_made <<-  c(seconds_per_iteration_of_each_call_made,
                                                                         seconds_per_iteration[index])
                          seconds_of_timed_iterations <-  max(minimum_seconds_of_this_call,
                                                              n_timed_iterations * seconds_per_iteration[index])
                          seconds_of_start_up_of_each_call_made <<-  c(seconds_of_start_up_of_each_call_made,
                                                                       max(0, seconds_of_call -
                                                                              seconds_of_timed_iterations))
                    }
                    attr(seconds_per_iteration, "details_of_each_timing") <-  do.call(rbind,
                                                                         details_of_each_timing)
                    return(seconds_per_iteration)
              }

              ## (under a budget: whether one more call fits in the seconds left)
              fn_another_timing_fits_in_the_budget <-  NULL
              if (budget_given) {
                    fn_another_timing_fits_in_the_budget <-  function() {

                          return(fn_seconds_left_of_the_budget() >=
                                 fn_seconds_of_the_shortest_call(L_of_the_calls = L_main_of_timings,
                                                                 seconds_per_lp_grad_call_of_model =
                                                                       lp_grad$seconds_per_lp_grad_call))
                    }

              }
              ##
              ## (the start of the search: the fastest of the threads per chain given by the hardware - 1, the
              ##  cores per L3 cache and the hardware threads per L3 cache, at most floor(CPUs / burn-in chains) -
              ##  each with the equations' N_chunks for its threads per chain)
              threads_per_core <-  fn_read_hardware_threads_per_core()
              threads_per_chain_of_hardware <-  c(1, 2,
                                                  rule$n_threads_per_chain_from_L3_cache / threads_per_core,
                                                  rule$n_threads_per_chain_from_L3_cache)
              threads_per_chain_of_hardware <-  2^floor(log2(pmin(rule$n_threads_per_chain_from_hardware,
                                                                   pmax(1, threads_per_chain_of_hardware))))
              starting_candidates <-  data.frame(threads_per_chain = sort(unique(threads_per_chain_of_hardware)))
              ## (only the candidates whose threads each get at least fewest_seconds_of_gradient_per_WCP_thread of
              ##  a gradient, when it is given; 1 thread per chain always)
              if (!is.null(fewest_seconds_of_gradient_per_WCP_thread) && !is.null(lp_grad)) {
                    share_of_a_gradient_per_thread <-  lp_grad$seconds_per_lp_grad_call /
                                                       starting_candidates$threads_per_chain
                    starting_candidates <-  starting_candidates[(starting_candidates$threads_per_chain == 1) |
                                                                (share_of_a_gradient_per_thread >=
                                                                 fewest_seconds_of_gradient_per_WCP_thread), ,
                                                                drop = FALSE]
              }
              starting_candidates$num_chunks <-  vapply(
                    X          = starting_candidates$threads_per_chain,
                    FUN        = function(threads_per_chain) {

                          # rule_at_threads <-  fn_compute_burnin_n_threads_WCP_and_num_chunks(
                          #                           M_bytes_per_chain     = M_bytes_measurement,
                          #                           N_units               = Stan_data_list$N,
                          #                           n_chains_burnin       = n_chains_burnin,
                          #                           pieces_per_thread     = 2,
                          #                           n_threads_WCP_burnin  = threads_per_chain,
                          #                           ...)
                          rule_at_threads <-  fn_burnin_num_chunks_of_Paper_1_equation_at_threads_per_chain(
                                                    M_bytes_per_chain             = M_bytes_per_chain_of_the_equations,
                                                    N_units                       = Stan_data_list$N,
                                                    n_chains_burnin               = n_chains_burnin,
                                                    n_threads_WCP_burnin          = threads_per_chain,
                                                    kind_of_tape                  = kind_of_tape,
                                                    N_cores_available             = N_cores_of_the_equations,
                                                    L2_cache_size_bytes_per_core  =
                                                          L2_cache_and_threads_per_core$L2_cache_size_bytes)
                          return(rule_at_threads$num_chunks_burnin)
                    },

                    FUN.VALUE  = 0)
              ## (the layout without chunking, 1 thread per chain and 1 chunk, is always the first one timed)
              starting_candidates <-  starting_candidates[!((starting_candidates$threads_per_chain == 1) &
                                                            (starting_candidates$num_chunks == 1)), , drop = FALSE]
              starting_candidates <-  rbind(data.frame(threads_per_chain = 1, num_chunks = 1),
                                            starting_candidates)
            #   ## (under a budget: when not even two of the shortest calls fit in the seconds left, Paper 1's
            #   ##  equations alone choose the layout, nothing timed)
            #   if (budget_given &&
            #       (fn_seconds_left_of_the_budget() <
            #        2 * fn_seconds_of_the_shortest_call(L_of_the_calls = L_main_of_timings,
            #                                            seconds_per_lp_grad_call_of_model =
            #                                                  lp_grad$seconds_per_lp_grad_call))) {
            #         equations_alone_because_of_the_budget <-  TRUE
            #         message(paste0("Burn-in layout from Paper 1's equations alone (",
            #                        n_threads_WCP_burnin_from_equations, " threads per chain, ",
            #                        num_chunks_burnin_from_equations, " chunks), nothing timed: ",
            #                        round(max(0, fn_seconds_left_of_the_budget()), 3), " s of the budget of ",
            #                        round(most_seconds_spent_choosing_the_burnin_layout, 3), " s left, less ",
            #                        "than two timed calls of ", L_main_of_timings, " leapfrog steps"))
            #   }
              ##  the equations' layout alone, untimed, was 57-186% slower than the ground truth at N = 500)

              ##  timed calls, costs at most largest_fraction_of_a_typical_fit_without_chunking_spent_searching of
              ##  a typical fit without chunking, its seconds from the lp_grad time (iterations_of_a_typical_fit_
              ##  without_chunking iterations of leapfrog_steps_per_timed_iteration leapfrog steps); otherwise
              ##  Paper 1's equations alone choose the layout, nothing timed. The lp_grad time is that of the
              ##  equations' N_chunks, never measured unchunked, so that a typical fit is if anything
              ##  underestimated and the search skipped sooner. The move threshold of the check and the tolerance
              ##  of the checks are unchanged)
              # if (!is.null(largest_fraction_of_a_typical_fit_without_chunking_spent_searching) &&
              #     !equations_alone_because_of_the_budget &&
              #     isTRUE(is.finite(lp_grad$seconds_per_lp_grad_call))) {
              #       seconds_of_a_typical_fit_without_chunking <-  iterations_of_a_typical_fit_without_chunking *
              #                                                     leapfrog_steps_per_timed_iteration *
              #                                                     lp_grad$seconds_per_lp_grad_call
              #       seconds_of_a_timed_call <-  if (budget_given) {
              #             fn_seconds_of_the_shortest_call(L_of_the_calls = L_main_of_timings,
              #                                             seconds_per_lp_grad_call_of_model =
              #                                                   lp_grad$seconds_per_lp_grad_call)
              #       } else {
              #             fn_seconds_of_start_up_of_a_call() +
              #             max(minimum_seconds_per_timing,
              #                 n_timed_iterations * L_main_of_timings * lp_grad$seconds_per_lp_grad_call)
              #       }
              #       seconds_of_the_cheapest_search <-  2 * seconds_of_a_timed_call
              #       if (seconds_of_the_cheapest_search >
              #           largest_fraction_of_a_typical_fit_without_chunking_spent_searching *
              #           seconds_of_a_typical_fit_without_chunking) {
              #             search_skipped_because_it_would_cost_more_than_its_share_of_a_typical_fit <-  TRUE
              #             equations_alone_because_of_the_budget <-  TRUE
              #             message(paste0("Burn-in layout: search skipped: a typical fit of ",
              #                            signif(seconds_of_a_typical_fit_without_chunking, 3), " s; the ",
              #                            "equations' layout ", n_chains_burnin, " x ",
              #                            n_threads_WCP_burnin_from_equations, " / ",
              #                            num_chunks_burnin_from_equations, "; any search would cost more than ",
              #                            "it can save (at least ", signif(seconds_of_the_cheapest_search, 3),
              #                            " s, more than ",
              #                            100 * largest_fraction_of_a_typical_fit_without_chunking_spent_searching,
              #                            "% of the typical fit)"))
              #       }
              # }
              if (!equations_alone_because_of_the_budget && is.null(JSON_text_of_Stan_data_list)) {
                    # JSON_text_of_Stan_data_list <-  fn_JSON_text_of_Stan_data_list(
                    #                                       Stan_data_list = Stan_data_list)
                    ## (the fit's data file read when it is of the Stan data given, otherwise written here)
                    data_file_of_init_object <-
                          fn_JSON_text_of_the_data_file_of_init_object_when_it_is_of_the_Stan_data_given(
                                init_object     = init_object,
                                Stan_data_list  = Stan_data_list)
                    JSON_text_of_Stan_data_list <-  if (!is.null(data_file_of_init_object))
                          data_file_of_init_object$JSON_text else
                          fn_JSON_text_of_Stan_data_list(Stan_data_list = Stan_data_list)
              }
              layout_search <-  if (equations_alone_because_of_the_budget) NULL else tryCatch(
                    expr  = {
                          ## starting_candidates$seconds <-  fn_seconds_per_burnin_iteration_of_each_layout(
                          ##                                      starting_candidates)
                          ## (under a budget, each candidate only when one more call fits: the candidates not
                          ##  timed are left out)
                          starting_candidates$seconds <-  NA_real_
                          for (index in seq_len(nrow(starting_candidates))) {
                                # if (!is.null(fn_another_timing_fits_in_the_budget) &&
                                #     !isTRUE(fn_another_timing_fits_in_the_budget())) {
                                ## (the first two layouts, without chunking and the equations' own, are
                                ##  always timed, whatever the time limit)
                                if ((index > 2) && !is.null(fn_another_timing_fits_in_the_budget) &&
                                    !isTRUE(fn_another_timing_fits_in_the_budget())) {
                                      break
                                }
                                starting_candidates$seconds[index] <-
                                      fn_seconds_per_burnin_iteration_of_each_layout(starting_candidates[index, ])
                          }
                          starting_candidates <-  starting_candidates[!is.na(starting_candidates$seconds), ,
                                                                      drop = FALSE]
                          if (nrow(starting_candidates) == 0) stop("no layout could be timed within the budget")
                          fastest_candidate <-  starting_candidates[which.min(starting_candidates$seconds), ]
                          fn_search_threads_per_chain_and_num_chunks_by_doubling_and_halving(
                                threads_per_chain_start         = fastest_candidate$threads_per_chain,
                                num_chunks_start                = fastest_candidate$num_chunks,
                                fn_seconds_of_each_layout       = fn_seconds_per_burnin_iteration_of_each_layout,
                                most_threads_per_chain          = rule$n_threads_per_chain_from_hardware,
                                most_num_chunks                 = Inf,
                                fraction_faster_needed_to_move  = fraction_faster_needed_to_move,
                                timings_already_made            = starting_candidates,
                                fn_another_timing_fits_in_the_budget  = fn_another_timing_fits_in_the_budget,
                                fraction_slower_within_which_a_neighbour_is_timed_again_in_the_last_round =
                                      fraction_slower_within_which_a_neighbour_is_timed_again_in_the_last_round,
                                time_the_doubling_after_a_doubling_of_num_chunks_within_the_move_threshold =
                                      time_the_doubling_after_a_doubling_of_num_chunks_within_the_move_threshold,
                                also_time_2_threads_per_chain_at_2_chunks_when_the_equations_say_no_chunking =
                                      TRUE)
                    },
                    error = function(condition) {

                          message(paste0("Burn-in layout not checked (", conditionMessage(condition),
                                         "); Paper 1's equations are used."))
                          NULL
                    })

              if (!is.null(workers_kept_for_reuse)) {
                    fn_remove_workers_kept_for_reuse(
                          environment_of_workers_kept_for_reuse = workers_kept_for_reuse)
              }
              if (!is.null(layout_search)) {
                    rule$n_threads_WCP_burnin <-  layout_search$threads_per_chain
                    rule$num_chunks_burnin <-  layout_search$num_chunks
                    rule$chunk_size_burnin <-  ceiling(Stan_data_list$N / layout_search$num_chunks)
                    rule$tape_bytes_per_piece <-  rule$M_bytes_per_chain / layout_search$num_chunks
                    num_chunks_check <-  layout_search
                    ## (every timing of the search, in order, with why each layout was timed and its details)
                    rule$every_timing_of_the_burnin_search <-  layout_search$every_timing_made
                    ## (every layout timed: its threads per chain, chunk count, the mean of its seconds per
                    ##  burn-in iteration over its timings, and its number of timings)
                    configuration_timings <-  data.frame(
                          n_threads_WCP_burnin                          = layout_search$timings$threads_per_chain,
                          num_chunks                                    = layout_search$timings$num_chunks,
                          mean_seconds_per_burnin_iteration_over_timings = layout_search$timings$mean_seconds,
                          n_timings                                     = layout_search$timings$n_timings)
              }
        }
        rule$n_threads_WCP_burnin_from_equations <-  n_threads_WCP_burnin_from_equations
        ## (the hardware candidates timed - 1, the cores per L3 cache and the hardware threads per L3 cache per
        ##  chain, each with the equations' N_chunks - and the fastest of them, i.e. the burn-in equation's
        ##  layout, from which the check starts)
        rule$burnin_hardware_candidates_timed <-  starting_candidates
        rule$n_threads_WCP_burnin_of_fastest_hardware_candidate <-  NA
        rule$num_chunks_burnin_of_fastest_hardware_candidate <-  NA
        if (!is.null(starting_candidates) && !is.null(starting_candidates$seconds)) {
              fastest_hardware_candidate <-  starting_candidates[which.min(starting_candidates$seconds), ]
              rule$n_threads_WCP_burnin_of_fastest_hardware_candidate <-
                    fastest_hardware_candidate$threads_per_chain
              rule$num_chunks_burnin_of_fastest_hardware_candidate <-  fastest_hardware_candidate$num_chunks
        }
        rule$L_main_of_timings <-  L_main_of_timings
        rule$seconds_per_lp_grad_call <-  if (is.null(lp_grad)) NA else lp_grad$seconds_per_lp_grad_call
        rule$num_chunks_burnin_from_equations <-  num_chunks_burnin_from_equations
        rule$num_chunks_check <-  num_chunks_check
        rule$M_bytes_measurement <-  M_bytes_measurement_as_returned
        rule$configuration_timings <-  configuration_timings
        rule$CPU_list <-  CPU_list
        rule$reused_saved_choice <-  FALSE
        rule$seconds_spent_choosing <-  proc.time()[["elapsed"]] - time_choice_start
        ## (where the seconds of the choice went: the memory measurement, then the leapfrog steps from the lp_grad
        ##  time and the layouts timed)
        rule$seconds_of_memory_measurement <-  seconds_of_memory_measurement
        ## (the budget, when given, whether it chose the equations' layout alone, and whether it ended the search)
        rule$most_seconds_spent_choosing_the_burnin_layout <-  NA
        if (budget_given) {
              rule$most_seconds_spent_choosing_the_burnin_layout <-  most_seconds_spent_choosing_the_burnin_layout
        }
        rule$burnin_layout_from_the_equations_alone_because_of_the_budget <-
              equations_alone_because_of_the_budget
        rule$seconds_of_a_typical_fit_without_chunking <-  seconds_of_a_typical_fit_without_chunking
        rule$seconds_of_the_cheapest_search <-  seconds_of_the_cheapest_search
        rule$search_skipped_because_it_would_cost_more_than_its_share_of_a_typical_fit <-
              search_skipped_because_it_would_cost_more_than_its_share_of_a_typical_fit
        rule$burnin_search_ended_by_the_budget <-  isTRUE(num_chunks_check$search_ended_by_the_budget)
        rule$n_timed_calls_of_the_burnin_choice <-  n_timed_calls_made
        n_layouts_timed <-  if (is.null(configuration_timings)) 0 else nrow(configuration_timings)
        message(paste0("Burn-in layout chosen in ", round(rule$seconds_spent_choosing, 2), " s",
                       if (budget_given) paste0(" (budget ",
                                                round(most_seconds_spent_choosing_the_burnin_layout, 2),
                                                " s)") else "",
                       ": the memory ",
                       "measurement ", round(seconds_of_memory_measurement, 2), " s, then ", n_layouts_timed,
                       " layouts timed (L = ", L_main_of_timings, ") in ",
                       round(rule$seconds_spent_choosing - seconds_of_memory_measurement, 2), " s",
                       if (rule$burnin_search_ended_by_the_budget) "; the search ended by the budget" else ""))
        seconds_of_the_rest <-  rule$seconds_spent_choosing - seconds_of_memory_measurement -
                                seconds_of_the_lp_grad_step - sum(seconds_of_each_timed_call) -
                                sum(seconds_of_the_gap_before_each_call)
        message(paste0("  the seconds of each step: the memory measurement ",
                       round(seconds_of_memory_measurement, 3), ", the data's JSON text and the lp_grad timing ",
                       round(seconds_of_the_lp_grad_step, 3), " (data file of the fit used: ",
                       data_file_of_the_fit_used, "; its model used: ", model_of_the_fit_used,
                       "; repeated calls made: ", repeated_calls_of_the_lp_grad_timing_made, "), ",
                       length(seconds_of_each_timed_call), " timed calls ",
                       round(sum(seconds_of_each_timed_call), 3),
                       " (their timed iterations ", round(sum(seconds_of_the_timed_iterations_of_each_call), 3),
                       "), the gaps before them ", round(sum(seconds_of_the_gap_before_each_call), 3),
                       ", the rest ", round(seconds_of_the_rest, 3), " s"))
        if (length(seconds_of_each_timed_call) > 0) {
              message(paste0("  each timed call, its timed iterations and the gap before it (s): ",
                             paste0(round(seconds_of_each_timed_call, 3), " (",
                                    round(seconds_of_the_timed_iterations_of_each_call, 3), "; gap ",
                                    round(seconds_of_the_gap_before_each_call, 3), ")", collapse = ", ")))
        }
        ## (saved for reuse when the configurations were timed, or when there was only one to choose)
        ## if (!is.null(configuration_timings) || nrow(configurations) == 1) {
        ## (saved only when the layout was timed)
        if (!is.null(configuration_timings)) {
              fn_save_burnin_choice(model_so_file           = model_so_file,
                                    N_units                 = Stan_data_list$N,
                                    n_chains_burnin         = n_chains_burnin,
                                    CPU_list                = CPU_list,
                                    rule                    = rule,
                                    seconds_spent_choosing  = rule$seconds_spent_choosing)
        }
        return(rule)

}




#' Whether Stan data follow NicoStan's N / chunk_size data contract
#'
#' @param Stan_data_list The Stan data.
#' @return TRUE when the data have positive whole-number fields N and chunk_size.
#' @keywords internal
fn_Stan_data_has_N_and_chunk_size <-  function( Stan_data_list ) {

        for (data_name in c("N", "chunk_size")) {
              data_value <-  Stan_data_list[[data_name]]
              if (!is.numeric(data_value) || length(x = data_value) != 1 || !is.finite(x = data_value) ||
                  data_value < 1 || data_value != floor(x = data_value)) {
                    return(FALSE)
              }
        }
        return(TRUE)

}




#### ---- M_bytes/chain of a built-in model, measured once for each model and data size, then reused -----------

#' The file in NicoStan's cache folder that holds the M_bytes/chain measured for each built-in model and data size
#' @return The path of the file.
#' @keywords internal
fn_saved_M_bytes_of_built_in_models_file <-  function() {

        return(file.path(tools::R_user_dir(package = "NicoStan", which = "cache"),
                         "M_bytes_per_chain_measured_per_built_in_model_and_data_size.rds"))

}

#' M_bytes/chain of a built-in model: the measurement saved for the same model and data size, or a new one
#'
#' Returns the M_bytes/chain of [fn_estimate_initial_M_bytes_per_chain_of_built_in_model()] for the built-in
#' model of init_object. A measurement is saved in NicoStan's cache folder and reused, without measuring again,
#' for the same Model_type, N, number of tests, number of main parameters and vect_type (the memory of one
#' gradient does not depend on the chains or on the CPUs).
#'
#' @param Model_type The built-in model: "MVP", "MVOP", "LC_MVP" or "LC_MVOP".
#' @param init_object The model's initialisation object.
#' @param y The data (default: the data of init_object).
#' @param reuse_saved_measurement TRUE (default): reuse the saved measurement when there is one.
#' @return A list: M_bytes_per_chain, reused_saved_measurement, seconds_spent_measuring (0 when reused) and
#'   measurement (the list of fn_estimate_initial_M_bytes_per_chain_of_built_in_model(); NULL when reused).
#' @export
fn_M_bytes_per_chain_of_built_in_model <-  function( Model_type,
                                                     init_object,
                                                     y                        = NULL,
                                                     reuse_saved_measurement  = TRUE
) {

        if (is.null(x = y)) y <-  init_object$model_args_list$y
        ##
        ## ---- the model and data size that identify a measurement:
        ##
        vect_type <-  as.character(init_object$Model_args_as_Rcpp_List$Model_args_strings[1])
        model_and_data_size <-  data.frame( Model_type        = Model_type,
                                            N                 = nrow(x = y),
                                            n_tests           = ncol(x = y),
                                            n_params_main     = init_object$n_params_main,
                                            vect_type         = vect_type,
                                            stringsAsFactors  = FALSE)
        ##
        saved_measurements_file <-  fn_saved_M_bytes_of_built_in_models_file()
        saved_measurements <-  if (file.exists(saved_measurements_file)) {
              tryCatch(expr = readRDS(file = saved_measurements_file), error = function(condition) NULL)
        } else NULL
        ##
        fn_rows_of_same_model_and_data_size <-  function(measurements) {

              measurements$Model_type == model_and_data_size$Model_type &
              measurements$N == model_and_data_size$N &
              measurements$n_tests == model_and_data_size$n_tests &
              measurements$n_params_main == model_and_data_size$n_params_main &
              measurements$vect_type == model_and_data_size$vect_type
        }

        ##
        ## ---- the saved measurement, reused as it is:
        ##
        if (isTRUE(reuse_saved_measurement) && is.data.frame(saved_measurements) &&
            (nrow(saved_measurements) > 0)) {
              same_model_and_data_size <-  fn_rows_of_same_model_and_data_size(measurements = saved_measurements)
              if (any(same_model_and_data_size)) {
                    saved_row <-  saved_measurements[which(same_model_and_data_size)[1], , drop = FALSE]
                    return(list( M_bytes_per_chain         = saved_row$M_bytes_per_chain,
                                 reused_saved_measurement  = TRUE,
                                 seconds_spent_measuring   = 0,
                                 measurement               = NULL))
              }
        }
        ##
        ## ---- a new measurement, saved for reuse (replacing a saved one for the same model and data size):
        ##
        time_measurement_start <-  proc.time()[["elapsed"]]
        measurement <-  fn_estimate_initial_M_bytes_per_chain_of_built_in_model( Model_type   = Model_type,
                                                                                 init_object  = init_object,
                                                                                 y            = y)
        seconds_spent_measuring <-  proc.time()[["elapsed"]] - time_measurement_start
        ##
        new_measurement <-  model_and_data_size
        new_measurement$M_bytes_per_chain <-  measurement$M_bytes_per_chain
        new_measurement$seconds_spent_measuring <-  seconds_spent_measuring
        ##
        if (is.data.frame(saved_measurements) && (nrow(saved_measurements) > 0)) {
              same_model_and_data_size <-  fn_rows_of_same_model_and_data_size(measurements = saved_measurements)
              saved_measurements <-  rbind(saved_measurements[!same_model_and_data_size, , drop = FALSE],
                                           new_measurement)
        } else {
              saved_measurements <-  new_measurement
        }
        dir.create(path = dirname(saved_measurements_file), recursive = TRUE, showWarnings = FALSE)
        ## (written through a temporary file renamed over it, as several fits can save at once)
        temporary_file <-  tempfile(pattern = "saved_M_bytes_of_built_in_models_",
                                    tmpdir = dirname(saved_measurements_file), fileext = ".rds.tmp")
        saveRDS(object = saved_measurements, file = temporary_file)
        file.rename(from = temporary_file, to = saved_measurements_file)
        ##
        return(list( M_bytes_per_chain         = measurement$M_bytes_per_chain,
                     reused_saved_measurement  = FALSE,
                     seconds_spent_measuring   = seconds_spent_measuring,
                     measurement               = measurement))

}




#### ---- the rule for a built-in model other than the binary LC-MVP, with its measurement --------------------
## (M_bytes/chain: counted from the workspace of the gradient, as Paper 1's M_bytes/chain for a manual gradient)

#' Burn-in within-chain threads and chunks of a built-in model other than the binary LC-MVP
#'
#' For every built-in model except the binary LC-MVP (whose threads and chunks are given in $sample(), from Paper
#' 1's E1 results), i.e. Model_type "MVP", "MVOP" or "LC_MVOP": M_bytes/chain is Paper 1's M_bytes/chain for a
#' manual gradient, the workspace of the gradient, counted
#' ([fn_M_bytes_per_chain_counted_from_workspace_of_built_in_model()]), and
#' [fn_compute_burnin_n_threads_WCP_and_num_chunks()] (Paper 1's burn-in rule: T threads and two pieces per
#' thread) gives the burn-in threads and chunks for the CPUs of this process. The rule's configurations are not
#' timed for built-in models.
#'
#' @param Model_type The built-in model: "MVP", "MVOP" or "LC_MVOP".
#' @param init_object The model's initialisation object.
#' @param n_chains_burnin Number of burn-in chains run in parallel (default 4).
#' @param ... Further arguments of [fn_compute_burnin_n_threads_WCP_and_num_chunks()].
#' @return The list of [fn_compute_burnin_n_threads_WCP_and_num_chunks()], with M_bytes_per_chain,
#'   reused_saved_measurement (FALSE), seconds_spent_choosing (0: M_bytes/chain is counted, not measured) and
#'   CPU_list.
#' @export
fn_compute_burnin_n_threads_WCP_and_num_chunks_for_built_in_model <-  function( Model_type,
                                                                                init_object,
                                                                                n_chains_burnin  = 4,
                                                                                ...
) {

        built_in_model_types_with_rule <-  c("MVP", "MVOP", "LC_MVOP")
        if (!is.character(Model_type) || (length(x = Model_type) != 1) ||
            !(Model_type %in% built_in_model_types_with_rule)) {
              stop(paste0("fn_compute_burnin_n_threads_WCP_and_num_chunks_for_built_in_model: Model_type must ",
                          "be one of ", paste(built_in_model_types_with_rule, collapse = ", "), " (the binary ",
                          "LC-MVP's threads and chunks are given in $sample(), from Paper 1's E1 results); ",
                          "got: ",
                          paste(as.character(Model_type), collapse = ", ")))
        }
        ##
        ## M_bytes <-  fn_M_bytes_per_chain_of_built_in_model( Model_type   = Model_type,
        ##                                                     init_object  = init_object)
        ## (Paper 1's M_bytes/chain for a manual gradient: the workspace of the gradient, counted, not measured)
        M_bytes <-  fn_M_bytes_per_chain_counted_from_workspace_of_built_in_model(
                          Model_type  = Model_type,
                          N           = nrow(x = init_object$model_args_list$y),
                          n_tests     = ncol(x = init_object$model_args_list$y))
        ##
        N_units <-  nrow(x = init_object$model_args_list$y)
        rule <-  fn_compute_burnin_n_threads_WCP_and_num_chunks( M_bytes_per_chain  =
                                                                       M_bytes$M_bytes_per_chain,
                                                                 N_units            = N_units,
                                                                 n_chains_burnin    = n_chains_burnin,
                                                                 pieces_per_thread  = 2,
                                                                 ...)
        ## rule$reused_saved_measurement <-  M_bytes$reused_saved_measurement
        ## rule$seconds_spent_choosing <-  M_bytes$seconds_spent_measuring
        ## (counted, nothing measured or reused)
        rule$reused_saved_measurement <-  FALSE
        rule$seconds_spent_choosing <-  0
        rule$CPU_list <-  fn_format_CPU_list(CPU_ids = rule$topology$CPU_ids_available)
        return(rule)

}




#### ---- the threads per core and the L2 cache size of the first CPU of this process ----------------------------

#' The hardware threads per core and the L2 cache size (unified or data) of the first CPU of this process
#'
#' Read from that CPU's own sysfs files (topology/thread_siblings_list and cache/index*/level, type and size).
#'
#' @param sysfs_CPU_directory The sysfs CPU directory (default "/sys/devices/system/cpu").
#' @return A list with N_threads_per_core and L2_cache_size_bytes, or NULL when either cannot be read.
#' @keywords internal
fn_read_threads_per_core_and_L2_cache_size_of_first_CPU_of_this_process <-  function(
        sysfs_CPU_directory  = "/sys/devices/system/cpu"
) {

        CPU_id <-  fn_read_L3_cache_topology_of_available_CPUs()$CPU_ids_available[1]
        CPU_directory <-  file.path(sysfs_CPU_directory, paste0("cpu", CPU_id))
        siblings_file <-  file.path(CPU_directory, "topology", "thread_siblings_list")
        if (!file.exists(siblings_file)) return(NULL)
        siblings_list <-  readLines(con = siblings_file, n = 1, warn = FALSE)
        N_threads_per_core <-  length(fn_parse_sysfs_CPU_list(CPU_list_string = siblings_list))
        L2_cache_size_bytes <-  NA_real_
        cache_index_directories <-  list.files(path = file.path(CPU_directory, "cache"),
                                               pattern = "^index[0-9]+$", full.names = TRUE)
        for (cache_index_directory in cache_index_directories) {
              fn_read_first_line_of_cache_file <-  function(file_name) {

                    return(readLines(con = file.path(cache_index_directory, file_name), n = 1, warn = FALSE))
              }

              cache_level <-  suppressWarnings(as.numeric(fn_read_first_line_of_cache_file("level")))
              cache_type <-  fn_read_first_line_of_cache_file("type")
              if (isTRUE(cache_level == 2) && (cache_type %in% c("Unified", "Data"))) {
                    L2_cache_size_bytes <-  fn_parse_sysfs_cache_size_bytes(
                                                  cache_size_string = fn_read_first_line_of_cache_file("size"))
              }
        }
        if (!is.finite(L2_cache_size_bytes) || (N_threads_per_core < 1)) return(NULL)
        return(list( N_threads_per_core   = N_threads_per_core,
                     L2_cache_size_bytes  = L2_cache_size_bytes))

}





#'
#' \preformatted{
#'   N_chunks  = 2^ceil( log2( max( 1, M_bytes/chain / (f x L2 per hardware thread) ) ) )
#' }
#'
#' with the L2 cache per hardware thread = L2_cache_size_bytes_per_core / N_threads_per_core, and f = 4.0 for the
#' autodiff tape of a Stan::math library, 2.3 for the tape of a nico::math (AVX) library; one thread per chain.
#' Fitted to the sampling ground truth of Weibull, robust-t, IRT, the hierarchical logistic model and the
#' four-class LC-MVOP on both machines (score_the_sampling_equation_with_one_target_for_each_kind_of_tape.py, in
#' the benchmark folder of Paper 3: 86 of 86 cases within 10% of the fastest layout, the worst 8.6%).
#'
#' @param M_bytes_per_chain Memory, in bytes, of one full-data log-density-and-gradient evaluation of one chain.
#' @param N_units The number of units reduce_sum_static sums over (the N of the Stan data).
#' @param kind_of_tape "Stan_math" (a Stan::math library) or "nico_math" (a nico::math, AVX, library).
#' @param L2_cache_size_bytes_per_core The L2 cache of one core, in bytes.
#' @param N_threads_per_core The hardware threads of one core.
#' @return A list with num_chunks_sampling and every intermediate quantity.
#' @export
fn_sampling_num_chunks_of_Paper_1_equation <-  function( M_bytes_per_chain,
                                                         N_units,
                                                         kind_of_tape,
                                                         L2_cache_size_bytes_per_core,
                                                         N_threads_per_core
) {

        fraction_of_the_L2_cache_of_each_kind_of_tape <-  c(Stan_math = 4.0, nico_math = 2.3)
        if (!(kind_of_tape %in% names(fraction_of_the_L2_cache_of_each_kind_of_tape))) {
              stop(paste0("fn_sampling_num_chunks_of_Paper_1_equation: kind_of_tape must be \"Stan_math\" or ",
                          "\"nico_math\"; got: ", paste(as.character(kind_of_tape), collapse = ", ")))
        }
        f <-  fraction_of_the_L2_cache_of_each_kind_of_tape[[kind_of_tape]]
        L2_cache_size_bytes_per_hardware_thread <-  L2_cache_size_bytes_per_core / N_threads_per_core
        N_pieces_for_the_L2_cache <-  M_bytes_per_chain / (f * L2_cache_size_bytes_per_hardware_thread)
        num_chunks_sampling <-  2^max(0, ceiling(log2(max(1, N_pieces_for_the_L2_cache))))
        ##
        return(list( num_chunks_sampling                      = num_chunks_sampling,
                     n_threads_WCP_sampling                   = 1,
                     chunk_size_sampling                      = ceiling(N_units / num_chunks_sampling),
                     fraction_of_the_L2_cache                 = f,
                     L2_cache_size_bytes_per_hardware_thread  = L2_cache_size_bytes_per_hardware_thread,
                     N_pieces_for_the_L2_cache                = N_pieces_for_the_L2_cache,
                     M_bytes_per_chain                        = M_bytes_per_chain,
                     N_units                                  = N_units,
                     kind_of_tape                             = kind_of_tape))

}


#'
#' The powers of two up to the hardware threads of one L3 cache, and floor(hardware threads / chains) when that
#' is larger (e.g. 45 on 180 hardware threads with 4 chains), each at most floor(hardware threads / chains).
#'
#' @param N_hardware_threads_available The hardware threads available.
#' @param N_hardware_threads_per_L3_available The largest number of available hardware threads in one L3 cache.
#' @param n_chains_burnin The burn-in chains run in parallel.
#' @return The candidates, in increasing order.
#' @export
fn_threads_per_chain_candidates_of_Paper_1_burnin_equation <-  function( N_hardware_threads_available,
                                                                         N_hardware_threads_per_L3_available,
                                                                         n_chains_burnin
) {

        most_threads_per_chain <-  max(1, floor(N_hardware_threads_available / n_chains_burnin))
        powers_of_two <-  2^(0:floor(log2(max(1, min(N_hardware_threads_per_L3_available,
                                                       most_threads_per_chain)))))
        return(sort(unique(c(powers_of_two, most_threads_per_chain))))

}


#'
#' \preformatted{
#'   T         = the largest candidate T with, within one L3 cache (T <= H_L3):
#'                 t_grad / T >= 12 us, or (t_grad / T >= 7 us and N_units / T >= 32),
#'               and across L3 caches (T > H_L3): t_grad / T >= 200 us; 1 when none passes
#'   L2/thread = L2_cache_size_bytes_per_core / ceiling(n_chains_burnin x T / N_cores_available)
#'   N_chunks  = 2^ceil( log2( max( M_bytes/chain / (g x L2/thread), T ) ) )
#' }
#'
#' with t_grad the measured seconds per lp_grad call of the library ([fn_seconds_per_lp_grad_call_of_compiled_
#' Stan_model()]), H_L3 the largest number of available hardware threads in one L3 cache, and g = 0.07 for the
#' autodiff tape of a Stan::math library, 0.5 for the tape of a nico::math (AVX) library. Fitted to the burn-in
#' ground truth (4 chains) of the same five models on both machines (fit_the_burnin_equations_with_units_per_
#' thread_and_a_largest_number_of_chunks_per_thread.py: 67 of 67 timed cases within 10%, the worst 9.5%).
#'
#' @param M_bytes_per_chain Memory, in bytes, of one full-data log-density-and-gradient evaluation of one chain.
#' @param N_units The number of units reduce_sum_static sums over (the N of the Stan data).
#' @param n_chains_burnin The burn-in chains run in parallel.
#' @param seconds_per_lp_grad_call The measured seconds of one log density and gradient of the library.
#' @param kind_of_tape "Stan_math" (a Stan::math library) or "nico_math" (a nico::math, AVX, library).
#' @param threads_per_chain_candidates The threads per chain tried (e.g. from
#'   [fn_threads_per_chain_candidates_of_Paper_1_burnin_equation()]).
#' @param N_hardware_threads_per_L3_available The largest number of available hardware threads in one L3 cache.
#' @param N_cores_available The cores available.
#' @param L2_cache_size_bytes_per_core The L2 cache of one core, in bytes.
#' @return A list with n_threads_WCP_burnin, num_chunks_burnin and every intermediate quantity.
#' @export
fn_burnin_threads_per_chain_and_num_chunks_of_Paper_1_equations <-  function(
        M_bytes_per_chain,
        N_units,
        n_chains_burnin,
        seconds_per_lp_grad_call,
        kind_of_tape,
        threads_per_chain_candidates,
        N_hardware_threads_per_L3_available,
        N_cores_available,
        L2_cache_size_bytes_per_core
) {

        if (!(kind_of_tape %in% c("Stan_math", "nico_math"))) {
              stop(paste0("fn_burnin_threads_per_chain_and_num_chunks_of_Paper_1_equations: kind_of_tape ",
                          "must be \"Stan_math\" or \"nico_math\"; got: ",
                          paste(as.character(kind_of_tape), collapse = ", ")))
        }
        ## (the seconds of gradient each thread needs: within one L3 cache, and across L3 caches)
        fewest_seconds_per_thread_within_one_L3                   <-  12e-6
        fewest_seconds_per_thread_within_one_L3_with_enough_units <-  7e-6
        fewest_units_per_thread_with_the_lower_seconds            <-  32
        fewest_seconds_per_thread_across_L3_caches                <-  200e-6
        ##
        ## ---- threads per chain: the largest candidate whose share of a gradient is enough:
        ##
        n_threads_WCP_burnin <-  1
        for (threads_per_chain in sort(threads_per_chain_candidates)) {
              seconds_per_thread <-  seconds_per_lp_grad_call / threads_per_chain
              if (threads_per_chain > N_hardware_threads_per_L3_available) {
                    enough <-  (seconds_per_thread >= fewest_seconds_per_thread_across_L3_caches)
              } else {
                    enough <-  (seconds_per_thread >= fewest_seconds_per_thread_within_one_L3) ||
                               ((seconds_per_thread >=
                                 fewest_seconds_per_thread_within_one_L3_with_enough_units) &&
                                (N_units / threads_per_chain >= fewest_units_per_thread_with_the_lower_seconds))
              }
              if (enough) n_threads_WCP_burnin <-  max(n_threads_WCP_burnin, threads_per_chain)
        }
        ##
        ## ---- chunks: from the tape and the L2 cache each thread has; no upper bound on N_chunks:
        ##
        chunks <-  fn_burnin_num_chunks_of_Paper_1_equation_at_threads_per_chain(
                         M_bytes_per_chain             = M_bytes_per_chain,
                         N_units                       = N_units,
                         n_chains_burnin               = n_chains_burnin,
                         n_threads_WCP_burnin          = n_threads_WCP_burnin,
                         kind_of_tape                  = kind_of_tape,
                         N_cores_available             = N_cores_available,
                         L2_cache_size_bytes_per_core  = L2_cache_size_bytes_per_core)
        num_chunks_burnin <-  chunks$num_chunks_burnin
        ##
        return(list( n_threads_WCP_burnin            = n_threads_WCP_burnin,
                     num_chunks_burnin               = num_chunks_burnin,
                     chunk_size_burnin               = ceiling(N_units / num_chunks_burnin),
                     fraction_of_the_L2_cache        = chunks$fraction_of_the_L2_cache,
                     L2_cache_size_bytes_per_thread  = chunks$L2_cache_size_bytes_per_thread,
                     N_pieces_for_the_L2_cache       = chunks$N_pieces_for_the_L2_cache,
                     threads_per_chain_candidates    = sort(threads_per_chain_candidates),
                     seconds_per_lp_grad_call        = seconds_per_lp_grad_call,
                     M_bytes_per_chain               = M_bytes_per_chain,
                     N_units                         = N_units,
                     n_chains_burnin                 = n_chains_burnin,
                     kind_of_tape                    = kind_of_tape))

}


#'
#' \preformatted{
#'   L2/thread = L2_cache_size_bytes_per_core / ceiling(n_chains_burnin x T / N_cores_available)
#'   N_chunks  = 2^ceil( log2( max( M_bytes/chain / (g x L2/thread), T ) ) )
#' }
#'
#' with g = 0.07 for the autodiff tape of a Stan::math library, 0.5 for the tape of a nico::math (AVX) library
#' (the chunks of [fn_burnin_threads_per_chain_and_num_chunks_of_Paper_1_equations()], for any T).
#'
#' @param M_bytes_per_chain Memory, in bytes, of one full-data log-density-and-gradient evaluation of one chain.
#' @param N_units The number of units reduce_sum_static sums over (the N of the Stan data).
#' @param n_chains_burnin The burn-in chains run in parallel.
#' @param n_threads_WCP_burnin T, the threads per chain.
#' @param kind_of_tape "Stan_math" (a Stan::math library) or "nico_math" (a nico::math, AVX, library).
#' @param N_cores_available The cores available.
#' @param L2_cache_size_bytes_per_core The L2 cache of one core, in bytes.
#' @return A list with num_chunks_burnin and every intermediate quantity.
#' @export
fn_burnin_num_chunks_of_Paper_1_equation_at_threads_per_chain <-  function( M_bytes_per_chain,
                                                                            N_units,
                                                                            n_chains_burnin,
                                                                            n_threads_WCP_burnin,
                                                                            kind_of_tape,
                                                                            N_cores_available,
                                                                            L2_cache_size_bytes_per_core
) {

        fraction_of_the_L2_cache_of_each_kind_of_tape <-  c(Stan_math = 0.07, nico_math = 0.5)
        if (!(kind_of_tape %in% names(fraction_of_the_L2_cache_of_each_kind_of_tape))) {
              stop(paste0("fn_burnin_num_chunks_of_Paper_1_equation_at_threads_per_chain: kind_of_tape must ",
                          "be \"Stan_math\" or \"nico_math\"; got: ",
                          paste(as.character(kind_of_tape), collapse = ", ")))
        }
        g <-  fraction_of_the_L2_cache_of_each_kind_of_tape[[kind_of_tape]]
        L2_cache_size_bytes_per_thread <-  L2_cache_size_bytes_per_core /
                                            max(1, ceiling(n_chains_burnin * n_threads_WCP_burnin /
                                                           N_cores_available))
        N_pieces_for_the_L2_cache <-  M_bytes_per_chain / (g * L2_cache_size_bytes_per_thread)
        num_chunks_burnin <-  2^max(0, ceiling(log2(max(1, max(N_pieces_for_the_L2_cache,
                                                                n_threads_WCP_burnin)))))
        ##
        return(list( num_chunks_burnin               = num_chunks_burnin,
                     fraction_of_the_L2_cache        = g,
                     L2_cache_size_bytes_per_thread  = L2_cache_size_bytes_per_thread,
                     N_pieces_for_the_L2_cache       = N_pieces_for_the_L2_cache))

}




#' The kind of autodiff tape of a compiled Stan model, read from the nico_ calls of its Stan file
#'
#' "nico_math" when the model was compiled with NicoStan's AVX version (BayesMVP_AVX_version_of_Stan_model given),
#' or when its Stan file calls any nico_ function outside a // comment; "Stan_math" otherwise.
#' Paper 1's equations use it to choose f (sampling) and g (burn-in).
#'
#' @param Stan_model_file_path The Stan file the library was compiled from.
#' @param BayesMVP_AVX_version_of_Stan_model The AVX version NicoStan wrote and compiled (init_object's field of
#'   that name), or NULL when the Stan file was compiled as it is.
#' @return "Stan_math" or "nico_math".
#' @export
fn_kind_of_tape_of_Stan_model_from_the_nico_calls_in_its_Stan_file <-  function(
                                                                    Stan_model_file_path,
                                                                    BayesMVP_AVX_version_of_Stan_model
) {

        if (!is.null(BayesMVP_AVX_version_of_Stan_model)) return("nico_math")
        if (!(is.character(Stan_model_file_path) && (length(Stan_model_file_path) == 1) &&
              file.exists(Stan_model_file_path))) {
              stop(paste0("fn_kind_of_tape_of_Stan_model_from_the_nico_calls_in_its_Stan_file: no Stan file at ",
                          paste(as.character(Stan_model_file_path), collapse = ", ")))
        }
        ##
        code <-  R_fn_Stan_parallel_call_names(code = paste(readLines(Stan_model_file_path, warn = FALSE),
                                                            collapse = "\n"))
        protected <-  "\"(?:[^\"\\\\]|\\\\.)*\"|/\\*[\\s\\S]*?\\*/|//[^\\n]*"
        lines_of_the_Stan_file <-  gsub(pattern = protected, replacement = " ", x = code, perl = TRUE)
        calls_a_nico_function <-  any(grepl(pattern = "\\bnico_[A-Za-z0-9_]+[[:space:]]*\\(",
                                            x = lines_of_the_Stan_file, perl = TRUE))
        ##
        return(if (calls_a_nico_function) "nico_math" else "Stan_math")

}


#### ---- the sampling chunks for the CPUs of this process ------------------------------------------------------

#' Sampling chunks for the CPUs of this process (Paper 1's rule, fn_compute_initial_N_chunks())
#'
#' Applies [fn_compute_initial_N_chunks()] to the sampling chains, with the topology of the CPUs available to
#' this R process (its affinity mask, e.g. the CPUs given to taskset): the L3 caches they belong to and the
#' largest number of available CPUs in one L3 cache. The threads per core, the L2 cache size and the L3 cache size
#' are read from Linux sysfs. Without this, fn_compute_initial_N_chunks() would size the chunks for every L3 cache
#' of the machine, even for a fit that runs on one of them.
#'
#' @param M_bytes_per_chain Memory, in bytes, of one full-data log-density-and-gradient evaluation of one chain.
#' @param n_chains_sampling Number of sampling chains run in parallel.
#' @param n_threads_WCP_sampling Within-chain threads of each sampling chain (default 1).
#' @param gradient_type "autodiff" (Stan models) or "manual" (built-in models).
#' @return The list of [fn_compute_initial_N_chunks()] (with N_chunks), and topology (the CPUs available and their
#'   L3 caches, from fn_read_L3_cache_topology_of_available_CPUs()).
#' @export
fn_compute_sampling_num_chunks_for_CPUs_of_this_process <-  function( M_bytes_per_chain,
                                                                      n_chains_sampling,
                                                                      n_threads_WCP_sampling  = 1,
                                                                      gradient_type
) {

        topology <-  fn_read_L3_cache_topology_of_available_CPUs()
        ##
        ## ---- the threads per core and the L2 cache size, read from sysfs by fn_compute_initial_N_chunks() (its
        ##      chunk count for the whole machine is not used):
        ##
        ## whole_machine <-  suppressWarnings(
        ##       fn_compute_initial_N_chunks( M_bytes_per_chain    = M_bytes_per_chain,
        ##                                    N_chains             = n_chains_sampling,
        ##                                    N_threads_per_chain  = n_threads_WCP_sampling,
        ##                                    gradient_type        = gradient_type))
        ## (the threads per core and the L2 cache size of the first CPU of this process, read from its own sysfs
        ##  files: the whole machine's reading, every CPU's cache directories, took about 0.3 s of every choice of
        ##  a layout on the local-HPC; on a machine whose cores all have the same threads and L2 cache, as the
        ##  local-HPC and the laptop, they are the same values; the whole machine's reading when they cannot be
        ##  read)
        whole_machine <-  fn_read_threads_per_core_and_L2_cache_size_of_first_CPU_of_this_process()
        if (is.null(whole_machine)) {
              whole_machine <-  suppressWarnings(
                    fn_compute_initial_N_chunks( M_bytes_per_chain    = M_bytes_per_chain,
                                                 N_chains             = n_chains_sampling,
                                                 N_threads_per_chain  = n_threads_WCP_sampling,
                                                 gradient_type        = gradient_type))
        }
        N_threads_per_core <-  whole_machine$N_threads_per_core
        N_cores_per_L3_available <-  max(1, ceiling(topology$N_hardware_threads_per_L3_available /
                                                    N_threads_per_core))
        ##
        ## ---- the rule for the CPUs of this process:
        ##
        rule <-  fn_compute_initial_N_chunks( M_bytes_per_chain    = M_bytes_per_chain,
                                              N_chains             = n_chains_sampling,
                                              N_threads_per_chain  = n_threads_WCP_sampling,
                                              gradient_type        = gradient_type,
                                              N_L3_caches          = topology$N_L3_caches_available,
                                              N_cores_per_L3       = N_cores_per_L3_available,
                                              N_threads_per_core   = N_threads_per_core,
                                              L3_cache_size_bytes  = topology$L3_cache_size_bytes,
                                              L2_cache_size_bytes  = whole_machine$L2_cache_size_bytes)
        rule$topology <-  topology
        return(rule)

}























