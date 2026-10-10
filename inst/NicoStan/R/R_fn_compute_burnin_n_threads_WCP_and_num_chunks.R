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
##   num_chunks_burnin     = 2^ceil( log2( max( ceil(M_bytes/chain / S_target), T * j ) ) ),
##                           at most 2^floor(log2(N_units))
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
fn_read_L3_cache_topology_of_available_CPUs <-  function( CPU_ids_available    = NULL,
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
        sysfs_CPU_directory                  = "/sys/devices/system/cpu"
) {

        ##
        ## ---- the list returned by fn_estimate_initial_M_bytes_per_chain() is accepted in place of the number:
        ##
        if (is.list(x = M_bytes_per_chain) && !is.null(x = M_bytes_per_chain$M_bytes_per_chain)) {
              M_bytes_per_chain <-  M_bytes_per_chain$M_bytes_per_chain
        }
        ##
        fn_check_positive_number <-  function(value, value_name, whole_number = FALSE) {
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
        if (is.null(x = n_threads_WCP_burnin)) {
              n_threads_WCP_burnin <-  2^floor(log2(min(n_threads_per_chain_from_hardware,
                                                        n_threads_per_chain_from_L3_cache)))
        }
        ##
        ## ---- the pieces: each fits in its thread's share of the L3 cache, and pieces_per_thread per thread:
        ##
        N_active_threads_per_L3 <-  min(topology$N_hardware_threads_per_L3_available,
                                        ceiling(n_chains_burnin * n_threads_WCP_burnin /
                                                topology$N_L3_caches_available))
        S_target_bytes <-  topology$L3_cache_size_bytes / N_active_threads_per_L3
        N_pieces_for_cache <-  ceiling(M_bytes_per_chain / S_target_bytes)
        N_pieces_for_load_balance <-  n_threads_WCP_burnin * pieces_per_thread
        ##
        num_chunks_burnin <-  2^ceiling(log2(max(N_pieces_for_cache, N_pieces_for_load_balance)))
        num_chunks_burnin <-  min(num_chunks_burnin, 2^floor(log2(N_units)))
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
#' (so that a drift of the machine affects every candidate alike). This is how Paper 1's burn-in benchmark timed
#' its configurations. Each candidate's chunk_size = ceiling(N / num_chunks) is written to a temporary JSON data
#' file.
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
#' @return A data frame: num_chunks and median_seconds_per_burnin_iteration.
#' @keywords internal
fn_time_burnin_iteration_for_each_num_chunks <-  function( init_object,
                                                           Stan_data_list,
                                                           n_chains_burnin,
                                                           n_threads_WCP_burnin,
                                                           num_chunks_candidates,
                                                           n_rounds            = 3,
                                                           n_timed_iterations  = 3,
                                                           minimum_seconds_per_timing = 0.2,
                                                           L_main              = 20,
                                                           eps_main            = 1e-5
) {

        n_params_main <-  init_object$n_params_main
        n_nuisance <-  init_object$n_nuisance
        ##
        ## ---- one JSON data file per candidate:
        ##
        json_file_paths <-  vapply(X = num_chunks_candidates, FUN = function(num_chunks) {
              Stan_data_for_candidate <-  Stan_data_list
              Stan_data_for_candidate$chunk_size <-  as.integer(ceiling(Stan_data_list$N / num_chunks))
              json_file_path <-  tempfile(pattern = paste0("burnin_chunks_", num_chunks, "_"), fileext = ".json")
              cmdstanr::write_stan_json(data = Stan_data_for_candidate, file = json_file_path)
              normalizePath(json_file_path, mustWork = TRUE)
        }, FUN.VALUE = "")
        on.exit(unlink(x = json_file_paths), add = TRUE)
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
                                                                    metric_shape_main = "dense")
        fn_small_values <-  function(n_values) 0.005 * (((seq_len(n_values) * 7) %% 5) - 2)
        initial_main <-  matrix(fn_small_values(n_params_main * n_chains_burnin), nrow = n_params_main)
        initial_nuisance <-  matrix(fn_small_values(n_nuisance * n_chains_burnin), nrow = n_nuisance)
        ##
        RcppParallel::setThreadOptions(numThreads = n_chains_burnin * n_threads_WCP_burnin)
        seconds <-  matrix(NA_real_, nrow = n_rounds, ncol = length(num_chunks_candidates))
        for (round_index in seq_len(n_rounds)) {
              for (candidate_index in seq_along(num_chunks_candidates)) {

                    Model_args_as_Rcpp_List <-  init_object$Model_args_as_Rcpp_List
                    Model_args_as_Rcpp_List$json_file_path <-  json_file_paths[candidate_index]
                    worker <-  fn_create_persistent_burnin_worker(
                                     n_threads_R = n_chains_burnin, partitioned_HMC_R = FALSE,
                                     diffusion_HMC_R = TRUE, Model_type_R = "Stan", sample_nuisance_R = TRUE,
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
                    ## (one untimed iteration, then at least n_timed_iterations and at least
                    ##  minimum_seconds_per_timing of timed iterations)
                    fn_persistent_burnin_run_one_iter(worker, 1000 * round_index, 1)
                    start_time <-  proc.time()[["elapsed"]]
                    n_iterations_timed <-  0
                    while (n_iterations_timed < n_timed_iterations ||
                           proc.time()[["elapsed"]] - start_time < minimum_seconds_per_timing) {
                          n_iterations_timed <-  n_iterations_timed + 1
                          fn_persistent_burnin_run_one_iter(worker, 1000 * round_index + 1 + n_iterations_timed,
                                                            1 + n_iterations_timed)
                    }
                    seconds_per_iteration <-  (proc.time()[["elapsed"]] - start_time) / n_iterations_timed
                    rm(worker)
                    invisible(gc(verbose = FALSE))
                    seconds[round_index, candidate_index] <-  mean(seconds_per_iteration)

              }
        }
        return(data.frame( num_chunks                          = num_chunks_candidates,
                           median_seconds_per_burnin_iteration = apply(X = seconds, MARGIN = 2,
                                                                       FUN = stats::median)))

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
#' N = 5,000). Without init_object (or if the timing fails), T threads and two pieces per thread.
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
#' @param init_object The model's initialisation object (default NULL: the chunk counts are not timed).
#' @param reuse_saved_choice TRUE (default): reuse the choice saved for the same model file, N, number of
#'   burn-in chains and CPUs, when there is one.
#' @param ... Further arguments of [fn_compute_burnin_n_threads_WCP_and_num_chunks()].
#' @return The list of [fn_compute_burnin_n_threads_WCP_and_num_chunks()] for the chosen configuration, with
#'   M_bytes_measurement and configuration_timings (NULL when not timed).
#' @export
fn_compute_burnin_n_threads_WCP_and_num_chunks_for_Stan_model <-  function( model_so_file,
                                                                           Stan_data_list,
                                                                           n_chains_burnin  = 4,
                                                                           init_list        = NULL,
                                                                           init_object      = NULL,
                                                                           reuse_saved_choice = TRUE,
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
        ##
        ## ---- the full data as one piece, so that the measured memory is that of the full-data gradient:
        ##
        Stan_data_list_one_piece <-  Stan_data_list
        Stan_data_list_one_piece$chunk_size <-  as.integer(Stan_data_list$N)
        M_bytes_measurement <-  fn_estimate_initial_M_bytes_per_chain( model_so_file   = model_so_file,
                                                                       Stan_data_list  = Stan_data_list_one_piece,
                                                                       init_list       = init_list)
        ##
        ## ---- the rule (T threads, 2 pieces per thread), and its configurations for 1, 2 and 4 pieces per thread
        ##      at T and T / 2 threads per chain:
        ##
        rule <-  fn_compute_burnin_n_threads_WCP_and_num_chunks( M_bytes_per_chain  = M_bytes_measurement,
                                                                 N_units            = Stan_data_list$N,
                                                                 n_chains_burnin    = n_chains_burnin,
                                                                 pieces_per_thread  = 2,
                                                                 ...)
        ## (1 thread per chain too: for a model whose likelihood has a long sequential part outside
        ##  reduce_sum_static, e.g. a time series' recursion or a survival model's risk-set scan, no within-chain
        ##  threads can be the fastest)
        thread_candidates <-  unique(c(rule$n_threads_WCP_burnin, max(1, rule$n_threads_WCP_burnin / 2), 1))
        rules <-  list()
        for (n_threads_candidate in thread_candidates) {
              for (pieces_per_thread in c(1, 2, 4)) {
                    rules[[length(rules) + 1]] <-  fn_compute_burnin_n_threads_WCP_and_num_chunks(
                                                         M_bytes_per_chain     = M_bytes_measurement,
                                                         N_units               = Stan_data_list$N,
                                                         n_chains_burnin       = n_chains_burnin,
                                                         pieces_per_thread     = pieces_per_thread,
                                                         n_threads_WCP_burnin  = n_threads_candidate,
                                                         ...)
              }
        }
        configurations <-  unique(data.frame(
              n_threads_WCP_burnin = vapply(X = rules, FUN = function(x) x$n_threads_WCP_burnin, FUN.VALUE = 0),
              num_chunks_burnin = vapply(X = rules, FUN = function(x) x$num_chunks_burnin, FUN.VALUE = 0)))
        ##
        ## ---- the fastest of the rule's chunk counts, timed in NicoStan's burn-in worker:
        ##
        configuration_timings <-  NULL
        if (!is.null(init_object) && nrow(configurations) > 1) {
              configuration_timings <-  tryCatch(
                    expr  = do.call(rbind, lapply(X = unique(configurations$n_threads_WCP_burnin),
                                                  FUN = function(n_threads_candidate) {
                          timings <-  fn_time_burnin_iteration_for_each_num_chunks(
                                            init_object           = init_object,
                                            Stan_data_list        = Stan_data_list,
                                            n_chains_burnin       = n_chains_burnin,
                                            n_threads_WCP_burnin  = n_threads_candidate,
                                            num_chunks_candidates = configurations$num_chunks_burnin[
                                                  configurations$n_threads_WCP_burnin == n_threads_candidate])
                          cbind(n_threads_WCP_burnin = n_threads_candidate, timings)
                    })),
                    error = function(condition) {
                          message(paste0("Burn-in configurations not timed (", conditionMessage(condition),
                                         "); the rule's threads and two pieces per thread are used."))
                          NULL
                    })
              if (!is.null(configuration_timings)) {
                    fastest_index <-  which.min(configuration_timings$median_seconds_per_burnin_iteration)
                    fastest <-  configuration_timings[fastest_index, ]
                    rule <-  rules[[which(vapply(X = rules, FUN = function(x) {
                          x$n_threads_WCP_burnin == fastest$n_threads_WCP_burnin &&
                                x$num_chunks_burnin == fastest$num_chunks
                    }, FUN.VALUE = FALSE))[1]]]
              }
        }
        rule$M_bytes_measurement <-  M_bytes_measurement
        rule$configuration_timings <-  configuration_timings
        rule$CPU_list <-  CPU_list
        rule$reused_saved_choice <-  FALSE
        rule$seconds_spent_choosing <-  proc.time()[["elapsed"]] - time_choice_start
        ## (saved for reuse when the configurations were timed, or when there was only one to choose)
        if (!is.null(configuration_timings) || nrow(configurations) == 1) {
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
























