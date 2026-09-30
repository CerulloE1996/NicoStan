#### =====================================================================================================================================
## R_fn_compute_initial_N_chunks.R
##
## ---- Starting number of chunks (N_chunks) from the memory of one full-data gradient evaluation per chain (M_bytes/chain) ------------
##
## The rule chooses the smallest number of chunks for which one chunk of one chain fits in that chain's share of a cache:
##
##   N_threads                  = N_chains * N_threads/chain
##   N_active_threads/L3        = min( N_threads/core * N_cores/L3 ,  ceil(N_threads / N_L3) )
##   S_target                   = S_L3 / N_active_threads/L3                              (autodiff (Stan) gradients)
##                              = S_L2 / max(1, N_active_threads/L3 / N_cores/L3)         (manual gradients)
##   N_chunks                   = N_threads/chain * ceil( max( ceil(M_bytes/chain / S_target), N_threads/chain ) / N_threads/chain )
##
## i.e. the threads are spread evenly over the L3 caches (at most every hardware thread of an L3 is active), an autodiff gradient
## shares its L3 with the other active threads of that L3, and a manual gradient shares the L2 of its core with its SMT sibling
## when more threads than cores are active on that L3. N_chunks is a multiple of N_threads/chain, and at least N_threads/chain,
## so that within-chain parallel threads receive equal numbers of chunks.
##
## The cache topology (number of L3 caches, cores per L3, threads per core and the L3 and L2 sizes) is read from Linux sysfs
## (/sys/devices/system/cpu/cpu*/cache/index*/{level,type,size,shared_cpu_list} and
## /sys/devices/system/cpu/cpu*/topology/thread_siblings_list); every hardware quantity can also be given as an argument, which
## then replaces the value read from sysfs (and is required on other operating systems). The thread topology (cores and threads
## per core) and the cache files are read separately, so that a missing cache file does not stop a call that needs only the
## thread topology. If neither thread_siblings_list nor core_cpus_list exists, each hardware thread is counted as its own core,
## and a warning asks for N_threads_per_core and N_cores_per_L3 to be given.
##




#### ---- sysfs helpers --------------------------------------------------------------------------------------------------------------------

#' Parse a Linux sysfs CPU list (e.g. "0-7,96-103") into a vector of CPU ids
#' @param CPU_list_string A sysfs CPU list string.
#' @return Sorted numeric vector of CPU ids.
#' @keywords internal
#' @noRd
fn_parse_sysfs_CPU_list <-  function( CPU_list_string ) {

        CPU_list_string <-  trimws(x = CPU_list_string)
        ##
        if (is.na(x = CPU_list_string) || !nzchar(x = CPU_list_string)) {
              return(numeric(0))
        }
        ##
        CPU_list_parts <-  strsplit(x = CPU_list_string, split = ",", fixed = TRUE)[[1]]
        ##
        CPU_ids <-  unlist(x = lapply( X   = CPU_list_parts,
                                       FUN = function(CPU_list_part) {

                                             range_ends <-  as.numeric(strsplit(x = trimws(x = CPU_list_part), split = "-", fixed = TRUE)[[1]])
                                             ##
                                             if (length(x = range_ends) == 1) {
                                                   return(range_ends)
                                             }
                                             ##
                                             return(seq(from = range_ends[1], to = range_ends[2]))

                                       }))
        ##
        return(sort(x = unique(x = CPU_ids)))

}




#' Parse a Linux sysfs cache size (e.g. "32768K", "16M") into bytes (K = 1024 bytes)
#' @param cache_size_string A sysfs cache size string.
#' @return Cache size in bytes.
#' @keywords internal
#' @noRd
fn_parse_sysfs_cache_size_bytes <-  function( cache_size_string ) {

        cache_size_string <-  toupper(x = trimws(x = cache_size_string))
        ##
        size_unit <-  sub(pattern = "^[0-9.]+[[:space:]]*", replacement = "", x = cache_size_string)
        size_unit <-  sub(pattern = "I?B$", replacement = "", x = size_unit)
        if (!nzchar(x = size_unit)) size_unit <-  "bytes"
        ##
        size_number <-  suppressWarnings(as.numeric(sub(pattern = "^([0-9.]+).*$", replacement = "\\1", x = cache_size_string)))
        ##
        size_multiplier <-  switch( EXPR  = size_unit,
                                    "K"     = 1024,
                                    "M"     = 1024^2,
                                    "G"     = 1024^3,
                                    "bytes" = 1,
                                    NA_real_)
        ##
        if (!is.finite(x = size_number) || !is.finite(x = size_multiplier)) {
              stop(paste0("fn_parse_sysfs_cache_size_bytes: unrecognised cache size '", cache_size_string, "'."))
        }
        ##
        return(size_number * size_multiplier)

}




#' Most common value of a numeric vector (the smallest such value on ties)
#' @keywords internal
#' @noRd
fn_most_common_value <-  function( values ) {

        value_counts <-  table(values)
        ##
        most_common_values <-  as.numeric(names(x = value_counts)[value_counts == max(value_counts)])
        ##
        return(min(most_common_values))

}




#### ---- CPU cache topology from sysfs -----------------------------------------------------------------------------------------------------

#' Read the CPU cache topology used by the N_chunks rule from Linux sysfs
#'
#' Counts the distinct L3 cache instances (by their shared_cpu_list), the cores in each L3 (distinct thread_siblings_list sets
#' among the CPUs sharing it), the hardware threads per core, and the L3 and L2 (unified or data) cache sizes. On CPUs where
#' these differ between cores or caches (for example hybrid designs, or L3 caches of different sizes), the most common count and
#' the smallest cache size are returned, and the differences are listed in topology_notes. Quantities that cannot be read are
#' returned as NA with a note. If neither thread_siblings_list nor core_cpus_list exists, each hardware thread is counted as its
#' own core (thread_topology_available = FALSE).
#'
#' @param sysfs_CPU_directory The sysfs CPU directory (default "/sys/devices/system/cpu").
#' @return A list with N_L3_caches, N_cores_per_L3, N_threads_per_core, L3_cache_size_bytes, L2_cache_size_bytes, N_cores,
#'   N_hardware_threads, thread_topology_available and topology_notes.
#' @keywords internal
#' @noRd
fn_read_CPU_cache_topology_from_sysfs <-  function( sysfs_CPU_directory = "/sys/devices/system/cpu" ) {

        if (!identical(x = Sys.info()[["sysname"]], y = "Linux") || !dir.exists(paths = sysfs_CPU_directory)) {
              stop(paste0("fn_read_CPU_cache_topology_from_sysfs: the cache topology is read from Linux sysfs (", sysfs_CPU_directory,
                          "), which is not available on this system. Supply N_L3_caches, N_cores_per_L3, N_threads_per_core, ",
                          "L3_cache_size_bytes and L2_cache_size_bytes as arguments instead."))
        }
        ##
        fn_read_first_line <-  function(file_path) {
              if (!file.exists(file_path)) return(NA_character_)
              first_line <-  tryCatch( expr  = readLines(con = file_path, n = 1, warn = FALSE),
                                       error = function(condition) character(0))
              if (length(x = first_line) == 0) return(NA_character_)
              return(first_line)
        }
        ##
        ## ---- online CPUs (hardware threads):
        ##
        online_CPU_list_string <-  fn_read_first_line(file_path = file.path(sysfs_CPU_directory, "online"))
        ##
        if (!is.na(x = online_CPU_list_string)) {
              online_CPU_ids <-  fn_parse_sysfs_CPU_list(CPU_list_string = online_CPU_list_string)
        } else {
              CPU_directory_names <-  list.files(path = sysfs_CPU_directory, pattern = "^cpu[0-9]+$")
              online_CPU_ids <-  sort(x = as.numeric(sub(pattern = "^cpu", replacement = "", x = CPU_directory_names)))
        }
        ##
        if (length(x = online_CPU_ids) == 0) {
              stop(paste0("fn_read_CPU_cache_topology_from_sysfs: no CPUs found under ", sysfs_CPU_directory, "."))
        }
        ##
        ## ---- one row per (CPU, cache index), and the thread siblings of each CPU:
        ##
        cache_rows <-  list()
        thread_siblings_list_strings <-  character(0)
        N_CPUs_without_thread_topology <-  0
        ##
        for (CPU_id in online_CPU_ids) {

              CPU_directory <-  file.path(sysfs_CPU_directory, paste0("cpu", CPU_id))
              ##
              thread_siblings_list_string <-  fn_read_first_line(file_path = file.path(CPU_directory, "topology", "thread_siblings_list"))
              if (is.na(x = thread_siblings_list_string)) {
                    thread_siblings_list_string <-  fn_read_first_line(file_path = file.path(CPU_directory, "topology", "core_cpus_list"))
              }
              if (is.na(x = thread_siblings_list_string)) {
                    thread_siblings_list_string <-  as.character(CPU_id)
                    N_CPUs_without_thread_topology <-  N_CPUs_without_thread_topology + 1
              }
              thread_siblings_list_strings[as.character(CPU_id)] <-  thread_siblings_list_string
              ##
              cache_index_directories <-  list.files(path = file.path(CPU_directory, "cache"), pattern = "^index[0-9]+$", full.names = TRUE)
              ##
              for (cache_index_directory in cache_index_directories) {

                    cache_level <-  suppressWarnings(as.numeric(fn_read_first_line(file_path = file.path(cache_index_directory, "level"))))
                    ##
                    if (is.na(x = cache_level) || !(cache_level %in% c(2, 3))) next
                    ##
                    cache_rows[[length(x = cache_rows) + 1]] <-
                          data.frame( CPU_id                  = CPU_id,
                                      cache_level             = cache_level,
                                      cache_type              = fn_read_first_line(file_path = file.path(cache_index_directory, "type")),
                                      cache_size_string       = fn_read_first_line(file_path = file.path(cache_index_directory, "size")),
                                      shared_CPU_list_string  = fn_read_first_line(file_path = file.path(cache_index_directory, "shared_cpu_list")),
                                      stringsAsFactors        = FALSE)

              }

        }
        ##
        topology_notes <-  character(0)
        ##
        ## (the cache files are read separately from the thread topology: when they are missing, only the cache quantities are NA):
        if (length(x = cache_rows) == 0) {

              cache_table <-  data.frame( CPU_id                  = numeric(0),
                                          cache_level             = numeric(0),
                                          cache_type              = character(0),
                                          cache_size_string       = character(0),
                                          shared_CPU_list_string  = character(0),
                                          stringsAsFactors        = FALSE)
              topology_notes <-  c(topology_notes, paste0("no L2 or L3 cache information was found under ", sysfs_CPU_directory))

        } else {

              cache_table <-  do.call(what = rbind, args = cache_rows)

        }
        ##
        cache_table <-  cache_table[cache_table$cache_type %in% c("Unified", "Data"), , drop = FALSE]
        cache_table <-  cache_table[!is.na(x = cache_table$shared_CPU_list_string) & !is.na(x = cache_table$cache_size_string), , drop = FALSE]
        ##
        thread_topology_available <-  N_CPUs_without_thread_topology == 0
        ##
        if (!thread_topology_available) {
              topology_notes <-  c(topology_notes,
                                   paste0("neither thread_siblings_list nor core_cpus_list was found for ", N_CPUs_without_thread_topology, " of ",
                                          length(x = online_CPU_ids), " CPUs, so each of those hardware threads is counted as its own core ",
                                          "(N_threads_per_core and N_cores_per_L3 are then wrong on a CPU with SMT)"))
        }
        ##
        ## ---- cores (distinct thread-sibling sets) and hardware threads per core:
        ##
        core_thread_sibling_sets <-  unique(x = unname(obj = thread_siblings_list_strings))
        N_cores <-  length(x = core_thread_sibling_sets)
        ##
        N_threads_per_each_core <-  sapply( X   = core_thread_sibling_sets,
                                            FUN = function(thread_siblings_list_string) {
                                                  sum(fn_parse_sysfs_CPU_list(CPU_list_string = thread_siblings_list_string) %in% online_CPU_ids)
                                            })
        N_threads_per_core <-  fn_most_common_value(values = N_threads_per_each_core)
        ##
        if (length(x = unique(x = N_threads_per_each_core)) > 1) {
              N_threads_per_core_values <-  paste(sort(x = unique(x = N_threads_per_each_core)), collapse = ", ")
              topology_notes <-  c(topology_notes,
                                   paste0("cores have different numbers of hardware threads (", N_threads_per_core_values,
                                          "); the most common (", N_threads_per_core, ") is used"))
        }
        ##
        ## ---- L3 caches: distinct instances, their sizes and the cores that share each one:
        ##
        L3_cache_table <-  cache_table[cache_table$cache_level == 3, , drop = FALSE]
        ##
        if (nrow(x = L3_cache_table) > 0) {

              L3_cache_table <-  L3_cache_table[!duplicated(x = L3_cache_table$shared_CPU_list_string), , drop = FALSE]
              N_L3_caches <-  nrow(x = L3_cache_table)
              ##
              L3_cache_size_bytes_each_instance <-  sapply(X = L3_cache_table$cache_size_string, FUN = fn_parse_sysfs_cache_size_bytes)
              L3_cache_size_bytes <-  min(L3_cache_size_bytes_each_instance)
              ##
              if (length(x = unique(x = L3_cache_size_bytes_each_instance)) > 1) {
                    topology_notes <-  c(topology_notes,
                                         paste0("L3 caches have different sizes; the smallest (", L3_cache_size_bytes / 1024^2, " MiB) is used"))
              }
              ##
              N_cores_per_each_L3 <-  sapply( X   = L3_cache_table$shared_CPU_list_string,
                                              FUN = function(shared_CPU_list_string) {
                                                    CPU_ids_sharing_this_L3 <-  fn_parse_sysfs_CPU_list(CPU_list_string = shared_CPU_list_string)
                                                    CPU_ids_sharing_this_L3 <-  CPU_ids_sharing_this_L3[CPU_ids_sharing_this_L3 %in% online_CPU_ids]
                                                    length(x = unique(x = thread_siblings_list_strings[as.character(CPU_ids_sharing_this_L3)]))
                                              })
              N_cores_per_L3 <-  fn_most_common_value(values = N_cores_per_each_L3)
              ##
              if (length(x = unique(x = N_cores_per_each_L3)) > 1) {
                    N_cores_per_L3_values <-  paste(sort(x = unique(x = N_cores_per_each_L3)), collapse = ", ")
                    topology_notes <-  c(topology_notes,
                                         paste0("L3 caches are shared by different numbers of cores (", N_cores_per_L3_values,
                                                "); the most common (", N_cores_per_L3, ") is used"))
              }

        } else {

              N_L3_caches <-  NA_real_
              N_cores_per_L3 <-  NA_real_
              L3_cache_size_bytes <-  NA_real_
              topology_notes <-  c(topology_notes, "no L3 cache was found in sysfs")

        }
        ##
        ## ---- L2 caches: size per instance:
        ##
        L2_cache_table <-  cache_table[cache_table$cache_level == 2, , drop = FALSE]
        ##
        if (nrow(x = L2_cache_table) > 0) {

              L2_cache_table <-  L2_cache_table[!duplicated(x = L2_cache_table$shared_CPU_list_string), , drop = FALSE]
              ##
              L2_cache_size_bytes_each_instance <-  sapply(X = L2_cache_table$cache_size_string, FUN = fn_parse_sysfs_cache_size_bytes)
              L2_cache_size_bytes <-  min(L2_cache_size_bytes_each_instance)
              ##
              if (length(x = unique(x = L2_cache_size_bytes_each_instance)) > 1) {
                    topology_notes <-  c(topology_notes,
                                         paste0("L2 caches have different sizes; the smallest (", L2_cache_size_bytes / 1024, " KiB) is used"))
              }

        } else {

              L2_cache_size_bytes <-  NA_real_
              topology_notes <-  c(topology_notes, "no L2 cache was found in sysfs")

        }
        ##
        return(list( N_L3_caches                = N_L3_caches,
                     N_cores_per_L3             = N_cores_per_L3,
                     N_threads_per_core         = N_threads_per_core,
                     L3_cache_size_bytes        = unname(obj = L3_cache_size_bytes),
                     L2_cache_size_bytes        = unname(obj = L2_cache_size_bytes),
                     N_cores                    = N_cores,
                     N_hardware_threads         = length(x = online_CPU_ids),
                     thread_topology_available  = thread_topology_available,
                     topology_notes             = topology_notes))

}




#### ---- N_chunks rule ----------------------------------------------------------------------------------------------------------------------

#' Starting number of chunks (N_chunks) from M_bytes/chain and the CPU cache topology
#'
#' Turns the memory used by one full-data log-density-and-gradient evaluation of one chain (M_bytes/chain, e.g. from
#' [fn_estimate_initial_M_bytes_per_chain()]) into the starting number of chunks, with the rule
#'
#' \preformatted{
#'   N_threads            = N_chains * N_threads/chain
#'   N_active_threads/L3  = min( N_threads/core * N_cores/L3 , ceil(N_threads / N_L3) )
#'   S_target             = S_L3 / N_active_threads/L3                              (gradient_type = "autodiff")
#'                        = S_L2 / max(1, N_active_threads/L3 / N_cores/L3)         (gradient_type = "manual")
#'   N_chunks             = N_threads/chain * ceil( max( ceil(M_bytes/chain / S_target), N_threads/chain ) / N_threads/chain )
#' }
#'
#' i.e. the smallest number of chunks for which one chunk of one chain fits in the share of the cache available to its thread:
#' the L3 share for autodiff (Stan) gradients, whose tape does not fit in L2 at useful chunk sizes, and the L2 of the core
#' (shared with an SMT sibling when more threads than cores are active on an L3) for manual gradients. N_chunks is rounded up
#' to a multiple of N_threads/chain, so that the within-chain threads receive equal numbers of chunks.
#'
#' The number of L3 caches, cores per L3, threads per core and the L3 and L2 sizes are read from Linux sysfs when not given;
#' any of them given as an argument replaces the value read from sysfs (on other operating systems all the quantities needed
#' must be given). Each hardware quantity given must be a single finite positive number (a whole number for the counts). If the
#' thread topology files are missing from sysfs, each hardware thread is counted as its own core and a warning is given. The
#' result is a starting value: the check during burn-in corrects it.
#'
#' @param M_bytes_per_chain Memory, in bytes, used by one full-data log-density-and-gradient evaluation of one chain, or the
#'   list returned by [fn_estimate_initial_M_bytes_per_chain()].
#' @param N_chains Number of chains run in parallel (n_chains_burnin or n_chains_sampling of NicoStan's $sample(), for the
#'   phase the chunk count is chosen for; written N_chains here to match the N_chunks and N_threads notation of the rule).
#' @param N_threads_per_chain Number of within-chain parallel threads per chain (default 1).
#' @param gradient_type "autodiff" (default; Stan / reverse-mode autodiff gradients, sized to the L3 share) or "manual"
#'   (manual gradients, sized to the L2 share).
#' @param N_L3_caches Number of L3 cache instances (default: read from sysfs).
#' @param N_cores_per_L3 Number of physical cores sharing one L3 cache (default: read from sysfs).
#' @param N_threads_per_core Number of hardware threads per core, i.e. 2 with SMT / hyper-threading (default: read from sysfs).
#' @param L3_cache_size_bytes Size of one L3 cache in bytes (default: read from sysfs).
#' @param L2_cache_size_bytes Size of the L2 cache of one core in bytes (default: read from sysfs).
#' @param sysfs_CPU_directory The sysfs CPU directory read for the quantities not given (default "/sys/devices/system/cpu").
#' @return A list with N_chunks and every intermediate quantity: N_chunks_for_one_chunk_to_fit_S_target, M_bytes_per_chain,
#'   bytes_per_chunk_at_N_chunks, gradient_type, N_chains, N_threads_per_chain, N_threads, N_L3_caches, N_cores_per_L3,
#'   N_threads_per_core, L3_cache_size_bytes, L2_cache_size_bytes, N_active_threads_per_L3, S_target_bytes,
#'   hardware_value_source (for each hardware quantity, "argument", "sysfs" or "not given (not used)") and topology_notes.
#' @examples
#' ## Stan gradient of the LC_MVP_bin_PartialLog_v5 model at N = 50,000 (19,152 bytes per individual, so
#' ## M_bytes/chain = 957.6e6) on a CPU with 12 L3 caches of 32 MiB,
#' ## 8 cores per L3 and 2 threads per core, 96 chains with one thread each (gives N_chunks = 229):
#' fn_compute_initial_N_chunks( M_bytes_per_chain    = 957.6e6,
#'                              N_chains             = 96,
#'                              N_threads_per_chain  = 1,
#'                              gradient_type        = "autodiff",
#'                              N_L3_caches          = 12,
#'                              N_cores_per_L3       = 8,
#'                              N_threads_per_core   = 2,
#'                              L3_cache_size_bytes  = 32 * 1024^2,
#'                              L2_cache_size_bytes  = 1024^2)
#'
#' \dontrun{
#' ## Topology read from sysfs, M_bytes/chain measured for the model and data of the real run:
#' M_estimate <- fn_estimate_initial_M_bytes_per_chain( model_so_file  = "model.so",
#'                                                      json_file_path = "data.json")
#' fn_compute_initial_N_chunks( M_bytes_per_chain = M_estimate,
#'                              N_chains          = 64)
#' }
#' @export
fn_compute_initial_N_chunks <-  function( M_bytes_per_chain,
                                          N_chains,
                                          N_threads_per_chain  = 1,
                                          gradient_type        = "autodiff",
                                          N_L3_caches          = NULL,
                                          N_cores_per_L3       = NULL,
                                          N_threads_per_core   = NULL,
                                          L3_cache_size_bytes  = NULL,
                                          L2_cache_size_bytes  = NULL,
                                          sysfs_CPU_directory  = "/sys/devices/system/cpu"
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
                    stop(paste0("fn_compute_initial_N_chunks: ", value_name, " must be a single positive ",
                                if (whole_number) "whole number" else "number", "; got: ", paste(as.character(value), collapse = ", ")))
              }
              invisible(TRUE)
        }
        ##
        fn_check_positive_number(value = M_bytes_per_chain,   value_name = "M_bytes_per_chain")
        fn_check_positive_number(value = N_chains,            value_name = "N_chains",            whole_number = TRUE)
        fn_check_positive_number(value = N_threads_per_chain, value_name = "N_threads_per_chain", whole_number = TRUE)
        ##
        if (!is.character(gradient_type) || length(x = gradient_type) != 1 || !(gradient_type %in% c("autodiff", "manual"))) {
              stop(paste0("fn_compute_initial_N_chunks: gradient_type must be \"autodiff\" (Stan gradients) or \"manual\"; got: ",
                          paste(as.character(gradient_type), collapse = ", ")))
        }
        ##
        ## ---- hardware quantities: arguments first, then sysfs for any that are missing (the cache size not used by
        ##      gradient_type is not needed):
        ##
        hardware_values <-  list( N_L3_caches          = N_L3_caches,
                                  N_cores_per_L3       = N_cores_per_L3,
                                  N_threads_per_core   = N_threads_per_core,
                                  L3_cache_size_bytes  = L3_cache_size_bytes,
                                  L2_cache_size_bytes  = L2_cache_size_bytes)
        ##
        hardware_value_names_needed <-  c("N_L3_caches", "N_cores_per_L3", "N_threads_per_core",
                                          if (gradient_type == "autodiff") "L3_cache_size_bytes" else "L2_cache_size_bytes")
        ##
        ## (each hardware quantity given as an argument is checked before anything is read from sysfs):
        for (hardware_value_name in names(x = hardware_values)) {
              if (!is.null(x = hardware_values[[hardware_value_name]])) {
                    fn_check_positive_number(value        = hardware_values[[hardware_value_name]],
                                             value_name   = hardware_value_name,
                                             whole_number = !grepl(pattern = "_bytes$", x = hardware_value_name))
              }
        }
        ##
        hardware_value_is_missing <-  sapply(X = hardware_values, FUN = is.null)
        hardware_value_source <-  ifelse(test = hardware_value_is_missing, yes = "sysfs", no = "argument")
        topology_notes <-  character(0)
        ##
        if (any(hardware_value_is_missing[hardware_value_names_needed])) {

              CPU_cache_topology <-  fn_read_CPU_cache_topology_from_sysfs(sysfs_CPU_directory = sysfs_CPU_directory)
              topology_notes <-  CPU_cache_topology$topology_notes
              ##
              for (hardware_value_name in names(x = hardware_values)[hardware_value_is_missing]) {
                    hardware_values[[hardware_value_name]] <-  CPU_cache_topology[[hardware_value_name]]
              }
              ##
              thread_topology_values_from_sysfs <-  intersect(x = c("N_threads_per_core", "N_cores_per_L3"),
                                                              y = names(x = hardware_values)[hardware_value_is_missing])
              ##
              if (!isTRUE(CPU_cache_topology$thread_topology_available) && length(x = thread_topology_values_from_sysfs) > 0) {
                    warning(paste0("fn_compute_initial_N_chunks: the thread topology files (thread_siblings_list, core_cpus_list) were not ",
                                   "found under ", sysfs_CPU_directory, ", so each hardware thread was counted as its own core (",
                                   paste(thread_topology_values_from_sysfs, collapse = " and "), " read as if the CPU had no SMT). ",
                                   "Give N_threads_per_core and N_cores_per_L3 as arguments if the CPU has SMT."))
              }

        } else {

              ## (every quantity the rule needs was given; the cache size it does not use stays unknown):
              for (hardware_value_name in names(x = hardware_values)[hardware_value_is_missing]) {
                    hardware_values[[hardware_value_name]] <-  NA_real_
                    hardware_value_source[hardware_value_name] <-  "not given (not used)"
              }

        }
        ##
        for (hardware_value_name in hardware_value_names_needed) {
              if (is.na(x = hardware_values[[hardware_value_name]])) {
                    stop(paste0("fn_compute_initial_N_chunks: ", hardware_value_name, " could not be read from sysfs (",
                                sysfs_CPU_directory, if (length(x = topology_notes) > 0) paste0(": ", paste(topology_notes, collapse = "; ")) else "",
                                "); supply it as an argument."))
              }
              fn_check_positive_number(value        = hardware_values[[hardware_value_name]],
                                       value_name   = hardware_value_name,
                                       whole_number = !grepl(pattern = "_bytes$", x = hardware_value_name))
        }
        ##
        ## ---- the rule:
        ##
        N_threads <-  N_chains * N_threads_per_chain
        ##
        N_hardware_threads_per_L3 <-  hardware_values$N_threads_per_core * hardware_values$N_cores_per_L3
        ##
        N_active_threads_per_L3 <-  min(N_hardware_threads_per_L3,
                                        ceiling(N_threads / hardware_values$N_L3_caches))
        ##
        if (gradient_type == "autodiff") {
              S_target_bytes <-  hardware_values$L3_cache_size_bytes / N_active_threads_per_L3
        } else {
              S_target_bytes <-  hardware_values$L2_cache_size_bytes / max(1, N_active_threads_per_L3 / hardware_values$N_cores_per_L3)
        }
        ##
        N_chunks_for_one_chunk_to_fit_S_target <-  ceiling(M_bytes_per_chain / S_target_bytes)
        ##
        N_chunks <-  N_threads_per_chain * ceiling(max(N_chunks_for_one_chunk_to_fit_S_target, N_threads_per_chain) / N_threads_per_chain)
        ##
        N_hardware_threads <-  hardware_values$N_L3_caches * N_hardware_threads_per_L3
        ##
        if (N_threads > N_hardware_threads) {
              warning(paste0("fn_compute_initial_N_chunks: N_threads = ", N_threads, " (", N_chains, " chains x ", N_threads_per_chain,
                             " thread(s) per chain) is more than the ", N_hardware_threads, " hardware threads (", hardware_values$N_L3_caches,
                             " L3 x ", hardware_values$N_cores_per_L3, " cores x ", hardware_values$N_threads_per_core, " threads per core)."))
        }
        ##
        ## ---- one-line summary:
        ##
        S_target_description <-  if (gradient_type == "autodiff") {
              paste0("S_L3 ", formatC(x = hardware_values$L3_cache_size_bytes / 1024^2, format = "f", digits = 2), " MiB / ",
                     N_active_threads_per_L3, " active threads per L3")
        } else {
              paste0("S_L2 ", formatC(x = hardware_values$L2_cache_size_bytes / 1024, format = "f", digits = 0), " KiB / max(1, ",
                     N_active_threads_per_L3, " active threads per L3 / ", hardware_values$N_cores_per_L3, " cores per L3)")
        }
        ##
        message(colourise(paste0("initial N_chunks = ", N_chunks,
                                 " (", gradient_type, " gradient; M_bytes/chain = ", formatC(x = M_bytes_per_chain / 1e6, format = "f", digits = 1),
                                 " MB; N_threads = ", N_threads, " = ", N_chains, " chains x ", N_threads_per_chain,
                                 " thread(s) per chain; S_target = ", formatC(x = S_target_bytes / 1024^2, format = "f", digits = 3),
                                 " MiB = ", S_target_description, "; ceil(M_bytes/chain / S_target) = ",
                                 N_chunks_for_one_chunk_to_fit_S_target, ")"), "cyan"))
        ##
        if (length(x = topology_notes) > 0) {
              message(colourise(paste0("CPU cache topology notes: ", paste(topology_notes, collapse = "; ")), "cyan"))
        }
        ##
        return(list( N_chunks                                = N_chunks,
                     N_chunks_for_one_chunk_to_fit_S_target  = N_chunks_for_one_chunk_to_fit_S_target,
                     M_bytes_per_chain                       = M_bytes_per_chain,
                     bytes_per_chunk_at_N_chunks             = M_bytes_per_chain / N_chunks,
                     gradient_type                           = gradient_type,
                     N_chains                                = N_chains,
                     N_threads_per_chain                     = N_threads_per_chain,
                     N_threads                               = N_threads,
                     N_L3_caches                             = hardware_values$N_L3_caches,
                     N_cores_per_L3                          = hardware_values$N_cores_per_L3,
                     N_threads_per_core                      = hardware_values$N_threads_per_core,
                     L3_cache_size_bytes                     = hardware_values$L3_cache_size_bytes,
                     L2_cache_size_bytes                     = hardware_values$L2_cache_size_bytes,
                     N_active_threads_per_L3                 = N_active_threads_per_L3,
                     S_target_bytes                          = S_target_bytes,
                     hardware_value_source                   = hardware_value_source,
                     topology_notes                          = topology_notes))

}























