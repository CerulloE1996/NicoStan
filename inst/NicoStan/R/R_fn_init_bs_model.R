
  
  
#' get_BayesMVP_Stan_paths
#' @export
get_BayesMVP_Stan_paths <- function() {
  
      Sys.setenv(STAN_THREADS = "true")
      
      # Get package directory paths
      pkg_dir <- system.file(package = "NicoStan")
      data_dir <- file.path(pkg_dir, "stan_data")  # directory to store data inc. JSON data files
      stan_dir <- file.path(pkg_dir, "stan_models")
      ##
      # print(paste("pkg_dir = ", pkg_dir))
      # print(paste("data_dir = ", data_dir))
      # print(paste("stan_dir = ", stan_dir))
      
      return(list(pkg_dir = pkg_dir,
                  data_dir = data_dir,
                  stan_dir = stan_dir))
  
}












#' Create a BridgeStan model while preserving one TBB runtime
#' @keywords internal
R_fn_create_BridgeStan_model <- function( lib,
                                          data,
                                          seed = 123,
                                          stanc_args = NULL,
                                          make_args = NULL
) {

        package_name <-  get0( x = ".nicostan_provider_package",
                               envir = environment(),
                               mode = "character",
                               inherits = TRUE,
                               ifnotfound = "NicoStan")
        ##
        R_fn_stop_if_several_TBB_copies_loaded(package_name = package_name)
        ##
        bs_model <-  tryCatch(
              bridgestan::StanModel$new( lib = lib,
                                         data = data,
                                         seed = seed,
                                         stanc_args = stanc_args,
                                         make_args = make_args),
              error = function(condition) {

                    R_fn_stop_if_several_TBB_copies_loaded(package_name = package_name)
                    stop(condition)

              })
        ##
        R_fn_stop_if_several_TBB_copies_loaded(package_name = package_name)
        ##
        return(bs_model)

}




#' init_bs_model_external
#' @export
init_bs_model_external <- function( stream = NULL,
                                    Stan_data_list,
                                    Stan_model_file_path,
                                    stanc_args = NULL,
                                    make_args = NULL
) {
  
        outs <- get_BayesMVP_Stan_paths()
        pkg_dir <- outs$pkg_dir
        data_dir <- outs$data_dir
        stan_dir <- outs$stan_dir
        ##
        if (length(Stan_data_list) == 0) {
          
              ## ---- a model with NO data: cmdstanr::write_stan_json() would write an
              ## empty ARRAY ([]), but BridgeStan requires an empty OBJECT ({}):
              json_file_path <- file.path( data_dir,
                                           paste0( "data_empty_",
                                                   if (is.null(stream)) 0 else stream,
                                                   ".json"))
              ##
              ## (written atomically - temporary file + rename - since the shared stan_data folder is read by
              ##  concurrent fits; see fn_write_file_atomically):
              fn_write_stan_json_atomically( stan_data_list = list(),
                                             json_file_path = json_file_path)
              ##
              validated_json_string <- "{}"
              
        } else {
          
              json_file_path <- convert_stan_data_list_to_JSON( stan_data_list = Stan_data_list,
                                                                pkg_data_dir = data_dir)
              if (!(is.null(stream))) {
                json_file_path <- copy_json_with_worker_id( original_path  = json_file_path, 
                                                            ii = stream)
              } 
              
              # Read the raw content
              raw_content <- paste(readLines(json_file_path), collapse = "")
              
              ##
              ## ---- Proper JSON object (the normal case): no parse / rewrite round trip.
              ##      convert_stan_data_list_to_JSON() writes the data with cmdstanr::write_stan_json(), i.e. a JSON
              ##      object starting with "{", which BridgeStan and CmdStan read natively. The fromJSON / unwrap /
              ##      write_json / re-read path below is only needed for LEGACY double-encoded files (a JSON string
              ##      wrapped in an array). For an N = 2500 data set that round trip took ~3.2 s per call, and one fit
              ##      can reach this function more than once (initialise_model() runs when the model object is created
              ##      with data, again in R_fn_sample_model, and once more there after a non-identity test re-ordering).
              ##      A JSON object is therefore only checked with jsonlite::validate() (a C-level syntax check that
              ##      builds no R object) and passed on unchanged; the file on disk is not rewritten:
              ##
              ## (only LEADING white space is trimmed for the "{" check: a two-sided trimws() scans the whole
              ##  multi-MB string, ~0.5 s for an N = 50000 data set):
              raw_content_is_JSON_object <-  startsWith( x      = trimws(x = raw_content, which = "left"),
                                                         prefix = "{")
              ##
              if (raw_content_is_JSON_object) {

                    raw_content_JSON_check <-  jsonlite::validate(raw_content)
                    ##
                    if (!isTRUE(raw_content_JSON_check)) {
                          stop(paste0("init_bs_model_external: invalid JSON in file ", json_file_path, ": ",
                                      attr(x = raw_content_JSON_check, which = "err")))
                    }
                    ##
                    validated_json_string <-  raw_content

              } else {

              ## ---- LEGACY path (JSON not starting with "{", e.g. a double-encoded file): parse, unwrap if
              ##      double-encoded, rewrite the file atomically and re-read it:
              # Try to parse it
              parsed <- tryCatch({
                jsonlite::fromJSON(raw_content, simplifyVector = FALSE)
              }, error = function(e) {
                stop("Invalid JSON in file: ", e$message)
              })
              
              # Check if it's wrapped in an array containing a string
              if (is.character(parsed) && length(parsed) == 1) {
                # It's double-encoded, extract the inner JSON
                json_data <- jsonlite::fromJSON(parsed[1], simplifyVector = FALSE)
              } else {
                # It's already proper JSON data
                json_data <- parsed
              }
              
              # # Validate it has expected structure
              # if (!all(c("N", "y") %in% names(json_data))) {
              #   warning("JSON data missing required fields N and y")
              # }
              
              # Preserve singleton/empty array ranks when unwrapping legacy JSON.
              # Rewritten ATOMICALLY (temporary file + rename): this file is shared by concurrent fits with the same
              # data, and an in-place rewrite let another process read it half-written ("premature EOF"):
              fn_write_file_atomically( target_file_path                = json_file_path,
                                        fn_write_to_temporary_file_path = function(temporary_file_path) {
                                              jsonlite::write_json( x = json_data, path = temporary_file_path, 
                                                                    auto_unbox = TRUE, digits = NA, pretty = FALSE)
                                        })
              
              # Read the validated string for bridgestan
              validated_json_string <- paste(readLines(json_file_path), collapse = "")

              } ## end of "else" (LEGACY path) of the "if (raw_content_is_JSON_object)"
              
        }
        ##
        # if (!(is.null(stream))) {
        #   Stan_model_file_path <- copy_.stan_with_worker_id(original_path  = Stan_model_file_path, 
        #                                                     ii = stream)
        # } 
        ##
        ## stanc_args  -> arguments for the STAN compiler (e.g. list("--O1", "--include-paths=..."))
        ## make_args   -> C++/make options for compiling the generated C++ (e.g.
        ##                list("STAN_THREADS=true") for reduce_sum models). These are
        ##                distinct: stanc_args never affect the C++ compilation flags.
        ##
        ## ---- STAN_THREADS: get_BayesMVP_Stan_paths() exports STAN_THREADS=true to make, so
        ## the model is built with a thread-local autodiff stack (safe for parallel chains).
        ## A model built WITHOUT it has ONE global autodiff stack: concurrent
        ## log_density_gradient calls from parallel chains race on
        ## start_nested()/recover_memory_nested() and abort the session
        ## ("empty_nested() must be false..."). make skips .so files it considers
        ## up-to-date, so a stale non-threads .so would never be rebuilt -- detect
        ## it by its embedded flag and delete it to force a fresh, threads build:
        page_release_linux <- identical(x = Sys.info()[["sysname"]], y = "Linux")
        if (page_release_linux) {
              memory_compile_options <- R_fn_bridge_memory_compile_options(
                    model_file = Stan_model_file_path,
                    make_args = make_args,
                    stanc_args = stanc_args,
                    release_header = NULL)
              original_bridgestan_path <- getFromNamespace(
                    "get_bridgestan_path", "bridgestan")(download = FALSE)
              page_release_runtime <- R_fn_bridge_memory_prepare_runtime(
                    release_header = NULL,
                    original_bridgestan_path = original_bridgestan_path)
              bridgestan::set_bridgestan_path(page_release_runtime$bridgestan_path)
              on.exit(bridgestan::set_bridgestan_path(original_bridgestan_path), add = TRUE)
              stanc_args <- memory_compile_options$stanc_args
              make_args <- memory_compile_options$make_args
              page_release_fingerprint_file <- paste0(
                    tools::file_path_sans_ext(Stan_model_file_path),
                    "_page_release_runtime.txt")
              page_release_fingerprint <- paste0(
                    "runtime_key=", page_release_runtime$runtime_key,
                    "; release_header_md5=", memory_compile_options$release_header_md5)
        }
        model_so_file <- normalizePath(transform_stan_path(Stan_model_file_path))
        ##
        if (file.exists(model_so_file)) {
              raw_so <- readBin( con = model_so_file,
                                 what = "raw",
                                 n = file.info(model_so_file)$size)
              ##
              if (length(grepRaw( pattern = "STAN_THREADS=false",
                                  x = raw_so,
                                  fixed = TRUE)) > 0) {
                    unlink(x = c(model_so_file, paste0(tools::file_path_sans_ext(Stan_model_file_path), ".o")))
              }
        }
        if (page_release_linux && file.exists(model_so_file) &&
            (!R_fn_bridge_memory_has_export(model_so_file) ||
             !file.exists(page_release_fingerprint_file) ||
             !identical(x = readLines(con = page_release_fingerprint_file, warn = FALSE),
                        y = page_release_fingerprint))) {
              unlink(x = c(model_so_file, paste0(tools::file_path_sans_ext(Stan_model_file_path), ".o")))
        }
        ##
        bs_model <- R_fn_create_BridgeStan_model( lib = Stan_model_file_path,
                                                  data = validated_json_string,
                                                  seed = 123,
                                                  stanc_args = stanc_args,
                                                  make_args = make_args)
        ## the fingerprint is written only when it differs, and through a temporary file renamed over
        ## it: several processes start fits of the same model at once, and rewriting it in place (writeLines()
        ## empties the file first) let another process read it empty, take the model as stale, delete its .so and
        ## rebuild it whilst other fits were loading it
        # if (page_release_linux) {
        #       writeLines(text = page_release_fingerprint,
        #                  con = page_release_fingerprint_file)
        # }
        if (page_release_linux &&
            !(file.exists(page_release_fingerprint_file) &&
              identical(x = readLines(con = page_release_fingerprint_file, warn = FALSE),
                        y = page_release_fingerprint))) {
              temporary_fingerprint_file <- tempfile(pattern = "page_release_fingerprint_",
                                                     tmpdir = dirname(page_release_fingerprint_file),
                                                     fileext = ".txt.tmp")
              writeLines(text = page_release_fingerprint,
                         con = temporary_fingerprint_file)
              stopifnot(file.rename(from = temporary_fingerprint_file, to = page_release_fingerprint_file))
        }
        
        json_file_path <- normalizePath(json_file_path)
        ##
        Stan_model_file_path <- normalizePath(Stan_model_file_path)
        model_so_file <- normalizePath(transform_stan_path(Stan_model_file_path))
        
        return(list(bs_model = bs_model, 
                    json_file_path = json_file_path, 
                    model_so_file = model_so_file,
                    Stan_model_file_path = Stan_model_file_path))
  
}






















