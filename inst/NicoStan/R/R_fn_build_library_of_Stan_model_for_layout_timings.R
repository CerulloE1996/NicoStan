##
## ==== R_fn_build_library_of_Stan_model_for_layout_timings.R ====================================================
##
## ---- Builds ONE library of a Stan model for the layout timings (Stan math, or the hand-written AVX version
##      with BayesMVP's AVX2 or AVX-512 functions), through NicoStan's own initialise_model(), so that the
##      timings never compile -
##
## The library is built with initialise_model() (the BridgeStan library with NicoStan's page-release runtime and
## its fingerprint file, which NicoStan's sampler needs: a library without them is rebuilt when the sampler loads
## it), from a plain copy of its .stan file with a fresh date (so that make never reuses a library older than its
## .stan file), with a base name of its own, in libraries_folder. A library already there is moved aside (never
## deleted).
##
## Compiler flags (set explicitly in every build, for every math backend, Stan math included):
##   - on a CPU with AVX-512 (the local-HPC): -O3 -march=native -mtune=native -mfma -mavx2 -mavx512f -mavx512vl
##     -mavx512dq, for Stan math, AVX2 and AVX-512 alike; the AVX2 version differs only by BayesMVP's AVX2
##     functions (CPPFLAGS_OPTIM=-DBAYESMVP_FORCE_AVX2);
##   - on a CPU without AVX-512 (the laptop): -O3 -march=native -mtune=native -mfma -mavx2, for every backend.
##
## The arguments of the build (.stan file, user header, make arguments, stanc arguments) are saved next to the
## library (<library_name>_arguments_of_build.rds), so that the timings call initialise_model() with the same
## arguments and the same library is loaded, never rebuilt.
##
##
#' The compiler flags of every build on this CPU: AVX-512 flags where the CPU has AVX-512, AVX2 flags where it has
#' AVX2, and no x86 instruction-set flags on ARM (where NicoSIMD's AVX2 functions run as NEON through SIMDe)
#'
#' The instruction sets come from R_fn_BayesMVP_AVX_instruction_sets_supported_by_this_CPU (the compiler R uses,
#' so Linux, macOS and Windows, x86-64 and ARM).
#' @return A list: instruction_set ("AVX512", "AVX2" or "NEON") and compiler_flags (one string).
#' @noRd
fn_compiler_flags_of_every_build_on_this_CPU <-  function() {

        instruction_sets_supported <-  R_fn_BayesMVP_AVX_instruction_sets_supported_by_this_CPU()
        on_ARM <-  grepl(pattern = "arm|aarch64", x = Sys.info()[["machine"]], ignore.case = TRUE)
        ##
        if (on_ARM) {
              return(list(instruction_set = "NEON",
                          compiler_flags  = "-O3 -march=native -mtune=native"))
        }
        if (instruction_sets_supported[["BayesMVP_AVX512"]]) {
              return(list(instruction_set = "AVX512",
                          compiler_flags  = paste0("-O3 -march=native -mtune=native -mfma -mavx2 ",
                                                   "-mavx512f -mavx512vl -mavx512dq")))
        }
        return(list(instruction_set = "AVX2",
                    compiler_flags  = "-O3 -march=native -mtune=native -mfma -mavx2"))

}

##
##
#' Builds one library of a Stan model for the layout timings, through initialise_model()
#'
#' @param source_Stan_model_file_path The .stan file: the model itself (math_backend = "Stan") or its hand-written
#'   AVX version (<model>_hand_written_BayesMVP_AVX.stan; math_backend = "AVX2" or "AVX512").
#' @param library_name The base name of the library (distinct for every model version and math backend).
#' @param math_backend "Stan", "AVX2" or "AVX512".
#' @param libraries_folder The folder of the libraries.
#' @param Stan_data_list A data list of the model (any N; used only by initialise_model()).
#' @param BayesMVP_include_folder Recorded in the arguments of the build only. BayesMVP holds no AVX functions
#'   of its own any more: the AVX versions (written with the fast_ names) use NicoSIMD's
#'   stan_external_functions_with_the_fast_names_of_BayesMVP.hpp, from the installed NicoSIMD.
#' @return The arguments of the build (invisibly), with the seconds of the build.
#' @noRd
fn_build_library_of_Stan_model_for_layout_timings <-  function( source_Stan_model_file_path,
                                                                library_name,
                                                                math_backend,
                                                                libraries_folder,
                                                                Stan_data_list,
                                                                BayesMVP_include_folder  = system.file(
                                                                      "include", package = "BayesMVP")) {

        ##
        stopifnot(math_backend %in% c("Stan", "AVX2", "AVX512"))
        stopifnot(file.exists(source_Stan_model_file_path))
        dir.create(libraries_folder, showWarnings = FALSE, recursive = TRUE)
        compiler_flags_of_this_CPU <-  fn_compiler_flags_of_every_build_on_this_CPU()
        if ((math_backend == "AVX512") && (compiler_flags_of_this_CPU$instruction_set != "AVX512")) {
              stop("This CPU has no AVX-512: the AVX-512 version cannot be built here.")
        }
        ##
        Stan_model_file_path <-  file.path(libraries_folder, paste0(library_name, ".stan"))
        so_file <-  sub("\\.stan$", "_model.so", Stan_model_file_path)
        ##
        ## ---- a library already there is moved aside (never deleted):
        ##
        if (file.exists(so_file)) {
              folder_of_old_libraries <-  file.path(libraries_folder, "old_libraries_moved_aside")
              dir.create(folder_of_old_libraries, showWarnings = FALSE)
              file.rename(so_file, file.path(folder_of_old_libraries,
                                             paste0(basename(so_file), ".",
                                                    format(Sys.time(), "%Y_%m_%d_%H%M%S"))))
        }
        ##
        ## ---- a plain copy of the .stan file, with a fresh date:
        ##
        stopifnot(file.copy(source_Stan_model_file_path, Stan_model_file_path, overwrite = TRUE,
                            copy.date = FALSE))
        Sys.setFileTime(Stan_model_file_path, Sys.time())
        ##
        ## ---- the arguments of the build (the compiler flags of this CPU for every math backend):
        ##
        make_args <-  c("STAN_THREADS=true", "PRECOMPILED_HEADERS=false",
                        paste0("CXXFLAGS_OPTIM=", shQuote(compiler_flags_of_this_CPU$compiler_flags)))
        Stan_cpp_user_header <-  NULL
        stanc_args <-  NULL
        if (math_backend != "Stan") {
              # Stan_cpp_user_header <-  system.file("include", "BayesMVP", "stan_external_functions.hpp",
              #                                      package = "BayesMVP")
              # Stan_cpp_user_header <-  file.path(BayesMVP_include_folder, "BayesMVP",
              #                                    "stan_external_functions.hpp")
              ##
              ## (NicoSIMD holds every AVX2 / AVX-512 function; the AVX versions declare the fast_ names, which
              ##  NicoSIMD's header below gives to its nico_ functions)
              Stan_cpp_user_header <-  system.file("include", "NicoSIMD",
                                                   "stan_external_functions_with_the_fast_names_of_BayesMVP.hpp",
                                                   package = "NicoSIMD")
              stopifnot(nzchar(Stan_cpp_user_header), file.exists(Stan_cpp_user_header))
              stanc_args <-  "--allow-undefined"
        }
        if (math_backend == "AVX2") {
              make_args <-  c(make_args, "CPPFLAGS_OPTIM=-DBAYESMVP_FORCE_AVX2")
        }
        arguments_of_build <-  list( Stan_model_file_path         = Stan_model_file_path,
                                     Stan_cpp_user_header         = Stan_cpp_user_header,
                                     make_args                    = make_args,
                                     stanc_args                   = stanc_args,
                                     math_backend                 = math_backend,
                                     compiler_flags               = compiler_flags_of_this_CPU$compiler_flags,
                                     source_Stan_model_file_path  = source_Stan_model_file_path,
                                     BayesMVP_include_folder      = BayesMVP_include_folder)
        ##
        ## ---- the build, through initialise_model():
        ##
        build_seconds <-  system.time(
              init_object <-  initialise_model( Model_type            = "Stan",
                                                sample_nuisance       = TRUE,
                                                Stan_data_list        = Stan_data_list,
                                                Stan_model_file_path  = Stan_model_file_path,
                                                Stan_cpp_user_header  = Stan_cpp_user_header,
                                                stanc_args            = stanc_args,
                                                make_args             = make_args,
                                                compile               = TRUE)
        )[["elapsed"]]
        if (!file.exists(so_file)) stop(paste0("The library ", so_file, " was not built."))
        arguments_of_build$build_seconds <-  build_seconds
        saveRDS(arguments_of_build, file.path(libraries_folder, paste0(library_name, "_arguments_of_build.rds")))
        message(paste0("\033[32mBuilt ", so_file, " in ", round(build_seconds), " s (", math_backend, "; ",
                       compiler_flags_of_this_CPU$compiler_flags, ")\033[0m"))
        ##
        return(invisible(arguments_of_build))
##
}






















