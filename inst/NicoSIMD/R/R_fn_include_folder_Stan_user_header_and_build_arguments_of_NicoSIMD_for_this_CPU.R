##
## ==== R_fn_include_folder_Stan_user_header_and_build_arguments_of_NicoSIMD_for_this_CPU.R ======================
##
## ---- What NicoStan (or BayesMVP, or any user) needs to build a Stan model with NicoSIMD's nico_ functions: the
##      include folder, the user header (stan_external_functions_of_NicoSIMD.hpp), the compiler flags of this CPU
##      and the make and stanc arguments -
##
## Compiler flags (set explicitly; the same rule as NicoStan's fn_compiler_flags_of_every_build_on_this_CPU()):
##   - on a CPU with AVX-512 (avx512f, avx512vl and avx512dq): -O3 -march=native -mtune=native -mfma -mavx2
##     -mavx512f -mavx512vl -mavx512dq, for AVX-512 and AVX2 alike; the AVX2 kernels are chosen with
##     CPPFLAGS_OPTIM=-DBAYESMVP_FORCE_AVX2, every compiler flag unchanged;
##   - on a CPU with AVX2 only: -O3 -march=native -mtune=native -mfma -mavx2;
##   - on a 64-bit ARM CPU (e.g. Apple Silicon, Snapdragon, AWS Graviton; every one has NEON): -O3 -mcpu=native,
##     and the AVX2 kernels are compiled through SIMDe, with NEON instructions (nico_simd_intrinsics.hpp);
##   - on any other CPU (e.g. an x86-64 CPU without AVX2): instruction_set = "Stan_math", with no user header and
##     no extra flags, i.e. the model is built on plain Stan math (and Eigen), as NicoSIMD has no kernels for it.
##
## The instruction sets of an x86-64 CPU are read from /proc/cpuinfo on Linux; on any other system (e.g. Windows,
## macOS), from the C++ compiler R uses, i.e. which of __AVX2__, __FMA__ and __AVX512F__ / __AVX512VL__ /
## __AVX512DQ__ it defines with -march=native (the flag every model is compiled with).
##
##
#' The include folder, user header and build arguments of NicoSIMD on this CPU
#'
#' @param instruction_set NULL (the default: AVX-512 where this CPU has it, otherwise AVX2; on a 64-bit ARM CPU,
#'   NEON through SIMDe), "AVX512", "AVX2" (the AVX2 kernels, also on a CPU with AVX-512) or
#'   "NEON_through_SIMDe" (the AVX2 kernels through SIMDe, on a 64-bit ARM CPU only); on a CPU with none of these,
#'   "Stan_math" (plain Stan math) is returned whatever is asked.
#' @return A list: include_folder (the folder holding NicoSIMD/), Stan_user_header (the path of
#'   stan_external_functions_of_NicoSIMD.hpp; both NULL for "Stan_math"), instruction_set ("AVX512", "AVX2",
#'   "NEON_through_SIMDe" or "Stan_math"),
#'   compiler_flags (one string),
#'   make_args (for BridgeStan / NicoStan: STAN_THREADS, CXXFLAGS_OPTIM with the compiler flags and, for AVX2 on a
#'   CPU with AVX-512, CPPFLAGS_OPTIM=-DBAYESMVP_FORCE_AVX2) and stanc_args ("--allow-undefined").
#' @export
fn_include_folder_Stan_user_header_and_build_arguments_of_NicoSIMD_for_this_CPU <-  function(
                                                                                    instruction_set  = NULL) {

        if (!is.null(instruction_set)) {
              stopifnot(instruction_set %in% c("AVX512", "AVX2", "NEON_through_SIMDe"))
        }
        ##
        ## ---- a 64-bit ARM CPU (every one has NEON; Linux reports "aarch64", macOS "arm64"):
        ##
        CPU_is_64_bit_ARM <-  tolower(Sys.info()[["machine"]]) %in% c("aarch64", "arm64")
        if (CPU_is_64_bit_ARM) {
              if (!is.null(instruction_set) && (instruction_set != "NEON_through_SIMDe")) {
                    stop(paste0("This is a 64-bit ARM CPU (no AVX): only ",
                                "instruction_set = \"NEON_through_SIMDe\" can be built here."))
              }
              compiler_flags <-  "-O3 -mcpu=native"
              include_folder <-  system.file("include", package = "NicoSIMD")
              Stan_user_header <-  file.path(include_folder, "NicoSIMD",
                                             "stan_external_functions_of_NicoSIMD.hpp")
              if (!nzchar(include_folder) || !file.exists(Stan_user_header)) {
                    stop("NicoSIMD's headers are not where they should be: reinstall NicoSIMD.")
              }
              return(list( include_folder    = include_folder,
                           Stan_user_header  = Stan_user_header,
                           instruction_set   = "NEON_through_SIMDe",
                           compiler_flags    = compiler_flags,
                           make_args         = c("STAN_THREADS=true", "PRECOMPILED_HEADERS=false",
                                                 paste0("CXXFLAGS_OPTIM=", shQuote(compiler_flags))),
                           stanc_args        = "--allow-undefined"))
        }
        if (identical(instruction_set, "NEON_through_SIMDe")) {
              stop("NEON through SIMDe is only for a 64-bit ARM CPU; this CPU is not one.")
        }
        ##
        ## ---- the instruction sets of this CPU:
        ##
        CPU_flags <-  if (file.exists("/proc/cpuinfo")) readLines("/proc/cpuinfo", warn = FALSE) else character(0)
        CPU_flags <-  CPU_flags[grepl("^flags", CPU_flags)][1]
        if (!is.na(CPU_flags)) {
              CPU_has_AVX2 <-  grepl("\\bavx2\\b", CPU_flags) && grepl("\\bfma\\b", CPU_flags)
              CPU_has_AVX512 <-  grepl("\\bavx512f\\b", CPU_flags) && grepl("\\bavx512vl\\b", CPU_flags) &&
                                 grepl("\\bavx512dq\\b", CPU_flags)
        } else {
              ##
              ## (no /proc/cpuinfo, e.g. Windows or macOS: the macros the compiler defines with -march=native)
              ##
              compiler <-  trimws(suppressWarnings(tools::Rcmd(c("config", "CXX17"), stdout = TRUE,
                                                                stderr = FALSE)))[1]
              compiler_words <-  strsplit(compiler, "\\s+")[[1]]
              empty_file <-  tempfile(fileext = ".cpp")
              writeLines("", empty_file)
              compiler_macros <-  suppressWarnings(system2(compiler_words[1],
                                                           c(compiler_words[-1], "-march=native", "-dM", "-E",
                                                             shQuote(empty_file)),
                                                           stdout = TRUE, stderr = FALSE))
              CPU_has_AVX2 <-  any(grepl("__AVX2__", compiler_macros)) && any(grepl("__FMA__", compiler_macros))
              CPU_has_AVX512 <-  any(grepl("__AVX512F__", compiler_macros)) &&
                                 any(grepl("__AVX512VL__", compiler_macros)) &&
                                 any(grepl("__AVX512DQ__", compiler_macros))
        }
        if (!CPU_has_AVX2) {
              message(paste0("\033[36mThis CPU has no AVX2 (and it is not a 64-bit ARM CPU): NicoSIMD has no ",
                             "kernels for it, so the model is built on plain Stan math.\033[0m"))
              return(list( include_folder    = NULL,
                           Stan_user_header  = NULL,
                           instruction_set   = "Stan_math",
                           compiler_flags    = "-O3 -march=native -mtune=native",
                           make_args         = c("STAN_THREADS=true", "PRECOMPILED_HEADERS=false",
                                                 paste0("CXXFLAGS_OPTIM=",
                                                        shQuote("-O3 -march=native -mtune=native"))),
                           stanc_args        = NULL))
        }
        if (is.null(instruction_set)) instruction_set <-  if (CPU_has_AVX512) "AVX512" else "AVX2"
        if ((instruction_set == "AVX512") && !CPU_has_AVX512) {
              stop("This CPU has no AVX-512: the AVX-512 version cannot be built here.")
        }
        ##
        ## ---- the compiler flags (those of this CPU, whichever kernels are chosen):
        ##
        compiler_flags <-  if (CPU_has_AVX512) {
              paste0("-O3 -march=native -mtune=native -mfma -mavx2 ", "-mavx512f -mavx512vl -mavx512dq")
        } else {
              "-O3 -march=native -mtune=native -mfma -mavx2"
        }
        make_args <-  c("STAN_THREADS=true", "PRECOMPILED_HEADERS=false",
                        paste0("CXXFLAGS_OPTIM=", shQuote(compiler_flags)))
        if ((instruction_set == "AVX2") && CPU_has_AVX512) {
              make_args <-  c(make_args, "CPPFLAGS_OPTIM=-DBAYESMVP_FORCE_AVX2")
        }
        ##
        ## ---- the include folder and the user header:
        ##
        include_folder <-  system.file("include", package = "NicoSIMD")
        Stan_user_header <-  file.path(include_folder, "NicoSIMD", "stan_external_functions_of_NicoSIMD.hpp")
        if (!nzchar(include_folder) || !file.exists(Stan_user_header)) {
              stop("NicoSIMD's headers are not where they should be: reinstall NicoSIMD.")
        }
        ##
        return(list( include_folder    = include_folder,
                     Stan_user_header  = Stan_user_header,
                     instruction_set   = instruction_set,
                     compiler_flags    = compiler_flags,
                     make_args         = make_args,
                     stanc_args        = "--allow-undefined"))

}





















