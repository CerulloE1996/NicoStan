##
## ==== R_fn_Stan_model_math_functions_BayesMVP_AVX.R ===========================================================
##
## ---- The math functions of a Stan model (Stan_model_math_functions of MVP_model$new() and initialise_model()):
##
##      Stan's own functions (the default), or BayesMVP's AVX2 / AVX-512 functions ----
##
## With Stan_model_math_functions = "BayesMVP_AVX2", "BayesMVP_AVX512" or
## "BayesMVP_AVX512_if_supported_else_AVX2", NicoStan:
##
##   1. writes the AVX version of the Stan model next to it (<model>_BayesMVP_AVX512.stan or
##      <model>_BayesMVP_AVX2.stan, as BridgeStan writes the compiled model next to it): every call of a Stan
##      function which BayesMVP replaces becomes BayesMVP's function of the same name with "fast_" in front,
##      declared in the functions block for the argument types it supports. stanc decides: a call which stanc
##      rejects (other argument types) gets Stan's own function back, one at a time, until the model compiles.
##      Comments and strings are left as they are. The generated quantity simd_lanes (the SIMD lanes of the
##      build) is added;
##   2. writes the user header of the AVX version next to it (<model>_BayesMVP_AVX512_user_header.hpp), which
##      includes BayesMVP's stan_external_functions.hpp, the header of each fused log density the AVX version
##      uses, and the user's own header, when one is given;
##   3. returns the AVX version, its user header and the make arguments of the instruction set; the AVX version is
##      then compiled with BridgeStan and used as the Stan model, as any Stan model;
##   4. after compilation, checks that the compiled model reports the SIMD lanes of the instruction set (8 for
##      AVX-512, 4 for AVX2).
##
## The AVX version is written again only when the Stan model or the declarations of BayesMVP's functions offered
## to stanc change: its first line records the md5 of both. A fused log density whose header is not in BayesMVP's
## header folder (e.g. an older BayesMVP) is not offered, so the model keeps Stan's own function there.
##
## The lists of BayesMVP's functions and their Stan declarations are in ONE place:
## fn_BayesMVP_AVX_functions_and_their_Stan_declarations() below.
##
##
##
##
#### ---- BayesMVP's AVX functions which replace Stan's functions, and their Stan declarations -----------------
#' BayesMVP's AVX functions which replace Stan's functions in the AVX version of a Stan model
#'
#' The ONE place of the lists: the elementwise functions (each declared for a vector argument), the fused log
#' densities (each declared for every argument type it supports; the name of each list element is the Stan
#' distribution it replaces) and the BayesMVP header which defines each fused log density.
#' @return A list: elementwise_functions, fused_distribution_declarations and
#'   fused_distribution_BayesMVP_header_files.
#' @keywords internal
fn_BayesMVP_AVX_functions_and_their_Stan_declarations <-  function() {

        ##
        ## ---- the elementwise functions (vector -> vector, stan_external_functions.hpp):
        ##
        # elementwise_functions <-  c("exp", "log", "log1p", "log1m", "log1p_exp", "log1m_exp", "inv_logit",
        #                             "log_inv_logit", "Phi", "Phi_approx", "inv_Phi", "tanh", "square", "sqrt",
        #                             "logit", "log1m_inv_logit", "expm1")
        ## (only the expensive functions: square and sqrt, which BayesMVP's header also has, and every basic
        ##  function (pow, abs, fmin, fmax, ...) stay on Stan math)
        elementwise_functions <-  c("exp", "log", "log1p", "log1m", "log1p_exp", "log1m_exp", "inv_logit",
                                    "log_inv_logit", "Phi", "Phi_approx", "inv_Phi", "tanh", "logit",
                                    "log1m_inv_logit", "expm1")
        ## (each declared for a vector, a row_vector and a matrix argument, returning the same shape)
        elementwise_function_argument_types <-  c("vector", "row_vector", "matrix")
        ##
        ## ---- the fused AVX log densities, and the Stan declarations of the argument types each supports:
        ##
        fused_distribution_declarations <-  list(
              normal = c("    real nico_normal_lpdf(vector y, vector mu, vector sigma);",
                         "    real nico_normal_lpdf(vector y, vector mu, real sigma);",
                         "    real nico_normal_lpdf(vector y, real mu, vector sigma);",
                         "    real nico_normal_lpdf(vector y, real mu, real sigma);",
                         "    real nico_normal_lpdf(array[] real y, vector mu, vector sigma);",
                         "    real nico_normal_lpdf(array[] real y, vector mu, real sigma);",
                         "    real nico_normal_lpdf(array[] real y, real mu, vector sigma);",
                         "    real nico_normal_lpdf(array[] real y, real mu, real sigma);"),
                         ## (a single real y, mu and sigma: Stan's own normal, as every call on reals)
                         # "    real nico_normal_lpdf(real y, real mu, real sigma);"),
              student_t =
                    c("    real nico_student_t_lpdf(vector y, real nu, vector mu, vector sigma);",
                      "    real nico_student_t_lpdf(vector y, real nu, vector mu, real sigma);",
                      "    real nico_student_t_lpdf(vector y, real nu, real mu, vector sigma);",
                      "    real nico_student_t_lpdf(vector y, real nu, real mu, real sigma);",
                      "    real nico_student_t_lpdf(array[] real y, real nu, vector mu, vector sigma);",
                      "    real nico_student_t_lpdf(array[] real y, real nu, vector mu, real sigma);",
                      "    real nico_student_t_lpdf(array[] real y, real nu, real mu, vector sigma);",
                      "    real nico_student_t_lpdf(array[] real y, real nu, real mu, real sigma);"),
              ## (Weibull: y vector or array[] real; alpha and sigma each real, vector or array[] real;
              ##  the log density and the log complementary CDF)
              weibull = as.vector(outer(
                    X = c("nico_weibull_lpdf", "nico_weibull_lccdf"),
                    Y = apply(expand.grid(y = c("vector y", "array[] real y"),
                                          alpha = c("real alpha", "vector alpha", "array[] real alpha"),
                                          sigma = c("real sigma", "vector sigma", "array[] real sigma"),
                                          stringsAsFactors = FALSE),
                              MARGIN = 1, FUN = paste, collapse = ", "),
                    FUN = function(function_name, arguments) {

                          paste0("    real ", function_name, "(", arguments, ");")
                    })),

              ordered_logistic =
                    c("    real nico_ordered_logistic_lpmf(int k, real eta, vector c);",
                      "    real nico_ordered_logistic_lpmf(array[] int k, vector eta, vector c);",
                      "    real nico_ordered_logistic_lpmf(array[] int k, vector eta, array[] vector c);"),
              bernoulli_logit = c("    real nico_bernoulli_logit_lpmf(array[] int y, vector f);"),
              ## (the Bernoulli-logit GLM: y | x, alpha, beta, as Stan's bernoulli_logit_glm)
              bernoulli_logit_glm = paste0("    real nico_bernoulli_logit_glm_lpmf(",
                                           "array[] int y, matrix x, ",
                                           c("real alpha", "vector alpha"), ", vector beta);"),
              ## (log_sum_exp: the reduction over a vector, row_vector, matrix or array[] real, and the
              ##  elementwise log_sum_exp of two arguments, at least one a vector; two reals keep Stan's own)
              log_sum_exp = c("    real nico_log_sum_exp(vector x);",
                              "    real nico_log_sum_exp(row_vector x);",
                              "    real nico_log_sum_exp(matrix x);",
                              "    real nico_log_sum_exp(array[] real x);",
                              ## (two reals: Stan's own log_sum_exp, as every call on reals)
                              # "    real nico_log_sum_exp(real a, real b);",
                              "    vector nico_log_sum_exp(vector a, vector b);",
                              "    vector nico_log_sum_exp(real a, vector b);",
                              "    vector nico_log_sum_exp(vector a, real b);"),
              ## (the row-wise log_sum_exp of a matrix: the calls of a model's own function log_sum_exp_2d(), as
              ##  in the latent class models, whose definition keeps its name)
              log_sum_exp_2d = c("    vector nico_log_sum_exp_2d(matrix x);"),
              ## (the squared exponential covariance of data points, as Stan's gp_exp_quad_cov)
              ## (the points must be data: BayesMVP's C++ does not compile with parameters there; not declared
              ##  data, so that a call in the model's own functions, whose arguments are not data, is accepted)
              gp_exp_quad_cov = c(paste0("    matrix nico_gp_exp_quad_cov(array[] real x, real sigma, ",
                                         "real length_scale);"),
                                  paste0("    matrix nico_gp_exp_quad_cov(array[] real x1, ",
                                         "array[] real x2, real sigma, real length_scale);")))
        ##
        ## ---- std_normal: BayesMVP's fused normal log density with mu = 0 and sigma = 1 (the same density, and
        ##      with propto the same constants dropped): "y ~ std_normal()" becomes "y ~ nico_normal(0, 1)" and
        ##      std_normal_lpdf(y) / std_normal_lupdf(y) become nico_normal_lpdf(y | 0, 1) /
        ##      nico_normal_lupdf(y | 0, 1), where a declaration of nico_normal_lpdf accepts y (normal above):
        ##
        std_normal_as_fused_normal_declarations <-  c(
              "    real nico_normal_lpdf(vector y, real mu, real sigma);",
              "    real nico_normal_lpdf(array[] real y, real mu, real sigma);")
        ##
        ## ---- the model's own functions which BayesMVP replaces: only where the model defines a function of this
        ##      name with exactly this signature; its calls (not its definitions) become the BayesMVP function:
        ##
        model_functions_replaced_by_BayesMVP <-  data.frame(
              model_function          = c("log_Phi",
                                          "log_Phi_approx",
                                          "inv_Phi_approx_from_logit_prob",
                                          "inv_Phi_approx_from_prob",
                                          "inv_Phi_from_log_p"),
              model_signature         = c("vector log_Phi(vector)",
                                          "vector log_Phi_approx(vector)",
                                          "vector inv_Phi_approx_from_logit_prob(vector)",
                                          "vector inv_Phi_approx_from_prob(vector)",
                                          "vector inv_Phi_from_log_p(vector, vector)"),
              BayesMVP_function       = c("nico_log_Phi",
                                          "nico_log_Phi_approx",
                                          "nico_inv_Phi_approx_from_logit_prob",
                                          "nico_inv_Phi_approx",
                                          "nico_inv_Phi_from_log_p"),
              BayesMVP_declaration    = c("    vector nico_log_Phi(vector x);",
                                          "    vector nico_log_Phi_approx(vector x);",
                                          "    vector nico_inv_Phi_approx_from_logit_prob(vector logit_p);",
                                          "    vector nico_inv_Phi_approx(vector p);",
                                          "    vector nico_inv_Phi_from_log_p(vector log_p, vector log_1m_p);"),
              stringsAsFactors = FALSE)
        ##
        ## ---- the math functions whose calls left on Stan math in the AVX version are reported (each with its
        ##      line and argument types): a call on a single real or int stays on Stan math and is counted; a
        ##      call on a vector, row_vector, matrix or array stops the writer. The functions, each with the
        ##      numbers of arguments of its calls on reals (no number: it has none, e.g. softmax), and the
        ##      distributions, each with the number of arguments of its log density and the suffix of its
        ##      declaration:
        ##
        math_functions_whose_calls_left_on_Stan_math_are_reported <-  list(
              exp = 1, log = 1, log1p = 1, log1m = 1, expm1 = 1, log1p_exp = 1, log1m_exp = 1, inv_logit = 1,
              logit = 1, log_inv_logit = 1, log1m_inv_logit = 1, Phi = 1, Phi_approx = 1, inv_Phi = 1,
              std_normal_log_qf = 1, std_normal_lcdf = 1, tanh = 1, sinh = 1, cosh = 1, sqrt = 1, square = 1,
              pow = 2, lgamma = 1, digamma = 1, log_sum_exp = 2, log_softmax = c(), softmax = c(),
              log_diff_exp = 2, log_mix = 3, erf = 1, erfc = 1, gp_exp_quad_cov = c(),
              log_Phi = 1, log_Phi_approx = 1, inv_Phi_approx_from_logit_prob = 1, inv_Phi_approx_from_prob = 1,
              inv_Phi_from_log_p = 2)
        distributions_whose_calls_left_on_Stan_math_are_reported <-  list(
              std_normal       = list(n_arguments = 1, declaration_suffix = "_lpdf"),
              normal           = list(n_arguments = 3, declaration_suffix = "_lpdf"),
              student_t        = list(n_arguments = 4, declaration_suffix = "_lpdf"),
              weibull          = list(n_arguments = 3, declaration_suffix = "_lpdf"),
              bernoulli_logit  = list(n_arguments = 2, declaration_suffix = "_lpmf"))
        ##
        ## ---- the BayesMVP header which defines each fused log density (in BayesMVP's header folder):
        ##
        fused_distribution_BayesMVP_header_files <-  c(
              normal               = "stan_external_fused_normal_lpdf.hpp",
              student_t            = "stan_external_fused_student_t_lpdf.hpp",
              weibull              = "stan_external_fused_weibull_lpdf.hpp",
              ordered_logistic     = "stan_external_fused_ordered_logistic_lpmf.hpp",
              bernoulli_logit      = "stan_external_fused_bernoulli_logit_lpmf.hpp",
              bernoulli_logit_glm  = "stan_external_fused_bernoulli_logit_and_probit_glm_lpmf.hpp",
              log_sum_exp          = "stan_external_log_sum_exp_with_nico_exponential.hpp",
              log_sum_exp_2d       = "stan_external_log_sum_exp_with_nico_exponential.hpp",
              gp_exp_quad_cov      = "stan_external_gp_exp_quad_cov.hpp")
        ##
        return(list( elementwise_functions                     = elementwise_functions,
                     elementwise_function_argument_types       = elementwise_function_argument_types,
                     fused_distribution_declarations           = fused_distribution_declarations,
                     fused_distribution_BayesMVP_header_files  = fused_distribution_BayesMVP_header_files,
                     std_normal_as_fused_normal_declarations   = std_normal_as_fused_normal_declarations,
                     model_functions_replaced_by_BayesMVP      = model_functions_replaced_by_BayesMVP,
                     math_functions_whose_calls_left_on_Stan_math_are_reported =
                           math_functions_whose_calls_left_on_Stan_math_are_reported,
                     distributions_whose_calls_left_on_Stan_math_are_reported =
                           distributions_whose_calls_left_on_Stan_math_are_reported))

}




#### ---- the instruction-set macros the compiler defines for this CPU -------------------------------------------
#' The instruction-set macros (__AVX2__, __FMA__, __AVX512F__, __AVX512VL__, __AVX512DQ__, __ARM_NEON) which the
#' C++ compiler R uses defines for this CPU with -march=native, i.e. the instruction sets every model built here
#' gets
#'
#' Works on Linux, macOS and Windows (Rtools), on x86-64 and on ARM, because it asks the compiler itself (the one
#' R CMD config CXX gives) rather than reading an operating-system file; when the compiler does not accept
#' -march=native (some clang versions on ARM), it is asked without it, which still reports __ARM_NEON on ARM.
#' @return A character vector of the macro names defined (possibly empty).
#' @keywords internal
R_fn_instruction_set_macros_defined_by_the_compiler_for_this_CPU <-  function() {

        macros_asked <-  c("__AVX2__", "__FMA__", "__AVX512F__", "__AVX512VL__", "__AVX512DQ__", "__ARM_NEON")
        R_binary <-  file.path(R.home("bin"), "R")
        compiler <-  tryCatch(suppressWarnings(system2(command = R_binary,
                                                       args = c("CMD", "config", "CXX"),
                                                       stdout = TRUE,
                                                       stderr = FALSE)),
                              error = function(e) character(0))
        compiler <-  trimws(paste(compiler, collapse = " "))
        if (!nzchar(compiler)) return(character(0))
        compiler_words <-  strsplit(x = compiler, split = "[[:space:]]+")[[1]]
        empty_source_file <-  tempfile(fileext = ".cpp")
        writeLines(text = character(0), con = empty_source_file)
        on.exit(unlink(empty_source_file), add = TRUE)
        fn_macros_defined_with_flags <-  function(flags) {

              output <-  tryCatch(suppressWarnings(system2(command = compiler_words[1],
                                                           args = c(compiler_words[-1], flags, "-dM", "-E",
                                                                    shQuote(empty_source_file)),
                                                           stdout = TRUE,
                                                           stderr = FALSE)),
                                  error = function(e) character(0))
              if (!is.null(attr(output, "status")) && attr(output, "status") != 0) return(NULL)
              macros_defined <-  sub(pattern = "^#define[[:space:]]+([A-Za-z0-9_]+).*$", replacement = "\\1",
                                     x = grep(pattern = "^#define", x = output, value = TRUE))
              return(macros_asked[macros_asked %in% macros_defined])

        }
        macros_defined <-  fn_macros_defined_with_flags(flags = "-march=native")
        if (is.null(macros_defined)) macros_defined <-  fn_macros_defined_with_flags(flags = character(0))
        if (is.null(macros_defined)) return(character(0))
        return(macros_defined)

}




#### ---- the instruction sets of the SIMD (nico_) functions which this CPU supports -----------------------------
#' The instruction sets of NicoSIMD's (and BayesMVP's) SIMD functions which this CPU supports
#'
#' Asked of the C++ compiler R uses (R_fn_instruction_set_macros_defined_by_the_compiler_for_this_CPU), so that
#' it works on Linux, macOS and Windows, x86-64 and ARM; when the compiler gives no answer, the operating
#' system is asked (Linux: the flags line of /proc/cpuinfo; macOS: sysctl machdep.cpu.features and
#' leaf7_features; Windows:
#' the PROCESSOR_ARCHITECTURE variable, which tells x86-64 from ARM only). AVX2 needs avx2 and fma; AVX-512 needs
#' avx2, fma, avx512f, avx512vl and avx512dq (the instruction-set flags the AVX version of a Stan model is
#' compiled with); on 64-bit ARM the AVX2 functions run as NEON through SIMDe, so ARM with NEON counts as AVX2.
#' @return A named logical vector: BayesMVP_AVX2 and BayesMVP_AVX512.
#' @keywords internal
R_fn_BayesMVP_AVX_instruction_sets_supported_by_this_CPU <-  function() {

        macros_defined <-  R_fn_instruction_set_macros_defined_by_the_compiler_for_this_CPU()
        if (length(macros_defined) > 0) {
              if ("__ARM_NEON" %in% macros_defined) {
                    return(c( BayesMVP_AVX2 = TRUE, BayesMVP_AVX512 = FALSE))
              }
              return(c( BayesMVP_AVX2    = all(c("__AVX2__", "__FMA__") %in% macros_defined),
                        BayesMVP_AVX512  = all(c("__AVX2__", "__FMA__", "__AVX512F__", "__AVX512VL__",
                                                 "__AVX512DQ__") %in% macros_defined)))
        }
        ##
        ## ---- the compiler gave no answer: the operating system is asked
        ##
        CPU_flags <-  character(0)
        operating_system <-  Sys.info()[["sysname"]]
        if (identical(operating_system, "Linux") && file.exists("/proc/cpuinfo")) {
              cpuinfo_lines <-  readLines(con = "/proc/cpuinfo", warn = FALSE)
              first_flags_line <-  grep(pattern = "^(flags|Features)", x = cpuinfo_lines, value = TRUE)[1]
              if (!is.na(first_flags_line)) {
                    CPU_flags <-  strsplit(x = sub(pattern = "^[A-Za-z]+[[:space:]]*:[[:space:]]*",
                                                   replacement = "", x = first_flags_line),
                                           split = "[[:space:]]+")[[1]]
              }
        } else if (identical(operating_system, "Darwin")) {
              sysctl_output <-  tryCatch(suppressWarnings(system2(command = "sysctl",
                                                                  args = c("-n", "machdep.cpu.features",
                                                                           "machdep.cpu.leaf7_features",
                                                                           "hw.optional.neon"),
                                                                  stdout = TRUE,
                                                                  stderr = FALSE)),
                                         error = function(e) character(0))
              CPU_flags <-  tolower(strsplit(x = paste(sysctl_output, collapse = " "),
                                             split = "[[:space:]]+")[[1]])
              on_Apple_silicon <-  identical(Sys.info()[["machine"]], "arm64") ||
                                   identical(sysctl_output[length(sysctl_output)], "1")
              if (on_Apple_silicon) {
                    CPU_flags <-  c(CPU_flags, "neon")
              }
        } else if (identical(operating_system, "Windows")) {
              if (grepl(pattern = "ARM", x = Sys.getenv("PROCESSOR_ARCHITECTURE"), ignore.case = TRUE)) {
                    CPU_flags <-  "neon"
              }
        }
        CPU_flags <-  tolower(CPU_flags)
        if (any(c("neon", "asimd") %in% CPU_flags)) {
              return(c( BayesMVP_AVX2 = TRUE, BayesMVP_AVX512 = FALSE))
        }
        if (length(CPU_flags) == 0) {
              stop(paste0("The instruction sets of this CPU could not be found: the C++ compiler R uses (R CMD ",
                          "config CXX) gave no answer to -march=native -dM -E, and the operating system gave no ",
                          "list of CPU flags. Set Stan_model_math_functions to \"NicoSIMD_AVX2\" or ",
                          "\"NicoSIMD_AVX512\" by hand."))
        }
        CPU_flags_needed_for_AVX2 <-    c("avx2", "fma")
        CPU_flags_needed_for_AVX512 <-  c("avx2", "fma", "avx512f", "avx512vl", "avx512dq")
        return(c( BayesMVP_AVX2    = all(CPU_flags_needed_for_AVX2 %in% CPU_flags),
                  BayesMVP_AVX512  = all(CPU_flags_needed_for_AVX512 %in% CPU_flags)))

}




#### ---- the AVX version of a Stan model: BayesMVP's functions in place of Stan's, where stanc accepts them ----
#' Write the AVX version of a Stan model, with BayesMVP's AVX functions in place of Stan's functions
#'
#'   (1) every call of an elementwise function (e.g. exp, log, inv_logit, Phi) becomes the BayesMVP function of
#'       the same name with "fast_" in front;
#'   (2) every fused distribution offered (e.g. normal, bernoulli_logit), as a ~ statement or an _lpdf / _lupdf /
#'       _lpmf / _lupmf call (and weibull_lccdf), becomes BayesMVP's fused log density of the same name with
#'       "fast_" in front;
#'   (3) the model is compiled with stanc (--allow-undefined) with every declaration offered, and every call that
#'       stanc rejects (argument types without a declaration) gets Stan's own function back, one at a time, until
#'       the model compiles;
#'   (4) the declarations of the BayesMVP functions used, and the generated quantity simd_lanes, are added.
#' Also: the calls of the model's own functions which BayesMVP replaces (model_functions_replaced_by_BayesMVP,
#' only where the model defines the function with that signature) become BayesMVP's function, and std_normal on
#' a vector or array[] real becomes BayesMVP's fused normal with mu = 0 and sigma = 1. Then (5): every call of a
#' math function or distribution of math_functions_whose_calls_left_on_Stan_math_are_reported and
#' distributions_whose_calls_left_on_Stan_math_are_reported which is left on Stan math is listed, with its line
#' and argument types (from stanc), in the report file; a call on a vector, row_vector, matrix or array (other
#' than in the definition of a model function whose calls all go to BayesMVP) stops the writer, and the AVX
#' version is not written.
#' Comments and strings are left as they are. The file is written through a temporary file renamed over it, so
#' that a process starting a fit never reads a partly written model.
#' @param source_Stan_model_file_path The Stan model.
#' @param AVX_Stan_model_file_path The AVX version written.
#' @param first_line_of_AVX_Stan_model The first line of the AVX version (a Stan comment recording the md5 of the
#'   Stan model and of the declarations offered).
#' @param elementwise_functions The elementwise functions offered.
#' @param fused_distribution_declarations The Stan declarations of the fused log densities offered (a named list;
#'   each name is the Stan distribution replaced).
#' @param stanc_file_path The stanc compiler (BridgeStan's, which compiles the model).
#' @param stanc_include_paths_arguments The --include-paths arguments of the model's stanc arguments (NULL: none).
#' @param elementwise_function_argument_types The argument types for which each elementwise function is declared.
#' @param model_functions_replaced_by_BayesMVP The model's own functions which BayesMVP replaces (a data frame).
#' @param std_normal_as_fused_normal_declarations The declarations of nico_normal_lpdf for std_normal (NULL:
#'   none).
#' @param math_functions_whose_calls_left_on_Stan_math_are_reported The functions (a named list: the numbers of
#'   arguments of their calls on reals).
#' @param distributions_whose_calls_left_on_Stan_math_are_reported The distributions (a named list).
#' @param report_file_path_of_calls_left_on_Stan_math The report written (NULL: none).
#' @return The BayesMVP functions used (the elementwise functions and fused distributions, by their Stan names).
#' @keywords internal
R_fn_write_AVX_version_of_Stan_model_with_BayesMVP_functions <-  function(
                    source_Stan_model_file_path,
                    AVX_Stan_model_file_path,
                    first_line_of_AVX_Stan_model,
                    elementwise_functions,
                    fused_distribution_declarations,
                    stanc_file_path,
                    stanc_include_paths_arguments                               = NULL,
                    elementwise_function_argument_types                         = "vector",
                    model_functions_replaced_by_BayesMVP                        = NULL,
                    std_normal_as_fused_normal_declarations                     = NULL,
                    math_functions_whose_calls_left_on_Stan_math_are_reported   = NULL,
                    distributions_whose_calls_left_on_Stan_math_are_reported    = NULL,
                    report_file_path_of_calls_left_on_Stan_math                 = NULL
) {

        source_Stan_model_file_name <-  basename(source_Stan_model_file_path)
        fused_distributions <-  names(fused_distribution_declarations)
        text <-  paste(readLines(con = source_Stan_model_file_path, warn = FALSE), collapse = "\n")
        ##
        ## ---- comments and strings are set aside (replaced by numbered markers) before the rewrite:
        ##
        protected_pattern <-  "\"(?:[^\"\\\\]|\\\\.)*\"|/\\*[\\s\\S]*?\\*/|//[^\\n]*"
        protected <-  regmatches(x = text,
                                 m = gregexpr(pattern = protected_pattern, text = text, perl = TRUE))[[1]]
        for (i in seq_along(protected)) {
              text <-  sub(pattern = protected[i], replacement = paste0("\001", i, "\001"), x = text,
                           fixed = TRUE)
        }
        ## (only the markers which are in the code are restored, so that each line of a long model is restored
        ##  quickly)
        fn_restore <-  function(code) {

              markers <-  regmatches(x = code, m = gregexpr(pattern = "\001[0-9]+\001", text = code))[[1]]
              for (marker in markers) {
                    i <-  as.numeric(gsub(pattern = "\001", replacement = "", x = marker, fixed = TRUE))
                    code <-  sub(pattern = marker, replacement = protected[i], x = code, fixed = TRUE)
              }
              return(code)

        }
        ##
        ## ---- the model's own functions which BayesMVP replaces: those which the model defines with exactly the
        ##      signature of model_functions_replaced_by_BayesMVP (e.g. vector log_Phi(vector x)):
        ##
        fn_pattern_of_definition <-  function(model_signature) {

              ## (e.g. "vector inv_Phi_from_log_p(vector, vector)": the return type, the name and the argument
              ##  types; each argument may be data, and has a name)
              return_type <-  sub(pattern = "^([A-Za-z_]+)\\s.*$", replacement = "\\1", x = model_signature)
              function_name <-  sub(pattern = "^[A-Za-z_]+\\s+([A-Za-z0-9_]+)\\(.*$", replacement = "\\1",
                                    x = model_signature)
              argument_types <-  strsplit(x = sub(pattern = "^.*\\((.*)\\)$", replacement = "\\1",
                                                  x = model_signature),
                                          split = ",\\s*")[[1]]
              arguments_pattern <-  paste(paste0("(data\\s+)?", argument_types, "\\s+[A-Za-z][A-Za-z0-9_]*"),
                                          collapse = "\\s*,\\s*")
              return(paste0("(?<![A-Za-z0-9_.])", return_type, "\\s+", function_name, "\\s*\\(\\s*",
                            arguments_pattern, "\\s*\\)\\s*\\{"))

        }
        if (is.null(model_functions_replaced_by_BayesMVP)) {
              model_functions_replaced_by_BayesMVP <-  data.frame( model_function        = character(0),
                                                                   model_signature       = character(0),
                                                                   BayesMVP_function     = character(0),
                                                                   BayesMVP_declaration  = character(0),
                                                                   stringsAsFactors = FALSE)
        }
        model_function_is_defined_with_its_signature <-  vapply(
              X = model_functions_replaced_by_BayesMVP$model_signature,
              FUN = function(model_signature) {

                    grepl(pattern = fn_pattern_of_definition(model_signature), x = text, perl = TRUE)
              },

              FUN.VALUE = logical(1))
        model_functions_offered <-  model_functions_replaced_by_BayesMVP[
                                          model_function_is_defined_with_its_signature, , drop = FALSE]
        ##
        ## ---- (4) the declarations (every one offered whilst stanc decides; only those used at the end):
        ##
        fn_declarations <-  function(used) {

              used_elementwise <-  intersect(used, elementwise_functions)
              used_fused <-  intersect(used, fused_distributions)
              used_model_functions <-  intersect(used, model_functions_offered$model_function)
              declarations <-  "    real bmvp_simd_lanes();"
              if (length(used_elementwise) > 0) {
                  # declarations <-  c(declarations, paste0("    vector fast_", used_elementwise, "(vector x);"))
                    ## (each for every argument type offered, e.g. vector, row_vector and matrix)
                    for (argument_type in elementwise_function_argument_types) {
                          declarations <-  c(declarations,
                                             paste0("    ", argument_type, " fast_", used_elementwise, "(",
                                                    argument_type, " x);"))
                    }
              }
              declarations <-  c(declarations,
                                 unlist(fused_distribution_declarations[used_fused], use.names = FALSE))
              declarations <-  c(declarations,
                                 model_functions_offered$BayesMVP_declaration[
                                       model_functions_offered$model_function %in% used_model_functions])
              return(unique(declarations))

        }
        ## (extra_declarations: the declarations of the probes of (0) and (5), below)
        fn_assemble <-  function(code, used, extra_declarations = NULL) {

              declarations <-  paste(c(fn_declarations(used), extra_declarations), collapse = "\n")
              if (grepl(pattern = "(^|\\n)\\s*functions\\s*\\{", x = code, perl = TRUE)) {
                    code <-  sub(pattern = "((^|\\n)\\s*functions\\s*\\{)",
                                 replacement = paste0("\\1\n", declarations),
                                 x = code, perl = TRUE)
              } else {
                    code <-  paste0("functions {\n", declarations, "\n}\n", code)
              }
              if (grepl(pattern = "(^|\\n)\\s*generated\\s+quantities\\s*\\{", x = code, perl = TRUE)) {
                    code <-  sub(pattern = "((^|\\n)\\s*generated\\s+quantities\\s*\\{)",
                                 replacement = "\\1\n    real simd_lanes = bmvp_simd_lanes();",
                                 x = code, perl = TRUE)
              } else {
                    code <-  paste0(code, "\ngenerated quantities {\n",
                                    "    real simd_lanes = bmvp_simd_lanes();\n}\n")
              }
              return(code)

        }
        fn_used <-  function(code) {

              used <-  c()
              for (elementwise_function in elementwise_functions) {
                    if (grepl(pattern = paste0("(?<![A-Za-z0-9_.])fast_", elementwise_function, "(?=\\s*\\()"),
                              x = code, perl = TRUE)) {
                          used <-  c(used, elementwise_function)
                    }
              }
              for (distribution in fused_distributions) {
                    if (grepl(pattern = paste0("(?<![A-Za-z0-9_.])fast_", distribution,
                                               "(_lpdf|_lupdf|_lpmf|_lupmf|_lccdf)?(?=\\s*\\()"),
                              x = code, perl = TRUE)) {
                          used <-  c(used, distribution)
                    }
              }
              for (i in seq_len(nrow(model_functions_offered))) {
                    if (grepl(pattern = paste0("(?<![A-Za-z0-9_.])", model_functions_offered$BayesMVP_function[i],
                                               "(?=\\s*\\()"),
                              x = code, perl = TRUE)) {
                          used <-  c(used, model_functions_offered$model_function[i])
                    }
              }
              return(used)

        }
        ##
        ## ---- stanc, on code with the declarations of the functions used (and of the probes of (0) and (5)); the
        ##      calls of named functions in the code; the lines of positions in the code; the end of a bracket:
        ##
        candidate_file <-  tempfile(pattern = "AVX_version_candidate_", fileext = ".stan")
        stanc_output_file <-  tempfile(pattern = "AVX_version_candidate_", fileext = ".hpp")
        fn_stanc_output <-  function(code) {

              writeLines(text = fn_restore(code), con = candidate_file)
              return(suppressWarnings(system2( command = stanc_file_path,
                                               args    = c("--allow-undefined",
                                                           stanc_include_paths_arguments,
                                                           paste0("--o=", shQuote(stanc_output_file)),
                                                           shQuote(candidate_file)),
                                               stdout  = TRUE,
                                               stderr  = TRUE)))

        }
        ## (the calls of the functions named, not their definitions (a name after a return type), each with its
        ##  position, name and suffix; with_distribution_suffixes: a distribution's name followed by _lpdf,
        ##  _lupdf, _lpmf, _lupmf, _lccdf, _lcdf or, in a ~ statement, nothing)
        fn_calls_of_functions <-  function(code, names_of_functions, with_distribution_suffixes) {

              suffix_pattern <-  if (with_distribution_suffixes) {
                    "(_lpdf|_lupdf|_lpmf|_lupmf|_lccdf|_lcdf)?"
              } else "()"
              pattern <-  paste0("(?<![A-Za-z0-9_.])(", paste(names_of_functions, collapse = "|"), ")",
                                 suffix_pattern, "(?=\\s*\\()")
              matches <-  gregexpr(pattern = pattern, text = code, perl = TRUE)[[1]]
              if (matches[1] == -1) {
                    return(data.frame( position = numeric(0), name = character(0), suffix = character(0),
                                       in_tilde_statement = logical(0), stringsAsFactors = FALSE))
              }
              starts_of_names <-  attr(matches, "capture.start")[, 1]
              starts_of_suffixes <-  attr(matches, "capture.start")[, 2]
              names_found <-  substring(text = code, first = starts_of_names,
                                        last = starts_of_names + attr(matches, "capture.length")[, 1] - 1)
              suffixes_found <-  substring(text = code, first = starts_of_suffixes,
                                           last = starts_of_suffixes + attr(matches, "capture.length")[, 2] - 1)
              text_before_call <-  substring(text = code, first = pmax(1, matches - 40), last = matches - 1)
              is_definition <-  grepl(pattern = "(vector|row_vector|matrix|real|int|void)\\s+$",
                                      x = text_before_call)
              calls <-  data.frame( position            = as.numeric(matches),
                                    name                = names_found,
                                    suffix              = suffixes_found,
                                    in_tilde_statement  = grepl(pattern = "~\\s*$", x = text_before_call),
                                    stringsAsFactors = FALSE)
              return(calls[!is_definition, , drop = FALSE])

        }
        ## (the line in the Stan model of each position in the code: the lines of the code before it, plus the
        ##  lines of the comments and strings set aside before it)
        fn_lines_in_Stan_model_of_positions <-  function(code, positions) {

              newline_positions <-  gregexpr(pattern = "\n", text = code, fixed = TRUE)[[1]]
              newline_positions <-  newline_positions[newline_positions > 0]
              lines <-  1 + findInterval(x = positions - 1, vec = newline_positions)
              markers <-  gregexpr(pattern = "\001[0-9]+\001", text = code)[[1]]
              if (markers[1] != -1) {
                    marker_numbers <-  as.numeric(gsub(pattern = "\001", replacement = "",
                                                       x = regmatches(x = code, m = list(markers))[[1]],
                                                       fixed = TRUE))
                    lines_in_markers <-  nchar(gsub(pattern = "[^\n]", replacement = "",
                                                    x = protected[marker_numbers]))
                    markers_before_each_position <-  findInterval(x = positions - 1, vec = as.numeric(markers))
                    lines <-  lines + c(0, cumsum(lines_in_markers))[markers_before_each_position + 1]
              }
              return(lines)

        }
        fn_position_of_matching_closing_bracket <-  function( code,
                                                              position_of_opening_bracket,
                                                              opening,
                                                              closing
        ) {

              characters <-  strsplit(x = substr(code, position_of_opening_bracket, nchar(code)), split = "")[[1]]
              depth <-  cumsum((characters == opening) - (characters == closing))
              return(position_of_opening_bracket - 1 + which(depth == 0)[1])

        }
        ## (the probes: each call k is renamed bmvp_probe_<k>_<its name> and declared with the signatures of
        ##  probe_declarations_of_each_call[[k]]; stanc rejects one call at a time, naming its argument types, and
        ##  that call gets its own name back, until stanc accepts the code. Returns the argument types of each
        ##  call stanc rejected, as stanc names them (e.g. "vector", "matrix, real"), and "" for every call it
        ##  accepted)
        fn_code_with_probe_names <-  function(code, calls, call_is_probed) {

              pieces <-  c()
              previous_end <-  0
              for (k in which(call_is_probed)) {
                    pieces <-  c(pieces, substr(code, previous_end + 1, calls$position[k] - 1),
                                 paste0("bmvp_probe_", k, "_"))
                    previous_end <-  calls$position[k] - 1
              }
              return(paste0(paste(pieces, collapse = ""), substr(code, previous_end + 1, nchar(code))))

        }
        fn_argument_types_of_calls_which_stanc_rejects <-  function(code, calls, probe_declarations_of_each_call,
                                                                    used) {

              argument_types <-  rep("", nrow(calls))
              call_is_probed <-  rep(TRUE, nrow(calls))
              for (attempt in seq_len(nrow(calls) + 1)) {

                    stanc_output <-  fn_stanc_output(code = fn_assemble(
                          code                = fn_code_with_probe_names(code = code, calls = calls,
                                                                         call_is_probed = call_is_probed),
                          used                = used,
                          extra_declarations  = unlist(probe_declarations_of_each_call[call_is_probed],
                                                       use.names = FALSE)))
                    if (is.null(attr(stanc_output, "status")) || attr(stanc_output, "status") == 0) {
                          return(argument_types)
                    }
                    stanc_message <-  paste(stanc_output, collapse = "\n")
                    rejected_call <-  regmatches(x = stanc_message,
                                                 m = regexec(pattern = paste0("Ill-typed arguments supplied to ",
                                                                              "function 'bmvp_probe_([0-9]+)_",
                                                                              "[A-Za-z0-9_]*':[[:space:]]*",
                                                                              "\\(([^)]*)\\)"),
                                                             text = stanc_message))[[1]]
                    if (length(rejected_call) < 3) {
                          stop(paste0("stanc rejected the AVX version of ", source_Stan_model_file_name,
                                      " for another reason than the argument types of a call (whilst finding ",
                                      "the argument types of the calls):\n", stanc_message))
                    }
                    k <-  as.numeric(rejected_call[2])
                    argument_types[k] <-  rejected_call[3]
                    call_is_probed[k] <-  FALSE

              }
              stop(paste0("stanc did not accept the probes of ", source_Stan_model_file_name, "."))

        }
        ##
        ## ---- (0) std_normal on a vector or array[] real becomes BayesMVP's fused normal with mu = 0 and
        ##      sigma = 1 ("y ~ std_normal()" -> "y ~ nico_normal(0, 1)", std_normal_lpdf(y) ->
        ##      nico_normal_lpdf(y | 0, 1), std_normal_lupdf(y) -> nico_normal_lupdf(y | 0, 1)), where stanc
        ##      accepts it with the probes declared for a vector and an array[] real; a call on a real (or a
        ##      row_vector) keeps Stan's std_normal:
        ##
        if (("normal" %in% fused_distributions) && (length(std_normal_as_fused_normal_declarations) > 0)) {

              std_normal_calls <-  fn_calls_of_functions(code = text, names_of_functions = "std_normal",
                                                         with_distribution_suffixes = TRUE)
              std_normal_calls <-  std_normal_calls[(std_normal_calls$suffix %in% c("_lpdf", "_lupdf")) |
                                                    (std_normal_calls$suffix == "" &
                                                     std_normal_calls$in_tilde_statement), , drop = FALSE]
              if (nrow(std_normal_calls) > 0) {
                    probe_declarations_of_each_call <-  lapply(X = seq_len(nrow(std_normal_calls)),
                                                               FUN = function(k) {

                                                                     paste0("    real bmvp_probe_", k,
                                                                            "_std_normal_lpdf(",
                                                                            c("vector y", "array[] real y"), ");")
                                                               })

                    argument_types_of_std_normal_calls_rejected <-
                          fn_argument_types_of_calls_which_stanc_rejects(
                          code                             = text,
                          calls                            = std_normal_calls,
                          probe_declarations_of_each_call  = probe_declarations_of_each_call,
                          used                             = c())
                    ## (from the last call to the first, so that the positions of the calls before it stay valid)
                    for (k in rev(which(argument_types_of_std_normal_calls_rejected == ""))) {

                          position_of_call <-  std_normal_calls$position[k]
                          position_of_opening_parenthesis <-  position_of_call - 1 +
                                regexpr(pattern = "(", text = substr(text, position_of_call, nchar(text)),
                                        fixed = TRUE)
                          position_of_closing_parenthesis <-  fn_position_of_matching_closing_bracket(
                                code = text, position_of_opening_bracket = position_of_opening_parenthesis,
                                opening = "(", closing = ")")
                          arguments_of_call <-  substr(text, position_of_opening_parenthesis + 1,
                                                       position_of_closing_parenthesis - 1)
                          call_with_BayesMVP_normal <-  if (std_normal_calls$suffix[k] == "") {
                                "nico_normal(0, 1)"
                          } else {
                                paste0("nico_normal", std_normal_calls$suffix[k], "(", arguments_of_call,
                                       " | 0, 1)")
                          }
                          text <-  paste0(substr(text, 1, position_of_call - 1), call_with_BayesMVP_normal,
                                          substr(text, position_of_closing_parenthesis + 1, nchar(text)))

                    }
              }

        }
        ##
        ## ---- (1) and (2): Stan's functions become BayesMVP's (a name followed by "(", not part of a longer
        ##      name):
        ##
        if (length(elementwise_functions) > 0) {
              text <-  gsub(pattern = paste0("(?<![A-Za-z0-9_.])(", paste(elementwise_functions, collapse = "|"),
                                             ")(?=\\s*\\()"),
                            replacement = "fast_\\1",
                            x = text,
                            perl = TRUE)
        }
        if (length(fused_distributions) > 0) {
              text <-  gsub(pattern = paste0("(?<![A-Za-z0-9_.])(", paste(fused_distributions, collapse = "|"),
                                             ")(_lpdf|_lupdf|_lpmf|_lupmf)?(?=\\s*\\()"),
                            replacement = "fast_\\1\\2",
                            x = text,
                            perl = TRUE)
        }
        if ("weibull" %in% fused_distributions) {
              text <-  gsub(pattern = "(?<![A-Za-z0-9_.])weibull_lccdf(?=\\s*\\()",
                            replacement = "nico_weibull_lccdf",
                            x = text,
                            perl = TRUE)
        }
        ## (a model's own function log_sum_exp_2d keeps its name in its definition and declarations, i.e. after
        ##  its return type vector: only its calls become BayesMVP's nico_log_sum_exp_2d)
        if ("log_sum_exp_2d" %in% fused_distributions) {
              text <-  gsub(pattern = "(?<![A-Za-z0-9_.])(vector\\s+)nico_log_sum_exp_2d(?=\\s*\\()",
                            replacement = "\\1log_sum_exp_2d",
                            x = text,
                            perl = TRUE)
        }
        ## (the model's own functions offered: their calls become BayesMVP's function, e.g.
        ##  inv_Phi_approx_from_prob -> nico_inv_Phi_approx; their definitions, after a return type, keep their
        ##  names)
        for (i in seq_len(nrow(model_functions_offered))) {
              text <-  gsub(pattern = paste0("(?<![A-Za-z0-9_.])", model_functions_offered$model_function[i],
                                             "(?=\\s*\\()"),
                            replacement = model_functions_offered$BayesMVP_function[i],
                            x = text,
                            perl = TRUE)
              text <-  gsub(pattern = paste0("(?<![A-Za-z0-9_.])((vector|row_vector|matrix|real|int|void)\\s+)",
                                             model_functions_offered$BayesMVP_function[i], "(?=\\s*\\()"),
                            replacement = paste0("\\1", model_functions_offered$model_function[i]),
                            x = text,
                            perl = TRUE)
        }
        ##
        ## ---- (3) stanc decides: a call it rejects gets Stan's own function back, one call at a time:
        ##
        candidate_file <-  tempfile(pattern = "AVX_version_candidate_", fileext = ".stan")
        stanc_output_file <-  tempfile(pattern = "AVX_version_candidate_", fileext = ".hpp")
        stanc_accepted_the_AVX_version <-  FALSE
        # for (attempt in seq_len(500)) {
        ## (at most every call of a BayesMVP function is given back: one attempt more than there are calls)
        number_of_calls_of_BayesMVP_functions <-  length(gregexpr(
              pattern = "(?<![A-Za-z0-9_.])fast_[A-Za-z0-9_]+(?=\\s*\\()", text = text, perl = TRUE)[[1]])
        for (attempt in seq_len(number_of_calls_of_BayesMVP_functions + 1)) {

              code <-  fn_assemble(code = text, used = c(elementwise_functions, fused_distributions,
                                                         model_functions_offered$model_function))
              writeLines(text = fn_restore(code), con = candidate_file)
              stanc_output <-  suppressWarnings(system2( command = stanc_file_path,
                                                         args    = c("--allow-undefined",
                                                                     stanc_include_paths_arguments,
                                                                     paste0("--o=", shQuote(stanc_output_file)),
                                                                     shQuote(candidate_file)),
                                                         stdout  = TRUE,
                                                         stderr  = TRUE))
              if (is.null(attr(stanc_output, "status")) || attr(stanc_output, "status") == 0) {
                    stanc_accepted_the_AVX_version <-  TRUE
                    break
              }
              stanc_message <-  paste(stanc_output, collapse = " ")
              position <-  regmatches(x = stanc_message,
                                      m = regexec(pattern = "line ([0-9]+), column ([0-9]+)",
                                                  text = stanc_message))[[1]]
              if (length(position) < 3 || !grepl(pattern = "fast_[A-Za-z0-9_]+", x = stanc_message)) {
                    stop(paste0("stanc rejected the AVX version of ", source_Stan_model_file_name,
                                " for another reason than a call of a BayesMVP function:\n",
                                paste(stanc_output, collapse = "\n")))
              }
              ## (the line of the rejected call in the rewritten text: the assembled code has the declarations
              ##  above it, so the line is found again by its content)
              code_lines <-  strsplit(x = fn_restore(code), split = "\n")[[1]]
              rejected_line <-  code_lines[as.numeric(position[2])]
              rejected_column <-  as.numeric(position[3])
              text_lines <-  strsplit(x = text, split = "\n")[[1]]
              matching_lines <-  which(vapply(X = text_lines,
                                              FUN = function(line) identical(fn_restore(line), rejected_line),
                                              FUN.VALUE = logical(1)))
              if (length(matching_lines) == 0) {

                    stop(paste0("The line of ", source_Stan_model_file_name, " which stanc rejected could not ",
                                "be found in the AVX version (e.g. a line with the start of a comment over ",
                                "several lines): ", rejected_line))
              }

              # line_index <-  matching_lines[1]
              ## (several lines with the same content, e.g. the vector and real versions of a function: the line
              ##  at stanc's line, i.e. the same one, counted in order, of the identical lines of the candidate,
              ##  which has the declarations above them)
              candidate_lines_with_the_same_content <-  which(code_lines == rejected_line)
              order_of_rejected_line <-  match(as.numeric(position[2]), candidate_lines_with_the_same_content)
              if (length(candidate_lines_with_the_same_content) != length(matching_lines) ||
                  is.na(order_of_rejected_line)) {
                    stop(paste0("The line of ", source_Stan_model_file_name, " which stanc rejected could not ",
                                "be told apart from other lines with the same content: ", rejected_line))
              }
              line_index <-  matching_lines[order_of_rejected_line]
              ## (the rejected call: the call of a BayesMVP function in the rejected line which starts nearest to
              ##  the column given by stanc; the name in stanc's message is not used, since the message also shows
              ##  the lines around the rejected one, which may call other BayesMVP functions; a ~ statement names
              ##  the distribution without its _lpdf / _lpmf suffix, as it is written in the line)
              calls <-  gregexpr(pattern = "fast_[A-Za-z0-9_]+(?=\\s*\\()", text = text_lines[line_index],
                                 perl = TRUE)[[1]]
              if (calls[1] == -1) {
                    stop(paste0("The line of ", source_Stan_model_file_name, " which stanc rejected has no ",
                                "call of a BayesMVP function: ", rejected_line, "\n",
                                paste(stanc_output, collapse = "\n")))
              }
              closest_call <-  calls[which.min(abs(calls - (rejected_column + 1)))]
              text_before_call <-  substr(x = text_lines[line_index], start = 1, stop = closest_call - 1)
              text_from_call_without_fast <-  substr(x = text_lines[line_index],
                                                     start = closest_call + nchar("fast_"),
                                                     stop = nchar(text_lines[line_index]))
              ## (a BayesMVP function which replaces a model's own function gives the model's function back, e.g.
              ##  nico_inv_Phi_approx -> inv_Phi_approx_from_prob)
              text_from_call <-  substr(x = text_lines[line_index], start = closest_call,
                                        stop = nchar(text_lines[line_index]))
              name_of_closest_call <-  regmatches(x = text_from_call,
                                                  m = regexpr(pattern = "^fast_[A-Za-z0-9_]+",
                                                              text = text_from_call))
              if (name_of_closest_call %in% model_functions_offered$BayesMVP_function) {
                    model_function <-  model_functions_offered$model_function[
                                             match(name_of_closest_call,
                                                   model_functions_offered$BayesMVP_function)]
                    text_from_call_without_fast <-  paste0(model_function,
                                                           substr(x = text_from_call,
                                                                  start = nchar(name_of_closest_call) + 1,
                                                                  stop = nchar(text_from_call)))
              }
              text_lines[line_index] <-  paste0(text_before_call, text_from_call_without_fast)
              text <-  paste(text_lines, collapse = "\n")

        }
        if (!stanc_accepted_the_AVX_version) {
              # stop(paste0("stanc did not accept the AVX version of ", source_Stan_model_file_name,
              #             " after 500 calls of BayesMVP functions were given back to Stan."))
              stop(paste0("stanc did not accept the AVX version of ", source_Stan_model_file_name,
                          " after every call of a BayesMVP function was given back to Stan."))
        }
        ##
        ## ---- (5) the calls left on Stan math: every call of a function of
        ##      math_functions_whose_calls_left_on_Stan_math_are_reported or of a distribution of
        ##      distributions_whose_calls_left_on_Stan_math_are_reported which is still in the AVX version, with
        ##      its line in the Stan model and its argument types (probes declared for reals and ints only: stanc
        ##      names the argument types of every call on a vector, row_vector, matrix or array). A call on a
        ##      single real or int stays on Stan math and is counted; any other call stops the writer, unless it
        ##      is in the definition of a model's function whose calls all go to BayesMVP (that code is not run):
        ##
        used <-  fn_used(code = text)
        function_calls_left <-  fn_calls_of_functions(
              code                        = text,
              names_of_functions          = names(math_functions_whose_calls_left_on_Stan_math_are_reported),
              with_distribution_suffixes  = FALSE)
        function_calls_left$kind <-  rep("function", nrow(function_calls_left))
        distribution_calls_left <-  fn_calls_of_functions(
              code                        = text,
              names_of_functions          = names(distributions_whose_calls_left_on_Stan_math_are_reported),
              with_distribution_suffixes  = TRUE)
        ## (a distribution's name without a suffix is a call of the distribution only in a ~ statement)
        distribution_calls_left <-  distribution_calls_left[(distribution_calls_left$suffix != "") |
                                                            distribution_calls_left$in_tilde_statement, ,
                                                            drop = FALSE]
        distribution_calls_left$kind <-  rep("distribution", nrow(distribution_calls_left))
        calls_left <-  rbind(function_calls_left, distribution_calls_left)
        calls_left <-  calls_left[order(calls_left$position), , drop = FALSE]
        calls_left$line_in_Stan_model <-  fn_lines_in_Stan_model_of_positions(code = text,
                                                                              positions = calls_left$position)
        ## (the probes: real and int arguments, in every combination, for each number of arguments of the calls on
        ##  reals; a function with none (e.g. softmax) gets a signature which no call has, a 4-dimensional int
        ##  array; the variate of a distribution, its first argument, is real for a density and int for a mass
        ##  function)
        fn_signatures_on_reals <-  function(n_arguments, type_of_first_argument = c("real", "int")) {

              if (length(n_arguments) == 0) return("array[,,,] int a1")
              signatures <-  c()
              for (n in n_arguments) {
                    argument_types <-  expand.grid(c(list(type_of_first_argument),
                                                     rep(list(c("real", "int")), n - 1)),
                                                   stringsAsFactors = FALSE)
                    for (r in seq_len(nrow(argument_types))) {
                          signatures <-  c(signatures, paste(paste0(unlist(argument_types[r, ]), " a", 1:n),
                                                             collapse = ", "))
                    }
              }
              return(signatures)

        }
        probe_declarations_of_each_call <-  lapply(X = seq_len(nrow(calls_left)), FUN = function(k) {

              name <-  calls_left$name[k]
              if (identical(calls_left$kind[k], "function")) {
                    numbers_of_arguments_on_reals <-  math_functions_whose_calls_left_on_Stan_math_are_reported[[
                                                            name]]
                    return(paste0("    real bmvp_probe_", k, "_", name, "(",
                                  fn_signatures_on_reals(numbers_of_arguments_on_reals), ");"))
              }
              distribution <-  distributions_whose_calls_left_on_Stan_math_are_reported[[name]]
              declaration_suffix <-  if (calls_left$suffix[k] %in% c("_lccdf", "_lcdf")) {
                    calls_left$suffix[k]
              } else distribution$declaration_suffix
              type_of_variate <-  if (identical(declaration_suffix, "_lpmf")) "int" else "real"
              return(paste0("    real bmvp_probe_", k, "_", name, declaration_suffix, "(",
                            fn_signatures_on_reals(n_arguments = distribution$n_arguments,
                                                   type_of_first_argument = type_of_variate), ");"))

        })
        calls_left$argument_types <-  if (nrow(calls_left) > 0) {
              fn_argument_types_of_calls_which_stanc_rejects(code = text, calls = calls_left,
                                                             probe_declarations_of_each_call =
                                                                   probe_declarations_of_each_call,
                                                             used = used)
        } else character(0)
        ##
        ## (the definitions which the AVX version does not call: those of the model's functions offered, and
        ##  log_sum_exp_2d where nico_log_sum_exp_2d is used, unless a call left on Stan math has their argument
        ##  types)
        signatures_of_definitions_not_called <-  model_functions_offered$model_signature
        if ("log_sum_exp_2d" %in% used) {
              signatures_of_definitions_not_called <-  c(signatures_of_definitions_not_called,
                                                         "vector log_sum_exp_2d(matrix)")
        }
        calls_left$in_definition_not_called <-  rep("", nrow(calls_left))
        for (model_signature in signatures_of_definitions_not_called) {

              definition <-  regexpr(pattern = fn_pattern_of_definition(model_signature), text = text,
                                     perl = TRUE)
              if (definition == -1) next
              function_name <-  sub(pattern = "^[A-Za-z_]+\\s+([A-Za-z0-9_]+)\\(.*$", replacement = "\\1",
                                    x = model_signature)
              argument_types_of_signature <-  sub(pattern = "^.*\\((.*)\\)$", replacement = "\\1",
                                                  x = model_signature)
              argument_types_of_calls_left <-  gsub(pattern = "data ", replacement = "",
                                                    x = calls_left$argument_types)
              calls_of_this_definition_left <-  (calls_left$name == function_name) &
                                                (argument_types_of_calls_left == argument_types_of_signature)
              if (any(calls_of_this_definition_left)) next
              position_of_opening_brace <-  definition + attr(definition, "match.length") - 1
              position_of_closing_brace <-  fn_position_of_matching_closing_bracket(
                    code = text, position_of_opening_bracket = position_of_opening_brace, opening = "{",
                    closing = "}")
              call_is_in_definition <-  (calls_left$position > position_of_opening_brace) &
                                        (calls_left$position < position_of_closing_brace)
              calls_left$in_definition_not_called[call_is_in_definition] <-  model_signature

        }
        ## (the calls which BayesMVP has a version of, as offered to stanc: the elementwise functions, the fused
        ##  log densities and functions (with their suffixes; weibull also _lccdf), std_normal as the fused
        ##  normal, and the model's own functions offered; a call of any other function (e.g. pow, fmin, erfc)
        ##  on a vector has no BayesMVP version, and is listed only)
        distributions_offered <-  fused_distributions
        if (("normal" %in% fused_distributions) && (length(std_normal_as_fused_normal_declarations) > 0)) {
              distributions_offered <-  c(distributions_offered, "std_normal")
        }
        BayesMVP_has_a_version_of_the_call <-  (calls_left$name %in% c(elementwise_functions,
                                                                       model_functions_offered$model_function,
                                                                       fused_distributions) &
                                                calls_left$suffix == "") |
                                               (calls_left$name %in% distributions_offered &
                                                calls_left$suffix %in% c("_lpdf", "_lupdf", "_lpmf", "_lupmf")) |
                                               (calls_left$name == "weibull" &
                                                "weibull" %in% fused_distributions &
                                                calls_left$suffix == "_lccdf")
        # calls_left$category <-  ifelse(calls_left$argument_types == "",
        #                                "real or int: stays on Stan math",
        #                                ifelse(calls_left$in_definition_not_called != "",
        #                                       paste0("in the definition of ",
        #                                              calls_left$in_definition_not_called,
        #                                              ", which the AVX version does not call"),
        #                                       "NOT SWAPPED: left on Stan math"))
        calls_left$category <-  ifelse(calls_left$argument_types == "",
                                       "real or int: stays on Stan math",
                                ifelse(calls_left$in_definition_not_called != "",
                                       paste0("in the definition of ", calls_left$in_definition_not_called,
                                              ", which the AVX version does not call"),
                                ifelse(!BayesMVP_has_a_version_of_the_call,
                                       "no BayesMVP version: stays on Stan math",
                                       "NOT SWAPPED: BayesMVP has this function, but stanc gave the call back")))
        calls_left_on_vectors_or_matrices <-  calls_left[grepl(pattern = "^NOT SWAPPED",
                                                               x = calls_left$category), , drop = FALSE]
        ##
        ## (the report: the calls of BayesMVP functions, then every call left on Stan math)
        ##
        BayesMVP_calls <-  regmatches(x = text,
                                      m = gregexpr(pattern = "(?<![A-Za-z0-9_.])fast_[A-Za-z0-9_]+(?=\\s*\\()",
                                                   text = text, perl = TRUE))[[1]]
        BayesMVP_calls_counted <-  table(BayesMVP_calls)
        calls_left_as_text <-  paste0(formatC(calls_left$line_in_Stan_model, width = 6), "  ",
                                      formatC(paste0(calls_left$name, calls_left$suffix,
                                                     ifelse(calls_left$in_tilde_statement, " (~)", "")),
                                              width = -34),
                                      formatC(ifelse(calls_left$argument_types == "", "real / int",
                                                     calls_left$argument_types),
                                              width = -28),
                                      calls_left$category)
        report_lines <-  c(paste0("Calls left on Stan math in the AVX version of ", source_Stan_model_file_name,
                                  ": ", basename(AVX_Stan_model_file_path)),
                           "",
                           paste0("Calls of BayesMVP functions: ", length(BayesMVP_calls), " (",
                                  paste(paste0(names(BayesMVP_calls_counted), " ",
                                               as.numeric(BayesMVP_calls_counted)),
                                        collapse = ", "),
                                  ")"),
                           paste0("Calls left on Stan math on a single real or int: ",
                                  sum(calls_left$argument_types == "")),
                           paste0("Calls left on Stan math in definitions which the AVX version does not call: ",
                                  sum(calls_left$argument_types != "" &
                                      calls_left$in_definition_not_called != "")),
                           paste0("Calls left on Stan math on a vector, row_vector, matrix or array, of ",
                                  "functions BayesMVP has no version of: ",
                                  sum(calls_left$category == "no BayesMVP version: stays on Stan math")),
                           paste0("Calls left on Stan math on a vector, row_vector, matrix or array, of ",
                                  "functions BayesMVP has (NOT SWAPPED): ",
                                  nrow(calls_left_on_vectors_or_matrices)),
                           "",
                           paste0(formatC("line", width = 6), "  ", formatC("function", width = -34),
                                  formatC("argument types", width = -28), "category"),
                           calls_left_as_text)
        if (!is.null(report_file_path_of_calls_left_on_Stan_math)) {
              writeLines(text = report_lines, con = report_file_path_of_calls_left_on_Stan_math)
        }
        message(colourise(paste0(source_Stan_model_file_name, ": ", length(BayesMVP_calls),
                                 " calls of BayesMVP functions; left on Stan math: ",
                                 sum(calls_left$argument_types == ""), " calls on reals, ",
                                 sum(calls_left$argument_types != "" & calls_left$in_definition_not_called != ""),
                                 " in definitions not called, ",
                                 sum(calls_left$category == "no BayesMVP version: stays on Stan math"),
                                 " on vectors, row_vectors, matrices or arrays with no BayesMVP version, ",
                                 nrow(calls_left_on_vectors_or_matrices),
                                 " NOT swapped although BayesMVP has the function",
                                 if (!is.null(report_file_path_of_calls_left_on_Stan_math)) {
                                       paste0(" (", report_file_path_of_calls_left_on_Stan_math, ")")
                                 }),
                          "cyan"))
        if (nrow(calls_left_on_vectors_or_matrices) > 0) {
              stop(paste0("The AVX version of ", source_Stan_model_file_name, " would leave ",
                          nrow(calls_left_on_vectors_or_matrices), " calls on a vector, row_vector, matrix or ",
                          "array of functions which BayesMVP has on Stan math (stanc gave them back), so it is ",
                          "not written (the model, or BayesMVP's declarations, must be changed):\n",
                          paste(paste0("  line ", calls_left_on_vectors_or_matrices$line_in_Stan_model, ": ",
                                       calls_left_on_vectors_or_matrices$name,
                                       calls_left_on_vectors_or_matrices$suffix, "(",
                                       calls_left_on_vectors_or_matrices$argument_types, ")"),
                                collapse = "\n")))
        }
        ##
        ## ---- the AVX version: its first line, the BayesMVP functions used, and the model with only their
        ##      declarations:
        ##
        used <-  fn_used(code = text)
        code <-  fn_restore(fn_assemble(code = text, used = used))
        BayesMVP_functions_used_as_text <-  if (length(used) > 0) paste(used, collapse = ", ") else "none"
        header_lines <-  c(first_line_of_AVX_Stan_model,
                           paste0("// BayesMVP functions used: ", BayesMVP_functions_used_as_text))
        temporary_file <-  tempfile(pattern = "AVX_version_being_written_",
                                    tmpdir = dirname(AVX_Stan_model_file_path),
                                    fileext = ".stan.tmp")
        writeLines(text = c(header_lines, code), con = temporary_file)
        if (!file.rename(from = temporary_file, to = AVX_Stan_model_file_path)) {
              stop(paste0("Could not write the AVX version of the Stan model: ", AVX_Stan_model_file_path))
        }
        ##
        return(used)

}




#### ---- the AVX version of a Stan model, its user header and its make arguments ------------------------------
#' Write the AVX version of a Stan model and its user header, and return them with the make arguments
#'
#' For Stan_model_math_functions = "BayesMVP_AVX2", "BayesMVP_AVX512" or
#' "BayesMVP_AVX512_if_supported_else_AVX2" (AVX-512 when this CPU supports it, else AVX2): checks the CPU,
#' writes (or reuses) the AVX version of the Stan model next to it and its user header, and returns the make
#' arguments of the instruction set (STAN_THREADS=true, PRECOMPILED_HEADERS=false, CXXFLAGS_OPTIM with the
#' instruction-set flags of NicoStan's examples and, for AVX2, CPPFLAGS_OPTIM=-DBAYESMVP_FORCE_AVX2, i.e.
#' BayesMVP's 4-lane kernels). The AVX version is written again only when the Stan model or the declarations
#' offered to stanc change (both md5s are in its first line); the user header is written again only when its
#' content changes, and the compiled model of the AVX version is then moved aside (with the suffix
#' .built_with_previous_user_header), so that BridgeStan builds it again with the new header.
#' @param Stan_model_math_functions "NicoSIMD_AVX2", "NicoSIMD_AVX512" or "NicoSIMD_AVX512_if_supported_else_AVX2"
#'   (NicoSIMD's functions: the same C++ as BayesMVP's AVX functions, with nico_ in place of fast_; NicoSIMD's
#'   names are put in at the end, from its names_renamed_from_fast_to_nico.txt), or "BayesMVP_AVX2",
#'   "BayesMVP_AVX512" or "BayesMVP_AVX512_if_supported_else_AVX2" (BayesMVP's own AVX functions).
#' @param Stan_model_file_path The Stan model.
#' @param Stan_cpp_user_header The user's own C++ header (NULL: none); included by the AVX version's user header.
#' @param stanc_args The model's stanc arguments (only their --include-paths arguments are used here).
#' @param make_args The model's make arguments (a USER_HEADER is included by the AVX version's user header; the
#'   instruction-set make arguments CXXFLAGS_OPTIM, CPPFLAGS_OPTIM and PRECOMPILED_HEADERS must not be given).
#' @param BayesMVP_AVX_headers_folder The folder of BayesMVP's AVX headers (stan_external_functions.hpp and
#'   stan_external_fused_*.hpp); NULL (default): the installed BayesMVP's include/BayesMVP folder. For BayesMVP's
#'   functions only.
#' @param NicoSIMD_headers_folder The folder of NicoSIMD's headers (the same file names as BayesMVP's, and
#'   names_renamed_from_fast_to_nico.txt); NULL (default): the installed NicoSIMD's include/NicoSIMD folder. For
#'   NicoSIMD's functions only.
#' @return A list: Stan_model_math_functions (as given), Stan_model_math_functions_used ("NicoSIMD_AVX2",
#'   "NicoSIMD_AVX512", "BayesMVP_AVX2" or "BayesMVP_AVX512"), package_of_the_SIMD_functions ("NicoSIMD" or
#'   "BayesMVP"), source_Stan_model_file_path, Stan_model_file_path (the AVX version), Stan_cpp_user_header (its
#'   user header), make_args, headers_folder_of_the_SIMD_functions, BayesMVP_AVX_headers_folder (the same folder,
#'   for BayesMVP's functions; NULL for NicoSIMD's), BayesMVP_functions_used (the SIMD functions used, by their
#'   Stan names, whichever package) and SIMD_lanes_expected (8 for AVX-512, 4 for AVX2).
#' @keywords internal
R_fn_write_BayesMVP_AVX_version_of_Stan_model_and_user_header <-  function( Stan_model_math_functions,
                                                                           Stan_model_file_path,
                                                                           Stan_cpp_user_header = NULL,
                                                                           stanc_args = NULL,
                                                                           make_args = NULL,
                                                                           BayesMVP_AVX_headers_folder = NULL,
                                                                           NicoSIMD_headers_folder = NULL
) {

        Stan_model_math_functions_with_BayesMVP_AVX <-  c("BayesMVP_AVX2",
                                                          "BayesMVP_AVX512",
                                                          "BayesMVP_AVX512_if_supported_else_AVX2")
        ## (NicoSIMD's functions: the same C++ as BayesMVP's AVX functions, with nico_ in place of fast_)
        Stan_model_math_functions_with_NicoSIMD <-  c("NicoSIMD_AVX2",
                                                      "NicoSIMD_AVX512",
                                                      "NicoSIMD_AVX512_if_supported_else_AVX2")
        if (!(is.character(Stan_model_math_functions) && length(Stan_model_math_functions) == 1 &&
              Stan_model_math_functions %in% c("Stan_math", Stan_model_math_functions_with_NicoSIMD,
                                               Stan_model_math_functions_with_BayesMVP_AVX))) {
              stop(paste0("Stan_model_math_functions must be one of: \"Stan_math\" (the default), \"",
                          paste(c(Stan_model_math_functions_with_NicoSIMD,
                                  Stan_model_math_functions_with_BayesMVP_AVX),
                                collapse = "\", \""), "\"."))
        }
        if (identical(Stan_model_math_functions, "Stan_math")) {
              stop(paste0("R_fn_write_BayesMVP_AVX_version_of_Stan_model_and_user_header: ",
                          "Stan_model_math_functions = \"Stan_math\" needs no AVX version."))
        }
        ##
        ## ---- the package of the SIMD functions: NicoSIMD (nico_ names) or BayesMVP (fast_ names). The checks of
        ##      this CPU and the AVX version are made with BayesMVP's names, and, for NicoSIMD, NicoSIMD's names
        ##      are put in at the end (R_fn_put_NicoSIMD_names_in_place_of_BayesMVP_names_in_AVX_version_of_Stan_
        ##      model() below):
        ##
        SIMD_functions_from_NicoSIMD <-  Stan_model_math_functions %in% Stan_model_math_functions_with_NicoSIMD
        package_of_the_SIMD_functions <-  if (SIMD_functions_from_NicoSIMD) "NicoSIMD" else "BayesMVP"
        ## ("AVX2", "AVX512" or "AVX512_if_supported_else_AVX2")
        instruction_set_asked <-  sub(pattern = "^(NicoSIMD|BayesMVP)_", replacement = "",
                                      x = Stan_model_math_functions)
        ##
        ## ---- NicoSIMD's headers: the installed NicoSIMD's, unless a folder is given (NicoSIMD is needed only
        ##      here, i.e. only for its SIMD functions):
        ##
        if (SIMD_functions_from_NicoSIMD) {
              if (is.null(NicoSIMD_headers_folder)) {
                    if (!requireNamespace("NicoSIMD", quietly = TRUE)) {
                          stop(paste0("Stan_model_math_functions = \"", Stan_model_math_functions, "\" needs ",
                                      "NicoSIMD's functions, but the NicoSIMD package is not installed. Install ",
                                      "NicoSIMD, give the folder of its headers in NicoSIMD_headers_folder, or ",
                                      "use Stan_model_math_functions = \"Stan_math\"."))
                    }
                    NicoSIMD_headers_folder <-  system.file("include", "NicoSIMD", package = "NicoSIMD")
              }
              headers_folder_of_the_SIMD_functions <-  normalizePath(path = NicoSIMD_headers_folder,
                                                                     mustWork = FALSE)
              BayesMVP_AVX_headers_folder <-  NULL
        }
        ##
        ## ---- BayesMVP's AVX headers: the installed BayesMVP's, unless a folder is given (BayesMVP is needed
        ##      only here, i.e. only for its AVX functions):
        ##
        # if (!SIMD_functions_from_NicoSIMD && is.null(BayesMVP_AVX_headers_folder)) {
        #       if (!requireNamespace("BayesMVP", quietly = TRUE)) {
        #             stop(paste0("Stan_model_math_functions = \"", Stan_model_math_functions, "\" needs ",
        #                         "BayesMVP's AVX functions, but the BayesMVP package is not installed. Install ",
        #                         "BayesMVP, give the folder of its AVX headers in BayesMVP_AVX_headers_folder, ",
        #                         "or use Stan_model_math_functions = \"Stan_math\"."))
        #       }
        #       BayesMVP_AVX_headers_folder <-  system.file("include", "BayesMVP", package = "BayesMVP")
        # }
        ##
        ## (BayesMVP holds no AVX functions of its own any more: the functions of its fast_ names are NicoSIMD's,
        ##  given their fast_ names by NicoSIMD's stan_external_functions_with_the_fast_names_of_BayesMVP.hpp, so
        ##  the folder of the headers of the fast_ names is the installed NicoSIMD's)
        if (!SIMD_functions_from_NicoSIMD && is.null(BayesMVP_AVX_headers_folder)) {
              if (!requireNamespace("NicoSIMD", quietly = TRUE)) {
                    stop(paste0("Stan_model_math_functions = \"", Stan_model_math_functions, "\" needs ",
                                "the AVX functions of NicoSIMD (under BayesMVP's fast_ names), but the NicoSIMD ",
                                "package is not installed. Install NicoSIMD, give the folder of its headers in ",
                                "BayesMVP_AVX_headers_folder, or use Stan_model_math_functions = \"Stan_math\"."))
              }
              BayesMVP_AVX_headers_folder <-  system.file("include", "NicoSIMD", package = "NicoSIMD")
        }
        if (!SIMD_functions_from_NicoSIMD) {
              BayesMVP_AVX_headers_folder <-  normalizePath(path = BayesMVP_AVX_headers_folder, mustWork = FALSE)
              headers_folder_of_the_SIMD_functions <-  BayesMVP_AVX_headers_folder
        }
        elementwise_functions_header_of_the_SIMD_functions <-  file.path(headers_folder_of_the_SIMD_functions,
                                                                         "stan_external_functions.hpp")
        ##
        ## (the fast_ names: NicoSIMD's header that gives them to its nico_ functions, read before every other
        ##  header of the AVX version, so that the fused headers included after it get the fast_ names too)
        header_of_the_fast_names <-  file.path(headers_folder_of_the_SIMD_functions,
                                               "stan_external_functions_with_the_fast_names_of_BayesMVP.hpp")
        if (!SIMD_functions_from_NicoSIMD && file.exists(header_of_the_fast_names)) {
              elementwise_functions_header_of_the_SIMD_functions <-  header_of_the_fast_names
        }
        if (!file.exists(elementwise_functions_header_of_the_SIMD_functions)) {
              stop(paste0(package_of_the_SIMD_functions, "'s header stan_external_functions.hpp is not in ",
                          headers_folder_of_the_SIMD_functions, "."))
        }
        ##
        ## ---- the instruction set, which this CPU must support:
        ##
        instruction_sets_supported <-  R_fn_BayesMVP_AVX_instruction_sets_supported_by_this_CPU()
        ## ("AVX2" or "AVX512")
        instruction_set_used <-  instruction_set_asked
        if (identical(instruction_set_asked, "AVX512_if_supported_else_AVX2")) {
              instruction_set_used <-  if (instruction_sets_supported[["BayesMVP_AVX512"]]) "AVX512" else "AVX2"
        }
        Stan_model_math_functions_used <-  paste0(package_of_the_SIMD_functions, "_", instruction_set_used)
        if (!instruction_sets_supported[[paste0("BayesMVP_", instruction_set_used)]]) {
              stop(paste0("This CPU does not support the instruction set of Stan_model_math_functions = \"",
                          Stan_model_math_functions_used, "\" (AVX2 needs the CPU flags avx2 and fma; ",
                          "AVX-512 also needs avx512f, avx512vl and avx512dq)."))
        }
        SIMD_lanes_expected <-  if (identical(instruction_set_used, "AVX512")) 8 else 4
        ##
        ## ---- the Stan model (the source of the AVX version; never an AVX version written by NicoStan itself):
        ##
        source_Stan_model_file_path <-  normalizePath(path = Stan_model_file_path, mustWork = TRUE)
        source_Stan_model_file_name <-  basename(source_Stan_model_file_path)
        first_line_of_source_Stan_model <-  readLines(con = source_Stan_model_file_path, n = 1, warn = FALSE)
        if (length(first_line_of_source_Stan_model) == 1 &&
            (startsWith(x = first_line_of_source_Stan_model, prefix = "// BayesMVP_AVX") ||
             startsWith(x = first_line_of_source_Stan_model, prefix = "// NicoSIMD_AVX")) &&
            grepl(pattern = "written by NicoStan", x = first_line_of_source_Stan_model, fixed = TRUE)) {
              stop(paste0(source_Stan_model_file_name, " is already an AVX version written by NicoStan; give ",
                          "the Stan model it was written from, with Stan_model_math_functions = \"",
                          Stan_model_math_functions, "\"."))
        }
        models_folder <-  dirname(source_Stan_model_file_path)
        if (file.access(names = models_folder, mode = 2) != 0) {
              stop(paste0("NicoStan writes the AVX version of the Stan model next to it (as BridgeStan writes ",
                          "the compiled model there), but its folder cannot be written to: ", models_folder))
        }
        ##
        ## ---- the declarations offered to stanc: every elementwise function, and each fused log density whose
        ##      header is in BayesMVP's header folder:
        ##
        BayesMVP_AVX_functions <-  fn_BayesMVP_AVX_functions_and_their_Stan_declarations()
        elementwise_functions <-  BayesMVP_AVX_functions$elementwise_functions
        header_file_of_each_fused_distribution <-  BayesMVP_AVX_functions$fused_distribution_BayesMVP_header_files
        fused_distribution_header_is_in_folder <-  file.exists(file.path(headers_folder_of_the_SIMD_functions,
                                                                         header_file_of_each_fused_distribution))
        fused_distributions_offered <-  names(header_file_of_each_fused_distribution)[
                                              fused_distribution_header_is_in_folder]
        fused_distribution_declarations_offered <-  BayesMVP_AVX_functions$fused_distribution_declarations[
                                                          fused_distributions_offered]
        # Stan_declarations_offered <-  c("    real bmvp_simd_lanes();",
        #                                 paste0("    vector fast_", elementwise_functions, "(vector x);"),
        #                                 unlist(fused_distribution_declarations_offered, use.names = FALSE))
        ## (with the declarations of the elementwise functions for each argument type, of the model's own
        ##  functions replaced and of std_normal as the fused normal, and the functions whose calls left on Stan
        ##  math are reported: a change of any of them writes the AVX version again)
        elementwise_function_argument_types <-  BayesMVP_AVX_functions$elementwise_function_argument_types
        Stan_declarations_offered <-  c("    real bmvp_simd_lanes();",
                                        as.vector(outer(X = elementwise_function_argument_types,
                                                        Y = elementwise_functions,
                                                        FUN = function(argument_type, elementwise_function) {

                                                              paste0("    ", argument_type, " fast_",
                                                                     elementwise_function, "(", argument_type,
                                                                     " x);")
                                                        })),

                                        unlist(fused_distribution_declarations_offered, use.names = FALSE),
                                        BayesMVP_AVX_functions$model_functions_replaced_by_BayesMVP$
                                              model_signature,
                                        BayesMVP_AVX_functions$model_functions_replaced_by_BayesMVP$
                                              BayesMVP_declaration,
                                        paste0("// std_normal as: ",
                                               BayesMVP_AVX_functions$std_normal_as_fused_normal_declarations),
                                        paste0("// calls left on Stan math reported: ",
                                               paste(names(BayesMVP_AVX_functions[[paste0(
                                                     "math_functions_whose_calls_left_on_Stan_math_",
                                                     "are_reported")]]),
                                                     collapse = ", "),
                                               "; ",
                                               paste(names(BayesMVP_AVX_functions[[paste0(
                                                     "distributions_whose_calls_left_on_Stan_math_",
                                                     "are_reported")]]),
                                                     collapse = ", ")))
        Stan_declarations_offered_md5 <-  digest::digest(object = paste(Stan_declarations_offered,
                                                                        collapse = "\n"),
                                                         algo = "md5",
                                                         serialize = FALSE)
        ##
        ## ---- the AVX version, written again only when the Stan model or the declarations offered change:
        ##
        source_Stan_model_md5 <-  unname(tools::md5sum(files = source_Stan_model_file_path))
        AVX_Stan_model_file_name <-  paste0(tools::file_path_sans_ext(source_Stan_model_file_name), "_",
                                            Stan_model_math_functions_used, ".stan")
        AVX_Stan_model_file_path <-  file.path(models_folder, AVX_Stan_model_file_name)
        first_line_of_AVX_Stan_model <-  paste0("// ", Stan_model_math_functions_used, " version of ",
                                                source_Stan_model_file_name, " written by NicoStan ",
                                                "(source model md5 ", source_Stan_model_md5,
                                                "; BayesMVP AVX function declarations md5 ",
                                                Stan_declarations_offered_md5, ")")
        AVX_version_is_up_to_date <-  file.exists(AVX_Stan_model_file_path) &&
                                      identical(readLines(con = AVX_Stan_model_file_path, n = 1, warn = FALSE),
                                                first_line_of_AVX_Stan_model)
        if (AVX_version_is_up_to_date) {

              ## (the BayesMVP functions it uses, from its second line)
              second_line_of_AVX_Stan_model <-  readLines(con = AVX_Stan_model_file_path, n = 2, warn = FALSE)[2]
              BayesMVP_functions_used_as_text <-  sub(pattern = "^// (BayesMVP|NicoSIMD) functions used: ",
                                                      replacement = "",
                                                      x = second_line_of_AVX_Stan_model)
              BayesMVP_functions_used <-  c()
              if (!identical(BayesMVP_functions_used_as_text, "none")) {
                    BayesMVP_functions_used <-  strsplit(x = BayesMVP_functions_used_as_text, split = ", ",
                                                         fixed = TRUE)[[1]]
              }
              message(colourise(paste0("Reusing the ", Stan_model_math_functions_used, " version of ",
                                       source_Stan_model_file_name, ": ", AVX_Stan_model_file_path),
                                "cyan"))

        } else {

              ## (the stanc of BridgeStan, which compiles the model; CmdStan's if BridgeStan has none)
              bridgestan_folder <-  utils::getFromNamespace("get_bridgestan_path", "bridgestan")(download = FALSE)
              stanc_file_path <-  file.path(bridgestan_folder, "bin", "stanc")
              if (!file.exists(stanc_file_path)) {
                    stanc_file_path <-  file.path(cmdstanr::cmdstan_path(), "bin", "stanc")
              }
              stanc_args_as_text <-  as.character(unlist(stanc_args, use.names = FALSE))
              stanc_include_paths_arguments <-  stanc_args_as_text[startsWith(x = stanc_args_as_text,
                                                                              prefix = "--include-paths")]
              message(colourise(paste0("Writing the ", Stan_model_math_functions_used, " version of ",
                                       source_Stan_model_file_name, " (", package_of_the_SIMD_functions,
                                       "'s SIMD functions where stanc ",
                                       "accepts them): ", AVX_Stan_model_file_path),
                                "cyan"))
              BayesMVP_functions_used <-  R_fn_write_AVX_version_of_Stan_model_with_BayesMVP_functions(
                    source_Stan_model_file_path      = source_Stan_model_file_path,
                    AVX_Stan_model_file_path         = AVX_Stan_model_file_path,
                    first_line_of_AVX_Stan_model     = first_line_of_AVX_Stan_model,
                    elementwise_functions            = elementwise_functions,
                    fused_distribution_declarations  = fused_distribution_declarations_offered,
                    stanc_file_path                  = stanc_file_path,
                    stanc_include_paths_arguments    = stanc_include_paths_arguments,
                    elementwise_function_argument_types      = elementwise_function_argument_types,
                    model_functions_replaced_by_BayesMVP     = BayesMVP_AVX_functions$
                                                                     model_functions_replaced_by_BayesMVP,
                    std_normal_as_fused_normal_declarations  = BayesMVP_AVX_functions$
                                                                     std_normal_as_fused_normal_declarations,
                    math_functions_whose_calls_left_on_Stan_math_are_reported = BayesMVP_AVX_functions$
                          math_functions_whose_calls_left_on_Stan_math_are_reported,
                    distributions_whose_calls_left_on_Stan_math_are_reported = BayesMVP_AVX_functions$
                          distributions_whose_calls_left_on_Stan_math_are_reported,
                    report_file_path_of_calls_left_on_Stan_math = paste0(
                          tools::file_path_sans_ext(AVX_Stan_model_file_path), "_calls_left_on_Stan_math.txt"))
              if (SIMD_functions_from_NicoSIMD) {
                    R_fn_put_NicoSIMD_names_in_place_of_BayesMVP_names_in_AVX_version_of_Stan_model(
                          AVX_Stan_model_file_path       = AVX_Stan_model_file_path,
                          NicoSIMD_headers_folder        = headers_folder_of_the_SIMD_functions,
                          stanc_file_path                = stanc_file_path,
                          stanc_include_paths_arguments  = stanc_include_paths_arguments)
              }

        }
        message(colourise(paste0(package_of_the_SIMD_functions, " functions used: ",
                                 if (length(BayesMVP_functions_used) > 0) {
                                       paste(BayesMVP_functions_used, collapse = ", ")
                                 } else "none (every call keeps Stan's own function)"),
                          "cyan"))
        ##
        ## ---- the user header of the AVX version: BayesMVP's elementwise functions, the header of each fused log
        ##      density used, and the user's own header (Stan_cpp_user_header or a USER_HEADER make argument):
        ##
        make_args_given <-  as.character(unlist(make_args, use.names = FALSE))
        make_arg_is_user_header <-  grepl(pattern = "^USER_HEADER=", x = make_args_given)
        user_headers_given <-  c(Stan_cpp_user_header,
                                 sub(pattern = "^USER_HEADER=", replacement = "",
                                     x = make_args_given[make_arg_is_user_header]))
        user_headers_given <-  unique(normalizePath(path = user_headers_given, mustWork = TRUE))
        if (length(user_headers_given) > 1) {
              stop(paste0("Stan_cpp_user_header and the USER_HEADER of make_args give different user headers (",
                          paste(user_headers_given, collapse = ", "), "); give the user header in one of the ",
                          "two only."))
        }
        fused_distributions_used <-  intersect(BayesMVP_functions_used,
                                               names(header_file_of_each_fused_distribution))
        fused_distribution_headers_used <-  file.path(headers_folder_of_the_SIMD_functions,
                                                      header_file_of_each_fused_distribution[
                                                            fused_distributions_used])
        AVX_user_header_lines <-  c("#pragma once",
                                    paste0("// User header of ", AVX_Stan_model_file_name,
                                           ", written by NicoStan: ", package_of_the_SIMD_functions,
                                           "'s SIMD functions",
                                           if (length(user_headers_given) == 1) " and the user's own header"),
                                    paste0("#include ",
                                           encodeString(x = elementwise_functions_header_of_the_SIMD_functions,
                                                        quote = "\"")))
        if (length(fused_distribution_headers_used) > 0) {
              AVX_user_header_lines <-  c(AVX_user_header_lines,
                                          paste0("#include ", encodeString(x = fused_distribution_headers_used,
                                                                           quote = "\"")))
        }
        if (length(user_headers_given) == 1) {
              AVX_user_header_lines <-  c(AVX_user_header_lines,
                                          paste0("#include ", encodeString(x = user_headers_given, quote = "\"")))
        }
        AVX_user_header_file <-  paste0(tools::file_path_sans_ext(AVX_Stan_model_file_path), "_user_header.hpp")
        AVX_user_header_is_up_to_date <-  file.exists(AVX_user_header_file) &&
                                          identical(readLines(con = AVX_user_header_file, warn = FALSE),
                                                    AVX_user_header_lines)
        if (!AVX_user_header_is_up_to_date) {

              ## (a compiled model of the AVX version was built with another user header, which make does not
              ##  track through NicoStan's wrapper header: it is moved aside, with the suffix
              ##  .built_with_previous_user_header, so that BridgeStan builds it again)
              AVX_model_so_file <-  paste0(tools::file_path_sans_ext(AVX_Stan_model_file_path), "_model.so")
              AVX_model_object_file <-  paste0(tools::file_path_sans_ext(AVX_Stan_model_file_path), ".o")
              if (file.exists(AVX_model_so_file) || file.exists(AVX_model_object_file)) {
                    ## message(colourise(paste0("The user header of ", AVX_Stan_model_file_name, " changed: its ",
                    ##                          "compiled model is removed and built again."),
                    ##                   "cyan"))
                    ## unlink(x = c(AVX_model_so_file, AVX_model_object_file))
                    message(colourise(paste0("The user header of ", AVX_Stan_model_file_name, " changed: its ",
                                             "compiled model is moved aside (.built_with_previous_user_header) ",
                                             "and built again."),
                                      "cyan"))
                    for (previous_build_file in c(AVX_model_so_file, AVX_model_object_file)) {
                          if (file.exists(previous_build_file)) {
                                file.rename(from = previous_build_file,
                                            to = paste0(previous_build_file, ".built_with_previous_user_header"))
                          }
                    }
              }
              temporary_file <-  tempfile(pattern = "AVX_user_header_being_written_", tmpdir = models_folder,
                                          fileext = ".hpp.tmp")
              writeLines(text = AVX_user_header_lines, con = temporary_file)
              if (!file.rename(from = temporary_file, to = AVX_user_header_file)) {
                    stop(paste0("Could not write the user header of the AVX version: ", AVX_user_header_file))
              }

        }
        ##
        ## ---- the make arguments of the instruction set (those of NicoStan's examples):
        ##
        make_args_without_user_header <-  make_args_given[!make_arg_is_user_header]
        names_of_make_args_given <-  sub(pattern = "=.*$", replacement = "", x = make_args_without_user_header)
        make_arg_names_set_for_instruction_set <-  c("CXXFLAGS_OPTIM", "CPPFLAGS_OPTIM", "PRECOMPILED_HEADERS")
        make_arg_is_set_for_instruction_set <-  (names_of_make_args_given %in%
                                                 make_arg_names_set_for_instruction_set)
        if (any(make_arg_is_set_for_instruction_set)) {
              stop(paste0("make_args hold ",
                          paste(make_args_without_user_header[make_arg_is_set_for_instruction_set],
                                collapse = ", "),
                          ", which Stan_model_math_functions = \"", Stan_model_math_functions, "\" sets ",
                          "itself (CXXFLAGS_OPTIM with the instruction-set flags, ",
                          "CPPFLAGS_OPTIM=-DBAYESMVP_FORCE_AVX2 for AVX2, PRECOMPILED_HEADERS=false); remove ",
                          "them from make_args."))
        }
        instruction_set_flags <-  if (identical(instruction_set_used, "AVX512")) {
              "-march=native -mtune=native -mfma -mavx2 -mavx512f -mavx512vl -mavx512dq"
        } else "-march=native -mtune=native -mfma -mavx2"
        make_args_of_AVX_version <-  make_args_without_user_header
        if (!("STAN_THREADS" %in% names_of_make_args_given)) {
              make_args_of_AVX_version <-  c(make_args_of_AVX_version, "STAN_THREADS=true")
        }
        make_args_of_AVX_version <-  c(make_args_of_AVX_version,
                                       "PRECOMPILED_HEADERS=false",
                                       paste0("CXXFLAGS_OPTIM=", shQuote(instruction_set_flags)))
        if (identical(instruction_set_used, "AVX2")) {
              make_args_of_AVX_version <-  c(make_args_of_AVX_version, "CPPFLAGS_OPTIM=-DBAYESMVP_FORCE_AVX2")
        }
        ##
        return(list( Stan_model_math_functions        = Stan_model_math_functions,
                     Stan_model_math_functions_used   = Stan_model_math_functions_used,
                     source_Stan_model_file_path      = source_Stan_model_file_path,
                     Stan_model_file_path             = AVX_Stan_model_file_path,
                     Stan_cpp_user_header             = AVX_user_header_file,
                     make_args                        = as.list(make_args_of_AVX_version),
                     package_of_the_SIMD_functions    = package_of_the_SIMD_functions,
                     headers_folder_of_the_SIMD_functions  = headers_folder_of_the_SIMD_functions,
                     BayesMVP_AVX_headers_folder      = BayesMVP_AVX_headers_folder,
                     BayesMVP_functions_used          = BayesMVP_functions_used,
                     SIMD_lanes_expected              = SIMD_lanes_expected))

}




#### ---- NicoSIMD's names in the AVX version of a Stan model written with BayesMVP's names ---------------------
#' Put NicoSIMD's names (nico_) in place of BayesMVP's (fast_) in the AVX version of a Stan model
#'
#' NicoSIMD holds BayesMVP's Stan-facing SIMD functions, with the same C++ and the names changed from fast_ to
#' nico_; its file names_renamed_from_fast_to_nico.txt lists every name changed. In the AVX version written with
#' BayesMVP's names, each name of that list, as a whole name (never inside a longer name), becomes its nico_ name,
#' in every line but the first (which records the source model and the declarations offered); the second line then
#' names NicoSIMD's functions. The AVX version is then checked again with stanc.
#' @param AVX_Stan_model_file_path The AVX version, written with BayesMVP's names.
#' @param NicoSIMD_headers_folder The folder of NicoSIMD's headers (with names_renamed_from_fast_to_nico.txt).
#' @param stanc_file_path The stanc compiler.
#' @param stanc_include_paths_arguments The --include-paths arguments of the model's stanc arguments.
#' @return The lines of the AVX version with NicoSIMD's names (invisibly).
#' @keywords internal
R_fn_put_NicoSIMD_names_in_place_of_BayesMVP_names_in_AVX_version_of_Stan_model <-  function(
                                                                          AVX_Stan_model_file_path,
                                                                          NicoSIMD_headers_folder,
                                                                          stanc_file_path,
                                                                          stanc_include_paths_arguments
) {

        file_of_the_names_renamed <-  file.path(NicoSIMD_headers_folder, "names_renamed_from_fast_to_nico.txt")
        if (!file.exists(file_of_the_names_renamed)) {
              stop(paste0("NicoSIMD's list of the names changed from fast_ to nico_ is not in ",
                          NicoSIMD_headers_folder, ": reinstall NicoSIMD."))
        }
        BayesMVP_names <-  readLines(con = file_of_the_names_renamed, warn = FALSE)
        BayesMVP_names <-  BayesMVP_names[nzchar(BayesMVP_names)]
        ## (the longest first, so that a name is never changed inside a longer one)
        BayesMVP_names <-  BayesMVP_names[order(nchar(BayesMVP_names), decreasing = TRUE)]
        ##
        lines_of_AVX_version <-  readLines(con = AVX_Stan_model_file_path, warn = FALSE)
        lines_after_the_first <-  lines_of_AVX_version[-1]
        for (BayesMVP_name in BayesMVP_names) {
              NicoSIMD_name <-  sub(pattern = "^fast_", replacement = "nico_", x = BayesMVP_name)
              lines_after_the_first <-  gsub(pattern = paste0("(?<![A-Za-z0-9_.])", BayesMVP_name,
                                                              "(?![A-Za-z0-9_])"),
                                             replacement = NicoSIMD_name,
                                             x = lines_after_the_first,
                                             perl = TRUE)
        }
        lines_after_the_first[1] <-  sub(pattern = "^// BayesMVP functions used: ",
                                         replacement = "// NicoSIMD functions used: ",
                                         x = lines_after_the_first[1])
        lines_of_AVX_version <-  c(lines_of_AVX_version[1], lines_after_the_first)
        ##
        ## ---- written through a temporary file next to it, then checked with stanc:
        ##
        temporary_file <-  tempfile(pattern = "AVX_version_with_NicoSIMD_names_",
                                    tmpdir = dirname(AVX_Stan_model_file_path), fileext = ".stan.tmp")
        writeLines(text = lines_of_AVX_version, con = temporary_file)
        if (!file.rename(from = temporary_file, to = AVX_Stan_model_file_path)) {
              stop(paste0("Could not write the AVX version with NicoSIMD's names: ", AVX_Stan_model_file_path))
        }
        stanc_output_file <-  tempfile(pattern = "AVX_version_with_NicoSIMD_names_", fileext = ".hpp")
        stanc_output <-  suppressWarnings(system2( command = stanc_file_path,
                                                   args    = c("--allow-undefined",
                                                               stanc_include_paths_arguments,
                                                               paste0("--o=", shQuote(stanc_output_file)),
                                                               shQuote(AVX_Stan_model_file_path)),
                                                   stdout  = TRUE,
                                                   stderr  = TRUE))
        if (!is.null(attr(stanc_output, "status")) || !file.exists(stanc_output_file)) {
              stop(paste0("stanc does not accept the AVX version with NicoSIMD's names (",
                          AVX_Stan_model_file_path, "):\n", paste(stanc_output, collapse = "\n")))
        }
        ##
        return(invisible(lines_of_AVX_version))

}




#### ---- the SIMD lanes which the compiled AVX version of a Stan model reports ---------------------------------
#' Check that the compiled AVX version of a Stan model reports the SIMD lanes of its instruction set
#'
#' Reads the generated quantity simd_lanes (bmvp_simd_lanes(): 8 with BayesMVP's AVX-512 kernels, 4 with the AVX2
#' kernels) of the compiled model, with BridgeStan, and stops when it differs from the instruction set's. The
#' generated quantities are evaluated at 0 on the unconstrained scale, or, if they cannot be evaluated there, at
#' the points where every coordinate is 0.5, -0.5, 1 or -1.
#' @param bs_model The BridgeStan model of the compiled AVX version.
#' @param SIMD_lanes_expected 8 (AVX-512) or 4 (AVX2).
#' @param Stan_model_math_functions_used "NicoSIMD_AVX512", "NicoSIMD_AVX2", "BayesMVP_AVX512" or
#'   "BayesMVP_AVX2".
#' @param AVX_Stan_model_file_path The AVX version (for the messages).
#' @return The SIMD lanes reported.
#' @keywords internal
R_fn_check_SIMD_lanes_reported_by_compiled_BayesMVP_AVX_Stan_model <-  function( bs_model,
                                                                                SIMD_lanes_expected,
                                                                                Stan_model_math_functions_used,
                                                                                AVX_Stan_model_file_path
) {

        AVX_Stan_model_file_name <-  basename(AVX_Stan_model_file_path)
        parameter_and_generated_quantity_names <-  bs_model$param_names(include_tp = FALSE, include_gq = TRUE)
        position_of_simd_lanes <-  which(parameter_and_generated_quantity_names == "simd_lanes")
        if (length(position_of_simd_lanes) != 1) {
              stop(paste0("The compiled model of ", AVX_Stan_model_file_name, " has no generated quantity ",
                          "simd_lanes, so its SIMD lanes cannot be checked."))
        }
        n_unconstrained <-  bs_model$param_unc_num()
        points_tried <-  list(rep(0, n_unconstrained),
                              rep(0.5, n_unconstrained),
                              rep(-0.5, n_unconstrained),
                              rep(1, n_unconstrained),
                              rep(-1, n_unconstrained))
        ##
        SIMD_lanes_reported <-  NA
        evaluation_errors <-  c()
        for (point in points_tried) {

              constrained_values <-  tryCatch(expr = bs_model$param_constrain( theta_unc   = point,
                                                                               include_tp  = FALSE,
                                                                               include_gq  = TRUE,
                                                                               rng         = bs_model$new_rng(
                                                                                                   seed = 1)),
                                              error = function(condition) conditionMessage(condition))
              if (is.numeric(constrained_values)) {

                    SIMD_lanes_reported <-  constrained_values[position_of_simd_lanes]
                    break
              }

              evaluation_errors <-  c(evaluation_errors, constrained_values)

        }
        if (is.na(SIMD_lanes_reported)) {
              stop(paste0("The generated quantities of ", AVX_Stan_model_file_name, " could not be evaluated at ",
                          "any of the points tried, so its SIMD lanes cannot be checked:\n",
                          paste(unique(evaluation_errors), collapse = "\n")))
        }
        ##
        if (SIMD_lanes_reported != SIMD_lanes_expected) {
              stop(paste0("The compiled model of ", AVX_Stan_model_file_name, " reports ", SIMD_lanes_reported,
                          " SIMD lanes, but ", Stan_model_math_functions_used, " needs ", SIMD_lanes_expected,
                          ": it was not built with the kernels of this instruction set (e.g. a model library ",
                          "built earlier with other flags). Remove ",
                          paste0(tools::file_path_sans_ext(AVX_Stan_model_file_path), "_model.so"),
                          " to build it again."))
        }
        message(colourise(paste0(AVX_Stan_model_file_name, ": the compiled model reports ", SIMD_lanes_reported,
                                 " SIMD lanes, as ", Stan_model_math_functions_used, " needs."),
                          "green"))
        ##
        return(SIMD_lanes_reported)

}




















