##
## ---- Backend argument routing without model construction:
##

workspace <-  path.expand("~/Documents/Work/PhD_work")
R_folder <-  file.path(workspace, "R_packages", "NicoStan", "inst", "NicoStan", "R")
initialiser_file <-  file.path(R_folder, "R_fn_initialise_model.R")
writer_file <-  file.path(R_folder, "R_fn_Stan_model_math_functions_BayesMVP_AVX.R")

fn_assignments_named <-  function(node,
                                  name) {

      matches <-  list()

      if (is.call(node)) {

            if (identical(node[[1]], as.name("<-")) && identical(node[[2]], as.name(name))) {
                  matches <-  list(node)
            }

            children <-  if (identical(node[[1]], as.name("function"))) {
                  list(node[[3]])
            } else as.list(node)[-1]

      } else if (is.expression(node)) {

            children <-  as.list(node)

      } else {

            return(matches)

      }

      for (i in seq_along(children)) {

            if (!identical(children[[i]], quote(expr = ))) {
                  matches <-  c(matches, fn_assignments_named(children[[i]], name))
            }

      }

      return(matches)

}

cat("[0/4] Parse current initializer and writer\n")
initialiser <-  parse(file = initialiser_file)
writer <-  parse(file = writer_file)
predicate <-  fn_assignments_named(initialiser, "Stan_model_math_functions_uses_NicoSIMD")
converter_calls <-  fn_assignments_named(initialiser, "BayesMVP_AVX_version_of_Stan_model")
converter_name <-  "R_fn_write_BayesMVP_AVX_version_of_Stan_model_and_user_header"
converter_calls <-  Filter(function(x) {

      return(is.call(x[[3]]) && identical(x[[3]][[1]], as.name(converter_name)))

}, converter_calls)
converter <-  fn_assignments_named(writer, converter_name)
stopifnot(length(predicate) == 1, length(converter_calls) == 1, length(converter) == 1)

##
## ---- Use the real validator, but no conversion or model evaluation:
##

converter_body <-  as.list(converter[[1]][[3]][[3]])[-1]
stopifnot(identical(converter_body[[1]][[2]], as.name("Stan_model_math_functions_with_BayesMVP_AVX")),
          identical(converter_body[[2]][[2]], as.name("Stan_model_math_functions_with_NicoSIMD")))
mock_body <-  as.call(c(list(as.name("{")), converter_body[1:4], list(quote(as.list(environment())))))
mock_converter <-  as.function(c(as.list(converter[[1]][[3]][[2]]), list(mock_body)))

fn_route_arguments <-  function(backend,
                                  folder) {

      environment_of_call <-  list2env(list(
            Stan_model_math_functions = backend,
            Stan_model_file_path = "model-sentinel",
            Stan_cpp_user_header = NULL,
            stanc_args = NULL,
            make_args = NULL,
            BayesMVP_AVX_headers_folder = folder,
            R_fn_write_BayesMVP_AVX_version_of_Stan_model_and_user_header = mock_converter))

      eval(predicate[[1]], envir = environment_of_call)
      eval(converter_calls[[1]], envir = environment_of_call)
      return(environment_of_call$BayesMVP_AVX_version_of_Stan_model)

}

cat("[1/4] Exact routing expressions and validator isolated\n")
accepted_backends <-  c("NicoSIMD_AVX2", "NicoSIMD_AVX512", "NicoSIMD_AVX512_if_supported_else_AVX2",
                        "BayesMVP_AVX2", "BayesMVP_AVX512", "BayesMVP_AVX512_if_supported_else_AVX2")

for (backend in accepted_backends) {

      for (folder in list("selected-folder", NULL)) {

            result <-  fn_route_arguments(backend, folder)
            native <-  startsWith(backend, "NicoSIMD_")
            stopifnot(identical(result$Stan_model_math_functions, backend),
                      identical(result$Stan_model_file_path, "model-sentinel"),
                      identical(result$NicoSIMD_headers_folder, if (native) folder else NULL),
                      identical(result$BayesMVP_AVX_headers_folder, if (native) NULL else folder))
            cat(paste0("PASS ", backend, ": ", if (is.null(folder)) "NULL" else "selected folder", "\n"))

      }

}

cat("[2/4] All supported labels routed correctly\n")
invalid_backends <-  list(native_unknown = "NicoSIMD_bad",
                          unknown = "bad",
                          missing = NA_character_,
                          empty = character(0),
                          multiple = c("NicoSIMD_AVX2", "NicoSIMD_AVX512"),
                          numeric = 1)

for (name in names(invalid_backends)) {

      error <-  tryCatch({

            fn_route_arguments(invalid_backends[[name]], "selected-folder")
            NULL

      }, error = identity)
      stopifnot(inherits(error, "error"),
                grepl("Stan_model_math_functions must be one of", conditionMessage(error), fixed = TRUE))
      cat(paste0("PASS rejected ", name, "\n"))

}

cat("[3/4] Invalid inputs rejected by the current validator\n")
cat("[4/4] PASS: routing only; no model, library, fit, build or timing evaluated\n")






















