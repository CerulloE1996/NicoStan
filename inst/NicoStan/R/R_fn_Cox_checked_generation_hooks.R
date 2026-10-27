##
## ---- Match the direct model call ---------------------------------------------------------------------------------------------------
##
R_fn_Cox_source_uses_checked_indices <- function(
      code) {

      node <- paste0(
            "nico_dot_products_of_affine_predictor_with_indexed_effects_",
            "and_log_cumulative_sum_exp_with_one_vector_of_N_values")
      return(grepl(
            pattern = paste0(
                  "\\bmodel\\s*\\{[\\s\\S]*?\\btarget\\s*\\+=\\s*",
                  node,
                  "\\s*\\("),
            x = code,
            perl = TRUE))

}

##
## ---- Prepare the model-specific generation functions -------------------------------------------------------------------------------
##
R_fn_Cox_checked_generation_hooks <- function(
      generate_Stan_model,
      patch_map_generation,
      check_generated_model) {

      ##
      ## ---- Require the existing function interfaces --------------------------------------------------------------------------------
      ##
      if (!identical(
                  x = names(formals(generate_Stan_model)),
                  y = c(
                        "arguments",
                        "automatic_header")) ||
          !identical(
                  x = names(formals(patch_map_generation)),
                  y = c(
                        "runtime_path",
                        "automatic_header")) ||
          !identical(
                  x = names(formals(check_generated_model)),
                  y = "code") ||
          !identical(
                  x = environment(generate_Stan_model),
                  y = environment(patch_map_generation))) {
            stop("The source-generation function interfaces are not recognised.")
      }
      generation_parts <- as.list(body(generate_Stan_model))
      map_statement <- quote(
            if (!is.null(automatic_header)) {
                  R_fn_bridge_memory_patch_generated_map_calls(
                        generated_file,
                        automatic_header,
                        code)
            })
      status_statement <- quote(
            if (status != 0) stop(paste0(
                  "Stan generation failed with exit ",
                  status,
                  ".")))
      map_position <- which(vapply(
            X = generation_parts,
            FUN = identical,
            FUN.VALUE = logical(1L),
            y = map_statement))
      ##
      ## ---- Require the post-compiler boundary --------------------------------------------------------------------------------------
      ##
      if (length(map_position) != 1L || map_position < 3L ||
          !identical(
                  x = generation_parts[[map_position - 1L]],
                  y = status_statement)) {
            stop("The post-compiler source-generation boundary is not recognised.")
      }
      checked_step <- quote({
            generated_cpp <- readChar(
                  con = generated_file,
                  nchars = file.info(generated_file)$size,
                  useBytes = TRUE)
            checked_cpp <- R_fn_Cox_checked_indices_in_generated_model(
                  code = generated_cpp)
            ##
            ## ---- Write only a changed model ----------------------------------------------------------------------------------------
            ##
            if (!identical(
                        x = generated_cpp,
                        y = checked_cpp)) {
                  writeChar(
                        object = checked_cpp,
                        con = generated_file,
                        nchars = nchar(
                              x = checked_cpp,
                              type = "bytes"),
                        eos = NULL,
                        useBytes = TRUE)
            }
      })
      serializer_parts <- as.list(body(patch_map_generation))
      functions_statement <- quote(
            functions <- list(
                  R_fn_Stan_parallel_call_names = R_fn_Stan_parallel_call_names,
                  R_fn_read_Stan_source_with_includes = R_fn_read_Stan_source_with_includes,
                  R_fn_Stan_map_type_in_helper_functions = R_fn_Stan_map_type_in_helper_functions,
                  R_fn_bridge_memory_patch_generated_map_calls = R_fn_bridge_memory_patch_generated_map_calls,
                  R_fn_bridge_memory_generate_Stan_model = R_fn_bridge_memory_generate_Stan_model))
      functions_position <- which(vapply(
            X = serializer_parts,
            FUN = identical,
            FUN.VALUE = logical(1L),
            y = functions_statement))
      ##
      ## ---- Require the existing helper list ----------------------------------------------------------------------------------------
      ##
      if (length(functions_position) != 1L) {
            stop("The serialized source-generation helpers are not recognised.")
      }
      hooks_environment <- new.env(
            parent = environment(generate_Stan_model))
      hooks_environment$R_fn_Cox_checked_indices_in_generated_model <- check_generated_model
      checked_generation <- generate_Stan_model
      body(checked_generation) <- as.call(append(
            x = generation_parts,
            values = list(checked_step),
            after = map_position - 1L))
      environment(checked_generation) <- hooks_environment
      hooks_environment$R_fn_bridge_memory_generate_Stan_model <- checked_generation
      functions_assignment <- serializer_parts[[functions_position]]
      helpers <- functions_assignment[[3L]]
      helpers[["R_fn_Cox_checked_indices_in_generated_model"]] <-
            as.name("R_fn_Cox_checked_indices_in_generated_model")
      functions_assignment[[3L]] <- helpers
      serializer_parts[[functions_position]] <- functions_assignment
      checked_serializer <- patch_map_generation
      body(checked_serializer) <- as.call(serializer_parts)
      environment(checked_serializer) <- hooks_environment
      return(list(
            generate_Stan_model = checked_generation,
            patch_map_generation = checked_serializer))

}
