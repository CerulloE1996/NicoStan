#### =====================================================================================================================================
## R_fn_check_sample_math_settings.R
##
## ---- $sample() / R_fn_sample_model math settings (vect_type, Phi_type, inv_Phi_type) - 2026-09-22:
##
##      These three arguments of $sample() / R_fn_sample_model used to be SILENTLY IGNORED:
##        - vect_type was overwritten with detect_vectorization_support() whatever the user passed;
##        - Phi_type / inv_Phi_type were only defaulted;
##        - all three were then handed to init_and_run_burnin_ChESSR, which never used them, and nothing wrote
##          them into the model arguments.
##      Where they actually live:
##        - Model_type = "Stan": nowhere. The .stan file defines its own maths (e.g. its own Phi / Phi_approx
##          choice, often passed as Stan data), so these settings cannot have any effect.
##        - built-in (native) models: model_args_list$vect_type / $Phi_type / $inv_Phi_type, read at MODEL
##          INITIALISATION by BayesMVP's init_hard_coded_model_args, which validates them
##          (fn_check_native_model_vect_types_and_Phi_types) and resolves the defaults.
##      So: an explicit (non-NULL) value for a Stan model stops; for a built-in model it must equal the value
##      that initialisation resolved into model_args_list, otherwise it stops and points to model_args_list.
##      NULL (the default) changes nothing.
##



#' fn_stop_if_sample_math_settings_supplied_for_Stan_model
#'
#' Stops when vect_type, Phi_type or inv_Phi_type is supplied (non-NULL) to $sample() / R_fn_sample_model for a
#' Model_type = "Stan" model, where these settings are not used (the .stan file defines its own maths).
#' @param Model_type the model type of the initialised model.
#' @param vect_type,Phi_type,inv_Phi_type the values supplied to $sample() / R_fn_sample_model (NULL = not supplied).
#' @return invisible(TRUE) if nothing was supplied (or the model is not a Stan model).
fn_stop_if_sample_math_settings_supplied_for_Stan_model <- function( Model_type,
                                                                     vect_type    = NULL,
                                                                     Phi_type     = NULL,
                                                                     inv_Phi_type = NULL) {

        ##
        if (!identical(x = Model_type, y = "Stan")) return(invisible(TRUE))
        ##
        requested_math_settings_list <- list( vect_type    = vect_type,
                                              Phi_type     = Phi_type,
                                              inv_Phi_type = inv_Phi_type)
        ##
        supplied_math_setting_names <- names(Filter(f = Negate(is.null), x = requested_math_settings_list))
        ##
        if (length(supplied_math_setting_names) > 0) {

              supplied_math_settings_as_text <- paste0( vapply( X = supplied_math_setting_names,
                                                                FUN = function(math_setting_name) {
                                                                  paste0( math_setting_name, " = '",
                                                                          paste(as.character(requested_math_settings_list[[math_setting_name]]), collapse = ", "),
                                                                          "'")
                                                                },
                                                                FUN.VALUE = character(1)),
                                                        collapse = ", ")
              ##
              stop(paste0( "$sample() / R_fn_sample_model: ", supplied_math_settings_as_text, " was supplied for a Model_type = 'Stan' ",
                           "model, but vect_type, Phi_type and inv_Phi_type are NOT used for Stan models - the .stan file defines its own ",
                           "maths (e.g. choose Phi vs Phi_approx inside the Stan program or through its Stan data), so this setting would be ",
                           "silently ignored. Remove ", paste(supplied_math_setting_names, collapse = ", "), " from the call (leave them NULL). ",
                           "They only apply to BayesMVP's built-in models, where they are set in model_args_list at model initialisation."))

        }
        ##
        return(invisible(TRUE))

}




#' fn_check_sample_math_settings_match_native_model_args
#'
#' For a BUILT-IN (native) model, checks that every vect_type / Phi_type / inv_Phi_type supplied (non-NULL) to
#' $sample() / R_fn_sample_model equals the value that model initialisation resolved into model_args_list (the
#' value BayesMVP's init_hard_coded_model_args validated and the C++ model actually receives). A differing value
#' stops and points the user to model_args_list, because $sample() cannot change these settings.
#' @param Model_type the model type of the initialised model.
#' @param model_args_list_in_effect init_object$model_args_list of the (re-)initialised model.
#' @param vect_type,Phi_type,inv_Phi_type the values supplied to $sample() / R_fn_sample_model (NULL = not supplied).
#' @return invisibly, a named list of the effective values (NULL for Stan models).
fn_check_sample_math_settings_match_native_model_args <- function( Model_type,
                                                                   model_args_list_in_effect,
                                                                   vect_type    = NULL,
                                                                   Phi_type     = NULL,
                                                                   inv_Phi_type = NULL) {

        ##
        if (identical(x = Model_type, y = "Stan")) return(invisible(NULL))
        ##
        requested_math_settings_list <- list( vect_type    = vect_type,
                                              Phi_type     = Phi_type,
                                              inv_Phi_type = inv_Phi_type)
        ##
        effective_math_settings_list <- list( vect_type    = model_args_list_in_effect[["vect_type"]],
                                              Phi_type     = model_args_list_in_effect[["Phi_type"]],
                                              inv_Phi_type = model_args_list_in_effect[["inv_Phi_type"]])
        ##
        for (math_setting_name in names(requested_math_settings_list)) {

              requested_math_setting_value <- requested_math_settings_list[[math_setting_name]]
              effective_math_setting_value <- effective_math_settings_list[[math_setting_name]]
              ##
              if (is.null(requested_math_setting_value)) next
              ##
              requested_value_is_one_string <- is.character(requested_math_setting_value) && (length(requested_math_setting_value) == 1)
              ##
              if (!requested_value_is_one_string ||
                  is.null(effective_math_setting_value) ||
                  !identical(x = requested_math_setting_value, y = as.character(effective_math_setting_value))) {

                    stop(paste0( "$sample() / R_fn_sample_model: ", math_setting_name, " = '",
                                 paste(as.character(requested_math_setting_value), collapse = ", "), "' was supplied, but for the built-in ",
                                 "models ", math_setting_name, " is set at MODEL INITIALISATION through model_args_list$", math_setting_name,
                                 " (validated by BayesMVP's init_hard_coded_model_args), not by $sample(). The initialised model uses ",
                                 math_setting_name, " = '",
                                 if (is.null(effective_math_setting_value)) "NULL" else paste(as.character(effective_math_setting_value), collapse = ", "),
                                 "' (Model_type = '", Model_type, "'), so this request would be silently ignored. Set model_args_list$",
                                 math_setting_name, " = '", paste(as.character(requested_math_setting_value), collapse = ", "),
                                 "' in MVP_model$new(model_args_list = ...) (or pass that model_args_list to $sample(model_args_list = ...)), ",
                                 "and leave the $sample() argument NULL."))

              }

        }
        ##
        supplied_math_setting_names <- names(Filter(f = Negate(is.null), x = requested_math_settings_list))
        ##
        if (length(supplied_math_setting_names) > 0) {
              message(paste0( "$sample() math settings match the initialised model (Model_type = '", Model_type, "'): ",
                              paste0( supplied_math_setting_names, " = '",
                                      vapply( X = supplied_math_setting_names,
                                              FUN = function(math_setting_name) as.character(effective_math_settings_list[[math_setting_name]]),
                                              FUN.VALUE = character(1)),
                                      "'",
                                      collapse = ", ")))
        }
        ##
        return(invisible(effective_math_settings_list))

}
