#### =====================================================================================================================================
## R_fn_model_backend.R
##
## ---- Bind the shared R implementation to a package's existing native/model operations ------------------------------------------------
##
.nicostan_model_types <-  "Stan"

#' Create a package-local interface to the common sampler
#'
#' The provider is the model package namespace. Shared R function bodies remain
#' in NicoStan; their enclosing environment supplies the provider's existing
#' native wrappers and model helpers. Gradient evaluations remain in C++.
#'
#' @param provider An environment containing the model package operations.
#' @param model_types Model types implemented by the provider.
#' @return An environment containing the shared functions and R6 generators.
#' @export
fn_create_model_backend <-  function( provider,
                                      model_types = "Stan") {

        stopifnot(is.environment(provider), is.character(model_types), length(model_types) > 0L)
        core_namespace <-  environment(fn_create_model_backend)
        backend <-  new.env(parent = core_namespace)
        ##
        ## Provider closures retain their own namespace, including registered .Call symbols.
        ##
        for (name in ls(provider, all.names = TRUE)) {
                value <-  get(name, envir = provider, inherits = FALSE)
                if (is.function(value)) assign(name, value, envir = backend)
        }
        ##
        ## The source-generated list excludes Rcpp wrappers and package load hooks.
        ## Each backend gets its own lexical bindings; there is no global provider switch.
        ##
        for (name in .nicostan_shared_function_names) {
                if (exists(name, envir = backend, inherits = FALSE)) next
                value <-  get(name, envir = core_namespace, inherits = FALSE)
                environment(value) <-  backend
                assign(name, value, envir = backend)
        }
        backend$.nicostan_model_types <-  model_types
        ##
        fn_bind_class <-  function( prototype) {

                public_methods <-  prototype$public_methods
                public_methods$clone <-  NULL
                return(R6::R6Class(classname = prototype$classname,
                                  public = c(prototype$public_fields, public_methods),
                                  private = c(prototype$private_fields, prototype$private_methods),
                                  active = prototype$active,
                                  lock_objects = prototype$lock_objects,
                                  class = prototype$class,
                                  portable = prototype$portable,
                                  lock_class = prototype$lock_class,
                                  cloneable = prototype$cloneable,
                                  parent_env = backend))

        }
        backend$MVP_model <-  fn_bind_class(prototype = MVP_model)
        backend$MVP_plot_and_diagnose <-  fn_bind_class(prototype = MVP_plot_and_diagnose)
        return(backend)

}
