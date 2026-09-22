\name{fn_create_model_backend}
\alias{fn_create_model_backend}
\title{Bind the shared sampler interface to a model package}
\usage{
fn_create_model_backend(provider, model_types = "Stan")
}
\arguments{
\item{provider}{An environment containing the model package's native wrappers and model helpers.}
\item{model_types}{The model types supported by this provider.}
}
\description{Creates isolated lexical bindings for NicoStan's maintained R sampler and R6 model classes.}
\details{Native wrappers retain their owning namespace. The shared R function bodies resolve those wrappers through a provider-local environment. Gradient evaluations remain in compiled code.}
\value{An environment containing shared R functions, provider operations, and the model and plotting R6 generators.}
