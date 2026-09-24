## ----
## Website banner text. The lines above the --- line are the big title; the lines below it are the
## paragraph under the title. Each line here is one line on the website. Lines starting with ## are ignored.
## ----
## The "description:" line is the one-line package description, used for the GitHub "About" box and the
## website's search-engine description.
## ----
description: Adaptive MCMC for Stan models, with advanced between-chain adaptation and diffusion-pathspace HMC
## ----
NicoStan:
Adaptive MCMC
for Stan models,
with advanced
between-chain
adaptation and
diffusion-pathspace HMC
---
## ---------------------------------------------------------------------------------
NicoStan uses standard HMC for the main model parameters,
and diffusion-pathspace HMC for the high-dimensional Gaussian latent variables
(or "nuisance" parameters), which many models require, such as:
binary/ordinal multivariate probit models
(e.g., for econometrics and diagnostic test accuracy with imperfect gold standards),
and stochastic volatility (e.g., for quantitative finance, econometrics, medicine).
## ---------------------------------------------------------------------------------
It also uses between-chain adaptation to achieve more efficient burnin/warmup,
as well as state-of-the-art burnin algorithms (such as ChEES-R-HMC, SNAPER-HMC).
## ---------------------------------------------------------------------------------
NicoStan also offers rapid parallel estimation of posterior summaries,
and only monitors/stores the trace for the main model parameters (by default).
## ---------------------------------------------------------------------------------
Note that NicoStan also works for general Stan models without nuisance parameters.
