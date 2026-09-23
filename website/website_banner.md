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
NicoStan uses standard HMC for the main model parameters,
and diffusion-pathspace HMC for the high-dimensional Gaussian latent variables
(or "nuisance" parameters), which many models require, such as:
binary/ordinal multivariate probit models
(used in e.g., econometrics and diagnostic/screening test accuracy without a gold standard),
and stochastic volatility models (used in e.g., quantitative finance, econometrics, medicine).
## ----
It also uses between-chain adaptation to achieve more efficient burnin/warmup,
and state-of-the-art burnin algorithms (such as ChEES-R-HMC and SNAPER-HMC).
## ----
NicoStan also offers rapid parallel estimation of posterior summaries,
and only monitors/stores the trace for the main model parameters (by default).
## ----
Note that NicoStan also works for general Stan models without nuisance parameters.
