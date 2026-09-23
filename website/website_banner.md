## Website banner text. The lines above the --- line are the big title; the lines below it are the
## paragraph under the title. Each line here is one line on the website. Lines starting with ## are ignored.
## The description: line is the one-line package description, used for the GitHub "About" box and the
## website's search-engine description.
description: Adaptive Hamiltonian Monte Carlo for general Stan models
NicoStan:
Adaptive HMC
for Stan models.
---
NicoStan uses standard HMC for the main model parameters,
and diffusion-pathspace HMC for high-dimensional nuisance parameters/Gaussian latent variables.
NicoStan also uses adaptive between-chain adaptation to achieve rapid burnin/warmup, 
and state-of-the-art burnin algorithms with a variety of options, such as ChEES-R-HMC and SNAPER-HMC.
NicoStan also offers rapid parallel estimation of posterior summaries, and only monitors and stores the trace for the main model parameters (by default).
NicoStan also works for general Stan models without nuisance parameters.
