
<!-- ------------------------------------------------------------------------------------------------------------------------------- -->
# NicoStan
<!-- ------------------------------------------------------------------------------------------------------------------------------- -->

[Installation](#installation) ·
[Examples](#examples) ·
[Benchmarks](#benchmarks) ·
[Models](#models-with-nuisance-parameters-diffusion-pathspace-hmc) ·
[NicoStan's efficient burnin algorithms](#efficient-burnin-algorithms) ·
[How to cite](#how-to-cite-nicostan) ·
[References](#references)


<!-- ------------------------------------------------------------------------------------------------------------------------------- -->
## What is NicoStan, and how is it different to Stan (e.g., cmdstanr or rstan)?
<!-- ------------------------------------------------------------------------------------------------------------------------------- -->


NicoStan is an R package for fitting Bayesian models written in the probabilistic programming language [Stan](https://mc-stan.org/).
NicoStan accesses Stan's log posterior and gradients through our integration of
[BridgeStan](https://roualdes.us/bridgestan/latest/) ([Roualdes et al., 2023](https://joss.theoj.org/papers/10.21105/joss.05236))
into NicoStan's R/C++ code;
more specifically, NicoStan's C++ sampler calls the compiled Stan model directly through BridgeStan's C/C++ interface.
NicoStan handles the burnin/warmup, sampling and posterior summaries.


For the burnin (or "warmup") phase, Stan uses a well-established, state-of-the-art No-U-Turn HMC
(NUTS-HMC; see [Hoffman and Gelman, 2014](https://www.jmlr.org/papers/v15/hoffman14a.html)) algorithm for adaptation;
that is, for automatically (or adaptively) tuning the HMC path length ($\tau$).
On the other hand, NicoStan provides state-of-the-art, between-chain adaptation algorithms,
such as SNAPER-HMC ([Sountsov and Hoffman, 2022](https://arxiv.org/abs/2110.11576v3)),
ChEES-HMC ([Hoffman et al., 2021](https://proceedings.mlr.press/v130/hoffman21a.html)),
and ChEES-R-HMC ([Sountsov and Hoffman, 2022](https://arxiv.org/abs/2110.11576v3)) -
see [this section below](#efficient-burnin-algorithms) for more details on NicoStan's burnin algorithms.


This difference in burnin adaptation algorithm makes NicoStan more efficient than Stan for most models
(benchmark results coming soon).
In our testing, NicoStan can also perform very well with a very short burnin
(100-125 iterations with just 4 chains; we are also currently testing shorter burnins).
Furthermore, the fact that NicoStan uses between-chain adaptation (as opposed to within-chain adaptation, like Stan's NUTS-HMC algorithm)
means that all chains finish the sampling phase at approximately the same time, avoiding the common issue of "stuck chains",
which is often seen with complex models when using Stan directly
(e.g., via [cmdstanr](https://mc-stan.org/cmdstanr/) or [rstan](https://mc-stan.org/rstan/)).


Additionally, for models with high-dimensional nuisance parameters
(see [this section below](#models-with-nuisance-parameters-diffusion-pathspace-hmc) for examples of such models),
NicoStan offers a hybrid diffusion-pathspace HMC sampling algorithm
(based on [Beskos et al., 2011](https://doi.org/10.1016/j.spa.2011.06.003),
and [Beskos et al., 2013](https://doi.org/10.1016/j.spa.2012.12.001)),
which can greatly increase efficiency - especially for large N.



Note that NicoStan can also fit any Stan model (i.e., any `.stan` model file);
however, expect efficiency gains (relative to Stan) to be less dramatic for models without high-dimensional nuisance parameters.


The NicoStan R package provides:

- **A general Stan interface**, so that existing Stan models can be fitted via NicoStan
(directly, using the user's existing `.stan` model files).
- **Hybrid diffusion-pathspace HMC** for models with suitable Gaussian latent/nuisance blocks,
based on [Beskos et al., 2011](https://doi.org/10.1016/j.spa.2011.06.003),
and [Beskos et al., 2013](https://doi.org/10.1016/j.spa.2012.12.001).
- **ChEES, ChEES-R and SNAPER-HMC trajectory-length adaptation**,
based on [Hoffman et al., 2021](https://proceedings.mlr.press/v130/hoffman21a.html),
and [Sountsov and Hoffman, 2022](https://arxiv.org/abs/2110.11576v3).
- **Custom AVX2 and AVX-512 maths functions**, supplied through the [BayesMVP](https://github.com/CerulloE1996/BayesMVP) extension, and available to general Stan models; however, note that your `.stan` model file will need to be re-written to declare the custom functions, with the NicoStan/BayesMVP C++ `.hpp` header file supplied when compiling (via `Stan_cpp_user_header`),
as well as replacing standard Stan math functions (e.g. `Phi()`) with their custom AVX2 or AVX-512 counterparts (e.g. `fast_Phi()`).
- **Parallel chains, diagonal/dense empirical or numerical-Hessian mass matrices for the main parameters, posterior summaries and MCMC diagnostics**
(see [Efficient burnin algorithms](#efficient-burnin-algorithms)); the main/nuisance metrics can be chosen separately.


NicoStan grew out of our work on efficient sampling for multivariate probit (MVP) models,
specifically their latent class extensions (e.g., the LC-MVP), which are used for the evaluation
of diagnostic/screening test accuracy without a perfect gold standard
(see e.g., [Xu et al., 2013](https://doi.org/10.1002/sim.5695); [Xu and Craig, 2009](https://doi.org/10.1111/j.1541-0420.2008.01194.x);
[Uebersax, 1999](https://doi.org/10.1177/01466219922031400); and [Cerullo et al., 2025](https://arxiv.org/abs/2509.18489v1)).


The [BayesMVP R package extension](https://github.com/CerulloE1996/BayesMVP) to NicoStan adds highly optimised,
specialised implementations of these models,
with manually implemented gradients
(see the [BayesMVP](#bayesmvp-multivariate-probit-models) section below).
For instance, we used [BayesMVP](https://github.com/CerulloE1996/BayesMVP) to fit the models in our simulation study ([Cerullo et al., 2025](https://arxiv.org/abs/2509.18489v1)),
which compared the LC-MVP and latent trait ([Qu et al., 1996](https://pubmed.ncbi.nlm.nih.gov/8805757/)) models.


<!-- ------------------------------------------------------------------------------------------------------------------------------- -->
## Installation
<!-- ------------------------------------------------------------------------------------------------------------------------------- -->

NicoStan requires:


- An R installation, a C++17 compiler and GNU make.
- The CmdStanR interface and CmdStan C++ source tree.
See the [CmdStanR installation guide](https://mc-stan.org/cmdstanr/articles/cmdstanr.html).
- The BridgeStan R interface and matching C++ library.
See the [BridgeStan installation guide](https://roualdes.us/bridgestan/latest/languages/r.html).


The current development build uses CmdStan 2.36.0 and BridgeStan 2.6.2.

We have tested NicoStan on Linux, using R 4.3.3 and GCC 11.4,
with the C++ source trees at `~/.cmdstan/cmdstan-2.36.0` and `~/.bridgestan/bridgestan-2.6.2`.

### Installation from local source files

If you have the NicoStan source files on your computer, use the administrator installer in `inst/examples`:


```r
source("path/to/NicoStan/inst/examples/NicoStan_admin_install.R")
```

Restart R after installation, then load `library(NicoStan)`.

The example source files are in `inst/examples`, and
the installation copies them into the compiled package.


### Installation from GitHub

For installation from GitHub, start with a fresh R session and a writable package library:

```r
install.packages(c("remotes", "devtools"))
remotes::install_github(repo = "CerulloE1996/NicoStan", upgrade = "never")
NicoStan::install_NicoStan()
## Restart R, then load library(NicoStan).
```

Note that the first step installs the outer R package and its installer; `install_NicoStan()` then compiles and installs the sampler itself. Also, make sure to restart R before switching between the installed and development copies.

### Installing the [BayesMVP](https://github.com/CerulloE1996/BayesMVP) extension

The AVX and native-model examples additionally require the NicoStan-compatible [BayesMVP](https://github.com/CerulloE1996/BayesMVP) extension.
After installing NicoStan, run the following in a fresh R session:

```r
remotes::install_github(repo = "CerulloE1996/BayesMVP",
                        ref = "main", upgrade = "never")
BayesMVP::install_BayesMVP()
## Restart R before loading the compiled BayesMVP package.
```

Note that this command installs the [main branch](https://github.com/CerulloE1996/BayesMVP/tree/main) of [BayesMVP](https://github.com/CerulloE1996/BayesMVP).


<!-- ------------------------------------------------------------------------------------------------------------------------------- -->
## Examples
<!-- ------------------------------------------------------------------------------------------------------------------------------- -->

The general examples are grouped by their parameterisation:


- **Hybrid/joint HMC with nuisance diffusion:** Cox shared frailty, hierarchical logistic regression,
joint longitudinal-survival modelling, and a discrete-time AR(1) stochastic-volatility model
([Stochastic_volatility_discrete_time.stan](inst/examples/models/Stochastic_volatility_discrete_time.stan)).
Even this non-centred discrete-time example produced 9 Stan NUTS divergences at `adapt_delta = 0.99` in a saved N = 60 validation check;
a separate, longer check on different simulated data had none at `0.999`
(see the [saved validation settings and results](inst/examples/README_benchmarks.md#discrete-time-stochastic-volatility-validation)).
Continuous-time SV can introduce the additional path-discretisation difficulties described
[below](#models-with-nuisance-parameters-diffusion-pathspace-hmc).
- **Standard HMC:** Weibull survival regression, robust Student-t regression and marginal Gaussian process (GP) regression.
Note that the GP example analytically integrates out the latent function, and the Student-t example evaluates its likelihood directly.


The logistic random-intercept example gives a short introduction to the API:


```r
source(system.file("examples", "random_intercepts.R", package = "NicoStan"))
fit <-  run_random_intercepts(burnin_algorithm = "CHEESR")
diagnostics <-  fit$summary(save_log_lik_trace = FALSE)
fit$model_fit_object$summaries$summary_tibbles$summary_tibble_main_params

## Other trajectory-length adaptation options:
## "CHEESR_log", "SNAPER", "ChEES"
```


This example simulates data with 20 groups and 200 observations, with a standard normal latent vector 
(i.e., the non-centred random intercepts) declared first in the Stan model.
See the complete [R example](inst/examples/random_intercepts.R) and [Stan model](inst/examples/random_intercepts.stan)
for the full parameterisation and settings.
Note that the R6 class is called `NicoStan::Nico_model` (`NicoStan::MVP_model` also still works, for compatibility);
`Model_type = "Stan"` must be selected when using a `.stan` model file; the other `Model_type` options require the
[BayesMVP R package extension](https://github.com/CerulloE1996/BayesMVP) to be installed
(see [BayesMVP: multivariate probit models](#bayesmvp-multivariate-probit-models)).


The wider comparison examples are available through:


```r
source(system.file("examples", "NicoStan_examples.R", package = "NicoStan"))
result <-  run_NicoStan_example(model = "hierarchical_logistic",
                                engine = "NicoStan", math_backend = "Stan")
```


To compare implementations:


- Use `engine = "cmdstanr"` for Stan/NUTS.
- Use `engine = "NicoStan", math_backend = "Stan"` for NicoStan with standard Stan mathematics.
- Use `engine = "NicoStan", math_backend = "AVX2"` for NicoStan with the custom AVX2 functions.
- Use `engine = "NicoStan", math_backend = "AVX512"` for NicoStan with the custom AVX-512 functions.

These AVX2/AVX-512 options select the prepared AVX versions of the supplied example models; they require the
[BayesMVP extension](https://github.com/CerulloE1996/BayesMVP) and a CPU supporting the selected instruction set.
For your own `.stan` model, you must first declare and call the custom AVX functions in the model and supply their C++ header;
selecting an AVX backend does not automatically replace standard Stan function calls
(see [Custom AVX2 and AVX-512 functions](#custom-avx2-and-avx-512-functions)).


The output records the model/data fingerprints, sampler settings, posterior summaries and timings.
Furthermore, the model definitions, prior scales and validation results are provided alongside the examples.

### Preparing your own Stan model

When preparing a model for NicoStan:

- Declare the nuisance block **first in the Stan parameters block**.
- Set `sample_nuisance = TRUE` (or `FALSE` for a model without a nuisance block).
- Note that NicoStan automatically detects the (unconstrained) dimension of the nuisance block, 
using the Stan compiler metadata and BridgeStan.
- For a non-centred Gaussian representation,
declare standard normal variables in this first block, and then introduce the scales/correlations in the transformed parameters block.
- Keep the complete posterior density (including the prior on the nuisance parameters) in the Stan model.


<!-- ------------------------------------------------------------------------------------------------------------------------------- -->
## [BayesMVP](https://github.com/CerulloE1996/BayesMVP): multivariate probit models
<!-- ------------------------------------------------------------------------------------------------------------------------------- -->


The [BayesMVP R package extension](https://github.com/CerulloE1996/BayesMVP) to NicoStan provides highly optimised,
specialised implementations (with manually implemented likelihood gradients) of some particularly difficult-to-sample models;
more specifically, it provides the following models:

- **MVP:** multivariate probit, using `Model_type = "MVP"`.
- **2LC-MVP:** 2-class multivariate probit, using `Model_type = "LC_MVP"`.
- **MVOP:** multivariate ordinal probit (which also supports mixed binary and ordinal outcomes), using `Model_type = "MVOP"`.
- **2LC-MVOP:** 2-class multivariate ordinal probit, using `Model_type = "LC_MVOP"`.
- **Latent trait:** the 2-class implementation, using `Model_type = "latent_trait"`.


These models handle correlated binary and/or ordinal outcomes; more specifically, the latent class extensions allow
diagnostic/screening test accuracy to be estimated when a perfect reference/gold standard is unavailable.


The ordinal latent class model is described in [Cerullo et al., 2022](https://doi.org/10.1002/jrsm.1567).
Furthermore, in [Cerullo et al., 2025](https://arxiv.org/abs/2509.18489v1) we compared the LC-MVP and latent trait models
in a comprehensive simulation study, in which the models were fitted using [BayesMVP](https://github.com/CerulloE1996/BayesMVP).


<!-- The **4LC-MVOP** model handles two related latent conditions. -->
<!-- It has a Stan implementation and a partial-manual-gradient/fused-function version. -->
<!-- Its partial-manual-gradient implementation runs through Stan/NicoStan, -->
<!-- whilst the five models above also have native BayesMVP implementations.  -->
<!-- Development and applications of 4LC-MVOP are covered in separate papers,  -->
<!-- whilst the model also provides a larger example for the general NicoStan sampler. -->


<!-- ------------------------------------------------------------------------------------------------------------------------------- -->
## Benchmarks
<!-- ------------------------------------------------------------------------------------------------------------------------------- -->


In our initial tests on models with high-dimensional nuisance parameters/blocks,
NicoStan has achieved speed-ups (relative to Stan, using cmdstanr) of more than **10×, without enabling AVX**.
These comparisons use the standard Stan maths functions, so the gains reflect the sampler and its burnin adaptation.
Furthermore, the custom AVX functions in the [BayesMVP extension](https://github.com/CerulloE1996/BayesMVP)
provide an additional route to speeding things up; more specifically, they reduce the cost of evaluating the model and its gradients.


**Detailed benchmarks are coming soon.** The comparisons will cover:


- Plain Stan (via the cmdstanr R package),
fitted using the default NUTS-HMC-based ([Hoffman and Gelman, 2014](https://www.jmlr.org/papers/v15/hoffman14a.html)) algorithm.
- NicoStan, using the same Stan model and the standard Stan maths functions (i.e., from the stan::math C++ library).
- NicoStan, with our custom AVX2 and/or AVX-512 functions (on supported CPUs).


All of these comparisons use the same data, priors and initial values.
The results will report sampling efficiency in terms of effective sample size (ESS); more specifically:
(i) sampling ESS/sec;
(ii) the time taken to reach a minimum ESS of 100, 1,000 and 10,000; and
(iii) sampling ESS per gradient evaluation (ESS/grad).
Additionally, we will report the burnin/warm-up costs,
as well as convergence diagnostics (e.g., R-hat and nested R-hat [nR-hat]; [Margossian et al., 2024](https://arxiv.org/abs/2110.13017v6)),
alongside the computational settings.


<!-- ------------------------------------------------------------------------------------------------------------------------------- -->
## Models with nuisance parameters (diffusion-pathspace HMC)
<!-- ------------------------------------------------------------------------------------------------------------------------------- -->


NicoStan is particularly aimed at models with a large block of latent/nuisance parameters for which analytic marginalisation is unavailable,
or numerical marginalisation is too expensive. Techniques such as non-centring can improve the posterior geometry for some of these models;
however, it does not remove the nuisance block - the corresponding latent variables still have to be handled during posterior computation.


Such blocks occur in many commonly-used models; more specifically:

- **MVP, LC-MVP, MVOP and LC-MVOP:** latent-Gaussian/auxiliary variables for correlated binary and/or ordinal outcomes.
The augmented state grows with the number of individuals and outcomes; evaluating the marginal likelihood instead involves multivariate Gaussian rectangle probabilities.
- **Generalised linear mixed models:** Gaussian random intercepts/slopes for binary, ordinal or count outcomes,
particularly when there are many subjects/groups or crossed random effects.
- **Generalised additive mixed models:** Gaussian spline coefficients and random effects,
with non-Gaussian observations retaining a nonconjugate posterior over those coefficients.
- **Gaussian processes with non-Gaussian likelihoods:** latent function values for binary, ordinal or count outcomes.
- **Spatial and spatiotemporal models:** Gaussian spatial fields, temporal effects and their interactions,
for instance in binomial/Poisson disease-mapping models.
- **Log-Gaussian Cox processes:** a Gaussian latent log-intensity field, often represented on a large spatial mesh/grid.
- **Stochastic volatility:** a latent log-volatility path in discrete-time models, or a latent volatility diffusion in continuous time.
For instance, path augmentation for a partially observed volatility diffusion introduces latent points between observations;
the discretisation then controls both approximation error and the size of the sampled path
(see [Beskos et al., 2013](https://doi.org/10.1016/j.spa.2012.12.001)).
The discrete-time AR(1) version has a simpler time structure, but neither centred nor non-centred parameterisations are uniformly efficient
across all regimes ([Kastner and Frühwirth-Schnatter, 2014](https://doi.org/10.1016/j.csda.2013.01.002)).
- **Nonlinear/non-Gaussian state-space models and partially observed diffusions:** latent states,
state innovations or a non-centred driving path.
- **Survival analysis with lognormal frailty:** Gaussian log-frailties at the individual/cluster level,
with the hazard/survival likelihood generally preventing analytic integration.
- **Joint longitudinal and survival models:** shared random effects or latent biomarker trajectories;
the survival component generally prevents the Gaussian marginalisation available for simpler longitudinal models.
- **Item-response and non-Gaussian factor models:** latent abilities/factors for binary, ordinal or count responses.
- **Nonlinear mixed-effects models:** Gaussian individual-level effects entering a nonlinear response model,
for instance in population pharmacokinetic/pharmacodynamic (PK/PD) models.
- **Measurement-error models with nonlinear/non-Gaussian outcomes:** unobserved true exposures/covariates
or residual processes that enter the outcome likelihood.
- **Nonlinear inverse problems with Gaussian-prior fields:** unknown spatial functions or initial conditions,
for instance log-permeability fields inferred through a groundwater-flow model.


For examples of these structures, see the [Stan item-response models](https://mc-stan.org/learn-stan/case-studies/rasch_and_2pl.html),
the [nonlinear mixed-effects PK models in Johnston et al., 2024](https://doi.org/10.1002/psp4.13088),
the diffusion bridge, stochastic volatility and latent diffusion survival applications in
[Beskos et al., 2013](https://doi.org/10.1016/j.spa.2012.12.001),
and the function-space applications in [Cotter et al., 2013](https://arxiv.org/abs/1202.0709).
The [runnable examples](#examples) show the implementations currently supplied with NicoStan.
The supplied discrete-time AR(1) stochastic-volatility example is a different model from the continuous-time experiments in
[Beskos et al., 2013](https://doi.org/10.1016/j.spa.2012.12.001) and [Beskos et al., 2015](https://doi.org/10.1093/biomet/asv051).


Note that some simpler Gaussian models allow the latent block to be integrated out analytically.
For instance, our Gaussian-outcome GP example uses the marginal likelihood and is fitted with standard HMC;
the latent-GP examples listed here instead have non-Gaussian likelihoods.


### Why some of these models are particularly difficult to sample


MVP and its latent class/ordinal extensions (LC-MVP, LC-MVOP, etc) are particularly demanding examples.
The augmented latent data can be very large, 
whilst strong correlations and outcome-dependent truncation create difficult dependencies between latent variables and model parameters.
Additionally, specifically for latent class models, poor identifiability can make sampling harder still
(see [Talhouk et al., 2012](https://www.stats.ox.ac.uk/~doucet/talhouk_doucet_murphy_sparseprobit.pdf)
and [Cerullo et al., 2025](https://arxiv.org/abs/2509.18489v1)).


Stochastic-volatility and nonlinear state-space models can also be difficult when latent processes are highly persistent
or their innovation scales are small. In frailty survival analysis models, joint longitudinal survival analysis, and latent-factor models,
limited information about individual effects can induce strong posterior dependence between the effects and their population-level parameters.
Techniques such as non-centring can help with these dependencies in some models;
however, the remaining high-dimensional latent block may nevertheless make the computation prohibitively expensive.


### GPU acceleration and serial dependencies


Several of these models also contain calculations that are difficult to map efficiently to a GPU.
For instance, in sequential-conditioning implementations of MVP/LC-MVP,
the calculation for one outcome depends on the preceding outcomes within the same individual;
more specifically, the conditional bounds and latent-variable transformations are built up in sequence.
Similarly, forward simulation/non-centred reconstruction of nonlinear state-space models and diffusion paths
often requires each state to be available before the next state can be calculated.
These dependencies limit parallelism within an individual/path, although independent individuals, paths and chains can still be processed in parallel.


Additionally, a model may require many small calculations, irregular indexing or repeated transfers between CPU and GPU memory,
so that the overhead of GPU execution is large relative to the work being parallelised.
This makes the structure of the likelihood/gradient calculation important, alongside the number of observations or latent parameters.


GPU suitability therefore depends on the implementation and the amount of parallel work available;
large dense-matrix operations and vectorised GLM likelihoods can benefit substantially.
For instance, [Stan's OpenCL interface](https://mc-stan.org/cmdstanr/articles/articles-online-only/opencl.html)
supports several GLM likelihoods directly.
For models whose serial dependencies or likelihood/gradient calculations limit effective GPU acceleration,
NicoStan provides a CPU-based solution.
Its hybrid/joint HMC and diffusion-pathspace dynamics, between-chain adaptation, 
and CPU parallelism address the cost of posterior sampling directly;
the optional [AVX2/AVX-512 functions](#custom-avx2-and-avx-512-functions) additionally accelerate model/gradient evaluation.
Furthermore, NicoStan can be used without having to restructure the model's calculations for GPU execution, or having to install CUDA.


<!--  Further parameterisation examples are given in the Stan User's Guide sections on   -->
<!--  [stochastic volatility](https://mc-stan.org/docs/2_33/stan-users-guide/stochastic-volatility-models.html), -->
<!--  [Gaussian processes](https://mc-stan.org/docs/stan-users-guide/gaussian-processes.html) and   -->
<!--  [reparameterisation](https://mc-stan.org/docs/stan-users-guide/efficiency-tuning.html).   -->


<!-- ------------------------------------------------------------------------------------------------------------------------------- -->
## How NicoStan works
<!-- ------------------------------------------------------------------------------------------------------------------------------- -->


Many Bayesian models have a relatively small number of parameters of direct interest (i.e., the "main" model parameters),
but a much larger (i.e., high-dimensional) block of latent variables (or "nuisance parameters");
for instance, models with random effects, stochastic volatility models,
and Gaussian process models (see the [Models with nuisance parameters](#models-with-nuisance-parameters-diffusion-pathspace-hmc)
section above for more examples).
As this nuisance block grows, standard HMC often becomes increasingly expensive.


To address this, NicoStan treats the nuisance parameters (in the unconstrained space) as a change of measure from a Gaussian reference measure
(based on [Beskos et al., 2011](https://doi.org/10.1016/j.spa.2011.06.003) and
[Beskos et al., 2013](https://doi.org/10.1016/j.spa.2012.12.001)) -
i.e., their posterior can be written as a Gaussian reference measure multiplied by a correction term.
This formulation is particularly useful when a Gaussian reference captures much of the nuisance-block structure;
for instance, a non-centred parameterisation such as `x = μ(θ) + L(θ)u`, with `u ~ N(0, I)`, gives a standard normal prior for `u`.
The nuisance posterior itself does not need to be Gaussian - its departure from the reference is retained in the correction term.


NicoStan samples the main parameters `θ` and the latent/nuisance parameters `u` **jointly**, within the same hybrid HMC/diffusion-pathspace trajectory:

- The main model parameters are updated using standard HMC dynamics.
- For models with nuisance parameters,
the nuisance parameters are updated using **diffusion-pathspace HMC** ([Beskos et al., 2013](https://doi.org/10.1016/j.spa.2012.12.001)),
where the Gaussian part of the dynamics is solved exactly (it is a rotation),
so this Gaussian substep adds no integration error, regardless of the size of the nuisance block.
The complete trajectory also contains numerical steps for the remaining terms; the Metropolis-Hastings acceptance step corrects their integration error.


Models without a nuisance block (e.g., standard univariate logistic regression) use standard HMC throughout.
In our initial tests, we have also found NicoStan to be more efficient than Stan (via cmdstanr) for some of these models;
we expect the different burnin/adaptation schemes to contribute to this, although the detailed comparisons are still in progress
(see [Benchmarks](#benchmarks) and [Efficient burnin algorithms](#efficient-burnin-algorithms)).


The hybrid/joint sampling scheme uses a "kick-flow-kick" splitting.
Its Gaussian-reference dynamics follow [Beskos et al., 2011](https://doi.org/10.1016/j.spa.2011.06.003)
and the diffusion-pathspace construction of [Beskos et al., 2013](https://doi.org/10.1016/j.spa.2012.12.001);
[Beskos et al., 2015](https://doi.org/10.1093/biomet/asv051) also describe joint updates of model parameters and latent driving variables.
Related approaches include Hilbert-space HMC with Ornstein-Uhlenbeck bridge velocities
([Pinski, 2021](https://doi.org/10.3390/e23050499)),
and pseudo-marginal HMC ([Alenlöv et al., 2021](https://jmlr.org/papers/v22/19-486.html)).


NicoStan also has the following two default options:


- **Shifted centre:** During burnin, the Gaussian reference is centred at a running estimate of the posterior mean of the nuisance block,
which is frozen before sampling (`theta_hat_us_rule = "running_mean_frozen"`), rather than at zero (`theta_hat_us_rule = "zero"`).
- **Nuisance mass:** each nuisance coordinate has its own mass,
estimated during burnin (`metric_type_nuisance = "Empirical"`);
alternatively, you can use a single common mass (`"uniform_diag"`) or unit mass (`"unit"`).


Note that neither of the above options changes the target distribution -
they change the reference dynamics used in the splitting.


<!-- Furthermore, the trajectory-length adaptation used during burnin 
(e.g., ChEES-R or SNAPER; see the [Efficient burnin algorithms](#efficient-burnin-algorithms) section below) 
is based on the main parameters only; hence, a large nuisance block does not dominate the adaptation.  -->


<!-- ------------------------------------------------------------------------------------------------------------------------------- -->
## Efficient burnin algorithms
<!-- ------------------------------------------------------------------------------------------------------------------------------- -->


NicoStan's burnin (i.e., warm-up) runs several chains in parallel,
and adapts the HMC tuning parameters using information pooled **across** all of the burnin chains (i.e., between-chain adaptation);
whereas Stan normally adapts each NUTS-HMC chain separately.


This between-chain adaptation is based on ChEES-HMC
([Hoffman et al., 2021](https://proceedings.mlr.press/v130/hoffman21a.html)),
and SNAPER-HMC ([Sountsov and Hoffman, 2022](https://arxiv.org/abs/2110.11576v3)).


More specifically, during burnin, NicoStan adapts:

- The step size, targeting a mean acceptance probability of `adapt_delta` (0.80 by default), using an ADAM-type update ([Kingma and Ba, 2015](https://arxiv.org/abs/1412.6980)).
- The mass matrix (i.e., metric) for the main parameters, using either empirical covariance/variance estimates from the burnin chains (`metric_type_main = "Empirical"`)
or a numerical Hessian (`metric_type_main = "Hessian"`); more specifically, the Hessian is computed by finite differences of the main-parameter gradients.
Both methods support a diagonal/dense Euclidean metric (`metric_shape_main = "diag"` or `"dense"`).
The nuisance masses and centre are adapted separately (see [How NicoStan works](#how-nicostan-works)).
- The trajectory length, using the criterion selected via `burnin_algorithm`
(see [Trajectory-length adaptation rules](#trajectory-length-adaptation-rules) below).


The main/nuisance separation means that NicoStan can use a dense empirical/Hessian metric for the main block
whilst retaining a diagonal/unit metric for a much larger nuisance block.
Stan's [standard HMC/NUTS metric interface](https://mc-stan.org/docs/cmdstan-guide/mcmc_config.html#metric)
selects a unit, diagonal or dense metric for the complete unconstrained parameter vector;
its dense option therefore includes the nuisance parameters too, rather than exposing separate main/nuisance metric choices.
This distinction allows NicoStan to use a dense main-parameter metric without constructing/storing a dense metric for the entire latent/nuisance block.


### Trajectory-length adaptation rules

Use `burnin_algorithm` to choose the trajectory-length adaptation rule:


- `CHEESR` (**ChEES-R**): The original ChEES-rate criterion,
using the squared change in the main block's centred squared radius per realised trajectory length.
- `CHEESR_log` (**Log-ChEES-R**): NicoStan's log-ratio formulation of the ChEES-rate criterion,
which normalises the numerator gradient by a running average of the ChEES numerator.
- `SNAPER` (**SNAPER**): Learns a difficult main-parameter direction with a metric-aware Oja update,
and adapts the trajectory length using squared changes along that direction per unit length.
- `ChEES` (**ChEES**): Uses the squared change in the main block's centred squared radius,
without dividing by trajectory length.


ChEES measures squared changes in the centred squared radius of the parameter vector
([Hoffman et al., 2021](https://proceedings.mlr.press/v130/hoffman21a.html)).
ChEES-R (the ChEES-Rate criterion of [Sountsov and Hoffman, 2022](https://arxiv.org/abs/2110.11576v3))
divides this change by the trajectory length, so that trajectory length is used as a proxy for computational cost.
SNAPER-HMC ([Sountsov and Hoffman, 2022](https://arxiv.org/abs/2110.11576v3))
focuses the corresponding criterion along the first principal component
(i.e., the direction of largest variance in the coordinates used for adaptation),
learned during warm-up with Oja's algorithm.


The position-based criteria use the main parameters in coordinates defined by the current mass matrix
(similarly to [Riou-Durand et al., 2023](https://proceedings.mlr.press/v206/riou-durand23a.html)).
<!-- If `BᵀB = M`, these coordinates are `z = B(θ - μ)`.  -->
<!-- Both diagonal and dense main-parameter metrics are supported, -->
<!-- and SNAPER-HMC learns its direction in the same coordinates. -->


<!-- `CHEESR_log` is our log-ratio formulation of the ChEES rate, -->
<!-- using a running average of the numerator to normalise its gradient. -->
Note that both `CHEESR` and `CHEESR_log` adapt a positive trajectory length on the log scale.
Their update rules differ; hence, both are available in NicoStan
(see the [adaptation notes](docs/adaptation-notes.md) for the derivation).


The default trajectory-length settings differ between burnin and sampling:


- **Burn-in:** fixed trajectory length (`randomize_tau_burnin = FALSE`)
for each burnin iteration at the current tuning setting.
- **Post-burn-in sampling:** randomised trajectory length (`randomize_tau_sampling = TRUE`) -
drawn uniformly between zero and twice the adapted scale -
with at least one integration step.


Note that `manual_tau` separately controls whether the trajectory length is adapted at all.


<!-- ------------------------------------------------------------------------------------------------------------------------------- -->
## Custom AVX2 and AVX-512 functions
<!-- ------------------------------------------------------------------------------------------------------------------------------- -->


NicoStan can use our custom vectorised (i.e., SIMD) mathematical functions, which are supplied through the [BayesMVP](https://github.com/CerulloE1996/BayesMVP) extension.
These include, for instance, exponentials (`exp()`), logarithms (`log()`)
and normal distribution functions (`Phi()`, `inv_Phi()`, `Phi_approx()` and `inv_Phi_approx()`),
as well as their derivatives.


- **AVX2:** four double-precision values per vector operation; supported by many x86 CPUs.
- **AVX-512:** eight double-precision values per vector operation on supported processors.
- **General Stan models:** declare and call the custom AVX functions in the `.stan` model, then supply their C++ header via `Stan_cpp_user_header`;
standard Stan function calls are not automatically rewritten to use them.
- **Specialised [BayesMVP](https://github.com/CerulloE1996/BayesMVP) models:** the same kernels are used within the native implementations.


Note that the AVX options are not a drop-in switch for an existing Stan model;
more specifically, the user has to re-write their Stan model (i.e., the `.stan` file) so that it calls the custom AVX2/AVX-512 functions from BayesMVP,
which is not straightforward for most models.
The comparison examples (whose Stan models have already been re-written in this way) support separate `math_backend = "Stan"`, `"AVX2"` and `"AVX512"` options.
The compiled AVX model reports its lane count (i.e., four for AVX2 and eight for AVX-512), so you can check which implementation is being used.
Note that you should compile the model on the same machine you will use for sampling, since the available instruction sets depend on the CPU;
for instance, [Intel's processor guidance](https://www.intel.com/content/www/us/en/support/articles/000090473/processors/intel-core-processors.html)
describes how to check which extensions your CPU supports.


<!-- ------------------------------------------------------------------------------------------------------------------------------- -->
## How to cite NicoStan
<!-- ------------------------------------------------------------------------------------------------------------------------------- -->


If you use NicoStan in your work, please cite the package as follows:

Cerullo, E. (2026). NicoStan: Adaptive Hamiltonian Monte Carlo for Stan Models. R package version 0.1.9000. https://github.com/CerulloE1996/NicoStan

```bibtex
@Manual{Cerullo2026NicoStan,
  title = {NicoStan: Adaptive Hamiltonian Monte Carlo for Stan Models},
  author = {Enzo Cerullo},
  year = {2026},
  note = {R package version 0.1.9000},
  url = {https://github.com/CerulloE1996/NicoStan},
}
```

Please also cite the methodological references relevant to the options used in your analysis (see [References](#references)).


<!-- ------------------------------------------------------------------------------------------------------------------------------- -->
## How to cite BayesMVP
<!-- ------------------------------------------------------------------------------------------------------------------------------- -->


If you use the BayesMVP extension (i.e., the specialised MVP models or the custom AVX2/AVX-512 functions), please also cite:

Cerullo, E. (2026). BayesMVP: Accelerated multivariate probit models using NicoStan. R package version 0.1.9000. https://github.com/CerulloE1996/BayesMVP

```bibtex
@Manual{Cerullo2026BayesMVP,
  title = {BayesMVP: Accelerated multivariate probit models using NicoStan},
  author = {Enzo Cerullo},
  year = {2026},
  note = {R package version 0.1.9000},
  url = {https://github.com/CerulloE1996/BayesMVP},
}
```


<!-- ------------------------------------------------------------------------------------------------------------------------------- -->
## References
<!-- ------------------------------------------------------------------------------------------------------------------------------- -->


1. Beskos, A., Pinski, F. J., Sanz-Serna, J. M. and Stuart, A. M. (2011). [Hybrid Monte Carlo on Hilbert spaces](https://doi.org/10.1016/j.spa.2011.06.003). Stochastic Processes and Their Applications, 121(10), 2201-2230.

2. Beskos, A., Kalogeropoulos, K. and Pazos, E. (2013). [Advanced MCMC methods for sampling on diffusion pathspace](https://doi.org/10.1016/j.spa.2012.12.001). Stochastic Processes and Their Applications, 123(4), 1415-1453.

3. Beskos, A., Dureau, J. and Kalogeropoulos, K. (2015). [Bayesian inference for partially observed stochastic differential equations driven by fractional Brownian motion](https://doi.org/10.1093/biomet/asv051). Biometrika, 102(4), 809-827.

4. Alenlöv, J., Doucet, A. and Lindsten, F. (2021). [Pseudo-Marginal Hamiltonian Monte Carlo](https://jmlr.org/papers/v22/19-486.html). Journal of Machine Learning Research, 22(141), 1-45.

5. Hoffman, M. D., Radul, A. and Sountsov, P. (2021). [An Adaptive-MCMC Scheme for Setting Trajectory Lengths in Hamiltonian Monte Carlo](https://proceedings.mlr.press/v130/hoffman21a.html). Proceedings of Machine Learning Research, 130, 3907-3915.

6. Sountsov, P. and Hoffman, M. D. (2022). [Focusing on Difficult Directions for Learning HMC Trajectory Lengths](https://arxiv.org/abs/2110.11576v3). arXiv:2110.11576, version 3, 6 May 2022.

7. Pinski, F. J. (2021). [A Novel Hybrid Monte Carlo Algorithm for Sampling Path Space](https://doi.org/10.3390/e23050499). Entropy, 23(5), 499.

8. Riou-Durand, L., Sountsov, P., Vogrinc, J., Margossian, C. and Power, S. (2023). [Adaptive Tuning for Metropolis Adjusted Langevin Trajectories](https://proceedings.mlr.press/v206/riou-durand23a.html). Proceedings of Machine Learning Research, 206, 8102-8116.

9. Cerullo, E., Jones, H. E., Carter, O., Quinn, T. J., Cooper, N. J. and Sutton, A. J. (2022). [Meta-analysis of dichotomous and ordinal tests with an imperfect gold standard](https://doi.org/10.1002/jrsm.1567). Research Synthesis Methods, 13(5), 595-611.

10. Cerullo, E., Pinkney, S., Sutton, A. J., Lucas, T., Cooper, N. J. and Jones, H. E. (2025). [Latent class multivariate probit and latent trait models for evaluating test accuracy without a gold standard: A simulation study](https://arxiv.org/abs/2509.18489v1). arXiv:2509.18489, version 1, 23 September 2025.

11. Hoffman, M. D. and Gelman, A. (2014). [The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo](https://www.jmlr.org/papers/v15/hoffman14a.html). Journal of Machine Learning Research, 15(47), 1593-1623.

12. Margossian, C. C., Hoffman, M. D., Sountsov, P., Riou-Durand, L., Vehtari, A. and Gelman, A. (2024). [Nested R-hat: Assessing the convergence of Markov chain Monte Carlo when running many short chains](https://arxiv.org/abs/2110.13017v6). Bayesian Analysis; arXiv:2110.13017, version 6, 30 May 2024.


13. Roualdes, E. A., Ward, B., Carpenter, B., Seyboldt, A. and Axen, S. D. (2023). [BridgeStan: Efficient in-memory access to the methods of a Stan model](https://doi.org/10.21105/joss.05236). Journal of Open Source Software, 8(87), 5236.

14. Qu, Y., Tan, M. and Kutner, M. H. (1996). [Random effects models in latent class analysis for evaluating accuracy of diagnostic tests](https://pubmed.ncbi.nlm.nih.gov/8805757/). Biometrics, 52(3), 797-810.

15. Xu, H. and Craig, B. A. (2009). [A probit latent class model with general correlation structures for evaluating accuracy of diagnostic tests](https://doi.org/10.1111/j.1541-0420.2008.01194.x). Biometrics, 65(4), 1145-1155.

16. Xu, H., Black, M. A. and Craig, B. A. (2013). [Evaluating accuracy of diagnostic tests with intermediate results in the absence of a gold standard](https://doi.org/10.1002/sim.5695). Statistics in Medicine, 32(15), 2571-2584.

17. Uebersax, J. S. (1999). [Probit latent class analysis with dichotomous or ordered category measures: Conditional independence/dependence models](https://doi.org/10.1177/01466219922031400). Applied Psychological Measurement, 23(4), 283-297.

18. Cotter, S. L., Roberts, G. O., Stuart, A. M. and White, D. (2013). [MCMC methods for functions: Modifying old algorithms to make them faster](https://arxiv.org/abs/1202.0709). Statistical Science, 28(3), 424-446.

19. Talhouk, A., Doucet, A. and Murphy, K. (2012). [Efficient Bayesian inference for multivariate probit models with sparse inverse correlation matrices](https://www.stats.ox.ac.uk/~doucet/talhouk_doucet_murphy_sparseprobit.pdf). Journal of Computational and Graphical Statistics, 21(3), 739-757.

20. Johnston, C. K., Waterhouse, T., Wiens, M., Mondick, J., French, J. and Gillespie, W. R. (2024). [Bayesian estimation in NONMEM](https://doi.org/10.1002/psp4.13088). CPT: Pharmacometrics & Systems Pharmacology, 13, 192-207.

21. Kingma, D. P. and Ba, J. (2015). [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980). International Conference on Learning Representations (ICLR); arXiv:1412.6980.


<!-- ------------------------------------------------------------------------------------------------------------------------------- -->
## Package citation and development
<!-- ------------------------------------------------------------------------------------------------------------------------------- -->


NicoStan is developed by Enzo Cerullo and licensed under GPL-3.
Package citation metadata is provided in [CITATION.cff](CITATION.cff),
and the references above are available as a [BibTeX file](docs/references.bib).


The source includes the example models and validation scripts.



