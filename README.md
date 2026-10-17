
<!-- ------------------------------------------------------------------------------------------------------------------------------- -->
# NicoStan <img src="docs/assets/NicoStan_logo_720.png" align="right" height="140" alt="NicoStan logo" />
<!-- ------------------------------------------------------------------------------------------------------------------------------- -->

[What is NicoStan?](#what-is-nicostan) ·
[How is NicoStan different to Stan?](#how-is-nicostan-different-to-stan-eg-cmdstanrrstan) ·
[Installation](#installation) ·
[Examples](#examples) ·
[BayesMVP: multivariate probit models](#bayesmvp-multivariate-probit-models) ·
[Automatic test (column) reordering](#automatic-test-column-reordering) ·
[Benchmarks](#benchmarks) ·
[Models with nuisance parameters (diffusion-pathspace HMC)](#models-with-nuisance-parameters-diffusion-pathspace-hmc) ·
[How NicoStan works](#how-nicostan-works) ·
[NicoStan's efficient burnin algorithms (SNAPER-HMC and ChEES-R-HMC)](#efficient-burnin-algorithms-snaper-hmc-and-chees-r-hmc) ·
[LQ_ESSR trajectory-length criterion](#lq_essr-a-linearquadratic-ess-rate-trajectory-length-criterion) ·
[SpESS-R trajectory-length criterion](#spess-r-a-spectral-linearquadratic-ess-rate-trajectory-length-criterion) ·
[Custom AVX2 and AVX-512 functions](#custom-avx2-and-avx-512-functions) ·
[How to cite NicoStan](#how-to-cite-nicostan) ·
[How to cite BayesMVP](#how-to-cite-bayesmvp) ·
[References](#references) ·
[Package citation and development](#package-citation-and-development)



| φ(x): whose jump is measured | ρ = −1 (no length penalty)¹ | ρ = 0 (÷ √τ) | ρ = 1 (÷ τ)² | ρ adaptive³ |
| --- | --- | --- | --- | --- |
| x | ESJD (Pasarica & Gelman, 2010) | Wang, Mohamed & de Freitas (2013) | ESJD rate (NicoStan "ESJD") | - |
| ‖x − m‖² | ChEES (Hoffman, Radul & Sountsov, 2021) | - | ChEES rate (NicoStan "CHESSR") | - |
| (wᵀ(x − m))² | - | - | SNAPER (Sountsov & Hoffman, 2021) | Adaptive MALT (Riou-Durand et al., 2023) |

The criterion family is Cρ(τ) = ESJDφ(τ) / τ^((1+ρ)/2) (Riou-Durand et al., 2023). m is the posterior mean and w is the leading principal direction.

¹ Outside the range ρ ∈ [0, 1] that Riou-Durand et al. consider; included here as the criteria with no length penalty.
² With jittered trajectory lengths, NicoStan divides each trajectory's squared jump by its own length.
"CHESSR_log" instead divides the average by τ.
Note that TensorFlow Probability's implementation divides by the length plus one step size,
i.e., it charges the extra gradient evaluation of each iteration;
NicoStan offers the same (and the exact expected gradient cost) via the R option `NicoStan_rate_criterion_cost_offset_steps`
("auto", or the number of extra steps), but divides by the length alone by default,
which was faster on our latent-class benchmarks
(see the validation note in [Efficient burnin algorithms](#efficient-burnin-algorithms-snaper-hmc-and-chees-r-hmc)).
³ ρ set to an estimate of the lag-1 autocorrelation of φ.

NicoStan's "ESJD_CHESSR" and "ESJD_SNAPER" are geometric means of two ρ = 1 criteria with different φ, so they aren't members of this family.



<!-- ------------------------------------------------------------------------------------------------------------------------------- -->
## What is NicoStan?
<!-- ------------------------------------------------------------------------------------------------------------------------------- -->


NicoStan is an R package for efficiently fitting Bayesian models written in the probabilistic programming language
[Stan](https://mc-stan.org/).
NicoStan accesses Stan's log posterior and gradients through our integration of
[BridgeStan](https://roualdes.us/bridgestan/latest/) ([Roualdes et al., 2023](https://joss.theoj.org/papers/10.21105/joss.05236))
into NicoStan's R/C++ code;
more specifically, NicoStan's C++ sampler calls the compiled Stan model directly through BridgeStan's C/C++ interface.
NicoStan handles the burnin/warmup, sampling and posterior summaries.


NicoStan grew out of our work on efficient sampling for multivariate probit (MVP) models,
specifically their latent class extensions (e.g., the LC-MVP), which are used for the evaluation
of diagnostic/screening test accuracy without a perfect gold standard
(see e.g., [Xu et al., 2013](https://doi.org/10.1002/sim.5695); [Xu and Craig, 2009](https://doi.org/10.1111/j.1541-0420.2008.01194.x);
[Uebersax, 1999](https://doi.org/10.1177/01466219922031400); and [Cerullo et al., 2025](https://arxiv.org/abs/2509.18489v1)).


Also, NicoStan is named after a cat, who provided great stress-relief whilst I was making NicoStan,
and made waiting 10+ minutes for C++ code to recompile tolerable.


NicoStan was formerly called "BayesMVP" (since we have mainly been working with multivariate probit [MVP] models);
however, it has now been split into 2 R packages: NicoStan, for the general Stan interface
(which relies heavily on [BridgeStan](https://roualdes.us/bridgestan/latest/)),
and [BayesMVP](https://github.com/CerulloE1996/BayesMVP), which is now an extension to NicoStan, rather than a standalone R package.


The [BayesMVP R package extension](https://github.com/CerulloE1996/BayesMVP) to NicoStan adds highly optimised,
specialised implementations of these models,
with manually implemented gradients
(see the [BayesMVP](#bayesmvp-multivariate-probit-models) section below).
For instance, we used [BayesMVP](https://github.com/CerulloE1996/BayesMVP) to fit the models in our simulation study
([Cerullo et al., 2025](https://arxiv.org/abs/2509.18489v1)),
which compared the LC-MVP and latent trait ([Qu et al., 1996](https://pubmed.ncbi.nlm.nih.gov/8805757/)) models,
which are used for the estimation of diagnostic/screening test accuracy without a perfect gold standard.


To summarise, the NicoStan R package provides:
<!-- ----------------------------------------------------------------------------------------- -->
- **A general Stan interface**, so that existing Stan models can be fitted via NicoStan
(directly, using the user's existing `.stan` model files).
<!-- ----------------------------------------------------------------------------------------- -->
- **Hybrid diffusion-pathspace HMC** for models with suitable Gaussian latent/nuisance blocks,
based on [Beskos et al., 2011](https://doi.org/10.1016/j.spa.2011.06.003),
and [Beskos et al., 2013](https://doi.org/10.1016/j.spa.2012.12.001).
<!-- ----------------------------------------------------------------------------------------- -->
- **ChEES, ChEES-R and SNAPER-HMC trajectory-length adaptation during burnin**,
based on [Hoffman et al., 2021](https://proceedings.mlr.press/v130/hoffman21a.html),
and [Sountsov and Hoffman, 2022](https://arxiv.org/abs/2110.11576v3).
See [Efficient burnin algorithms](#efficient-burnin-algorithms-snaper-hmc-and-chees-r-hmc) for more information.
<!-- ----------------------------------------------------------------------------------------- -->
- **Custom AVX2 and AVX-512 maths functions**, supplied through the
[BayesMVP](https://github.com/CerulloE1996/BayesMVP) R package extension to NicoStan,
and available to general Stan models;
however, note that your `.stan` model file will need to be re-written to declare the custom functions,
with the NicoStan/BayesMVP C++ `.hpp` header file supplied when compiling (via `Stan_cpp_user_header`),
as well as replacing standard Stan math functions (e.g. `Phi()`)
with their custom AVX2 or AVX-512 counterparts (e.g. `fast_Phi()`).
<!-- ----------------------------------------------------------------------------------------- -->
- **Partially-dense metric:**
NicoStan allows you to set a dense HMC mass matrix for the main model parameters,
combined with a diagonal M for the high-dimensional nuisance parameters.
In other words, the main and nuisance metrics can be chosen separately.
This is ideal, because it is often intractable to fit a dense M on the entire parameter set,
and a dense M for the main model parameters often leads to better sampling than a diagonal M,
since it takes posterior correlations into account.
Furthermore, users can set either an Empirical M or a Hessian-based
(using numerical differentiation) M for the main model parameters.
<!-- ----------------------------------------------------------------------------------------- -->


<!-- ------------------------------------------------------------------------------------------------------------------------------- -->
## How is NicoStan different to Stan (e.g., cmdstanr/rstan)?
<!-- ------------------------------------------------------------------------------------------------------------------------------- -->


For the burnin (or "warmup") phase, Stan uses a well-established, state-of-the-art No-U-Turn HMC
(NUTS-HMC; see [Hoffman and Gelman, 2014](https://www.jmlr.org/papers/v15/hoffman14a.html)) algorithm for adaptation;
that is, for automatically (or adaptively) tuning the HMC path length ($\tau$).
On the other hand, NicoStan provides state-of-the-art, between-chain adaptation algorithms,
such as SNAPER-HMC ([Sountsov and Hoffman, 2022](https://arxiv.org/abs/2110.11576v3)),
ChEES-HMC ([Hoffman et al., 2021](https://proceedings.mlr.press/v130/hoffman21a.html)),
and ChEES-R-HMC ([Sountsov and Hoffman, 2022](https://arxiv.org/abs/2110.11576v3)) -
see [this section below](#efficient-burnin-algorithms-snaper-hmc-and-chees-r-hmc) for more details on NicoStan's burnin algorithms.


This difference in burnin adaptation algorithm makes NicoStan more efficient than Stan for most models
(see the [Benchmarks](#benchmarks) section below).
In our testing, NicoStan can also perform very well with a very short burnin
(100-125 iterations with just 4 chains; we are also currently testing shorter burnins).
Furthermore, the fact that NicoStan uses between-chain adaptation
(as opposed to within-chain adaptation, like Stan's NUTS-HMC algorithm)
means that all chains finish the sampling phase at approximately the same time, avoiding the common issue of "stuck chains",
which is often seen with complex models when using Stan directly
(e.g., via [cmdstanr](https://mc-stan.org/cmdstanr/) or [rstan](https://mc-stan.org/rstan/)).


Additionally, for models with high-dimensional nuisance parameters
(see [this section below](#models-with-nuisance-parameters-diffusion-pathspace-hmc) for examples of such models),
NicoStan offers a hybrid diffusion-pathspace HMC sampling algorithm
(based on [Beskos et al., 2011](https://doi.org/10.1016/j.spa.2011.06.003),
and [Beskos et al., 2013](https://doi.org/10.1016/j.spa.2012.12.001)),
which can greatly increase efficiency - especially for large N.

Furthermore, as we mentioned above (see [this section](#what-is-nicostan)), unlike Stan,
NicoStan allows you to set a dense HMC mass matrix for the main model parameters,
combined with a diagonal M for the high-dimensional nuisance parameters.


Note that NicoStan can also fit any Stan model (i.e., any `.stan` model file);
however, expect efficiency gains (relative to Stan) to be less dramatic for models without high-dimensional nuisance parameters.


<!-- ------------------------------------------------------------------------------------------------------------------------------- -->
## Installation
<!-- ------------------------------------------------------------------------------------------------------------------------------- -->

**Warning:** these R packages (NicoStan, NicoSIMD and BayesMVP) have not been tested on Windows in over 12 months,
and have never been tested on macOS; almost all of the testing and development has been done on Linux
(with x86 AMD CPUs). If you have any issues, please send an email to: enzo.cerullo@bath.edu.


NicoStan requires:


- An R installation, a C++17 compiler and GNU make.
- The CmdStanR interface and CmdStan C++ source tree.
See the [CmdStanR installation guide](https://mc-stan.org/cmdstanr/articles/cmdstanr.html).
- The BridgeStan R interface and matching C++ library.
See the [BridgeStan installation guide](https://roualdes.us/bridgestan/latest/languages/r.html).


The current development build uses CmdStan 2.36.0 and BridgeStan 2.6.2.

We have tested NicoStan mostly on Linux, using R 4.3.3 and GCC 11.4,
with the C++ source trees at `~/.cmdstan/cmdstan-2.36.0` and `~/.bridgestan/bridgestan-2.6.2`.


If you have any installation issues (e.g., C++ compiler errors) or bugs, please email me at:
enzo.cerullo@bath.edu, or open an issue on [GitHub](https://github.com/CerulloE1996/NicoStan/issues).

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

Note that this command installs the
[main branch](https://github.com/CerulloE1996/BayesMVP/tree/main) of [BayesMVP](https://github.com/CerulloE1996/BayesMVP).


<!-- ------------------------------------------------------------------------------------------------------------------------------- -->
## Compatibility across devices
<!-- ------------------------------------------------------------------------------------------------------------------------------- -->


**Warning:** these R packages (NicoStan, NicoSIMD and BayesMVP) have not been tested on Windows in over 12 months,
and have never been tested on macOS; almost all of the testing and development has been done on Linux
(with x86 AMD CPUs). If you have any issues, please send an email to: enzo.cerullo@bath.edu.


This section applies to all three packages:

- **NicoStan** (the sampler): written to compile on any 64-bit CPU with a C++17 compiler
(i.e., x86-64 and 64-bit ARM CPUs, on Linux, Windows and macOS).
- **NicoSIMD** (the SIMD functions, installed with NicoStan): uses AVX-512, AVX2, or NEON through SIMDe,
whichever the CPU supports, and otherwise plain Stan math (see
[NicoSIMD: the SIMD functions on every device](#nicosimd-the-simd-functions-on-every-device)).
- **[BayesMVP](https://github.com/CerulloE1996/BayesMVP)** (the built-in, manual-gradient models; see
[BayesMVP: multivariate probit models](#bayesmvp-multivariate-probit-models)): uses its own AVX2 / AVX-512 C++
functions on x86-64 CPUs, and plain C++ (i.e., no NEON version yet) on ARM CPUs.


| Device / operating system | SIMD functions used | Notes |
|---|---|---|
| Linux, Intel / AMD CPU | AVX-512 or AVX2 | tested (AMD EPYC 9654, Zen 4; AMD Ryzen 7 5800H, Zen 3) |
| Windows, Intel / AMD CPU | AVX-512 or AVX2 | requires the Clang compiler of Rtools (`clang64`); with GCC, the AVX functions are switched off |
| macOS, Intel CPU | AVX2 (or AVX-512 on some Mac Pro and iMac Pro models) | not yet tested |
| macOS, Apple Silicon (M1-M4) | NEON (via SIMDe) | not yet tested; the within-chain parallelisation of NicoStan and of the BayesMVP models uses OpenMP, which Apple's compiler does not include by default (libomp may need to be installed, e.g., via Homebrew) |
| Windows on ARM (e.g., Snapdragon X) | NEON (via SIMDe) | not yet tested |
| Any other CPU (e.g., an x86-64 CPU without AVX2) | none: plain Stan math (and Eigen) | |


Note that we have not yet tested NicoStan, BayesMVP or the SIMD functions on any ARM CPU 
(i.e., neither on Apple Silicon Macs nor on Snapdragon CPUs), nor on macOS;
hence, bug reports are very welcome
(please email me - see the [Installation](#installation) section for my email address - 
or open an issue on [GitHub](https://github.com/CerulloE1996/NicoStan/issues)).
Furthermore, our benchmarks were run on an AMD EPYC 9654 (Zen 4) and an AMD Ryzen 7 5800H (Zen 3),
so they do not apply to ARM CPUs;
the speed-ups of the NEON versions may be smaller than those of the AVX2 and AVX-512 versions.


<!-- ------------------------------------------------------------------------------------------------------------------------------- -->
## Examples
<!-- ------------------------------------------------------------------------------------------------------------------------------- -->

The general examples are grouped by their parameterisation:


- **Hybrid/joint HMC with nuisance diffusion:** Cox shared frailty, hierarchical logistic regression,
joint longitudinal-survival modelling, and a discrete-time AR(1) stochastic-volatility model
([Stochastic_volatility_discrete_time.stan](inst/examples/models/Stochastic_volatility_discrete_time.stan)).
Continuous-time SV can introduce the additional path-discretisation difficulties described
[below](#models-with-nuisance-parameters-diffusion-pathspace-hmc).
- **Standard HMC:** Weibull survival regression, robust Student-t regression and marginal Gaussian process (GP) regression.


The logistic random-intercept example gives a short introduction to the API:


```r
source(system.file("examples", "random_intercepts.R", package = "NicoStan"))
fit <-  run_random_intercepts(burnin_algorithm = "CHEESR")
diagnostics <-  fit$summary(save_log_lik_trace = FALSE)
fit$model_fit_object$summaries$summary_tibbles$summary_tibble_main_params

## Other trajectory-length adaptation options:
## "CHEESR_log", "SNAPER", "ChEES"
```


The summary tibbles report, for each parameter, the posterior mean, SD and quantiles,
the bulk and tail ESS (`n_eff` and `n_eff_tail`; [Vehtari et al., 2021](https://doi.org/10.1214/20-BA1221)),
the ESS of the centred squared draws (`n_eff_sd`, i.e., the ESS for estimating the posterior SD,
which is also the quantity the ChEES-HMC and SNAPER-HMC papers use to score samplers),
and the R-hat and nested R-hat convergence diagnostics (`Rhat` and `n_Rhat`).


This example simulates data with 20 groups and 200 observations, with a standard normal latent vector 
(i.e., the non-centred random intercepts) declared first in the Stan model.
See the complete [R example](inst/examples/random_intercepts.R) and [Stan model](inst/examples/random_intercepts.stan)
for the full parameterisation and settings.
Note that the R6 class is called `NicoStan::Nico_model` (`NicoStan::MVP_model` also still works, for compatibility reasons);
`Model_type = "Stan"` must be selected when using a `.stan` model file; the other `Model_type` options require the
[BayesMVP R package extension](https://github.com/CerulloE1996/BayesMVP) to be installed
(see [BayesMVP: multivariate probit models](#bayesmvp-multivariate-probit-models)).


The wider comparison examples are available through:


```r
source(system.file("examples", "NicoStan_examples.R", package = "NicoStan"))
result <-  run_NicoStan_example(model = "hierarchical_logistic",
                                engine = "NicoStan", math_backend = "Stan")
```


To select implementation:


- Use `engine = "cmdstanr"` for Stan/NUTS.
- Use `engine = "NicoStan", math_backend = "Stan"` for NicoStan with standard Stan mathematics.
- Use `engine = "NicoStan", math_backend = "AVX2"` for NicoStan with the custom AVX2 functions.
- Use `engine = "NicoStan", math_backend = "AVX512"` for NicoStan with the custom AVX-512 functions.

These AVX2/AVX-512 options select the prepared AVX versions of the supplied example models; they require the
[BayesMVP extension](https://github.com/CerulloE1996/BayesMVP) and a CPU supporting the selected instruction set.
For your own `.stan` model, you must first declare and call the custom AVX functions in the model and supply their C++ header;
selecting an AVX backend does not automatically replace standard Stan function calls
(see [Custom AVX2 and AVX-512 functions](#custom-avx2-and-avx-512-functions)).


### Preparing your own Stan model

When preparing a model for NicoStan:

- Declare the nuisance block **first in the Stan parameters block**.
- Set `sample_nuisance = TRUE` (or `FALSE` for a model without a nuisance block).
- Note that NicoStan automatically detects the (unconstrained) dimension of the nuisance block, 
using the Stan compiler metadata and BridgeStan.


<!-- ------------------------------------------------------------------------------------------------------------------------------- -->
## Worked examples
<!-- ------------------------------------------------------------------------------------------------------------------------------- -->


Every example below is in NicoStan's `inst/examples` folder
(i.e., `system.file("examples", package = "NicoStan")` once NicoStan is installed),
with its Stan model, its AVX version and an R script which simulates the data, fits the model and summarises it.


### Example 1: one nuisance block (logistic regression with random intercepts)


The nuisance parameters (here, the standardised random intercepts `z`) are declared **first** in the `parameters` block;
NicoStan then detects them automatically (i.e., the first declared parameter), and samples them with
diffusion-pathspace HMC (`diffusion_HMC = TRUE`), whilst the main parameters use standard HMC.
The Stan model ([random_intercepts.stan](inst/examples/random_intercepts.stan)) is:

```stan
data {
  int<lower=1> N;
  int<lower=1> J;
  array[N] int<lower=1, upper=J> group;
  vector[N] x;
  array[N] int<lower=0, upper=1> y;
}
parameters {
  vector[J] z;                  // the nuisance block: declared first
  real alpha;
  real beta;
  real<lower=0> sigma;
}
transformed parameters {
  vector[J] group_effect = sigma * z;   // non-centred random intercepts
}
model {
  z ~ std_normal();
  alpha ~ normal(0, 1.5);
  beta ~ normal(0, 1);
  sigma ~ normal(0, 1);
  y ~ bernoulli_logit(alpha + beta * x + group_effect[group]);
}
```

and it is fitted with (see [random_intercepts.R](inst/examples/random_intercepts.R) for the data simulation):

```r
model <-  NicoStan::Nico_model$new(Model_type = "Stan",
                                    Stan_data_list = stan_data,
                                    Stan_model_file_path = "random_intercepts.stan",
                                    sample_nuisance = TRUE)
initial_values <-  lapply(X = 1:4, FUN = function(chain) {
                       list(z = rep(0, n_groups), alpha = 0.1 * (chain - 2.5), beta = 0, sigma = 0.6)
                   })
model$sample(n_burnin = 125, n_iter = 250,
             n_chains_burnin = 4, n_chains_sampling = 4, n_superchains = 2,
             init_lists_per_chain = initial_values,
             burnin_algorithm = "CHESSR",
             diffusion_HMC = TRUE, partitioned_HMC = FALSE,
             seed = 2026)
fit_summary <-  model$summary(save_log_lik_trace = FALSE)
```

This model has no `reduce_sum()`, so each chain runs on one thread, with no chunks.
For a model that uses `reduce_sum()`, NicoStan finds the optimal within-chain threads (WCP) and
$N_{\text{chunks}}$ of the burn-in and of the sampling for you, from the L2 and L3 caches of the computer,
either on their own before the fit or inside `$sample()`; see step (d) of
[Example 3](#example-3-the-multivariate-ordered-probit-model-with-the-fused-ghk-function-and-bayesmvp).


### Example 2: several nuisance blocks (`nuisance_params`)


A model may have any number of nuisance blocks, declared anywhere in the `parameters` block;
they are given by name with `nuisance_params`, e.g., for random intercepts and random slopes:

```stan
parameters {
  real alpha;
  real beta;
  real<lower=0> sigma_intercept;
  real<lower=0> sigma_slope;
  vector[J] z_intercept;        // nuisance block 1
  vector[J] z_slope;            // nuisance block 2
}
model {
  z_intercept ~ std_normal();
  z_slope ~ std_normal();
  alpha ~ normal(0, 1.5);
  beta ~ normal(0, 1);
  sigma_intercept ~ normal(0, 1);
  sigma_slope ~ normal(0, 1);
  y ~ bernoulli_logit(alpha + sigma_intercept * z_intercept[group]
                      + (beta + sigma_slope * z_slope[group]) .* x);
}
```

```r
model <-  NicoStan::Nico_model$new(Model_type = "Stan",
                                    Stan_data_list = stan_data,
                                    Stan_model_file_path = "random_intercepts_and_slopes.stan",
                                    nuisance_params = c("z_intercept", "z_slope"),
                                    sample_nuisance = TRUE)
```

The initial values of each block are given by name, in its own shape (e.g., `z_intercept = rep(0, J)`);
a block left out starts at 0. Example 3 shows every step of a full fit (the priors, the nuisance blocks, the
initial values, the optimal layouts and the summaries).

With `nuisance_params`, NicoStan writes a version of the model (`<model>_reduce_sum_chunked.stan`, next to it)
whose nuisance blocks are held by NicoStan, rather than declared as Stan parameters
(the blocks one after the other, each in Stan's own order of its values);
the rest of the model is unchanged
(see [reduce_sum_chunked](#reduce_sum_chunked-nuisance-blocks-held-inside-the-chunks) for what this gives when
the model uses `reduce_sum`).
Note that:

- **How many blocks:** any number (each one named in `nuisance_params`).
- **Which shapes:** unconstrained `vector`, `row_vector`, `array[] real`, `array[] vector` and `array[] row_vector`
blocks, and tuples of these (see below); any other block (e.g., with a constraint, a matrix, or an array of tuples)
stops with an error saying so.
- **Tuples:** a nuisance block may also be a tuple, so that blocks of different containers can be declared (and
named in `nuisance_params`) at once, e.g.:

```stan
parameters {
  real alpha;
  real beta;
  real<lower=0> sigma_intercept;
  real<lower=0> sigma_slope;
  tuple(vector[J], vector[J]) z;   // z.1: random intercepts, z.2: random slopes
}
model {
  z.1 ~ std_normal();
  z.2 ~ std_normal();
  ...
}
```

with `nuisance_params = c("z")`. Each component of the tuple (including those of any tuple inside it) is then held
as a nuisance block of its own, in the tuple's own order (which is also Stan's order of its values).
Note that an array of tuples (e.g., `array[N] tuple(real, vector[K])`) is not supported;
declare it as a tuple of arrays instead (e.g., `tuple(array[N] real, array[N] vector[K])`).


### Example 3: the multivariate ordered probit model with the fused GHK function (and BayesMVP)


The multivariate ordered probit (MVOP) model - i.e., a truncated multivariate normal for each subject, whose
truncation region is set by the subject's binary and ordinal results - can be written in Stan with one of NicoSIMD's
fused functions, `nico_finite_mixture_of_multivariate_ordered_probit_GHK_log_likelihood`, which computes the GHK
log likelihood of a chunk of subjects, and its gradient, in one autodiff node.
It runs on any CPU with AVX-512 or AVX2, and on 64-bit ARM CPUs (e.g., Apple Silicon Macs), where NicoSIMD's
functions run with NEON instructions (through SIMDe).
The same function also takes 2 or 4 latent classes (i.e., the LC-MVP and the LC-MVOP), binary, ordinal and mixed
outcomes, each outcome's own covariates, missing results (coded -1, handled by the GHK method) and
interval-censored ordinal results.


This example fits a one-class MVOP to the real NHANES data (the 9575 US adults of the hexbin R package),
with four outcomes, each with its own covariates:
iron deficiency, low albumin and cancer (binary), and anaemia (ordinal: the WHO categories of haemoglobin for each
sex, i.e., none / mild / moderate / severe).
The function is declared in the `functions` block without a body, and the nuisance block `u_raw` (one GHK value for
each subject and outcome) is the sliced argument of `reduce_sum_static`, so that everything whose size grows with N
is inside the chunks; the priors (normal priors on the coefficients, an LKJ prior on the correlation matrix and an
induced Dirichlet prior on the cutpoints of each ordinal outcome) are those of BayesMVP's MVOP model.
The key parts of the Stan model
([one_class_MVOP_with_the_fused_GHK_log_likelihood.stan](inst/examples/models/one_class_MVOP_with_the_fused_GHK_log_likelihood.stan))
are:

```stan
functions {
  real nico_finite_mixture_of_multivariate_ordered_probit_GHK_log_likelihood(...);   // declared, no body
  real partial_sum_of_the_GHK_log_likelihood_lpdf(array[] row_vector u_raw_slice, int start, int end, ...) {
    matrix[end - start + 1, 1] log_class_probability = rep_matrix(0.0, end - start + 1, 1);
    return nico_finite_mixture_of_multivariate_ordered_probit_GHK_log_likelihood(
             u_raw_slice, 1, intercept, coefficients, set_of_class_and_test,
             covariates[:, :, start:end], n_covariates, cutpoints, set_of_class_and_test,
             position_of_first_cutpoint_of_test, L_Omega, log_class_probability,
             y[start:end], y[start:end], test_is_binary, n_categories_of_test, test_at_GHK_position,
             Phi_type, overflow_threshold, underflow_threshold);
  }
}
parameters {
  array[N] row_vector[n_tests] u_raw;      // the nuisance block (given with nuisance_params)
  vector[n_coefficients_total] beta_vec;   // the coefficients of each outcome (its intercept first)
  cholesky_factor_corr[n_tests] L_Omega;
  vector<lower=C_raw_lower, upper=C_raw_upper>[n_cutpoints_total] C_raw;   // first cutpoint, then log gaps
}
model {
  ...   // the priors, including the induced Dirichlet prior on the cutpoints
  target += reduce_sum_static(partial_sum_of_the_GHK_log_likelihood_lpdf, u_raw, chunk_size, ...);
}
```

NicoSIMD gives the user header and the build arguments for the CPU in use, so that the same R code runs on every
device. The full R script
([Run_one_class_MVOP_with_the_fused_GHK_log_likelihood_and_BayesMVP.R](inst/examples/Run_one_class_MVOP_with_the_fused_GHK_log_likelihood_and_BayesMVP.R))
builds the NHANES outcomes and covariates (`y`, `X_of_each_outcome`, `n_covariates_of_test`), and then fits the
model in five steps: (a) the priors; (b) the nuisance parameters; (c) the initial values;
(d) the optimal layouts of the burn-in and of the sampling; and (e) the fit and its summaries.


**(a) The priors.**
The priors are set in the Stan model's `model` block, and their values are passed to it as data,
so that they can be changed without editing the model (to change the form of a prior, edit the `model` block):

```r
## ---- The priors (the same for NicoStan and BayesMVP, hence the same posterior):
prior_LKJ_eta                       <-  2     ## L_Omega ~ lkj_corr_cholesky(prior_LKJ_eta)
prior_mean_of_each_coefficient      <-  0     ## beta_vec ~ normal(prior_beta_mean_vec, prior_beta_sd_vec)
prior_SD_of_each_intercept          <-  2.5
prior_SD_of_each_other_coefficient  <-  1
prior_Dirichlet_alpha               <-  1     ## the induced Dirichlet prior on the cutpoints of each ordinal test
C_raw_lower                         <-  -10   ## the bounds of the first cutpoint and of the logs of the gaps
C_raw_upper                         <-  2.5
##
## ---- The priors of the coefficients as matrices (covariates x outcomes; the intercepts in the first row):
prior_beta_mean <-  matrix(prior_mean_of_each_coefficient, nrow = n_covariates_max, ncol = n_tests)
prior_beta_sd <-  matrix(prior_SD_of_each_other_coefficient, nrow = n_covariates_max, ncol = n_tests)
prior_beta_sd[1, ] <-  prior_SD_of_each_intercept
##
Stan_data_list <-  list(N = N, n_tests = n_tests,
                        test_is_binary = c(1, 1, 1, 0), n_categories_of_test = c(2, 2, 2, 4),
                        y = y, n_covariates_max = n_covariates_max, n_covariates_of_test = n_covariates_of_test,
                        X = X_of_each_outcome,
                        ## (the priors' values)
                        prior_beta_mean = prior_beta_mean, prior_beta_sd = prior_beta_sd,
                        prior_LKJ_eta = prior_LKJ_eta, prior_Dirichlet_alpha = prior_Dirichlet_alpha,
                        C_raw_lower = C_raw_lower, C_raw_upper = C_raw_upper,
                        ## (Phi_type 1: the exact normal CDF; the bounds of the log-scale tails)
                        Phi_type = 1, overflow_threshold = 7.5, underflow_threshold = -7.5,
                        ## (reduce_sum's data: NicoStan sets chunk_size for each layout)
                        chunk_size = N)
```


**(b) The nuisance parameters.**
The nuisance block `u_raw` (declared in the `parameters` block as `array[N] row_vector[n_tests] u_raw;`) is given
to NicoStan by name with `nuisance_params`; any number of nuisance blocks can be given this way
(e.g., `nuisance_params = c("z_intercept", "z_slope")`, see
[Example 2](#example-2-several-nuisance-blocks-nuisance_params)).
NicoStan then samples the nuisance blocks with diffusion-pathspace HMC (`diffusion_HMC = TRUE` in `$sample()`) and
the main parameters with standard HMC.
Since `u_raw` is the sliced argument of `reduce_sum_static`, NicoStan holds its values inside the chunks
(reduce_sum_chunked), i.e., off Stan's main autodiff tape.
Note that a single nuisance block declared first in the `parameters` block is also found with
`sample_nuisance = TRUE` alone (as in
[Example 1](#example-1-one-nuisance-block-logistic-regression-with-random-intercepts)):

```r
NicoSIMD_build <-  NicoSIMD::fn_include_folder_Stan_user_header_and_build_arguments_of_NicoSIMD_for_this_CPU()
model_NicoStan <-  NicoStan::Nico_model$new(Model_type            = "Stan",
                                            Stan_data_list        = Stan_data_list,
                                            Stan_model_file_path  = Stan_model_file_path,
                                            Stan_cpp_user_header  = NicoSIMD_build$Stan_user_header,
                                            make_args             = NicoSIMD_build$make_args,
                                            nuisance_params       = "u_raw",
                                            sample_nuisance       = TRUE)
```


**(c) The initial values** of each burn-in chain are given as a list, by parameter name; a nuisance block is given
in its own shape (here, `u_raw` as an N x n_tests matrix, one row per subject), and a nuisance block left out
starts at 0:

```r
n_chains_burnin <-  4
initial_values_NicoStan <-  rep(list(list(u_raw     = matrix(0, nrow = N, ncol = n_tests),
                                          beta_vec  = rep(0, sum(n_covariates_of_test)),
                                          L_Omega   = diag(n_tests),
                                          C_raw     = c(1.0, log(0.5), log(1.5)))), n_chains_burnin)
```


**(d) The optimal layouts of the burn-in and of the sampling.**
NicoStan finds the optimal number of within-chain threads (WCP) and the optimal $N_{\text{chunks}}$,
separately for the burn-in and for the sampling, from the L2 and L3 caches of the computer it runs on,
and then checks them by timing a few layouts around them, within a budget of 2000 of the model's own gradient
evaluations. Nothing is chosen by hand. The layouts can be found on their own, before the fit:

```r
layouts_NicoStan <-
      NicoStan::fn_compute_burnin_and_sampling_layouts_of_Stan_model_within_a_budget_of_lp_grad_calls(
            model_so_file    = model_NicoStan$init_object$model_so_file,
            Stan_data_list   = Stan_data_list,
            init_object      = model_NicoStan$init_object,
            n_chains_burnin  = n_chains_burnin)
## (layouts_NicoStan$n_threads_WCP_burnin, $num_chunks_burnin, $n_chains_sampling, $n_threads_WCP_sampling,
##  $num_chunks_sampling, and $seconds_spent_choosing_the_burnin_and_sampling_layouts)
```


**(e) The fit**, with these layouts carried over into `$sample()`, and its summaries:

```r
model_NicoStan$sample(init_lists_per_chain     = initial_values_NicoStan,
                      n_burnin                 = 500,
                      n_iter                   = 500,
                      n_chains_burnin          = n_chains_burnin,
                      burnin_algorithm         = "CHESSR",
                      diffusion_HMC            = TRUE,
                      partitioned_HMC          = FALSE,
                      n_threads_WCP_burnin     = layouts_NicoStan$n_threads_WCP_burnin,
                      num_chunks_burnin        = layouts_NicoStan$num_chunks_burnin,
                      n_chains_sampling        = layouts_NicoStan$n_chains_sampling,
                      n_threads_WCP_sampling   = layouts_NicoStan$n_threads_WCP_sampling,
                      num_chunks_sampling      = layouts_NicoStan$num_chunks_sampling,
                      seed                     = 123)
fit_summary_NicoStan <-  model_NicoStan$summary(save_log_lik_trace = FALSE)
fit_summary_NicoStan$get_summary_main()           ## the main parameters
fit_summary_NicoStan$get_summary_transformed()    ## the transformed parameters (e.g., the correlation matrix)
```

> **Note:** if you leave these out (or set them to `NULL`), NicoStan will find them for you, as long as your model
> uses `reduce_sum()` (with `N` and `chunk_size` in its data) or is a built-in BayesMVP model.


**The same model with BayesMVP (manual gradients).**
BayesMVP's built-in MVOP model has the same likelihood and priors, but its gradients are derived by hand (rather
than by autodiff), which makes it faster still. With the BayesMVP extension installed, the same five steps are:

```r
## ---- (a) The priors, in model_args_list (each a list with one element per latent class; one class here), with
##      the same values as above:
model_args_list <-  list(y                             = y,
                         n_cat_per_ord_test            = 4,
                         X                             = list(X_of_each_outcome),
                         n_covariates_per_outcome_mat  = matrix(n_covariates_of_test, nrow = 1),
                         prior_coeffs_mean_mat         = list(prior_beta_mean),
                         prior_coeffs_sd_mat           = list(prior_beta_sd),
                         lkj_cholesky_eta              = matrix(prior_LKJ_eta, nrow = 1, ncol = 1),
                         prior_dirichlet_alpha         = list(matrix(prior_Dirichlet_alpha, nrow = 4, ncol = 1)),
                         C_raw_lower                   = C_raw_lower,
                         C_raw_upper                   = C_raw_upper,
                         Phi_type                      = "Phi")
##
## ---- (b) The nuisance parameters are built in (u_raw); sample_nuisance = TRUE samples them with diffusion HMC:
model_BayesMVP <-  BayesMVP::MVP_model$new(Model_type       = "MVOP",
                                           model_args_list  = model_args_list,
                                           sample_nuisance  = TRUE)
##
## ---- (c) The same initial values (BayesMVP's cutpoints are unconstrained by tanh, between the bounds):
C_unc_initial <-  atanh(2 * (c(1.0, log(0.5), log(1.5)) - C_raw_lower) / (C_raw_upper - C_raw_lower) - 1)
initial_values_BayesMVP <-  rep(list(list(u_raw                    = matrix(0, nrow = N, ncol = n_tests),
                                          beta                     = list(prior_beta_mean),
                                          Omega_unconstrained_vec  = list(rep(0, choose(n_tests, 2))),
                                          C_unc_vec                = list(C_unc_initial))), n_chains_burnin)
##
## ---- (d) The optimal layouts of the burn-in and of the sampling, found by BayesMVP from the L2 and L3 caches:
burnin_layout_BayesMVP <-  BayesMVP::fn_compute_burnin_n_threads_WCP_and_num_chunks_for_BayesMVP_model(
                                 Model_type       = "MVOP",
                                 init_object      = model_BayesMVP$init_object,
                                 n_chains_burnin  = n_chains_burnin)
sampling_layout_BayesMVP <-
      BayesMVP::fn_compute_sampling_n_chains_n_threads_WCP_and_num_chunks_for_BayesMVP_model(
            Model_type   = "MVOP",
            init_object  = model_BayesMVP$init_object)
##
## ---- (e) The fit, with these layouts carried over into $sample(), and its summaries:
model_BayesMVP$sample(init_lists_per_chain     = initial_values_BayesMVP,
                      n_burnin                 = 500,
                      n_iter                   = 500,
                      n_chains_burnin          = n_chains_burnin,
                      burnin_algorithm         = "CHESSR",
                      diffusion_HMC            = TRUE,
                      partitioned_HMC          = FALSE,
                      n_threads_WCP_burnin     = burnin_layout_BayesMVP$n_threads_WCP_burnin,
                      num_chunks_burnin        = burnin_layout_BayesMVP$num_chunks_burnin,
                      n_chains_sampling        = sampling_layout_BayesMVP$n_chains_sampling,
                      n_threads_WCP_sampling   = sampling_layout_BayesMVP$n_threads_WCP_sampling,
                      num_chunks_sampling      = sampling_layout_BayesMVP$num_chunks_sampling,
                      seed                     = 123,
                      reorder_cols_MVP         = FALSE)
fit_summary_BayesMVP <-  model_BayesMVP$summary(save_log_lik_trace = FALSE)
fit_summary_BayesMVP$get_summary_main()
```

> **Note:** here too, if you leave the layouts out (or set them to `NULL`), BayesMVP will find them for you.

The R script prints the posterior means of the correlation matrix from both fits, side by side.


### Example 4: network meta-analysis of test accuracy with multiple thresholds (MetaOrdDTA's Xu model)


MetaOrdDTA's network meta-analysis (NMA) model of diagnostic test accuracy - i.e., Nyaga et al.'s NMA model for
several tests at once, with Xu et al.'s fixed cutpoints (one set of cutpoints per test and disease class,
shared by every study) - is a Stan model with two nuisance blocks of standard-normal between-study values, which
NicoStan can fit unchanged.
Each study reports, for each of its tests, the number of non-diseased and of diseased patients above each of the
test's thresholds; the thresholds a study did not report are coded -1 (the "missing thresholds" the model is made
for).


This example fits it to simulated data: 4 ordinal anxiety screening tests (GAD-2, GAD-7, HADS and BAI) in 80
studies, each with 1 to 3 of them, simulated with MetaOrdDTA's own simulator
(`R_fn_sim_NMA_data_with_covariates_varying()`) from a realistic design, with the missing thresholds of a real
NMA.
MetaOrdDTA prepares the data and the priors exactly as it gives them to Stan (`MetaOrd_model$new(..., compile =
FALSE)`: nothing is compiled or fitted by MetaOrdDTA), and NicoStan fits its Stan model
(`DTA_NMA_Nyaga_Xu_FIXEDthr.stan`, with its `#include` files from MetaOrdDTA's `stan_functions` folder).
The full R script
([Run_network_meta_analysis_of_test_accuracy_Xu_fixed_cutpoints_with_MetaOrdDTA_and_NicoStan.R](inst/examples/Run_network_meta_analysis_of_test_accuracy_Xu_fixed_cutpoints_with_MetaOrdDTA_and_NicoStan.R))
reads the data, and then fits the model in five steps: (a) the priors; (b) the nuisance parameters; (c) the
initial values; (d) the layouts; and (e) the fit and its summaries.

```r
simulated_NMA_data <-  readRDS(system.file("examples", "data",
                                           paste0("simulated_network_meta_analysis_of_four_ordinal_screening_",
                                                  "tests_with_missing_thresholds.rds"),
                                           package = "NicoStan"))
## (x[[t]][[c]]: test t, class c (1 = non-diseased, 2 = diseased), one row per study; its first column is the
##  number of patients of the class, the others the number above each threshold, -1 when it was not reported)
x <-  lapply(simulated_NMA_data$x, unname)
indicator_index_test_in_study <-  simulated_NMA_data$indicator_index_test_in_study
n_studies <-  simulated_NMA_data$n_studies
##
model_MetaOrdDTA <-  MetaOrdDTA::MetaOrd_model$new(x                              = x,
                                                   indicator_index_test_in_study  = indicator_index_test_in_study,
                                                   intercept_only                 = TRUE,
                                                   network                        = TRUE,
                                                   model_parameterisation         = "Xu",
                                                   random_thresholds              = FALSE,
                                                   Dirichlet_random_effects_type  = "fixed",
                                                   n_chains                       = 4,
                                                   compile                        = FALSE)
```


**(a) The priors.**
The priors are set in the Stan model's `model` block, and their values are passed to it as data (MetaOrdDTA's
defaults, written out below so that they can be changed; to change the form of a prior, edit the `model` block):

```r
## ---- The priors (MetaOrdDTA's defaults for this model):
prior_mean_of_each_beta_mu    <-  0   ## beta_mu[t][c, ] ~ normal(prior_beta_mu_mean, prior_beta_mu_SD)
prior_SD_of_each_beta_mu      <-  1
prior_SD_of_each_beta_sigma   <-  1   ## beta_sigma[c] ~ normal(0, prior_beta_sigma_SD[c])     (half-normal)
prior_SD_of_each_beta_tau     <-  1   ## beta_tau_raw ~ normal(0, prior_beta_tau_SD)           (half-normal)
prior_LKJ_eta_of_beta_corr    <-  2   ## beta_L_Omega ~ lkj_corr_cholesky(prior_beta_corr_LKJ)
prior_Dirichlet_alpha         <-  1   ## the induced Dirichlet prior on the cutpoints of each test and class
##
priors <-  model_MetaOrdDTA$priors
for (t in seq_along(priors$prior_beta_mu_mean)) {
      priors$prior_beta_mu_mean[[t]][] <-  prior_mean_of_each_beta_mu
      priors$prior_beta_mu_SD[[t]][] <-  prior_SD_of_each_beta_mu
}
priors$prior_beta_sigma_SD[] <-  prior_SD_of_each_beta_sigma
priors$prior_beta_tau_SD[] <-  prior_SD_of_each_beta_tau
priors$prior_beta_corr_LKJ <-  prior_LKJ_eta_of_beta_corr
## (one matrix per class, tests x categories, as the Stan model declares it)
n_thr_max <-  model_MetaOrdDTA$internal_obj$outs_data$stan_data_list$n_thr_max
alpha_of_each_class <-  matrix(prior_Dirichlet_alpha, nrow = length(x), ncol = n_thr_max + 1)
priors$prior_dirichlet_alpha <-  list(alpha_of_each_class, alpha_of_each_class)
##
## ---- The data list, then the priors (names given twice kept once, as MetaOrdDTA does), with the model's
##      0 / 1 switches as numbers:
Stan_data_list <-  MetaOrdDTA:::R_fn_remove_duplicates_from_list(c(model_MetaOrdDTA$internal_obj$outs_data$
                                                                       stan_data_list,
                                                                   priors))
for (name_of_switch in c("softplus", "hetero_sigma", "compound_symmetry")) {
      Stan_data_list[[name_of_switch]] <-  as.integer(Stan_data_list[[name_of_switch]])
}
```


**(b) The nuisance parameters.**
The model's two nuisance blocks are declared first in its `parameters` block,

```stan
parameters {
  matrix[2, n_studies] beta_eta_z;                 // each study's effect in each class
  array[2] vector[n_total_obs] beta_delta_z_vec;   // each study-test pair's effect in each class
  ...
}
```

with `n_total_obs` the number of study-test pairs. As they are the first `2 n_studies + 2 n_total_obs`
parameters, `sample_nuisance = TRUE` with `n_nuisance_override = 2 n_studies + 2 n_total_obs` gives both blocks to
NicoStan, which samples them with diffusion-pathspace HMC (`diffusion_HMC = TRUE` in `$sample()`) and the main
parameters with standard HMC; the `#include` files are found with `stanc_args`:

```r
Stan_model_file_path <-  file.path(tempdir(), "DTA_NMA_Nyaga_Xu_FIXEDthr.stan")
file.copy(from = system.file("stan_models", "stan_models_NMA", "DTA_NMA_Nyaga_Xu_FIXEDthr.stan",
                             package = "MetaOrdDTA"),
          to = Stan_model_file_path, overwrite = TRUE)
Stan_functions_folder <-  system.file("stan_functions", package = "MetaOrdDTA")
n_total_obs <-  sum(indicator_index_test_in_study)
model_NicoStan <-  NicoStan::Nico_model$new(Model_type            = "Stan",
                                            Stan_data_list        = Stan_data_list,
                                            Stan_model_file_path  = Stan_model_file_path,
                                            stanc_args            = list(paste0("--include-paths=",
                                                                                Stan_functions_folder)),
                                            sample_nuisance       = TRUE,
                                            n_nuisance_override   = 2 * n_studies + 2 * n_total_obs)
```

Here this gives 432 nuisance parameters (2 x 80 studies + 2 x 136 study-test pairs) and 241 main parameters.


**(c) The initial values.**
The initial values of the burn-in chains are given by name (the nuisance blocks at 0, i.e., every study at the
average):

```r
n_index_tests <-  length(x)
initial_values_NicoStan <-  rep(list(list(beta_eta_z        = matrix(0, nrow = 2, ncol = n_studies),
                                          beta_delta_z_vec  = list(rep(0, n_total_obs), rep(0, n_total_obs)),
                                          beta_mu           = rep(list(matrix(0, nrow = 2, ncol = 1)),
                                                                  n_index_tests),
                                          C_raw_vec         = list(rep(0, sum(Stan_data_list$n_thr)),
                                                                   rep(0, sum(Stan_data_list$n_thr))),
                                          beta_sigma        = c(0.25, 0.25),
                                          beta_tau_raw      = matrix(0.25, nrow = n_index_tests, ncol = 2),
                                          beta_corr         = 0)),
                                4)
```


**(d) The layouts.**
This model has no `reduce_sum()` (its N is the number of studies, which is small), so every chain runs on one
thread, without chunks, and NicoStan runs one chain per CPU in the sampling.
For a model whose likelihood is in `reduce_sum()` chunks, see Example 3, where NicoStan finds the optimal
within-chain threads (WCP) and `N_chunks` of the burn-in and of the sampling by itself, from the L2 and L3 caches
of the computer.


**(e) The fit and its summaries.**

```r
model_NicoStan$sample(init_lists_per_chain     = initial_values_NicoStan,
                      n_burnin                 = 500,
                      n_iter                   = 500,
                      n_chains_burnin          = 4,
                      burnin_algorithm         = "CHESSR",
                      diffusion_HMC            = TRUE,
                      partitioned_HMC          = FALSE,
                      seed                     = 123)
fit_summary <-  model_NicoStan$summary(save_log_lik_trace = FALSE)
fit_summary$get_summary_main()
```

> **Note:** if you leave the layouts out (or set them to `NULL`), NicoStan will find them for you, as long as your
> model uses `reduce_sum()` or is a built-in BayesMVP model.


### The example models


| Model | Nuisance block (declared first) | Stan model | AVX version | R script |
|---|---|---|---|---|
| Logistic regression with random intercepts | `z[J]` | [random_intercepts.stan](inst/examples/random_intercepts.stan) | - | [random_intercepts.R](inst/examples/random_intercepts.R) |
| Stochastic volatility (discrete-time AR(1)) | `h_std[T]` | [EASY_Stochastic_volatility.stan](inst/examples/models/EASY_Stochastic_volatility.stan) | [_avx](inst/examples/models/EASY_Stochastic_volatility_avx.stan) | [Run_EASY_Stochastic_volatility.R](inst/examples/Run_EASY_Stochastic_volatility.R) |
| Shared-frailty Cox survival | `log_frailty_raw[J]` | [Frailty_surv_shared_frailty_Cox.stan](inst/examples/models/Frailty_surv_shared_frailty_Cox.stan) | [_avx](inst/examples/models/Frailty_surv_shared_frailty_Cox_avx.stan) | [Run_Frailty_surv_shared_Cox.R](inst/examples/Run_Frailty_surv_shared_Cox.R) |
| Latent diffusion survival | `w[M]` | [Latent_diffusion_survival.stan](inst/examples/models/Latent_diffusion_survival.stan) | [_avx](inst/examples/models/Latent_diffusion_survival_avx.stan) | [Run_Latent_diffusion_survival.R](inst/examples/Run_Latent_diffusion_survival.R) |
| Joint longitudinal and survival | `b_raw[2, N_subj]` | [Joint_longitudinal_surv.stan](inst/examples/models/Joint_longitudinal_surv.stan) | [_avx](inst/examples/models/Joint_longitudinal_surv_avx.stan) | [Run_joint_longitudinal.R](inst/examples/Run_joint_longitudinal.R) |
| Hierarchical logistic regression | `beta_raw[K, 2]` | [hierarchical_logistic_regression.stan](inst/examples/models/hierarchical_logistic_regression.stan) | [_avx](inst/examples/models/hierarchical_logistic_regression_avx.stan) | [Run_hierarchical_logistic_regression.R](inst/examples/Run_hierarchical_logistic_regression.R) |
| Weibull survival | none | [standard_surv_Weibull.stan](inst/examples/models/standard_surv_Weibull.stan) | [_avx](inst/examples/models/standard_surv_Weibull_avx.stan) | [Run_standard_surv_Weibull.R](inst/examples/Run_standard_surv_Weibull.R) |
| Robust Student-t regression | none | [robust_regression_t4.stan](inst/examples/models/robust_regression_t4.stan) | [_avx](inst/examples/models/robust_regression_t4_avx.stan) | [Run_robust_t4_regression.R](inst/examples/Run_robust_t4_regression.R) |
| Gaussian process regression (marginal) | none | [Gaussian_process.stan](inst/examples/models/Gaussian_process.stan) | [_avx](inst/examples/models/Gaussian_process_avx.stan) | [Run_Gaussian_process.R](inst/examples/Run_Gaussian_process.R) |
| One-class multivariate ordered probit (real NHANES data) | `u_raw[N]` (row vectors) | [one_class_MVOP_with_the_fused_GHK_log_likelihood.stan](inst/examples/models/one_class_MVOP_with_the_fused_GHK_log_likelihood.stan) | (NicoSIMD's fused GHK function) | [Run_one_class_MVOP_with_the_fused_GHK_log_likelihood_and_BayesMVP.R](inst/examples/Run_one_class_MVOP_with_the_fused_GHK_log_likelihood_and_BayesMVP.R) |


The model definitions, priors and nuisance blocks of each are described in
[MODEL_NOTES.md](inst/examples/MODEL_NOTES.md).
The full code of every example model and R script (each file a click away, with a Copy button) is on
[one page of the website](https://cerulloe1996.github.io/NicoStan/example_models_full_code.html).


### The BayesMVP models


With the [BayesMVP](https://github.com/CerulloE1996/BayesMVP) extension installed, the built-in
(manual-gradient) models each have an R script, which runs three sample sizes:
[Run_MVP.R](inst/examples/Run_MVP.R), [Run_2LC_MVP.R](inst/examples/Run_2LC_MVP.R),
[Run_MVOP.R](inst/examples/Run_MVOP.R), [Run_2LC_MVOP.R](inst/examples/Run_2LC_MVOP.R) and
[Run_latent_trait.R](inst/examples/Run_latent_trait.R).
Furthermore, the four-class LC-MVOP (4LC-MVOP), which extends the LC-MVOP to two conditions at once
(i.e., its latent classes are the 2 x 2 combinations of the two conditions),
is in [four_class/](inst/examples/four_class), with its plain, AVX and fused Stan models and synthetic data
([Run_4LC_MVOP.R](inst/examples/Run_4LC_MVOP.R)).
The BayesMVP R package also has two example data sets:
binary data for the LC-MVP, simulated from real, publicly available COVID-19 data;
and ordinal data for the LC-MVOP, based on a real South African depression data set
(with the PHQ-9 and the CES-D-10 as the two ordinal index tests, and the MINI as the binary reference test).


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


These models handle correlated binary and/or ordinal outcomes, and their latent class extensions (LC-MVP, LC-MVOP) allow
diagnostic/screening test accuracy to be estimated when a perfect reference/gold standard is unavailable.

The correlation matrices use the flexible Cholesky parameterisation of [Sean Pinkney](https://github.com/spinkney)
([Pinkney, 2024](https://arxiv.org/abs/2405.07286)), which also allows the correlations to be constrained to be positive
(`corr_force_positive = TRUE`), as well as custom element-specific bounds (via the `lb_corr` and `ub_corr` arguments, with one
matrix per latent class). Furthermore, any of the correlations can be fixed to known values, via the `known_values_indicator_list`
argument (which indicates which correlations are known) and the `known_values_list` argument (their values).


The ordinal latent class model is described in [Cerullo et al., 2022](https://doi.org/10.1002/jrsm.1567).
Furthermore, in [Cerullo et al., 2025](https://arxiv.org/abs/2509.18489v1) we compared the LC-MVP and latent trait models
in a comprehensive simulation study, in which the models were fitted using [BayesMVP](https://github.com/CerulloE1996/BayesMVP).


<!-- The **4LC-MVOP** model handles two related latent conditions. -->
<!-- It has a Stan implementation and a partial-manual-gradient/fused-function version. -->
<!-- Its partial-manual-gradient implementation runs through Stan/NicoStan, -->
<!-- whilst the five models above also have native BayesMVP implementations.  -->
<!-- Development and applications of 4LC-MVOP are covered in separate papers,  -->
<!-- whilst the model also provides a larger example for the general NicoStan sampler. -->


### Automatic test (column) reordering


For the latent class models (LC-MVP and LC-MVOP), BayesMVP chooses the order of the tests
(i.e., the columns of `y`) automatically, before the main burnin (`reorder_cols_MVP = TRUE` by default).
The model itself does not depend on the order of the tests;
however, the unconstrained parameters which HMC samples do - and hence so does the efficiency of HMC.

- A short pre-burnin gives rough estimates of the correlations, intercepts and (for ordinal tests) cutpoints;
the most nearly deterministic test is then placed last,
and the other tests follow the greedy correlation order (the most strongly correlated tests first).
- This is because, in the GHK construction, each test's truncation bounds depend on the tests placed before it,
and a nearly deterministic test has steep tail values which would otherwise propagate to every later test;
the order also sets the order of the correlation parameterisation
([Pinkney, 2024](https://arxiv.org/abs/2405.07286)).
- On our binary LC-MVP model (N = 10,000), the slower order of the old greedy-only rule needed 1.60× the
gradient evaluations to reach the target min ESS (0.59× the min ESS per 1000 gradient evaluations);
on our ordinal LC-MVOP model (N = 5,000), placing the binary test last needed 0.67-0.75× the gradient evaluations
of the input order.
- The summaries and traces are returned in the original test order.

See the [BayesMVP README](https://github.com/CerulloE1996/BayesMVP#automatic-test-column-reordering)
for more details.


<!-- ------------------------------------------------------------------------------------------------------------------------------- -->
## Benchmarks
<!-- ------------------------------------------------------------------------------------------------------------------------------- -->


In our initial tests, NicoStan - combined with the [BayesMVP extension](https://github.com/CerulloE1996/BayesMVP) -
was **over 150× faster than Stan**, and around **50× more efficient than Mplus**.
These numbers are for the latent class multivariate probit model (LC-MVP),
using an N = 10,000 simulated binary dataset with 6 binary tests in total (hence a total of 60,000 observations),
which is based on real, publicly available COVID-19 data.
These models (LC-MVP) are used in both human and veterinary medicine for estimating diagnostic/screening test accuracy
without a perfect gold standard.


For our ordinal example dataset (using the LC-MVOP model),
which is based on a real dataset on three tests to screen and/or diagnose depression
(specifically, the MINI as the imperfect gold standard, and the PHQ-9 and CES-D-10 as the two ordinal tests),
for N = 5,000 (hence 15,000 total observations since there's 3 tests), we found that NicoStan/BayesMVP (using the LC-MVOP model)
was **over 50× more efficient than Stan** and **over 1000× more efficient than Mplus**.
The latter is because, desite being very efficient for the binary LC-MVP,
Mplus's Gibbs-based algorithm struggles greatly with ordinal outcomes,
and Mplus does not let you fit ordinal outcomes with more than 10 categories;
hence, we had to group categories together just to attempt to measure its efficiency.
Note that this is not ideal, and we would never recommend grouping categories in real-life data analysis,
nor would we suggest assuming continuity just because there are a lot of categories
(especially for single studies and for latent class models).


Furthermore, even when using NicoStan **without** BayesMVP
(i.e., using the `.stan` model file directly - without the manually coded C++ gradients and AVX functions, etc.),
we still found that NicoStan was **over 10× more efficient than Stan** (via cmdstanr),
for both the LC-MVP and the LC-MVOP examples described above.
These benchmarks use the standard Stan maths functions, so the gains reflect the sampler and its burnin adaptation.
Furthermore, note that the custom AVX functions in the [BayesMVP extension](https://github.com/CerulloE1996/BayesMVP)
provide an additional route to speeding things up.


**More detailed and comprehensive benchmarks are coming soon.** The comparisons will cover:


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


### Stan vs NicoStan (diffusion HMC and standard HMC) on eight general models


We compared NicoStan with Stan (i.e., NUTS, via cmdstanr) in terms of wall-clock time on eight models,
using the same Stan model files, data and initial values.
Five of these models have a nuisance block -
a stochastic volatility model (T = 1,000; SV), a shared-frailty Cox model (N = 250; COX),
a latent diffusion survival model (N = 200; LDS), a joint longitudinal and survival model (N = 75; JLS)
and a hierarchical logistic regression (N = 1,200; HLR) -
and three do not -
a Gaussian process regression (N = 80; GP), a Weibull survival model (N = 250; WEI)
and a robust regression with t₄ errors (N = 1,000; RT4).
For the models with a nuisance block, NicoStan sampled the nuisance block using either diffusion HMC
(i.e., hybrid HMC; NicoStan's default) or standard HMC;
for the models without one, NicoStan samples all of the parameters using standard HMC.
NicoStan used its default trajectory-length criterion (ChEES-R) and the SpESS-R criterion
(see [SpESS-R](#spess-r-a-spectral-linearquadratic-ess-rate-trajectory-length-criterion) below).


Both samplers used 4 chains for the warm-up (or burnin), followed by 180 sampling chains of 100 iterations each
(45 sampling chains starting from the final state of each warm-up chain),
one fit at a time on a 96-core (192-thread) machine, with 5 seeds per configuration.
Stan ran in two stages: the warm-up with 4 chains, then the sampling with 180 chains, each using the step size,
metric and final state of its warm-up chain (adaptation off); adapt_delta = 0.8, maximum tree depth 10,
and 250, 500 or 1,000 warm-up iterations (vs. NicoStan's 125, 250 or 500 burnin iterations,
since Stan's warm-up is unreliable with only 125 iterations).
For COX, both samplers evaluated the log-likelihood in 3 chunks (reduce_sum_static),
as given by NicoStan's cache-based chunking rule (`fn_compute_initial_N_chunks()`) for 180 chains;
for the other models, the rule gave 1 chunk.


The table below shows the (projected) time needed to reach a minimum ESS of 1,000
(the measured warm-up or burnin time, plus 1,000 times the measured sampling time divided by the minimum ESS;
in seconds) / the minimum ESS per 1,000 sampling gradient evaluations
(geometric means over 5 seeds; the minimum is over the bulk ESS, the tail ESS and the ESS of the squared deviations,
i.e., of the SD, of the main parameters).
Stan's times are those of its CmdStan processes (from the first chain's start to the last chain's end;
for the sampling, this includes the staggered launch of its 180 processes),
and NicoStan's those of its burnin and sampling
(excluding, for both, the compilation and loading of the model; and, for SpESS-R, the one-off compilation of its
C++ code, which is then cached, and its loading, which took about 0.01 s per fit).
**Bold**: the best of the row for each endpoint (and any within 1% of it).


| Model | Stan warm-up / NicoStan burnin | Stan (NUTS) | NicoStan, ChEES-R, diffusion HMC | NicoStan, ChEES-R, standard HMC | NicoStan, SpESS-R, diffusion HMC | NicoStan, SpESS-R, standard HMC |
|---|---|---|---|---|---|---|
| SV | 250 / 125 | 2.08 / 7.0 | 0.52 / 11.4 | 0.55 / 7.8 | **0.47** / **19.5** | 0.51 / 12.1 |
|  | 500 / 250 | 2.90 / 6.0 | **0.72** / 11.3 | 0.76 / 8.7 | 0.79 / **15.2** | 0.86 / 13.1 |
|  | 1000 / 500 | 3.93 / 7.9 | **1.22** / 14.3 | 1.27 / 8.5 | 1.63 / **15.2** | 1.72 / 12.6 |
| COX | 250 / 125 | 10.5 / 23.7 | **1.58** / 28.8 | 1.89 / 17.9 | 1.61 / **30.4** | 1.69 / 26.8 |
|  | 500 / 250 | 14.4 / 23.3 | 2.77 / 29.9 | 2.76 / 32.1 | **2.68** / **34.9** | 2.76 / 31.8 |
|  | 1000 / 500 | 22.3 / 23.6 | **4.23** / 39.9 | 5.06 / 26.8 | 4.56 / **40.5** | 4.92 / 31.2 |
| LDS | 250 / 125 | 1.17 / 2.8 | 0.30 / 7.2 | **0.30** / **7.6** | 0.31 / 7.2 | **0.30** / 6.2 |
|  | 500 / 250 | 1.47 / 2.8 | **0.50** / **7.3** | 0.51 / 6.4 | 0.54 / 5.9 | 0.53 / 6.6 |
|  | 1000 / 500 | 1.91 / 2.8 | **0.89** / **8.7** | **0.89** / 7.3 | 0.97 / 7.1 | 0.96 / 6.6 |
| JLS | 250 / 125 | 13.1 / 6.7 | 1.07 / 8.6 | 1.11 / 9.0 | 1.13 / 12.6 | **1.04** / **13.9** |
|  | 500 / 250 | 15.3 / 6.2 | **1.65** / 10.1 | 1.67 / 8.0 | 1.73 / **15.4** | 1.81 / 14.6 |
|  | 1000 / 500 | 19.7 / 6.1 | **2.69** / 7.2 | **2.71** / 5.9 | 3.29 / **16.3** | 3.43 / 14.3 |
| HLR | 250 / 125 | 1.29 / 8.8 | **0.28** / 15.0 | 0.29 / 12.4 | **0.28** / **15.5** | 0.30 / 13.1 |
|  | 500 / 250 | 1.60 / 8.7 | **0.50** / **15.8** | **0.51** / 11.8 | 0.55 / 14.3 | 0.55 / 13.8 |
|  | 1000 / 500 | 2.45 / 9.5 | **0.92** / 15.0 | 0.93 / 11.9 | 1.02 / **16.7** | 1.04 / 13.2 |
| GP | 250 / 125 | 0.89 / 86.5 | (no nuisance block) | **0.38** / 101.9 | (no nuisance block) | 0.39 / **112.8** |
|  | 500 / 250 | 1.39 / 86.3 | (no nuisance block) | **0.70** / 116.4 | (no nuisance block) | 0.74 / **126.8** |
|  | 1000 / 500 | 2.29 / 78.9 | (no nuisance block) | **1.32** / 107.5 | (no nuisance block) | 1.42 / **129.1** |
| WEI | 250 / 125 | 0.37 / 78.9 | (no nuisance block) | 0.16 / 108.9 | (no nuisance block) | **0.15** / **127.8** |
|  | 500 / 250 | 0.48 / 79.7 | (no nuisance block) | 0.28 / 132.2 | (no nuisance block) | **0.27** / **149.1** |
|  | 1000 / 500 | 0.58 / 86.6 | (no nuisance block) | **0.52** / 116.2 | (no nuisance block) | 0.54 / **155.8** |
| RT4 | 250 / 125 | 0.46 / 114.3 | (no nuisance block) | 0.15 / 171.2 | (no nuisance block) | **0.15** / **184.2** |
|  | 500 / 250 | 0.68 / 110.3 | (no nuisance block) | **0.26** / **224.1** | (no nuisance block) | 0.26 / 207.2 |
|  | 1000 / 500 | 1.07 / 114.0 | (no nuisance block) | **0.48** / **232.7** | (no nuisance block) | 0.50 / 221.2 |


NicoStan reached the target faster than Stan for every model and warm-up/burnin length.
For the fastest configuration of each sampler (for every model, Stan's 250 warm-up iterations and NicoStan's
125 burnin iterations), NicoStan was between 2.3× (GP) and 12.6× (JLS) faster
(2.5× for WEI, 3.2× for RT4, 3.9× for LDS, 4.4× for SV, 4.6× for HLR and 6.6× for COX),
and its highest minimum ESS per gradient was between 1.5× and 3.1× that of Stan.


Within NicoStan, diffusion HMC reached the target faster than standard HMC in 24 of the 30 cells with a nuisance
block (combinations of model, criterion and burnin length), with a higher minimum ESS per gradient in 25 of them
(geometric means of 0.97× and 1.19× those of standard HMC):
it was faster for COX, SV and HLR (0.92×, 0.94× and 0.97× the time of standard HMC),
whilst the two samplers were essentially level for JLS and LDS (0.99× and 1.01×).


Compared to ChEES-R (with the same model, sampler and burnin length), SpESS-R had a higher minimum ESS per gradient
in 30 of the 39 cells (20% higher, on average) and needed fewer gradient evaluations to reach the target in 28 of them
(11% fewer, on average).
However, it chose longer trajectories during the burnin (9%, 14% and 29% more burnin gradient evaluations than
ChEES-R, for burnins of 125, 250 and 500 iterations); hence, it reached the target slightly faster than ChEES-R with
125 burnin iterations (0.98× its time, on average), but more slowly with 250 and 500 (1.05× and 1.13×).
For the fastest configuration of each model, SpESS-R was faster for SV, WEI, RT4 and JLS
(0.91×, 0.92×, 0.96× and 0.97× the time of ChEES-R), essentially joint-best for HLR and LDS (within 1%),
and slower for COX and GP (1.02× and 1.03×).


SpESS-R's own computation (its update in each burnin iteration, which runs in C++) took between 0.18 and 0.21 ms
per burnin iteration on average, vs. between 0.09 and 0.11 ms for ChEES-R (for burnins of 125 to 500 iterations);
in other words, it added 0.03 s per burnin, on average, i.e., 3.1% of the average burnin time of ChEES-R (0.96 s;
between 0.6% and 6.1%, depending on the model).


Note that:

- We counted Stan's gradient evaluations as its number of leapfrog steps,
and NicoStan's as the expected number of leapfrog steps of its (jittered) trajectories,
plus one for every iteration in which the sampler evaluates the gradient at the start of the trajectory
(i.e., during the burnin, for standard HMC, and for the models without a nuisance block).
- The nested R-hat ([Margossian et al., 2024](https://arxiv.org/abs/2110.13017v6); with the 4 warm-up chains as the
superchains) exceeded 1.01 in 51 of the 510 fits, 46 of which were for LDS (45 of NicoStan's 60 LDS fits, with a
median of 1.018 and a maximum of 1.044, and 1 of Stan's 15, with 1.012); for the other models, it was at most 1.038
for NicoStan (COX) and 1.003 for Stan.
- Stan had 77 divergent transitions (in 9 of its 120 fits: 8 for HLR and 1 for LDS),
and NicoStan had 22 (in 17 of its 390 fits: 13 for LDS and 4 for SV), out of 18,000 sampling draws per fit.


### Trajectory-length criteria within NicoStan


On the same eight models, we also compared NicoStan's trajectory-length criteria:
the SpESS-R criterion (see [SpESS-R](#spess-r-a-spectral-linearquadratic-ess-rate-trajectory-length-criterion) below),
ChEES-R (NicoStan's default), ESJD, SNAPER and the geometric mean of ESJD and SNAPER (`"ESJD_SNAPER"`).
For the models with a nuisance block, we ran every criterion with both diffusion HMC and standard HMC;
hence, with burnins of 500, 250 and 125 iterations,
there were 39 cells (i.e., combinations of model, sampler and burnin length), with 10 seeds per cell
(and, in each fit, 4 burnin chains, followed by 16 sampling chains of 250 iterations each).


For each cell, we computed each criterion's loss relative to the best criterion of the cell
(for the gradient evaluations needed to reach a minimum ESS of 1,000, the criterion's divided by the best;
for the minimum ESS per gradient, the best divided by the criterion's),
and whether it was within noise of the best
(i.e., whether the 95% interval of its matched-seed ratio to the best criterion included 1).
The tables below give the geometric mean of the losses over the 39 cells ("mean loss"), the worst loss,
and the number of cells in which the criterion was best or within noise of the best
(a descriptive count, which does not require the interval to be narrow).


**Ordered by the gradient evaluations needed to reach the target:**

| Criterion | Gradients to target: mean loss | worst loss | best or within noise | ESS per gradient: mean loss | worst loss | best or within noise |
|---|---|---|---|---|---|---|
| SpESS-R | 1.081 | 1.39 | 36/39 | 1.075 | 1.43 | 38/39 |
| ESJD | 1.091 | 1.59 | 34/39 | 1.109 | 1.75 | 35/39 |
| ESJD_SNAPER | 1.103 | 1.49 | 34/39 | 1.132 | 1.59 | 35/39 |
| SNAPER | 1.135 | 1.67 | 30/39 | 1.170 | 1.83 | 31/39 |
| ChEES-R (default) | 1.146 | 1.71 | 29/39 | 1.191 | 2.04 | 29/39 |

**Ordered by the minimum ESS per gradient:**

| Criterion | Gradients to target: mean loss | worst loss | best or within noise | ESS per gradient: mean loss | worst loss | best or within noise |
|---|---|---|---|---|---|---|
| SpESS-R | 1.081 | 1.39 | 36/39 | 1.075 | 1.43 | 38/39 |
| ESJD | 1.091 | 1.59 | 34/39 | 1.109 | 1.75 | 35/39 |
| ESJD_SNAPER | 1.103 | 1.49 | 34/39 | 1.132 | 1.59 | 35/39 |
| SNAPER | 1.135 | 1.67 | 30/39 | 1.170 | 1.83 | 31/39 |
| ChEES-R (default) | 1.146 | 1.71 | 29/39 | 1.191 | 2.04 | 29/39 |


The SpESS-R criterion had the smallest mean loss for both endpoints
(1.081 and 1.075, vs. 1.091 and 1.109 for ESJD, the second best);
however, for the gradient evaluations needed to reach the target, ESJD was within 1% of it
(i.e., the two were essentially joint-best).
It also had the smallest worst-case losses (1.39 and 1.43, vs. 1.49 and 1.59 for ESJD_SNAPER, the second best),
and it was best or within noise of the best in the most cells (36 and 38 of the 39 cells, respectively,
vs. 34 and 35 for ESJD and for ESJD_SNAPER; descriptive counts).


More specifically, when the nuisance block was sampled using diffusion HMC,
the SpESS-R criterion was best on average for both endpoints
(1.089 and 1.078, vs. 1.104 and 1.123 for ESJD), and it was best or within noise of the best in all 15 of these cells;
similarly, with standard HMC for the nuisance block, it was best on average for both endpoints
(1.109 and 1.097, vs. 1.123 and 1.144 for ESJD).


However, it was not uniformly best:
for the models without a nuisance block, ESJD, ESJD_SNAPER and the SpESS-R criterion were essentially
joint-best for both endpoints (within 1% of each other; 1.020 and 1.031 for ESJD, vs. 1.026 and 1.035 for the
SpESS-R criterion);
and it was not within noise of the best in 3 cells for the gradient evaluations needed to reach the target
(RT4 with a 250-iteration burnin, and JLS and SV with standard HMC and a 500-iteration burnin; losses of up to 1.19),
and in 1 cell (RT4 with a 250-iteration burnin) for the minimum ESS per gradient.


<!-- ------------------------------------------------------------------------------------------------------------------------------- -->
## Models with nuisance parameters (diffusion-pathspace HMC)
<!-- ------------------------------------------------------------------------------------------------------------------------------- -->


NicoStan is particularly aimed at models with a large block of latent/nuisance parameters for which analytic marginalisation is unavailable,
or numerical marginalisation is too expensive. Techniques such as non-centring can improve the posterior geometry for some of these models;
however, it does not remove the nuisance block - the corresponding latent variables still have to be handled during posterior computation.


Such blocks occur in many commonly-used models; such as:

- **MVP, LC-MVP, MVOP and LC-MVOP:** latent-Gaussian/auxiliary variables for correlated binary and/or ordinal outcomes.
The augmented state grows with the number of individuals and outcomes;
evaluating the marginal likelihood instead involves multivariate Gaussian rectangle probabilities.
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
Note that the supplied discrete-time AR(1) stochastic-volatility example is a different model from the continuous-time experiments in
[Beskos et al., 2013](https://doi.org/10.1016/j.spa.2012.12.001) and [Beskos et al., 2015](https://doi.org/10.1093/biomet/asv051).


Note that some simpler Gaussian models allow the latent block to be integrated out analytically.
For instance, our Gaussian-outcome GP example uses the marginal likelihood and is fitted with standard HMC;
the latent-GP examples listed here instead have non-Gaussian likelihoods.


### Which parameters are nuisance and which are main: the Paper 3 benchmark models


Every parameter of the `parameters` block of each benchmark model is listed below, with its notation, its plain-English
meaning (taken from the comment on its declaration in the Stan file), its size, and whether NicoStan holds it as a
nuisance block (sampled with diffusion-pathspace HMC) or in the main block. Every N-sized latent block is a nuisance
block in every timing, benchmark and fit; the main block holds only the fixed-size parameters.
The joint longitudinal-survival, population PK and hierarchical logistic regression models are written in the array /
vector forms that NicoStan's reduce_sum_chunked writer holds, with the same values and the same posterior. For instance,
the joint longitudinal-survival model's $2 \times N_{\text{subj}}$ matrix of standardised random effects is declared
as two vectors of length $N_{\text{subj}}$ (`b_raw_intercept`, `b_raw_slope`), since the writer cannot hold a matrix
nuisance block.


| Model (N) | Parameter (Stan name) | Plain English | Size | Block (why) |
|---|---|---|---|---|
| Weibull survival (250) | $\alpha$ (`alpha`) | Weibull shape parameter | 1 | main (fixed size) |
| | $\boldsymbol{\beta}$ (`beta`) | regression coefficients | $K$ | main (fixed size) |
| | $\mu$ (`mu`) | intercept | 1 | main (fixed size) |
| Robust Student-t regression (500) | $\alpha$ (`alpha`) | intercept | 1 | main (fixed size) |
| | $\boldsymbol{\beta}$ (`beta`) | regression coefficients | $K$ | main (fixed size) |
| | $\sigma$ (`sigma`) | scale parameter | 1 | main (fixed size) |
| GP regression (50) | $\ell$ (`lengthscale`) | lengthscale | 1 | main (fixed size; the GP is marginalised, so there is no latent block) |
| | $\sigma_f$ (`sigma_f`) | signal standard deviation | 1 | main (fixed size) |
| | $\sigma_n$ (`sigma_n`) | noise standard deviation | 1 | main (fixed size) |
| Stochastic volatility, non-centred (T = 500) | $\tilde{\mathbf{h}} = (\tilde{h}_1, \ldots, \tilde{h}_T)$ (`h_std`) | standard normal innovations of the log-volatility path | $T$ | nuisance (standard normal latent path, grows with $T$) |
| | $\mu$ (`mu`) | mean log volatility | 1 | main (fixed size) |
| | $\phi$ (`phi`) | persistence of volatility, in $(-1, 1)$ | 1 | main (fixed size) |
| | $\sigma$ (`sigma`) | white noise shock scale | 1 | main (fixed size) |
| Stochastic volatility, centred (T = 500) | $\mathbf{h} = (h_1, \ldots, h_T)$ (`h`) | latent log-volatility path | $T$ | nuisance (latent path, grows with $T$) |
| | $\mu$ (`mu`) | mean log volatility | 1 | main (fixed size) |
| | $\phi$ (`phi`) | persistence of volatility, in $(-1, 1)$ | 1 | main (fixed size) |
| | $\sigma$ (`sigma`) | white noise shock scale | 1 | main (fixed size) |
| Shared-frailty Cox survival (250) | $\tilde{\mathbf{f}}$ (`log_frailty_raw`) | raw (standard normal) log frailties of the clusters | $J$ | nuisance (standard normal frailties, grow with N) |
| | $\boldsymbol{\beta}$ (`beta`) | regression coefficients (log hazard ratios) | $K$ | main (fixed size) |
| | $\sigma_{\text{frailty}}$ (`sigma_frailty`) | standard deviation of the frailties | 1 | main (fixed size) |
| Latent diffusion survival (500) | $\mathbf{w} = (w_1, \ldots, w_M)$ (`w`) | standard normal innovations of the path of the log baseline hazard | $M$ | nuisance (standard normal path, grows with the grid) |
| | $x_0$ (`x_0`) | log baseline hazard at diagnosis (per year) | 1 | main (fixed size) |
| | $\mu_x$ (`mu_x`) | long-run log baseline hazard (per year) | 1 | main (fixed size) |
| | $\kappa$ (`kappa`) | rate of return to $\mu_x$ (per year) | 1 | main (fixed size) |
| | $s$ (`diffusion_scale`) | diffusion of the log baseline hazard (per square root of a year) | 1 | main (fixed size) |
| | $\boldsymbol{\beta}$ (`beta`) | log hazard ratios of the covariates | $K$ | main (fixed size) |
| Joint longitudinal-survival (50) | $\tilde{\mathbf{b}}_0$ (`b_raw_intercept`) | standard normal random intercepts of the subjects | $N_{\text{subj}}$ | nuisance (standard normal random effects, grow with N) |
| | $\tilde{\mathbf{b}}_1$ (`b_raw_slope`) | standard normal random slopes of the subjects | $N_{\text{subj}}$ | nuisance (standard normal random effects, grow with N) |
| | $\boldsymbol{\beta}_{\text{long}}$ (`beta_long`) | fixed effects of the longitudinal submodel | $K_{\text{long}}$ | main (fixed size) |
| | $\sigma_{\text{long}}$ (`sigma_long`) | residual error SD of the longitudinal submodel | 1 | main (fixed size) |
| | $\sigma_{b_0}$ (`sigma_b0`) | random intercept SD | 1 | main (fixed size) |
| | $\sigma_{b_1}$ (`sigma_b1`) | random slope SD | 1 | main (fixed size) |
| | $\rho_b$ (`rho_b`) | correlation between the random intercepts and slopes | 1 | main (fixed size) |
| | $\boldsymbol{\beta}_{\text{surv}}$ (`beta_surv`) | effects of the covariates of the survival submodel | $K_{\text{surv}}$ | main (fixed size) |
| | $\lambda$ (`lambda`) | baseline hazard rate | 1 | main (fixed size) |
| | $\alpha_{\text{assoc}}$ (`alpha_assoc`) | association between the longitudinal trajectory and the hazard | 1 | main (fixed size) |
| Hierarchical logistic regression (500) | $\tilde{\boldsymbol{\beta}}_k \in \mathbb{R}^2$, $k = 1, \ldots, K$ (`beta_raw`) | standard normal [intercept, slope] of each group | $2K$ | nuisance (standard normal group effects, grow with N) |
| | $\boldsymbol{\mu} \in \mathbb{R}^2$ (`mu`) | population means of the [intercept, slope] | 2 | main (fixed size) |
| | $\boldsymbol{\sigma} \in \mathbb{R}^2$ (`sigma`) | population standard deviations of the [intercept, slope] | 2 | main (fixed size) |
| Graded response IRT (500) | $\boldsymbol{\theta} = (\theta_1, \ldots, \theta_N)$ (`theta`) | abilities of the respondents (standard normal) | $N$ | nuisance (standard normal abilities, grow with N) |
| | $\mathbf{a}$ (`a`) | item discriminations | $J$ | main (fixed size) |
| | $\mathbf{c}_j$, $j = 1, \ldots, J$ (`c`) | item cutpoints (logit scale), ordered within each item | $J(K - 1)$ | main (fixed size) |
| Two-compartment population PK (250) | $\mathbf{z}_n \in \mathbb{R}^5$, $n = 1, \ldots, N$ (`z`) | standard normal effects of each patient (one vector per patient) | $5N$ | nuisance (standard normal patient effects, grow with N) |
| | $\mu_{\text{CL}}$ (`mu_CL`) | population mean of log CL (clearance, L/h) | 1 | main (fixed size) |
| | $\mu_{Q}$ (`mu_Q`) | population mean of log Q (intercompartmental clearance, L/h) | 1 | main (fixed size) |
| | $\mu_{V_1}$ (`mu_V1`) | population mean of log V1 (central volume, L) | 1 | main (fixed size) |
| | $\mu_{V_2}$ (`mu_V2`) | population mean of log V2 (peripheral volume, L) | 1 | main (fixed size) |
| | $\mu_{k_a}$ (`mu_ka`) | population mean of log ka (absorption rate, 1/h), above the elimination rate (no flip-flop) | 1 | main (fixed size) |
| | $\boldsymbol{\omega} \in \mathbb{R}^5$ (`omega`) | between-patient SDs of log CL, log Q, log V1, log V2 and log ka | 5 | main (fixed size) |
| | $\mathbf{L}_{\Omega}$ (`L_Omega`) | Cholesky factor of the correlation matrix of the patient effects | $5 \times 5$ (10 free) | main (fixed size) |
| | $\sigma$ (`sigma`) | residual SD of the log concentrations | 1 | main (fixed size) |
| GP classification (50) | $\boldsymbol{\eta} = (\eta_1, \ldots, \eta_N)$ (`eta`) | standard normal values of the latent GP | $N$ | nuisance (standard normal latent GP, grows with N) |
| | $\mu$ (`mu`) | constant mean of the latent log-odds | 1 | main (fixed size) |
| | $\ell$ (`lengthscale`) | lengthscale of the squared-exponential covariance | 1 | main (fixed size) |
| | $\sigma_f$ (`sigma_f`) | marginal standard deviation of the latent log-odds | 1 | main (fixed size) |
| Spatial HSGP logistic (500) | $\mathbf{z} = (z_1, \ldots, z_M)$ (`z`) | standard normal coefficients of the basis functions | $M$ | nuisance (standard normal basis coefficients, grow with the basis) |
| | $\alpha$ (`alpha`) | intercept (log-odds at the average location and covariates) | 1 | main (fixed size) |
| | $\boldsymbol{\beta} \in \mathbb{R}^3$ (`beta`) | coefficients of the covariates | 3 | main (fixed size) |
| | $\sigma_{\text{gp}}$ (`sigma_gp`) | marginal standard deviation of the spatial effect | 1 | main (fixed size) |
| | $\ell$ (`length_scale`) | length scale of the spatial effect (units of location) | 1 | main (fixed size) |
| NMA, Xu fixed cutpoints (25 studies) | $\mathbf{Z}_{\eta}$ (`beta_eta_z`) | standard normal study effects, shared by the tests (one per class and study) | $2 \times n_{\text{studies}}$ | nuisance (standard normal study effects, grow with the studies) |
| | $\mathbf{z}_{\delta, c}$, $c = 1, 2$ (`beta_delta_z_vec`) | standard normal test-specific study effects | $2 \times n_{\text{obs}}$ | nuisance (standard normal, grow with the studies) |
| | $\boldsymbol{\beta}_{\mu, t}$ (`beta_mu`) | mean locations of each test and class (with the covariate effects) | $n_{\text{tests}} \times 2 \times n_{\text{cov}}$ | main (fixed size) |
| | $\mathbf{C}^{\text{raw}}_c$, $c = 1, 2$ (`C_raw_vec`) | raw log-differences of the cutpoints, shared by the studies (fixed cutpoints) | $2 \times n_C$ | main (fixed size) |
| | $\boldsymbol{\sigma}_{\beta} \in \mathbb{R}^2$ (`beta_sigma`) | between-study SDs (Nyaga's $\sigma$) | 2 | main (fixed size) |
| | $\boldsymbol{\tau}^{\text{raw}}$ (`beta_tau_raw`) | test-specific between-study SDs | $n_{\text{tests}} \times 2$ | main (fixed size) |
| | $\rho_{\beta}$ (`beta_corr`) | between-study correlation of the two classes (possibly restricted) | 1 | main (fixed size) |
| Four-class LC-MVOP (250) | $\tilde{\mathbf{u}}_i \in \mathbb{R}^{n_{\text{tests}}}$ (`u_raw`) | GHK nuisance variables of each row (one row vector per row) | $n_{\text{rows}} \times n_{\text{tests}}$ | nuisance (GHK latent variables, grow with N) |
| | $\boldsymbol{\omega}$ (`Omega_unconstrained_vec`) | free entries of the correlation matrices (class, then row, then column) | $n_{\text{corrs}}$ | main (fixed size) |
| | $\boldsymbol{\beta}^{\text{own}}_{\text{absent}}$ (`beta_own_absent_free`) | mean of each test when its own condition is absent | $n_{\text{tests}}$ | main (fixed size) |
| | $\boldsymbol{\beta}^{\text{own}}_{\text{present}}$ (`beta_own_present_free`) | mean of each test when its own condition is present (above the absent one) | $n_{\text{tests}}$ | main (fixed size) |
| | $\boldsymbol{\Delta}^{\text{cross}}$ (`delta_cross_free`) | effects of the other condition on each test, by own-condition status | $2 \times n_{\text{tests}}$ | main (fixed size) |
| | $\mathbf{B}^{\text{cov}}$ (`beta_cov_raw`) | covariate effects on each test (by own-condition status if allowed) | $(1 \text{ or } 2) \times n_{\text{cov}} \times n_{\text{tests}}$ | main (fixed size) |
| | $\boldsymbol{\gamma}_A$, $\boldsymbol{\gamma}_B$ (`beta_prevalence_A`, `beta_prevalence_B`) | prevalence regression coefficients (zero length without prevalence covariates) | $n_{\text{prev cov}}$ each | main (fixed size) |
| | $\boldsymbol{\omega}_{\text{global}}$ (`Omega_global_unconstrained_vec`) | free entries of the global correlation matrix (correlation pooling) | $n_{\text{corrs, global}}$ | main (fixed size) |
| | $\lambda_k$ (`lambda_class`) | pooling weight of each class (only when the correlations are pooled) | $n_{\text{class}}$ or 0 | main (fixed size) |
| | $\boldsymbol{\pi}_p$ (`prev_simplex`) | class probabilities of each population (simplex form) | $n_{\text{pops}} \times n_{\text{class}}$ or 0 | main (fixed size) |
| | $z_{A, p}$, $z_{B, p}$ (`z_logit_A`, `z_logit_B`) | non-centred stratum deviations of the logit prevalences (odds-ratio form) | $n_{\text{pops}}$ each or 0 | main (fixed size) |
| | $\mu_A$, $\mu_B$ (`mu_logit_A`, `mu_logit_B`) | population-typical logit prevalences (odds-ratio form) | 1 each or 0 | main (fixed size) |
| | $\sigma_A$, $\sigma_B$ (`sigma_logit_A`, `sigma_logit_B`) | between-stratum spread of the logit prevalences (odds-ratio form) | 1 each or 0 | main (fixed size) |
| | $\log \text{OR}$ (`log_OR_par`) | log odds ratio between the two conditions (shared or one per population) | 1, $n_{\text{pops}}$ or 0 | main (fixed size) |
| | $\mathbf{C}^{\text{raw}}$ (`C_raw_vec`) | own condition absent: first cutpoint and log-increments of each test | $n_C$ | main (fixed size) |
| | $\mathbf{d}^{\text{raw}}$ (`d_raw_present`) | own condition present: raw shift-down of each threshold | $n_C$ | main (fixed size) |
| One-class MVOP (1C-MVOP) | $\tilde{\mathbf{u}}_i \in \mathbb{R}^{n_{\text{tests}}}$, $i = 1, \ldots, N$ (`u_raw`) | GHK nuisance variables of each individual | $N \times n_{\text{tests}}$ | nuisance (GHK latent variables, grow with N) |
| | $\boldsymbol{\omega}$ (`Omega_unconstrained_vec`) | free entries of the correlation matrix | $n_{\text{corrs}}$ | main (fixed size) |
| | $\boldsymbol{\beta}$ (`beta_vec`) | coefficients of every test (intercepts and covariates) | $n_{\text{cov, total}}$ | main (fixed size) |
| | $\mathbf{p}^{\text{raw}}$ (`p_raw`) | prevalence (empty with one class) | 0 | none |
| | $\mathbf{C}^{\text{unc}}$ (`C_unc_vec`) | unconstrained cutpoints of the ordinal tests | $n_C$ | main (fixed size) |
| Two-class LC-MVOP (2LC-MVOP) | $\tilde{\mathbf{u}}_i \in \mathbb{R}^{n_{\text{tests}}}$, $i = 1, \ldots, N$ (`u_raw`) | GHK nuisance variables of each individual | $N \times n_{\text{tests}}$ | nuisance (GHK latent variables, grow with N) |
| | $\boldsymbol{\omega}_c$, $c = 1, 2$ (`Omega_unconstrained_vec`) | free entries of the correlation matrix of each class | $2 \times n_{\text{corrs}}$ | main (fixed size) |
| | $\boldsymbol{\beta}$ (`beta_vec`) | coefficients of every test and class (intercepts and covariates) | $n_{\text{cov, total}}$ | main (fixed size) |
| | $\mathbf{p}^{\text{raw}}$ (`p_raw`) | raw prevalence of each population | $n_{\text{pops}}$ | main (fixed size) |
| | $\mathbf{C}^{\text{unc}}_c$, $c = 1, 2$ (`C_unc_vec`) | unconstrained cutpoints of the ordinal tests, per class | $2 \times n_C$ | main (fixed size) |


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
These dependencies limit parallelism within an individual/path, although independent individuals,
paths and chains can still be processed in parallel.


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
## reduce_sum_chunked: nuisance blocks held inside the chunks
<!-- ------------------------------------------------------------------------------------------------------------------------------- -->


Stan's `reduce_sum` splits a sum (e.g., the log likelihood) into chunks, each evaluated on its own autodiff tape;
this is the basis of tape chunking (i.e., chunks small enough that each one's tape fits in the CPU's caches;
see Paper 1).
However, with `reduce_sum` alone, some of the work still happens outside the chunks, on the main tape:
for a nuisance block with one or more values per unit (e.g., per individual, group or time point),
the main tape still holds one node per value, `reduce_sum` copies each chunk's slice of the block,
and its bookkeeping (the list of the parameters' addresses, the zeroing of their gradients, and adding each chunk's
gradients back into them) runs over every value of the block, on the main thread.
These passes over the whole block grow with N, and at large N they no longer fit in the caches.


`reduce_sum_chunked` moves this work inside the chunks:

- **Chunk-local bookkeeping:** `reduce_sum`'s bookkeeping (the addresses, zeroing and adding back) is done inside
each chunk, for that chunk's slice only.
- **Nuisance blocks held by NicoStan:** the nuisance blocks are taken out of the Stan parameters, and held by
NicoStan instead; each chunk reads only its own slice of the block, which lives on the chunk's own tape only,
so that the main tape holds no node per nuisance value.


Note that `map_rect` is not a substitute for this:
`map_rect` returns a vector from each job (rather than a sum), and keeps every job's shared inputs and their
Jacobian on the main tape; in other words, a shared N-sized parameter is copied into every job.


### What it gives


For the stochastic volatility model (non-centred, with one latent volatility per time point as its nuisance block),
`reduce_sum_chunked` is faster than plain `reduce_sum` at every N, and its gain grows with N
(sampling throughput, i.e., iterations per second over all chains, each at its own fastest layout of chains,
threads per chain and chunks; see Paper 1):

| N | Gain over plain `reduce_sum`: local-HPC (AVX-512) | Gain over plain `reduce_sum`: laptop (AVX2) |
|---:|---:|---:|
| 500 | 1.15x | 1.19x |
| 5000 | 1.26x | 1.27x |
| 10,000 | 1.28x | 1.47x |
| 25,000 | 1.74x | 1.95x |
| 50,000 | 2.12x | 2.74x |

With `reduce_sum_chunked`, the AVX-512 version of this model is 3.06x, 3.61x, 4.14x, 4.87x and 5.23x faster than
its Stan math version (local-HPC, N = 500, 5000, 10,000, 25,000 and 50,000), i.e., rising with N.


### What it changes inside Stan's autodiff system


`reduce_sum_chunked` directly modifies Stan's internal automatic differentiation (autodiff) system, i.e., the part
of the Stan math library that computes the gradient of the log density, rather than only the Stan model.
Stan uses reverse-mode autodiff: each operation in the model is recorded as a node (a `vari`) on a tape (i.e., a list
of every operation, in the order in which it was evaluated, kept in Stan's own block of memory, the "arena"), and
the gradient is then computed by one sweep backwards over the tape, in which each node passes its adjoint (i.e., the
derivative of the log density with respect to that node's value) back to the nodes it was computed from.


More specifically, NicoStan changes two parts of this system:

- **Stan math's `reduce_sum`** (the C++ header file `stan/math/rev/functor/reduce_sum.hpp`): the bookkeeping of
the sliced argument (the list of its nodes' addresses, the zeroing of its partial derivatives, and adding each
chunk's partial derivatives back into its nodes) is done by each chunk, for its own slice only; furthermore, for a
nuisance block held by NicoStan, each chunk makes the nodes of its own nuisance values on its own tape (from
NicoStan's array of values), and adds their adjoints straight into NicoStan's array of the gradient, so that the
main tape holds no node per nuisance value.
- **BridgeStan's gradient call** (`model_rng.cpp` and `bridgestan.cpp`), so that the gradient of the nuisance block
is returned from NicoStan's array, without extra copies.


The gradient is the same as that of the original model, up to the order of a few floating-point additions; we
checked this for the stochastic volatility, hierarchical logistic regression, joint longitudinal and survival,
population pharmacokinetic and IRT models (96 cases per model; largest relative difference 4.9e-16;
see Paper 3).


### Why it does not change your own Stan, BridgeStan or cmdstanr installation


None of these changes are made to your own installation of Stan.
Stan's C++ code is "header-only" for the most part, i.e., it is not a library installed once and shared by every
model, but source code that is compiled into each model's own library (the `.so` file) when that model is built;
hence a change to Stan's code only affects the models that are compiled with the changed copy of it.
NicoStan therefore makes its own private copy of your BridgeStan installation (which contains its own copy of
the Stan math library), in R's cache folder for NicoStan (`tools::R_user_dir("NicoStan", "cache")`), makes its
changes to that copy only, and points BridgeStan at the copy only whilst it compiles a NicoStan model, then points it
back at your own installation straight away.
In other words:

- your own BridgeStan folder is only read (to be copied), never written;
- CmdStan (which cmdstanr and CmdStanPy use, and which has its own separate copy of the Stan math library) and rstan
are never read or written by NicoStan at all;
- any model you compile with cmdstanr, rstan or BridgeStan itself is compiled against its own, original copy of
Stan, exactly as before.


Note that each change is only made when the lines of the copied file are exactly those NicoStan expects; otherwise
NicoStan stops with an error, rather than changing a file which differs from the one it expects (e.g., one from a
newer version of Stan).
Each private copy is named after the path of the original BridgeStan installation and the content of NicoStan's
changes, so that a new version of either makes a new copy (rather than reusing an old one); deleting the cache
folder removes every copy, and NicoStan simply makes a new one the next time it is needed.
These changes are currently made on Linux only.


### How to write a model for it


1. Declare each nuisance block in the `parameters` block, as one of: `vector[E]`, `row_vector[E]`,
`array[E] real`, `array[E] vector[K]` or `array[E] row_vector[K]` (unconstrained), with `E` its number of units.
2. Make the nuisance block the **sliced** argument of `reduce_sum` (or `reduce_sum_static`),
and put everything whose size grows with N (e.g., the nuisance block's prior, the likelihood terms) inside the
partial-sum function; the shared arguments should only be the main parameters and data.
Note that `reduce_sum` slices arrays only, so a `vector[E]` block should be declared as `array[E] real` here.
3. Give the names of the nuisance blocks with `nuisance_params` (see
[Example 2](#example-2-several-nuisance-blocks-nuisance_params)).


For instance, for the hierarchical logistic regression (with a random intercept and a random slope for each
group, in non-centred form), the nuisance block `beta_raw` is sliced by groups,
and each chunk evaluates the prior of its groups' `beta_raw` and the likelihood of their observations
(the data are sorted by group in `transformed data`, so that each chunk's observations are contiguous):

```stan
functions {
  real partial_sum_groups_lpdf(array[] row_vector beta_raw_slice, int start, int end,
                               vector mu, vector sigma,
                               data array[] int y_sorted, data vector x_sorted,
                               data array[] int group_sorted,
                               data array[] int first_of_group, data array[] int last_of_group) {
    int first = first_of_group[start];
    int last = last_of_group[end];
    matrix[end - start + 1, 2] beta_raw_of_slice = to_matrix(beta_raw_slice);
    matrix[end - start + 1, 2] beta_of_slice;
    for (i in 1:2) {
      beta_of_slice[, i] = mu[i] + sigma[i] * beta_raw_of_slice[, i];
    }
    array[last - first + 1] int position_in_slice
      = to_int(to_array_1d(to_vector(group_sorted[first:last]) - (start - 1)));
    return std_normal_lupdf(to_vector(beta_raw_of_slice))
           + bernoulli_logit_lupmf(y_sorted[first:last] | beta_of_slice[position_in_slice, 1]
                                                          + beta_of_slice[position_in_slice, 2]
                                                            .* x_sorted[first:last]);
  }
}
parameters {
  array[K] row_vector[2] beta_raw;   // the nuisance block (one row per group)
  vector[2] mu;
  vector<lower=0>[2] sigma;
}
model {
  mu ~ normal(0, 2);
  sigma ~ normal(0, 2);
  target += reduce_sum_static(partial_sum_groups_lupdf, beta_raw, groups_per_chunk, mu, sigma,
                              y_sorted, x_sorted, group_sorted, first_of_group, last_of_group);
}
```

(the full model, with its `data` and `transformed data` blocks, is in Paper 3's benchmark models), and it is
fitted with `nuisance_params = c("beta_raw")`.
NicoStan then writes the `reduce_sum_chunked` version of the model next to it
(`<model>_reduce_sum_chunked.stan`): the block's declaration is kept as a comment, the `reduce_sum` call is given the
block's term indices as its sliced argument, and a copy of the partial-sum function reads the chunk's own values
at its start, under the name of its sliced argument, so that the rest of the function is unchanged.
Any other use of a nuisance block (outside `reduce_sum`) reads its values where it is used; these values are
correct, but not chunked.


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


NicoStan samples the main parameters `θ` and the latent/nuisance parameters `u` **jointly**,
within the same hybrid HMC/diffusion-pathspace trajectory:

- The main model parameters are updated using standard HMC dynamics.
- For models with nuisance parameters,
the nuisance parameters are updated using **diffusion-pathspace HMC** ([Beskos et al., 2013](https://doi.org/10.1016/j.spa.2012.12.001)),
where the Gaussian part of the dynamics is solved exactly (it is a rotation),
so this Gaussian substep adds no integration error, regardless of the size of the nuisance block.
The complete trajectory also contains numerical steps for the remaining terms; 
the Metropolis-Hastings acceptance step corrects their integration error.


Models without a nuisance block (e.g., standard univariate logistic regression) use standard HMC throughout.
In our initial tests, we have also found NicoStan to be more efficient than Stan (via cmdstanr) for some of these models;
we expect the different burnin/adaptation schemes to contribute to this, although the detailed comparisons are still in progress
(see [Benchmarks](#benchmarks) and [Efficient burnin algorithms](#efficient-burnin-algorithms-snaper-hmc-and-chees-r-hmc)).


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


<!-- Furthermore, the trajectory-length adaptation used during burnin 
(e.g., ChEES-R or SNAPER; see the [Efficient burnin algorithms](#efficient-burnin-algorithms-snaper-hmc-and-chees-r-hmc) section below) 
is based on the main parameters only; hence, a large nuisance block does not dominate the adaptation.  -->


<!-- ------------------------------------------------------------------------------------------------------------------------------- -->
## Efficient burnin algorithms (SNAPER-HMC and ChEES-R-HMC)
<!-- ------------------------------------------------------------------------------------------------------------------------------- -->


NicoStan's burnin (i.e., warm-up) runs several chains in parallel,
and adapts the HMC tuning parameters using information pooled **across** all of the burnin chains (i.e., between-chain adaptation);
whereas Stan normally adapts each NUTS-HMC chain separately.


This between-chain adaptation is based on ChEES-HMC
([Hoffman et al., 2021](https://proceedings.mlr.press/v130/hoffman21a.html)),
and SNAPER-HMC ([Sountsov and Hoffman, 2022](https://arxiv.org/abs/2110.11576v3)).


More specifically, during burnin, NicoStan adapts:

- The step size, targeting a mean acceptance probability of `adapt_delta` (0.80 by default),
using an ADAM-type update ([Kingma and Ba, 2015](https://arxiv.org/abs/1412.6980));
the ADAM moments are reset, and the learning-rate schedule restarted, at the final metric update,
so that the step size settles on the final metric rather than on the acceptance crashes of the earlier metric windows.
- The mass matrix (i.e., metric) for the main parameters, 
using either empirical covariance/variance estimates from the burnin chains (`metric_type_main = "Empirical"`)
or a numerical Hessian (`metric_type_main = "Hessian"`); 
more specifically, the Hessian is computed by finite differences of the main-parameter gradients.
Both methods support a diagonal or dense Euclidean metric (`metric_shape_main = "diag"` or `"dense"`).
With the (default) empirical metric pooled across the burnin chains, NicoStan updates the metric
once a metric window holds 8 draws, and rescales the step size and trajectory length at each metric update
(i.e., multiplies them by $1/\sqrt{c}$, where $c$ is the geometric mean of the change in the diagonal of the inverse metric);
more specifically, a few draws are enough to replace an initial (unit) metric which is far off (e.g., for our ordinal LC-MVOP models),
whilst the rescaling means that these early updates do not disturb models whose initial metric is already close (e.g., our binary LC-MVP models).
These can be changed via the R options `NicoStan_pooled_metric_warm_up`
(`"hard"`, a number of draws, `"soft"`, `"first_window"`, `"n_over_n_plus_k"` or `"signal_to_noise"`)
and `NicoStan_metric_update_rescales_eps_and_tau` (`TRUE` by default).
The nuisance masses and centre are adapted separately (see [How NicoStan works](#how-nicostan-works)).
- The trajectory length, using the criterion selected via `burnin_algorithm`
(see [Trajectory-length adaptation algorithms](#hmc-trajectory-length-tau-adaptation-algorithms) below).


The main/nuisance separation means that NicoStan can use a dense empirical or Hessian metric for the main block,
whilst retaining a diagonal for a much larger nuisance block (where computing a dense metric would be prohibitively expensive).
Stan's [standard HMC/NUTS metric interface](https://mc-stan.org/docs/cmdstan-guide/mcmc_config.html#metric)
selects a unit, diagonal or dense metric for the complete unconstrained parameter vector;
its dense option therefore includes the nuisance parameters too, rather than exposing separate main/nuisance metric choices.
This distinction allows NicoStan to use a dense main-parameter metric without constructing/storing a dense metric 
for the entire latent/nuisance block.


### HMC Trajectory length ($\tau$) adaptation algorithms


Use `burnin_algorithm` to choose the trajectory-length adaptation algorithm:


- `CHEESR` (**ChEES-R**): The original ChEES-rate criterion,
using the squared change in the monitored parameters' centred squared radius per realised trajectory length.
Proposed by [Sountsov and Hoffman, 2022](https://arxiv.org/abs/2110.11576v3).
- `SNAPER` (**SNAPER**): Learns a difficult main-parameter direction,
and adapts the trajectory length using squared changes along that direction per unit length.
Proposed by [Sountsov and Hoffman, 2022](https://arxiv.org/abs/2110.11576v3).
- `ChEES` (**ChEES**): Uses the squared change in the monitored parameters' centred squared radius,
without dividing by trajectory length.
Proposed by [Hoffman et al., 2021](https://proceedings.mlr.press/v130/hoffman21a.html).
- `ESJD` (**ESJD rate**): Uses the squared jumped distance of the monitored parameters
(in the coordinates defined by the current mass matrix) per unit trajectory length,
i.e., each trajectory's squared jump divided by its own length.
Based on the expected squared jumped distance of Pasarica and Gelman, 2010.
- `LQ_ESSR` (**LQ_ESSR**): Experimental; the soft minimum, over the monitored parameters,
of the lag-one ESS bounds of the linear and quadratic statistics, per unit trajectory length
(see [LQ_ESSR](#lq_essr-a-linearquadratic-ess-rate-trajectory-length-criterion) below).
- `SpESSR` (**SpESS-R**, the spectral linear/quadratic ESS-rate criterion; also available as
`LQ_ESSR_spec_bins99_evid_expand`): Experimental; LQ_ESSR with the ESS of the linear
statistics estimated from the spectrum of each parameter's motion
(i.e., from how quickly it decorrelates as a function of the trajectory length), per gradient evaluation
(see [SpESS-R](#spess-r-a-spectral-linearquadratic-ess-rate-trajectory-length-criterion) below).


ChEES measures squared changes in the centred squared radius of the parameter vector
([Hoffman et al., 2021](https://proceedings.mlr.press/v130/hoffman21a.html)).
ChEES-R (the ChEES-Rate criterion of [Sountsov and Hoffman, 2022](https://arxiv.org/abs/2110.11576v3))
divides this change by the trajectory length, so that trajectory length is used as a proxy for computational cost.
SNAPER-HMC ([Sountsov and Hoffman, 2022](https://arxiv.org/abs/2110.11576v3))
focuses the corresponding criterion along the first principal component
(i.e., the direction of largest variance in the coordinates used for adaptation).


The position-based criteria use the main parameters in coordinates defined by the current mass matrix
(similarly to [Riou-Durand et al., 2023](https://proceedings.mlr.press/v206/riou-durand23a.html)).
<!-- If `BᵀB = M`, these coordinates are `z = B(θ - μ)`.  -->
<!-- Both diagonal and dense main-parameter metrics are supported, -->
<!-- and SNAPER-HMC learns its direction in the same coordinates. -->


By default, the trajectory-length criterion monitors every main parameter;
however, for the LC-MVP and LC-MVOP models of the [BayesMVP extension](#bayesmvp-multivariate-probit-models)
(`Model_type = "LC_MVP"` and `"LC_MVOP"`),
it instead monitors the parameters which test sensitivity, specificity and disease prevalence depend on -
more specifically, `beta` and `p_raw` for the LC-MVP, and `beta`, `p_raw` and `C_unc_vec` for the LC-MVOP.
Note that `beta` contains the class-specific coefficients of the latent test variables
(i.e., their means, when there are no covariates), which give each test's sensitivity and specificity,
and `p_raw` is the disease prevalence (on the unconstrained scale);
for the ordinal tests of the LC-MVOP, the sensitivity and specificity at each threshold
also depend on the cutpoints (`C_unc_vec`, on the unconstrained scale).
For any model and any `burnin_algorithm`, `interest_only` (e.g., `interest_only = c("beta", "p_raw")`)
restricts the criterion to the named main-parameter families,
whilst `tau_adaptation_block = "main"` (without `interest_only`) monitors every main parameter.


We checked NicoStan's SNAPER and ChEES-R implementations against the authors' own JAX implementation
(the SNAPER-HMC notebook accompanying [Sountsov and Hoffman, 2022](https://arxiv.org/abs/2110.11576v3))
on the German Credit logistic regression of the Inference Gym;
more specifically, we compared the minimum (over parameters) ESS of the centred squared draws per gradient evaluation,
i.e., the quantity their paper reports.
NicoStan's SNAPER agrees with their implementation to within 5%,
and their ChEES-R value is reproduced once two differences are accounted for:
(i) their implementation divides the ChEES-R and SNAPER criteria by the trajectory length plus one step size
(i.e., it charges the extra gradient evaluation of each iteration), which NicoStan offers via the R option
`NicoStan_rate_criterion_cost_offset_steps`; and
(ii) their sampling phase draws the jittered trajectory lengths from a Halton sequence rather than independently,
which is worth around 15% ESS per gradient on this model (NicoStan offers this via `tau_jitter_sampling_type = "halton"`; see below).
Note that on our latent-class benchmarks the first of these makes the adapted trajectories longer
and the ESS per gradient lower; hence, NicoStan divides by the trajectory length alone by default.


The default trajectory-length settings differ between burnin and sampling:


- **Burn-in:** randomised trajectory length by default (`randomize_tau_burnin = TRUE`),
drawn as in the sampling phase; setting `randomize_tau_burnin = FALSE` instead fixes the trajectory length
for each burnin iteration at the current tuning setting, which saves burnin time compared to using a randomized $\tau$.
- **Post-burnin (sampling phase):** randomised trajectory length (`randomize_tau_sampling = TRUE`) -
drawn uniformly between zero and twice the adapted scale (i.e., $\tau \sim \text{uniform}(0, 2 \bar\tau)$) -
with at least one integration step ($L \ge 1$).
By default, these lengths are drawn independently (`tau_jitter_sampling_type = "uniform"`);
alternatively, `tau_jitter_sampling_type = "halton"` takes them from a Halton (base-2 van der Corput) sequence over the same range,
as in the SNAPER-HMC authors' implementation.
In our testing, this gave about 10% more ESS of the posterior SD per gradient on the German Credit regression,
and no measurable change on our binary LC-MVP models.


Note that `manual_tau` separately controls whether the trajectory length is adapted at all -
you can instead set $\tau$ manually, by specifying `manual_tau = TRUE`, and it's fixed value by specifying
`tau_if_manual = 3.0` (if one wishes to fix $\tau = 3$).


### LQ_ESSR: a linear/quadratic ESS-rate trajectory-length criterion


NicoStan also offers an experimental criterion, `burnin_algorithm = "LQ_ESSR"` (the linear/quadratic ESS rate),
which targets the **worst**-mixing monitored parameters - via a soft minimum -
rather than a single summary of all of them
(as ESJD, ChEES-R and SNAPER do):

- For each monitored parameter (in the coordinates defined by the current mass matrix),
LQ_ESSR estimates the lag-one autocorrelation, ρ, of both the parameter (the linear statistic)
and its squared deviation (the quadratic statistic), using the acceptance-weighted jumps of the burnin chains.
- Each ρ is then converted into an upper bound on the ESS per draw, $(1 - \rho)/(1 + \rho)$
(see [Sountsov and Hoffman, 2022](https://arxiv.org/abs/2110.11576v3)
and [Riou-Durand et al., 2023](https://proceedings.mlr.press/v206/riou-durand23a.html));
the bound is exact along the normal modes of a Gaussian target with exact dynamics, and an upper bound otherwise.
- The criterion is the soft minimum of all of these bounds (i.e., of two bounds per parameter),
divided by the mean trajectory length; hence, good mixing of the posterior means cannot compensate
for poor mixing of the squared deviations, and the best-mixing parameters cannot hide the worst ones.
- The monitored parameters are chosen in the same way as for the other criteria
(see [Trajectory-length adaptation algorithms](#hmc-trajectory-length-tau-adaptation-algorithms) above).
<!-- - On our binary LC-MVP model (N = 10,000; 3 seeds per configuration), with a 125-iteration burnin,
LQ_ESSR needed fewer gradients than ESJD to reach a minimum ESS of 1,000
(0.87 [95% CI: 0.82, 0.92] times, averaged over the grid) and had a higher minimum ESS per 1,000 gradients
(1.17 [1.10, 1.25] times); however, the single best configuration was an ESJD one
(72.2 vs. 81.4 thousand gradients, and 15.68 vs. 14.03 minimum ESS per 1,000 gradients, for the best
ESJD and LQ_ESSR configurations; within seed noise).
With a 500-iteration burnin, LQ_ESSR was level with ESJD on average
(1.00 [0.93, 1.07] times the gradients; 0.99 [0.92, 1.07] times the minimum ESS per 1,000 gradients),
whilst its best configuration was the best of all on both endpoints
(85.3 thousand gradients; 16.34 minimum ESS per 1,000 gradients),
although the best configuration of every other criterion was within seed noise of it. -->


The table below compares LQ_ESSR with the other trajectory-length criteria in NicoStan.
Note that z denotes the parameters in the coordinates defined by the current mass matrix, z₀ and z′ the start
and end of a trajectory, m the running estimate of the posterior mean and t the (jittered) length of a trajectory;
by default, every criterion weights each chain's proposal by its acceptance probability.


| `burnin_algorithm` | What it maximises | Per unit trajectory length? | Statistic used | Means (bulk) vs. second moments (tail)¹ | τ gradient estimator | Cost normalisation |
| --- | --- | --- | --- | --- | --- | --- |
| `"ESJD"` | Expected squared jumped distance rate | Yes (each trajectory's jump ÷ its own t) | Jump distance, ‖z′ − z₀‖², over all monitored coordinates | Means (linear), averaged over all monitored coordinates | End-point (default) or both ends (`tau_gradient_estimator = "two_ended"`) | ÷ t (optional step offset² and `tau_cost_exponent`) |
| `"ChEES"` | Expected squared change in the centred squared radius, ½‖z − m‖² | No (per iteration) | Squared radius of all monitored coordinates | Second moments (quadratic), weighted towards the largest-variance directions | End-point (default) or both ends | None |
| `"CHESSR"` (ChEES-R) | ChEES per unit trajectory length | Yes (÷ its own t) | Squared radius of all monitored coordinates | Second moments (quadratic), weighted towards the largest-variance directions | End-point (default) or both ends | ÷ t (optional step offset² and `tau_cost_exponent`) |
| `"SNAPER"` | Expected squared change in (wᵀ(z − m))² per unit trajectory length | Yes (÷ its own t) | Squared projection onto a learned leading principal direction, w | Second moments (quadratic), along one direction | End-point (default) or both ends | ÷ t (optional step offset² and `tau_cost_exponent`) |
| `"LQ_ESSR"` | Soft minimum of the lag-one ESS bounds, (1 − ρ)/(1 + ρ), per unit trajectory length | Yes (÷ the mean t over the chains) | Per-coordinate jumps of z and of its squared deviation, converted into ESS bounds | Both (linear and quadratic), for the worst coordinate | End-point only | ÷ mean t (optional step offset² and `tau_cost_exponent`) |
| `"SpESSR"` (SpESS-R) | Soft minimum of the per-coordinate ESS fractions, per gradient evaluation | Per gradient evaluation (÷ the expected number of leapfrog steps) | Per-coordinate squared jumps of z, binned by trajectory length and fitted by a mixture of cosines (linear); jumps of the squared deviation, as for LQ_ESSR (quadratic) | Both (linear and quadratic), for the worst coordinate | None (a grid of candidate lengths; τ moves part of the way towards the best) | ÷ expected leapfrog steps |


¹ "Second moments" refers to the squared (quadratic) statistics, which are closer to the tail ESS than the means;
however, they are not the same as the quantile-based tail ESS.

² Via the R option `NicoStan_rate_criterion_cost_offset_steps` (0, i.e., no offset, by default).


### SpESS-R: a spectral linear/quadratic ESS-rate trajectory-length criterion


NicoStan also offers a spectral version of LQ_ESSR, SpESS-R (`burnin_algorithm = "SpESSR"`;
also available as `"LQ_ESSR_spec_bins99_evid_expand"`), which keeps the soft minimum over the monitored parameters
(and over the linear and quadratic statistics),
but estimates the ESS of the linear statistics from the **spectrum** of each parameter's motion -
i.e., from how quickly it decorrelates as a function of the trajectory length -
rather than from its lag-one autocorrelation alone:

- The proposals of the burnin chains are binned by their (jittered) trajectory length, t
(in bins a quarter of a doubling wide), and each monitored parameter's mean squared jump is tracked per bin,
with older updates down-weighted (by a factor of 0.99 per update).
- For each parameter, a mixture of cosines, m(t) = Σₖ wₖ (1 − cos ωₖt) (with Σₖ wₖ = 1),
is fitted to these binned jumps by non-negative least squares;
in other words, each parameter's motion is split into "modes", with frequencies ωₖ and weights wₖ.
- The autocorrelation of each mode at a candidate trajectory length, λₖ,
follows from the (running) mean acceptance probability and the jitter,
and the ESS per draw of the parameter is then 1 / Σₖ wₖ (1 + λₖ)/(1 − λₖ).
For a Gaussian target with exact dynamics, this is exact for each normal mode;
furthermore, it corrects the optimism of the lag-one bound for parameters which combine fast and slow modes.
- The quadratic statistics (squared deviations) keep the lag-one ESS bound of LQ_ESSR.
- The criterion is the soft minimum of all of these ESS fractions, divided by the expected number of leapfrog steps
(i.e., per gradient evaluation), over a grid of candidate trajectory lengths;
τ then moves part of the way towards the best candidate at each burnin iteration,
and moves beyond the longest trajectory lengths tried so far only if a (parametric) bootstrap
shows evidence that the criterion is still rising.


SpESS-R's update runs entirely in C++ (compiled once, when it is first used, and then cached);
in our 180-chain benchmark (see
[Stan vs NicoStan](#stan-vs-nicostan-diffusion-hmc-and-standard-hmc-on-eight-general-models)),
it added 3.1% to the average burnin time of ChEES-R (0.03 s per burnin),
and it reached the target slightly faster than ChEES-R with 125 burnin iterations (0.98×),
but more slowly with 250 and 500 (1.05× and 1.13×), because it chose longer trajectories during the burnin.


In our benchmark on eight models
(see [Trajectory-length criteria within NicoStan](#trajectory-length-criteria-within-nicostan)),
the SpESS-R criterion had the smallest mean loss for both endpoints
(with ESJD within 1% of it for the number of gradient evaluations needed to reach a minimum ESS of 1,000)
and the smallest worst-case losses;
however, it was not uniformly best
(e.g., for the models without a nuisance block, ESJD, ESJD_SNAPER and the SpESS-R criterion were
essentially joint-best).


<!-- ------------------------------------------------------------------------------------------------------------------------------- -->
## Custom AVX2 and AVX-512 functions
<!-- ------------------------------------------------------------------------------------------------------------------------------- -->


NicoStan can also use our custom C++, vectorised (i.e., SIMD) mathematical functions,
which are supplied through the [BayesMVP](https://github.com/CerulloE1996/BayesMVP) extension R package.
These include, for instance, exponentials (`exp()`), logarithms (`log()`),
and normal distribution functions (`Phi()`, `inv_Phi()`, `Phi_approx()` and `inv_Phi_approx()`),
as well as their derivatives and some other maths functions.


- **AVX2:** four double-precision values per vector operation; supported by many x86 CPUs.
- **AVX-512:** eight double-precision values per vector operation on supported processors.
- **General Stan models:** declare and call the custom AVX functions in the `.stan` model, 
then supply their C++ header via `Stan_cpp_user_header`.
- **Specialised [BayesMVP](https://github.com/CerulloE1996/BayesMVP) models:** 
the same AVX functions are used within the native implementations.


Note that the AVX options are not a drop-in switch for an existing Stan model;
more specifically, the user has to re-write their Stan model (i.e., the `.stan` file), 
so that it calls the custom AVX2/AVX-512 functions from BayesMVP.


The comparison examples (whose Stan models have already been re-written in this way) 
support separate `math_backend = "Stan"`, `"AVX2"` and `"AVX512"` options.
The compiled AVX model reports its vector size (i.e., four for AVX2 and eight for AVX-512), 
so you can check which implementation is being used.
Note that you should compile the model on the same machine you will use for sampling,
since the available instruction sets depend on the CPU;
for instance, 
[Intel's processor guidance](https://www.intel.com/content/www/us/en/support/articles/000090473/processors/intel-core-processors.html)
describes how to check which extensions your CPU supports.


### NicoSIMD: the SIMD functions on every device


SIMD (single instruction, multiple data) instructions apply one operation 
(for instance, an addition, a multiplication, or one step of the polynomial approximation of `exp()`) 
to several numbers at once, all held in one wide register of the CPU. 
Which SIMD instructions are available depends on the CPU's instruction set;
more specifically:

- **AVX2** (Advanced Vector Extensions 2; 256-bit registers, i.e., four double-precision values per instruction) 
is an x86 (Intel and AMD) instruction set. 
It is supported by every Intel CPU since the 4th-generation Core CPUs ("Haswell", 2013), 
and by every AMD CPU since 2015, including every AMD Ryzen, Threadripper and EPYC CPU;
note that some low-power Intel CPUs (e.g., older Atom, Celeron and Pentium models) do not support it.
- **AVX-512** (512-bit registers, i.e., eight double-precision values per instruction) 
is also an x86 instruction set. 
It is supported by AMD's Zen 4 and Zen 5 CPUs (e.g., Ryzen 7000 and 9000, and EPYC 9004 and 9005), 
by Intel's Xeon (server) CPUs since 2017, 
and by some 10th- and 11th-generation Intel laptop and desktop CPUs;
however, it is switched off in Intel's 12th-generation (and later) consumer CPUs.
Note that Zen 4 CPUs run each 512-bit instruction in two 256-bit halves;
nevertheless, on our Zen 4 server, the AVX-512 versions were at least as fast as the AVX2 versions 
in almost every case.
- **NEON** (also called Advanced SIMD; 128-bit registers, i.e., two double-precision values per instruction) 
is the SIMD instruction set of ARM CPUs, which have no AVX. 
It is supported by every 64-bit ARM CPU, for instance:
Apple Silicon (M1-M4) Macs,
Qualcomm Snapdragon CPUs (including the Snapdragon X CPUs of Windows on ARM laptops),
the Raspberry Pi 3, 4 and 5,
and ARM servers such as AWS Graviton, NVIDIA Grace and Ampere Altra.
Some ARM server CPUs also have SVE / SVE2 (scalable vector extensions), which NicoSIMD does not use.


The custom vectorised functions (the AVX2 and AVX-512 functions, the fused log densities, 
and the mathematical functions such as `exp()`, `log()` and `Phi()`) 
are provided by NicoSIMD, a separate, header-only R package, 
which is installed automatically with NicoStan (by `install_NicoStan()`).
Note that neither NicoStan nor BayesMVP requires them: 
both run on the plain Stan math functions, just more slowly.


Since every model is compiled on the user's own computer, 
the instruction set is chosen when the model is compiled, based on what the CPU supports
(i.e., there is no run-time dispatch: the compiled model contains one set of SIMD functions only).
More specifically, `NicoSIMD::fn_include_folder_Stan_user_header_and_build_arguments_of_NicoSIMD_for_this_CPU()`
reads the CPU's instruction sets 
(from `/proc/cpuinfo` on Linux; otherwise, e.g., on Windows and macOS, from the macros which R's own C++ compiler
defines with `-march=native`, i.e., the flag each model is compiled with),
and chooses one of these four options:


| Option | CPUs (examples) | Vector width (doubles per instruction) | Compiler flags | What is compiled |
|---|---|---|---|---|
| **AVX-512** | AMD Zen 4 and Zen 5 (e.g., Ryzen 7000 and 9000, EPYC 9004 and 9005); Intel Xeon (server) CPUs since 2017 (Skylake-SP onwards); some 10th- and 11th-generation Intel laptop and desktop CPUs | 8 (512-bit `zmm` registers) | `-O3 -march=native -mtune=native -mfma -mavx2 -mavx512f -mavx512vl -mavx512dq` | the AVX-512 kernels (`__m512d`), natively |
| **AVX2** | practically every x86-64 (Intel and AMD) CPU made since 2013-2015: Intel since Haswell (2013), and AMD since Excavator (2015), including every Ryzen, Threadripper and EPYC CPU | 4 (256-bit `ymm` registers) | `-O3 -march=native -mtune=native -mfma -mavx2` | the AVX2 kernels (`__m256d`), natively |
| **NEON (through SIMDe)** | every 64-bit ARM (AArch64) CPU: Apple Silicon (M1-M4) Macs, Qualcomm Snapdragon (e.g., the Snapdragon X CPUs of Windows on ARM laptops), AWS Graviton, NVIDIA Grace, Ampere Altra, and the Raspberry Pi 4 and 5 (with a 64-bit operating system) | 2 (128-bit NEON registers, `float64x2_t`) | `-O3 -mcpu=native` | the AVX2 kernels, translated into NEON instructions by [SIMDe](https://github.com/simd-everywhere/simde) |
| **Plain Stan math** | any other CPU (e.g., an x86-64 CPU without AVX2, or a 32-bit ARM CPU) | Eigen's own (see below) | `-O3 -march=native -mtune=native` | no NicoSIMD functions: the model uses Stan's own math library and Eigen |


Some more detail on each option:

- **AVX-512** requires three of its extensions: AVX-512F (the foundation), AVX-512VL (the 512-bit instructions
also on 256-bit registers) and AVX-512DQ (e.g., the conversions between 64-bit integers and doubles, which the
`exp()` and `log()` kernels use). Note that Intel switched AVX-512 off in its 12th-generation (Alder Lake) and later
consumer CPUs, since their efficiency cores do not support it; it is expected to return in future Intel desktop
CPUs, through AVX10. Furthermore, Zen 4 CPUs run each 512-bit instruction as two 256-bit halves;
nevertheless, on our Zen 4 server, the AVX-512 versions were at least as fast as the AVX2 versions in almost every
case (with half as many instructions to decode).
- **AVX2** always comes with FMA3 (fused multiply-add, i.e., `a * b + c` in one instruction, with one rounding),
which the polynomial approximations of the kernels (e.g., of `exp()`, `log()` and `Phi()`) use at every step.
- **NEON (through SIMDe):** [SIMDe](https://github.com/simd-everywhere/simde) ("SIMD Everywhere"; header-only,
MIT licence, included in NicoSIMD) implements each x86 intrinsic with the target CPU's own instructions.
With `SIMDE_ENABLE_NATIVE_ALIASES`, the kernels' own intrinsics (e.g., `_mm256_fmadd_pd`) and types (`__m256d`)
refer to SIMDe's versions, so that the AVX2 kernels compile unchanged on ARM CPUs;
since a NEON register holds two doubles (vs. four for AVX2), each 256-bit AVX2 instruction becomes two 128-bit
NEON instructions (e.g., `_mm256_fmadd_pd` becomes two `vfmaq_f64`).
This is close to hand-written NEON code for the arithmetic, although a few AVX2 instructions without a single NEON
equivalent (e.g., the gathers) take several NEON instructions.
Note that NicoSIMD does not use the scalable vector extensions (SVE / SVE2) of some ARM server CPUs.
- **Plain Stan math:** the model is compiled with Stan's own math library
(i.e., exactly as Stan compiles it), whose linear algebra uses [Eigen](https://eigen.tuxfamily.org);
Eigen itself vectorises its matrix and vector operations with whatever SIMD instructions the compiler enables
(e.g., SSE2, AVX or NEON), but the elementwise transcendental functions (e.g., `exp()`, `log()` and `Phi()`)
and the log densities are Stan's own scalar versions.


In every case, a compiled model reports the vector width of its NicoSIMD functions through `bmvp_simd_lanes()`
(i.e., 8 for AVX-512, and 4 for AVX2 and for NEON through SIMDe, since SIMDe keeps the AVX2 kernels' width),
so that one can check which version is being used.


To check which of these your CPU supports:
on Linux, `grep -o -w -E 'avx2|avx512f|asimd' /proc/cpuinfo | sort -u` 
(where `asimd` means NEON, on ARM CPUs);
on Windows, a tool such as CPU-Z;
on an Intel Mac, `sysctl -a | grep machdep.cpu`
(every Apple Silicon Mac supports NEON, and none supports AVX).


<!-- ------------------------------------------------------------------------------------------------------------------------------- -->
## How to cite NicoStan
<!-- ------------------------------------------------------------------------------------------------------------------------------- -->


If you use NicoStan in your work, you can cite the package as follows:


Cerullo, E. (2026). NicoStan: Efficient MCMC for Stan models, with advanced between-chain adaptation and diffusion-pathspace HMC.
R package version 0.1.9000. https://github.com/CerulloE1996/NicoStan


```bibtex
@Manual{Cerullo2026NicoStan,
  title = {NicoStan: Efficient MCMC for Stan models,
  with advanced between-chain adaptation and diffusion-pathspace HMC},
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


If you use the BayesMVP extension (i.e., the specialised MVP/MVOP-based models and/or the custom AVX2/AVX-512 functions), you can also cite:


Cerullo, E. (2026). BayesMVP: Accelerated multivariate probit models using NicoStan.
R package version 0.1.9000. https://github.com/CerulloE1996/BayesMVP

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

22. Pinkney, S. (2024). [A Short Note on a Flexible Cholesky Parameterization of Correlation Matrices](https://arxiv.org/abs/2405.07286). arXiv:2405.07286.


<!-- ------------------------------------------------------------------------------------------------------------------------------- -->
## Package citation and development
<!-- ------------------------------------------------------------------------------------------------------------------------------- -->


NicoStan is developed by Enzo Cerullo and licensed under GPL-3.
Package citation metadata is provided in [CITATION.cff](CITATION.cff),
and the references above are available as a [BibTeX file](docs/references.bib).



