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
[nico_reduce_sum and nico_map_rect](#nico_reduce_sum-and-nico_map_rect) ·
[AVX-512 and AVX2 speed-ups over Stan::math](#avx-512-and-avx2-speed-ups-over-stanmath) ·
[How NicoStan works](#how-nicostan-works) ·
[NicoStan's efficient burnin algorithms (SNAPER-HMC and ChEES-R-HMC)](#efficient-burnin-algorithms-snaper-hmc-and-chees-r-hmc) ·
[Blended trajectory-length criteria (geometric means of ESJD, ChEES-R and SNAPER)](#blended-trajectory-length-criteria-geometric-means-of-esjd-chees-r-and-snaper) ·
[Custom AVX2 and AVX-512 functions](#custom-avx2-and-avx-512-functions) ·
[How to cite NicoStan](#how-to-cite-nicostan) ·
[How to cite BayesMVP](#how-to-cite-bayesmvp) ·
[References](#references) ·
[Package citation and development](#package-citation-and-development)



The trajectory-length criteria that NicoStan offers, and where each one sits in the criterion family of
[Riou-Durand et al., 2023](https://proceedings.mlr.press/v206/riou-durand23a.html):

| φ(x): whose jump is measured | ρ = −1 (no length penalty)¹ | ρ = 0 (÷ √τ) | ρ = 1 (÷ τ)² | ρ adaptive³ |
| --- | --- | --- | --- | --- |
| x | ESJD ([Pasarica & Gelman, 2010](https://www3.stat.sinica.edu.tw/statistica/J20N1/J20N113/J20N113.html)) | [Wang, Mohamed & de Freitas (2013)](https://arxiv.org/abs/1302.6182) | ESJD rate (NicoStan "ESJD") | - |
| ‖x − m‖² | ChEES ([Hoffman, Radul & Sountsov, 2021](https://proceedings.mlr.press/v130/hoffman21a.html)) | - | ChEES rate (NicoStan "CHESSR") | - |
| (wᵀ(x − m))² | - | - | SNAPER ([Sountsov & Hoffman, 2022](https://arxiv.org/abs/2110.11576v3)) | Adaptive MALT ([Riou-Durand et al., 2023](https://proceedings.mlr.press/v206/riou-durand23a.html)) |

The criterion family is Cρ(τ) = ESJDφ(τ) / τ^((1+ρ)/2) (Riou-Durand et al., 2023).
m is the posterior mean and w is the leading principal direction.

¹ Outside the range ρ ∈ [0, 1] that Riou-Durand et al. consider; included here as the criteria with no length penalty.

² With jittered trajectory lengths, NicoStan divides each trajectory's squared jump by its own length;
"CHESSR_log" instead divides the average by τ.
Note that TensorFlow Probability's implementation divides by the length plus one step size,
i.e., it charges the extra gradient evaluation of each iteration.
NicoStan offers the same (and the exact expected gradient cost) via the `$sample()` argument
`rate_criterion_cost_offset_steps` (`"auto"`, or the number of extra steps);
however, by default (`rate_criterion_cost_offset_steps = 0`) it divides by the length alone,
which was faster on our latent-class benchmarks
(see the validation note in [Efficient burnin algorithms](#efficient-burnin-algorithms-snaper-hmc-and-chees-r-hmc)).

³ ρ set to an estimate of the lag-1 autocorrelation of φ.

NicoStan's blended criteria, "ESJD_CHESSR" and "ESJD_SNAPER" (geometric means of two ρ = 1 criteria),
and "ESJD_CHESSR_SNAPER" and "ESJD_w50_CHESSR_w25_SNAPER_w25" (equally and 0.50 / 0.25 / 0.25 weighted geometric
means of all three), combine criteria with different φ, so they are not members of this family
(see [Blended trajectory-length criteria](#blended-trajectory-length-criteria-geometric-means-of-esjd-chees-r-and-snaper)).



<!-- ------------------------------------------------------------------------------------------------------------------------------- -->
## What is NicoStan?
<!-- ------------------------------------------------------------------------------------------------------------------------------- -->


NicoStan is an R package for efficiently fitting Bayesian models written in the probabilistic programming language
[Stan](https://mc-stan.org/).


NicoStan accesses Stan's log posterior and its gradient through
[BridgeStan](https://roualdes.us/bridgestan/latest/) ([Roualdes et al., 2023](https://doi.org/10.21105/joss.05236));
more specifically, NicoStan's C++ sampler calls the compiled Stan model directly through BridgeStan's C/C++ interface.
NicoStan itself handles the burnin, the sampling and the posterior summaries.


NicoStan grew out of our work on efficient sampling for multivariate probit (MVP) models,
specifically their latent class extensions (e.g., the LC-MVP), which are used for the evaluation
of diagnostic/screening test accuracy without a perfect gold standard
(see e.g., [Xu et al., 2013](https://doi.org/10.1002/sim.5695); [Xu and Craig, 2009](https://doi.org/10.1111/j.1541-0420.2008.01194.x);
[Uebersax, 1999](https://doi.org/10.1177/01466219922031400); and Cerullo et al., 2025,
[Latent class multivariate probit and latent trait models for evaluating test accuracy without a gold standard: A simulation study](https://arxiv.org/abs/2509.18489)).


Also, NicoStan is named after a cat, who provided great stress-relief whilst I was making NicoStan,
and made waiting 10+ minutes for C++ code to recompile tolerable.


NicoStan was formerly called "BayesMVP" (since we have mainly been working with multivariate probit [MVP] models);
however, it is now split into 3 R packages:
NicoStan, the general Stan interface (which relies heavily on [BridgeStan](https://github.com/roualdes/bridgestan));
[NicoSIMD](#nicosimd-the-simd-functions-on-every-device), which holds every AVX2 / AVX-512 function
and is installed with NicoStan;
and [BayesMVP](https://github.com/CerulloE1996/BayesMVP), which is now an extension to NicoStan,
rather than a standalone R package.


The [BayesMVP R package extension](https://github.com/CerulloE1996/BayesMVP) to NicoStan adds highly optimised,
specialised implementations of these models, with manually implemented gradients
(see the [BayesMVP](#bayesmvp-multivariate-probit-models) section below).
For instance, we used BayesMVP to fit the models in our simulation study
([Cerullo et al., 2025](https://arxiv.org/abs/2509.18489)),
which compared the LC-MVP and latent trait ([Qu et al., 1996](https://pubmed.ncbi.nlm.nih.gov/8805757/)) models.


To summarise, the NicoStan R package provides:
<!-- ----------------------------------------------------------------------------------------- -->
- **A general Stan interface**, so that existing Stan models can be fitted via NicoStan
(directly, using the user's existing `.stan` model files).
<!-- ----------------------------------------------------------------------------------------- -->
- **nico_reduce_sum and nico_map_rect:** NicoStan's versions of Stan's
[`reduce_sum`](https://mc-stan.org/docs/stan-users-guide/parallelization.html) and `map_rect`,
which move `reduce_sum`'s bookkeeping inside the chunks;
furthermore, the nuisance blocks are held by NicoStan inside the chunks,
so that the main autodiff tape holds no node per nuisance value.
NicoStan also chooses the number of threads per chain and the number of chunks,
separately for the burnin and for the sampling,
based on the tape-chunking method of
"Cache-aware chunking dramatically improves parallel scaling for HMC: Application to multivariate probit model"
(in preparation).
See [nico_reduce_sum and nico_map_rect](#nico_reduce_sum-and-nico_map_rect) for more information.

<!-- ----------------------------------------------------------------------------------------- -->
**How the implementations are compared.** Every model is compared on the following axes, where applicable:

| Comparison | Choices |
|---|---|
| Math functions | nico::math (NicoSIMD's custom AVX2, AVX-512 or NEON (via SIMDe) C++ functions) versus Stan::math |
| Fusion | Fused functions wherever available versus non-fused |
| Sum calculation | nico_reduce_sum versus Stan's reduce_sum |
| Map calculation (Cox only) | nico_map_rect versus Stan's map_rect |

Usually eight combinations, four when an axis does not apply, and up to sixteen for the Cox model with the map axis;
a missing implementation is not "N/A". The table names the axes only; the timings are reported separately.
<!-- ----------------------------------------------------------------------------------------- -->
- **Hybrid diffusion-pathspace HMC** for models with suitable Gaussian latent/nuisance blocks,
based on [Beskos et al., 2011](https://doi.org/10.1016/j.spa.2011.06.003),
and [Beskos et al., 2013](https://doi.org/10.1016/j.spa.2012.12.001).
See [Models with nuisance parameters](#models-with-nuisance-parameters-diffusion-pathspace-hmc).
<!-- ----------------------------------------------------------------------------------------- -->
- **Between-chain adaptation during burnin:** all chains share the same step size ε, trajectory length τ and
metric M; at each burnin iteration, the gradients which Adam ([Kingma and Ba, 2015](https://arxiv.org/abs/1412.6980))
uses to adapt ε and τ are averaged across the chains, and M is estimated from all of the chains,
so that each burnin iteration gives as much information for the adaptation as N_chains iterations of one chain
([Hoffman et al., 2021](https://proceedings.mlr.press/v130/hoffman21a.html)).
<!-- ----------------------------------------------------------------------------------------- -->
- **ChEES, ChEES-R and SNAPER-HMC trajectory-length adaptation during burnin**,
based on [Hoffman et al., 2021](https://proceedings.mlr.press/v130/hoffman21a.html),
and [Sountsov and Hoffman, 2022](https://arxiv.org/abs/2110.11576v3),
as well as our blended criteria (geometric means of ESJD, ChEES-R and SNAPER).
See [Efficient burnin algorithms](#efficient-burnin-algorithms-snaper-hmc-and-chees-r-hmc) and
[Blended trajectory-length criteria](#blended-trajectory-length-criteria-geometric-means-of-esjd-chees-r-and-snaper)
for more information.
<!-- ----------------------------------------------------------------------------------------- -->
- **Custom AVX2 and AVX-512 maths functions**, supplied by
[NicoSIMD](#nicosimd-the-simd-functions-on-every-device) (installed with NicoStan),
and available to general Stan models
(on 64-bit ARM CPUs, through [SIMDe](https://github.com/simd-everywhere/simde));
however, note that your `.stan` model file will need to be re-written to declare the custom functions
(e.g. `vector nico_Phi(vector x);`, with no body), and to replace standard Stan math functions (e.g. `Phi()`)
with their nico_ counterparts (e.g. `nico_Phi()`),
with NicoSIMD's C++ `.hpp` header file supplied when compiling (via `Stan_cpp_user_header`).
See [Custom AVX2 and AVX-512 functions](#custom-avx2-and-avx-512-functions).
<!-- ----------------------------------------------------------------------------------------- -->
- **Partially-dense metric:**
NicoStan allows you to set a dense HMC mass matrix (M) for the main model parameters
(`metric_shape_main = "dense"`),
combined with a diagonal M for the high-dimensional nuisance parameters.
In other words, the main and nuisance metrics can be chosen separately.
This is useful, because it is often intractable to fit a dense M on the entire parameter set,
and a dense M for the main model parameters often leads to better sampling than a diagonal M,
since it takes posterior correlations into account.
Furthermore, users can set either an empirical M or a Hessian-based M
(the Hessian computed by central differences of the gradient) for the main model parameters
(`metric_type_main = "Empirical"` or `"Hessian"`).
<!-- ----------------------------------------------------------------------------------------- -->


<!-- ------------------------------------------------------------------------------------------------------------------------------- -->
## How is NicoStan different to Stan (e.g., cmdstanr/rstan)?
<!-- ------------------------------------------------------------------------------------------------------------------------------- -->


For the burnin (or "warmup") phase, [Stan](https://mc-stan.org/) uses the No-U-Turn sampler
(NUTS-HMC; see [Hoffman and Gelman, 2014](https://www.jmlr.org/papers/v15/hoffman14a.html)),
which sets the HMC path length ($\tau$) automatically, within each chain.

On the other hand, NicoStan tunes $\tau$ with state-of-the-art, between-chain adaptation algorithms,
such as SNAPER-HMC ([Sountsov and Hoffman, 2022](https://arxiv.org/abs/2110.11576v3)),
ChEES-HMC ([Hoffman et al., 2021](https://proceedings.mlr.press/v130/hoffman21a.html))
and ChEES-R-HMC ([Sountsov and Hoffman, 2022](https://arxiv.org/abs/2110.11576v3)),
which is NicoStan's default (`burnin_algorithm = "CHESSR"`).

NicoStan also offers geometric means of these criteria with the ESJD criterion
(`burnin_algorithm = "ESJD_CHESSR"`, `"ESJD_SNAPER"`, `"ESJD_CHESSR_SNAPER"` and `"ESJD_w50_CHESSR_w25_SNAPER_w25"`).
See [this section below](#efficient-burnin-algorithms-snaper-hmc-and-chees-r-hmc) for more details on NicoStan's burnin algorithms.


This difference in burnin adaptation algorithm makes NicoStan more efficient than Stan for most models
(see the [Benchmarks](#benchmarks) section below).

In our testing, NicoStan can also perform very well with a very short burnin
(100-125 iterations with just 4 chains; we are also currently testing shorter burnins).

Furthermore, since NicoStan uses between-chain adaptation
(as opposed to within-chain adaptation, like Stan's NUTS-HMC algorithm),
all chains finish the sampling phase at approximately the same time.
This avoids the common issue of "stuck chains",
which is often seen with complex models when using Stan directly
(e.g., via [cmdstanr](https://mc-stan.org/cmdstanr/) or [rstan](https://mc-stan.org/rstan/)).


Additionally, for models with high-dimensional nuisance parameters
(see [this section below](#models-with-nuisance-parameters-diffusion-pathspace-hmc) for examples of such models),
NicoStan offers a hybrid diffusion-pathspace HMC sampling algorithm
(based on [Beskos et al., 2011](https://doi.org/10.1016/j.spa.2011.06.003)
and [Beskos et al., 2013](https://doi.org/10.1016/j.spa.2012.12.001)),
which can greatly increase efficiency, especially for large N.

Furthermore, as we mentioned above (see [this section](#what-is-nicostan)), unlike Stan,
NicoStan allows you to set a dense HMC mass matrix ($\mathbf{M}$) for the main model parameters
(`metric_shape_main = "dense"`),
combined with a diagonal $\mathbf{M}$ for the high-dimensional nuisance parameters (`metric_shape_nuisance = "diag"`).


Note that NicoStan can also fit any Stan model (i.e., any `.stan` model file);
however, expect efficiency gains (relative to Stan) to be less dramatic for models without high-dimensional nuisance parameters.


<!-- ------------------------------------------------------------------------------------------------------------------------------- -->
## Installation
<!-- ------------------------------------------------------------------------------------------------------------------------------- -->

**Warning:** these R packages (NicoStan, NicoSIMD and BayesMVP) have not been tested on Windows in over 12 months,
and have never been tested on macOS; almost all of the testing and development has been done on Linux
(with x86 AMD CPUs). If you have any issues, please send an email to: enzo.cerullo@bath.edu.


NicoStan requires:


- An [R](https://www.r-project.org/) installation, a C++17 compiler (the build uses
[GCC](https://gcc.gnu.org/)'s `g++` with `-std=gnu++17`) and [GNU make](https://www.gnu.org/software/make/).
- The [cmdstanr](https://mc-stan.org/cmdstanr/) R interface and the [CmdStan](https://mc-stan.org/users/interfaces/cmdstan)
C++ source tree. See the [cmdstanr installation guide](https://mc-stan.org/cmdstanr/articles/cmdstanr.html).
- The [BridgeStan](https://github.com/roualdes/bridgestan) R interface and its matching C++ source tree.
See the [BridgeStan installation guide](https://roualdes.us/bridgestan/latest/languages/r.html).


NicoStan compiles against the [Stan math library](https://github.com/stan-dev/math) of your CmdStan source tree,
including the copies of [Eigen](https://eigen.tuxfamily.org/) (3.4.0) and [TBB](https://github.com/uxlfoundation/oneTBB)
that come with it.

The NicoSIMD package (the header-only C++ of the nico_ AVX2 / AVX-512 functions, with NEON on 64-bit ARM CPUs through
[SIMDe](https://github.com/simd-everywhere/simde)) is bundled with NicoStan;
`install_NicoStan()` installs it first, before NicoStan itself, so it needs no separate installation.


The current development build uses CmdStan 2.36.0 and BridgeStan 2.6.2.
NicoStan uses `~/.cmdstan/cmdstan-2.36.0` when that folder exists (otherwise the CmdStan path of cmdstanr),
and the newest `~/.bridgestan/bridgestan-*` folder.

We have tested NicoStan mostly on Linux, using R 4.3.3 and R 4.4.0 with GCC 11.4,
with the C++ source trees at `~/.cmdstan/cmdstan-2.36.0` and `~/.bridgestan/bridgestan-2.6.2`.


The compiler flags are set by the installer itself, for the CPU it is built on:
`-O3 -march=native -mtune=native`, plus `-mfma` when the CPU has FMA, and then:

- on a CPU with AVX-512: `-mavx -mavx2 -mavx512f -mavx512cd -mavx512bw -mavx512dq -mavx512vl`;
- on a CPU with AVX2 (but not AVX-512): `-mavx -mavx2`;
- on a 64-bit ARM CPU: `-O3 -mcpu=native` instead.

These can be overridden with the `CUSTOM_FLAGS` argument of `install_NicoStan()`
(a named list; the accepted names include `CXX_COMPILER`, `CXX_STD`, `CPU_BASE_FLAGS`, `FMA_FLAGS` and `AVX_FLAGS`).


If you have any installation issues (e.g., C++ compiler errors) or bugs, please email me at:
enzo.cerullo@bath.edu, or open an issue on [GitHub](https://github.com/CerulloE1996/NicoStan/issues).

### Installation from local source files

If you have the NicoStan source files on your computer, source the administrator installer in `inst/examples`
([`NicoStan_admin_install.R`](inst/examples/NicoStan_admin_install.R)):


```r
source("path/to/NicoStan/inst/examples/NicoStan_admin_install.R")
```

The whole build runs in a clean R process, so that nothing is loaded or unloaded in your own R session.

**Restart R after the installation, then load `library(NicoStan)`.**

The example source files are in `inst/examples`, and
the installation copies them into the compiled package
(after installation, they are in the folder given by `system.file("examples", package = "NicoStan")`).


### Installation from GitHub

For installation from GitHub, start with a fresh R session and a writable package library,
and install [remotes](https://remotes.r-lib.org/) (or [devtools](https://devtools.r-lib.org/)) first:

```r
install.packages("remotes")
remotes::install_github(repo = "CerulloE1996/NicoStan",
                        upgrade = "never")
NicoStan::install_NicoStan()
```

**Restart R, then load `library(NicoStan)`.**

Note that `remotes::install_github()` installs the outer R package and its installer;
`install_NicoStan()` then installs NicoSIMD and compiles and installs the sampler itself.

Also, make sure to restart R before switching between the installed and development copies.

### Installing the [BayesMVP](https://github.com/CerulloE1996/BayesMVP) extension

The native (manually differentiated) multivariate probit and ordinal probit models and their examples
additionally require the NicoStan-compatible [BayesMVP](https://github.com/CerulloE1996/BayesMVP) extension.
(The nico_ AVX functions themselves come from NicoSIMD, which is installed with NicoStan.)

After installing NicoStan, run the following in a fresh R session:

```r
remotes::install_github(repo = "CerulloE1996/BayesMVP",
                        ref = "main",
                        upgrade = "never")
BayesMVP::install_BayesMVP()
```

**Restart R before loading the compiled BayesMVP package.**

`install_BayesMVP()` stops if NicoStan (version 0.1.9000 or later) is not installed.

Note that this command installs the
[main branch](https://github.com/CerulloE1996/BayesMVP/tree/main) of [BayesMVP](https://github.com/CerulloE1996/BayesMVP).


<!-- ------------------------------------------------------------------------------------------------------------------------------- -->
## Compatibility across devices
<!-- ------------------------------------------------------------------------------------------------------------------------------- -->


**Warning:** these R packages (NicoStan, NicoSIMD and BayesMVP) have not been tested on Windows in over 12 months,
and have never been tested on macOS; almost all of the testing and development has been done on Linux
(with x86-64 AMD CPUs). If you have any issues, please send an email to: enzo.cerullo@bath.edu.


This section applies to all three packages:

- **NicoStan** (the sampler): written to compile on any 64-bit CPU with a C++17 compiler
(i.e., x86-64 and 64-bit ARM CPUs, on Linux, Windows and macOS).
- **NicoSIMD** (the SIMD functions, installed with NicoStan): which SIMD functions are used is decided when the
model is compiled, from the instruction sets the compiler is told to use (`-march=native` on x86-64,
`-mcpu=native` on 64-bit ARM), not whilst the model runs:
the AVX-512 functions (8 doubles per operation) when the model is compiled with AVX-512F, AVX-512VL and AVX-512DQ;
otherwise the AVX2 functions (4 doubles per operation) when it is compiled with AVX2;
on a 64-bit ARM CPU the same AVX2 functions, translated into NEON instructions by
[SIMDe](https://github.com/simd-everywhere/simde) (each 256-bit AVX2 operation becomes two 128-bit NEON operations);
on any other CPU, no SIMD functions, and the model runs on Stan::math
(see [NicoSIMD: the SIMD functions on every device](#nicosimd-the-simd-functions-on-every-device)).
- **[BayesMVP](https://github.com/CerulloE1996/BayesMVP)** (the built-in, manual-gradient models; see
[BayesMVP: multivariate probit models](#bayesmvp-multivariate-probit-models)): uses the AVX-512 or AVX2
functions on x86-64 CPUs, and scalar C++ (one double per operation; i.e., no NEON version yet) on ARM CPUs.


| Device / operating system | SIMD functions used | Notes |
|---|---|---|
| Linux, Intel / AMD CPU | AVX-512 (if the CPU has AVX-512F, AVX-512VL and AVX-512DQ), else AVX2 (with FMA) | tested (AMD EPYC 9654, Zen 4; AMD Ryzen 7 5800H, Zen 3) |
| Windows, Intel / AMD CPU | AVX-512 or AVX2 | requires the Clang compiler of [Rtools](https://cran.r-project.org/bin/windows/Rtools/) (`clang64`); with GCC, the AVX functions are switched off |
| macOS, Intel CPU | AVX2 (or AVX-512 on some Mac Pro and iMac Pro models) | not yet tested |
| macOS, Apple Silicon (M1-M4) | the AVX2 functions as NEON instructions (via SIMDe) | not yet tested; NicoStan runs its chains and their within-chain threads with [OpenMP](https://www.openmp.org), which Apple's compiler does not include by default ([libomp](https://formulae.brew.sh/formula/libomp) may need to be installed, e.g., via Homebrew) |
| Windows on ARM (e.g., Snapdragon X) | the AVX2 functions as NEON instructions (via SIMDe) | not yet tested |
| Any other CPU (e.g., an x86-64 CPU without AVX2) | none: Stan::math (and [Eigen](https://eigen.tuxfamily.org)) | |


Note that the automatic AVX version of a Stan model (`Stan_model_math_functions = "NicoSIMD_AVX512"`,
`"NicoSIMD_AVX2"` or `"NicoSIMD_AVX512_if_supported_else_AVX2"` in `Nico_model$new()`) currently reads the
instruction sets of the CPU from `/proc/cpuinfo`, so it works on Linux with an x86-64 CPU only;
on Windows, macOS and ARM CPUs it stops with an error saying so.


Note that we have not yet tested NicoStan, BayesMVP or the SIMD functions on any ARM CPU
(i.e., neither on Apple Silicon Macs nor on Snapdragon CPUs), nor on macOS;
hence, bug reports are very welcome
(please email me, see the [Installation](#installation) section for my email address,
or open an issue on [GitHub](https://github.com/CerulloE1996/NicoStan/issues)).
Furthermore, our benchmarks were run on an AMD EPYC 9654 (Zen 4) and an AMD Ryzen 7 5800H (Zen 3),
so they do not apply to ARM CPUs;
the speed-ups of the NEON versions may be smaller than those of the AVX2 and AVX-512 versions.


<!-- ------------------------------------------------------------------------------------------------------------------------------- -->
## Examples
<!-- ------------------------------------------------------------------------------------------------------------------------------- -->

The general examples are grouped by their parameterisation:


- **Hybrid/joint HMC with nuisance diffusion:** Cox shared frailty, hierarchical logistic regression,
joint longitudinal-survival modelling, a discrete-time AR(1) stochastic-volatility model
([EASY_Stochastic_volatility.stan](inst/examples/models/EASY_Stochastic_volatility.stan))
and a latent-diffusion survival model (its nuisance block: standard normal Brownian increments).
Continuous-time SV can introduce the additional path-discretisation difficulties described
[below](#models-with-nuisance-parameters-diffusion-pathspace-hmc).
- **Standard HMC:** Weibull survival regression, robust Student-t regression and marginal Gaussian process (GP)
regression (the latent GP integrated out, so these three models have no nuisance block).


The logistic random-intercept example gives a short introduction to the API:


```r
source(system.file("examples", "random_intercepts.R", package = "NicoStan"))
fit <-  run_random_intercepts(burnin_algorithm = "CHESSR")
diagnostics <-  fit$summary(save_log_lik_trace = FALSE)
fit$model_fit_object$summaries$summary_tibbles$summary_tibble_main_params

## Other trajectory-length adaptation options (burnin_algorithm), e.g.:
## "SNAPER", "ChEES", "ESJD_CHESSR", "ESJD_CHESSR_SNAPER", "ESJD_w50_CHESSR_w25_SNAPER_w25"
```


The trajectory-length criteria that `burnin_algorithm` accepts (the default is `"CHESSR"`, i.e., ChEES-R;
the names are not case-sensitive, e.g., `"CHEESR"` is also accepted):

- `"CHESSR"` (the default): ChEES-R, the ChEES criterion ([Hoffman et al., 2021](https://proceedings.mlr.press/v130/hoffman21a.html)) per unit trajectory length
(see [Efficient burnin algorithms (SNAPER-HMC and ChEES-R-HMC)](#efficient-burnin-algorithms-snaper-hmc-and-chees-r-hmc)).
- `"ChEES"`: the original ChEES criterion ([Hoffman et al., 2021](https://proceedings.mlr.press/v130/hoffman21a.html)).
- `"CHESSR_log"`: a smoothed log-rate variant of ChEES-R.
- `"SNAPER"`: the SNAPER criterion ([Sountsov and Hoffman, 2022](https://arxiv.org/abs/2110.11576v3)).
- `"CHESSR_time"` and `"SNAPER_time"`: the time-to-target-ESS variants of ChEES-R and SNAPER (joint sampler only;
settings in `time_criterion_settings`).
- `"ESJD"`: the expected squared jumped distance ([Pasarica and Gelman, 2010](https://doi.org/10.2139/ssrn.1010403)) per unit trajectory length.
- `"ESJD_CHESSR"`: the equally weighted geometric mean of the ESJD and ChEES-R criteria.
- `"ESJD_SNAPER"`: the equally weighted geometric mean of the ESJD and SNAPER criteria.
- `"ESJD_CHESSR_SNAPER"`: the equally weighted geometric mean of the ESJD, ChEES-R and SNAPER criteria.
- `"ESJD_w50_CHESSR_w25_SNAPER_w25"`: the same, with the weights 0.50, 0.25 and 0.25.

The four blended criteria are described in
[Blended trajectory-length criteria](#blended-trajectory-length-criteria-geometric-means-of-esjd-chees-r-and-snaper).


The summary tibbles report, for each parameter, the posterior mean, SD and quantiles,
the bulk and tail ESS (`n_eff` and `n_eff_tail`; [Vehtari et al., 2021](https://doi.org/10.1214/20-BA1221)),
the ESS of the centred squared draws (`n_eff_sd`, i.e., the ESS for estimating the posterior SD),
and the R-hat and nested R-hat ([Margossian et al., 2024](https://arxiv.org/abs/2110.13017v6))
convergence diagnostics (`Rhat` and `n_Rhat`).


This example simulates data with 20 groups and 200 observations, with a standard normal latent vector
(i.e., the non-centred random intercepts) as the nuisance block, named with `nuisance_params = "z"`.
See the complete [R example](inst/examples/random_intercepts.R) and [Stan model](inst/examples/random_intercepts.stan)
for the full parameterisation and settings.
Note that the [R6](https://r6.r-lib.org) class is called `NicoStan::Nico_model` (`NicoStan::MVP_model` also still
works, for compatibility reasons);
`Model_type = "Stan"` must be selected when using a `.stan` model file; the other `Model_type` options require the
[BayesMVP R package extension](https://github.com/CerulloE1996/BayesMVP) to be installed
(see [BayesMVP: multivariate probit models](#bayesmvp-multivariate-probit-models)).


The wider comparison examples are available through:


```r
source(system.file("examples", "NicoStan_examples.R", package = "NicoStan"))
result <-  run_NicoStan_example(model = "hierarchical_logistic",
                                engine = "NicoStan",
                                math_backend = "Stan")
```


To select the version that runs:


- Use `engine = "cmdstanr"` for [Stan](https://mc-stan.org)'s NUTS ([Hoffman and Gelman, 2014](https://www.jmlr.org/papers/v15/hoffman14a.html)),
through [cmdstanr](https://mc-stan.org/cmdstanr/).
- Use `engine = "NicoStan", math_backend = "Stan"` for NicoStan with Stan::math.
- Use `engine = "NicoStan", math_backend = "AVX2"` for NicoStan with nico::math (AVX2).
- Use `engine = "NicoStan", math_backend = "AVX512"` for NicoStan with nico::math (AVX-512).

The `"AVX2"` and `"AVX512"` options select the prepared AVX versions of the supplied example models
(the `*_avx.stan` files); they require the
[BayesMVP extension](https://github.com/CerulloE1996/BayesMVP) and a CPU supporting the selected instruction set.
For your own `.stan` model, either call the nico_ functions in the model yourself and supply their C++ header,
or let NicoStan write the AVX version with `Stan_model_math_functions` in `Nico_model$new()`
(see [Custom AVX2 and AVX-512 functions](#custom-avx2-and-avx-512-functions)).


### Preparing your own Stan model

When preparing a model for NicoStan:

- Name every nuisance block with `nuisance_params = c(...)` in `Nico_model$new()`, in any order
(e.g., `nuisance_params = c("z_intercept", "z_slope")`); the blocks can be declared anywhere in the
`parameters` block.
NicoStan never chooses the nuisance blocks itself: with `sample_nuisance = TRUE` and no `nuisance_params`,
it stops with an error saying so.
- Set `sample_nuisance = TRUE` (or `FALSE` for a model without a nuisance block, i.e., every parameter is main).
- Note that NicoStan then writes a version of the model, in the file NicoStan writes next to the model, whose
nuisance blocks are held by NicoStan rather than declared as Stan parameters
(each declaration is kept as a comment, and the model reads the values where it uses them).
- For the nuisance blocks to be held by NicoStan inside the chunks, the model's
[reduce_sum](https://mc-stan.org/docs/stan-users-guide/parallelization.html) (or `reduce_sum_static`) must slice
the nuisance block; any other use of a block reads its values on the main tape
(see [Example 2](#example-2-several-nuisance-blocks-nuisance_params) and
[nico_reduce_sum and nico_map_rect](#nico_reduce_sum-and-nico_map_rect)).


<!-- ------------------------------------------------------------------------------------------------------------------------------- -->
## Worked examples
<!-- ------------------------------------------------------------------------------------------------------------------------------- -->


The files of Examples 1, 3 and 4 are in NicoStan's `inst/examples` folder
(i.e., `system.file("examples", package = "NicoStan")` once NicoStan is installed):
each has its Stan model and an R script which simulates or loads the data and fits the model.
Example 2 is a sketch in the text only, with no file of its own.


### Example 1: one nuisance block (logistic regression with random intercepts)


The nuisance parameters (here, the standardised random intercepts `z`) are named with `nuisance_params = "z"`.
With `diffusion_HMC = TRUE` and `partitioned_HMC = FALSE`, NicoStan samples the main and the nuisance parameters
jointly, within the same trajectory and with one Metropolis-Hastings step:
the nuisance block moves with diffusion-pathspace dynamics
([Beskos et al., 2013](https://doi.org/10.1016/j.spa.2012.12.001)) and the main parameters with standard HMC dynamics,
with the kick-flow-kick integrator of [Beskos et al., 2015](https://doi.org/10.1093/biomet/asv051).

Note that NicoStan never chooses the nuisance blocks itself:
with `sample_nuisance = TRUE` they are always named, wherever they are declared in the `parameters` block,
and a Stan model with `sample_nuisance = TRUE` and no named block stops with an error saying so.

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
  vector[J] z;
  real alpha;
  real beta;
  real<lower=0> sigma;
}
transformed parameters {
  vector[J] group_effect = sigma * z;
}
model {
  z ~ std_normal();
  alpha ~ normal(0, 1.5);
  beta ~ normal(0, 1);
  sigma ~ normal(0, 1);
  y ~ bernoulli_logit(alpha + beta * x + group_effect[group]);
}
generated quantities {
  real mean_probability = mean(inv_logit(alpha + beta * x + group_effect[group]));
}
```

Here `z` (one value per group, $J = 20$) is the nuisance block, and `alpha`, `beta` and `sigma` are the main
parameters; `group_effect = sigma * z` are the non-centred random intercepts.

It is fitted as in [random_intercepts.R](inst/examples/random_intercepts.R), which gives every sampler setting
explicitly. The model is first copied into a writable temporary folder, because NicoStan writes its version of the
model (and BridgeStan compiles it) next to the model file, which cannot be done inside the installed package:

```r
model_directory <-  tempfile(pattern = "nicostan_random_intercepts_")
dir.create(path = model_directory)
model_file <-  file.path(model_directory, "random_intercepts.stan")
stopifnot(file.copy(from = system.file("examples", "random_intercepts.stan", package = "NicoStan"),
                    to = model_file))
##
## ---- the data (N = 200 observations in J = 20 groups):
##
set.seed(seed = 2026)
n_groups <-  20
n_observations <-  200
group <-  rep(x = seq_len(n_groups), each = n_observations / n_groups)
x <-  rnorm(n = n_observations)
group_effect <-  rnorm(n = n_groups, sd = 0.6)
probability <-  plogis(q = -0.4 + 0.8 * x + group_effect[group])
stan_data <-  list(N = n_observations, J = n_groups, group = group, x = x,
                   y = rbinom(n = n_observations, size = 1, prob = probability))
##
## ---- the model, with its nuisance block named:
##
model <-  NicoStan::Nico_model$new(Model_type = "Stan",
                                    Stan_data_list = stan_data,
                                    Stan_model_file_path = model_file,
                                    nuisance_params = "z",
                                    sample_nuisance = TRUE)
##
## ---- fixed initial values of each of the 4 burn-in chains (the nuisance block z by name, in its own shape):
##
initial_values <-  lapply(X = seq_len(4), FUN = function(chain) {

        list(z = rep(x = 0, times = n_groups), alpha = 0.1 * (chain - 2.5), beta = 0, sigma = 0.6)

})
##
## ---- the fit:
##
model$sample(n_burnin = 125,
             n_iter = 250,
             n_chains_burnin = 4,
             n_chains_sampling = 4,
             n_superchains = 2,
             init_lists_per_chain = initial_values,
             burnin_algorithm = "CHESSR",
             diffusion_HMC = TRUE,
             partitioned_HMC = FALSE,
             metric_type_main = "Empirical",
             metric_shape_main = "diag",
             metric_type_nuisance = "uniform_diag",
             M_decay_type = "inverse",
             M_decay_power = 0.5,
             M_decay_scale = 1.13,
             ratio_M_main = 0.9,
             ratio_M_nuisance = 0.9,
             learning_rate = 0.05,
             learning_rate_initial = 0.1,
             n_threads_WCP_burnin = 1,
             n_threads_WCP_sampling = 1,
             adapt_delta = 0.80,
             n_refresh = 25,
             seed = 2026,
             stream = 2026,
             reorder_cols_MVP = FALSE)
##
## ---- the summaries:
##
fit_summary <-  model$summary(save_log_lik_trace = FALSE)
```

`NicoStan::Nico_model` and `NicoStan::MVP_model` are the same class (the script uses `MVP_model`).
`burnin_algorithm = "CHESSR"` is the ChEES-R criterion of
[Sountsov and Hoffman, 2022](https://arxiv.org/abs/2110.11576v3),
and `n_superchains = 2` gives the nested R-hat of [Margossian et al., 2024](https://arxiv.org/abs/2110.13017v6).

This model has no `reduce_sum()`, so each chain runs on one thread, with no chunks.
For a model that uses [`reduce_sum()`](https://mc-stan.org/docs/stan-users-guide/parallelization.html#reduce-sum)
(with the data `N` and `chunk_size`), NicoStan finds the optimal within-chain threads (WCP) and
$N_{\text{chunks}}$ of the burn-in and of the sampling for you, from the L2 and L3 caches of the computer,
either on their own before the fit or inside `$sample()` (when `n_threads_WCP_burnin`, `num_chunks_burnin`,
`n_threads_WCP_sampling` and `num_chunks_sampling` are not given); see step (d) of
[Example 3](#example-3-the-multivariate-ordered-probit-model-with-the-fused-ghk-function-and-bayesmvp).


### Example 2: several nuisance blocks (`nuisance_params`)


A model may have any number of nuisance blocks, declared anywhere in the `parameters` block;
they are given by name with `nuisance_params`, e.g., for random intercepts and random slopes
(with the `data` block of Example 1):

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

Note that the nuisance blocks of `nuisance_params` are sampled by NicoStan, so `sample_nuisance` must be `TRUE`.

The initial values of each block are given by name, in its own shape (e.g., `z_intercept = rep(0, J)`; for an
`array[E] vector[K]` block, a matrix with one row per term, $E \times K$, or a list of its $E$ vectors);
the initial values of a constrained block are given on its own, constrained scale, as for Stan;
a block left out of the initial values starts at 0 on its unconstrained scale.
Example 3 shows every step of a full fit (the priors, the nuisance blocks, the initial values, the optimal layouts
and the summaries).

With `nuisance_params`, NicoStan writes its `nico_reduce_sum` version of the model (and its C++ user header) next to
the model file. In it, the nuisance blocks are held by NicoStan, rather than declared as Stan parameters
(the blocks one after the other, each in Stan's own order of its values),
and every use of a block reads NicoStan's values where it is used; the rest of the model's code is kept as it is.
When the model uses `reduce_sum`, a nuisance block which is its sliced argument is then held by NicoStan inside the
chunks (see [nico_reduce_sum](#nico_reduce_sum-and-nico_map_rect) for what this gives).
Note that:

- **How many blocks:** any number (each one named in `nuisance_params`).
- **Which shapes:** `vector[E]`, `row_vector[E]`, `array[E] real`, `array[E] vector[K]`, `array[E] row_vector[K]`,
`matrix[R, C]` and `array[A] matrix[R, C]` blocks, and tuples of these (see below).
- **Constraints:** each of these may have the constraint `<lower = a>`, `<upper = b>`, `<lower = a, upper = b>`,
`<offset = a>`, `<multiplier = b>` or `<offset = a, multiplier = b>`, whose bounds do not use a nuisance block:
NicoStan then holds the block's unconstrained values, as Stan's own sampler does,
every value the model reads is constrained with Stan math's own transform,
and the log-Jacobian of the transform is added to the target.
Any other block (e.g., an array of tuples, or an array of more dimensions) stops with an error saying so.
- **Slicing:** Stan's `reduce_sum` slices arrays only, so a nuisance block which is the sliced argument of
`reduce_sum` must be an array (e.g., `array[N] real`, `array[N] vector[K]` or `array[N] row_vector[K]`).
- **Tuples:** a nuisance block may also be a
[tuple](https://mc-stan.org/docs/reference-manual/types.html#tuple-data-type),
so that blocks of different containers can be declared (and named in `nuisance_params`) at once, e.g.:

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
as a nuisance block of its own, in the tuple's own order (which is also Stan's order of its values);
its initial values are given as the tuple, by its name, as a list with one element per component.
Note that an array of tuples (e.g., `array[N] tuple(real, vector[K])`) is not supported;
declare it as a tuple of arrays instead (e.g., `tuple(array[N] real, array[N] vector[K])`).
### Example 3: the multivariate ordered probit model with the fused GHK function (and BayesMVP)


The multivariate ordered probit (MVOP) model - i.e., a truncated multivariate normal for each subject, whose
truncation region is set by the subject's binary and ordinal results - can be written in Stan with one of
NicoSIMD's fused functions, `nico_finite_mixture_of_multivariate_ordered_probit_GHK_log_likelihood`, which
computes the [GHK](https://doi.org/10.1016/0304-4076%2894%2901716-6) log likelihood of a chunk of subjects, and
its gradient, in one autodiff node.

It runs on any CPU with AVX-512 or AVX2, and on 64-bit ARM CPUs (e.g., Apple Silicon Macs), where NicoSIMD's
functions run with NEON instructions (through [SIMDe](https://github.com/simd-everywhere/simde)). On any other
CPU, NicoSIMD gives no user header, so a Stan model that calls the function cannot be built there.

The same function also takes 2 or 4 latent classes (i.e., the LC-MVP and the LC-MVOP), binary, ordinal and mixed
outcomes, each outcome's own covariates, missing results (coded -1, handled by the GHK method) and
interval-censored ordinal results.


This example fits a one-class MVOP to the real NHANES data (the 9575 people of the NHANES data in the
[hexbin](https://cran.r-project.org/package=hexbin) R package), with four outcomes, each with its own covariates:
iron deficiency, low albumin and cancer (binary), and anaemia (ordinal: the WHO categories of haemoglobin for each
sex, i.e., none / mild / moderate / severe).

The function is declared in the `functions` block without a body, and the nuisance block `u_raw` (one GHK value
for each subject and outcome) is the sliced argument of
[`reduce_sum_static`](https://mc-stan.org/docs/stan-users-guide/parallelization.html), so that everything whose
size grows with N is inside the chunks.

The priors are those of [BayesMVP](https://github.com/CerulloE1996/BayesMVP)'s MVOP model: normal priors on the
coefficients, an [LKJ](https://doi.org/10.1016/j.jmva.2009.04.008) prior on the correlation matrix and an [induced
Dirichlet](https://betanalpha.github.io/assets/case_studies/ordinal_regression.html) prior on the cutpoints of
each ordinal outcome.

The key parts of the Stan model (`inst/examples/models/one_class_MVOP_with_the_fused_GHK_log_likelihood.stan`)
are:

```stan
functions {
      ////
      //// ---- NicoSIMD's fused GHK log likelihood, declared without a body (its 20 arguments: the full file)
      ////
      real nico_finite_mixture_of_multivariate_ordered_probit_GHK_log_likelihood( ... );
      ////
      //// ---- the partial sum of reduce_sum_static: the log likelihood of the subjects start, ..., end
      ////
      real partial_sum_of_the_GHK_log_likelihood_lpdf( array[] row_vector u_raw_slice, int start, int end,
                                                       ... ) {
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
      array[N] row_vector[n_tests] u_raw;     //// the nuisance block (given with nuisance_params)
      vector[n_coefficients_total] beta_vec;  //// the coefficients of each outcome (its intercept first)
      cholesky_factor_corr[n_tests] L_Omega;
      vector<lower=C_raw_lower, upper=C_raw_upper>[n_cutpoints_total] C_raw;   //// first cutpoint, then log gaps
}
model {
      ...   //// the priors, including the induced Dirichlet prior on the cutpoints
      target += reduce_sum_static( partial_sum_of_the_GHK_log_likelihood_lpdf, u_raw, chunk_size, ... );
}
generated quantities {
      matrix[n_tests, n_tests] Omega = multiply_lower_tri_self_transpose(L_Omega);
}
```

NicoSIMD gives the user header and the build arguments for the CPU in use, so that the same R code runs on every
device.

The full R script (`inst/examples/Run_one_class_MVOP_with_the_fused_GHK_log_likelihood_and_BayesMVP.R`) builds the
NHANES outcomes and covariates (`y`, `X_of_each_outcome`, `n_covariates_of_test`), and then fits the model in five
steps: (a) the priors; (b) the nuisance parameters; (c) the initial values; (d) the optimal layouts of the burn-in
and of the sampling; and (e) the fit and its summaries.


**(a) The priors.** The priors are set in the Stan model's `model` block, and their values are passed to it as
data, so that they can be changed without editing the model (to change the form of a prior, edit the `model`
block):

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
Stan_data_list <-  list(N                      = N,
                        n_tests                = n_tests,
                        test_is_binary         = c(1, 1, 1, 0),
                        n_categories_of_test   = c(2, 2, 2, 4),
                        y                      = y,
                        n_covariates_max       = n_covariates_max,
                        n_covariates_of_test   = n_covariates_of_test,
                        X                      = X_of_each_outcome,
                        ## (the priors' values)
                        prior_beta_mean        = prior_beta_mean,
                        prior_beta_sd          = prior_beta_sd,
                        prior_LKJ_eta          = prior_LKJ_eta,
                        prior_Dirichlet_alpha  = prior_Dirichlet_alpha,
                        C_raw_lower            = C_raw_lower,
                        C_raw_upper            = C_raw_upper,
                        ## (Phi_type 1: the exact normal CDF; the bounds of the log-scale tails)
                        Phi_type               = 1,
                        overflow_threshold     = 7.5,
                        underflow_threshold    = -7.5,
                        ## (NicoStan's reduce_sum data: N and chunk_size, which NicoStan sets for each layout)
                        chunk_size             = N)
```


**(b) The nuisance parameters.** The nuisance block `u_raw` (declared in the `parameters` block as `array[N]
row_vector[n_tests] u_raw;`) is given to NicoStan by name with `nuisance_params`; any number of nuisance blocks
can be given this way (e.g., `nuisance_params = c("z_intercept", "z_slope")`, see [Example
2](#example-2-several-nuisance-blocks-nuisance_params)).

NicoStan then samples the nuisance blocks with [diffusion-pathspace
HMC](https://doi.org/10.1016/j.spa.2012.12.001) (`diffusion_HMC = TRUE` in `$sample()`), and the main parameters
with standard HMC.

Since `u_raw` is the sliced argument of `reduce_sum_static`, its values are held by NicoStan inside the chunks
(`nico_reduce_sum`), i.e., off Stan's main autodiff tape.

Note that NicoStan never chooses the nuisance blocks itself, so they are always named, as here:

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


**(c) The initial values** of each burn-in chain are given as a list, by parameter name, on each parameter's own
(constrained) scale, as for Stan; a nuisance block is given in its own shape (here, `u_raw` as an N x n_tests
matrix, one row per subject), and a block left out starts at 0 on the unconstrained scale:

```r
n_chains_burnin <-  4
C_raw_initial <-  c(1.0, log(0.5), log(1.5))   ## the first cutpoint of anaemia, then the logs of its 2 gaps
initial_values_NicoStan <-  rep(list(list(u_raw     = matrix(0, nrow = N, ncol = n_tests),
                                          beta_vec  = rep(0, sum(n_covariates_of_test)),
                                          L_Omega   = diag(n_tests),
                                          C_raw     = C_raw_initial)), n_chains_burnin)
```


**(d) The optimal layouts of the burn-in and of the sampling.** NicoStan finds the optimal number of within-chain
threads (WCP) and the optimal $N_{\text{chunks}}$, separately for the burn-in and for the sampling (and the number
of sampling chains, so that every CPU is used), from the equations of *Cache-aware chunking dramatically improves
parallel scaling for HMC: Application to multivariate probit model* (in preparation), which use the L2 and L3
caches of the computer it runs on.

It then checks them by timing a few layouts around them, all within at most max(0.5 s, 2000 x the model's own
seconds per gradient call). Nothing is chosen by hand.

The layouts can be found on their own, before the fit (`kind_of_tape = "nico_math"`, since the model calls a nico_
function; `"Stan_math"` for a model that does not):

```r
layouts_NicoStan <-
      NicoStan::fn_compute_burnin_and_sampling_layouts_of_Stan_model_within_a_budget_of_lp_grad_calls(
            model_so_file    = model_NicoStan$init_object$model_so_file,
            Stan_data_list   = Stan_data_list,
            init_object      = model_NicoStan$init_object,
            kind_of_tape     = "nico_math",
            n_chains_burnin  = n_chains_burnin)
## (layouts_NicoStan$n_threads_WCP_burnin, $num_chunks_burnin, $n_chains_sampling, $n_threads_WCP_sampling,
##  $num_chunks_sampling, and $seconds_spent_choosing_the_burnin_and_sampling_layouts)
```


**(e) The fit**, with these layouts carried over into `$sample()`, and its summaries (the sampler: ChEES-R,
`burnin_algorithm = "CHESSR"`, of [Hoffman et al. (2021)](https://proceedings.mlr.press/v130/hoffman21a.html)):

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
fit_summary_NicoStan$get_summary_main()                   ## the main parameters
fit_summary_NicoStan$get_summary_transformed()            ## the transformed parameters (beta and the cutpoints)
fit_summary_NicoStan$get_summary_generated_quantities()   ## the generated quantities (the correlation matrix)
```

> **Note:** if you leave these out (or set them to `NULL`), NicoStan will find them for you, with the same
> function and within the same time limit, as long as your model uses `reduce_sum` or `reduce_sum_static` (with
> `N` and `chunk_size` in its data) or is a built-in BayesMVP model.


**The same model with BayesMVP (manual gradients).** BayesMVP's built-in MVOP model has the same likelihood and
priors, but its gradients are derived by hand (rather than by autodiff). With the BayesMVP extension installed,
the same five steps are:

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
## ---- (b) The nuisance parameters are built in (u_raw); sample_nuisance = TRUE samples them with
##      diffusion-pathspace HMC:
model_BayesMVP <-  BayesMVP::MVP_model$new(Model_type       = "MVOP",
                                           model_args_list  = model_args_list,
                                           sample_nuisance  = TRUE)
##
## ---- (c) The same initial values (BayesMVP's cutpoints are unconstrained by tanh, between the bounds):
C_unc_initial <-  atanh(2 * (C_raw_initial - C_raw_lower) / (C_raw_upper - C_raw_lower) - 1)
initial_values_BayesMVP <-  rep(list(list(u_raw                    = matrix(0, nrow = N, ncol = n_tests),
                                          beta                     = list(prior_beta_mean),
                                          Omega_unconstrained_vec  = list(rep(0, choose(n_tests, 2))),
                                          C_unc_vec                = list(C_unc_initial))), n_chains_burnin)
##
## ---- (d) The optimal layouts of the burn-in and of the sampling, found by BayesMVP from the same equations
##      (for a manual gradient), then checked by timing a few layouts around them:
burnin_layout_BayesMVP <-  BayesMVP::fn_compute_burnin_n_threads_WCP_and_num_chunks_for_BayesMVP_model(
                                 Model_type       = "MVOP",
                                 init_object      = model_BayesMVP$init_object,
                                 n_chains_burnin  = n_chains_burnin)
sampling_layout_BayesMVP <-
      BayesMVP::fn_compute_sampling_n_chains_n_threads_WCP_and_num_chunks_for_BayesMVP_model(
            Model_type   = "MVOP",
            init_object  = model_BayesMVP$init_object)
## (burnin_layout_BayesMVP$n_threads_WCP_burnin, $num_chunks_burnin, $seconds_spent_choosing;
##  sampling_layout_BayesMVP$n_chains_sampling, $n_threads_WCP_sampling, $num_chunks_sampling,
##  $seconds_spent_choosing)
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

The R script prints the posterior means of the correlation matrix from both fits, side by side (they should agree
within MCMC error).


### Example 4: network meta-analysis of test accuracy with multiple thresholds (MetaOrdDTA's Xu model)


[MetaOrdDTA](https://github.com/CerulloE1996/MetaOrdDTA)'s network meta-analysis (NMA) model of diagnostic test
accuracy, i.e., the NMA model of Nyaga et al.
([ANOVA model for network meta-analysis of diagnostic test accuracy data](https://doi.org/10.1177/0962280216669182))
with the fixed cutpoints of Xu et al.
([Evaluating accuracy of diagnostic tests with intermediate results in the absence of a gold standard](https://doi.org/10.1002/sim.5695)),
is a [Stan](https://mc-stan.org/) model which NicoStan fits unchanged.
The model and MetaOrdDTA are described in
[Ordinal regression for meta-analysis of test accuracy: a flexible approach for utilising all threshold data](https://arxiv.org/abs/2505.23393).
In this model, each test has one set of cutpoints per disease class, shared by every study.


Each study reports, for each of its tests, the number of non-diseased and of diseased patients above each of the
test's thresholds.
The thresholds a study did not report are coded -1: these are the "missing thresholds" the model is made for.


This example fits the model to simulated data: 4 ordinal anxiety screening tests (GAD-2, GAD-7, HADS and BAI) in
80 studies, each study with 1 to 3 of the tests (136 study-test pairs in all).
The data were simulated with MetaOrdDTA's own simulator (`R_fn_sim_NMA_data_with_covariates_varying()`), with a
design (which tests each study has, and the proportion of missing thresholds of each test) taken from a real NMA.


MetaOrdDTA prepares the data and the priors exactly as it gives them to Stan
(`MetaOrd_model$new(..., compile = FALSE)`: nothing is compiled or fitted by MetaOrdDTA here).
NicoStan then fits MetaOrdDTA's Stan model, `DTA_NMA_Nyaga_Xu_FIXEDthr.stan`
([on GitHub](https://github.com/CerulloE1996/MetaOrdDTA/blob/main/inst/stan_models/stan_models_NMA/DTA_NMA_Nyaga_Xu_FIXEDthr.stan)),
with its `#include` files from MetaOrdDTA's `stan_functions` folder.
The full R script,
`inst/examples/Run_network_meta_analysis_of_test_accuracy_Xu_fixed_cutpoints_with_MetaOrdDTA_and_NicoStan.R`,
reads the data and then fits the model in five steps: (a) the priors; (b) the nuisance parameters; (c) the
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
The priors are set in the `model` block of the Stan model, and their values are passed to it as data.
The values below are MetaOrdDTA's defaults for this model (with its default probit link), written out so that
they can be changed; to change the form of a prior, edit the `model` block.
The cutpoints have an induced Dirichlet prior
([Betancourt, ordinal regression](https://betanalpha.github.io/assets/case_studies/ordinal_regression.html)),
and the between-study correlation an LKJ prior
([Stan's `lkj_corr_cholesky`](https://mc-stan.org/docs/functions-reference/correlation_matrix_distributions.html)):

```r
## ---- The priors (MetaOrdDTA's defaults for this model):
## (beta_mu[t][c, ] ~ normal(prior_beta_mu_mean[t][c, ], prior_beta_mu_SD[t][c, ]))
prior_mean_of_each_beta_mu    <-  0
prior_SD_of_each_beta_mu      <-  1
## (beta_sigma[c] ~ normal(0, prior_beta_sigma_SD[c]): half-normal, as beta_sigma >= 0)
prior_SD_of_each_beta_sigma   <-  1
## (to_vector(beta_tau_raw) ~ normal(0, to_vector(prior_beta_tau_SD)): half-normal, as beta_tau_raw >= 0)
prior_SD_of_each_beta_tau     <-  1
## (beta_L_Omega ~ lkj_corr_cholesky(prior_beta_corr_LKJ), beta_L_Omega made from beta_corr)
prior_LKJ_eta_of_beta_corr    <-  2
## (Ind_Dir_ord_prob ~ induced_dirichlet_given_C(C_vec_t, prior_dirichlet_alpha[c][t, ], ...): every alpha 1)
prior_Dirichlet_alpha         <-  1
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
The model has two nuisance blocks, both of standard-normal between-study values (the z blocks):

```stan
parameters {
  matrix[2, n_studies] beta_eta_z;                 // each study's effect in each class
  array[2] vector[n_total_obs] beta_delta_z_vec;   // each study-test pair's effect in each class
  ...
}
```

with `n_total_obs` the number of study-test pairs.
They are given to NicoStan by name, with `nuisance_params = c("beta_eta_z", "beta_delta_z_vec")`.
The raw cutpoints `C_raw_vec` are a main parameter here, not a nuisance one: in this fixed-cutpoint model they do
not grow with the number of studies (the cutpoints are nuisance only in MetaOrdDTA's random-cutpoint models).
NicoStan samples the two nuisance blocks with diffusion-pathspace HMC
([Beskos et al., 2013](https://doi.org/10.1016/j.spa.2012.12.001); `diffusion_HMC = TRUE` in `$sample()`), and the
main parameters with standard HMC.


NicoStan writes the version of the model with these two blocks held by NicoStan
(`DTA_NMA_Nyaga_Xu_FIXEDthr_nico_reduce_sum.stan`, next to the model; the rest of the model is unchanged), and
the `#include` files are found with `stanc_args`:

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
                                            nuisance_params       = c("beta_eta_z", "beta_delta_z_vec"),
                                            sample_nuisance       = TRUE)
```

Here this gives 432 nuisance parameters (2 x 80 studies + 2 x 136 study-test pairs) and 241 main parameters
(`beta_mu`: 8; `C_raw_vec`: 2 x 111 cutpoints, i.e., 6 + 21 + 21 + 63 thresholds of the 4 tests; `beta_sigma`: 2;
`beta_tau_raw`: 8; `beta_corr`: 1).


**(c) The initial values.**
The initial values of the burn-in chains are given by name: the nuisance blocks at 0 (i.e., every study at the
average), the raw cutpoints at 0, and the between-study SDs and the correlation small:

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
This model has no [`reduce_sum()`](https://mc-stan.org/docs/stan-users-guide/parallelization.html) (its N is the
number of studies, which is small), so every chain runs on one thread, without chunks, and NicoStan runs one chain
per CPU in the sampling.
For a model whose likelihood is in `reduce_sum()` chunks, see
[Example 3](#example-3-the-multivariate-ordered-probit-model-with-the-fused-ghk-function-and-bayesmvp), where
NicoStan finds the optimal number of within-chain threads (WCP) and the optimal $N_{\text{chunks}}$ of the burn-in
and of the sampling by itself, from the L2 and L3 caches of the computer.


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


MetaOrdDTA also has its own NicoStan option, `model_MetaOrdDTA$sample(algorithm = "NicoStan", NicoStan_args =
list(...))`, which names the nuisance blocks of its own models by itself and fits the nico::math (AVX) version of
the model with the fused nico_ functions (`DTA_NMA_Nyaga_Xu_FIXEDthr_NicoStan_fused_AVX.stan`).
This example calls NicoStan directly instead, so that every step is shown.

### The example models


Every example below is in [inst/examples/](https://github.com/CerulloE1996/NicoStan/tree/main/inst/examples);
the Stan models are in [inst/examples/models/](https://github.com/CerulloE1996/NicoStan/tree/main/inst/examples/models).
Each model has a [Stan](https://mc-stan.org) file for Stan::math and, where it exists, a nico::math (AVX) file
(the `_avx.stan` file: the same model, with each Stan function swapped for its nico_ copy from NicoSIMD).
Each R script fits its model at the three sample sizes given at its top (`N_values`), which can be edited.

In the scripts of the eight models with an `_avx.stan` file, a nuisance block (where the model has one) is the first
parameter declared, and NicoStan finds it by itself (`sample_nuisance = TRUE`);
the other three scripts name their nuisance blocks with `nuisance_params`.


| Model | Nuisance block(s) | Stan::math | nico::math (AVX) | nico::math (AVX) with fused nico_ functions | R script (N) |
|---|---|---|---|---|---|
| Logistic regression with random intercepts | `z[J]` | [random_intercepts.stan](inst/examples/random_intercepts.stan) | - | - | [random_intercepts.R](inst/examples/random_intercepts.R) (N = 200, J = 20) |
| Stochastic volatility (discrete-time AR(1)) | `h_std[T]` | [EASY_Stochastic_volatility.stan](inst/examples/models/EASY_Stochastic_volatility.stan) | [EASY_Stochastic_volatility_avx.stan](inst/examples/models/EASY_Stochastic_volatility_avx.stan) (`nico_exp`) | - | [Run_EASY_Stochastic_volatility.R](inst/examples/Run_EASY_Stochastic_volatility.R) (T = 250, 1000, 4000) |
| Shared-frailty Cox survival | `log_frailty_raw[J]` | [Frailty_surv_shared_frailty_Cox.stan](inst/examples/models/Frailty_surv_shared_frailty_Cox.stan) | [Frailty_surv_shared_frailty_Cox_avx.stan](inst/examples/models/Frailty_surv_shared_frailty_Cox_avx.stan) (`nico_exp`) | - | [Run_Frailty_surv_shared_Cox.R](inst/examples/Run_Frailty_surv_shared_Cox.R) (N = 100, 250, 500) |
| Latent diffusion survival | `w[M]` | [Latent_diffusion_survival.stan](inst/examples/models/Latent_diffusion_survival.stan) | [Latent_diffusion_survival_avx.stan](inst/examples/models/Latent_diffusion_survival_avx.stan) (`nico_log`) | - | [Run_Latent_diffusion_survival.R](inst/examples/Run_Latent_diffusion_survival.R) (N = 100, 200, 800) |
| Joint longitudinal and survival | `b_raw[2, N_subj]` | [Joint_longitudinal_surv.stan](inst/examples/models/Joint_longitudinal_surv.stan) | [Joint_longitudinal_surv_avx.stan](inst/examples/models/Joint_longitudinal_surv_avx.stan) (`nico_exp`) | - | [Run_joint_longitudinal.R](inst/examples/Run_joint_longitudinal.R) (N = 25, 75, 200) |
| Hierarchical logistic regression | `beta_raw[K, 2]` | [hierarchical_logistic_regression.stan](inst/examples/models/hierarchical_logistic_regression.stan) | [hierarchical_logistic_regression_avx.stan](inst/examples/models/hierarchical_logistic_regression_avx.stan) (`nico_log1p_exp`) | - | [Run_hierarchical_logistic_regression.R](inst/examples/Run_hierarchical_logistic_regression.R) (N = 300, 1200, 4000) |
| Weibull survival | none | [standard_surv_Weibull.stan](inst/examples/models/standard_surv_Weibull.stan) | [standard_surv_Weibull_avx.stan](inst/examples/models/standard_surv_Weibull_avx.stan) (`nico_exp`) | - | [Run_standard_surv_Weibull.R](inst/examples/Run_standard_surv_Weibull.R) (N = 100, 250, 500) |
| Robust Student-t regression | none | [robust_regression_t4.stan](inst/examples/models/robust_regression_t4.stan) | [robust_regression_t4_avx.stan](inst/examples/models/robust_regression_t4_avx.stan) (`nico_log1p`) | - | [Run_robust_t4_regression.R](inst/examples/Run_robust_t4_regression.R) (N = 250, 1000, 3000) |
| Gaussian process regression (marginal) | none (the latent GP is integrated out) | [Gaussian_process.stan](inst/examples/models/Gaussian_process.stan) | [Gaussian_process_avx.stan](inst/examples/models/Gaussian_process_avx.stan) (`nico_exp`) | - | [Run_Gaussian_process.R](inst/examples/Run_Gaussian_process.R) (N = 40, 80, 160) |
| One-class multivariate ordered probit (the real NHANES data of the [hexbin](https://cran.r-project.org/package=hexbin) package) | `u_raw[N]` (row vectors) | - | - | `one_class_MVOP_with_the_fused_GHK_log_likelihood.stan` (NicoSIMD's fused GHK function) | `Run_one_class_MVOP_with_the_fused_GHK_log_likelihood_and_BayesMVP.R` (N = 9575) |
| Network meta-analysis of test accuracy, Xu model with fixed cutpoints ([MetaOrdDTA](https://github.com/CerulloE1996/MetaOrdDTA)) | `beta_eta_z`, `beta_delta_z_vec` | [DTA_NMA_Nyaga_Xu_FIXEDthr.stan](https://github.com/CerulloE1996/MetaOrdDTA/blob/main/inst/stan_models/stan_models_NMA/DTA_NMA_Nyaga_Xu_FIXEDthr.stan) (in MetaOrdDTA) | - | [DTA_NMA_Nyaga_Xu_FIXEDthr_NicoStan_fused_AVX.stan](https://github.com/CerulloE1996/MetaOrdDTA/blob/main/inst/stan_models/stan_models_NMA/DTA_NMA_Nyaga_Xu_FIXEDthr_NicoStan_fused_AVX.stan) (in MetaOrdDTA) | `Run_network_meta_analysis_of_test_accuracy_Xu_fixed_cutpoints_with_MetaOrdDTA_and_NicoStan.R` (simulated data, 80 studies) |


The one-class multivariate ordered probit has no Stan::math version, because Stan has no fused GHK function;
its R script also fits the same model with BayesMVP's built-in MVOP model (manual gradients), with the same priors.
The network meta-analysis script fits the Stan::math version.

The model definitions, priors and nuisance blocks of each are described in
[MODEL_NOTES.md](inst/examples/MODEL_NOTES.md).
The full code of every example model and R script (each file a click away, with a Copy button) is on
[one page of the website](https://cerulloe1996.github.io/NicoStan/example_models_full_code.html).


### The BayesMVP models


With the [BayesMVP](https://github.com/CerulloE1996/BayesMVP) extension installed, each of its built-in
(manual-gradient) models has an R script, which fits it at N = 48, 150 and 500 (`N_values`, which can be edited):
[Run_MVP.R](inst/examples/Run_MVP.R), [Run_2LC_MVP.R](inst/examples/Run_2LC_MVP.R),
[Run_MVOP.R](inst/examples/Run_MVOP.R), [Run_2LC_MVOP.R](inst/examples/Run_2LC_MVOP.R) and
[Run_latent_trait.R](inst/examples/Run_latent_trait.R).

Furthermore, the four-class LC-MVOP (4LC-MVOP), which extends the LC-MVOP to two conditions at once
(i.e., its latent classes are the 2 x 2 combinations of the two conditions),
is in [four_class/](https://github.com/CerulloE1996/NicoStan/tree/main/inst/examples/four_class), with synthetic data
at N = 24, 48 and 96 ([Run_4LC_MVOP.R](inst/examples/Run_4LC_MVOP.R)).
Its Stan models are in [four_class/models/](https://github.com/CerulloE1996/NicoStan/tree/main/inst/examples/four_class/models):
the Stan::math version; the nico::math (AVX) version (`four_class_avx.stan`);
and the nico::math (AVX) version with the fused nico_ functions (`four_class_avx_fused.stan`, e.g.
`nico_normal_binary`, `nico_normal_interval` and `nico_truncated_normal_quantile_and_log_mass`).

The BayesMVP R package also has two example data sets:
binary data for the LC-MVP, simulated from real, publicly available COVID-19 data;
and ordinal data for the LC-MVOP, based on a real South African depression data set
(with the PHQ-9 and the CES-D-10 as the two ordinal index tests, and the MINI as the binary reference test).


<!-- ------------------------------------------------------------------------------------------------------------------------------- -->
## [BayesMVP](https://github.com/CerulloE1996/BayesMVP): multivariate probit models
<!-- ------------------------------------------------------------------------------------------------------------------------------- -->


The [BayesMVP R package extension](https://github.com/CerulloE1996/BayesMVP) to NicoStan provides specialised
implementations, with manually derived likelihood gradients hand-coded in C++, of some particularly difficult-to-sample
models; more specifically, it provides the following models:

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
matrix per latent class).

Furthermore, any of the correlations can be fixed to known values, via the `known_values_indicator_list`
argument (which indicates which correlations are known) and the `known_values_list` argument (their values).


**Chunks and within-chain threads.**
BayesMVP's models work through the subjects in chunks: the forward pass and the (manual) gradients of each chunk
are finished before the next chunk starts, in one workspace which is reused for every chunk
(one workspace per thread, with within-chain parallelisation, i.e., WCP),
so that, with the number of chunks chosen as below, each chunk's buffers stay in the CPU's L2 cache.

Similarly to the Stan models, the number of chunks and of within-chain threads of the burn-in and of the sampling
are found by BayesMVP itself, from the L2 and L3 caches of the computer, followed by a short timed check of the
layouts around them
(`BayesMVP::fn_compute_burnin_n_threads_WCP_and_num_chunks_for_BayesMVP_model()` and
`BayesMVP::fn_compute_sampling_n_chains_n_threads_WCP_and_num_chunks_for_BayesMVP_model()`, see
[Example 3](#example-3-the-multivariate-ordered-probit-model-with-the-fused-ghk-function-and-bayesmvp)),
or by `$sample()` when they are left out.

The log density and its gradient do not depend on the number of chunks: in the evaluation the sampler uses,
the largest relative difference from one chunk was 1.1e-14 (LC-MVP and LC-MVOP, with and without missing results,
with nico::math (AVX-512) and with Stan::math, at 2 to 64 chunks against 1 chunk; the check is
`inst/BayesMVP/tests/check_that_the_BayesMVP_manual_gradient_models_give_the_same_log_density_and_gradient_at_every_number_of_chunks.R`
in BayesMVP).


The ordinal latent class model is described in
[Meta-analysis of dichotomous and ordinal tests with an imperfect gold standard](https://doi.org/10.1002/jrsm.1567)
(Cerullo et al., 2022).
Furthermore, in
[Latent class multivariate probit and latent trait models for evaluating test accuracy without a gold standard: A simulation study](https://arxiv.org/abs/2509.18489)
(Cerullo et al., 2025), we compared the LC-MVP and latent trait models in a comprehensive simulation study,
in which the models were fitted using [BayesMVP](https://github.com/CerulloE1996/BayesMVP).


<!-- The **4LC-MVOP** model handles two related latent conditions. -->
<!-- It has a Stan implementation and a partial-manual-gradient/fused-function version. -->
<!-- Its partial-manual-gradient implementation runs through Stan/NicoStan, -->
<!-- whilst the five models above also have native BayesMVP implementations.  -->
<!-- Development and applications of 4LC-MVOP are covered in separate papers,  -->
<!-- whilst the model also provides a larger example for the general NicoStan sampler. -->


### Automatic test (column) reordering


For the latent class models (LC-MVP and LC-MVOP), BayesMVP chooses the order of the tests
(i.e., the columns of `y`) automatically, before the main burn-in (`reorder_cols_MVP = TRUE` by default;
`reorder_cols_MVP = FALSE` keeps the original order).
The same is done, by default, for a Stan model whose data has a `latent_test_order` field
(e.g. the four-class LC-MVOP).

The model itself does not depend on the order of the tests;
however, the unconstrained parameters which [HMC](https://arxiv.org/abs/1206.1901) samples do - and hence so does
the efficiency of HMC.

- A short pre-burn-in gives rough estimates of the correlations, intercepts and (for ordinal tests) cutpoints;
the most nearly deterministic test is then placed last,
and the other tests follow the greedy correlation order (the most strongly correlated tests first).
- This is because, in the [GHK](https://en.wikipedia.org/wiki/GHK_algorithm) construction, each test's truncation
bounds depend on the tests placed before it,
and a nearly deterministic test has steep tail values which would otherwise propagate to every later test;
the order also sets the order of the correlation parameterisation
([Pinkney, 2024](https://arxiv.org/abs/2405.07286)).
- The rule can be chosen with `test_order_rule_for_reorder_cols_MVP`; e.g. the old greedy-only rule is
`test_order_rule_for_reorder_cols_MVP = "greedy_correlation_order"`.
- On our binary LC-MVP model (N = 10,000), the slower order of the old greedy-only rule needed 1.60x the
gradient evaluations to reach the target min ESS (0.59x the min ESS per 1000 gradient evaluations);
on our ordinal LC-MVOP model (N = 5,000), placing the binary test last needed 0.67-0.75x the gradient evaluations
of the input order.
- The summaries and traces are returned in the original test order.

See the [BayesMVP README](https://github.com/CerulloE1996/BayesMVP#automatic-test-column-reordering)
for more details.


<!-- ------------------------------------------------------------------------------------------------------------------------------- -->
## Benchmarks
<!-- ------------------------------------------------------------------------------------------------------------------------------- -->


In our initial tests, NicoStan, combined with the [BayesMVP extension](https://github.com/CerulloE1996/BayesMVP),
was **over 150x faster than [Stan](https://mc-stan.org/)**, and around **50x more efficient than [Mplus](https://www.statmodel.com/)**.
These numbers are for the latent class multivariate probit model (LC-MVP),
using an N = 10,000 simulated binary dataset with 6 binary tests in total (hence a total of 60,000 observations),
which is based on real, publicly available COVID-19 data.
These models (LC-MVP) are used in both human and veterinary medicine for estimating diagnostic and screening test accuracy
without a perfect gold standard.


For our ordinal example dataset (using the LC-MVOP model),
which is based on a real dataset on three tests to screen for and/or diagnose depression
(more specifically, the MINI as the imperfect gold standard, and the PHQ-9 and CES-D-10 as the two ordinal tests),
for N = 5,000 (hence 15,000 observations in total, since there are 3 tests), we found that NicoStan with BayesMVP
(using the LC-MVOP model) was **over 50x more efficient than Stan** and **over 1000x more efficient than Mplus**.
The latter is because, despite being very efficient for the binary LC-MVP,
the Gibbs-based algorithm of Mplus struggles greatly with ordinal outcomes,
and Mplus does not let you fit ordinal outcomes with more than 10 categories;
hence, we had to group categories together just to attempt to measure its efficiency.
Note that this is not ideal: we would never recommend grouping categories in real-life data analysis,
nor would we suggest assuming continuity just because there are a lot of categories
(especially for single studies and for latent class models).


Furthermore, even when using NicoStan **without** BayesMVP
(i.e., using the `.stan` model file directly, without the hand-coded C++ gradients and the AVX functions),
we still found that NicoStan was **over 10x more efficient than Stan** (via [cmdstanr](https://mc-stan.org/cmdstanr/)),
for both the LC-MVP and the LC-MVOP examples described above.
These fits use Stan::math (i.e., Stan's own C++ maths library), so the gains come from the sampler and its burn-in adaptation.
The nico::math (AVX) versions, i.e., the same models with the AVX2 / AVX-512 nico_ functions of NicoSIMD
(which BayesMVP uses), give a further route to speeding things up.


**More detailed and comprehensive benchmarks are coming soon.** The comparisons will cover:


- Stan (NUTS, via cmdstanr),
fitted using the default NUTS-HMC algorithm ([Hoffman and Gelman, 2014](https://www.jmlr.org/papers/v15/hoffman14a.html)).
- NicoStan, using the same Stan model with Stan::math.
- NicoStan, using the same Stan model with nico::math (AVX2 and/or AVX-512, on CPUs that support them).


All of these comparisons use the same data, priors and initial values.
The results will report sampling efficiency in terms of effective sample size (ESS); more specifically:
(i) sampling ESS/sec;
(ii) the time taken to reach a minimum ESS of 100, 1,000 and 10,000; and
(iii) sampling ESS per gradient evaluation (ESS/grad).
Additionally, we will report the warm-up (Stan) and burn-in (NicoStan) costs,
as well as convergence diagnostics (e.g., R-hat and nested R-hat [nR-hat]; [Margossian et al., 2024](https://arxiv.org/abs/2110.13017v6)),
alongside the computational settings.


### Stan vs NicoStan (diffusion HMC and standard HMC) on eight general models


We compared NicoStan with Stan (i.e., NUTS, via cmdstanr) in terms of wall-clock time on eight models,
using the same Stan model files, data and initial values
(from the NicoStan paper, *NicoStan: between-chain adaptive hybrid diffusion-pathspace HMC for general Stan models*, in preparation).
Five of these models have a nuisance block:

- a stochastic volatility model (`stochastic_volatility`, T = 1,000; SV;
example model file [EASY_Stochastic_volatility.stan](inst/examples/models/EASY_Stochastic_volatility.stan));
- a shared-frailty Cox model (`shared_frailty_Cox_survival`, N = 250; COX;
example model file [Frailty_surv_shared_frailty_Cox.stan](inst/examples/models/Frailty_surv_shared_frailty_Cox.stan));
- a latent diffusion survival model (`latent_diffusion_survival`, N = 200; LDS;
example model file [Latent_diffusion_survival.stan](inst/examples/models/Latent_diffusion_survival.stan));
- a joint longitudinal and survival model (`joint_longitudinal_survival`, N = 75; JLS;
example model file [Joint_longitudinal_surv.stan](inst/examples/models/Joint_longitudinal_surv.stan));
- a hierarchical logistic regression (`hierarchical_logistic_regression`, N = 1,200; HLR;
example model file [hierarchical_logistic_regression.stan](inst/examples/models/hierarchical_logistic_regression.stan)).

The other three do not:

- a Gaussian process regression (`Gaussian_process_regression`, N = 80; GP;
example model file [Gaussian_process.stan](inst/examples/models/Gaussian_process.stan));
- a Weibull survival model (`Weibull_survival`, N = 250; WEI;
example model file [standard_surv_Weibull.stan](inst/examples/models/standard_surv_Weibull.stan));
- a robust regression with t₄ errors (`robust_Student_t_regression`, N = 1,000; RT4;
example model file [robust_regression_t4.stan](inst/examples/models/robust_regression_t4.stan)).


For the models with a nuisance block, NicoStan sampled the nuisance block using either diffusion HMC
(i.e., hybrid HMC; NicoStan's default; see [Models with nuisance parameters](#models-with-nuisance-parameters-diffusion-pathspace-hmc))
or standard HMC;
for the models without one, NicoStan sampled all of the parameters using standard HMC.
NicoStan used its default trajectory-length criterion, ChEES-R
(see [Efficient burnin algorithms](#efficient-burnin-algorithms-snaper-hmc-and-chees-r-hmc)).


Both samplers used 4 chains for the warm-up (Stan) or burn-in (NicoStan), followed by 180 sampling chains of 100 iterations each
(45 sampling chains starting from the final state of each warm-up / burn-in chain),
one fit at a time on a 96-core (192-thread) machine, with 5 seeds per configuration.
Stan ran in two stages: the warm-up with 4 chains, then the sampling with 180 chains, each using the step size,
metric and final state of its warm-up chain (adaptation off); adapt_delta = 0.8, maximum tree depth 10,
and 250, 500 or 1,000 warm-up iterations (vs. NicoStan's 125, 250 or 500 burn-in iterations,
since we found Stan's warm-up to be unreliable with only 125 iterations).


For COX, both samplers evaluated the log-likelihood in 3 chunks
(using [reduce_sum_static](https://mc-stan.org/docs/stan-users-guide/parallelization.html)),
as given by NicoStan's cache-based chunking rule (`fn_compute_initial_N_chunks()`) for 180 chains;
for the other seven models, the rule gave 1 chunk.


The table below shows the (projected) time needed to reach a minimum ESS of 1,000
(the measured warm-up or burn-in time, plus 1,000 times the measured sampling time divided by the minimum ESS;
in seconds) / the minimum ESS per 1,000 sampling gradient evaluations
(geometric means over 5 seeds; the minimum is over the bulk ESS, the tail ESS and the ESS of the squared deviations,
i.e., of the SD, of the main parameters).


Stan's times are those of its CmdStan processes (from the first chain's start to the last chain's end;
for the sampling, this includes the staggered launch of its 180 processes),
and NicoStan's those of its burn-in and sampling;
every timed phase starts its clock at its first iteration (0 untimed iterations),
and, for both samplers, the times exclude the compilation and loading of the model.
**Bold**: the best of the row for each endpoint (and any within 1% of it).


| Model | Stan warm-up / NicoStan burn-in | Stan (NUTS) | NicoStan, ChEES-R, diffusion HMC | NicoStan, ChEES-R, standard HMC |
|---|---|---|---|---|
| SV | 250 / 125 | 2.08 / 7.0 | **0.52** / **11.4** | 0.55 / 7.8 |
|  | 500 / 250 | 2.90 / 6.0 | **0.72** / **11.3** | 0.76 / 8.7 |
|  | 1000 / 500 | 3.93 / 7.9 | **1.22** / **14.3** | 1.27 / 8.5 |
| COX | 250 / 125 | 10.5 / 23.7 | **1.58** / **28.8** | 1.89 / 17.9 |
|  | 500 / 250 | 14.4 / 23.3 | **2.77** / 29.9 | **2.76** / **32.1** |
|  | 1000 / 500 | 22.3 / 23.6 | **4.23** / **39.9** | 5.06 / 26.8 |
| LDS | 250 / 125 | 1.17 / 2.8 | 0.30 / 7.2 | **0.30** / **7.6** |
|  | 500 / 250 | 1.47 / 2.8 | **0.50** / **7.3** | 0.51 / 6.4 |
|  | 1000 / 500 | 1.91 / 2.8 | **0.89** / **8.7** | **0.89** / 7.3 |
| JLS | 250 / 125 | 13.1 / 6.7 | **1.07** / 8.6 | 1.11 / **9.0** |
|  | 500 / 250 | 15.3 / 6.2 | **1.65** / **10.1** | 1.67 / 8.0 |
|  | 1000 / 500 | 19.7 / 6.1 | **2.69** / **7.2** | **2.71** / 5.9 |
| HLR | 250 / 125 | 1.29 / 8.8 | **0.28** / **15.0** | 0.29 / 12.4 |
|  | 500 / 250 | 1.60 / 8.7 | **0.50** / **15.8** | **0.51** / 11.8 |
|  | 1000 / 500 | 2.45 / 9.5 | **0.92** / **15.0** | 0.93 / 11.9 |
| GP | 250 / 125 | 0.89 / 86.5 | (no nuisance block) | **0.38** / **101.9** |
|  | 500 / 250 | 1.39 / 86.3 | (no nuisance block) | **0.70** / **116.4** |
|  | 1000 / 500 | 2.29 / 78.9 | (no nuisance block) | **1.32** / **107.5** |
| WEI | 250 / 125 | 0.37 / 78.9 | (no nuisance block) | **0.16** / **108.9** |
|  | 500 / 250 | 0.48 / 79.7 | (no nuisance block) | **0.28** / **132.2** |
|  | 1000 / 500 | 0.58 / 86.6 | (no nuisance block) | **0.52** / **116.2** |
| RT4 | 250 / 125 | 0.46 / 114.3 | (no nuisance block) | **0.15** / **171.2** |
|  | 500 / 250 | 0.68 / 110.3 | (no nuisance block) | **0.26** / **224.1** |
|  | 1000 / 500 | 1.07 / 114.0 | (no nuisance block) | **0.48** / **232.7** |


NicoStan reached the target faster than Stan for every model and warm-up / burn-in length.


For the fastest configuration of each sampler (for every model, Stan's 250 warm-up iterations and NicoStan's
125 burn-in iterations), NicoStan was between 2.3x (GP and WEI) and 12.3x (JLS) faster
(3.0x for RT4, 3.9x for LDS, 4.0x for SV, 4.6x for HLR and 6.6x for COX).
Its highest minimum ESS per gradient was between 1.3x (GP) and 3.1x (LDS) that of Stan.


Within NicoStan, diffusion HMC reached the target faster than standard HMC in 12 of the 15 cells with a nuisance
block (i.e., combinations of model and burn-in length; for LDS with 500 burn-in iterations, the two are level at 0.89 s),
and it had a higher minimum ESS per gradient in 12 of them.


Note that:

- We counted Stan's gradient evaluations as its number of leapfrog steps,
and NicoStan's as the expected number of leapfrog steps of its (jittered) trajectories,
plus one for every iteration in which the sampler evaluates the gradient at the start of the trajectory
(i.e., during the burn-in, for standard HMC, and for the models without a nuisance block).
- The nested R-hat ([Margossian et al., 2024](https://arxiv.org/abs/2110.13017v6); with the 4 warm-up / burn-in chains as the
superchains) exceeded 1.01 in 51 of the 510 fits of this benchmark (Stan's 120 fits and NicoStan's 390, which also include
fits with a second trajectory-length criterion that NicoStan no longer offers), 46 of which were for LDS
(45 of NicoStan's 60 LDS fits, with a median of 1.018 and a maximum of 1.044, and 1 of Stan's 15, with 1.012);
for the other models, it was at most 1.038 for NicoStan (COX) and 1.003 for Stan.
- Over the same fits, Stan had 77 divergent transitions (in 9 of its 120 fits: 8 for HLR and 1 for LDS),
and NicoStan had 22 (in 17 of its 390 fits: 13 for LDS and 4 for SV), out of 18,000 sampling draws per fit.


### Trajectory-length criteria within NicoStan


We also compared NicoStan's trajectory-length criteria (the `burnin_algorithm` option strings in brackets):
ChEES ([Hoffman et al., 2021](https://proceedings.mlr.press/v130/hoffman21a.html); `"ChEES"`),
ChEES-R (NicoStan's default; `"CHESSR"`),
SNAPER ([Sountsov and Hoffman, 2022](https://arxiv.org/abs/2110.11576v3); `"SNAPER"`),
the ESJD rate ([Pasarica and Gelman, 2010](https://www.jstor.org/stable/24308995); `"ESJD"`),
and the two blends that we proposed:
ESJD x ChEES-R = sqrt(C_ESJD * C_ChEESR) (`"ESJD_CHESSR"`) and
ESJD x SNAPER = sqrt(C_ESJD * C_SNAPER) (`"ESJD_SNAPER"`).


Maximising a geometric mean is the same as maximising the average of the logarithms of the criteria;
in other words, each criterion contributes equally to the choice of the trajectory length (tau), whatever its scale.


NicoStan also offers two three-way blends of ESJD, ChEES-R and SNAPER:
the equally weighted geometric mean (`"ESJD_CHESSR_SNAPER"`)
and the weighted one with weights 0.50 / 0.25 / 0.25, i.e., exp(0.5 log C_ESJD + 0.25 log C_ChEESR + 0.25 log C_SNAPER)
(`"ESJD_w50_CHESSR_w25_SNAPER_w25"`).
Both are in the preliminary tables below.


The benchmark used the models of the previous section at other N:
each model at a first N (SV with T = 500; LDS, HLR, RT4 and WEI with N = 500; GP and JLS with N = 50) and at 10 x N,
with burn-ins of 125 and 250 iterations and 10 seeds per cell (14 cells per N).


For each cell, a criterion's loss is its geometric mean over the seeds relative to the best of the eight criteria in that cell
(1.00 = the best; for the minimum ESS per gradient, the best's value divided by the criterion's, so that above 1 is always worse),
and its mean loss is the geometric mean of its losses over the cells.
"Cells OK" counts the cells in which the criterion was the best, or within noise of the best with the 95% interval of
its seed-matched ratio within plus or minus 25%.
**Bold**: the best of a column (and any mean or worst loss within 1% of it).


**Preliminary results, first N of each model (14 cells):**

| Criterion | Time to min ESS 1,000: mean loss | worst loss | cells OK | Time to min ESS 10,000: mean loss | Gradients to min ESS 1,000: mean loss | Min ESS per 1,000 gradients: mean loss | worst loss | cells OK |
|---|---|---|---|---|---|---|---|---|
| ESJD | 1.073 | 1.30 | **11** | 1.186 | 1.178 | 1.214 | 1.87 | 6 |
| ChEES | 1.107 | 1.76 | 5 | 1.206 | 1.379 | 1.483 | 2.43 | 3 |
| ChEES-R (default) | 1.086 | 1.36 | 7 | 1.179 | 1.171 | 1.211 | 1.77 | 5 |
| SNAPER | **1.045** | **1.19** | 10 | **1.105** | **1.104** | **1.136** | **1.39** | 6 |
| ESJD x ChEES-R | 1.067 | 1.30 | 10 | 1.151 | 1.123 | 1.154 | 1.60 | 6 |
| ESJD x SNAPER | 1.068 | 1.39 | 5 | 1.131 | **1.106** | **1.128** | 1.88 | **10** |
| ESJD x ChEES-R x SNAPER (equal weights) | 1.242 | 1.43 | 1 | 1.230 | 1.120 | 1.145 | 1.70 | 7 |
| ESJD x ChEES-R x SNAPER (0.50 / 0.25 / 0.25) | 1.242 | 1.45 | 3 | 1.235 | 1.130 | 1.160 | 1.75 | 8 |

**Preliminary results, 10 x N (14 cells):**

| Criterion | Time to min ESS 1,000: mean loss | worst loss | cells OK | Time to min ESS 10,000: mean loss | Gradients to min ESS 1,000: mean loss | Min ESS per 1,000 gradients: mean loss | worst loss | cells OK |
|---|---|---|---|---|---|---|---|---|
| ESJD | 1.119 | 2.32 | 9 | 1.161 | 1.135 | 1.167 | 2.69 | 8 |
| ChEES | 1.233 | 1.74 | 3 | 1.379 | 1.361 | 1.489 | 2.72 | 2 |
| ChEES-R (default) | **1.076** | 1.21 | 9 | 1.125 | 1.095 | 1.130 | 1.35 | 8 |
| SNAPER | 1.090 | 1.43 | 8 | 1.144 | 1.117 | 1.159 | 1.47 | 4 |
| ESJD x ChEES-R | **1.069** | 1.21 | 10 | 1.110 | 1.080 | 1.107 | 1.31 | 11 |
| ESJD x SNAPER | 1.082 | 1.48 | **12** | 1.110 | 1.091 | 1.109 | 1.52 | 9 |
| ESJD x ChEES-R x SNAPER (equal weights) | 1.096 | 1.21 | 8 | 1.093 | 1.065 | 1.088 | **1.25** | 10 |
| ESJD x ChEES-R x SNAPER (0.50 / 0.25 / 0.25) | 1.084 | **1.16** | 7 | **1.072** | **1.045** | **1.062** | **1.25** | **12** |


At the first N of each model, SNAPER had the smallest mean loss for the time to a minimum ESS of 1,000 (1.045)
and of 10,000 (1.105), and for the gradient evaluations to a minimum ESS of 1,000 (1.104, with ESJD x SNAPER within 1%: 1.106);
ESJD x SNAPER had the smallest mean loss for the minimum ESS per gradient (1.128, with SNAPER within 1%: 1.136).
The two three-way blends had the largest mean losses for the time to a minimum ESS of 1,000 (1.242 for both)
and of 10,000 (1.230 and 1.235).


At 10 x N, ESJD x ChEES-R had the smallest mean loss for the time to a minimum ESS of 1,000
(1.069, with ChEES-R within 1%: 1.076); the weighted three-way blend (0.50 / 0.25 / 0.25) had the smallest mean loss for
the time to a minimum ESS of 10,000 (1.072), for the gradient evaluations to a minimum ESS of 1,000 (1.045)
and for the minimum ESS per gradient (1.062), the smallest worst losses for the time to a minimum ESS of 1,000 (1.16)
and for the minimum ESS per gradient (1.25, tied with the equally weighted blend),
and the most cells OK for the minimum ESS per gradient (12 of 14).


In short, no single criterion was best everywhere: SNAPER and ESJD x SNAPER led at the first N of each model,
whilst the weighted three-way blend led for three of the four endpoints at 10 x N;
ChEES had the largest mean loss for every endpoint at 10 x N.


<!-- ------------------------------------------------------------------------------------------------------------------------------- -->
## Models with nuisance parameters (diffusion-pathspace HMC)
<!-- ------------------------------------------------------------------------------------------------------------------------------- -->


NicoStan is particularly aimed at models with a large block of latent / nuisance parameters for which analytic marginalisation is unavailable,
or numerical marginalisation is too expensive.


Techniques such as non-centring can improve the posterior geometry of some of these models;
however, they do not remove the nuisance block: the latent variables still have to be sampled.


Such blocks occur in many commonly used models, e.g.:

- **MVP, LC-MVP, MVOP and LC-MVOP:** latent Gaussian / auxiliary variables for correlated binary and / or ordinal outcomes.
The augmented state grows with the number of individuals and outcomes;
evaluating the marginal likelihood instead involves multivariate Gaussian rectangle probabilities.
- **Generalised linear mixed models:** Gaussian random intercepts / slopes for binary, ordinal or count outcomes,
particularly when there are many subjects / groups or crossed random effects.
- **Generalised additive mixed models:** Gaussian spline coefficients and random effects,
with non-Gaussian observations leaving a non-conjugate posterior over those coefficients.
- **Gaussian processes with non-Gaussian likelihoods:** latent function values for binary, ordinal or count outcomes.
- **Spatial and spatio-temporal models:** Gaussian spatial fields, temporal effects and their interactions,
e.g. in binomial / Poisson disease-mapping models.
- **Log-Gaussian Cox processes:** a Gaussian latent log-intensity field, often represented on a large spatial mesh / grid.
- **Stochastic volatility:** a latent log-volatility path in discrete-time models, or a latent volatility diffusion in continuous time.
E.g., path augmentation for a partially observed volatility diffusion introduces latent points between the observations;
the discretisation then controls both the approximation error and the size of the sampled path
(see [Beskos et al., 2013](https://doi.org/10.1016/j.spa.2012.12.001)).
The discrete-time AR(1) version has a simpler time structure, but neither the centred nor the non-centred parameterisation is uniformly efficient
across all regimes ([Kastner and Frühwirth-Schnatter, 2014](https://doi.org/10.1016/j.csda.2013.01.002)).
- **Non-linear / non-Gaussian state-space models and partially observed diffusions:** latent states,
state innovations or a non-centred driving path.
- **Survival models with log-normal frailty:** Gaussian log-frailties at the individual / cluster level,
with the hazard / survival likelihood generally preventing analytic integration.
- **Joint longitudinal and survival models:** shared random effects or latent biomarker trajectories;
the survival component generally prevents the Gaussian marginalisation available for simpler longitudinal models.
- **Item-response and non-Gaussian factor models:** latent abilities / factors for binary, ordinal or count responses.
- **Non-linear mixed-effects models:** Gaussian individual-level effects entering a non-linear response model,
e.g. in population pharmacokinetic / pharmacodynamic (PK/PD) models.
- **Measurement-error models with non-linear / non-Gaussian outcomes:** unobserved true exposures / covariates
or residual processes that enter the outcome likelihood.
- **Non-linear inverse problems with Gaussian-prior fields:** unknown spatial functions or initial conditions,
e.g. log-permeability fields inferred through a groundwater-flow model.


For examples of these structures, see the [Stan item-response models](https://mc-stan.org/learn-stan/case-studies/rasch_and_2pl.html),
the [non-linear mixed-effects PK models of Johnston et al., 2024](https://doi.org/10.1002/psp4.13088),
the diffusion bridge, stochastic volatility and latent diffusion survival applications of
[Beskos et al., 2013](https://doi.org/10.1016/j.spa.2012.12.001),
and the function-space applications of [Cotter et al., 2013](https://arxiv.org/abs/1202.0709).


The [runnable examples](#examples) show the implementations currently supplied with NicoStan.
Note that the supplied discrete-time AR(1) stochastic volatility example is a different model from the continuous-time experiments of
[Beskos et al., 2013](https://doi.org/10.1016/j.spa.2012.12.001) and [Beskos et al., 2015](https://doi.org/10.1093/biomet/asv051).


Note that some simpler Gaussian models allow the latent block to be integrated out analytically.
E.g., our Gaussian-outcome GP regression example uses the marginal likelihood and is fitted with standard HMC;
the latent-GP examples listed here instead have non-Gaussian likelihoods.


### How NicoStan samples the nuisance block: diffusion-pathspace HMC and hybrid HMC


Many models have a relatively small number of main parameters, $\boldsymbol{\theta}$,
and a much larger block of nuisance parameters, $\mathbf{u}$, whose (non-centred) prior is standard normal,
i.e. $\mathbf{u} \sim \text{multi\_normal}(\mathbf{0}, \mathbf{I})$.


As the number of nuisance parameters, $D_{\text{nuisance}}$, grows, standard HMC needs an increasingly small step size,
of order $D_{\text{nuisance}}^{-1/4}$, to keep its acceptance rate
(see [Beskos et al., 2013](https://doi.org/10.1016/j.spa.2012.12.001), section 3.3, and the references therein).


Diffusion-pathspace HMC ([Beskos et al., 2011](https://doi.org/10.1016/j.spa.2011.06.003);
[Beskos et al., 2013](https://doi.org/10.1016/j.spa.2012.12.001)) avoids this by integrating the Gaussian part of the
Hamiltonian dynamics exactly: the potential of $\mathbf{u}$ is written as $\frac{1}{2} \mathbf{u}' \mathbf{u} + \Phi(\mathbf{u})$,
and each step is a "kick" (which updates only the momentum, using the gradient of $\Phi$), an exact "rotation" of the
Gaussian part, and a second kick.


Since the rotation leaves the Gaussian reference measure invariant, the integration error only comes from $\Phi$;
hence, under the conditions given by Beskos et al., the step size does not need to decrease as $D_{\text{nuisance}}$ grows.


NicoStan samples the main parameters and the nuisance parameters **jointly**, i.e. within the same trajectory, with a single
Metropolis-Hastings step, using the "advanced hybrid Monte Carlo" algorithm of
[Beskos et al., 2015](https://doi.org/10.1093/biomet/asv051) (section 3.2 and algorithm 3):
standard HMC dynamics for $\boldsymbol{\theta}$, and diffusion-pathspace dynamics for $\mathbf{u}$.


Each step is a kick-flow-kick Strang splitting ([Beskos et al., 2015](https://doi.org/10.1093/biomet/asv051), equations 14-16):
a half kick of both momenta, a flow (the leapfrog drift for $\boldsymbol{\theta}$, an exact rotation for each nuisance parameter),
then a second half kick.
Hence, each step needs a single evaluation of the gradient of $\log \pi$, as for the leapfrog integrator.


By default, NicoStan makes two changes to the Gaussian reference of Beskos et al., neither of which changes the target
(they only change how the Hamiltonian is split):
(i) the reference is centred at a running estimate of the posterior mean of $\mathbf{u}$, made during burn-in and frozen
before the sampling phase, rather than at $\mathbf{0}$;
and (ii) each nuisance parameter has its own mass, estimated during burn-in, so that each rotates with its own frequency.


Note that Beskos et al. use unit masses for the nuisance parameters, which, as they show, makes their algorithm well-posed
as the number of nuisance parameters goes to infinity
([Beskos et al., 2015](https://doi.org/10.1093/biomet/asv051), remark 2 and section 3.3);
our per-coordinate masses do not necessarily keep this property.


[Alenlöv et al., 2021](https://jmlr.org/papers/v22/19-486.html) later used the same split of the Hamiltonian in their
pseudo-marginal HMC algorithm, but with the order of the operators reversed (flow-kick-flow, their equation 19);
NicoStan uses the kick-flow-kick order of Beskos et al.
Other dimension-robust HMC-type algorithms include that of [Pinski, 2021](https://doi.org/10.3390/e23050499)
and the manifold MCMC method of [Beskos, 2014](https://doi.org/10.1016/j.spl.2014.03.016).


In NicoStan, the nuisance blocks of a Stan model are named with `nuisance_params` when the model is set up, and
diffusion-pathspace HMC is switched on with `diffusion_HMC = TRUE` (with `diffusion_HMC_integrator = "kick_flow_kick"`) in
`$sample()`; when `diffusion_HMC` is not given, NicoStan warns and samples the nuisance block with standard HMC.
With the standard leapfrog drift for the nuisance parameters instead of the rotation, the sampler is standard HMC for both
blocks; the NicoStan paper compares both
(*NicoStan: between-chain adaptive hybrid diffusion-pathspace HMC for general Stan models*, in preparation).


<a id="which-parameters-are-nuisance-and-which-are-main-the-paper-3-benchmark-models"></a>
### Which parameters are nuisance and which are main: the benchmark models of the NicoStan paper


Every parameter of the `parameters` block of each benchmark model is listed below, with its notation, its meaning in words
(taken from the comment on its declaration in the Stan file), its size, and whether NicoStan holds it as a nuisance block
(sampled with diffusion-pathspace HMC) or in the main block.
Every N-sized latent block is a nuisance block in every timing, benchmark and fit; the main block holds only the fixed-size
parameters. The models without a nuisance block (Weibull survival, robust Student-t regression and GP regression) have
nothing for nico_reduce_sum to hold: for them, only [Stan's reduce_sum](https://mc-stan.org/docs/stan-users-guide/parallelization.html) applies.


The rows follow the Stan::math versions of the models
(`inst/examples/benchmark_models/<model>_unchunked_no_loops_over_N_with_Stan_fused_functions.stan`;
the 1C-MVOP and 2LC-MVOP from BayesMVP's MVOP and LC-MVOP Stan files).
In the nico_reduce_sum versions (the `_chunked.stan` files), NicoStan holds the nuisance blocks inside the chunks.
Since the nico_reduce_sum writer reads a `matrix` nuisance block whole wherever it is used (i.e. on the main tape,
not chunked), the matrix blocks are declared there as arrays of row vectors, with the same values:
the hierarchical logistic regression's `beta_raw` as `array[K] row_vector[2]`,
the joint longitudinal-survival model's `b_raw` as `array[N_subj] row_vector[2]`,
and the population PK model's `z` as `array[N] row_vector[5]`.
Similarly, the nico_reduce_sum version of the NMA holds its study effects as four blocks, one for the studies with each number
(1 to 4) of index tests (`beta_z_of_the_studies_with_1_index_test`, ..., `beta_z_of_the_studies_with_4_index_tests`).


| Model | Parameter (Stan name) | In words | Size | Block (why) |
|---|---|---|---|---|
| Weibull survival | $\alpha$ (`alpha`) | Weibull shape parameter | 1 | main (fixed size) |
| | $\boldsymbol{\beta}$ (`beta`) | regression coefficients | $K$ | main (fixed size) |
| | $\mu$ (`mu`) | intercept | 1 | main (fixed size) |
| Robust Student-t regression | $\alpha$ (`alpha`) | intercept | 1 | main (fixed size) |
| | $\boldsymbol{\beta}$ (`beta`) | regression coefficients | $K$ | main (fixed size) |
| | $\sigma$ (`sigma`) | scale parameter | 1 | main (fixed size) |
| GP regression | $\ell$ (`lengthscale`) | lengthscale | 1 | main (fixed size; the GP is marginalised, so there is no latent block) |
| | $\sigma_f$ (`sigma_f`) | signal standard deviation | 1 | main (fixed size) |
| | $\sigma_n$ (`sigma_n`) | noise standard deviation | 1 | main (fixed size) |
| Stochastic volatility, non-centred | $\tilde{\mathbf{h}} = (\tilde{h}_1, \ldots, \tilde{h}_T)$ (`h_std`) | standard normal innovations of the log-volatility path | $T$ | nuisance (standard normal latent path, grows with $T$) |
| | $\mu$ (`mu`) | mean log volatility | 1 | main (fixed size) |
| | $\phi$ (`phi`) | persistence of volatility, in $(-1, 1)$ | 1 | main (fixed size) |
| | $\sigma$ (`sigma`) | white noise shock scale | 1 | main (fixed size) |
| Stochastic volatility, centred | $\mathbf{h} = (h_1, \ldots, h_T)$ (`h`) | latent log-volatility path | $T$ | nuisance (latent path, grows with $T$) |
| | $\mu$ (`mu`) | mean log volatility | 1 | main (fixed size) |
| | $\phi$ (`phi`) | persistence of volatility, in $(-1, 1)$ | 1 | main (fixed size) |
| | $\sigma$ (`sigma`) | white noise shock scale | 1 | main (fixed size) |
| Shared-frailty Cox survival | $\tilde{\mathbf{f}}$ (`log_frailty_raw`) | raw (standard normal) log frailties of the clusters | $J$ | nuisance (standard normal frailties, grow with N) |
| | $\boldsymbol{\beta}$ (`beta`) | regression coefficients (log hazard ratios) | $K$ | main (fixed size) |
| | $\sigma_{\text{frailty}}$ (`sigma_frailty`) | standard deviation of the frailties | 1 | main (fixed size) |
| Latent diffusion survival | $\mathbf{w} = (w_1, \ldots, w_M)$ (`w`) | standard normal innovations of the path of the log baseline hazard | $M$ | nuisance (standard normal path, grows with the grid) |
| | $x_0$ (`x_0`) | log baseline hazard at diagnosis (per year) | 1 | main (fixed size) |
| | $\mu_x$ (`mu_x`) | long-run log baseline hazard (per year) | 1 | main (fixed size) |
| | $\kappa$ (`kappa`) | rate of return to $\mu_x$ (per year) | 1 | main (fixed size) |
| | $s$ (`diffusion_scale`) | diffusion of the log baseline hazard (per square root of a year) | 1 | main (fixed size) |
| | $\boldsymbol{\beta}$ (`beta`) | log hazard ratios of the covariates | $K$ | main (fixed size) |
| Latent diffusion survival, the form of [Beskos et al., 2013](https://doi.org/10.1016/j.spa.2012.12.001) (`latent_diffusion_survival_chunked.stan`) | $\mathbf{w} = (w_1, \ldots, w_M)$ (`w`) | standard normal Brownian increments of the latent diffusion path | $M$ | nuisance (standard normal path, grows with the grid) |
| | $\theta_1$ (`drift_sin_coefficient`) | coefficient of $\sin(x)$ in the drift, $-(\theta_1 \sin(x) + \theta_2)$ | 1 | main (fixed size) |
| | $\theta_2$ (`drift_constant`) | constant of the drift | 1 | main (fixed size) |
| | $\kappa$ (`hazard_scale`) | hazard scale, $h(u) = \kappa (x(u)^2 + h_{\text{offset}})$ | 1 | main (fixed size) |
| Joint longitudinal-survival | $\tilde{\mathbf{B}}$ (`b_raw`) | standard normal random effects of the subjects (row 1: intercepts; row 2: slopes, before the correlation $\rho_b$) | $2 \times N_{\text{subj}}$ | nuisance (standard normal random effects, grow with N) |
| | $\boldsymbol{\beta}_{\text{long}}$ (`beta_long`) | fixed effects of the longitudinal submodel | $K_{\text{long}}$ | main (fixed size) |
| | $\sigma_{\text{long}}$ (`sigma_long`) | residual error SD of the longitudinal submodel | 1 | main (fixed size) |
| | $\sigma_{b_0}$ (`sigma_b0`) | random intercept SD | 1 | main (fixed size) |
| | $\sigma_{b_1}$ (`sigma_b1`) | random slope SD | 1 | main (fixed size) |
| | $\rho_b$ (`rho_b`) | correlation between the random intercepts and slopes | 1 | main (fixed size) |
| | $\boldsymbol{\beta}_{\text{surv}}$ (`beta_surv`) | effects of the covariates of the survival submodel | $K_{\text{surv}}$ | main (fixed size) |
| | $\lambda$ (`lambda`) | baseline hazard rate | 1 | main (fixed size) |
| | $\alpha_{\text{assoc}}$ (`alpha_assoc`) | association between the longitudinal trajectory and the hazard | 1 | main (fixed size) |
| Hierarchical logistic regression | $\tilde{\boldsymbol{\beta}}$ (`beta_raw`) | standard normal [intercept, slope] of each of the $K$ groups | $K \times 2$ | nuisance (standard normal group effects, grow with N) |
| | $\boldsymbol{\mu} \in \mathbb{R}^2$ (`mu`) | population means of the [intercept, slope] | 2 | main (fixed size) |
| | $\boldsymbol{\sigma} \in \mathbb{R}^2$ (`sigma`) | population standard deviations of the [intercept, slope] | 2 | main (fixed size) |
| Graded response IRT | $\boldsymbol{\theta} = (\theta_1, \ldots, \theta_N)$ (`theta`) | abilities of the respondents (standard normal) | $N$ | nuisance (standard normal abilities, grow with N) |
| | $\mathbf{a}$ (`a`) | item discriminations (positive) | $J$ | main (fixed size) |
| | $\mathbf{c}_j$, $j = 1, \ldots, J$ (`c`) | item cutpoints (logit scale), ordered within each item | $J(K - 1)$ | main (fixed size) |
| Two-compartment population PK | $\mathbf{Z}$ (`z`) | standard normal effects of the patients (one column of 5 per patient) | $5 \times N$ | nuisance (standard normal patient effects, grow with N) |
| | $\mu_{\text{CL}}$ (`mu_CL`) | population mean of log CL (clearance, L/h) | 1 | main (fixed size) |
| | $\mu_{Q}$ (`mu_Q`) | population mean of log Q (intercompartmental clearance, L/h) | 1 | main (fixed size) |
| | $\mu_{V_1}$ (`mu_V1`) | population mean of log V1 (central volume, L) | 1 | main (fixed size) |
| | $\mu_{V_2}$ (`mu_V2`) | population mean of log V2 (peripheral volume, L) | 1 | main (fixed size) |
| | $\mu_{k_a}$ (`mu_ka`) | population mean of log ka (absorption rate, 1/h), constrained above the population disposition rate $\alpha$, which rules out the flip-flop solution | 1 | main (fixed size) |
| | $\boldsymbol{\omega} \in \mathbb{R}^5$ (`omega`) | between-patient SDs of log CL, log Q, log V1, log V2 and log ka | 5 | main (fixed size) |
| | $\mathbf{L}_{\Omega}$ (`L_Omega`) | Cholesky factor of the correlation matrix of the patient effects | $5 \times 5$ (10 free) | main (fixed size) |
| | $\sigma$ (`sigma`) | residual SD of the log concentrations (5 per patient at every N, at 0.5, 1.5, 4, 8 and 24 h after the dose) | 1 | main (fixed size) |
| GP classification | $\boldsymbol{\eta} = (\eta_1, \ldots, \eta_N)$ (`eta`) | standard normal values of the latent GP | $N$ | nuisance (standard normal latent GP, grows with N) |
| | $\mu$ (`mu`) | constant mean of the latent log-odds | 1 | main (fixed size) |
| | $\ell$ (`lengthscale`) | lengthscale of the squared-exponential covariance | 1 | main (fixed size) |
| | $\sigma_f$ (`sigma_f`) | marginal standard deviation of the latent log-odds | 1 | main (fixed size) |
| Spatial HSGP logistic | $\mathbf{z} = (z_1, \ldots, z_M)$ (`z`) | standard normal coefficients of the basis functions | $M$ | nuisance (standard normal basis coefficients, grow with the basis) |
| | $\alpha$ (`alpha`) | intercept (log-odds at the average location and covariates) | 1 | main (fixed size) |
| | $\boldsymbol{\beta} \in \mathbb{R}^3$ (`beta`) | coefficients of the covariates | 3 | main (fixed size) |
| | $\sigma_{\text{gp}}$ (`sigma_gp`) | marginal standard deviation of the spatial effect | 1 | main (fixed size) |
| | $\ell$ (`length_scale`) | length scale of the spatial effect (units of location) | 1 | main (fixed size) |
| NMA, Xu fixed cutpoints | $\mathbf{Z}_{\eta}$ (`beta_eta_z`) | standard normal study effects, shared by the tests (one per class and study) | $2 \times n_{\text{studies}}$ | nuisance (standard normal study effects, grow with the studies) |
| | $\mathbf{z}_{\delta, c}$, $c = 1, 2$ (`beta_delta_z_vec`) | standard normal test-specific study effects | $2 \times n_{\text{obs}}$ | nuisance (standard normal, grow with the studies) |
| | $\boldsymbol{\beta}_{\mu, t}$ (`beta_mu`) | mean locations of each test and class (with the covariate effects) | $n_{\text{tests}} \times 2 \times n_{\text{cov}}$ | main (fixed size) |
| | $\mathbf{C}^{\text{raw}}_c$, $c = 1, 2$ (`C_raw_vec`) | raw log-differences of the cutpoints, shared by the studies (fixed cutpoints) | $2 \times n_C$ | main (fixed size) |
| | $\boldsymbol{\sigma}_{\beta} \in \mathbb{R}^2$ (`beta_sigma`) | between-study SDs (Nyaga's $\sigma$) | 2 | main (fixed size) |
| | $\boldsymbol{\tau}^{\text{raw}}$ (`beta_tau_raw`) | test-specific between-study SDs | $n_{\text{tests}} \times 2$ | main (fixed size) |
| | $\rho_{\beta}$ (`beta_corr`) | between-study correlation of the two classes (possibly restricted) | 1 | main (fixed size) |
| Four-class LC-MVOP | $\tilde{\mathbf{u}}_i \in \mathbb{R}^{n_{\text{tests}}}$ (`u_raw`) | GHK nuisance variables of each row (one row vector per row) | $n_{\text{rows}} \times n_{\text{tests}}$ | nuisance (GHK latent variables, grow with N) |
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


The MVP and its latent class / ordinal extensions (LC-MVP, LC-MVOP, etc.) are particularly demanding examples.
The augmented latent data can be very large,
whilst strong correlations and outcome-dependent truncation create difficult dependencies between the latent variables and the model parameters.
Additionally, specifically for latent class models, poor identifiability can make sampling harder still
(see [Talhouk et al., 2012](https://www.stats.ox.ac.uk/~doucet/talhouk_doucet_murphy_sparseprobit.pdf)
and [Cerullo et al., 2025](https://arxiv.org/abs/2509.18489v1)).


Stochastic volatility and non-linear state-space models can also be difficult when the latent processes are highly persistent
or their innovation scales are small.


In frailty survival models, joint longitudinal-survival models and latent-factor models,
limited information about the individual effects can induce strong posterior dependence between those effects and their population-level parameters.
Techniques such as non-centring can help with these dependencies in some models;
however, the remaining high-dimensional latent block can nevertheless make the computation prohibitively expensive.


### GPU acceleration and serial dependencies


Some of these models contain calculations that are done in sequence, which limits how much of them can run in parallel
(e.g. on a GPU).
E.g., in the GHK implementations of the MVP / LC-MVP, the conditional bounds and latent-variable transformations of each
outcome depend on the preceding outcomes of the same individual, so they are built up one outcome after another.
Similarly, the non-centred reconstruction of a latent path (e.g. the AR(1) log-volatility path of stochastic volatility)
needs each state before the next state can be calculated.
These dependencies limit the parallelism within an individual / path, although independent individuals, paths and chains
can still be processed in parallel.


GPU suitability therefore depends on the model and on the amount of parallel work available:
e.g., [Stan's OpenCL support](https://mc-stan.org/cmdstanr/articles/articles-online-only/opencl.html) is expected to give
its largest speed-ups with Stan's specialised GLM functions (e.g. `bernoulli_logit_glm`).


NicoStan runs on the CPU, and needs no changes to the model for GPU execution.
Its hybrid diffusion-pathspace HMC, its between-chain adaptation and its CPU parallelism
(many chains, with nico_reduce_sum within each chain) address the cost of posterior sampling directly;
the optional [AVX2 / AVX-512 functions](#custom-avx2-and-avx-512-functions) additionally speed up the evaluation of the
log density and its gradient.


<!--  Further parameterisation examples are given in the Stan User's Guide sections on   -->
<!--  [stochastic volatility](https://mc-stan.org/docs/2_33/stan-users-guide/stochastic-volatility-models.html), -->
<!--  [Gaussian processes](https://mc-stan.org/docs/stan-users-guide/gaussian-processes.html) and   -->
<!--  [reparameterisation](https://mc-stan.org/docs/stan-users-guide/efficiency-tuning.html).   -->


<!-- ------------------------------------------------------------------------------------------------------------------------------- -->
## nico_reduce_sum and nico_map_rect
<!-- ------------------------------------------------------------------------------------------------------------------------------- -->


Stan's [`reduce_sum`](https://mc-stan.org/docs/stan-users-guide/parallelization.html) splits a sum (e.g., the
log likelihood) into chunks, each evaluated on its own autodiff tape, in parallel with
[oneTBB](https://github.com/oneapi-src/oneTBB);
this is the basis of tape chunking (i.e., chunks small enough that each one's tape fits in the CPU's caches;
see *Cache-aware chunking dramatically improves parallel scaling for HMC: Application to multivariate probit
model*, in preparation).


However, with Stan's `reduce_sum`, some of the work still happens outside the chunks, on the main tape.
For a nuisance block with one or more values per unit (e.g., per individual, group or time point),
the main tape still holds one node per value, and `reduce_sum` copies each chunk's slice of the block.
Its bookkeeping (the list of the addresses of the sliced argument's nodes, the zeroing of their partial
derivatives, and adding each chunk's partial derivatives back into them) also runs over every value of the
block, on one thread.
These passes over the whole block grow with N, and at large N they no longer fit in the caches.


`nico_reduce_sum` moves this work inside the chunks:

- **Chunk-local bookkeeping:** the bookkeeping of the sliced argument (the addresses, the zeroing and the adding
back) is done inside each chunk, for that chunk's slice only.
- **Nuisance blocks held by NicoStan inside the chunks:** the nuisance blocks named in `nuisance_params` are taken
out of the Stan parameters and held by NicoStan in one array; each chunk reads only its own slice of the block,
whose values become nodes of the chunk's own tape only, and the chunk's nuisance derivatives are added straight
into NicoStan's array of the gradient, so that the main tape holds no node per nuisance value.


`nico_map_rect` is the counterpart of Stan's
[`map_rect`](https://mc-stan.org/docs/stan-users-guide/parallelization.html), for models whose jobs must each
return a vector (rather than one sum).
Stan's `map_rect` keeps every job's inputs and their Jacobian on the main tape, so that a nuisance block read by
the jobs is first gathered into job arguments on the main tape; `nico_map_rect` reads the nuisance values of each
job inside that job, on the job's own nested tape, instead (see
[Map jobs in a shared-frailty Cox model](#map-jobs-in-a-shared-frailty-cox-model)).


### What nico_reduce_sum gives


For the stochastic volatility model (non-centred, with one latent innovation per time point as its nuisance block),
`nico_reduce_sum` is faster than Stan's `reduce_sum` at every N, and its gain grows with N.
The table gives the sampling throughput (`TP_iter`, i.e., iterations per second over all chains) of the
nico::math (AVX) version with `nico_reduce_sum` over that of the same version with Stan's `reduce_sum`, each at its
own fastest layout of chains, threads per chain and chunks (from the saved layout timings):

| N | Gain over Stan's `reduce_sum`: local-HPC (AVX-512) | Gain over Stan's `reduce_sum`: laptop (AVX2) |
|---:|---:|---:|
| 500 | 1.13x | 1.17x |
| 10,000 | 1.30x | 1.36x |
| 50,000 | 2.11x | 2.54x |

With `nico_reduce_sum`, the AVX-512 version of this model is 3.02x, 4.23x and 5.21x faster than its Stan::math
version (local-HPC, N = 500, 10,000 and 50,000), and the AVX2 version 2.97x, 4.52x and 4.69x (laptop), i.e.,
rising with N.


The speed-ups of the nico::math (AVX) versions over Stan::math for every benchmark model are in [AVX-512 and AVX2 speed-ups over Stan::math](#avx-512-and-avx2-speed-ups-over-stanmath).


### Map jobs in a shared-frailty Cox model


`nico_reduce_sum` applies a partial-sum function to slices and returns their sum; `nico_map_rect` applies a job
function and returns a vector of job results.
In this Cox model, the observations are sorted by decreasing survival time and split into blocks.
Each map job calculates the log-sum-exp of eta within one block, giving that block's log risk.
A cumulative log-sum-exp scan of the block results then gives each later block its risk-set offset.


The scalar reducer uses each block's offset to calculate its Cox partial-likelihood contribution and the prior
contribution of that block's frailty groups.
In other words, the map jobs produce the blocks' risk summaries, and the reducer sums the blocks' contributions.
With `nico_map_rect`, the C++ that NicoStan generates reads the nuisance values needed by each map job inside that
job, rather than first gathering the nuisance values of all blocks into job-argument vectors on the main tape.


For each pair, the scalar reducer and its nuisance-reader code are identical; only the map changes.
Each arm uses its own layout, chosen by the same search.
The comparison covers both math backends, both machines, and N = 10,000, 25,000, 50,000 and 200,000, with
J = N/2 (two subjects per frailty group).


The table gives the ratio of the median `TP_iter` (at L = 10) of `nico_map_rect` to that of Stan's `map_rect`;
each median is over three repeats at that arm's own fastest layout.
Each row compares the same machine, math backend and N.
Layouts are given as chains x within-chain threads (WCP) / requested chunks.


| Machine | Math backend | N | Stan's `map_rect` | `nico_map_rect` | Speed-up |
|---|---|---:|---:|---:|---:|
| local-HPC | Stan::math | 10,000 | 180 x 1 / 8 | 180 x 1 / 4 | 1.007x |
| local-HPC | Stan::math | 25,000 | 180 x 1 / 32 | 180 x 1 / 16 | 1.002x |
| local-HPC | Stan::math | 50,000 | 180 x 1 / 64 | 180 x 1 / 32 | 1.070x |
| local-HPC | Stan::math | 200,000 | 180 x 1 / 512 | 180 x 1 / 256 | 1.239x |
| local-HPC | nico::math (AVX-512) | 10,000 | 180 x 1 / 2 | 180 x 1 / 2 | 1.084x |
| local-HPC | nico::math (AVX-512) | 25,000 | 180 x 1 / 32 | 180 x 1 / 8 | 1.036x |
| local-HPC | nico::math (AVX-512) | 50,000 | 180 x 1 / 64 | 180 x 1 / 32 | 1.123x |
| local-HPC | nico::math (AVX-512) | 200,000 | 180 x 1 / 512 | 180 x 1 / 256 | 1.414x |
| laptop | Stan::math | 10,000 | 16 x 1 / 8 | 16 x 1 / 16 | 0.960x |
| laptop | Stan::math | 25,000 | 16 x 1 / 32 | 16 x 1 / 32 | 1.034x |
| laptop | Stan::math | 50,000 | 8 x 2 / 64 | 8 x 2 / 32 | 1.060x |
| laptop | Stan::math | 200,000 | 8 x 2 / 512 | 8 x 2 / 256 | 1.208x |
| laptop | nico::math (AVX2) | 10,000 | 8 x 2 / 8 | 16 x 1 / 8 | 1.044x |
| laptop | nico::math (AVX2) | 25,000 | 16 x 1 / 32 | 16 x 1 / 16 | 1.109x |
| laptop | nico::math (AVX2) | 50,000 | 16 x 1 / 32 | 16 x 1 / 64 | 1.271x |
| laptop | nico::math (AVX2) | 200,000 | 16 x 1 / 256 | 8 x 2 / 256 | 1.496x |


At N = 10,000 the gain is small; on the laptop with Stan::math, `nico_map_rect` has 4.0% lower throughput.
The gains are larger at N = 50,000 and 200,000, for both math backends and both machines.


### What it changes inside Stan's autodiff system


`nico_reduce_sum` directly modifies Stan's internal automatic differentiation (autodiff) system, i.e., the part
of the [Stan math library](https://github.com/stan-dev/math) that computes the gradient of the log density
([Carpenter et al., 2015](https://arxiv.org/abs/1509.07164)), rather than only the Stan model.
Stan uses reverse-mode autodiff ([Margossian, 2019](https://arxiv.org/abs/1811.05031)): each operation in the
model is recorded as a node (a `vari`) on a tape (i.e., a list of every operation, in the order in which it was
evaluated, kept in Stan's own block of memory, the "arena").
The gradient is then computed by one sweep backwards over the tape, in which each node passes its adjoint (i.e.,
the derivative of the log density with respect to that node's value) back to the nodes it was computed from.


More specifically, for a model with nuisance blocks NicoStan changes these parts of this system:

- **Stan math's `reduce_sum`** (the C++ header file `stan/math/rev/functor/reduce_sum.hpp`): the bookkeeping of
the sliced argument (the list of its nodes' addresses, the zeroing of their partial derivatives, and adding each
chunk's partial derivatives back into its nodes) is done by each chunk, for its own slice only; furthermore, for a
nuisance block held by NicoStan, each chunk makes the nodes of its own nuisance values on its own tape (from
NicoStan's array of values), and their derivatives are added into NicoStan's array of the gradient, so that the
main tape holds no node per nuisance value.
- **Stan math's `map_rect`** (`stan/math/rev/functor/map_rect_concurrent.hpp`), for `nico_map_rect`: each job
reads its own nuisance values on its own nested tape.
- **BridgeStan's gradient call** ([BridgeStan](https://github.com/roualdes/bridgestan)'s `src/model_rng.cpp` and
`src/bridgestan.cpp`), so that the gradient of the nuisance block is returned from NicoStan's array, without extra
copies.


On Linux, every NicoStan model is also compiled with a change to Stan math's arena
(`stan/math/memory/stack_alloc.hpp`),
which lets NicoStan release the memory of each thread's tape after its sampling call.


The gradient is the same as that of the original model, up to the order of a few floating-point additions; we
checked this for the stochastic volatility, hierarchical logistic regression, joint longitudinal and survival,
population pharmacokinetic and IRT models (96 cases per model; largest relative difference 4.9e-16;
see *NicoStan: between-chain adaptive hybrid diffusion-pathspace HMC for general Stan models*, in preparation).


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
- [CmdStan](https://github.com/stan-dev/cmdstan) (which [cmdstanr](https://mc-stan.org/cmdstanr/) and
[CmdStanPy](https://mc-stan.org/cmdstanpy/) use, and which has its own separate copy of the Stan math library)
and [rstan](https://mc-stan.org/rstan/) are never written by NicoStan at all;
- any model you compile with cmdstanr, rstan or BridgeStan itself is compiled against its own, original copy of
Stan, exactly as before.


Note that each change is only made when the lines of the copied file are exactly those NicoStan expects; otherwise
NicoStan stops with an error, rather than changing a file which differs from the one it expects (e.g., one from a
newer version of Stan).
Each private copy is named after the path of the original BridgeStan installation, its arena file and the content
of NicoStan's changes, so that a new version of either makes a new copy (rather than reusing an old one); deleting
the cache folder removes every copy, and NicoStan simply makes a new one the next time it is needed.
These changes are currently made on Linux only.


### How to write a model for it


1. Declare each nuisance block in the `parameters` block, as one of: `vector[E]`, `row_vector[E]`,
`array[E] real`, `array[E] vector[K]`, `array[E] row_vector[K]`, `matrix[R, C]` (read whole where it is used),
`array[A] matrix[R, C]`, or a tuple of these (each component of a tuple becomes a nuisance block of its own),
with `E` its number of units.
Each may have one constraint, `<lower = a>`, `<upper = b>`, `<lower = a, upper = b>`, `<offset = a>`,
`<multiplier = b>` or `<offset = a, multiplier = b>`, with bounds that do not use a nuisance block: NicoStan then
holds the block's unconstrained values, as Stan's own sampler does, and adds the log-Jacobian of the transform to
the target once per value.
Any other block (e.g., an array of tuples, or an array of more dimensions) stops with an error saying so.
2. Make the nuisance block the **sliced** argument of `reduce_sum` or `reduce_sum_static` (which may also be
written `nico_reduce_sum_static`), and put everything whose size grows with N (e.g., the nuisance block's prior,
the likelihood terms) inside the partial-sum function.
Note that `reduce_sum` slices arrays only, so a `vector[E]` block should be declared as `array[E] real` here
(otherwise NicoStan stops with an error).
A nuisance block that is a shared argument of the partial-sum function is also read inside the chunks.
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

(the full model, with its `data` and `transformed data` blocks, is
`inst/examples/benchmark_models/hierarchical_logistic_regression_chunked.stan`), and it is fitted with
`nuisance_params = c("beta_raw")`.


NicoStan then writes the `nico_reduce_sum` version of the model next to it: each nuisance block's declaration is
kept as a comment, the `reduce_sum` call is given the block's term indices as its sliced argument, and a copy of
the partial-sum function (named `<function>_nicostan_chunked_of_<block>`) reads the chunk's own values at its start,
under the name of its sliced argument, so that the rest of the function is unchanged.
Any other use of a nuisance block (outside `reduce_sum`) reads its values where it is used; these values are
correct, but not chunked.


<!-- ------------------------------------------------------------------------------------------------------------------------------- -->
## AVX-512 and AVX2 speed-ups over Stan::math

<a id="avx-512-and-avx2-speed-ups-over-stanmath"></a>

The tables below give, for every benchmark model and every N with saved ground-truth layout timings, how much faster sampling is with nico::math (AVX) than with Stan::math. Two machines were used: the local-HPC (AMD EPYC 9654, 192 threads), which runs the AVX-512 version, and the laptop (AMD Ryzen 7 5800H, 16 threads), which runs the AVX2 version.

A speed-up is TP_iter(AVX) / TP_iter(Stan::math), where TP_iter is the number of sampling iterations per second over all chains, at L = 10 leapfrog steps per iteration. Each version is timed at its own fastest layout (chains, threads per chain and N_chunks), found by searching the layouts separately for each version, machine and N.

Every timed call starts its clock at its first iteration. A timed call with a divergent transition is not used. Where the fastest layouts were timed again 3 times, the speed-up uses the median of the 3 repeats ("median of 3"); otherwise it uses the single timing of the ground-truth search ("single"). Where the two sides differ, the column gives the first version named in "Versions compared" first; a few cases have only 2 repeats ("median of 2").

Like is compared with like: nico::math (AVX) with Stan's reduce_sum against Stan::math with Stan's reduce_sum, and nico_reduce_sum against nico_reduce_sum. The second table compares nico::math (AVX) with the fused nico_ functions (e.g. the fused GHK) against Stan::math with Stan's fused functions (e.g. bernoulli_logit_glm), for the models where Stan has such a function. The third table gives the gain of nico_reduce_sum over Stan's reduce_sum on the same math library. Where a model has more than one Stan file for the same version, the fastest one is used.

### nico::math (AVX) over Stan::math


| Model | N | Versions compared | AVX-512, local-HPC | timing | AVX2, laptop | timing |
|---|---:|---|---:|---|---:|---|
| Four-class LC-MVOP | 500 | nico::math (AVX), reduce_sum over Stan::math, reduce_sum | 1.17x | median of 3 | 1.16x | median of 3 |
| Four-class LC-MVOP | 500 | nico::math (AVX), nico_reduce_sum over Stan::math, nico_reduce_sum | 1.17x | median of 3 | 1.16x | median of 3 |
| Four-class LC-MVOP | 5,000 | nico::math (AVX), reduce_sum over Stan::math, reduce_sum | 1.25x | median of 3 | 1.17x | median of 3 |
| Four-class LC-MVOP | 5,000 | nico::math (AVX), nico_reduce_sum over Stan::math, nico_reduce_sum | 1.21x | median of 3 | 1.23x | median of 3 |
| Four-class LC-MVOP | 25,000 | nico::math (AVX), reduce_sum over Stan::math, reduce_sum | 1.24x | median of 3 | 1.19x | median of 3 |
| Four-class LC-MVOP | 25,000 | nico::math (AVX), nico_reduce_sum over Stan::math, nico_reduce_sum | 1.23x | median of 3 | 1.24x | median of 3 |
| Gaussian process classification | 50 | nico::math (AVX), reduce_sum over Stan::math, reduce_sum | 1.75x | single | 1.63x | single / median of 3 |
| Gaussian process classification | 50 | nico::math (AVX), nico_reduce_sum over Stan::math, nico_reduce_sum | 1.72x | single | 1.62x | single |
| Gaussian process classification | 250 | nico::math (AVX), reduce_sum over Stan::math, reduce_sum | 3.41x | single | 3.92x | median of 3 |
| Gaussian process classification | 250 | nico::math (AVX), nico_reduce_sum over Stan::math, nico_reduce_sum | 3.45x | single | 3.63x | single |
| Gaussian process classification | 500 | nico::math (AVX), reduce_sum over Stan::math, reduce_sum | 4.47x | median of 3 | 3.75x | median of 3 |
| Gaussian process classification | 500 | nico::math (AVX), nico_reduce_sum over Stan::math, nico_reduce_sum | 3.98x | single | 3.27x | single |
| Graded response IRT | 500 | nico::math (AVX), reduce_sum over Stan::math, reduce_sum | 4.40x | median of 3 | 3.85x | median of 3 |
| Graded response IRT | 500 | nico::math (AVX), nico_reduce_sum over Stan::math, nico_reduce_sum | 4.36x | median of 3 / median of 2 | 3.76x | median of 3 |
| Graded response IRT | 10,000 | nico::math (AVX), reduce_sum over Stan::math, reduce_sum | 4.69x | median of 3 | 3.95x | median of 3 |
| Graded response IRT | 10,000 | nico::math (AVX), nico_reduce_sum over Stan::math, nico_reduce_sum | 4.53x | median of 3 / median of 2 | 3.93x | median of 3 |
| Graded response IRT | 50,000 | nico::math (AVX), reduce_sum over Stan::math, reduce_sum | 4.95x | median of 3 | 4.05x | median of 3 |
| Graded response IRT | 50,000 | nico::math (AVX), nico_reduce_sum over Stan::math, nico_reduce_sum | 4.72x | median of 3 | 3.94x | median of 3 |
| Hierarchical logistic regression | 500 | nico::math (AVX), reduce_sum over Stan::math, reduce_sum | 1.45x | median of 3 | 1.38x | median of 3 |
| Hierarchical logistic regression | 500 | nico::math (AVX), nico_reduce_sum over Stan::math, nico_reduce_sum | 1.42x | single / median of 3 | no numerator row |  |
| Hierarchical logistic regression | 10,000 | nico::math (AVX), reduce_sum over Stan::math, reduce_sum | 1.53x | median of 3 | 1.45x | median of 3 |
| Hierarchical logistic regression | 10,000 | nico::math (AVX), nico_reduce_sum over Stan::math, nico_reduce_sum | 1.34x | single / median of 3 | no numerator row |  |
| Hierarchical logistic regression | 50,000 | nico::math (AVX), reduce_sum over Stan::math, reduce_sum | 1.52x | median of 3 | 1.45x | median of 3 |
| Hierarchical logistic regression | 50,000 | nico::math (AVX), nico_reduce_sum over Stan::math, nico_reduce_sum | 1.32x | single / median of 3 | no numerator row |  |
| Latent diffusion survival | 500 | nico::math (AVX), reduce_sum over Stan::math, reduce_sum | 3.09x | median of 3 / single | 2.96x | single |
| Latent diffusion survival | 500 | nico::math (AVX), nico_reduce_sum over Stan::math, nico_reduce_sum | 3.03x | single | 2.87x | single |
| Latent diffusion survival | 10,000 | nico::math (AVX), reduce_sum over Stan::math, reduce_sum | 6.01x | median of 3 / single | 5.64x | single |
| Latent diffusion survival | 10,000 | nico::math (AVX), nico_reduce_sum over Stan::math, nico_reduce_sum | 5.86x | single | 5.80x | single |
| Latent diffusion survival | 50,000 | nico::math (AVX), reduce_sum over Stan::math, reduce_sum | 6.18x | median of 3 / single | 5.64x | single |
| Latent diffusion survival | 50,000 | nico::math (AVX), nico_reduce_sum over Stan::math, nico_reduce_sum | 6.24x | single | 5.58x | single |
| Network meta-analysis (Xu, fixed cutpoints) | 10 | nico::math (AVX), reduce_sum over Stan::math, reduce_sum | 0.84x | single | 0.92x | single |
| Network meta-analysis (Xu, fixed cutpoints) | 10 | nico::math (AVX), nico_reduce_sum over Stan::math, nico_reduce_sum | 1.07x | single | not timed |  |
| Network meta-analysis (Xu, fixed cutpoints) | 25 | nico::math (AVX), reduce_sum over Stan::math, reduce_sum | 0.78x | single | 0.83x | single |
| Network meta-analysis (Xu, fixed cutpoints) | 25 | nico::math (AVX), nico_reduce_sum over Stan::math, nico_reduce_sum | 1.53x | single | not timed |  |
| One-class MVOP | 500 | nico::math (AVX), reduce_sum over Stan::math, reduce_sum | 1.27x | single | 1.17x | single |
| One-class MVOP | 10,000 | nico::math (AVX), reduce_sum over Stan::math, reduce_sum | 1.27x | single | 1.21x | single |
| One-class MVOP | 50,000 | nico::math (AVX), reduce_sum over Stan::math, reduce_sum | 1.18x | single | 1.11x | single |
| Robust Student-t regression | 500 | nico::math (AVX), reduce_sum over Stan::math, reduce_sum | 1.37x | median of 3 | 1.48x | median of 3 |
| Robust Student-t regression | 10,000 | nico::math (AVX), reduce_sum over Stan::math, reduce_sum | 1.69x | median of 3 | 1.63x | median of 3 |
| Robust Student-t regression | 50,000 | nico::math (AVX), reduce_sum over Stan::math, reduce_sum | 1.79x | median of 3 | 1.70x | median of 3 |
| Shared-frailty Cox survival | 500 | nico::math (AVX), reduce_sum over Stan::math, reduce_sum | not timed |  | 1.20x | median of 3 |
| Shared-frailty Cox survival | 10,000 | nico::math (AVX), reduce_sum over Stan::math, reduce_sum | 1.17x | median of 3 / single | 1.08x | median of 3 / single |
| Shared-frailty Cox survival | 50,000 | nico::math (AVX), reduce_sum over Stan::math, reduce_sum | 1.06x | median of 3 / single | 0.93x | median of 3 / single |
| Two-class LC-MVOP | 500 | nico::math (AVX), reduce_sum over Stan::math, reduce_sum | 1.33x | single / median of 2 | 1.20x | single |
| Two-class LC-MVOP | 5,000 | nico::math (AVX), reduce_sum over Stan::math, reduce_sum | 1.31x | single / median of 2 | 1.22x | single |
| Two-class LC-MVOP | 10,000 | nico::math (AVX), reduce_sum over Stan::math, reduce_sum | 1.25x | single / median of 2 | 1.18x | single |
| Two-class LC-MVOP | 25,000 | nico::math (AVX), reduce_sum over Stan::math, reduce_sum | 1.27x | single / median of 2 | 1.23x | single |
| Two-class LC-MVOP | 50,000 | nico::math (AVX), reduce_sum over Stan::math, reduce_sum | 1.25x | single / median of 2 | 1.30x | single |
| Two-class LC-MVOP | 200,000 | nico::math (AVX), reduce_sum over Stan::math, reduce_sum | 1.27x | single / median of 2 | not timed |  |
| Weibull survival | 500 | nico::math (AVX), reduce_sum over Stan::math, reduce_sum | 1.97x | median of 3 | 2.07x | median of 3 |
| Weibull survival | 10,000 | nico::math (AVX), reduce_sum over Stan::math, reduce_sum | 2.09x | median of 3 | 2.33x | median of 3 |
| Weibull survival | 50,000 | nico::math (AVX), reduce_sum over Stan::math, reduce_sum | 2.06x | median of 3 | 2.32x | median of 3 |


Geometric means over the models at each N (number of models in brackets):


| Versions compared | N | AVX-512, local-HPC (models) | AVX2, laptop (models) |
|---|---:|---:|---:|
| nico::math (AVX), reduce_sum over Stan::math, reduce_sum | 10 | 0.84x (1) | 0.92x (1) |
| nico::math (AVX), reduce_sum over Stan::math, reduce_sum | 25 | 0.78x (1) | 0.83x (1) |
| nico::math (AVX), reduce_sum over Stan::math, reduce_sum | 50 | 1.75x (1) | 1.63x (1) |
| nico::math (AVX), reduce_sum over Stan::math, reduce_sum | 250 | 3.41x (1) | 3.92x (1) |
| nico::math (AVX), reduce_sum over Stan::math, reduce_sum | 500 | 1.98x (9) | 1.80x (10) |
| nico::math (AVX), reduce_sum over Stan::math, reduce_sum | 5,000 | 1.28x (2) | 1.19x (2) |
| nico::math (AVX), reduce_sum over Stan::math, reduce_sum | 10,000 | 2.02x (8) | 1.92x (8) |
| nico::math (AVX), reduce_sum over Stan::math, reduce_sum | 25,000 | 1.25x (2) | 1.21x (2) |
| nico::math (AVX), reduce_sum over Stan::math, reduce_sum | 50,000 | 2.01x (8) | 1.91x (8) |
| nico::math (AVX), reduce_sum over Stan::math, reduce_sum | 200,000 | 1.27x (1) |  |
| nico::math (AVX), nico_reduce_sum over Stan::math, nico_reduce_sum | 10 | 1.07x (1) |  |
| nico::math (AVX), nico_reduce_sum over Stan::math, nico_reduce_sum | 25 | 1.53x (1) |  |
| nico::math (AVX), nico_reduce_sum over Stan::math, nico_reduce_sum | 50 | 1.72x (1) | 1.62x (1) |
| nico::math (AVX), nico_reduce_sum over Stan::math, nico_reduce_sum | 250 | 3.45x (1) | 3.63x (1) |
| nico::math (AVX), nico_reduce_sum over Stan::math, nico_reduce_sum | 500 | 2.45x (5) | 2.53x (4) |
| nico::math (AVX), nico_reduce_sum over Stan::math, nico_reduce_sum | 5,000 | 1.21x (1) | 1.23x (1) |
| nico::math (AVX), nico_reduce_sum over Stan::math, nico_reduce_sum | 10,000 | 3.29x (3) | 4.78x (2) |
| nico::math (AVX), nico_reduce_sum over Stan::math, nico_reduce_sum | 25,000 | 1.23x (1) | 1.24x (1) |
| nico::math (AVX), nico_reduce_sum over Stan::math, nico_reduce_sum | 50,000 | 3.38x (3) | 4.69x (2) |


### nico::math (AVX) with the fused nico_ functions over Stan::math with Stan's fused functions


| Model | N | Versions compared | AVX-512, local-HPC | timing | AVX2, laptop | timing |
|---|---:|---|---:|---|---:|---|
| Graded response IRT | 500 | nico::math (AVX) fused, reduce_sum over Stan::math fused, reduce_sum | 5.86x | median of 3 | 4.93x | median of 3 |
| Graded response IRT | 500 | nico::math (AVX) fused, nico_reduce_sum over Stan::math fused, nico_reduce_sum | 5.81x | median of 3 | 4.51x | median of 3 |
| Graded response IRT | 10,000 | nico::math (AVX) fused, reduce_sum over Stan::math fused, reduce_sum | 6.82x | median of 3 | 5.30x | median of 3 |
| Graded response IRT | 10,000 | nico::math (AVX) fused, nico_reduce_sum over Stan::math fused, nico_reduce_sum | 6.92x | median of 3 | 5.10x | median of 3 |
| Graded response IRT | 50,000 | nico::math (AVX) fused, reduce_sum over Stan::math fused, reduce_sum | 6.73x | median of 3 | 5.22x | median of 3 |
| Graded response IRT | 50,000 | nico::math (AVX) fused, nico_reduce_sum over Stan::math fused, nico_reduce_sum | 6.87x | median of 3 | 5.00x | median of 3 |
| Hierarchical logistic regression | 500 | nico::math (AVX) fused, reduce_sum over Stan::math fused, reduce_sum | 2.08x | median of 3 | 2.03x | median of 3 |
| Hierarchical logistic regression | 500 | nico::math (AVX) fused, nico_reduce_sum over Stan::math fused, nico_reduce_sum | 1.78x | median of 3 | 1.73x | median of 3 |
| Hierarchical logistic regression | 10,000 | nico::math (AVX) fused, reduce_sum over Stan::math fused, reduce_sum | 2.48x | median of 3 | 2.14x | median of 3 |
| Hierarchical logistic regression | 10,000 | nico::math (AVX) fused, nico_reduce_sum over Stan::math fused, nico_reduce_sum | 2.66x | median of 3 | 2.74x | median of 3 |
| Hierarchical logistic regression | 50,000 | nico::math (AVX) fused, reduce_sum over Stan::math fused, reduce_sum | 2.59x | median of 3 | 2.12x | median of 3 |
| Hierarchical logistic regression | 50,000 | nico::math (AVX) fused, nico_reduce_sum over Stan::math fused, nico_reduce_sum | 2.75x | median of 3 | 2.64x | median of 3 |
| Joint longitudinal and survival | 500 | nico::math (AVX) fused, reduce_sum over Stan::math fused, reduce_sum | 2.75x | single | 2.78x | single |
| Joint longitudinal and survival | 500 | nico::math (AVX) fused, nico_reduce_sum over Stan::math fused, nico_reduce_sum | 2.76x | median of 3 | 2.86x | median of 3 |
| Joint longitudinal and survival | 10,000 | nico::math (AVX) fused, reduce_sum over Stan::math fused, reduce_sum | 2.66x | single | 2.46x | single |
| Joint longitudinal and survival | 10,000 | nico::math (AVX) fused, nico_reduce_sum over Stan::math fused, nico_reduce_sum | 2.80x | median of 3 | 2.87x | median of 3 |
| Joint longitudinal and survival | 50,000 | nico::math (AVX) fused, reduce_sum over Stan::math fused, reduce_sum | 2.29x | single | 2.27x | single |
| Joint longitudinal and survival | 50,000 | nico::math (AVX) fused, nico_reduce_sum over Stan::math fused, nico_reduce_sum | 2.73x | median of 3 | 2.79x | median of 3 |
| Spatial logistic regression (HSGP) | 500 | nico::math (AVX) fused, reduce_sum over Stan::math fused, reduce_sum | 1.21x | median of 3 / single | 1.25x | median of 3 / single |
| Spatial logistic regression (HSGP) | 500 | nico::math (AVX) fused, nico_reduce_sum over Stan::math fused, nico_reduce_sum | 3.15x | single | 1.22x | single |
| Spatial logistic regression (HSGP) | 10,000 | nico::math (AVX) fused, reduce_sum over Stan::math fused, reduce_sum | 1.51x | median of 3 / single | 1.78x | median of 3 / single |
| Spatial logistic regression (HSGP) | 10,000 | nico::math (AVX) fused, nico_reduce_sum over Stan::math fused, nico_reduce_sum | 4.50x | single | 1.78x | single |
| Spatial logistic regression (HSGP) | 50,000 | nico::math (AVX) fused, reduce_sum over Stan::math fused, reduce_sum | 2.21x | median of 3 / single | 4.75x | median of 3 |
| Spatial logistic regression (HSGP) | 50,000 | nico::math (AVX) fused, nico_reduce_sum over Stan::math fused, nico_reduce_sum | 4.49x | single | 4.78x | single |
| Weibull survival | 500 | nico::math (AVX) fused, reduce_sum over Stan::math fused, reduce_sum | 2.26x | median of 3 | 2.93x | median of 3 |
| Weibull survival | 10,000 | nico::math (AVX) fused, reduce_sum over Stan::math fused, reduce_sum | 6.67x | median of 3 | 4.40x | median of 3 |
| Weibull survival | 50,000 | nico::math (AVX) fused, reduce_sum over Stan::math fused, reduce_sum | 6.60x | median of 3 | 4.70x | median of 3 |


Geometric means over the models at each N (number of models in brackets):


| Versions compared | N | AVX-512, local-HPC (models) | AVX2, laptop (models) |
|---|---:|---:|---:|
| nico::math (AVX) fused, reduce_sum over Stan::math fused, reduce_sum | 500 | 2.47x (5) | 2.52x (5) |
| nico::math (AVX) fused, reduce_sum over Stan::math fused, reduce_sum | 10,000 | 3.40x (5) | 2.94x (5) |
| nico::math (AVX) fused, reduce_sum over Stan::math fused, reduce_sum | 50,000 | 3.57x (5) | 3.55x (5) |
| nico::math (AVX) fused, nico_reduce_sum over Stan::math fused, nico_reduce_sum | 500 | 3.08x (4) | 2.28x (4) |
| nico::math (AVX) fused, nico_reduce_sum over Stan::math fused, nico_reduce_sum | 10,000 | 3.90x (4) | 2.91x (4) |
| nico::math (AVX) fused, nico_reduce_sum over Stan::math fused, nico_reduce_sum | 50,000 | 3.90x (4) | 3.64x (4) |


### nico_reduce_sum over Stan's reduce_sum

Here both sides use the same math library (Stan::math or nico::math (AVX)); the two columns are the two machines, each with its own math library.


| Model | N | Versions compared | local-HPC | timing | laptop | timing |
|---|---:|---|---:|---|---:|---|
| Four-class LC-MVOP | 500 | Stan::math, nico_reduce_sum over Stan::math, reduce_sum | 1.03x | median of 3 | 1.04x | median of 3 |
| Four-class LC-MVOP | 500 | nico::math (AVX), nico_reduce_sum over nico::math (AVX), reduce_sum | 1.03x | median of 3 | 1.04x | median of 3 |
| Four-class LC-MVOP | 500 | nico::math (AVX) fused, nico_reduce_sum over nico::math (AVX) fused, reduce_sum | 0.88x | single | 0.81x | single / median of 3 |
| Four-class LC-MVOP | 5,000 | Stan::math, nico_reduce_sum over Stan::math, reduce_sum | 1.04x | median of 3 | 1.03x | median of 3 |
| Four-class LC-MVOP | 5,000 | nico::math (AVX), nico_reduce_sum over nico::math (AVX), reduce_sum | 1.01x | median of 3 | 1.08x | median of 3 |
| Four-class LC-MVOP | 5,000 | nico::math (AVX) fused, nico_reduce_sum over nico::math (AVX) fused, reduce_sum | 0.88x | single | 0.75x | single / median of 3 |
| Four-class LC-MVOP | 25,000 | Stan::math, nico_reduce_sum over Stan::math, reduce_sum | 1.03x | median of 3 | 1.05x | median of 3 |
| Four-class LC-MVOP | 25,000 | nico::math (AVX), nico_reduce_sum over nico::math (AVX), reduce_sum | 1.03x | median of 3 | 1.09x | median of 3 |
| Four-class LC-MVOP | 25,000 | nico::math (AVX) fused, nico_reduce_sum over nico::math (AVX) fused, reduce_sum | 0.89x | single | 0.80x | single / median of 3 |
| Gaussian process classification | 50 | Stan::math, nico_reduce_sum over Stan::math, reduce_sum | 0.98x | single | 0.97x | single / median of 3 |
| Gaussian process classification | 50 | nico::math (AVX), nico_reduce_sum over nico::math (AVX), reduce_sum | 0.96x | single | 0.97x | single |
| Gaussian process classification | 50 | nico::math (AVX) fused, nico_reduce_sum over nico::math (AVX) fused, reduce_sum | no numerator row |  | 0.99x | single |
| Gaussian process classification | 250 | Stan::math, nico_reduce_sum over Stan::math, reduce_sum | 0.98x | single | 0.99x | single / median of 3 |
| Gaussian process classification | 250 | nico::math (AVX), nico_reduce_sum over nico::math (AVX), reduce_sum | 1.00x | single | 0.92x | single / median of 3 |
| Gaussian process classification | 250 | nico::math (AVX) fused, nico_reduce_sum over nico::math (AVX) fused, reduce_sum | no numerator row |  | 1.00x | single |
| Gaussian process classification | 500 | Stan::math, nico_reduce_sum over Stan::math, reduce_sum | 0.97x | single / median of 3 | 0.99x | single / median of 3 |
| Gaussian process classification | 500 | nico::math (AVX), nico_reduce_sum over nico::math (AVX), reduce_sum | 0.87x | single / median of 3 | 0.86x | single / median of 3 |
| Gaussian process classification | 500 | nico::math (AVX) fused, nico_reduce_sum over nico::math (AVX) fused, reduce_sum | no numerator row |  | 1.00x | single |
| Graded response IRT | 500 | Stan::math, nico_reduce_sum over Stan::math, reduce_sum | 1.00x | median of 2 / median of 3 | 1.03x | median of 3 |
| Graded response IRT | 500 | Stan::math fused, nico_reduce_sum over Stan::math fused, reduce_sum | 1.04x | median of 3 | 1.04x | median of 3 |
| Graded response IRT | 500 | nico::math (AVX), nico_reduce_sum over nico::math (AVX), reduce_sum | 0.99x | median of 3 | 1.01x | median of 3 |
| Graded response IRT | 500 | nico::math (AVX) fused, nico_reduce_sum over nico::math (AVX) fused, reduce_sum | 1.03x | median of 3 | 0.95x | median of 3 |
| Graded response IRT | 10,000 | Stan::math, nico_reduce_sum over Stan::math, reduce_sum | 1.04x | median of 2 / median of 3 | 1.04x | median of 3 |
| Graded response IRT | 10,000 | Stan::math fused, nico_reduce_sum over Stan::math fused, reduce_sum | 1.02x | median of 3 | 1.04x | median of 3 |
| Graded response IRT | 10,000 | nico::math (AVX), nico_reduce_sum over nico::math (AVX), reduce_sum | 1.00x | median of 3 | 1.03x | median of 3 |
| Graded response IRT | 10,000 | nico::math (AVX) fused, nico_reduce_sum over nico::math (AVX) fused, reduce_sum | 1.04x | median of 3 | 1.00x | median of 3 |
| Graded response IRT | 50,000 | Stan::math, nico_reduce_sum over Stan::math, reduce_sum | 1.03x | median of 3 | 1.04x | median of 3 |
| Graded response IRT | 50,000 | Stan::math fused, nico_reduce_sum over Stan::math fused, reduce_sum | 1.03x | median of 3 | 1.04x | median of 3 |
| Graded response IRT | 50,000 | nico::math (AVX), nico_reduce_sum over nico::math (AVX), reduce_sum | 0.98x | median of 3 | 1.01x | median of 3 |
| Graded response IRT | 50,000 | nico::math (AVX) fused, nico_reduce_sum over nico::math (AVX) fused, reduce_sum | 1.05x | median of 3 | 1.00x | median of 3 |
| Hierarchical logistic regression | 500 | Stan::math, nico_reduce_sum over Stan::math, reduce_sum | 1.01x | median of 3 | 0.91x | median of 3 |
| Hierarchical logistic regression | 500 | Stan::math fused, nico_reduce_sum over Stan::math fused, reduce_sum | 1.19x | median of 3 | 1.09x | median of 3 |
| Hierarchical logistic regression | 500 | nico::math (AVX), nico_reduce_sum over nico::math (AVX), reduce_sum | 0.98x | single / median of 3 | no numerator row |  |
| Hierarchical logistic regression | 500 | nico::math (AVX) fused, nico_reduce_sum over nico::math (AVX) fused, reduce_sum | 1.02x | median of 3 | 0.93x | median of 3 |
| Hierarchical logistic regression | 10,000 | Stan::math, nico_reduce_sum over Stan::math, reduce_sum | 1.14x | median of 3 | 1.12x | median of 3 |
| Hierarchical logistic regression | 10,000 | Stan::math fused, nico_reduce_sum over Stan::math fused, reduce_sum | 1.24x | median of 3 | 1.15x | median of 3 |
| Hierarchical logistic regression | 10,000 | nico::math (AVX), nico_reduce_sum over nico::math (AVX), reduce_sum | 1.00x | single / median of 3 | no numerator row |  |
| Hierarchical logistic regression | 10,000 | nico::math (AVX) fused, nico_reduce_sum over nico::math (AVX) fused, reduce_sum | 1.34x | median of 3 | 1.47x | median of 3 |
| Hierarchical logistic regression | 50,000 | Stan::math, nico_reduce_sum over Stan::math, reduce_sum | 1.16x | median of 3 | 1.12x | median of 3 |
| Hierarchical logistic regression | 50,000 | Stan::math fused, nico_reduce_sum over Stan::math fused, reduce_sum | 1.21x | median of 3 | 1.15x | median of 3 |
| Hierarchical logistic regression | 50,000 | nico::math (AVX), nico_reduce_sum over nico::math (AVX), reduce_sum | 1.00x | single / median of 3 | no numerator row |  |
| Hierarchical logistic regression | 50,000 | nico::math (AVX) fused, nico_reduce_sum over nico::math (AVX) fused, reduce_sum | 1.28x | median of 3 | 1.43x | median of 3 |
| Joint longitudinal and survival | 500 | Stan::math fused, nico_reduce_sum over Stan::math fused, reduce_sum | 1.02x | median of 3 / single | 1.02x | median of 3 / single |
| Joint longitudinal and survival | 500 | nico::math (AVX) fused, nico_reduce_sum over nico::math (AVX) fused, reduce_sum | 1.02x | median of 3 / single | 1.05x | median of 3 / single |
| Joint longitudinal and survival | 10,000 | Stan::math fused, nico_reduce_sum over Stan::math fused, reduce_sum | 1.03x | median of 3 / single | 1.00x | median of 3 / single |
| Joint longitudinal and survival | 10,000 | nico::math (AVX) fused, nico_reduce_sum over nico::math (AVX) fused, reduce_sum | 1.08x | median of 3 / single | 1.16x | median of 3 / single |
| Joint longitudinal and survival | 50,000 | Stan::math fused, nico_reduce_sum over Stan::math fused, reduce_sum | 1.07x | median of 3 / single | 1.04x | median of 3 / single |
| Joint longitudinal and survival | 50,000 | nico::math (AVX) fused, nico_reduce_sum over nico::math (AVX) fused, reduce_sum | 1.27x | median of 3 / single | 1.28x | median of 3 / single |
| Latent diffusion survival | 500 | Stan::math, nico_reduce_sum over Stan::math, reduce_sum | 0.96x | single | 0.95x | single |
| Latent diffusion survival | 500 | nico::math (AVX), nico_reduce_sum over nico::math (AVX), reduce_sum | 0.95x | single / median of 3 | 0.93x | single |
| Latent diffusion survival | 500 | nico::math (AVX) fused, nico_reduce_sum over nico::math (AVX) fused, reduce_sum | 0.96x | single | 0.93x | single |
| Latent diffusion survival | 10,000 | Stan::math, nico_reduce_sum over Stan::math, reduce_sum | 1.01x | single | 0.97x | single |
| Latent diffusion survival | 10,000 | nico::math (AVX), nico_reduce_sum over nico::math (AVX), reduce_sum | 0.99x | single / median of 3 | 0.99x | single |
| Latent diffusion survival | 10,000 | nico::math (AVX) fused, nico_reduce_sum over nico::math (AVX) fused, reduce_sum | 1.02x | single | 1.00x | single |
| Latent diffusion survival | 50,000 | Stan::math, nico_reduce_sum over Stan::math, reduce_sum | 1.00x | single | 0.95x | single |
| Latent diffusion survival | 50,000 | nico::math (AVX), nico_reduce_sum over nico::math (AVX), reduce_sum | 1.01x | single / median of 3 | 0.94x | single |
| Latent diffusion survival | 50,000 | nico::math (AVX) fused, nico_reduce_sum over nico::math (AVX) fused, reduce_sum | 1.02x | single | 0.98x | single |
| Network meta-analysis (Xu, fixed cutpoints) | 10 | Stan::math, nico_reduce_sum over Stan::math, reduce_sum | 0.78x | single | no numerator row |  |
| Network meta-analysis (Xu, fixed cutpoints) | 10 | nico::math (AVX), nico_reduce_sum over nico::math (AVX), reduce_sum | 0.99x | single | no numerator row |  |
| Network meta-analysis (Xu, fixed cutpoints) | 25 | Stan::math, nico_reduce_sum over Stan::math, reduce_sum | 0.75x | single | no numerator row |  |
| Network meta-analysis (Xu, fixed cutpoints) | 25 | nico::math (AVX), nico_reduce_sum over nico::math (AVX), reduce_sum | 1.47x | single | no numerator row |  |
| One-class MVOP | 500 | nico::math (AVX) fused, nico_reduce_sum over nico::math (AVX) fused, reduce_sum | 1.03x | single | 0.84x | single |
| One-class MVOP | 10,000 | nico::math (AVX) fused, nico_reduce_sum over nico::math (AVX) fused, reduce_sum | 1.22x | single | 0.94x | single |
| One-class MVOP | 50,000 | nico::math (AVX) fused, nico_reduce_sum over nico::math (AVX) fused, reduce_sum | 1.46x | single | 1.03x | single |
| Shared-frailty Cox survival | 500 | Stan::math, nico_reduce_sum over Stan::math, reduce_sum | not timed |  | 0.89x | median of 3 |
| Shared-frailty Cox survival | 500 | nico::math (AVX) fused, nico_reduce_sum over nico::math (AVX) fused, reduce_sum | not timed |  | 0.28x | median of 3 / single |
| Shared-frailty Cox survival | 10,000 | Stan::math, nico_reduce_sum over Stan::math, reduce_sum | 0.74x | median of 3 / single | 0.62x | median of 3 / single |
| Shared-frailty Cox survival | 10,000 | nico::math (AVX) fused, nico_reduce_sum over nico::math (AVX) fused, reduce_sum | 0.14x | median of 3 / single | 0.11x | median of 3 / single |
| Shared-frailty Cox survival | 50,000 | Stan::math, nico_reduce_sum over Stan::math, reduce_sum | 0.60x | median of 3 / single | 0.60x | median of 3 / single |
| Shared-frailty Cox survival | 50,000 | nico::math (AVX) fused, nico_reduce_sum over nico::math (AVX) fused, reduce_sum | 0.14x | median of 3 / single | 0.19x | median of 3 / single |
| Spatial logistic regression (HSGP) | 500 | Stan::math fused, nico_reduce_sum over Stan::math fused, reduce_sum | 0.35x | single | 0.92x | single |
| Spatial logistic regression (HSGP) | 500 | nico::math (AVX) fused, nico_reduce_sum over nico::math (AVX) fused, reduce_sum | 0.91x | single / median of 3 | 0.90x | single / median of 3 |
| Spatial logistic regression (HSGP) | 10,000 | Stan::math fused, nico_reduce_sum over Stan::math fused, reduce_sum | 0.34x | single | 0.98x | single |
| Spatial logistic regression (HSGP) | 10,000 | nico::math (AVX) fused, nico_reduce_sum over nico::math (AVX) fused, reduce_sum | 1.00x | single / median of 3 | 0.98x | single / median of 3 |
| Spatial logistic regression (HSGP) | 50,000 | Stan::math fused, nico_reduce_sum over Stan::math fused, reduce_sum | 0.49x | single | 1.00x | single / median of 3 |
| Spatial logistic regression (HSGP) | 50,000 | nico::math (AVX) fused, nico_reduce_sum over nico::math (AVX) fused, reduce_sum | 1.00x | single / median of 3 | 1.00x | single / median of 3 |
| Stochastic volatility | 500 | Stan::math, nico_reduce_sum over Stan::math, reduce_sum | 1.02x | median of 3 / single | 1.05x | median of 3 / single |
| Stochastic volatility | 500 | nico::math (AVX) fused, nico_reduce_sum over nico::math (AVX) fused, reduce_sum | 1.13x | single | 1.17x | single |
| Stochastic volatility | 10,000 | Stan::math, nico_reduce_sum over Stan::math, reduce_sum | 1.04x | median of 3 / single | 1.12x | median of 3 / single |
| Stochastic volatility | 10,000 | nico::math (AVX) fused, nico_reduce_sum over nico::math (AVX) fused, reduce_sum | 1.30x | single | 1.36x | single |
| Stochastic volatility | 50,000 | Stan::math, nico_reduce_sum over Stan::math, reduce_sum | 1.21x | median of 3 / single | 1.23x | median of 3 / single |
| Stochastic volatility | 50,000 | nico::math (AVX) fused, nico_reduce_sum over nico::math (AVX) fused, reduce_sum | 2.11x | single | 2.45x | single |
| Two-class LC-MVOP | 500 | nico::math (AVX) fused, nico_reduce_sum over nico::math (AVX) fused, reduce_sum | 0.81x | single / median of 2 | 0.79x | single |
| Two-class LC-MVOP | 5,000 | nico::math (AVX) fused, nico_reduce_sum over nico::math (AVX) fused, reduce_sum | 0.81x | single / median of 2 | 0.84x | single |
| Two-class LC-MVOP | 10,000 | nico::math (AVX) fused, nico_reduce_sum over nico::math (AVX) fused, reduce_sum | 0.87x | single / median of 2 | 0.84x | single |
| Two-class LC-MVOP | 25,000 | nico::math (AVX) fused, nico_reduce_sum over nico::math (AVX) fused, reduce_sum | 0.95x | single / median of 2 | 0.86x | single |
| Two-class LC-MVOP | 50,000 | nico::math (AVX) fused, nico_reduce_sum over nico::math (AVX) fused, reduce_sum | 1.01x | single / median of 2 | 0.85x | single |
| Two-compartment population PK | 50 | Stan::math, nico_reduce_sum over Stan::math, reduce_sum | no numerator row |  | 1.28x | single |
| Two-compartment population PK | 50 | nico::math (AVX) fused, nico_reduce_sum over nico::math (AVX) fused, reduce_sum | not timed |  | 0.66x | single |
| Two-compartment population PK | 500 | Stan::math, nico_reduce_sum over Stan::math, reduce_sum | no numerator row |  | 1.40x | single |
| Two-compartment population PK | 500 | nico::math (AVX) fused, nico_reduce_sum over nico::math (AVX) fused, reduce_sum | no numerator row |  | 0.54x | single |


Geometric means over the models at each N (number of models in brackets):


| Versions compared | N | local-HPC (models) | laptop (models) |
|---|---:|---:|---:|
| Stan::math, nico_reduce_sum over Stan::math, reduce_sum | 10 | 0.78x (1) |  |
| Stan::math, nico_reduce_sum over Stan::math, reduce_sum | 25 | 0.75x (1) |  |
| Stan::math, nico_reduce_sum over Stan::math, reduce_sum | 50 | 0.98x (1) | 1.12x (2) |
| Stan::math, nico_reduce_sum over Stan::math, reduce_sum | 250 | 0.98x (1) | 0.99x (1) |
| Stan::math, nico_reduce_sum over Stan::math, reduce_sum | 500 | 1.00x (6) | 1.02x (8) |
| Stan::math, nico_reduce_sum over Stan::math, reduce_sum | 5,000 | 1.04x (1) | 1.03x (1) |
| Stan::math, nico_reduce_sum over Stan::math, reduce_sum | 10,000 | 0.98x (5) | 0.95x (5) |
| Stan::math, nico_reduce_sum over Stan::math, reduce_sum | 25,000 | 1.03x (1) | 1.05x (1) |
| Stan::math, nico_reduce_sum over Stan::math, reduce_sum | 50,000 | 0.97x (5) | 0.96x (5) |
| Stan::math fused, nico_reduce_sum over Stan::math fused, reduce_sum | 500 | 0.81x (4) | 1.02x (4) |
| Stan::math fused, nico_reduce_sum over Stan::math fused, reduce_sum | 10,000 | 0.82x (4) | 1.04x (4) |
| Stan::math fused, nico_reduce_sum over Stan::math fused, reduce_sum | 50,000 | 0.90x (4) | 1.06x (4) |
| nico::math (AVX), nico_reduce_sum over nico::math (AVX), reduce_sum | 10 | 0.99x (1) |  |
| nico::math (AVX), nico_reduce_sum over nico::math (AVX), reduce_sum | 25 | 1.47x (1) |  |
| nico::math (AVX), nico_reduce_sum over nico::math (AVX), reduce_sum | 50 | 0.96x (1) | 0.97x (1) |
| nico::math (AVX), nico_reduce_sum over nico::math (AVX), reduce_sum | 250 | 1.00x (1) | 0.92x (1) |
| nico::math (AVX), nico_reduce_sum over nico::math (AVX), reduce_sum | 500 | 0.96x (5) | 0.96x (4) |
| nico::math (AVX), nico_reduce_sum over nico::math (AVX), reduce_sum | 5,000 | 1.01x (1) | 1.08x (1) |
| nico::math (AVX), nico_reduce_sum over nico::math (AVX), reduce_sum | 10,000 | 1.00x (3) | 1.01x (2) |
| nico::math (AVX), nico_reduce_sum over nico::math (AVX), reduce_sum | 25,000 | 1.03x (1) | 1.09x (1) |
| nico::math (AVX), nico_reduce_sum over nico::math (AVX), reduce_sum | 50,000 | 1.00x (3) | 0.98x (2) |
| nico::math (AVX) fused, nico_reduce_sum over nico::math (AVX) fused, reduce_sum | 50 |  | 0.81x (2) |
| nico::math (AVX) fused, nico_reduce_sum over nico::math (AVX) fused, reduce_sum | 250 |  | 1.00x (1) |
| nico::math (AVX) fused, nico_reduce_sum over nico::math (AVX) fused, reduce_sum | 500 | 0.97x (9) | 0.80x (12) |
| nico::math (AVX) fused, nico_reduce_sum over nico::math (AVX) fused, reduce_sum | 5,000 | 0.84x (2) | 0.80x (2) |
| nico::math (AVX) fused, nico_reduce_sum over nico::math (AVX) fused, reduce_sum | 10,000 | 0.87x (9) | 0.83x (9) |
| nico::math (AVX) fused, nico_reduce_sum over nico::math (AVX) fused, reduce_sum | 25,000 | 0.92x (2) | 0.83x (2) |
| nico::math (AVX) fused, nico_reduce_sum over nico::math (AVX) fused, reduce_sum | 50,000 | 0.97x (9) | 0.96x (9) |


Over the models timed at N = 10,000 and 50,000, the geometric mean of the speed-ups of nico::math (AVX) over Stan::math with Stan's reduce_sum is 2.02x and 2.01x with AVX-512 (8 models) and 1.92x and 1.91x with AVX2 (8 models), and, with the fused functions on both sides, 3.40x and 3.57x with AVX-512 and 2.94x and 3.55x with AVX2 (5 models).
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
In our initial tests, we have also found NicoStan to be more efficient than [Stan](https://mc-stan.org/)
(via [cmdstanr](https://mc-stan.org/cmdstanr/)) for some of these models;
we expect the different burnin/adaptation schemes to contribute to this, although the detailed comparisons are still in progress
(see [Benchmarks](#benchmarks) and [Efficient burnin algorithms](#efficient-burnin-algorithms-snaper-hmc-and-chees-r-hmc)).


The hybrid/joint sampling scheme uses a "kick-flow-kick" splitting
(`diffusion_HMC_integrator = "kick_flow_kick"`, the default of `$sample()`).
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
whereas Stan normally adapts each [NUTS-HMC](https://www.jmlr.org/papers/v15/hoffman14a.html) chain separately.


This between-chain adaptation is based on ChEES-HMC
([Hoffman et al., 2021](https://proceedings.mlr.press/v130/hoffman21a.html))
and SNAPER-HMC ([Sountsov and Hoffman, 2022](https://arxiv.org/abs/2110.11576v3)).


More specifically, during burnin, NicoStan adapts:

- The step size, targeting a mean acceptance probability of `adapt_delta` (0.80 by default),
using an ADAM-type update ([Kingma and Ba, 2015](https://arxiv.org/abs/1412.6980));
the ADAM moments are reset, and the learning-rate schedule restarted, at the final metric update
(`eps_adam_reset_at_metric_end` and `eps_learning_rate_restart_at_metric_end`, both `TRUE` by default),
so that the step size settles on the final metric rather than on the acceptance crashes of the earlier metric windows.
- The mass matrix (i.e., metric) for the main parameters,
using either empirical covariance/variance estimates from the burnin chains (`metric_type_main = "Empirical"`, the default)
or a numerical Hessian (`metric_type_main = "Hessian"`);
more specifically, the Hessian is computed by finite (central) differences of the main-parameter gradients.
Both methods support a diagonal or dense Euclidean metric (`metric_shape_main = "diag"`, the default, or `"dense"`).
With the (default) empirical metric pooled across the burnin chains, NicoStan updates the metric
once a metric window holds 8 draws, and rescales the step size and trajectory length at each metric update
(i.e., multiplies them by $1/\sqrt{c}$, where $c$ is the geometric mean of the change in the diagonal of the inverse metric;
for a dense metric, the geometric mean of the ratios of its eigenvalues);
more specifically, a few draws are enough to replace an initial (unit) metric which is far off (e.g., for our ordinal LC-MVOP models),
whilst the rescaling means that these early updates do not disturb models whose initial metric is already close (e.g., our binary LC-MVP models).
These can be changed via the `$sample()` arguments `pooled_metric_warm_up`
(a number of draws, 8 by default, or `"hard"`, `"soft"`, `"first_window"`, `"n_over_n_plus_k"` or `"signal_to_noise"`)
and `metric_update_rescales_eps_and_tau` (`TRUE` by default).
The nuisance masses and centre are adapted separately (see [How NicoStan works](#how-nicostan-works)).
- The trajectory length, using the criterion selected via `burnin_algorithm`
(see [Trajectory-length adaptation algorithms](#hmc-trajectory-length-tau-adaptation-algorithms) below),
also with an ADAM-type update.


The main/nuisance separation means that NicoStan can use a dense empirical or Hessian metric for the main block,
whilst retaining a diagonal for a much larger nuisance block (where computing a dense metric would be prohibitively expensive).
Stan's [standard HMC/NUTS metric interface](https://mc-stan.org/docs/cmdstan-guide/mcmc_config.html#metric)
selects a unit, diagonal or dense metric for the complete unconstrained parameter vector;
its dense option therefore includes the nuisance parameters too, rather than exposing separate main/nuisance metric choices.
This distinction allows NicoStan to use a dense main-parameter metric without constructing/storing a dense metric
for the entire latent/nuisance block.


### HMC Trajectory length ($\tau$) adaptation algorithms


Use `burnin_algorithm` (an argument of `$sample()`) to choose the trajectory-length adaptation algorithm;
the default is `"CHESSR"` (ChEES-R):


- `"CHESSR"` (**ChEES-R**, the default): The ChEES-rate criterion,
using the squared change in the monitored parameters' centred squared radius per realised trajectory length.
Proposed by [Sountsov and Hoffman, 2022](https://arxiv.org/abs/2110.11576v3).
- `"SNAPER"` (**SNAPER**): Learns a difficult main-parameter direction,
and adapts the trajectory length using squared changes along that direction per unit length.
Proposed by [Sountsov and Hoffman, 2022](https://arxiv.org/abs/2110.11576v3).
- `"ChEES"` (**ChEES**): Uses the squared change in the monitored parameters' centred squared radius,
without dividing by trajectory length.
Proposed by [Hoffman et al., 2021](https://proceedings.mlr.press/v130/hoffman21a.html).
- `"ESJD"` (**ESJD rate**): Uses the squared jumped distance of the monitored parameters
(in the coordinates defined by the current mass matrix) per unit trajectory length,
i.e., each trajectory's squared jump divided by its own length.
Based on the expected squared jumped distance of [Pasarica and Gelman, 2010](https://www.jstor.org/stable/24308995).
- `"ESJD_CHESSR"` (**ESJD × ChEES-R**), `"ESJD_SNAPER"` (**ESJD × SNAPER**),
`"ESJD_CHESSR_SNAPER"` and `"ESJD_w50_CHESSR_w25_SNAPER_w25"`: the geometric means (blends) of the ESJD-rate criterion
with ChEES-R and / or SNAPER
(see [Blended trajectory-length criteria](#blended-trajectory-length-criteria-geometric-means-of-esjd-chees-r-and-snaper) below).


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


The table below compares the trajectory-length criteria in NicoStan.
Note that z denotes the parameters in the coordinates defined by the current mass matrix, z₀ and z′ the start
and end of a trajectory, m the running estimate of the posterior mean and t the (jittered) length of a trajectory;
by default, every criterion weights each chain's proposal by its acceptance probability.


| `burnin_algorithm` | What it maximises | Per unit trajectory length? | Statistic used | Means (bulk) vs. second moments (tail)¹ | τ gradient estimator | Cost normalisation |
| --- | --- | --- | --- | --- | --- | --- |
| `"ESJD"` | Expected squared jumped distance rate | Yes (each trajectory's jump ÷ its own t) | Jump distance, ‖z′ − z₀‖², over all monitored coordinates | Means (linear), averaged over all monitored coordinates | End-point (`tau_gradient_estimator = "forward"`, the default) or both ends (`"two_ended"`) | ÷ t (optional step offset² and `tau_cost_exponent`) |
| `"ChEES"` | Expected squared change in the centred squared radius, ½‖z − m‖² | No (per iteration) | Squared radius of all monitored coordinates | Second moments (quadratic), weighted towards the largest-variance directions | End-point (default) or both ends | None |
| `"CHESSR"` (ChEES-R) | ChEES per unit trajectory length | Yes (÷ its own t) | Squared radius of all monitored coordinates | Second moments (quadratic), weighted towards the largest-variance directions | End-point (default) or both ends | ÷ t (optional step offset² and `tau_cost_exponent`) |
| `"SNAPER"` | Expected squared change in (wᵀ(z − m))² per unit trajectory length | Yes (÷ its own t) | Squared projection onto a learned leading principal direction, w | Second moments (quadratic), along one direction | End-point (default) or both ends | ÷ t (optional step offset² and `tau_cost_exponent`) |
| `"ESJD_CHESSR"`, `"ESJD_SNAPER"` | The geometric mean of the ESJD rate and the ChEES-R (or SNAPER) criterion | Yes (each component ÷ its own t) | Those of the two components | Both (linear and quadratic), each with weight ½ on the log scale | End-point (default) or both ends | As for the components |
| `"ESJD_CHESSR_SNAPER"`, `"ESJD_w50_CHESSR_w25_SNAPER_w25"` | The weighted geometric mean of the ESJD rate, ChEES-R and SNAPER | Yes (each component ÷ its own t) | Those of the three components | Both (linear and quadratic); weights ⅓, ⅓, ⅓ or 0.50, 0.25, 0.25 on the log scale | End-point (default) or both ends | As for the components |


¹ "Second moments" refers to the squared (quadratic) statistics, which are closer to the tail ESS than the means;
however, they are not the same as the quantile-based tail ESS.

² Via the `$sample()` argument `rate_criterion_cost_offset_steps` (0, i.e., no offset, by default).


For the ESJD rate (and for the ESJD component of every blend), `esjd_jump_power` (2 by default) also allows
the jump to be raised to the power of 3 or 4, rather than 2.


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


We checked NicoStan's SNAPER and ChEES-R implementations against the authors' own [JAX](https://github.com/jax-ml/jax) implementation
(the SNAPER-HMC notebook accompanying [Sountsov and Hoffman, 2022](https://arxiv.org/abs/2110.11576v3))
on the German Credit logistic regression of the [Inference Gym](https://github.com/tensorflow/probability/tree/main/spinoffs/inference_gym);
more specifically, we compared the minimum (over parameters) ESS of the centred squared draws per gradient evaluation,
i.e., the quantity their paper reports.
NicoStan's SNAPER agrees with their implementation to within 5%,
and their ChEES-R value is reproduced once two differences are accounted for:
(i) their implementation divides the ChEES-R and SNAPER criteria by the trajectory length plus one step size
(i.e., it charges the extra gradient evaluation of each iteration), which NicoStan offers via the `$sample()` argument
`rate_criterion_cost_offset_steps`; and
(ii) their sampling phase draws the jittered trajectory lengths from a Halton sequence rather than independently,
which is worth around 15% ESS per gradient on this model (NicoStan offers this via `tau_jitter_sampling_type = "halton"`; see below).
Note that on our latent-class benchmarks the first of these makes the adapted trajectories longer
and the ESS per gradient lower; hence, NicoStan divides by the trajectory length alone by default.


The default trajectory-length settings differ between burnin and sampling:


- **Burnin:** randomised trajectory length by default (`randomize_tau_burnin = TRUE`),
drawn as in the sampling phase; setting `randomize_tau_burnin = FALSE` instead fixes the trajectory length
for each burnin iteration at the current tuning setting, which saves burnin time compared to using a randomised $\tau$.
- **Post-burnin (sampling phase):** randomised trajectory length (`randomize_tau_sampling = TRUE`) -
drawn uniformly between zero and twice the adapted scale (i.e., $\tau \sim \text{uniform}(0, 2 \bar\tau)$) -
with at least one integration step ($L \ge 1$).
By default, these lengths are drawn independently (`tau_jitter_sampling_type = "uniform"`);
alternatively, `tau_jitter_sampling_type = "halton"` takes them from a Halton (base-2 van der Corput) sequence over the same range,
as in the SNAPER-HMC authors' implementation.
In our testing, this gave about 10% more ESS of the posterior SD per gradient on the German Credit regression,
and no measurable change on our binary LC-MVP models.


Note that `manual_tau` separately controls whether the trajectory length is adapted at all -
you can instead set $\tau$ manually, by specifying `manual_tau = TRUE`, and its fixed value by specifying
`tau_if_manual` (e.g., `tau_if_manual = 3.0` to fix $\tau = 3$).


### Blended trajectory-length criteria (geometric means of ESJD, ChEES-R and SNAPER)


NicoStan also offers four blended criteria, which are geometric means of the ESJD-rate criterion
and the ChEES-R and / or SNAPER criteria.
They are described in more detail in
*NicoStan: between-chain adaptive hybrid diffusion-pathspace HMC for general Stan models* (in preparation).


**The three criteria they blend.**
Each criterion uses the main parameters in the coordinates defined by the current mass matrix,
$z = B(\theta - m)$, where $B$ is the (upper-triangular) Cholesky factor of the metric (i.e., $B^\top B = M$)
and $m$ is the running estimate of the posterior mean;
$\tau^{(i)}$ is the (jittered) trajectory length of iteration $i$:

- **ESJD rate** (`"ESJD"`): the expected squared jump of $z$ itself, per unit trajectory length,
$C_{ESJD} = \mathbb{E}\left[ \lVert z^{(i+1)} - z^{(i)} \rVert_2^2 / \tau^{(i)} \right]$.
- **ChEES-R** (`"CHESSR"`; [Sountsov and Hoffman, 2022](https://arxiv.org/abs/2110.11576v3)):
the expected squared change in the ChEES function, $\varphi_{ChEES} = \tfrac{1}{2} \lVert z \rVert_2^2$
([Hoffman et al., 2021](https://proceedings.mlr.press/v130/hoffman21a.html)), per unit trajectory length,
$C_{ChEESR} = \mathbb{E}\left[ \left( \varphi_{ChEES}(\theta^{(i+1)}) - \varphi_{ChEES}(\theta^{(i)}) \right)^2 \right] / \tau$.
- **SNAPER** (`"SNAPER"`; [Sountsov and Hoffman, 2022](https://arxiv.org/abs/2110.11576v3)):
the same along the learned first principal component $v$, with $\varphi_{SNAPER} = (v^\top z)^2$,
$C_{SNAPER} = \mathbb{E}\left[ \left( \varphi_{SNAPER}(\theta^{(i+1)}) - \varphi_{SNAPER}(\theta^{(i)}) \right)^2 \right] / \tau$.


**The blends.**

| `burnin_algorithm` | Criterion |
| --- | --- |
| `"ESJD_CHESSR"` (ESJD × ChEES-R) | $C_{ESJD \times ChEESR} = \sqrt{C_{ESJD} \cdot C_{ChEESR}}$ |
| `"ESJD_SNAPER"` (ESJD × SNAPER) | $C_{ESJD \times SNAPER} = \sqrt{C_{ESJD} \cdot C_{SNAPER}}$ |
| `"ESJD_CHESSR_SNAPER"` | $\left( C_{ESJD} \cdot C_{ChEESR} \cdot C_{SNAPER} \right)^{1/3}$, i.e., the equally weighted geometric mean of the three |
| `"ESJD_w50_CHESSR_w25_SNAPER_w25"` | $\exp\left( 0.50 \log C_{ESJD} + 0.25 \log C_{ChEESR} + 0.25 \log C_{SNAPER} \right)$, i.e., weights 0.50, 0.25 and 0.25 on the logarithms |


In each blend, every component is computed exactly as it is for that criterion alone,
and $\tau$ is updated (by ADAM) along the same weighted average of the components' derivatives of $\log C$ with respect to $\log \tau$;
a burnin iteration where any component has no positive (acceptance-weighted) value leaves $\tau$ unchanged.


**Why a geometric mean.**
Maximising a geometric mean is equivalent to maximising the average of the logarithms of the criteria;
in other words, each criterion contributes equally (or with its stated weight) to the choice of $\tau$, regardless of its scale.
ESJD measures the jumps of the parameters themselves (a first-moment, i.e., linear, statistic),
whereas ChEES-R and SNAPER measure the changes of squared (quadratic) statistics, which are closer to the ESS of the posterior SD;
hence, a blend asks the trajectory length to serve both.
In our preliminary pilot runs on the latent class models of [BayesMVP](https://github.com/CerulloE1996/BayesMVP)
(see below), ESJD chose shorter trajectories than ChEES-R and SNAPER in all 4 cells,
and the blends' trajectory lengths lay between (or at) those of their two components in 7 of the 8 comparisons
(e.g., for the binary LC-MVP with $N = 10{,}000$ and a 125-iteration burnin: $\tau = 1.51$ for ESJD,
1.81 for ESJD × ChEES-R, 1.77 for ESJD × SNAPER, 2.43 for ChEES-R and 2.50 for SNAPER).
Note that this is not a property of ESJD on every target: for a unit Gaussian with exact dynamics,
the ESJD rate is maximised at a longer (fixed) trajectory length than ChEES-R ($t = 2.33$ vs. $1.17$),
since the squared radius oscillates twice as fast as the position.


**Preliminary results: models with and without a nuisance block.**
The table below summarises a preliminary benchmark of eight criteria within NicoStan on seven models
(stochastic volatility, latent diffusion survival, joint longitudinal and survival,
hierarchical logistic regression, Gaussian process regression, Weibull survival, and robust regression with Student-t errors),
each at its first $N$ and at $10 \times N$, with burnin lengths of 250 and 125 iterations
(i.e., 14 cells per $N$; 10 seeds per cell; diffusion HMC for the nuisance blocks).
The loss is the criterion's geometric mean over the seeds relative to the best criterion of the cell
(1.00 = the best; for the minimum ESS per 1,000 gradients, the best criterion's value divided by the criterion's);
"average loss" is the geometric mean of the loss over the 14 cells, "worst loss" the largest,
and "cells OK" the number of cells where the criterion was the best, or within noise of it with the 95% interval within ±25%.
The three-way blends are `"ESJD_CHESSR_SNAPER"` (equal weights) and `"ESJD_w50_CHESSR_w25_SNAPER_w25"` (0.50 / 0.25 / 0.25).


| $N$ | Criterion | Time to min ESS of 1,000: average loss | worst loss | cells OK (of 14) | Min ESS per 1,000 gradients: average loss | worst loss | cells OK (of 14) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| First $N$ | ChEES | 1.107 | 1.76 | 5 | 1.483 | 2.43 | 3 |
| First $N$ | ChEES-R (default) | 1.086 | 1.36 | 7 | 1.211 | 1.77 | 5 |
| First $N$ | SNAPER | **1.045** | **1.19** | 10 | **1.136** | **1.39** | 6 |
| First $N$ | ESJD | 1.073 | 1.30 | **11** | 1.214 | 1.87 | 6 |
| First $N$ | ESJD × ChEES-R | 1.067 | 1.30 | 10 | 1.154 | 1.60 | 6 |
| First $N$ | ESJD × SNAPER | 1.068 | 1.39 | 5 | **1.128** | 1.88 | **10** |
| First $N$ | ESJD × ChEES-R × SNAPER (equal weights) | 1.242 | 1.43 | 1 | 1.145 | 1.70 | 7 |
| First $N$ | ESJD × ChEES-R × SNAPER (0.50 / 0.25 / 0.25) | 1.242 | 1.45 | 3 | 1.160 | 1.75 | 8 |
| $10 \times N$ | ChEES | 1.233 | 1.74 | 3 | 1.489 | 2.72 | 2 |
| $10 \times N$ | ChEES-R (default) | **1.076** | 1.21 | 9 | 1.130 | 1.35 | 8 |
| $10 \times N$ | SNAPER | 1.090 | 1.43 | 8 | 1.159 | 1.47 | 4 |
| $10 \times N$ | ESJD | 1.119 | 2.32 | 9 | 1.167 | 2.69 | 8 |
| $10 \times N$ | ESJD × ChEES-R | **1.069** | 1.21 | 10 | 1.107 | 1.31 | 11 |
| $10 \times N$ | ESJD × SNAPER | 1.082 | 1.48 | **12** | 1.109 | 1.52 | 9 |
| $10 \times N$ | ESJD × ChEES-R × SNAPER (equal weights) | 1.096 | 1.21 | 8 | 1.088 | **1.25** | 10 |
| $10 \times N$ | ESJD × ChEES-R × SNAPER (0.50 / 0.25 / 0.25) | 1.084 | **1.16** | 7 | **1.062** | **1.25** | **12** |

(Bold: the best of each column at each $N$, and any average or worst loss within 1% of it. Preliminary.)


In other words (preliminary): no criterion was uniformly best;
at the first $N$, SNAPER had the smallest average loss for the time to a minimum ESS of 1,000 (1.045)
and the smallest worst losses for both endpoints,
whilst ESJD × SNAPER had the smallest average loss for the minimum ESS per 1,000 gradients (1.128, with SNAPER within 1%: 1.136);
at $10 \times N$, ESJD × ChEES-R had the smallest average loss for the time to a minimum ESS of 1,000
(1.069, with ChEES-R within 1%: 1.076),
whilst the weighted three-way blend (0.50 / 0.25 / 0.25) had the smallest average loss for the minimum ESS per 1,000 gradients
(1.062), the smallest worst losses for both endpoints (1.16 and 1.25) and the most cells OK for that endpoint (12 of 14).
ChEES (without the division by the trajectory length) had the largest average loss for the minimum ESS per 1,000 gradients
at both $N$, and for the time to a minimum ESS of 1,000 at $10 \times N$;
at the first $N$, the two three-way blends had the largest average loss for the time to a minimum ESS of 1,000 (1.242).


**Preliminary results: latent class models (BayesMVP).**
In a preliminary pilot study on the binary LC-MVP ($N = 10{,}000$) and the ordinal LC-MVOP ($N = 1000$),
with burnin lengths of 125 and 250 iterations (4 cells; 20 seeds per criterion and cell; 400 runs in all),
the average ratios to the best criterion of each cell (above 1 = a loss) were:

| Criterion | Gradient evaluations to the target min ESS: average ratio | worst ratio | cells OK (of 4) | Min ESS per 1,000 gradients: average ratio | worst ratio | cells OK (of 4) |
| --- | --- | --- | --- | --- | --- | --- |
| ESJD | **1.072** | **1.171** | **3** | **1.091** | **1.204** | **3** |
| ESJD × SNAPER | 1.139 | 1.265 | 2 | 1.156 | 1.265 | 1 |
| SNAPER | 1.161 | 1.405 | 2 | 1.162 | 1.405 | 2 |
| ChEES-R (default) | 1.253 | 1.765 | 1 | 1.262 | 1.766 | 1 |
| ESJD × ChEES-R | 1.293 | 1.646 | 1 | 1.316 | 1.647 | 0 |

("Cells OK": the best of the cell, or within noise of it with the 95% interval within ±25%. Preliminary.)

Hence, on these latent class models, ESJD × SNAPER was second only to ESJD,
whilst ESJD × ChEES-R was last on average;
the three-way blends were not part of this pilot study.


**How to choose a blend.** NicoStan's default is ChEES-R (`burnin_algorithm = "CHESSR"`);
to use a blend, pass its name to `$sample()`, e.g. (the call of [Example 1](#example-1-one-nuisance-block-logistic-regression-with-random-intercepts) with the blend in place of the default):

```r
model$sample(n_burnin = 125, n_iter = 250,
             n_chains_burnin = 4, n_chains_sampling = 4, n_superchains = 2,
             init_lists_per_chain = initial_values,
             burnin_algorithm = "ESJD_CHESSR_SNAPER",
             diffusion_HMC = TRUE, partitioned_HMC = FALSE,
             seed = 2026)
```

where `burnin_algorithm` is one of `"ESJD_CHESSR"`, `"ESJD_SNAPER"`, `"ESJD_CHESSR_SNAPER"` or
`"ESJD_w50_CHESSR_w25_SNAPER_w25"` (the names are not case-sensitive; e.g., `"esjd_chessr_snaper"` also works).


<!-- ------------------------------------------------------------------------------------------------------------------------------- -->
## Custom AVX2 and AVX-512 functions
<!-- ------------------------------------------------------------------------------------------------------------------------------- -->


NicoStan can also use our custom C++, vectorised (i.e., SIMD) mathematical functions,
which are supplied by [NicoSIMD](#nicosimd-the-simd-functions-on-every-device),
the SIMD library of both NicoStan and [BayesMVP](https://github.com/CerulloE1996/BayesMVP).
These include, for instance, exponentials (`nico_exp()`), logarithms (`nico_log()`),
and normal distribution functions (`nico_Phi()`, `nico_inv_Phi()`, `nico_Phi_approx()` and `nico_inv_Phi_approx()`),
as well as their derivatives, log densities, GLMs and fused likelihood functions.


- **AVX2:** four double-precision values per vector operation; supported by most x86-64 CPUs.
- **AVX-512:** eight double-precision values per vector operation, on supported processors.
- **General Stan models:** declare and call the `nico_` functions in the `.stan` model,
then supply their C++ header via `Stan_cpp_user_header`
(`NicoSIMD::fn_include_folder_Stan_user_header_and_build_arguments_of_NicoSIMD_for_this_CPU()` gives the header
and the build arguments for the CPU in use, see
[Example 3](#example-3-the-multivariate-ordered-probit-model-with-the-fused-ghk-function-and-bayesmvp)).
- **Specialised [BayesMVP](https://github.com/CerulloE1996/BayesMVP) models:**
the same NicoSIMD functions are used within the native implementations.


Note that the AVX versions are not a drop-in switch for an existing Stan model;
more specifically, the user has to re-write their Stan model (i.e., the `.stan` file),
so that it calls the `nico_` functions of NicoSIMD in place of the Stan::math ones.


The comparison examples (whose Stan models have already been re-written in this way)
support separate `math_backend = "Stan"`, `"AVX2"` and `"AVX512"` options.
The compiled AVX model reports its vector size (i.e., four for AVX2 and eight for AVX-512),
so you can check which version is being used.
Note that you should compile the model on the same machine you will use for sampling,
since the available instruction sets depend on the CPU
(see the end of this section for how to check which ones your CPU supports).


### NicoSIMD: the SIMD functions on every device


SIMD (single instruction, multiple data) instructions apply one operation
(for instance, an addition, a multiplication, or one step of the polynomial approximation of `exp()`)
to several numbers at once, all held in one wide register of the CPU.
Which SIMD instructions are available depends on the CPU's instruction set;
more specifically:

- **AVX2** (Advanced Vector Extensions 2; 256-bit registers, i.e., four double-precision values per instruction)
is an x86 (Intel and AMD) instruction set.
It is supported by every Intel Core CPU since the 4th-generation Core CPUs ("Haswell"),
and by every AMD CPU since "Excavator", including every AMD Ryzen, Threadripper and EPYC CPU;
note that some low-power Intel CPUs (e.g., older Atom, Celeron and Pentium models) do not support it.
- **AVX-512** (512-bit registers, i.e., eight double-precision values per instruction)
is also an x86 instruction set.
It is supported by AMD's Zen 4 and Zen 5 CPUs (e.g., Ryzen 7000 and 9000, and EPYC 9004 and 9005),
by Intel's Xeon (server) CPUs since Skylake-SP,
and by some 10th- and 11th-generation Intel laptop and desktop CPUs;
however, it is switched off in Intel's 12th-generation (and later) consumer CPUs.
- **NEON** (also called Advanced SIMD; 128-bit registers, i.e., two double-precision values per instruction)
is the SIMD instruction set of ARM CPUs, which have no AVX.
It is supported by every 64-bit ARM CPU, for instance:
Apple Silicon (M1-M4) Macs,
Qualcomm Snapdragon CPUs (including the Snapdragon X CPUs of Windows on ARM laptops),
the Raspberry Pi 3, 4 and 5 (with a 64-bit operating system),
and ARM servers such as AWS Graviton, NVIDIA Grace and Ampere Altra.


NicoSIMD (named, like NicoStan, after Nico) is the one SIMD library of both NicoStan and BayesMVP:
a separate, header-only R package, which both packages import, and which is bundled with NicoStan
(in its `inst/NicoSIMD` folder), so that `install_NicoStan()` installs it first.
Every SIMD function is called from Stan by its `nico_` name, as an external function
(declared in the `functions` block with no body, and compiled with `--allow-undefined` and NicoSIMD's header).
More specifically, NicoSIMD holds:


| Category | `nico_` functions (examples) | Fused |
|---|---|---|
| Elementwise functions (for a `vector`, a `row_vector` and a `matrix`) | `nico_exp`, `nico_expm1`, `nico_log`, `nico_log1p`, `nico_log1m`, `nico_log1p_exp`, `nico_log1m_exp`, `nico_inv_logit`, `nico_log_inv_logit`, `nico_log1m_inv_logit`, `nico_logit`, `nico_Phi`, `nico_Phi_approx`, `nico_log_Phi`, `nico_log_Phi_approx`, `nico_inv_Phi`, `nico_inv_Phi_from_log_p`, `nico_inv_Phi_approx`, `nico_inv_Phi_approx_from_logit_prob`, `nico_tanh`, `nico_sqrt`, `nico_square` | no |
| Reductions and matrix functions | `nico_log_sum_exp`, `nico_sum_log`, `nico_sum_log_square_plus`, `nico_dot_product_exp`, `nico_cholesky_decompose`, `nico_gp_exp_quad_cov` | no |
| Log densities of whole vectors | `nico_normal_lpdf`, `nico_student_t_lpdf`, `nico_weibull_lpdf`, `nico_weibull_lccdf`, `nico_bernoulli_logit_lpmf`, `nico_binomial_lpmf`, `nico_ordered_logistic_lpmf` | no |
| GLMs of Stan::math | `nico_normal_id_glm_lpdf`, `nico_bernoulli_logit_glm_lpmf`, `nico_ordered_logistic_glm_lpmf` | yes |
| Our own fused likelihoods (no Stan built-in) | `nico_bernoulli_probit_glm_lpmf`, `nico_bernoulli_logit_glm_with_coefficients_by_group_lpmf`, `nico_bernoulli_logit_grouped_regression_lpmf`, `nico_bernoulli_logit_glm_with_a_tensor_product_basis_lpmf`, `nico_student_t_id_glm_lpdf`, `nico_weibull_log_glm_lpdf`, `nico_weibull_log_glm_lccdf`, `nico_normal_log_scale_lpdf`, `nico_normal_affine_location_and_log_scale_lpdf`, `nico_normal_affine_recurrence_log_scale_joint_lpdf`, `nico_gp_exp_quad_cov_normal_marginal_lpdf`, `nico_gp_normal_marginal_lpdf` (with a choice of kernel), `nico_finite_mixture_of_multivariate_ordered_probit_GHK_log_likelihood` (the fused GHK), `nico_binomial_counts_above_successive_cutpoints_of_a_cumulative_link_lpmf`, `nico_proportional_hazards_log_likelihood_with_a_piecewise_linear_baseline_hazard_on_a_grid` | yes |
| Our own fused priors | `nico_induced_dirichlet_prior_of_ordered_cutpoints_lpdf` | yes |
| Our own fused multi-step building blocks | `nico_AR1_recursion`, `nico_stationary_noncentred_AR1_path`, `nico_affine_recurrence_at_positions`, `nico_last_value_of_affine_recurrence`, `nico_log_sum_exp_of_affine_predictor`, `nico_log_cumulative_sum_exp`, `nico_dot_products_of_affine_predictor_and_log_cumulative_sum_exp`, `nico_dot_product_exp_of_affine_predictor_by_column`, `nico_sum_of_exponentials_by_group`, `nico_two_compartment_first_order_input_exponents_and_coefficients`, `nico_rows_of_mean_plus_random_effects_times_transposed_loadings`, `nico_gp_exp_quad_cov_cholesky`, `nico_truncated_normal_quantile_and_log_mass` | yes |


### Mathematical backend and reverse-mode fusion

A fused function combines two or more expensive mathematical operations internally in C++ into
one autodiff operation with a combined reverse pass, instead of recording those expensive
operations separately on the AD tape. This must happen at the C++ stan::math::var / reverse-mode
AD level.

Cheap operations - sqrt, pow, multiplication, division, addition and subtraction - do not count
towards the two-operation requirement. Squares and matrix products remain cheap as well.

A wrapper around separate AD calls is not fusion. Computing log1p(exp(x)) inside a C++ function
does not qualify when exp and log1p still create separate AD operations. Computing their values
internally and registering one operation with combined derivative propagation does qualify.

### Four experimental categories

NicoSIMD is the implementation of the nico::math mathematical backend in this table.

| Mathematical backend | Fusion category | Definition |
|---|---|---|
| Stan::math | Non-fused | Use Stan::math mathematical implementations. No call may combine two or more expensive mathematical operations internally in C++ into one reverse-mode autodiff operation with a combined reverse pass. When a calculation requires several expensive operations, evaluate them as separate autodiff operations. |
| Stan::math | Fused | Use Stan::math mathematical implementations and maximise the use of its applicable functions that combine two or more expensive mathematical operations internally in C++ into one reverse-mode autodiff operation with a combined reverse pass. Use ordinary Stan::math calls for remaining work. The fused category exists only when at least one applicable call satisfies this implementation criterion. |
| NicoSIMD | Non-fused | Preserve the non-fused Stan model's structure and replace each expensive mathematical call with its NicoSIMD implementation. Each replacement must remain non-fused under the same C++ reverse-mode criterion; a matching function name or interface is not proof. Keep cheap operations on Stan::math. |
| NicoSIMD | Fused | Maximise the use of applicable NicoSIMD functions that combine two or more expensive mathematical operations internally in C++ into one reverse-mode autodiff operation with a combined reverse pass. Use ordinary NicoSIMD calls for remaining expensive work and Stan::math for cheap operations. The fused category exists only when at least one applicable call satisfies this implementation criterion. |

### Required accompanying qualifications

1. Preserve the statistical model, priors, constraints and parameterisation. Do not change a model
   merely to make a fused function applicable.
2. NicoSIMD has many more fused functions than Stan::math. Each fused cell maximises its own
   backend's applicable fusions. The files often differ; where no additional Nico fusion applies,
   corresponding Nico calls may use the same structure as the Stan-fused implementation.
3. Mathematical backend, fusion, summation and mapping are independent axes. Summation and map
   calls follow the design row; they do not establish mathematical fusion. A Stan-mathematics
   hybrid using Nico summation, mapping or nuisance readers is not a 100% Stan-only comparator.
4. A speed comparator uses only Stan::math mathematics and Stan's own summation/mapping, or none.
5. Inspect the actual C++ overload, parameter/data roles, proportional specialization and combined
   reverse operation. A function name, vectorisation, a GLM label or a README classification is
   not proof. Inspect the mathematical operations combined by the implementation, including
   operations eliminated analytically rather than retained as literal function calls. No particular
   callback API is required. A combined reverse operation alone does not establish fusion.
6. Parameter values extracted into double or SIMD buffers retain their derivative dependence.
   Do not count only the explicit expensive calls left after analytic simplification. In particular,
   poisson_log_lpmf and poisson_log_glm_lpmf / poisson_log_glm_lupmf combine the exponential link
   and Poisson log probability in one C++ autodiff operation. Done separately, exp followed by
   poisson_lpmf records separate operations. These Poisson log functions are fused. Supporting
   counting conventions never override this definition or the user's explicit rulings.
7. stanc runs at its default transformation level, without --O1 or --Oexperimental. The specified
   C++ compiler optimisation flags remain unchanged. Inspect generated C++ for introduced calls;
   separate-looking Stan expressions are not proof of separate reverse-mode operations.
8. Numerical log-density and gradient agreement are separate build checks, not conversion or
   fusion proof. Declarations without calls and double-only generated quantities do not establish
   reverse-mode likelihood fusion; they still obey the mathematical-backend requirements.
9. If an overload or input regime is not verified, mark it unverified. Do not infer classifications
   for every density, GLM, recurrence or inventory group by analogy. This category table is not a
   certification that existing model cells conform.

The function inventory above requires overload-specific checking against this definition;
its group labels are not certification of every listed function or argument regime.
The fused GHK function covers the multivariate (ordered) probit family
(1, 2 or 4 latent classes; binary, ordinal and mixed outcomes; missing results).


NicoSIMD also includes [SIMDe](https://github.com/simd-everywhere/simde), for NEON on ARM CPUs (see below).
Every SIMD local variable is declared aligned (`ALIGN32` for AVX2 and `ALIGN64` for AVX-512),
so that the AVX code is also safe on Windows.
Note that a model compiled without NicoSIMD's functions runs on the Stan::math functions, just more slowly.


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
| **AVX-512** | AMD Zen 4 and Zen 5 (e.g., Ryzen 7000 and 9000, EPYC 9004 and 9005); Intel Xeon (server) CPUs since Skylake-SP; some 10th- and 11th-generation Intel laptop and desktop CPUs | 8 (512-bit `zmm` registers) | `-O3 -march=native -mtune=native -mfma -mavx2 -mavx512f -mavx512vl -mavx512dq` | the AVX-512 kernels (`__m512d`), natively |
| **AVX2** | practically every x86-64 (Intel and AMD) CPU of the last decade: Intel since Haswell, and AMD since Excavator, including every Ryzen, Threadripper and EPYC CPU | 4 (256-bit `ymm` registers) | `-O3 -march=native -mtune=native -mfma -mavx2` | the AVX2 kernels (`__m256d`), natively |
| **NEON (through SIMDe)** | every 64-bit ARM (AArch64) CPU: Apple Silicon (M1-M4) Macs, Qualcomm Snapdragon (e.g., the Snapdragon X CPUs of Windows on ARM laptops), AWS Graviton, NVIDIA Grace, Ampere Altra, and the Raspberry Pi 3, 4 and 5 (with a 64-bit operating system) | 2 (128-bit NEON registers, `float64x2_t`) | `-O3 -mcpu=native` | the AVX2 kernels, translated into NEON instructions by [SIMDe](https://github.com/simd-everywhere/simde) |
| **Stan::math** | any other CPU (e.g., an x86-64 CPU without AVX2, or a 32-bit ARM CPU) | Eigen's own (see below) | `-O3 -march=native -mtune=native` | no NicoSIMD functions: the model uses Stan's own math library and Eigen |


Some more detail on each option:

- **AVX-512** requires three of its extensions: AVX-512F (the foundation), AVX-512VL (the 512-bit instructions
also on 256-bit registers) and AVX-512DQ (e.g., the conversions between 64-bit integers and doubles, which the
`exp()` and `log()` kernels use). Note that Intel switched AVX-512 off in its 12th-generation (Alder Lake) and later
consumer CPUs, since their efficiency cores do not support it; it is expected to return in future Intel desktop
CPUs, through AVX10. Furthermore, Zen 4 CPUs run each 512-bit instruction as two 256-bit halves;
nevertheless, on our Zen 4 server, the AVX-512 versions were at least as fast as the AVX2 versions in almost every
case.
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
- **Stan::math:** the model is compiled with Stan's own math library
(i.e., exactly as Stan compiles it), whose linear algebra uses [Eigen](https://eigen.tuxfamily.org);
Eigen itself vectorises its matrix and vector operations with whatever SIMD instructions the compiler enables
(e.g., SSE2, AVX or NEON), but the elementwise transcendental functions (e.g., `exp()`, `log()` and `Phi()`)
and the log densities are Stan's own scalar versions.


In every case, a compiled model reports the vector width of its NicoSIMD functions through `bmvp_simd_lanes()`
(declared in Stan as `real bmvp_simd_lanes();`;
i.e., 8 for AVX-512, and 4 for AVX2 and for NEON through SIMDe, since SIMDe keeps the AVX2 kernels' width),
so that one can check which version is being used.


To check which of these your CPU supports:
on Linux, `grep -o -w -E 'avx2|avx512f|asimd' /proc/cpuinfo | sort -u`
(where `asimd` means NEON, on ARM CPUs);
on Windows, a tool such as [CPU-Z](https://www.cpuid.com/softwares/cpu-z.html);
on an Intel Mac, `sysctl -a | grep machdep.cpu`
(every Apple Silicon Mac supports NEON, and none supports AVX).


<!-- ------------------------------------------------------------------------------------------------------------------------------- -->
## How to cite NicoStan
<!-- ------------------------------------------------------------------------------------------------------------------------------- -->


If you use NicoStan in your work, you can cite the package as follows
(the same entry as in its [`CITATION.cff`](CITATION.cff)):


Cerullo, E. (2026). NicoStan: Adaptive Hamiltonian Monte Carlo for Stan Models.
R package version 0.1.9000. https://github.com/CerulloE1996/NicoStan


```bibtex
@Manual{Cerullo2026NicoStan,
  title = {NicoStan: Adaptive Hamiltonian Monte Carlo for Stan Models},
  author = {Enzo Cerullo},
  year = {2026},
  note = {R package version 0.1.9000},
  url = {https://github.com/CerulloE1996/NicoStan},
}
```

NicoSIMD (the SIMD functions) is installed with NicoStan; if you use its `nico_` functions, please cite NicoStan.
Please also cite the methodological references relevant to the options used in your analysis
(see [References](#references)).


<!-- ------------------------------------------------------------------------------------------------------------------------------- -->
## How to cite BayesMVP
<!-- ------------------------------------------------------------------------------------------------------------------------------- -->


If you use the BayesMVP extension (i.e., the specialised MVP / MVOP models), you can also cite:


Cerullo, E. (2026). BayesMVP: Specialised Multivariate Probit Models Using NicoStan.
R package version 0.1.9000. https://github.com/CerulloE1996/BayesMVP

```bibtex
@Manual{Cerullo2026BayesMVP,
  title = {BayesMVP: Specialised Multivariate Probit Models Using NicoStan},
  author = {Enzo Cerullo},
  year = {2026},
  note = {R package version 0.1.9000},
  url = {https://github.com/CerulloE1996/BayesMVP},
}
```

The cache-aware chunking of BayesMVP's models is described in
"Cache-aware chunking dramatically improves parallel scaling for HMC: Application to multivariate probit model"
(in preparation).


<!-- ------------------------------------------------------------------------------------------------------------------------------- -->
## References
<!-- ------------------------------------------------------------------------------------------------------------------------------- -->


1. Beskos, A., Pinski, F. J., Sanz-Serna, J. M. and Stuart, A. M. (2011). [Hybrid Monte Carlo on Hilbert spaces](https://doi.org/10.1016/j.spa.2011.06.003). Stochastic Processes and Their Applications, 121(10), 2201-2230.

2. Beskos, A., Kalogeropoulos, K. and Pazos, E. (2013). [Advanced MCMC methods for sampling on diffusion pathspace](https://doi.org/10.1016/j.spa.2012.12.001). Stochastic Processes and Their Applications, 123(4), 1415-1453.

3. Beskos, A., Dureau, J. and Kalogeropoulos, K. (2015). [Bayesian inference for partially observed stochastic differential equations driven by fractional Brownian motion](https://doi.org/10.1093/biomet/asv051). Biometrika, 102(4), 809-827.

4. Alenlöv, J., Doucet, A. and Lindsten, F. (2021). [Pseudo-Marginal Hamiltonian Monte Carlo](https://jmlr.org/papers/v22/19-486.html). Journal of Machine Learning Research, 22(141), 1-45.

5. Hoffman, M. D., Radul, A. and Sountsov, P. (2021). [An Adaptive-MCMC Scheme for Setting Trajectory Lengths in Hamiltonian Monte Carlo](https://proceedings.mlr.press/v130/hoffman21a.html). Proceedings of Machine Learning Research, 130, 3907-3915.

6. Sountsov, P. and Hoffman, M. D. (2022). [Focusing on Difficult Directions for Learning HMC Trajectory Lengths](https://arxiv.org/abs/2110.11576v3). arXiv:2110.11576, version 3.

7. Pinski, F. J. (2021). [A Novel Hybrid Monte Carlo Algorithm for Sampling Path Space](https://doi.org/10.3390/e23050499). Entropy, 23(5), 499.

8. Riou-Durand, L., Sountsov, P., Vogrinc, J., Margossian, C. and Power, S. (2023). [Adaptive Tuning for Metropolis Adjusted Langevin Trajectories](https://proceedings.mlr.press/v206/riou-durand23a.html). Proceedings of Machine Learning Research, 206, 8102-8116.

9. Cerullo, E., Jones, H. E., Carter, O., Quinn, T. J., Cooper, N. J. and Sutton, A. J. (2022). [Meta-analysis of dichotomous and ordinal tests with an imperfect gold standard](https://doi.org/10.1002/jrsm.1567). Research Synthesis Methods, 13(5), 595-611.

10. Cerullo, E., Pinkney, S., Sutton, A. J., Lucas, T., Cooper, N. J. and Jones, H. E. (2025). [Latent class multivariate probit and latent trait models for evaluating test accuracy without a gold standard: A simulation study](https://arxiv.org/abs/2509.18489). arXiv:2509.18489.

11. Hoffman, M. D. and Gelman, A. (2014). [The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo](https://www.jmlr.org/papers/v15/hoffman14a.html). Journal of Machine Learning Research, 15(47), 1593-1623.

12. Margossian, C. C., Hoffman, M. D., Sountsov, P., Riou-Durand, L., Vehtari, A. and Gelman, A. (2024). [Nested R-hat: Assessing the convergence of Markov chain Monte Carlo when running many short chains](https://arxiv.org/abs/2110.13017v6). Bayesian Analysis; arXiv:2110.13017, version 6.

13. Roualdes, E. A., Ward, B., Carpenter, B., Seyboldt, A. and Axen, S. D. (2023). [BridgeStan: Efficient in-memory access to the methods of a Stan model](https://doi.org/10.21105/joss.05236). Journal of Open Source Software, 8(87), 5236.

14. Qu, Y., Tan, M. and Kutner, M. H. (1996). [Random effects models in latent class analysis for evaluating accuracy of diagnostic tests](https://doi.org/10.2307/2533043). Biometrics, 52(3), 797-810.

15. Xu, H. and Craig, B. A. (2009). [A probit latent class model with general correlation structures for evaluating accuracy of diagnostic tests](https://doi.org/10.1111/j.1541-0420.2008.01194.x). Biometrics, 65(4), 1145-1155.

16. Xu, H., Black, M. A. and Craig, B. A. (2013). [Evaluating accuracy of diagnostic tests with intermediate results in the absence of a gold standard](https://doi.org/10.1002/sim.5695). Statistics in Medicine, 32(15), 2571-2584.

17. Uebersax, J. S. (1999). [Probit latent class analysis with dichotomous or ordered category measures: Conditional independence/dependence models](https://doi.org/10.1177/01466219922031400). Applied Psychological Measurement, 23(4), 283-297.

18. Cotter, S. L., Roberts, G. O., Stuart, A. M. and White, D. (2013). [MCMC methods for functions: Modifying old algorithms to make them faster](https://doi.org/10.1214/13-STS421). Statistical Science, 28(3), 424-446; [arXiv:1202.0709](https://arxiv.org/abs/1202.0709).

19. Talhouk, A., Doucet, A. and Murphy, K. (2012). [Efficient Bayesian inference for multivariate probit models with sparse inverse correlation matrices](https://doi.org/10.1080/10618600.2012.679239). Journal of Computational and Graphical Statistics, 21(3), 739-757.

20. Johnston, C. K., Waterhouse, T., Wiens, M., Mondick, J., French, J. and Gillespie, W. R. (2024). [Bayesian estimation in NONMEM](https://doi.org/10.1002/psp4.13088). CPT: Pharmacometrics & Systems Pharmacology, 13, 192-207.

21. Kingma, D. P. and Ba, J. (2015). [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980). International Conference on Learning Representations (ICLR); arXiv:1412.6980.

22. Pinkney, S. (2024). [A Short Note on a Flexible Cholesky Parameterization of Correlation Matrices](https://arxiv.org/abs/2405.07286). arXiv:2405.07286.

23. Kastner, G. and Frühwirth-Schnatter, S. (2014). [Ancillarity-sufficiency interweaving strategy (ASIS) for boosting MCMC estimation of stochastic volatility models](https://doi.org/10.1016/j.csda.2013.01.002). Computational Statistics & Data Analysis, 76, 408-423.


<!-- ------------------------------------------------------------------------------------------------------------------------------- -->
## Package citation and development
<!-- ------------------------------------------------------------------------------------------------------------------------------- -->


NicoStan has one author, who is also its only contributor (see [How to cite NicoStan](#how-to-cite-nicostan)),
and is licensed under GPL-3 (see [LICENSE](LICENSE)).
Package citation metadata is provided in [CITATION.cff](CITATION.cff),
and the references above are available as a [BibTeX file](docs/references.bib).
