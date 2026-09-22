# NicoStan examples and local installation

This is the source directory for the NicoStan examples and administrator installer. Edit and run the examples here. The installer copies this directory into the compiled package, where it is available through `system.file("examples", package = "NicoStan")`.

- **Install/update NicoStan:** in a fresh R session, source `NicoStan_admin_install.R`, then restart R and load `library(NicoStan)`.
- **Short introduction:** source `random_intercepts.R` and call `run_random_intercepts()`. Its standard normal nuisance vector is declared first, and its dimension is detected automatically.
- **General models:** `models/` contains all nine supplied Stan filenames and their nine `_avx.stan` counterparts. The duplicate Cox-frailty and Weibull filenames refer to the same respective models. `Latent_diffusion_survival.stan` and its `_avx.stan` version (Beskos et al., 2013) were added for the Paper 3 benchmarks; `Run_Latent_diffusion_survival.R` runs it.
- **Model definitions and priors:** see `MODEL_NOTES.md` for the nuisance/main split, model corrections, exact prior changes and validation checks.
- **Native multivariate probit examples:** `Run_MVP.R`, `Run_2LC_MVP.R`, `Run_MVOP.R`, `Run_2LC_MVOP.R` and `Run_latent_trait.R` use the BayesMVP extension and run three editable sample sizes. Their ESS summaries exclude the nuisance block.
- **Four-class ordinal example:** `four_class/` contains the plain, AVX and fused 4LC-MVOP models, synthetic fixtures and a three-size comparison runner.
- **Benchmark functions:** `paper3_examples.R` runs an individual arm; `paper3_benchmarks.R` runs the three sample sizes and compares CmdStanR, NicoStan and NicoStan with AVX-512. AVX2 is available on supported CPUs.

## Install from the local source

```r
source("path/to/NicoStan/inst/examples/NicoStan_admin_install.R")
```

After the installer completes, restart R:

```r
library(NicoStan)
source(system.file("examples", "random_intercepts.R", package = "NicoStan"))
fit <- run_random_intercepts(burnin_algorithm = "CHESSR")
fit$summary(save_log_lik_trace = FALSE)
```

The administrator script automatically finds the outer NicoStan source directory. `NICOSTAN_SOURCE_ROOT` selects a different checkout, and `NICOSTAN_INSTALL_LIB` selects a different R library.
