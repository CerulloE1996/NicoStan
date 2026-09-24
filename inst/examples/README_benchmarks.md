# NicoStan example benchmarks

The benchmark launcher compares the same simulated data, priors, user-provided initial values, and chain seeds across:

- CmdStanR with the plain Stan model;
- NicoStan with the plain Stan model; and
- NicoStan with the custom AVX-512 Stan functions.

AVX-2 is available as an optional fourth arm when the host supports it. The default size grid contains three values for every model, with smaller sizes for the joint longitudinal-survival example and larger sizes for the inexpensive regression examples. The grid is editable through `NicoStan_benchmark_registry()` or by passing a named `N_grid` list to `NicoStan_benchmark_run()`.

Each completed arm is recorded under a configuration fingerprint. A later call with the same model, size, arm, seed, sampler settings, Stan source hash, and benchmark harness hash reuses the saved result. A partial run can therefore be resumed after a compilation or sampling failure.

The fingerprint also includes the benchmark orchestration source, installed package RDB/RDX/shared-library hashes, AVX header hashes, CmdStan path and version, R C++17 configuration, compiler environment, and the exact compiler flags. Results are stored below a fingerprint-specific directory, and a saved record is reused only when its embedded fingerprint agrees with the manifest.

The ESS summaries use only the explicitly declared non-nuisance parameters in `example$main_parameters`. The benchmark writes:

- fitting time and post-warmup sampling time;
- minimum bulk and tail ESS over the main parameters;
- minimum main-parameter ESS per second of sampling and fitting;
- maximum main-parameter R-hat and divergences;
- nested R-hat for NicoStan when the sampling design has replicated starting-state groups;
- minimum main-parameter ESS per 1,000 units of gradient work; and
- posterior agreement summaries against CmdStanR.

For NicoStan, gradient work is the APMS-compatible estimate based on adapted trajectory length, iterations, and chains. For CmdStanR, it is the sum of `n_leapfrog__` during sampling. These are labelled proxies, and the exact number of gradient evaluations is retained as `NA` until an exact counter is available. The time to ESS 100, 1,000, and 10,000 columns are linear extrapolations from the observed minimum main-parameter ESS per second, so their diagnostic status remains visible beside them.

The initial-value identity check refers to the user-provided input lists before adaptation. NicoStan then constructs post-burn-in sampling starts from its burn-in endpoints, so those adapted starts are not claimed to be identical to CmdStanR's direct starts. The NicoStan benchmark uses fewer superchains than sampling chains so nested R-hat compares replicated chains that share a recorded burn-in endpoint group. CmdStanR has no corresponding nested-starting-state diagnostic, so its field is recorded as not applicable. Four independent chains are not labelled as a replicated nested-R-hat design.

Sampling efficiency is calculated only when the engine reports a positive post-warmup sampling interval. NicoStan uses the unrounded `result$time_sampling` interval around its native sampling call; CmdStanR uses `fit$time()$chains$sampling`. A missing timer is recorded as `NA` with an explicit basis, rather than being replaced by total fitting time.

The `Run_*.R` files in this directory are benchmark drivers. Each has an editable `N_values` vector near the top and runs all three sample sizes. For example, after installing NicoStan:

```r
source("path/to/NicoStan/inst/examples/Run_robust_t4_regression.R")
```

The drivers use the analysis profile by default. Set `NICOSTAN_EXAMPLE_PROFILE=smoke` for a shorter execution check. On AVX2-only hardware, the driver replaces the unavailable AVX-512 comparison with an explicitly labelled AVX2 arm. On hardware without either instruction set, it runs the two plain-model arms. The generic `NicoStan_benchmark_run()` function also accepts a custom arm table and `include_avx2 = TRUE`.

| Model | Three default sample sizes |
| --- | --- |
| Cox shared frailty | 100, 250, 500 observations |
| Weibull survival | 100, 250, 500 observations |
| Robust Student-t regression | 250, 1000, 3000 observations |
| Hierarchical logistic regression | 300, 1200, 4000 observations |
| Joint longitudinal-survival | 25, 75, 200 subjects |
| Marginal Gaussian process | 40, 80, 160 observations |
| Stochastic volatility (discrete-time AR(1)) | 250, 1000, 4000 time points |
| Latent diffusion survival | 100, 200, 800 subjects (50 path increments for every size) |

### Latent diffusion survival validation

The model was changed: a hazard offset of 0.01 and drift priors centred at the Beskos et al. values. The old model diverged in both engines. See `MODEL_NOTES.md` for what changed and why. Earlier LDS benchmark or validation results are for the old model and should not be mixed with new ones. The benchmark runner's resume check now refuses a saved run whose `.stan` sha256 differs from the current file.

Validation with `alg_paper_3_validate_estimates.R`, run unchanged apart from command-line options:

- Data: N = 200, data seed 123.
- cmdstanr: 4 x (1000 + 1000).
- NicoStan: CHESSR, 16 sampling chains.
- Main parameters: `drift_sin_coefficient`, `drift_constant` and `hazard_scale`.
- z = (mean_NicoStan - mean_cmdstanr) / sqrt(MCSE_NicoStan^2 + MCSE_cmdstanr^2).

| Run | Engine | Divergences | min ESS (bulk) | max R-hat | max abs(z) |
| --- | --- | --- | --- | --- | --- |
| adapt_delta 0.9, NicoStan 16 x 250 | cmdstanr | 0 | 822 | 1.002 | 0.83 |
| | NicoStan | 0 | 166 (`drift_constant`) | 1.064 | |
| adapt_delta 0.8, NicoStan 16 x 250 | cmdstanr | 0 | 890 | 1.003 | 2.24 (`hazard_scale`) |
| | NicoStan | 3 | 557 | 1.027 | |
| adapt_delta 0.9, NicoStan 16 x 1000 | cmdstanr | 0 | 822 | 1.002 | 0.91 |
| | NicoStan | 0 | 1177 | 1.013 | |
| Before the change: adapt_delta 0.9, round 3 | cmdstanr | 128 | 80 | 1.034 | 1.09 |
| | NicoStan | 64 | 1375 | 1.012 | |

For the new model, no |z| exceeds 3.

With 250 draws per chain, NicoStan's `drift_constant` has not mixed well (R-hat 1.064 at adapt_delta 0.9). The cause is the path-sign mixing described in `MODEL_NOTES.md`. With 1000 draws per chain it reaches R-hat 1.013 and ESS 1177.

The old NicoStan ESS of 1375 was not a better result. Its chains could not cross the zero-hazard walls, so they never moved between the sign patterns and explored less of the posterior.

Extra cmdstanr checks of the new model: 4 x (1000 + 1000), same diagnostic script as the diagnosis.

| N | Data seed | adapt_delta | Divergences |
| --- | --- | --- | --- |
| 100 | 123 | 0.8 | 0 |
| 100 | 2026 | 0.8 | 0 |
| 200 | 7 | 0.8 | 0 |
| 200 | 123 | 0.9 | 0 |
| 800 | 123 | 0.9 | 0 |
| 200 | 123 | 0.8 | 0 (test copy of the final model, identical apart from its comments) |
| 800 | 123 | 0.8 | 0 (test copy of the final model, identical apart from its comments) |

In every one of these runs, main-parameter R-hat was at most 1.01.

The output directory contains `benchmark_summary.csv`, `efficiency_comparisons.csv`, `posterior_agreement.csv`, `comparison_checks.csv`, and the resumable `benchmark_manifest.csv`.

## Discrete-time stochastic-volatility validation

These are previously saved checks of `Stochastic_volatility_discrete_time.stan`, using the non-centred AR(1) path and the priors in `MODEL_NOTES.md`.
The plain model SHA256 is `ccd3b69909d3e2e832aaf926569fef1de7251837481567b1dc87beab6326b45f`.
Each row uses four chains; warmup and sampling counts below are per chain. R-hat is the maximum across `mu`, `phi` and `sigma`.

| Data/run seed | N (time points) | Warmup | Sampling | adapt_delta | Engine / maths | Divergences | Maximum main R-hat |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026 | 60 | 500 | 500 | 0.99 | CmdStanR NUTS / Stan | 9 | 1.012822 |
| 2026 | 60 | 500 | 500 | 0.99 | NicoStan / Stan | 0 | 1.006198 |
| 2026 | 60 | 500 | 500 | 0.99 | NicoStan / AVX-512 | 8 | 1.004116 |
| 9201 | 60 | 1000 | 1000 | 0.999 | CmdStanR NUTS / Stan | 0 | 1.000776 |
| 9201 | 60 | 1000 | 1000 | 0.999 | NicoStan / Stan | 0 | 1.002802 |

Within the seed-2026 comparison, the data, initial values and plain model hashes match.
The seed-9201 check uses different simulated data and longer warmup/sampling; it is not a controlled comparison isolating `adapt_delta`.
These results show that this non-centred discrete-time example can produce divergences even at `0.99`;
they do not establish a threshold that is necessary or sufficient for all data/parameter regimes, or a general speed-up.
The AVX-512 row is retained alongside the plain NicoStan result so that the check is reported in full.

The [saved-check metadata](validation/stochastic_volatility_discrete_time_checks.json) records the configurations, hashes and diagnostic values.
The model's regime-dependent parameterisation issues are discussed by
[Kastner and Frühwirth-Schnatter (2014)](https://doi.org/10.1016/j.csda.2013.01.002); their study is not a NUTS comparison.
