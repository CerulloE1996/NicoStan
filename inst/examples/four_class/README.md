# Four-class LC-MVOP example

This is a self-contained synthetic example for the four-class joint latent-class
multivariate ordinal probit model. It contains no clinical data and does not need
the APMS project at runtime.

- `models/four_class_plain.stan` is the ordinary Stan model.
- `models/four_class_avx.stan` is the AVX special-function variant.
- `models/four_class_avx_fused.stan` is the optional fused variant.
- `fixture/N24`, `fixture/N48`, and `fixture/N96` contain the same synthetic
  model at three editable sample sizes. Each fixture is stored as text in
  `fixture.R`, so the installed example does not depend on RDS files.
- `run_four_class_example.R` runs CmdStanR plain, NicoStan plain, or NicoStan
  AVX512 for one of those sample sizes.
- `run_four_class_benchmark_grid.R` runs all three arms at all three sample sizes
  and writes a CSV to a user supplied writable output directory.

The first parameter declaration is the `u_raw` augmented nuisance block. NicoStan
detects its unconstrained dimension automatically when `sample_nuisance = TRUE`.
The runner never passes `n_nuisance_override`.

The supplied N=24 fixture has 163 unconstrained coordinates: 96 nuisance
coordinates from `u_raw` and 67 main coordinates. The N=48 and N=96 fixtures have
192 and 384 nuisance coordinates respectively, with the same 67-coordinate main
block.

N=24 is the base synthetic fixture. N=48 and N=96 are deterministic bootstrap
resamples of its subject rows, generated with `set.seed(2026 + N)` so that the
three executable benchmark sizes share the same synthetic model contract. They
are benchmark fixtures rather than independent simulation-study replicates.

Run the three arms from this directory with:

```sh
Rscript run_four_class_example.R cmdstanr Stan 24
Rscript run_four_class_example.R NicoStan Stan 24
Rscript run_four_class_example.R NicoStan AVX512 24
```

To run the complete three-arm by three-size grid and keep all compiled libraries
outside the installed example directory:

```sh
Rscript run_four_class_benchmark_grid.R /tmp/nicostan_four_class_benchmarks
```

The source-friendly `Run_4LC_MVOP.R` entrypoint accepts a selected subset when
called explicitly, for example `run_4LC_MVOP(N_values = c(24L, 96L))`. The
selection is passed to the child grid runner and the entrypoint stops if that
child exits unsuccessfully.

The same commands work with `48` and `96`. Each run uses two chains, 60 warmup
iterations, and 80 sampling iterations as a short executable smoke check. The
validation harness measures ESS and R-hat over the explicit main-parameter set;
the `u_raw` nuisance block is excluded from those summaries.

The benchmark CSV records sampling and total ESS/sec, divergences, the main only
minimum ESS, and `1000 * min(ESS) / gradient_work`. The gradient-work field is
labelled with its arm specific proxy: CmdStanR uses the sum of `n_leapfrog__`,
while NicoStan uses adapted main trajectory length times iterations and chains.

The analysis grid sets `Phi_type = 2` for all three arms, which activates the
vectorized approximation branch used by the AVX source. A `Phi_type = 1` run is
an exact-CDF baseline: the AVX model compiles with the external header, but its
custom vector functions are bypassed on that branch. Each CSV record reports
`phi_type` and `avx_kernel_path_active` explicitly.

The analysis grid also records `metric_type_main` and `M_decay_type`. Its NicoStan
arms use the empirical main metric with inverse mass decay, including the guarded
fallback that preserves the last valid metric when an early moment proposal is
undefined. CmdStanR records its NUTS metric separately.

The AVX run expects an installed BayesMVP extension package providing
`BayesMVP/stan_external_functions.hpp`. The fused source is included for separate
compilation and comparison; it is not silently substituted into the ordinary AVX
run.

The APMS helper used to generate the fused source is a build-time development
tool. The generated Stan files and synthetic fixtures in this directory are the
runtime inputs.
