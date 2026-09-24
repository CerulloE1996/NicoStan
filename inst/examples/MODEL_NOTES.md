# General Stan examples: models, nuisance blocks and priors

The nine supplied filenames represent seven distinct models. The two Cox-frailty filenames are aliases, as are the two Weibull filenames. A tenth file, `Latent_diffusion_survival.stan`, is included for the example benchmarks. Each filename has a corresponding `_avx.stan` version. The plain and AVX versions use the same data, priors and initial values.

| Model | First parameter block | Automatically detected nuisance dimension | Main parameters used for ESS |
| --- | --- | --- | --- |
| Cox shared frailty | `log_frailty_raw[J]` | `J` | Regression coefficients and frailty scale |
| Hierarchical logistic regression | `beta_raw[K,2]` | `2*K` | Population means and scales |
| Joint longitudinal-survival model | `b_raw[2,N_subj]` | `2*N_subj` | Fixed effects, scales, random-effect correlation, baseline hazard and association |
| Stochastic volatility (discrete-time AR(1)) | `h_std[T]` | `T` | `mu`, `phi`, `sigma` |
| Latent diffusion survival | `w[M]` (Brownian increments) | `M` (50 in the examples) | `drift_sin_coefficient`, `drift_constant`, `hazard_scale` |
| Weibull survival regression | Shape parameter; no nuisance block | `0` | Shape, intercept and regression coefficients |
| Robust Student-t regression | Intercept; no nuisance block | `0` | Intercept, regression coefficients and residual scale |
| Marginal Gaussian process | Covariance hyperparameters; no nuisance block | `0` | Length scale, signal scale and observation-noise scale |

For nuisance models, the first declaration contains the complete standard normal reference block. NicoStan identifies the declaration from compiler metadata and counts its unconstrained BridgeStan coordinates. No manual dimension is supplied by the examples. Models without a nuisance block use `sample_nuisance = FALSE`.

## Model corrections

- **Cox shared frailty:** retained the Cox risk-set partial likelihood and its Breslow treatment of ties. Removed the unused malformed parallel helper. The simulator supplies one event or censoring time per subject and a shared group frailty. Risk-set contributions are dependent; they are not independent subject-level likelihood terms for ordinary leave-one-out scoring.
- **Weibull:** event times contribute a Weibull density and censored times contribute the survival probability. The AVX version writes the same proportional-hazards log likelihood explicitly.
- **Hierarchical logistic regression:** changed the group coefficients to the equivalent non-centred representation `beta = mu + sigma * beta_raw` and placed `beta_raw` first. Updated the array syntax. The model still contains a group intercept and slope.
- **Joint longitudinal-survival model:** replaced the use of `X_long[1]` for every subject and survival time with subject-specific `X_long_event` and `X_long_quad` inputs. These evaluate the fixed-effect design at the event/censoring time and each quadrature time. Correlated random intercepts/slopes use a standard normal first block. The example uses finite follow-up; extrapolating a linearly declining associated trajectory indefinitely can imply a positive probability of surviving forever.
- **Gaussian process:** the marginal likelihood integrates out the latent function, so this example has no explicit nuisance block. The prediction calculation now includes the fixed diagonal variance used by the fitted model. The name `sigma_intercept` was changed to `fixed_nugget_sd`, because the term is independent diagonal noise rather than a shared intercept. The model requires at least two observations and nonconstant predictors/outcomes because it standardises both.
- **Stochastic volatility (discrete-time AR(1)):** placed the innovation vector first, required `T >= 1`, and guarded the recursion for `T = 1`. The AR(1) initial state uses its stationary variance.
- **Latent diffusion survival:** the model of Beskos, Kalogeropoulos and Pazos (2013, Sections 5 and 6.3), after Roberts and Sangalli (2010). All subjects share one hazard path `h(u) = hazard_scale * (x(u)^2 + x_squared_hazard_offset)`, where `dX = -(drift_sin_coefficient * sin(X) + drift_constant) du + sigma_x dB`, `X_0 = x_0`, on `[0, t_max]`. Beskos et al. use `(1.4, 1)`, `hazard_scale = 1`, unit diffusion coefficient, `X_0 = 2` and `t_max = 1`, and hold them fixed; here the drift coefficients and the hazard scale are estimated jointly with the path, while `sigma_x` and `x_0` are data. The path uses a non-centred Euler-Maruyama discretisation on `M` steps, with the standard normal increments `w` first. The hazard at an event time uses the linearly interpolated path; the integrated hazard is the trapezoid rule on the grid plus a partial trapezoid up to the event/censoring time. Right-censored subjects contribute `-H(t)`. The examples fix `M = 50` (step 0.02, the only step Beskos et al. report, which is for their diffusion-bridge experiment; they give none for the survival experiment). The hazard scale is identified only through the fixed `x_0` and the fixed diffusion coefficient; in the validation data (N = 200) its posterior SD is about 0.19. The simulator draws a fine path (20 sub-steps per model step) and exact event times given that path, using the same hazard offset as the model. The AVX version replaces only the vector log of `x^2 + x_squared_hazard_offset` at the event times.
- **Latent diffusion survival: differences from Beskos et al.:**
  - *Why:* the first version diverged in both engines at adapt_delta 0.9 (cmdstanr: 128 divergences in 4000 draws, min ESS 80, R-hat 1.034; NicoStan: 64 divergences). The diagnosis used cmdstanr, 4 x (1000 + 1000), on the example validation data (`make_NicoStan_example(N = 200, seed = 123)`). Scripts, logs and fits are in `BayesMVP_migration_2026_09_21/audit/claude_2026_09_22/validation_round4/LDS/`.
  - *Cause 1, the main one: the `x^2` hazard at `x = 0`.* The log hazard at an event time is `2 log|x(t)|`, which is `-Inf` at `x(t) = 0`, and `h(x) = h(-x)`. The drift pulls X towards its stable point -0.795. The drift-only path from `x_0 = 2` reaches 0 at t = 1.07, and about 60% of simulated paths cross 0 before t = 1. So the posterior splits into sign patterns of the path at the late, sparse event times, and these are separated by `-Inf` walls. Trajectories diverge near the walls and cannot cross them. In the old model one chain spent 30% of its draws in the "path negative at the last event" pattern, while the other three never visited it (x_path R-hat 1.11, min ESS 23). With every parameter fixed at the Beskos et al. values, i.e. their exact model, cmdstanr still gave 43 divergences and x_path R-hat 1.53: one chain sat entirely in the other sign pattern. Fixing `hazard_scale = 1` did not help (227 divergences), and neither did tighter drift priors alone (80). Replacing the linear interpolation or refining the Euler step would not remove the wall either, because the wall is in the hazard function itself.
  - *Fix 1:* `h(u) = hazard_scale * (x(u)^2 + x_squared_hazard_offset)`, with the offset as data (0.01 in the simulator and in the examples; 0 gives the exact Beskos et al. form, and its divergences). The log hazard is then finite and smooth, so the sign patterns are connected. In a test with `hazard_scale` fixed at 1 and the old `normal(0, 2)` drift priors, on the old data, the offset alone cut cmdstanr's divergences from 227 to 0, with x_path R-hat 1.002 (offset 0.01) or 1.003 (offset 0.1). The data are simulated with the same offset, so the model is correctly specified.
  - *Cause 2, the drift ridge.* One path on [0, 1] carries little information about the drift. With `normal(0, 2)` priors the posterior SDs were about 1.5, close to the prior, with correlation -0.7 between the two coefficients. With the offset but these priors, cmdstanr at adapt_delta 0.8 still gave 1-4 divergences at N = 100-200 and 49 at N = 800. At N = 800, 30 of the 49 came from the 5% of draws with `drift_sin_coefficient <= -1.5`, the far tail of the ridge.
  - *Fix 2:* `drift_sin_coefficient ~ normal(1.4, 1)`, `drift_constant ~ normal(1, 1)`. These relax Beskos et al.'s "drift known at (1.4, 1)" to "drift known to within about 1". The alternative `normal(0, 1)` also gave 0 divergences in the same runs (N = 200 and 800, adapt_delta 0.8), but it pulls `drift_sin_coefficient` towards 0 (posterior mean 0.6 against 1.4).
  - *Unchanged:* `hazard_scale` stays estimated with its `lognormal(0, 1)` prior. `M = 50`, the Euler-Maruyama recursion, the linear interpolation at event times, the trapezoid integrated hazard, `x_0 = 2`, `sigma_x = 1` and `t_max = 1` are all as before. The non-centred increments `w` are still the first parameter block.
  - *Result:* see "Latent diffusion survival validation" in `README_benchmarks.md`.
  - *Residual caveat:* the likelihood still depends on the path only through `x^2`, so the sign of the path after it nears zero is identified only by the prior. The sign patterns are now connected, but how fast a chain moves between them depends on the data set: it is slow when late events sit near the zero crossing. `drift_constant` is correlated with that sign, and was the slowest-mixing main parameter in every run. Sign-invariant quantities (`hazard_path`, `survival_path`) are the meaningful path summaries; `x_path` R-hat near the end of the window can be above 1.01 even when these are fine.

## Prior choices and changes

These priors are calibrated to the units used by the synthetic examples. In particular, the GP standardises its inputs/outcomes, the survival examples use their simulated time scale, and the stochastic-volatility example uses returns in its specified units.

| Model | Prior choices |
| --- | --- |
| Cox shared frailty | `beta ~ normal(0,2)`; frailty scale `~ exponential(1)`; standard normal raw frailties |
| Weibull | Shape `~ lognormal(0,0.5)`; `beta ~ normal(0,1)`; intercept `~ normal(-1,1.5)` |
| Robust Student-t | Fixed four degrees of freedom; intercept `~ normal(0,10)`; coefficients `~ normal(0,2.5)`; residual scale `~ exponential(1)` |
| Hierarchical logistic | Population means `~ normal(0,2)`; positive group scales `~ normal(0,2)`; standard normal raw coefficients |
| Joint longitudinal-survival | Fixed coefficients `~ normal(0,2)`; residual/random-effect scales `~ exponential(1)`; random-effect correlation `~ uniform(-1,1)`; baseline hazard `~ lognormal(log(0.2),1)`; association `~ normal(0,1)` |
| Marginal GP | Length scale `~ inv_gamma(5,5)`; positive signal/noise scales `~ normal(0,1)`; fixed nugget SD `0.05` in standardised outcome units |
| Stochastic volatility (discrete-time AR(1)) | `phi ~ uniform(-1,1)`; positive innovation scale `~ normal(0,0.5)`; level `~ normal(0,2)` |
| Latent diffusion survival | `drift_sin_coefficient ~ normal(1.4,1)`, `drift_constant ~ normal(1,1)`; hazard scale `~ lognormal(0,1)`; standard normal Brownian increments; fixed hazard offset `0.01` |

The prior revisions were:

- Stochastic-volatility scale: `cauchy(0,5)` to `normal(0,0.5)` on its positive support.
- Stochastic-volatility level: `cauchy(0,10)` to `normal(0,2)`.
- Weibull shape: `exponential(1)` to `lognormal(0,0.5)`.
- Weibull coefficients: `normal(0,2)` to `normal(0,1)`.
- Weibull intercept: `normal(0,5)` to `normal(-1,1.5)`.
- Joint-model baseline hazard: `exponential(0.1)` to `lognormal(log(0.2),1)`.
- Latent diffusion survival drift coefficients: `normal(0,2)` each to `normal(1.4,1)` and `normal(1,1)`, centred at the Beskos et al. values; see the model note above.

The other listed priors were retained. These revisions apply to the new general examples, and do not alter the archived diagnostic-test simulation study.

## Mathematical and numerical checks

The validation harness checks:

- Complete log densities against independent R implementations, including the Jacobians for constrained parameters and the corresponding non-centred representations.
- BridgeStan gradients against central finite differences, and the native NicoStan evaluator against BridgeStan.
- Plain/AVX log densities and gradients at the initial state and five perturbed unconstrained states.
- The actual SIMD lane count returned by the compiled AVX model.
- The first parameter declaration, detected nuisance dimension and main-parameter selection.
- Complete warm-up, sampling and posterior-summary execution.

Efficiency calculations select only the explicit main-parameter list above. Generated quantities, nuisance coordinates and padding are excluded. The benchmark tables retain convergence diagnostics alongside efficiency values. The gradient-work comparison follows the APMS convention: `1000 * min(ESS) / gradient_work`, using the recorded work proxy specified for each sampler.
