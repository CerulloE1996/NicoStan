# Trajectory-length adaptation in NicoStan

NicoStan provides `burnin_algorithm = "ChEES"`, `"CHESSR"`, `"CHESSR_log"` and `"SNAPER"`. The code also accepts `"KE"`, a kinetic-energy criterion of our own; it is experimental (unpublished and not recommended) and is kept only for comparison. The trajectory-length criteria use the main parameters. The nuisance block is excluded from these criteria, including when the transition jointly updates both blocks.

## Positive trajectory lengths and the log parameterisation

Let the trajectory-length scale be $\bar\tau>0$ and define

$$
\eta=\log\bar\tau,\qquad \bar\tau=\exp(\eta).
$$

This is a one-to-one reparameterisation. For any differentiable objective $J$,

$$
\frac{\partial J}{\partial\eta}
=\frac{\partial J}{\partial\bar\tau}\frac{\partial\bar\tau}{\partial\eta}
=\bar\tau\frac{\partial J}{\partial\bar\tau}.
$$

Therefore, optimising the same objective in $\eta$ or $\bar\tau$ gives the same interior stationary points. An additive update in $\eta$ is a multiplicative update in $\bar\tau$:

$$
\eta_{k+1}=\eta_k+\delta_k
\quad\Longleftrightarrow\quad
\bar\tau_{k+1}=\bar\tau_k\exp(\delta_k).
$$

Algorithm 1 in [Sountsov and Hoffman (2022)](https://arxiv.org/abs/2110.11576v3) already adapts the trajectory length on the log scale. Both NicoStan options, `CHESSR` and `CHESSR_log`, do this. The extra `log` in our option name refers to the log-rate objective and its normalisation, rather than the positive parameterisation alone.

## ChEES and ChEES-R

For the current main-parameter mass matrix $M$, choose $B$ such that $B^\mathsf{T}B=M$, and write

$$
z=B(\theta-\mu),\qquad f(z)=\frac12\lVert z\rVert^2.
$$

For chain $i$, let $z_i'$ be the proposal endpoint, $\alpha_i$ its acceptance probability and

$$
C_i=\{f(z_i')-f(z_i)\}^2.
$$

- ChEES uses the expected squared change, estimated by $m^{-1}\sum_i\alpha_i C_i$.
- ChEES-R uses the corresponding rate, estimated by $m^{-1}\sum_i\alpha_i C_i/\tau_i$.
- The endpoint velocity is transformed by the same $B$. Since this velocity already equals $d\theta/dt$, it does not receive a second multiplication by $M^{-1}$.

These criteria follow [Hoffman et al. (2021)](https://proceedings.mlr.press/v130/hoffman21a.html) and [Sountsov and Hoffman (2022)](https://arxiv.org/abs/2110.11576v3), applied to the main block in the coordinates above.

Define $G_i=\tau_i\,\partial C_i/\partial\tau_i$. With $\tau_i=u_i\bar\tau$, the pathwise derivative with respect to $\eta$ is

$$
\frac{\partial}{\partial\eta}\frac{C_i}{\tau_i}
=\frac{G_i-C_i}{\tau_i}.
$$

The `CHESSR` update uses the mean of $\alpha_i(G_i-C_i)/\tau_i$. Acceptance probabilities and the current centring/metric estimates are held fixed when computing this endpoint derivative.

## Our log-rate formulation

For a shared trajectory length, write $A(\bar\tau)=\mathbb E[\alpha C]$ and $R(\bar\tau)=A(\bar\tau)/\bar\tau$. Then

$$
\frac{\partial R}{\partial\eta}=A'(\bar\tau)-\frac{A(\bar\tau)}{\bar\tau},
$$

and, whenever $A>0$,

$$
\frac{\partial\log R}{\partial\eta}
=\frac{\bar\tau A'(\bar\tau)}{A(\bar\tau)}-1
=\frac{1}{R}\frac{\partial R}{\partial\eta}.
$$

Consequently, taking the logarithm preserves the positive objective's stationary points and the sign of its gradient. It rescales the gradient, so finite learning-rate updates and ADAM trajectories can differ.

`CHESSR_log` estimates this derivative using

$$
g_k=\frac{m^{-1}\sum_i\alpha_iG_i}{s_k}-1,
\qquad
s_k=0.9s_{k-1}+0.1\left(m^{-1}\sum_i\alpha_iC_i\right).
$$

The running average starts at the first positive, finite numerator estimate. Zero-acceptance proposals contribute zero to the numerator while remaining in the number of chains used in the average. This preserves the acceptance weighting.

When trajectory lengths are randomised during adaptation, the two options also use different aggregation:

- `CHESSR` estimates $\mathbb E[\alpha C/\tau]$.
- `CHESSR_log` normalises the numerator of $\mathbb E[\alpha C]/\bar\tau$ using the running average above.

In general, $\mathbb E[\alpha C/\tau]\ne\mathbb E[\alpha C]/\bar\tau$. Their fixed-length relationship therefore does not imply identical objectives under trajectory jitter. NicoStan makes both options available for empirical comparison.

## SNAPER and phase defaults

SNAPER uses $f(z)=(w^\mathsf{T}z)^2$, with a unit direction $w$ learned during warm-up. The direction and endpoints use the same mass-matrix coordinates. NicoStan updates the direction with an Oja iteration and uses the resulting squared-change rate for trajectory adaptation.

The default phase settings are:

- `randomize_tau_burnin = FALSE`: fixed length within each transition at the current tuning setting; adaptation changes the setting between iterations.
- `randomize_tau_sampling = TRUE`: randomised post-burn-in trajectory lengths, with at least one integration step.
- `manual_tau` controls whether the trajectory-length scale is adapted.

## Rescaling tau for randomised sampling (`tau_sampling_scale`)

**Provenance.** This option is our own heuristic, introduced on 2026-09-22. It is not a published method. It rests on the exact-dynamics Gaussian analysis in [Hoffman et al. (2021)](https://proceedings.mlr.press/v130/hoffman21a.html) and on the $\mathrm{Unif}(0,2\bar\tau)$ jitter discussed by [Sountsov and Hoffman (2022)](https://arxiv.org/abs/2110.11576v3). The two optima and their ratios below are our own derivation.

**Why.** By default NicoStan adapts $\tau$ with a fixed length during burn-in (`randomize_tau_burnin = FALSE`). This is deliberate: the burn-in chains run in lockstep, so per-chain jitter would make every iteration wait for the longest chain. Sampling then draws $\tau_i\sim\mathrm{Unif}(0,2\bar\tau)$ with $\bar\tau$ equal to the adapted value (`randomize_tau_sampling = TRUE`). The fixed-length optimum is not the optimal mean of a jittered length, so the adapted $\bar\tau$ is not the value the same criterion would pick for the jittered sampler.

**Derivation.** Take a unit Gaussian with exact dynamics. A trajectory of length $t$ gives $\theta'=\theta\cos t+r\sin t$, and ChEES (with its factor of $1/4$) is $\sin^2 t$. The kinetic-energy criterion `KE` gives the same value there. With $\tau\sim\mathrm{Unif}(0,2T)$,

$$
\mathbb E[\sin^2\tau]=\frac12-\frac{\sin 4T}{8T}.
$$

The rate criteria divide by the trajectory length. `CHESSR` and SNAPER, as coded, average $C_i/\tau_i$ over chains, which gives $\mathbb E[\sin^2\tau/\tau]=(2T)^{-1}\int_0^{2T}\sin^2 t/t\,dt$. `CHESSR_log` uses the ratio of means, $\mathbb E[\sin^2\tau]/T$. The first maxima are:

| Criterion | Fixed-length optimum | Jittered-mean optimum | Factor |
|---|---|---|---|
| `KE`, `ChEES` | $\pi/2=1.5708$ | $1.1234$ | $0.7151$ |
| `CHESSR`, `SNAPER` (mean of $C_i/\tau_i$) | $1.1656$ | $0.8950$ | $0.7678$ |
| `CHESSR_log` ($\mathbb E[C]/\bar\tau$) | $1.1656$ | $\pi/4=0.7854$ | $0.6738$ |

`fn_tau_sampling_scale_gaussian_factor()` in `R_fn_sample.R` computes these factors from the formulas with `optimize()` and `integrate()`, so they are not typed-in constants.

**Behaviour.**

- `tau_sampling_scale = "none"` (default): factor 1. This is the behaviour before the option existed.
- `"gaussian_matched"`: multiplies $\tau$ by the factor for the run's `burnin_algorithm`.
- A positive number: multiplies $\tau$ by that number.

The factor is applied once, to `tau_main` (and to `tau_us`, which the joint sampler keeps equal to `tau_main`), at the switch from burn-in to sampling. It is applied only when $\tau$ was adapted (`manual_tau = FALSE`) with a fixed length (`randomize_tau_burnin = FALSE`) and sampling is randomised. Otherwise the factor is 1: when $\tau$ was adapted under jitter, $\bar\tau$ already targets the jittered criterion. The step size is not re-initialised, and the result is capped at `max_tau_main`.

`R_fn_sample_model()` returns the following, so that runners can check what was applied:

- `tau_sampling_scale` (requested) and `tau_sampling_scale_effective`
- `tau_sampling_scale_factor` and `tau_sampling_scale_not_applied_reason`
- `tau_main_before_sampling_scale` and `tau_main_after_sampling_scale`, with the `tau_us` equivalents
- the effective `tau_initial` and `learning_rate_initial`

When the factor is not 1, `burnin_object$EHMC_args_as_Rcpp_List$tau_main` carries the scaled value, because the summaries compute the sampling $L$ and gradient counts from it. `burnin_object$tau_main` and `tau_main_during_burnin_vec` keep the burn-in values.

**Caveat.** The factor is exact only for a unit Gaussian under exact dynamics, and whether it helps on real posteriors is an empirical question. Pilot study 7 compares it with `"none"`, and with jittered burn-in (`randomize_tau_burnin = TRUE` with one $\tau$ shared by all chains in each iteration).
