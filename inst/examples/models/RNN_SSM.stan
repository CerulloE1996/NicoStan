// Non-centred Bayesian single-gate GRU / nonlinear state-space model (RNN-SSM),
// adapted from stan-dev/example-models BPA/Ch.05/ssm.stan with an input-dependent
// update gate and a tanh recurrent nonlinearity:
//   a[t] = inv_logit(g0 + gx * x[t])
//   h[1] = h0 + sigma_proc * eps[1]
//   h[t] = a[t] * h[t - 1] + (1 - a[t]) * tanh(wh * h[t - 1] + wx * x[t] + b)
//          + sigma_proc * eps[t],   t = 2, ..., T
//   y[t] ~ normal(h[t], sigma_obs)
// Nuisance parameter block eps[1..T] ~ std_normal() declared first.
data {
    int<lower=2> T;               // # time points (equally spaced)
    vector[T] x;                  // exogenous input sequence
    vector[T] y;                  // observed output sequence
}
parameters {
    vector[T] eps;                // standard normal nuisance block, declared first
    real g0;                      // update-gate intercept
    real gx;                      // update-gate input weight
    real wh;                      // recurrent weight inside tanh
    real wx;                      // input weight inside tanh
    real b;                       // candidate-state bias
    real h0;                      // initial latent state
    real<lower=0> sigma_proc;     // state process noise scale
    real<lower=0> sigma_obs;      // observation noise scale
}
model {
    g0 ~ normal(0, 2);
    gx ~ normal(0, 2);
    wh ~ normal(0, 2);
    wx ~ normal(0, 2);
    b ~ normal(0, 2);
    h0 ~ normal(0, 2);
    sigma_proc ~ normal(0, 1);
    sigma_obs ~ normal(0, 1);
    eps ~ std_normal();
    {
      vector[T] a = inv_logit(g0 + gx * x);
      vector[T] one_minus_a = 1.0 - a;
      vector[T] drive = wx * x + b;
      vector[T] shock = sigma_proc * eps;
      vector[T] h;
      h[1] = h0 + sigma_proc * eps[1];
      for (t in 2:T) {
        h[t] = a[t] * h[t - 1] + one_minus_a[t] * tanh(wh * h[t - 1] + drive[t]) + shock[t];
      }
      y ~ normal(h, sigma_obs);
    }
}
generated quantities {
    vector[T] a;
    vector[T] h;
    vector[T] y_rep;
    a = inv_logit(g0 + gx * x);
    h[1] = h0 + sigma_proc * eps[1];
    for (t in 2:T) {
      h[t] = a[t] * h[t - 1] + (1.0 - a[t]) * tanh(wh * h[t - 1] + wx * x[t] + b) + sigma_proc * eps[t];
    }
    y_rep = to_vector(normal_rng(h, sigma_obs));
}
