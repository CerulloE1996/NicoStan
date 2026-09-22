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
