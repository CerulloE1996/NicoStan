

// stochastic_volatility.stan
data {
    int<lower=1> T;               // # time points (equally spaced)
    vector[T] y;                  // mean corrected return at time t
}


parameters {
    vector[T] h_std;              // standard normal nuisance block, declared first
    real mu;                      // mean log volatility
    real<lower=-1, upper=1> phi;  // persistence of volatility
    real<lower=0> sigma;          // white noise shock scale
}


transformed parameters {
    vector[T] h;                  // log volatility at time t
    h = h_std * sigma;            // now h ~ normal(0, sigma)
    h[1] /= sqrt(1 - phi * phi);  // rescale h[1]
    h += mu;
    if (T > 1) {
      for (t in 2:T) {
        h[t] += phi * (h[t-1] - mu);
      }
    }
}


model {
    // priors
    phi ~ uniform(-1, 1);
    // Example-scale priors for mean-corrected returns measured in standardised units.
    sigma ~ normal(0, 0.5);
    mu ~ normal(0, 2);
    h_std ~ std_normal();
    
    // likelihood
    y ~ normal(0, exp(h / 2));
}


generated quantities {
    vector[T] y_rep;
    vector[T] volatility;
    
    for (t in 1:T) {
      y_rep[t] = normal_rng(0, exp(h[t] / 2));
      volatility[t] = exp(h[t] / 2);
    }
}
