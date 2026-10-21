// stochastic_volatility_centred_unchunked_no_loops_over_N_with_Stan_fused_functions.stan
// Unchunked Stan::math version of the CENTRED stochastic volatility model of the chunking paper's layout timings
// (stochastic_volatility_centred_sliced_latent_path_automatic_reduce_sum_chunked, from its source
// stochastic_volatility_centred_sliced_latent_path): the same log density as that source at N_chunks = 1 (one
// leaf: no transition across a chunk boundary), as one pass over all T time points, without reduce_sum_static.
// The log volatilities h are the parameters (the same posterior over (mu, phi, sigma, h) as the non-centred
// model with h_std; the sampler moves in other coordinates).
// The same data (N and chunk_size, when they are in the data, are not read; y is read as a vector, as in
// stochastic_volatility.stan) and the same unconstrained parameters, in the same order and with the same names
// (h, mu, phi, sigma), as the source; h is a vector (the source's array[T] real h, the same T values in the same
// order, exists only so that reduce_sum_static can slice it).
// The transitions h[t] ~ normal(mu + phi (h[t - 1] - mu), sigma), t = 2, ..., T, are one call of Stan's fused
// normal GLM: mu + phi (h[t - 1] - mu) = mu (1 - phi) + phi h[t - 1], i.e. normal_id_glm with the outcomes
// h[2:T], the covariates h[1:(T - 1)] (a (T - 1) x 1 matrix of the same values), the intercept mu (1 - phi), the
// coefficient phi and the scale sigma (the same density). The returns are Stan's vectorised normal, whose scale
// exp(h / 2) varies by time point (no fused function applies); h[1]'s stationary term as in the source.
// No loop over N (T) anywhere.


data {
    int<lower=1> T;               // # time points (equally spaced)
    vector[T] y;                  // mean corrected return at time t
}


parameters {
    vector[T] h;                  // log volatility at time t (the latent path, centred)
    real mu;                      // mean log volatility
    real<lower=-1, upper=1> phi;  // persistence of volatility
    real<lower=0> sigma;          // white noise shock scale
}


model {
    // priors
    phi ~ uniform(-1, 1);
    sigma ~ normal(0, 0.5);
    mu ~ normal(0, 2);

    // the stationary first state, the transitions (Stan's fused normal GLM; see the header) and the returns
    h[1] ~ normal(mu, sigma / sqrt(1 - square(phi)));
    if (T > 1) {
        h[2:T] ~ normal_id_glm(to_matrix(h[1:(T - 1)]), mu * (1 - phi), [phi]', sigma);
    }
    y ~ normal(0, exp(h / 2));
}


generated quantities {
    vector[T] volatility = exp(h / 2);
}






















