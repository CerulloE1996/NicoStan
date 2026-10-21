// (v2: mu and the division by -alpha folded into the coefficients of one product per observation, with the
//  covariates' first column of ones made once in transformed data; the same values)
// Weibull_survival_chunked_no_loops_over_N_hand_written_BayesMVP_AVX.stan: the hand-written BayesMVP AVX version
// of Weibull_survival_chunked_no_loops_over_N.stan (the same model): every element-wise call on a vector written
// by hand as BayesMVP's nico_exp and nico_log (exp and log of vectors) and BayesMVP's fused nico_weibull_lpdf,
// nico_weibull_lccdf and nico_normal_lpdf (the log densities of vectors); every scalar call stays on Stan's math
// library. BayesMVP's functions are declared in the functions block as NicoStan's AVX writer declares them; one
// text for AVX2 and AVX-512 (the instruction set comes from the compiler flags, as for NicoStan's AVX versions),
// and simd_lanes (generated quantities) reports the SIMD lanes compiled (4 for AVX2, 8 for AVX-512).
//
// Weibull_survival_chunked.stan
// (with no loop over the observations: the model block as before, already vectorised in the chunks; the
//  censored observations of the chunks, censored_before, computed with whole-vector operations in transformed
//  data, and the generated quantities vectorised, the same values; the loops before are kept below, commented
//  out)
// Tape-chunked version of Weibull_survival.stan: the same log density, with the likelihood summed over the
// observations by reduce_sum_static, so that each chunk's autodiff tape is built, differentiated and released
// before the next one (the chunking paper's chunking; N_chunks from
// NicoStan::fn_compute_initial_N_chunks()). The data
// hold the observed failures (N: t, X) and the censored observations (N_cens: t_cens, X_cens) apart; the observed
// failures are the units of the chunks, in chunks of chunk_size failures (data), and each chunk also takes its
// share of the censored observations, in proportion to its failures: the chunk of failures start to end takes the
// censored observations censored_before[start] + 1 to censored_before[end + 1], with
// censored_before[e + 1] = floor(e * N_cens / N), so consecutive chunks take consecutive censored observations,
// every censored observation is in exactly one chunk and every chunk holds about the same number of observations.
// Each chunk computes its observations' scales and terms with the whole-vector operations of
// Weibull_survival.stan; lambda and lambda_cens, transformed parameters there, are reported as generated
// quantities (the same values).
// Nothing per observation is computed outside reduce_sum_static: the outer tape holds the priors on alpha, beta
// and mu (K + 2 values) and the one node of reduce_sum_static.
// The chunking data are NicoStan's for Stan models: N (the observed failures, the units of the chunks; already in
// the data) and chunk_size, which NicoStan sets to ceiling(N / N_chunks) in each phase (num_chunks_burnin,
// num_chunks_sampling).
//
// The model, as in Weibull_survival.stan: the vectorised version of NicoStan's example
// (inst/examples/models/standard_surv_Weibull.stan).

functions {
    real bmvp_simd_lanes();
    vector nico_exp(vector x);
    vector nico_log(vector x);
    real nico_normal_lpdf(vector y, vector mu, vector sigma);
    real nico_normal_lpdf(vector y, vector mu, real sigma);
    real nico_normal_lpdf(vector y, real mu, vector sigma);
    real nico_normal_lpdf(vector y, real mu, real sigma);
    real nico_normal_lpdf(array[] real y, vector mu, vector sigma);
    real nico_normal_lpdf(array[] real y, vector mu, real sigma);
    real nico_normal_lpdf(array[] real y, real mu, vector sigma);
    real nico_normal_lpdf(array[] real y, real mu, real sigma);
    real nico_normal_lpdf(real y, real mu, real sigma);
    real nico_weibull_lpdf(vector y, real alpha, real sigma);
    real nico_weibull_lccdf(vector y, real alpha, real sigma);
    real nico_weibull_lpdf(array[] real y, real alpha, real sigma);
    real nico_weibull_lccdf(array[] real y, real alpha, real sigma);
    real nico_weibull_lpdf(vector y, vector alpha, real sigma);
    real nico_weibull_lccdf(vector y, vector alpha, real sigma);
    real nico_weibull_lpdf(array[] real y, vector alpha, real sigma);
    real nico_weibull_lccdf(array[] real y, vector alpha, real sigma);
    real nico_weibull_lpdf(vector y, array[] real alpha, real sigma);
    real nico_weibull_lccdf(vector y, array[] real alpha, real sigma);
    real nico_weibull_lpdf(array[] real y, array[] real alpha, real sigma);
    real nico_weibull_lccdf(array[] real y, array[] real alpha, real sigma);
    real nico_weibull_lpdf(vector y, real alpha, vector sigma);
    real nico_weibull_lccdf(vector y, real alpha, vector sigma);
    real nico_weibull_lpdf(array[] real y, real alpha, vector sigma);
    real nico_weibull_lccdf(array[] real y, real alpha, vector sigma);
    real nico_weibull_lpdf(vector y, vector alpha, vector sigma);
    real nico_weibull_lccdf(vector y, vector alpha, vector sigma);
    real nico_weibull_lpdf(array[] real y, vector alpha, vector sigma);
    real nico_weibull_lccdf(array[] real y, vector alpha, vector sigma);
    real nico_weibull_lpdf(vector y, array[] real alpha, vector sigma);
    real nico_weibull_lccdf(vector y, array[] real alpha, vector sigma);
    real nico_weibull_lpdf(array[] real y, array[] real alpha, vector sigma);
    real nico_weibull_lccdf(array[] real y, array[] real alpha, vector sigma);
    real nico_weibull_lpdf(vector y, real alpha, array[] real sigma);
    real nico_weibull_lccdf(vector y, real alpha, array[] real sigma);
    real nico_weibull_lpdf(array[] real y, real alpha, array[] real sigma);
    real nico_weibull_lccdf(array[] real y, real alpha, array[] real sigma);
    real nico_weibull_lpdf(vector y, vector alpha, array[] real sigma);
    real nico_weibull_lccdf(vector y, vector alpha, array[] real sigma);
    real nico_weibull_lpdf(array[] real y, vector alpha, array[] real sigma);
    real nico_weibull_lccdf(array[] real y, vector alpha, array[] real sigma);
    real nico_weibull_lpdf(vector y, array[] real alpha, array[] real sigma);
    real nico_weibull_lccdf(vector y, array[] real alpha, array[] real sigma);
    real nico_weibull_lpdf(array[] real y, array[] real alpha, array[] real sigma);
    real nico_weibull_lccdf(array[] real y, array[] real alpha, array[] real sigma);
    real partial_sum_observations_lpdf(array[] real t_slice, int start, int end,
                                       vector t_cens, matrix X_with_ones, matrix X_cens_with_ones,
                                       array[] int censored_before,
                                       real alpha, vector beta, real mu) {
        // the chunk's censored observations: c1 to c2 (none when c2 < c1)
        int c1 = censored_before[start] + 1;
        int c2 = censored_before[end + 1];
        // the chunk's observed failures (start to end), with scale exp(-(mu + X * beta) / alpha)
        // real lp = weibull_lupdf(t_slice | alpha, exp(-(mu + X[start:end] * beta) / alpha));
        // (the same values, with the sign moved from the linear predictors to alpha: -(x) / alpha equals
        //  x / (-alpha) exactly, and the negation is then one node on alpha in place of one node per observation)
        real minus_alpha = -alpha;
        // real lp = nico_weibull_lupdf(t_slice | alpha, nico_exp((mu + X[start:end] * beta) / minus_alpha));
        // (the same values, with mu and the division by -alpha folded into the coefficients:
        //  (mu + X * beta) / (-alpha) = [1, X] * (append_row(mu, beta) / (-alpha)), so that each observation's
        //  scale takes one product, in place of a product, an addition and a division per observation)
        vector[cols(X_with_ones)] coefficients_over_minus_alpha = append_row(mu, beta) / minus_alpha;
        real lp = nico_weibull_lupdf(t_slice | alpha,
                                  nico_exp(X_with_ones[start:end] * coefficients_over_minus_alpha));
        // the chunk's censored observations
        if (c2 >= c1) {
          // lp += weibull_lccdf(t_cens[c1:c2] | alpha, exp(-(mu + X_cens[c1:c2] * beta) / alpha));
          // lp += nico_weibull_lccdf(t_cens[c1:c2] | alpha, nico_exp((mu + X_cens[c1:c2] * beta) / minus_alpha));
          lp += nico_weibull_lccdf(t_cens[c1:c2] | alpha,
                                nico_exp(X_cens_with_ones[c1:c2] * coefficients_over_minus_alpha));
        }
        return lp;
    }
}


data {
  int<lower=1> N;                    // number of observations (the observed failures; at least one, as the
                                     // units of the chunks)
  int<lower=0> N_cens;               // number of censored observations
  int<lower=0> K;                    // number of covariates
  vector[N] t;                       // observed failure times
  vector[N_cens] t_cens;             // censored times
  matrix[N, K] X;                    // covariate matrix for failures
  matrix[N_cens, K] X_cens;          // covariate matrix for censored
  int<lower=1> chunk_size;           // observed failures per chunk (reduce_sum_static)
}


transformed data {
  // the observed failure times, the sliced argument of reduce_sum_static
  array[N] real t_array = to_array_1d(t);
  // the censored observations of the chunks: the chunk of failures start to end takes the censored observations
  // censored_before[start] + 1 to censored_before[end + 1], with censored_before[e + 1] = floor(e * N_cens / N)
  // (counted in whole numbers without the product e * N_cens, so that it cannot overflow)
  // array[N + 1] int censored_before;
  // {
  //   int quotient = 0;
  //   int remainder = 0;
  //   censored_before[1] = 0;
  //   for (e in 1:N) {
  //     remainder += N_cens;
  //     while (remainder >= N) {
  //       remainder -= N;
  //       quotient += 1;
  //     }
  //     censored_before[e + 1] = quotient;
  //   }
  // }
  // (now with whole-vector operations, the same values: e * N_cens is computed exactly as a real (below 2^53),
  //  and floor(e * N_cens / N) of the real division is exact, as e * N_cens / N is at least 1 / N from the next
  //  whole number when it is not one, far more than its rounding error)
  array[N + 1] int censored_before = to_int(floor(to_array_1d(linspaced_vector(N + 1, 0, N) * N_cens / N)));
  // the covariates with a first column of ones, so that mu is the first coefficient of the chunks' products
  matrix[N, K + 1] X_with_ones = append_col(rep_vector(1, N), X);
  matrix[N_cens, K + 1] X_cens_with_ones = append_col(rep_vector(1, N_cens), X_cens);
}


parameters {
  real<lower=0> alpha;               // Weibull shape parameter
  vector[K] beta;                    // regression coefficients
  real mu;                           // intercept
}


model {
  // Priors
  // Example-scale priors: centred/unit-scale covariates and survival times in the chosen reference unit.
  alpha ~ lognormal(0, 0.5);
  beta ~ nico_normal(0, 1);
  // mu ~ normal(-1, 1.5);
  mu ~ normal(0, 2);

  // Likelihood for observed failures and for censored observations (reduce_sum_static over the observed
  // failures, in chunks of chunk_size, each chunk with its share of the censored observations; see the header)
  target += reduce_sum_static(partial_sum_observations_lupdf, t_array, chunk_size,
                              // t_cens, X, X_cens, censored_before, alpha, beta, mu);
                              t_cens, X_with_ones, X_cens_with_ones, censored_before, alpha, beta, mu);
}


generated quantities {
  real simd_lanes = bmvp_simd_lanes();
  // (lambda and lambda_cens: the transformed parameters of Weibull_survival.stan, the same values)
  vector[N] lambda;                  // scale parameters for failures
  vector[N_cens] lambda_cens;        // scale parameters for censored
  vector[N + N_cens] log_lik;
  vector[N + N_cens] t_pred;         // posterior predictive samples

  // Weibull parameterization: scale = exp(-(mu + X*beta)/alpha)
  lambda = nico_exp(-(mu + X * beta) / alpha);
  lambda_cens = nico_exp(-(mu + X_cens * beta) / alpha);

  // // Log-likelihood for observed
  // for (n in 1:N) {
  //   log_lik[n] = weibull_lpdf(t[n] | alpha, lambda[n]);
  //   t_pred[n] = weibull_rng(alpha, lambda[n]);
  // }
  //
  // // Log-likelihood for censored
  // for (n in 1:N_cens) {
  //   log_lik[N + n] = weibull_lccdf(t_cens[n] | alpha, lambda_cens[n]);
  //   t_pred[N + n] = weibull_rng(alpha, lambda_cens[n]);
  // }

  // Log-likelihood for observed (weibull_lpdf(t[n] | alpha, lambda[n]) of every observation, whole-vector:
  // log(alpha) + (alpha - 1) * log(t) - alpha * log(lambda) - (t / lambda)^alpha), and its posterior predictive
  // samples (weibull_rng over the vector lambda: the same draws, in the same order)
  log_lik[1:N] = log(alpha) + (alpha - 1) * nico_log(t) - alpha * nico_log(lambda) - (t ./ lambda) .^ alpha;
  t_pred[1:N] = to_vector(weibull_rng(alpha, lambda));

  // Log-likelihood for censored (weibull_lccdf(t_cens[n] | alpha, lambda_cens[n]) of every censored observation,
  // whole-vector: -(t_cens / lambda_cens)^alpha), and its posterior predictive samples
  if (N_cens > 0) {
    log_lik[(N + 1):(N + N_cens)] = -(t_cens ./ lambda_cens) .^ alpha;
    t_pred[(N + 1):(N + N_cens)] = to_vector(weibull_rng(alpha, lambda_cens));
  }

}
























