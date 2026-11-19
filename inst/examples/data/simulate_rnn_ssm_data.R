args <- commandArgs(trailingOnly = TRUE)
out_dir <- if (length(args) >= 1) args[1] else "."
stan_file <- if (length(args) >= 2) args[2] else ""
script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
generator_file <- if (length(script_arg) >= 1) normalizePath(sub("^--file=", "", script_arg[1]), mustWork = FALSE) else ""
generator_file_md5 <- if (nzchar(generator_file) && file.exists(generator_file)) as.character(tools::md5sum(generator_file)) else ""

true_vals <- c(
  g0 = 0.8,
  gx = 0.7,
  wh = 1.1,
  wx = 0.9,
  b = 0.1,
  h0 = 0.0,
  sigma_proc = 0.25,
  sigma_obs = 0.35
)

simulate_one <- function(T_len, seed = 123L) {
  set.seed(seed)
  z_x <- rnorm(T_len)
  x <- numeric(T_len)
  x[1] <- z_x[1]
  phi_x <- 0.7
  sd_x <- sqrt(1 - phi_x^2)
  for (t in 2:T_len) x[t] <- phi_x * x[t - 1] + sd_x * z_x[t]

  eps <- rnorm(T_len)
  a <- plogis(true_vals["g0"] + true_vals["gx"] * x)
  h <- numeric(T_len)
  h[1] <- true_vals["h0"] + true_vals["sigma_proc"] * eps[1]
  for (t in 2:T_len) {
    h[t] <- a[t] * h[t - 1] +
      (1.0 - a[t]) * tanh(true_vals["wh"] * h[t - 1] + true_vals["wx"] * x[t] + true_vals["b"]) +
      true_vals["sigma_proc"] * eps[t]
  }
  y <- rnorm(T_len, mean = h, sd = true_vals["sigma_obs"])

  stan_data_list <- list(T = as.integer(T_len), x = as.numeric(x), y = as.numeric(y))
  tmp_rds <- tempfile(fileext = ".rds")
  saveRDS(stan_data_list, tmp_rds)
  sha <- sub(" .*", "", system2("sha256sum", tmp_rds, stdout = TRUE)[1])
  unlink(tmp_rds)

  stan_md5 <- if (nzchar(stan_file) && file.exists(stan_file)) {
    as.character(tools::md5sum(stan_file))
  } else ""

  obj <- list(
    stan_data_list = stan_data_list,
    rnn_ssm_case = sprintf("T%d_ds%d", T_len, seed),
    data_sha256 = sha,
    data_source = sprintf("simulate_rnn_ssm_data.R, analysis profile (T%d_seed%d)", T_len, seed),
    T = as.integer(T_len),
    data_seed = as.numeric(seed),
    profile = "analysis",
    main_parameters = names(true_vals),
    nuisance_parameter = "eps",
    n_nuisance = as.integer(T_len),
    initial_main_values = list(
      g0 = 0.5, gx = 0.5, wh = 0.8, wx = 0.6, b = 0.0, h0 = 0.0,
      sigma_proc = 0.3, sigma_obs = 0.4
    ),
    initial_eps_sd = 0.05,
    true_values = true_vals,
    generator = sprintf("simulate_rnn_ssm_data.R(T = %d, seed = %d)", T_len, seed),
    generator_file = generator_file,
    generator_file_md5 = generator_file_md5,
    rng_kind = RNGkind(),
    R_version = R.version.string,
    stan_model_file = stan_file,
    stan_model_md5 = stan_md5,
    created_at = Sys.time()
  )
  out_path <- file.path(out_dir, sprintf("rnn_ssm_data_T%d_seed%d.rds", T_len, seed))
  saveRDS(obj, out_path)
  cat(sprintf("WROTE %s (sha=%s, stan_md5=%s)\n", out_path, substr(sha, 1, 12), substr(stan_md5, 1, 12)))
}

for (T_len in c(500L, 10000L, 100000L)) simulate_one(T_len, 123L)
