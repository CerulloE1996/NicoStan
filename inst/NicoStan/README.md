# NicoStan compiled package

This directory contains NicoStan's compiled sampler package. The outer NicoStan package supplies the installer.

- The maintained documentation is the [NicoStan README](https://github.com/CerulloE1996/NicoStan).
- Edit and run the examples from `R_packages/NicoStan/inst/examples`. The installer copies that source directory into the compiled package.
- Use `inst/examples/NicoStan_admin_install.R` from the outer source package for a complete local installation.
- NicoStan provides general Stan/BridgeStan sampling; the separate BayesMVP extension supplies the specialised multivariate probit implementations and reusable AVX functions.
