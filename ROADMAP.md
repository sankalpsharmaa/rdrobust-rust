# Roadmap

Planned work for rdrobust-rust, in priority order. Dated 2026-10-18. Items move to [CHANGELOG.md](CHANGELOG.md) when they ship. Open an issue to discuss or claim one.

## 1. Match CRAN rdrobust 4.1.1

The parity fixtures pin upstream commit `a86d11a`, labelled 4.1.0. Upstream never released 4.1.0. CRAN rdrobust 4.1.1 (2026-10-01) changed results and added errors. Until this item ships, the parity claim refers to an unreleased reference.

- Re-pin `tools/parity.py` and the README fixture instructions to the 4.1.1 release. Regenerate `fixtures/*.json`.
- Sharp RD with covariates and `deriv >= 2`: multiply the estimate by `factorial(deriv)`. The port returns half the correct value for `deriv = 2` with covariates (`rdrobust-core/src/rdrobust.rs`, sharp-plus-covariates branch).
- Pad the `bwcheck` floor and the first pilot bandwidth by `1 + sqrt(eps)`, as upstream does. Selected bandwidths move by less than `1e-6` relative.
- Change the oracle `ginv.tol` from `1e-20` to `1e-15` and check whether `linalg.rs` needs the same change.
- `rdplot`: keep bin edges with their own bins when a side has empty bins.
- New errors: "Not enough variability" when a pilot window has fewer distinct `x` values than polynomial coefficients. A side with fewer than `q + 1` distinct `x` values stops with an error.
- New warning channel: fewer than 10 clusters with positive weight on a side, and a stronger warning at `p + 1` or fewer.
- Refresh the README Stata parity table against Stata 11.1.1 (the current table used `rdrobust.ado` 10.0.0).

## 2. Repository hygiene

Small items that each take under an hour.

- GitHub shows the license as "Other" because `LICENSE` starts with a custom header. Move the header to `NOTICE` and keep `LICENSE` as the verbatim GPL-3.0 text.
- Add repository topics: `regression-discontinuity`, `econometrics`, `causal-inference`, `rust`, `python`, `pyo3`.
- Add `CITATION.cff` and a Zenodo DOI per release.
- Add `CONTRIBUTING.md`, issue templates, and a pull request template.
- Add `.github/dependabot.yml` for Cargo, pip, and GitHub Actions.
- Run the Rust CI job on macOS and Windows as well as Linux. The 0.1.1 macOS loader failure was not caught because the Rust job runs on Linux only.
- Move to Rust 1.98 or later and drop `strip = "none"` from the release profile. Rust 1.98 fixed the `rust-objcopy` bug that misaligned the `__LINKEDIT` string pool.

## 3. Complete the `rdrobust` output

The Python result has 6 keys. R returns about 40. The values below already exist inside the side fits, so this is plumbing.

- `N`, `N_h`, `N_b`, `M`: sample sizes per side, effective sample sizes in `h` and `b`, and unique-value counts.
- `tau_cl` and `tau_bc` per side, and `bias` per side.
- `beta_Y_p_l`, `beta_Y_p_r`: side polynomial coefficients, needed to draw fitted lines.
- Fuzzy first stage: `tau_T`, `se_T`, `z_T`, `pv_T`, `ci_T`, `beta_T_p_l`, `beta_T_p_r`.
- `coef_covs`, `n_clust`, and the variance matrices `V_cl_l`, `V_cl_r`, `V_rb_l`, `V_rb_r`.
- Echo of the options used: `kernel`, `vce`, `bwselect`, `level`, `masspoints`, `rdmodel`.

## 4. Close the input gaps

- Expose `rdbwselect` in Python. The Rust crate already exports it.
- `all=TRUE`: return every bandwidth selector in one call, as R and Python do.
- `subset` mask, to keep the API parallel to R.
- Document or change the `covs_drop` default. The port uses `false`. R and Python use `TRUE`.
- Expose `ginv.tol`.
- Replace the `bwselect` and `masspoints` strings in `RdRobustOptions` with enums. An unknown selector currently panics.
- Accept a scalar or a pair for `h` and `b` in the Rust API, as the Python binding already does.
- `rdplot`: per-bin confidence intervals (`ci`), `covs` and `covs_eval`, `h`, `support`, `J_IMSE`, `J_MV`, `bin_avg`, `bin_med`, and the full per-bin and curve tables (`vars_bins`, `vars_poly`).

## 5. Python package layer

- Add a pure-Python package (`python/rdrobust_rust/__init__.py`) around the extension.
- Ship type stubs: a hand-written `rdrobust_rust.pyi` plus `py.typed`, or PyO3 `experimental-inspect` with `maturin build --generate-stubs`.
- Return a frozen dataclass with `.coef`, `.se`, `.ci`, `.N_h`, a `__repr__`, and a `summary()`.
- Accept pandas and polars inputs and column names with a `data=` argument. Upstream R 4.0.0 and Python 2.x both added `data=`.
- Optional matplotlib helper that draws an RD plot from `rdplot` output, matching upstream Python's `plot_rdrobust`.
- Upgrade PyO3 from 0.26 to the current release before adding wheels.

## 6. Wheels and platforms

PyPI currently holds three wheels per release: macOS arm64, manylinux_2_35 x86_64, and Windows x64.

- Build Linux wheels in a manylinux container (`manylinux: 2_28` or `2_17` in `maturin-action`). The current glibc 2.35 floor blocks RHEL 8 and Rocky 8.
- Add Linux aarch64 and a macOS `universal2` wheel for Intel Macs.
- Add a musllinux wheel for Alpine containers.
- Add a `cp314t` wheel for free-threaded Python 3.14 and an `abi3t` wheel for 3.15 and later. abi3 wheels do not install on 3.14t. The Rayon batch path is the use case that gains most.
- Submit a conda-forge feedstock.

## 7. Benchmarks

- Add the official Python `rdrobust` package to the benchmark table, with and without its numba `speed` extra. Python users ask for this comparison first.
- Re-run the R and Stata timings against R 4.1.1 and Stata 11.1.1. The README numbers are from July 2026 on R 3.0.0 and `rdrobust.ado` 10.0.0.
- Benchmark at `n = 1e6` to confirm whether the nearest-neighbor residual search is the bottleneck for large samples.

## 8. Core API

- `try_rdrobust_batch` returning `Vec<Result<RdRobust, RdError>>`, so one bad row fails alone instead of poisoning the batch with `NaN`. The raw `rdrobust_batch` still panics on a `NaN` in `x` during the sort.
- A `serde` feature on `RdRobust`, `RdRobustOptions`, and `RdPlot`.
- A batch bandwidth-selection path.
- A public Monte Carlo helper that returns coverage and bias tables, based on `examples/monte_carlo.rs`.

## 9. New estimators

All seven packages on rdpackages.github.io have R, Stata, and Python versions. None has a compiled core. In order of fit with the existing batch design:

1. **rdpower**: power, sample size, and minimum detectable effect. Power curves call `rdrobust` hundreds of times, which is the batch workload this crate is built for.
2. **rdmulti**: `rdmc` (multiple cutoffs, pooled and per cutoff), `rdms` (multiple scores), `rdmcplot`. `rdrobust_multi_cutoff` already runs the per-cutoff fits in parallel.
3. **rddensity**: manipulation test at the cutoff. Every RD paper runs it once.
4. **RDHonest**: honest confidence intervals under a smoothness bound (Armstrong and Kolesár). R and Stata exist. No Python implementation exists.
5. **rdlocrand**: randomization inference in a window. Window selection runs many permutation tests.
6. **rdhte** and **rd2d**: heterogeneous effects and boundary designs. Newer methods, more work.

Smaller method additions that reuse the batch path: placebo-cutoff sweeps and bandwidth-sensitivity tables, donut-hole RD, and a wild cluster bootstrap.

## 10. Other bindings

- **R package through extendr.** R users are the largest RD audience. CRAN needs vendored crates, offline builds, and `SystemRequirements: Cargo, rustc`.
- **Stata plugin.** Stata plugins are C shared libraries with a `stata_call(argc, argv)` entry. Rust can export it with `extern "C"` and compile `stplugin.c` through the `cc` crate.
- **Polars expression plugin** through `pyo3-polars`, so a group-by aggregation can run the batch path without the GIL.
- **WebAssembly demo page** with a bandwidth slider, for teaching and for the README.

## Not planned

- A zero-copy input path. The implementation sorts and owns its working data.
- Bit-for-bit identity with R or Stata. The target is agreement within documented tolerances.
- Publishing the Python extension crate (`rdrobust-py`) to crates.io. It ships through PyPI.
