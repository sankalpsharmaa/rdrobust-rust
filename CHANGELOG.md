# Changelog

All notable user-visible changes are documented here. This project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- `rdrobust` no longer panics in debug builds (or returns silent `NaN` in release
  builds) when a pilot window is empty, e.g. a cutoff at `max(x)`. The validated
  entry points (`try_rdrobust_with` and the Python binding) now reject a cutoff
  that is not strictly inside the range of `x`, matching R.
- `rdrobust` applies R's small-sample rule: with fewer than 20 observations the
  full support is used as the bandwidth on both sides instead of running
  data-driven selection on a sample too small for it.
- `rdrobust` and the Python binding reject a fuzzy treatment that is constant on
  both sides of the cutoff with no jump (not identified), as R does. One-sided
  compliance is detected by exact equality rather than a floating-point variance.
- `rdbwselect` now drops a treatment that is constant on either side and selects
  a sharp bandwidth, as R's `rdbwselect` does; `rdrobust` already did this.
- `rdplot` keeps the order-4 pilot regression when its Gram matrix is finite but
  ill-conditioned (R falls back to the pseudoinverse at order 4) and shares the
  pilot order across sides; it previously dropped to order 3 or 2 on one side,
  changing the `es`/`espr`/`esmvpr` bin counts for offset running variables. A
  side with a constant outcome now gets one bin as in R, instead of an infinite
  `es` bin count.
- `rdbwselect` propagates a failed pilot stage as `NaN` through the
  `bwrestrict`/`bwcheck` clamps and the `comb` selectors, as R's `min`/`max`/
  `median` do, instead of silently substituting the clamp bound.
- `try_rdrobust_with` rejects a ragged `covs` slice whose length is not `n * dz`.
- Python binding: `level` must be strictly positive; `h`/`b` accept a length-1
  sequence as R does.
- Fixture metadata now records the pinned R reference as rdrobust 4.1.0.

## [0.1.0]

### Added

- Pure-Rust local-polynomial estimation for sharp and fuzzy RD designs, with
  conventional and robust bias-corrected inference.
- Data-driven bandwidth selection, covariates, user weights, mass-point and
  standardization controls, and nearest-neighbor, HC0--HC3, CR1, CRV2, and CRV3
  variance estimators.
- RD plot data and ES-family bin selection.
- Parallel batch and multiple-cutoff estimation APIs.
- Python abi3 bindings for `rdrobust`, `rdplot`, `rdrobust_batch`, and
  `rdrobust_multi_cutoff`.
- GIL release around Python binding computations.
- R-parity fixtures, Rust tests, and Python binding tests.
