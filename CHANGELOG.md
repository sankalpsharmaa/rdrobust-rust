# Changelog

All notable user-visible changes are documented here. This project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Initial release contents are complete; see the release checklist.

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
