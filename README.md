# rdrobust-rust

> Independent Rust reimplementation; not an official rdpackages release.

`rdrobust-rust` provides regression-discontinuity estimation in a compiled Rust
core with a NumPy-facing Python API. It is intended for research pipelines that
run RD repeatedly: across outcomes, subgroups, specifications, simulations, or
cutoffs.

This repository is currently a private source release. The Python and Rust
packages have not been published to PyPI or crates.io.

## Why this repository exists

The reference [rdrobust](https://github.com/rdpackages/rdrobust) packages remain
the standard implementation. This project serves a narrower computational need:
running the same numerical work many times without making Python manage every
fit.

- The numerical core is compiled Rust. Rust's
  [ownership model](https://doc.rust-lang.org/book/ch04-01-what-is-ownership.html)
  releases memory when values leave scope without a garbage collector.
- `rdrobust_batch` and `rdrobust_multi_cutoff` use
  [Rayon](https://docs.rs/rayon/latest/rayon/) to schedule independent fits on a
  shared, work-stealing thread pool.
- The Python binding
  [releases the interpreter](https://pyo3.rs/main/parallelism) during numerical
  work, so the Rust computations can run in parallel and other Python threads
  can continue.
- A threaded batch stays in one process. This avoids starting a separate Python
  interpreter for each worker and can avoid some of the serialization and data
  duplication associated with
  [`multiprocessing`](https://docs.python.org/3/library/multiprocessing.html).
- Rust checks ownership and common data-race mistakes at compile time. That
  makes the parallel path easier to maintain than hand-managed shared state.
- The same Rust core backs the single-fit, batch, and multiple-cutoff APIs.
  Deterministic fixtures compare its results with the upstream R package.

Rust does not make every RD call faster or smaller. A single small fit may see
little benefit after Python-to-Rust conversion costs. The implementation also
sorts and owns working data internally; it is not zero-copy. The main target is
a large dataset or research design that requires many independent fits. Measure
your actual workload before treating speed or memory use as settled.

## Fastest Python setup

The reliable path is to clone the repository and install an optimized extension
inside a virtual environment.

### 1. Install the one-time prerequisites

You need:

- GitHub access to `sankalpsharmaa/rdrobust-rust`;
- [GitHub CLI](https://cli.github.com/) authenticated with `gh auth login`;
- [Rust through rustup](https://rustup.rs/); and
- [uv](https://docs.astral.sh/uv/getting-started/installation/).

On macOS or Linux, install Rust and uv if they are not already available:

```sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Open a new shell, then confirm the tools are on your path:

```sh
rustc --version
uv --version
gh auth status
```

### 2. Clone the private repository

```sh
gh repo clone sankalpsharmaa/rdrobust-rust
cd rdrobust-rust
```

### 3. Create the Python environment

Python 3.9 through 3.14 is supported. Python 3.12 is a conservative default:

```sh
uv venv --python 3.12
source .venv/bin/activate
uv pip install numpy
```

On Windows PowerShell, activate with:

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Compile and install the extension

[`maturin develop`](https://www.maturin.rs/local_development) builds the Rust
extension and installs it directly into the active environment:

```sh
env -u CONDA_PREFIX uvx maturin develop \
  --release \
  --manifest-path rdrobust-py/Cargo.toml
```

`env -u CONDA_PREFIX` prevents an active Conda base environment from competing
with `.venv`. It has no effect when Conda is not active. On Windows PowerShell,
run `Remove-Item Env:CONDA_PREFIX -ErrorAction SilentlyContinue` once, then run
the `uvx maturin develop` command without the `env -u CONDA_PREFIX` prefix.

### 5. Verify the installation

```sh
python -c 'import rdrobust_rust as rr; print(rr.__name__)'
```

Expected output:

```text
rdrobust_rust
```

## First RD estimate in Python

```python
import numpy as np
import rdrobust_rust as rr

rng = np.random.default_rng(0)
x = np.ascontiguousarray(rng.uniform(-1, 1, 10_000), dtype=np.float64)
y = np.ascontiguousarray(
    0.5 * x + 2.0 * (x >= 0) + rng.standard_normal(x.size),
    dtype=np.float64,
)

result = rr.rdrobust(y, x, c=0.0)

print("Bias-corrected estimate:", result["coef"][1])
print("Robust 95% CI:", result["ci"][2])
print("Selected bandwidths:", result["bws"])
```

The result is a dictionary:

|Key|Contents|
|-|-|
|`coef`|Conventional and bias-corrected estimates|
|`se`|Conventional and robust standard errors|
|`z`|Test statistics|
|`pv`|p-values|
|`ci`|Conventional, bias-corrected, and robust confidence intervals|
|`bws`|Selected estimation and bias bandwidths|

## Run many RD fits in parallel

Use `rdrobust_batch` instead of starting one Python process per fit. Each row is
an independent dataset. Both arrays must have shape `(number_of_fits,
observations_per_fit)`.

```python
import numpy as np
import rdrobust_rust as rr

rng = np.random.default_rng(1)
fits, observations = 100, 5_000

x = np.ascontiguousarray(
    rng.uniform(-1, 1, size=(fits, observations)),
    dtype=np.float64,
)
y = np.ascontiguousarray(
    0.3 * x + 1.5 * (x >= 0) + rng.standard_normal(x.shape),
    dtype=np.float64,
)

results = rr.rdrobust_batch(y, x, c=0.0)
estimates = np.array([result["coef"][1] for result in results])
print(estimates)
```

Rayon uses the machine's logical CPU count by default. To cap the shared thread
pool, set `RAYON_NUM_THREADS` before starting Python:

```sh
RAYON_NUM_THREADS=8 python analysis.py
```

Do not wrap `rdrobust_batch` in a second multiprocessing pool. Divide the work
into rows and let the native batch function parallelize it once.

## Run one dataset at many cutoffs

Use `rdrobust_multi_cutoff` when the outcome and running variable are shared:

```python
import numpy as np
import rdrobust_rust as rr

rng = np.random.default_rng(2)
x = np.ascontiguousarray(rng.uniform(-1, 1, 10_000), dtype=np.float64)
y = np.ascontiguousarray(
    0.4 * x + 1.0 * (x >= 0) + rng.standard_normal(x.size),
    dtype=np.float64,
)
cutoffs = np.array([-0.25, 0.0, 0.25], dtype=np.float64)
results = rr.rdrobust_multi_cutoff(y, x, cutoffs)

for cutoff, result in zip(cutoffs, results):
    print(cutoff, result["coef"][1], result["ci"][2])
```

Here `x` and `y` are one-dimensional arrays. The function evaluates the cutoffs
in parallel and returns one result dictionary per cutoff.

## Python input checklist

Before calling the extension:

1. Convert numerical inputs to `np.float64`.
2. Make arrays C-contiguous with `np.ascontiguousarray`.
3. Give `x` and `y` the same number of observations.
4. Include observations on both sides of the cutoff.
5. Remove `NaN` and infinite values.
6. Give `fuzzy`, `cluster`, and `weights` the same length as `x`.
7. Give `covs` shape `(observations, covariates)` for one fit, or
   `(fits, observations, covariates)` for a batch.
8. Encode cluster labels as numbers before passing them to the extension.

The extension validates these conditions and raises `ValueError` when an input
is invalid.

## Supported features

- Sharp and fuzzy RD estimation with robust bias-corrected inference.
- Data-driven bandwidth selection.
- Covariates and observation weights.
- Nearest-neighbor, heteroskedasticity-consistent, and cluster-robust variance
  estimators.
- RD plot binning and polynomial fit data through `rdplot`.
- Parallel batch and multiple-cutoff estimation.

## Known limitations

The public output focuses on estimates, standard errors, tests, confidence
intervals, and bandwidths. Per-side decompositions, fuzzy first-stage
diagnostics, multi-row `rdbwselect(all=TRUE)` output, and several `rdplot`
extensions are not exposed. The Python binding accepts NumPy arrays rather than
DataFrames or `subset` and missing-value convenience arguments.

## Benchmark your workload

Run the included benchmark from the repository root:

```sh
python3 tools/parity.py benchmark
```

Treat this as a local engineering check, not as a general performance claim.
Report the array sizes, number of fits, CPU count, thread count, and peak memory
when comparing implementations.

## Repository layout

|Path|Purpose|
|-|-|
|`rdrobust-core/`|Pure-Rust numerical crate|
|`rdrobust-py/`|PyO3 and maturin Python binding|
|`fixtures/`|Deterministic, schema-versioned R parity fixtures|
|`tools/`|Fixture generation, verification, and benchmark tools|

## Test and reproduce

Run the standard checks from the repository root:

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-targets --all-features
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps
python3 -m unittest tools.parity
python3 tools/parity.py verify
python rdrobust-py/tests/test_bindings.py
```

Regenerating the fixtures also requires `Rscript`, `jsonlite`, and the upstream R
implementation pinned to the 4.1.0 source revision used for the fixtures. Keep
them isolated from your normal R library:

```sh
mkdir -p .r-lib
R_LIBS_USER="$PWD/.r-lib" Rscript -e \
  'install.packages(c("jsonlite", "remotes"), repos="https://cloud.r-project.org");
   remotes::install_github("rdpackages/rdrobust",
     ref="a86d11aa5a89f28923c9b1ed4b8ba5c0d83d4514",
     subdir="R/rdrobust", dependencies=FALSE, upgrade="never")'
R_LIBS_USER="$PWD/.r-lib" Rscript -e \
  'stopifnot(packageVersion("rdrobust") == "4.1.0")'
R_LIBS_USER="$PWD/.r-lib" python3 tools/parity.py generate --check
```

See [RELEASING.md](RELEASING.md) for packaging, publication, and clean-install
checks.

## License and attribution

This project is distributed under GPL-3.0-only. See [LICENSE](LICENSE),
[NOTICE](NOTICE), and [AUTHORS](AUTHORS). The original rdrobust authors are
Sebastian Calonico, Matias D. Cattaneo, Max H. Farrell, and Rocio Titiunik.

## Security

Report suspected vulnerabilities privately through this repository's GitHub
Security Advisories interface rather than a public issue.
