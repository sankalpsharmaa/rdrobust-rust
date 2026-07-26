# Releasing rdrobust-rust

This is an independent project. Before publishing, review the release diff and
confirm the version is consistent in the workspace and Python metadata.

## Local release gate

Run these commands from the repository root, in this order:

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-targets --all-features
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps
cargo package --workspace
cargo publish --dry-run -p rdrobust-core
uvx maturin build --release --sdist --manifest-path rdrobust-py/Cargo.toml --out dist
uvx twine check dist/*
python rdrobust-py/tests/test_bindings.py
```

Inspect every `.crate`, `.whl`, and `.tar.gz` before upload. Each archive must
contain `LICENSE`, `NOTICE`, and `AUTHORS` (wheel files may place them in the
standard `.dist-info/licenses/` directory).

## TestPyPI and trusted publishing

First exercise the Python distribution on TestPyPI. Configure a TestPyPI
Trusted Publisher for this repository and `release.yml`, or upload the checked
artifacts with a scoped TestPyPI token. Install the resulting package in a
fresh virtual environment and run `rdrobust-py/tests/test_bindings.py` before
publishing to PyPI.

For PyPI, configure a Trusted Publisher for project `rdrobust-rust` with the
repository workflow `.github/workflows/release.yml` and environment `pypi`.
The workflow uses GitHub's OIDC token; do not add a PyPI API token to repository
secrets. The `pypi` environment should require the desired reviewer(s).

## crates.io

Publish `rdrobust-core` first: it has no internal crate dependency. Wait for
crates.io to index the release before publishing any dependent crate. The
release workflow performs `cargo publish --dry-run -p rdrobust-core` before the
guarded upload to the `crates-io` environment. The Python extension is
distributed to PyPI and is not published to crates.io by this release process.

## Tag, publish, and verify

1. Create and push an annotated tag such as `v0.1.0` for the reviewed commit.
2. Publish a GitHub release from that tag, which runs `release.yml`.
3. Confirm that [docs.rs](https://docs.rs/rdrobust-core) built the new
   `rdrobust-core` documentation.
4. In a clean Rust project, run `cargo add rdrobust-core` and compile a small
   program.
5. In a clean Python environment, install `rdrobust-rust`, then run
   `python rdrobust-py/tests/test_bindings.py` from this checkout.

Record the tag, crates.io version, PyPI version, and clean-install results in
the release notes.
