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
3. Confirm the release assets include Linux, macOS, and Windows wheels plus the
   source distribution.
4. Confirm that [docs.rs](https://docs.rs/rdrobust-core) built the new
   `rdrobust-core` documentation after the crates.io job succeeds.
5. In a clean Rust project, run `cargo add rdrobust-core` and compile a small
   program.
6. In a clean Python environment, install a wheel from the GitHub release (or
   from PyPI once published), then run `rdrobust-py/tests/test_bindings.py`
   from this checkout.

Record the tag, GitHub release asset names, crates.io version, PyPI version,
and clean-install results in the release notes.

## GitHub Releases (Python packages)

Publishing a GitHub release from a version tag builds abi3 wheels for Linux,
macOS, and Windows, builds the sdist, and attaches all of them to that release.
This path does not require PyPI credentials.

```sh
git tag -a v0.1.0 -m "rdrobust-rust 0.1.0"
git push origin v0.1.0
gh release create v0.1.0 --title "v0.1.0" --generate-notes
```

After `release.yml` finishes, install a platform wheel from the release Assets
list:

```sh
uv pip install https://github.com/sankalpsharmaa/rdrobust-rust/releases/download/v0.1.0/<wheel-file>.whl
```

Replace `<wheel-file>` with the asset that matches your OS and architecture.
The `github-release` job uploads the built `.whl` and `.tar.gz` files.

The `crates-io` and `pypi` jobs stay skipped until you set repository variables
`PUBLISH_CRATES_IO=true` and/or `PUBLISH_PYPI=true` and configure the matching
environments and credentials. Re-runs are idempotent: crates.io skips a version
that already exists, and PyPI uses `skip-existing`.
