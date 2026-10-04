#!/usr/bin/env python3
"""Import, regenerate, verify, and benchmark the consolidated parity fixtures."""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

SCHEMA_VERSION = 1
FAMILY_FILES = (
    "bandwidth.json",
    "estimation.json",
    "clustering.json",
    "plotting.json",
    "preprocessing.json",
    "weights.json",
)
FAMILY_SOURCES = {
    "bandwidth.json": ("bw", "bwselect", "rho"),
    "estimation.json": ("rdrobust", "perfcomp", "_root_estimation"),
    "clustering.json": ("cluster", "cluster_bw"),
    "plotting.json": ("rdplot", "weights_rdplot"),
    "preprocessing.json": ("prep", "_root_preprocessing"),
    "weights.json": ("weights",),
}
ORACLE_METADATA = {"package": "rdrobust", "version": "4.1.0"}
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURES = REPO_ROOT / "fixtures"
DEFAULT_LEGACY = REPO_ROOT / "tests" / "fixtures"
ORACLE_SCRIPT = Path(__file__).with_name("oracle.R")


def validate_bundle(name: str, bundle: dict) -> None:
    """Validate the stable fixture envelope and its dispatch keys."""
    if not isinstance(bundle, dict):
        raise ValueError(f"{name}: bundle must be a JSON object")
    envelope_keys = {"schema_version", "oracle", "cases"}
    if set(bundle) != envelope_keys:
        raise ValueError(f"{name}: envelope keys must be {sorted(envelope_keys)!r}")
    if bundle.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"{name}: schema_version must be {SCHEMA_VERSION}, "
            f"got {bundle.get('schema_version')!r}"
        )
    if bundle.get("oracle") != ORACLE_METADATA:
        raise ValueError(f"{name}: oracle must be {ORACLE_METADATA!r}")
    cases = bundle.get("cases")
    if not isinstance(cases, list):
        raise ValueError(f"{name}: cases must be a list")
    previous_id: str | None = None
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(f"{name}: case {index} must be an object")
        case_id = case.get("id")
        kind = case.get("kind")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"{name}: case {index} has no non-empty string id")
        if not isinstance(kind, str) or not kind:
            raise ValueError(f"{name}: case {case_id!r} has no non-empty string kind")
        if previous_id is not None and case_id < previous_id:
            raise ValueError(f"{name}: cases must be sorted lexicographically by id")
        previous_id = case_id


def load_bundles(root: Path) -> dict[str, dict]:
    """Load and validate all six fixture bundles below *root*."""
    bundles: dict[str, dict] = {}
    for name in FAMILY_FILES:
        path = root / name
        try:
            bundle = json.loads(path.read_text())
        except FileNotFoundError as exc:
            raise ValueError(f"missing fixture bundle: {path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: invalid JSON: {exc}") from exc
        validate_bundle(name, bundle)
        bundles[name] = bundle
    return bundles


def run_oracle(requests: list[dict], rscript: str = "Rscript") -> list[dict]:
    """Send one request array to the R oracle and return its result array."""
    command = [rscript, str(ORACLE_SCRIPT)]
    try:
        completed = subprocess.run(
            command,
            input=json.dumps(requests, allow_nan=False),
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise RuntimeError(f"failed to run {rscript}: {exc}") from exc
    if completed.returncode != 0:
        diagnostic = completed.stderr.strip() or "(no standard error)"
        raise RuntimeError(
            f"R oracle failed with exit code {completed.returncode}: {diagnostic}"
        )
    try:
        results = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"R oracle returned invalid JSON: {exc}") from exc
    if not isinstance(results, list):
        raise RuntimeError("R oracle result must be a JSON array")
    return results


def _legacy_cases(fixtures_dir: Path, source: str) -> list[dict]:
    if source == "_root_estimation":
        paths = sorted(
            path
            for path in fixtures_dir.glob("*.json")
            if not path.name.startswith("tie_")
        )
        kind = "cross_language"
    elif source == "_root_preprocessing":
        paths = sorted(fixtures_dir.glob("tie_*.json"))
        kind = "prep"
    else:
        paths = sorted((fixtures_dir / source).glob("*.json"))
        kind = source

    cases = []
    for path in paths:
        case = json.loads(path.read_text())
        if not isinstance(case, dict):
            raise ValueError(f"{path}: fixture must be a JSON object")
        if "id" not in case:
            raise ValueError(f"{path}: fixture has no id")
        case["kind"] = kind
        cases.append(case)
    return cases


def _bundle_legacy(fixtures_dir: Path) -> dict[str, dict]:
    bundles: dict[str, dict] = {}
    for name, sources in FAMILY_SOURCES.items():
        cases = [
            case for source in sources for case in _legacy_cases(fixtures_dir, source)
        ]
        cases.sort(key=lambda case: case["id"])
        bundle = {
            "schema_version": SCHEMA_VERSION,
            "oracle": dict(ORACLE_METADATA),
            "cases": cases,
        }
        validate_bundle(name, bundle)
        bundles[name] = bundle
    return bundles


def _write_bundle(path: Path, bundle: dict) -> None:
    path.write_text(
        json.dumps(bundle, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def import_legacy(fixtures_dir: Path, output_dir: Path) -> int:
    """Consolidate legacy one-case files into the six deterministic bundles."""
    bundles = _bundle_legacy(fixtures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, bundle in bundles.items():
        _write_bundle(output_dir / name, bundle)
    return sum(len(bundle["cases"]) for bundle in bundles.values())


def _expectation_field(case: dict) -> str:
    if "expected" in case:
        return "expected"
    if "expected_r" in case:
        return "expected_r"
    raise ValueError(f"{case.get('kind')}/{case.get('id')}: no expectation field")


def _regenerated_bundles(fixtures_dir: Path, rscript: str) -> dict[str, dict]:
    bundles = load_bundles(fixtures_dir)
    requests: list[dict] = []
    original_expectations: dict[tuple[str, str], dict] = {}
    for bundle in bundles.values():
        for case in bundle["cases"]:
            request = copy.deepcopy(case)
            expectation_field = _expectation_field(case)
            original_expectation = case[expectation_field]
            request["_expectation_field"] = expectation_field
            request["_expected_keys"] = sorted(original_expectation)
            request.pop("expected", None)
            request.pop("expected_r", None)
            requests.append(request)
            original_expectations[(case["kind"], case["id"])] = original_expectation

    results = run_oracle(requests, rscript)
    by_key: dict[tuple[str, str], dict] = {}
    for result in results:
        if not isinstance(result, dict):
            raise RuntimeError("R oracle returned a non-object result")
        key = (result.get("kind"), result.get("id"))
        if not all(isinstance(value, str) for value in key):
            raise RuntimeError("R oracle result is missing string kind/id")
        if key in by_key:
            raise RuntimeError(f"R oracle returned duplicate result {key}")
        if not isinstance(result.get("expected"), dict):
            raise RuntimeError(f"R oracle result {key} has no expected object")
        if key not in original_expectations:
            raise RuntimeError(f"R oracle returned unexpected result {key}")
        by_key[key] = _coerce_numeric_shapes(
            result["expected"], original_expectations[key]
        )

    expected_keys = {
        (case["kind"], case["id"])
        for bundle in bundles.values()
        for case in bundle["cases"]
    }
    actual_keys = set(by_key)
    if expected_keys != actual_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise RuntimeError(
            f"R oracle result mismatch; missing={missing}, extra={extra}"
        )

    regenerated = copy.deepcopy(bundles)
    for bundle in regenerated.values():
        for case in bundle["cases"]:
            case[_expectation_field(case)] = by_key[(case["kind"], case["id"])]
    return regenerated


def _coerce_numeric_shapes(value: Any, template: Any) -> Any:
    """Preserve legacy JSON integer/float spelling while accepting R numbers."""
    if isinstance(template, dict) and isinstance(value, dict):
        if set(value) != set(template):
            raise RuntimeError(
                "R oracle changed expectation fields from "
                f"{sorted(template)!r} to {sorted(value)!r}"
            )
        return {
            key: _coerce_numeric_shapes(value[key], template_value)
            for key, template_value in template.items()
        }
    if isinstance(template, list) and isinstance(value, list):
        if len(value) != len(template):
            raise RuntimeError(
                f"R oracle changed vector length from {len(template)} to {len(value)}"
            )
        return [
            _coerce_numeric_shapes(item, template_item)
            for item, template_item in zip(value, template)
        ]
    if isinstance(template, bool):
        return bool(value)
    if isinstance(template, int):
        return int(value)
    if isinstance(template, float):
        numeric = float(value)
        scale = max(abs(numeric), abs(template), 1.0)
        # R/BLAS versions can reorder floating-point work below the parity gate.
        # Retaining the existing spelling in that range keeps --check genuinely
        # byte-stable while still surfacing any numerically meaningful change.
        if abs(numeric - template) <= 1e-10 * scale:
            return template
        return numeric
    return value


def generate(fixtures_dir: Path, output_dir: Path, rscript: str) -> None:
    """Regenerate all bundles, validate them, then atomically replace outputs."""
    regenerated = _regenerated_bundles(fixtures_dir, rscript)
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="rdrobust-fixtures-") as tmp:
        candidate_dir = Path(tmp)
        backup_dir = candidate_dir / "originals"
        for name, bundle in regenerated.items():
            validate_bundle(name, bundle)
            _write_bundle(candidate_dir / name, bundle)
        load_bundles(candidate_dir)

        backup_dir.mkdir()
        existed: dict[str, bool] = {}
        for name in FAMILY_FILES:
            destination = output_dir / name
            existed[name] = destination.exists()
            if existed[name]:
                shutil.copy2(destination, backup_dir / name)

        replaced: list[str] = []
        try:
            for name in FAMILY_FILES:
                (candidate_dir / name).replace(output_dir / name)
                replaced.append(name)
        except BaseException as replacement_error:
            rollback_errors = []
            for name in reversed(replaced):
                destination = output_dir / name
                try:
                    if existed[name]:
                        (backup_dir / name).replace(destination)
                    else:
                        destination.unlink(missing_ok=True)
                except OSError as rollback_error:
                    rollback_errors.append(f"{name}: {rollback_error}")
            if rollback_errors:
                raise RuntimeError(
                    "fixture replacement failed and rollback was incomplete: "
                    + "; ".join(rollback_errors)
                ) from replacement_error
            raise


def _verify_python_binding(bundles: dict[str, dict]) -> tuple[int, int] | None:
    try:
        numpy = importlib.import_module("numpy")
        binding = importlib.import_module("rdrobust_rust")
    except ImportError:
        return None

    checked = 0
    failures = 0
    for case in bundles["estimation.json"]["cases"]:
        if case["kind"] != "rdrobust" or "expected" not in case:
            continue
        y = numpy.asarray(case["y"], dtype=float)
        x = numpy.asarray(case["x"], dtype=float)
        fuzzy = numpy.asarray(case["t"], dtype=float) if "t" in case else None
        covs = None
        if "z" in case:
            covs = numpy.ascontiguousarray(
                numpy.asarray(case["z"], dtype=float).reshape(-1, case["dz"])
            )
        output = binding.rdrobust(
            y,
            x,
            fuzzy=fuzzy,
            covs=covs,
            c=case["c"],
            p=case["p"],
            q=case["q"],
            deriv=case["deriv"],
            kernel=case["kernel"],
            vce=case["vce"],
            nnmatch=3,
            scaleregul=case["scaleregul"],
            scalepar=case["scalepar"],
            level=case["level"],
            bwselect=case["bwselect"],
            masspoints="off",
            stdvars=False,
            bwrestrict=False,
        )
        expected = case["expected"]
        for index in range(3):
            scale = max(abs(output["coef"][index]), abs(expected["coef"][index]), 1.0)
            if abs(output["coef"][index] - expected["coef"][index]) > 1e-8 * scale:
                failures += 1
        checked += 1
    return checked, failures


def verify(fixtures_dir: Path) -> None:
    """Validate envelopes, uniqueness, and the binding when it is importable."""
    bundles = load_bundles(fixtures_dir)
    seen: set[tuple[str, str]] = set()
    for name, bundle in bundles.items():
        for case in bundle["cases"]:
            key = (case["kind"], case["id"])
            if key in seen:
                raise ValueError(f"{name}: duplicate fixture key {key}")
            seen.add(key)
    binding_result = _verify_python_binding(bundles)
    if binding_result is not None:
        checked, failures = binding_result
        if failures:
            raise RuntimeError(
                f"rdrobust_rust parity: {checked} fixtures, {failures} failures"
            )


def _benchmark_case(n: int = 2_000, seed: int = 1) -> dict[str, Any]:
    numpy = importlib.import_module("numpy")
    rng = numpy.random.default_rng(seed)
    x_array = numpy.sort(rng.uniform(-1.0, 1.0, n))
    y_array = 0.5 * x_array + 2.0 * (x_array >= 0) + rng.standard_normal(n)
    return {
        "id": "benchmark",
        "kind": "rdrobust",
        "_expectation_field": "expected",
        "_expected_keys": ["coef"],
        "x": x_array.tolist(),
        "y": y_array.tolist(),
        "c": 0.0,
        "p": 1,
        "q": 2,
        "deriv": 0,
        "kernel": "tri",
        "vce": "nn",
        "scaleregul": 1.0,
        "scalepar": 1.0,
        "level": 95.0,
        "bwselect": "mserd",
    }


def benchmark(rscript: str) -> None:
    """Benchmark the fixed seeded workload in R and the installed Rust binding."""
    numpy = importlib.import_module("numpy")
    binding = importlib.import_module("rdrobust_rust")
    case = _benchmark_case()
    y = numpy.asarray(case["y"], dtype=float)
    x = numpy.asarray(case["x"], dtype=float)

    rust_times = []
    rust_output = None
    for _ in range(7):
        started = time.perf_counter()
        rust_output = binding.rdrobust(
            y,
            x,
            c=0.0,
            p=1,
            q=2,
            deriv=0,
            kernel="tri",
            vce="nn",
            nnmatch=3,
            scaleregul=1.0,
            scalepar=1.0,
            level=95.0,
            bwselect="mserd",
            masspoints="off",
            stdvars=False,
            bwrestrict=False,
        )
        rust_times.append((time.perf_counter() - started) * 1_000)

    r_times = []
    r_result = None
    for _ in range(3):
        started = time.perf_counter()
        r_result = run_oracle([case], rscript)[0]
        r_times.append((time.perf_counter() - started) * 1_000)

    assert rust_output is not None and r_result is not None
    rust_coef = float(rust_output["coef"][1])
    r_coef = float(r_result["expected"]["coef"][1])
    difference = abs(rust_coef - r_coef)
    print(f"R median: {statistics.median(r_times):.3f} ms")
    print(f"Rust median: {statistics.median(rust_times):.3f} ms")
    print(f"coefficient difference: {difference:.3e}")
    if difference > 1e-8:
        raise RuntimeError(f"coefficient difference {difference:.3e} exceeds 1e-8")


def _check_regeneration(fixtures_dir: Path, rscript: str) -> None:
    regenerated = _regenerated_bundles(fixtures_dir, rscript)
    with tempfile.TemporaryDirectory(prefix="rdrobust-fixtures-check-") as tmp:
        candidate_dir = Path(tmp)
        for name, bundle in regenerated.items():
            _write_bundle(candidate_dir / name, bundle)
        differences = [
            name
            for name in FAMILY_FILES
            if (candidate_dir / name).read_bytes() != (fixtures_dir / name).read_bytes()
        ]
    if differences:
        raise RuntimeError(f"regenerated bundles differ: {', '.join(differences)}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    import_parser = subparsers.add_parser("import-legacy")
    import_parser.add_argument("--source", type=Path, default=DEFAULT_LEGACY)
    import_parser.add_argument("--output", type=Path, default=DEFAULT_FIXTURES)

    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    generate_parser.add_argument("--output", type=Path)
    generate_parser.add_argument("--rscript", default="Rscript")
    generate_parser.add_argument("--check", action="store_true")

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)

    benchmark_parser = subparsers.add_parser("benchmark")
    benchmark_parser.add_argument("--rscript", default="Rscript")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "import-legacy":
            total = import_legacy(args.source, args.output)
            print(f"imported {total} legacy fixtures into {len(FAMILY_FILES)} bundles")
        elif args.command == "generate":
            if args.check:
                _check_regeneration(args.fixtures, args.rscript)
                print("regenerated bundles are byte-identical")
            else:
                generate(args.fixtures, args.output or args.fixtures, args.rscript)
                print(f"regenerated {len(FAMILY_FILES)} fixture bundles")
        elif args.command == "verify":
            verify(args.fixtures)
            bundles = load_bundles(args.fixtures)
            total = sum(len(bundle["cases"]) for bundle in bundles.values())
            print(f"verified {total} fixtures across {len(FAMILY_FILES)} bundles")
        elif args.command == "benchmark":
            benchmark(args.rscript)
        return 0
    except (ImportError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


class BundleValidationTests(unittest.TestCase):
    def test_rejects_wrong_schema_version(self) -> None:
        with self.assertRaises(ValueError):
            validate_bundle(
                "example.json",
                {"schema_version": 2, "oracle": {}, "cases": []},
            )

    def test_accepts_valid_bundle(self) -> None:
        validate_bundle(
            "example.json",
            {
                "schema_version": 1,
                "oracle": {"package": "rdrobust", "version": "4.1.0"},
                "cases": [{"id": "example", "kind": "rdrobust"}],
            },
        )

    def test_generate_rolls_back_when_final_replacement_fails(self) -> None:
        def bundle(case_id: str) -> dict:
            return {
                "schema_version": 1,
                "oracle": {"package": "rdrobust", "version": "4.1.0"},
                "cases": [{"id": case_id, "kind": "rdrobust"}],
            }

        with tempfile.TemporaryDirectory(prefix="rdrobust-rollback-test-") as tmp:
            output_dir = Path(tmp)
            originals = {}
            regenerated = {}
            for index, name in enumerate(FAMILY_FILES):
                _write_bundle(output_dir / name, bundle(f"original-{index}"))
                originals[name] = (output_dir / name).read_bytes()
                regenerated[name] = bundle(f"replacement-{index}")

            real_replace = Path.replace

            def fail_final_replace(source: Path, target: Path) -> Path:
                target = Path(target)
                if (
                    source.parent.name.startswith("rdrobust-fixtures-")
                    and source.name == FAMILY_FILES[-1]
                    and target.parent == output_dir
                ):
                    raise OSError("injected final replacement failure")
                return real_replace(source, target)

            with (
                mock.patch.object(
                    sys.modules[__name__],
                    "_regenerated_bundles",
                    return_value=regenerated,
                ),
                mock.patch.object(Path, "replace", fail_final_replace),
                self.assertRaisesRegex(OSError, "injected final replacement failure"),
            ):
                generate(output_dir, output_dir, "unused")

            self.assertEqual(
                {name: (output_dir / name).read_bytes() for name in FAMILY_FILES},
                originals,
            )


if __name__ == "__main__":
    raise SystemExit(main())
