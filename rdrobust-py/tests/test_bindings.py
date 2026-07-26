#!/usr/bin/env python3
"""End-to-end checks for the rdrobust_rust Python bindings."""

import json
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np
import rdrobust_rust


FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "estimation.json"
X = np.linspace(-1.0, 1.0, 40)
Y = 0.5 * X + (X >= 0.0)


def close(a, b):
    return abs(a - b) <= 1e-8 * max(abs(a), abs(b), 1.0)


def assert_value_error(name, fn):
    try:
        fn()
    except ValueError:
        return
    except BaseException as exc:
        raise AssertionError(
            f"{name}: expected ValueError, got {type(exc).__name__}: {exc}"
        ) from exc
    raise AssertionError(f"{name}: expected ValueError, call succeeded")


def assert_finite_result(name, out):
    values = []
    for key in ("coef", "se", "z", "pv"):
        values.extend(out[key])
    values.extend(value for row in out["ci"] for value in row)
    values.extend(value for row in out["bws"] for value in row)
    for value in values:
        if not math.isfinite(value):
            raise AssertionError(f"{name}: non-finite result: {out}")


def assert_close_dict(name, got, exp):
    for key in ("coef", "se", "z", "pv"):
        for i, (a, b) in enumerate(zip(got[key], exp[key])):
            if abs(a - b) > 1e-10 * max(abs(a), abs(b), 1.0):
                raise AssertionError(f"{name} {key}[{i}]: {a} != {b}")
    for key in ("ci", "bws"):
        for i, (row_g, row_e) in enumerate(zip(got[key], exp[key])):
            for j, (a, b) in enumerate(zip(row_g, row_e)):
                if abs(a - b) > 1e-10 * max(abs(a), abs(b), 1.0):
                    raise AssertionError(f"{name} {key}[{i}][{j}]: {a} != {b}")


def make_batch(batch=5, n=180):
    xs = []
    ys = []
    for seed in range(batch):
        rng = np.random.default_rng(seed)
        x = np.sort(rng.uniform(-1.0, 1.0, n))
        y = 0.4 * x + 1.5 * (x >= 0.0) + 0.05 * np.sin(np.arange(n) + seed)
        perm = rng.permutation(n)
        xs.append(x[perm])
        ys.append(y[perm])
    return np.ascontiguousarray(ys), np.ascontiguousarray(xs)


def test_import_smoke() -> None:
    for name in ("rdrobust", "rdplot", "rdrobust_batch", "rdrobust_multi_cutoff"):
        if not callable(getattr(rdrobust_rust, name, None)):
            raise AssertionError(f"rdrobust_rust.{name} is not callable")


def test_r_parity() -> None:
    bundle = json.loads(FIXTURES.read_text())
    assert bundle["schema_version"] == 1
    cases = [case for case in bundle["cases"] if case["kind"] == "rdrobust"]
    assert len(cases) == 17

    for f in cases:
        y = np.asarray(f["y"], dtype=float)
        x = np.asarray(f["x"], dtype=float)
        t = np.asarray(f["t"], dtype=float) if "t" in f else None
        covs = (
            np.ascontiguousarray(np.asarray(f["z"], dtype=float).reshape(-1, f["dz"]))
            if "z" in f
            else None
        )
        out = rdrobust_rust.rdrobust(
            y,
            x,
            fuzzy=t,
            covs=covs,
            c=f["c"],
            p=f["p"],
            q=f["q"],
            deriv=f["deriv"],
            kernel=f["kernel"],
            vce=f["vce"],
            nnmatch=3,
            scaleregul=f["scaleregul"],
            scalepar=f["scalepar"],
            level=f["level"],
            bwselect=f["bwselect"],
            masspoints="off",
            stdvars=False,
            bwrestrict=False,
        )
        exp = f["expected"]
        for i in range(3):
            for key in ("coef", "se", "z", "pv"):
                if not close(out[key][i], exp[key][i]):
                    raise AssertionError(
                        f"{f['id']} {key}[{i}]: {out[key][i]} vs R {exp[key][i]}"
                    )
            if not close(out["ci"][i][0], exp["ci_l"][i]):
                raise AssertionError(f"{f['id']} ci_l[{i}]")
            if not close(out["ci"][i][1], exp["ci_u"][i]):
                raise AssertionError(f"{f['id']} ci_u[{i}]")


def test_supported_configurations() -> None:
    assert_finite_result("baseline", rdrobust_rust.rdrobust(Y, X))

    n = 240
    x = np.linspace(-1.0, 1.0, n)
    y = 0.4 * x + 0.25 * x * x + 1.5 * (x >= 0.0) + 0.05 * np.sin(np.arange(n))
    fuzzy = 0.25 + 0.5 * (x >= 0.0) + 0.05 * np.cos(np.arange(n))
    covs = np.ascontiguousarray(np.column_stack([np.sin(x), x * x]))
    weights = 0.5 + 0.25 * (1.0 + np.cos(np.arange(n)))
    cluster = (np.arange(n) // 6).astype(float)

    configs = []
    for bwselect in (
        "mserd",
        "msesum",
        "msetwo",
        "msecomb1",
        "msecomb2",
        "cerrd",
        "cersum",
        "certwo",
        "cercomb1",
        "cercomb2",
    ):
        configs.append((f"bwselect_{bwselect}", {"bwselect": bwselect}))
    for kernel in ("tri", "epa", "uni"):
        configs.append((f"kernel_{kernel}", {"kernel": kernel}))
    for vce in ("nn", "hc0", "hc1", "hc2", "hc3"):
        configs.append((f"vce_{vce}", {"vce": vce}))
    configs.extend(
        [
            ("weights", {"weights": weights, "vce": "hc1"}),
            ("covs", {"covs": covs, "vce": "hc1"}),
            ("fuzzy", {"fuzzy": fuzzy, "vce": "hc1"}),
            (
                "fuzzy_covs_weights",
                {"fuzzy": fuzzy, "covs": covs, "weights": weights, "vce": "hc2"},
            ),
            ("cluster_cr1", {"cluster": cluster, "vce": "cr1"}),
            ("cluster_cr2", {"cluster": cluster, "vce": "cr2"}),
            ("cluster_cr3", {"cluster": cluster, "vce": "cr3"}),
            ("rho_manual", {"rho": 0.8, "h": 0.45}),
        ]
    )

    for name, kwargs in configs:
        assert_finite_result(name, rdrobust_rust.rdrobust(y, x, **kwargs))
    for binselect in ("es", "espr", "esmv", "esmvpr"):
        out = rdrobust_rust.rdplot(y, x, binselect=binselect, weights=weights)
        if out["j_l"] < 1 or out["j_r"] < 1:
            raise AssertionError(
                f"rdplot_{binselect}: invalid bin counts {out['j_l']}, {out['j_r']}"
            )


def test_invalid_inputs() -> None:
    y, x = make_batch(batch=2, n=80)
    cases = {
        "y_x_length_mismatch_short_y": lambda: rdrobust_rust.rdrobust(Y[:-1], X),
        "y_x_length_mismatch_short_x": lambda: rdrobust_rust.rdrobust(Y, X[:-1]),
        "fuzzy_length_mismatch": lambda: rdrobust_rust.rdrobust(
            Y, X, fuzzy=np.ones(X.size - 1)
        ),
        "cluster_length_mismatch": lambda: rdrobust_rust.rdrobust(
            Y, X, cluster=np.ones(X.size - 1)
        ),
        "weights_length_mismatch": lambda: rdrobust_rust.rdrobust(
            Y, X, weights=np.ones(X.size - 1)
        ),
        "covariate_row_mismatch": lambda: rdrobust_rust.rdrobust(
            Y, X, covs=np.ones((X.size - 1, 2))
        ),
        "nonfinite_x": lambda: rdrobust_rust.rdrobust(Y, np.r_[X[:-1], np.nan]),
        "nonfinite_y": lambda: rdrobust_rust.rdrobust(np.r_[Y[:-1], np.inf], X),
        "no_left_of_cutoff": lambda: rdrobust_rust.rdrobust(Y, X + 2.0),
        "no_right_of_cutoff": lambda: rdrobust_rust.rdrobust(Y, X - 2.0),
        "negative_weight": lambda: rdrobust_rust.rdrobust(
            Y, X, weights=np.r_[np.ones(X.size - 1), -1.0]
        ),
        "rho_zero": lambda: rdrobust_rust.rdrobust(Y, X, rho=0.0),
        "manual_h_zero": lambda: rdrobust_rust.rdrobust(Y, X, h=0.0),
        "manual_b_negative": lambda: rdrobust_rust.rdrobust(Y, X, h=0.5, b=-0.5),
        "level_100": lambda: rdrobust_rust.rdrobust(Y, X, level=100.0),
        "nnmatch_zero": lambda: rdrobust_rust.rdrobust(Y, X, nnmatch=0),
        "bad_bwselect": lambda: rdrobust_rust.rdrobust(Y, X, bwselect="bogus"),
        "bad_masspoints": lambda: rdrobust_rust.rdrobust(Y, X, masspoints="bogus"),
        "bad_order_q": lambda: rdrobust_rust.rdrobust(Y, X, p=2, q=2),
        "bad_order_deriv": lambda: rdrobust_rust.rdrobust(Y, X, p=1, deriv=2),
        "rdplot_y_x_length_mismatch": lambda: rdrobust_rust.rdplot(Y[:-1], X),
        "rdplot_weights_length_mismatch": lambda: rdrobust_rust.rdplot(
            Y, X, weights=np.ones(X.size - 1)
        ),
        "rdplot_bad_binselect": lambda: rdrobust_rust.rdplot(Y, X, binselect="qs"),
        "rdplot_nonfinite_x": lambda: rdrobust_rust.rdplot(Y, np.r_[X[:-1], np.nan]),
        "rdplot_no_left_of_cutoff": lambda: rdrobust_rust.rdplot(Y, X + 2.0),
        "rdplot_no_right_of_cutoff": lambda: rdrobust_rust.rdplot(Y, X - 2.0),
        "x_shape_mismatch": lambda: rdrobust_rust.rdrobust_batch(y, x[:, :-1]),
        "batch_fuzzy_shape_mismatch": lambda: rdrobust_rust.rdrobust_batch(
            y, x, fuzzy=x[:, :-1]
        ),
        "batch_cluster_shape_mismatch": lambda: rdrobust_rust.rdrobust_batch(
            y, x, cluster=x[:, :-1]
        ),
        "batch_weights_shape_mismatch": lambda: rdrobust_rust.rdrobust_batch(
            y, x, weights=x[:, :-1]
        ),
        "batch_negative_weight": lambda: rdrobust_rust.rdrobust_batch(
            y, x, weights=-np.ones_like(x)
        ),
        "bad_cutoff_split": lambda: rdrobust_rust.rdrobust_multi_cutoff(
            y[0], x[0], np.array([2.0])
        ),
    }
    assert len(cases) == 32
    for name, fn in cases.items():
        assert_value_error(name, fn)


def test_batch_matches_serial() -> None:
    y, x = make_batch()
    got = rdrobust_rust.rdrobust_batch(
        y, x, vce="hc1", masspoints="off", stdvars=False, bwrestrict=False
    )
    if len(got) != y.shape[0]:
        raise AssertionError(f"batch length {len(got)} != {y.shape[0]}")
    for i, out in enumerate(got):
        assert_finite_result(f"batch[{i}]", out)
        exp = rdrobust_rust.rdrobust(
            y[i], x[i], vce="hc1", masspoints="off", stdvars=False, bwrestrict=False
        )
        assert_close_dict(f"batch[{i}]", out, exp)


def test_multi_cutoff_matches_serial() -> None:
    rng = np.random.default_rng(11)
    x = np.sort(rng.uniform(-1.0, 1.0, 240))
    y = 0.2 * x + 1.0 * (x >= 0.0) + 0.02 * np.cos(np.arange(x.size))
    cutoffs = np.array([-0.35, 0.0, 0.35])
    got = rdrobust_rust.rdrobust_multi_cutoff(
        y, x, cutoffs, vce="hc0", masspoints="off", stdvars=False, bwrestrict=False
    )
    if len(got) != cutoffs.size:
        raise AssertionError(f"multi-cutoff length {len(got)} != {cutoffs.size}")
    for i, cutoff in enumerate(cutoffs):
        assert_finite_result(f"multi_cutoff[{i}]", got[i])
        exp = rdrobust_rust.rdrobust(
            y,
            x,
            c=float(cutoff),
            vce="hc0",
            masspoints="off",
            stdvars=False,
            bwrestrict=False,
        )
        assert_close_dict(f"multi_cutoff[{i}]", got[i], exp)


def test_optional_inputs_match_serial() -> None:
    y, x = make_batch(batch=3, n=240)
    idx = np.arange(x.shape[1], dtype=float)
    fuzzy = np.ascontiguousarray(0.25 + 0.55 * (x >= 0.0) + 0.02 * np.sin(idx)[None, :])
    covs = np.ascontiguousarray(np.stack([np.sin(x), x * x], axis=2))
    weights = np.ascontiguousarray(
        np.broadcast_to(0.75 + 0.2 * (1.0 + np.cos(idx))[None, :], x.shape)
    )
    got = rdrobust_rust.rdrobust_batch(
        y,
        x,
        fuzzy=fuzzy,
        covs=covs,
        weights=weights,
        vce="hc1",
        masspoints="off",
        stdvars=False,
        bwrestrict=False,
    )
    for i, out in enumerate(got):
        exp = rdrobust_rust.rdrobust(
            y[i],
            x[i],
            fuzzy=fuzzy[i],
            covs=covs[i],
            weights=weights[i],
            vce="hc1",
            masspoints="off",
            stdvars=False,
            bwrestrict=False,
        )
        assert_close_dict(f"batch_optional[{i}]", out, exp)

    cluster = np.ascontiguousarray(
        np.broadcast_to((idx // 8).astype(float), x.shape)
        + np.arange(x.shape[0])[:, None] * 1000.0
    )
    got_cluster = rdrobust_rust.rdrobust_batch(
        y,
        x,
        cluster=cluster,
        vce="cr1",
        masspoints="off",
        stdvars=False,
        bwrestrict=False,
    )
    for i, out in enumerate(got_cluster):
        exp = rdrobust_rust.rdrobust(
            y[i],
            x[i],
            cluster=cluster[i],
            vce="cr1",
            masspoints="off",
            stdvars=False,
            bwrestrict=False,
        )
        assert_close_dict(f"batch_cluster[{i}]", out, exp)

    rng = np.random.default_rng(13)
    x = np.sort(rng.uniform(-1.0, 1.0, 260))
    y = 0.3 * x + 1.2 * (x >= 0.0) + 0.02 * np.sin(np.arange(x.size))
    fuzzy = 0.25 + 0.55 * (x >= 0.0) + 0.02 * np.cos(np.arange(x.size))
    covs = np.ascontiguousarray(np.column_stack([np.sin(x), x * x]))
    weights = 0.75 + 0.2 * (1.0 + np.cos(np.arange(x.size)))
    cutoffs = np.array([-0.25, 0.0, 0.25])
    got = rdrobust_rust.rdrobust_multi_cutoff(
        y,
        x,
        cutoffs,
        fuzzy=fuzzy,
        covs=covs,
        weights=weights,
        vce="hc1",
        masspoints="off",
        stdvars=False,
        bwrestrict=False,
    )
    for i, cutoff in enumerate(cutoffs):
        exp = rdrobust_rust.rdrobust(
            y,
            x,
            c=float(cutoff),
            fuzzy=fuzzy,
            covs=covs,
            weights=weights,
            vce="hc1",
            masspoints="off",
            stdvars=False,
            bwrestrict=False,
        )
        assert_close_dict(f"multi_cutoff_optional[{i}]", got[i], exp)


def test_gil_release() -> None:
    y, x = make_batch(batch=256, n=1600)
    progress = 0
    running = True

    def spin():
        nonlocal progress
        while running:
            progress += 1

    thread = threading.Thread(target=spin)
    thread.start()
    deadline = time.time() + 1.0
    while progress == 0 and time.time() < deadline:
        time.sleep(0.001)

    before = progress
    rdrobust_rust.rdrobust_batch(
        y, x, masspoints="off", stdvars=False, bwrestrict=False
    )
    after = progress
    running = False
    thread.join(timeout=1.0)
    if after <= before:
        raise AssertionError(
            "background Python thread made no progress during rdrobust_batch"
        )


TESTS = (
    test_import_smoke,
    test_r_parity,
    test_supported_configurations,
    test_invalid_inputs,
    test_batch_matches_serial,
    test_multi_cutoff_matches_serial,
    test_optional_inputs_match_serial,
    test_gil_release,
)


def main() -> int:
    try:
        for test in TESTS:
            test()
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(
        "rdrobust_rust bindings: 17 parity cases, 30 supported configurations, "
        "32 invalid-input cases, batch, multi-cutoff, optional inputs, and GIL release passed"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
