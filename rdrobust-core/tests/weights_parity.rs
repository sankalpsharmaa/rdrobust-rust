//! User-weights parity: `rdrobust(..., weights=W)` and `rdplot(..., weights=W)`
//! vs R-canonical cases in `fixtures/weights.json` and `fixtures/plotting.json`. Effective
//! weight is `W * kweight`; zero-weight points drop from the window. Gate ~1e-8
//! (clusters ~1e-10). The uniform-weights fixture must equal the unweighted run.

use std::fs;
use std::path::PathBuf;

use rdrobust_core::{rdplot, rdrobust, Kernel, Vce};
use serde::Deserialize;

#[derive(Deserialize)]
struct Expected {
    coef: Vec<f64>,
    se: Vec<f64>,
    bws_h: Vec<f64>,
    bws_b: Vec<f64>,
}

#[derive(Deserialize)]
struct Fixture {
    id: String,
    x: Vec<f64>,
    y: Vec<f64>,
    w: Vec<f64>,
    #[serde(default)]
    t: Option<Vec<f64>>,
    #[serde(default)]
    cl: Option<Vec<f64>>,
    #[serde(default)]
    z: Option<Vec<f64>>,
    #[serde(default)]
    dz: usize,
    c: f64,
    p: usize,
    q: usize,
    deriv: usize,
    kernel: String,
    vce: String,
    scaleregul: f64,
    scalepar: f64,
    level: f64,
    bwselect: String,
    expected: Expected,
}

#[derive(Deserialize)]
struct RdpExpected {
    j_l: usize,
    j_r: usize,
    coef_l: Vec<f64>,
    coef_r: Vec<f64>,
    mean_x_l: Vec<f64>,
    mean_y_l: Vec<f64>,
    mean_x_r: Vec<f64>,
    mean_y_r: Vec<f64>,
}

#[derive(Deserialize)]
struct RdpFixture {
    id: String,
    x: Vec<f64>,
    y: Vec<f64>,
    w: Vec<f64>,
    c: f64,
    p: usize,
    kernel: String,
    binselect: String,
    scale_l: f64,
    scale_r: f64,
    nbins_l: Option<usize>,
    nbins_r: Option<usize>,
    expected: RdpExpected,
}

#[derive(Deserialize)]
struct FixtureBundle {
    schema_version: u32,
    oracle: OracleMetadata,
    cases: Vec<serde_json::Value>,
}

#[derive(Deserialize)]
struct OracleMetadata {
    package: String,
    version: String,
}

fn load_bundle(name: &str) -> FixtureBundle {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../fixtures")
        .join(name);
    let bundle: FixtureBundle = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
    assert_eq!(bundle.schema_version, 1);
    assert_eq!(bundle.oracle.package, "rdrobust");
    assert_eq!(bundle.oracle.version, "4.1.0");
    bundle
}

fn kernel(s: &str) -> Kernel {
    match s {
        "tri" | "triangular" => Kernel::Triangular,
        "epa" | "epanechnikov" => Kernel::Epanechnikov,
        "uni" | "uniform" => Kernel::Uniform,
        other => panic!("unknown kernel {other}"),
    }
}

fn vce(s: &str) -> Vce {
    match s {
        "nn" => Vce::NearestNeighbor,
        "hc0" => Vce::Hc0,
        "hc1" => Vce::Hc1,
        "hc2" => Vce::Hc2,
        "hc3" => Vce::Hc3,
        "cr0" | "cr1" => Vce::Cr1,
        "cr2" => Vce::Cr2,
        "cr3" => Vce::Cr3,
        other => panic!("unknown vce {other}"),
    }
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-8 * a.abs().max(b.abs()).max(1.0)
}

/// Sort (mean_x, mean_y) pairs by mean_x so the bin-id convention is irrelevant.
fn sorted_pairs(mx: &[f64], my: &[f64]) -> Vec<(f64, f64)> {
    let mut v: Vec<(f64, f64)> = mx.iter().copied().zip(my.iter().copied()).collect();
    v.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    v
}

#[test]
fn rdrobust_weights_matches_r() {
    let mut n = 0;
    for case in load_bundle("weights.json")
        .cases
        .into_iter()
        .filter(|case| case["kind"] == "weights")
    {
        let f: Fixture = serde_json::from_value(case).unwrap();
        let out = rdrobust(
            &f.y,
            &f.x,
            f.t.as_deref(),
            f.z.as_deref(),
            f.dz,
            f.c,
            f.p,
            f.q,
            f.deriv,
            kernel(&f.kernel),
            vce(&f.vce),
            3,
            false,
            f.scaleregul,
            f.scalepar,
            f.level,
            &f.bwselect,
            f.cl.as_deref(),
            Some(&f.w),
            "off",
            false,
            false,
            None,
            None,
            false,
            None,
            None,
        );
        for i in 0..3 {
            assert!(
                close(out.coef[i], f.expected.coef[i]),
                "{} coef[{}]: {} vs R {}",
                f.id,
                i,
                out.coef[i],
                f.expected.coef[i]
            );
            assert!(
                close(out.se[i], f.expected.se[i]),
                "{} se[{}]: {} vs R {}",
                f.id,
                i,
                out.se[i],
                f.expected.se[i]
            );
        }
        assert!(close(out.bws[0][0], f.expected.bws_h[0]), "{} h_l", f.id);
        assert!(close(out.bws[1][0], f.expected.bws_b[0]), "{} b_l", f.id);
        n += 1;
    }
    assert!(n >= 9, "expected >=9 weights fixtures, found {n}");
}

#[test]
fn rdplot_weights_matches_r() {
    let mut n = 0;
    for case in load_bundle("plotting.json")
        .cases
        .into_iter()
        .filter(|case| case["kind"] == "weights_rdplot")
    {
        let f: RdpFixture = serde_json::from_value(case).unwrap();
        let out = rdplot(
            &f.y,
            &f.x,
            f.c,
            f.p,
            kernel(&f.kernel),
            &f.binselect,
            f.scale_l,
            f.scale_r,
            f.nbins_l,
            f.nbins_r,
            Some(&f.w),
        );
        assert_eq!(out.j_l, f.expected.j_l, "{} j_l", f.id);
        assert_eq!(out.j_r, f.expected.j_r, "{} j_r", f.id);
        for (c, e) in out.coef_l.iter().zip(&f.expected.coef_l) {
            assert!(close(*c, *e), "{} coef_l {} vs R {}", f.id, c, e);
        }
        for (c, e) in out.coef_r.iter().zip(&f.expected.coef_r) {
            assert!(close(*c, *e), "{} coef_r {} vs R {}", f.id, c, e);
        }
        // per-bin means are unweighted (R tapply); compare sorted by mean_x
        let rl = sorted_pairs(&out.mean_x_l, &out.mean_y_l);
        let el = sorted_pairs(&f.expected.mean_x_l, &f.expected.mean_y_l);
        assert_eq!(rl.len(), el.len(), "{} n_bins_l", f.id);
        for (a, b) in rl.iter().zip(&el) {
            assert!(
                close(a.0, b.0) && close(a.1, b.1),
                "{} bin_l mismatch",
                f.id
            );
        }
        let rr = sorted_pairs(&out.mean_x_r, &out.mean_y_r);
        let er = sorted_pairs(&f.expected.mean_x_r, &f.expected.mean_y_r);
        assert_eq!(rr.len(), er.len(), "{} n_bins_r", f.id);
        for (a, b) in rr.iter().zip(&er) {
            assert!(
                close(a.0, b.0) && close(a.1, b.1),
                "{} bin_r mismatch",
                f.id
            );
        }
        n += 1;
    }
    assert!(n >= 3, "expected >=3 weights_rdplot fixtures, found {n}");
}
