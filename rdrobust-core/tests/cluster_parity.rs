//! Cluster-robust parity: `rdrobust(..., cluster=C)` and `rdbwselect(..., cluster=C)`
//! across CR1 (clustered hc/nn), CRV2, and CRV3, vs R-canonical cases in
//! `fixtures/clustering.json`. CR1 sums and the CRV2/CRV3
//! eigendecompositions reorder floating-point ops relative to R's LAPACK, so the
//! gate is ~1e-8 relative (spec 05 §1 documents bit-parity as unachievable).

use std::fs;
use std::path::PathBuf;

use rdrobust_core::{rdbwselect, rdrobust, Kernel, Vce};
use serde::Deserialize;

#[derive(Deserialize)]
struct Expected {
    coef: Vec<f64>,
    se: Vec<f64>,
    ci_l: Vec<f64>,
    ci_u: Vec<f64>,
    bws_h: Vec<f64>,
    bws_b: Vec<f64>,
}

#[derive(Deserialize)]
struct Fixture {
    id: String,
    x: Vec<f64>,
    y: Vec<f64>,
    cl: Vec<f64>,
    #[serde(default)]
    t: Option<Vec<f64>>,
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
struct BwExpected {
    h_l: f64,
    h_r: f64,
    b_l: f64,
    b_r: f64,
}

#[derive(Deserialize)]
struct BwFixture {
    id: String,
    x: Vec<f64>,
    y: Vec<f64>,
    cl: Vec<f64>,
    c: f64,
    p: usize,
    q: usize,
    deriv: usize,
    kernel: String,
    vce: String,
    scaleregul: f64,
    bwselect: String,
    expected: BwExpected,
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

fn load_bundle() -> FixtureBundle {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../fixtures/clustering.json");
    let bundle: FixtureBundle = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
    assert_eq!(bundle.schema_version, 1);
    assert_eq!(bundle.oracle.package, "rdrobust");
    assert_eq!(bundle.oracle.version, "1.3.0");
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

/// Map the user-facing vce string to the residual/method enum. The cluster
/// vector (passed separately) is what activates clustering: hc/nn -> CR1 meat,
/// cr2/cr3 -> CRV2/CRV3 meat.
fn vce(s: &str) -> Vce {
    match s {
        "nn" => Vce::NearestNeighbor,
        "hc0" => Vce::Hc0,
        "hc1" => Vce::Hc1,
        "hc2" => Vce::Hc2,
        "hc3" => Vce::Hc3,
        // cr0/cr1: CR1 meat with a raw residual (hii=0 -> w=1 in rdrobust_res)
        "cr0" | "cr1" => Vce::Cr1,
        "cr2" => Vce::Cr2,
        "cr3" => Vce::Cr3,
        other => panic!("unknown vce {other}"),
    }
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-8 * a.abs().max(b.abs()).max(1.0)
}

#[test]
fn rdrobust_cluster_matches_r() {
    let mut n = 0;
    for case in load_bundle()
        .cases
        .into_iter()
        .filter(|case| case["kind"] == "cluster")
    {
        let f: Fixture = serde_json::from_value(case).unwrap();
        let out = rdrobust(
            &f.y,
            &f.x,
            f.t.as_deref(),
            None,
            0,
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
            Some(&f.cl),
            None,
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
            assert!(
                close(out.ci[i][0], f.expected.ci_l[i]) && close(out.ci[i][1], f.expected.ci_u[i]),
                "{} ci[{}]: [{},{}] vs R [{},{}]",
                f.id,
                i,
                out.ci[i][0],
                out.ci[i][1],
                f.expected.ci_l[i],
                f.expected.ci_u[i]
            );
        }
        assert!(close(out.bws[0][0], f.expected.bws_h[0]), "{} h_l", f.id);
        assert!(close(out.bws[0][1], f.expected.bws_h[1]), "{} h_r", f.id);
        assert!(close(out.bws[1][0], f.expected.bws_b[0]), "{} b_l", f.id);
        assert!(close(out.bws[1][1], f.expected.bws_b[1]), "{} b_r", f.id);
        n += 1;
    }
    assert!(n >= 10, "expected >=10 cluster fixtures, found {n}");
}

#[test]
fn rdbwselect_cluster_matches_r() {
    let mut n = 0;
    for case in load_bundle()
        .cases
        .into_iter()
        .filter(|case| case["kind"] == "cluster_bw")
    {
        let f: BwFixture = serde_json::from_value(case).unwrap();
        let out = rdbwselect(
            &f.y,
            &f.x,
            None,
            None,
            0,
            f.c,
            f.p,
            f.q,
            f.deriv,
            kernel(&f.kernel),
            vce(&f.vce),
            3,
            false,
            f.scaleregul,
            &f.bwselect,
            Some(&f.cl),
            None,
            "off",
            false,
            false,
            None,
        );
        assert!(
            close(out.h_l, f.expected.h_l),
            "{} h_l: {} vs R {}",
            f.id,
            out.h_l,
            f.expected.h_l
        );
        assert!(
            close(out.h_r, f.expected.h_r),
            "{} h_r: {} vs R {}",
            f.id,
            out.h_r,
            f.expected.h_r
        );
        assert!(
            close(out.b_l, f.expected.b_l),
            "{} b_l: {} vs R {}",
            f.id,
            out.b_l,
            f.expected.b_l
        );
        assert!(
            close(out.b_r, f.expected.b_r),
            "{} b_r: {} vs R {}",
            f.id,
            out.b_r,
            f.expected.b_r
        );
        n += 1;
    }
    assert!(n >= 5, "expected >=5 cluster bw fixtures, found {n}");
}
