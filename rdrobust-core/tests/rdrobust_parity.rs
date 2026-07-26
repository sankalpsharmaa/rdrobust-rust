//! End-to-end parity test: the ported `rdrobust` (auto-selecting bandwidth via
//! `rdbwselect`, scoped option set) vs R. Cases come from `fixtures/estimation.json`.

use std::fs;
use std::path::PathBuf;

use rdrobust_core::{rdrobust, Kernel, Vce};
use serde::Deserialize;

#[derive(Deserialize)]
struct Expected {
    coef: Vec<f64>,
    se: Vec<f64>,
    z: Vec<f64>,
    pv: Vec<f64>,
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
    #[serde(default)]
    t: Option<Vec<f64>>,
    #[serde(default)]
    z: Option<Vec<f64>>,
    #[serde(default)]
    dz: usize,
    #[serde(default)]
    covs_drop_coll: bool,
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

fn kernel(s: &str) -> Kernel {
    match s {
        "tri" => Kernel::Triangular,
        "epa" => Kernel::Epanechnikov,
        "uni" => Kernel::Uniform,
        other => panic!("kernel {other}"),
    }
}

fn vce(s: &str) -> Vce {
    match s {
        "hc0" => Vce::Hc0,
        "hc1" => Vce::Hc1,
        "hc2" => Vce::Hc2,
        "hc3" => Vce::Hc3,
        "nn" => Vce::NearestNeighbor,
        other => panic!("vce {other}"),
    }
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-8 * a.abs().max(b.abs()).max(1.0)
}

#[test]
fn rdrobust_matches_r() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../fixtures/estimation.json");
    let bundle: FixtureBundle = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
    assert_eq!(bundle.schema_version, 1);
    assert_eq!(bundle.oracle.package, "rdrobust");
    assert_eq!(bundle.oracle.version, "1.3.0");
    let mut n = 0;
    for case in bundle
        .cases
        .into_iter()
        .filter(|case| case["kind"] == "rdrobust")
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
            f.covs_drop_coll,
            f.scaleregul,
            f.scalepar,
            f.level,
            &f.bwselect,
            None,
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
                close(out.z[i], f.expected.z[i]),
                "{} z[{}]: {} vs R {}",
                f.id,
                i,
                out.z[i],
                f.expected.z[i]
            );
            assert!(
                close(out.pv[i], f.expected.pv[i]),
                "{} pv[{}]: {} vs R {}",
                f.id,
                i,
                out.pv[i],
                f.expected.pv[i]
            );
            assert!(
                close(out.ci[i][0], f.expected.ci_l[i]),
                "{} ci_l[{}]: {} vs R {}",
                f.id,
                i,
                out.ci[i][0],
                f.expected.ci_l[i]
            );
            assert!(
                close(out.ci[i][1], f.expected.ci_u[i]),
                "{} ci_u[{}]: {} vs R {}",
                f.id,
                i,
                out.ci[i][1],
                f.expected.ci_u[i]
            );
        }
        assert!(
            close(out.bws[0][0], f.expected.bws_h[0]) && close(out.bws[0][1], f.expected.bws_h[1]),
            "{} bws_h",
            f.id
        );
        assert!(
            close(out.bws[1][0], f.expected.bws_b[0]) && close(out.bws[1][1], f.expected.bws_b[1]),
            "{} bws_b",
            f.id
        );
        n += 1;
    }
    assert!(n >= 8, "expected >= 8 rdrobust fixtures, found {n}");
    eprintln!("rdrobust parity: {n} fixtures matched R to 1e-8");
}
