//! Data-prep parity: `rdrobust` under the default-ON prep options (`masspoints`,
//! `stdvars`, `bwrestrict`, `bwcheck`) vs R-canonical cases in
//! `fixtures/preprocessing.json`. Covers the continuous regime (where stdvars/bwrestrict
//! are scale-equivariant no-ops) and a tie-heavy regime that trips
//! `masspoints="adjust"` and the auto `bwcheck=10` floor. Gate ~1e-8 relative.

use std::fs;
use std::path::PathBuf;

use rdrobust_core::{rdrobust, Kernel, Vce};
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
    masspoints: String,
    stdvars: bool,
    bwrestrict: bool,
    bwcheck: Option<usize>,
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
        other => panic!("unknown vce {other}"),
    }
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-8 * a.abs().max(b.abs()).max(1.0)
}

#[test]
fn rdrobust_prep_matches_r() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../fixtures/preprocessing.json");
    let bundle: FixtureBundle = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
    assert_eq!(bundle.schema_version, 1);
    assert_eq!(bundle.oracle.package, "rdrobust");
    assert_eq!(bundle.oracle.version, "4.1.0");
    let mut n = 0;
    for case in bundle
        .cases
        .into_iter()
        .filter(|case| case["kind"] == "prep" && case.get("expected").is_some())
    {
        let f: Fixture = serde_json::from_value(case).unwrap();
        let out = rdrobust(
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
            f.scalepar,
            f.level,
            &f.bwselect,
            None,
            None,
            &f.masspoints,
            f.stdvars,
            f.bwrestrict,
            f.bwcheck,
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
        assert!(
            close(out.bws[0][0], f.expected.bws_h[0]),
            "{} h_l: {} vs R {}",
            f.id,
            out.bws[0][0],
            f.expected.bws_h[0]
        );
        assert!(close(out.bws[0][1], f.expected.bws_h[1]), "{} h_r", f.id);
        assert!(
            close(out.bws[1][0], f.expected.bws_b[0]),
            "{} b_l: {} vs R {}",
            f.id,
            out.bws[1][0],
            f.expected.bws_b[0]
        );
        assert!(close(out.bws[1][1], f.expected.bws_b[1]), "{} b_r", f.id);
        n += 1;
    }
    assert!(n >= 10, "expected >=10 prep fixtures, found {n}");
}
