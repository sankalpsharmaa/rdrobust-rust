//! rho parity: `rdrobust(..., rho=, h=)` vs R-canonical cases in
//! `fixtures/bandwidth.json`. Auto-bw path replaces the selected b with h/rho
//! (rdrobust.R:501-504); manual-h path sets b = h/rho (R:296-300). Gate ~1e-8.

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
    rho: f64,
    #[serde(default)]
    h: Option<f64>,
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
fn rdrobust_rho_matches_r() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../fixtures/bandwidth.json");
    let bundle: FixtureBundle = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
    assert_eq!(bundle.schema_version, 1);
    assert_eq!(bundle.oracle.package, "rdrobust");
    assert_eq!(bundle.oracle.version, "1.3.0");
    let mut n = 0;
    for case in bundle
        .cases
        .into_iter()
        .filter(|case| case["kind"] == "rho")
    {
        let f: Fixture = serde_json::from_value(case).unwrap();
        let h_pair = f.h.map(|v| (v, v));
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
            None,
            None,
            "off",
            false,
            false,
            None,
            Some(f.rho),
            false,
            h_pair,
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
        assert!(
            close(out.bws[1][0], f.expected.bws_b[0]),
            "{} b_l: {} vs R {}",
            f.id,
            out.bws[1][0],
            f.expected.bws_b[0]
        );
        n += 1;
    }
    assert!(n >= 6, "expected >=6 rho fixtures, found {n}");
}
