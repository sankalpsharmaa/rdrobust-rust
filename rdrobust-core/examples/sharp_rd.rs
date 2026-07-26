//! A single sharp regression-discontinuity fit, printing the conventional,
//! bias-corrected, and robust estimates with their standard errors and intervals.
//!
//! Run with: `cargo run --release --example sharp_rd`

use rdrobust_core::{rdrobust_batch, Dataset, RdRobustOptions};

fn main() {
    // Sharp RD: the outcome jumps by 2 at the cutoff c = 0, plus a smooth trend
    // and a deterministic wiggle so the variance is well defined.
    let n = 1000;
    let x: Vec<f64> = (0..n)
        .map(|i| -1.0 + 2.0 * (i as f64 + 0.5) / n as f64)
        .collect();
    let y: Vec<f64> = x
        .iter()
        .enumerate()
        .map(|(i, &xi)| 0.5 * xi + if xi >= 0.0 { 2.0 } else { 0.0 } + 0.5 * (i as f64 * 0.7).sin())
        .collect();

    // RdRobustOptions::default mirrors the R defaults (p=1, q=2, triangular, nn).
    let opts = RdRobustOptions::default();
    let ds = Dataset {
        y: &y,
        x: &x,
        t: None,
        z: None,
        dz: 0,
        cluster: None,
        weights: None,
    };
    let out = &rdrobust_batch(&[ds], &opts)[0];

    println!("Sharp RD estimate (true jump = 2.0), n = {n}");
    println!(
        "{:<16} {:>9} {:>9} {:>22}",
        "method", "coef", "se", "95% CI"
    );
    for (i, label) in ["Conventional", "Bias-Corrected", "Robust"]
        .iter()
        .enumerate()
    {
        println!(
            "{:<16} {:>9.4} {:>9.4}   [{:>8.4}, {:>8.4}]",
            label, out.coef[i], out.se[i], out.ci[i][0], out.ci[i][1]
        );
    }
    println!(
        "bandwidths  h = ({:.4}, {:.4})   b = ({:.4}, {:.4})",
        out.bws[0][0], out.bws[0][1], out.bws[1][0], out.bws[1][1]
    );
}
