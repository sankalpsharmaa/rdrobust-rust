//! Parallel Monte-Carlo: fit many independent RD datasets at once. This is the
//! regime where the compiled core plus rayon fan-out beats interpreted R/Python by
//! a wide margin (see `BENCHMARKS.md`).
//!
//! Run with: `cargo run --release --example monte_carlo`

use std::time::Instant;

use rdrobust_core::{rdrobust_batch, Dataset, RdRobustOptions};

/// One deterministic sharp-RD replication (true jump = 2.0), seeded by `seed`.
fn gen(seed: u64, n: usize) -> (Vec<f64>, Vec<f64>) {
    let x: Vec<f64> = (0..n)
        .map(|i| -1.0 + 2.0 * (i as f64 + 0.5) / n as f64)
        .collect();
    let y: Vec<f64> = x
        .iter()
        .enumerate()
        .map(|(i, &xi)| {
            0.5 * xi + if xi >= 0.0 { 2.0 } else { 0.0 } + ((seed as f64 + i as f64) * 0.9).sin()
        })
        .collect();
    (x, y)
}

fn main() {
    let reps = 256;
    let n = 2000;

    let data: Vec<(Vec<f64>, Vec<f64>)> = (0..reps as u64).map(|s| gen(s, n)).collect();
    let datasets: Vec<Dataset> = data
        .iter()
        .map(|(x, y)| Dataset {
            y,
            x,
            t: None,
            z: None,
            dz: 0,
            cluster: None,
            weights: None,
        })
        .collect();
    let opts = RdRobustOptions::default();

    let t0 = Instant::now();
    let results = rdrobust_batch(&datasets, &opts);
    let dt = t0.elapsed().as_secs_f64() * 1e3;

    let mean = results.iter().map(|r| r.coef[0]).sum::<f64>() / reps as f64;
    println!(
        "{reps} fits of n = {n} in {dt:.1} ms ({:.3} ms/fit, parallel across cores)",
        dt / reps as f64
    );
    println!("mean conventional coefficient = {mean:.4}  (true jump = 2.0)");
}
