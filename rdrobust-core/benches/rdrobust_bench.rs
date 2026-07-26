//! Benchmarks for the rdrobust numerical core.
//!
//! Run with `cargo bench`. Four groups document the performance story:
//!   - `single_latency`  -- per-call latency as n grows (nn variance, the default)
//!   - `single_vce`      -- cost of each variance estimator at fixed n
//!   - `single_design`   -- sharp vs fuzzy at fixed n
//!   - `batch_scaling`   -- serial vs rayon-parallel throughput; the multicore win over R/Python in the Monte-Carlo / many-cutoff regime
//!
//! The single-fit groups run with `masspoints="adjust"`, `bwrestrict=true`
//! (R/Python defaults). They use an equispaced DGP (`gen` below), whereas the
//! cross-language benchmark in `tools/parity.py` uses a random-uniform running
//! variable, so absolute times are not comparable across the two; only
//! within-benchmark comparisons hold. The batch group runs the library's
//! `rdrobust_batch` with `masspoints="off"` (the simulation regime where the DGP
//! is controlled), so its per-fit cost is not comparable to the single-fit groups
//! -- only its serial-vs-parallel ratio is meaningful.

use std::hint::black_box;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rdrobust_core::{
    rdrobust_batch, rdrobust_multi_cutoff, rdrobust_with, Dataset, RdRobust, RdRobustOptions, Vce,
};

/// Deterministic sorted sharp-RD data (cutoff 0), no RNG dependency: the running
/// variable is equispaced-distinct, the outcome a smooth signal plus a jump of 2.
fn gen(seed: u64, n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut x: Vec<f64> = (0..n)
        .map(|i| -1.0 + 2.0 * (i as f64 + 0.5) / (n as f64))
        .collect();
    x.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let y: Vec<f64> = x
        .iter()
        .enumerate()
        .map(|(i, &xi)| {
            let treat = if xi >= 0.0 { 1.0 } else { 0.0 };
            0.5 * xi + 2.0 * treat + ((seed as f64 + i as f64) * 0.9).sin()
        })
        .collect();
    (x, y)
}

/// Deterministic sorted fuzzy-RD data: imperfect compliance (treatment take-up
/// jumps from ~0.2 below the cutoff to ~0.8 above), outcome driven by the
/// realized treatment.
fn gen_fuzzy(seed: u64, n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let (x, _) = gen(seed, n);
    let t: Vec<f64> = x
        .iter()
        .enumerate()
        .map(|(i, &xi)| {
            // pseudo-random compliance, deterministic in (seed, i)
            let u = ((seed as f64 + 1.0) * (i as f64 + 1.0) * 12.9898)
                .sin()
                .abs();
            let p = if xi >= 0.0 { 0.8 } else { 0.2 };
            if u < p {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    let y: Vec<f64> = x
        .iter()
        .zip(&t)
        .enumerate()
        .map(|(i, (&xi, &ti))| 0.5 * xi + 1.5 * ti + ((seed as f64 + i as f64) * 0.9).sin())
        .collect();
    (x, y, t)
}

/// Single-fit call mirroring R/Python defaults (masspoints adjust, bwrestrict on).
/// The `opts` passed in must carry `masspoints: "adjust", stdvars: false,
/// bwrestrict: true` so every single-fit group measures the same workload.
fn fit(y: &[f64], x: &[f64], t: Option<&[f64]>, opts: &RdRobustOptions) -> RdRobust {
    rdrobust_with(y, x, t, None, 0, None, None, opts)
}

/// Per-call latency at n in {500, 2000, 10000, 50000}, nn variance (the default).
fn single_latency(c: &mut Criterion) {
    let opts = RdRobustOptions {
        vce: Vce::NearestNeighbor,
        masspoints: "adjust",
        stdvars: false,
        bwrestrict: true,
        ..Default::default()
    };
    let mut g = c.benchmark_group("single_latency");
    for &n in &[500usize, 2_000, 10_000, 50_000] {
        let (x, y) = gen(1, n);
        g.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| fit(black_box(&y), black_box(&x), None, &opts))
        });
    }
    g.finish();
}

/// Variance-estimator cost at n=2000: nn vs hc0/hc1/hc3.
fn single_vce(c: &mut Criterion) {
    let (x, y) = gen(1, 2_000);
    let mut g = c.benchmark_group("single_vce");
    let cases: [(&str, Vce); 4] = [
        ("nn", Vce::NearestNeighbor),
        ("hc0", Vce::Hc0),
        ("hc1", Vce::Hc1),
        ("hc3", Vce::Hc3),
    ];
    for (name, vce) in cases {
        let opts = RdRobustOptions {
            vce,
            masspoints: "adjust",
            stdvars: false,
            bwrestrict: true,
            ..Default::default()
        };
        g.bench_function(name, |b| {
            b.iter(|| fit(black_box(&y), black_box(&x), None, &opts))
        });
    }
    g.finish();
}

/// Sharp vs fuzzy at n=2000, nn variance.
fn single_design(c: &mut Criterion) {
    let opts = RdRobustOptions {
        vce: Vce::NearestNeighbor,
        masspoints: "adjust",
        stdvars: false,
        bwrestrict: true,
        ..Default::default()
    };
    let (xs, ys) = gen(1, 2_000);
    let (xf, yf, tf) = gen_fuzzy(1, 2_000);
    let mut g = c.benchmark_group("single_design");
    g.bench_function("sharp", |b| {
        b.iter(|| fit(black_box(&ys), black_box(&xs), None, &opts))
    });
    g.bench_function("fuzzy", |b| {
        b.iter(|| fit(black_box(&yf), black_box(&xf), Some(&tf), &opts))
    });
    g.finish();
}

/// Wrap (x, y) pairs as sharp `Dataset`s borrowing from `data`.
fn make_datasets(data: &[(Vec<f64>, Vec<f64>)]) -> Vec<Dataset<'_>> {
    data.iter()
        .map(|(x, y)| Dataset {
            y,
            x,
            t: None,
            z: None,
            dz: 0,
            cluster: None,
            weights: None,
        })
        .collect()
}

/// Serial vs rayon-parallel batch throughput, and many-cutoff, at n=2000.
fn batch_scaling(c: &mut Criterion) {
    // The simulation regime: nn variance, masspoints "off", no standardization or
    // bandwidth clamp. This matches the workload the old `rdrobust_batch` fixed
    // internally, so the measured numbers stay valid.
    let batch_opts = RdRobustOptions {
        vce: Vce::NearestNeighbor,
        masspoints: "off",
        stdvars: false,
        bwrestrict: false,
        ..Default::default()
    };
    let mut g = c.benchmark_group("batch_scaling");
    g.sample_size(30);

    for &k in &[16usize, 64, 256] {
        let data: Vec<(Vec<f64>, Vec<f64>)> = (0..k as u64).map(|s| gen(s, 2_000)).collect();

        // parallel (rayon) -- the shipped path
        g.bench_with_input(BenchmarkId::new("parallel", k), &k, |b, _| {
            b.iter(|| rdrobust_batch(black_box(&make_datasets(&data)), &batch_opts))
        });

        // serial reference -- same work, one core, for the speedup ratio
        g.bench_with_input(BenchmarkId::new("serial", k), &k, |b, _| {
            b.iter(|| {
                make_datasets(&data)
                    .iter()
                    .map(|d| fit_off(d.y, d.x, &batch_opts))
                    .collect::<Vec<_>>()
            })
        });
    }

    // many-cutoff: 32 cutoffs on one n=10000 dataset (geographic / multi-cutoff RD)
    let (x, y) = gen(7, 10_000);
    let cutoffs: Vec<f64> = (0..32).map(|i| -0.5 + i as f64 * (1.0 / 32.0)).collect();
    g.bench_function("multi_cutoff_32_n10000", |b| {
        b.iter(|| {
            rdrobust_multi_cutoff(
                black_box(&y),
                black_box(&x),
                None,
                None,
                0,
                None,
                None,
                &cutoffs,
                &batch_opts,
            )
        })
    });

    g.finish();
}

/// The serial counterpart of `rdrobust_batch` (masspoints "off"), for the ratio.
/// `opts` must carry the same batch workload (`masspoints: "off", stdvars: false,
/// bwrestrict: false`) as the parallel path.
fn fit_off(y: &[f64], x: &[f64], opts: &RdRobustOptions) -> RdRobust {
    rdrobust_with(y, x, None, None, 0, None, None, opts)
}

criterion_group! {
    name = benches;
    config = Criterion::default().warm_up_time(Duration::from_millis(500)).measurement_time(Duration::from_secs(3));
    targets = single_latency, single_vce, single_design, batch_scaling
}
criterion_main!(benches);
