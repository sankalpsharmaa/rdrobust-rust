//! Batched and multi-cutoff entry points.

use rayon::prelude::*;

use crate::rdrobust::{rdrobust_with, RdRobust, RdRobustOptions};

/// One dataset in a batch: sharp design when `t` is `None`, fuzzy otherwise.
///
/// A `Dataset` carries everything a single [`crate::rdrobust::rdrobust_with`] call
/// needs from the data side (the per-fit knobs live in the shared
/// [`RdRobustOptions`]), so a batch fit is a true repeated `rdrobust`.
pub struct Dataset<'a> {
    /// Outcome variable, one value per observation.
    pub y: &'a [f64],
    /// Running (forcing) variable. For nearest-neighbor `vce`, must be sorted ascending.
    pub x: &'a [f64],
    /// Treatment-status variable for a fuzzy design; `None` runs a sharp design.
    pub t: Option<&'a [f64]>,
    /// Covariates flattened row-major as `n x dz`; `None` runs without covariate adjustment.
    pub z: Option<&'a [f64]>,
    /// Number of covariate columns in `z` (0 when `z` is `None`).
    pub dz: usize,
    /// Cluster ids; presence remaps the variance estimator to its cluster-robust form. `None` for no clustering.
    pub cluster: Option<&'a [f64]>,
    /// User weights multiplying the kernel; `None` for an unweighted fit.
    pub weights: Option<&'a [f64]>,
}

/// Runs [`crate::rdrobust::rdrobust_with`] on many datasets in parallel.
///
/// Every dataset shares the same `opts` (including `opts.bwselect`), so each fit is
/// a repeated `rdrobust` differing only in its data. The results preserve the input
/// order of `datasets`.
pub fn rdrobust_batch(datasets: &[Dataset], opts: &RdRobustOptions) -> Vec<RdRobust> {
    datasets
        .par_iter()
        .map(|d| rdrobust_with(d.y, d.x, d.t, d.z, d.dz, d.cluster, d.weights, opts))
        .collect()
}

/// Runs [`crate::rdrobust::rdrobust_with`] at many cutoffs on one dataset.
///
/// Each fit copies `opts` and overrides `opts.c` with the cutoff for that fit; every
/// other knob (and the data) is shared. The results preserve the order of `cutoffs`.
#[allow(clippy::too_many_arguments)]
pub fn rdrobust_multi_cutoff(
    y: &[f64],
    x: &[f64],
    fuzzy: Option<&[f64]>,
    covs: Option<&[f64]>,
    dz: usize,
    cluster: Option<&[f64]>,
    weights: Option<&[f64]>,
    cutoffs: &[f64],
    opts: &RdRobustOptions,
) -> Vec<RdRobust> {
    cutoffs
        .par_iter()
        .map(|&c| {
            let mut o = *opts;
            o.c = c;
            rdrobust_with(y, x, fuzzy, covs, dz, cluster, weights, &o)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Vce;

    fn dgp(seed: u64, n: usize) -> (Vec<f64>, Vec<f64>) {
        let mut x: Vec<f64> = (0..n)
            .map(|i| -1.0 + 2.0 * (i as f64) / (n as f64))
            .collect();
        x.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let y: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, &xi)| {
                let treat = if xi >= 0.0 { 1.0 } else { 0.0 };
                0.5 * xi + 2.0 * treat + ((seed as f64 + i as f64) * 0.7).sin()
            })
            .collect();
        (x, y)
    }

    #[test]
    fn batch_equals_serial() {
        let data: Vec<(Vec<f64>, Vec<f64>)> = (0..6).map(|s| dgp(s, 400)).collect();
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
        let opts = RdRobustOptions {
            vce: Vce::Hc0,
            ..Default::default()
        };

        let parallel = rdrobust_batch(&datasets, &opts);
        let serial: Vec<RdRobust> = datasets
            .iter()
            .map(|d| rdrobust_with(d.y, d.x, d.t, d.z, d.dz, d.cluster, d.weights, &opts))
            .collect();

        assert_eq!(parallel.len(), serial.len());
        for (a, b) in parallel.iter().zip(serial.iter()) {
            assert_eq!(a.coef, b.coef);
            assert_eq!(a.se, b.se);
        }
    }
}
