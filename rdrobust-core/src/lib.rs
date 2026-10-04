//! Pure-Rust numerical core for robust local-polynomial regression-discontinuity
//! (RD) estimation: a faithful port of the Calonico-Cattaneo-Farrell-Titiunik
//! `rdrobust` package (R source `R/rdrobust/R/{functions,rdrobust,rdbwselect,rdplot}.R`).
//!
//! # What it computes
//!
//! - [`rdrobust()`]: the sharp or fuzzy RD point estimate, with conventional and
//!   bias-corrected coefficients, CCT robust variance, z-statistics, p-values, and
//!   confidence intervals.
//! - [`rdbwselect`]: MSE- and CER-optimal bandwidth selection (all ten selectors).
//! - [`rdplot()`]: the binned-scatter quantities and the global polynomial curve.
//! - [`rdrobust_batch`] and [`rdrobust_multi_cutoff`]: rayon-parallel fan-out over
//!   independent fits for Monte-Carlo and many-cutoff workloads.
//!
//! Outputs match the R reference to ~1e-8 on continuous quantities and ~1e-11 on
//! clustered variances, and exactly on discrete decisions (neighbor sets,
//! mass-point grouping, bin counts).
//!
//! # Example
//!
//! ```
//! use rdrobust_core::{rdrobust_batch, Dataset, RdRobustOptions};
//!
//! // A small sharp RD: the outcome jumps by about 2 at the cutoff c = 0.
//! let x: Vec<f64> = (0..200).map(|i| -1.0 + 2.0 * (i as f64 + 0.5) / 200.0).collect();
//! let y: Vec<f64> = x
//!     .iter()
//!     .enumerate()
//!     .map(|(i, &xi)| 0.5 * xi + if xi >= 0.0 { 2.0 } else { 0.0 } + 0.3 * (i as f64).sin())
//!     .collect();
//!
//! let opts = RdRobustOptions::default(); // R defaults: p=1, q=2, triangular kernel, nn variance
//! let ds = Dataset { y: &y, x: &x, t: None, z: None, dz: 0, cluster: None, weights: None };
//! let out = rdrobust_batch(&[ds], &opts);
//!
//! // Each result's rows are [Conventional, Bias-Corrected, Robust].
//! assert_eq!(out.len(), 1);
//! assert!(out[0].coef[0].is_finite());
//! assert!(out[0].se[2] >= 0.0);
//! ```
//!
//! # Scope and limitations
//!
//! The main output set (`coef`/`se`/`z`/`pv`/`ci`/`bws`) is complete. Per-side
//! decompositions, fuzzy first-stage diagnostics, and the `rdplot` QS / covariate /
//! CI extensions are not ported. The raw [`rdrobust`]/[`rdrobust_with`] entry points
//! still return `NaN` on degenerate inputs (a cutoff outside the support, a fuzzy
//! treatment with no first stage); [`try_rdrobust_with`] adds a typed validation
//! channel ([`RdError`]) that rejects those inputs up front. See `CHANGELOG.md`
//! for the full list.

#![deny(missing_docs)]
#![forbid(unsafe_code)]

pub(crate) mod batch;
pub(crate) mod bw;
pub(crate) mod bwselect;
pub(crate) mod kernel;
pub(crate) mod linalg;
pub(crate) mod rdplot;
pub(crate) mod rdrobust;
mod stats;
mod vce;

pub use batch::{rdrobust_batch, rdrobust_multi_cutoff, Dataset};
pub use bw::{rdrobust_bw, BwParts};
pub use bwselect::{rdbwselect, Bws};
pub use kernel::kweight;
pub use rdplot::{rdplot, RdPlot};
pub use rdrobust::{
    rdrobust, rdrobust_with, try_rdrobust_with, RdError, RdRobust, RdRobustOptions,
};

/// Local-polynomial weighting kernel (mirrors `rdrobust_kweight`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kernel {
    /// `(1 - |u|) * 1{|u| <= 1}` -- the rdrobust default.
    Triangular,
    /// `0.75 * (1 - u^2) * 1{|u| <= 1}`.
    Epanechnikov,
    /// `0.5 * 1{|u| <= 1}`.
    Uniform,
}

/// Variance-covariance estimator (mirrors the `vce` argument).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Vce {
    /// Nearest-neighbor (default), `nnmatch` neighbors per side.
    NearestNeighbor,
    /// Heteroskedasticity-consistent HC0.
    Hc0,
    /// Heteroskedasticity-consistent HC1.
    Hc1,
    /// Heteroskedasticity-consistent HC2.
    Hc2,
    /// Heteroskedasticity-consistent HC3.
    Hc3,
    /// Cluster-robust CRV1.
    Cr1,
    /// Cluster-robust CRV2.
    Cr2,
    /// Cluster-robust CRV3.
    Cr3,
}

impl Vce {
    /// HC order (`Hc0`->0 .. `Hc3`->3), `None` for non-HC variants.
    pub(crate) fn hc_order(self) -> Option<u8> {
        match self {
            Vce::Hc0 => Some(0),
            Vce::Hc1 => Some(1),
            Vce::Hc2 => Some(2),
            Vce::Hc3 => Some(3),
            _ => None,
        }
    }

    /// Cluster order (`Cr1`->1 .. `Cr3`->3), `None` for non-cluster variants.
    pub(crate) fn cluster_order(self) -> Option<u8> {
        match self {
            Vce::Cr1 => Some(1),
            Vce::Cr2 => Some(2),
            Vce::Cr3 => Some(3),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_variants_are_distinct() {
        assert_ne!(Kernel::Triangular, Kernel::Uniform);
        assert_ne!(Kernel::Epanechnikov, Kernel::Uniform);
    }

    #[test]
    fn vce_carries_order() {
        assert_eq!(Vce::Hc2, Vce::Hc2);
        assert_ne!(Vce::Hc2, Vce::Hc3);
        assert_eq!(Vce::Hc2.hc_order(), Some(2));
        assert_eq!(Vce::Cr3.cluster_order(), Some(3));
        assert_eq!(Vce::NearestNeighbor.hc_order(), None);
    }
}
