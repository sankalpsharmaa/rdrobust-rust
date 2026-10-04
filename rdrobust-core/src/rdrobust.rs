//! Main RD estimator, a port of `rdrobust` (R rdrobust.R:540-907): the
//! point estimate (conventional + bias-corrected) with CCT robust variance and
//! confidence intervals.
//!
//! Implemented: sharp/fuzzy, covariates, clusters (CR1/CRV2/CRV3), nn/hc0-hc3, user
//! weights, `rho`, `perf_comp`/`sharpbw`, the prep options (`masspoints`,
//! `stdvars`, `bwrestrict`, `bwcheck`), and manual/auto bandwidths.
//!
//! Parity gate: coef/se/ci to ~1e-8 vs R (~1e-11 for clusters).

use nalgebra::{DMatrix, DVector};

use crate::bwselect::rdbwselect;
use crate::kernel::eff_weight;
use crate::linalg::{qr_xx_inv, vander};
use crate::stats::{pnorm, qnorm};
use crate::vce;
use crate::{Kernel, Vce};

/// Headline `rdrobust` results, one row per inference method (rows are
/// Conventional, Bias-Corrected, Robust; ports rdrobust.R:854-863).
///
/// The three rows are not three independent estimates. The point estimate
/// repeats (`coef = [tau_cl, tau_bc, tau_bc]`: Bias-Corrected and Robust share
/// `tau_bc`) and the standard error repeats (`se = [se_cl, se_cl, se_rb]`:
/// Conventional and Bias-Corrected share `se_cl`). `z`, `pv`, and `ci` follow
/// row-wise from `coef` and `se`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RdRobust {
    /// Point estimates by row: `[tau_cl, tau_bc, tau_bc]` (Conventional uses the
    /// p-fit estimate; Bias-Corrected and Robust both use the bias-corrected one).
    pub coef: [f64; 3],
    /// Standard errors by row: `[se_cl, se_cl, se_rb]` (Conventional and
    /// Bias-Corrected share the conventional SE; Robust uses the CCT robust SE).
    pub se: [f64; 3],
    /// z-statistics by row, `coef[i] / se[i]`.
    pub z: [f64; 3],
    /// Two-sided normal p-values by row, `2 * Phi(-|z[i]|)`.
    pub pv: [f64; 3],
    /// Confidence intervals by row, each `[lower, upper]` at `level`% using the
    /// matching row's `coef` and `se`.
    pub ci: [[f64; 2]; 3],
    /// Bandwidths used: rows are `h` (point estimate) then `b` (bias correction),
    /// columns are left then right, i.e. `[[h_l, h_r], [b_l, b_r]]`.
    pub bws: [[f64; 2]; 2],
}

/// Per-side fit pieces at the common window (the larger of the h- and b-windows).
struct SideFit {
    r_p: DMatrix<f64>, // Vandermonde order p, n x (p+1)
    r_q: DMatrix<f64>, // Vandermonde order q, n x (q+1)
    inv_g_p: DMatrix<f64>,
    inv_g_q: DMatrix<f64>,
    rp_wh: DMatrix<f64>,   // R_p * W_h (row-scaled) -- conventional meat design
    rp_swh: DMatrix<f64>,  // R_p * sqrt(W_h) -- crv2/crv3 conventional sqrtRX
    rq_wb: DMatrix<f64>,   // R_q * W_b (row-scaled) -- robust hc residual design
    rq_swb: DMatrix<f64>,  // R_q * sqrt(W_b) -- crv2/crv3 h==b robust sqrtRX
    w_b: Vec<f64>,         // in-window W_b kernel weights (for qq_cluster)
    q_q: DMatrix<f64>,     // bias-correction design, n x (p+1)
    d: DMatrix<f64>,       // response design [Y] or [Y, T]
    beta_p: DMatrix<f64>,  // (p+1) x ncols
    beta_q: DMatrix<f64>,  // (q+1) x ncols
    beta_bc: DMatrix<f64>, // (p+1) x ncols
    zwz_p: Option<DMatrix<f64>>, // covariate FWL inputs (per side; combined in main)
    zwy_p: Option<DMatrix<f64>>,
    ex: Vec<f64>,
    ec: Vec<f64>, // windowed cluster ids (empty if no clustering)
    en: usize,
    edups: Vec<usize>,
    edupsid: Vec<usize>,
}

/// Fit one side at bandwidths (h, b): build the common window, the p/q fits, the
/// bias-correction design Q_q, and the three coefficient matrices.
#[allow(clippy::too_many_arguments)]
fn fit_side(
    y: &[f64],
    x: &[f64],
    t: Option<&[f64]>,
    z: Option<&[f64]>,
    cl: Option<&[f64]>,
    wt: Option<&[f64]>,
    dz: usize,
    c: f64,
    p: usize,
    q: usize,
    h: f64,
    b: f64,
    kernel: Kernel,
    vce_kind: Vce,
) -> SideFit {
    let w_h_full = eff_weight(x, c, h, kernel, wt);
    let w_b_full = eff_weight(x, c, b, kernel, wt);
    // common window: the larger bandwidth's support (R:546-548)
    let ind: Vec<usize> = if h > b {
        (0..x.len()).filter(|&i| w_h_full[i] > 0.0).collect()
    } else {
        (0..x.len()).filter(|&i| w_b_full[i] > 0.0).collect()
    };
    let en = ind.len();
    let ex: Vec<f64> = ind.iter().map(|&i| x[i]).collect();
    let ec: Vec<f64> = cl.map_or(Vec::new(), |c| ind.iter().map(|&i| c[i]).collect());
    let w_h: Vec<f64> = ind.iter().map(|&i| w_h_full[i]).collect();
    let w_b: Vec<f64> = ind.iter().map(|&i| w_b_full[i]).collect();

    let (edups, edupsid) = match vce_kind {
        Vce::NearestNeighbor => {
            let (du, ds) = vce::compute_dups_dupsid(x); // full side (sorted)
            (
                ind.iter().map(|&i| du[i]).collect(),
                ind.iter().map(|&i| ds[i]).collect(),
            )
        }
        _ => (Vec::new(), Vec::new()),
    };

    let u: Vec<f64> = ex.iter().map(|&xi| xi - c).collect();
    let r_q = vander(&u, q);
    let r_p = r_q.columns(0, p + 1).into_owned();

    // row-scaled designs
    let row_scale = |m: &DMatrix<f64>, w: &[f64]| {
        let mut out = m.clone();
        for i in 0..en {
            for j in 0..m.ncols() {
                out[(i, j)] *= w[i];
            }
        }
        out
    };
    let sqrt_w = |w: &[f64]| w.iter().map(|v| v.sqrt()).collect::<Vec<_>>();
    let rp_swh = row_scale(&r_p, &sqrt_w(&w_h));
    let rq_swb = row_scale(&r_q, &sqrt_w(&w_b));
    let inv_g_p = qr_xx_inv(&rp_swh);
    let inv_g_q = qr_xx_inv(&rq_swb);
    let rp_wh = row_scale(&r_p, &w_h);
    let rq_wb = row_scale(&r_q, &w_b);

    // L = (R_p*W_h)' u^(p+1) ; vcorr = W_b * col_{p+1}(R_q invG_q) ;
    // Q_q[k,i] = (R_p*W_h)[k,i] - h^(p+1) * vcorr[k] * L[i]   (R:572-577)
    let u_pow = DVector::from_iterator(en, u.iter().map(|&uu| (uu / h).powi((p + 1) as i32)));
    let l_vec = rp_wh.transpose() * &u_pow; // (p+1)
    let rq_ig = &r_q * &inv_g_q; // n x (q+1)
    let hp1 = h.powi((p + 1) as i32);
    let mut q_q = rp_wh.clone();
    for k in 0..en {
        let vcorr = w_b[k] * rq_ig[(k, p + 1)];
        for i in 0..=p {
            q_q[(k, i)] -= hp1 * vcorr * l_vec[i];
        }
    }

    // response design [Y, T?, Z..] and the three fits
    let dt = t.is_some() as usize;
    let mut d = DMatrix::zeros(en, 1 + dt + dz);
    for (r, &i) in ind.iter().enumerate() {
        d[(r, 0)] = y[i];
    }
    if let Some(tt) = t {
        for (r, &i) in ind.iter().enumerate() {
            d[(r, 1)] = tt[i];
        }
    }
    if let Some(zz) = z {
        for (r, &i) in ind.iter().enumerate() {
            for j in 0..dz {
                d[(r, 1 + dt + j)] = zz[i * dz + j];
            }
        }
    }
    let beta_p = &inv_g_p * (rp_wh.transpose() * &d);
    let beta_q = &inv_g_q * (rq_wb.transpose() * &d);
    let beta_bc = &inv_g_p * (q_q.transpose() * &d);

    // per-side covariate partialling inputs (combined across sides in `rdrobust`)
    let (zwz_p, zwy_p) = if dz > 0 {
        let (zwz, zwy) = vce::partialling(&rp_wh, &inv_g_p, &d, &w_h, dt, dz);
        (Some(zwz), Some(zwy))
    } else {
        (None, None)
    };

    SideFit {
        r_p,
        r_q,
        inv_g_p,
        inv_g_q,
        rp_wh,
        rp_swh,
        rq_wb,
        rq_swb,
        w_b,
        q_q,
        d,
        beta_p,
        beta_q,
        beta_bc,
        zwz_p,
        zwy_p,
        ex,
        ec,
        en,
        edups,
        edupsid,
    }
}

/// Scalar/config options for [`rdrobust_with`], a typed alternative to the raw
/// [`rdrobust`] positional arguments. The data inputs (`y`, `x`, `fuzzy`, `covs`,
/// `dz`, `cluster`, `weights`) stay separate; this struct carries everything else.
///
/// [`Default`] matches the R/Python wrapper defaults, so `RdRobustOptions {
/// p: 2, q: 3, ..Default::default() }` overrides one knob and leaves the rest at
/// their R defaults. Each field forwards to the matching positional argument of
/// [`rdrobust`]; see that function for the per-argument semantics.
#[derive(Debug, Clone, Copy)]
pub struct RdRobustOptions {
    /// RD cutoff on the running variable (R `c`, default 0).
    pub c: f64,
    /// Polynomial order for the point estimate (R `p`, default 1).
    pub p: usize,
    /// Polynomial order for the bias correction; must exceed `p` (R `q`, default 2).
    pub q: usize,
    /// Derivative order of the estimand: 0 = level, 1 = kink (R `deriv`, default 0).
    pub deriv: usize,
    /// Kernel for the local-polynomial weights (R `kernel`, default triangular).
    pub kernel: Kernel,
    /// Variance estimator before any cluster remap (R `vce`, default nn).
    pub vce: Vce,
    /// Minimum neighbors for `vce = nn` (R `nnmatch`, default 3).
    pub nnmatch: usize,
    /// Drop collinear covariates in the FWL partialling (R `covs_drop`, default false here).
    pub covs_drop_coll: bool,
    /// Regularization scale in bandwidth selection; 0 removes it (R `scaleregul`, default 1).
    pub scaleregul: f64,
    /// Scaling applied to the RD parameter (R `scalepar`, default 1).
    pub scalepar: f64,
    /// Confidence-interval level in percent, e.g. 95 (R `level`, default 95).
    pub level: f64,
    /// Bandwidth-selection method when `h` is `None`, e.g. `"mserd"` (R `bwselect`, default "mserd").
    pub bwselect: &'static str,
    /// Mass-point handling: `"off"`, `"check"`, or `"adjust"` (R `masspoints`, default "adjust").
    pub masspoints: &'static str,
    /// Standardize `x` and `y` before bandwidth selection (R `stdvars`, default true).
    pub stdvars: bool,
    /// Clamp selected bandwidths to the range of `x` (R `bwrestrict`, default true).
    pub bwrestrict: bool,
    /// Minimum unique observations in the pilot bandwidth; `None` skips the check (R `bwcheck`, default NULL).
    pub bwcheck: Option<usize>,
    /// Sets `b = h / rho`; `None` leaves `b` from `b`/selection (R `rho`, default NULL).
    pub rho: Option<f64>,
    /// Force sharp-RD bandwidth selection in a fuzzy design (R `sharpbw`, default false).
    pub sharpbw: bool,
    /// Main bandwidths `(left, right)`; `None` triggers selection (R `h`, default NULL).
    pub h: Option<(f64, f64)>,
    /// Bias bandwidths `(left, right)`; `None` defaults to `h` (or `h / rho`) (R `b`, default NULL).
    pub b: Option<(f64, f64)>,
}

impl Default for RdRobustOptions {
    /// R/Python wrapper defaults: `c=0, p=1, q=2, deriv=0`, triangular kernel, nn
    /// variance, `nnmatch=3`, `scaleregul=1`, `scalepar=1`, `level=95`,
    /// `bwselect="mserd"`, `masspoints="adjust"`, `stdvars`/`bwrestrict` on, and the
    /// remaining knobs off/unset.
    fn default() -> Self {
        Self {
            c: 0.0,
            p: 1,
            q: 2,
            deriv: 0,
            kernel: Kernel::Triangular,
            vce: Vce::NearestNeighbor,
            nnmatch: 3,
            covs_drop_coll: false,
            scaleregul: 1.0,
            scalepar: 1.0,
            level: 95.0,
            bwselect: "mserd",
            masspoints: "adjust",
            stdvars: true,
            bwrestrict: true,
            bwcheck: None,
            rho: None,
            sharpbw: false,
            h: None,
            b: None,
        }
    }
}

/// Ergonomic, options-struct entry point to [`rdrobust`]: the seven data inputs
/// stay explicit while every scalar/config knob lives in [`RdRobustOptions`], so a
/// caller no longer threads 27 positional arguments (where swapping two
/// bools/strings silently changes the estimate).
///
/// This forwards verbatim to the raw [`rdrobust`]; it changes ergonomics, not
/// numerics. `fuzzy` is the treatment status (`None` = sharp), `covs` the
/// flattened row-major `n x dz` covariate slice (`None` = no covariates, `dz = 0`).
///
/// # Example
///
/// ```
/// use rdrobust_core::{rdrobust_with, RdRobustOptions};
///
/// // A small sharp RD with a jump of about 2 at the cutoff c = 0; x sorted ascending.
/// let x: Vec<f64> = (0..200).map(|i| -1.0 + 2.0 * (i as f64 + 0.5) / 200.0).collect();
/// let y: Vec<f64> = x
///     .iter()
///     .enumerate()
///     .map(|(i, &xi)| 0.5 * xi + if xi >= 0.0 { 2.0 } else { 0.0 } + 0.3 * (i as f64).sin())
///     .collect();
///
/// // Override just the polynomial orders (q must exceed p); keep every other R default.
/// let opts = RdRobustOptions { p: 2, q: 3, ..Default::default() };
/// let out = rdrobust_with(&y, &x, None, None, 0, None, None, &opts);
///
/// assert!(out.coef[0].is_finite());
/// ```
#[allow(clippy::too_many_arguments)]
pub fn rdrobust_with(
    y: &[f64],
    x: &[f64],
    fuzzy: Option<&[f64]>,
    covs: Option<&[f64]>,
    dz: usize,
    cluster: Option<&[f64]>,
    weights: Option<&[f64]>,
    opts: &RdRobustOptions,
) -> RdRobust {
    rdrobust(
        y,
        x,
        fuzzy,
        covs,
        dz,
        opts.c,
        opts.p,
        opts.q,
        opts.deriv,
        opts.kernel,
        opts.vce,
        opts.nnmatch,
        opts.covs_drop_coll,
        opts.scaleregul,
        opts.scalepar,
        opts.level,
        opts.bwselect,
        cluster,
        weights,
        opts.masspoints,
        opts.stdvars,
        opts.bwrestrict,
        opts.bwcheck,
        opts.rho,
        opts.sharpbw,
        opts.h,
        opts.b,
    )
}

/// Validation failure for [`try_rdrobust_with`]: a degenerate input or option that
/// the raw [`rdrobust`]/[`rdrobust_with`] path would otherwise carry into `NaN`
/// results. Each variant mirrors a check the Python binding already performs, with
/// the same message text, so the two channels agree on what they reject.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RdError {
    /// Fewer than two observations in `y`/`x`.
    TooFewObs,
    /// A length-`n` input has a length other than `expected` (the length of `y`).
    LengthMismatch {
        /// The offending input's name (`"x"`, `"fuzzy"`, `"cluster"`, `"weights"`, `"covs"`).
        name: &'static str,
        /// The actual length supplied.
        len: usize,
        /// The expected length (the length of `y`).
        expected: usize,
    },
    /// An input contains a non-finite value (NaN or infinity).
    NonFinite {
        /// The offending input's name (`"y"`, `"x"`, `"fuzzy"`, `"cluster"`, `"weights"`, `"covs"`).
        name: &'static str,
    },
    /// The cutoff `c` is not strictly inside the support of `x` (`min(x) < c < max(x)`).
    OneSidedCutoff,
    /// Invalid polynomial/derivative orders: `q <= p` or `deriv > p`.
    InvalidOrder,
    /// `nnmatch` is less than 1.
    InvalidNnmatch,
    /// `level` is outside the open interval `(0, 100)`.
    InvalidLevel,
    /// A bandwidth-like scalar (`h`, `b`, or `rho`) is not strictly positive.
    NonPositiveBandwidth {
        /// The offending scalar's name (`"h"`, `"b"`, or `"rho"`).
        name: &'static str,
    },
    /// A weight is negative.
    NonNegativeWeightRequired,
    /// Covariates were supplied (`covs` is `Some`) but with zero columns (`dz == 0`).
    EmptyCovariates,
    /// The fuzzy first stage is constant on both sides of the cutoff with no jump,
    /// so the treatment effect is not identified (R stops with the same diagnosis).
    DegenerateFuzzy,
}

impl std::fmt::Display for RdError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RdError::TooFewObs => {
                write!(f, "y and x must have at least two observations")
            }
            RdError::LengthMismatch {
                name,
                len,
                expected,
            } => write!(f, "{name} has length {len}, expected {expected}"),
            RdError::NonFinite { name } => write!(f, "{name} must be finite"),
            RdError::OneSidedCutoff => {
                write!(f, "c must lie strictly inside the range of x")
            }
            RdError::InvalidOrder => write!(f, "q must be > p and deriv must be <= p"),
            RdError::InvalidNnmatch => write!(f, "nnmatch must be >= 1"),
            RdError::InvalidLevel => write!(f, "level must be in (0, 100)"),
            RdError::NonPositiveBandwidth { name } => write!(f, "{name} must be positive"),
            RdError::NonNegativeWeightRequired => write!(f, "weights must be non-negative"),
            RdError::EmptyCovariates => write!(f, "covs must have at least one column"),
            RdError::DegenerateFuzzy => write!(
                f,
                "fuzzy has no variation and no jump at the cutoff; the fuzzy estimator is not identified"
            ),
        }
    }
}

impl std::error::Error for RdError {}

/// Validate the [`try_rdrobust_with`] inputs and options without running the fit.
/// Returns `Ok(())` when every check the Python binding enforces passes, otherwise
/// the first failing [`RdError`]. `dz` is the row stride of `covs` (0 when `covs`
/// is `None`); `n` is taken from `y`.
#[allow(clippy::too_many_arguments)]
fn validate_rdrobust_inputs(
    y: &[f64],
    x: &[f64],
    fuzzy: Option<&[f64]>,
    covs: Option<&[f64]>,
    dz: usize,
    cluster: Option<&[f64]>,
    weights: Option<&[f64]>,
    opts: &RdRobustOptions,
) -> Result<(), RdError> {
    let n = y.len();
    if n < 2 {
        return Err(RdError::TooFewObs);
    }
    let ensure_len = |name: &'static str, len: usize| -> Result<(), RdError> {
        if len == n {
            Ok(())
        } else {
            Err(RdError::LengthMismatch {
                name,
                len,
                expected: n,
            })
        }
    };
    let ensure_finite = |name: &'static str, values: &[f64]| -> Result<(), RdError> {
        if values.iter().all(|v| v.is_finite()) {
            Ok(())
        } else {
            Err(RdError::NonFinite { name })
        }
    };

    ensure_len("x", x.len())?;
    ensure_finite("y", y)?;
    ensure_finite("x", x)?;
    if let Some(values) = fuzzy {
        ensure_len("fuzzy", values.len())?;
        ensure_finite("fuzzy", values)?;
    }
    if let Some(values) = cluster {
        ensure_len("cluster", values.len())?;
        ensure_finite("cluster", values)?;
    }
    if let Some(values) = weights {
        ensure_len("weights", values.len())?;
        ensure_finite("weights", values)?;
        if values.iter().any(|&v| v < 0.0) {
            return Err(RdError::NonNegativeWeightRequired);
        }
    }
    if let Some(values) = covs {
        if dz == 0 {
            return Err(RdError::EmptyCovariates);
        }
        // Exact: `values.len() / dz` would silently accept a ragged slice.
        if values.len() != n * dz {
            return Err(RdError::LengthMismatch {
                name: "covs",
                len: values.len() / dz,
                expected: n,
            });
        }
        ensure_finite("covs", values)?;
    }

    if opts.q <= opts.p || opts.deriv > opts.p {
        return Err(RdError::InvalidOrder);
    }
    if opts.nnmatch == 0 {
        return Err(RdError::InvalidNnmatch);
    }
    if !(0.0..100.0).contains(&opts.level) {
        return Err(RdError::InvalidLevel);
    }
    if let Some(r) = opts.rho {
        if r <= 0.0 {
            return Err(RdError::NonPositiveBandwidth { name: "rho" });
        }
    }
    if let Some((hl, hr)) = opts.h {
        if hl <= 0.0 || hr <= 0.0 {
            return Err(RdError::NonPositiveBandwidth { name: "h" });
        }
    }
    if let Some((bl, br)) = opts.b {
        if bl <= 0.0 || br <= 0.0 {
            return Err(RdError::NonPositiveBandwidth { name: "b" });
        }
    }

    // R rejects `c <= min(x) | c >= max(x)` (rdrobust.R:244): the cutoff must lie
    // strictly inside the support so both sides have a window of positive width.
    // A cutoff at `max(x)` passes a plain "both sides non-empty" test but leaves
    // the right side with zero range and every pilot fit empty (NaN output).
    let x_min = x.iter().cloned().fold(f64::INFINITY, f64::min);
    let x_max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if !(opts.c > x_min && opts.c < x_max) {
        return Err(RdError::OneSidedCutoff);
    }

    // R stops when the first stage has no variation on either side and no jump
    // (rdrobust.R:175-177): the fuzzy estimator divides by a zero first stage.
    if let Some(tt) = fuzzy {
        let side = |on_left: bool| -> Vec<f64> {
            (0..n)
                .filter(|&i| (x[i] < opts.c) == on_left)
                .map(|i| tt[i])
                .collect()
        };
        let (tl, tr) = (side(true), side(false));
        let constant = |v: &[f64]| v.iter().all(|&u| u == v[0]);
        if constant(&tl) && constant(&tr) && (tl[0] - tr[0]).abs() < f64::EPSILON.sqrt() {
            return Err(RdError::DegenerateFuzzy);
        }
    }

    Ok(())
}

/// Validated wrapper over [`rdrobust_with`]: run `validate_rdrobust_inputs` first
/// and, when every check passes, forward verbatim to [`rdrobust_with`] and return
/// `Ok(result)`. On any degenerate input or option it returns the first matching
/// [`RdError`] instead of letting the raw path produce `NaN`. The numerics on valid
/// inputs are identical to [`rdrobust_with`]; only the rejection of bad inputs differs.
///
/// `fuzzy`, `covs`, `dz`, `cluster`, and `weights` carry the same meaning as in
/// [`rdrobust_with`]. `opts` holds every scalar/config knob via [`RdRobustOptions`].
///
/// # Example
///
/// ```
/// use rdrobust_core::{try_rdrobust_with, RdError, RdRobustOptions};
///
/// // A small sharp RD with a jump of about 2 at the cutoff c = 0; x sorted ascending.
/// let x: Vec<f64> = (0..200).map(|i| -1.0 + 2.0 * (i as f64 + 0.5) / 200.0).collect();
/// let y: Vec<f64> = x
///     .iter()
///     .enumerate()
///     .map(|(i, &xi)| 0.5 * xi + if xi >= 0.0 { 2.0 } else { 0.0 } + 0.3 * (i as f64).sin())
///     .collect();
///
/// // Valid options (defaults have q = 2 > p = 1): an Ok result.
/// let ok = RdRobustOptions::default();
/// let out = try_rdrobust_with(&y, &x, None, None, 0, None, None, &ok).unwrap();
/// assert!(out.coef[0].is_finite());
///
/// // q <= p (here q = p = 1) is rejected before any fit runs.
/// let bad = RdRobustOptions { q: 1, ..Default::default() };
/// let err = try_rdrobust_with(&y, &x, None, None, 0, None, None, &bad);
/// assert_eq!(err, Err(RdError::InvalidOrder));
/// ```
#[allow(clippy::too_many_arguments)]
pub fn try_rdrobust_with(
    y: &[f64],
    x: &[f64],
    fuzzy: Option<&[f64]>,
    covs: Option<&[f64]>,
    dz: usize,
    cluster: Option<&[f64]>,
    weights: Option<&[f64]>,
    opts: &RdRobustOptions,
) -> Result<RdRobust, RdError> {
    validate_rdrobust_inputs(y, x, fuzzy, covs, dz, cluster, weights, opts)?;
    Ok(rdrobust_with(y, x, fuzzy, covs, dz, cluster, weights, opts))
}

/// Port of `rdrobust` (rdrobust.R:540-907) for the scoped option set: the RD
/// point estimate (conventional and bias-corrected) with CCT robust variance and
/// confidence intervals.
///
/// `h`/`b` as `Some((left, right))` supplies bandwidths directly; `None` selects
/// them via [`rdbwselect`]. For `vce = nn`, `x` must be sorted ascending.
///
/// With fewer than 20 observations R skips bandwidth selection and uses the full
/// support, `h = b = max(c - min(x), max(x) - c)`, on both sides (overriding any
/// `h`, `b`, or `rho`); this port does the same.
///
/// The core takes every argument as a required positional; the "R-default" column
/// below is the value the R/Python wrapper supplies when the user omits the arg.
///
/// # Arguments
///
/// * `y` - Outcome values, one per observation.
/// * `x` - Running variable, one per observation (sorted ascending when `vce = nn`).
/// * `t` - Fuzzy treatment status; `None` for a sharp design (R `fuzzy`, default NULL).
/// * `z` - Covariates as a flattened row-major `n x dz` slice; `None` for no covariates (R `covs`, default NULL).
/// * `dz` - Covariate column count (the row stride of `z`); 0 when `z` is `None`. Rust-core detail, no R default.
/// * `c` - RD cutoff (R `c`, default 0).
/// * `p` - Polynomial order for the point estimate (R `p`, default 1, or `deriv + 1`).
/// * `q` - Polynomial order for the bias correction (R `q`, default `p + 1`, must exceed `p`).
/// * `deriv` - Derivative order of the estimand: 0 = level, 1 = kink (R `deriv`, default 0).
/// * `kernel` - Kernel for the local-polynomial weights (R `kernel`, default triangular).
/// * `vce_kind` - Variance estimator before clustering remap: nn, hc0-hc3 (R `vce`, default nn).
/// * `nnmatch` - Minimum neighbors for `vce = nn` (R `nnmatch`, default 3).
/// * `covs_drop_coll` - Drop collinear covariates in the FWL partialling. This port
///   defaults to false (R's default is true); the flag only switches the ginv versus
///   Cholesky solve in the FWL step, and the projection onto col(Z) is unique, so it
///   changes no coefficient. R `covs_drop`; the flag is estimate-identical, not a
///   default-parity match.
/// * `scaleregul` - Regularization scale in bandwidth selection; 0 removes it (R `scaleregul`, default 1).
/// * `scalepar` - Scaling applied to the RD parameter (R `scalepar`, default 1).
/// * `level` - Confidence-interval level in percent, e.g. 95 (R `level`, default 95).
/// * `bwselect` - Bandwidth-selection method when `h` is `None`, e.g. `"mserd"` (R `bwselect`, default "mserd").
/// * `cluster` - Cluster ids; presence remaps `vce_kind` to CR1/CRV2/CRV3 (R `cluster`, default NULL).
/// * `weights` - User weights multiplying the kernel; `None` for unweighted (R `weights`, default NULL).
/// * `masspoints` - Mass-point handling: `"off"`, `"check"`, or `"adjust"` (R `masspoints`, default "adjust").
/// * `stdvars` - Standardize `x` and `y` before bandwidth selection (R `stdvars`, default true).
/// * `bwrestrict` - Clamp selected bandwidths to the range of `x` (R `bwrestrict`, default true).
/// * `bwcheck` - Minimum unique observations required in the pilot bandwidth; `None` to skip (R `bwcheck`, default NULL).
/// * `rho` - Sets `b = h / rho`; `None` leaves `b` from `b`/selection (R `rho`, default NULL).
/// * `sharpbw` - Force sharp-RD bandwidth selection in a fuzzy design (R `sharpbw`, default false).
/// * `h` - Main bandwidths `(left, right)`; `None` triggers selection via [`rdbwselect`] (R `h`, default NULL).
/// * `b` - Bias bandwidths `(left, right)`; `None` defaults to `h` (or `h / rho`) (R `b`, default NULL).
///
/// # Returns
///
/// An [`RdRobust`] with the three-row coef/se/z/pv/ci and the 2x2 bandwidths used.
#[allow(clippy::too_many_arguments)]
pub fn rdrobust(
    y: &[f64],
    x: &[f64],
    t: Option<&[f64]>,
    z: Option<&[f64]>,
    dz: usize,
    c: f64,
    p: usize,
    q: usize,
    deriv: usize,
    kernel: Kernel,
    vce_kind: Vce,
    nnmatch: usize,
    covs_drop_coll: bool,
    scaleregul: f64,
    scalepar: f64,
    level: f64,
    bwselect: &str,
    cluster: Option<&[f64]>,
    weights: Option<&[f64]>,
    masspoints: &str,
    stdvars: bool,
    bwrestrict: bool,
    bwcheck: Option<usize>,
    rho: Option<f64>,
    sharpbw: bool,
    h: Option<(f64, f64)>,
    b: Option<(f64, f64)>,
) -> RdRobust {
    // perf_comp: a fuzzy design with one-sided (constant-T) compliance. R selects
    // the bandwidth as if SHARP (treatment dropped) but estimates fuzzy at it
    // (rdrobust.R:166-185, 581-591); `sharpbw=true` forces the same regardless.
    let perf_comp = t.is_some_and(|tt| crate::bwselect::perf_comp(x, tt, c));
    // treatment used for BANDWIDTH SELECTION (dropped -> sharp BW when perf_comp/sharpbw)
    let bw_t = if perf_comp || sharpbw { None } else { t };

    // Small samples (rdrobust.R:303-307): with fewer than 20 observations R skips
    // data-driven selection entirely and uses the full support on both sides,
    // h = b = max(range_l, range_r), overriding any user-supplied h/b/rho. The
    // ranges are taken on the original x scale (R measures them on the stdvars
    // scale but then applies them to the de-standardized data, which is a
    // units mismatch; this port keeps the bandwidths in x units).
    let h = if y.len() < 20 {
        let x_min = x.iter().cloned().fold(f64::INFINITY, f64::min);
        let x_max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let full = (c - x_min).abs().max((x_max - c).abs());
        Some((full, full))
    } else {
        h
    };
    let (rho, b) = if y.len() < 20 { (None, None) } else { (rho, b) };

    // bandwidths (rho: b = h/rho; rdrobust.R:296-300 manual-h path, 501-504 auto path)
    let (h_l, h_r, b_l, b_r) = match h {
        Some((hl, hr)) => {
            let (bl, br) = match (rho, b) {
                (Some(r), _) => (hl / r, hr / r),   // rho overrides any b
                (None, Some((bl, br))) => (bl, br), // explicit b
                (None, None) => (hl, hr),           // rho defaults to 1 -> b = h
            };
            (hl, hr, bl, br)
        }
        None => {
            let bw = rdbwselect(
                y,
                x,
                bw_t,
                z,
                dz,
                c,
                p,
                q,
                deriv,
                kernel,
                vce_kind,
                nnmatch,
                covs_drop_coll,
                scaleregul,
                bwselect,
                cluster,
                weights,
                masspoints,
                stdvars,
                bwrestrict,
                bwcheck,
            );
            match rho {
                Some(r) => (bw.h_l, bw.h_r, bw.h_l / r, bw.h_r / r),
                None => (bw.h_l, bw.h_r, bw.b_l, bw.b_r),
            }
        }
    };

    // A failed selection (NaN bandwidth: an empty pilot window or a 0/0 stage)
    // yields NaN everywhere in R, since `w = K((x - c)/NaN)` is NA. A NaN
    // bandwidth here would instead empty every window and fit zeros, so short-
    // circuit to the same NaN outcome rather than report a spurious 0 +/- 0.
    if [h_l, h_r, b_l, b_r].iter().any(|v| v.is_nan()) {
        return RdRobust {
            coef: [f64::NAN; 3],
            se: [f64::NAN; 3],
            z: [f64::NAN; 3],
            pv: [f64::NAN; 3],
            ci: [[f64::NAN; 2]; 3],
            bws: [[h_l, h_r], [b_l, b_r]],
        };
    }

    // split sides (x<c left, x>=c right); z is flattened row-major (n x dz)
    let (mut yl, mut xl, mut tl, mut zl, mut cll, mut fwl) = (
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
    );
    let (mut yr, mut xr, mut tr, mut zr, mut clr, mut fwr) = (
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
    );
    for i in 0..y.len() {
        if x[i] < c {
            yl.push(y[i]);
            xl.push(x[i]);
            if let Some(tt) = t {
                tl.push(tt[i]);
            }
            if let Some(zz) = z {
                zl.extend_from_slice(&zz[i * dz..i * dz + dz]);
            }
            if let Some(cc) = cluster {
                cll.push(cc[i]);
            }
            if let Some(ww) = weights {
                fwl.push(ww[i]);
            }
        } else {
            yr.push(y[i]);
            xr.push(x[i]);
            if let Some(tt) = t {
                tr.push(tt[i]);
            }
            if let Some(zz) = z {
                zr.extend_from_slice(&zz[i * dz..i * dz + dz]);
            }
            if let Some(cc) = cluster {
                clr.push(cc[i]);
            }
            if let Some(ww) = weights {
                fwr.push(ww[i]);
            }
        }
    }
    let tl_opt = t.map(|_| tl.as_slice());
    let tr_opt = t.map(|_| tr.as_slice());
    let zl_opt = z.map(|_| zl.as_slice());
    let zr_opt = z.map(|_| zr.as_slice());
    let fwl_opt = weights.map(|_| fwl.as_slice());
    let fwr_opt = weights.map(|_| fwr.as_slice());
    let cll_opt = cluster.map(|_| cll.as_slice());
    let clr_opt = cluster.map(|_| clr.as_slice());

    let fl = fit_side(
        &yl, &xl, tl_opt, zl_opt, cll_opt, fwl_opt, dz, c, p, q, h_l, b_l, kernel, vce_kind,
    );
    let fr = fit_side(
        &yr, &xr, tr_opt, zr_opt, clr_opt, fwr_opt, dz, c, p, q, h_r, b_r, kernel, vce_kind,
    );

    let fac = vce::factorial(deriv);
    let beta_p_diff = &fr.beta_p - &fl.beta_p;
    let beta_bc_diff = &fr.beta_bc - &fl.beta_bc;
    let fuzzy = t.is_some();
    let covs = dz > 0;

    // tau + s_Y by (fuzzy, covs) (R:619-717)
    let (tau_cl, tau_bc, s_y): (f64, f64, DVector<f64>) = match (fuzzy, covs) {
        (false, false) => (
            scalepar * fac * beta_p_diff[(deriv, 0)],
            scalepar * fac * beta_bc_diff[(deriv, 0)],
            DVector::from_element(1, 1.0),
        ),
        (true, false) => {
            let tau_y_cl = scalepar * fac * beta_p_diff[(deriv, 0)];
            let tau_t_cl = fac * beta_p_diff[(deriv, 1)];
            let tau_y_bc = scalepar * fac * beta_bc_diff[(deriv, 0)];
            let tau_t_bc = fac * beta_bc_diff[(deriv, 1)];
            let tcl = tau_y_cl / tau_t_cl;
            let s = DVector::from_vec(vec![1.0 / tau_t_cl, -(tau_y_cl / (tau_t_cl * tau_t_cl))]);
            let b_f = DVector::from_vec(vec![tau_y_cl - tau_y_bc, tau_t_cl - tau_t_bc]);
            let tbc = tcl - s.dot(&b_f);
            (tcl, tbc, s)
        }
        (false, true) => {
            // sharp + covariates: FWL gamma combined across sides (rdrobust.R:668-686).
            // tau = scalepar * s_Y . beta_diff[deriv,:] (R:675, no factorial -> deriv=0).
            let zwz = fl.zwz_p.as_ref().unwrap() + fr.zwz_p.as_ref().unwrap();
            let zwy = fl.zwy_p.as_ref().unwrap() + fr.zwy_p.as_ref().unwrap();
            let gamma = vce::gamma_from(&zwz, &zwy, covs_drop_coll); // dz x 1
            let mut s = DVector::zeros(1 + dz);
            s[0] = 1.0;
            for j in 0..dz {
                s[1 + j] = -gamma[(j, 0)];
            }
            let dot = |bd: &DMatrix<f64>| (0..s.len()).map(|k| s[k] * bd[(deriv, k)]).sum::<f64>();
            (
                scalepar * dot(&beta_p_diff),
                scalepar * dot(&beta_bc_diff),
                s,
            )
        }
        (true, true) => {
            // fuzzy + covariates (rdrobust.R:688-722): combined FWL gamma (col 0 = Y
            // partial, col 1 = T partial), then the delta-method ratio over the
            // covariate-adjusted Y/T derivative estimates.
            let zwz = fl.zwz_p.as_ref().unwrap() + fr.zwz_p.as_ref().unwrap();
            let zwy = fl.zwy_p.as_ref().unwrap() + fr.zwy_p.as_ref().unwrap();
            let gamma = vce::gamma_from(&zwz, &zwy, covs_drop_coll); // dz x 2
            let cz = 2; // Z block start in beta_diff (after Y, T)
            let dot = |bd: &DMatrix<f64>, yt: usize, gcol: usize| -> f64 {
                bd[(deriv, yt)]
                    + (0..dz)
                        .map(|j| -gamma[(j, gcol)] * bd[(deriv, cz + j)])
                        .sum::<f64>()
            };
            let tau_y_cl = scalepar * fac * dot(&beta_p_diff, 0, 0);
            let tau_t_cl = fac * dot(&beta_p_diff, 1, 1);
            let tau_y_bc = scalepar * fac * dot(&beta_bc_diff, 0, 0);
            let tau_t_bc = fac * dot(&beta_bc_diff, 1, 1);
            let tcl = tau_y_cl / tau_t_cl;
            let s_delta =
                DVector::from_vec(vec![1.0 / tau_t_cl, -(tau_y_cl / (tau_t_cl * tau_t_cl))]);
            let b_f = DVector::from_vec(vec![tau_y_cl - tau_y_bc, tau_t_cl - tau_t_bc]);
            let tbc = tcl - s_delta.dot(&b_f);
            let mut s = DVector::zeros(2 + dz);
            s[0] = 1.0 / tau_t_cl;
            s[1] = -(tau_y_cl / (tau_t_cl * tau_t_cl));
            for j in 0..dz {
                s[2 + j] = -(1.0 / tau_t_cl) * gamma[(j, 0)]
                    + (tau_y_cl / (tau_t_cl * tau_t_cl)) * gamma[(j, 1)];
            }
            (tcl, tbc, s)
        }
    };

    // variance: conventional uses res_h (p-fit) + (R_p*W_h) meat; robust uses
    // res_b (q-fit; res_b=res_h for nn) + Q_q meat; both with invG_p (R:750-794).
    // Clustering routes the meat through the CR1/CRV2/CRV3 path; the robust
    // variance has four cases (h==b vs h!=b) x (CR1 vs CRV2/3) (R:769-795).
    let has_cluster = cluster.is_some();
    let kind = vce::cluster_kind(vce_kind);
    let crv2 = kind == vce::ClusterKind::Crv2;
    let hb_match = h_l == b_l && h_r == b_r;
    let side_v = |f: &SideFit| -> (DMatrix<f64>, DMatrix<f64>) {
        let res_h = vce::residuals(
            &f.ex,
            &f.r_p,
            &f.inv_g_p,
            &f.rp_wh,
            &f.beta_p,
            &f.d,
            f.en,
            vce_kind,
            &f.edups,
            &f.edupsid,
            nnmatch,
            has_cluster,
        );
        // nn shares res_b=res_h, but clustering remaps nn away (rdrobust.R:320),
        // so a clustered fit always takes the q-fit residual branch.
        let res_b = if matches!(vce_kind, Vce::NearestNeighbor) && !has_cluster {
            res_h.clone()
        } else {
            vce::residuals(
                &f.ex,
                &f.r_q,
                &f.inv_g_q,
                &f.rq_wb,
                &f.beta_q,
                &f.d,
                f.en,
                vce_kind,
                &f.edups,
                &f.edupsid,
                nnmatch,
                has_cluster,
            )
        };
        if !has_cluster {
            let v_cl = vce::v_matrix(&f.inv_g_p, &f.rp_wh, &res_h, &s_y);
            let v_rb = vce::v_matrix(&f.inv_g_p, &f.q_q, &res_b, &s_y);
            return (v_cl, v_rb);
        }
        let groups = vce::cluster_idx(&f.ec);
        // conventional: p-design at h, bread invG_p (R:769-770)
        let v_cl = vce::v_cluster(
            &f.inv_g_p,
            &f.rp_wh,
            &res_h,
            &s_y,
            &groups,
            kind,
            Some(&f.rp_swh),
            None,
        );
        let v_rb = if hb_match {
            match kind {
                // h==b: direct q-regression (R:779-783); CR1 bread invG_q, no override
                vce::ClusterKind::Cr1 => vce::v_cluster(
                    &f.inv_g_q, &f.rq_wb, &res_b, &s_y, &groups, kind, None, None,
                ),
                _ => vce::v_cluster(
                    &f.inv_g_q,
                    &f.rq_wb,
                    &res_b,
                    &s_y,
                    &groups,
                    kind,
                    Some(&f.rq_swb),
                    None,
                ),
            }
        } else {
            match kind {
                // h!=b CR1: Q_q sandwich, bread invG_p, k_override=q+1 (R:793-794)
                vce::ClusterKind::Cr1 => vce::v_cluster(
                    &f.inv_g_p,
                    &f.q_q,
                    &res_b,
                    &s_y,
                    &groups,
                    kind,
                    None,
                    Some(q + 1),
                ),
                // h!=b CRV2/3: decoupled two-stage qq-cluster meat (R:785-788)
                _ => {
                    let m = vce::vce_meat_qq_cluster(
                        &f.q_q, &f.r_q, &f.w_b, &f.inv_g_q, &res_b, &groups, crv2, &s_y,
                    );
                    &f.inv_g_p * (m * &f.inv_g_p)
                }
            }
        };
        (v_cl, v_rb)
    };
    let (v_cl_l, v_rb_l) = side_v(&fl);
    let (v_cl_r, v_rb_r) = side_v(&fr);

    let scale2 = scalepar * scalepar * fac * fac;
    let v_tau_cl = scale2 * (v_cl_l[(deriv, deriv)] + v_cl_r[(deriv, deriv)]);
    let v_tau_rb = scale2 * (v_rb_l[(deriv, deriv)] + v_rb_r[(deriv, deriv)]);
    let se_cl = v_tau_cl.sqrt();
    let se_rb = v_tau_rb.sqrt();

    // assemble coef / se / z / pv / ci (R:854-863)
    let coef = [tau_cl, tau_bc, tau_bc];
    let se = [se_cl, se_cl, se_rb];
    let quant = -qnorm((1.0 - level / 100.0) / 2.0);
    let mut z = [0.0; 3];
    let mut pv = [0.0; 3];
    let mut ci = [[0.0; 2]; 3];
    for i in 0..3 {
        z[i] = coef[i] / se[i];
        pv[i] = 2.0 * pnorm(-z[i].abs());
        ci[i] = [coef[i] - quant * se[i], coef[i] + quant * se[i]];
    }

    RdRobust {
        coef,
        se,
        z,
        pv,
        ci,
        bws: [[h_l, h_r], [b_l, b_r]],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dgp(n: usize) -> (Vec<f64>, Vec<f64>) {
        let x: Vec<f64> = (0..n)
            .map(|i| -1.0 + 2.0 * (i as f64 + 0.5) / n as f64)
            .collect();
        let y: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, &xi)| 0.5 * xi + if xi >= 0.0 { 2.0 } else { 0.0 } + 0.3 * (i as f64).sin())
            .collect();
        (x, y)
    }

    #[test]
    fn cutoff_must_be_strictly_inside_support() {
        let (x, y) = dgp(200);
        let at = |c: f64| {
            try_rdrobust_with(
                &y,
                &x,
                None,
                None,
                0,
                None,
                None,
                &RdRobustOptions {
                    c,
                    ..Default::default()
                },
            )
        };
        assert_eq!(at(x[x.len() - 1]).map(|_| ()), Err(RdError::OneSidedCutoff));
        assert_eq!(at(x[0]).map(|_| ()), Err(RdError::OneSidedCutoff));
        assert_eq!(at(2.0).map(|_| ()), Err(RdError::OneSidedCutoff));
        assert!(at(0.0).is_ok());
        assert!(at(x[1]).is_ok());
    }

    #[test]
    fn cutoff_at_max_x_does_not_panic_in_raw_path() {
        // The raw path has no validation; an empty pilot window must not panic
        // (debug-mode `n - 1` underflow in the nn residual). The right side is a
        // single point with zero range, so selection fails and, as in R, the
        // whole result is NaN rather than a spurious 0 +/- 0.
        let (x, y) = dgp(200);
        let c = x[x.len() - 1];
        let out = rdrobust_with(
            &y,
            &x,
            None,
            None,
            0,
            None,
            None,
            &RdRobustOptions {
                c,
                ..Default::default()
            },
        );
        assert!(out.coef.iter().chain(out.se.iter()).all(|v| v.is_nan()));
        assert!(out.bws[0][0].is_nan());
    }

    #[test]
    fn ragged_covariates_rejected() {
        let (x, y) = dgp(100);
        let z = vec![0.0; 2 * 100 - 1];
        let err = try_rdrobust_with(
            &y,
            &x,
            None,
            Some(&z),
            2,
            None,
            None,
            &RdRobustOptions::default(),
        );
        assert!(matches!(
            err,
            Err(RdError::LengthMismatch { name: "covs", .. })
        ));
    }

    #[test]
    fn degenerate_fuzzy_rejected_but_one_sided_compliance_allowed() {
        let (x, y) = dgp(200);
        let flat = vec![0.5; 200];
        let err = try_rdrobust_with(
            &y,
            &x,
            Some(&flat),
            None,
            0,
            None,
            None,
            &RdRobustOptions::default(),
        );
        assert_eq!(err.map(|_| ()), Err(RdError::DegenerateFuzzy));
        // constant on both sides WITH a jump is a (sharp-like) fuzzy design R accepts
        let step: Vec<f64> = x
            .iter()
            .map(|&v| if v >= 0.0 { 1.0 } else { 0.0 })
            .collect();
        let out = try_rdrobust_with(
            &y,
            &x,
            Some(&step),
            None,
            0,
            None,
            None,
            &RdRobustOptions::default(),
        )
        .unwrap();
        let sharp = rdrobust_with(
            &y,
            &x,
            None,
            None,
            0,
            None,
            None,
            &RdRobustOptions::default(),
        );
        assert!((out.coef[0] - sharp.coef[0]).abs() < 1e-12);
        assert_eq!(out.bws, sharp.bws);
    }

    #[test]
    fn fewer_than_twenty_obs_use_full_support() {
        let x: Vec<f64> = (0..12).map(|i| -0.8 + 2.0 * i as f64 / 11.0).collect();
        let y: Vec<f64> = x
            .iter()
            .map(|&v| 0.3 * v + if v >= 0.0 { 1.0 } else { 0.0 })
            .collect();
        let full = (x[11] - 0.0).abs().max((0.0 - x[0]).abs());
        for opts in [
            RdRobustOptions::default(),
            RdRobustOptions {
                h: Some((0.1, 0.1)),
                ..Default::default()
            },
            RdRobustOptions {
                h: Some((0.1, 0.1)),
                rho: Some(0.5),
                ..Default::default()
            },
        ] {
            let out = rdrobust_with(&y, &x, None, None, 0, None, None, &opts);
            assert_eq!(out.bws, [[full, full], [full, full]]);
            assert!(out.coef[0].is_finite() && out.se[2].is_finite());
        }
        // at n = 20 selection runs again
        let (x20, y20) = dgp(20);
        let out = rdrobust_with(
            &y20,
            &x20,
            None,
            None,
            0,
            None,
            None,
            &RdRobustOptions::default(),
        );
        assert!(out.bws[0][0] < 1.0);
    }
}
