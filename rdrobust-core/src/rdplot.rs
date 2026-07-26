//! RD plot port -- `rdplot` (R rdplot.R): the binned scatter (per-bin means),
//! the IMSE/MV bin-count selection, and the global polynomial curve coefficients.
//!
//! Implemented: the ES family of binselect (`es`, `esmv` (default),
//! `espr`, `esmvpr`) with no covariates, no CI bands. QS spacing, covariate
//! adjustment, and CI/sd are later increments. Parity: exact bin counts
//! `J_l`/`J_r`, poly coefficients and per-bin means ~1e-8 vs R.

use nalgebra::{DMatrix, DVector};

use crate::kernel::eff_weight;
use crate::linalg::{qr_xx_inv, vander};
use crate::Kernel;

/// Numeric outputs behind an RD plot: selected bin counts, global polynomial curve
/// coefficients, and per-bin scatter means for each side.
///
/// Ports the subset of the R `rdplot` return object the numeric core needs (R
/// rdplot.R:557-607): the selected counts `J`, the curve coefficients `coef`, and
/// the per-bin x/y means that place each scatter dot. Plotting, CI bands, per-bin
/// N/sd/SE, bin midpoints, and the QS/covariate paths are not part of this slice.
/// Bin counts match R exactly; coefficients and per-bin means match R to ~1e-8.
#[derive(Debug, Clone)]
pub struct RdPlot {
    /// Selected number of bins on the left side (`x < c`), post scale/nbins, clamped
    /// to at least 1. R `J[1]` (rdplot.R:449-466).
    pub j_l: usize,
    /// Selected number of bins on the right side (`x >= c`), post scale/nbins, clamped
    /// to at least 1. R `J[2]`.
    pub j_r: usize,
    /// Left-side global polynomial coefficients, length `p + 1` (orders 0..=p in `x - c`).
    /// R `gamma_p1_l` / `coef[,"Left"]` (rdplot.R:206-225).
    pub coef_l: Vec<f64>,
    /// Right-side global polynomial coefficients, length `p + 1`. R `gamma_p1_r` /
    /// `coef[,"Right"]`.
    pub coef_r: Vec<f64>,
    /// Sorted unique non-empty left bin ids (negative: `-j_l` leftmost, `-1` just below
    /// `c`). R `sort(unique(bin_x_l))` (rdplot.R:484).
    pub bin_l: Vec<i64>,
    /// Sorted unique non-empty right bin ids (`1..=j_r`, ascending in x). R
    /// `sort(unique(bin_x_r))`.
    pub bin_r: Vec<i64>,
    /// Per-bin mean of `x` on the left, ordered by sorted bin id (the scatter dot's
    /// x-coordinate). One entry per non-empty bin. R `tapply(x_l, bin_x_l, mean)`
    /// (rdplot.R:488-490).
    pub mean_x_l: Vec<f64>,
    /// Per-bin mean of `y` on the left, ordered by sorted bin id (the scatter dot's
    /// y-coordinate). R `tapply(y_l, bin_x_l, mean)`.
    pub mean_y_l: Vec<f64>,
    /// Per-bin mean of `x` on the right, ordered by sorted bin id. R
    /// `tapply(x_r, bin_x_r, mean)`.
    pub mean_x_r: Vec<f64>,
    /// Per-bin mean of `y` on the right, ordered by sorted bin id. R
    /// `tapply(y_r, bin_x_r, mean)`.
    pub mean_y_r: Vec<f64>,
}

/// Row-scale a matrix by a per-row vector (`diag(w) * m`).
fn row_scale(m: &DMatrix<f64>, w: &[f64]) -> DMatrix<f64> {
    let mut out = m.clone();
    for i in 0..m.nrows() {
        for j in 0..m.ncols() {
            out[(i, j)] *= w[i];
        }
    }
    out
}

/// Sample variance (ddof=1), R `var`.
fn var_y(y: &[f64]) -> f64 {
    let n = y.len() as f64;
    let mean = y.iter().sum::<f64>() / n;
    y.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / (n - 1.0)
}

/// Global weighted local-polynomial fit of order `p` over a side (kernel weights
/// at bandwidth `h`, times user weights `wt`; points outside the window get weight
/// 0). Returns the (p+1) coefficients (R rdplot.R:206-225,212-214).
fn poly_fit(
    x: &[f64],
    y: &[f64],
    c: f64,
    p: usize,
    h: f64,
    kernel: Kernel,
    wt: Option<&[f64]>,
) -> Vec<f64> {
    let w = eff_weight(x, c, h, kernel, wt);
    let u: Vec<f64> = x.iter().map(|&xi| xi - c).collect();
    let r_p = vander(&u, p);
    let wsqrt: Vec<f64> = w.iter().map(|v| v.sqrt()).collect();
    let inv_g = qr_xx_inv(&row_scale(&r_p, &wsqrt));
    let rw = row_scale(&r_p, &w);
    let yv = DVector::from_column_slice(y);
    let gamma = &inv_g * (rw.transpose() * &yv);
    gamma.iter().copied().collect()
}

/// The four ES bin-count quantities for one side (rdplot.R:276-388, ES half).
struct EsPilot {
    hat_dw: f64, // J_es_hat_dw (IMSE, spacings sigma2)
    chk_dw: f64, // J_es_chk_dw (IMSE, poly-reg sigma2)
    hat_mv: f64, // J_es_hat_mv (mimicking-variance, spacings)
    chk_mv: f64, // J_es_chk_mv (mimicking-variance, poly-reg)
}

/// Vandermonde derivative design: col j = j * x^(j-1) (col 0 = 0), n x (k+1).
fn dvander(x: &[f64], k: usize) -> DMatrix<f64> {
    let n = x.len();
    let mut d = DMatrix::zeros(n, k + 1);
    for i in 0..n {
        for j in 1..=k {
            d[(i, j)] = j as f64 * x[i].powi((j - 1) as i32);
        }
    }
    d
}

/// ES pilot for one side (R rdplot.R:276-388). `x`/`y` are the side's points,
/// `range = |c - x_endpoint|` (support half-range), `n` is the TOTAL sample size.
fn es_pilot(x: &[f64], y: &[f64], range: f64, n: usize, vy: f64) -> EsPilot {
    // pilot order k (4 -> 3 -> 2 on singular Gram), UNWEIGHTED, RAW x
    let mut k = 4usize;
    let (rk, inv_gk) = loop {
        let rk = vander(x, k);
        let g = rk.transpose() * &rk;
        match g.cholesky() {
            Some(_) => break (rk.clone(), qr_xx_inv(&rk)),
            None if k > 2 => k -= 1,
            None => break (rk.clone(), qr_xx_inv(&rk)),
        }
    };
    let yv = DVector::from_column_slice(y);
    let y2: Vec<f64> = y.iter().map(|&v| v * v).collect();
    let y2v = DVector::from_column_slice(&y2);
    let gamma_k1 = &inv_gk * (rk.transpose() * &yv);
    let gamma_k2 = &inv_gk * (rk.transpose() * &y2v);

    // mu's at raw points
    let drk = dvander(x, k);
    let mu1_hat = &drk * &gamma_k1; // fitted first derivative

    // sort (x, y) ascending by x for the spacings estimators
    let mut order: Vec<usize> = (0..x.len()).collect();
    order.sort_by(|&a, &b| x[a].total_cmp(&x[b]));
    let xs: Vec<f64> = order.iter().map(|&i| x[i]).collect();
    let ys: Vec<f64> = order.iter().map(|&i| y[i]).collect();
    let m = xs.len();
    let dxi: Vec<f64> = (1..m).map(|i| xs[i] - xs[i - 1]).collect();
    let dyi: Vec<f64> = (1..m).map(|i| ys[i] - ys[i - 1]).collect();
    let x_bar: Vec<f64> = (1..m).map(|i| 0.5 * (xs[i] + xs[i - 1])).collect();

    // sigma2 at midpoints (poly-reg estimator), negatives -> var_y
    let rk_bar = vander(&x_bar, k);
    let mu0_i = &rk_bar * &gamma_k1;
    let mu2_i = &rk_bar * &gamma_k2;
    let sigma2_bar: Vec<f64> = (0..x_bar.len())
        .map(|i| {
            let s = mu2_i[i] - mu0_i[i] * mu0_i[i];
            if s < 0.0 {
                vy
            } else {
                s
            }
        })
        .collect();

    let nf = n as f64;
    let sum_mu1_sq: f64 = mu1_hat.iter().map(|v| v * v).sum();
    let b_es = (range * range / (12.0 * nf)) * sum_mu1_sq;
    let v_hat = (0.5 / range)
        * dxi
            .iter()
            .zip(&dyi)
            .map(|(&dx, &dy)| dx * dy * dy)
            .sum::<f64>();
    let v_chk = (1.0 / range)
        * dxi
            .iter()
            .zip(&sigma2_bar)
            .map(|(&dx, &s)| dx * s)
            .sum::<f64>();

    let j_fun = |b: f64, v: f64| (((2.0 * b) / v) * nf).powf(1.0 / 3.0).ceil();
    let nlogn = nf / (nf.ln() * nf.ln());
    EsPilot {
        hat_dw: j_fun(b_es, v_hat),
        chk_dw: j_fun(b_es, v_chk),
        hat_mv: ((vy / v_hat) * nlogn).ceil(),
        chk_mv: ((vy / v_chk) * nlogn).ceil(),
    }
}

/// R `findInterval(v, knots, rightmost.closed=TRUE)`: count of knots <= v, with the
/// top knot folded into the last interval. knots ascending, length J+1.
fn find_interval(v: f64, knots: &[f64]) -> usize {
    let j = knots.len() - 1;
    if v < knots[0] {
        return 0;
    }
    if v >= knots[j] {
        return j;
    }
    knots.iter().filter(|&&k| k <= v).count()
}

/// Per-bin means grouped by bin id (ascending), dropping empty bins (R `tapply`).
///
/// Single forward pass over `bins` in increasing index order, accumulating
/// `(sum_x, sum_y, count)` per id in a `BTreeMap` (keys iterate ascending, matching
/// the sorted-unique-id order). Accumulating in index order keeps each bin's
/// floating-point sum bit-identical to the per-id-scan form.
fn bin_means(x: &[f64], y: &[f64], bins: &[i64]) -> (Vec<i64>, Vec<f64>, Vec<f64>) {
    use std::collections::BTreeMap;
    let mut acc: BTreeMap<i64, (f64, f64, usize)> = BTreeMap::new();
    for k in 0..bins.len() {
        let e = acc.entry(bins[k]).or_insert((0.0, 0.0, 0));
        e.0 += x[k];
        e.1 += y[k];
        e.2 += 1;
    }
    let mut ids = Vec::with_capacity(acc.len());
    let mut mx = Vec::with_capacity(acc.len());
    let mut my = Vec::with_capacity(acc.len());
    for (&id, &(sx, sy, cnt)) in &acc {
        ids.push(id);
        mx.push(sx / cnt as f64);
        my.push(sy / cnt as f64);
    }
    (ids, mx, my)
}

/// Computes the data behind an RD plot for the evenly-spaced (ES) binselect family:
/// the global polynomial curve coefficients, the IMSE/MV bin counts, and the per-bin
/// scatter means. Ports R `rdplot` (rdplot.R), ES half only, no covariates and no CI.
///
/// Splits the sample at `c` (`x < c` left, `x >= c` right), fits one weighted
/// order-`p` polynomial per side over the full side range, selects `J_l`/`J_r` via the
/// order-4 ES pilot regressions, lays evenly-spaced breakpoints, assigns each point
/// with R's `findInterval` convention, and returns per-bin x/y means (empty bins
/// dropped). Bin counts are exact vs R; coefficients and means match to ~1e-8.
///
/// # Arguments
/// - `y`: outcome values.
/// - `x`: running variable, parallel to `y`, need not be sorted.
/// - `c`: cutoff; the point `x == c` falls on the right side.
/// - `p`: global polynomial order for the curve (R default 4).
/// - `kernel`: kernel for the polynomial-fit weights (uniform/triangular/Epanechnikov).
/// - `binselect`: ES variant string, one of `es`, `espr`, `esmv` (also matched by `""`),
///   or `esmvpr`. Any other value, including the QS variants, triggers `unimplemented!`.
/// - `scale_l`, `scale_r`: multipliers on the selected per-side bin count (R default 1).
/// - `nbins_l`, `nbins_r`: optional manual bin counts that override binselect per side.
/// - `wt`: optional unit weights that multiply the kernel weights in the curve fit only;
///   they do not enter the pilot regressions or bin construction.
///
/// # Panics
/// Panics via `unimplemented!` when `binselect` is not one of the four supported ES
/// strings (the QS family is a later increment).
#[allow(clippy::too_many_arguments)]
pub fn rdplot(
    y: &[f64],
    x: &[f64],
    c: f64,
    p: usize,
    kernel: Kernel,
    binselect: &str,
    scale_l: f64,
    scale_r: f64,
    nbins_l: Option<usize>,
    nbins_r: Option<usize>,
    wt: Option<&[f64]>,
) -> RdPlot {
    let x_min = x.iter().cloned().fold(f64::INFINITY, f64::min);
    let x_max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range_l = c - x_min;
    let range_r = x_max - c;

    // side split: x<c left, x>=c right (carry user weights for the poly fit only)
    let (mut xl, mut yl, mut wl, mut xr, mut yr, mut wr) = (
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
    );
    for i in 0..x.len() {
        if x[i] < c {
            xl.push(x[i]);
            yl.push(y[i]);
            if let Some(ww) = wt {
                wl.push(ww[i]);
            }
        } else {
            xr.push(x[i]);
            yr.push(y[i]);
            if let Some(ww) = wt {
                wr.push(ww[i]);
            }
        }
    }
    let wl_opt = wt.map(|_| wl.as_slice());
    let wr_opt = wt.map(|_| wr.as_slice());
    let n = x.len();

    // global poly curve coefficients (h = full side range, default; weighted)
    let coef_l = poly_fit(&xl, &yl, c, p, range_l, kernel, wl_opt);
    let coef_r = poly_fit(&xr, &yr, c, p, range_r, kernel, wr_opt);

    // ES pilot + binselect dispatch
    let vy_l = var_y(&yl);
    let vy_r = var_y(&yr);
    let pl = es_pilot(&xl, &yl, range_l, n, vy_l);
    let pr = es_pilot(&xr, &yr, range_r, n, vy_r);
    let pick = |pe: &EsPilot| -> f64 {
        match binselect {
            "es" => pe.hat_dw,
            "espr" => pe.chk_dw,
            "esmv" | "" => pe.hat_mv,
            "esmvpr" => pe.chk_mv,
            other => unimplemented!("binselect {other} not yet ported (ES family only)"),
        }
    };
    let j_l = nbins_l
        .unwrap_or_else(|| (scale_l * pick(&pl)).ceil() as usize)
        .max(1);
    let j_r = nbins_r
        .unwrap_or_else(|| (scale_r * pick(&pr)).ceil() as usize)
        .max(1);

    // ES breakpoints: x_min + i*jump  (J+1 knots), last = endpoint
    let jump_l = range_l / j_l as f64;
    let jump_r = range_r / j_r as f64;
    let knots_l: Vec<f64> = (0..=j_l).map(|i| x_min + i as f64 * jump_l).collect();
    let knots_r: Vec<f64> = (0..=j_r).map(|i| c + i as f64 * jump_r).collect();

    // bin assignment (R findInterval; left ids negative)
    let bins_l: Vec<i64> = xl
        .iter()
        .map(|&v| find_interval(v, &knots_l) as i64 - j_l as i64 - 1)
        .collect();
    let bins_r: Vec<i64> = xr
        .iter()
        .map(|&v| find_interval(v, &knots_r) as i64)
        .collect();

    let (bin_l, mean_x_l, mean_y_l) = bin_means(&xl, &yl, &bins_l);
    let (bin_r, mean_x_r, mean_y_r) = bin_means(&xr, &yr, &bins_r);

    RdPlot {
        j_l,
        j_r,
        coef_l,
        coef_r,
        bin_l,
        bin_r,
        mean_x_l,
        mean_y_l,
        mean_x_r,
        mean_y_r,
    }
}
