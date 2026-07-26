//! Local-polynomial fit engine -- port of `rdrobust_bw` (R functions.R:213-361),
//! the per-side engine producing the `V`/`B`/`R`/`rate` ingredients of the
//! MSE/CER bandwidth formula. Residuals and the sandwich meat live in `vce`.
//!
//! Implemented: sharp/fuzzy, covariates, clusters (CR1/CRV2/CRV3), user weights,
//! nn/hc0-hc3. Split into `compute_vfit` (V-window, cacheable per `(o,nu)`) +
//! `assemble_from_vfit` (per-`h_B` B-fit) for the `vcache`. Parity: V/B/R/rate
//! ~1e-8 vs R.

use nalgebra::{DMatrix, DVector};

use crate::kernel::eff_weight;
use crate::linalg::{ginv, qr_xx_inv, vander};
use crate::vce;
use crate::{Kernel, Vce};

/// The four scalars `rdrobust_bw` returns (R functions.R:359): the variance,
/// bias, regularization, and rate ingredients of the MSE/CER bandwidth formula.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BwParts {
    /// Variance term `(2*nu+1) * h_V^(2*nu+1) * V_V` (R:356).
    pub v: f64,
    /// Bias term `sqrt(2*(o+1-nu)) * BConst * (s . beta_B[last])` (R:355).
    pub b: f64,
    /// Regularization term `scale * 2*(o+1-nu) * BWreg`, with `BWreg = 3*BConst^2*V_B` (R:357, 353). Zero when `scale <= 0`.
    pub r: f64,
    /// MSE-rate exponent `1/(2*o+3)`, the order-dependent power in the bandwidth formula (R:358).
    pub rate: f64,
}

/// Kernel-weighted local-polynomial geometry on the in-window points: the pieces
/// depending only on `(x, c, o, h, kernel)`, not on the response columns.
pub(crate) struct Geom {
    pub inv_g: DMatrix<f64>,  // (R'WR)^{-1}, (o+1)x(o+1)
    pub rx: DMatrix<f64>,     // R * eW (row-scaled), n_w x (o+1)
    pub rxsqrt: DMatrix<f64>, // R * sqrt(eW) (row-scaled) -- crv2/crv3 sqrtRX
    pub r_mat: DMatrix<f64>,  // Vandermonde R, n_w x (o+1)
    pub idx: Vec<usize>,      // indices of in-window points into the full side array
    pub ex: Vec<f64>,         // in-window running variable
    pub ew: Vec<f64>,         // in-window kernel weights (for covariate partialling)
    pub n_w: usize,           // window size
}

/// Build the fit geometry of order `o` at bandwidth `h` (R:232-239). `R*sqrt(eW)`
/// and `R*eW` are ROW scalings (R's length-n x matrix recycling = `diag(.) %*% R`).
pub(crate) fn fit_geom(
    x: &[f64],
    c: f64,
    o: usize,
    h: f64,
    kernel: Kernel,
    wt: Option<&[f64]>,
) -> Geom {
    let w = eff_weight(x, c, h, kernel, wt);
    let idx: Vec<usize> = (0..x.len()).filter(|&i| w[i] > 0.0).collect();
    let n_w = idx.len();
    let ex: Vec<f64> = idx.iter().map(|&i| x[i]).collect();
    let ew: Vec<f64> = idx.iter().map(|&i| w[i]).collect();

    let u: Vec<f64> = ex.iter().map(|&xi| xi - c).collect();
    let r_mat = vander(&u, o);

    let mut rwsq = r_mat.clone();
    for i in 0..n_w {
        let s = ew[i].sqrt();
        for j in 0..=o {
            rwsq[(i, j)] *= s;
        }
    }
    let inv_g = qr_xx_inv(&rwsq);

    let mut rx = r_mat.clone();
    for i in 0..n_w {
        let wi = ew[i];
        for j in 0..=o {
            rx[(i, j)] *= wi;
        }
    }
    Geom {
        inv_g,
        rx,
        rxsqrt: rwsq,
        r_mat,
        idx,
        ex,
        ew,
        n_w,
    }
}

/// Window-subset response design `D`, columns `[Y, T?, Z1..ZdZ]`, n_w x ncols.
/// `z` is the full-side covariate matrix flattened ROW-MAJOR (n x dz), so the
/// covariate row for full-side index `i` is `z[i*dz .. i*dz+dz]`.
pub(crate) fn build_design(
    g: &Geom,
    y: &[f64],
    t: Option<&[f64]>,
    z: Option<&[f64]>,
    dz: usize,
) -> DMatrix<f64> {
    let dt = t.is_some() as usize;
    let mut d = DMatrix::zeros(g.n_w, 1 + dt + dz);
    for (r, &i) in g.idx.iter().enumerate() {
        d[(r, 0)] = y[i];
    }
    if let Some(tt) = t {
        for (r, &i) in g.idx.iter().enumerate() {
            d[(r, 1)] = tt[i];
        }
    }
    if let Some(zz) = z {
        for (r, &i) in g.idx.iter().enumerate() {
            for j in 0..dz {
                d[(r, 1 + dt + j)] = zz[i * dz + j];
            }
        }
    }
    d
}

/// WLS coefficients `(R'WR)^{-1} R'W D`, (o+1) x ncols (R:269).
pub(crate) fn fit_beta(g: &Geom, d: &DMatrix<f64>) -> DMatrix<f64> {
    &g.inv_g * (g.rx.transpose() * d)
}

/// The s / delta-method gradient (R:240-279). Four cases by (fuzzy, covs):
/// sharp `[1]`; fuzzy `[1/tau_T, -tau_Y/tau_T^2]`; covs-only `[1, -gamma[,0]]`;
/// fuzzy+covs the 3-block form. Covariates are partialled out by FWL on the
/// kernel-weighted Vandermonde, per side (R:247-263). `cz` = 0-based Z-block start.
pub(crate) fn compute_s(
    g: &Geom,
    beta: &DMatrix<f64>,
    d: &DMatrix<f64>,
    nu: usize,
    dt: usize,
    dz: usize,
    covs_drop_coll: bool,
) -> DVector<f64> {
    let fac = vce::factorial(nu);
    if dz == 0 {
        if dt == 0 {
            return DVector::from_element(1, 1.0);
        }
        let tau_y = fac * beta[(nu, 0)];
        let tau_t = fac * beta[(nu, 1)];
        return DVector::from_vec(vec![1.0 / tau_t, -(tau_y / (tau_t * tau_t))]);
    }

    // FWL partialling (R:251-263). cz = start of the Z block in D.
    let cz = 1 + dt;
    let ez = d.columns(cz, dz).into_owned();
    let mut ezw = ez;
    for i in 0..g.n_w {
        let w = g.ew[i];
        for j in 0..dz {
            ezw[(i, j)] *= w;
        }
    }
    let u = g.rx.transpose() * d; // (o+1) x ncols  = crossprod(R*eW, D)
    let zwd = ezw.transpose() * d; // dz x ncols     = crossprod(eZ*eW, D)
    let u_cz = u.columns(cz, dz).into_owned(); // (o+1) x dz
    let ig_u = &g.inv_g * &u; // (o+1) x ncols
    let uigu = u_cz.transpose() * ig_u; // dz x ncols
    let zwz = zwd.columns(cz, dz) - uigu.columns(cz, dz); // dz x dz
    let zwy = zwd.columns(0, 1 + dt) - uigu.columns(0, 1 + dt); // dz x (1+dt)
    let gamma = if covs_drop_coll {
        ginv(&zwz) * zwy
    } else {
        match zwz.clone().cholesky() {
            Some(ch) => ch.inverse() * zwy,
            None => ginv(&zwz) * zwy,
        }
    }; // dz x (1+dt)

    if dt == 0 {
        // covs only: s = [1, -gamma[,0]]
        let mut s = DVector::zeros(1 + dz);
        s[0] = 1.0;
        for j in 0..dz {
            s[1 + j] = -gamma[(j, 0)];
        }
        return s;
    }

    // fuzzy + covs (R:275-279). tau over [beta_Y/T, beta_Z-block] partialled by s/s_T.
    let beta_z: Vec<f64> = (0..dz).map(|j| beta[(nu, cz + j)]).collect();
    let dot_y = beta[(nu, 0)] + (0..dz).map(|j| -gamma[(j, 0)] * beta_z[j]).sum::<f64>();
    let dot_t = beta[(nu, 1)] + (0..dz).map(|j| -gamma[(j, 1)] * beta_z[j]).sum::<f64>();
    let tau_y = fac * dot_y;
    let tau_t = fac * dot_t;
    let mut s = DVector::zeros(2 + dz);
    s[0] = 1.0 / tau_t;
    s[1] = -(tau_y / (tau_t * tau_t));
    for j in 0..dz {
        s[2 + j] = -(1.0 / tau_t) * gamma[(j, 0)] + (tau_y / (tau_t * tau_t)) * gamma[(j, 1)];
    }
    s
}

/// Subset full-side `dups`/`dupsid` to a fit's window (R `dups[ind_V]`); empty
/// when there is no run-length structure (hc ignores it).
fn window_dups(g: &Geom, dups: &[usize], dupsid: &[usize]) -> (Vec<usize>, Vec<usize>) {
    if dups.is_empty() {
        return (Vec::new(), Vec::new());
    }
    (
        g.idx.iter().map(|&i| dups[i]).collect(),
        g.idx.iter().map(|&i| dupsid[i]).collect(),
    )
}

/// Cached V-fit pieces (R `list(V_V, BConst, s)`, functions.R:307): the part of
/// `rdrobust_bw` that depends only on `(o, nu)` when `h_V` is fixed across a side's
/// pilot calls. Reused by every pilot sharing an `(o, nu)` key (the `vcache`).
pub(crate) struct VFit {
    pub v_v: f64,
    pub bconst: f64,
    pub s: DVector<f64>,
}

/// Window-subset a fit's cluster ids into groups (empty when unclustered).
fn fit_groups(g: &Geom, cl: Option<&[f64]>) -> Vec<Vec<usize>> {
    match cl {
        Some(c) => vce::cluster_idx(&g.idx.iter().map(|&i| c[i]).collect::<Vec<_>>()),
        None => Vec::new(),
    }
}

/// V-fit half of `rdrobust_bw` (R functions.R:232-305): order-`o` geometry at
/// `h_V`, the s-vector, `V_V`, and `BConst`. A pure function of `(o, nu, h_V)` and
/// the side data, so it is computed once per `(o, nu)` and reused across pilots.
#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_vfit(
    y: &[f64],
    x: &[f64],
    t: Option<&[f64]>,
    z: Option<&[f64]>,
    dz: usize,
    c: f64,
    o: usize,
    nu: usize,
    h_v: f64,
    vce_kind: Vce,
    nnmatch: usize,
    covs_drop_coll: bool,
    kernel: Kernel,
    cl: Option<&[f64]>,
    wt: Option<&[f64]>,
) -> VFit {
    let dt = t.is_some() as usize;
    let has_cluster = cl.is_some();
    let kind = vce::cluster_kind(vce_kind);
    let (dups, dupsid) = match vce_kind {
        Vce::NearestNeighbor => vce::compute_dups_dupsid(x),
        _ => (Vec::new(), Vec::new()),
    };

    let gv = fit_geom(x, c, o, h_v, kernel, wt);
    let dv = build_design(&gv, y, t, z, dz);
    let beta_v = fit_beta(&gv, &dv);
    let s = compute_s(&gv, &beta_v, &dv, nu, dt, dz, covs_drop_coll);
    let (dw, sw) = window_dups(&gv, &dups, &dupsid);
    let res_v = vce::residuals(
        &gv.ex,
        &gv.r_mat,
        &gv.inv_g,
        &gv.rx,
        &beta_v,
        &dv,
        gv.n_w,
        vce_kind,
        &dw,
        &sw,
        nnmatch,
        has_cluster,
    );
    let v_v = if has_cluster {
        vce::v_cluster(
            &gv.inv_g,
            &gv.rx,
            &res_v,
            &s,
            &fit_groups(&gv, cl),
            kind,
            Some(&gv.rxsqrt),
            None,
        )[(nu, nu)]
    } else {
        vce::v_matrix(&gv.inv_g, &gv.rx, &res_v, &s)[(nu, nu)] // R:301
    };

    // BConst (R:302-305): v = RX' ((eX-c)/h_V)^(o+1); BConst = (Hp * invG v)[nu].
    let uvec = DVector::from_iterator(
        gv.n_w,
        gv.ex
            .iter()
            .map(|&xi| ((xi - c) / h_v).powi((o + 1) as i32)),
    );
    let igv = &gv.inv_g * (gv.rx.transpose() * &uvec);
    let bconst = h_v.powi(nu as i32) * igv[nu];

    VFit { v_v, bconst, s }
}

/// B-fit + assembly half (R functions.R:312-358): given a cached `VFit`, fit the
/// order-`o_b` geometry at `h_B` and assemble `V`/`B`/`R`/`rate`. The B-fit varies
/// with `h_B` so it runs per pilot call.
#[allow(clippy::too_many_arguments)]
pub(crate) fn assemble_from_vfit(
    vf: &VFit,
    y: &[f64],
    x: &[f64],
    t: Option<&[f64]>,
    z: Option<&[f64]>,
    dz: usize,
    c: f64,
    o: usize,
    nu: usize,
    o_b: usize,
    h_v: f64,
    h_b: f64,
    scale: f64,
    vce_kind: Vce,
    nnmatch: usize,
    kernel: Kernel,
    cl: Option<&[f64]>,
    wt: Option<&[f64]>,
) -> BwParts {
    let has_cluster = cl.is_some();
    let kind = vce::cluster_kind(vce_kind);
    let (dups, dupsid) = match vce_kind {
        Vce::NearestNeighbor => vce::compute_dups_dupsid(x),
        _ => (Vec::new(), Vec::new()),
    };
    let s = &vf.s;
    let bconst = vf.bconst;

    let gb = fit_geom(x, c, o_b, h_b, kernel, wt);
    let db = build_design(&gb, y, t, z, dz);
    let beta_b = fit_beta(&gb, &db);
    let mut bwreg = 0.0;
    if scale > 0.0 {
        let (dwb, swb) = window_dups(&gb, &dups, &dupsid);
        let res_b = vce::residuals(
            &gb.ex,
            &gb.r_mat,
            &gb.inv_g,
            &gb.rx,
            &beta_b,
            &db,
            gb.n_w,
            vce_kind,
            &dwb,
            &swb,
            nnmatch,
            has_cluster,
        );
        let v_b = if has_cluster {
            vce::v_cluster(
                &gb.inv_g,
                &gb.rx,
                &res_b,
                s,
                &fit_groups(&gb, cl),
                kind,
                Some(&gb.rxsqrt),
                None,
            )[(o + 1, o + 1)]
        } else {
            vce::v_matrix(&gb.inv_g, &gb.rx, &res_b, s)[(o + 1, o + 1)]
        };
        bwreg = 3.0 * bconst * bconst * v_b; // R:353
    }
    let last = o + 1;
    let sb: f64 = (0..beta_b.ncols())
        .map(|col| s[col] * beta_b[(last, col)])
        .sum();

    // ----- assembly (R:355-358) -----
    let of = o as f64;
    let nuf = nu as f64;
    let b = (2.0 * (of + 1.0 - nuf)).sqrt() * bconst * sb;
    let v = (2.0 * nuf + 1.0) * h_v.powi((2 * nu + 1) as i32) * vf.v_v;
    let r = scale * (2.0 * (of + 1.0 - nuf)) * bwreg;
    let rate = 1.0 / (2.0 * of + 3.0);

    BwParts { v, b, r, rate }
}

/// Port of `rdrobust_bw` (R functions.R:213-361) for the sharp/fuzzy,
/// optional-covariate, nn/hc0-hc3/crv2/crv3 case: the per-side local-polynomial
/// fit engine producing the `V`/`B`/`R`/`rate` ingredients of the MSE/CER
/// bandwidth formula.
///
/// `t = Some(..)` selects fuzzy; `z = Some(..)` adds covariates (full-side n x dZ);
/// `wt = Some(..)` applies user weights. For `vce = nn`, `x` MUST be sorted
/// ascending. Thin wrapper over `compute_vfit` (the V-fit, cacheable per `(o, nu)`)
/// and `assemble_from_vfit` (the per-`h_b` B-fit); the bandwidth selector caches
/// the V-fit across pilot calls instead of calling this wrapper.
///
/// # Arguments
/// * `y` - response, full-side sample (length n).
/// * `x` - running variable (length n). Sorted ascending for `vce = nn`.
/// * `t` - treatment vector for the fuzzy design; `None` selects sharp.
/// * `z` - covariates flattened ROW-MAJOR (n x dz); `None` selects no covariates.
/// * `dz` - covariate-column count, the width of each `z` row.
/// * `c` - cutoff.
/// * `o` - main polynomial order (`p` or `q`).
/// * `nu` - derivative order (`deriv`, `p+1`, or `q+1`).
/// * `o_b` - bias polynomial order; must equal `o + 1`.
/// * `h_v` - V-fit (variance) bandwidth.
/// * `h_b` - B-fit (bias) bandwidth.
/// * `scale` - regularization scale; `> 0` triggers the `BWreg` block, `<= 0` sets `R = 0`.
/// * `vce_kind` - variance estimator (nn, hc0-hc3, crv2, crv3).
/// * `nnmatch` - nearest-neighbor match count (used by `vce = nn`).
/// * `covs_drop_coll` - covariate partialling solver: `true` uses `ginv` (Moore-Penrose), `false` uses Cholesky with a `ginv` fallback (see `compute_s`).
/// * `kernel` - weighting kernel (triangular default, epanechnikov, uniform).
/// * `cl` - cluster ids; `None` selects no clustering.
/// * `wt` - user weights applied multiplicatively to the kernel weights; `None` skips.
///
/// # Panics
/// Panics if `o_b != o + 1` (the structural invariant the R caller always satisfies).
#[allow(clippy::too_many_arguments)]
pub fn rdrobust_bw(
    y: &[f64],
    x: &[f64],
    t: Option<&[f64]>,
    z: Option<&[f64]>,
    dz: usize,
    c: f64,
    o: usize,
    nu: usize,
    o_b: usize,
    h_v: f64,
    h_b: f64,
    scale: f64,
    vce_kind: Vce,
    nnmatch: usize,
    covs_drop_coll: bool,
    kernel: Kernel,
    cl: Option<&[f64]>,
    wt: Option<&[f64]>,
) -> BwParts {
    assert_eq!(o_b, o + 1, "rdrobust_bw invariant: o_B == o + 1");
    let vf = compute_vfit(
        y,
        x,
        t,
        z,
        dz,
        c,
        o,
        nu,
        h_v,
        vce_kind,
        nnmatch,
        covs_drop_coll,
        kernel,
        cl,
        wt,
    );
    assemble_from_vfit(
        &vf, y, x, t, z, dz, c, o, nu, o_b, h_v, h_b, scale, vce_kind, nnmatch, kernel, cl, wt,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rate_depends_on_order() {
        let x: Vec<f64> = (0..200).map(|i| (i as f64) * 0.01).collect();
        let y: Vec<f64> = x.iter().map(|&xi| 1.0 + 0.5 * xi).collect();
        let out = rdrobust_bw(
            &y,
            &x,
            None,
            None,
            0,
            0.0,
            1,
            0,
            2,
            0.8,
            1.0,
            1.0,
            Vce::Hc0,
            3,
            false,
            Kernel::Triangular,
            None,
            None,
        );
        assert!((out.rate - 1.0 / 5.0).abs() < 1e-15);
    }
}
