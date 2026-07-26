//! Bandwidth selector -- port of `rdbwselect` (R rdbwselect.R), the orchestration
//! that runs ~14 `rdrobust_bw` pilot fits and combines their `V`/`B`/`R` into the
//! MSE/CER-optimal `h`/`b` bandwidths.
//!
//! Implemented: sharp/fuzzy, covariates, clusters, user weights, nn/hc0-hc3, all 10
//! selectors + CER, and the prep options (`stdvars` rescaling, mass-point
//! `M`-counts, `bwrestrict` range clamps, `bwcheck` floor). Implements the
//! `vcache`: each side's 3 distinct `(o,nu)` V-fits are computed once and reused
//! across the 7 pilots (`compute_vfit` + `assemble_from_vfit`), bit-identical to
//! the per-call form.
//!
//! Parity gate: continuous `h`/`b` to ~1e-8 vs R, exact on the discrete selector
//! branch.

use crate::bw::{assemble_from_vfit, compute_vfit, VFit};
use crate::{Kernel, Vce};

/// Selected bandwidths: `h`/`b` for the left and right sides (common selectors
/// duplicate the scalar into both sides).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bws {
    /// Main bandwidth `h` (point-estimate window), left side.
    pub h_l: f64,
    /// Main bandwidth `h` (point-estimate window), right side.
    pub h_r: f64,
    /// Bias (pilot) bandwidth `b` (bias-correction window), left side.
    pub b_l: f64,
    /// Bias (pilot) bandwidth `b` (bias-correction window), right side.
    pub b_r: f64,
}

/// Pilot kernel constant `C_c` (rdbwselect.R:265-274).
fn c_c(kernel: Kernel) -> f64 {
    match kernel {
        Kernel::Epanechnikov => 2.34,
        Kernel::Uniform => 1.843,
        Kernel::Triangular => 2.576,
    }
}

/// Sample standard deviation (ddof = 1), R `sd`.
fn sd(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mean = x.iter().sum::<f64>() / n;
    (x.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / (n - 1.0)).sqrt()
}

/// R quantile type 2 on a SORTED slice (rdbwselect.R:117 `quantile(.,type=2)`):
/// j = floor(n p) (1-based); if the fractional part g>0, return x_(j+1); if g==0,
/// average x_(j) and x_(j+1).
fn quantile_type2(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    let np = n as f64 * p;
    let j = np.floor() as usize; // 1-based floor index
    let g = np - j as f64;
    if g > 0.0 {
        sorted[j.min(n - 1)] // x_(j+1), 0-based = sorted[j]
    } else {
        let lo = j.max(1) - 1; // x_(j)
        let hi = j.min(n - 1); // x_(j+1)
        0.5 * (sorted[lo] + sorted[hi])
    }
}

/// Median of three finite values (R `median` of an odd count = the middle).
fn median3(a: f64, b: f64, c: f64) -> f64 {
    let mut v = [a, b, c];
    v.sort_by(|x, y| x.total_cmp(y));
    v[1]
}

/// Select the RD bandwidths `h` (point estimate) and `b` (bias correction) by
/// running the ~14 `rdrobust_bw` pilot fits and combining their `V`/`B`/`R` per the
/// chosen criterion (R rdbwselect.R). `x` must be sorted ascending when `vce = nn`.
///
/// # Arguments
///
/// - `y`, `x`: outcome and running variable (length n; `x` sorted for nn).
/// - `t`: treatment take-up for a fuzzy design; `None` for sharp.
/// - `z`, `dz`: covariates as a flat row-major `n x dz` slice, and the column count
///   (`None` / `0` for no covariates).
/// - `c`: the cutoff.
/// - `p`: local-polynomial order of the point estimate (R default 1).
/// - `q`: order of the bias-correction fit (R default 2, `q > p`).
/// - `deriv`: derivative order of the estimand (R default 0).
/// - `kernel`: weighting kernel (drives the pilot constants).
/// - `vce`: variance estimator used in the pilot `V` (a `cluster` vector remaps it).
/// - `nnmatch`: neighbors per side for nearest-neighbor variance.
/// - `covs_drop_coll`: drop collinear covariate columns (ginv path) vs Cholesky.
/// - `scaleregul`: regularization scale on the bias-bandwidth denominator (R default 1).
/// - `bwselect`: selector name -- `mserd`, `msetwo`, `msesum`, `msecomb1`,
///   `msecomb2`, `cerrd`, `certwo`, `cersum`, `cercomb1`, or `cercomb2`.
/// - `cluster`: cluster ids; their presence remaps `vce` to CR1/CRV2/CRV3 and uses
///   the cluster count in the CER rate.
/// - `weights`: user weights `W` (multiplied into the kernel weights).
/// - `masspoints`: `"adjust"` uses the unique-point count `M` for pilot rates and
///   may auto-set `bwcheck`; `"off"` disables both.
/// - `stdvars`: standardize `x`/`y`/`c` before selection, then rescale the chosen
///   bandwidths by the `x` standard deviation.
/// - `bwrestrict`: clamp each pilot stage to the side's support range.
/// - `bwcheck`: floor the pilot and final bandwidths at the k-th nearest unique
///   distance (`None` to disable).
#[allow(clippy::too_many_arguments)]
pub fn rdbwselect(
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
    vce: Vce,
    nnmatch: usize,
    covs_drop_coll: bool,
    scaleregul: f64,
    bwselect: &str,
    cluster: Option<&[f64]>,
    weights: Option<&[f64]>,
    masspoints: &str,
    stdvars: bool,
    bwrestrict: bool,
    bwcheck: Option<usize>,
) -> Bws {
    let n = y.len();

    // stdvars (rdrobust.R:146-153): run selection on standardized x/y/c, then
    // multiply the chosen bandwidths by x_sd at the end. No-op when stdvars=false.
    let (x_sd, y_sd) = if stdvars { (sd(x), sd(y)) } else { (1.0, 1.0) };
    let xv: Vec<f64> = x.iter().map(|v| v / x_sd).collect();
    let yv: Vec<f64> = y.iter().map(|v| v / y_sd).collect();
    let x = xv.as_slice();
    let y = yv.as_slice();
    let c = c / x_sd;

    // BWp = min(sd(x), IQR_type2(x)/1.349); c_bw = C_c * BWp * N^(-1/5) (or M^-1/5).
    let mut xs = x.to_vec();
    xs.sort_by(|a, b| a.total_cmp(b));
    let x_iq = quantile_type2(&xs, 0.75) - quantile_type2(&xs, 0.25);
    let bwp = sd(x).min(x_iq / 1.349);

    // split L (x<c) / R (x>=c), preserving (sorted) order; z flat row-major (n x dz)
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
    for i in 0..n {
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
    let cll_opt = cluster.map(|_| cll.as_slice());
    let clr_opt = cluster.map(|_| clr.as_slice());
    let fwl_opt = weights.map(|_| fwl.as_slice());
    let fwr_opt = weights.map(|_| fwr.as_slice());

    // distinct-cluster counts per side -> the CER bandwidth factor (rdrobust.R:477)
    let n_unique = |v: &[f64]| -> usize {
        let mut s = v.to_vec();
        s.sort_by(|a, b| a.total_cmp(b));
        s.dedup();
        s.len()
    };
    let g_l = cluster.map(|_| n_unique(&cll));
    let g_r = cluster.map(|_| n_unique(&clr));
    let range_l = (c - xs[0]).abs();
    let range_r = (c - xs[n - 1]).abs();

    // ----- masspoints / bwrestrict / bwcheck prep (spec 10; rdrobust.R:361-406) -----
    let nl = xl.len();
    let nr = xr.len();
    let mp_on = masspoints == "check" || masspoints == "adjust";
    let (m_l, m_r) = if mp_on || bwcheck.is_some() {
        (n_unique(&xl), n_unique(&xr))
    } else {
        (nl, nr)
    };
    let m = m_l + m_r;
    // auto-set bwcheck=10 when adjust detects mass points (>=20% duplicates)
    let mut bwcheck_eff = bwcheck;
    if mp_on {
        let mass_l = 1.0 - m_l as f64 / nl as f64;
        let mass_r = 1.0 - m_r as f64 / nr as f64;
        if (mass_l >= 0.2 || mass_r >= 0.2) && masspoints == "adjust" && bwcheck.is_none() {
            bwcheck_eff = Some(10);
        }
    }
    // pilot bandwidth: M^(-1/5) under masspoints=adjust, else N^(-1/5)
    let pilot_pow = if masspoints == "adjust" {
        m as f64
    } else {
        n as f64
    };
    let mut c_bw = c_c(kernel) * bwp * pilot_pow.powf(-1.0 / 5.0);
    // bwrestrict clamp bounds (reused for all per-stage clamps)
    let bw_max_l = range_l;
    let bw_max_r = range_r;
    let bw_max = bw_max_l.max(bw_max_r);
    if bwrestrict {
        c_bw = c_bw.min(bw_max);
    }
    // bwcheck floor: distance to the k-th nearest unique point per side
    let (bw_min_l, bw_min_r) = match bwcheck_eff {
        Some(bc) => {
            let mut ul = xl.clone();
            ul.sort_by(|a, b| a.total_cmp(b));
            ul.dedup();
            let mut ur = xr.clone();
            ur.sort_by(|a, b| a.total_cmp(b));
            ur.dedup();
            // X_uniq_l descending (closest to c last); X_uniq_r ascending
            let bcl = bc.min(m_l);
            let bcr = bc.min(m_r);
            let lhs = (ul[ul.len() - bcl] - c).abs(); // descending[bcl-1] = ascending[len-bcl]
            let rhs = (ur[bcr - 1] - c).abs();
            (lhs, rhs)
        }
        None => (0.0, 0.0),
    };
    if bwcheck_eff.is_some() {
        c_bw = c_bw.max(bw_min_l).max(bw_min_r);
    }
    // per-stage clamp helpers (no-ops when the option is off)
    let restrict = |v: f64, bmax: f64| if bwrestrict { v.min(bmax) } else { v };
    let floor_side = |v: f64, bmin: f64| {
        if bwcheck_eff.is_some() {
            v.max(bmin)
        } else {
            v
        }
    };
    let floor_comb = |v: f64| {
        if bwcheck_eff.is_some() {
            v.max(bw_min_l).max(bw_min_r)
        } else {
            v
        }
    };

    // vcache (functions.R:213-231): the V-fit depends only on (o,nu) since h_V=c_bw
    // is fixed across pilots. The 7 pilot calls per side use only 3 distinct (o,nu)
    // V-fits -- precompute each once, then run the B-fit per pilot via h_B.
    let mk_vfit = |yy: &[f64], xx: &[f64], tt, zz, cc, ww, o, nu| {
        compute_vfit(
            yy,
            xx,
            tt,
            zz,
            dz,
            c,
            o,
            nu,
            c_bw,
            vce,
            nnmatch,
            covs_drop_coll,
            kernel,
            cc,
            ww,
        )
    };
    let vd_l = mk_vfit(&yl, &xl, tl_opt, zl_opt, cll_opt, fwl_opt, q + 1, q + 1);
    let vb_l = mk_vfit(&yl, &xl, tl_opt, zl_opt, cll_opt, fwl_opt, q, p + 1);
    let vh_l = mk_vfit(&yl, &xl, tl_opt, zl_opt, cll_opt, fwl_opt, p, deriv);
    let vd_r = mk_vfit(&yr, &xr, tr_opt, zr_opt, clr_opt, fwr_opt, q + 1, q + 1);
    let vb_r = mk_vfit(&yr, &xr, tr_opt, zr_opt, clr_opt, fwr_opt, q, p + 1);
    let vh_r = mk_vfit(&yr, &xr, tr_opt, zr_opt, clr_opt, fwr_opt, p, deriv);

    // one pilot B-fit + assembly against a cached V-fit (h_V = c_bw fixed)
    let asm = |vf: &VFit, yy: &[f64], xx: &[f64], tt, zz, cc, ww, o, nu, o_b, h_b, scale| {
        assemble_from_vfit(
            vf, yy, xx, tt, zz, dz, c, o, nu, o_b, c_bw, h_b, scale, vce, nnmatch, kernel, cc, ww,
        )
    };

    // ----- d-stage (o=q+1, scale=0, no regularization) -----
    let cdl = asm(
        &vd_l,
        &yl,
        &xl,
        tl_opt,
        zl_opt,
        cll_opt,
        fwl_opt,
        q + 1,
        q + 1,
        q + 2,
        range_l,
        0.0,
    );
    let cdr = asm(
        &vd_r,
        &yr,
        &xr,
        tr_opt,
        zr_opt,
        clr_opt,
        fwr_opt,
        q + 1,
        q + 1,
        q + 2,
        range_r,
        0.0,
    );
    let rate_d = cdl.rate;
    let d_bw_l = (cdl.v / cdl.b.powi(2)).powf(rate_d);
    let d_bw_r = (cdr.v / cdr.b.powi(2)).powf(rate_d);
    let d_bw_s = ((cdl.v + cdr.v) / (cdr.b + cdl.b).powi(2)).powf(rate_d);
    let d_bw_d = ((cdl.v + cdr.v) / (cdr.b - cdl.b).powi(2)).powf(rate_d);
    // d-stage clamps: bwrestrict cap then bwcheck floor (TWO per-side, SUM/RD combined)
    let d_bw_l = floor_side(restrict(d_bw_l, bw_max_l), bw_min_l);
    let d_bw_r = floor_side(restrict(d_bw_r, bw_max_r), bw_min_r);
    let d_bw_s = floor_comb(restrict(d_bw_s, bw_max));
    let d_bw_d = floor_comb(restrict(d_bw_d, bw_max));

    // helper for the regularized b/h combine (RD uses Br-Bl, SUM uses Br+Bl)
    let reg_two =
        |cv: f64, cb: f64, cr: f64, rate: f64| (cv / (cb.powi(2) + scaleregul * cr)).powf(rate);
    let reg_comb = |vl: f64, vr: f64, b_combined: f64, rl: f64, rr: f64, rate: f64| {
        ((vl + vr) / (b_combined.powi(2) + scaleregul * (rl + rr))).powf(rate)
    };

    // ----- TWO (per-side h_B = d_bw_l/_r) -----
    let cbl = asm(
        &vb_l,
        &yl,
        &xl,
        tl_opt,
        zl_opt,
        cll_opt,
        fwl_opt,
        q,
        p + 1,
        q + 1,
        d_bw_l,
        scaleregul,
    );
    let cbr = asm(
        &vb_r,
        &yr,
        &xr,
        tr_opt,
        zr_opt,
        clr_opt,
        fwr_opt,
        q,
        p + 1,
        q + 1,
        d_bw_r,
        scaleregul,
    );
    let rate_b = cbl.rate;
    let b_bw_l = restrict(reg_two(cbl.v, cbl.b, cbl.r, rate_b), bw_max_l);
    let b_bw_r = restrict(reg_two(cbr.v, cbr.b, cbr.r, rate_b), bw_max_r);
    let chl = asm(
        &vh_l, &yl, &xl, tl_opt, zl_opt, cll_opt, fwl_opt, p, deriv, q, b_bw_l, scaleregul,
    );
    let chr = asm(
        &vh_r, &yr, &xr, tr_opt, zr_opt, clr_opt, fwr_opt, p, deriv, q, b_bw_r, scaleregul,
    );
    let rate_h = chl.rate;
    let h_bw_l = restrict(reg_two(chl.v, chl.b, chl.r, rate_h), bw_max_l);
    let h_bw_r = restrict(reg_two(chr.v, chr.b, chr.r, rate_h), bw_max_r);

    // ----- SUM (shared h_B = d_bw_s, Br+Bl) -----
    let cbl_s = asm(
        &vb_l,
        &yl,
        &xl,
        tl_opt,
        zl_opt,
        cll_opt,
        fwl_opt,
        q,
        p + 1,
        q + 1,
        d_bw_s,
        scaleregul,
    );
    let cbr_s = asm(
        &vb_r,
        &yr,
        &xr,
        tr_opt,
        zr_opt,
        clr_opt,
        fwr_opt,
        q,
        p + 1,
        q + 1,
        d_bw_s,
        scaleregul,
    );
    let b_bw_s = restrict(
        reg_comb(
            cbl_s.v,
            cbr_s.v,
            cbr_s.b + cbl_s.b,
            cbl_s.r,
            cbr_s.r,
            rate_b,
        ),
        bw_max,
    );
    let chl_s = asm(
        &vh_l, &yl, &xl, tl_opt, zl_opt, cll_opt, fwl_opt, p, deriv, q, b_bw_s, scaleregul,
    );
    let chr_s = asm(
        &vh_r, &yr, &xr, tr_opt, zr_opt, clr_opt, fwr_opt, p, deriv, q, b_bw_s, scaleregul,
    );
    let h_bw_s = restrict(
        reg_comb(
            chl_s.v,
            chr_s.v,
            chr_s.b + chl_s.b,
            chl_s.r,
            chr_s.r,
            rate_h,
        ),
        bw_max,
    );

    // ----- RD (shared h_B = d_bw_d, Br-Bl) -----
    let cbl_d = asm(
        &vb_l,
        &yl,
        &xl,
        tl_opt,
        zl_opt,
        cll_opt,
        fwl_opt,
        q,
        p + 1,
        q + 1,
        d_bw_d,
        scaleregul,
    );
    let cbr_d = asm(
        &vb_r,
        &yr,
        &xr,
        tr_opt,
        zr_opt,
        clr_opt,
        fwr_opt,
        q,
        p + 1,
        q + 1,
        d_bw_d,
        scaleregul,
    );
    let b_bw_d = restrict(
        reg_comb(
            cbl_d.v,
            cbr_d.v,
            cbr_d.b - cbl_d.b,
            cbl_d.r,
            cbr_d.r,
            rate_b,
        ),
        bw_max,
    );
    let chl_d = asm(
        &vh_l, &yl, &xl, tl_opt, zl_opt, cll_opt, fwl_opt, p, deriv, q, b_bw_d, scaleregul,
    );
    let chr_d = asm(
        &vh_r, &yr, &xr, tr_opt, zr_opt, clr_opt, fwr_opt, p, deriv, q, b_bw_d, scaleregul,
    );
    let h_bw_d = restrict(
        reg_comb(
            chl_d.v,
            chr_d.v,
            chr_d.b - chl_d.b,
            chl_d.r,
            chr_d.r,
            rate_h,
        ),
        bw_max,
    );

    // ----- combine + CER -----
    let (h_rd, b_rd) = (h_bw_d, b_bw_d);
    let (h_sum, b_sum) = (h_bw_s, b_bw_s);
    let h_c1 = h_rd.min(h_sum);
    let b_c1 = b_rd.min(b_sum);
    let h_c2_l = median3(h_rd, h_sum, h_bw_l);
    let h_c2_r = median3(h_rd, h_sum, h_bw_r);
    let b_c2_l = median3(b_rd, b_sum, b_bw_l);
    let b_c2_r = median3(b_rd, b_sum, b_bw_r);
    let pf = p as f64;
    // CER rate factor: cluster count (g_l+g_r) replaces N when clustered (R:476-477)
    let cer_base = match (g_l, g_r) {
        (Some(gl), Some(gr)) => (gl + gr) as f64,
        _ => n as f64,
    };
    let cer_h = cer_base.powf(-(pf / ((3.0 + pf) * (3.0 + 2.0 * pf))));
    let cer_b = 1.0;

    let common = |h: f64, b: f64| Bws {
        h_l: h,
        h_r: h,
        b_l: b,
        b_r: b,
    };
    let bws = match bwselect {
        "mserd" | "" => common(h_rd, b_rd),
        "msesum" => common(h_sum, b_sum),
        "msetwo" => Bws {
            h_l: h_bw_l,
            h_r: h_bw_r,
            b_l: b_bw_l,
            b_r: b_bw_r,
        },
        "msecomb1" => common(h_c1, b_c1),
        "msecomb2" => Bws {
            h_l: h_c2_l,
            h_r: h_c2_r,
            b_l: b_c2_l,
            b_r: b_c2_r,
        },
        "cerrd" => common(h_rd * cer_h, b_rd * cer_b),
        "cersum" => common(h_sum * cer_h, b_sum * cer_b),
        "certwo" => Bws {
            h_l: h_bw_l * cer_h,
            h_r: h_bw_r * cer_h,
            b_l: b_bw_l * cer_b,
            b_r: b_bw_r * cer_b,
        },
        "cercomb1" => common(h_c1 * cer_h, b_c1 * cer_b),
        "cercomb2" => Bws {
            h_l: h_c2_l * cer_h,
            h_r: h_c2_r * cer_h,
            b_l: b_c2_l * cer_b,
            b_r: b_c2_r * cer_b,
        },
        other => panic!("unknown bwselect {other}"),
    };
    // de-standardize: bandwidths return on the original x scale (rdrobust.R:467-469)
    Bws {
        h_l: bws.h_l * x_sd,
        h_r: bws.h_r * x_sd,
        b_l: bws.b_l * x_sd,
        b_r: bws.b_r * x_sd,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantile_type2_even_n_averages() {
        // n=4, p=0.5: np=2, g=0 -> avg of x_(2),x_(3)
        let s = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(quantile_type2(&s, 0.5), 2.5);
    }

    #[test]
    fn median3_picks_middle() {
        assert_eq!(median3(3.0, 1.0, 2.0), 2.0);
    }
}
