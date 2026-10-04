//! Shared variance machinery: residuals (nn / hc0 / hc1) and the sandwich meat,
//! used by both the bandwidth engine (`bw.rs`) and the main estimator
//! (`rdrobust.rs`). Ports of `rdrobust_res` (functions.R:146-210) and the
//! non-cluster branch of `rdrobust_vce` (functions.R:380-392).

use nalgebra::{DMatrix, DVector};

use crate::Vce;

/// Exact integer factorial as f64 (R `factorial(nu)`; factorial(0)=1).
pub(crate) fn factorial(n: usize) -> f64 {
    (1..=n).map(|k| k as f64).product()
}

/// Run-length structure of a SORTED running variable (R `dups`/`dupsid`,
/// rdrobust.R:207-214). `dups[i]` = tie-block size of `x[i]`; `dupsid[i]` = its
/// 1-based rank within that block.
pub(crate) fn compute_dups_dupsid(x: &[f64]) -> (Vec<usize>, Vec<usize>) {
    let n = x.len();
    let mut dups = vec![0usize; n];
    let mut dupsid = vec![0usize; n];
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && x[j] == x[i] {
            j += 1;
        }
        let len = j - i;
        for k in i..j {
            dups[k] = len;
            dupsid[k] = k - i + 1;
        }
        i = j;
    }
    (dups, dupsid)
}

/// Nearest-neighbor leave-one-out residuals for every column of `d` (R
/// `rdrobust_res` nn branch, functions.R:153-187). `x` is the windowed SORTED
/// running variable; `dups`/`dupsid` are the window-subset of the full-side
/// run-length structure (R subsets `dups[ind]`, it does not recompute).
pub(crate) fn nn_residuals(
    x: &[f64],
    d: &DMatrix<f64>,
    dups: &[usize],
    dupsid: &[usize],
    matches: usize,
) -> DMatrix<f64> {
    let n = x.len();
    let ncols = d.ncols();
    let nn_tol_eps = f64::EPSILON.sqrt();
    // An empty window (every point outside the kernel support, e.g. a pilot
    // bandwidth of 0 when one side collapses onto the cutoff) has no residuals.
    // `n - 1` would underflow; the loop below never runs for n == 0 anyway.
    let target = matches.min(n.saturating_sub(1)) as isize;
    let mut res = DMatrix::zeros(n, ncols);

    for pos in 1..=n {
        let p = pos as isize;
        let mut rpos = dups[pos - 1] as isize - dupsid[pos - 1] as isize;
        let mut lpos = dupsid[pos - 1] as isize - 1;
        while lpos + rpos < target {
            if p - lpos - 1 <= 0 {
                rpos += dups[(p + rpos) as usize] as isize;
            } else if p + rpos + 1 > n as isize {
                lpos += dups[(p - lpos - 2) as usize] as isize;
            } else {
                let dleft = x[pos - 1] - x[(p - lpos - 2) as usize];
                let dright = x[(p + rpos) as usize] - x[pos - 1];
                let nn_tol = dleft.max(dright) * nn_tol_eps;
                if dleft - dright > nn_tol {
                    rpos += dups[(p + rpos) as usize] as isize;
                } else if dright - dleft > nn_tol {
                    lpos += dups[(p - lpos - 2) as usize] as isize;
                } else {
                    rpos += dups[(p + rpos) as usize] as isize;
                    lpos += dups[(p - lpos - 2) as usize] as isize;
                }
            }
        }
        let lo0 = (p - lpos - 1) as usize;
        let hi0 = (p + rpos - 1) as usize;
        let ji = (lpos + rpos) as f64;
        for col in 0..ncols {
            let mut sum = 0.0;
            for k in lo0..=hi0 {
                sum += d[(k, col)];
            }
            let d_j = sum - d[(pos - 1, col)];
            res[(pos - 1, col)] = (ji / (ji + 1.0)).sqrt() * (d[(pos - 1, col)] - d_j / ji);
        }
    }
    res
}

/// Residual matrix (n_w x ncols) for the given variance estimator.
/// hc0: `D - R beta`; hc1: scaled by `sqrt(n/(n-dof))`, dof = `r_mat.ncols()`
/// (the poly count R passes as `d`); nn: leave-one-out neighbor residuals.
/// `has_cluster` suppresses the hc1 dof scaling (R functions.R:202 — clustering
/// applies its own df factor in the meat, so doubling is avoided); the cluster
/// variants (crv2/crv3) return the RAW residual (functions.R:189-199), the cluster-level
/// leverage adjustment being deferred to the meat.
#[allow(clippy::too_many_arguments)]
pub(crate) fn residuals(
    ex: &[f64],
    r_mat: &DMatrix<f64>,
    inv_g: &DMatrix<f64>,
    rx: &DMatrix<f64>,
    beta: &DMatrix<f64>,
    d: &DMatrix<f64>,
    n_w: usize,
    vce: Vce,
    dups_w: &[usize],
    dupsid_w: &[usize],
    nnmatch: usize,
    has_cluster: bool,
) -> DMatrix<f64> {
    // Clustering remaps every vce to a raw-residual path (rdrobust.R:318-344):
    // nn/hc0/hc1/cr1 -> hc1 (w=1 -> raw), hc2/hc3 -> crv2/crv3 (raw). The cluster
    // leverage adjustment lives in the meat, not the residual.
    if has_cluster {
        return d - r_mat * beta;
    }
    // crv2/crv3 use raw residuals; the hat-matrix adjustment is cluster-wise.
    if vce.cluster_order().is_some() {
        return d - r_mat * beta;
    }
    match vce.hc_order() {
        Some(0) => d - r_mat * beta,
        Some(1) => {
            let dof = r_mat.ncols() as f64;
            let w = (n_w as f64 / (n_w as f64 - dof)).sqrt();
            (d - r_mat * beta) * w
        }
        // hc2/hc3 (R functions.R:203-204): w = sqrt(1/max(1-hii,1e-8)) / (1/...).
        // hii = rowSums((R invG) * (R*W)) -- the hat-matrix diagonal.
        Some(2) | Some(3) => {
            let hc3 = vce.hc_order() == Some(3);
            let rig = r_mat * inv_g; // n_w x (o+1)
            let mut res = d - r_mat * beta;
            let ncols = res.ncols();
            for k in 0..n_w {
                let hii: f64 = (0..r_mat.ncols()).map(|j| rig[(k, j)] * rx[(k, j)]).sum();
                let denom = (1.0 - hii).max(1e-8);
                let w = if hc3 {
                    1.0 / denom
                } else {
                    (1.0 / denom).sqrt()
                };
                for col in 0..ncols {
                    res[(k, col)] *= w;
                }
            }
            res
        }
        // Only NearestNeighbor reaches here: Hc0-3 have hc_order Some(_), and the
        // Cr1-3 variants are handled by the earlier cluster_order check.
        _ => match vce {
            Vce::NearestNeighbor => nn_residuals(ex, d, dups_w, dupsid_w, nnmatch),
            _ => unreachable!("non-NearestNeighbor vce {vce:?} handled above"),
        },
    }
}

/// Covariate FWL partialling inputs for one fit (R functions.R:251-256 /
/// rdrobust.R:659-667). `rx` = R*W (row-scaled design), `ew` = its row weights,
/// `d` = `[Y, T?, Z..]`. Returns `(ZWZ, ZWY)` = (dz x dz, dz x (1+dt)); the main
/// estimator SUMS these across sides before inverting, the bw engine does not.
pub(crate) fn partialling(
    rx: &DMatrix<f64>,
    inv_g: &DMatrix<f64>,
    d: &DMatrix<f64>,
    ew: &[f64],
    dt: usize,
    dz: usize,
) -> (DMatrix<f64>, DMatrix<f64>) {
    let cz = 1 + dt;
    let mut ezw = d.columns(cz, dz).into_owned();
    for i in 0..d.nrows() {
        for j in 0..dz {
            ezw[(i, j)] *= ew[i];
        }
    }
    let u = rx.transpose() * d; // crossprod(R*W, D)
    let zwd = ezw.transpose() * d; // crossprod(eZ*W, D)
    let u_cz = u.columns(cz, dz).into_owned();
    let uigu = u_cz.transpose() * (inv_g * &u);
    let zwz = zwd.columns(cz, dz) - uigu.columns(cz, dz);
    let zwy = zwd.columns(0, 1 + dt) - uigu.columns(0, 1 + dt);
    (zwz, zwy)
}

/// `gamma = inv(ZWZ) ZWY`: chol2inv when SPD, ginv when `covs_drop_coll`.
pub(crate) fn gamma_from(
    zwz: &DMatrix<f64>,
    zwy: &DMatrix<f64>,
    covs_drop_coll: bool,
) -> DMatrix<f64> {
    if covs_drop_coll {
        crate::linalg::ginv(zwz) * zwy
    } else {
        match zwz.clone().cholesky() {
            Some(ch) => ch.inverse() * zwy,
            None => crate::linalg::ginv(zwz) * zwy,
        }
    }
}

/// Sandwich `invG * M * invG` (full matrix), meat `M = crossprod(c(res %*% s) * rx)`
/// (R rdrobust_vce non-cluster branch; the d>0 form reduces to d==0 when s=[1]).
pub(crate) fn v_matrix(
    inv_g: &DMatrix<f64>,
    rx: &DMatrix<f64>,
    res: &DMatrix<f64>,
    s: &DVector<f64>,
) -> DMatrix<f64> {
    let r_comb = res * s; // n-vector
    let mut rsc = rx.clone();
    let nrow = rx.nrows();
    let ncol = rx.ncols();
    for i in 0..nrow {
        let ri = r_comb[i];
        for j in 0..ncol {
            rsc[(i, j)] *= ri;
        }
    }
    let m = rsc.transpose() * &rsc;
    inv_g * (m * inv_g)
}

/// Which cluster-robust meat to build (mirrors R's `vce` -> branch selection).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ClusterKind {
    /// Liang-Zeger CR1 with the rdrobust small-sample df factor.
    Cr1,
    /// Bell-McCaffrey CRV2 (eigendecomposition half-inverse).
    Crv2,
    /// Pustejovsky-Tipton CRV3 (Sherman-Morrison-Woodbury).
    Crv3,
}

/// Map a `Vce` to its cluster meat (only meaningful when a cluster vector is
/// present). R remaps the user's vce when a cluster is supplied (rdrobust.R:
/// 318-344): `nn`/`hc0`/`hc1`/`cr1` -> CR1, `hc2`/`cr2` -> CRV2, `hc3`/`cr3` ->
/// CRV3. The residual is RAW in every clustered branch (hc1 is suppressed,
/// crv2/crv3 use raw) -- see `residuals`.
pub(crate) fn cluster_kind(vce: Vce) -> ClusterKind {
    match (vce.hc_order(), vce.cluster_order()) {
        (Some(2), _) | (_, Some(2)) => ClusterKind::Crv2,
        (Some(3), _) | (_, Some(3)) => ClusterKind::Crv3,
        _ => ClusterKind::Cr1,
    }
}

/// Group row indices by cluster id (R `.rdrobust_cluster_idx`, functions.R:96-108,
/// numeric path only). Returns groups in ascending-cluster-value order; each is
/// the original 0-based row indices in that cluster, in stable (sorted) order.
/// Group order and intra-group order do not affect the meat (sum of rank-1 terms).
pub(crate) fn cluster_idx(c: &[f64]) -> Vec<Vec<usize>> {
    let n = c.len();
    if n == 0 {
        return Vec::new();
    }
    let mut ord: Vec<usize> = (0..n).collect();
    ord.sort_by(|&a, &b| c[a].total_cmp(&c[b]));
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut start = 0;
    while start < n {
        let v = c[ord[start]];
        let mut end = start + 1;
        while end < n && c[ord[end]] == v {
            end += 1;
        }
        groups.push(ord[start..end].to_vec());
        start = end;
    }
    groups
}

/// `solve(m)` with R's fallback: LU inverse, else Moore-Penrose `ginv`.
fn solve_or_ginv(m: &DMatrix<f64>) -> DMatrix<f64> {
    match m.clone().try_inverse() {
        Some(inv) => inv,
        None => crate::linalg::ginv(m),
    }
}

/// Combine the `s`-weighted residual columns of cluster `idx` into a single
/// score vector via `f(l)` = column-`l` k-score, summed as `Σ_l s[l] f(l)`.
fn cluster_score(
    s: &DVector<f64>,
    ncols: usize,
    mut f: impl FnMut(usize) -> DVector<f64>,
) -> DVector<f64> {
    let mut sv = f(0) * s[0];
    for l in 1..ncols {
        sv += f(l) * s[l];
    }
    sv
}

/// Cluster-robust MEAT `w * M` (k x k). Mirrors the cluster branches of
/// `rdrobust_vce` (functions.R:363-592). The caller applies the bread
/// (`inv_g * (w*M) * inv_g`). `rx` (n x k) is the weighted score design (R*W);
/// `res` is n x (1+d); `s` is length (1+d). For CRV2/CRV3, `sqrt_rx` (R*sqrt(W),
/// n x k) and `inv_g` (k x k bread of that design) are required. `k_override`
/// sets the CR1 df dimension (else `rx.ncols()`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn vce_meat_cluster(
    s: &DVector<f64>,
    rx: &DMatrix<f64>,
    res: &DMatrix<f64>,
    groups: &[Vec<usize>],
    kind: ClusterKind,
    inv_g: Option<&DMatrix<f64>>,
    sqrt_rx: Option<&DMatrix<f64>>,
    k_override: Option<usize>,
) -> DMatrix<f64> {
    let k = rx.ncols();
    let ncols = res.ncols();
    let mut m = DMatrix::zeros(k, k);

    match kind {
        ClusterKind::Cr1 => {
            // M = Σ_g sv_g sv_g'; sv_g = Σ_l s[l] (rx_g' res_g[,l]).
            for idx in groups {
                let xi = rows(rx, idx); // n_g x k
                let ri = rows(res, idx); // n_g x ncols
                let sv = cluster_score(s, ncols, |l| {
                    (xi.transpose() * ri.column(l)).column(0).into_owned()
                });
                m += &sv * sv.transpose();
            }
            let n = res.nrows() as f64;
            let k_df = k_override.unwrap_or(k) as f64;
            let g = groups.len() as f64;
            let w = ((n - 1.0) / (n - k_df)) * (g / (g - 1.0));
            m * w
        }
        ClusterKind::Crv3 => {
            // SMW: score_g = u_g + L_g (G - L_g)^{-1} u_g.  w = 1.
            let sqrt_rx = sqrt_rx.expect("crv3 requires sqrt_rx");
            let inv_g = inv_g.expect("crv3 requires inv_g");
            let g_gram = solve_or_ginv(inv_g);
            for idx in groups {
                let srx_g = rows(sqrt_rx, idx); // n_g x k
                let sw_g = srx_g.column(0).into_owned(); // sqrt(eW)
                let ri = rows(res, idx);
                let l_g = srx_g.transpose() * &srx_g; // k x k
                let m_g = solve_or_ginv(&(&g_gram - &l_g));
                let sv = cluster_score(s, ncols, |l| {
                    let e = ri.column(l);
                    let swe =
                        DVector::from_iterator(idx.len(), (0..idx.len()).map(|i| sw_g[i] * e[i]));
                    let u_g = srx_g.transpose() * swe; // k x 1
                    &u_g + &l_g * (&m_g * &u_g)
                });
                m += &sv * sv.transpose();
            }
            m
        }
        ClusterKind::Crv2 => {
            // Bell-McCaffrey: score_g = u_g + L_g R' V diag(c) V' R u_g.  w = 1.
            let sqrt_rx = sqrt_rx.expect("crv2 requires sqrt_rx");
            let inv_g = inv_g.expect("crv2 requires inv_g");
            // C_half = chol(invG) (R upper); nalgebra gives lower L (L L' = invG),
            // so C_half = L', tC_half = L. crv2_ok = chol succeeded.
            let c_half_lower = nalgebra::Cholesky::new(inv_g.clone());
            for idx in groups {
                let srx_g = rows(sqrt_rx, idx);
                let sw_g = srx_g.column(0).into_owned();
                let ri = rows(res, idx);
                let l_g = srx_g.transpose() * &srx_g; // k x k

                let u_of = |l: usize| -> DVector<f64> {
                    let e = ri.column(l);
                    let swe =
                        DVector::from_iterator(idx.len(), (0..idx.len()).map(|i| sw_g[i] * e[i]));
                    srx_g.transpose() * swe
                };

                let sv = match &c_half_lower {
                    None => cluster_score(s, ncols, u_of), // chol failed -> raw score
                    Some(ch) => {
                        let lower = ch.l(); // tC_half
                        let upper = lower.transpose(); // C_half
                        let f_sq = &upper * &l_g * &lower; // C_half L_g tC_half (symmetric)
                        let eig = nalgebra::SymmetricEigen::new(0.5 * (&f_sq + f_sq.transpose()));
                        let kk = f_sq.nrows();
                        let mut c_coef = DVector::zeros(kk);
                        for j in 0..kk {
                            let sigma2 = eig.eigenvalues[j].max(0.0);
                            c_coef[j] = if sigma2 < 1e-14 {
                                0.0
                            } else {
                                (1.0 / (1.0 - sigma2).max(1e-8).sqrt() - 1.0) / sigma2
                            };
                        }
                        let v_g = &eig.eigenvectors;
                        let lt_chalf = &l_g * &lower; // L_g tC_half
                        cluster_score(s, ncols, |l| {
                            let u_g = u_of(l);
                            let ru_g = &upper * &u_g; // C_half u_g
                                                      // adj = V diag(c) V' (R u_g)
                            let vt_ru = v_g.transpose() * &ru_g;
                            let scaled =
                                DVector::from_iterator(kk, (0..kk).map(|j| c_coef[j] * vt_ru[j]));
                            let adj = v_g * scaled;
                            &u_g + &lt_chalf * adj
                        })
                    }
                };
                m += &sv * sv.transpose();
            }
            m
        }
    }
}

/// Decoupled cluster-robust meat for the bias-corrected (h != b) estimator
/// (R `.rdrobust_vce_qq_cluster`, functions.R:607-683). Returns `M` (k x k,
/// k = ncol(q)); the caller wraps `inv_g_p * M * inv_g_p`. Uses the mixed-bandwidth
/// score `Q` with leverage from the q-regression at bandwidth b.
#[allow(clippy::too_many_arguments)]
pub(crate) fn vce_meat_qq_cluster(
    q: &DMatrix<f64>,
    r_q: &DMatrix<f64>,
    w_b: &[f64],
    inv_g_q: &DMatrix<f64>,
    res: &DMatrix<f64>,
    groups: &[Vec<usize>],
    crv2: bool,
    s: &DVector<f64>,
) -> DMatrix<f64> {
    let k = q.ncols();
    let ncols = res.ncols();
    let mut m = DMatrix::zeros(k, k);
    let g_q = solve_or_ginv(inv_g_q);

    for idx in groups {
        let q_g = rows(q, idx); // n_g x k
        let r_g = rows(r_q, idx); // n_g x k_R
        let wb_g: Vec<f64> = idx.iter().map(|&i| w_b[i]).collect();
        let sqrt_wb: Vec<f64> = wb_g.iter().map(|v| v.sqrt()).collect();
        let mut sr_g = r_g.clone();
        for i in 0..idx.len() {
            for j in 0..sr_g.ncols() {
                sr_g[(i, j)] *= sqrt_wb[i];
            }
        }
        let l_g = sr_g.transpose() * &sr_g; // k_R x k_R
        let p_g = q_g.transpose() * &r_g; // k x k_R
        let ri = rows(res, idx);

        // T2: per-cluster operator depends only on L_g/G_q -- compute once.
        let mut use_simple = false;
        let mut lambda: Option<DMatrix<f64>> = None;
        let mut m_g: Option<DMatrix<f64>> = None;
        if crv2 {
            let eig_l = nalgebra::SymmetricEigen::new(0.5 * (&l_g + l_g.transpose()));
            let kr = l_g.nrows();
            let sigma: Vec<f64> = (0..kr)
                .map(|j| eig_l.eigenvalues[j].max(0.0).sqrt())
                .collect();
            let max_sig = sigma.iter().cloned().fold(0.0_f64, f64::max);
            let keep: Vec<usize> = (0..kr).filter(|&j| sigma[j] > max_sig * 1e-10).collect();
            let r = keep.len();
            if r == 0 {
                use_simple = true;
            } else {
                // V_r (kr x r), Dsig (r x r)
                let mut v_r = DMatrix::zeros(kr, r);
                for (col, &j) in keep.iter().enumerate() {
                    v_r.set_column(col, &eig_l.eigenvectors.column(j));
                }
                let sig: Vec<f64> = keep.iter().map(|&j| sigma[j]).collect();
                let dsig = DMatrix::from_diagonal(&DVector::from_vec(sig.clone()));
                let mut t_mat = &dsig * (v_r.transpose() * (inv_g_q * &v_r)) * &dsig;
                t_mat = 0.5 * (&t_mat + t_mat.transpose());
                let eig_t = nalgebra::SymmetricEigen::new(t_mat);
                let mut gamma = DVector::zeros(r);
                for j in 0..r {
                    let tau = eig_t.eigenvalues[j].clamp(0.0, 1.0 - 1e-10);
                    gamma[j] = 1.0 / (1.0 - tau).sqrt() - 1.0;
                }
                let w_g = &eig_t.eigenvectors;
                let dinv =
                    DMatrix::from_diagonal(&DVector::from_iterator(r, sig.iter().map(|v| 1.0 / v)));
                let a = &v_r * &dinv * w_g; // kr x r
                                            // Lambda = A diag(gamma) A'
                let mut a_scaled = a.clone();
                for col in 0..r {
                    for row in 0..a_scaled.nrows() {
                        a_scaled[(row, col)] *= gamma[col];
                    }
                }
                lambda = Some(&a_scaled * a.transpose());
            }
        } else {
            m_g = Some(solve_or_ginv(&(&g_q - &l_g)));
        }

        let sv = cluster_score(s, ncols, |l| {
            let e_gl = ri.column(l).into_owned();
            let u_qq = q_g.transpose() * &e_gl; // k x 1
            if use_simple {
                u_qq.column(0).into_owned()
            } else {
                let wbe =
                    DVector::from_iterator(idx.len(), (0..idx.len()).map(|i| wb_g[i] * e_gl[i]));
                let u_q = r_g.transpose() * wbe; // k_R x 1
                let adj = if crv2 {
                    lambda.as_ref().unwrap() * &u_q
                } else {
                    m_g.as_ref().unwrap() * &u_q
                };
                (&u_qq + &p_g * adj).column(0).into_owned()
            }
        });
        m += &sv * sv.transpose();
    }
    m
}

/// Subset matrix rows by index (R `M[idx,,drop=FALSE]`).
fn rows(m: &DMatrix<f64>, idx: &[usize]) -> DMatrix<f64> {
    let mut out = DMatrix::zeros(idx.len(), m.ncols());
    for (r, &i) in idx.iter().enumerate() {
        out.set_row(r, &m.row(i));
    }
    out
}

/// Full sandwiched cluster V `inv_g * (w*M) * inv_g` for the direct (h==b style)
/// path when a cluster vector is present.
#[allow(clippy::too_many_arguments)]
pub(crate) fn v_cluster(
    inv_g: &DMatrix<f64>,
    rx: &DMatrix<f64>,
    res: &DMatrix<f64>,
    s: &DVector<f64>,
    groups: &[Vec<usize>],
    kind: ClusterKind,
    sqrt_rx: Option<&DMatrix<f64>>,
    k_override: Option<usize>,
) -> DMatrix<f64> {
    let crv_inv_g = match kind {
        ClusterKind::Cr1 => None,
        _ => Some(inv_g),
    };
    let meat = vce_meat_cluster(s, rx, res, groups, kind, crv_inv_g, sqrt_rx, k_override);
    inv_g * (meat * inv_g)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nn_residuals_empty_window_is_empty() {
        let d = DMatrix::<f64>::zeros(0, 1);
        let res = nn_residuals(&[], &d, &[], &[], 3);
        assert_eq!((res.nrows(), res.ncols()), (0, 1));
    }

    #[test]
    fn nn_residuals_single_point_is_nan_like_r() {
        // Ji = 0 -> sqrt(0/1) * (y - 0/0): R yields NaN too.
        let d = DMatrix::from_row_slice(1, 1, &[2.0]);
        let res = nn_residuals(&[0.5], &d, &[1], &[1], 3);
        assert!(res[(0, 0)].is_nan());
    }

    #[test]
    fn dups_dupsid_runs() {
        let (d, id) = compute_dups_dupsid(&[1.0, 1.0, 2.0, 3.0, 3.0, 3.0]);
        assert_eq!(d, vec![2, 2, 1, 3, 3, 3]);
        assert_eq!(id, vec![1, 2, 1, 1, 2, 3]);
    }
}
