//! PyO3 bindings for `rdrobust-core`.
//!
//! Exposes `rdrobust`, `rdplot`, batch, and multi-cutoff entry points as
//! NumPy-in, dict-out Python functions.

use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use rdrobust_core::{
    rdplot as core_rdplot, rdrobust_batch as core_rdrobust_batch,
    rdrobust_multi_cutoff as core_rdrobust_multi_cutoff, rdrobust_with as core_rdrobust_with,
    Dataset as CoreDataset, Kernel, RdRobust, RdRobustOptions as CoreRdRobustOptions, Vce,
};

/// Map a kernel string to `Kernel`, accepting R's aliases (rdrobust.R kernel
/// validation). `tri`/`triangular`/`""` give Triangular, `epa`/`epanechnikov`
/// give Epanechnikov, `uni`/`uniform` give Uniform; anything else raises
/// `ValueError`.
fn parse_kernel(s: &str) -> PyResult<Kernel> {
    match s {
        "tri" | "triangular" | "" => Ok(Kernel::Triangular),
        "epa" | "epanechnikov" => Ok(Kernel::Epanechnikov),
        "uni" | "uniform" => Ok(Kernel::Uniform),
        other => Err(PyValueError::new_err(format!("unknown kernel '{other}'"))),
    }
}

/// Parse the vce string. `cr1`/`cr2`/`cr3` are cluster estimators; without a
/// cluster vector R falls back to `hc1`/`hc2`/`hc3` (rdrobust.R:336-344), which
/// we replicate. With a cluster vector the core remaps hc/nn -> CR1/CRV2/CRV3.
fn parse_vce(s: &str, has_cluster: bool) -> PyResult<Vce> {
    match s {
        "nn" => Ok(Vce::NearestNeighbor),
        "hc0" => Ok(Vce::Hc0),
        "hc1" => Ok(Vce::Hc1),
        "hc2" => Ok(Vce::Hc2),
        "hc3" => Ok(Vce::Hc3),
        "cr0" | "cr1" => Ok(if has_cluster { Vce::Cr1 } else { Vce::Hc1 }),
        "cr2" => Ok(if has_cluster { Vce::Cr2 } else { Vce::Hc2 }),
        "cr3" => Ok(if has_cluster { Vce::Cr3 } else { Vce::Hc3 }),
        other => Err(PyValueError::new_err(format!(
            "vce '{other}' not supported (nn, hc0-hc3, cr1-cr3)"
        ))),
    }
}

fn ensure_len(name: &str, len: usize, expected: usize) -> PyResult<()> {
    if len == expected {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "{name} has length {len}, expected {expected}"
        )))
    }
}

fn ensure_finite_scalar(name: &str, value: f64) -> PyResult<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!("{name} must be finite")))
    }
}

fn ensure_positive_scalar(name: &str, value: f64) -> PyResult<()> {
    ensure_finite_scalar(name, value)?;
    if value > 0.0 {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!("{name} must be positive")))
    }
}

fn ensure_finite_slice(name: &str, values: &[f64]) -> PyResult<()> {
    for (i, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(PyValueError::new_err(format!("{name}[{i}] must be finite")));
        }
    }
    Ok(())
}

fn ensure_nonnegative_slice(name: &str, values: &[f64]) -> PyResult<()> {
    ensure_finite_slice(name, values)?;
    for (i, &value) in values.iter().enumerate() {
        if value < 0.0 {
            return Err(PyValueError::new_err(format!(
                "{name}[{i}] must be non-negative"
            )));
        }
    }
    Ok(())
}

fn ensure_choice(name: &str, value: &str, choices: &[&str]) -> PyResult<()> {
    if choices.contains(&value) {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!("unknown {name} '{value}'")))
    }
}

/// R's `c<=x_min | c>=x_max` check (rdrobust.R:244): the cutoff must lie strictly
/// inside the support. `c == max(x)` leaves a right side with zero range, so every
/// pilot window is empty and the estimate is NaN.
fn ensure_cutoff_split(x: &[f64], c: f64) -> PyResult<()> {
    let x_min = x.iter().cloned().fold(f64::INFINITY, f64::min);
    let x_max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if c > x_min && c < x_max {
        Ok(())
    } else {
        Err(PyValueError::new_err(
            "c must lie strictly inside the range of x",
        ))
    }
}

/// R's degenerate-first-stage stop (rdrobust.R:175-177): `fuzzy` constant on both
/// sides with no jump at the cutoff divides by a zero first stage.
fn ensure_fuzzy_identified(x: &[f64], t: &[f64], c: f64) -> PyResult<()> {
    let side = |on_left: bool| -> Vec<f64> {
        (0..t.len())
            .filter(|&i| (x[i] < c) == on_left)
            .map(|i| t[i])
            .collect()
    };
    let (tl, tr) = (side(true), side(false));
    let constant = |v: &[f64]| v.iter().all(|&u| u == v[0]);
    if constant(&tl) && constant(&tr) && (tl[0] - tr[0]).abs() < f64::EPSILON.sqrt() {
        Err(PyValueError::new_err(
            "fuzzy has no variation and no jump at the cutoff; the fuzzy estimator is not identified",
        ))
    } else {
        Ok(())
    }
}

/// Parse a bandwidth argument that may be a scalar or a (left, right) sequence,
/// as R accepts `h`/`b` of length 1 or 2. A float or a length-1 sequence gives
/// `(v, v)`; a length-2 tuple/list/ndarray gives `(l, r)`; anything else raises
/// `ValueError`.
fn parse_bandwidth(name: &str, value: Option<Bound<'_, PyAny>>) -> PyResult<Option<(f64, f64)>> {
    match value {
        None => Ok(None),
        Some(obj) => {
            if let Ok(v) = obj.extract::<f64>() {
                return Ok(Some((v, v)));
            }
            if let Ok(pair) = obj.extract::<Vec<f64>>() {
                match pair.as_slice() {
                    [v] => return Ok(Some((*v, *v))),
                    [l, r] => return Ok(Some((*l, *r))),
                    _ => {}
                }
            }
            Err(PyValueError::new_err(format!(
                "{name} must be a scalar or a length-2 (left, right) sequence"
            )))
        }
    }
}

struct RdrobustValidationOptions<'a> {
    n: usize,
    c: f64,
    p: usize,
    q: usize,
    deriv: usize,
    nnmatch: usize,
    scaleregul: f64,
    scalepar: f64,
    level: f64,
    bwselect: &'a str,
    masspoints: &'a str,
    bwcheck: Option<usize>,
    rho: Option<f64>,
    h: Option<(f64, f64)>,
    b: Option<(f64, f64)>,
}

fn validate_rdrobust_options(opts: RdrobustValidationOptions<'_>) -> PyResult<()> {
    if opts.n < 2 {
        return Err(PyValueError::new_err(
            "y and x must have at least two observations",
        ));
    }
    if opts.deriv > opts.p {
        return Err(PyValueError::new_err("deriv must be <= p"));
    }
    if opts.q <= opts.p {
        return Err(PyValueError::new_err("q must be > p"));
    }
    if opts.nnmatch == 0 {
        return Err(PyValueError::new_err("nnmatch must be >= 1"));
    }
    ensure_finite_scalar("c", opts.c)?;
    ensure_finite_scalar("scalepar", opts.scalepar)?;
    ensure_finite_scalar("scaleregul", opts.scaleregul)?;
    if opts.scaleregul < 0.0 {
        return Err(PyValueError::new_err("scaleregul must be non-negative"));
    }
    ensure_finite_scalar("level", opts.level)?;
    if !(opts.level > 0.0 && opts.level < 100.0) {
        return Err(PyValueError::new_err("level must be in (0, 100)"));
    }
    ensure_choice(
        "bwselect",
        opts.bwselect,
        &[
            "", "mserd", "msesum", "msetwo", "msecomb1", "msecomb2", "cerrd", "cersum", "certwo",
            "cercomb1", "cercomb2",
        ],
    )?;
    ensure_choice("masspoints", opts.masspoints, &["off", "check", "adjust"])?;
    if matches!(opts.bwcheck, Some(0)) {
        return Err(PyValueError::new_err("bwcheck must be >= 1 when supplied"));
    }
    if let Some(value) = opts.rho {
        ensure_positive_scalar("rho", value)?;
    }
    if let Some((l, r)) = opts.h {
        ensure_positive_scalar("h", l)?;
        ensure_positive_scalar("h", r)?;
    }
    if let Some((l, r)) = opts.b {
        ensure_positive_scalar("b", l)?;
        ensure_positive_scalar("b", r)?;
    }
    Ok(())
}

fn parse_bwselect(s: &str) -> PyResult<&'static str> {
    match s {
        "" => Ok(""),
        "mserd" => Ok("mserd"),
        "msesum" => Ok("msesum"),
        "msetwo" => Ok("msetwo"),
        "msecomb1" => Ok("msecomb1"),
        "msecomb2" => Ok("msecomb2"),
        "cerrd" => Ok("cerrd"),
        "cersum" => Ok("cersum"),
        "certwo" => Ok("certwo"),
        "cercomb1" => Ok("cercomb1"),
        "cercomb2" => Ok("cercomb2"),
        other => Err(PyValueError::new_err(format!("unknown bwselect '{other}'"))),
    }
}

fn parse_masspoints(s: &str) -> PyResult<&'static str> {
    match s {
        "off" => Ok("off"),
        "check" => Ok("check"),
        "adjust" => Ok("adjust"),
        other => Err(PyValueError::new_err(format!(
            "unknown masspoints '{other}'"
        ))),
    }
}

#[allow(clippy::too_many_arguments)]
fn build_core_options(
    n: usize,
    c: f64,
    p: usize,
    q: usize,
    deriv: usize,
    kernel: &str,
    vce: &str,
    has_cluster: bool,
    nnmatch: usize,
    covs_drop: bool,
    scaleregul: f64,
    scalepar: f64,
    level: f64,
    bwselect: &str,
    masspoints: &str,
    stdvars: bool,
    bwrestrict: bool,
    bwcheck: Option<usize>,
    rho: Option<f64>,
    sharpbw: bool,
    h: Option<(f64, f64)>,
    b: Option<(f64, f64)>,
) -> PyResult<CoreRdRobustOptions> {
    validate_rdrobust_options(RdrobustValidationOptions {
        n,
        c,
        p,
        q,
        deriv,
        nnmatch,
        scaleregul,
        scalepar,
        level,
        bwselect,
        masspoints,
        bwcheck,
        rho,
        h,
        b,
    })?;

    Ok(CoreRdRobustOptions {
        c,
        p,
        q,
        deriv,
        kernel: parse_kernel(kernel)?,
        vce: parse_vce(vce, has_cluster)?,
        nnmatch,
        covs_drop_coll: covs_drop,
        scaleregul,
        scalepar,
        level,
        bwselect: parse_bwselect(bwselect)?,
        masspoints: parse_masspoints(masspoints)?,
        stdvars,
        bwrestrict,
        bwcheck,
        rho,
        sharpbw,
        h,
        b,
    })
}

struct OwnedDataset {
    y: Vec<f64>,
    x: Vec<f64>,
    t: Option<Vec<f64>>,
    z: Option<Vec<f64>>,
    dz: usize,
    cluster: Option<Vec<f64>>,
    weights: Option<Vec<f64>>,
}

impl OwnedDataset {
    fn as_core(&self) -> CoreDataset<'_> {
        CoreDataset {
            y: &self.y,
            x: &self.x,
            t: self.t.as_deref(),
            z: self.z.as_deref(),
            dz: self.dz,
            cluster: self.cluster.as_deref(),
            weights: self.weights.as_deref(),
        }
    }
}

fn sorted_owned_dataset(
    ys: &[f64],
    xs: &[f64],
    t: Option<&[f64]>,
    z: Option<&[f64]>,
    dz: usize,
    cluster: Option<&[f64]>,
    weights: Option<&[f64]>,
) -> OwnedDataset {
    let n = ys.len();
    let mut perm: Vec<usize> = (0..n).collect();
    perm.sort_by(|&a, &b| xs[a].total_cmp(&xs[b]));
    let gather1 = |v: &[f64]| -> Vec<f64> { perm.iter().map(|&i| v[i]).collect() };
    let z_sorted = z.map(|values| {
        let mut buf = Vec::with_capacity(n * dz);
        for &orig in &perm {
            buf.extend_from_slice(&values[orig * dz..orig * dz + dz]);
        }
        buf
    });
    OwnedDataset {
        y: gather1(ys),
        x: gather1(xs),
        t: t.map(gather1),
        z: z_sorted,
        dz,
        cluster: cluster.map(gather1),
        weights: weights.map(gather1),
    }
}

fn read_optional_matrix(
    name: &str,
    value: &Option<PyReadonlyArray2<'_, f64>>,
    batch: usize,
    n: usize,
    nonnegative: bool,
) -> PyResult<Option<Vec<Vec<f64>>>> {
    let Some(array) = value else {
        return Ok(None);
    };
    let arr = array.as_array();
    let shape = arr.shape();
    if shape != [batch, n] {
        return Err(PyValueError::new_err(format!(
            "{name} has shape ({}, {}), expected ({batch}, {n})",
            shape[0], shape[1]
        )));
    }
    let values = array.as_slice()?;
    if nonnegative {
        ensure_nonnegative_slice(name, values)?;
    } else {
        ensure_finite_slice(name, values)?;
    }
    Ok(Some(
        (0..batch)
            .map(|i| values[i * n..(i + 1) * n].to_vec())
            .collect(),
    ))
}

fn read_optional_covariates(
    value: &Option<PyReadonlyArray3<'_, f64>>,
    batch: usize,
    n: usize,
) -> PyResult<(Option<Vec<Vec<f64>>>, usize)> {
    let Some(array) = value else {
        return Ok((None, 0));
    };
    let arr = array.as_array();
    let shape = arr.shape();
    if shape[0] != batch || shape[1] != n {
        return Err(PyValueError::new_err(format!(
            "covs has shape ({}, {}, {}), expected ({batch}, {n}, dz)",
            shape[0], shape[1], shape[2]
        )));
    }
    let dz = shape[2];
    if dz == 0 {
        return Err(PyValueError::new_err("covs must have at least one column"));
    }
    let values = array.as_slice()?;
    ensure_finite_slice("covs", values)?;
    Ok((
        Some(
            (0..batch)
                .map(|i| values[i * n * dz..(i + 1) * n * dz].to_vec())
                .collect(),
        ),
        dz,
    ))
}

fn rdrobust_to_dict<'py>(py: Python<'py>, out: &RdRobust) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("coef", out.coef.to_vec())?;
    d.set_item("se", out.se.to_vec())?;
    d.set_item("z", out.z.to_vec())?;
    d.set_item("pv", out.pv.to_vec())?;
    d.set_item("ci", out.ci.iter().map(|r| r.to_vec()).collect::<Vec<_>>())?;
    d.set_item(
        "bws",
        out.bws.iter().map(|r| r.to_vec()).collect::<Vec<_>>(),
    )?;
    Ok(d)
}

fn rdrobust_vec_to_list<'py>(
    py: Python<'py>,
    outputs: &[RdRobust],
) -> PyResult<Bound<'py, PyList>> {
    let list = PyList::empty(py);
    for out in outputs {
        list.append(rdrobust_to_dict(py, out)?)?;
    }
    Ok(list)
}

fn validate_rdplot_options(n: usize, c: f64, scale: f64, binselect: &str) -> PyResult<()> {
    if n < 2 {
        return Err(PyValueError::new_err(
            "y and x must have at least two observations",
        ));
    }
    ensure_finite_scalar("c", c)?;
    ensure_positive_scalar("scale", scale)?;
    ensure_choice(
        "binselect",
        binselect,
        &["", "es", "espr", "esmv", "esmvpr"],
    )?;
    Ok(())
}

/// Estimate a sharp or fuzzy RD treatment effect.
///
/// `x` need not be sorted; observations are reordered by ascending `x`
/// internally (a stable sort matching R's `order()`), so nearest-neighbor
/// variance is computed correctly regardless of input order. `h` and `b` each
/// accept either a scalar (applied to both sides) or a length-2 (left, right)
/// pair for asymmetric bandwidths. The cutoff must lie strictly inside the range
/// of `x`. With fewer than 20 observations the full support is used as the
/// bandwidth on both sides, as in R.
#[pyfunction]
#[pyo3(signature = (
    y, x, fuzzy=None, covs=None, cluster=None, weights=None, c=0.0, p=1, q=2, deriv=0,
    kernel="tri".to_string(), vce="nn".to_string(), nnmatch=3, covs_drop=false,
    scaleregul=1.0, scalepar=1.0, level=95.0, bwselect="mserd".to_string(),
    masspoints="adjust".to_string(), stdvars=true, bwrestrict=true, bwcheck=None,
    rho=None, sharpbw=false, h=None, b=None,
))]
#[allow(clippy::too_many_arguments)]
fn rdrobust<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    x: PyReadonlyArray1<'py, f64>,
    fuzzy: Option<PyReadonlyArray1<'py, f64>>,
    covs: Option<PyReadonlyArray2<'py, f64>>,
    cluster: Option<PyReadonlyArray1<'py, f64>>,
    weights: Option<PyReadonlyArray1<'py, f64>>,
    c: f64,
    p: usize,
    q: usize,
    deriv: usize,
    kernel: String,
    vce: String,
    nnmatch: usize,
    covs_drop: bool,
    scaleregul: f64,
    scalepar: f64,
    level: f64,
    bwselect: String,
    masspoints: String,
    stdvars: bool,
    bwrestrict: bool,
    bwcheck: Option<usize>,
    rho: Option<f64>,
    sharpbw: bool,
    h: Option<Bound<'py, PyAny>>,
    b: Option<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    let ys = y.as_slice()?;
    let xs = x.as_slice()?;
    let n = ys.len();
    ensure_len("x", xs.len(), n)?;
    ensure_finite_slice("y", ys)?;
    ensure_finite_slice("x", xs)?;
    let t = match &fuzzy {
        Some(f) => {
            let values = f.as_slice()?;
            ensure_len("fuzzy", values.len(), n)?;
            ensure_finite_slice("fuzzy", values)?;
            Some(values)
        }
        None => None,
    };
    let cl = match &cluster {
        Some(c) => {
            let values = c.as_slice()?;
            ensure_len("cluster", values.len(), n)?;
            ensure_finite_slice("cluster", values)?;
            Some(values)
        }
        None => None,
    };
    let wt = match &weights {
        Some(w) => {
            let values = w.as_slice()?;
            ensure_len("weights", values.len(), n)?;
            ensure_nonnegative_slice("weights", values)?;
            Some(values)
        }
        None => None,
    };
    let (zflat, dz) = match &covs {
        Some(cc) => {
            let arr = cc.as_array();
            let shape = arr.shape();
            if shape[0] != n {
                return Err(PyValueError::new_err(format!(
                    "covs has {rows} rows, expected {n}",
                    rows = shape[0]
                )));
            }
            if shape[1] == 0 {
                return Err(PyValueError::new_err("covs must have at least one column"));
            }
            let values = cc.as_slice()?;
            ensure_finite_slice("covs", values)?;
            (Some(values), shape[1])
        }
        None => (None, 0),
    };
    let h_pair = parse_bandwidth("h", h)?;
    let b_pair = parse_bandwidth("b", b)?;
    let opts = build_core_options(
        n,
        c,
        p,
        q,
        deriv,
        &kernel,
        &vce,
        cl.is_some(),
        nnmatch,
        covs_drop,
        scaleregul,
        scalepar,
        level,
        &bwselect,
        &masspoints,
        stdvars,
        bwrestrict,
        bwcheck,
        rho,
        sharpbw,
        h_pair,
        b_pair,
    )?;
    ensure_cutoff_split(xs, c)?;
    if let Some(tt) = t {
        ensure_fuzzy_identified(xs, tt, c)?;
    }
    let data = sorted_owned_dataset(ys, xs, t, zflat, dz, cl, wt);
    let out = py.detach(|| {
        core_rdrobust_with(
            &data.y,
            &data.x,
            data.t.as_deref(),
            data.z.as_deref(),
            data.dz,
            data.cluster.as_deref(),
            data.weights.as_deref(),
            &opts,
        )
    });
    rdrobust_to_dict(py, &out)
}

/// Compute binned RD plot data.
#[pyfunction]
#[pyo3(signature = (
    y, x, c=0.0, p=4, kernel="uni".to_string(), binselect="esmv".to_string(),
    scale=1.0, nbins_l=None, nbins_r=None, weights=None,
))]
#[allow(clippy::too_many_arguments)]
fn rdplot<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    x: PyReadonlyArray1<'py, f64>,
    c: f64,
    p: usize,
    kernel: String,
    binselect: String,
    scale: f64,
    nbins_l: Option<usize>,
    nbins_r: Option<usize>,
    weights: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let ys = y.as_slice()?;
    let xs = x.as_slice()?;
    let n = ys.len();
    ensure_len("x", xs.len(), n)?;
    ensure_finite_slice("y", ys)?;
    ensure_finite_slice("x", xs)?;
    let wt = match &weights {
        Some(w) => {
            let values = w.as_slice()?;
            ensure_len("weights", values.len(), n)?;
            ensure_nonnegative_slice("weights", values)?;
            Some(values)
        }
        None => None,
    };
    validate_rdplot_options(n, c, scale, &binselect)?;
    ensure_cutoff_split(xs, c)?;
    let kernel = parse_kernel(&kernel)?;
    let out = py.detach(|| {
        core_rdplot(
            ys, xs, c, p, kernel, &binselect, scale, scale, nbins_l, nbins_r, wt,
        )
    });
    let d = PyDict::new(py);
    d.set_item("j_l", out.j_l)?;
    d.set_item("j_r", out.j_r)?;
    d.set_item("coef_l", out.coef_l)?;
    d.set_item("coef_r", out.coef_r)?;
    d.set_item("bin_l", out.bin_l)?;
    d.set_item("bin_r", out.bin_r)?;
    d.set_item("mean_x_l", out.mean_x_l)?;
    d.set_item("mean_y_l", out.mean_y_l)?;
    d.set_item("mean_x_r", out.mean_x_r)?;
    d.set_item("mean_y_r", out.mean_y_r)?;
    Ok(d)
}

/// Estimate many independent RD fits in parallel.
///
/// `y` and `x` must be contiguous `batch x n` arrays. Optional `fuzzy`,
/// `cluster`, and `weights` arrays use the same `batch x n` shape; optional
/// `covs` uses `batch x n x dz`. The scalar options are shared by every fit.
#[pyfunction]
#[pyo3(signature = (
    y, x, fuzzy=None, covs=None, cluster=None, weights=None, c=0.0, p=1, q=2, deriv=0,
    kernel="tri".to_string(), vce="nn".to_string(), nnmatch=3, covs_drop=false,
    scaleregul=1.0, scalepar=1.0, level=95.0, bwselect="mserd".to_string(),
    masspoints="adjust".to_string(), stdvars=true, bwrestrict=true, bwcheck=None,
    rho=None, sharpbw=false, h=None, b=None,
))]
#[allow(clippy::too_many_arguments)]
fn rdrobust_batch<'py>(
    py: Python<'py>,
    y: PyReadonlyArray2<'py, f64>,
    x: PyReadonlyArray2<'py, f64>,
    fuzzy: Option<PyReadonlyArray2<'py, f64>>,
    covs: Option<PyReadonlyArray3<'py, f64>>,
    cluster: Option<PyReadonlyArray2<'py, f64>>,
    weights: Option<PyReadonlyArray2<'py, f64>>,
    c: f64,
    p: usize,
    q: usize,
    deriv: usize,
    kernel: String,
    vce: String,
    nnmatch: usize,
    covs_drop: bool,
    scaleregul: f64,
    scalepar: f64,
    level: f64,
    bwselect: String,
    masspoints: String,
    stdvars: bool,
    bwrestrict: bool,
    bwcheck: Option<usize>,
    rho: Option<f64>,
    sharpbw: bool,
    h: Option<Bound<'py, PyAny>>,
    b: Option<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyList>> {
    let y_arr = y.as_array();
    let y_shape = y_arr.shape();
    let batch = y_shape[0];
    let n = y_shape[1];
    if batch == 0 {
        return Err(PyValueError::new_err(
            "batch must contain at least one dataset",
        ));
    }
    let ys = y.as_slice()?;
    let x_arr = x.as_array();
    let x_shape = x_arr.shape();
    if x_shape != [batch, n] {
        return Err(PyValueError::new_err(format!(
            "x has shape ({}, {}), expected ({batch}, {n})",
            x_shape[0], x_shape[1]
        )));
    }
    let xs = x.as_slice()?;
    ensure_finite_slice("y", ys)?;
    ensure_finite_slice("x", xs)?;
    let t_rows = read_optional_matrix("fuzzy", &fuzzy, batch, n, false)?;
    let cluster_rows = read_optional_matrix("cluster", &cluster, batch, n, false)?;
    let weight_rows = read_optional_matrix("weights", &weights, batch, n, true)?;
    let (cov_rows, dz) = read_optional_covariates(&covs, batch, n)?;
    let h_pair = parse_bandwidth("h", h)?;
    let b_pair = parse_bandwidth("b", b)?;
    let opts = build_core_options(
        n,
        c,
        p,
        q,
        deriv,
        &kernel,
        &vce,
        cluster_rows.is_some(),
        nnmatch,
        covs_drop,
        scaleregul,
        scalepar,
        level,
        &bwselect,
        &masspoints,
        stdvars,
        bwrestrict,
        bwcheck,
        rho,
        sharpbw,
        h_pair,
        b_pair,
    )?;

    let mut owned = Vec::with_capacity(batch);
    for i in 0..batch {
        let row_y = &ys[i * n..(i + 1) * n];
        let row_x = &xs[i * n..(i + 1) * n];
        ensure_cutoff_split(row_x, c)?;
        if let Some(rows) = &t_rows {
            ensure_fuzzy_identified(row_x, &rows[i], c)?;
        }
        owned.push(sorted_owned_dataset(
            row_y,
            row_x,
            t_rows.as_ref().map(|rows| rows[i].as_slice()),
            cov_rows.as_ref().map(|rows| rows[i].as_slice()),
            dz,
            cluster_rows.as_ref().map(|rows| rows[i].as_slice()),
            weight_rows.as_ref().map(|rows| rows[i].as_slice()),
        ));
    }

    let outputs = py.detach(|| {
        let datasets: Vec<CoreDataset<'_>> = owned.iter().map(OwnedDataset::as_core).collect();
        core_rdrobust_batch(&datasets, &opts)
    });
    rdrobust_vec_to_list(py, &outputs)
}

/// Estimate one dataset at many cutoffs in parallel.
#[pyfunction]
#[pyo3(signature = (
    y, x, cutoffs, fuzzy=None, covs=None, cluster=None, weights=None, p=1, q=2, deriv=0,
    kernel="tri".to_string(), vce="nn".to_string(), nnmatch=3, covs_drop=false,
    scaleregul=1.0, scalepar=1.0, level=95.0, bwselect="mserd".to_string(),
    masspoints="adjust".to_string(), stdvars=true, bwrestrict=true, bwcheck=None,
    rho=None, sharpbw=false, h=None, b=None,
))]
#[allow(clippy::too_many_arguments)]
fn rdrobust_multi_cutoff<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    x: PyReadonlyArray1<'py, f64>,
    cutoffs: PyReadonlyArray1<'py, f64>,
    fuzzy: Option<PyReadonlyArray1<'py, f64>>,
    covs: Option<PyReadonlyArray2<'py, f64>>,
    cluster: Option<PyReadonlyArray1<'py, f64>>,
    weights: Option<PyReadonlyArray1<'py, f64>>,
    p: usize,
    q: usize,
    deriv: usize,
    kernel: String,
    vce: String,
    nnmatch: usize,
    covs_drop: bool,
    scaleregul: f64,
    scalepar: f64,
    level: f64,
    bwselect: String,
    masspoints: String,
    stdvars: bool,
    bwrestrict: bool,
    bwcheck: Option<usize>,
    rho: Option<f64>,
    sharpbw: bool,
    h: Option<Bound<'py, PyAny>>,
    b: Option<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyList>> {
    let ys = y.as_slice()?;
    let xs = x.as_slice()?;
    let n = ys.len();
    ensure_len("x", xs.len(), n)?;
    ensure_finite_slice("y", ys)?;
    ensure_finite_slice("x", xs)?;
    let cutoff_values = cutoffs.as_slice()?;
    if cutoff_values.is_empty() {
        return Err(PyValueError::new_err("cutoffs must not be empty"));
    }
    ensure_finite_slice("cutoffs", cutoff_values)?;
    for &cutoff in cutoff_values {
        ensure_cutoff_split(xs, cutoff)?;
    }
    let t = match &fuzzy {
        Some(f) => {
            let values = f.as_slice()?;
            ensure_len("fuzzy", values.len(), n)?;
            ensure_finite_slice("fuzzy", values)?;
            Some(values)
        }
        None => None,
    };
    let cl = match &cluster {
        Some(c) => {
            let values = c.as_slice()?;
            ensure_len("cluster", values.len(), n)?;
            ensure_finite_slice("cluster", values)?;
            Some(values)
        }
        None => None,
    };
    let wt = match &weights {
        Some(w) => {
            let values = w.as_slice()?;
            ensure_len("weights", values.len(), n)?;
            ensure_nonnegative_slice("weights", values)?;
            Some(values)
        }
        None => None,
    };
    let (zflat, dz) = match &covs {
        Some(cc) => {
            let arr = cc.as_array();
            let shape = arr.shape();
            if shape[0] != n {
                return Err(PyValueError::new_err(format!(
                    "covs has {rows} rows, expected {n}",
                    rows = shape[0]
                )));
            }
            if shape[1] == 0 {
                return Err(PyValueError::new_err("covs must have at least one column"));
            }
            let values = cc.as_slice()?;
            ensure_finite_slice("covs", values)?;
            (Some(values), shape[1])
        }
        None => (None, 0),
    };
    if let Some(tt) = t {
        for &cutoff in cutoff_values {
            ensure_fuzzy_identified(xs, tt, cutoff)?;
        }
    }
    let h_pair = parse_bandwidth("h", h)?;
    let b_pair = parse_bandwidth("b", b)?;
    let opts = build_core_options(
        n,
        0.0,
        p,
        q,
        deriv,
        &kernel,
        &vce,
        cl.is_some(),
        nnmatch,
        covs_drop,
        scaleregul,
        scalepar,
        level,
        &bwselect,
        &masspoints,
        stdvars,
        bwrestrict,
        bwcheck,
        rho,
        sharpbw,
        h_pair,
        b_pair,
    )?;
    let data = sorted_owned_dataset(ys, xs, t, zflat, dz, cl, wt);
    let cutoffs_owned = cutoff_values.to_vec();
    let outputs = py.detach(|| {
        core_rdrobust_multi_cutoff(
            &data.y,
            &data.x,
            data.t.as_deref(),
            data.z.as_deref(),
            data.dz,
            data.cluster.as_deref(),
            data.weights.as_deref(),
            &cutoffs_owned,
            &opts,
        )
    });
    rdrobust_vec_to_list(py, &outputs)
}

/// Register the `rdrobust_rust` Python module.
#[pymodule]
fn rdrobust_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rdrobust, m)?)?;
    m.add_function(wrap_pyfunction!(rdplot, m)?)?;
    m.add_function(wrap_pyfunction!(rdrobust_batch, m)?)?;
    m.add_function(wrap_pyfunction!(rdrobust_multi_cutoff, m)?)?;
    Ok(())
}
