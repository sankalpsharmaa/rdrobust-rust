//! Kernel weights -- port of `rdrobust_kweight` (R functions.R:134-144).

use crate::Kernel;

/// Computes local-polynomial kernel weights `w_i = K((x_i - c)/h) / h`, porting `rdrobust_kweight` (functions.R:134-144).
///
/// Returns one weight per element of `x`, element-wise: `w[i]` depends only on
/// `x[i]`. Three details hold parity bit-exact with R:
/// - The `|u| <= 1` support boundary is inclusive. Only the uniform kernel is
///   nonzero at `|u| == 1` (triangular and Epanechnikov both yield `0` there),
///   so it is the sole kernel whose result would change under `<`.
/// - `u^2` is computed as `u*u` because R's `^2` special-cases to `x*x`
///   (`R_pow` returns `x*x` for exponent 2), so the square matches to the bit.
/// - The `/h` divides last, matching R's left-to-right operator order. `h`
///   enters twice: once forming `u = (x_i - c)/h`, once in this final divide.
///
/// Points with `|u| > 1` get weight `0.0`. Downstream code keeps only `w > 0`,
/// excluding them from every fit (R does the same via `ind_V = w > 0`).
///
/// # Arguments
/// - `x`: the running-variable values, length n.
/// - `c`: the cutoff.
/// - `h`: the bandwidth, expected `> 0`.
/// - `kernel`: the kernel choice (triangular, Epanechnikov, or uniform).
pub fn kweight(x: &[f64], c: f64, h: f64, kernel: Kernel) -> Vec<f64> {
    x.iter()
        .map(|&xi| {
            let u = (xi - c) / h;
            let au = u.abs();
            if au > 1.0 {
                0.0
            } else {
                match kernel {
                    Kernel::Triangular => (1.0 - au) / h,
                    Kernel::Epanechnikov => 0.75 * (1.0 - u * u) / h,
                    Kernel::Uniform => 0.5 / h,
                }
            }
        })
        .collect()
}

/// Computes effective weights `W_i * kweight_i`, porting R's `if (!is.null(W)) w = W*w` (functions.R:233).
///
/// Multiplies each kernel weight from [`kweight`] by the matching user weight
/// when `wt` is supplied. Returns one weight per element of `x`.
///
/// Points with effective weight `<= 0` (zero user weight or `|u| > 1`) drop
/// downstream via the `w > 0` filter.
///
/// # Arguments
/// - `x`: the running-variable values, length n.
/// - `c`: the cutoff.
/// - `h`: the bandwidth, expected `> 0`.
/// - `kernel`: the kernel choice passed through to [`kweight`].
/// - `wt`: the full-array user weights aligned element-wise with `x`; `None`
///   returns kernel weights alone.
pub fn eff_weight(x: &[f64], c: f64, h: f64, kernel: Kernel, wt: Option<&[f64]>) -> Vec<f64> {
    let mut w = kweight(x, c, h, kernel);
    if let Some(u) = wt {
        for (wi, &ui) in w.iter_mut().zip(u.iter()) {
            *wi *= ui;
        }
    }
    w
}

#[cfg(test)]
mod tests {
    use super::*;

    // u = (x-c)/h at x = {0, 0.5, 1, 2}, c=0, h=1  ->  u = {0, 0.5, 1, 2}
    #[test]
    fn triangular_matches_closed_form() {
        let w = kweight(&[0.0, 0.5, 1.0, 2.0], 0.0, 1.0, Kernel::Triangular);
        assert_eq!(w, vec![1.0, 0.5, 0.0, 0.0]);
    }

    #[test]
    fn epanechnikov_matches_closed_form() {
        let w = kweight(&[0.0, 0.5, 1.0, 2.0], 0.0, 1.0, Kernel::Epanechnikov);
        // 0.75*(1-u^2): u=0 ->0.75; u=0.5 ->0.75*0.75=0.5625; u>=1 ->0
        assert_eq!(w, vec![0.75, 0.5625, 0.0, 0.0]);
    }

    #[test]
    fn uniform_is_inclusive_at_boundary() {
        let w = kweight(&[0.5, 1.0, 1.0001], 0.0, 1.0, Kernel::Uniform);
        // nonzero exactly at |u| == 1, zero just past it
        assert_eq!(w, vec![0.5, 0.5, 0.0]);
    }

    #[test]
    fn bandwidth_scales_weights() {
        // h=2: u at x=1 is 0.5; triangular weight = (1-0.5)/2 = 0.25
        let w = kweight(&[1.0], 0.0, 2.0, Kernel::Triangular);
        assert_eq!(w, vec![0.25]);
    }
}
