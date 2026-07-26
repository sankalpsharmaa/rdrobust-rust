//! Standard normal CDF (`pnorm`) and quantile (`qnorm`) for p-values and CI
//! critical values. `qnorm` uses Acklam's rational approximation refined by one
//! Halley step, giving full double precision (matches R `qnorm`/`pnorm` to ~1e-15).

use std::f64::consts::{PI, SQRT_2};

/// Standard normal CDF, `P(Z <= x) = 0.5 * erfc(-x / sqrt 2)`.
pub(crate) fn pnorm(x: f64) -> f64 {
    0.5 * libm::erfc(-x / SQRT_2)
}

/// Standard normal quantile (inverse CDF), `p in (0, 1)`.
#[allow(clippy::excessive_precision)]
pub(crate) fn qnorm(p: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];
    const P_LOW: f64 = 0.02425;

    let mut x = if p < P_LOW {
        let qv = (-2.0 * p.ln()).sqrt();
        (((((C[0] * qv + C[1]) * qv + C[2]) * qv + C[3]) * qv + C[4]) * qv + C[5])
            / ((((D[0] * qv + D[1]) * qv + D[2]) * qv + D[3]) * qv + 1.0)
    } else if p <= 1.0 - P_LOW {
        let qv = p - 0.5;
        let r = qv * qv;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * qv
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let qv = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * qv + C[1]) * qv + C[2]) * qv + C[3]) * qv + C[4]) * qv + C[5])
            / ((((D[0] * qv + D[1]) * qv + D[2]) * qv + D[3]) * qv + 1.0)
    };

    // one Halley refinement step -> full double precision
    let e = pnorm(x) - p;
    let u = e * (2.0 * PI).sqrt() * (x * x / 2.0).exp();
    x -= u / (1.0 + x * u / 2.0);
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pnorm_known_values() {
        assert!((pnorm(0.0) - 0.5).abs() < 1e-15);
        assert!((pnorm(1.959963984540054) - 0.975).abs() < 1e-12);
    }

    #[test]
    fn qnorm_inverts_pnorm() {
        // R qnorm(0.975) = 1.959963984540054
        assert!((qnorm(0.975) - 1.959963984540054).abs() < 1e-12);
        assert!((qnorm(0.025) + 1.959963984540054).abs() < 1e-12);
        for &p in &[0.001, 0.1, 0.3, 0.5, 0.7, 0.9, 0.999] {
            assert!((pnorm(qnorm(p)) - p).abs() < 1e-12);
        }
    }
}
