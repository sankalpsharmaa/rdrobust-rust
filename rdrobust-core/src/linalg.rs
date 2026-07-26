//! Linear-algebra primitives: ports of `.rdrobust_vander` (R functions.R:85-94)
//! and `qrXXinv` (R functions.R:128-132).
//!
//! `qr_xx_inv` mirrors R's `qrXXinv` branch for branch: try a Cholesky of the
//! Gram matrix `X'X` and invert through the factor; on Cholesky failure (the
//! Gram is not numerically positive definite) fall back to the Moore-Penrose
//! pseudoinverse `ginv`. The pseudoinverse uses MASS `ginv`'s relative cutoff,
//! keeping singular values `> sqrt(eps) * max(d)`. R's package-level `ginv.tol`
//! default (1e-20) threads only into the covariate-adjustment solve, never into
//! `qrXXinv`, so it is not used here (port spec 01 sec 5.4).
//!
//! Which branch fires (Cholesky vs `ginv`) is a discrete decision; the inverse
//! itself matches R to about 1e-8 rather than bit for bit, since nalgebra's
//! factorizations are not bit identical to R's LAPACK (port spec 01 sec 0).
//!
//! The small-matrix path uses nalgebra: the Gram is `(p+1)x(p+1)`, typically
//! 3x3 to 5x5. faer is reserved for the larger weighted and covariate systems
//! in later modules.

use nalgebra::DMatrix;

/// Builds the Vandermonde design `[1, u, u^2, ..., u^p]` as an `n x (p+1)` matrix.
///
/// Ports `.rdrobust_vander` (R functions.R:85-94). Column 0 is all `1.0`; each
/// later column multiplies the previous one elementwise by `u`
/// (`out[,j] <- out[,j-1] * u`), preserving R's exact left-to-right product
/// order for bit-exact f64 parity. `p == 0` returns an `n x 1` column of ones
/// (R's `p < 1` guard).
///
/// # Arguments
/// * `u` - the evaluation points, length `n`.
/// * `p` - the polynomial order (`>= 0`); the result has `p + 1` columns.
pub fn vander(u: &[f64], p: usize) -> DMatrix<f64> {
    let n = u.len();
    let mut m = DMatrix::<f64>::zeros(n, p + 1);
    for i in 0..n {
        m[(i, 0)] = 1.0;
    }
    for j in 1..=p {
        for i in 0..n {
            m[(i, j)] = m[(i, j - 1)] * u[i];
        }
    }
    m
}

/// Returns `(X'X)^{-1}` via Cholesky, falling back to the pseudoinverse when
/// `X'X` is not positive definite.
///
/// Ports R's `qrXXinv` (functions.R:128-132). Forms the Gram `G = X'X`, attempts
/// a Cholesky factorization, and inverts through the factor on success. On
/// Cholesky failure (G not numerically positive definite, matching R's `try`
/// around `chol`), it falls back to [`ginv`], the Moore-Penrose pseudoinverse.
/// The result is the `(p+1) x (p+1)` inverse Gram, not `X^{-1}`.
///
/// # Arguments
/// * `x` - the `n x (p+1)` design matrix whose inverse Gram is returned.
pub fn qr_xx_inv(x: &DMatrix<f64>) -> DMatrix<f64> {
    let g = x.transpose() * x;
    match g.clone().cholesky() {
        Some(chol) => chol.inverse(),
        None => ginv(&g),
    }
}

/// Computes the Moore-Penrose pseudoinverse via SVD with MASS `ginv`'s relative
/// singular-value cutoff.
///
/// Ports `MASS::ginv` as `qrXXinv` calls it (port spec 01 sec 5.4). From the SVD
/// `g = U D V'`, it keeps singular values with `d_i > sqrt(eps) * max(d)` (where
/// `eps` is `f64::EPSILON`, so the cutoff is `sqrt(eps) * max(d)` with
/// `sqrt(eps) ~= 1.49e-8`), inverts the kept values, zeros the rest, and returns
/// `V diag(1/d_i) U'`. The keep cutoff uses MASS's default tolerance, not the
/// package-level `ginv.tol` (1e-20), which never reaches this path.
///
/// # Arguments
/// * `g` - the matrix to pseudoinvert, in practice the symmetric Gram `X'X`.
///
/// # Panics
/// Panics if the SVD does not yield the requested `U` and `V'` factors.
pub fn ginv(g: &DMatrix<f64>) -> DMatrix<f64> {
    let svd = g.clone().svd(true, true);
    let d = &svd.singular_values;
    let dmax = d.iter().cloned().fold(0.0_f64, f64::max);
    let tol = f64::EPSILON.sqrt() * dmax;
    let r = d.len();
    let u = svd.u.as_ref().expect("u was requested");
    let vt = svd.v_t.as_ref().expect("v_t was requested");
    let mut dinv = DMatrix::<f64>::zeros(r, r);
    for i in 0..r {
        if d[i] > tol {
            dinv[(i, i)] = 1.0 / d[i];
        }
    }
    vt.transpose() * dinv * u.transpose()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vander_built_by_successive_mult() {
        let m = vander(&[2.0, 3.0], 3);
        // row 0: [1, 2, 4, 8]; row 1: [1, 3, 9, 27]
        assert_eq!(
            (m[(0, 0)], m[(0, 1)], m[(0, 2)], m[(0, 3)]),
            (1.0, 2.0, 4.0, 8.0)
        );
        assert_eq!((m[(1, 1)], m[(1, 2)], m[(1, 3)]), (3.0, 9.0, 27.0));
    }

    #[test]
    fn vander_degree_zero_is_intercept() {
        let m = vander(&[5.0, 9.0], 0);
        assert_eq!((m.nrows(), m.ncols()), (2, 1));
        assert_eq!((m[(0, 0)], m[(1, 0)]), (1.0, 1.0));
    }

    #[test]
    fn qr_xx_inv_inverts_well_conditioned_gram() {
        let x = vander(&[0.0, 1.0, 2.0, 3.0, 4.0], 2); // 5x3, full rank
        let inv = qr_xx_inv(&x);
        let g = x.transpose() * &x;
        let id = &inv * &g;
        let i3 = DMatrix::<f64>::identity(3, 3);
        assert!((id - i3).amax() < 1e-9);
    }

    #[test]
    fn ginv_satisfies_moore_penrose_on_singular() {
        // rank-2 symmetric 3x3 (first two rows identical)
        let g = DMatrix::from_row_slice(3, 3, &[2.0, 2.0, 0.0, 2.0, 2.0, 0.0, 0.0, 0.0, 1.0]);
        let pinv = ginv(&g);
        // Moore-Penrose identity: G G^+ G == G
        let recon = &g * &pinv * &g;
        assert!((recon - g).amax() < 1e-9);
    }
}
