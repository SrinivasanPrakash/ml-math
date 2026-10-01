"""Linear regression from zero -- NumPy only.

Equation numbers (E1, E2, ...) refer to README.md of this directory.
Notation: X is n x p design matrix, y in R^n, w in R^p, H = hat matrix.
"""
import numpy as np


# ---------------------------------------------------------------- data helpers
def add_intercept(X):
    """Prepend a column of ones so that w[0] is the intercept (E1)."""
    X = np.asarray(X, dtype=float)
    return np.column_stack([np.ones(len(X)), X])


def poly_features(x, degree):
    """Vandermonde matrix [1, x, ..., x^degree] for 1-D x (polynomial regression)."""
    x = np.asarray(x, dtype=float).ravel()
    return np.vander(x, degree + 1, increasing=True)


# ---------------------------------------------------------------- loss & gradient
def mse(X, y, w):
    """Mean squared error L(w) = (1/2n)||y - Xw||^2   (E3)."""
    r = y - X @ w
    return float(r @ r) / (2 * len(y))


def mse_grad(X, y, w):
    """Gradient of E3:  (1/n) X^T (Xw - y)   (E8)."""
    return X.T @ (X @ w - y) / len(y)


def mse_hessian(X):
    """Hessian of E3:  (1/n) X^T X   (E9). Constant in w."""
    return X.T @ X / len(X)


# ---------------------------------------------------------------- OLS solvers
def ols_normal(X, y):
    """Solve the normal equations X^T X w = X^T y   (E10). Squares kappa(X)."""
    return np.linalg.solve(X.T @ X, X.T @ y)


def ols_qr(X, y):
    """OLS via thin QR: X = QR, solve R w = Q^T y by back-substitution   (E24)."""
    Q, R = np.linalg.qr(X, mode="reduced")
    return _back_substitute(R, Q.T @ y)


def ols_svd(X, y, rcond=1e-12):
    """OLS via SVD: w = V diag(1/s) U^T y, dropping tiny singular values  (E26)."""
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    keep = s > rcond * s[0]
    return Vt[keep].T @ ((U[:, keep].T @ y) / s[keep])


def _back_substitute(R, b):
    """Solve upper-triangular R w = b."""
    p = len(b)
    w = np.zeros(p)
    for i in range(p - 1, -1, -1):
        w[i] = (b[i] - R[i, i + 1:] @ w[i + 1:]) / R[i, i]
    return w


# ---------------------------------------------------------------- geometry
def hat_matrix(X):
    """H = X (X^T X)^{-1} X^T, orthogonal projector onto col(X)   (E14)."""
    return X @ np.linalg.solve(X.T @ X, X.T)


def hat_matrix_qr(X):
    """H = Q Q^T computed stably from thin QR (same matrix as hat_matrix)."""
    Q, _ = np.linalg.qr(X, mode="reduced")
    return Q @ Q.T


def residuals(X, y, w):
    return y - X @ w


# ---------------------------------------------------------------- gradient descent
def gradient_descent(X, y, lr, n_iter=1000, w0=None):
    """Batch GD on E3: w <- w - lr * grad   (E12). Returns (w, path) with
    path of shape (n_iter+1, p)."""
    p = X.shape[1]
    w = np.zeros(p) if w0 is None else np.array(w0, dtype=float)
    path = [w.copy()]
    for _ in range(n_iter):
        w = w - lr * mse_grad(X, y, w)
        path.append(w.copy())
    return w, np.array(path)


def gd_lr_bounds(X):
    """Return (lambda_min, lambda_max, lr_max=2/lambda_max, lr_opt=2/(lmax+lmin))  (E15-E17)
    of the Hessian (1/n) X^T X."""
    ev = np.linalg.eigvalsh(mse_hessian(X))
    lmin, lmax = ev[0], ev[-1]
    return lmin, lmax, 2.0 / lmax, 2.0 / (lmax + lmin)


def gd_contraction_rate(X, lr):
    """Per-step contraction factor max_i |1 - lr*lambda_i|  (E16)."""
    ev = np.linalg.eigvalsh(mse_hessian(X))
    return float(np.max(np.abs(1 - lr * ev)))


# ---------------------------------------------------------------- ridge
def ridge_closed_form(X, y, lam, penalize_intercept=True):
    """Ridge  w = (X^T X + lam I)^{-1} X^T y   (E30).
    If penalize_intercept is False the first coefficient is left unpenalised."""
    p = X.shape[1]
    D = np.eye(p)
    if not penalize_intercept:
        D[0, 0] = 0.0
    return np.linalg.solve(X.T @ X + lam * D, X.T @ y)


def ridge_svd(X, y, lam):
    """Ridge via SVD: w = V diag(s/(s^2+lam)) U^T y   (E33)."""
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    return Vt.T @ ((s / (s**2 + lam)) * (U.T @ y))


def ridge_shrinkage_factors(X, lam):
    """Filter factors f_i = s_i^2/(s_i^2+lam) in [0,1]   (E34)."""
    s = np.linalg.svd(X, compute_uv=False)
    return s**2 / (s**2 + lam)


def ridge_effective_dof(X, lam):
    """df(lam) = trace of ridge hat matrix = sum_i f_i   (E35)."""
    return float(ridge_shrinkage_factors(X, lam).sum())


# ---------------------------------------------------------------- diagnostics
def r_squared(y, y_hat):
    """R^2 = 1 - SSE/SST   (E19)."""
    sse = np.sum((y - y_hat) ** 2)
    sst = np.sum((y - y.mean()) ** 2)
    return float(1 - sse / sst)


def adjusted_r_squared(y, y_hat, p):
    """Adjusted R^2 with p = number of columns of X including intercept   (E20)."""
    n = len(y)
    r2 = r_squared(y, y_hat)
    return float(1 - (1 - r2) * (n - 1) / (n - p))


def leverages(X):
    """Diagonal of the hat matrix, h_ii   (E21)."""
    Q, _ = np.linalg.qr(X, mode="reduced")
    return np.sum(Q**2, axis=1)


def standardized_residuals(X, y, w):
    """Internally studentised residuals r_i / (sigma_hat sqrt(1-h_ii))   (E22)."""
    n, p = X.shape
    r = y - X @ w
    s2 = r @ r / (n - p)
    return r / np.sqrt(s2 * (1 - leverages(X)))


def condition_number(A):
    """2-norm condition number s_max / s_min."""
    s = np.linalg.svd(A, compute_uv=False)
    return float(s[0] / s[-1])


def vif(X):
    """Variance inflation factors for the columns of X WITHOUT intercept
    (VIF_j = [R^{-1}]_jj for the correlation matrix R)   (E28)."""
    Z = (X - X.mean(0)) / X.std(0)
    R = Z.T @ Z / len(Z)
    return np.diag(np.linalg.inv(R))


def coef_covariance(X, sigma2):
    """Cov(w_hat) = sigma^2 (X^T X)^{-1}   (E27)."""
    return sigma2 * np.linalg.inv(X.T @ X)


# ---------------------------------------------------------------- bias-variance
def bias_variance_poly(f, degrees, n_train=30, sigma=0.3, n_datasets=300,
                       n_test=200, lam=0.0, seed=0):
    """Monte-Carlo estimate of bias^2, variance and noise for polynomial
    regression of each degree against the true function f on [-1,1]   (E37).
    Returns dict of arrays indexed by degree."""
    rng = np.random.default_rng(seed)
    x_test = np.linspace(-1, 1, n_test)
    f_test = f(x_test)
    out = {"degree": np.array(degrees), "bias2": [], "var": [], "noise": sigma**2}
    preds = {d: np.empty((n_datasets, n_test)) for d in degrees}
    for k in range(n_datasets):
        x = rng.uniform(-1, 1, n_train)
        y = f(x) + sigma * rng.standard_normal(n_train)
        for d in degrees:
            X = poly_features(x, d)
            w = ridge_svd(X, y, lam) if lam > 0 else ols_svd(X, y)
            preds[d][k] = poly_features(x_test, d) @ w
    for d in degrees:
        mean_pred = preds[d].mean(0)
        v = preds[d].var(0, ddof=1)
        # subtract v/n_datasets: the Monte-Carlo mean has its own variance
        out["bias2"].append(np.mean((mean_pred - f_test) ** 2 - v / n_datasets))
        out["var"].append(np.mean(v))
    out["bias2"] = np.array(out["bias2"])
    out["var"] = np.array(out["var"])
    return out
