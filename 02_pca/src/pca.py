"""Principal Component Analysis, NumPy only.

Equation numbers (E1, E2, ...) refer to README.md in this directory.

Conventions: X is (n, d), rows are samples. C = X_c^T X_c / (n-1) is the sample covariance.
Components are the rows of ``components_`` (shape (k, d)), unit length, ordered by decreasing variance.
"""
import numpy as np


# ---------------------------------------------------------------- basics
def center(X):
    """Subtract the column means (E1). Returns (Xc, mean)."""
    mean = X.mean(axis=0)
    return X - mean, mean


def covariance(Xc):
    """Sample covariance C = Xc^T Xc / (n-1) of already-centred data (E2)."""
    n = Xc.shape[0]
    return Xc.T @ Xc / (n - 1)


def standardize(X):
    """Z-scores: centre and divide by the std (ddof=1). PCA of Z is PCA of the correlation matrix (E30)."""
    Xc, mean = center(X)
    std = Xc.std(axis=0, ddof=1)
    return Xc / std, mean, std


def fix_signs(components):
    """Deterministic sign convention: the largest-|.| entry of each component is positive.
    (Eigenvectors are only defined up to sign, E14.)"""
    comps = components.copy()
    for i, v in enumerate(comps):
        if v[np.argmax(np.abs(v))] < 0:
            comps[i] = -v
    return comps


# ---------------------------------------------------------------- three solvers
def pca_eig(Xc, k=None):
    """Eigendecomposition of the covariance matrix (E12). Returns (eigvals desc, components (k,d))."""
    C = covariance(Xc)
    vals, vecs = np.linalg.eigh(C)  # ascending, orthonormal columns
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    vals = np.clip(vals, 0.0, None)  # tiny negatives from round-off
    k = len(vals) if k is None else k
    return vals[:k], vecs[:, :k].T


def pca_svd(Xc, k=None):
    """Thin SVD of the centred data matrix, X = U S V^T, lambda_i = s_i^2/(n-1) (E21-E23).
    Never forms the d x d covariance, so it is numerically better for ill-conditioned data."""
    n = Xc.shape[0]
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    vals = s**2 / (n - 1)
    k = len(s) if k is None else k
    return vals[:k], Vt[:k]


def pca_power(Xc, k, n_iter=2000, tol=1e-12, seed=0):
    """Power iteration with deflation on the covariance matrix (E27-E28).

    v <- C v / ||C v||  converges to the top eigenvector (if lambda_1 > lambda_2);
    then C <- C - lambda v v^T removes it and we repeat.
    """
    rng = np.random.default_rng(seed)
    C = covariance(Xc).copy()
    d = C.shape[0]
    vals, comps = [], []
    for _ in range(k):
        v = rng.standard_normal(d)
        v /= np.linalg.norm(v)
        for _ in range(n_iter):
            w = C @ v
            nw = np.linalg.norm(w)
            if nw == 0:  # remaining matrix is zero
                break
            w /= nw
            # v and -v are equivalent: compare up to sign
            done = min(np.linalg.norm(w - v), np.linalg.norm(w + v)) < tol
            v = w
            if done:
                break
        lam = float(v @ C @ v)  # Rayleigh quotient (E5)
        vals.append(lam)
        comps.append(v)
        C = C - lam * np.outer(v, v)  # deflation (E28)
    return np.array(vals), np.array(comps)


# ---------------------------------------------------------------- estimator
class PCA:
    """PCA(n_components, solver in {'eig','svd','power'}, whiten=False)."""

    def __init__(self, n_components=None, solver="svd", whiten=False):
        self.n_components = n_components
        self.solver = solver
        self.whiten = whiten

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        Xc, self.mean_ = center(X)
        n, d = X.shape
        k = min(n, d) if self.n_components is None else self.n_components
        if self.solver == "eig":
            vals, comps = pca_eig(Xc, k)
        elif self.solver == "svd":
            vals, comps = pca_svd(Xc, k)
        elif self.solver == "power":
            vals, comps = pca_power(Xc, k)
        else:
            raise ValueError(self.solver)
        self.components_ = fix_signs(comps)
        self.explained_variance_ = vals
        self.total_variance_ = float(np.trace(covariance(Xc)))  # = sum of ALL eigenvalues (E8)
        self.explained_variance_ratio_ = vals / self.total_variance_  # E24
        return self

    def transform(self, X):
        """Scores Z = (X - mean) W^T (E9); divided by sqrt(lambda) if whiten (E26)."""
        Z = (np.asarray(X, float) - self.mean_) @ self.components_.T
        if self.whiten:
            Z = Z / np.sqrt(self.explained_variance_)
        return Z

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, Z):
        """Reconstruction x_hat = mean + z W (E10)."""
        if self.whiten:
            Z = Z * np.sqrt(self.explained_variance_)
        return Z @ self.components_ + self.mean_


# ---------------------------------------------------------------- diagnostics
def reconstruction_error(X, k, solver="svd"):
    """Mean squared reconstruction error (1/(n-1)) sum ||x - x_hat||^2 for rank k (E17)."""
    p = PCA(k, solver).fit(X)
    R = X - p.inverse_transform(p.transform(X))
    return float((R**2).sum() / (X.shape[0] - 1))


def choose_k(eigvals, threshold=0.95):
    """Smallest k with cumulative explained variance ratio >= threshold (E25)."""
    ratio = np.cumsum(eigvals) / np.sum(eigvals)
    return int(np.searchsorted(ratio, threshold - 1e-12) + 1)


def kaiser_k(eigvals):
    """Kaiser rule for correlation-PCA: keep eigenvalues > mean eigenvalue (= 1 when standardised)."""
    return int(np.sum(eigvals > np.mean(eigvals)))


# ---------------------------------------------------------------- probabilistic PCA
def ppca_fit(X, k):
    """Closed-form ML solution of Tipping & Bishop (E31-E33).

    sigma2 = mean of the d-k discarded eigenvalues; W = U_k (Lambda_k - sigma2 I)^{1/2} (R = I).
    Returns dict(mean, W (d,k), sigma2).
    """
    Xc, mean = center(np.asarray(X, float))
    vals, comps = pca_eig(Xc)
    d = len(vals)
    sigma2 = float(vals[k:].mean()) if k < d else 0.0
    W = comps[:k].T * np.sqrt(np.maximum(vals[:k] - sigma2, 0.0))
    return {"mean": mean, "W": W, "sigma2": sigma2}


def ppca_posterior_mean(model, X):
    """E[z|x] = M^{-1} W^T (x - mu), M = W^T W + sigma2 I (E34)."""
    W, s2 = model["W"], model["sigma2"]
    M = W.T @ W + s2 * np.eye(W.shape[1])
    return (np.asarray(X, float) - model["mean"]) @ W @ np.linalg.inv(M)


def ppca_loglik(model, X):
    """Average log-likelihood under x ~ N(mu, W W^T + sigma2 I) (E32)."""
    X = np.asarray(X, float)
    W, s2 = model["W"], model["sigma2"]
    d = X.shape[1]
    Cm = W @ W.T + s2 * np.eye(d)
    Xc = X - model["mean"]
    _, logdet = np.linalg.slogdet(Cm)
    quad = np.einsum("ij,jk,ik->i", Xc, np.linalg.inv(Cm), Xc)
    return float(np.mean(-0.5 * (d * np.log(2 * np.pi) + logdet + quad)))


# ---------------------------------------------------------------- synthetic data
def make_glyph_data(n=300, size=12, noise=0.15, seed=0):
    """Digit-like synthetic images: random mixtures of 5 smooth strokes on a size x size grid + noise.
    The true rank is 5, so PCA should need ~5 components. Returns (n, size*size)."""
    rng = np.random.default_rng(seed)
    g = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(g, g)
    strokes = [
        np.exp(-((xx) ** 2) / 0.05),  # vertical bar
        np.exp(-((yy) ** 2) / 0.05),  # horizontal bar
        np.exp(-((xx - yy) ** 2) / 0.05),  # diagonal
        np.exp(-((xx + yy) ** 2) / 0.05),  # anti-diagonal
        np.exp(-((xx**2 + yy**2 - 0.6) ** 2) / 0.05),  # ring
    ]
    S = np.array([s.ravel() for s in strokes])
    coef = rng.random((n, len(strokes))) * (rng.random((n, len(strokes))) < 0.5)
    return coef @ S + noise * rng.standard_normal((n, size * size))
