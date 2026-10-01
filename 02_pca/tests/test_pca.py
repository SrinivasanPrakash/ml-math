import os, sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pca import (PCA, center, covariance, pca_eig, pca_svd, pca_power, fix_signs, reconstruction_error,
                 choose_k, standardize, ppca_fit, ppca_loglik, ppca_posterior_mean, make_glyph_data)


@pytest.fixture
def X():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((6, 6))
    return rng.standard_normal((200, 6)) @ A + rng.standard_normal(6) * 3


def test_hand_example():
    # README worked example: eigenvalues 3 and 1/3, components (1,1)/sqrt2 and (1,-1)/sqrt2
    X = np.array([[1, 1], [2, 3], [3, 2], [4, 4.0]])
    p = PCA(2, "eig").fit(X)
    assert np.allclose(p.explained_variance_, [3, 1 / 3])
    s = 1 / np.sqrt(2)
    assert np.allclose(np.abs(p.components_), [[s, s], [s, s]])
    assert np.isclose(p.explained_variance_ratio_[0], 0.9)


@pytest.mark.parametrize("solver", ["eig", "svd", "power"])
def test_orthonormal(X, solver):
    W = PCA(4, solver).fit(X).components_
    assert np.allclose(W @ W.T, np.eye(4), atol=1e-8)


def test_eig_svd_agree(X):
    Xc, _ = center(X)
    v1, c1 = pca_eig(Xc)
    v2, c2 = pca_svd(Xc)
    assert np.allclose(v1, v2)
    assert np.allclose(np.abs(fix_signs(c1) @ fix_signs(c2).T), np.eye(6), atol=1e-8)
    # s^2/(n-1) = eigenvalue
    s = np.linalg.svd(Xc, compute_uv=False)
    assert np.allclose(s**2 / (len(X) - 1), v1)


def test_power_matches_eigh(X):
    Xc, _ = center(X)
    v1, c1 = pca_eig(Xc, 3)
    v2, c2 = pca_power(Xc, 3)
    assert np.allclose(v1, v2, rtol=1e-6)
    assert np.allclose(np.abs(np.sum(c1 * c2, axis=1)), 1, atol=1e-6)


@pytest.mark.parametrize("solver", ["eig", "svd", "power"])
def test_reconstruction_error_is_discarded_eigenvalues(X, solver):
    Xc, _ = center(X)
    vals, _ = pca_eig(Xc)
    for k in range(1, 7):
        assert np.isclose(reconstruction_error(X, k, solver), vals[k:].sum(), atol=1e-8)


def test_variance_preserved(X):
    p = PCA(None, "svd").fit(X)
    Z = p.transform(X)
    # total variance of scores equals total variance of data (rotation)
    assert np.isclose(Z.var(axis=0, ddof=1).sum(), X.var(axis=0, ddof=1).sum())
    # scores are uncorrelated with variance = eigenvalue
    assert np.allclose(np.cov(Z.T), np.diag(p.explained_variance_), atol=1e-8)
    # variance of the k-score projection = sum of top-k eigenvalues
    k = 2
    assert np.isclose(Z[:, :k].var(axis=0, ddof=1).sum(), p.explained_variance_[:k].sum())
    assert np.isclose(p.explained_variance_ratio_.sum(), 1.0)


def test_first_component_maximises_variance(X):
    Xc, _ = center(X)
    C = covariance(Xc)
    w = PCA(1).fit(X).components_[0]
    best = w @ C @ w
    rng = np.random.default_rng(0)
    for _ in range(500):
        u = rng.standard_normal(6)
        u /= np.linalg.norm(u)
        assert u @ C @ u <= best + 1e-12


def test_pythagoras_variance_plus_error_is_total(X):
    Xc, _ = center(X)
    total = np.trace(covariance(Xc))
    rng = np.random.default_rng(3)
    for _ in range(5):  # holds for ANY orthonormal subspace, not just PCA
        Q, _ = np.linalg.qr(rng.standard_normal((6, 2)))
        P = Xc @ Q @ Q.T
        var_proj = (P**2).sum() / (len(X) - 1)
        err = ((Xc - P) ** 2).sum() / (len(X) - 1)
        assert np.isclose(var_proj + err, total)


def test_roundtrip_full_rank(X):
    p = PCA(None, "eig").fit(X)
    assert np.allclose(p.inverse_transform(p.transform(X)), X)


def test_whitening(X):
    Z = PCA(None, "svd", whiten=True).fit_transform(X)
    assert np.allclose(np.cov(Z.T), np.eye(6), atol=1e-8)
    p = PCA(3, "svd", whiten=True).fit(X)
    assert np.allclose(p.inverse_transform(p.transform(X)), PCA(3, "svd").fit(X).inverse_transform(PCA(3, "svd").fit(X).transform(X)))


def test_standardised_pca_is_correlation_pca(X):
    Z, _, _ = standardize(X)
    vals, _ = pca_eig(Z)
    assert np.allclose(vals, np.linalg.eigvalsh(np.corrcoef(X.T))[::-1])
    assert np.isclose(vals.sum(), X.shape[1])  # trace of correlation matrix = d


def test_scale_dependence(X):
    X2 = X.copy()
    X2[:, 0] *= 1000
    assert abs(PCA(1).fit(X2).components_[0, 0]) > 0.999  # dominated by the rescaled feature


def test_choose_k():
    assert choose_k(np.array([6, 3, 1.0]), 0.9) == 2
    assert choose_k(np.array([6, 3, 1.0]), 0.6) == 1
    assert choose_k(np.array([6, 3, 1.0]), 1.0) == 3


def test_sign_convention_deterministic(X):
    a = PCA(3, "eig").fit(X).components_
    b = PCA(3, "svd").fit(X).components_
    assert np.allclose(a, b, atol=1e-6)


def test_ppca(X):
    m = ppca_fit(X, 3)
    Xc, _ = center(X)
    vals, _ = pca_eig(Xc)
    assert np.isclose(m["sigma2"], vals[3:].mean())
    # model covariance reproduces top-k eigenvalues and spreads the rest equally
    Cm = m["W"] @ m["W"].T + m["sigma2"] * np.eye(6)
    ev = np.sort(np.linalg.eigvalsh(Cm))[::-1]
    assert np.allclose(ev[:3], vals[:3]) and np.allclose(ev[3:], m["sigma2"])
    # ML: perturbing W lowers likelihood
    base = ppca_loglik(m, X)
    rng = np.random.default_rng(0)
    for _ in range(5):
        m2 = dict(m, W=m["W"] + 0.05 * rng.standard_normal(m["W"].shape))
        assert ppca_loglik(m2, X) < base
    # as sigma2 -> 0 posterior mean tends to PCA scores (up to scale convention)
    assert ppca_posterior_mean(m, X).shape == (200, 3)


def test_glyph_data_low_rank():
    G = make_glyph_data()
    vals, _ = pca_svd(center(G)[0])
    r = np.cumsum(vals) / vals.sum()
    assert r[4] > 0.5 and vals[4] > 3 * vals[5]  # visible gap after 5 components


def test_matches_sklearn(X):
    sk = pytest.importorskip("sklearn.decomposition")
    ref = sk.PCA(n_components=4, svd_solver="full").fit(X)
    for solver in ["eig", "svd", "power"]:
        p = PCA(4, solver).fit(X)
        assert np.allclose(p.explained_variance_, ref.explained_variance_, rtol=1e-6)
        assert np.allclose(p.explained_variance_ratio_, ref.explained_variance_ratio_, rtol=1e-6)
        assert np.allclose(np.abs(np.sum(p.components_ * ref.components_, axis=1)), 1, atol=1e-6)
        s = np.sign(np.sum(p.components_ * ref.components_, axis=1))[:, None]
        assert np.allclose(p.components_, s * ref.components_, atol=1e-5)


def test_exercise_facts():
    X = np.array([[1, 1], [2, 3], [3, 2], [4, 4.0]])
    m = ppca_fit(X, 1)
    assert np.isclose(m["sigma2"], 1 / 3) and np.isclose(np.linalg.norm(m["W"]), np.sqrt(8 / 3))
    Xw = np.random.default_rng(5).standard_normal((4, 100))
    assert (pca_eig(center(Xw)[0])[0] > 1e-10).sum() == 3  # rank <= n-1 after centring
    C = np.array([[5 / 3, 4 / 3], [4 / 3, 5 / 3]])
    v = np.array([1.0, 0.0])
    for t in range(1, 5):
        v = C @ v; v /= np.linalg.norm(v)
        assert np.isclose(abs((v[0] - v[1]) / (v[0] + v[1])), (1 / 9) ** t)
