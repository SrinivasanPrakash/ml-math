"""Numeric claims made in README.md (worked example and exercises)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import linear_regression as lr  # noqa: E402

X = lr.add_intercept(np.array([1.0, 2, 3])[:, None])
y = np.array([1.0, 2, 2])


def test_worked_example_leverage_and_sigma():
    h = np.diag(lr.hat_matrix(X))
    assert np.allclose(h, [5 / 6, 1 / 3, 5 / 6])
    x = X[:, 1]
    assert np.allclose(h, 1 / 3 + (x - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2))
    w = lr.ols_normal(X, y)
    sse = np.sum((y - X @ w) ** 2)
    assert np.isclose(sse / (3 - 2), 1 / 6) and np.isclose(sse / 3, 1 / 18)


def test_worked_example_gd_first_step_and_bounds():
    _, path = lr.gradient_descent(X, y, 0.1, 1)
    assert np.allclose(path[1], [1 / 6, 11 / 30])
    lmin, lmax, lrmax, lropt = lr.gd_lr_bounds(X)
    assert np.isclose(lmax, (17 / 3 + np.sqrt(265) / 3) / 2)
    assert np.isclose(lmin, (17 / 3 - np.sqrt(265) / 3) / 2)
    assert np.isclose(lmax / lmin, 46.14, atol=0.01)
    assert np.isclose(lrmax, 0.3606, atol=1e-4)
    k = lmax / lmin
    assert np.isclose(lr.gd_contraction_rate(X, lropt), (k - 1) / (k + 1))
    # eta = 0.4 > 2/lmax diverges
    w, _ = lr.gradient_descent(X, y, 0.4, 200)
    assert np.linalg.norm(w) > 1e3


def test_worked_example_ridge():
    assert np.allclose(lr.ridge_closed_form(X, y, 1.0, penalize_intercept=False), [1.0, 1 / 3])
    assert np.allclose(lr.ridge_closed_form(X, y, 1.0), [3 / 8, 7 / 12])


def test_lauchli_matrix_normal_equations_fail_but_qr_works():
    d = 1e-8
    L = np.array([[1.0, 1.0], [d, 0.0], [0.0, d]])
    yy = L @ np.array([1.0, 1.0])
    assert 1.0 + d * d == 1.0                      # d^2 is lost when forming L^T L
    assert np.linalg.matrix_rank(L.T @ L) == 1     # numerically singular
    assert np.allclose(lr.ols_qr(L, yy), [1, 1], atol=1e-6)
    assert np.allclose(lr.ols_svd(L, yy), [1, 1], atol=1e-6)


def test_ridge_1d_optimal_lambda():
    s2, alpha, sigma2 = 4.0, 1.5, 0.7
    mse = lambda lam: (lam**2 * alpha**2 + sigma2 * s2) / (s2 + lam) ** 2
    lams = np.linspace(1e-3, 5, 200001)
    assert np.isclose(lams[np.argmin(mse(lams))], sigma2 / alpha**2, atol=1e-3)


def test_ridge_norm_decreasing_via_svd():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((30, 5)); b = rng.standard_normal(30)
    n = [np.linalg.norm(lr.ridge_svd(A, b, l)) for l in np.logspace(-3, 3, 30)]
    assert all(a > c for a, c in zip(n, n[1:]))


def test_r2_nondecreasing_when_adding_columns():
    rng = np.random.default_rng(1)
    A = lr.add_intercept(rng.standard_normal((40, 6))); b = rng.standard_normal(40)
    r2 = [lr.r_squared(b, A[:, :k] @ lr.ols_qr(A[:, :k], b)) for k in range(1, 8)]
    assert all(a <= c + 1e-12 for a, c in zip(r2, r2[1:]))
