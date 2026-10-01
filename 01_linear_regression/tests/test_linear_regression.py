import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import linear_regression as lr  # noqa: E402


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    n, p = 60, 4
    X = lr.add_intercept(rng.standard_normal((n, p - 1)))
    w = np.array([1.0, -2.0, 0.5, 3.0])
    y = X @ w + 0.1 * rng.standard_normal(n)
    return X, y


def test_worked_example_exact():
    X = lr.add_intercept(np.array([1.0, 2, 3])[:, None])
    y = np.array([1.0, 2, 2])
    assert np.allclose(X.T @ X, [[3, 6], [6, 14]])
    w = lr.ols_normal(X, y)
    assert np.allclose(w, [2 / 3, 1 / 2])
    r = lr.residuals(X, y, w)
    assert np.allclose(r, [-1 / 6, 1 / 3, -1 / 6])
    assert np.isclose(r @ r, 1 / 6)
    assert np.isclose(lr.r_squared(y, X @ w), 3 / 4)


def test_solvers_match_lstsq(data):
    X, y = data
    ref = np.linalg.lstsq(X, y, rcond=None)[0]
    for solver in (lr.ols_normal, lr.ols_qr, lr.ols_svd):
        assert np.allclose(solver(X, y), ref, atol=1e-10)


def test_matches_sklearn(data):
    sk = pytest.importorskip("sklearn.linear_model")
    X, y = data
    m = sk.LinearRegression(fit_intercept=False).fit(X, y)
    assert np.allclose(lr.ols_qr(X, y), m.coef_)
    r = sk.Ridge(alpha=2.5, fit_intercept=False).fit(X, y)
    assert np.allclose(lr.ridge_closed_form(X, y, 2.5), r.coef_)


def test_gradient_check(data):
    X, y = data
    rng = np.random.default_rng(1)
    w = rng.standard_normal(X.shape[1])
    g = lr.mse_grad(X, y, w)
    eps = 1e-6
    num = np.array([(lr.mse(X, y, w + eps * e) - lr.mse(X, y, w - eps * e)) / (2 * eps)
                    for e in np.eye(len(w))])
    assert np.allclose(g, num, atol=1e-7)


def test_gradient_zero_at_ols(data):
    X, y = data
    assert np.allclose(lr.mse_grad(X, y, lr.ols_qr(X, y)), 0, atol=1e-10)


def test_residual_orthogonality(data):
    X, y = data
    r = lr.residuals(X, y, lr.ols_qr(X, y))
    assert np.allclose(X.T @ r, 0, atol=1e-10)
    assert abs(r.sum()) < 1e-10  # intercept column => residuals sum to zero


def test_hat_matrix_properties(data):
    X, y = data
    for H in (lr.hat_matrix(X), lr.hat_matrix_qr(X)):
        assert np.allclose(H, H.T)
        assert np.allclose(H @ H, H)
        assert np.isclose(np.trace(H), X.shape[1])
        ev = np.linalg.eigvalsh(H)
        assert np.allclose(np.sort(ev)[:-X.shape[1]], 0, atol=1e-10)
        assert np.allclose(ev[-X.shape[1]:], 1)
        assert np.allclose(H @ X, X)
    H = lr.hat_matrix(X)
    assert np.allclose(H @ y, X @ lr.ols_qr(X, y))
    assert np.allclose(np.diag(H), lr.leverages(X))
    assert np.all((np.diag(H) >= 0) & (np.diag(H) <= 1))


def test_pythagoras(data):
    X, y = data
    yh = X @ lr.ols_qr(X, y)
    assert np.isclose(y @ y, yh @ yh + (y - yh) @ (y - yh))


def test_gd_converges_to_ols(data):
    X, y = data
    _, _, _, lr_opt = lr.gd_lr_bounds(X)
    w, path = lr.gradient_descent(X, y, lr_opt, 2000)
    assert np.allclose(w, lr.ols_qr(X, y), atol=1e-8)
    losses = [lr.mse(X, y, p) for p in path]
    assert all(a >= b - 1e-14 for a, b in zip(losses, losses[1:]))


def test_gd_learning_rate_bound(data):
    X, y = data
    _, lmax, lr_max, _ = lr.gd_lr_bounds(X)
    w_ols = lr.ols_qr(X, y)
    ok, _ = lr.gradient_descent(X, y, 0.99 * lr_max, 3000)
    bad, _ = lr.gradient_descent(X, y, 1.05 * lr_max, 300)
    assert np.linalg.norm(ok - w_ols) < 1e-3
    assert np.linalg.norm(bad - w_ols) > 1e3
    assert lr.gd_contraction_rate(X, 1.05 * lr_max) > 1


def test_gd_rate_matches_theory():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((100, 3)) * np.array([1.0, 0.5, 0.2])
    y = rng.standard_normal(100)
    lmin, lmax, _, eta = lr.gd_lr_bounds(X)
    kappa = lmax / lmin
    rate = lr.gd_contraction_rate(X, eta)
    assert np.isclose(rate, (kappa - 1) / (kappa + 1))
    w_star = lr.ols_qr(X, y)
    _, path = lr.gradient_descent(X, y, eta, 60)
    e = np.linalg.norm(path - w_star, axis=1)
    # error never exceeds rate^k * e0 (up to tiny slack) -- the bound is tight
    assert np.all(e[:40] <= rate ** np.arange(40) * e[0] * (1 + 1e-9))


def test_kappa_squared():
    rng = np.random.default_rng(4)
    X = rng.standard_normal((50, 5)) @ np.diag([1, 1, 1, 1e-1, 1e-3])
    assert np.isclose(lr.condition_number(X.T @ X), lr.condition_number(X) ** 2, rtol=1e-6)


def test_qr_svd_beat_normal_equations_when_ill_conditioned():
    x = np.linspace(0, 1, 40)
    X = lr.poly_features(x, 11)
    w_true = np.ones(12)
    y = X @ w_true
    errs = {f.__name__: np.linalg.norm(f(X, y) - w_true)
            for f in (lr.ols_normal, lr.ols_qr, lr.ols_svd)}
    assert errs["ols_qr"] < errs["ols_normal"]
    assert errs["ols_svd"] < errs["ols_normal"]


def test_ridge_to_ols_as_lambda_to_zero(data):
    X, y = data
    w_ols = lr.ols_qr(X, y)
    errs = [np.linalg.norm(lr.ridge_closed_form(X, y, l) - w_ols) for l in (1, 1e-2, 1e-4, 1e-8)]
    assert errs == sorted(errs, reverse=True)
    assert errs[-1] < 1e-6


def test_ridge_svd_equals_closed_form(data):
    X, y = data
    for l in (0.1, 1.0, 50.0):
        assert np.allclose(lr.ridge_svd(X, y, l), lr.ridge_closed_form(X, y, l))


def test_ridge_norm_monotone_and_to_zero(data):
    X, y = data
    norms = [np.linalg.norm(lr.ridge_closed_form(X, y, l)) for l in (0, 0.1, 1, 10, 100, 1e6)]
    assert all(a >= b for a, b in zip(norms, norms[1:]))
    assert norms[-1] < 1e-3


def test_ridge_dof_limits(data):
    X, _ = data
    assert np.isclose(lr.ridge_effective_dof(X, 0.0), X.shape[1])
    assert lr.ridge_effective_dof(X, 1e12) < 1e-6
    f = lr.ridge_shrinkage_factors(X, 3.0)
    assert np.all((f > 0) & (f < 1))


def test_ridge_as_augmented_ols(data):
    # MAP view: ridge == OLS on data augmented with sqrt(lam) I pseudo-observations of 0
    X, y = data
    lam = 4.0
    Xa = np.vstack([X, np.sqrt(lam) * np.eye(X.shape[1])])
    ya = np.concatenate([y, np.zeros(X.shape[1])])
    assert np.allclose(lr.ols_qr(Xa, ya), lr.ridge_closed_form(X, y, lam))


def test_ridge_stabilises_collinear():
    rng = np.random.default_rng(5)
    a = rng.standard_normal(50)
    X = np.column_stack([a, a + 1e-9 * rng.standard_normal(50)])
    y = a + 0.1 * rng.standard_normal(50)
    w = lr.ridge_closed_form(X, y, 1.0)
    assert np.all(np.isfinite(w)) and np.linalg.norm(w) < 5


def test_r2_properties(data):
    X, y = data
    yh = X @ lr.ols_qr(X, y)
    assert 0 <= lr.r_squared(y, yh) <= 1
    assert np.isclose(lr.r_squared(y, np.full_like(y, y.mean())), 0)
    # adding a pure-noise column never lowers R^2 but may lower adjusted R^2
    rng = np.random.default_rng(7)
    X2 = np.column_stack([X, rng.standard_normal(len(y))])
    yh2 = X2 @ lr.ols_qr(X2, y)
    assert lr.r_squared(y, yh2) >= lr.r_squared(y, yh) - 1e-12


def test_vif_detects_collinearity():
    rng = np.random.default_rng(8)
    a = rng.standard_normal(500)
    b = a + 0.05 * rng.standard_normal(500)
    c = rng.standard_normal(500)
    v = lr.vif(np.column_stack([a, b, c]))
    assert v[0] > 100 and v[1] > 100 and v[2] < 1.2


def test_coef_covariance_monte_carlo():
    rng = np.random.default_rng(9)
    X = lr.add_intercept(rng.standard_normal((30, 2)))
    w = np.array([1.0, 2.0, -1.0])
    sigma = 0.5
    W = np.array([lr.ols_qr(X, X @ w + sigma * rng.standard_normal(30)) for _ in range(20000)])
    assert np.allclose(W.mean(0), w, atol=0.01)  # unbiased
    assert np.allclose(np.cov(W.T), lr.coef_covariance(X, sigma**2), rtol=0.05, atol=1e-4)


def test_standardized_residuals_scale(data):
    X, y = data
    z = lr.standardized_residuals(X, y, lr.ols_qr(X, y))
    assert 0.7 < z.std() < 1.3


def test_bias_variance_trend():
    out = lr.bias_variance_poly(lambda x: np.sin(np.pi * x), [1, 3, 5], n_datasets=200, seed=0)
    assert out["bias2"][0] > out["bias2"][1] > out["bias2"][2]  # sin(pi x) is odd; deg 3 beats 1
    assert out["var"][0] < out["var"][2]  # variance grows with flexibility (not strictly per step)


def test_ols_is_invariant_to_column_rescale(data):
    X, y = data
    D = np.diag([1, 10, 0.1, 5])
    assert np.allclose(lr.hat_matrix(X @ D), lr.hat_matrix(X))
