"""End-to-end demo; every number quoted in README.md is printed here (runs in a few seconds)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import linear_regression as lr  # noqa: E402

np.set_printoptions(precision=6, suppress=True)


def head(s):
    print("\n=== " + s)


head("1. Worked example: x=[1,2,3], y=[1,2,2]")
X = lr.add_intercept(np.array([1.0, 2, 3])[:, None]); y = np.array([1.0, 2, 2])
print("X^T X =", X.T @ X, " X^T y =", X.T @ y)
w = lr.ols_normal(X, y); r = y - X @ w
print("w =", w, "(exact 2/3, 1/2)  y_hat =", X @ w, " r =", r)
print("X^T r =", X.T @ r, " SSE =", r @ r, " SST =", np.sum((y - y.mean())**2),
      " R2 =", lr.r_squared(y, X @ w))
print("hat diag =", np.diag(lr.hat_matrix(X)), " trace =", np.trace(lr.hat_matrix(X)))
print("ridge lam=1 (unpenalised intercept):", lr.ridge_closed_form(X, y, 1.0, penalize_intercept=False))
print("ridge lam=1 (all penalised)        :", lr.ridge_closed_form(X, y, 1.0))
print("GD eta=0.1, 5 steps (loss 1/2n):", end=" ")
_, path = lr.gradient_descent(X, y, 0.1, 5); print(path[-1], " loss0 =", lr.mse(X, y, path[0]))
lmin, lmax, lrmax, lropt = lr.gd_lr_bounds(X)
print(f"Hessian eigs: {lmin:.6f} {lmax:.6f}, 2/lmax={lrmax:.6f}, 2/(lmax+lmin)={lropt:.6f}")

head("2. Solvers agree on random well-conditioned data")
rng = np.random.default_rng(0)
Xr = lr.add_intercept(rng.standard_normal((100, 3))); wt = np.array([1.0, -2.0, 0.5, 3.0])
yr = Xr @ wt + 0.3 * rng.standard_normal(100)
ref = np.linalg.lstsq(Xr, yr, rcond=None)[0]
for f in (lr.ols_normal, lr.ols_qr, lr.ols_svd):
    print(f"{f.__name__:11s} max|w - lstsq| = {np.abs(f(Xr, yr) - ref).max():.2e}")
H = lr.hat_matrix(Xr)
print("||H^2-H||_max =", np.abs(H @ H - H).max(), " trace(H) =", np.trace(H),
      " ||X^T r||_max =", np.abs(Xr.T @ (yr - Xr @ ref)).max())
yh = Xr @ ref
print("R2 =", lr.r_squared(yr, yh), " adjR2 =", lr.adjusted_r_squared(yr, yh, 4))
print("sigma2_hat =", np.sum((yr - yh)**2) / (100 - 4), " se(w) =", np.sqrt(np.diag(lr.coef_covariance(Xr, np.sum((yr-yh)**2)/96))))

head("3. kappa squared and solver accuracy (poly degree d on 50 pts in [0,1], exact w = 1)")
x = np.linspace(0, 1, 50)
print("deg  kappa(X)     kappa(X)^2   kappa(X^TX)   err_normal  err_QR     err_SVD")
for d in (3, 6, 9, 12):
    Xp = lr.poly_features(x, d); wt1 = np.ones(d + 1); yp = Xp @ wt1
    e = [np.linalg.norm(f(Xp, yp) - wt1) / np.linalg.norm(wt1) for f in (lr.ols_normal, lr.ols_qr, lr.ols_svd)]
    k = lr.condition_number(Xp)
    print(f"{d:3d}  {k:.2e}  {k**2:.2e}  {lr.condition_number(Xp.T @ Xp):.2e}   " + "  ".join(f"{v:.2e}" for v in e))

head("4. Gradient descent: conditioning controls the rate")
n = 200; z = np.random.default_rng(0).standard_normal((n, 2)); z = (z - z.mean(0)) / z.std(0)
for sc in (0.8, 0.3, 0.1):
    Xg = z @ np.diag([1.0, sc]); yg = Xg @ np.array([1.0, 1.0]) + 0.1 * np.random.default_rng(1).standard_normal(n)
    ws = lr.ols_qr(Xg, yg); lmin, lmax, lrmax, eta = lr.gd_lr_bounds(Xg)
    kappa = lmax / lmin; rate = lr.gd_contraction_rate(Xg, eta)
    _, path = lr.gradient_descent(Xg, yg, eta, 5000, w0=ws + 1.0)
    e = np.linalg.norm(path - ws, axis=1)
    k = int(np.argmax(e < 1e-6 * e[0]))
    pred = int(np.ceil(np.log(1e-6) / np.log(rate)))
    print(f"scale {sc}: kappa={kappa:7.1f}  lmax={lmax:.3f}  2/lmax={lrmax:.3f}  eta*={eta:.3f}  "
          f"rate={rate:.4f}  iters to 1e-6: measured {k}, predicted {pred}")
Xg = z @ np.diag([1.0, 0.3]); yg = Xg @ np.ones(2); ws = lr.ols_qr(Xg, yg); _, lmax, lrmax, _ = lr.gd_lr_bounds(Xg)
for f in (0.99, 1.01):
    w_, _ = lr.gradient_descent(Xg, yg, f * lrmax, 2000)
    print(f"eta = {f}*2/lmax -> |w-w*| after 2000 steps = {np.linalg.norm(w_ - ws):.3e}")

head("5. Ridge")
rng = np.random.default_rng(1); m = 60; t = rng.standard_normal(m)
Xc = np.column_stack([t, t + 0.01 * rng.standard_normal(m)]); yc = t + 0.3 * rng.standard_normal(m)
print("collinear pair: corr =", np.corrcoef(Xc.T)[0, 1], " VIF =", lr.vif(Xc), " kappa(X)=", lr.condition_number(Xc))
for lam in (0, 1e-3, 1e-1, 10):
    wv = lr.ridge_closed_form(Xc, yc, lam) if lam else lr.ols_svd(Xc, yc)
    print(f"lam={lam:<6}: w = {wv}  ||w|| = {np.linalg.norm(wv):.3f}  df = {lr.ridge_effective_dof(Xc, lam):.3f}")
reps = []
for _ in range(2000):
    tt = rng.standard_normal(m); Xb = np.column_stack([tt, tt + 0.01 * rng.standard_normal(m)])
    yb = tt + 0.3 * rng.standard_normal(m); reps.append([lr.ols_svd(Xb, yb), lr.ridge_closed_form(Xb, yb, 1.0)])
reps = np.array(reps)
print("true effect w1+w2=1. Over 2000 redraws: OLS std(w1)=%.3f  ridge std(w1)=%.3f; mean w1+w2 OLS %.3f ridge %.3f" %
      (reps[:, 0, 0].std(), reps[:, 1, 0].std(), reps[:, 0].sum(1).mean(), reps[:, 1].sum(1).mean()))
Xs = rng.standard_normal((50, 4)); ys = Xs @ np.array([1, 0, -1, 2.0]) + 0.5 * rng.standard_normal(50)
print("ridge->OLS: ||w_ridge(1e-8) - w_ols|| =", np.linalg.norm(lr.ridge_closed_form(Xs, ys, 1e-8) - lr.ols_qr(Xs, ys)))

head("6. Bias-variance (y = sin(pi x) + N(0,0.3^2), n=30, 400 datasets)")
out = lr.bias_variance_poly(lambda x: np.sin(np.pi * x), list(range(1, 10)), n_train=30, sigma=0.3, n_datasets=400, seed=0)
print("deg  bias^2    var       bias^2+var+sigma^2")
for d, b, v in zip(out["degree"], out["bias2"], out["var"]):
    print(f"{d:3d}  {b:8.4f}  {v:8.4f}  {b + v + out['noise']:8.4f}")

head("7. Polynomial overfitting (15 pts)")
rng = np.random.default_rng(2); f = lambda x: np.sin(np.pi * x)
xs = np.sort(rng.uniform(-1, 1, 15)); ysn = f(xs) + 0.25 * rng.standard_normal(15)
xte = rng.uniform(-1, 1, 500); yte = f(xte) + 0.25 * rng.standard_normal(500)
for d in (1, 3, 9, 14):
    wd = lr.ols_svd(lr.poly_features(xs, d), ysn)
    print(f"degree {d:2d}: train MSE {np.mean((ysn - lr.poly_features(xs, d) @ wd)**2):.4f}  "
          f"test MSE {np.mean((yte - lr.poly_features(xte, d) @ wd)**2):.4f}")
