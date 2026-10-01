"""Regenerate every figure in figures/ deterministically (fixed seeds)."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))
import linear_regression as lr  # noqa: E402

OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=130)
    plt.close(fig)
    print("wrote", name)


# ------------------------------------------------------------ 1. projection geometry
def fig_projection():
    X = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])  # col(X) = the x-y plane
    X = np.array([[1.0, 0.2], [0.3, 1.0], [0.1, 0.1]])
    y = np.array([1.2, 0.9, 1.1])
    w = lr.ols_qr(X, y)
    yh = X @ w
    r = y - yh
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    a, b = np.meshgrid(np.linspace(-0.3, 1.2, 6), np.linspace(-0.3, 1.2, 6))
    P = a[..., None] * X[:, 0] + b[..., None] * X[:, 1]
    ax.plot_surface(P[..., 0], P[..., 1], P[..., 2], alpha=0.2, color="tab:blue")
    o = np.zeros(3)
    def arrow(v, c, lab, frm=o, ls="-", at=1.0, off=(0.03, 0.03, 0.03)):
        ax.quiver(*frm, *v, color=c, arrow_length_ratio=0.08, linestyle=ls, linewidth=2)
        ax.text(*(frm + at * v + np.array(off)), lab, color=c)
    arrow(X[:, 0], "tab:blue", r"$x_1$ (col 1)")
    arrow(X[:, 1], "tab:cyan", r"$x_2$ (col 2)")
    arrow(y, "k", r"$y$", off=(-0.15, 0, 0.05))
    arrow(yh, "tab:green", r"$\hat y = Hy = Xw$")
    arrow(r, "tab:red", r"$r=(I-H)y\ \perp\ \mathrm{col}(X)$", frm=yh, at=0.5, off=(0.1, 0.0, 0.0))
    ax.set_xlabel("obs 1"); ax.set_ylabel("obs 2"); ax.set_zlabel("obs 3")
    ax.set_title("OLS = orthogonal projection of $y\\in\\mathbb{R}^3$ onto the plane col$(X)$\n"
                 f"$|X^\\top r|_\\infty$ = {np.abs(X.T @ r).max():.1e}  (right angle)")
    ax.set_xlim(-0.2, 1.4); ax.set_ylim(-0.2, 1.4); ax.set_zlim(-0.2, 1.4)
    ax.view_init(elev=22, azim=-60)
    save(fig, "projection_geometry.png")


# ------------------------------------------------------------ 2. loss surfaces + GD
def fig_gd():
    rng = np.random.default_rng(0)
    n = 200
    z = rng.standard_normal((n, 2))
    z = (z - z.mean(0)) / z.std(0)
    cases = {"well-conditioned": z @ np.diag([1.0, 0.8]),
             "ill-conditioned": z @ np.diag([1.0, 0.1])}
    w_true = np.array([1.0, 1.0])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    for ax, (name, X) in zip(axes, cases.items()):
        y = X @ w_true + 0.1 * rng.standard_normal(n)
        wstar = lr.ols_qr(X, y)
        lmin, lmax, _, eta = lr.gd_lr_bounds(X)
        kappa = lmax / lmin
        K = 200 if name == "well-conditioned" else 200
        _, path = lr.gradient_descent(X, y, eta, K, w0=wstar + np.array([-1.5, -2.0]))
        g1 = np.linspace(-1.0, 3.0, 200); g2 = np.linspace(-2.2, 4.2, 200) if False else np.linspace(-1.5, 6.0, 200)
        A, B = np.meshgrid(g1 + wstar[0] - 1.0, g2 + wstar[1] - 1.0)
        Lv = np.array([[lr.mse(X, y, np.array([a, b])) for a, b in zip(ra, rb)]
                       for ra, rb in zip(A, B)])
        ax.contour(A, B, Lv, levels=np.geomspace(Lv.min() + 1e-3, Lv.max(), 25), cmap="Blues_r")
        ax.plot(path[:, 0], path[:, 1], ".-", color="tab:red", ms=3, lw=1, label=f"GD, $\\eta=2/(\\lambda_{{max}}+\\lambda_{{min}})$={eta:.2f}")
        ax.plot(*wstar, "k*", ms=14, label=r"$\hat w_{OLS}$")
        ax.plot(*path[0], "go", label="start")
        rate = lr.gd_contraction_rate(X, eta)
        err = np.linalg.norm(path - wstar, axis=1)
        k90 = int(np.argmax(err < 1e-3 * err[0])) if (err < 1e-3 * err[0]).any() else K
        ax.set_title(f"{name}: $\\kappa(X^\\top X)$={kappa:.0f}, rate={rate:.3f}/step\n"
                     f"steps to reduce error 1000x: {k90 if k90 < K else '>'+str(K)}")
        ax.set_xlabel("$w_1$"); ax.set_ylabel("$w_2$"); ax.legend(loc="upper right", fontsize=8)
        ax.set_aspect("equal", adjustable="box")
    save(fig, "gd_loss_surface.png")

    # convergence curves vs theory for several conditioning levels
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for s2, c in zip([0.8, 0.3, 0.1], ["tab:blue", "tab:orange", "tab:red"]):
        X = z @ np.diag([1.0, s2])
        y = X @ w_true + 0.1 * rng.standard_normal(n)
        wstar = lr.ols_qr(X, y)
        lmin, lmax, _, eta = lr.gd_lr_bounds(X)
        rate = lr.gd_contraction_rate(X, eta)
        _, path = lr.gradient_descent(X, y, eta, 300, w0=wstar + np.array([1.0, 1.0]))
        e = np.linalg.norm(path - wstar, axis=1)
        ax.semilogy(e / e[0], c=c, label=f"$\\kappa$={lmax/lmin:.0f}")
        ax.semilogy(rate ** np.arange(301), "--", c=c, lw=1)
    ax.set_ylim(1e-12, 2)
    ax.set_xlabel("iteration k"); ax.set_ylabel(r"$\|w_k-\hat w\|/\|w_0-\hat w\|$")
    ax.set_title("Solid: measured GD error.  Dashed: $((\\kappa-1)/(\\kappa+1))^k$")
    ax.legend()
    save(fig, "gd_convergence.png")


# ------------------------------------------------------------ 3. ridge paths
def fig_ridge():
    rng = np.random.default_rng(1)
    n = 60
    t = rng.standard_normal(n)
    X = np.column_stack([t, t + 0.1 * rng.standard_normal(n), rng.standard_normal(n),
                         rng.standard_normal(n), 0.5 * rng.standard_normal(n)])
    X = (X - X.mean(0)) / X.std(0)
    y = 2 * X[:, 0] + 1 * X[:, 2] - 1.5 * X[:, 3] + 0.5 * rng.standard_normal(n)
    y = y - y.mean()
    lams = np.logspace(-3, 4, 120)
    W = np.array([lr.ridge_closed_form(X, y, l) for l in lams])
    dof = np.array([lr.ridge_effective_dof(X, l) for l in lams])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for j in range(X.shape[1]):
        axes[0].semilogx(lams, W[:, j], label=f"$w_{j+1}$" + (" (collinear pair)" if j < 2 else ""))
    axes[0].axhline(0, c="k", lw=0.5)
    axes[0].set_xlabel(r"$\lambda$"); axes[0].set_ylabel("coefficient")
    axes[0].set_title("Ridge paths: $w_1,w_2$ (nearly collinear) are wild\nat $\\lambda\\to0$ and shrink smoothly"); axes[0].legend(fontsize=8)
    axes[1].semilogx(lams, dof)
    axes[1].set_xlabel(r"$\lambda$"); axes[1].set_ylabel(r"df$(\lambda)=\sum_i s_i^2/(s_i^2+\lambda)$")
    axes[1].set_title("Effective degrees of freedom: 5 -> 0")
    s = np.linalg.svd(X, compute_uv=False)
    for i, si in enumerate(s):
        axes[2].semilogx(lams, si**2 / (si**2 + lams), label=f"$s_{i+1}$={si:.1f}")
    axes[2].set_xlabel(r"$\lambda$"); axes[2].set_ylabel(r"filter $f_i=s_i^2/(s_i^2+\lambda)$")
    axes[2].set_title("SVD shrinkage: small-$s_i$ directions die first"); axes[2].legend(fontsize=8)
    save(fig, "ridge_paths.png")


# ------------------------------------------------------------ 4. bias-variance
def fig_bias_variance():
    f = lambda x: np.sin(np.pi * x)
    degs = list(range(1, 10))
    out = lr.bias_variance_poly(f, degs, n_train=30, sigma=0.3, n_datasets=400, seed=0)
    tot = out["bias2"] + out["var"] + out["noise"]
    lams = np.logspace(-6, 1, 15)
    ridge = [lr.bias_variance_poly(f, [9], n_train=30, sigma=0.3, n_datasets=300, lam=l, seed=1)
             for l in lams]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    ax.semilogy(degs, np.maximum(out["bias2"], 1e-4), "o-", label=r"bias$^2$ (floored at 1e-4: MC noise)")
    ax.semilogy(degs, out["var"], "s-", label="variance")
    ax.semilogy(degs, tot, "k^-", label=r"expected test MSE = bias$^2$+var+$\sigma^2$")
    ax.axhline(out["noise"], c="gray", ls=":", label=r"noise $\sigma^2$ (irreducible)")
    ax.set_xlabel("polynomial degree"); ax.set_ylabel("error (log scale)")
    ax.set_title("OLS: bias falls, variance explodes ($y=\\sin\\pi x+\\varepsilon$, n=30)")
    ax.legend(fontsize=8)
    ax = axes[1]
    b = np.array([o["bias2"][0] for o in ridge]); v = np.array([o["var"][0] for o in ridge])
    ax.loglog(lams, np.maximum(b, 1e-4), "o-", label=r"bias$^2$"); ax.loglog(lams, v, "s-", label="variance")
    ax.loglog(lams, b + v + 0.09, "k^-", label="expected test MSE")
    ax.set_xlabel(r"ridge $\lambda$ (degree-9 model)"); ax.set_ylabel("error")
    ax.set_title("Ridge trades a little bias for a lot of variance"); ax.legend(fontsize=8)
    save(fig, "bias_variance.png")


# ------------------------------------------------------------ 5. polynomial overfit
def fig_poly():
    rng = np.random.default_rng(2)
    f = lambda x: np.sin(np.pi * x)
    x = np.sort(rng.uniform(-1, 1, 15)); y = f(x) + 0.25 * rng.standard_normal(15)
    xt = np.linspace(-1, 1, 400); xte = rng.uniform(-1, 1, 500); yte = f(xte) + 0.25 * rng.standard_normal(500)
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), sharey=True)
    for ax, d in zip(axes, [1, 3, 9, 14]):
        w = lr.ols_svd(lr.poly_features(x, d), y)
        tr = np.mean((y - lr.poly_features(x, d) @ w) ** 2)
        te = np.mean((yte - lr.poly_features(xte, d) @ w) ** 2)
        ax.plot(xt, f(xt), "g--", label="truth"); ax.plot(xt, lr.poly_features(xt, d) @ w, "r", label="fit")
        ax.plot(x, y, "ko", ms=4); ax.set_ylim(-2, 2)
        ax.set_title(f"degree {d}: train MSE {tr:.3f}\ntest MSE {te:.2f}"); ax.set_xlabel("x")
    axes[0].legend(); axes[0].set_ylabel("y")
    save(fig, "poly_overfit.png")


# ------------------------------------------------------------ 6. numerics: kappa^2
def fig_solvers():
    x = np.linspace(0, 1, 50)
    degs = np.arange(2, 16)
    rows = []
    for d in degs:
        X = lr.poly_features(x, d)
        wt = np.ones(d + 1); y = X @ wt
        rows.append([lr.condition_number(X), lr.condition_number(X.T @ X)] +
                    [np.linalg.norm(f(X, y) - wt) / np.linalg.norm(wt)
                     for f in (lr.ols_normal, lr.ols_qr, lr.ols_svd)])
    R = np.array(rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    axes[0].semilogy(degs, R[:, 0], "o-", label=r"$\kappa(X)$")
    axes[0].semilogy(degs, R[:, 1], "s-", label=r"$\kappa(X^\top X)$")
    axes[0].semilogy(degs, R[:, 0] ** 2, "k:", label=r"$\kappa(X)^2$")
    axes[0].set_xlabel("polynomial degree"); axes[0].set_ylabel("condition number")
    axes[0].set_title("Forming $X^\\top X$ squares the condition number"); axes[0].legend()
    for j, nm in enumerate(["normal eq.", "QR", "SVD"]):
        axes[1].semilogy(degs, R[:, 2 + j] + 1e-17, "o-", label=nm)
    axes[1].set_xlabel("polynomial degree"); axes[1].set_ylabel("relative error in $w$")
    axes[1].set_title("Noise-free data, exact $w=\\mathbf{1}$: solver accuracy"); axes[1].legend()
    save(fig, "solver_accuracy.png")


# ------------------------------------------------------------ 7. diagnostics
def fig_diagnostics():
    rng = np.random.default_rng(3)
    n = 80
    x = rng.uniform(0, 4, n)
    cases = {"correct model (linear)": 1 + 2 * x + 0.5 * rng.standard_normal(n),
             "misspecified (truth quadratic)": 1 + 2 * x - 0.6 * x**2 + 0.5 * rng.standard_normal(n),
             "heteroscedastic noise": 1 + 2 * x + (0.1 + 0.4 * x) * rng.standard_normal(n)}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    X = lr.add_intercept(x[:, None])
    for ax, (nm, y) in zip(axes, cases.items()):
        w = lr.ols_qr(X, y); yh = X @ w
        ax.scatter(yh, lr.standardized_residuals(X, y, w), s=12)
        ax.axhline(0, c="k", lw=0.7)
        ax.set_title(f"{nm}\n$R^2$={lr.r_squared(y, yh):.3f}")
        ax.set_xlabel(r"fitted $\hat y$"); ax.set_ylabel("standardised residual")
    save(fig, "residual_diagnostics.png")


if __name__ == "__main__":
    fig_projection(); fig_gd(); fig_ridge(); fig_bias_variance(); fig_poly(); fig_solvers(); fig_diagnostics()
