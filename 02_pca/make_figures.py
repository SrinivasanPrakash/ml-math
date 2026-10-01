"""Regenerate every figure in figures/ (deterministic)."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")  # small matrices: threads only add overhead
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))
from pca import (PCA, center, covariance, pca_eig, pca_svd, standardize, pca_power, make_glyph_data, choose_k)

OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.25, "font.size": 10})


def data2d(n=200, seed=0):
    rng = np.random.default_rng(seed)
    t = np.deg2rad(35)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    return (rng.standard_normal((n, 2)) * [2.5, 0.8]) @ R.T + [3, 2]


def fig_ellipse():
    X = data2d()
    p = PCA(2).fit(X)
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.scatter(*X.T, s=12, alpha=0.5, label="data")
    ax.plot(*p.mean_, "k+", ms=14, mew=2, label="mean")
    th = np.linspace(0, 2 * np.pi, 200)
    for r, ls in [(1, "-"), (2, "--")]:
        E = p.mean_ + (np.sqrt(p.explained_variance_) * r * np.c_[np.cos(th), np.sin(th)]) @ p.components_
        ax.plot(*E.T, "gray", ls=ls, lw=1, label=f"{r}-sigma ellipse")
    for i, c in enumerate(["C3", "C2"]):
        v = p.components_[i] * np.sqrt(p.explained_variance_[i]) * 2
        ax.annotate("", p.mean_ + v, p.mean_, arrowprops=dict(arrowstyle="->", color=c, lw=2.5))
        ax.text(*(p.mean_ + 1.12 * v), f"PC{i+1}\n$\\lambda$={p.explained_variance_[i]:.2f}", color=c, ha="center")
    ax.set_aspect("equal"); ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    ax.set_title("Principal axes = eigenvectors of covariance;\nsemi-axis length = $\\sqrt{\\lambda}$")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT}/ellipse_axes.png"); plt.close(fig)


def fig_variance_vs_error():
    X = data2d(60, seed=2)
    Xc, mu = center(X)
    C = covariance(Xc)
    n = len(X)
    angles = np.linspace(0, np.pi, 361)
    var = np.array([np.array([np.cos(a), np.sin(a)]) @ C @ np.array([np.cos(a), np.sin(a)]) for a in angles])
    tot = np.trace(C)
    err = tot - var
    best = angles[np.argmax(var)]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, a, title in [(axs[0], np.deg2rad(110), "Poor direction: small variance, long residuals"),
                         (axs[1], best, "PC1: max variance = min residuals")]:
        w = np.array([np.cos(a), np.sin(a)])
        proj = Xc @ w
        P = np.outer(proj, w) + mu
        ax.scatter(*X.T, s=14, zorder=3)
        for x, q in zip(X, P):
            ax.plot([x[0], q[0]], [x[1], q[1]], "C3", lw=0.7, alpha=0.8)
        ax.scatter(*P.T, s=8, c="k", zorder=4)
        L = np.array([-1, 1])[:, None] * 6 * w + mu
        ax.plot(*L.T, "k-", lw=1)
        v = (proj**2).sum() / (n - 1); e = ((Xc - np.outer(proj, w)) ** 2).sum() / (n - 1)
        ax.set_title(f"{title}\nvar={v:.2f}, err={e:.2f}, sum={v+e:.2f}", fontsize=9)
        ax.set_aspect("equal"); ax.set_xlim(X[:, 0].min() - 1, X[:, 0].max() + 1); ax.set_ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)
        ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    ax = axs[2]
    ax.plot(np.rad2deg(angles), var, label="variance of projection $w^\\top C w$")
    ax.plot(np.rad2deg(angles), err, label="mean squared residual")
    ax.plot(np.rad2deg(angles), var + err, "k--", label="sum = trace(C) (constant)")
    ax.axvline(np.rad2deg(best), color="gray", ls=":")
    ax.text(np.rad2deg(best) + 3, tot * 0.5, "PC1 angle:\nmax var & min err\nat the SAME angle", fontsize=9)
    ax.set_xlabel("projection direction angle (deg)"); ax.set_ylabel("value"); ax.legend(fontsize=8, loc="center right")
    ax.set_title("Pythagoras: variance + error = constant")
    fig.tight_layout(); fig.savefig(f"{OUT}/variance_vs_error.png"); plt.close(fig)


def fig_scree():
    G = make_glyph_data()
    Xc, _ = center(G)
    vals, _ = pca_svd(Xc)
    r = np.cumsum(vals) / vals.sum()
    k95 = choose_k(vals, 0.95)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    m = 25
    axs[0].plot(range(1, m + 1), vals[:m], "o-")
    axs[0].axvline(5.5, color="C3", ls="--"); axs[0].text(6, vals[0] * 0.6, "elbow: true rank = 5\n(noise floor after)", color="C3")
    axs[0].set_xlabel("component index $i$"); axs[0].set_ylabel("eigenvalue $\\lambda_i$"); axs[0].set_title("Scree plot (synthetic glyphs, 144 pixels)")
    axs[1].plot(range(1, m + 1), r[:m], "o-")
    axs[1].axhline(0.95, color="gray", ls=":"); axs[1].axvline(k95, color="C2", ls="--")
    axs[1].text(k95 + 0.5, 0.5, f"k={k95} reaches 95%", color="C2")
    axs[1].set_xlabel("number of components $k$"); axs[1].set_ylabel("cumulative explained variance ratio"); axs[1].set_title("Cumulative explained variance")
    fig.tight_layout(); fig.savefig(f"{OUT}/scree.png"); plt.close(fig)


def fig_reconstruction():
    G = make_glyph_data(seed=0)
    idx = [0, 1, 2, 3]
    ks = [1, 2, 3, 5, 10, 30]
    fig, axs = plt.subplots(len(idx), len(ks) + 1, figsize=(1.7 * (len(ks) + 1), 1.8 * len(idx)))
    for j, k in enumerate([None] + ks):
        if k is None:
            R = G
        else:
            p = PCA(k).fit(G); R = p.inverse_transform(p.transform(G))
        for i, s in enumerate(idx):
            ax = axs[i, j]; ax.imshow(R[s].reshape(12, 12), cmap="gray_r", vmin=-0.2, vmax=1.2); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            if i == 0:
                ax.set_title("original" if k is None else f"k={k}", fontsize=9)
    fig.suptitle("Rank-k PCA reconstruction of noisy glyphs (true rank 5): structure appears by k=5, then only noise is added", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{OUT}/reconstruction.png"); plt.close(fig)


def fig_standardisation():
    rng = np.random.default_rng(0)
    n = 200
    h = rng.standard_normal(n)
    height_cm = 170 + 8 * h + 2 * rng.standard_normal(n)
    weight_kg = 70 + 10 * h + 4 * rng.standard_normal(n)
    X = np.c_[height_cm, weight_kg * 1000]  # weight in grams: arbitrary unit change
    Z, _, _ = standardize(X)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, D, title in [(axs[0], X, "Raw (height cm, weight g)"), (axs[1], Z, "Standardised (z-scores)")]:
        p = PCA(2).fit(D)
        ax.scatter(*D.T, s=10, alpha=0.5)
        for i, c in enumerate(["C3", "C2"]):
            sc = np.sqrt(p.explained_variance_[i]) * 2.5
            ax.annotate("", p.mean_ + p.components_[i] * sc, p.mean_, arrowprops=dict(arrowstyle="->", color=c, lw=2.5))
        ax.set_title(f"{title}\nPC1 = ({p.components_[0,0]:.3f}, {p.components_[0,1]:.3f}), ratio={p.explained_variance_ratio_[0]:.3f}", fontsize=9)
    axs[0].set_xlabel("height (cm)"); axs[0].set_ylabel("weight (g)"); axs[1].set_xlabel("height (z)"); axs[1].set_ylabel("weight (z)")
    axs[1].set_aspect("equal")
    fig.suptitle("Changing a unit rotates PC1 to the large-variance axis; standardising removes the unit dependence", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{OUT}/standardisation.png"); plt.close(fig)


def fig_pitfalls():
    rng = np.random.default_rng(1)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
    # outlier
    X = rng.standard_normal((100, 2)) * [3, 0.7]
    Xo = np.vstack([X, [[0, 30]]])
    for D, c, lab in [(X, "C0", "clean"), (Xo, "C3", "with 1 outlier")]:
        p = PCA(1).fit(D); v = p.components_[0]
        axs[0].plot(*(np.array([-1, 1])[:, None] * 8 * v + p.mean_).T, c, lw=2, label=f"PC1 {lab}")
    axs[0].set_ylim(-4, 32); axs[0].scatter(*X.T, s=10, c="C0"); axs[0].scatter(*Xo[-1], s=60, c="C3", marker="x")
    axs[0].legend(fontsize=8); axs[0].set_title("Outliers: one point swings PC1"); axs[0].set_aspect("equal")
    # nonlinear: noisy circle
    t = rng.uniform(0, 2 * np.pi, 300)
    C = np.c_[np.cos(t), np.sin(t)] + 0.05 * rng.standard_normal((300, 2))
    p = PCA(2).fit(C)
    axs[1].scatter(*C.T, s=8, c=t, cmap="twilight")
    for i in range(2):
        axs[1].plot(*(np.array([-1, 1])[:, None] * 1.3 * p.components_[i]).T, "k-")
    axs[1].set_title(f"Non-linear: circle is 1-D, but eigenvalues are\n{p.explained_variance_[0]:.2f}, {p.explained_variance_[1]:.2f} (no preferred axis)", fontsize=9); axs[1].set_aspect("equal")
    # sign ambiguity: raw power-iteration PC1 from 40 random starts (no sign convention)
    X = data2d(60, seed=4)
    Xc, _ = center(X)
    V = np.array([pca_power(Xc, 1, seed=s)[1][0] for s in range(40)])
    axs[2].scatter(*V.T, s=25, alpha=0.6)
    axs[2].plot(*PCA(1).fit(X).components_[0], "r*", ms=14, label="after fix_signs")
    axs[2].legend(fontsize=8)
    axs[2].set_title("Sign ambiguity: PC1 from 40 random starts\nlands on $+v$ or $-v$; fix a convention before comparing", fontsize=9)
    axs[2].set_xlabel("v[0]"); axs[2].set_ylabel("v[1]"); axs[2].set_aspect("equal")
    for a in axs[:2]: a.set_xlabel("$x_1$"); a.set_ylabel("$x_2$")
    fig.tight_layout(); fig.savefig(f"{OUT}/pitfalls.png"); plt.close(fig)


if __name__ == "__main__":
    for f in [fig_ellipse, fig_variance_vs_error, fig_scree, fig_reconstruction, fig_standardisation, fig_pitfalls]:
        f(); print("done", f.__name__)
