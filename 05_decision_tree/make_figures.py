"""Regenerate every figure in figures/ deterministically (fixed seeds).

Run:  python make_figures.py        (about 1 minute; the timing figure depends on your CPU)
"""
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))
from tree import (WORKED_X, WORKED_Y_CLASS, BaggedTrees, DecisionTreeClassifier,
                  DecisionTreeRegressor, best_split_classification,
                  best_split_classification_naive, entropy, gini, impurity_from_counts,
                  make_friedman1, make_moons, make_sine, misclassification,
                  permutation_importance, variance_experiment)

OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)
C0, C1, C2, C3, GREY = "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#7f7f7f"
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.25, "font.size": 10})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=130)
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- 1 impurity
def fig_impurity():
    p = np.linspace(0, 1, 501)
    P = np.c_[1 - p, p]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(p, entropy(P), C0, lw=2, label="entropy $H$ (bits)")
    ax[0].plot(p, entropy(P) / 2, C0, lw=1.5, ls="--", label="entropy / 2 (rescaled)")
    ax[0].plot(p, gini(P), C1, lw=2, label="Gini $2p(1-p)$")
    ax[0].plot(p, misclassification(P), C2, lw=2, label="misclassification $1-\\max(p,1-p)$")
    ax[0].annotate("kink at $p=1/2$: piecewise linear,\nonly sees the majority class", (0.5, 0.5), (0.08, 0.72),
                   arrowprops=dict(arrowstyle="->", color=C2), color=C2)
    ax[0].set_xlabel("$p$ = fraction of class 1 in the node"); ax[0].set_ylabel("impurity")
    ax[0].set_title("Two-class impurity measures (all concave)")
    ax[0].legend(loc="upper right", fontsize=8, framealpha=0.95)
    # Jensen picture
    h = lambda q: entropy(np.stack([1 - np.asarray(q), np.asarray(q)], -1))
    pl, pr, lam = 0.15, 0.85, 0.4
    pm = lam * pl + (1 - lam) * pr
    ax[1].plot(p, h(p), C0, lw=2, label="$H(p)$")
    ax[1].plot([pl, pr], [h(pl), h(pr)], "k--", lw=1.2, label="chord")
    ax[1].plot([pl, pr, pm], [h(pl), h(pr), h(pm)], "ko")
    chord = lam * h(pl) + (1 - lam) * h(pr)
    ax[1].plot([pm, pm], [chord, h(pm)], C1, lw=3)
    ax[1].annotate(f"information gain\n$= H(p_t)-\\sum\\frac{{n_c}}{{n}}H(p_c)$\n$= {h(pm) - chord:.3f}\\geq 0$",
                   (pm, (chord + h(pm)) / 2), (0.62, 0.30), color=C1,
                   arrowprops=dict(arrowstyle="->", color=C1))
    ax[1].text(pl, h(pl) + 0.04, "$p_L$", ha="center"); ax[1].text(pr, h(pr) + 0.04, "$p_R$", ha="center")
    ax[1].text(pm, h(pm) + 0.04, "$p_t=\\lambda p_L+(1-\\lambda)p_R$", ha="center")
    ax[1].set_xlabel("$p$"); ax[1].set_ylabel("entropy (bits)")
    ax[1].set_title("Jensen: parent lies above the chord of its children")
    ax[1].legend(loc="lower center", fontsize=8)
    save(fig, "impurity_curves.png")


# ---------------------------------------------------------------- 2 worked
def fig_worked():
    x = WORKED_X[:, 0]
    thr = np.arange(1.5, 10, 1.0)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for c, col in (("gini", C1), ("entropy", C0), ("misclassification", C2)):
        g = [best_split_classification(x, WORKED_Y_CLASS, 2, c)]
        gains = []
        for t in thr:
            L, R = WORKED_Y_CLASS[x <= t], WORKED_Y_CLASS[x > t]
            parent = impurity_from_counts(np.bincount(WORKED_Y_CLASS), c)
            w = (len(L) * impurity_from_counts(np.bincount(L, minlength=2), c)
                 + len(R) * impurity_from_counts(np.bincount(R, minlength=2), c)) / 10
            gains.append(parent - w)
        ax[1].plot(thr, gains, "o-", color=col, label=c)
        ax[1].plot(thr[int(np.argmax(gains))], max(gains), "*", ms=16, color=col, mec="k")
    ax[1].set_xlabel("threshold on $x_1$"); ax[1].set_ylabel("impurity decrease $\\Delta$")
    ax[1].set_title("Worked example: gain per threshold (star = chosen split)")
    ax[1].legend()
    for k, mk, col in ((0, "o", C0), (1, "s", C1)):
        m = WORKED_Y_CLASS == k
        ax[0].scatter(WORKED_X[m, 0], WORKED_X[m, 1], marker=mk, c=col, s=70, label=f"class {k}")
    for i, (a, b) in enumerate(WORKED_X):
        ax[0].annotate(str(i + 1), (a, b), (4, 4), textcoords="offset points", fontsize=8)
    ax[0].axvline(5.5, color="k", lw=1.5); ax[0].plot([0.5, 5.5], [2.5, 2.5], "k--", lw=1)
    ax[0].text(5.6, 10.2, "$x_1\\leq 5.5$", fontsize=9); ax[0].text(0.7, 2.7, "left node: $x_2\\leq 2$ isolates row 4", fontsize=8)
    ax[0].set_xlabel("$x_1$"); ax[0].set_ylabel("$x_2$"); ax[0].set_title("The 10 rows and the Gini tree")
    ax[0].legend(loc="lower right")
    save(fig, "worked_example.png")


# ---------------------------------------------------------------- 3 regions
def fig_regions():
    rng = np.random.default_rng(1)
    X, y = make_moons(200, 0.3, rng, flip=0.15)
    gx, gy = np.meshgrid(np.linspace(-1.6, 2.6, 300), np.linspace(-1.1, 1.6, 300))
    G = np.c_[gx.ravel(), gy.ravel()]
    depths = [1, 2, 3, 5, 8, None]
    fig, axs = plt.subplots(2, 3, figsize=(12, 7.5))
    for ax, d in zip(axs.ravel(), depths):
        t = DecisionTreeClassifier(max_depth=d).fit(X, y)
        ax.contourf(gx, gy, t.predict(G).reshape(gx.shape), levels=[-.5, .5, 1.5],
                    colors=["#c6dbef", "#fcbba1"], alpha=0.9)
        ax.scatter(X[y == 0, 0], X[y == 0, 1], c=C0, s=14, edgecolor="k", lw=0.3)
        ax.scatter(X[y == 1, 0], X[y == 1, 1], c=C1, s=14, edgecolor="k", lw=0.3)
        ax.set_title(f"max_depth={d}: {t.get_n_leaves()} leaves, train acc {t.score(X, y):.2f}")
        ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.grid(False)
    fig.suptitle("Axis-aligned rectangles: deeper trees carve out single noisy points", y=1.0)
    save(fig, "decision_regions.png")


# ---------------------------------------------------------------- 4 depth curves
def fig_depth():
    depths = list(range(1, 16))
    tr, te = [], []
    for s in range(20):
        rng = np.random.default_rng(100 + s)
        X, y = make_moons(200, 0.3, rng, flip=0.15)
        Xt, yt = make_moons(2000, 0.3, rng, flip=0.15)
        r = [DecisionTreeClassifier(max_depth=d).fit(X, y) for d in depths]
        tr.append([1 - t.score(X, y) for t in r]); te.append([1 - t.score(Xt, yt) for t in r])
    tr, te = np.array(tr), np.array(te)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for a, col, lab in ((tr, C0, "train error"), (te, C1, "test error")):
        ax.plot(depths, a.mean(0), "o-", color=col, label=lab)
        ax.fill_between(depths, a.mean(0) - a.std(0), a.mean(0) + a.std(0), color=col, alpha=0.15)
    ax.axhline(0.15, color=GREY, ls=":"); ax.text(5.5, 0.155, "15% of labels were flipped: error cannot be much lower", color=GREY)
    k = int(te.mean(0).argmin())
    ax.annotate("sweet spot", (depths[k], te.mean(0)[k]), (depths[k] + 1.5, 0.05),
                arrowprops=dict(arrowstyle="->"))
    ax.annotate("memorises noise:\ntrain 0, test worse", (13, te.mean(0)[12]), (6.2, 0.345),
                arrowprops=dict(arrowstyle="->"))
    ax.set_xlabel("max depth"); ax.set_ylabel("misclassification rate")
    ax.set_title("Overfitting vs depth (mean $\\pm$ sd over 20 datasets, n=200)")
    ax.legend(); save(fig, "train_test_vs_depth.png")


# ---------------------------------------------------------------- 5 pruning
def fig_pruning():
    rng = np.random.default_rng(1)
    Xtr, ytr = make_moons(200, 0.3, rng, flip=0.15)
    Xte, yte = make_moons(4000, 0.3, rng, flip=0.15)
    Xva, yva = make_moons(1000, 0.3, rng, flip=0.15)
    full = DecisionTreeClassifier().fit(Xtr, ytr)
    p = full.cost_complexity_path()
    pr = [full.prune(a) for a in p["alphas"]]
    tr = [1 - t.score(Xtr, ytr) for t in pr]
    va = [1 - t.score(Xva, yva) for t in pr]
    te = [1 - t.score(Xte, yte) for t in pr]
    k = int(np.argmin(va))
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    ax[0].step(p["alphas"], p["n_leaves"], where="post", color=C0, marker="o", ms=4)
    ax[0].set_xlabel("$\\alpha$ (price per leaf)"); ax[0].set_ylabel("leaves $|T_\\alpha|$")
    ax[0].set_title("Weakest-link path: larger $\\alpha$, smaller tree")
    ax[1].plot(p["n_leaves"], p["impurities"], "o-", color=C3, ms=4)
    ax[1].invert_xaxis(); ax[1].set_xlabel("leaves (pruning goes left to right)")
    ax[1].set_ylabel("$R(T_k)$ (weighted Gini of leaves)")
    ax[1].set_title("$R(T_k)$ rises monotonically as we prune")
    ax[2].plot(p["n_leaves"], tr, "o-", color=C0, ms=4, label="train")
    ax[2].plot(p["n_leaves"], va, "s-", color=C2, ms=4, label="validation")
    ax[2].plot(p["n_leaves"], te, "^-", color=C1, ms=4, label="test")
    ax[2].axvline(p["n_leaves"][k], color=GREY, ls="--")
    ax[2].annotate(f"chosen: {p['n_leaves'][k]} leaves", (p["n_leaves"][k], va[k]),
                   (p["n_leaves"][k] + 14, va[k] - 0.08), arrowprops=dict(arrowstyle="->"))
    ax[2].invert_xaxis(); ax[2].set_xlabel("leaves"); ax[2].set_ylabel("error")
    ax[2].set_title("Choosing $\\alpha$ on a validation set"); ax[2].legend()
    save(fig, "pruning_path.png")


# ---------------------------------------------------------------- 6 bagging
def fig_bagging():
    xt = make_friedman1(40, 0, np.random.default_rng(99))[0]
    t0 = time.time()
    res = variance_experiment(lambda r: make_friedman1(80, 1.0, r), xt,
                              [{}, {"max_features": 2}], n_estimators=30, n_datasets=40, seed=0)
    print(f"  bagging experiment {time.time() - t0:.0f}s")
    B = np.arange(1, 31)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    for r, col, lab in ((res[0], C0, "bagging (all features)"), (res[1], C2, "random forest ($m=2$ of 5)")):
        ax[0].plot(B, r["var_by_B"], "o", color=col, ms=4, label=f"{lab}, $\\hat\\rho$={r['rho']:.2f}")
        ax[0].plot(B, r["rho"] * r["sigma2"] + (1 - r["rho"]) * r["sigma2"] / B, "-", color=col, lw=1)
        ax[0].axhline(r["rho"] * r["sigma2"], color=col, ls=":", lw=1)
    ax[0].set_xlabel("number of trees $B$"); ax[0].set_ylabel("prediction variance over training sets")
    ax[0].set_title("Variance $=\\rho\\sigma^2+\\frac{1-\\rho}{B}\\sigma^2$ (lines) vs measured (dots)\ndotted: floor $\\rho\\sigma^2$")
    ax[0].legend(fontsize=8)
    # 1-D spaghetti
    xg = np.linspace(0, 1, 200)[:, None]
    for j, (title, mk) in enumerate((("single unpruned tree", lambda s, X, y: DecisionTreeRegressor().fit(X, y)),
                                      ("bagged, $B=50$", lambda s, X, y: BaggedTrees(DecisionTreeRegressor, 50, s).fit(X, y)))):
        a = ax[1 + j]
        a.plot(xg[:, 0], np.sin(2 * np.pi * xg[:, 0]), "k", lw=2.5, label="truth")
        for s in range(8):
            X, y = make_sine(60, 0.3, np.random.default_rng(500 + s))
            a.plot(xg[:, 0], mk(s, X, y).predict(xg), color=C0 if j == 0 else C2, lw=1, alpha=0.65)
        a.set_ylim(-1.8, 1.8); a.set_xlabel("$x$"); a.set_ylabel("$\\hat f(x)$")
        a.set_title(f"{title}: 8 different training sets"); a.legend(loc="upper right")
    save(fig, "bagging_variance.png")


# ---------------------------------------------------------------- 7 timing
def fig_timing():
    rng = np.random.default_rng(0)
    ns = [400, 800, 1600, 3200, 6400, 12800]
    fast, naive = [], []

    def best_time(f, reps):
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter(); f(); ts.append(time.perf_counter() - t0)
        return min(ts)

    for n in ns:
        x, y = rng.normal(size=n), rng.integers(0, 2, n)
        fast.append(best_time(lambda: best_split_classification(x, y, 2), 15))
        naive.append(best_time(lambda: best_split_classification_naive(x, y, 2), 2))
    ns_a = np.array(ns, float)
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    ax.loglog(ns, naive, "o-", color=C1, label="naive: recompute per threshold")
    ax.loglog(ns, fast, "s-", color=C0, label="sort + sweep")
    ax.loglog(ns, naive[-1] * (ns_a / ns_a[-1]) ** 2, ":", color=C1, label="slope 2 ($n^2$)")
    ax.loglog(ns, fast[-1] * (ns_a / ns_a[-1]), ":", color=C0, label="slope 1 ($n$, $\\log n$ factor invisible)")
    ax.set_xlabel("samples in node $n$"); ax.set_ylabel("seconds for one feature (best of repeats)")
    ax.set_title("Best split of one continuous feature"); ax.legend(fontsize=8)
    save(fig, "split_timing.png")


# ---------------------------------------------------------------- 8 MDI bias
def fig_mdi():
    mdi, perm = [], []
    for s in range(30):
        rng = np.random.default_rng(s)
        n = 300
        sig = rng.integers(0, 2, n).astype(float)
        y = np.where(rng.random(n) < 0.8, sig, 1 - sig).astype(int)
        X = np.c_[sig, rng.normal(size=n), rng.integers(0, 2, n)]
        t = DecisionTreeClassifier().fit(X, y)
        sg = rng.integers(0, 2, 2000).astype(float)
        Xt = np.c_[sg, rng.normal(size=2000), rng.integers(0, 2, 2000)]
        yt = np.where(rng.random(2000) < 0.8, sg, 1 - sg).astype(int)
        mdi.append(t.feature_importances_); perm.append(permutation_importance(t, Xt, yt, 3, s))
    mdi, perm = np.array(mdi), np.array(perm)
    names = ["signal\n(binary)", "noise\n(continuous)", "noise\n(binary)"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for a, d, title, yl in ((ax[0], mdi, "MDI on training data (normalised)", "importance"),
                            (ax[1], perm, "Permutation importance on held-out data", "accuracy drop")):
        a.bar(names, d.mean(0), yerr=d.std(0), color=[C2, C1, C1], capsize=4)
        a.set_title(title); a.set_ylabel(yl)
    ax[0].set_ylim(0, 0.9)
    ax[0].text(0.5, 0.98, "unrestricted tree gives the useless\ncontinuous feature the most credit", ha="center",
               transform=ax[0].transAxes, color=C1, va="top")
    fig.suptitle("MDI is biased toward features with many split points (30 repetitions, n=300)")
    save(fig, "mdi_bias.png")


# ---------------------------------------------------------------- 9 regression
def fig_regression():
    rng = np.random.default_rng(3)
    X, y = make_sine(60, 0.3, rng)
    xg = np.linspace(0, 1, 400)[:, None]
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
    for ax, d in zip(axs, [1, 2, 4, None]):
        t = DecisionTreeRegressor(max_depth=d).fit(X, y)
        ax.scatter(X[:, 0], y, s=12, c=GREY)
        ax.plot(xg[:, 0], np.sin(2 * np.pi * xg[:, 0]), "k--", lw=1, label="truth")
        ax.plot(xg[:, 0], t.predict(xg), C1, lw=2, label="tree")
        ax.set_title(f"depth {d}: {t.get_n_leaves()} leaves (piecewise constant)")
        ax.set_xlabel("$x$")
    axs[0].set_ylabel("$y$"); axs[0].legend()
    save(fig, "regression_tree_fits.png")


if __name__ == "__main__":
    fig_impurity(); fig_worked(); fig_regions(); fig_depth(); fig_pruning()
    fig_bagging(); fig_timing(); fig_mdi(); fig_regression()
