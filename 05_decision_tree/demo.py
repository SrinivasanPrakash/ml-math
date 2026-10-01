"""End-to-end demo (< 30 s).  Every number quoted in README.md is printed here."""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from tree import (WORKED_X, WORKED_Y_CLASS, WORKED_Y_REG, BaggedTrees, DecisionTreeClassifier,
                  DecisionTreeRegressor, best_split_classification,
                  best_split_classification_naive, best_split_regression,
                  best_split_regression_naive, categorical_split, impurity_from_counts,
                  make_friedman1, make_moons, make_sine, permutation_importance,
                  variance_experiment)


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


# --------------------------------------------------------------------------- #
section("1. Worked example (README section 12): 10 rows, 2 features")
x, y = WORKED_X, WORKED_Y_CLASS
print("class counts (0,1):", np.bincount(y).tolist())
for c in ("gini", "entropy", "misclassification"):
    print(f"  root {c:18s}= {float(impurity_from_counts(np.bincount(y), c)):.4f}")
for c in ("gini", "entropy", "misclassification"):
    for j in range(2):
        g, t = best_split_classification(x[:, j], y, 2, c)
        print(f"  best split on x{j + 1} [{c:17s}]: threshold {t:4.1f}  gain {g:.4f}")
print("\nGini candidate thresholds on x1 (children weighted impurity, gain):")
for t in np.arange(1.5, 10, 1.0):
    L, R = y[x[:, 0] <= t], y[x[:, 0] > t]
    gl = impurity_from_counts(np.bincount(L, minlength=2), "gini")
    gr = impurity_from_counts(np.bincount(R, minlength=2), "gini")
    w = (len(L) * gl + len(R) * gr) / 10
    print(f"  x1 <= {t:4.1f}: L={np.bincount(L, minlength=2).tolist()} R={np.bincount(R, minlength=2).tolist()}"
          f"  weighted={w:.4f}  gain={0.48 - w:.4f}")
clf = DecisionTreeClassifier("gini").fit(x, y)
print("\nFull Gini tree:\n" + clf.export_text(["x1", "x2"]))
print("MDI importances (x1, x2):", np.round(clf.feature_importances_, 4))
reg = DecisionTreeRegressor(max_depth=1).fit(x, WORKED_Y_REG)
g, t = best_split_regression(x[:, 0], WORKED_Y_REG)
L, R = WORKED_Y_REG[x[:, 0] <= t], WORKED_Y_REG[x[:, 0] > t]
print(f"\nRegression: y={WORKED_Y_REG.tolist()}  var(y)={WORKED_Y_REG.var():.4f}"
      f"  SSE(y)={((WORKED_Y_REG - WORKED_Y_REG.mean()) ** 2).sum():.2f}")
print(f"  best split x1 <= {t}: means {L.mean():.2f} / {R.mean():.2f}, SSE {((L - L.mean()) ** 2).sum():.2f}"
      f" + {((R - R.mean()) ** 2).sum():.2f}, variance reduction {g:.4f}"
      f" (closed form {len(L) * len(R) / 100 * (L.mean() - R.mean()) ** 2:.4f})")
print(reg.export_text(["x1", "x2"]))

# --------------------------------------------------------------------------- #
section("2. Fast sweep vs naive O(n^2) split search (one continuous feature, Gini)")
rng = np.random.default_rng(0)
for n in (500, 1000, 2000):
    xs, ys = rng.normal(size=n), rng.integers(0, 2, n)
    t0 = time.perf_counter(); f = best_split_classification(xs, ys, 2); t1 = time.perf_counter()
    g = best_split_classification_naive(xs, ys, 2); t2 = time.perf_counter()
    print(f"  n={n:5d}: fast {1e3 * (t1 - t0):7.2f} ms   naive {1e3 * (t2 - t1):8.1f} ms"
          f"   same split: {np.isclose(f[0], g[0]) and np.isclose(f[1], g[1])}")

# --------------------------------------------------------------------------- #
section("3. Overfitting: train vs test error by depth (noisy moons, 15% label flips)")
rng = np.random.default_rng(1)
Xtr, ytr = make_moons(200, 0.3, rng, flip=0.15)
Xte, yte = make_moons(4000, 0.3, rng, flip=0.15)
for d in (1, 2, 3, 4, 6, 8, 12, None):
    t = DecisionTreeClassifier(max_depth=d).fit(Xtr, ytr)
    print(f"  depth {str(d):>4s}: leaves {t.get_n_leaves():3d}  train err {1 - t.score(Xtr, ytr):.3f}"
          f"  test err {1 - t.score(Xte, yte):.3f}")

# --------------------------------------------------------------------------- #
section("4. Cost-complexity pruning path (validation set chooses alpha)")
Xva, yva = make_moons(1000, 0.3, rng, flip=0.15)
full = DecisionTreeClassifier().fit(Xtr, ytr)
path = full.cost_complexity_path()
val_err = [1 - full.prune(a).score(Xva, yva) for a in path["alphas"]]
print(f"  full tree: {full.get_n_leaves()} leaves, path has {len(path['alphas'])} trees")
print("  alpha is nondecreasing:", bool(np.all(np.diff(path['alphas']) >= 0)),
      "| R(T_k) nondecreasing:", bool(np.all(np.diff(path['impurities']) >= -1e-12)))
k = int(np.argmin(val_err))
best = full.prune(path["alphas"][k])
print(f"  best alpha={path['alphas'][k]:.5f} -> {best.get_n_leaves()} leaves, validation err {val_err[k]:.3f},"
      f" test err {1 - best.score(Xte, yte):.3f}  (unpruned test err {1 - full.score(Xte, yte):.3f})")
print(best.export_text(["x", "y"]))

# --------------------------------------------------------------------------- #
section("5. Categorical feature: optimal subset split via ordering by mean response")
rng = np.random.default_rng(2)
cat = rng.integers(0, 6, 300).astype(float)
means = np.array([0, 3, 1, 5, 2, 4.0])
yc = means[cat.astype(int)] + rng.normal(0, 0.5, 300)
gain, S = categorical_split(cat, yc, "regression")
print(f"  categories ordered by mean: {np.argsort(means).tolist()} -> best subset {sorted(S)}, gain {gain:.4f}")
print(f"  scanned 5 prefix splits instead of 2^5-1 = 31 subsets")

# --------------------------------------------------------------------------- #
section("6. MDI bias: informative binary feature vs pure-noise continuous feature")
rng = np.random.default_rng(0)
n = 300
sig = rng.integers(0, 2, n).astype(float)
yy = np.where(rng.random(n) < 0.8, sig, 1 - sig).astype(int)
X = np.c_[sig, rng.normal(size=n), rng.integers(0, 2, n)]
t = DecisionTreeClassifier().fit(X, yy)
sg = rng.integers(0, 2, 5000).astype(float)
Xt = np.c_[sg, rng.normal(size=5000), rng.integers(0, 2, 5000)]
yt = np.where(rng.random(5000) < 0.8, sg, 1 - sg).astype(int)
print("  features: [signal(binary), noise(continuous), noise(binary)]")
print("  MDI (train, unrestricted tree):", np.round(t.feature_importances_, 3))
print("  permutation importance (test):  ", np.round(permutation_importance(t, Xt, yt, 5), 3))

# --------------------------------------------------------------------------- #
section("7. Bagging: variance of predictions across fresh training sets (Friedman #1)")
t0 = time.time()
xt = make_friedman1(40, 0, np.random.default_rng(99))[0]
res = variance_experiment(lambda r: make_friedman1(80, 1.0, r), xt,
                          [{}, {"max_features": 2}], n_estimators=20, n_datasets=25, seed=0)
for name, r in zip(("bagging", "random forest (m=2)"), res):
    v = r["var_by_B"]
    print(f"  {name:20s}: Var(B=1)={v[0]:.3f}  Var(B=5)={v[4]:.3f}  Var(B=20)={v[-1]:.3f}"
          f"   rho_hat={r['rho']:.3f}  rho*s2+(1-rho)*s2/20={r['rho'] * r['sigma2'] + (1 - r['rho']) * r['sigma2'] / 20:.3f}")
print(f"  ({time.time() - t0:.1f} s)")
