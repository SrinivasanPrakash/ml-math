"""Decision trees from scratch (NumPy only).

Equation numbers "(n)" refer to README.md in the parent directory.

Contents
--------
* impurity measures             entropy (1), Gini (2), misclassification (3)
* split search                  fast O(n log n) sweep (10) and naive O(n^2) reference
* categorical splits            Fisher / Breiman ordering trick (README section 9)
* DecisionTreeClassifier / DecisionTreeRegressor (recursive partitioning, section 3)
* cost-complexity pruning       weakest link (11)-(12)
* feature importance            MDI (13) and permutation importance
* BaggedTrees                   bootstrap aggregation, variance (14)
* toy data generators           used by tests, demo and figures
"""
import sys
from copy import deepcopy

import numpy as np

# --------------------------------------------------------------------------- #
# 1. Impurity measures.  All take class probabilities p (last axis = classes).
# --------------------------------------------------------------------------- #


def entropy(p):
    """Shannon entropy in bits, H(p) = -sum_k p_k log2 p_k   (eq. 1), 0 log 0 = 0."""
    p = np.asarray(p, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0, p * np.log2(p), 0.0)
    return -terms.sum(axis=-1)


def gini(p):
    """Gini impurity, G(p) = 1 - sum_k p_k^2 = sum_k p_k (1 - p_k)   (eq. 2)."""
    p = np.asarray(p, dtype=float)
    return 1.0 - (p ** 2).sum(axis=-1)


def misclassification(p):
    """Misclassification error, E(p) = 1 - max_k p_k   (eq. 3)."""
    p = np.asarray(p, dtype=float)
    return 1.0 - p.max(axis=-1)


CRITERIA = {"entropy": entropy, "gini": gini, "misclassification": misclassification}


def impurity_from_counts(counts, criterion="gini"):
    """Impurity of class-count vectors (last axis = classes); empty nodes get 0."""
    counts = np.asarray(counts, dtype=float)
    n = counts.sum(axis=-1)
    p = counts / np.where(n > 0, n, 1.0)[..., None]
    return CRITERIA[criterion](p) * (n > 0)


# --------------------------------------------------------------------------- #
# 2. Split search on ONE continuous feature.
#    Both searchers return (gain, threshold) or None; gain = parent impurity
#    minus size-weighted child impurity, i.e. Delta in eq. (5).
# --------------------------------------------------------------------------- #

_TOL = 1e-12


def _midpoint(lo, hi):
    """Threshold between two adjacent distinct sorted values; guards float rounding."""
    thr = 0.5 * (lo + hi)
    return np.where(thr >= hi, lo, thr)


def _pick(child, parent, lo, hi):
    """Choose the smallest-child-impurity candidate (first one within tolerance)."""
    best = child.min()
    if not np.isfinite(best):
        return None
    i = int(np.argmax(child <= best + _TOL * max(1.0, abs(parent))))
    return max(parent - best, 0.0), float(_midpoint(lo[i], hi[i]))


def best_split_classification(x, y, n_classes, criterion="gini", min_samples_leaf=1):
    """Fast exact split search: sort once, sweep with running class counts.

    Moving the split point one sample to the right adds that sample's one-hot
    label to the left counts and removes it from the right counts (eq. 10);
    ``np.cumsum`` performs all those running updates at once.
    Cost: O(n log n) for the sort + O(n K) for the sweep.
    """
    n = len(x)
    if n < 2:
        return None
    order = np.argsort(x, kind="stable")
    xs, ys = x[order], y[order]
    nl = np.arange(1, n)
    valid = (xs[1:] > xs[:-1]) & (nl >= min_samples_leaf) & (n - nl >= min_samples_leaf)
    if not valid.any():
        return None
    onehot = np.eye(n_classes)[ys]
    left = np.cumsum(onehot, axis=0)[:-1]            # counts left of split i
    total = onehot.sum(axis=0)
    right = total - left                             # eq. (10)
    child = (nl * impurity_from_counts(left, criterion)
             + (n - nl) * impurity_from_counts(right, criterion)) / n
    child = np.where(valid, child, np.inf)
    parent = float(impurity_from_counts(total, criterion))
    return _pick(child, parent, xs[:-1], xs[1:])


def best_split_classification_naive(x, y, n_classes, criterion="gini", min_samples_leaf=1):
    """Reference implementation: recompute both children from scratch per threshold. O(n^2)."""
    n = len(x)
    vals = np.unique(x)
    if len(vals) < 2:
        return None
    lo, hi = vals[:-1], vals[1:]
    thr = _midpoint(lo, hi)
    child = np.full(len(thr), np.inf)
    for i, t in enumerate(thr):
        mask = x <= t
        nl = int(mask.sum())
        if nl < min_samples_leaf or n - nl < min_samples_leaf:
            continue
        cl = np.bincount(y[mask], minlength=n_classes)
        cr = np.bincount(y[~mask], minlength=n_classes)
        child[i] = (nl * impurity_from_counts(cl, criterion)
                    + (n - nl) * impurity_from_counts(cr, criterion)) / n
    parent = float(impurity_from_counts(np.bincount(y, minlength=n_classes), criterion))
    return _pick(child, parent, lo, hi)


def best_split_regression(x, y, min_samples_leaf=1):
    """Fast exact variance-reduction split: sort once, running sums S1, S2 (eq. 10).

    SSE of a group = sum y^2 - (sum y)^2 / m, so only cumulative sums are needed.
    y is centred first to avoid catastrophic cancellation.
    """
    n = len(x)
    if n < 2:
        return None
    order = np.argsort(x, kind="stable")
    xs = x[order]
    ys = y[order] - y.mean()
    nl = np.arange(1, n)
    valid = (xs[1:] > xs[:-1]) & (nl >= min_samples_leaf) & (n - nl >= min_samples_leaf)
    if not valid.any():
        return None
    s1 = np.cumsum(ys)[:-1]
    s2 = np.cumsum(ys ** 2)[:-1]
    t1, t2 = ys.sum(), (ys ** 2).sum()
    sse_l = s2 - s1 ** 2 / nl
    sse_r = (t2 - s2) - (t1 - s1) ** 2 / (n - nl)
    child = np.where(valid, (sse_l + sse_r) / n, np.inf)
    parent = t2 / n - (t1 / n) ** 2
    return _pick(child, parent, xs[:-1], xs[1:])


def best_split_regression_naive(x, y, min_samples_leaf=1):
    """Reference implementation, O(n^2): recompute means and SSEs for each threshold."""
    n = len(x)
    vals = np.unique(x)
    if len(vals) < 2:
        return None
    lo, hi = vals[:-1], vals[1:]
    thr = _midpoint(lo, hi)
    child = np.full(len(thr), np.inf)
    for i, t in enumerate(thr):
        mask = x <= t
        nl = int(mask.sum())
        if nl < min_samples_leaf or n - nl < min_samples_leaf:
            continue
        yl, yr = y[mask], y[~mask]
        child[i] = (((yl - yl.mean()) ** 2).sum() + ((yr - yr.mean()) ** 2).sum()) / n
    parent = float(y.var())
    return _pick(child, parent, lo, hi)


def categorical_split(x, y, kind, n_classes=None, criterion="gini", min_samples_leaf=1):
    """Best subset split ``x in S`` for a categorical feature (README section 9).

    Categories are ordered by mean response (regression) or by P(class 1) (binary
    classification); then only the q-1 'prefix' subsets are scanned instead of all
    2^(q-1)-1 subsets.  This is exact for regression and for two classes; for >2
    classes we order by the share of the globally most common class (a heuristic).
    Returns (gain, frozenset_of_left_categories) or None.
    """
    cats, inv = np.unique(x, return_inverse=True)
    if len(cats) < 2:
        return None
    if kind == "regression":
        score_c = np.bincount(inv, weights=y) / np.bincount(inv)
        res = best_split_regression(score_c[inv], y, min_samples_leaf)
    else:
        target = 1 if n_classes == 2 else int(np.argmax(np.bincount(y, minlength=n_classes)))
        score_c = np.bincount(inv, weights=(y == target).astype(float)) / np.bincount(inv)
        res = best_split_classification(score_c[inv], y, n_classes, criterion, min_samples_leaf)
    if res is None:
        return None
    gain, thr = res
    return gain, frozenset(cats[score_c <= thr].tolist())


# --------------------------------------------------------------------------- #
# 3. Trees
# --------------------------------------------------------------------------- #


class Node:
    """One tree node.  Leaves have ``left is None``.

    value : class-count vector (classifier) or mean (regressor)
    impurity : I(t) in the tree's criterion (variance for regression)
    cats : frozenset for categorical splits (left = x in cats), else None
    """
    __slots__ = ("feature", "threshold", "cats", "left", "right", "n", "impurity", "value")

    def __init__(self, n, impurity, value):
        self.feature = self.threshold = self.cats = self.left = self.right = None
        self.n, self.impurity, self.value = n, impurity, value

    @property
    def is_leaf(self):
        return self.left is None

    def collapse(self):
        """Turn this node into a leaf (used by pruning)."""
        self.feature = self.threshold = self.cats = self.left = self.right = None


def _walk(node):
    yield node
    if not node.is_leaf:
        yield from _walk(node.left)
        yield from _walk(node.right)


class _BaseTree:
    """Greedy recursive partitioning (README section 3)."""

    _kind = None

    def __init__(self, criterion, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 min_impurity_decrease=0.0, max_features=None, categorical_features=(),
                 random_state=None):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features
        self.categorical_features = tuple(categorical_features)
        self.random_state = random_state

    # ---- fitting ----
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = self._prepare_y(np.asarray(y))
        self.n_features_, self.n_samples_ = X.shape[1], X.shape[0]
        self._rng = np.random.default_rng(self.random_state)
        sys.setrecursionlimit(max(sys.getrecursionlimit(), 20000))
        self.root_ = self._build(X, y, 0)
        return self

    def _n_try(self):
        d, mf = self.n_features_, self.max_features
        if mf is None:
            return d
        if mf == "sqrt":
            return max(1, int(np.sqrt(d)))
        if isinstance(mf, float):
            return max(1, int(mf * d))
        return min(int(mf), d)

    def _build(self, X, y, depth):
        n = len(y)
        value, imp = self._leaf_stats(y)
        node = Node(n, imp, value)
        if (imp <= 1e-14 or n < self.min_samples_split or n < 2 * self.min_samples_leaf
                or (self.max_depth is not None and depth >= self.max_depth)):
            return node
        split = self._find_split(X, y)
        if split is None:
            return node
        gain, j, thr, cats = split
        if n / self.n_samples_ * gain < self.min_impurity_decrease:
            return node
        mask = np.isin(X[:, j], list(cats)) if cats is not None else X[:, j] <= thr
        node.feature, node.threshold, node.cats = j, thr, cats
        node.left = self._build(X[mask], y[mask], depth + 1)
        node.right = self._build(X[~mask], y[~mask], depth + 1)
        return node

    def _find_split(self, X, y):
        d, k = self.n_features_, self._n_try()
        feats = np.arange(d) if self.max_features is None else self._rng.permutation(d)
        best, tried = None, 0
        for j in feats:
            if self.max_features is not None and tried >= k and best is not None:
                break
            x = X[:, j]
            if x.min() == x.max():
                continue
            tried += 1
            if j in self.categorical_features:
                res = self._cat_search(x, y)
                cand = None if res is None else (res[0], j, None, res[1])
            else:
                res = self._num_search(x, y)
                cand = None if res is None else (res[0], j, res[1], None)
            if cand is not None and (best is None or cand[0] > best[0] + _TOL):
                best = cand
        return best

    # ---- prediction ----
    def _route(self, node, X, idx, out):
        if node.is_leaf:
            out[idx] = node.value
            return
        col = X[idx, node.feature]
        go_left = np.isin(col, list(node.cats)) if node.cats is not None else col <= node.threshold
        self._route(node.left, X, idx[go_left], out)
        self._route(node.right, X, idx[~go_left], out)

    def _raw_predict(self, X):
        X = np.asarray(X, dtype=float)
        out = np.zeros((len(X),) + np.shape(self.root_.value))
        self._route(self.root_, X, np.arange(len(X)), out)
        return out

    # ---- inspection ----
    def get_n_leaves(self):
        return sum(nd.is_leaf for nd in _walk(self.root_))

    def get_depth(self):
        def depth(nd):
            return 0 if nd.is_leaf else 1 + max(depth(nd.left), depth(nd.right))
        return depth(self.root_)

    @property
    def feature_importances_(self):
        """Mean decrease in impurity, normalised to sum to 1   (eq. 13)."""
        imp = np.zeros(self.n_features_)
        for nd in _walk(self.root_):
            if not nd.is_leaf:
                dec = (nd.n * nd.impurity - nd.left.n * nd.left.impurity
                       - nd.right.n * nd.right.impurity) / self.n_samples_
                imp[nd.feature] += dec
        s = imp.sum()
        return imp / s if s > 0 else imp

    def export_text(self, feature_names=None, class_names=None, decimals=2):
        """Text rendering of the tree in the style of sklearn.tree.export_text."""
        names = feature_names or [f"x{j}" for j in range(self.n_features_)]
        lines = []

        def leaf_text(nd):
            if self._kind == "classification":
                k = int(np.argmax(nd.value))
                label = class_names[k] if class_names is not None else self.classes_[k]
                return f"class: {label}  (n={nd.n}, counts={nd.value.astype(int).tolist()})"
            return f"value: {nd.value:.{decimals}f}  (n={nd.n}, var={nd.impurity:.{decimals}f})"

        def rec(nd, indent):
            if nd.is_leaf:
                lines.append(f"{indent}|--- {leaf_text(nd)}")
                return
            nm = names[nd.feature]
            if nd.cats is not None:
                cs = "{" + ", ".join(str(int(c)) if float(c).is_integer() else str(c)
                                     for c in sorted(nd.cats)) + "}"
                lt, rt = f"{nm} in {cs}", f"{nm} not in {cs}"
            else:
                lt, rt = f"{nm} <= {nd.threshold:.{decimals}f}", f"{nm}  > {nd.threshold:.{decimals}f}"
            lines.append(f"{indent}|--- {lt}")
            rec(nd.left, indent + "|   ")
            lines.append(f"{indent}|--- {rt}")
            rec(nd.right, indent + "|   ")

        rec(self.root_, "")
        return "\n".join(lines)

    # ---- cost-complexity pruning (README section 11) ----
    def _R(self, nd):
        """Resubstitution cost of node t as a leaf: R(t) = (n_t/N) I(t)."""
        return nd.n / self.n_samples_ * nd.impurity

    def total_impurity(self):
        """R(T) = sum over leaves of (n_t/N) I(t)."""
        return sum(self._R(nd) for nd in _walk(self.root_) if nd.is_leaf)

    def cost_complexity(self, alpha):
        """R_alpha(T) = R(T) + alpha |T|   (eq. 11)."""
        return self.total_impurity() + alpha * self.get_n_leaves()

    def _links(self):
        """List of (g(t), node) for every internal node, g from eq. (12)."""
        out = []

        def rec(nd):
            if nd.is_leaf:
                return self._R(nd), 1
            rl, ll = rec(nd.left)
            rr, lr = rec(nd.right)
            r_sub, leaves = rl + rr, ll + lr
            out.append(((self._R(nd) - r_sub) / (leaves - 1), nd))
            return r_sub, leaves

        rec(self.root_)
        return out

    def _weakest(self):
        links = self._links()
        return min(links, key=lambda t: t[0])  # first minimum in post-order

    def cost_complexity_path(self):
        """Weakest-link sequence T_0 > T_1 > ... > root.

        Returns dict(alphas, impurities, n_leaves): T_k is optimal for
        alpha in [alphas[k], alphas[k+1]).
        """
        t = deepcopy(self)
        alphas, imps, leaves = [0.0], [t.total_impurity()], [t.get_n_leaves()]
        while not t.root_.is_leaf:
            g, nd = t._weakest()
            nd.collapse()
            alphas.append(g)
            imps.append(t.total_impurity())
            leaves.append(t.get_n_leaves())
        return {"alphas": np.maximum.accumulate(np.array(alphas)),
                "impurities": np.array(imps), "n_leaves": np.array(leaves)}

    def prune(self, alpha):
        """Return a copy pruned to the smallest subtree minimising R_alpha."""
        t = deepcopy(self)
        while not t.root_.is_leaf:
            g, nd = t._weakest()
            if g > alpha:
                break
            nd.collapse()
        return t


class DecisionTreeClassifier(_BaseTree):
    _kind = "classification"

    def __init__(self, criterion="gini", **kw):
        if criterion not in CRITERIA:
            raise ValueError(f"criterion must be one of {list(CRITERIA)}")
        super().__init__(criterion, **kw)

    def _prepare_y(self, y):
        self.classes_, yi = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        return yi

    def _leaf_stats(self, y):
        counts = np.bincount(y, minlength=self.n_classes_).astype(float)
        return counts, float(impurity_from_counts(counts, self.criterion))

    def _num_search(self, x, y):
        return best_split_classification(x, y, self.n_classes_, self.criterion,
                                         self.min_samples_leaf)

    def _cat_search(self, x, y):
        return categorical_split(x, y, "classification", self.n_classes_, self.criterion,
                                 self.min_samples_leaf)

    def predict_proba(self, X):
        c = self._raw_predict(X)
        return c / c.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self._raw_predict(X), axis=1)]

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))


class DecisionTreeRegressor(_BaseTree):
    _kind = "regression"

    def __init__(self, **kw):
        super().__init__("squared_error", **kw)

    def _prepare_y(self, y):
        return y.astype(float)

    def _leaf_stats(self, y):
        return float(y.mean()), float(y.var())

    def _num_search(self, x, y):
        return best_split_regression(x, y, self.min_samples_leaf)

    def _cat_search(self, x, y):
        return categorical_split(x, y, "regression", min_samples_leaf=self.min_samples_leaf)

    def predict(self, X):
        return self._raw_predict(X)

    def score(self, X, y):
        """Coefficient of determination R^2."""
        y = np.asarray(y, dtype=float)
        return float(1 - ((y - self.predict(X)) ** 2).sum() / ((y - y.mean()) ** 2).sum())


# --------------------------------------------------------------------------- #
# 4. Permutation importance (the unbiased alternative to MDI)
# --------------------------------------------------------------------------- #


def permutation_importance(model, X, y, n_repeats=10, random_state=0):
    """Drop in held-out score when one column is shuffled."""
    rng = np.random.default_rng(random_state)
    X = np.asarray(X, dtype=float)
    base = model.score(X, y)
    out = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            drops.append(base - model.score(Xp, y))
        out[j] = np.mean(drops)
    return out


# --------------------------------------------------------------------------- #
# 5. Bagging
# --------------------------------------------------------------------------- #


class BaggedTrees:
    """Bootstrap aggregation of trees; with max_features it is a random forest."""

    def __init__(self, tree_class=DecisionTreeRegressor, n_estimators=25, random_state=None,
                 **tree_kwargs):
        self.tree_class, self.n_estimators = tree_class, n_estimators
        self.random_state, self.tree_kwargs = random_state, tree_kwargs

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        rng = np.random.default_rng(self.random_state)
        n = len(X)
        self.estimators_ = []
        for _ in range(self.n_estimators):
            idx = rng.integers(0, n, size=n)                  # bootstrap sample
            seed = int(rng.integers(0, 2 ** 31 - 1))
            self.estimators_.append(
                self.tree_class(random_state=seed, **self.tree_kwargs).fit(X[idx], y[idx]))
        if self.tree_class is DecisionTreeClassifier:
            self.classes_ = np.unique(y)
        return self

    def predict_all(self, X):
        """Per-tree regression predictions, shape (n_estimators, n_samples)."""
        return np.array([t.predict(X) for t in self.estimators_])

    def predict_proba(self, X):
        P = np.zeros((len(X), len(self.classes_)))
        for t in self.estimators_:
            P[:, np.searchsorted(self.classes_, t.classes_)] += t.predict_proba(X)
        return P / len(self.estimators_)

    def predict(self, X):
        if self.tree_class is DecisionTreeClassifier:
            return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
        return self.predict_all(X).mean(axis=0)


def variance_experiment(make_data, x_test, tree_kwargs_list, n_estimators, n_datasets, seed=0):
    """Variance of predictions over fresh training sets, as a function of ensemble size.

    For each of ``n_datasets`` training sets and each config in ``tree_kwargs_list``
    we fit ``n_estimators`` bootstrap trees and record prefix averages, so one fit
    gives the variance for every ensemble size B = 1..n_estimators.
    Returns, per config, dict(var_by_B, rho, sigma2, mean_pred) where
    var_by_B[B-1] = average over test points of Var_datasets(mean of first B trees).
    """
    rng = np.random.default_rng(seed)
    preds = [np.zeros((n_datasets, n_estimators, len(x_test))) for _ in tree_kwargs_list]
    for d in range(n_datasets):
        X, y = make_data(rng)
        s = int(rng.integers(0, 2 ** 31 - 1))
        for c, kw in enumerate(tree_kwargs_list):
            preds[c][d] = BaggedTrees(DecisionTreeRegressor, n_estimators, s, **kw).fit(
                X, y).predict_all(x_test)
    results = []
    for P in preds:                                     # P: (datasets, B, test)
        cum = np.cumsum(P, axis=1) / np.arange(1, n_estimators + 1)[None, :, None]
        var_by_B = cum.var(axis=0).mean(axis=1)         # (B,)
        sigma2 = P.var(axis=(0, 1)).mean()              # variance of one tree (data + bootstrap)
        c = P - P.mean(axis=(0, 1))
        pair = ((c.sum(axis=1) ** 2 - (c ** 2).sum(axis=1)) / (n_estimators * (n_estimators - 1)))
        rho = pair.mean(axis=0).mean() / sigma2         # average pairwise correlation
        results.append({"var_by_B": var_by_B, "rho": rho, "sigma2": sigma2,
                        "mean_pred": P.mean(axis=(0, 1))})
    return results


# --------------------------------------------------------------------------- #
# 6. Toy data
# --------------------------------------------------------------------------- #


def make_moons(n, noise, rng, flip=0.0):
    """Two interleaving half circles; ``flip`` = fraction of labels flipped at random."""
    n1 = n // 2
    t1 = rng.uniform(0, np.pi, n1)
    t2 = rng.uniform(0, np.pi, n - n1)
    a = np.c_[np.cos(t1), np.sin(t1)]
    b = np.c_[1 - np.cos(t2), 0.5 - np.sin(t2)]
    X = np.vstack([a, b]) + rng.normal(0, noise, (n, 2))
    y = np.r_[np.zeros(n1, int), np.ones(n - n1, int)]
    f = rng.random(n) < flip
    y[f] = 1 - y[f]
    p = rng.permutation(n)
    return X[p], y[p]


def make_sine(n, noise, rng):
    """y = sin(2 pi x) + noise on x ~ U(0,1); X has shape (n, 1)."""
    x = rng.uniform(0, 1, n)
    return x[:, None], np.sin(2 * np.pi * x) + rng.normal(0, noise, n)


def make_friedman1(n, noise, rng, d=5):
    """Friedman #1: 10 sin(pi x0 x1) + 20 (x2-.5)^2 + 10 x3 + 5 x4 + noise."""
    X = rng.uniform(0, 1, (n, d))
    y = (10 * np.sin(np.pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - 0.5) ** 2
         + 10 * X[:, 3] + 5 * X[:, 4] + rng.normal(0, noise, n))
    return X, y


# Worked example of README section 12 (10 rows, 2 features).
WORKED_X = np.array([[1, 5], [2, 3], [3, 8], [4, 1], [5, 9],
                     [6, 2], [7, 7], [8, 4], [9, 6], [10, 10]], dtype=float)
WORKED_Y_CLASS = np.array([0, 0, 0, 1, 0, 1, 1, 1, 1, 1])
WORKED_Y_REG = np.array([2, 3, 1, 2, 9, 10, 8, 9, 10, 11], dtype=float)
