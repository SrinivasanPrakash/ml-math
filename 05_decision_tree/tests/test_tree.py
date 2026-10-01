"""Tests: closed forms, naive == fast, sklearn agreement (optional), invariants."""
import itertools

import numpy as np
import pytest

from tree import (CRITERIA, WORKED_X, WORKED_Y_CLASS, WORKED_Y_REG, BaggedTrees,
                  DecisionTreeClassifier, DecisionTreeRegressor, best_split_classification,
                  best_split_classification_naive, best_split_regression,
                  best_split_regression_naive, categorical_split, entropy, gini,
                  impurity_from_counts, make_friedman1, make_moons, make_sine,
                  misclassification, permutation_importance, variance_experiment, _walk)


# ---------------- impurity measures ----------------
def test_impurity_closed_forms():
    p = np.array([0.5, 0.5])
    assert entropy(p) == pytest.approx(1.0)
    assert gini(p) == pytest.approx(0.5)
    assert misclassification(p) == pytest.approx(0.5)
    u = np.full(4, 0.25)
    assert entropy(u) == pytest.approx(2.0)          # log2 K
    assert gini(u) == pytest.approx(0.75)            # 1 - 1/K
    assert misclassification(u) == pytest.approx(0.75)


@pytest.mark.parametrize("name", list(CRITERIA))
def test_impurity_zero_when_pure_and_max_when_uniform(name):
    f = CRITERIA[name]
    rng = np.random.default_rng(0)
    K = 5
    assert f(np.eye(K)[2]) == pytest.approx(0.0)
    for _ in range(50):
        p = rng.dirichlet(np.ones(K))
        assert 0 <= f(p) <= f(np.full(K, 1 / K)) + 1e-12


@pytest.mark.parametrize("name", list(CRITERIA))
def test_concavity_and_information_gain_nonnegative(name):
    """Jensen: I(sum lam_j p_j) >= sum lam_j I(p_j)  (README eq. 6)."""
    f = CRITERIA[name]
    rng = np.random.default_rng(1)
    for _ in range(300):
        K = rng.integers(2, 6)
        pl, pr = rng.dirichlet(np.ones(K)), rng.dirichlet(np.ones(K))
        lam = rng.random()
        mix = lam * pl + (1 - lam) * pr
        assert f(mix) >= lam * f(pl) + (1 - lam) * f(pr) - 1e-12


def test_gini_is_expected_misclassification_under_random_labelling():
    rng = np.random.default_rng(2)
    p = rng.dirichlet(np.ones(4))
    truth = rng.choice(4, size=400000, p=p)
    guess = rng.choice(4, size=400000, p=p)          # label drawn from the node's own distribution
    assert np.mean(truth != guess) == pytest.approx(gini(p), abs=4e-3)


def test_misclassification_can_fail_to_see_a_useful_split():
    parent = np.array([400, 400.0])
    a_l, a_r = np.array([300, 100.0]), np.array([100, 300.0])
    b_l, b_r = np.array([200, 400.0]), np.array([200, 0.0])
    def child(l, r, c):
        return (l.sum() * impurity_from_counts(l, c) + r.sum() * impurity_from_counts(r, c)) / 800
    assert child(a_l, a_r, "misclassification") == pytest.approx(child(b_l, b_r, "misclassification"))
    assert child(b_l, b_r, "gini") < child(a_l, a_r, "gini") - 0.01
    assert child(b_l, b_r, "entropy") < child(a_l, a_r, "entropy") - 0.01


# ---------------- split search ----------------
@pytest.mark.parametrize("crit", list(CRITERIA))
@pytest.mark.parametrize("seed", range(5))
def test_fast_split_equals_naive_classification(crit, seed):
    rng = np.random.default_rng(seed)
    n, K = 60, 3
    x, y = rng.normal(size=n), rng.integers(0, K, n)
    f = best_split_classification(x, y, K, crit, 2)
    g = best_split_classification_naive(x, y, K, crit, 2)
    assert f[0] == pytest.approx(g[0], abs=1e-10)
    assert f[1] == pytest.approx(g[1])


@pytest.mark.parametrize("seed", range(5))
def test_fast_split_equals_naive_regression(seed):
    rng = np.random.default_rng(seed)
    x, y = rng.normal(size=70), rng.normal(size=70)
    f, g = best_split_regression(x, y, 3), best_split_regression_naive(x, y, 3)
    assert f[0] == pytest.approx(g[0], abs=1e-10) and f[1] == pytest.approx(g[1])


def test_fast_split_equals_naive_with_ties_in_x():
    rng = np.random.default_rng(7)
    x, y = rng.integers(0, 6, 80).astype(float), rng.integers(0, 2, 80)
    f, g = best_split_classification(x, y, 2), best_split_classification_naive(x, y, 2)
    assert f[0] == pytest.approx(g[0], abs=1e-10)
    yr = rng.normal(size=80)
    f, g = best_split_regression(x, yr), best_split_regression_naive(x, yr)
    assert f[0] == pytest.approx(g[0], abs=1e-10)


def test_variance_reduction_closed_form():
    """Delta = (nL nR / n^2)(mean_L - mean_R)^2  (README eq. 9)."""
    x, y = WORKED_X[:, 0], WORKED_Y_REG
    gain, thr = best_split_regression(x, y)
    L, R = y[x <= thr], y[x > thr]
    n = len(y)
    assert gain == pytest.approx(len(L) * len(R) / n ** 2 * (L.mean() - R.mean()) ** 2)
    assert gain == pytest.approx(13.5)


def test_no_split_on_constant_feature():
    assert best_split_classification(np.ones(10), np.arange(10) % 2, 2) is None
    assert best_split_regression(np.ones(10), np.arange(10.0)) is None


def test_optimal_leaf_value_is_the_mean():
    rng = np.random.default_rng(3)
    y = rng.normal(size=30)
    cs = np.linspace(y.min(), y.max(), 2001)
    sse = ((y[None, :] - cs[:, None]) ** 2).sum(1)
    assert abs(cs[sse.argmin()] - y.mean()) < (cs[1] - cs[0])
    t = DecisionTreeRegressor(max_depth=0).fit(rng.normal(size=(30, 1)), y)
    assert t.predict(np.zeros((1, 1)))[0] == pytest.approx(y.mean())


@pytest.mark.parametrize("kind", ["regression", "gini", "entropy"])
def test_categorical_split_matches_brute_force_over_subsets(kind):
    rng = np.random.default_rng(4)
    q, n = 6, 120
    x = rng.integers(0, q, n).astype(float)
    if kind == "regression":
        y = np.array([0, 3, 1, 5, 2, 4.0])[x.astype(int)] + rng.normal(0, 1, n)
        res = categorical_split(x, y, "regression")
        parent = y.var()
    else:
        y = (rng.random(n) < np.array([.1, .8, .3, .9, .5, .2])[x.astype(int)]).astype(int)
        res = categorical_split(x, y, "classification", 2, kind)
        parent = float(impurity_from_counts(np.bincount(y, minlength=2), kind))
    best = 0.0
    for r in range(1, q):
        for S in itertools.combinations(range(q), r):
            m = np.isin(x, S)
            if kind == "regression":
                c = (m.sum() * y[m].var() + (~m).sum() * y[~m].var()) / n
            else:
                c = (m.sum() * impurity_from_counts(np.bincount(y[m], minlength=2), kind)
                     + (~m).sum() * impurity_from_counts(np.bincount(y[~m], minlength=2), kind)) / n
            best = max(best, parent - c)
    assert res[0] == pytest.approx(best, abs=1e-10)


def test_categorical_tree_predicts_with_subset_rule():
    x = np.array([0, 1, 2, 3] * 10, dtype=float)[:, None]
    y = np.where(np.isin(x[:, 0], [0, 3]), 5.0, -5.0)   # not an interval in code order
    t = DecisionTreeRegressor(max_depth=1, categorical_features=[0]).fit(x, y)
    assert np.allclose(t.predict(x), y)
    assert t.root_.cats in (frozenset({0.0, 3.0}), frozenset({1.0, 2.0}))
    assert "in {" in t.export_text()


# ---------------- tree properties ----------------
def test_pure_leaf_property_and_exact_fit_classifier():
    rng = np.random.default_rng(5)
    X, y = make_moons(200, 0.3, rng, flip=0.15)
    for crit in CRITERIA:
        t = DecisionTreeClassifier(crit).fit(X, y)
        assert t.score(X, y) == 1.0
        for nd in _walk(t.root_):
            if nd.is_leaf:
                assert (nd.value > 0).sum() == 1            # pure
                assert nd.impurity == pytest.approx(0.0)


def test_exact_fit_regressor_and_leaf_is_mean():
    rng = np.random.default_rng(6)
    X, y = make_sine(100, 0.3, rng)
    t = DecisionTreeRegressor().fit(X, y)
    assert np.allclose(t.predict(X), y)
    t2 = DecisionTreeRegressor(min_samples_leaf=10).fit(X, y)
    leaf_of = t2.predict(X)
    for v in np.unique(leaf_of):
        assert v == pytest.approx(y[leaf_of == v].mean())  # leaf value = mean of its samples


def test_duplicate_x_conflict_cannot_be_fit():
    X = np.array([[1.0], [1.0], [2.0]])
    t = DecisionTreeClassifier().fit(X, [0, 1, 0])
    assert t.score(X, [0, 1, 0]) < 1.0


def test_xor_needs_depth_two_despite_zero_first_gain():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]] * 5, dtype=float)
    y = np.array([0, 1, 1, 0] * 5)
    assert best_split_classification(X[:, 0], y, 2, "gini")[0] == pytest.approx(0.0)
    assert DecisionTreeClassifier().fit(X, y).score(X, y) == 1.0


def test_hyperparameter_limits_respected():
    rng = np.random.default_rng(8)
    X, y = make_moons(300, 0.3, rng, flip=0.1)
    t = DecisionTreeClassifier(max_depth=3, min_samples_leaf=7).fit(X, y)
    assert t.get_depth() <= 3
    assert min(nd.n for nd in _walk(t.root_) if nd.is_leaf) >= 7
    t = DecisionTreeClassifier(min_impurity_decrease=0.01).fit(X, y)
    for nd in _walk(t.root_):
        if not nd.is_leaf:
            dec = nd.n * nd.impurity - nd.left.n * nd.left.impurity - nd.right.n * nd.right.impurity
            assert dec / 300 >= 0.01 - 1e-12      # every kept split meets the threshold
    assert t.get_n_leaves() < DecisionTreeClassifier().fit(X, y).get_n_leaves()


def test_impurity_decrease_nonnegative_and_importances_sum_to_one():
    rng = np.random.default_rng(9)
    X, y = make_friedman1(150, 1.0, rng)
    for t in (DecisionTreeRegressor(max_depth=5).fit(X, y),
              DecisionTreeClassifier().fit(X, (y > np.median(y)).astype(int))):
        for nd in _walk(t.root_):
            if not nd.is_leaf:
                dec = nd.n * nd.impurity - nd.left.n * nd.left.impurity - nd.right.n * nd.right.impurity
                assert dec >= -1e-9
        assert t.feature_importances_.sum() == pytest.approx(1.0)


def test_worked_example_numbers():
    x, y = WORKED_X, WORKED_Y_CLASS
    assert impurity_from_counts(np.bincount(y), "gini") == pytest.approx(0.48)
    assert impurity_from_counts(np.bincount(y), "entropy") == pytest.approx(0.970951, abs=1e-6)
    g, thr = best_split_classification(x[:, 0], y, 2, "gini")
    assert (thr, g) == (5.5, pytest.approx(0.32))
    g, thr = best_split_classification(x[:, 0], y, 2, "entropy")
    assert (thr, g) == (5.5, pytest.approx(0.609987, abs=1e-6))
    g, thr = best_split_classification(x[:, 0], y, 2, "misclassification")
    assert (thr, g) == (3.5, pytest.approx(0.3))
    assert best_split_classification(x[:, 1], y, 2, "gini")[0] == pytest.approx(0.08)


def test_text_printer_has_expected_lines():
    t = DecisionTreeClassifier(max_depth=1).fit(WORKED_X, WORKED_Y_CLASS)
    s = t.export_text(["x1", "x2"])
    assert "x1 <= 5.50" in s and "class: 1" in s and "counts=[4, 1]" in s


# ---------------- pruning ----------------
def _all_subtree_options(nd, R):
    """All (R, leaves) pairs achievable by collapsing nodes (independent brute force)."""
    opts = [(R(nd), 1)]
    if not nd.is_leaf:
        for (rl, ll), (rr, lr) in itertools.product(_all_subtree_options(nd.left, R),
                                                    _all_subtree_options(nd.right, R)):
            opts.append((rl + rr, ll + lr))
    return opts


def _noisy_tree(seed=0, n=120):
    rng = np.random.default_rng(seed)
    X, y = make_moons(n, 0.35, rng, flip=0.1)
    return DecisionTreeClassifier().fit(X, y), X, y


def test_pruning_path_monotone():
    t, X, y = _noisy_tree()
    p = t.cost_complexity_path()
    assert p["n_leaves"][0] == t.get_n_leaves() and p["n_leaves"][-1] == 1
    assert np.all(np.diff(p["alphas"]) >= 0)             # alphas nondecreasing
    assert np.all(np.diff(p["impurities"]) >= -1e-12)    # R(T_k) nondecreasing
    assert np.all(np.diff(p["n_leaves"]) < 0)            # strictly smaller trees


def test_pruned_tree_is_nested_and_training_error_monotone():
    t, X, y = _noisy_tree(1)
    p = t.cost_complexity_path()
    errs, prev_leaves = [], 10 ** 9
    for a in p["alphas"]:
        pt = t.prune(a)
        assert pt.get_n_leaves() <= prev_leaves
        prev_leaves = pt.get_n_leaves()
        errs.append(1 - pt.score(X, y))
    # the unpruned tree has zero training error; pruning never decreases it along the path
    assert errs[0] == 0.0
    assert np.all(np.diff(errs) >= -1e-12)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_pruned_tree_minimises_cost_complexity_exactly(seed):
    """prune(alpha) attains min over ALL subtrees of R + alpha |T| (brute force)."""
    rng = np.random.default_rng(seed)
    X, y = make_moons(40, 0.4, rng, flip=0.15)
    t = DecisionTreeClassifier().fit(X, y)
    opts = _all_subtree_options(t.root_, t._R)
    for a in [0.0, 0.002, 0.01, 0.03, 0.08, 0.2]:
        best = min(r + a * l for r, l in opts)
        assert t.prune(a).cost_complexity(a) == pytest.approx(best, abs=1e-12)


def test_prune_large_alpha_gives_root():
    t, X, y = _noisy_tree()
    assert t.prune(10.0).get_n_leaves() == 1
    assert t.prune(0.0).total_impurity() == pytest.approx(t.total_impurity())


# ---------------- bagging ----------------
def test_bagging_reduces_variance():
    xt = make_sine(30, 0, np.random.default_rng(0))[0]
    res = variance_experiment(lambda r: make_sine(60, 0.3, r), xt, [{}], 15, 25, seed=0)[0]
    v = res["var_by_B"]
    assert v[-1] < 0.6 * v[0]
    assert 0 <= res["rho"] <= 1


def test_bagged_classifier_and_regressor_run():
    rng = np.random.default_rng(0)
    X, y = make_moons(150, 0.3, rng)
    b = BaggedTrees(DecisionTreeClassifier, 15, 0, max_features=1).fit(X, y)
    assert (b.predict(X) == y).mean() > 0.9
    assert b.predict_proba(X).shape == (150, 2)
    Xr, yr = make_sine(100, 0.2, rng)
    assert BaggedTrees(DecisionTreeRegressor, 10, 0).fit(Xr, yr).predict(Xr).shape == (100,)


def test_mdi_bias_and_permutation_importance():
    rng = np.random.default_rng(0)
    n = 300
    signal = rng.integers(0, 2, n).astype(float)
    y = np.where(rng.random(n) < 0.8, signal, 1 - signal).astype(int)
    X = np.c_[signal, rng.normal(size=n)]                # informative binary, useless continuous
    t = DecisionTreeClassifier().fit(X, y)
    imp = t.feature_importances_
    assert imp[1] > imp[0]                               # MDI prefers the noise feature
    Xte = np.c_[rng.integers(0, 2, 2000).astype(float), rng.normal(size=2000)]
    yte = np.where(rng.random(2000) < 0.8, Xte[:, 0], 1 - Xte[:, 0]).astype(int)
    pi = permutation_importance(t, Xte, yte, 5)
    assert pi[0] > pi[1] + 0.05


# ---------------- sklearn agreement (optional) ----------------
sk = pytest.importorskip("sklearn.tree", reason="sklearn not installed")


@pytest.mark.parametrize("crit", ["gini", "entropy"])
@pytest.mark.parametrize("depth", [1, 3, 5])
def test_matches_sklearn_classifier(crit, depth):
    rng = np.random.default_rng(depth)
    X = rng.normal(size=(200, 4))
    y = ((X[:, 0] + 0.5 * X[:, 1] ** 2 + rng.normal(0, 0.7, 200)) > 0.5).astype(int)
    ours = DecisionTreeClassifier(crit, max_depth=depth).fit(X, y)
    ref = sk.DecisionTreeClassifier(criterion=crit, max_depth=depth, random_state=0).fit(X, y)
    Xt = rng.normal(size=(500, 4))
    assert np.array_equal(ours.predict(X), ref.predict(X))
    assert ours.get_n_leaves() == ref.get_n_leaves()
    if depth <= 3:   # deeper trees contain exact-gain ties that sklearn breaks by random feature order
        assert np.array_equal(ours.predict(Xt), ref.predict(Xt))
        assert np.allclose(ours.feature_importances_, ref.feature_importances_, atol=1e-6)


@pytest.mark.parametrize("depth", [1, 3, 6])
def test_matches_sklearn_regressor(depth):
    rng = np.random.default_rng(10 + depth)
    X, y = make_friedman1(200, 1.0, rng)
    ours = DecisionTreeRegressor(max_depth=depth).fit(X, y)
    ref = sk.DecisionTreeRegressor(max_depth=depth, random_state=0).fit(X, y)
    Xt = rng.uniform(size=(300, 5))
    assert np.allclose(ours.predict(X), ref.predict(X), atol=1e-8)
    assert ours.get_n_leaves() == ref.get_n_leaves()
    if depth <= 3:   # deeper trees contain exact-gain ties that sklearn breaks by random feature order
        assert np.allclose(ours.predict(Xt), ref.predict(Xt), atol=1e-6)
        assert np.allclose(ours.feature_importances_, ref.feature_importances_, atol=1e-6)


def test_pruning_matches_sklearn_ccp_path():
    rng = np.random.default_rng(11)
    X, y = make_moons(150, 0.35, rng, flip=0.1)
    ours = DecisionTreeClassifier().fit(X, y).cost_complexity_path()
    ref = sk.DecisionTreeClassifier(random_state=0).fit(X, y).cost_complexity_pruning_path(X, y)
    # same set of distinct alphas (sklearn reports one entry per distinct alpha)
    assert np.allclose(np.unique(np.round(ours["alphas"], 10)),
                       np.unique(np.round(ref.ccp_alphas, 10)), atol=1e-9)
