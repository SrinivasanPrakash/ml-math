"""Every number quoted in the README exercises / derivations is reproduced here."""
import numpy as np
import pytest

from tree import (DecisionTreeRegressor, Node, entropy, gini, impurity_from_counts, misclassification,
                  WORKED_X, WORKED_Y_CLASS, best_split_classification, DecisionTreeClassifier)


def test_exercise_1_three_class_node():
    p = np.array([0.5, 0.3, 0.2])
    assert gini(p) == pytest.approx(0.62)
    assert entropy(p) == pytest.approx(1.4855, abs=1e-4)
    assert misclassification(p) == pytest.approx(0.5)


def test_gini_gain_identity_weighted_squared_distance():
    """G(p_t) - sum w_c G(p_c) = sum w_c ||p_c - p_t||^2   (README eq. 6a)."""
    rng = np.random.default_rng(0)
    for _ in range(100):
        pl, pr = rng.dirichlet(np.ones(4)), rng.dirichlet(np.ones(4))
        w = rng.random()
        pt = w * pl + (1 - w) * pr
        lhs = gini(pt) - (w * gini(pl) + (1 - w) * gini(pr))
        rhs = w * ((pl - pt) ** 2).sum() + (1 - w) * ((pr - pt) ** 2).sum()
        assert lhs == pytest.approx(rhs)


def test_information_gain_equals_weighted_kl():
    rng = np.random.default_rng(1)
    pl, pr, w = rng.dirichlet(np.ones(3)), rng.dirichlet(np.ones(3)), 0.3
    pt = w * pl + (1 - w) * pr
    kl = lambda a, b: float((a * np.log2(a / b)).sum())
    assert entropy(pt) - (w * entropy(pl) + (1 - w) * entropy(pr)) == pytest.approx(
        w * kl(pl, pt) + (1 - w) * kl(pr, pt))


def test_exercise_cost_ratio_and_bootstrap_and_bagging():
    n = 10 ** 5
    assert n ** 2 / (n * np.log2(n)) == pytest.approx(6020, rel=0.01)
    assert (1 - 1 / 10) ** 10 == pytest.approx(0.3487, abs=1e-4)
    assert (1 - 1 / 1000) ** 1000 == pytest.approx(np.exp(-1), abs=2e-4)
    rho, s2 = 0.3, 1.0
    var = lambda B: rho * s2 + (1 - rho) * s2 / B
    assert var(10) == pytest.approx(0.37) and var(35) <= 0.32 < var(34)
    assert 2 ** 19 - 1 == 524287


def test_exercise_pruning_alpha():
    """Hand example: R(t)=0.10, two leaves with R=0.03, 0.04 -> alpha = 0.03."""
    t = DecisionTreeRegressor()
    t.n_samples_ = 100
    root = Node(50, 0.2, 0.0)
    root.feature, root.threshold = 0, 0.0
    root.left, root.right = Node(20, 0.15, 0.0), Node(30, 0.04 * 100 / 30, 0.0)
    t.root_ = root
    (g, node), = t._links()
    assert g == pytest.approx(0.03) and node is root
    # three-leaf subtree: R(t)=0.11, R(T_t)=0.05 -> alpha = 0.06/2 = 0.03
    assert (0.11 - 0.05) / (3 - 1) == pytest.approx(0.03)


def test_median_minimises_absolute_loss_numerically():
    y = np.array([1.0, 2.0, 2.5, 9.0, 30.0])
    cs = np.linspace(0, 31, 3101)
    assert cs[np.abs(y[None] - cs[:, None]).sum(1).argmin()] == pytest.approx(np.median(y), abs=0.02)


def test_readme_misclassification_tie_on_worked_example():
    x, y = WORKED_X[:, 0], WORKED_Y_CLASS
    gains = {}
    for t in (3.5, 5.5):
        L, R = y[x <= t], y[x > t]
        w = (len(L) * impurity_from_counts(np.bincount(L, minlength=2), "misclassification")
             + len(R) * impurity_from_counts(np.bincount(R, minlength=2), "misclassification")) / 10
        gains[t] = 0.4 - w
    assert gains[3.5] == pytest.approx(0.3) and gains[5.5] == pytest.approx(0.3)
    t = DecisionTreeClassifier().fit(WORKED_X, WORKED_Y_CLASS)
    imp = t.feature_importances_
    assert imp[0] == pytest.approx(2 / 3) and imp[1] == pytest.approx(1 / 3)
