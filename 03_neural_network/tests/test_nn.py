"""Tests for src/nn.py: gradient checks, learning tasks, invariants."""
import numpy as np
import pytest

import nn

EPS = 1e-5
TOL = 1e-6


def rel_err(a, b):
    """Norm-wise relative error ||a-b|| / (||a|| + ||b||)."""
    denom = np.linalg.norm(a) + np.linalg.norm(b)
    return 0.0 if denom == 0 else np.linalg.norm(a - b) / denom


def num_grad(f, x, eps=EPS):
    """Centred differences, perturbing x in place entry by entry."""
    g = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        old = x[i]
        x[i] = old + eps
        fp = f()
        x[i] = old - eps
        fm = f()
        x[i] = old
        g[i] = (fp - fm) / (2 * eps)
    return g


# ----------------------------------------------------------------- layers
def check_layer(layer, X, rng):
    """Scalar s = <G, layer(X)> with random G; compare dX and dparams."""
    G = rng.standard_normal(layer.forward(X).shape)

    def s():
        return np.sum(G * layer.forward(X))

    layer.forward(X)
    dX = layer.backward(G)
    assert rel_err(dX, num_grad(s, X)) < TOL
    for p, g in layer.parameters():
        layer.forward(X)
        layer.backward(G)
        assert rel_err(g.copy(), num_grad(s, p)) < TOL


@pytest.mark.parametrize("make", [
    lambda rng: nn.Dense(5, 4, "xavier", rng),
    lambda rng: nn.Dense(3, 7, "he", rng),
    lambda rng: nn.Sigmoid(),
    lambda rng: nn.Tanh(),
    lambda rng: nn.ReLU(),
    lambda rng: nn.Softmax(),
])
def test_layer_gradients(make):
    rng = np.random.default_rng(1)
    layer = make(rng)
    n_in = layer.W.shape[0] if isinstance(layer, nn.Dense) else 4
    X = rng.standard_normal((6, n_in))
    if isinstance(layer, nn.ReLU):
        X[np.abs(X) < 0.05] = 0.5       # stay away from the kink at 0
    check_layer(layer, X, rng)


# ----------------------------------------------------------------- losses
def check_loss(loss, pred, target):
    loss.forward(pred, target)
    analytic = loss.backward()

    def f():
        return loss.forward(pred, target)

    numeric = num_grad(f, pred)
    assert rel_err(analytic, numeric) < TOL


def test_mse_gradient():
    rng = np.random.default_rng(2)
    check_loss(nn.MSE(), rng.standard_normal((5, 3)), rng.standard_normal((5, 3)))


def test_softmax_ce_gradient():
    rng = np.random.default_rng(3)
    check_loss(nn.SoftmaxCrossEntropy(), rng.standard_normal((6, 4)),
               rng.integers(0, 4, 6))


def test_ce_on_probs_gradient():
    rng = np.random.default_rng(4)
    P = nn.softmax(rng.standard_normal((6, 4)))
    # perturbing P directly (rows need not stay normalised: derivative is pointwise)
    check_loss(nn.CrossEntropy(), P, rng.integers(0, 4, 6))


def test_bce_logits_gradient():
    rng = np.random.default_rng(5)
    check_loss(nn.BCEWithLogits(), rng.standard_normal((7, 1)),
               rng.integers(0, 2, 7))


def test_softmax_plus_ce_equals_p_minus_y():
    """The headline identity (E12): chain Softmax -> CE equals fused (P-Y)/N."""
    rng = np.random.default_rng(6)
    Z = rng.standard_normal((8, 5))
    y = rng.integers(0, 5, 8)
    fused = nn.SoftmaxCrossEntropy()
    fused.forward(Z, y)
    g_fused = fused.backward()
    sm, ce = nn.Softmax(), nn.CrossEntropy()
    ce.forward(sm.forward(Z), y)
    g_chain = sm.backward(ce.backward())
    assert np.allclose(g_fused, g_chain, atol=1e-12)
    assert np.allclose(g_fused, (nn.softmax(Z) - nn.one_hot(y, 5)) / 8)


# ---------------------------------------------------- whole-network check
@pytest.mark.parametrize("act", ["sigmoid", "tanh", "relu"])
def test_full_network_gradients(act):
    rng = np.random.default_rng(7)
    model = nn.mlp([3, 5, 4, 3], activation=act, rng=rng)
    X = rng.standard_normal((6, 3))
    y = rng.integers(0, 3, 6)
    loss = nn.SoftmaxCrossEntropy()
    loss.forward(model.forward(X), y)
    model.backward(loss.backward())
    analytic = [g.copy() for _, g in model.parameters()]

    def f():
        return loss.forward(model.forward(X), y)

    for (p, _), a in zip(model.parameters(), analytic):
        assert rel_err(a, num_grad(f, p)) < TOL


# ------------------------------------------------------------ properties
def test_softmax_properties():
    rng = np.random.default_rng(8)
    Z = rng.standard_normal((5, 6))
    P = nn.softmax(Z)
    assert np.allclose(P.sum(axis=1), 1) and (P > 0).all()
    assert np.allclose(P, nn.softmax(Z + 100.0))        # shift invariance
    assert np.isfinite(nn.softmax(np.array([[1e4, 0.0, -1e4]]))).all()


def test_ce_gradient_rows_sum_to_zero():
    rng = np.random.default_rng(9)
    loss = nn.SoftmaxCrossEntropy()
    loss.forward(rng.standard_normal((6, 4)), rng.integers(0, 4, 6))
    assert np.allclose(loss.backward().sum(axis=1), 0)


def test_batch_gradient_is_mean_of_per_sample_gradients():
    rng = np.random.default_rng(10)
    model = nn.mlp([2, 4, 3], "tanh", rng=rng)
    X = rng.standard_normal((5, 2))
    y = rng.integers(0, 3, 5)
    loss = nn.SoftmaxCrossEntropy()

    def grads(Xb, yb):
        loss.forward(model.forward(Xb), yb)
        model.backward(loss.backward())
        return [g.copy() for _, g in model.parameters()]

    full = grads(X, y)
    acc = [np.zeros_like(g) for g in full]
    for i in range(5):
        for a, g in zip(acc, grads(X[i:i + 1], y[i:i + 1])):
            a += g / 5
    for a, f in zip(acc, full):
        assert np.allclose(a, f, atol=1e-12)


def test_sample_order_invariance():
    rng = np.random.default_rng(11)
    model = nn.mlp([2, 4, 3], "relu", rng=rng)
    X = rng.standard_normal((7, 2))
    y = rng.integers(0, 3, 7)
    loss = nn.SoftmaxCrossEntropy()
    loss.forward(model.forward(X), y)
    model.backward(loss.backward())
    g1 = [g.copy() for _, g in model.parameters()]
    perm = rng.permutation(7)
    loss.forward(model.forward(X[perm]), y[perm])
    model.backward(loss.backward())
    for a, (_, b) in zip(g1, model.parameters()):
        assert np.allclose(a, b, atol=1e-12)


def test_activation_derivative_identities():
    z = np.linspace(-4, 4, 9).reshape(1, -1)
    s = nn.Sigmoid()
    s.forward(z)
    assert np.allclose(s.deriv(), s.out * (1 - s.out))
    assert s.deriv().max() <= 0.25 + 1e-12            # sigma' <= 1/4
    t = nn.Tanh()
    t.forward(z)
    assert t.deriv().max() <= 1 + 1e-12
    assert np.allclose(np.tanh(z), 2 * nn.Sigmoid().forward(2 * z) - 1)


def test_sigmoid_stable_for_extreme_inputs():
    out = nn.Sigmoid().forward(np.array([[-1000.0, 0.0, 1000.0]]))
    assert np.allclose(out, [[0.0, 0.5, 1.0]])


@pytest.mark.parametrize("scheme,fan", [("xavier", 2 / (400 + 300)),
                                        ("he", 2 / 400)])
def test_init_variance(scheme, fan):
    W = nn.init_weights(400, 300, scheme, np.random.default_rng(0))
    assert abs(W.var() / fan - 1) < 0.05


def test_he_init_preserves_relu_second_moment():
    """E[a^2] stays ~constant through a deep ReLU net with He init (Sec. 8)."""
    rng = np.random.default_rng(0)
    A = rng.standard_normal((2000, 256))
    q0 = np.mean(A ** 2)
    for _ in range(20):
        W = nn.init_weights(256, 256, "he", rng)
        A = np.maximum(A @ W, 0)
    assert 0.5 < np.mean(A ** 2) / q0 < 2.0


# ------------------------------------------------------------ optimisers
def test_sgd_step():
    p, g = np.array([1.0, 2.0]), np.array([0.5, -1.0])
    nn.SGD(0.1).step([(p, g)])
    assert np.allclose(p, [0.95, 2.1])


def test_momentum_two_steps():
    p, g = np.array([1.0]), np.array([1.0])
    opt = nn.Momentum(lr=0.1, beta=0.9)
    opt.step([(p, g)])           # v=1      p=0.9
    opt.step([(p, g)])           # v=1.9    p=0.9-0.19
    assert np.allclose(p, 0.71)


def test_adam_first_step_is_lr_times_sign():
    """With bias correction the first Adam step is lr * g/(|g|+eps) ~ lr*sign(g)."""
    p, g = np.array([0.0, 0.0, 0.0]), np.array([1e-3, -5.0, 200.0])
    nn.Adam(lr=0.01).step([(p, g)])
    assert np.allclose(p, -0.01 * np.sign(g), atol=1e-6)


def test_optimisers_minimise_quadratic():
    for opt in [nn.SGD(0.1), nn.Momentum(0.05, 0.9), nn.Adam(0.1)]:
        p = np.array([3.0, -2.0])
        for _ in range(500):
            opt.step([(p, 2 * p)])
        assert np.linalg.norm(p) < 1e-2


# --------------------------------------------------------------- learning
def test_hand_worked_example():
    """The numbers in README Sec. 13 (checked here)."""
    import hand_example
    r = hand_example.run()
    assert np.isclose(r["z3"], -0.75) and np.isclose(r["loss"], 1.53125)
    assert np.allclose(r["delta2"], [-1.75, 0.875])
    assert np.allclose(r["delta1"], [0.0, 1.3125])
    assert np.allclose(r["dW1"], [[0, 1.3125], [0, 2.625]])
    assert np.allclose(r["dW2"], [[-3.5, 1.75], [-1.75, 0.875]])
    assert np.allclose(r["dW3"], [[-0.875], [-4.375]])


def test_learns_xor():
    X, y = nn.make_xor()
    rng = np.random.default_rng(0)
    model = nn.mlp([2, 8, 2], "tanh", rng=rng)
    nn.fit(model, nn.SoftmaxCrossEntropy(), nn.Adam(0.05), X, y, epochs=300)
    assert nn.accuracy(model, X, y) == 1.0


def test_logistic_regression_is_single_neuron():
    """Base case: Dense(2,1)+BCE on separable data reaches 100%."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 2))
    y = (X @ np.array([2.0, -1.0]) + 0.5 > 0).astype(int)
    model = nn.Sequential([nn.Dense(2, 1, "xavier", rng)])
    nn.fit(model, nn.BCEWithLogits(), nn.Adam(0.1), X, y, epochs=200)
    pred = (model.forward(X)[:, 0] > 0).astype(int)
    assert np.mean(pred == y) > 0.98


def test_learns_spirals():
    Xtr, ytr = nn.make_spirals(100, 3, seed=0)
    Xte, yte = nn.make_spirals(100, 3, seed=1)
    rng = np.random.default_rng(0)
    model = nn.mlp([2, 64, 64, 3], "relu", rng=rng)
    nn.fit(model, nn.SoftmaxCrossEntropy(), nn.Adam(0.01), Xtr, ytr, epochs=1500)
    assert nn.accuracy(model, Xtr, ytr) > 0.95
    assert nn.accuracy(model, Xte, yte) > 0.95


def test_loss_decreases_monotonically_for_small_lr():
    X, y = nn.make_moons(100, 0.1, seed=0)
    rng = np.random.default_rng(0)
    model = nn.mlp([2, 16, 16, 2], "tanh", rng=rng)
    hist = nn.fit(model, nn.SoftmaxCrossEntropy(), nn.SGD(0.05), X, y,
                  epochs=300)
    d = np.diff(hist)
    assert (d <= 1e-12).all()
    assert hist[-1] < 0.8 * hist[0]


def test_training_is_deterministic():
    def run():
        X, y = nn.make_moons(60, 0.1, seed=0)
        m = nn.mlp([2, 8, 2], "tanh", rng=np.random.default_rng(3))
        return nn.fit(m, nn.SoftmaxCrossEntropy(), nn.Adam(0.01), X, y,
                      epochs=20, batch_size=16, rng=np.random.default_rng(4))
    assert run() == run()


def test_relu_interpolant_construction():
    """Universal-approximation construction: error <= h^2/8 max|f''| (Sec. 11)."""
    import experiments
    a, b = -np.pi, np.pi
    xs = np.linspace(a, b, 2001).reshape(-1, 1)
    prev = np.inf
    for H in [2, 4, 8, 16, 32]:
        net = experiments.relu_interpolant(np.sin, a, b, H)
        err = np.max(np.abs(net.forward(xs)[:, 0] - np.sin(xs[:, 0])))
        h = (b - a) / H
        assert err <= h ** 2 / 8 + 1e-12 and err < prev
        prev = err
    # exact at the knots
    net = experiments.relu_interpolant(np.sin, a, b, 8)
    kn = np.linspace(a, b, 9)
    assert np.allclose(net.forward(kn.reshape(-1, 1))[:, 0], np.sin(kn))


def test_constant_init_keeps_hidden_units_identical():
    """Symmetry pitfall (Sec. 16): equal initial weights => equal units forever."""
    rng = np.random.default_rng(0)
    model = nn.mlp([2, 4, 2], "tanh", rng=rng)
    for layer in model.layers:
        if isinstance(layer, nn.Dense):
            layer.W[...] = 0.3
    X, y = nn.make_moons(40, 0.1, seed=0)
    nn.fit(model, nn.SoftmaxCrossEntropy(), nn.SGD(0.1), X, y, epochs=50)
    W1 = model.layers[0].W
    assert np.allclose(W1, W1[:, :1])        # all 4 hidden columns equal
