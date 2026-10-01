"""Experiments shared by make_figures.py and demo.py (so README numbers and
figures come from the same code)."""
import numpy as np

import nn


# ---- decision boundary snapshots ------------------------------------------
def boundary_run(dataset, snapshots, hidden, epochs, lr, seed=0):
    """Train full-batch with Adam; return grid probabilities at given epochs."""
    if dataset == "moons":
        X, y = nn.make_moons(200, 0.15, seed=seed)
        classes = 2
    else:
        X, y = nn.make_spirals(100, 3, seed=seed)
        classes = 3
    lim = 1.6 if dataset == "moons" else 1.1
    xs = np.linspace(-lim, lim, 120) if dataset == "spirals" else np.linspace(-1.6, 2.6, 120)
    ys = np.linspace(-lim, lim, 120) if dataset == "spirals" else np.linspace(-1.2, 1.7, 120)
    gx, gy = np.meshgrid(xs, ys)
    grid = np.c_[gx.ravel(), gy.ravel()]
    rng = np.random.default_rng(seed)
    model = nn.mlp([2] + hidden + [classes], "tanh" if dataset == "moons" else "relu",
                   rng=rng)
    snaps = {}

    def cb(epoch, m):
        if epoch in snapshots:
            snaps[epoch] = (m.predict(grid).reshape(gx.shape),
                            nn.accuracy(m, X, y))

    hist = nn.fit(model, nn.SoftmaxCrossEntropy(), nn.Adam(lr), X, y,
                  epochs=epochs, callback=cb)
    return X, y, gx, gy, snaps, hist


# ---- initialisation experiments --------------------------------------------
def init_loss_curves(schemes=("small", "normal", "xavier", "he"), depth=8,
                     width=32, epochs=400, lr=0.02, seed=0):
    X, y = nn.make_spirals(100, 3, seed=0)
    out = {}
    for s in schemes:
        rng = np.random.default_rng(seed)
        model = nn.mlp([2] + [width] * depth + [3], "relu", init=s, rng=rng)
        with np.errstate(all="ignore"):
            hist = nn.fit(model, nn.SoftmaxCrossEntropy(), nn.SGD(lr), X, y,
                          epochs=epochs)
        out[s] = (np.array(hist), nn.accuracy(model, X, y))
    return out


def layer_statistics(activation, init, depth=20, width=100, N=512, seed=0):
    """RMS of A_l (forward) and of dL/dZ_l (backward) for each layer l."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, width))
    y = rng.integers(0, 3, N)
    model = nn.mlp([width] + [width] * depth + [3], activation, init=init, rng=rng)
    dense = [l for l in model.layers if isinstance(l, nn.Dense)]
    loss = nn.SoftmaxCrossEntropy()
    with np.errstate(all="ignore"):
        loss.forward(model.forward(X), y)
        G = loss.backward()
        deltas = []                          # delta_l = dL/dZ_l, output layer first
        for layer in reversed(model.layers):
            if isinstance(layer, nn.Dense):
                deltas.append(np.sqrt(np.mean(G ** 2)))
            G = layer.backward(G)
        # first/last entries: drop the output layer, reorder input -> output
        deltas = deltas[::-1][:-1]
        acts = [np.sqrt(np.mean((l.X @ l.W + l.b) ** 2)) for l in dense][:-1]
    return np.array(acts), np.array(deltas)


# ---- optimiser comparison on a ravine ---------------------------------------
RAVINE_A, RAVINE_B = 1.0, 100.0


def ravine_f(p):
    return 0.5 * (RAVINE_A * p[0] ** 2 + RAVINE_B * p[1] ** 2)


def ravine_grad(p):
    return np.array([RAVINE_A * p[0], RAVINE_B * p[1]])


def ravine_run(opt, steps=300, start=(-9.0, 1.5)):
    p = np.array(start, dtype=float)
    path, fs = [p.copy()], [ravine_f(p)]
    for _ in range(steps):
        opt.step([(p, ravine_grad(p))])
        path.append(p.copy())
        fs.append(ravine_f(p))
    return np.array(path), np.array(fs)


def ravine_optimisers():
    return {
        "SGD (lr=0.019)": nn.SGD(0.019),
        "Momentum (lr=0.01, b=0.9)": nn.Momentum(0.01, 0.9),
        "Adam (lr=0.3)": nn.Adam(0.3),
    }


# ---- universal approximation: explicit ReLU construction -------------------
def relu_interpolant(f, a, b, H):
    """Build 1-H-1 ReLU net equal to the piecewise-linear interpolant of f
    on H equal segments of [a, b] (README Sec. 11). No training involved.

    f_hat(x) = f(x_0) + sum_k c_k ReLU(x - x_k),  c_0 = s_0, c_k = s_k - s_{k-1}
    where s_k is the slope of segment k.
    """
    knots = np.linspace(a, b, H + 1)
    vals = f(knots)
    slopes = np.diff(vals) / np.diff(knots)
    c = np.r_[slopes[0], np.diff(slopes)]
    d1, d2 = nn.Dense(1, H), nn.Dense(H, 1)
    d1.W = np.ones((1, H))
    d1.b = -knots[:-1]
    d2.W = c.reshape(H, 1)
    d2.b = np.array([vals[0]])
    return nn.Sequential([d1, nn.ReLU(), d2])
