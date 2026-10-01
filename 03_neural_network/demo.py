"""End-to-end demo: prints every number quoted in README.md (runs in <30 s)."""
import os

# Small matrices: single-threaded BLAS is much faster than thread contention.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import experiments as E  # noqa: E402
import hand_example  # noqa: E402
import nn  # noqa: E402

np.set_printoptions(precision=4, suppress=True)


def section(t):
    print("\n=== " + t)


section("Sec. 3: single neuron / logistic regression  x=(2,1) w=(1,-1) b=0 y=1")
x, w, b, y = np.array([2.0, 1.0]), np.array([1.0, -1.0]), 0.0, 1.0
z = w @ x + b
a = 1 / (1 + np.exp(-z))
print(f"z={z:.4f}  a={a:.4f}  L={-np.log(a):.4f}  dz={a - y:.4f}  "
      f"dw={(a - y) * x}  db={a - y:.4f}")

section("Sec. 5: activation facts")
zz = np.linspace(-10, 10, 200001).reshape(1, -1)
s = nn.Sigmoid(); s.forward(zz)
t = nn.Tanh(); t.forward(zz)
print(f"max sigmoid' = {s.deriv().max():.4f} (at z=0);  max tanh' = {t.deriv().max():.4f}")
print(f"0.25^10 = {0.25 ** 10:.3e},  0.25^20 = {0.25 ** 20:.3e}")

section("Sec. 6: softmax + cross-entropy, logits (2,1,0), true class 0")
Z = np.array([[2.0, 1.0, 0.0]])
ce = nn.SoftmaxCrossEntropy()
L = ce.forward(Z, np.array([0]))
print(f"exp = {np.exp(Z[0])}, sum={np.exp(Z[0]).sum():.4f}")
print(f"p = {nn.softmax(Z)[0]}  L = {L:.4f}  grad (p-y) = {ce.backward()[0]}")

section("Sec. 13: hand-worked 2-2-2-1 ReLU net, x=(1,2), y=1, MSE")
r = hand_example.run()
for k in ["z1", "a1", "z2", "a2", "z3", "loss", "delta3", "delta2", "delta1",
          "dW3", "dW2", "dW1", "db3", "db2", "db1"]:
    print(f"{k:7s}", np.round(r[k], 5).tolist())

# finite-difference check of one hand-example entry
def _loss_with(dw):
    x_ = np.array([[1.0, 2.0]]); y_ = np.array([[1.0]])
    W1 = np.array([[1.0, -1.0], [0.5, 1.0]]); W1[1, 1] += dw
    a1 = np.maximum(x_ @ W1, 0)
    a2 = np.maximum(a1 @ np.array([[0.5, 1.0], [-0.5, 0.5]]), 0)
    return 0.5 * np.sum((a2 @ np.array([[1.0], [-0.5]]) - y_) ** 2)
print("finite difference dL/dW1[1,1] =", round((_loss_with(1e-5) - _loss_with(-1e-5)) / 2e-5, 6))

section("Gradient check (centred differences, eps=1e-5): full 3-8-8-3 nets")
rng = np.random.default_rng(0)
for act in ["sigmoid", "tanh", "relu"]:
    model = nn.mlp([3, 8, 8, 3], act, rng=rng)
    X = rng.standard_normal((10, 3)); yy = rng.integers(0, 3, 10)
    loss = nn.SoftmaxCrossEntropy()
    loss.forward(model.forward(X), yy); model.backward(loss.backward())
    worst = 0.0
    for p, g in model.parameters():
        g = g.copy(); ng = np.zeros_like(p)
        for i in np.ndindex(p.shape):
            old = p[i]
            p[i] = old + 1e-5; fp = loss.forward(model.forward(X), yy)
            p[i] = old - 1e-5; fm = loss.forward(model.forward(X), yy)
            p[i] = old; ng[i] = (fp - fm) / 2e-5
        worst = max(worst, np.linalg.norm(g - ng) / (np.linalg.norm(g) + np.linalg.norm(ng)))
    print(f"{act:8s} worst relative error = {worst:.2e}")

section("Sec. 8: initialisation variance (width 400 -> 300)")
for sch in ["xavier", "he"]:
    W = nn.init_weights(400, 300, sch, np.random.default_rng(0))
    print(f"{sch:7s} empirical Var(w) = {W.var():.5f}   target = "
          f"{2 / (700) if sch == 'xavier' else 2 / 400:.5f}")
for act, init in [("relu", "xavier"), ("relu", "he"), ("relu", "normal"),
                  ("tanh", "xavier"), ("tanh", "normal"), ("sigmoid", "xavier")]:
    a_, d_ = E.layer_statistics(act, init)
    print(f"{act:7s}/{init:7s} RMS Z_1={a_[0]:.3g} Z_19={a_[-1]:.3g}   "
          f"RMS Delta_1={d_[0]:.3g} Delta_19={d_[-1]:.3g}")
print(f"theory relu/xavier: RMS(Z_19)/RMS(Z_1) = 2^-9 = {2 ** -9.0:.4g}")

print("per-layer RMS factor of Delta in sigmoid/xavier net: "
      f"{(E.layer_statistics('sigmoid', 'xavier')[1][-1] / E.layer_statistics('sigmoid', 'xavier')[1][0]) ** (-1 / 18):.3f}"
      " (bound 0.25)")
a_, _ = E.layer_statistics("relu", "normal")
print(f"theory relu/normal: RMS(Z_19) ~ Z_1 * sqrt(100/2)^18 = {a_[0] * np.sqrt(50.0) ** 18:.2e}")

section("Sec. 8/9: 8 hidden ReLU layers, SGD lr=0.02, 400 epochs (spirals)")
for sch, (h, acc) in E.init_loss_curves().items():
    print(f"{sch:7s} loss[0]={h[0]:.4g}  loss[-1]={h[-1]:.4g}  train acc={acc:.1%}")

section("Sec. 10: ravine f=0.5(x^2+100y^2), start (-9,1.5)")
for name, opt in E.ravine_optimisers().items():
    _, f = E.ravine_run(opt)
    print(f"{name:28s} steps to f<1e-3: {int((f < 1e-3).argmax()):4d}   f[300]={f[-1]:.2e}")
print("SGD stability limit 2/100 = 0.02;  contraction factors |1-lr*h|: "
      f"x: {abs(1 - 0.019):.3f}  y: {abs(1 - 0.019 * 100):.3f}")
p = np.array([0.0]); nn.Adam(lr=0.01).step([(p, np.array([123.0]))])
print(f"Adam first step on g=123 moves parameter by {p[0]:.6f} (= -lr)")

section("Sec. 11: ReLU interpolant of sin on [-pi, pi]")
xs = np.linspace(-np.pi, np.pi, 4001).reshape(-1, 1)
for H in [2, 4, 8, 16, 32]:
    net = E.relu_interpolant(np.sin, -np.pi, np.pi, H)
    err = np.max(np.abs(net.forward(xs)[:, 0] - np.sin(xs[:, 0])))
    print(f"H={H:2d} hidden units: max error {err:.4f}   bound h^2/8 = "
          f"{((2 * np.pi / H) ** 2) / 8:.4f}")

section("Exercises: numbers")
zc = -6.0; ac = 1 / (1 + np.exp(-zc))
print(f"z=-6, y=1: a={ac:.5f}  BCE dL/dz={ac - 1:.5f}  MSE dL/dz={(ac - 1) * ac * (1 - ac):.6f}")
sizes = [784, 128, 64, 10]
print("784-128-64-10 parameters:", [(sizes[i] * sizes[i + 1], sizes[i + 1]) for i in range(3)],
      "total", sum(sizes[i] * sizes[i + 1] + sizes[i + 1] for i in range(3)))
print(f"momentum rate sqrt(beta) = {np.sqrt(0.9):.4f}; effective lr multiplier 1/(1-beta) = {1 / 0.1:.0f}")

section("Learning tasks")
X, y = nn.make_xor()
m = nn.mlp([2, 8, 2], "tanh", rng=np.random.default_rng(0))
h = nn.fit(m, nn.SoftmaxCrossEntropy(), nn.Adam(0.05), X, y, epochs=300)
print(f"XOR: loss {h[0]:.3f} -> {h[-1]:.5f}, accuracy {nn.accuracy(m, X, y):.0%}")
Xtr, ytr = nn.make_spirals(100, 3, seed=0)
Xte, yte = nn.make_spirals(100, 3, seed=1)
m = nn.mlp([2, 64, 64, 3], "relu", rng=np.random.default_rng(0))
nn.fit(m, nn.SoftmaxCrossEntropy(), nn.Adam(0.01), Xtr, ytr, epochs=1500)
print(f"Spirals 2-64-64-3: train acc {nn.accuracy(m, Xtr, ytr):.1%}, "
      f"test acc {nn.accuracy(m, Xte, yte):.1%}")
Xm, ym = nn.make_moons(100, 0.1, seed=0)
m = nn.mlp([2, 16, 16, 2], "tanh", rng=np.random.default_rng(0))
h = np.array(nn.fit(m, nn.SoftmaxCrossEntropy(), nn.SGD(0.05), Xm, ym, epochs=300))
print(f"Moons SGD lr=0.05: loss {h[0]:.4f} -> {h[-1]:.4f}, "
      f"max increase between epochs = {max(0, np.diff(h).max()):.1e}")
