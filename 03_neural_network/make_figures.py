"""Regenerate every figure in figures/ (deterministic, fixed seeds)."""
import os

# Small matrices: single-threaded BLAS is much faster than thread contention.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))
import experiments as E  # noqa: E402
import nn  # noqa: E402

OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.25,
                     "font.size": 10})


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def fig_activations():
    z = np.linspace(-6, 6, 400).reshape(1, -1)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for name, cls, col in [("sigmoid", nn.Sigmoid, "C0"), ("tanh", nn.Tanh, "C1"),
                           ("ReLU", nn.ReLU, "C2")]:
        a = cls()
        out = a.forward(z)
        ax[0].plot(z[0], out[0], col, label=name)
        ax[1].plot(z[0], a.deriv()[0], col, label=name + "'")
    ax[0].set(title="Activations f(z)", xlabel="z", ylabel="f(z)", ylim=(-1.3, 3))
    ax[1].set(title="Derivatives f'(z)", xlabel="z", ylabel="f'(z)", ylim=(-0.1, 1.2))
    ax[1].annotate("sigmoid' <= 1/4:\nevery sigmoid layer\nshrinks gradients >= 4x",
                   (0, 0.25), (1.5, 0.6), arrowprops=dict(arrowstyle="->"))
    ax[1].annotate("tanh' = 1 only at z=0", (0, 1), (1.5, 1.05),
                   arrowprops=dict(arrowstyle="->"))
    ax[1].annotate("ReLU' = 0 for z<0\n(dead unit)", (-3, 0), (-5.8, 0.35),
                   arrowprops=dict(arrowstyle="->"))
    ax[0].legend()
    ax[1].legend(loc="upper left")
    save(fig, "activations.png")


def fig_graph():
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.axis("off")
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 3.2)
    nodes = [("x", 0.6), ("Z1=xW1+b1", 2.0), ("A1=f(Z1)", 3.7), ("Z2=A1W2+b2", 5.4),
             ("A2=f(Z2)", 7.1), ("Z3=A2W3+b3", 8.8), ("L", 10.4)]
    for t, x in nodes:
        ax.text(x, 2.0, t, ha="center", va="center", fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.35", fc="#dbe9f6", ec="k"))
    hw = lambda t: 0.065 * len(t) + 0.2          # approximate half box width
    for (t0, x0), (t1, x1) in zip(nodes[:-1], nodes[1:]):
        ax.annotate("", (x1 - hw(t1), 2.0), (x0 + hw(t0), 2.0),
                    arrowprops=dict(arrowstyle="->", color="C0"))
    labs = ["", "D1", "", "D2", "", "D3", ""]
    for (_, x), lab in zip(nodes, labs):
        if lab:
            ax.text(x, 1.0, r"$\Delta_" + lab[1] + r"=\partial L/\partial Z_" + lab[1] + "$",
                    ha="center", fontsize=9, color="C3")
    for (t0, x0), (t1, x1) in zip(nodes[:-1], nodes[1:]):
        ax.annotate("", (x0 + hw(t0), 1.55), (x1 - hw(t1), 1.55),
                    arrowprops=dict(arrowstyle="->", color="C3"))
    ax.text(5.5, 2.85, "forward: values flow left to right (cache Z, A)", color="C0",
            ha="center")
    ax.text(5.5, 0.35, r"backward: $\Delta_l=(\Delta_{l+1}W_{l+1}^\top)\odot f'(Z_l)$,"
            r"  $dW_l=A_{l-1}^\top\Delta_l$", color="C3", ha="center")
    save(fig, "computational_graph.png")


def fig_boundaries():
    cfg = [("moons", [0, 20, 100, 500], [8, 8], 500, 0.01),
           ("spirals", [0, 10, 25, 1500], [64, 64], 1500, 0.01)]
    fig, axes = plt.subplots(2, 4, figsize=(14, 7.5))
    cmap = plt.get_cmap("coolwarm")
    for r, (ds, snaps, hid, ep, lr) in enumerate(cfg):
        X, y, gx, gy, sn, hist = E.boundary_run(ds, snaps, hid, ep, lr)
        for c, e in enumerate(snaps):
            ax = axes[r, c]
            pred, acc = sn[e]
            ax.contourf(gx, gy, pred, alpha=0.35, cmap="viridis" if ds == "spirals" else cmap,
                        levels=np.arange(-0.5, pred.max() + 1.5, 1))
            ax.scatter(X[:, 0], X[:, 1], c=y, s=12, cmap="viridis" if ds == "spirals" else cmap,
                       edgecolor="k", linewidth=0.3)
            ax.set(title=f"{ds}, epoch {e}, acc {acc:.0%}", xlabel="$x_1$", ylabel="$x_2$")
            ax.grid(False)
    fig.suptitle("Decision regions during training (moons: 2-8-8-2 tanh; "
                 "spirals: 2-64-64-3 ReLU; Adam, full batch)")
    fig.tight_layout()
    save(fig, "decision_boundaries.png")


def fig_init():
    res = E.init_loss_curves()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    labels = {"small": "N(0, 0.01^2)  (too small)", "normal": "N(0, 1)  (too large)",
              "xavier": "Xavier", "he": "He"}
    for s, (h, acc) in res.items():
        ax[0].plot(h, label=f"{labels[s]}  final acc {acc:.0%}")
    ax[0].axhline(np.log(3), color="gray", ls="--")
    ax[0].text(250, np.log(3) + 0.05, "ln 3 = chance level", color="gray")
    ax[0].set(ylim=(0, 3), xlabel="epoch", ylabel="training cross-entropy",
              title="8 hidden ReLU layers, plain SGD lr=0.02")
    ax[0].legend(fontsize=8)
    ax[1].set_title("Log scale, clipped (N(0,1): ~1e4 at epoch 1, NaN soon after)")
    for s, (h, acc) in res.items():
        ax[1].semilogy(h, label=labels[s])
    ax[1].set(xlabel="epoch", ylabel="training cross-entropy (log)", ylim=(2e-2, 3e4))
    save(fig, "init_loss_curves.png")


def fig_layer_stats():
    cases = [("sigmoid", "xavier"), ("tanh", "xavier"), ("tanh", "normal"),
             ("relu", "normal"), ("relu", "xavier"), ("relu", "he")]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for act, init in cases:
        a, d = E.layer_statistics(act, init)
        ax[0].semilogy(np.arange(1, len(a) + 1), a, marker="o", ms=3, label=f"{act}/{init}")
        ax[1].semilogy(np.arange(1, len(d) + 1), d, marker="o", ms=3, label=f"{act}/{init}")
    ax[0].set_yscale("log")
    ax[0].set(title="Forward: RMS of pre-activations $Z_l$", xlabel="layer l (1 = input side)",
              ylabel="RMS")
    ax[1].set(title=r"Backward: RMS of $\Delta_l=\partial L/\partial Z_l$",
              xlabel="layer l (1 = input side)", ylabel="RMS")
    ax[1].annotate("sigmoid/xavier: vanishing\n(1e-16 at layer 1)", (1, 5e-17), (3, 1e-13),
                   arrowprops=dict(arrowstyle="->"), fontsize=8)
    ax[1].annotate("relu/normal: exploding\n(1e13 at layer 1)", (1, 1.8e13), (4, 1e9),
                   arrowprops=dict(arrowstyle="->"), fontsize=8)
    ax[0].annotate("relu/xavier: signal\nhalves in power\nper layer", (12, 0.02), (4, 1e-2),
                   arrowprops=dict(arrowstyle="->"), fontsize=8)
    ax[0].legend(fontsize=8)
    fig.suptitle("20 hidden layers, width 100, Gaussian inputs (one batch of 512)")
    save(fig, "activation_gradient_per_layer.png")


def fig_optimisers():
    gx, gy = np.meshgrid(np.linspace(-10, 10, 200), np.linspace(-2, 2, 200))
    F = 0.5 * (E.RAVINE_A * gx ** 2 + E.RAVINE_B * gy ** 2)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].contour(gx, gy, F, levels=np.logspace(-1, 2.3, 14), cmap="Greys", linewidths=0.6)
    for (name, opt), col in zip(E.ravine_optimisers().items(), ["C0", "C1", "C3"]):
        path, f = E.ravine_run(opt)
        ax[0].plot(path[:60, 0], path[:60, 1], ".-", color=col, ms=3, lw=0.8, label=name)
        ax[1].semilogy(f, color=col, label=f"{name}: f<1e-3 at step {int((f < 1e-3).argmax())}")
    ax[0].set(title=r"Ravine $f=\frac{1}{2}(x^2+100y^2)$, first 60 steps",
              xlabel="x", ylabel="y")
    ax[0].legend(fontsize=8, loc="lower right")
    ax[0].annotate("SGD zigzags across\nthe steep direction", (-8, 1.2), (-9.5, -1.6),
                   arrowprops=dict(arrowstyle="->"), fontsize=8)
    ax[1].set(title="Loss vs step", xlabel="step", ylabel="f", ylim=(1e-10, 1e3))
    ax[1].legend(fontsize=8)
    save(fig, "optimiser_ravine.png")


if __name__ == "__main__":
    fig_activations()
    fig_graph()
    fig_boundaries()
    fig_init()
    fig_layer_stats()
    fig_optimisers()
