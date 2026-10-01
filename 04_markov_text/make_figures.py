"""Regenerate every figure in figures/ deterministically:  python make_figures.py"""
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
import experiments as ex  # noqa: E402
import markov as mk  # noqa: E402

FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 10, "figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})


def fig_heatmap():
    seqs = ex.load_sentences("word")
    freq = Counter(t for s in seqs for t in s)
    top = [w for w, _ in freq.most_common(14)]
    mapped = [[mk.BOS] + [t if t in top else "<other>" for t in s] + [mk.EOS, mk.BOS] for s in seqs]   # EOS -> BOS wrap closes the chain
    states = [mk.BOS] + top + ["<other>", mk.EOS]
    C = mk.count_matrix(mapped, states)
    P = mk.mle_transition_matrix(C)
    fig, ax = plt.subplots(figsize=(8.2, 7))
    from matplotlib.colors import PowerNorm
    im = ax.imshow(P, cmap="viridis", norm=PowerNorm(0.5, vmin=0, vmax=1))
    ax.set_xticks(range(len(states)), states, rotation=70, ha="right")
    ax.set_yticks(range(len(states)), states)
    for i in range(len(states)):
        for j in range(len(states)):
            if P[i, j] >= 0.2:
                ax.text(j, i, f"{P[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if P[i, j] < 0.7 else "black")
    ax.set_xlabel("next word $j$")
    ax.set_ylabel("current word $i$")
    ax.set_title("Word-level transition matrix $P_{ij}=c_{ij}/\\sum_j c_{ij}$ (14 commonest words + rest)\n"
                 "each row sums to 1; bright = likely successor (colour scale $\\sqrt{P}$)", fontsize=10)
    fig.colorbar(im, ax=ax, label="$P_{ij}$", fraction=0.04)
    fig.tight_layout()
    fig.savefig(FIG / "transition_heatmap.png")
    plt.close(fig)


def fig_graph():
    P, names = ex.TOY, ex.TOY_NAMES
    pos = np.array([[0.0, 1.0], [-0.87, -0.5], [0.87, -0.5]]) * 1.6
    fig, ax = plt.subplots(figsize=(6.4, 6))
    cols = ["#f2c14e", "#9aa5b1", "#4f86c6"]
    R = 0.38
    for i, (x, y) in enumerate(pos):
        ax.add_patch(Circle((x, y), R, fc=cols[i], ec="k", zorder=3))
        ax.text(x, y, names[i], ha="center", va="center", zorder=4, fontweight="bold")
    for i in range(3):
        for j in range(3):
            if i == j:
                v = pos[i] / np.linalg.norm(pos[i])
                c = pos[i] + v * (R + 0.28)
                ax.add_patch(Circle(c, 0.28, fill=False, ec="k", zorder=2))
                ax.text(*(pos[i] + v * (R + 0.75)), f"{P[i, i]:.1f}", ha="center", va="center", fontsize=10)
                continue
            a = FancyArrowPatch(pos[i], pos[j], connectionstyle="arc3,rad=0.22", arrowstyle="-|>",
                                mutation_scale=16, shrinkA=R * 95, shrinkB=R * 95, lw=1 + 4 * P[i, j], color="#444", zorder=1)
            ax.add_patch(a)
            mid = (pos[i] + pos[j]) / 2
            d = pos[j] - pos[i]
            nrm = np.array([d[1], -d[0]]) / np.linalg.norm(d)
            ax.text(*(mid + nrm * 0.33), f"{P[i, j]:.1f}", ha="center", va="center", fontsize=10,
                    bbox=dict(fc="white", ec="none", pad=1))
    pi = mk.stationary_eig(P)
    ax.set_xlim(-2.6, 2.6); ax.set_ylim(-1.7, 3.0); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("Toy 3-state chain: arrow $i\\to j$ labelled $P_{ij}$ (outgoing labels sum to 1)\n"
                 f"stationary $\\pi$ = ({pi[0]:.3f}, {pi[1]:.3f}, {pi[2]:.3f})", fontsize=10)
    fig.savefig(FIG / "state_graph.png")
    plt.close(fig)


def fig_convergence():
    P = ex.TOY
    pi = mk.stationary_eig(P)
    lam2 = sorted(np.abs(np.linalg.eigvals(P)))[-2]
    N = 25
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2))
    d = np.array([1.0, 0, 0])
    traj = [d]
    for _ in range(N):
        traj.append(traj[-1] @ P)
    traj = np.array(traj)
    for j, nm in enumerate(ex.TOY_NAMES):
        a.plot(traj[:, j], marker="o", ms=3, label=f"$P(X_n=$ {nm}$)$")
        a.axhline(pi[j], ls=":", c=f"C{j}")
    a.set_xlabel("step $n$"); a.set_ylabel("probability")
    a.set_title("Start sunny: $\\pi_0 P^n$ approaches $\\pi$ (dotted)")
    a.legend()
    for s, nm in enumerate(ex.TOY_NAMES):
        e = [np.abs(mk.n_step(P, n)[s] - pi).sum() for n in range(N + 1)]
        b.semilogy(e, marker="o", ms=3, label=f"start {nm}")
    ref = 1.0 * lam2 ** np.arange(N + 1)
    b.semilogy(ref * np.abs(mk.n_step(P, 1)[0] - pi).sum() / lam2, "k--", label=f"$|\\lambda_2|^n$, $|\\lambda_2|$={lam2:.3f}")
    b.set_xlabel("step $n$"); b.set_ylabel("$\\|\\pi_0P^n-\\pi\\|_1$")
    b.set_title("Geometric convergence at rate $|\\lambda_2|$")
    b.legend()
    fig.tight_layout()
    fig.savefig(FIG / "convergence_stationary.png")
    plt.close(fig)


def fig_perplexity():
    seqs = ex.load_sentences("word")
    train, test = ex.train_test_split(seqs)
    orders = list(range(0, 6))
    rows = ex.perplexity_table(train, test, orders)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.semilogy(orders, [rows[o]["addk_train"] for o in orders], "o-", c="C0", label="add-0.05: train")
    ax.semilogy(orders, [rows[o]["addk_test"] for o in orders], "o--", c="C0", label="add-0.05: test")
    ax.semilogy(orders, [rows[o]["kn_train"] for o in orders], "s-", c="C3", label="Kneser-Ney: train")
    ax.semilogy(orders, [rows[o]["kn_test"] for o in orders], "s--", c="C3", label="Kneser-Ney: test")
    V = mk.MarkovChain(0).fit(train).V
    ax.axhline(V, c="gray", ls=":")
    ax.text(2.5, V * 0.8, f"uniform model: PP = V = {V}", ha="center", color="gray", fontsize=9)
    ax.set_ylim(1.3, V * 1.3)
    ax.set_xlabel("context length $k$ (order)"); ax.set_ylabel("perplexity (log scale)")
    ax.set_title(f"Train falls with order (memorisation); add-k test PP turns up, KN test PP flattens\n"
                 f"({len(train)} training / {len(test)} test sentences; MLE test PP = inf for every $k\\geq1$)", fontsize=10)
    ax.legend(); fig.tight_layout()
    fig.savefig(FIG / "perplexity_vs_order.png")
    plt.close(fig)


def fig_overlap():
    seqs = ex.load_sentences("word")
    train, _ = ex.train_test_split(seqs)
    orders = list(range(1, 6))
    rows = ex.overlap_table(train, orders)
    fig, ax = plt.subplots(figsize=(7.5, 5.6))
    ax.plot(orders, [rows[o]["overlap"] for o in orders], "o-", label="generated 4-grams found in training text")
    ax.plot(orders, [rows[o]["verbatim"] for o in orders], "s-", label="generated sentences copied verbatim")
    ax.plot(orders, [rows[o]["deterministic"] for o in orders], "^--", label="contexts with only one successor")
    ax.set_xticks(orders)
    ax.set_xlabel("order $k$"); ax.set_ylabel("fraction")
    ax.set_title("Low order: novel but incoherent.  High order: fluent but copied\n(300 samples per order, unsmoothed chain)", fontsize=10)
    ax.annotate("incoherent\nbut original", (1, rows[1]["overlap"]), xytext=(1.3, 0.12), arrowprops=dict(arrowstyle="->"))
    ax.annotate("fluent\nbut memorised", (5, rows[5]["overlap"]), xytext=(3.7, 0.25), arrowprops=dict(arrowstyle="->"))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=1); fig.tight_layout()
    fig.savefig(FIG / "ngram_overlap_vs_order.png")
    plt.close(fig)


if __name__ == "__main__":
    for f in (fig_heatmap, fig_graph, fig_convergence, fig_perplexity, fig_overlap):
        f()
        print("wrote", f.__name__)
