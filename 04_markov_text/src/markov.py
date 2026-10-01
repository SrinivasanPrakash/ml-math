"""Markov chain text generator, from scratch (stdlib + NumPy).

Equation numbers refer to README.md:
  (1) Markov property            P(X_{t+1}=j | X_t=i, history) = P_ij
  (2) row-stochastic matrix      P >= 0, P 1 = 1
  (3) MLE                        P_ij = c_ij / sum_j c_ij
  (4) Chapman-Kolmogorov         P^(n+m) = P^(n) P^(m)
  (5) stationary distribution    pi P = pi
  (6) detailed balance           pi_i P_ij = pi_j P_ji
  (7) inverse-CDF sampling       j = min{ j : F_j > u }
  (8) add-k smoothing            (c + k) / (n + k V)
  (9) interpolated Kneser-Ney
 (10) cross-entropy              H = -(1/N) sum log2 p(w_t | context)
 (11) perplexity                 PP = 2^H
 (12) temperature                p_j ~ p_j^(1/T)

Part A: dense-matrix tools for a first-order chain on a finite state set.
Part B: order-k n-gram language model (word or character level).
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Dict, Hashable, List, Sequence, Tuple

import numpy as np

BOS = "<s>"      # padding that fills the context at the start of a sentence
EOS = "</s>"     # end-of-sentence symbol (predictable)
UNK = "<unk>"    # stand-in for tokens never seen in training


# ---------------------------------------------------------------------------
# Part A: matrices
# ---------------------------------------------------------------------------
def count_matrix(sequences: Sequence[Sequence[Hashable]], states: Sequence[Hashable]) -> np.ndarray:
    """c_ij = number of observed transitions i -> j, summed over sequences."""
    idx = {s: i for i, s in enumerate(states)}
    C = np.zeros((len(states), len(states)))
    for seq in sequences:
        for a, b in zip(seq[:-1], seq[1:]):
            C[idx[a], idx[b]] += 1
    return C


def mle_transition_matrix(C: np.ndarray, k: float = 0.0) -> np.ndarray:
    """Eq. (3) (k=0) and Eq. (8) (k>0): P_ij = (c_ij + k) / (sum_j c_ij + k V).

    A row with no observations (and k=0) is undefined under the MLE; we make it
    uniform so that the result is still row-stochastic.
    """
    C = np.asarray(C, dtype=float)
    V = C.shape[1]
    num = C + k
    den = num.sum(axis=1, keepdims=True)
    P = np.where(den > 0, num / np.where(den > 0, den, 1), 1.0 / V)
    return P


def is_row_stochastic(P: np.ndarray, tol: float = 1e-9) -> bool:
    """Eq. (2)."""
    return bool(np.all(P >= -tol) and np.allclose(P.sum(axis=1), 1.0, atol=tol))


def n_step(P: np.ndarray, n: int) -> np.ndarray:
    """Eq. (4): the n-step transition matrix is the matrix power P^n."""
    return np.linalg.matrix_power(P, n)


def stationary_eig(P: np.ndarray) -> np.ndarray:
    """Eq. (5): left eigenvector of P for eigenvalue 1 == right eigenvector of P^T."""
    w, v = np.linalg.eig(P.T)
    i = int(np.argmin(np.abs(w - 1.0)))
    pi = np.real(v[:, i])
    return pi / pi.sum()          # normalise (also fixes the sign)


def stationary_power(P: np.ndarray, pi0: np.ndarray | None = None, tol: float = 1e-12,
                     max_iter: int = 100000) -> Tuple[np.ndarray, List[float]]:
    """Power iteration pi_{t+1} = pi_t P.  Returns (pi, history of ||pi_t - pi_{t-1}||_1)."""
    V = P.shape[0]
    pi = np.full(V, 1.0 / V) if pi0 is None else np.asarray(pi0, float)
    hist = []
    for _ in range(max_iter):
        nxt = pi @ P
        d = float(np.abs(nxt - pi).sum())
        hist.append(d)
        pi = nxt
        if d < tol:
            break
    return pi, hist


def detailed_balance_gap(P: np.ndarray, pi: np.ndarray) -> float:
    """Eq. (6): max_ij |pi_i P_ij - pi_j P_ji| (zero iff the chain is reversible w.r.t. pi)."""
    F = pi[:, None] * P
    return float(np.abs(F - F.T).max())


def is_irreducible(P: np.ndarray) -> bool:
    """Every state reaches every other: (I + A)^(V-1) has no zero entry, A = [P > 0]."""
    V = P.shape[0]
    M = np.eye(V) + (P > 0)
    return bool(np.all(np.linalg.matrix_power(M, V - 1) > 0))


def period(P: np.ndarray, state: int = 0) -> int:
    """gcd of the return times to `state` (aperiodic iff 1); looks at lengths up to 2V."""
    from math import gcd
    V = P.shape[0]
    A = (P > 0).astype(float)
    cur = np.eye(V)
    g = 0
    for n in range(1, 2 * V + 1):
        cur = (cur @ A > 0).astype(float)
        if cur[state, state] > 0:
            g = gcd(g, n)
    return g


def simulate_states(P: np.ndarray, start: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample a path X_0=start, X_1, ..., X_n using inverse-CDF sampling (Eq. 7)."""
    cdf = np.cumsum(P, axis=1)
    path = np.empty(n + 1, dtype=int)
    path[0] = start
    for t in range(n):
        path[t + 1] = sample_inverse_cdf(cdf[path[t]], rng.random(), is_cdf=True)
    return path


def sample_inverse_cdf(p: np.ndarray, u: float, is_cdf: bool = False) -> int:
    """Eq. (7): smallest j with F_j > u, where F = cumsum(p) and u ~ Uniform[0,1)."""
    F = p if is_cdf else np.cumsum(p)
    j = int(np.searchsorted(F, u, side="right"))
    return min(j, len(F) - 1)      # guards against F[-1] = 1 - 1e-16 < u


def temperature_scale(p: np.ndarray, T: float) -> np.ndarray:
    """Eq. (12): p_j^(1/T), renormalised.  T=1 identity, T->0 argmax, T->inf uniform."""
    if T <= 0:
        out = np.zeros_like(p)
        out[int(np.argmax(p))] = 1.0
        return out
    with np.errstate(divide="ignore"):
        logit = np.log(p) / T
    logit -= logit[np.isfinite(logit)].max()
    q = np.exp(logit)
    return q / q.sum()


def lift_to_tuples(sequences: Sequence[Sequence[Hashable]], order: int):
    """Order-k chain as a first-order chain on k-tuples.

    Returns (states, lifted_sequences): each lifted sequence is the sequence of
    sliding k-tuples, so a transition (a,b,c) -> (b,c,d) is one first-order step.
    """
    lifted, seen = [], {}
    for seq in sequences:
        t = [tuple(seq[i:i + order]) for i in range(len(seq) - order + 1)]
        lifted.append(t)
        for s in t:
            seen.setdefault(s, len(seen))
    return list(seen), lifted


# ---------------------------------------------------------------------------
# Part B: n-gram language model
# ---------------------------------------------------------------------------
def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"[.!?]+", text.lower()) if s.strip()]


def word_tokens(sentence: str) -> List[str]:
    return re.findall(r"[a-z']+|[,;]", sentence.lower())


def char_tokens(sentence: str) -> List[str]:
    return list(re.sub(r"\s+", " ", sentence.lower()).strip())


def detokenize_words(tokens: Sequence[str]) -> str:
    s = " ".join(tokens)
    s = re.sub(r" ([,;])", r"\1", s)
    return s[:1].upper() + s[1:] + "."


class MarkovChain:
    """Order-k Markov chain over tokens: P(w | previous k tokens).

    The state is the k-tuple of preceding tokens (padded with BOS), i.e. a
    first-order chain on tuples.  order=0 is the unigram model.

    smoothing: 'none' (MLE, Eq. 3), 'addk' (Eq. 8) or 'kn' (Eq. 9).
    """

    def __init__(self, order: int = 2, smoothing: str = "none", k: float = 1.0, discount: float = 0.75):
        assert smoothing in ("none", "addk", "kn")
        self.order, self.smoothing, self.k, self.d = order, smoothing, k, discount

    # ---- training ---------------------------------------------------------
    def fit(self, sequences: Sequence[Sequence[str]]) -> "MarkovChain":
        n = self.order
        self.vocab = sorted({t for s in sequences for t in s} | {EOS, UNK})
        self.V = len(self.vocab)
        self.tok2id = {t: i for i, t in enumerate(self.vocab)}
        # full-order counts c(context, w), the sufficient statistic for Eq. (3)
        self.counts: Dict[tuple, Counter] = defaultdict(Counter)
        grams = set()
        for s in sequences:
            padded = [BOS] * n + list(s) + [EOS]
            for i in range(n, len(padded)):
                self.counts[tuple(padded[i - n:i])][padded[i]] += 1
                grams.add(tuple(padded[i - n:i + 1]))
        self.totals = {h: sum(c.values()) for h, c in self.counts.items()}
        if self.smoothing == "kn":
            self._build_kn(grams)
        self.seen_vocab = set(self.vocab)
        return self

    def _build_kn(self, grams):
        """Continuation counts for the lower orders of Eq. (9).

        level[L] maps an L-gram (context + word) to its count: raw count at the
        top level L=order+1, otherwise N1+(. g) = number of distinct left words v
        such that (v,)+g was seen.
        """
        n = self.order
        levels: Dict[int, Counter] = {n + 1: Counter()}
        for s_counts_h, c in self.counts.items():
            for w, cnt in c.items():
                levels[n + 1][s_counts_h + (w,)] = cnt
        # distinct j-grams for lower j are needed only through their extensions
        distinct = {n + 1: set(levels[n + 1])}
        for L in range(n, 0, -1):
            distinct[L] = {g[1:] for g in distinct[L + 1]}   # drop the oldest token
            lv = Counter()
            for g in distinct[L + 1]:
                lv[g[1:]] += 1                               # one new left word per distinct (L+1)-gram
            levels[L] = lv
        self.levels = levels
        self.ctx_tot, self.ctx_types = {}, {}
        for L, lv in levels.items():
            tot, typ = Counter(), Counter()
            for g, c in lv.items():
                tot[g[:-1]] += c
                typ[g[:-1]] += 1
            self.ctx_tot[L], self.ctx_types[L] = tot, typ

    # ---- probabilities ----------------------------------------------------
    def _context(self, context: Sequence[str]) -> tuple:
        """Last `order` tokens, BOS-padded on the left, unseen tokens mapped to UNK."""
        n = self.order
        padded = [BOS] * n + [t if t in self.seen_vocab else UNK for t in context]
        return tuple(padded[len(padded) - n:]) if n else ()

    def prob(self, context: Sequence[str], w: str) -> float:
        """P(w | last `order` tokens of context) under the chosen smoothing."""
        h = self._context(context)
        w = w if w in self.seen_vocab else UNK
        if self.smoothing == "kn":
            return self._kn(h, w, self.order + 1)
        c = self.counts.get(h)
        cw = c[w] if c else 0
        tot = self.totals.get(h, 0)
        if self.smoothing == "addk":
            return (cw + self.k) / (tot + self.k * self.V)          # Eq. (8)
        return cw / tot if tot else 0.0                             # Eq. (3)

    def _kn(self, h: tuple, w: str, L: int) -> float:
        """Eq. (9), recursive over the context length."""
        if L == 1:       # empty context: continuation unigram mixed with uniform
            tot, typ, c = self.ctx_tot[1].get((), 0), self.ctx_types[1].get((), 0), self.levels[1].get((w,), 0)
            return max(c - self.d, 0) / tot + self.d * typ / tot / self.V
        lower = self._kn(h[1:], w, L - 1)
        tot = self.ctx_tot[L].get(h, 0)
        if tot == 0:
            return lower
        c = self.levels[L].get(h + (w,), 0)
        return max(c - self.d, 0) / tot + self.d * self.ctx_types[L][h] / tot * lower

    def distribution(self, context: Sequence[str]) -> np.ndarray:
        """Full distribution over self.vocab given context (sums to 1 when defined)."""
        return np.array([self.prob(context, w) for w in self.vocab])

    # ---- evaluation -------------------------------------------------------
    def log2_probs(self, sequences: Sequence[Sequence[str]]) -> np.ndarray:
        out = []
        for s in sequences:
            toks = list(s) + [EOS]
            for i, w in enumerate(toks):
                p = self.prob(toks[max(0, i - self.order):i], w)
                out.append(np.log2(p) if p > 0 else -np.inf)
        return np.array(out)

    def cross_entropy(self, sequences) -> float:
        """Eq. (10), in bits per token."""
        return float(-self.log2_probs(sequences).mean())

    def perplexity(self, sequences) -> float:
        """Eq. (11)."""
        return float(2.0 ** self.cross_entropy(sequences))

    # ---- generation -------------------------------------------------------
    def generate(self, rng: np.random.Generator, temperature: float = 1.0, max_len: int = 60,
                 prefix: Sequence[str] = ()) -> List[str]:
        """Sample tokens until EOS.  Unseen contexts back off to shorter ones."""
        out = list(prefix)
        while len(out) < max_len:
            ctx = out[-self.order:] if self.order else []
            ids, p = self._next(ctx)
            q = temperature_scale(p, temperature)
            w = ids[sample_inverse_cdf(q, rng.random())]
            if w == EOS:
                break
            out.append(w)
        return out

    def _next(self, ctx):
        """(candidate tokens, probabilities) for the next token."""
        if self.smoothing == "none":
            c = self.counts.get(self._context(ctx))
            if not c:                       # user prefix never seen: fall back to unigram counts
                c = Counter()
                for cc in self.counts.values():
                    c.update(cc)
            ids = list(c)
            p = np.array([c[t] for t in ids], float)
            return ids, p / p.sum()
        p = self.distribution(ctx)
        return self.vocab, p / p.sum()


def ngram_overlap(generated: Sequence[Sequence[str]], train: Sequence[Sequence[str]], n: int) -> float:
    """Fraction of generated n-grams that occur verbatim in the training text (1 = pure copying)."""
    def grams(seqs):
        return [tuple(s[i:i + n]) for s in seqs for i in range(len(s) - n + 1)]
    ref = set(grams(train))
    g = grams(generated)
    return float(np.mean([x in ref for x in g])) if g else float("nan")
