"""End-to-end demo (< 30 s):  python demo.py   -- every number quoted in the README comes from here."""
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import experiments as ex  # noqa: E402
import markov as mk  # noqa: E402

np.set_printoptions(precision=4, suppress=True)
t0 = time.time()


def head(s):
    print("\n" + "=" * 8, s, "=" * 8)


# ---------------------------------------------------------------- 1. worked example
head("1. Worked example: three sentences, counts -> MLE")
tiny = [["the", "cat", "sat"], ["the", "cat", "ran"], ["the", "dog", "sat"]]
seqs = [[mk.BOS] + s + [mk.EOS] for s in tiny]
states = [mk.BOS, "the", "cat", "dog", "sat", "ran", mk.EOS]
C = mk.count_matrix(seqs, states)
P = mk.mle_transition_matrix(C)
print("states:", states)
print("counts C:\n", C.astype(int))
print("P = C / row sums:\n", P)
print("row-stochastic:", mk.is_row_stochastic(P))
m1 = mk.MarkovChain(1, "none").fit(tiny)
m2 = mk.MarkovChain(2, "none").fit(tiny)
print("P(cat|the)=%.4f  P(dog|the)=%.4f  P(sat|cat)=%.4f" % (m1.prob(["the"], "cat"), m1.prob(["the"], "dog"), m1.prob(["cat"], "sat")))
print("order 2: P(sat|the cat)=%.4f  P(sat|the dog)=%.4f" % (m2.prob(["the", "cat"], "sat"), m2.prob(["the", "dog"], "sat")))
sent = ["the", "cat", "sat"]
lp = m1.log2_probs([sent])
print("sentence 'the cat sat </s>' log2-probs:", lp, " cross-entropy %.4f bits, perplexity %.4f" % (m1.cross_entropy([sent]), m1.perplexity([sent])))
ak = mk.MarkovChain(1, "addk", k=1.0).fit(tiny)
print("V=%d  add-1: P(cat|the)=%.4f  P(ran|the)=%.4f (MLE: %.1f)" % (ak.V, ak.prob(["the"], "cat"), ak.prob(["the"], "ran"), m1.prob(["the"], "ran")))
print("MLE perplexity of unseen 'the dog ran':", m1.perplexity([["the", "dog", "ran"]]), "| add-1:", round(ak.perplexity([["the", "dog", "ran"]]), 4))

padded2 = [[mk.BOS, mk.BOS] + t + [mk.EOS] for t in tiny]
st2, _ = mk.lift_to_tuples(padded2, 2)
print("order-2 chain on the tiny corpus = first-order chain on %d bigram states" % len(st2))
Cex = np.array([[6, 2], [1, 3]])
Pex = mk.mle_transition_matrix(Cex)
print("exercise 1: counts [[6,2],[1,3]] -> P=%s pi=%s" % (Pex.tolist(), mk.stationary_eig(Pex)))

# ---------------------------------------------------------------- 2. toy chain
head("2. Toy weather chain: n-step, stationary, convergence")
P = ex.TOY
print("P:\n", P)
print("P^2:\n", mk.n_step(P, 2))
print("P^5 row sunny:", mk.n_step(P, 5)[0])
print("P^50 rows:\n", mk.n_step(P, 50))
pi = mk.stationary_eig(P)
print("pi (eig of P^T):", pi, " pi P - pi max abs:", np.abs(pi @ P - pi).max())
pw, hist = mk.stationary_power(P, np.array([1.0, 0, 0]), tol=1e-12)
print("power iteration: %d iterations to 1e-12, pi=%s" % (len(hist), pw))
ev = np.linalg.eigvals(P)
print("eigenvalues:", np.sort(np.abs(ev))[::-1], " irreducible:", mk.is_irreducible(P), " period:", mk.period(P))
print("detailed-balance gap (P is not reversible):", round(mk.detailed_balance_gap(P, pi), 4))
W = np.array([[0, 2, 1, 0], [2, 0, 3, 1], [1, 3, 0, 4], [0, 1, 4, 0]], float)
PW = W / W.sum(1, keepdims=True)
piW = W.sum(1) / W.sum()
print("random walk on weighted graph: pi=W-row-sums/total=%s, detailed-balance gap %.1e" % (piW, mk.detailed_balance_gap(PW, piW)))
cyc = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0.0]])
print("3-cycle: period", mk.period(cyc), " P^n[0] for n=1..4:", [mk.n_step(cyc, n)[0].astype(int).tolist() for n in range(1, 5)])
P2 = np.array([[0.7, 0.3], [0.1, 0.9]])                          # two-state chain: a=0.3, b=0.1
print("two-state chain a=0.3,b=0.1: pi=%s (closed form (b,a)/(a+b)=(0.25,0.75)), lambda_2=%s (1-a-b=0.6)" % (mk.stationary_eig(P2), np.sort(np.linalg.eigvals(P2).real)[0]))
print("  detailed balance gap (any 2-state chain is reversible): %.1e" % mk.detailed_balance_gap(P2, mk.stationary_eig(P2)))
rng = np.random.default_rng(0)
ends = np.array([mk.simulate_states(P, 0, 5, rng)[-1] for _ in range(20000)])
print("P^5[sunny] exact:", mk.n_step(P, 5)[0], " simulated (20000 runs):", np.bincount(ends, minlength=3) / 20000)
path = mk.simulate_states(P, 0, 100000, np.random.default_rng(1))
print("long-run time fractions:", np.bincount(path) / len(path))

# ---------------------------------------------------------------- 3. corpus
head("3. Corpus and the word chain")
words = ex.load_sentences("word")
N = sum(len(s) + 1 for s in words)
print("%d sentences, %d tokens (+EOS: %d), %d word types" % (len(words), sum(map(len, words)), N, len({t for s in words for t in s})))
wrapped = [[mk.BOS] + s + [mk.EOS, mk.BOS] for s in words]       # close the chain EOS -> BOS
st = sorted({t for s in wrapped for t in s})
Pw = mk.mle_transition_matrix(mk.count_matrix(wrapped, st))
piw = mk.stationary_eig(Pw)
freq = Counter(t for s in words for t in s)
freq[mk.EOS] = len(words)
freq[mk.BOS] = len(words)
tot = sum(freq.values())
emp = np.array([freq[t] / tot for t in st])
print("word chain: irreducible=%s period=%d" % (mk.is_irreducible(Pw), mk.period(Pw)))
print("max |pi - empirical frequency| = %.2e  (pi_w = count_w / total)" % np.abs(piw - emp).max())
top = np.argsort(-piw)[:5]
print("top stationary states:", [(st[i], round(float(piw[i]), 4)) for i in top])

# ---------------------------------------------------------------- 4. perplexity
head("4. Perplexity vs order (train 80% / test 20% of sentences)")
train, test = ex.train_test_split(words)
print("train %d sentences, test %d sentences" % (len(train), len(test)))
uni = mk.MarkovChain(1, "addk", k=1e9).fit(train)
print("V = %d;  perplexity of the uniform model = %.4f" % (uni.V, uni.perplexity(test)))
unk = sum(w not in uni.seen_vocab for s in test for w in s) / sum(len(s) for s in test)
print("test tokens unseen in training: %.1f%%" % (100 * unk))
rows = ex.perplexity_table(train, test, range(0, 6))
print("order | add-0.05 train  test | KN train  test | MLE train  test")
for o, r in rows.items():
    print("  %d   | %8.2f %8.2f | %7.2f %7.2f | %7.2f %s" % (o, r["addk_train"], r["addk_test"], r["kn_train"], r["kn_test"], r["mle_train"], "inf" if np.isinf(r["mle_test"]) else "%.2f" % r["mle_test"]))

# ---------------------------------------------------------------- 5. overlap
head("5. Coherence vs memorisation (300 samples/order, unsmoothed chain trained on train split)")
orows = ex.overlap_table(train, range(1, 6))
print("order | contexts | single-successor | 4-gram overlap | verbatim sentences | mean len")
for o, r in orows.items():
    print("  %d   | %6d   | %8.3f         | %8.3f       | %8.3f           | %5.1f" % (o, r["contexts"], r["deterministic"], r["overlap"], r["verbatim"], r["mean_len"]))
    print("      sample:", r["sample"])

# ---------------------------------------------------------------- 6. temperature
head("6. Temperature (word order 2, full corpus, seed 7)")
m = mk.MarkovChain(2, "kn").fit(words)
mm = mk.MarkovChain(2, "none").fit(words)
for T in (0.3, 1.0, 2.0):
    r = np.random.default_rng(7)
    print("T=%.1f:" % T, mk.detokenize_words(mm.generate(r, temperature=T)))
p = mm.distribution(["the", "harbor"])
for T in (0.5, 1.0, 2.0):
    q = mk.temperature_scale(p[p > 0], T)
    print("T=%.1f next-word dist after 'the harbor' (%d candidates): max prob %.3f, entropy %.3f bits" % (T, (p > 0).sum(), q.max(), -(q * np.log2(q)).sum()))
print("--- KN-smoothed generation (order 2) puts mass on unseen words:")
print(mk.detokenize_words(m.generate(np.random.default_rng(7))))

# ---------------------------------------------------------------- 7. char level
head("7. Character-level chains")
chars = ex.load_sentences("char")
ctr, cte = ex.train_test_split(chars)
print("character vocabulary (train + EOS + UNK):", mk.MarkovChain(1).fit(ctr).V)
print("order | KN test perplexity | sample (unsmoothed chain, full text)")
for o in (0, 1, 2, 3, 5, 8):
    pp = mk.MarkovChain(o, "kn").fit(ctr).perplexity(cte)
    g = mk.MarkovChain(o, "none").fit(chars).generate(np.random.default_rng(3), max_len=90)
    s = "".join(g)
    print("  %d   | %8.3f           | %s" % (o, pp, s))

print("\nelapsed %.1f s" % (time.time() - t0))
