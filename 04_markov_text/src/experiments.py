"""Shared experiment code for demo.py, make_figures.py and the README numbers."""
from pathlib import Path

import numpy as np

import markov as mk

ROOT = Path(__file__).resolve().parents[1]
TOY = np.array([[0.7, 0.2, 0.1], [0.3, 0.4, 0.3], [0.2, 0.3, 0.5]])   # sunny / cloudy / rainy
TOY_NAMES = ["sunny", "cloudy", "rainy"]


def load_sentences(level="word"):
    text = (ROOT / "data" / "corpus.txt").read_text()
    tok = mk.word_tokens if level == "word" else mk.char_tokens
    return [tok(s) for s in mk.split_sentences(text)]


def train_test_split(seqs, test_frac=0.2, seed=0):
    idx = np.random.default_rng(seed).permutation(len(seqs))
    n_test = int(round(test_frac * len(seqs)))
    test = [seqs[i] for i in sorted(idx[:n_test])]
    train = [seqs[i] for i in sorted(idx[n_test:])]
    return train, test


def perplexity_table(train, test, orders, k=0.05):
    """rows: order -> dict of perplexities for add-k (train/test), KN (train/test), MLE test."""
    rows = {}
    for o in orders:
        ak = mk.MarkovChain(o, "addk", k=k).fit(train)
        kn = mk.MarkovChain(o, "kn").fit(train)
        ml = mk.MarkovChain(o, "none").fit(train)
        rows[o] = dict(addk_train=ak.perplexity(train), addk_test=ak.perplexity(test),
                       kn_train=kn.perplexity(train), kn_test=kn.perplexity(test),
                       mle_train=ml.perplexity(train), mle_test=ml.perplexity(test))
    return rows


def overlap_table(train, orders, n_gen=300, n=4, seed=0):
    """Generate with the unsmoothed chain; measure copying of the training text."""
    rows = {}
    train_sents = {tuple(s) for s in train}
    for o in orders:
        m = mk.MarkovChain(o, "none").fit(train)
        rng = np.random.default_rng(seed)
        gens = [m.generate(rng) for _ in range(n_gen)]
        det = np.mean([len(c) == 1 for c in m.counts.values()])     # contexts with a single successor
        rows[o] = dict(overlap=mk.ngram_overlap(gens, train, n),
                       verbatim=float(np.mean([tuple(g) in train_sents for g in gens])),
                       deterministic=float(det), contexts=len(m.counts),
                       mean_len=float(np.mean([len(g) for g in gens])),
                       sample=mk.detokenize_words(gens[0]))
    return rows
