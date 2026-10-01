import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import markov as mk  # noqa: E402

TINY = [["the", "cat", "sat"], ["the", "cat", "ran"], ["the", "dog", "sat"]]
TOY = np.array([[0.7, 0.2, 0.1], [0.3, 0.4, 0.3], [0.2, 0.3, 0.5]])


@pytest.fixture(scope="module")
def corpus():
    text = (ROOT / "data" / "corpus.txt").read_text()
    return [mk.word_tokens(s) for s in mk.split_sentences(text)]


# ---- rows sum to 1 --------------------------------------------------------
def test_rows_sum_to_one_mle_and_smoothed(corpus):
    toks = [[mk.BOS] + s + [mk.EOS] for s in corpus]
    states = sorted({t for s in toks for t in s})
    C = mk.count_matrix(toks, states)
    for k in (0.0, 0.1, 1.0):
        P = mk.mle_transition_matrix(C, k)
        assert mk.is_row_stochastic(P)


@pytest.mark.parametrize("smoothing", ["none", "addk", "kn"])
@pytest.mark.parametrize("order", [0, 1, 2])
def test_model_distribution_sums_to_one(corpus, smoothing, order):
    m = mk.MarkovChain(order, smoothing, k=0.5).fit(corpus)
    for ctx in (["the"], ["the", "harbor"], [], ["zzz-unseen"]):
        p = m.distribution(ctx)
        assert np.all(p >= 0)
        if smoothing != "none" or ctx in (["the"], ["the", "harbor"], []):
            assert p.sum() == pytest.approx(1.0, abs=1e-9)


# ---- MLE vs hand count ----------------------------------------------------
def test_mle_matches_hand_count():
    seqs = [[mk.BOS] + s + [mk.EOS] for s in TINY]
    states = [mk.BOS, "the", "cat", "dog", "sat", "ran", mk.EOS]
    P = mk.mle_transition_matrix(mk.count_matrix(seqs, states))
    ix = {s: i for i, s in enumerate(states)}
    assert P[ix["the"], ix["cat"]] == pytest.approx(2 / 3)
    assert P[ix["the"], ix["dog"]] == pytest.approx(1 / 3)
    assert P[ix["cat"], ix["sat"]] == pytest.approx(1 / 2)
    assert P[ix["sat"], ix[mk.EOS]] == pytest.approx(1.0)
    assert P[ix[mk.BOS], ix["the"]] == pytest.approx(1.0)


def test_mle_model_matches_hand_count():
    m = mk.MarkovChain(1, "none").fit(TINY)
    assert m.prob(["the"], "cat") == pytest.approx(2 / 3)
    assert m.prob(["the"], "dog") == pytest.approx(1 / 3)
    assert m.prob(["cat"], "ran") == pytest.approx(1 / 2)
    assert m.prob(["dog"], "ran") == 0.0
    m2 = mk.MarkovChain(2, "none").fit(TINY)
    assert m2.prob(["the", "cat"], "sat") == pytest.approx(1 / 2)
    assert m2.prob(["the", "dog"], "sat") == pytest.approx(1.0)


def test_mle_maximises_likelihood():
    """Property: no other row distribution gives a higher multinomial log-likelihood."""
    rng = np.random.default_rng(0)
    c = np.array([5.0, 2.0, 0.0, 3.0])
    mle = c / c.sum()
    ll = lambda p: float(np.sum(c[c > 0] * np.log(p[c > 0])))
    for _ in range(200):
        q = rng.dirichlet(np.ones(4))
        assert ll(q) <= ll(mle) + 1e-12


def test_add_k_formula():
    m = mk.MarkovChain(1, "addk", k=0.5).fit(TINY)
    V = m.V                                   # 5 words + EOS + UNK = 7
    assert V == 7
    assert m.prob(["the"], "cat") == pytest.approx((2 + 0.5) / (3 + 0.5 * V))
    assert m.prob(["the"], "ran") == pytest.approx(0.5 / (3 + 0.5 * V))


# ---- stationary distribution ---------------------------------------------
def test_stationary_satisfies_pi_P_equals_pi():
    pi = mk.stationary_eig(TOY)
    assert pi.sum() == pytest.approx(1.0) and np.all(pi > 0)
    np.testing.assert_allclose(pi @ TOY, pi, atol=1e-12)


def test_power_iteration_matches_eig_and_rows_of_Pn():
    pi, hist = mk.stationary_power(TOY)
    np.testing.assert_allclose(pi, mk.stationary_eig(TOY), atol=1e-10)
    Pn = mk.n_step(TOY, 60)
    for row in Pn:
        np.testing.assert_allclose(row, pi, atol=1e-10)
    assert hist[-1] < hist[0]


def test_convergence_rate_is_second_eigenvalue():
    lam2 = sorted(np.abs(np.linalg.eigvals(TOY)))[-2]
    pi = mk.stationary_eig(TOY)
    e = [np.abs(mk.n_step(TOY, n)[0] - pi).sum() for n in (10, 11)]
    assert e[1] / e[0] == pytest.approx(lam2, rel=0.2)


def test_detailed_balance_random_walk_on_graph():
    W = np.array([[0, 2, 1, 0], [2, 0, 3, 1], [1, 3, 0, 4], [0, 1, 4, 0]], float)
    P = W / W.sum(1, keepdims=True)
    pi = W.sum(1) / W.sum()
    assert mk.detailed_balance_gap(P, pi) < 1e-15
    np.testing.assert_allclose(pi @ P, pi, atol=1e-15)


def test_two_state_closed_form():
    a, b = 0.3, 0.1
    P = np.array([[1 - a, a], [b, 1 - b]])
    np.testing.assert_allclose(mk.stationary_eig(P), [b / (a + b), a / (a + b)], atol=1e-12)
    assert sorted(np.linalg.eigvals(P).real)[0] == pytest.approx(1 - a - b)
    assert mk.detailed_balance_gap(P, mk.stationary_eig(P)) < 1e-15


def test_stationary_of_closed_text_chain_is_token_frequency(corpus):
    wrapped = [[mk.BOS] + s + [mk.EOS, mk.BOS] for s in corpus]
    st = sorted({t for s in wrapped for t in s})
    P = mk.mle_transition_matrix(mk.count_matrix(wrapped, st))
    pi = mk.stationary_eig(P)
    from collections import Counter
    f = Counter(t for s in corpus for t in s)
    f[mk.EOS] = f[mk.BOS] = len(corpus)
    tot = sum(f.values())
    np.testing.assert_allclose(pi, [f[t] / tot for t in st], atol=1e-10)


def test_irreducible_and_period():
    cycle = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0.0]])
    assert mk.is_irreducible(cycle) and mk.period(cycle) == 3
    assert mk.is_irreducible(TOY) and mk.period(TOY) == 1
    reducible = np.array([[1, 0], [0.5, 0.5]])
    assert not mk.is_irreducible(reducible)


# ---- matrix power vs simulation ------------------------------------------
def test_matrix_power_vs_simulation():
    rng = np.random.default_rng(123)
    n, runs = 5, 20000
    ends = np.array([mk.simulate_states(TOY, 0, n, rng)[-1] for _ in range(runs)])
    emp = np.bincount(ends, minlength=3) / runs
    np.testing.assert_allclose(emp, mk.n_step(TOY, n)[0], atol=0.015)   # > 4 sigma


def test_chapman_kolmogorov():
    np.testing.assert_allclose(mk.n_step(TOY, 7), mk.n_step(TOY, 3) @ mk.n_step(TOY, 4), atol=1e-14)


# ---- sampling ------------------------------------------------------------
def test_inverse_cdf_boundaries():
    p = np.array([0.2, 0.5, 0.3])
    assert [mk.sample_inverse_cdf(p, u) for u in (0.0, 0.19, 0.2, 0.69, 0.7, 0.999)] == [0, 0, 1, 1, 2, 2]


def test_inverse_cdf_frequencies():
    rng = np.random.default_rng(1)
    p = np.array([0.1, 0.6, 0.3])
    draws = np.array([mk.sample_inverse_cdf(p, rng.random()) for _ in range(20000)])
    np.testing.assert_allclose(np.bincount(draws) / 20000, p, atol=0.015)


def test_temperature_limits():
    p = np.array([0.1, 0.6, 0.3])
    np.testing.assert_allclose(mk.temperature_scale(p, 1.0), p)
    np.testing.assert_allclose(mk.temperature_scale(p, 1e-3), [0, 1, 0], atol=1e-9)
    assert np.abs(mk.temperature_scale(p, 1e3) - 1 / 3).max() < 0.01
    assert mk.temperature_scale(p, 2.0).sum() == pytest.approx(1)
    assert mk.temperature_scale(np.array([0.0, 0.5, 0.5]), 0.5)[0] == 0.0


def test_generation_only_uses_seen_transitions(corpus):
    m = mk.MarkovChain(2, "none").fit(corpus)
    rng = np.random.default_rng(0)
    for _ in range(20):
        g = m.generate(rng)
        padded = [mk.BOS, mk.BOS] + g + ([mk.EOS] if len(g) < 60 else [])  # 60 = max_len cut-off
        for i in range(2, len(padded)):
            assert m.counts[tuple(padded[i - 2:i])][padded[i]] > 0


def test_generation_is_seeded(corpus):
    m = mk.MarkovChain(2, "none").fit(corpus)
    a = m.generate(np.random.default_rng(5))
    b = m.generate(np.random.default_rng(5))
    assert a == b


# ---- order-k as first-order on tuples -------------------------------------
def test_order_k_equals_first_order_on_tuples():
    seqs = [[mk.BOS, mk.BOS] + s + [mk.EOS] for s in TINY]
    states, lifted = mk.lift_to_tuples(seqs, 2)
    P = mk.mle_transition_matrix(mk.count_matrix(lifted, states))
    ix = {s: i for i, s in enumerate(states)}
    m = mk.MarkovChain(2, "none").fit(TINY)
    # (the,cat) -> (cat,sat) has probability P(sat | the cat)
    assert P[ix[("the", "cat")], ix[("cat", "sat")]] == pytest.approx(m.prob(["the", "cat"], "sat"))
    assert mk.is_row_stochastic(P[[ix[s] for s in states if s[1] != mk.EOS]])


# ---- perplexity -----------------------------------------------------------
class Uniform:
    def __init__(self, V): self.V = V
    def prob(self, ctx, w): return 1.0 / self.V


def test_perplexity_of_uniform_model_is_vocab_size():
    for V in (2, 7, 228):
        lp = np.log2([Uniform(V).prob([], "x")] * 50)
        assert 2 ** (-lp.mean()) == pytest.approx(V)


def test_perplexity_of_heavily_smoothed_model_is_vocab_size(corpus):
    m = mk.MarkovChain(1, "addk", k=1e9).fit(corpus)
    assert m.perplexity(corpus) == pytest.approx(m.V, rel=1e-6)


def test_perplexity_of_deterministic_model_is_one():
    m = mk.MarkovChain(1, "none").fit([["a", "b", "c"]] * 3)
    assert m.perplexity([["a", "b", "c"]]) == pytest.approx(1.0)


def test_mle_gives_infinite_test_perplexity_when_unseen():
    m = mk.MarkovChain(1, "none").fit(TINY)
    assert np.isinf(m.perplexity([["the", "dog", "ran"]]))
    assert np.isfinite(mk.MarkovChain(1, "addk", k=0.1).fit(TINY).perplexity([["the", "dog", "ran"]]))


def test_kn_is_a_distribution_and_beats_addk_on_heldout(corpus):
    train, test = corpus[:30], corpus[30:]
    kn = mk.MarkovChain(2, "kn").fit(train)
    for ctx in (["the"], ["the", "harbor"], ["unseen", "tokens"]):
        assert kn.distribution(ctx).sum() == pytest.approx(1.0)
    ak = mk.MarkovChain(2, "addk", k=1.0).fit(train)
    assert kn.perplexity(test) < ak.perplexity(test)


def test_ngram_overlap():
    assert mk.ngram_overlap([["a", "b", "c"]], [["a", "b", "c", "d"]], 2) == 1.0
    assert mk.ngram_overlap([["a", "c", "b"]], [["a", "b", "c", "d"]], 2) == 0.0
