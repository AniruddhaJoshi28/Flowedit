"""Unit tests for word-level multi-scale context profiles (pure functions, CPU only)."""

import torch

from flowedit.memory import context_profile as cp


def test_word_spans_are_generic():
    assert cp.word_char_spans("I'll read it, 42 times.") == [(0, 4), (5, 9), (10, 12), (14, 16), (17, 22)]


def test_alignment_follows_tokenizer_insertions_and_rewrites():
    text = "Hi;there"
    # tokenizer rewrote ';' -> ',' and inserted a space before the second segment
    toks = list("Hi, there")
    m = cp.align_tokens_to_chars(text, toks)
    assert m[0] == 0 and m[1] == 1
    assert m[4:] == [3, 4, 5, 6, 7]
    assert -1 in m[2:4]


def test_radius_actually_expands_beyond_neighbours():
    d = 8
    vecs = torch.eye(d)[:7]  # 7 orthogonal words, target = 3
    valid = torch.ones(7, dtype=torch.bool)
    p = cp.build_profile(vecs, valid, target=3, radii=(1, 2, 4))
    # word 0 (3 to the left) is invisible at r=1, visible at r=4
    assert float(p["L1"][0]) == 0.0
    assert float(p["L4"][0]) > 0.0
    assert float(p["L1"][2]) > 0.0
    # target word never contributes
    for k in p:
        assert float(p[k][3]) == 0.0


def test_function_words_are_kept():
    """No stopword filtering: 'will' vs 'have' produce different left contexts."""
    torch.manual_seed(0)
    table = {w: torch.randn(16) for w in ["i", "will", "have", "read", "the", "book"]}

    def prof(sentence):
        words = sentence.split()
        vecs = torch.stack([table[w] for w in words])
        return cp.build_profile(vecs, torch.ones(len(words), dtype=torch.bool), words.index("read"))

    a, b = prof("i will read the book"), prof("i have read the book")
    assert float(torch.dot(a["L1"], b["L1"])) < 0.99
    assert torch.allclose(a["R1"], b["R1"])


def test_profile_similarity_uniform_and_symmetric():
    p = {"L1": torch.tensor([1.0, 0.0]), "R1": torch.tensor([0.0, 1.0])}
    q = {"L1": torch.tensor([1.0, 0.0]), "R1": torch.tensor([1.0, 0.0])}
    assert abs(cp.profile_similarity(p, q) - 0.5) < 1e-6
    assert cp.profile_similarity(p, q) == cp.profile_similarity(q, p)
    assert cp.profile_similarity(p, None) == 0.0


def test_group_preserving_attention_keeps_other_groups_exact():
    base = torch.tensor([2.0, 1.5, 0.3, -1.0, 0.7])
    group = [0, 1]
    a_base, a_new = cp.group_preserving_attention(base, group, torch.tensor([0.0, 5.0]))
    others = [2, 3, 4]
    assert torch.equal(a_new[others], a_base[others])
    assert torch.isclose(a_new[group].sum(), a_base[group].sum())
    assert torch.isclose(a_new.sum(), torch.tensor(1.0))
    assert a_new[1] > a_new[0]  # mass redistributed towards the contextually preferred entry


def test_ema_profile_normalised():
    a = {"L1": torch.tensor([1.0, 0.0])}
    b = {"L1": torch.tensor([0.0, 1.0]), "S": torch.tensor([1.0, 0.0])}
    m = cp.ema_profile(a, b, 0.8)
    assert torch.isclose(m["L1"].norm(), torch.tensor(1.0)) and "S" in m
