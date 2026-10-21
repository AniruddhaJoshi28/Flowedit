"""
Unit tests for the context sense model (flowedit/memory/sense_context.py) and its memory integration.
"""

import torch

from flowedit.config import MemoryConfig
from flowedit.memory import sense_context as sc
from flowedit.memory.hopfield_memory import HopfieldMemory

W = 2


def ctx(text, word="wound"):
    return sc.context(text, text.lower().index(word), W)


def background(*texts):
    bg = sc.Background()
    for t in texts:
        bg.observe(t)
    return bg


def test_context_features_and_edges():
    assert ctx("She wound the clock.") == [(-1, "she"), (1, "the"), (-2, "<s>"), (2, "clock")]
    assert ctx("Clean the wound")[1] == (1, "</s>")


def test_lambda_prior_without_repeat_corrections():
    s = sc.SenseData("woond", [ctx("The nurse cleaned the wound.")])
    lam = sc.fit_lambda([s], background("The nurse cleaned the wound."), W)
    assert all(abs(v - 0.5) < 1e-9 for v in lam.values())


def test_lambda_learns_which_positions_recur():
    texts = ["She wound the clock.", "He wound the rope.", "They wound the cable.", "I wound the string."]
    s = sc.SenseData("wownd", [ctx(t) for t in texts])
    lam = sc.fit_lambda([s], background(*texts, "The weather is nice today and the sky is blue."), W)
    assert lam[1] > 0.8                      # "the" always follows: informative
    assert lam[-1] < lam[1] and lam[2] < lam[1]  # subjects and objects vary: less informative


def test_single_pronunciation_applies_only_with_context_support():
    learned = ["She wound the clock.", "He wound the rope."]
    s = sc.SenseData("wownd", [ctx(t) for t in learned])
    bg = background(*learned, "The nurse cleaned the wound.", "A deep wound in his leg.")
    lam = sc.fit_lambda([s], bg, W)
    verdict, _, _ = sc.decide(ctx("They wound the cable tightly."), [s], lam, bg)
    assert verdict == "choose"
    verdict, _, info = sc.decide(ctx("The nurse cleaned the wound."), [s], lam, bg)
    assert verdict == "abstain" and info["posterior"][1] > info["posterior"][0]


def test_several_pronunciations_compete_and_abstain_from_guessing():
    a = sc.SenseData("woond", [ctx("The nurse cleaned the wound carefully."), ctx("A deep wound in his leg.")])
    b = sc.SenseData("wownd", [ctx("She wound the clock."), ctx("He wound the rope.")])
    bg = background("The nurse cleaned the wound carefully.", "A deep wound in his leg.",
                    "She wound the clock.", "He wound the rope.")
    lam = sc.fit_lambda([a, b], bg, W)
    assert sc.decide(ctx("We wound the yarn."), [a, b], lam, bg)[:2] == ("choose", 1)
    assert sc.decide(ctx("Keep the wound clean."), [a, b], lam, bg)[:2] == ("choose", 0)
    assert sc.decide(ctx("Doctors say wound tissue heals slowly."), [a, b], lam, bg)[0] == "undecided"


_TABLE = torch.randn(256, 32, generator=torch.Generator().manual_seed(1))


def _enc(text, language="en"):
    return _TABLE[torch.tensor([ord(c) % 256 for c in text])].unsqueeze(0)


def _memory():
    mem = HopfieldMemory(MemoryConfig(homograph_sim_threshold=0.0, sense_context_window=W),
                         embedding_dim=32, text_encoder=_enc)
    for carrier, spelling in (("She wound the clock.", "wownd"), ("He wound the rope.", "wownd")):
        mem.write(key=mem._carrier_key(carrier, "wound"), value=torch.zeros(32), word="wound",
                  carrier_text=carrier, phonetic_text=spelling)
    return mem


def _applied(mem, text):
    r = mem.retrieve(_enc(text), text=text)
    return [s.phonetic_text for s in r.matched_spans]


def test_memory_applies_correction_only_where_context_supports_it():
    mem = _memory()
    assert _applied(mem, "They wound the cable.") == ["wownd"]
    assert _applied(mem, "The nurse cleaned the wound.") == []
    assert mem.last_sense_decisions[-1]["verdict"] == "abstain"


def test_background_survives_save_and_load(tmp_path):
    mem = _memory()
    mem.retrieve(_enc("Totally new words here."), text="Totally new words here.")
    path = tmp_path / "mem.pt"
    mem.save(str(path))
    other = HopfieldMemory(MemoryConfig(sense_context_window=W), embedding_dim=32, text_encoder=_enc)
    other.load(str(path))
    assert other.background.counts == mem.background.counts
    assert other.background.counts["totally"] == 1
