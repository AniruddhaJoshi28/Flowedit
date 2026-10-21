"""
Unit tests for exemplar sense keys: a homograph sense keeps one context key per correction sentence.
"""

import torch

from flowedit.config import MemoryConfig
from flowedit.memory.hopfield_memory import HopfieldMemory

DIM = 64
_TABLE = torch.randn(256, DIM, generator=torch.Generator().manual_seed(0))


def char_encoder(text, language="en"):
    """Deterministic character-level stand-in for a backbone text encoder."""
    return _TABLE[torch.tensor([ord(c) % 256 for c in text])].unsqueeze(0)


def make_memory(**overrides):
    cfg = MemoryConfig(homograph_sim_threshold=0.0, **overrides)
    return HopfieldMemory(cfg, embedding_dim=DIM, text_encoder=char_encoder, sense_encoder=None, sense_lexicon=None)


def learn(mem, carrier, spelling):
    key = mem._carrier_key(carrier, "lead")
    return mem.write(key=key, value=torch.zeros(DIM), word="lead", carrier_text=carrier, phonetic_text=spelling)


def occurrence_key(mem, text):
    span = mem._find_word_char_spans(text, "lead")[0]
    return mem.compute_context_key(char_encoder(text), list(range(*span)))


def senses(mem):
    return [(i, e) for i, e in enumerate(mem.entries)]


def build(mem):
    learn(mem, "This toy is made of lead.", "led")
    learn(mem, "Paint that contains lead is unsafe.", "led")   # same sense, second context
    learn(mem, "She is the lead of data science.", "leed")


def test_off_by_default_uses_pooled_key():
    mem = make_memory()
    build(mem)
    assert not mem.config.sense_exemplar_keys
    text = "The toy is made of lead."
    chosen = mem._select_sense(senses(mem), None, lambda: occurrence_key(mem, text))
    assert chosen is not None
    assert mem.last_sense_decisions == []


def test_second_correction_keeps_first_context():
    mem = make_memory(sense_exemplar_keys=True)
    build(mem)
    led = mem.entries[0]
    assert len(led.carrier_texts) == 2 and mem.num_entries == 2  # merged, not a new sense
    text = "This toy is made of lead."
    chosen = mem._select_sense(senses(mem), None, lambda: occurrence_key(mem, text))
    d = mem.last_sense_decisions[-1]
    assert chosen[0] == 0
    assert abs(d["exemplar_sims"][0] - 1.0) < 1e-5            # exact stored context still recalled
    assert d["exemplar_sims"][0] >= d["pooled_sims"][0]       # pooled key drifted toward the newer carrier


def test_exemplar_choice_follows_nearest_context():
    mem = make_memory(sense_exemplar_keys=True)
    build(mem)
    for text, expected in (("She is the lead of data science.", 1), ("Paint that contains lead is unsafe.", 0)):
        chosen = mem._select_sense(senses(mem), None, lambda t=text: occurrence_key(mem, t))
        assert chosen[0] == expected
        assert mem.entries[chosen[0]].phonetic_text == ("leed" if expected else "led")


def test_margin_abstains_on_close_call():
    mem = make_memory(sense_exemplar_keys=True, sense_exemplar_margin=10.0)
    build(mem)
    chosen = mem._select_sense(senses(mem), None, lambda: occurrence_key(mem, "We will lead them home."))
    assert chosen is None
    assert mem.last_sense_decisions[-1]["abstained"]


def test_single_pronunciation_unaffected():
    mem = make_memory(sense_exemplar_keys=True)
    learn(mem, "This toy is made of lead.", "led")
    chosen = mem._select_sense(senses(mem), None, lambda: occurrence_key(mem, "Old pipes contain lead."))
    assert chosen[0] == 0
    assert mem.last_sense_decisions == []
