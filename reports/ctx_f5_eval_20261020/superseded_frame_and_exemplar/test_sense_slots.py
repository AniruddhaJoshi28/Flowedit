"""
Unit tests for grammatical-frame sense memory (flowedit/memory/sense_slots.py) and its retrieval hook.
"""

import torch

from flowedit.config import MemoryConfig
from flowedit.memory import sense_slots as ss
from flowedit.memory.hopfield_memory import HopfieldMemory


def at(text, word="wound"):
    return text.lower().index(word)


def ex(sense, text, word="wound"):
    return ss.SenseExample(sense, text, at(text, word))


def test_frames():
    assert ss.frame("The nurse cleaned the wound.", at("The nurse cleaned the wound.")) == ("DET", "END")
    assert ss.frame("He had a deep wound in his leg.", at("He had a deep wound in his leg.")) == ("DET_MOD", "PREP")
    assert ss.frame("She wound the clock.", at("She wound the clock.")) == ("PRON", "DET")
    assert ss.frame("I have already read it.", at("I have already read it.", "read")) == ("PERF", "OBJ")
    assert ss.frame("Read the label.", 0) == ("START", "DET")


def test_frame_generalises_across_content_words():
    examples = [ex(0, "The nurse cleaned the wound carefully."), ex(1, "She wound the clock before bed.")]
    for text, want in (("Keep the wound clean.", 0), ("They wound the cable onto the spool.", 1),
                       ("Grandpa wound the old watch.", 1)):
        verdict, k, _ = ss.decide(examples, [0, 1], text, at(text))
        assert (verdict, k) == ("choose", want), text


def test_single_sense_not_forced_on_other_role():
    examples = [ex(0, "The nurse cleaned the wound carefully.")]
    verdict, _, info = ss.decide(examples, [0], "She wound the clock.", at("She wound the clock."))
    assert verdict == "abstain" and info["role"] == "verb"
    verdict, k, _ = ss.decide(examples, [0], "A deep wound needs stitches.", at("A deep wound needs stitches."))
    assert verdict == "choose" and k == 0


def test_undecided_without_evidence():
    examples = [ex(0, "A wound like that hurts."), ex(1, "The wound string snapped.")]
    verdict, _, _ = ss.decide(examples, [0, 1], "Wound, he said.", 0)
    assert verdict == "undecided"


_TABLE = torch.randn(256, 32, generator=torch.Generator().manual_seed(1))


def _enc(text, language="en"):
    return _TABLE[torch.tensor([ord(c) % 256 for c in text])].unsqueeze(0)


def _memory(slots):
    mem = HopfieldMemory(MemoryConfig(homograph_sim_threshold=0.0, sense_slots=slots), embedding_dim=32,
                         text_encoder=_enc, sense_encoder=None, sense_lexicon=None)
    carrier = "The nurse cleaned the wound carefully."
    mem.write(key=mem._carrier_key(carrier, "wound"), value=torch.zeros(32), word="wound",
              carrier_text=carrier, phonetic_text="woond")
    return mem


def test_retrieval_hook_off_by_default_and_on_when_enabled():
    text = "She wound the clock."
    cands = lambda m: [(0, m.entries[0])]  # noqa: E731
    key = lambda m: m.compute_context_key(_enc(text), list(range(4, 9)))  # noqa: E731
    off = _memory(False)
    assert off._select_sense(cands(off), None, lambda: key(off), occ_text=text, occ_start=4) is not None
    on = _memory(True)
    assert on._select_sense(cands(on), None, lambda: key(on), occ_text=text, occ_start=4) is None
    assert on.last_sense_decisions[-1]["verdict"] == "abstain"
