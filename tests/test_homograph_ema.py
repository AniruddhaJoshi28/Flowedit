"""
EMA sense-cluster homograph disambiguation (shared by XTTS and F5-TTS).

Covers: EMA merging vs. sense splitting, per-occurrence sense selection (the same spelling with
two meanings in one sentence), backbone-space isolation between XTTS (d=1024) and F5-TTS (d=512),
legacy memory files, F5-TTS span-anchored delta placement, per-occurrence spelling-store
resolution, the encoder-unavailable fallback, and a guard against built-in word tables.
"""

import os
import re
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from flowedit.config import MemoryConfig
from flowedit.memory.hopfield_memory import HopfieldMemory
from flowedit.memory.sense_encoder import get_sense_encoder
from flowedit.refiner.hopfield_refiner import HopfieldRefiner
from flowedit.backbone.f5tts_wrapper import F5TTSBackbone

SENTENCE = "The old paint contains lead, and Priya is the lead researcher studying it."
METAL = "This toy is made of lead"
LEADER = "Anurag is the lead of data science"


def _encoder_or_skip():
    enc = get_sense_encoder(MemoryConfig())
    if enc is None or not enc.available:
        pytest.skip("contextual sense encoder model not available locally")
    return enc


class CharBackbone:
    """Deterministic character-level stand-in backbone (F5-TTS-like)."""

    def __init__(self, dim=512, char_level=True):
        self.embedding_dim = dim
        self.is_character_level = char_level
        self.embedding_space = f"Stub{'Char' if char_level else 'Tok'}:{dim}"
        self.delta_bounds = (0.35, 3.5)
        self.calls = []
        gen = torch.Generator().manual_seed(0)
        self.table = torch.randn(256, dim, generator=gen)

    def encode_text(self, text, language="en"):
        ids = torch.tensor([ord(c) % 256 for c in text])
        return self.table[ids].unsqueeze(0)

    def synthesize_direct(self, text, speaker_conditioning, **kwargs):
        self.calls.append((text, kwargs.get("text_embedding_delta")))
        return torch.zeros(1, 2400), 24000


def _write(memory, backbone, word, carrier, phonetic=None, word_delta=None, occurrence=0):
    emb = backbone.encode_text(carrier)
    spans = memory._find_word_char_spans(carrier, word)
    cs, ce = spans[occurrence]
    ts, te = memory._char_span_to_token_span(cs, ce, len(carrier), emb.shape[1])
    key = memory.compute_context_key(emb, list(range(ts, te)))
    return memory.write(
        key=key,
        value=torch.zeros(memory.d),
        word=word,
        carrier_text=carrier,
        token_indices=list(range(ts, te)),
        phonetic_text=phonetic,
        word_delta=word_delta,
        char_span=(cs, ce),
    )


# ─────────────────────────────────────────────────────────────────────────────
# EMA clustering
# ─────────────────────────────────────────────────────────────────────────────

def test_same_pronunciation_is_ema_merged_and_prototype_stays_normalised():
    _encoder_or_skip()
    bb = CharBackbone()
    mem = HopfieldMemory(MemoryConfig(), embedding_dim=bb.embedding_dim, embedding_space=bb.embedding_space)
    i1, a1 = _write(mem, bb, "lead", METAL, phonetic="led", word_delta=torch.ones(1, 4, 512))
    p1 = mem.entries[i1].sense_prototype.clone()
    i2, a2 = _write(mem, bb, "lead", "Lead pipes were banned decades ago.", phonetic="led", word_delta=torch.ones(1, 4, 512) * 3)
    assert (a1, a2) == ("inserted", "averaged") and i1 == i2 and mem.num_entries == 1
    e = mem.entries[i1]
    cfg = MemoryConfig()
    # warm-up EMA: decay = ema_warmup_decay → δ = 1·0.2 + 3·0.8
    expected = 1.0 * cfg.ema_warmup_decay + 3.0 * (1 - cfg.ema_warmup_decay)
    assert torch.allclose(e.word_delta, torch.full((1, 4, 512), expected), atol=1e-5)
    assert torch.isclose(e.sense_prototype.norm(), torch.tensor(1.0), atol=1e-5)
    assert not torch.allclose(e.sense_prototype, p1), "prototype must move towards the new context"
    assert len(e.carrier_texts) == 2


def test_different_pronunciation_creates_separate_sense_even_in_similar_context():
    _encoder_or_skip()
    bb = CharBackbone()
    mem = HopfieldMemory(MemoryConfig(), embedding_dim=512, embedding_space=bb.embedding_space)
    _write(mem, bb, "lead", METAL, phonetic="led")
    _, action = _write(mem, bb, "lead", "This toy is made of lead and nothing else", phonetic="leed")
    assert action == "inserted" and mem.num_entries == 2


# ─────────────────────────────────────────────────────────────────────────────
# Per-occurrence disambiguation (the reported failure)
# ─────────────────────────────────────────────────────────────────────────────

def _two_sense_memory(bb):
    mem = HopfieldMemory(MemoryConfig(), embedding_dim=bb.embedding_dim, embedding_space=bb.embedding_space)
    _write(mem, bb, "lead", METAL, phonetic="led")
    _write(mem, bb, "lead", LEADER, phonetic="leed")
    return mem


def test_two_occurrences_in_one_sentence_get_different_senses():
    _encoder_or_skip()
    bb = CharBackbone()
    mem = _two_sense_memory(bb)
    res = mem.retrieve(bb.encode_text(SENTENCE), text=SENTENCE)
    by_pos = {s.char_start: s.phonetic_text for s in res.matched_spans}
    first, second = [m.start() for m in re.finditer(r"\blead\b", SENTENCE)]
    assert by_pos == {first: "led", second: "leed"}


@pytest.mark.parametrize("char_level", [True, False])
def test_refiner_respells_each_occurrence_for_both_backbone_types(char_level):
    _encoder_or_skip()
    bb = CharBackbone(char_level=char_level)
    mem = _two_sense_memory(bb)
    res = HopfieldRefiner(mem, MemoryConfig()).forward(bb, SENTENCE, {"audio_path": "x.wav"})
    assert res.is_modified
    assert bb.calls[-1][0] == "The old paint contains led, and Priya is the leed researcher studying it."


def test_single_learned_sense_leaves_other_meaning_untouched():
    _encoder_or_skip()
    bb = CharBackbone()
    mem = HopfieldMemory(MemoryConfig(), embedding_dim=512, embedding_space=bb.embedding_space)
    _write(mem, bb, "lead", METAL, phonetic="led")
    HopfieldRefiner(mem, MemoryConfig()).forward(bb, SENTENCE, {"audio_path": "x.wav"})
    assert bb.calls[-1][0] == "The old paint contains led, and Priya is the lead researcher studying it."


def test_context_free_entries_apply_everywhere():
    """Names learned without a sentence (carrier == word) are unconditional, as before."""
    bb = CharBackbone()
    mem = HopfieldMemory(MemoryConfig(), embedding_dim=512, embedding_space=bb.embedding_space)
    _write(mem, bb, "Rybrevant", "Rybrevant", phonetic="Rye-breh-vant")
    text = "Rybrevant was prescribed, and Rybrevant worked."
    res = mem.retrieve(bb.encode_text(text), text=text)
    assert [s.phonetic_text for s in res.matched_spans] == ["Rye-breh-vant", "Rye-breh-vant"]


def test_f5_delta_lands_on_its_own_occurrence_only():
    _encoder_or_skip()
    bb = CharBackbone()
    mem = HopfieldMemory(MemoryConfig(), embedding_dim=512, embedding_space=bb.embedding_space)
    _write(mem, bb, "lead", METAL, phonetic="led", word_delta=torch.full((1, 4, 512), 0.1))
    _write(mem, bb, "lead", LEADER, phonetic="leed")
    HopfieldRefiner(mem, MemoryConfig()).forward(bb, SENTENCE, {"audio_path": "x.wav"})
    synth, delta = bb.calls[-1]
    active = (delta[0].abs().sum(-1) > 0).nonzero().flatten().tolist()
    start = synth.index("led,")
    assert active == list(range(start, start + 3)), "δ* must cover exactly the respelled metal occurrence"


# ─────────────────────────────────────────────────────────────────────────────
# Backbone spaces: XTTS (1024) ↔ F5-TTS (512)
# ─────────────────────────────────────────────────────────────────────────────

def test_cross_backbone_transfer_keeps_senses_and_isolates_perturbations(tmp_path):
    _encoder_or_skip()
    xtts = CharBackbone(dim=1024, char_level=False)
    f5 = CharBackbone(dim=512, char_level=True)
    path = str(tmp_path / "corrections.pt")

    mem_x = HopfieldMemory(MemoryConfig(), 1024, xtts.embedding_space, text_encoder=xtts.encode_text)
    _write(mem_x, xtts, "lead", METAL, phonetic="led", word_delta=torch.full((1, 4, 1024), 0.2))
    mem_x.save(path)

    mem_f = HopfieldMemory(MemoryConfig(), 512, f5.embedding_space, text_encoder=f5.encode_text)
    mem_f.load(path)
    e = mem_f.entries[0]
    assert e.key.shape == (512,) and e.word_delta is None and e.delta_space is None
    assert e.phonetic_text == "led" and e.sense_prototype is not None
    expected_key = mem_f._rekey(e)
    assert torch.allclose(e.key, expected_key), "foreign entries are re-keyed in the local space"

    # F5 learns its own perturbation for the same sense; XTTS' δ* must survive the round trip
    _write(mem_f, f5, "lead", "Lead pipes were banned decades ago.", phonetic="led", word_delta=torch.full((1, 4, 512), 0.05))
    assert mem_f.num_entries == 1
    mem_f.save(path)

    mem_x2 = HopfieldMemory(MemoryConfig(), 1024, xtts.embedding_space, text_encoder=xtts.encode_text)
    mem_x2.load(path)
    ex = mem_x2.entries[0]
    assert ex.delta_space == xtts.embedding_space
    assert torch.allclose(ex.word_delta, torch.full((1, 4, 1024), 0.2))
    assert set(ex.space_data) == {xtts.embedding_space, f5.embedding_space}

    res = mem_f.retrieve(f5.encode_text(SENTENCE), text=SENTENCE)
    assert res.matched_spans[0].has_delta  # F5's own δ*, not XTTS'


def test_legacy_file_from_other_backbone_is_rekeyed_without_foreign_deltas(tmp_path):
    path = str(tmp_path / "legacy.pt")
    torch.save({"entries": [{
        "key": F.normalize(torch.randn(1024), dim=0), "value": torch.randn(1024), "word": "lead",
        "carrier_text": METAL, "token_indices": [11, 12, 13], "word_delta": torch.randn(1, 3, 1024),
        "phonetic_text": "led",
    }], "current_step": 1, "gate_threshold": torch.tensor(4.0)}, path)
    f5 = CharBackbone(dim=512)
    mem = HopfieldMemory(MemoryConfig(), 512, f5.embedding_space, text_encoder=f5.encode_text)
    mem.load(path)
    e = mem.entries[0]
    assert e.key.shape == (512,) and e.word_delta is None and e.value.abs().sum() == 0
    assert "legacy:d1024" in e.space_data


# ─────────────────────────────────────────────────────────────────────────────
# F5-TTS span-anchored injection
# ─────────────────────────────────────────────────────────────────────────────

def test_f5_span_locator_handles_prompt_prefix_chunking_and_repeats():
    text = "the lead pipe and the lead role"
    target = [ord(c) for c in text]
    spans = [(4, 8), (22, 26)]  # both "lead"
    ref = [ord(c) + 1 for c in "reference prompt. "]
    # whole text after the prompt (+1 vocab shift)
    row = ref + [t + 1 for t in target]
    assert F5TTSBackbone._locate_delta_spans(row, target, spans) == [(4, 8, len(ref) + 4), (22, 26, len(ref) + 22)]
    # chunked generation: only the second half of the text is in this forward pass
    chunk_start = text.index("and")
    row2 = ref + [t + 1 for t in target[chunk_start:]]
    assert F5TTSBackbone._locate_delta_spans(row2, target, spans) == [(22, 26, len(ref) + 22 - chunk_start)]


def test_f5_active_delta_spans():
    d = torch.zeros(1, 10, 4)
    d[0, 2:4] = 1.0
    d[0, 7] = 1.0
    assert F5TTSBackbone._active_delta_spans(d) == [(2, 4), (7, 8)]


# ─────────────────────────────────────────────────────────────────────────────
# Spelling store: per-occurrence respelling
# ─────────────────────────────────────────────────────────────────────────────

def test_spelling_store_resolves_each_occurrence_independently():
    _encoder_or_skip()
    from flowedit.memory.s3_storage import S3SpellingStore
    store = S3SpellingStore.__new__(S3SpellingStore)
    from flowedit.alignment.homograph_resolver import HomographContextResolver
    store.resolver = HomographContextResolver()
    store._dictionary = {"lead": {"word": "lead", "default_spell": "led", "senses": [
        {"sense_id": "a", "spell_as": "led", "carrier_text": METAL},
        {"sense_id": "b", "spell_as": "leed", "carrier_text": LEADER},
    ]}}
    out, applied = store.apply_corrections_to_text(SENTENCE)
    assert out == "The old paint contains led, and Priya is the leed researcher studying it."
    assert [a["spell_as"] for a in applied] == ["led", "leed"]


# ─────────────────────────────────────────────────────────────────────────────
# Fallback & no-hardcoding guards
# ─────────────────────────────────────────────────────────────────────────────

def test_encoder_unavailable_falls_back_to_backbone_context_keys():
    torch.manual_seed(0)
    d = 64
    mem = HopfieldMemory(MemoryConfig(), embedding_dim=d, sense_encoder=None)
    text = "aaaa lead bbbb cccc dddd lead eeee"
    emb = torch.randn(1, len(text), d)
    first, second = [m.span() for m in re.finditer(r"\blead\b", text)]
    for (cs, ce), pron in ((first, "led"), (second, "leed")):
        key = mem.compute_context_key(emb, list(range(cs, ce)))
        mem.write(key, torch.zeros(d), word="lead", carrier_text=text, phonetic_text=pron, char_span=(cs, ce))
    res = mem.retrieve(emb, text=text)
    assert [(s.char_start, s.phonetic_text) for s in res.matched_spans] == [(first[0], "led"), (second[0], "leed")]


def test_no_builtin_homograph_tables_in_package():
    root = Path(__file__).resolve().parents[1] / "flowedit"
    banned = re.compile(r"HOMOGRAPH_SENSE_MAP|KNOWN_HOMOGRAPHS|TEMPORAL_PAST|INFINITIVE_MODALS")
    offenders = [str(p) for p in root.rglob("*.py") if banned.search(p.read_text(encoding="utf-8"))]
    assert offenders == []
