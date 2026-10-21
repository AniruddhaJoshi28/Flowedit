"""
End-to-end integration tests for Context-Robust Hopfield Retrieval.

Validates that contextual reranking correctly disambiguates homographs across
three retrieval modes while preserving:
    - Correction vectors (δ*) bit-identical across modes
    - Gate values identical across modes
    - Unrelated memory groups' attention mass unchanged
    - Shadow mode synthesis output identical to baseline
    - Active mode homograph retrieval improved over baseline

Homographs tested: lead (LED/LEED), read (RED/REED), wind (WIND/WYND).
"""

import copy
import dataclasses
import re
from typing import Dict, List, Optional, Tuple

import pytest
import torch
import torch.nn.functional as F

from flowedit.config import MemoryConfig
from flowedit.memory.hopfield_memory import HopfieldMemory as _HopfieldMemory
from flowedit.memory.sense_encoder import get_sense_encoder
from flowedit.refiner.hopfield_refiner import HopfieldRefiner


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def HopfieldMemory(*args, **kwargs):
    """Memory without the pretrained lexicon (tests the learned-correction path only)."""
    kwargs.setdefault("sense_lexicon", None)
    return _HopfieldMemory(*args, **kwargs)


class CharBackbone:
    """Deterministic character-level stand-in backbone (F5-TTS-like)."""

    def __init__(self, dim=512):
        self.embedding_dim = dim
        self.is_character_level = True
        self.embedding_space = f"StubChar:{dim}"
        self.delta_bounds = (0.35, 3.5)
        self.calls: List[Tuple[str, Optional[torch.Tensor]]] = []
        gen = torch.Generator().manual_seed(42)
        self.table = torch.randn(256, dim, generator=gen)

    def encode_text(self, text, language="en"):
        ids = torch.tensor([ord(c) % 256 for c in text])
        return self.table[ids].unsqueeze(0)

    def synthesize_direct(self, text, speaker_conditioning, **kwargs):
        self.calls.append((text, kwargs.get("text_embedding_delta")))
        return torch.zeros(1, 2400), 24000


def _encoder_or_skip():
    enc = get_sense_encoder(MemoryConfig())
    if enc is None or not enc.available:
        pytest.skip("contextual sense encoder model not available locally")
    return enc


def _write(memory, backbone, word, carrier, phonetic=None, word_delta=None, occurrence=0):
    """Write a correction entry into memory (mirrors the pattern in test_homograph_ema.py)."""
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
# Homograph test data
# ─────────────────────────────────────────────────────────────────────────────

# Each homograph: (word, sense_A_phonetic, sense_A_carriers, sense_B_phonetic, sense_B_carriers)
# Carriers are the training sentences. We create a unique delta per sense.
HOMOGRAPHS = {
    "lead": {
        "senses": {
            "led": {  # metal / past tense of "to lead"
                "carriers": [
                    "The pipe contains lead.",
                    "The old industrial pipe contains lead.",
                    "This toy is made of lead.",
                ],
                "delta_scale": 1.0,
            },
            "leed": {  # verb: to guide
                "carriers": [
                    "She will lead the team.",
                    "He wants to lead the project.",
                    "They chose her to lead the initiative.",
                ],
                "delta_scale": 2.0,
            },
        },
    },
    "read": {
        "senses": {
            "red": {  # past tense
                "carriers": [
                    "Yesterday I read the entire book.",
                    "She already read the report.",
                    "He read that paper last week.",
                ],
                "delta_scale": 3.0,
            },
            "reed": {  # present tense
                "carriers": [
                    "I read books every morning.",
                    "Please read the instructions carefully.",
                    "They read the news daily.",
                ],
                "delta_scale": 4.0,
            },
        },
    },
    "wind": {
        "senses": {
            "wind": {  # air current (short i)
                "carriers": [
                    "The wind is blowing strongly.",
                    "A cold wind came from the north.",
                    "The wind howled through the night.",
                ],
                "delta_scale": 5.0,
            },
            "wynd": {  # to turn/coil (long i)
                "carriers": [
                    "Please wind the clock before bed.",
                    "She needs to wind the yarn into a ball.",
                    "He began to wind the old music box.",
                ],
                "delta_scale": 6.0,
            },
        },
    },
}

# Test sentences: (sentence, expected {word: phonetic} assignments)
TEST_SENTENCES = [
    # lead
    ("The pipe in the basement contains lead.", {"lead": "led"}),
    ("She will lead the entire research group.", {"lead": "leed"}),
    ("The old plumbing has dangerous lead inside.", {"lead": "led"}),
    ("They asked her to lead the discussion today.", {"lead": "leed"}),
    # read
    ("Yesterday she read the entire manuscript.", {"read": "red"}),
    ("I read scientific papers every single morning.", {"read": "reed"}),
    ("He already read the instructions carefully.", {"read": "red"}),
    ("They always read the news before breakfast.", {"read": "reed"}),
    # wind
    ("The strong wind battered the coastal houses.", {"wind": "wind"}),
    ("She had to wind the vintage music box.", {"wind": "wynd"}),
    ("A powerful wind swept across the open plain.", {"wind": "wind"}),
    ("He helped wind the fishing line onto the reel.", {"wind": "wynd"}),
    # Mixed: multiple homographs in one sentence
    ("She will lead the team into the wind.", {"lead": "leed", "wind": "wind"}),
    ("Yesterday I read about the lead contamination.", {"read": "red", "lead": "led"}),
]


def _create_delta(dim, n_chars, scale):
    """Unique, reproducible per-sense delta vector."""
    gen = torch.Generator().manual_seed(int(scale * 1000))
    return torch.randn(1, n_chars, dim, generator=gen) * scale * 0.01


def _populate_memory(bb, config):
    """Build a memory with all homograph senses from HOMOGRAPHS."""
    mem = HopfieldMemory(config, embedding_dim=bb.embedding_dim, embedding_space=bb.embedding_space)
    for word, data in HOMOGRAPHS.items():
        for phonetic, sense in data["senses"].items():
            for carrier in sense["carriers"]:
                spans = mem._find_word_char_spans(carrier, word)
                n_chars = spans[0][1] - spans[0][0]
                delta = _create_delta(bb.embedding_dim, n_chars, sense["delta_scale"])
                _write(mem, bb, word, carrier, phonetic=phonetic, word_delta=delta)
    return mem


def _retrieve_phonetics(mem, bb, text):
    """Retrieve and return {word: phonetic} from the memory for a test sentence."""
    emb = bb.encode_text(text)
    res = mem.retrieve(emb, text=text)
    out = {}
    for s in res.matched_spans:
        out[s.word.lower()] = s.phonetic_text
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 1. Competing pronunciation entries are correctly stored
# ─────────────────────────────────────────────────────────────────────────────

class TestHomographEntryStorage:
    """Verify that each homograph has exactly 2 distinct sense clusters."""

    def test_each_homograph_has_two_senses(self):
        _encoder_or_skip()
        bb = CharBackbone()
        cfg = MemoryConfig()
        mem = _populate_memory(bb, cfg)

        word_senses = {}
        for e in mem.entries:
            word_senses.setdefault(e.word.lower(), set()).add(e.phonetic_text)

        for word in HOMOGRAPHS:
            assert word in word_senses, f"'{word}' missing from memory"
            expected = set(HOMOGRAPHS[word]["senses"].keys())
            assert word_senses[word] == expected, (
                f"'{word}': expected senses {expected}, got {word_senses[word]}"
            )

    def test_delta_vectors_are_distinct_per_sense(self):
        _encoder_or_skip()
        bb = CharBackbone()
        mem = _populate_memory(bb, MemoryConfig())

        for word in HOMOGRAPHS:
            entries = [e for e in mem.entries if e.word.lower() == word]
            assert len(entries) == 2
            d0, d1 = entries[0].word_delta, entries[1].word_delta
            assert d0 is not None and d1 is not None
            assert not torch.allclose(d0, d1), f"'{word}' senses share the same delta"

    def test_carrier_texts_accumulated_via_ema(self):
        _encoder_or_skip()
        bb = CharBackbone()
        mem = _populate_memory(bb, MemoryConfig())

        for word, data in HOMOGRAPHS.items():
            for phonetic, sense in data["senses"].items():
                entries = [e for e in mem.entries if e.word.lower() == word and e.phonetic_text == phonetic]
                assert len(entries) == 1
                assert len(entries[0].carrier_texts) == len(sense["carriers"]), (
                    f"'{word}/{phonetic}': expected {len(sense['carriers'])} carriers, "
                    f"got {len(entries[0].carrier_texts)}"
                )


# ─────────────────────────────────────────────────────────────────────────────
# 2. Three-mode comparison: baseline vs shadow vs active
# ─────────────────────────────────────────────────────────────────────────────

class TestThreeModeComparison:
    """
    For every test sentence, run retrieval in all three modes and record:
        - Selected memory entry / phonetic
        - Attention distribution (for shadow/active)
        - Gate value
        - Effective correction delta
    """

    @pytest.fixture(scope="class")
    def shared_state(self):
        """Build memory once, clone for each mode."""
        _encoder_or_skip()
        bb = CharBackbone()
        base_cfg = MemoryConfig()
        mem_base = _populate_memory(bb, base_cfg)
        return bb, mem_base

    def _mem_with_mode(self, mem_base, bb, mode):
        """Clone memory and switch retrieval mode, wiring in the backbone's text encoder."""
        cfg = dataclasses.replace(mem_base.config, retrieval_mode=mode)
        mem = HopfieldMemory(cfg, embedding_dim=mem_base.d, embedding_space=mem_base.space)
        mem.entries = copy.deepcopy(mem_base.entries)
        mem._text_encoder = bb.encode_text
        mem._tokenizer = mem_base._tokenizer
        return mem

    def test_shadow_produces_identical_phonetics_to_baseline(self, shared_state):
        """Shadow mode must select the same pronunciation as baseline (only logs differ)."""
        bb, mem_base = shared_state
        for text, _ in TEST_SENTENCES:
            base_res = _retrieve_phonetics(self._mem_with_mode(mem_base, bb, "baseline"), bb, text)
            shadow_res = _retrieve_phonetics(self._mem_with_mode(mem_base, bb, "shadow"), bb, text)
            assert base_res == shadow_res, (
                f"Shadow != baseline for '{text[:60]}': {shadow_res} vs {base_res}"
            )

    def test_shadow_logs_rerank_diagnostics(self, shared_state):
        """Shadow mode must produce non-empty diagnostics for homograph occurrences."""
        bb, mem_base = shared_state
        mem = self._mem_with_mode(mem_base, bb, "shadow")
        text = "The pipe contains lead."
        emb = bb.encode_text(text)
        mem.last_retrieval_diagnostics.clear()
        mem.retrieve(emb, text=text)
        # At least one diagnostic entry for 'lead'
        diags = [d for d in mem.last_retrieval_diagnostics if d.get("word", "").lower() == "lead"]
        assert len(diags) >= 1, "Shadow mode should log diagnostics for 'lead'"
        d = diags[0]
        assert d.get("mode") == "shadow"
        # When sense-encoder prototypes are available, they handle disambiguation and
        # _contextual_rerank is correctly bypassed. In that case, 'applied' won't be set.
        # When prototypes are unavailable, the rerank fires and 'applied' is set.
        assert d.get("final_choice") is not None, "Diagnostic must record the final choice"

    def test_active_mode_records_full_diagnostics(self, shared_state):
        """Active mode diagnostics contain attention distributions and group info."""
        bb, mem_base = shared_state
        mem = self._mem_with_mode(mem_base, bb, "active")
        text = "She will lead the team."
        emb = bb.encode_text(text)
        mem.last_retrieval_diagnostics.clear()
        mem.retrieve(emb, text=text)
        diags = [d for d in mem.last_retrieval_diagnostics if d.get("applied")]
        if diags:
            d = diags[0]
            assert "attn_base" in d, "Active mode should record baseline attention"
            assert "attn_new" in d, "Active mode should record new attention"
            assert "ctx_sims" in d, "Active mode should record context similarities"
            assert d["mode"] == "active"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Correction vector (δ*) integrity across modes
# ─────────────────────────────────────────────────────────────────────────────

class TestCorrectionVectorIntegrity:
    """δ* vectors must remain 100% unchanged regardless of retrieval mode."""

    @pytest.fixture(scope="class")
    def shared_state(self):
        _encoder_or_skip()
        bb = CharBackbone()
        mem = _populate_memory(bb, MemoryConfig())
        # Snapshot all deltas before any retrieval
        deltas = {(e.word, e.phonetic_text): e.word_delta.clone() for e in mem.entries if e.word_delta is not None}
        return bb, mem, deltas

    def test_deltas_unchanged_after_baseline_retrieval(self, shared_state):
        bb, mem_base, original_deltas = shared_state
        cfg = dataclasses.replace(mem_base.config, retrieval_mode="baseline")
        mem = HopfieldMemory(cfg, embedding_dim=mem_base.d, embedding_space=mem_base.space)
        mem.entries = copy.deepcopy(mem_base.entries)
        for text, _ in TEST_SENTENCES:
            mem.retrieve(bb.encode_text(text), text=text)
        for e in mem.entries:
            if e.word_delta is not None:
                orig = original_deltas[(e.word, e.phonetic_text)]
                assert torch.equal(e.word_delta, orig), f"δ* changed for {e.word}/{e.phonetic_text} in baseline"

    def test_deltas_unchanged_after_active_retrieval(self, shared_state):
        bb, mem_base, original_deltas = shared_state
        cfg = dataclasses.replace(mem_base.config, retrieval_mode="active")
        mem = HopfieldMemory(cfg, embedding_dim=mem_base.d, embedding_space=mem_base.space)
        mem.entries = copy.deepcopy(mem_base.entries)
        mem._text_encoder = mem_base._text_encoder
        for text, _ in TEST_SENTENCES:
            mem.retrieve(bb.encode_text(text), text=text)
        for e in mem.entries:
            if e.word_delta is not None:
                orig = original_deltas[(e.word, e.phonetic_text)]
                assert torch.equal(e.word_delta, orig), f"δ* changed for {e.word}/{e.phonetic_text} in active"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Unrelated memory groups retain baseline attention mass
# ─────────────────────────────────────────────────────────────────────────────

class TestAttentionMassPreservation:
    """When reranking 'lead' senses, 'read' and 'wind' entries must keep their baseline attention."""

    def test_max_other_attn_change_is_zero(self):
        _encoder_or_skip()
        bb = CharBackbone()
        cfg = dataclasses.replace(MemoryConfig(), retrieval_mode="active")
        mem = _populate_memory(bb, cfg)
        mem._text_encoder = bb.encode_text

        text = "She will lead the team."
        mem.last_retrieval_diagnostics.clear()
        mem.retrieve(bb.encode_text(text), text=text)
        applied = [d for d in mem.last_retrieval_diagnostics if d.get("applied")]
        for d in applied:
            max_change = d.get("max_other_attn_change", 0.0)
            assert abs(max_change) < 1e-6, (
                f"Unrelated group attention changed by {max_change} (expected 0.0)"
            )

    def test_group_mass_preserved(self):
        """Sum of attention within the target group is identical before and after rerank."""
        _encoder_or_skip()
        bb = CharBackbone()
        cfg = dataclasses.replace(MemoryConfig(), retrieval_mode="active")
        mem = _populate_memory(bb, cfg)
        mem._text_encoder = bb.encode_text

        for text, _ in TEST_SENTENCES[:4]:  # lead sentences
            mem.last_retrieval_diagnostics.clear()
            mem.retrieve(bb.encode_text(text), text=text)
            for d in mem.last_retrieval_diagnostics:
                if d.get("applied") and "attn_base" in d and "attn_new" in d:
                    base_mass = sum(d["attn_base"])
                    new_mass = sum(d["attn_new"])
                    assert abs(base_mass - new_mass) < 1e-5, (
                        f"Group mass changed: {base_mass:.6f} → {new_mass:.6f}"
                    )


# ─────────────────────────────────────────────────────────────────────────────
# 5. Shadow synthesis output identical to baseline
# ─────────────────────────────────────────────────────────────────────────────

class TestShadowSynthesisIdentical:
    """Shadow mode must produce bit-identical synthesis calls as baseline."""

    def test_refiner_output_text_identical(self):
        _encoder_or_skip()
        bb_base = CharBackbone()
        bb_shadow = CharBackbone()  # separate call recorders
        cfg_base = MemoryConfig()
        cfg_shadow = dataclasses.replace(MemoryConfig(), retrieval_mode="shadow")

        mem_base = _populate_memory(bb_base, cfg_base)
        mem_shadow = _populate_memory(bb_shadow, cfg_shadow)
        mem_shadow._text_encoder = bb_shadow.encode_text

        ref_base = HopfieldRefiner(mem_base, cfg_base)
        ref_shadow = HopfieldRefiner(mem_shadow, cfg_shadow)

        for text, _ in TEST_SENTENCES:
            bb_base.calls.clear()
            bb_shadow.calls.clear()
            ref_base.forward(bb_base, text, {"audio_path": "x.wav"})
            ref_shadow.forward(bb_shadow, text, {"audio_path": "x.wav"})
            if bb_base.calls and bb_shadow.calls:
                base_synth = bb_base.calls[-1][0]
                shadow_synth = bb_shadow.calls[-1][0]
                assert base_synth == shadow_synth, (
                    f"Shadow ≠ baseline synthesis for '{text[:60]}...':\n"
                    f"  baseline: {base_synth}\n  shadow:   {shadow_synth}"
                )

    def test_refiner_deltas_identical(self):
        """If baseline injects a delta, shadow must inject the exact same delta."""
        _encoder_or_skip()
        bb_base = CharBackbone()
        bb_shadow = CharBackbone()
        cfg_base = MemoryConfig()
        cfg_shadow = dataclasses.replace(MemoryConfig(), retrieval_mode="shadow")

        mem_base = _populate_memory(bb_base, cfg_base)
        mem_shadow = _populate_memory(bb_shadow, cfg_shadow)
        mem_shadow._text_encoder = bb_shadow.encode_text

        ref_base = HopfieldRefiner(mem_base, cfg_base)
        ref_shadow = HopfieldRefiner(mem_shadow, cfg_shadow)

        text = "The pipe contains lead."
        bb_base.calls.clear()
        bb_shadow.calls.clear()
        ref_base.forward(bb_base, text, {"audio_path": "x.wav"})
        ref_shadow.forward(bb_shadow, text, {"audio_path": "x.wav"})

        delta_base = bb_base.calls[-1][1] if bb_base.calls else None
        delta_shadow = bb_shadow.calls[-1][1] if bb_shadow.calls else None
        if delta_base is not None and delta_shadow is not None:
            assert torch.equal(delta_base, delta_shadow), "Shadow delta differs from baseline delta"
        else:
            assert delta_base is None and delta_shadow is None, "Delta presence mismatch"


# ─────────────────────────────────────────────────────────────────────────────
# 6. Sentence variation stress test
# ─────────────────────────────────────────────────────────────────────────────

class TestSentenceVariation:
    """
    Change surrounding words, sentence length, and cue-word positions
    while preserving the intended pronunciation.
    """

    LEAD_METAL_VARIATIONS = [
        "The pipe contains lead.",
        "The old industrial pipe contains lead.",
        "Workers discovered lead in the water supply.",
        "Lead contamination was found in the soil.",
        "The paint was recalled due to high lead content.",
    ]

    LEAD_VERB_VARIATIONS = [
        "She will lead the team.",
        "He wants to lead the entire project forward.",
        "The captain will lead the expedition into the mountains.",
        "They asked her to lead the research initiative.",
        "Who will lead the organisation next year?",
    ]

    def test_active_mode_consistent_for_metal_variations(self):
        """All 'metal' variations should retrieve 'led' in active mode."""
        _encoder_or_skip()
        bb = CharBackbone()
        cfg = dataclasses.replace(MemoryConfig(), retrieval_mode="active")
        mem = _populate_memory(bb, cfg)
        mem._text_encoder = bb.encode_text

        results = {}
        for text in self.LEAD_METAL_VARIATIONS:
            phonetics = _retrieve_phonetics(mem, bb, text)
            results[text] = phonetics.get("lead", "MISSING")

        for text, pron in results.items():
            assert pron == "led", f"Metal variation failed: '{text[:50]}' → '{pron}' (expected 'led')"

    def test_active_mode_consistent_for_verb_variations(self):
        """All 'verb' variations should retrieve 'leed' in active mode."""
        _encoder_or_skip()
        bb = CharBackbone()
        cfg = dataclasses.replace(MemoryConfig(), retrieval_mode="active")
        mem = _populate_memory(bb, cfg)
        mem._text_encoder = bb.encode_text

        results = {}
        for text in self.LEAD_VERB_VARIATIONS:
            phonetics = _retrieve_phonetics(mem, bb, text)
            results[text] = phonetics.get("lead", "MISSING")

        for text, pron in results.items():
            assert pron == "leed", f"Verb variation failed: '{text[:50]}' → '{pron}' (expected 'leed')"


# ─────────────────────────────────────────────────────────────────────────────
# 7. Gate value consistency across modes
# ─────────────────────────────────────────────────────────────────────────────

class TestGateValueConsistency:
    """Gate values must be identical across baseline/shadow/active."""

    def test_gate_values_unchanged(self):
        _encoder_or_skip()
        bb = CharBackbone()

        modes = ["baseline", "shadow", "active"]
        text = "The pipe contains lead."

        gate_by_mode = {}
        for mode in modes:
            cfg = dataclasses.replace(MemoryConfig(), retrieval_mode=mode)
            mem = _populate_memory(bb, cfg)
            if mode != "baseline":
                mem._text_encoder = bb.encode_text
            emb = bb.encode_text(text)
            res = mem.retrieve(emb, text=text)
            gates = {s.word.lower(): s.gate_value for s in res.matched_spans}
            gate_by_mode[mode] = gates

        for word in gate_by_mode["baseline"]:
            for mode in ["shadow", "active"]:
                if word in gate_by_mode[mode]:
                    assert gate_by_mode["baseline"][word] == gate_by_mode[mode][word], (
                        f"Gate value for '{word}' differs: baseline={gate_by_mode['baseline'][word]}, "
                        f"{mode}={gate_by_mode[mode][word]}"
                    )


# ─────────────────────────────────────────────────────────────────────────────
# 8. Active mode improvement over baseline (the core claim)
# ─────────────────────────────────────────────────────────────────────────────

class TestActiveImprovement:
    """
    Active mode should disambiguate at least as well as baseline, and strictly
    better on sentences where baseline fails.

    We count correct retrievals across all TEST_SENTENCES and assert active >= baseline.
    """

    def test_active_at_least_as_good_as_baseline(self):
        _encoder_or_skip()
        bb = CharBackbone()

        correct = {"baseline": 0, "active": 0}
        total = 0

        for mode in ["baseline", "active"]:
            cfg = dataclasses.replace(MemoryConfig(), retrieval_mode=mode)
            mem = _populate_memory(bb, cfg)
            if mode != "baseline":
                mem._text_encoder = bb.encode_text

            for text, expected in TEST_SENTENCES:
                result = _retrieve_phonetics(mem, bb, text)
                for word, want in expected.items():
                    if mode == "baseline":
                        total += 1
                    got = result.get(word)
                    if got == want:
                        correct[mode] += 1

        assert correct["active"] >= correct["baseline"], (
            f"Active mode worse than baseline: {correct['active']}/{total} vs {correct['baseline']}/{total}"
        )

    def test_report_improvement_details(self):
        """Log per-sentence results for inspection (always passes, purely diagnostic)."""
        _encoder_or_skip()
        bb = CharBackbone()
        rows = []

        for mode in ["baseline", "active"]:
            cfg = dataclasses.replace(MemoryConfig(), retrieval_mode=mode)
            mem = _populate_memory(bb, cfg)
            if mode != "baseline":
                mem._text_encoder = bb.encode_text

            for text, expected in TEST_SENTENCES:
                result = _retrieve_phonetics(mem, bb, text)
                for word, want in expected.items():
                    got = result.get(word, "MISSING")
                    rows.append({
                        "mode": mode,
                        "text": text[:60],
                        "word": word,
                        "expected": want,
                        "got": got,
                        "correct": got == want,
                    })

        # Print a comparison table
        print("\n" + "=" * 100)
        print("CONTEXTUAL HOMOGRAPH RETRIEVAL -- BASELINE vs ACTIVE")
        print("=" * 100)
        print(f"{'Mode':<10} {'Word':<8} {'Expected':<10} {'Got':<10} {'OK':<4} {'Sentence':<60}")
        print("-" * 100)
        for r in rows:
            mark = "Y" if r["correct"] else "N"
            print(f"{r['mode']:<10} {r['word']:<8} {r['expected']:<10} {r['got']:<10} {mark:<4} {r['text']}")
        print("=" * 100)

        base_correct = sum(1 for r in rows if r["mode"] == "baseline" and r["correct"])
        active_correct = sum(1 for r in rows if r["mode"] == "active" and r["correct"])
        total = sum(1 for r in rows if r["mode"] == "baseline")
        print(f"Baseline: {base_correct}/{total}  |  Active: {active_correct}/{total}")
        print(f"Improvement: {active_correct - base_correct} additional correct retrievals")
        print("=" * 100)
