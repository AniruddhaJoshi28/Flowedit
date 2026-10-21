"""
Shadow-Mode Comparison: Baseline vs Context-Robust Hopfield Retrieval.

Runs every test sentence through all three retrieval modes (baseline, shadow, active)
with TWO configurations:
    1. WITH sense-encoder prototypes (the default production path)
    2. WITHOUT sense-encoder prototypes (exercises the contextual rerank path)

Records per-occurrence:
    - Selected memory entry / phonetic
    - Attention distribution (base vs new)
    - Group mass preservation
    - Context similarity scores
    - Whether the rerank changed the decision

Produces a structured JSON report + readable terminal summary.

Usage:
    python -m pytest tests/test_shadow_mode_comparison.py -v -s
"""

import copy
import dataclasses
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest
import torch
import torch.nn.functional as F

from flowedit.config import MemoryConfig
from flowedit.memory.hopfield_memory import HopfieldMemory as _HopfieldMemory
from flowedit.memory.sense_encoder import get_sense_encoder
from flowedit.refiner.hopfield_refiner import HopfieldRefiner


def HopfieldMemory(*args, **kwargs):
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


# ─── Homograph definitions ──────────────────────────────────────────────────

HOMOGRAPHS = {
    "lead": {
        "senses": {
            "led": {
                "carriers": [
                    "The pipe contains lead.",
                    "The old industrial pipe contains lead.",
                    "This toy is made of lead.",
                ],
                "delta_scale": 1.0,
            },
            "leed": {
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
            "red": {
                "carriers": [
                    "Yesterday I read the entire book.",
                    "She already read the report.",
                    "He read that paper last week.",
                ],
                "delta_scale": 3.0,
            },
            "reed": {
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
            "wind": {
                "carriers": [
                    "The wind is blowing strongly.",
                    "A cold wind came from the north.",
                    "The wind howled through the night.",
                ],
                "delta_scale": 5.0,
            },
            "wynd": {
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

TEST_SENTENCES = [
    ("The pipe in the basement contains lead.", {"lead": "led"}),
    ("She will lead the entire research group.", {"lead": "leed"}),
    ("The old plumbing has dangerous lead inside.", {"lead": "led"}),
    ("They asked her to lead the discussion today.", {"lead": "leed"}),
    ("Yesterday she read the entire manuscript.", {"read": "red"}),
    ("I read scientific papers every single morning.", {"read": "reed"}),
    ("He already read the instructions carefully.", {"read": "red"}),
    ("They always read the news before breakfast.", {"read": "reed"}),
    ("The strong wind battered the coastal houses.", {"wind": "wind"}),
    ("She had to wind the vintage music box.", {"wind": "wynd"}),
    ("A powerful wind swept across the open plain.", {"wind": "wind"}),
    ("He helped wind the fishing line onto the reel.", {"wind": "wynd"}),
    ("She will lead the team into the wind.", {"lead": "leed", "wind": "wind"}),
    ("Yesterday I read about the lead contamination.", {"read": "red", "lead": "led"}),
]


def _create_delta(dim, n_chars, scale):
    gen = torch.Generator().manual_seed(int(scale * 1000))
    return torch.randn(1, n_chars, dim, generator=gen) * scale * 0.01


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


def _populate_memory(bb, config):
    mem = HopfieldMemory(config, embedding_dim=bb.embedding_dim, embedding_space=bb.embedding_space)
    for word, data in HOMOGRAPHS.items():
        for phonetic, sense in data["senses"].items():
            for carrier in sense["carriers"]:
                spans = mem._find_word_char_spans(carrier, word)
                n_chars = spans[0][1] - spans[0][0]
                delta = _create_delta(bb.embedding_dim, n_chars, sense["delta_scale"])
                _write(mem, bb, word, carrier, phonetic=phonetic, word_delta=delta)
    return mem


def _clone_memory(mem_base, bb, mode, strip_prototypes=False):
    """Clone memory with a new mode. Optionally strip prototypes AND the sense encoder
    to force the backbone-key + contextual-rerank path."""
    cfg = dataclasses.replace(mem_base.config, retrieval_mode=mode)
    mem = HopfieldMemory(cfg, embedding_dim=mem_base.d, embedding_space=mem_base.space)
    mem.entries = copy.deepcopy(mem_base.entries)
    mem._text_encoder = bb.encode_text
    mem._tokenizer = mem_base._tokenizer
    if strip_prototypes:
        for e in mem.entries:
            e.sense_prototype = None
        # Disable the sense encoder so occ_vec is None during retrieval.
        # This forces the key-based path and lets _contextual_rerank fire.
        mem._sense_encoder_spec = None
    return mem


def _run_comparison(bb, mem_base, strip_prototypes=False):
    """Run all test sentences through all three modes, collecting full diagnostics."""
    results = []
    for mode in ["baseline", "shadow", "active"]:
        mem = _clone_memory(mem_base, bb, mode, strip_prototypes=strip_prototypes)
        for text, expected in TEST_SENTENCES:
            mem.last_retrieval_diagnostics.clear()
            emb = bb.encode_text(text)
            res = mem.retrieve(emb, text=text)

            for s in res.matched_spans:
                word = s.word.lower()
                want = expected.get(word)
                if want is None:
                    continue

                # Find matching diagnostic
                diag = next(
                    (d for d in mem.last_retrieval_diagnostics
                     if d.get("word", "").lower() == word
                     and d.get("char_start") == s.char_start),
                    {}
                )

                results.append({
                    "mode": mode,
                    "text": text,
                    "word": word,
                    "expected": want,
                    "got": s.phonetic_text,
                    "correct": s.phonetic_text == want,
                    "score": s.score,
                    "gate_value": s.gate_value,
                    "sense_similarity": s.sense_similarity,
                    "strip_prototypes": strip_prototypes,
                    # Rerank diagnostics (only present when rerank fires)
                    "rerank_applied": diag.get("applied", False),
                    "ctx_sims": diag.get("ctx_sims"),
                    "orig_sims": diag.get("orig_sims"),
                    "new_scores": diag.get("new_scores"),
                    "attn_base": diag.get("attn_base"),
                    "attn_new": diag.get("attn_new"),
                    "group_mass": diag.get("group_mass"),
                    "max_other_attn_change": diag.get("max_other_attn_change"),
                    "baseline_choice": diag.get("baseline_choice"),
                    "new_choice": diag.get("new_choice"),
                })
    return results


# ─── Tests ───────────────────────────────────────────────────────────────────

class TestShadowModeComparison:
    """Full shadow-mode comparison with detailed diagnostics."""

    @pytest.fixture(scope="class")
    def base_state(self):
        _encoder_or_skip()
        bb = CharBackbone()
        mem = _populate_memory(bb, MemoryConfig())
        return bb, mem

    def test_with_prototypes(self, base_state):
        """Path 1: sense-encoder prototypes handle disambiguation (production path)."""
        bb, mem_base = base_state
        results = _run_comparison(bb, mem_base, strip_prototypes=False)
        self._print_report(results, "WITH SENSE-ENCODER PROTOTYPES")

        # Core invariants
        baseline = [r for r in results if r["mode"] == "baseline"]
        shadow = [r for r in results if r["mode"] == "shadow"]
        active = [r for r in results if r["mode"] == "active"]

        # Shadow must match baseline exactly
        for b, s in zip(baseline, shadow):
            assert b["got"] == s["got"], f"Shadow != baseline: {b['text'][:40]}"
            assert b["gate_value"] == s["gate_value"]

        # Active must be >= baseline
        base_score = sum(1 for r in baseline if r["correct"])
        active_score = sum(1 for r in active if r["correct"])
        assert active_score >= base_score

    def test_without_prototypes(self, base_state):
        """Path 2: prototypes stripped — contextual rerank decides (the contribution)."""
        bb, mem_base = base_state
        results = _run_comparison(bb, mem_base, strip_prototypes=True)
        self._print_report(results, "WITHOUT PROTOTYPES (CONTEXTUAL RERANK PATH)")

        baseline = [r for r in results if r["mode"] == "baseline"]
        shadow = [r for r in results if r["mode"] == "shadow"]
        active = [r for r in results if r["mode"] == "active"]

        # Shadow must match baseline (rerank is logged but not applied)
        for b, s in zip(baseline, shadow):
            assert b["got"] == s["got"], f"Shadow != baseline: {b['text'][:40]}"

        # Verify rerank diagnostics are logged in shadow/active
        shadow_applied = [r for r in shadow if r["rerank_applied"]]
        active_applied = [r for r in active if r["rerank_applied"]]

        # Verify attention mass preservation where rerank fired
        for r in active_applied:
            if r["max_other_attn_change"] is not None:
                assert abs(r["max_other_attn_change"]) < 1e-6, (
                    f"Unrelated group attention changed: {r['max_other_attn_change']}"
                )
            if r["attn_base"] is not None and r["attn_new"] is not None:
                assert abs(sum(r["attn_base"]) - sum(r["attn_new"])) < 1e-5, (
                    f"Group mass not preserved"
                )

        base_score = sum(1 for r in baseline if r["correct"])
        active_score = sum(1 for r in active if r["correct"])
        assert active_score >= base_score

    def test_delta_vectors_identical_across_all_modes_and_paths(self, base_state):
        """Correction vectors must be 100% unchanged in all configurations."""
        bb, mem_base = base_state
        original_deltas = {
            (e.word, e.phonetic_text): e.word_delta.clone()
            for e in mem_base.entries if e.word_delta is not None
        }

        for strip in [False, True]:
            for mode in ["baseline", "shadow", "active"]:
                mem = _clone_memory(mem_base, bb, mode, strip_prototypes=strip)
                for text, _ in TEST_SENTENCES:
                    mem.retrieve(bb.encode_text(text), text=text)
                for e in mem.entries:
                    if e.word_delta is not None:
                        orig = original_deltas[(e.word, e.phonetic_text)]
                        assert torch.equal(e.word_delta, orig), (
                            f"delta changed: {e.word}/{e.phonetic_text} mode={mode} strip={strip}"
                        )

    def test_gate_values_identical_across_modes(self, base_state):
        """Gate values must be identical across all three modes."""
        bb, mem_base = base_state
        for strip in [False, True]:
            gate_data = {}
            for mode in ["baseline", "shadow", "active"]:
                mem = _clone_memory(mem_base, bb, mode, strip_prototypes=strip)
                for text, expected in TEST_SENTENCES:
                    res = mem.retrieve(bb.encode_text(text), text=text)
                    for s in res.matched_spans:
                        key = (text, s.word.lower(), s.char_start)
                        if key not in gate_data:
                            gate_data[key] = {}
                        gate_data[key][mode] = s.gate_value

            for key, gates in gate_data.items():
                if "baseline" in gates and "shadow" in gates:
                    assert gates["baseline"] == gates["shadow"], (
                        f"Gate mismatch baseline/shadow: {key}"
                    )
                if "baseline" in gates and "active" in gates:
                    assert gates["baseline"] == gates["active"], (
                        f"Gate mismatch baseline/active: {key}"
                    )

    def test_save_full_report(self, base_state):
        """Save the complete comparison report as JSON for analysis."""
        bb, mem_base = base_state
        report = {
            "with_prototypes": _run_comparison(bb, mem_base, strip_prototypes=False),
            "without_prototypes": _run_comparison(bb, mem_base, strip_prototypes=True),
        }

        # Summary statistics
        for path_name, results in report.items():
            for mode in ["baseline", "shadow", "active"]:
                mode_results = [r for r in results if r["mode"] == mode]
                correct = sum(1 for r in mode_results if r["correct"])
                total = len(mode_results)
                print(f"  {path_name}/{mode}: {correct}/{total}")

        # Save to reports directory
        report_dir = Path(__file__).parent.parent / "reports"
        report_dir.mkdir(exist_ok=True)
        report_path = report_dir / "shadow_mode_comparison.json"

        # Convert tensors for JSON serialisation
        def _clean(obj):
            if isinstance(obj, torch.Tensor):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: _clean(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_clean(v) for v in obj]
            return obj

        with open(report_path, "w") as f:
            json.dump(_clean(report), f, indent=2, default=str)
        print(f"\nFull report saved to: {report_path}")

    @staticmethod
    def _print_report(results, title):
        """Terminal-friendly comparison table."""
        print(f"\n{'=' * 110}")
        print(f"  {title}")
        print(f"{'=' * 110}")
        print(f"{'Mode':<10} {'Word':<8} {'Want':<8} {'Got':<8} {'OK':<4} "
              f"{'Rerank':<8} {'CtxSim':<20} {'Sentence':<45}")
        print("-" * 110)

        for r in results:
            mark = "Y" if r["correct"] else "N"
            rerank = "Yes" if r["rerank_applied"] else "-"
            ctx = ""
            if r["ctx_sims"]:
                ctx = " ".join(f"{v:.3f}" for v in r["ctx_sims"])
            text_short = r["text"][:44]
            print(f"{r['mode']:<10} {r['word']:<8} {r['expected']:<8} {r['got']:<8} {mark:<4} "
                  f"{rerank:<8} {ctx:<20} {text_short}")

        print("=" * 110)
        for mode in ["baseline", "shadow", "active"]:
            mode_r = [r for r in results if r["mode"] == mode]
            c = sum(1 for r in mode_r if r["correct"])
            t = len(mode_r)
            reranked = sum(1 for r in mode_r if r["rerank_applied"])
            print(f"  {mode}: {c}/{t} correct, {reranked} reranked")
        print("=" * 110)
