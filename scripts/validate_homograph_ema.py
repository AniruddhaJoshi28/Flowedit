#!/usr/bin/env python
"""
Validate EMA sense-cluster homograph disambiguation on XTTS and F5-TTS.

Two modes:

  --mode text   (runs anywhere, no TTS weights): loads the memory file into an XTTS-space and an
                F5-space memory (stub text encoders), runs the full retrieval + refiner path and
                prints, for every occurrence of every stored word, the chosen sense, its
                similarity and the text actually sent to the synthesizer. Also reports latency.

  --mode audio  (TTS server): loads the real XTTS and F5-TTS inference pipelines and writes
                baseline / corrected WAVs for every sentence and backbone plus manifest.csv for
                listening comparison.

Senses can be registered for the run without touching the production memory file:
    --sense "lead=led@This toy is made of lead" --sense "lead=leed@Priya is the lead researcher"

Example:
    python scripts/validate_homograph_ema.py --mode text --memory ../corrections.pt \\
        --sense "lead=leed@Priya is the lead researcher on the project"
    python scripts/validate_homograph_ema.py --mode audio --memory ../corrections.pt --out ./homograph_eval \\
        --sense "lead=leed@Priya is the lead researcher on the project" --speaker michael
"""

import argparse
import csv
import os
import shutil
import sys
import tempfile
import time
from typing import List, Optional, Tuple

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flowedit.config import FlowEditConfig, MemoryConfig  # noqa: E402
from flowedit.memory.hopfield_memory import HopfieldMemory  # noqa: E402
from flowedit.refiner.hopfield_refiner import HopfieldRefiner  # noqa: E402

DEFAULT_SENTENCES = [
    "The old paint contains lead, and Priya is the lead researcher studying it.",
    "The lead researcher found lead in the old pipes.",
    "Our team lead says the water contains lead.",
    "Lead pipes were banned decades ago.",
    "She will lead the team next year.",
    "This product was developed by Samsung team lead by Anurag.",
    "Bispecific antibodies are becoming an important approach in cancer treatment.",
    "The weather is pleasant today.",
]


def parse_sense(spec: str) -> Tuple[str, str, str]:
    """'word=spelling@carrier sentence' -> (word, spelling, carrier)."""
    head, carrier = spec.split("@", 1)
    word, spelling = head.split("=", 1)
    return word.strip(), spelling.strip(), carrier.strip()


def register_sense(memory: HopfieldMemory, encode, word: str, spelling: str, carrier: str) -> str:
    emb = encode(carrier, "en")
    spans = memory._find_word_char_spans(carrier, word)
    if not spans:
        raise ValueError(f"'{word}' not found in '{carrier}'")
    cs, ce = spans[0]
    ts, te = memory._char_span_to_token_span(cs, ce, len(carrier), emb.shape[1])
    key = memory.compute_context_key(emb, list(range(ts, te)))
    _, action = memory.write(
        key=key, value=torch.zeros(memory.d), word=word, carrier_text=carrier,
        token_indices=list(range(ts, te)), phonetic_text=spelling, char_span=(cs, ce),
        metadata={"source": "validation_run"},
    )
    return action


class StubBackbone:
    """Text-only stand-in: deterministic embeddings of the right granularity and dimension."""

    def __init__(self, name: str, dim: int, char_level: bool):
        self.embedding_dim = dim
        self.is_character_level = char_level
        self.embedding_space = f"{name}:{dim}"
        self.delta_bounds = (0.35, 3.5) if char_level else (0.30, 4.0)
        gen = torch.Generator().manual_seed(dim)
        self.table = torch.randn(256, dim, generator=gen)
        self.last = None

    def encode_text(self, text, language="en"):
        ids = torch.tensor([ord(c) % 256 for c in text])
        return self.table[ids].unsqueeze(0)

    def synthesize_direct(self, text, speaker_conditioning, **kwargs):
        delta = kwargs.get("text_embedding_delta")
        self.last = (text, None if delta is None else float(delta.norm()))
        return torch.zeros(1, 240), 24000


def run_text(args) -> int:
    backbones = [StubBackbone("XTTSBackbone", 1024, False), StubBackbone("F5TTSBackbone", 512, True)]
    workdir = tempfile.mkdtemp(prefix="homograph_eval_")
    mem_path = os.path.join(workdir, "corrections.pt")
    if args.memory and os.path.exists(args.memory):
        shutil.copy(args.memory, mem_path)

    for bb in backbones:
        mem = HopfieldMemory(MemoryConfig(), bb.embedding_dim, bb.embedding_space, text_encoder=bb.encode_text)
        mem.load(mem_path)
        for spec in args.sense:
            w, sp, car = parse_sense(spec)
            print(f"  [{bb.embedding_space}] register '{w}'->'{sp}' ({register_sense(mem, bb.encode_text, w, sp, car)})")
        refiner = HopfieldRefiner(mem, MemoryConfig())

        print(f"\n=== {bb.embedding_space}  ({mem.num_entries} sense clusters) ===")
        for e in mem.entries:
            print(f"  - {e.word!r:14} phonetic={e.phonetic_text!r:16} n={e.access_count} local_delta={e.delta_space == mem.space} "
                  f"carriers={e.carrier_texts[:2]}")
        for text in args.sentences:
            t0 = time.perf_counter()
            res = refiner.forward(bb, text, {"audio_path": ""})
            ms = (time.perf_counter() - t0) * 1000
            synth, dnorm = bb.last
            print(f"\n  IN : {text}\n  OUT: {synth}   [{ms:.1f} ms, delta-norm={dnorm}]")
            for s in res.diagnostics.get("matched_spans", []):
                print(f"       '{s['word']}'@{s['char_start']}: sense-sim={s['sense_similarity']:.3f} -> {s['phonetic_text']!r} (delta={s['has_delta']})")
    shutil.rmtree(workdir, ignore_errors=True)
    return 0


def run_audio(args) -> int:
    from flowedit.pipeline.inference import FlowEditInference
    from flowedit.utils.voice_finder import get_hardcoded_voice_path

    os.makedirs(args.out, exist_ok=True)
    mem_path = os.path.join(args.out, "corrections_eval.pt")
    if args.memory and os.path.exists(args.memory):
        shutil.copy(args.memory, mem_path)
    speaker = args.speaker if os.path.isfile(args.speaker) else get_hardcoded_voice_path(args.speaker)

    rows = []
    for btype in args.backbones:
        cfg = FlowEditConfig()
        cfg.backbone.backbone_type = btype
        inf = FlowEditInference(cfg)
        inf.load()
        inf.memory.load(mem_path)
        for spec in args.sense:
            w, sp, car = parse_sense(spec)
            register_sense(inf.memory, inf.backbone.encode_text, w, sp, car)
        cond = inf.backbone.get_speaker_embedding(speaker, language="en")
        for i, text in enumerate(args.sentences):
            base_path = os.path.join(args.out, f"{btype}_{i:02d}_baseline.wav")
            corr_path = os.path.join(args.out, f"{btype}_{i:02d}_corrected.wav")
            t0 = time.perf_counter()
            wav, sr = inf.backbone.synthesize_direct(text=text, speaker_conditioning=cond, language="en", seed=args.seed)
            t_base = time.perf_counter() - t0
            import soundfile as sf
            sf.write(base_path, wav.squeeze().cpu().numpy(), sr)
            t0 = time.perf_counter()
            res = inf.synthesize(text=text, speaker_wav=speaker, output_path=corr_path, seed=args.seed)
            t_corr = time.perf_counter() - t0
            diag = res["diagnostics"]
            rows.append({
                "backbone": btype, "idx": i, "text": text, "synth_text": diag.get("synth_text", text),
                "occurrences": "; ".join(f"{s['word']}@{s['char_start']}={s['phonetic_text']}({s['sense_similarity']:.2f})"
                                         for s in diag.get("matched_spans", [])),
                "baseline_wav": base_path, "corrected_wav": corr_path,
                "baseline_s": round(t_base, 3), "corrected_s": round(t_corr, 3),
            })
            print(f"[{btype}] {text}\n   -> {rows[-1]['synth_text']}  ({t_base:.2f}s baseline / {t_corr:.2f}s corrected)")
        del inf
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with open(os.path.join(args.out, "manifest.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} sentence pairs and manifest.csv to {args.out}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("text", "audio"), default="text")
    ap.add_argument("--memory", default="./corrections.pt", help="memory file to evaluate (copied, never modified)")
    ap.add_argument("--spellings", default=None, help="spelling_dictionary.json to use as the registered-spelling store")
    ap.add_argument("--sense", action="append", default=[], help="extra sense for this run: 'word=spelling@carrier'")
    ap.add_argument("--sentence", dest="sentences", action="append", help="test sentence (repeatable)")
    ap.add_argument("--backbones", nargs="+", default=["xtts", "f5tts"])
    ap.add_argument("--speaker", default="blessing")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="./homograph_eval")
    args = ap.parse_args(argv)
    args.sentences = args.sentences or DEFAULT_SENTENCES
    if args.spellings:
        import json
        from flowedit.memory.s3_storage import s3_spelling_store
        with open(args.spellings, encoding="utf-8") as f:
            s3_spelling_store._dictionary = json.load(f).get("words", {})
        print(f"Registered spellings from {args.spellings}: {sorted(s3_spelling_store._dictionary)}")
    return run_text(args) if args.mode == "text" else run_audio(args)


if __name__ == "__main__":
    sys.exit(main())
