#!/usr/bin/env python
"""
Probe: do frozen F5-TTS representations separate homograph pronunciations from text context?

Representations (genuine F5-TTS checkpoint, all weights frozen, no training of any kind):
  text_embed   output of the DiT text embedding (character embedding + ConvNeXt), 512-d
  L00 … L21    output of each of the 22 DiT transformer blocks, 1024-d
For one occurrence, a representation is the mean over the sequence rows that carry the word's
characters. F5-TTS places character k of (reference text + sentence) at sequence row k; every
extracted span is checked to decode back to the word's characters.

Leakage control (no audio containing a target pronunciation ever enters the model):
  x     fixed Gaussian noise, the same noise row at the same position for every sentence
  cond  the speaker prompt mel (same prompt for all sentences; its transcript is checked to
        contain none of the probed words)
  t     0.0 for the main probe; 0.5 and 0.9 with the same inputs to measure timestep dependence

Retrieval (isolates information already present, no classifier / projection):
  exemplar   sense of the most cosine-similar correction-sentence representation
  prototype  sense whose mean correction representation is most cosine-similar

Protocol: the correction sentences and held-out sentences of scripts/sense_context_benchmark.py.
Held-out sentences are split per (word, sense) into alternating validation / test halves.
Representation and retrieval method are selected on validation only; test is reported once,
next to the context sense model's and pooled keys' choices on the same test sentences.
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flowedit.config import FlowEditConfig  # noqa: E402
from flowedit.backbone import create_backbone  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sense_context_benchmark import DATA  # noqa: E402

TIMES = (0.0, 0.5, 0.9)
KS = (1, 2, 3)
NOISE_SEED = 1234


def split(tests):
    return tests[0::2], tests[1::2]   # validation, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--speaker", required=True)
    ap.add_argument("--context-rows", required=True, help="selection_rows.csv of sense_context_benchmark.py")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.backends.cudnn.deterministic = True

    cfg = FlowEditConfig()
    cfg.backbone.backbone_type = "f5tts"
    bb = create_backbone(cfg.backbone)
    bb.load_model()
    ema = bb.tts_api.ema_model
    dit = ema.transformer
    dev = next(ema.parameters()).device
    dtype = next(ema.parameters()).dtype
    for p in ema.parameters():
        p.requires_grad_(False)
    n_layers = len(dit.transformer_blocks)

    from f5_tts.model.utils import convert_char_to_pinyin, list_str_to_idx
    import soundfile as sf

    # Speaker prompt exactly as the backbone prepares it for synthesis
    spk = bb.get_speaker_embedding(args.speaker, language="en")
    ref_path = spk.get("processed_audio_path") or spk.get("audio_path")
    ref_text = bb._preset_prompt_text(spk.get("audio_path", "") or ref_path, (spk.get("text") or ".").strip())
    if not ref_text.endswith(". "):
        ref_text = ref_text.rstrip() + (" " if ref_text.rstrip().endswith(".") else ". ")
    for w in DATA:
        assert not re.search(rf"\b{w}\b", ref_text, flags=re.IGNORECASE), f"prompt transcript contains '{w}'"
    audio, sr = sf.read(ref_path, dtype="float32", always_2d=True)
    audio = torch.from_numpy(audio.mean(axis=1))[None]
    if sr != 24000:
        import torchaudio
        audio = torchaudio.functional.resample(audio, sr, 24000)
    rms = torch.sqrt(torch.mean(audio ** 2))
    if rms < 0.1:
        audio = audio * 0.1 / rms
    with torch.no_grad():
        ref_mel = ema.mel_spec(audio.to(dev)).permute(0, 2, 1).to(dtype)   # [1, n_ref, mel]
    n_ref = ref_mel.shape[1]
    hop = 256
    gen = torch.Generator().manual_seed(NOISE_SEED)
    noise_bank = torch.randn(8192, ref_mel.shape[-1], generator=gen)

    captured = {}
    hooks = [dit.text_embed.register_forward_hook(lambda m, i, o: captured.__setitem__("text_embed", o.detach()))]
    for li, blk in enumerate(dit.transformer_blocks):
        hooks.append(blk.register_forward_hook(lambda m, i, o, li=li: captured.__setitem__(f"L{li:02d}", o.detach())))
    reps = ["text_embed"] + [f"L{i:02d}" for i in range(n_layers)]

    def represent(sentence, word, t):
        """{rep: unit vector} for the first occurrence of `word` in `sentence` at timestep t."""
        # F5's tokenizer may insert spaces and rewrite some punctuation, so locate the word in the
        # token sequence itself: the sentence's own tokens end the full sequence
        tokens = convert_char_to_pinyin([ref_text + sentence])[0]
        sent_tokens = convert_char_to_pinyin([sentence])[0]
        assert all(len(tok) == 1 for tok in tokens), "expected one character per token"
        joined, sent_joined = "".join(tokens), "".join(sent_tokens)
        assert joined.endswith(sent_joined), "sentence tokens must end the sequence"
        offset = len(joined) - len(sent_joined)
        start = offset + re.search(rf"\b{word}\b", sent_joined, flags=re.IGNORECASE).start()
        span = list(range(start, start + len(word)))
        text = list_str_to_idx([tokens], ema.vocab_char_map).to(dev)
        inv = {v: k for k, v in ema.vocab_char_map.items()}
        decoded = "".join(inv[int(text[0, k])] for k in span)
        assert decoded.lower() == word.lower(), f"span decodes to '{decoded}', not '{word}'"
        duration = n_ref + int(n_ref / len(ref_text.encode("utf-8")) * len(sentence.encode("utf-8")))
        duration = max(duration, text.shape[1] + 1)
        cond = F.pad(ref_mel, (0, 0, 0, duration - n_ref))
        x = noise_bank[:duration].unsqueeze(0).to(dev, dtype)
        captured.clear()
        with torch.no_grad():
            dit(x=x, cond=cond, text=text, time=torch.tensor(t, device=dev, dtype=dtype),
                drop_audio_cond=False, drop_text=False)
        return {r: F.normalize(captured[r][0, span].float().mean(0), dim=0).cpu() for r in reps}

    # Extract every sentence once per timestep
    sentences = set()
    for word, senses in DATA.items():
        for s in senses.values():
            sentences.update((word, x) for x in s["corrections"] + s["tests"])
    feats = {t: {} for t in TIMES}
    for t in TIMES:
        for word, sent in sorted(sentences):
            feats[t][(word, sent)] = represent(sent, word, t)
        print(f"t={t}: extracted {len(feats[t])} sentences", flush=True)

    def predict(t, rep, method, word, k, sent):
        names = list(DATA[word])
        ex = {s: torch.stack([feats[t][(word, c)][rep] for c in DATA[word][s]["corrections"][:k]]) for s in names}
        q = feats[t][(word, sent)][rep]
        if method == "exemplar":
            score = {s: float((ex[s] @ q).max()) for s in names}
        else:
            score = {s: float(F.normalize(ex[s].mean(0), dim=0) @ q) for s in names}
        return max(score, key=score.get)

    rows = []
    for t in TIMES:
        for rep in reps:
            for method in ("exemplar", "prototype"):
                for k in KS:
                    for word, senses in DATA.items():
                        for s, d in senses.items():
                            val, test = split(d["tests"])
                            for part, sents in (("val", val), ("test", test)):
                                for sent in sents:
                                    rows.append({"t": t, "rep": rep, "method": method, "k": k, "word": word,
                                                 "split": part, "expected": s, "text": sent,
                                                 "pred": predict(t, rep, method, word, k, sent)})
    with open(os.path.join(args.out, "probe_rows.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def acc(sel):
        return sum(r["pred"] == r["expected"] for r in sel) / max(1, len(sel)), len(sel)

    by = defaultdict(list)
    for r in rows:
        by[(r["t"], r["rep"], r["method"], r["k"], r["split"])].append(r)

    # Selection on validation at t = 0 (mean over k), then one look at test
    val_scores = {(rep, m): np.mean([acc(by[(0.0, rep, m, k, "val")])[0] for k in KS])
                  for rep in reps for m in ("exemplar", "prototype")}
    best_rep, best_m = max(val_scores, key=val_scores.get)

    # Context model and pooled keys on the same test sentences
    test_set = {(w, x) for w, s in DATA.items() for d in s.values() for x in split(d["tests"])[1]}
    val_set = {(w, x) for w, s in DATA.items() for d in s.values() for x in split(d["tests"])[0]}
    ctx = defaultdict(lambda: [0, 0])
    with open(args.context_rows, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if not r["scenario"].startswith("both_k"):
                continue
            key = (r["word"], r["text"])
            part = "test" if key in test_set else "val" if key in val_set else None
            if part is None:
                continue
            for agg in ((r["method"], int(r["scenario"][-1]), part), (r["method"], int(r["scenario"][-1]), part, r["word"])):
                ctx[agg][0] += int(r["applied"] == r["expected"])
                ctx[agg][1] += 1

    summary = {
        "protocol": {"times": TIMES, "ks": KS, "noise_seed": NOISE_SEED, "prompt_text": ref_text,
                     "n_val": len(val_set), "n_test": len(test_set), "layers": n_layers},
        "selected_on_validation": {"rep": best_rep, "method": best_m, "val_mean_acc": val_scores[(best_rep, best_m)]},
        "validation_ranking_top10": sorted(([f"{r}|{m}", round(v, 3)] for (r, m), v in val_scores.items()),
                                           key=lambda x: -x[1])[:10],
        "test": {},
    }
    for k in KS:
        row = {"probe_selected": acc(by[(0.0, best_rep, best_m, k, "test")]),
               "probe_selected_t0.5": acc(by[(0.5, best_rep, best_m, k, "test")]),
               "probe_selected_t0.9": acc(by[(0.9, best_rep, best_m, k, "test")]),
               "probe_text_embed_exemplar": acc(by[(0.0, "text_embed", "exemplar", k, "test")])}
        for m in ("context", "pooled"):
            c, n = ctx[(m, k, "test")]
            row[f"{m}_model"] = (c / n if n else None, n)
        per_word = {}
        for word in DATA:
            sel = [r for r in by[(0.0, best_rep, best_m, k, "test")] if r["word"] == word]
            c, n = ctx[("context", k, "test", word)]
            per_word[word] = {"probe": acc(sel), "context": (c / n if n else None, n)}
        row["per_word"] = per_word
        summary["test"][f"k{k}"] = row
    # Full layer curve (validation only, t=0) for the record
    summary["validation_curve_t0"] = {f"{rep}|{m}": round(v, 3) for (rep, m), v in val_scores.items()}
    json.dump(summary, open(os.path.join(args.out, "probe_summary.json"), "w"), indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "validation_curve_t0"}, indent=2))
    for h in hooks:
        h.remove()


if __name__ == "__main__":
    main()
